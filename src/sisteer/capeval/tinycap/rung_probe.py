"""The truncation ladder's RUNG PROBE — pure.

When more than 1 % of a pinned leg's responses stop at the completion cap
(``tinycap_ladder.py select``, slice C), the capped items are re-run at the
16,384 cap. That rung needs a LONGER window — ``LADDER_WINDOW`` = 32,768 — and
the measured Gemma budget (589,824 batched tokens) buys exactly
``LADDER_SEQS[0]`` = 18 sequences there; a KV cache that turns out smaller than
the budget refuses the standup at readiness (``vllm_server.kv_coverage_problem``
naming the numbers), and the ladder then REDUCES S along ``LADDER_SEQS``
(18 -> 16 -> 14 -> 12), never the window: the window is what the rung is FOR.

Source vocabulary — reuse, never widen: the geometry
predicates know exactly ``"model_budget"`` and ``"override"`` for
``seqs_source``; a reduced S is ``"override"`` — true to ``overridden``,
``is_identity == False``, ``over_budget == False`` — and WHY rides a separate
``rung_source``: ``identity`` (S = 18), ``probe`` (S < 18 accepted after a KV
refusal), ``operator`` (``--ladder-rung L,S``, never masquerading as a probe).

The probe's verdict is a run-level CERTIFICATE, ``<VOL_GEN_ROOT>/<model>/_ladder/
<suffix>/rung_probe.json`` (exact schema :data:`CERTIFICATE_FIELDS`), consulted
BEFORE any GPU: :func:`certificate_is_valid` validates structure AND semantics
against the launch's provenance, and every scoring container takes
its ``accepted`` block as a FIXED footprint. This module is torch-free (the
geometry module + the contracts) so the launcher process, its CPU preflight and
the tests all read one implementation.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

from sisteer.capeval.manifest import canonical_json
from sisteer.capeval.serving import geometry as _sg
from sisteer.capeval.serving import serving_attestation as _sa  # stdlib-only leaf: the LoRA kernel index ceiling
from sisteer.capeval.tinycap import contracts as C

__all__ = [
    # the NATIVE ladder — the engine-served rungs (each requested rung through the LoRA kernel's
    # int32 index ceiling, coalesced), its engine-bound `ladder` block and the reader of a
    # recorded block
    "SERVING_ENGINE_LENS", "SERVING_ENGINE_NATIVE", "LADDER_BLOCK_FIELDS_NATIVE", "native_ladder_spec",
    "spec_from_block",
    # the S the launcher ASKS the standup for a served rung — the budget's rung it stands for
    "requested_rung_for",
    "LADDER_WINDOW", "LADDER_SEQS", "LADDER_MODEL_KEY", "LADDER_CAP_TO", "LADDER_WINDOW_SOURCE",
    "SEQS_SOURCE_MODEL_BUDGET", "SEQS_SOURCE_OVERRIDE", "SEQS_SOURCES",
    "RUNG_SOURCE_IDENTITY", "RUNG_SOURCE_PROBE", "RUNG_SOURCE_OPERATOR", "RUNG_SOURCES",
    "KV_REFUSAL_MARKERS", "KV_UNIFORM_VERDICT_RE", "KV_HYBRID_VERDICT_RE", "KV_VERDICT_RES", "kv_verdict_match",
    "kv_coverage_refusal", "parse_kv_tokens", "next_rung", "ladder_task_partial_reason", "draw_subset_task_reason", "verdict_geometry_reason",
    "ladder_budget", "identity_S", "seqs_source_for", "rung_source_for", "accepted_block",
    "parse_ladder_rung", "format_ladder_rung",
    # bcb-bfcl-200-widen S6: the model-aware ladder, the per-cell
    # certificate, the re-probe at the common rung, the wave admission record
    "LadderSpec", "ladder_spec", "ladder_seqs_for", "LADDER_SEQS_FLOOR", "LADDER_SEQS_STEP", "LADDER_SEQS_DEPTH",
    "LADDER_BLOCK_FIELDS", "REPROBED_FROM_FIELDS", "certificate_filename", "certificate_sha256", "common_rung",
    "ADMISSION_SCHEMA", "ADMISSION_FILENAME", "ADMISSION_FIELDS", "ADMISSION_CELL_FIELDS", "admission_path",
    "build_admission", "admission_is_valid", "read_admission", "write_admission", "admission_state",
    "CERTIFICATE_SCHEMA", "CERTIFICATE_FILENAME", "LADDER_SUBDIR", "CERTIFICATE_FIELDS",
    "ACCEPTED_FIELDS", "WALKED_FIELDS", "BUDGET_FIELDS", "WALK_OUTCOMES", "SELECTION_BLOCK_FIELDS",
    "ladder_dir", "certificate_path", "superseded_certificate_path", "selection_digest",
    "read_ladder_selection", "fit_guard_reason", "Provenance", "walked_entry", "build_certificate",
    "certificate_is_valid", "read_certificate", "write_certificate", "certificate_state",
    "CERTIFICATE_VALID", "CERTIFICATE_ABSENT", "LadderError",
]


class LadderError(ValueError):
    """A refusal of the ladder's pure half: a rung off the ladder, a selection
    file that does not recompute, a certificate that cannot be written."""


# --------------------------------------------------------------------------- #
# the ladder                                                                   #
# --------------------------------------------------------------------------- #

#: The rung-2 window: the 16,384 cap plus every selected item's prompt — the
#: ONE window every rung of the ladder serves (reducing S, never L).
LADDER_WINDOW = 32_768
#: The S candidates, walked downward on a KV refusal. 18 IS the identity S of
#: the measured Gemma budget at the window (asserted at import below); 12 is the
#: lowest S the probe admits — a rung FLOOR, never an item count: nothing in the
#: ladder refuses on the selection's size.
LADDER_SEQS: Tuple[int, ...] = (18, 16, 14, 12)
#: The measured model the ladder is sized for (``geometry.BUDGET_BY_MODEL``).
LADDER_MODEL_KEY = "gemma4e4b"
#: The rung-2 cap (the contracts' one number).
LADDER_CAP_TO = C.LADDER_CAP_TO
#: ``ResolvedGeometry.window_source`` of a ladder geometry (a new WINDOW source
#: is fine — no predicate keys on window sources beyond the footprint ones).
LADDER_WINDOW_SOURCE = "ladder_window"
#: ``ResolvedGeometry.seqs_source`` — exactly the geometry module's two words.
SEQS_SOURCE_MODEL_BUDGET = "model_budget"
SEQS_SOURCE_OVERRIDE = "override"
SEQS_SOURCES: Tuple[str, ...] = (SEQS_SOURCE_MODEL_BUDGET, SEQS_SOURCE_OVERRIDE)
#: WHY the rung's S is what it is (the certificate's / run record's word).
RUNG_SOURCE_IDENTITY = "identity"
RUNG_SOURCE_PROBE = "probe"
RUNG_SOURCE_OPERATOR = "operator"
RUNG_SOURCES: Tuple[str, ...] = (RUNG_SOURCE_IDENTITY, RUNG_SOURCE_PROBE, RUNG_SOURCE_OPERATOR)

assert LADDER_SEQS == tuple(sorted(set(LADDER_SEQS), reverse=True)), LADDER_SEQS
assert LADDER_WINDOW % _sg.VLLM_BLOCK_SIZE == 0

# --------------------------------------------------------------------------- #
# the MODEL-AWARE ladder                                                      #
# --------------------------------------------------------------------------- #

#: The rung list of a model's ladder walks DOWN from its identity S at the window
#: in steps of :data:`LADDER_SEQS_STEP` to a floor of ``max(LADDER_SEQS_FLOOR,
#: identity_S - LADDER_SEQS_DEPTH)`` — gemma (identity 18) -> (18, 16, 14, 12), the
#: first lane's ladder unchanged; nemotron (identity 24) -> (24, 22, 20, 18); qwen
#: after its S6e measurement -> from its measured S. 12 stays the lowest S any
#: probe admits (a rung FLOOR, never an item count).
LADDER_SEQS_FLOOR = 12
LADDER_SEQS_STEP = 2
LADDER_SEQS_DEPTH = 6


def ladder_seqs_for(identity: int, *, floor: int = LADDER_SEQS_FLOOR, step: int = LADDER_SEQS_STEP,
                    depth: int = LADDER_SEQS_DEPTH) -> Tuple[int, ...]:
    """The S candidates of a ladder whose identity S is ``identity``: strictly
    descending from ``identity`` by ``step`` to ``max(floor, identity - depth)``."""
    ident = int(identity)
    if ident < 1:
        raise LadderError(f"identity S {identity!r} is not a positive int")
    if ident < int(floor):
        # code review: an identity BELOW the floor has no ladder — it used to return
        # `(identity,)`, a rung under the documented floor, instead of refusing
        raise LadderError(f"identity S {ident} is below the ladder floor {int(floor)} — no rung of this ladder "
                          "is admissible; the budget row is not one the ladder can be sized from")
    lo = max(int(floor), ident - int(depth))
    seqs = tuple(range(ident, lo - 1, -int(step)))
    assert seqs and seqs[0] == ident, seqs
    return seqs


#: the engine a ladder is walked on. ``None`` / ``"lens"`` on a :class:`LadderSpec` is the
#: historical lens ladder (its ``block()`` byte-identical); ``"lora_native"`` is the
#: ENGINE-SERVED ladder :func:`native_ladder_spec` derives. The literals are restated here
#: (the ``mamba_backend`` pattern) so this pure module keeps importing nothing of the
#: launcher; ``contracts.SERVING_ENGINE_*`` spell the same two words.
SERVING_ENGINE_LENS = "lens"
SERVING_ENGINE_NATIVE = "lora_native"


class LadderSpec(NamedTuple):
    """ONE model's ladder (bcb-bfcl-200-widen S6): the MEASURED budget
    row it is sized from, the rung-2 window, the S candidates walked downward on
    a KV refusal and the identity S (what the budget buys at the window). Threaded
    through the certificate (its ``ladder`` block), the launcher's ladder gate /
    provenance / admission / dry run / probe loop and the meter, so a Nemotron or
    qwen ladder is never sized from Gemma's numbers.

    The five trailing fields are the NATIVE ladder's alone (:func:`native_ladder_spec`; every
    field defaults to "not native", so every existing construction and every lens ``block()``
    is byte-identical). On the native engine a standup lowers S under the LoRA kernel's int32
    index ceiling BEFORE it launches, so the rung the engine SERVES is ``min(S, ceiling //
    window)`` — ``seqs`` / ``identity_S`` are then the SERVED rungs (coalesced: gemma's
    requested 18 and 16 are both served at 15 at 32,768, one rung), ``requested_seqs`` /
    ``requested_identity_S`` the budget's ladder they were mapped from, ``served_rungs`` the
    mapping (``((15, (18, 16)), (14, (14,)), (12, (12,)))``) and ``kernel_ceiling`` the numbers
    behind it. The ``block()`` then carries the engine, so a native certificate is never a lens
    launch's (nor the reverse) even where the engine lowers nothing (llama8b's 14 and 12
    stand)."""

    model_key: str
    budget: "_sg.ModelBudget"
    window: int
    seqs: Tuple[int, ...]
    identity_S: int
    serving_engine: Optional[str] = None
    requested_seqs: Tuple[int, ...] = ()
    requested_identity_S: Optional[int] = None
    served_rungs: Tuple[Tuple[int, Tuple[int, ...]], ...] = ()
    kernel_ceiling: Optional[Mapping[str, int]] = None

    @property
    def native(self) -> bool:
        return self.serving_engine == SERVING_ENGINE_NATIVE

    def block(self) -> Dict[str, Any]:
        """The certificate's ``ladder`` block: exact keys :data:`LADDER_BLOCK_FIELDS` on the lens
        (byte-identical to before), :data:`LADDER_BLOCK_FIELDS_NATIVE` on the native engine."""
        block: Dict[str, Any] = {"model_key": self.model_key, "window": int(self.window),
                                 "seqs": [int(s) for s in self.seqs], "identity_S": int(self.identity_S)}
        if not self.native:
            return block
        block.update({
            "serving_engine": str(self.serving_engine),
            "requested_seqs": [int(s) for s in self.requested_seqs],
            "requested_identity_S": int(self.requested_identity_S),
            "served_rungs": [{"served": int(s), "requested": [int(r) for r in reqs]} for s, reqs in self.served_rungs],
            "kernel_ceiling": {k: int(v) for k, v in dict(self.kernel_ceiling or {}).items()},
        })
        return block

    def as_dict(self) -> Dict[str, Any]:
        return {**self.block(), "batched_token_budget": int(self.budget.batched_token_budget),
                "budget_source": str(self.budget.source)}


def ladder_spec(model_key: str = LADDER_MODEL_KEY, *, window: int = LADDER_WINDOW) -> LadderSpec:
    """The ladder of ``model_key`` from its MEASURED budget row (``geometry.BUDGET_BY_MODEL``).
    A missing or provisional row REFUSES (the ladder serves nothing the model did not
    measure — qwen35_4b only after S6e); the identity S is what the row buys at the
    window (the hybrid per-slot cap applied by ``seqs_at``), the rung list
    :func:`ladder_seqs_for` of it."""
    row = _sg.BUDGET_BY_MODEL.get(model_key)
    if row is None or row.source != _sg.MEASURED_BUDGET_SOURCE:
        raise LadderError(
            f"the ladder needs the MEASURED budget row of {model_key!r}; the geometry table "
            f"holds {row.source if row is not None else None!r}"
            + (" — measure it first (scripts/modal/tinycap_budget_measure.py, plan S6e) and replace the "
               "provisional row" if row is not None else ""))
    L = int(window)
    if L % _sg.VLLM_BLOCK_SIZE:
        raise LadderError(f"ladder window {L} is not a multiple of the vLLM block size {_sg.VLLM_BLOCK_SIZE}")
    ident = int(row.seqs_at(L))
    return LadderSpec(model_key=str(model_key), budget=row, window=L, seqs=ladder_seqs_for(ident), identity_S=ident)


def _spec(spec: Optional[LadderSpec]) -> LadderSpec:
    """The spec in force: the one given, else the default Gemma ladder (every
    pre-existing caller — ``identity_S()``, ``accepted_block(S)`` — keeps its meaning)."""
    return spec if spec is not None else ladder_spec(LADDER_MODEL_KEY)


def native_ladder_spec(spec: LadderSpec) -> LadderSpec:
    """The ladder the NATIVE engine SERVES for the model whose lens ladder is ``spec``.
    A native standup lowers S under the LoRA kernel's int32 index ceiling before it launches
    (``geometry.kernel_ceiling_adjustment``, the pure step: ``to_S = min(S, ((2^31 - 1) // D)
    // window)`` from the model's widest adapted module ``D``), so the rung it serves for a
    requested S is known BEFORE any standup. Each requested rung is mapped through that step
    and the rungs that collapse onto one served S are COALESCED (gemma at 32,768: 18 -> 15,
    16 -> 15, 14, 12 => served (15, 14, 12), identity 15; nemotron, D = 5120: every rung of
    (24, 22, 20, 18) -> 12, the floor — one rung); the served identity is the served image of
    the requested identity. Refuses (:class:`LadderError`) by name a model with no pinned
    width (the step cannot be re-derived: never a guess), a window the ceiling admits no row
    at, and a ceiling that puts every served rung BELOW the ladder floor (no rung the ladder
    admits is servable natively). The returned spec's ``block()`` carries the engine and the
    requested -> served mapping, so the certificate names both."""
    if spec.native:
        raise LadderError(f"the ladder of {spec.model_key!r} is already the native engine's")
    W, key = int(spec.window), str(spec.model_key)
    try:
        D = int(_sg.LORA_MAX_MODULE_DIM_BY_MODEL[key])
    except KeyError:
        raise LadderError(
            f"the native ladder of {key!r} cannot be sized: the model has no pinned LoRA max module dim "
            f"(geometry.LORA_MAX_MODULE_DIM_BY_MODEL knows {sorted(_sg.LORA_MAX_MODULE_DIM_BY_MODEL)}), so the kernel "
            "ceiling step every native standup applies cannot be pre-computed — pin its attested width, never a guess") from None
    ceiling = int(_sa.lora_kernel_index_ceiling(D))
    max_S = ceiling // W
    if max_S < 1:
        raise LadderError(f"the native ladder of {key!r} cannot be sized: the ladder window {W} alone exceeds the LoRA kernel "
                          f"index ceiling {ceiling:,} tokens (D = {D}); no S serves that window natively")
    by_served: Dict[int, List[int]] = {}
    for S in spec.seqs:
        try:
            served = int(_sg.adjusted_seqs(key, W, int(S)))
        except (KeyError, ValueError) as e:              # geometry.UnknownModelKey is a KeyError
            raise LadderError(f"the native ladder of {key!r} cannot be sized at rung S {S}: {e}") from None
        by_served.setdefault(served, []).append(int(S))
    seqs = tuple(by_served)                          # descending: spec.seqs is, and min(S, cap) is monotone
    assert seqs == tuple(sorted(seqs, reverse=True)), seqs
    if seqs[-1] < LADDER_SEQS_FLOOR:
        raise LadderError(
            f"the native ladder of {key!r} has no admissible rung: the LoRA kernel index ceiling serves at most S = {max_S} at "
            f"window {W} (D = {D}, ceiling {ceiling:,} tokens), below the ladder floor {LADDER_SEQS_FLOOR} — every requested rung "
            f"{list(spec.seqs)} would be served below the floor; the ladder cannot run on the native engine for this model")
    identity = int(_sg.adjusted_seqs(key, W, int(spec.identity_S)))
    assert identity == seqs[0], (identity, seqs)
    return LadderSpec(model_key=key, budget=spec.budget, window=W, seqs=seqs, identity_S=identity,
                      serving_engine=SERVING_ENGINE_NATIVE, requested_seqs=tuple(int(s) for s in spec.seqs),
                      requested_identity_S=int(spec.identity_S),
                      served_rungs=tuple((int(s), tuple(reqs)) for s, reqs in by_served.items()),
                      kernel_ceiling={"D": D, "ceiling": ceiling, "max_S": int(max_S)})


def spec_from_block(block: Mapping[str, Any], budget: "_sg.ModelBudget") -> LadderSpec:
    """A :class:`LadderSpec` re-stated from a certificate's recorded ``ladder`` block over ``budget``
    (the readers' fallback when the shared table refuses the model — ``tinycap_ladder``'s and the
    analysis's ``spec_for_certificate``): the lens block's four keys, or the native block's nine
    (its engine and mapping restored, so a native certificate is judged as one). A block with
    another key set refuses."""
    keys = set(block)
    if keys == set(LADDER_BLOCK_FIELDS):
        return LadderSpec(model_key=str(block["model_key"]), budget=budget, window=int(block["window"]),
                          seqs=tuple(int(s) for s in block["seqs"]), identity_S=int(block["identity_S"]))
    if keys == set(LADDER_BLOCK_FIELDS_NATIVE):
        if block["serving_engine"] != SERVING_ENGINE_NATIVE:
            raise LadderError(f"a ladder block with the native keys names engine {block['serving_engine']!r}")
        # the native block is RE-DERIVED, never restated verbatim — the requested ladder it records (the budget's
        # rungs at the window) through the same kernel-ceiling step every native standup applies
        # (:func:`native_ladder_spec`); a recorded mapping that no longer re-derives (a re-pinned width, an edited
        # block) refuses by name, so a certificate is judged as the engine's ladder, not as its own claim
        key = str(block["model_key"])
        requested = LadderSpec(model_key=key, budget=budget, window=int(block["window"]),
                               seqs=tuple(int(s) for s in block["requested_seqs"]),
                               identity_S=int(block["requested_identity_S"]))
        try:
            spec = native_ladder_spec(requested)
        except LadderError as e:
            raise LadderError(f"the recorded native ladder block of {key!r} does not re-derive: {e}") from None
        if spec.block() != dict(block):
            raise LadderError(f"the recorded native ladder block of {key!r} {dict(block)} != the ladder the {SERVING_ENGINE_NATIVE} "
                              f"engine serves for its requested rungs {list(requested.seqs)} today {spec.block()} — a mapping "
                              "that does not re-derive (a re-pinned width, or an edited block) certifies nothing")
        return spec
    raise LadderError(f"ladder block keys {sorted(keys)} are neither the lens ladder's {sorted(LADDER_BLOCK_FIELDS)} nor the "
                      f"native ladder's {sorted(LADDER_BLOCK_FIELDS_NATIVE)}")


def requested_rung_for(spec: LadderSpec, S: int) -> int:
    """The S the launcher ASKS the standup for rung ``S`` of ``spec``. On the lens ``S`` itself. On the
    native engine ``S`` is a SERVED rung and the answer is the LARGEST requested rung the engine serves
    at it (``served_rungs``: gemma's served 15 <- (18, 16) is asked as 18 — the budget's identity S — so
    the engine's own kernel step (18 -> 15) lowers it and is recorded in the standup's ``adjustments``
    exactly as on every native production standup, and the served identity rung IS production's
    footprint; 14 and 12 are asked as themselves). The probe, the scoring worker, the CPU gate and the
    dry run all ask through this one function, so the standup record a certificate is issued on is the
    record production re-derives. Refuses (:class:`LadderError`) an S off the ladder."""
    S = int(S)
    if S not in spec.seqs:
        raise LadderError(f"ladder rung S {S} is not on the ladder {list(spec.seqs)}")
    if not spec.native:
        return S
    for served, requested in spec.served_rungs:
        if int(served) == S:
            return max(int(r) for r in requested)
    raise LadderError(f"the native ladder of {spec.model_key!r} maps no requested rung onto the served S {S} "
                      f"(served_rungs {list(spec.served_rungs)}) — a torn spec")


# the geometry module restates the two below-identity rung sources for its ladder-reduction record (it imports
# nothing of the ladder); the two spellings are held equal here, at import
assert tuple(_sg.LADDER_REDUCTION_SOURCES) == (RUNG_SOURCE_PROBE, RUNG_SOURCE_OPERATOR), (
    _sg.LADDER_REDUCTION_SOURCES, (RUNG_SOURCE_PROBE, RUNG_SOURCE_OPERATOR))

assert LADDER_SEQS == ladder_spec(LADDER_MODEL_KEY).seqs, (LADDER_SEQS, ladder_spec(LADDER_MODEL_KEY).seqs)
assert LADDER_SEQS[0] == _sg.identity_seqs(LADDER_WINDOW, model_key=LADDER_MODEL_KEY), (
    LADDER_SEQS, _sg.identity_seqs(LADDER_WINDOW, model_key=LADDER_MODEL_KEY))


def ladder_budget(spec: Optional[LadderSpec] = None) -> "_sg.ModelBudget":
    """The MEASURED row the ladder is sized from — ``spec.budget``, the Gemma row by
    default (``BUDGET_BY_MODEL[gemma4e4b]``). A missing or provisional row refuses
    (:func:`ladder_spec`): the ladder serves nothing the model did not measure."""
    return _spec(spec).budget


def identity_S(spec: Optional[LadderSpec] = None) -> int:
    """The S the measured budget buys at the spec's window (gemma: 18 at 32,768)."""
    return int(_spec(spec).identity_S)


def seqs_source_for(S: int, identity: int) -> str:
    return SEQS_SOURCE_MODEL_BUDGET if int(S) == int(identity) else SEQS_SOURCE_OVERRIDE


def rung_source_for(S: int, identity: int, *, operator: bool = False) -> str:
    """``identity`` at the budget's own S, ``probe`` below it, ``operator`` below
    it when the operator named the rung. An operator rung AT the identity S
    refuses: the identity rung needs no override (and ``operator`` <=> ``S <
    identity_S`` is what the gate and the certificate hold)."""
    if int(S) == int(identity):
        if operator:
            raise LadderError(f"an operator rung at the identity S {identity} is not an override — run without "
                              f"{C.FLAG_LADDER_RUNG}")
        return RUNG_SOURCE_IDENTITY
    return RUNG_SOURCE_OPERATOR if operator else RUNG_SOURCE_PROBE


def accepted_block(S: int, *, identity: Optional[int] = None, operator: bool = False,
                   max_model_len: Optional[int] = None, spec: Optional[LadderSpec] = None) -> Dict[str, Any]:
    """The certificate's ``accepted`` block — and a configuration's
    ``ladder_rung`` — for rung ``S`` at the spec's window (the Gemma ladder by
    default). Refuses a window other than the spec's or an S off its rungs."""
    sp = _spec(spec)
    L, S = int(sp.window if max_model_len is None else max_model_len), int(S)
    if L != sp.window:
        raise LadderError(f"ladder rung window {L} != LADDER_WINDOW {sp.window}")
    if S not in sp.seqs:
        raise LadderError(f"ladder rung S {S} is not on the ladder {list(sp.seqs)}")
    ident = sp.identity_S if identity is None else int(identity)
    return {"max_model_len": L, "max_num_seqs": S,
            "seqs_source": seqs_source_for(S, ident),
            "rung_source": rung_source_for(S, ident, operator=operator)}


def parse_ladder_rung(spec: str) -> Tuple[int, int]:
    """``"32768,16"`` -> ``(32768, 16)``; refuses anything but two ints."""
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise LadderError(f"{C.FLAG_LADDER_RUNG}: {spec!r} is not <max_model_len>,<max_num_seqs>")
    return int(parts[0]), int(parts[1])


def format_ladder_rung(rung: Tuple[int, int]) -> str:
    return f"{int(rung[0])},{int(rung[1])}"


# --------------------------------------------------------------------------- #
# the KV refusal (the standup's verdict tail, parsed here)                     #
# --------------------------------------------------------------------------- #

#: The standup's readiness refusal for a geometry the KV cache cannot cover:
#: EXACTLY ``vllm_server.kv_coverage_problem``'s two INSUFFICIENT-CAPACITY
#: verdicts, as ``validate_scheduler_reports`` appends them per worker under
#: the no-chunking header (``"- worker {i}: {problem}"``) — the ladder steps
#: down on THESE and nothing else. The header itself
#: ("resolved scheduler state does not match the composed no-chunking
#: invariant") is generic: a prefix-caching / scheduler-class / partial-prefill
#: mismatch raises it with a healthy cache, and an informational "KV cache" line
#: states no shortfall — neither is capacity evidence. There is no structured
#: channel: the standup process prints the ``VllmServerError`` into its log, so
#: the verdict SENTENCES are what a reader can hold to.
#: Uniform cache: ``KV cache holds {kv_tokens} tokens < the {budget}-token budget (= {S} x {L})``.
KV_UNIFORM_VERDICT_RE = re.compile(
    r"KV cache holds (?P<kv>[0-9][0-9,]*) tokens < the (?P<budget>[0-9][0-9,]*)-token budget "
    r"\(= (?P<S>[0-9]+) x (?P<L>[0-9]+)\)")
#: Hybrid cache: ``hybrid KV cache ({desc}) supports only {mc}-way concurrency at
#: max_model_len={L} (~{tokens} tokens) < the requested max_num_seqs={S}``.
KV_HYBRID_VERDICT_RE = re.compile(
    r"hybrid KV cache \(.*\) supports only (?P<mc>[0-9.]+)-way concurrency at max_model_len=(?P<L>[0-9]+) "
    r"\(~(?P<kv>[0-9][0-9,]*) tokens\) < the requested max_num_seqs=(?P<S>[0-9]+)")
KV_VERDICT_RES: Tuple["re.Pattern[str]", ...] = (KV_UNIFORM_VERDICT_RE, KV_HYBRID_VERDICT_RE)
#: The two verdicts' opening words — a cheap pre-filter and the reader's
#: statement of WHAT it recognises; a line must still match a verdict regex.
KV_REFUSAL_MARKERS: Tuple[str, ...] = ("KV cache holds", "hybrid KV cache (")


def kv_verdict_match(line: str) -> Optional["re.Match[str]"]:
    """The insufficient-capacity verdict a log line carries, or None."""
    if not any(m in line for m in KV_REFUSAL_MARKERS):
        return None
    for pat in KV_VERDICT_RES:
        m = pat.search(line)
        if m:
            return m
    return None


def kv_coverage_refusal(standup_log: str) -> Optional[str]:
    """The line(s) of a standup log that carry an EXPLICIT insufficient-capacity
    verdict (:data:`KV_VERDICT_RES`), or None — a standup that died for any
    other reason (a scheduler / config mismatch under the same header, an
    engine crash, an informational KV-size line) is NOT a refusal the ladder
    may step down on."""
    hits = [ln.strip() for ln in (standup_log or "").splitlines() if kv_verdict_match(ln) is not None]
    return "\n".join(hits[-6:]) if hits else None


def parse_kv_tokens(refusal_text: Optional[str]) -> Optional[int]:
    """The resolved KV-cache size (tokens) a refusal names — the SMALLEST when
    several workers report (the most constraining one sizes the rung) — or None
    when the text names none (a refusal that does not say how much fits)."""
    if not refusal_text:
        return None
    found = []
    for ln in refusal_text.splitlines():
        m = kv_verdict_match(ln)
        if m is not None:
            found.append(int(m.group("kv").replace(",", "")))
    found = [n for n in found if n > 0]
    return min(found) if found else None


def next_rung(refusal_text: Optional[str], remaining: Sequence[int], *, window: int = LADDER_WINDOW) -> Optional[int]:
    """The next S to try after a refusal: the LARGEST remaining S whose
    ``S x window`` the refusal's KV cache holds (skipping rungs the numbers
    already rule out); with no readable KV size, the next candidate; None when
    no remaining rung fits (no ladder). ``remaining`` is the tail of the spec's
    rung list after the refused rung."""
    rest = [int(s) for s in remaining]
    if not rest:
        return None
    kv = parse_kv_tokens(refusal_text)
    if kv is None:
        return rest[0]
    fits = [s for s in rest if s * int(window) <= kv]
    return max(fits) if fits else None


# --------------------------------------------------------------------------- #
# the certificate                                                              #
# --------------------------------------------------------------------------- #

#: Schema 2 (bcb-bfcl-200-widen S6): the certificate carries the ``ladder`` block it was
#: walked on (model key, window, rung list, identity S — :meth:`LadderSpec.block`) and
#: ``reprobed_from`` (null, or the FIRST certificate's sha256 + accepted S when this one is the
#: single-rung RE-PROBE at the wave's common footprint). A schema-1 certificate is
#: refused: it names no ladder, so nothing can hold it to the model's.
CERTIFICATE_SCHEMA = 2
CERTIFICATE_FILENAME = "rung_probe.json"
LADDER_SUBDIR = "_ladder"
#: Exact key set (``CERTIFICATE_KEYS``).
CERTIFICATE_FIELDS: Tuple[str, ...] = (
    "schema", "suffix", "accepted", "walked", "tree_sha", "model_slug", "hf", "rev", "cfg_path",
    "cfg_sha256", "object_digests", "selection_sha256", "cap_overrides_by_task", "budget", "ladder",
    "probed_cell", "probe_run_id", "standup_s", "reprobed_from", "at")
ACCEPTED_FIELDS: Tuple[str, ...] = ("max_model_len", "max_num_seqs", "seqs_source", "rung_source")
LADDER_BLOCK_FIELDS: Tuple[str, ...] = ("model_key", "window", "seqs", "identity_S")
#: the NATIVE ladder's block: the lens keys — now the SERVED rungs and identity — plus the
#: engine, the requested ladder they were mapped from, the requested -> served mapping and the
#: kernel-ceiling numbers behind it (:func:`native_ladder_spec`). A certificate's block is held
#: to exactly the key set of the engine its launch serves on, so a lens certificate never
#: admits a native launch, nor the reverse, whatever the numbers.
LADDER_BLOCK_FIELDS_NATIVE: Tuple[str, ...] = LADDER_BLOCK_FIELDS + (
    "serving_engine", "requested_seqs", "requested_identity_S", "served_rungs", "kernel_ceiling")
REPROBED_FROM_FIELDS: Tuple[str, ...] = ("certificate_sha256", "accepted_S")
#: The WAVE record (plan S6): ``_ladder/<suffix>/rung_admission.json`` —
#: the common rung every rerun of the wave launches at (``min`` accepted S over the fitted cells'
#: certificates), and per fitted cell its FINAL certificate's sha256, the S its first probe
#: accepted and whether it was re-probed at the common footprint. Every scoring container's
#: ``ladder_rung`` must equal ``common``; the preflight refuses a record whose certificates do not.
ADMISSION_SCHEMA = 1
ADMISSION_FILENAME = "rung_admission.json"
ADMISSION_FIELDS: Tuple[str, ...] = ("schema", "suffix", "common", "cells", "at")
ADMISSION_CELL_FIELDS: Tuple[str, ...] = ("certificate_sha256", "first_accepted_S", "reprobed")
WALKED_FIELDS: Tuple[str, ...] = ("S", "outcome", "kv_tokens", "refusal_tail")
BUDGET_FIELDS: Tuple[str, ...] = ("batched_token_budget", "budget_source")
WALK_REFUSED, WALK_ACCEPTED = "refused", "accepted"
WALK_OUTCOMES: Tuple[str, ...] = (WALK_REFUSED, WALK_ACCEPTED)
#: ``ladder_selection.json``'s per-task block (slice C's ``select`` writes it;
#: the digest binds the FULL block, so a tampered prompt bound changes it).
SELECTION_BLOCK_FIELDS: Tuple[str, ...] = (
    "ids", "n", "n_selected", "ids_sha256", "cap_from", "cap_to", "max_prompt_tokens_rung1")
CERTIFICATE_VALID = "valid"
CERTIFICATE_ABSENT = "absent"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def ladder_dir(gen_root: str, model_slug: str, suffix: str) -> str:
    """``<gen_root>/<model_slug>/_ladder/<suffix>`` — the run-level ladder
    records of one ``--out-suffix`` (slice C reads the certificate there)."""
    s = str(suffix or "").strip("/")
    if not s:
        raise LadderError("a ladder run needs --out-suffix (the certificate is filed per suffix)")
    return f"{str(gen_root).rstrip('/')}/{model_slug}/{LADDER_SUBDIR}/{s}"


def certificate_filename(cell: Optional[str] = None) -> str:
    """``rung_probe.<cell>.json`` — ONE certificate per probed cell (bcb-bfcl-200-widen S6:
    every fitted cell of a wave is certified by its own probe); ``rung_probe.json`` when no
    cell is named (the earlier single-certificate layout, still readable)."""
    if cell is None:
        return CERTIFICATE_FILENAME
    c = str(cell).strip()
    if not c or "/" in c:
        raise LadderError(f"certificate cell {cell!r} is not a cell id")
    return f"rung_probe.{c}.json"


def certificate_path(gen_root: str, model_slug: str, suffix: str, cell: Optional[str] = None) -> Path:
    return Path(ladder_dir(gen_root, model_slug, suffix)) / certificate_filename(cell)


def admission_path(gen_root: str, model_slug: str, suffix: str) -> Path:
    return Path(ladder_dir(gen_root, model_slug, suffix)) / ADMISSION_FILENAME


def superseded_certificate_path(path: Path, at: Optional[float] = None) -> Path:
    """``rung_probe.<ts>.superseded.json`` beside ``path`` — where
    ``--ladder-reprobe`` moves an INVALID certificate (a recorded act)."""
    ts = _dt.datetime.fromtimestamp(at, tz=_dt.timezone.utc) if at is not None else _dt.datetime.now(_dt.timezone.utc)
    # the per-cell stem is kept: rung_probe.<cell>.<ts>.superseded.json (legacy: rung_probe.<ts>.superseded.json)
    return Path(path).with_name(f"{Path(path).stem}.{ts.strftime('%Y%m%dT%H%M%SZ')}.superseded.json")


def selection_digest(selection: Mapping[str, Mapping[str, Any]]) -> str:
    """``sha256(canonical_json({task: dict(block)}))`` over the FULL per-task
    blocks — byte-identical to slice C's ``tinycap_ladder.selection_digest``
    (both through ``manifest.canonical_json``)."""
    return hashlib.sha256(canonical_json({k: dict(v) for k, v in selection.items()}).encode("utf-8")).hexdigest()


def _positive_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def read_ladder_selection(path: str | Path) -> Dict[str, Any]:
    """The ``--item-selection`` file (``tinycap_ladder.py select``'s
    ``ladder_selection.json``), read to its ACTUAL shape and recomputed: a
    ``selection`` map whose every block carries exactly
    :data:`SELECTION_BLOCK_FIELDS`, names a ladder row task, ``n ==
    n_selected == len(ids)`` (> 0, distinct), ``ids_sha256 ==
    contracts.ids_sha256(ids)``, ``cap_to == LADDER_CAP_TO``, ``cap_from <
    cap_to``, a positive ``max_prompt_tokens_rung1``; and ``selection_sha256``
    recomputing over the selection. Refuses (:class:`LadderError`) otherwise."""
    p = Path(path).expanduser()
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LadderError(f"{p}: {type(e).__name__}: {e}") from e
    if not isinstance(doc, Mapping):
        raise LadderError(f"{p}: not an object")
    for key in ("schema", "selection", "selection_sha256"):
        if key not in doc:
            raise LadderError(f"{p}: missing {key!r}")
    sel = doc["selection"]
    if not isinstance(sel, Mapping) or not sel:
        raise LadderError(f"{p}: selection must be a non-empty object (nothing fired -> no ladder)")
    for task, block in sel.items():
        if task not in C.LADDER_ROW_TASKS:
            raise LadderError(f"{p}: selection names {task!r}, not a ladder row task {list(C.LADDER_ROW_TASKS)}")
        if not isinstance(block, Mapping) or set(block) != set(SELECTION_BLOCK_FIELDS):
            raise LadderError(f"{p}: selection[{task!r}] keys "
                              f"{sorted(block) if isinstance(block, Mapping) else block!r} != {sorted(SELECTION_BLOCK_FIELDS)}")
        ids = block["ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise LadderError(f"{p}: selection[{task!r}].ids must be a non-empty list of distinct strings")
        if block["n"] != len(ids) or block["n_selected"] != len(ids):
            raise LadderError(f"{p}: selection[{task!r}] n / n_selected / len(ids) disagree")
        if block["ids_sha256"] != C.ids_sha256(ids):
            raise LadderError(f"{p}: selection[{task!r}].ids_sha256 does not recompute")
        if block["cap_to"] != LADDER_CAP_TO:
            raise LadderError(f"{p}: selection[{task!r}].cap_to {block['cap_to']!r} != the ladder's {LADDER_CAP_TO}")
        if not _positive_int(block["cap_from"]) or block["cap_from"] >= LADDER_CAP_TO:
            raise LadderError(f"{p}: selection[{task!r}].cap_from {block['cap_from']!r} is not a cap below {LADDER_CAP_TO}")
        if not _positive_int(block["max_prompt_tokens_rung1"]):
            raise LadderError(f"{p}: selection[{task!r}].max_prompt_tokens_rung1 must be a positive int")
        try:
            C.ladder_log_ids(task, ids)                   # every id in the task's pinned vocabulary
        except ValueError as e:
            raise LadderError(f"{p}: selection[{task!r}].ids: {e}") from None
    if doc["selection_sha256"] != selection_digest(sel):
        raise LadderError(f"{p}: selection_sha256 does not recompute over the selection block")
    return dict(doc)


def ladder_task_partial_reason(cell_dir: str | Path, task: str, *, log_ids: Sequence[str], cap: int,
                               n_expected_full: Optional[int]) -> Optional[str]:
    """The ONE strict per-task ladder validator — why ``task`` under
    ``cell_dir`` is NOT a validly complete rung-2 result over ``log_ids`` (the
    selection in LOG vocabulary) at ``cap``; None when it is.
    No done marker; a PARTIAL marker with exactly ``PARTIAL_MARKER_FIELDS``
    naming this task, ``limits == LADDER_LIMITS``, ``n_items == len(log_ids)``,
    ``n_expected_full`` = the leg's frozen full count; the summary present with
    ``n == n_items`` and ``max_tokens == cap``; the outcome record present,
    shape-checked, at ``cap``, its item ids == the selection EXACTLY. Run by
    the launcher (``ladder_partial_reason`` / ``ladder_resume_report``: the
    preflight's admission and the container's post-driver step) AND by the
    driver at task dispatch (``inspect_served.run_one_task`` skips a task this
    returns None for — a resumed two-leg rung re-generates nothing already
    filed)."""
    from sisteer.capeval.tinycap import completion
    cell_dir = Path(cell_dir)
    name = completion.marker_filename(task)
    if (cell_dir / C.DONE_DIR / name).is_file():
        return "a done marker (a rung-2 selection run ends PARTIAL by contract)"
    marker_path = cell_dir / C.PARTIAL_DIR / name
    if not marker_path.is_file():
        return "no partial marker"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"partial marker unreadable ({type(e).__name__})"
    if not isinstance(marker, Mapping):
        return "partial marker is not an object"
    if set(marker) != set(C.PARTIAL_MARKER_FIELDS):
        return (f"partial marker fields {sorted(marker)} != exactly the contract's "
                f"{sorted(C.PARTIAL_MARKER_FIELDS)}")
    if marker["task"] != task:
        return f"partial marker names task {marker['task']!r}"
    if not isinstance(marker["limits"], Mapping) or dict(marker["limits"]) != dict(C.LADDER_LIMITS):
        return (f"partial marker limits {marker['limits']!r} != {dict(C.LADDER_LIMITS)} (a smoke PARTIAL is "
                "not a ladder store)")
    ids = [str(i) for i in log_ids]
    n = len(ids)
    if marker["n_items"] != n:
        return f"partial marker n_items {marker['n_items']!r} != the selection's {n}"
    if marker["n_expected_full"] != n_expected_full:
        return f"partial marker n_expected_full {marker['n_expected_full']!r} != the frozen full count {n_expected_full}"
    summary = completion.read_task_summary(cell_dir, task)
    if summary is None:
        return f"no {C.SUMMARY_DIR}/{name} beside the partial marker"
    if not isinstance(summary, Mapping) or not summary:
        return "summary payload empty"
    if summary.get(C.METRIC_N) != n:
        return f"summary n {summary.get(C.METRIC_N)!r} != the selection's {n}"
    if summary.get("max_tokens") != int(cap):
        return f"summary max_tokens {summary.get('max_tokens')!r} != the rung's cap {int(cap)}"
    try:
        outcomes = completion.read_task_outcomes(cell_dir, task)
    except completion.CompletionError as e:
        return f"outcomes record refused ({str(e)[:200]})"
    if outcomes is None:
        return "no outcomes record beside the partial marker"
    if outcomes.get("max_tokens") != int(cap):
        return f"outcomes max_tokens {outcomes.get('max_tokens')!r} != the rung's cap {int(cap)}"
    try:
        completion.verify_outcomes_against_summary(outcomes, summary)   # the canonical field-by-field verifier
    except completion.CompletionError as e:
        return f"outcomes disagree with the summary ({str(e)[:200]})"
    want, got = set(ids), {str(it.get("item_id")) for it in outcomes.get("items") or ()}
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        return (f"outcomes items != the selection (missing {missing[:5]}{'…' if len(missing) > 5 else ''}, "
                f"extra {extra[:5]}{'…' if len(extra) > 5 else ''})")
    return None


def draw_subset_task_reason(cell_dir: str | Path, task: str, *, draws: Sequence[Sequence[object]], cap: int,
                            source_identity_hash: Optional[str] = None) -> Optional[str]:
    """The ONE strict validator of a DRAW-SUBSET rung store —
    :func:`ladder_task_partial_reason`'s twin for a rung that re-draws ``draws`` (``[[item_id, sample_idx], ...]``, the
    ``rung_subset/2`` file's) at ``cap``. Why ``task`` under ``cell_dir`` is NOT a validly complete rung store over exactly
    those draws; None when it is. Then the runner issues ZERO requests on resume, the launcher admits the store, and the
    analysis reads it as a rung leg. The contract, every seam: a DONE marker (a draw-subset rung ends DONE — its
    ``expected_n`` IS the subset's distinct items) whose ``n_items`` is the subset's distinct items and whose
    ``n_draws == n_expected_draws == len(draws)`` (``ifeval_contracts.DONE_MARKER_DRAW_FIELDS``); the summary present with
    ``n == n_items``, ``n_scored_samples + n_unscored_samples == n_draws`` and ``max_tokens == cap``; the outcome record
    present, shape-checked, at ``cap``, agreeing with the summary (``completion.verify_outcomes_against_summary`` with
    ``n_draws``), its ``(item_id, sample_idx)`` set == the subset's draws EXACTLY (neither more nor fewer — a store whose
    outcomes are not the draws refuses BY NAME; repair = delete the partial store and re-run, never a silent widen); and,
    when the manifest is readable, its ``item_subset`` block a v2 block over the SAME draws bound to ``source_identity_hash``
    when given."""
    from sisteer.capeval.tinycap import completion
    from sisteer.capeval.tinycap import ifeval_contracts as IC
    cell_dir = Path(cell_dir)
    name = completion.marker_filename(task)
    want = sorted({(str(d[0]), int(d[1])) for d in draws})
    if not want:
        return "an empty draw subset names no rung"
    if len(want) != len(list(draws)):
        return "the draw subset repeats a draw"
    ids = sorted({i for i, _ in want})
    if (cell_dir / C.PARTIAL_DIR / name).is_file():
        return "a partial marker (a draw-subset rung ends DONE over exactly its draws; a PARTIAL store is not one)"
    marker_path = cell_dir / C.DONE_DIR / name
    if not marker_path.is_file():
        return "no done marker"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"done marker unreadable ({type(e).__name__})"
    if not isinstance(marker, Mapping) or marker.get("task") != task:
        return f"done marker names task {marker.get('task') if isinstance(marker, Mapping) else None!r}"
    for f in IC.DONE_MARKER_DRAW_FIELDS:
        if f not in marker:
            return f"done marker carries no {f!r} — not a draw-subset store (the marker of a run that declared no expected draw count)"
    if marker["n_items"] != len(ids):
        return f"done marker n_items {marker['n_items']!r} != the subset's {len(ids)} distinct items"
    if marker["n_draws"] != len(want) or marker["n_expected_draws"] != len(want):
        return (f"done marker n_draws {marker['n_draws']!r} / n_expected_draws {marker['n_expected_draws']!r} != the "
                f"subset's {len(want)} draws")
    summary = completion.read_task_summary(cell_dir, task)
    if summary is None:
        return f"no {C.SUMMARY_DIR}/{name} beside the done marker"
    if not isinstance(summary, Mapping) or not summary:
        return "summary payload empty"
    if summary.get(C.METRIC_N) != len(ids):
        return f"summary n {summary.get(C.METRIC_N)!r} != the subset's {len(ids)} distinct items"
    if summary.get("n_scored_samples", 0) + summary.get("n_unscored_samples", 0) != len(want):
        return (f"summary draws {summary.get('n_scored_samples')!r} + {summary.get('n_unscored_samples')!r} != the subset's "
                f"{len(want)} draws")
    if summary.get("max_tokens") != int(cap):
        return f"summary max_tokens {summary.get('max_tokens')!r} != the rung's cap {int(cap)}"
    try:
        outcomes = completion.read_task_outcomes(cell_dir, task)
    except completion.CompletionError as e:
        return f"outcomes record refused ({str(e)[:200]})"
    if outcomes is None:
        return "no outcomes record beside the done marker"
    if outcomes.get("max_tokens") != int(cap):
        return f"outcomes max_tokens {outcomes.get('max_tokens')!r} != the rung's cap {int(cap)}"
    try:
        completion.verify_outcomes_against_summary(outcomes, summary, n_draws=len(want))
    except completion.CompletionError as e:
        return f"outcomes disagree with the summary ({str(e)[:200]})"
    got = sorted({(str(it.get("item_id")), int(it.get("sample_idx"))) for it in outcomes.get("items") or ()})
    if got != want:
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        return (f"outcomes draws != the subset (missing {missing[:5]}{'…' if len(missing) > 5 else ''}, extra "
                f"{extra[:5]}{'…' if len(extra) > 5 else ''}) — a rung's population is its draws, whole; repair = delete "
                "the store and re-run, never widen")
    manifest = completion.read_store_manifest(cell_dir)
    if manifest is not None:
        block = manifest.get(C.MANIFEST_KEY_ITEM_SUBSET)
        if not isinstance(block, Mapping):
            return f"manifest carries no {C.MANIFEST_KEY_ITEM_SUBSET} block — not a rung store"
        if block.get("schema") != IC.ITEM_SUBSET_SCHEMA_V2:
            return f"manifest {C.MANIFEST_KEY_ITEM_SUBSET}.schema {block.get('schema')!r} is not {IC.ITEM_SUBSET_SCHEMA_V2!r} (an item-level rung)"
        block_draws = sorted((str(d[0]), int(d[1])) for d in (block.get("draws") or ()))
        if block_draws != want:
            return f"manifest {C.MANIFEST_KEY_ITEM_SUBSET}.draws ({len(block_draws)}) != the subset's draws ({len(want)})"
        if source_identity_hash is not None and block.get("source_identity_hash") != source_identity_hash:
            return (f"manifest {C.MANIFEST_KEY_ITEM_SUBSET}.source_identity_hash {block.get('source_identity_hash')!r} != the "
                    f"source store's {source_identity_hash!r}")
    return None


def fit_guard_reason(selection: Mapping[str, Mapping[str, Any]], max_model_len: int) -> Optional[str]:
    """The ``assert_pass_fits_window`` precondition, re-derived from the
    selection the digest binds: ``max(max_prompt_tokens_rung1) + cap_to <= L``
    per task. A reason, or None."""
    for task, block in sorted(selection.items()):
        need = int(block["max_prompt_tokens_rung1"]) + int(block["cap_to"])
        if need > int(max_model_len):
            return (f"{task}: the longest selected prompt ({block['max_prompt_tokens_rung1']} tokens) + cap_to "
                    f"{block['cap_to']} = {need} > the rung window {max_model_len}")
    return None


class Provenance(NamedTuple):
    """What THIS launch is — the facts a certificate must equal to admit it."""

    tree_sha: str
    model_slug: str
    hf: str
    rev: str
    cfg_path: str
    cfg_sha256: str
    object_digests: Mapping[str, str]
    selection: Mapping[str, Mapping[str, Any]]
    cap_overrides_by_task: Mapping[str, int]
    batched_token_budget: int
    budget_source: str
    #: the out-suffix the certificate is filed under (a valid certificate
    #: copied into another suffix's path names the old namespace)
    suffix: str
    #: the cell the launch probes — its STEERED cell (a certificate written
    #: by probing the base cell is not this launch's)
    probed_cell: str
    #: bcb-bfcl-200-widen S6: the MODEL's ladder the certificate was walked on
    #: (None = the default Gemma ladder, so every pre-existing caller keeps its meaning)
    ladder: Optional[LadderSpec] = None

    @property
    def spec(self) -> LadderSpec:
        return _spec(self.ladder)

    @property
    def selection_sha256(self) -> str:
        return selection_digest(self.selection)

    @property
    def identity_S(self) -> int:
        return int(identity_S(self.ladder))

    def as_dict(self) -> Dict[str, Any]:
        return {"tree_sha": self.tree_sha, "model_slug": self.model_slug, "hf": self.hf, "rev": self.rev,
                "cfg_path": self.cfg_path, "cfg_sha256": self.cfg_sha256,
                "object_digests": dict(sorted(self.object_digests.items())),
                "selection_sha256": self.selection_sha256,
                "cap_overrides_by_task": dict(sorted(self.cap_overrides_by_task.items())),
                "budget": {"batched_token_budget": int(self.batched_token_budget),
                           "budget_source": self.budget_source},
                "ladder": self.spec.block(),
                "suffix": str(self.suffix).strip("/"), "probed_cell": str(self.probed_cell)}


def verdict_geometry_reason(refusal_tail: Optional[str], S: int, *, window: int = LADDER_WINDOW) -> Optional[str]:
    """Why the verdict(s) in ``refusal_tail`` are NOT evidence about rung ``S``
    at the ladder window — None when every verdict line names exactly that
    geometry and a real shortfall (a walk entry saying S = 18 over a verdict
    produced for S = 16 is not a refusal of 18). Uniform: the captured
    requested S == ``S``, L == ``LADDER_WINDOW``, budget == S x L and
    kv < budget; hybrid: S and L likewise, the allocator's concurrency < S and
    its ~tokens < S x L."""
    L = int(window)
    want_budget = int(S) * L
    lines = [ln for ln in (refusal_tail or "").splitlines() if kv_verdict_match(ln) is not None]
    if not lines:
        return "no KV-coverage verdict in the tail"
    for ln in lines:
        m = kv_verdict_match(ln)
        got_S, got_L = int(m.group("S")), int(m.group("L"))
        if got_S != int(S) or got_L != L:
            return (f"the verdict names S {got_S} x L {got_L}, not this rung's {int(S)} x {L} — "
                    "evidence about another geometry")
        kv = int(m.group("kv").replace(",", ""))
        if m.re is KV_UNIFORM_VERDICT_RE:
            budget = int(m.group("budget").replace(",", ""))
            if budget != want_budget:
                return f"the verdict's budget {budget} != S x L = {want_budget}"
            if not kv < budget:
                return f"the verdict states no shortfall ({kv} tokens >= the {budget}-token budget)"
        else:
            mc = float(m.group("mc"))
            if not (mc < int(S) and kv < want_budget):
                return f"the hybrid verdict states no shortfall ({mc}-way, ~{kv} tokens vs S {S} x {L})"
    return None


def walked_entry(S: int, outcome: str, refusal_tail: Optional[str] = None, *,
                 window: int = LADDER_WINDOW) -> Dict[str, Any]:
    if outcome not in WALK_OUTCOMES:
        raise LadderError(f"walk outcome {outcome!r} not in {list(WALK_OUTCOMES)}")
    if outcome == WALK_REFUSED:
        why = verdict_geometry_reason(refusal_tail, S, window=window)
        if why is not None:
            raise LadderError(f"a refused rung S={S} needs its OWN KV-coverage verdict: {why} ({refusal_tail!r})")
    tail = refusal_tail if outcome == WALK_REFUSED else None
    return {"S": int(S), "outcome": outcome,
            "kv_tokens": parse_kv_tokens(tail) if tail else None,
            "refusal_tail": tail}


def build_certificate(*, accepted: Mapping[str, Any], walked: Sequence[Mapping[str, Any]],
                      provenance: Provenance, probe_run_id: str, standup_s: Optional[float],
                      suffix: Optional[str] = None, probed_cell: Optional[str] = None,
                      at: Optional[str] = None, reprobed_from: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The certificate document, keys exactly :data:`CERTIFICATE_FIELDS`. The
    suffix and the probed cell are the PROVENANCE's (a writer cannot file a
    certificate naming another namespace or another probe target); passing
    them is allowed only when they agree. ``reprobed_from`` (S6) names
    the FIRST certificate this single-rung re-probe supersedes (its sha256 and the
    S it accepted); the ``ladder`` block is the provenance spec's."""
    pd = provenance.as_dict()
    if suffix is not None and str(suffix).strip("/") != pd["suffix"]:
        raise LadderError(f"certificate suffix {suffix!r} != the launch's {pd['suffix']!r}")
    if probed_cell is not None and str(probed_cell) != pd["probed_cell"]:
        raise LadderError(f"certificate probed_cell {probed_cell!r} != the launch's probe cell {pd['probed_cell']!r}")
    rf = None
    if reprobed_from is not None:
        rf = {"certificate_sha256": str(reprobed_from["certificate_sha256"]),
              "accepted_S": int(reprobed_from["accepted_S"])}
    cert = {"schema": CERTIFICATE_SCHEMA, "suffix": pd["suffix"],
            "accepted": {k: accepted[k] for k in ACCEPTED_FIELDS},
            "walked": [{k: w[k] for k in WALKED_FIELDS} for w in walked],
            **{k: pd[k] for k in ("tree_sha", "model_slug", "hf", "rev", "cfg_path", "cfg_sha256", "object_digests",
                                  "selection_sha256", "cap_overrides_by_task", "budget", "ladder")},
            "probed_cell": pd["probed_cell"], "probe_run_id": str(probe_run_id),
            "standup_s": (None if standup_s is None else float(standup_s)),
            "reprobed_from": rf,
            "at": at or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    assert tuple(cert) == CERTIFICATE_FIELDS, tuple(cert)
    return cert


def certificate_sha256(cert: Mapping[str, Any]) -> str:
    """sha256 over the certificate's canonical JSON — what the wave record and a re-probe's
    ``reprobed_from`` name."""
    return hashlib.sha256(canonical_json(dict(cert)).encode("utf-8")).hexdigest()


def _iso8601(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _exact_keys(block: Any, fields: Sequence[str], what: str) -> Optional[str]:
    if not isinstance(block, Mapping):
        return f"{what} is not an object ({type(block).__name__})"
    if set(block) != set(fields):
        missing, extra = sorted(set(fields) - set(block)), sorted(set(block) - set(fields))
        return (f"{what} keys != {sorted(fields)}"
                + (f" (missing {missing})" if missing else "") + (f" (extra {extra})" if extra else ""))
    return None


def certificate_is_valid(cert: Any, provenance: Provenance, *,
                         operator_rung: Optional[Tuple[int, int]] = None) -> Optional[str]:
    """Why ``cert`` does NOT admit a launch with ``provenance`` — None when it
    does. Checks IN ORDER (every rule reads the MODEL's ladder off
    ``provenance.spec``): (1) the exact key set at every level (incl. the
    ``ladder`` block and ``reprobed_from``); (2) JSON types — ints where ints,
    positive dims / ``kv_tokens``, ISO-8601 ``at``; (3)
    ``accepted.max_model_len == spec.window``; (4) ``accepted.max_num_seqs in
    spec.seqs``; (5) source consistency — ``identity`` <=> S == identity_S <=>
    ``model_budget``; ``probe`` <=> S < identity_S <=> ``override``;
    ``operator`` only when THIS launch passes ``--ladder-rung`` equal to
    ``accepted``; (6) the budget is the measured row's and ``S x L <=`` it;
    (6b) the ``ladder`` block equals the spec's; (7) a LEGAL walk
    (``walked[0].S == identity_S``, on the ladder, strictly descending, every
    non-terminal entry ``refused`` with a recognised tail whose ``next_rung``
    IS the next entry, exactly one ``accepted`` — the last — at
    ``accepted.max_num_seqs``; ``[]`` for an operator certificate; a RE-PROBE
    (``reprobed_from`` set) walks exactly its one accepted rung BELOW the S its
    first certificate accepted); (8) provenance equal to this launch's; (9) the
    fit guard over the selection the digest binds."""
    spec = provenance.spec
    # (1) exact keys, every level
    bad = _exact_keys(cert, CERTIFICATE_FIELDS, "certificate")
    if bad:
        return bad
    acc, walked, budget = cert["accepted"], cert["walked"], cert["budget"]
    # the `ladder` block's key set is the ENGINE's: a lens certificate's four keys never admit
    # a native launch, whose block names the engine and the requested -> served mapping — nor
    # the reverse, whatever the rung numbers happen to be
    bad = (_exact_keys(acc, ACCEPTED_FIELDS, "accepted") or _exact_keys(budget, BUDGET_FIELDS, "budget")
           or _exact_keys(cert["ladder"], LADDER_BLOCK_FIELDS_NATIVE if spec.native else LADDER_BLOCK_FIELDS,
                          f"ladder ({'the native engine' if spec.native else 'the lens'}: this launch's)"))
    if bad:
        return bad
    rf = cert["reprobed_from"]
    if rf is not None:
        bad = _exact_keys(rf, REPROBED_FROM_FIELDS, "reprobed_from")
        if bad:
            return bad
        if not C.is_sha256_hex(rf["certificate_sha256"]):
            return f"reprobed_from.certificate_sha256 {rf['certificate_sha256']!r} is not a sha256 hex"
        if not _positive_int(rf["accepted_S"]):
            return f"reprobed_from.accepted_S {rf['accepted_S']!r} is not a positive int"
    if not isinstance(walked, list):
        return f"walked is not a list ({type(walked).__name__})"
    for i, w in enumerate(walked):
        bad = _exact_keys(w, WALKED_FIELDS, f"walked[{i}]")
        if bad:
            return bad
    # (2) types
    if cert["schema"] != CERTIFICATE_SCHEMA or isinstance(cert["schema"], bool):
        return f"schema {cert['schema']!r} != {CERTIFICATE_SCHEMA}"
    if not isinstance(cert["suffix"], str) or not cert["suffix"].strip("/"):
        return f"suffix {cert['suffix']!r} is not a non-empty string"
    for k in ("max_model_len", "max_num_seqs"):
        if not _positive_int(acc[k]):
            return f"accepted.{k} {acc[k]!r} is not a positive int"
    for k in ("seqs_source", "rung_source"):
        if not isinstance(acc[k], str):
            return f"accepted.{k} {acc[k]!r} is not a string"
    if acc["seqs_source"] not in SEQS_SOURCES:
        return f"accepted.seqs_source {acc['seqs_source']!r} not in {list(SEQS_SOURCES)}"
    if acc["rung_source"] not in RUNG_SOURCES:
        return f"accepted.rung_source {acc['rung_source']!r} not in {list(RUNG_SOURCES)}"
    for i, w in enumerate(walked):
        if not _positive_int(w["S"]):
            return f"walked[{i}].S {w['S']!r} is not a positive int"
        if w["outcome"] not in WALK_OUTCOMES:
            return f"walked[{i}].outcome {w['outcome']!r} not in {list(WALK_OUTCOMES)}"
        if w["kv_tokens"] is not None and not _positive_int(w["kv_tokens"]):
            return f"walked[{i}].kv_tokens {w['kv_tokens']!r} is not a positive int or null"
        if w["refusal_tail"] is not None and not isinstance(w["refusal_tail"], str):
            return f"walked[{i}].refusal_tail is not a string or null"
    for k in ("tree_sha", "model_slug", "hf", "rev", "cfg_path", "cfg_sha256", "selection_sha256",
              "probed_cell", "probe_run_id"):
        if not isinstance(cert[k], str) or not cert[k]:
            return f"{k} {cert[k]!r} is not a non-empty string"
    if not _SHA40.match(cert["tree_sha"]):
        return f"tree_sha {cert['tree_sha']!r} is not a 40-hex commit sha"
    for k in ("cfg_sha256", "selection_sha256"):
        if not C.is_sha256_hex(cert[k]):
            return f"{k} {cert[k]!r} is not a sha256 hex"
    od = cert["object_digests"]
    if not isinstance(od, Mapping) or not all(isinstance(c, str) and C.is_sha256_hex(d) for c, d in od.items()):
        return f"object_digests {od!r} is not a {{cell: sha256}} object"
    caps = cert["cap_overrides_by_task"]
    if not isinstance(caps, Mapping) or not caps or not all(isinstance(t, str) and _positive_int(v) for t, v in caps.items()):
        return f"cap_overrides_by_task {caps!r} is not a non-empty {{row task: positive int}} object"
    if not _positive_int(budget["batched_token_budget"]) or not isinstance(budget["budget_source"], str):
        return f"budget {budget!r} is not {{batched_token_budget: int > 0, budget_source: str}}"
    ss = cert["standup_s"]
    if ss is not None and (isinstance(ss, bool) or not isinstance(ss, (int, float)) or ss < 0):
        return f"standup_s {ss!r} is not a non-negative number or null"
    if not _iso8601(cert["at"]):
        return f"at {cert['at']!r} is not ISO-8601"
    # (3) (4) the rung is on the MODEL's ladder
    L, S = acc["max_model_len"], acc["max_num_seqs"]
    if L != spec.window:
        return f"accepted.max_model_len {L} != LADDER_WINDOW {spec.window}"
    if S not in spec.seqs:
        return f"accepted.max_num_seqs {S} is not on the ladder {list(spec.seqs)}"
    # (5) source consistency
    ident = provenance.identity_S
    src = acc["rung_source"]
    if src == RUNG_SOURCE_OPERATOR:
        if operator_rung is None:
            return (f"accepted.rung_source is {RUNG_SOURCE_OPERATOR!r} but this launch passed no "
                    f"{C.FLAG_LADDER_RUNG} — an operator certificate never admits a launch that did not ask for it")
        if (int(operator_rung[0]), int(operator_rung[1])) != (L, S):
            return (f"{C.FLAG_LADDER_RUNG} {format_ladder_rung(operator_rung)} != the operator certificate's "
                    f"accepted ({L}, {S})")
    elif operator_rung is not None:
        return (f"this launch passed {C.FLAG_LADDER_RUNG} {format_ladder_rung(operator_rung)} but the certificate's "
                f"rung_source is {src!r} — an operator rung needs an operator certificate ({C.FLAG_LADDER_REPROBE} "
                "supersedes the old one)")
    if src == RUNG_SOURCE_IDENTITY and S != ident:
        return f"accepted.rung_source {RUNG_SOURCE_IDENTITY!r} but S {S} != identity S {ident}"
    if src == RUNG_SOURCE_PROBE and not S < ident:
        return f"accepted.rung_source {RUNG_SOURCE_PROBE!r} but S {S} is not below identity S {ident}"
    if acc["seqs_source"] != seqs_source_for(S, ident):
        return (f"accepted.seqs_source {acc['seqs_source']!r} != {seqs_source_for(S, ident)!r} for S {S} "
                f"(identity S {ident})")
    # (6) the budget
    want_budget = int(provenance.batched_token_budget)
    if budget["batched_token_budget"] != want_budget or budget["budget_source"] != provenance.budget_source:
        return (f"budget {dict(budget)} != this launch's measured row "
                f"{{'batched_token_budget': {want_budget}, 'budget_source': {provenance.budget_source!r}}}")
    if S * L > want_budget:
        return f"accepted S x L = {S} x {L} = {S * L} > the budget {want_budget}"
    # (6b) the ladder block IS this launch's model ladder (a certificate walked on
    # another model's rungs never admits this one) — on THIS launch's engine: the native block
    # names it and the requested -> served mapping, re-derived here from the model's pinned width
    if cert["ladder"].get("serving_engine", SERVING_ENGINE_LENS) != (spec.serving_engine or SERVING_ENGINE_LENS):
        return (f"the certificate's ladder is the {cert['ladder'].get('serving_engine', SERVING_ENGINE_LENS)!r} engine's; this "
                f"launch serves on {spec.serving_engine or SERVING_ENGINE_LENS!r} — a certificate never admits the other engine")
    if dict(cert["ladder"]) != spec.block():
        return f"ladder {dict(cert['ladder'])} != this launch's {spec.block()}"
    # (7) the walk
    if src == RUNG_SOURCE_OPERATOR:
        if walked:
            return f"an operator certificate walks nothing; walked has {len(walked)} entries"
        if rf is not None:
            return "an operator certificate is never a re-probe (reprobed_from must be null)"
    elif rf is not None:
        # a RE-PROBE at the wave's common footprint: ONE accepted rung, below
        # the S the first certificate accepted (so rung_source is `probe`)
        if len(walked) != 1 or walked[0]["outcome"] != WALK_ACCEPTED or walked[0]["S"] != S:
            return (f"a re-probe certificate walks exactly its one accepted rung S {S}; walked "
                    f"{[(w['S'], w['outcome']) for w in walked]}")
        if not rf["accepted_S"] > S:
            return (f"reprobed_from.accepted_S {rf['accepted_S']} is not above the re-probed S {S} — a re-probe "
                    "exists only where the first certificate accepted a LARGER rung than the wave's common one")
        if rf["accepted_S"] not in spec.seqs:
            return f"reprobed_from.accepted_S {rf['accepted_S']} is not on the ladder {list(spec.seqs)}"
        if walked[0]["kv_tokens"] is not None or walked[0]["refusal_tail"] is not None:
            return "the accepted walk entry carries a refusal (kv_tokens / refusal_tail must be null)"
    else:
        if not walked:
            return "walked is empty (a probe certificate records the rungs it stood up)"
        if walked[0]["S"] != ident:
            return f"walked[0].S {walked[0]['S']} != identity S {ident} (the probe starts at the identity rung)"
        seqs = [w["S"] for w in walked]
        if any(s not in spec.seqs for s in seqs):
            return f"walked S {seqs} leaves the ladder {list(spec.seqs)}"
        if any(a <= b for a, b in zip(seqs, seqs[1:])):
            return f"walked S {seqs} is not strictly descending"
        n_acc = sum(1 for w in walked if w["outcome"] == WALK_ACCEPTED)
        if n_acc != 1 or walked[-1]["outcome"] != WALK_ACCEPTED:
            return f"walked must end in exactly one {WALK_ACCEPTED!r} entry (found {n_acc}, last {walked[-1]['outcome']!r})"
        if walked[-1]["S"] != S:
            return f"the accepted walk entry's S {walked[-1]['S']} != accepted.max_num_seqs {S}"
        for i, w in enumerate(walked[:-1]):
            tail = w["refusal_tail"]
            if kv_coverage_refusal(tail or "") is None:
                return f"walked[{i}] (S {w['S']}) refused with no KV-coverage refusal in its tail"
            geo = verdict_geometry_reason(tail, w["S"], window=spec.window)
            if geo is not None:
                return f"walked[{i}] (S {w['S']}): {geo}"
            if w["kv_tokens"] != parse_kv_tokens(tail):
                return f"walked[{i}].kv_tokens {w['kv_tokens']!r} != the tail's {parse_kv_tokens(tail)!r}"
            remaining = [s for s in spec.seqs if s < w["S"]]
            want_next = next_rung(tail, remaining, window=spec.window)
            if walked[i + 1]["S"] != want_next:
                return (f"walked[{i + 1}].S {walked[i + 1]['S']} != next_rung({w['kv_tokens']} tokens, "
                        f"{remaining}) = {want_next} — the skip does not agree with the recorded refusal")
        if walked[-1]["kv_tokens"] is not None or walked[-1]["refusal_tail"] is not None:
            return "the accepted walk entry carries a refusal (kv_tokens / refusal_tail must be null)"
    # (8) provenance — the suffix (the namespace the certificate is filed under;
    # a certificate copied into another suffix's path names the old one) and the
    # probed cell (the launch's STEERED cell) included
    want = provenance.as_dict()
    for k in ("tree_sha", "model_slug", "hf", "rev", "cfg_path", "cfg_sha256", "selection_sha256", "suffix",
              "probed_cell"):
        if cert[k] != want[k]:
            return f"{k} {cert[k]!r} != this launch's {want[k]!r}"
    if dict(cert["object_digests"]) != want["object_digests"]:
        return f"object_digests {dict(cert['object_digests'])} != this launch's {want['object_digests']}"
    if dict(cert["cap_overrides_by_task"]) != want["cap_overrides_by_task"]:
        return f"cap_overrides_by_task {dict(cert['cap_overrides_by_task'])} != this launch's {want['cap_overrides_by_task']}"
    # (9) the fit guard, from the selection the digest binds
    return fit_guard_reason(provenance.selection, L)


def read_certificate(path: str | Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(certificate, None)`` / ``(None, None)`` when absent / ``(None, why)``
    when present but unreadable (an unreadable certificate is INVALID, never
    absent — it is not silently re-probed)."""
    p = Path(path)
    if not p.is_file():
        return None, None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"{p}: unreadable ({type(e).__name__}: {str(e)[:120]})"
    if not isinstance(doc, Mapping):
        return None, f"{p}: not an object"
    return dict(doc), None


def write_certificate(path: str | Path, cert: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(cert), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return p


def certificate_state(cert: Optional[Mapping[str, Any]], read_error: Optional[str], provenance: Provenance,
                      *, operator_rung: Optional[Tuple[int, int]] = None) -> str:
    """``"valid"`` | ``"absent"`` | ``"invalid(<reason>)"`` — what the preflight
    reports under ``ladder_admission.certificate``."""
    if cert is None and read_error is None:
        return CERTIFICATE_ABSENT
    if cert is None:
        return f"invalid({read_error})"
    reason = certificate_is_valid(cert, provenance, operator_rung=operator_rung)
    return CERTIFICATE_VALID if reason is None else f"invalid({reason})"


# --------------------------------------------------------------------------- #
# the WAVE admission record                                                   #
# --------------------------------------------------------------------------- #


def common_rung(certificates: Mapping[str, Mapping[str, Any]]) -> int:
    """The wave's common footprint: the MINIMUM accepted S over the fitted cells'
    certificates (every rerun of the wave launches at it — what ``merge`` requires)."""
    if not certificates:
        raise LadderError("no certificate to take a common rung from")
    return min(int(c["accepted"]["max_num_seqs"]) for c in certificates.values())


def build_admission(*, suffix: str, common: Mapping[str, Any], certificates: Mapping[str, Mapping[str, Any]],
                    at: Optional[str] = None) -> Dict[str, Any]:
    """The wave record: ``common`` = the accepted block every final certificate carries;
    per cell its final certificate's sha256, the S its FIRST probe accepted
    (``reprobed_from.accepted_S`` when re-probed, else its own) and whether it was re-probed."""
    cells: Dict[str, Any] = {}
    for cell, cert in sorted(certificates.items()):
        rf = cert.get("reprobed_from")
        cells[str(cell)] = {"certificate_sha256": certificate_sha256(cert),
                            "first_accepted_S": int(rf["accepted_S"]) if rf else int(cert["accepted"]["max_num_seqs"]),
                            "reprobed": rf is not None}
    doc = {"schema": ADMISSION_SCHEMA, "suffix": str(suffix).strip("/"),
           "common": {k: common[k] for k in ACCEPTED_FIELDS}, "cells": cells,
           "at": at or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    assert tuple(doc) == ADMISSION_FIELDS, tuple(doc)
    return doc


def admission_is_valid(doc: Any, *, suffix: str, spec: LadderSpec,
                       certificates: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """Why ``doc`` does NOT admit a wave whose fitted cells' FINAL certificates are
    ``certificates`` — None when it does: exact keys; the suffix; ``common`` a rung of
    the spec's ladder; the cell set == the certificates' (a cell probed but not recorded,
    or recorded but unprobed, refuses); per cell the recorded sha256 == the certificate's,
    the certificate's accepted == ``common`` (the re-probe rule: every final certificate
    carries the common rung), ``reprobed`` iff the certificate says so, and
    ``first_accepted_S`` == the re-probe's ``reprobed_from.accepted_S`` / the accepted S."""
    bad = _exact_keys(doc, ADMISSION_FIELDS, "admission")
    if bad:
        return bad
    if doc["schema"] != ADMISSION_SCHEMA or isinstance(doc["schema"], bool):
        return f"admission schema {doc['schema']!r} != {ADMISSION_SCHEMA}"
    if str(doc["suffix"]).strip("/") != str(suffix).strip("/"):
        return f"admission suffix {doc['suffix']!r} != this launch's {str(suffix).strip('/')!r}"
    bad = _exact_keys(doc["common"], ACCEPTED_FIELDS, "admission.common")
    if bad:
        return bad
    common = doc["common"]
    if common["max_model_len"] != spec.window or common["max_num_seqs"] not in spec.seqs:
        return (f"admission.common {dict(common)} is not a rung of the ladder {list(spec.seqs)} at window "
                f"{spec.window}")
    if not _iso8601(doc["at"]):
        return f"admission at {doc['at']!r} is not ISO-8601"
    cells = doc["cells"]
    if not isinstance(cells, Mapping) or not cells:
        return "admission.cells is not a non-empty object"
    if set(cells) != set(certificates):
        return (f"admission.cells {sorted(cells)} != the certificates given {sorted(certificates)} — every "
                "probed cell is recorded, and only probed cells")
    for cell, rec in cells.items():
        bad = _exact_keys(rec, ADMISSION_CELL_FIELDS, f"admission.cells[{cell!r}]")
        if bad:
            return bad
        cert = certificates[cell]
        if rec["certificate_sha256"] != certificate_sha256(cert):
            return f"admission.cells[{cell!r}].certificate_sha256 != the final certificate's"
        if dict(cert.get("accepted") or {}) != dict(common):
            return (f"certificate of {cell!r} accepted {dict(cert.get('accepted') or {})} != the common rung "
                    f"{dict(common)} — re-probe it at the common footprint")
        rf = cert.get("reprobed_from")
        if bool(rec["reprobed"]) != (rf is not None):
            return f"admission.cells[{cell!r}].reprobed {rec['reprobed']!r} disagrees with the certificate"
        first = int(rf["accepted_S"]) if rf else int(cert["accepted"]["max_num_seqs"])
        if rec["first_accepted_S"] != first:
            return f"admission.cells[{cell!r}].first_accepted_S {rec['first_accepted_S']!r} != {first}"
        if first < int(common["max_num_seqs"]):
            return f"admission.cells[{cell!r}]: first accepted S {first} is below the common rung {common['max_num_seqs']}"
    return None


def read_admission(path: str | Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return read_certificate(path)


def write_admission(path: str | Path, doc: Mapping[str, Any]) -> Path:
    return write_certificate(path, doc)


def admission_state(doc: Optional[Mapping[str, Any]], read_error: Optional[str], *, suffix: str, spec: LadderSpec,
                    certificates: Mapping[str, Mapping[str, Any]]) -> str:
    """``"valid"`` | ``"absent"`` | ``"invalid(<reason>)"`` for the wave record."""
    if doc is None and read_error is None:
        return CERTIFICATE_ABSENT
    if doc is None:
        return f"invalid({read_error})"
    reason = admission_is_valid(doc, suffix=suffix, spec=spec, certificates=certificates)
    return CERTIFICATE_VALID if reason is None else f"invalid({reason})"
