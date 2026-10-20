"""Engine protocol + ArmHandle.

Stage-0 FROZEN interfaces, pinned by ``tests/test_capeval_protocol.py``.
Implementation: serving/hf_engine.py (every v0 scored path).

Engine-matched rule: deltas only ever pair rows from the same engine —
fingerprint refuses cross-engine pairs (loud error naming both cells).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    ContextManager,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from sisteer.capeval.manifest import AblationSpec


@dataclass
class GenOut:
    """One generation (canary / target-plugin paths).

    ``gen_token_ids`` are the RAW generated token ids, including the
    terminal EOS when ``ended_with_eos`` (additive): the canary computes
    repetition and content length from these raw ids — ``text`` is decoded
    with special tokens stripped, so a special-token degeneration (e.g. 256
    BOS tokens) is invisible in ``text`` but not in the ids."""

    text: str
    gen_token_logprobs: List[float]
    ended_with_eos: bool
    gen_token_ids: List[int] = field(default_factory=list)


class PositionScore(NamedTuple):
    """One scored position from score_corpus (§2.4): the teacher-forced
    next-token NLL, plus the Tier-0 atom quantities when ``atom_ids`` were
    requested (None otherwise). Unpacks as the plan's
    ``(token_nll, atom_probs|None)`` tuple."""

    token_nll: float
    atom_probs: Optional[Any]  # the atom container (cache schema)


@dataclass
class ArmHandle:
    """Returned by engine.arm(...) — one (arm, coeff) binding.

    Concrete engines subclass and implement the two methods; the four
    fields are the frozen attribute schema.
    """

    name: str
    kind: str
    coeff: float
    intervention_hash: str

    def ctx(self) -> ContextManager[None]:
        """Context manager entering obj.apply(model, coeff) or
        hooks.steering(...). FRESH per call: a long-lived context is a
        contract violation, and callers wrap each forward/generate call
        individually (F10)."""
        raise NotImplementedError("engine implementations provide (plan §2.4)")

    def transform_prompt(self, s: str) -> str:
        """obj.inject() for token kinds; identity otherwise. Only Tier-0
        and target-plugin paths call this."""
        raise NotImplementedError("engine implementations provide (plan §2.4)")


@runtime_checkable
class Engine(Protocol):
    """The serving seam (§2.4). One model load per process."""

    name: str  # "hf" | "vllm"

    def load(self) -> None:
        """Load model+tokenizer (models.load_model_and_tokenizer with the
        manifest's pinned revision). One load per process."""
        ...

    def arm(self, spec: AblationSpec, coeff: float) -> ArmHandle:
        """Resolve one manifested arm at one coefficient (through
        methods.load_object for sisteer artifacts; stats/controls.py for
        control:* kinds)."""
        ...

    def score_corpus(
        self,
        corpus: Any,  # a corpora.py corpus slice (that module defines the type)
        arm: Optional[ArmHandle],
        atom_ids: Optional[Any] = None,
    ) -> Iterator[PositionScore]:
        """Tier 0: teacher-forced per-position scoring. arm=None scores
        the base model. atom_ids (the base cache's per-position top-k ids)
        requests the coarsened-KL quantities (§4.3)."""
        ...

    def generate(
        self,
        prompts: Sequence[str],
        arm: Optional[ArmHandle],
        gen_cfg: Mapping[str, Any],
    ) -> List[GenOut]:
        """Batch generation with scores (canary §4.4, target plugins §9)."""
        ...


#: Engine-phase steering policy (2026-08-20): the arm ctx — hooks, weight
#: edits, whatever the method installs — is entered for the DECODE phase
#: only; the prefill forward runs ENTIRELY OUTSIDE it (the arm machinery
#: is not invoked at any prompt position — never merely
#: invoked-with-zero-coefficient), so every prompt position is served by
#: the plain base model and steered decode attends over unsteered
#: base-computed prompt KV. For weight-edit arms the reading is "prompt
#: encoded by base weights, generation computed by adapted weights".
#: BOS-skip stays an INDEPENDENT within-hooked-region position mask —
#: it is not collapsed into this phase flag. Method-generic BY
#: CONSTRUCTION: it gates ctx enter/exit, never per-method position
#: logic. The semantic is shared with the vLLM proxy lane (branch
#: integrate-main) by cross-path agreement, 2026-08-20.
GENERATED_POSITIONS = "generated"

#: The RESPONSE MASK (2026-08-28): the position-mask decode-only serving --
#: the assistant-header positions at the tail of the prompt PLUS every
#: generated position are steered, so the prefill forward that emits the
#: FIRST generated token runs steered (hooks.SteeringHandle._position_mask
#: with response_header_len=H; jailbreak._set_response_positions is the one
#: seam that arms it, H<=0 refused on ANY tokenizer). This is the mechanism
#: the JB/lens lanes already serve ("position_mask" in
#: sweep.DECODE_MECHANISMS); GENERATED_POSITIONS above is the OTHER,
#: incompatible decode-only implementation (unsteered prefill, token 1
#: unsteered -- retired for defense serving by the 2026-08-21 ruling). The
#: two must never be pooled or compared; an arm resolves to at most one.
RESPONSE_POSITIONS = "response"


def arm_position_policy(arm: Optional[ArmHandle]) -> str:
    """The arm's resolved steering-position policy, ``"all"`` for the
    base model (arm=None) and for handles that carry no resolved spec.
    The engine-phase ``steer_positions`` policy (``"generated"``) wins
    over the object's own position mode when present — it changes WHICH
    forwards run inside the ctx, which is the coarser fact a caller must
    respect first.

    Teacher-forced scorers must refuse every policy but ``"all"``: the
    full scored sequence is forwarded, so a selector policy like ``"last"``
    steers the appended target token rather than its predictor position —
    and under ``"generated"`` a teacher-forced forward contains no
    generated positions at all."""
    if arm is None:
        return "all"
    spec = getattr(arm, "resolved_spec", None) or {}
    phase = str(spec.get("steer_positions", "all"))
    if phase != "all":
        return phase
    return str(spec.get("positions", "all"))
