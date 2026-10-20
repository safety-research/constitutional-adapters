"""The at-cap rule — ONE definition, shared by the carry-over bundle reader
(:mod:`.carry_over`) and the cap ladder (:mod:`.cap_ladder`).

Stdlib-only on purpose. The sampled shell's ``ladder_gate.py`` imports
``cap_ladder`` on every gate call between rungs (tens of spawns per job, on
the GPU-idle gap between standups), and ``carry_over`` reaches ``judged`` ->
``tier0`` / ``sisteer.models`` (torch, transformers: ~7 s and ~860 MB per
process). Holding the three token counters here is what keeps ``cap_ladder``
free of that chain.
"""
from __future__ import annotations

#: the geometry measurement's completion-count convention, stated so the
#: rerun's at-cap set is provably the measurement's. Recorded on artifacts
#: under its historical name ``carry_over.AT_CAP_RULE`` (still exported there).
AT_CAP_RULE = "min(ntok(text) + 1, cap) >= cap"


def ntok(tok, text: str) -> int:
    """Tokens of ``text`` alone (no specials) — the re-tokenized completion."""
    return len(tok(text, add_special_tokens=False)["input_ids"])


def completion_tokens(tok, text: str, cap: int) -> int:
    """Completion tokens as vLLM would count them for a text generated under
    ``cap``: the re-tokenized text +1 EOS on a stop finish, clamped at the
    cap (a length finish emits no EOS) — ``common.comp_tokens``'s rule."""
    return min(ntok(tok, text) + 1, int(cap))


def is_at_cap(tok, text: str, cap: int) -> bool:
    """:data:`AT_CAP_RULE`."""
    return completion_tokens(tok, text, cap) >= int(cap)
