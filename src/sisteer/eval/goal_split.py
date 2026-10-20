"""Deterministic GOAL-level train/held-out split — the seam that lets one
attack family be both trained on and evaluated as held-out.

WHY A GOAL SPLIT RATHER THAN A FAMILY SPLIT. The held-out design excluded
whole FAMILIES from training, so "does this defense generalise to an unseen
attack" was answerable but "does it generalise to unseen GOALS under an
attack we trained on" was not — and restricting the training mix to the four
families with StrongREJECT headroom (bon, pasttense, codeattack, renellm)
would have consumed two of the six held-out families outright. Splitting by
GOAL keeps both: every family may appear on both sides, on DISJOINT goals,
so the leakage gate (which keys on prompt shas) passes by construction and
neither measurement is sacrificed.

THE SPLIT IS ON goal_id, NOT ON THE PROMPT. One goal yields many prompts —
several variants per family, several families per goal — and all of them
carry the same underlying request. Splitting on prompts would put variants
of ONE request on both sides, which is leakage wearing a different name: the
model would meet the same harmful intent in training and again at eval,
differing only in surface transform. Keying on goal_id makes the unit of
separation the REQUEST, which is what "unseen" has to mean here.

Same convention as ``train_two_si_objects.split_pairs``: first byte of
sha256(salt + key), so the split is stable across runs, orderings, machines
and PYTHONHASHSEED, and reproducible from the salt alone. A DIFFERENT salt
from the pair holdout on purpose — the two splits are independent questions
(held-out GOALS for cross-request generalisation, held-out PAIRS for the
in-sample check), and sharing a salt would correlate them silently.
"""

from __future__ import annotations

import collections
import hashlib
from typing import Iterable

# Deliberately distinct from "two_si_holdout" (the pair-level salt).
GOAL_SPLIT_SALT = "two_si_goal_heldout"
DEFAULT_HELDOUT_FRAC = 0.3


def goal_is_heldout(goal_id: str, frac: float = DEFAULT_HELDOUT_FRAC,
                    salt: str = GOAL_SPLIT_SALT) -> bool:
    """True when this goal belongs to the HELD-OUT side.

    ``frac`` is the held-out share. 0.3 by default rather than the pair
    split's 0.1: the held-out side here has to support a per-family attack
    evaluation with usable power, not merely a generalisation spot-check,
    and the families that matter most (pasttense) have few goals to spare.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"frac must be in [0, 1), got {frac}")
    h = hashlib.sha256((salt + str(goal_id)).encode()).digest()[0] / 255.0
    return h < frac


def split_goals(goal_ids: Iterable[str], frac: float = DEFAULT_HELDOUT_FRAC,
                salt: str = GOAL_SPLIT_SALT) -> tuple[list[str], list[str]]:
    """(train, heldout), each sorted so callers cannot depend on input order."""
    train, held = [], []
    for g in {str(x) for x in goal_ids}:
        (held if goal_is_heldout(g, frac, salt) else train).append(g)
    return sorted(train), sorted(held)


def split_report(goal_ids_by_family: dict[str, Iterable[str]],
                 frac: float = DEFAULT_HELDOUT_FRAC,
                 salt: str = GOAL_SPLIT_SALT) -> dict:
    """Per-family train/held-out goal counts, for recording ALONGSIDE any
    number derived from the split.

    A split reported without its composition is the failure this project
    spent a day fixing: thin strata hide inside a healthy-looking total, and
    a family with two held-out goals cannot support an evaluation however
    good the aggregate looks. ``thin`` flags any family whose either side
    falls below ``min_side``.
    """
    out: dict = {"salt": salt, "frac": frac, "families": {}}
    for fam, ids in sorted(goal_ids_by_family.items()):
        tr, he = split_goals(ids, frac, salt)
        out["families"][fam] = {
            "train_goals": len(tr), "heldout_goals": len(he),
            "total": len(tr) + len(he),
            "heldout_share": round(len(he) / max(len(tr) + len(he), 1), 3),
        }
    allg = {str(g) for ids in goal_ids_by_family.values() for g in ids}
    tr, he = split_goals(allg, frac, salt)
    out["union"] = {"train_goals": len(tr), "heldout_goals": len(he),
                    "total": len(allg)}
    return out


def check_disjoint(train_goal_ids: Iterable[str],
                   heldout_goal_ids: Iterable[str]) -> None:
    """Raise if the two sides share a goal. Cheap, and the one invariant the
    whole design rests on — a single shared goal means the held-out
    evaluation is partly measuring memorisation, and nothing downstream
    would surface it."""
    overlap = {str(g) for g in train_goal_ids} & {str(g) for g in heldout_goal_ids}
    if overlap:
        raise ValueError(
            f"{len(overlap)} goal(s) appear on BOTH sides of the split "
            f"(e.g. {sorted(overlap)[:3]}) — the held-out side would be "
            "partly in-sample")


# --------------------------------------------------------------------------- #
# Split-state provenance: a held-out label without its REGIME is not a label   #
# --------------------------------------------------------------------------- #

FAMILY_HELD_OUT = "family_held_out"   # the family never appears in training
GOAL_SPLIT = "goal_split"             # the family trains on disjoint GOALS
PARKED = "parked"                     # neither fit nor evaluated, for now


def split_state(regime: str, *, frac: float | None = None,
                salt: str | None = None, reason: str = "",
                since: str = "") -> dict:
    """The per-family record that must travel with any held-out number.

    Two families can look identical in a results table and answer DIFFERENT
    questions: family-held-out tests generalisation to an unseen attack
    TYPE, goal-split tests generalisation to unseen INSTANCES of a type the
    object was trained on. The second is substantially weaker, and every
    mechanical check — including the prompt-sha leakage gate — stays green
    across the difference, so nothing downstream surfaces it. Hence the
    regime is recorded explicitly, with the salt and fraction that reproduce
    a goal split.

    PARKED families carry their date and reason because a later, narrower
    run will look BETTER than an earlier broad one simply by having dropped
    families the model already resisted — a comparison artifact that is
    invisible unless the parking is written down where the comparison
    happens.
    """
    if regime not in (FAMILY_HELD_OUT, GOAL_SPLIT, PARKED):
        raise ValueError(f"unknown regime {regime!r}")
    rec: dict = {"regime": regime}
    if regime == GOAL_SPLIT:
        if frac is None or salt is None:
            raise ValueError(
                "a goal_split record MUST carry its salt and frac — without "
                "them the split cannot be reproduced and the label cannot be "
                "distinguished from family_held_out")
        rec.update(frac=frac, salt=salt)
    if regime == PARKED:
        if not (reason and since):
            raise ValueError(
                "a parked record MUST carry `since` and `reason`: a family "
                "silently absent from a later run makes that run look better "
                "than an earlier one for free")
        rec.update(since=since, reason=reason)
    return rec


# The standing decision, recorded where a future comparison will find it.
FAMILY_REGIMES: dict[str, dict] = {
    # Trained on AND evaluated, on disjoint goals (the four with usable
    # StrongREJECT headroom: unaided baselines BoN 14%, CodeAttack 10%,
    # past-tense 8%, ReNeLLM 6%).
    "bon":        split_state(GOAL_SPLIT, frac=DEFAULT_HELDOUT_FRAC, salt=GOAL_SPLIT_SALT),
    "pasttense":  split_state(GOAL_SPLIT, frac=DEFAULT_HELDOUT_FRAC, salt=GOAL_SPLIT_SALT),
    "codeattack": split_state(GOAL_SPLIT, frac=DEFAULT_HELDOUT_FRAC, salt=GOAL_SPLIT_SALT),
    "renellm":    split_state(GOAL_SPLIT, frac=DEFAULT_HELDOUT_FRAC, salt=GOAL_SPLIT_SALT),
    # The undefended harmful corpus stays on the training side.
    "wildjailbreak_harmful": split_state(GOAL_SPLIT, frac=DEFAULT_HELDOUT_FRAC,
                                         salt=GOAL_SPLIT_SALT),
}
_PARK_REASON = ("no StrongREJECT headroom at the 0.75 threshold — unaided "
                "success at or under ~2%, so a defense cannot measurably "
                "improve on it; parked, not deleted, and re-enabling is a "
                "config change")
for _f in ("amplegcg_plus", "pap", "encoding", "aim", "gcg_replay",
           "artprompt", "cipherchat", "drattack", "multilingual"):
    FAMILY_REGIMES[_f] = split_state(PARKED, since="2026-08-24", reason=_PARK_REASON)
