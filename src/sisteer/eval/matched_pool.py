"""Matched training-pool construction for the multi-SI comparison.

This is the **frozen training-pool artifact** the two-SI aggregate
(``scripts/two_si_run.py aggregate``) reads FIRST and refuses to run without
(``two_si_run._load_pool_regime``). It exists to make the N SI
objects' training pools *provably identical except for the ``pos``
completion*, because ``pos`` is **SI-specific by construction** — it is the
completion the *SI-prompted* teacher gave — so left to their own devices the
SIs thwart *different* attack subsets and would train on *different*
pools. A held-out difference could then come from the training data rather
than from the instruction, which destroys the headline.

Originally two SIs; the machinery is N-SI (stage 3 extends
the matched-intersection invariant to THREE SIs — the two safety SIs plus the
length-matched irrelevant control — with the data held constant across all of
them: identical prompts, identical ``neg``, only ``pos`` differs).

This module is the **CPU-only computation (piece A)**: it works over already
graded rows (no model, no GPU, no network), computes the matched
intersection and the native per-SI pools, runs the identity assertions, and
freezes the result to a counts-only manifest. The GPU-side generation of the
missing per-SI ``pos`` rows (piece B) lives in
``scripts/build_two_si_pool.py generate-pos`` and only fills gaps that this
computation reports; A always runs standalone against whatever rows exist.

The frozen 8-step ordering, implemented exactly:

1. Freeze the train attack families and the train goal ids (:class:`PoolSpec`).
2. Generate/grade the undefended (``none``) responses once (an input here).
3. Generate or reuse **every** SI-prompted response on **every identical**
   ``none``-successful prompt (inputs here; B fills the gaps A reports).
4. ``matched_ids = intersection over SIs of thwarted_by[si]``.
5. **Assert** the pools have identical ordered prompt ids, identical
   per-family counts, identical per-goal counts, and identical ``neg``
   responses — so they differ ONLY in ``pos``
   (:func:`assert_pools_identical_except_pos`).
6. **Hash and freeze** the manifest (``manifest_sha256``) before training reads it.
7. Record the **native** per-SI pools as a clearly-labelled secondary.
8. State the regime (matched vs native fallback) prominently in BOTH cases.

Stage-3 extensions:

- **A benign-compliance half** (``PoolSpec.benign_source`` /
  ``benign_fraction``): benign prompts get their OWN all-SIs-comply
  intersection — a benign item is admitted only where EVERY SI-prompted
  teacher fully complied (XSTest rubric ``C`` only, consumed through
  ``content_sets.select_xstest_complied``) — then the two sides are balanced
  to the declared benign fraction, counted in pairs, by a deterministic
  sha-ordered subsample of the LARGER side (identical across SIs, so the
  identity invariant is untouched). ``neg`` for a benign pair is the
  undefended (``none``) response, exactly as on the harmful side.
- **The >=200 hard floor** (``hard_floor_pairs``): counted
  AFTER every filter (grading, the N-way SI intersection, train/scoring
  disjointness), BOTH sides must clear the floor or the build raises
  :class:`PoolFloorError` naming the realised counts per SI and per side.
  With the floor (or a benign half) configured the native fallback is
  DISABLED: a thin intersection FAILS, it never silently ships a confounded
  native pool.
- **The composition assertion**: when
  ``benign_fraction`` is declared, the realised pool must contain every
  declared train family with a NON-ZERO count and a realised benign fraction
  within ``benign_fraction_tol`` — else :class:`PoolCompositionError`. This
  cannot be satisfied vacuously: an empty pool raises.
- **Train/scoring disjointness**: any candidate whose
  NORMALISED prompt matches ``scoring_prompt_shas`` (the eval-control
  prompts the objects are scored on) is removed BEFORE the intersections and
  the floor count, and the removal is recorded.
- **Rejection sampling**: an SI's ``pos`` rows may carry
  multiple ``candidate_idx`` values per prompt — the irrelevant SI reaches
  parity with the safety SIs by sampling K candidates and accepting only
  THWARTING completions (which is also what keeps a harmful-compliance
  target out of the mix). The reduction accepts a prompt iff ANY candidate
  thwarts, picks the smallest accepted ``candidate_idx`` as the frozen
  ``pos``, and reports the realised K and per-prompt yield per SI.

Containment (JB4 / task hard rule): the manifest holds **counts, ids and
hashes only — NEVER harmful text**. Attack prompts and completions stay in
the mode-0700 machine-local store; here they exist only in memory long enough
to compute a sha. Every string that reaches the manifest is a safe opaque id
(family / goal id / variant key), a hex sha, or one of this module's own fixed
label strings — enforced by :func:`assert_no_store_text` before the manifest
is returned.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .jailbreak_data import normalized_text_sha

MANIFEST_VERSION = 2

# The two — and only two — regimes the frozen manifest may declare. These
# strings are the contract with ``two_si_run._load_pool_regime``.
REGIME_MATCHED = "matched_intersection"
REGIME_NATIVE = "native_per_si"
POOL_REGIMES = (REGIME_MATCHED, REGIME_NATIVE)

# The anti-pattern, named so it can never be reintroduced: equal size is not
# equal composition, so "subsample to a common size" is rejected BY NAME.
# (The stage-3 benign/harmful balance is NOT this pattern: it trims one SIDE
# of the pool identically for every SI; it never trims one SI to another.)
FORBIDDEN_REGIME = "subsample_to_common_size"

_REGIME_NOTE = {
    REGIME_MATCHED: (
        "Class A objects trained on the MATCHED INTERSECTION: identical ordered "
        "prompt ids, per-family counts, per-goal counts and neg responses for "
        "every SI; ONLY the pos completion differs (D8). The SI contrast is not "
        "confounded by pool composition."),
    REGIME_NATIVE: (
        "D8 FALLBACK: the matched intersection was too small to train on, so "
        "Class A objects train on NATIVE per-SI pools whose sizes and "
        "composition DIFFER. The headline SI contrast is therefore CONFOUNDED "
        "by pool composition (reported, never silent — D26). Subsampling to a "
        "common size is explicitly NOT done: equal size is not equal "
        "composition."),
}

_REPLAY_NOTE = (
    "adaptive/non-static families; excluded from the primary pool by "
    "construction (D0). Never mixed into the static pairs.")

_BENIGN_NOTE = (
    "benign-compliance half (stage-3 fixed decisions 1-3): admitted only "
    "where EVERY SI-prompted teacher fully complied (XSTest rubric C; P and "
    "I rejected alike). neg is the undefended response, shared across SIs.")

_RELAXED_NOTE = (
    "relaxed requirement-(a) stratum (user amendment 2026-08-20): prompts the "
    "UNDEFENDED model already REFUSED, thwarted by every SI. Its neg is a base "
    "refusal and its pos an SI-styled refusal, so the object learns 'refuse in "
    "this SI's style' rather than 'refuse instead of complying'. A SEPARATE "
    "stratum, never mixed into the primary matched intersection — report with "
    "and without it.")

_ANTI_NEG_NOTE = (
    "bidpo anti-safety negative source (bidpo amendment 2026-08-20): a genuine "
    "harmful COMPLIANCE elicited under safety_refusal.anti_text, one per harmful "
    "matched pair, accepted only when the production grader scored it a real "
    "compliance (StrongREJECT > threshold). It is bidpo's DISPREFERRED side for "
    "EVERY harmful prompt, lifting the DPO preference off the undefended refusal "
    "(which was itself a refusal ~94% of the time). Recorded here counts/shas "
    "ONLY; the completions live in the mode-0700 store. This block is bidpo-only "
    "and NEVER alters matched_intersection's neg — the identity invariant across "
    "SIs is untouched.")

_BIDPO_BENIGN_NOTE = (
    "bidpo's OWN training pool (spec-bidpo-benign 2026-08-21, section-13 "
    "final mix rule). HARMFUL side: every prompt THIS SI thwarted (per-SI "
    "thwart, deliberately NOT the three-way matched intersection and NOT the "
    "pool's 50/50 balance subsample - those exist for the distillation arms) "
    "that carries an accepted anti-safety compliance; chosen = this SI's "
    "refusal, rejected = the anti compliance. BENIGN side, polarity "
    "REVERSED: benign prompts this SI REFUSED (reduced XSTest label in the "
    "accepted refusal set - any-C compliance wins, else the smallest "
    "candidate's label, the same reduction the benign intersection uses) "
    "while the UNDEFENDED model fully complied (its own XSTest label is C, "
    "VERIFIED by grading - never assumed); chosen = the undefended "
    "compliance, rejected = this SI's refusal. Mix rule: benign = "
    "min(harmful, available) per SI by default; the harmful side is never "
    "subsampled. The three SIs therefore train at DIFFERENT harmful:benign "
    "RATIOS as well as counts - ratio shapes what a preference objective "
    "learns, so the per-SI counts AND ratios recorded here must be stated "
    "alongside any bidpo cross-SI number. bidpo-only: the matched "
    "intersection and every other arm's inputs are byte-identical with this "
    "block on or off, and bidpo's per-SI files are EXPLICITLY OUTSIDE the "
    "identical-except-pos guarantee. Counts/shas only; completions stay in "
    "the mode-0700 store.")

# The acceptance-direction label the anti-neg block records: unlike a pos row
# (accepted when it THWARTS, score <= threshold), an anti-neg is accepted when
# it COMPLIES (score > threshold) — the opposite direction, stated so a reader
# never has to infer it.
_ANTI_ACCEPT_LABEL = "strongreject_gt_threshold_compliance"

# The fixed vocabulary of manifest string LEAVES that are our own labels /
# structural tokens (everything else must be a safe id or a hex sha). Used by
# assert_no_store_text so a stray free-text string fails loudly.
#: the goal-level-holdout note the family record carries (since 2026-08-28
#: family-level holdout is retired; do not reintroduce it).
_GOAL_HOLDOUT_NOTE = ("families may appear on both sides by design; "
                      "the holdout is the GOAL split (2026-08-28)")

_LABEL_STRINGS: frozenset[str] = frozenset(
    [*POOL_REGIMES, *_REGIME_NOTE.values(), _REPLAY_NOTE, _BENIGN_NOTE,
     _RELAXED_NOTE, _ANTI_NEG_NOTE, _BIDPO_BENIGN_NOTE, _ANTI_ACCEPT_LABEL,
     _GOAL_HOLDOUT_NOTE,
     "relaxed", "strict", "benign",
     "intersection_below_min_pool_size", "source_floor_violation"])

# A safe opaque id: no whitespace, bounded length, from a small punctuation set
# (matches jailbreak_artifacts._SAFE_KEY in spirit). A harmful prompt/response
# (long, spaced) can never masquerade as one of these.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:/@+\-]{1,128}$")
_HEX_SHA = re.compile(r"^[0-9a-f]{64}$")

# XSTest C/P/I rubric labels a benign SI row may carry (only "C" — full
# compliance — admits a row; P and I are rejected alike).
XSTEST_LABELS = ("C", "P", "I")


class MatchedPoolError(ValueError):
    """A hard failure in matched-pool construction (fail loud, never degrade)."""


class PoolAssertionError(MatchedPoolError):
    """One or more of the five identity invariants was violated."""


class PoolFloorError(MatchedPoolError):
    """The >=N hard floor was violated: below the floor the build FAILS —
    it never falls back and never ships thin."""


class PoolDimensionError(PoolFloorError):
    """The DIMENSION-AWARE floor was violated: an arm's realised fit-set
    size ``n`` is below the dimension ``d`` of the object it fits, so the
    closed-form solve is rank-deficient (the regime that produced a defence
    with no measurable effect elsewhere in this programme). A subclass of
    :class:`PoolFloorError` so any ``except PoolFloorError`` still catches
    it; raised with every arm's realised n and its d in the message. The old
    scalar ">= 200 per side per SI" floor is replaced as the AUTHORITATIVE
    gate by "n >= d per fitted arm" — for the d = 4096 alphakl probes that
    is ~4,600 matched pairs per SI after the holdout and calibration carves;
    ~45,000 reaches n ~ 10d."""


class PoolCompositionError(MatchedPoolError):
    """The realised pool composition does not match what the config
    declared: a missing train family or an off-target benign fraction
    fails the build rather than freezing."""


def text_sha(text: str) -> str:
    """A RAW sha256 over the exact bytes of ``text``.

    Deliberately NOT ``jailbreak_data.normalized_text_sha``: that lowercases and
    collapses whitespace, which would MERGE distinct BoN augmentations (BoN's
    whole mechanism is case/whitespace perturbation) into one sha and silently
    fuse different attack prompts. Prompt/response identity here must be exact,
    so the hash is over the raw bytes. (The train/scoring-disjointness filter
    DOES use the normalised sha — a trivially reformatted eval prompt must
    still be caught there.)"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Inputs                                                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GradedRow:
    """One already-graded (prompt, condition) observation.

    ``prompt`` and ``response`` are LIVE-ONLY (never serialised — only their
    shas reach the manifest), the same discipline ``jailbreak_data.Goal`` uses
    for harmful text. The pairing unit is ``prompt_id = (family, goal_id,
    variant_key)``: ``variant_key`` identifies the attack INSTANCE within a
    (family, goal) — e.g. BoN's ``"<sample_idx>:<trial>"`` — and MUST be a safe
    opaque id, never the raw variant text (that would leak).

    ``candidate_idx`` distinguishes rejection-sampling candidates for the SAME
    prompt under the SAME SI: the irrelevant SI's ``pos`` is sampled K times
    and only thwarting completions are accepted. Rows with the default 0
    behave exactly as before.

    ``label`` carries the XSTest C/P/I rubric verdict for a BENIGN SI row;
    harmful rows use ``score`` (StrongREJECT) instead.
    """

    family: str
    goal_id: str
    variant_key: str
    condition: str            # "none" (undefended) | "si" (SI-prompted)
    si_id: str | None         # which SI for condition=="si"; None for "none"
    prompt: str               # live-only; hashed to prompt_sha
    response: str             # live-only; hashed to neg_sha/pos_sha
    score: float | None = None
    source: str | None = None  # pool source/stratum label; defaults to family
    label: str | None = None   # XSTest C/P/I for benign SI rows
    candidate_idx: int = 0     # rejection-sampling candidate

    @property
    def prompt_id(self) -> tuple[str, str, str]:
        return (self.family, self.goal_id, self.variant_key)

    @property
    def source_label(self) -> str:
        return self.source if self.source is not None else self.family


@dataclass(frozen=True)
class PoolSpec:
    """The frozen inputs that define the pool.

    ``static_families`` is the allowlist of static/replayable families — the
    ONLY families whose rows may enter the primary pool: adaptive attackers
    emit a different prompt per condition, so "the same attack under both
    conditions" is undefined, measured 657 for BoN vs 0 for the adaptive
    four. Rows from any other family are routed to the ``replay_transfer``
    stratum, never mixed in silently.

    Stage-3 fields (all default OFF so legacy two-SI builds are unchanged):

    - ``benign_source`` / ``benign_fraction`` / ``benign_fraction_tol``: the
      DECLARED pool composition. Setting them turns on the benign half, the
      balancing, and the composition assertion — and DISABLES the native
      fallback. ``benign_source`` is either a single family label (the legacy
      PHTest-only stratum) OR a collection of labels (PHTest +
      jailbreak-styled benign + the AlphaSteer benign pool, each its OWN
      family so the manifest's per-family counts and per-source floors keep
      every corpus's contribution visible). ``benign_families`` normalises
      either form to a frozenset.
    - ``hard_floor_pairs``: the >=N floor per side per SI; 0 disables.
      Setting it also disables the native fallback. It is a COARSE per-side
      lower guard — the AUTHORITATIVE size gate is the dimension-aware floor
      below.
    - ``fit_dims`` / ``fit_holdout_frac`` / ``fit_calibration_rows``: the
      DIMENSION-AWARE floor. ``fit_dims`` maps an arm id to the dimension
      ``d`` of the object it fits (e.g. the alphakl probes at d = 4096 =
      hidden_size). For each such arm the realised fit-set size ``n =
      floor((1 - fit_holdout_frac) * n_matched) - fit_calibration_rows[arm]``
      (the early-stopping holdout carve and, for the alphakl arms, the 64-row
      calibration carve) is compared against ``d``; ``n < d`` for ANY arm
      raises :class:`PoolDimensionError` naming every arm's realised n and d.
      Empty ``fit_dims`` disables it (legacy).
    - ``relaxed_stratum_enabled``: admit the RELAXED requirement-(a) harmful
      stratum — other refusals of harm besides jailbreaks. Prompts the
      UNDEFENDED model already REFUSED (score <= threshold) but every SI
      still thwarts are collected into a SEPARATELY LABELLED
      ``relaxed_requirement_a`` manifest block — NEVER mixed into the primary
      matched intersection, so a result can be reported with and without it.
      Under the default it is FOLDED into the primary pool instead, so the
      SEPARATE block is emitted only in the ``require_undefended_compliance``
      regime.
    - ``require_undefended_compliance``: whether the primary harmful
      intersection REQUIRES the undefended model to have complied. Default
      False — the requirement is DROPPED, so the base is every harmful prompt
      with a none row (the relaxed pool is the DEFAULT) and requirement (a)
      survives only as an identifiable STRICT SUBSET in the manifest (per-pair
      ``requirement_a`` + ``n_strict``). True restores the strict base (bidpo
      needs a genuine harmful compliance as its dispreferred example), and
      then the relaxed stratum is available as the separate labelled block.
    - ``scoring_prompt_shas``: NORMALISED shas (``jailbreak_data.
      normalized_text_sha``) of every prompt the objects are scored on; any
      candidate matching one is removed before the intersections — training
      items are never reused for scoring.
    - ``anti_si_id``: the pseudo-SI id under which the anti-safety negatives
      were generated (``safety_refusal.anti_text`` as system). When set,
      SI-prompted rows carrying this id are routed to a SEPARATE anti-neg
      reduction — NOT into ``si_cands`` — and reduced by the opposite
      acceptance direction (accept a candidate iff it COMPLIES, score >
      threshold; the frozen anti-neg is the smallest accepted
      ``candidate_idx``, rejection-sampling semantics). It produces the
      ``bidpo_anti_neg`` manifest block and NEVER enters the matched
      intersection, so ``anti_si_id`` MUST NOT be one of ``si_ids``. None
      (default) disables the block entirely.
    - ``bidpo_benign_*``: bidpo's OWN per-SI training pool, frozen as the
      ``bidpo_benign`` manifest block. HARMFUL side: every prompt THIS SI
      thwarted — the per-SI thwart, deliberately NOT the three-way joint and
      NOT the 50/50 balance subsample (those exist for the distillation
      arms) — that carries an accepted anti-safety compliance, so
      ``anti_si_id`` is REQUIRED; chosen = the SI's refusal, rejected = the
      anti compliance. BENIGN side, polarity REVERSED: benign prompts this SI
      REFUSED (reduced label in ``bidpo_benign_refusal_labels`` — ``("I",)``
      by default, ``("I", "P")`` the muddier opt-in) while the UNDEFENDED
      model fully complied (its own XSTest label is ``C`` — VERIFIED, so the
      undefended benign ``none`` rows must have been graded first; an
      ungraded one that is needed RAISES, never assumed ``C``): chosen = the
      undefended compliance, rejected = this SI's refusal. Per-SI and
      UNEQUAL: no cross-SI parity subsample — the per-SI counts and the
      projected harmful:benign ratios are recorded instead, and the earlier
      >1000 benign floor is SUPERSEDED: ``bidpo_benign_floor`` defaults to 0,
      so only an EMPTY side (benign or harmful, any SI) fails the build.
      Requires a declared benign stratum. The block is bidpo-only and leaves
      every other manifest block byte-identical.
    """

    si_ids: tuple[str, ...]
    train_families: tuple[str, ...]
    eval_families: tuple[str, ...]
    train_goal_ids: tuple[str, ...]
    eval_goal_ids: tuple[str, ...]
    static_families: frozenset[str]
    # (family_disjointness is RETIRED — user ruling 2026-08-28: the goal
    # split IS the holdout, families may appear on both sides by design, and
    # the field is gone rather than dormant so nothing can re-enable it.)
    # Minimum share of the pool's goals a train family must reach. Presence
    # alone is not a check: a frozen-corpus family covering 3 of 300 goals
    # scores identically to one covering all 300, and thinness LOOKS LIKE
    # DATA while absence is visible. 0.0 preserves the old presence-only
    # behaviour for callers that have not set it.
    min_family_goal_coverage: float = 0.0
    threshold: float = 0.5
    min_pool_size: int = 1
    source_floors: Mapping[str, int] = field(default_factory=dict)
    default_source_floor: int = 0
    benign_source: "str | tuple[str, ...] | frozenset[str] | None" = None
    benign_fraction: float | None = None
    benign_fraction_tol: float = 0.02
    hard_floor_pairs: int = 0
    fit_dims: Mapping[str, int] = field(default_factory=dict)
    fit_holdout_frac: float = 0.0
    fit_calibration_rows: Mapping[str, int] = field(default_factory=dict)
    relaxed_stratum_enabled: bool = False
    require_undefended_compliance: bool = False
    scoring_prompt_shas: frozenset[str] = frozenset()
    anti_si_id: str | None = None
    bidpo_benign_enabled: bool = False
    bidpo_benign_refusal_labels: tuple[str, ...] = ("I",)
    bidpo_benign_floor: int = 0
    # Per-family hard caps on matched pairs: {family -> max matched pairs},
    # applied deterministically BEFORE the 50/50 balance so the cap can change
    # which side is binding. Empty (the default) is a no-op — every existing
    # config/test stays byte-identical without it.
    family_caps: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.si_ids) < 2:
            raise MatchedPoolError(
                f"PoolSpec needs at least two si_ids to compare, got {self.si_ids}")
        bad = set(self.train_families) - set(self.static_families)
        if bad:
            raise MatchedPoolError(
                f"train_families must be a subset of static_families (only "
                f"static/replayable attacks qualify — D0); offenders: {sorted(bad)}")
        if (self.benign_source is None) != (self.benign_fraction is None):
            raise MatchedPoolError(
                "benign_source and benign_fraction must be declared together "
                f"(got benign_source={self.benign_source!r}, "
                f"benign_fraction={self.benign_fraction!r}) — the config must "
                "declare the intended composition, not half of it")
        if self.benign_source is not None and not self.benign_families:
            raise MatchedPoolError(
                "benign_source declares no families — pass a family label or a "
                "non-empty collection of them")
        if self.benign_fraction is not None and not (
                0.0 < float(self.benign_fraction) < 1.0):
            raise MatchedPoolError(
                f"benign_fraction must be in (0, 1), got {self.benign_fraction}")
        clash = self.benign_families & (
            set(self.static_families) | set(self.train_families)
            | set(self.eval_families))
        if clash:
            raise MatchedPoolError(
                f"benign_source {sorted(clash)} collides with an attack "
                "family — the benign half is its own stratum, never an attack")
        if self.hard_floor_pairs < 0:
            raise MatchedPoolError(
                f"hard_floor_pairs must be >= 0, got {self.hard_floor_pairs}")
        if not (0.0 <= float(self.fit_holdout_frac) < 1.0):
            raise MatchedPoolError(
                "fit_holdout_frac must be in [0, 1), got "
                f"{self.fit_holdout_frac}")
        bad_dims = {a: d for a, d in self.fit_dims.items() if int(d) <= 0}
        if bad_dims:
            raise MatchedPoolError(
                f"fit_dims must be positive dimensions, got {bad_dims}")
        if self.anti_si_id is not None and self.anti_si_id in self.si_ids:
            raise MatchedPoolError(
                f"anti_si_id {self.anti_si_id!r} must NOT be one of the compared "
                f"si_ids {self.si_ids} — the anti-safety negatives are bidpo's "
                "dispreferred side, never a comparison SI, and mixing them into "
                "the matched intersection would break the identity invariant")
        if self.bidpo_benign_enabled and self.benign_source is None:
            raise MatchedPoolError(
                "bidpo_benign_enabled requires a declared benign stratum "
                "(benign_source/benign_fraction): the reversed-polarity side "
                "is built from the benign store rows, so enabling it against "
                "a pool with no benign half is a configuration contradiction "
                "— refusing rather than silently producing an empty side")
        if self.bidpo_benign_enabled and self.anti_si_id is None:
            raise MatchedPoolError(
                "bidpo_benign_enabled requires anti_si_id (pool.anti_neg): "
                "the section-13 harmful side's REJECTED completions are the "
                "anti-safety compliances — the legacy undefended-strict "
                "source measures ~1.5% of the pool and is unusable, so a "
                "bidpo pool without the anti source is a configuration "
                "contradiction")
        labels = tuple(self.bidpo_benign_refusal_labels)
        bad_labels = set(labels) - {"I", "P"}
        if not labels or bad_labels:
            raise MatchedPoolError(
                "bidpo_benign_refusal_labels must be a non-empty subset of "
                f"('I', 'P') — the REJECTED side of a reversed benign pair is "
                f"a refusal, never a compliance; got {labels!r} (spec-bidpo-"
                "benign DECISION 3: default ('I',) — unambiguous refusals "
                "only; ('I', 'P') is the muddier opt-in)")
        if self.bidpo_benign_floor < 0:
            raise MatchedPoolError(
                f"bidpo_benign_floor must be >= 0, got {self.bidpo_benign_floor}")

    @property
    def benign_families(self) -> frozenset[str]:
        """The benign stratum's family labels, normalised from
        ``benign_source`` (a single str, a collection, or None)."""
        if self.benign_source is None:
            return frozenset()
        if isinstance(self.benign_source, str):
            return frozenset({self.benign_source})
        return frozenset(self.benign_source)

    def floor_for(self, source: str) -> int:
        return int(self.source_floors.get(source, self.default_source_floor))

    def fit_rows(self, n_matched: int, arm: str) -> int:
        """Realised fit-set size for ``arm`` at a balanced pool of
        ``n_matched`` pairs: the nominal holdout carve then the arm's own
        calibration carve. The estimate the pre-GPU dimension floor gates on
        (the deployed split hashes per prompt, so the realised split is
        within rounding of this — the gate uses the nominal fraction and
        says so)."""
        kept = int(math.floor((1.0 - float(self.fit_holdout_frac)) * n_matched))
        return kept - int(self.fit_calibration_rows.get(arm, 0))

    @property
    def strict(self) -> bool:
        """True when the stage-3 hard gates are on: the native fallback is
        disabled and a thin/confounded pool RAISES instead of shipping."""
        return (self.hard_floor_pairs > 0 or self.benign_source is not None
                or bool(self.fit_dims))


# --------------------------------------------------------------------------- #
# The five identity assertions (step 5)                                       #
# --------------------------------------------------------------------------- #


def _pool_prompt_ids(pool: list["PairRecord"]) -> list[tuple[str, str, str]]:
    return [(p.family, p.goal_id, p.variant_key) for p in pool]


def assert_pools_identical_except_pos(pools: Mapping[str, list["PairRecord"]]) -> None:
    """The five invariants across the per-SI matched pools (step 5).

    Every SI's pool must have IDENTICAL ordered prompt ids, per-family counts,
    per-goal counts, ``neg`` responses, and prompt texts — so the ONLY thing
    that differs between the N objects' training inputs is which SI supplied
    the ``pos`` completion. Each of the five invariants is checked
    independently and ALL violations are collected, so a single raise names
    every broken invariant (an assertion nobody has seen fail is not known to
    work — every invariant has its own negative test).

    ``pools`` maps ``si_id -> ordered list[PairRecord]`` for ANY number of
    SIs >= 2 (stage 3 runs it across three). The first SI (spec order) is the
    reference; every other SI is compared against it.
    """
    si_ids = list(pools)
    if len(si_ids) < 2:
        raise PoolAssertionError(
            f"need >=2 SI pools to assert identity, got {si_ids}")
    ref = si_ids[0]
    ref_pool = pools[ref]
    ref_ids = _pool_prompt_ids(ref_pool)
    ref_family = Counter(p.family for p in ref_pool)
    ref_goal = Counter(p.goal_id for p in ref_pool)
    # neg / prompt keyed by prompt_id (order-independent) so a pure REORDER trips
    # only `ordered_prompt_ids`, not these.
    ref_neg = {(p.family, p.goal_id, p.variant_key): p.neg_sha for p in ref_pool}
    ref_prompt = {(p.family, p.goal_id, p.variant_key): p.prompt_sha
                  for p in ref_pool}

    violations: list[str] = []
    for si in si_ids[1:]:
        pool = pools[si]
        if _pool_prompt_ids(pool) != ref_ids:
            violations.append(f"ordered_prompt_ids[{ref} vs {si}]")
        if Counter(p.family for p in pool) != ref_family:
            violations.append(f"per_family_counts[{ref} vs {si}]")
        if Counter(p.goal_id for p in pool) != ref_goal:
            violations.append(f"per_goal_counts[{ref} vs {si}]")
        neg = {(p.family, p.goal_id, p.variant_key): p.neg_sha for p in pool}
        if neg != ref_neg:
            violations.append(f"neg_responses[{ref} vs {si}]")
        prompt = {(p.family, p.goal_id, p.variant_key): p.prompt_sha
                  for p in pool}
        if prompt != ref_prompt:
            violations.append(f"differ_only_in_pos[{ref} vs {si}]")
    if violations:
        raise PoolAssertionError(
            "matched pools are NOT identical-except-pos (D8 step 5) — violated "
            f"invariants: {violations}")


@dataclass(frozen=True)
class PairRecord:
    """One matched/native training pair, by id + hash (no text ever)."""

    family: str
    goal_id: str
    variant_key: str
    prompt_sha: str
    neg_sha: str
    pos_sha: str

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "goal_id": self.goal_id,
            "variant_key": self.variant_key,
            "prompt_sha": self.prompt_sha,
            "neg_sha": self.neg_sha,
            "pos_sha": self.pos_sha,
        }


# --------------------------------------------------------------------------- #
# Containment self-check                                                         #
# --------------------------------------------------------------------------- #


def assert_no_store_text(manifest, *, allowed_labels: Iterable[str] = ()) -> None:
    """Walk ``manifest`` and refuse any string leaf that is not a safe opaque
    id, a hex sha, or one of this module's fixed label strings. This is the
    positive containment guarantee (JB4): a harmful prompt or completion — long
    and space-bearing — can never satisfy :data:`_SAFE_ID`, so it fails loudly
    at build time rather than reaching a committed artifact."""
    allowed = _LABEL_STRINGS | set(allowed_labels)

    def leaf(path: str, value: str) -> None:
        if value in allowed or _HEX_SHA.match(value) or _SAFE_ID.match(value):
            return
        raise MatchedPoolError(
            f"manifest leaf {path or '<root>'} is not a safe id / sha / label "
            f"(len={len(value)}) — refusing to persist possible store text")

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if not isinstance(k, str):
                    raise MatchedPoolError(f"{path}: non-string dict key")
                leaf(f"{path}.{k}" if path else k, k)
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, bool) or node is None or isinstance(node, (int, float)):
            return
        elif isinstance(node, str):
            leaf(path, node)
        else:
            raise MatchedPoolError(f"{path}: disallowed leaf type {type(node).__name__}")

    walk(manifest, "")


# --------------------------------------------------------------------------- #
# The computation (piece A)                                                     #
# --------------------------------------------------------------------------- #


def _index_rows(rows: Iterable[GradedRow], spec: PoolSpec):
    """Deduplicate and index the in-scope rows.

    Returns ``(none_by_pid, si_cands, replay_rows, benign_none, benign_cands,
    anti_cands)``:

    - ``none_by_pid[pid] -> GradedRow`` — undefended HARMFUL rows;
    - ``si_cands[si][pid] -> {candidate_idx: GradedRow}`` — SI-prompted
      harmful rows, one bucket per rejection-sampling candidate;
    - ``replay_rows`` — the non-static (adaptive) rows, kept apart;
    - ``benign_none[pid] -> GradedRow`` — undefended BENIGN rows (the shared
      ``neg``; no verdict is required of them);
    - ``benign_cands[si][pid] -> {candidate_idx: GradedRow}`` — SI-prompted
      benign rows carrying an XSTest C/P/I ``label``;
    - ``anti_cands[pid] -> {candidate_idx: GradedRow}`` — harmful rows generated
      under ``spec.anti_si_id`` (safety_refusal.anti_text), bidpo's anti-safety
      negatives, kept in their OWN bucket (never in ``si_cands``, so they never
      enter the matched intersection). Empty when ``spec.anti_si_id`` is None.

    A prompt_id seen twice under the same (condition, si, candidate_idx) with
    CONFLICTING verdicts raises — an ambiguous input must never silently pick
    one (authoritative-source rule). Identical-verdict duplicates are collapsed
    deterministically (smallest response sha). Distinct ``candidate_idx``
    values are NOT conflicts: they are rejection-sampling candidates and are
    reduced later."""
    thr = spec.threshold
    train_fams = set(spec.train_families)
    train_goals = set(spec.train_goal_ids)
    static = set(spec.static_families)
    benign_source = spec.benign_source
    benign_families = spec.benign_families

    none_by_pid: dict[tuple, GradedRow] = {}
    si_cands: dict[str, dict[tuple, dict[int, GradedRow]]] = {
        si: {} for si in spec.si_ids}
    replay_rows: list[GradedRow] = []
    benign_none: dict[tuple, GradedRow] = {}
    benign_cands: dict[str, dict[tuple, dict[int, GradedRow]]] = {
        si: {} for si in spec.si_ids}
    anti_cands: dict[tuple, dict[int, GradedRow]] = {}

    def keep_smaller_sha(prev: GradedRow, row: GradedRow) -> GradedRow:
        return row if text_sha(row.response) < text_sha(prev.response) else prev

    def place_none(bucket: dict, pid, row: GradedRow, *, check_verdict: bool) -> None:
        prev = bucket.get(pid)
        if prev is None:
            bucket[pid] = row
            return
        if check_verdict and (prev.score > thr) != (row.score > thr):
            raise MatchedPoolError(
                f"conflicting graded verdicts for prompt_id {pid} "
                f"(condition={row.condition}, si_id={row.si_id}): scores "
                f"{prev.score} vs {row.score} straddle threshold {thr}")
        bucket[pid] = keep_smaller_sha(prev, row)

    def place_cand(bucket: dict, pid, row: GradedRow, *, verdict) -> None:
        cands = bucket.setdefault(pid, {})
        prev = cands.get(row.candidate_idx)
        if prev is None:
            cands[row.candidate_idx] = row
            return
        if verdict(prev) != verdict(row):
            raise MatchedPoolError(
                f"conflicting graded verdicts for prompt_id {pid} "
                f"(condition={row.condition}, si_id={row.si_id}, "
                f"candidate_idx={row.candidate_idx}): the duplicate rows "
                "disagree — an ambiguous input must never silently pick one")
        cands[row.candidate_idx] = keep_smaller_sha(prev, row)

    for row in rows:
        if benign_source is not None and row.family in benign_families:
            # -- benign stratum: outside the attack-family/goal taxonomy -------
            if row.condition == "none":
                place_none(benign_none, row.prompt_id, row, check_verdict=False)
            elif row.condition == "si":
                if row.si_id not in benign_cands:
                    continue               # an SI we are not comparing
                if row.label not in XSTEST_LABELS:
                    raise MatchedPoolError(
                        f"benign SI row {row.prompt_id} (si_id={row.si_id}) "
                        f"carries label {row.label!r}; expected one of "
                        f"{XSTEST_LABELS} (fixed decision 1 — ungraded rows "
                        "must be filtered out before pool computation)")
                place_cand(benign_cands[row.si_id], row.prompt_id, row,
                           verdict=lambda r: r.label == "C")
            else:
                raise MatchedPoolError(
                    f"unknown condition {row.condition!r} (expected none|si)")
            continue
        if row.family not in static:
            replay_rows.append(row)      # adaptive -> replay_transfer, never primary
            continue
        if row.family not in train_fams or row.goal_id not in train_goals:
            continue                     # out of the frozen train scope
        if row.score is None:
            raise MatchedPoolError(
                f"harmful row {row.prompt_id} (condition={row.condition}, "
                f"si_id={row.si_id}) has no score — ungraded rows must be "
                "filtered out before pool computation (three-state rule)")
        pid = row.prompt_id
        if row.condition == "none":
            place_none(none_by_pid, pid, row, check_verdict=True)
        elif row.condition == "si":
            if spec.anti_si_id is not None and row.si_id == spec.anti_si_id:
                # bidpo's anti-safety negative — its OWN bucket, never a
                # comparison SI. Conflict verdict keys on compliance (score >
                # thr), the acceptance direction the reduction uses.
                place_cand(anti_cands, pid, row,
                           verdict=lambda r: r.score > thr)
            elif row.si_id in si_cands:
                place_cand(si_cands[row.si_id], pid, row,
                           verdict=lambda r: r.score > thr)
            # else: an SI we are not comparing -> drop
        else:
            raise MatchedPoolError(
                f"unknown condition {row.condition!r} (expected none|si)")
    return (none_by_pid, si_cands, replay_rows, benign_none, benign_cands,
            anti_cands)


def _pair_record(none_row: GradedRow, si_row: GradedRow) -> PairRecord:
    return PairRecord(
        family=none_row.family, goal_id=none_row.goal_id,
        variant_key=none_row.variant_key,
        prompt_sha=text_sha(none_row.prompt),
        neg_sha=text_sha(none_row.response),
        pos_sha=text_sha(si_row.response))


def _counts(records: list[PairRecord]) -> tuple[dict, dict]:
    return (dict(Counter(r.family for r in records)),
            dict(Counter(r.goal_id for r in records)))


def _pid_str(pid: tuple[str, str, str]) -> str:
    """A single opaque token for a prompt_id, for the coverage/missing lists.
    Kept url-safe (``_SAFE_ID``) so it survives the containment self-check."""
    return "/".join(pid)


def _benign_source_repr(spec: "PoolSpec"):
    """A JSON-safe rendering of ``spec.benign_source`` for the manifest: the
    single label verbatim (legacy PHTest-only), or a sorted list of family
    labels for the multi-corpus scale-up. Never a frozenset/tuple, which
    would trip the containment walk and the canonical hash."""
    if isinstance(spec.benign_source, str):
        return spec.benign_source
    return sorted(spec.benign_families)


def _det_take(pids: Iterable[tuple], n: int) -> set[tuple]:
    """A deterministic, seedless n-subset: order by the sha of the prompt_id
    (the repo's ``_det_sample`` sha-sort convention — pseudo-random but
    byte-reproducible), take the first n. Used to trim the LARGER side to the
    declared benign fraction; the same subset is used for EVERY SI, so the
    identity invariant is untouched."""
    ordered = sorted(pids, key=lambda p: text_sha(_pid_str(p)))
    return set(ordered[:n])


def _reduce_side(pids, cands_by_si, spec, accept_fn):
    """Reduce per-candidate rows to per-prompt accept/representative, per SI.

    For each SI: a prompt is ACCEPTED iff any candidate satisfies
    ``accept_fn`` (thwarting on the harmful side, XSTest ``C`` on the benign
    side); the representative ``pos`` row is the accepted candidate with the
    smallest ``candidate_idx`` (rejection-sampling semantics: the first
    accepted draw is the frozen one). Returns
    ``(accepted_by_si, rep_by_si, stats_by_si)`` where the stats carry the
    realised K and the per-prompt yield."""
    accepted_by: dict[str, set] = {}
    rep_by: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    for si in spec.si_ids:
        acc: set = set()
        rep: dict = {}
        n_prompts = n_cands = n_acc_prompts = n_acc_cands = k_max = 0
        for pid in pids:
            cands = cands_by_si[si].get(pid)
            if not cands:
                continue
            n_prompts += 1
            n_cands += len(cands)
            k_max = max(k_max, len(cands))
            ok = {i: r for i, r in cands.items() if accept_fn(r)}
            n_acc_cands += len(ok)
            if ok:
                n_acc_prompts += 1
                acc.add(pid)
                rep[pid] = ok[min(ok)]
        accepted_by[si] = acc
        rep_by[si] = rep
        stats[si] = {
            "n_prompts_with_candidates": n_prompts,
            "n_candidates": n_cands,
            "k_max": k_max,
            "k_mean": round(n_cands / n_prompts, 4) if n_prompts else 0.0,
            "n_prompts_accepted": n_acc_prompts,
            "n_candidates_accepted": n_acc_cands,
            "per_prompt_yield": (round(n_acc_prompts / n_prompts, 4)
                                 if n_prompts else 0.0),
            "per_candidate_accept_rate": (round(n_acc_cands / n_cands, 4)
                                          if n_cands else 0.0),
        }
    return accepted_by, rep_by, stats


def _bidpo_pool_block(spec: PoolSpec, benign_candidates, benign_none,
                      benign_cands, *, harmful_base, thwarted_by, si_rep,
                      anti_cands, none_success) -> dict:
    """bidpo's OWN per-SI training pool, frozen as the ``bidpo_benign``
    manifest block.

    HARMFUL side, per SI: every prompt THIS SI thwarted (the per-SI thwart
    over the full harmful base — deliberately NOT the three-way joint and NOT
    the 50/50 balance subsample, which exist for the distillation arms) that
    carries an accepted anti-safety compliance (score > threshold, smallest
    accepted candidate_idx — the bidpo_anti_neg reduction, applied over the
    per-SI set instead of the balanced selection). chosen = this SI's frozen
    thwarting refusal, rejected = the anti compliance.

    BENIGN side, per SI, polarity REVERSED: benign prompts this SI REFUSED
    while the undefended model VERIFIABLY complied. chosen = the undefended
    compliance, rejected = the SI's refusal. The refusal read reuses the
    benign intersection's EXACT reduction (any candidate labelled ``C`` ->
    the SI complied; otherwise the reduced label is the smallest candidate's
    own label): a prompt qualifies iff that reduced label is in
    ``spec.bidpo_benign_refusal_labels`` AND the undefended ``none`` row's
    XSTest label is ``C`` (``P`` is excluded from BOTH sides under the
    default, so each side is unambiguous). Loud guards:

    - an undefended row that is NEEDED (this SI refused) but UNGRADED raises,
      naming the count and the grading pass to run — ``C`` is never assumed;
    - an undefended row that itself refused/hedged (label ``I``/``P``)
      contributes nothing (no compliance to prefer): DROPPED, counted per
      label, and reported in the block;
    - an EMPTY side — benign or harmful, any SI — fails the build
      (``bidpo_benign_floor`` defaults to 0 = the zero-raise; the section-5
      >1000 floor is SUPERSEDED by section 13, which accepts si_irrelevant's
      thin benign side and records it instead).

    Per-SI, UNEQUAL: no cross-SI parity subsample.
    The three SIs train at DIFFERENT harmful:benign ratios as well as counts
    (~1:1, ~1:1, ~4.5:1 projected), so the block records, per SI, the
    harmful count, the benign count, AND the projected ratio at the default
    1:1 mix rule (benign = min(harmful, available); harmful never
    subsampled) — the confound context every bidpo cross-SI number must be
    stated with.
    """
    thr = spec.threshold
    labels_ok = tuple(spec.bidpo_benign_refusal_labels)
    ok = set(labels_ok)
    floor = spec.bidpo_benign_floor

    # -- the anti reduction over the FULL harmful base (not selected_h):
    #    accept iff COMPLIES (score > threshold), freeze the smallest
    #    accepted candidate_idx — identical rule to the bidpo_anti_neg block.
    anti_accepted: dict[tuple, GradedRow] = {}
    for pid, cands in anti_cands.items():
        if pid not in harmful_base:
            continue                     # out of scope / scoring-stripped
        acc = {i: r for i, r in cands.items() if r.score > thr}
        if acc:
            anti_accepted[pid] = acc[min(acc)]

    per_si: dict[str, dict] = {}
    ungraded_by_si: dict[str, int] = {}
    for si in spec.si_ids:
        # ---- harmful side: this SI's OWN thwart set + an accepted anti-neg
        thwarted = thwarted_by[si]
        h_entries: list[dict] = []
        for pid in sorted(thwarted):
            anti_row = anti_accepted.get(pid)
            if anti_row is None:
                continue                 # no accepted anti compliance at K
            si_row = si_rep[si][pid]
            h_entries.append({
                "family": pid[0], "goal_id": pid[1], "variant_key": pid[2],
                "prompt_sha": text_sha(si_row.prompt),
                "chosen_sha": text_sha(si_row.response),
                "rejected_sha": text_sha(anti_row.response),
                "requirement_a": ("strict" if pid in none_success
                                  else "relaxed"),
            })
        harmful_block = {
            "n_thwarted": len(thwarted),
            "n_with_anti_neg": len(h_entries),
            "n_missing_anti_neg": len(thwarted) - len(h_entries),
            "n_strict": sum(1 for e in h_entries
                            if e["requirement_a"] == "strict"),
            "ordered_pairs": h_entries,
        }

        # ---- benign side: this SI refused, the undefended VERIFIABLY complied
        n_with_rows = n_si_refused = n_ungraded = 0
        dropped = {"P": 0, "I": 0}
        b_entries: list[dict] = []
        for pid in sorted(benign_candidates):
            cands = benign_cands[si].get(pid)
            if not cands:
                continue
            n_with_rows += 1
            if any(r.label == "C" for r in cands.values()):
                continue            # the SI complied — nothing to reverse
            rep = cands[min(cands)]  # the reduced label's own row
            if rep.label not in ok:
                continue            # e.g. a P under the I-only default
            n_si_refused += 1
            none_row = benign_none[pid]
            if none_row.label is None:
                n_ungraded += 1     # counted; raised for ALL SIs below
                continue
            if none_row.label not in XSTEST_LABELS:
                raise MatchedPoolError(
                    f"benign none row {pid} carries XSTest label "
                    f"{none_row.label!r}; expected one of {XSTEST_LABELS} — "
                    "a grading bug must never silently admit or drop a "
                    "chosen side")
            if none_row.label != "C":
                dropped[none_row.label] += 1   # undefended also refused/hedged
                continue
            b_entries.append({
                "family": pid[0], "goal_id": pid[1], "variant_key": pid[2],
                "prompt_sha": text_sha(none_row.prompt),
                "chosen_sha": text_sha(none_row.response),
                "rejected_sha": text_sha(rep.response),
                "rejected_label": rep.label,
            })
        if n_ungraded:
            ungraded_by_si[si] = n_ungraded
        benign_block = {
            "n_prompts_with_si_rows": n_with_rows,
            "n_si_refused": n_si_refused,
            "n_dropped_undefended_not_complied": dropped,
            "n_qualified": len(b_entries),
            "ordered_pairs": b_entries,
        }
        per_si[si] = {"harmful": harmful_block, "benign": benign_block}

    if ungraded_by_si:
        raise MatchedPoolError(
            "bidpo benign reversed side needs the UNDEFENDED benign responses "
            "graded, but "
            + ", ".join(f"{si}: {n}" for si, n in sorted(ungraded_by_si.items()))
            + " needed `none` row(s) carry no XSTest label — the chosen side "
            "must be a VERIFIED compliance, never an assumed one. Run the "
            "benign-none grading pass (build_two_si_pool.py grade-store with "
            "pool.bidpo_benign.enabled) before freezing the manifest.")

    failing_h = {si: e["harmful"]["n_with_anti_neg"]
                 for si, e in per_si.items()
                 if e["harmful"]["n_with_anti_neg"] == 0}
    if failing_h:
        thw = ", ".join(f"{si}={per_si[si]['harmful']['n_thwarted']}"
                        for si in spec.si_ids)
        raise PoolFloorError(
            "bidpo harmful side is EMPTY for "
            f"{sorted(failing_h)} — no thwarted prompt carries an accepted "
            f"anti-safety compliance (thwarted per SI: {thw}). Run/raise-K "
            "on generate-anti-neg + grade-anti-neg, or disable "
            "pool.bidpo_benign; a bidpo with no harmful preferences is "
            "untrainable and the build fails rather than shipping it.")
    failing_b = {si: e["benign"]["n_qualified"] for si, e in per_si.items()
                 if e["benign"]["n_qualified"] <= floor}
    if failing_b:
        counts = ", ".join(f"{si}={per_si[si]['benign']['n_qualified']}"
                           for si in spec.si_ids)
        raise PoolFloorError(
            "bidpo benign reversed side is EMPTY or below the configured "
            f"floor (> {floor} qualified pairs per SI; refusal labels "
            f"{list(labels_ok)}): qualified per SI — {counts}; failing: "
            f"{sorted(failing_b)}. Section 13 accepts a THIN side (the "
            "section-5 >1000 floor is superseded; the realised count is "
            "recorded instead) but never an empty one: widen the refusal-"
            "label setting to ('I', 'P'), grade/generate more benign rows, "
            "or disable pool.bidpo_benign — never ship a harmful-only bidpo "
            "wearing the benign-balanced configuration.")

    # -- the projected mix at the DEFAULT 1:1 rule (benign = min(harmful,
    #    available); harmful never subsampled). The trainer records the
    #    REALISED numbers per object; these projections put all three SIs'
    #    ratios in one frozen place.
    for si in spec.si_ids:
        n_h = per_si[si]["harmful"]["n_with_anti_neg"]
        n_b = min(n_h, per_si[si]["benign"]["n_qualified"])
        per_si[si]["projected_mix_at_default_1to1"] = {
            "n_harmful": n_h,
            "n_benign": n_b,
            "ratio_harmful_to_benign": round(n_h / n_b, 4),
        }

    return {
        "note": _BIDPO_BENIGN_NOTE,
        "enabled": True,
        "anti_si_id": spec.anti_si_id,
        "refusal_labels": list(labels_ok),
        "undefended_required_label": "C",
        "floor": floor,
        "per_si": per_si,
    }


def build_matched_pool(rows: Iterable[GradedRow], spec: PoolSpec) -> dict:
    """Compute the frozen matched-pool manifest from already-graded rows.

    Runs the 8-step ordering (steps 4–8 here; 1–3 are the frozen spec +
    graded inputs) across ALL of ``spec.si_ids`` (N-SI: the intersection is
    over every SI's thwarted set), asserts the five identity invariants on
    the matched intersection, computes the benign all-comply intersection and
    the declared-fraction balance when a benign half is declared, enforces the
    hard floor and the composition assertion (stage-3), records the native
    per-SI pools as a labelled secondary, decides the regime (matched vs the
    explicit native fallback — never a silent subsample, and never ANY
    fallback under the stage-3 gates), and hashes everything load-bearing into
    ``manifest_sha256``. The returned dict is byte-reproducible from the same
    inputs and carries counts/ids/hashes only."""
    rows = list(rows)

    # -- no evaluation-only corpus enters a training pool: the ReNeLLM paper
    #    corpus ships under terms that forbid training on it, so a row whose
    #    prompt is one of its rows refuses the whole build (counts only). ----
    from .attacks import eval_only
    eval_only.refuse_texts(
        (r.get("prompt") if isinstance(r, dict) else getattr(r, "prompt", None)
         for r in rows),
        consumer="the training-pool builder (build_matched_pool)")

    # -- the holdout is GOAL-LEVEL, not family-level (since 2026-08-28;
    #    family disjointness is retired — do not reintroduce it). The
    #    attack families (bon/pasttense/codeattack/renellm/grok420) are
    #    deliberately IDENTICAL between train and eval; generalization is
    #    measured on goals the fit never saw. --------------------------
    goal_overlap = sorted(set(spec.train_goal_ids) & set(spec.eval_goal_ids))
    if goal_overlap:
        raise MatchedPoolError(
            f"train/eval harm goals overlap (the goal split IS the holdout): "
            f"{goal_overlap}")
    # When families are shared, an empty eval goal set is not a thin
    # holdout — it is NO holdout, and every "held-out" number downstream
    # would be in-sample while still labelled held-out. Refuse rather than
    # produce a pool that cannot support the claim its consumers will make.
    fam_overlap = sorted(set(spec.train_families) & set(spec.eval_families))
    if fam_overlap and not set(spec.eval_goal_ids):
        raise MatchedPoolError(
            f"nothing is held out: families {fam_overlap} appear on both "
            f"sides (family-level holdout is retired) and eval_goal_ids is "
            f"EMPTY. Every 'held-out' number from this pool would be "
            f"in-sample while still labelled held out. Provide a goal split.")

    (none_by_pid, si_cands, replay_rows,
     benign_none, benign_cands, anti_cands) = _index_rows(rows, spec)
    thr = spec.threshold

    # -- train/scoring disjointness: remove any candidate
    #    whose NORMALISED prompt is a scoring (eval-control) prompt, BEFORE the
    #    intersections and the floor count. Everything downstream keys off the
    #    none-row candidates, so stripping here strips the pool. --------------
    def _strip_scoring(bucket: dict) -> tuple[dict, int]:
        if not spec.scoring_prompt_shas:
            return bucket, 0
        kept = {pid: r for pid, r in bucket.items()
                if normalized_text_sha(r.prompt) not in spec.scoring_prompt_shas}
        return kept, len(bucket) - len(kept)

    none_by_pid, n_scoring_removed_h = _strip_scoring(none_by_pid)
    benign_none, n_scoring_removed_b = _strip_scoring(benign_none)

    none_success = {pid for pid, r in none_by_pid.items() if r.score > thr}

    # -- harmful side: the primary
    #    matched intersection no longer REQUIRES the undefended model to have
    #    complied. The base is now EVERY harmful prompt with a none row (the
    #    relaxed pool is the DEFAULT); requirement (a) — a genuine harmful
    #    compliance as neg — survives as (i) an identifiable STRICT SUBSET
    #    recorded in the manifest so any result can be reported on it alone,
    #    and (ii) the opt-in `require_undefended_compliance` regime bidpo uses
    #    (a preference arm needs a real dispreferred example). The
    #    per-SI thwart reduction runs over the whole base either way. ----------
    harmful_base = (set(none_success) if spec.require_undefended_compliance
                    else set(none_by_pid))
    thwarted_by, si_rep, rs_stats = _reduce_side(
        harmful_base, si_cands, spec, lambda r: not (r.score > thr))

    matched_ids = set(harmful_base)
    for si in spec.si_ids:
        matched_ids &= thwarted_by[si]

    # -- per-family caps: deterministically trim any family whose matched
    #    pairs exceed its configured cap, BEFORE the 50/50 balance. The cap
    #    changes which side is binding: trimming a dominant harmful family
    #    below the benign matched count inverts the balance so it cuts BENIGN
    #    down to the harmful total rather than the reverse. Reuses the
    #    sha-ordered `_det_take` convention (identical to the balance's own
    #    subset rule), so the kept pairs are reproducible across runs,
    #    orderings and machines. Empty `family_caps` -> no-op. NEVER a silent
    #    truncation: the cap value, the drop count and the surviving
    #    composition are all recorded in the `family_caps` manifest block below.
    family_cap_report: dict[str, dict] = {}
    if spec.family_caps:
        by_fam_matched: dict[str, set] = defaultdict(set)
        for pid in matched_ids:
            by_fam_matched[pid[0]].add(pid)
        for fam in sorted(spec.family_caps):
            cap = int(spec.family_caps[fam])
            present = by_fam_matched.get(fam, set())
            n_before = len(present)
            if n_before > cap:
                matched_ids -= (present - _det_take(present, cap))
            family_cap_report[fam] = {
                "cap": cap,
                "n_before": n_before,
                "n_after": min(n_before, cap),
                "n_dropped": max(0, n_before - cap),
            }

    # the identifiable requirement-(a) subset: prompts the UNDEFENDED model
    # actually COMPLIED with, so neg is a genuine harmful compliance.
    strict_matched = matched_ids & none_success

    # -- the RELAXED requirement-(a) stratum, config-gated and SEPARATELY
    #    LABELLED, NEVER mixed into the primary intersection. It admits
    #    prompts the UNDEFENDED model already REFUSED (score <= thr, so
    #    `none_refused` — the base model declined without a jailbreak) that
    #    every SI ALSO thwarts. Its neg is a base refusal and its pos an
    #    SI-styled refusal, so the pair teaches "refuse in this instruction's
    #    style" rather than "refuse instead of complying" — a DIFFERENT
    #    learning object, which is exactly why it is a stratum a result can
    #    be reported with and without, never a silent mix-in. ----------------
    relaxed_matched: set = set()
    relaxed_thwarted_by: dict[str, set] = {}
    relaxed_rep: dict[str, dict] = {}
    relaxed_rs_stats: dict[str, dict] = {}
    if spec.relaxed_stratum_enabled and spec.require_undefended_compliance:
        # Only a SEPARATE stratum in the strict regime; under the
        # default these already live in the primary intersection (folded in),
        # so emitting them here too would double-count.
        none_refused = {pid for pid, r in none_by_pid.items()
                        if not (r.score > thr)}
        relaxed_thwarted_by, relaxed_rep, relaxed_rs_stats = _reduce_side(
            none_refused, si_cands, spec, lambda r: not (r.score > thr))
        relaxed_matched = set(none_refused)
        for si in spec.si_ids:
            relaxed_matched &= relaxed_thwarted_by[si]

    # -- benign side (stage-3): the all-SIs-comply intersection ----------------
    benign_active = spec.benign_source is not None
    benign_candidates = set(benign_none)
    accepted_b: dict[str, set] = {}
    b_rep: dict[str, dict] = {}
    xstest_counts: dict[str, dict] = {}
    benign_matched: set = set()
    if benign_active:
        from ..data.content_sets import select_xstest_complied

        _, b_rep, _b_stats = _reduce_side(
            benign_candidates, benign_cands, spec, lambda r: r.label == "C")
        for si in spec.si_ids:
            # reduced per-prompt label: "C" iff any candidate complied, else
            # the smallest candidate's own label — fed through the SHARED
            # C-only acceptance helper so admission and its counts are the
            # loader module's, never a re-implementation.
            reduced: dict[tuple, str] = {}
            for pid in sorted(benign_candidates):
                cands = benign_cands[si].get(pid)
                if not cands:
                    continue
                if any(r.label == "C" for r in cands.values()):
                    reduced[pid] = "C"
                else:
                    reduced[pid] = cands[min(cands)].label
            accepted, counts = select_xstest_complied(reduced)
            accepted_b[si] = set(accepted)
            xstest_counts[si] = counts
        benign_matched = set(benign_candidates)
        for si in spec.si_ids:
            benign_matched &= accepted_b[si]

    # -- bidpo's OWN per-SI pool:
    #    a SEPARATE block. Its HARMFUL side is the per-SI thwart set (NOT the
    #    joint intersection, NOT the balance subsample) with an accepted anti
    #    compliance as the rejected side; its BENIGN side is the reversed-
    #    polarity material the intersection above DISCARDS (this-SI-refused
    #    is the complement of all-SIs-complied) — so the primary pool and
    #    every other arm are untouched by construction. Computed AFTER the
    #    scoring-disjointness strip (which binds bidpo too),
    #    raising loudly on ungraded undefended rows and on any EMPTY side. --
    bidpo_benign_block = None
    if spec.bidpo_benign_enabled:
        bidpo_benign_block = _bidpo_pool_block(
            spec, benign_candidates, benign_none, benign_cands,
            harmful_base=harmful_base, thwarted_by=thwarted_by,
            si_rep=si_rep, anti_cands=anti_cands, none_success=none_success)

    # -- the hard floor: counted AFTER grading, the N-way
    #    intersection and train/scoring disjointness; below EITHER floor the
    #    build FAILS — never a fallback, never a thin pool. --------------------
    n_h, n_b = len(matched_ids), len(benign_matched)
    if spec.hard_floor_pairs > 0:
        floor = spec.hard_floor_pairs
        sides = [("harmful-refusal", n_h)]
        if benign_active:
            sides.append(("benign-compliance", n_b))
        failing = [(name, n) for name, n in sides if n < floor]
        if failing:
            per_si_h = ", ".join(
                f"{si}={len(thwarted_by[si])}" for si in spec.si_ids)
            per_si_b = (", ".join(
                f"{si}={len(accepted_b[si])}" for si in spec.si_ids)
                if benign_active else "n/a (no benign half declared)")
            side_report = "; ".join(
                f"{name} matched pairs = {n} ({'FAIL' if n < floor else 'ok'}, "
                f"floor {floor})" for name, n in sides)
            raise PoolFloorError(
                f"hard floor violated (fixed decision 4: >= {floor} "
                "harmful-refusal AND benign-compliance pairs per SI, counted "
                "after grading, the N-way SI intersection, and train/scoring "
                f"disjointness): {side_report}. Realised per SI — harmful "
                f"thwarted: {per_si_h}; benign complied: {per_si_b}. The build "
                "FAILS rather than shipping a thin or confounded pool: it "
                "never falls back to native_per_si. Raise the rejection-"
                "sampling candidate count K, generate/grade more rows, or add "
                "families — never lower the floor.")

    # -- the declared-fraction balance, counted in pairs:
    #    trim the LARGER side to the declared fraction with a deterministic
    #    sha-ordered subset, identical for every SI. Take the full available
    #    pool above the floor. ------------------------------------------------
    if benign_active:
        f = float(spec.benign_fraction)
        h_take = min(n_h, int(n_b * (1.0 - f) / f))
        b_take = min(n_b, int(n_h * f / (1.0 - f)))
        selected_h = _det_take(matched_ids, h_take)
        selected_b = _det_take(benign_matched, b_take)
    else:
        selected_h, selected_b = set(matched_ids), set()

    def order(pids: Iterable[tuple]) -> list[tuple]:
        return sorted(pids)  # (family, goal_id, variant_key) — stable, seedless

    matched_order = order(selected_h | selected_b)
    n_matched = len(matched_order)

    # build the per-SI matched pools (only pos differs) and assert (step 5)
    matched_pools: dict[str, list[PairRecord]] = {}
    for si in spec.si_ids:
        matched_pools[si] = [
            _pair_record(none_by_pid[pid], si_rep[si][pid])
            if pid in selected_h else
            _pair_record(benign_none[pid], b_rep[si][pid])
            for pid in matched_order]
    assert_pools_identical_except_pos(matched_pools)

    # Per-pair requirement-(a) label so a result can be reported on the
    # STRICT ("undefended complied") subset alone. Harmful pids the
    # undefended model complied with are "strict"; the rest of the harmful
    # side is "relaxed" (undefended already refused); the benign half is
    # "benign".
    def _req_a(pid):
        if pid in selected_b:
            return "benign"
        return "strict" if pid in none_success else "relaxed"

    # -- bidpo anti-safety negatives (bidpo amendment 2026-08-20). Reduce the
    #    anti candidates (generated under spec.anti_si_id = safety_refusal.
    #    anti_text) over the HARMFUL pairs actually in the balanced pool
    #    (selected_h), accepting a candidate iff it COMPLIES (score > thr — the
    #    OPPOSITE direction from a pos row, which is accepted when it thwarts).
    #    The frozen anti-neg is the smallest accepted candidate_idx (the first
    #    accepted draw — the same rejection-sampling rule as the pos side). This
    #    is a SEPARATE, bidpo-only block; matched_intersection's neg is never
    #    touched, so the identity invariant across SIs holds. Records counts +
    #    shas + the realised per-prompt yield; NEVER any completion text. -----
    anti_block = None
    if spec.anti_si_id is not None:
        anti_accepted: dict[tuple, GradedRow] = {}
        a_prompts = a_cands = a_acc_prompts = a_acc_cands = a_kmax = 0
        for pid in selected_h:
            cands = anti_cands.get(pid)
            if not cands:
                continue
            a_prompts += 1
            a_cands += len(cands)
            a_kmax = max(a_kmax, len(cands))
            ok = {i: r for i, r in cands.items() if r.score > thr}  # COMPLY
            a_acc_cands += len(ok)
            if ok:
                a_acc_prompts += 1
                anti_accepted[pid] = ok[min(ok)]
        anti_ordered = [pid for pid in matched_order if pid in selected_h]
        anti_block = {
            "note": _ANTI_NEG_NOTE,
            "anti_si_id": spec.anti_si_id,
            "acceptance": _ANTI_ACCEPT_LABEL,
            "threshold": thr,
            "n_harmful_pairs": len(anti_ordered),
            "n_with_anti_neg": sum(1 for pid in anti_ordered
                                   if pid in anti_accepted),
            "n_missing_anti_neg": sum(1 for pid in anti_ordered
                                      if pid not in anti_accepted),
            "rejection_sampling": {
                "n_prompts_with_candidates": a_prompts,
                "n_candidates": a_cands,
                "k_max": a_kmax,
                "k_mean": round(a_cands / a_prompts, 4) if a_prompts else 0.0,
                "n_prompts_accepted": a_acc_prompts,
                "n_candidates_accepted": a_acc_cands,
                "per_prompt_yield": (round(a_acc_prompts / a_prompts, 4)
                                     if a_prompts else 0.0),
                "per_candidate_accept_rate": (round(a_acc_cands / a_cands, 4)
                                              if a_cands else 0.0),
            },
            "ordered_pairs": [
                {"family": pid[0], "goal_id": pid[1], "variant_key": pid[2],
                 "anti_neg_sha": text_sha(anti_accepted[pid].response)}
                for pid in anti_ordered if pid in anti_accepted],
        }

    # -- the DIMENSION-AWARE floor: the AUTHORITATIVE size gate, replacing
    #    the old scalar ">= 200 per side" as the binding one. The closed-form
    #    arms fit a d-dimensional object (the alphakl probes at d = 4096);
    #    n < d is rank-deficient and produced a defence with no measurable
    #    effect elsewhere. Each arm's realised fit-set n (the balanced pool
    #    minus the nominal holdout carve and its own calibration carve) is
    #    compared against its d, and n < d for ANY arm FAILS the build with
    #    every arm's realised n and d named. Empty fit_dims disables it
    #    (legacy). Counted on the balanced total, which is what the trainers'
    #    split actually consumes. ------------------------------------------
    dimension_floor_block = None
    if spec.fit_dims:
        arm_rows = []
        for arm in sorted(spec.fit_dims):
            d = int(spec.fit_dims[arm])
            n_fit = spec.fit_rows(n_matched, arm)
            arm_rows.append({
                "arm": arm, "d": d, "n_fit": n_fit,
                "calibration_rows": int(spec.fit_calibration_rows.get(arm, 0)),
                "ok": n_fit >= d,
            })
        binding = max(arm_rows, key=lambda r: r["d"] + r["calibration_rows"])
        dimension_floor_block = {
            "enabled": True,
            "holdout_frac": float(spec.fit_holdout_frac),
            "n_matched": n_matched,
            "binding_arm": binding["arm"],
            "per_arm": arm_rows,
            "all_arms_meet_dimension": all(r["ok"] for r in arm_rows),
        }
        failing = [r for r in arm_rows if not r["ok"]]
        if failing:
            table = "; ".join(
                f"{r['arm']}: n = {r['n_fit']} vs d = {r['d']} "
                f"({'FAIL' if not r['ok'] else 'ok'})" for r in arm_rows)
            raise PoolDimensionError(
                "dimension-aware floor violated (user amendment 2026-08-20: "
                "every fitted arm needs n >= d, counted on the balanced pool "
                f"of {n_matched} pairs after the nominal holdout carve "
                f"(frac={float(spec.fit_holdout_frac)}) and each arm's "
                f"calibration carve): {table}. Realised fit rows per arm are "
                "below the fitted dimension, so the closed-form solve is "
                "rank-deficient — the regime that produced a defence with no "
                "measurable effect. The build FAILS rather than shipping an "
                "underdetermined pool: generate/grade more matched pairs "
                "(raise the pool target), never lower d.")

    # -- composition assertion: the realised pool must
    #    match the DECLARED intent — every train family present with a
    #    non-zero count, benign fraction on target — and can never be
    #    satisfied vacuously (an empty pool raises). --------------------------
    fam_counts = dict(Counter(p.family for p in matched_pools[spec.si_ids[0]]))
    composition_block = None
    if spec.benign_fraction is not None:
        if n_matched == 0:
            raise PoolCompositionError(
                "composition assertion cannot pass vacuously: the balanced "
                "matched pool is EMPTY (declared train_families="
                f"{list(spec.train_families)}, benign_source="
                f"{spec.benign_source!r}, benign_fraction="
                f"{spec.benign_fraction}) — nothing to freeze.")
        n_benign_final = sum(1 for p in matched_pools[spec.si_ids[0]]
                             if p.family in spec.benign_families)
        realised_fraction = n_benign_final / n_matched
        missing_fams = [fam for fam in spec.train_families
                        if fam_counts.get(fam, 0) == 0]
        # ...and the SILENTLY-THIN case, which presence cannot see. Coverage
        # is measured over distinct GOALS, not rows: a family emitting six
        # variants of one goal has a healthy row count and one goal's worth
        # of signal. renellm is the live example -- it is frozen-corpus, so
        # it can only reach goals its pinned corpus already contains.
        pool0 = matched_pools[spec.si_ids[0]]
        fam_goals = {fam: len({p.goal_id for p in pool0 if p.family == fam})
                     for fam in spec.train_families}
        # PER-FAMILY denominator: the goals THIS family was offered, not the
        # pool-wide goal union. The union is wrong whenever families do not
        # share a goal space, and this pool is exactly that case: the five
        # transform families are crossed with ~300 shared JBB/HarmBench goals,
        # while wildjailbreak_harmful is a standalone corpus whose every row
        # is its own goal and which therefore contributed 2,907 of the pool's
        # 3,091 goals. Measured against the union, NO transform family could
        # reach 10% however complete it was -- bon reported "184/3091 (6.0%)"
        # when it had realised 184 of its own 300. That is the assertion
        # firing on pool SHAPE rather than on thinness (2026-08-26).
        # What "thin" means is: of the goals this family offered, few survived.
        offered = {}
        for r in rows:
            fam = getattr(r, "family", None) or (
                r.get("family") if isinstance(r, dict) else None)
            gid = getattr(r, "goal_id", None) or (
                r.get("goal_id") if isinstance(r, dict) else None)
            if fam is not None and gid is not None:
                offered.setdefault(fam, set()).add(gid)
        thin_fams = []
        if spec.min_family_goal_coverage > 0:
            for fam in spec.train_families:
                if fam in spec.benign_families or fam_counts.get(fam, 0) == 0:
                    continue
                denom = len(offered.get(fam, ())) or None
                if not denom:
                    continue   # nothing offered: the missing-family check owns this
                frac = fam_goals[fam] / denom
                if frac < spec.min_family_goal_coverage:
                    thin_fams.append(
                        f"{fam} covers {fam_goals[fam]}/{denom} of its OWN "
                        f"offered goals ({frac:.1%} < "
                        f"{spec.min_family_goal_coverage:.0%})")
        off_target = abs(realised_fraction - float(spec.benign_fraction)) \
            > float(spec.benign_fraction_tol)
        if missing_fams or thin_fams or off_target:
            problems = []
            if thin_fams:
                problems.append(
                    "train families present but SILENTLY THIN (a thin family "
                    "looks like data; an absent one is visible): "
                    + "; ".join(thin_fams))
            if missing_fams:
                problems.append(
                    f"declared train families with ZERO realised pairs: "
                    f"{missing_fams}")
            if off_target:
                problems.append(
                    f"realised benign fraction {realised_fraction:.4f} is "
                    f"outside declared {float(spec.benign_fraction)} +/- "
                    f"{float(spec.benign_fraction_tol)}")
            raise PoolCompositionError(
                "realised pool composition does not match the declared config "
                f"(stage-3 success criterion 1): {'; '.join(problems)}. "
                f"Declared train_families={list(spec.train_families)}, "
                f"benign_source={spec.benign_source!r}, benign_fraction="
                f"{spec.benign_fraction}; realised per-family counts="
                f"{fam_counts}, pool size={n_matched}. The build FAILS "
                "rather than freezing a pool narrower than its spec (the "
                "stage-2 BoN-only regression this assertion exists to stop).")
        composition_block = {
            "declared": {
                "train_families": sorted(spec.train_families),
                "benign_source": _benign_source_repr(spec),
                "benign_fraction": float(spec.benign_fraction),
                "benign_fraction_tol": float(spec.benign_fraction_tol),
            },
            "realised": {
                "per_family_counts": fam_counts,
                "n_pairs": n_matched,
                "n_benign": n_benign_final,
                "n_harmful": n_matched - n_benign_final,
                "benign_fraction": round(realised_fraction, 6),
            },
            "pre_balance": {
                "n_harmful_matched": n_h,
                "n_benign_matched": n_b,
                "harmful_taken": len(selected_h),
                "benign_taken": len(selected_b),
            },
            "asserted": True,
        }

    # step 7: native per-SI pools (all-thwarted per SI) as labelled secondary
    native_pools: dict[str, list[PairRecord]] = {}
    for si in spec.si_ids:
        native_pools[si] = [
            _pair_record(none_by_pid[pid], si_rep[si][pid])
            for pid in order(thwarted_by[si])]

    # per-source yields (D: each source's own floor, never abundance masking
    # emptiness). A "source" is a labelled subset (defaults to the family).
    sources = sorted({r.source_label for r in none_by_pid.values()}
                     | {r.source_label for si in spec.si_ids
                        for cands in si_cands[si].values()
                        for r in cands.values()})
    per_source: dict[str, dict] = {}
    all_floors_ok = True
    for src in sources:
        cand = {pid for pid, r in none_by_pid.items() if r.source_label == src}
        succ = cand & none_success
        # thwart is now counted over the whole base, not gated on undefended
        # compliance; n_none_success stays reported for the strict view.
        thw = {si: (cand & thwarted_by[si]) for si in spec.si_ids}
        matched_src = cand & matched_ids
        floor = spec.floor_for(src)
        ok = len(matched_src) >= floor
        all_floors_ok = all_floors_ok and ok
        entry = {
            "n_candidates": len(cand),
            "n_none_success": len(succ),
            "n_matched": len(matched_src),
            "n_strict_matched": len(cand & strict_matched),
            "floor": floor,
            "floor_ok": ok,
        }
        for si in spec.si_ids:
            entry[f"n_thwarted_{si}"] = len(thw[si])
        per_source[src] = entry

    # benign per-source yields + floors (scale-up amendment 2026-08-20): the
    # benign half draws from several corpora (PHTest, jailbreak-styled benign,
    # the AlphaSteer benign pool), each its own family/source, so one corpus's
    # abundance can never mask another's emptiness. Same source_floors seam as
    # the harmful side; counted on the all-SIs-comply intersection.
    benign_per_source: dict[str, dict] = {}
    benign_floors_ok = True
    if benign_active:
        # every DECLARED benign family plus any source label seen, so a
        # declared-but-empty corpus shows n=0 and trips its floor rather than
        # being silently invisible (abundance never masks emptiness).
        benign_sources = sorted(
            spec.benign_families | {r.source_label for r in benign_none.values()})
        for src in benign_sources:
            cand = {pid for pid, r in benign_none.items()
                    if r.source_label == src}
            matched_src = cand & benign_matched
            floor = spec.floor_for(src)
            ok = len(matched_src) >= floor
            benign_floors_ok = benign_floors_ok and ok
            benign_per_source[src] = {
                "n_candidates": len(cand),
                "n_matched_all_si_comply": len(matched_src),
                "floor": floor,
                "floor_ok": ok,
            }

    # coverage: what piece B still has to generate — none-successful prompts
    # lacking any graded SI candidate (per SI). This is exactly the gap
    # generate-pos fills. `n_all_candidates_rejected` counts prompts whose
    # candidates ALL failed acceptance — the ones a HIGHER K could recover.
    coverage: dict[str, dict] = {}
    for si in spec.si_ids:
        present = {pid for pid in harmful_base if si_cands[si].get(pid)}
        missing = harmful_base - present
        coverage[si] = {
            "n_harmful_base": len(harmful_base),
            "n_none_success": len(none_success),
            "n_si_rows_present": len(present),
            "n_missing": len(missing),
            "n_all_candidates_rejected": len(present - thwarted_by[si]),
            "missing_prompt_ids": [_pid_str(pid) for pid in order(missing)],
        }

    benign_block = None
    if benign_active:
        b_coverage: dict[str, dict] = {}
        for si in spec.si_ids:
            present = {pid for pid in benign_candidates
                       if benign_cands[si].get(pid)}
            missing = benign_candidates - present
            b_coverage[si] = {
                "n_candidates": len(benign_candidates),
                "n_si_rows_present": len(present),
                "n_missing": len(missing),
                "missing_prompt_ids": [_pid_str(pid) for pid in order(missing)],
            }
        benign_block = {
            "note": _BENIGN_NOTE,
            "source": _benign_source_repr(spec),
            "sources": sorted(spec.benign_families),
            "n_candidates": len(benign_candidates),
            "n_matched_all_si_comply": n_b,
            "per_si_xstest_counts": xstest_counts,
            "per_si_n_complied": {si: len(accepted_b[si])
                                  for si in spec.si_ids},
            "per_source_yields": benign_per_source,
            "all_sources_meet_floor": benign_floors_ok,
            "coverage": b_coverage,
        }

    # -- the RELAXED requirement-(a) stratum, built and labelled. Its per-SI
    #    pools obey the same identity invariant (only pos differs), so it is
    #    a drop-in extra training block a downstream run can add OR omit —
    #    but it lives in its OWN manifest key and is NEVER folded into
    #    matched_intersection. neg is a base refusal (the undefended model
    #    already declined); pos is the SI-styled refusal, so the object
    #    learns "refuse in this SI's style". --------------------------------
    relaxed_block = None
    if spec.relaxed_stratum_enabled and spec.require_undefended_compliance:
        relaxed_order = order(relaxed_matched)
        relaxed_pools: dict[str, list[PairRecord]] = {
            si: [_pair_record(none_by_pid[pid], relaxed_rep[si][pid])
                 for pid in relaxed_order]
            for si in spec.si_ids}
        if relaxed_pools[spec.si_ids[0]]:
            assert_pools_identical_except_pos(relaxed_pools)
        relaxed_block = {
            "note": _RELAXED_NOTE,
            "enabled": True,
            "requirement_a": "relaxed",
            "n": len(relaxed_order),
            "ordered_pairs": [
                {**relaxed_pools[spec.si_ids[0]][i].as_dict(),
                 "pos_sha": {si: relaxed_pools[si][i].pos_sha
                             for si in spec.si_ids}}
                for i in range(len(relaxed_order))],
            "per_family_counts": _counts(relaxed_pools[spec.si_ids[0]])[0],
            "per_goal_counts": _counts(relaxed_pools[spec.si_ids[0]])[1],
            "per_si_thwarted": {si: len(relaxed_thwarted_by[si])
                                for si in spec.si_ids},
            "rejection_sampling": relaxed_rs_stats,
        }

    # replay_transfer stratum (adaptive material) — labelled, NEVER in primary
    replay_families = Counter(r.family for r in replay_rows)
    replay_block = {
        "note": _REPLAY_NOTE,
        "families": sorted(replay_families),
        "per_family_counts": dict(replay_families),
        "n_rows": len(replay_rows),
    }

    # -- regime decision (step 8): matched, or the EXPLICIT native fallback.
    #    Under the stage-3 gates (hard floor / benign half) the fallback is
    #    DISABLED: a thin pool has already raised, and min_pool_size raises
    #    too rather than shipping a confounded native pool. --------------------
    fallback_reason = None
    if spec.strict:
        if n_matched < spec.min_pool_size:
            raise PoolFloorError(
                f"matched pool ({n_matched} pairs) is below min_pool_size "
                f"({spec.min_pool_size}) and the native fallback is DISABLED "
                "under the stage-3 gates (hard_floor_pairs/benign half): the "
                "build fails rather than shipping a confounded native pool.")
        regime = REGIME_MATCHED
    elif n_matched < spec.min_pool_size:
        regime = REGIME_NATIVE
        fallback_reason = "intersection_below_min_pool_size"
    else:
        regime = REGIME_MATCHED
    if regime == REGIME_MATCHED:
        pool_sizes = {si: n_matched for si in spec.si_ids}
    else:
        pool_sizes = {si: len(native_pools[si]) for si in spec.si_ids}

    manifest = {
        "version": MANIFEST_VERSION,
        # -- the contract two_si_run._load_pool_regime reads --------------------
        "regime": regime,
        "pool_sizes": pool_sizes,
        "headline_confounded_by_pool_composition": regime == REGIME_NATIVE,
        "regime_note": _REGIME_NOTE[regime],
        "fallback_reason": fallback_reason,
        "si_ids": list(spec.si_ids),
        "threshold": spec.threshold,
        "min_pool_size": spec.min_pool_size,
        # -- per-family caps, recorded so a capped pool never reads as "this
        #    is all we had": the configured caps, the per-family
        #    before/after/dropped counts, and the total dropped. Empty when
        #    no caps are configured. ---------------------------------------
        "family_caps": {
            "configured": {f: int(n) for f, n in spec.family_caps.items()},
            "applied": family_cap_report,
            "n_dropped_total": sum(e["n_dropped"]
                                   for e in family_cap_report.values()),
        },
        # -- the stage-3 hard floor, recorded ---------------------------------
        "hard_floor": {
            "floor_pairs": spec.hard_floor_pairs,
            "enforced": spec.hard_floor_pairs > 0,
            "n_harmful_matched": n_h,
            "n_strict_harmful_matched": len(strict_matched),
            "n_benign_matched": n_b if benign_active else None,
        },
        # -- the frozen matched intersection (the primary pool). With a benign
        #    half declared this is the BALANCED harmful+benign pool. ----------
        "matched_intersection": {
            "n": n_matched,
            # Requirement (a) survives as a per-pair label, so any result can
            # be reported on the STRICT subset ("undefended complied", neg is
            # a genuine harmful compliance) alone.
            "n_strict": sum(1 for pid in matched_order if pid in strict_matched),
            "n_relaxed": sum(1 for pid in matched_order
                             if pid in selected_h and pid not in none_success),
            "n_benign": sum(1 for pid in matched_order if pid in selected_b),
            "requirement_a_required": spec.require_undefended_compliance,
            "ordered_pairs": [
                {**matched_pools[spec.si_ids[0]][i].as_dict(),
                 "pos_sha": {si: matched_pools[si][i].pos_sha
                             for si in spec.si_ids},
                 "requirement_a": _req_a(matched_order[i])}
                for i in range(n_matched)],
            "per_family_counts": _counts(matched_pools[spec.si_ids[0]])[0],
            "per_goal_counts": _counts(matched_pools[spec.si_ids[0]])[1],
        },
        # -- the native per-SI pools (labelled secondary, unequal by design;
        #    harmful side only — the benign half exists only in the matched
        #    regime) ----------------------------------------------------------
        "native_per_si": {
            si: {
                "n": len(native_pools[si]),
                "ordered_pairs": [r.as_dict() for r in native_pools[si]],
                "per_family_counts": _counts(native_pools[si])[0],
                "per_goal_counts": _counts(native_pools[si])[1],
            } for si in spec.si_ids},
        # -- per-source yields + floors ---------------------------------------
        "per_source_yields": per_source,
        "all_sources_meet_floor": all_floors_ok,
        # -- rejection-sampling accounting: realised K and per-prompt yield,
        #    per SI, on the harmful side ---------------------------------------
        "rejection_sampling": rs_stats,
        # -- train/scoring disjointness, recorded -----------------------------
        "scoring_disjointness": {
            "n_scoring_prompt_shas": len(spec.scoring_prompt_shas),
            "n_removed_harmful": n_scoring_removed_h,
            "n_removed_benign": n_scoring_removed_b,
        },
        # -- holdout record: goal-level split (family lists recorded as
        #    plain facts; shared families are the design, not a leak) ---------
        "disjointness": {
            "family": {
                "n_train": len(set(spec.train_families)),
                "n_eval": len(set(spec.eval_families)),
                "train": sorted(set(spec.train_families)),
                "eval": sorted(set(spec.eval_families)),
                "disjoint": not fam_overlap,
                "overlap": fam_overlap,
                "note": _GOAL_HOLDOUT_NOTE,
            },
            "goal": {
                "n_train": len(set(spec.train_goal_ids)),
                "n_eval": len(set(spec.eval_goal_ids)),
                "disjoint": True,
            },
        },
        # -- what piece B still has to generate -------------------------------
        "coverage": coverage,
        # -- adaptive material, quarantined -----------------------------------
        "replay_transfer": replay_block,
        # -- the five identity invariants held (they raise above if not) ------
        "assertions": {
            "matched_pools_identical_except_pos": True,
            "invariants": ["ordered_prompt_ids", "per_family_counts",
                           "per_goal_counts", "neg_responses",
                           "differ_only_in_pos"],
        },
    }
    if benign_block is not None:
        manifest["benign"] = benign_block
    if composition_block is not None:
        manifest["composition"] = composition_block
    if dimension_floor_block is not None:
        manifest["dimension_floor"] = dimension_floor_block
    if relaxed_block is not None:
        manifest["relaxed_requirement_a"] = relaxed_block
    if anti_block is not None:
        manifest["bidpo_anti_neg"] = anti_block
    if bidpo_benign_block is not None:
        manifest["bidpo_benign"] = bidpo_benign_block

    # containment self-check: the goal ids / variant keys are the only free-ish
    # strings, so allowlist exactly those safe ids plus the source labels.
    allowed_labels = (
        set(spec.si_ids) | set(spec.train_families) | set(spec.eval_families)
        | set(spec.train_goal_ids) | set(spec.eval_goal_ids)
        | {r.family for r in rows} | {r.goal_id for r in rows}
        | {r.variant_key for r in rows} | {r.source_label for r in rows}
        | set(sources) | set(replay_families) | set(spec.fit_dims)
        | (spec.benign_families if benign_active else frozenset())
        | ({spec.anti_si_id} if spec.anti_si_id is not None else frozenset()))
    assert_no_store_text(manifest, allowed_labels=allowed_labels)

    manifest["manifest_sha256"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: Mapping) -> str:
    """A deterministic sha256 over everything load-bearing in the manifest
    (the whole dict except a pre-existing ``manifest_sha256``). Canonical
    serialisation (sorted keys, no whitespace) makes it byte-reproducible."""
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dumps_manifest(manifest: Mapping) -> str:
    """The canonical on-disk JSON: sorted keys, stable indent — byte-identical
    across runs on identical inputs."""
    return json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=True)


def write_manifest(manifest: Mapping, path: str | Path) -> Path:
    """Freeze the manifest to disk (atomic tmp+replace), byte-stable. Verifies
    the embedded ``manifest_sha256`` matches the content before writing, so a
    hand-edited or stale manifest fails loudly rather than being trusted."""
    path = Path(path)
    embedded = manifest.get("manifest_sha256")
    recomputed = manifest_hash(manifest)
    if embedded is not None and embedded != recomputed:
        raise MatchedPoolError(
            f"manifest_sha256 mismatch (embedded {embedded} != content "
            f"{recomputed}) — refusing to freeze a manifest whose hash does not "
            "cover its content")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(dumps_manifest(manifest))
    tmp.replace(path)
    return path
