"""Shared scaffolding for the HELD-OUT jailbreak evaluation families.

Held-out families are EVAL-ONLY attack families that must NEVER enter any
training pool (that is what "held out" means — a family the trained defense has
never seen). This module is the seam that lets a NEW held-out family be slotted
in fast and evaluated identically to every other family, and it enforces the
invariants that make a held-out result trustworthy.

THE CENTRAL DESIGN PRINCIPLE — separate BUILD from REPLAY.
    Every family, however expensive its provenance, reduces to a FROZEN,
    CONTENT-HASHED prompt corpus that REPLAY treats identically to today's
    static families (``static_common.run_static``). Build once (possibly
    expensive — GPU, an LLM rewrite, a sub-env, an ingested artifact), replay
    forever at ONE generation per prompt. That is what makes "slot in and
    evaluate quickly" true, and it makes every family comparable across
    conditions because every condition replays byte-identical prompts.

THREE BUILD KINDS (the ``build_kind`` field of :class:`HeldOutFamily` — how the
family's corpus is PRODUCED; deliberately NOT called "class A/B/C", which in
this programme is the defense-OBJECT taxonomy, a different axis — see below):
    deterministic  a pure function of the goal, no model at build time.
                   Implemented via the ``<name>_variants(goal, seed)`` seam
                   exactly like ``pasttense`` / ``encoding``; the "corpus" is
                   materialised on the fly for free (:func:`materialize_corpus`).
    rewriter       produced by an LLM at BUILD time (``llm_rewriter``), then
                   FROZEN. Replay reads the frozen corpus; it never touches the
                   LLM. Provenance names the model + pin.
    ingested       an EXTERNAL pinned artifact (``frozen_corpus``): expensive /
                   adaptive to generate elsewhere, but static once produced.
                   Provenance names the artifact path / HF-pin + sha.

The ``rewriter`` and ``ingested`` kinds SHARE the frozen-corpus replay engine
(``frozen_corpus`` / ``FrozenCorpus``) — they differ only in how the corpus was
produced, not in how it is evaluated. A single adapter unlocks the whole of the
``ingested`` kind at once.

THREE EVALUATION AXES a held-out result reads along — keep them distinct:
    1. TRANSFORM axis (always).  The family is a NEW attack transform the defense
       never trained on. Held-out-ness at this axis is enforced by the leakage
       gate (below) — the transformed PROMPTS are not themselves training prompts.
    2. GOAL axis (harm-category holdout).  The stage-3 pool holds out 33% of harm
       CATEGORIES wholesale (``category_holdout: {seed: 0, frac: 0.33}``), an
       AUTHORITATIVE, enforced split (``build_two_si_pool._goal_split_from_store``
       reads each store row's ``category_group`` into the PoolSpec, and the pool
       build filters on the train goals). So a held-out result splits two ways —
       and this module labels them for a reader who does not know the raw group
       names (:func:`goal_axis_label`):
         held_in  goals -> TRANSFORM GENERALISATION: a new transform applied to
                           goals whose harm categories the defense DID train on.
         held_out goals -> FULL GENERALISATION: a new transform AND a harm
                           category the defense NEVER trained on (strictly the
                           stronger claim).
       A per-family, per-cell number pooled over the two GOAL groups throws away
       the axis that makes the result interesting — so the manifest records the
       behaviour count PER axis, and the ASR breakout routes through the existing
       AUTHORITATIVE ``by_category_group`` emission (``jailbreak_run._category_groups``),
       never a re-derivation here (duplicating the split is how the two drift).
    These coexist and are ORTHOGONAL to the defense-OBJECT class (below).

    NOTE: a third "transfer"/cross-model axis (flagging corpora optimised
    against another model) was defined here and REMOVED ENTIRELY by user
    directive 2026-08-21 as not relevant to this programme. An ingested corpus
    is reported as an ordinary held-out family with no such annotation. This is
    a decision, not an omission — do not reintroduce it.

DON'T POOL ACROSS DEFENSE-OBJECT CLASSES. The condition's ``cell`` fixes the
defense OBJECT, and objects intervene at DIFFERENT phases of the forward by
construction (alphasteer / the matched CAA floor never touch decode positions;
antidote edits one prefill position plus decode; decode-only fits are
response-positions-only). A held-out number pooled across object classes
silently averages arms that intervene at different phases — a confound this
scaffolding exists to prevent. So the manifest records the OBJECT class + cell +
arm (read from the registry, ``defense_cell``, never inferred from a raw arm
string), and a pooled headline is never emitted without the per-cell breakout.

WHAT THIS MODULE GUARANTEES (the hard requirements, enforced not asserted):
    1. LEAKAGE GATE — :func:`check_corpus_leakage` checks every corpus prompt
       against the stage-3 TRAINING-POOL fingerprint by normalised-exact match
       (``jailbreak_data.normalized_text_sha`` — the SAME primitive as the
       OR-Bench 80k-minus-hard-1k subtraction in ``content_sets``) and REFUSES
       on overlap. A held-out family that leaks is worse than no family.
    2. ONE SHARED GRADER — every family persists through ``run_static`` into the
       same ``responses.jsonl`` shape, so the authoritative grading host StrongREJECT
       post-step grades them all identically. No family brings its own criterion.
    3. MANIFEST — :func:`build_manifest` records the corpus content hash, build
       provenance, row + behaviour counts, the leakage result, the TRANSFER
       flag, the GOAL-axis behaviour counts, and the defense-OBJECT class/cell.
       Sanitised: numbers, bools and enum/id strings only — never harmful text.
    4. CONFORMANCE — :func:`conformance_report` gives the checks any registered
       family must pass; ``tests/test_heldout_conformance.py`` runs it over every
       registered family so adding one is one module + one pin + one fixture.

GOAL-HOLDOUT COVERAGE GAP (a limitation, not a bug). The harm-category holdout
only splits CATEGORISED goals. Store rows with NO harm category — the
``wildjailbreak_harmful`` family, ~4,000 sampled rows — are stamped ``held_in``
(train) by design, so the held-out GOAL axis is drawn ONLY from the categorised
HarmBench / JBB goals; wildjailbreak contributes training goals and never appears
on the held-out goal axis. The manifest stamps this
(``goal_holdout_coverage="categorised_harmbench_jbb_only"``).

CONTAINMENT. Materialised prompts and frozen corpora contain harmful text and
therefore live ONLY in mode-0700 private dirs (``static_common``'s discipline,
and ``frozen_corpus`` for the on-disk corpora). The MANIFEST is the sanitised,
committable/servable artifact; it never carries a prompt or a goal string.

OUT OF SCOPE — families deliberately NOT implemented here (do not re-add):
  Already implemented on the static/replay scaffolding:
    GCG (``gcg``/``gcg_replay``), AmpleGCG (``amplegcg_plus``), BoN (``bon``),
    PastTense (``pasttense``), PAP (``pap``), and the
    encoding subset — Base64 / FlipAttack / payload-splitting (``encoding``).
    One name resolves to exactly one scaffolding (the dispatch guard in
    ``attacks/__init__``), so these run through their existing adapters.
    NOTE (2026-08-28): this is a DISPATCH boundary, not a holdout — the
    holdout is GOAL-level, and a family may be trained on and evaluated on.
  Not static / different threat model (cannot go through the prompt-based runner):
    - Simple Adaptive Attacks (random search needs per-target logprob feedback);
    - GPTFuzzer as a LIVE mutation-search loop (its ARCHIVE is ingestible, below);
    - PGD / embedding-space (Schwinn) — operates on embeddings, not text, so it
      cannot go through a prompt-based runner AT ALL. It needs a separate
      white-box interface; noted as a KNOWN GAP, not forced.

INGESTIBLE THROUGH THE FROZEN-CORPUS ADAPTER — reachable once someone supplies a
pinned artifact (do NOT generate the artifacts here); each needs its published
adversarial-prompt table (ideally keyed by HarmBench BehaviorID), pinned by path
or HF pin + sha, plus a ``frozen_corpus.MappingSpec``:
    Rainbow Teaming, Ruby Teaming, Ferret, DiveR-CT, Curiosity-driven Red
    Teaming, GPTFuzzer archives, I-GCG, COLD-Attack, GBDA, AdvPrompter. See
    ``frozen_corpus`` for the adapter and ``frozen_demo`` for the worked example.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from ..jailbreak_artifacts import assert_sanitized
from ..jailbreak_data import normalized_text_sha


# --------------------------------------------------------------------------- #
# The family spec                                                             #
# --------------------------------------------------------------------------- #


VALID_BUILD_KINDS = ("deterministic", "rewriter", "ingested")


@dataclass(frozen=True)
class HeldOutFamily:
    """Registry record for one held-out family.

    ``variants_fn(goal, seed) -> [(variant_name, prompt), ...]`` is the REPLAY
    seam every family shares with ``static_common`` — deterministic and
    defense-independent (same seed -> identical prompts), so the exact same
    variant is presented under every defense condition. ``run_fn`` is the
    ``run_<name>`` entry the driver dispatches (it delegates to
    ``static_common.run_static``).

    ``build_kind`` is how the corpus is PRODUCED (deterministic / rewriter /
    ingested — NOT the defense-object "class A/B/C", a different axis).
    ``provenance`` is the sanitised build description folded
    into the manifest — every value must be a number/bool or an enum/id string.
    ``build_fn`` is the ``rewriter``-kind build step (None otherwise).
    """

    name: str
    build_kind: str
    variants_fn: Callable[..., Sequence[tuple[str, str]]]
    run_fn: Callable[..., dict]
    provenance: dict = field(default_factory=dict)
    #: ``rewriter``-kind build step (``build_fn(goals, *, rewriter, fingerprint,
    #: ...)``) — None for ``deterministic`` (no build) and ``ingested`` (not
    #: built here). The conformance test uses it to build a stub corpus so every
    #: kind is exercised end to end.
    build_fn: Optional[Callable[..., dict]] = None

    def __post_init__(self):
        if self.build_kind not in VALID_BUILD_KINDS:
            raise ValueError(
                f"held-out family {self.name!r} has build_kind "
                f"{self.build_kind!r} (expected one of {VALID_BUILD_KINDS})")
        # Provenance must be manifest-safe up front — catch a free-text value at
        # registration, not when the manifest is written mid-run.
        assert_sanitized(self.provenance, allowed=_provenance_allowed(self))


def _provenance_allowed(fam: "HeldOutFamily") -> set[str]:
    """The provenance allowlist: the family/build-kind enums plus any string
    VALUES the family declared in provenance (model ids, artifact names, pins —
    all fixed vocabulary chosen by the family author, not model output)."""
    allowed = {fam.name, fam.build_kind, *VALID_BUILD_KINDS}
    for v in fam.provenance.values():
        if isinstance(v, str):
            allowed.add(v)
    return allowed


# --------------------------------------------------------------------------- #
# Goal-axis labels (reader-friendly names for the harm-category holdout)       #
# --------------------------------------------------------------------------- #

#: Map the AUTHORITATIVE ``category_group`` (``jailbreak_run._category_groups`` /
#: the store's own label) to a reader-facing goal-axis name. A reader must not
#: have to know what "held_in"/"held_out" mean.
_GOAL_AXIS_LABELS = {
    "held_in": "transform_generalisation",   # new transform, trained harm category
    "held_out": "full_generalisation",       # new transform AND untrained category
}


def goal_axis_label(category_group: Optional[str]) -> str:
    """Reader-friendly goal-axis name for a ``category_group``. An uncategorised
    goal (no harm category — e.g. a wildjailbreak row, stamped train by design)
    maps to ``uncategorised``."""
    return _GOAL_AXIS_LABELS.get(category_group or "", "uncategorised")


# --------------------------------------------------------------------------- #
# Materialisation + content hash                                               #
# --------------------------------------------------------------------------- #


def materialize_corpus(variants_fn, goals, *, seed: int = 0) -> list[dict]:
    """The frozen corpus a family reduces to, over ``goals`` at ``seed``.

    Returns an ORDERED list of rows ``{goal_id, sample_idx, variant_name,
    prompt}`` — ``variants_fn`` applied to each goal, sub-variants numbered by
    position (the ``sample_idx`` ``static_common`` persists). For a
    ``deterministic`` family this is the whole "build" (free); for
    ``rewriter``/``ingested`` ``variants_fn`` reads a frozen corpus so this just
    re-expresses it in the canonical shape used for hashing, leakage and
    conformance."""
    rows: list[dict] = []
    for g in goals:
        for i, (name, prompt) in enumerate(variants_fn(g, seed)):
            rows.append({
                "goal_id": g.goal_id,
                "sample_idx": i,
                "variant_name": name,
                "prompt": prompt,
            })
    return rows


def corpus_content_hash(rows: Iterable[dict]) -> str:
    """A STABLE sha256 over a corpus's ``(goal_id, sample_idx, variant_name,
    prompt)`` rows, independent of row order or dict-key order. Two builds that
    produce the same prompts hash identically — the pin the manifest records and
    replay verifies."""
    canon = sorted(
        (str(r["goal_id"]), int(r["sample_idx"]), str(r["variant_name"]),
         str(r["prompt"]))
        for r in rows
    )
    h = hashlib.sha256()
    for gid, idx, name, prompt in canon:
        for part in (gid, str(idx), name, prompt):
            b = part.encode("utf-8")
            h.update(str(len(b)).encode("ascii"))
            h.update(b"\0")
            h.update(b)
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The leakage gate (hard requirement 1)                                        #
# --------------------------------------------------------------------------- #


class LeakageError(RuntimeError):
    """A held-out corpus overlaps the training pool by normalised-exact match.
    Raised (never warned) so a leaked family cannot silently be evaluated — a
    held-out family that leaks is worse than no family."""


@dataclass(frozen=True)
class LeakageReport:
    """The leakage-check outcome. Leak-free by construction: COUNTS and the
    offending GOAL IDS only, never a prompt string."""

    n_checked: int
    n_overlap: int
    overlap_goal_ids: tuple[str, ...]
    training_fingerprint_sha: str

    @property
    def passed(self) -> bool:
        return self.n_overlap == 0

    def as_manifest(self) -> dict:
        return {
            "n_checked": self.n_checked,
            "n_overlap": self.n_overlap,
            "passed": self.passed,
            "training_fingerprint_sha": self.training_fingerprint_sha,
            "overlap_goal_ids": list(self.overlap_goal_ids[:32]),
        }


def fingerprint_sha(fingerprint: Iterable[str]) -> str:
    """A stable sha256 over a training fingerprint (a set of normalised-text
    shas), so the manifest can record WHICH training pool a corpus was checked
    against without embedding the whole set."""
    h = hashlib.sha256()
    for s in sorted(set(fingerprint)):
        h.update(s.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def training_fingerprint(texts: Iterable[str]) -> frozenset[str]:
    """The normalised-exact-match fingerprint of a set of training prompt texts
    — ``{normalized_text_sha(t) for t in texts}``. The SAME primitive the
    OR-Bench hard-1k subtraction uses, so a leak here is a leak there."""
    return frozenset(normalized_text_sha(t) for t in texts)


def check_corpus_leakage(rows, fingerprint, *, refuse: bool = True) -> LeakageReport:
    """Check a materialised corpus's PROMPTS against the training-pool
    ``fingerprint`` by normalised-exact match. Returns a :class:`LeakageReport`;
    if ``refuse`` (the default) and any prompt collides, raises
    :class:`LeakageError` — the corpus is NOT usable as held-out."""
    fp = frozenset(fingerprint)
    overlap: list[str] = []
    n = 0
    for r in rows:
        n += 1
        if normalized_text_sha(str(r["prompt"])) in fp:
            overlap.append(str(r.get("goal_id", "")))
    report = LeakageReport(
        n_checked=n,
        n_overlap=len(overlap),
        overlap_goal_ids=tuple(dict.fromkeys(overlap)),
        training_fingerprint_sha=fingerprint_sha(fp),
    )
    if refuse and not report.passed:
        raise LeakageError(
            f"held-out corpus leaks: {report.n_overlap} of {report.n_checked} "
            f"prompts normalised-exact-match the training pool "
            f"(offending goal ids: {report.overlap_goal_ids[:8]}). Refusing — a "
            "held-out family that leaks is worse than no family.")
    return report


# --------------------------------------------------------------------------- #
# The manifest (hard requirement 3)                                            #
# --------------------------------------------------------------------------- #


def _goal_axis_counts(rows, category_group_of) -> Optional[dict]:
    """Behaviour count per GOAL axis (transform-generalisation vs
    full-generalisation vs uncategorised) using the AUTHORITATIVE category-group
    split. ``category_group_of`` maps ``goal_id -> "held_in"|"held_out"`` (from
    ``jailbreak_run._category_groups`` / the store); None when the split is not
    available on this node."""
    if category_group_of is None:
        return None
    behaviors = {str(r["goal_id"]) for r in rows}
    counts = {"transform_generalisation": 0, "full_generalisation": 0,
              "uncategorised": 0}
    for gid in behaviors:
        counts[goal_axis_label(category_group_of.get(gid))] += 1
    return {k: {"n_behaviors": v} for k, v in counts.items()}


def build_manifest(
    family: "HeldOutFamily",
    rows,
    *,
    seed: int,
    leakage: LeakageReport,
    category_group_of: Optional[dict] = None,
    object_class: Optional[str] = None,
    cell: Optional[str] = None,
    arm: Optional[str] = None,
    goal_set_sha: Optional[str] = None,
    built_at: Optional[str] = None,
    attack_regime: str = "fixed_prompt",
) -> dict:
    """The per-family, per-cell corpus manifest: content hash, build provenance,
    row + behaviour counts, the leakage result, the TRANSFER flag, the GOAL-axis
    behaviour counts (when the authoritative split is supplied), the defense
    OBJECT class / cell / arm (when supplied), and the ATTACK REGIME. Sanitised
    (numbers/bools/enum-ids only) — validate with :func:`assert_manifest_safe`
    before serving.

    This stamps the CORPUS (build time) AND records the extra axes a reader must
    not pool over / must not misread: the GOAL-holdout axis (per-axis n), the
    defense-OBJECT axis (which object class this cell is), and ``attack_regime``
    (``fixed_prompt`` — the held-out transform families are all fixed prompts, an
    attacker that does not react to the defense; the qualifier travels ON the
    manifest, not in prose). The sweep's engine stamp records the SERVING identity
    (replay time), separately."""
    rows = list(rows)
    n_behaviors = len({str(r["goal_id"]) for r in rows})
    manifest = {
        "family": family.name,
        "build_kind": family.build_kind,
        "corpus_content_hash": corpus_content_hash(rows),
        "row_count": len(rows),
        "n_behaviors": n_behaviors,
        "seed": int(seed),
        "built_at": built_at or _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "provenance": dict(family.provenance),
        "leakage_check": leakage.as_manifest(),
        # the goal-holdout coverage gap (a limitation, stamped so a reader sees
        # the held-out goal axis excludes uncategorised/wildjailbreak goals)
        "goal_holdout_coverage": "categorised_harmbench_jbb_only",
        # ATTACK REGIME on the manifest (fixed_prompt for held-out transforms) —
        # so a quoted family's numbers carry their own "not an adaptive adversary"
        # caveat.
        "attack_regime": str(attack_regime),
    }
    axis = _goal_axis_counts(rows, category_group_of)
    if axis is not None:
        manifest["goal_axis"] = axis
    if object_class is not None:
        manifest["object_class"] = object_class
    if cell is not None:
        manifest["cell"] = cell
    if arm is not None:
        manifest["arm"] = arm
    if goal_set_sha is not None:
        manifest["goal_set_sha"] = goal_set_sha
    return manifest


#: The defense-OBJECT class enum a manifest may stamp (reader-friendly names for
#: ``defense_cell`` class_ A/B/baseline; forward-compatible with finer training-
#: lane classes). Never the attack build_kind.
OBJECT_CLASS_VALUES = ("si_distilled", "si_agnostic_baseline", "baseline", "other")

#: The fixed reason label the skipped-leakage branch stamps (fingerprint not
#: loadable on this node). A CONSTANT, allowlisted below by NAME: the first GPU
#: smoke of the held-out leg (2026-08-21, slurm 173196) found the skipped branch
#: had NEVER run end to end — the reason string was not in the sanitisation
#: allowlist, so every fingerprint-absent node failed assert_manifest_safe
#: instead of recording the skip loudly. Keep it a constant so the writer and the
#: allowlist can never drift apart again.
LEAKAGE_SKIPPED_REASON = "no_training_fingerprint_available_on_this_node"


def _manifest_allowed(manifest: dict) -> set[str]:
    """The manifest string-enum allowlist: build-kind/family/coverage labels, the
    object-class + cell + arm enums, the provenance string values, content
    hashes, the ISO timestamp, and any overlap goal ids (all safe to serve)."""
    allowed: set[str] = {
        manifest.get("family", ""), manifest.get("build_kind", ""),
        *VALID_BUILD_KINDS, *OBJECT_CLASS_VALUES, "true", "false",
        "categorised_harmbench_jbb_only",
        # attack-regime enum values are safe labels, not prompt text.
        "fixed_prompt", "adaptive",
        # the skipped-leakage reason label (a constant, no content rides in it).
        LEAKAGE_SKIPPED_REASON,
        manifest.get("corpus_content_hash", ""),
        str(manifest.get("built_at", "")),
    }
    for key in ("object_class", "cell", "arm"):
        v = manifest.get(key)
        if isinstance(v, str):
            allowed.add(v)
    prov = manifest.get("provenance", {})
    for v in prov.values():
        if isinstance(v, str):
            allowed.add(v)
    lk = manifest.get("leakage_check", {})
    allowed.add(str(lk.get("training_fingerprint_sha", "")))
    allowed.update(str(g) for g in lk.get("overlap_goal_ids", []))
    if "goal_set_sha" in manifest:
        allowed.add(str(manifest["goal_set_sha"]))
    allowed.discard("")
    return allowed


def assert_manifest_safe(manifest: dict) -> None:
    """Raise unless the manifest is servable — every leaf a number/bool/None or
    an allowed enum/id string, so no harmful prompt or goal text can ride along
    (JB4). The goal-axis sub-dicts are safe-identifier keys + int counts."""
    assert_sanitized(manifest, allowed=_manifest_allowed(manifest))


# --------------------------------------------------------------------------- #
# The driver preflight                                                          #
# --------------------------------------------------------------------------- #


def write_family_manifest(
    family: "HeldOutFamily",
    goals,
    *,
    manifest_path,
    fingerprint=None,
    category_group_of: Optional[dict] = None,
    object_class: Optional[str] = None,
    cell: Optional[str] = None,
    arm: Optional[str] = None,
    seed: int = 0,
    attack_regime: str = "fixed_prompt",
) -> dict:
    """The driver-facing preflight for one held-out family: materialise the
    corpus over ``goals``, run the LEAKAGE GATE (refuse on overlap) when a
    training ``fingerprint`` is available, build the sanitised manifest (with the
    GOAL-axis split, defense-OBJECT class, and ATTACK REGIME when supplied), and
    write it.

    ``category_group_of`` is the AUTHORITATIVE goal split
    (``jailbreak_run._category_groups``); ``object_class`` / ``cell`` / ``arm``
    identify the defense object (``defense_cell``); ``attack_regime`` is the
    fixed_prompt/adaptive qualifier stamped on the manifest (held-out transforms
    are ``fixed_prompt``). When ``fingerprint`` is None the gate cannot run here —
    the manifest records ``leakage_check.skipped=True`` LOUDLY rather than a
    fabricated pass."""
    import json as _json
    from pathlib import Path as _Path

    rows = materialize_corpus(family.variants_fn, goals, seed=seed)
    kw = dict(category_group_of=category_group_of, object_class=object_class,
              cell=cell, arm=arm, attack_regime=attack_regime)
    if fingerprint is not None:
        leak = check_corpus_leakage(rows, fingerprint, refuse=True)
        manifest = build_manifest(family, rows, seed=seed, leakage=leak, **kw)
    else:
        manifest = build_manifest(
            family, rows, seed=seed,
            leakage=LeakageReport(n_checked=len(rows), n_overlap=0,
                                  overlap_goal_ids=(), training_fingerprint_sha=""),
            **kw)
        manifest["leakage_check"] = {
            "skipped": True,
            "reason": LEAKAGE_SKIPPED_REASON,
            "n_checked": len(rows), "n_overlap": 0, "passed": False,
        }
    assert_manifest_safe(manifest)
    p = _Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


# --------------------------------------------------------------------------- #
# Conformance (hard requirement 4)                                            #
# --------------------------------------------------------------------------- #


def conformance_report(
    family: "HeldOutFamily",
    goals,
    fingerprint,
    *,
    seed: int = 0,
    min_variants_per_goal: int = 1,
) -> dict:
    """Run the checks EVERY registered held-out family must pass, returning a
    ``{check_name: bool}`` report (plus counts). ``tests/test_heldout_conformance``
    asserts every value is True over every registered family. Checks:

    - ``determinism``          same seed -> byte-identical materialised corpus
    - ``variant_count``        every covered goal yields >= ``min_variants_per_goal``
    - ``no_leakage``           no prompt normalised-exact-matches the pool
    - ``manifest_complete``    the manifest has every required field and is
                               servable (:func:`assert_manifest_safe`)
    - ``content_hash_stable``  the hash is invariant to row/goal ordering

    Replay-resume idempotence is exercised separately by the conformance TEST."""
    rows = materialize_corpus(family.variants_fn, goals, seed=seed)
    rows2 = materialize_corpus(family.variants_fn, goals, seed=seed)

    determinism = rows == rows2

    covered = {str(r["goal_id"]) for r in rows}
    per_goal_counts = {gid: 0 for gid in covered}
    for r in rows:
        per_goal_counts[str(r["goal_id"])] += 1
    variant_count = all(c >= min_variants_per_goal for c in per_goal_counts.values())

    leak = check_corpus_leakage(rows, fingerprint, refuse=False)

    import random as _random
    shuffled = list(rows)
    _random.Random(0).shuffle(shuffled)
    content_hash_stable = corpus_content_hash(rows) == corpus_content_hash(shuffled)

    manifest = build_manifest(family, rows, seed=seed, leakage=leak)
    required = {"family", "build_kind", "corpus_content_hash",
                "row_count", "n_behaviors", "seed", "built_at", "provenance",
                "leakage_check", "goal_holdout_coverage"}
    manifest_complete = required.issubset(manifest)
    if manifest_complete:
        try:
            assert_manifest_safe(manifest)
        except ValueError:
            manifest_complete = False

    return {
        "determinism": determinism,
        "variant_count": variant_count,
        "no_leakage": leak.passed,
        "manifest_complete": manifest_complete,
        "content_hash_stable": content_hash_stable,
        "n_rows": len(rows),
        "n_behaviors_covered": len(covered),
    }
