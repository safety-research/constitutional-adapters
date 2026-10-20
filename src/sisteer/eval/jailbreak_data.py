"""Offline loaders for the jailbreak-robustness benchmarks (S2-JB).

Three standard benchmarks, each loaded the SAME way every other stage-2
dataset is: via
``hf_hub_download(repo, file, repo_type="dataset", revision=<full sha>)``
+ a direct pyarrow/pandas read, NEVER ``load_dataset`` — ``datasets>=5``
refuses script datasets, ``hf://…@rev`` hits the Hub API even when cached,
and ``load_dataset`` cache-config hashes differ online-vs-offline. Every
loader therefore works unchanged under ``HF_HUB_OFFLINE=1`` on a compute
node once Phase 0 has warmed the cache on a login node. The pattern is the
one ``scripts/fetch_stage2_data.ensure_hf_snapshot`` verifies.

Revisions are PINNED to a full commit sha (verified on the Hub 2026-08-14):

- HarmBench (Mazeika 2402.04249) — ``mariagrandury/harmbench`` (UNGATED
  mirror of the official text behaviours; ``walledai/HarmBench`` is gated
  and unusable offline without a token). Carries the official taxonomy
  columns (Behavior, FunctionalCategory ∈ {standard, contextual,
  copyright}, SemanticCategory ∈ the 7 semantic categories, Tags,
  ContextString, BehaviorID) and the official 320/80 test/val split under
  ``DirectRequest/``. BoN + the multi-turn tiers sweep the STANDARD test
  behaviours; ``copyright`` is hash-matched (not LLM-judged) and is
  excluded from the harmfulness comparison.
- JBB-Behaviors — ``JailbreakBench/JBB-Behaviors``: 100 harmful + 100
  benign, 10 categories × 10, columns Index/Goal/Target/Behavior/Category/
  Source.
- XSTest (Röttger et al.) — ``Paul/XSTest``: the 250 SAFE prompts are the
  benign fairness control (the file also holds 200 unsafe contrast prompts,
  which we never use here).

**Dual-use containment.** A loaded ``Goal`` carries the
harmful-behaviour ``text`` in memory so the target/judge can be run on it,
but that text is NEVER written to a committed or served file. Only
``goal_id`` + category label + aggregate numbers are ever persisted to
``out/…`` (enforced by ``eval/jailbreak_artifacts``). ``goal_content_sha``
gives a stable, opaque handle for cache keys / resume keys that leaks
nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# -- pinned dataset revisions (full shas — see module docstring) ---------------

HARMBENCH_REPO = "mariagrandury/harmbench"
HARMBENCH_REVISION = "4aa1f46a9d43e441e238c1d01dc52413094fc84e"
HARMBENCH_FILES = {
    "test": "DirectRequest/test-00000-of-00001.parquet",   # 320 behaviours
    "val": "DirectRequest/val-00000-of-00001.parquet",     # 80 behaviours
}

JBB_REPO = "JailbreakBench/JBB-Behaviors"
JBB_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"
JBB_FILES = {
    "harmful": "data/harmful-behaviors.csv",
    "benign": "data/benign-behaviors.csv",
}

XSTEST_REPO = "Paul/XSTest"
XSTEST_REVISION = "f600c994b256f12867dfa5b3eb3d545a3e62f8b5"
XSTEST_FILE = "xstest_prompts.csv"

# The seven HarmBench semantic categories (copyright is the 8th functional
# bucket, excluded from the harmfulness read). Frozen so a served
# aggregate JSON's category strings are a fixed enum (artifact allowlist).
HARMBENCH_SEMANTIC_CATEGORIES = (
    "chemical_biological",
    "cybercrime_intrusion",
    "harassment_bullying",
    "harmful",
    "illegal",
    "misinformation_disinformation",
    "copyright",
)

JBB_CATEGORIES = (
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Economic harm",
    "Fraud/Deception",
    "Disinformation",
    "Sexual/Adult content",
    "Privacy",
    "Expert advice",
    "Government decision-making",
)


@dataclass(frozen=True)
class Goal:
    """One benchmark item. ``text`` is live-only (never serialised); id +
    category are the only human-readable fields safe to persist."""

    goal_id: str
    text: str
    category: str
    dataset: str        # "harmbench" | "jbb" | "xstest"
    revision: str       # the pinned sha the text came from
    split: str = ""     # "test"|"val" (harmbench); "harmful"|"benign" (jbb); "safe" (xstest)
    source: str = ""    # provenance where the benchmark records it
    functional: str = ""  # harmbench FunctionalCategory (standard/contextual/copyright)


def goal_content_sha(goal: Goal) -> str:
    """Stable opaque handle over (dataset, revision, text) — the resume /
    cache namespace component that leaks no harmful text (JB1)."""
    h = hashlib.sha256()
    h.update(goal.dataset.encode())
    h.update(b"\0")
    h.update(goal.revision.encode())
    h.update(b"\0")
    h.update(goal.text.encode())
    return h.hexdigest()


def normalized_text(text: str) -> str:
    """Case- and whitespace-normalised prompt text: lowercase, collapse all
    runs of whitespace to a single space, strip. The canonical form for the
    content-disjointness check so the SAME behaviour phrased identically across
    two datasets/revisions collides (a dataset+revision-keyed sha would miss
    it — B/HIGH content-disjoint splits)."""
    return " ".join(text.lower().split())


def normalized_text_sha(text: str) -> str:
    """A TEXT-ONLY sha over ``normalized_text(text)`` — no dataset/revision
    component, so it is disjoint-across-datasets by construction. Used to
    guarantee the teacher prompts share no PROMPT with any HarmBench / JBB /
    XSTest eval prompt, regardless of which dataset each came from."""
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


# Cross-taxonomy synonyms: the SAME harm topic under HarmBench's and JBB's
# different label vocabularies. `category_holdout_split` shuffles each
# taxonomy INDEPENDENTLY, so without linking these a topic can be held out
# under one source's name while its synonym stays held in under the other's
# -- and a "generalises to unseen harm types" read is then measured on a
# topic the model trained on. At the shipped seed 0 / frac 0.33 that is
# live: JBB `Disinformation` (10 goals) is HELD OUT while HarmBench
# `misinformation_disinformation` (34 goals) is HELD IN.
#
# Only pairs that name the same topic are listed. HarmBench `harmful` is a
# catch-all and is deliberately NOT linked to anything; the partial overlaps
# (`illegal` against JBB `Fraud/Deception` + `Economic harm`,
# `chemical_biological` against `Physical harm`) are real but not synonymy,
# and pretending otherwise would hold out most of the taxonomy. They remain
# an open leak, documented rather than silently half-fixed.
CROSS_TAXONOMY_SYNONYMS: tuple[frozenset[str], ...] = (
    frozenset({"misinformation_disinformation", "Disinformation"}),
    frozenset({"cybercrime_intrusion", "Malware/Hacking"}),
    frozenset({"harassment_bullying", "Harassment/Discrimination"}),
)


def category_holdout_split(taxonomies: dict[str, list[str]], *, seed,
                           frac: float,
                           link_synonyms: bool = False,
                           ) -> tuple[list[str], list[str]]:
    """Partition harm categories into held-IN vs held-OUT, seed-shuffled and
    ~``frac`` held out PER taxonomy (so both HarmBench and JBB contribute
    held-in AND held-out categories). Reproducible from ``(seed, frac)``, never
    hand-picked — so a "does the compiled safety vector generalise to harm TYPES
    it was never distilled to refuse?" read cannot be cherry-picked.

    ``taxonomies`` maps a taxonomy name -> its category list. Returns
    ``(held_in, held_out)`` as sorted category-NAME lists over the union.
    Category names are safe to serve (never harmful prompt text)."""
    import random

    held_in: set[str] = set()
    held_out: set[str] = set()
    for name in sorted(taxonomies):
        cats = sorted(set(taxonomies[name]))
        if not cats:
            continue
        order = list(cats)
        random.Random(f"{seed}|{name}").shuffle(order)
        n_out = max(1, round(frac * len(order)))
        held_out.update(order[:n_out])
        held_in.update(order[n_out:])
    if link_synonyms:
        # A synonym group is held OUT only if EVERY member was independently
        # assigned out; a split group collapses to held-IN. Either direction
        # closes the leak -- a held-out topic then has no training members by
        # construction -- but they charge different sets. Collapsing IN costs
        # the held-out set one topic; collapsing OUT would move HarmBench's
        # 34 misinformation goals into held-out and cut the training budget
        # ~18%, which is the binding constraint (measured: ~118 usable
        # training goals after both splits compound). IN is the cheaper
        # correction and the default here; flip this block to collapse OUT if
        # held-out COVERAGE ever matters more than training volume.
        for group in CROSS_TAXONOMY_SYNONYMS:
            present = {c for c in group if c in held_in or c in held_out}
            if len(present) < 2 or present <= held_out:
                continue
            held_out -= present
            held_in |= present
    return sorted(held_in), sorted(held_out)


def harm_category_holdout(*, seed, frac: float,
                          link_synonyms: bool = False,
                          ) -> tuple[list[str], list[str]]:
    """The eval's category holdout over the STANDARD harm taxonomies: HarmBench
    SemanticCategory (copyright excluded — it is hash-matched, not judged) and
    JBB category. Returns ``(held_in, held_out)`` category-name lists."""
    return category_holdout_split(
        {
            "harmbench": [c for c in HARMBENCH_SEMANTIC_CATEGORIES
                          if c != "copyright"],
            "jbb": list(JBB_CATEGORIES),
        },
        seed=seed, frac=frac, link_synonyms=link_synonyms,
    )


def _download(repo: str, file: str, revision: str) -> str:
    """hf_hub_download at a pinned revision — cache-first, HF_HUB_OFFLINE
    honoured (never load_dataset)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, file, repo_type="dataset", revision=revision)


# -- HarmBench -----------------------------------------------------------------


def load_harmbench(
    split: str = "test",
    *,
    revision: str = HARMBENCH_REVISION,
    path: str | None = None,
    functional: str | None = "standard",
    exclude_copyright: bool = True,
) -> list[Goal]:
    """HarmBench text behaviours for one official split.

    ``functional`` filters FunctionalCategory (default ``"standard"`` — the
    200/159/41-behaviour standard set BoN and the multi-turn tiers sweep;
    pass ``None`` for all functional categories). ``exclude_copyright``
    drops the hash-matched copyright behaviours from the harmfulness read
    even when ``functional is None``.

    ``path`` overrides the download (a local parquet) for offline tests.
    Contextual behaviours carry a ``ContextString`` that must precede the
    behaviour text; it is prepended when present (empty for standard).
    """
    import pandas as pd

    if split not in HARMBENCH_FILES:
        raise ValueError(f"unknown HarmBench split {split!r}; "
                         f"known: {sorted(HARMBENCH_FILES)}")
    if path is None:
        path = _download(HARMBENCH_REPO, HARMBENCH_FILES[split], revision)
    df = pd.read_parquet(path)
    goals: list[Goal] = []
    for _, row in df.iterrows():
        func = str(row["FunctionalCategory"])
        if exclude_copyright and func == "copyright":
            continue
        if functional is not None and func != functional:
            continue
        behavior = str(row["Behavior"])
        context = row.get("ContextString")
        context = "" if context is None or (isinstance(context, float)) else str(context)
        text = f"{context}\n\n{behavior}" if context.strip() else behavior
        goals.append(Goal(
            goal_id=str(row["BehaviorID"]),
            text=text,
            category=str(row["SemanticCategory"]),
            dataset="harmbench",
            revision=revision,
            split=split,
            source="HarmBench",
            functional=func,
        ))
    return goals


# -- JBB-Behaviors -------------------------------------------------------------


def load_jbb(
    kind: str = "harmful",
    *,
    revision: str = JBB_REVISION,
    path: str | None = None,
) -> list[Goal]:
    """JBB-Behaviors, harmful (100) or benign (100). ``path`` overrides the
    download for offline tests."""
    import pandas as pd

    if kind not in JBB_FILES:
        raise ValueError(f"unknown JBB kind {kind!r}; known: {sorted(JBB_FILES)}")
    if path is None:
        path = _download(JBB_REPO, JBB_FILES[kind], revision)
    df = pd.read_csv(path)
    goals: list[Goal] = []
    for _, row in df.iterrows():
        goals.append(Goal(
            goal_id=f"jbb_{kind}_{int(row['Index'])}",
            text=str(row["Goal"]),
            category=str(row["Category"]),
            dataset="jbb",
            revision=revision,
            split=kind,
            source=str(row.get("Source", "")),
        ))
    return goals


def stratify_jbb(goals: list[Goal], n: int = 50, seed: int = 0) -> list[Goal]:
    """A category-balanced n-item subset (the 10-category
    stratified subset of 50, scaled to 100 as needed). Round-robin across
    categories after a per-category deterministic shuffle, so the same n
    goals are drawn every run and the split is reproducible."""
    import random

    by_cat: dict[str, list[Goal]] = {}
    for g in goals:
        by_cat.setdefault(g.category, []).append(g)
    rng = random.Random(seed)
    for rows in by_cat.values():
        rows.sort(key=lambda g: g.goal_id)
        rng.shuffle(rows)
    cats = sorted(by_cat)
    picked: list[Goal] = []
    i = 0
    while len(picked) < n and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            picked.append(by_cat[c].pop(0))
        i += 1
    return sorted(picked, key=lambda g: g.goal_id)


# -- XSTest --------------------------------------------------------------------


def load_xstest_safe(
    *,
    revision: str = XSTEST_REVISION,
    path: str | None = None,
) -> list[Goal]:
    """The 250 SAFE XSTest prompts — the benign fairness control.
    Only ``label == "safe"`` rows are kept; the 200 unsafe contrast prompts
    are never used here. ``path`` overrides the download for offline tests.
    ``category`` carries the XSTest prompt ``type`` (homonyms, figurative_
    language, safe_targets, …) so the fairness gate can be read per type."""
    import pandas as pd

    if path is None:
        path = _download(XSTEST_REPO, XSTEST_FILE, revision)
    df = pd.read_csv(path)
    goals: list[Goal] = []
    for _, row in df.iterrows():
        if str(row["label"]).strip().lower() != "safe":
            continue
        goals.append(Goal(
            goal_id=f"xstest_{row['id']}",
            text=str(row["prompt"]),
            category=str(row["type"]),
            dataset="xstest",
            revision=revision,
            split="safe",
            source="XSTest",
        ))
    return goals
