"""Content-based contrast sets for SI-AGNOSTIC (fit-once) steering extraction.

The jailbreak-defense methods AlphaSteer (2506.07022) and Antidote/RepE
(2410.02298) are fit ONCE on labeled harmful-vs-benign CONTENT — never per
system-instruction. Their SI-mode registry trainers (extraction/
alphasteer.py, extraction/antidote.py) contrast SI-present vs
SI-absent renderings of the same prompts, which is a deliberate SI steering
adaptation; this module supplies the paper-faithful content sets so the same
methods can be fit SI-agnostically and applied to any SI comparison,
including the two-SI baselines.

The content mapping, the cached analogue of
AlphaSteer's own recipe (benign = borderline pseudo-harmful + CoCoNot;
harmful = harmful_train_1000 + jailbreak_train; r = a precomputed refusal
pickle — verified against their calc_steering_matrix.py):

    benign   = Alpaca (utility) + OR-Bench-80k (over-refusal-adjacent benign)
    harmful  = HarmBench-val (direct malicious requests)
    r        = base refusals-vs-compliances = diff-in-means of last-token
               activations, harmful - benign (the Arditi/AdaSteer refusal
               direction the AlphaSteer pickle encodes). Computed in the
               trainer from these same sets unless a precomputed r is passed.

All prompts are returned as RAW instruction strings (no chat template, no
system prompt) — the trainer renders them SI-free with system=None, which is
what makes the fit SI-agnostic.

build_content_contrast also takes explicit prompt-list files
(``benign_prompts_path`` / ``harmful_prompts_path``) plus a loud
``min_prompts`` floor. They guard the failure that motivated them: the
default assembly fed 41 harmful prompts (HarmBench-val's 80 rows silently
reduced to the standard-only functional subset by load_harmbench's
default) to AlphaSteer's d x d fit and nothing surfaced the count. The
motivating lists are the AlphaSteer authors' own 14,000-benign /
2,000-harmful fit composition, materialised by
scripts/materialize_alphasteer_contrast.py.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.jailbreak_data import load_harmbench, normalized_text_sha
from .prompts import load_hf_prompts

# OR-Bench (bench-llm/or-bench, CC-BY-4.0), pinned sha; configs are root CSVs
# with `prompt`, `category` columns (facts.md H; verified cached on cluster A).
OR_BENCH_REPO = "bench-llm/or-bench"
OR_BENCH_REVISION = "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
OR_BENCH_FILES = {
    "or-bench-80k": "or-bench-80k.csv",
    "or-bench-hard-1k": "or-bench-hard-1k.csv",
    "or-bench-toxic": "or-bench-toxic.csv",
}
# Alpaca pin: the campaign's benign prompt source already flows through
# load_hf_prompts; None keeps its "latest cached" behavior (the fit-once
# object is not a resume unit, so a pin is optional — pass one for
# reproducibility across re-fits).
ALPACA_DATASET = "tatsu-lab/alpaca"

# PHTest (furonghuang-lab/PHTest, MIT), pinned sha; a single root CSV with
# `ID`, `Request`, `Harmfulness` columns — 3,269 rows, 2,077 labelled
# ``harmless`` + 1,192 ``controversial`` (verified in the shared cluster A
# cache at this revision, 2026-08-19). Stage 3's hard-benign source:
# borderline pseudo-harmful prompts, chosen so OR-Bench stays 100% reserved
# for evaluation with all 1,319 hard-1k rows intact.
PHTEST_REPO = "furonghuang-lab/PHTest"
PHTEST_REVISION = "edb80210812ecc7dab2219aa2e70921710c01888"
PHTEST_FILE = "PHTest.csv"
PHTEST_SPLITS = ("all", "harmless", "controversial")

# -- SCALE-UP benign sources. PHTest alone yields only ~2,952 rows after the
# 10-300 char filter — too few for the raised pool target at 50/50 — so the
# benign stratum draws from these additional corpora.
#
# WildJailbreak (allenai/wildjailbreak, ODC-BY per the dataset card): its train
# split's ``adversarial_benign`` rows are jailbreak-STYLED benign requests, so
# stylistically closer to the harmful side than PHTest. GATED (accept terms on
# the hub) and NOT cached on the grading host. Verified against cluster A's
# shared model cache 2026-08-20 (snapshot below) over ssh to the login host:
# schema, per-stratum counts and the ODC-BY licence confirmed there (the
# grading host does not mount the cluster's shared storage). The gated terms
# are accepted, so it is a live source; the revision is PINNED to the verified
# snapshot per the repo convention (a prior pass left it None, which must not
# ship). The train file is a TSV whose ``data_type`` column carries the four
# strata and whose ``adversarial`` column holds the jailbreak-styled prompt.
WILDJAILBREAK_REPO = "allenai/wildjailbreak"
WILDJAILBREAK_REVISION = "5ddc12a7894f842b0619b8e1c7ee496b198af009"  # cohort cache, verified 2026-08-20
WILDJAILBREAK_TRAIN_FILE = "train/train.tsv"
WILDJAILBREAK_BENIGN_TYPE = "adversarial_benign"
#: The train split's four `data_type` strata (verified on the cluster
#: 2026-08-20: vanilla_harmful 50,050 / adversarial_harmful 82,728 /
#: vanilla_benign 50,050 / adversarial_benign 78,731; 261,559 rows total).
WILDJAILBREAK_TYPES = ("vanilla_harmful", "adversarial_harmful",
                       "vanilla_benign", "adversarial_benign")
WILDJAILBREAK_PROMPT_COL = "adversarial"
WILDJAILBREAK_TYPE_COL = "data_type"

# The AlphaSteer authors' benign pool (14,000 prompts claimed), a flat JSON
# list of strings materialised on the cluster by an earlier development branch
# (see class_b.benign_prompts_path). Same file the Class B arms fit on — see
# the config note: sharing it makes the Class-A-vs-Class-B benign composition
# less independent, acceptable because both are FIT sets (never a scoring set).
# It lives on cluster storage, not the grading host, so its count is unverified there.
# The default is in placeholder form; _load_prompt_list expands $SHARED (sisteer.site).
ALPHASTEER_BENIGN_PATH = "$SHARED/exp/jailbreak/contrast/benign_alphasteer.json"

# THE LEAKAGE GUARD:
# OR-Bench-hard-1k and XSTest are the over-refusal EVAL CONTROLS —
# the held-out prompts the fitted defenses are judged on. Fitting the benign
# set on either optimises the defense not to fire on those exact prompts,
# which silently invalidates the over-refusal read for precisely the
# defenses most likely to over-refuse. So the fit path REFUSES them outright
# (raise, never warn) rather than relying on the or-bench-80k default
# holding. load_orbench itself stays permissive: the EVAL side legitimately
# loads hard-1k through it.
#
# Two KNOWN LIMITATIONS of this guard, both accepted (2026-08-19): (1) it
# never inspects an explicitly-passed prompt LIST — alpaca_path_prompts or
# any pre-built list bypasses it entirely; (2) it is NAME-based, so a
# hard-1k CSV saved under another filename passes. The normalised-content
# subtraction below (subtract_hard_1k, wired into load_orbench) is the
# content-level backstop for the or-bench-80k fit path; nothing
# content-level guards XSTest.
_EVAL_CONTROL_MARKERS = ("or-bench-hard-1k", "xstest")

# Fit-set size floor applied whenever a side comes from an explicit
# prompt-list file (see build_content_contrast) or a caller passes
# min_prompts. The d x d covariance / null-space fits these sets feed need
# n >> d (d = 4096 for Llama-3.1-8B): a 41-prompt harmful set once fit a
# 4096 x 4096 AlphaSteer matrix without a peep and the failure only
# surfaced at eval. (Floor design lifted from an earlier development
# branch's benign_prompts_path/harmful_prompts_path seam, commit 9d95cb5.)
MIN_PROMPTS_DEFAULT = 500


def _refuse_eval_controls(**sources) -> None:
    """Raise if any named fit-set source is an over-refusal eval control."""
    for arg, value in sources.items():
        if value is None:
            continue
        norm = str(value).lower().replace("_", "-")
        for marker in _EVAL_CONTROL_MARKERS:
            if marker in norm:
                raise ValueError(
                    f"{arg}={value!r} names {marker!r}, an over-refusal EVAL "
                    "CONTROL: the fitted defenses are judged for over-refusal "
                    "on OR-Bench-hard-1k and XSTest, so fitting the benign "
                    "set on them would bias the defense not to fire on the "
                    "very prompts that measure it, invalidating the control. "
                    "Fit on or-bench-80k (the default) instead."
                )


def _det_sample(items: list[str], n: int | None, seed: int) -> list[str]:
    """Deterministic length-n subsample (or all, shuffled) — stable across
    runs so a re-fit reproduces the same content set."""
    if n is None or n >= len(items):
        picked = list(items)
    else:
        rng = random.Random(seed)
        picked = rng.sample(items, n)
    picked.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    return picked


def _hard_1k_prompts(*, revision: str = OR_BENCH_REVISION,
                     path: str | None = None) -> list[str]:
    """ALL OR-Bench-hard-1k rows, stripped but NOT char-filtered — the eval
    set is judged intact, so every row must be held out of any fit sample."""
    import pandas as pd

    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(OR_BENCH_REPO,
                               OR_BENCH_FILES["or-bench-hard-1k"],
                               repo_type="dataset", revision=revision)
    df = pd.read_csv(path)
    return [str(p).strip() for p in df["prompt"].tolist()]


def subtract_hard_1k(prompts: list[str], hard_1k: list[str]) -> list[str]:
    """Drop every prompt that appears in OR-Bench-hard-1k, by NORMALISED
    exact match — ``jailbreak_data.normalized_text_sha`` (strip, lowercase,
    collapse whitespace), the repo's cross-dataset prompt-disjointness
    convention. ``matched_pool.text_sha`` is deliberately raw-bytes for
    attack-variant identity and would miss a trivially reformatted duplicate
    here, so it is the wrong hash for this job.

    THE LEAKAGE FIX (measured 2026-08-19 at the pinned revision):
    OR-Bench-hard-1k is a near-verbatim
    subset of or-bench-80k — 1,318 of its 1,319 rows appear exactly in the
    80k file — so _refuse_eval_controls' advice to "fit on or-bench-80k
    instead" did NOT hold the eval items out; ~21 verbatim hard-1k eval
    prompts sat in the seed-0 Class-B fit sample. Every 80k fit sample must
    therefore subtract hard-1k, BEFORE the deterministic subsample so the
    realised sample size is honest (the subtraction removes 1,318 of the
    80,355 char-filtered 80k rows)."""
    held_out = {normalized_text_sha(p) for p in hard_1k}
    return [p for p in prompts if normalized_text_sha(p) not in held_out]


def load_orbench(
    config: str = "or-bench-80k",
    n: int | None = None,
    *,
    seed: int = 0,
    revision: str = OR_BENCH_REVISION,
    max_chars: int = 300,
    path: str | None = None,
    hard_1k_path: str | None = None,
) -> list[str]:
    """OR-Bench prompts (over-refusal-adjacent benign). `path` overrides the
    download for offline tests. Filtered to single-line prompts within
    `max_chars` to match the Alpaca prompt shape, then deterministically
    subsampled to `n`.

    The ``or-bench-80k`` config SUBTRACTS the hard-1k eval rows first (see
    subtract_hard_1k): always when 80k comes from the hub, and on a ``path``
    override when ``hard_1k_path`` supplies the hard-1k side. An offline
    ``path`` override WITHOUT ``hard_1k_path`` skips the subtraction — the
    caller owns that file's contents, and tests must never hit the hub."""
    import pandas as pd

    if config not in OR_BENCH_FILES:
        raise ValueError(f"unknown OR-Bench config {config!r}; "
                         f"known: {sorted(OR_BENCH_FILES)}")
    if hard_1k_path is not None and config != "or-bench-80k":
        raise ValueError(
            f"hard_1k_path is only meaningful for the or-bench-80k fit "
            f"pool, not config={config!r} (subtracting the eval set from "
            "itself would silently empty it)")
    downloaded = path is None
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(OR_BENCH_REPO, OR_BENCH_FILES[config],
                               repo_type="dataset", revision=revision)
    df = pd.read_csv(path)
    prompts = [
        str(p).strip() for p in df["prompt"].tolist()
        if 10 <= len(str(p).strip()) <= max_chars
    ]
    if config == "or-bench-80k" and (downloaded or hard_1k_path is not None):
        prompts = subtract_hard_1k(
            prompts, _hard_1k_prompts(revision=revision, path=hard_1k_path))
    return _det_sample(prompts, n, seed)


def load_phtest(
    n: int | None = None,
    *,
    seed: int = 0,
    split: str = "all",
    max_chars: int = 300,
    revision: str = PHTEST_REVISION,
    path: str | None = None,
) -> list[str]:
    """PHTest prompts — the Stage-3 hard-benign source (borderline
    pseudo-harmful requests a well-calibrated model should answer).

    ``split`` selects the ``Harmfulness`` label: ``"harmless"``,
    ``"controversial"``, or ``"all"`` (both, 3,269 rows at the pinned
    revision). ``path`` overrides the download for offline tests. Same
    conventions as load_orbench: single-line prompts within 10..``max_chars``
    chars, deterministically subsampled to ``n`` (seed-stable, so a re-fit
    reproduces the same benign set)."""
    import pandas as pd

    if split not in PHTEST_SPLITS:
        raise ValueError(f"unknown PHTest split {split!r}; "
                         f"known: {PHTEST_SPLITS}")
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(PHTEST_REPO, PHTEST_FILE,
                               repo_type="dataset", revision=revision)
    df = pd.read_csv(path)
    if split != "all":
        df = df[df["Harmfulness"] == split]
    prompts = [
        str(p).strip() for p in df["Request"].tolist()
        if 10 <= len(str(p).strip()) <= max_chars
    ]
    return _det_sample(prompts, n, seed)


def select_xstest_complied(graded) -> tuple[list, dict[str, int]]:
    """The benign-acceptance filter: keep ONLY the items the SI-prompted
    teacher fully COMPLIED with under the XSTest C/P/I rubric.

    ``graded`` is what the graders already return —
    ``judge_batch.grade_xstest`` / ``LiveStrongREJECTGrader.grade_xstest``'s
    ``{item: "C" | "P" | "I"}`` mapping (an iterable of ``(item, label)``
    pairs also works; do NOT reimplement the grading, this consumes it).
    Returns ``(accepted, counts)``: the ``"C"``-labelled items in input
    order, and a ``{"C": _, "P": _, "I": _}`` tally of every label seen so
    the rejection yield is reportable. ``P`` (partial refusal) is REJECTED
    exactly like ``I`` — nothing is distilled from hedged half-answers. An
    unknown label raises (fail loud; a grading bug must never silently drop
    or admit rows). Deliberately importable without the judge client:
    selection is pure bookkeeping over labels."""
    rows = graded.items() if hasattr(graded, "items") else graded
    accepted: list = []
    counts = {"C": 0, "P": 0, "I": 0}
    for item, label in rows:
        if label not in counts:
            raise ValueError(f"unknown XSTest label {label!r} for item "
                             f"{item!r}; expected one of C/P/I")
        counts[label] += 1
        if label == "C":
            accepted.append(item)
    return accepted, counts


def _char_filtered(texts: list[str], max_chars: int) -> list[str]:
    """The shared benign-prompt shape filter: strip, keep 10..``max_chars``
    chars. NOTE this is a LENGTH filter only — despite the "single-line"
    wording in the older loaders' docstrings it does NOT drop multi-line
    prompts (verified 2026-08-20: 70 multi-line PHTest rows survive). Kept
    identical to load_phtest / load_orbench so the corpora mix on one shape."""
    return [str(p).strip() for p in texts
            if 10 <= len(str(p).strip()) <= max_chars]


def subtract_eval_controls(
    prompts: list[str], control_texts: list[str],
) -> list[str]:
    """Drop every prompt whose NORMALISED text matches an eval-control prompt
    (``normalized_text_sha`` — strip/lowercase/collapse-whitespace, the repo's
    cross-dataset disjointness convention). The content-level backstop for the
    scale-up benign corpora: the name-based ``_refuse_eval_controls`` cannot
    inspect an authored list (its known limitation), and the pool builder's
    ``_strip_scoring`` is the authoritative pool-level enforcement — this lets
    a loader ALSO subtract OR-Bench-hard-1k / XSTest-safe up front so a corpus
    is disjoint before it is ever generated on."""
    held_out = {normalized_text_sha(t) for t in control_texts}
    return [p for p in prompts if normalized_text_sha(p) not in held_out]


def load_wildjailbreak_split(
    data_type: str,
    n: int | None = None,
    *,
    seed: int = 0,
    max_chars: int = 300,
    revision: str | None = WILDJAILBREAK_REVISION,
    path: str | None = None,
    control_texts: list[str] | None = None,
) -> list[str]:
    """One WildJailbreak train stratum as a prompt list.

    ``data_type`` is one of ``WILDJAILBREAK_TYPES``. The ``adversarial`` column
    carries the jailbreak-styled prompt for the two ``adversarial_*`` strata;
    the ``vanilla_*`` strata carry theirs in ``vanilla``, so the column is
    chosen from the stratum rather than hardcoded (reading ``adversarial`` for
    a vanilla row yields the empty string for every row — a silent empty
    corpus, which this picks apart instead).

    Verified against the cluster snapshot 2026-08-20: gated on the hub, so the
    read is offline-cache-only on cluster B; the loader RAISES on a missing
    column or an unknown stratum rather than guessing.
    """
    import pandas as pd

    if data_type not in WILDJAILBREAK_TYPES:
        raise ValueError(f"unknown WildJailbreak data_type {data_type!r}; "
                         f"known: {list(WILDJAILBREAK_TYPES)}")
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(WILDJAILBREAK_REPO, WILDJAILBREAK_TRAIN_FILE,
                               repo_type="dataset", revision=revision)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    col = "adversarial" if data_type.startswith("adversarial") else "vanilla"
    for need in (WILDJAILBREAK_TYPE_COL, col):
        if need not in df.columns:
            raise ValueError(
                f"WildJailbreak TSV {path} is missing the {need!r} column "
                f"(have {list(df.columns)}) — refusing to guess the schema; "
                "verify the snapshot on the cluster and re-pin.")
    rows = df[df[WILDJAILBREAK_TYPE_COL] == data_type]
    if rows.empty:
        raise ValueError(
            f"WildJailbreak TSV {path} has no {data_type!r} rows (present: "
            f"{sorted(df[WILDJAILBREAK_TYPE_COL].unique())[:8]}) — refusing a "
            "silently empty corpus.")
    prompts = _char_filtered(rows[col].tolist(), max_chars)
    if control_texts:
        prompts = subtract_eval_controls(prompts, control_texts)
    return _det_sample(prompts, n, seed)


#: Char cap for the two ADVERSARIAL WildJailbreak strata. The benign shape
#: filter's 300 comes from PHTest, whose prompts are plain one-line requests
#: (median 75 chars for wildjailbreak's own vanilla_benign). The adversarial
#: strata are jailbreak-STYLED and long by construction — measured on the
#: pinned snapshot 2026-08-20: adversarial_benign median 727 chars, of which
#: only 423 of 78,710 rows (0.5%) fall under 300; adversarial_harmful median
#: 862, only 118 of 82,728 (0.1%). Applying the PHTest cap therefore discards
#: ~99.5% of the corpus AND biases what survives toward the shortest, least
#: representative attacks. 2000 keeps 99.1% / 97.7%.
WILDJAILBREAK_ADVERSARIAL_MAX_CHARS = 2000


def load_wildjailbreak_benign(
    n: int | None = None,
    *,
    seed: int = 0,
    max_chars: int = WILDJAILBREAK_ADVERSARIAL_MAX_CHARS,
    revision: str | None = WILDJAILBREAK_REVISION,
    path: str | None = None,
    control_texts: list[str] | None = None,
) -> list[str]:
    """WildJailbreak ``adversarial_benign`` prompts — jailbreak-styled benign
    requests, stylistically closer to the harmful side than PHTest (scale-up
    amendment 2026-08-20). A thin wrapper over ``load_wildjailbreak_split`` so
    the benign registry entry and the harmful sampler share one reader."""
    return load_wildjailbreak_split(
        WILDJAILBREAK_BENIGN_TYPE, n, seed=seed, max_chars=max_chars,
        revision=revision, path=path, control_texts=control_texts)


def load_alphasteer_benign(
    n: int | None = None,
    *,
    seed: int = 0,
    max_chars: int = 300,
    path: str | None = None,
    control_texts: list[str] | None = None,
) -> list[str]:
    """The AlphaSteer authors' benign pool — a flat JSON list of strings
    (scale-up make-up source, amendment 2026-08-20). Reuses the verbatim
    prompt-list reader (``_load_prompt_list``: loud on a missing/empty/
    non-string-list file), then the shared char filter, optional eval-control
    subtraction, and the deterministic subsample. ``path`` defaults to the
    cluster location (``$SHARED/...``, expanded on read); the file is not on the
    grading host, so its count is unverified there. Shared with the Class B fit
    set — see ALPHASTEER_BENIGN_PATH."""
    if path is None:
        path = ALPHASTEER_BENIGN_PATH
    prompts, _sha = _load_prompt_list(path, "alphasteer_benign")
    prompts = _char_filtered(prompts, max_chars)
    if control_texts:
        prompts = subtract_eval_controls(prompts, control_texts)
    return _det_sample(prompts, n, seed)


#: The benign-corpus loader registry (scale-up amendment 2026-08-20): the pool
#: builder's ``pool.benign_sources[].loader`` names resolve here. Each loader
#: takes ``(n, *, seed, path, ...)`` and returns a list of prompt strings.
BENIGN_LOADERS = {
    "phtest": load_phtest,
    "wildjailbreak_benign": load_wildjailbreak_benign,
    "alphasteer_benign": load_alphasteer_benign,
}


def load_harmbench_texts(
    split: str = "val", n: int | None = None, *, seed: int = 0,
    path: str | None = None,
) -> list[str]:
    """HarmBench behaviour texts for the harmful (fire) set. Default split
    ``val`` (80 behaviours) keeps the ``test`` split free for evaluation —
    the fit set must be disjoint from what the object is judged on."""
    goals = load_harmbench(split=split, path=path)
    texts = [g.text for g in goals]
    return _det_sample(texts, n, seed)


def _load_prompt_list(path: str | Path, side: str) -> tuple[list[str], str]:
    """A VERBATIM prompt list from a JSON file: a flat non-empty list of
    strings, used exactly as stored — no sampling, no shuffling, no
    truncation, no dataset access. Every malformation fails LOUDLY
    (missing file, non-list JSON, empty list, non-string elements): a
    silently empty or truncated fit set is the exact failure this seam
    exists to prevent. Returns ``(prompts, sha256-of-file-bytes)`` so the
    fit metadata can pin exactly which list was consumed. A site placeholder
    in ``path`` (``$SHARED/...``) is expanded first and refuses by name when
    it cannot be (sisteer.site)."""
    from sisteer import site as _site

    p = _site.expand_path(path, key=f"{side} prompts path")
    if not p.is_file():
        raise FileNotFoundError(f"{side} prompts file not found: {p}")
    raw = p.read_bytes()
    data = json.loads(raw)  # malformed JSON raises JSONDecodeError: loud
    if not isinstance(data, list):
        raise ValueError(
            f"{side} prompts file {p} must hold a flat JSON list of "
            f"strings, got {type(data).__name__}")
    if not data:
        raise ValueError(
            f"{side} prompts file {p} holds an EMPTY list — refusing to "
            f"fit on an empty {side} set")
    bad = [i for i, x in enumerate(data) if not isinstance(x, str)]
    if bad:
        raise ValueError(
            f"{side} prompts file {p}: every element must be a string; "
            f"{len(bad)} are not (first at index {bad[0]}: "
            f"{type(data[bad[0]]).__name__})")
    return list(data), hashlib.sha256(raw).hexdigest()


@dataclass
class ContentContrast:
    """A labeled harmful/benign content contrast — the SI-agnostic fit input.
    ``benign`` and ``harmful`` are raw instruction strings; the trainer
    renders them SI-free."""

    benign: list[str]
    harmful: list[str]
    meta: dict = field(default_factory=dict)


def build_content_contrast(
    *,
    n_benign: int = 2000,
    alpaca_frac: float = 0.5,
    n_harmful: int | None = None,
    harmbench_split: str = "val",
    orbench_config: str = "or-bench-80k",
    seed: int = 0,
    alpaca_revision: str | None = None,
    orbench_revision: str = OR_BENCH_REVISION,
    alpaca_path_prompts: list[str] | None = None,
    orbench_path: str | None = None,
    orbench_hard_1k_path: str | None = None,
    harmbench_path: str | None = None,
    benign_prompts_path: str | None = None,
    harmful_prompts_path: str | None = None,
    min_prompts: int | None = None,
) -> ContentContrast:
    """Assemble the fit-once contrast: benign = Alpaca + OR-Bench-80k
    (mixed by ``alpaca_frac``), harmful = HarmBench.

    Refuses outright (ValueError) any benign source that is one of this
    lane's over-refusal eval controls — see _refuse_eval_controls above.

    ``alpaca_path_prompts`` injects the Alpaca prompts directly (tests /
    offline); otherwise they come from load_hf_prompts. Deterministic in
    ``seed`` end to end.

    ``benign_prompts_path`` / ``harmful_prompts_path``: the explicit
    prompt-list seam (lifted from an earlier development branch, commit
    9d95cb5, 2026-08-20 — same design, same motivation). Each is a JSON
    file holding a flat list of prompt strings, loaded VERBATIM for that
    side (no sampling, no shuffling, no truncation, no dataset access) —
    so an arm can be fit on externally-authored lists such as the
    AlphaSteer authors' own 14,000/2,000 composition. The two sides are
    independently overridable; a side without a path keeps the assembly
    above exactly, so existing callers are unaffected. Each consumed
    file's sha256 lands in ``meta`` (``benign_sha256``/``harmful_sha256``)
    so the fit record pins exactly which list it saw. NOTE the name-based
    ``_refuse_eval_controls`` guard cannot inspect an authored list — its
    known limitation (1) above — so the caller owns eval-control
    disjointness for these files.

    ``min_prompts``: hard floor on the size of BOTH fit sets. ON by
    default (at :data:`MIN_PROMPTS_DEFAULT` = 500) whenever either
    prompts-path is given — a mixed configuration that pairs an authored
    file with the 41-behaviour HarmBench-val default is precisely the
    silent under-fit this exists to stop — and OFF when neither is given
    (back-compat with every legacy caller). Pass an int to set the floor
    explicitly in any mode (tests/smoke runs pass a small one; the raise
    stays loud by default)."""
    benign_sha = harmful_sha = None
    if benign_prompts_path is not None:
        benign, benign_sha = _load_prompt_list(benign_prompts_path, "benign")
        benign_sources: dict[str, int] = {
            f"file:{benign_prompts_path}": len(benign)}
        hard1k_subtracted = False
    else:
        _refuse_eval_controls(orbench_config=orbench_config,
                              orbench_path=orbench_path)
        n_alpaca = int(round(n_benign * alpaca_frac))
        n_orbench = n_benign - n_alpaca
        if alpaca_path_prompts is not None:
            alpaca = _det_sample(alpaca_path_prompts, n_alpaca, seed)
        else:
            alpaca = _det_sample(
                load_hf_prompts(ALPACA_DATASET,
                                n=max(n_alpaca * 2, n_alpaca + 50),
                                seed=seed, revision=alpaca_revision),
                n_alpaca, seed)
        orbench = load_orbench(orbench_config, n_orbench, seed=seed,
                               revision=orbench_revision, path=orbench_path,
                               hard_1k_path=orbench_hard_1k_path)
        benign = _det_sample(alpaca + orbench, None, seed)  # interleave stably
        benign_sources = {"alpaca": len(alpaca), orbench_config: len(orbench)}
        # was the hard-1k eval set subtracted from the 80k
        # fit pool? (mirrors load_orbench's condition)
        hard1k_subtracted = (
            orbench_config == "or-bench-80k"
            and (orbench_path is None or orbench_hard_1k_path is not None))
    if harmful_prompts_path is not None:
        harmful, harmful_sha = _load_prompt_list(harmful_prompts_path,
                                                 "harmful")
        harmful_source = f"file:{harmful_prompts_path}"
    else:
        harmful = load_harmbench_texts(harmbench_split, n_harmful, seed=seed,
                                       path=harmbench_path)
        harmful_source = f"harmbench:{harmbench_split}"

    guard_on = (min_prompts is not None or benign_prompts_path is not None
                or harmful_prompts_path is not None)
    floor = MIN_PROMPTS_DEFAULT if min_prompts is None else int(min_prompts)
    if guard_on:
        for side, prompts in (("benign", benign), ("harmful", harmful)):
            if len(prompts) < floor:
                raise ValueError(
                    f"{side} fit set has only {len(prompts)} prompts, below "
                    f"min_prompts={floor}. The d x d covariance/null-space "
                    "fits this contrast feeds need n >> d (d = 4096 for "
                    "Llama-3.1-8B); an undersized set fits without error "
                    "and only fails at eval — a 41-prompt harmful set "
                    "(HarmBench-val silently reduced to standard-only) sank "
                    "an AlphaSteer fit this way. Supply more prompts, or "
                    "pass min_prompts=<small int> for a deliberate "
                    "smoke/test run.")

    return ContentContrast(
        benign=benign, harmful=harmful,
        meta={
            "benign_sources": benign_sources,
            "harmful_source": harmful_source,
            "n_benign": len(benign), "n_harmful": len(harmful),
            "benign_sha256": benign_sha, "harmful_sha256": harmful_sha,
            "min_prompts": floor if guard_on else None,
            "seed": seed, "alpaca_frac": alpaca_frac,
            "orbench_revision": orbench_revision,
            "alpaca_revision": alpaca_revision,
            "orbench_hard1k_subtracted": hard1k_subtracted,
        },
    )
