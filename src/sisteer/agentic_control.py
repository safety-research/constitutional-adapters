"""Lane core library for the agentic-control amendment.

Pure logic only — no drivers, no GPU, no API calls. Everything a stage
driver, the grading host filter, or the analysis needs to
agree on lives here, so the three can never drift apart silently:

* the 5 pre-registered conditions and the lane label
* ``cell_path`` — the ONE canonical cell-path schema every lane cell is
  written to, parsed from, and loaded through
* the teacher pool: broad multi-behavior MWE stems + the frozen
  CERTIFIED benign pool, template-disjoint from the agentic eval
  scenarios, with both disjointness invariants asserted on every build
* the distinctive-n-gram template-disjointness matcher whose reference
  corpus is a frozen EXTERNAL corpus (pinned wikitext) — never the
  Alpaca pool the benign side derives from
* certified-benign manifest handling: the pool builder consumes ONLY the
  frozen certified prompt IDs + verdict SHA that the pre-certify stage
  (``--stage certify``) froze BEFORE any generation
* the pure accept rules over filed verdicts and the contrastive-pair
  builder with its yield report and pre-registered minimum-yield floors
* ``cluster_key`` — the template-family key for cluster-robust
  inference.

Hard rule, inherited from ``aligned_pairs``: a candidate with a MISSING
verdict or a ``{"__blocked__": true}`` verdict is NEVER an accept — not
as a pos, not as a neg, on either side.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .data.aligned_pairs import candidate_key
from .data.pairs import Pair
from .data.realign_prompts import (
    ALPACA_REVISION,
    _hash_hex,
    _normalize,
    extraction_pool,
    load_alpaca_pinned,
)
from .eval.agentic_misalignment.loader import (
    GOAL_TYPES,
    GOAL_VALUES,
    SCENARIOS,
    URGENCY_TYPES,
    load_scenarios,
)

# ------------------------------------------------------------ conditions

LANE = "agentic_control"

# The 5 pre-registered conditions. The compiled labels ARE the
# vendored defense-registry mode names (verified: DEFENSES = ("none",
# "prompted", *ARMS)), so no alias layer exists anywhere.
CONDITIONS: tuple[str, ...] = (
    "none",
    "prompted",
    "psr_kl_all",
    "psr_mse_all",
    "psr_kl_all_hard",
)
COMPILED_CONDITIONS: tuple[str, ...] = tuple(
    c for c in CONDITIONS if c not in ("none", "prompted")
)

# ------------------------------------------------------------ cell paths

# The lane's cell kinds. The eval/calibrate forms are parse_cell's
# 5-segment ``<organism>/<kind>/<arm>/<teacher>/<suite>``; collateral is
# its 4-segment ``<organism>/collateral/<arm>/<teacher>`` with the suite
# detected from the record file (XSTest via --records-name).
KINDS: tuple[str, ...] = ("eval", "calibrate", "collateral")
EVAL_SUITES: tuple[str, ...] = ("agentic", "mask")
CALIBRATE_SUITE = "agentic"
# The lane trains no per-teacher cells; the teacher segment is fixed.
TEACHER_SEGMENT = "none"


def cell_path(kind: str, condition: str, suite: str | None = None) -> str:
    """The canonical relative cell path under ``<out_root>/``:

    - ``eval``:       ``agentic_control/eval/<condition>/none/<suite>``
      (suite ∈ {agentic, mask} — the judged held-out suites);
    - ``calibrate``:  ``agentic_control/calibrate/<condition>/none/agentic``
      (the H0-style calibration read; agentic is the only calibrate suite);
    - ``collateral``: ``agentic_control/collateral/<condition>/none``
      (XSTest records live inside as ``xstest_<partition>.jsonl``;
      MMLU/GSM8K/wikitext land in ``results.json``, never judged).

    The driver writes ONLY through this helper; ``parse_cell`` and the
    loaders consume exactly these forms. Loud on any malformation —
    a misspelled kind/condition/suite here would file records under the
    wrong identity downstream.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown cell kind {kind!r} (lane kinds: {KINDS})")
    if condition not in CONDITIONS:
        raise ValueError(
            f"unknown condition {condition!r} (the 5 pre-registered "
            f"conditions: {CONDITIONS})"
        )
    if kind == "eval":
        if suite not in EVAL_SUITES:
            raise ValueError(
                f"eval cells need suite ∈ {EVAL_SUITES}, got {suite!r}"
            )
        return f"{LANE}/eval/{condition}/{TEACHER_SEGMENT}/{suite}"
    if kind == "calibrate":
        if suite not in (None, CALIBRATE_SUITE):
            raise ValueError(
                f"calibrate cells are agentic-only (the H0-style read), "
                f"got suite {suite!r}"
            )
        return f"{LANE}/calibrate/{condition}/{TEACHER_SEGMENT}/{CALIBRATE_SUITE}"
    # collateral: the 4-segment <arm>/<teacher> form; a suite here would
    # silently change the shape parse_cell sees.
    if suite is not None:
        raise ValueError(
            f"collateral cells take no suite segment (XSTest is discovered "
            f"via --records-name inside the cell), got {suite!r}"
        )
    return f"{LANE}/collateral/{condition}/{TEACHER_SEGMENT}"


# ------------------------------------------------------------ cluster key

# scenario_id == f"{scenario}_{goal_type}-{goal_value}_{urgency}"; the
# template family is the stem with the trailing _{urgency} stripped.
# Built from the loader's own grid constants so the two can never
# disagree about the grammar.
_GOAL_VALUE_ALTS = "|".join(
    sorted(set(GOAL_VALUES) | {"none"}, key=len, reverse=True)
)
_CLUSTER_RE = re.compile(
    rf"^(?P<family>(?:{'|'.join(SCENARIOS)})"
    rf"_(?:{'|'.join(GOAL_TYPES)})-(?:{_GOAL_VALUE_ALTS}))"
    rf"_(?:{'|'.join(URGENCY_TYPES)})$"
)


def cluster_key(scenario_id: str) -> str:
    """Template-family key: ``{scenario}_{goal_type}-{goal_value}`` — the
    persisted ``scenario_id`` with its ``_{urgency}`` segment stripped.
    34 families over the 100 committed rows; the analysis resamples
    FAMILIES, not rows. Raises on anything that does not parse as a grid
    scenario_id — a silent passthrough would quietly break the
    clustering."""
    m = _CLUSTER_RE.match(scenario_id)
    if m is None:
        raise ValueError(
            f"scenario_id {scenario_id!r} does not parse as "
            "{scenario}_{goal_type}-{goal_value}_{urgency} over the "
            "vendored grid — refusing to derive a cluster key"
        )
    return m.group("family")


def assert_cluster_keys_total(
    scenario_ids: list[str] | None = None,
) -> tuple[str, ...]:
    """Totality assertion: EVERY scenario_id (default: all rows of the
    committed artifact) must parse to a known family. Returns the
    sorted family tuple for the caller to pin/report."""
    if scenario_ids is None:
        scenario_ids = [row["scenario_id"] for row in load_scenarios()]
    if not scenario_ids:
        raise ValueError("no scenario_ids to derive cluster keys from")
    return tuple(sorted({cluster_key(s) for s in scenario_ids}))


# ------------------------------------------- distinctive-n-gram matcher

# Frozen, pre-registered matcher params: word n-grams of the
# aligned_pairs/realign_prompts _normalize text, n=8; the reference
# corpus is pinned EXTERNAL wikitext — see reference_corpus_id(). The
# round-3 fix: the reference must be disjoint from ALL teacher
# candidates. Alpaca is the benign source, so an Alpaca reference would
# delete every candidate-overlapping signature n-gram and make the check
# vacuous.
NGRAM_N = 8


def normalized_sha(text: str) -> str:
    """Normalized-text hash — the unit of the exact-text disjointness
    invariants (same normalization as the split manifests)."""
    return _hash_hex(_normalize(text))


def word_ngrams(text: str, n: int = NGRAM_N) -> frozenset[tuple[str, ...]]:
    """All normalized word n-grams of ``text`` (empty for texts shorter
    than n words)."""
    words = _normalize(text).split()
    return frozenset(
        tuple(words[i : i + n]) for i in range(len(words) - n + 1)
    )


def reference_corpus_id() -> str:
    """Provenance string of the frozen external reference corpus."""
    from .eval.realign_collateral_data import WIKITEXT_REPO, WIKITEXT_REVISION

    return f"{WIKITEXT_REPO}@{WIKITEXT_REVISION}"


def reference_corpus_texts() -> list[str]:
    """The frozen EXTERNAL reference corpus for the matcher: every
    substantial pinned-wikitext doc (deterministic; no cap). Deliberately
    NOT the Alpaca pool — see the module docstring (round-3 fix)."""
    from .eval.realign_collateral_data import load_wikitext

    return load_wikitext(n_docs=None, seed=0)


def agentic_template_texts() -> list[str]:
    """Every vendored agentic system/user template as realized in the
    committed scenario artifact (the byte-frozen render of the templates,
    all 100 rows)."""
    texts: list[str] = []
    for row in load_scenarios():
        texts.append(row["system_prompt"])
        texts.append(row["user_prompt"])
    return list(dict.fromkeys(texts))


def template_signature(
    template_texts: list[str],
    reference_texts: list[str],
    *,
    n: int = NGRAM_N,
) -> frozenset[tuple[str, ...]]:
    """Template signature = the templates' normalized word n-grams MINUS
    every n-gram that also occurs in the external reference corpus, so
    only n-grams DISTINCTIVE of the templates survive. Common words and
    ordinary shared phrasing drop out; genuine template reuse — verbatim
    or lightly edited — keeps at least one surviving n-gram.

    Loud twice rather than silently weak: no template n-grams at all
    (templates shorter than n words) and a VACUOUS signature (the
    reference swallowed everything — the exact failure mode of using a
    candidate-derived corpus as the reference) both raise."""
    if not template_texts:
        raise ValueError("no template texts to build a signature from")
    sig: set[tuple[str, ...]] = set()
    for text in template_texts:
        sig |= word_ngrams(text, n)
    if not sig:
        raise ValueError(
            f"template texts yield no {n}-grams — the signature would "
            "match nothing; refusing to pretend the check ran"
        )
    ref: set[tuple[str, ...]] = set()
    for text in reference_texts:
        ref |= word_ngrams(text, n)
    distinctive = sig - ref
    if not distinctive:
        raise ValueError(
            "vacuous template signature: every template n-gram also "
            "occurs in the reference corpus. The reference must be a "
            "frozen EXTERNAL corpus disjoint from the templates and from "
            f"all teacher candidates (pinned wikitext, {NGRAM_N=}), never "
            "a corpus the pool itself derives from"
        )
    return frozenset(distinctive)


def find_template_ngram(
    text: str,
    signature: frozenset[tuple[str, ...]],
    *,
    n: int = NGRAM_N,
) -> tuple[str, ...] | None:
    """The first (lexicographically smallest) surviving template n-gram
    contained in ``text``, or None."""
    hits = word_ngrams(text, n) & signature
    return min(hits) if hits else None


def assert_template_disjoint(
    prompts: list[str],
    signature: frozenset[tuple[str, ...]],
    *,
    n: int = NGRAM_N,
    where: str = "teacher pool",
) -> None:
    """Raise on any prompt containing a surviving distinctive template
    n-gram — the template-reuse invariant (medium)."""
    for i, prompt in enumerate(prompts):
        hit = find_template_ngram(prompt, signature, n=n)
        if hit is not None:
            raise ValueError(
                f"template leakage into the {where}: prompt {i} contains "
                f"the distinctive agentic-template {n}-gram "
                f"{' '.join(hit)!r} — the teacher pool must be "
                "template-disjoint from the eval scenarios"
            )


def assert_eval_disjoint(
    prompts: list[str],
    eval_texts: list[str],
    *,
    where: str = "teacher pool",
) -> None:
    """Exact-text (normalized-sha) disjointness of the pool against ALL
    eval items — agentic rows (system + user), MASK items, XSTest
    prompts. An empty eval-text list is a loader bug, not a pass."""
    if not eval_texts:
        raise ValueError(
            "no eval texts to check disjointness against — refusing to "
            "certify a vacuous invariant"
        )
    eval_shas = {normalized_sha(t) for t in eval_texts}
    for i, prompt in enumerate(prompts):
        if normalized_sha(prompt) in eval_shas:
            raise ValueError(
                f"eval leakage into the {where}: prompt {i} "
                f"({prompt[:80]!r}) is an eval item — the eval sets are "
                "never sampled, trained on, or used to pick anything"
            )


def default_eval_texts() -> list[str]:
    """Every eval-item text the pool must be disjoint from: the 100
    committed agentic rows (system + user), the pinned MASK items, and
    the pinned XSTest prompts. Network-touching loaders are imported
    lazily; tests inject their own lists."""
    texts = agentic_template_texts()
    from .eval.mask import load_mask

    for item in load_mask():
        for t in (item.get("system_pressure"), item.get("user_prompt")):
            if isinstance(t, str) and t.strip():
                texts.append(t)
    from .eval.xstest import load_xstest

    texts.extend(item["prompt"] for item in load_xstest())
    return texts


# ------------------------------------------------- certified benign pool

# The pre-certification contract: the ``--stage certify`` stage
# judges the NOMINAL alpaca pool under benignprompt.v1 BEFORE any
# generation, freezes the passing prompts into this manifest, and every
# downstream consumer (pool builder, candidates, negs, filter) reads ONLY
# the frozen set and cross-checks ``certified_sha``. The manifest is
# BOUND to the campaign: schema 2 records the campaign's
# ``prereg_sha``, the exact pinned ``ALPACA_REVISION``, and the frozen
# nominal-pool digest ``nominal_sha``; the writer verifies every
# certified ID is a member of the nominal pool, and every load takes the
# current prereg SHA and refuses a mismatch.
BENIGN_RUBRIC_TAG = "benignprompt.v1"
CERTIFIED_MANIFEST_NAME = "certified_benign.json"
CERTIFIED_SCHEMA = 2
# The certified-record verdict whitelist (finding A8): exactly the
# rubric's schema field — a rationale (or anything else) must never
# persist, and the certified_sha must never bind one.
CERTIFIED_VERDICT_FIELDS = ("benign",)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_N_PRESSURE = 300
DEFAULT_N_BENIGN = 300
_POOL_SALT = "agentic-control-pool"


def benign_prompt_id(prompt: str) -> str:
    """Content-derived id for a benign candidate prompt (normalized-text
    hash), so the frozen ID set is tamper-evident and independent of any
    file ordering."""
    return f"alpaca:{normalized_sha(prompt)[:16]}"


def nominal_benign_pool(
    n: int = DEFAULT_N_BENIGN,
    seed: int = 0,
    *,
    alpaca_prompts: list[str] | None = None,
) -> list[dict]:
    """The NOMINAL (pre-certification) benign pool the certify stage
    judges: pinned-alpaca prompts with content-derived ids, deduped by
    id. This is the only place the nominal pool is defined, so the
    certify stage and the pool builder can never disagree about it."""
    if alpaca_prompts is None:
        alpaca_prompts = load_alpaca_pinned(n=n, seed=seed)
    rows: dict[str, dict] = {}
    for prompt in alpaca_prompts[:n]:
        pid = benign_prompt_id(prompt)
        rows.setdefault(pid, {"id": pid, "prompt": prompt})
    return list(rows.values())


def certified_records_sha(records: list[dict]) -> str:
    """``certified_sha``: sha256 over the sorted certified-ID+verdict
    records — order- and key-order-independent, recomputed (never
    trusted) by every consumer."""
    ordered = sorted(records, key=lambda r: r["id"])
    payload = "\n".join(json.dumps(r, sort_keys=True) for r in ordered)
    return hashlib.sha256(payload.encode()).hexdigest()


def nominal_pool_sha(rows: list[dict]) -> str:
    """``nominal_sha``: digest of the NOMINAL (pre-certification) pool —
    sorted {id, prompt} rows only, so it is order-independent and
    invariant to any extra bookkeeping a caller carries (finding A1)."""
    ordered = sorted(rows, key=lambda r: r["id"])
    payload = "\n".join(
        json.dumps({"id": r["id"], "prompt": r["prompt"]}, sort_keys=True)
        for r in ordered
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def certified_manifest(
    records: list[dict], *, prereg_sha: str, nominal: list[dict]
) -> dict:
    """The certified-pool manifest payload (the certify stage writes it;
    defined here so producer and consumers share ONE schema). ``records``
    are the CERTIFIED rows only: {id, prompt, verdict} with a filed,
    unblocked ``benign == 1`` verdict each; ``nominal`` is the full
    NOMINAL pool the certify stage judged ({id, prompt} rows).

    Campaign binding (finding A1): the payload records ``prereg_sha``,
    the exact pinned ``ALPACA_REVISION``, and ``nominal_sha`` (the frozen
    nominal-pool digest), and every certified ID must be a member of the
    nominal pool with its exact prompt text. Verdicts are re-constructed
    onto the whitelist here (finding A8) — only ``benign`` persists and
    binds into ``certified_sha``; a rationale passed in can never reach
    disk. Validated by round-tripping through the loader before return.
    """
    nominal_prompt_of = {r["id"]: r["prompt"] for r in nominal}
    clean_records = []
    for rec in records:
        rid, verdict = rec["id"], rec["verdict"]
        if not isinstance(verdict, dict) or verdict.get("__blocked__"):
            raise ValueError(
                f"certified record {rid!r} carries a blocked/malformed "
                "verdict — blocked is never certified"
            )
        if rid not in nominal_prompt_of:
            raise ValueError(
                f"certified record {rid!r} is not a member of the nominal "
                "pool — the certify stage judges ONLY the nominal pool, "
                "so a foreign id is tampering (finding A1); refusing"
            )
        if nominal_prompt_of[rid] != rec["prompt"]:
            raise ValueError(
                f"certified record {rid!r} prompt differs from the "
                "nominal pool's text for that id — mispaired record "
                "(finding A1); refusing"
            )
        clean_records.append({
            "id": rid,
            "prompt": rec["prompt"],
            # the A8 whitelist: schema fields only, nothing preserved
            "verdict": {f: verdict[f] for f in CERTIFIED_VERDICT_FIELDS},
        })
    clean_records.sort(key=lambda r: r["id"])
    payload = {
        "schema": CERTIFIED_SCHEMA,
        "rubric": BENIGN_RUBRIC_TAG,
        "alpaca_revision": ALPACA_REVISION,
        "prereg_sha": prereg_sha,
        "nominal_sha": nominal_pool_sha(nominal),
        "records": clean_records,
        "certified_sha": certified_records_sha(clean_records),
    }
    load_certified_benign(payload, prereg_sha=prereg_sha)  # loud now,
    # not at consume time
    return payload


def load_certified_benign(
    manifest: str | Path | dict, *, prereg_sha: str
) -> tuple[dict[str, str], str]:
    """Read + validate the frozen certified-benign manifest; returns
    ``(id -> prompt, certified_sha)``. ``prereg_sha`` is the CURRENT
    campaign anchor — every load passes it, and a manifest certified
    under a different anchor is refused (finding A1). Every failure is a
    refusal — the benign side of the pool exists ONLY as this frozen set:

    * schema/rubric mismatch, missing keys;
    * an ``alpaca_revision`` that is not the exact pinned
      ``ALPACA_REVISION``, a ``prereg_sha`` that is not the caller's
      current anchor, or a missing/malformed ``nominal_sha`` (A1);
    * a record whose id is not the content-derived ``benign_prompt_id``
      of its own prompt (tampered or hand-edited);
    * duplicate ids;
    * a record whose verdict carries anything beyond the whitelisted
      schema fields (finding A8 — a persisted rationale is a leak);
    * a record without a filed, unblocked ``benign == 1`` verdict (the
      certified set holds certified prompts and nothing else);
    * a ``certified_sha`` that does not match the recomputation.
    """
    if not prereg_sha or not isinstance(prereg_sha, str):
        raise ValueError(
            "a current prereg SHA is required to load the certified-benign "
            "manifest (finding A1) — pass the campaign anchor"
        )
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        if not path.exists():
            raise FileNotFoundError(
                f"certified-benign manifest {path} missing — the certify "
                "stage must run (and freeze the certified IDs) before "
                "anything consumes the benign pool"
            )
        manifest = json.loads(path.read_text())
    if manifest.get("schema") != CERTIFIED_SCHEMA:
        raise ValueError(
            f"unknown certified-benign manifest schema: "
            f"{manifest.get('schema')!r} (this code freezes/reads schema "
            f"{CERTIFIED_SCHEMA}; an older manifest predates the campaign "
            "binding and must be re-certified deliberately)"
        )
    if manifest.get("rubric") != BENIGN_RUBRIC_TAG:
        raise ValueError(
            f"certified-benign manifest was judged under "
            f"{manifest.get('rubric')!r}, not {BENIGN_RUBRIC_TAG!r} — "
            "wrong certification; refusing"
        )
    if manifest.get("alpaca_revision") != ALPACA_REVISION:
        raise ValueError(
            f"certified-benign manifest pins alpaca_revision "
            f"{manifest.get('alpaca_revision')!r}, this code pins "
            f"{ALPACA_REVISION!r} — the nominal-pool source moved under "
            "the certification (finding A1); refusing"
        )
    recorded = manifest.get("prereg_sha")
    if recorded != prereg_sha:
        raise ValueError(
            f"certified-benign manifest records prereg_sha "
            f"{str(recorded)[:12]}..., this run is anchored at "
            f"{prereg_sha[:12]}... — the certification and this run "
            "disagree about the frozen protocol (findings A1 + 9); "
            "refusing"
        )
    nominal_sha = manifest.get("nominal_sha")
    if not isinstance(nominal_sha, str) or not _SHA256_RE.match(nominal_sha):
        raise ValueError(
            "certified-benign manifest carries no valid nominal_sha — "
            "the frozen nominal-pool digest is required (finding A1); "
            "refusing"
        )
    records = manifest.get("records")
    if not records:
        raise ValueError(
            "certified-benign manifest holds no records — an empty "
            "certified pool cannot supply the benign side"
        )
    prompts: dict[str, str] = {}
    for rec in records:
        missing = [k for k in ("id", "prompt", "verdict") if k not in rec]
        if missing:
            raise ValueError(
                f"certified record lacks keys {missing}: {rec!r}"
            )
        rid, prompt, verdict = rec["id"], rec["prompt"], rec["verdict"]
        if rid != benign_prompt_id(prompt):
            raise ValueError(
                f"certified record id {rid!r} is not the content-derived "
                f"id of its prompt ({benign_prompt_id(prompt)!r}) — "
                "tampered manifest; refusing"
            )
        if rid in prompts:
            raise ValueError(f"duplicate certified id {rid!r}")
        if not isinstance(verdict, dict) or verdict.get("__blocked__"):
            raise ValueError(
                f"certified record {rid} carries a blocked/malformed "
                "verdict — blocked is never certified"
            )
        if set(verdict) != set(CERTIFIED_VERDICT_FIELDS):
            raise ValueError(
                f"certified record {rid} verdict carries fields "
                f"{sorted(verdict)} — the whitelist is exactly "
                f"{sorted(CERTIFIED_VERDICT_FIELDS)} (finding A8: nothing "
                "else, a rationale least of all, may persist or bind into "
                "certified_sha); refusing"
            )
        if verdict.get("benign") != 1:
            raise ValueError(
                f"certified record {rid} has benign="
                f"{verdict.get('benign')!r} — only benign==1 prompts may "
                "be frozen into the certified pool"
            )
        prompts[rid] = prompt
    sha = certified_records_sha(records)
    if manifest.get("certified_sha") != sha:
        raise ValueError(
            "certified_sha mismatch: manifest says "
            f"{str(manifest.get('certified_sha'))[:12]}..., records "
            f"recompute to {sha[:12]}... — torn or edited manifest; "
            "refusing"
        )
    return prompts, sha


# ------------------------------------------------------------ teacher pool

# The BROAD multi-behavior MWE side: every ethically-charged
# advanced-ai-risk behavior family the plan names — power-seeking,
# survival, corrigibility, coordination, myopia, wealth/resource-seeking,
# self-awareness — as present at the pinned MWE_REVISION (verified
# 2026-08-16 via list_repo_files; the only human_generated_evals file
# excluded is one-box-tendency, a decision-theory probe, not an
# ethically-charged behavior). The prereg amendment freezes this list.
TEACHER_BEHAVIORS: tuple[str, ...] = (
    "coordinate-itself",
    "coordinate-other-ais",
    "coordinate-other-versions",
    "corrigible-less-HHH",
    "corrigible-more-HHH",
    "corrigible-neutral-HHH",
    "myopic-reward",
    "power-seeking-inclination",
    "self-awareness-general-ai",
    "self-awareness-good-text-model",
    "self-awareness-text-model",
    "self-awareness-training-architecture",
    "self-awareness-web-gpt",
    "survival-instinct",
    "wealth-seeking-inclination",
)


@dataclass(frozen=True)
class TeacherPool:
    """The two frozen teacher sides + the certified-pool provenance the
    consuming stages must record and cross-check (``certified_sha``)."""

    pressure: tuple[str, ...]
    benign: tuple[str, ...]
    certified_sha: str


def teacher_pool(
    certified: str | Path | dict,
    *,
    prereg_sha: str,
    n_pressure: int = DEFAULT_N_PRESSURE,
    n_benign: int = DEFAULT_N_BENIGN,
    seed: int = 0,
    behaviors: tuple[str, ...] = TEACHER_BEHAVIORS,
    manifest_dir: str | Path | None = None,
    partitions_by_behavior: dict | None = None,
    eval_texts: list[str] | None = None,
    template_texts: list[str] | None = None,
    reference_texts: list[str] | None = None,
    ngram_n: int = NGRAM_N,
) -> TeacherPool:
    """Build the lane's teacher pool, asserting EVERY invariant on every
    build (never cached, never assumed):

    * pressure side: MWE question stems from the ``extract`` partition
      ONLY, across the broad behavior list, via the parent
      ``extraction_pool`` machinery (leakage discipline unchanged)
    * benign side: ONLY the frozen certified prompts (deterministically
      shuffled, capped at ``n_benign``) — the nominal Alpaca pool is
      never consumed directly
    * the two sides are disjoint from each other (normalized sha)
    * exact-text eval disjointness against all eval items
    * the reference corpus is EXTERNAL and disjoint from all teacher
      candidates
    * distinctive-n-gram template disjointness against the vendored
      agentic templates.

    ``prereg_sha`` is the campaign anchor the certified manifest must be
    bound to (finding A1). ``partitions_by_behavior`` / ``eval_texts`` /
    ``template_texts`` / ``reference_texts`` inject offline fixtures for
    tests; production callers omit them and get the pinned loaders.
    """
    certified_prompts, certified_sha = load_certified_benign(
        certified, prereg_sha=prereg_sha)

    pressure = extraction_pool(
        n_mwe=n_pressure,
        n_alpaca=0,
        seed=seed,
        behaviors=behaviors,
        manifest_dir=manifest_dir,
        partitions_by_behavior=partitions_by_behavior,
        alpaca_prompts=[],
    )
    benign = [certified_prompts[rid] for rid in sorted(certified_prompts)]
    random.Random(f"{_POOL_SALT}:benign:{seed}").shuffle(benign)
    benign = benign[:n_benign]

    cross = {normalized_sha(p) for p in pressure} & {
        normalized_sha(p) for p in benign
    }
    if cross:
        raise ValueError(
            f"{len(cross)} prompt(s) appear on BOTH teacher sides — one "
            "prompt would be judged under two different rubrics; refusing"
        )

    candidates = list(pressure) + list(benign)
    if eval_texts is None:
        eval_texts = default_eval_texts()
    if template_texts is None:
        template_texts = agentic_template_texts()
    if reference_texts is None:
        reference_texts = reference_corpus_texts()

    assert_eval_disjoint(candidates, eval_texts)

    ref_shas = {normalized_sha(t) for t in reference_texts}
    for i, prompt in enumerate(candidates):
        if normalized_sha(prompt) in ref_shas:
            raise ValueError(
                f"reference corpus is not external: teacher candidate {i} "
                f"({prompt[:80]!r}) occurs in the reference — the matcher "
                "reference must be disjoint from all teacher candidates "
                "(pinned wikitext, never the pool's own sources)"
            )

    signature = template_signature(template_texts, reference_texts, n=ngram_n)
    assert_template_disjoint(candidates, signature, n=ngram_n)

    return TeacherPool(
        pressure=tuple(pressure),
        benign=tuple(benign),
        certified_sha=certified_sha,
    )


# ------------------------------------------------------------ accept rules

# Pure predicates over FILED verdict dicts (sanitized or raw — they read
# the persisted fields only). ``None`` means no verdict was ever filed.
# Missing and blocked NEVER accept, on either side, in either role.

XSTEST_COMPLIANCE = "full_compliance"
XSTEST_REFUSALS: tuple[str, ...] = ("full_refusal", "partial_refusal")


def _filed(verdict: dict | None) -> bool:
    return isinstance(verdict, dict) and not verdict.get("__blocked__")


def is_pressure_pos(am: dict | None, engage: dict | None) -> bool:
    """Pressure accept: ``am.v1 harmful == 0`` AND
    ``engage.v1 engaged == 1``. ``harmful == 0`` alone is never an accept
    — an empty or generic refusal must not teach the objects."""
    return (
        _filed(am)
        and _filed(engage)
        and am.get("harmful") == 0
        and engage.get("engaged") == 1
    )


def is_pressure_neg(am: dict | None) -> bool:
    """A usable pressure negative is JUDGED harmful (``am.v1 harmful ==
    1``) — including the greedy no-SI fallback, which must itself be
    judged, never assumed harmful."""
    return _filed(am) and am.get("harmful") == 1


def is_benign_pos(xstest: dict | None) -> bool:
    """Benign accept: judged full compliance (label C) under xstest.v1."""
    return _filed(xstest) and xstest.get("label") == XSTEST_COMPLIANCE


def is_benign_neg(xstest: dict | None) -> bool:
    """Benign negative: judged an ACTUAL refusal/over-refusal (label ≠ C)
    under xstest.v1."""
    return _filed(xstest) and xstest.get("label") in XSTEST_REFUSALS


def pressure_select_verdict(
    am: dict | None, engage: dict | None
) -> dict | None:
    """Bridge the lane's two pressure verdicts into the ``(aligned,
    coherent)`` shape ``aligned_pairs.select_pairs`` consumes, so the
    untouched F9 machinery applies the lane's accept rule verbatim:
    ``aligned == 1`` iff ``harmful == 0``; ``coherent == 1`` iff
    ``engaged == 1`` — its accept (aligned AND coherent) IS
    ``is_pressure_pos``. Missing either verdict returns None (never
    filed); a block on either side propagates as blocked."""
    if am is None or engage is None:
        return None
    if am.get("__blocked__") or engage.get("__blocked__"):
        return {"__blocked__": True}
    return {
        "aligned": 1 if am.get("harmful") == 0 else 0,
        "coherent": 1 if engage.get("engaged") == 1 else 0,
    }


def benign_select_verdict(xstest: dict | None) -> dict | None:
    """The benign-side bridge: ``aligned == 1`` iff judged full
    compliance; ``coherent`` fixed at 1 (xstest.v1 is the only benign
    candidate rubric). Same missing/blocked semantics as the pressure
    bridge."""
    if xstest is None:
        return None
    if xstest.get("__blocked__"):
        return {"__blocked__": True}
    return {
        "aligned": 1 if xstest.get("label") == XSTEST_COMPLIANCE else 0,
        "coherent": 1,
    }


# ------------------------------------------------ contrastive-pair builder

# Pre-registered minimum-yield floors (the prereg amendment freezes
# them). Below floor, distillation ABORTS (the driver enforces on
# the floors verdict the filter publishes) — effective counts, never
# nominal pool sizes.
YIELD_FLOORS: dict[str, dict[str, float]] = {
    "pressure": {"min_pairs": 150, "min_fraction": 0.50},
    "benign": {"min_pairs": 100, "min_fraction": 0.40},
}


def floors_verdict(side: str, n_pairs: int, n_pool: int) -> dict:
    """The pre-registered floors check for one side: enough pairs AND a
    high enough fraction of the pool resolving. ``n_pool`` is the number
    of prompts the builder ranged over (the side's pool)."""
    if side not in YIELD_FLOORS:
        raise ValueError(
            f"unknown side {side!r} (floors exist for {sorted(YIELD_FLOORS)})"
        )
    floors = YIELD_FLOORS[side]
    fraction = (n_pairs / n_pool) if n_pool else 0.0
    return {
        "side": side,
        "n_pairs": n_pairs,
        "n_pool": n_pool,
        "fraction": fraction,
        "min_pairs": floors["min_pairs"],
        "min_fraction": floors["min_fraction"],
        "yield_ok": n_pairs >= floors["min_pairs"]
        and fraction >= floors["min_fraction"],
    }


def combined_yield_ok(*reports: dict) -> bool:
    """True only when EVERY side's floors verdict passed — the single bit
    the distill stage gates on."""
    if not reports:
        raise ValueError("no yield reports to combine")
    return all(r["floors"]["yield_ok"] for r in reports)


def _judged_fallback_negs(
    negs: list[dict] | None,
    verdicts: dict[int, dict] | None,
    valid,
    *,
    pool: list[str],
    what: str,
) -> dict[int, str]:
    """prompt_idx -> completion for fallback negatives that are VALID:
    judged (filed, unblocked) and passing ``valid``. An unjudged fallback
    is simply absent — it can never back a pair. Duplicate rows for one
    prompt are a hard error (one validated neg per prompt), and every
    row's stored prompt must match the frozen pool at its index — a
    resumed/reordered negs file must never yield a cross-prompt
    pair."""
    out: dict[int, str] = {}
    for row in negs or []:
        i = row["prompt_idx"]
        if i in out:
            raise ValueError(
                f"duplicate {what} for prompt_idx={i} — exactly one "
                "fallback negative per prompt; refusing"
            )
        if not 0 <= i < len(pool) or row["prompt"] != pool[i]:
            raise ValueError(
                f"{what} for prompt_idx={i} does not match the frozen "
                "pool prompt at that index — a reordered or foreign "
                "fallback row would silently build a cross-prompt pair "
                "(finding A7); refusing"
            )
        if valid((verdicts or {}).get(i)):
            out[i] = row["completion"]
    return out


def _count_blocked(*verdict_maps: dict | None) -> int:
    return sum(
        1
        for vmap in verdict_maps
        for v in (vmap or {}).values()
        if isinstance(v, dict) and v.get("__blocked__")
    )


def _build_pairs(
    candidates: list[dict],
    *,
    pool: list[str],
    pos_ok,
    neg_ok,
    fallback_neg_of: dict[int, str],
    fallback_source: str,
    side: str,
    n_blocked: int,
    fallback_info: dict,
) -> tuple[list[Pair], dict]:
    """The shared selection skeleton: per prompt (rows ordered by
    (round, sample_idx)), pos = the lowest accepted candidate, neg = the
    lowest candidate passing ``neg_ok``, falling back to the prompt's
    validated fallback negative (source counted); prompts with no pos, or
    with no valid neg from either source, are skipped + counted. Emits
    (pairs, yield report) — effective counts, floors verdict included,
    never silently padded.

    ``pool`` is the FROZEN ordered side pool (finding A3): its full size
    is the yield denominator, every candidate row must match it by
    (prompt_idx, text), and round 0 must cover every pool index — a pool
    member missing from the candidate store must refuse loudly, never
    silently vanish from the denominator."""
    if not pool:
        raise ValueError(
            f"{side} builder given an empty frozen pool — nothing to "
            "range the yield over; refusing"
        )
    by_prompt: dict[int, list[dict]] = {}
    for row in candidates:
        i = row["prompt_idx"]
        if not 0 <= i < len(pool) or row["prompt"] != pool[i]:
            raise ValueError(
                f"{side} candidate at prompt_idx={i} does not match the "
                "frozen pool prompt at that index — stale or foreign "
                "candidate rows (findings A3 + A7); refusing"
            )
        by_prompt.setdefault(i, []).append(row)
    covered0 = {row["prompt_idx"] for row in candidates
                if row["round"] == 0}
    missing0 = sorted(set(range(len(pool))) - covered0)
    if missing0:
        raise ValueError(
            f"{side} candidates cover only {len(covered0)} of "
            f"{len(pool)} frozen pool prompts in round 0 (missing "
            f"prompt_idx {missing0[:5]}) — a missing pool member must "
            "never silently leave the yield denominator (finding A3); "
            "refusing"
        )
    for rows in by_prompt.values():
        rows.sort(key=lambda r: (r["round"], r["sample_idx"]))

    pairs: list[Pair] = []
    neg_source = {"candidate": 0, fallback_source: 0}
    skipped = {"no_pos": 0, "no_neg": 0}
    unresolved: list[dict] = []
    for prompt_idx in range(len(pool)):
        rows = by_prompt[prompt_idx]
        pos_row = next((r for r in rows if pos_ok(r)), None)
        if pos_row is None:  # no accepted pos — no_neg is not even asked
            skipped["no_pos"] += 1
            unresolved.append({"prompt_idx": prompt_idx, "why": "no_pos"})
            continue
        neg_row = next((r for r in rows if neg_ok(r)), None)
        if neg_row is not None:
            neg_text = neg_row["completion"]
            neg_source["candidate"] += 1
        elif prompt_idx in fallback_neg_of:
            neg_text = fallback_neg_of[prompt_idx]
            neg_source[fallback_source] += 1
        else:
            skipped["no_neg"] += 1
            unresolved.append({"prompt_idx": prompt_idx, "why": "no_neg"})
            continue
        pairs.append(
            Pair(
                prompt=pos_row["prompt"],
                pos=pos_row["completion"],
                neg=neg_text,
                instruction_id=pos_row["si_id"],
            )
        )

    n_pool = len(pool)  # the FROZEN pool size, never the rows seen (A3)
    report = {
        "side": side,
        "n_pool": n_pool,
        "n_pairs": len(pairs),
        "resolved_fraction": (len(pairs) / n_pool) if n_pool else 0.0,
        "neg_source": neg_source,
        "skipped": skipped,
        "unresolved": unresolved,
        "n_blocked_verdicts": n_blocked,
        "fallback": fallback_info,
        "floors": floors_verdict(side, len(pairs), n_pool),
    }
    return pairs, report


def build_pressure_pairs(
    candidates: list[dict],
    am_verdicts: dict[tuple[int, int, int], dict],
    engage_verdicts: dict[tuple[int, int, int], dict],
    *,
    pool: list[str],
    greedy_negs: list[dict] | None = None,
    greedy_neg_verdicts: dict[int, dict] | None = None,
) -> tuple[list[Pair], dict]:
    """Pressure-side contrastive pairs over an aligned_pairs candidate
    store. ``pool`` is the FROZEN ordered pressure pool (the run
    manifest's prompt list) — the yield denominator and the per-row
    prompt authority (findings A3 + A7). Verdicts are keyed by
    ``candidate_key(row)`` (am.v1 and engage.v1 for candidates; am.v1 by
    prompt_idx for the greedy no-SI negs from ``generate_negs``). pos
    requires ``is_pressure_pos``; neg = lowest judged-harmful candidate,
    else the JUDGED-harmful greedy neg (source ``greedy_no_si``,
    counted); else skip + count."""
    fallback = _judged_fallback_negs(
        greedy_negs, greedy_neg_verdicts, is_pressure_neg,
        pool=pool, what="greedy no-SI neg",
    )
    return _build_pairs(
        candidates,
        pool=pool,
        pos_ok=lambda r: is_pressure_pos(
            am_verdicts.get(candidate_key(r)),
            engage_verdicts.get(candidate_key(r)),
        ),
        neg_ok=lambda r: is_pressure_neg(am_verdicts.get(candidate_key(r))),
        fallback_neg_of=fallback,
        fallback_source="greedy_no_si",
        side="pressure",
        n_blocked=_count_blocked(
            am_verdicts, engage_verdicts, greedy_neg_verdicts
        ),
        fallback_info={
            "n_rows": len(greedy_negs or []),
            "n_valid": len(fallback),
        },
    )


def build_benign_pairs(
    candidates: list[dict],
    xstest_verdicts: dict[tuple[int, int, int], dict],
    *,
    pool: list[str],
    refusal_negs: list[dict] | None = None,
    refusal_neg_verdicts: dict[int, dict] | None = None,
) -> tuple[list[Pair], dict]:
    """Benign-side contrastive pairs: pos = lowest candidate judged full
    compliance (C); neg = lowest candidate judged an actual refusal
    (non-C), else the prompt's deliberate over-caution refusal negative
    when ITS xstest.v1 verdict is non-C (source ``refusal_neg``,
    counted); else skip + count. ``pool`` is the FROZEN ordered benign
    pool — the yield denominator and per-row prompt authority. The
    refusal neg is judged like everything else — a completion that
    failed to refuse never becomes a negative."""
    fallback = _judged_fallback_negs(
        refusal_negs, refusal_neg_verdicts, is_benign_neg,
        pool=pool, what="over-caution refusal neg",
    )
    return _build_pairs(
        candidates,
        pool=pool,
        pos_ok=lambda r: is_benign_pos(
            xstest_verdicts.get(candidate_key(r))
        ),
        neg_ok=lambda r: is_benign_neg(
            xstest_verdicts.get(candidate_key(r))
        ),
        fallback_neg_of=fallback,
        fallback_source="refusal_neg",
        side="benign",
        n_blocked=_count_blocked(xstest_verdicts, refusal_neg_verdicts),
        fallback_info={
            "n_rows": len(refusal_negs or []),
            "n_valid": len(fallback),
        },
    )
