#!/usr/bin/env python
"""Held-out split for the sdf-base-steering campaign.

Carves ~5% of each corpus family into a held-out set at leakage-safe
granularity:

- advice: the A3-call group ``(principle_id, prompt_type_id)``
  (adjudicated 2026-09-01) — ``scenario_id`` is 1:1 with records, so a
  scenario carve would put A3 siblings (which share one generation context)
  on both sides of the split.
- stories: ``subcategory_id`` — ``call_id`` embeds the subcategory prefix
  (verified at build time, fail-loud), so a subcategory carve can never
  split a call.

Mechanics are fully deterministic: group keys are sorted, sha256-ranked
with a fixed salt, and whole groups are walked in rank order onto the
held-out side until the held-out SET's official size >= frac of the
family's — both sides measured by the AUTHORITATIVE set rule
(corpus_generator ``tokens.official_tokens``): min over the three
reference tokenizers of the SUM of the set's per-tokenizer counts. Never
the sum of per-record mins (``record_min_tokens`` is the per-record
ESTIMATOR; the limiting tokenizer can flip between subsets).

Also freezes the shared held-out EVAL subset: all held-out doc
ids sha-ranked under their own salt, first 128 taken; the realized family
counts are recorded, not assumed.

Seed replicates (the Qwen base seed-replicate battery): a
per-seed carve runs the SAME walk under a seed-specific salt
(``--salt sdfbase-heldout-v1-s<s>`` / ``--eval-salt
sdfbase-heldout-eval-v1-s<s>``) with the contrastive probes' source
documents PINNED into the held-out side (``--pin-groups-from
PROBE.jsonl``, repeatable): every group that contains a
``source == heldout_doc`` ``source_id`` enters held-out first (sha order
among themselves), then the salted walk continues over the remaining
groups until the stop rule. Pinning is at the carve's own group
granularity — doc-level pins would put a pinned doc's A3 siblings /
subcategory mates on the training side, the leakage the granularity rule
exists to prevent. The artifact then carries a ``pins`` block (probe file
shas, pinned source ids, pinned vs seed-drawn groups per family); a build
WITHOUT pins emits exactly the historical shape, so the default arguments
reproduce ``data/sdfbase/heldout_split.json`` byte for byte.

Output: ``data/sdfbase/heldout_split.json`` — ids only (synthetic pipeline
ids, no corpus text), committed to the worktree by the orchestrator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

SALT_HELDOUT = "sdfbase-heldout-v1"
SALT_EVAL = "sdfbase-heldout-eval-v1"
FRAC = 0.05
EVAL_N = 128
TOK_NAMES = ("llama3", "gemma", "qwen")

WORKTREE_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Corpus CODE root vs corpus DATA root.
#
# The corpus-generation package lives at <repo>/corpus_generator (its own
# pyproject, locked with the harness); ``corpus_generator.tokens`` — the renderers and the official
# token rule — is imported from ``<code root>/src`` by path. ONE code root per
# builder process: ``build_pools.py`` and ``build_ctl2_pools.py`` import this
# module and inherit its import-time ``GENERATOR_ROOT``, so the process-wide
# override is the environment variable ``CORPUS_GENERATOR_ROOT`` (the front
# door exports it for every builder it launches); ``--code-root`` is the
# explicit per-invocation override of this script alone.
#
# The DATA (the released finals, the staged reference tokenizers) is NOT in
# the generator tree — stage 02 materialises the finals from the committed
# shards (corpora/) into ``$SISTEER_DATA_ROOT/finals`` and stages the
# tokenizers into ``$SISTEER_DATA_ROOT/tokenizers``, then passes the paths
# EXPLICITLY (``--advice-final/--stories-final``, ``--tokenizers-dir``). The
# module-level defaults below exist for the grading host workflow and the tests that
# gate on ``ADVICE_FINAL.exists()``: they resolve under
# ``CORPUS_GENERATOR_DATA_ROOT`` (else ``<code root>/data``, the generator's
# layout) and a CLI run whose resolved final does not exist REFUSES by name
# rather than guessing another location.
# --------------------------------------------------------------------------
GENERATOR_ROOT = Path(os.environ.get("CORPUS_GENERATOR_ROOT") or (WORKTREE_ROOT / "corpus_generator"))
GENERATOR_DATA_ROOT = Path(os.environ.get("CORPUS_GENERATOR_DATA_ROOT") or (GENERATOR_ROOT / "data"))
ADVICE_FINAL = GENERATOR_DATA_ROOT / "advice/final-production-20260831-v2.jsonl"
STORIES_FINAL = GENERATOR_DATA_ROOT / "stories/final-production-20260831-v2.jsonl"
SPLIT_OUT = WORKTREE_ROOT / "data/sdfbase/heldout_split.json"


def set_code_root(code_root) -> Path:
    """``--code-root``: repoint the corpus CODE root for this process (the
    data root and the finals' defaults are unaffected). Returns the new root."""
    global GENERATOR_ROOT
    GENERATOR_ROOT = Path(code_root)
    return GENERATOR_ROOT


def tokenizers_dir(explicit=None) -> Path:
    """The staged reference tokenizers: ``explicit`` (a ``--tokenizers-dir``),
    else ``$CORPUS_GENERATOR_TOKENIZERS_DIR``, else ``<data root>/tokenizers``."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("CORPUS_GENERATOR_TOKENIZERS_DIR")
    return Path(env) if env else GENERATOR_DATA_ROOT / "tokenizers"


def require_final(path, flag: str) -> Path:
    """A final that must exist: refuse by name (the flag to pass and the env
    root it was resolved under) instead of building from a guessed file."""
    path = Path(path)
    if not path.is_file():
        raise SplitError(
            f"corpus final not found: {path} — pass {flag} <file> (stage 02 "
            f"materialises the finals from corpora/ with "
            f"reproduce/tools/reassemble_corpus.py) or set "
            f"CORPUS_GENERATOR_DATA_ROOT to a directory holding the generator's "
            f"data/ layout"
        )
    return path


class SplitError(RuntimeError):
    """Loud failure — never degrade to a silently-wrong split."""


# --------------------------------------------------------------------------
# Generator-tree provenance (production builds refuse a dirty tree)
# --------------------------------------------------------------------------

def tree_state(repo: Path) -> dict:
    """{"commit", "dirty", "diff_sha256"} of one generator repo.

    ``dirty`` is True when ``git status --porcelain`` reports anything
    (untracked files included — they can be imported by the builders);
    ``diff_sha256`` then hashes the porcelain listing + ``git diff HEAD``
    bytes, so two builds from the SAME dirty tree record the same diff
    identity. A tree git cannot describe (not a repo, no git) records
    commit="unknown", dirty=None (unverifiable, stated honestly) — tests
    and exported checkouts must not be taken hostage."""
    import hashlib as _hashlib
    import subprocess as _sp

    def _git(*argv: str) -> str:
        out = _sp.run(["git", "-C", str(repo), *argv],
                      capture_output=True, text=True, timeout=30,
                      check=True)
        return out.stdout

    try:
        commit = _git("rev-parse", "HEAD").strip()
        porcelain = _git("status", "--porcelain")
        dirty = bool(porcelain.strip())
        diff_sha = None
        if dirty:
            diff = _sp.run(["git", "-C", str(repo), "diff", "HEAD"],
                           capture_output=True, timeout=60, check=True)
            diff_sha = _hashlib.sha256(
                porcelain.encode("utf-8") + b"\x00" + diff.stdout
            ).hexdigest()
        return {"commit": commit, "dirty": dirty, "diff_sha256": diff_sha}
    except Exception:  # noqa: BLE001 — not a repo / no git: unverifiable
        return {"commit": "unknown", "dirty": None, "diff_sha256": None}


def _generator_repos() -> tuple[tuple[str, Path], ...]:
    """The generator repos every sdfbase artifact records: this worktree
    (the builder code) and the corpus repo (renderers + corpus data)."""
    # Late-bound so a test overriding GENERATOR_ROOT via env still resolves.
    return (("sisteer", WORKTREE_ROOT), ("corpus_generator", GENERATOR_ROOT))


def generator_provenance() -> dict:
    """Flat provenance keys for the artifact manifests —
    ``<repo>_commit`` (the existing shape, kept), plus ``<repo>_dirty``
    and ``<repo>_diff_sha256`` so a --allow-dirty build is honestly
    marked and re-identifiable."""
    out: dict = {}
    for name, repo in _generator_repos():
        st = tree_state(repo)
        out[f"{name}_commit"] = st["commit"]
        out[f"{name}_dirty"] = st["dirty"]
        out[f"{name}_diff_sha256"] = st["diff_sha256"]
    return out


def refuse_dirty_generator_tree(allow_dirty: bool, *, tag: str) -> dict:
    """The production-build gate: a DIRTY generator tree refuses unless
    ``--allow-dirty`` — an artifact stamped with a commit its bytes were
    not built from is false provenance. With the override, the build
    proceeds and the caller records ``dirty=true`` + the tree-diff sha
    (via :func:`generator_provenance`, returned here for convenience).
    An UNVERIFIABLE tree (dirty=None — no git) passes: it is recorded as
    unknown rather than blocking non-repo checkouts."""
    prov = generator_provenance()
    dirty = [name for name, _repo in _generator_repos()
             if prov.get(f"{name}_dirty") is True]
    if dirty and not allow_dirty:
        raise SplitError(
            f"[{tag}] generator tree(s) DIRTY: {', '.join(dirty)} — a "
            "production build must run from committed code (the recorded "
            "commit is the artifact's provenance). Commit first, or pass "
            "--allow-dirty to record dirty=true + the tree-diff sha "
            "instead.")
    return prov


def generator_tokens(code_root=None):
    """Import ``corpus_generator.tokens`` from the corpus code root (the
    ``<repo>/corpus_generator`` by default; ``code_root`` / ``--code-root`` /
    ``$CORPUS_GENERATOR_ROOT`` override it).

    Import by path: importing the corpus pipeline's own renderers
    is what gives byte-parity by construction.
    """
    root = Path(code_root) if code_root else GENERATOR_ROOT
    src = root / "src"
    if not src.is_dir():
        raise SplitError(
            f"corpus_generator code root not found at {root} (expected "
            f"{src} — the corpus_generator/ of this repo; set "
            "CORPUS_GENERATOR_ROOT or pass --code-root to override)"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from corpus_generator import tokens  # noqa: PLC0415

    return tokens


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def sha_rank(salt: str, key: str) -> str:
    """Deterministic rank digest for a key under a fixed salt."""
    return hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()


def doc_id(rec: dict) -> str:
    return rec["lineage"]["custom_id"]


def advice_group_key(rec: dict) -> str:
    """Canonical string form of the A3-call group (principle_id, prompt_type_id)."""
    return f"{rec['principle_id']}|{rec['prompt_type_id']}"


def story_group_key(rec: dict) -> str:
    return rec["subcategory_id"]


def record_token_counts(rec: dict) -> dict[str, int]:
    counts = rec.get("token_counts") or {}
    if set(counts) != set(TOK_NAMES):
        raise SplitError(
            f"{doc_id(rec)}: token_counts keys {sorted(counts)} != {sorted(TOK_NAMES)}"
        )
    return {k: int(counts[k]) for k in TOK_NAMES}


def set_totals(records: list[dict]) -> dict[str, int]:
    """Per-tokenizer SUM over a set of records — the set rule's raw totals."""
    totals = {k: 0 for k in TOK_NAMES}
    for rec in records:
        for k, n in record_token_counts(rec).items():
            totals[k] += n
    return totals


def official_of(totals: dict[str, int]) -> tuple[int, str]:
    """(official token count, limiting tokenizer) of a set's totals.

    Official = min across per-tokenizer totals (the AUTHORITATIVE rule,
    corpus_generator tokens.py:132-140). Ties break by sorted name.
    """
    limiting = min(sorted(totals), key=lambda k: totals[k])
    return totals[limiting], limiting


def verify_structure(advice: list[dict], stories: list[dict]) -> None:
    """Fail loudly on any structural assumption the carve depends on."""
    all_ids = [doc_id(r) for r in advice] + [doc_id(r) for r in stories]
    if len(set(all_ids)) != len(all_ids):
        raise SplitError("duplicate lineage.custom_id across the corpus")
    for r in advice:
        if not r["prompt_type_id"].startswith(r["principle_id"] + "-"):
            raise SplitError(
                f"{doc_id(r)}: prompt_type_id {r['prompt_type_id']!r} does not "
                f"embed principle_id {r['principle_id']!r}"
            )
        if not r["scenario_id"].startswith(r["prompt_type_id"] + "-"):
            raise SplitError(
                f"{doc_id(r)}: scenario_id {r['scenario_id']!r} does not embed "
                f"prompt_type_id {r['prompt_type_id']!r}"
            )
        if not r["user"] or not r["assistant"]:
            raise SplitError(f"{doc_id(r)}: empty user/assistant field")
    for r in stories:
        # call_id embeds the subcategory prefix — this is what makes a
        # subcategory carve unable to split a call.
        if not r["call_id"].startswith(r["subcategory_id"] + "-"):
            raise SplitError(
                f"{doc_id(r)}: call_id {r['call_id']!r} does not embed "
                f"subcategory_id {r['subcategory_id']!r} — subcategory carve "
                "would no longer guarantee call integrity"
            )
        if not r["subcategory_id"].startswith(r["category_id"] + "-"):
            raise SplitError(
                f"{doc_id(r)}: subcategory_id does not embed category_id"
            )
        if not r["text"]:
            raise SplitError(f"{doc_id(r)}: empty text field")


def carve_family(records: list[dict], key_fn, salt: str, frac: float,
                 pinned_groups=()) -> dict:
    """Carve one family: whole groups, sha-rank walk, authoritative stop rule.

    ``pinned_groups`` (seed replicates): group keys that enter the
    held-out side FIRST, in sha order among themselves under ``salt``; the
    salted walk then continues over the REMAINING groups until
    ``min(cum) >= frac × family_official``. When the pins alone already
    meet the stop rule no group is seed-drawn (the historical 5 % carve
    with the probe groups pinned is exactly this case). A pinned key the
    family does not have refuses. With no pins the walk is the historical
    one, byte for byte.

    Returns a dict with the held-out group keys (walk order: pinned first,
    then seed-drawn), held-out doc ids (corpus file order), the pinned /
    seed-drawn partition of the groups, and full per-tokenizer accounting.
    """
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(key_fn(rec), []).append(rec)

    family_totals = set_totals(records)
    family_official, family_limiting = official_of(family_totals)
    threshold = frac * family_official

    pinned = set(pinned_groups)
    unknown = sorted(pinned - set(groups))
    if unknown:
        raise SplitError(
            f"pinned group(s) not in this family: {unknown[:10]}"
            f"{' …' if len(unknown) > 10 else ''} — a pin must name a "
            "group the carve can move")

    def _rank(keys) -> list[str]:
        return sorted(sorted(keys), key=lambda k: (sha_rank(salt, k), k))

    pinned_ranked = _rank(pinned)
    ranked = _rank(g for g in groups if g not in pinned)
    cum = {k: 0 for k in TOK_NAMES}
    heldout_groups: list[str] = []

    def _take(g: str) -> None:
        for rec in groups[g]:
            for k, n in record_token_counts(rec).items():
                cum[k] += n
        heldout_groups.append(g)

    for g in pinned_ranked:
        _take(g)
    pinned_cum = dict(cum)
    seed_drawn: list[str] = []
    if not (pinned and min(cum.values()) >= threshold):
        for g in ranked:
            _take(g)
            seed_drawn.append(g)
            if min(cum.values()) >= threshold:
                break
        else:
            raise SplitError(
                f"walked every group without reaching {threshold:.1f} official "
                f"tokens (family official {family_official}) — frac {frac} is "
                "not attainable"
            )

    heldout_set = set(heldout_groups)
    heldout_ids = [doc_id(r) for r in records if key_fn(r) in heldout_set]
    heldout_official, heldout_limiting = official_of(cum)
    train_totals = {k: family_totals[k] - cum[k] for k in TOK_NAMES}
    train_official, train_limiting = official_of(train_totals)
    pinned_set = set(pinned_ranked)
    pinned_official, _ = official_of(pinned_cum)
    n_pinned_records = sum(len(groups[g]) for g in pinned_ranked)
    return {
        "groups": heldout_groups,
        "ids": heldout_ids,
        "pinned_groups": pinned_ranked,
        "seed_drawn_groups": seed_drawn,
        "pin_accounting": {
            "n_pinned_groups": len(pinned_ranked),
            "n_seed_drawn_groups": len(seed_drawn),
            "pinned_records": n_pinned_records,
            "seed_drawn_records": len(heldout_ids) - n_pinned_records,
            "pinned_per_tokenizer": pinned_cum,
            "pinned_official": pinned_official,
            "pinned_frac_of_family": pinned_official / family_official,
            # the seed-drawn share of the HELD-OUT mass (O2's "≈ 40 %")
            "seed_drawn_share_of_heldout": (
                (heldout_official - pinned_official) / heldout_official
                if heldout_official else 0.0),
            "pinned_ids": [doc_id(r) for r in records
                           if key_fn(r) in pinned_set],
        },
        "accounting": {
            "family_records": len(records),
            "family_groups": len(groups),
            "family_per_tokenizer": family_totals,
            "family_official": family_official,
            "family_limiting_tokenizer": family_limiting,
            "heldout_records": len(heldout_ids),
            "heldout_groups": len(heldout_groups),
            "heldout_per_tokenizer": cum,
            "heldout_official": heldout_official,
            "heldout_limiting_tokenizer": heldout_limiting,
            "heldout_frac_realized": heldout_official / family_official,
            "train_records": len(records) - len(heldout_ids),
            "train_per_tokenizer": train_totals,
            "train_official": train_official,
            "train_limiting_tokenizer": train_limiting,
        },
    }


def select_eval_subset(heldout_ids_by_family: dict[str, list[str]],
                       salt: str = SALT_EVAL, n: int = EVAL_N) -> dict:
    """The shared held-out EVAL pool ids: sha-rank ALL held-out docs, take
    the first n. Uniform ranking approximates the natural mix; realized
    family counts are recorded, not assumed."""
    pooled = [(i, fam) for fam, ids in heldout_ids_by_family.items() for i in ids]
    pooled.sort(key=lambda t: (sha_rank(salt, t[0]), t[0]))
    take = pooled[:n]
    if len(take) < n:
        raise SplitError(
            f"held-out set has only {len(take)} docs, need {n} for the eval subset"
        )
    return {
        "salt": salt,
        "n": n,
        "ids": [i for i, _ in take],
        # Counter preserves first-seen order — same dict as the manual fold.
        "family_counts": dict(Counter(fam for _, fam in take)),
    }


PIN_SOURCE = "heldout_doc"


def load_probe_source_ids(probe_paths) -> list[str]:
    """The sorted, de-duplicated ``source_id``s of every probe item whose
    ``source`` is ``heldout_doc`` across the given probe files (the
    contrastive probes' continuation items are built from held-out
    documents; the ``constitution_section`` items pin nothing). A probe
    file with no held-out-doc item, or an item missing its ``source_id``,
    refuses — a silent empty pin would quietly un-pin every seed."""
    ids: set[str] = set()
    for p in probe_paths:
        p = Path(p)
        if not p.is_file():
            raise SplitError(f"probe file {p} missing")
        n_here = 0
        for rec in load_jsonl(p):
            if rec.get("source") != PIN_SOURCE:
                continue
            sid = rec.get("source_id")
            if not sid:
                raise SplitError(
                    f"{p.name}: a {PIN_SOURCE} probe item ({rec.get('id')!r}) "
                    "carries no source_id — cannot pin it")
            ids.add(str(sid))
            n_here += 1
        if n_here == 0:
            raise SplitError(
                f"{p.name}: no probe item with source == {PIN_SOURCE!r} — "
                "nothing to pin (wrong file?)")
    return sorted(ids)


def resolve_pinned_groups(source_ids, advice: list[dict],
                          stories: list[dict]) -> dict[str, list[str]]:
    """Map probe source doc ids onto the carve's GROUP keys per family
    (``advice_group_key`` / ``story_group_key``). Every source id must be a
    corpus doc id (``lineage.custom_id``); an unknown id refuses — a probe
    built from a document this corpus does not carry cannot be pinned, and
    pretending otherwise would leave it unpinned silently."""
    where: dict[str, tuple[str, str]] = {}
    for rec in advice:
        where[doc_id(rec)] = ("advice", advice_group_key(rec))
    for rec in stories:
        where[doc_id(rec)] = ("stories", story_group_key(rec))
    groups: dict[str, set[str]] = {"advice": set(), "stories": set()}
    unknown = []
    for sid in source_ids:
        hit = where.get(sid)
        if hit is None:
            unknown.append(sid)
            continue
        groups[hit[0]].add(hit[1])
    if unknown:
        raise SplitError(
            f"{len(unknown)} probe source id(s) are not corpus doc ids "
            f"(first: {unknown[:5]}) — cannot pin what the corpus does not "
            "carry")
    return {fam: sorted(g) for fam, g in groups.items()}


def build_split(advice_final: Path, stories_final: Path, *,
                frac: float = FRAC, salt: str = SALT_HELDOUT,
                eval_salt: str = SALT_EVAL, eval_n: int = EVAL_N,
                pin_probe_files=None) -> dict:
    """Build the full split artifact dict (deterministic, no timestamps,
    NO generator state — a committed split must byte-match a fresh
    rebuild regardless of the builder's commit, which is exactly what
    tests pin. Generator provenance — commit/dirty/diff-sha — lives in
    the CLI-written ``.provenance.json`` sidecar instead).

    ``pin_probe_files`` (seed replicates): probe jsonl paths whose
    ``heldout_doc`` source documents are pinned into the held-out side at
    group granularity (:func:`carve_family` ``pinned_groups``). The
    artifact then carries a ``pins`` block; without pins the artifact's
    shape is the historical one, byte for byte."""
    advice_final = require_final(advice_final, "--advice-final")
    stories_final = require_final(stories_final, "--stories-final")
    advice = load_jsonl(advice_final)
    stories = load_jsonl(stories_final)
    verify_structure(advice, stories)

    pin_probe_files = [Path(p) for p in (pin_probe_files or [])]
    pinned: dict[str, list[str]] = {"advice": [], "stories": []}
    pinned_source_ids: list[str] = []
    if pin_probe_files:
        pinned_source_ids = load_probe_source_ids(pin_probe_files)
        pinned = resolve_pinned_groups(pinned_source_ids, advice, stories)

    adv = carve_family(advice, advice_group_key, salt, frac,
                       pinned_groups=pinned["advice"])
    sto = carve_family(stories, story_group_key, salt, frac,
                       pinned_groups=pinned["stories"])
    eval_subset = select_eval_subset(
        {"advice": adv["ids"], "stories": sto["ids"]}, salt=eval_salt, n=eval_n
    )
    split = {
        "salt": salt,
        "frac": frac,
        "advice": {
            "group_key": "(principle_id, prompt_type_id)",
            "groups": adv["groups"],
            "ids": adv["ids"],
        },
        "stories": {
            "group_key": "subcategory_id",
            "groups": sto["groups"],
            "ids": sto["ids"],
        },
        "token_totals": {
            "advice": adv["accounting"],
            "stories": sto["accounting"],
        },
        "eval_subset": eval_subset,
        "built_from_sha256": {
            "advice_final": sha256_file(advice_final),
            "stories_final": sha256_file(stories_final),
        },
    }
    if pin_probe_files:
        # Emitted ONLY for a pinned build — the unpinned artifact keeps its
        # historical bytes (the committed seed-0 split is that build).
        split["pins"] = {
            "rule": "every carve group containing a probe source document "
                    "(source == heldout_doc) enters held-out first, in sha "
                    "order under the split salt; the salted walk then "
                    "continues over the remaining groups to the stop rule",
            "source": PIN_SOURCE,
            "probe_files": [{"name": p.name, "sha256": sha256_file(p)}
                            for p in pin_probe_files],
            "pinned_source_ids": pinned_source_ids,
            "n_pinned_source_ids": len(pinned_source_ids),
            "advice": {"pinned_groups": adv["pinned_groups"],
                       "seed_drawn_groups": adv["seed_drawn_groups"],
                       **adv["pin_accounting"]},
            "stories": {"pinned_groups": sto["pinned_groups"],
                        "seed_drawn_groups": sto["seed_drawn_groups"],
                        **sto["pin_accounting"]},
        }
    return split


def split_provenance(split: dict, *, artifact: Path, generator: dict,
                     salt: str | None = None, eval_salt: str | None = None,
                     frac: float | None = None, extra: dict | None = None
                     ) -> dict:
    """The ``.provenance.json`` sidecar body: artifact sha + generator
    tree state (the historical keys), plus a ``carve`` summary — salts,
    requested and realized fractions, train-side official tokens, and the
    pinned / seed-drawn group counts when the split was pinned. ``extra``
    (e.g. a lane's fraction-override reason) is recorded under ``carve``
    verbatim. Tolerates a split lacking any of the summary fields (older
    or stubbed artifacts): a missing field records null, never a guess."""
    totals = split.get("token_totals") or {}
    pins = split.get("pins") or {}
    fams = ("advice", "stories")
    carve = {
        "salt": split.get("salt", salt),
        "eval_salt": (split.get("eval_subset") or {}).get("salt", eval_salt),
        "frac_requested": split.get("frac", frac),
        "frac_realized": {f: (totals.get(f) or {}).get("heldout_frac_realized")
                          for f in fams},
        "heldout_records": {f: (totals.get(f) or {}).get("heldout_records")
                            for f in fams},
        "train_side_official_tokens": {
            f: (totals.get(f) or {}).get("train_official") for f in fams},
        "pinned": bool(pins),
        "probe_files": [p.get("name") for p in pins.get("probe_files", [])],
        "n_pinned_source_ids": pins.get("n_pinned_source_ids", 0),
        "pinned_groups": {
            f: len((pins.get(f) or {}).get("pinned_groups") or [])
            for f in fams},
        "seed_drawn_groups": {
            f: len((pins.get(f) or {}).get("seed_drawn_groups") or [])
            for f in fams},
        "seed_drawn_share_of_heldout": {
            f: (pins.get(f) or {}).get("seed_drawn_share_of_heldout")
            for f in fams},
    }
    if extra:
        carve.update(extra)
    return {"artifact": Path(artifact).name,
            "artifact_sha256": sha256_file(Path(artifact)),
            "carve": carve,
            "generator": generator}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--advice-final", type=Path, default=ADVICE_FINAL,
                    help="the constitution advice final (stage 02 passes the "
                         "materialised file explicitly; a missing default "
                         "refuses)")
    ap.add_argument("--stories-final", type=Path, default=STORIES_FINAL,
                    help="the constitution stories final (as above)")
    ap.add_argument("--code-root", type=Path, default=None,
                    help="corpus_generator code root (default <repo>/corpus_generator, "
                         "or $CORPUS_GENERATOR_ROOT)")
    ap.add_argument("--out", type=Path, default=SPLIT_OUT)
    ap.add_argument("--frac", type=float, default=FRAC)
    ap.add_argument("--salt", default=SALT_HELDOUT,
                    help="held-out carve salt (seed replicates: "
                         f"{SALT_HELDOUT}-s<s>); default = the campaign's")
    ap.add_argument("--eval-salt", default=SALT_EVAL,
                    help="eval-subset salt (seed replicates: "
                         f"{SALT_EVAL}-s<s>); default = the campaign's")
    ap.add_argument("--pin-groups-from", type=Path, action="append",
                    default=[], metavar="PROBE.jsonl",
                    help="pin every carve group containing a probe item's "
                         "source == heldout_doc source_id into held-out "
                         "(repeatable; the artifact then carries a 'pins' "
                         "block)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build from a DIRTY generator tree anyway, "
                         "recording dirty=true + the tree-diff sha in the "
                         "artifact (production builds refuse otherwise)")
    args = ap.parse_args(argv)

    try:
        if args.code_root:
            set_code_root(args.code_root)
        generator = refuse_dirty_generator_tree(args.allow_dirty,
                                                tag="build-heldout-split")
        split = build_split(args.advice_final, args.stories_final,
                            frac=args.frac, salt=args.salt,
                            eval_salt=args.eval_salt,
                            pin_probe_files=args.pin_groups_from or None)
    except SplitError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    write_split(args.out, split, generator=generator, salt=args.salt,
                eval_salt=args.eval_salt, frac=args.frac)
    print_split_summary(split)
    print(f"wrote {args.out}")
    return 0


def write_split(out: Path, split: dict, *, generator: dict,
                salt: str | None = None, eval_salt: str | None = None,
                frac: float | None = None, extra: dict | None = None) -> Path:
    """Write the split artifact + its ``.provenance.json`` sidecar.

    Generator provenance rides the SIDECAR, never the split artifact
    itself: the artifact is deterministic BY DESIGN (a committed split
    must byte-match a fresh rebuild at any commit), while the sidecar
    records which tree built it — commit per repo, dirty flag, the
    tree-diff sha when --allow-dirty overrode the refusal — and the carve
    summary (:func:`split_provenance`). Returns the sidecar path."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(split, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    prov_path = out.with_suffix(".provenance.json")
    prov_path.write_text(
        json.dumps(split_provenance(split, artifact=out, generator=generator,
                                    salt=salt, eval_salt=eval_salt, frac=frac,
                                    extra=extra),
                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return prov_path


def print_split_summary(split: dict) -> None:
    for fam in ("advice", "stories"):
        acc = split["token_totals"][fam]
        pins = (split.get("pins") or {}).get(fam)
        pin_txt = (f"; pinned {pins['n_pinned_groups']} groups "
                   f"({pins['pinned_frac_of_family']:.4f} of family) + "
                   f"{pins['n_seed_drawn_groups']} seed-drawn "
                   f"({pins['seed_drawn_share_of_heldout']:.3f} of held-out)"
                   if pins else "")
        print(
            f"{fam}: {acc['heldout_groups']}/{acc['family_groups']} groups, "
            f"{acc['heldout_records']}/{acc['family_records']} records held out; "
            f"official {acc['heldout_official']:,}/{acc['family_official']:,} "
            f"({acc['heldout_frac_realized']:.4f}, limiting "
            f"{acc['heldout_limiting_tokenizer']}); train official "
            f"{acc['train_official']:,}{pin_txt}"
        )
    ev = split["eval_subset"]
    print(f"eval subset: {ev['n']} docs, family counts {ev['family_counts']}")


if __name__ == "__main__":
    raise SystemExit(main())
