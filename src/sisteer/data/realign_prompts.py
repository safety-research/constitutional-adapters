"""Canonical MWE item manifest + deterministic prompt splits (realign suite).

The crux is ONE canonical item manifest per behavior: every consumer
derives its items from the same canonical listing, and the three
partitions — ``extract`` (free-form extraction/training stems),
``calibration``, and ``id_test`` (in-distribution A/B eval) — are
carved from that single manifest by a deterministic hash on each item's
normalized question STEM. This fixes a known leakage bug: extraction
previously drew free-form MWE stems from the same pool the A/B eval
items came from, so the same question could train the vector and then
score it.

Two disjointness guarantees, asserted (never assumed) at build time and
re-asserted on every load from disk:

1. canonical ids never span two partitions (nor repeat within one);
2. normalized stem hashes never span two partitions — the stem is what
   extraction actually trains on, so stem-level overlap IS the leakage
   even when the lettered items differ.

Assignment hashes the stem with a behavior-INDEPENDENT salt, so items
sharing a stem co-locate in one partition even across behaviors, and
``balance_letters`` is applied strictly WITHIN each partition (on a
canonically ordered listing, so the subsample is reproducible).

Dataset reads are pinned by revision. ``mwe.load_mwe`` and
``prompts.load_hf_prompts`` don't expose their loaders' ``revision``
parameter and those modules belong to other workstreams, so thin pinned
mirrors live here (``load_mwe_pinned`` / ``load_alpaca_pinned``);
parsing/filtering logic is shared or replicated exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

from .mwe import _CHOICES_RE, ABItem, balance_letters, parse_mwe_lines

# -- pinned sources ---------------------------------------------------------------

REALIGN_BEHAVIORS: tuple[str, ...] = (
    "power-seeking-inclination",
    "survival-instinct",
)
MWE_SOURCE = "human_generated_evals"
# Anthropic/model-written-evals @ main, resolved 2026-08-14.
MWE_REVISION = "d533f626cc321c92175a58ee570aa3cdb87238d1"
ALPACA_DATASET = "tatsu-lab/alpaca"
# tatsu-lab/alpaca @ main, resolved 2026-08-14.
ALPACA_REVISION = "dce01c9b08f87459cf36a430d809084718273017"

PARTITIONS: tuple[str, str, str] = ("extract", "calibration", "id_test")
DEFAULT_MWE_FRACS: tuple[float, float, float] = (0.5, 0.25, 0.25)
# Deliberately behavior-independent: a stem shared across behaviors must
# land in the same partition in both, or it could train in one behavior
# and test in the other.
_MWE_SALT = "realign-mwe"

MWE_MANIFEST_NAME = "mwe_manifest.json"
SPLIT_MANIFEST_NAME = "split_manifest.json"


# -- hashing helpers ----------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _hash_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _unit_hash(key: str, salt: str) -> float:
    """Deterministic [0, 1) value for a key — prompts.split_prompts'
    scheme, widened from its single digest byte to 64 bits so realized
    fractions track the requested ones closely."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def mwe_stem(question: str) -> str:
    """The free-form question stem: the MWE question with its lettered
    choice block stripped — located with mwe._CHOICES_RE, the exact
    pattern load_mwe parses choices with — plus any trailing 'Choices:'
    label. Raises rather than returning a stem that still carries
    choice text (that text would leak eval structure into free-form
    extraction prompts)."""
    m = _CHOICES_RE.search(question)
    stem = question[: m.start()] if m else question
    stem = re.sub(r"choices\s*:?\s*$", "", stem.strip(), flags=re.IGNORECASE)
    stem = stem.strip()
    if not stem:
        raise ValueError(f"empty stem for MWE question {question!r}")
    if "(A)" in stem or "(B)" in stem:
        raise ValueError(f"stem still contains choice text: {stem!r}")
    return stem


def stem_hash(question: str) -> str:
    """Normalized-stem hash — the unit of the leakage guarantee."""
    return _hash_hex(_normalize(mwe_stem(question)))


def canonical_item_id(behavior: str, item: ABItem) -> str:
    """Stable content-derived id, invariant to file ordering AND to the
    letter arrangement: hashing the normalized stem plus the SORTED
    normalized choice texts means an A/B-swapped duplicate of an item
    maps to the same id (and is deduped from the manifest) instead of
    slipping into a different partition."""
    choices = sorted(_normalize(c or "") for c in (item.choice_a, item.choice_b))
    key = "\x1f".join([_normalize(mwe_stem(item.question)), *choices])
    return f"{behavior}:{_hash_hex(key)[:16]}"


# -- pinned loaders -----------------------------------------------------------------


def load_mwe_pinned(
    behavior: str,
    source: str = MWE_SOURCE,
    strict_ab: bool = True,
    revision: str = MWE_REVISION,
) -> list[ABItem]:
    """mwe.load_mwe at a pinned dataset revision (which load_mwe doesn't
    expose); parsing is shared via parse_mwe_lines. A failed download
    raises — never silently falls back to an unpinned read."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Anthropic/model-written-evals",
        repo_type="dataset",
        filename=f"advanced-ai-risk/{source}/{behavior}.jsonl",
        revision=revision,
    )
    with open(path) as f:
        return parse_mwe_lines(f.readlines(), strict_ab=strict_ab)


def load_alpaca_pinned(
    n: int = 2000,
    seed: int = 0,
    max_chars: int = 300,
    revision: str = ALPACA_REVISION,
) -> list[str]:
    """prompts.load_hf_prompts at a pinned dataset revision (which that
    loader doesn't expose), with its filtering replicated exactly:
    instruction-only rows (no `input` context), 10..max_chars long."""
    from datasets import load_dataset

    ds = load_dataset(ALPACA_DATASET, split="train", revision=revision)
    ds = ds.shuffle(seed=seed)
    prompts: list[str] = []
    for row in ds:
        instr = row.get("instruction") or row.get("prompt") or ""
        if row.get("input"):
            continue  # keep single-turn, context-free prompts
        if 10 <= len(instr) <= max_chars:
            prompts.append(instr.strip())
        if len(prompts) >= n:
            break
    return prompts


# -- the canonical manifest and its partitions ----------------------------------------


def mwe_canonical_manifest(
    behavior: str, seed: int = 0, *, items: list[ABItem] | None = None
) -> list[tuple[str, ABItem]]:
    """The ONE canonical (canonical_id, item) listing for a behavior.

    Membership and ids are content-derived: exact-content duplicates
    (letter arrangement included — see canonical_item_id) are dropped,
    then entries are ordered canonically. `seed` only shuffles the
    LISTING order (deterministically), for consumers that iterate the
    manifest directly; partition membership never depends on it,
    because mwe_partitions re-canonicalizes the order before hashing.

    `items` injects pre-parsed items (tests / offline callers); by
    default the pinned-revision loader is used and a failure to load
    raises rather than degrading to unpinned data.
    """
    if items is None:
        items = load_mwe_pinned(behavior, strict_ab=True)
    by_id: dict[str, ABItem] = {}
    for item in items:
        by_id.setdefault(canonical_item_id(behavior, item), item)
    manifest = sorted(by_id.items())
    random.Random(f"{_MWE_SALT}:{behavior}:{seed}").shuffle(manifest)
    return manifest


def _partition_of(question: str, fracs: tuple[float, float, float]) -> str:
    u = _unit_hash(stem_hash(question), _MWE_SALT)
    if u < fracs[0]:
        return "extract"
    if u < fracs[0] + fracs[1]:
        return "calibration"
    return "id_test"


def _check_fracs(fracs, n_expected: int) -> None:
    if len(fracs) != n_expected:
        raise ValueError(f"fracs must have {n_expected} entries, got {fracs!r}")
    if any(f < 0 for f in fracs) or abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"fracs must be non-negative and sum to 1, got {fracs!r}")


def mwe_partitions(
    behavior: str,
    fracs: tuple[float, float, float] = DEFAULT_MWE_FRACS,
    seed: int = 0,
    *,
    items: list[ABItem] | None = None,
) -> dict[str, list[tuple[str, ABItem]]]:
    """Three DISJOINT partitions {extract, calibration, id_test} of the
    canonical manifest.

    Assignment is a deterministic hash on each item's normalized stem,
    so membership depends on item content only — never on file order,
    and never on `seed`. balance_letters is applied WITHIN each
    partition (on the canonically ordered listing, so the subsample is
    reproducible); `seed` affects only that subsampling. Reseeding can
    therefore drop different items, but can never move an item between
    partitions. Disjointness — by canonical id AND by normalized stem
    hash — is asserted at the end, not assumed.
    """
    _check_fracs(fracs, len(PARTITIONS))
    manifest = mwe_canonical_manifest(behavior, seed=seed, items=items)
    assigned: dict[str, list[tuple[str, ABItem]]] = {p: [] for p in PARTITIONS}
    for cid, item in sorted(manifest, key=lambda e: e[0]):
        assigned[_partition_of(item.question, fracs)].append((cid, item))
    parts: dict[str, list[tuple[str, ABItem]]] = {}
    for name, entries in assigned.items():
        # balance_letters operates on bare items and preserves object
        # identity (it shuffles/slices the same objects), so an id() map
        # recovers each survivor's canonical id.
        cid_of = {id(item): cid for cid, item in entries}
        balanced = balance_letters([item for _, item in entries], seed=seed)
        parts[name] = [(cid_of[id(item)], item) for item in balanced]
    assert_partitions_disjoint(parts)
    return parts


def assert_partitions_disjoint(
    parts: dict[str, list[tuple[str, ABItem]]],
) -> None:
    """Raise (loudly) on any train/eval leakage between partitions: a
    canonical id that repeats anywhere, or a normalized stem hash that
    spans two partitions (same-stem items WITHIN one partition are
    allowed — they cannot leak)."""
    cid_where: dict[str, str] = {}
    stem_where: dict[str, str] = {}
    for name, entries in parts.items():
        for cid, item in entries:
            if cid in cid_where:
                kind = (
                    "spans partitions "
                    f"{cid_where[cid]!r} and {name!r}"
                    if cid_where[cid] != name
                    else f"repeats within partition {name!r}"
                )
                raise ValueError(f"MWE partition leakage: canonical id {cid} {kind}")
            cid_where[cid] = name
            sh = stem_hash(item.question)
            if sh in stem_where and stem_where[sh] != name:
                raise ValueError(
                    f"MWE partition leakage: stem of {cid} (hash {sh[:12]}...) "
                    f"spans partitions {stem_where[sh]!r} and {name!r}"
                )
            stem_where[sh] = name


# -- frozen on-disk manifest -----------------------------------------------------------


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Atomic publish (tmp + fsync + os.replace — the save_pairs
    pattern): consumers resume from these files by existence, so a
    crash mid-write must leave the old complete file or none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w") as f:
            json.dump(payload, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_mwe_manifest(
    out_dir: str | Path,
    behaviors: tuple[str, ...] = REALIGN_BEHAVIORS,
    fracs: tuple[float, float, float] = DEFAULT_MWE_FRACS,
    seed: int = 0,
    *,
    items_by_behavior: dict[str, list[ABItem]] | None = None,
) -> Path:
    """Freeze the partitions to <out_dir>/mwe_manifest.json so every
    consumer reads the SAME splits from disk. Disjointness is asserted
    per behavior (inside mwe_partitions) and again across the union of
    behaviors before anything is written."""
    payload: dict = {
        "schema": 1,
        "salt": _MWE_SALT,
        "behaviors": list(behaviors),
        "fracs": list(fracs),
        "seed": seed,
        "partitions": {},
    }
    combined: dict[str, list[tuple[str, ABItem]]] = {p: [] for p in PARTITIONS}
    for behavior in behaviors:
        items = None if items_by_behavior is None else items_by_behavior[behavior]
        parts = mwe_partitions(behavior, fracs=fracs, seed=seed, items=items)
        payload["partitions"][behavior] = {
            name: [{"canonical_id": cid, **item.__dict__} for cid, item in entries]
            for name, entries in parts.items()
        }
        for name in PARTITIONS:
            combined[name].extend(parts[name])
    assert_partitions_disjoint(combined)
    path = Path(out_dir) / MWE_MANIFEST_NAME
    _atomic_write_json(payload, path)
    return path


def load_mwe_manifest(
    out_dir: str | Path,
) -> dict[str, dict[str, list[tuple[str, ABItem]]]]:
    """Read the frozen partitions back: {behavior: {partition: [(cid,
    item), ...]}}. Disjointness is re-asserted on EVERY load — a
    manifest that was edited, merged, or corrupted on disk must fail
    loudly here, not silently leak downstream."""
    path = Path(out_dir) / MWE_MANIFEST_NAME
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1:
        raise ValueError(f"unknown mwe manifest schema in {path}: {payload.get('schema')!r}")
    out: dict[str, dict[str, list[tuple[str, ABItem]]]] = {}
    combined: dict[str, list[tuple[str, ABItem]]] = {p: [] for p in PARTITIONS}
    for behavior, parts in payload["partitions"].items():
        out[behavior] = {}
        for name in PARTITIONS:
            entries = [
                (
                    row["canonical_id"],
                    ABItem(**{k: v for k, v in row.items() if k != "canonical_id"}),
                )
                for row in parts[name]
            ]
            out[behavior][name] = entries
            combined[name].extend(entries)
    assert_partitions_disjoint(combined)
    return out


# -- extraction pool --------------------------------------------------------------------


def extraction_pool(
    n_mwe: int = 300,
    n_alpaca: int = 300,
    seed: int = 0,
    *,
    behaviors: tuple[str, ...] = REALIGN_BEHAVIORS,
    manifest_dir: str | Path | None = None,
    partitions_by_behavior: dict[str, dict[str, list[tuple[str, ABItem]]]] | None = None,
    alpaca_prompts: list[str] | None = None,
) -> list[str]:
    """Free-form extraction/training prompts: MWE question STEMS drawn
    from the `extract` partition ONLY — never calibration or id_test;
    that separation is the leakage fix this module exists for — mixed
    with pinned-revision alpaca benign prompts.

    Partition source, in priority order: `partitions_by_behavior`
    (injected — tests/offline), `manifest_dir` (the frozen on-disk
    manifest; the intended production path so every consumer sees the
    same splits), else freshly computed pinned-revision partitions at
    the default fracs. If the extract partitions hold fewer than n_mwe
    distinct stems, all of them are used (the pool is visibly shorter;
    nothing is silently substituted from other partitions).
    """
    if partitions_by_behavior is None:
        if manifest_dir is not None:
            partitions_by_behavior = load_mwe_manifest(manifest_dir)
        else:
            partitions_by_behavior = {
                b: mwe_partitions(b, seed=seed) for b in behaviors
            }
    stems: list[str] = []
    for behavior in sorted(partitions_by_behavior):
        for _cid, item in partitions_by_behavior[behavior]["extract"]:
            stems.append(mwe_stem(item.question))  # raises if choice text survives
    stems = sorted(dict.fromkeys(stems))  # dedupe; canonical order pre-shuffle
    rng = random.Random(f"realign-extraction-pool:{seed}")
    rng.shuffle(stems)
    stems = stems[:n_mwe]
    if alpaca_prompts is None:
        alpaca_prompts = load_alpaca_pinned(n=n_alpaca, seed=seed)
    pool = stems + list(alpaca_prompts)[:n_alpaca]
    rng.shuffle(pool)
    return pool


# -- three-way free-form prompt split ------------------------------------------------------


def three_way_split(
    prompts: list[str],
    fracs: tuple[float, float, float] = (0.6, 0.2, 0.2),
    salt: str = "realign",
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic hash-based (train, calib, held) split — the repo's
    prompts.split_prompts generalized from two ways to three. Each
    prompt is assigned independently by its salted hash, so membership
    is stable across runs and across prompt-list orderings, and a
    duplicated prompt always travels to the same bucket."""
    _check_fracs(fracs, 3)
    train: list[str] = []
    calib: list[str] = []
    held: list[str] = []
    edges = (fracs[0], fracs[0] + fracs[1])
    for p in prompts:
        u = _unit_hash(p, salt)
        (train if u < edges[0] else calib if u < edges[1] else held).append(p)
    return train, calib, held


def write_split_manifest(
    out_dir: str | Path,
    train: list[str],
    calib: list[str],
    held: list[str],
    *,
    salt: str = "realign",
    fracs: tuple[float, float, float] | None = None,
) -> Path:
    """Freeze a three_way_split to <out_dir>/split_manifest.json (atomic
    publish) so every consumer reads the same frozen split. Disjointness
    across the three lists is asserted before writing."""
    splits = {"train": list(train), "calib": list(calib), "held": list(held)}
    _assert_splits_disjoint(splits)
    payload = {
        "schema": 1,
        "salt": salt,
        "fracs": list(fracs) if fracs is not None else None,
        "splits": splits,
    }
    path = Path(out_dir) / SPLIT_MANIFEST_NAME
    _atomic_write_json(payload, path)
    return path


def load_split_manifest(
    out_dir: str | Path,
) -> tuple[list[str], list[str], list[str]]:
    """Read a frozen split back as (train, calib, held), re-asserting
    disjointness on every load — tampered or hand-merged manifests fail
    loudly rather than silently leaking."""
    path = Path(out_dir) / SPLIT_MANIFEST_NAME
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1:
        raise ValueError(f"unknown split manifest schema in {path}: {payload.get('schema')!r}")
    splits = payload["splits"]
    _assert_splits_disjoint(splits)
    return splits["train"], splits["calib"], splits["held"]


def _assert_splits_disjoint(splits: dict[str, list[str]]) -> None:
    """Raise on a prompt (normalized) appearing in two different splits;
    duplicates WITHIN one split are allowed (they cannot leak)."""
    where: dict[str, str] = {}
    for name, prompts in splits.items():
        for p in prompts:
            key = _hash_hex(_normalize(p))
            if key in where and where[key] != name:
                raise ValueError(
                    f"prompt split leakage: a prompt appears in both "
                    f"{where[key]!r} and {name!r}: {p[:80]!r}"
                )
            where[key] = name
