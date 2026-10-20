"""Pinned, offline corpus loading + deterministic slicing.

Every hub read goes through huggingface_hub at a PINNED full-SHA
revision (``snapshot_download`` for pattern-pinned corpora,
``hf_hub_download`` for single files) + a DIRECT pyarrow/pandas read —
never ``load_dataset`` (the S2-proven pattern: script datasets are
refused by the locked ``datasets>=5``, and HfFileSystem paths break
under HF_HUB_OFFLINE even when cached).

Pins live in ``configs/capeval/pins.yaml``, WRITTEN BY
``scripts/capeval_fetch_data.py`` at Phase 0 — this module only
READS them, and refuses loudly when the file is absent (authoritative-
source rule: no silent fallback to unpinned revisions).

pins.yaml shape (matching what the fetch script actually writes; its
docstring documents the freeze semantics)::

    datasets:
      <name>:                     # wikitext103, stack_py, ultrachat,
        repo_id: Salesforce/wikitext        #   canary_prompts, alpaca, …
        revision: <full 40-hex commit sha>  # REQUIRED, full length
        patterns: [<glob>, ...] | null      # snapshot allow_patterns;
                                            #   null = whole repo
        verify: parquet | json | jsonl | jsonl_gz | csv   # read format
        bytes: <int>              # fetch-script bookkeeping (not read)
        files: <int>              # fetch-script bookkeeping (not read:
                                  #   its counting convention is the
                                  #   fetch script's own)
    evalplus: {...}               # not read here
    models: {...}
    resolved_at: "..."

The content COLUMN for each named corpus is this module's knowledge
(wikitext103→text, stack_py→content, ultrachat→messages,
canary_prompts/alpaca→instruction), refused loudly when absent from the
read frame. Files resolve through ``snapshot_download`` at the pinned
revision (offline-safe once cached — never ``load_dataset``), then
direct pyarrow/pandas reads.

Unknown names, short revisions, and missing keys are refusals. Slices
are deterministic (sort + seed-0 shuffle + take-N, or an order-preserving
prefix for contiguous corpora) and every slice's RESOLVED spec — doc ids
included — is content-hashed into ``CorpusSlice.slice_hash``, which feeds
the Tier-0 cache digests (§4.3) and ``harness_config_hash``.

Windowing/tokenization is deliberately NOT here: tier0.py owns
token budgets (1024-token windows, assistant masking); corpora.py hands
it deterministic document/conversation/prompt slices.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from sisteer.capeval.manifest import (
    CorpusRef,
    _is_full_sha,
    canonical_json,
    sha256_hex16,
)

PINS_ENV_VAR = "CAPEVAL_PINS"
PINS_RELPATH = Path("configs") / "capeval" / "pins.yaml"

# Default slice parameters. Character budgets are the
# document-level stand-ins for the token budgets tier0.yaml owns (~4-5
# chars/token on English text/code); tier0.py trims to exact token
# budgets downstream.
WIKITEXT_CHAR_BUDGET = 1_500_000  # ~250k tokens of 1024-token windows
STACK_PY_CHAR_BUDGET = 600_000  # ~100k tokens
ULTRACHAT_MAX_CONVERSATIONS = 800  # >= the ~100k-assistant-token budget
CANARY_N_PROMPTS = 300
COMPLIANCE_POOL_N = 600  # S2 convention: prompts.n=600, holdout_frac=0.3
COMPLIANCE_HOLDOUT_FRAC = 0.3
COMPLIANCE_N_PROMPTS = 100


class CorpusError(ValueError):
    """A corpus/pins refusal (loud, never a silent fallback)."""


@dataclass(frozen=True)
class CorpusSlice:
    """One resolved, deterministic corpus slice.

    ``items`` are raw documents (kind="text"), conversations — lists of
    {role, content} dicts — (kind="chat"), or prompt strings
    (kind="prompts"). ``spec`` is the fully-resolved slice spec;
    ``slice_hash`` covers spec + item ids + item CONTENT, so any upstream
    drift (a re-uploaded file at the same revision path, a changed
    filter) changes the hash and breaks cache/​config-hash reuse (F8).
    """

    name: str
    kind: str  # "text" | "chat" | "prompts"
    item_ids: Tuple[str, ...]
    items: Tuple[Any, ...]
    spec: Dict[str, Any]
    slice_hash: str

    def __len__(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------
# pins.yaml
# --------------------------------------------------------------------------


def find_pins_path() -> Path:
    """$CAPEVAL_PINS if set, else <repo root>/configs/capeval/pins.yaml
    (repo root located from this file's src-layout position)."""
    env = os.environ.get(PINS_ENV_VAR)
    if env:
        return Path(env)
    # src/sisteer/capeval/corpora.py -> parents[3] == repo root (editable
    # install; a non-editable install must set $CAPEVAL_PINS).
    return Path(__file__).resolve().parents[3] / PINS_RELPATH


_PIN_REQUIRED_KEYS = ("repo_id", "revision", "verify")
_PIN_ALLOWED_KEYS = _PIN_REQUIRED_KEYS + ("patterns", "bytes", "files")
# "csv" added for the judged-tier or_bench pin: load_pins validates EVERY
# entry, so the new kind must be legal here or one csv pin in pins.yaml
# breaks every pins consumer.
_PIN_FORMATS = ("parquet", "json", "jsonl", "jsonl_gz", "csv")
# file suffixes each verify format reads (also filters snapshot noise —
# READMEs, .gitattributes — when patterns is null)
_VERIFY_SUFFIXES = {
    "parquet": (".parquet",),
    "json": (".json",),
    "jsonl": (".jsonl",),
    "jsonl_gz": (".json.gz", ".jsonl.gz"),
    "csv": (".csv",),
}


def load_pins(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Read + validate pins.yaml. Absent file is the loud Phase-0 error."""
    path = Path(path) if path is not None else find_pins_path()
    if not path.is_file():
        raise CorpusError(
            f"pins file not found at {path} — run `python "
            "scripts/capeval_fetch_data.py` first (Phase 0 resolves and "
            "pins every dataset revision; corpora.py never guesses one). "
            f"Set ${PINS_ENV_VAR} to point at a pins.yaml elsewhere.")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping) or not isinstance(raw.get("datasets"), Mapping):
        raise CorpusError(f"{path}: expected a mapping with a 'datasets' key")
    for name, entry in raw["datasets"].items():
        where = f"{path}:datasets.{name}"
        if not isinstance(entry, Mapping):
            raise CorpusError(f"{where}: not a mapping")
        unknown = sorted(set(entry) - set(_PIN_ALLOWED_KEYS))
        if unknown:
            raise CorpusError(f"{where}: unknown key(s) {unknown}")
        missing = [k for k in _PIN_REQUIRED_KEYS if k not in entry]
        if missing:
            raise CorpusError(f"{where}: missing key(s) {missing}")
        rev = entry["revision"]
        if not _is_full_sha(rev):
            raise CorpusError(f"{where}: revision must be a full 40-hex "
                              f"commit sha, got {rev!r}")
        if entry["verify"] not in _PIN_FORMATS:
            raise CorpusError(f"{where}: verify {entry['verify']!r} not in "
                              f"{_PIN_FORMATS}")
        patterns = entry.get("patterns")
        if patterns is not None and (
                not isinstance(patterns, list) or not patterns
                or not all(isinstance(p, str) and p for p in patterns)):
            raise CorpusError(f"{where}: patterns must be null or a "
                              "non-empty list of globs")
    return dict(raw)


def pin_entry(pins: Mapping[str, Any], name: str) -> Dict[str, Any]:
    datasets = pins["datasets"]
    if name not in datasets:
        raise CorpusError(f"no pin named {name!r} in pins.yaml — the fetch "
                          f"script pins: {sorted(datasets)}")
    return dict(datasets[name])


# --------------------------------------------------------------------------
# reading (hf_hub_download + direct pyarrow/pandas ONLY)
# --------------------------------------------------------------------------


def _read_file(path: Union[str, Path], fmt: str) -> "pandas.DataFrame":  # noqa: F821
    import pandas as pd

    path = Path(path)
    if fmt == "parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pandas()
    if fmt == "json":
        return pd.read_json(path)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True)
    if fmt == "jsonl_gz":
        return pd.read_json(path, lines=True, compression="gzip")
    if fmt == "csv":
        return pd.read_csv(path)
    raise CorpusError(f"unknown format {fmt!r} for {path}")


def pinned_snapshot_files(entry: Mapping[str, Any], name: str) -> List[Path]:
    """Resolve a pins entry to concrete local files: snapshot_download at
    the pinned full-SHA revision with ``local_files_only=True`` — this is
    an EVALUATION-TIME loader, so an unfetched cache must refuse
    immediately rather than reach for the network (with HF_HUB_OFFLINE
    accidentally unset, a Slurm job would otherwise hang on hub metadata
    instead of failing loudly; network fetching belongs exclusively to
    scripts/capeval_fetch_data.py). Then match the entry's glob patterns
    (or, when patterns is null, every file whose suffix fits the verify
    format). Zero matches is a refusal — an unfetched corpus must fail
    here, not read an empty frame. Sorted for determinism."""
    import fnmatch

    from huggingface_hub import snapshot_download

    try:
        snap = Path(snapshot_download(
            entry["repo_id"], repo_type="dataset", revision=entry["revision"],
            allow_patterns=entry.get("patterns") or None,
            local_files_only=True))
    except Exception as e:
        raise CorpusError(
            f"corpus {name!r}: {entry['repo_id']} at "
            f"{entry['revision'][:12]} is not in the local HF cache "
            f"({type(e).__name__}: {e}) — evaluation never touches the "
            "network; run scripts/capeval_fetch_data.py first") from e
    suffixes = _VERIFY_SUFFIXES[entry["verify"]]
    patterns = entry.get("patterns")
    out = []
    for p in sorted(snap.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(snap).as_posix()
        if not any(rel.endswith(s) for s in suffixes):
            continue
        if patterns and not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            continue
        out.append(p)
    if not out:
        raise CorpusError(
            f"corpus {name!r}: no files matching patterns={patterns!r} / "
            f"verify={entry['verify']!r} in the {entry['repo_id']} snapshot "
            f"at {entry['revision'][:12]} — run capeval_fetch_data first")
    return out


def _fetch_pinned_frames(entry: Mapping[str, Any],
                         name: str) -> "pandas.DataFrame":  # noqa: F821
    """Read every pinned file (direct pyarrow/pandas — never
    load_dataset) and concatenate."""
    import pandas as pd

    frames = [_read_file(p, entry["verify"])
              for p in pinned_snapshot_files(entry, name)]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _column(df, field: str, where: str) -> List[Any]:
    if field not in df.columns:
        raise CorpusError(f"{where}: column {field!r} not in {list(df.columns)} "
                          "— refusing to guess a schema")
    return list(df[field])


# --------------------------------------------------------------------------
# deterministic slicing + hashing
# --------------------------------------------------------------------------


def deterministic_order(items: Sequence[str], seed: int = 0) -> List[int]:
    """Indices of ``items`` in the canonical order: sort by content (ties
    by original index — duplicates keep a stable order), then a seeded
    permutation. Independent of upstream row order."""
    import numpy as np

    by_content = sorted(range(len(items)),
                        key=lambda i: (_content_key(items[i]), i))
    perm = np.random.default_rng(seed).permutation(len(by_content))
    return [by_content[int(p)] for p in perm]


def _content_key(item: Any) -> str:
    return item if isinstance(item, str) else canonical_json(item)


def _item_id(item: Any) -> str:
    return sha256_hex16(_content_key(item).encode("utf-8"))


def _make_slice(name: str, kind: str, items: Sequence[Any],
                spec: Mapping[str, Any]) -> CorpusSlice:
    ids = tuple(_item_id(it) for it in items)
    dup = len(ids) != len(set(ids))
    if dup:
        raise CorpusError(f"corpus {name!r}: duplicate items after slicing — "
                          "item_ids must be unique for per-item pairing "
                          "(dedupe upstream)")
    resolved = dict(spec)
    resolved["item_ids"] = list(ids)
    content_digest = sha256_hex16(
        canonical_json([_content_key(it) for it in items]).encode("utf-8"))
    resolved["content_digest"] = content_digest
    slice_hash = sha256_hex16(canonical_json(
        {"name": name, "kind": kind, "spec": resolved}).encode("utf-8"))
    return CorpusSlice(name=name, kind=kind, item_ids=ids, items=tuple(items),
                       spec=resolved, slice_hash=slice_hash)


def _dedupe_keep_first(items: Sequence[Any]) -> List[Any]:
    seen, out = set(), []
    for it in items:
        k = _content_key(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _pin_spec(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The hash-bearing part of a pins entry (bytes/files bookkeeping
    excluded — they are fetch-script metadata, not content identity)."""
    return {"repo_id": entry["repo_id"], "revision": entry["revision"],
            "patterns": entry.get("patterns"), "verify": entry["verify"]}


# --------------------------------------------------------------------------
# the pinned tier-0 / target corpora
# --------------------------------------------------------------------------


def wikitext103(pins: Optional[Mapping[str, Any]] = None,
                char_budget: int = WIKITEXT_CHAR_BUDGET) -> CorpusSlice:
    """Contiguous stored-order prefix of the wikitext-103 test split.
    Deliberately NOT shuffled: rows are lines of a continuous document
    stream, and shuffling would destroy the context the NLL windows
    score. Determinism comes from the pinned revision + prefix rule."""
    pins = pins if pins is not None else load_pins()
    entry = pin_entry(pins, "wikitext103")
    texts = _column(_fetch_pinned_frames(entry, "wikitext103"), "text",
                    "wikitext103")
    taken, total = [], 0
    for t in texts:
        taken.append(t)
        total += len(t)
        if total >= char_budget:
            break
    # one contiguous text item: the windowing unit is tier0.py's (F14 items
    # are 1024-token windows derived downstream)
    doc = "".join(taken)
    spec = {"pin": _pin_spec(entry),
            "mode": "contiguous_prefix", "char_budget": char_budget,
            "n_source_rows": len(taken)}
    return _make_slice("wikitext103", "text", [doc], spec)


def stack_py(pins: Optional[Mapping[str, Any]] = None,
             char_budget: int = STACK_PY_CHAR_BUDGET,
             seed: int = 0) -> CorpusSlice:
    """The pinned python-code corpus (codeparrot/codeparrot-clean-valid;
    the-stack-smol is gated, and the pins file is authoritative). Whole
    files are self-contained docs — sort + seed-0 shuffle, take files
    until the char budget."""
    pins = pins if pins is not None else load_pins()
    entry = pin_entry(pins, "stack_py")
    docs = _dedupe_keep_first(
        [t for t in _column(_fetch_pinned_frames(entry, "stack_py"),
                            "content", "stack_py") if t and t.strip()])
    order = deterministic_order(docs, seed=seed)
    taken, total = [], 0
    for i in order:
        taken.append(docs[i])
        total += len(docs[i])
        if total >= char_budget:
            break
    spec = {"pin": _pin_spec(entry),
            "mode": "sorted_shuffled_take", "seed": seed,
            "char_budget": char_budget}
    return _make_slice("stack_py", "text", taken, spec)


def ultrachat(pins: Optional[Mapping[str, Any]] = None,
              max_conversations: int = ULTRACHAT_MAX_CONVERSATIONS,
              seed: int = 0) -> CorpusSlice:
    """ultrachat_200k test_sft conversations, sort + seed-0 shuffle +
    take-N. The ≥100k-assistant-token budget is applied downstream by
    tier0.py (token counting needs the tokenizer); ``max_conversations``
    just has to comfortably exceed it."""
    pins = pins if pins is not None else load_pins()
    entry = pin_entry(pins, "ultrachat")
    raw = _column(_fetch_pinned_frames(entry, "ultrachat"), "messages",
                  "ultrachat")
    convs = []
    for msgs in raw:
        conv = [{"role": str(m["role"]), "content": str(m["content"])}
                for m in msgs]
        if conv and any(m["role"] == "assistant" for m in conv):
            convs.append(conv)
    convs = _dedupe_keep_first(convs)
    order = deterministic_order(convs, seed=seed)
    taken = [convs[i] for i in order[:max_conversations]]
    spec = {"pin": _pin_spec(entry),
            "mode": "sorted_shuffled_take", "seed": seed,
            "max_conversations": max_conversations}
    return _make_slice("ultrachat", "chat", taken, spec)


def canary_prompts(pins: Optional[Mapping[str, Any]] = None,
                   n: int = CANARY_N_PROMPTS, seed: int = 0) -> CorpusSlice:
    """The §4.4 canary set: deterministic n-subset (sort + seed-0
    shuffle) of the pinned alpaca_eval instructions."""
    pins = pins if pins is not None else load_pins()
    entry = pin_entry(pins, "canary_prompts")
    prompts = _dedupe_keep_first(
        [p for p in _column(_fetch_pinned_frames(entry, "canary_prompts"),
                            "instruction", "canary_prompts")
         if p and p.strip()])
    if len(prompts) < n:
        raise CorpusError(f"canary_prompts: only {len(prompts)} distinct "
                          f"prompts at the pin, need {n}")
    order = deterministic_order(prompts, seed=seed)
    taken = [prompts[i] for i in order[:n]]
    spec = {"pin": _pin_spec(entry),
            "mode": "sorted_shuffled_take", "seed": seed, "n": n}
    return _make_slice("canary_prompts", "prompts", taken, spec)


def alpaca_holdout_prompts(pins: Optional[Mapping[str, Any]] = None,
                           n: int = COMPLIANCE_N_PROMPTS,
                           pool_n: int = COMPLIANCE_POOL_N,
                           holdout_frac: float = COMPLIANCE_HOLDOUT_FRAC,
                           seed: int = 0) -> CorpusSlice:
    """The sisteer_compliance prompt set (§9): the pinned tatsu-lab/alpaca
    dataset the S2 campaign drew from, same filter (input-free, 10-300
    chars) and the same hash-based holdout split (data.prompts.
    split_prompts — order-independent), same pool/holdout constants as
    configs/stage2_llama8b.yaml (n=600, holdout_frac=0.3, eval=first 100).

    NOT bit-identical to the S2 eval set: S2's pool order came from
    datasets.shuffle(seed=0), which is not reproducible without
    load_dataset (forbidden here); this loader's pool order is the
    canonical sort+seed-0 shuffle. Same pinned data, same selection
    logic, deterministic — and compliance is always a paired arm-vs-base
    delta on the SAME prompts, so cross-campaign prompt identity is not
    load-bearing (the parity probes carry that check)."""
    from sisteer.data.prompts import split_prompts

    pins = pins if pins is not None else load_pins()
    entry = pin_entry(pins, "alpaca")
    df = _fetch_pinned_frames(entry, "alpaca")
    instructions = _column(df, "instruction", "alpaca")
    inputs = list(df["input"]) if "input" in df.columns else [""] * len(instructions)
    pool = _dedupe_keep_first([
        instr.strip() for instr, inp in zip(instructions, inputs)
        if not (inp and str(inp).strip()) and 10 <= len(str(instr)) <= 300
    ])
    order = deterministic_order(pool, seed=seed)
    pool = [pool[i] for i in order[:pool_n]]
    _, heldout = split_prompts(pool, holdout_frac=holdout_frac)
    if len(heldout) < n:
        raise CorpusError(f"alpaca_holdout_prompts: holdout has only "
                          f"{len(heldout)} prompts, need {n} — raise pool_n")
    taken = heldout[:n]
    spec = {"pin": _pin_spec(entry),
            "mode": "filtered_pool_hash_holdout", "seed": seed,
            "pool_n": pool_n, "holdout_frac": holdout_frac, "n": n}
    return _make_slice("alpaca_holdout", "prompts", taken, spec)


# --------------------------------------------------------------------------
# CorpusRef resolution (target.relevant_corpus, §4.5 / F20)
# --------------------------------------------------------------------------

_SLICE_KEYS_PATH = frozenset({"first", "field"})
_SLICE_KEYS_HUB = frozenset({"first", "field", "file", "format"})
#: local-file suffix -> read format (resolve_corpus_ref, both modes)
_SUFFIX_FORMATS = {".jsonl": "jsonl", ".json": "json", ".parquet": "parquet"}


def resolve_corpus_ref(ref: CorpusRef, name: str = "relevant_corpus") -> CorpusSlice:
    """Resolve a manifest CorpusRef + slice spec into a CorpusSlice.

    Path mode reads a local file (jsonl/json/parquet by suffix); hub mode
    reads ``repo_id@revision`` via hf_hub_download, with the repo
    filename in ``slice.file`` (and optional ``slice.format``). The v0
    slice spec is ``{first: N, field: <column>}`` — an order-preserving
    prefix ("first 200 prompts", §4.1); unknown slice keys are refusals.
    """
    slice_spec = dict(ref.slice)
    if ref.path is not None:
        unknown = sorted(set(slice_spec) - _SLICE_KEYS_PATH)
        if unknown:
            raise CorpusError(f"{name}: unknown slice key(s) {unknown} "
                              f"(path mode allows {sorted(_SLICE_KEYS_PATH)})")
        # a site placeholder ($SHARED/...) expands through sisteer.site; an unset one refuses by name
        from sisteer import site as _site
        try:
            path = _site.expand_path(ref.path, key=f"{name}.path")
        except _site.PlaceholderError as exc:
            raise CorpusError(f"{name}: {exc}") from None
        if not path.is_file():
            raise CorpusError(f"{name}: corpus file does not exist: {path}")
        fmt = _SUFFIX_FORMATS.get(path.suffix)
        if fmt is None:
            raise CorpusError(f"{name}: unsupported corpus suffix "
                              f"{path.suffix!r} on {path}")
        df = _read_file(path, fmt)
        source = {"path": str(path), "format": fmt}
    else:
        unknown = sorted(set(slice_spec) - _SLICE_KEYS_HUB)
        if unknown:
            raise CorpusError(f"{name}: unknown slice key(s) {unknown} "
                              f"(hub mode allows {sorted(_SLICE_KEYS_HUB)})")
        if "file" not in slice_spec:
            raise CorpusError(f"{name}: hub mode (repo_id) requires "
                              "slice.file naming the repo filename")
        fmt = (slice_spec.get("format")
               or _SUFFIX_FORMATS.get(Path(slice_spec["file"]).suffix))
        if fmt is None:
            raise CorpusError(f"{name}: cannot infer format of "
                              f"{slice_spec['file']!r}; set slice.format")
        from huggingface_hub import hf_hub_download
        try:
            # evaluation-time loader: cache-only, never the network (r1 #11)
            local = hf_hub_download(ref.repo_id, slice_spec["file"],
                                    repo_type="dataset",
                                    revision=ref.revision,
                                    local_files_only=True)
        except Exception as e:
            raise CorpusError(
                f"{name}: {ref.repo_id}@{str(ref.revision)[:12]} file "
                f"{slice_spec['file']!r} is not in the local HF cache "
                f"({type(e).__name__}: {e}) — evaluation never touches the "
                "network; run scripts/capeval_fetch_data.py first") from e
        df = _read_file(local, fmt)
        source = {"repo_id": ref.repo_id, "revision": ref.revision,
                  "file": slice_spec["file"], "format": fmt}

    for req in ("first", "field"):
        if req not in slice_spec:
            raise CorpusError(f"{name}: slice spec missing {req!r}")
    n, field = slice_spec["first"], slice_spec["field"]
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise CorpusError(f"{name}: slice.first must be a positive int")
    raw_values = _column(df, field, name)
    if len(raw_values) < n:
        raise CorpusError(f"{name}: asked for first {n} of {field!r} but the "
                          f"corpus has only {len(raw_values)} rows")
    # every SELECTED value must already be a string: str()-coercing a
    # pandas null/container would fabricate a plausible prompt like "nan"
    # or "['x']" and hash it into the slice — a malformed corpus row must
    # refuse, not localize on garbage.
    values: List[str] = []
    for i, v in enumerate(raw_values[:n]):
        if not isinstance(v, str):
            raise CorpusError(
                f"{name}: row {i} of {field!r} is not a string "
                f"({type(v).__name__}: {v!r}) — refusing to coerce; fix the "
                "corpus or point slice.field at a string column")
        values.append(v)
    taken = _dedupe_keep_first(values)
    if len(taken) < n:
        raise CorpusError(f"{name}: first {n} rows of {field!r} contain "
                          f"duplicates ({n - len(taken)}) — per-item pairing "
                          "needs distinct items; fix the corpus or the slice")
    spec = {"source": source, "mode": "first", "n": n, "field": field}
    return _make_slice(name, "prompts", taken, spec)
