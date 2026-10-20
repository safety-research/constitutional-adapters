"""Neutral long-context PADDING for the STATIC single-shot families.

Inserts a MEASURED number of neutral filler tokens (the PG-19 cache the
long-context suite already uses) into the user turn, before or after the
family's transformed prompt. A prompted cell carries its instruction as the
authoritative SYSTEM message, so "before" places the filler between that
instruction and the payload; a compiled cell has no instruction text at all and
sees the same filler as a long prefix. That asymmetry IS the question the knob
exists to ask, and it needs no per-arm special casing here.

OFF unless a caller passes a :class:`PadPlan`: with ``padding=None`` every byte
``static_common.run_static`` writes is exactly as before.

METHOD (the approach the earlier long-context suite proved, kept here without
torch): the filler is a slice of ONE fixed token stream, and the slice length is adjusted
iteratively until the measured token delta lands within a tolerance of the
target. Growing the target therefore grows the padding while the payload stays
byte-identical -- the same-sample-grown-padding design of 2402.14848.

STORE IDENTITY. Padding is part of what a store MEASURES, so one store holds
one padding contract: :func:`assert_static_padding_consistent` refuses to
append padded rows onto an unpadded store, unpadded onto padded, or rows under
a different block, the same rule ``static_common`` already applies to the
engine and the decoder. A padded run needs its own ``private_run``.

WHAT A ROW RECORDS. The row's ``variant`` / ``attack_prompt`` stay the
UNPADDED transform output -- one meaning for those fields across padded and
unpadded stores, and a store that does not grow by the padding width on every
row. What was actually sent is pinned by ``padded_prompt_sha256`` inside the
row's ``padding`` block, and is reproducible from the block alone (filler
sha256 + offset + count + placement + separator).

COST. Two things are cached. Neither changes a byte of what is sent, and
neither weakens what a row records -- ``pad_tokens_actual`` stays a measurement:

* THE FILLER'S TOKEN STREAM (the ids of the whole corpus under the tokenizer:
  ~635k tokens, ~7 s of CPU per process on the pinned Qwen3.5 tokenizer, paid
  by every driver of a wave) is read from a PRE-TOKENIZED SIDECAR beside the
  corpus file -- ``<corpus stem>.ids.<tokenizer sha256[:16]>.json`` -- when one
  exists for this exact corpus TEXT and this exact tokenizer DEFINITION; else
  the corpus is tokenized in-process, as before, and the sidecar is written for
  the next process (best-effort: an unwritable directory costs a warning, never
  the run). The sidecar is keyed by the sha256 of the corpus text and the
  sha256 of the tokenizer's serialized definition (vocab, merges, normalizer,
  pre-tokenizer, added tokens -- everything that decides the ids); a mismatch on
  either REBUILDS, never reuses. The block's ``filler_sha256`` keeps identifying
  the CORPUS file; the sidecar is a cache and enters no row. Build it ahead of a
  wave with ``scripts/fetch_stage2_data.py --only pg19-filler
  --sidecar-tokenizer MODEL --sidecar-revision REV`` (:func:`ensure_filler_ids`).

* THE DECODED SLICE and its exact token count are cached per (offset, take) --
  in ``mode="fixed"`` every item of a level shares one -- so per item only the
  payload and the filler EDGE beside it are tokenized. :meth:`PadPlan.apply`
  states the exactness guarantee and the full-measurement sample that checks it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

#: the row/manifest block's schema label
SCHEMA = "static_padding/1"
#: the pre-tokenized filler sidecar's schema label
IDS_SCHEMA = "static_padding_ids/1"
#: what joins the filler to the payload (recorded, so a reader never guesses)
SEPARATOR = "\n\n"
#: "before" = between the system instruction and the payload; "after" = behind it
PLACEMENTS = ("before", "after")
#: "fixed" = the same filler prefix for every item (one text, grown);
#: "per_goal" = a deterministic per-goal offset into the stream
MODES = ("fixed", "per_goal")
DEFAULT_TOLERANCE = 0.02
DEFAULT_MAX_ITERS = 8
#: headroom the window preflight leaves for the payload + chat template
DEFAULT_RESERVE_TOKENS = 2048
#: chars of the filler's EDGE (the side touching the payload) re-tokenized per
#: item together with the separator and the payload -- every seam merge lies
#: inside this text, so it is measured, not assumed (see PadPlan.apply)
SEAM_WINDOW_CHARS = 1024
#: every Nth item on a cached slice is ALSO tokenized in full and compared; the
#: first item of every slice always is. 1 = every item measured in full.
DEFAULT_VERIFY_EVERY = 16
#: decoded slices kept per plan (one per (offset, take); per_goal cycles)
_SLICE_CACHE_SIZE = 8
#: the keys of the STORE-IDENTITY block (what every row of one store shares);
#: the per-item keys ride beside them in the row and are NOT compared
BLOCK_KEYS = ("schema", "tokens", "placement", "mode", "source",
              "filler_sha256", "separator", "tolerance")
#: env override read by :func:`resolve_filler_path`
FILLER_ENV = "SISTEER_PG19_FILLER"
#: the repo-relative default the Phase-0 fetch writes
REPO_FILLER_REL = "data/pg19_filler.json"


class StaticPaddingError(RuntimeError):
    """A padding request that cannot be honoured EXACTLY -- a missing or short
    filler, an unreachable target, a malformed spec, a measurement the cached
    path cannot stand behind. Always loud: a run that quietly padded by the
    wrong amount would be a different measurement wearing this one's name."""


def _warn(msg: str) -> None:
    print(f"[static_padding] {msg}", file=sys.stderr, flush=True)


# -- the filler corpus ---------------------------------------------------------


def load_filler(path) -> tuple[str, str]:
    """``(filler_text, sha256 of the file's bytes)``.

    Accepts the forms the Phase-0 builder and the long-context runner already
    use -- a JSON string, a JSON object with a ``"text"`` key -- plus plain
    text, so any neutral corpus file can stand in. Refuses loudly on a missing
    file, an unrecognized shape, or empty text."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise StaticPaddingError(f"filler cache unreadable at {p}: {e}") from e
    sha = hashlib.sha256(raw).hexdigest()
    try:
        text_bytes = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise StaticPaddingError(f"filler cache at {p} is not UTF-8 text: {e}") from e
    text: Optional[str] = None
    try:
        data = json.loads(text_bytes)
    except json.JSONDecodeError:
        text = text_bytes          # a plain-text corpus
    else:
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict) and isinstance(data.get("text"), str):
            text = data["text"]
        else:
            raise StaticPaddingError(
                f"unrecognized filler cache format in {p} (expected a JSON "
                "string, a JSON object with a 'text' key, or plain text)")
    if not text or not text.strip():
        raise StaticPaddingError(f"filler cache at {p} holds no text")
    return text, sha


def _repo_root() -> Path:
    # src/sisteer/eval/attacks/static_padding.py -> the repo
    return Path(__file__).resolve().parents[4]


def resolve_filler_path(explicit=None, cfg_value=None, repo_root=None) -> Path:
    """The first EXISTING of: ``explicit``, ``cfg_value``, ``$SISTEER_PG19_FILLER``, then
    ``<repo_root>/data/pg19_filler.json`` (the checkout a job runs from).

    ``explicit`` and ``cfg_value`` may carry a site placeholder (``$SHARED/...``); it is expanded
    through ``sisteer.site`` first, and an unset or unknown placeholder refuses by name rather than
    being searched as a literal path. An explicitly named path that does not exist refuses NAMING
    IT, rather than silently falling through to a default someone else wrote -- a run padded from
    an unintended corpus is not the run that was asked for. Refuses naming every path searched."""
    from sisteer import site as _site

    def _expanded(value, key: str) -> Path:
        try:
            return _site.expand_path(value, key=key)
        except _site.PlaceholderError as exc:
            raise StaticPaddingError(str(exc)) from None

    if explicit:
        p = _expanded(explicit, "--static-padding-filler")
        if p.is_file():
            return p
        raise StaticPaddingError(
            f"--static-padding-filler names {p}, which is not a file "
            "(refusing to fall back to a default corpus: the filler is part "
            "of what the run measures)")
    candidates: list[Path] = []
    if cfg_value:
        candidates.append(_expanded(cfg_value, "filler_cache"))
    env = os.environ.get(FILLER_ENV)
    if env:
        candidates.append(Path(env))
    candidates.append(Path(repo_root or _repo_root()) / REPO_FILLER_REL)
    for c in candidates:
        if c.is_file():
            return c
    raise StaticPaddingError(
        "no PG-19 filler cache found; searched "
        + ", ".join(str(c) for c in candidates)
        + f" (build one with scripts/fetch_stage2_data.py --only pg19-filler, "
        f"point at one with --static-padding-filler, or set ${FILLER_ENV})")


# -- the pre-tokenized sidecar -------------------------------------------------


def tokenizer_identity(tok) -> Optional[dict]:
    """``{"class", "name_or_path", "sha256"}`` for a fast HF tokenizer -- the
    sha256 is of its serialized definition (vocab, merges, normalizer,
    pre-tokenizer, post-processor, added tokens: everything that decides which
    ids a text becomes), so two tokenizers with the same hash tokenize
    identically whatever model id or revision they were loaded under, and any
    difference in the definition is a different hash. ``None`` for anything
    without a serializable definition (a slow tokenizer, a test double): no
    identity means no sidecar, and the corpus is tokenized in-process."""
    backend = getattr(tok, "backend_tokenizer", None)
    if not getattr(tok, "is_fast", False) or backend is None:
        return None
    try:
        blob = backend.to_str()
    except Exception:  # noqa: BLE001 - an unserializable backend simply has no identity
        return None
    return {"class": type(tok).__name__,
            "name_or_path": str(getattr(tok, "name_or_path", "") or ""),
            "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()}


def filler_ids_sidecar_path(filler_path, identity: Mapping[str, Any]) -> Path:
    """``<dir>/<corpus stem>.ids.<tokenizer sha256[:16]>.json`` -- one sidecar per
    tokenizer definition beside the corpus, so two models' sidecars never
    clobber each other (the full hash is checked inside the file)."""
    p = Path(filler_path)
    return p.with_name(f"{p.stem}.ids.{identity['sha256'][:16]}.json")


def _sidecar_mismatch(d, *, text_sha256: str, identity: Mapping[str, Any]) -> Optional[str]:
    if not isinstance(d, Mapping):
        return "not a JSON object"
    if d.get("schema") != IDS_SCHEMA:
        return f"schema {d.get('schema')!r} != {IDS_SCHEMA!r}"
    if d.get("text_sha256") != text_sha256:
        return "built from a different corpus text (text_sha256 mismatch)"
    t = d.get("tokenizer")
    if not isinstance(t, Mapping) or t.get("sha256") != identity["sha256"]:
        who = t.get("name_or_path") if isinstance(t, Mapping) else "?"
        return f"built by a different tokenizer definition ({who}; sha256 mismatch)"
    ids = d.get("ids")
    if (not isinstance(ids, list) or d.get("n_tokens") != len(ids)
            or not all(type(i) is int for i in ids)):
        return "ids are not the recorded count of ints"
    return None


def read_filler_ids_sidecar(path, *, text_sha256: str, identity: Mapping[str, Any]) -> Optional[list]:
    """The sidecar's ids, or ``None`` when it is absent or NOT usable. A present
    sidecar is used only when its schema, corpus text sha256, tokenizer sha256
    and token count all match; any mismatch (another corpus, another tokenizer,
    a torn or hand-edited file) is named on stderr and answered with ``None``
    -- the caller rebuilds from the corpus, which is authoritative. Ids from
    another tokenizer never feed a plan: that would be a silently mis-sized
    pad, the one failure this module exists to prevent."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        _warn(f"ignoring unreadable filler sidecar {p} ({type(e).__name__}: {e}); "
              "rebuilding from the corpus")
        return None
    why = _sidecar_mismatch(d, text_sha256=text_sha256, identity=identity)
    if why is not None:
        _warn(f"ignoring filler sidecar {p}: {why}; rebuilding from the corpus")
        return None
    return d["ids"]


def write_filler_ids_sidecar(path, ids, *, text_sha256: str, filler_sha256: Optional[str],
                             identity: Mapping[str, Any], tokenizer_revision: Optional[str] = None) -> None:
    """Write the sidecar atomically (temp file + rename in the same directory: a
    killed or concurrent build never leaves a torn file, and two drivers
    racing to build the same sidecar write identical bytes). Raises ``OSError``
    on an unwritable location; callers decide whether that is fatal."""
    p = Path(path)
    payload = {"schema": IDS_SCHEMA,
               "filler_sha256": filler_sha256,       # the CORPUS file, as the block records it
               "text_sha256": text_sha256,           # the text these ids tokenize
               "tokenizer": {**dict(identity), "revision": tokenizer_revision},
               "n_tokens": len(ids),
               "add_special_tokens": False,
               "builder": "sisteer.eval.attacks.static_padding",
               "ids": list(ids)}
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)


def ensure_filler_ids(filler_path, tok, *, filler_text: Optional[str] = None,
                      filler_sha256: Optional[str] = None, tokenizer_revision: Optional[str] = None,
                      write: bool = True) -> tuple[list, str]:
    """``(ids of the corpus under tok, how they were obtained)``.

    The ONE code path a plan and the Phase-0 builder share. Reads the sidecar
    beside ``filler_path`` when a matching one exists; otherwise tokenizes the
    whole corpus in-process (the stream's definition -- a prefix tokenization is
    not provably the prefix of the whole's, so the build is never bounded) and,
    with ``write``, writes the sidecar for the next process. The status names
    the route: ``"sidecar"`` (read), ``"built"`` (tokenized + written),
    ``"built-unwritten"`` (tokenized; the write failed -- warned, not fatal),
    ``"tokenized"`` (no sidecar possible: no corpus file at ``filler_path``,
    the given text is not that file's text, or the tokenizer has no identity).

    ``filler_text`` given: the text a plan was handed (the sidecar applies only
    when it IS the file's text). Absent: read from ``filler_path``."""
    p = Path(filler_path) if filler_path else None
    file_text = file_sha = None
    if p is not None and p.is_file():
        file_text, file_sha = load_filler(p)
    if filler_text is None:
        if file_text is None:
            raise StaticPaddingError(f"no filler text given and no corpus file at {filler_path!r}")
        filler_text = file_text
    identity = tokenizer_identity(tok)
    sidecar = None
    if identity is not None and file_text is not None and file_text == filler_text:
        sidecar = filler_ids_sidecar_path(p, identity)
    text_sha = hashlib.sha256(filler_text.encode("utf-8")).hexdigest()
    if sidecar is not None:
        ids = read_filler_ids_sidecar(sidecar, text_sha256=text_sha, identity=identity)
        if ids is not None:
            return ids, "sidecar"
    ids = list(tok(filler_text, add_special_tokens=False)["input_ids"])
    if sidecar is None or not write:
        return ids, "tokenized"
    try:
        write_filler_ids_sidecar(sidecar, ids, text_sha256=text_sha,
                                 filler_sha256=file_sha or filler_sha256 or None,
                                 identity=identity, tokenizer_revision=tokenizer_revision)
    except OSError as e:
        _warn(f"could not write the pre-tokenized filler sidecar {sidecar} ({e}); this "
              "process tokenized the corpus itself and the next one will too")
        return ids, "built-unwritten"
    return ids, "built"


# -- the spec ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PadSpec:
    """WHAT to pad with and by how much. ``tokens`` is the padding ADDED, not a
    total prompt length, so one number means the same thing whatever the
    payload's own length is."""

    tokens: int
    placement: str = "before"
    mode: str = "fixed"
    source: str = "pg19"
    filler_path: str = ""
    filler_sha256: str = ""
    tolerance: float = DEFAULT_TOLERANCE
    max_iters: int = DEFAULT_MAX_ITERS

    def __post_init__(self) -> None:
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 1:
            raise StaticPaddingError(
                f"padding tokens must be a positive int, got {self.tokens!r} "
                "(0 or absent means the knob is off; never a PadSpec)")
        if self.placement not in PLACEMENTS:
            raise StaticPaddingError(
                f"placement must be one of {list(PLACEMENTS)}, got {self.placement!r}")
        if self.mode not in MODES:
            raise StaticPaddingError(
                f"mode must be one of {list(MODES)}, got {self.mode!r}")
        if isinstance(self.tolerance, bool) or not (0 < float(self.tolerance) < 1):
            raise StaticPaddingError(
                f"tolerance must be in (0, 1), got {self.tolerance!r}")
        if isinstance(self.max_iters, bool) or int(self.max_iters) < 1:
            raise StaticPaddingError(
                f"max_iters must be >= 1, got {self.max_iters!r}")


# One filler string's ids per (tokenizer, text) per process: the corpus is ~635k
# tokens, and re-tokenizing it per plan costs seconds each with the GPU idle.
# Across processes the sidecar plays this role. Values are ``(ids, source)``.
# The value HOLDS the tokenizer: the key's ``id(tok)`` is only unique while that object is alive, and a tokenizer freed and
# another allocated at the same address (CPython reuses the slot) would otherwise read the first one's ids -- with a vocabulary
# that never produced them (a decode then indexes past it). Checked by identity on every hit.
_filler_ids_cache: dict[tuple[int, int], tuple[Any, list, str]] = {}


def _filler_token_ids(tok, text: str, *, filler_path: str = "", filler_sha256: str = "") -> tuple[list, str]:
    key = (id(tok), hash(text))
    hit = _filler_ids_cache.get(key)
    if hit is None or hit[0] is not tok:
        ids, source = ensure_filler_ids(filler_path, tok, filler_text=text, filler_sha256=filler_sha256 or None)
        hit = (tok, ids, source)
        _filler_ids_cache[key] = hit
    return hit[1], hit[2]


@dataclasses.dataclass
class _Slice:
    """One decoded filler slice: ``ids[offset:offset+take]`` as text, its exact
    standalone token count, and the EDGE that touches the payload with its own
    standalone count. ``uses`` drives the full-measurement sample."""

    text: str
    n_tokens: int
    edge: str
    n_edge: int
    uses: int = 0


class PadPlan:
    """A :class:`PadSpec` bound to a tokenizer and the filler text: renders one
    item's padded prompt, deterministically.

    ``verify_every`` (default :data:`DEFAULT_VERIFY_EVERY`): how often an item
    on a cached slice is ALSO tokenized in full and compared with the cached
    path's count -- the first item of every slice always is; ``1`` measures
    every item in full (today's cost). See :meth:`apply`.

    ``ids_source`` says where the filler's token stream came from
    (``"sidecar"`` / ``"built"`` / ``"built-unwritten"`` / ``"tokenized"``;
    :func:`ensure_filler_ids`); ``stats`` counts ``items`` rendered,
    ``verified`` in full, and ``slices`` decoded."""

    def __init__(self, spec: PadSpec, tok, filler_text: str, *,
                 verify_every: int = DEFAULT_VERIFY_EVERY) -> None:
        if not isinstance(spec, PadSpec):
            raise StaticPaddingError(
                f"PadPlan needs a PadSpec, got {type(spec).__name__}")
        if not filler_text or not filler_text.strip():
            raise StaticPaddingError("PadPlan needs non-empty filler text")
        if isinstance(verify_every, bool) or not isinstance(verify_every, int) or verify_every < 1:
            raise StaticPaddingError(
                f"verify_every must be a positive int, got {verify_every!r} "
                "(1 = every item measured in full)")
        self.spec = spec
        self._tok = tok
        self._filler = filler_text
        self.verify_every = verify_every
        self._ids, self.ids_source = _filler_token_ids(
            tok, filler_text, filler_path=spec.filler_path, filler_sha256=spec.filler_sha256)
        if not self._ids:
            raise StaticPaddingError("the filler text tokenized to nothing")
        self._slices: dict[tuple[int, int], _Slice] = {}
        self.stats = {"items": 0, "verified": 0, "slices": 0}

    # the store-identity block: NO per-item key may enter it (every row of one
    # store is compared against it for equality)
    @property
    def block(self) -> dict:
        s = self.spec
        return {"schema": SCHEMA, "tokens": int(s.tokens), "placement": s.placement,
                "mode": s.mode, "source": s.source, "filler_sha256": s.filler_sha256,
                "separator": SEPARATOR, "tolerance": float(s.tolerance)}

    def _count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def _offset(self, goal_id: str, room: int) -> int:
        if self.spec.mode == "fixed":
            return 0
        span = len(self._ids) - room
        if span <= 0:
            raise StaticPaddingError(
                f"filler too short for mode=per_goal: {len(self._ids)} tokens, "
                f"but an item needs up to {room} with room to vary the offset "
                f"(target {self.spec.tokens}); extend the filler cache or use "
                "mode=fixed")
        digest = hashlib.sha256(f"{self.spec.tokens}|{goal_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % span

    def _compose(self, filler_text: str, prompt: str) -> str:
        if self.spec.placement == "before":
            return f"{filler_text}{SEPARATOR}{prompt}"
        return f"{prompt}{SEPARATOR}{filler_text}"

    def _slice(self, offset: int, take: int) -> _Slice:
        """The decoded slice for (offset, take), decoded and counted ONCE: in
        mode=fixed every item of a level shares it; in mode=per_goal each
        goal's variants share theirs (a bounded FIFO, so a long per_goal run
        cycles through decodes at today's cost rather than growing)."""
        key = (offset, take)
        sl = self._slices.get(key)
        if sl is None:
            text = self._tok.decode(self._ids[offset:offset + max(0, take)], skip_special_tokens=True)
            edge = text[-SEAM_WINDOW_CHARS:] if self.spec.placement == "before" else text[:SEAM_WINDOW_CHARS]
            sl = _Slice(text=text, n_tokens=self._count(text), edge=edge, n_edge=self._count(edge))
            while len(self._slices) >= _SLICE_CACHE_SIZE:
                self._slices.pop(next(iter(self._slices)))
            self._slices[key] = sl
            self.stats["slices"] += 1
        return sl

    def _measure(self, sl: _Slice, sent: str, prompt: str, base: int, *, goal_id: str,
                 offset: int, take: int) -> int:
        """``tokens(sent) - tokens(prompt)`` -- see :meth:`apply` for the guarantee."""
        sl.uses += 1
        self.stats["items"] += 1
        if self.verify_every == 1:
            self.stats["verified"] += 1
            return self._count(sent) - base
        # the seam window: the filler edge in its real position beside the
        # separator and the payload, so every merge across the seam is measured
        seam = self._count(self._compose(sl.edge, prompt))
        fast = sl.n_tokens - sl.n_edge + seam - base
        if sl.uses == 1 or (sl.uses - 1) % self.verify_every == 0:
            self.stats["verified"] += 1
            full = self._count(sent) - base
            if full != fast:
                raise StaticPaddingError(
                    f"padding measurement disagrees for goal {goal_id!r}: the seam-window "
                    f"count gives {fast} padding tokens but the full tokenization of the "
                    f"sent text gives {full} (offset {offset}, {take} filler tokens, "
                    f"placement {self.spec.placement}) -- this tokenizer is not local "
                    "across the filler edge, so the cached-slice path cannot stand in for "
                    "a full measurement; build the plan with verify_every=1 (every item "
                    "tokenized in full) or report the tokenizer")
        return fast

    def apply(self, prompt: str, *, goal_id: str) -> tuple[str, dict]:
        """``(text actually sent, per-item record)``.

        The per-item record is ``{pad_tokens_target, pad_tokens_actual,
        filler_offset_tokens, padded_prompt_sha256}``; ``pad_tokens_actual`` is
        the measured delta ``tokens(sent) - tokens(prompt)`` and must land
        within ``spec.tolerance`` of the target or this raises. Deterministic:
        the same (spec, tokenizer, filler, prompt, goal_id) always yields the
        identical text and record, whatever the plan rendered before -- the
        slice search always starts at the target, and the caches hold only
        what a cold plan would recompute.

        HOW ``pad_tokens_actual`` IS MEASURED -- one of two routes, the same
        number either way:

        * ``verify_every=1``: the sent text is tokenized in full for every
          item. Exact per item; ~0.2-0.5 s of CPU per item at a 65,536-token pad.
        * otherwise (the default): the decoded slice and its exact standalone
          count are cached per (offset, take), and per item only the filler
          EDGE beside the payload (:data:`SEAM_WINDOW_CHARS` chars) is
          re-tokenized in its real position with the separator and the payload:
          ``count(slice) - count(edge) + count(edge + sep + payload)`` (or the
          mirror image for placement=after). Every merge across the
          filler/separator/payload seam lies inside that re-tokenized text, so
          it is MEASURED per item, not corrected for. The formula equals the
          full count whenever the tokenizer is LOCAL -- tokens more than a
          window of prose away from the seam do not depend on the payload --
          which every pre-tokenized BPE / SentencePiece tokenizer is (a
          pre-token never spans a window's worth of prose, and the slice's own
          start/end artefacts cancel because both counts share them). That
          assumption is CHECKED against a full tokenization on the first item
          of every cached slice and on every ``verify_every``-th item after
          it; a disagreement raises :class:`StaticPaddingError` and nothing is
          recorded. Guarantee: exact per item under locality, with locality
          verified on a deterministic 1-in-``verify_every`` sample of items
          (plus the first of each slice) -- never an estimate recorded as a
          measurement, and a deviation is refused, not absorbed.

        ``mode="per_goal"`` gives each goal its own offset, so its slice is
        decoded and fully measured on the goal's first item (today's cost)
        and cached for the goal's remaining variants."""
        target = int(self.spec.tokens)
        base = self._count(prompt)
        # room to grow while converging: the payload's own tokens do not move,
        # so the adjustment only ever chases the separator/merge boundary
        room = target + max(64, target // 8)
        offset = self._offset(goal_id, room)
        available = len(self._ids) - offset
        if available < target:
            raise StaticPaddingError(
                f"filler too short: {available} tokens available from offset "
                f"{offset} < {target} needed; extend the filler cache "
                "(scripts/fetch_stage2_data.py --only pg19-filler)")
        take = target
        actual = -1
        for _ in range(int(self.spec.max_iters)):
            if take > available:
                raise StaticPaddingError(
                    f"filler exhausted while adjusting to {target} padding "
                    f"tokens (needed {take} from offset {offset}, have "
                    f"{available}); extend the filler cache")
            sl = self._slice(offset, take)
            sent = self._compose(sl.text, prompt)
            actual = self._measure(sl, sent, prompt, base, goal_id=goal_id, offset=offset, take=take)
            if abs(actual - target) <= self.spec.tolerance * target:
                rec = {"pad_tokens_target": target,
                       "pad_tokens_actual": int(actual),
                       "filler_offset_tokens": int(offset),
                       "padded_prompt_sha256":
                           hashlib.sha256(sent.encode("utf-8")).hexdigest()}
                return sent, rec
            take += (target - actual)
            if take < 1:
                take = 1
        raise StaticPaddingError(
            f"could not hit {target} padding tokens within "
            f"{self.spec.tolerance:.0%} after {self.spec.max_iters} "
            f"adjustments (last actual={actual}); raise --static-padding-"
            "tolerance or report the tokenizer")


# -- the store guard -----------------------------------------------------------


def block_identity(value) -> Optional[dict]:
    """The STORE-IDENTITY subset of a row's ``padding`` block (per-item keys
    dropped), or ``None`` for a row that carries none."""
    if not isinstance(value, Mapping):
        return None
    return {k: value[k] for k in BLOCK_KEYS if k in value}


def assert_static_padding_consistent(responses_path, expected_block: Optional[Mapping[str, Any]]) -> None:
    """ONE PADDING CONTRACT PER STORE.

    Refuse to append padded rows onto a store holding unpadded rows, unpadded
    rows onto a padded store, or rows under a DIFFERENT block (another token
    count, placement, mode or filler). Padded and unpadded are the contrast
    being measured, so one store must never hold both -- the same rule
    ``static_common`` applies to the engine and the decoder. A torn tail line
    is skipped, matching resume."""
    from .static_common import StaticStoreIncompleteError

    p = Path(responses_path)
    if not p.exists():
        return
    want = dict(expected_block) if expected_block is not None else None
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail from a SIGKILL mid-append
        got = block_identity(rec.get("padding"))
        if got == want:
            continue
        if want is None:
            raise StaticStoreIncompleteError(
                f"{p} holds a PADDED row ({got.get('tokens') if got else '?'} "
                "padding tokens) while this run pads NOTHING — refusing to mix "
                "padded and unpadded rows in one responses file (that contrast "
                "is measured across stores); run into a fresh private_run")
        if got is None:
            raise StaticStoreIncompleteError(
                f"{p} holds an UNPADDED row while this run pads "
                f"{want.get('tokens')} tokens — refusing to mix padded and "
                "unpadded rows in one responses file; a padded run needs its "
                "own private_run, never an unpadded store")
        raise StaticStoreIncompleteError(
            f"{p} holds a row padded under {got} while this run pads under "
            f"{want} — refusing to mix two padding contracts (tokens / "
            "placement / mode / filler) in one responses file; run into a "
            "fresh private_run")


# -- the window preflight ------------------------------------------------------


def window_problem(*, pad_tokens: int, cap: int, window: Optional[int],
                   reserve: int = DEFAULT_RESERVE_TOKENS) -> Optional[str]:
    """``None`` when ``pad_tokens + cap + reserve <= window``, else a message
    naming all four numbers and the flags that move them.

    ``window=None`` (no launcher-resolved geometry) checks nothing: the
    server's own rejection governs, exactly as it does today. The point of the
    check is that a padded run which cannot fit should die in seconds, not
    after the GPU time is spent turning every goal into a window error."""
    if window is None:
        return None
    need = int(pad_tokens) + int(cap) + int(reserve)
    if need <= int(window):
        return None
    return (f"padding {int(pad_tokens)} tokens + completion cap {int(cap)} + "
            f"reserve {int(reserve)} = {need} tokens exceeds the served window "
            f"{int(window)} — every request would be refused by the server. "
            "Stand the target up at a larger --max-model-len (and pass it as "
            "--target-max-model-len), lower --static-padding-tokens, or lower "
            "the config's attacks.static_max_new_tokens; --static-padding-"
            "reserve changes the headroom this check leaves for the payload "
            "and the chat template.")
