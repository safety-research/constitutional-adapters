"""Candidate system-instruction loader (the candidate-SI set).

Reads a ``candidate_sis.json`` snapshot -- 60 candidate system instructions
(A1-A20 safety, B1-B20 constitution-derived, C1-C20 irrelevant controls) --
into frozen :class:`~sisteer.data.instructions.Instruction` objects. The
snapshot itself is NOT part of this repository (B1 carries the full
~142k-character constitution verbatim): the caller names it, by path or by
``$SISTEER_CANDIDATE_SIS``, and can pin its bytes by sha (``expect_sha``).
One of its entries, the short irrelevant control C16, is the teacher
instruction of a distillation fit (``scripts/train_two_si_objects.py`` with
``SISTEER_LOAD_CANDIDATE_SIS`` set, which registers the 60 ids; see the end
of ``instructions.py``).

Every load re-validates the file and FAILS LOUDLY on any drift:

- exactly 60 entries whose ids are exactly {A1..A20, B1..B20, C1..C20},
  with the id prefix BOUND to the category (A->safety, B->constitution,
  C->irrelevant; exactly 20 per category);
- every id free of ``__`` (cell ids are ``prompted__<si_id>``, so a ``__``
  inside an SI id would make the cell-id grammar ambiguous);
- no id already registered in ``instructions.BY_ID`` with DIFFERENT
  content (re-loading the same file after the env-guarded registration in
  instructions.py is idempotent, like ``instructions._register_instruction``);
- 60 distinct sha256(text) -- text identity is the intervention identity;
- B2 text == ``CONSTITUTION_PROMPT`` and C3 text == ``IRRELEVANT_PROMPT``
  byte-for-byte (a free cross-check against the repo's own constants);
- no entry is a ``<<...>>`` pointer sentinel -- B1 must be the inlined
  constitution, never a ``<<FULL_CONSTITUTION_VERBATIM>>`` placeholder.

THE SUBSET MODE (opt-in; added 2026-11-15). A host that
needs only some of the 60 texts -- the Modal C16 cells, whose volume every
member of the workspace can read, so the full snapshot must not go there --
loads a SUBSET file instead. It is derived from the full snapshot by
``scripts/sikl4/make_candidate_sis_subset.py``, never hand-written, and shaped
``{"meta": {"subset_of_sha256", "ids", "text_sha256"}, "system_instructions":
[<the entries, copied verbatim>]}``. The mode is entered ONLY when
``$SISTEER_CANDIDATE_SIS_SUBSET_SHA256`` is set (non-empty):
:func:`load_candidate_instructions` then dispatches to
:func:`load_candidate_subset`, and with it unset every check above runs exactly
as before. The subset mode refuses, each by name:

- file bytes that do not hash to that variable;
- ``meta.subset_of_sha256`` other than :data:`CANDIDATE_SIS_FULL_SHA256`;
- an id outside {A1..C20}, or one with no pinned text sha in
  :data:`SUBSET_TEXT_SHA256` (today only C16, :data:`C16_TEXT_SHA256`, read
  once from the full snapshot): nothing else would tie its text to the full
  snapshot's;
- entry ids other than exactly ``meta.ids``, and ``meta.text_sha256`` other
  than the pinned shas;
- a text whose sha256 is not the pinned one, so the subset's text IS the full
  snapshot's and the registered Instruction (id, category, text, checker) is
  the one the full snapshot registers.

Every per-entry check above applies in the subset mode too; the 60-count, the
20-per-category count and the B2/C3 cross-checks are skipped in this mode only
(B2 and C3 have no pinned subset sha, so a subset never carries them).

Who sets the pair (``SISTEER_LOAD_CANDIDATE_SIS=<subset>`` and the sha
variable): every process that builds a cell registry from a config naming a
subset id, not only the processes that serve that id's cells. The
explicit-cells registry (``defense_cell.registry_from_explicit_cells``)
resolves the SI of every cell the config lists and refuses the WHOLE config
when one does not resolve, so a container serving a safety cell over a config
that also lists ``C16__lora_r4`` refuses at standup without the pair. Setting
it where C16 is not served changes nothing else: the subset registers its ids
alone, and every other cell's registry row is the same with or without it.

``checker="external"`` is the registry's sentinel for "no programmatic
checker -- scored by an external judge suite"; tests/test_data.py exempts
it from the checker-exists sweep via DEFERRED_CHECKERS, and an accidental
``checkers.check("external", ...)`` fails loudly (KeyError).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, NoReturn

from .safety_texts import CONSTITUTION_PROMPT, IRRELEVANT_PROMPT

# Instruction/BY_ID are imported LAZILY inside the loading helpers,
# never at module top: instructions.py's env-guarded tail imports THIS
# module, so a top-level import here would make the pair order-dependent --
# with SISTEER_LOAD_CANDIDATE_SIS set, `import candidate_sis` before
# `import instructions` would raise ImportError (candidate_sis triggering
# instructions' tail mid-way through its own init). The lazy import runs only
# when the loader is CALLED, by which point instructions.py has defined both
# names (its tail is at module end), so either import order works.
if TYPE_CHECKING:  # annotation-only; never executed at runtime
    from .instructions import Instruction

# The snapshot's path when the caller passes none (an explicit `path` wins).
PATH_ENV = "SISTEER_CANDIDATE_SIS"

N_EXPECTED = 60
EXPECTED_IDS = frozenset(f"{p}{n}" for p in "ABC" for n in range(1, 21))

# The id prefix IS the category: consumers group the candidates by category,
# so a mislabeled entry (an A-id filed as "constitution") would silently land
# in the wrong group. The loader binds prefix -> category per entry and
# requires exactly 20 per category; consumers take THIS loader's categories
# rather than re-extracting unvalidated ones.
CATEGORY_BY_PREFIX = {"A": "safety", "B": "constitution", "C": "irrelevant"}
N_PER_CATEGORY = 20

# Entries that must be byte-identical to constants already in the repo
# (sha16 21e06f482a6559b3 / 71ff67aa2bc000f1). A mismatch means the
# candidate file drifted, so it is a loud stop, not a warning.
_CROSS_CHECKS: dict[str, tuple[str, str]] = {
    "B2": ("safety_texts.CONSTITUTION_PROMPT", CONSTITUTION_PROMPT),
    "C3": ("safety_texts.IRRELEVANT_PROMPT", IRRELEVANT_PROMPT),
}

# The retired pointer form ("<<FULL_CONSTITUTION_VERBATIM>>").
_POINTER_RE = re.compile(r"^\s*<<.*>>\s*$", re.DOTALL)

# -- the subset mode (see the module docstring) --------------------------------
#: Set (non-empty) to the subset FILE's sha256 to enter the subset mode, and only then.
SUBSET_SHA_ENV = "SISTEER_CANDIDATE_SIS_SUBSET_SHA256"
#: The FILE sha256 of the one full snapshot a subset may be derived from: the
#: snapshot the sikl4 refits were fit against and the registry's si_text rows
#: name.
CANDIDATE_SIS_FULL_SHA256 = "ba8efe82ece5a35a97e558e31ca9c2660c220bc3677a02193e4b4a126560d60e"
#: sha256 of C16's text in that snapshot (517 characters, the sikl4 refits'
#: teacher text; the value the full path registers and the registry's si_text
#: row for C16 carries).
C16_TEXT_SHA256 = "ba214825889190f8b83ab705187e4eeb66174da3c4af36f95fd60d77e34d139a"
#: The ids a subset may carry, each with the sha256 of its text in the full
#: snapshot. An id with no entry here is refused.
SUBSET_TEXT_SHA256: Mapping[str, str] = MappingProxyType({"C16": C16_TEXT_SHA256})
#: A subset file's meta keys, exactly.
SUBSET_META_KEYS = frozenset({"subset_of_sha256", "ids", "text_sha256"})
#: Matched with ``fullmatch``: ``$`` under ``match`` would let a trailing newline through.
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def default_path() -> str:
    """The snapshot path when the caller names none: ``$SISTEER_CANDIDATE_SIS``.
    The snapshot is not part of this repository, so there is no fallback: an
    unset variable refuses by name rather than guessing a location."""
    path = os.environ.get(PATH_ENV)
    if not path:
        raise FileNotFoundError(
            f"no candidate-SI snapshot named: pass a path, or set {PATH_ENV} "
            "(or SISTEER_LOAD_CANDIDATE_SIS=<path>) to the "
            "candidate_sis.snapshot.json the fit was specified against -- the "
            "snapshot is not part of this repository")
    return path


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha16(text: str) -> str:
    return _sha256(text)[:16]


def load_candidate_instructions(
    path: str | None = None, *, expect_sha: str | None = None
) -> list[Instruction]:
    """Read candidate_sis.json → 60 frozen Instruction objects. Fails loudly.

    ``path=None`` resolves via :func:`default_path`. ``expect_sha`` (full
    sha256 hex of the FILE BYTES) pins the exact snapshot. The returned
    instructions carry ``checker="external"`` (externally judged) and are
    NOT registered here — registration goes through
    ``instructions._register_instruction`` (see the env-guarded block at the end
    of instructions.py, keyed on ``SISTEER_LOAD_CANDIDATE_SIS``).

    With ``$SISTEER_CANDIDATE_SIS_SUBSET_SHA256`` set, the file is read as a
    SUBSET instead (:func:`load_candidate_subset`, pinned by that variable)
    and only its ids are returned; ``expect_sha`` still applies first.
    """
    resolved = path or default_path()
    raw = Path(resolved).read_bytes()

    if expect_sha is not None:
        got = hashlib.sha256(raw).hexdigest()
        if got != expect_sha:
            raise ValueError(
                f"candidate_sis sha mismatch at {resolved}: "
                f"expected {expect_sha}, got {got}"
            )

    subset_sha = os.environ.get(SUBSET_SHA_ENV)
    if subset_sha:
        return _load_subset(resolved, raw, subset_sha)

    doc = json.loads(raw)
    # Snapshot shape: {"meta": ..., "system_instructions": [...]}; a bare
    # list of entries is accepted for fixtures.
    if isinstance(doc, dict):
        try:
            entries = doc["system_instructions"]
        except KeyError:
            raise ValueError(
                f"{resolved}: dict form must carry a 'system_instructions' "
                f"key (found keys {sorted(doc)})"
            ) from None
    else:
        entries = doc
    if not isinstance(entries, list):
        raise ValueError(
            f"{resolved}: system_instructions must be a list, "
            f"got {type(entries).__name__}"
        )

    instructions = _entries_to_instructions(resolved, entries)

    # -- exactly the 60 expected ids -------------------------------------
    ids = {i.id for i in instructions}
    if len(instructions) != N_EXPECTED or ids != EXPECTED_IDS:
        raise ValueError(
            f"{resolved}: expected exactly {N_EXPECTED} entries with ids "
            f"A1..A20,B1..B20,C1..C20; got {len(instructions)} entries "
            f"(missing: {sorted(EXPECTED_IDS - ids)}, "
            f"unexpected: {sorted(ids - EXPECTED_IDS)})"
        )

    # -- exactly 20 per category ------------------------------------------
    # Implied by the id check + the per-entry prefix binding, but asserted in its own
    # right so the invariant consumers lean on is named, not inferred.
    per_cat: dict[str, int] = {}
    for instr in instructions:
        per_cat[instr.category] = per_cat.get(instr.category, 0) + 1
    want = {c: N_PER_CATEGORY for c in CATEGORY_BY_PREFIX.values()}
    if per_cat != want:
        raise ValueError(
            f"{resolved}: category counts {per_cat} != the expected "
            f"{want} (prefix-category binding)"
        )

    _check_registry_and_distinct(resolved, instructions)

    # -- cross-checks against the repo constants --------------------------
    by_id = {i.id: i for i in instructions}
    for sid, (const_name, const_text) in _CROSS_CHECKS.items():
        if by_id[sid].text != const_text:
            raise ValueError(
                f"{resolved}: {sid} text (sha16 {_sha16(by_id[sid].text)}) "
                f"!= {const_name} (sha16 {_sha16(const_text)}) — the "
                "candidate file drifted from the snapshot this loader expects"
            )

    return instructions


def load_candidate_subset(path: str | Path, *, subset_sha256: str) -> list[Instruction]:
    """Read a candidate-SI SUBSET file → its frozen Instruction objects, the
    file pinned by ``subset_sha256`` (the sha256 of its bytes). Fails loudly
    on every refusal the module docstring lists. Nothing is registered here.
    This is the subset mode's entry point: :func:`load_candidate_instructions`
    calls it when ``$SISTEER_CANDIDATE_SIS_SUBSET_SHA256`` is set."""
    return _load_subset(str(path), Path(path).read_bytes(), subset_sha256)


def _entries_to_instructions(resolved: str, entries: list) -> list[Instruction]:
    """The frozen Instruction of each entry, after the per-entry checks (shape,
    id grammar, duplicates, the prefix-category binding, no pointer sentinel),
    each naming the offending entry. Shared by the full and the subset path."""
    # Lazy on purpose — see the module-top note.
    from .instructions import Instruction

    # -- per-entry shape + id grammar (checked before set-level asserts so
    # the error names the offending entry, not just the aggregate) --------
    instructions: list[Instruction] = []
    seen: dict[str, int] = {}
    for pos, e in enumerate(entries):
        missing = [k for k in ("id", "category", "text") if not e.get(k)]
        if missing:
            raise ValueError(
                f"{resolved}: entry {pos} ({e.get('id', '<no id>')!r}) is "
                f"missing/empty {missing}"
            )
        sid, text = e["id"], e["text"]
        if "__" in sid:
            raise ValueError(
                f"{resolved}: id {sid!r} contains '__' — cell ids are "
                "'prompted__<si_id>', so '__' inside an SI id is forbidden"
            )
        if sid in seen:
            raise ValueError(
                f"{resolved}: duplicate id {sid!r} (entries {seen[sid]} "
                f"and {pos})"
            )
        seen[sid] = pos
        # -- the id prefix binds the category ----------------------------
        bound = CATEGORY_BY_PREFIX.get(sid[:1])
        if bound is not None and e["category"] != bound:
            raise ValueError(
                f"{resolved}: {sid} is filed under category "
                f"{e['category']!r} but its id prefix {sid[:1]!r} binds it "
                f"to {bound!r} — a mislabeled candidate would be grouped "
                "with the wrong category (prefix-category binding)"
            )
        if _POINTER_RE.match(text):
            raise ValueError(
                f"{resolved}: {sid} text is a '<<...>>' pointer sentinel — "
                "the file must inline the full text (B1 is inlined, never "
                "a pointer)"
            )
        instructions.append(
            Instruction(
                id=sid,
                category=e["category"],
                text=text,
                anti_text=None,
                checker="external",
                checker_kwargs={"suite": "align"},
            )
        )
    return instructions


def _check_registry_and_distinct(resolved: str, instructions: list[Instruction]) -> None:
    """No collision with an id already registered under different content, and
    distinct text hashes. Shared by the full and the subset path."""
    # Lazy on purpose — see the module-top note.
    from .instructions import BY_ID

    # -- no collision with pre-existing registry ids ---------------------
    # Identical re-registration is fine (the env-guarded load in
    # instructions.py already ran); different content under a known id is
    # a real collision.
    for instr in instructions:
        existing = BY_ID.get(instr.id)
        if existing is not None and existing != instr:
            raise ValueError(
                f"{resolved}: id {instr.id!r} already registered in "
                "instructions.BY_ID with different content"
            )

    # -- distinct text hashes ---------------------------------------------
    by_hash: dict[str, list[str]] = {}
    for instr in instructions:
        by_hash.setdefault(_sha16(instr.text), []).append(instr.id)
    dupes = {h: ids_ for h, ids_ in by_hash.items() if len(ids_) > 1}
    if dupes:
        raise ValueError(
            f"{resolved}: candidate texts are not distinct — duplicate "
            f"sha256(text) groups: {sorted(dupes.values())}"
        )


def _keys_or_type(value: object) -> object:
    """What a subset refusal says it got: a dict's sorted keys, else the value's type name."""
    return sorted(value) if isinstance(value, dict) else type(value).__name__


def _load_subset(resolved: str, raw: bytes, subset_sha: str) -> list[Instruction]:
    """The subset mode over ``raw`` (the bytes read from ``resolved``)."""

    def refuse(why: str) -> NoReturn:
        raise ValueError(f"{resolved}: candidate-SI subset refused: {why}")

    if not isinstance(subset_sha, str) or not _SHA256_RE.fullmatch(subset_sha):
        raise ValueError(
            f"{SUBSET_SHA_ENV} must be the subset file's sha256 (64 lowercase "
            f"hex), got {subset_sha!r}")
    got = hashlib.sha256(raw).hexdigest()
    if got != subset_sha:
        raise ValueError(
            f"candidate_sis subset sha mismatch at {resolved}: {SUBSET_SHA_ENV} "
            f"pins {subset_sha}, the file hashes to {got}")

    doc = json.loads(raw)
    if not isinstance(doc, dict) or set(doc) != {"meta", "system_instructions"}:
        refuse("a subset file is a dict with exactly the keys 'meta' and "
               f"'system_instructions' (got {_keys_or_type(doc)})")
    meta, entries = doc["meta"], doc["system_instructions"]
    if not isinstance(meta, dict) or set(meta) != SUBSET_META_KEYS:
        refuse(f"meta must carry exactly the keys {sorted(SUBSET_META_KEYS)} "
               f"(got {_keys_or_type(meta)})")
    if meta["subset_of_sha256"] != CANDIDATE_SIS_FULL_SHA256:
        refuse(f"meta.subset_of_sha256 {meta['subset_of_sha256']!r} is not the "
               f"pinned full snapshot {CANDIDATE_SIS_FULL_SHA256} "
               "(candidate_sis.CANDIDATE_SIS_FULL_SHA256)")

    ids = meta["ids"]
    if (not isinstance(ids, list) or not ids
            or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids)):
        refuse(f"meta.ids must be a non-empty list of distinct id strings, got {ids!r}")
    foreign = [i for i in ids if i not in EXPECTED_IDS]
    if foreign:
        refuse(f"meta.ids {foreign} are not candidate ids (A1..A20, B1..B20, C1..C20)")
    unpinned = [i for i in ids if i not in SUBSET_TEXT_SHA256]
    if unpinned:
        refuse(f"meta.ids {unpinned} have no pinned text sha256 in "
               f"candidate_sis.SUBSET_TEXT_SHA256 (pinned: {sorted(SUBSET_TEXT_SHA256)}) "
               "— nothing would tie their text to the full snapshot's")
    claimed = meta["text_sha256"]
    if not isinstance(claimed, dict) or set(claimed) != set(ids):
        refuse(f"meta.text_sha256 must map exactly meta.ids {ids} to their text "
               f"sha256 (got {_keys_or_type(claimed)})")
    wrong = {i: claimed[i] for i in ids if claimed[i] != SUBSET_TEXT_SHA256[i]}
    if wrong:
        refuse(f"meta.text_sha256 {wrong} differ from the pinned full-snapshot text "
               f"sha256 {({i: SUBSET_TEXT_SHA256[i] for i in wrong})}")

    if not isinstance(entries, list):
        refuse(f"system_instructions must be a list, got {type(entries).__name__}")
    instructions = _entries_to_instructions(resolved, entries)
    got_ids = [i.id for i in instructions]
    if sorted(got_ids) != sorted(ids):  # each list is duplicate-free by now
        refuse(f"the entries' ids {got_ids} are not exactly meta.ids {ids} "
               f"(missing: {sorted(set(ids) - set(got_ids))}, "
               f"unexpected: {sorted(set(got_ids) - set(ids))})")
    for instr in instructions:
        h = _sha256(instr.text)
        if h != SUBSET_TEXT_SHA256[instr.id]:
            refuse(f"{instr.id} text sha256 {h} != {SUBSET_TEXT_SHA256[instr.id]}, the "
                   f"sha256 of {instr.id}'s text in the full snapshot — a subset carries "
                   "the full snapshot's text byte for byte")

    _check_registry_and_distinct(resolved, instructions)
    return instructions
