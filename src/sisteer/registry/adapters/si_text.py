"""SI texts: ``configs/registry/si_texts.yaml`` -> ``si_text`` records.

Runs with the dimensions, before objects (a prompted object references its ``si_text`` row). Every row's
``si_text_sha256`` is the sha256 of the EXACT rendered text bytes (utf-8 of ``Instruction.text``); the seed
carries the value it computed, and :func:`verify_seed_shas` recomputes it from the code that ships the text
(``sisteer.data.instructions``) and REFUSES a mismatch by name — the seed is never trusted over the bytes.
An SI prefix hash met in a manifest that no row explains refuses naming the hash
(:func:`si_id_for_prefix_hash`), never a guess.

Texts the code does NOT ship (the candidate-SI teachers A10 / B3 / C16 of the ``candsi`` objects) carry a
``text_source`` block naming the out-of-tree snapshot that holds them (path, ``$ENV`` override, the snapshot FILE's
sha256, the entry id). Verification reads the snapshot when it is present (file sha checked first, then the entry's
text bytes) and refuses a mismatch exactly as for shipped texts. When the snapshot is NOT on this host the row is
loaded with its seeded sha and says so — ``field_confidence_json.si_text_sha256`` records that the value was
verified at seed time and not re-verified here (:class:`SnapshotUnavailable`; the loader reports it as a warning,
never silently). A ``text_source`` of an unknown kind refuses.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Mapping, Optional, Tuple

from .. import identity
from .. import records as R
from .. import site_forms as SF
from .base import AdapterRefusal

#: the seed's row-only keys that are not ``si_text`` columns: kept in ``extra_json`` under their own names
_EXTRA_KEYS = ("si_text_sha16_prefix_hash", "length", "producer_spellings", "used_by_objects", "text_source")
#: the one out-of-tree text source the adapter knows how to read (the candidate-SI screen's frozen snapshot,
#: ``sisteer.data.candidate_sis`` on the candidate-si-refit lineage; ``{"meta": …, "system_instructions": [{id, text, …}]}``)
EXTERNAL_TEXT_SOURCE_KINDS = ("candidate_sis_snapshot",)


class SiTextRefusal(AdapterRefusal):
    """A seed row disagrees with the bytes the code ships, or names a text the code does not."""


class UnknownPrefixHash(LookupError):
    """A 16-hex SI prefix hash no ``si_text`` row explains (the adapter refuses rather than guessing)."""


class SnapshotUnavailable(SiTextRefusal):
    """A row's out-of-tree text snapshot is not present on this host: the sha cannot be re-verified HERE (it was at
    seed time). Distinct from a mismatch, which is a plain :class:`SiTextRefusal`."""


def external_snapshot_path(text_source: Mapping[str, object]) -> str:
    """The snapshot path a ``text_source`` block names: the block's override key (``env``) when the environment or this
    host's site config sets it, else the seed's path with its site placeholder (``$STATE/...``) expanded through the site
    config, then any other variable and ``~``. A placeholder this host does not resolve stays as written, so the snapshot
    is reported absent by that name. The seed never names a host, a user or a session directory."""
    env = str(text_source.get("env") or "")
    override = SF.config_value(env) if env else None
    if override:
        return os.path.expanduser(override)
    path = str(text_source.get("path") or "")
    return os.path.expanduser(os.path.expandvars(SF.expansion(path) or path))


def external_text_bytes(si_id: str, text_source: Mapping[str, object]) -> bytes:
    """The text bytes of ``si_id`` from its out-of-tree snapshot: the FILE sha is checked before anything is read out
    of it, the entry is looked up by id, and every failure names what it saw."""
    kind = str(text_source.get("kind") or "")
    if kind not in EXTERNAL_TEXT_SOURCE_KINDS:
        raise SiTextRefusal(si_id, f"text_source.kind {kind!r} is not one the adapter reads ({EXTERNAL_TEXT_SOURCE_KINDS})")
    path = external_snapshot_path(text_source)
    if not path or not os.path.isfile(path):
        raise SnapshotUnavailable(si_id, f"snapshot {path or '<no path>'} is not present on this host (set ${text_source.get('env')} "
                                         f"or restore the file; the seeded sha was verified against it at seed time)")
    with open(path, "rb") as fh:
        raw = fh.read()
    want_file = str(text_source.get("file_sha256") or "")
    got_file = hashlib.sha256(raw).hexdigest()
    if want_file and got_file != want_file:
        raise SiTextRefusal(si_id, f"snapshot {path} sha256 {got_file[:16]}… != the seed's file_sha256 {want_file[:16]}… (a drifted snapshot is not the text the objects distilled)")
    doc = json.loads(raw.decode("utf-8"))
    entries = doc.get("system_instructions") if isinstance(doc, dict) else doc
    if not isinstance(entries, list):
        raise SiTextRefusal(si_id, f"snapshot {path} carries no system_instructions list")
    entry_id = str(text_source.get("entry_id") or si_id)
    for ent in entries:
        if isinstance(ent, Mapping) and str(ent.get("id")) == entry_id:
            text = ent.get("text")
            if not isinstance(text, str):
                raise SiTextRefusal(si_id, f"snapshot entry {entry_id!r} has no text")
            return text.encode("utf-8")
    raise SiTextRefusal(si_id, f"snapshot {path} has no entry with id {entry_id!r}")


def text_bytes_for(si_id: str, row: Optional[Mapping[str, object]] = None) -> bytes:
    """The exact rendered text bytes of ``si_id`` as ``sisteer.data.instructions`` ships them — or, for a row that
    carries a ``text_source`` block, as its out-of-tree snapshot holds them (:func:`external_text_bytes`).

    ``safety_refusal`` is the module-level ``SAFETY_SI`` (not in ``BY_ID``; the seed's note) — resolved through the
    module attribute. Any other id resolves through ``BY_ID``; an unknown id refuses by name."""
    if row is not None and isinstance(row.get("text_source"), Mapping):
        return external_text_bytes(si_id, row["text_source"])
    from sisteer.data import instructions as I  # torch-free (verified: importing it loads no torch)
    if si_id == "safety_refusal":
        return I.SAFETY_SI.text.encode("utf-8")
    try:
        return I.BY_ID[si_id].text.encode("utf-8")
    except KeyError:
        raise SiTextRefusal(si_id, "no Instruction with this id in sisteer.data.instructions (BY_ID) and it is not SAFETY_SI") from None


def sha256_of_text(si_id: str, row: Optional[Mapping[str, object]] = None) -> str:
    return hashlib.sha256(text_bytes_for(si_id, row)).hexdigest()


def verify_seed_shas(rows: List[Mapping[str, object]], seed_file: str = "si_texts.yaml",
                     unverifiable: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Recompute every row's sha from the bytes (the code's, or the row's out-of-tree snapshot); refuse the first
    mismatch naming id, seed sha and computed sha. Returns ``{si_id: verified sha}``.

    A row whose snapshot is not on this host cannot be re-verified: with ``unverifiable`` given, its reason is
    recorded there (``{si_id: why}``) and the row is skipped — the caller stamps the record; without it, the
    :class:`SnapshotUnavailable` propagates (a caller that did not ask to tolerate absence is refused loudly)."""
    out: Dict[str, str] = {}
    for row in rows:
        sid = str(row.get("si_id"))
        want = str(row.get("si_text_sha256"))
        try:
            got = sha256_of_text(sid, row)
        except SnapshotUnavailable as exc:
            if unverifiable is None:
                raise
            unverifiable[sid] = str(exc)
            continue
        if got != want:
            raise SiTextRefusal(f"{seed_file}:{sid}", f"si_text_sha256 {want[:16]}… in the seed != sha256 of the text bytes {got[:16]}…")
        out[sid] = got
    return out


def prefix_hash_index(rows: List[Mapping[str, object]]) -> Dict[str, str]:
    """``{sha256_hex16: si_id}`` — the 16-hex prefix every Petri manifest / served proxy pins (H8)."""
    idx: Dict[str, str] = {}
    for row in rows:
        h16 = str(row["si_text_sha256"])[:16]
        if h16 in idx and idx[h16] != row["si_id"]:
            raise SiTextRefusal("si_texts.yaml", f"prefix hash {h16} names two ids ({idx[h16]}, {row['si_id']})")
        idx[h16] = str(row["si_id"])
    return idx


def si_id_for_prefix_hash(hash16: str, index: Mapping[str, str]) -> str:
    """The si_id behind a manifest's 16-hex prefix hash; an unknown hash refuses BY NAME (never a default)."""
    try:
        return index[str(hash16)]
    except KeyError:
        raise UnknownPrefixHash(f"SI prefix hash {hash16!r} matches no si_text row (known: {sorted(index)}); "
                                "refusing to guess the instruction — register the text or name it in si_texts.yaml") from None


def si_text_payload(row: Mapping[str, object], normalise) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Map one seed row onto the ``si_text`` columns. ``normalise`` is the dimensions loader's column normaliser
    (dates -> UTC ms, ``unverified`` -> the column's representation, confidence -> field_confidence_json)."""
    payload: Dict[str, object] = {}
    extra: Dict[str, object] = {}
    for k, v in row.items():
        if k == "si_token_count_by_model":
            payload["si_token_count_by_model_json"] = v if isinstance(v, dict) else {"_all": v}
        elif k in _EXTRA_KEYS or k == "source":
            extra["seed_source" if k == "source" else k] = v
        elif k in ("confidence", "inferred_reason"):
            continue  # folded into field_confidence_json by the caller's normaliser
        else:
            payload[k] = v
    return normalise("si_text", payload, row, extra)


def records_for(rows: List[Mapping[str, object]], producer: Mapping[str, object], written_utc: str, normalise,
                verify: bool = True, unverifiable: Optional[Dict[str, str]] = None) -> List[Dict[str, object]]:
    """One ``si_text`` record per row. With ``verify`` the shas are recomputed first; a row whose out-of-tree snapshot
    is absent is loaded with its seeded sha and ``field_confidence_json.si_text_sha256`` says so (the reason also
    lands in ``unverifiable`` when the caller passes a dict, so the loader can report it)."""
    unv: Dict[str, str] = unverifiable if unverifiable is not None else {}
    if verify:
        verify_seed_shas(rows, unverifiable=unv)
    out = []
    for row in rows:
        payload, extra = si_text_payload(row, normalise)
        payload = dict(payload)
        sid = str(row.get("si_id"))
        if sid in unv:
            ts = row.get("text_source") if isinstance(row.get("text_source"), Mapping) else {}
            fc = payload.get("field_confidence_json")
            fc = dict(fc) if isinstance(fc, Mapping) else {}
            fc["si_text_sha256"] = {"confidence": "recorded",
                                    "reason": (f"verified against the snapshot (file sha256 {str(ts.get('file_sha256') or '')[:16]}…) when "
                                               f"seeded; NOT re-verified at this load: {unv[sid]}")}
            payload["field_confidence_json"] = fc
        payload["extra"] = extra
        out.append(R.make_record("si_text", payload, producer, identity.adapter_record_id(payload, "si_text"), written_utc))
    return out
