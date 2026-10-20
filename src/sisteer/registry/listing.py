"""Directory LISTINGS as attested facts (adapter re-anchoring plan commit 7): the one
definition of what a directory's listing is and how it is fingerprinted, shared by the readers' listing inputs
(:class:`adapters.base.ListingInput`: a group's directory listing is an input of the group) and by the store manifest
(:mod:`stores`: a generation attests, per directory under its component, the canonical directory's listing, empty directories
included), so a shard's listing input can be compared with the manifest's attestation of the same directory and never minted from
a staged copy the canonical does not match.

A listing is the sorted ``(name, kind)`` entries of ONE directory -- names as they are, kinds from ``lstat`` (:data:`LISTING_KINDS`:
``dir`` / ``file`` / ``link`` / ``other``; a symlink is ``link`` whatever it points at) -- nothing below it and no attribute of the
entries beyond their kind. Its fingerprint is the sha256 of the sorted ``[name, kind]`` pairs as canonical JSON, so two listings of
the same entries fingerprint the same whatever root they were listed under.
"""
from __future__ import annotations

import os
import stat
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from . import identity

LISTING_KINDS = ("dir", "file", "link", "other")
Entries = Tuple[Tuple[str, str], ...]


def normalize_entries(entries: Iterable[Sequence[str]]) -> Entries:
    """The sorted ``(name, kind)`` tuple a listing is compared and fingerprinted as, whatever sequence shape it arrived in (a manifest's
    JSON lists, a reader's tuples)."""
    return tuple(sorted((str(e[0]), str(e[1])) for e in entries))


def listing_entries(directory: str) -> Entries:
    """The sorted ``(name, kind)`` entries of ``directory`` (``kind`` one of :data:`LISTING_KINDS`, from ``lstat``: a symlink is ``link``
    whatever it points at); ``()`` for a directory that does not exist or cannot be listed."""
    try:
        names = os.listdir(directory)
    except OSError:
        return ()
    out: List[Tuple[str, str]] = []
    for n in names:
        try:
            st = os.lstat(os.path.join(directory, n))
        except OSError:
            out.append((n, "other"))
            continue
        kind = "link" if stat.S_ISLNK(st.st_mode) else "dir" if stat.S_ISDIR(st.st_mode) else "file" if stat.S_ISREG(st.st_mode) else "other"
        out.append((n, kind))
    return tuple(sorted(out))


def listing_fingerprint(entries: Iterable[Sequence[str]]) -> str:
    """sha256 over the sorted ``[name, kind]`` pairs as canonical JSON: the listing's identity, whatever root it was listed under."""
    return identity.sha256_hex(identity.canonical_json([[n, k] for n, k in normalize_entries(entries)]).encode("utf-8"))


def listings_sha256(directories: Mapping[str, Iterable[Sequence[str]]]) -> str:
    """sha256 over ``{relative directory: listing fingerprint}`` as canonical JSON: the identity of a whole tree's attested listings
    (a manifest's ``directories`` field, :mod:`stores`)."""
    return identity.sha256_hex(identity.canonical_json({str(rel): listing_fingerprint(ents) for rel, ents in directories.items()}).encode("utf-8"))


def directories_json(directories: Mapping[str, Iterable[Sequence[str]]]) -> Dict[str, List[List[str]]]:
    """The ``directories`` field's JSON shape: ``{rel: [[name, kind], ...]}``, sorted."""
    return {str(rel): [[n, k] for n, k in normalize_entries(ents)] for rel, ents in sorted(directories.items())}
