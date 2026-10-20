"""The adapter contract shared by every ingest adapter and the runner.

Rules every adapter follows:

* refuse with the path on an unreadable / missing input (:class:`AdapterRefusal`); never default a
  provenance field (``unverified`` + reason in ``field_confidence_json``);
* register every input as an ``artifact`` under the input rules — :func:`check_input`: a non-
  regular file is REFUSED; a symlink is admitted only when the adapter declares it legitimate, and
  then BOTH the link path and the target's sha256 + ``st_dev`` are recorded; a target on tmpfs /
  ramfs is refused outright; a ``_sources.json`` sha that no longer matches the bytes is a refusal; an input whose
  full-resolution stat identity (size, mtime_ns, inode, ctime_ns) equals the pass's :class:`InputHashCache` entry keeps the
  recorded sha without a read (S2), and a file whose identity moves while it is being hashed is hashed once more, then refused;
* mint ``run_id`` ONLY through :func:`identity.run_id`; record ids are deterministic
  (:func:`identity.adapter_record_id`) so a re-run is idempotent;
* declare ``discover()`` roots, ``comparison_group(input)`` (the refresh boundary) and ``rng_scope``
  (``none`` | ``group`` | ``reader_traversal``) so the incremental pass re-runs exactly what a full
  reader run would (S3);
* spell IDENTITY in the canonical (grading-host) form and ADDRESS files locally:
  an :class:`Input` carries ``uri`` / ``path`` (where the adapter opened the file, wherever the read runs) beside
  ``identity_uri`` (the grading-host spelling the pass's :class:`IdentityMap` derives); revision ids, watermarks, ``inputs_json``,
  state and status rows and path-valued comparison groups (:func:`comparison_group_identity`) use the canonical spelling, so a
  read under a relocated home mints the ids the grading host minted. With no map installed (every grading-host pass) ``identity_uri == uri``;
* NEVER READ A STORE WITHOUT A STORE ACCESS SCOPE (spec S2): adapters read on Slurm. A process-level
  :class:`StoreAccessScope` is REQUIRED for any store access -- installed by the cluster shard (``cluster_reads.run_shard``), or by the
  operator's explicit override ``REGISTRY_STORE_READ_OVERRIDE=<reason>`` (logged once per process). Because :class:`Adapter` is a
  Protocol no concrete adapter subclasses, the guard is applied where adapters are CONSTRUCTED (:func:`registered` hands out
  :func:`guarded_factory` wrappers whose instances refuse at the first ``discover()`` / ``read()``), at every dispatch site (the runner's
  pass, ``registry objects scan``, the smoke scripts), in :func:`check_input`, and in the shared directory walkers the adapters route
  through (:func:`store_listdir`, :func:`store_scandir`, :func:`store_walk`, :func:`store_glob`, :func:`store_open`). Without a scope the
  refusal is :class:`GradingHostStoreReadRefused`, by name, BEFORE any byte is read or any directory listed; ``REGISTRY_STORE_AUDIT=<log>``
  (+ ``REGISTRY_STORE_AUDIT_ROOTS``) installs a ``sys.addaudithook`` that records every ``open`` / listing under the named roots, so a
  test or an operator can prove the refusal fired first.
"""
from __future__ import annotations

import contextlib
import glob as _glob
import json
import logging
import os
import stat
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Set, Tuple
from urllib.parse import quote, unquote

from .. import identity
from .. import records as R

log = logging.getLogger(__name__)


class AdapterRefusal(Exception):
    """An adapter cannot read a source: carries the path and why."""

    def __init__(self, path: str, why: str):
        super().__init__(f"{path}: {why}")
        self.path = path
        self.why = why


class InputRefusal(AdapterRefusal):
    """An input rule refused the file (non-regular, tmpfs target, sha mismatch)."""


@dataclass(frozen=True)
class Input:
    uri: str                 # the path as the adapter names it (a symlink's LINK path) -- LOCAL addressing, wherever the read runs
    path: str                # the resolved on-disk path that was hashed -- local addressing
    sha256: str
    size: int
    mtime_utc: str
    role: str = "store"      # artifact kind hint: report | store_root | generations | judge_items | rows | ...
    is_symlink: bool = False
    target_device: Optional[int] = None
    #: the CANONICAL spelling of ``uri`` (the grading host's): what revision ids, watermarks, ``inputs_json``, state rows and the artifact's
    #: primary location are keyed by, so a read under a relocated home (a cluster shard) mints the grading host's ids. Derived by
    #: :func:`check_input` from the pass's :class:`IdentityMap`; equal to ``uri`` when no map is installed (every grading host pass).
    identity_uri: str = ""

    def __post_init__(self) -> None:
        if not self.identity_uri:
            object.__setattr__(self, "identity_uri", self.uri)

    def with_uri(self, uri: str) -> "Input":
        """This input under another spelling -- the ONLY way an adapter aliases an input. The identity FOLLOWS the new spelling (the
        pass's :class:`IdentityMap` applied to it), so an alias minted under a relocated root is the grading host's alias: a
        ``dataclasses.replace(inp, uri=...)`` kept the unsuffixed ``identity_uri`` and every per-group alias of one file shared one
        identity."""
        return replace(self, uri=uri, identity_uri=identity_map().canonical(uri))

    def with_alias(self, suffix: str) -> "Input":
        """The per-group alias ``<uri>#<suffix>`` of one shared input (a manifest declaring several groups reads once per group)."""
        return self.with_uri(f"{self.uri}#{suffix}")

    def artifact_payload(self) -> Dict[str, object]:
        extra: Dict[str, object] = {}
        if self.is_symlink:
            extra = {"realpath": self.path, "target_sha256": self.sha256, "target_device": self.target_device}
        locations: List[Dict[str, object]] = [{"uri": self.identity_uri, "verified_utc": self.mtime_utc, "bytes": self.size}]
        if self.identity_uri != self.uri:
            # the copy the bytes were read from, as a SECOND location: the identity stays the canonical spelling
            locations.append({"uri": self.uri, "verified_utc": self.mtime_utc, "bytes": self.size, "read_root": True})
        return {"artifact_id": self.sha256, "kind": self.role, "uri_primary": self.identity_uri,
                "locations_json": locations, "bytes": self.size, "verified_utc": self.mtime_utc, "extra": extra}


@dataclass
class RecordBatch:
    """Records an adapter yields for one comparison group: deterministic ids, plus the run-scale
    observation keys the group carries (item-scale membership rides the item rows themselves)."""
    source_id: str
    group_id: str
    records: List[Dict[str, object]] = field(default_factory=list)
    observation_keys: List[Tuple[str, Dict[str, object]]] = field(default_factory=list)  # (table, natural key)
    item_keys: List[Dict[str, object]] = field(default_factory=list)                     # item_result keys carried

    def add(self, record_type: str, payload: Mapping[str, object], producer: Mapping[str, object], written_utc: str,
            observe: bool = True) -> Dict[str, object]:
        rec = R.make_record(record_type, payload, producer, identity.adapter_record_id(payload, record_type), written_utc)
        self.records.append(rec)
        table = R.RECORD_TYPE_TABLE.get(record_type)
        if observe and table:
            cols = R.ITEM_MEMBERSHIP_KEY if table == "item_result" else R.NATURAL_KEYS[table]
            key = {c: payload.get(c, _sentinel(c)) for c in cols}
            if table in R.ITEM_SCALE_TABLES:
                if table == "item_result":
                    self.item_keys.append(key)
            else:
                self.observation_keys.append((table, key))
        return rec


def _sentinel(col: str):
    from .. import schema_def as S
    return S.SENTINELS.get(col)


@dataclass
class SourceStateRow:
    """What a source file names NOW under a JSON path (F8) — the runner writes it as a source_state record."""
    source_id: str
    input_uri: str
    json_path: str
    found: int
    path_resolves: int
    current_members: List[str]
    input_sha256: Optional[str]


# ---- directory listings as INPUTS (adapter re-anchoring plan commit 7) --------------------------------------
#: A reader that states what a directory CONTAINS (the entries it saw, read or not: em_misalign's `$.entries`, gsm8k_p200's root-wide `$.dirs`)
#: states it as the content of a tracked INPUT, never as a side observation of a read: the directory's listing -- its entries' names and kinds,
#: sorted -- is a synthetic :class:`Input` discovered beside the files, ``uri = identity_uri = registry://listing/<source_id>/<rel dir>``,
#: ``sha256`` = :func:`listing_fingerprint` of the listing, so a listing change is a revision change (the group re-reads), reverting the
#: listing re-activates the revision whose state row matches, and the row keyed on the listing URI is GROUP-OWNED under the runner's
#: state-key ownership rule (its key is one of the revision's inputs). The runner registers no artifact for it (no file), hands it to the
#: adapter's ``read`` and ``source_states`` with the group's files, watermarks it like any input, and the silent-read rule counts it as no
#: store. ``<source_id>`` is percent-quoted in the URI (the adapters' ``<source>/<name>`` groups); ``<rel dir>`` is the adapter's stable label for the directory.
LISTING_URI_PREFIX = "registry://listing/"
LISTING_ROLE = "listing"
LISTING_MTIME = "1970-01-01T00:00:00.000Z"
# the one definition of a listing and its fingerprint is `sisteer.registry.listing` (the store manifest attests the same shape); re-exported here
from ..listing import LISTING_KINDS, listing_entries, listing_fingerprint  # noqa: E402,F401


def listing_uri(source_id: str, rel: str) -> str:
    """``registry://listing/<source_id>/<rel dir>``; the source id is percent-quoted (``/`` -> ``%2F``, the adapters' ``<source>/<name>`` group
    spellings) so the first ``/`` after it always starts the directory label."""
    if not str(source_id) or ":" not in str(source_id):
        raise AdapterRefusal(str(source_id), "a listing input's source id must be `<adapter>:<group>`")
    if not str(rel) or str(rel).startswith("/"):
        raise AdapterRefusal(str(source_id), f"a listing input's directory label must be a non-empty relative label, not {rel!r}")
    return f"{LISTING_URI_PREFIX}{quote(str(source_id), safe=':')}/{rel}"


def is_listing_uri(uri: object) -> bool:
    return isinstance(uri, str) and uri.startswith(LISTING_URI_PREFIX)


def listing_source_id(uri: str) -> str:
    """The ``<source_id>`` a listing URI names (``registry://listing/<source_id>/<rel dir>`` -> ``<source_id>``, unquoted)."""
    if not is_listing_uri(uri):
        raise ValueError(f"{uri!r} is not a listing URI ({LISTING_URI_PREFIX}<source_id>/<rel dir>)")
    return unquote(uri[len(LISTING_URI_PREFIX):].split("/", 1)[0])


def listing_rel(uri: str) -> str:
    """The ``<rel dir>`` label a listing URI names."""
    if not is_listing_uri(uri):
        raise ValueError(f"{uri!r} is not a listing URI ({LISTING_URI_PREFIX}<source_id>/<rel dir>)")
    return uri[len(LISTING_URI_PREFIX):].split("/", 1)[1]


def listing_group(uri: str, adapter_name: str) -> str:
    """The comparison GROUP a listing URI belongs to, for an adapter's ``comparison_group``: the source id's group part."""
    sid = listing_source_id(uri)
    if not sid.startswith(f"{adapter_name}:"):
        raise ValueError(f"listing URI {uri!r} names source {sid!r}, not one of adapter {adapter_name!r}")
    return sid[len(adapter_name) + 1:]


@dataclass(frozen=True)
class ListingInput(Input):
    """A directory listing as an input (the module note above): ``entries`` are the listing's names (the kinds are in the fingerprint)."""
    entries: Tuple[str, ...] = ()

    @classmethod
    def of(cls, source_id: str, rel: str, entries: Sequence[Tuple[str, str]], path: str = "") -> "ListingInput":
        """``path`` is the listed directory itself (the local spelling), when the entries ARE its whole listing: on a cluster shard the
        runner compares the input's fingerprint with the store manifest's attested listing of that directory (``runner.ClusterListingRefusal``);
        a listing built from a FILTERED or synthesized entry set carries no path and is never a shard's to publish."""
        ents = tuple(sorted((str(n), str(k)) for n, k in entries))
        uri = listing_uri(source_id, rel)
        return cls(uri=uri, path=str(path or ""), sha256=listing_fingerprint(ents), size=len(ents), mtime_utc=LISTING_MTIME, role=LISTING_ROLE,
                   identity_uri=uri, entries=tuple(n for n, _k in ents))

    @classmethod
    def of_directory(cls, source_id: str, rel: str, directory: str) -> "ListingInput":
        return cls.of(source_id, rel, listing_entries(directory), path=directory)


def is_listing_input(inp: Input) -> bool:
    return isinstance(inp, ListingInput) or is_listing_uri(getattr(inp, "identity_uri", None))


def is_synthetic_uri(uri: object) -> bool:
    """Whether a URI names a synthetic input (a listing, an anchor decision): no file, so no artifact and no hash cache."""
    return isinstance(uri, str) and uri.startswith("registry://")


#: The per-row refusal ledger's ``source_state`` row (``json_path`` :data:`ROW_REFUSALS_PATH`) is keyed on the GROUP, never on one of its
#: files: ``input_uri = registry://row-refusals/<source_id>`` (the group's own identity, the way the anchor decision input is
#: ``registry://anchor-decision/<source_id>``), ``input_sha256`` = :func:`inputs_fingerprint` of the read's inputs (their sorted shas). ``source_state`` is keyed
#: by (source_id, input_uri, json_path) and says what the source names NOW, so a ledger keyed on the group's first input moved to a new key
#: whenever that file left or discovery re-ordered, and the old row stood found = 1 with a stale list beside the new one (live: two
#: ``$.row_refusals`` rows per source for bfcl_full / bigcodebench_full / strength_mtdiff groups, 2026-10-03). The group key never moves.
ROW_REFUSALS_URI_PREFIX = "registry://row-refusals/"
ROW_REFUSALS_PATH = "$.row_refusals"


def row_refusals_uri(source_id: str) -> str:
    """The ``$.row_refusals`` ledger's ``input_uri`` for a source: ``registry://row-refusals/<source_id>``."""
    return ROW_REFUSALS_URI_PREFIX + str(source_id)


def is_row_refusals_uri(uri: object) -> bool:
    return isinstance(uri, str) and uri.startswith(ROW_REFUSALS_URI_PREFIX)


def state_owned_paths_of(adapter) -> Tuple[str, ...]:
    """The ``json_path`` spellings an adapter DECLARES group-owned (``state_owned_paths``) beyond the structural rule the runner applies
    (``runner`` module docstring, the state-key ownership rule: a state row keyed on one of the revision's inputs or on the group's
    ledger URI is the group's; anything else -- a directory listing, an absent file, a ruling-spelled path -- is auxiliary). For a row
    whose subject IS one of the group's inputs but whose key spells it another way (``r2_export``'s ``$.arm_operating_point_id`` under the
    rulings' ``~``-spelling of the parquet). Absent means none; a declaration that is not a sequence of non-empty strings is a bug in the
    adapter and refuses BY NAME rather than guarding under a value nobody chose."""
    v = getattr(adapter, "state_owned_paths", None)
    if v is None:
        return ()
    name = getattr(adapter, "name", adapter)
    if isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple)) or not all(isinstance(x, str) and x for x in v):
        raise AdapterRefusal(str(name), f"state_owned_paths must be a sequence of non-empty json_path strings, not {v!r}")
    return tuple(v)


def inputs_fingerprint(inputs: Sequence["Input"]) -> str:
    """sha256 over the SORTED sha256s of the read's inputs (as a canonical JSON list, duplicates kept) -- the ``input_sha256`` of a state row
    whose subject is the group's input SET rather than one file (the ``$.row_refusals`` ledger): the set's CONTENT identity, whatever the
    files are spelled or ordered. Never a path: the same bytes under another root (a test's temporary mirror, a staged copy) fingerprint
    the same, so a pinned emission is reproducible and the row never spells a read root."""
    return identity.sha256_hex(identity.canonical_json(sorted(str(i.sha256) for i in inputs)).encode("utf-8"))


@dataclass
class AdapterContext:
    root: str                                  # the registry root (records/ lives under it)
    now: str
    producer: Dict[str, object]
    legacy_spellings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    conn: object = None                        # read-only sqlite connection, when the adapter needs dimension ids
    #: the pass's anchor currency (:mod:`anchor_current`): loaded ONCE from ``conn`` on the first
    #: :meth:`anchor_currency` call and shared by every contrast-filing adapter of the pass, so a pass whose adapters file no
    #: contrast never pays the fold of ``v_run_current`` and a pass with several pays it once. ``None`` until then.
    anchors: object = None
    #: the comparison group being read (``None`` outside a read) and the anchor decisions ``anchor_currency().choose(...)`` filed for it:
    #: the runner calls :meth:`begin_group` before ``adapter.read`` and, after it, records the log on the revision marker as
    #: ``extra.anchor_decisions`` and fingerprints it into the group's synthetic decision input. One list for the pass, cleared IN PLACE
    #: per group -- the loaded frame holds a reference to this very list (:attr:`anchor_current.AnchorCurrency.decision_log`).
    group: Optional[str] = None
    decision_log: List[Dict[str, object]] = field(default_factory=list)

    def begin_group(self, group: str) -> None:
        """Start the read of ``group``: name it and clear the decision log (in place, so a frame already bound to it sees the new read)."""
        self.group = group
        self.decision_log.clear()

    def anchor_currency(self):
        """The pass's :class:`anchor_current.AnchorCurrency`, loaded on first call from ``conn`` and cached on the context, its
        :attr:`decision_log` bound to this context's. Raises :class:`anchor_current.AnchorCurrencyUnavailable` without a connection
        (``declared_common.smoke`` passes none: the smoke reports it per group instead of crashing) and on a database or sidecar without
        ``v_run_current``; a failed load is not cached, so the next call asks again."""
        if self.anchors is None:
            from .anchor_current import AnchorCurrency, AnchorCurrencyUnavailable
            if self.conn is None:
                raise AnchorCurrencyUnavailable("no registry connection: the anchor currency cannot be loaded (a smoke read, or a pass without a database)")
            self.anchors = AnchorCurrency.load(self.conn)   # type: ignore[arg-type]
        self.anchors.decision_log = self.decision_log      # type: ignore[attr-defined]
        return self.anchors


class Adapter(Protocol):
    name: str
    covers_sources: Tuple[str, ...]
    roots: Tuple[str, ...]
    rng_scope: str
    allow_symlinks: bool
    #: OPTIONAL: the producing session the adapter reads for (the paper's status `owner`); the runner files `registry` when absent.
    owner: str
    #: OPTIONAL: the version of what the adapter's CODE emits — an int or a short string, :data:`DEFAULT_EMISSION_VERSION`
    #: when absent. The runner folds it into every revision id beside the input shas (:func:`runner.revision_id_for`), so a change
    #: to what the adapter PARKS or SPELLS, bumped here, publishes a NEW revision over unchanged inputs and the old head is
    #: superseded; a change that only fixes a refusal needs no bump (the corrected read publishes under the same id).
    EMISSION_VERSION: object

    def discover(self) -> List[Input]: ...
    def comparison_group(self, inp: Input) -> str: ...
    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]: ...
    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]: ...
    #: OPTIONAL: the ``json_path`` spellings of state rows that are the GROUP's although their key is not one of its inputs' identity
    #: URIs (:func:`state_owned_paths_of`; the runner's state-key ownership rule). Absent means none.
    state_owned_paths: Sequence[str]

    # -- OPTIONAL status hooks: the runner files one `source_status` row per (paper source string, group) after each
    #    accepted read. Both default in the runner (:func:`status_extra_of`, :func:`status_sources_of`) when an adapter lacks them.
    def status_extra(self, group: str) -> Mapping[str, object]:
        """The FACTS of the last ``read`` of ``group`` — what the adapter computed off the producer's records and the paper's
        reader-side guards consume (ruled operating points, dose tables, the record's roster, engines, `missing` / `torn`
        lists). Stored verbatim in ``source_status.extra_json``; JSON-serialisable, finite numbers only. Default ``{}``."""
        ...

    def status_sources(self, group: str) -> Tuple[str, ...]:
        """The paper source strings ``group`` reports under. Default: the group itself when it IS one of ``covers_sources``
        (an adapter whose groups are its sources), else every ``covers_sources`` entry, else the adapter's name."""
        ...

    #: OPTIONAL: the reader's declaration that a group carries NO record BY DESIGN — a facts-only group whose product is the status
    #: row's facts or the marker's ``source_state`` rows, an artifact-only group whose inputs the runner registers and nothing more:
    #: ``True`` for every group of the adapter, or a callable ``(group) -> bool`` asked per group. The runner's emitted-nothing guard
    #: (:func:`runner.emitted_nothing_unexplained`) never fires on a declared group; WITHOUT the declaration a group that read store
    #: inputs, emitted no record and recorded no ``refuse`` / ``missing`` / ``torn`` / ``held`` entry is filed ``ok = 0`` (fail loud).
    emits_no_runs: object
    #: OPTIONAL: the share of the previous accepted revision's inputs a read may drop before the runner REFUSES it as
    #: ``inputs_dropped_unexplained`` (:func:`inputs_dropped_threshold_of`; :data:`INPUTS_DROPPED_THRESHOLD` when absent). A number in
    #: [0, 1], or a callable asked per comparison group.
    inputs_dropped_threshold: object
    #: OPTIONAL: inputs a read drops ON PURPOSE (:func:`inputs_retired_of`): input URIs or ``fnmatch`` globs over them, or a callable
    #: asked per comparison group. A dropped input covered here does not count toward the threshold — a deliberate retirement is
    #: declared, never inferred from the read. A RENAME is a retirement: an adapter whose code changes which file it reads (a new
    #: store file name under an ``EMISSION_VERSION`` bump, a moved tree) declares the OLD names here in the same commit, or its next
    #: read of every source that lost more than the threshold share refuses until it does.
    inputs_retired: object
    #: OPTIONAL: run ids a read drops ON PURPOSE (:func:`runs_retired_of`): ``{"run_ids": [<run id or fnmatch glob>, ...] | "*",
    #: "reason": <why>}``, or a callable asked per comparison group. The runner writes it on the marker as ``extra.declared_run_drops``
    #: (``harvest.DECLARED_RUN_DROPS_KEY``); WITHOUT it a revision that no longer names a run id the source's accepted head carries is
    #: HELD by the harvester -- published, not accepted, the head current -- until a declaration or ``registry revision accept``
    #: releases it. A reader whose change retires rows on purpose -- a stricter row rule, a run-id scheme change under an
    #: ``EMISSION_VERSION`` bump -- declares them here in the same commit, the way a renamed input goes under ``inputs_retired``.
    runs_retired: object


def status_extra_of(adapter, group: str) -> Dict[str, object]:
    """``adapter.status_extra(group)`` as a plain dict, ``{}`` for an adapter without the hook (never a refusal: a status
    row with no facts is still a status row)."""
    fn = getattr(adapter, "status_extra", None)
    if not callable(fn):
        return {}
    out = fn(group)
    return dict(out) if isinstance(out, Mapping) else {}


def status_sources_of(adapter, group: str) -> Tuple[str, ...]:
    """The paper source strings a group's status rows are filed under (see :meth:`Adapter.status_sources`)."""
    fn = getattr(adapter, "status_sources", None)
    if callable(fn):
        out = tuple(str(s) for s in fn(group))
        if out:
            return out
    covers = tuple(str(s) for s in (getattr(adapter, "covers_sources", ()) or ()))
    if group in covers:
        return (group,)
    return covers or (str(adapter.name),)


def emits_no_runs_of(adapter, group: str) -> bool:
    """Whether the adapter declares ``group`` as emitting no record by design (:attr:`Adapter.emits_no_runs`): a callable is asked
    per group, a bare truthy value covers every group, an absent attribute declares nothing."""
    marker = getattr(adapter, "emits_no_runs", None)
    if marker is None:
        return False
    if callable(marker):
        return bool(marker(group))
    return bool(marker)


def owner_of(adapter) -> str:
    """The adapter's declared owner, ``registry`` when it declares none (the paper's own fallback spelling)."""
    owner = getattr(adapter, "owner", None)
    return str(owner) if isinstance(owner, str) and owner.strip() else "registry"


#: The share of the previous accepted revision's inputs a read may drop, undeclared, before the runner refuses the read
#: (:func:`runner.inputs_dropped_comparison`): 0.10. The guard exists for the 2026-09-21 class: the six `relu_paired` reads
#: that withdrew the paper's over-refusal rows had dropped 14.6-25.0 % of their sources' inputs (a store tree absent from discovery
#: for one read), and every one of them must refuse — at 0.20 two of the six would have published. The price is a handful more
#: historical refusals, each an ``inputs_retired`` declaration away, which is the loud outcome wanted.
INPUTS_DROPPED_THRESHOLD = 0.10


def inputs_dropped_threshold_of(adapter, group: str) -> float:
    """The share of the previous accepted revision's inputs a read of ``group`` may drop before it is refused
    (:attr:`Adapter.inputs_dropped_threshold`): a callable is asked per group, a number covers every group, an absent
    attribute means :data:`INPUTS_DROPPED_THRESHOLD`. A declaration that is not a number in [0, 1] is a bug in the adapter and
    refuses BY NAME (the way :func:`emission_version_of` does) rather than guarding under a value nobody chose."""
    v = getattr(adapter, "inputs_dropped_threshold", None)
    if v is None:
        return INPUTS_DROPPED_THRESHOLD
    if callable(v):
        v = v(group)
        if v is None:
            return INPUTS_DROPPED_THRESHOLD
    name = getattr(adapter, "name", adapter)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
        raise ValueError(f"{name}: inputs_dropped_threshold must be a number in [0, 1], not {v!r}")
    return float(v)


def inputs_retired_of(adapter, group: str) -> Tuple[str, ...]:
    """The input URIs or ``fnmatch`` globs a read of ``group`` drops on purpose (:attr:`Adapter.inputs_retired`): a callable is
    asked per group, a sequence of strings covers every group, an absent attribute declares nothing. Anything else — a bare
    string (one glob is a one-element list, never a string a loop would spell out), a non-string member — refuses BY NAME."""
    v = getattr(adapter, "inputs_retired", None)
    if v is None:
        return ()
    if callable(v):
        v = v(group)
        if v is None:
            return ()
    name = getattr(adapter, "name", adapter)
    if isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple, set, frozenset)) or not all(isinstance(x, str) and x for x in v):
        raise ValueError(f"{name}: inputs_retired must be a list of input URIs or globs (strings), not {v!r}")
    return tuple(sorted(set(v)))


def runs_retired_of(adapter, group: str) -> Optional[Dict[str, object]]:
    """The run ids a read of ``group`` drops ON PURPOSE (:attr:`Adapter.runs_retired`), as the declaration the runner writes on the
    marker (``extra.declared_run_drops``, ``harvest.DECLARED_RUN_DROPS_KEY``): ``{"run_ids": [<run id or fnmatch glob>, ...] | "*",
    "reason": <why>}`` -- a dict covers every group, a callable is asked per group (None declares nothing), an absent attribute
    declares nothing. Anything else -- a bare list with no reason, an empty reason, a ``run_ids`` string other than ``"*"``, a
    non-string member -- refuses BY NAME, the way :func:`inputs_retired_of` does, rather than guarding under a shape nobody chose."""
    v = getattr(adapter, "runs_retired", None)
    if v is None:
        return None
    if callable(v):
        v = v(group)
        if v is None:
            return None
    name = getattr(adapter, "name", adapter)
    if not isinstance(v, Mapping):
        raise ValueError(f"{name}: runs_retired must be a dict {{'run_ids': [...] | '*', 'reason': ...}}, not {v!r}")
    ids, reason = v.get("run_ids"), v.get("reason")
    if ids != "*" and not (isinstance(ids, (list, tuple)) and ids and all(isinstance(x, str) and x for x in ids)):
        raise ValueError(f"{name}: runs_retired.run_ids must be '*' or a non-empty list of run ids or globs, not {ids!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{name}: runs_retired.reason must be a non-empty string, not {reason!r}")
    return {"run_ids": "*" if ids == "*" else sorted(set(ids)), "reason": reason.strip()}


#: The emission version every adapter carries until it declares one. It is the version every revision published before
#: the rule existed was minted under, so :func:`runner.revision_id_for` folds NOTHING for it: an adapter that never bumped
#: reproduces its published ids byte for byte (a ``--reread`` of an unchanged head still drops the read's records).
DEFAULT_EMISSION_VERSION = 1

#: the longest ``EMISSION_VERSION`` string admitted: a version is a label, never a changelog
EMISSION_VERSION_MAX_LEN = 64


def emission_version_of(adapter) -> str:
    """The adapter's ``EMISSION_VERSION`` as the string the revision identity folds in: ``'1'`` when it declares none, else
    the ``str()`` of its int or short string. A declaration that is neither — a bool, a float, an empty or overlong string, a
    container — is a bug in the adapter and refuses BY NAME rather than minting ids under a spelling nobody chose."""
    v = getattr(adapter, "EMISSION_VERSION", DEFAULT_EMISSION_VERSION)
    name = getattr(adapter, "name", adapter)
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise TypeError(f"{name}: EMISSION_VERSION must be an int or a short string, not {type(v).__name__} ({v!r})")
    out = str(v).strip()
    if not out or len(out) > EMISSION_VERSION_MAX_LEN or any(ch.isspace() for ch in out):
        raise ValueError(f"{name}: EMISSION_VERSION must be a non-empty label of at most {EMISSION_VERSION_MAX_LEN} characters "
                         f"without whitespace, not {v!r}")
    return out


# --------------------------------------------------------------------------------------------
# input rules
# --------------------------------------------------------------------------------------------

_VOLATILE_FSTYPES = ("tmpfs", "ramfs", "devtmpfs")


def _mount_table() -> List[Tuple[str, str, str]]:
    """``(mount point, mount point as a directory prefix, fstype)`` per line of ``/proc/mounts``, LONGEST mount point first so the
    first match is the longest — read once per pass (:class:`_PassScope`; it was re-read for every one of a pass's ~27k inputs),
    per call outside one; empty when it cannot be read."""
    scope = _SCOPE
    if scope is not None and scope.mounts is not None:
        return scope.mounts
    mounts: List[Tuple[str, str, str]] = []
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    mounts.append((parts[1], parts[1].rstrip("/") + "/", parts[2]))
    except OSError:
        pass
    mounts.sort(key=lambda m: len(m[0]), reverse=True)      # stable: equal lengths keep /proc/mounts order, as the scan did
    if scope is not None:
        scope.mounts = mounts
    return mounts


def _fstype_of_real(real: str) -> Optional[str]:
    """The filesystem type of the mount holding the already-resolved ``real`` (longest prefix match), or None."""
    for mp, prefix, fstype in _mount_table():
        if real == mp or real.startswith(prefix) or mp == "/":
            return fstype
    return None


def mount_fstype(path: str) -> Optional[str]:
    """The filesystem type of the mount holding ``path`` (longest prefix match in /proc/mounts)."""
    return _fstype_of_real(os.path.realpath(path))


def _mtime_utc(st: os.stat_result) -> str:
    return R._fmt_utc(datetime.fromtimestamp(st.st_mtime, tz=timezone.utc))


# --------------------------------------------------------------------------------------------
# the input hash cache (S2): sha256 by full-resolution stat identity, so an unchanged input is never read again
# --------------------------------------------------------------------------------------------

#: The sidecar under the registry root's ``state/`` where the cache persists between passes (loaded and saved by the runner, once
#: per pass, atomically). JSON, like the harvester's ``file_stat.json``: at the live inventory (~27k distinct input paths,
#: ~300 bytes each) it is ~8 MB, loaded and saved in well under a second beside a pass that used to hash ~20 GB — and the
#: harvester's own sidecar carries 434k entries in the same shape.
INPUT_HASH_CACHE_NAME = "input_hash_cache.json"

#: The version of the sidecar's shape; a file of another version is dropped, never reinterpreted.
INPUT_HASH_CACHE_VERSION = 1

#: The stat fields that make an input's IDENTITY at full resolution: the harvester's ``file_stat.json`` rule (size, mtime_ns,
#: inode; ``harvest.new_or_changed_files``) plus ``ctime_ns``, so a same-size edit that restores the mtime (a ``cp -p`` or
#: ``touch -r`` over the file) still reads as changed — any write, rename or replacement moves ctime or the inode. The cache is
#: NEVER seeded from ``source_watermark``: its ``mtime_utc`` is ms-truncated and it carries no inode or ctime, so the first pass
#: after this lands hashes every input once (the accepted cost) and later passes reuse what that pass recorded.
INPUT_STAT_FIELDS: Tuple[str, ...] = ("size", "mtime_ns", "ino", "ctime_ns")


def stat_identity(st: os.stat_result) -> Dict[str, int]:
    """The :data:`INPUT_STAT_FIELDS` of a stat result, as the cache stores and compares them."""
    return {f: int(getattr(st, "st_" + f)) for f in INPUT_STAT_FIELDS}


class InputHashCache:
    """sha256 by REAL path and full-resolution stat identity (:data:`INPUT_STAT_FIELDS`), consulted by :func:`check_input` for
    every input a pass discovers: an entry whose four fields equal the file's stat now supplies the sha without a read; anything
    else is hashed (:func:`hash_input`) and recorded. Keyed by the resolved path, so a shared file declared under several
    ``#<group>`` aliases is hashed once per pass, not once per alias. ``bypass`` (``adapters run --full``) reuses nothing and
    re-hashes bytes — the documented answer to an in-place edit that preserved every stat field — while still recording what it
    hashed. ``reused`` / ``hashed`` are the pass's counters (the summary line reports them)."""

    def __init__(self, entries: Optional[Mapping[str, Mapping[str, object]]] = None):
        self.entries: Dict[str, Dict[str, object]] = {}
        for real, ent in (entries or {}).items():
            if isinstance(ent, Mapping) and all(isinstance(ent.get(f), int) for f in INPUT_STAT_FIELDS) and isinstance(ent.get("sha256"), str):
                self.entries[str(real)] = {**{f: int(ent[f]) for f in INPUT_STAT_FIELDS}, "sha256": str(ent["sha256"])}
        self.bypass = False
        self.seen: Set[str] = set()
        self.reused = 0
        self.hashed = 0
        self.dirty = False

    @staticmethod
    def path(root: str) -> str:
        return os.path.join(root, "state", INPUT_HASH_CACHE_NAME)

    @classmethod
    def load(cls, root: str) -> "InputHashCache":
        """The sidecar of ``root``, or an empty cache when there is none, it is unreadable, or its version is not this code's
        (a stale or torn sidecar costs one hashing pass, never a wrong sha: every entry is re-verified by stat before reuse)."""
        data = R.read_json_or_none(cls.path(root))
        if not isinstance(data, dict) or data.get("version") != INPUT_HASH_CACHE_VERSION or not isinstance(data.get("entries"), dict):
            return cls()
        return cls(data["entries"])

    def save(self, root: str) -> None:
        """Write the sidecar whole (a unique temp name, fsync, rename: :func:`records.write_json_atomic`)."""
        R.write_json_atomic(self.path(root), {"version": INPUT_HASH_CACHE_VERSION, "fields": list(INPUT_STAT_FIELDS), "entries": self.entries},
                            indent=None, sort_keys=True, separators=(",", ":"))
        self.dirty = False

    def lookup(self, real: str, st: os.stat_result) -> Optional[str]:
        """The recorded sha256 of ``real`` when its entry's identity equals ``st``'s; None when there is none, it moved, or the
        cache is bypassed."""
        self.seen.add(real)
        if self.bypass:
            return None
        ent = self.entries.get(real)
        if ent is None or any(ent[f] != v for f, v in stat_identity(st).items()):
            return None
        self.reused += 1
        return str(ent["sha256"])

    def record(self, real: str, st: os.stat_result, sha256: str) -> None:
        """Remember the sha of ``real`` under the identity ``st`` — the stat taken AFTER the hash and found equal to the one before it."""
        self.seen.add(real)
        self.hashed += 1
        ent: Dict[str, object] = dict(stat_identity(st), sha256=str(sha256))
        if self.entries.get(real) != ent:
            self.entries[real] = ent
            self.dirty = True

    def admit(self, uri: str, real: str) -> None:
        """The seam at which a cache may REFUSE an input by its canonical spelling and its real path, before any byte is read
        (:func:`check_input` calls it right after :func:`resolve_input`): the grading host cache admits everything; the cluster read's
        ``cluster_reads.GenerationHashCache`` refuses an input spelled under a pinned component's base that does not address that
        component's pinned generation. Raise :class:`InputRefusal` to refuse."""
        return None

    def prune(self) -> int:
        """Drop the entries of files that are no longer on disk; return how many. An entry NOT consulted this pass is kept while
        its file exists: a pass over one adapter (``--adapter A``, the way a landing window runs them) must not evict every other
        adapter's entries, or the next full pass would hash everything again."""
        gone = [real for real in self.entries if real not in self.seen and not os.path.lexists(real)]
        for real in gone:
            del self.entries[real]
        if gone:
            self.dirty = True
        return len(gone)


# --------------------------------------------------------------------------------------------
# canonical identity beside local addressing (registry-slurm-fast-windows Phase 3)
# --------------------------------------------------------------------------------------------

#: The environment variable a relocated read sets to the grading host home (``/home/<user>``): :meth:`IdentityMap.from_environ` maps the
#: process's own home (``$HOME``, where every adapter root resolves at import) onto it. Unset on the grading host: no map, identity == uri.
URI_HOME_ENV = "SISTEER_REGISTRY_URI_HOME"


@dataclass(frozen=True)
class IdentityMap:
    """``local prefix -> canonical prefix`` pairs (longest local prefix first). :meth:`canonical` rewrites a path (or a ``path#alias``
    URI, or a path-valued comparison group) that IS a local prefix or lies under one into the canonical spelling; :meth:`local` is the
    inverse -- the local side's spelling of a canonical path, which under the home tree a shard lays out is the pinned
    GENERATION, a spelling the adapter never addresses through (a prior-only group is spelled by ``runner.prior_group_spelling``, from
    the adapter's roots). Empty = the identity function, byte for byte."""
    pairs: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def build(cls, mapping: Mapping[str, str]) -> "IdentityMap":
        pairs: List[Tuple[str, str]] = []
        for local, canonical in mapping.items():
            l_, c_ = os.path.abspath(str(local)).rstrip("/"), str(canonical).rstrip("/")
            if not l_ or not c_ or not c_.startswith("/"):
                raise ValueError(f"identity map: {local!r} -> {canonical!r} is not a pair of absolute prefixes")
            if l_ != c_:
                pairs.append((l_, c_))
        pairs.sort(key=lambda pr: len(pr[0]), reverse=True)
        return cls(tuple(pairs))

    @classmethod
    def from_environ(cls, environ: Optional[Mapping[str, str]] = None) -> "IdentityMap":
        """``{$HOME: $SISTEER_REGISTRY_URI_HOME}`` when the variable names a home other than this process's; else empty."""
        env = os.environ if environ is None else environ
        canonical_home = env.get(URI_HOME_ENV)
        if not canonical_home:
            return cls()
        return cls.build({os.path.expanduser("~"): canonical_home})

    def __bool__(self) -> bool:
        return bool(self.pairs)

    @staticmethod
    def _swap(s: str, a: str, b: str) -> Optional[str]:
        if s == a:
            return b
        if s.startswith(a + "/"):
            return b + s[len(a):]
        return None

    def canonical(self, s: str) -> str:
        for local, canonical in self.pairs:
            out = self._swap(s, local, canonical)
            if out is not None:
                return out
        return s

    def local(self, s: str) -> str:
        """The local side's spelling of a canonical path: the inverse of :meth:`canonical` -- and NOT INJECTIVE. Two pairs may share a
        canonical prefix (``cluster_reads.run_shard`` maps two pinned components' generations onto ONE home-relative base when they
        share it, the base laid out as a real directory with one link per entry), and this returns whichever pair sorts first, so the
        answer is not the spelling anything resolves. A caller must therefore never use it to spell a comparison group for an adapter:
        under the home tree a shard lays out the adapter addresses through the canonical spelling and this is the pinned
        generation, a spelling no root and no declaration knows -- ``runner.prior_group_spelling`` is the rule, from the adapter's
        roots. Its ONE production caller is that rule, which takes this answer only when it lies under one of the adapter's declared
        roots (the re-pointed-root shape) and keeps the identity otherwise, so an arbitrary pair's spelling is never addressed through;
        ``runner.plan_groups`` called it directly for every prior-only group until the spelling defect of 2026-10-08."""
        for local, canonical in sorted(self.pairs, key=lambda pr: len(pr[1]), reverse=True):
            out = self._swap(s, canonical, local)
            if out is not None:
                return out
        return s

    def canonical_text(self, s: str) -> str:
        """Free text (a status ``detail``, a refusal, a ``why``) with every mapped local prefix rewritten wherever it appears -- a local
        prefix is a read root nothing else spells. Identities use :meth:`canonical` (whole-prefix), never this."""
        for local, canonical in self.pairs:
            if local in s:
                s = s.replace(local + "/", canonical + "/")
                s = canonical if s == local else s
        return s

    def canonical_source_id(self, source_id: str) -> str:
        """``<adapter>:<group>`` with a path-valued group canonicalized (the group is everything after the first colon)."""
        name, sep, group = str(source_id).partition(":")
        return f"{name}{sep}{self.canonical(group)}" if sep else str(source_id)

    def canonical_deep(self, obj):
        """Every string of a JSON-like value that spells a mapped local prefix, rewritten; everything else returned as is. Applied to the
        records a relocated read emits, so no record ever spells the read root (the copy is on the marker's ``extra.read_copy``). A value
        nothing under which changed is returned AS THE SAME OBJECT, so a caller can tell an untouched payload by identity (and skip
        re-minting its id)."""
        if not self.pairs:
            return obj
        if isinstance(obj, str):
            return self.canonical_text(obj)
        if isinstance(obj, dict):
            out = {k: self.canonical_deep(v) for k, v in obj.items()}
            return obj if all(out[k] is v for k, v in obj.items()) else out
        if isinstance(obj, (list, tuple)):
            out = [self.canonical_deep(v) for v in obj]
            if all(a is b for a, b in zip(out, obj)):
                return obj
            return out if isinstance(obj, list) else tuple(out)
        return obj

    def as_dict(self) -> Dict[str, str]:
        return dict(self.pairs)


@dataclass
class _PassScope:
    """What one pass shares across every :func:`check_input` call (installed by :func:`using_hash_cache`): the hash cache, the
    mount table read once, and the resolved path of every input directory — ``realpath`` walks every component of a path and
    the inputs sit ~13 deep, so a pass that resolved each of ~27k inputs on its own spent seconds in ``lstat`` alone."""
    cache: Optional[InputHashCache]
    mounts: Optional[List[Tuple[str, str, str]]] = None
    real_dirs: Dict[str, str] = field(default_factory=dict)
    #: the pass's identity map (canonical spelling beside local addressing; registry-slurm-fast-windows Phase 3): empty = identity
    identity: IdentityMap = field(default_factory=IdentityMap)



#: The scope of the running pass; None outside one (a bare ``check_input`` call, the tests' fixtures): every input is then
#: hashed and every path resolved on its own.
_SCOPE: Optional[_PassScope] = None


@contextlib.contextmanager
def using_hash_cache(cache: Optional[InputHashCache]) -> Iterator[Optional[InputHashCache]]:
    """Install ``cache`` for every :func:`check_input` call inside the block — the runner wraps one whole pass (discovery and
    reads) in it; the block also shares the mount table and the inputs' resolved directories (:class:`_PassScope`)."""
    global _SCOPE
    previous = _SCOPE
    _SCOPE = _PassScope(cache, identity=(previous.identity if previous is not None else IdentityMap()))
    try:
        yield cache
    finally:
        _SCOPE = previous


def identity_map() -> IdentityMap:
    """The identity map installed for the current pass (empty -- the identity function -- outside a relocated pass)."""
    return _SCOPE.identity if _SCOPE is not None else IdentityMap()


@contextlib.contextmanager
def using_identity_map(imap: Optional[IdentityMap]) -> Iterator[IdentityMap]:
    """Install ``imap`` for every :func:`check_input` call and every identity derivation inside the block -- the runner wraps one whole
    pass (discovery and reads) in it, beside :func:`using_hash_cache` (which carries it into the pass scope it opens)."""
    global _SCOPE
    previous = _SCOPE
    imap = imap if imap is not None else IdentityMap()
    if previous is None:
        _SCOPE = _PassScope(None, identity=imap)
    else:
        _SCOPE = _PassScope(previous.cache, previous.mounts, previous.real_dirs, identity=imap)
    try:
        yield imap
    finally:
        _SCOPE = previous


def comparison_group_identity(group: str) -> str:
    """The canonical identity of a comparison group: the pass's map applied when the group is a path under a mapped root (sampled_static's
    groups are absolute directories), the group itself otherwise. Source ids and revision ids are minted from this, never from the
    local spelling, so relocation mints no new source."""
    return identity_map().canonical(str(group))


def hash_input(real: str, st: os.stat_result, path: Optional[str] = None) -> Tuple[str, os.stat_result]:
    """sha256 of ``real``'s bytes bound to ONE stat identity: the file is stat'ed again after the read and, when its identity moved
    against ``st`` (a writer landed between the stat and the hash), it is hashed ONCE more against the new identity; a second move
    refuses by name — the bytes cannot be bound to an identity, so nothing is recorded. Returns ``(sha256, stat)`` with the
    stat the sha is bound to."""
    for _attempt in range(2):
        sha = R.file_sha256(real)
        after = os.stat(real)
        if stat_identity(after) == stat_identity(st):
            return sha, st
        st = after
    raise InputRefusal(path or real, f"changed while being hashed ({real}: its size, mtime, inode or ctime moved twice during the read); "
                                     "read it again once its writer is done (D12)")


def _real_dir(scope: _PassScope, d: str) -> str:
    """``os.path.realpath(d)`` for a directory, memoized per pass and built from its parent's: one ``lstat`` per directory the pass
    meets for the first time (a link is resolved by ``realpath`` itself), where ``realpath`` walks every component of every input
    — 16 ``lstat`` per input at the live depth, most of a quiet pass."""
    real = scope.real_dirs.get(d)
    if real is not None:
        return real
    head, tail = os.path.split(d)
    if not head or not tail or tail in (os.curdir, os.pardir):
        real = os.path.realpath(d)
    else:
        try:
            is_link = stat.S_ISLNK(os.lstat(d).st_mode)
        except OSError:
            is_link = True                                    # absent or unreadable: let realpath resolve what it can
        real = os.path.realpath(d) if is_link else os.path.join(_real_dir(scope, head), tail)
    scope.real_dirs[d] = real
    return real


def _real_path(path: str, is_link: bool) -> str:
    """``os.path.realpath(path)`` — through the pass's directory memo (:func:`_real_dir`) when the leaf is not a link."""
    scope = _SCOPE
    head, tail = os.path.split(path)
    if scope is None or is_link or not head or not tail or tail in (os.curdir, os.pardir):
        return os.path.realpath(path)
    return os.path.join(_real_dir(scope, head), tail)


def resolve_input(path: str, allow_symlink: bool = False) -> Tuple[str, bool, os.stat_result]:
    """The input rules that need no read: the input exists, is a regular file (through a symlink only when the adapter admits
    them) and does not resolve to a volatile mount. Returns ``(real path, is_symlink, stat)`` — the one ``lstat`` a non-link
    needs, or the target's stat through a link; refuses by name otherwise."""
    try:
        lst = os.lstat(path)
    except OSError:
        raise AdapterRefusal(path, "input missing") from None
    is_link = stat.S_ISLNK(lst.st_mode)
    if is_link and not allow_symlink:
        raise InputRefusal(path, "input is a symlink and this adapter does not admit symlinks (D12)")
    real = _real_path(path, is_link)
    st: Optional[os.stat_result] = lst
    if is_link:
        try:
            st = os.stat(real)
        except OSError:
            st = None
    if st is None or not stat.S_ISREG(st.st_mode):
        raise InputRefusal(path, "input is not a regular file (D12)")
    fstype = _fstype_of_real(real)
    if fstype in _VOLATILE_FSTYPES:
        raise InputRefusal(path, f"input resolves to a {fstype} mount ({real}); a volatile target is refused (D12)")
    return real, is_link, st


def check_input(path: str, role: str = "store", allow_symlink: bool = False,
                expected_sha256: Optional[str] = None) -> Input:
    """Apply the input rules and return the hashed :class:`Input`; refuse by name otherwise. The sha comes from the pass's
    :class:`InputHashCache` when the file's full-resolution stat identity equals the recorded one (no read), else from
    :func:`hash_input` (a read, bound to one identity, recorded for the next pass); ``expected_sha256`` is checked against the
    resulting sha in EVERY case, cache hit included. The :class:`Input` names the file as the adapter spelled it (``uri``) and as
    it was hashed (``path``) -- local addressing -- and its canonical identity (``identity_uri``) through the pass's :class:`IdentityMap`."""
    require_store_access("hashing an input", path)     # the S2 chokepoint (C6): no store byte without a store access scope
    real, is_link, st = resolve_input(path, allow_symlink)
    cache = _SCOPE.cache if _SCOPE is not None else None
    if cache is not None:
        cache.admit(path, real)                        # the escape refusal (C5): by spelling and real path, before any read
    sha = cache.lookup(real, st) if cache is not None else None
    if sha is None:
        sha, st = hash_input(real, st, path)
        if cache is not None:
            cache.record(real, st, sha)
    if expected_sha256 is not None and not sha.startswith(expected_sha256):
        raise InputRefusal(path, f"recorded sha {expected_sha256[:12]}… no longer matches the bytes on disk ({sha[:12]}…) (D12)")
    return Input(uri=path, path=real, sha256=sha, size=st.st_size, mtime_utc=_mtime_utc(st), role=role,
                 is_symlink=is_link, target_device=st.st_dev if is_link else None, identity_uri=identity_map().canonical(path))


def watermark_of(inp: Input) -> Tuple[str, int, str]:
    return (inp.sha256, inp.size, inp.mtime_utc)


# --------------------------------------------------------------------------------------------
# adapter registry
# --------------------------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[[], Adapter]] = {}


def register(name: str, factory: Callable[[], Adapter]) -> None:
    _REGISTRY[name] = factory


def registered() -> Dict[str, Callable[[], Adapter]]:
    """Every registered adapter's factory, each wrapped by :func:`guarded_factory`: the instance a factory
    returns refuses its first ``discover()`` / ``read()`` without a :class:`StoreAccessScope`. ``load_all`` and every caller that bypasses
    it (the smoke scripts, the backfill, a test) get the same wrapper; construction itself is never refused (``stores coverage`` constructs
    every adapter to read its declared roots and opens nothing)."""
    return {name: guarded_factory(name, factory) for name, factory in _REGISTRY.items()}


def registered_raw() -> Dict[str, Callable[[], Adapter]]:
    """The factories as registered, unwrapped (what the registration tests read of the two conventions)."""
    return dict(_REGISTRY)


# --------------------------------------------------------------------------------------------
# the store access scope (spec S2: no grading host store read at any entry point)
# --------------------------------------------------------------------------------------------

#: the operator's explicit override: ``REGISTRY_STORE_READ_OVERRIDE=<reason>`` admits every store read of the process and is logged once
STORE_OVERRIDE_ENV = "REGISTRY_STORE_READ_OVERRIDE"
#: the audit instrument: ``REGISTRY_STORE_AUDIT=<log file>`` records every ``open`` / ``os.listdir`` / ``os.scandir`` / ``glob.glob`` under
#: the roots ``REGISTRY_STORE_AUDIT_ROOTS`` names (``:``-separated; default this home) as one JSON line each
STORE_AUDIT_ENV = "REGISTRY_STORE_AUDIT"
STORE_AUDIT_ROOTS_ENV = "REGISTRY_STORE_AUDIT_ROOTS"
#: the audit events that are a store access
STORE_AUDIT_EVENTS: Tuple[str, ...] = ("open", "os.listdir", "os.scandir", "glob.glob")
#: the scope kinds: the cluster shard's, the operator's override (environment or explicit), an explicit operator scope a caller installs
STORE_SCOPE_KINDS: Tuple[str, ...] = ("shard", "override", "operator")


class GradingHostStoreReadRefused(AdapterRefusal):
    """A store access on a host with no :class:`StoreAccessScope` installed: adapters read on Slurm (``registry cluster-reads shard``);
    a deliberate read here sets ``REGISTRY_STORE_READ_OVERRIDE=<reason>`` (logged). Raised by name BEFORE any byte is read."""


@dataclass(frozen=True)
class StoreAccessScope:
    """WHO may read a store in this process and why: ``kind`` one of :data:`STORE_SCOPE_KINDS`, ``reason`` the human sentence (the
    shard names its operation, the override its operator's reason), ``window`` the read operation's name when there is one."""
    kind: str
    reason: str
    window: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in STORE_SCOPE_KINDS:
            raise ValueError(f"store access scope kind {self.kind!r} is not one of {STORE_SCOPE_KINDS}")
        if not str(self.reason).strip():
            raise ValueError("a store access scope needs a reason")

    def describe(self) -> str:
        return f"{self.kind}: {self.reason}" + (f" (operation {self.window})" if self.window else "")


#: the scope installed explicitly (:func:`using_store_access_scope`); while ``_SCOPE_EXPLICIT`` is set the environment override is NOT
#: consulted, so a block that installs ``None`` is an UNSCOPED host whatever the environment says (the refusal tests, the refusal drill)
_STORE_SCOPE: Optional[StoreAccessScope] = None
_SCOPE_EXPLICIT = False
_OVERRIDE_LOGGED: Set[str] = set()


def environment_override(environ: Optional[Mapping[str, str]] = None) -> Optional[StoreAccessScope]:
    """The operator's override scope from ``REGISTRY_STORE_READ_OVERRIDE=<reason>``, or None (an empty value is no override)."""
    reason = str((environ if environ is not None else os.environ).get(STORE_OVERRIDE_ENV) or "").strip()
    return StoreAccessScope("override", reason) if reason else None


def store_access_scope() -> Optional[StoreAccessScope]:
    """The scope in force: the one installed by :func:`using_store_access_scope` (None inside such a block = no scope), else the
    environment override (logged once per process and reason), else None."""
    if _SCOPE_EXPLICIT:
        return _STORE_SCOPE
    override = environment_override()
    if override is not None and override.reason not in _OVERRIDE_LOGGED:
        _OVERRIDE_LOGGED.add(override.reason)
        log.warning("%s=%r: store reads admitted on this host by operator override (registry-slurm-only S2)", STORE_OVERRIDE_ENV, override.reason)
    return override


@contextlib.contextmanager
def using_store_access_scope(scope: Optional[StoreAccessScope]) -> Iterator[Optional[StoreAccessScope]]:
    """Install ``scope`` for every store access inside the block -- the cluster shard wraps its pass in one. ``None`` installs NO scope
    and hides the environment override for the block: the way a test or a drill makes this process an unscoped host."""
    global _STORE_SCOPE, _SCOPE_EXPLICIT
    previous, previous_explicit = _STORE_SCOPE, _SCOPE_EXPLICIT
    _STORE_SCOPE, _SCOPE_EXPLICIT = scope, True
    try:
        yield scope
    finally:
        _STORE_SCOPE, _SCOPE_EXPLICIT = previous, previous_explicit


def without_store_access_scope() -> "contextlib.AbstractContextManager[Optional[StoreAccessScope]]":
    """An unscoped host for the block (the environment override hidden): every store access inside refuses by name."""
    return using_store_access_scope(None)


def require_store_access(what: str, path: Optional[str] = None) -> StoreAccessScope:
    """The chokepoint: return the scope in force, or refuse by name (:class:`GradingHostStoreReadRefused`) when there is none."""
    scope = store_access_scope()
    if scope is None:
        where = f" ({path})" if path else ""
        raise GradingHostStoreReadRefused(path or what, f"store access refused: {what}{where} -- no store access scope is installed on this host; "
                                     "adapters read on Slurm (registry cluster-reads shard), and a deliberate read here sets "
                                     f"{STORE_OVERRIDE_ENV}=<reason> (logged) (registry-slurm-only C6, S2)")
    return scope


def site_root(spec: str) -> str:
    """An adapter's store root written with a site placeholder (``$GRADE/<rel>``; :mod:`sisteer.site`), as the path it names on
    this host. It resolves when the adapter module loads, so a module's root constants stay plain strings and importing a module
    never needs a site config. Where the placeholder does not resolve (no site config here and no value in the environment) the
    root keeps its placeholder form, and the adapter's first ``discover()`` / ``read()`` refuses by name
    (:func:`require_resolved_roots`): nothing reads a literal ``$GRADE`` directory."""
    from ... import site as _site
    try:
        return str(_site.expand_path(spec))
    except (_site.PlaceholderError, _site.SiteConfigError) as exc:
        UNRESOLVED_SITE_ROOTS[spec] = str(exc)
        return spec


#: every :func:`site_root` spelling that did not resolve in this process, with the reason: one unresolved placeholder means the
#: process started without the site's values, so no adapter reads (a root, a staging dir or a grade dir an adapter derives from it
#: would silently name nothing)
UNRESOLVED_SITE_ROOTS: Dict[str, str] = {}


def require_resolved_roots(adapter) -> None:
    """Refuse when any adapter store path in this process still holds a site placeholder (:func:`site_root` could not resolve it
    when its module loaded), or when the adapter's own declared roots do: the process must start with the variable set (in the
    environment or the site config)."""
    name = getattr(adapter, "name", type(adapter).__name__)
    if UNRESOLVED_SITE_ROOTS:
        spec, why = sorted(UNRESOLVED_SITE_ROOTS.items())[0]
        raise AdapterRefusal(spec, f"{name}: {len(UNRESOLVED_SITE_ROOTS)} adapter store path(s) hold an unresolved site placeholder "
                                   f"({why}); start the process with the site's values set")
    for root in getattr(adapter, "roots", None) or ():
        if isinstance(root, str) and "$" in root:
            raise AdapterRefusal(root, f"{name}: the declared root holds an unresolved site placeholder")


def _guarded_method(adapter, name: str, fn: Callable):
    def guarded(*args, **kwargs):
        require_store_access(f"{getattr(adapter, 'name', type(adapter).__name__)}.{name}()")
        require_resolved_roots(adapter)
        return fn(*args, **kwargs)
    guarded.__name__ = getattr(fn, "__name__", name)
    guarded.__doc__ = getattr(fn, "__doc__", None)
    guarded.__wrapped__ = fn       # type: ignore[attr-defined]
    return guarded


def guard_adapter(adapter):
    """Wrap the instance's ``discover`` and ``read`` so the first call refuses without a scope (:func:`require_store_access`); idempotent.
    The instance is returned as is (``isinstance`` holds; the adapter's own attributes are untouched)."""
    if getattr(adapter, "_store_access_guarded", False):
        return adapter
    for name in ("discover", "read"):
        fn = getattr(adapter, name, None)
        if callable(fn):
            try:
                setattr(adapter, name, _guarded_method(adapter, name, fn))
            except AttributeError:
                # a frozen or slotted instance: the dispatch-site guards (run_pass, check_input, the walkers) still hold
                continue
    try:
        setattr(adapter, "_store_access_guarded", True)
    except AttributeError:
        pass
    return adapter


def guarded_factory(name: str, factory: Callable[[], Adapter]) -> Callable[[], Adapter]:
    """``factory`` with its product passed through :func:`guard_adapter`; ``__wrapped__`` names the registered factory."""
    if getattr(factory, "_store_access_guarded_factory", False):
        return factory

    def make(*args, **kwargs):
        return guard_adapter(factory(*args, **kwargs))
    make.__name__ = f"guarded:{name}"
    make.__wrapped__ = factory         # type: ignore[attr-defined]
    make._store_access_guarded_factory = True   # type: ignore[attr-defined]
    return make


def store_listdir(path) -> List[str]:
    """``os.listdir`` behind the chokepoint (the same exceptions; the refusal is :class:`GradingHostStoreReadRefused`, never ``OSError``)."""
    require_store_access("listing a store directory", os.fsdecode(path) if isinstance(path, (str, bytes)) else str(path))
    return os.listdir(path)


def store_listdir_if_scoped(path) -> List[str]:
    """A bounded listing for a CONSTRUCTION-time or IMPORT-TIME root discovery (an adapter whose ``__init__`` lists a parent directory to
    find the roots it will read beside the ones it declares): on an unscoped host -- the grading host, which reads no store (C6, S2) -- the
    listing is NOT attempted and an empty list is returned, so the adapter constructs with its declared roots alone (``stores coverage``
    constructs every adapter and opens nothing); the shard installs its scope before the adapters import and construct, so there the
    discovery runs. An absent or unreadable directory lists as empty either way. A read-time listing uses :func:`store_listdir`, which
    refuses by name instead -- never this."""
    if store_access_scope() is None:
        return []
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def store_glob_if_scoped(pattern, **kwargs) -> List[str]:
    """:func:`store_listdir_if_scoped`'s shape for a construction-time ``glob``: no match on an unscoped host, ``glob.glob`` under a scope."""
    if store_access_scope() is None:
        return []
    return _glob.glob(pattern, **kwargs)


def store_scandir(path):
    """``os.scandir`` behind the chokepoint."""
    require_store_access("scanning a store directory", os.fsdecode(path) if isinstance(path, (str, bytes)) else str(path))
    return os.scandir(path)


def store_walk(top, **kwargs):
    """``os.walk`` behind the chokepoint (checked once, when the walk is requested)."""
    require_store_access("walking a store tree", os.fsdecode(top) if isinstance(top, (str, bytes)) else str(top))
    return os.walk(top, **kwargs)


def store_glob(pattern, **kwargs) -> List[str]:
    """``glob.glob`` behind the chokepoint."""
    require_store_access("matching store paths", str(pattern))
    return _glob.glob(pattern, **kwargs)


def store_iglob(pattern, **kwargs):
    """``glob.iglob`` behind the chokepoint (checked when the iterator is requested)."""
    require_store_access("matching store paths", str(pattern))
    return _glob.iglob(pattern, **kwargs)


def store_open(path, mode: str = "r", **kwargs):
    """``open`` behind the chokepoint, for a STORE file (never the registry's own records or state)."""
    require_store_access("opening a store file", os.fsdecode(path) if isinstance(path, (str, bytes)) else str(path))
    return open(path, mode, **kwargs)


# -- the audit instrument -------------------------------------------------------------------------------------------------------------

_AUDIT: Dict[str, object] = {"path": None, "roots": ()}
_AUDIT_HOOKED = False


def _audit_hook(event: str, args) -> None:
    path = _AUDIT.get("path")
    if path is None or event not in STORE_AUDIT_EVENTS or not args:
        return
    target = args[0]
    if isinstance(target, bytes):
        target = os.fsdecode(target)
    if not isinstance(target, str):
        return                                   # a file descriptor, a Path-like already resolved elsewhere, a non-path argument
    ap = os.path.abspath(target)
    roots = _AUDIT.get("roots") or ()
    if not any(ap == r or ap.startswith(r + os.sep) for r in roots):   # type: ignore[union-attr]
        return
    if ap == os.path.abspath(str(path)):
        return                                   # the log itself
    try:
        with open(str(path), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "path": ap}, sort_keys=True) + "\n")
    except OSError:
        pass


def install_store_audit(log_path: str, roots: Sequence[str]) -> None:
    """Record every store access under ``roots`` (:data:`STORE_AUDIT_EVENTS`) as one JSON line in ``log_path``. The ``sys.addaudithook``
    is registered once per process and reads this module's configuration, so :func:`disable_store_audit` turns it off again (an audit
    hook itself cannot be removed)."""
    global _AUDIT_HOOKED
    _AUDIT["path"] = os.path.abspath(log_path)
    _AUDIT["roots"] = tuple(sorted({os.path.abspath(os.path.expanduser(str(r))).rstrip(os.sep) or os.sep for r in roots if str(r).strip()}))
    if not _AUDIT_HOOKED:
        sys.addaudithook(_audit_hook)
        _AUDIT_HOOKED = True


def disable_store_audit() -> None:
    _AUDIT["path"] = None
    _AUDIT["roots"] = ()


def install_store_audit_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Install the audit from ``REGISTRY_STORE_AUDIT`` / ``REGISTRY_STORE_AUDIT_ROOTS`` when the first is set (the CLI's and the smoke
    scripts' first act); returns whether it was. Default roots: this home."""
    env = environ if environ is not None else os.environ
    path = str(env.get(STORE_AUDIT_ENV) or "").strip()
    if not path:
        return False
    roots = [r for r in str(env.get(STORE_AUDIT_ROOTS_ENV) or "").split(":") if r.strip()] or [os.path.expanduser("~")]
    install_store_audit(path, roots)
    return True
