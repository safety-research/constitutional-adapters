"""Slurm-sharded reads of the heavy adapters. The grading host plans a window, the cluster reads it as an array of shards against
immutable store generations (:mod:`sisteer.registry.stores`) and a shipped reader-state sidecar (:mod:`sisteer.registry.reader_state`),
and the grading host pulls what was published -- marker last, behind a failure barrier. THE GRADING HOST READS NO STORE: an adapter
the cluster could not read is RECORDED REFUSED with its reason (``cluster-reads refusals``; the window stamps ``-- adapter A REFUSED``
and ends ABORTED naming them) and is never read on the grading host; the one thing the grading host still does for a cluster read is
re-activate, from COMMITTED BYTES, the refused-known revisions a shard returned (:func:`reactivate_refused_known` ->
``runner.reactivate_by_id``).

THE LAYOUT ON THE NODE (identity vs addressing, as built). Every adapter root hangs off ``$HOME`` at import, and two live
store indexes (``sampled/report/analysis/final_rung.json`` under ``sampled`` and under ``relu_paired``) embed ~3,000 absolute
paths under the home directory that the readers follow. So the task does NOT relocate the home: the CANONICAL spelling itself is made
to resolve on the node -- the node-local home receives, per pinned component, a symlink from the component's home-relative base onto
its verified generation (``$SHARED/registry/stores/gen/<slug>/<sha8>``; a base two components share becomes a real directory with
one link per top-level entry), so ``Input.uri`` IS the grading host's spelling while ``Input.path`` (the real path) addresses the
generation. The pass's :class:`base.IdentityMap` then carries only the pairs that are NOT the home: the exported tree onto the
checkout on the grading host (a repo-resident declaration file is an input too) and each generation path onto its canonical base (a
reader that resolves a real path into a record -- relu_paired's object weights location -- still spells the canonical home).
``SISTEER_REGISTRY_URI_HOME`` stays the general mechanism for a home that differs (it is checked against the node's home and REFUSED
when they differ: the embedded paths would not resolve).

THE ATTESTATION (``scripts/registry/landing/attest.sbatch`` -> ``stores.attest_plan``): ONCE per read
operation, before the array (``cluster-reads submit`` submits the attestation job first and the array behind it with
``--dependency=afterok``), every pinned generation of the plan is attested where it stands -- the authority facts (`current` names the pin;
MANIFEST / VERIFIED / DEST-STAT / SEALED / PUBLISHED beside it, agreeing with the manifest), a FULL DIRECTORY LISTING against the manifest
(every extra, missing or re-typed entry and every differing link target refuses by name), the directory seal (every directory 0555, the
count the SEALED record's), and the content -- the full ``admit_generation`` walk the first time (recorded in ``ATTESTED-<sha8>.json`` beside
the generation) or a seeded 64-file stat-and-hash sample within a week of it -- in parallel threads, into ``<window root>/attestation.json``.
A refusal is a RESULT (the job exits 0 once the record is written); the shards read it.

THE SHARD (:func:`run_shard`, one array task): check the operation's attestation record for every pinned generation (:func:`admit_pins`:
O(1) -- the record exists, is THIS operation's and attests each pin ok; else refuse by name, no walk, no hash) -> lay the home tree out ->
run the pass against the sidecar with the shard deal (``runner.deal_shards``) and the BYTE GUARANTEE (:class:`GenerationHashCache`: every
input under a pinned generation is hashed and its sha compared with the APPROVED manifest; a mismatch refuses by name; and the ESCAPE
REFUSAL: an input spelled under a pinned component's base whose real path is not inside that component's pinned generation is refused
before a byte is read) -> check the record again when the read completes -> write the shard summary
``<window root>/<adapter>/<i>-of-<K>.json``. A refused admission, a refused-known group, a held group: all named in the summary; a
refused admission or shard is the adapter's RECORDED REFUSAL this window (nothing reads it elsewhere), a refused-known group is the
grading host's to re-activate from committed bytes at pull. The shard runs under the one :class:`adapters.base.StoreAccessScope` the grading host
never has (``kind = shard``, installed before the adapters import): every store access of a pass passes the S2 chokepoints with it.
Publishes land in the window root's spool ``root/records/adapters/<a>/<rev16>/`` by one rename.
ANY DROPPED INPUT REFUSES THE GROUP on the cluster (``runner.ClusterInputDropRefusal``): the shard reads a staged copy,
and a copy that lacks an input the previous accepted revision read is INCOMPLETE -- the grading host's 10 % threshold is for a producer's own
vanished files and does not apply here. No revision, no marker; the refusal names each missing input with the staged component that would
have covered it (``ClusterRead.components``, filled from the plan's pins), the summary carries it (``refused[].inputs_dropped``), and the
pull names it as the adapter's refusal reason. A group whose every input is absent is refused the same way UNLESS the adapter declares
every one of them retired (``inputs_retired``; ``runner.retire_vanished_group``, registry-hardening 3a): then its one owning shard commits
the retirement -- the empty accepted revision, a marker-only directory the pull delivers like any other -- with the declaration on the
marker (``extra.retirement_declared``); an undeclared one is the adapter's to declare or the grading host's.

COVERAGE IS THE SHARD'S: the plan runs NO discovery on the grading host any more (the rule: no store read on the grading host at any
entry point -- the plan's discovery was one). The shard's own discovery over the home tree is the coverage
check: an input the staged components do not hold is absent on the node, so a group whose head read it is refused there by name
(:class:`runner.ClusterInputDropRefusal`, the covering component or ``no staged component covers it``) and the pull records the adapter's
refusal. :func:`input_coverage` / :func:`uncovered_reason` stay as the rule's pure reference implementation (a caller holding a discovered
input set -- a test, a shard-side report -- can judge it against the manifest copies); ``plan_window`` no longer calls them.
A LISTING input (``adapters.base.ListingInput``: a directory's listing, a synthetic ``registry://listing/...`` identity) is covered by its
DIRECTORY: under a pinned component's base, with the manifest copy of that generation attesting the listing of that directory
(``stores.MANIFEST_RULES["directories"]``) at the fingerprint the reader computes -- the shard's own guard (``runner.ClusterListingRefusal``,
over :meth:`GenerationHashCache.attested_listing`) compares every listing input against the attestation the pin names, and refuses the group
by name (``staged_listing_unattested`` / ``staged_listing_mismatch``) when the directory is unattested or its listing moved since the push.

THE PIN NAMES ITS ATTESTATION: a content sha cannot tell two directory listings of the same files apart, so
beside ``pins = {slug: content_sha8}`` the plan records ``attestations = {slug: listings_sha256 | None}`` -- the attestation the published
generation was VERIFIED against (``stores.published_attestations``, recorded by the publish). Attestations are immutable per (content,
listings) (``MANIFEST-<sha8>.<lsha8>.json``, the grading host state dir and destination alike; ``stores.MANIFEST_RULES["directories"]``), and every
pin-side check reads the pin's: the plan's grading host revalidation, the node's manifest index (so the shard's listing guard compares with what
the window pinned), and the post-run revalidation -- a canonical rebuilt as the same files over other listings while a window pinned to
the old attestation ran refuses that window's components ``listing_changed``, and its pulled revisions are quarantined. The shard's
ADMISSION is the operation's attestation record (C5, :func:`admit_pins`: ``attestation.json`` written by ``attest.sbatch`` before the
array), O(1) per pin; the pins' listing attestations ride the plan and the summaries beside it.

THE TRANSPORT (:func:`pull_window`): the grading host pulls. Pass 1 rsyncs every revision directory WITHOUT ``*source_revision*``
markers; a revision's marker is eligible only after pass 1 returned 0 AND the inbox directory holds exactly the non-marker files
(names and sizes) the shard summary lists for it; pass 2 pulls the eligible markers alone and re-checks each arrived whole. Any rsync
error or list mismatch FAILS CLOSED: no marker moves, the partial directory is moved under ``inbox/<name>/_failed/<window>/`` (operator
space ``scan_inbox`` never merges), and the adapter is recorded refused. The harvester's own marker-wait is the second guard; its
backstop ``pull_clusters`` of the producers' spool is untouched and never sees a window root. The pull then re-activates every
REFUSED-KNOWN revision the shards returned whose ``source_revision`` row is not its source's current one, from the grading host's
committed bytes (:func:`reactivate_refused_known`; the report's ``reactivations``) -- the reactivation the cluster left to the grading
host, without a store read. The ids are GATED like the publications: a refused-known id is replayed only when the shard that returned
it attests ``ok``, its adapter passed the post-run revalidation, the operation's attestation record ADMITS the adapter on the grading
host's own re-check (:func:`attestation_premise`, FAIL-CLOSED: a readable record of THIS operation pulled beside the summaries, naming
every pin the adapter reads at its pinned generation and attesting it ok; a missing or unreadable record, another operation's, a pin the
record does not name or attests at another generation, or one it refused, voids every claim the adapter's shards made), and the summary
transport returned 0 -- a replay writes the live record log past the inbox barrier, and the window's refusal would not undo it; every other
id is withheld by name (the report's ``refused_known_withheld``). The same attestation premise gates DELIVERY: a publication of an adapter
the record does not admit is never pulled (``withheld``) and quarantined under ``_failed/`` if a directory of it is already in the inbox
-- the shard summary's ``ok`` is the shard's claim, the record is what the pull verifies for itself. Three more rules of the transport:
the summaries and the record are received into a FRESH per-attempt directory and installed whole
(:func:`install_received`), so a retry over the same window cannot read an earlier attempt's record; a summary transfer that did not
return 0 is a WHOLE-PULL barrier (the report's ``barrier``: no pass 1, no pass 2, nothing replayed, every cluster adapter refused with
the rc and rsync's words); and pass 1 pulls BY OWNERSHIP -- exactly the directories this window's summaries list and nothing withholds
-- so a directory in the shared inbox that no summary of this window names (an earlier window's delivery interrupted before its
inbox-merge) is left for inbox-merge, never quarantined (``unlisted``). And ``runner.reactivate_by_id`` refuses, by name, an
accepted superseded revision that carries NO ``source_status`` row (committed before the status fact existed): a re-emission of committed
bytes cannot supply the reader's status facts, so re-activating it would make it the head while absent from ``v_source_status``; the
operator's ``--reread --force`` (a fresh read under ``REGISTRY_STORE_READ_OVERRIDE``) files the row.

THE WINDOW (``scripts/registry/landing/window.sh``; its cluster stages run only for the adapters the window NAMES that
have a cluster policy, and are skipped in one stamped line when none has): inbox-merge of the STANDING inbox (an earlier
window's complete, never-merged delivery is in records/ before the reader-state export, so no shard reissues its id; a marker-less partial
stays by ``harvest.scan_inbox``'s rule) -> reader-state export -> plan (admission pass 1 on the grading host: ``stores.revalidate`` per
component) -> stage (the sidecar, the plan and the rulings symlink onto the record-log mirror, verified) -> submit the array -> watch ->
re-validate (pass 2; a difference RECORDS that adapter refused -- the pull withholds every revision it published this window and
quarantines a listed one already in the inbox under ``inbox/<name>/_stale/<window>/adapters/``; nothing is moved before the pull) ->
pull + inbox-merge ->
:func:`refused_adapters` names every adapter that was NOT read this window, with the reason (the window stamps each ``REFUSED`` and ends
ABORTED when any is) -> one ingest. The plan file ``<root>/state/cluster_reads/<window>.json`` records every route and refusal
(``refusals``); the status page's "Cluster reads" section reads the latest one.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .. import scrub_map as SM
from . import db as DB
from . import harvest as H
from . import reader_state as RS
from . import records as R
from . import stores as ST
from .adapters.base import AdapterRefusal, InputHashCache, StoreAccessScope, install_store_audit_from_env, is_listing_input, using_store_access_scope
from .listing import Entries, listing_fingerprint
from .emit import producer_block, substrate_from_env, substrate_override
from .vocab import SUBSTRATES
from .mirrors import RSYNC_SSH, SSH_OPTS

PLAN_SCHEMA = "cluster_reads_plan/1"
SUMMARY_SCHEMA = "shard_summary/2"          # /2: `published[<rev dir>].sha256` -- a per-file digest beside each size
STATE_SUBDIR = "cluster_reads"                 # <root>/state/cluster_reads/<window>.json
READS_REL = "registry/reads"                   # $SHARED/registry/reads/<window>/
RULINGS_MIRROR_REL = "registry/log/rulings"    # the record-log mirror's rulings
TREES_REL = "gates/trees"                      # $SHARED/gates/trees/<sha> (tree_export.sh)
DEFAULT_ADAPTERS: Tuple[str, ...] = ("relu_paired", "sampled_static", "multiturn_grades", "em_misalign", "bigcodebench_full", "bfcl_full")
MARKER_TOKEN = "source_revision"
INBOX_NAME = "cluster_a"
FAILED_DIR = "_failed"
STALE_DIR = "_stale"
PARTIAL_DIR = "_partial"      # inbox/<name>/_partial/<window>/<rev dir>: an earlier window's marker-less partial, set aside when this window's verified copy installs
REFUSED_DIR = "_refused"      # records/adapters/<a>/_refused/<rev16>/: a refused read's record (runner.refused_dir); a shard's travels with the pull
STANDING_REFUSAL = "standing_refusal"   # the per-group state of a refusal re-confirmed byte for byte under the same inputs and tree: recorded, never an
                                        # abort. Distinct from `refused_standing`, the state of an emit-time-refused REVISION ID left unread (string ids)
RECEIVING_FILE = "RECEIVING.json"        # <receive dir>/RECEIVING.json: what a later window needs to finish an interrupted delivery
RECEIVING_SCHEMA = "receiving/1"
RECV_PREFIX = ".recv-"                   # inbox/<name>/.recv-<window>-<pid>-<rand>/: a pull's attempt-local receive directory
#: the pull report's reason for an inbox directory no summary of this window lists (ownership): left for inbox-merge
UNLISTED_REASON = "no shard summary of this window lists it; left in the inbox for inbox-merge (an earlier window's delivery, or a summary that never arrived)"
CONFIG_STAGED = "cluster_reads.yaml"                     # <window root>/cluster_reads.yaml: the exact configuration the plan was built from
#: the site placeholders a node must resolve for the staged configuration's component bases and the adapters' store roots: a node
#: carries no site config, so the plan carries the grading host's values (:func:`shard_site_env`) and the shard sets each one its
#: environment lacks (:func:`apply_shard_site_env`) before it loads the staged configuration or imports an adapter module
SHARD_SITE_VARS: Dict[str, str] = {"STATE": "SISTEER_STATE_ROOT", "RUNSTATE": "SISTEER_RUN_STATE_ROOT", "STAGE": "SISTEER_STAGE_DIR",
                                   "WORK": "SISTEER_WORK_DIR", "GRADE": "SISTEER_GRADE_DIR"}
NAMESPACE_MARKER = ".registry-reads-namespace.json"     # <home>/: the window's home tree as built on this node
NAMESPACE_LOCK = ".registry-reads-namespace.lock"
#: the status page's "Cluster reads" section when no plan was recorded (or the plan file cannot be read)
EMPTY_SECTION: Dict[str, Any] = {"present": False, "window": None, "rows": [], "stages": {}}


class ClusterReadsError(RuntimeError):
    """A refusal by name: the plan, the layout, the transport or a shard could not proceed."""


_write_json = ST._write_json


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ClusterReadsError(f"{path}: not a JSON object")
    return data


def cluster_producer(substrate: Optional[str] = None, code_sha: Optional[str] = None) -> Dict[str, object]:
    """The producer every record of a cluster read carries: the substrate and host it was read on (``emit.substrate_from_env`` when not
    given) -- nothing pretends to be the grading host (spec "Provenance")."""
    return producer_block(H.HARVESTER_PRODUCER_SCRIPT, code_sha=code_sha or os.environ.get("SISTEER_TREE_SHA") or "unverified", substrate=substrate)


# --------------------------------------------------------------------------------------------
# the manifest-backed hash cache
# --------------------------------------------------------------------------------------------


class GenerationHashCache(InputHashCache):
    """The pass's hash cache as the BYTE GUARANTEE of a cluster read (``admit_generation`` is only a precheck): every input
    is hashed as ever (:meth:`lookup` answers nothing, so no sha is ever taken on trust) and, when it lies under a pinned generation,
    its computed sha and size are compared with the generation's APPROVED manifest (``stores.manifest_index``: the manifest hashes
    recomputed and matched against the VERIFIED report) -- a mismatch REFUSES the input by name (:class:`base.InputRefusal`, which
    refuses the adapter's discovery, so the adapter falls back to the grading host this window and the summary says why). An input under
    no pinned generation (a repo-resident declaration file) is hashed and admitted as on the grading host. Never saved back."""

    def __init__(self, indexes: Sequence[object]):
        super().__init__(None)
        self.indexes: List[Tuple[str, object]] = sorted(((str(ix.tree).rstrip("/"), ix) for ix in indexes), key=lambda t: len(t[0]), reverse=True)
        self.aliases: Dict[str, str] = {}             # a home-tree COPY of a generation file -> the generation's path (its manifest entry)
        # (a canonical base, EVERY generation pinned at it), longest base first: two components may share a base with disjoint members
        # (`$GRADE`: grade_alphasteer + grade_arditi), and then both generations stand at that base
        self.bases: List[Tuple[str, Tuple[str, ...]]] = []
        self.checked_against_manifest = 0
        self.refused: List[str] = []
        self.escaped: List[str] = []                  # the inputs the escape refusal named (their canonical spelling)

    def bind_bases(self, bases: "Mapping[str, object] | Iterable[Sequence[str]]") -> None:
        """The shard's pinned components as ``(canonical base, pinned generation tree)`` pairs -- an iterable of pairs, or a mapping whose
        value is one generation or several -- the home tree's bases onto their generations: what :meth:`admit`'s escape refusal judges an
        input's canonical spelling against. EVERY generation pinned at a base is kept (a ``{base: generation}`` dict kept one of
        the two components sharing ``$GRADE``, and a valid input of the other was refused as escaping). Longest base first, so a
        component whose base nests inside another's (``full22/sampled`` under ``full22``) claims its own subtree."""
        grouped: Dict[str, List[str]] = {}
        items = bases.items() if isinstance(bases, Mapping) else bases
        for b, g in items:
            key = os.path.abspath(str(b)).rstrip("/")
            for gen in ([g] if isinstance(g, str) else list(g)):      # type: ignore[union-attr]
                gen = str(gen).rstrip("/")
                if gen not in grouped.setdefault(key, []):
                    grouped[key].append(gen)
        self.bases = sorted(((b, tuple(gs)) for b, gs in grouped.items()), key=lambda t: len(t[0]), reverse=True)

    def admit(self, uri: str, real: str) -> None:
        """The ESCAPE REFUSAL, before a byte is read: an input whose canonical spelling lies under a pinned component's base
        must address bytes INSIDE a generation pinned at that base -- its real path under the generation tree, or a home-tree copy aliased
        onto a generation entry (:meth:`alias_files`); when two components share the base (disjoint members, each in its own generation),
        either generation. Anything else at such a spelling -- a file written into a real directory of the home tree, a link out of it, a
        node-local path -- is refused by name (:class:`base.InputRefusal`), and so is an input that resolves into a generation pinned at
        a DIFFERENT base (nothing approved those bytes at this spelling). The longest base claims the spelling: an input under a nested
        component's base (``full22/sampled``) must lie in THAT component's generation, never in the enclosing base's. An input under no
        pinned base (a repo-resident declaration file) is admitted as ever."""
        from .adapters.base import InputRefusal
        u = os.path.abspath(uri)
        for base, gens in self.bases:
            if u == base or u.startswith(base + "/"):
                r = self.aliases.get(os.path.abspath(real), os.path.abspath(real))
                if any(r == gen or r.startswith(gen + "/") for gen in gens):
                    return
                self.escaped.append(uri)
                where = (f"that component's pinned generation {gens[0]}" if len(gens) == 1
                         else f"any of the {len(gens)} generations pinned at that base ({', '.join(gens)})")
                raise InputRefusal(uri, f"escapes the pinned generation: the canonical spelling lies under {base} and its real path {real} is not under "
                                        f"{where}; nothing approved these bytes")

    def alias_files(self, pairs: Iterable[Sequence[str]]) -> None:
        """``(copy, generation path)`` pairs (:func:`home_tree_spec`'s ``files``): a regular-file member of a shared base is a copy in the
        home tree (a file symlink would be an input refused for an adapter that admits none), and its bytes are checked against the
        generation entry it was copied from -- the byte guarantee holds for copies too."""
        for copy, src in pairs:
            self.aliases[os.path.abspath(str(copy))] = str(src)

    @classmethod
    def from_pins(cls, gen_dir: str, pins: Mapping[str, str], attestations: Optional[Mapping[str, Optional[str]]] = None) -> "GenerationHashCache":
        """One approved index per pinned generation, its directory listings from the attestation the pin names (``attestations``;
        ``stores.manifest_index``), never the destination's latest ``.json`` of that content."""
        indexes = []
        for slug, sha8 in sorted(pins.items()):
            try:
                indexes.append(ST.manifest_index(f"{gen_dir}/{slug}", sha8, listings_sha256=(attestations or {}).get(slug)))
            except ST.StoresError as exc:
                raise ClusterReadsError(f"pinned generation {slug}@{sha8}: {exc}") from exc
        return cls(indexes)

    @classmethod
    def from_pins_file(cls, path: str) -> "GenerationHashCache":
        doc = _load_json(path)
        if not doc.get("gen_dir") or not isinstance(doc.get("pins"), dict):
            raise ClusterReadsError(f"{path}: a pins file carries gen_dir and pins {{slug: sha8}}")
        att = doc.get("attestations") if isinstance(doc.get("attestations"), dict) else {}
        return cls.from_pins(str(doc["gen_dir"]), {str(k): str(v) for k, v in doc["pins"].items()}, {str(k): (str(v) if v else None) for k, v in att.items()})

    def index_of(self, real: str) -> Optional[Tuple[str, object]]:
        """The (true rel, index) of a real path inside a pinned generation: the on-disk name is unescaped (``stores.NAME_ESCAPE_RULE``) so
        the manifest -- which spells true names -- is consulted for the entry the bytes belong to."""
        real = self.aliases.get(real, real)
        for tree, ix in self.indexes:
            if real == tree:
                return "", ix
            if real.startswith(tree + "/"):
                return ST.unescape_rel(real[len(tree) + 1:]), ix
        return None

    def attested_listing(self, path: str) -> Optional[Sequence[Sequence[str]]]:
        """The canonical directory listing the pinned generation's manifest attests for ``path`` (a directory of the home tree, resolved
        onto the generation), or None when the directory lies under no pinned generation or its manifest attests no listing for it -- the
        shard's `ClusterRead.listing_attestation` (``runner.ClusterListingRefusal``: a listing input is compared with THIS, never with a
        previous revision)."""
        real = os.path.realpath(str(path))
        hit = self.index_of(real)
        if hit is None:
            return None
        rel, ix = hit
        return ix.attested_listing(rel)

    def lookup(self, real: str, st: os.stat_result) -> Optional[str]:
        self.seen.add(real)
        return None                       # every byte is read: nothing under a generation is taken on trust

    def record(self, real: str, st: os.stat_result, sha256: str) -> None:
        """The manifest check comes FIRST: a sha the approved manifest contradicts is never stored (the cache would otherwise hold the
        refused bytes' sha as if recorded), only counted as hashed and named in ``refused``."""
        from .adapters.base import InputRefusal
        hit = self.index_of(real)
        if hit is not None:
            rel, ix = hit
            why = ix.check(rel, sha256, int(st.st_size))
            if why is not None:
                self.seen.add(real)
                self.hashed += 1
                self.refused.append(real)
                raise InputRefusal(real, f"verified-manifest check: {why}")
        super().record(real, st, sha256)
        if hit is not None:
            self.checked_against_manifest += 1


# --------------------------------------------------------------------------------------------
# the home tree on the node
# --------------------------------------------------------------------------------------------


def _manifest_copy_path(root: str, slug: str, sha8: str, what: str) -> str:
    mpath = os.path.join(ST.component_state_dir(root, slug), f"MANIFEST-{sha8}.sha256")
    if not os.path.isfile(mpath):
        raise ClusterReadsError(f"component {slug}: no manifest copy for generation {sha8} at {mpath}; {what}")
    return mpath


def manifest_entries(root: str, slug: str, sha8: str, rel_dir: str = "") -> List[Tuple[str, bool]]:
    """The entries ``(name, is_dir)`` of one directory of the pinned generation (``rel_dir``: base-relative, ``""`` = the top level),
    from the grading host's own copy of its manifest (``state/store_manifests/<slug>/MANIFEST-<sha8>.sha256`` + the symlinks in its ``.json``)
    -- the plan lays the home tree out without listing a directory on the cluster; true names, never the destination's escaped
    spelling. Refuses by name when the copy is absent."""
    mpath = _manifest_copy_path(root, slug, sha8, "the home tree cannot be laid out from it")
    prefix = rel_dir.strip("/") + "/" if rel_dir.strip("/") else ""
    tops: Dict[str, bool] = {}
    for _sha, _size, rel in ST.read_manifest_lines(mpath):
        if prefix and not rel.startswith(prefix):
            continue
        head, sep, _rest = rel[len(prefix):].partition("/")
        tops[head] = tops.get(head, False) or bool(sep)
    for rel in ST.read_manifest_symlinks(mpath):
        if prefix and not rel.startswith(prefix):
            continue
        head, sep, _rest = rel[len(prefix):].partition("/")
        tops[head] = tops.get(head, False) or bool(sep) or head not in tops     # a symlink of the generation is linked like a directory entry
    return sorted(tops.items())


def generation_escapes(root: str, slug: str, sha8: str) -> Dict[str, str]:
    """``{true rel: escaped rel}`` the pinned generation's manifest records (``stores.read_manifest_escapes``) -- the entries whose
    on-disk names differ from their canonical spelling (the destination refuses ':'), from the grading host's own manifest copy."""
    return ST.read_manifest_escapes(_manifest_copy_path(root, slug, sha8, "its name escapes cannot be read"))


def _opened_dirs(escapes: Mapping[str, str]) -> set:
    """The true directories the home tree must lay out as REAL directories so that every escaped entry gets its own node-local link: every
    proper ancestor of each escaped path component (a whole link above an escaped name would resolve the canonical spelling to a path
    the destination does not have)."""
    opened: set = set()
    for rel in escapes:
        parts = rel.split("/")
        for i, part in enumerate(parts):
            if ST.escape_name(part) != part:
                for j in range(1, i + 1):
                    opened.add("/".join(parts[:j]))
    return opened


def home_tree_spec(components: Mapping[str, object], pins: Mapping[str, str], gen_dir: str, home: str,
                   entries_of: Callable[[str, str], Sequence[Tuple[str, bool]]],
                   subentries_of: Optional[Callable[[str, str, str], Sequence[Tuple[str, bool]]]] = None,
                   escapes_of: Optional[Callable[[str, str], Mapping[str, str]]] = None) -> Dict[str, Any]:
    """ONE namespace for the window: what every pinned component's canonical spelling resolves to under ``home``, computed from
    the UNION of every cluster-routed adapter's components so tasks of different adapters on one node build the same tree. A base no
    other pinned component shares (nor nests under) links whole onto its generation; a shared or nesting base is a REAL directory whose
    top-level entries are laid out one by one -- a directory entry as a link onto the generation's, a regular-file entry as a COPY of it
    (a file symlink is an input refused for an adapter that admits none; :meth:`GenerationHashCache.alias_files` checks the copy's
    bytes against the generation). Two components that need different things at one path REFUSE by name, and so does an entry that
    would be created inside a whole-linked base (nothing is ever written into a generation). ``entries_of(slug, sha8)`` lists a
    generation's top-level entries (:func:`manifest_entries` on the grading host).

    THE NAME ESCAPE (``stores.NAME_ESCAPE_RULE``): an entry whose canonical name the destination refuses (``:``) lives in the
    generation under an escaped name, so its canonical spelling is a NODE-LOCAL LINK onto the escaped path, and every directory above
    it is a REAL directory of the home tree laid out entry by entry (``subentries_of(slug, sha8, rel_dir)``: :func:`manifest_entries`
    with a directory; ``escapes_of(slug, sha8)``: :func:`generation_escapes`). A component with escapes and no ``subentries_of`` refuses
    by name. Every link target is the generation's ESCAPED path; every link and copy keeps the canonical name. Returns ``{home, gen_dir,
    dirs, links: [[link, target]], files: [[copy, source]], escaped: [[canonical, escaped]]}`` -- a pure function of its inputs, carried in
    the plan and initialized on the node by :func:`init_home_tree` (``gen_dir``: the generations root every target lies under, what the
    builder vouches a link it finds against)."""
    pinned = {slug: components[slug] for slug in pins if slug in components}
    missing = sorted(set(pins) - set(pinned))
    if missing:
        raise ClusterReadsError(f"pinned components not declared in the configuration: {missing}")
    home = home.rstrip("/")
    bases: Dict[str, List[str]] = {}
    for slug, comp in pinned.items():
        if comp.home_rel is None:
            raise ClusterReadsError(f"component {slug}: base {comp.base} is not under the home; the home tree cannot lay it out")
        bases.setdefault(comp.home_rel, []).append(slug)
    escapes: Dict[str, Dict[str, str]] = {slug: dict((escapes_of or (lambda s, g: {}))(slug, pins[slug])) for slug in pinned}
    for slug, esc in escapes.items():
        bad = sorted(rel for rel, e in esc.items() if ST.escape_rel(rel) != e)
        if bad:
            raise ClusterReadsError(f"component {slug}: the manifest's name_escapes disagree with the escape rule for {bad[:5]}; the home tree cannot lay it out")
        if esc and subentries_of is None:
            raise ClusterReadsError(f"component {slug}: {len(esc)} escaped name(s) in generation {pins[slug]} and no directory listing to lay them out with")
    opened = {slug: _opened_dirs(esc) for slug, esc in escapes.items()}
    claims: Dict[str, Tuple[str, Optional[str], str]] = {}          # path -> (kind: dir | link | file, target, the component(s) needing it)
    escaped_links: List[List[str]] = []

    def claim(path: str, kind: str, target: Optional[str], slug: str) -> None:
        prev = claims.get(path)
        if prev is not None and (prev[0], prev[1]) != (kind, target):
            raise ClusterReadsError(f"components {prev[2]} and {slug} need different things at {path}: {prev[0]} {prev[1] or ''} vs {kind} {target or ''}".rstrip())
        claims[path] = (kind, target, slug)

    def lay_out(slug: str, gen: str, base_dir: str, rel_dir: str) -> None:
        """One directory of a component laid out entry by entry; an opened directory (above an escaped name) recurses as a real one."""
        listing = entries_of(slug, pins[slug]) if not rel_dir else subentries_of(slug, pins[slug], rel_dir)   # type: ignore[misc]
        for name, is_dir in listing:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            path = os.path.join(base_dir, rel)
            if rel in opened[slug]:
                claim(path, "dir", None, slug)
                lay_out(slug, gen, base_dir, rel)
                continue
            target = f"{gen}/{ST.escape_rel(rel)}"
            claim(path, "link" if is_dir else "file", target, slug)
            if ST.rel_needs_escape(rel):
                escaped_links.append([path, target])

    for rel, slugs in sorted(bases.items()):
        nested = [r for r in bases if r != rel and r.startswith(rel.rstrip("/") + "/")]
        base_dir = os.path.join(home, rel)
        if len(slugs) == 1 and not nested and not escapes[slugs[0]]:
            claim(base_dir, "link", f"{gen_dir}/{slugs[0]}/{pins[slugs[0]]}", slugs[0])
            continue
        claim(base_dir, "dir", None, "+".join(sorted(slugs)))
        for slug in sorted(slugs):
            lay_out(slug, f"{gen_dir}/{slug}/{pins[slug]}", base_dir, "")
    # every directory between the home and a claimed path is a REAL one: a whole-linked base is never an ancestor of anything created
    dirs = {p for p, (kind, _t, _s) in claims.items() if kind == "dir"}
    for p, (kind, _t, slug) in list(claims.items()):
        parent = os.path.dirname(p)
        while len(parent) > len(home):
            if parent in claims and claims[parent][0] != "dir":
                raise ClusterReadsError(f"{parent} is linked whole onto a generation (for {claims[parent][2]}) and {slug} needs {os.path.basename(p)} "
                                        "inside it; nothing is written into a generation")
            dirs.add(parent)
            parent = os.path.dirname(parent)
    return {"home": home, "gen_dir": gen_dir.rstrip("/"), "dirs": sorted(dirs),
            "links": sorted([p, t] for p, (k, t, _s) in claims.items() if k == "link"),
            "files": sorted([p, t] for p, (k, t, _s) in claims.items() if k == "file"),
            "escaped": sorted(escaped_links)}


def _spec_sha(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()


def _verify_home_tree(spec: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Every entry of the built namespace, as the spec says: a real directory, a link onto its target, a copy the size of its source."""
    problems: List[Dict[str, str]] = []
    for d in spec["dirs"]:
        if os.path.islink(d) or not os.path.isdir(d):
            problems.append({"path": d, "why": "not a real directory"})
    for link, target in spec["links"]:
        if not os.path.islink(link) or os.readlink(link) != target:
            problems.append({"path": link, "why": f"not a link onto {target}"})
    for dst, src in spec["files"]:
        try:
            ok = os.path.isfile(dst) and not os.path.islink(dst) and os.path.getsize(dst) == os.path.getsize(src)
        except OSError:
            ok = False
        if not ok:
            problems.append({"path": dst, "why": f"not a copy of {src}"})
    return problems


def _link_ancestor(path: str, home: str) -> Optional[str]:
    """The TOPMOST ancestor of ``path`` strictly below ``home`` that is a symlink, or None: beneath it the spelling resolves INTO a
    generation (a base a build promoted to a whole link), where nothing of the builder's stands and nothing is ever removed."""
    ancestors: List[str] = []
    parent = os.path.dirname(path)
    while len(parent) > len(home):
        ancestors.append(parent)
        parent = os.path.dirname(parent)
    for a in reversed(ancestors):                              # from the home down: the first link is the one everything beneath runs through
        if os.path.islink(a):
            return a
    return None


def _recorded_entries(path: str, links: Set[str], files: Set[str], dirs: Set[str]) -> Optional[Tuple[List[str], List[str]]]:
    """Every entry under ``path``, recursively, as the previous window's recorded namespace knows it: ``(unlink, rmdir)`` -- its recorded
    links and copies, then its recorded directories bottom-up -- or None the moment an entry that record does NOT name stands there (a
    symlink, a file, a directory of someone else's): the directory is then not the builder's to remove. Nothing is touched here."""
    unlink: List[str] = []
    rmdir: List[str] = []
    with os.scandir(path) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        if entry.is_symlink():
            if entry.path not in links:
                return None
            unlink.append(entry.path)
        elif entry.is_dir(follow_symlinks=False):
            if entry.path not in dirs:
                return None
            sub = _recorded_entries(entry.path, links, files, dirs)
            if sub is None:
                return None
            unlink += sub[0]
            rmdir += sub[1] + [entry.path]
        elif entry.is_file(follow_symlinks=False) and entry.path in files:
            unlink.append(entry.path)
        else:
            return None
    return unlink, rmdir


def _same_bytes(a: str, b: str, chunk: int = 1 << 20) -> bool:
    """Two regular files hold the same bytes: sizes first, then a streamed compare (``filecmp.cmp`` keeps a process-wide cache keyed on
    size and mtime, which a same-size rewrite under a restored mtime would answer from)."""
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                return False
            if not x:
                return True


def _build_home_tree(spec: Mapping[str, Any], previous: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Lay the namespace out (the FIRST task of a window on this node, under :func:`init_home_tree`'s lock): the previous window's
    entries this one does not name are removed (a link or a copy a previous build made -- never a real directory), its whole links
    where this window needs a real directory are removed, its links are re-pointed onto this window's generations; a real path where
    a link belongs refuses by name -- except a REAL DIRECTORY the builder itself made, which the link replaces: one the previous window's
    recorded namespace names among its ``dirs`` (a base laid out entry by entry) under which, once that window's entries are withdrawn,
    nothing stands that the record does not ALSO name (its ``links``, ``files`` and ``dirs``, checked entry by entry: ``makedirs`` accepts a
    directory that already existed, so ``dirs`` records what was REQUIRED, not what was made); a component re-based from ``<base>/hpo``
    onto ``<base>`` met the real ``<base>/`` its previous build had made and refused every shard of every tick, 2026-10-07 13:48Z. An
    unrecorded entry anywhere under it -- a file, an unmanaged alias, a directory -- keeps the refusal, nothing removed. A recorded entry
    whose spelling now runs THROUGH a symlink (a base an earlier attempt of this window promoted whole, its marker withheld by a refusal
    elsewhere) is skipped (``skipped_under_link``), never stat'ed or removed through the link into the generation -- and the promoted link
    ITSELF, when this window names it neither as a link nor as a directory, is removed by name (a symlink: the generation is never touched)
    if the previous window recorded a real directory at its spelling AND its literal target lies under the plan's generations root
    (``gen_dir``; the builder's own targets, never resolved further), else REFUSED by name and left standing: left alone it would outlive
    every marker as an UNRECORDED link at a canonical spelling onto an old generation, whose bytes a later adapter rooted beneath it could
    discover and :meth:`GenerationHashCache.admit` would accept (a spelling under no pinned base is admitted as ever). A COPY path that
    already exists is replaced only when the previous window's recorded namespace (``previous``: its ``links`` and ``files``, the MANAGED
    paths) names it -- a previous window's copy, read-only because ``copy2`` kept the sealed generation's mode (eleven EACCES refusals), or
    a link where a copy now belongs; an existing path no previous window recorded is REFUSED by name and left untouched (a file this
    machinery did not write at that spelling is not its to delete) -- EXCEPT a regular file (no symlink, no directory) whose bytes ARE the
    generation's file's, which is ADOPTED (``adopted``): a copy a build of this very layout made while a refusal elsewhere withheld its
    marker (the 13:48-17:04Z outage, its builds refusing at a link and copying all the same) stands unrecorded at the spelling this window
    names, and every adapter sharing its base refused on it thereafter, 2026-10-07 17:34Z. An adopted copy whose mtime is the generation's
    is not written at all; one whose mtime differs is re-copied through the replace path, because the pass compares an input's
    ``(sha256, size, mtime_utc)`` with its stored watermark (``runner.plan_groups``) and a drifted mtime would read as a changed input the
    grading host never saw. Bytes that differ, a symlink or a directory keep today's refusal. The marker this build writes on success
    carries the spec, so the adopted path is managed from then on. TWO PHASES, AND A REFUSED BUILD CHANGES NOTHING ON DISK: a PREFLIGHT
    evaluates every refusal condition of every pass above -- the promoted-link vouching, ``exists and is not a directory``, a link's
    parent, ``exists and is not a symlink`` with the owned-directory check, a copy's parent, directory and unmanaged-and-not-adoptable
    checks, the byte compare included -- against the tree AS IT STANDS, touching nothing, and collects the writes it would make; then the
    writes run, in pass order, ONLY when nothing refused. Otherwise every planned write -- the withdrawal of a previous window's entries
    included -- is listed under ``not_attempted`` and the tree is left byte for byte as found: a refused build advances no marker, so
    anything it changed would stand unrecorded for the next build to refuse on. Two faces of that, both live on 2026-10-07: the unrecorded
    copies (a build refusing at a link ran its copy loop all the same), and the inverse of the promoted base -- a recorded whole link the
    dirs pass remade as a real directory for a build that then refused, which no marker's ``dirs`` names, so a later window linking the
    base whole refuses it for good. Where phase 2 would change what the preflight sees, the preflight reads the tree as phase 2 leaves it:
    a path beneath a whole link this window remakes as a real directory does not exist yet, and a parent this window's ``dirs`` create
    counts as a real directory. A copy that fails in phase 2 (``cannot copy``) refuses the build and withholds the copies after it. Every
    path created lies under the home."""
    out: Dict[str, Any] = {"created": [], "repointed": [], "unchanged": [], "removed": [], "copied": [], "adopted": [], "refused": [],
                           "skipped_under_link": [], "not_attempted": []}
    home = str(spec["home"]).rstrip("/")
    keep = {p for p, _t in spec["links"]} | {p for p, _t in spec["files"]} | set(spec["dirs"])
    previous_links = {p for p, _t in (previous or {}).get("links") or []}
    previous_files = {p for p, _t in (previous or {}).get("files") or []}
    previous_dirs = set((previous or {}).get("dirs") or [])          # the real directories a previous build REQUIRED (recorded since the marker exists)
    managed = previous_links | previous_files
    gen_root = os.path.normpath(str(spec["gen_dir"])).rstrip("/") if spec.get("gen_dir") else None
    plan: List[Tuple[Any, ...]] = []                           # phase 2's writes in pass order: (what, path, ...)

    def ancestor_in(path: str, paths: Set[str]) -> bool:
        """A proper ancestor of ``path``, strictly below the home, is one of ``paths``."""
        parent = os.path.dirname(path)
        while len(parent) > len(home):
            if parent in paths:
                return True
            parent = os.path.dirname(parent)
        return False

    # ---- phase 1: the preflight, against the tree as it stands; nothing is touched ----
    withdraw: Set[str] = set()
    promoted: Dict[str, List[str]] = {}                       # a symlinked ancestor -> the recorded entries whose spelling runs through it
    for p, _t in list((previous or {}).get("links") or []) + list((previous or {}).get("files") or []):
        if p in keep:
            continue
        above = _link_ancestor(p, home)
        if above is not None:
            # an earlier attempt of this window promoted an ancestor base to a whole link and a refusal elsewhere withheld its marker: the
            # spelling now resolves into the generation, where isfile would find the generation's own file and remove would reach it
            out["skipped_under_link"].append(p)
            promoted.setdefault(above, []).append(p)
            continue
        if os.path.islink(p) or os.path.isfile(p):
            withdraw.add(p)
            plan.append(("remove", p))
    for above in sorted(promoted):
        if above in keep:
            continue                                           # this window names it: a real directory again (the dirs pass) or a whole link (re-pointed below)
        target = os.readlink(above)
        literal = os.path.normpath(target) if os.path.isabs(target) else None
        why = ([] if above in previous_dirs else ["no previous window recorded a real directory there"]) \
            + (["the plan carries no generations root"] if gen_root is None else
               [] if literal is not None and (literal == gen_root or literal.startswith(gen_root + "/")) else [f"its target does not lie under the generations root {gen_root}"])
        if why:
            out["refused"].append({"path": above, "why": f"is a link onto {target} this window does not name ({'; '.join(why)}); "
                                                           "refusing to remove a link this machinery cannot vouch for"})
        else:
            plan.append(("remove", above))                     # the builder's own promoted link, dropped by this window: a symlink, so the generation is never touched
    dirs_ok: Set[str] = set()                                  # the real directories phase 2 leaves standing
    remade: Set[str] = set()                                   # whole links this window remakes as real directories: nothing stands beneath them afterwards
    for d in spec["dirs"]:                                     # sorted: parents before children
        if ancestor_in(d, remade):
            dirs_ok.add(d)
            plan.append(("mkdir", d))
        elif os.path.islink(d):                                # a previous window linked this base whole; this one needs a real directory
            remade.add(d)
            dirs_ok.add(d)
            plan += [("remove", d), ("mkdir", d)]
        elif os.path.lexists(d) and not os.path.isdir(d):
            out["refused"].append({"path": d, "why": "exists and is not a directory; refusing to replace a real path"})
        else:
            dirs_ok.add(d)
            if not os.path.isdir(d):
                plan.append(("mkdir", d))

    def parent_ok(path: str) -> bool:
        parent = os.path.dirname(path)
        return parent in dirs_ok or (not os.path.islink(parent) and os.path.isdir(parent))

    for link, target in spec["links"]:
        if not parent_ok(link):
            out["refused"].append({"path": link, "why": f"its parent {os.path.dirname(link)} is not a real directory"})
        elif ancestor_in(link, remade) or not os.path.lexists(link):
            plan.append(("symlink", link, target))
        elif os.path.islink(link):
            if os.readlink(link) == target:
                out["unchanged"].append(link)
            else:
                plan.append(("repoint", link, target))
        else:
            # a real directory a previous window laid out entry by entry (recorded in its `dirs`) where this one links the base whole -- a
            # component re-based onto its parent (#470, 2026-10-07) -- is the builder's to replace only if everything under it is recorded
            # too (the entries the withdrawal above removes are recorded, and are not removed twice); the check runs whole, nothing removed
            owned = _recorded_entries(link, previous_links, previous_files, previous_dirs) if (os.path.isdir(link) and link in previous_dirs) else None
            if owned is None:
                out["refused"].append({"path": link, "why": "exists and is not a symlink; refusing to replace a real path"})
            else:
                plan.append(("replace", link, target, [p for p in owned[0] if p not in withdraw], owned[1]))
    for dst, src in spec["files"]:
        if not parent_ok(dst):
            out["refused"].append({"path": dst, "why": f"its parent {os.path.dirname(dst)} is not a real directory"})
            continue
        if ancestor_in(dst, remade) or not os.path.lexists(dst):
            plan.append(("copy", dst, src))
            continue
        if os.path.isdir(dst) and not os.path.islink(dst):
            out["refused"].append({"path": dst, "why": "is a directory; refusing to replace a real path with a file copy"})
            continue
        if dst in managed:
            plan.append(("copy", dst, src))                    # a RECORDED copy or link is replaced (removed, then copied) in phase 2
            continue
        # a regular file no window recorded whose bytes are the generation's file's: a copy a withheld build left -- adopted
        try:
            adopt = os.path.isfile(dst) and not os.path.islink(dst) and _same_bytes(src, dst)
        except OSError:
            adopt = False                                      # unreadable on either side: not proven ours, so today's refusal
        if not adopt:
            out["refused"].append({"path": dst, "why": "exists and is not a copy or link a previous window recorded in this node's namespace; "
                                                       "refusing to replace a path this machinery does not manage"})
            continue
        plan.append(("adopt", dst, src))
    if out["refused"]:
        for action in plan:                                    # every planned write, once, in the order it would have run
            if action[1] not in out["not_attempted"]:
                out["not_attempted"].append(action[1])
        return out

    # ---- phase 2: the writes, in pass order ----
    for action in plan:
        what, path = action[0], action[1]
        if what == "remove":
            os.remove(path)
            out["removed"].append(path)
        elif what == "mkdir":
            os.makedirs(path, exist_ok=True)
        elif what == "symlink":
            os.symlink(action[2], path)
            out["created"].append(path)
        elif what == "repoint":
            tmp = f"{path}.{os.getpid()}.tmp"
            os.symlink(action[2], tmp)
            os.replace(tmp, path)
            out["repointed"].append(path)
        elif what == "replace":
            for p in action[3]:
                os.remove(p)
                out["removed"].append(p)
            for d in list(action[4]) + [path]:                 # recorded directories bottom-up, the base itself last
                os.rmdir(d)
                out["removed"].append(d)
            os.symlink(action[2], path)
            out["created"].append(path)
        else:                                                  # copy | adopt
            if out["refused"]:
                out["not_attempted"].append(path)              # a copy before this one failed: nothing more is written
                continue
            src = action[2]
            try:
                if what == "adopt" and os.stat(path).st_mtime_ns == os.stat(src).st_mtime_ns:
                    out["adopted"].append(path)                # the generation's bytes AND mtime: nothing written; this build's marker records it
                    continue
                if os.path.lexists(path):
                    # a previous window's RECORDED copy or link is REPLACED, never written over: copy2 preserves the sealed generation's
                    # read-only mode, so writing over an earlier window's copy failed every shard of three adapters with EACCES (2026-09-29)
                    os.remove(path)
                shutil.copy2(src, path)                        # bytes + mtime: the watermark's stat reads as the grading host's
                out["copied"].append(path)
                if what == "adopt":
                    out["adopted"].append(path)                # the generation's bytes under another mtime: re-copied (the rule above)
            except OSError as exc:
                out["refused"].append({"path": path, "why": f"cannot copy {src}: {exc}"})
    return out


def init_home_tree(spec: Mapping[str, Any], home: str, window: str) -> Dict[str, Any]:
    """The window's namespace on THIS node, once: under a node-local lock the first task builds it (:func:`_build_home_tree`)
    and records it in ``<home>/.registry-reads-namespace.json``; every later task of the same window finds the record and VERIFIES the
    tree is the one the plan describes -- nothing is re-pointed, removed or created within a window, so concurrent shards of different
    adapters never pull a base from under each other. A record of the same window built from ANOTHER plan refuses by name."""
    home = home.rstrip("/")
    os.makedirs(home, exist_ok=True)
    lock_path, marker_path = os.path.join(home, NAMESPACE_LOCK), os.path.join(home, NAMESPACE_MARKER)
    sha = _spec_sha(spec)
    counts = {"links": len(spec["links"]), "files": len(spec["files"]), "dirs": len(spec["dirs"])}
    with open(lock_path, "a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        marker: Optional[Dict[str, Any]] = None
        if os.path.isfile(marker_path):
            try:
                marker = _load_json(marker_path)
            except (OSError, ValueError, ClusterReadsError):
                marker = None
        if marker is not None and marker.get("window") == window:
            if marker.get("spec_sha256") != sha:
                return {"mode": "refused", **counts, "refused": [{"path": marker_path, "why": f"window {window}'s namespace on this node was built from "
                                                                  f"another plan (sha {str(marker.get('spec_sha256'))[:12]} != {sha[:12]}); nothing is re-pointed within a window"}]}
            return {"mode": "verified", **counts, "refused": _verify_home_tree(spec)}
        built = _build_home_tree(spec, (marker or {}).get("spec"))
        built.update(mode="built", previous_window=(marker or {}).get("window"), **counts)
        if not built["refused"]:
            _write_json(marker_path, {"window": window, "spec_sha256": sha, "spec": dict(spec), "built_utc": R.utc_now(), "host": socket.gethostname()})
        return built


def admit_pins(attestation_path: str, window: str, pins: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """The shard's admission, O(1) per pin: the operation's attestation record (``stores.attest_plan``, written by
    ``attest.sbatch`` before the array) must exist at ``attestation_path``, be THIS operation's (``window``) and attest every pinned
    generation ok -- refusals by name (``stores.attestation_admits``: ``no_attestation``, ``attestation_window``, ``attestation_missing``,
    ``attestation_pin``, ``attestation_refused`` carrying the attestation's own refusal). No walk and no hash here: the attestation did the
    listing, the seal check and the content walk or sample once for the operation (replacing the per-shard ``admit_generation`` walk that
    cost 837 s per shard), and the reader's per-input hash against the pinned manifest stays the byte guarantee."""
    return ST.attestation_admits(ST.read_attestation(attestation_path), window, pins, path=attestation_path)


def _refused_roots(admission: Mapping[str, Mapping[str, Any]], **when: str) -> List[Dict[str, Any]]:
    """The components an admission refused, by name, for the shard summary's ``refused_roots``."""
    return [{"component": s, "refusal": a.get("refusal"), "reason": a.get("reason"), **when} for s, a in sorted(admission.items()) if not a.get("ok")]


# --------------------------------------------------------------------------------------------
# the shard summary
# --------------------------------------------------------------------------------------------


def _rel_under(path: str, base: str) -> Optional[str]:
    base = base.rstrip("/") + "/"
    return path[len(base):] if path.startswith(base) else None


def _summary_skeleton(adapters: Sequence[str], shard: Optional[Tuple[int, int]], substrate: str) -> Dict[str, Any]:
    """Every key a shard summary carries, empty: the pass's summary and a refused shard's fill the same document."""
    return {"schema": SUMMARY_SCHEMA, "adapters": list(adapters), "shard": ({"index": shard[0], "count": shard[1]} if shard else None),
            "host": socket.gethostname(), "substrate": substrate, "written_utc": R.utc_now(), "pass": None, "published": {},
            "refused_known": [], "refused": [], "held": [], "refused_standing": [], "standing_refusals": [], "refused_records": {}, "ok": True}


def shard_summary(rep, summary: Mapping[str, Any], adapters: Sequence[str], shard: Optional[Tuple[int, int]], root: str,
                  producer: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The shard summary (``shard_summary/2``): what the pass saw, changed, published (per revision directory: its files with their sizes
    AND their sha256 digests -- the transport's barrier verifies what it received against exactly this, byte for byte, before anything
    enters the inbox), held, refused, refused as known, left to other shards; wall, host, substrate.

    ``refused_standing`` is the groups the plan left standing UNREAD (the standing-refusal rule of ``runner.plan_groups``; no ``retry_refused``),
    as before. ``standing_refusals`` is the groups the shard RE-READ and refused again byte for byte under the same inputs and the
    same reading tree -- per group ``state = standing_refusal``, the refusal's words (``refusal``) and ``prior_refusal`` (the window that
    recorded the refusal this one equals, its utc, its digest, the attempt id and the tree; ``runner.standing_refusal_of``). Neither is in
    ``refused`` and neither refuses the adapter at the pull. ``refused_records`` lists every ``_refused/<rev>/`` record this shard wrote
    (a NEW or CHANGED refusal's and a standing one's alike) in the ``published`` shape -- files, sizes, digests, the marker -- so the pull
    delivers them like any read result: the harvester folds the stamps, the next sidecar carries them, and the next shard can tell a
    standing refusal from a new one."""
    adapters_root = os.path.join(H.records_dir(root), "adapters")
    published: Dict[str, Dict[str, Any]] = {}
    refused_records: Dict[str, Dict[str, Any]] = {}
    refused_known: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    standing: List[Dict[str, Any]] = []
    held: List[Dict[str, Any]] = []

    def listed(into: Dict[str, Dict[str, Any]], r, standing_: bool) -> None:
        for f in r.files:
            rel = _rel_under(f, adapters_root)
            if rel is None:
                continue
            rev_dir = os.path.dirname(rel)
            d = into.setdefault(rev_dir, {"adapter": r.adapter, "group": r.group_identity or r.group, "revision_id": r.revision_id,
                                          "files": {}, "sha256": {}, "marker": None, **({"standing": standing_} if into is refused_records else {})})
            name = os.path.basename(f)
            d["files"][name] = int(os.path.getsize(f))
            d["sha256"][name] = R.file_sha256(f)        # the bytes the shard wrote; the pull verifies what it received against these
            if MARKER_TOKEN in name:
                d["marker"] = name

    for r in rep.results:
        entry = {"adapter": r.adapter, "group": r.group_identity or r.group, "revision_id": r.revision_id}
        if r.refused_known:
            # a RETIREMENT's refused-known result carries the shard's claim (runner.retirement_claim: the head it compared against, the
            # vanished inputs, the covering patterns); the pull's re-activation by id is gated on it (registry-hardening 3a, round 1)
            refused_known.append(dict(entry, why=r.error, **({"retirement": dict(r.retirement)} if r.retirement else {})))
        elif r.held:
            held.append(dict(entry, why=r.error))
        elif r.standing_refusal:
            standing.append(dict(entry, state=STANDING_REFUSAL, refusal=r.error, prior_refusal=dict(r.prior_refusal or {})))
        elif not r.accepted:
            # ``inputs_dropped``: the staged copy lacked an input of the group's head (runner.ClusterInputDropRefusal); the pull names it
            refused.append(dict(entry, error=r.error, inputs_dropped=bool(r.inputs_dropped)))
        if r.accepted and r.files:
            listed(published, r, False)
        elif r.files and not r.accepted and not r.refused_known and not r.held:
            listed(refused_records, r, bool(r.standing_refusal))
    doc = _summary_skeleton(adapters, shard, str(producer.get("substrate") or substrate_from_env()))
    doc.update({"pass": dict(summary), "published": published, "refused_known": refused_known, "refused": refused, "held": held,
                "refused_standing": [{"adapter": a, "group": g} for a, groups in sorted(dict(rep.refused_standing).items()) for g in groups],
                "standing_refusals": standing, "refused_records": refused_records})
    doc.update(extra or {})
    return doc


def write_shard_summary(path: str, rep, summary: Mapping[str, Any], adapters: Sequence[str], shard: Optional[Tuple[int, int]], root: str,
                        producer: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    doc = shard_summary(rep, summary, adapters, shard, root, producer, extra)
    _write_json(path, doc)
    return doc


def summary_path(summaries_dir: str, adapter: str, index: int, count: int) -> str:
    return os.path.join(summaries_dir, adapter, f"{index}-of-{count}.json")


# --------------------------------------------------------------------------------------------
# the shard task (cluster side)
# --------------------------------------------------------------------------------------------


def shard_site_env() -> Dict[str, str]:
    """This host's text for each :data:`SHARD_SITE_VARS` placeholder, as the site writes it (a ``~`` stays: a shard runs only under the
    plan's canonical home, so a ``~`` value names the same directory on the node as here). A placeholder this host cannot resolve is
    left out, and the node then refuses the value that needs it by name, as this host would."""
    from .. import site as _site
    out: Dict[str, str] = {}
    for name, var in SHARD_SITE_VARS.items():
        try:
            out[var] = _site.expand("$" + name)
        except (_site.PlaceholderError, _site.SiteConfigError):
            continue
    return out


def config_identity_refusal(recorded_sha256: Optional[str], path: str, as_path: Optional[str] = None) -> Optional[str]:
    """Why the configuration file at ``path`` is not the one a plan recorded (None when its bytes hash to exactly that sha). The
    plan's tree, home tree and component bases were derived from those bytes and every shard runs that tree, so staging and the shard
    hold this one rule: equal bytes, or a new plan. A certified scrub-map rewrite of the file is named as such and needs a new plan
    too, because the plan's tree read the bytes before it. ``as_path`` is the file the plan named when ``path`` is a copy of it (the
    staged copy beside the plan): a map row names the file at its repo path, so the lookup uses that path."""
    want, got = str(recorded_sha256 or ""), R.file_sha256(path)
    if got == want:
        return None
    if SM.check(want, got, path=as_path or path) == SM.VIA_MAP:
        return (f"sha {got[:12]} is the certified scrub-map rewrite of the recorded {want[:12]}; a plan binds the exact bytes its "
                "tree read, so plan again from the rewritten file")
    return f"sha {got[:12]}, recorded {want[:12] or 'unrecorded'}; {SM.mismatch_note(want, got)}"


def apply_shard_site_env(plan: Mapping[str, Any]) -> Dict[str, str]:
    """Set in this process each :data:`SHARD_SITE_VARS` value the plan carries and the environment lacks; the values it set."""
    applied: Dict[str, str] = {}
    known = set(SHARD_SITE_VARS.values())
    for var, value in sorted((plan.get("site_env") or {}).items()):
        if var in known and isinstance(value, str) and value and not os.environ.get(var):
            os.environ[var] = value
            applied[var] = value
    return applied


def run_shard(plan_path: str, index: int, work: Optional[str] = None, summaries_dir: Optional[str] = None, home: Optional[str] = None,
              log: Callable[[str], None] = lambda s: print(s, flush=True)) -> int:
    """One array task: admission -> home tree -> the sharded pass against the sidecar -> admission again -> the shard summary. Every
    refusal is a RESULT (the summary says so, rc 0); only a crash exits non-zero. ``home`` defaults to this process's home, which must be
    the plan's canonical home (the layout resolves embedded paths only there)."""
    t0 = time.monotonic()
    plan = _load_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ClusterReadsError(f"{plan_path}: schema {plan.get('schema')!r} is not {PLAN_SCHEMA!r}")
    tasks = plan.get("tasks") or []
    if not 0 <= index < len(tasks):
        raise ClusterReadsError(f"task index {index} is not one of the plan's {len(tasks)} tasks")
    task = tasks[index]
    adapter, i, k = str(task["adapter"]), int(task["shard"]), int(task["shards"])
    # the ONE store access scope (C6, S2): the shard's, for this task -- installed before the adapters import (an import-time discovery
    # of a cap-step directory runs here and nowhere else) and held through the pass; nothing on the grading host ever installs one
    scope = StoreAccessScope("shard", f"cluster read shard {i}/{k} of {adapter} (task {index})", window=str(plan.get("window")))
    with using_store_access_scope(scope):
        return _run_shard_scoped(plan_path, plan, index, task, adapter, i, k, t0, work=work, summaries_dir=summaries_dir, home=home, log=log)


def _run_shard_scoped(plan_path: str, plan: Dict[str, Any], index: int, task: Mapping[str, Any], adapter: str, i: int, k: int, t0: float,
                      work: Optional[str] = None, summaries_dir: Optional[str] = None, home: Optional[str] = None,
                      log: Callable[[str], None] = lambda s: print(s, flush=True)) -> int:
    """:func:`run_shard`'s body, under the shard's store access scope."""
    entry = plan["adapters"][adapter]
    pins = {str(s): str(g) for s, g in (entry.get("pins") or {}).items()}
    attestations = {str(s): (str(a) if a else None) for s, a in (entry.get("attestations") or {}).items()}      # the pin's attestation per component
    gen_dir, root = str(plan["gen_dir"]), str(plan["root"])
    from . import config as CFG
    from . import site_aliases as SA
    SA.install(CFG.load(root))             # the alias table stage_window put beside the shard's root (an empty one when the site has none)
    summaries_dir = summaries_dir or str(plan["summaries_dir"])
    out_path = summary_path(summaries_dir, adapter, i, k)
    home = os.path.abspath(home or os.path.expanduser("~"))
    # the substrate the shard reads on: a declared one, else the plan's destination (a [[clusters]] entry's registry label), else the probes
    dest_label = str(plan.get("destination") or "")
    substrate = substrate_override() or (dest_label if dest_label in SUBSTRATES else substrate_from_env())
    host_name = socket.gethostname()
    stamps: Dict[str, float] = {}
    attestation = str(plan.get("attestation") or os.path.join(os.path.dirname(os.path.abspath(plan_path)), ST.ATTESTATION_FILE))
    extra: Dict[str, Any] = {"window": plan.get("window"), "adapter": adapter, "task_index": index, "pins": pins, "attestations": attestations,
                             "admission_pre": {}, "admission_post": {}, "attestation": attestation,
                             "home_tree": None, "refused_roots": [], "timings_s": stamps, "started_utc": R.utc_now(), "plan": plan_path}

    def refuse(why: str) -> int:
        extra.update(ok=False, refusal=why, finished_utc=R.utc_now(), wall_s=round(time.monotonic() - t0, 3))
        _write_json(out_path, {**_summary_skeleton([adapter], (i, k), substrate), **extra})
        log(f"[read] REFUSED {adapter} shard {i}/{k}: {why}")
        return 0

    if str(plan.get("uri_home", "")).rstrip("/") != home.rstrip("/"):
        return refuse(f"the plan's canonical home {plan.get('uri_home')} is not this node's home {home}; the store indexes' embedded paths would not resolve")
    if entry.get("route") != "cluster":
        return refuse(f"the plan routes {adapter} to the grading host ({entry.get('reason')}); nothing to read here")
    t1 = time.monotonic()
    pre = admit_pins(attestation, str(plan.get("window")), pins)
    stamps["admission_pre"] = round(time.monotonic() - t1, 3)
    extra["admission_pre"] = pre
    bad = sorted(s for s, a in pre.items() if not a.get("ok"))
    if bad:
        extra["refused_roots"] = _refused_roots(pre)
        return refuse("generation admission refused: " + "; ".join(f"{s}: {pre[s].get('refusal')} ({pre[s].get('reason')})" for s in bad))
    # the configuration the plan was built from: the copy staged beside the plan, bound by sha (a node has no grading host checkout)
    staged = str(plan.get("config_staged") or os.path.join(os.path.dirname(plan_path), CONFIG_STAGED))
    if not os.path.isfile(staged):
        return refuse(f"the staged configuration {staged} is missing (cluster-reads stage lays it beside the plan)")
    why = config_identity_refusal(plan.get("config_sha256"), staged, as_path=str(plan.get("config") or "") or None)
    if why:
        return refuse(f"the staged configuration {staged} is not the one the plan was built from ({why})")
    extra["site_env"] = apply_shard_site_env(plan)      # before the staged configuration's bases or any adapter root expands
    cfg = ST.load_config(staged)
    # the window's ONE namespace, from the plan: built once per node under a lock, verified by every later task
    spec = plan.get("home_tree")
    if not isinstance(spec, dict) or str(spec.get("home", "")).rstrip("/") != home.rstrip("/"):
        return refuse("the plan carries no home tree for this node's home; nothing to lay out")
    built = init_home_tree(spec, home, str(plan.get("window")))
    extra["home_tree"] = {**{k_: (len(v) if isinstance(v, list) else v) for k_, v in built.items() if k_ != "refused"}, "refusals": built["refused"]}
    if built["refused"]:
        return refuse("home tree: " + "; ".join(f"{r['path']}: {r['why']}" for r in built["refused"]))
    # identity pairs beyond the home: the exported tree onto the grading host checkout, each generation onto its canonical base
    pairs: Dict[str, str] = {}
    if plan.get("tree") and plan.get("repo_root") and plan["tree"].rstrip("/") != plan["repo_root"].rstrip("/"):
        pairs[str(plan["tree"])] = str(plan["repo_root"])
    for slug, sha8 in pins.items():
        pairs[f"{gen_dir}/{slug}/{sha8}"] = os.path.join(home, cfg.components[slug].home_rel)
    from .adapters import load_all
    from .adapters import dimensions as DIM
    from .adapters.base import IdentityMap
    from .adapters.runner import ClusterRead, run_pass
    reg = load_all()
    if adapter not in reg:
        return refuse(f"adapter {adapter!r} is not registered in this tree")
    read_copy = {"substrate": substrate, "host": host_name, "canonical_home": home, "window": plan.get("window"),
                 "components": {s: {"origin": cfg.components[s].origin, "route": "generation", "content_sha8": sha8, "verified": True,
                                    "generation": f"{gen_dir}/{s}/{sha8}", "attestation": attestations.get(s)} for s, sha8 in sorted(pins.items())}}
    producer = cluster_producer(substrate=substrate, code_sha=plan.get("tree_sha"))
    imap = IdentityMap.build(pairs)
    cache = GenerationHashCache.from_pins(gen_dir, pins, attestations)
    cache.alias_files(spec.get("files") or [])
    # PAIRS, never a dict keyed by base: two pinned components may share a base, and every generation at it must be admitted
    cache.bind_bases([(os.path.join(home, cfg.components[s].home_rel), f"{gen_dir}/{s}/{sha8}") for s, sha8 in sorted(pins.items())])
    legacy = DIM.adapter_legacy_spellings(DIM.DEFAULT_SEEDS_DIR)
    policy = dict(entry.get("policy") or {})                   # the window's per-adapter read policy travels in the plan
    full = bool(plan.get("full")) or bool(policy.get("full"))
    conn = RS.open_reader_state(str(plan["reader_state"]), [adapter])
    try:
        cluster = ClusterRead.from_sidecar(conn, read_copy)
        # the staged components by canonical base: what a refusal for a missing input names as the component that would have covered it
        cluster.components = {s: {"base": os.path.join(home, cfg.components[s].home_rel), "generation": sha8} for s, sha8 in sorted(pins.items())}
        # the pinned generations' attested directory listings: what every listing input of the read is compared with (ClusterListingRefusal)
        cluster.listing_attestation = cache.attested_listing
        t1 = time.monotonic()
        rep = run_pass(root, [reg[adapter]()], conn=conn, full=full, retry_refused=bool(policy.get("retry_refused", True)), legacy_spellings=legacy,
                       producer=producer, hash_cache=cache, identity_map_=imap, shard=(i, k), cluster=cluster, save_cache=False)
        stamps["pass"] = round(time.monotonic() - t1, 3)
    finally:
        conn.close()
    t1 = time.monotonic()
    post = admit_pins(attestation, str(plan.get("window")), pins)
    stamps["admission_post"] = round(time.monotonic() - t1, 3)
    extra["admission_post"] = post
    stale = sorted(s for s, a in post.items() if not a.get("ok"))
    summary = rep.as_dict()
    summary.update(shard={"index": i, "count": k}, identity_map=imap.as_dict(), inputs_checked_against_manifest=cache.checked_against_manifest,
                   inputs_refused_by_manifest=list(cache.refused), inputs_escaped=list(cache.escaped), policy=dict(policy, full=full))
    extra.update(ok=not stale, finished_utc=R.utc_now(), wall_s=round(time.monotonic() - t0, 3), read_copy=read_copy)
    if stale:
        extra["refused_roots"] = _refused_roots(post, when="after the read")
        extra["refusal"] = "the attestation no longer admits a generation after the read: " + "; ".join(f"{s}: {post[s].get('refusal')}" for s in stale)
    doc = write_shard_summary(out_path, rep, summary, [adapter], (i, k), root, producer, extra)
    log(f"[read] {adapter} shard {i}/{k}: seen {rep.groups_seen} not_owned {rep.groups_not_owned} published {len(doc['published'])} held {rep.groups_held} "
        f"refused {rep.groups_refused} refused_known {rep.groups_refused_known} records {rep.records_written} hashed {rep.inputs_hashed} "
        f"checked_against_manifest {cache.checked_against_manifest} wall {extra['wall_s']}s ok={extra['ok']}")
    return 0


# --------------------------------------------------------------------------------------------
# the plan (the grading host side)
# --------------------------------------------------------------------------------------------


def plan_state_dir(root: str) -> str:
    return os.path.join(root, "state", STATE_SUBDIR)


def plan_path(root: str, window: str) -> str:
    return os.path.join(plan_state_dir(root), f"{window}.json")


def latest_plan(root: str) -> Optional[Dict[str, Any]]:
    d = plan_state_dir(root)
    if not os.path.isdir(d):
        return None
    names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    if not names:
        return None
    try:
        return _load_json(os.path.join(d, names[-1]))
    except (OSError, ValueError, ClusterReadsError):
        return None


def _refused_components(results: Mapping[str, Any]) -> Tuple[List[str], str]:
    """The components an admission (``stores.adapter_admission``) refused, and the reason naming each."""
    refused = sorted(s for s, rv in results.items() if not rv.ok)
    return refused, "; ".join(f"{s}: {results[s].refusal} ({results[s].reason})" for s in refused)


REREAD_FORCE_REASON = ("reread --force reads the store on the grading host (a committed revision read again); an operator override "
                       "(REGISTRY_STORE_READ_OVERRIDE=<reason>) is the one way to run it, never a cluster read")
#: the plan's reason for an adapter whose entry is ``cluster: refused`` (stores.CLUSTER_REFUSED): recognized before any admission,
#: the declaration's own reason appended; the adapter is recorded REFUSED this window (never read on the grading host; S2)
REFUSED_DECL_PREFIX = "declared cluster: refused in the configuration -- "
#: a plan entry routed to the cluster with no task would be "read" by nobody: every consumer of a plan (the submission, the refusal set,
#: the pull) records it refused, naming this reason
NO_TASK_REASON = "the plan routes the adapter to the cluster with no task (no shard would read it); recorded refused"
UNCOVERED_PREFIX = "input(s) outside the staged components"
#: how many uncovered inputs the plan names per root (the counts are exact; the names are a sample)
UNCOVERED_SAMPLE = 5


def _under(path: str, base: str) -> bool:
    base = str(base).rstrip("/")
    return bool(base) and (path == base or path.startswith(base + "/"))


def generation_members(root: str, slug: str, sha8: str) -> Tuple[set, Dict[str, str]]:
    """The base-relative paths a pinned generation holds, from the grading host's own manifest copy (``state/store_manifests/<slug>/MANIFEST-
    <sha8>.sha256`` and the symlinks in its ``.json``): ``(files, {symlink: target})``. Refuses by name when the copy is absent."""
    mpath = os.path.join(ST.component_state_dir(root, slug), f"MANIFEST-{sha8}.sha256")
    if not os.path.isfile(mpath):
        raise ClusterReadsError(f"component {slug}: no manifest copy for generation {sha8} at {mpath}; its members cannot be checked")
    return {rel for _sha, _size, rel in ST.read_manifest_lines(mpath)}, ST.read_manifest_symlinks(mpath)


def generation_listings(root: str, slug: str, sha8: str, listings_sha256: Optional[str] = None) -> Optional[Dict[str, Entries]]:
    """The directory listings the pin's attestation carries, from the grading host's own copy (``MANIFEST-<sha8>.<lsha8>.json`` for the named
    ``listings_sha256``, the plain ``.json`` when none is named; :data:`stores.MANIFEST_RULES` ["directories"]): ``{relpath: entries}``, or
    None when it attests no listing (written before the field) -- the attestation :func:`run_shard` installs on the node from the same
    keyed copy, so the plan judges what the shard will. Refuses by name a pin naming an attestation the grading host never wrote."""
    mpath = os.path.join(ST.component_state_dir(root, slug), f"MANIFEST-{sha8}.sha256")
    if not os.path.isfile(mpath):
        raise ClusterReadsError(f"component {slug}: no manifest copy for generation {sha8} at {mpath}; its listings cannot be checked")
    attestation = ST.read_manifest_attestation(mpath, listings_sha256)
    if listings_sha256 and attestation is None:
        raise ClusterReadsError(f"component {slug}: no attestation of generation {sha8} with listings {listings_sha256[:8]} beside {mpath}")
    return ST.attested_directories(attestation)


def _staged_member(rel: str, files: set, links: Mapping[str, str], hops: int = 0) -> bool:
    """Whether a base-relative path is a FILE the generation holds: listed as is, or reached through an attested directory link whose
    target -- relative and inside the base, the manifest refuses any other (``stores.link_escapes``) -- is followed the way the node's
    filesystem will follow it, to a listed file. The manifest attests a link WITHOUT walking through it, so a link into a directory the
    component EXCLUDES is in the generation while the files beneath it are not: those are not members (the review of the follow-ups; a
    NEW group discovered through such a link has no head for the shard's dropped-input rule to compare against)."""
    if rel in files:
        return True
    if hops > 16:
        return False
    parts = rel.split("/")
    for i in range(1, len(parts)):
        head = "/".join(parts[:i])
        target = links.get(head)
        if target is None:
            continue
        if os.path.isabs(target):
            return False
        resolved = os.path.normpath(os.path.join(os.path.dirname(head), target))
        if resolved == ".." or resolved.startswith("../"):
            return False
        return _staged_member(os.path.normpath(os.path.join(resolved, *parts[i:])), files, links, hops + 1)
    return False


def dangling_links(root: str, slug: str, sha8: str) -> Dict[str, str]:
    """The attested directory links of a pinned generation whose target holds NO member of the generation -- a link into a directory the
    component excludes, a climbing or an absolute target -- from the grading host's MANIFEST COPY alone (no store read):
    ``{link: target}``. On the node such a link dangles, so a group the grading host read through it is simply not discovered by the
    shard -- it has no head for the dropped-input rule to compare against and would vanish from the cluster read without a word. The
    plan refuses the adapter by name instead (:func:`plan_window`), the way the plan-time discovery used to."""
    files, links = generation_members(root, slug, sha8)
    out: Dict[str, str] = {}
    for link, target in sorted(links.items()):
        if os.path.isabs(target):
            out[link] = target
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(link), target))
        if resolved == ".." or resolved.startswith("../"):
            out[link] = target
            continue
        if resolved in files or any(f.startswith(resolved + "/") for f in files):
            continue                                  # a file, or a directory with members: the link resolves on the node
        out[link] = target
    return out


def input_coverage(cfg, root: str, pins: Mapping[str, str], inputs: Sequence[Any], repo_root: str, home: str,
                   adapter_roots: Sequence[str] = (), attestations: Optional[Mapping[str, Optional[str]]] = None) -> Dict[str, Any]:
    """Every discovered input against what the node WILL have. An input is COVERED when its canonical spelling lies
    under the exported tree (``repo_root``: a repo-resident declaration file) or under a pinned component's base AND its base-relative
    path is a member of that generation -- a file the grading host's manifest copy lists, or a path reached through a directory link it
    attests, RESOLVED through the link's target to a listed file (:func:`_staged_member`: a link into an excluded directory resolves to
    files the node will not have) -- and, for a symlink input, when its target is covered too. A LISTING input is covered by its
    directory (``ListingInput.path``): under a pinned base, with that generation's manifest copy attesting the directory's listing
    (:func:`generation_listings`) at the input's fingerprint, the comparison the shard's guard makes (``runner.ClusterListingRefusal``);
    a listing naming no directory (filtered or synthesized) is never a shard's. Every other
    input the staged copy would not resolve, and the plan routes the adapter to the grading host naming the ROOTS they lie under: a pinned
    component's base (a member its generation lacks: excluded, not a declared member, or newer than the mirror; a listing its generation
    does not attest or attests otherwise), else the adapter's declared root, else the input's top-level directory under the home, else its
    directory. Returns ``{discovered, covered, uncovered, roots: {root: {count, why, sample}}}``; a per-group alias (``<uri>#<group>``) is
    judged by its file."""
    members = {slug: generation_members(root, slug, sha8) for slug, sha8 in sorted(pins.items())}
    listings = {slug: generation_listings(root, slug, sha8, (attestations or {}).get(slug)) for slug, sha8 in sorted(pins.items())}
    bases = sorted(((cfg.components[s].base.rstrip("/"), s) for s in pins if s in cfg.components), key=lambda t: -len(t[0]))
    home = str(home).rstrip("/")
    declared = sorted((str(r).rstrip("/") for r in (adapter_roots or ()) if r), key=len, reverse=True)

    def covered(path: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """``(covered, the pinned base the path lies under or None, its slug)``."""
        if _under(path, repo_root):
            return True, None, None
        under: Optional[Tuple[str, str]] = None
        for base, slug in bases:
            if not _under(path, base):
                continue
            under = under or (base, slug)
            rel = os.path.relpath(path, base)
            files, links = members[slug]
            if rel in links or _staged_member(rel, files, links):     # an attested link entry itself: a symlink input, whose target is judged below
                return True, base, slug
        return False, (under[0] if under else None), (under[1] if under else None)

    def listing_covered(inp: Any) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """``(covered, the pinned base the listed directory lies under or None, its slug, why not)`` of a listing input."""
        path = str(getattr(inp, "path", "") or "").rstrip("/")
        if not path:
            return False, None, None, "a listing naming no directory of its own (filtered or synthesized): never a shard's to read"
        for base, slug in bases:
            if not _under(path, base):
                continue
            rel = "" if path == base else os.path.relpath(path, base)
            attested = listings[slug]
            if attested is None:
                return False, base, slug, f"a directory listing generation {pins[slug]}'s manifest does not attest (written before the field)"
            if rel not in attested:
                return False, base, slug, f"a directory generation {pins[slug]} attests no listing for"
            if listing_fingerprint(attested[rel]) != str(inp.sha256):
                return False, base, slug, f"a directory whose listing moved since generation {pins[slug]} was pushed"
            return True, base, slug, None
        return False, None, None, None

    out: Dict[str, Any] = {"discovered": len(inputs), "covered": 0, "uncovered": 0, "roots": {}}
    for inp in inputs:
        uri = str(inp.uri).split("#", 1)[0]
        why: Optional[str] = None
        where = uri                                       # the path the uncovered root is derived from
        if is_listing_input(inp):
            ok, base, slug, why = listing_covered(inp)
            where = str(getattr(inp, "path", "") or "") or uri
        else:
            ok, base, slug = covered(uri)
            if ok and getattr(inp, "is_symlink", False):
                ok, _b, _s = covered(str(inp.path))
                if not ok:
                    why = f"a symlink whose target {inp.path} the staged components do not hold"
        if ok:
            out["covered"] += 1
            continue
        out["uncovered"] += 1
        if base is not None:
            key, why = base, why or f"under component {slug}'s base but not a member of its generation {pins[slug]}"
        else:
            key = next((r for r in declared if _under(where, r)), None)
            if key is None:
                key = os.path.join(home, os.path.relpath(where, home).split("/")[0]) if _under(where, home) else os.path.dirname(where)
            why = why or "no staged component covers it"
        r = out["roots"].setdefault(key, {"count": 0, "why": why, "sample": []})
        r["count"] += 1
        if len(r["sample"]) < UNCOVERED_SAMPLE:
            r["sample"].append(uri)
    return out


def uncovered_reason(cov: Mapping[str, Any]) -> str:
    """The plan's reason for an adapter with uncovered inputs: the counts, then every uncovered root with its count and why."""
    parts = [f"{r['count']} under {root} ({r['why']})" for root, r in sorted(cov["roots"].items(), key=lambda kv: (-kv[1]["count"], kv[0]))]
    return f"{UNCOVERED_PREFIX}: {cov['uncovered']} of {cov['discovered']} -- " + "; ".join(parts)


def plan_time_checks(name: str, decl, cfg, registry: Mapping[str, Callable[[], Any]], home: Optional[str] = None) -> Dict[str, Any]:
    """The plan-time checks on the CONSTRUCTED adapter instance: ``{rng_scope, shards, roots, uncovered_roots,
    missing_components, construction_error, refusal}`` -- ``refusal`` is None when the entry may route to the cluster, else the reason by name.
    (a) An adapter declaring ``rng_scope = reader_traversal`` (one shared RNG over its whole traversal) must be planned as exactly ONE shard:
    the shard runner (``runner.run_pass``) admits it at K = 1 only, and a larger count crashes every task before a summary is written -- the
    configuration carries the count, this check makes a wrong one impossible to plan. (b) Input coverage: every root the instance declares it
    opens (``adapter.roots``) must be covered by the ENTRY's components -- :func:`stores.coverage` over this one adapter, the roots-only
    static check (``missing_components``: a held root none of whose holders the entry names; ``uncovered``: a root no component holds). An
    adapter that is not registered in this tree, or whose construction fails, is refused by name too (the shard would refuse it). ``home``
    is the home the roots are judged under -- :data:`stores.HOME`, the home ``Component.home_rel`` is relative to, by default (the grading host's)."""
    home = home or ST.HOME
    out: Dict[str, Any] = {"rng_scope": None, "shards": int(decl.shards), "roots": [], "uncovered_roots": [], "missing_components": [],
                           "construction_error": None, "refusal": None}
    factory = registry.get(name)
    if factory is None:
        out["refusal"] = f"{name} is not registered in this tree (plan-time check on the constructed adapter: nothing to construct)"
        return out
    try:
        adapter = factory()
    except Exception as exc:          # noqa: BLE001 -- the refusal names it
        out["construction_error"] = f"{type(exc).__name__}: {exc}"
        out["refusal"] = f"{name} could not be constructed for the plan-time checks: {out['construction_error']}"
        return out
    scope = str(getattr(adapter, "rng_scope", "none") or "none")
    out["rng_scope"] = scope
    if scope == "reader_traversal" and int(decl.shards) != 1:
        out["refusal"] = (f"{name} declares rng_scope = reader_traversal (one shared RNG over its whole traversal) and {decl.shards} shards in {cfg.path}; "
                          "such an adapter is planned as exactly ONE shard (the shard runner admits K = 1 only) -- set shards: 1")
        return out
    cov = ST.coverage(cfg, registry={name: factory}, home=home, only=[name])
    entry = cov.adapters.get(name) or {}
    out["roots"] = [{"root": r.get("root"), "state": r.get("state"), "components": list(r.get("components") or [])} for r in (entry.get("roots") or [])]
    out["uncovered_roots"] = [str(r.get("root")) for r in (entry.get("roots") or []) if r.get("state") in (ST.UNCOVERED, ST.UNDECLARED)]
    out["missing_components"] = list(entry.get("missing_components") or [])
    if entry.get("construction_error"):
        out["construction_error"] = entry["construction_error"]
        out["refusal"] = f"{name} could not be constructed for the plan-time checks: {entry['construction_error']}"
    elif out["uncovered_roots"] or out["missing_components"]:
        parts = []
        if out["uncovered_roots"]:
            parts.append("root(s) no component holds: " + ", ".join(out["uncovered_roots"]))
        if out["missing_components"]:
            parts.append("root(s) held by component(s) the entry does not name: " + ", ".join(out["missing_components"]))
        out["refusal"] = (f"{UNCOVERED_PREFIX}: {name} declares it opens " + "; ".join(parts)
                          + f" -- the plan pins generations from the entry in {cfg.path} alone, so these would be read nowhere on the node")
    return out


def holds_seen_of(sidecar: Optional[str], adapters: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
    """Per adapter, the retirement markers the harvester HOLDS because the head they were computed against moved before the fold
    (``harvest.retirements_held_for_moved_head``, ``blocked_records.kind = retirement_head``) as the READER-STATE SIDECAR at ``sidecar``
    carries them -- each the hold's identity (``harvest.retirement_hold_identity``: ``revision_id``; ``record_uri``, the held carrier;
    ``blocking_head``, the head it was computed against) -- what :func:`plan_window` writes on every adapter entry as ``holds_seen`` and the
    receipt carries (``reads_protocol.receipt_from_plan``).
    WHY the SIDECAR and never the live database: a hold is resolved by the next shard that judges the group against today's head, and the
    tick keeps the adapter unread until a window has done so (``reads_protocol.held_retirements_unread``); the shard judges against the
    sidecar -- the snapshot the tick exported before it planned -- so the holds a window had IN HAND are the holds of THAT snapshot, which the
    sidecar carries (its ``blocked_records`` and ``source_revision`` tables are the two the query reads). An earlier version read the live
    database here, after the export: a hold the harvester recorded between the two steps (another writer accepted a newer head of the source
    with an added input; the fold held the retirement) was named by the plan while the shard read the earlier head and watermarks -- an
    unchanged present input, no claim -- and the otherwise whole receipt named the hold and cleared it, so unchanged pins and module parked
    the adapter while the newer head's measurements stayed current after its input vanished. Read off the snapshot, such a hold is absent
    from this window's ``holds_seen`` and stands unread until a later window's export carries it. ``None`` (a caller with no sidecar: the
    refusal drill plans against a scratch root) records every adapter as having seen nothing -- the plan vouches for no hold and clears
    none; a sidecar that cannot be opened, is not one, or was not exported for one of ``adapters`` raises :class:`ClusterReadsError` -- a
    plan that cannot read the snapshot's queue says so instead of recording that it saw nothing."""
    out: Dict[str, List[Dict[str, str]]] = {str(a): [] for a in adapters}
    if not out or sidecar is None:
        return out
    try:
        conn = RS.open_reader_state(sidecar, list(out))
    except (RS.ReaderStateError, sqlite3.Error) as exc:
        raise ClusterReadsError(f"plan: the reader-state sidecar {sidecar} could not be opened for the standing retirement holds: {exc}") from exc
    try:
        for a in out:
            try:
                holds = H.retirements_held_for_moved_head(conn, a)
            except sqlite3.Error as exc:
                raise ClusterReadsError(f"plan: the standing retirement holds of {a} could not be read from the sidecar {sidecar}: {exc}") from exc
            out[a] = [H.retirement_hold_identity(h) for h in holds]
    finally:
        conn.close()
    return out


def holds_seen_source(sidecar: Optional[str]) -> Optional[Dict[str, object]]:
    """The snapshot a plan's ``holds_seen`` was read from -- ``plan["holds_seen_from"]``, copied onto the receipt: the sidecar's path and
    its meta row's ``built_utc`` and ``ledger_head`` (the live database's record-ingest head at the export); None when the plan had no
    sidecar and so recorded no hold seen. An unreadable sidecar raises :class:`ClusterReadsError`, as :func:`holds_seen_of` does."""
    if sidecar is None:
        return None
    try:
        conn = RS.open_reader_state(sidecar)
    except (RS.ReaderStateError, sqlite3.Error) as exc:
        raise ClusterReadsError(f"plan: the reader-state sidecar {sidecar} could not be opened: {exc}") from exc
    try:
        m = RS.meta(conn)
    finally:
        conn.close()
    return {"sidecar": os.path.abspath(sidecar), "built_utc": m.get("built_utc"), "ledger_head": m.get("ledger_head")}


def plan_window(root: str, window: str, adapters: Sequence[str], cfg, dest, tree_sha: str, uri_home: Optional[str] = None,
                repo_root: Optional[str] = None, full: bool = False, published: Optional[Mapping[str, str]] = None,
                full_adapters: Sequence[str] = (), reread_force_adapters: Sequence[str] = (),
                attestations: Optional[Mapping[str, Optional[str]]] = None, registry: Optional[Mapping[str, Callable[[], Any]]] = None,
                sidecar: Optional[str] = None) -> Dict[str, Any]:
    """The window's plan: per adapter its pinned generations (the published ``current`` of every component, from ``state/store_mirror.json``),
    admission pass 1 on the grading host (``stores.revalidate`` of the canonical copy against each pin), its READ POLICY (the
    window's ``--retry-refused`` always; ``--full`` for ``full_adapters``; ``--reread --force`` for ``reread_force_adapters`` -- a store read
    on the grading host, an operator override, so those are recorded refused with the reason) and the ROUTE -- ``cluster`` with K >= 1 shards, or
    ``grading_host`` = NOT READ THIS WINDOW, with the reason (an undeclared adapter, an entry declared ``cluster: refused`` -- recognized FIRST,
    before any admission, with the declaration's reason (:data:`REFUSED_DECL_PREFIX`) -- a grading-host-only mode, no published generation, a
    refused component, a generation attesting a directory link whose target holds no member -- :func:`dangling_links`, from the manifest
    copy alone). A ``cluster`` route always carries at least one task: an entry with none is never a read
    (:func:`cluster_routes_without_tasks`, checked by the submission, the refusal set and the pull). The plan runs NO discovery (the
    grading host reads no store; coverage is the shard's). Then ONE home tree for the window from the
    union of every cluster-routed adapter's pins (:func:`home_tree_spec`; a conflict records them all refused by name) and the sha of
    the configuration the plan was built from (staged beside it). Tasks = one per (adapter, shard) in adapter order. Every
    ``grading_host`` route is a recorded REFUSAL (``plan["refusals"]``, ``plan: <reason>``). Written to ``<root>/state/cluster_reads/<window>.json``
    by the caller.

    ``registry`` (``{name: factory}``; the CLI's ``plan`` and the tick pass ``adapters.load_all()``) enables THE PLAN-TIME CHECKS ON THE
    CONSTRUCTED INSTANCE (:func:`plan_time_checks`): the adapter is constructed -- never discovered, no store
    opened, the rule ``stores coverage`` already follows -- and refused by name when it declares ``rng_scope = reader_traversal`` with
    other than exactly one shard (the shard runner admits such an adapter at K = 1 only; a larger count crashed every task of three adapters
    before any summary was written), or when a root it declares it opens is not covered by the entry's components (the plan pins
    generations from the entry alone, so an uncovered root is read nowhere on the node). None (the default) constructs nothing:
    the plan discovers nothing either way.

    ``sidecar`` is the READER-STATE SIDECAR exported for this window's shards (the tick's, the landing window's ``plan --reader-state``):
    every adapter entry's ``holds_seen`` -- the harvester's standing retirement holds the window has in hand -- is read off it, the snapshot
    the shards judge against, never the live database (:func:`holds_seen_of`; registry-hardening 3a, closing pass), and
    ``plan["holds_seen_from"]`` names the snapshot (:func:`holds_seen_source`). None records no hold seen and ``holds_seen_from`` None: the
    plan vouches for no hold, and the reads protocol clears none on its receipt."""
    if published is None:
        published = {slug: rec.get("published") for slug, rec in (ST.load_state(root).get("components") or {}).items() if rec.get("published")}
    if attestations is None:
        attestations = ST.published_attestations(root)          # the attestation each published generation was verified against
    shared = dest.shared
    window_root = f"{shared}/{READS_REL}/{window}"
    gen_dir = f"{dest.stores_dir}/gen"
    plan: Dict[str, Any] = {"schema": PLAN_SCHEMA, "window": window, "built_utc": R.utc_now(), "uri_home": (uri_home or os.path.expanduser("~")).rstrip("/"),
                            "repo_root": (repo_root or ST.REPO_ROOT).rstrip("/"), "tree_sha": tree_sha, "tree": f"{shared}/{TREES_REL}/{tree_sha}",
                            "shared": shared, "host": dest.host, "destination": dest.name, "window_root": window_root, "root": f"{window_root}/root",
                            "reader_state": f"{window_root}/reader_state.sqlite", "summaries_dir": window_root, "gen_dir": gen_dir,
                            "rulings_mirror": f"{shared}/{RULINGS_MIRROR_REL}", "config": cfg.path, "config_sha256": R.file_sha256(cfg.path),
                            "config_staged": f"{window_root}/{CONFIG_STAGED}", "attestation": f"{window_root}/{ST.ATTESTATION_FILE}", "full": bool(full),
                            "home_tree": None, "holds_seen_from": None, "adapters": {}, "tasks": [], "refusals": {}, "stages": {},
                            "site_env": shard_site_env()}
    # the site's alias table (site_aliases.toml beside the registry root), staged into the shard root so a shard reads state written
    # under the site's older cluster and host names as the grading host does; None when the site has none
    aliases = os.path.join(root, "site_aliases.toml")
    plan["site_aliases"] = aliases if os.path.isfile(aliases) else None
    index = 0
    for name in adapters:
        entry: Dict[str, Any] = {"route": "grading_host", "reason": None, "refused": False, "shards": 0, "pins": {}, "attestations": {}, "revalidation_pre": {},
                                 "revalidation_post": {}, "task_indices": [], "holds_seen": [],
                                 "policy": {"retry_refused": True, "full": bool(full) or name in full_adapters, "reread_force": name in reread_force_adapters}}
        plan["adapters"][name] = entry
        decl = cfg.adapters.get(name)
        if decl is None:
            entry["reason"] = f"{name} is not declared in {cfg.path}"
            continue
        if decl.refused:
            # declared NOT readable on the cluster: the declaration's reason is the route's, recorded before any admission or
            # discovery -- nothing is pinned, revalidated or discovered for it; the grading host reads it this window
            entry["refused"] = True
            entry["reason"] = REFUSED_DECL_PREFIX + decl.reason
            continue
        entry["shards"] = int(decl.shards)
        if entry["shards"] < 1:
            # the loader refuses this shape; a configuration that reaches here some other way still never yields a task-less cluster route
            entry["reason"] = f"{name} declares {entry['shards']} shard(s) in {cfg.path}; a cluster route needs at least one task"
            continue
        if entry["policy"]["reread_force"]:
            entry["reason"] = REREAD_FORCE_REASON
            continue
        missing = [s for s in decl.components if not published.get(s)]
        entry["pins"] = {s: str(published[s]) for s in decl.components if published.get(s)}
        entry["attestations"] = {s: (str(attestations[s]) if attestations.get(s) else None) for s in entry["pins"]}   # the pin's attestation
        if missing:
            entry["reason"] = "no published generation for component(s): " + ", ".join(missing)
            continue
        results = ST.adapter_admission(cfg, name, entry["pins"], root=root, attestations=entry["attestations"])
        entry["revalidation_pre"] = {s: rv.as_dict() for s, rv in results.items()}
        refused, why = _refused_components(results)
        if refused:
            entry["reason"] = "admission refused component(s): " + why
            continue
        # no grading host discovery here (C6, S2): the shard's discovery over the home tree is the coverage check, and its refusal is the record.
        # The one coverage fact the MANIFEST COPY alone can tell is checked: an attested directory link whose target holds no member dangles
        # on the node, and what the grading host reads through it would vanish from the shard's discovery without a word -- refused by name
        dangling = {}
        try:
            for slug, sha8 in sorted(entry["pins"].items()):
                for link, target in dangling_links(root, slug, sha8).items():
                    dangling[f"{slug}:{link}"] = target
        except ClusterReadsError as exc:
            entry["reason"] = f"the manifest copy could not be read for the coverage check: {exc}"
            continue
        if dangling:
            entry["reason"] = (f"{UNCOVERED_PREFIX}: the generation attests {len(dangling)} directory link(s) whose target holds no member, so the "
                               "staged copy would hide what the grading host reads through them -- "
                               + "; ".join(f"{k} -> {v}" for k, v in sorted(dangling.items())))
            entry["dangling_links"] = dangling
            continue
        if registry is not None:
            # the plan-time checks on the CONSTRUCTED instance, last before the route: the pins and the admission above keep
            # their precedence (a stale pin names the component, as the refusal drill expects), then the code's own declarations are checked
            checks = plan_time_checks(name, decl, cfg, registry)
            entry["plan_checks"] = checks
            if checks["refusal"]:
                entry["reason"] = checks["refusal"]
                continue
        entry["route"] = "cluster"
        for shard in range(int(decl.shards)):
            plan["tasks"].append({"index": index, "adapter": name, "shard": shard, "shards": int(decl.shards)})
            entry["task_indices"].append(index)
            index += 1
    cluster = [a for a, e in plan["adapters"].items() if e["route"] == "cluster"]
    if cluster:
        union: Dict[str, str] = {}
        for a in cluster:
            union.update(plan["adapters"][a]["pins"])          # one published map: a slug pins the same generation for every adapter
        try:
            plan["home_tree"] = home_tree_spec(cfg.components, union, gen_dir, plan["uri_home"], lambda s, g: manifest_entries(root, s, g),
                                               subentries_of=lambda s, g, d: manifest_entries(root, s, g, d), escapes_of=lambda s, g: generation_escapes(root, s, g))
        except ClusterReadsError as exc:
            for a in cluster:
                plan["adapters"][a].update(route="grading_host", reason=f"home tree: {exc}", task_indices=[])
            plan["tasks"] = []
    # the retirement holds the SIDECAR carries -- the snapshot exported for this window's shards, never the live database: on every
    # entry, whatever its route -- the receipt copies them, and the tick reads an ATTEMPTED entry's list as "this window's shards
    # judged the adapter with these holds in hand" (reads_protocol.held_retirements_unread). A hold the harvester records after the
    # export is not in the snapshot: absent here, it stands unread until a later window's export carries it
    seen = holds_seen_of(sidecar, list(plan["adapters"]))
    plan["holds_seen_from"] = holds_seen_source(sidecar)
    for name, entry in plan["adapters"].items():
        entry["holds_seen"] = list(seen.get(name) or [])
        if entry["route"] == "grading_host":
            plan["refusals"][name] = [f"plan: {entry['reason']}"]
    return plan


def revalidate_plan(root: str, plan: Dict[str, Any], cfg) -> List[str]:
    """Admission pass 2, after the array and BEFORE the pulled revisions are merged: every cluster-routed adapter's components
    re-validated against their pins; a difference names the adapter for quarantine and records its refusal. Returns the adapters refused."""
    stale: List[str] = []
    for name, entry in plan["adapters"].items():
        if entry.get("route") != "cluster":
            continue
        results = ST.adapter_admission(cfg, name, entry["pins"], root=root, attestations=entry.get("attestations") or {})
        entry["revalidation_post"] = {s: rv.as_dict() for s, rv in results.items()}
        refused, why = _refused_components(results)
        if refused:
            stale.append(name)
            plan.setdefault("refusals", {}).setdefault(name, []).append("post-run revalidation refused component(s): " + why)
    record_stage(plan, "revalidate_post", stale=stale)
    return stale


# --------------------------------------------------------------------------------------------
# staging and submission (the grading host side; the grading host initiates every connection)
# --------------------------------------------------------------------------------------------


def _is_local(host: Optional[str]) -> bool:
    return not host or host == "local"


def _ssh(host: Optional[str], script: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, timeout_s: float = 300.0,
         input: Optional[str] = None) -> subprocess.CompletedProcess:
    argv = ["bash", "-c", script] if _is_local(host) else ["ssh", *SSH_OPTS, host, script]
    return runner(argv, capture_output=True, text=True, timeout=timeout_s, input=input)


def _remote(host: Optional[str], path: str) -> str:
    return path if _is_local(host) else f"{host}:{path}"


def stage_window(plan: Dict[str, Any], sidecar_path: str, plan_file: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 timeout_s: float = 1800.0) -> Dict[str, Any]:
    """Lay the window root out on the cluster: ``root/records`` (the spool), ``root/rulings`` -> an ABSOLUTE symlink onto the record-log
    mirror's rulings (verified: ``realpath`` equals the mirror and the mirror is non-empty), then rsync the sidecar, the plan and
    the exact configuration the plan was built from (its sha checked first) into the window root. Every step refuses by name."""
    host, window_root, root, mirror = plan.get("host"), plan["window_root"], plan["root"], plan["rulings_mirror"]
    config = str(plan.get("config") or "")
    if not os.path.isfile(config):
        raise ClusterReadsError(f"stage: the configuration the plan was built from, {config}, is not a file")
    # the rule every shard applies to the staged copy (config_identity_refusal): equal bytes, so a window that stages is one its shards admit
    why = config_identity_refusal(plan.get("config_sha256"), config)
    if why:
        raise ClusterReadsError(f"stage: {config} changed since the plan was built ({why}); plan again")
    script = (f"set -e; mkdir -p {shlex.quote(root)}/records {shlex.quote(root)}/state; "
              f"[ -d {shlex.quote(mirror)} ] || {{ echo 'ERR no rulings mirror at {mirror}'; exit 2; }}; "
              f"[ -n \"$(ls -A {shlex.quote(mirror)})\" ] || {{ echo 'ERR rulings mirror {mirror} is empty'; exit 2; }}; "
              f"ln -sfn {shlex.quote(mirror)} {shlex.quote(root)}/rulings; "
              f"echo RULINGS \"$(readlink -f {shlex.quote(root)}/rulings)\"; "
              f"ls {shlex.quote(mirror)} | wc -l")
    r = _ssh(host, script, runner, timeout_s)
    if r.returncode != 0 or "ERR" in (r.stdout + r.stderr):
        raise ClusterReadsError(f"stage: laying out {window_root} failed (rc {r.returncode}): {(r.stdout + r.stderr).strip()[-800:]}")
    resolved = next((line.split(" ", 1)[1] for line in r.stdout.splitlines() if line.startswith("RULINGS ")), "")
    if os.path.normpath(resolved) != os.path.normpath(mirror):
        raise ClusterReadsError(f"stage: {root}/rulings resolves to {resolved!r}, not the mirror {mirror}")
    n_rulings = int(r.stdout.strip().splitlines()[-1] or 0)
    staged = [(sidecar_path, "reader_state.sqlite"), (plan_file, "plan.json"), (config, CONFIG_STAGED)]
    if plan.get("site_aliases"):
        staged.append((str(plan["site_aliases"]), "root/site_aliases.toml"))       # beside the shard's registry root
    for src, name in staged:
        cmd = ["rsync", "-t", "--timeout=600"] + ([] if _is_local(host) else ["-e", RSYNC_SSH]) + [src, _remote(host, f"{window_root}/{name}")]
        r2 = runner(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r2.returncode != 0:
            raise ClusterReadsError(f"stage: rsync of {name} into {window_root} failed (rc {r2.returncode}): {r2.stderr.strip()[-800:]}")
    return {"window_root": window_root, "rulings_resolved": resolved, "rulings_entries": n_rulings, "sidecar_bytes": os.path.getsize(sidecar_path)}


def _sbatch_job_id(r: subprocess.CompletedProcess, what: str) -> str:
    if r.returncode != 0:
        raise ClusterReadsError(f"submit: {what} sbatch failed (rc {r.returncode}): {(r.stdout + r.stderr).strip()[-800:]}")
    job = (r.stdout or "").strip().splitlines()[-1].split(";")[0].strip() if (r.stdout or "").strip() else ""
    if not job.isdigit():
        raise ClusterReadsError(f"submit: {what} sbatch printed no job id: {(r.stdout or '').strip()[-200:]!r}")
    return job


def submit_attest(plan: Dict[str, Any], runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, timeout_s: float = 300.0,
                  job_name: Optional[str] = None) -> str:
    """``sbatch --parsable`` the operation's ATTESTATION job (C5; ``attest.sbatch WINDOW_ROOT TREE``: ``stores.attest_plan`` over every pinned
    generation of the plan, the record into ``<window root>/attestation.json``), positional arguments only (never ``--export``). Returns
    its job id -- what :func:`submit_array` makes the array depend on."""
    if not (plan.get("tasks") or []):
        raise ClusterReadsError("submit: the plan has no cluster task (every adapter routes to the grading host); nothing to attest")
    sbatch = f"{plan['tree']}/scripts/registry/landing/attest.sbatch"
    name = job_name or f"attest-{plan.get('window')}"
    script = f"sbatch --parsable -J {shlex.quote(name)} {shlex.quote(sbatch)} {shlex.quote(plan['window_root'])} {shlex.quote(plan['tree'])}"
    return _sbatch_job_id(_ssh(plan.get("host"), script, runner, timeout_s), "attestation")


def submit_array(plan: Dict[str, Any], venv: str, job_name: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 timeout_s: float = 300.0, local_venv: bool = False, dependency: Optional[str] = None) -> str:
    """``sbatch --parsable`` the read array on the cluster: one task per plan task, positional arguments only (never ``--export``);
    ``local_venv`` asks each task for a node-local copy of the venv (opt-in: the torch-cpu gate venv's copy was measured at > 7 min).
    ``dependency`` is the attestation job's id (C5): the array runs ``--dependency=afterok:<id>`` and ``--kill-on-invalid-dep=yes``, so a
    crashed attestation (no record written, non-zero exit) cancels the array instead of leaving it pending forever -- the window then
    finds no shard summaries and every cluster adapter falls back by name; an attestation that REFUSED a generation exits 0 with the record,
    the array runs, and each shard reading that pin refuses by name. Returns the array job id."""
    n = len(plan.get("tasks") or [])
    if n == 0:
        raise ClusterReadsError("submit: the plan has no cluster task (every adapter routes to the grading host)")
    sbatch = f"{plan['tree']}/scripts/registry/landing/read_array.sbatch"
    dep = f"--dependency=afterok:{shlex.quote(str(dependency))} --kill-on-invalid-dep=yes " if dependency else ""
    script = (f"sbatch --parsable -J {shlex.quote(job_name)} --array=0-{n - 1} {dep}{shlex.quote(sbatch)} "
              f"{shlex.quote(plan['window_root'])} {shlex.quote(plan['tree'])} {shlex.quote(venv)} {1 if local_venv else 0}")
    return _sbatch_job_id(_ssh(plan.get("host"), script, runner, timeout_s), "array")


# --------------------------------------------------------------------------------------------
# the transport: the grading host pulls, marker last, behind the barrier
# --------------------------------------------------------------------------------------------


def _rsync(argv: Sequence[str], runner: Callable[..., subprocess.CompletedProcess], timeout_s: float) -> subprocess.CompletedProcess:
    try:
        return runner(list(argv), capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(list(argv), 255, stdout="", stderr=f"{type(exc).__name__}: {exc}")


def _stderr_tail(r: subprocess.CompletedProcess, lines: int = 3, width: int = 400) -> str:
    """The last ``lines`` of a transport's stderr on one line (the barrier's reason carries rsync's own words), bounded."""
    tail = " | ".join(ln.strip() for ln in str(r.stderr or "").strip().splitlines()[-lines:] if ln.strip())
    return tail[:width]


def install_received(recv: str, expect_dir: str) -> None:
    """Install a pull attempt's receive directory as the expect directory, WHOLE: the previous attempt's directory, if any,
    moves aside to ``<expect_dir>.prev`` (replacing an older ``.prev``) and ``recv`` takes its place by one rename. rsync without
    ``--delete`` leaves a previous attempt's files where the source no longer has them, so a retry received INTO the standing directory
    could read an earlier attempt's attestation record (or shard summary) as this attempt's; received fresh and installed whole, a file
    absent from the source this attempt is absent here."""
    prev = expect_dir.rstrip("/") + ".prev"
    if os.path.isdir(prev) and not os.path.islink(prev):
        shutil.rmtree(prev)
    elif os.path.lexists(prev):
        os.remove(prev)
    if os.path.lexists(expect_dir):
        os.rename(expect_dir, prev)
    os.rename(recv, expect_dir)


def _revision_dirs(top: str) -> List[str]:
    """Every revision directory (``<a>/<rev16>``, ``<a>/_activations/<rev16>/<stamp>`` or a refusal record's ``<a>/_refused/<rev16>``) under
    an ``adapters/`` tree, relative."""
    out: List[str] = []
    if not os.path.isdir(top):
        return out
    for a in sorted(os.listdir(top)):
        ad = os.path.join(top, a)
        if not os.path.isdir(ad):
            continue
        for name in sorted(os.listdir(ad)):
            p = os.path.join(ad, name)
            if RS.REV16_RE.match(name) and os.path.isdir(p):
                out.append(f"{a}/{name}")
            elif name == REFUSED_DIR and os.path.isdir(p):
                out += [f"{a}/{REFUSED_DIR}/{rev}" for rev in sorted(os.listdir(p)) if RS.REV16_RE.match(rev) and os.path.isdir(os.path.join(p, rev))]
            elif name == "_activations" and os.path.isdir(p):
                for rev in sorted(os.listdir(p)):
                    rd = os.path.join(p, rev)
                    if RS.REV16_RE.match(rev) and os.path.isdir(rd):
                        out += [f"{a}/_activations/{rev}/{stamp}" for stamp in sorted(os.listdir(rd)) if os.path.isdir(os.path.join(rd, stamp))]
    return out


def _listing(d: str) -> Dict[str, int]:
    return {fn: int(os.path.getsize(os.path.join(d, fn))) for fn in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, fn))}


def _inbox_shas(d: str) -> Dict[str, str]:
    """name -> sha256 of every file in a revision directory as it stands in the inbox (an earlier delivery reused by this operation: the bytes
    the merge must move, which are not this window's shard summary's)."""
    return {fn: R.file_sha256(os.path.join(d, fn)) for fn in _listing(d)}


def _depends_entry(adapter: str, how: str, files: Mapping[str, str]) -> Dict[str, Any]:
    """One ``report["depends"]`` entry: ``how`` is ``copied`` (this window's verified copy) or ``reused`` (an earlier complete
    delivery found in the inbox); ``files`` the name -> sha256 the merge must leave in ``records/`` for the revision to count as merged."""
    return {"adapter": adapter, "how": how, "files": {str(k): str(v) for k, v in sorted(files.items())}}


def expected_from_summaries(expect_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """``{revision dir: {adapter, files, sha256, marker, summary, ok, refusal}}`` over every shard summary under ``expect_dir/<adapter>/*.json``
    (``files``: name -> size; ``sha256``: name -> digest, ``shard_summary/2`` -- empty for a summary that carries none, which the pull
    refuses by name), and the summaries per adapter (with their paths) so the caller can count the shards that reported."""
    expected: Dict[str, Dict[str, Any]] = {}
    per_adapter: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(expect_dir):
        return expected, per_adapter
    for a in sorted(os.listdir(expect_dir)):
        ad = os.path.join(expect_dir, a)
        if not os.path.isdir(ad):
            continue
        for fn in sorted(os.listdir(ad)):
            if not fn.endswith(".json"):
                continue
            try:
                doc = _load_json(os.path.join(ad, fn))
            except (OSError, ValueError, ClusterReadsError) as exc:
                per_adapter.setdefault(a, []).append({"_path": os.path.join(ad, fn), "error": str(exc)})    # the same key a readable one carries
                continue
            doc["_path"] = os.path.join(ad, fn)
            per_adapter.setdefault(a, []).append(doc)
            # a published revision directory and a `_refused/<rev>/` record travel the same way: listed with every file's size
            # and digest, pulled by ownership, verified on receipt, merged into records/ for the harvester
            for kind, key in (("revision", "published"), ("refused", "refused_records")):
                for rev_dir, info in (doc.get(key) or {}).items():
                    expected[rev_dir] = {"adapter": info.get("adapter", a), "files": {k: int(v) for k, v in (info.get("files") or {}).items()},
                                         "sha256": {k: str(v) for k, v in (info.get("sha256") or {}).items()},
                                         "marker": info.get("marker"), "summary": doc["_path"], "ok": doc.get("ok") is True,
                                         "refusal": doc.get("refusal"), "kind": kind}
    return expected, per_adapter


def pull_window(root: str, plan: Dict[str, Any], expect_dir: str, inbox_name: str = INBOX_NAME, source: Optional[str] = None,
                runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, timeout_s: float = 1800.0) -> Dict[str, Any]:
    """The two-pass pull with the failure barriers (module docstring).
    ``source`` overrides ``<host>:<window root>`` (a local directory in tests). Pass 0 receives the shard summaries and the operation's
    attestation record into a FRESH per-attempt directory and installs it as ``expect_dir`` whole (:func:`install_received`: the previous
    attempt's directory becomes ``<expect_dir>.prev``), so a retry over the same window never reads an earlier attempt's record. A pass 0
    that did not return 0 ends the pull (``barrier``): nothing is pulled or replayed and every cluster adapter is refused with the rc and
    rsync's stderr tail. Otherwise a revision is delivered only when THREE prerequisites hold: its adapter passed the post-run revalidation
    (``stages.revalidate_post``, recorded by ``cluster-reads revalidate``; none recorded = nothing is delivered), its publishing shard's
    summary attests ``ok``, AND the pulled attestation record ADMITS its adapter (:func:`attestation_premise`; FAIL-CLOSED:
    a readable record of this operation naming every pin of the adapter at its pinned generation, ok). Pass 1 pulls BY OWNERSHIP:
    exactly the revision directories this window's summaries list and the prerequisites do not withhold (``--files-from`` of directories,
    markers excluded); a withheld revision already in the inbox is quarantined (a stale adapter's under ``_stale/``, a refusing shard's and
    an unadmitted adapter's under ``_failed/``); a directory in the inbox that no summary of this window lists is left for inbox-merge and
    reported (``unlisted``), never quarantined. Pass 2 pulls the eligible markers alone (pass 1 returned 0, the inbox directory holds
    exactly the listed batches, a marker is named). Returns the report: per revision directory ``pulled`` / ``quarantined`` / ``withheld``
    (with the reason) and ``unlisted``, the adapters recorded REFUSED (``adapters_refused``, with the reasons; nothing reads them
    elsewhere), the refused-known revision ids the shards returned (``refused_known_ids``, per adapter: the grading host's to re-activate from
    committed bytes, :func:`reactivate_refused_known`) -- GATED like a publication: only an id a shard attesting ``ok`` returned,
    for an adapter that passed the post-run revalidation and that this operation's attestation record admits, over a summary transport
    that returned 0; every other id is listed under ``refused_known_withheld`` (per adapter ``{revision id: reason}``) and never replayed
    -- the ``barrier`` (or None) and the rsync return codes."""
    host = plan.get("host")
    window_root = source if source is not None else _remote(host, plan["window_root"])
    local_source = source is not None and ":" not in source.split("/", 1)[0]
    ssh_e = [] if local_source else ["-e", RSYNC_SSH]
    window = str(plan.get("window") or "window")
    inbox = os.path.join(H.inbox_dir(root), inbox_name)
    dest_adapters = os.path.join(inbox, "adapters")
    src_adapters, dst_adapters = window_root.rstrip("/") + "/root/records/adapters/", dest_adapters.rstrip("/") + "/"
    os.makedirs(dest_adapters, exist_ok=True)
    expect_dir = os.path.abspath(expect_dir)
    os.makedirs(os.path.dirname(expect_dir), exist_ok=True)
    report: Dict[str, Any] = {"window": window, "source": window_root, "expect_dir": expect_dir, "pulled": {}, "quarantined": {}, "withheld": {},
                              "unlisted": {}, "partial": {}, "partial_aside": {}, "adapters_refused": {}, "refused_known_ids": {}, "refused_known_withheld": {},
                              "refused_known_claims": {}, "standing_refusals": {},
                              "rsync": {}, "summaries": {}, "barrier": None, "utc": R.utc_now(),
                              # every revision directory whose MERGE this operation depends on -- newly copied AND an earlier complete
                              # delivery reused as it stands -- with the files (name -> sha256) the merge must leave in records/; the receipt's
                              # precondition (reads_protocol.merge_evidence) is read from records/ against THIS, never from scan_inbox's return
                              "depends": {}}
    adapters_cluster = [a for a, e in plan["adapters"].items() if e.get("route") == "cluster"]
    # the post-run revalidation is a PREREQUISITE of delivery: without its record nothing moves
    reval = (plan.get("stages") or {}).get("revalidate_post")
    if not isinstance(reval, dict):
        for a in adapters_cluster:
            report["adapters_refused"][a] = ["no post-run revalidation is recorded for this window (cluster-reads revalidate); nothing is delivered"]
        return report
    stale_adapters = {str(a) for a in (reval.get("stale") or [])}
    # 0. the shard summaries and the operation's attestation record, received into a FRESH per-attempt directory and installed whole
    # (:func:`install_received`): nothing a previous attempt over the same window received survives into this one, so a record
    # or summary absent from the window root at THIS pull is absent here too.
    # anchored includes: only the adapters' own directories are entered (an `*/` include would walk the whole spool for nothing)
    include = ([f"--include=/{a}/" for a in adapters_cluster] + [f"--include=/{a}/*.json" for a in adapters_cluster]
               + [f"--include=/{ST.ATTESTATION_FILE}", "--exclude=*"])
    recv = f"{expect_dir}.recv-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    os.makedirs(recv)
    r0 = _rsync(["rsync", "-rt", "--timeout=300", *ssh_e, "--prune-empty-dirs", *include, window_root.rstrip("/") + "/", recv + "/"], runner, timeout_s)
    report["rsync"]["summaries"] = r0.returncode
    install_received(recv, expect_dir)
    expected, per_adapter = expected_from_summaries(expect_dir)
    # the operation's attestation record (C5), pulled beside the summaries: what the shards read before anything else, re-checked HERE per
    # adapter (:func:`attestation_premise`, fail-closed): a record that is missing, unreadable, another operation's or does not admit a pin
    # voids every claim the adapter's shards made -- its publications and its refused-known ids alike -- and names why, so a window whose
    # array never ran (an attestation crash cancels it) still says why
    att_path = os.path.join(expect_dir, ST.ATTESTATION_FILE)
    att = ST.read_attestation(att_path)
    report["attestation"] = attestation_summary(att, window)
    attestation_void: Dict[str, str] = {}      # adapter -> why the operation's attestation does not admit it (nothing of it is delivered or replayed)
    # THE TRANSPORT BARRIER ON THE SUMMARIES. The summaries and the record are the evidence every later step reads; a transfer
    # that did not return 0 (a partial transfer: one adapter's summary unreadable while another's and the record arrived) cannot say which
    # claims are whole. So it is not a per-adapter reason but the end of this pull: no pass 1, no pass 2, nothing replayed, every cluster
    # adapter refused with the rc and rsync's own words, every listed revision withheld; a directory already in the inbox is left where it is
    # (an earlier clean delivery is inbox-merge's; a marker-less partial is never activated). A clean retry delivers.
    barrier_why = None
    if r0.returncode != 0:
        tail = _stderr_tail(r0)
        barrier_why = (f"the shard summaries and the attestation record could not be pulled whole (rsync rc {r0.returncode}"
                       + (f": {tail}" if tail else "") + "); nothing is delivered or replayed this pull")
        report["barrier"] = {"stage": "summaries", "rc": r0.returncode, "stderr_tail": tail}
    idle = set(cluster_routes_without_tasks(plan))
    for a in adapters_cluster:
        want = plan["adapters"][a]["shards"]
        docs = per_adapter.get(a, [])
        have = [d for d in docs if "error" not in d]
        standing = [x for d in have for x in (d.get("standing_refusals") or []) if isinstance(x, dict) and x.get("group")]
        report["summaries"][a] = {"expected": want, "present": len(have), "ok": sum(1 for d in have if d.get("ok") is True),
                                  "refused": [d.get("refusal") for d in have if d.get("ok") is not True],
                                  "refused_known": sum(len(d.get("refused_known") or []) for d in have),
                                  "refused_groups": sum(len(d.get("refused") or []) for d in have),
                                  "inputs_dropped": sum(1 for d in have for x in (d.get("refused") or []) if isinstance(x, dict) and x.get("inputs_dropped")),
                                  "standing_refusals": len(standing),
                                  "held": sum(len(d.get("held") or []) for d in have), "unreadable": [d["_path"] for d in docs if "error" in d]}
        if standing:
            # a refusal re-confirmed byte for byte under the same inputs and tree is a RESULT, recorded for the receipt and the status
            # page (`state = standing_refusal`, the words, the prior refusal it equals) and never a reason below -- a NEW or CHANGED refusal is in
            # `refused_groups`
            report["standing_refusals"][a] = [{"group": str(x.get("group")), "revision_id": str(x.get("revision_id")), "state": STANDING_REFUSAL,
                                               "refusal": x.get("refusal"), "prior_refusal": dict(x.get("prior_refusal") or {})}
                                              for x in sorted(standing, key=lambda x: str(x.get("group")))]
        s = report["summaries"][a]
        reasons = []
        # the adapter-level failures that void EVERY claim its shards made this window (the refused-known gate below)
        premise: List[str] = []
        if a in idle:
            reasons.append(NO_TASK_REASON)
        if r0.returncode != 0:
            # the whole-pull barrier below: this is every adapter's reason, and the pull ends after this loop
            reasons.append(barrier_why)
            premise.append(barrier_why)
        if s["present"] < want:
            reasons.append(f"{want - s['present']} of {want} shard summaries missing")
        if s["unreadable"]:
            reasons.append(f"{len(s['unreadable'])} shard summaries unreadable")
        if s["ok"] < s["present"]:
            reasons.append(f"{s['present'] - s['ok']} shard(s) refused: " + "; ".join(str(x) for x in s["refused"] if x))
        if s["inputs_dropped"]:
            # the staged copy lacked an input the grading host read; the shard's own words (the paths and the components) travel
            first = next(str(x.get("error") or "") for d in have for x in (d.get("refused") or []) if isinstance(x, dict) and x.get("inputs_dropped"))
            reasons.append(f"{s['inputs_dropped']} group(s) refused on the cluster: the staged copy lacks input(s) the grading host read -- {first}"
                           + (f" (+{s['inputs_dropped'] - 1} more group(s))" if s["inputs_dropped"] > 1 else ""))
        if s["refused_groups"] > s["inputs_dropped"]:
            reasons.append(f"{s['refused_groups'] - s['inputs_dropped']} group(s) refused by the reader on the cluster")
        if a in stale_adapters:
            reasons.append("post-run revalidation refused a component (the canonical copy moved under the read); nothing of it is delivered")
            premise.append("post-run revalidation refused the adapter (the canonical copy moved under the read)")
        # THE ATTESTATION PREMISE (FAIL-CLOSED). The shard's admission re-run on the grading host over the record the pull brought
        # back: no readable record of THIS operation, or a pin of the adapter it does not name, attests at another generation or refused,
        # and nothing of the adapter is delivered or replayed. Before this the gate fired only on a window mismatch or a refused pin of a
        # PRESENT record: a missing or unreadable record (present = False) passed every claim of an `ok` shard through, and a superseded
        # revision could be re-activated into the live log on a pull with no verifiable attestation.
        pins = {str(s_): str(g_) for s_, g_ in (plan["adapters"][a].get("pins") or {}).items()}
        void = attestation_premise(att, att_path, window, pins)
        if void is not None:
            reasons.append(void)
            premise.append(void)
            attestation_void[a] = void
        # THE REFUSED-KNOWN GATE. A refused-known id is a CLAIM of the shard that returned it -- the id it minted from the bytes
        # it read is one the grading host holds COMMITTED (accepted, or held / unfolded; an id the grading host refused at emit time
        # and never committed is read fresh by the shard, never returned) -- and its replay
        # (:func:`reactivate_refused_known` -> ``runner.reactivate_by_id``) writes the LIVE record log directly, past the inbox barrier:
        # the window's later refusal does not undo it. So the claim is trusted exactly as far as the shard's publications would be
        # delivered and no further: the shard attests ``ok``, its adapter passed the post-run revalidation, the attestation record
        # pulled is this operation's and admits every pin the adapter reads (the premise above), and the summary transport returned 0.
        # Any other id is WITHHELD by name (``refused_known_withheld``, with the reason; the adapter is refused for the cause already).
        # Before the gate a stale adapter's or a refused shard's ids were replayed, and a superseded revision became the head on a read
        # the window refused. A refused-known group is otherwise not a refusal of the adapter: the id is the grading host's to
        # re-activate.
        trusted: set = set()
        withheld: Dict[str, str] = {}
        claims: Dict[str, Dict[str, Any]] = {}
        for d in have:
            shard_why = None if d.get("ok") is True else f"the shard that returned it is not ok ({d.get('refusal') or 'no attestation'})"
            for x in d.get("refused_known") or []:
                if not (isinstance(x, dict) and x.get("revision_id")):
                    continue
                rid = str(x["revision_id"])
                why = premise[0] if premise else shard_why
                if why is None:
                    trusted.add(rid)
                    if isinstance(x.get("retirement"), dict):
                        # a RETIREMENT's claim (runner.retirement_claim) travels with the id: the replay withholds the retirement unless the
                        # claim is exactly the live state (registry-hardening 3a, round 1); an id with no claim is replayed and withheld by name
                        claims[rid] = dict(x["retirement"])
                else:
                    withheld[rid] = why
        report["refused_known_ids"][a] = sorted(trusted - set(withheld))       # an id a withheld shard also returned stays withheld
        kept_claims = {rid: claims[rid] for rid in report["refused_known_ids"][a] if rid in claims}
        if kept_claims:
            report["refused_known_claims"][a] = kept_claims
        if withheld:
            report["refused_known_withheld"][a] = dict(sorted(withheld.items()))
            reasons.append(f"{len(withheld)} refused-known revision(s) withheld from re-activation: " + "; ".join(sorted(set(withheld.values()))))
        if reasons:
            report["adapters_refused"][a] = reasons
    if barrier_why is not None:
        # the whole-pull barrier (above): every listed revision is named withheld, nothing moves
        report["withheld"].update({rev_dir: barrier_why for rev_dir in expected})
        return report
    # what is never delivered: a stale adapter's directories, a refusing shard's publications, and every publication of an adapter
    # the operation's attestation does not admit on the grading host's re-check -- never pulled, quarantined if already present
    never: Dict[str, str] = {}
    for rev_dir, exp in expected.items():
        if exp["adapter"] in stale_adapters:
            report["withheld"][rev_dir] = "post-run revalidation refused the adapter"
        elif not exp["ok"]:
            never[rev_dir] = f"the shard that published it is not ok ({exp.get('refusal') or 'no attestation'})"
        elif exp["adapter"] in attestation_void:
            never[rev_dir] = attestation_void[exp["adapter"]]
    report["withheld"].update(never)
    # 1. pass 1, BY OWNERSHIP: exactly the revision directories THIS window's summaries list and nothing above withholds --
    # a --files-from of directories, recursive, markers excluded. The inbox under `adapters/` is shared across windows, and a directory this
    # window's summaries do not name (an earlier window's delivery interrupted before its inbox-merge, or a summary that never arrived) is
    # neither pulled, marked nor quarantined here: it stays for inbox-merge (a complete one is merged and folds; a marker-less one is left by
    # inbox-merge too, `harvest.scan_inbox`'s marker rule, for a window whose summaries list the id to complete). Before this, pass 1 pulled
    # the whole spool behind excludes and the inbox loop quarantined every directory of the adapter it found, an earlier window's valid,
    # marker-bearing delivery included -- stranded under _failed/.
    # A LISTED directory that already holds its marker in the inbox is an earlier COMPLETED delivery of this very revision id:
    # this window's pass 2 has not run, so the marker can only be an earlier window's (or an earlier attempt's) whose inbox-merge never ran,
    # and the shard that listed it minted the same id from the same inputs because `reader_state.published_dirs_of` reads records/, not the
    # inbox. It is complete as it stands: reported so, not re-pulled (pass 1 would mix this window's bytes into it), never compared against
    # this window's batch list (the marker would read as an extra file), never quarantined; inbox-merge takes it.
    for rev_dir in sorted(expected):
        if rev_dir in report["withheld"] or expected[rev_dir].get("kind") == "refused":
            continue            # a refusal record directory is shared by every window that refuses the id: an earlier window's file there is no delivery of this one's
        d = os.path.join(dest_adapters, rev_dir)
        if os.path.isdir(d) and any(MARKER_TOKEN in fn for fn in os.listdir(d)):
            report["pulled"][rev_dir] = {"state": "complete (an earlier delivery already in the inbox)", "adapter": expected[rev_dir]["adapter"],
                                         "files": len(_listing(d))}
            report["depends"][rev_dir] = _depends_entry(expected[rev_dir]["adapter"], "reused", _inbox_shas(d))     # the inbox copy's own bytes
    deliverable = sorted(rev_dir for rev_dir in expected if rev_dir not in report["withheld"] and rev_dir not in report["pulled"])
    # THE ATTEMPT-LOCAL RECEIVE DIRECTORY. Passes 1 and 2 land every deliverable revision under `inbox/<name>/.recv-<window>-…/`,
    # never in the shared `adapters/` tree. Before this the inbox directory itself was the staging destination: a marker-less partial an
    # earlier window had left there (its pass 1 landed, its pass 2 never ran; the standing merge rightly leaves it; the next export cannot
    # see it, so the next shard legitimately reissues the SAME id) kept its batches under pass 1 `--ignore-existing` -- same names, same
    # sizes, DIFFERENT bytes: the earlier window's written_utc / observed_utc, possibly other facts -- and pass 2 brought this window's marker
    # over them, so a mixed revision folded under this window's provenance. Now a revision enters the inbox only WHOLE: every listed batch
    # and the marker received here and verified by size AND sha256 against the shard summary (`shard_summary/2`), then one rename; the
    # earlier partial is set aside under `_partial/<window>/` first (never deleted, for diagnosis). `scan_inbox` skips the receive directory
    # (a dot-prefixed top-level entry) like the `_` ones.
    recv_inbox = os.path.join(inbox, f"{RECV_PREFIX}{window}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    os.makedirs(recv_inbox, exist_ok=True)
    # THE RECEIVING MANIFEST: everything a later window needs to finish this delivery if this process dies between pass 2 and
    # the install -- the window, where its summaries and attestation record were, the revalidation verdict, the attestation premise per
    # adapter (and the pins it was computed from), and per listed revision the files, sizes, digests and marker the summary promised.
    # :func:`recover_receive_dirs` (the window's `-- cluster reads inbox-merge` stage, before the standing merge) re-verifies against THIS
    # record, never against the shared inbox, and installs or fails loudly by name.
    _write_json(os.path.join(recv_inbox, RECEIVING_FILE), {
        "schema": RECEIVING_SCHEMA, "window": window, "utc": R.utc_now(), "pid": os.getpid(), "expect_dir": expect_dir, "attestation": att_path,
        "revalidate_post": {"stale": sorted(stale_adapters)}, "attestation_void": dict(attestation_void),
        "pins": {a: {str(s_): str(g_) for s_, g_ in (plan["adapters"][a].get("pins") or {}).items()} for a in adapters_cluster},
        "revisions": {rev_dir: {"adapter": expected[rev_dir]["adapter"], "files": dict(expected[rev_dir]["files"]), "sha256": dict(expected[rev_dir].get("sha256") or {}),
                                "marker": expected[rev_dir].get("marker"), "kind": expected[rev_dir].get("kind") or "revision"} for rev_dir in deliverable}})
    if deliverable:
        list_path = os.path.join(inbox, f".dirs-{window}-{os.getpid()}.txt")
        with open(list_path, "w", encoding="utf-8") as fh:
            fh.write("".join(d + "\n" for d in deliverable))
        try:
            r1 = _rsync(["rsync", "-rt", "--timeout=600", *ssh_e, f"--exclude=*{MARKER_TOKEN}*", f"--files-from={list_path}", src_adapters, recv_inbox + "/"],
                        runner, timeout_s)
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass
    else:
        r1 = subprocess.CompletedProcess([], 0, stdout="", stderr="")          # nothing of this window to pull
    report["rsync"]["pass1"] = r1.returncode

    def quarantine(rev_dir: str, why: str, where: str = FAILED_DIR, src_root: str = dest_adapters) -> None:
        src = os.path.join(src_root, rev_dir)
        dst = os.path.join(inbox, where, window, *(("adapters",) if where == STALE_DIR else ()), rev_dir)
        if os.path.isdir(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                dst = f"{dst}.{int(time.time())}"
            shutil.move(src, dst)
        report["quarantined"][rev_dir] = {"why": why, "moved_to": dst if os.path.isdir(dst) else None}
        a = rev_dir.split("/", 1)[0]
        report["adapters_refused"].setdefault(a, []).append(f"revision {rev_dir}: {why}")

    verify = verify_received      # the shard summary's promise against what was received: names, sizes AND sha256

    # the shared inbox, by ownership: what is ALREADY there, and whose it is
    present = _revision_dirs(dest_adapters)
    partial_in_inbox: Dict[str, str] = {}
    for rev_dir in present:
        exp = expected.get(rev_dir)
        d = os.path.join(dest_adapters, rev_dir)
        have = _listing(d)
        if exp is not None and exp.get("kind") == "refused":
            # a refusal record directory is SHARED by every window that refuses the id -- one `<stamp>-source_revision` file per
            # window, every one marker-shaped -- so an earlier window's file already in the inbox is no finished delivery of THIS window's:
            # this window's file is pulled and installed beside it (:func:`install_refusal_record`); inbox-merge takes them file by file
            continue
        if any(MARKER_TOKEN in fn for fn in have):
            # an earlier COMPLETED delivery: this window's pass 2 has not run, so the marker is an earlier window's (or an
            # earlier attempt's) whose inbox-merge never ran. It is inbox-merge's whatever this window holds about the id -- stale, refused,
            # unadmitted or barred, the verdict is about THIS window's read, and nothing of this window was pulled into it: never compared
            # against this window's batch list, never quarantined. Listed: reported complete as it stands (and released from `withheld`:
            # there is no delivery of this window to withhold); unlisted: reported `unlisted`.
            if exp is not None:
                report["withheld"].pop(rev_dir, None)
                report["pulled"][rev_dir] = {"state": "complete (an earlier delivery already in the inbox)", "adapter": exp["adapter"], "files": len(have)}
                report["depends"][rev_dir] = _depends_entry(exp["adapter"], "reused", _inbox_shas(d))
            elif rev_dir.split("/", 1)[0] in adapters_cluster:
                report["unlisted"][rev_dir] = UNLISTED_REASON
            continue
        if exp is None:
            # not this window's (ownership): an earlier window's partial delivery interrupted before its pass 2, or a directory of
            # a shard whose summary never arrived -- never pulled by this pass 1, never given a marker here, never quarantined here;
            # inbox-merge leaves a marker-less one too (`harvest.scan_inbox`'s rule) for a window whose summaries list the id to complete
            if rev_dir.split("/", 1)[0] in adapters_cluster:
                report["unlisted"][rev_dir] = UNLISTED_REASON
            continue
        if rev_dir.split("/", 1)[0] in stale_adapters:
            quarantine(rev_dir, "post-run revalidation refused the adapter; a directory of it in the inbox is never merged", where=STALE_DIR)
            continue
        if rev_dir in never:
            quarantine(rev_dir, f"{never[rev_dir]}; never delivered")
            continue
        # a listed, marker-less directory of a DELIVERABLE revision: an earlier window's partial of this very id. Never a staging
        # destination; set aside under _partial/<window>/ when this window's verified copy installs, left where it is otherwise
        partial_in_inbox[rev_dir] = d
        report["partial"][rev_dir] = ("a marker-less directory of this revision is already in the inbox (an earlier window's interrupted delivery): "
                                      f"set aside under {PARTIAL_DIR}/{window}/ when this window's verified copy installs, left otherwise")
    # what pass 1 received, revision by revision, verified in the receive directory
    eligible: List[str] = []
    for rev_dir in deliverable:
        exp = expected[rev_dir]
        d = os.path.join(recv_inbox, rev_dir)
        if r1.returncode != 0:
            quarantine(rev_dir, f"pass 1 rsync returned {r1.returncode}; the batch files cannot be trusted whole", src_root=recv_inbox)
            continue
        if not os.path.isdir(d):
            # published on the cluster, absent after pass 1: nothing arrived (gone from the spool? the caller sees the revision on its own
            # disk) -- named, never invented
            report["pulled"].setdefault(rev_dir, {"state": "absent after pass 1", "adapter": exp["adapter"]})
            continue
        why = verify(exp, d, with_marker=False)
        if why is not None:
            quarantine(rev_dir, why, src_root=recv_inbox)
            continue
        if not exp.get("marker"):
            quarantine(rev_dir, "the shard summary names no marker for this revision", src_root=recv_inbox)
            continue
        eligible.append(rev_dir)
    # 2. pass 2: the eligible markers alone, into the receive directory; then the whole-revision check and the install by one rename
    if eligible:
        markers = [f"{rev_dir}/{expected[rev_dir]['marker']}" for rev_dir in eligible]
        list_path = os.path.join(inbox, f".markers-{window}-{os.getpid()}.txt")
        with open(list_path, "w", encoding="utf-8") as fh:
            fh.write("".join(m + "\n" for m in markers))
        try:
            r2 = _rsync(["rsync", "-rt", "--timeout=600", *ssh_e, f"--files-from={list_path}", src_adapters, recv_inbox + "/"], runner, timeout_s)
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass
        report["rsync"]["pass2"] = r2.returncode
        for rev_dir in eligible:
            exp = expected[rev_dir]
            d = os.path.join(recv_inbox, rev_dir)
            if r2.returncode != 0:
                quarantine(rev_dir, f"the marker did not arrive whole (pass 2 rc {r2.returncode})", src_root=recv_inbox)
                continue
            why = verify(exp, d, with_marker=True)
            if why is not None:
                quarantine(rev_dir, f"the marker did not arrive whole: {why}", src_root=recv_inbox)
                continue
            aside = install_whole(inbox, window, d, rev_dir)
            if aside is not None:
                report["partial_aside"][rev_dir] = aside
            report["pulled"][rev_dir] = {"state": "complete", "adapter": exp["adapter"], "files": len(exp["files"])}
            report["depends"][rev_dir] = _depends_entry(exp["adapter"], "copied", dict(exp.get("sha256") or {}))   # the shard's digests, verified on receipt
    shutil.rmtree(recv_inbox, ignore_errors=True)       # whatever did not install was quarantined by name above; the rest is empty directories
    for a in adapters_cluster:
        if a in report["adapters_refused"]:
            report["adapters_refused"][a] = sorted(set(report["adapters_refused"][a]))
    return report


def verify_received(exp: Mapping[str, Any], d: str, with_marker: bool) -> Optional[str]:
    """None when ``d`` holds exactly the files the shard summary lists for the revision (its batches; the marker too when ``with_marker``),
    each with the listed size AND sha256; else the reason. A summary that carries no digest for a listed file admits nothing: name and
    size alone are not the test any more. ``exp`` is an ``expected_from_summaries`` entry or a RECEIVING manifest's promise
    (``files``, ``sha256``)."""
    what = "file" if with_marker else "batch"
    want = {k: int(v) for k, v in (exp.get("files") or {}).items() if with_marker or MARKER_TOKEN not in k}
    have = _listing(d) if os.path.isdir(d) else {}
    if have != want:
        missing = sorted(set(want) - set(have))
        extra = sorted(set(have) - set(want))
        sized = sorted(k for k in set(have) & set(want) if have[k] != want[k])
        return f"{what} list mismatch against the shard summary: missing {missing[:5]}, extra {extra[:5]}, size differs {sized[:5]}"
    shas = exp.get("sha256") or {}
    undigested = sorted(k for k in want if k not in shas)
    if undigested:
        return (f"the shard summary carries no sha256 for {undigested[:5]} (a {SUMMARY_SCHEMA} summary lists every published file's digest); "
                "nothing is admitted on name and size alone")
    bad = sorted(k for k in want if R.file_sha256(os.path.join(d, k)) != shas[k])
    if bad:
        return f"sha256 mismatch against the shard summary: {bad[:5]} (the bytes received are not the bytes the shard published)"
    return None


def install_whole(inbox: str, window: str, src: str, rev_dir: str) -> Optional[str]:
    """Install a verified, complete revision directory ``src`` (in a receive directory) at ``<inbox>/adapters/<rev_dir>`` by ONE rename.
    A marker-less directory already at the destination -- an earlier window's partial -- is set aside first under
    ``_partial/<window>/<rev_dir>`` (never deleted, never merged); returns that path, or None when nothing was aside. A destination that
    already holds its marker is a complete delivery: nothing is installed over it (``ClusterReadsError``). If the rename fails after the
    set-aside, the partial goes back where it was and the error is raised -- the two moves are the one gap a crash can open, and
    :func:`recover_receive_dirs` closes it from the manifest (the complete copy still in the receive directory, the partial under
    ``_partial/``)."""
    dst = os.path.join(inbox, "adapters", rev_dir)
    if f"/{REFUSED_DIR}/" in rev_dir:
        return install_refusal_record(inbox, src, rev_dir)
    aside: Optional[str] = None
    if os.path.lexists(dst):
        if os.path.isdir(dst) and any(MARKER_TOKEN in fn for fn in _listing(dst)):
            raise ClusterReadsError(f"{dst} already holds a complete revision (its marker is present); the received copy is not installed over it")
        aside = os.path.join(inbox, PARTIAL_DIR, window, rev_dir)
        os.makedirs(os.path.dirname(aside), exist_ok=True)
        if os.path.exists(aside):
            aside = f"{aside}.{int(time.time())}"
        shutil.move(dst, aside)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError:
        if aside is not None and not os.path.lexists(dst):
            shutil.move(aside, dst)              # the partial back where it was; the copy stays in the receive directory for recovery
        raise
    return aside


def install_refusal_record(inbox: str, src: str, rev_dir: str) -> Optional[str]:
    """Install a verified refusal record directory ``src`` (``<a>/_refused/<rev16>``) at ``<inbox>/adapters/<rev_dir>`` FILE BY FILE:
    the directory is shared by every window that refuses the id, each window's record its own ``<stamp>-source_revision`` file, so an
    earlier window's file already there (not yet merged) is left in place and this window's lands beside it. A file already present with
    the same bytes is a duplicate (the received copy dropped); the same name with other bytes is a conflict and refuses by name
    (``ClusterReadsError``: nothing of ``src`` moved). Returns None (nothing is ever set aside)."""
    dst = os.path.join(inbox, "adapters", rev_dir)
    if not os.path.lexists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
        return None
    if not os.path.isdir(dst):
        raise ClusterReadsError(f"{dst} exists and is not a directory; the refusal record is not installed over it")
    for name in sorted(os.listdir(src)):
        there = os.path.join(dst, name)
        if os.path.exists(there) and R.file_sha256(there) != R.file_sha256(os.path.join(src, name)):
            raise ClusterReadsError(f"{there} already holds other bytes under this name; the received refusal record is not installed over it")
    for name in sorted(os.listdir(src)):
        there = os.path.join(dst, name)
        if os.path.exists(there):
            os.remove(os.path.join(src, name))          # the same bytes: a duplicate of a file already waiting for inbox-merge
        else:
            os.replace(os.path.join(src, name), there)
    try:
        os.rmdir(src)
    except OSError:
        pass
    return None


def recover_receive_dirs(root: str, window: str, inbox_name: str = INBOX_NAME, log: Callable[[str], None] = lambda s: print(s, flush=True)) -> Dict[str, Any]:
    """Crash recovery for the pull's receive directories: run by the window's ``-- cluster reads inbox-merge`` stage BEFORE
    the standing merge, and by ``cluster-reads recover``. A pull that died between pass 2 and the install leaves a complete, verified
    revision under ``inbox/<name>/.recv-<window>-…/`` -- a dot-prefixed directory ``scan_inbox`` skips, the reader-state export cannot
    see, and no later pull opens (each allocates its own): without this step a finished read never reached the database. For every
    receive directory present: with its RECEIVING manifest, each revision directory inside is re-verified against the manifest's promise
    (names, sizes, sha256, marker -- the shard summary's words, persisted), its adapter must have passed THAT window's post-run
    revalidation and attestation premise (the manifest's verdicts; the attestation record re-checked when it is still on disk), and a
    complete, admitted copy is installed by the same rename as the pull (:func:`install_whole`, set-aside rule included). Anything else --
    no manifest, an unlisted directory, a mismatch, a stale or unadmitted adapter, a destination already complete -- is moved under
    ``_failed/<window>/recv-<receive dir>/`` with a loud ``UNRECOVERABLE`` line naming it, never deleted (the manifest copied beside it);
    an emptied receive directory is removed. Returns ``{receive dir: {manifest, installed, partial_aside, failed, removed}}``."""
    inbox = os.path.join(H.inbox_dir(root), inbox_name)
    out: Dict[str, Any] = {}
    if not os.path.isdir(inbox):
        return out
    for name in sorted(os.listdir(inbox)):
        recv = os.path.join(inbox, name)
        if not (name.startswith(RECV_PREFIX) and os.path.isdir(recv)):
            continue
        rep: Dict[str, Any] = {"manifest": None, "installed": [], "partial_aside": {}, "failed": {}, "removed": False}
        out[name] = rep
        failed_root = os.path.join(inbox, FAILED_DIR, window, "recv-" + name[len(RECV_PREFIX):])

        def fail(rel: str, why: str, _rep: Dict[str, Any] = rep, _recv: str = recv, _failed_root: str = failed_root, _name: str = name) -> None:
            src, dst = os.path.join(_recv, rel), os.path.join(_failed_root, rel)
            if os.path.lexists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    dst = f"{dst}.{int(time.time())}"
                shutil.move(src, dst)
            _rep["failed"][rel] = why
            log(f"[recover] UNRECOVERABLE {_name}/{rel}: {why} -> {dst}")

        try:
            manifest = _load_json(os.path.join(recv, RECEIVING_FILE))
        except (OSError, ValueError, ClusterReadsError):
            manifest = None
        if not isinstance(manifest, dict) or manifest.get("schema") != RECEIVING_SCHEMA or not isinstance(manifest.get("revisions"), dict):
            rep["manifest"] = "missing"
            for rel in _revision_dirs(recv):
                fail(rel, f"no readable {RECEIVING_FILE} in the receive directory: the window, its summary and its verdicts are unknown; not installed")
        else:
            rep["manifest"] = {"window": manifest.get("window"), "utc": manifest.get("utc")}
            stale = {str(a) for a in (manifest.get("revalidate_post") or {}).get("stale") or []}
            void = {str(a): str(w) for a, w in (manifest.get("attestation_void") or {}).items()}
            att_path = str(manifest.get("attestation") or "")
            att = ST.read_attestation(att_path) if att_path and os.path.isfile(att_path) else None
            for rel in _revision_dirs(recv):
                promise = manifest["revisions"].get(rel)
                if not isinstance(promise, dict):
                    fail(rel, "the receive directory holds a revision the manifest does not list; not installed")
                    continue
                a = str(promise.get("adapter") or rel.split("/", 1)[0])
                if a in stale:
                    fail(rel, f"adapter {a} was refused by window {manifest.get('window')}'s post-run revalidation; not installed")
                    continue
                if a in void:
                    fail(rel, f"window {manifest.get('window')}'s attestation did not admit adapter {a}: {void[a]}; not installed")
                    continue
                if att is not None:
                    why_att = attestation_premise(att, att_path, str(manifest.get("window")), dict((manifest.get("pins") or {}).get(a) or {}))
                    if why_att is not None:
                        fail(rel, f"the attestation record re-checked now does not admit adapter {a}: {why_att}; not installed")
                        continue
                why = verify_received(promise, os.path.join(recv, rel), with_marker=True)
                if why is not None:
                    fail(rel, f"{why}; not installed")
                    continue
                try:
                    aside = install_whole(inbox, window, os.path.join(recv, rel), rel)
                except ClusterReadsError as exc:
                    fail(rel, f"{exc}; not installed")
                    continue
                rep["installed"].append(rel)
                if aside is not None:
                    rep["partial_aside"][rel] = aside
                log(f"[recover] installed {name}/{rel} (window {manifest.get('window')}, received {manifest.get('utc')})"
                    + (f"; the earlier partial aside at {aside}" if aside else ""))
        if rep["failed"] and os.path.isfile(os.path.join(recv, RECEIVING_FILE)):
            os.makedirs(failed_root, exist_ok=True)
            shutil.copy2(os.path.join(recv, RECEIVING_FILE), os.path.join(failed_root, RECEIVING_FILE))
        if not _revision_dirs(recv):
            shutil.rmtree(recv, ignore_errors=True)
            rep["removed"] = not os.path.exists(recv)
    return out


def reactivate_refused_known(root: str, plan: Mapping[str, Any], report: Dict[str, Any], conn=None, now: Optional[str] = None,
                             adapter_of: Optional[Callable[[str], Any]] = None) -> Dict[str, Dict[str, Any]]:
    """The grading host's half of a refused-known group: for every revision id the shards refused as known
    (``report["refused_known_ids"]``) whose ``source_revision`` row is NOT its source's current one, ``runner.reactivate_by_id`` rebuilds the
    re-activation bundle from the COMMITTED bytes -- no store read; the marker's ``inputs_json`` and the committed watermark facts stand in
    for a discovery. A current id is recorded ``current`` (nothing to do), a pending / held / unfolded one ``held`` with the runner's reason,
    an id the grading host REFUSED AT EMIT TIME and never committed ``refused_standing`` (its row is the fold of the ``_refused/<rev>/``
    record and no directory or bundle exists -- nothing to re-activate from; the refusal stands as it was, and a shard on this tree reads
    such an id fresh instead of returning it, so one lands here only from an older shard's claim), a RETIREMENT whose shard claim
    (``report["refused_known_claims"]``, ``runner.retirement_claim``: the head the shard compared against, the vanished inputs it saw
    declared retired) is not exactly the live state -- or that came with no claim -- ``withheld`` by name (another writer accepted a newer
    revision of the source between the sidecar's export and the pull; nothing written, the next window's shard judges the group against
    today's head; the receipt lists the id under ``withheld`` and the adapter is not read whole), a marker without the facts ``refused`` by
    name -- and only a refusal IS a recorded refusal of the adapter (``report["adapters_refused"]``: the revision the cluster refused and
    the grading host could not re-activate stays as it was); ``refused_standing`` and ``withheld`` never are. ``conn`` is a read-only
    connection to the registry's database (opened here when None; the bundle is written to the record log only). Returns and records
    (``report["reactivations"]``) ``{adapter: {revision_id: {state, error, files}}}``; ``adapter_of`` (tests) constructs the adapter
    instead of the registry's factory."""
    from .adapters import load_all
    from .adapters.runner import ReactivationBudget, reactivate_by_id
    out: Dict[str, Dict[str, Any]] = {}
    ids = {a: list(revs) for a, revs in (report.get("refused_known_ids") or {}).items() if revs}
    if not ids:
        report["reactivations"] = out
        return out
    own = conn is None
    if own:
        db_path = DB.db_path(root)
        conn = DB.connect(db_path, readonly=True) if os.path.exists(db_path) else None
    try:
        reg = None
        for a, revs in sorted(ids.items()):
            budget = ReactivationBudget()
            per: Dict[str, Any] = {}
            try:
                if adapter_of is not None:
                    adapter = adapter_of(a)
                else:
                    reg = reg if reg is not None else load_all()
                    if a not in reg:
                        raise ClusterReadsError(f"adapter {a!r} is not registered in this tree")
                    adapter = reg[a]()        # constructed only: reactivate_by_id never discovers or reads
            except (ClusterReadsError, Exception) as exc:    # noqa: BLE001 -- a construction failure is this adapter's refusal, by name
                for rev in sorted(revs):
                    per[rev] = {"state": "refused", "error": f"adapter could not be constructed for the re-activation: {exc}", "files": []}
                    report["adapters_refused"].setdefault(a, []).append(f"refused-known revision {rev}: not re-activated -- {exc}")
                out[a] = per
                continue
            claims = (report.get("refused_known_claims") or {}).get(a) or {}
            for rev in sorted(revs):
                res = reactivate_by_id(root, adapter, rev, conn, now=now, budget=budget, retirement_claim=claims.get(rev))
                if res.accepted and not res.files:
                    state = "current"
                elif res.accepted:
                    state = "reactivated"
                elif res.deferred:
                    state = "deferred"
                elif res.held:
                    state = "held"
                elif res.withheld:
                    state = "withheld"               # a retirement whose shard claim is not the live state: nothing written, not the adapter's refusal
                elif res.refused_standing:
                    state = "refused_standing"        # refused at emit time, never committed: nothing to re-activate, not the adapter's refusal
                else:
                    state = "refused"
                    report["adapters_refused"].setdefault(a, []).append(f"refused-known revision {rev}: not re-activated from committed bytes -- {res.error}")
                per[rev] = {"state": state, "error": res.error, "files": len(res.files), "records": res.records_written, "withdrawals": res.withdrawals}
            out[a] = per
            if a in report["adapters_refused"]:
                report["adapters_refused"][a] = sorted(set(report["adapters_refused"][a]))
    finally:
        if own and conn is not None:
            conn.close()
    report["reactivations"] = out
    return out


def attestation_premise(att: Optional[Mapping[str, Any]], path: str, window: str, pins: Mapping[str, str]) -> Optional[str]:
    """The grading host's re-check of the shard's admission over the record the pull brought back (FAIL-CLOSED): the reason the
    operation's attestation record does NOT admit an adapter, or None when it admits every pin the adapter reads. Refused by name: no
    readable record at ``path`` (missing, or not a ``store_attestation/1`` record), another operation's record, and -- through
    ``stores.attestation_admits``, the very check the shard ran before and after its read -- a pin the record names no generation for
    (``attestation_missing``), attests at another generation (``attestation_pin``) or refused (``attestation_refused``, with the record's
    own refusal and reason). A shard summary's ``ok`` is the shard's claim; the record is what the pull verifies for itself, and without
    it nothing of the adapter is delivered or replayed."""
    if att is None:
        state = f"unreadable (not a {ST.ATTESTATION_SCHEMA} record)" if os.path.isfile(path) else "missing"
        return (f"no readable attestation record for this operation ({path} is {state}): the attestation job wrote none, or the record did "
                f"not arrive with the shard summaries")
    if str(att.get("window")) != str(window):
        return f"the attestation record pulled is operation {att.get('window')}'s, not this one's"
    admission = ST.attestation_admits(att, window, pins, path=path)
    bad = {s: r for s, r in sorted(admission.items()) if r.get("ok") is not True}
    if not bad:
        return None
    parts: List[str] = []
    own = True              # every refusal is the record's own verdict on the pinned generation (vs a pin it does not name / attests elsewhere)
    for slug, r in bad.items():
        if r.get("refusal") == "attestation_refused":
            g = (att.get("generations") or {}).get(slug) or {}
            parts.append(f"{slug}: {g.get('refusal')} ({g.get('reason')})")
        else:
            own = False
            parts.append(f"{slug}: {r.get('refusal')} ({r.get('reason')})")
    head = "the attestation refused component(s) this adapter reads: " if own else "the attestation does not admit component(s) this adapter reads: "
    return head + "; ".join(parts)


def attestation_summary(att: Optional[Mapping[str, Any]], window: str) -> Dict[str, Any]:
    """What the pull report and the plan's ``pull`` stage carry of the operation's attestation record (C5): present?, this window's?, ok,
    the stamp, the host, and per component the verdict (``ok`` / ``refusal`` / ``reason`` / the content mode / the listing's entries and
    seconds) -- the record itself stays in the expect directory."""
    if not att:
        return {"present": False, "window": None, "ok": None, "utc": None, "host": None, "refused": [], "generations": {}}
    gens = {}
    for slug, g in sorted((att.get("generations") or {}).items()):
        if not isinstance(g, Mapping):
            continue
        lst, content = g.get("listing") or {}, g.get("content") or {}
        gens[slug] = {"content_sha8": g.get("content_sha8"), "ok": g.get("ok") is True, "refusal": g.get("refusal"), "reason": g.get("reason"),
                      "content_mode": content.get("mode"), "listing_entries": lst.get("entries"), "listing_s": lst.get("elapsed_s"), "elapsed_s": g.get("elapsed_s")}
    return {"present": True, "window": att.get("window"), "this_window": str(att.get("window")) == str(window), "ok": att.get("ok") is True, "utc": att.get("utc"),
            "host": att.get("host"), "job_id": att.get("job_id"), "elapsed_s": att.get("elapsed_s"), "refused": sorted(att.get("refused") or []), "generations": gens}


#: The pre-pull ``quarantine_stale`` is GONE: it moved the stale adapter's WHOLE inbox subtree under ``_stale/<window>/``
#: before the pull -- and before the pull nothing of THIS window is in the inbox, so the only directories it could ever move were an
#: earlier window's deliveries interrupted before their inbox-merge (stranded: ``scan_inbox`` skips ``_stale/``). ``cluster-reads
#: revalidate`` records the stale adapters; :func:`pull_window` withholds their LISTED revisions (never pulled) and quarantines a listed
#: one already in the inbox under ``_stale/<window>/adapters/`` -- ownership by listing, never by adapter subtree.


def cluster_routes_without_tasks(plan: Mapping[str, Any]) -> List[str]:
    """The adapters a plan routes to the cluster with NO task: ``shards`` < 1, or an EMPTY ``task_indices`` where the entry carries the
    key, or a ``tasks`` list the plan carries that names none of theirs -- nobody would read them, so every consumer treats them as the
    grading host's (:data:`NO_TASK_REASON`). :func:`plan_window` never writes such an entry (a ``cluster: refused`` or zero-shard declaration
    routes to the grading host first); a plan FILE is data, so the submission, the refusal set and the pull check it rather than trust it. A
    plan that carries neither key (a minimal one: ``{route: cluster, shards: K}``) is judged by its shards alone."""
    tasks = plan.get("tasks")
    tasks_of: Dict[str, int] = {}
    for t in tasks or []:
        tasks_of[str(t.get("adapter"))] = tasks_of.get(str(t.get("adapter")), 0) + 1
    out: List[str] = []
    for a, e in (plan.get("adapters") or {}).items():
        if e.get("route") != "cluster":
            continue
        if int(e.get("shards") or 0) < 1 or ("task_indices" in e and not e["task_indices"]) or (tasks is not None and not tasks_of.get(a)):
            out.append(a)
    return sorted(out)


def plan_refusals(plan: Mapping[str, Any]) -> Dict[str, List[str]]:
    """The plan's recorded refusals (``refusals``; a pre-C6 plan file spelled the key ``fallback`` -- read for the status page)."""
    rec = plan.get("refusals")
    if rec is None:
        rec = plan.get("fallback")
    return {str(a): list(r) for a, r in (rec or {}).items()}


def refused_adapters(plan: Dict[str, Any], pull_report: Optional[Mapping[str, Any]] = None) -> Dict[str, List[str]]:
    """Every adapter NOT read this window, with the reasons (C6: nothing falls back to the grading host -- a refusal is recorded, stamped by the
    window as ``-- adapter A REFUSED`` and the window ends ABORTED naming them): routed ``grading_host`` at plan time (an entry declared ``cluster:
    refused`` among them, with the declaration's reason), a cluster route with no task, a missing / failed / refusing shard, a failed pull
    barrier, a post-run revalidation refusal, a refused-known revision the grading host could not re-activate from committed bytes."""
    out: Dict[str, List[str]] = plan_refusals(plan)
    for a in cluster_routes_without_tasks(plan):
        out.setdefault(a, []).append(f"plan: {NO_TASK_REASON}")
    for a, reasons in ((pull_report or {}).get("adapters_refused") or {}).items():
        out.setdefault(a, [])
        out[a] += [f"pull: {r}" for r in reasons]
    return {a: sorted(set(r)) for a, r in out.items()}


def record_stage(plan: Dict[str, Any], name: str, **fields: Any) -> None:
    plan.setdefault("stages", {})[name] = {"utc": R.utc_now(), **fields}


# --------------------------------------------------------------------------------------------
# the status page's section
# --------------------------------------------------------------------------------------------


def status_section(root: str) -> Dict[str, Any]:
    """"Cluster reads" from the latest plan file: per adapter the route, shards, pins, the reasons it was recorded refused; the stages."""
    plan = latest_plan(root)
    if plan is None:
        return dict(EMPTY_SECTION)
    rows = []
    fb = plan_refusals(plan)
    for a, e in sorted((plan.get("adapters") or {}).items()):
        rows.append({"adapter": a, "route": e.get("route"), "shards": e.get("shards"), "refusals": "; ".join(fb.get(a, [])) or "", "reason": e.get("reason") or "",
                     "refused": bool(e.get("refused")), "pins": ", ".join(f"{s}@{g}" for s, g in sorted((e.get("pins") or {}).items()))})
    return {"present": True, "window": plan.get("window"), "built_utc": plan.get("built_utc"), "tree_sha": plan.get("tree_sha"), "rows": rows,
            "stages": plan.get("stages") or {}, "job_id": (plan.get("stages") or {}).get("submit", {}).get("job_id")}


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("plan", help="the grading host: pin the published generations, re-validate every component (admission pass 1), route each adapter "
                                    "(cluster, or recorded refused with the reason; no discovery runs here -- the grading host reads no store), "
                                    "write state/cluster_reads/<window>.json")
    s.add_argument("--window", required=True, help="the window's name (an ISO stamp)")
    s.add_argument("--adapter", nargs="*", help=f"the adapters to plan (default: {', '.join(DEFAULT_ADAPTERS)})")
    s.add_argument("--tree-sha", required=True, help="the exported tree's commit (40 hex)")
    s.add_argument("--dest", help="destination name from cluster_reads.yaml (the only one when a single destination is declared)")
    s.add_argument("--config", help="cluster_reads.yaml (default the checkout's)")
    s.add_argument("--repo-root", help="the grading host checkout the cluster's tree stands for (default this checkout)")
    s.add_argument("--full", action="store_true", help="the shards read every group and hash every byte (the cluster --full)")
    s.add_argument("--full-adapter", nargs="*", default=[], metavar="A", help="the window's $FULL_ADAPTERS: these adapters read --full on the cluster")
    s.add_argument("--reread-force-adapter", nargs="*", default=[], metavar="A",
                   help="the window's $REREAD_FORCE_ADAPTERS: a store read on the grading host (an operator override), recorded refused here")
    s.add_argument("--reader-state", metavar="FILE",
                   help="the window's reader-state sidecar, as `reader-state export` wrote it for these adapters: the harvester's standing "
                        "retirement holds the plan records (holds_seen) are read off it -- the snapshot the shards judge against, never the "
                        "live database; without it the plan records no hold seen and the reads protocol clears none on its receipt")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("stage", help="the grading host: lay the window root out on the cluster (spool, the rulings symlink onto the record-log mirror, verified) "
                                     "and rsync the sidecar and the plan into it")
    s.add_argument("--window", required=True)
    s.add_argument("--reader-state", required=True, metavar="FILE")
    s = sub.add_parser("submit", help="the grading host: sbatch the operation's attestation job (attest.sbatch: every pinned generation listed against its "
                                      "manifest, its seal and its content checked once for this operation) and then the read array (one task per "
                                      "plan task) behind it with --dependency=afterok; records both job ids in the plan and prints the array's; "
                                      "prints `none` (rc 0) for a plan with no cluster task -- a failed submission is a non-zero exit, never `none`")
    s.add_argument("--window", required=True)
    s.add_argument("--venv", required=True, help="the gate venv on the cluster (imported in place by every task)")
    s.add_argument("--job-name", help="default reads-<window>")
    s.add_argument("--local-venv", action="store_true", help="every task first copies the venv node-local (read_array.sbatch LOCAL_VENV=1): opt-in, "
                   "MEASURED 2026-09-28 at > 7.5 min for two concurrent copies of the torch-cpu gate venv on one node (~563 s per task)")
    s.add_argument("--op", help="the operation id the reservation must name (C7: refused by name, exit 23, when another operation holds it)")
    s.add_argument("--state", help="the landing state directory the reservation lives under (default $REGISTRY_LANDING_STATE)")
    s = sub.add_parser("shard", help="CLUSTER: one array task -- admission, the home tree, the sharded pass against the sidecar, the summary")
    s.add_argument("--plan", required=True, metavar="FILE")
    s.add_argument("--index", type=int, required=True)
    s.add_argument("--work", help="the task's node-local scratch (read_array.sbatch passes it; the pass itself does not use it)")
    s.add_argument("--summaries-dir", help="override the plan's summaries directory (tests)")
    s.add_argument("--home", help="override the node's home (tests)")
    s = sub.add_parser("revalidate", help="the grading host: admission pass 2 after the array; a difference names the adapter for quarantine + its recorded refusal")
    s.add_argument("--window", required=True)
    s.add_argument("--config")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("pull", help="the grading host: pull the published revisions marker-last behind the failure barrier, quarantine the stale ones, and "
                                    "re-activate every refused-known revision from committed bytes (no store read)")
    s.add_argument("--window", required=True)
    s.add_argument("--source", help="override <host>:<window root> (a local directory in tests)")
    s.add_argument("--expect", help="where the shard summaries are pulled (default state/cluster_reads/<window>.summaries/)")
    s.add_argument("--op", help="the operation id the reservation must name before anything is pulled or merged (C7; exit 23 otherwise)")
    s.add_argument("--state", help="the landing state directory the reservation lives under (default $REGISTRY_LANDING_STATE)")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("refusals", help="the grading host: print the adapters recorded REFUSED this window -- not read anywhere (one per line; --json for the reasons)")
    s.add_argument("--window", required=True)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("stamp", help="the grading host: record a stage in the plan (name=value pairs)")
    s.add_argument("--window", required=True)
    s.add_argument("--stage", required=True)
    s.add_argument("fields", nargs="*", metavar="KEY=VALUE")
    s = sub.add_parser("status", help="the latest plan, as the status page shows it")
    s.add_argument("--json", action="store_true")
    from . import reads_protocol as RP
    RP.add_arguments(sub)      # C7: reserve / release / receipt / tick / changed -- the ownership protocol of every read operation
    s = sub.add_parser("recover", help="the grading host: finish the deliveries an interrupted pull left in the inbox's receive directories (.recv-*): each "
                                       "revision re-verified against its RECEIVING manifest and installed, or moved under _failed/<window>/recv-*/ "
                                       "with a loud line (never deleted); the window runs this before its standing inbox-merge")
    s.add_argument("--window", help="the window doing the recovery (names the _failed/ and _partial/ destinations; default recover-<utc>)")
    s.add_argument("--json", action="store_true")


def _plan_file(root: str, window: str) -> Tuple[str, Dict[str, Any]]:
    path = plan_path(root, window)
    if not os.path.isfile(path):
        raise ClusterReadsError(f"no plan for window {window!r} at {path} (run `cluster-reads plan` first)")
    return path, _load_json(path)


def _cli_pull(root: str, plan: Dict[str, Any], path: str, args) -> int:
    """The ``pull`` stage's body: the transfer, the refused-known re-activations, the recorded stage, the refusal set, the line."""
    expect = args.expect or (path[:-len(".json")] + ".summaries")
    rep = pull_window(root, plan, expect, source=args.source)
    # C6: the refused-known revisions the shards returned are the grading host's to re-activate from committed bytes -- here, before the merge
    reactivations = reactivate_refused_known(root, plan, rep)
    n_react = sum(1 for per in reactivations.values() for v in per.values() if v.get("state") == "reactivated")
    withheld = rep.get("refused_known_withheld") or {}
    n_withheld = sum(len(v) for v in withheld.values())
    n_retirements_withheld = sum(1 for per in reactivations.values() for v in per.values() if v.get("state") == "withheld")
    record_stage(plan, "pull", pulled=len(rep["pulled"]), pulled_dirs=sorted(d for d, v in rep["pulled"].items() if v.get("state") == "complete"),
                 depends_dirs=dict(rep.get("depends") or {}),
                 quarantined=len(rep["quarantined"]), withheld=len(rep["withheld"]), unlisted=sorted(rep.get("unlisted") or {}),
                 partial=sorted(rep.get("partial") or {}), partial_aside=dict(rep.get("partial_aside") or {}),
                 barrier=rep.get("barrier"), rsync=rep["rsync"], summaries=rep["summaries"], attestation=rep.get("attestation"),
                 reactivations={a: {r: v.get("state") for r, v in per.items()} for a, per in reactivations.items()}, refused_known_withheld=withheld,
                 standing_refusals=dict(rep.get("standing_refusals") or {}))
    plan["refusals"] = refused_adapters(plan, rep)
    _write_json(path, plan)
    barrier = rep.get("barrier") or {}
    print(json.dumps(rep, indent=1, default=str) if args.json else
          f"pull: {sum(1 for v in rep['pulled'].values() if v.get('state') == 'complete')} revision(s) complete, {len(rep['quarantined'])} quarantined, "
          f"{n_react} refused-known revision(s) re-activated from committed bytes, {n_withheld} withheld; "
          + (f"{n_retirements_withheld} retirement re-activation(s) withheld by name (the shard's claim is not the live head); " if n_retirements_withheld else "")
          + (f"BARRIER at the {barrier.get('stage')} transfer (rsync rc {barrier.get('rc')}): nothing delivered or replayed; " if barrier else "")
          + (f"{len(rep['unlisted'])} inbox directory(ies) of an earlier window left for inbox-merge; " if rep.get("unlisted") else "")
          + (f"{len(rep['partial_aside'])} earlier partial(s) set aside under {PARTIAL_DIR}/; " if rep.get("partial_aside") else "")
          + ("refused: " + ", ".join(sorted(rep["adapters_refused"])) if rep["adapters_refused"] else "no refusal from the pull"))
    return 0


def cmd_cluster_reads(args, root: str) -> int:
    try:
        if args.action == "shard":
            return run_shard(args.plan, args.index, work=args.work, summaries_dir=args.summaries_dir, home=args.home)
        if args.action == "status":
            sec = status_section(root)
            print(json.dumps(sec, indent=1, default=str) if args.json else
                  ("cluster reads: no plan recorded" if not sec["present"] else
                   f"cluster reads: window {sec['window']} (tree {str(sec.get('tree_sha') or '')[:9]}) built {sec['built_utc']}\n" +
                   "\n".join(f"  {r['adapter']:<20} {r['route']:<8} K={r['shards']:<2} {('REFUSED: ' + r['refusals']) if r['refusals'] else ''}" for r in sec["rows"])))
            return 0
        if args.action in ("reserve", "release", "receipt", "tick", "changed"):
            from . import reads_protocol as RP
            return RP.cmd_reads(args, root)
        if args.action == "recover":
            # no plan is needed -- every receive directory carries its own RECEIVING manifest; a refusal is a result (rc 0, loud)
            window = str(getattr(args, "window", None) or f"recover-{R.utc_now().replace(':', '').replace('-', '')[:15]}Z")
            rec = recover_receive_dirs(root, window)
            n_inst = sum(len(v["installed"]) for v in rec.values())
            n_fail = sum(len(v["failed"]) for v in rec.values())
            print(json.dumps(rec, indent=1, default=str) if args.json else
                  (f"recover: {len(rec)} receive directory(ies): {n_inst} revision(s) installed, {n_fail} UNRECOVERABLE (under {FAILED_DIR}/{window}/recv-*/)"
                   if rec else "recover: no receive directory left by an interrupted pull"))
            return 0
        if args.action == "plan":
            cfg = ST.load_config(getattr(args, "config", None), root=root)      # a destination joins this root's [[clusters]] entry
            dest = cfg.destination(args.dest)
            adapters = list(args.adapter) if args.adapter else list(DEFAULT_ADAPTERS)
            from .adapters import load_all
            plan = plan_window(root, args.window, adapters, cfg, dest, args.tree_sha, repo_root=args.repo_root, full=bool(args.full),
                               full_adapters=list(args.full_adapter or []), reread_force_adapters=list(args.reread_force_adapter or []), registry=load_all(),
                               sidecar=getattr(args, "reader_state", None))
            path = plan_path(root, args.window)
            _write_json(path, plan)
            routes = {a: e["route"] for a, e in plan["adapters"].items()}
            print(json.dumps({"plan": path, "routes": routes, "tasks": len(plan["tasks"]), "refusals": plan["refusals"]}, indent=1) if args.json else
                  f"plan {path}: " + ", ".join(f"{a}={r}" for a, r in routes.items()) + f"; {len(plan['tasks'])} task(s)" +
                  ("".join(f"\n  {a} -> REFUSED: {'; '.join(why)}" for a, why in plan["refusals"].items()) if plan["refusals"] else ""))
            return 0
        path, plan = _plan_file(root, args.window)
        if args.action == "submit" and getattr(args, "op", None):
            # The reservation must still name this operation before a job is submitted; a window whose
            # reservation another operation took stops here by name (the pull's check is inside its critical section, below)
            from . import reads_protocol as RP
            try:
                RP.require_ownership(RP.reads_dir(getattr(args, "state", None)), str(args.op))
            except RP.OwnershipLost as exc:
                print(f"cluster-reads submit: ownership -- {exc}; nothing submitted", file=sys.stderr)
                return RP.EXIT_OWNERSHIP
        if args.action == "stage":
            info = stage_window(plan, args.reader_state, path)
            record_stage(plan, "stage", **info)
            _write_json(path, plan)
            print(json.dumps(info))
            return 0
        if args.action == "submit":
            if not plan.get("tasks"):
                idle = cluster_routes_without_tasks(plan)
                if idle:
                    # `none` would read as success to the window (nothing watched, the plan's refusals recorded) while these
                    # adapters sit routed to the cluster with nobody reading them: a failed submission instead, which refuses them all
                    raise ClusterReadsError(f"submit: the plan has no task but routes {', '.join(idle)} to the cluster; nothing would read "
                                            f"them -- re-plan (a cluster route carries at least one task)")
                record_stage(plan, "submit", job_id=None, tasks=0)     # every adapter routes to the grading host: nothing to submit, not a failure
                _write_json(path, plan)
                print("none")
                return 0
            # C5: the attestation job first, the array behind it (afterok); the array id is what the window watches -- its terminal state
            # follows the attestation's, and a shard reads the attestation's record before anything else
            attest = submit_attest(plan)
            job = submit_array(plan, args.venv, args.job_name or f"reads-{args.window}", local_venv=bool(getattr(args, "local_venv", False)), dependency=attest)
            record_stage(plan, "submit", job_id=job, attest_job_id=attest, tasks=len(plan["tasks"]), local_venv=bool(getattr(args, "local_venv", False)),
                         op=getattr(args, "op", None))
            if getattr(args, "op", None):
                from . import reads_protocol as RP
                RP.update_op(RP.reads_dir(getattr(args, "state", None)), str(args.op), state="submitted", jobs={"attest": attest, "reads": job}, submitted_utc=R.utc_now())
            _write_json(path, plan)
            print(job)
            return 0
        if args.action == "revalidate":
            cfg = ST.load_config(getattr(args, "config", None) or plan.get("config"), root=root)
            # recorded only: the pull withholds a stale adapter's LISTED revisions and quarantines a listed one already in the
            # inbox; nothing is moved here -- before the pull the inbox holds only earlier windows' deliveries, which are inbox-merge's
            stale = revalidate_plan(root, plan, cfg)
            _write_json(path, plan)
            print(json.dumps({"stale": stale, "refusals": plan.get("refusals") or {}}, indent=1) if args.json else
                  ("revalidation: every component still matches its pin" if not stale else "revalidation REFUSED: " + ", ".join(stale)
                   + " (recorded refused; the pull withholds every revision these adapters published this window)"))
            return 0
        if args.action == "pull":
            if not getattr(args, "op", None):
                return _cli_pull(root, plan, path, args)
            # under `--op` the ownership check and the pull are ONE critical section -- the same lock the tick's
            # pull / merge / receipt and every reservation hand-over hold -- so the reservation it reads cannot change hands underneath the
            # transfer, and a takeover that arrives now waits for the pull to end
            from . import reads_protocol as RP
            rd = RP.reads_dir(getattr(args, "state", None))
            try:
                with RP.critical_section(rd, str(args.op), "window", "pull", log=lambda s: print(s, file=sys.stderr)):
                    try:
                        RP.require_ownership(rd, str(args.op))
                    except RP.OwnershipLost as exc:
                        print(f"cluster-reads pull: ownership -- {exc}; nothing pulled or merged", file=sys.stderr)
                        return RP.EXIT_OWNERSHIP
                    rc = _cli_pull(root, plan, path, args)
                    if rc != 0:
                        return rc
                    # under `--op` the transfer AND the inbox-merge are two RECORDED stages of this operation, both
                    # inside this one section -- the plan's `pull` (recorded by _cli_pull) and `inbox_merge`, mirrored on the op record
                    # (`pulled` -> `merged`) -- so the window's `cluster-reads receipt` has the same precondition as the tick's receipt
                    # (invariant I7: no receipt without a recorded merge of this operation's pull); a window that dies between its pull and
                    # its receipt is resumed by the next reconcile at the first missing stage, never at the receipt
                    from . import harvest as H
                    RP.record_op_stage(rd, str(args.op), "pull", by=str(args.op), owner="window", tree_sha=plan.get("tree_sha"))
                    merged = H.scan_inbox(root)
                    counts = {k: sum(1 for _, st in merged if st == k) for k in ("merged", "duplicate", "conflict", "incomplete")}
                    # I7: the receipt's precondition is DURABLE state -- every revision this operation depends on (newly copied
                    # or an earlier delivery reused) read back from records/ against its recorded files, never the scan's return value
                    evidence = RP.merge_evidence(root, RP.depends_of_plan(plan))
                    problems = RP.apply_merge_problems(plan, evidence)
                    record_stage(plan, "inbox_merge", by=str(args.op), problems=problems, evidence={r: v["state"] for r, v in evidence.items()}, **counts)
                    _write_json(path, plan)
                    RP.record_op_stage(rd, str(args.op), "inbox_merge", by=str(args.op), stage_fields={"problems": problems}, owner="window")
                    print(f"inbox-merge: {counts['merged']} merged, {counts['duplicate']} duplicate, {counts['conflict']} conflict, "
                          f"{counts['incomplete']} incomplete -- recorded as this operation's inbox_merge stage (the receipt's precondition)"
                          + ("; REFUSED in the receipt: " + "; ".join(f"{a}: {', '.join(w)}" for a, w in sorted(problems.items())) if problems else ""))
                    return 0
            except RP.CriticalSectionBusy as exc:
                print(f"cluster-reads pull: {exc}; nothing pulled or merged", file=sys.stderr)
                return RP.EXIT_SECTION_BUSY
        if args.action == "refusals":
            fb = refused_adapters(plan)
            print(json.dumps(fb, indent=1) if args.json else "\n".join(sorted(fb)))
            return 0
        if args.action == "stamp":
            fields = dict(f.split("=", 1) for f in args.fields if "=" in f)
            record_stage(plan, args.stage, **fields)
            _write_json(path, plan)
            return 0
        return 2
    except (ClusterReadsError, ST.StoresError, OSError, ValueError) as exc:
        print(f"cluster-reads {args.action}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    install_store_audit_from_env()
    ap = argparse.ArgumentParser(prog="python -m sisteer.registry.cluster_reads")
    ap.add_argument("--root", default=os.environ.get("REGISTRY_ROOT") or os.path.expanduser("~/registry"))
    add_arguments(ap)
    a = ap.parse_args()
    raise SystemExit(cmd_cluster_reads(a, os.path.abspath(a.root)))
