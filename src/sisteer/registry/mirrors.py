"""Complete record-log fan-out to the mirrors: immutable records move separately from a
VERSIONED checksum manifest whose pointer switches only after every named record has arrived.

* immutable records — ``records/**`` (incl. ``harvester/``, ``adapters/``, plain and ``.gz``) and
  ``rulings/**`` — are copied with ignore-existing semantics (resumable; a record file is never
  rewritten, corrections are new files);
* ``manifests/CHECKSUMS-<ingest_seq>.sha256`` holds sorted ``<logical sha>  <logical relpath>`` lines
  where the sha is ALWAYS over the DECOMPRESSED content, so a file's logical sha is the same before
  and after ``gzip_after_ingest``; ``manifests/CHECKSUMS.CURRENT`` (one line, the manifest name) is
  written ``.tmp`` + rename ONLY after :func:`verify_manifest` confirmed every named record on that
  destination;
* ONE stdlib verifier (:func:`verify_manifest`) runs everywhere — locally, over ssh on the clusters
  (``python3 -S -m sisteer.registry.mirrors verify <root> <manifest>``) — never ``sha256sum -c``.

Destinations come from ``<root>/config.toml`` (``[[clusters]]``; :mod:`sisteer.registry.config`) or an older ``config.json`` →
``mirrors: [{name, kind: local|ssh, path, host, remote_src}]`` (taken verbatim). The record log's copies are the grading host and
the clusters (the 2026-09-28 ruling); the Modal destination and its chunked push, ledger, in-flight marker and probes were
REMOVED with the transport -- a ``kind: modal`` entry is refused by name at load
(``config.check_destinations``), and :func:`sync` raises on an unknown kind rather than skipping it. Tests use ``kind: local``
destinations (a directory) and fake ``ssh``/``rsync`` runners; ``ssh`` rsyncs to ``<host>:<path>`` and runs the verifier there
under ``PYTHONPATH=<remote_src>`` (the repo mirror ``mirror-repos.sh`` maintains). A full re-hash is ``registry mirrors verify``.
The sync stops starting copy chunks when the wall-clock budget (the unit's ``TimeoutStartSec``) would not fit another one and
exits 0 — a partial run that made progress, resumed by the next timer fire.

The ssh (cluster) destinations verify INCREMENTALLY per cycle (2026-09-28): the destination's ``CHECKSUMS.CURRENT``
names the manifest it last verified clean (the base), and the verifier is asked ``--since <base>``: presence of EVERY
manifest entry comes from ONE names-only directory walk (cluster A: ~22 s for 430,000 files, against ~150 s of per-entry
``stat`` or ~65 min of hashing at 110 files/s — which is why the whole-log verify crossed the 3,600 s timeout a third of
the time and, on a login node, ran the heavy work outside Slurm); hashed are the entries the base did not carry (new files),
every carried entry with BOTH a plain and a ``.gz`` copy on the destination (a file gzipped after the base was acked; the
plain copy is pruned once both hash clean, as before), and a seeded random ``--sample`` (:data:`VERIFY_SAMPLE`) of the
carried rest, so silent loss of an old file is caught by the walk and silent corruption by the sample over successive
cycles. A FULL verify (every entry hashed, a process pool of ``--workers``) runs when the destination has no base, on
``--full`` (the weekly ``registry-mirrors-full.timer``), and for ``registry mirrors verify`` — under ``srun`` on the
cluster (:func:`_slurm_wrap`: partition / QoS / CPUs / memory from the ``[[clusters]]`` entry, ``--time`` sized from the
entry count, never ``--export``) when the cluster entry has ``slurm = true`` (the default), so no hash pass of the whole
log runs on a login node. The report says what rested on what: ``mode``, ``since``, ``hashed``, ``presence_only``,
``sampled``, ``sample_seed``.

The ssh COPY is incremental too (2026-09-28, later the same day): the one ssh that reads the destination's ``CHECKSUMS.CURRENT``
also fixes the copy's base, and the run sends exactly the root-relative paths the run's manifest names that the base does not
carry with the same sha (:func:`added_since` — the same delta the verify hashes), as their on-disk files (plain and/or ``.gz``),
in manifest order, through ``rsync --files-from=-`` (:func:`_rsync_files_cmd`; ``-rt --partial-dir --ignore-existing`` kept, so a
record file is still never rewritten) in chunks of :data:`COPY_CHUNK_FILES` paths, one journal line per chunk.
The whole-tree ``rsync -rt --ignore-existing records/ rulings/`` that ran every cycle stat'ed every destination entry — O(total
files), 461,769 of them on cluster A's VAST through a login node at load 40 — and from 2026-09-28 ~22:00Z on did not finish inside
its 3,600 s bound, so the unit hit its 2 h ``TimeoutStartSec`` every run and the cluster A copy stopped advancing (the grading host log
~6,000 records ahead of the last acked manifest). It now runs ONLY when the destination has no acked manifest (or the acked
manifest's file is gone from the grading host) and under ``--full`` (the weekly ``registry-mirrors-full.timer``), which is also what
brings a ``.gz`` twin to a destination that holds a carried file's plain copy. The incremental verify's ``missing`` entries are
re-copied (the corrupt ones' ``--checksum`` repair, one list) and verified again, so a copy interrupted mid-chunk heals in the next
cycle; the copy stops starting chunks when the run's budget would not fit another (``stopped: budget``, the rest rides the next
run). rsync's partial-transfer exits (23, 24: a listed file the harvester gzipped meanwhile) are a journal line, not the
destination's error — the verify is the authority on what arrived. The row says ``copy_mode`` (``incremental`` / ``full``),
``copy_since``, ``copy_files`` (paths listed), ``chunks_done`` / ``chunks_total``, and ``copied`` (files rsync's ``--stats``
reports transferred; -1 when it printed none).

MANIFEST RETENTION (2026-10-08). Every pass wrote a new ~80 MB manifest and nothing pruned them: the grading host held 1,248 (72 GB on a
root disk at 96 %; a full disk pauses the registry) and each cluster mirror ~1,110 copies. :func:`prune_manifests` -- run by
:func:`sync_all` after every destination's sync and by ``registry mirrors prune-manifests [--keep K] [--dry-run] [--dest NAME]`` --
applies ONE rule (:func:`plan_manifest_retention`) to the grading host's ``manifests/`` and to each destination's: keep the side's own
``CHECKSUMS.CURRENT``, the newest :data:`MANIFEST_KEEP` by ingest_seq, and every manifest a live reference names -- each
destination's pointer (the ``--since`` base of its next incremental copy and verify, whose file the grading host must still hold for
:func:`added_since`), known live from the pass's results, from the destination's own listing, or from ``state/mirror_state.json``
-- and delete the rest. The pointers are a durable per-destination RECORD, ``state/mirror_pointers.json`` (:func:`record_pointer`),
merged on every observation and never erased by an unknown one, so an outage, a filtered ``--dest`` run or a verify-only cycle
cannot lose them -- nor by an OLDER one: every observation is stamped when it is made (a read as it begins, a switch as it
completes) and a merge older than the recorded observation is refused, so a listing that overlapped a switch cannot put the old
pointer back; the destination listing fails closed (explicit markers, stderr kept), refusing the side -- and the grading host when a
listed destination's pointer is unknown from every source, or while any file under ``state/mirror_pointers.unrecorded/`` cannot be
read or validated (:func:`unreadable_holds`: a hold the retention cannot read may name the base it would otherwise delete, so the
grading host deletes nothing until a merge records the pointer of the EXACT destination the file's name encodes, which removes it -- a name
that decodes to no exact destination is an operator's to remove; the pointer record and every hold are written durably -- temp file,
fsync, rename, the file and its directory fsynced -- so a crash leaves a whole file or the previous one). The reader census behind the rule: :func:`_remote_base` (a destination's pointer), :func:`latest_manifest`
(the newest; ``mirrors verify`` and the status page), and ``replica._latest_manifest`` (the log root's pointer only; no replica
head or build state names a manifest), so nothing ever names an older manifest and no pin file is needed. A destination's
listing or deletion failing is that side's ``error`` in the pass's state, never the pass's failure; the next pass retries.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import db as DB
from . import records as R
from . import site_aliases as SA

MANIFEST_DIR = "manifests"
CURRENT_NAME = "CHECKSUMS.CURRENT"
LOG_SUBDIRS = ("records", "rulings")

#: Wall-clock budget of one `mirrors sync` (the unit's TimeoutStartSec=2h); overridable with `--budget-s`.
DEFAULT_SYNC_BUDGET_S = 2 * 3600
#: No new copy chunk starts unless this multiple of the last chunk's duration is left in the budget.
BUDGET_MARGIN = 1.5
#: Incremental verify: carried entries (present in the base manifest with the same sha, one copy on the destination) re-hashed
#: per cycle, chosen at random with a recorded seed; the rest are confirmed by presence from the directory walk. Measured
#: 2026-09-28: 110 files/s on cluster A's login node, 194 on cluster B's, so 2,000 is ~20 s. Per destination: `verify_sample`.
VERIFY_SAMPLE = 2000
#: The remote verifier's wall-clock bound (queue wait + run for the Slurm path); the systemd unit's TimeoutStartSec is 2 h.
VERIFY_TIMEOUT_S = 3600
#: Full verify under Slurm: `--time` is sized from the entry count at this many files/s PER WORKER (measured 110 serial on
#: cluster A; the pool is I/O-bound, so per-worker throughput is taken at the serial rate), padded 2x, within these bounds.
SLURM_FILES_PER_S_PER_WORKER = 100.0
SLURM_TIME_MIN_S = 600
SLURM_TIME_MAX_S = VERIFY_TIMEOUT_S
SLURM_PARTITION = "general"
SLURM_VERIFY_CPUS = 4
SLURM_VERIFY_MEM = "16G"


@dataclass
class Destination:
    name: str
    kind: str = "local"          # local | ssh (the Modal kind was removed with its transport)
    path: str = ""
    host: str = ""
    remote_src: str = ""         # ssh: PYTHONPATH of the mirrored repo's src/ for the remote verifier
    environment: Optional[str] = None
    sample: int = VERIFY_SAMPLE          # ssh: carried entries re-hashed per incremental verify
    slurm: bool = False                  # ssh: run a FULL verify under `srun` (config-derived clusters default to True)
    qos: str = ""                        # ssh/slurm: the QoS of that job (the cluster's own QoS name)
    partition: str = SLURM_PARTITION     # ssh/slurm
    cpus: int = SLURM_VERIFY_CPUS        # ssh/slurm: --cpus-per-task, and the verifier's --workers
    mem: str = SLURM_VERIFY_MEM          # ssh/slurm: --mem (never 0)

    @staticmethod
    def from_config(d: Mapping[str, object]) -> "Destination":
        return Destination(str(d["name"]), str(d.get("kind", "local")), str(d.get("path", "")), str(d.get("host", "")),
                           str(d.get("remote_src", "") or ""), (str(d["environment"]) if d.get("environment") else None),
                           sample=int(d["verify_sample"]) if d.get("verify_sample") is not None else VERIFY_SAMPLE,
                           slurm=bool(d.get("slurm", False)), qos=str(d.get("qos", "") or ""),
                           partition=str(d.get("partition") or SLURM_PARTITION),
                           cpus=int(d.get("verify_cpus") or SLURM_VERIFY_CPUS), mem=str(d.get("verify_mem") or SLURM_VERIFY_MEM))

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "kind": self.kind, "path": self.path, "host": self.host, "remote_src": self.remote_src,
                "environment": self.environment, "verify_sample": self.sample, "slurm": self.slurm,
                "qos": self.qos, "partition": self.partition, "verify_cpus": self.cpus, "verify_mem": self.mem}


@dataclass
class VerifyReport:
    manifest: str
    ok: int = 0
    missing: List[str] = field(default_factory=list)
    corrupt: List[str] = field(default_factory=list)
    pruned_plain: int = 0
    bytes: int = 0                       # on-disk bytes of the files this verify HASHED (every file, in full mode)
    mode: str = "full"                   # "full" | "incremental"
    since: Optional[str] = None          # incremental: the base manifest's name
    hashed: Optional[int] = None         # entries whose content was hashed
    presence_only: Optional[int] = None  # entries confirmed by presence alone (carried from the base, one copy on disk)
    sampled: Optional[int] = None        # carried entries hashed as the random sample (a subset of `hashed`)
    sample_seed: Optional[int] = None
    seconds: float = 0.0
    note: Optional[str] = None           # e.g. the base was asked for but the verifier is older than the flag

    @property
    def clean(self) -> bool:
        return not self.missing and not self.corrupt

    def as_json(self) -> Dict[str, object]:
        return {"manifest": self.manifest, "ok": self.ok, "missing": self.missing, "corrupt": self.corrupt,
                "pruned_plain": self.pruned_plain, "bytes": self.bytes, "mode": self.mode, "since": self.since,
                "hashed": self.hashed, "presence_only": self.presence_only, "sampled": self.sampled,
                "sample_seed": self.sample_seed, "seconds": round(self.seconds, 1), "note": self.note}


def destinations(root: str) -> List[Destination]:
    """The configured destinations; also installs the root's site alias table (:mod:`site_aliases`), so the destination-keyed
    state below reads a destination recorded under its site's older name as the same destination (:func:`read_pointers`)."""
    from . import config as CFG
    from .harvest import load_config
    SA.install(CFG.load(root))
    return [Destination.from_config(d) for d in (load_config(root).get("mirrors") or [])]


# --------------------------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------------------------


def log_files(root: str) -> List[Tuple[str, str]]:
    """[(logical relpath, on-disk path)] of every record file under records/ and rulings/."""
    out = []
    for sub in LOG_SUBDIRS:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for path in R.iter_record_files(base):
            out.append((R.logical_uri(os.path.relpath(path, root)), path))
    # a plain and a gz copy of the same logical file: keep one entry
    seen: Dict[str, str] = {}
    for rel, path in out:
        seen.setdefault(rel, path)
    return sorted(seen.items())


@dataclass
class ManifestWrite:
    name: str
    entries: int = 0
    bytes: int = 0                                   # on-disk bytes of the files the manifest names (as hashed here)
    vanished: List[str] = field(default_factory=list)   # listed, then gone with no ``.gz`` twin before it could be hashed


def _sha_or_twin(path: str) -> Tuple[Optional[str], Optional[str]]:
    """(logical sha, the path hashed) — of ``path``, or of its ``.gz`` twin when the plain file vanished between the listing
    and the hash (the harvester's ``gzip_after_ingest`` writes ``<f>.gz`` then removes ``<f>``; the logical content is the
    same); (None, None) when neither is there any more."""
    try:
        return R.file_logical_sha(path), path
    except FileNotFoundError:
        twin = path + ".gz" if not path.endswith(".gz") else None
        if twin and os.path.exists(twin):
            try:
                return R.file_logical_sha(twin), twin
            except FileNotFoundError:
                return None, None
        return None, None


def write_manifest_report(root: str, ingest_seq: int) -> ManifestWrite:
    """Write ``manifests/CHECKSUMS-<ingest_seq>.sha256`` (a NEW immutable file per sync). A file listed and then gzipped by the
    harvester before its hash is read from the ``.gz`` twin (same logical sha); one gone with no twin is SKIPPED and named in
    ``vanished`` (the next manifest picks it up) — on 2026-09-28 the ``FileNotFoundError`` killed the whole ``mirrors sync``."""
    lines = []
    out = ManifestWrite("")
    for rel, path in log_files(root):
        sha, hashed = _sha_or_twin(path)
        if sha is None:
            out.vanished.append(rel)
            continue
        lines.append(f"{sha}  {rel}")
        out.entries += 1
        try:
            out.bytes += os.path.getsize(hashed)
        except OSError:
            pass
    name = f"CHECKSUMS-{int(ingest_seq):012d}.sha256"
    out.name = name
    d = os.path.join(root, MANIFEST_DIR)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, p)
    return out


def write_manifest(root: str, ingest_seq: int) -> str:
    """:func:`write_manifest_report`'s name only (the historical signature)."""
    return write_manifest_report(root, ingest_seq).name


def read_manifest(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            sha, _, rel = line.partition("  ")
            out.append((sha, rel))
    return out


manifest_sha = R.file_sha256


def current_manifest(root: str) -> Optional[str]:
    return R.read_pointer(os.path.join(root, MANIFEST_DIR, CURRENT_NAME))


def switch_current(root: str, manifest_name: str) -> None:
    R.write_pointer(os.path.join(root, MANIFEST_DIR, CURRENT_NAME), manifest_name)


# --------------------------------------------------------------------------------------------
# manifest retention
# --------------------------------------------------------------------------------------------

#: Manifests kept per side besides the side's CHECKSUMS.CURRENT and the referenced ones: the newest this many by ingest_seq, about
#: two days of half-hourly passes (measured 2026-10-08: a pass every ~20 min, so ~32 h). `registry mirrors prune-manifests --keep K`
#: and `sync_all(keep=)` override it.
MANIFEST_KEEP = 96
_MANIFEST_NAME = re.compile(r"^CHECKSUMS-(\d{12})\.sha256$")
#: Names of a side's deletions the state row carries (the full list is in the log line's counts).
DELETED_NAMES_IN_ROW = 50
#: ssh bounds of a destination's prune: ONE listing (`find -maxdepth 1 -printf`, with the pointer) and ONE deletion (`xargs rm -f` over
#: the names on stdin).
PRUNE_LIST_TIMEOUT_S = 120
PRUNE_DELETE_TIMEOUT_S = 600


def is_manifest_name(name: str) -> bool:
    """``CHECKSUMS-<12 digits>.sha256`` exactly: what the prune may list and delete; a ``.tmp`` mid-write or the pointer never."""
    return bool(_MANIFEST_NAME.match(name or ""))


def list_manifests(directory: str) -> Dict[str, int]:
    """``{name: bytes}`` of the manifests under ``directory`` from one ``scandir`` (an absent directory is empty)."""
    out: Dict[str, int] = {}
    try:
        with os.scandir(directory) as it:
            for e in it:
                if is_manifest_name(e.name) and e.is_file(follow_symlinks=False):
                    try:
                        out[e.name] = e.stat(follow_symlinks=False).st_size
                    except OSError:
                        out[e.name] = 0
    except FileNotFoundError:
        return {}
    return out


@dataclass
class RetentionPlan:
    """The rule applied to ONE side's listing: what stays, what goes, and why the referenced ones stay."""
    side: str                                 # "grading_host" or a destination's name
    location: str                             # the manifests directory as read (host:path for ssh)
    keep: int
    current: Optional[str]                    # the side's CHECKSUMS.CURRENT (None: no pointer, as on the grading host)
    kept: List[str] = field(default_factory=list)       # ascending by ingest_seq
    deleted: List[str] = field(default_factory=list)    # ascending by ingest_seq
    kept_bytes: int = 0
    deleted_bytes: int = 0
    referenced: Dict[str, List[str]] = field(default_factory=dict)   # kept by a live reference: name -> reasons
    error: Optional[str] = None               # the listing or the deletion failed; logged, the next pass retries
    removed: int = 0                          # names actually removed (0 on a dry run)
    dry_run: bool = False

    @property
    def oldest_kept(self) -> Optional[str]:
        return self.kept[0] if self.kept else None

    @property
    def newest_kept(self) -> Optional[str]:
        return self.kept[-1] if self.kept else None

    def as_dict(self) -> Dict[str, object]:
        return {"side": self.side, "location": self.location, "keep": self.keep, "current": self.current,
                "manifests": len(self.kept) + len(self.deleted), "kept": len(self.kept), "deleted": len(self.deleted),
                "kept_bytes": self.kept_bytes, "deleted_bytes": self.deleted_bytes, "oldest_kept": self.oldest_kept,
                "newest_kept": self.newest_kept, "referenced": {k: list(v) for k, v in self.referenced.items()},
                "deleted_names": self.deleted[:DELETED_NAMES_IN_ROW], "removed": self.removed, "dry_run": self.dry_run,
                "error": self.error}


def plan_manifest_retention(listing: Mapping[str, int], current: Optional[str], references: Mapping[str, Sequence[str]],
                            keep: int = MANIFEST_KEEP, side: str = "grading_host", location: str = "") -> RetentionPlan:
    """THE rule, pure. Of the manifests in ``listing`` (``{name: bytes}``; a name not shaped ``CHECKSUMS-<12 digits>.sha256`` is
    ignored, so a ``.tmp`` mid-write or the pointer is never touched) keep (a) ``current``, the side's own ``CHECKSUMS.CURRENT``,
    (b) the newest ``keep`` by ingest_seq (the zero-padded name order), and (c) every name ``references`` carries -- a destination's
    pointer is the ``--since`` base of its next incremental copy and verify and the grading host must still hold that file for the delta
    (:func:`added_since`); ``replica._latest_manifest`` reads the log root's pointer only -- and delete the rest. A referenced name
    the listing lacks has nothing to keep; ``referenced`` reports the present ones with their reasons, inside the newest ``keep`` or not."""
    k = int(keep)
    if k < 1:
        raise ValueError(f"manifest retention: keep must be at least 1 (got {keep}); the newest manifest is never deleted")
    names = sorted(n for n in listing if is_manifest_name(n))
    keep_set = set(names[-k:])
    if current and current in listing and is_manifest_name(current):
        keep_set.add(current)
    plan = RetentionPlan(side, location, k, current)
    for name in names:
        reasons = list(references.get(name) or [])
        if reasons:
            keep_set.add(name)
            plan.referenced[name] = reasons
    for name in names:
        if name in keep_set:
            plan.kept.append(name)
            plan.kept_bytes += int(listing[name] or 0)
        else:
            plan.deleted.append(name)
            plan.deleted_bytes += int(listing[name] or 0)
    return plan


POINTERS_NAME = "mirror_pointers.json"
POINTERS_LOCK_NAME = "mirror_pointers.lock"
#: How long one observer waits for another's merge of the pointer record (a read, a dump and a rename: milliseconds) before
#: giving up with ``TimeoutError``.
POINTERS_LOCK_TIMEOUT_S = 60.0


def pointers_path(root: str) -> str:
    return os.path.join(root, "state", POINTERS_NAME)


def pointers_lock_path(root: str) -> str:
    return os.path.join(root, "state", POINTERS_LOCK_NAME)


@contextlib.contextmanager
def pointers_lock(root: str, timeout_s: float = POINTERS_LOCK_TIMEOUT_S):
    """An exclusive ``flock`` on ``state/mirror_pointers.lock`` around every read-modify-write of the pointer record. The
    scheduled pass (:func:`sync_all`), a ``mirrors verify`` and an on-demand ``prune-manifests`` may overlap, and two observers
    that each read the whole record and replace it would keep only the later one's observation -- the lost one being, at worst,
    an unreachable destination's base, which the grading host then deletes. Held for the merge
    alone; a holder that dies releases it (the kernel's); ``TimeoutError`` (an ``OSError``) after ``timeout_s``."""
    p = pointers_lock_path(root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fh = open(p, "a+")
    deadline = time.monotonic() + float(timeout_s)
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"the pointer record's lock {p} was held by another observer for over {float(timeout_s):.0f} s; "
                                       f"this observation was not recorded")
                time.sleep(0.02)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


#: ``state/mirror_pointers.unrecorded/<destination>~<destination digest>~<stamp>~<digest>.json``: ONE file per OBSERVATION whose merge into the record
#: FAILED (the lock held past its timeout, the record unwritable, two names in the same millisecond), written outside the lock and
#: never overwritten by another observation of the destination: the file is the observation's identity (destination, name, stamp),
#: so two observers of one destination each keep what they saw. A file per DESTINATION let an older observation written later
#: replace the fresher one -- the pass's switch to B held, then an on-demand prune's read of A from before the switch held over it
#: -- after which the record and the hold both named A and the grading host deleted B, the destination's actual base (closing review of
#: 8eef8bbb2, MEDIUM). A merge that records a NAME for the destination releases the observations it covers -- the held name itself,
#: or one observed strictly before it (:func:`release_unrecorded`) -- judging each file by its own content, so a hold written while a
#: release runs is either not yet listed or judged on its stamp, never removed with a file another observation owned. The retention
#: keeps every held name as a reference meanwhile. A file there that cannot be READ or validated (zero bytes, not JSON, no
#: destination, not a manifest name: a storage error, or a crash that tore it) is never taken for no hold -- it may be the fresher
#: observation, and the record's name alone is then the OLDER pointer -- so the grading host's deletions are refused while any stands
#: (:func:`unreadable_holds`, :func:`prune_manifests`), and only a merge that records a NAME for the destination the file is named for
#: removes it (:func:`release_unrecorded`; re-check of 360d61f4b, MEDIUM: ignored, a torn hold of B let the record's A stand alone, and
#: B, the destination's actual base, went once it left the newest K). The file's name carries the destination EXACTLY --
#: percent-encoded (:func:`_encode_segment`) and followed by a digest of the exact name that a release recomputes -- so a destination
#: whose plain form coincides with another's (``eu west`` / ``eu_west``) never releases the other's unreadable file, and a name that
#: decodes to no exact destination (a stray file; the three-segment names written before this rule) is released by no merge
#: (re-check of 614abd0b4, MEDIUM: the lossy plain prefix let the second destination's merge release the first's torn hold). Each
#: hold, and the record, is written durably (:func:`_write_json_durably`).
POINTERS_UNRECORDED_DIR = "mirror_pointers.unrecorded"
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]")
_PERCENT_BYTE = re.compile(r"%([0-9A-F]{2})")
_HOLD_SEP = "~"      # outside the plain and the encoded sets, so a hold file's name splits into its segments exactly
_HOLD_SUFFIX = ".json"
_HOLD_TMP_SUFFIX = ".tmp"     # a hold's temp name ends here, never in ``.json``, so a write in progress is never listed as a hold


def unrecorded_dir(root: str) -> str:
    return os.path.join(root, "state", POINTERS_UNRECORDED_DIR)


def _plain_segment(text: object) -> str:
    """``text`` in plain file-name characters, for the eye only: LOSSY (``eu west`` and ``eu_west`` coincide), so never an identity."""
    return _UNSAFE_IN_FILENAME.sub("_", str(text)) or "_"


def _encode_segment(text: object) -> str:
    """``text`` as a file-name segment that decodes back EXACTLY (:func:`_decode_segment`): every character outside ``[A-Za-z0-9._-]``
    -- the separator ``~`` and ``%`` among them -- as ``%`` and the two upper-case hex digits of each of its UTF-8 bytes."""
    return "".join(ch if not _UNSAFE_IN_FILENAME.search(ch) else "".join(f"%{b:02X}" for b in ch.encode("utf-8")) for ch in str(text))


def _decode_segment(segment: str) -> Optional[str]:
    """The exact text :func:`_encode_segment` wrote, or None when ``segment`` is not its strict encoding: a ``%`` not followed by two
    upper-case hex digits, a character the encoder would have escaped, bytes that are not UTF-8, or anything that does not re-encode to
    the very same segment."""
    buf = bytearray()
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch == "%":
            m = _PERCENT_BYTE.match(segment, i)
            if not m:
                return None
            buf.append(int(m.group(1), 16))
            i += 3
        elif _UNSAFE_IN_FILENAME.search(ch):
            return None
        else:
            buf.extend(ch.encode("ascii"))
            i += 1
    try:
        text = buf.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if _encode_segment(text) == segment else None


def _dest_digest(dest_name: object) -> str:
    """Eight hex digits of the exact destination name, the hold file name's check on its first segment."""
    return hashlib.sha1(str(dest_name).encode("utf-8")).hexdigest()[:8]


def unrecorded_path(root: str, dest_name: str, current: str, observed_utc: str) -> str:
    """The hold file of ONE observation -- ``dest_name``'s ``CHECKSUMS.CURRENT`` read or switched as ``current`` at ``observed_utc``:
    the destination EXACTLY (:func:`_encode_segment`, readable for a plain name, decodable for any), a digest of the exact destination
    (:func:`_dest_digest`, what a release recomputes and compares), the stamp in plain characters for the eye, then a digest of the real
    three, so two destinations whose plain forms coincide, a name the plain form bends, and one name at two instants all get distinct
    files and no file is ever taken for another destination's (the file carries the real values; :func:`unrecorded_pointers` reads
    those, not the name; :func:`_hold_name_destination` reads the name when the content cannot be read). Two observers of the same
    pointer at the same instant made the same observation and share the file."""
    digest = hashlib.sha1(json.dumps([str(dest_name), str(current), str(observed_utc)]).encode("utf-8")).hexdigest()[:8]
    fn = _HOLD_SEP.join((_encode_segment(dest_name), _dest_digest(dest_name), _plain_segment(observed_utc), digest)) + _HOLD_SUFFIX
    return os.path.join(unrecorded_dir(root), fn)


def _write_json_durably(p: str, obs: Mapping[str, object]) -> None:
    """Write ONE JSON file -- a hold, or the pointer record -- durably and atomically: a per-process, per-write temp name beside it
    (``<path>.<pid>.<hex>.tmp``, outside the ``.json`` shape :func:`_hold_files` lists, so a write in progress is never read as a hold),
    the document flushed and fsynced, ``os.replace`` into place, then the file and its directory fsynced (``DB.fsync_file_and_dir``: a
    rename is durable only once its directory entry is) -- the shape of ``records.write_json_atomic`` plus the directory half. A hold
    written without this could be torn or gone after a crash while the merge it stands in for had already failed, and the retention
    then trusted the record's older name (re-check of 360d61f4b, MEDIUM); a record written without this could revert to the older name
    after a crash with the newer name's hold already released (re-check of 614abd0b4, MEDIUM). A write that fails at any step removes
    its temp file (best effort) and raises: the directory holds the whole new file or the previous one, never a partial ``.json``."""
    tmp = f"{p}.{os.getpid()}.{uuid.uuid4().hex[:8]}{_HOLD_TMP_SUFFIX}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obs, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    DB.fsync_file_and_dir(p)


def hold_unrecorded(root: str, dest_name: str, current: str, source: str, now: Optional[str] = None, error: str = "") -> str:
    """Hold ONE observation of ``dest_name``'s ``CHECKSUMS.CURRENT`` (a manifest name, observed at ``now``) that could not be merged
    into the record: a durable write (:func:`_write_json_durably`) into the observation's own file (:func:`unrecorded_path`), no lock,
    nothing read and nothing removed -- another observation of the destination, older or fresher, keeps its own file, and only a
    merge releases one (:func:`release_unrecorded`). Returns the file's path."""
    if not is_manifest_name(str(current)):
        raise ValueError(f"hold_unrecorded: {current!r} is not a manifest name")
    observed = now or R.utc_now()
    p = unrecorded_path(root, dest_name, str(current), observed)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    _write_json_durably(p, {"destination": str(dest_name), "current": str(current), "observed_utc": observed, "source": str(source),
                            "error": str(error)[-400:], "held_utc": R.utc_now()})
    return p


def _hold_files(root: str) -> List[Tuple[str, Optional[Dict[str, object]], Optional[str]]]:
    """Every ``.json`` under the hold directory, in name order, as ``(path, observation, problem)``: the observation's
    ``destination``, ``current``, ``observed_utc``, ``source`` and ``error`` with no problem, or no observation and the problem for a
    file that is not a readable hold -- it could not be read (a storage error), is not JSON (zero bytes, a torn write) or is not a
    hold (no destination, not a manifest name). A ``.tmp`` mid-write is not listed, and a file gone between the listing and its read
    was released by a merge meanwhile, so it is nothing. The one read behind :func:`unrecorded_pointers`, :func:`unreadable_holds`
    and :func:`release_unrecorded`."""
    d = unrecorded_dir(root)
    try:
        names = sorted(os.listdir(d))
    except (FileNotFoundError, NotADirectoryError):     # no hold was ever written there (a file in the directory's place refuses the write, loudly)
        return []
    out: List[Tuple[str, Optional[Dict[str, object]], Optional[str]]] = []
    for fn in names:
        if not fn.endswith(_HOLD_SUFFIX):
            continue
        p = os.path.join(d, fn)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:                       # released between the listing and here: a merge covered it
            continue
        except OSError as exc:
            out.append((p, None, f"cannot be read: {exc.strerror or exc}"))
            continue
        except ValueError:                              # json's decode errors, a wrong encoding included
            out.append((p, None, "not JSON"))
            continue
        if not isinstance(data, Mapping) or not data.get("destination"):
            out.append((p, None, "not a hold: no destination"))
        elif not is_manifest_name(str(data.get("current") or "")):
            out.append((p, None, f"not a hold: {data.get('current')!r} is not a manifest name"))
        else:
            out.append((p, {"destination": str(data["destination"]), "current": str(data["current"]), "observed_utc": data.get("observed_utc"),
                            "source": data.get("source"), "error": data.get("error")}, None))
    return out


def _hold_name_destination(filename: str) -> Optional[str]:
    """The EXACT destination a hold file's NAME encodes (:func:`unrecorded_path`: four segments, the first the destination
    percent-encoded and the second a digest of the exact destination), or None when the name does not decode to one -- another shape
    (the three-segment names written before this rule, a stray file), a first segment that is not a strict encoding, or a decoded name
    whose digest is not the second segment -- the only thing left to read of a hold whose content cannot be, so a release acts on an
    exact identity or not at all, never on a lossy match (re-check of 614abd0b4, MEDIUM)."""
    stem = filename[:-len(_HOLD_SUFFIX)] if filename.endswith(_HOLD_SUFFIX) else filename
    parts = stem.split(_HOLD_SEP)
    if len(parts) != 4:
        return None
    name = _decode_segment(parts[0])
    return name if name is not None and _dest_digest(name) == parts[1] else None


def unreadable_holds(root: str) -> List[Dict[str, object]]:
    """``[{"path", "destination", "problem"}, ...]``, in name order, for every ``.json`` under the hold directory that is NOT a readable
    hold (:func:`_hold_files`): ``destination`` is the exact destination its name encodes (:func:`_hold_name_destination`), or None when
    the name decodes to none. Such a file may be the fresher observation of its destination's pointer, so :func:`prune_manifests`
    refuses the grading host's deletions while any stands, and only a merge recording a name for exactly that destination removes it
    (:func:`release_unrecorded`); a file whose name decodes to no destination is an operator's to read and remove."""
    return [{"path": p, "destination": _canonical_dest(_hold_name_destination(os.path.basename(p))), "problem": problem}
            for p, obs, problem in _hold_files(root) if obs is None]


def _stamp_ms(utc: object) -> int:
    """Epoch ms of a registry stamp, for ordering the holds freshest first; one that cannot be read sorts oldest."""
    try:
        return R.utc_to_epoch_ms(str(utc))
    except (ValueError, TypeError):
        return -1


def unrecorded_pointers(root: str) -> Dict[str, List[Dict[str, object]]]:
    """``{destination: [{"current", "observed_utc", "source", "error", "path"}, ...]}`` of every held observation
    (:func:`hold_unrecorded`), freshest first, one entry per held NAME -- a name held at several instants (a lock stuck across
    passes) is reported at its freshest, every file standing until a merge covers it. A file that is not a readable hold is not
    here: :func:`unreadable_holds` lists it, and the retention refuses the grading host's deletions while it stands."""
    by_dest: Dict[str, Dict[str, Dict[str, object]]] = {}
    for p, obs, _problem in _hold_files(root):
        if obs is None:
            continue
        entry = {"current": obs["current"], "observed_utc": obs["observed_utc"], "source": obs["source"], "error": obs["error"], "path": p}
        names = by_dest.setdefault(str(_canonical_dest(obs["destination"])), {})
        prior = names.get(str(obs["current"]))
        if prior is None or (_utc_order(entry["observed_utc"], prior["observed_utc"]) or 0) > 0:
            names[str(obs["current"])] = entry
    return {dest: sorted(names.values(), key=lambda e: -_stamp_ms(e["observed_utc"])) for dest, names in sorted(by_dest.items())}


def release_unrecorded(root: str, dest_name: str, current: str, observed: object) -> int:
    """Remove the holds of ``dest_name`` that a merge recording ``current``, observed at ``observed``, COVERS: a hold of that name
    (answered -- the name is on record, whenever it was observed) or one observed strictly before ``observed`` (the destination
    moved past it); a fresher one stays, and so does one whose stamp cannot be read, until a merge of its name. Each file is judged
    by its own content, never by which observation wrote first, so a hold written while this runs is either not yet listed or
    judged on its stamp. An UNREADABLE file whose name encodes EXACTLY this destination (:func:`_hold_name_destination`, the only
    thing left to read) goes too -- here and nowhere else, so a hold the retention cannot read stands, refusing the grading host's
    deletions, until a merge records a name for that very destination; a merge of another destination (one whose plain form
    coincides included), a recorded absence or a refused merge removes nothing (this runs only after a name was written durably), and
    a name that decodes to no exact destination is released by no merge. Returns how many files went."""
    n = 0
    for p, obs, _problem in _hold_files(root):
        if obs is None:
            covered = _canonical_dest(_hold_name_destination(os.path.basename(p))) == str(dest_name)
        elif _canonical_dest(obs["destination"]) != str(dest_name):
            continue
        else:
            covered = obs["current"] == str(current) or (_utc_order(observed, obs["observed_utc"]) or 0) > 0
        if not covered:
            continue
        try:
            os.remove(p)
        except FileNotFoundError:          # another release took it between the listing and here
            continue
        n += 1
    return n


def _canonical_dest(name: Optional[object]) -> Optional[str]:
    """A destination name read back from the state as the destination it is now: the site alias table maps a name recorded before
    the site renamed its destinations (:func:`site_aliases.canonical`); None stays None."""
    return None if name is None else str(SA.canonical(str(name)))


def _utc_order(a: object, b: object) -> Optional[int]:
    """-1, 0 or 1 for the registry timestamp ``a`` before, in the same millisecond as, or after ``b``; None when either is not one
    (a hand-edited file), which leaves the order undecided."""
    try:
        x, y = R.utc_to_epoch_ms(str(a)), R.utc_to_epoch_ms(str(b))
    except (ValueError, TypeError):
        return None
    return (x > y) - (x < y)


def read_pointers(root: str) -> Dict[str, Dict[str, object]]:
    """``{destination: {"current", "observed_utc", "source"}}`` -- the last OBSERVED ``CHECKSUMS.CURRENT`` per destination
    (``state/mirror_pointers.json``), a record independent of the latest outcome rows. ``current`` is a manifest name, or None:
    the pointer was observed ABSENT (a fail-closed listing without a pointer file, a local destination without one) -- an
    ESTABLISHED state, unlike a destination with no entry, whose pointer nothing ever observed. An entry whose ``current`` is
    anything else is ignored. A destination recorded under its site's older name (before the destinations were renamed) is read
    as the destination it is now (:func:`_canonical_dest`), and an entry under the current name wins over it, so the next merge
    rewrites the record under the current name and the pointer, the ``--since`` base the retention keeps, is never orphaned."""
    data = R.read_json_or_none(pointers_path(root))
    recs = data.get("destinations") if isinstance(data, Mapping) else None
    out: Dict[str, Dict[str, object]] = {}
    if isinstance(recs, Mapping):
        for name, rec in recs.items():
            if isinstance(rec, Mapping) and "current" in rec and (rec["current"] is None or is_manifest_name(str(rec["current"]))):
                key = str(_canonical_dest(name))
                if key in out and str(name) != key:
                    continue
                out[key] = dict(rec)
    return out


def record_pointer(root: str, dest_name: str, current: Optional[str], source: str, now: Optional[str] = None,
                   absent: bool = False) -> bool:
    """Merge ONE observation of a destination's ``CHECKSUMS.CURRENT`` into ``state/mirror_pointers.json`` (whole file, tmp +
    rename, under :func:`pointers_lock`). The record is per destination and survives every later outcome: an UNKNOWN observation
    (None, or not a manifest name) never erases a known pointer -- a sync that failed before its pointer probe leaves the record
    as it was, where the state file's row would say None; a destination a filtered ``mirrors sync --dest`` omits keeps its
    record; and every observer writes the same record -- a sync's result (``SyncResult.dest_current``), a verify's read
    (``verify_destination``'s ``remote_current``) and a prune's listing. ``absent`` says a None
    was a DEFINITE observation (the fail-closed listing's ``POINTER-ABSENT`` / ``DIR-ABSENT``, a local destination without a
    pointer file): recorded as ``{"current": null}`` for a destination without an entry, so the retention counts its pointer as
    established (nothing to keep for it) rather than unknown (hold the grading host's deletions), while a recorded NAME stands until a
    new name is observed -- a destination that lost its pointer switches to a new one at its next clean sync, and a listing that
    saw no directory while a mount was away must not release the base. Returns whether a
    record was written.

    A merge that FAILS (the lock held past its timeout, the record unwritable) must not lose a NAME it observed: the record would
    keep the OLDER name and the retention prune against it while the destination's actual pointer -- the ``--since`` base of its
    next copy -- fell out of the newest K and was deleted on the grading host, its next sync a whole-tree rsync (record A, a
    switch to B whose merge timed out, then an outage row). So the observation is HELD first
    (:func:`hold_unrecorded`: a file of its own, per observation, under ``state/mirror_pointers.unrecorded/``, written without the
    lock) and the error is re-raised naming the hold; the retention keeps a held name as a reference (:func:`manifest_references`)
    and reports it until a merge records a name for that destination -- this function, on success -- which releases the hold. A
    recorded ABSENCE releases
    nothing (a mount away must not release the base), and an absence that fails to merge is not held (nothing to keep for it; an
    unknown destination holds the grading host's deletions by itself).

    The merge is FRESHNESS-AWARE: the lock serializes the writes, not the observations
    behind them, so an on-demand prune that listed a destination's pointer A while the scheduled pass was switching it to B, and
    took the lock after the pass had recorded B, wrote its earlier A over B -- and released B's hold, had the pass's merge failed;
    an outage row then hid B from the state, the record's A was preferred, and B, the destination's actual base, was deleted on
    the grading host once it fell outside the newest K. ``now`` is therefore the instant the observation was MADE, stamped by the
    observer -- a read (a listing, a verify's read, the sync's probe of its base) as it BEGINS, so a switch during the read is
    the fresher; a switch as it completes -- and a name observed before the recorded one (``observed_utc``) is refused: nothing
    written, nothing released. Two names in the same millisecond cannot be ordered: the incoming one is held besides the recorded
    (both kept until a fresher merge decides). A name always replaces a recorded absence, and an absence never a name, whenever
    either was observed (an absence is less information than a name, not a later state). A hold is released
    only by a merge of the held name or one observed strictly after the held observation; one with a stamp that cannot be read
    stays (loud, each pass) until a merge of its name. Each observation is held in a file of its own, never overwritten by another
    (a file per destination let an older observation held later replace the fresher, live one,
    and a release that removed the destination's file took a fresher hold written between its read and its remove), and a release
    judges each file by its own content (:func:`release_unrecorded`). A hold is written DURABLY (:func:`_write_json_durably`: temp
    file, fsync, rename, the file and its directory fsynced) and one that cannot be read or validated is never ignored: the retention
    refuses the grading host's deletions while it stands (:func:`unreadable_holds`: a torn hold of B read as
    no hold, the record's A stood alone, and B went), and only this function's successful merge of a name for EXACTLY the destination
    the file's name encodes removes it (the lossy plain prefix let a destination whose plain form
    coincided release another's torn hold). The RECORD is written the same way, under the lock, before the merge counts as made and
    before any hold is released: a record whose rename was not yet durable could revert to the older name after a
    crash, with the newer name's hold already gone -- the very loss the hold exists to prevent."""
    if current is not None and not is_manifest_name(str(current)):
        return False
    if current is None and not absent:
        return False
    observed = now or R.utc_now()
    try:
        with pointers_lock(root):
            recs = read_pointers(root)
            prior = recs.get(str(dest_name))
            if prior is not None:
                if current is None and prior.get("current") is not None:
                    return False                           # a definite absence does not release a recorded name
                if current is not None and prior.get("current") is None:
                    pass                                   # a name replaces an absence, whenever either was observed
                else:                                      # name against name, absence against absence: the fresher observation stands
                    order = _utc_order(observed, prior.get("observed_utc"))
                    if order is not None and order < 0:
                        return False                       # STALE: observed before the recorded one; nothing written, nothing released
                    if order == 0 and current is not None and str(current) != str(prior.get("current")):
                        # the same millisecond, two names: which came first is unknown, so this one is held besides the recorded
                        hold_unrecorded(root, dest_name, str(current), source, observed,
                                        error=f"observed in the same millisecond as the recorded {prior.get('current')}: the order is unknown")
                        return False
            recs[str(dest_name)] = {"current": (str(current) if current else None), "observed_utc": observed, "source": str(source)}
            _write_json_durably(pointers_path(root), {"utc": now or R.utc_now(), "destinations": recs})   # durable before anything is released
    except OSError as exc:
        if current is None:
            raise
        try:
            where = hold_unrecorded(root, dest_name, str(current), source, observed, error=str(exc))
        except OSError as exc2:
            raise OSError(f"{exc}; and the observation ({dest_name}: {current}, {source}) could NOT be held either: {exc2}") from exc
        raise OSError(f"{exc}; the observation ({dest_name}: {current}, {source}) is held in {where} and kept by the retention until a "
                      f"merge records {dest_name}'s pointer") from exc
    if current is not None:
        release_unrecorded(root, dest_name, str(current), observed)   # the holds this merge covers: its name, or an older observation
    return True


def last_known_currents(root: str) -> Dict[str, Optional[str]]:
    """Each destination whose ``CHECKSUMS.CURRENT`` is ESTABLISHED, with it: the pointer RECORD first (:func:`read_pointers`,
    merged on every observation; a name, or None for a pointer observed absent), then -- for a destination with no record yet
    (a registry before the record existed) -- the latest ``state/mirror_state.json`` row when it NAMES one: ``dest_current``,
    else the manifest it switched to, else the base it read, else a ``mirrors verify`` row's ``remote_current``. A destination
    in neither is absent from the dict: nothing says, and the retention holds the grading host's deletions for it
    (:func:`prune_manifests`); a row that names nothing (an outage before the probe) establishes nothing."""
    out: Dict[str, Optional[str]] = {name: (str(rec["current"]) if rec["current"] else None) for name, rec in read_pointers(root).items()}
    st = read_state(root) or {}
    for row in (st.get("destinations") or []):
        if not isinstance(row, Mapping) or not row.get("destination") or str(_canonical_dest(row["destination"])) in out:
            continue
        cur = (row.get("dest_current") or (row.get("manifest") if row.get("switched") else None) or row.get("since")
               or row.get("copy_since") or row.get("remote_current"))
        if cur and is_manifest_name(str(cur)):
            out[str(_canonical_dest(row["destination"]))] = str(cur)
    return out


def manifest_references(root: str, currents: Mapping[str, Optional[str]],
                        unrecorded: Optional[Mapping[str, Sequence[Mapping[str, object]]]] = None) -> Dict[str, List[str]]:
    """``{manifest name: [reasons]}`` of every manifest a live pointer names: the grading host's own ``CHECKSUMS.CURRENT`` when it has one
    (the local-destination layout; the grading host root normally has none), each destination's (``currents``), and every HELD
    observation (``unrecorded``, :func:`unrecorded_pointers`: a destination's pointer as an observer read or switched it, whose
    merge into the record failed -- each kept besides whatever the record says for it, until a merge covers it)."""
    refs: Dict[str, List[str]] = {}
    cur = current_manifest(root)
    if cur and is_manifest_name(cur):
        refs.setdefault(cur, []).append("the grading host CHECKSUMS.CURRENT")
    for dest, name in sorted(currents.items()):
        if name and is_manifest_name(name):
            refs.setdefault(name, []).append(f"{dest}'s CHECKSUMS.CURRENT: the --since base of its next incremental copy and verify")
    for dest, entries in sorted((unrecorded or {}).items()):
        for held in entries:
            name = str(held.get("current") or "")
            if is_manifest_name(name):
                refs.setdefault(name, []).append(f"{dest}'s CHECKSUMS.CURRENT as {held.get('source')} observed it at {held.get('observed_utc')}, "
                                                 f"not yet recorded: kept until a merge records {dest}'s pointer")
    return refs


#: The listing shell's markers (:func:`_remote_listing_cmd`): what the parser requires, so a failure can never read as an empty
#: side (P2: ``2>/dev/null …; true`` made a failed ``cat`` of an EXISTING pointer look absent).
_LISTING_DIR_ABSENT = "DIR-ABSENT"
_LISTING_POINTER = "POINTER"
_LISTING_POINTER_ABSENT = "POINTER-ABSENT"
_LISTING_POINTER_ERROR = "POINTER-ERROR"
_LISTING_SEP = "---"
_LISTING_FIND_OK = "FIND-OK"
_LISTING_FIND_ERROR = "FIND-ERROR"


def _remote_listing_cmd(dest: Destination) -> List[str]:
    """ONE ssh, FAIL-CLOSED: ``DIR-ABSENT`` when the destination has no ``manifests/`` yet (an empty side); else a first line
    ``POINTER <name>`` (the pointer's content, its newline stripped by the shell), ``POINTER-ABSENT`` (no pointer file) or
    ``POINTER-ERROR`` (a pointer file that EXISTS but could not be read -- never mistaken for an absent one); a ``---`` line;
    ``<bytes> <name>`` per manifest from ``find -maxdepth 1 -printf``; and a last line ``FIND-OK`` or ``FIND-ERROR``. Nothing is
    sent to ``/dev/null``: a failing ``cat`` or ``find`` reports on stderr, which the error carries. No ``test -f``, so it is not
    the pointer probe of :func:`_remote_base`."""
    m = f"{dest.path.rstrip('/')}/{MANIFEST_DIR}"
    return _ssh(dest, (f"m='{m}'; if [ ! -d \"$m\" ]; then echo {_LISTING_DIR_ABSENT}; exit 0; fi; "
                       f"if [ -e \"$m/{CURRENT_NAME}\" ]; then p=$(cat \"$m/{CURRENT_NAME}\") && printf '{_LISTING_POINTER} %s\\n' \"$p\" "
                       f"|| echo {_LISTING_POINTER_ERROR}; else echo {_LISTING_POINTER_ABSENT}; fi; echo {_LISTING_SEP}; "
                       f"find \"$m\" -maxdepth 1 -type f -name 'CHECKSUMS-*.sha256' -printf '%s %f\\n' && echo {_LISTING_FIND_OK} "
                       f"|| echo {_LISTING_FIND_ERROR}"))


def parse_remote_listing(text: str) -> Tuple[Optional[str], Dict[str, int]]:
    """``(pointer, {name: bytes})`` from :func:`_remote_listing_cmd`'s stdout, STRICT: ``DIR-ABSENT`` alone is an empty side;
    otherwise the pointer line, the ``---`` line and the closing ``FIND-OK`` are all required, and ``POINTER-ERROR``,
    ``FIND-ERROR``, an empty or cut-short output, a pointer that is not a manifest name, or a malformed size all raise
    ``ValueError`` -- the side is then refused, never pruned as if empty. An empty pointer FILE reads as no pointer
    (``records.read_pointer``'s rule); a listed name that is not a manifest's shape is not ours and is skipped."""
    lines = [l.rstrip("\r") for l in (text or "").splitlines()]
    if not lines or not any(l.strip() for l in lines):
        raise ValueError("empty listing output: the destination's pointer and manifests could not be established")
    head = lines[0].strip()
    if head == _LISTING_DIR_ABSENT:
        return None, {}
    if head == _LISTING_POINTER_ABSENT:
        cur: Optional[str] = None
    elif head == _LISTING_POINTER_ERROR:
        raise ValueError(f"the destination's {CURRENT_NAME} exists but could not be read")
    elif head.startswith(_LISTING_POINTER + " ") or head == _LISTING_POINTER:
        name = head[len(_LISTING_POINTER):].strip()
        if not name:
            cur = None
        elif is_manifest_name(name):
            cur = name
        else:
            raise ValueError(f"the destination's {CURRENT_NAME} names {name!r}, not a manifest")
    else:
        raise ValueError(f"listing without its pointer line: {text.strip()[:200]!r}")
    if len(lines) < 2 or lines[1].strip() != _LISTING_SEP:
        raise ValueError(f"listing without its separator line: {text.strip()[:200]!r}")
    tail = lines[-1].strip()
    if tail == _LISTING_FIND_ERROR:
        raise ValueError("the destination's manifests could not be listed (find failed)")
    if tail != _LISTING_FIND_OK:
        raise ValueError(f"listing without its completion marker (cut short?): {text.strip()[-200:]!r}")
    listing: Dict[str, int] = {}
    for line in lines[2:-1]:
        if not line.strip():
            continue
        size, _, name = line.strip().partition(" ")
        if not size.isdigit():
            raise ValueError(f"malformed listing line {line.strip()[:120]!r}")
        if is_manifest_name(name):
            listing[name] = int(size)
    return cur, listing


def read_remote_listing(dest: Destination, run) -> Tuple[Optional[str], Dict[str, int]]:
    """The listing of an ssh destination through ``run``: a nonzero exit (ssh itself) or an unparsable / failed listing raises,
    with the remote stderr attached, so the caller refuses that side."""
    proc = run(_remote_listing_cmd(dest), PRUNE_LIST_TIMEOUT_S)
    rc = int(getattr(proc, "returncode", 0) or 0)
    err = str(getattr(proc, "stderr", "") or "").strip()
    if rc != 0:
        raise subprocess.SubprocessError(f"listing {dest.name}'s manifests exited {rc}: {err[:300]}")
    try:
        return parse_remote_listing(str(getattr(proc, "stdout", "") or ""))
    except ValueError as exc:
        raise ValueError(f"{exc}" + (f" :: {err[-300:]}" if err else "")) from None


def _remote_delete_cmd(dest: Destination) -> List[str]:
    """ONE ssh deleting the manifests named on stdin, one per line, inside the destination's ``manifests/`` (``cd`` first, so a
    missing directory fails loudly instead of deleting elsewhere)."""
    return _ssh(dest, f"cd {dest.path.rstrip('/')}/{MANIFEST_DIR} && xargs -r rm -f --")


def _checked_names(names: Sequence[str]) -> List[str]:
    """The names a deletion may carry: every one shaped ``CHECKSUMS-<12 digits>.sha256``; anything else refuses the whole deletion."""
    bad = [n for n in names if not is_manifest_name(n)]
    if bad:
        raise ValueError(f"manifest retention refuses to delete a name that is not a manifest: {bad[:3]}")
    return list(names)


def _apply_local(plan: RetentionPlan, directory: str) -> None:
    for name in _checked_names(plan.deleted):
        try:
            os.remove(os.path.join(directory, name))
            plan.removed += 1
        except FileNotFoundError:
            plan.removed += 1          # a concurrent pass took it
        except OSError as exc:
            plan.error = f"{name}: {exc}"
            return


def _apply_remote(plan: RetentionPlan, dest: Destination, run) -> None:
    names = _checked_names(plan.deleted)
    if not names:
        return
    proc = run(_remote_delete_cmd(dest), PRUNE_DELETE_TIMEOUT_S, input="\n".join(names) + "\n", check=False)
    rc = int(getattr(proc, "returncode", 0) or 0)
    if rc != 0:
        raise subprocess.SubprocessError(f"deleting {len(names)} manifest(s) on {dest.name} exited {rc}: "
                                         f"{str(getattr(proc, 'stderr', '') or '').strip()[:300]}")
    plan.removed = len(names)


def _late_references(root: str, live: Mapping[str, Optional[str]]) -> Dict[str, List[str]]:
    """The references as the record, the state and the holds say RIGHT NOW, ``live`` (a pass's own results) over the record: the
    last look :func:`prune_manifests` takes before a file goes, so a pointer merged since its plan -- a concurrent pass's switch, a
    verify's read -- still protects what it names."""
    known = dict(last_known_currents(root))
    known.update({d: c for d, c in live.items() if c})
    return manifest_references(root, known, unrecorded_pointers(root))


def _unreadable_refusal(unreadable: Sequence[Mapping[str, object]]) -> str:
    """The grading host plan's refusal while hold files that cannot be read or validated stand (:func:`unreadable_holds`): each by name,
    with what is wrong with it and how it is released."""
    parts: List[str] = []
    for u in unreadable:
        dest = u.get("destination")
        parts.append(f"{os.path.basename(str(u['path']))} ({u.get('problem')}; "
                     + (f"named for {dest} by its file name, released once a sync, verify or prune listing records {dest}'s pointer"
                        if dest else "its name does not decode to one exact destination, so no merge releases it: an operator reads it and "
                                     "removes it by hand")
                     + ")")
    return (f"{len(unreadable)} hold file(s) under {os.path.dirname(str(unreadable[0]['path']))} cannot be read or validated: "
            + ", ".join(parts) + "; a hold is a destination's pointer whose merge into the record failed, and one that cannot be read "
            "may name the base the grading host would otherwise delete, so nothing is deleted on the grading host while any stands")


def prune_manifests(root: str, dests: Sequence[Destination], keep: Optional[int] = None, dry_run: bool = False, runner=None,
                    currents: Optional[Mapping[str, Optional[str]]] = None, destinations_when_idle: bool = True,
                    configured: Optional[Sequence[str]] = None, now: Optional[str] = None) -> Dict[str, object]:
    """The retention rule applied to the grading host's ``manifests/`` and to each of ``dests``' -- the ONE function behind the pass
    (:func:`sync_all`, after every destination's sync) and ``registry mirrors prune-manifests``. The destinations are listed first
    (a local one by ``scandir``, an ssh one by ONE fail-closed ssh that also returns its pointer), so the references are the live
    pointers: ``currents`` (the pass's results) win, then a destination's own listing, then the pointer RECORD and the state file
    (:func:`last_known_currents`, which also protects a destination not in ``dests``); a pointer a listing establishes is recorded
    (:func:`record_pointer`; a listing without a pointer file as a definite absence; not on a dry run). A destination whose listing
    failed is refused (an empty plan with the error). Then one plan per side (:func:`plan_manifest_retention`), applied unless
    ``dry_run``: the grading host's deletions by ``os.remove``, a destination's by one ssh with the names on stdin. A destination's listing
    or deletion failing is that side's ``error``; nothing here raises for a destination, and a dry run writes nothing anywhere.
    ``destinations_when_idle`` False (the pass) skips the destinations when the grading host plan deletes nothing -- their copies are the
    grading host's, in lockstep, and the steady state deletes one per pass -- so a pass over a young registry adds no ssh; the CLI lists
    them regardless, so a one-off catch-up reaches names the grading host no longer holds.

    FAIL CLOSED for the grading host: EVERY destination is accounted for before its ``manifests/`` loses a file -- the ones listed
    here, the ones ``currents`` names (a pass's destinations, the unreached among them) and every configured one (``configured``;
    default the root's config, so a filtered ``--dest`` run accounts for the destinations it omits). A destination whose pointer is
    unknown from every source -- neither this run (a name, or a definite absence from its listing) nor the record nor a state row
    names it -- may name a manifest the grading host would otherwise delete, so when any is unknown and its plan would delete, the
    plan is refused (its error names each with why: the listing failed, the pass did not reach it, the run did not list it)
    and nothing is deleted there until a reachable sync or listing establishes it; the destinations whose own listing succeeded
    are still pruned by the rule: before this, only a listed destination was checked, so one the
    pass could not reach, with no record yet and an outage row in the state, held nothing.

    Every HELD observation (:func:`unrecorded_pointers`: a destination's pointer as an observer read or switched it, whose merge
    into the record failed -- a file each, so two observers of one destination both stand) is a reference besides the record's name
    for that destination -- the record may be the OLDER pointer, and pruning against it alone deleted the destination's actual base
    -- and a destination nothing else names is not unknown while a hold names its pointer; the
    summary reports each under ``unrecorded`` (per destination, freshest first) and the text a ``[pointers] HELD`` line, until a
    merge covers it.

    A hold file that cannot be READ or validated (:func:`unreadable_holds`: zero bytes, not JSON, no destination, not a manifest
    name -- a storage error, or a crash that tore it) is never taken for no hold: it may be the destination's fresher pointer, the
    record's name then the OLDER one, and the retention trusting the record alone deleted the destination's actual base (re-check of
    360d61f4b, MEDIUM: a switch to B held, the hold torn, the record's A trusted, B gone once it left the newest K; a whole-tree copy
    on recovery). So while any stands and the grading host plan would delete, the grading host plan is REFUSED -- its error names each file, what
    is wrong with it and the destination its name says, with how it is released: a merge that records that destination's pointer
    (:func:`release_unrecorded`); a file whose name says no destination is an operator's to read and remove -- and the destinations
    whose own listing succeeded are still pruned by the rule; the summary reports each under ``unreadable`` and the text as a
    ``[pointers] UNREADABLE`` line. The last look before a deletion reads the holds once more: one unreadable since the plan refuses
    too, and one released since lifts no refusal here (the next pass does).

    A listing is an OBSERVATION stamped as it BEGINS (``now`` when the caller fixed the run's time, else the clock), and its merge
    is freshness-aware (:func:`record_pointer`): one that read the pointer while a pass was switching it, and merged after the pass
    recorded the new name, is refused -- before this the lock serialized the writes, not the observations, and the stale write put
    the old name back over the new one. So the references are taken from the record as it
    stands AFTER the listings' merges (``currents`` over it, then what this run's listings established that the record does not
    carry -- a dry run records nothing, a merge may have failed -- and the holds); a listing the record moved past meanwhile is
    reported under ``stale`` (and a ``[pointers] STALE`` line) with the record's name standing. And right before any file goes,
    the record and the holds are read ONCE MORE (:func:`_late_references`): a pointer merged since the plan is added to the
    references and the plans rebuilt -- a reference is only ever added there, never removed. Returns ``{utc, keep, dry_run,
    references, unrecorded, unreadable, stale, sides: [plan dicts, the grading host first]}``."""
    run = runner or _default_runner
    k = MANIFEST_KEEP if keep is None else int(keep)
    if k < 1:
        raise ValueError(f"manifest retention: keep must be at least 1 (got {keep})")
    known: Dict[str, Optional[str]] = dict(last_known_currents(root))     # established: a name, or None = observed absent
    held = unrecorded_pointers(root)                                       # every held observation: observed, not recorded, a reference each
    live = {d: c for d, c in (currents or {}).items() if c}
    known.update(live)
    accounted = ({d.name for d in dests} | set(currents or {})
                 | set(configured if configured is not None else (d.name for d in destinations(root))))
    grading_host_dir = os.path.join(root, MANIFEST_DIR)
    grading_host_listing = list_manifests(grading_host_dir)
    provisional = plan_manifest_retention(grading_host_listing, current_manifest(root), manifest_references(root, known, held), k)
    skip = not destinations_when_idle and not provisional.deleted
    listed: Dict[str, Tuple[Optional[str], Dict[str, int], Optional[str]]] = {}
    began: Dict[str, str] = {}                         # when each listing began: the instant its observation is stamped with
    for d in dests:
        if skip:
            break
        seen = now or R.utc_now()                      # BEFORE the read: a pointer switched while it runs is the fresher observation
        try:
            if d.kind == "local":
                cur, lst = current_manifest(d.path), list_manifests(os.path.join(d.path, MANIFEST_DIR))
            elif d.kind == "ssh":
                cur, lst = read_remote_listing(d, run)
            else:
                raise ValueError(f"unknown destination kind {d.kind!r} (local | ssh)")
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            listed[d.name] = (None, {}, f"listing: {str(exc)[-400:]}")
            continue
        listed[d.name] = (cur, lst, None)
        began[d.name] = seen
        if not dry_run:
            try:
                record_pointer(root, d.name, cur, "listing", now=seen, absent=True)
            except OSError as exc:                     # the record is a cache of what this run read; the next observer retries
                _say(f"[pointers] ERROR recording {d.name}'s pointer: {exc}")
    # the references, from the record as it stands after the listings' merges (a stale listing was refused there, a fresher one
    # merged, a failed one held), the pass's results over it, what this run established that nothing recorded, and the holds
    recs = read_pointers(root)
    known = dict(last_known_currents(root))
    known.update(live)
    held = unrecorded_pointers(root)
    unreadable = unreadable_holds(root)        # a hold the retention cannot read: the grading host's deletions are refused while it stands
    stale: Dict[str, Dict[str, object]] = {}
    for dname, (cur, _lst, err) in listed.items():
        if err or dname in live:
            continue
        rec = recs.get(dname) or {}
        if cur and rec.get("current") and rec["current"] != cur and (_utc_order(rec.get("observed_utc"), began[dname]) or 0) > 0:
            # the record moved past this listing while it ran (a switch merged with a later stamp): the record's name stands
            stale[dname] = {"listed": cur, "observed_utc": began[dname], "recorded": rec["current"],
                            "recorded_utc": rec.get("observed_utc"), "source": rec.get("source")}
            continue
        if cur:
            known[dname] = cur
        elif dname not in known:
            known[dname] = None                        # established: the listing saw no pointer, nothing to keep for it
    refs = manifest_references(root, known, held)
    unknown = sorted(name for name in accounted if name not in known and name not in held)     # a held name establishes too

    def build(refs_: Mapping[str, Sequence[str]], unreadable_: Sequence[Mapping[str, object]]) -> List[RetentionPlan]:
        plans_ = [plan_manifest_retention(grading_host_listing, current_manifest(root), refs_, k, side="grading_host", location=grading_host_dir)]
        if plans_[0].deleted and (unknown or unreadable_):
            reasons: List[str] = []
            if unknown:
                def why(name: str) -> str:
                    if name in listed:
                        return "its listing failed"
                    if name in (currents or {}):
                        return "this pass did not reach it"
                    return "this run did not list it"
                reasons.append(f"the {CURRENT_NAME} of " + ", ".join(f"{n} ({why(n)})" for n in unknown) + " is unknown from "
                               f"every source (no sync, verify or listing ever recorded it, and the state file names none); nothing "
                               f"deleted on the grading host until a reachable sync or `registry mirrors prune-manifests` establishes it")
            if unreadable_:
                reasons.append(_unreadable_refusal(unreadable_))
            plans_[0].error = "refused: " + "; and ".join(reasons)
        for d in dests:
            if d.name not in listed:
                continue
            cur, lst, err = listed[d.name]
            loc = f"{d.host}:{d.path.rstrip('/')}/{MANIFEST_DIR}" if d.kind == "ssh" else os.path.join(d.path, MANIFEST_DIR)
            p = plan_manifest_retention(lst, cur, refs_, k, side=d.name, location=loc)
            p.error = err
            plans_.append(p)
        for p in plans_:
            p.dry_run = bool(dry_run)
        return plans_

    plans = build(refs, unreadable)
    if not dry_run:
        # the last look before a file goes: a pointer merged since the plan (a concurrent pass's switch, a verify's read) may name a
        # manifest a plan deletes, so the record and the holds are read again and every new reference is added -- none removed; a
        # hold unreadable since the plan refuses the grading host too, and one released since lifts no refusal here (the next pass does)
        late = _late_references(root, live)
        added = {n: list(r) for n, r in late.items() if n not in refs and any(n in p.deleted for p in plans)}
        seen = {str(u["path"]) for u in unreadable}
        late_unreadable = [u for u in unreadable_holds(root) if str(u["path"]) not in seen]
        if added or late_unreadable:
            refs.update(added)
            unreadable = list(unreadable) + late_unreadable
            plans = build(refs, unreadable)
        if not plans[0].error:
            _apply_local(plans[0], grading_host_dir)
        by_name = {d.name: d for d in dests}
        for p in plans[1:]:
            if p.error:
                continue
            d = by_name[p.side]
            try:
                if d.kind == "local":
                    _apply_local(p, os.path.join(d.path, MANIFEST_DIR))
                else:
                    _apply_remote(p, d, run)
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                p.error = f"deletion: {str(exc)[-400:]}"
    return {"utc": R.utc_now(), "keep": k, "dry_run": bool(dry_run), "references": refs,
            "unrecorded": {name: [dict(h) for h in hs] for name, hs in sorted(held.items())},
            "unreadable": [dict(u) for u in unreadable], "stale": stale,
            "sides": [p.as_dict() for p in plans]}


def _fmt_bytes(n: int) -> str:
    n = int(n or 0)
    if n >= 10**9:
        return f"{n / 10**9:.1f} GB"
    if n >= 10**6:
        return f"{n / 10**6:.1f} MB"
    return f"{n} B"


def format_retention(summary: Mapping[str, object], header: bool = True) -> str:
    """The plan as text: a header with the rule and the referenced manifests with their reasons (``header``), then one line per
    side -- its count and bytes, what is kept (count, bytes, oldest, newest, how many by reference), what is deleted (count,
    bytes), and ``dry run`` / ``removed N`` / ``ERROR``. Between them one ``[pointers] HELD`` line per observation whose merge
    into the record failed (``unrecorded``), one ``[pointers] UNREADABLE`` line per hold file that cannot be read or validated
    (``unreadable``: the grading host's deletions are refused while it stands) and one ``[pointers] STALE`` line per listing the record
    moved past while it ran (``stale``), with or without the header. :func:`sync_all` logs the HELD, UNREADABLE, STALE and side
    lines; the CLI prints the whole."""
    lines: List[str] = []
    refs = summary.get("references") or {}
    if header:
        lines.append(f"manifest retention: keep the newest {summary.get('keep')} by ingest_seq, each side's {CURRENT_NAME} and every "
                     f"manifest a pointer names; " + ("DRY RUN: nothing deleted" if summary.get("dry_run") else "applied"))
        if refs:
            lines.append("referenced (kept wherever present):")
            for name in sorted(refs):
                lines.append(f"  {name}: " + "; ".join(refs[name]))
        else:
            lines.append("referenced: none (no pointer names a manifest)")
    unrecorded = summary.get("unrecorded") or {}
    for dest in sorted(unrecorded):
        for h in (unrecorded[dest] or []):
            lines.append(f"[pointers] HELD {dest}'s {CURRENT_NAME} {h.get('current')} (observed by {h.get('source')} at {h.get('observed_utc')}, "
                         f"not recorded: {h.get('error')}); kept on every side until a sync, verify or prune listing records {dest}'s pointer")
    for u in summary.get("unreadable") or []:
        dest = u.get("destination")
        lines.append(f"[pointers] UNREADABLE hold {u.get('path')}: {u.get('problem')}; "
                     + (f"named for {dest} by its file name: the grading host deletes nothing until a sync, verify or prune listing records "
                        f"{dest}'s pointer, which releases it" if dest
                        else "its name does not decode to one exact destination, so no merge releases it: the grading host deletes nothing until "
                             "an operator reads it and removes it by hand"))
    stale = summary.get("stale") or {}
    for dest in sorted(stale):
        s = stale[dest] or {}
        lines.append(f"[pointers] STALE {dest}'s {CURRENT_NAME} read {s.get('listed')} by a listing begun at {s.get('observed_utc')}; "
                     f"{s.get('source')} recorded {s.get('recorded')} observed at {s.get('recorded_utc')} meanwhile, which stands")
    for s in summary.get("sides") or []:
        total = int(s.get("manifests") or 0)
        lines.append(f"[retention] {s.get('side')} {s.get('location')}: {total} manifest(s) "
                     f"({_fmt_bytes(int(s.get('kept_bytes') or 0) + int(s.get('deleted_bytes') or 0))}); "
                     f"keep {s.get('kept')} ({_fmt_bytes(s.get('kept_bytes') or 0)}; oldest {s.get('oldest_kept') or '-'}, "
                     f"newest {s.get('newest_kept') or '-'}; {len(s.get('referenced') or {})} by reference), "
                     f"delete {s.get('deleted')} ({_fmt_bytes(s.get('deleted_bytes') or 0)})"
                     + (": dry run" if s.get("dry_run") else f": removed {s.get('removed')}")
                     + (f"; ERROR {s.get('error')}" if s.get("error") else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# the one verifier
# --------------------------------------------------------------------------------------------


def _resolve(root: str, rel: str) -> List[str]:
    out = []
    for cand in (os.path.join(root, rel), os.path.join(root, rel + ".gz")):
        if os.path.exists(cand):
            out.append(cand)
    return out


def walk_present(root: str) -> set:
    """Root-relative paths of every record file under ``records/`` and ``rulings/`` from ONE names-only directory walk
    (``os.scandir`` ``d_type``, no per-entry ``stat``): the presence check of the incremental verify. Measured 2026-09-28:
    412,000 files in 21.5 s on cluster A's login node (VAST/NFS; a ``stat`` per entry there ran at 2,800/s) and 12.4 s on
    cluster B's (MooseFS)."""
    present = set()
    for sub in LOG_SUBDIRS:
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        stack = [base]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.name.endswith(".jsonl") or e.name.endswith(".jsonl.gz"):
                            present.add(os.path.relpath(e.path, root))
            except FileNotFoundError:
                continue
    return present


def _candidates(rel: str, present: set) -> List[str]:
    """The on-disk root-relative paths (plain first, then ``.gz``) behind a logical entry, from the walk's set."""
    return [c for c in (rel, rel + ".gz") if c in present]


def _hash_entry(args: Tuple[str, str, str, List[str]]) -> Tuple[str, List[str], int]:
    """Pool worker: ``(root, rel, sha, candidates)`` -> ``(rel, [candidates whose logical sha matches], bytes of the last)``."""
    root, rel, sha, cands = args
    good: List[str] = []
    for c in cands:
        try:
            if R.file_logical_sha(os.path.join(root, c)) == sha:
                good.append(c)
        except (OSError, EOFError):
            continue
    size = 0
    if good:
        try:
            size = os.path.getsize(os.path.join(root, good[-1]))
        except OSError:
            size = 0
    return rel, good, size


def choose_sample(rels: Sequence[str], k: int, seed: Optional[int] = None) -> Tuple[List[str], int]:
    """``min(k, len(rels))`` of ``rels`` drawn without replacement by ``random.Random(seed)``; a None seed draws one from
    ``os.urandom`` so successive cycles cover different files. Returns (chosen, the seed used) — the seed is reported so
    a run can be repeated. Pure given the seed; the tests pin it."""
    seed = int.from_bytes(os.urandom(4), "big") if seed is None else int(seed)
    k = max(0, int(k))
    rels = list(rels)
    if k >= len(rels):
        return rels, seed
    return random.Random(seed).sample(rels, k), seed


def _hash_many(root: str, todo: Sequence[Tuple[str, str, List[str]]], workers: int):
    """Yield ``_hash_entry`` results for ``[(rel, sha, cands)]``: in-process when ``workers`` <= 1 (the login node path),
    else a process pool — the verify is I/O- and sha256-bound and a compute node has the cores."""
    jobs = [(root, rel, sha, cands) for rel, sha, cands in todo]
    if int(workers) <= 1 or len(jobs) < 2:
        for j in jobs:
            yield _hash_entry(j)
        return
    from concurrent.futures import ProcessPoolExecutor
    chunk = max(1, min(500, len(jobs) // (int(workers) * 8) or 1))
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        for r in ex.map(_hash_entry, jobs, chunksize=chunk):
            yield r


def verify_manifest(root: str, manifest_path: str, prune_plain: bool = True, since: Optional[str] = None,
                    sample: int = 0, seed: Optional[int] = None, workers: int = 1) -> VerifyReport:
    """Verify every entry of ``manifest_path`` under ``root``. Presence comes from ONE directory walk (:func:`walk_present`).
    FULL mode (``since`` None): the DECOMPRESSED content of every present entry is hashed (``workers`` > 1: a process pool).
    INCREMENTAL mode (``since`` = the base manifest the destination last verified clean): hashed are the entries the base
    does not carry with the same sha, every carried entry with BOTH a plain and a ``.gz`` copy (gzipped after the base was
    acked), and a random ``sample`` of the carried rest (:func:`choose_sample`, the seed reported); every other carried entry
    counts as ``presence_only``. A plain copy beside a hashed-clean gz copy is pruned (a later manifest names only the gz);
    presence never prunes. A ``since`` file that does not exist falls back to full mode with a ``note``."""
    t0 = time.monotonic()
    rep = VerifyReport(os.path.basename(manifest_path))
    base: Dict[str, str] = {}
    if since:
        if os.path.exists(since):
            base = {rel: sha for sha, rel in read_manifest(since)}
            rep.mode, rep.since = "incremental", os.path.basename(since)
        else:
            rep.note = f"base manifest {since} is not on the destination; every entry hashed"
    present = walk_present(root)
    todo: List[Tuple[str, str, List[str]]] = []
    carried_single: List[Tuple[str, str, List[str]]] = []
    for sha, rel in read_manifest(manifest_path):
        cands = _candidates(rel, present)
        if not cands:
            rep.missing.append(rel)
            continue
        if rep.mode == "incremental" and base.get(rel) == sha and len(cands) == 1:
            carried_single.append((rel, sha, cands))
        else:
            todo.append((rel, sha, cands))
    sampled_rels: set = set()
    if rep.mode == "incremental":
        chosen, rep.sample_seed = choose_sample([rel for rel, _, _ in carried_single], sample, seed)
        sampled_rels = set(chosen)
        rep.sampled = len(sampled_rels)
        todo += [t for t in carried_single if t[0] in sampled_rels]
        presence = [t for t in carried_single if t[0] not in sampled_rels]
        rep.presence_only = len(presence)
        rep.ok += len(presence)
    else:
        rep.presence_only = 0
    rep.hashed = 0
    for rel, good, size in _hash_many(root, todo, workers):
        rep.hashed += 1
        if not good:
            rep.corrupt.append(rel)
            continue
        rep.ok += 1
        rep.bytes += size
        if prune_plain and len(good) == 2:
            for c in good:
                if not c.endswith(".gz"):
                    try:
                        os.remove(os.path.join(root, c))
                        rep.pruned_plain += 1
                    except FileNotFoundError:
                        pass
    rep.seconds = time.monotonic() - t0
    return rep


# --------------------------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------------------------


@dataclass
class SyncResult:
    destination: str
    manifest: str
    copied: int
    verify: VerifyReport
    switched: bool
    error: Optional[str] = None
    chunks_done: int = 0          # ssh: --files-from chunks rsync'd
    chunks_total: int = 0         # ssh: chunks the copy delta was split into
    stopped: Optional[str] = None  # ssh: "budget" when the run stopped starting chunks; the next run resumes
    seconds: float = 0.0
    hashed: Optional[int] = None         # entries the verify hashed (ssh: the delta + twins + sample)
    presence_only: Optional[int] = None  # entries confirmed by presence alone (ssh: carried from the base)
    mode: Optional[str] = None           # ssh: "incremental" | "full"; local: "full"
    since: Optional[str] = None          # ssh incremental: the base manifest (the destination's CHECKSUMS.CURRENT before this sync)
    sampled: Optional[int] = None        # ssh incremental: carried entries re-hashed as the random sample
    slurm: bool = False                  # ssh: the verify ran under srun
    note: Optional[str] = None           # ssh: a verifier older than the incremental flags, or a base fallback
    vanished: List[str] = field(default_factory=list)   # files listed for the manifest and gone before hashing (no .gz twin)
    copy_mode: Optional[str] = None      # ssh: "incremental" (the delta since copy_since, --files-from) | "full" (the whole tree); local: "full"
    copy_since: Optional[str] = None     # ssh incremental: the destination's acked manifest the delta was computed from
    copy_files: Optional[int] = None     # ssh incremental: on-disk paths listed for rsync (the delta); None for a full copy
    dest_current: Optional[str] = None   # the destination's CHECKSUMS.CURRENT after this sync (the new manifest when switched, else the
                                         # base it read; None when unknown): what the manifest retention rule must keep for it
    observed_utc: Optional[str] = None   # when dest_current was observed: the switch as it completed, else the base read as it BEGAN
                                         # (a read is stamped as it starts, so a pointer switched meanwhile is the fresher observation)


def _copy_local(src_root: str, dest_root: str) -> int:
    """Ignore-existing copy of records/ and rulings/ (resumable: a partial file is re-copied when its
    logical sha differs)."""
    n = 0
    for rel, path in log_files(src_root):
        dest = os.path.join(dest_root, os.path.relpath(path, src_root))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            try:
                if R.file_logical_sha(dest) == R.file_logical_sha(path):
                    continue
            except (OSError, EOFError):
                pass
        try:
            shutil.copy2(path, dest + ".part")
        except FileNotFoundError:
            continue        # gzipped by the harvester meanwhile; the .gz rides the next sync
        os.replace(dest + ".part", dest)
        n += 1
    return n


SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
RSYNC_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=15"
#: Partial transfers are staged under this directory ON THE DESTINATION, never at a final name: an interrupted
#: copy leaves nothing `--ignore-existing` would skip forever, and rsync promotes a finished file atomically.
PARTIAL_DIR = ".partial"
#: The ssh copy lists the delta (:func:`added_since`) for ``rsync --files-from=-`` in chunks of this many on-disk paths — one rsync
#: call, one journal line each, so a cycle's cost is O(new files) and a killed run loses at most a chunk's partial files (staged under
#: `PARTIAL_DIR`, never at a final name). 461,769-file whole-tree passes did not finish in an hour on 2026-09-28; a 5-min cycle's delta
#: is a few hundred files, and a day's backlog ~6,000 — one chunk.
COPY_CHUNK_FILES = 20000
#: One rsync call's wall-clock bound (a chunk of the incremental copy, or one subtree of the full copy): the pre-existing 3,600 s.
COPY_TIMEOUT_S = 3600
#: rsync exits the copy tolerates: 23 (partial transfer: a listed file the harvester gzipped between the listing and the transfer)
#: and 24 (a source file vanished during the walk). The verify decides what arrived; anything else is the destination's error.
RSYNC_PARTIAL_EXITS = (23, 24)
_RSYNC_TRANSFERRED = re.compile(r"^Number of (?:regular )?files transferred: (\d[\d,]*)", re.M)


def _rsync_cmd(src: str, dest: str) -> List[str]:
    """The whole-tree copy of one subtree (records/ or rulings/): recursive, ignore-existing, partials staged, ``--stats`` so the
    transferred count can be read back. Runs only for a destination with no acked manifest and under ``--full`` (O(total files))."""
    return ["rsync", "-rt", f"--partial-dir={PARTIAL_DIR}", "--ignore-existing", "--timeout=120", "-e", RSYNC_SSH, "--stats", src, dest]


def _rsync_files_cmd(src_root: str, dest_root: str) -> List[str]:
    """The incremental copy: the same never-overwrite rsync as :func:`_rsync_cmd` (``-rt --partial-dir --ignore-existing``) over
    exactly the root-relative paths on stdin (``--files-from=-``, which implies ``--relative``, so ``records/x/f.jsonl`` under
    ``src_root`` lands at the same path under ``dest_root``, directories created as needed). O(listed files)."""
    return ["rsync", "-rt", f"--partial-dir={PARTIAL_DIR}", "--ignore-existing", "--timeout=120", "-e", RSYNC_SSH, "--stats",
            "--files-from=-", src_root, dest_root]


def _rsync_transferred(stdout: str) -> Optional[int]:
    """The ``Number of (regular) files transferred`` line of an rsync ``--stats`` report, or None when there is none."""
    m = _RSYNC_TRANSFERRED.search(stdout or "")
    return int(m.group(1).replace(",", "")) if m else None


def added_since(manifest_path: str, base_path: str) -> List[str]:
    """The logical relpaths ``manifest_path`` names that ``base_path`` (the destination's acked manifest) does not carry with the same
    sha, in manifest order: ONE definition of the delta, what the incremental copy lists and what the incremental verify hashes."""
    base = {rel: sha for sha, rel in read_manifest(base_path)}
    return [rel for sha, rel in read_manifest(manifest_path) if base.get(rel) != sha]


def _repair_cmd(src_root: str, dest_root: str, relpaths: Sequence[str]) -> List[str]:
    """Re-copy the named files (root-relative) WITHOUT ignore-existing and WITH ``--checksum`` (a truncated or damaged
    file whose size and mtime happen to match is still replaced): the verifier found them corrupt on the destination;
    the list rides stdin (``--files-from=-``)."""
    return ["rsync", "-rt", "--checksum", f"--partial-dir={PARTIAL_DIR}", "--timeout=600", "-e", RSYNC_SSH, "--files-from=-",
            src_root, dest_root]


def _default_runner(argv: Sequence[str], timeout_s: float, input: Optional[str] = None,
                    check: bool = True) -> subprocess.CompletedProcess:
    """The one subprocess shape behind every ``ssh`` / ``rsync`` call; tests inject a recorder with this signature."""
    return subprocess.run(list(argv), check=check, capture_output=True, text=True, timeout=timeout_s, input=input)


def _run_copy(run, cmd: Sequence[str], timeout_s: float, label: str, input: Optional[str] = None) -> Optional[int]:
    """One copy rsync: exit 0 or a :data:`RSYNC_PARTIAL_EXITS` code (a journal line) is a result whose transferred count (``--stats``)
    is returned; any other exit raises :class:`subprocess.CalledProcessError` (the destination's error, as before)."""
    proc = run(list(cmd), timeout_s, input=input, check=False)
    rc = int(getattr(proc, "returncode", 0) or 0)
    if rc in RSYNC_PARTIAL_EXITS:
        _say(f"{label}: rsync exit {rc} (partial transfer; the verify decides what arrived): "
             + str(getattr(proc, "stderr", "") or "").strip()[-300:])
    elif rc != 0:
        raise subprocess.CalledProcessError(rc, list(cmd), getattr(proc, "stdout", ""), getattr(proc, "stderr", ""))
    return _rsync_transferred(getattr(proc, "stdout", "") or "")


def _rsync_list(run, cmd: Sequence[str], rels: Sequence[str], timeout_s: float, label: str, deadline: Optional[float] = None,
                chunk_files: Optional[int] = None) -> Tuple[int, int, Optional[int], Optional[str]]:
    """``cmd`` (a ``--files-from=-`` rsync) once per chunk of ``chunk_files`` paths of ``rels`` — stdin, one per line, in the given
    order — with one journal line per chunk. Returns ``(chunks_done, chunks_total, transferred, stopped)``:
    ``transferred`` is the sum of rsync's ``--stats`` counts (None when a call printed none: a fake runner, an older rsync);
    ``stopped`` is "budget" when ``deadline`` (monotonic seconds) is set and would not fit another chunk (:data:`BUDGET_MARGIN` × the
    last chunk's time) — the chunks done stand (a finished file is at its final name), the rest rides the next run."""
    step = max(1, int(COPY_CHUNK_FILES if chunk_files is None else chunk_files))
    chunks = [list(rels[i:i + step]) for i in range(0, len(rels), step)]
    done, total, transferred, last_s, counted = 0, len(chunks), 0, 0.0, True
    for k, chunk in enumerate(chunks, 1):
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (last_s and remaining < last_s * BUDGET_MARGIN):
                _say(f"{label}: budget: stopped after {done}/{total} chunks ({remaining:.0f} s left, last chunk {last_s:.0f} s); "
                     f"the rest rides the next run")
                return done, total, (transferred if counted else None), "budget"
        t0 = time.monotonic()
        n = _run_copy(run, cmd, timeout_s, f"{label} chunk {k}/{total}", input="\n".join(chunk) + "\n")
        if n is None:
            counted = False
        else:
            transferred += n
        last_s = time.monotonic() - t0
        done += 1
        _say(f"{label} chunk {k}/{total}: {len(chunk)} files listed, {n if n is not None else '?'} transferred, {last_s:.0f} s")
    return done, total, (transferred if counted else None), None


def _run_verifier(cmd: Sequence[str], name: str = "", runner=None) -> VerifyReport:
    """Run the remote verifier and PARSE its report: exit 0 (clean) and exit 1 (missing / corrupt files) are both
    results; any other exit, or unparsable output, is a transport / tooling error and raises."""
    proc = (runner or _default_runner)(list(cmd), 3600, check=False)
    rc = int(getattr(proc, "returncode", 0) or 0)
    if rc not in (0, 1):
        raise subprocess.SubprocessError(f"verifier exited {rc}: {str(getattr(proc, 'stderr', '') or '').strip()[:500]}")
    data = json.loads(getattr(proc, "stdout", "") or "")
    return _report_from_json(name, data)


def _on_disk_relpaths(root: str, logical_rels: Sequence[str]) -> List[str]:
    """The on-disk root-relative paths (plain and/or .gz) behind logical manifest entries."""
    out: List[str] = []
    for rel in logical_rels:
        for cand in _resolve(root, rel):
            out.append(os.path.relpath(cand, root))
    return out


def _ssh(dest: Destination, remote_cmd: str) -> List[str]:
    return ["ssh"] + SSH_OPTS + [dest.host, remote_cmd]


def _remote_python(dest: Destination) -> List[str]:
    """The stdlib interpreter on a cluster, importing from the repo mirror when one is configured."""
    return (["env", f"PYTHONPATH={dest.remote_src}"] if dest.remote_src else []) + ["python3", "-S"]


def slurm_time(n_entries: int, workers: int) -> str:
    """``--time`` for a full verify of ``n_entries`` with ``workers`` hashing processes: 2x the estimate at
    :data:`SLURM_FILES_PER_S_PER_WORKER` per worker, clamped to [:data:`SLURM_TIME_MIN_S`, :data:`SLURM_TIME_MAX_S`], as
    ``H:MM:SS``. Right-sized so backfill can place it; never the 24 h default."""
    est = 2.0 * max(0, int(n_entries)) / (SLURM_FILES_PER_S_PER_WORKER * max(1, int(workers)))
    secs = int(min(SLURM_TIME_MAX_S, max(SLURM_TIME_MIN_S, est)))
    return f"{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"


def _slurm_wrap(dest: Destination, n_entries: int) -> List[str]:
    """The ``srun`` prefix that runs the remote verifier as a Slurm step instead of on the login node: CPU-only, the
    cluster entry's partition / QoS / CPUs / memory, ``--time`` from :func:`slurm_time`, ``--quiet`` so stdout is the
    verifier's JSON alone. Never ``--export`` (crashes slurmd on this cluster), never ``--mem=0``, no GPUs."""
    return ["srun", f"--partition={dest.partition}", f"--qos={dest.qos}", f"--cpus-per-task={int(dest.cpus)}", f"--mem={dest.mem}",
            f"--time={slurm_time(n_entries, dest.cpus)}", f"--job-name=registry-verify-{dest.name}", "--quiet"]


def _verify_cmd(dest: Destination, name: str, no_prune: bool = False, since: Optional[str] = None, sample: Optional[int] = None,
                slurm: bool = False, n_entries: int = 0) -> List[str]:
    """``ssh <host> [srun …] [env PYTHONPATH=…] python3 -S -m sisteer.registry.mirrors verify <root> <manifest> --json
    [--no-prune] [--since <root>/manifests/<base> --sample N] [--workers N]`` as ONE argv (ssh joins the remote words), so
    ``verify`` and the manifest path are readable elements of it. ``since`` is the base manifest's NAME (its file sits beside
    the manifest on the destination); ``slurm`` wraps the verifier in :func:`_slurm_wrap` and hands it the step's CPUs as
    ``--workers``."""
    remote_root = dest.path.rstrip("/")
    cmd = ["ssh"] + SSH_OPTS + [dest.host]
    if slurm:
        cmd += _slurm_wrap(dest, n_entries)
    cmd += _remote_python(dest)
    cmd += ["-m", "sisteer.registry.mirrors", "verify", remote_root, f"{remote_root}/{MANIFEST_DIR}/{name}", "--json"]
    if no_prune:
        cmd.append("--no-prune")
    if since:
        cmd += ["--since", f"{remote_root}/{MANIFEST_DIR}/{since}", "--sample", str(int(VERIFY_SAMPLE if sample is None else sample))]
    if slurm:
        cmd += ["--workers", str(int(dest.cpus))]
    return cmd


def _remote_base(dest: Destination, run) -> Optional[str]:
    """The manifest the destination last verified clean — its ``CHECKSUMS.CURRENT`` — but only when that manifest's FILE is
    still beside it (the incremental verify reads it there); None otherwise (a fresh mirror, or a pointer to a pruned file),
    which means a full verify. One ssh."""
    remote_root = dest.path.rstrip("/")
    m = f"{remote_root}/{MANIFEST_DIR}"
    proc = run(_ssh(dest, f"b=$(cat {m}/{CURRENT_NAME} 2>/dev/null) && test -n \"$b\" && test -f {m}/$b && printf %s \"$b\" || true"), 120)
    out = (getattr(proc, "stdout", "") or "").strip()
    return out if out.startswith("CHECKSUMS-") and out.endswith(".sha256") else None


def _opt_int(v: object) -> Optional[int]:
    return None if v is None else int(v)


def _report_from_json(name: str, data: Mapping[str, object]) -> VerifyReport:
    """A verifier's JSON as a report. A report WITHOUT ``mode`` came from a verifier older than the incremental flags (it
    ignores unknown argv and hashes everything): ``mode`` is then "full" and ``hashed`` / ``presence_only`` stay None."""
    return VerifyReport(str(data.get("manifest") or name), int(data.get("ok", 0)), list(data.get("missing", [])),
                        list(data.get("corrupt", [])), int(data.get("pruned_plain", 0)), int(data.get("bytes", 0)),
                        mode=str(data.get("mode") or "full"), since=(str(data["since"]) if data.get("since") else None),
                        hashed=_opt_int(data.get("hashed")), presence_only=_opt_int(data.get("presence_only")),
                        sampled=_opt_int(data.get("sampled")), sample_seed=_opt_int(data.get("sample_seed")),
                        seconds=float(data.get("seconds") or 0.0), note=(str(data["note"]) if data.get("note") else None))


def _say(msg: str) -> None:
    print(msg, flush=True)


def sync(root: str, dest: Destination, ingest_seq: int, manifest_name: Optional[str] = None, runner=None,
         deadline: Optional[float] = None, full: bool = False,
         manifest: Optional[ManifestWrite] = None) -> SyncResult:
    """Copy records (ignore-existing), copy the manifest, VERIFY on the destination, then switch
    ``CHECKSUMS.CURRENT`` there. Never switches on a failed verification. ``runner`` (tests) replaces the
    subprocess call for ``ssh`` destinations. ``deadline`` (``time.monotonic()`` seconds) bounds the ssh copy's chunks;
    the local kind ignores it. A kind this module has no transport for (the removed ``modal`` among them) raises
    ``ValueError`` by name rather than being skipped (fail-closed).

    ssh: ONE ssh reads the destination's own ``CHECKSUMS.CURRENT`` (:func:`_remote_base`), the BASE of both the copy and the
    verify. The copy is INCREMENTAL — the on-disk files behind :func:`added_since` (the manifest's entries the base does not
    carry), listed for ``rsync --files-from=-`` in :data:`COPY_CHUNK_FILES` chunks, ``deadline`` honored between chunks — and
    the verify is INCREMENTAL against the same base: the delta, the gz twins and a random sample hashed, the rest confirmed
    by presence; ``missing`` and ``corrupt`` entries are re-copied and verified again. Unless ``full`` is asked, the destination
    has no usable base, or the base's file is gone from the grading host (copy only): then the whole tree is rsync'd (O(total files))
    and, in the first two cases, every entry is hashed, under ``srun`` when ``dest.slurm``. A report without
    ``mode`` came from a remote verifier older than the flags (the repo mirror lags): it hashed everything, which is
    a valid verify, and the result carries a ``note`` saying so. ``manifest`` (:class:`ManifestWrite`) is the run's
    manifest when the caller wrote it; ``manifest_name`` alone still works."""
    if manifest is None and manifest_name is None:
        manifest = write_manifest_report(root, ingest_seq)
    name = manifest_name or (manifest.name if manifest else None)
    assert name
    vanished = list(manifest.vanished) if manifest else []
    mpath = os.path.join(root, MANIFEST_DIR, name)
    if dest.kind == "local":
        os.makedirs(dest.path, exist_ok=True)
        copied = _copy_local(root, dest.path)
        os.makedirs(os.path.join(dest.path, MANIFEST_DIR), exist_ok=True)
        shutil.copy2(mpath, os.path.join(dest.path, MANIFEST_DIR, name))
        rep = verify_manifest(dest.path, os.path.join(dest.path, MANIFEST_DIR, name))
        switched = False
        if rep.clean:
            switch_current(dest.path, name)
            switched = True
        seen = R.utc_now()                 # after the switch, before the read of the pointer: the observation's instant
        return SyncResult(dest.name, name, copied, rep, switched, seconds=rep.seconds, hashed=rep.hashed,
                          presence_only=rep.presence_only, mode=rep.mode, vanished=vanished, copy_mode="full",
                          dest_current=current_manifest(dest.path), observed_utc=seen)
    if dest.kind == "ssh":
        t_start = time.monotonic()
        run = runner or _default_runner
        remote_root = dest.path.rstrip("/")
        target = f"{dest.host}:{remote_root}"
        src_root = root.rstrip("/") + "/"
        notes: List[str] = []
        copy_mode, copy_since, copy_files, copied, done, total = "full", None, None, -1, 0, 0
        base: Optional[str] = None     # the destination's pointer as read below; the retention rule keeps it until a switch
        base_seen: Optional[str] = None    # when the base read BEGAN (the pointer's observation, unless this sync switches it)
        switch_seen: Optional[str] = None  # when the switch completed

        def result(rep: VerifyReport, switched: bool, **kw) -> SyncResult:
            return SyncResult(dest.name, name, copied, rep, switched, seconds=time.monotonic() - t_start, vanished=vanished,
                              chunks_done=done, chunks_total=total, copy_mode=copy_mode, copy_since=copy_since, copy_files=copy_files,
                              dest_current=(name if switched else base), observed_utc=(switch_seen if switched else (base_seen if base else None)),
                              **kw)

        try:
            run(_ssh(dest, f"mkdir -p {remote_root}/{MANIFEST_DIR} " + " ".join(f"{remote_root}/{sub}" for sub in LOG_SUBDIRS)), 120)
            # ONE ssh reads the destination's acked manifest: the base of the copy's delta AND of the incremental verify
            base_seen = R.utc_now()
            base = None if full else _remote_base(dest, run)
            base_local = os.path.join(root, MANIFEST_DIR, base) if base else None
            if base_local and not os.path.exists(base_local):
                notes.append(f"the destination's acked manifest {base} is not under {os.path.join(root, MANIFEST_DIR)}; the whole "
                             f"tree was rsync'd")
                base_local = None
            if base_local:
                copy_mode, copy_since = "incremental", base
                rels = _on_disk_relpaths(root, added_since(mpath, base_local))
                copy_files = len(rels)
                done, total, transferred, stopped = _rsync_list(run, _rsync_files_cmd(src_root, f"{target}/"), rels, COPY_TIMEOUT_S,
                                                                f"[{dest.name}] copy", deadline)
                copied = -1 if transferred is None else transferred
                _say(f"[{dest.name}] copy incremental since {base}: {copy_files} files listed in {total} chunk(s), "
                     f"{done} chunk(s) rsync'd, {copied if copied >= 0 else '?'} transferred")
                if stopped:
                    return result(VerifyReport(name), False, stopped=stopped)
            else:
                counts: List[Optional[int]] = []
                for sub in LOG_SUBDIRS:
                    src = os.path.join(root, sub) + "/"
                    if os.path.isdir(src):
                        counts.append(_run_copy(run, _rsync_cmd(src, f"{target}/{sub}/"), COPY_TIMEOUT_S, f"[{dest.name}] copy full {sub}/"))
                copied = sum(c for c in counts if c is not None) if counts and all(c is not None for c in counts) else -1
                _say(f"[{dest.name}] copy full ({'--full' if full else 'no acked manifest on the destination'}): the whole records/ "
                     f"and rulings/ trees rsync'd, {copied if copied >= 0 else '?'} transferred")
            run(_rsync_cmd(mpath, f"{target}/{MANIFEST_DIR}/"), 600)
            n_entries = manifest.entries if manifest else len(read_manifest(mpath))
            use_slurm = bool(dest.slurm) and base is None
            verify_cmd = _verify_cmd(dest, name, since=base, sample=dest.sample, slurm=use_slurm, n_entries=n_entries)
            rep = _run_verifier(verify_cmd, name, run)
            if rep.corrupt or rep.missing:
                # a truncated or damaged file at its final name, or one the destination lacks (lost, or never sent by an interrupted
                # incremental copy): re-copy exactly those files (checksummed), verify again — how the copy heals itself
                rels = _on_disk_relpaths(root, rep.corrupt + rep.missing)
                if rels:
                    _rsync_list(run, _repair_cmd(src_root, f"{target}/", rels), rels, 3600, f"[{dest.name}] repair")
                    rep = _run_verifier(verify_cmd, name, run)
            if rep.note:
                notes.append(rep.note)
            if base and rep.hashed is None:
                notes.append(f"asked for an incremental verify since {base} but the remote verifier reported no mode: it predates the "
                             f"flag (the site's repo-mirror tool refreshes {dest.remote_src or 'the repo mirror'}) and hashed every entry")
            switched = False
            if rep.clean:
                run(_ssh(dest, f"printf %s '{name}' > {remote_root}/{MANIFEST_DIR}/{CURRENT_NAME}.tmp && "
                               f"mv -f {remote_root}/{MANIFEST_DIR}/{CURRENT_NAME}.tmp {remote_root}/{MANIFEST_DIR}/{CURRENT_NAME}"), 120)
                switched = True
                switch_seen = R.utc_now()
            return result(rep, switched, hashed=rep.hashed, presence_only=rep.presence_only, mode=rep.mode,
                          since=rep.since if rep.mode == "incremental" else None, sampled=rep.sampled, slurm=use_slurm,
                          note="; ".join(notes) or None)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            msg = str(exc)
            if getattr(exc, "stderr", None):
                msg += " :: " + str(getattr(exc, "stderr"))[-400:]
            return result(VerifyReport(name), False, error=msg)
    raise ValueError(f"mirrors.sync: unknown destination kind {dest.kind!r} (local | ssh; the Modal destination was removed, registry-slurm-only S3)")


def ack_payload(result: SyncResult, ingest_seq: int, manifest_path: str, now: Optional[str] = None,
                manifest: Optional[ManifestWrite] = None) -> Dict[str, object]:
    """The ``mirror_ack`` harvester record for a verified sync. ``bytes`` is what the destination holds: the verify's own
    count when it hashed every file, else (an incremental verify hashes a few thousand) the manifest's on-disk bytes as
    the grading host measured them writing it."""
    nbytes = result.verify.bytes
    if manifest is not None and (result.verify.mode == "incremental" or result.verify.hashed is None and result.verify.bytes == 0):
        nbytes = manifest.bytes
    return {"destination": result.destination, "acked_through_seq": int(ingest_seq), "manifest_name": result.manifest,
            "file_count": result.verify.ok, "bytes": nbytes, "checksums_sha256": manifest_sha(manifest_path),
            "acked_utc": now or R.utc_now(), "last_error": result.error}


def latest_manifest(root: str) -> Optional[str]:
    d = os.path.join(root, MANIFEST_DIR)
    names = sorted(f for f in os.listdir(d) if f.startswith("CHECKSUMS-") and f.endswith(".sha256")) if os.path.isdir(d) else []
    return names[-1] if names else None


def verify_destination(root: str, dest: Destination, runner=None) -> Dict[str, object]:
    """Per-destination lag against the LATEST LOCAL MANIFEST: files it names that the destination lacks
    or corrupts (``lag_files``), plus ``unsynced_local_files`` (local files newer than that manifest — they
    ride the next sync, not lag). ``local`` verifies in-process; ``ssh`` runs the one verifier there
    (``--no-prune``); any other kind is an error row by name."""
    latest = latest_manifest(root)
    manifest_count = len(read_manifest(os.path.join(root, MANIFEST_DIR, latest))) if latest else 0
    local_count = len(log_files(root))
    base: Dict[str, object] = {"destination": dest.name, "kind": dest.kind, "latest_manifest": latest, "remote_current": None,
                               "lag_files": manifest_count, "unsynced_local_files": max(0, local_count - manifest_count),
                               "missing": [], "corrupt": [], "ok": 0, "error": None}
    if not latest:
        base["error"] = "no manifest written yet (run `registry mirrors sync`)"
        return base
    seen: Optional[str] = None     # when the pointer read BEGAN: the observation's instant for the record
    try:
        if dest.kind == "local":
            seen = R.utc_now()
            base["remote_current"] = current_manifest(dest.path)
            rp = os.path.join(dest.path, MANIFEST_DIR, latest)
            rep = verify_manifest(dest.path, rp if os.path.exists(rp) else os.path.join(root, MANIFEST_DIR, latest), prune_plain=False)
        elif dest.kind == "ssh":
            run = runner or _default_runner
            remote_root = dest.path.rstrip("/")
            run(_ssh(dest, f"mkdir -p {remote_root}/{MANIFEST_DIR}"), 120)
            run(_rsync_cmd(os.path.join(root, MANIFEST_DIR, latest), f"{dest.host}:{remote_root}/{MANIFEST_DIR}/"), 600)
            # a FULL hash of every entry: under srun when the cluster entry says so, never on the login node by default
            rep = _run_verifier(_verify_cmd(dest, latest, no_prune=True, slurm=bool(dest.slurm), n_entries=manifest_count), latest, run)
            base["slurm"] = bool(dest.slurm)
            seen = R.utc_now()
            cur = run(_ssh(dest, f"cat {remote_root}/{MANIFEST_DIR}/{CURRENT_NAME} 2>/dev/null || true"), 120)
            base["remote_current"] = (getattr(cur, "stdout", "") or "").strip() or None
        else:
            raise ValueError(f"unknown destination kind {dest.kind!r} (local | ssh; the Modal destination was removed, registry-slurm-only S3)")
    except Exception as exc:
        base["error"] = str(exc)[-600:]
        return base
    base.update({"lag_files": manifest_count - rep.ok, "missing": rep.missing, "corrupt": rep.corrupt, "ok": rep.ok,
                 "mode": rep.mode, "hashed": rep.hashed, "seconds": round(rep.seconds, 1)})
    try:        # the retention rule's record of this destination's pointer (a local one's missing file is a definite absence; the
        # ssh read above is lossy, so its None is unknown); the record is housekeeping, never the verify's failure
        record_pointer(root, dest.name, base.get("remote_current"), "verify", now=seen, absent=(dest.kind == "local"))
    except OSError as exc:
        base["pointer_record_error"] = str(exc)[-300:]
    return base


def state_path(root: str) -> str:
    return os.path.join(root, "state", "mirror_state.json")


def write_state(root: str, results: Sequence[Mapping[str, object]], now: Optional[str] = None,
                extra: Optional[Mapping[str, object]] = None) -> str:
    """``state/mirror_state.json``: the last sync/verify outcome per destination (the status page reads it), plus ``extra``
    top-level sections (the pass's ``manifest_retention`` summary)."""
    p = state_path(root)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    payload: Dict[str, object] = {"utc": now or R.utc_now(), "destinations": list(results)}
    payload.update(dict(extra or {}))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    os.replace(tmp, p)
    return p


def read_state(root: str) -> Optional[Dict[str, object]]:
    p = state_path(root)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


VANISHED_NAMES_IN_ROW = 50


def sync_all(root: str, ingest_seq: int, dests: Optional[Sequence[Destination]] = None, runner=None,
             now: Optional[str] = None, deadline: Optional[float] = None,
             full: bool = False, keep: Optional[int] = None, prune: bool = True) -> List[Dict[str, object]]:
    """One manifest, every destination, an ack record per verified sync, the state file. Returns the
    per-destination outcome dicts. ``deadline`` (monotonic seconds; the CLI sets it from the budget at
    process start, so the manifest write counts against it) bounds the ssh copies' chunks; ``full`` asks the ssh
    destinations for a full verify (every entry hashed, under srun when configured) instead of the incremental one.
    A destination's error (a verify that timed out, an ssh outage) is that destination's row; the next destination
    still runs. After every destination, the manifest retention rule (:func:`prune_manifests`, ``keep`` manifests, default
    :data:`MANIFEST_KEEP`) prunes the grading host's ``manifests/`` and those of the destinations this pass reached without error, with
    the pointers this pass left (``dest_current``) as the live references -- and with every destination of the pass accounted for,
    the unreached included: one whose pointer is unknown from every source holds the grading host's deletions until a pass reaches it
    (:func:`prune_manifests`); its outcome is the state file's ``manifest_retention`` section and one log line per side that
    deleted or errored, and it never fails the pass (``prune`` False skips it)."""
    from . import harvest as H
    dests = list(dests if dests is not None else destinations(root))
    mw = write_manifest_report(root, int(ingest_seq))
    name = mw.name
    mpath = os.path.join(root, MANIFEST_DIR, name)
    if mw.vanished:
        _say(f"[manifest] {name}: {len(mw.vanished)} file(s) listed and gone before hashing (no .gz twin); skipped, the next "
             f"manifest picks them up: " + ", ".join(mw.vanished[:5]) + (" …" if len(mw.vanished) > 5 else ""))
    out: List[Dict[str, object]] = []
    live_currents: Dict[str, Optional[str]] = {}
    reached: List[Destination] = []
    for d in dests:
        res = sync(root, d, int(ingest_seq), name, runner=runner, deadline=deadline, full=full, manifest=mw)
        live_currents[d.name] = res.dest_current
        try:    # stamped when the sync observed it (the switch, or the start of its base read) unless the caller fixed the pass's time;
            # a None (an outage before the probe) erases nothing
            record_pointer(root, d.name, res.dest_current, "sync", now or res.observed_utc)
        except OSError as exc:                                           # housekeeping: never the pass's failure
            _say(f"[pointers] ERROR recording {d.name}'s pointer: {exc}")
        if not res.error:
            reached.append(d)
        row = {"destination": res.destination, "kind": d.kind, "manifest": res.manifest, "copied": res.copied, "ok": res.verify.ok,
               "missing": len(res.verify.missing), "corrupt": len(res.verify.corrupt), "switched": res.switched, "error": res.error,
               "chunks_done": res.chunks_done, "chunks_total": res.chunks_total, "stopped": res.stopped,
               "hashed": res.hashed, "presence_only": res.presence_only, "mode": res.mode, "since": res.since, "sampled": res.sampled,
               "slurm": res.slurm, "note": res.note, "vanished": len(mw.vanished), "vanished_names": mw.vanished[:VANISHED_NAMES_IN_ROW],
               "copy_mode": res.copy_mode, "copy_since": res.copy_since, "copy_files": res.copy_files,
               "dest_current": res.dest_current,
               "seconds": round(res.seconds, 1), "utc": now or R.utc_now()}
        if res.switched:
            row["ack_record"] = H.write_harvester_record(root, "mirror_ack", ack_payload(res, int(ingest_seq), mpath, now, manifest=mw), now)
        if d.kind == "ssh":
            _say(f"[{d.name}] verify {name}: copy {res.copy_mode} {res.copy_files if res.copy_files is not None else '-'} listed / "
                 f"{res.copied} transferred, mode {res.mode}, ok {res.verify.ok}, missing {len(res.verify.missing)}, corrupt "
                 f"{len(res.verify.corrupt)}, hashed {res.hashed}, presence-only {res.presence_only}, sampled {res.sampled}, "
                 f"slurm {res.slurm}, switched {res.switched}, {res.seconds:.0f} s" + (f"; STOPPED {res.stopped}" if res.stopped else "")
                 + (f"; NOTE {res.note}" if res.note else "") + (f"; ERROR {res.error}" if res.error else ""))
        out.append(row)
    retention: Optional[Dict[str, object]] = None
    if prune:
        try:
            retention = prune_manifests(root, reached, keep=keep, runner=runner, currents=live_currents, destinations_when_idle=False,
                                        now=now)
            # a line per side that deleted or errored; an idle side is silent (a local-only pass printed nothing before this rule)
            busy = [side for side in retention["sides"] if side.get("deleted") or side.get("error")]
            for line in format_retention({"sides": busy, "unrecorded": retention.get("unrecorded"), "stale": retention.get("stale"),
                                          "unreadable": retention.get("unreadable")}, header=False).splitlines():
                _say(line)
        except Exception as exc:        # housekeeping never costs a verified sync its state and acks
            retention = {"utc": now or R.utc_now(), "keep": MANIFEST_KEEP if keep is None else int(keep), "dry_run": False,
                         "references": {}, "sides": [], "error": f"{type(exc).__name__}: {str(exc)[-400:]}"}
            _say(f"[retention] ERROR {retention['error']}; the next pass retries")
    write_state(root, out, now, extra={"manifest_retention": retention} if retention is not None else None)
    return out


USAGE = ("usage: python3 -S -m sisteer.registry.mirrors verify <root> <manifest> [--json] [--no-prune] "
         "[--since <base manifest>] [--sample N] [--seed N] [--workers N]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python3 -S -m sisteer.registry.mirrors verify <root> <manifest> [--json] [--no-prune] [--since <base>] [--sample N]
    [--seed N] [--workers N]`` — the remote verifier (stdlib only; ``-S`` keeps site-packages out). ``--workers`` defaults to
    ``$SLURM_CPUS_PER_TASK`` inside a Slurm step, else 1."""
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "verify":
        print(USAGE, file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog="python3 -S -m sisteer.registry.mirrors verify", add_help=False)
    ap.add_argument("root")
    ap.add_argument("manifest")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--since", default=None)
    ap.add_argument("--sample", type=int, default=VERIFY_SAMPLE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    try:
        args, unknown = ap.parse_known_args(argv[1:])
    except SystemExit:
        print(USAGE, file=sys.stderr)
        return 2
    workers = args.workers if args.workers is not None else int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    rep = verify_manifest(args.root, args.manifest, prune_plain=not args.no_prune, since=args.since, sample=args.sample,
                          seed=args.seed, workers=workers)
    if unknown:
        rep.note = ((rep.note + "; ") if rep.note else "") + "ignored unknown arguments: " + " ".join(unknown)
    if args.json:
        print(json.dumps(rep.as_json()))
    else:
        print(f"{rep.manifest}: mode={rep.mode} ok={rep.ok} missing={len(rep.missing)} corrupt={len(rep.corrupt)} "
              f"hashed={rep.hashed} presence_only={rep.presence_only} sampled={rep.sampled} {rep.seconds:.0f}s")
    return 0 if rep.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
