"""Immutable, manifest-addressed store GENERATIONS on cluster A for the cluster-read adapters.

The freshness contract (spec "Interfaces and contracts"): a cluster copy of a store root is admitted for a read only
when evidence proves it EQUAL IN CONTENT to the source's canonical copy at read time -- a whole-root content attestation
(every file's path, size and sha256; the two-manifest form) computed at the canonical copy and matching the cluster
copy, checkable without the grading host's database. Authority is declared PER ROOT COMPONENT in
``configs/registry/cluster_reads.yaml``:

* ``grading_host_copy`` -- the canonical copy is the grading host tree the registry reads today (Modal-, cluster-B- and grading-host-produced
  components; the grading host judge outputs beside cluster-generated responses make the grading host tree the complete one). It is
  CARRIED to cluster A as an immutable generation::

      $SHARED/registry/stores/gen/<slug>/<content_sha8>/       the tree (never written after its verify)
      $SHARED/registry/stores/gen/<slug>/MANIFEST-<sha8>.sha256  `<sha256>  <size>  <relpath>` per regular file
      $SHARED/registry/stores/gen/<slug>/MANIFEST-<sha8>.json    n_files, bytes, name_size_sha256, content_sha256, built_utc, ...
      $SHARED/registry/stores/gen/<slug>/MANIFEST-<sha8>.<lsha8>.json   the same, keyed by its listings (lsha8 = listings_sha256[:8]): one
                                                                     IMMUTABLE attestation per (content, listings); the plain .json is the
                                                                     attestation the verify job ran against
      $SHARED/registry/stores/gen/<slug>/VERIFIED-<sha8>.json    written by store_verify.sbatch after hashing EVERY entry
      $SHARED/registry/stores/gen/<slug>/DEST-STAT-<sha8>.json   the verified tree's own stat snapshot (the seal the reader checks)
      $SHARED/registry/stores/gen/<slug>/VERIFIED-<sha8>.<lsha8>.json   a RE-ATTESTATION of the published generation's directory listings
      $SHARED/registry/stores/gen/<slug>/DEST-STAT-<sha8>.<lsha8>.json  (`registry stores attest`): the pinned verifier run once more over the
                                                                     sealed tree against MANIFEST-<sha8>.<lsha8>.json, its report and snapshot
                                                                     keyed like the attestation; the live VERIFIED / DEST-STAT are never rewritten
      $SHARED/registry/stores/gen/<slug>/SEALED-<sha8>.json      written at publish after every DIRECTORY was chmod 0555
      $SHARED/registry/stores/gen/<slug>/PUBLISHED-<sha8>.json   written by EVERY publication (push, --on-cluster, mirror, the carry, the
                                                                  move, an explicit adoption) INSIDE the publish lock, before `current`
                                                                  switches: change_utc (the producer's committed change), published_utc,
                                                                  previous, producer, note. A `current` without its record
                                                                  is refused by name (current_unrecorded) until adopted explicitly -- by
                                                                  every publisher and by the reversed mirror's verify, which also
                                                                  requires the matching VERIFIED report and the SEALED record beside a
                                                                  `current` before it is green (current_unverified / current_unsealed)
      $SHARED/registry/stores/gen/<slug>/current -> <sha8>       the ONE publication step (a symlink rename) -- REFUSED unless MANIFEST,
                                                                  VERIFIED, DEST-STAT and SEALED stand beside the tree (the publish guard);
                                                                  a compare-and-switch under gen/<slug>/.publish.lock for EVERY publisher,
                                                                  with the record check, the producer rule and the record write inside it

  A manifest attests every regular file (path, size, sha256) AND every symlink (path and literal target; a target that
  escapes the component base is refused by name -- the copy would resolve it to bytes nobody attested). On the destination
  a symlink's target is spelled per the name escape below (:func:`escape_target`: its referent is renamed there, so the true
  spelling would dangle); the verify and the admission check THAT spelling, the manifest keeps the true target. ``registry
  stores mirror``: (i) manifest + stat snapshot of the canonical copy through a full-resolution ``(size, mtime_ns, ino, ctime_ns)``
  stat cache (only changed files are read); (ii) ``rsync -rlt --partial --relative`` (never ``--delete``; ``nice`` /
  ``ionice -c3``) into ``gen/<slug>/<sha8>/`` with ``--link-dest`` against the published generation, so unchanged files
  are hard links and a generation costs its delta, then a second pass that FORCES every file whose manifest sha differs
  from the published generation's (``--files-from`` + ``--ignore-times``: rsync's size/mtime quick check would otherwise
  hard-link a same-size, mtime-preserving edit to the old bytes); (iii) a post-copy stat walk -- any file whose stat moved
  INVALIDATES the attempt (the directory is renamed ``.invalid-<sha8>-<stamp>``, never published, named in the report);
  (iv) the manifest is pushed beside the tree and ``store_verify.sbatch`` (16 CPU, ``high``, no GPU, ``--time`` 2x the
  measured rate) SEALS every regular file (``chmod a-w``), hashes every manifest entry and checks every attested link --
  a generation is verified in full exactly once -- and writes ``VERIFIED-<sha8>.json`` plus ``DEST-STAT-<sha8>.json``;
  (v) only a generation whose VERIFIED report matches its manifest is PUBLISHED by the ``current`` rename, and a
  ``current`` that names a generation WITHOUT a matching report is reported (``current_unverified``), never called
  unchanged. The report is read with a BOUNDED WAIT after the watch says DONE (:func:`_await_report`: a compute node's
  write is not visible from the login node for up to ~60 s; re-read every 15 s for up to 120 s while absent, stale or
  unreadable). A PRESENT report that finds a missing / corrupt / extra file renames the tree ``.failed-<sha8>-<stamp>``
  and publishes nothing; a report still not visible inside the bound is ``verify_report_missing`` -- the tree is LEFT
  IN PLACE and the next tick publishes it when the report is visible. Both shortcuts probe the tree (``test -d``), so a
  ``current`` or a report naming a tree that was renamed away falls through to a fresh copy and verify, never
  ``unchanged`` / ``published_existing_verified`` over a directory that is not there. The recorded state is never trusted
  over that probe (a tick that read ``published`` from the record while ``current`` dangled left every cluster read
  of the component refused as ``generation_changed`` until the content happened to change): a DANGLING ``current`` --
  the pointer names this content but ``gen/<slug>/<sha8>`` is not a directory -- is named in the log together with the
  ``.failed-<sha8>-*`` / ``.invalid-<sha8>-*`` trees found beside it, the record drops ``published`` and carries
  ``current_dangling`` (``stores status`` prints it as a PROBLEM line with no probe) until the fresh copy is verified and
  ``current`` republished under the outcome ``republished_dangling_current`` -- the same sha8 path, re-created empty and
  verified in full again, so "never written after its verify" holds for the new tree as it did for the renamed one; every
  report and snapshot that spoke for the replaced tree, the keyed re-attestations included, is moved aside first
  (:meth:`Remote.reports_aside`), so no report of an older tree ever attests the fresh one. A
  repair whose copy is cut short (``rsync_failed`` with the dangle recorded) is RESUMED by the next tick from that record
  -- ``current`` still names the sha8 and the tree is the mirror's own partial copy -- never reported ``current_unverified``
  over it; a pointer published behind the mirror's back (no such record) still is.

  :func:`admit_generation` is the destination change guard -- a fast PRECHECK -- the cluster-read task runs beside
  :func:`revalidate`, before it reads a pinned generation and again when the read completes: the VERIFIED report must be
  ok and match the manifest; a stat walk of the tree against ``DEST-STAT`` (size, mtime_ns, ino, mode, and ctime for a
  regular file with one link -- a later generation hard-linking the same inode moves ctime and nlink, so a hard-linked
  file's ctime is out of the key) refuses any moved / missing / extra file or link by name; moved files and a seeded
  sample are re-hashed and every link's target re-read. The seal cannot stop the owning user from rewriting same-length
  bytes and restoring mode + mtime, so the precheck is not the guarantee: **the READER is** -- it hashes every input it
  reads (``adapters/base.py check_input``) and compares that sha with :func:`manifest_index`'s entry for the path, loaded
  from the pinned generation's ``MANIFEST-<sha8>.sha256`` and approved by its VERIFIED report; a mismatch refuses that
  input by name ("bytes differ from the verified generation"). Every byte read is thus a byte the verifier hashed.

  Step 0 of every tick, before any copy or submit, PINS THE CODE: the running checkout's revision is pushed to the
  cluster repo mirror (``mirror-repos.sh``, else ``git push <mirror> <sha>:refs/heads/gate/<sha7>``), the verifier's
  presence at that revision is checked on the mirror (``git cat-file -e <sha>:scripts/registry/landing/store_verify.sbatch``;
  absent -> the tick is refused by name before anything is copied), ``git archive <sha>`` of the verifier (the sbatch,
  ``src/sisteer/__init__.py``, ``src/sisteer/registry``) is extracted into ``$SHARED/registry/stores/trees/<sha>/`` and
  THAT sbatch is submitted, so the job runs exactly the code the grading host runs.
* ``cluster_a_origin`` -- the canonical copy is the producer's original on cluster A; the grading host holds a read-only mirror
  of it. No copy: ``registry stores check`` compares the grading host mirror's manifest with a manifest of the original built by
  ``store_verify.sbatch manifest-only`` and records ``match`` / ``mismatch`` in ``state/store_manifests/<slug>/authority.json``
  -- a mismatch is what later refuses that component's in-place cluster route (the adapter reads on the grading host).

* ``cluster_a`` -- the canonical copy IS the generation tree on cluster A, written by ``registry stores push`` (the producer's own
  push from the grading host, or ``--on-cluster`` / ``python3 -S -m sisteer.registry.stores push`` inside the producer's own job:
  local copy into ``.incoming-<sha8>``, in-process verify, seal, guard, PUBLISHED; the transitional carry; or the one-time verified
  move: ``stores check-authority`` reads every `current` in one batch, ``stores verify-authority`` proves a generation equal to its
  copy on the grading host and adopts it) and read in place by the shards; the grading host holds a verified mirror at the
  component's ``grading_host_mirror`` (default: the base). Such a component names its ``producer`` (``grading_host:<session>`` / ``cluster`` /
  ``frozen``), and while a ``grading_host`` producer has not adopted the push tool, ``transitional_carry: true`` with a ``carry_deadline``.
  Every registered adapter has an ``adapters:`` entry -- shards and components, or ``cluster: refused`` with a written reason -- and
  ``registry stores coverage`` checks every adapter's DECLARED roots against the components by constructing the adapters and reading
  ``roots`` (never ``discover()``, never a listing): a root under the home must lie inside a component (or above its members:
  ``restricted``, the shard's own discovery being the exact check); a root outside the home must be retired by the adapter's
  ``inputs_retired`` or declared ``volatile_roots`` with a reason; the test pins the live file at zero UNCOVERED.

  **The REVERSED mirror.** ``registry stores mirror`` keeps its unit and its command and dispatches
  on each component's authority (:func:`run_mirror`): ``grading_host_copy`` -> the carry above (:func:`mirror_component`); ``cluster_a_origin``
  -> skipped (`stores check` compares); ``cluster_a`` -> :func:`mirror_authority`, by producer. A ``grading_host:<session>`` producer: VERIFY
  ONLY -- the grading host mirror's manifest (the stat cache) against the published generation's; equal -> ``verified`` (green; the grading host
  manifest files are saved as the pin's local copy); unequal -> ``unpushed_changes`` (amber: the producer pushes; nothing is copied) --
  except while ``transitional_carry`` holds, when the registry PUSHES on the producer's behalf (mode ``carry``; ``change_utc`` = the
  grading host mtime of the changed files; ``min_publish_interval_s`` defers it like a push; a carry past ``carry_deadline`` still runs and
  is reported ``carry_overdue``). A ``frozen`` producer: verify only; a difference is ``frozen_differs`` (red), never repaired; nothing
  published is ``no_current`` (red). A ``cluster`` producer: the registry PULLS the published generation into the registry-owned mirror
  ``<root>/stores/mirror/<slug>/`` (:func:`pull_component`: rsync, then every escaped name and link target restored to its true
  spelling and every entry the generation does not name removed, by name, from that registry-owned directory and nowhere else),
  hash-verified -- the pulled tree's manifest must BE the generation's
  (``pulled`` / ``pull_verify_failed``). ``--full`` (the weekly ``registry-store-mirror-full.timer``, Sunday 03:00 UTC): the grading host side
  is hashed with the stat cache IGNORED (every file read; the cache refreshed) and the published generation is RE-HASHED on the
  destination by the pinned verifier into ``<stores>/full/<slug>/VERIFIED-<sha8>-<stamp>.json`` (the live report untouched); a
  mismatch is ``full_verify_failed`` (red): named, nothing renamed, `current` untouched -- the operator decides. The outcome's
  level (:data:`OUTCOME_LEVELS`) rides in every result and in `stores status`, which also shows the last `check-authority` reading per
  component (the authority rows). A verify job that completes with no report VISIBLE yet (NFS, seconds after the job) is
  ``verify_report_missing`` after a bounded re-read: the tree is LEFT for the next run, never renamed `.failed-` (the live
  bigcodebench_full misread of 2026-10-04); only a report that is present and does not match renames.

**Re-attestation of a published generation's directory listings** (`registry stores attest`, :func:`attest_listings`). A generation
verified before the 2026-10-04 directory-listings rule (``manifest_rules`` without ``directories``; its report attests no listing) is
``listings_unattested`` on the mirror (amber: its files stand published and verified; the window checks no listing for it). The attest
builds the grading host manifest, requires `current` to be its content and the live report to admit its files, writes the keyed
``MANIFEST-<sha8>.<lsha8>.json`` beside the tree (once), runs the PINNED verifier over the sealed tree against THAT attestation into
``VERIFIED-<sha8>.<lsha8>.json`` (the live report and snapshot are never rewritten), and on a matching report records
``published_attestation`` so the window's plan pins it and every pin-side check reads it from then on. THE RULE: the attest never changes
the tree. A directory set that differs from the canonical's in EITHER direction is refused by name, ``listing_only_drift`` -- a directory
the generation holds that the canonical lacks (nothing removes an entry from an immutable generation) or an attested directory the
generation lacks (nothing is created inside a published, sealed tree: the A -> B -> A rebuild copies a fresh tree rather than rewriting
one a window may be reading) -- and the next FILE change publishes a generation that attests the current listings. The mirror reads the
keyed report on every tick (:func:`listings_verdict`): an attested listing is ``unchanged``, a refusing one ``listing_only_drift``, a
listing moved from the attested one ``listings_changed``.

:func:`revalidate` is the admission helper the cluster-read window calls TWICE (a verified generation can be stale by
admission): a full stat walk of the canonical copy on the grading host against the pinned generation's stat snapshot, plus
a hash of every file whose stat moved and a seeded 1 % / >= 64-file sample; it returns the refusal BY NAME
(``no_component``, ``no_manifest``, ``no_stat_snapshot``, ``file_set_changed``, ``content_changed``, ``changed_during_hash``).

Everything the cluster side runs is stdlib-only (``python3 -S -m sisteer.registry.stores verify|manifest``, the way the
record-log mirror's verifier runs); PyYAML is imported only when the configuration is read on the grading host. Nothing here
writes the live database, a producer's root, or the record log: the grading host side writes ``<root>/state/`` only.
Tests drive the whole flow with a local destination (a directory) and a runner that intercepts ``sbatch`` and the watcher.
"""
from __future__ import annotations

import argparse
import fcntl
import fnmatch
import getpass
import glob
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import stat as statmod
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import records as R
from . import site_aliases as SA
from .db import DEFAULT_ROOT
from .listing import Entries, directories_json, listing_entries, listing_fingerprint, listings_sha256, normalize_entries
from .mirrors import PARTIAL_DIR, RSYNC_SSH, SSH_OPTS

CONFIG_SCHEMA = "cluster_reads/2"
#: the schemas :func:`load_config` reads: v1 (the six heavy adapters, two authorities) and v2 (every registered adapter declared, the
#: ``cluster_a`` authority with its producer, the per-adapter cluster refusal, the volatile roots). The rules apply by content, so a
#: v1 document loads unchanged and a v2 document may still declare only ``grading_host_copy`` components.
CONFIG_SCHEMAS = ("cluster_reads/1", CONFIG_SCHEMA)
MANIFEST_SCHEMA = "store_manifest/1"
VERIFIED_SCHEMA = "store_verified/1"
AUTHORITY_SCHEMA = "store_authority/1"
STATE_SCHEMA = "store_mirror_state/1"
#: ``cluster_a`` -- the canonical copy is the generation tree on cluster A, written by `registry stores push` (its
#: component names its ``producer`` and the grading host's verified ``grading_host_mirror``); ``cluster_a_origin`` -- the producer's original on
#: cluster A, the grading host a read-only mirror of it (kept as a valid value; no member today); ``grading_host_copy`` -- the grading host tree the
#: registry reads today, carried to cluster A as a generation (valid during the transition to ``cluster_a``)
AUTHORITIES = ("cluster_a", "cluster_a_origin", "grading_host_copy")
#: ``producer`` of a component: ``grading_host:<session>`` (a grading-host-side producing session), ``cluster`` (written on cluster A by a job),
#: ``frozen`` (nobody writes it any more); REQUIRED for a ``cluster_a`` component, informational for the others
PRODUCER_KINDS = ("grading_host", "cluster", "frozen")
_PRODUCER_RE = re.compile(r"^(grading_host:[A-Za-z0-9][A-Za-z0-9._-]*|cluster|frozen)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
#: the keys a component / an adapter entry may carry (a typo is refused by name, never silently ignored)
COMPONENT_KEYS = frozenset({"base", "members", "exclude", "authority", "origin", "original", "note", "producer", "grading_host_mirror",
                            "transitional_carry", "carry_deadline", "min_publish_interval_s"})
#: a ``cluster_a``-only key on another authority is refused: the push, the carry and the mirror exist for that authority alone
CLUSTER_A_ONLY_KEYS = ("grading_host_mirror", "transitional_carry", "carry_deadline", "min_publish_interval_s")
ADAPTER_KEYS = frozenset({"shards", "components", "cluster", "reason", "volatile_roots", "note"})
#: ``adapters.<name>.cluster: refused`` -- the adapter is declared NOT readable on the cluster, with a written ``reason``
CLUSTER_REFUSED = "refused"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "registry", "cluster_reads.yaml")
CONFIG_ENV = "REGISTRY_CLUSTER_READS"
STATE_SUBDIR = "store_manifests"          # <root>/state/store_manifests/<slug>/
STATE_FILE = "store_mirror.json"          # <root>/state/store_mirror.json (the status page reads it)
LOCK_FILE = "store_mirror.lock"
STAT_CACHE = "stat_cache.json"
AUTHORITY_FILE = "authority.json"
HOME = os.path.expanduser("~")
#: the full-resolution stat key of the CANONICAL copy, in the order the cache entries, the stat snapshots and :func:`stat_key` spell it
STAT_FIELDS = ("size", "mtime_ns", "ino", "ctime_ns")
#: the stat key of a VERIFIED generation on the destination (``DEST-STAT-<sha8>.json``, :func:`dest_stat_key`): the mode records
#: the seal; ctime rides along ONLY for a regular file with one link (the owner can restore mode and mtime after a same-length
#: rewrite, never ctime), because a later generation hard-linking the same inode moves ctime and nlink without touching a byte --
#: a hard-linked file's bytes are the reader's per-input hash against :func:`manifest_index` to guarantee
DEST_STAT_FIELDS = ("size", "mtime_ns", "ino", "mode", "ctime_ns_if_one_link")
#: what the tick pins on the cluster: the paths ``git archive <sha>`` extracts into ``$SHARED/registry/stores/trees/<sha>/``
PINNED_PATHS = ("scripts/registry/landing/store_verify.sbatch", "src/sisteer/__init__.py", "src/sisteer/registry")
MIRROR_REPOS = os.path.join(HOME, "scripts", "mirror-repos.sh")   # the grading host's repo-mirror push (all branches, both clusters)
MIRROR_TIMEOUT_S = 1800.0
MANIFEST_RULES = {
    "files": "regular files only (other entries are counted), relpath under the component base, members expanded and excludes "
             "applied as declared; sorted by the UTF-8 bytes of the relpath (LC_ALL=C order)",
    "directories": "ADDITIVE (2026-10-04): every directory the walk visits, keyed by its relpath under the base ('' for the base itself), "
                   "with the canonical directory's WHOLE listing as sorted [name, kind] pairs (kind from lstat: dir / file / link / other; "
                   "excluded entries included, since the listing is what the canonical directory contains), empty directories included; "
                   "listings_sha256 = sha256 over {relpath: listing fingerprint}; not part of the two D269 hashes (a generation is named by "
                   "its file content) -- a manifest without the field attests no listing; the verify job reproduces attested empty "
                   "directories, verifies every listing, and a shard compares its listing inputs against this attestation. An attestation is IMMUTABLE "
                   "per (content, listings): the grading host state dir and the destination hold MANIFEST-<sha8>.<lsha8>.json for every distinct "
                   "listing of a content generation, written once and never rewritten, and a pin names the one it was verified against "
                   "(a content sha alone cannot tell two listings of the same files apart; the plain MANIFEST-<sha8>.json is the FIRST "
                   "attestation on the grading host and the verify job's on the destination)",
    "symlinks": "every symlink as '<relpath> -> <literal target>\\n', sorted the same way, appended to BOTH texts below; a target "
                "that escapes the component base (absolute, or resolving above it) is refused, never attested; on the destination the "
                "target is spelled per name_escape_rule (its referent is renamed there), the manifest keeps the true target",
    "name_size": "sha256 over the text of one '<relpath> <size>\\n' line per file, then the symlink lines",
    "content": "sha256 over the text of one '<sha256>  <relpath>\\n' line per file (the sha256sum output form), then the symlink lines",
    "manifest_lines": "'<sha256>  <size>  <relpath>' per file, in the same order; the symlinks are listed in MANIFEST-<sha8>.json",
}
#: the sample :func:`revalidate` hashes beside every file whose stat moved
SAMPLE_FRACTION = 0.01
SAMPLE_MIN = 64
SEC_PER_FILE = 0.002       # per-file overhead in the verify --time estimate (NFS stat + open)
SHELL_TIMEOUT_S = 300.0    # one shell step or one sbatch on the destination
SEAL_TIMEOUT_S = 1800.0    # the directory seal of one generation (a `find -exec chmod` over NFS; relu_paired_run_state has 75k files)
LIST_CAP = 50              # how many paths a report lists (counts are always complete)
#: the records the publish step writes beside a generation: ``PUBLISHED-<sha8>.json`` carries the
#: producer's committed change (``change_utc``) and the publication's identity -- the publication clock's start; ``SEALED-<sha8>.json`` records
#: the directory seal (every directory of the generation ``chmod 0555`` after the verify, deepest first; the parent ``gen/<slug>/`` stays
#: writable so `current` can switch and a failed tree can be renamed)
PUBLISHED_SCHEMA = "store_published/1"
SEALED_SCHEMA = "store_sealed/1"
#: ``<root>/state/store_authority.json`` -- what `registry stores check-authority` read of every component's `current` in ONE batch
AUTHORITY_STATE_SCHEMA = "store_authority_state/1"
AUTHORITY_STATE_FILE = "store_authority.json"
MOVE_SUBDIR = "store_move"          # <root>/state/store_move/<slug>.{grading_host,cluster_a}.sha256 + .verified.json (the one-time move's evidence)
DIR_SEAL_MODE = 0o555
#: the rsync `--chmod` of every STAGING copy (:func:`rsync_argv`, :func:`rsync_forced_argv`): a directory of the unpublished tree is
#: owner-writable whatever the source's mode, so the verify can complete it (an attested empty directory the copy left out); the
#: publication seal (:data:`DIR_SEAL_MODE`, :func:`seal_generation`) is applied after the verify and is never weakened by this
STAGING_DIR_CHMOD = "Du+w"
#: the modes a PUBLISHED record names: the producer's push from the grading host, the producer's push inside its own cluster job, an existing
#: verified generation adopted (sealed + recorded) by a push that found it unchanged, the one-time verified move, the transitional carry
PUBLISH_MODES = ("push", "push_on_cluster", "mirror", "adopt", "move", "carry")
#: the outcomes `registry stores push` exits 0 on -- plus ``deferred`` (``min_publish_interval_s`` not yet elapsed: the
#: producer's next push publishes; nothing is wrong) and ``listings_unattested`` (the content is published and verified; only its
#: directory listings were never attested, which is the registry operator's `stores attest`, not the producer's push)
PUSH_OK_OUTCOMES = ("published", "published_existing_verified", "unchanged", "deferred", "listings_unattested")
#: THE REVERSED MIRROR. ``<root>/stores/mirror/<slug>/`` is the registry-owned pull target of a
#: ``cluster_a`` component whose producer is ``cluster`` (its ``grading_host_mirror`` unless the yaml names one); ``<stores>/full/<slug>/`` on
#: the destination holds the weekly `--full` re-hash reports (the live ``VERIFIED-<sha8>.json`` is never rewritten).
MIRROR_SUBDIR = os.path.join("stores", "mirror")
FULL_SUBDIR = "full"
#: every mirror outcome's level, the status page's color: green = the canonical copy and its mirror agree (or nothing was to be done),
#: amber = the grading host producer has not pushed yet (nothing is wrong with the mirror; the delay is the producer's), red = a disagreement or a
#: failure the operator reads. ``skipped``, ``dry_run``, ``invalidated`` (retried next tick) and ``deferred`` are green
OUTCOME_LEVELS = {
    "unchanged": "green", "published": "green", "published_existing_verified": "green", "verified": "green", "pulled": "green",
    "dry_run": "green", "skipped": "green", "invalidated": "green", "deferred": "green",
    "republished_dangling_current": "green",
    "unpushed_changes": "amber",
    # the generation states a run records while it is in flight (or left behind by a crash; the next tick overwrites them): not green --
    # `publishing` is written BEFORE the directory seal, so a tick that dies between its verify and its switch leaves it, never `published`
    "copying": "amber", "awaiting_verify": "amber", "verifying_inline": "amber", "publishing": "amber",
    "verify_cancelled": "amber", "verify_in_flight": "amber", "verify_unconfirmed": "red",
    "frozen_differs": "red", "no_current": "red", "dangling": "red", "no_manifest": "red", "no_grading_host_mirror": "red",
    # the carry that would republish a DANGLING pointer, held by `min_publish_interval_s`: nothing is published and the shards cannot
    # read the component, so it is never the plain `deferred` (green) -- a deferred REPAIR is not a successful deferral
    "dangling_carry_deferred": "red",
    # the tree stands under the pointer but it is the push flow's unfinished repair of it (its copy, its seal or its guarded switch did
    # not complete): a present directory is not a repaired `current` -- the flow that owns the repair finishes it
    "dangling_repair_pending": "red",
    "pull_failed": "red", "pull_verify_failed": "red", "full_verify_failed": "red", "verify_report_missing": "red",
    "rsync_failed": "red", "tar_failed": "red", "verify_failed": "red", "verify_job_failed": "red", "verify_watch_timeout": "red",
    "current_moved": "red", "lock_busy": "red", "producer_mismatch": "red", "current_unrecorded": "red", "current_unverified": "red",
    "current_unsealed": "red",
    # the published generation's FILES stand verified; its directory LISTINGS were never attested (a generation verified before the
    # 2026-10-04 directory-listings rule): nothing is wrong with the mirror -- `stores attest` attests them (amber, exit 0)
    "listings_unattested": "amber",
    "listings_changed": "red",
    # a re-attestation found the generation's directory set differs from the canonical's while its files match: named by the keyed report
    # `stores attest` wrote, a listing-COVERAGE limitation (the listing readers keep the grading host route; the pin is unchanged), not an
    # integrity failure -- amber, exit 0; the next file change publishes a generation that attests the current listings
    "listing_only_drift": "amber",
    # a re-attestation hashed every file of the published generation and found the FILES wrong (missing / corrupt / extra): an integrity
    # failure the live report predates, standing on every tick until a later full hash (a fresh `stores attest`, `--full`) passes
    "attest_integrity_failed": "red",
    "refused": "red", "error": "red",
}
#: ``last_run.outcome`` in ``state/store_mirror.json`` is NOT always a mirror outcome. Two auxiliary commands stamp the same slot in a
#: namespace of their own -- `stores check` writes ``check:<state>`` (match / mismatch / error) and the one-time `stores verify-authority`
#: writes ``verify-authority:<state>`` (verified / not_applicable / unequal / no_current / no_manifest / verify_failed / current_moved /
#: source_changed / error) -- and a mirror run stamps ``verify_inline`` / ``verify_submitted`` while its verify is in flight (a crash
#: leaves one behind; the next run overwrites it). Read the slot through :func:`last_run_level`, never :data:`OUTCOME_LEVELS` alone: the
#: status page's authority rows did exactly that and called a correctly VERIFIED store, and a matching in-place check, red.
CHECK_OUTCOME_PREFIX = "check:"
VERIFY_AUTHORITY_OUTCOME_PREFIX = "verify-authority:"
#: `stores attest` (the re-attestation of a published generation's directory listings; :func:`attest_listings`) stamps ``attest:<outcome>``:
#: ``attested`` / ``already_attested`` green (the current listings are attested; nothing left to do), ``content_differs`` amber (the grading host
#: content is not the published generation's: the mirror's next tick publishes it, which attests the current listings; nothing to attest
#: here), ``skipped`` green (an authority the attest does not cover), ``verify_submitted`` amber (in flight; a crash leaves it behind),
#: everything else red by name -- ``current_unsealed`` / ``current_unrecorded`` among them: a `current` without its evidence
#: (:data:`CURRENT_EVIDENCE`) is never attested, so the attest cannot turn the mirror's red into a green (round 1 on 2c)
ATTEST_OUTCOME_PREFIX = "attest:"
ATTEST_OUTCOME_LEVELS: Dict[str, str] = {
    "attested": "green", "already_attested": "green", "skipped": "green",
    "content_differs": "amber", "verify_submitted": "amber", "listing_only_drift": "amber",
    "attest_verify_failed": "red", "no_current": "red", "current_dangling": "red", "current_unverified": "red",
    "current_unsealed": "red", "current_unrecorded": "red",
    "no_listings": "red", "verify_job_failed": "red", "verify_watch_timeout": "red", "verify_report_missing": "red", "lock_busy": "red",
    "error": "red",
}
AUXILIARY_OUTCOME_LEVELS: Dict[str, Dict[str, str]] = {
    CHECK_OUTCOME_PREFIX: {"match": "green", "not_applicable": "green", "mismatch": "red", "error": "red"},
    VERIFY_AUTHORITY_OUTCOME_PREFIX: {"verified": "green", "not_applicable": "green",
                                      "unequal": "red", "no_current": "red", "no_manifest": "red", "verify_failed": "red",
                                      "current_moved": "red", "source_changed": "red", "error": "red"},
    ATTEST_OUTCOME_PREFIX: ATTEST_OUTCOME_LEVELS,
}
IN_FLIGHT_OUTCOME_LEVELS = {"verify_inline": "amber", "verify_submitted": "amber"}


def last_run_level(outcome: Optional[str]) -> Optional[str]:
    """green / amber / red for a component's ``last_run.outcome`` as ``state/store_mirror.json`` records it, whichever command stamped it:
    a mirror / push / pull outcome through :data:`OUTCOME_LEVELS`; ``check:<state>`` and ``verify-authority:<state>`` through
    :data:`AUXILIARY_OUTCOME_LEVELS` (a matching check and a verified move are green, not a failed mirror; their failures red by name);
    the in-flight stamps amber. None for no outcome; a state no table names is red (an unnamed state is never green)."""
    if not outcome:
        return None
    for prefix, table in AUXILIARY_OUTCOME_LEVELS.items():
        if outcome.startswith(prefix):
            return table.get(outcome[len(prefix):], "red")
    return IN_FLIGHT_OUTCOME_LEVELS.get(outcome) or OUTCOME_LEVELS.get(outcome, "red")


#: a verify job that reads DONE and whose report is not visible yet (NFS close-to-open / attribute cache: the file the job wrote may stay
#: invisible from the login node for up to ~60 s -- measured 2026-10-05 against every one of the 48 `verify_failed` outcomes since 09-29,
#: all of them this misread; the live bigcodebench_full tree was renamed `.failed-` on it, 2026-10-04): the report is re-read this many
#: times, this far apart (8 x 15 s = 120 s, twice the measured delay), before the run gives up as ``verify_report_missing`` -- and a
#: missing report NEVER renames the tree (only a present, mismatching one does); the next run publishes the verified tree
REPORT_REREAD_TRIES = 8
REPORT_REREAD_PAUSE_S = 15.0
#: the pull's restore stages every escaped entry of a directory under this prefix + a nonce BEFORE any is renamed to its true name:
#: a generation may hold both `a:b` and `a%3Ab`, spelled `a%3Ab` and `a%253Ab` on the destination, and restoring `a%253Ab` ->
#: `a%3Ab` while the other `a%3Ab` still stands would take it for the previous pull's copy and delete it. The prefix carries no ':' or
#: '%', so a staged name is never itself unescaped; a leftover (a crash between the two phases) names nothing in the manifest, so the
#: next pull's prune removes it
RESTORE_STAGE_PREFIX = ".restore-"
#: a `stores mirror` run asked to wait for the mirror lock (`--lock-wait-s`; the weekly `--full`) polls this often
MIRROR_LOCK_POLL_S = 5.0
#: what must stand beside a `current` before the reversed mirror calls it green (the push flow's rule and check-authority's problem
#: set): a VERIFIED report that admits the generation's manifest, SEALED-<sha8>.json, PUBLISHED-<sha8>.json. Each absence is a red
#: outcome by name, in this order; `--adopt-unrecorded` adopts a VERIFIED `current` (the seal when missing, the record when missing) for a
#: grading_host or frozen producer, never for a cluster producer (its job records and seals before it switches, so a bare `current` there is a
#: publisher mid-flight or a crash)
CURRENT_EVIDENCE = ("current_unverified", "current_unsealed", "current_unrecorded")
#: ADMISSION BY ATTESTATION. Once per READ OPERATION (a window or a tick) and before
#: the read array, ``attest.sbatch`` runs :func:`attest_plan` over every pinned generation of the plan: (a) the authority facts -- `current`
#: names the pin, the tree exists, MANIFEST / VERIFIED / DEST-STAT / SEALED / PUBLISHED stand beside it and agree with the manifest; (b) a
#: FULL DIRECTORY LISTING of the generation (recursive ``os.scandir``: names, ``d_type``, every symlink's target; no stat, no hash) compared
#: with the manifest's entries and attested links -- an extra, missing or re-typed entry, or a link target that differs or escapes, refuses
#: by name; (c) the directory seal (``lstat`` of every directory == 0555) and the directory count against the SEALED record; (d) the content
#: check -- the FIRST attestation of a generation (no ``ATTESTED-<sha8>.json`` beside it, or one older than :data:`ATTEST_FULL_WALK_MAX_AGE_S`)
#: runs the full :func:`admit_generation` walk (the DEST-STAT stat walk + the 1 % sample hash) and records it; a later one runs a seeded
#: :data:`ATTEST_SAMPLE_N`-file stat-and-hash sample (:func:`sample_generation`); the weekly `stores mirror --full` re-hashes everything.
#: The record ``<window root>/attestation.json`` (:data:`ATTESTATION_SCHEMA`) is written PER OPERATION; the shards check it in O(1)
#: (:func:`attestation_admits`) and refuse by name without it -- nothing is reused across operations but the knowledge that a full content
#: walk happened. The reader's per-input hash against the pinned manifest stays the byte guarantee (``cluster_reads.GenerationHashCache``).
ATTESTATION_SCHEMA = "store_attestation/1"
ATTESTED_SCHEMA = "store_attested/1"
ATTESTATION_FILE = "attestation.json"          # <window root>/attestation.json: this operation's record
ATTEST_FULL_WALK_MAX_AGE_S = 7 * 86400.0       # a generation's full content walk is good for a week of sampled attestations
ATTEST_SAMPLE_N = 64
ATTEST_THREADS = 4                              # the attestation job lists the pinned generations in parallel threads (NFS-bound)
#: THE NAME ESCAPE (measured 2026-10-04: VAST's nfs4.1 export refuses ':' in a name with EINVAL on mkdir and touch, single or double
#: colon, while '%3A' is accepted). Applied PER PATH COMPONENT, to a component containing ':' or '%': '%' -> '%25' first, then ':' -> '%3A'.
#: Escaping '%' too makes the rule a bijection (a true name `a%3Ab` lands as `a%253Ab`, never colliding with a true `a:b`), so the
#: destination spelling of any true name is computed, and the true name of any destination spelling is recovered, without a table.
#: The manifest lines, the stat snapshots, the VERIFIED / DEST-STAT files and every reader-facing index keep the TRUE names; the
#: generation on disk carries the ESCAPED names; MANIFEST-<sha8>.json records the affected entries as `name_escapes` (true -> escaped).
#: A SYMLINK'S TARGET is spelled the same way on the destination (:func:`escape_target`; `target_escapes` in the .json): its referent is
#: renamed there, so a link carrying the true target would dangle -- the shape the committed fixture pins
#: (`link:to -> d:x/sub%y/f:1.txt`), which the verify had passed because it compared the literal. The verify and the admission compare
#: the on-disk target with the escaped spelling, and the manifest keeps the true target (the content hashes are unchanged).
#: Affected entries -- every entry whose NAME needs the escape, and every symlink whose TARGET does -- travel by `tar --transform` with
#: the `s` flag (member names AND symlink targets rewritten, the same character substitutions, '/' untouched), the rest by rsync; the
#: cluster-read home tree lays one node-local link per escaped NAME (cluster_reads.home_tree_spec; a translated target needs nothing
#: there: the link resolves inside the generation).
#: the characters the destination cannot hold in a name, each spelled '%' + its two-digit hex on disk, '%' itself first (so the
#: spelling stays reversible). A run on 2026-10-04 measured ':' (EINVAL on VAST); a run on 2026-10-06 measured the rest of the set
#: with one `touch` per character in the drills namespace: `| < > " ? * \` refused (EINVAL), `% ; & # { } [ ] @ ! = + ,` accepted --
#: VAST applies the Windows-reserved set. fullscale_consolidated carries 477 attested names with '|' and was the component that found it.
DESTINATION_REFUSED_CHARS = (":", "|", "<", ">", '"', "?", "*", "\\")
NAME_ESCAPES: Dict[str, str] = {"%": "%25", **{c: "%%%02X" % ord(c) for c in DESTINATION_REFUSED_CHARS}}
NAME_ESCAPE_CHARS = tuple(NAME_ESCAPES)
NAME_ESCAPE_RULE = ("per path component containing '%' or a character the destination refuses (: | < > \" ? * \\ -- VAST's Windows-reserved set, "
                    "measured by R1 and the R2b run): '%' -> '%25' first, then each refused character -> '%' + its two-digit hex (':' -> '%3A', "
                    "'|' -> '%7C', ...), reversible; the generation on disk carries the escaped names and spells a symlink's in-tree target the "
                    "same way, every manifest and index the true names and the true target (registry-slurm-only C3; R1: VAST refuses ':' in a name; "
                    "the R2b run: '|' and the rest)")
_UNESCAPE_RE = re.compile("|".join(re.escape(e) for e in NAME_ESCAPES.values()))
_UNESCAPE: Dict[str, str] = {e: c for c, e in NAME_ESCAPES.items()}


def _sed_literal(ch: str) -> str:
    """One character as a literal in a GNU sed / tar `--transform` basic regular expression (`*` and `\\` are the specials in the set)."""
    return "\\" + ch if ch in "*\\.[]^$/" else ch


#: the GNU tar `--transform` expressions that spell :data:`NAME_ESCAPE_RULE` on extraction, '%' first; `s` = symlink targets rewritten
#: too (a `S` left `link%3Ato -> d:x/sub%y/f:1.txt` dangling beside its renamed referent; hard-link targets follow the default `h`)
TAR_TRANSFORMS = tuple(f"s/{_sed_literal(c)}/{e}/gs" for c, e in NAME_ESCAPES.items())
#: the destination-side serialization of every publisher of ONE slug: an exclusive flock on ``gen/<slug>/.publish.lock``.
#: Where the tree is (a local destination: the producer's own job, `--on-cluster`) it is held from before the snapshot (the manifest
#: build) until `current` has switched (:class:`PublishLock`); the grading-host-driven publisher, whose steps are one ssh each, takes
#: the SAME file for its publish step (:func:`locked_step`) and holds it across the readlink, the comparison with what it read at its
#: start and the rename -- so no publisher's compare-and-switch can interleave with another's switch (an earlier form read `current`
#: and renamed in one shell step but outside the lock, and an on-cluster switch between the two was overwritten). A waiter blocks up
#: to ``lock_wait_s`` (``lock_busy`` after that, naming the holder; nothing copied where the whole flow waits, `current` not switched
#: where the step waits). The SAME critical section requires the PUBLISHED record of the `current` it read
#: (``current_unrecorded`` when absent: a publisher is mid-flight or crashed there, unless an explicit adoption), applies the producer
#: rule to that record and to the generation's own, and writes the new generation's record (temp name + rename) BEFORE the switch -- so
#: no observer ever sees a `current` without its record, and nothing is read outside the lock that the switch depends on
PUBLISH_LOCK_FILE = ".publish.lock"
PUBLISH_LOCK_WAIT_S = 1800.0
PUBLISH_LOCK_POLL_S = 2.0
PUBLISH_LOCK_FD = 9        # the file descriptor a locked destination step holds the slug's publish lock on (:func:`locked_step`)


def escape_name(name: str) -> str:
    """One path component as the destination spells it (:data:`NAME_ESCAPE_RULE`); unchanged when it carries no escaped character.
    Character by character, so the order of substitutions cannot matter: a '%' in the true name becomes '%25' and never re-reads
    as the start of another escape."""
    if not any(c in name for c in NAME_ESCAPE_CHARS):
        return name
    return "".join(NAME_ESCAPES.get(c, c) for c in name)


def unescape_name(name: str) -> str:
    """The true name of one destination path component -- the exact inverse of :func:`escape_name`: every '%' on the destination is
    followed by one of the table's two-digit codes, decoded in ONE pass (so '%2525' is the true name '%25', never '%')."""
    if "%" not in name:
        return name
    return _UNESCAPE_RE.sub(lambda m: _UNESCAPE[m.group(0)], name)


def escape_rel(rel: str) -> str:
    return "/".join(escape_name(c) for c in rel.split("/"))


def unescape_rel(rel: str) -> str:
    return "/".join(unescape_name(c) for c in rel.split("/"))


def rel_needs_escape(rel: str) -> bool:
    return escape_rel(rel) != rel


def escape_target(target: str) -> str:
    """A symlink's attested (true, relative, in-tree -- :func:`link_escapes` refuses any other) target as the destination spells it:
    every path component escaped like a name (`.` and `..` carry neither ':' nor '%', so a climb is spelled the same). The link's
    referent lives under its escaped name there, so a link carrying the true target would dangle; :func:`verify_generation` and
    :func:`admit_generation` compare the on-disk target with this spelling, :func:`tar_copy_argv` writes it (tar's `s` flag)."""
    return escape_rel(target)


def escape_tops(rels: Iterable[str]) -> List[str]:
    """For every affected rel the shortest prefix that ends at its FIRST escaped component -- the subtrees rsync must leave alone (it
    cannot create their names on the destination) and tar carries; deduplicated, sorted by bytes."""
    tops = set()
    for rel in rels:
        parts = rel.split("/")
        for i, part in enumerate(parts):
            if escape_name(part) != part:
                tops.add("/".join(parts[:i + 1]))
                break
    return sorted(tops, key=_by_bytes)


class StoresError(Exception):
    """A configuration or transport error the command reports by name (never a silent skip)."""


class ManifestRefusal(StoresError):
    """The manifest of a canonical copy could not be built: a file changed while being hashed, a path is unhashable, ..."""


class CurrentMoved(StoresError):
    """The publish step found `current` naming something other than what this run read at its start: another publisher switched it
    meanwhile, and a run acting on that stale reading must not switch it back. Outcome ``current_moved``; the verified,
    sealed tree stays for a later run that reads the new `current` to publish through the same guard."""


class PublishLockBusy(StoresError):
    """The destination-side publish lock of a slug stayed held by another publisher for the whole wait (:class:`PublishLock`). Outcome
    ``lock_busy``: nothing was copied or published; the producer pushes again when the other publication has finished."""


class CurrentUnrecorded(StoresError):
    """`current` names a generation with no ``PUBLISHED-<sha8>.json`` beside it: a publisher is mid-flight or died
    between its switch and its record (the old flow), or the generation predates the record. Outcome ``current_unrecorded``: nothing is
    copied or switched -- wait for the publisher, or adopt the generation explicitly (``stores push --adopt-unrecorded``: the record is
    written under the publish lock without switching when it is this push's content; a differing content then supersedes it)."""

    def __init__(self, message: str, generation: Optional[str] = None):
        super().__init__(message)
        self.generation = generation


class ProducerMismatch(StoresError):
    """A grading-host-driven publication would switch `current` onto, or away from, a generation whose PUBLISHED record names a cluster producer
    while the component is not declared cluster-produced (:func:`_foreign_producer`); checked inside the locked publish step as well as
    before the copy. Outcome ``producer_mismatch``: nothing switched."""

    def __init__(self, message: str, generation: Optional[str] = None):
        super().__init__(message)
        self.generation = generation


# --------------------------------------------------------------------------------------------
# configuration: configs/registry/cluster_reads.yaml
# --------------------------------------------------------------------------------------------


@dataclass
class Component:
    slug: str
    base: str                                  # absolute, ``~`` expanded
    authority: str                             # cluster_a | cluster_a_origin | grading_host_copy (:data:`AUTHORITIES`)
    origin: str = ""                           # where the bytes were produced (informational: modal / cluster_b / grading_host / cluster_a / mixed)
    members: List[str] = field(default_factory=lambda: ["."])   # glob patterns relative to base ("." = the whole base)
    exclude: List[str] = field(default_factory=list)            # "/rel/path" anchored at base, or a basename pattern matched anywhere
    original: Optional[str] = None             # cluster_a_origin: the original on the destination (absolute, or relative to its shared root)
    adapters: List[str] = field(default_factory=list)
    note: str = ""
    producer: str = ""                         # grading_host:<session> | cluster | frozen (REQUIRED for `cluster_a`; informational otherwise)
    grading_host_mirror: Optional[str] = None        # cluster_a: the grading host's verified mirror of the generation (default: the base itself)
    transitional_carry: bool = False           # cluster_a + grading_host:* producer: the registry pushes on the producer's behalf until it adopts the push tool
    carry_deadline: Optional[str] = None       # cluster_a + transitional_carry: YYYY-MM-DD, the date the producer is expected to push itself
    min_publish_interval_s: Optional[int] = None   # cluster_a: a producer that writes every few minutes publishes at most this often

    @property
    def home_rel(self) -> Optional[str]:
        """The base's path relative to the grading host home (where the cluster-read task's relocated home mounts the generation)."""
        if self.base == HOME or self.base.startswith(HOME + os.sep):
            return os.path.relpath(self.base, HOME)
        return None

    @property
    def root_spelling(self) -> str:
        return ("~" + self.base[len(HOME):]) if self.home_rel is not None else self.base

    def anchored_excludes(self) -> set:
        return {e.strip("/") for e in self.exclude if e.startswith("/")}

    def name_excludes(self) -> List[str]:
        return [e for e in self.exclude if not e.startswith("/")]

    def member_tops(self) -> Tuple[List[str], List[str]]:
        """The declared members expanded ONCE for both the walk and the rsync source list: (the absolute tops, deduplicated,
        in declared order; the patterns that matched nothing)."""
        tops: List[str] = []
        unmatched: List[str] = []
        for pattern in self.members:
            matches = [self.base] if pattern in (".", "") else sorted(glob.glob(os.path.join(self.base, pattern)))
            if not matches:
                unmatched.append(pattern)
            for top in map(os.path.normpath, matches):
                if top not in tops:
                    tops.append(top)
        return tops, unmatched


#: git's administrative directory for a linked checkout's own git directory (``<main>/.git/<this>/<name>``): read to name the checkout
#: a linked worktree was added from, never a path into a worktree
LINKED_GITDIR_PARENT = "worktrees"


def checkout_dir_name(path: str) -> str:
    """The directory name of the checkout ``path`` is: a linked worktree (a ``.git`` FILE naming ``<main>/.git/worktrees/<name>``) names
    the checkout it was added from, anything else its own directory. The repo-mirror tool keys a clone on a cluster by this name
    (``<shared>/repos/<name>``), and :func:`pin_cluster_tree` hands the tool the same name, so a destination's default mirror is the
    clone the tool fills."""
    p = os.path.abspath(path).rstrip(os.sep) or os.sep
    dotgit = os.path.join(p, ".git")
    if os.path.isfile(dotgit):
        try:
            with open(dotgit, "r", encoding="utf-8") as fh:
                line = fh.read().strip()
        except OSError:
            line = ""
        if line.startswith("gitdir:"):
            gitdir = os.path.abspath(os.path.join(p, line[len("gitdir:"):].strip()))
            marker = f"{os.sep}.git{os.sep}{LINKED_GITDIR_PARENT}{os.sep}"
            if marker in gitdir:
                return os.path.basename(gitdir.split(marker)[0].rstrip(os.sep))
    return os.path.basename(p)


def mirror_selector(dest_name: str, environ: Optional[Mapping[str, str]] = None) -> str:
    """The name the site's repo-mirror tool knows the cluster with registry label ``dest_name`` by (``mirror-repos.sh --cluster <it>``;
    the tool names the clusters its own way, not by their registry labels): ``$MIRROR_REPOS_CLUSTER`` when ``$CLUSTER_DEST`` is that
    label (the pair the landing scripts export, scripts/registry/landing/site_env.sh), else the cluster block that carries the label
    -- its ``CLUSTER_MIRROR_SELECTOR``, else its site label -- among every block file in the site's clusters/ directory, listed in
    SISTEER_CLUSTERS or not (that list is the front door's default fan-out; a store destination names its cluster by registry label
    whichever clusters the front door fans out to), read through the site's cluster reader (reproduce/lib/clusters.py: load_addressable,
    resolve_label), else the label itself: no site config, no block carrying the label, or more than one. A name the tool does not know
    is refused by the tool, and :func:`pin_cluster_tree` then pushes the revision to the mirror itself."""
    env = os.environ if environ is None else environ
    if env.get("MIRROR_REPOS_CLUSTER") and env.get("CLUSTER_DEST") == dest_name:
        return str(env["MIRROR_REPOS_CLUSTER"])
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_site_clusters", os.path.join(REPO_ROOT, "reproduce", "lib", "clusters.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
        # the site config file: SISTEER_CONFIG, else SISTEER_SITE_DIR's config.env, else the reader's own default
        if env.get("SISTEER_CONFIG"):
            cfg_path = os.path.abspath(env["SISTEER_CONFIG"])
        elif env.get("SISTEER_SITE_DIR"):
            cfg_path = os.path.join(env["SISTEER_SITE_DIR"], "config.env")
        else:
            cfg_path = mod.default_config_path()
        blocks = mod.load_addressable(os.path.dirname(cfg_path), mod.site_config_file(cfg_path))
        label = mod.resolve_label(blocks, dest_name)
    except (Exception, SystemExit):          # no site config, one the reader refuses, or a label no block (or more than one) carries:
        return dest_name                     # the label itself, which the tool then judges
    return str(blocks[label].get("CLUSTER_MIRROR_SELECTOR") or label)


@dataclass
class Destination:
    name: str
    host: Optional[str]                        # None -> a local directory (tests); else the ssh alias of the cluster's login host
    shared: str                                # the cluster's shared root ($SHARED) or the local root, no trailing slash
    stores_rel: str = "registry/stores"
    repo_rel: str = ""                         # the repo mirror on the destination the running revision is pushed to (mirror-repos.sh);
                                               # empty -> repos/<source_repo's checkout name> (__post_init__, checkout_dir_name)
    source_repo: str = REPO_ROOT               # the checkout whose HEAD is the running code (the unit runs from the repository root)
    sbatch_rel: str = "scripts/registry/landing/store_verify.sbatch"
    log_rel: str = "exp/logs"                  # the verifier jobs' logs: <log dir>/<job name>_<job id>.out (the sbatch names no log path)
    watch_job: str = os.path.join(HOME, "scripts", "watch-job.sh")
    partition: str = "general"
    qos: str = "high"
    verify_cpus: int = 16
    verify_mem: str = "32G"
    verify_rate_mb_s: float = 100.0            # the default until a VERIFIED report has measured the slug's rate
    verify_min_time_s: int = 600
    verify_max_time_s: int = 4 * 3600
    rsync_timeout_s: int = 6 * 3600
    watch_max_hours: float = 4.0
    watch_poll_s: int = 30
    #: after the watch says DONE, the VERIFIED report is re-read every ``report_poll_s`` for up to ``report_wait_s`` while it is
    #: absent, stale or unreadable: a report the compute node wrote is not visible from the login node for up to ~60 s (NFS
    #: attribute / negative-entry caching; measured 2026-10-05: absent at 17-50 s after the job's End, present at 18-243 s)
    report_wait_s: float = 120.0
    report_poll_s: float = 15.0

    def __post_init__(self) -> None:
        if not self.repo_rel:
            self.repo_rel = f"repos/{checkout_dir_name(self.source_repo)}"

    @property
    def is_local(self) -> bool:
        return not self.host or self.host == "local"

    def resolve(self, path: str) -> str:
        """An absolute path as given, else ``path`` under the shared root."""
        return path if path.startswith("/") else f"{self.shared}/{path}"

    @property
    def stores_dir(self) -> str:
        return self.resolve(self.stores_rel)

    def gen_dir(self, slug: str) -> str:
        return f"{self.stores_dir}/gen/{slug}"

    def origin_manifest_dir(self, slug: str) -> str:
        return f"{self.stores_dir}/origin_manifests/{slug}"

    @property
    def sbatch_path(self) -> str:
        """The verifier in the repo mirror's working tree: the fallback when no revision is pinned (direct calls, tests)."""
        return self.resolve(f"{self.repo_rel}/{self.sbatch_rel}")

    @property
    def log_dir(self) -> str:
        """Where the verifier jobs write their logs: ``<shared root>/exp/logs`` by default, the site's job-log directory."""
        return self.resolve(self.log_rel)

    @property
    def mirror_dir(self) -> str:
        return self.resolve(self.repo_rel)

    @property
    def mirror_url(self) -> str:
        """Where ``git push`` sends the running revision: the mirror's path, over ssh for a remote destination."""
        return self.mirror_dir if self.is_local else f"{self.host}:{self.mirror_dir}"

    def tree_dir(self, revision: str) -> str:
        """``$SHARED/registry/stores/trees/<sha>/``: ``git archive <sha>`` of :data:`PINNED_PATHS`, extracted once, never rewritten."""
        return f"{self.stores_dir}/trees/{revision}"

    def pinned_sbatch(self, revision: str) -> str:
        return f"{self.tree_dir(revision)}/{self.sbatch_rel}"


@dataclass
class VolatileRoot:
    """A declared root OUTSIDE the home that no component can hold (a ``/tmp`` working tree, a volume peek): named in the configuration
    with the reason it is not a coverage gap. A root under the home is never volatile -- it is a component or it is UNCOVERED."""
    root: str
    reason: str


@dataclass
class AdapterEntry:
    name: str
    shards: int
    components: List[str]
    refused: bool = False                      # ``cluster: refused`` -- declared not readable on the cluster, with the reason
    reason: str = ""
    volatile_roots: List[VolatileRoot] = field(default_factory=list)
    note: str = ""


@dataclass
class Config:
    path: str
    destinations: Dict[str, Destination]
    components: Dict[str, Component]
    adapters: Dict[str, AdapterEntry]
    #: destinations that spell neither ``host`` nor ``shared``: name -> their other keys. Each takes its login host, shared root and repo
    #: mirror from the registry configuration's ``[[clusters]]`` entry of the same name when it is asked for (:meth:`destination`), so a
    #: read of the components alone (a cluster read shard, the coverage check) needs no registry configuration.
    joined: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: the registry root whose ``config.toml`` names the clusters (None: ``$REGISTRY_ROOT``, else the default root)
    root: Optional[str] = None

    def declared(self) -> List[str]:
        return sorted(set(self.destinations) | set(self.joined))

    def destination(self, name: Optional[str]) -> Destination:
        names = self.declared()
        if name is None:
            if len(names) != 1:
                raise StoresError(f"--dest is required (declared: {', '.join(names) or 'none'})")
            name = names[0]
        if name in self.destinations:
            return self.destinations[name]
        if name not in self.joined:
            raise StoresError(f"destination {name!r} is not declared in {self.path} (declared: {', '.join(names) or 'none'})")
        clusters = registry_clusters(self.root)
        entry = clusters.get(name)
        if entry is None:
            raise StoresError(f"{self.path}: destination {name!r} spells no `host` or `shared`, and the registry configuration "
                              f"{registry_config_path(self.root)} names no [[clusters]] entry {name!r} "
                              f"(declared: {', '.join(sorted(clusters)) or 'none'}); add one, or spell host and shared in {os.path.basename(self.path)}")
        opts = dict(self.joined[name])
        if "repo_rel" not in opts:
            shared, mirror = str(entry["shared"]), str(entry["mirror"])
            opts["repo_rel"] = mirror[len(shared.rstrip("/")) + 1:] if mirror.startswith(shared.rstrip("/") + "/") else mirror
        self.destinations[name] = Destination(name, (str(entry["host"]) if entry.get("host") else None), str(entry["shared"]), **opts)
        return self.destinations[name]

    def select(self, slugs: Optional[Sequence[str]]) -> List[Component]:
        if not slugs:
            return list(self.components.values())
        out = []
        for s in slugs:
            if s not in self.components:
                raise StoresError(f"component {s!r} is not declared in {self.path} (declared: {', '.join(self.components)})")
            out.append(self.components[s])
        return out


def config_path(explicit: Optional[str] = None) -> str:
    return os.path.abspath(os.path.expanduser(explicit or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG))


def _site_path(value, p, slug: str, key: str) -> str:
    """A component path from cluster_reads.yaml: a site placeholder ($STATE, $STAGE, $GRADE, $WORK, ...) expanded through
    sisteer.site, then `~`; an unset or unknown placeholder refuses by name (StoresError), never a literal `$STATE` path. A
    value with no `$` is a plain path (`~` expanded) and needs no site module."""
    if "$" not in str(value):
        return os.path.expanduser(str(value))
    from .. import site as _site
    try:
        return str(_site.expand_path(str(value), key=f"components.{slug}.{key}"))
    except _site.PlaceholderError as exc:
        raise StoresError(f"{p}: {exc}") from exc


def registry_config_path(root: Optional[str] = None) -> str:
    from . import config as RC
    return RC.config_path(_registry_root(root))


def _registry_root(root: Optional[str]) -> str:
    return os.path.abspath(os.path.expanduser(root or os.environ.get("REGISTRY_ROOT") or DEFAULT_ROOT))


def registry_clusters(root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The registry configuration's ``[[clusters]]`` entries by name (each cluster's registry label): its login host, its shared root and
    the path of its repo mirror (the entry's ``remote_src`` without ``/src``, else ``<shared>/repos/<the directory name of [repo]
    checkout>``). ``root`` is the registry root (default ``$REGISTRY_ROOT``, else the default root, as the CLI resolves it)."""
    from . import config as RC
    try:
        cfg = RC.load(_registry_root(root))
    except (ValueError, OSError) as exc:          # a ConfigError, or a file that is not TOML
        raise StoresError(f"{registry_config_path(root)}: the registry configuration names the clusters a destination joins, and it is "
                          f"refused: {exc}") from exc
    out: Dict[str, Dict[str, Any]] = {}
    for c in cfg.get("clusters") or []:
        if not c.get("name") or not c.get("shared"):
            continue
        shared = str(c["shared"]).rstrip("/") or "/"
        src = RC.remote_src(dict(c, shared=shared), cfg).rstrip("/")          # the entry's remote_src, else <shared>/repos/<checkout name>/src
        out[str(c["name"])] = {"host": c.get("host"), "shared": shared, "mirror": src[: -len("/src")] if src.endswith("/src") else src}
    return out


def load_config(path: Optional[str] = None, root: Optional[str] = None) -> Config:
    """Read and validate ``cluster_reads.yaml`` (PyYAML, imported here only so the cluster-side entry points stay stdlib). A destination
    that spells neither ``host`` nor ``shared`` takes both, and its ``repo_rel``, from the registry configuration's ``[[clusters]]`` entry of
    the same name (:func:`registry_clusters` of ``root``) when it is asked for (:meth:`Config.destination`), which refuses one that names
    no such entry by name."""
    import yaml
    p = config_path(path)
    with open(p, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict) or doc.get("schema") not in CONFIG_SCHEMAS:
        raise StoresError(f"{p}: schema {(doc or {}).get('schema') if isinstance(doc, dict) else None!r} is not one of {CONFIG_SCHEMAS}")
    dests: Dict[str, Destination] = {}
    joined: Dict[str, Dict[str, Any]] = {}
    known = {f.name for f in Destination.__dataclass_fields__.values()} - {"name", "host", "shared"}
    for name, d in (doc.get("destinations") or {}).items():
        d = dict(d or {})
        host = d.pop("host", None)
        shared = d.pop("shared", None)
        bad = sorted(set(d) - known)
        if bad:
            raise StoresError(f"{p}: destination {name!r} has unknown keys {bad}")
        for key in ("watch_job", "source_repo"):
            if key in d:
                d[key] = os.path.abspath(os.path.expanduser(str(d[key])))
        if host is None and shared is None:
            joined[str(name)] = d              # the cluster of this registry label: its [[clusters]] entry, when asked for
            continue
        if not shared:
            raise StoresError(f"{p}: destination {name!r} lacks `shared`")
        # the per-user shared root, as config.py spells it
        shared = os.path.expanduser(str(shared)).replace("{user}", getpass.getuser()).rstrip("/") or "/"
        dests[str(name)] = Destination(str(name), (str(host) if host else None), shared, **d)
    comps: Dict[str, Component] = {}
    for slug, c in (doc.get("components") or {}).items():
        c = dict(c or {})
        bad = sorted(set(c) - COMPONENT_KEYS)
        if bad:
            raise StoresError(f"{p}: component {slug!r} has unknown keys {bad} (known: {sorted(COMPONENT_KEYS)})")
        base = c.get("base")
        if not base:
            raise StoresError(f"{p}: component {slug!r} lacks `base`")
        base = os.path.abspath(_site_path(base, p, slug, "base"))
        authority = str(c.get("authority") or "")
        if authority not in AUTHORITIES:
            raise StoresError(f"{p}: component {slug!r} authority {authority!r} is not one of {AUTHORITIES}")
        original = c.get("original")
        if authority == "cluster_a_origin" and not original:
            raise StoresError(f"{p}: component {slug!r} is cluster_a_origin and names no `original`")
        if authority != "cluster_a_origin" and original:
            raise StoresError(f"{p}: component {slug!r} is {authority} and must not name an `original` (a cluster_a_origin key)")
        producer = str(c.get("producer") or "")
        if authority == "cluster_a" and not producer:
            raise StoresError(f"{p}: component {slug!r} is cluster_a and names no `producer` (grading_host:<session> | cluster | frozen)")
        if producer and not _PRODUCER_RE.match(producer):
            raise StoresError(f"{p}: component {slug!r} producer {producer!r} is not grading_host:<session>, cluster or frozen")
        if authority != "cluster_a":
            stray = [k for k in CLUSTER_A_ONLY_KEYS if k in c]
            if stray:
                raise StoresError(f"{p}: component {slug!r} is {authority} and carries {stray}, which belong to authority cluster_a")
        grading_host_mirror = c.get("grading_host_mirror")
        # a cluster_a component's grading host mirror: the declared path, else the base (a grading-host-side or frozen producer's own tree); a CLUSTER
        # producer's default is the registry-owned pull target under the registry root, resolved at run time (:func:`grading_host_mirror_path`)
        grading_host_mirror = (os.path.abspath(_site_path(grading_host_mirror, p, slug, "grading_host_mirror")) if grading_host_mirror
                         else (base if authority == "cluster_a" and producer != "cluster" else None))
        carry = c.get("transitional_carry", False)
        if not isinstance(carry, bool):
            raise StoresError(f"{p}: component {slug!r} transitional_carry {carry!r} is not a boolean")
        deadline = c.get("carry_deadline")
        if carry:
            if not producer.startswith("grading_host:"):
                raise StoresError(f"{p}: component {slug!r} declares transitional_carry with producer {producer!r}; a carry is for a grading host:<session> "
                                  f"producer that has not adopted the push tool")
            if not deadline:
                raise StoresError(f"{p}: component {slug!r} declares transitional_carry and no `carry_deadline` (YYYY-MM-DD)")
        if deadline is not None:
            deadline = str(deadline)
            if not carry:
                raise StoresError(f"{p}: component {slug!r} names carry_deadline {deadline!r} without transitional_carry: true")
            if not _DATE_RE.match(deadline):
                raise StoresError(f"{p}: component {slug!r} carry_deadline {deadline!r} is not YYYY-MM-DD")
        interval = c.get("min_publish_interval_s")
        if interval is not None:
            if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
                raise StoresError(f"{p}: component {slug!r} min_publish_interval_s {interval!r} is not a non-negative integer")
        members = [str(m) for m in (c.get("members") or ["."])]
        exclude = [str(e) for e in (c.get("exclude") or [])]
        for e in exclude:
            if "\n" in e or e.strip("/") == "":
                raise StoresError(f"{p}: component {slug!r} exclude {e!r} is not a path or a name pattern")
        comps[str(slug)] = Component(str(slug), base, authority, origin=str(c.get("origin") or ""), members=members, exclude=exclude,
                                     original=(str(original) if original else None), adapters=[], note=str(c.get("note") or ""),
                                     producer=producer, grading_host_mirror=grading_host_mirror, transitional_carry=carry, carry_deadline=deadline,
                                     min_publish_interval_s=(int(interval) if interval is not None else None))
    adapters: Dict[str, AdapterEntry] = {}
    for name, a in (doc.get("adapters") or {}).items():
        a = dict(a or {})
        bad = sorted(set(a) - ADAPTER_KEYS)
        if bad:
            raise StoresError(f"{p}: adapter {name!r} has unknown keys {bad} (known: {sorted(ADAPTER_KEYS)})")
        note = str(a.get("note") or "")
        if "cluster" in a:
            # declared NOT readable on the cluster, with the reason written down: no shards, no components, no volatile roots
            if a.get("cluster") != CLUSTER_REFUSED:
                raise StoresError(f"{p}: adapter {name!r} cluster {a.get('cluster')!r} is not {CLUSTER_REFUSED!r} (the only value; omit the key otherwise)")
            reason = str(a.get("reason") or "").strip()
            if not reason:
                raise StoresError(f"{p}: adapter {name!r} is cluster: refused and gives no `reason`")
            stray = [k for k in ("shards", "components", "volatile_roots") if k in a]
            if stray:
                raise StoresError(f"{p}: adapter {name!r} is cluster: refused and carries {stray}: a refused adapter reads on no cluster")
            adapters[str(name)] = AdapterEntry(str(name), 0, [], refused=True, reason=reason, note=note)
            continue
        if "reason" in a:
            raise StoresError(f"{p}: adapter {name!r} gives a `reason` without cluster: refused (use `note` for prose)")
        shards = int(a.get("shards") or 0)
        if shards < 1:
            raise StoresError(f"{p}: adapter {name!r} declares shards {shards}; at least 1")
        names = [str(x) for x in (a.get("components") or [])]
        if not names:
            raise StoresError(f"{p}: adapter {name!r} names no component (every adapter that reads on the cluster reads a component; "
                              f"declare `cluster: refused` with a reason otherwise)")
        for x in names:
            if x not in comps:
                raise StoresError(f"{p}: adapter {name!r} names component {x!r}, which is not declared")
            comps[x].adapters.append(str(name))
        volatile: List[VolatileRoot] = []
        for v in (a.get("volatile_roots") or []):
            if not isinstance(v, Mapping) or not v.get("root") or not str(v.get("reason") or "").strip():
                raise StoresError(f"{p}: adapter {name!r} volatile_roots entry {v!r} is not {{root: <absolute path>, reason: <why>}}")
            vr = os.path.abspath(os.path.expanduser(str(v["root"])))
            if vr == HOME or vr.startswith(HOME + os.sep):
                raise StoresError(f"{p}: adapter {name!r} volatile root {vr} lies under the home {HOME}: a root under the home is a component, "
                                  f"never a volatile exception")
            volatile.append(VolatileRoot(vr, str(v["reason"]).strip()))
        adapters[str(name)] = AdapterEntry(str(name), shards, names, volatile_roots=volatile, note=note)
    return Config(p, dests, comps, adapters, joined, root)


def grading_host_mirror_path(root: Optional[str], comp: Component) -> str:
    """Where the grading host's copy of a component lives: the declared ``grading_host_mirror``; else, for a ``cluster_a`` component produced on the
    cluster, the registry-owned pull target ``<root>/stores/mirror/<slug>`` (:data:`MIRROR_SUBDIR`; the registry root the CLI resolved,
    so a scratch ``--root`` pulls under itself and never into the live root); else the base."""
    if comp.grading_host_mirror:
        return comp.grading_host_mirror
    if comp.authority == "cluster_a" and comp.producer == "cluster":
        return os.path.join(root or DEFAULT_ROOT, MIRROR_SUBDIR, comp.slug)
    return comp.base


# --------------------------------------------------------------------------------------------
# the component walk, the stat cache and the manifest
# --------------------------------------------------------------------------------------------


def stat_key(st: os.stat_result) -> Tuple[int, int, int, int]:
    return (int(st.st_size), int(st.st_mtime_ns), int(st.st_ino), int(st.st_ctime_ns))


def _by_bytes(rel: str) -> bytes:
    return rel.encode("utf-8", "surrogateescape")


@dataclass
class Walk:
    files: List[Tuple[str, str, os.stat_result]]      # (rel, abs, lstat) sorted by rel bytes
    links: List[Tuple[str, str, os.stat_result, str]] = field(default_factory=list)   # (rel, abs, lstat, literal target), sorted
    other: int = 0
    unmatched_members: List[str] = field(default_factory=list)
    #: every directory the walk visited -> its WHOLE listing (:func:`listing.listing_entries`; excluded entries included), '' the base:
    #: the manifest's additive ``directories`` field (MANIFEST_RULES), empty directories included
    dirs: Dict[str, Entries] = field(default_factory=dict)

    def stats(self) -> List[Tuple[str, os.stat_result]]:
        """(rel, lstat) of every attested entry -- files and links -- the stat snapshots and :func:`stat_changes` cover."""
        return [(rel, st) for rel, _p, st in self.files] + [(rel, st) for rel, _p, st, _t in self.links]


def link_escapes(rel: str, target: str) -> bool:
    """Does a symlink at ``rel`` (base-relative) with the literal ``target`` resolve outside the base? Absolute targets and
    targets that climb above the base do; the copy would resolve them to bytes no manifest attests."""
    if os.path.isabs(target):
        return True
    resolved = os.path.normpath(os.path.join(os.path.dirname(rel), target))
    return resolved == ".." or resolved.startswith("../")


def exclusion_test(comp: Component) -> Callable[[str], bool]:
    """The component's exclude rules as a predicate over a base-relative path: anchored excludes match the path or any prefix of it,
    name excludes match any component of it (the walk's rule, and the staged-listing expectation's: :func:`listing_mismatches`)."""
    anchored = comp.anchored_excludes()
    names = comp.name_excludes()

    def excluded(rel: str) -> bool:
        if not anchored and not names:
            return False
        parts = rel.split("/")
        for i in range(1, len(parts) + 1):
            if "/".join(parts[:i]) in anchored:
                return True
        return any(fnmatch.fnmatchcase(part, pat) for part in parts for pat in names)

    return excluded


def staged_listing(rel: str, attested: Iterable[Sequence[str]], excluded: Optional[Callable[[str], bool]] = None) -> Entries:
    """What a staged generation's directory ``rel`` is expected to list, from the canonical's attested listing: the entries the copy
    deliberately leaves out -- the component's excluded paths and ``other`` kinds (sockets, fifos: rsync carries neither) -- removed.
    A reader's listing input on a shard still fingerprints the WHOLE staged listing and is compared with the WHOLE attestation, so a
    directory that holds an excluded or ``other`` entry is never a shard's to publish (``runner.ClusterListingRefusal``: mismatch)."""
    out = []
    for name, kind in normalize_entries(attested):
        if kind == "other":
            continue
        if excluded is not None and excluded(f"{rel}/{name}" if rel else name):
            continue
        out.append((name, kind))
    return tuple(out)


def manifest_component(meta: Mapping[str, Any], base: str) -> Component:
    """A :class:`Component` standing for a manifest's declared ``members`` / ``exclude`` over another base (a generation on the node):
    the exclusion rules the staged-listing expectation needs."""
    return Component(str(meta.get("slug") or os.path.basename(base.rstrip("/"))), base, str(meta.get("authority") or "grading_host_copy"),
                     members=list(meta.get("members") or ["."]), exclude=list(meta.get("exclude") or []))


def walk_component(comp: Component) -> Walk:
    """Every regular file and every symlink of the component: the declared members (globs relative to the base) expanded in
    sorted order, each directory walked without following symlinks, excludes applied; sorted by the relpath's UTF-8 bytes."""
    if not os.path.isdir(comp.base):
        raise ManifestRefusal(f"{comp.slug}: base {comp.base} is not a directory")
    excluded = exclusion_test(comp)

    walk = Walk([])

    def take(rel: str, p: str, st: os.stat_result) -> None:
        if statmod.S_ISLNK(st.st_mode):
            walk.links.append((rel, p, st, os.readlink(p)))
        elif statmod.S_ISREG(st.st_mode):
            walk.files.append((rel, p, st))
        else:
            walk.other += 1

    tops, walk.unmatched_members = comp.member_tops()
    for top in tops:
        rel_top = "" if top == comp.base else os.path.relpath(top, comp.base)
        if ".." in rel_top.split("/"):          # a path COMPONENT, never a prefix: a member named `..cache` is legitimate
            raise ManifestRefusal(f"{comp.slug}: member {top} resolves outside the base {comp.base}")
        if rel_top and excluded(rel_top):
            continue
        st = os.lstat(top)
        if not statmod.S_ISDIR(st.st_mode):
            take(rel_top, top, st)
            continue
        for dirpath, dirnames, filenames in os.walk(top, topdown=True, followlinks=False):
            rel_dir = os.path.relpath(dirpath, comp.base)
            rel_dir = "" if rel_dir == "." else rel_dir
            walk.dirs[rel_dir] = listing_entries(dirpath)       # the canonical directory's whole listing, empty directories included
            keep = []
            for d in sorted(dirnames):
                rd = f"{rel_dir}/{d}" if rel_dir else d
                if excluded(rd):
                    continue
                dp = os.path.join(dirpath, d)
                if os.path.islink(dp):
                    take(rd, dp, os.lstat(dp))       # a directory link is attested by its target, never followed
                    continue
                keep.append(d)
            dirnames[:] = keep
            for fn in sorted(filenames):
                rel = f"{rel_dir}/{fn}" if rel_dir else fn
                if excluded(rel):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    take(rel, p, os.lstat(p))
                except FileNotFoundError:
                    continue          # vanished between the listing and the stat: the post-copy walk decides
    walk.files.sort(key=lambda t: _by_bytes(t[0]))
    walk.links.sort(key=lambda t: _by_bytes(t[0]))
    return walk


def escaping_links(comp: Component) -> List[Tuple[str, str]]:
    """The symlink census a push applies, as a stat walk and nothing more: every attested symlink of the component (declared members,
    excludes applied) whose literal target :func:`link_escapes` refuses -- absolute, or climbing above the base -- as ``(rel, target)``
    in manifest order. :func:`build_manifest` refuses the whole component at the first such link, AFTER hashing nothing; this is the
    same test over the whole walk, so a precert can list every refusal of every component before any push is attempted and without
    reading a byte. The R2b precert (2026-10-06) counted links and built manifests only for bases under 2 GiB, so four refusals
    (hpo_llama, hpo_r1, grade_strength, full22_seedrep) were found by the pushes instead; this is the census it should have run."""
    walk = walk_component(comp)
    return [(rel, target) for rel, _p, _st, target in walk.links if link_escapes(rel, target)]


def stat_changes(walk: Walk, snapshot: Mapping[str, Sequence[int]],
                 key: Callable[[os.stat_result], Tuple[int, ...]] = stat_key) -> Tuple[List[str], List[str], List[str]]:
    """The walk (files and links) against a stat snapshot ``{rel: [key fields]}`` -- the ONE comparison behind the stat
    cache, the post-copy invalidation, the admission re-validation and the destination guard. Returns (moved, missing,
    extra): entries in both whose key differs, entries only the snapshot names, entries only the walk found; each sorted
    by the relpath's bytes."""
    keys = {rel: list(key(st)) for rel, st in sorted(walk.stats(), key=lambda t: _by_bytes(t[0]))}
    moved = [rel for rel, k in keys.items() if rel in snapshot and list(snapshot[rel]) != k]
    missing = sorted((rel for rel in snapshot if rel not in keys), key=_by_bytes)
    extra = [rel for rel in keys if rel not in snapshot]
    return moved, missing, extra


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_text(path: str, text: str) -> None:
    """Atomic: a PER-PROCESS unique temp name, ``fsync``, ``os.replace`` -- the shape of ``mirrors._write_json_atomic`` (a
    fixed ``<path>.tmp`` let two writers interleave into one file, the ledger, 2026-09-19); stdlib, so the cluster side shares it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_json(path: str, data: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")


def _stamp(utc: str) -> str:
    """A registry UTC stamp as a file-name suffix: digits, ``T`` and ``Z`` only."""
    return re.sub(r"[^0-9TZ]", "", utc)


def component_state_dir(root: str, slug: str) -> str:
    return os.path.join(root, "state", STATE_SUBDIR, slug)


def manifest_lines(entries: Iterable[Tuple[str, int, str]]) -> str:
    """``<sha256>  <size>  <relpath>`` per file (:data:`MANIFEST_RULES`)."""
    return "".join(f"{sha}  {size}  {rel}\n" for sha, size, rel in entries)


def two_manifest_hashes(entries: Iterable[Tuple[str, int, str]], links: Iterable[Tuple[str, str]] = ()) -> Tuple[str, str]:
    """The pair (``name_size_sha256``, ``content_sha256``) over the sorted file entries, then the sorted symlink
    lines (:data:`MANIFEST_RULES`): a retargeted link is a different generation."""
    ns = hashlib.sha256()
    ct = hashlib.sha256()
    for sha, size, rel in entries:
        ns.update(f"{rel} {size}\n".encode("utf-8", "surrogateescape"))
        ct.update(f"{sha}  {rel}\n".encode("utf-8", "surrogateescape"))
    for rel, target in links:
        line = f"{rel} -> {target}\n".encode("utf-8", "surrogateescape")
        ns.update(line)
        ct.update(line)
    return ns.hexdigest(), ct.hexdigest()


def parse_manifest_lines(text: str, where: str) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        parts = line.split("  ", 2)
        if len(parts) != 3 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]) or not parts[1].isdigit():
            raise StoresError(f"{where}:{n}: not a manifest line")
        out.append((parts[0], int(parts[1]), parts[2]))
    return out


def read_manifest_lines(path: str) -> List[Tuple[str, int, str]]:
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
        return parse_manifest_lines(fh.read(), path)


def manifest_json_path(manifest_path: str) -> str:
    """``MANIFEST-<sha8>.json`` beside a ``MANIFEST-<sha8>.sha256`` (or any ``<prefix>.sha256``)."""
    return manifest_path[:-len(".sha256")] + ".json" if manifest_path.endswith(".sha256") else manifest_path + ".json"


def read_manifest_symlinks(manifest_path: str) -> Dict[str, str]:
    """The symlinks a manifest attests (``symlinks: {rel: target}`` in its ``.json``); ``{}`` when the ``.json`` is absent."""
    meta = _load_json(manifest_json_path(manifest_path)) or {}
    links = meta.get("symlinks")
    return {str(k): str(v) for k, v in links.items()} if isinstance(links, dict) else {}


def read_manifest_escapes(manifest_path: str) -> Dict[str, str]:
    """The name escapes a manifest records (``name_escapes: {true rel: escaped rel}`` in its ``.json``; :data:`NAME_ESCAPE_RULE`). A
    manifest written before the rule existed records none -- and had none: VAST refused every such name, so no generation holds one."""
    meta = _load_json(manifest_json_path(manifest_path)) or {}
    esc = meta.get("name_escapes")
    return {str(k): str(v) for k, v in esc.items()} if isinstance(esc, dict) else {}


def attestation_json_path(manifest_path: str, listings_sha256: Optional[str] = None) -> str:
    """The attestation keyed by its listings beside a ``MANIFEST-<sha8>.sha256``: ``MANIFEST-<sha8>.<lsha8>.json`` (``lsha8`` the first eight
    hex digits of ``listings_sha256``; MANIFEST_RULES["directories"]), or the plain ``MANIFEST-<sha8>.json`` when no listings are named."""
    base = manifest_path[:-len(".sha256")] if manifest_path.endswith(".sha256") else manifest_path
    return f"{base}.{str(listings_sha256)[:8]}.json" if listings_sha256 else base + ".json"


def read_manifest_attestation(manifest_path: str, listings_sha256: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The attestation (a manifest ``.json``) a pin names, or None when it is not on disk. With ``listings_sha256`` the keyed file,
    else the plain file when IT carries that ``listings_sha256`` (a content whose first attestation predates the keyed files); never
    "the latest attestation of that content", which a rebuild of the same files over other listings would have made the wrong one.
    With no listings named, the plain file."""
    if listings_sha256:
        meta = _load_json(attestation_json_path(manifest_path, listings_sha256))
        if meta:
            return meta
        plain = _load_json(manifest_json_path(manifest_path))
        return plain if plain and plain.get("listings_sha256") == listings_sha256 else None
    return _load_json(manifest_json_path(manifest_path)) or None


def attested_directories(meta: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Entries]]:
    """The directory listings an attestation carries (``directories: {rel: [[name, kind], ...]}``); None when there is no attestation or
    it carries no field -- a manifest written before the field attests NO listing (never ``{}``, which would read as "a tree with no
    directory")."""
    dirs = (meta or {}).get("directories")
    if not isinstance(dirs, dict):
        return None
    return {str(rel): normalize_entries(ents) for rel, ents in dirs.items() if isinstance(ents, list)}


def read_manifest_directories(manifest_path: str, listings_sha256: Optional[str] = None) -> Optional[Dict[str, Entries]]:
    """:func:`attested_directories` of :func:`read_manifest_attestation` -- the plain ``.json``, or the attestation ``listings_sha256`` names."""
    return attested_directories(read_manifest_attestation(manifest_path, listings_sha256))


def listing_mismatches(gen_path: str, attested: Mapping[str, Entries], excluded: Optional[Callable[[str], bool]] = None,
                       create_empty: bool = False) -> Tuple[List[str], List[str]]:
    """Every attested directory whose listing on disk under ``gen_path`` is not what the stage is expected to list (:func:`staged_listing`:
    the attestation less the component's excluded entries and ``other`` kinds), by relpath ('' the tree itself); with ``create_empty``
    an attested directory expected EMPTY that is absent is created first (rsync carries empty directories, the forced files-from pass
    does not, and the attestation is what verifies them) and named in the second list. The tree's entries beyond the attested
    directories are the file walk's."""
    mismatched: List[str] = []
    created: List[str] = []
    wants = {rel: staged_listing(rel, attested[rel], excluded) for rel in attested}
    # the attestation spells the canonical's TRUE names; the destination spells an escaped entry per NAME_ESCAPE_RULE (an earlier verify
    # compared the two spellings directly and refused every escaped generation `listing_mismatch`). An attested directory is found at its
    # ESCAPED path and its on-disk listing is read back in true names before the comparison -- two spellings of one name on disk read as
    # a duplicate entry, a mismatch
    paths = {rel: (os.path.join(gen_path, escape_rel(rel)) if rel else gen_path) for rel in attested}
    if create_empty:
        # every creation BEFORE any comparison: a parent's listing names the empty child the copy left out, so comparing the parent first
        # would refuse a tree the creation pass is about to complete. Nothing is created inside a PUBLISHED generation: its directories are
        # sealed (DIR_SEAL_MODE, no write bit), and a re-attestation's verify cannot -- and must not -- add one. The seal is READ, never
        # probed: a process the kernel does not refuse (root / CAP_DAC_OVERRIDE) would otherwise create inside the sealed tree and attest a
        # listing neither attestation describes. A creation refused for any other reason (another owner's directory, a read-only mount) is
        # not an error either: the directory stays absent and the comparison below names it, which is the re-attestation's
        # `listing_only_drift`
        for rel in sorted(attested, key=_by_bytes):
            if not wants[rel] and not os.path.lexists(paths[rel]):
                if _under_sealed_directory(paths[rel]):
                    continue
                try:
                    os.makedirs(paths[rel], exist_ok=True)
                except OSError:
                    continue
                created.append(rel)
    for rel in sorted(attested, key=_by_bytes):
        d = paths[rel]
        if not os.path.isdir(d) or os.path.islink(d) or _true_listing(d) != wants[rel]:
            mismatched.append(rel)
    return mismatched, created


def _under_sealed_directory(path: str) -> bool:
    """Whether creating ``path`` (absent) would create inside a SEALED directory: the nearest existing ancestor is a directory with no write
    bit at all (the publish seal, :data:`DIR_SEAL_MODE`; a user, group or other write bit means the kernel decides). Read from the mode, so
    the answer is the same whatever the caller's privilege; an ancestor that cannot be stat'ed answers False and the kernel decides."""
    parent = os.path.dirname(path) or os.curdir
    while not os.path.lexists(parent):
        up = os.path.dirname(parent) or os.curdir
        if up == parent:
            return False
        parent = up
    try:
        st = os.stat(parent)
    except OSError:
        return False
    return statmod.S_ISDIR(st.st_mode) and not (st.st_mode & 0o222)


def _true_listing(directory: str) -> Entries:
    """A destination directory's listing in TRUE names (:func:`unescape_name` of each on-disk entry), the form the attestation spells."""
    return normalize_entries((unescape_name(n), k) for n, k in listing_entries(directory))


@dataclass
class Manifest:
    component: Component
    entries: List[Tuple[str, int, str]]        # (sha256, size, rel) sorted by rel bytes
    links: List[Tuple[str, str]]               # (rel, literal target) sorted by rel bytes; none escapes the base
    snapshot: Dict[str, List[int]]             # rel -> [size, mtime_ns, ino, ctime_ns] at build time, files and links
    name_size_sha256: str
    content_sha256: str
    built_utc: str
    built_host: str
    hashed: int
    from_cache: int
    other_skipped: int
    unmatched_members: List[str]
    directories: Dict[str, Entries] = field(default_factory=dict)   # rel -> the canonical directory's whole listing (MANIFEST_RULES["directories"])
    listings_sha256: str = ""                                       # listings_sha256(directories)

    @property
    def slug(self) -> str:
        return self.component.slug

    @property
    def n_directories(self) -> int:
        return len(self.directories)

    @property
    def n_empty_directories(self) -> int:
        return sum(1 for ents in self.directories.values() if not ents)

    @property
    def content_sha8(self) -> str:
        return self.content_sha256[:8]

    @property
    def n_files(self) -> int:
        return len(self.entries)

    @property
    def bytes(self) -> int:
        return sum(size for _s, size, _r in self.entries)

    def text(self) -> str:
        return manifest_lines(self.entries)

    @property
    def rels(self) -> List[str]:
        """Every attested rel -- files then links -- in manifest order (true names)."""
        return [rel for _s, _z, rel in self.entries] + [rel for rel, _t in self.links]

    def name_escapes(self) -> Dict[str, str]:
        """``{true rel: escaped rel}`` for every attested entry whose destination spelling differs (:data:`NAME_ESCAPE_RULE`)."""
        return {rel: escape_rel(rel) for rel in self.rels if rel_needs_escape(rel)}

    def target_escapes(self) -> Dict[str, str]:
        """``{rel: escaped target}`` for every attested symlink whose TARGET the destination spells differently (:func:`escape_target`):
        the links tar must carry whatever their own names, because rsync would write the true target, which dangles beside the renamed
        referent."""
        return {rel: escape_target(target) for rel, target in self.links if escape_target(target) != target}

    def tar_carried(self) -> Tuple[List[str], List[str]]:
        """What travels by :func:`tar_copy_argv` instead of rsync: (the rels -- every entry whose name needs the escape and every symlink
        whose target does, sorted by bytes; the rsync EXCLUDES -- :func:`escape_tops` of the escaped names and of the escaped attested
        directories plus, by exact path, each plain-named link with an escaped target)."""
        names = self.name_escapes()
        targets = self.target_escapes()
        carried = sorted(set(names) | set(targets), key=_by_bytes)
        # an attested directory whose name needs the escape and holds no carried entry (an EMPTY one; MANIFEST_RULES["directories"]) is
        # excluded from rsync as well -- rsync would try its literal name on the destination -- and nothing carries it: the verify creates
        # it at its escaped path (:func:`listing_mismatches`, ``create_empty``)
        dir_tops = escape_tops(rel for rel in self.directories if rel and rel_needs_escape(rel))
        excludes = sorted(set(escape_tops(names)) | set(dir_tops) | {rel for rel in targets if rel not in names}, key=_by_bytes)
        return carried, excludes

    def max_mtime_utc(self, rels: Optional[Iterable[str]] = None) -> Optional[str]:
        """The latest ``mtime_ns`` among ``rels`` (default: every attested file) from the build-time stat snapshot, as a registry UTC
        stamp -- the producer's committed store change a PUBLISHED record carries as ``change_utc``."""
        from datetime import datetime, timezone
        picks = [rel for rel in (rels if rels is not None else (r for _s, _z, r in self.entries)) if rel in self.snapshot]
        if not picks:
            return None
        ns = max(int(self.snapshot[rel][1]) for rel in picks)        # STAT_FIELDS: size, mtime_ns, ino, ctime_ns
        return R._fmt_utc(datetime.fromtimestamp(ns / 1e9, timezone.utc))

    def summary(self, files_path: str) -> Dict[str, Any]:
        """The state file's digest of a manifest (``latest_manifest`` / ``grading_host``): hashes, stamp, counts, where the lines are."""
        return {"content_sha8": self.content_sha8, "content_sha256": self.content_sha256, "name_size_sha256": self.name_size_sha256,
                "built_utc": self.built_utc, "n_files": self.n_files, "bytes": self.bytes, "hashed": self.hashed, "from_cache": self.from_cache,
                "files": files_path, "listings_sha256": self.listings_sha256 or None,
                "attestation": attestation_json_path(files_path, self.listings_sha256)}

    def meta(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        c = self.component
        return {
            "schema": MANIFEST_SCHEMA, "slug": self.slug, "n_files": self.n_files, "bytes": self.bytes,
            "name_size_sha256": self.name_size_sha256, "content_sha256": self.content_sha256, "content_sha8": self.content_sha8,
            "built_utc": self.built_utc, "built_host": self.built_host, "hashed": self.hashed, "from_cache": self.from_cache,
            "n_symlinks": len(self.links), "symlinks": dict(self.links), "other_skipped": self.other_skipped,
            "unmatched_members": self.unmatched_members,
            "name_escapes": self.name_escapes(), "n_name_escapes": len(self.name_escapes()), "name_escape_rule": NAME_ESCAPE_RULE,
            "target_escapes": self.target_escapes(), "n_target_escapes": len(self.target_escapes()),
            "n_directories": self.n_directories, "n_empty_directories": self.n_empty_directories, "listings_sha256": self.listings_sha256,
            "directories": directories_json(self.directories),
            "manifest_rules": MANIFEST_RULES, "files": f"MANIFEST-{self.content_sha8}.sha256",
            "stat_snapshot": f"MANIFEST-{self.content_sha8}.stat.json",
            "root": c.root_spelling, "base": c.base, "home_rel": c.home_rel, "authority": c.authority, "origin": c.origin,
            "members": c.members, "exclude": c.exclude, "original": c.original, "adapters": c.adapters,
            **(extra or {}),
        }


def _hash_many(paths: Sequence[str], hasher: Callable[[str], str], workers: int) -> List[str]:
    if workers <= 1 or len(paths) < 2:
        return [hasher(p) for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(hasher, paths))


def build_manifest(comp: Component, state_dir: Optional[str], hasher: Callable[[str], str] = R.file_sha256, workers: int = 1,
                   full: bool = False, write_cache: bool = True) -> Manifest:
    """The canonical copy's manifest through the stat cache ``<state_dir>/stat_cache.json``: a file whose
    ``(size, mtime_ns, ino, ctime_ns)`` equal its cache entry keeps the cached sha and is not read; every other file is
    hashed, re-stat'ed, hashed ONCE more if it moved meanwhile, and refused by name if it is still moving. The cache is
    pruned to the files seen and saved atomically. ``state_dir=None`` disables the cache (the cluster side, ``manifest-only``).
    ``full`` IGNORES the cache's entries -- every file is read (the weekly `--full`; the refreshed cache is still saved);
    ``write_cache=False`` reads the cache and never writes it (a dry run writes nothing)."""
    walk = walk_component(comp)
    if not walk.files:
        raise ManifestRefusal(f"{comp.slug}: no regular file under {comp.base} for members {comp.members} (excludes {comp.exclude})")
    for rel, _st in walk.stats():
        if "\n" in rel or "\r" in rel:
            raise ManifestRefusal(f"{comp.slug}: path {rel!r} carries a line break; the manifest cannot spell it")
    links = [(rel, target) for rel, _p, _st, target in walk.links]
    for rel, target in links:
        if "\n" in target or "\r" in target:
            raise ManifestRefusal(f"{comp.slug}: symlink {rel!r} target carries a line break; the manifest cannot spell it")
        if link_escapes(rel, target):
            raise ManifestRefusal(f"{comp.slug}: symlink {rel} -> {target} escapes the component base {comp.base}; "
                                  f"the copy would resolve it to bytes no manifest attests (exclude it or point it inside)")
    cache_path = os.path.join(state_dir, STAT_CACHE) if state_dir else None
    cache: Dict[str, Any] = (_load_json(cache_path) or {}) if cache_path else {}
    cached = {} if full else {rel: ent for rel, ent in cache.items() if isinstance(ent, dict) and isinstance(ent.get("sha256"), str)}
    moved, _gone, new = stat_changes(walk, {rel: [ent.get(k) for k in STAT_FIELDS] for rel, ent in cached.items()})
    stale = set(moved) | set(new)
    to_hash = [(rel, p, st) for rel, p, st in walk.files if rel in stale]
    fresh = {rel: cached[rel] for rel, _p, _st in walk.files if rel not in stale}     # the cache, pruned to the files seen
    hashed = 0
    for (rel, p, st), sha in zip(to_hash, _hash_many([p for _r, p, _s in to_hash], hasher, workers)):
        after = os.lstat(p)
        if stat_key(after) != stat_key(st):
            sha = hasher(p)                       # moved once across the hash: read it once more
            hashed += 1
            st, after = after, os.lstat(p)
            if stat_key(after) != stat_key(st):
                raise ManifestRefusal(f"{comp.slug}: {rel} changed while being hashed (twice); the canonical copy is being written")
        hashed += 1
        fresh[rel] = dict(zip(STAT_FIELDS, stat_key(after)), sha256=sha)
    entries = [(fresh[rel]["sha256"], fresh[rel]["size"], rel) for rel, _p, _s in walk.files]
    snapshot = {rel: [fresh[rel][k] for k in STAT_FIELDS] for rel, _p, _s in walk.files}
    snapshot.update({rel: list(stat_key(st)) for rel, _p, st, _t in walk.links})
    ns, ct = two_manifest_hashes(entries, links)
    if cache_path and write_cache and fresh != cache:
        _write_json(cache_path, fresh)
    return Manifest(comp, entries, links, snapshot, ns, ct, R.utc_now(), socket.gethostname(), hashed, len(walk.files) - len(to_hash),
                    walk.other, walk.unmatched_members, dict(walk.dirs), listings_sha256(walk.dirs))


def write_manifest_files(m: Manifest, out_dir: str) -> str:
    """``MANIFEST-<sha8>.sha256`` + ``.stat.json`` (the stat snapshot :func:`revalidate` needs; both are functions of the content, so a
    rewrite restates nothing) and the ATTESTATIONS under ``out_dir``: the plain ``MANIFEST-<sha8>.json`` when none exists yet (the FIRST
    attestation of this content) and ``MANIFEST-<sha8>.<lsha8>.json`` when none exists for these listings -- written once, never rewritten
    (MANIFEST_RULES["directories"]): a pin names the attestation it was verified against and :func:`revalidate` reads THAT one, so a
    rebuild of the same files over other listings adds an attestation and changes none. A build whose (content, listings) pair is already
    attested writes nothing. Returns the ``.sha256`` path."""
    base = os.path.join(out_dir, f"MANIFEST-{m.content_sha8}")
    _write_text(base + ".sha256", m.text())
    if not os.path.isfile(base + ".json"):
        _write_json(base + ".json", m.meta())
    if m.listings_sha256:
        keyed = attestation_json_path(base + ".sha256", m.listings_sha256)
        if not os.path.isfile(keyed):
            _write_json(keyed, m.meta())
    _write_json(base + ".stat.json", {"schema": MANIFEST_SCHEMA + "/stat", "slug": m.slug, "content_sha8": m.content_sha8,
                                      "built_utc": m.built_utc, "files": m.snapshot})
    return base + ".sha256"


# --------------------------------------------------------------------------------------------
# the cluster side: verify a generation against its manifest; manifest-only of an in-place original
# --------------------------------------------------------------------------------------------


def _walk_tree(top: str) -> Walk:
    """The component walk over a bare directory (no members, no excludes): a generation's own walk, keyed by TRUE names -- the
    destination spells an escaped entry per :data:`NAME_ESCAPE_RULE`, and every manifest, snapshot and index spells the true name, so
    the rels are unescaped here (the paths stay the on-disk ones, and so does a symlink's TARGET: the callers compare it with
    :func:`escape_target` of the attested one, which is what a correctly written link carries)."""
    top = os.path.abspath(top)
    walk = walk_component(Component(os.path.basename(top), top, "grading_host_copy"))
    walk.files = [(unescape_rel(rel), p, st) for rel, p, st in walk.files]
    walk.links = [(unescape_rel(rel), p, st, target) for rel, p, st, target in walk.links]
    walk.files.sort(key=lambda t: _by_bytes(t[0]))
    walk.links.sort(key=lambda t: _by_bytes(t[0]))
    return walk


def _walk_plain_dir(top: str) -> Dict[str, os.stat_result]:
    """Every regular file under ``top`` as rel -> lstat."""
    return {rel: st for rel, _p, st in _walk_tree(top).files}


def dest_stat_key(st: os.stat_result) -> Tuple[int, int, int, int, int]:
    """:data:`DEST_STAT_FIELDS` of an entry of a verified generation: the mode records the seal; ctime is in the key for a
    regular file with ONE link (a same-length rewrite that restores mode and mtime still moves ctime) and 0 otherwise (a
    later generation hard-linking the inode moves ctime and nlink without touching a byte)."""
    one_link = statmod.S_ISREG(st.st_mode) and int(st.st_nlink) == 1
    return (int(st.st_size), int(st.st_mtime_ns), int(st.st_ino), int(st.st_mode), int(st.st_ctime_ns) if one_link else 0)


def _seal_files(walk: Walk) -> set:
    """``chmod a-w`` every regular file of a generation before it is hashed. Directories stay writable so a copy can resume
    into the tree and a failed tree can be renamed; an inode a published generation already sealed (a hard link) is left
    alone. Returns the rels sealed."""
    sealed = set()
    for rel, p, st in walk.files:
        if st.st_mode & 0o222:
            os.chmod(p, statmod.S_IMODE(st.st_mode) & ~0o222)
            sealed.add(rel)
    return sealed


def verify_generation(gen_path: str, manifest_path: str, workers: int = 16, job_id: Optional[str] = None,
                      seal: bool = True, attestation: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
    """Seal the generation's files, hash EVERY manifest entry (bytes, never inode identity), re-read every attested symlink
    and walk the tree for entries the manifest does not name. ``ok`` iff no missing, corrupt or extra file or link; the
    recomputed two-manifest hashes ride along. Returns (the VERIFIED report, the destination stat snapshot of every file
    and link -- :data:`DEST_STAT_FIELDS`, taken after the seal -- that :func:`admit_generation` checks against).

    The directory listings verified are the ones ``attestation`` names -- the path of a manifest ``.json``; default the plain
    ``MANIFEST-<sha8>.json`` beside the lines, the attestation the publish's verify runs against. A RE-ATTESTATION (`stores attest`,
    :func:`attest_listings`) names the keyed ``MANIFEST-<sha8>.<lsha8>.json`` of the canonical's current listings and writes its report
    to ``VERIFIED-<sha8>.<lsha8>.json``; the report's ``dest_stat`` is then keyed the same way (``DEST-STAT-<sha8>.<lsha8>.json``), so the
    live snapshot the admission checks against is never rewritten by a verify that did not publish. When listings are attested the report
    also names the DIRECTION of any difference between the attested directory set and the tree's: ``dirs_generation_only`` (directories
    the tree holds that the attestation does not name) and ``dirs_canonical_only`` (attested directories absent from the tree) -- what
    the re-attestation's refusal reads."""
    t0 = time.monotonic()
    entries = read_manifest_lines(manifest_path)
    want_links = read_manifest_symlinks(manifest_path)
    att_path = attestation or manifest_json_path(manifest_path)
    # the attested LISTINGS (MANIFEST_RULES["directories"]): an attested directory expected empty that the copy did not reproduce is created,
    # then every attested directory's listing on disk must be the canonical's less the entries the copy leaves out (the component's excludes,
    # `other` kinds); a manifest without the field attests none and verifies as before
    meta = _load_json(att_path) or {}
    want_dirs = attested_directories(meta)
    excluded = exclusion_test(manifest_component(meta, gen_path))
    listing_bad, created_dirs = (listing_mismatches(gen_path, want_dirs, excluded, create_empty=True) if (want_dirs is not None and os.path.isdir(gen_path))
                                 else ([], []))
    walk = _walk_tree(gen_path) if os.path.isdir(gen_path) else Walk([])
    sealed = _seal_files(walk) if seal else set()
    present = {rel: (p, st) for rel, p, st in walk.files}
    links = {rel: target for rel, _p, _st, target in walk.links}
    named = {rel for _s, _z, rel in entries}
    missing = [rel for _s, _z, rel in entries if rel not in present] + sorted((rel for rel in want_links if rel not in links), key=_by_bytes)
    extra = sorted([rel for rel in present if rel not in named] + [rel for rel in links if rel not in want_links], key=_by_bytes)
    to_hash = [(sha, size, rel) for sha, size, rel in entries if rel in present]
    corrupt = []
    hashed_bytes = 0
    # the on-disk path of an entry (its name escaped on the destination), never the true rel joined onto the tree
    for (sha, size, rel), got in zip(to_hash, _hash_many([present[rel][0] for _s, _z, rel in to_hash], R.file_sha256, workers)):
        st_size = present[rel][1].st_size
        hashed_bytes += st_size
        if got != sha or st_size != size:
            corrupt.append(rel)
    # a link is corrupt unless its on-disk target is the DESTINATION spelling of the attested one (escape_target): the literal true
    # target dangles beside a renamed referent, and a verify that accepted it published a link no reader could open
    corrupt += sorted((rel for rel, target in want_links.items()
                       if rel in links and (links[rel] != escape_target(target) or link_escapes(rel, links[rel]))), key=_by_bytes)
    ns, ct = two_manifest_hashes(entries, sorted(want_links.items(), key=lambda t: _by_bytes(t[0])))
    dest_stat = {rel: list(dest_stat_key(os.lstat(p) if rel in sealed else st)) for rel, (p, st) in present.items()}
    dest_stat.update({rel: list(dest_stat_key(st)) for rel, _p, st, _t in walk.links})
    # the direction of a listing difference (true names; the tree's directories are spelled escaped on disk): what a re-attestation
    # refuses by, and what an operator reads
    tree_dirs = {unescape_rel(rel) for rel in walk.dirs} if want_dirs is not None else set()
    gen_only = sorted(tree_dirs - set(want_dirs or {}), key=_by_bytes) if want_dirs is not None else []
    canon_only = sorted(set(want_dirs or {}) - tree_dirs, key=_by_bytes) if want_dirs is not None else []
    lsha = listings_sha256(want_dirs) if want_dirs is not None else None
    # a verify against an explicit attestation never rewrites the live DEST-STAT-<sha8>.json (the snapshot the admission checks the sealed
    # tree against): its snapshot is keyed like the attestation
    dest_stat_name = f"DEST-STAT-{ct[:8]}.json" if attestation is None else f"DEST-STAT-{ct[:8]}.{(lsha or 'unattested')[:8]}.json"
    elapsed = time.monotonic() - t0
    report = {"schema": VERIFIED_SCHEMA, "generation": os.path.abspath(gen_path), "manifest": os.path.abspath(manifest_path),
              "attestation": os.path.abspath(att_path),
              "content_sha8": ct[:8], "content_sha256": ct, "name_size_sha256": ns, "n_files": len(entries), "n_symlinks": len(want_links),
              "bytes": sum(z for _s, z, _r in entries), "ok": not (missing or corrupt or extra or listing_bad), "missing": missing, "corrupt": corrupt,
              "extra": extra, "hashed": len(to_hash), "hashed_bytes": hashed_bytes, "sealed": len(sealed), "elapsed_s": round(elapsed, 3),
              "name_escapes": len(read_manifest_escapes(manifest_path)),
              "target_escapes": sum(1 for target in want_links.values() if escape_target(target) != target),
              "n_directories": (len(want_dirs) if want_dirs is not None else None), "listings_attested": want_dirs is not None,
              "listings_sha256": lsha, "listing_mismatch": listing_bad,
              "created_dirs": created_dirs,
              "dirs_generation_only": gen_only[:LIST_CAP], "n_dirs_generation_only": len(gen_only),
              "dirs_canonical_only": canon_only[:LIST_CAP], "n_dirs_canonical_only": len(canon_only),
              "rate_mb_s": round(hashed_bytes / 1e6 / elapsed, 3) if elapsed > 0 else None, "workers": workers,
              "dest_stat": dest_stat_name, "dest_stat_fields": list(DEST_STAT_FIELDS),
              "host": socket.gethostname(), "job_id": job_id, "verified_utc": R.utc_now()}
    return report, dest_stat


def write_verified(out: str, report: Mapping[str, Any], dest_stat: Mapping[str, Sequence[int]]) -> str:
    """``DEST-STAT-<sha8>.json`` beside the VERIFIED report, then the report itself (its presence is the completion signal);
    returns the snapshot's path."""
    spath = os.path.join(os.path.dirname(os.path.abspath(out)), str(report["dest_stat"]))
    _write_json(spath, {"schema": VERIFIED_SCHEMA + "/dest_stat", "generation": report["generation"], "content_sha8": report["content_sha8"],
                        "fields": list(DEST_STAT_FIELDS), "files": dest_stat})
    _write_json(out, report)
    return spath


@dataclass
class Admission:
    generation: str
    refusal: Optional[str] = None       # None = admitted; else no_manifest | no_verified_report | verified_mismatch | no_dest_stat |
                                        # unsealed | sealed_mismatch | unsealed_directories | no_attestation | generation_changed |
                                        # listing_changed | content_changed | changed_during_hash
    reason: str = ""
    n_files: int = 0
    n_symlinks: int = 0
    moved: int = 0
    sampled: int = 0
    hashed: int = 0
    mismatched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    elapsed_s: float = 0.0
    n_directories: Optional[int] = None                 # attested directory listings checked (None: the manifest attests none)
    listing_mismatched: List[str] = field(default_factory=list)
    listings_sha256: Optional[str] = None               # the attestation the listings were checked against (the pin's, else the plain .json's)

    @property
    def ok(self) -> bool:
        return self.refusal is None

    def as_dict(self) -> Dict[str, Any]:
        d = {"generation": self.generation, "ok": self.ok, **asdict(self)}
        d.update(mismatched=self.mismatched[:LIST_CAP], missing=self.missing[:LIST_CAP], extra=self.extra[:LIST_CAP],
                 listing_mismatched=self.listing_mismatched[:LIST_CAP], elapsed_s=round(self.elapsed_s, 3))
        return d


def admit_generation(gen_dir: str, sha8: str, sample_fraction: float = SAMPLE_FRACTION, sample_min: int = SAMPLE_MIN,
                     seed: Optional[int] = None, listings_sha256: Optional[str] = None) -> Admission:
    """The destination change guard, a fast PRECHECK (stdlib; the cluster-read task runs it beside :func:`revalidate`
    before it reads a pinned generation and again when the read completes). ``gen/<slug>/<sha8>`` is admitted when:
    ``MANIFEST-<sha8>.sha256`` is there; ``VERIFIED-<sha8>.json`` is ``ok`` and both hashes equal the manifest's;
    ``DEST-STAT-<sha8>.json`` is there; ``SEALED-<sha8>.json`` is there and names this generation, and the generation's top
    directory is not writable; a stat walk of the tree against it (:data:`DEST_STAT_FIELDS`) finds no missing /
    extra entry; every symlink's target equals the attested one and stays inside the tree; every file whose stat moved and
    a seeded sample (``max(sample_min, ceil(sample_fraction * n))``) re-hash to the manifest's sha. Any difference refuses
    BY NAME.

    THE SEAL GATES ADMISSION as it gates publication (the final pass on 2c). A same-sha8 dangle repair whose directory seal failed is
    copied, verified and left WRITABLE, with the previous tree's SEALED record moved aside beside its reports -- a tree the mirror
    calls ``current_unsealed`` (red) and publishes nothing for. Read on its VERIFIED and DEST-STAT alone it looked admissible, so a
    window that had pinned it read it; ``unsealed`` / ``sealed_mismatch`` refuse it by name instead. The top directory's mode is the
    CHEAP half of the seal (one ``lstat``): every directory's mode and the recorded count are the attestation's check
    (:func:`attest_generation` step (c)), and a tree un-sealed deeper down is still that check's to name.

    What it does NOT guarantee: a hard-linked file (nlink > 1; its ctime is out of the key) rewritten with same-length bytes
    and its mode + mtime restored, outside the seeded sample, passes. The byte guarantee is the reader's: every input it
    reads is hashed (``adapters/base.py check_input``) and compared with :func:`manifest_index`'s entry; see there."""
    t0 = time.monotonic()
    gen_path = f"{gen_dir}/{sha8}"
    ad = Admission(gen_path, seed=int(sha8, 16) if seed is None else seed)

    def refuse(name: str, reason: str) -> Admission:
        ad.refusal, ad.reason, ad.elapsed_s = name, reason, time.monotonic() - t0
        return ad

    mpath = f"{gen_dir}/MANIFEST-{sha8}.sha256"
    if not os.path.isfile(mpath):
        return refuse("no_manifest", f"no {mpath}")
    entries = read_manifest_lines(mpath)
    want = {rel: (sha, size) for sha, size, rel in entries}
    want_links = read_manifest_symlinks(mpath)
    ad.n_files, ad.n_symlinks = len(entries), len(want_links)
    ns, ct = two_manifest_hashes(entries, sorted(want_links.items(), key=lambda t: _by_bytes(t[0])))
    report = _load_json(f"{gen_dir}/VERIFIED-{sha8}.json") or {}
    if not report:
        return refuse("no_verified_report", f"no VERIFIED-{sha8}.json under {gen_dir}; the generation was never verified")
    if not (report.get("ok") is True and report.get("content_sha256") == ct and report.get("name_size_sha256") == ns and ct[:8] == sha8
            and not (report.get("missing") or report.get("corrupt") or report.get("extra"))):
        return refuse("verified_mismatch", f"VERIFIED-{sha8}.json is not an ok report of manifest {ct[:8]} (ok={report.get('ok')!r})")
    snap = _load_json(f"{gen_dir}/DEST-STAT-{sha8}.json") or {}
    if not isinstance(snap.get("files"), dict):
        return refuse("no_dest_stat", f"no DEST-STAT-{sha8}.json under {gen_dir}; the seal cannot be checked")
    sealed = _load_json(f"{gen_dir}/SEALED-{sha8}.json") or {}
    if not sealed:
        return refuse("unsealed", f"no SEALED-{sha8}.json under {gen_dir}: no publication of THIS tree sealed its directories (a repair whose "
                                  f"seal did not complete is verified and writable, the tree it replaced having taken its seal aside); the "
                                  f"mirror publishes nothing for it -- `stores push` or `stores mirror --adopt-unrecorded` seals it")
    if sealed.get("content_sha8") != sha8:
        return refuse("sealed_mismatch", f"SEALED-{sha8}.json names content {sealed.get('content_sha8')}, not {sha8}")
    if not os.path.isdir(gen_path):
        return refuse("generation_changed", f"{gen_path} is not a directory")
    top_mode = statmod.S_IMODE(os.lstat(gen_path).st_mode)
    if top_mode & 0o222:
        return refuse("unsealed_directories", f"{gen_path} is mode {top_mode:04o}, not {DIR_SEAL_MODE:04o}: the directory seal is not on the "
                                              f"generation's top directory, so entries can be added or removed under it")
    walk = _walk_tree(gen_path)
    moved, ad.missing, ad.extra = stat_changes(walk, snap["files"], key=dest_stat_key)
    if ad.missing or ad.extra:
        return refuse("generation_changed", f"{len(ad.missing)} verified entr(y/ies) missing, {len(ad.extra)} unverified entr(y/ies) present under {gen_path}")
    # the attested directory listings (MANIFEST_RULES["directories"]) of the attestation the PIN names (``listings_sha256``; the plain .json when
    # none is named): an empty directory removed or an entry added under the tree is a change; a tree re-copied under another attestation
    # of the same content is not this pin's
    attestation = read_manifest_attestation(mpath, listings_sha256)
    if listings_sha256 and attestation is None:
        return refuse("no_attestation", f"no attestation of generation {sha8} with listings {listings_sha256[:8]} under {gen_dir}; the pin names one this destination never received")
    ad.listings_sha256 = (attestation or {}).get("listings_sha256")
    want_dirs = attested_directories(attestation)
    if want_dirs is not None:
        ad.n_directories = len(want_dirs)
        ad.listing_mismatched, _created = listing_mismatches(gen_path, want_dirs, exclusion_test(manifest_component(attestation or {}, gen_path)))
        if ad.listing_mismatched:
            return refuse("listing_changed", f"{len(ad.listing_mismatched)} director(y/ies) under {gen_path} list other than manifest {sha8} attests (first: {ad.listing_mismatched[0]!r})")
    ad.moved = len(moved)
    moved_set = set(moved)
    files = {rel: (p, st) for rel, p, st in walk.files}
    links = {rel: target for rel, _p, _st, target in walk.links}
    # every link's on-disk target must be the destination spelling of the attested one (escape_target), and stay inside the tree
    ad.mismatched += sorted((rel for rel, target in links.items()
                             if rel not in want_links or escape_target(want_links[rel]) != target or link_escapes(rel, target)), key=_by_bytes)
    pool = [rel for rel in want if rel not in moved_set]
    k = min(len(entries), max(sample_min, math.ceil(sample_fraction * len(entries))))
    sample = random.Random(ad.seed).sample(pool, min(k, len(pool)))
    ad.sampled = len(sample)
    for rel in [r for r in moved if r in files] + sample:
        p, st = files[rel]
        sha = R.file_sha256(p)
        ad.hashed += 1
        if dest_stat_key(os.lstat(p)) != dest_stat_key(st):
            return refuse("changed_during_hash", f"{rel} changed while being hashed; the generation is being written")
        if (sha, st.st_size) != want[rel]:
            ad.mismatched.append(rel)
    if ad.mismatched:
        return refuse("content_changed", f"{len(ad.mismatched)} entr(y/ies) differ from manifest {sha8} (first: {ad.mismatched[0]})")
    ad.elapsed_s = time.monotonic() - t0
    return ad


@dataclass
class ManifestIndex:
    """The verified generation's ``{relpath: (sha256, size)}`` -- what the READER compares every input's computed sha with.
    The cluster-reads worker installs one per pinned generation in the pass-scoped seam and passes the entry's sha to
    ``check_input(path, expected_sha256=...)`` (``adapters/base.py``), which already hashes every input at discovery and
    refuses a mismatch by name; :meth:`check` is the same rule for a caller that hashed the bytes itself."""
    gen_dir: str
    sha8: str
    content_sha256: str
    name_size_sha256: str
    files: Dict[str, Tuple[str, int]]
    symlinks: Dict[str, str]
    directories: Optional[Dict[str, Entries]] = None   # the attested listings (MANIFEST_RULES["directories"]); None: the manifest attests none
    listings_sha256: Optional[str] = None              # whose attestation: the pin's, else the plain .json's

    @property
    def tree(self) -> str:
        return f"{self.gen_dir}/{self.sha8}"

    @property
    def listings_attested(self) -> bool:
        return self.directories is not None

    def attested_listing(self, rel: str) -> Optional[Entries]:
        """The canonical listing the manifest attests for the directory ``rel`` under the tree ('' the tree itself), or None when the
        manifest attests no listings or none for that directory -- what a shard compares a listing input's fingerprint with
        (``runner.ClusterListingRefusal``)."""
        if self.directories is None:
            return None
        return self.directories.get(rel)

    def expected(self, relpath: str) -> Optional[Tuple[str, int]]:
        return self.files.get(relpath)

    def check(self, relpath: str, sha256: str, size: Optional[int] = None) -> Optional[str]:
        """None when the computed ``sha256`` (and ``size``, when given) equal the verified manifest's entry for ``relpath``;
        else the refusal, by name, for the reader to file against that input."""
        want = self.files.get(relpath)
        if want is None:
            return f"{relpath}: not in the verified generation {self.sha8}'s manifest ({len(self.files)} files); nothing approved these bytes"
        if sha256 != want[0] or (size is not None and int(size) != want[1]):
            return (f"{relpath}: bytes differ from the verified generation {self.sha8} (sha {sha256[:12]}... != manifest {want[0][:12]}..."
                    f"{'' if size is None or int(size) == want[1] else f'; size {size} != {want[1]}'})")
        return None

    def path_of(self, relpath: str) -> str:
        """The on-disk path of a true rel inside the generation (its name escaped per :data:`NAME_ESCAPE_RULE`)."""
        return os.path.join(self.tree, escape_rel(relpath))

    def check_file(self, relpath: str) -> Optional[str]:
        """Hash the entry's on-disk path now and :meth:`check` it -- a reader that does not hash for itself."""
        p = self.path_of(relpath)
        try:
            st = os.stat(p)
        except OSError as exc:
            return f"{relpath}: cannot be read from the verified generation {self.sha8}: {type(exc).__name__}: {exc}"
        return self.check(relpath, R.file_sha256(p), st.st_size)


def manifest_index(gen_dir: str, sha8: str, listings_sha256: Optional[str] = None) -> ManifestIndex:
    """The pinned generation's manifest as an index, loaded from ``gen/<slug>/MANIFEST-<sha8>.sha256`` (+ the symlinks in its
    ``.json``) on the cluster and APPROVED: its recomputed hashes must equal the ones the VERIFIED report hashed every
    entry against, and the report must be ``ok``. Raises :class:`StoresError` by name (``no_manifest``, ``manifest_mismatch``,
    ``no_verified_report``, ``verified_mismatch``, ``no_attestation``) when they do not -- an index nobody verified is never handed
    to a reader. The directory listings are the attestation the pin names (``listings_sha256``; :func:`read_attestation`), never the
    latest ``.json`` of that content."""
    mpath = f"{gen_dir}/MANIFEST-{sha8}.sha256"
    if not os.path.isfile(mpath):
        raise StoresError(f"no_manifest: no {mpath}")
    entries = read_manifest_lines(mpath)
    links = read_manifest_symlinks(mpath)
    ns, ct = two_manifest_hashes(entries, sorted(links.items(), key=lambda t: _by_bytes(t[0])))
    if ct[:8] != sha8:
        raise StoresError(f"manifest_mismatch: {mpath} hashes to {ct[:8]}, not {sha8}")
    report = _load_json(f"{gen_dir}/VERIFIED-{sha8}.json") or {}
    if not report:
        raise StoresError(f"no_verified_report: no VERIFIED-{sha8}.json under {gen_dir}; the generation was never verified")
    if not (report.get("ok") is True and report.get("content_sha256") == ct and report.get("name_size_sha256") == ns
            and not (report.get("missing") or report.get("corrupt") or report.get("extra"))):
        raise StoresError(f"verified_mismatch: VERIFIED-{sha8}.json is not an ok report of manifest {sha8} (ok={report.get('ok')!r})")
    attestation = read_manifest_attestation(mpath, listings_sha256)
    if listings_sha256 and attestation is None:
        raise StoresError(f"no_attestation: no attestation of generation {sha8} with listings {listings_sha256[:8]} under {gen_dir}")
    return ManifestIndex(gen_dir, sha8, ct, ns, {rel: (sha, size) for sha, size, rel in entries}, links, attested_directories(attestation),
                         (attestation or {}).get("listings_sha256"))


# --------------------------------------------------------------------------------------------
# admission by attestation, once per read operation: the listing, the seal, the content, the record
# --------------------------------------------------------------------------------------------


def _implied_dirs(rels: Iterable[str]) -> set:
    """Every proper ancestor directory of the given base-relative paths -- the directories a manifest's files and links imply."""
    out: set = set()
    for rel in rels:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            out.add("/".join(parts[:i]))
    return out


def list_generation(gen_path: str, entries: Sequence[Tuple[str, int, str]], want_links: Mapping[str, str], seal_check: bool = True) -> Dict[str, Any]:
    """The NAMESPACE check of a generation: ONE recursive ``os.scandir`` listing -- names, ``d_type`` and every
    symlink's target; no ``stat``, no hash -- compared with the manifest's entries and attested links, then the directory seal (``lstat``
    of every directory, mode == :data:`DIR_SEAL_MODE`). On-disk names are the destination's escaped spelling (:data:`NAME_ESCAPE_RULE`) and
    are unescaped before the comparison, so the manifest's TRUE names are what is compared. Differences, by name: ``missing`` (a manifest
    file or link not on disk), ``extra`` (a file or link the manifest does not name), ``retyped`` (a manifest entry present under another
    type), ``link_mismatch`` (an on-disk target that is not the destination spelling of the attested one -- :func:`escape_target` -- or
    escapes the tree), ``other_entries`` (neither file, directory nor symlink). A DIRECTORY that no manifest entry implies is counted
    (``extra_dirs``) and is NOT a mismatch on its own: the mirror copies a member directory whole (``rsync -r``), so an empty source
    directory lands in the generation with no manifest line attesting it; its contents, if any, are ``extra`` entries, and the directory
    COUNT is checked against the SEALED record by :func:`attest_generation` (an added or removed directory moves it). ``ok`` is the
    listing's verdict alone; the seal's is ``n_unsealed_dirs`` (``.`` names the generation root). The listing of the largest
    live generation (86,978 entries) measured 22.8 s cold over NFS, plus 8.6 s for the directory modes -- against the 837-s per-shard stat walk it replaces."""
    t0 = time.monotonic()
    gen_path = os.path.abspath(gen_path)
    files: set = set()
    dirs: set = set()
    links: Dict[str, str] = {}
    other: List[str] = []
    dir_paths: List[str] = [gen_path]
    n_entries = 0
    stack: List[Tuple[str, str]] = [(gen_path, "")]
    while stack:
        d, prefix = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                n_entries += 1
                rel = (prefix + "/" if prefix else "") + unescape_name(e.name)
                if e.is_symlink():
                    links[rel] = os.readlink(e.path)
                elif e.is_dir(follow_symlinks=False):
                    dirs.add(rel)
                    dir_paths.append(e.path)
                    stack.append((e.path, rel))
                elif e.is_file(follow_symlinks=False):
                    files.add(rel)
                else:
                    other.append(rel)
    t1 = time.monotonic()
    want_files = {rel for _s, _z, rel in entries}
    want_link_set = set(want_links)
    link_set = set(links)
    missing = sorted(want_files - files, key=_by_bytes) + sorted(want_link_set - link_set, key=_by_bytes)
    extra = sorted(files - want_files, key=_by_bytes) + sorted(link_set - want_link_set, key=_by_bytes)
    retyped = sorted((want_files & (dirs | link_set)) | (want_link_set & (files | dirs)), key=_by_bytes)
    link_mismatch = sorted((rel for rel, t in links.items() if rel in want_links and (t != escape_target(want_links[rel]) or link_escapes(rel, t))),
                           key=_by_bytes)
    extra_dirs = sorted(dirs - _implied_dirs(list(want_files) + list(want_link_set)), key=_by_bytes)
    unsealed: List[str] = []
    t2 = t1
    if seal_check:
        for p in dir_paths:
            if statmod.S_IMODE(os.lstat(p).st_mode) != DIR_SEAL_MODE:
                unsealed.append(os.path.relpath(p, gen_path))
        t2 = time.monotonic()
    ok = not (missing or extra or retyped or link_mismatch or other)
    return {"generation": gen_path, "entries": n_entries, "files": len(files), "dirs": len(dir_paths), "symlinks": len(links), "other": len(other),
            "elapsed_s": round(t1 - t0, 3), "ok": ok,
            "missing": missing[:LIST_CAP], "extra": extra[:LIST_CAP], "retyped": retyped[:LIST_CAP], "link_mismatch": link_mismatch[:LIST_CAP],
            "other_entries": other[:LIST_CAP], "n_missing": len(missing), "n_extra": len(extra), "n_retyped": len(retyped),
            "n_link_mismatch": len(link_mismatch), "extra_dirs": extra_dirs[:LIST_CAP], "n_extra_dirs": len(extra_dirs),
            "seal_checked": bool(seal_check), "unsealed_dirs": unsealed[:LIST_CAP], "n_unsealed_dirs": len(unsealed), "seal_elapsed_s": round(t2 - t1, 3)}


def _sample_seed(window: str, sha8: str) -> int:
    """The seed of a later attestation's content sample: a function of the OPERATION and the generation, so successive operations over one
    generation sample different files (and two attestations of one operation sample the same ones)."""
    return int(hashlib.sha256(f"{window}:{sha8}".encode("utf-8")).hexdigest()[:8], 16)


def sample_generation(gen_dir: str, sha8: str, n: int = ATTEST_SAMPLE_N, seed: Optional[int] = None) -> Admission:
    """The content check of a LATER attestation (within :data:`ATTEST_FULL_WALK_MAX_AGE_S` of the recorded full walk): a seeded
    sample of ``n`` manifest entries, each ``lstat``'d against the sealed tree's DEST-STAT snapshot and hashed against the manifest -- no
    walk of the tree (the listing is the namespace check; the first attestation's :func:`admit_generation` was the full walk). Refusals by
    name as :func:`admit_generation`'s: ``no_manifest``, ``no_dest_stat``, ``generation_changed`` (a sampled entry gone or no longer a
    regular file), ``changed_during_hash``, ``content_changed``. A moved stat over intact bytes is admitted and counted (``moved``)."""
    t0 = time.monotonic()
    gen_path = f"{gen_dir}/{sha8}"
    ad = Admission(gen_path, seed=int(sha8, 16) if seed is None else int(seed))

    def refuse(name: str, reason: str) -> Admission:
        ad.refusal, ad.reason, ad.elapsed_s = name, reason, time.monotonic() - t0
        return ad

    mpath = f"{gen_dir}/MANIFEST-{sha8}.sha256"
    if not os.path.isfile(mpath):
        return refuse("no_manifest", f"no {mpath}")
    entries = read_manifest_lines(mpath)
    want = {rel: (sha, size) for sha, size, rel in entries}
    want_links = read_manifest_symlinks(mpath)
    ad.n_files, ad.n_symlinks = len(entries), len(want_links)
    snap = _load_json(f"{gen_dir}/DEST-STAT-{sha8}.json") or {}
    if not isinstance(snap.get("files"), dict):
        return refuse("no_dest_stat", f"no DEST-STAT-{sha8}.json under {gen_dir}; the seal cannot be checked")
    rels = sorted(want, key=_by_bytes)
    sample = random.Random(ad.seed).sample(rels, min(max(0, int(n)), len(rels)))
    ad.sampled = len(sample)
    for rel in sample:
        p = os.path.join(gen_path, escape_rel(rel))
        try:
            st = os.lstat(p)
        except OSError as exc:
            ad.missing.append(rel)
            return refuse("generation_changed", f"sampled entry {rel} cannot be read under {gen_path}: {type(exc).__name__}")
        if not statmod.S_ISREG(st.st_mode):
            ad.mismatched.append(rel)
            return refuse("generation_changed", f"sampled entry {rel} is not a regular file under {gen_path}")
        recorded = snap["files"].get(rel)
        if not isinstance(recorded, list) or list(dest_stat_key(st)) != [int(x) for x in recorded]:
            ad.moved += 1
        sha = R.file_sha256(p)
        ad.hashed += 1
        if dest_stat_key(os.lstat(p)) != dest_stat_key(st):
            return refuse("changed_during_hash", f"{rel} changed while being hashed; the generation is being written")
        if (sha, int(st.st_size)) != want[rel]:
            ad.mismatched.append(rel)
    if ad.mismatched:
        return refuse("content_changed", f"{len(ad.mismatched)} sampled entr(y/ies) differ from manifest {sha8} (first: {ad.mismatched[0]})")
    ad.elapsed_s = time.monotonic() - t0
    return ad


def attested_path(gen_dir: str, sha8: str) -> str:
    """``gen/<slug>/ATTESTED-<sha8>.json``: the one fact an attestation keeps across operations -- when the generation's full content walk
    last passed (:data:`ATTESTED_SCHEMA`). Written by the attestation job only; the parent ``gen/<slug>/`` is writable by design."""
    return f"{gen_dir}/ATTESTED-{sha8}.json"


def _attestation_record(gen_path: str, sha8: str, window: str, utc: str) -> Dict[str, Any]:
    return {"generation": gen_path, "content_sha8": sha8, "window": window, "ok": False, "refusal": None, "reason": "", "utc": utc,
            "authority": {}, "listing": None, "seal": None, "content": None, "elapsed_s": 0.0}


def attest_generation(gen_dir: str, sha8: str, window: str, now: Callable[[], str] = R.utc_now,
                      full_walk_max_age_s: float = ATTEST_FULL_WALK_MAX_AGE_S, sample_n: int = ATTEST_SAMPLE_N, history: bool = True,
                      seed: Optional[int] = None, log: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """ONE pinned generation's attestation for operation ``window``, in this order, stopping at the first refusal (what ran
    stays in the record): (a) the AUTHORITY FACTS -- `current` names the pin (``no_current`` / ``current_moved``: the plan is stale), the
    tree exists (``dangling``), MANIFEST-<sha8> is there and hashes to the pin (``no_manifest`` / ``manifest_mismatch``), VERIFIED is an ok
    report of that manifest (``unverified`` / ``verified_mismatch``), DEST-STAT is there (``no_dest_stat``), SEALED and PUBLISHED stand
    beside it and name the pin (``unsealed`` / ``sealed_mismatch`` / ``unrecorded`` / ``published_mismatch``); (b) the full directory LISTING
    against the manifest (:func:`list_generation`; ``listing_mismatch``); (c) the directory SEAL -- every directory 0555
    (``unsealed_directories``) and the count equal to the SEALED record's (``directory_count_changed``); (d) the CONTENT -- the full
    :func:`admit_generation` walk when no ``ATTESTED-<sha8>.json`` records a passed full walk within ``full_walk_max_age_s`` (``mode: full``;
    the record is then written, ``history`` permitting), else :func:`sample_generation` over ``sample_n`` seeded entries (``mode: sample``;
    the seed a function of the operation and the generation). ``history=False`` writes nothing (a read-only rehearsal). Every refusal is
    a RESULT by name; only a crash raises."""
    t0 = time.monotonic()
    gen_path = f"{gen_dir}/{sha8}"
    rec = _attestation_record(gen_path, sha8, window, now())

    def refuse(name: str, reason: str) -> Dict[str, Any]:
        rec.update(ok=False, refusal=name, reason=reason, elapsed_s=round(time.monotonic() - t0, 3))
        log(f"[attest] REFUSED {gen_path}: {name}: {reason}")
        return rec

    # (a) the authority facts, all read where the tree is
    cur_path = f"{gen_dir}/current"
    current = os.readlink(cur_path) if os.path.islink(cur_path) else None
    rec["authority"]["current"] = current
    if current is None:
        return refuse("no_current", f"nothing is published under {gen_dir} (no `current`); the plan pinned {sha8}")
    if current != sha8:
        return refuse("current_moved", f"`current` under {gen_dir} names {current}, not the pinned {sha8}: another generation was published since the plan; plan again")
    if not os.path.isdir(gen_path):
        return refuse("dangling", f"`current` -> {sha8} under {gen_dir} names a tree that is not there")
    mpath = f"{gen_dir}/MANIFEST-{sha8}.sha256"
    if not os.path.isfile(mpath):
        return refuse("no_manifest", f"no {mpath}")
    entries = read_manifest_lines(mpath)
    want_links = read_manifest_symlinks(mpath)
    ns, ct = two_manifest_hashes(entries, sorted(want_links.items(), key=lambda t: _by_bytes(t[0])))
    if ct[:8] != sha8:
        return refuse("manifest_mismatch", f"{mpath} hashes to {ct[:8]}, not {sha8}")
    report = _load_json(f"{gen_dir}/VERIFIED-{sha8}.json") or {}
    if not report:
        return refuse("unverified", f"no VERIFIED-{sha8}.json under {gen_dir}; the generation was never verified")
    if not (report.get("ok") is True and report.get("content_sha256") == ct and report.get("name_size_sha256") == ns
            and not (report.get("missing") or report.get("corrupt") or report.get("extra"))):
        return refuse("verified_mismatch", f"VERIFIED-{sha8}.json is not an ok report of manifest {sha8} (ok={report.get('ok')!r})")
    if not os.path.isfile(f"{gen_dir}/DEST-STAT-{sha8}.json"):
        return refuse("no_dest_stat", f"no DEST-STAT-{sha8}.json under {gen_dir}; the content check has no stat snapshot")
    sealed = _load_json(f"{gen_dir}/SEALED-{sha8}.json") or {}
    if not sealed:
        return refuse("unsealed", f"no SEALED-{sha8}.json under {gen_dir}: the generation's directories were never sealed (every publication seals; "
                                  f"`stores verify-authority` and `stores mirror --adopt-unrecorded` adopt a pre-seal generation)")
    if sealed.get("content_sha8") != sha8:
        return refuse("sealed_mismatch", f"SEALED-{sha8}.json names content {sealed.get('content_sha8')}, not {sha8}")
    pub = _load_json(f"{gen_dir}/PUBLISHED-{sha8}.json") or {}
    if not pub:
        return refuse("unrecorded", f"no PUBLISHED-{sha8}.json under {gen_dir}: the publication is unrecorded (a publisher mid-flight or crashed, or a "
                                    f"pre-record generation `stores mirror --adopt-unrecorded` has not adopted)")
    if pub.get("content_sha8") != sha8:
        return refuse("published_mismatch", f"PUBLISHED-{sha8}.json names content {pub.get('content_sha8')}, not {sha8}")
    rec["authority"].update(verified_utc=report.get("verified_utc"), n_files=len(entries), n_symlinks=len(want_links), bytes=int(report.get("bytes") or 0),
                            sealed_utc=sealed.get("sealed_utc"), sealed_n_dirs=sealed.get("n_dirs"), published_utc=pub.get("published_utc"),
                            change_utc=pub.get("change_utc"), publish_mode=pub.get("mode"), producer=pub.get("producer"))
    # (b) the listing against the manifest, (c) the seal -- one scandir pass, then the directory modes
    lst = list_generation(gen_path, entries, want_links)
    rec["listing"] = {k: lst[k] for k in ("entries", "files", "dirs", "symlinks", "other", "elapsed_s", "ok", "missing", "extra", "retyped", "link_mismatch",
                                          "other_entries", "n_missing", "n_extra", "n_retyped", "n_link_mismatch", "extra_dirs", "n_extra_dirs")}
    rec["seal"] = {"ok": lst["n_unsealed_dirs"] == 0, "n_dirs": lst["dirs"], "recorded_n_dirs": sealed.get("n_dirs"), "unsealed_dirs": lst["unsealed_dirs"],
                   "n_unsealed_dirs": lst["n_unsealed_dirs"], "elapsed_s": lst["seal_elapsed_s"]}
    if not lst["ok"]:
        first = (lst["missing"] or lst["extra"] or lst["retyped"] or lst["link_mismatch"] or lst["other_entries"] or ["?"])[0]
        return refuse("listing_mismatch", f"the directory listing of {gen_path} is not manifest {sha8}'s: {lst['n_missing']} missing, {lst['n_extra']} extra, "
                                          f"{lst['n_retyped']} re-typed, {lst['n_link_mismatch']} link target(s) differ, {lst['other']} other (first: {first})")
    if lst["n_unsealed_dirs"]:
        return refuse("unsealed_directories", f"{lst['n_unsealed_dirs']} of {lst['dirs']} director(y/ies) under {gen_path} are not {DIR_SEAL_MODE:04o} "
                                              f"(first: {lst['unsealed_dirs'][0]}); the directory seal was lifted")
    sealed_n = sealed.get("n_dirs")
    if isinstance(sealed_n, int) and sealed_n != lst["dirs"]:
        return refuse("directory_count_changed", f"{lst['dirs']} director(y/ies) under {gen_path}, SEALED-{sha8}.json recorded {sealed_n} at the seal "
                                                 f"({lst['n_extra_dirs']} the manifest implies by no entry); a directory was added or removed")
    # (d) the content: the full walk the first time (or after a week), else the seeded sample
    hist = _load_json(attested_path(gen_dir, sha8)) if history else None
    age = _age_s((hist or {}).get("full_walk_utc"), rec["utc"]) if hist else None
    sample_mode = bool(hist and hist.get("ok") is True and hist.get("content_sha8") == sha8 and age is not None and 0 <= age <= float(full_walk_max_age_s))
    if sample_mode:
        ad = sample_generation(gen_dir, sha8, n=sample_n, seed=seed if seed is not None else _sample_seed(window, sha8))
        why = f"full walk recorded {hist.get('full_walk_utc')} ({age / 86400:.1f} d ago, within {full_walk_max_age_s / 86400:.0f} d)"   # type: ignore[union-attr]
    else:
        ad = admit_generation(gen_dir, sha8, seed=seed)
        why = ("no full walk recorded" if not hist else "the recorded full walk is not this generation's or not ok" if not (hist.get("ok") is True and hist.get("content_sha8") == sha8)
               else "the recorded full walk's stamp does not parse" if age is None else f"the recorded full walk is {age / 86400:.1f} d old (> {full_walk_max_age_s / 86400:.0f} d)")
    rec["content"] = {"mode": "sample" if sample_mode else "full", "why": why, "ok": ad.ok, "refusal": ad.refusal, "reason": ad.reason, "n_files": ad.n_files,
                      "moved": ad.moved, "sampled": ad.sampled, "hashed": ad.hashed, "mismatched": ad.mismatched[:LIST_CAP], "missing": ad.missing[:LIST_CAP],
                      "extra": ad.extra[:LIST_CAP], "seed": ad.seed, "elapsed_s": round(ad.elapsed_s, 3),
                      "full_walk_utc": (hist or {}).get("full_walk_utc") if sample_mode else None, "history_written": False}
    if not ad.ok:
        return refuse(str(ad.refusal), f"content ({rec['content']['mode']}): {ad.reason}")
    if not sample_mode:
        rec["content"]["full_walk_utc"] = rec["utc"]
        if history:
            try:
                _write_json(attested_path(gen_dir, sha8), {"schema": ATTESTED_SCHEMA, "generation": gen_path, "content_sha8": sha8, "ok": True,
                                                           "full_walk_utc": rec["utc"], "window": window, "host": socket.gethostname(), "n_files": ad.n_files,
                                                           "moved": ad.moved, "sampled": ad.sampled, "hashed": ad.hashed, "elapsed_s": round(ad.elapsed_s, 3)})
                rec["content"]["history_written"] = True
            except OSError as exc:
                # the record is a convenience (the next operation walks in full again), never a reason to refuse a generation that just passed
                rec["content"]["history_error"] = f"{type(exc).__name__}: {exc}"
    rec.update(ok=True, elapsed_s=round(time.monotonic() - t0, 3))
    log(f"[attest] ok {gen_path}: {lst['entries']} entries listed in {lst['elapsed_s']}s, {lst['dirs']} directories sealed ({lst['seal_elapsed_s']}s), "
        f"content {rec['content']['mode']} ({ad.hashed} hashed, {ad.elapsed_s:.1f}s)")
    return rec


def plan_pins(plan: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """The UNION of every cluster-routed adapter's pins in a cluster-reads plan (``{slug: sha8}``) and the conflicts (``{slug: [sha8, ...]}``
    where two adapters pin one component at different generations -- the plan never writes that; a plan FILE is data)."""
    pins: Dict[str, str] = {}
    conflicts: Dict[str, set] = {}
    for _name, entry in sorted((plan.get("adapters") or {}).items()):
        if not isinstance(entry, Mapping) or entry.get("route") != "cluster":
            continue
        for slug, sha8 in (entry.get("pins") or {}).items():
            slug, sha8 = str(slug), str(sha8)
            if slug in pins and pins[slug] != sha8:
                conflicts.setdefault(slug, {pins[slug]}).add(sha8)
            pins.setdefault(slug, sha8)
    return pins, {s: sorted(v) for s, v in conflicts.items()}


def attest_plan(plan_path: str, out: str, now: Callable[[], str] = R.utc_now, threads: int = ATTEST_THREADS, job_id: Optional[str] = None,
                log: Callable[[str], None] = lambda s: None, **kw: Any) -> Dict[str, Any]:
    """The attestation JOB's body (``attest.sbatch``): every pinned generation of the plan's cluster-routed adapters attested for THIS
    operation (:func:`attest_generation`, in ``threads`` parallel threads -- the listings are NFS-bound), the record written atomically to
    ``out`` (:data:`ATTESTATION_SCHEMA`: the window, the plan and its sha, per component the generation's record, the refused components,
    ``ok``). A component two adapters pin differently is refused ``pin_conflict``; a crash inside one generation's check is THAT
    generation's refusal (``error``), never the whole record's. Raises only when the plan cannot be read -- then nothing is written and
    the job fails, which the read array's ``afterok`` dependency turns into its cancellation."""
    t0 = time.monotonic()
    plan = _load_json(plan_path)
    if not plan:
        raise StoresError(f"attest: {plan_path} is not a readable JSON object")
    gen_dir, window = str(plan.get("gen_dir") or ""), str(plan.get("window") or "")
    if not gen_dir or not window:
        raise StoresError(f"attest: {plan_path} carries no gen_dir / window (not a cluster-reads plan)")
    pins, conflicts = plan_pins(plan)
    rec: Dict[str, Any] = {"schema": ATTESTATION_SCHEMA, "window": window, "plan": os.path.abspath(plan_path), "plan_sha256": R.file_sha256(plan_path),
                           "gen_dir": gen_dir, "utc": now(), "host": socket.gethostname(), "job_id": job_id, "threads": int(threads),
                           "n_generations": len(pins), "generations": {}, "refused": [], "ok": True, "elapsed_s": 0.0}

    def one(slug: str, sha8: str) -> Dict[str, Any]:
        if slug in conflicts:
            r = _attestation_record(f"{gen_dir}/{slug}", sha8, window, now())
            r.update(refusal="pin_conflict", reason=f"the plan pins component {slug} at {conflicts[slug]} for different adapters; one generation per component per operation")
            return r
        try:
            return attest_generation(f"{gen_dir}/{slug}", sha8, window, now=now, log=log, **kw)
        except Exception as exc:        # noqa: BLE001 -- one generation's crash is its refusal; the other generations are still attested
            r = _attestation_record(f"{gen_dir}/{slug}/{sha8}", sha8, window, now())
            r.update(refusal="error", reason=f"{type(exc).__name__}: {exc}")
            return r

    with ThreadPoolExecutor(max_workers=max(1, min(int(threads), len(pins) or 1))) as ex:
        futures = {slug: ex.submit(one, slug, sha8) for slug, sha8 in sorted(pins.items())}
        for slug, fut in futures.items():
            rec["generations"][slug] = fut.result()
    rec["refused"] = sorted(s for s, g in rec["generations"].items() if g.get("ok") is not True)
    rec["ok"] = not rec["refused"]
    rec["elapsed_s"] = round(time.monotonic() - t0, 3)
    _write_json(out, rec)
    log(f"[attest] {window}: {'ok' if rec['ok'] else 'REFUSED ' + ', '.join(rec['refused'])} -- {len(pins)} generation(s), {rec['elapsed_s']}s -> {out}")
    return rec


def read_attestation(path: str) -> Optional[Dict[str, Any]]:
    """The operation's attestation record, or None when absent, unreadable or not one (:data:`ATTESTATION_SCHEMA`)."""
    rec = _load_json(path)
    return rec if rec and rec.get("schema") == ATTESTATION_SCHEMA and isinstance(rec.get("generations"), dict) else None


def attestation_admits(rec: Optional[Mapping[str, Any]], window: str, pins: Mapping[str, str], path: str = "") -> Dict[str, Dict[str, Any]]:
    """The SHARD'S admission, O(1) per pin: is every pinned component attested ok by THIS operation's record? Per slug ``{ok, refusal,
    reason, ...}``; refusals by name: ``no_attestation`` (no record), ``attestation_window`` (another operation's record), ``attestation_missing``
    (the record names no such component), ``attestation_pin`` (it attests another generation), ``attestation_refused`` (the attestation
    refused it -- the record's refusal and reason ride along). No walk, no hash."""
    out: Dict[str, Dict[str, Any]] = {}
    for slug, sha8 in sorted(pins.items()):
        base: Dict[str, Any] = {"slug": slug, "generation": f"{slug}/{sha8}", "ok": False, "refusal": None, "reason": "", "attested_utc": None,
                                "content_mode": None, "attestation": path or None}
        if rec is None:
            out[slug] = {**base, "refusal": "no_attestation", "reason": f"no attestation record at {path or '(unknown path)'}: the operation's attestation job wrote none"}
            continue
        if str(rec.get("window")) != str(window):
            out[slug] = {**base, "refusal": "attestation_window", "reason": f"the attestation record is operation {rec.get('window')}'s, not {window}'s"}
            continue
        g = (rec.get("generations") or {}).get(slug)
        if not isinstance(g, Mapping):
            out[slug] = {**base, "refusal": "attestation_missing", "reason": f"the attestation record of {window} names no component {slug}"}
            continue
        base.update(attested_utc=g.get("utc"), content_mode=(g.get("content") or {}).get("mode"))
        if g.get("content_sha8") != sha8:
            out[slug] = {**base, "refusal": "attestation_pin", "reason": f"the attestation record attests {slug}@{g.get('content_sha8')}, not the pinned {sha8}"}
            continue
        if g.get("ok") is not True:
            out[slug] = {**base, "refusal": "attestation_refused", "attestation_refusal": g.get("refusal"), "reason": f"{g.get('refusal')}: {g.get('reason')}"}
            continue
        out[slug] = {**base, "ok": True}
    return out


def manifest_of_directory(path: str, members: Sequence[str] = (".",), exclude: Sequence[str] = (), workers: int = 16) -> Manifest:
    """A manifest of an in-place original (the cluster side of ``stores check``): the same walk rules as the grading host
    mirror's component, no stat cache (every file is hashed)."""
    path = os.path.abspath(path)
    comp = Component(os.path.basename(path), path, "cluster_a_origin", members=list(members), exclude=list(exclude), original=path)
    return build_manifest(comp, None, workers=workers)


def manifest_in_place(directory: str, out_prefix: str, members: Sequence[str], exclude: Sequence[str], workers: int,
                      job_id: Optional[str] = None, grading_host_root: Optional[str] = None) -> Manifest:
    """What ``store_verify.sbatch manifest-only`` runs: manifest the original in place and write ``<out_prefix>.{sha256,json}``.
    The ``.json`` records ``root`` as the grading host mirror's spelling passed in (``--grading-host-root``; None when not passed) and NO
    ``home_rel``: both would otherwise be computed against the compute node's home, which means nothing."""
    man = manifest_of_directory(directory, members=members or ["."], exclude=exclude, workers=workers)
    _write_text(out_prefix + ".sha256", man.text())
    _write_json(out_prefix + ".json", man.meta({"job_id": job_id, "in_place_original": os.path.abspath(directory), "root": grading_host_root, "home_rel": None,
                                                "spelling": "built on the cluster: `root` is the grading host mirror's spelling as passed (--grading-host-root); "
                                                            "`home_rel` is the grading host's and is not recorded here"}))
    return man


# --------------------------------------------------------------------------------------------
# the destination: rsync, ssh shell steps, sbatch + the watcher (every process through ONE runner)
# --------------------------------------------------------------------------------------------


def _default_runner(argv: Sequence[str], timeout_s: float, input: Optional[str] = None) -> subprocess.CompletedProcess:
    """The one subprocess shape behind every shell step, rsync, sbatch and watch; tests inject a recorder with this signature."""
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s, input=input)


def locked_step(gen_dir: str, body: str, wait_s: float, holder: str, utc: str) -> str:
    """ONE rendering, for every publisher, of a destination shell step that runs ``body`` inside ``gen_dir`` under the slug's publish
    lock: an exclusive ``flock`` on the SAME ``gen/<slug>/.publish.lock`` :class:`PublishLock` takes (the file's name and placement are
    shared by construction -- :data:`PUBLISH_LOCK_FILE` under the slug's generation directory), held on fd :data:`PUBLISH_LOCK_FD` from
    before the body's first read until the shell exits, so a reading the body takes and the rename it ends in are one critical section
    against every other publisher of the slug, on-cluster or grading-host-driven. The holder line is written into the
    lock file the way PublishLock writes it (``<holder> <utc>``), so a waiter on either side names who holds it. Exit codes the caller
    maps: 60 ``gen_dir`` is not a directory; 69 the lock file cannot be opened; 70 the lock stayed held for ``wait_s`` (the holder line
    is on stdout; ``wait_s <= 0`` tries once); 71 ``flock`` failed for another reason (a filesystem without locks). Everything else is
    the body's."""
    q = shlex.quote
    fd = PUBLISH_LOCK_FD
    take = f"flock -n -E 70 {fd}" if wait_s <= 0 else f"flock -w {max(0.001, float(wait_s)):.3f} -E 70 {fd}"
    return (f"cd {q(gen_dir)} || exit 60; exec {fd}>>{q(PUBLISH_LOCK_FILE)} || exit 69; {take}; rc=$?; "
            f"if test \"$rc\" = 70; then cat {q(PUBLISH_LOCK_FILE)} 2>/dev/null; exit 70; elif test \"$rc\" != 0; then exit 71; fi; "
            f"printf '%s %s\\n' {q(holder)} {q(utc)} > {q(PUBLISH_LOCK_FILE)}; {body}")


class Remote:
    """The handful of shell steps the grading host runs on the destination (locally for a directory destination), one process each."""

    def __init__(self, dest: Destination, runner=None):
        self.dest = dest
        self.run = runner or _default_runner

    def _on_dest(self, argv: Sequence[str]) -> List[str]:
        """``argv`` as run on the destination: itself for a local directory, else through one ssh."""
        return list(argv) if self.dest.is_local else ["ssh", *SSH_OPTS, self.dest.host or "", shlex.join(argv)]

    def _cmd(self, cmd: str, what: Optional[str] = None, input: Optional[str] = None, ok: Sequence[int] = (0,),
             timeout_s: Optional[float] = None) -> subprocess.CompletedProcess:
        """One shell step; with ``what`` set, a return code outside ``ok`` raises :class:`StoresError` naming the step. ``timeout_s``
        overrides :data:`SHELL_TIMEOUT_S` for a step that may legitimately wait (the locked publish step)."""
        r = self.run(self._on_dest(["sh", "-c", cmd]), timeout_s or SHELL_TIMEOUT_S, input=input)
        if what and r.returncode not in ok:
            raise StoresError(f"{what} on {self.dest.name}: rc {r.returncode}: {(r.stderr or '').strip()[:300]}")
        return r

    def mkdir(self, path: str) -> None:
        self._cmd(f"mkdir -p {shlex.quote(path)}", what=f"mkdir -p {path}")

    def read(self, path: str) -> Optional[str]:
        """The file's text, or None when it does not exist."""
        q = shlex.quote(path)
        r = self._cmd(f"if test -f {q}; then cat {q}; else exit 44; fi", what=f"read {path}", ok=(0, 44))
        return None if r.returncode == 44 else r.stdout

    def read_json(self, path: str) -> Dict[str, Any]:
        """The file's JSON object, or ``{}`` when it does not exist (or is not an object); malformed JSON is an error."""
        text = self.read(path)
        if text is None:
            return {}
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise StoresError(f"{path} on {self.dest.name} is not JSON: {exc}")
        return data if isinstance(data, dict) else {}

    def readlink(self, path: str) -> Optional[str]:
        """The symlink's target; None when nothing is at ``path``. A transport failure (ssh rc 255, a shell that did not run)
        or a ``path`` that exists but is not a symlink is an ERROR by name -- never read as "no link": a `current` misread as
        absent would re-copy a whole root with no hard links."""
        q = shlex.quote(path)
        r = self._cmd(f"if test -L {q}; then readlink {q}; elif test -e {q}; then exit 45; else exit 44; fi", what=f"readlink {path}", ok=(0, 44, 45))
        if r.returncode == 45:
            raise StoresError(f"{path} on {self.dest.name} exists but is not a symlink; not a layout this command wrote")
        return None if r.returncode == 44 else r.stdout.strip()

    def isdir(self, path: str) -> bool:
        """Is ``path`` a directory on the destination (``test -d``)? Like :meth:`readlink`, a transport failure is an ERROR by
        name, never read as "absent": a present generation misread as absent would be copied again in full."""
        r = self._cmd(f"if test -d {shlex.quote(path)}; then exit 0; else exit 44; fi", what=f"test -d {path}", ok=(0, 44))
        return r.returncode == 0

    def renamed_aside(self, gen_dir: str, sha8: str) -> List[str]:
        """The ``.failed-<sha8>-*`` and ``.invalid-<sha8>-*`` entries under ``gen_dir`` -- what a rename of this generation's
        tree left behind -- as sorted basenames (the stamp orders them, oldest first). One stat-class shell step, run ONLY
        when ``current`` dangles (never on the unchanged path); a transport failure is an ERROR by name."""
        if not re.fullmatch(r"[0-9a-f]{8}", sha8):
            raise StoresError(f"content sha8 {sha8!r} is not eight hex digits; not a generation name this command wrote")
        cmd = (f"cd {shlex.quote(gen_dir)} && for p in .failed-{sha8}-* .invalid-{sha8}-*; do "
               f"if test -e \"$p\"; then printf '%s\\n' \"$p\"; fi; done")
        r = self._cmd(cmd, what=f"list the renamed trees of {sha8} under {gen_dir}")
        return sorted(line.strip() for line in (r.stdout or "").splitlines() if line.strip())

    def reports_aside(self, gen_dir: str, sha8: str, stamp: str) -> List[str]:
        """Every report and snapshot that speaks for the tree at ``gen_dir/<sha8>`` -- the live ``VERIFIED-<sha8>.json`` and
        ``DEST-STAT-<sha8>.json``, every keyed ``VERIFIED-<sha8>.<lsha8>.json`` / ``DEST-STAT-<sha8>.<lsha8>.json`` a `stores attest`
        wrote over it, AND its directory seal ``SEALED-<sha8>.json`` -- moved aside as ``.stale-<name>-<stamp>.json`` in ONE shell step,
        run right before that content is copied afresh (its tree renamed aside or gone): a report describes the tree it was run over, and
        the fresh tree earns its own (round 3 on 2c: a keyed report of a superseded tree attested the fresh one; the closing pass: the
        replaced tree's SEALED record stood beside the replacement, so a repair whose own seal failed read `unchanged` over writable
        directories the next tick -- the seal counts and dates the directories of the tree it sealed, and the fresh tree earns its own).
        The keyed ``MANIFEST-<sha8>.<lsha8>.json`` attestations stay: they name a (content, listings) pair, not a tree; and the
        ``PUBLISHED-<sha8>.json`` record stays: it records the publication of the POINTER (the locked publish step requires the record of
        the `current` it compares against, and reads the producer from it), and the replacement's own publication rewrites it inside that
        step. Returns the new paths, oldest name first; nothing stood -> ``[]``. A transport failure is an ERROR by name, never "nothing to
        move"."""
        if not re.fullmatch(r"[0-9a-f]{8}", sha8):
            raise StoresError(f"content sha8 {sha8!r} is not eight hex digits; not a generation name this command wrote")
        if not re.fullmatch(r"[0-9A-Za-z]+", stamp):
            raise StoresError(f"stamp {stamp!r} is not a file-name stamp this command writes")
        cmd = (f"cd {shlex.quote(gen_dir)} || exit 61; for p in VERIFIED-{sha8}.json VERIFIED-{sha8}.*.json DEST-STAT-{sha8}.json DEST-STAT-{sha8}.*.json SEALED-{sha8}.json; do "
               f"if test -f \"$p\"; then s=\".stale-${{p%.json}}-{stamp}.json\"; mv -T \"$p\" \"$s\" || exit 62; printf '%s\\n' \"$s\"; fi; done")
        r = self._cmd(cmd, what=f"move the reports of {sha8} under {gen_dir} aside")
        return [f"{gen_dir}/{line.strip()}" for line in (r.stdout or "").splitlines() if line.strip()]

    def write(self, path: str, text: str) -> None:
        q = shlex.quote(path)
        # a per-process temp name on the destination too (`$$` is the remote shell's pid), then one rename
        self._cmd(f"mkdir -p {shlex.quote(os.path.dirname(path))} && cat > {q}.$$.tmp && mv -f {q}.$$.tmp {q}", what=f"write {path}", input=text)

    def rename(self, src: str, dst: str) -> None:
        self._cmd(f"mv -T {shlex.quote(src)} {shlex.quote(dst)}", what=f"mv -T {src} {dst}")

    def exists(self, path: str) -> bool:
        """Whether anything (a file, a directory, a link) is at ``path``; a transport failure is an error by name, never "absent"."""
        r = self._cmd(f"if test -e {shlex.quote(path)} || test -L {shlex.quote(path)}; then exit 0; else exit 44; fi", what=f"test -e {path}", ok=(0, 44))
        return r.returncode == 0

    #: what the publish guard requires beside ``gen/<slug>/<sha8>`` (exit code -> the missing piece), checked ON the destination in the
    #: same shell step that switches `current`, so the directory is re-read at the publish moment
    PUBLISH_REQUIRES = ((61, "the generation directory"), (62, "MANIFEST-<sha8>.sha256"), (63, "VERIFIED-<sha8>.json"),
                        (64, "DEST-STAT-<sha8>.json"), (65, "SEALED-<sha8>.json"))

    #: how the step reads a record's producer: one `"producer": "<value>"` pair, indented or compact (every record is written by
    #: :func:`published_record`, one key per line); a record naming no producer, or `null`, yields nothing
    _PRODUCER_GREP = "grep -o '\"producer\": *\"[^\"]*\"' {rec} | head -1 | sed 's/.*: *\"//; s/\"$//'"

    @staticmethod
    def _record_write(rec_name: str, tmp_rec: str) -> str:
        """The record from stdin to a temporary name, then one rename (exit 74 when either fails)."""
        q = shlex.quote
        return f"cat > {q(tmp_rec)} && mv -f {q(tmp_rec)} {q(rec_name)} || {{ rm -f {q(tmp_rec)}; exit 74; }}; "

    def publish(self, gen_dir: str, sha8: str, record_text: str, expect_current: Optional[str] = None, lock_wait_s: float = PUBLISH_LOCK_WAIT_S,
                lock_holder: Optional[str] = None, lock_held: bool = False, utc: Optional[str] = None, adopt_unrecorded: bool = False,
                producer_guard: bool = True) -> Dict[str, Any]:
        """The one publication step: ``current -> <sha8>`` by a symlink rename -- REFUSED by name unless ``gen/<slug>/<sha8>`` is a
        directory with its manifest, VERIFIED report, DEST-STAT snapshot and SEALED record beside it (:func:`seal_generation` writes the
        last after the directory seal). Nothing switches `current` onto a tree nobody verified and sealed.

        The switch is a COMPARE-AND-SWITCH: in the same shell step `current` is re-read and must name ``expect_current``
        (None: there must be none) -- what the caller read at its start -- else :class:`CurrentMoved`: another publisher switched it
        meanwhile, and a run acting on that stale reading never moves `current` back onto its own, older snapshot. The temporary
        pointer is unique per publication (``.current.<uuid>.tmp``), so two publishers never share one; it is removed on a failure.

        The step runs UNDER THE SLUG'S PUBLISH LOCK: :func:`locked_step` takes ``gen/<slug>/.publish.lock`` -- the
        same file :class:`PublishLock` takes -- before the readlink and holds it through the rename, so the reading, the comparison and
        the switch are one critical section against every other publisher of the slug; a publisher whose reading went stale waits for
        the other's switch and then refuses, never overwrites. ``lock_held``: the caller already holds the slug's PublishLock on this
        (local) destination for its whole flow, and the step runs under it (a second flock from a new open file description would wait
        on the caller's own lock). The lock stays held at most ``lock_wait_s``: :class:`PublishLockBusy` after that, naming the holder,
        `current` not switched.

        INSIDE the same critical section, in this order: the `current` it read must have its ``PUBLISHED`` record
        beside it, else :class:`CurrentUnrecorded` (a publisher is mid-flight or crashed there; ``adopt_unrecorded`` is the explicit
        acceptance of a legacy, unrecorded `current`, and this publication then supersedes it); with ``producer_guard`` (a grading-host-driven
        publisher) that record, and the generation's own if it was published before, must not name ``producer: cluster``, else
        :class:`ProducerMismatch`; then ``record_text`` -- the NEW generation's record, whose ``previous`` is ``expect_current``, exactly
        the `current` the step just compared -- is written to a temporary name and renamed into ``PUBLISHED-<sha8>.json``; only then
        does `current` switch. So no observer ever sees a `current` without its record, and nothing the switch depends on is read
        outside the lock. Returns the lock record (``path``, ``holder``, ``scope``)."""
        q = shlex.quote
        nonce = uuid.uuid4().hex[:12]
        tmp, tmp_rec, rec_name = f".current.{nonce}.tmp", f".PUBLISHED-{sha8}.{nonce}.tmp", f"PUBLISHED-{sha8}.json"
        body = (f"test -d {q(sha8)} || exit 61; test -f {q(f'MANIFEST-{sha8}.sha256')} || exit 62; "
                f"test -f {q(f'VERIFIED-{sha8}.json')} || exit 63; test -f {q(f'DEST-STAT-{sha8}.json')} || exit 64; "
                f"test -f {q(f'SEALED-{sha8}.json')} || exit 65; "
                f"C=''; if test -L current; then C=$(readlink current); elif test -e current; then exit 67; fi; "
                f"if test \"$C\" != {q(expect_current or '')}; then printf '%s' \"$C\"; exit 66; fi; ")
        if not adopt_unrecorded:
            body += "if test -n \"$C\" && ! test -f \"PUBLISHED-$C.json\"; then printf '%s' \"$C\"; exit 72; fi; "
        if producer_guard:
            body += ("P=''; if test -n \"$C\" && test -f \"PUBLISHED-$C.json\"; then P=$(" + self._PRODUCER_GREP.format(rec='"PUBLISHED-$C.json"') + "); fi; "
                     "if test \"$P\" = cluster; then printf 'current:%s' \"$C\"; exit 73; fi; "
                     f"P=''; if test -f {q(rec_name)}; then P=$(" + self._PRODUCER_GREP.format(rec=q(rec_name)) + "); fi; "
                     f"if test \"$P\" = cluster; then printf 'generation:%s' {q(sha8)}; exit 73; fi; ")
        body += self._record_write(rec_name, tmp_rec)
        body += f"ln -sfn {q(sha8)} {q(tmp)} && mv -T {q(tmp)} current || {{ rm -f {q(tmp)}; exit 68; }}"
        holder = lock_holder or f"{getpass.getuser()}@{socket.gethostname()}:{os.getpid()}"
        lock = {"path": f"{gen_dir}/{PUBLISH_LOCK_FILE}", "holder": holder, "scope": "flow" if lock_held else "step"}
        ok = (0, 60, 61, 62, 63, 64, 65, 66, 67, 68, 72, 73, 74)
        if lock_held:
            cmd = f"cd {q(gen_dir)} || exit 60; {body}"
            r = self._cmd(cmd, ok=ok, input=record_text)
        else:
            cmd = locked_step(gen_dir, body, lock_wait_s, holder, utc or R.utc_now())
            r = self._cmd(cmd, ok=ok + (69, 70, 71), input=record_text, timeout_s=max(0.0, float(lock_wait_s)) + SHELL_TIMEOUT_S)
        if r.returncode == 60:
            raise StoresError(f"publish_refused: {gen_dir} on {self.dest.name} is not a directory; `current` is not switched")
        if r.returncode == 70:
            line = next((l for l in (r.stdout or "").splitlines() if l.strip()), "") or "an unnamed publisher"
            raise PublishLockBusy(f"lock_busy: {lock['path']} on {self.dest.name} is held by {line.strip()[:200]} after {max(0.0, float(lock_wait_s)):.0f} s; "
                                  f"`current` is not switched onto {sha8} -- push again when that publication has finished (the verified tree stays)")
        if r.returncode in (69, 71):
            raise StoresError(f"publish {sha8} under {gen_dir} on {self.dest.name}: the publish lock {PUBLISH_LOCK_FILE} could not be "
                              f"{'opened' if r.returncode == 69 else 'taken (flock failed; a filesystem without locks?)'}: {(r.stderr or '').strip()[:300]}")
        missing = dict(self.PUBLISH_REQUIRES).get(r.returncode)
        if missing:
            raise StoresError(f"publish_refused: {gen_dir}/{sha8} on {self.dest.name} lacks {missing.replace('<sha8>', sha8)}; "
                              f"`current` is not switched onto an unverified or unsealed generation")
        if r.returncode == 66:
            actual = (r.stdout or "").strip()
            raise CurrentMoved(f"publish_refused: current_moved -- `current` under {gen_dir} on {self.dest.name} is {actual or 'absent'}, not "
                               f"{expect_current or 'absent'} as this run read it: another publisher switched it meanwhile; `current` is not "
                               f"switched onto this run's generation {sha8}")
        if r.returncode == 67:
            raise StoresError(f"publish_refused: {gen_dir}/current on {self.dest.name} exists but is not a symlink; not a layout this command wrote")
        if r.returncode == 72:
            cur = (r.stdout or "").strip() or expect_current
            raise CurrentUnrecorded(f"publish_refused: current_unrecorded -- `current` under {gen_dir} on {self.dest.name} names {cur} and no "
                                    f"PUBLISHED-{cur}.json stands beside it: a publisher is mid-flight or crashed there, or the generation predates "
                                    f"the record; `current` is not switched onto {sha8} -- wait for that publisher, or adopt the generation explicitly "
                                    f"(`stores push --adopt-unrecorded`, from the producer whose content it is)", generation=cur)
        if r.returncode == 73:
            which, _, gen = (r.stdout or "").strip().partition(":")
            raise ProducerMismatch(f"publish_refused: producer_mismatch -- PUBLISHED-{gen}.json ({'the `current` this run read' if which == 'current' else 'this generation, published before'}) "
                                   f"under {gen_dir} on {self.dest.name} names producer cluster and this component is not declared cluster-produced: "
                                   f"a grading host copy of a slug a cluster job publishes is a lagging mirror and never switches `current` onto or away "
                                   f"from a cluster publication; `current` is not switched onto {sha8}", generation=gen)
        if r.returncode == 74:
            raise StoresError(f"publish {sha8} under {gen_dir} on {self.dest.name}: the publication record could not be written: {(r.stderr or '').strip()[:300]}")
        if r.returncode != 0:
            raise StoresError(f"publish {sha8} under {gen_dir} on {self.dest.name}: rc {r.returncode}: {(r.stderr or '').strip()[:300]}")
        return lock

    def record_published(self, gen_dir: str, sha8: str, record_text: str, lock_wait_s: float = PUBLISH_LOCK_WAIT_S, lock_holder: Optional[str] = None,
                         lock_held: bool = False, utc: Optional[str] = None) -> Dict[str, Any]:
        """The explicit ADOPTION of an unrecorded `current`: ``record_text`` becomes ``PUBLISHED-<sha8>.json`` under the
        slug's publish lock WITHOUT switching anything -- the step requires `current` to name ``sha8`` (else :class:`CurrentMoved`: the
        record would describe a generation that is not current) and writes nothing when a record already stands (another publisher
        recorded it meanwhile; ``written`` False). The same lock, the same temp-name-and-rename as :meth:`publish`. Returns the lock
        record plus ``written``."""
        q = shlex.quote
        nonce = uuid.uuid4().hex[:12]
        rec_name, tmp_rec = f"PUBLISHED-{sha8}.json", f".PUBLISHED-{sha8}.{nonce}.tmp"
        body = (f"test -d {q(sha8)} || exit 61; C=''; if test -L current; then C=$(readlink current); elif test -e current; then exit 67; fi; "
                f"if test \"$C\" != {q(sha8)}; then printf '%s' \"$C\"; exit 66; fi; if test -f {q(rec_name)}; then exit 76; fi; "
                + self._record_write(rec_name, tmp_rec) + "exit 0")
        holder = lock_holder or f"{getpass.getuser()}@{socket.gethostname()}:{os.getpid()}"
        lock = {"path": f"{gen_dir}/{PUBLISH_LOCK_FILE}", "holder": holder, "scope": "flow" if lock_held else "step"}
        ok = (0, 60, 61, 66, 67, 74, 76)
        if lock_held:
            r = self._cmd(f"cd {q(gen_dir)} || exit 60; {body}", ok=ok, input=record_text)
        else:
            r = self._cmd(locked_step(gen_dir, body, lock_wait_s, holder, utc or R.utc_now()), ok=ok + (69, 70, 71), input=record_text,
                          timeout_s=max(0.0, float(lock_wait_s)) + SHELL_TIMEOUT_S)
        if r.returncode == 70:
            line = next((l for l in (r.stdout or "").splitlines() if l.strip()), "") or "an unnamed publisher"
            raise PublishLockBusy(f"lock_busy: {lock['path']} on {self.dest.name} is held by {line.strip()[:200]} after {max(0.0, float(lock_wait_s)):.0f} s; "
                                  f"the record of {sha8} is not written -- adopt again when that publication has finished")
        if r.returncode == 66:
            actual = (r.stdout or "").strip()
            raise CurrentMoved(f"adopt_refused: current_moved -- `current` under {gen_dir} on {self.dest.name} is {actual or 'absent'}, not {sha8}: a "
                               f"record is written only for the generation that is current")
        if r.returncode not in (0, 76):
            raise StoresError(f"adopt {sha8} under {gen_dir} on {self.dest.name}: rc {r.returncode}: {(r.stderr or '').strip()[:300]}")
        return {**lock, "written": r.returncode == 0}

    def seal_directories(self, gen_path: str) -> Dict[str, Any]:
        """``chmod 0555`` every directory of the generation, deepest first (the generation root included; its parent untouched), then
        count the directories and the ones still not 0555. Returns ``{n_dirs, unsealed, host}``; a non-empty ``unsealed`` is the
        caller's refusal (the filesystem did not honor the mode)."""
        q = shlex.quote
        cmd = (f"G={q(gen_path)}; test -d \"$G\" || exit 61; find \"$G\" -depth -type d -exec chmod {DIR_SEAL_MODE:04o} {{}} + || exit 62; "
               f"n=$(find \"$G\" -type d | wc -l) && u=$(find \"$G\" -type d ! -perm {DIR_SEAL_MODE:04o} | wc -l) && printf '%s %s %s\\n' \"$n\" \"$u\" \"$(hostname)\"")
        r = self.run(self._on_dest(["sh", "-c", cmd]), SEAL_TIMEOUT_S)
        if r.returncode == 61:
            raise StoresError(f"seal: {gen_path} on {self.dest.name} is not a directory")
        if r.returncode != 0:
            raise StoresError(f"seal {gen_path} on {self.dest.name}: rc {r.returncode}: {(r.stderr or '').strip()[:300]}")
        parts = (r.stdout or "").split()
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise StoresError(f"seal {gen_path} on {self.dest.name}: unreadable count {(r.stdout or '').strip()[:100]!r}")
        return {"n_dirs": int(parts[0]), "unsealed": int(parts[1]), "host": parts[2]}

    def unsealed_directories(self, gen_path: str) -> int:
        """How many directories of the generation are not ``0555`` -- the deep seal check (the attestation's; `verify-authority` and the
        R14 drill run it); -1 when the tree is absent."""
        q = shlex.quote
        r = self._cmd(f"G={q(gen_path)}; test -d \"$G\" || exit 61; find \"$G\" -type d ! -perm {DIR_SEAL_MODE:04o} | wc -l", ok=(0, 61))
        return -1 if r.returncode == 61 else int((r.stdout or "0").split()[0])

    def submit(self, job_name: str, script_args: Sequence[str], n_bytes: int, n_files: int, rate_mb_s: float,
               script: Optional[str] = None) -> Tuple[str, int]:
        """``sbatch --parsable`` of ``store_verify.sbatch`` with the resource lines spelled here and nowhere else: CPU only
        (no GPU, so no GPU budget), ``--dependency=singleton`` (one job per name at a time), ``--time`` twice the estimate at
        ``rate_mb_s``; every script argument POSITIONAL, never an export flag. ``script`` is the pinned tree's sbatch
        (:func:`pin_cluster_tree`); the repo mirror's working-tree copy only when nothing is pinned. The log goes to the
        destination's :attr:`Destination.log_dir` (created first; a job whose log directory is missing fails before it runs), since
        the sbatch names no log path a site with another shared root could not open. Returns (job id, the time limit in seconds)."""
        d = self.dest
        tlim = verify_time_s(n_bytes, n_files, rate_mb_s, d.verify_min_time_s, d.verify_max_time_s)
        self.mkdir(d.log_dir)
        argv = ["sbatch", "--parsable", f"--job-name={job_name}", "--dependency=singleton", f"--partition={d.partition}", f"--qos={d.qos}",
                f"--cpus-per-task={d.verify_cpus}", f"--mem={d.verify_mem}", f"--time={slurm_time(tlim)}", f"--output={d.log_dir}/%x_%j.out",
                script or d.sbatch_path, *script_args]
        r = self.run(self._on_dest(argv), SHELL_TIMEOUT_S)
        if r.returncode != 0:
            raise StoresError(f"sbatch on {d.name} failed (rc {r.returncode}): {(r.stderr or r.stdout or '').strip()[:400]}")
        m = re.search(r"\d+", r.stdout or "")
        if not m:
            raise StoresError(f"sbatch on {d.name} printed no job id: {(r.stdout or '').strip()[:200]!r}")
        return m.group(0), tlim

    def watch_argv(self, job_id: str, bound_s: Optional[float] = None) -> Tuple[List[str], float, int]:
        """The watcher's argv, the bound in seconds THIS PROCESS enforces on it, and the script's own whole-hour stop, for one job:
        ``watch-job.sh --slurm <id> --poll P --stall-min 0 --pending-min 0 --max-hours H [--host HOST]``. The bound is ``bound_s`` (the
        controller's `--bound-s`), else the destination's ``watch_max_hours``; the runner kills the watcher at it and :meth:`watch` reads
        WATCH-TIMEOUT. ``--max-hours`` is the script's own hard stop and is a WHOLE NUMBER: the script computes its deadline with Bash
        integer arithmetic, which refuses ``1.0`` or ``0.5`` with a syntax error before its first poll -- so the window's carries
        (`--bound-s 3600` / `1800`, rendered ``1.0`` / ``0.5`` here) got a dead watcher, read UNKNOWN and cancelled every verifier they
        submitted (the default ``4.0`` rendered the same way). It is the next whole hour PAST the bound, a backstop
        that never races the in-process bound, and the watch record carries both (``bound_s``, ``max_hours``)."""
        bound = float(bound_s) if bound_s and float(bound_s) > 0 else float(self.dest.watch_max_hours) * 3600.0
        max_hours = int(bound // 3600) + 1
        argv = [self.dest.watch_job, "--slurm", str(job_id), "--poll", str(int(self.dest.watch_poll_s)), "--stall-min", "0",
                "--pending-min", "0", "--max-hours", str(max_hours)]
        if not self.dest.is_local:
            argv += ["--host", self.dest.host or ""]
        return argv, bound, max_hours

    def watch(self, job_id: str, bound_s: Optional[float] = None) -> Dict[str, Any]:
        """``watch-job.sh`` (:meth:`watch_argv`) in the foreground, bounded IN THIS PROCESS at ``bound_s`` (else the destination's
        ``watch_max_hours``): the terminal line (DONE / FAILED / MISSING / WATCH-TIMEOUT) and the tail. Past the bound the runner kills the
        watcher and the state is WATCH-TIMEOUT with ``rc`` None -- the job may still be running and the caller owns what happens to it
        (:func:`mirror_component` cancels it by id and confirms it terminal). MISSING is the watcher LOSING the job (no accounting row for
        ten minutes; it never consults the queue), not a terminal state -- the caller reads the queue and the accounting itself."""
        argv, bound, max_hours = self.watch_argv(job_id, bound_s)
        try:
            r = self.run(argv, bound)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout
            text = out.decode(errors="replace") if isinstance(out, bytes) else (out or "")
            lines = [l for l in text.splitlines() if l.strip()]
            terminal = (f"WATCH-TIMEOUT job={job_id} -- the controller's bound of {int(bound)} s expired in this process and the watcher was "
                        "killed; the job may still be alive")
            return {"job_id": job_id, "state": "WATCH-TIMEOUT", "terminal": terminal, "rc": None, "tail": (lines + [terminal])[-12:],
                    "bound_s": bound, "max_hours": max_hours}
        lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
        terminal = next((l for l in reversed(lines) if l.startswith(("DONE", "FAILED", "WATCH-TIMEOUT", "MISSING"))), None)
        state = (terminal.split()[0] if terminal else ("WATCH-TIMEOUT" if r.returncode == 3 else "UNKNOWN"))
        return {"job_id": job_id, "state": state, "terminal": terminal, "rc": r.returncode, "tail": lines[-12:], "bound_s": bound, "max_hours": max_hours}


def _rsync_literal(pattern: str) -> str:
    """A path as an rsync exclude PATTERN that matches the name literally. rsync reads a backslash as an escape ONLY in a pattern that
    contains a wildcard (`*`, `?`, `[`); in a pattern with none every character, a backslash included, is literal (its man page: "an
    extra level of backslash removal when a pattern contains wildcard characters compared to a pattern that has none"). So a name
    with no wildcard is returned as it is -- doubling its backslashes made the exclude of a `\\`-named entry miss, and rsync copied
    the true name beside tar's escaped copy (the R2b run) -- and a name with a wildcard has the wildcards AND its backslashes escaped."""
    if not re.search(r"[*?\[]", pattern):
        return pattern
    return re.sub(r"([*?\[\]\\])", r"\\\1", pattern)


def rsync_argv(dest: Destination, comp: Component, gen_path: str, link_dest: Optional[str], escape_tops_: Sequence[str] = (),
               extra_link_dests: Sequence[str] = ()) -> List[str]:
    """``nice ionice -c3 rsync -rlt --partial --relative [--link-dest] <base>/./<member>... <dest>/`` -- never ``--delete``. The
    subtrees whose names the destination refuses, and the links whose targets it spells differently (``escape_tops_``, base-relative;
    :meth:`Manifest.tar_carried`), are EXCLUDED here and carried by :func:`tar_copy_argv` under their escaped spelling. ``extra_link_dests``: the trees a retry may hard-link against too
    (rsync accepts several `--link-dest`, tried in order): the tree of THIS content a previous attempt left standing and this tick renamed
    aside -- its complete files are byte-identical by the content sha, so they link instead of
    transferring, and the fresh verify stays the gate."""
    # the STAGING copy's directories are owner-writable (`--chmod=Du+w`: without -p rsync gives a new directory the source's mode masked by the
    # umask, so a tree copied from a SEALED generation -- 0555 directories -- inherited the seal before it was published, and the verify
    # could not create an attested empty directory the copy left out (an escaped name: rsync excludes it, tar carries files and links only,
    # the verify creates it); the publication seal is applied afterwards by seal_generation, every directory 0555 and refused when the mode
    # does not hold, so nothing of the seal is weakened here). Files keep their source mode: the verify seals them (chmod a-w) itself
    argv = ["nice", "-n", "10", "ionice", "-c3", "rsync", "-rlt", f"--chmod={STAGING_DIR_CHMOD}", "--partial", f"--partial-dir={PARTIAL_DIR}", "--relative",
            "--stats", "--timeout=600"]
    for e in comp.exclude:
        argv.append(f"--exclude=/{e.strip('/')}" if e.startswith("/") else f"--exclude={e}")
    for top in escape_tops_:
        argv.append(f"--exclude=/{_rsync_literal(top)}")
    if link_dest:
        argv.append(f"--link-dest={link_dest}")
    for extra in extra_link_dests:
        if extra and extra != link_dest:
            argv.append(f"--link-dest={extra}")
    if not dest.is_local:
        argv += ["-e", RSYNC_SSH]
    tops, _unmatched = comp.member_tops()
    argv += [f"{comp.base}/./" if top == comp.base else f"{comp.base}/./{os.path.relpath(top, comp.base)}" for top in tops]
    argv.append(f"{gen_path}/" if dest.is_local else f"{dest.host}:{gen_path}/")
    return argv


def rsync_forced_argv(dest: Destination, comp: Component, gen_path: str) -> List[str]:
    """The second pass: ``rsync -lt --ignore-times --files-from=-`` of the base-relative files on stdin -- the files whose
    manifest sha differs from the published generation's, which the first pass may have hard-linked to the OLD bytes when
    size and mtime happened to match. No ``--link-dest``: these are copied. Never ``--delete``."""
    argv = ["nice", "-n", "10", "ionice", "-c3", "rsync", "-lt", "--ignore-times", "--files-from=-", f"--chmod={STAGING_DIR_CHMOD}", "--partial",
            f"--partial-dir={PARTIAL_DIR}", "--stats", "--timeout=600"]
    if not dest.is_local:
        argv += ["-e", RSYNC_SSH]
    argv += [f"{comp.base}/", f"{gen_path}/" if dest.is_local else f"{dest.host}:{gen_path}/"]
    return argv


def tar_copy_argv(dest: Destination, comp: Component, gen_path: str) -> List[str]:
    """The copy of the entries rsync cannot write correctly on the destination (:data:`NAME_ESCAPE_RULE`): ``tar -C <base> --null -T -
    --no-recursion -cf -`` of exactly the affected files and links (NUL-separated on stdin, excludes already applied by the walk) piped
    into ``tar -x -C <generation> --transform ...`` on the destination (over one ssh for a remote one), the transforms escaping member
    names AND symlink targets (the ``s`` flag: a link's referent is renamed, so its target must be too -- :func:`escape_target`; the
    manifest keeps the true target). ``bash -o pipefail`` so a failure of either side is the step's. These entries are copied in full
    every generation (no hard link against the published one); the forced pass skips them."""
    extract = ["tar", "-x", "-C", gen_path, "--no-same-owner"] + [f"--transform={t}" for t in TAR_TRANSFORMS]
    extract_cmd = shlex.join(extract) if dest.is_local else shlex.join(["ssh", *SSH_OPTS, dest.host or "", shlex.join(extract)])
    create = ["nice", "-n", "10", "ionice", "-c3", "tar", "-C", comp.base, "--no-recursion", "--null", "-T", "-", "-cf", "-"]
    return ["bash", "-o", "pipefail", "-c", f"{shlex.join(create)} | {extract_cmd}"]


def _rsync_stats(stdout: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, pat in (("files_transferred", r"Number of regular files transferred:\s*([\d,]+)"),
                     ("bytes_transferred", r"Total transferred file size:\s*([\d,]+)"),
                     ("bytes_per_s", r"([\d,.]+) bytes/sec")):
        m = re.search(pat, stdout or "")
        if m:
            try:
                out[key] = float(m.group(1).replace(",", "")) if key == "bytes_per_s" else int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return out


def verify_time_s(n_bytes: int, n_files: int, rate_mb_s: float, min_s: int = 600, max_s: int = 4 * 3600) -> int:
    """``--time`` for the verify job: TWICE the estimate at the measured (or default) hash rate, bounded."""
    est = n_bytes / (max(rate_mb_s, 1.0) * 1e6) + n_files * SEC_PER_FILE + 60.0
    return int(min(max(math.ceil(2 * est), min_s), max_s))


def slurm_time(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------------------------
# step 0 of a tick: pin the running code revision on the destination
# --------------------------------------------------------------------------------------------


def running_revision(run: Callable[..., subprocess.CompletedProcess], source_repo: str) -> Tuple[str, List[str]]:
    """The checkout's HEAD sha and its uncommitted changes to TRACKED files (``git status --porcelain --untracked-files=no``);
    a dirty checkout has no exact revision the cluster could run."""
    r = run(["git", "-C", source_repo, "rev-parse", "HEAD"], SHELL_TIMEOUT_S)
    sha = (r.stdout or "").strip()
    if r.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise StoresError(f"no_revision: {source_repo} is not a git checkout with a HEAD (rc {r.returncode}: {(r.stderr or '').strip()[:200]})")
    r = run(["git", "-C", source_repo, "status", "--porcelain", "--untracked-files=no"], SHELL_TIMEOUT_S)
    if r.returncode != 0:
        raise StoresError(f"no_revision: git status in {source_repo} failed (rc {r.returncode}: {(r.stderr or '').strip()[:200]})")
    return sha, [line for line in (r.stdout or "").splitlines() if line.strip()]


@dataclass
class PinnedTree:
    revision: str
    tree: str                 # $SHARED/registry/stores/trees/<sha>/
    sbatch_path: str          # the verifier inside it, what `sbatch` is handed
    pushed_via: str           # mirror-repos.sh | git push gate/<sha7> | already on the mirror


def pin_cluster_tree(dest: Destination, remote: Remote, log: Callable[[str], None] = lambda s: None) -> PinnedTree:
    """Step 0 of a tick, before any copy or submit: make the destination run EXACTLY the code running here. (1) the running
    checkout's HEAD; a dirty checkout is refused by name (``dirty_checkout``). (2) push it to the repo mirror: the grading host's
    ``mirror-repos.sh --cluster <selector> <checkout name>`` (:func:`mirror_selector`, :func:`checkout_dir_name`) when the destination
    is remote and the script exists, else -- and whenever the
    revision is still not on the mirror -- a direct ``git push <mirror> <sha>:refs/heads/gate/<sha7>``. (3) on the mirror,
    ``git cat-file -e <sha>^{commit}`` (``revision_not_on_mirror``) and ``git cat-file -e <sha>:<sbatch>``
    (``verifier_missing_at_revision``): a mirror without the verifier at this revision refuses the tick HERE, not after a
    112 GiB copy. (4) ``git archive <sha> <PINNED_PATHS> | tar -x`` into ``trees/<sha>/`` (incoming dir + one rename; a tree
    already there with its ``REVISION`` file is reused, never rewritten). Returns the tree and the sbatch to submit."""
    q = shlex.quote
    source = dest.source_repo
    sha, dirty = running_revision(remote.run, source)
    if dirty:
        raise StoresError(f"dirty_checkout: {source} has {len(dirty)} uncommitted change(s) to tracked files (first: {dirty[0].strip()}); "
                          f"the cluster cannot run the code that is running here -- commit first")
    mirror = dest.mirror_dir

    def on_mirror() -> bool:
        return remote._cmd(f"git -C {q(mirror)} cat-file -e {sha}^{{commit}}").returncode == 0

    via = "already on the mirror" if on_mirror() else ""
    if not via and not dest.is_local and os.path.isfile(MIRROR_REPOS):
        # the repo-mirror tool knows the cluster by its own name (the site's mirror selector) and keys the clone by the checkout's
        # directory name, which for a linked worktree is the checkout it was added from and names the destination's default mirror
        selector, name = mirror_selector(dest.name), checkout_dir_name(source)
        r = remote.run([MIRROR_REPOS, "--cluster", selector, name], MIRROR_TIMEOUT_S)
        log(f"[pin] {os.path.basename(MIRROR_REPOS)} --cluster {selector} {name}: rc {r.returncode}")
        if r.returncode == 0 and on_mirror():
            via = "mirror-repos.sh"
    if not via:
        r = remote.run(["git", "-C", source, "push", "-q", dest.mirror_url, f"{sha}:refs/heads/gate/{sha[:7]}"], MIRROR_TIMEOUT_S)
        if r.returncode != 0:
            raise StoresError(f"mirror_push_failed: git push {dest.mirror_url} {sha[:7]} -> refs/heads/gate/{sha[:7]} rc {r.returncode}: "
                              f"{(r.stderr or '').strip()[:300]}")
        via = f"git push gate/{sha[:7]}"
        if not on_mirror():
            raise StoresError(f"revision_not_on_mirror: {sha[:7]} is not on {mirror} after the push")
    if remote._cmd(f"git -C {q(mirror)} cat-file -e {sha}:{q(dest.sbatch_rel)}").returncode != 0:
        raise StoresError(f"verifier_missing_at_revision: {dest.sbatch_rel} does not exist at {sha[:7]} on {mirror}; "
                          f"the tick cannot verify what it would copy")
    tree = dest.tree_dir(sha)
    paths = " ".join(q(p) for p in PINNED_PATHS)
    remote._cmd(f'set -e; T={q(tree)}; if test -f "$T/REVISION"; then exit 0; fi; I="$T.incoming.$$"; rm -rf "$I"; mkdir -p "$I"; '
                f'git -C {q(mirror)} archive {sha} {paths} | tar -x -C "$I"; printf "%s\\n" {sha} > "$I/REVISION"; '
                f'mv -T "$I" "$T" 2>/dev/null || rm -rf "$I"; test -f "$T/REVISION"',
                what=f"extract {sha[:7]} into {tree}")
    log(f"[pin] revision {sha[:7]} ({via}); verifier {dest.pinned_sbatch(sha)}")
    return PinnedTree(sha, tree, dest.pinned_sbatch(sha), via)


# --------------------------------------------------------------------------------------------
# state: <root>/state/store_mirror.json and the lock
# --------------------------------------------------------------------------------------------


def state_path(root: str) -> str:
    return os.path.join(root, "state", STATE_FILE)


def load_state(root: str) -> Dict[str, Any]:
    return _load_json(state_path(root)) or {"schema": STATE_SCHEMA, "components": {}}


def published_attestations(root: str) -> Dict[str, Optional[str]]:
    """Per component with a published generation, the attestation it was verified against (``published_attestation``: the VERIFIED
    report's ``listings_sha256``, recorded by the publish; None for a generation published before attestations were keyed) -- what the
    window's plan pins beside the content sha and every pin-side check reads (MANIFEST_RULES["directories"])."""
    return {slug: (rec.get("published_attestation") or None) for slug, rec in (load_state(root).get("components") or {}).items() if rec.get("published")}


def update_state(root: str, slug: str, patch: Mapping[str, Any], destination: str) -> Dict[str, Any]:
    st = load_state(root)
    st.setdefault("components", {})
    cur = dict(st["components"].get(slug) or {})
    cur.update(patch)
    st["components"][slug] = cur
    st["schema"] = STATE_SCHEMA
    st["updated_utc"] = R.utc_now()
    st["destination"] = destination
    _write_json(state_path(root), st)
    return st


def measured_rate(root: str, slug: str, dest: Destination) -> float:
    """The slug's last VERIFIED hash rate from the state file, else the destination's default (the ``--time`` estimate's rate)."""
    return float((load_state(root).get("components", {}).get(slug) or {}).get("verify_rate_mb_s") or dest.verify_rate_mb_s)


class MirrorLock:
    """A ``flock`` on ``<root>/state/store_mirror.lock``, non-blocking by default: the loser of two concurrent runs logs and exits 0 (the
    30-min tick yields to whatever holds it). ``wait_s`` > 0 polls for the lock that long (:data:`MIRROR_LOCK_POLL_S` apart) before
    giving up -- the weekly `--full` (registry-store-mirror-full.service, `--lock-wait-s`) waits out a tick that holds it instead of
    skipping its week, and the two units need no systemd `Conflicts=`: a conflict would let the tick's timer start STOP the weekly
    mid-verify, its Slurm job left without its watcher. ``take=False`` (the dry run) takes nothing and writes nothing, and
    ``held`` is True. ``waited_s`` is how long the wait took; :meth:`holder_line` reads who holds it."""

    def __init__(self, root: str, take: bool = True, wait_s: float = 0.0, poll_s: Optional[float] = None):
        self.path = os.path.join(root, "state", LOCK_FILE)
        self.take = take
        self.wait_s = max(0.0, float(wait_s or 0.0))
        self.poll_s = MIRROR_LOCK_POLL_S if poll_s is None else max(0.001, float(poll_s))
        self.fh = None
        self.held = not take
        self.waited_s = 0.0

    def holder_line(self) -> str:
        """The holder's own line in the lock file (``<pid> <utc>``), or ``unknown``."""
        try:
            with open(self.path) as fh:
                return (fh.readline().strip() or "unknown")[:120]
        except OSError:
            return "unknown"

    def __enter__(self) -> "MirrorLock":
        if not self.take:
            return self
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, "a+")
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                self.waited_s = time.monotonic() - t0
                if self.waited_s >= self.wait_s:
                    return self
                time.sleep(min(self.poll_s, self.wait_s - self.waited_s))
        self.waited_s = time.monotonic() - t0
        self.held = True
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"{os.getpid()} {R.utc_now()}\n")
        self.fh.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self.fh is not None:
            if self.held:
                try:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            self.fh.close()


class PublishLock:
    """The destination-side serialization of every publisher of ONE slug: an exclusive ``flock`` on
    ``gen/<slug>/.publish.lock``, held from BEFORE the snapshot (the manifest build) until `current` has switched, so a later
    publisher's snapshot is always taken after the earlier publication and `current` moves forward in snapshot order -- two overlapping
    producer jobs finishing in reverse order would otherwise switch it back onto the older content, and share one temporary pointer.
    Taken where the tree is: a LOCAL destination (the producer's own job, `--on-cluster`; the tests' directory destination). The kernel
    releases it when the holder dies, so nothing stale survives a killed job (VAST is nfs4.1 with server-side locks, so it holds across
    nodes). The grading-host-driven path, whose steps are one ssh each, cannot hold a lock across them; it takes THIS SAME FILE for its publish
    step instead (:func:`locked_step` through :meth:`Remote.publish`: flock before the readlink, released after the rename), so every
    publisher's compare-and-switch is serialized at the destination against every other's; its own side is
    serialized by :class:`MirrorLock`. A waiter blocks, polling, up to ``wait_s`` (logging once who holds it), then :class:`PublishLockBusy`."""

    def __init__(self, gen_dir: str, wait_s: float = PUBLISH_LOCK_WAIT_S, holder: Optional[str] = None,
                 log: Callable[[str], None] = lambda s: None, poll_s: float = PUBLISH_LOCK_POLL_S):
        self.path = os.path.join(gen_dir, PUBLISH_LOCK_FILE)
        self.wait_s, self.poll_s, self.log = float(wait_s), float(poll_s), log
        self.holder = holder or f"{socket.gethostname()}:{os.getpid()}"
        self.fh = None
        self.waited_s = 0.0
        self.waited_for: Optional[str] = None

    def _holder_line(self) -> str:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return fh.read().strip()[:200]
        except OSError:
            return ""

    def acquire(self) -> "PublishLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fh = open(self.path, "a+")
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                pass
            waited = time.monotonic() - t0
            if self.waited_for is None:
                self.waited_for = self._holder_line() or "an unnamed publisher"
                self.log(f"waiting for the publisher {self.waited_for} holding {self.path} (up to {self.wait_s:.0f} s)")
            if waited >= self.wait_s:
                fh.close()
                raise PublishLockBusy(f"lock_busy: {self.path} is held by {self.waited_for} after {waited:.0f} s; nothing copied or published -- "
                                      f"push again when that publication has finished")
            time.sleep(max(0.01, min(self.poll_s, self.wait_s - waited)))
        self.fh = fh
        self.waited_s = time.monotonic() - t0
        fh.seek(0)
        fh.truncate()
        fh.write(f"{self.holder} {R.utc_now()}\n")
        fh.flush()
        return self

    def release(self) -> None:
        if self.fh is not None:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            self.fh.close()
            self.fh = None

    def record(self) -> Dict[str, Any]:
        return {"path": self.path, "holder": self.holder, "waited_s": round(self.waited_s, 3), "waited_for": self.waited_for, "scope": "flow"}

    def __enter__(self) -> "PublishLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# --------------------------------------------------------------------------------------------
# stores mirror: one grading_host_copy component -> one immutable generation
# --------------------------------------------------------------------------------------------


@dataclass
class MirrorResult:
    slug: str
    outcome: str                      # unchanged | published | published_existing_verified | republished_dangling_current | dry_run |
                                      # invalidated | rsync_failed | tar_failed | verify_failed | verify_report_missing | verify_job_failed |
                                      # verify_watch_timeout | current_moved | lock_busy | producer_mismatch | current_unrecorded |
                                      # current_unverified | current_unsealed (`current` recorded but its directories not certified sealed:
                                      # a repair whose seal did not complete, or a generation that predates the seal) | dangling |
                                      # dangling_carry_deferred (the carry that would republish a dangling pointer, held by its publish
                                      # interval) | dangling_repair_pending (the tree stands, but it is the push flow's unfinished repair of
                                      # the pointer) | listings_unattested (the file content stands published and verified; its
                                      # listings were never attested -- `stores attest`) | listings_changed (the canonical's directory
                                      # listings moved from the attested ones) | listing_only_drift (a re-attestation found the
                                      # generation's directory set differs from the canonical's) | refused | skipped | error
    content_sha8: Optional[str] = None
    n_files: int = 0
    bytes: int = 0
    hashed: int = 0
    from_cache: int = 0
    published: Optional[str] = None   # the `current` target after this run
    job_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def level(self) -> str:
        """green / amber / red (:data:`OUTCOME_LEVELS`; an outcome the table does not name is red: an unnamed state is never green)."""
        return OUTCOME_LEVELS.get(self.outcome, "red")

    @property
    def ok(self) -> bool:
        """Exit 0 for the mirror: green outcomes and the amber ``unpushed_changes`` (the producer's, not the mirror's, to act on)."""
        return self.level in ("green", "amber")

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        detail = d.pop("detail")
        return {**d, "level": self.level, "elapsed_s": round(self.elapsed_s, 3), **detail}


def _report_admits(report: Mapping[str, Any], content_sha256: Optional[str], name_size_sha256: Optional[str], n_files: int) -> bool:
    """Does a VERIFIED report admit these manifest facts: ``ok``, BOTH hashes and the file count equal, nothing missing / corrupt /
    extra (an absent hash never matches)."""
    return (bool(content_sha256) and bool(name_size_sha256) and report.get("schema") == VERIFIED_SCHEMA and report.get("ok") is True
            and report.get("content_sha256") == content_sha256 and report.get("name_size_sha256") == name_size_sha256
            and int(report.get("n_files") or -1) == n_files and not (report.get("missing") or report.get("corrupt") or report.get("extra")))


def _verified_matches(report: Mapping[str, Any], m: Manifest, listings: bool = True) -> bool:
    """Does a VERIFIED report admit the manifest: ``ok``, BOTH hashes and the file count equal, nothing missing / corrupt / extra --
    and, when the manifest attests directory listings (``listings``; MANIFEST_RULES["directories"]), the report verified THOSE listings
    (its ``listings_sha256`` equals the manifest's, no ``listing_mismatch``). A report of the same files that verified other listings, or
    none, does not admit a manifest that attests listings. With ``listings=False`` the question is the FILES alone: a report whose only
    refusal is a ``listing_mismatch`` (``ok`` false for that reason and no other -- a re-attestation that found the directory set moved,
    :func:`attest_listings`) admits them."""
    files_ok = (report.get("schema") == VERIFIED_SCHEMA and report.get("content_sha256") == m.content_sha256
                and report.get("name_size_sha256") == m.name_size_sha256 and int(report.get("n_files") or -1) == m.n_files
                and not (report.get("missing") or report.get("corrupt") or report.get("extra"))
                and (report.get("ok") is True or (report.get("ok") is False and bool(report.get("listing_mismatch")))))
    if not files_ok or not listings or not m.directories:
        return files_ok
    return report.get("listings_sha256") == m.listings_sha256 and not report.get("listing_mismatch")


@dataclass(frozen=True)
class ManifestFacts:
    """The three facts of a generation's manifest that a VERIFIED report is matched by (:func:`_files_failed`, :func:`attest_integrity`):
    read from the generation's own ``MANIFEST-<sha8>.json`` (:func:`generation_facts`) when the grading host manifest is NOT the generation's
    content -- the reversed mirror's unequal branches (round 2 on 2c), where no :class:`Manifest` of the generation is at hand."""
    content_sha256: str
    name_size_sha256: str
    listings_sha256: str = ""


ManifestLike = Union[Manifest, ManifestFacts]


def generation_facts(meta: Optional[Mapping[str, Any]], sha8: str) -> Optional[ManifestFacts]:
    """:class:`ManifestFacts` of the published generation ``sha8`` from its ``MANIFEST-<sha8>.json`` meta; None when the meta does not
    name that content (absent, or another layout's)."""
    c, n = str((meta or {}).get("content_sha256") or ""), str((meta or {}).get("name_size_sha256") or "")
    if not c or not n or c[:8] != sha8:
        return None
    return ManifestFacts(c, n, str((meta or {}).get("listings_sha256") or ""))


def keyed_report_path(gen_dir: str, sha8: str, listings_sha256: str) -> str:
    """``VERIFIED-<sha8>.<lsha8>.json`` beside the generation: the report of a RE-ATTESTATION of its directory listings (`stores attest`,
    :func:`attest_listings`), keyed like the attestation it verified against (:func:`attestation_json_path`). The live
    ``VERIFIED-<sha8>.json`` is written once, by the publish's verify, and never rewritten."""
    return f"{gen_dir}/VERIFIED-{sha8}.{str(listings_sha256)[:8]}.json"


def listings_verdict(remote: Remote, gen_dir: str, sha8: str, verified: Mapping[str, Any], m: Manifest) -> Tuple[str, Optional[Dict[str, Any]]]:
    """How the published generation's directory listings stand against the manifest ``m`` (the canonical's current listings), given the
    live report ``verified`` admits its files: ``("attested", report)`` when a report verified THESE listings -- the live one, or the
    keyed ``VERIFIED-<sha8>.<lsha8>.json`` a re-attestation wrote; ``("drift", report)`` when the keyed report for these listings is
    present, admits the files and names a ``listing_mismatch`` (the generation's directory set is not the canonical's: the
    re-attestation's refusal, read by the mirror on every tick without another job); ``("none", None)`` when no report speaks for
    these listings. A manifest that attests no listings is ``attested`` by the live report alone."""
    if _verified_matches(verified, m):
        return "attested", dict(verified)
    if not m.listings_sha256:
        return "none", None
    keyed = remote.read_json(keyed_report_path(gen_dir, sha8, m.listings_sha256))
    if _verified_matches(keyed, m):
        return "attested", keyed
    if _files_failed(keyed, m):
        # a keyed report of THIS content that found the files wrong is never "absent": it is an integrity failure the mirror
        # names until a later full hash passes (:func:`attest_integrity`)
        return "integrity_failed", keyed
    if _verified_matches(keyed, m, listings=False) and keyed.get("listings_sha256") == m.listings_sha256 and keyed.get("listing_mismatch"):
        return "drift", keyed
    return "none", None


def _files_failed(report: Mapping[str, Any], m: ManifestLike) -> bool:
    """Is ``report`` a VERIFIED report of THIS content (both hashes the manifest's) that found the FILES wrong: a missing, corrupt or
    extra entry, or ``ok`` false for a reason other than a listing mismatch alone? A report of another content, or none, is not."""
    if not report or report.get("schema") != VERIFIED_SCHEMA or report.get("content_sha256") != m.content_sha256 or report.get("name_size_sha256") != m.name_size_sha256:
        return False
    return bool(report.get("missing") or report.get("corrupt") or report.get("extra")) or (report.get("ok") is False and not report.get("listing_mismatch"))


def _failure_record(path: str, report: Mapping[str, Any], sha8: str, found_utc: str) -> Dict[str, Any]:
    counts = {k: len(report.get(k) or []) for k in ("missing", "corrupt", "extra")}
    return {"content_sha8": sha8, "report_path": path, "verified_utc": report.get("verified_utc"), "job_id": report.get("job_id"),
            "listings_sha256": report.get("listings_sha256"), **counts, "first": {k: (report.get(k) or [])[:10] for k in counts}, "found_utc": found_utc}


def attest_integrity(remote: Remote, gen_dir: str, sha8: str, m: ManifestLike, prior_rec: Mapping[str, Any], verdict: str,
                     att_report: Optional[Mapping[str, Any]], full_passed: bool, now: Callable[[], str] = R.utc_now) -> Optional[Dict[str, Any]]:
    """The re-attestation INTEGRITY FAILURE that stands for the published generation ``sha8``, or None. A `stores attest` whose verify hashed
    every file and found some wrong (``attest_verify_failed``) records it in the component's state (``attest_integrity_failed``: the keyed
    report's path, stamp, job and counts); the mirror re-reads that report every tick, and also the keyed report for the canonical's CURRENT
    listings (``verdict == "integrity_failed"``, the record missing when the attest crashed before writing it), so a failure is never folded
    into "absent". It STANDS until a later full hash of every file passes: this tick's `--full` re-hash (``full_passed``), the
    last recorded `--full` pass of this content (``last_full_pass`` in the state) or an attesting report (``verdict == "attested"``) stamped
    later than the failing one; a report rewritten ok (a fresh attest of the same listings) or gone no longer fails. A record naming another
    content is a superseded generation's and is dropped. ``m`` is the generation's manifest, or its :class:`ManifestFacts` when the grading host
    manifest is not the generation's (the reversed mirror's unequal branches, ``verdict == "none"``: the record alone is read)."""
    recorded = prior_rec.get("attest_integrity_failed") or None
    if recorded and recorded.get("content_sha8") != sha8:
        recorded = None
    last_pass = prior_rec.get("last_full_pass") or {}
    passes = [str(last_pass.get("verified_utc") or "")] if last_pass.get("content_sha8") == sha8 else []
    if verdict == "attested" and att_report:
        passes.append(str(att_report.get("verified_utc") or ""))
    failing: List[Tuple[str, Mapping[str, Any]]] = []
    if verdict == "integrity_failed" and att_report:
        failing.append((keyed_report_path(gen_dir, sha8, m.listings_sha256), att_report))
    if recorded and recorded.get("report_path") and all(recorded["report_path"] != p for p, _r in failing):
        rep = remote.read_json(str(recorded["report_path"]))
        if _files_failed(rep, m):
            failing.append((str(recorded["report_path"]), rep))
    if not failing or full_passed:
        return None
    path, rep = max(failing, key=lambda t: str(t[1].get("verified_utc") or ""))
    if any(p > str(rep.get("verified_utc") or "") for p in passes):
        return None                      # a later full hash of every file passed: recovery proven
    found = recorded.get("found_utc") if (recorded and recorded.get("report_path") == path) else None
    return _failure_record(path, rep, sha8, str(found or now()))


def _integrity_why(fail: Mapping[str, Any], gen_path: str) -> str:
    first = next((v[0] for k in ("corrupt", "missing", "extra") for v in [fail.get("first", {}).get(k) or []] if v), None)
    return (f"attest_integrity_failed: a re-attestation of {gen_path} on {fail.get('verified_utc')} (job {fail.get('job_id') or '-'}, report "
            f"{fail.get('report_path')}) hashed every file and found {fail.get('missing', 0)} missing, {fail.get('corrupt', 0)} corrupt, "
            f"{fail.get('extra', 0)} extra entr(y/ies)" + (f" (first: {first!r})" if first else "") + ": the published generation's content is not "
            f"its manifest's and the live VERIFIED report predates the finding; red until a fresh `stores attest` or `stores mirror --full` re-hashes "
            f"the generation and passes -- nothing renamed, `current` untouched, the operator decides")


def _differs_note(differs: str, deferral: Optional[Mapping[str, Any]], grading_host_sha8: str, counts: Mapping[str, Any], comp: Component) -> str:
    """The reversed mirror's unequal outcome a standing integrity failure outranks, appended to :func:`_integrity_why`: what the grading host
    copy does and what supersedes the failing `current`."""
    if deferral:
        return (f"; the carry that would publish the grading host copy ({grading_host_sha8}; {counts}) over it is deferred -- {deferral.get('why')} -- "
                f"and a deferred carry is this red, never `deferred`")
    if differs == "frozen_differs":
        return (f"; the frozen component's grading host copy ({grading_host_sha8}) differs from it too ({counts}; frozen_differs): a frozen store is never "
                f"repaired by the mirror -- the operator decides which copy is the truth")
    return (f"; the grading host copy ({grading_host_sha8}) differs from it ({counts}) and nothing is copied: the producer's `stores push {comp.slug}` publishes "
            f"a fresh, fully verified generation that supersedes it (unpushed_changes)")


def _drift_why(report: Mapping[str, Any], gen_path: str) -> str:
    """The re-attestation's refusal, by direction (:func:`verify_generation` names both): directories the generation holds that the
    canonical lacks -- nothing removes an entry from an immutable generation -- and attested directories the generation lacks -- nothing is
    created inside a published, sealed tree (the A -> B -> A rebuild copies a fresh tree rather than rewriting one a window may be
    reading). Either way the next FILE change publishes a generation that attests the current listings."""
    gen_only, canon_only = list(report.get("dirs_generation_only") or []), list(report.get("dirs_canonical_only") or [])
    n_gen, n_canon = int(report.get("n_dirs_generation_only") or len(gen_only)), int(report.get("n_dirs_canonical_only") or len(canon_only))
    parts = []
    if n_gen:
        parts.append(f"the generation holds {n_gen} director(y/ies) the canonical no longer has (first: {gen_only[0]!r}): nothing removes an entry "
                     f"from an immutable generation")
    if n_canon:
        parts.append(f"the canonical holds {n_canon} director(y/ies) the generation lacks (first: {canon_only[0]!r}): nothing is created inside a "
                     f"published, sealed generation")
    if not parts:
        mism = list(report.get("listing_mismatch") or [])
        parts.append(f"{len(mism)} attested director(y/ies) under {gen_path} list other than the canonical's" + (f" (first: {mism[0]!r})" if mism else ""))
    return ("listing_only_drift: the file content of the published generation stands verified; its directory listings are not the canonical's: "
            + "; ".join(parts) + " -- a listing-coverage limitation, not an integrity failure: the listings are not attested, the listing readers "
            "keep the grading host route, and the next file change publishes a generation that attests the current listings")


def current_evidence(remote: Remote, gen_dir: str, sha8: str, meta: Mapping[str, Any], n_files: int) -> Dict[str, Any]:
    """What stands beside `current` -> ``sha8`` (:data:`CURRENT_EVIDENCE`): the VERIFIED report read and compared with the
    generation's manifest facts (``meta`` = its ``MANIFEST-<sha8>.json``; ``n_files`` the parsed entry count), the SEALED and the
    PUBLISHED records read. Returns the three documents, the ``problems`` by outcome name in the order the mirror reports them, and the
    ``record`` the result carries (``generation_record``)."""
    verified = remote.read_json(f"{gen_dir}/VERIFIED-{sha8}.json")
    sealed = remote.read_json(f"{gen_dir}/SEALED-{sha8}.json")
    pub = remote.read_json(f"{gen_dir}/PUBLISHED-{sha8}.json")
    matches = _report_admits(verified, (meta or {}).get("content_sha256"), (meta or {}).get("name_size_sha256"), n_files) and (
        not (meta or {}).get("directories")       # a manifest that attests listings is admitted only by a report that verified THOSE listings
        or (verified.get("listings_sha256") == (meta or {}).get("listings_sha256") and not verified.get("listing_mismatch")))
    problems = [name for name, missing in (("current_unverified", not matches), ("current_unsealed", not sealed), ("current_unrecorded", not pub)) if missing]
    record = {"verified": bool(verified), "verified_matches": matches, "verified_utc": verified.get("verified_utc"), "sealed": bool(sealed),
              "sealed_utc": sealed.get("sealed_utc"), "published_record": bool(pub), "change_utc": pub.get("change_utc"),
              "published_utc": pub.get("published_utc"), "publish_mode": pub.get("mode"), "problems": problems}
    return {"verified": verified, "sealed": sealed, "published": pub, "problems": problems, "record": record}


def evidence_why(problems: Sequence[str], gen_dir: str, sha8: str, comp: Component, adoptable: bool) -> str:
    """The reversed mirror's refusal of a `current` without its evidence, by name, with the step that resolves it."""
    missing = {"current_unverified": f"no VERIFIED-{sha8}.json admitting its manifest", "current_unsealed": f"no SEALED-{sha8}.json",
               "current_unrecorded": f"no PUBLISHED-{sha8}.json"}
    why = (f"`current` -> {sha8} under {gen_dir} stands without its evidence: {'; '.join(missing[p] for p in problems)} -- a publisher "
           f"mid-flight or crashed there, or a generation that predates the rule; not called verified")
    if adoptable:
        why += (f"; `stores mirror --component {comp.slug} --adopt-unrecorded` adopts a VERIFIED current (the seal and the record written "
                f"under the publish lock, nothing switched)")
    elif "current_unverified" in problems:
        why += (f"; nothing adopts an unverified generation: the producer re-publishes it (`stores push {comp.slug}`), or "
                f"`stores verify-authority {comp.slug}` verifies and adopts it")
    return why


def adopt_current(remote: Remote, gen_dir: str, sha8: str, comp: Component, man: Manifest, ev: Mapping[str, Any], source: str,
                  lock_wait_s: float = PUBLISH_LOCK_WAIT_S, now: Callable[[], str] = R.utc_now, log: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """`--adopt-unrecorded` for a verify-only component -- the push flow's adoption applied by the mirror: a `current`
    whose VERIFIED report admits its manifest (``ev`` from :func:`current_evidence`; the caller refuses otherwise) is SEALED when its seal is
    missing (:func:`seal_generation`) and RECORDED when its record is missing (``PUBLISHED-<sha8>.json``, mode ``adopt``, ``change_utc`` the
    grading host copy's latest mtime, written under the slug's publish lock WITHOUT switching: :meth:`Remote.record_published`); nothing else
    changes. Returns what was adopted."""
    adopted: Dict[str, Any] = {}
    if not ev["sealed"]:
        log(f"[{comp.slug}] adopting {sha8}: sealing its directories")
        adopted["sealed"] = {k: v for k, v in seal_generation(remote, gen_dir, sha8, now=now).items() if k in ("n_dirs", "sealed_utc", "host")}
    if not ev["published"]:
        session = comp.producer.split(":", 1)[1] if comp.producer and comp.producer.startswith("grading_host:") else None
        meta = {"mode": "adopt", "note": f"adopted by `stores mirror --adopt-unrecorded`: a verified `current` of producer {comp.producer} that predates "
                                         f"the publication record", "producer_session": session, "producer_host": socket.gethostname(),
                "source": source, "declared_base": comp.base}
        rec = published_record(gen_dir, sha8, comp, man, None, man.rels, meta, now=now)
        log(f"[{comp.slug}] adopting {sha8}: its PUBLISHED record under the publish lock")
        step = remote.record_published(gen_dir, sha8, record_text(rec), lock_wait_s=lock_wait_s, utc=now())
        pub = rec if step["written"] else remote.read_json(f"{gen_dir}/PUBLISHED-{sha8}.json")       # another publisher recorded it meanwhile
        adopted["published_record"] = _published_digest(pub)
        adopted["publish_lock"] = {k: step[k] for k in ("path", "holder", "scope")}
    return adopted


def read_verified_report(remote: Remote, path: str, tries: Optional[int] = None, pause_s: Optional[float] = None) -> Tuple[Dict[str, Any], int]:
    """A VERIFIED report right after its job read DONE: ``{}`` the first time is re-read up to :data:`REPORT_REREAD_TRIES` more times,
    :data:`REPORT_REREAD_PAUSE_S` apart (NFS close-to-open: the file the job wrote a second ago may not be visible yet). Returns (the
    report, how many re-reads it took); an absent report after the last re-read is ``{}`` -- the caller never treats that as a failed
    verify."""
    tries = REPORT_REREAD_TRIES if tries is None else tries
    pause_s = REPORT_REREAD_PAUSE_S if pause_s is None else pause_s
    report = remote.read_json(path)
    rereads = 0
    while not report and rereads < tries:
        time.sleep(pause_s)
        rereads += 1
        report = remote.read_json(path)
    return report, rereads


def _published_entries(sdir: str, remote: Remote, gen_dir: str, sha8: str) -> Optional[Dict[str, Tuple[str, int]]]:
    """``{rel: (sha, size)}`` of the published generation's manifest: the grading host's own copy in the state dir, else the one
    beside the tree on the destination; None when neither exists (every file is then forced)."""
    local = os.path.join(sdir, f"MANIFEST-{sha8}.sha256")
    if os.path.isfile(local):
        entries = read_manifest_lines(local)
    else:
        text = remote.read(f"{gen_dir}/MANIFEST-{sha8}.sha256")
        if text is None:
            return None
        entries = parse_manifest_lines(text, f"{gen_dir}/MANIFEST-{sha8}.sha256")
    return {rel: (sha, size) for sha, size, rel in entries}


def seal_generation(remote: Remote, gen_dir: str, sha8: str, now: Callable[[], str] = R.utc_now, by: Optional[str] = None) -> Dict[str, Any]:
    """The directory seal at publish: every directory of ``gen/<slug>/<sha8>`` ``chmod 0555``, deepest first -- after the verify
    sealed the files -- then ``SEALED-<sha8>.json`` beside the generation (directory count, utc, host). Refuses by name when a directory
    is still writable after the chmod (a filesystem that ignores the mode: cluster B's MooseFS; VAST honors it, rehearsal R14). The
    parent ``gen/<slug>/`` is never touched: `current` still switches and a failed tree is still renamed. Idempotent: a re-seal records
    ``resealed`` and keeps the first seal's stamp."""
    gen_path = f"{gen_dir}/{sha8}"
    prior = remote.read_json(f"{gen_dir}/SEALED-{sha8}.json")
    r = remote.seal_directories(gen_path)
    if r["unsealed"]:
        raise StoresError(f"seal_refused: {r['unsealed']} of {r['n_dirs']} director(y/ies) under {gen_path} are not {DIR_SEAL_MODE:04o} after the "
                          f"chmod; the filesystem does not honor the directory seal, so the generation is not published")
    stamp = now()
    rec = {"schema": SEALED_SCHEMA, "generation": gen_path, "content_sha8": sha8, "n_dirs": r["n_dirs"], "mode": f"{DIR_SEAL_MODE:04o}",
           "sealed_utc": stamp, "first_sealed_utc": (prior.get("first_sealed_utc") or prior.get("sealed_utc") or stamp) if prior else stamp,
           "resealed": bool(prior), "host": r["host"], "sealed_by": by or f"{getpass.getuser()}@{socket.gethostname()}"}
    remote.write(f"{gen_dir}/SEALED-{sha8}.json", json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def published_record(gen_dir: str, sha8: str, comp: Component, man: Manifest, previous: Optional[str], changed_rels: Sequence[str],
                     meta: Mapping[str, Any], now: Callable[[], str] = R.utc_now) -> Dict[str, Any]:
    """The ``PUBLISHED-<sha8>.json`` record of a publication: the publication clock's start and the publication's identity. ``change_utc`` is the
    PRODUCER'S committed store change -- the latest mtime among the files that differ from the previous published generation (every file
    when there is none, or for an adoption / the one-time move), from the stat snapshot the manifest already took; ``published_utc`` is
    now. ``meta`` names the ``mode`` (:data:`PUBLISH_MODES`), ``note``, ``producer_session``, ``producer_host``, ``pushed_by``, ``job_id``.
    The record is WRITTEN by the destination step that publishes or adopts (:meth:`Remote.publish`, :meth:`Remote.record_published`), inside
    the publish lock, never by a plain write: :func:`record_text` spells it for that step."""
    mode = str(meta.get("mode") or "push")
    if mode not in PUBLISH_MODES:
        raise StoresError(f"publish mode {mode!r} is not one of {PUBLISH_MODES}")
    change_utc = man.max_mtime_utc(changed_rels) or man.max_mtime_utc(man.rels)
    return {"schema": PUBLISHED_SCHEMA, "slug": comp.slug, "content_sha8": sha8, "generation": f"{gen_dir}/{sha8}",
            "change_utc": change_utc, "published_utc": now(), "mode": mode,
            "producer_host": str(meta.get("producer_host") or socket.gethostname()), "producer_session": meta.get("producer_session"),
            "pushed_by": str(meta.get("pushed_by") or f"{getpass.getuser()}@{socket.gethostname()}"), "note": str(meta.get("note") or ""),
            "job_id": meta.get("job_id"), "previous": previous, "changed_files": len(list(changed_rels)), "n_files": man.n_files, "bytes": man.bytes,
            "source": man.component.base, "declared_base": str(meta.get("declared_base") or comp.base), "authority": comp.authority, "producer": comp.producer or None,
            "transitional_carry": comp.transitional_carry, "carry_deadline": comp.carry_deadline, "n_name_escapes": len(man.name_escapes()),
            "n_target_escapes": len(man.target_escapes())}


def record_text(rec: Mapping[str, Any]) -> str:
    """A record as the destination step writes it: one key per line (:data:`Remote._PRODUCER_GREP` reads the producer from that shape)."""
    return json.dumps(rec, indent=1, sort_keys=True) + "\n"


def _published_digest(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: rec.get(k) for k in ("change_utc", "published_utc", "mode", "previous", "changed_files", "note", "producer_session")}


def _foreign_producer(rec: Mapping[str, Any], comp: Component) -> Optional[str]:
    """Why a grading-host-driven publisher may not switch `current` onto, or away from, the generation whose PUBLISHED record is ``rec``; None
    when it may. The record names ``producer: cluster`` -- a cluster job published that generation -- and this component is not declared
    cluster-produced: a grading host copy of such a slug is a lagging mirror of what the cluster writes, and publishing it would move `current`
    onto older content (``published_existing_verified`` re-points at a tree a cluster job verified earlier; a fresh copy supersedes the
    job's newer generation). The on-cluster publisher is never refused by this rule (it is where the tree is), and a record naming no
    producer, or a ``grading_host:*`` / ``frozen`` one, is never foreign to the grading host: the grading host tree is the grading host tree whichever session
    wrote it, so a renamed producer session strands nothing. This is the PRE-check that saves
    the copy; the locked publish step applies the same rule again, under the lock, to the record it reads there (round 3)."""
    producer = rec.get("producer") if rec else None
    if producer == "cluster" and comp.producer != "cluster":
        return (f"producer_mismatch: PUBLISHED-{rec.get('content_sha8')}.json names producer cluster (mode {rec.get('mode')}, job "
                f"{rec.get('job_id') or '-'}, {rec.get('published_utc')}) and this component declares producer {comp.producer or 'none'}: a cluster "
                f"job publishes this slug, so the grading host copy is a lagging mirror and never switches `current` onto or away from a cluster "
                f"publication -- declare producer: cluster (the job pushes; the grading host holds a pull mirror) or stop the cluster producer")
    return None
def _await_report(remote: Remote, path: str, job_id: Optional[str], wait_s: float, poll_s: float,
                  sleep: Callable[[float], None], log: Callable[[str], None]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The VERIFIED report THIS job wrote, read with a bounded wait after the watch said DONE. A report the compute node
    wrote is not visible from the login node at once (NFS attribute / negative-entry caching, up to ~60 s measured; the
    first read right after DONE returned ``{}`` for every one of the 48 ``verify_failed`` outcomes of 2026-09-29..10-05,
    none a real mismatch), so an ABSENT report is re-read every ``poll_s`` for up to ``wait_s``; so is one that is not yet
    readable (a transport failure, text that is not JSON) and one whose ``job_id`` is another job's -- the report a RENAMED
    tree left behind, which the verifier overwrites and which the cache can still show. Returns ``(report, wait)``: the
    fresh report (``{}`` when none was seen inside the bound) and ``{attempts, waited_s, last}`` for the record."""
    attempts, waited, last = 0, 0.0, "absent"
    while True:
        attempts += 1
        try:
            report = remote.read_json(path)
        except StoresError as exc:
            report, last = {}, f"unreadable: {exc}"[:300]
        else:
            if report and str(report.get("job_id")) == str(job_id):
                return report, {"attempts": attempts, "waited_s": waited, "last": "present"}
            last = "absent" if not report else f"stale (written by job {report.get('job_id')!r}, not {job_id!r})"
        if waited + poll_s > wait_s:
            return {}, {"attempts": attempts, "waited_s": waited, "last": last}
        log(f"[await] {os.path.basename(path)} {last} {waited:.0f} s after DONE; re-reading in {poll_s:.0f} s (bound {wait_s:.0f} s)")
        sleep(poll_s)
        waited += poll_s


def _slurm_of(dest: Destination, remote: "Remote"):
    """The reads protocol's Slurm transport on the mirror's own runner (one code path for `scancel` / `squeue` / `sacct`)."""
    from . import reads_protocol as RPROTO
    return RPROTO.Slurm(None if dest.is_local else dest.host, runner=remote.run, timeout_s=SHELL_TIMEOUT_S)


def names_live_verifier(generation: Optional[Mapping[str, Any]]) -> bool:
    """Does a component's generation record name a verify job no run has proved terminal: ``state`` ``awaiting_verify`` with a ``job_id``
    -- what a controller cut short leaves behind, and what a bounded run writes back when its cancel could not be confirmed or its
    watcher lost the job (``verify_unconfirmed`` / ``verify_in_flight`` are OUTCOMES of a run that found such a record; the record itself
    keeps ``awaiting_verify``). Every other state names a job that ended (``published``, ``verify_cancelled``, ``verify_job_failed``, ...)
    or no job at all. The one predicate :func:`settle_awaiting_verify` settles by and :func:`mirror_authority`'s ``finish`` refuses to
    write over."""
    g = generation or {}
    return str(g.get("state") or "") == "awaiting_verify" and bool(g.get("job_id"))


def settle_awaiting_verify(generation: Mapping[str, Any], dest: Destination, remote: "Remote", log: Callable[[str], None] = lambda s: None,
                           recorded_by: str = "an earlier run") -> Dict[str, Any]:
    """A recorded ``awaiting_verify`` job of a component, settled before this run copies, renames or switches anything:
    ``{job_id, state, where, refusal, why}`` -- no record: nothing to settle (``job_id`` None). The job is read ONCE from the queue
    and the accounting together (``reads_protocol.Slurm.queue_and_accounting``) and judged by THE terminal-confirmation rule
    (:func:`reads_protocol.terminal_verdicts`, the one ``wait_terminal`` polls on): an explicit terminal accounting row AND no queue row
    -- settled, no refusal, the run goes on and the existing-verified / recopy paths decide; a known non-terminal state (the queue's, or
    an accounting row not yet terminal) -- ``refusal = "verify_in_flight"`` (its verifier reads the tree this run would rename); the
    queue / accounting unreadable, OR no row for the job in EITHER (the accounting not yet caught up with a job that left the queue, or
    a row it purged) -- ``refusal = "verify_unconfirmed"`` (fail closed: the record keeps ``awaiting_verify`` with the job id and the
    next run asks again). An earlier form read ``Slurm.states`` and took its None ("no row anywhere") for settled, so a
    verifier the accounting had not yet shown was treated as gone and the carry renamed and recopied the generation it may still read.
    ``recorded_by`` is the wording's author of the record: :func:`mirror_component` passes its own run when its watcher LOST the job
    (MISSING) and settles it by this same rule instead of taking the loss for a terminal state."""
    out: Dict[str, Any] = {"job_id": None, "state": None, "where": None, "refusal": None, "why": None}
    if not names_live_verifier(generation):
        return out
    from . import reads_protocol as RPROTO
    job_id = str(generation["job_id"])
    out["job_id"] = job_id
    try:
        queued, acct = _slurm_of(dest, remote).queue_and_accounting([job_id])
    except RPROTO.SlurmQueryFailed as exc:
        out.update(refusal="verify_unconfirmed", why=f"verify job {job_id} recorded awaiting_verify by {recorded_by} and its state could not be read ({exc}); "
                                                     "nothing copied, renamed or switched until it reads terminal")
        log(out["why"])
        return out
    verdict = RPROTO.terminal_verdicts([job_id], queued, acct)[job_id]
    out.update(state=verdict["state"], where=verdict["where"])
    if verdict["confirmed"]:
        out["why"] = f"verify job {job_id} recorded awaiting_verify by {recorded_by} reads {verdict['why']}: settled"
        log(out["why"])
        return out
    if verdict["state"] is not None:
        out.update(refusal="verify_in_flight", why=f"verify job {job_id} recorded awaiting_verify by {recorded_by} is {verdict['why']}: its verifier still "
                                                   "reads the generation this run would rename and recopy; nothing copied, renamed or switched")
        log(out["why"])
        return out
    out.update(refusal="verify_unconfirmed",
               why=f"verify job {job_id} recorded awaiting_verify by {recorded_by} has {verdict['why']} and no queue row: not proved terminal (the "
                   "accounting lags a job that just left the queue, or purged its row); nothing copied, renamed or switched until `sacct -j "
                   f"{job_id}` reads it terminal -- the next run asks again")
    log(out["why"])
    return out


def cancel_and_confirm(job_id: str, dest: Destination, remote: "Remote", sleep: Callable[[float], None] = time.sleep,
                       log: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """``scancel`` the job this run submitted and wait until it is CONFIRMED terminal (``reads_protocol.Slurm.wait_terminal``: an explicit
    terminal accounting row and no row in the queue). ``{job_id, confirmed, state, why}``; an unconfirmed cancel names why."""
    from . import reads_protocol as RPROTO
    slurm = _slurm_of(dest, remote)
    out: Dict[str, Any] = {"job_id": job_id, "confirmed": False, "state": None, "why": None}
    try:
        slurm.cancel([job_id])
        states = slurm.wait_terminal([job_id], sleep=sleep)
        out.update(confirmed=True, state=states.get(job_id), why=f"verify job {job_id} cancelled at the bound and confirmed {states.get(job_id)}")
    except (RPROTO.SlurmQueryFailed, RPROTO.ReconcileRefusal) as exc:
        out["why"] = f"verify job {job_id} cancelled at the bound but NOT confirmed terminal: {exc}"
    log(out["why"])
    return out


#: why a `current` a hold read is NOT advertised as the component's published generation, one phrase per problem
#: (:func:`held_publication`); ``{sha8}`` is the pointer's target.
HELD_PUBLICATION_WHY = {
    "no_current": "`current` names no generation",
    "dangling": "`current` -> {sha8} names no tree",
    "no_manifest": "no MANIFEST-{sha8}.sha256 stands beside it",
    "current_unverified": "no VERIFIED-{sha8}.json admits its manifest",
    "current_unsealed": "no SEALED-{sha8}.json seals its directories",
    "current_unrecorded": "no PUBLISHED-{sha8}.json records its publication",
    "current_dangling": "a recorded repair of a dangling pointer is unresolved (the dangle DROPPED `published`; a publish restores it)",
}


def held_publication(remote: "Remote", gen_dir: str, current: Optional[str], recorded: Mapping[str, Any]) -> Dict[str, Any]:
    """What a run that copies, renames and switches NOTHING may advertise as the component's published generation -- the hold of a recorded
    verifier it could not prove terminal, on either route (:func:`mirror_authority`'s ``hold``, :func:`mirror_component`'s settle refusal).
    The live `current` is a PUBLICATION only with the evidence a publish leaves beside it: its tree a directory, a live
    ``VERIFIED-<sha8>.json`` admitting the generation's manifest (:func:`current_evidence`, the one predicate this code calls a generation
    verified by), its ``SEALED-<sha8>.json`` and its ``PUBLISHED-<sha8>.json`` record. AND a recorded dangling repair no run has resolved
    (``current_dangling``, which the dangle itself DROPS ``published`` for) keeps it dropped whatever the pointer reads now: only a
    publish, or the verdict that the pointer names a directory, clears that record -- neither of which a hold is.

    THE SEAL GATES PUBLICATION HERE TOO, as it does on every other route (the sibling rule from registry-hardening-2c, which this route
    was written without). A `current` whose own ``SEALED-<sha8>.json`` is not beside it is ``current_unsealed`` and is not advertised: that
    absence IS a publication left unfinished -- the repair of a dangling pointer moves the replaced tree's seal aside with its reports
    (:meth:`Remote.reports_aside`) and the replacement earns its own inside the publication step, so an unsealed `current` names a tree
    whose seal died on the transport and whose directories are still writable. :func:`admit_generation` refuses such a generation at the
    far end (``unsealed``), so advertising it would buy the window a refusal rather than a read.

    ``{published, problems, why, record, current_dangling}``: ``published`` is the sha8 with every problem absent, else None; ``problems``
    names each one (:data:`HELD_PUBLICATION_WHY`); ``record`` is :func:`current_evidence`'s, when the manifest could be read.

    Review re-check MEDIUM: the hold took the `current` target for a published generation with none of this. The dangling-pointer repair
    drops ``published``, recopies the same sha under `current` and submits the verify; when the watch cannot prove that job terminal it
    leaves ``awaiting_verify`` and NO publication -- and the next tick's hold wrote that UNVERIFIED sha back into ``published``.
    ``cluster_reads.plan`` pins any non-empty ``published`` and validates the grading host copy against the pin, so an adapter was routed at an
    incomplete destination tree and refused by the cluster attestation at the far end, the read wasted and the window's component refused,
    instead of the component declining at plan time."""
    dangle = (recorded or {}).get("current_dangling") or None
    out: Dict[str, Any] = {"published": None, "problems": [], "why": None, "record": None, "current_dangling": dangle}
    problems: List[str] = []
    if not current:
        problems.append("no_current")
    elif not remote.isdir(f"{gen_dir}/{current}"):
        problems.append("dangling")
    else:
        got = _generation_manifest(remote, gen_dir, current)
        if got is None:
            problems.append("no_manifest")
        else:
            _text, theirs, theirs_meta = got
            ev = current_evidence(remote, gen_dir, current, theirs_meta, len(theirs))
            out["record"] = ev["record"]
            problems += list(ev["problems"])
    if dangle:
        problems.append("current_dangling")
    out["problems"] = problems
    if not problems:
        out["published"] = current
        return out
    out["why"] = (f"`current` under {gen_dir} is not advertised as this component's published generation: "
                  + "; ".join(HELD_PUBLICATION_WHY[p].format(sha8=current) for p in problems)
                  + " -- a run that copies, renames and switches nothing never raises `published` on evidence it does not have")
    return out


def mirror_component(root: str, dest: Destination, comp: Component, runner=None, dry_run: bool = False,
                     hasher: Callable[[str], str] = R.file_sha256, log: Callable[[str], None] = lambda s: None,
                     now: Callable[[], str] = R.utc_now, pin: Optional[PinnedTree] = None,
                     allow: Sequence[str] = ("grading_host_copy",), publish_meta: Optional[Mapping[str, Any]] = None,
                     verify_inline: bool = False, incoming: bool = False, workers: int = 16,
                     lock_wait_s: float = PUBLISH_LOCK_WAIT_S, lock_holder: Optional[str] = None, adopt_unrecorded: bool = False,
                     full: bool = False, sleep: Callable[[float], None] = time.sleep, bound_s: Optional[float] = None) -> MirrorResult:
    """The generation flow for ONE component whose canonical (or mirror) copy is the local tree ``comp.base`` (module docstring).

    ``bound_s`` (`stores mirror --bound-s`; the window's carry): THE CONTROLLER OWNS ITS BOUND. The verify job's watch is bounded to it;
    past the bound the controller CANCELS its own recorded job by id and CONFIRMS it terminal (an explicit terminal accounting row and
    out of the queue; ``reads_protocol.Slurm``) before it returns ``verify_cancelled`` -- never an outer ``timeout`` that kills the
    controller and its watcher while the submitted verifier runs on with the MirrorLock released (the branch review's P2: the next carry
    would rename and repopulate the generation the live verifier still reads). A cancel that cannot be confirmed is ``verify_unconfirmed``
    and the generation record KEEPS ``awaiting_verify`` with the job id. And every run first SETTLES a recorded ``awaiting_verify`` job of
    the component (:func:`settle_awaiting_verify`, by the terminal-confirmation rule ``wait_terminal`` itself polls on): while it reads a
    non-terminal state the run is refused by name (``verify_in_flight``: nothing copied, renamed or switched); while the queue and the
    accounting cannot be read, or NEITHER has a row for it, the run is ``verify_unconfirmed`` (absence proves nothing; the record stands);
    once an accounting row reads terminal and the queue is silent the run goes on (the existing-verified path publishes a matching report;
    a renamed tree is recopied). The watch itself is bounded IN THIS PROCESS (:meth:`Remote.watch`; the script's ``--max-hours`` is a
    whole-hour backstop past the bound -- its arithmetic is integer), and a watcher that LOSES the job (MISSING: no accounting row for ten
    minutes) is no terminal verdict: the run reads the queue and the accounting once by the same rule -- a live or unproven job keeps the
    ``awaiting_verify`` record (``verify_in_flight`` / ``verify_unconfirmed``; nothing cancelled, renamed or switched) and a proved
    terminal state stands in for the watcher's line.
    ``full`` (the weekly `--full`): the manifest reads every file (the stat cache ignored, then refreshed) and an ``unchanged``
    generation is RE-HASHED on the destination (:func:`full_rehash`; ``full_verify_failed`` names a mismatch, renames nothing).
    Every outcome is recorded in ``state/store_mirror.json``; nothing is deleted on either side (an invalidated or failed tree is
    RENAMED). ``pin`` is the tick's pinned code tree (:func:`pin_cluster_tree`): its sbatch is what the verify job runs.

    ``allow`` names the authorities this call carries (`stores mirror`: ``grading_host_copy``; `stores push`: ``grading_host_copy`` and
    ``cluster_a``). ``publish_meta`` (the push) names the publication's mode and identity in ``PUBLISHED-<sha8>.json`` (:func:`published_record`) and makes an
    ``unchanged`` run ADOPT a generation that lacks its seal or its record (sealed + recorded, mode ``adopt``). ``verify_inline`` runs
    :func:`verify_generation` in this process instead of a Slurm job (the producer's own cluster job; a local destination only);
    ``incoming`` copies into ``.incoming-<sha8>`` and renames it into place after the post-copy walk (nothing half-copied ever stands at
    the generation's name). Every publication seals the directories (:func:`seal_generation`) and passes the guard (:meth:`Remote.publish`),
    whose switch is a compare-and-switch against the `current` this run read at its start (``current_moved`` when another publisher
    moved it meanwhile: nothing is switched back). On a LOCAL destination the whole flow -- snapshot, copy, verify, publish -- runs under
    the slug's :class:`PublishLock` (``lock_wait_s``, ``lock_holder``; ``lock_busy`` when it stays held), so overlapping publishers
    publish in snapshot order; on a REMOTE destination the publish step takes the same lock for the readlink, the comparison and the
    rename (:func:`locked_step`; ``lock_busy`` with the verified tree left standing when it stays held), so no switch interleaves with
    another publisher's. EVERY publication writes ``PUBLISHED-<sha8>.json`` (mode ``mirror`` for a plain mirror) inside that step, before
    the switch (:func:`published_record`), and the step requires the record of the `current` it read: a `current` with none is
    ``current_unrecorded`` -- nothing copied or switched -- until adopted explicitly (``adopt_unrecorded``: `stores push --adopt-unrecorded`
    writes the record under the lock without switching when the generation is this run's content, and a differing content then
    supersedes the unrecorded legacy `current`). A grading-host-driven run (every mode but ``push_on_cluster``) refuses ``producer_mismatch`` --
    nothing copied -- when `current`'s PUBLISHED record, or the record of the verified generation it would re-point at, names a cluster
    producer and the component is not declared cluster-produced (:func:`_foreign_producer`; the step re-applies both rules under the lock).
    Entries whose names the destination refuses, and links whose targets it spells differently, travel by :func:`tar_copy_argv` under
    their escaped spelling (:data:`NAME_ESCAPE_RULE`).

    ``sleep`` paces the
    bounded wait for the VERIFIED report after the watch says DONE (:func:`_await_report`; tests inject a recorder).

    Both shortcuts PROBE THE TREE (``test -d gen/<slug>/<sha8>``), never the report alone: ``current -> <sha8>`` with a
    matching report is ``unchanged`` only while the tree is a directory, and a matching report publishes an existing
    generation only while its tree is there. A ``current`` or report that names a tree that is gone (renamed away) falls
    through to the copy, with ``--link-dest`` pointing only at a generation that IS a directory (rsync warns about a
    missing link-dest and copies in full anyway) and every report that spoke for the replaced tree moved aside (the live one and the
    keyed re-attestations, with their snapshots: :meth:`Remote.reports_aside`), so the fresh verify is the only report.
    A DANGLING ``current`` is never read from the record as published: the renamed trees beside it are listed
    (:meth:`Remote.renamed_aside`) and named in the log, the record's ``published`` is dropped and ``current_dangling``
    set for as long as the pointer dangles, and the copy that repairs it lands as ``republished_dangling_current``.

    THE REPAIR IS PENDING UNTIL ITS OWN SEAL AND PUBLICATION STAND (the closing pass on 2c). The replacement tree is copied under
    the SAME ``<sha8>`` the pointer names, so once its fresh verify report is beside it the destination reads exactly like a healthy
    `current`; what tells the two apart is the record: ``current_dangling`` names this content from the moment the dangle is
    established until the repair's publication clears it (nothing else clears a dangle of this content -- not a `stores attest`, not
    the reversed mirror, which keep the record of a pending repair). A tick that finds the pointer, a present tree, a matching report
    AND that record (the repair's seal, or its publication step, failed last tick: a transient destination error) is ``pending``: it
    runs the publication -- the directory seal, then the guarded switch, which rewrites ``PUBLISHED-<sha8>.json`` -- and lands
    ``republished_dangling_current``; never ``unchanged``. The previous tree's records never certify the replacement: its SEALED
    record goes aside with its reports when the repair starts (:meth:`Remote.reports_aside`), so a `current` over the unsealed
    replacement is ``current_unsealed`` (red) to every reader even when the record is gone, and the publication's in-flight state
    (``generation.state = publishing``, the dangle's ``phase``) is written before the seal.

    THE SEAL GATES PUBLICATION, ON EVERY ROUTE (the final pass on 2c). ``published`` -- what `cluster_reads.plan_window` pins and what a
    shard then reads -- is DROPPED for a `current` whose own ``SEALED-<sha8>.json`` is not beside it, and written only after a seal
    succeeds: the ``current_unsealed`` refusal leaves it None (the tree standing does not restore it), an adoption's seal runs BEFORE the
    ``published`` transition so a seal that dies on the transport publishes nothing, `stores attest` drops it by the same rule, and a run
    that settles nothing advertises nothing without it either (:func:`held_publication`).
    :func:`admit_generation` carries the other end of it: a pinned generation with no SEALED record, or whose top directory is writable,
    is refused by name (``unsealed`` / ``sealed_mismatch`` / ``unsealed_directories``) instead of admitted on its VERIFIED and DEST-STAT
    alone. Before this a failed seal left a published pin over a verified-but-WRITABLE replacement while the mirror reported the
    publication incomplete."""
    t0 = time.monotonic()
    if comp.authority not in allow:
        why = (f"authority {comp.authority}: the canonical copy is the generation on the destination (`stores push` publishes it); nothing is "
               f"carried from the grading host" if comp.authority == "cluster_a" else f"authority {comp.authority}: no copy; `stores check` compares manifests")
        return MirrorResult(comp.slug, "skipped", detail={"why": why})
    if verify_inline and not dest.is_local:
        raise StoresError(f"verify_inline runs where the tree is: destination {dest.name} is remote ({dest.host})")
    gen_dir = dest.gen_dir(comp.slug)
    if dry_run:
        walk = walk_component(comp)
        esc = sorted(rel for rel, _p, _s in walk.files if rel_needs_escape(rel)) + sorted(rel for rel, _p, _s, _t in walk.links if rel_needs_escape(rel))
        tgt = sorted(rel for rel, _p, _s, t in walk.links if escape_target(t) != t)
        return MirrorResult(comp.slug, "dry_run", n_files=len(walk.files), bytes=sum(st.st_size for _r, _p, st in walk.files),
                            detail={"would_rsync_to": f"{gen_dir}/<content_sha8>/", "members": comp.members, "exclude": comp.exclude,
                                    "n_symlinks": len(walk.links), "unmatched_members": walk.unmatched_members, "n_name_escapes": len(esc),
                                    "name_escape_tops": escape_tops(esc), "n_target_escapes": len(tgt), "target_escaped_links": tgt[:LIST_CAP],
                                    "sbatch": dest.sbatch_path, "writes": "nothing"}, elapsed_s=time.monotonic() - t0)
    sdir = component_state_dir(root, comp.slug)
    remote = Remote(dest, runner)
    res = MirrorResult(comp.slug, "error")
    # the ONE state record every exit writes (the `finally`): the component, its latest manifest, the generation, what is
    # published, and last_run = {utc, outcome, ...}; each exit fills in its part
    state: Dict[str, Any] = {"authority": comp.authority, "base": comp.base, "home_rel": comp.home_rel, "adapters": comp.adapters,
                             "revision": pin.revision if pin else None, "producer": comp.producer or None}
    last_run: Dict[str, Any] = {}
    if pin:
        res.detail["revision"] = pin.revision
    pre: Optional[Manifest] = None
    current: Optional[str] = None
    dangling = False                      # `current` names this content and its tree is gone, partial, or the pending replacement
    unsealed = False                      # `current` names a generation whose own SEALED record is not beside it (the seal gates publication)
    changed_rels: List[str] = []
    plock: Optional[PublishLock] = None
    mode = str((publish_meta or {}).get("mode") or "mirror")
    # every publication is recorded: a plain mirror's record names mode `mirror` (the record is what the next publisher requires of `current`)
    meta: Dict[str, Any] = dict(publish_meta) if publish_meta is not None else {"mode": "mirror", "producer_host": socket.gethostname()}
    grading_host_driven = mode != "push_on_cluster"
    # the lock holder's name, written into the lock file for whoever waits on it: the same on both paths
    holder = lock_holder or f"{getpass.getuser()}@{socket.gethostname()}:{os.getpid()} mode={mode}"

    def generation(**fields: Any) -> None:
        state["generation"] = {"content_sha8": res.content_sha8, **fields}

    def refuse(outcome: str, why: str, **fields: Any) -> MirrorResult:
        """A refusal by name that switches nothing: `current` stays what it is, the state says why and what it read. While the pointer
        dangles (its tree gone, partial, or the pending replacement of an unfinished repair) nothing is published, whatever `current`
        says: ``published`` stays dropped and the dangle record stands. The SAME holds for a `current` whose own ``SEALED-<sha8>.json``
        is not beside it, directory standing or not: the seal gates publication, so ``published`` is dropped there too -- before this a
        ``current_unsealed`` refusal re-advertised the pointer it had just called red, and the window pinned it."""
        res.outcome = outcome
        res.published = None if (dangling or unsealed) else current
        res.detail["why"] = last_run["why"] = why
        res.detail.update(fields)
        state["published"] = None if (dangling or unsealed) else current
        generation(state=outcome, current_read=current, **fields)
        return res

    def unrecorded_why(gen: str) -> str:
        return (f"current_unrecorded: `current` -> {gen} under {gen_dir} has no publication record (PUBLISHED-{gen}.json): a publisher is mid-flight "
                f"or crashed there, or the generation predates the record; nothing copied or switched -- wait for that publisher, or adopt the "
                f"generation explicitly with `stores push {comp.slug} --adopt-unrecorded` (from the producer whose content it is: the record is "
                f"written under the publish lock without switching; a differing content then supersedes it)")

    def publish(outcome: str, report: Mapping[str, Any], **fields: Any) -> MirrorResult:
        """The one publication step: the directory seal, then ``current -> <sha8>`` through the guard, which -- under the slug's publish
        lock (held for the whole flow on a local destination, taken by the step itself on a remote one) -- re-reads the directory
        (manifest, VERIFIED, DEST-STAT, SEALED), `current` against what this run read at its start (a stale reading never switches
        `current` back), the record of that `current` (an unrecorded `current` is never superseded without an explicit adoption), the
        producer rule, and writes THIS publication's record before it switches."""
        if grading_host_driven and outcome == "published_existing_verified":
            # the tree this run would re-point at was published before: a cluster job's generation is never re-published from the grading host
            # (the pre-check that saves the seal; the step applies the rule again under the lock)
            why = _foreign_producer(remote.read_json(f"{gen_dir}/PUBLISHED-{res.content_sha8}.json"), comp)
            if why:
                return refuse("producer_mismatch", why, foreign_generation=res.content_sha8)
        # the publication's in-flight state, written BEFORE the directory seal: a tick that dies between its verify and its switch leaves
        # `publishing` (amber) behind, and a dangle repair leaves its record naming the phase -- so the next tick finishes the seal and the
        # publication of the replacement (`pending` above) rather than reading the tree, its fresh report and the previous tree's records
        # as `unchanged` (the closing pass on 2c). `last_run` keeps the in-flight stamp the verify wrote
        if dangling:
            state["current_dangling"] = {**(state.get("current_dangling") or {}), "phase": "publishing"}
        generation(state="publishing", job_id=report.get("job_id"), verified_utc=report.get("verified_utc"),
                   **({"repaired_dangling_current": True} if dangling else {}))
        update_state(root, comp.slug, dict(state), dest.name)
        sealed = seal_generation(remote, gen_dir, res.content_sha8, now=now)
        res.detail["sealed"] = {k: sealed.get(k) for k in ("n_dirs", "sealed_utc", "resealed")}
        assert pre is not None
        rec = published_record(gen_dir, res.content_sha8, comp, pre, current, changed_rels, meta, now=now)
        try:
            # the producer rule inside the step, as _foreign_producer states it: a record naming `cluster` is foreign only to a component
            # NOT declared cluster-produced (a cluster-produced component's own records name cluster)
            lock_rec = remote.publish(gen_dir, res.content_sha8, record_text(rec), expect_current=current, lock_wait_s=lock_wait_s, lock_holder=holder,
                                      lock_held=plock is not None, utc=now(), adopt_unrecorded=adopt_unrecorded,
                                      producer_guard=grading_host_driven and comp.producer != "cluster")
            if plock is None:
                res.detail["publish_lock"] = lock_rec
        except PublishLockBusy as exc:
            # the step waited its bound for another publisher's lock: nothing switched; the verified, sealed tree stays for the next run
            # (published_existing_verified against the `current` it reads then)
            res.outcome = "lock_busy"
            res.detail["why"] = last_run["why"] = str(exc)
            res.detail["current_read"] = current
            generation(state="lock_busy", current_read=current, verified_utc=report.get("verified_utc"))
            return res
        except CurrentUnrecorded as exc:
            # the `current` this run read has no record (the step saw it under the lock): the verified tree stays; nothing switched
            return refuse("current_unrecorded", str(exc), unrecorded_generation=exc.generation, verified_utc=report.get("verified_utc"))
        except ProducerMismatch as exc:
            return refuse("producer_mismatch", str(exc), foreign_generation=exc.generation, verified_utc=report.get("verified_utc"))
        except CurrentMoved as exc:
            # the verified, sealed tree stays: a later run that reads the NEW `current` publishes it through the same guard if its
            # content is still that run's (published_existing_verified); nothing here moves `current` onto an older snapshot
            res.outcome = "current_moved"
            res.published = remote.readlink(f"{gen_dir}/current")
            res.detail["why"] = last_run["why"] = str(exc)
            res.detail["current_read"] = current
            state["published"] = res.published                  # what IS published now: the other publisher's generation
            generation(state="current_moved", current_read=current, current_now=res.published, verified_utc=report.get("verified_utc"))
            return res
        res.outcome, res.published, res.job_id = outcome, res.content_sha8, report.get("job_id")
        # the attestation the generation was VERIFIED against (its listings_sha256): what a window pins and revalidates against; a publish
        # clears a recorded dangle
        state.update(published=res.content_sha8, published_attestation=report.get("listings_sha256"), verify_rate_mb_s=report.get("rate_mb_s"), current_dangling=None,
                     attest_integrity_failed=None)
        res.detail["published_record"] = _published_digest(rec)
        fields = dict(fields, published_record=res.detail["published_record"])
        generation(state="published", job_id=res.job_id, verified_utc=report.get("verified_utc"), sealed=res.detail["sealed"],
                   listings_sha256=report.get("listings_sha256"), **fields)
        return res

    try:
        # the destination's directory first: the slug's publish lock lives in it, and it is taken BEFORE the snapshot so that what this
        # run manifests is the producer's state after the previous publication, never a snapshot an overlapping publisher outran
        remote.mkdir(gen_dir)
        if dest.is_local:
            plock = PublishLock(gen_dir, wait_s=lock_wait_s, holder=holder, log=lambda line: log(f"[{comp.slug}] {line}")).acquire()
            res.detail["publish_lock"] = plock.record()
        try:
            pre = build_manifest(comp, sdir, hasher=hasher, full=full)
        except ManifestRefusal as exc:
            res.outcome = "refused"
            res.detail["reason"] = last_run["reason"] = str(exc)
            return res
        sha8 = pre.content_sha8
        res.content_sha8, res.n_files, res.bytes, res.hashed, res.from_cache = sha8, pre.n_files, pre.bytes, pre.hashed, pre.from_cache
        state["latest_manifest"] = pre.summary(write_manifest_files(pre, sdir))       # attestations are immutable: this adds, never restates
        escapes = pre.name_escapes()
        if escapes:
            res.detail["name_escapes"] = {"n": len(escapes), "tops": escape_tops(escapes)[:LIST_CAP]}
        target_escapes = pre.target_escapes()
        if target_escapes:
            res.detail["target_escapes"] = {"n": len(target_escapes), "links": sorted(target_escapes, key=_by_bytes)[:LIST_CAP]}
        carried, tar_excludes = pre.tar_carried()
        gen_path = f"{gen_dir}/{sha8}"
        recorded = ((load_state(root).get("components") or {}).get(comp.slug) or {})
        # a verify job an earlier run recorded and did not see to its end (its controller cut short) is SETTLED before anything moves
        settled = settle_awaiting_verify(recorded.get("generation") or {}, dest, remote, log=lambda line: log(f"[{comp.slug}] {line}"))
        if settled["refusal"]:
            # the settle refused: nothing is copied, renamed or switched, and `current` is advertised as published ONLY with the evidence
            # of its publication beside it (held_publication) -- a recorded dangling repair this run has not resolved keeps `published`
            # dropped, so the window's plan pins nothing for the component rather than a tree nothing attests
            res.outcome = settled["refusal"]
            current_now = remote.readlink(f"{gen_dir}/current")
            held = held_publication(remote, gen_dir, current_now, recorded)
            res.published = held["published"]
            res.detail.update(awaiting_verify=settled)
            if current_now and held["problems"]:
                res.detail["not_published"] = {"problems": held["problems"], "why": held["why"]}
                last_run["not_published"] = held["why"]
            last_run["why"] = settled["why"]
            state["published"] = held["published"]
            if held["current_dangling"]:
                state["current_dangling"] = held["current_dangling"]       # the dangle note stands until a publish clears it
            state["generation"] = dict(recorded.get("generation") or {}, settle=settled)    # the record stands: the job id is still the fact
            return res
        if settled.get("job_id"):
            res.detail["awaiting_verify_settled"] = settled
        # what the LAST tick recorded about the pointer: a repair it began and did not finish (its copy cut short) is resumed, and
        # the attestation a publish or a `stores attest` recorded for the published listings (``published_attestation``)
        prior = recorded.get("current_dangling") or {}
        recorded_attestation = recorded.get("published_attestation") or None
        current = remote.readlink(f"{gen_dir}/current")
        verified = remote.read_json(f"{gen_dir}/VERIFIED-{sha8}.json")
        tree_present = remote.isdir(gen_path)          # both shortcuts probe the TREE: a report outlives a renamed tree
        # `current` can name a tree that is not there (the live bigcodebench_full on 2026-10-04: code without the guard switched `current`
        # onto a tree a verify-report misread had renamed `.failed-`); such a pointer is never "unchanged" -- the content is re-copied,
        # re-verified and re-published through the guard, and nothing hard-links against the missing tree
        current_exists = bool(current) and (tree_present if current == sha8 else remote.isdir(f"{gen_dir}/{current}"))
        # `current` names this content but its tree is gone (renamed away) -- or the tree is the last tick's PARTIAL repair copy
        # (that tick recorded the dangle for this sha8 and no report matches the tree): the copy resumes, never
        # `current_unverified` over the mirror's own unfinished copy (a pointer published behind the mirror's back still is)
        resuming = current == sha8 and tree_present and prior.get("content_sha8") == sha8 and not _verified_matches(verified, pre)
        # the tree is the REPLACEMENT the last tick's repair copied and verified (that tick recorded the dangle for this sha8 and the
        # fresh report matches) whose seal or publication step did not complete: PENDING -- the publication runs below, never `unchanged`
        # over the previous tree's records (the closing pass on 2c)
        pending = current == sha8 and tree_present and prior.get("content_sha8") == sha8 and _verified_matches(verified, pre)
        dangling = current == sha8 and (not tree_present or resuming or pending)
        if current and current != sha8 and not current_exists:
            # `current` names ANOTHER content whose tree is gone: nothing hard-links against it (the same-content dangle is handled below)
            res.detail["current_dangling"] = current
            log(f"[{comp.slug}] current -> {current} names no tree under {gen_dir}; nothing links against it")
        if current == sha8 and tree_present and not resuming and not pending:
            # the pointer names a directory and no repair of this content is in flight: no dangle, whatever the record said (a recorded
            # dangle is cleared ONLY by this verdict -- for ANOTHER content -- or by a publish; a tick that errors before either leaves
            # the record saying what it said)
            state["published"], state["current_dangling"] = current, None
            state["published_attestation"] = verified.get("listings_sha256")
            if not _verified_matches(verified, pre, listings=False):
                # `current` names this content but no matching report stands beside it: never "unchanged"
                res.outcome, res.published = "current_unverified", current
                res.detail.update(verified_present=bool(verified),
                                  why="`current` names this generation but no VERIFIED report matching its manifest is beside it")
                generation(state="current_unverified", verified_utc=verified.get("verified_utc"))
                return res
            # the same FILE content, published and verified. Its directory LISTINGS (MANIFEST_RULES["directories"]) are judged by the report
            # that verified the canonical's CURRENT listings against this tree -- the live report, or the keyed VERIFIED-<sha8>.<lsha8>.json a
            # `stores attest` wrote (:func:`listings_verdict`); a generation is named by its file content, so nothing new is ever published
            # under this name for a listing alone
            # The verdict is READ here and APPLIED last: the publication record, the adoption and the weekly re-hash run for every listing
            # state (an amber listing outcome returned before them let `--full` report success without re-hashing and a push return an
            # accepted outcome without its record)
            verdict, att_report = listings_verdict(remote, gen_dir, sha8, verified, pre)
            attested_by_live = verdict == "attested" and _verified_matches(verified, pre)
            sealed_rec = remote.read_json(f"{gen_dir}/SEALED-{sha8}.json")
            pub_rec = remote.read_json(f"{gen_dir}/PUBLISHED-{sha8}.json")
            unsealed = not sealed_rec
            if unsealed:
                # THE SEAL GATES PUBLICATION (the final pass on 2c): from the moment this run reads a `current` without its own SEALED
                # record, nothing is advertised published -- not by the refusal below, not by an adoption whose seal then fails. Before
                # this the state kept ``published = current`` through both, so `cluster_reads.plan_window` pinned, and a shard read, a
                # verified-but-WRITABLE replacement while the mirror's own outcome said the publication was incomplete
                state["published"] = None
            if not pub_rec and not adopt_unrecorded:
                # `current` names this content but no record stands beside it: never "unchanged" -- a publisher may be mid-flight, or the
                # generation predates the record; the adoption is explicit
                return refuse("current_unrecorded", unrecorded_why(sha8), unrecorded_generation=sha8, verified_utc=verified.get("verified_utc"))
            if not sealed_rec and publish_meta is None and not adopt_unrecorded:
                # `current` names this content, recorded, but no SEALED record stands beside it: never "unchanged" either -- the evidence the
                # mirror calls a `current` green by (CURRENT_EVIDENCE; the reversed mirror and the attest already name it) includes the seal.
                # A repair of this content whose seal failed moved the previous tree's seal aside (reports_aside) and left the replacement
                # writable: with the state's record gone this is what remains of it, and a plain tick must not read it green (the closing
                # pass on 2c). The seal is written by a push, or by `--adopt-unrecorded` (the record stands; the seal is what it adopts)
                return refuse("current_unsealed", f"current_unsealed: `current` -> {sha8} under {gen_dir} stands recorded but without its SEALED-{sha8}.json: "
                                                  f"its directories are not certified sealed (a repair whose seal did not complete, or a generation that predates "
                                                  f"the seal); never called unchanged -- `stores push {comp.slug}` or `stores mirror --component {comp.slug} "
                                                  f"--adopt-unrecorded` seals it", verified_utc=verified.get("verified_utc"), authority_problems=["current_unsealed"])
            adopted: Dict[str, Any] = {}
            if unsealed and (publish_meta is not None or not pub_rec or adopt_unrecorded):
                # a push seals an unsealed generation it finds current (an older flow left it writable); an adoption seals before it records,
                # and `--adopt-unrecorded` seals a recorded `current` whose seal alone is missing (as the reversed mirror's adoption does).
                # The SEAL RUNS FIRST and ``published`` follows it (the final pass on 2c): a seal that dies on the transport leaves an error
                # with nothing published, where the state had already called the generation published before the attempt was made
                adopted["sealed"] = {k: v for k, v in seal_generation(remote, gen_dir, sha8, now=now).items() if k in ("n_dirs", "sealed_utc", "host")}
                unsealed = False
                state["published"] = current
            res.outcome, res.published = "unchanged", current
            generation(state="published", verified_utc=verified.get("verified_utc"), listings_sha256=state["published_attestation"])
            if not pub_rec:
                # the explicit adoption: the record of THIS generation, written under the publish lock, `current` untouched
                pub_rec = published_record(gen_dir, sha8, comp, pre, None, [rel for _s, _z, rel in pre.entries], {**meta, "mode": "adopt"}, now=now)
                adopt = remote.record_published(gen_dir, sha8, record_text(pub_rec), lock_wait_s=lock_wait_s, lock_holder=holder, lock_held=plock is not None, utc=now())
                if not adopt["written"]:
                    pub_rec = remote.read_json(f"{gen_dir}/PUBLISHED-{sha8}.json")      # another publisher recorded it meanwhile
                adopted["published_record"] = _published_digest(pub_rec)
                if plock is None:
                    res.detail["publish_lock"] = {k: adopt[k] for k in ("path", "holder", "scope")}
            if adopted:
                res.detail["adopted"] = adopted
                state["generation"]["adopted"] = adopted
            res.detail["published_record"] = _published_digest(pub_rec)
            full_passed = False
            if full:
                # the weekly full verify: the unchanged, published generation re-hashed on the destination by the pinned verifier -- against the
                # ATTESTATION that verified the canonical's current listings (the keyed MANIFEST-<sha8>.<lsha8>.json when a re-attestation did,
                # the plain .json when the live report did), its listings expected only when one is attested; the files alone otherwise
                att_path = None if (verdict != "attested" or attested_by_live) else attestation_json_path(f"{gen_dir}/MANIFEST-{sha8}.sha256", pre.listings_sha256)
                fr = full_rehash(root, dest, comp, remote, gen_dir, sha8, pre, pin=pin, log=log, now=now, workers=workers, inline=verify_inline,
                                 attestation=att_path, listings_sha256=(pre.listings_sha256 if verdict == "attested" else None))
                if not fr["ok"]:
                    return refuse("full_verify_failed", f"the published generation {sha8} under {gen_dir} does not re-hash to its manifest ({fr['counts']}; "
                                                        f"report {'present' if fr['report_present'] else 'ABSENT'} at {fr['report_path']}): nothing renamed, "
                                                        f"`current` untouched -- the operator decides", full=fr)
                res.detail["full"] = fr
                state["generation"]["full"] = {k: fr.get(k) for k in ("ok", "report_path", "job_id")}
                full_passed = True
                # remembered: a re-attestation failure stamped before this pass is superseded on every later tick (attest_integrity)
                state["last_full_pass"] = {"content_sha8": sha8, "verified_utc": (fr.get("report") or {}).get("verified_utc"), "report_path": fr["report_path"]}
            carried = {k: state["generation"][k] for k in ("adopted", "full") if k in state["generation"]}
            # a re-attestation that hashed every file and found some WRONG stands, red, until a later full hash passes (HIGH 1)
            fail = attest_integrity(remote, gen_dir, sha8, pre, recorded, verdict, att_report, full_passed, now=now)
            state["attest_integrity_failed"] = fail
            if fail:
                # the attestation recorded for the generation stands through the red (as the drift and listings_changed paths keep it): a
                # pin that names none checks NO listing, so dropping it would widen what a reader admits, not narrow it (round 2 on 2c)
                state["published_attestation"] = verified.get("listings_sha256") or recorded_attestation
                res.outcome, res.published = "attest_integrity_failed", current
                res.detail.update(why=_integrity_why(fail, gen_path), attest_failure=fail, listings_sha256=pre.listings_sha256)
                generation(state="attest_integrity_failed", verified_utc=verified.get("verified_utc"), attest_failure=fail, **carried)
                return res
            # the same FILE content, published and verified. Its directory LISTINGS (MANIFEST_RULES["directories"]) are judged by the report
            # that verified the canonical's CURRENT listings against this tree -- the live report, or the keyed VERIFIED-<sha8>.<lsha8>.json a
            # `stores attest` wrote (:func:`listings_verdict`); a generation is named by its file content, so nothing new is ever published
            # under this name for a listing alone
            listing_fields = dict(listings_sha256=pre.listings_sha256, n_directories=pre.n_directories, n_empty_directories=pre.n_empty_directories)
            if verdict == "attested":
                state["published_attestation"] = (att_report or {}).get("listings_sha256")
                generation(state="published", verified_utc=verified.get("verified_utc"), listings_sha256=state["published_attestation"], **carried)
                return res
            if verdict == "drift":
                # a re-attestation of exactly these listings refused them: the generation's directory set is not the canonical's (its keyed
                # report names the direction) -- a listing-COVERAGE limitation, amber: the pin is unchanged, the listing readers keep the grading host
                # route, the generation is not shown listing-attested; read on every tick without another job until a file change publishes a
                # fresh generation
                state["published_attestation"] = verified.get("listings_sha256") or recorded_attestation
                res.outcome, res.published = "listing_only_drift", current
                res.detail.update(published_listings_sha256=state["published_attestation"], attest_report=keyed_report_path(gen_dir, sha8, pre.listings_sha256),
                                  listing_mismatch=list((att_report or {}).get("listing_mismatch") or [])[:LIST_CAP],
                                  dirs_generation_only=list((att_report or {}).get("dirs_generation_only") or [])[:LIST_CAP],
                                  dirs_canonical_only=list((att_report or {}).get("dirs_canonical_only") or [])[:LIST_CAP],
                                  why=_drift_why(att_report or {}, gen_path), **listing_fields)
                generation(state="listing_only_drift", verified_utc=verified.get("verified_utc"), listings_sha256=pre.listings_sha256, **carried)
                return res
            if verified.get("listings_sha256") is None and not recorded_attestation:
                # the generation was verified BEFORE the directory-listings rule (its report attests no listing) and nothing has attested
                # one since: its file content stands published and verified, nothing is wrong with the mirror, and the window's revalidate
                # checks no listing for it (the pin names no attestation) -- `stores attest` attests the current listings. Before this
                # outcome existed the compare read `None == <digest>` and called every such generation `listings_changed` (red, 2026-10-05..07)
                res.outcome, res.published = "listings_unattested", current
                res.detail.update(published_listings_sha256=None, attest=f"registry stores attest --dest {dest.name} --component {comp.slug}",
                                  why="the published generation was verified before its directory listings were attested; its file content stands "
                                      "published and verified -- `stores attest` attests the current listings", **listing_fields)
                generation(state="listings_unattested", verified_utc=verified.get("verified_utc"), listings_sha256=pre.listings_sha256, **carried)
                return res
            # the canonical's directory LISTINGS moved from the attested ones since (an empty directory landed or left): nothing new can be
            # published under this name -- recorded as its own outcome, and the window's revalidate refuses the component (its listings are
            # not the pinned attestation's), so the readers that state listings take the grading host route until `stores attest` attests the
            # current listings or a file change publishes the next generation, which attests them
            state["published_attestation"] = verified.get("listings_sha256") or recorded_attestation
            res.outcome, res.published = "listings_changed", current
            res.detail.update(published_listings_sha256=state["published_attestation"], attest=f"registry stores attest --dest {dest.name} --component {comp.slug}",
                              why="the canonical copy's directory listings moved while its file content stands; the published generation attests "
                                  "the old listings -- `stores attest` attests the current ones (or refuses a drift by name)", **listing_fields)
            generation(state="listings_changed", verified_utc=verified.get("verified_utc"), listings_sha256=pre.listings_sha256, **carried)
            return res
        if current:
            # the record of the `current` this run read: absent -> a publisher is mid-flight or crashed there (or a legacy generation);
            # nothing is copied until it is adopted explicitly. The pre-check that saves the copy; the step applies it again under the lock
            cur_rec = remote.read_json(f"{gen_dir}/PUBLISHED-{current}.json")
            if not cur_rec and not adopt_unrecorded:
                return refuse("current_unrecorded", unrecorded_why(current), unrecorded_generation=current)
        if current and grading_host_driven:
            # `current` is a cluster job's publication: the grading host copy is a lagging mirror of what that job writes; nothing is copied
            why = _foreign_producer(cur_rec, comp)
            if why:
                return refuse("producer_mismatch", why, foreign_generation=current)
        # the files that differ from the PUBLISHED generation: FORCED in the copy (rsync's size/mtime quick check would hard-link a
        # same-size, mtime-preserving edit to the old bytes) and the PUBLISHED record's `change_utc` witnesses
        prev = _published_entries(sdir, remote, gen_dir, current) if current else None
        changed_rels = [rel for sha, size, rel in pre.entries if prev is None or prev.get(rel) != (sha, size)]
        if pending:
            # the replacement tree of this content's repair, verified by the fresh report beside it, its seal or its publication cut short
            # last tick: the dangle record follows the destination here as on the first tick -- `published` stays dropped (nothing is
            # published until the replacement's own seal and record stand) and the dangle keeps its first finding -- then the publication
            # runs: the directory seal, the guarded switch (which rewrites PUBLISHED-<sha8>.json), republished_dangling_current. Never
            # `unchanged`: before this the fresh report, the present tree and the PREVIOUS tree's SEALED / PUBLISHED records read as a
            # healthy `current`, and the repair's writable directories and stale record stood green
            found = ", ".join(prior.get("renamed_aside") or []) or ("no renamed tree" if prior.get("renamed_aside") is not None else "not yet listed")
            tree = f"{gen_path} is the verified replacement of the repair begun {prior.get('found_utc')} (its seal or its publication did not complete)"
            state["published"] = None
            state["current_dangling"] = {**prior, "resumed": False, "phase": "publishing", "why": f"current -> {sha8} but {tree}; beside it: {found}"}
            generation(state="current_dangling", found_utc=prior.get("found_utc"))
            res.detail.update(current_dangling=True, resumed_repair=False, pending_publication=True, renamed_aside=prior.get("renamed_aside"),
                              why_republished=f"`current -> {sha8}` but {tree}; beside it: {found}; sealed and republished through the guard, "
                                              f"never called unchanged over the previous tree's records")
            log(f"[{comp.slug}] current -> {sha8} is dangling: {tree}; beside it: {found}; nothing is published until the replacement's own "
                f"seal and record stand; sealing and republishing")
            res.detail["verified"] = {k: verified.get(k) for k in ("hashed", "elapsed_s", "rate_mb_s", "host", "verified_utc")}
            return publish("republished_dangling_current", verified, repaired_dangling_current=True, renamed_aside=prior.get("renamed_aside"),
                           pending_publication=True)
        if tree_present and _verified_matches(verified, pre):      # a run that died between the verify and the publish
            return publish("published_existing_verified", verified)
        if dangling:
            # the record said `published`; the destination says the pointer dangles. The record follows the destination
            # THE MOMENT the dangle is established -- `published` dropped, a minimal `current_dangling` and the generation marked --
            # before any further step that can raise (the listing below is one more ssh), so a tick that dies here still leaves an
            # honest record (an error merged over `published: <sha8>` kept the plan pinning a missing tree).
            # Nothing is published until the fresh copy is verified and `current` republished.
            tree = (f"{gen_path} is the partial copy of the repair begun {prior.get('found_utc')} (no report matches it)" if resuming
                    else f"{gen_path} is not a directory")
            found_utc = prior.get("found_utc") if resuming else now()
            state["published"] = None
            state["current_dangling"] = {"content_sha8": sha8, "found_utc": found_utc, "renamed_aside": None, "resumed": bool(resuming),
                                         "why": f"current -> {sha8} but {tree}; beside it: not yet listed"}
            generation(state="current_dangling", found_utc=found_utc)
            res.detail.update(current_dangling=True, resumed_repair=bool(resuming))
            # then the trees a rename left beside the pointer are named (one more stat-class step, on this path only)
            aside = remote.renamed_aside(gen_dir, sha8)
            found = ", ".join(aside) if aside else "no renamed tree"
            state["current_dangling"].update(renamed_aside=aside, why=f"current -> {sha8} but {tree}; beside it: {found}")
            res.detail.update(renamed_aside=aside,
                              why_copied=f"`current -> {sha8}` but {tree} (renamed away; beside it: {found}); "
                                         f"{'the copy resumed' if resuming else 'copied afresh'} and re-verified")
            log(f"[{comp.slug}] current -> {sha8} is dangling: {tree}; beside it: {found}; "
                f"nothing is published until the fresh copy verifies; {'resuming the copy' if resuming else 'copying afresh'}")
        aside_link: Optional[str] = None       # the tree of this content renamed aside THIS tick: the retry's second --link-dest
        if tree_present:
            # this content was published before under OTHER listings (an A -> B -> A reversion whose directories moved) and its tree still
            # stands, not current: a generation tree is never rewritten in place (a window pinned to it would read a tree neither
            # attestation describes) -- it is renamed aside and the copy builds a fresh tree the new attestation verifies
            superseded = f"{gen_dir}/.superseded-{sha8}-{_stamp(now())}"
            remote.rename(gen_path, superseded)
            res.detail["superseded"] = superseded
            aside_link = superseded
            log(f"[{comp.slug}] {gen_path} stands from an earlier attestation of this content; renamed to {superseded} (the fresh copy links against it: "
                f"a same-content file transfers nothing, the verify decides)")
        # EVERY report and snapshot that spoke for a tree at this name is moved aside before the fresh copy, in one step: the live
        # VERIFIED-<sha8>.json a renamed-away tree left behind (so the fresh verify's report is the only one) with its DEST-STAT, AND the keyed
        # VERIFIED-<sha8>.<lsha8>.json / DEST-STAT-<sha8>.<lsha8>.json a `stores attest` wrote over the REPLACED tree. Round 3 on 2c: the keyed
        # report of a superseded A tree stayed beside the generation, and when the canonical's listings returned to the ones it had verified
        # it attested the FRESH A tree -- built from the canonical under other listings -- so the mirror read `unchanged` and recorded them as
        # the attestation while admit_generation refused that pin `listing_changed`. A report speaks for the tree it was run over; the fresh
        # tree earns its own. The keyed MANIFEST-<sha8>.<lsha8>.json attestations stay: they name a (content, listings) pair, not a tree
        aside_reports = remote.reports_aside(gen_dir, sha8, _stamp(now()))
        if aside_reports:
            res.detail["stale_reports_renamed_to"] = aside_reports
            live_aside = next((p for p in aside_reports if os.path.basename(p).startswith(f".stale-VERIFIED-{sha8}-")), None)
            if live_aside:
                res.detail["stale_report_renamed_to"] = live_aside
        # (ii) the copy: --link-dest against the PUBLISHED generation -- only when that generation IS a directory and not the dangling
        # pointer's (rsync warns about a missing link-dest and copies in full anyway) -- then the files whose sha differs from it FORCED
        # (rsync's size/mtime quick check would hard-link a same-size, mtime-preserving edit to the old bytes); the entries the destination
        # cannot name (the escape) and the links whose targets it spells differently by tar under their escaped spelling, every generation
        link_dest = f"{gen_dir}/{current}" if (current_exists and not dangling) else None
        forced = [rel for rel in changed_rels if rel not in escapes] if link_dest else []
        copy_path = f"{gen_dir}/.incoming-{sha8}" if incoming else gen_path
        n_escaped_files = sum(1 for _s, _z, rel in pre.entries if rel in escapes)
        log(f"[{comp.slug}] rsync {pre.n_files - n_escaped_files} files -> {copy_path} (link-dest {current or 'none'}; aside {aside_link or 'none'}; "
            f"{len(forced)} forced; {len(carried)} entr(y/ies) escaped by tar)")
        r = remote.run(rsync_argv(dest, comp, copy_path, link_dest, escape_tops_=tar_excludes, extra_link_dests=([aside_link] if aside_link else [])),
                       dest.rsync_timeout_s)
        stats = _rsync_stats(r.stdout or "")
        if r.returncode == 0 and forced:
            r = remote.run(rsync_forced_argv(dest, comp, copy_path), dest.rsync_timeout_s, input="\n".join(forced) + "\n")
            stats["forced"] = {"n": len(forced), **_rsync_stats(r.stdout or "")}
        res.detail.update(rsync=stats, link_dest=link_dest, link_dest_aside=aside_link)
        if r.returncode != 0:
            res.outcome = "rsync_failed"
            res.detail.update(rc=r.returncode, stderr=(r.stderr or "").strip()[-600:])
            last_run["stderr"] = res.detail["stderr"]
            generation(state="copying", rsync_rc=r.returncode)
            return res
        if carried:
            r = remote.run(tar_copy_argv(dest, comp, copy_path), dest.rsync_timeout_s, input="\0".join(carried) + "\0")
            res.detail["tar"] = {"n": len(carried), "rc": r.returncode}
            if r.returncode != 0:
                res.outcome = "tar_failed"
                res.detail.update(rc=r.returncode, stderr=(r.stderr or "").strip()[-600:])
                last_run["stderr"] = res.detail["stderr"]
                generation(state="copying", tar_rc=r.returncode)
                return res
        # (iii) canonical mutation during the copy invalidates the generation
        moved, missing, extra = stat_changes(walk_component(comp), pre.snapshot)
        changed = sorted(moved + missing + extra, key=_by_bytes)
        if changed:
            invalid = f"{gen_dir}/.invalid-{sha8}-{_stamp(now())}"
            remote.rename(copy_path, invalid)
            res.outcome = "invalidated"
            res.detail.update(moved=len(changed), moved_first=changed[:20], renamed_to=invalid,
                              why="the canonical copy changed during the copy; retried on the next run")
            last_run["moved_first"] = changed[:20]
            generation(state="invalidated", renamed_to=invalid, moved=len(changed))
            return res
        if incoming:
            remote.rename(copy_path, gen_path)
        # (iv) the manifest beside the tree, then the verify: a Slurm job watched in the foreground, or in this process (the producer's job)
        remote.write(f"{gen_dir}/MANIFEST-{sha8}.sha256", pre.text())
        remote.write(f"{gen_dir}/MANIFEST-{sha8}.json", json.dumps(pre.meta({"rsync": stats, "link_dest": link_dest, "tar": res.detail.get("tar"), "link_dest_aside": aside_link}),
                                                                   indent=1, sort_keys=True) + "\n")
        if pre.listings_sha256:
            # the IMMUTABLE attestation of this (content, listings) pair beside the tree (MANIFEST_RULES["directories"]): the node's admission and
            # index read the one the plan pins, never the plain .json a later push of the same content may have replaced
            keyed = attestation_json_path(f"{gen_dir}/MANIFEST-{sha8}.sha256", pre.listings_sha256)
            if remote.read(keyed) is None:
                remote.write(keyed, json.dumps(pre.meta(), indent=1, sort_keys=True) + "\n")
        if verify_inline:
            generation(state="verifying_inline")
            update_state(root, comp.slug, {**state, "last_run": {"utc": now(), "outcome": "verify_inline"}}, dest.name)
            report, dest_stat = verify_generation(gen_path, f"{gen_dir}/MANIFEST-{sha8}.sha256", workers=workers, job_id=os.environ.get("SLURM_JOB_ID"))
            write_verified(f"{gen_dir}/VERIFIED-{sha8}.json", report, dest_stat)
            res.job_id = report.get("job_id")
            verified = remote.read_json(f"{gen_dir}/VERIFIED-{sha8}.json")       # written by this process: visible at once
            waited = {"attempts": 1, "waited_s": 0.0, "last": "present" if verified else "absent"}
        else:
            rate = measured_rate(root, comp.slug, dest)
            script_args = ["verify", gen_path, f"{gen_dir}/MANIFEST-{sha8}.sha256", f"{gen_dir}/VERIFIED-{sha8}.json", str(dest.verify_cpus)]
            res.job_id, tlim = remote.submit(f"store-verify-{comp.slug}", script_args, pre.bytes, pre.n_files, rate, script=pin.sbatch_path if pin else None)
            log(f"[{comp.slug}] verify job {res.job_id} submitted (--time {slurm_time(tlim)} at {rate:.0f} MB/s x2); watching")
            generation(state="awaiting_verify", job_id=res.job_id, time_limit_s=tlim)
            update_state(root, comp.slug, {**state, "last_run": {"utc": now(), "outcome": "verify_submitted"}}, dest.name)
            watched = remote.watch(res.job_id, bound_s=bound_s)
            res.detail["watch"] = watched
            if watched["state"] == "MISSING":
                # the watcher LOST the job (no accounting row for ten minutes; it never consults the queue): NOT a terminal state. The
                # controller reads the queue and the accounting together once, by the one terminal rule (before this it was read as
                # FAILED, the record became verify_job_failed, the next run settled nothing and renamed the tree a live verifier read):
                # a known non-terminal state or no proof keeps the record (awaiting_verify + job id) for the next run's settle -- nothing
                # is cancelled, renamed or switched; a PROVED terminal state stands in for the watcher's line
                lost = settle_awaiting_verify({"state": "awaiting_verify", "job_id": res.job_id}, dest, remote,
                                              log=lambda line: log(f"[{comp.slug}] {line}"), recorded_by="this run, whose watcher lost it (MISSING)")
                res.detail["lost_watch"] = lost
                last_run["terminal"] = watched.get("terminal")
                if lost["refusal"]:
                    res.outcome = lost["refusal"]                        # verify_in_flight | verify_unconfirmed
                    last_run["why"] = lost["why"]
                    generation(state="awaiting_verify", job_id=res.job_id, time_limit_s=tlim, watch=watched, settle=lost, bound_s=bound_s)
                    return res
                watched = {**watched, "state": "DONE" if str(lost["state"]).startswith("COMPLETED") else "FAILED", "settled_by": "accounting",
                           "accounting_state": lost["state"]}
                res.detail["watch"] = watched
            if watched["state"] != "DONE":
                if watched["state"] == "FAILED":
                    res.outcome = "verify_job_failed"
                    last_run["terminal"] = watched.get("terminal")
                    generation(state=res.outcome, job_id=res.job_id, watch=watched)
                    return res
                # WATCH-TIMEOUT (or an unknown watcher exit): the job may still be running. The controller owns the bound: cancel the job
                # it recorded, confirm it terminal (sacct row + out of the queue), and only then return; the tree is left where it is
                cancelled = cancel_and_confirm(str(res.job_id), dest, remote, sleep=sleep, log=lambda line: log(f"[{comp.slug}] {line}"))
                res.detail["cancel"] = cancelled
                last_run["terminal"] = watched.get("terminal")
                if cancelled["confirmed"]:
                    res.outcome = "verify_cancelled"
                    generation(state=res.outcome, job_id=res.job_id, watch=watched, terminal=cancelled.get("state"), bound_s=bound_s)
                else:
                    res.outcome = "verify_unconfirmed"
                    generation(state="awaiting_verify", job_id=res.job_id, time_limit_s=tlim, watch=watched, cancel=cancelled, bound_s=bound_s)
                return res
            # the report THIS job wrote, with a bounded wait: the compute node's write is not visible from the login node at once
            verified, waited = _await_report(remote, f"{gen_dir}/VERIFIED-{sha8}.json", res.job_id, dest.report_wait_s, dest.report_poll_s, sleep, log)
        res.detail["report_wait"] = waited
        if not verified:
            # no report of this job inside the bound: the tree is LEFT IN PLACE (never renamed -- the verify ran and the
            # report exists on the cluster); the next tick's existing-verified path publishes it once the report is visible
            res.outcome = "verify_report_missing"
            res.detail.update(verified_present=False, tree_left=gen_path,
                              why=f"verify job {res.job_id} finished (watch DONE) but VERIFIED-{sha8}.json written by it was not readable "
                                  f"from {dest.name} within {dest.report_wait_s:.0f} s ({waited['attempts']} reads, {dest.report_poll_s:.0f} s apart; "
                                  f"last: {waited['last']}); the tree is left in place and the next tick publishes it when the report is visible")
            last_run["last_read"] = waited["last"]
            generation(state="verify_report_missing", job_id=res.job_id, waited_s=waited["waited_s"], attempts=waited["attempts"], last_read=waited["last"])
            return res
        if _verified_matches(verified, pre):
            res.detail["verified"] = {k: verified.get(k) for k in ("hashed", "elapsed_s", "rate_mb_s", "host", "verified_utc")}
            return publish("republished_dangling_current" if dangling else "published", verified, link_dest=link_dest, rsync=stats,
                           **({"repaired_dangling_current": True, "renamed_aside": res.detail.get("renamed_aside")} if dangling else {}))
        # a PRESENT report of this job that does not match: a real content mismatch
        failed = f"{gen_dir}/.failed-{sha8}-{_stamp(now())}"
        remote.rename(gen_path, failed)
        res.outcome = "verify_failed"
        counts = {k: len(verified.get(k) or []) for k in ("missing", "corrupt", "extra")}
        res.detail.update(renamed_to=failed, **counts, first={k: (verified.get(k) or [])[:10] for k in counts}, verified_present=True,
                          why="the verify report does not match the manifest; nothing published")
        last_run["first"] = res.detail["first"]
        generation(state="verify_failed", job_id=res.job_id, renamed_to=failed, counts=counts)
        return res
    except PublishLockBusy as exc:
        res.outcome = "lock_busy"
        res.detail["why"] = last_run["why"] = str(exc)
        return res
    except (StoresError, subprocess.SubprocessError, OSError) as exc:
        res.outcome = "error"
        res.detail["error"] = last_run["error"] = f"{type(exc).__name__}: {exc}"[:800]
        return res
    finally:
        if plock is not None:
            plock.release()
        res.elapsed_s = time.monotonic() - t0
        update_state(root, comp.slug, {**state, "last_run": {"utc": now(), "outcome": res.outcome, **last_run}}, dest.name)


# --------------------------------------------------------------------------------------------
# the reversed mirror: `stores mirror` dispatches on each component's authority
# --------------------------------------------------------------------------------------------


def _manifest_diff(mine: Manifest, theirs: Sequence[Tuple[str, int, str]], theirs_links: Mapping[str, str]) -> Dict[str, Any]:
    """:func:`_entries_diff` plus the attested symlinks compared -- the shape `verify-authority` records (``only_grading_host`` is the grading host
    side, ``only_original`` the generation's)."""
    d = _entries_diff(mine.entries, theirs)
    link_diff = sorted(set(dict(mine.links).items()) ^ set(theirs_links.items()))
    d["symlinks"] = [list(t) for t in link_diff[:LIST_CAP]]
    d["counts"]["symlinks"] = len(link_diff)
    return d


def _generation_manifest(remote: Remote, gen_dir: str, sha8: str) -> Optional[Tuple[str, List[Tuple[str, int, str]], Dict[str, Any]]]:
    """(the text, the parsed entries, the ``.json`` meta) of a generation's manifest on the destination; None when the ``.sha256`` is absent."""
    path = f"{gen_dir}/MANIFEST-{sha8}.sha256"
    text = remote.read(path)
    if text is None:
        return None
    return text, parse_manifest_lines(text, path), remote.read_json(f"{gen_dir}/MANIFEST-{sha8}.json")


def _manifest_links(meta: Mapping[str, Any]) -> Dict[str, str]:
    links = meta.get("symlinks") if meta else None
    return {str(k): str(v) for k, v in links.items()} if isinstance(links, dict) else {}


def _save_generation_manifest(sdir: str, sha8: str, text: str, meta: Mapping[str, Any]) -> None:
    """The published generation's manifest copied beside the grading host state when no copy is there: the window's home tree reads the pin's
    manifest from ``<state>/store_manifests/<slug>/MANIFEST-<sha8>.*`` (cluster_reads.manifest_entries), and a generation the grading host
    copy is NOT equal to (unpushed changes, a frozen difference) has none here otherwise."""
    base = os.path.join(sdir, f"MANIFEST-{sha8}")
    if not os.path.isfile(base + ".sha256"):
        _write_text(base + ".sha256", text)
    if meta and not os.path.isfile(base + ".json"):
        _write_json(base + ".json", dict(meta))


def full_rehash(root: str, dest: Destination, comp: Component, remote: Remote, gen_dir: str, sha8: str, mine: Manifest,
                pin: Optional[PinnedTree] = None, log: Callable[[str], None] = lambda s: None, now: Callable[[], str] = R.utc_now,
                workers: int = 16, inline: bool = False, attestation: Optional[str] = None, listings_sha256: Optional[str] = None) -> Dict[str, Any]:
    """`--full`: the PUBLISHED generation re-hashed on the destination -- every manifest entry's bytes, every attested link, a walk for
    unlisted entries (:func:`verify_generation`) -- by the pinned verifier (``store_verify.sbatch verify``, a CPU job at ``high`` watched
    in the foreground; ``inline`` -- in this process, inside a producer's own job) into ``<stores>/full/<slug>/VERIFIED-<sha8>-<stamp>.json``: the live
    ``VERIFIED-<sha8>.json`` is never rewritten. The listings are verified against ``attestation`` -- the keyed ``MANIFEST-<sha8>.<lsha8>.json``
    when a re-attestation is what verified the canonical's current listings (the mirror selects it from :func:`listings_verdict`), else the
    plain ``.json`` beside the lines -- and ``ok`` iff the report is ``ok`` (every file its manifest's, every listing it verified in place),
    the FILES match ``mine`` (the grading host's manifest, already equal to the generation's) and, when ``listings_sha256`` names the
    attested listings, the report verified THOSE; a generation whose listings are unattested or moved is judged on its files alone (before
    this the weekly failed forever on a healthy re-attested generation, and an amber listing state skipped the re-hash). The counts and the
    first paths of what differs ride along. A job that does not complete raises :class:`StoresError`."""
    full_dir = f"{dest.stores_dir}/{FULL_SUBDIR}/{comp.slug}"
    remote.mkdir(full_dir)
    out = f"{full_dir}/VERIFIED-{sha8}-{_stamp(now())}.json"
    manifest = f"{gen_dir}/MANIFEST-{sha8}.sha256"
    rec: Dict[str, Any] = {"report_path": out, "generation": f"{gen_dir}/{sha8}", "attestation": attestation, "expected_listings_sha256": listings_sha256}
    if inline:
        if not dest.is_local:
            raise StoresError(f"full re-hash inline runs where the tree is: destination {dest.name} is remote ({dest.host})")
        report, dest_stat = verify_generation(f"{gen_dir}/{sha8}", manifest, workers=workers, job_id=os.environ.get("SLURM_JOB_ID"), attestation=attestation)
        write_verified(out, report, dest_stat)
    else:
        script_args = ["verify", f"{gen_dir}/{sha8}", manifest, out, str(dest.verify_cpus)] + (["--attestation", attestation] if attestation else [])
        job_id, tlim = remote.submit(f"store-full-{comp.slug}", script_args, mine.bytes, mine.n_files,
                                     measured_rate(root, comp.slug, dest), script=pin.sbatch_path if pin else None)
        rec.update(job_id=job_id, time_limit_s=tlim)
        log(f"[{comp.slug}] full re-hash job {job_id} over {gen_dir}/{sha8} (--time {slurm_time(tlim)}); watching")
        watched = remote.watch(job_id)
        rec["watch"] = watched
        if watched["state"] != "DONE":
            raise StoresError(f"full re-hash job {job_id}: {watched.get('terminal') or watched['state']}")
        report, rec["report_rereads"] = read_verified_report(remote, out)
    files_ok = report.get("ok") is True and _verified_matches(report, mine, listings=False)
    rec["ok"] = files_ok and (listings_sha256 is None or report.get("listings_sha256") == listings_sha256)
    rec["report"] = {k: report.get(k) for k in ("ok", "hashed", "hashed_bytes", "elapsed_s", "rate_mb_s", "host", "verified_utc", "job_id",
                                                "listings_sha256", "n_directories")}
    rec["counts"] = {k: len(report.get(k) or []) for k in ("missing", "corrupt", "extra")}
    rec["n_listing_mismatch"] = len(report.get("listing_mismatch") or [])
    rec["first"] = {k: (report.get(k) or [])[:10] for k in ("missing", "corrupt", "extra")}
    rec["report_present"] = bool(report)
    return rec


@dataclass
class AttestResult:
    """The outcome of `stores attest` for one component (:func:`attest_listings`): attested | already_attested | content_differs |
    listing_only_drift | attest_verify_failed | no_current | current_dangling | current_unverified | current_unsealed | current_unrecorded |
    no_listings | verify_job_failed | verify_watch_timeout | verify_report_missing | skipped | error; the level from
    :data:`ATTEST_OUTCOME_LEVELS`."""
    slug: str
    outcome: str
    content_sha8: Optional[str] = None
    listings_sha256: Optional[str] = None            # the canonical's current listings (the grading host manifest's)
    published_attestation: Optional[str] = None      # the attestation recorded for the published generation after this run
    job_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def level(self) -> str:
        return ATTEST_OUTCOME_LEVELS.get(self.outcome, "red")

    @property
    def ok(self) -> bool:
        return self.level in ("green", "amber")

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        detail = d.pop("detail")
        return {**d, "level": self.level, "elapsed_s": round(self.elapsed_s, 3), **detail}


def attest_listings(root: str, dest: Destination, comp: Component, runner=None, log: Callable[[str], None] = lambda s: None,
                    now: Callable[[], str] = R.utc_now, pin: Optional[PinnedTree] = None, hasher: Callable[[str], str] = R.file_sha256,
                    sleep: Callable[[float], None] = time.sleep) -> AttestResult:
    """`registry stores attest`: attest the PUBLISHED generation's directory listings (MANIFEST_RULES["directories"]) for ONE component whose
    grading-host copy is that generation's content -- the designed path for a generation verified before the listings rule
    (``listings_unattested`` on the mirror) and for a listing moved since its attestation (``listings_changed``). The steps: (1) the grading
    host's manifest through the stat cache, its files written to the state dir (the keyed ``MANIFEST-<sha8>.<lsha8>.json`` added when absent;
    attestations are immutable: this adds, never restates); (2) something must be published (``no_current``), `current` must name that
    content (``content_differs`` otherwise: the mirror's next tick publishes it, which attests the current listings), its tree must
    stand (``current_dangling``), the live ``VERIFIED-<sha8>.json`` must admit its files (``current_unverified``) -- and each of those
    three refusals RECORDS what the mirror records when it reads the same facts (before this the attest returned the red
    and wrote only ``last_run``, so a generation whose tree had been renamed away, or whose live report was gone, stayed
    ``generation.state = published`` with ``published = <sha8>`` and `stores status` showed it green and verified): ``no_current`` drops
    ``published`` and names the state; ``current_dangling`` drops ``published`` and carries the ``current_dangling`` dangle record (the
    renamed trees beside the pointer listed, as the carry lists them; `stores status` prints the PROBLEM line, the plan pins nothing,
    the mirror's next tick repairs it); ``current_unverified`` keeps ``published`` (the tree stands) and names the state, red, as the
    carry does -- the recorded ``published_attestation`` is left alone in every case (it describes the tree, which a pin naming it
    checks the listing of; dropping it would widen what a reader admits) -- and its SEALED and PUBLISHED records must stand beside it
    (``current_unsealed`` / ``current_unrecorded``, red, no job: the evidence the mirror calls a `current` green by -- CURRENT_EVIDENCE --
    is never restored by an attestation of the listings, so the attest refuses before the ``already_attested`` and the verified paths
    alike and records the problem as the generation's state; `stores mirror --adopt-unrecorded` writes what is missing, then the attest
    runs -- before this the attest wrote ``generation.state = published`` over a `current_unrecorded` the mirror had just
    named, and `stores status` read green while the cluster's authority check refused the generation); (3) a report that already verified THESE listings -- the live
    one, or a keyed re-attestation -- is ``already_attested`` (the attestation recorded, no job); a keyed report that refused them is
    ``listing_only_drift`` (no job: the verdict stands until the canonical's listings move again) -- NEITHER while a re-attestation
    integrity failure stands for the generation (:func:`attest_integrity`; an older report for these listings is never
    reused over a later failure -- the attest re-verifies in full, and a pass is remembered as ``last_full_pass``); (4) else the keyed attestation is
    written beside the tree (once, as the publish path writes it) and the PINNED verifier runs over the sealed tree against THAT
    attestation into ``VERIFIED-<sha8>.<lsha8>.json`` (:func:`verify_generation` with ``attestation``; its snapshot keyed too) -- the live
    report and snapshot are never rewritten, the tree is never renamed; (5) a report that admits the files AND the listings records
    ``published_attestation = <listings_sha256>`` in ``state/store_mirror.json`` (``attested``), so the window's plan pins it and
    :func:`revalidate`, :func:`admit_generation` and :func:`manifest_index` check listings for it from then on.

    THE RULE for a listing the tree does not satisfy: the attest NEVER changes the generation. A ``listing_mismatch`` in the report with
    the files matching is refused by name, ``listing_only_drift`` (AMBER: a listing-coverage limitation, not an integrity failure -- the pin
    is unchanged, the listing readers keep the grading host route, the generation is not shown listing-attested, the unit does not exit 1 for
    it), naming the direction the verifier recorded -- directories the
    generation holds that the canonical lacks (the live ``.partial/`` case: nothing removes an entry from an immutable generation), and
    attested directories the generation lacks (the canonical added an empty directory: nothing is created inside a published, sealed tree
    either; the tree's directories are 0555 and its directory count is sealed, and the precedent -- the A -> B -> A rebuild -- copies a
    fresh tree rather than rewriting one a window may be reading). Both wait for the next FILE change, whose publication attests the
    current listings. The alternative considered and rejected: un-sealing the parent, creating the attested empty directory and
    re-sealing (a write to a tree "never written after its verify", a moved directory count in SEALED, and a parent listing changed
    under any window pinned to the earlier attestation). A listing change the tree DOES satisfy -- an excluded entry (a cache
    directory) landing or leaving under an attested directory, which the staged expectation never carries -- is attested. Nothing here
    switches `current`, renames a tree, or touches the live report; a report that finds the FILES wrong is ``attest_verify_failed``,
    named and left to the operator like the weekly full verify's failure. Every exit stamps ``last_run.outcome = attest:<outcome>``."""
    t0 = time.monotonic()
    res = AttestResult(comp.slug, "error")
    if comp.authority not in ("grading_host_copy", "cluster_a") or comp.producer == "cluster":
        res.outcome = "skipped"
        res.detail["why"] = (f"authority {comp.authority}, producer {comp.producer or 'none'}: the grading host holds no canonical copy of this component "
                             f"whose listings this command could attest")
        res.elapsed_s = time.monotonic() - t0
        return res
    gen_dir = dest.gen_dir(comp.slug)
    sdir = component_state_dir(root, comp.slug)
    remote = Remote(dest, runner)
    source = comp if comp.authority == "grading_host_copy" else replace(comp, base=grading_host_mirror_path(root, comp))
    prior_rec = (load_state(root).get("components") or {}).get(comp.slug) or {}
    gen_rec: Dict[str, Any] = dict(prior_rec.get("generation") or {})
    res.published_attestation = prior_rec.get("published_attestation") or None
    patch: Dict[str, Any] = {}
    last_run: Dict[str, Any] = {}

    def refuse(outcome: str, why: str, **fields: Any) -> AttestResult:
        res.outcome = outcome
        res.detail["why"] = last_run["why"] = why
        res.detail.update(fields)
        return res

    def drift(report: Mapping[str, Any], gen_path: str, report_path: str) -> AttestResult:
        res.outcome = "listing_only_drift"
        res.detail.update(listing_mismatch=list(report.get("listing_mismatch") or [])[:LIST_CAP],
                          dirs_generation_only=list(report.get("dirs_generation_only") or [])[:LIST_CAP],
                          n_dirs_generation_only=report.get("n_dirs_generation_only"),
                          dirs_canonical_only=list(report.get("dirs_canonical_only") or [])[:LIST_CAP],
                          n_dirs_canonical_only=report.get("n_dirs_canonical_only"), report_path=report_path, why=_drift_why(report, gen_path))
        last_run["why"] = res.detail["why"]
        gen_rec.update(content_sha8=res.content_sha8, state="listing_only_drift", listings_sha256=res.listings_sha256, attest_report=report_path)
        patch["generation"] = gen_rec
        return res

    try:
        pre = build_manifest(source, sdir, hasher=hasher)
        sha8 = pre.content_sha8
        res.content_sha8, res.listings_sha256 = sha8, pre.listings_sha256 or None
        patch["latest_manifest"] = pre.summary(write_manifest_files(pre, sdir))
        current = remote.readlink(f"{gen_dir}/current")
        if current is None:
            # nothing published: the state says so (the reversed mirror's `no_current` shape), nothing is advertised to a plan
            patch.update(published=None, current_dangling=None,
                         generation={"content_sha8": None, "state": "no_current", "current_read": None, "grading_host_content_sha8": sha8})
            return refuse("no_current", f"nothing is published under {gen_dir}; the mirror publishes first, and that publication attests the listings")
        if current != sha8:
            return refuse("content_differs", f"`current` -> {current} under {gen_dir} but the grading host copy's content is {sha8}: the mirror's next tick "
                                             f"publishes it, and that publication attests the current listings; nothing to attest here", current=current)
        gen_path = f"{gen_dir}/{sha8}"
        # the mirror's repair of THIS pointer began and no publication has ended it (the dangle record names this content): the tree the
        # attest reads may be that repair's partial copy or its verified-but-unsealed replacement, and the refusals below keep the record
        # -- and `published` dropped -- so the mirror's next tick resumes or finishes the repair instead of reading a healthy `current`
        # (before this the refusals cleared the record and re-advertised the pointer)
        prior_dangle = prior_rec.get("current_dangling") or {}
        repair_pending = prior_dangle.get("content_sha8") == sha8
        if not remote.isdir(gen_path):
            # the dangle record, as the carry writes it THE MOMENT the dangle is established (`published` dropped, the generation marked) and
            # before the one further step that can raise (the listing of the renamed trees beside the pointer), so a refusal that dies there
            # still leaves an honest record; a dangle the last tick already found keeps its `found_utc`
            found_utc = (prior_dangle.get("found_utc") if repair_pending else None) or now()
            dangle = {"content_sha8": sha8, "found_utc": found_utc, "renamed_aside": None, "resumed": False,
                      "why": f"current -> {sha8} but {gen_path} is not a directory; beside it: not yet listed"}
            patch.update(published=None, current_dangling=dangle,
                         generation={"content_sha8": sha8, "state": "current_dangling", "current_read": current, "found_utc": found_utc})
            aside = remote.renamed_aside(gen_dir, sha8)
            found = ", ".join(aside) if aside else "no renamed tree"
            dangle.update(renamed_aside=aside, why=f"current -> {sha8} but {gen_path} is not a directory; beside it: {found}")
            return refuse("current_dangling", f"`current` -> {sha8} under {gen_dir} but {gen_path} is not a directory (beside it: {found}); nothing is "
                                              f"published until the mirror repairs it (D882: its next tick copies afresh, re-verifies and republishes); "
                                              f"nothing attests listings over a tree that is not there", renamed_aside=aside, found_utc=found_utc)
        verified = remote.read_json(f"{gen_dir}/VERIFIED-{sha8}.json")
        if not _verified_matches(verified, pre, listings=False):
            # the carry's `current_unverified` shape: the tree stands (no dangle), `published` names it, the generation's state is the red --
            # unless the tree is a pending repair's partial copy, which keeps the dangle's record and nothing published (the mirror resumes it)
            patch.update(published=None if repair_pending else current, current_dangling=prior_dangle if repair_pending else None,
                         generation={"content_sha8": sha8, "state": "current_unverified", "current_read": current, "verified_utc": verified.get("verified_utc")})
            return refuse("current_unverified", f"no VERIFIED-{sha8}.json admitting the generation's files stands under {gen_dir}; nothing attests "
                                                f"listings over unverified content", verified_present=bool(verified))
        sealed_rec = remote.read_json(f"{gen_dir}/SEALED-{sha8}.json")
        pub_rec = remote.read_json(f"{gen_dir}/PUBLISHED-{sha8}.json")
        problems = [name for name, missing in (("current_unsealed", not sealed_rec), ("current_unrecorded", not pub_rec)) if missing]
        if problems:
            # a `current` without its seal or its publication record is the mirror's red by name (CURRENT_EVIDENCE); an attestation of
            # its listings restores neither, so nothing here is attested or recorded as published -- the generation's state names the
            # problem the attest read, `stores mirror --adopt-unrecorded` writes what is missing, and the attest runs after it (round 1 on 2c)
            gen_rec.update(content_sha8=sha8, state=problems[0], current_read=current, authority_problems=problems)
            # the replacement tree of a pending repair stands exactly so (its seal went aside with the previous tree's reports; its own is
            # not written yet): the dangle's record and `published` dropped stay for the mirror to finish the publication
            # the seal gates publication on every route (the final pass on 2c): a `current` without its SEALED record advertises nothing
            # here either, pending repair or not -- the mirror's own refusal drops it, and an attest must not put it back
            patch.update(generation=gen_rec, published=None if (repair_pending or not sealed_rec) else current,
                         current_dangling=prior_dangle if repair_pending else None)
            return refuse(problems[0], "nothing attests listings over a `current` without its evidence -- "
                          + evidence_why(problems, gen_dir, sha8, comp, adoptable=True) + "; then `stores attest` attests the listings",
                          authority_problems=problems, adoptable=True)
        if not pre.listings_sha256:
            return refuse("no_listings", f"the grading host manifest of {comp.slug} attests no directory listing; nothing to attest")
        out = keyed_report_path(gen_dir, sha8, pre.listings_sha256)
        verdict, att = listings_verdict(remote, gen_dir, sha8, verified, pre)
        # a report that already speaks for THESE listings is reused only while no re-attestation INTEGRITY FAILURE stands for the generation
        # (attest_integrity: the recorded attest_integrity_failed, re-read, superseded only by a full hash stamped LATER than it) -- round 2
        # on 2c: the canonical's listings returned to a set an OLDER keyed report had attested after a later re-attestation of other
        # listings had hashed a corrupt file, and `already_attested` wrote `generation.state = published` over the mirror's standing red
        # (the cached drift return would have demoted it to amber the same way). The attest re-verifies in full instead: a later full hash
        # of every file is the only thing that clears the failure, and this is one
        standing = attest_integrity(remote, gen_dir, sha8, pre, prior_rec, verdict, att, False, now=now) if verdict in ("attested", "drift") else None
        if standing:
            res.detail["standing_failure"] = standing
            log(f"[{comp.slug}] {os.path.basename(str(standing.get('report_path')))} found the files wrong on {standing.get('verified_utc')}, after the "
                f"report for these listings; re-verifying in full")
        elif verdict == "attested":
            res.outcome = "already_attested"
            res.published_attestation = (att or {}).get("listings_sha256")
            by = f"VERIFIED-{sha8}.json" if verified.get("listings_sha256") == res.published_attestation else os.path.basename(out)
            res.detail.update(attested_by=by, n_directories=pre.n_directories,
                              why=f"{by} already verified the canonical's current listings {str(res.published_attestation)[:8]} against {gen_path}; recorded")
            patch["published_attestation"] = res.published_attestation
            gen_rec.update(content_sha8=sha8, state="published", listings_sha256=res.published_attestation)
            patch["generation"] = gen_rec
            return res
        elif verdict == "drift":
            return drift(att or {}, gen_path, out)
        if verdict == "integrity_failed":
            log(f"[{comp.slug}] {os.path.basename(out)} found the files wrong on {(att or {}).get('verified_utc')}; re-verifying in full")
        # the keyed attestation beside the tree, once (as the publish path writes it): what the verifier reads the listings from
        keyed_manifest = attestation_json_path(f"{gen_dir}/MANIFEST-{sha8}.sha256", pre.listings_sha256)
        if remote.read(keyed_manifest) is None:
            remote.write(keyed_manifest, json.dumps(pre.meta(), indent=1, sort_keys=True) + "\n")
        res.detail["keyed_manifest"] = keyed_manifest
        rate = measured_rate(root, comp.slug, dest)
        script_args = ["verify", gen_path, f"{gen_dir}/MANIFEST-{sha8}.sha256", out, str(dest.verify_cpus), "--attestation", keyed_manifest]
        res.job_id, tlim = remote.submit(f"store-attest-{comp.slug}", script_args, pre.bytes, pre.n_files, rate, script=pin.sbatch_path if pin else None)
        log(f"[{comp.slug}] attest job {res.job_id} submitted over {gen_path} against {os.path.basename(keyed_manifest)} (--time {slurm_time(tlim)}); watching")
        gen_rec.update(content_sha8=sha8, attest_job_id=res.job_id)
        update_state(root, comp.slug, {**patch, "generation": gen_rec,
                                       "last_run": {"utc": now(), "outcome": f"{ATTEST_OUTCOME_PREFIX}verify_submitted", "job_id": res.job_id}}, dest.name)
        watched = remote.watch(res.job_id)
        res.detail["watch"] = watched
        if watched["state"] != "DONE":
            res.outcome = "verify_job_failed" if watched["state"] in ("FAILED", "MISSING") else "verify_watch_timeout"
            last_run["terminal"] = watched.get("terminal")
            return res
        report, waited = _await_report(remote, out, res.job_id, dest.report_wait_s, dest.report_poll_s, sleep, log)
        res.detail["report_wait"] = waited
        if not report:
            return refuse("verify_report_missing", f"attest job {res.job_id} finished (watch DONE) but {os.path.basename(out)} written by it was not readable "
                                                   f"from {dest.name} within {dest.report_wait_s:.0f} s ({waited['attempts']} reads; last: {waited['last']}); nothing "
                                                   f"recorded -- the next `stores attest` (and the mirror's next tick) reads it once it is visible", report_path=out)
        res.detail["report_path"] = out
        res.detail["report"] = {k: report.get(k) for k in ("ok", "hashed", "elapsed_s", "rate_mb_s", "host", "verified_utc", "job_id", "n_directories", "listings_sha256")}
        if _verified_matches(report, pre, listings=False):
            # every file hashed and its manifest's: a standing integrity failure is recovered from, and the pass is remembered as `--full`'s
            # is (last_full_pass), so a failing keyed report of OTHER listings left beside the generation stays superseded when the
            # canonical's listings return to it (attest_integrity reads the record and the current listings' report, never every report)
            patch.update(attest_integrity_failed=None,
                         last_full_pass={"content_sha8": sha8, "verified_utc": report.get("verified_utc"), "report_path": out})
        if _verified_matches(report, pre):
            res.outcome = "attested"
            res.published_attestation = pre.listings_sha256
            res.detail.update(n_directories=pre.n_directories, n_empty_directories=pre.n_empty_directories)
            patch["published_attestation"] = pre.listings_sha256
            gen_rec.update(content_sha8=sha8, state="published", listings_sha256=pre.listings_sha256, attested_utc=report.get("verified_utc"),
                           attest_report=out, attest_job_id=res.job_id)
            patch["generation"] = gen_rec
            return res
        if _verified_matches(report, pre, listings=False):
            return drift(report, gen_path, out)
        counts = {k: len(report.get(k) or []) for k in ("missing", "corrupt", "extra")}
        # an INTEGRITY failure: recorded in the state so the mirror names it, red, on every tick until a later full hash passes (attest_integrity)
        patch["attest_integrity_failed"] = _failure_record(out, report, sha8, now())
        gen_rec.update(content_sha8=sha8, state="attest_integrity_failed", attest_failure=patch["attest_integrity_failed"])
        patch["generation"] = gen_rec
        return refuse("attest_verify_failed", f"the re-verify of {gen_path} does not match its manifest ({counts}; report {out}): nothing renamed, `current` "
                                              f"untouched; recorded as attest_integrity_failed, which the mirror names on every tick until a fresh `stores attest` "
                                              f"or `stores mirror --full` re-hashes the generation and passes -- the operator decides",
                      **counts, first={k: (report.get(k) or [])[:10] for k in counts})
    except (StoresError, subprocess.SubprocessError, OSError) as exc:
        res.outcome = "error"
        res.detail["error"] = last_run["error"] = f"{type(exc).__name__}: {exc}"[:800]
        return res
    finally:
        res.elapsed_s = time.monotonic() - t0
        update_state(root, comp.slug, {**patch, "last_run": {"utc": now(), "outcome": f"{ATTEST_OUTCOME_PREFIX}{res.outcome}", **last_run}}, dest.name)


def pull_rsync_argv(dest: Destination, gen_path: str, target: str) -> List[str]:
    """The PULL of a published generation into the registry-owned mirror directory: ``nice ionice -c3 rsync -rlt --partial
    <generation>/ <target>/``. rsync deletes nothing here either (the standing rule of this module): an entry the generation no longer
    holds is removed afterwards, by name, by :func:`prune_to_manifest` -- in that registry-owned directory and nowhere else."""
    # the generation is sealed (0555 directories, read-only files; without -p rsync carries the source mode through the umask) and the
    # mirror is the registry's own writable copy: user-writable so the escaped names can be restored in place and the next pull can update
    argv = ["nice", "-n", "10", "ionice", "-c3", "rsync", "-rlt", "--chmod=Du+w,Fu+w", "--partial", f"--partial-dir={PARTIAL_DIR}",
            "--stats", "--timeout=600"]
    if not dest.is_local:
        argv += ["-e", RSYNC_SSH, f"{dest.host}:{gen_path}/"]
    else:
        argv.append(f"{gen_path}/")
    argv.append(f"{target}/")
    return argv


def _remove_entry(path: str) -> None:
    """One entry of the registry-owned mirror removed by name: a directory tree, else the file or link."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def clear_escaped_tops(target: str, rels: Iterable[str]) -> Dict[str, Any]:
    """BEFORE the pull's rsync: every entry standing in the mirror at the ESCAPED spelling of an escaped top (:func:`escape_tops` of the
    generation's rels) is removed by name. What stands there is never current content -- rsync cannot match a restored `d:x` against the
    source's `d%3Ax`, so an escaped subtree arrives afresh every pull -- but the previous pull's RESTORED copy of a chain neighbor (a true
    `a%3Ab`, which this generation spells `a%253Ab`) or an escaped copy a crash left between the rsync and the restore. Left standing,
    rsync would merge a directory's old entries into the fresh copy, or refuse a type change outright (measured 2026-10-04: `could not make
    way for new regular file`, rc 23, when a non-empty directory stands where the source has a file -- rsync deletes nothing without
    --delete, by this module's rule), and the pull would fail the same way every tick. Returns the count and the first names."""
    cleared: List[str] = []
    for top in escape_tops(rels):
        esc = escape_rel(top)
        p = os.path.join(target, esc)
        if os.path.lexists(p):
            _remove_entry(p)
            cleared.append(esc)
    return {"cleared": len(cleared), "cleared_first": cleared[:10]}


def restore_escaped_names(target: str, true_links: Mapping[str, str]) -> Dict[str, int]:
    """A pulled tree carries the DESTINATION spellings (:data:`NAME_ESCAPE_RULE`): every path component is unescaped in place, deepest
    first (the rule is a bijection, so no table is needed: on the destination every '%' is followed by '25' or '3A'), and every attested
    symlink whose on-disk target is the escaped spelling of its true target is rewritten to the true one (``true_links``: rel -> target
    from the generation's manifest). Per directory in TWO PHASES: first every escaped entry is moved to a collision-free
    staging name (:data:`RESTORE_STAGE_PREFIX`), then each staged entry to its true name -- a generation may hold both `a:b` and `a%3Ab`,
    spelled `a%3Ab` and `a%253Ab` here, and a one-pass rename that met `a%253Ab` first would have taken the escaped `a%3Ab` beside it
    for a stale copy and deleted it. Once every escaped sibling is staged away, a true-named entry already standing is the PREVIOUS
    pull's restored copy (rsync cannot match `d:x` against the source's `d%3Ax`, so an escaped subtree arrives afresh every pull, as the
    push carries it by tar every generation): it is replaced by the fresh copy -- the mirror holds nothing but the generation. Returns the
    counts."""
    renamed = replaced = 0
    for dirpath, dirnames, filenames in os.walk(target, topdown=False, followlinks=False):
        escaped = [(name, unescape_name(name)) for name in list(dirnames) + list(filenames) if unescape_name(name) != name]
        if not escaped:
            continue
        staged: List[Tuple[str, str]] = []
        for name, true in escaped:                      # phase 1: every escaped entry of this directory staged away
            while True:
                tmp = os.path.join(dirpath, f"{RESTORE_STAGE_PREFIX}{uuid.uuid4().hex[:12]}")
                if not os.path.lexists(tmp):
                    break
            os.rename(os.path.join(dirpath, name), tmp)
            staged.append((tmp, true))
        for tmp, true in staged:                        # phase 2: each to its true name; what stands there is the previous pull's copy
            dst = os.path.join(dirpath, true)
            if os.path.lexists(dst):
                _remove_entry(dst)
                replaced += 1
            os.rename(tmp, dst)
            renamed += 1
    relinked = 0
    for rel, true_target in true_links.items():
        p = os.path.join(target, rel)
        if os.path.islink(p) and os.readlink(p) != true_target and os.readlink(p) == escape_target(true_target):
            os.remove(p)
            os.symlink(true_target, p)
            relinked += 1
    return {"renamed": renamed, "replaced": replaced, "relinked": relinked}


def prune_to_manifest(target: str, keep_rels: Iterable[str]) -> Dict[str, int]:
    """After a pull, the registry-owned mirror holds exactly the generation: every regular file and symlink under ``target`` whose true
    rel the manifest does not name is removed, then every directory left empty (deepest first). The rsync partial directory is left
    alone. The target is the registry's own copy of an immutable generation and nothing else lives in it; a producer's root is never
    touched by this (the mirror of a cluster producer is never its base). Returns the counts; the hash verification that follows is
    what proves the result IS the generation."""
    keep = set(keep_rels)
    removed_files = removed_dirs = 0
    for dirpath, dirnames, filenames in os.walk(target, topdown=False, followlinks=False):
        rel_dir = os.path.relpath(dirpath, target)
        rel_dir = "" if rel_dir == "." else rel_dir
        if rel_dir.split("/")[0] == PARTIAL_DIR:
            continue
        for name in filenames:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if rel not in keep:
                os.remove(os.path.join(dirpath, name))
                removed_files += 1
        for name in dirnames:
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                rel = f"{rel_dir}/{name}" if rel_dir else name
                if rel not in keep:
                    os.remove(p)
                    removed_files += 1
            elif name != PARTIAL_DIR and not os.listdir(p):
                os.rmdir(p)
                removed_dirs += 1
    return {"pruned_files": removed_files, "pruned_dirs": removed_dirs}


def _carry_deferral(remote: Remote, gen_dir: str, current: Optional[str], comp: Component, now: str) -> Optional[Dict[str, Any]]:
    """``min_publish_interval_s`` for the carry, as :func:`push_component` applies it: the `current` record's age; the deferral detail, or None."""
    if not (comp.min_publish_interval_s and current):
        return None
    pub = remote.read_json(f"{gen_dir}/PUBLISHED-{current}.json")
    age = _age_s(pub.get("published_utc"), now) if pub else None
    if age is not None and age < comp.min_publish_interval_s:
        return {"why": f"published {age:.0f} s ago; min_publish_interval_s {comp.min_publish_interval_s}: the next tick carries",
                "published_utc": pub.get("published_utc"), "min_publish_interval_s": comp.min_publish_interval_s}
    return None


def mirror_authority(root: str, dest: Destination, comp: Component, runner=None, dry_run: bool = False, full: bool = False,
                     hasher: Callable[[str], str] = R.file_sha256, log: Callable[[str], None] = lambda s: None,
                     now: Callable[[], str] = R.utc_now, pin: Optional[PinnedTree] = None, lock_wait_s: float = PUBLISH_LOCK_WAIT_S,
                     adopt_unrecorded: bool = False, workers: int = 16, bound_s: Optional[float] = None) -> MirrorResult:
    """`stores mirror` for a ``cluster_a`` component: the canonical copy is the published generation, so nothing is carried
    from the grading host -- the grading host mirror is VERIFIED against it. By producer:

    * ``grading_host:<session>``: the grading host mirror's manifest (the stat cache; `--full` reads every file) against the generation's. Equal ->
      ``verified``. Unequal (or nothing published) -> ``unpushed_changes`` (amber: the producer's `stores push` publishes; the mirror
      copies nothing) -- unless ``transitional_carry``, when the registry publishes on the producer's behalf through the push flow
      (:func:`mirror_component`, mode ``carry``, the producer's session recorded; ``change_utc`` = the grading host mtime of the changed files;
      ``min_publish_interval_s`` defers it like a push). A carry past ``carry_deadline`` still runs and is reported ``carry_overdue`` --
      stopping it would leave the shards on stale data over a date; the status row shows the overdue carry.
    * ``frozen``: verify only. A difference is ``frozen_differs`` (red), never repaired by the mirror -- the operator decides which copy is
      the truth; nothing published is ``no_current`` (red; the one-time move publishes it).
    * ``cluster``: :func:`pull_component`.

    The published generation's manifest is copied beside the grading host's state when no copy is there (the window's home tree reads the
    pin's manifest from it). `--full` adds :func:`full_rehash` of the published generation when the two sides agree (``full_verify_failed``
    otherwise: named, nothing renamed, `current` untouched) -- against the attestation that verified the canonical's current listings,
    as the carry's does (:func:`listings_verdict`), and a pass is remembered (``last_full_pass``). Two sides that agree are ``verified``
    only when no re-attestation INTEGRITY FAILURE stands for the generation (:func:`attest_integrity`: the ``attest_integrity_failed``
    record a `stores attest` wrote, and the keyed ``VERIFIED-<sha8>.<lsha8>.json`` of the canonical's current listings read on every
    tick) -- ``attest_integrity_failed`` (red) otherwise, zero submissions, until a later full hash of every file passes (this tick's
    `--full`, a remembered pass, a fresh `stores attest`). Before this the reversed mirror compared the two manifests
    and returned ``verified`` over a generation whose re-attestation had hashed a corrupt file the day before; the standing red the
    carry keeps held only for ``grading_host_copy`` components. Two sides that DIFFER name the same standing failure first:
    the record is re-read against the published generation's own manifest facts (:func:`generation_facts`; the grading host's manifest is
    not the generation's there), and ``attest_integrity_failed`` outranks ``unpushed_changes`` (amber), ``frozen_differs`` and a DEFERRED
    carry (``deferred``, green) while `current` still names the failing generation -- a carry that is not deferred runs, because the
    fresh, fully verified generation it publishes supersedes `current` (as the producer's push does). A dry run reads only (the stat
    cache read, never written) and reports what a tick would conclude (``would``). ``bound_s`` (`stores mirror --bound-s`, the window's
    carry) rides into the transitional carry's :func:`mirror_component` call unchanged, so the carry's verify watch is bounded by the
    controller's bound exactly as a ``grading_host_copy``
    carry's is (before this, this route dropped it, and a stalled verifier of a changed transitional component held the window --
    timers stopped, the maintenance flag up -- for the destination's four-hour default instead of the window's hour).

    EVERY run first SETTLES a verify job the component's record names and no run has proved terminal (:func:`names_live_verifier`:
    ``awaiting_verify`` with a job id -- a `stores push` of this component cut short, a transitional carry's unconfirmed cancel or lost
    watch), by the one terminal rule :func:`settle_awaiting_verify` applies, BEFORE any outcome of this route can replace that record: a
    job reading non-terminal is ``verify_in_flight``, a job the queue and the accounting cannot prove either way ``verify_unconfirmed``
    -- nothing copied, renamed or switched, the record standing with its job id and this run's ``settle`` reading, the grading host mirror's
    absence (when it is absent too) noted beside it under ``also``, never instead of it -- and a proved terminal job lets the run go on
    (``awaiting_verify_settled``). And ``finish`` never writes over such a record on its own: an outcome that is not a settlement of
    the job rides beside it (``also``), whichever path reached it. Review round 4 MEDIUM: a tick that found the grading host directory
    absent wrote ``no_grading_host_mirror`` over an ``awaiting_verify`` record, job id and all (``finish`` set the generation unconditionally;
    ``update_state`` merges shallowly), and when the directory returned the carry's guard, which reads only the record, saw no job and
    renamed and recopied the generation the verifier still read.

    A DANGLING ``current`` PUBLISHES NOTHING. While the tree the pointer names is not there, ``published`` stays
    unset on every exit (``finish``), and it is DROPPED the moment the pointer and the absent tree are read -- before the listing of what
    a rename left beside it, whose remote call can fail and take the run out through the error path rather than through ``finish``, which
    left the previous sha pinnable over a tree already read as absent. The dangle record stands -- written here the moment the dangle is
    established when no record of this content stands, kept as it stands when one does (the push flow's own repair writes its phase into
    it) -- until a tree is verified or republished. The outcome is ``dangling`` (red), or, for a transitional carry whose repair
    ``min_publish_interval_s`` holds, ``dangling_carry_deferred`` (red): a deferred REPAIR is not a successful deferral. A tree standing
    under the pointer does not by itself end the dangle: while the dangle record names `current` and the pointer's own evidence or the
    record's ``phase`` says the push flow's repair never finished its publication, the tree is that repair's partial copy or its
    unpublished replacement -- ``dangling_repair_pending`` (red), or the carry run again, which finishes the seal and the guarded switch.
    A tree restored WHOLE, with its own seal and record, ends the dangle as before. Before this, a `stores attest` that had found the
    tree gone and dropped ``published`` was undone by the next tick, which named the dangle red and restored ``published = current`` in the
    same breath -- so the window's plan (``cluster_reads.plan_window``) pinned a generation with no tree, and a deferred carry over a dangle
    reported ``deferred`` (green, exit 0) while nothing was readable."""
    if comp.producer == "cluster":
        return pull_component(root, dest, comp, runner=runner, dry_run=dry_run, full=full, hasher=hasher, log=log, now=now, pin=pin, workers=workers)
    t0 = time.monotonic()
    gen_dir = dest.gen_dir(comp.slug)
    sdir = component_state_dir(root, comp.slug)
    remote = Remote(dest, runner)
    source = grading_host_mirror_path(root, comp)
    frozen = comp.producer == "frozen"
    stamp = now()
    overdue = bool(comp.transitional_carry and comp.carry_deadline and comp.carry_deadline < stamp[:10])
    res = MirrorResult(comp.slug, "error", detail={"authority": comp.authority, "producer": comp.producer or None, "grading_host_mirror": source, "full": bool(full)})
    if comp.transitional_carry:
        res.detail["carry"] = {"deadline": comp.carry_deadline, "overdue": overdue, "producer": comp.producer}
    state: Dict[str, Any] = {"authority": comp.authority, "base": comp.base, "home_rel": comp.home_rel, "adapters": comp.adapters,
                             "revision": pin.revision if pin else None, "producer": comp.producer or None, "grading_host_mirror": source,
                             "transitional_carry": comp.transitional_carry, "carry_deadline": comp.carry_deadline, "carry_overdue": overdue}
    last_run: Dict[str, Any] = {}
    current: Optional[str] = None
    dangles: Optional[bool] = None          # None until `current` is read this tick; then whether it names a tree that is not there
    carried = False
    # the generation record an earlier run left for this component (the standing attest_integrity_failed / last_full_pass with it),
    # read ONCE: a verify job it names that no run has proved terminal is settled before anything else, and no outcome of this run
    # writes over it unless this run proved that job terminal
    recorded: Dict[str, Any] = dict((load_state(root).get("components") or {}).get(comp.slug) or {})
    pending: Dict[str, Any] = dict(recorded.get("generation") or {})
    settled: Dict[str, Any] = {"job_id": None, "state": None, "where": None, "refusal": None, "why": None}

    def settled_terminal() -> bool:
        """Did THIS run prove the recorded job terminal (the settle read it and refused nothing)?"""
        return bool(settled["job_id"]) and not settled["refusal"]

    def standing_record(also: Sequence[str]) -> Dict[str, Any]:
        """The recorded verifier's record carried forward: everything the run that submitted it wrote (``awaiting_verify``, the job id,
        the bound, the watch, the cancel), THIS run's settle reading in place of the last one, and what this run read beside the held
        job (``also``: outcomes noted beside the record, never instead of it -- this run's, not an accumulation)."""
        held = {k: v for k, v in pending.items() if k not in ("settle", "also")}
        if settled["job_id"]:
            held["settle"] = settled
        if also:
            held["also"] = list(also)
        return held

    def finish(outcome: str, why: Optional[str] = None, **fields: Any) -> MirrorResult:
        """Every exit but the carry's and the hold's: `current` is what the shards read (``published``), the generation record names the
        outcome -- unless the record names a verify job this run has not proved terminal, which STANDS with the outcome noted beside it
        (:func:`names_live_verifier`; the record is the only evidence of that job the next carry's guard reads). A dry run's outcome is
        always ``dry_run``: what a tick would conclude rides as ``would`` (and it writes nothing). A pointer read this tick that does NOT
        dangle ends a recorded ``current_dangling`` (the dangle's record, written by the carry or by a `stores attest` that found the
        tree gone): the tree standing is the event that ends a dangle, and a record left standing after it would keep
        `stores status` on its PROBLEM line and `verified: false` over a verified generation.

        A POINTER THAT DANGLES PUBLISHES NOTHING (the closing pass on 2c): while the tree `current` names is not there, ``published`` and
        ``res.published`` stay unset whatever the pointer says, and the dangle's record stands until a tree is verified or republished.
        Before this every exit here wrote ``published = current``, so a `stores attest` that had found the tree gone and dropped
        ``published`` was undone by the very next tick -- it named the dangle red and RE-ADVERTISED the pointer in the same breath, and
        the window's plan pinned a generation whose tree was absent."""
        if dry_run and outcome != "dry_run":
            fields.setdefault("would", outcome)
            fields["writes"] = "nothing"
            outcome = "dry_run"
        res.outcome = outcome
        if why:
            res.detail["why"] = last_run["why"] = why
        res.detail.update(fields)
        res.published = None if dangles else current
        state["published"] = None if dangles else current
        if dangles is False:
            state["current_dangling"] = None
        if names_live_verifier(pending) and not settled_terminal():
            # this outcome is no settlement of the recorded job: the record stands (`last_run.outcome` names the outcome too)
            res.detail["also"] = last_run["also"] = [outcome]
            state["generation"] = standing_record([outcome])
            return res
        state["generation"] = {"content_sha8": current, "state": outcome, "current_read": current, "grading_host_content_sha8": res.content_sha8,
                               **{k: v for k, v in fields.items() if k in ("verified_utc", "would", "unpushed_files", "differing", "full",
                                                                           "authority_problems", "adopted", "adoptable", "attest_failure")}}
        return res

    def hold() -> MirrorResult:
        """The run refused by the recorded verifier's name (``verify_in_flight`` / ``verify_unconfirmed``; ``settled`` says which and why):
        nothing copied, renamed or switched; the record stands with its job id and this settle's reading; the grading host mirror's absence, when
        it is absent too, is noted beside it (``also``), never instead of it. `current` is advertised as ``published`` -- what the shards
        read and what the window's plan pins -- ONLY with the evidence of its publication beside it (:func:`held_publication`); without it
        (an unverified, unsealed or unrecorded pointer, a tree that is gone, a recorded dangling repair no run has resolved) ``published``
        stays None
        and the record's dangle note stands, so the plan pins nothing for this component instead of routing an adapter at a tree nothing
        attests. A dry run reports the hold as ``would`` and writes nothing."""
        outcome = settled["refusal"]
        cur = remote.readlink(f"{gen_dir}/current")
        held = held_publication(remote, gen_dir, cur, recorded)
        also = [] if os.path.isdir(source) else ["no_grading_host_mirror"]
        if dry_run:
            res.outcome = "dry_run"
            res.detail.update(would=outcome, writes="nothing")
        else:
            res.outcome = outcome
        res.published = held["published"]
        res.detail.update(current=cur, awaiting_verify=settled)
        if held["record"]:
            res.detail["generation_record"] = held["record"]
        if cur and held["problems"]:
            res.detail["not_published"] = {"problems": held["problems"], "why": held["why"]}
            last_run["not_published"] = held["why"]
        res.detail["why"] = last_run["why"] = settled["why"]
        if also:
            res.detail["also"] = last_run["also"] = also
        state["published"] = held["published"]
        if held["current_dangling"]:
            state["current_dangling"] = held["current_dangling"]           # the dangle note stands until a publish clears it
        state["generation"] = standing_record(also)
        return res

    def carry() -> MirrorResult:
        """The transitional carry: the grading host copy published on the producer's behalf through the push flow (mode ``carry``).

        A deferral of the REPAIR of a dangling pointer is not a successful deferral (the closing pass on 2c): the tree `current` names is
        not there, so nothing is published and no shard can read the component -- ``dangling_carry_deferred`` (red) names that the publish
        interval holds the repair until it elapses, where ``deferred`` (green, exit 0) read as a healthy wait over an absent tree."""
        nonlocal carried
        deferral = _carry_deferral(remote, gen_dir, current, comp, now())
        if deferral and dangles:
            return finish("dangling_carry_deferred",
                          f"`current` -> {current} under {gen_dir} names no tree AND the carry that would republish it is deferred "
                          f"({deferral['why']}): nothing is published and no shard can read this component until the interval elapses "
                          f"and a tick carries the grading host copy ({mine.content_sha8}, {mine.n_files} files)",
                          **{k: v for k, v in deferral.items() if k != "why"})
        if deferral:
            return finish("deferred", deferral["why"], **{k: v for k, v in deferral.items() if k != "why"})
        session = comp.producer.split(":", 1)[1]
        meta = {"mode": "carry", "note": f"transitional carry on behalf of {comp.producer} (deadline {comp.carry_deadline}{'; OVERDUE' if overdue else ''})",
                "producer_session": session, "producer_host": socket.gethostname(), "source": source, "declared_base": comp.base}
        carried = True
        r = mirror_component(root, dest, replace(comp, base=source), runner=runner, hasher=hasher, log=log, now=now, pin=pin, allow=("cluster_a",),
                             publish_meta=meta, lock_wait_s=lock_wait_s, adopt_unrecorded=adopt_unrecorded, workers=workers, bound_s=bound_s)
        r.detail["carry"] = {"deadline": comp.carry_deadline, "overdue": overdue, "producer": comp.producer}
        update_state(root, comp.slug, {"grading_host_mirror": source, "carry_overdue": overdue, "carry": r.detail["carry"]}, dest.name)
        return r

    try:
        # a verify job the record names and no run has proved terminal is settled FIRST -- before the grading host directory is even looked for:
        # the record is the only evidence of that job, and a `no_grading_host_mirror` (or any other outcome of this route) written over it would
        # let the next carry's guard, which reads only the record, rename and recopy the generation its verifier still reads
        settled = settle_awaiting_verify(pending, dest, remote, log=lambda line: log(f"[{comp.slug}] {line}"))
        if settled["refusal"]:
            return hold()
        if settled["job_id"]:
            res.detail["awaiting_verify_settled"] = settled
        if not os.path.isdir(source):
            return finish("no_grading_host_mirror", f"{source} is not a directory: the grading host has no copy to verify against the published generation")
        try:
            mine = build_manifest(replace(comp, base=source), sdir, hasher=hasher, full=full, write_cache=not dry_run)
        except ManifestRefusal as exc:
            res.outcome = "refused"
            res.detail["reason"] = last_run["reason"] = str(exc)
            return res
        res.content_sha8, res.n_files, res.bytes, res.hashed, res.from_cache = mine.content_sha8, mine.n_files, mine.bytes, mine.hashed, mine.from_cache
        if not dry_run:
            state["latest_manifest"] = mine.summary(write_manifest_files(mine, sdir))
        current = remote.readlink(f"{gen_dir}/current")
        current_exists = bool(current) and remote.isdir(f"{gen_dir}/{current}")
        dangles = bool(current) and not current_exists
        prior_dangle = recorded.get("current_dangling") or {}
        repair_recorded = bool(current) and prior_dangle.get("content_sha8") == current     # the dangle's record names THIS pointer
        res.detail["current"] = current
        if current and not current_exists:
            # NOTHING IS PUBLISHED FROM THIS MOMENT, written into the state BEFORE the listing step below (the final pass on 2c). `finish`
            # drops ``published`` on every exit that reaches it, but the listing is a REMOTE call and an SSH outage there raises: the
            # exception path flushes this record and ``last_run = error`` without passing through `finish`, so the previous sha stayed
            # pinnable over a tree this tick had already read as absent -- a window pinned it and the shards read nothing
            state["published"] = None
            # The dangle's record follows the destination here as it does on the push flow's path: nothing is published
            # while the pointer dangles (`finish` above), and the dangle is RECORDED the moment it is established when no record of this
            # content stands -- minimally first, then the trees a rename left beside the pointer in one listing step, so a tick that dies
            # there still leaves an honest record -- which is what carries `stores status`'s PROBLEM line and keeps the next reader (a
            # later tick, a `stores attest`) on the same finding. A record that already names this content is KEPT as it stands: the push
            # flow's own repair writes its `resumed` / `phase` into it, and this tick learned nothing newer
            if not repair_recorded:
                why = f"current -> {current} but {gen_dir}/{current} is not a directory"
                state["current_dangling"] = {"content_sha8": current, "found_utc": now(), "renamed_aside": None, "resumed": False,
                                             "why": f"{why}; beside it: not yet listed"}
                if re.fullmatch(r"[0-9a-f]{8}", current):          # a generation name this command wrote; any other pointer is listed by nobody
                    aside = remote.renamed_aside(gen_dir, current)
                    state["current_dangling"].update(renamed_aside=aside, why=f"{why}; beside it: {', '.join(aside) if aside else 'no renamed tree'}")
            if comp.transitional_carry and not dry_run:
                return carry()                  # the push flow re-copies and re-points a dangling `current`
            return finish("dangling", f"`current` -> {current} under {gen_dir} names no tree; nothing is published and the shards cannot read this "
                                      f"component until it is re-published "
                                      f"({'`stores push ' + comp.slug + '`' if not frozen else 'the one-time move'})")
        if not current:
            if frozen:
                return finish("no_current", f"nothing is published under {gen_dir}: a frozen component's generation is written by the one-time move "
                                            f"(`stores push {comp.slug}`); the mirror never publishes it")
            if comp.transitional_carry and not dry_run:
                return carry()
            return finish("unpushed_changes", f"nothing is published under {gen_dir}; the producer {comp.producer} publishes the grading host copy "
                                              f"({mine.content_sha8}, {mine.n_files} files) with `stores push {comp.slug}`; nothing copied",
                          unpushed_files=mine.n_files, **({"would": "carry"} if comp.transitional_carry else {}))
        got = _generation_manifest(remote, gen_dir, current)
        if got is None:
            return finish("no_manifest", f"`current` -> {current} under {gen_dir} has no MANIFEST-{current}.sha256 beside it; not a layout this command wrote")
        text, theirs, theirs_meta = got
        diff = _manifest_diff(mine, theirs, _manifest_links(theirs_meta))
        equal = mine.content_sha8 == current and not any(diff["counts"].values())

        def published_failure() -> Optional[Dict[str, Any]]:
            """The re-attestation integrity failure standing for the PUBLISHED generation when the grading host copy is NOT its content: the
            record `stores attest` wrote, re-read against the generation's own manifest facts (no listing verdict: the canonical's listings
            are not this generation's, so no keyed report of them speaks for it)."""
            facts = generation_facts(theirs_meta, current)
            return attest_integrity(remote, gen_dir, current, facts, recorded, "none", None, False, now=now) if facts else None

        def carry_deferral() -> Optional[Dict[str, Any]]:
            return _carry_deferral(remote, gen_dir, current, comp, now()) if (comp.transitional_carry and not frozen) else None

        ev = current_evidence(remote, gen_dir, current, theirs_meta, len(theirs))
        verified, pub = ev["verified"], ev["published"]
        res.detail["generation_record"] = ev["record"]
        # THE PUSH FLOW'S REPAIR OF THIS POINTER IS PENDING. The dangle's record names `current` and no
        # publication has ended it, so the tree standing here is that repair's partial copy, or its replacement under the SAME name whose
        # own seal or guarded switch did not complete -- never a repaired `current`. Which of the two is read off the pointer: a repair
        # that ran to the end leaves its own seal and record beside the tree, and the one thing a same-content replacement inherits from
        # the tree it replaced is the publication record, so the repair's own `phase` is what tells that last case apart. Nothing is
        # published, the dangle's record stands, and the repair is finished by the flow that owns it -- the carry here, `stores push`
        # for a producer that publishes itself. Before this the mirror read `current_exists` alone and called the dangle over: it
        # restored `published`, CLEARED the record (so the push flow could no longer see its own unfinished repair) and, with the seal
        # written and only the switch missing, read the whole thing `verified` over the previous tree's publication record
        if repair_recorded and (prior_dangle.get("phase") == "publishing" or ev["problems"]):
            dangles = True
            if comp.transitional_carry and not dry_run:
                return carry()          # the push flow finishes the seal and the guarded switch of its own replacement (its pending path)
            missing = ", ".join(ev["problems"]) or "its seal and its record stand; its publication did not"
            return finish("dangling_repair_pending",
                          f"`current` -> {current} under {gen_dir} stands over the repair begun {prior_dangle.get('found_utc')}, which no "
                          f"publication has ended ({missing}): a present directory is not a repaired `current` -- nothing is published and "
                          f"the shards cannot read this component until the replacement's own seal and publication stand, which "
                          f"`stores push {comp.slug}` completes",
                          authority_problems=list(ev["problems"]), repair=dict(prior_dangle))
        # a `current` the grading host copy equals is green only with its evidence beside it (the push flow's rule): the
        # matching VERIFIED report, SEALED-<sha8>.json and PUBLISHED-<sha8>.json -- each absence is red by name (CURRENT_EVIDENCE), and
        # `--adopt-unrecorded` adopts a VERIFIED one (the seal when missing, the record when missing), as `stores push --adopt-unrecorded` does
        problems: List[str] = list(ev["problems"]) if equal else []
        adoptable = bool(problems) and "current_unverified" not in problems
        if dry_run:
            would = (problems[0] if problems else "verified") if equal else ("frozen_differs" if frozen else ("carry" if comp.transitional_carry else "unpushed_changes"))
            if would == "verified":
                # a standing re-attestation integrity failure is what a tick would name (read-only: the record and the keyed report)
                verdict, att_report = listings_verdict(remote, gen_dir, current, verified, mine)
                if attest_integrity(remote, gen_dir, current, mine, recorded, verdict, att_report, False, now=now):
                    would = "attest_integrity_failed"
                if full:
                    would += " + full re-hash"
            elif not equal and published_failure() and (would != "carry" or carry_deferral()):
                would = "attest_integrity_failed"            # the standing failure outranks the unequal outcomes (round 2 on 2c)
            return finish("dry_run", would=would, differing=(None if equal else diff), writes="nothing",
                          **({"authority_problems": problems, "adoptable": adoptable} if problems else {}))
        if equal:
            fields: Dict[str, Any] = {}
            if problems and adoptable and adopt_unrecorded:
                fields["adopted"] = adopt_current(remote, gen_dir, current, comp, mine, ev, source, lock_wait_s=lock_wait_s, now=now, log=log)
                ev = current_evidence(remote, gen_dir, current, theirs_meta, len(theirs))         # re-read: what stands beside `current` now
                verified, pub, problems = ev["verified"], ev["published"], list(ev["problems"])
                res.detail["generation_record"] = ev["record"]
            if problems:
                return finish(problems[0], evidence_why(problems, gen_dir, current, comp, adoptable), authority_problems=problems, adoptable=adoptable)
            fields.update(verified_utc=verified.get("verified_utc"), published_record=_published_digest(pub) if pub else None)
            # the report that verified the canonical's CURRENT listings against this generation (the live one, or the keyed report a
            # `stores attest` wrote): what `--full` re-hashes against, and where a re-attestation's integrity failure is read from
            verdict, att_report = listings_verdict(remote, gen_dir, current, verified, mine)
            attested_by_live = verdict == "attested" and _verified_matches(verified, mine)
            full_passed = False
            if full:
                att_path = None if (verdict != "attested" or attested_by_live) else attestation_json_path(f"{gen_dir}/MANIFEST-{current}.sha256", mine.listings_sha256)
                fr = full_rehash(root, dest, comp, remote, gen_dir, current, mine, pin=pin, log=log, now=now, workers=workers,
                                 attestation=att_path, listings_sha256=(mine.listings_sha256 if verdict == "attested" else None))
                if not fr["ok"]:
                    return finish("full_verify_failed", f"the published generation {current} under {gen_dir} does not re-hash to its manifest "
                                                        f"({fr['counts']}; report {'present' if fr['report_present'] else 'ABSENT'} at {fr['report_path']}): "
                                                        f"nothing renamed, `current` untouched -- the operator decides", full=fr, **fields)
                fields["full"] = fr
                full_passed = True
                # remembered: a re-attestation failure stamped before this pass is superseded on every later tick (attest_integrity)
                state["last_full_pass"] = {"content_sha8": current, "verified_utc": (fr.get("report") or {}).get("verified_utc"), "report_path": fr["report_path"]}
            # two sides that agree are verified only when no re-attestation that hashed every file and found some WRONG stands for this
            # generation: the record `stores attest` wrote, or the keyed report of the canonical's current listings alone -- red by name on
            # every tick, zero submissions, until a later full hash passes (round 1 on 2c: the reversed mirror returned verified over it)
            fail = attest_integrity(remote, gen_dir, current, mine, recorded, verdict, att_report, full_passed, now=now)
            state["attest_integrity_failed"] = fail
            if fail:
                return finish("attest_integrity_failed", _integrity_why(fail, f"{gen_dir}/{current}"), attest_failure=fail, listings_sha256=mine.listings_sha256, **fields)
            return finish("verified", **fields)
        # the two sides differ: the pin's manifest copy for the window, then the outcome by producer -- but a re-attestation INTEGRITY
        # FAILURE standing for the PUBLISHED generation (what the shards read) is named FIRST, red, whatever the grading host copy does (round 2
        # on 2c: a producer's unpushed edit took this branch and `unpushed_changes`, amber, exit 0, stood over a `current` whose attest had
        # hashed a corrupt file, the record still in the state). It stands until a full hash of that generation passes or a fresh, fully
        # verified generation supersedes `current`: the producer's push, or the carry when it is not deferred -- a deferred carry is this
        # red, never `deferred` (green)
        _save_generation_manifest(sdir, current, text, theirs_meta)
        counts = diff["counts"]
        unpushed = sum(counts.get(k, 0) for k in ("only_grading_host", "sha_or_size_differs"))
        differs = "frozen_differs" if frozen else ("carry" if comp.transitional_carry else "unpushed_changes")
        fail = published_failure()
        deferral = carry_deferral() if (fail and differs == "carry") else None
        if fail and (differs != "carry" or deferral):
            state["attest_integrity_failed"] = fail
            return finish("attest_integrity_failed", _integrity_why(fail, f"{gen_dir}/{current}") + _differs_note(differs, deferral, mine.content_sha8, counts, comp),
                          attest_failure=fail, differing=diff, differs_outcome=("deferred" if deferral else differs),
                          **({"unpushed_files": unpushed} if not frozen else {}), **({"deferral": deferral} if deferral else {}))
        if differs != "carry":
            state["attest_integrity_failed"] = None
        if frozen:
            return finish("frozen_differs", f"the frozen component's grading host copy ({mine.content_sha8}) differs from the published generation {current} "
                                            f"({counts}); a frozen store is never repaired by the mirror -- the operator decides which copy is the truth",
                          differing=diff)
        if comp.transitional_carry:
            return carry()
        return finish("unpushed_changes", f"the grading host copy ({mine.content_sha8}) differs from the published generation {current} ({counts}); the "
                                          f"producer {comp.producer} publishes it with `stores push {comp.slug}`; nothing copied",
                      differing=diff, unpushed_files=unpushed)
    except PublishLockBusy as exc:
        res.outcome = "lock_busy"
        res.detail["why"] = last_run["why"] = str(exc)
        return res
    except (StoresError, subprocess.SubprocessError, OSError) as exc:
        res.outcome = "error"
        res.detail["error"] = last_run["error"] = f"{type(exc).__name__}: {exc}"[:800]
        return res
    finally:
        res.elapsed_s = time.monotonic() - t0
        if not dry_run and not carried:
            update_state(root, comp.slug, {**state, "last_run": {"utc": now(), "outcome": res.outcome, **last_run}}, dest.name)


def pull_component(root: str, dest: Destination, comp: Component, runner=None, dry_run: bool = False, full: bool = False,
                   hasher: Callable[[str], str] = R.file_sha256, log: Callable[[str], None] = lambda s: None,
                   now: Callable[[], str] = R.utc_now, pin: Optional[PinnedTree] = None, workers: int = 16) -> MirrorResult:
    """`stores mirror` for a ``cluster_a`` component produced ON THE CLUSTER: the published generation is PULLED into the
    registry-owned mirror :func:`grading_host_mirror_path` (``<root>/stores/mirror/<slug>/``) -- :func:`pull_rsync_argv`, then every escaped
    name and link target restored to its true spelling (:func:`restore_escaped_names`) and every entry the generation does not name
    removed from that directory (:func:`prune_to_manifest`; nowhere else) -- and HASH-VERIFIED:
    the pulled tree's manifest (every pulled file read) must BE the generation's (``content_sha8`` = `current`, no entry differing, the
    links equal). ``pulled`` when the pull ran and verified, ``unchanged`` when the mirror already was the generation (the stat cache; a
    stat walk, nothing copied), ``pull_failed`` / ``pull_verify_failed`` (red) otherwise; the pulled tree is left in place either way
    (the next tick re-pulls what differs). The grading host manifest files are written beside the state (the pin's local copy; the stat
    snapshot lets `revalidate` check the pull mirror too). `--full` reads every file of the mirror and adds :func:`full_rehash`."""
    t0 = time.monotonic()
    gen_dir = dest.gen_dir(comp.slug)
    sdir = component_state_dir(root, comp.slug)
    remote = Remote(dest, runner)
    target = grading_host_mirror_path(root, comp)
    mirror_comp = Component(comp.slug, target, "cluster_a", origin=comp.origin, members=["."], exclude=[], adapters=list(comp.adapters), producer="cluster")
    res = MirrorResult(comp.slug, "error", detail={"authority": comp.authority, "producer": "cluster", "grading_host_mirror": target, "full": bool(full)})
    state: Dict[str, Any] = {"authority": comp.authority, "base": comp.base, "home_rel": comp.home_rel, "adapters": comp.adapters,
                             "revision": pin.revision if pin else None, "producer": "cluster", "grading_host_mirror": target}
    last_run: Dict[str, Any] = {}
    current: Optional[str] = None
    dangles = False                         # `current` names a tree that is not there: nothing is published, whatever the pointer says

    def finish(outcome: str, why: Optional[str] = None, **fields: Any) -> MirrorResult:
        """A dry run's outcome is always ``dry_run``; what a tick would conclude rides as ``would`` (and it writes nothing). A pointer that
        DANGLES publishes nothing (the closing pass on 2c, the same rule as the reversed mirror's): ``published`` stays unset so no window
        pins a generation whose tree is not there."""
        if dry_run and outcome != "dry_run":
            fields.setdefault("would", outcome)
            fields["writes"] = "nothing"
            outcome = "dry_run"
        res.outcome = outcome
        if why:
            res.detail["why"] = last_run["why"] = why
        res.detail.update(fields)
        res.published = None if dangles else current
        state["published"] = None if dangles else current
        state["generation"] = {"content_sha8": current, "state": outcome, "current_read": current, "grading_host_content_sha8": res.content_sha8,
                               **{k: v for k, v in fields.items() if k in ("verified_utc", "would", "pulled_now", "differing", "full", "pull",
                                                                           "authority_problems")}}
        return res

    def mirror_manifest(full_read: bool, write_cache: bool = True) -> Optional[Manifest]:
        if not os.path.isdir(target):
            return None
        try:
            return build_manifest(mirror_comp, sdir, hasher=hasher, full=full_read, write_cache=write_cache)
        except ManifestRefusal:
            return None                      # an empty or half-pulled directory: pulled afresh below

    try:
        current = remote.readlink(f"{gen_dir}/current")
        res.detail["current"] = current
        if not current:
            return finish("no_current", f"nothing is published under {gen_dir}: the cluster producer publishes inside its job "
                                        f"(`python3 -S -m sisteer.registry.stores push {comp.slug} --from DIR --stores-dir ...`); nothing to pull")
        if not remote.isdir(f"{gen_dir}/{current}"):
            dangles = True
            return finish("dangling", f"`current` -> {current} under {gen_dir} names no tree; nothing pulled and nothing published")
        got = _generation_manifest(remote, gen_dir, current)
        if got is None:
            return finish("no_manifest", f"`current` -> {current} under {gen_dir} has no MANIFEST-{current}.sha256 beside it; nothing pulled")
        text, theirs, theirs_meta = got
        theirs_links = _manifest_links(theirs_meta)
        res.n_files, res.bytes = len(theirs), sum(z for _s, z, _r in theirs)
        ev = current_evidence(remote, gen_dir, current, theirs_meta, len(theirs))
        verified, pub = ev["verified"], ev["published"]
        res.detail["generation_record"] = ev["record"]
        if ev["problems"]:
            # nothing is pulled from a `current` without its evidence: the cluster producer records and seals inside its job
            # BEFORE it switches, so a bare `current` there is a publisher mid-flight or a crash -- never adopted by the mirror
            return finish(ev["problems"][0], evidence_why(ev["problems"], gen_dir, current, comp, adoptable=False)
                          + f"; the producer publishes inside its job (`python3 -S -m sisteer.registry.stores push {comp.slug} ...`); nothing pulled, the mirror left as it is",
                          authority_problems=list(ev["problems"]))
        mine = mirror_manifest(full_read=full, write_cache=not dry_run)
        already = mine is not None and mine.content_sha8 == current and not any(_manifest_diff(mine, theirs, theirs_links)["counts"].values())
        if dry_run:
            return finish("dry_run", would=("unchanged" if already else "pull"), would_pull_from=f"{gen_dir}/{current}", would_pull_to=target, writes="nothing")
        pulled_now = False
        if not already:
            os.makedirs(target, exist_ok=True)
            rels = [rel for _s, _z, rel in theirs] + list(theirs_links)
            cleared = clear_escaped_tops(target, rels)          # before the rsync: nothing of ours stands at an escaped spelling
            log(f"[{comp.slug}] pull {len(theirs)} files <- {gen_dir}/{current} into {target}"
                + (f" ({cleared['cleared']} entr(y/ies) at an escaped spelling cleared first)" if cleared["cleared"] else ""))
            r = remote.run(pull_rsync_argv(dest, f"{gen_dir}/{current}", target), dest.rsync_timeout_s)
            stats = _rsync_stats(r.stdout or "")
            if r.returncode != 0:
                return finish("pull_failed", f"rsync from {gen_dir}/{current} into {target} failed (rc {r.returncode})", pull={"rsync": stats, **cleared},
                              rc=r.returncode, stderr=(r.stderr or "").strip()[-600:])
            restored = restore_escaped_names(target, theirs_links)
            pruned = prune_to_manifest(target, rels)
            res.detail["pull"] = {"rsync": stats, **cleared, **restored, **pruned}
            pulled_now = True
            mine = mirror_manifest(full_read=True)            # every pulled byte read: the verification
            if mine is None:
                return finish("pull_verify_failed", f"the pulled tree under {target} holds no regular file; the generation {current} names {len(theirs)}")
        assert mine is not None
        res.content_sha8, res.hashed, res.from_cache = mine.content_sha8, mine.hashed, mine.from_cache
        diff = _manifest_diff(mine, theirs, theirs_links)
        if mine.content_sha8 != current or any(diff["counts"].values()):
            return finish("pull_verify_failed", f"the pulled tree hashes to {mine.content_sha8}, the generation is {current} ({diff['counts']}); the "
                                                f"mirror is not the generation -- re-pulled next tick", differing=diff, pulled_now=pulled_now)
        state["latest_manifest"] = mine.summary(write_manifest_files(mine, sdir))
        fields: Dict[str, Any] = {"pulled_now": pulled_now, "verified_utc": verified.get("verified_utc"),
                                  "published_record": _published_digest(pub) if pub else None}
        if full:
            fr = full_rehash(root, dest, comp, remote, gen_dir, current, mine, pin=pin, log=log, now=now, workers=workers)
            if not fr["ok"]:
                return finish("full_verify_failed", f"the published generation {current} under {gen_dir} does not re-hash to its manifest ({fr['counts']}); "
                                                    f"nothing renamed, `current` untouched -- the operator decides", full=fr, **fields)
            fields["full"] = fr
        return finish("pulled" if pulled_now else "unchanged", **fields)
    except (StoresError, subprocess.SubprocessError, OSError) as exc:
        res.outcome = "error"
        res.detail["error"] = last_run["error"] = f"{type(exc).__name__}: {exc}"[:800]
        return res
    finally:
        res.elapsed_s = time.monotonic() - t0
        if not dry_run:
            update_state(root, comp.slug, {**state, "last_run": {"utc": now(), "outcome": res.outcome, **last_run}}, dest.name)


def run_mirror(root: str, dest: Destination, comp: Component, runner=None, dry_run: bool = False, full: bool = False,
               hasher: Callable[[str], str] = R.file_sha256, log: Callable[[str], None] = lambda s: None, now: Callable[[], str] = R.utc_now,
               pin: Optional[PinnedTree] = None, lock_wait_s: float = PUBLISH_LOCK_WAIT_S, adopt_unrecorded: bool = False, workers: int = 16,
               bound_s: Optional[float] = None) -> MirrorResult:
    """ONE component of `stores mirror`, dispatched on its authority: ``grading_host_copy`` -> :func:`mirror_component` (the carry
    to the destination; `--full` reads every file on the grading host and re-hashes an unchanged generation on the destination);
    ``cluster_a`` -> :func:`mirror_authority` (verify only / the transitional carry / the pull); ``cluster_a_origin`` -> ``skipped``
    (`stores check`). ``bound_s`` rides BOTH routes: the controller's bound on a verify watch is the same whether the carry is a
    ``grading_host_copy``'s or a ``cluster_a`` component's transitional one (before this it reached only the first)."""
    if comp.authority == "cluster_a":
        return mirror_authority(root, dest, comp, runner=runner, dry_run=dry_run, full=full, hasher=hasher, log=log, now=now, pin=pin,
                                lock_wait_s=lock_wait_s, adopt_unrecorded=adopt_unrecorded, workers=workers, bound_s=bound_s)
    return mirror_component(root, dest, comp, runner=runner, dry_run=dry_run, hasher=hasher, log=log, now=now, pin=pin, lock_wait_s=lock_wait_s,
                            adopt_unrecorded=adopt_unrecorded, workers=workers, full=full, bound_s=bound_s)


# --------------------------------------------------------------------------------------------
# stores push: the producer's publication (on the grading host, or inside the producer's own cluster job)
# --------------------------------------------------------------------------------------------


def _push_refusal(comp: Component) -> Optional[str]:
    """Why a component cannot be pushed from the grading host, by name; None when it can."""
    if comp.authority == "cluster_a_origin":
        return "authority cluster_a_origin: the original lives on the destination; nothing is pushed from the grading host (`stores check` compares)"
    if comp.authority == "cluster_a" and comp.producer == "cluster":
        return ("producer cluster: the producer publishes inside its own job (`python3 -S -m sisteer.registry.stores push --from DIR --stores-dir ...` "
                "on the node); the grading host holds a pull mirror, never the canonical copy")
    return None


def _age_s(stamp: Optional[str], now: str) -> Optional[float]:
    """Seconds from a registry UTC stamp to ``now`` (both ``_fmt_utc`` spellings); None when either does not parse."""
    from datetime import datetime, timezone
    try:
        a = datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        b = datetime.strptime(str(now), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (b - a).total_seconds()


def push_component(root: str, dest: Destination, comp: Component, runner=None, from_dir: Optional[str] = None, note: str = "",
                   session: Optional[str] = None, hasher: Callable[[str], str] = R.file_sha256, log: Callable[[str], None] = lambda s: None,
                   now: Callable[[], str] = R.utc_now, pin: Optional[PinnedTree] = None, lock_wait_s: float = PUBLISH_LOCK_WAIT_S,
                   adopt_unrecorded: bool = False) -> MirrorResult:
    """`registry stores push <slug>` from the grading host: :func:`mirror_component` for one component (``grading_host_copy``, or ``cluster_a`` with a
    grading-host-side or frozen producer) from its ``grading_host_mirror`` (``cluster_a``) or its base, ``--from DIR`` overriding either, plus the
    ``PUBLISHED-<sha8>.json`` record (:func:`published_record`, mode ``push``, written by the locked step). A ``min_publish_interval_s``
    component whose `current` was published more recently than that is ``deferred`` (nothing copied; the next push publishes).
    ``lock_wait_s`` bounds the publish step's wait for the destination lock (``lock_busy`` after it); ``adopt_unrecorded``
    (`--adopt-unrecorded`) is the explicit adoption of a `current` that has no publication record. Exit 0 only on :data:`PUSH_OK_OUTCOMES`."""
    why = _push_refusal(comp)
    if why:
        return MirrorResult(comp.slug, "refused", detail={"reason": why})
    source = os.path.abspath(os.path.expanduser(from_dir)) if from_dir else grading_host_mirror_path(root, comp)
    src = replace(comp, base=source)
    remote = Remote(dest, runner)
    if comp.min_publish_interval_s:
        current = remote.readlink(f"{dest.gen_dir(comp.slug)}/current")
        pub = remote.read_json(f"{dest.gen_dir(comp.slug)}/PUBLISHED-{current}.json") if current else {}
        age = _age_s(pub.get("published_utc"), now()) if pub else None
        if age is not None and age < comp.min_publish_interval_s:
            return MirrorResult(comp.slug, "deferred", content_sha8=current, published=current,
                                detail={"why": f"published {age:.0f} s ago; min_publish_interval_s {comp.min_publish_interval_s}: the next push publishes",
                                        "published_utc": pub.get("published_utc"), "min_publish_interval_s": comp.min_publish_interval_s})
    # the record names the DECLARED base beside the source it was pushed from (`--from DIR` may differ): mirror_component sees only the source
    meta = {"mode": "push", "note": note, "producer_session": session, "producer_host": socket.gethostname(), "source": source, "declared_base": comp.base}
    return mirror_component(root, dest, src, runner=runner, hasher=hasher, log=log, now=now, pin=pin, allow=("grading_host_copy", "cluster_a"), publish_meta=meta,
                            lock_wait_s=lock_wait_s, adopt_unrecorded=adopt_unrecorded)


def push_on_cluster(slug: str, from_dir: str, stores_dir: str, note: str = "", session: Optional[str] = None, workers: int = 16,
                    members: Sequence[str] = (), exclude: Sequence[str] = (), log: Callable[[str], None] = lambda s: None,
                    now: Callable[[], str] = R.utc_now, lock_wait_s: float = PUBLISH_LOCK_WAIT_S, adopt_unrecorded: bool = False) -> MirrorResult:
    """`python3 -S -m sisteer.registry.stores push <slug> --from DIR --stores-dir DIR` -- the producer's publication INSIDE its own
    cluster job (under its `srun`; stdlib only, no yaml, no credential, no ssh): the local destination (``$SHARED/registry/stores``),
    the copy into ``.incoming-<sha8>`` and the rename into place, the manifest, :func:`verify_generation` in this process, the directory
    seal, the guard, ``PUBLISHED-<sha8>.json`` (mode ``push_on_cluster``). The producer's stat cache and state live under
    ``<stores_dir>/push_state/<slug>/``. The whole flow runs under the slug's :class:`PublishLock` on the destination (``lock_wait_s``:
    how long to wait for another publisher of the slug; ``lock_busy`` after that), so two overlapping jobs of one producer publish in
    snapshot order and never share a temporary pointer."""
    stores_dir = os.path.abspath(stores_dir)
    dest = Destination("on-cluster", None, os.path.dirname(stores_dir), stores_rel=stores_dir)
    comp = Component(slug, os.path.abspath(os.path.expanduser(from_dir)), "cluster_a", origin="cluster_a", members=list(members or ["."]),
                     exclude=list(exclude or []), producer="cluster")
    root = os.path.join(stores_dir, "push_state", slug)
    job_id = os.environ.get("SLURM_JOB_ID")
    meta = {"mode": "push_on_cluster", "note": note, "producer_session": session, "producer_host": socket.gethostname(),
            "job_id": job_id, "source": comp.base}
    holder = f"{socket.gethostname()}:{os.getpid()} job={job_id or '-'} session={session or '-'}"
    return mirror_component(root, dest, comp, log=log, now=now, allow=("cluster_a",), publish_meta=meta, verify_inline=True, incoming=True,
                            workers=workers, lock_wait_s=lock_wait_s, lock_holder=holder, adopt_unrecorded=adopt_unrecorded)


# --------------------------------------------------------------------------------------------
# stores check-authority: every component's `current` in ONE batch -> state/store_authority.json
# --------------------------------------------------------------------------------------------


def _authority_batch_script(dest: Destination, comps: Sequence[Component]) -> str:
    """One shell script, run once on the destination: per component a ``C`` line (slug, `current`, its real path, exists, VERIFIED,
    PUBLISHED, SEALED present) and, when a PUBLISHED record exists, a ``P`` line carrying it on one line."""
    q = shlex.quote
    parts = ["set +e"]
    for c in comps:
        g = dest.gen_dir(c.slug)
        parts.append(
            f"S={q(c.slug)}; G={q(g)}; T=''; R=''; if test -L \"$G/current\"; then T=$(readlink \"$G/current\"); R=$(readlink -f \"$G/current\" 2>/dev/null); fi; "
            f"E=0; V=0; P=0; Z=0; if test -n \"$T\"; then test -d \"$G/$T\" && E=1; test -f \"$G/VERIFIED-$T.json\" && V=1; "
            f"test -f \"$G/PUBLISHED-$T.json\" && P=1; test -f \"$G/SEALED-$T.json\" && Z=1; fi; "
            f"printf 'C\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' \"$S\" \"$T\" \"$R\" \"$E\" \"$V\" \"$P\" \"$Z\"; "
            f"if test \"$P\" = 1; then printf 'P\\t%s\\t' \"$S\"; tr -d '\\n' < \"$G/PUBLISHED-$T.json\"; printf '\\n'; fi")
    return "; ".join(parts)


def check_authority(root: Optional[str], cfg: Config, dest: Destination, runner=None, now: Callable[[], str] = R.utc_now,
                    require_certified: bool = False) -> Dict[str, Any]:
    """Read, in ONE batch on the destination, every ``cluster_a`` / ``grading_host_copy`` component's `current` (its target, the real path, the
    target's existence) and whether VERIFIED, PUBLISHED and SEALED stand beside it; record ``<root>/state/store_authority.json``. A
    component's problems, by name: ``no_current`` (``cluster_a``: the canonical copy is the generation and none is published),
    ``dangling`` (`current` names a tree that is not there), ``unverified``, ``unsealed``, ``no_published_record`` (``cluster_a``),
    ``no_grading_host_mirror`` (``cluster_a``: the declared mirror is not a directory here). A ``grading_host_copy`` component with no `current` is
    ``never_published`` (its canonical copy is here; informational), one without a PUBLISHED record ``unrecorded`` (a problem).
    The report is ``ok`` iff no component has a problem; the CLI exits non-zero otherwise, naming every one.

    A component is CERTIFIED when its `current` exists and VERIFIED, SEALED and PUBLISHED all stand beside it (``certified`` per row,
    ``n_certified`` in the summary). ``require_certified`` (`--require-certified`; the one-time move's after-check, R2b) is the strict
    reading: a ``grading_host_copy`` component with no `current` is then the PROBLEM ``not_certified_never_published``, not state ok -- the
    default reading has no authority to check for such a component and calls it ok, which is how the R2b run's after-check read
    "0 problems; 49 ok" with five components never pushed."""
    remote = Remote(dest, runner)
    comps = [c for c in cfg.components.values() if c.authority in ("cluster_a", "grading_host_copy")]
    rows: Dict[str, Dict[str, Any]] = {}
    checked = now()
    today = checked[:10]
    if comps:
        r = remote._cmd(_authority_batch_script(dest, comps), what="check-authority batch")
        facts: Dict[str, Dict[str, Any]] = {}
        published: Dict[str, Dict[str, Any]] = {}
        for line in (r.stdout or "").splitlines():
            cells = line.split("\t")
            if cells[0] == "C" and len(cells) == 8:
                facts[cells[1]] = {"current": cells[2] or None, "real": cells[3] or None, "exists": cells[4] == "1", "verified": cells[5] == "1",
                                   "published_record": cells[6] == "1", "sealed": cells[7] == "1"}
            elif cells[0] == "P" and len(cells) >= 3:
                try:
                    doc = json.loads("\t".join(cells[2:]))
                    published[cells[1]] = doc if isinstance(doc, dict) else {}
                except ValueError:
                    published[cells[1]] = {"unreadable": True}
        for c in comps:
            f = facts.get(c.slug)
            if f is None:
                raise StoresError(f"check-authority: the batch returned nothing for component {c.slug}")
            pub = published.get(c.slug) or {}
            problems: List[str] = []
            notes: List[str] = []
            if not f["current"]:
                if c.authority == "cluster_a":
                    problems.append("no_current")
                elif require_certified:
                    problems.append("not_certified_never_published")        # the strict reading: nothing on the destination to certify
                else:
                    notes.append("never_published")
            else:
                if not f["exists"]:
                    problems.append("dangling")
                else:
                    if not f["verified"]:
                        problems.append("unverified")
                    if not f["sealed"]:
                        problems.append("unsealed")
                if not f["published_record"]:
                    # a `current` without its PUBLISHED record is a PROBLEM for every authority: every publisher records
                    # inside the lock now, so an unrecorded `current` is a publisher mid-flight, a crash, or a pre-rule generation the
                    # cutover's one-time `stores mirror --adopt-unrecorded` has not adopted yet
                    problems.append("no_published_record" if c.authority == "cluster_a" else "unrecorded")
            mirror_present: Optional[bool] = None
            if c.authority == "cluster_a":
                mirror_present = os.path.isdir(grading_host_mirror_path(root, c))
                if not mirror_present:
                    problems.append("no_grading_host_mirror")
            overdue = bool(c.transitional_carry and c.carry_deadline and c.carry_deadline < today)
            if overdue:
                notes.append("carry_overdue")
            certified = bool(f["current"]) and f["exists"] and f["verified"] and f["sealed"] and f["published_record"]
            rows[c.slug] = {"authority": c.authority, "producer": c.producer or None, "gen_dir": dest.gen_dir(c.slug), **f, "certified": certified,
                            "change_utc": pub.get("change_utc"), "published_utc": pub.get("published_utc"), "publish_mode": pub.get("mode"),
                            "grading_host_mirror": grading_host_mirror_path(root, c) if c.authority == "cluster_a" else c.grading_host_mirror,
                            "grading_host_mirror_present": mirror_present, "transitional_carry": c.transitional_carry,
                            "carry_deadline": c.carry_deadline, "carry_overdue": overdue, "problems": problems, "notes": notes,
                            "state": ("ok" if not problems else problems[0])}
    for c in cfg.components.values():
        if c.authority == "cluster_a_origin":
            rows[c.slug] = {"authority": c.authority, "state": "not_applicable", "problems": [], "notes": ["no generation: `stores check` compares the original"],
                            "authority_state": authority_state(root, c.slug) if root else None}
    problems = sorted(f"{slug}: {', '.join(row['problems'])}" for slug, row in rows.items() if row["problems"])
    n_checked = sum(1 for row in rows.values() if row.get("state") != "not_applicable")
    rec = {"schema": AUTHORITY_STATE_SCHEMA, "checked_utc": checked, "destination": dest.name, "stores_dir": dest.stores_dir, "config": cfg.path,
           "ok": not problems, "n_components": len(rows), "n_problems": len(problems), "problems": problems,
           "require_certified": bool(require_certified), "n_checked": n_checked,
           "n_certified": sum(1 for row in rows.values() if row.get("certified")),
           "counts": {k: sum(1 for row in rows.values() if row["state"] == k) for k in sorted({row["state"] for row in rows.values()})},
           "components": dict(sorted(rows.items()))}
    if root:
        _write_json(os.path.join(root, "state", AUTHORITY_STATE_FILE), rec)
    return rec


def format_authority(rec: Mapping[str, Any]) -> str:
    lines = [f"store authority on {rec.get('destination')} ({rec.get('stores_dir')}), checked {rec.get('checked_utc')}: "
             f"{rec.get('n_components')} components, {rec.get('n_problems')} with a problem"
             + (f"; certified {rec['n_certified']} of {rec.get('n_checked', rec.get('n_components'))} (current exists, VERIFIED, SEALED, PUBLISHED)"
                if "n_certified" in rec else "")
             + ("; strict: never-published is a problem" if rec.get("require_certified") else ""),
             "slug | authority | current | exists | verified | sealed | published record | change_utc | state | notes"]
    for slug, row in (rec.get("components") or {}).items():
        if row.get("state") == "not_applicable":
            lines.append(f"{slug} | {row['authority']} | - | - | - | - | - | - | not_applicable | {'; '.join(row.get('notes') or [])}")
            continue
        lines.append(" | ".join([slug, row["authority"], row.get("current") or "-", *("yes" if row.get(k) else "no" for k in ("exists", "verified", "sealed", "published_record")),
                                 row.get("change_utc") or "-", ", ".join(row["problems"]) if row["problems"] else "ok", "; ".join(row.get("notes") or [])]))
    lines.append("check-authority OK" if rec.get("ok") else "check-authority NOT OK: " + "; ".join(rec.get("problems") or []))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# stores verify-authority: the one-time verified move of a component
# --------------------------------------------------------------------------------------------

MOVE_SCHEMA = "store_move/1"


def verify_authority(root: str, dest: Destination, comp: Component, out_dir: str, runner=None, hasher: Callable[[str], str] = R.file_sha256,
                     log: Callable[[str], None] = lambda s: None, now: Callable[[], str] = R.utc_now, pin: Optional[PinnedTree] = None) -> Dict[str, Any]:
    """Is the component's CURRENT generation on the destination the grading host copy, byte for byte? (i) a fresh grading host manifest of the
    ``grading_host_mirror`` (``cluster_a``) or the base -> ``<out>/<slug>.grading_host.sha256``; (ii) the generation's manifest ->
    ``<out>/<slug>.cluster_a.sha256``; the two diffed by :func:`_entries_diff` (and the attested symlinks compared) -- unequal ->
    ``<out>/<slug>.diff.json``, state ``unequal`` (push first; no job is spent hashing a generation that is not this content); (iii) equal
    -> ``store_verify.sbatch verify`` over the CURRENT generation (the pinned verifier; the report to ``<stores>/move/<slug>/``, never over
    the live VERIFIED file) proves the generation's bytes are its manifest's; (iv) then the generation is ADOPTED -- the directory seal
    and ``SEALED-<sha8>.json`` when missing, ``PUBLISHED-<sha8>.json`` (mode ``move``; ``change_utc`` = the grading host copy's latest mtime)
    when missing -- and ``<out>/<slug>.verified.json`` is written: the evidence the yaml's flip to ``authority: cluster_a`` rests on.

    The certificate attests the PAIRING AT THE MOMENT IT IS FILED, not at the start: the verify job can run for a long time,
    and nothing on the grading host stops an on-cluster producer from publishing a new generation, or the grading host mirror from moving, meanwhile.
    So `current` is re-read after the job (before anything is adopted) and, immediately before the certificate is written, `current` is
    re-read AND the source is re-manifested (through the stat cache) against the hashes captured at the start; a difference refuses by
    name -- ``current_moved`` / ``source_changed`` (``<out>/<slug>.diff.json``, a stale certificate retired) -- and the rechecks ride in
    the certificate."""
    t0 = time.monotonic()
    rec: Dict[str, Any] = {"schema": MOVE_SCHEMA, "slug": comp.slug, "authority": comp.authority, "destination": dest.name, "checked_utc": now(),
                           "state": "error", "out": os.path.abspath(out_dir), "revision": pin.revision if pin else None}
    if comp.authority == "cluster_a_origin":
        rec.update(state="not_applicable", why="authority cluster_a_origin: no generation; `stores check` compares the original")
        return rec
    os.makedirs(out_dir, exist_ok=True)
    remote = Remote(dest, runner)
    gen_dir = dest.gen_dir(comp.slug)
    try:
        source = grading_host_mirror_path(root, comp)
        mine = build_manifest(replace(comp, base=source), component_state_dir(root, comp.slug), hasher=hasher)
        grading_host_lines = os.path.join(out_dir, f"{comp.slug}.grading_host.sha256")
        _write_text(grading_host_lines, mine.text())
        _write_json(os.path.join(out_dir, f"{comp.slug}.grading_host.json"), mine.meta({"source": source}))
        rec["grading_host"] = mine.summary(grading_host_lines)
        current = remote.readlink(f"{gen_dir}/current")
        rec["current"] = current
        if not current:
            rec.update(state="no_current", why=f"nothing is published under {gen_dir}; `stores push {comp.slug}` first")
            return rec
        theirs_text = remote.read(f"{gen_dir}/MANIFEST-{current}.sha256")
        if theirs_text is None:
            rec.update(state="no_manifest", why=f"`current` -> {current} has no MANIFEST-{current}.sha256 beside it")
            return rec
        cluster_a_lines = os.path.join(out_dir, f"{comp.slug}.cluster_a.sha256")
        _write_text(cluster_a_lines, theirs_text)
        theirs = parse_manifest_lines(theirs_text, cluster_a_lines)
        theirs_meta = remote.read_json(f"{gen_dir}/MANIFEST-{current}.json")
        theirs_links = {str(k): str(v) for k, v in (theirs_meta.get("symlinks") or {}).items()} if isinstance(theirs_meta.get("symlinks"), dict) else {}
        rec["cluster_a"] = {"content_sha8": current, "n_files": len(theirs), "bytes": sum(z for _s, z, _r in theirs), "files": cluster_a_lines,
                            "n_symlinks": len(theirs_links)}
        rec["differing"] = _entries_diff(mine.entries, theirs)
        link_diff = sorted(set(dict(mine.links).items()) ^ set(theirs_links.items()))
        rec["differing"]["symlinks"] = [list(t) for t in link_diff[:LIST_CAP]]
        rec["differing"]["counts"]["symlinks"] = len(link_diff)
        equal = mine.content_sha8 == current and not any(rec["differing"]["counts"].values())
        if not equal:
            rec.update(state="unequal", why=(f"the grading host copy hashes to {mine.content_sha8}, `current` is {current}: the generation is not the grading host "
                                              f"content; `stores push {comp.slug}` first, then verify again"))
            _write_json(os.path.join(out_dir, f"{comp.slug}.diff.json"), rec)
            return rec
        # (iii) the generation's bytes are its manifest's: the pinned verifier over the CURRENT generation, its report in the move dir
        move_dir = f"{dest.stores_dir}/move/{comp.slug}"
        remote.mkdir(move_dir)
        out_report = f"{move_dir}/VERIFIED-{current}-{_stamp(now())}.json"
        job_id, tlim = remote.submit(f"store-verify-{comp.slug}", ["verify", f"{gen_dir}/{current}", f"{gen_dir}/MANIFEST-{current}.sha256", out_report,
                                                                    str(dest.verify_cpus)], mine.bytes, mine.n_files, measured_rate(root, comp.slug, dest),
                                     script=pin.sbatch_path if pin else None)
        rec["job_id"], rec["time_limit_s"], rec["report_path"] = job_id, tlim, out_report
        log(f"[{comp.slug}] verify-authority job {job_id} over {gen_dir}/{current} (--time {slurm_time(tlim)}); watching")
        watched = remote.watch(job_id)
        rec["watch"] = watched
        if watched["state"] != "DONE":
            raise StoresError(f"verify job {job_id}: {watched.get('terminal') or watched['state']}")
        report = remote.read_json(out_report)
        if not _verified_matches(report, mine):
            counts = {k: len(report.get(k) or []) for k in ("missing", "corrupt", "extra")}
            rec.update(state="verify_failed", report={"ok": report.get("ok"), **counts, "first": {k: (report.get(k) or [])[:10] for k in counts}},
                       why="the generation's bytes do not match its manifest (or the report is absent); nothing adopted")
            _write_json(os.path.join(out_dir, f"{comp.slug}.diff.json"), rec)
            return rec
        rec["report"] = {k: report.get(k) for k in ("hashed", "hashed_bytes", "elapsed_s", "rate_mb_s", "host", "verified_utc", "job_id", "sealed")}

        def recheck(stage: str, source_too: bool) -> Optional[str]:
            """Is the pairing the job verified still the pairing NOW? `current` re-read; with ``source_too`` the grading host source re-manifested
            (the stat cache: a stat walk, only moved files read) against the hashes captured at the start. The refusal by name, or None."""
            again_current = remote.readlink(f"{gen_dir}/current")
            check: Dict[str, Any] = {"stage": stage, "utc": now(), "current": again_current, "current_at_start": current}
            again: Optional[Manifest] = None
            if source_too:
                again = build_manifest(replace(comp, base=source), component_state_dir(root, comp.slug), hasher=hasher)
                check.update(grading_host_content_sha8=again.content_sha8, grading_host_content_sha8_at_start=mine.content_sha8, grading_host_hashed=again.hashed,
                             grading_host_name_size_equal=again.name_size_sha256 == mine.name_size_sha256)
            rec.setdefault("rechecks", []).append(check)
            if again_current != current:
                rec["why"] = (f"`current` under {gen_dir} moved from {current} to {again_current or 'absent'} while the verify job ran ({stage}): "
                              f"the certificate would attest a pairing that no longer stands; nothing adopted or certified -- run verify-authority again")
                return "current_moved"
            if again is not None and (again.content_sha256, again.name_size_sha256) != (mine.content_sha256, mine.name_size_sha256):
                rec["why"] = (f"the grading host copy at {source} changed while the verify job ran ({stage}): {mine.content_sha8} -> {again.content_sha8}; the "
                              f"generation {current} is no longer its content -- `stores push {comp.slug}` first, then verify again")
                return "source_changed"
            return None

        refused = recheck("after_verify", source_too=False)
        if refused:
            rec["state"] = refused
            _write_json(os.path.join(out_dir, f"{comp.slug}.diff.json"), rec)
            return rec
        # (iv) adopt: the directory seal and the PUBLISHED record, each only when missing
        sealed = remote.read_json(f"{gen_dir}/SEALED-{current}.json")
        if not sealed or remote.unsealed_directories(f"{gen_dir}/{current}") != 0:
            sealed = seal_generation(remote, gen_dir, current, now=now, by=f"verify-authority {getpass.getuser()}@{socket.gethostname()}")
            rec["sealed_now"] = True
        rec["sealed"] = {k: sealed.get(k) for k in ("n_dirs", "sealed_utc", "first_sealed_utc", "host")}
        pub = remote.read_json(f"{gen_dir}/PUBLISHED-{current}.json")
        if not pub:
            # the move's adoption record, written under the slug's publish lock for the generation that IS current (never a plain write)
            pub = published_record(gen_dir, current, comp, mine, None, [rel for _s, _z, rel in mine.entries],
                                   {"mode": "move", "note": "one-time verified move (registry-slurm-only plan §1.2 (iii))", "producer_host": socket.gethostname()}, now=now)
            adopt = remote.record_published(gen_dir, current, record_text(pub), lock_holder=f"verify-authority {getpass.getuser()}@{socket.gethostname()}:{os.getpid()}", utc=now())
            if not adopt["written"]:
                pub = remote.read_json(f"{gen_dir}/PUBLISHED-{current}.json")
            rec["published_now"] = True
        rec["published_record"] = _published_digest(pub)
        rec["unsealed_after"] = remote.unsealed_directories(f"{gen_dir}/{current}")
        if rec["unsealed_after"]:
            raise StoresError(f"{rec['unsealed_after']} director(y/ies) under {gen_dir}/{current} are not {DIR_SEAL_MODE:04o} after the seal")
        # the certificate is filed only if the pairing still stands NOW: `current` and the source re-read right before the write
        refused = recheck("before_certificate", source_too=True)
        if refused:
            rec["state"] = refused
            _write_json(os.path.join(out_dir, f"{comp.slug}.diff.json"), rec)
            return rec
        rec["state"] = "verified"
        rec["verified_utc"] = now()
        _write_json(os.path.join(out_dir, f"{comp.slug}.verified.json"), rec)
        return rec
    except (StoresError, subprocess.SubprocessError, OSError) as exc:
        rec["state"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:800]
        return rec
    finally:
        rec["elapsed_s"] = round(time.monotonic() - t0, 3)
        stale = os.path.join(out_dir, f"{comp.slug}.verified.json")
        if rec["state"] != "verified" and os.path.isfile(stale):
            # the evidence the yaml flip rests on never outlives a later outcome that is not `verified`
            os.replace(stale, f"{stale}.stale-{_stamp(now())}")
            rec["stale_verified_renamed"] = True
        update_state(root, comp.slug, {"authority": comp.authority, "base": comp.base, "home_rel": comp.home_rel, "adapters": comp.adapters,
                                       "move": {k: rec.get(k) for k in ("state", "current", "checked_utc", "job_id", "error", "why")},
                                       "last_run": {"utc": now(), "outcome": f"{VERIFY_AUTHORITY_OUTCOME_PREFIX}{rec['state']}"}}, dest.name)


# --------------------------------------------------------------------------------------------
# stores check: a cluster_a_origin component -> authority.json (match | mismatch)
# --------------------------------------------------------------------------------------------


def _entries_diff(a: Sequence[Tuple[str, int, str]], b: Sequence[Tuple[str, int, str]]) -> Dict[str, Any]:
    da = {rel: (sha, size) for sha, size, rel in a}
    db = {rel: (sha, size) for sha, size, rel in b}
    only_a = sorted(set(da) - set(db), key=_by_bytes)
    only_b = sorted(set(db) - set(da), key=_by_bytes)
    differ = sorted((rel for rel in set(da) & set(db) if da[rel] != db[rel]), key=_by_bytes)
    return {"only_grading_host": only_a[:LIST_CAP], "only_original": only_b[:LIST_CAP], "sha_or_size_differs": differ[:LIST_CAP],
            "counts": {"only_grading_host": len(only_a), "only_original": len(only_b), "sha_or_size_differs": len(differ)}}


def _authority_summary(rec: Mapping[str, Any], keys: Sequence[str] = ("state", "checked_utc", "job_id", "error")) -> Dict[str, Any]:
    """The digest of an ``authority.json`` record that the state file, :func:`authority_state` and the CLI print."""
    return {k: rec.get(k) for k in keys} | {"counts": (rec.get("differing") or {}).get("counts")}


def check_component(root: str, dest: Destination, comp: Component, runner=None, hasher: Callable[[str], str] = R.file_sha256,
                    log: Callable[[str], None] = lambda s: None, now: Callable[[], str] = R.utc_now,
                    pin: Optional[PinnedTree] = None) -> Dict[str, Any]:
    """Compare the grading host mirror's manifest with a manifest of the cluster A original built by
    ``store_verify.sbatch manifest-only`` (the same walk rules), and record the authority state."""
    t0 = time.monotonic()
    if comp.authority != "cluster_a_origin":
        return {"slug": comp.slug, "state": "not_applicable", "why": f"authority {comp.authority}: carried as a generation by `stores mirror`"}
    sdir = component_state_dir(root, comp.slug)
    remote = Remote(dest, runner)
    original = dest.resolve(comp.original or "")
    record: Dict[str, Any] = {"schema": AUTHORITY_SCHEMA, "slug": comp.slug, "authority": comp.authority, "base": comp.base, "home_rel": comp.home_rel,
                              "original": original, "destination": dest.name, "checked_utc": now(), "state": "error",
                              "revision": pin.revision if pin else None}
    try:
        mine = build_manifest(comp, sdir, hasher=hasher)
        record["grading_host"] = mine.summary(write_manifest_files(mine, sdir))
        stamp = _stamp(now())
        out_prefix = f"{dest.origin_manifest_dir(comp.slug)}/MANIFEST-{stamp}"
        remote.mkdir(dest.origin_manifest_dir(comp.slug))
        script_args = ["manifest-only", original, out_prefix, str(dest.verify_cpus)]
        for m in comp.members:
            script_args += ["--member", m]
        for e in comp.exclude:
            script_args += ["--exclude", e]
        script_args += ["--grading-host-root", comp.root_spelling]      # the mirror's spelling, recorded in the cluster-built manifest
        job_id, _tlim = remote.submit(f"store-manifest-{comp.slug}", script_args, mine.bytes, mine.n_files, measured_rate(root, comp.slug, dest),
                                      script=pin.sbatch_path if pin else None)
        record["job_id"] = job_id
        log(f"[{comp.slug}] manifest-only job {job_id} over {original}; watching")
        watched = remote.watch(job_id)
        record["watch"] = watched
        if watched["state"] != "DONE":
            raise StoresError(f"manifest-only job {job_id}: {watched.get('terminal') or watched['state']}")
        theirs_meta = remote.read_json(out_prefix + ".json")
        theirs_text = remote.read(out_prefix + ".sha256")
        if not theirs_meta or theirs_text is None:
            raise StoresError(f"manifest-only job {job_id} left no {out_prefix}.json / .sha256")
        theirs_lines = os.path.join(sdir, f"ORIGINAL-{stamp}.sha256")          # kept beside the grading host manifest as evidence
        _write_text(theirs_lines, theirs_text)
        theirs = read_manifest_lines(theirs_lines)
        record["original_manifest"] = {k: theirs_meta.get(k) for k in ("content_sha8", "content_sha256", "name_size_sha256", "built_utc",
                                                                        "built_host", "n_files", "bytes", "job_id", "root", "home_rel")} | {"files": theirs_lines}
        equal = (theirs_meta.get("content_sha256") == mine.content_sha256 and theirs_meta.get("name_size_sha256") == mine.name_size_sha256)
        record["state"] = "match" if equal else "mismatch"
        record["differing"] = _entries_diff(mine.entries, theirs)
        if not equal:
            record["why"] = ("the grading host mirror's two-manifest hashes differ from the original's; the component's in-place cluster route is "
                             "refused until they agree (the adapter reads on the grading host)")
    except (StoresError, subprocess.SubprocessError, OSError) as exc:
        record["state"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:800]
    record["elapsed_s"] = round(time.monotonic() - t0, 3)
    _write_json(os.path.join(sdir, AUTHORITY_FILE), record)
    update_state(root, comp.slug, {"authority": comp.authority, "base": comp.base, "home_rel": comp.home_rel, "adapters": comp.adapters,
                                   "original": original, "authority_state": _authority_summary(record), "latest_manifest": record.get("grading_host"),
                                   "last_run": {"utc": now(), "outcome": f"{CHECK_OUTCOME_PREFIX}{record['state']}"}}, dest.name)
    return record


def authority_state(root: str, slug: str) -> Dict[str, Any]:
    """The recorded authority state of a component (``unchecked`` when `stores check` never ran)."""
    rec = _load_json(os.path.join(component_state_dir(root, slug), AUTHORITY_FILE))
    return _authority_summary(rec, ("state", "checked_utc", "job_id", "error", "original")) if rec else {"state": "unchecked"}


# --------------------------------------------------------------------------------------------
# admission re-validation (the cluster-read window's helper)
# --------------------------------------------------------------------------------------------


@dataclass
class Revalidation:
    slug: str
    generation: str
    refusal: Optional[str] = None       # None = admitted; else no_component | no_manifest | no_stat_snapshot | no_attestation |
                                        # file_set_changed | listing_changed | content_changed | changed_during_hash
    reason: str = ""
    n_files: int = 0
    moved: int = 0
    moved_unchanged: int = 0
    sampled: int = 0
    hashed: int = 0
    mismatched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    elapsed_s: float = 0.0
    skipped: bool = False               # a ``cluster_a`` component is admitted by the per-operation attestation on the cluster, not here
    n_directories: Optional[int] = None                 # attested directory listings compared with the canonical's (None: none attested)
    listing_mismatched: List[str] = field(default_factory=list)
    listings_sha256: Optional[str] = None               # the attestation compared against (the pin's; the plain .json's when none is named)

    @property
    def ok(self) -> bool:
        return self.refusal is None

    def as_dict(self) -> Dict[str, Any]:
        d = {"slug": self.slug, "generation": self.generation, "ok": self.ok, **asdict(self)}
        d.update(mismatched=self.mismatched[:LIST_CAP], missing=self.missing[:LIST_CAP], extra=self.extra[:LIST_CAP],
                 listing_mismatched=self.listing_mismatched[:LIST_CAP], elapsed_s=round(self.elapsed_s, 3))
        return d


def revalidate(slug: str, generation: str, root: Optional[str] = None, cfg: Optional[Config] = None,
               sample_fraction: float = SAMPLE_FRACTION, sample_min: int = SAMPLE_MIN, seed: Optional[int] = None,
               hasher: Callable[[str], str] = R.file_sha256, listings_sha256: Optional[str] = None) -> Revalidation:
    """Is the grading host canonical copy of ``slug`` still the content the pinned ``generation`` (a content_sha8) carries?
    A full stat walk against the generation's stat snapshot; every file whose stat moved is hashed, plus a seeded sample
    of ``max(sample_min, ceil(sample_fraction * n))`` files; any missing / extra file or any sha that differs from the
    manifest REFUSES by name. A moved stat over unchanged bytes (a touch) is admitted and counted. ``root`` and ``cfg``
    default to the registry root (``$REGISTRY_ROOT``) and the declared configuration, the two-argument call the window makes.
    The canonical's directory listings are compared with the attestation the PIN names (``listings_sha256``: the one the generation
    was verified against, recorded by the publish and carried in the window's plan) -- never with the latest attestation of that
    content: a rebuild of the same files over other listings adds ``MANIFEST-<sha8>.<lsha8>.json`` and leaves the pin's untouched,
    so a window pinned to the old attestation refuses ``listing_changed`` when the canonical left it, and a pin naming an attestation
    this grading host never wrote refuses ``no_attestation``. With none named (an older pin), the plain ``.json`` -- the FIRST attestation."""
    t0 = time.monotonic()
    root = root or os.path.abspath(os.environ.get("REGISTRY_ROOT", DEFAULT_ROOT))
    gen = generation[:8]
    rv = Revalidation(slug, gen, seed=int(gen, 16) if seed is None else seed)

    def refuse(name: str, reason: str) -> Revalidation:
        rv.refusal, rv.reason, rv.elapsed_s = name, reason, time.monotonic() - t0
        return rv

    cfg = cfg or load_config(root=root)
    comp = cfg.components.get(slug)
    if comp is None:
        return refuse("no_component", f"{slug} is not declared in {cfg.path}")
    sdir = component_state_dir(root, slug)
    mpath = os.path.join(sdir, f"MANIFEST-{gen}.sha256")
    spath = os.path.join(sdir, f"MANIFEST-{gen}.stat.json")
    if not os.path.isfile(mpath):
        return refuse("no_manifest", f"no {mpath}: the generation was never built from this grading host state")
    snap_doc = _load_json(spath)
    if not snap_doc or not isinstance(snap_doc.get("files"), dict):
        return refuse("no_stat_snapshot", f"no stat snapshot {spath}; the moved-file test cannot run")
    entries = read_manifest_lines(mpath)
    want = {rel: (sha, size) for sha, size, rel in entries}
    want_links = read_manifest_symlinks(mpath)
    rv.n_files = len(entries)
    try:
        walk = walk_component(comp)
    except ManifestRefusal as exc:
        return refuse("file_set_changed", str(exc))
    moved, rv.missing, rv.extra = stat_changes(walk, snap_doc["files"])
    if rv.missing or rv.extra:
        return refuse("file_set_changed", f"{len(rv.missing)} manifest entr(y/ies) missing, {len(rv.extra)} unlisted entr(y/ies) present under {comp.base}")
    attestation = read_manifest_attestation(mpath, listings_sha256)
    if listings_sha256 and attestation is None:
        return refuse("no_attestation", f"no attestation of generation {gen} with listings {listings_sha256[:8]} under {sdir}; the pin names one this grading host never wrote")
    rv.listings_sha256 = (attestation or {}).get("listings_sha256")
    want_dirs = attested_directories(attestation)
    if want_dirs is not None:
        # the canonical's directory listings against the PIN'S attestation (MANIFEST_RULES["directories"]): an empty directory that landed or
        # left since that push is a change the pinned generation does not carry (its file content is unchanged, so no new generation was
        # published: `listings_changed`; or the same content was re-published under another attestation), and the readers that state
        # listings must not take -- or keep -- the cluster route against it
        rv.n_directories = len(want_dirs)
        now_dirs = {rel: walk.dirs.get(rel) for rel in want_dirs}
        rv.listing_mismatched = sorted((rel for rel, ents in now_dirs.items() if ents is None or normalize_entries(ents) != normalize_entries(want_dirs[rel])), key=_by_bytes)
        rv.listing_mismatched += sorted((rel for rel in walk.dirs if rel not in want_dirs), key=_by_bytes)     # a directory the generation never attested
        if rv.listing_mismatched:
            return refuse("listing_changed", f"{len(rv.listing_mismatched)} director(y/ies) under {comp.base} list other than generation {gen} attests (first: {rv.listing_mismatched[0]!r})")
    moved_set = set(moved)
    rv.moved = len(moved)
    # every symlink's literal target must still be the attested one (a retarget moves its stat too, but the target is the identity)
    links = {rel: target for rel, _p, _st, target in walk.links}
    rv.mismatched += sorted((rel for rel, target in links.items() if want_links.get(rel) != target), key=_by_bytes)
    rv.moved_unchanged += sum(1 for rel in moved if rel in links and rel not in rv.mismatched)
    k = min(len(entries), max(sample_min, math.ceil(sample_fraction * len(entries))))
    pool = [rel for rel in want if rel not in moved_set]
    sample = random.Random(rv.seed).sample(pool, min(k, len(pool)))
    rv.sampled = len(sample)
    now_by_rel = {rel: (p, st) for rel, p, st in walk.files}
    for rel in [r for r in moved if r in now_by_rel] + sample:
        p, st = now_by_rel[rel]
        sha = hasher(p)
        rv.hashed += 1
        if stat_key(os.lstat(p)) != stat_key(st):
            return refuse("changed_during_hash", f"{rel} changed while being hashed; the canonical copy is being written")
        if (sha, st.st_size) != want[rel]:
            rv.mismatched.append(rel)
        elif rel in moved_set:
            rv.moved_unchanged += 1
    if rv.mismatched:
        return refuse("content_changed", f"{len(rv.mismatched)} entr(y/ies) differ from generation {gen}'s manifest (first: {rv.mismatched[0]})")
    rv.elapsed_s = time.monotonic() - t0
    return rv


def adapter_admission(cfg: Config, adapter: str, pins: Mapping[str, str], root: Optional[str] = None,
                      attestations: Optional[Mapping[str, Optional[str]]] = None, **kw: Any) -> Dict[str, Revalidation]:
    """Every component the adapter declares, re-validated against its pinned generation (``pins = {slug: content_sha8}``) and the
    attestation the pin names (``attestations = {slug: listings_sha256 | None}``; :func:`revalidate`): the adapter takes the cluster
    route only when EVERY one is admitted, so a stale or absent input under any of its components -- relu_paired's controls index
    under ``sampled``, say -- refuses the whole adapter by the component's name. A component with no pin is refused as
    ``no_manifest`` (nothing was pinned to read)."""
    if adapter not in cfg.adapters:
        raise StoresError(f"adapter {adapter!r} is not declared in {cfg.path}")
    out: Dict[str, Revalidation] = {}
    for slug in cfg.adapters[adapter].components:
        if slug not in pins:
            out[slug] = Revalidation(slug, "", "no_manifest", f"{adapter}: no generation pinned for component {slug}")
            continue
        comp = cfg.components.get(slug)
        if comp is not None and comp.authority == "cluster_a":
            # C5: the published generation IS the canonical copy of a ``cluster_a`` component; the grading host holds a mirror that may
            # lag or lead the producer's push (``unpushed_changes`` is the reversed mirror's amber, never a reason to refuse the route or to
            # quarantine what the shards read from the attested generation). Its admission is the per-operation attestation on the cluster,
            # so both the grading host passes -- the plan's and the post-run revalidation -- skip it BY NAME, and the plan records that they did.
            out[slug] = Revalidation(slug, pins[slug][:8], None, "authority cluster_a: the generation is the canonical copy; admitted by the "
                                     "per-operation attestation on the cluster (attestation.json), not by a grading host revalidation", skipped=True)
            continue
        out[slug] = revalidate(slug, pins[slug], root=root, cfg=cfg, listings_sha256=(attestations or {}).get(slug), **kw)
    return out


# --------------------------------------------------------------------------------------------
# stores coverage: every registered adapter's DECLARED roots against the components -- no discovery, no store opened
# --------------------------------------------------------------------------------------------

COVERAGE_SCHEMA = "store_coverage/1"
#: the state of one declared root: a root under the home lies inside a component's declared members
#: (``covered``), or above them (``restricted``: the component holds only those members of it -- the shard's own discovery is the exact
#: check), or under no component at all (``UNCOVERED``); a root outside the home is ``retired`` when the adapter's own ``inputs_retired``
#: declaration drops everything under it, ``volatile`` when the configuration names it so with a reason, else ``UNCOVERED``; a ``None``
#: root is ``undeclared`` (the adapter's declaration is broken); every root of a ``cluster: refused`` adapter is ``refused``
COVERED, RESTRICTED, RETIRED, VOLATILE, REFUSED_ROOT, UNCOVERED, UNDECLARED = "covered", "restricted", "retired", "volatile", "refused", "UNCOVERED", "undeclared"
ROOT_STATES = (COVERED, RESTRICTED, RETIRED, VOLATILE, REFUSED_ROOT, UNCOVERED, UNDECLARED)


def _segments(rel: str) -> List[str]:
    return [x for x in os.path.normpath(rel).split("/") if x not in ("", ".")]


def member_match(rel: str, pattern: str) -> Optional[str]:
    """How a declared member (a glob relative to the base; ``.`` = the whole base) relates to a base-relative root path, by path algebra
    alone (segment-wise ``fnmatch``; nothing is listed): :data:`COVERED` when the root lies at or below the member, :data:`RESTRICTED`
    when the member lies strictly below the root (the component holds only that part of the root), None when they diverge."""
    want = _segments(pattern)
    have = _segments(rel)
    if not want:
        return COVERED
    n = min(len(want), len(have))
    if any(not fnmatch.fnmatchcase(have[i], want[i]) for i in range(n)):
        return None
    return COVERED if len(have) >= len(want) else RESTRICTED


def root_coverage(root: str, comp: Component) -> Optional[Tuple[str, str]]:
    """Does ``comp`` hold ``root``? ``(state, detail)`` -- :data:`COVERED` / :data:`RESTRICTED` with the member that decided it -- or None
    when the root lies outside the base, is excluded by the component, or diverges from every member."""
    base = comp.base.rstrip("/") or "/"
    if root == base:
        rel = ""
    elif root.startswith(base + "/"):
        rel = root[len(base) + 1:]
    else:
        return None
    anchored = comp.anchored_excludes()
    names = comp.name_excludes()
    parts = _segments(rel)
    for i in range(1, len(parts) + 1):
        if "/".join(parts[:i]) in anchored:
            return None
    if any(fnmatch.fnmatchcase(part, pat) for part in parts for pat in names):
        return None
    best: Optional[Tuple[str, str]] = None
    for m in comp.members:
        state = member_match(rel, m)
        if state == COVERED:
            return (COVERED, m)
        if state == RESTRICTED and best is None:
            best = (RESTRICTED, m)
    return best


def _retired_globs(adapter) -> Tuple[str, ...]:
    """The adapter's own ``inputs_retired`` declaration, read without a comparison group (a callable is asked with the empty group; one
    that cannot answer declares nothing here) -- never a discovery."""
    from .adapters.base import inputs_retired_of
    try:
        return tuple(inputs_retired_of(adapter, ""))
    except Exception:          # noqa: BLE001 -- a declaration that needs a real group is simply not usable for the static check
        return ()


def _retired_root(root: str, globs: Sequence[str]) -> Optional[str]:
    for g in globs:
        if fnmatch.fnmatchcase(root, g) or fnmatch.fnmatchcase(root + "/", g) or fnmatch.fnmatchcase(root + "/_", g):
            return g
    return None


@dataclass
class RootCoverage:
    adapter: str
    root: Optional[str]
    state: str
    components: List[str] = field(default_factory=list)   # the components that hold (or restrict) the root
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoverageReport:
    config: str
    adapters: Dict[str, Dict[str, Any]]
    roots: List[RootCoverage]
    summary: Dict[str, Any]
    ok: bool
    home: str = HOME                 # the home the roots were judged against (``~`` in the text and YAML forms)
    schema: str = COVERAGE_SCHEMA

    def as_dict(self) -> Dict[str, Any]:
        return {"schema": self.schema, "config": self.config, "home": self.home, "ok": self.ok, "summary": self.summary, "adapters": self.adapters,
                "roots": [r.as_dict() for r in self.roots]}


def coverage(cfg: Config, registry: Optional[Mapping[str, Callable[[], Any]]] = None, home: Optional[str] = None,
             only: Optional[Sequence[str]] = None) -> CoverageReport:
    """Every registered adapter's declared ``roots`` (the instance attribute its constructor sets) against the configuration's components:
    the adapters are CONSTRUCTED and nothing else -- no ``discover()``, no store opened, no directory listed (the §1.5 rule: the only
    store access of a cluster read is the shard's). ``registry`` defaults to ``adapters.load_all()``; ``home`` to this host's home.

    Per adapter: whether the configuration declares it (an undeclared adapter FAILS the report), ``cluster: refused`` + reason, its
    shards and components, each root's :data:`ROOT_STATES` state with the component(s) that hold it, every held root none of whose
    holders the entry names (``missing_components``, the alternatives joined by ``|``: FAILS -- the plan pins generations from the entry)
    and the entry's components that hold none of its roots (``extra_components``: listed, a dependency read through another root's index
    is legitimate). ``ok`` iff no
    root is UNCOVERED or undeclared, every adapter is declared, every adapter constructs, and no entry misses a component."""
    if registry is None:
        from .adapters import load_all
        registry = load_all()
    home = (home or HOME).rstrip("/")
    names = sorted(registry)
    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise StoresError(f"adapter(s) not registered: {', '.join(unknown)}")
        names = [n for n in names if n in set(only)]
    rows: List[RootCoverage] = []
    adapters: Dict[str, Dict[str, Any]] = {}
    counts = {st: 0 for st in ROOT_STATES}
    undeclared_adapters: List[str] = []
    failed_construction: Dict[str, str] = {}
    missing_components: Dict[str, List[str]] = {}
    for name in names:
        decl = cfg.adapters.get(name)
        entry: Dict[str, Any] = {"declared": decl is not None, "refused": bool(decl and decl.refused), "reason": (decl.reason if decl else ""),
                                 "shards": (decl.shards if decl else None), "components": (list(decl.components) if decl else []),
                                 "roots": [], "missing_components": [], "extra_components": []}
        adapters[name] = entry
        if decl is None:
            undeclared_adapters.append(name)
        try:
            adapter = registry[name]()
        except Exception as exc:          # noqa: BLE001 -- the report names it; the battery decides
            failed_construction[name] = f"{type(exc).__name__}: {exc}"
            entry["construction_error"] = failed_construction[name]
            continue
        roots = list(getattr(adapter, "roots", ()) or ())
        retired = _retired_globs(adapter)
        volatile = {v.root: v.reason for v in (decl.volatile_roots if decl else [])}
        needed: List[List[str]] = []          # per held root, the components that hold it (one of them must be in the entry)
        for root in roots:
            if decl is not None and decl.refused:
                row = RootCoverage(name, root, REFUSED_ROOT, [], f"cluster: refused -- {decl.reason}")
            elif root is None or not str(root):
                row = RootCoverage(name, None, UNDECLARED, [], "the adapter's roots carry None: the declaration is broken")
            else:
                root = str(root)
                if root == home or root.startswith(home + "/"):
                    holds = [(c.slug, st) for c in cfg.components.values() for st in [root_coverage(root, c)] if st is not None]
                    full = [slug for slug, (st, _m) in holds if st == COVERED]
                    part = [slug for slug, (st, _m) in holds if st == RESTRICTED]
                    if full:
                        detail = "; ".join(f"{slug}: member {m}" for slug, (st, m) in holds if st == COVERED)
                        row = RootCoverage(name, root, COVERED, full, detail)
                    elif part:
                        detail = "members restrict: " + "; ".join(f"{slug}: {m}" for slug, (st, m) in holds if st == RESTRICTED) \
                            + " -- the shard's own discovery is the exact check"
                        row = RootCoverage(name, root, RESTRICTED, part, detail)
                    else:
                        row = RootCoverage(name, root, UNCOVERED, [], "no component's base holds it")
                else:
                    g = _retired_root(root, retired)
                    if g is not None:
                        row = RootCoverage(name, root, RETIRED, [], f"outside the home; the adapter retires it (inputs_retired {g})")
                    elif root in volatile:
                        row = RootCoverage(name, root, VOLATILE, [], f"outside the home; declared volatile: {volatile[root]}")
                    else:
                        row = RootCoverage(name, root, UNCOVERED, [], "outside the home: no component can hold it (declare it volatile with a reason, "
                                                                     "or the adapter retires it)")
            rows.append(row)
            entry["roots"].append(row.as_dict())
            counts[row.state] += 1
            if row.components:
                needed.append(list(row.components))
        if decl is not None and not decl.refused:
            # every held root must have at least ONE of its holders in the entry (a root above several members-restricted components is
            # held by the one(s) whose members it opens -- the entry names those; the shard's own discovery is the exact check); the
            # alternatives are reported together when none is named
            missing = [holders for holders in needed if not any(h in decl.components for h in holders)]
            entry["missing_components"] = sorted({" | ".join(h) for h in missing})
            held = {h for holders in needed for h in holders}
            entry["extra_components"] = [slug for slug in decl.components if slug not in held]
            if entry["missing_components"]:
                missing_components[name] = entry["missing_components"]
    # the per-state root counts are keyed by the state names (one of them is "refused": every root of a refused adapter); the ADAPTER
    # counts carry their own names, so the two never collide
    summary = {"adapters": len(names), "declared_adapters": sum(1 for e in adapters.values() if e["declared"]),
               "refused_adapters": sum(1 for e in adapters.values() if e["refused"]), "roots": len(rows), **counts,
               "undeclared_adapters": undeclared_adapters, "failed_construction": failed_construction, "missing_components": missing_components}
    ok = not (counts[UNCOVERED] or counts[UNDECLARED] or undeclared_adapters or failed_construction or missing_components)
    return CoverageReport(cfg.path, adapters, rows, summary, ok, home=home)


def _tilde(path: Optional[str], home: str = HOME) -> str:
    if path is None:
        return "None"
    return ("~" + path[len(home):]) if path == home or path.startswith(home + "/") else path


def format_coverage(rep: CoverageReport) -> str:
    lines = [f"store coverage ({rep.config}): every registered adapter's declared roots against the components (constructed, never discovered)"]
    for name, e in rep.adapters.items():
        if not e["declared"]:
            head = f"{name}: NOT DECLARED in the configuration"
        elif e["refused"]:
            head = f"{name}: cluster: refused -- {e['reason']}"
        else:
            head = f"{name}: shards={e['shards']} components=[{', '.join(e['components'])}]"
        lines.append(head)
        if e.get("construction_error"):
            lines.append(f"  CONSTRUCTION FAILED: {e['construction_error']}")
        for r in e["roots"]:
            comps = f" [{', '.join(r['components'])}]" if r["components"] else ""
            lines.append(f"  {_tilde(r['root'], rep.home)} -> {r['state']}{comps}  ({r['detail']})")
        if e["missing_components"]:
            lines.append(f"  MISSING components (the roots lie under them; the entry must name them): {', '.join(e['missing_components'])}")
        if e["extra_components"]:
            lines.append(f"  extra components (no declared root lies under them): {', '.join(e['extra_components'])}")
    sm = rep.summary
    lines.append(f"coverage: {sm['adapters']} adapters ({sm['declared_adapters']} declared, {sm['refused_adapters']} refused), {sm['roots']} roots: "
                 + ", ".join(f"{st} {sm[st]}" for st in ROOT_STATES)
                 + f"; undeclared adapters: {sm['undeclared_adapters'] or 'none'}; construction failures: {sorted(sm['failed_construction']) or 'none'}; "
                 f"missing components: {sm['missing_components'] or 'none'}")
    lines.append("coverage OK" if rep.ok else "coverage NOT OK")
    return "\n".join(lines)


def coverage_yaml_fragment(rep: CoverageReport) -> str:
    """A YAML fragment to paste into the configuration for what the report found wanting: an ``adapters:`` stub for every undeclared
    adapter (shards 1 -- size it by the rule; the components its roots lie under; an UNCOVERED root as a comment) and a ``components:``
    stub per UNCOVERED root under the home. Nothing for a clean report."""
    out: List[str] = []
    stubs = [(n, e) for n, e in rep.adapters.items() if not e["declared"] or e["missing_components"]]
    if stubs:
        out.append("adapters:")
        for name, e in stubs:
            comps = []
            for r in e["roots"]:
                for c in r["components"]:
                    if c not in comps:
                        comps.append(c)
            out.append(f"  {name}:")
            out.append(f"    shards: {e['shards'] or 1}   # size it: >= 4 GB or >= 10,000 files -> 8; >= 500 MB -> 4; else 1")
            out.append(f"    components: [{', '.join(comps)}]")
            for r in e["roots"]:
                if r["state"] == UNCOVERED:
                    out.append(f"    # UNCOVERED root: {_tilde(r['root'], rep.home)}")
    unc = [r for r in rep.roots if r.state == UNCOVERED and r.root and (r.root == rep.home or r.root.startswith(rep.home + "/"))]
    if unc:
        out.append("components:")
        seen = set()
        for r in unc:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", _tilde(r.root, rep.home).strip("~/")).strip("_").lower()
            if slug in seen:
                continue
            seen.add(slug)
            out.append(f"  {slug}:")
            out.append(f"    base: {_tilde(r.root, rep.home)}")
            out.append("    authority: grading_host_copy")
            out.append(f"    note: \"UNCOVERED root of {r.adapter}\"")
    return "\n".join(out) + ("\n" if out else "")


# --------------------------------------------------------------------------------------------
# stores status
# --------------------------------------------------------------------------------------------


def _latest_manifest_meta(sdir: str) -> Optional[Dict[str, Any]]:
    best = None
    for p in glob.glob(os.path.join(sdir, "MANIFEST-*.json")):
        m = _load_json(p)
        if m and m.get("schema") == MANIFEST_SCHEMA and (best is None or str(m.get("built_utc")) > str(best.get("built_utc"))):
            best = m
    return best


#: the per-component digest of the last `check-authority` reading `stores status` shows (the AUTHORITY ROWS): from
#: ``state/store_authority.json`` (:func:`check_authority`), never re-read from the destination here (`--probe` does that)
AUTHORITY_ROW_KEYS = ("state", "problems", "notes", "current", "exists", "verified", "sealed", "published_record", "change_utc", "published_utc",
                      "publish_mode", "grading_host_mirror_present", "carry_overdue")


def install_site_aliases(root: Optional[str]) -> None:
    """Install the root's site alias table (:mod:`site_aliases`) for the reads of state this module wrote: a state file written
    before the site renamed its clusters and grading host carries keys and values under the older names, read through
    :func:`site_aliases.get_compat`. A refused configuration or alias file raises (``ConfigError``), so no read of that state goes
    ahead without the table the site declared."""
    from . import config as CFG
    if not root:
        return
    SA.install(CFG.load(root))


def canonical_authority(value: Optional[object]) -> Optional[str]:
    """An authority read back from a state file written before the site renamed its clusters and grading host: the value of
    :data:`AUTHORITIES` whose older spelling it is (:func:`site_aliases.legacy_keys` under the installed table), else the value as read;
    None stays None."""
    if value is None:
        return None
    v = str(value)
    if v in AUTHORITIES:
        return v
    return next((a for a in AUTHORITIES if v in SA.legacy_keys(a)), v)


def authority_rows(root: str) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    """(when `check-authority` last ran, its per-component digest) from ``state/store_authority.json``; (None, {}) when it never ran.
    A key written under the site's older names reads as its current one (:func:`install_site_aliases`)."""
    rec = _load_json(os.path.join(root, "state", AUTHORITY_STATE_FILE))
    if not rec or rec.get("schema") != AUTHORITY_STATE_SCHEMA:
        return None, {}
    install_site_aliases(root)
    return rec.get("checked_utc"), {slug: {k: SA.get_compat(row, k) for k in AUTHORITY_ROW_KEYS}
                                    for slug, row in (rec.get("components") or {}).items()}


def status(root: str, cfg: Config, dest: Optional[Destination] = None, probe: bool = False) -> Dict[str, Any]:
    """Per component: authority and producer, the latest grading host manifest, the generation state (the mirror's last outcome and its
    level), what is published, the ``cluster_a_origin`` authority state, the last run, and the AUTHORITY ROW -- the last
    `check-authority` reading of its `current` (:func:`authority_rows`). From ``state/store_mirror.json``, ``state/store_authority.json``
    and the per-slug state dir; ``probe=True`` also asks the destination for ``current``."""
    st = load_state(root)
    remote = Remote(dest) if (probe and dest is not None) else None
    checked_utc, auth = authority_rows(root)
    rows: Dict[str, Any] = {}
    for slug, comp in cfg.components.items():
        rec = dict(st.get("components", {}).get(slug) or {})
        latest = rec.get("latest_manifest") or _latest_manifest_meta(component_state_dir(root, slug))
        gen = rec.get("generation") or {}
        gstate = str(gen.get("state") or "")
        row = {"authority": comp.authority, "root": comp.root_spelling, "home_rel": comp.home_rel, "adapters": comp.adapters, "origin": comp.origin,
               "latest_manifest": ({k: latest.get(k) for k in ("content_sha8", "built_utc", "n_files", "bytes", "hashed", "from_cache")} if latest else None),
               "generation": rec.get("generation"), "published": rec.get("published"),
               "verified": gstate in ("published", "verified", "pulled") and not rec.get("current_dangling"),
               "current_dangling": rec.get("current_dangling"),
               # the mirror's level for this component: the generation state is an outcome name for the reversed mirror (verified /
               # unpushed_changes / frozen_differs / pulled / ...) and `published` for the carry; an unknown state is not green
               "level": ("green" if gstate == "published" else OUTCOME_LEVELS.get(gstate, "red")) if gstate else None,
               "authority_state": authority_state(root, slug) if comp.authority == "cluster_a_origin" else None,
               "authority_check": auth.get(slug),
               "last_run": rec.get("last_run"), "original": comp.original, "producer": comp.producer or None,
               "grading_host_mirror": grading_host_mirror_path(root, comp) if comp.authority == "cluster_a" else comp.grading_host_mirror,
               "transitional_carry": comp.transitional_carry, "carry_deadline": comp.carry_deadline}
        if remote is not None:
            gen_dir = dest.gen_dir(slug)
            cur = remote.readlink(f"{gen_dir}/current")
            # `current` is a pointer; whether the directory it names is there is the question (a dangling pointer is what a
            # renamed-away tree leaves behind, and every read of the component is refused until the next tick repairs it)
            tree = remote.isdir(f"{gen_dir}/{cur}") if cur else None
            row["probe"] = {"current": cur, "gen_dir": gen_dir, "current_tree_present": tree,
                            "problem": (None if (cur is None or tree) else
                                        f"current -> {cur} but {gen_dir}/{cur} is not a directory (dangling; the next tick copies afresh)")}
        rows[slug] = row
    return {"schema": STATE_SCHEMA, "config": cfg.path, "destination": st.get("destination"), "updated_utc": st.get("updated_utc"),
            "authority_checked_utc": checked_utc,
            "adapters": {a.name: {"shards": a.shards, "components": a.components} for a in cfg.adapters.values()}, "components": rows}


def format_status(data: Mapping[str, Any]) -> str:
    lines = [f"store copies ({data.get('config')}); destination {data.get('destination') or '-'}; state updated {data.get('updated_utc') or 'never'}; "
             f"authority checked {data.get('authority_checked_utc') or 'never'}",
             "slug | authority | published | generation | latest manifest (files / MB / built) | authority state | last run | producer | level | "
             "authority check (current / verified / sealed / recorded / problems)"]
    for slug, r in data.get("components", {}).items():
        lm = r.get("latest_manifest") or {}
        g = r.get("generation") or {}
        a = r.get("authority_state") or {}
        lr = r.get("last_run") or {}
        ac = r.get("authority_check") or {}
        mb = f"{(lm.get('bytes') or 0) / 1e6:.0f}" if lm else "-"
        if ac and ac.get("state") == "not_applicable":
            check = "not_applicable"
        elif ac:
            flags = "/".join("yes" if ac.get(k) else "no" for k in ("verified", "sealed", "published_record"))
            check = f"{ac.get('current') or '-'} {flags} {', '.join(ac.get('problems') or []) or 'ok'}"
            if ac.get("carry_overdue"):
                check += " carry_overdue"
        else:
            check = "unchecked"
        producer = r.get("producer") or "-"
        if r.get("transitional_carry"):
            producer += f" (carry until {r.get('carry_deadline')})"
        lines.append(" | ".join([slug, r["authority"], str(r.get("published") or "-"), f"{g.get('content_sha8') or '-'}:{g.get('state') or '-'}",
                                 f"{lm.get('content_sha8') or '-'} {lm.get('n_files') or '-'} / {mb} / {lm.get('built_utc') or '-'}",
                                 str(a.get("state") or ("unchecked" if r["authority"] == "cluster_a_origin" else "-")),
                                 f"{lr.get('utc') or '-'} {lr.get('outcome') or ''}".strip(), producer, str(r.get("level") or "-"), check]))
    # one PROBLEM line per dangling `current`: from the RECORD when the last tick found the pointer dangling and did not get to
    # republish it (no probe needed), and from `--probe` when the destination is asked directly
    for slug, r in data.get("components", {}).items():
        dangle = r.get("current_dangling") or {}
        if dangle:
            lr = r.get("last_run") or {}
            lines.append(f"PROBLEM {slug}: {dangle.get('why') or 'current dangles'} (found {dangle.get('found_utc') or '-'}; "
                         f"last run {lr.get('outcome') or '-'}; nothing is published until the next tick republishes it)")
        problem = (r.get("probe") or {}).get("problem")
        if problem:
            lines.append(f"PROBLEM {slug}: {problem}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI: `registry stores ...` (registered by cli.build_parser) and the stdlib entry points the sbatch runs
# --------------------------------------------------------------------------------------------


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help=f"cluster_reads.yaml to read (default ${CONFIG_ENV}, else the checkout's configs/registry/cluster_reads.yaml)")
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("mirror", help="the store mirror, by each component's authority: grading_host_copy -> carried to the destination as an immutable, "
                                      "verified generation (rsync --link-dest, a Slurm verify job watched in the foreground, `current` renamed only "
                                      "when the verify matches the manifest); cluster_a -> the grading host copy VERIFIED against the published generation "
                                      "(a grading host producer: verified / unpushed_changes, or pushed on its behalf under transitional_carry; frozen: "
                                      "frozen_differs is red and never repaired; cluster: the generation PULLED into <root>/stores/mirror/<slug>, "
                                      "hash-verified); cluster_a_origin -> skipped (`stores check`)")
    s.add_argument("--dest", help="destination name from the config (the only one when a single destination is declared)")
    s.add_argument("--component", dest="slugs", action="append", metavar="SLUG", help="only this component (repeatable); `--root` stays the registry root")
    s.add_argument("--dry-run", action="store_true", help="read only: stat the canonical copies (and, for a cluster_a component, compare the grading host "
                                                          "mirror with the published generation through the stat cache), print what would happen; write nothing")
    s.add_argument("--full", action="store_true", help="the weekly full verify (registry-store-mirror-full.timer): hash the grading host side with the stat "
                                                       "cache ignored and RE-HASH every published generation on the destination with the pinned verifier "
                                                       "(the report under <stores>/full/<slug>/; the live VERIFIED file untouched); a mismatch is "
                                                       "full_verify_failed, named, nothing renamed")
    s.add_argument("--adopt-unrecorded", action="store_true",
                   help="adopt every selected component whose `current` has no PUBLISHED record (written under the publish lock, nothing switched; "
                        "a differing copy then supersedes it) -- the one-time step for generations published before records existed; for a verify-only "
                        "cluster_a component (a grading host or frozen producer) the seal is written too when missing, and only a `current` whose VERIFIED "
                        "report admits its manifest is adopted (current_unverified is never adopted; a cluster producer's `current` is never adopted here)")
    s.add_argument("--bound-s", type=float, default=0.0, metavar="S",
                   help="the controller's own bound on each component's verify watch (a grading_host_copy carry and a cluster_a transitional carry alike): "
                        "past it the submitted verify job is cancelled BY ID and "
                        "confirmed terminal (sacct row + out of the queue) before this process returns (outcome verify_cancelled; an unconfirmed "
                        "cancel keeps the awaiting_verify record, and every later run -- on either route, a cluster_a component's before its grading host "
                        "mirror is even looked for -- settles that record by the same rule before it copies, "
                        "renames or switches anything: verify_in_flight while the job reads non-terminal, verify_unconfirmed while neither the "
                        "queue nor the accounting has a row for it; no other outcome writes over that record while the job is unproved). "
                        "0 = the destination's watch bound (watch_max_hours). The bound is enforced in "
                        "this process -- the watcher is killed at it; its own --max-hours is the next whole hour past it, the script's arithmetic "
                        "being integer. Never wrap this command in `timeout`: a "
                        "killed controller abandons its verifier with the lock released (the landing window passes WINDOW_MIRROR_WAIT_S here)")
    s.add_argument("--lock-wait-s", type=float, default=0.0, metavar="S",
                   help="wait up to S seconds for another `stores mirror` run to release <root>/state/store_mirror.lock before giving up (default 0: "
                        "a held lock exits 0 at once -- the 30-min tick yields); a run that waited and still finds it held exits 3, so the weekly "
                        "unit's Restart=on-failure retries it instead of skipping its week")
    s = sub.add_parser("check", help="cluster_a_origin components: compare the grading host mirror's manifest with a manifest of the original built on the "
                                     "cluster (store_verify.sbatch manifest-only) and record match / mismatch in state/store_manifests/<slug>/authority.json")
    s.add_argument("--dest")
    s.add_argument("--component", dest="slugs", action="append", metavar="SLUG")
    s = sub.add_parser("status", help="per component: latest manifest, generation state, what is published, authority state")
    s.add_argument("--dest")
    s.add_argument("--json", action="store_true")
    s.add_argument("--probe", action="store_true", help="also read the destination's `current` pointer over ssh and check that the generation it "
                                                        "names is a directory (a dangling pointer is a PROBLEM line)")
    s = sub.add_parser("revalidate", help="admission helper: is the grading host canonical copy still the content of the pinned generation? "
                                          "(stat walk + moved files hashed + a seeded sample; the refusal by name)")
    s.add_argument("--component", dest="slugs", action="append", metavar="SLUG", required=True)
    s.add_argument("--generation", required=True, metavar="SHA8")
    s.add_argument("--seed", type=int)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("coverage", help="every registered adapter's DECLARED roots against the components: the adapters are constructed and "
                                        "nothing is discovered or opened; exit 1 when a root is UNCOVERED, an adapter is undeclared or an entry "
                                        "misses a component its roots lie under")
    s.add_argument("--adapter", dest="adapters", action="append", metavar="NAME", help="only this adapter (repeatable)")
    s.add_argument("--json", action="store_true")
    s.add_argument("--emit-yaml", action="store_true", help="print a YAML fragment for what the report found wanting (stubs to size and paste)")
    s = sub.add_parser("push", help="the producer's publication of ONE component: its grading host copy (or --from DIR) becomes a verified, sealed generation "
                                    "on the destination with PUBLISHED-<sha8>.json (change_utc = the producer's committed change) beside it; exit 0 only "
                                    "on published / unchanged (/ deferred by min_publish_interval_s). --on-cluster runs the same flow where the tree is "
                                    "(inside the producer's job: local copy, in-process verify, no ssh, no credential)")
    s.add_argument("slug", metavar="SLUG", help="the component (declared in the config; with --on-cluster any slug the stores directory may hold)")
    s.add_argument("--dest", help="destination name from the config (the only one when a single destination is declared)")
    s.add_argument("--from", dest="from_dir", metavar="DIR", help="the tree to publish instead of the component's grading_host_mirror / base (REQUIRED with --on-cluster)")
    s.add_argument("--on-cluster", action="store_true", help="run on the cluster, in the producer's job: --from DIR into --stores-dir (default: the destination's stores "
                                                             "directory), verify in this process, seal, publish; `python3 -S -m sisteer.registry.stores push` is the stdlib spelling")
    s.add_argument("--stores-dir", metavar="DIR", help="with --on-cluster: the local stores directory ($SHARED/registry/stores)")
    s.add_argument("--note", default="", help="recorded in PUBLISHED-<sha8>.json")
    s.add_argument("--session", help="the producing session's name, recorded as producer_session")
    s.add_argument("--workers", type=int, default=16, help="with --on-cluster: hashing threads of the in-process verify")
    s.add_argument("--lock-wait-s", type=float, default=PUBLISH_LOCK_WAIT_S, metavar="S",
                   help="how long to wait for another publisher of the same component before refusing as lock_busy (the destination-side publish "
                        f"lock: the whole flow with --on-cluster, the publish step from the grading host; default {PUBLISH_LOCK_WAIT_S:.0f})")
    s.add_argument("--adopt-unrecorded", action="store_true",
                   help="a `current` with no PUBLISHED record is refused (current_unrecorded: a publisher may be mid-flight there); with this flag the "
                        "generation is ADOPTED -- when it is this push's content its record is written under the publish lock without switching, "
                        "else this publication supersedes it (the operator asserts it is a legacy generation, not a publisher in flight)")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("check-authority", help="read, in ONE batch on the destination, every component's `current` (target, real path, existence) and whether "
                                               "VERIFIED, PUBLISHED and SEALED stand beside it -> state/store_authority.json; exit 1 naming every dangling, "
                                               "unverified or unsealed `current` (and a cluster_a component with none)")
    s.add_argument("--dest")
    s.add_argument("--require-certified", action="store_true",
                   help="the strict reading for the one-time move's after-check: a grading_host_copy component with no `current` is the problem "
                        "not_certified_never_published (by default it has no authority to check and reads ok); the summary counts certified = "
                        "current exists + VERIFIED + SEALED + PUBLISHED")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("attest", help="re-attest a PUBLISHED generation's directory listings: for a component whose grading host copy is the published "
                                      "generation's content, write MANIFEST-<sha8>.<lsha8>.json beside the tree, run the pinned verifier once more over "
                                      "the sealed tree against that attestation into VERIFIED-<sha8>.<lsha8>.json (the live report untouched, nothing "
                                      "renamed) and record the attestation the window pins (published_attestation); a tree whose directory set is not the "
                                      "canonical's is refused by name (listing_only_drift) and nothing changes -- the next file change publishes a "
                                      "generation attesting the current listings; exit 0 iff every component is attested, already attested or has nothing "
                                      "to attest (content_differs: the mirror publishes first)")
    s.add_argument("--dest", help="destination name from the config (the only one when a single destination is declared)")
    s.add_argument("--component", dest="slugs", action="append", metavar="SLUG", help="this component (repeatable)")
    s.add_argument("--all-unattested", action="store_true",
                   help="every component whose last mirror tick recorded listings_unattested (a generation verified before the directory-listings "
                        "rule) or listings_changed (a listing moved from the attested one) in state/store_mirror.json -- the one-time step after "
                        "the rule landed; nothing selected exits 0")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("verify-authority", help="the one-time verified move of a component: a fresh grading host manifest vs the CURRENT generation's (both saved "
                                                "under --out), equal -> store_verify.sbatch re-hashes the generation, the generation is sealed and recorded, "
                                                "<slug>.verified.json is written; unequal -> <slug>.diff.json (push first); exit 0 iff every component verified")
    s.add_argument("--dest")
    s.add_argument("--component", dest="slugs", action="append", metavar="SLUG", required=True)
    s.add_argument("--out", metavar="DIR", help=f"where the evidence files go (default <root>/state/{MOVE_SUBDIR}/)")
    s.add_argument("--json", action="store_true")


def _log(line: str) -> None:
    print(line, file=sys.stderr)


def _cli_push(args, root: str, cfg: Config) -> int:
    if args.on_cluster:
        if not args.from_dir:
            print("stores push --on-cluster: --from DIR is required (the tree the producer publishes)", file=sys.stderr)
            return 2
        stores_dir = args.stores_dir or cfg.destination(args.dest).stores_dir
        comps = cfg.components.get(args.slug)
        # --from DIR is the whole component; the declared excludes (caches) still apply when the slug is declared
        r = push_on_cluster(args.slug, args.from_dir, stores_dir, note=args.note, session=args.session, workers=args.workers,
                            exclude=(comps.exclude if comps else ()), log=_log, lock_wait_s=args.lock_wait_s, adopt_unrecorded=bool(args.adopt_unrecorded))
        print(json.dumps(r.as_dict(), default=str) if args.json else f"{r.slug}: {r.outcome} {r.content_sha8 or ''} {r.detail.get('why') or r.detail.get('reason') or r.detail.get('error') or ''}".rstrip())
        return 0 if r.outcome in PUSH_OK_OUTCOMES else 1
    dest = cfg.destination(args.dest)
    comp = cfg.select([args.slug])[0]
    why = _push_refusal(comp)
    if why:
        print(f"stores push {comp.slug}: refused -- {why}", file=sys.stderr)
        return 1
    # ONE writer of state/store_mirror.json and of the destination's generations: a push never runs beside a mirror tick or another push
    with MirrorLock(root) as lock:
        if not lock.held:
            print(f"stores push: another run holds {lock.path}; nothing pushed -- retry when it finishes", file=sys.stderr)
            return 3
        remote = Remote(dest)
        pin = pin_cluster_tree(dest, remote, log=_log)             # step 0: the verify job runs exactly this code, or nothing is copied
        r = push_component(root, dest, comp, from_dir=args.from_dir, note=args.note, session=args.session, log=_log, pin=pin, lock_wait_s=args.lock_wait_s,
                           adopt_unrecorded=bool(args.adopt_unrecorded))
        print(json.dumps(r.as_dict(), default=str) if args.json else
              f"{r.slug}: {r.outcome} {r.content_sha8 or ''} {r.detail.get('why') or r.detail.get('reason') or r.detail.get('error') or ''}".rstrip())
        return 0 if r.outcome in PUSH_OK_OUTCOMES else 1


def _cli_check_authority(args, root: str, cfg: Config) -> int:
    dest = cfg.destination(args.dest)
    rec = check_authority(root, cfg, dest, require_certified=bool(getattr(args, "require_certified", False)))
    print(json.dumps(rec, indent=1, default=str) if args.json else format_authority(rec))
    return 0 if rec["ok"] else 1


def _cli_verify_authority(args, root: str, cfg: Config) -> int:
    dest = cfg.destination(args.dest)
    comps = cfg.select(args.slugs)
    out_dir = os.path.abspath(os.path.expanduser(args.out)) if args.out else os.path.join(root, "state", MOVE_SUBDIR)
    with MirrorLock(root) as lock:
        if not lock.held:
            print(f"stores verify-authority: another run holds {lock.path}; nothing verified -- retry when it finishes", file=sys.stderr)
            return 3
        pin = pin_cluster_tree(dest, Remote(dest), log=_log)
        rc = 0
        for c in comps:
            rec = verify_authority(root, dest, c, out_dir, log=_log, pin=pin)
            print(json.dumps(rec, default=str) if args.json else
                  f"{c.slug}: {rec['state']} current={rec.get('current') or '-'} {rec.get('why') or rec.get('error') or ''}".rstrip())
            rc = rc or (0 if rec["state"] == "verified" else 1)
        return rc


def _cli_status(args, root: str, cfg: Config) -> int:
    dest = cfg.destination(args.dest) if (args.probe or args.dest) else None
    data = status(root, cfg, dest, probe=bool(args.probe))
    print(json.dumps(data, indent=1, default=str) if args.json else format_status(data))
    return 0


def _cli_coverage(args, root: str, cfg: Config) -> int:
    rep = coverage(cfg, only=args.adapters)
    if args.emit_yaml:
        print(coverage_yaml_fragment(rep), end="")
    elif args.json:
        print(json.dumps(rep.as_dict(), indent=1, default=str))
    else:
        print(format_coverage(rep))
    return 0 if rep.ok else 1


def _cli_revalidate(args, root: str, cfg: Config) -> int:
    rc = 0
    for slug in args.slugs:
        rv = revalidate(slug, args.generation, root=root, cfg=cfg, seed=args.seed)
        print(json.dumps(rv.as_dict(), default=str) if args.json else
              f"{slug} @ {rv.generation}: {'ok' if rv.ok else 'REFUSED ' + str(rv.refusal)} -- {rv.reason or f'{rv.n_files} files, {rv.moved} moved, {rv.sampled} sampled'}")
        rc = rc or (0 if rv.ok else 1)
    return rc


def _cli_check(args, root: str, cfg: Config) -> int:
    dest = cfg.destination(args.dest)
    comps = [c for c in cfg.select(args.slugs) if c.authority == "cluster_a_origin"]
    if not comps:
        print("stores check: no cluster_a_origin component declared" + (f" among {args.slugs}" if args.slugs else "") + "; nothing to compare")
        return 0
    # ONE writer of state/store_mirror.json: a check never runs beside a mirror tick (`status` only reads and takes nothing)
    with MirrorLock(root) as lock:
        if not lock.held:
            print(f"stores check: another run holds {lock.path}; nothing compared -- retry when it finishes", file=sys.stderr)
            return 3
        pin = pin_cluster_tree(dest, Remote(dest), log=_log)       # step 0: the job runs exactly this code, or nothing runs
        rc = 0
        for c in comps:
            rec = check_component(root, dest, c, log=_log, pin=pin)
            print(json.dumps(_authority_summary(rec, ("slug", "state", "checked_utc", "job_id", "error", "why")), default=str))
            rc = rc or (0 if rec["state"] in ("match", "mismatch") else 1)
        return rc


def _cli_attest(args, root: str, cfg: Config) -> int:
    dest = cfg.destination(args.dest)
    slugs = list(args.slugs or [])
    if getattr(args, "all_unattested", False):
        for slug, rec in (load_state(root).get("components") or {}).items():
            gstate = str((rec.get("generation") or {}).get("state") or "")
            if slug in cfg.components and slug not in slugs and rec.get("published") and gstate in ("listings_unattested", "listings_changed"):
                slugs.append(slug)
    if not slugs:
        if getattr(args, "all_unattested", False):
            print("stores attest: nothing to attest (no component's last mirror tick recorded listings_unattested or listings_changed)")
            return 0
        print("stores attest: name --component SLUG (repeatable) or --all-unattested", file=sys.stderr)
        return 2
    comps = cfg.select(slugs)
    # ONE writer of state/store_mirror.json and of the destination's generations: an attest never runs beside a mirror tick or a push
    with MirrorLock(root) as lock:
        if not lock.held:
            print(f"stores attest: another run holds {lock.path}; nothing attested -- retry when it finishes", file=sys.stderr)
            return 3
        remote = Remote(dest)
        pin = pin_cluster_tree(dest, remote, log=_log)             # step 0: the verify job runs exactly this code, or nothing is submitted
        results = []
        for c in comps:
            r = attest_listings(root, dest, c, log=_log, pin=pin)
            print(json.dumps(r.as_dict(), default=str) if args.json else
                  f"{r.slug}: {r.outcome} {r.content_sha8 or ''} listings={(r.listings_sha256 or '-')[:8]} {r.detail.get('why') or r.detail.get('error') or ''}".rstrip())
            results.append(r)
        summary = {"summary": True, "destination": dest.name, "components": len(results), "ok": sum(1 for r in results if r.ok),
                   "outcomes": {o: sum(1 for r in results if r.outcome == o) for o in sorted({r.outcome for r in results})},
                   "levels": {lv: sum(1 for r in results if r.level == lv) for lv in ("green", "amber", "red")}, "revision": pin.revision}
        print(json.dumps(summary) if args.json else
              f"stores attest: {summary['ok']} of {len(results)} ok -- " + ", ".join(f"{o} {n}" for o, n in summary["outcomes"].items()))
        return 0 if all(r.ok for r in results) else 1


def _cli_mirror(args, root: str, cfg: Config) -> int:
    dest = cfg.destination(args.dest)
    comps = cfg.select(args.slugs)
    # a dry run acts on nothing and takes no lock, so it writes nothing at all. A run asked to wait (`--lock-wait-s`: the weekly --full)
    # that still finds the lock held exits 3 -- a failure its unit's Restart=on-failure retries -- where the default (no wait) yields with 0
    wait_s = max(0.0, float(getattr(args, "lock_wait_s", 0.0) or 0.0))
    with MirrorLock(root, take=not args.dry_run, wait_s=wait_s) as lock:
        if not lock.held:
            if wait_s > 0:
                print(f"stores mirror: another run ({lock.holder_line()}) still holds {lock.path} after {lock.waited_s:.0f} s of --lock-wait-s "
                      f"{wait_s:.0f}; nothing done (exit 3: the unit's Restart=on-failure retries)", file=sys.stderr)
                return 3
            print(f"stores mirror: another run holds {lock.path}; exiting without acting")
            return 0
        if lock.waited_s >= 1.0:
            _log(f"stores mirror: waited {lock.waited_s:.0f} s for {lock.path}")
        remote = Remote(dest)
        if args.dry_run:      # pins nothing (a push writes on the destination); reports what a tick would pin
            pin, (sha, dirty) = None, running_revision(remote.run, dest.source_repo)
            revision: Dict[str, Any] = {"revision": sha, "dirty_tracked": len(dirty), "would_pin": dest.pinned_sbatch(sha)}
        else:                 # step 0, before any copy or submit: the destination runs exactly this code, or the tick refuses
            pin = pin_cluster_tree(dest, remote, log=_log)
            revision = {"revision": pin.revision, "pushed_via": pin.pushed_via, "verifier": pin.sbatch_path}
        results = []
        for c in comps:
            r = run_mirror(root, dest, c, dry_run=bool(args.dry_run), full=bool(getattr(args, "full", False)), log=_log, pin=pin,
                           adopt_unrecorded=bool(getattr(args, "adopt_unrecorded", False)),
                           bound_s=(float(getattr(args, "bound_s", 0.0) or 0.0) or None))
            print(json.dumps(r.as_dict(), default=str))
            results.append(r)
        # the components whose published generation stands with unattested listings (amber; the operator's `stores attest`), and whose
        # re-attestation refused a drift (amber, a coverage limitation), named so a reader of the summary knows what to run and what to read
        unattested = sorted(r.slug for r in results if r.outcome == "listings_unattested")
        drifted = sorted(r.slug for r in results if r.outcome == "listing_only_drift")
        summary = {"summary": True, "destination": dest.name, "components": len(results), "ok": sum(1 for r in results if r.ok),
                   "outcomes": {o: sum(1 for r in results if r.outcome == o) for o in sorted({r.outcome for r in results})},
                   "levels": {lv: sum(1 for r in results if r.level == lv) for lv in ("green", "amber", "red")},
                   "listings_unattested": unattested, "listing_only_drift": drifted,
                   "dry_run": bool(args.dry_run), "full": bool(getattr(args, "full", False)), **revision}
        if unattested:
            summary["attest"] = f"registry stores attest --dest {dest.name} --all-unattested"
            _log(f"stores mirror: {len(unattested)} component(s) published and verified with unattested directory listings "
                 f"({', '.join(unattested[:8])}{', ...' if len(unattested) > 8 else ''}); `{summary['attest']}` attests them")
        print(json.dumps(summary))
        return 0 if all(r.ok for r in results) else 1


def cmd_stores(args, root: str) -> int:
    """The `registry stores` subcommands; ``root`` is the registry root the CLI resolved (state lives under ``<root>/state/``)."""
    try:
        install_site_aliases(root)          # state written under the site's older names reads as its current names
        cfg = load_config(getattr(args, "config", None), root=root)
    except (StoresError, OSError, ValueError) as exc:
        print(f"stores: {exc}", file=sys.stderr)
        return 2
    try:
        return {"status": _cli_status, "revalidate": _cli_revalidate, "check": _cli_check, "mirror": _cli_mirror,
                "coverage": _cli_coverage, "push": _cli_push, "check-authority": _cli_check_authority,
                "verify-authority": _cli_verify_authority, "attest": _cli_attest}[args.action](args, root, cfg)
    except StoresError as exc:
        print(f"stores {args.action}: {exc}", file=sys.stderr)
        return 2


def _cluster_main(argv: Optional[Sequence[str]] = None) -> int:
    """``python3 -S -m sisteer.registry.stores verify|manifest|admit|push ...`` -- the stdlib entry points ``store_verify.sbatch`` and a
    producer's own job run."""
    p = argparse.ArgumentParser(prog="python3 -S -m sisteer.registry.stores")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="hash every manifest entry of a generation and write the VERIFIED report")
    v.add_argument("generation")
    v.add_argument("manifest")
    v.add_argument("--out", required=True)
    v.add_argument("--workers", type=int, default=16)
    v.add_argument("--job-id")
    v.add_argument("--attestation", metavar="FILE", help="the manifest .json whose directory listings to verify (default: the plain MANIFEST-<sha8>.json "
                                                        "beside the lines); a re-attestation names MANIFEST-<sha8>.<lsha8>.json and writes --out "
                                                        "VERIFIED-<sha8>.<lsha8>.json, its DEST-STAT keyed the same way (the live snapshot untouched)")
    m = sub.add_parser("manifest", help="manifest of a directory in place (MANIFEST rules), no stat cache")
    m.add_argument("directory")
    m.add_argument("--out-prefix", required=True)
    m.add_argument("--workers", type=int, default=16)
    m.add_argument("--member", action="append", default=[])
    m.add_argument("--exclude", action="append", default=[])
    m.add_argument("--job-id")
    m.add_argument("--grading-host-root", help="the grading host mirror's spelling of the base (`~/...`), recorded as `root` in the .json; never the compute node's home")
    g = sub.add_parser("admit", help="the destination change guard: is gen/<slug>/<sha8> still the verified, sealed generation? (exit 1 = refused)")
    g.add_argument("gen_dir")
    g.add_argument("sha8")
    g.add_argument("--seed", type=int)
    g.add_argument("--json", action="store_true")
    t = sub.add_parser("attest", help="the per-operation attestation of every pinned generation of a cluster-reads plan (attest.sbatch runs it before "
                                      "the read array): the authority facts, the full directory listing against the manifest, the directory seal, "
                                      "the content walk (first time, or after a week) or a seeded sample; writes --out; exit 0 once the record is "
                                      "written (a refusal is a RESULT the shards read), non-zero only when no record could be written")
    t.add_argument("--plan", required=True, metavar="FILE", help="<window root>/plan.json")
    t.add_argument("--out", required=True, metavar="FILE", help="<window root>/attestation.json")
    t.add_argument("--threads", type=int, default=ATTEST_THREADS)
    t.add_argument("--job-id")
    t.add_argument("--json", action="store_true")
    u = sub.add_parser("push", help="the producer's publication inside its own cluster job: copy --from DIR into <stores dir>/gen/<slug>/.incoming-<sha8>, "
                                    "manifest, verify in this process, seal the directories, publish through the guard, write PUBLISHED-<sha8>.json; "
                                    "exit 0 only on published / unchanged")
    u.add_argument("slug")
    u.add_argument("--from", dest="from_dir", required=True, metavar="DIR")
    u.add_argument("--stores-dir", required=True, metavar="DIR", help="$SHARED/registry/stores")
    u.add_argument("--note", default="")
    u.add_argument("--session")
    u.add_argument("--workers", type=int, default=16)
    u.add_argument("--member", action="append", default=[])
    u.add_argument("--exclude", action="append", default=[])
    u.add_argument("--lock-wait-s", type=float, default=PUBLISH_LOCK_WAIT_S, metavar="S",
                   help=f"how long to wait for another publisher of this slug to finish before refusing as lock_busy (default {PUBLISH_LOCK_WAIT_S:.0f})")
    u.add_argument("--adopt-unrecorded", action="store_true", help="adopt a `current` that has no PUBLISHED record (the record under the lock, nothing switched; "
                                                                   "a differing tree then supersedes it) instead of refusing current_unrecorded")
    u.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "push":
        r = push_on_cluster(a.slug, a.from_dir, a.stores_dir, note=a.note, session=a.session, workers=a.workers, members=a.member, exclude=a.exclude,
                            log=lambda line: print(line, file=sys.stderr), lock_wait_s=a.lock_wait_s, adopt_unrecorded=bool(a.adopt_unrecorded))
        print(json.dumps(r.as_dict(), default=str) if a.json else
              f"{r.slug}: {r.outcome} {r.content_sha8 or ''} {r.detail.get('why') or r.detail.get('reason') or r.detail.get('error') or ''}".rstrip())
        return 0 if r.outcome in PUSH_OK_OUTCOMES else 1
    if a.cmd == "verify":
        rep, dest_stat = verify_generation(a.generation, a.manifest, workers=a.workers, job_id=a.job_id, attestation=a.attestation)
        write_verified(a.out, rep, dest_stat)
        print(json.dumps({"ok": rep["ok"], "n_files": rep["n_files"], "n_symlinks": rep["n_symlinks"], "hashed": rep["hashed"], "sealed": rep["sealed"],
                          "elapsed_s": rep["elapsed_s"], "rate_mb_s": rep["rate_mb_s"], "missing": len(rep["missing"]), "corrupt": len(rep["corrupt"]),
                          "extra": len(rep["extra"]), "out": a.out, "dest_stat": rep["dest_stat"]}))
        return 0 if rep["ok"] else 1
    if a.cmd == "attest":
        rec = attest_plan(a.plan, a.out, threads=a.threads, job_id=a.job_id, log=lambda line: print(line, file=sys.stderr))
        print(json.dumps(rec, default=str) if a.json else
              f"attestation {rec['window']}: {'ok' if rec['ok'] else 'REFUSED ' + ', '.join(rec['refused'])} -- {rec['n_generations']} generation(s) in "
              f"{rec['elapsed_s']}s -> {a.out}" + "".join(f"\n  {slug}: {g.get('refusal')}: {g.get('reason')}" for slug, g in rec["generations"].items() if not g.get("ok")))
        return 0
    if a.cmd == "admit":
        ad = admit_generation(a.gen_dir, a.sha8, seed=a.seed)
        print(json.dumps(ad.as_dict(), default=str) if a.json else
              f"{ad.generation}: {'ok' if ad.ok else 'REFUSED ' + str(ad.refusal)} -- {ad.reason or f'{ad.n_files} files, {ad.n_symlinks} symlinks, {ad.moved} moved, {ad.sampled} sampled'}")
        return 0 if ad.ok else 1
    man = manifest_in_place(a.directory, a.out_prefix, a.member, a.exclude, a.workers, job_id=a.job_id, grading_host_root=a.grading_host_root)
    print(json.dumps({"n_files": man.n_files, "bytes": man.bytes, "content_sha8": man.content_sha8, "out": a.out_prefix + ".{sha256,json}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cluster_main())
