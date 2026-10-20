#!/usr/bin/env python3
"""reproduce/tools/stage07_mirror.py — stage 07's MIRROR: the WHOLE run stores a receipt names, copied off the Modal volume to this host and
to the primary cluster, sha-verified, so the filing points at a durable copy (the rule of 2026-10-12: the stores are COPIED off the
volume to the grading host and the cluster; the registry side stays the analysis filing, which records where the verified copy lives).

    python reproduce/tools/stage07_mirror.py --state-root S [--receipt F] [--volume twosi-fits] [--environment ENV] [--modal BIN]
                                             --cluster-root ROOT --cluster-label L --ssh-host HOST --cluster-dest DIR [--dry-run]

Runs AFTER `07 --phase receipt` and BEFORE `07 --phase file`; in neither `all` nor any other phase. Everything it pulls or checks comes from
the published receipt (<state>/07_store/receipt.json, or --receipt) and from the stores' OWN files — never a hard-coded control, suite or
marker name (nor a receipt schema version: any `stage07-receipt/*` receipt carrying stores is read): the receipt's `recs[<cell>].out` is each
store root (a container path under the volume mount, converted once by the receipt module's CHECKED conversion), `recs[<cell>].done_in` the
store keys it credits, `tree_sha` the tree, `pulled` the files it already hashed. <state> is THE VIEW'S state directory as 07 passes it — the
model's state dir for the baseline wave, <model state>/07_views/<view>/ for a dose view — the directory that view's receipt lives in, so the
mirror it publishes, the build it leaves on a refusal and the cluster record it installs are that view's; --cluster-dest follows the same
layout on the cluster (<reproduce root>/<model>/07_mirror for the wave, <reproduce root>/<model>/07_views/<view>/07_mirror for a dose view), and
--cluster-root is that expanded reproduce root itself, the directory every component of the destination is checked under (THE CLUSTER COPY).
Nothing here knows a view by name: the tool lays everything under the state root and the destination it is given.

THE PULL. `modal volume get --force` on each store root (VolumeCLI.get_tree, the full22 stager's CLI seam: the Modal CLI downloads a folder
recursively, laying it under the destination as <destination>/<folder name>/; verified on the smoke stores 2026-10-12) into
<state>/07_mirror.building/<receipt sha12>/<cell>/ — the store root's content directly under the cell. Nothing is excluded, nothing on the
volume is written, moved or deleted (`ls` and `get` only). Before the first byte: the stores are LISTED (`modal volume ls --json`, one call
per directory, recursively) for an inventory of every file and the size the CLI reports, the bytes are estimated from those sizes, and the
pull REFUSES by name when the state filesystem's free space is below estimate x --free-margin + --free-headroom. A leftover
07_mirror.building/<sha12>/ from a refused build is never removed here: it refuses by name (inspect it, then remove it by hand).

THE CHECKS, every one a refusal by name that leaves the .building directory in place for inspection:
  * every pulled file is sha256'd; every file the receipt hashed (`pulled`: the done and summary markers, the per-cell merged summary, the
    attempt pointers and completion records) must be present in the mirror at the same place under its store and hash BYTE FOR BYTE to
    the receipt's value (a flipped byte refuses);
  * every store key the receipt credits (`recs[<cell>].done_in` values) has cells/<key>/manifest.json in its store, whose
    producer.tree_sha is the receipt's tree and whose cell_id is the cell (the launcher's cell manifest, contracts.MANIFEST_KEY_PRODUCER /
    PRODUCER_BLOCK_FIELDS; the manifest carries no store-key field, so the KEY is checked as the directory the receipt names — read off the
    smoke stores 2026-10-12);
  * each store's RESULT (contracts.RESULT_FILENAME) ends on a line whose `result=` token is COMPLETE (contracts.RESULT_COMPLETE) and whose
    `sha=` token, when present, is the tree;
  * the file set pulled equals the file set listed, and each file's size RENDERED as the Modal CLI renders it (humanize_filesize: bytes
    exact below 1 KiB, one decimal above) equals the listing's string — Modal exposes no content hash, so this is a size check at the CLI's
    rounding, and the byte-for-byte guarantee is the receipt's sha256 over the files it pulled plus `sha256sum -c` on the cluster copy;
  * the DIRECTORY set pulled equals the directory set listed: `modal volume ls --json` is run once per directory,
    recursively, so every directory under the store root is reported exactly once, as a `type: dir` entry of its parent's listing (the CLI
    prints each entry's path from the volume root; there are no implicit parents to add — a file's parent is itself a listed directory of
    the walk), and the CLI's download creates a directory for every DIRECTORY entry, an empty one included; so a directory the listing
    reported that the pull did not create (an empty one omitted), or one the pull created that no listing reported, refuses by name — the
    directory set recorded in mirror.json below is the LISTED set, confirmed on disk, never the pulled tree alone (which certified both);
  * the pulled tree holds REGULAR FILES IN DIRECTORIES and nothing else (`walk_tree`): a symlink, a fifo, a socket or a device anywhere in
    it refuses by name (the CLI writes files and directories only — modal/cli/_download.py, 1.5.4 — so any other entry is not the store's).
On success MIRROR.sha256 (sha256sum format, paths relative to the mirror root, every data file) and mirror.json (schema stage07-mirror/2:
the volume paths, the file count, the bytes, the tree sha, the receipt's sha256, the time, per-cell manifests and RESULT lines, the size
check's wording, and `directories` — EVERY directory of the mirror, relative paths sorted, the CLI's layout as pulled: the CLI creates a
directory for every DIRECTORY entry it downloads, an empty one included, so the directory set is recorded rather than inferred from the file
list) are written into the build, which is then PUBLISHED as <state>/07_mirror/<receipt sha12>/ by one rename. A
previous mirror (another receipt's) sits beside it and is never deleted. A re-run for the same receipt VERIFIES the published copy (every
file against MIRROR.sha256, nothing extra, the directories exactly mirror.json's, mirror.json's receipt sha) and the CLUSTER copy its record
names (over ssh, below) and does nothing else — except finish the cluster push when no cluster record exists yet. A mirror.json of the
earlier schema (stage07-mirror/1, no `directories`) refuses by name: move that mirror aside and run the mirror again.

THE CLUSTER COPY. Before anything is written there, THE DESTINATION IS GUARDED (`guard_command`): over the same ssh, every
component between --cluster-root (the site's expanded SISTEER_REPRODUCE_ROOT, which 07 passes) and <cluster-dest>/<receipt sha12> — the
<model>, a dose view's 07_views/<view>, 07_mirror and <sha12> itself — is tested with `test -L`, and once the directory exists its `realpath`
must equal the root's own realpath plus the literal relative path; a symlink anywhere on that walk, or a destination that resolves elsewhere,
refuses by name (exit 61 / 62 there) and nothing is pushed, recorded or deleted. Without it `mkdir -p` accepted a destination that was a symlink,
the trailing-slash rsync wrote through it to wherever it pointed, `cd` followed it and the inventory read only its children, so the push and
every later filing recorded the symlink path as the durable copy. The root itself may be or lie under a symlink (its realpath is the anchor);
only what lies below it is held plain. The same guard runs, before the `cd`, in EVERY check command below (the push's and the re-verifications'):
a copy whose path has since become a symlink is REDIRECTED, never verified. Then the published mirror is pushed with rsync (no --delete; `-a
--partial`, so a re-run resumes/completes) over the site's cluster ssh (CLUSTER_SSH of SISTEER_PRIMARY_CLUSTER; `local:<dir>` is the repro_ssh
test seam: the string runs with `bash -c` here and the rsync target is the local path) to <cluster-dest>/<receipt sha12>/ — 07 passes
<expanded SISTEER_REPRODUCE_ROOT>/<model>/07_mirror — then `sha256sum -c MIRROR.sha256` runs THERE over ssh, followed by the directory's
INVENTORY OF EVERY ENTRY BY TYPE (`find . -mindepth 1
-printf "%y %p\0"` — GNU find's type letter and path, NUL-terminated so a path is never ambiguous — read back and held to the mirror: a
regular file must be one MIRROR.sha256 or the records name, a directory one mirror.json's `directories` lists, and a symlink, fifo, socket
or device is never the copy's, whatever its name. `sha256sum -c` reads the listed paths only, and rsync without
--delete adds and never removes, so a file left in the destination before the push, or added after it, was certified as part of the copy;
an inventory of `find . -type f`, regular files alone, left a symlink (to anywhere), a special file or a stray directory in the
destination or the copy unread, and the copy was certified with an unrecorded entry in it, unlike the local mirror, which
refuses a symlink); a mismatch, or an entry there that is not the mirror's, refuses by name and deletes nothing. Only after it verifies is cluster.json (schema
stage07-mirror-cluster/2: the label, the host, the cluster ROOT the destination was guarded under, the cluster path, the digest of
MIRROR.sha256, the sha256 of mirror.json, the receipt sha, the counts, the time; a /1 record — written before the guard — refuses by name,
naming the remedy: move it out of the mirror and run the mirror again, which re-pushes the record over a guarded, re-checked copy, nothing
re-pulled) composed — STAGED OUTSIDE the published mirror (<state>/07_mirror.building/<receipt sha12>.cluster.json), rsynced
to the cluster path on its own (--ignore-times: a stale record there matching it in size and mtime must still be replaced), read
back there by digest, and only then INSTALLED beside mirror.json by one rename (the
first cut wrote the record into the mirror before its transfer: a transfer that failed left a `verified: true` record a re-run
and the filing accepted; now a failed transfer leaves no record, and the re-run finishes the push). `07 --phase file` requires that record
for the receipt it files and copies its facts into the analysis record's `extra.mirror`.

WHAT A RE-RUN WITH A CLUSTER RECORD CHECKS (`verified_mirror`, which the filing calls too): every data file re-hashed against MIRROR.sha256
and nothing extra or missing locally (`verify_published`), mirror.json this receipt's, and cluster.json's digests of BOTH records equal to the files
beside it (`mirror_sha256sum_digest`, `mirror_json_sha256`), its tree and file count the mirror's, `verified: true` — a store byte or a
record changed after the cluster saw it refuses by name (the first cut hashed the two records only). And the record's TARGET must
be THIS run's — the same cluster label, ssh host, cluster root and <cluster-dest>/<sha12> — else the re-run refuses by name (a site whose
destination had moved reported the mirror as done while the copy sat at the former path): a mirror has one cluster record; a copy to a
second destination is a deliberate act (move cluster.json aside, recorded, and run the mirror again; the earlier copy is never deleted).
THEN THE CLUSTER COPY ITSELF, over ssh (`verify_cluster_copy`; a re-run and the filing re-hashed the grading-host
mirror and read cluster.json, and never touched the copy the record names — a cluster copy removed, altered or on a host that did not
answer passed as "on record"): `<the guard under the recorded cluster root> && cd <cluster_path> && sha256sum -c --quiet MIRROR.sha256 &&
sha256sum MIRROR.sha256 mirror.json cluster.json && echo inventory: && find . -mindepth 1 -printf "%y %p\0"` runs THERE — the guard BEFORE
the `cd`, so a path that has become a symlink is never entered; the three records' digests read back must be the local mirror's
(MIRROR.sha256 and mirror.json by cluster.json's digests, cluster.json by the installed record's), and the inventory read back must be
exactly MIRROR.sha256's paths plus the three records as regular files and mirror.json's `directories` as directories, no other entry of any
type. A copy that is REDIRECTED (a symlink at the copy or at a component between the recorded root and it, or a copy whose
realpath is not its literal path under the root's), MISSING (no such directory or file there; a recorded directory or a listed regular file
not found as such), ALTERED (a sum that did not match, a record whose digest is not the local one), CONTAMINATED (an entry there that is not
the mirror's: a regular file in neither MIRROR.sha256 nor the records, a directory mirror.json does not list, or a symlink, fifo, socket or
device — nothing here deletes it) or UNREACHABLE (ssh exited 255, or could not run) refuses by name, the local mirror and its record left
as they are — an unreachable cluster refuses; it never passes. `07 --phase file` runs the same check before it composes.

THE GRADING-HOST PATH IS GUARDED TOO (`local_guard`, the cluster guard's twin): <state> is the anchor — it may be or lie
under a symlink, the site's layout, and its realpath anchors the comparison — and each component below it, 07_mirror then <receipt sha12>,
must not be a symlink and, once it exists, must resolve to the root's realpath plus its literal relative path. The guard runs BEFORE the
published mirror is followed on a re-run, before the pull, before the rename that publishes (and again once 07_mirror exists), and first
thing in `verified_mirror`, which the re-run and `07 --phase file` call — so a mirror moved elsewhere with a link left at
<state>/07_mirror/<sha12> (byte-identical through the link: `is_dir()`, `verify_published`, the local walk and the cluster re-check all
passed, and the filing certified the link's path as `extra.mirror.local_path`), a link at 07_mirror itself, or a first publication through
one (which laid the mirror wherever the link pointed) is REDIRECTED, refused by name with the link and its target (or the path and where it
resolves), nothing followed, moved or deleted: put the copy back at its path by hand, recorded, or move the whole state root and point
--state-root at it. `extra.mirror.local_path` is the literal <state>/07_mirror/<sha12> as before.

--dry-run prints the planned pulls, the rsync and the ssh check (DRY-RUN: lines in the shape of lib/common.sh's helpers) from the receipt
alone: no volume access, no secrets, nothing written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MIRROR_DIR = "07_mirror"
#: the pulls land here and are published as MIRROR_DIR/<sha12> only once every check passes; a refused build STAYS for inspection
MIRROR_BUILD_DIR = "07_mirror.building"
SUMS_NAME = "MIRROR.sha256"
MIRROR_NAME = "mirror.json"
CLUSTER_NAME = "cluster.json"
#: /2: mirror.json carries `directories`, every directory of the mirror, which the local and the cluster checks hold
#: the directory set to; a /1 mirror.json refuses by name (the mirror is run again)
MIRROR_SCHEMA = "stage07-mirror/2"
#: /2: cluster.json carries `cluster_root`, the reproduce root the destination was guarded under (every component between
#: it and the copy a plain directory), which the re-verifications guard under again; a /1 record refuses by name (the record is re-pushed)
CLUSTER_SCHEMA = "stage07-mirror-cluster/2"
CLUSTER_SCHEMA_PRE_GUARD = "stage07-mirror-cluster/1"
#: the guard's own exit codes on the cluster (round 6): 61 — a component between the root and the destination (or the destination) is a symlink;
#: 62 — the destination exists but its realpath is not its literal path under the root's realpath. Distinct from sha256sum's, cd's and ssh's.
GUARD_SYMLINK_RC = 61
GUARD_RESOLVE_RC = 62
GUARD_RCS = (GUARD_SYMLINK_RC, GUARD_RESOLVE_RC)
#: the launcher's per-store cell manifest (`cells/<store key>/manifest.json`; contracts documents its keys, not its filename)
CELL_MANIFEST = "manifest.json"
CELLS_DIR = "cells"
#: the pull's staging parent inside the build: the CLI lays a folder under its destination as <destination>/<folder name>/, and the store
#: root is then moved to <build>/<cell>/ (one rename)
PULL_STAGING = ".pull"
RECORDS = (SUMS_NAME, MIRROR_NAME, CLUSTER_NAME)
SIZE_CHECK = ("every pulled file's size rendered as the Modal CLI renders it (humanize_filesize: bytes exact below 1 KiB, one decimal above) equals "
              "the `modal volume ls --json` size string, and the file set pulled equals the file set listed; Modal exposes no content hash, so the "
              "byte-for-byte checks are the receipt's sha256 over the files it pulled and `sha256sum -c MIRROR.sha256` on the cluster copy")
DIR_CHECK = ("the directory set pulled under each store equals the directory set `modal volume ls --json` listed (one listing per directory, recursively: every "
             "directory reported once as a `type: dir` entry of its parent's listing, no implicit parents), so an empty directory the download omitted or a directory "
             "it created that was never listed refuses by name; `directories` is that listed set under each cell, confirmed on disk, plus the cells")
EXIT_REFUSED = 2
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=30")
RSYNC_FLAGS = ("-a", "--partial")
#: ssh's own exit status for a connection it could not make (the remote command's status otherwise): the cluster copy is UNREACHABLE, never "fine"
SSH_UNREACHABLE_RC = 255
#: the words sha256sum / the shell print when a listed file or the directory is not there (the copy is MISSING, not altered)
_MISSING_WORDS = ("No such file or directory", "could not be read", "FAILED open or read")
#: the line the remote check prints between the records' digests and the directory's inventory
INVENTORY_MARK = "inventory:"
#: the inventory of EVERY entry under the copy, by type (round 5): GNU find's `%y` type letter (f regular file, d directory, l symlink, p fifo,
#: s socket, b/c device), a space, the path (`./…`), NUL-terminated — a newline in a name cannot split or forge an entry. Double quotes, so
#: the remote shell hands find the `\0` and the DRY-RUN line pastes back whole. GNU findutils on the clusters and the grading host.
INVENTORY_CMD = 'find . -mindepth 1 -printf "%y %p\\0"'
_TYPE_WORDS = {"f": "regular file", "d": "directory", "l": "symlink", "p": "fifo", "s": "socket", "b": "block device", "c": "character device",
               "D": "door", "U": "unknown type"}


def type_word(letter: str) -> str:
    """`find -printf %y`'s letter as a word (an unlisted letter is shown as such)."""
    return _TYPE_WORDS.get(letter, f"type {letter!r}")


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"

ShRunner = Callable[..., "subprocess.CompletedProcess[str]"]


def ssh_argv(host: str, remote: str) -> List[str]:
    """The argv that runs `remote` on the cluster: `ssh -n <SSH_OPTS> <host> <remote>`, or `bash -c <remote>` under the `local:<dir>` seam."""
    if host.startswith("local:"):
        return ["bash", "-c", remote]
    return ["ssh", "-n", *SSH_OPTS, host, remote]


def dest_components(root: str, path: str) -> List[str]:
    """The directory names between the cluster's reproduce root and a destination under it, in order — what the guard walks. Refused by name
    when either is not absolute, when the path does not lie strictly under the root, or when a component is empty, `.`, `..` or carries a
    newline (a name the guard could not print whole on one line)."""
    r, p = str(root).rstrip("/"), str(path).rstrip("/")
    if not r.startswith("/"):
        raise Refusal(f"--cluster-root {root!r} is not an absolute path on the cluster (the site's expanded SISTEER_REPRODUCE_ROOT, under which --cluster-dest lies)")
    if not p.startswith(r + "/"):
        raise Refusal(f"the cluster destination {path} does not lie under the cluster root {root} — 07 passes <expanded SISTEER_REPRODUCE_ROOT> as the root and "
                      "<root>/<model>[/07_views/<view>]/07_mirror as the destination; nothing here guards a destination outside the root, so nothing is pushed")
    comps = p[len(r) + 1:].split("/")
    for c in comps:
        if c in ("", ".", "..") or "\n" in c or "\0" in c:
            raise Refusal(f"the cluster destination {path} has a component {c!r} under the root {root} that is not a plain directory name — nothing pushed")
    return comps


def guard_command(root: str, path: str) -> str:
    """What runs ON the cluster before anything enters or writes <path>: a walk from the reproduce root down every
    component of the destination — `$p` the path so far, `$s` the relative path so far — refusing (exit GUARD_SYMLINK_RC, the entry named with
    its target) the moment a component is a symlink, then, when the destination exists, its `realpath` held to the root's realpath plus the
    literal relative path (exit GUARD_RESOLVE_RC otherwise; a destination that does not exist yet passes this half — `mkdir -p` creates it, and
    the push's command runs the comparison again after). The root itself may be or lie under a symlink: its realpath is the anchor. Leaves
    `$r`, `$p` and `$s` set for what the caller chains after it. POSIX sh: `test`, `readlink`, `realpath` (coreutils on the clusters)."""
    comps = " ".join(shlex.quote(c) for c in dest_components(root, path))
    return (f'r={shlex.quote(str(root).rstrip("/"))}; p="$r"; s=""; for c in {comps}; do p="$p/$c"; s="$s/$c"; '
            f'! [ -L "$p" ] || {{ echo "SYMLINK: $p -> $(readlink -- "$p")"; exit {GUARD_SYMLINK_RC}; }}; done && {RESOLVE_CHECK}')


#: the resolved-path half of the guard (needs `$r`, `$p`, `$s` from guard_command's walk): nothing to compare while the destination is absent
RESOLVE_CHECK = ('{ ! [ -e "$p" ] || [ "$(realpath -- "$p")" = "$(realpath -- "$r")$s" ] || { echo "RESOLVES: $p -> $(realpath -- "$p"), not $(realpath -- "$r")$s"; '
                 f'exit {GUARD_RESOLVE_RC}; }}; }}')


def prepare_command(root: str, path: str) -> str:
    """What makes the destination before the push: the guard (no symlink between the root and the destination, nothing to resolve while it is
    absent), `mkdir -p`, then the resolved-path comparison on the directory that now exists — so a destination `mkdir -p` would have accepted as
    a symlink to elsewhere is refused by name and nothing is written through it."""
    return f"{guard_command(root, path)} && mkdir -p {shlex.quote(path)} && {RESOLVE_CHECK}"


def check_command(path: str, records: Sequence[str], *, root: str) -> str:
    """What runs ON the cluster to verify a copy: the guard under the reproduce root BEFORE the `cd` (a copy whose path has become a
    symlink is never entered), then every data file against the sums there, the named records' digests, then the directory's inventory of
    every entry by type after INVENTORY_MARK (`sha256sum -c` reads the listed paths only; a regular-file inventory never saw a symlink, a
    special file or a stray directory). Whatever is outside the mirror is read back here and refused by name, never deleted."""
    return (f"{guard_command(root, path)} && cd {shlex.quote(path)} && sha256sum -c --quiet {SUMS_NAME} && sha256sum {' '.join(records)} && echo {INVENTORY_MARK} && "
            f"{INVENTORY_CMD}")


def copy_check_command(path: str, *, root: str) -> str:
    """The re-check of a RECORDED copy (the re-run, the filing): the guard under the recorded root, the three records' digests and the inventory."""
    return check_command(path, RECORDS, root=root)


def guard_verdict(rc: int, text: str) -> Optional[str]:
    """The guard's own finding when the remote command exited with one of GUARD_RCS: the `SYMLINK: … -> …` or `RESOLVES: …` line it printed
    (the output's tail when the line is not there); None for any other exit status."""
    if rc not in GUARD_RCS:
        return None
    hit = [ln for ln in (text or "").splitlines() if ln.startswith(("SYMLINK: ", "RESOLVES: "))]
    return hit[-1] if hit else ((text or "").strip()[-300:] or f"exit {rc}, no output")


Entry = Tuple[str, str]      # (find's type letter, the path relative to the copy)


def _digest_lines(text: str) -> Dict[str, str]:
    return {ln.split()[-1]: ln.split()[0] for ln in text.splitlines() if re.match(r"[0-9a-f]{64}\s", ln)}


def parse_check_output(stdout: str) -> Tuple[Dict[str, str], Optional[List[Entry]]]:
    """(the records' digests by name, the inventory as (type letter, path relative to the copy) entries) from the remote check's stdout — the
    inventory None when the command never reached INVENTORY_MARK (it did not run whole), and a Refusal when what follows the mark is not what
    INVENTORY_CMD prints: NUL-terminated `<letter> ./<path>` records, the last one terminated (a cut-off inventory is not readable)."""
    text = stdout or ""
    mark = INVENTORY_MARK + "\n"
    if text.startswith(mark):
        at = 0
    else:
        i = text.find("\n" + mark)
        at = -1 if i < 0 else i + 1
    if at < 0:
        return _digest_lines(text), None
    digests, tail = _digest_lines(text[:at]), text[at + len(mark):]
    if not tail:
        return digests, []
    if not tail.endswith("\0"):
        raise Refusal(f"the remote check's inventory after `{INVENTORY_MARK}` does not end on a NUL-terminated entry ({tail[-120:]!r}) — the inventory is cut off and "
                      "is not readable")
    entries: List[Entry] = []
    for rec in tail[:-1].split("\0"):
        m = re.fullmatch(r"([A-Za-z?]) \./(.+)", rec, re.S)
        if not m:
            raise Refusal(f"the remote check printed an entry after `{INVENTORY_MARK}` that `{INVENTORY_CMD}` does not print: {rec[:120]!r} — the inventory is not readable")
        entries.append((m.group(1), m.group(2)))
    return digests, entries


def unexpected_entries(entries: Sequence[Entry], sums: Mapping[str, str], *, directories: Sequence[str], records: Sequence[str] = RECORDS) -> Tuple[List[str], List[str]]:
    """(the entries that are not the mirror's, each `<path> (<type>)`, sorted; the mirror's entries not found there as such, sorted) — a regular
    file is the mirror's when MIRROR.sha256 or the records name it, a directory when mirror.json's `directories` lists it, and a symlink, fifo,
    socket or device never is, whatever its name (round 5). The second list: a listed data file not present as a regular file (one replaced by a
    symlink reads back in both lists) and a recorded directory not present as a directory."""
    files_ok, dirs_ok = set(sums) | set(records), set(directories)
    unexpected = sorted(f"{p} ({type_word(t)})" for t, p in entries if not ((t == "f" and p in files_ok) or (t == "d" and p in dirs_ok)))
    files_there, dirs_there = {p for t, p in entries if t == "f"}, {p for t, p in entries if t == "d"}
    lacking = sorted(set(sums) - files_there) + sorted(f"{d} (directory)" for d in dirs_ok - dirs_there)
    return unexpected, lacking


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RT = _load_by_path("_stage07_mirror_receipt", HERE / "stage07_receipt.py")     # volume_rel (the checked conversion), Refusal, the stager M, read_json
M = RT.M
Refusal = RT.Refusal
volume_rel = RT.volume_rel
read_json = RT.read_json
sha256_file = RT.sha256_file


def _contracts():
    return RT._contracts()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sh_runner(argv: Sequence[str], **kw: Any) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(list(argv), capture_output=True, text=True, **kw)


# --------------------------------------------------------------------------------------------------------- the Modal CLI's size rendering
_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")


def humanize_filesize(value: int) -> str:
    """modal/cli/volume.py humanize_filesize (modal 1.5.4), verbatim: what `modal volume ls --json` prints as `size`."""
    if value < 0:
        raise ValueError("value should be >= 0")
    base = 1024
    size = float(value)
    for unit in _UNITS:
        if size < base:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= base
    return f"{size:.1f} ZiB"


def parse_filesize(text: str) -> int:
    """A rendered size back to (approximate) bytes, for the disk estimate: `157 B` -> 157, `53.6 KiB` -> 54886."""
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(B|KiB|MiB|GiB|TiB|PiB|EiB|ZiB)\s*", str(text))
    if not m:
        raise Refusal(f"modal volume ls --json reported a size {text!r} this tool cannot read")
    units = (*_UNITS, "ZiB")
    return int(round(float(m.group(1)) * (1024 ** units.index(m.group(2)))))


# ------------------------------------------------------------------------------------------------------------------------ the receipt
def read_receipt(path: Path) -> Tuple[dict, str]:
    """The published receipt and its sha256 — refused by name when absent or not a receipt carrying stores."""
    if not path.is_file():
        raise Refusal(f"no receipt at {path} — run `07 --phase receipt` first (the mirror pulls the stores the receipt names; nothing else is pulled)")
    doc = read_json(path, "receipt")
    schema = str(doc.get("schema") or "")
    if not schema.startswith("stage07-receipt/"):
        raise Refusal(f"{path} is not a stage 07 receipt (schema {doc.get('schema')!r})")
    for k in ("recs", "tree_sha", "pulled", "volume", "cells"):
        if k not in doc:
            raise Refusal(f"the receipt {path} carries no `{k}` — not a receipt this mirror can read; run --phase receipt again")
    if not isinstance(doc["recs"], dict) or not doc["recs"]:
        raise Refusal(f"the receipt {path} names no run cells (`recs`)")
    if not isinstance(doc["tree_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", doc["tree_sha"]):
        raise Refusal(f"the receipt's tree_sha {doc.get('tree_sha')!r} is not a 40-hex sha")
    return doc, sha256_file(path)


def plan_stores(receipt: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{cell: {out (container path), vrel (volume-root path), keys (the store keys the receipt credits)}} — from the receipt alone."""
    plan: Dict[str, Dict[str, Any]] = {}
    seen_out: Dict[str, str] = {}
    for cell, rec in receipt["recs"].items():
        out = rec.get("out") if isinstance(rec, dict) else None
        if not isinstance(out, str) or not out:
            raise Refusal(f"the receipt's recs[{cell!r}] names no store root (`out`)")
        if out in seen_out:
            raise Refusal(f"the receipt names one store root {out} for two cells ({seen_out[out]}, {cell}) — one store is one cell's")
        seen_out[out] = cell
        done_in = rec.get("done_in") if isinstance(rec, dict) else None
        if not isinstance(done_in, dict) or not done_in or not all(isinstance(v, str) and v for v in done_in.values()):
            raise Refusal(f"the receipt's recs[{cell!r}] credits no store keys (`done_in`)")
        if "/" in cell or cell in (".", "..") or not cell:
            raise Refusal(f"the receipt's cell {cell!r} is not a directory name")
        plan[cell] = {"out": out, "vrel": volume_rel(out), "keys": sorted(dict.fromkeys(done_in.values()))}
    return plan


def receipt_files_by_cell(receipt: Mapping[str, Any], plan: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, dict]]:
    """The receipt's `pulled` entries placed under their store: {cell: {path relative to the store root: {sha256, bytes, container}}}.
    A pulled file under no store root refuses (the receipt pulled it from somewhere this mirror does not copy)."""
    out: Dict[str, Dict[str, dict]] = {c: {} for c in plan}
    for cpath, rec in (receipt.get("pulled") or {}).items():
        home = [c for c, p in plan.items() if cpath.startswith(p["out"].rstrip("/") + "/")]
        if len(home) != 1:
            raise Refusal(f"the receipt pulled {cpath}, which lies under {'no' if not home else 'more than one'} store root the receipt names ({[p['out'] for p in plan.values()]})")
        cell = home[0]
        rel = cpath[len(plan[cell]["out"].rstrip("/")) + 1:]
        if not isinstance(rec, dict) or not isinstance(rec.get("sha256"), str):
            raise Refusal(f"the receipt's pulled[{cpath!r}] carries no sha256")
        out[cell][rel] = {"sha256": rec["sha256"], "bytes": rec.get("bytes"), "container": cpath}
    return out


# ------------------------------------------------------------------------------------------------------------------ the volume listing
def list_tree(cli, vrel: str) -> Tuple[Dict[str, str], List[str]]:
    """Every file under a volume directory, recursively, by one `modal volume ls --json` per directory: ({path relative to vrel: the CLI's
    size string}, [relative directories]). The root must list (a store root that does not exist refuses by name)."""
    root = vrel.strip("/")
    files: Dict[str, str] = {}
    dirs: List[str] = []
    todo = [root]
    while todo:
        d = todo.pop(0)
        try:
            entries = cli.ls_json(d)
        except Refusal as e:
            raise Refusal(f"cannot list /{d}: {e}") from None
        for e in entries:
            name = str(e.get("filename") or "")
            if not name:
                continue
            rel = name.strip("/")
            if rel.startswith(root + "/"):
                rel = rel[len(root) + 1:]
            elif rel == root:
                continue
            else:                                   # the CLI lists paths from the volume root; a bare name is relative to the listed directory
                rel = f"{d[len(root) + 1:]}/{rel}".strip("/") if d != root else rel
            if e.get("type") == "dir":
                dirs.append(rel)
                todo.append(f"{root}/{rel}")
            elif e.get("type") == "file":
                files[rel] = str(e.get("size") or "")
            else:                                   # a link, FIFO, socket or unknown entry: `volume get` would not reproduce it, so no copy could be certified complete
                raise Refusal(f"/{root}/{rel} is listed as {e.get('type')!r}, neither a file nor a directory: the mirror copies files and "
                              "directories only, so a store holding it cannot be certified complete; nothing is pulled")
    return files, sorted(dict.fromkeys(dirs))


def estimate_bytes(listing: Mapping[str, str]) -> int:
    return sum(parse_filesize(s) for s in listing.values())


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return shutil.disk_usage(str(p)).free


def _rmdir_quiet(d: Path) -> None:
    """Remove an EMPTY directory (the build parent once nothing of it remains); anything inside — another receipt's build, a staged record — stays."""
    try:
        os.rmdir(d)
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------------------------------ the checks
def _entry_kind(p: Path) -> str:
    mode = os.lstat(p).st_mode
    for test, word in ((stat.S_ISFIFO, "fifo"), (stat.S_ISSOCK, "socket"), (stat.S_ISBLK, "block device"), (stat.S_ISCHR, "character device")):
        if test(mode):
            return word
    return f"mode {oct(stat.S_IFMT(mode))}"


def walk_tree(root: Path) -> Tuple[Dict[str, Path], List[str]]:
    """({path relative to root (posix): the file}, [every directory relative to root, sorted]) — regular files in directories and nothing else:
    a symlink (to anything) or a special file (fifo, socket, device) anywhere under root refuses by name. The Modal CLI writes files and
    directories only, so any other entry is not the store's; and the cluster copy is held to the same two kinds (round 5)."""
    files: Dict[str, Path] = {}
    dirs: List[str] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            raise Refusal(f"{p} is a symlink — a store holds regular files in directories and nothing else; nothing is published")
        if p.is_dir():
            dirs.append(rel)
        elif p.is_file():
            files[rel] = p
        else:
            raise Refusal(f"{p} is a {_entry_kind(p)}, neither a regular file nor a directory — a store holds regular files in directories and nothing else; "
                          "nothing is published")
    return files, sorted(dirs)


def walk_files(root: Path) -> Dict[str, Path]:
    """{path relative to root (posix): the file}, every regular file under root (walk_tree's files; its refusals included)."""
    return walk_tree(root)[0]


def result_tokens(line: str) -> Dict[str, str]:
    return {k: v for k, v in (t.split("=", 1) for t in line.split() if "=" in t)}


def check_store(cell: str, store: Path, *, keys: Sequence[str], tree_sha: str, receipt_files: Mapping[str, dict], listing: Mapping[str, str],
                listed_dirs: Sequence[str], C) -> Dict[str, Any]:
    """Every check on one pulled store (the docstring's list); returns the per-cell record for mirror.json: {files: {rel: {sha256, bytes}},
    bytes, manifests, result_last_line, receipt_files_checked, directories (the store's directory set, the listing's confirmed on disk)}.
    `listed_dirs` is every directory `modal volume ls --json` reported under the store root (list_tree; round 6): the pulled tree must hold
    exactly those directories — a listed one the download did not create (an empty directory omitted) or one it created that no listing
    reported refuses by name, before the file set is read."""
    files, dirs = walk_tree(store)
    if not files:
        raise Refusal(f"{cell}: the pull of {store} holds no file")
    # (0) the directory set: the listing's, exactly (the CLI creates a directory for every DIRECTORY entry it downloads, an empty one included)
    d_missing = sorted(set(listed_dirs) - set(dirs))
    d_extra = sorted(set(dirs) - set(listed_dirs))
    if d_missing or d_extra:
        raise Refusal(f"{cell}: the pulled directory set is not the listed one (listed on the volume but not pulled: {d_missing[:8]}{'…' if len(d_missing) > 8 else ''}; pulled but "
                      f"not listed: {d_extra[:8]}{'…' if len(d_extra) > 8 else ''}) — `modal volume ls --json` reported every directory under {cell}'s store root and the "
                      "CLI's download creates one for every DIRECTORY entry, so the two sets agree for the store as the volume holds it; a directory listed but absent "
                      "from the pull, or present in the pull but never listed, is not that store; nothing published")
    hashes = {rel: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for rel, p in files.items()}
    # (1) the receipt's files, byte for byte
    for rel, want in sorted(receipt_files.items()):
        got = hashes.get(rel)
        if got is None:
            raise Refusal(f"{cell}: the receipt pulled {want['container']} (sha {want['sha256'][:12]}) but the mirrored store has no {rel} — the store on the volume is "
                          "not the one the receipt described; nothing published")
        if got["sha256"] != want["sha256"]:
            raise Refusal(f"{cell}: {rel} hashes to {got['sha256'][:12]} in the mirror, the receipt recorded {want['sha256'][:12]} for {want['container']} — the bytes "
                          "differ from the receipt's; nothing published")
    # (2) the credited store keys' manifests
    manifests: Dict[str, dict] = {}
    for key in keys:
        rel = f"{CELLS_DIR}/{key}/{CELL_MANIFEST}"
        if rel not in files:
            raise Refusal(f"{cell}: the receipt credits store key {key!r} but the mirrored store has no {rel} (cells/ holds "
                          f"{sorted({r.split('/', 2)[1] for r in files if r.startswith(CELLS_DIR + '/') and r.count('/') >= 2})}) — nothing published")
        man = read_json(files[rel], f"{cell} {rel}")
        producer = man.get(C.MANIFEST_KEY_PRODUCER)
        got_tree = producer.get("tree_sha") if isinstance(producer, dict) else None
        if got_tree != tree_sha:
            raise Refusal(f"{cell}: {rel} producer.tree_sha {str(got_tree)[:12]!r} is not the receipt's tree {tree_sha[:12]} — the store was produced at another tree; "
                          "nothing published")
        if man.get("cell_id") != cell:
            raise Refusal(f"{cell}: {rel} cell_id {man.get('cell_id')!r} is not the cell {cell!r} — nothing published")
        manifests[key] = {"path": rel, "producer": dict(producer), "cell_id": man.get("cell_id"), "harness_cell_id": man.get("harness_cell_id"),
                          "intervention_hash": man.get("intervention_hash"), "harness_config_hash": man.get("harness_config_hash"), "sha256": hashes[rel]["sha256"]}
    # (3) RESULT's last line
    if C.RESULT_FILENAME not in files:
        raise Refusal(f"{cell}: the mirrored store has no {C.RESULT_FILENAME} — not a finished store; nothing published")
    lines = [ln for ln in files[C.RESULT_FILENAME].read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise Refusal(f"{cell}: {C.RESULT_FILENAME} is empty — nothing published")
    last = lines[-1]
    toks = result_tokens(last)
    if toks.get("result") != C.RESULT_COMPLETE:
        raise Refusal(f"{cell}: the last {C.RESULT_FILENAME} line does not read result={C.RESULT_COMPLETE} ({last[:200]!r}) — nothing published")
    if "sha" in toks and toks["sha"] != tree_sha:
        raise Refusal(f"{cell}: the last {C.RESULT_FILENAME} line names sha={toks['sha'][:12]}, not the receipt's tree {tree_sha[:12]} ({last[:200]!r}) — nothing published")
    # (4) the listing: the same file set, the same rendered sizes
    missing = sorted(set(listing) - set(files))
    extra = sorted(set(files) - set(listing))
    if missing or extra:
        raise Refusal(f"{cell}: the pulled file set is not the listed one (listed but not pulled: {missing[:8]}{'…' if len(missing) > 8 else ''}; pulled but not listed: "
                      f"{extra[:8]}{'…' if len(extra) > 8 else ''}) — nothing published")
    for rel, size in sorted(listing.items()):
        rendered = humanize_filesize(hashes[rel]["bytes"])
        if rendered != size.strip():
            raise Refusal(f"{cell}: {rel} is {hashes[rel]['bytes']} B locally ({rendered}), the volume listing says {size!r} — nothing published")
    return {"files": hashes, "bytes": sum(h["bytes"] for h in hashes.values()), "manifests": manifests, "result_last_line": last,
            "receipt_files_checked": len(receipt_files), "directories": sorted(dirs)}


def write_sums(mirror: Path, per_cell: Mapping[str, Mapping[str, Any]]) -> Path:
    """MIRROR.sha256: sha256sum's format (`<sha256>  <path>`), every data file, the paths relative to the mirror root and sorted."""
    rows = {f"{cell}/{rel}": h["sha256"] for cell, rec in per_cell.items() for rel, h in rec["files"].items()}
    p = mirror / SUMS_NAME
    p.write_text("".join(f"{rows[path]}  {path}\n" for path in sorted(rows)), encoding="utf-8")
    return p


def read_sums(p: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        m = re.fullmatch(r"([0-9a-f]{64})  (.+)", ln)
        if not m:
            raise Refusal(f"{p}: a line this tool did not write: {ln[:120]!r}")
        out[m.group(2)] = m.group(1)
    return out


def mirror_directories(mj: Mapping[str, Any], mj_p: Path) -> List[str]:
    """mirror.json's `directories` (every directory of the mirror, relative, sorted) — refused by name when the record carries none (a mirror
    of the earlier schema) or carries something else."""
    dirs = mj.get("directories")
    if not isinstance(dirs, list) or not all(isinstance(d, str) and d and not d.startswith("/") for d in dirs):
        raise Refusal(f"{mj_p} records no `directories` list ({str(dirs)[:80]!r}) — not a mirror.json of schema {MIRROR_SCHEMA}; move the mirror aside (named in "
                      "decisions.md) and run the mirror again")
    return sorted(dirs)


def verify_published(mirror: Path, *, receipt_sha: str) -> Tuple[dict, Dict[str, str]]:
    """The published copy: every data file hashes to MIRROR.sha256's line and nothing is listed that is absent or present that is unlisted;
    the directories are exactly mirror.json's `directories` (round 5: nothing extra, nothing missing); no symlink or special file anywhere
    (walk_tree); mirror.json is this receipt's. Returns (mirror.json, the sums)."""
    sums_p, mj_p = mirror / SUMS_NAME, mirror / MIRROR_NAME
    for p in (sums_p, mj_p):
        if not p.is_file():
            raise Refusal(f"{mirror} is published but holds no {p.name} — not a mirror this tool wrote; move it aside and run the mirror again")
    mj = read_json(mj_p, "mirror.json")
    if mj.get("schema") != MIRROR_SCHEMA:
        raise Refusal(f"{mj_p} schema {mj.get('schema')!r} is not {MIRROR_SCHEMA!r} — a mirror written by an earlier cut of this tool (before the directory inventory, "
                      f"code review round 5) or by something else; move {mirror} aside (named in decisions.md) and run the mirror again, which pulls the stores afresh")
    if mj.get("receipt_sha256") != receipt_sha:
        raise Refusal(f"{mj_p} records receipt {str(mj.get('receipt_sha256'))[:12]}, this receipt is {receipt_sha[:12]} — the published directory is named for another receipt's "
                      "first twelve hex (or was moved); nothing filed against it")
    sums = read_sums(sums_p)
    want_dirs = mirror_directories(mj, mj_p)
    all_files, dirs = walk_tree(mirror)
    files = {rel: p for rel, p in all_files.items() if rel not in RECORDS}
    missing = sorted(set(sums) - set(files))
    extra = sorted(set(files) - set(sums))
    if missing or extra:
        raise Refusal(f"the published mirror {mirror} is not whole (listed in {SUMS_NAME} but absent: {missing[:8]}; present but unlisted: {extra[:8]}) — do not file against it")
    d_missing = sorted(set(want_dirs) - set(dirs))
    d_extra = sorted(set(dirs) - set(want_dirs))
    if d_missing or d_extra:
        raise Refusal(f"the published mirror {mirror} is not whole in its directories (listed in {MIRROR_NAME} but absent: {d_missing[:8]}; present but unlisted: {d_extra[:8]}) — "
                      "a directory was added to or removed from the copy after it was verified; do not file against it")
    for rel, p in sorted(files.items()):
        got = sha256_file(p)
        if got != sums[rel]:
            raise Refusal(f"the published mirror {mirror}: {rel} hashes to {got[:12]}, {SUMS_NAME} says {sums[rel][:12]} — the copy changed after it was verified; do not file against it")
    return mj, sums


#: the realpath the local guard anchors on and compares with (a seam: a test stands in for a mount that resolves a plain directory elsewhere)
_realpath = os.path.realpath


def local_guard(state: Path, sha12: str) -> Path:
    """The grading-host mirror's path, <state>/07_mirror/<sha12>, held PLAIN under the state root before anything follows it or records it
    (the cluster guard's twin: `mirror.is_dir()`, `verify_published` and the local walk all followed a symlink at <sha12>
    or at 07_mirror, and the cluster re-check passed too, so a mirror moved elsewhere with a link left at its path re-ran and FILED as the copy
    at the recorded location, and a first publication through a symlinked 07_mirror laid the mirror wherever the link pointed while local_path
    named the link). The state root is the anchor — it may be or lie under a symlink (the site's layout; its realpath anchors the comparison) —
    and each component below it, 07_mirror then <sha12>, must not be a symlink and, once it exists, must resolve to the root's realpath plus
    its literal relative path. REDIRECTED refuses by name (SYMLINK: the link and its target; RESOLVES: the path and where it resolves); nothing
    here follows, moves or deletes anything. Returns the literal mirror path."""
    state = Path(state)
    literal = state / MIRROR_DIR / sha12
    anchor = _realpath(state)
    remedy = (f"the state root {state} is the anchor (it may itself be a symlink) and every component below it through {MIRROR_DIR}/{sha12} must be a plain "
              "directory; nothing here follows, moves or deletes it: inspect it, put the copy back at its path by hand (recorded in decisions.md) — or move "
              "the whole state root and point --state-root at it — and run again")
    p, rel = state, ""
    for comp in (MIRROR_DIR, sha12):
        p, rel = p / comp, f"{rel}/{comp}"
        if p.is_symlink():
            raise Refusal(f"the grading-host mirror of receipt {sha12} at {literal} is REDIRECTED: SYMLINK: {p} -> {os.readlink(p)} — the recorded path names a link, "
                          "not the copy at that location (a filing would certify the link's path as the durable copy, and retargeting or removing the link later "
                          f"would lose or change what it points to); {remedy}")
        if p.exists():
            real = _realpath(p)
            if real != f"{anchor}{rel}":
                raise Refusal(f"the grading-host mirror of receipt {sha12} at {literal} is REDIRECTED: RESOLVES: {p} -> {real}, not {anchor}{rel} — the directory's realpath "
                              f"is not its literal path under the state root's realpath, so the recorded path does not name the copy at that location; {remedy}")
    return literal


def record_staging_path(mirror: Path) -> Path:
    """Where a cluster record is STAGED before its transfer verifies: <state>/07_mirror.building/<sha12>.cluster.json — outside the published
    mirror, so a record whose push did not finish is never read as one that did (a leftover here is overwritten by the next run's push)."""
    mirror = Path(mirror)
    return mirror.parent.parent / MIRROR_BUILD_DIR / f"{mirror.name}.{CLUSTER_NAME}"


def verified_mirror(state: Path, receipt_sha: str) -> Dict[str, Any]:
    """What `07 --phase file` reads: the published mirror of this receipt WITH its cluster record, both consistent — the pointer the filing
    records ({local_path, cluster_label, ssh_host, cluster_path, mirror_sha256sum_digest, mirror_json_sha256, receipt_sha256, files, bytes,
    tree_sha, verified_utc}). Refuses by name when the mirror or its cluster record is absent or inconsistent. The WHOLE copy is verified
    here (header: WHAT A RE-RUN WITH A CLUSTER RECORD CHECKS): FIRST the path held plain under the state root (`local_guard`, round 7: a
    symlink at 07_mirror or at <sha12> is REDIRECTED before anything follows it), then every data file against MIRROR.sha256
    (`verify_published`), mirror.json this receipt's, and cluster.json's digests of both records equal to the files beside it — so a store byte
    or a record changed after the cluster check cannot be filed as the verified copy."""
    mirror = local_guard(state, receipt_sha[:12])         # round 7: the path is held plain under the state root BEFORE it is followed or recorded
    if not mirror.is_dir():
        raise Refusal(f"no mirror of receipt {receipt_sha[:12]} at {mirror}")
    sums_p, mj_p, cl_p = mirror / SUMS_NAME, mirror / MIRROR_NAME, mirror / CLUSTER_NAME
    for p in (sums_p, mj_p):
        if not p.is_file():
            raise Refusal(f"{mirror} holds no {p.name} — not a published mirror")
    if not cl_p.is_file():
        raise Refusal(f"the mirror at {mirror} has no {CLUSTER_NAME}: its cluster copy was never verified (the push or its `sha256sum -c` did not finish)")
    mj, sums = verify_published(mirror, receipt_sha=receipt_sha)                 # every data file re-hashed; nothing extra, nothing missing
    cl = read_json(cl_p, "cluster.json")
    digest, mj_sha = sha256_file(sums_p), sha256_file(mj_p)
    if cl.get("schema") == CLUSTER_SCHEMA_PRE_GUARD:
        raise Refusal(f"{cl_p} is a cluster record of schema {CLUSTER_SCHEMA_PRE_GUARD!r}, written before the destination was guarded against a symlink at the copy or at a "
                      f"component under the reproduce root (code review round 6; this tool writes {CLUSTER_SCHEMA!r}, which records the root the copy was guarded under): "
                      f"record it in decisions.md, move {cl_p} OUT of the mirror directory (anywhere else, named in the note) and run `07 --phase mirror` again, which "
                      "guards the recorded destination, re-checks the copy there and pushes a new record over it — nothing is re-pulled and the earlier copy is never deleted")
    if cl.get("schema") != CLUSTER_SCHEMA or cl.get("receipt_sha256") != receipt_sha or cl.get("mirror_sha256sum_digest") != digest or cl.get("verified") is not True:
        raise Refusal(f"{cl_p} does not record a verified cluster copy of this mirror (schema {cl.get('schema')!r}, receipt {str(cl.get('receipt_sha256'))[:12]}, "
                      f"digest {str(cl.get('mirror_sha256sum_digest'))[:12]} vs {SUMS_NAME}'s {digest[:12]}, verified {cl.get('verified')!r})")
    if cl.get("mirror_json_sha256") != mj_sha:
        raise Refusal(f"{cl_p} records {MIRROR_NAME} as {str(cl.get('mirror_json_sha256'))[:12]}, the {MIRROR_NAME} beside it hashes to {mj_sha[:12]} — the mirror record "
                      "changed after the cluster copy was verified; not a verified mirror (restore the record, or move cluster.json aside and run the mirror again)")
    if cl.get("tree_sha") != mj.get("tree_sha") or cl.get("files") != mj.get("files") or len(sums) != mj.get("files"):
        raise Refusal(f"{cl_p} and {mj_p} disagree (tree {str(cl.get('tree_sha'))[:12]} vs {str(mj.get('tree_sha'))[:12]}; files {cl.get('files')} vs {mj.get('files')}; "
                      f"{SUMS_NAME} lists {len(sums)}) — not a verified mirror")
    root, cpath = cl.get("cluster_root"), cl.get("cluster_path")
    if not isinstance(root, str) or not root.startswith("/") or not isinstance(cpath, str):
        raise Refusal(f"{cl_p} records no absolute `cluster_root` ({root!r}) — not a record of schema {CLUSTER_SCHEMA}; the re-verification guards the copy's path under that root "
                      f"and cannot without it (move {cl_p} out of the mirror, recorded, and run the mirror again)")
    try:
        dest_components(root, cpath)
    except Refusal as e:
        raise Refusal(f"{cl_p}: the recorded cluster_path does not lie plainly under the recorded cluster_root ({e}) — not a verified mirror") from None
    return {"local_path": str(mirror), "cluster_label": cl.get("cluster_label"), "ssh_host": cl.get("ssh_host"), "cluster_root": root, "cluster_path": cpath,
            "mirror_sha256sum_digest": digest, "mirror_json_sha256": mj_sha, "receipt_sha256": receipt_sha, "files": mj.get("files"), "bytes": mj.get("bytes"),
            "tree_sha": mj.get("tree_sha"), "verified_utc": cl.get("verified_utc"), "volume": mj.get("volume"), "modal_environment": mj.get("modal_environment"),
            "store_volume_paths": sorted(c.get("volume_path") for c in (mj.get("cells") or {}).values())}


#: what names a cluster target: the label, the ssh host, the reproduce root the destination is guarded under, and <dest>/<sha12>
TARGET_KEYS = ("cluster_label", "ssh_host", "cluster_root", "cluster_path")


def target_of(cluster: "Cluster", sha12: str) -> Dict[str, str]:
    return {"cluster_label": cluster.label, "ssh_host": cluster.host, "cluster_root": cluster.root, "cluster_path": f"{cluster.dest}/{sha12}"}


def check_target(ptr: Mapping[str, Any], target: Mapping[str, str], *, what: str) -> None:
    """The recorded cluster target against the one a call names — refused by name on any difference (a mirror has one cluster record); the
    root among them (round 6): the re-verification guards the copy under the RECORDED root, so a site whose root moved names another target."""
    diffs = [(k, ptr.get(k), target.get(k)) for k in TARGET_KEYS if ptr.get(k) != target.get(k)]
    if diffs:
        shown = "; ".join(f"{k}: recorded {a!r}, this call {b!r}" for k, a, b in diffs)
        raise Refusal(f"the cluster record of receipt {str(ptr.get('receipt_sha256'))[:12]}'s mirror names another target than {what} ({shown}) — the verified copy sits "
                      f"where {CLUSTER_NAME} says, {ptr.get('cluster_label')} {ptr.get('ssh_host')}:{ptr.get('cluster_path')}, and nothing was copied to this call's "
                      "target; a mirror has ONE cluster record: run with the recorded target, or — when the site's destination has moved — record that in "
                      f"decisions.md, move {ptr.get('local_path')}/{CLUSTER_NAME} OUT of the mirror directory (anywhere else, named in the note) and run "
                      "`07 --phase mirror` again to push the published mirror to the new destination (the earlier copy is never deleted here)")


def verify_cluster_copy(ptr: Mapping[str, Any], *, runner: ShRunner = sh_runner, log=print) -> Dict[str, Any]:
    """The CLUSTER copy a verified mirror's record names, re-verified THERE before it is trusted (header: THEN THE CLUSTER COPY ITSELF —
    a re-run and the filing rehashed the grading-host mirror and read cluster.json, never the copy): `copy_check_command` over
    the recorded ssh host at the recorded cluster path, and the three records' digests read back held to the local mirror's (`ptr` as
    `verified_mirror` returns it: MIRROR.sha256 by `mirror_sha256sum_digest`, mirror.json by `mirror_json_sha256`, cluster.json by the installed
    record under `local_path`), and the copy's inventory of every entry by type read back and held to the mirror — the regular files to the
    local MIRROR.sha256's paths plus the three records, the directories to the local mirror.json's `directories`, and no symlink, fifo, socket
    or device. Refuses by name — UNREACHABLE when ssh exits 255 or cannot run (never a pass), MISSING when the directory, a
    listed file or a recorded directory is not there as such, ALTERED when a sum did not match or a record's digest is not the local one,
    CONTAMINATED when an entry there is not the mirror's (nothing here deletes it). Returns the facts of the check (the host, the path, the
    command, the file count, the inventory's sizes, the time); nothing is written anywhere."""
    host, path, root = str(ptr.get("ssh_host") or ""), str(ptr.get("cluster_path") or ""), str(ptr.get("cluster_root") or "")
    if not host or not path.startswith("/"):
        raise Refusal(f"the cluster record names no ssh host / absolute cluster path to check ({host!r}, {path!r})")
    if not root.startswith("/"):
        raise Refusal(f"the cluster record names no absolute cluster root to guard {path} under ({root!r}) — the copy is checked under the root it was pushed under, or not at all")
    local = Path(str(ptr.get("local_path")))
    cl_local = local / CLUSTER_NAME
    if not cl_local.is_file():
        raise Refusal(f"{cl_local} is not a file — the installed cluster record the copy's is held to is gone")
    for name in (SUMS_NAME, MIRROR_NAME):
        if not (local / name).is_file():
            raise Refusal(f"{local / name} is not a file — the record the copy's inventory is held to is gone")
    sums = read_sums(local / SUMS_NAME)
    directories = mirror_directories(read_json(local / MIRROR_NAME, "mirror.json"), local / MIRROR_NAME)
    want = {SUMS_NAME: str(ptr.get("mirror_sha256sum_digest")), MIRROR_NAME: str(ptr.get("mirror_json_sha256")), CLUSTER_NAME: sha256_file(cl_local)}
    check = copy_check_command(path, root=root)
    where = f"{ptr.get('cluster_label')} {host}:{path}"
    log(f"  {ssh_shown(host, check)}")
    try:
        r = runner(ssh_argv(host, check))
    except OSError as e:
        raise Refusal(f"the cluster copy at {where} is UNREACHABLE: {ssh_argv(host, check)[0]} could not run ({e}) — an unreachable copy is never taken as verified") from None
    text = ((r.stderr or "") + (r.stdout or "")).strip()
    if r.returncode == SSH_UNREACHABLE_RC:
        raise Refusal(f"the cluster copy at {where} is UNREACHABLE: ssh exited {r.returncode} ({text[-300:] or 'no output'}) — an unreachable copy is never taken as "
                      "verified; nothing here changes until the host answers")
    redirected = guard_verdict(r.returncode, r.stdout or "")
    if redirected is not None:
        raise Refusal(f"the cluster copy at {where} is REDIRECTED: {redirected} — the copy itself, or a component between the recorded cluster root {root} and it, is a "
                      "symlink (or the copy's realpath is not its literal path under the root's), so the recorded path no longer names the directory the push verified; "
                      f"the check never entered it and nothing here follows or deletes the link: inspect it, move it OUT by hand (recorded in decisions.md) so every "
                      f"component under {root} is a plain directory, and run the check again (the local mirror at {ptr.get('local_path')} stands and verifies; its "
                      "cluster record is left as it is)")
    if r.returncode != 0:
        kind = "MISSING" if any(w in text for w in _MISSING_WORDS) else "ALTERED"
        raise Refusal(f"the cluster copy at {where} is {kind}: `{check}` exited {r.returncode} there ({text[-400:] or 'no output'}) — the copy on record is not "
                      f"the verified mirror any more (the local mirror at {ptr.get('local_path')} stands and verifies; its cluster record is left as it is)")
    got, entries = parse_check_output(r.stdout or "")
    diffs = [(n, got.get(n), want[n]) for n in (SUMS_NAME, MIRROR_NAME, CLUSTER_NAME) if got.get(n) != want[n]]
    if diffs:
        shown = "; ".join(f"{n} there {str(a)[:12]}, here {b[:12]}" for n, a, b in diffs)
        raise Refusal(f"the cluster copy at {where} is ALTERED: its records' digests are not the local mirror's ({shown}) — the copy on record is not the verified "
                      f"mirror any more (the local mirror at {ptr.get('local_path')} stands and verifies; its cluster record is left as it is)")
    if entries is None:
        raise Refusal(f"the cluster copy at {where} could not be inventoried: `{check}` exited 0 there but printed no `{INVENTORY_MARK}` line — the check did not run whole; "
                      "nothing is taken as verified")
    unexpected, lacking = unexpected_entries(entries, sums, directories=directories, records=RECORDS)
    stands = f"(the local mirror at {ptr.get('local_path')} stands and verifies; its cluster record is left as it is)"
    if unexpected:
        raise Refusal(f"the cluster copy at {where} is CONTAMINATED: {_n(len(unexpected), 'entry', 'entries')} there {'is' if len(unexpected) == 1 else 'are'} not the mirror's "
                      f"({unexpected[:8]}{'…' if len(unexpected) > 8 else ''}) — a regular file in neither {SUMS_NAME} nor the records, a directory {MIRROR_NAME} does not list, "
                      f"or a symlink, fifo, socket or device, which the mirror never holds; the copy holds entries outside the recorded snapshot and is not the verified mirror "
                      f"as it stands; nothing here deletes them: inspect the copy, move the unexpected entries OUT by hand (recorded in decisions.md) and run the check again {stands}")
    if lacking:
        raise Refusal(f"the cluster copy at {where} is MISSING {_n(len(lacking), 'entry', 'entries')} of the mirror ({lacking[:8]}{'…' if len(lacking) > 8 else ''}) — a listed "
                      f"file not there as a regular file, or a recorded directory not there; the copy on record is not the verified mirror any more {stands}")
    n_files, n_dirs = sum(1 for t, _p in entries if t == "f"), sum(1 for t, _p in entries if t == "d")
    return {"ssh_host": host, "cluster_path": path, "cluster_label": ptr.get("cluster_label"), "check_command": check, "files": ptr.get("files"), "inventory": n_files,
            "directories": n_dirs, "checked_utc": utc_now()}


def ssh_shown(host: str, remote: str) -> str:
    return f"ssh {host} '{remote}'"


# ----------------------------------------------------------------------------------------------------------------- the cluster copy
class Cluster:
    """rsync + ssh to the site's cluster, or the `local:<dir>` seam (lib/common.sh repro_ssh: `bash -c` here, the rsync target a local path)."""

    def __init__(self, label: str, host: str, dest: str, *, root: Optional[str] = None, runner: ShRunner = sh_runner, log=print):
        if not host:
            raise Refusal(f"cluster {label} has no CLUSTER_SSH — the mirror has nowhere to push")
        if not dest or not dest.startswith("/"):
            raise Refusal(f"--cluster-dest {dest!r} is not an absolute path on the cluster")
        if not root or not str(root).startswith("/"):
            raise Refusal(f"--cluster-root {root!r} is not an absolute path on the cluster — the site's expanded SISTEER_REPRODUCE_ROOT, under which every component of "
                          "--cluster-dest/<receipt sha12> is guarded against a symlink before the push (07 passes it); nothing pushed without it")
        self.label, self.host, self.dest, self.root, self.runner, self.log = label, host, dest.rstrip("/"), str(root).rstrip("/"), runner, log
        if self.dest != self.root:
            dest_components(self.root, self.dest)             # the destination is the root or lies plainly under it (the guard walks <dest>/<sha12> from the root), or nothing runs
        self.local = host.startswith("local:")

    def ssh_argv(self, remote: str) -> List[str]:
        return ssh_argv(self.host, remote)

    def ssh_shown(self, remote: str) -> str:
        return ssh_shown(self.host, remote)

    def rsync_argv(self, src: Path, sub: str) -> List[str]:
        target = f"{self.dest}/{sub}/" if self.local else f"{self.host}:{self.dest}/{sub}/"
        argv = ["rsync", *RSYNC_FLAGS]
        if not self.local:
            argv += ["-e", "ssh " + " ".join(SSH_OPTS)]
        return [*argv, f"{src}/", target]

    def run(self, argv: Sequence[str], what: str) -> "subprocess.CompletedProcess[str]":
        try:
            r = self.runner(list(argv))
        except OSError as e:
            raise Refusal(f"{what}: {argv[0]} could not run ({e})") from None
        if r.returncode != 0:
            raise Refusal(f"{what} failed (rc {r.returncode}): {((r.stderr or '') + (r.stdout or '')).strip()[-600:]}")
        return r

    def check_command(self, sub: str) -> str:
        """The check after the data push: the guard under the root, the sums, the two records pushed with the mirror, the inventory (cluster.json
        is pushed afterwards)."""
        return check_command(f"{self.dest}/{sub}", (SUMS_NAME, MIRROR_NAME), root=self.root)

    def prepare_command(self, sub: str) -> str:
        """What makes <dest>/<sub> before the push: the guard, `mkdir -p`, the resolved-path comparison (prepare_command)."""
        return prepare_command(self.root, f"{self.dest}/{sub}")

    def prepare(self, sub: str) -> None:
        """The guarded mkdir (round 6): a symlink at the destination or at any component between the root and it, or a destination that resolves
        elsewhere, refuses by name BEFORE any rsync — `mkdir -p` would accept a symlink to a directory, the trailing-slash rsync would write
        through it, and the record would name the symlink path as the durable copy. Nothing is deleted or moved here."""
        path = f"{self.dest}/{sub}"
        cmd = self.prepare_command(sub)
        self.log(f"  {self.ssh_shown(cmd)}")
        try:
            r = self.runner(self.ssh_argv(cmd))
        except OSError as e:
            raise Refusal(f"ssh {self.host}: the guarded mkdir of {path}: {self.ssh_argv(cmd)[0]} could not run ({e})") from None
        redirected = guard_verdict(r.returncode, r.stdout or "")
        if redirected is not None:
            raise Refusal(f"the cluster destination {self.host}:{path} is REDIRECTED: {redirected} — the destination, or a component between the cluster root {self.root} "
                          "and it, is a symlink (or the destination's realpath is not its literal path under the root's); `mkdir -p` would accept it, rsync would write the "
                          "mirror through it to wherever it points, and the record would name the symlink path as the durable copy; nothing pushed, nothing recorded and "
                          "nothing deleted: inspect the link, move it OUT by hand (recorded in decisions.md) so every component under the root is a plain directory, then "
                          "run the mirror again")
        if r.returncode != 0:
            raise Refusal(f"ssh {self.host}: the guarded mkdir of {path} failed (rc {r.returncode}): {((r.stderr or '') + (r.stdout or '')).strip()[-600:]}")

    def push_and_check(self, mirror: Path, sub: str, *, sums: Mapping[str, str], mj_sha: str, directories: Sequence[str]) -> Dict[str, Any]:
        """The guarded mkdir (prepare), rsync the mirror, `sha256sum -c MIRROR.sha256` there behind the same guard, read back the two records'
        digests and the directory's inventory of every entry by type (rounds 4-5: the regular files held to the sums plus the three records — a
        cluster.json already there is the record of an earlier push whose install did not finish, replaced by push_record — the directories to
        mirror.json's `directories`, and no other kind of entry; anything else refuses by name and is never deleted). Returns the facts for
        cluster.json."""
        path = f"{self.dest}/{sub}"
        self.prepare(sub)
        argv = self.rsync_argv(mirror, sub)
        self.log(f"  {' '.join(shlex.quote(a) for a in argv)}")
        r = self.run(argv, f"rsync to {self.host}:{path}")
        check = self.check_command(sub)
        self.log(f"  {self.ssh_shown(check)}")
        what = f"the cluster check `sha256sum -c {SUMS_NAME}` on {self.host}:{path}"
        try:
            rc = self.runner(self.ssh_argv(check))
        except OSError as e:
            raise Refusal(f"{what}: {self.ssh_argv(check)[0]} could not run ({e})") from None
        redirected = guard_verdict(rc.returncode, rc.stdout or "")
        if redirected is not None:
            raise Refusal(f"the cluster destination {self.host}:{path} is REDIRECTED after the push: {redirected} — a symlink appeared at the destination or at a component "
                          f"between the cluster root {self.root} and it between the guarded mkdir and the check (or the destination resolves elsewhere); the check never "
                          "entered it; nothing recorded and nothing deleted: inspect the link, move it OUT by hand (recorded in decisions.md) and run the mirror again")
        if rc.returncode != 0:
            raise Refusal(f"{what} failed (rc {rc.returncode}): {((rc.stderr or '') + (rc.stdout or '')).strip()[-600:]}")
        digests, entries = parse_check_output(rc.stdout or "")
        local_sums = sha256_file(mirror / SUMS_NAME)
        if digests.get(SUMS_NAME) != local_sums or digests.get(MIRROR_NAME) != mj_sha:
            raise Refusal(f"the cluster copy's records differ from the published ones ({SUMS_NAME} {str(digests.get(SUMS_NAME))[:12]} vs {local_sums[:12]}; {MIRROR_NAME} "
                          f"{str(digests.get(MIRROR_NAME))[:12]} vs {mj_sha[:12]}) — the push is not the mirror; nothing recorded")
        if entries is None:
            raise Refusal(f"the cluster check on {self.host}:{path} exited 0 but printed no `{INVENTORY_MARK}` line — the check did not run whole; nothing recorded")
        unexpected, lacking = unexpected_entries(entries, sums, directories=directories, records=RECORDS)
        if unexpected:
            raise Refusal(f"the cluster destination {self.host}:{path} holds {_n(len(unexpected), 'entry', 'entries')} outside the mirror ({unexpected[:8]}"
                          f"{'…' if len(unexpected) > 8 else ''}) — a regular file {SUMS_NAME} does not list, a directory {MIRROR_NAME} does not list, or a symlink, fifo, "
                          "socket or device, which the mirror never holds; rsync adds and never deletes, so the copy there is not the mirror alone; nothing recorded and "
                          "nothing deleted: move those entries OUT by hand (recorded in decisions.md), or point --cluster-dest at an empty directory, then run the mirror "
                          "again (the pushed files stay and the re-run resumes the push)")
        if lacking:
            raise Refusal(f"the cluster copy at {self.host}:{path} lacks {_n(len(lacking), 'entry', 'entries')} of the mirror after the push ({lacking[:8]}"
                          f"{'…' if len(lacking) > 8 else ''}) — a listed file not there as a regular file, or a recorded directory not there; the push is not the mirror; "
                          "nothing recorded")
        n_files, n_dirs = sum(1 for t, _p in entries if t == "f"), sum(1 for t, _p in entries if t == "d")
        return {"cluster_path": path, "checked": len(sums), "inventory": n_files, "directories": n_dirs, "rsync_argv": argv, "check_command": check,
                "rsync_stderr_tail": (r.stderr or "").strip()[-300:]}

    def rsync_file_argv(self, src: Path, sub: str, name: str) -> List[str]:
        """One file to <dest>/<sub>/<name> (the staged cluster record): the same flags plus --ignore-times, never --delete. --ignore-times (round 4,
        found by the real-binaries test): a record already at the destination — an earlier push's, replaced here — can match the staged one in
        size and mtime to the second, and rsync's quick check then skips the transfer; the read-back refused ("the record did not land whole"),
        so nothing wrong was installed, but the record must land as staged on the first push, not on a later second."""
        target = f"{self.dest}/{sub}/{name}" if self.local else f"{self.host}:{self.dest}/{sub}/{name}"
        argv = ["rsync", *RSYNC_FLAGS, "--ignore-times"]
        if not self.local:
            argv += ["-e", "ssh " + " ".join(SSH_OPTS)]
        return [*argv, str(src), target]

    def push_record(self, staged: Path, sub: str) -> None:
        """The STAGED cluster record (record_staging_path: outside the published mirror) over as <cluster path>/cluster.json, then its digest
        read back there and held to the staged file's. The caller installs the record into the mirror only after this returns."""
        path = f"{self.dest}/{sub}"
        argv = self.rsync_file_argv(staged, sub, CLUSTER_NAME)
        self.log(f"  {' '.join(shlex.quote(a) for a in argv)}")
        self.run(argv, f"rsync of {CLUSTER_NAME} to {self.host}:{path}")
        cmd = f"sha256sum {shlex.quote(path + '/' + CLUSTER_NAME)}"
        r = self.run(self.ssh_argv(cmd), f"ssh {self.host}: {cmd}")
        got = (r.stdout or "").split()[:1]
        want = sha256_file(staged)
        if got != [want]:
            raise Refusal(f"{CLUSTER_NAME} on {self.host}:{path} hashes to {str(got[0] if got else None)[:12]}, the staged one {want[:12]} — the record did not land whole")


# ------------------------------------------------------------------------------------------------------------------------- the run
def _modal_cli(modal: str, volume: str, environment: Optional[str], log, runner=None):
    parts = str(modal).split()
    run = runner or M.default_runner
    if len(parts) > 1 and runner is None:
        prefix = parts
        run = lambda argv, **kw: M.default_runner([*prefix, *list(argv)[1:]], **kw)  # noqa: E731
    return M.VolumeCLI(Path(parts[0]), volume, runner=run, log=log, environment=environment or None)


def run(*, state: Path, receipt_path: Optional[Path], volume: Optional[str], environment: Optional[str], modal: str, cluster_label: str, ssh_host: str,
        cluster_dest: str, dry_run: bool, cluster_root: Optional[str] = None, free_margin: float = 1.25, free_headroom: int = 256 << 20, modal_runner=None,
        cluster_runner: ShRunner = sh_runner, log=print) -> Dict[str, Any]:
    C = _contracts()
    state = Path(state)
    rpath = receipt_path or state / RT.STORE_DIR / RT.RECEIPT_NAME
    receipt, rsha = read_receipt(rpath)
    sha12 = rsha[:12]
    tree = receipt["tree_sha"]
    # this call's volume / environment must be the receipt's (the mirror describes the receipt's execution, never this call's)
    r_vol, r_env = receipt.get("volume"), receipt.get("modal_environment") or None
    if volume is not None and volume != r_vol:
        raise Refusal(f"--volume {volume} is not the receipt's: the run wrote volume {r_vol} ({rpath}) — pass --volume {r_vol}")
    if environment is not None and (environment or None) != r_env:
        raise Refusal(f"--environment {environment or 'none'} is not the receipt's: the run ran under Modal environment {r_env or 'none'} ({rpath}) — pass --environment {r_env or ''}")
    volume, environment = r_vol, r_env
    if not isinstance(volume, str) or not volume:
        raise Refusal(f"the receipt {rpath} names no volume")
    plan = plan_stores(receipt)
    rfiles = receipt_files_by_cell(receipt, plan)
    cluster = Cluster(cluster_label, ssh_host, cluster_dest, root=cluster_root, runner=cluster_runner, log=log)
    mirror = state / MIRROR_DIR / sha12
    build = state / MIRROR_BUILD_DIR / sha12
    cli = _modal_cli(modal, volume, environment, log, runner=modal_runner)
    log(f"receipt {rpath} (sha256 {sha12}…; tree {tree[:7]}; {len(plan)} store(s) on volume {volume}, Modal environment {environment or 'the CLI default'})")
    if dry_run:
        for cell, p in plan.items():
            shown = [modal, *cli.argv("volume", "get", "--force", volume, p["vrel"], str(build / PULL_STAGING / cell))[1:]]     # the CLI as given (`<python> -m modal` whole)
            log(f"DRY-RUN: {' '.join(shown)}   (-> {build / cell}/; first `modal volume ls --json` per directory for the inventory, the sizes and the disk check)")
        log(f"DRY-RUN: sha256 every pulled file; cross-check the {sum(len(v) for v in rfiles.values())} file(s) the receipt hashed byte for byte; "
            f"{CELLS_DIR}/<key>/{CELL_MANIFEST} producer.tree_sha == {tree[:7]} and cell_id for keys {sorted({k for p in plan.values() for k in p['keys']})}; "
            f"{C.RESULT_FILENAME} last line result={C.RESULT_COMPLETE}; every size as `modal volume ls --json` renders it")
        log(f"DRY-RUN: write {build / SUMS_NAME} and {build / MIRROR_NAME}; publish {mirror}/ by rename (a previous mirror beside it is kept; {MIRROR_DIR}/{sha12} held plain "
            f"under the state root {state} — no symlink at either component, the directory's realpath its literal path under the root's realpath — before the pull, "
            "before the rename and on every local check of the published mirror; a link there is REDIRECTED, refused by name, nothing followed or deleted)")
        log(f"DRY-RUN: {cluster.ssh_shown(cluster.prepare_command(sha12))}   (the guard under the cluster root {cluster.root}: no symlink at the destination or at any "
            "component between the root and it, then mkdir -p, then the destination's realpath held to the root's realpath plus its literal relative path; a symlink "
            "refuses by name before any rsync)")
        log(f"DRY-RUN: {' '.join(shlex.quote(a) for a in cluster.rsync_argv(mirror, sha12))}")
        log(f"DRY-RUN: {cluster.ssh_shown(cluster.check_command(sha12))}   (the guard again before the cd; the sums, the two records' digests and the inventory of every "
            f"entry by type read back — the regular files held to {SUMS_NAME} plus the records, the directories to {MIRROR_NAME}'s, no other kind of entry)")
        staged = record_staging_path(mirror)
        log(f"DRY-RUN: write {staged} (the cluster path, the digests of {SUMS_NAME} and {MIRROR_NAME}, the time) only after the check passes")
        log(f"DRY-RUN: {' '.join(shlex.quote(a) for a in cluster.rsync_file_argv(staged, sha12, CLUSTER_NAME))}")
        log(f"DRY-RUN: {cluster.ssh_shown(f'sha256sum {cluster.dest}/{sha12}/{CLUSTER_NAME}')}   (read back by digest; only then is the record installed as {mirror / CLUSTER_NAME})")
        return {"dry": True, "receipt_sha256": rsha, "mirror": str(mirror), "cluster_path": f"{cluster.dest}/{sha12}"}
    # the grading-host path held plain under the state root (round 7) BEFORE a published mirror is followed, before the pull and again before the rename
    local_guard(state, sha12)
    # a published mirror of this receipt: verify it, finish the cluster record if it is missing, and do nothing else
    if mirror.exists():
        mj, sums = verify_published(mirror, receipt_sha=rsha)
        log(f"the published mirror {mirror} verifies ({len(sums)} file(s) against {SUMS_NAME}; receipt {sha12})")
        if (mirror / CLUSTER_NAME).is_file():
            ptr = verified_mirror(state, rsha)
            check_target(ptr, target_of(cluster, sha12), what=f"this call's ({cluster.label} {cluster.host}:{cluster.dest}/{sha12})")   # one cluster record per mirror
            log(f"cluster copy on record: {ptr['cluster_label']} {ptr['cluster_path']} (verified {ptr['verified_utc']}); re-verifying it there")
            facts = verify_cluster_copy(ptr, runner=cluster.runner, log=log)          # the copy itself, over ssh: missing / altered / contaminated / unreachable refuses (rounds 3-5)
            log(f"cluster copy re-verified: {ptr['ssh_host']}:{ptr['cluster_path']} (sha256sum -c {SUMS_NAME}: {ptr['files']} file(s) OK; {SUMS_NAME}, {MIRROR_NAME} and "
                f"{CLUSTER_NAME} read back by digest, the local mirror's; inventory {facts['inventory']} regular file(s) and {facts['directories']} "
                f"director{'y' if facts['directories'] == 1 else 'ies'}, none unexpected); nothing else to do")
            return {"dry": False, "mirror": str(mirror), "receipt_sha256": rsha, "reused": True, **ptr, "cluster_check": facts}
        log(f"no {CLUSTER_NAME} yet: finishing the cluster push of the verified mirror")
        return _cluster_step(mirror, sha12, cluster, sums=sums, mj=mj, rsha=rsha, tree=tree, log=log)
    if build.exists():
        raise Refusal(f"a build of this receipt's mirror already sits at {build} — a refused run left it for inspection; read it, then remove it by hand "
                      f"(rm -rf {shlex.quote(str(build))}) and run the mirror again; nothing pulled")
    # the inventory + the disk check BEFORE the first byte: every file with its rendered size AND every directory (round 6: the directory set is
    # held to this listing after the pull, so an empty directory the download omits, or a directory it creates that was never listed, refuses)
    listings: Dict[str, Dict[str, str]] = {}
    listed_dirs: Dict[str, List[str]] = {}
    estimate = 0
    for cell, p in plan.items():
        files, dirs = list_tree(cli, p["vrel"])
        if not files:
            raise Refusal(f"{cell}: the store root {p['out']} lists no file on volume {volume} — nothing to mirror")
        listings[cell], listed_dirs[cell] = files, dirs
        est = estimate_bytes(files)
        estimate += est
        log(f"  {cell}: {len(files)} file(s) in {len(dirs) + 1} directories, about {est:,} B listed under {p['out']}")
    need = int(estimate * free_margin) + int(free_headroom)
    free = free_bytes(state)
    if free < need:
        raise Refusal(f"the state filesystem under {state} has {free:,} B free; the pull needs about {estimate:,} B x {free_margin} + {int(free_headroom):,} B headroom = "
                      f"{need:,} B — free space first (or point --state-root at a roomier filesystem); nothing pulled")
    log(f"disk: {estimate:,} B listed, {need:,} B needed with the margin, {free:,} B free under {state}")
    build.mkdir(parents=True)
    try:
        per_cell: Dict[str, Dict[str, Any]] = {}
        for cell, p in plan.items():
            staging = build / PULL_STAGING / cell
            staging.mkdir(parents=True)
            log(f"== {cell} <- {p['out']} (volume path {p['vrel']})")
            got = cli.get_tree(p["vrel"], staging)
            os.rename(got, build / cell)
            shutil.rmtree(staging, ignore_errors=True)
            per_cell[cell] = check_store(cell, build / cell, keys=p["keys"], tree_sha=tree, receipt_files=rfiles[cell], listing=listings[cell],
                                         listed_dirs=listed_dirs[cell], C=C)
            log(f"  {cell}: {len(per_cell[cell]['files'])} file(s) in {len(per_cell[cell]['directories']) + 1} directories, {per_cell[cell]['bytes']:,} B; "
                f"{per_cell[cell]['receipt_files_checked']} receipt file(s) byte-identical; manifests {sorted(per_cell[cell]['manifests'])}; {C.RESULT_FILENAME}: "
                f"result={C.RESULT_COMPLETE}; file set, sizes and directory set as listed")
        shutil.rmtree(build / PULL_STAGING, ignore_errors=True)
        if (build / PULL_STAGING).exists():
            raise Refusal(f"the pull's staging directory {build / PULL_STAGING} could not be removed — the mirror's directories would not be the stores'; nothing published")
        sums_p = write_sums(build, per_cell)
        # every directory of the mirror: the LISTED directory set of every store (confirmed on disk by check_store, round 6) under its cell,
        # plus the cells themselves — recorded so the local re-check and the cluster inventory hold the directory set to a record rather than
        # to the file list (round 5); the build's own directories must be exactly that set (a stray directory at the build root refuses)
        directories = sorted([cell for cell in plan] + [f"{cell}/{d}" for cell in plan for d in listed_dirs[cell]])
        on_disk = walk_tree(build)[1]
        if on_disk != directories:
            raise Refusal(f"the build {build} holds directories that are not the stores' listed ones (present but not listed: {sorted(set(on_disk) - set(directories))[:8]}; "
                          f"listed but absent: {sorted(set(directories) - set(on_disk))[:8]}) — nothing published")
        mj = {
            "schema": MIRROR_SCHEMA, "receipt_sha256": rsha, "receipt_path": str(rpath), "tree_sha": tree, "app": receipt.get("app"), "volume": volume,
            "modal_environment": environment, "written_utc": utc_now(),
            "cells": {cell: {"volume_path": plan[cell]["out"], "volume_rel": plan[cell]["vrel"], "store_keys": plan[cell]["keys"], "files": len(per_cell[cell]["files"]),
                             "directories": len(per_cell[cell]["directories"]), "bytes": per_cell[cell]["bytes"], "manifests": per_cell[cell]["manifests"],
                             "result_last_line": per_cell[cell]["result_last_line"], "receipt_files_checked": per_cell[cell]["receipt_files_checked"]} for cell in plan},
            "files": sum(len(per_cell[c]["files"]) for c in plan), "bytes": sum(per_cell[c]["bytes"] for c in plan), "directories": directories,
            "receipt_files_checked": sum(per_cell[c]["receipt_files_checked"] for c in plan), "size_check": SIZE_CHECK, "directory_check": DIR_CHECK, "sums": SUMS_NAME,
            "written_by": "reproduce/tools/stage07_mirror.py",
        }
        (build / MIRROR_NAME).write_text(json.dumps(mj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        sums = read_sums(sums_p)
    except Refusal as e:
        raise Refusal(f"{e} (the build stays at {build} for inspection)") from None
    try:
        local_guard(state, sha12)                         # round 7: a link that appeared at 07_mirror or <sha12> while the build ran is refused before the rename
        mirror.parent.mkdir(parents=True, exist_ok=True)
        local_guard(state, sha12)                         # 07_mirror exists now: its realpath its literal path under the root's, or nothing is published
    except Refusal as e:
        raise Refusal(f"{e} (the build stays at {build} for inspection)") from None
    if mirror.exists():
        raise Refusal(f"{mirror} appeared while this build ran — another mirror of the same receipt; the build stays at {build}")
    os.rename(build, mirror)
    _rmdir_quiet(build.parent)                        # the build parent, when this was its only build (another receipt's leftover build or staged record stays)
    log(f"published {mirror} ({mj['files']} file(s), {mj['bytes']:,} B; {SUMS_NAME} + {MIRROR_NAME})")
    return _cluster_step(mirror, sha12, cluster, sums=sums, mj=mj, rsha=rsha, tree=tree, log=log)


def _cluster_step(mirror: Path, sha12: str, cluster: Cluster, *, sums: Mapping[str, str], mj: Mapping[str, Any], rsha: str, tree: str, log) -> Dict[str, Any]:
    mj_sha = sha256_file(mirror / MIRROR_NAME)
    log(f"== cluster {cluster.label} ({cluster.host}): {cluster.dest}/{sha12}")
    facts = cluster.push_and_check(mirror, sha12, sums=sums, mj_sha=mj_sha, directories=mirror_directories(mj, mirror / MIRROR_NAME))
    record = {"schema": CLUSTER_SCHEMA, "cluster_label": cluster.label, "ssh_host": cluster.host, "cluster_root": cluster.root, "cluster_path": facts["cluster_path"],
              "verified": True, "guard": f"no symlink at any component between {cluster.root} and the destination, the destination's realpath its literal path under the "
                                         "root's (before the mkdir, the rsync and the check; the same guard runs before every re-check)",
              "verified_utc": utc_now(), "check": f"sha256sum -c {SUMS_NAME} ({facts['checked']} file(s) OK) + the two records' digests read back + the inventory of every "
                                                  f"entry by type ({_n(facts['inventory'], 'regular file', 'regular files')}, {_n(facts['directories'], 'directory', 'directories')}; "
                                                  f"none unexpected)",
              "mirror_sha256sum_digest": sha256_file(mirror / SUMS_NAME), "mirror_json_sha256": mj_sha, "receipt_sha256": rsha, "tree_sha": tree,
              "files": mj.get("files"), "bytes": mj.get("bytes"), "rsync_argv": facts["rsync_argv"], "check_command": facts["check_command"]}
    # the record is STAGED outside the mirror, pushed on its own and read back by digest, and only then installed: a transfer that fails
    # leaves the published mirror with no cluster.json (the re-run finishes the push), never a `verified: true` record the filing would accept
    staged = record_staging_path(mirror)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    try:
        cluster.push_record(staged, sha12)
    except Refusal as e:
        staged.unlink(missing_ok=True)
        _rmdir_quiet(staged.parent)
        raise Refusal(f"{e}; {CLUSTER_NAME} was NOT installed at {mirror} (the mirror stands published and verified; run the mirror again to finish the push)") from None
    os.replace(staged, mirror / CLUSTER_NAME)
    _rmdir_quiet(staged.parent)
    ptr = verified_mirror(mirror.parent.parent, rsha)
    log(f"cluster copy verified: {cluster.host}:{facts['cluster_path']} (sha256sum -c {SUMS_NAME}: {facts['checked']} file(s) OK; inventory {facts['inventory']} regular file(s) "
        f"and {facts['directories']} director{'y' if facts['directories'] == 1 else 'ies'}, none unexpected); {CLUSTER_NAME} pushed, read back and installed")
    log(f"mirror: {mirror} — receipt {sha12}, {mj.get('files')} file(s), {mj.get('bytes'):,} B, {SUMS_NAME} digest {ptr['mirror_sha256sum_digest'][:12]}; `07 --phase file` records it")
    return {"dry": False, "mirror": str(mirror), "receipt_sha256": rsha, "reused": False, **ptr}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-root", type=Path, required=True, help="the VIEW's state directory (07: the model's state dir for the baseline wave, <state>/07_views/<view> for a "
                                                                      "dose view): the receipt is read and the mirror written under it")
    ap.add_argument("--receipt", type=Path, default=None, help=f"default <state-root>/{RT.STORE_DIR}/{RT.RECEIPT_NAME} (the published receipt)")
    ap.add_argument("--volume", default=None, help="a cross-check: must be the receipt's volume (default: the receipt's)")
    ap.add_argument("--environment", default=None, help="a cross-check: must be the Modal environment the receipt records (default: the receipt's)")
    ap.add_argument("--modal", default=str(M.DEFAULT_MODAL), help="the Modal CLI (`<python> -m modal` accepted)")
    ap.add_argument("--cluster-root", required=True, help="the site's expanded SISTEER_REPRODUCE_ROOT on the cluster (07 passes it): every component between it and "
                                                          "--cluster-dest/<receipt sha12> is guarded against a symlink before the push and before every re-check (recorded)")
    ap.add_argument("--cluster-label", required=True, help="the site's label of the primary cluster (recorded)")
    ap.add_argument("--ssh-host", required=True, help="CLUSTER_SSH of that cluster; `local:<dir>` is the test seam (bash -c here, a local rsync target)")
    ap.add_argument("--cluster-dest", required=True, help="the directory on the cluster under which <receipt sha12>/ lands (07: <reproduce root>/<model>/07_mirror for the baseline "
                                                          "wave, <reproduce root>/<model>/07_views/<view>/07_mirror for a dose view — the state tree's layout)")
    ap.add_argument("--free-margin", type=float, default=1.25, help="the pull refuses when free space < listed bytes x this + --free-headroom (default 1.25)")
    ap.add_argument("--free-headroom", type=int, default=256 << 20, help="bytes of headroom on top of the margin (default 256 MiB)")
    ap.add_argument("--dry-run", action="store_true", help="print the planned pulls, the rsync and the ssh check from the receipt; no volume access, nothing written")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    log = lambda s: print(f"[stage07-mirror] {s}", flush=True)  # noqa: E731
    try:
        run(state=Path(a.state_root), receipt_path=a.receipt, volume=a.volume, environment=a.environment, modal=a.modal, cluster_label=a.cluster_label,
            ssh_host=a.ssh_host, cluster_dest=a.cluster_dest, cluster_root=a.cluster_root, dry_run=a.dry_run, free_margin=a.free_margin, free_headroom=a.free_headroom, log=log)
        return 0
    except Refusal as e:
        log(f"REFUSED: {e}")
        log("this run published nothing and recorded nothing; the filing refuses until a verified mirror with its cluster record stands for the receipt and the cluster copy it names verifies there")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
