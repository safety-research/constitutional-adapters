#!/usr/bin/env python3
"""site_cutover.py — install a prepared set of site files and rename the per-cluster inbox directories, inside a landing
window's drain, with a byte-for-byte way back.

    site_cutover.py check    --set DIR --root ROOT --state STATE [--own-op OP] [--unit U]...
    site_cutover.py apply    --set DIR --root ROOT --state STATE --backup DIR [--own-op OP] [--unit U]...
                             [--new-py PY --new-pypath PATH] [--clusters-py FILE] [--expect-site-aliases-sha256 H]
    site_cutover.py restore  --backup DIR
    site_cutover.py rehearse --set DIR --scratch DIR [--new-py PY --new-pypath PATH] [--clusters-py FILE] [--plant-failure]

The PREPARED SET is a directory holding the new files and MANIFEST.tsv, one row per change (TAB-separated; blank lines and
lines starting with # are ignored; a path may start with ~/):

    file    <source, relative to the set>  <target path>  <sha256 of the target now, or "absent">  <sha256 of the source>
    rename  <old directory>                <new directory>

A `file` row replaces the target with the source (a new file when the target is "absent"). A `rename` row moves one inbox
directory to its new name with rename(2), so both names must be on one filesystem.

check refuses, each reason on its own `REFUSED:` line and exit 31, when any of these holds; nothing is touched:
  - the manifest does not parse, or a source file's sha256 is not the one the manifest records (the set changed after it
    was written)
  - a target's sha256 is not the manifest's "before", or an "absent" target exists (a live file changed after the set was
    prepared), or a target's directory does not exist
  - a rename's old directory is missing, its new name already exists, or the two are on different filesystems
  - an old inbox holds a receive directory (`.recv-*`) or a RECEIVING.json: an interrupted delivery that only a later window
    can finish, and never one to move by hand
  - a reads operation record ($STATE/reads/ops/*.json) is not in a terminal state (the window's own operation, --own-op, aside)
  - $STATE/reads/critical.lock or ROOT/harvester.lock is held (a non-blocking flock probe)
  - a named unit (--unit) reads active, activating, deactivating or reloading (`systemctl --user is-active`)

apply runs check, then backs every existing target up under --backup (a new directory) with its sha256, installs each source
atomically (a temporary file beside the target, then a rename) and verifies it, renames the inbox directories, and validates:
`registry config --check` under the new code (--new-py with PYTHONPATH=--new-pypath), the new site config through the
cluster-block reader (--clusters-py), and the site-alias digest of ROOT equal to that of the set's site_aliases.toml (and to
--expect-site-aliases-sha256 when given). The ledger --backup/cutover.json is written ahead: each install and each rename is
recorded before it runs and marked done after it, so a step that ran is always in the ledger, whether or not its mark was
written. Any failure restores everything recorded so far, byte for byte, and exits 31. Success prints INSTALLED=1, BACKUP, FILES,
RENAMES, DIGEST.

restore reads --backup/cutover.json and settles every recorded step by what the filesystem shows, never by its done mark (a
step can have run with its mark unwritten: a full disk, or a stop between the two), in reverse order:
  - a rename: the new name present and the old one absent is moved back; the old name present and the new one absent is
    already in place (never renamed, or already moved back); both names present, or neither, is refused by name and moves
    nothing (records may sit under either name)
  - a file that existed before: put back from its backup (sha-verified) unless it already reads as the backup
  - a file that was absent before: removed when it holds the installed bytes, left (and refused by name) when it holds others
  - a temporary file a stopped install or restore left beside a target is removed
Run again, it finishes what an interrupted restore left. It prints RESTORED=1 and exits 0, or names what it could not restore
(the ledger left unwritten included) and exits 3.

rehearse runs the same apply and restore on byte copies of the targets inside --scratch (each target path re-rooted under it,
each old inbox an empty directory), then checks that every file is back byte for byte. --plant-failure fails the validation
after the install, so the automatic restore is what is checked.

Exit: 0 done; 2 usage; 3 a restore that could not finish; 31 a refusal or a failed apply (restored).
Standard library only: it runs before the base checkout moves and must not import the registry package."""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

MANIFEST = "MANIFEST.tsv"
LEDGER = "cutover.json"
LEDGER_SCHEMA = "site_cutover/1"
ABSENT = "absent"
REFUSED_RC = 31
RESTORE_FAILED_RC = 3
#: op record states after which nothing of a reads operation is outstanding (sisteer.registry.reads_protocol.OP_TERMINAL_STATES;
#: a test keeps the two equal)
OP_TERMINAL_STATES = ("receipted", "cancelled", "yielded", "unchanged", "failed", "reconciled", "ownership_lost", "interrupted", "dry_run")
BUSY_UNIT_STATES = ("active", "activating", "deactivating", "reloading", "refreshing")
RECV_PREFIX = ".recv-"
RECEIVING_FILE = "RECEIVING.json"
TMP_INFIX = ".tmp-site-cutover-"           # atomic_copy's temporary file beside a target: <target><TMP_INFIX><pid>
ALIASES_FILE = "site_aliases.toml"
DIGEST_SNIPPET = ("import sys\n"
                  "from sisteer.registry import site_aliases as A\n"
                  "root_t = A.load_root(sys.argv[1]); A.validate(root_t)\n"
                  "set_t = A.load_table(sys.argv[2]); A.validate(set_t)\n"
                  "print('ROOT_DIGEST=' + A.digest(root_t)); print('SET_DIGEST=' + A.digest(set_t))\n")


class Refused(Exception):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expand(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def read_manifest(set_dir: str) -> dict:
    path = os.path.join(set_dir, MANIFEST)
    if not os.path.isfile(path):
        raise Refused(f"no {MANIFEST} in the prepared set {set_dir}")
    files, renames = [], []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if cols[0] == "file" and len(cols) == 5:
                src, target, before, after = cols[1:]
                if os.path.isabs(src) or ".." in src.split("/"):
                    raise Refused(f"{path}:{n}: the source {src!r} must be a path inside the set")
                files.append({"source": os.path.join(set_dir, src), "target": expand(target), "before": before, "after": after})
            elif cols[0] == "rename" and len(cols) == 3:
                renames.append({"old": expand(cols[1]), "new": expand(cols[2])})
            else:
                raise Refused(f"{path}:{n}: not a `file` row (5 columns) or a `rename` row (3 columns): {line.rstrip()!r}")
    targets = [f["target"] for f in files]
    if len(set(targets)) != len(targets):
        raise Refused(f"{path}: a target is named twice")
    if not files and not renames:
        raise Refused(f"{path}: no rows")
    return {"path": path, "sha256": sha256_file(path), "files": files, "renames": renames}


def lock_held(path: str) -> bool:
    """True when another process holds an flock on ``path``; an absent file is not held."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return True
        raise
    finally:
        os.close(fd)
    return False


def unit_state(unit: str) -> str:
    r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
    return (r.stdout or "").strip().splitlines()[0] if (r.stdout or "").strip() else "unknown"


def interrupted_deliveries(old: str) -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(old):
        for d in dirnames:
            if d.startswith(RECV_PREFIX):
                found.append(os.path.join(dirpath, d))
        if RECEIVING_FILE in filenames:
            found.append(os.path.join(dirpath, RECEIVING_FILE))
    return sorted(set(found))


def refusals(man: dict, root: str, state: str, own_op: str, units: list) -> list:
    out = []
    for f in man["files"]:
        if not os.path.isfile(f["source"]):
            out.append(f"the prepared file {f['source']} is missing")
        elif sha256_file(f["source"]) != f["after"]:
            out.append(f"the prepared file {f['source']} is not the one the manifest records (sha256 {sha256_file(f['source'])}, manifest {f['after']}); regenerate its row")
        parent = os.path.dirname(f["target"])
        if not os.path.isdir(parent):
            out.append(f"the directory of {f['target']} does not exist")
        if f["before"] == ABSENT:
            if os.path.lexists(f["target"]):
                out.append(f"{f['target']} exists, and the manifest says it is absent (a file appeared after the set was prepared)")
        elif not os.path.isfile(f["target"]):
            out.append(f"{f['target']} is missing, and the manifest records sha256 {f['before']} for it")
        else:
            now = sha256_file(f["target"])
            if now != f["before"]:
                out.append(f"{f['target']} changed after the set was prepared (sha256 now {now}, manifest {f['before']})")
    for r in man["renames"]:
        if not os.path.isdir(r["old"]):
            out.append(f"the inbox directory {r['old']} does not exist")
            continue
        if os.path.lexists(r["new"]):
            out.append(f"the inbox target {r['new']} already exists")
        elif os.path.isdir(os.path.dirname(r["new"])) and os.stat(r["old"]).st_dev != os.stat(os.path.dirname(r["new"])).st_dev:
            out.append(f"{r['old']} and {r['new']} are on different filesystems (a rename must stay on one)")
        for hit in interrupted_deliveries(r["old"]):
            out.append(f"an interrupted delivery is in {r['old']}: {hit} (only a later window can finish it; never move it by hand)")
    ops_dir = os.path.join(state, "reads", "ops")
    if os.path.isdir(ops_dir):
        for fn in sorted(os.listdir(ops_dir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(ops_dir, fn), encoding="utf-8") as fh:
                    op = json.load(fh)
            except (OSError, ValueError) as exc:
                out.append(f"the reads op record {os.path.join(ops_dir, fn)} cannot be read ({exc})")
                continue
            op_id = str(op.get("op_id") or fn[:-5])
            if op_id == own_op:
                continue
            if str(op.get("state")) not in OP_TERMINAL_STATES:
                out.append(f"the reads operation {op_id} is not terminal (state {op.get('state')!r}, owner {op.get('owner')!r}; {os.path.join(ops_dir, fn)})")
    for lock in (os.path.join(state, "reads", "critical.lock"), os.path.join(root, "harvester.lock")):
        if lock_held(lock):
            out.append(f"{lock} is held by another process")
    for u in units:
        try:
            st = unit_state(u)
        except OSError as exc:
            out.append(f"the state of the unit {u} cannot be read ({exc})")
            continue
        if st in BUSY_UNIT_STATES:
            out.append(f"the unit {u} reads {st}")
    return out


def atomic_copy(src: str, dst: str) -> None:
    """``src``'s bytes at ``dst`` through a temporary file and one rename; the mode is ``dst``'s when it exists (a private site
    config stays private), else ``src``'s."""
    tmp = f"{dst}{TMP_INFIX}{os.getpid()}"
    mode = os.stat(dst).st_mode if os.path.isfile(dst) else os.stat(src).st_mode
    try:
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo)
            fo.flush()
            os.fsync(fo.fileno())
        os.chmod(tmp, mode & 0o7777)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_ledger(backup: str, ledger: dict) -> None:
    tmp = os.path.join(backup, LEDGER + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, os.path.join(backup, LEDGER))


def install(man: dict, backup: str) -> dict:
    """Back up, install and rename, the ledger written ahead: each step is recorded before it runs (its intent) and marked done after
    it, so a step that ran is in the ledger even when its mark could not be written. Raises on the first failure."""
    os.makedirs(backup)
    ledger = {"schema": LEDGER_SCHEMA, "manifest": man["path"], "manifest_sha256": man["sha256"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "files": [], "renames": [], "restored": False}
    save_ledger(backup, ledger)
    for i, f in enumerate(man["files"]):
        entry = {"target": f["target"], "before": f["before"], "after": f["after"], "backup": None, "installed": False}
        if f["before"] != ABSENT:
            name = f"{i:02d}-{os.path.basename(f['target'])}"
            shutil.copy2(f["target"], os.path.join(backup, name))      # bytes and mode
            if sha256_file(os.path.join(backup, name)) != f["before"]:
                raise Refused(f"the backup of {f['target']} does not read back as sha256 {f['before']}")
            entry["backup"] = name
        ledger["files"].append(entry)
        save_ledger(backup, ledger)                # the intent, before the target is touched
        atomic_copy(f["source"], f["target"])
        entry["installed"] = True
        save_ledger(backup, ledger)
        if sha256_file(f["target"]) != f["after"]:
            raise Refused(f"{f['target']} does not read back as sha256 {f['after']} after the install")
    for r in man["renames"]:
        entry = {"old": r["old"], "new": r["new"], "done": False}
        ledger["renames"].append(entry)
        save_ledger(backup, ledger)                # the intent, before the old name goes away
        started_ns = time.time_ns()                # read just before the old name goes away: a write stamped later went nowhere under it
        os.rename(r["old"], r["new"])
        entry["done"] = True
        entry["rename_started_ns"] = started_ns
        save_ledger(backup, ledger)
    return ledger


def remove_temporaries(target: str) -> None:
    """The temporary files a stopped install or restore left beside ``target`` (``<target>.tmp-site-cutover-<pid>``), removed."""
    parent, base = os.path.split(target)
    try:
        names = os.listdir(parent)
    except FileNotFoundError:
        return
    for name in names:
        if name.startswith(base + TMP_INFIX) and name[len(base + TMP_INFIX):].isdigit():
            os.unlink(os.path.join(parent, name))


def restore_rename(r: dict) -> str:
    """One recorded rename settled by what the filesystem shows, whatever its done mark says; returns a problem, or "" when the old name
    holds the inbox and the new one is free."""
    old, new = r["old"], r["new"]
    old_there, new_there = os.path.lexists(old), os.path.lexists(new)
    if old_there and not new_there:
        return "" if os.path.isdir(old) and not os.path.islink(old) else f"{old} is not a directory; {new} is free"
    if new_there and not old_there:
        if not os.path.isdir(new) or os.path.islink(new):
            return f"{new} is not a directory, and {old} is absent: neither is moved"
        os.rename(new, old)
        return ""
    if old_there and new_there:
        return (f"both {old} and {new} exist, so records may sit under either name: neither is moved, and the timers must not start "
                f"until one directory holds them all under {old}")
    return f"neither {old} nor {new} exists: the inbox is under neither name"


def restore_file(f: dict, backup: str) -> str:
    """One recorded install settled by what the filesystem shows, whatever its installed mark says; returns a problem, or "" when the
    target reads as it did before the cutover."""
    target = f["target"]
    remove_temporaries(target)
    if f["before"] == ABSENT:
        if not os.path.lexists(target):
            return ""
        if os.path.isfile(target) and not os.path.islink(target) and sha256_file(target) == f["after"]:
            os.unlink(target)
            return ""
        return f"{target} was absent before and now holds bytes that are not the installed file's; left as it is"
    if not f.get("backup"):
        return f"{target} has no backup in the ledger; left as it is"
    if os.path.isfile(target) and sha256_file(target) == f["before"]:
        return ""                                  # never installed, or already put back
    src = os.path.join(backup, f["backup"])
    if not os.path.isfile(src) or sha256_file(src) != f["before"]:
        return f"the backup {src} does not read as sha256 {f['before']}; {target} left as it is"
    atomic_copy(src, target)
    if sha256_file(target) != f["before"]:
        return f"{target} does not read back as sha256 {f['before']} after the restore"
    return ""


def restore(backup: str) -> list:
    """Every recorded rename and install put back, in reverse order, by what the filesystem shows (never by a done mark, which a full
    disk or a stop can leave unwritten after its step ran); returns the problems (empty = restored). Idempotent: run again, it finishes
    what an interrupted restore left. Raises only when the ledger itself cannot be read."""
    with open(os.path.join(backup, LEDGER), encoding="utf-8") as fh:
        ledger = json.load(fh)
    problems = []
    for r in reversed(ledger.get("renames") or []):
        try:
            p = restore_rename(r)
        except OSError as exc:                        # named, and every other step still runs
            p = f"the inbox directory {r['new']} could not be moved back to {r['old']} ({exc})"
        if p:
            problems.append(p)
        else:
            r["done"] = False
    for f in reversed(ledger.get("files") or []):
        try:
            p = restore_file(f, backup)
        except OSError as exc:
            p = f"{f['target']} could not be restored ({exc})"
        if p:
            problems.append(p)
        else:
            f["installed"] = False
    ledger["restored"] = not problems
    ledger["restored_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        save_ledger(backup, ledger)
    except OSError as exc:                            # the filesystem is settled above; the ledger saying so is part of a finished restore
        problems.append(f"the ledger {os.path.join(backup, LEDGER)} could not record the restore ({exc}); run the restore again once it can be written")
    return problems


def validate(man: dict, root: str, new_py: str, new_pypath: str, clusters_py: str, expect_digest: str, plant_failure: bool) -> str:
    """The installed configuration read back under the new code; returns the site-alias digest, raises Refused on a failure."""
    if plant_failure:
        raise Refused("a planted failure after the install (--plant-failure)")
    if clusters_py:
        for f in man["files"]:
            if os.path.basename(f["target"]) == "config.env":
                r = subprocess.run([sys.executable, clusters_py, "--config", f["target"], "--labels"], capture_output=True, text=True)
                if r.returncode != 0:
                    raise Refused(f"the installed site config {f['target']} does not load: {(r.stderr or r.stdout).strip()[-300:]}")
    if not new_py:
        return ""
    env = dict(os.environ, PYTHONPATH=new_pypath)
    r = subprocess.run([new_py, "-m", "sisteer.registry.cli", "--root", root, "config", "--check"], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise Refused(f"registry config --check under the new code failed (rc {r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")
    set_aliases = next((f["source"] for f in man["files"] if os.path.basename(f["target"]) == ALIASES_FILE), "")
    if not set_aliases:
        return ""
    r = subprocess.run([new_py, "-c", DIGEST_SNIPPET, root, set_aliases], capture_output=True, text=True, env=env)
    got = dict(line.split("=", 1) for line in (r.stdout or "").splitlines() if "=" in line)
    if r.returncode != 0 or not got.get("ROOT_DIGEST") or not got.get("SET_DIGEST"):
        raise Refused(f"the site-alias digest could not be read under the new code (rc {r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")
    if got["ROOT_DIGEST"] != got["SET_DIGEST"]:
        raise Refused(f"the installed site-alias table reads digest {got['ROOT_DIGEST']}, the prepared one {got['SET_DIGEST']}")
    if expect_digest and got["ROOT_DIGEST"] != expect_digest:
        raise Refused(f"the site-alias digest {got['ROOT_DIGEST']} is not the expected {expect_digest}")
    return got["ROOT_DIGEST"]


def cmd_check(a) -> int:
    try:
        man = read_manifest(a.set)
        out = refusals(man, expand(a.root), expand(a.state), a.own_op or "", a.unit or [])
    except Refused as exc:
        out = [str(exc)]
    for line in out:
        print(f"REFUSED: {line}")
    if out:
        return REFUSED_RC
    print(f"CHECKED=1\nFILES={len(man['files'])}\nRENAMES={len(man['renames'])}")
    return 0


def run_apply(man: dict, root: str, state: str, backup: str, own_op: str, units: list, new_py: str, new_pypath: str,
              clusters_py: str, expect_digest: str, plant_failure: bool = False) -> int:
    out = refusals(man, root, state, own_op, units)
    if os.path.lexists(backup):
        out.append(f"the backup directory {backup} already exists (one cutover per backup directory)")
    for line in out:
        print(f"REFUSED: {line}")
    if out:
        return REFUSED_RC
    try:
        install(man, backup)
        digest = validate(man, root, new_py, new_pypath, clusters_py, expect_digest, plant_failure)
    except (Refused, OSError, subprocess.SubprocessError) as exc:
        print(f"FAILED: {exc}")
        try:
            problems = restore(backup) if os.path.isfile(os.path.join(backup, LEDGER)) else []
        except (OSError, ValueError) as rexc:           # the ledger itself unreadable or unwritable: say so, exit 3
            problems = [f"the restore could not run ({rexc}); {os.path.join(backup, LEDGER)} says what is where"]
        for p in problems:
            print(f"RESTORE PROBLEM: {p}")
        print(f"RESTORED={0 if problems else 1}\nBACKUP={backup}")
        return RESTORE_FAILED_RC if problems else REFUSED_RC
    print(f"INSTALLED=1\nBACKUP={backup}\nFILES={len(man['files'])}\nRENAMES={len(man['renames'])}\nDIGEST={digest}")
    return 0


def cmd_apply(a) -> int:
    try:
        man = read_manifest(a.set)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return REFUSED_RC
    return run_apply(man, expand(a.root), expand(a.state), expand(a.backup), a.own_op or "", a.unit or [], a.new_py or "", a.new_pypath or "",
                     a.clusters_py or "", a.expect_site_aliases_sha256 or "")


def cmd_restore(a) -> int:
    backup = expand(a.backup)
    if not os.path.isfile(os.path.join(backup, LEDGER)):
        print(f"REFUSED: no {LEDGER} under {backup}")
        return RESTORE_FAILED_RC
    try:
        problems = restore(backup)
    except (OSError, ValueError) as exc:
        problems = [f"the restore could not run ({exc}); {os.path.join(backup, LEDGER)} says what is where"]
    for p in problems:
        print(f"RESTORE PROBLEM: {p}")
    print(f"RESTORED={0 if problems else 1}\nBACKUP={backup}")
    return RESTORE_FAILED_RC if problems else 0


def cmd_rehearse(a) -> int:
    try:
        man = read_manifest(a.set)
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return REFUSED_RC
    scratch = expand(a.scratch)
    if os.path.lexists(scratch) and os.listdir(scratch):
        print(f"REFUSED: the scratch directory {scratch} is not empty")
        return REFUSED_RC
    os.makedirs(scratch, exist_ok=True)
    under = lambda p: os.path.join(scratch, p.lstrip("/"))   # noqa: E731
    copies = {}
    for f in man["files"]:
        os.makedirs(os.path.dirname(under(f["target"])), exist_ok=True)
        if os.path.isfile(f["target"]):
            shutil.copy2(f["target"], under(f["target"]))
            copies[under(f["target"])] = sha256_file(under(f["target"]))
        else:
            copies[under(f["target"])] = ABSENT
    for r in man["renames"]:
        os.makedirs(under(r["old"]), exist_ok=True)
        os.makedirs(os.path.dirname(under(r["new"])), exist_ok=True)
    smap = {"path": man["path"], "sha256": man["sha256"],
            "files": [dict(f, target=under(f["target"])) for f in man["files"]],
            "renames": [{"old": under(r["old"]), "new": under(r["new"])} for r in man["renames"]]}
    root = next((os.path.dirname(f["target"]) for f in smap["files"] if os.path.basename(f["target"]) in ("config.toml", ALIASES_FILE)), scratch)
    state = os.path.join(scratch, "state")
    backup = os.path.join(scratch, "backup")
    rc = run_apply(smap, root, state, backup, "", [], a.new_py or "", a.new_pypath or "", a.clusters_py or "", "", plant_failure=a.plant_failure)
    if a.plant_failure:
        if rc != REFUSED_RC:
            print(f"REHEARSAL=FAIL (the planted failure ended with rc {rc}, not the restored refusal {REFUSED_RC})")
            return REFUSED_RC
    else:
        if rc != 0:
            print("REHEARSAL=FAIL (the apply did not succeed on the copies)")
            return REFUSED_RC
        problems = restore(backup)
        for p in problems:
            print(f"RESTORE PROBLEM: {p}")
        if problems:
            print("REHEARSAL=FAIL (the restore did not finish)")
            return RESTORE_FAILED_RC
    bad = []
    for path, sha in copies.items():
        if sha == ABSENT:
            if os.path.lexists(path):
                bad.append(f"{path} should be absent again")
        elif not os.path.isfile(path) or sha256_file(path) != sha:
            bad.append(f"{path} is not back byte for byte")
    for r in smap["renames"]:
        if not os.path.isdir(r["old"]) or os.path.lexists(r["new"]):
            bad.append(f"{r['old']} is not back under its old name")
    for b in bad:
        print(f"MISMATCH: {b}")
    print(f"REHEARSAL={'PASS' if not bad else 'FAIL'} ({'planted failure after the install' if a.plant_failure else 'install, validate, restore'}; "
          f"{len(copies)} file(s), {len(smap['renames'])} rename(s); scratch {scratch})")
    return 0 if not bad else REFUSED_RC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--set", required=True)
        p.add_argument("--root", required=True, help="the registry root (harvester.lock, the installed config.toml)")
        p.add_argument("--state", required=True, help="the landing state directory ($STATE/reads/ops, critical.lock)")
        p.add_argument("--own-op", default="", help="this window's reads operation, which is live by design")
        p.add_argument("--unit", action="append", default=[], help="a unit that must not be running (repeatable)")
        if name == "apply":
            p.add_argument("--backup", required=True)
            p.add_argument("--new-py", default="")
            p.add_argument("--new-pypath", default="")
            p.add_argument("--clusters-py", default="")
            p.add_argument("--expect-site-aliases-sha256", default="")
    p = sub.add_parser("restore")
    p.add_argument("--backup", required=True)
    p = sub.add_parser("rehearse")
    p.add_argument("--set", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--new-py", default="")
    p.add_argument("--new-pypath", default="")
    p.add_argument("--clusters-py", default="")
    p.add_argument("--plant-failure", action="store_true")
    a = ap.parse_args(argv)
    return {"check": cmd_check, "apply": cmd_apply, "restore": cmd_restore, "rehearse": cmd_rehearse}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
