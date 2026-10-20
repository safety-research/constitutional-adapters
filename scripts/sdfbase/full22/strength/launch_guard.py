#!/usr/bin/env python
"""STRENGTH FRONTIER: the launch guard — before ANY dose-ladder leg is submitted, the object THIS RUN will serve, read
three ways ON the cluster it will run on, and refused unless all three agree.

  launch_guard.py --cluster LABEL [--site-dir DIR] --target T --views v1,v2 [--pins PATH] [--json OUT]
  launch_guard.py --local-root BANK_ROOT     --target T --views v1,v2 [--pins PATH] [--json OUT]   (a pulled or synthetic bank)

The sbatch reads the bank VIEW: <shared>/exp/objects/sdfbase/full22/<target>/cells_view/<view>/<cell_id>/object.pt (a
symlink into cells/<staged_id>, built by make_bank_view.py). For every entry of the view's VIEW_MANIFEST.json the guard
requires

  disk      sha256 of object.pt's bytes, hashed on the host at guard time          — what the job will load
  manifest  entries[].object_sha256, the sha the view recorded when it was built   — what the bank claims it holds
  pin       configs/sdfbase_full22_pins.json[target][staged_id].object_sha256      — what the roster says the cell IS

to be ONE value. It prints one line per (view, cell)

  OK|REFUSE target/view cell disk=… manifest=… pin=… [reason]

and a final `GUARD: PASS (n cells)` or `GUARD: REFUSE (k of n cells)`. Exit 0 on PASS, 3 on REFUSE, 2 on an operator
error. PASS is said out loud, one OK line per cell, on purpose: a guard that only speaks on failure leaves no evidence
it ran.

The failure this catches does not announce itself: a wrong object runs to completion and yields a clean-looking number
attributed to an artifact it never touched. It exists because "staged" was relayed twice on 2026-09-13 while the object
was not yet in the serving bank.

Refusal reasons — exactly one per cell, the first that holds in this order:
  NO VIEW         the manifest names the cell but object.pt is not on disk (a dangling symlink, a staged copy since
                  removed, a "staged" that never reached this bank) — a refusal, never an exception
  disk!=manifest  the bytes on disk are not the bytes the view recorded (the bank disagrees with itself)
  NO PIN          the roster pins nothing for (target, staged_id)
  disk!=pin       the bytes on disk are not the bytes the roster pinned

THE PIN IS KEYED BY (target, staged_id), NEVER BY staged_id ALONE. Staged ids repeat across targets — 36 of the 50 ids
in the pins file (`base__…` under all five targets; `dm1__constv2-mix-full__lora_r4_ce` under nemotron, glm and llama),
each a DIFFERENT object with a different sha. A lookup keyed by the id alone matched another target's pin and refused
four correct cells on 2026-09-13. `pinned_sha` reads only pins[target] (one level of nesting, lane -> id, admitted);
an id pinned twice inside one block with different shas is an operator error, never a pick.

Operator errors (exit 2; nothing to put a verdict on): the pins file has no block for the target, a view directory or
its VIEW_MANIFEST.json is missing, a view with no entries, an unreachable host. A missing view is not a per-cell refusal
because there is no cell to refuse — the launch is still blocked, since only exit 0 gets through this guard.

With --cluster <label> the check runs ON that cluster, in screen_verdict.py's shape: this file's own source goes over ONE
ssh as `python3 -`, the request (bank root, target, views, the target's pins block) rides base64 in argv, one JSON line
comes back. The bytes are hashed where the job will read them; nothing is copied to the grading host, nothing is written on
the cluster. The label's block in the SITE config (reproduce/lib/clusters.py — `clusters/<label>.env`: CLUSTER_SSH is
the host, CLUSTER_SHARED_ROOT the root; --site-dir, default $SISTEER_CONFIG's directory / $SISTEER_SITE_DIR / the site
loader's own default, clusters.default_site_dir — this file names no user config path and reads none itself) is the ONLY source
of the host and the bank root, which is `<CLUSTER_SHARED_ROOT>/exp/objects/sdfbase/full22`
VERBATIM — the guard expands nothing on the host, so a block whose root carries a `$` or `{` placeholder is refused, as
are a site dir with no config.env and a label SISTEER_CLUSTERS does not name (demo-import Q11: no literal host table, no
fallback; the former host-alias flag with its literal table is gone and argparse rejects it). Every such refusal is `GUARD: ERROR …`, exit 2,
before any ssh. --local-root runs the same function against a bank on this machine (the tests; a pulled copy) and needs
no site config.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # shipped over ssh the source runs as `python3 -`: __file__ is "<stdin>" and there is no repo around it
    HERE: Optional[Path] = Path(__file__).resolve() if __file__ != "<stdin>" else None
except NameError:
    HERE = None
REPO: Optional[Path] = HERE.parents[4] if HERE is not None and len(HERE.parents) > 4 else None
DEFAULT_PINS: Optional[Path] = (REPO / "configs" / "sdfbase_full22_pins.json") if REPO is not None else None

#: the bank under a cluster's shared root (stage_strength.py's REL), appended VERBATIM to the site block's CLUSTER_SHARED_ROOT
REL = "exp/objects/sdfbase/full22"
VIEW_MANIFEST = "VIEW_MANIFEST.json"
OBJECT = "object.pt"
SSH_TIMEOUT = 900

EXIT_PASS, EXIT_OPERATOR, EXIT_REFUSE = 0, 2, 3
DISK_NE_MANIFEST, DISK_NE_PIN, NO_PIN, NO_VIEW = "disk!=manifest", "disk!=pin", "NO PIN", "NO VIEW"
REASONS = (DISK_NE_MANIFEST, DISK_NE_PIN, NO_PIN, NO_VIEW)


class GuardError(Exception):
    """An operator error: nothing this guard can put a verdict on (exit 2). Never a PASS, never a REFUSE."""


@dataclass
class CellCheck:
    target: str
    view: str
    cell_id: str
    staged_id: str
    object_path: str
    disk: Optional[str]       # sha256 of object.pt on disk; None when it is not there
    manifest: Optional[str]   # the view's recorded sha; None when the entry records none
    pin: Optional[str]        # the roster's pin for (target, staged_id); None when there is none
    ok: bool
    reason: str               # "" when ok, else exactly one of REASONS

    def line(self) -> str:
        def short(v: Optional[str], absent: str) -> str:
            return v[:12] if v else absent
        head = "OK    " if self.ok else "REFUSE"
        s = (f"{head} {self.target}/{self.view:<9} {self.cell_id:<45} disk={short(self.disk, 'MISSING')} "
             f"manifest={short(self.manifest, 'MISSING')} pin={short(self.pin, 'NONE')}")
        return s + (f" {self.reason}" if self.reason else "")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------ #
# the pure check                                                                                                 #
# ------------------------------------------------------------------------------------------------------------ #

def pins_block(pins: dict, target: str) -> dict:
    """The target's own block of the pins file — the ONLY place its cells' pins are read from."""
    blk = pins.get(target) if isinstance(pins, dict) else None
    if not isinstance(blk, dict):
        have = sorted(pins) if isinstance(pins, dict) else type(pins).__name__
        raise GuardError(f"NO PINS BLOCK for target {target!r} (the pins file has: {have})")
    return blk


def pinned_sha(pins: dict, target: str, staged_id: str) -> Optional[str]:
    """The pinned object_sha256 for (target, staged_id): pins[target][staged_id], or pins[target][<lane>][staged_id]
    (one level of nesting). Reads NOTHING outside pins[target]: the same staged id under another target is a different
    object. None when the target pins no such id; an id pinned more than once inside the block with different shas is
    a GuardError — a guard does not pick."""
    blk = pins_block(pins, target)
    found: List[str] = []
    direct = blk.get(staged_id)
    if isinstance(direct, dict) and direct.get("object_sha256"):
        found.append(str(direct["object_sha256"]))
    for group in blk.values():                       # one level of nesting: a lane's dict of ids
        if isinstance(group, dict) and "object_sha256" not in group:
            nested = group.get(staged_id)
            if isinstance(nested, dict) and nested.get("object_sha256"):
                found.append(str(nested["object_sha256"]))
    distinct = sorted(set(found))
    if len(distinct) > 1:
        raise GuardError(f"AMBIGUOUS PIN: pins[{target!r}] pins {staged_id!r} {len(found)} times with "
                         f"{len(distinct)} different shas — fix the pins file; this guard does not pick")
    return distinct[0] if distinct else None


def check_view(bank_root: Path, target: str, view: str, pins: dict) -> List[CellCheck]:
    """Every cell of ONE view. The view directory and its manifest must exist and name at least one entry (else a
    GuardError); a cell whose object is missing is a `NO VIEW` refusal, not an exception."""
    view_dir = Path(bank_root) / target / "cells_view" / view
    manifest_p = view_dir / VIEW_MANIFEST
    if not view_dir.is_dir():
        raise GuardError(f"NO VIEW DIR: {view_dir} — the view is not built on this bank (or host/target/view is wrong)")
    if not manifest_p.is_file():
        raise GuardError(f"NO MANIFEST: {manifest_p} — the view directory exists but was never built to completion")
    try:
        vm = json.loads(manifest_p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise GuardError(f"unreadable {manifest_p}: {e}")
    entries = vm.get("entries") if isinstance(vm, dict) else None
    if not isinstance(entries, list) or not entries:
        raise GuardError(f"NO ENTRIES: {manifest_p} names no cells — nothing to check is not a PASS")
    out: List[CellCheck] = []
    for e in entries:
        cell_id, sid = str(e.get("cell_id") or ""), str(e.get("staged_id") or "")
        if not cell_id or not sid:
            raise GuardError(f"{manifest_p}: an entry without cell_id/staged_id: {e!r}")
        recorded = e.get("object_sha256") or None
        obj = view_dir / cell_id / OBJECT
        try:
            disk = sha256_file(obj) if obj.is_file() else None
        except OSError as err:
            raise GuardError(f"cannot read {obj}: {err}")
        pin = pinned_sha(pins, target, sid)
        if disk is None:
            reason = NO_VIEW
        elif disk != recorded:
            reason = DISK_NE_MANIFEST
        elif pin is None:
            reason = NO_PIN
        elif disk != pin:
            reason = DISK_NE_PIN
        else:
            reason = ""
        out.append(CellCheck(target=target, view=view, cell_id=cell_id, staged_id=sid, object_path=str(obj),
                             disk=disk, manifest=recorded, pin=pin, ok=(reason == ""), reason=reason))
    return out


def check_views(bank_root, target: str, views: Iterable[str], pins: dict) -> List[CellCheck]:
    """The pure check: every (view, cell) of `views` under `bank_root`/`target`, against `pins` (the pins FILE as a
    dict, keyed by target). Raises GuardError for an operator error; a per-cell problem is a CellCheck with ok=False
    and one of REASONS. Pure: reads files under bank_root, writes nothing."""
    pins_block(pins, target)   # the target must be pinned at all before any view is opened
    out: List[CellCheck] = []
    for view in views:
        out.extend(check_view(Path(bank_root), target, view, pins))
    return out


def verdict(checks: List[CellCheck]) -> dict:
    bad = [c for c in checks if not c.ok]
    reasons: Dict[str, int] = {}
    for c in bad:
        reasons[c.reason] = reasons.get(c.reason, 0) + 1
    return {"result": "PASS" if not bad else "REFUSE", "n_cells": len(checks), "n_refused": len(bad),
            "n_views": len({c.view for c in checks}), "reasons": reasons}


def verdict_line(v: dict) -> str:
    if v["result"] == "PASS":
        return f"GUARD: PASS ({v['n_cells']} cells)"
    return f"GUARD: REFUSE ({v['n_refused']} of {v['n_cells']} cells)"


# ------------------------------------------------------------------------------------------------------------ #
# the remote leg — screen_verdict.py's shape: this source over ssh as `python3 -`, the request base64 in argv   #
# ------------------------------------------------------------------------------------------------------------ #

def gather(req: dict) -> dict:
    """req = {bank_root, target, views, pins}. Runs the pure check where the bytes are. The bank root is used VERBATIM — it is
    the site block's literal CLUSTER_SHARED_ROOT plus REL; nothing is formatted or expanded on the host (demo-import Q11)."""
    user = getpass.getuser()
    bank_root = str(req["bank_root"])
    checks = check_views(bank_root, req["target"], req["views"], req["pins"])
    return {"ok": True, "hostname": platform.node(), "user": user, "bank_root": bank_root,
            "checks": [asdict(c) for c in checks]}


# ------------------------------------------------------------------------------------------------------------ #
# the grading host leg's ONE source of a cluster's host and root: the site's cluster block (reproduce/lib/clusters.py)  #
# ------------------------------------------------------------------------------------------------------------ #

def site_cluster_block(label: str, site_dir: Optional[str]) -> dict:
    """The site block of `label` — `{host, shared_root, bank_root, site_dir, cluster_block_label}` — read through
    reproduce/lib/clusters.py (load_addressable over --site-dir / clusters.default_site_dir(): every block under clusters/, so a
    cluster outside SISTEER_CLUSTERS' default fan-out is checked too). Refuses (GuardError, exit 2, before any ssh): no checkout
    around this file (the lib is imported from it); no config.env under the site dir; a malformed site config or block; a label
    with no block; a CLUSTER_SHARED_ROOT carrying `$` or `{` (a placeholder the guard would have to expand on the host — it
    expands nothing: clusters.py substitutes `$SHARED` alone and never `$USER`)."""
    if HERE is None or REPO is None:
        raise GuardError("--cluster resolves the site's cluster block through reproduce/lib/clusters.py and needs this file on disk "
                         "inside its checkout; run it from the checkout (or --local-root <bank> for a bank on this machine)")
    lib = str(REPO / "reproduce" / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    try:
        import clusters as CL  # noqa: WPS433 — the ONE reader of the site's cluster blocks
    except ImportError as e:
        raise GuardError(f"cannot import reproduce/lib/clusters.py from {lib}: {e}")
    sd = site_dir or CL.default_site_dir()
    if not os.path.isfile(os.path.join(sd, "config.env")):
        raise GuardError(f"NO SITE CONFIG: no config.env under {sd} — pass --site-dir, or --local-root <bank> for a bank on this machine")
    try:
        blocks = CL.load_addressable(sd)
    except SystemExit as e:            # clusters.ClusterConfigError: a malformed site config refuses by name, never a silent fallback
        raise GuardError(f"SITE CONFIG under {sd}: {e}")
    if label not in blocks:
        raise GuardError(f"UNKNOWN CLUSTER: {label!r} has no block under {sd}/clusters (blocks: {' '.join(blocks)})")
    b = blocks[label]
    root = str(b["CLUSTER_SHARED_ROOT"])
    if "$" in root or "{" in root:
        raise GuardError(f"cluster {label!r}: CLUSTER_SHARED_ROOT={root!r} carries a placeholder — a cluster block's shared root is a "
                         "literal absolute path; the guard expands nothing on the host (clusters.py substitutes $SHARED alone, never $USER)")
    return {"host": str(b["CLUSTER_SSH"]), "shared_root": root, "bank_root": root.rstrip("/") + "/" + REL, "site_dir": sd,
            "cluster_block_label": str(b.get("CLUSTER_LABEL") or "")}


def gather_remote(host: str, req: dict, ssh_timeout: int = SSH_TIMEOUT) -> dict:
    """ONE ssh round trip: this file's source on stdin, the request base64 in argv, JSON back."""
    if HERE is None:
        raise GuardError("--cluster ships this file's own source and needs it on disk; run it from the checkout")
    b64 = base64.b64encode(json.dumps(req).encode("utf-8")).decode("ascii")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, f"python3 - --gather {b64}"]
    try:
        p = subprocess.run(cmd, input=HERE.read_text(encoding="utf-8"), capture_output=True, text=True,
                           timeout=ssh_timeout)
    except subprocess.TimeoutExpired:
        raise GuardError(f"ssh {host}: no answer within {ssh_timeout}s — an unreachable cluster is an operator error, "
                         "not a PASS")
    if p.returncode != 0:
        raise GuardError(f"ssh {host} failed rc={p.returncode}: {p.stderr.strip()[-1500:]}")
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        raise GuardError(f"ssh {host}: the remote leg printed no JSON (stdout tail {p.stdout[-500:]!r}; "
                         f"stderr {p.stderr[-500:]!r})")
    if not isinstance(rec, dict) or not rec.get("ok"):
        err = rec.get("error") if isinstance(rec, dict) else None
        raise GuardError(f"{host}: {err or 'the remote leg failed without a reason'}")
    return rec


# ------------------------------------------------------------------------------------------------------------ #
# CLI                                                                                                            #
# ------------------------------------------------------------------------------------------------------------ #

def _git_sha() -> Optional[str]:
    if REPO is None:
        return None
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Optional[str], record: dict) -> None:
    if path:
        Path(path).write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gather", help=argparse.SUPPRESS)   # the remote leg: base64 JSON request -> one JSON line
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cluster", metavar="LABEL",
                     help="the SITE cluster label to check ON (clusters/<label>.env: CLUSTER_SSH + CLUSTER_SHARED_ROOT, reproduce/lib/clusters.py): "
                          "the bank under its shared root, hashed where the job will read it")
    src.add_argument("--local-root", help="a bank root on this machine (<root>/<target>/cells_view/<view>/…); needs no site config")
    ap.add_argument("--site-dir", default=None,
                    help="the site config dir whose clusters/<label>.env --cluster reads (default: $SISTEER_CONFIG's directory, else $SISTEER_SITE_DIR, "
                         "else the site loader's own default, reproduce/lib/clusters.py default_site_dir)")
    ap.add_argument("--target", help="the roster target, e.g. nemotron3nano4b / llama8b")
    ap.add_argument("--views", help="comma list of the views the sbatch will read, e.g. dm1,d0p1,d0p25,d0p5,d1,d1p5")
    ap.add_argument("--pins", default=str(DEFAULT_PINS) if DEFAULT_PINS else None,
                    help="the pins file (default: this checkout's configs/sdfbase_full22_pins.json)")
    ap.add_argument("--json", dest="json_out", help="write the full record here (full shas, host, pins sha, verdict)")
    args = ap.parse_args(argv)

    if args.gather:
        req = json.loads(base64.b64decode(args.gather).decode("utf-8"))
        try:
            out = gather(req)
        except GuardError as e:
            out = {"ok": False, "error": str(e)}
        print(json.dumps(out, sort_keys=True))
        return 0

    if not args.target or not args.views:
        ap.error("--target and --views are required")
    if not (args.cluster or args.local_root):
        ap.error("one of --cluster / --local-root is required")
    if args.site_dir and not args.cluster:
        ap.error("--site-dir goes with --cluster (--local-root reads a bank on this machine and needs no site config)")
    views: List[str] = []
    for v in (x.strip() for x in args.views.split(",")):
        if v and v not in views:
            views.append(v)
    if not views:
        ap.error("--views names no view")

    tool = {"path": str(HERE) if HERE else "<stdin>", "sha256": sha256_file(HERE) if HERE else None,
            "git_sha": _git_sha(), "argv": list(argv) if argv is not None else sys.argv[1:]}
    record: dict = {"written_at": _now(), "tool": tool, "target": args.target, "views": views,
                    "pins": {"path": args.pins, "sha256": None}, "source": None, "checks": [], "verdict": None}
    try:
        pins_path = Path(args.pins) if args.pins else None
        if pins_path is None or not pins_path.is_file():
            raise GuardError(f"NO PINS FILE at {pins_path}")
        record["pins"]["sha256"] = sha256_file(pins_path)
        pins = json.loads(pins_path.read_text(encoding="utf-8"))
        block = pins_block(pins, args.target)          # refused here, before any ssh
        if args.local_root:
            bank_root = str(Path(args.local_root).resolve())
            checks = check_views(bank_root, args.target, views, pins)
            source = {"mode": "local", "host": None, "bank_root": bank_root, "hostname": platform.node(),
                      "user": getpass.getuser()}
        else:
            site = site_cluster_block(args.cluster, args.site_dir)       # refused here, before any ssh (demo-import Q11)
            req = {"bank_root": site["bank_root"], "target": args.target, "views": views, "pins": {args.target: block}}
            g = gather_remote(site["host"], req)
            checks = [CellCheck(**c) for c in g["checks"]]
            source = {"mode": "ssh", "cluster_label": args.cluster, "host": site["host"], "shared_root": site["shared_root"],
                      "site_dir": site["site_dir"], "cluster_block_label": site["cluster_block_label"], "bank_root": g["bank_root"],
                      "hostname": g.get("hostname"), "user": g.get("user")}
    except GuardError as e:
        record["verdict"] = {"result": "ERROR", "error": str(e)}
        record["exit_code"] = EXIT_OPERATOR
        _write_json(args.json_out, record)
        print(f"GUARD: ERROR {e}")
        return EXIT_OPERATOR

    v = verdict(checks)
    record.update({"source": source, "checks": [asdict(c) for c in checks], "verdict": v,
                   "exit_code": EXIT_PASS if v["result"] == "PASS" else EXIT_REFUSE})
    _write_json(args.json_out, record)
    where = (f"ssh {source['cluster_label']} ({source['host']})" if source["mode"] == "ssh" else f"local {source['hostname']}")
    print(f"# launch guard: {where} bank={source['bank_root']} target={args.target} views={','.join(views)} "
          f"pins={args.pins} (sha {record['pins']['sha256'][:12]}) tool sha {(tool['sha256'] or '-')[:12]} "
          f"git {(tool['git_sha'] or '-')[:12]}")
    for c in checks:
        print(c.line())
    print(verdict_line(v))
    return record["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
