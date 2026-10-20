#!/usr/bin/env python3
"""operating_scripts_inventory.py inventory [--root DIR] [--out JSON]
   operating_scripts_inventory.py retire --inventory JSON [--root DIR] [--apply] [--manifest FILE]

The operating scripts' INVENTORY and their RETIREMENT PLAN. The operating tree (its root given by --root)
holds the interim landing scripts, the gate copies and ~60 queue scripts that drove the
registry's landings before the repo chain; the cutover's post-window operator stage moves the historical ones out of the way
and turns the two entry points into exec wrappers onto the repo chain, so no sentinel or stale copy is ever load-bearing again.

`inventory` walks ROOT/scripts and ROOT/gates (never ROOT/retired, never __pycache__) and classifies EVERY file:
  class      executable (.sh / .py / .sbatch, or an executable file with a shebang) | document (.md / .txt) | data (everything else)
  kind       (executables) gate (gates/**, scripts/gate-*.sh) | queue (scripts/serial-queue-*.sh, scripts/land-chain-*.sh) |
             landing (scripts/land*.sh) | window (scripts/reg-ff-chain.sh) | ruling (scripts/ruling-*) | check (scripts/post-landing-*) | other
  role       (executables) wrapper (the two entry points the operator stage rewrites: scripts/land.sh, gates/slurm/gate_slurm.sh) |
             live (a RUNNING process names the file in its command line -- /proc/<pid>/cmdline, read-only, the pids recorded --
             or a running queue script's text references it) | historical (everything else)
  modal_hits (executables) every line matching the no-Modal scan's regex (the same composed pattern as check_no_modal.sh; the
             line numbers and text are recorded so the operator can read them)
The RETIREMENT PLAN derived from the inventory: `moves` = every historical executable that has a Modal hit OR is a queue / gate
script (the 26 pattern-matching executables and every historical queue / gate script), each to
ROOT/retired/<original path>; `wrappers` = the two entry points with the exec-wrapper text that replaces them (their previous
bodies go to retired/ beside the rest); `blocked_by_live` = files the plan would move that a running process still names -- the
retirement REFUSES while any exists (the operator waits for the queue to finish); `kept` = historical executables the plan leaves
(rulings, post-landing checks, helpers without a hit). Documents and data are never moved.
`retire` applies the plan: a dry run by default (prints every move); --apply re-reads liveness, refuses by name when a move
target is live or a destination exists, moves with os.replace (same filesystem), writes the two wrappers (mode 0755), and appends
one line per action to --manifest (default ROOT/retired/retired-manifest.txt). Exit: 0; 2 usage / an unreadable inventory;
3 a refusal (live blockers, a destination that exists, a root that is not the inventory's).
Each wrapper finds the repo chain when it runs: $SISTEER_REPO, else the site config's SISTEER_REPO, else the main checkout of the
repository this tool ran from (a linked worktree's main working tree, so removing a release worktree never strands a wrapper).
Standard library only; nothing here reads outside ROOT and /proc, except `git rev-parse` on this tool's own repository."""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import time

EXEC_SUFFIXES = (".sh", ".py", ".sbatch")
DOC_SUFFIXES = (".md", ".txt")
SCAN_DIRS = ("scripts", "gates")
WRAPPER_PATHS = ("scripts/land.sh", "gates/slurm/gate_slurm.sh")
RETIRED_DIR = "retired"
SCHEMA = "operating_scripts_inventory/1"

# the no-Modal regex, composed from fragments so this module's own text is never a hit (the same pattern check_no_modal.sh builds)
_M, _U, _S, _V, _R, _G, _L, _C = "modal", "MODAL", "_", "volume", "registry", "gates/", "land/", "cpu"
PATTERN = (rf"import {_M}\b|from {_M}\b|{_M}\.(App|Function|Volume|Image|Cls)|{_M} run|{_M} {_V}|venvs/{_M}|{_U}{_S}TOKEN|"
           rf"{_R}{_S}{_V}|{_R}{_S}smoke|{_M}{_S}ops|{_V}{_S}ops|{_G}{_L}|{_C}{_S}suite|\[{_M}\]")
PATTERN_RE = re.compile(PATTERN)

#: this tool's directory (scripts/registry/landing/ of the checkout it runs from)
HERE = os.path.dirname(os.path.realpath(__file__))
#: the script each wrapper execs, relative to the checkout it resolves
WRAPPER_TARGETS = {"scripts/land.sh": "scripts/registry/landing/land.sh",
                   "gates/slurm/gate_slurm.sh": "scripts/registry/landing/gate_slurm.sh"}


def main_checkout(here: str = HERE) -> str:
    """The persistent checkout the wrappers fall back on: the main working tree of the repository holding ``here``. A retirement run
    from a linked worktree (a release branch's) records the checkout that worktree was added from, never the worktree itself, which
    is removed once its branch lands; site_env.sh's site_env_main_checkout resolves SITE_REPO's last fallback the same way. Without
    git, or with a git directory that is not <tree>/.git, ``here``'s own checkout."""
    own = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    try:
        got = subprocess.run(["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return own
    common = got.stdout.strip()
    return common[:-len("/.git")] if got.returncode == 0 and common.endswith("/.git") else own


def _dq(text: str) -> str:
    """``text`` escaped for a bash double-quoted word (``$``, a backquote, ``"`` and ``\\`` stay literal)."""
    return re.sub(r'([$`"\\])', r"\\\1", text)


def _default_site_dir_sh() -> str:
    """sisteer.site's default site directory, which the site config's one reader names, spelled for bash (a leading ``~`` as
    ``$HOME``). Read through scripts/site_paths.py, which loads this checkout's site module by file path."""
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from site_paths import _site
    d = _site().DEFAULT_SITE_DIR
    return "$HOME" + d[1:] if d.startswith("~") else d


def _exec_lines(target: str, checkout: str) -> str:
    """The wrappers' checkout, resolved when the wrapper RUNS, in site_env.sh's SITE_REPO order: $SISTEER_REPO, else the site
    config's SISTEER_REPO (the file the front door reads: $SISTEER_CONFIG, else config.env in $SISTEER_SITE_DIR or the site
    module's default directory, sourced in a subshell as the front door sources it), else ``checkout``, recorded at retire time.
    A missing target refuses by name (exit 2) rather than failing the exec."""
    return ("# the checkout: $SISTEER_REPO, else the site config's SISTEER_REPO (read now), else the main checkout recorded when this wrapper was written.\n"
            f"R=${{SISTEER_REPO:-}}; C=${{SISTEER_CONFIG:-${{SISTEER_SITE_DIR:-{_default_site_dir_sh()}}}/config.env}}\n"
            "[ -n \"$R\" ] || [ ! -r \"$C\" ] || R=$(set -a; . \"$C\" >/dev/null 2>&1; printf '%s' \"${SISTEER_REPO:-}\")\n"
            f"R=${{R:-\"{_dq(checkout)}\"}}; case \"$R\" in \"~\") R=$HOME ;; \"~/\"*) R=$HOME/${{R#\\~/}} ;; esac; R=${{R%/}}\n"
            f"[ -x \"$R/{target}\" ] || {{ echo \"REFUSED: $0: no $R/{target} (set SISTEER_REPO, in the environment or in $C, to the checkout)\" >&2; exit 2; }}\n"
            f"exec \"$R/{target}\" \"$@\"\n")


def wrapper_text(rel: str, checkout: str | None = None) -> str:
    """The wrapper that replaces ``rel`` at retire time: it lays the /tmp verdict symlink and execs the repo chain's script under the
    checkout it resolves when it runs (:func:`_exec_lines`); ``checkout`` (default :func:`main_checkout`) is the last fallback."""
    checkout = checkout if checkout is not None else main_checkout()
    if rel == "scripts/land.sh":
        return ("#!/bin/bash\n"
                "# wrapper (registry-slurm-only C9, plan section 1.8): the repo chain is the landing step; this file's previous body is retired/scripts/land.sh.\n"
                "# /tmp/gate-land-<branch>.out is laid as a symlink onto the chain's verdict file for the queue scripts' WAIT_VERDICT.\n"
                "S=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}; mkdir -p \"$S/verdicts\"; [ -n \"${1:-}\" ] && ln -sfn \"$S/verdicts/gate-land-$1.out\" \"/tmp/gate-land-$1.out\"\n"
                + _exec_lines(WRAPPER_TARGETS[rel], checkout))
    if rel == "gates/slurm/gate_slurm.sh":
        return ("#!/bin/bash\n"
                "# wrapper (registry-slurm-only C9, plan section 1.8): the repo chain is the gate; this file's previous body is retired/gates/slurm/gate_slurm.sh.\n"
                "# /tmp/gate-land-<branch>.out is laid as a symlink onto the chain's verdict file (--branch B) for the queue scripts' WAIT_VERDICT.\n"
                "S=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}; mkdir -p \"$S/verdicts\"; B=\"\"; a=(\"$@\"); for ((i=0;i<${#a[@]};i++)); do [ \"${a[$i]}\" = --branch ] && B=${a[$((i+1))]:-}; done; [ -n \"$B\" ] && ln -sfn \"$S/verdicts/gate-land-$B.out\" \"/tmp/gate-land-$B.out\"\n"
                + _exec_lines(WRAPPER_TARGETS[rel], checkout))
    raise KeyError(rel)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def classify(rel: str, path: str) -> str:
    suffix = os.path.splitext(rel)[1].lower()
    if suffix in EXEC_SUFFIXES:
        return "executable"
    if suffix in DOC_SUFFIXES:
        return "document"
    try:
        st = os.stat(path)
        if st.st_mode & stat.S_IXUSR and st.st_size > 2:
            with open(path, "rb") as fh:
                if fh.read(2) == b"#!":
                    return "executable"
    except OSError:
        pass
    return "data"


def kind_of(rel: str) -> str:
    name = os.path.basename(rel)
    if rel.startswith("gates/") or re.match(r"scripts/gate-[^/]+\.sh$", rel):
        return "gate"
    if re.match(r"scripts/(serial-queue|land-chain)-[^/]+\.sh$", rel):
        return "queue"
    if re.match(r"scripts/land[^/]*\.sh$", rel):
        return "landing"
    if rel == "scripts/reg-ff-chain.sh":
        return "window"
    if name.startswith("ruling-"):
        return "ruling"
    if name.startswith("post-landing-"):
        return "check"
    return "other"


def modal_hits(path: str) -> list:
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                if PATTERN_RE.search(line):
                    out.append({"line": n, "text": line.rstrip("\n")[:200]})
    except OSError:
        pass
    return out


def running_cmdlines(proc_root: str = "/proc") -> list:
    """Every live process's (pid, argv) this user can read; read-only."""
    out = []
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            with open(os.path.join(proc_root, pid, "cmdline"), "rb") as fh:
                argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if argv:
            out.append((int(pid), argv))
    return out


def live_pids(root: str, rels: list, cmdlines: list) -> dict:
    """rel -> the pids whose command line names THIS tree's file: its absolute path under ``root``, or the bare tree-relative path
    (a relative invocation). A token naming the same relative path under ANOTHER root is not this tree's (R7 ran the retirement on a
    COPY while a landing ran the real tree's scripts/land.sh: a suffix match refused the copy for a process that never touched it);
    those are reported by :func:`live_elsewhere` as information, never as a blocker."""
    found: dict = {}
    for rel in rels:
        absp = os.path.join(root, rel)
        for pid, argv in cmdlines:
            if any(tok == absp or tok == rel for tok in argv):
                found.setdefault(rel, []).append(pid)
    return found


def live_elsewhere(root: str, rels: list, cmdlines: list) -> dict:
    """rel -> the (pid, token) pairs naming the same tree-relative path under another root (information for a rehearsal on a copy)."""
    found: dict = {}
    for rel in rels:
        absp = os.path.join(root, rel)
        for pid, argv in cmdlines:
            for tok in argv:
                if tok != absp and tok.endswith("/" + rel):
                    found.setdefault(rel, []).append([pid, tok])
    return found


def references(text: str, rels: list) -> list:
    """The inventory paths a script's text names (by tree-relative path, e.g. scripts/land.sh)."""
    return sorted(r for r in rels if r in text)


def build_inventory(root: str, proc_root: str = "/proc") -> dict:
    root = os.path.abspath(os.path.expanduser(root))
    files = []
    for sub in SCAN_DIRS:
        top = os.path.join(root, sub)
        if not os.path.isdir(top):
            continue
        for dp, dn, fn in os.walk(top):
            dn[:] = sorted(d for d in dn if d not in ("__pycache__", RETIRED_DIR))
            for f in sorted(fn):
                path = os.path.join(dp, f)
                if os.path.islink(path) and not os.path.exists(path):
                    continue
                rel = os.path.relpath(path, root)
                cls = classify(rel, path)
                entry = {"path": rel, "class": cls, "bytes": os.path.getsize(path)}
                if cls == "executable":
                    entry["kind"] = kind_of(rel)
                    entry["modal_hits"] = modal_hits(path)
                files.append(entry)
    rels = [f["path"] for f in files if f["class"] == "executable"]
    cmdlines = running_cmdlines(proc_root)
    live = live_pids(root, rels, cmdlines)
    elsewhere = live_elsewhere(root, rels, cmdlines)
    # a running queue's text references other executables: those are live too (one hop; the queue calls them)
    texts = {}
    for rel in rels:
        try:
            with open(os.path.join(root, rel), "r", encoding="utf-8", errors="replace") as fh:
                texts[rel] = fh.read()
        except OSError:
            texts[rel] = ""
    via_queue: dict = {}
    for rel, pids in list(live.items()):
        for ref in references(texts[rel], rels):
            if ref != rel:
                via_queue.setdefault(ref, []).append(rel)
    for f in files:
        if f["class"] != "executable":
            continue
        rel = f["path"]
        f["references"] = references(texts[rel], [r for r in rels if r != rel])
        if rel in WRAPPER_PATHS:
            f["role"] = "wrapper"
        elif rel in live or rel in via_queue:
            f["role"] = "live"
        else:
            f["role"] = "historical"
        if rel in live:
            f["live_pids"] = sorted(live[rel])
        if rel in via_queue:
            f["live_via"] = sorted(set(via_queue[rel]))
        if rel in elsewhere:
            f["live_elsewhere"] = elsewhere[rel]      # the same relative path running under another root: not this tree's process
    moves, blocked, kept = [], [], []
    for f in files:
        if f["class"] != "executable":
            continue
        rel = f["path"]
        would_move = bool(f["modal_hits"]) or f["kind"] in ("queue", "gate")
        if f["role"] == "wrapper":
            continue
        if would_move and f["role"] == "live":
            blocked.append({"path": rel, "pids": f.get("live_pids", []), "via": f.get("live_via", [])})
        elif would_move:
            moves.append({"path": rel, "to": f"{RETIRED_DIR}/{rel}", "why": ("modal hit" if f["modal_hits"] else "") + ("; " if f["modal_hits"] and f["kind"] in ("queue", "gate") else "") + (f"{f['kind']} script" if f["kind"] in ("queue", "gate") else "")})
        else:
            kept.append(rel)
    wrappers = []
    checkout = main_checkout()
    for rel in WRAPPER_PATHS:
        present = any(f["path"] == rel for f in files)
        wrappers.append({"path": rel, "present": present, "body_to": f"{RETIRED_DIR}/{rel}", "text": wrapper_text(rel, checkout)})
    counts = {"files": len(files),
              "executable": sum(1 for f in files if f["class"] == "executable"),
              "document": sum(1 for f in files if f["class"] == "document"),
              "data": sum(1 for f in files if f["class"] == "data"),
              "modal_hit_executables": sum(1 for f in files if f.get("modal_hits")),
              "modal_hits": sum(len(f.get("modal_hits") or []) for f in files),
              "live": sum(1 for f in files if f.get("role") == "live"),
              "historical": sum(1 for f in files if f.get("role") == "historical"),
              "wrapper": sum(1 for f in files if f.get("role") == "wrapper"),
              "moves": len(moves), "blocked_by_live": len(blocked), "kept": len(kept)}
    rules = {"class": "executable = .sh/.py/.sbatch or an executable file with a shebang; document = .md/.txt; data = the rest",
             "kind": "gate = gates/** or scripts/gate-*.sh; queue = scripts/serial-queue-*.sh or scripts/land-chain-*.sh; landing = scripts/land*.sh; "
                     "window = scripts/reg-ff-chain.sh; ruling = ruling-*; check = post-landing-*; other",
             "role": "wrapper = the two entry points; live = a running process names THIS tree's file by its absolute or bare relative path (or a running script's "
                     "text references it); historical = the rest; live_elsewhere = the same relative path running under another root (information only)",
             "moves": "historical executables with a Modal hit or of kind queue/gate -> retired/<path>; wrappers' bodies -> retired/<path>; documents and data stay",
             "pattern": PATTERN}
    return {"schema": SCHEMA, "root": root, "utc": utc_now(), "rules": rules, "counts": counts, "files": files,
            "retirement": {"moves": moves, "wrappers": wrappers, "blocked_by_live": blocked, "kept": kept}}


def cmd_inventory(a) -> int:
    if "$" in a.root:   # the default: expanded through the site config (scripts/site_paths.py); an unset $WORK refuses by name
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        from site_paths import site_path
        a.root = str(site_path(a.root, "--root"))
    inv = build_inventory(a.root, a.proc_root)
    text = json.dumps(inv, indent=1, sort_keys=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        os.replace(tmp, a.out)
    c = inv["counts"]
    print(f"[inventory] {inv['root']}: {c['files']} files = {c['executable']} executable / {c['document']} document / {c['data']} data; "
          f"{c['modal_hit_executables']} executable(s) with {c['modal_hits']} Modal hit(s); roles: {c['live']} live, {c['historical']} historical, "
          f"{c['wrapper']} wrapper; plan: {c['moves']} move(s), {c['blocked_by_live']} blocked by a live process, {c['kept']} kept"
          + (f" -> {a.out}" if a.out else ""))
    for b in inv["retirement"]["blocked_by_live"]:
        print(f"[inventory] LIVE: {b['path']} pids={b['pids']} via={b['via']}")
    if not a.out:
        print(text)
    return 0


def cmd_retire(a) -> int:
    try:
        with open(a.inventory, "r", encoding="utf-8") as fh:
            inv = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[retire] cannot read the inventory {a.inventory}: {exc}", file=sys.stderr)
        return 2
    if inv.get("schema") != SCHEMA:
        print(f"[retire] {a.inventory} is not a {SCHEMA} document", file=sys.stderr)
        return 2
    root = os.path.abspath(os.path.expanduser(a.root)) if a.root else inv["root"]
    if a.root and os.path.abspath(os.path.expanduser(a.root)) != inv["root"] and not a.allow_other_root:
        print(f"[retire] --root {root} is not the inventory's root {inv['root']} (pass --allow-other-root for a rehearsal on a copy)", file=sys.stderr)
        return 3
    plan = inv["retirement"]
    manifest = a.manifest or os.path.join(root, RETIRED_DIR, "retired-manifest.txt")
    rels = [f["path"] for f in inv["files"] if f.get("class") == "executable"]
    # liveness re-read at apply time: the inventory is a snapshot
    live_now = live_pids(root, rels, running_cmdlines(a.proc_root)) if a.apply else {}
    blocked = list(plan.get("blocked_by_live") or [])
    for m in plan["moves"]:
        if m["path"] in live_now:
            blocked.append({"path": m["path"], "pids": live_now[m["path"]], "via": []})
    for w in plan["wrappers"]:
        if w["path"] in live_now:
            blocked.append({"path": w["path"], "pids": live_now[w["path"]], "via": []})
    actions = [("move", m["path"], m["to"]) for m in plan["moves"]]
    for w in plan["wrappers"]:
        if w.get("present"):
            actions.append(("move", w["path"], w["body_to"]))
        actions.append(("wrapper", w["path"], ""))
    print(f"[retire] {'APPLY' if a.apply else 'DRY RUN'} on {root}: {len(plan['moves'])} move(s), {len(plan['wrappers'])} wrapper(s), "
          f"{len(blocked)} blocked by a live process; manifest {manifest}")
    for kind, src, dst in actions:
        print(f"[retire] {kind} {src}" + (f" -> {dst}" if dst else ""))
    if blocked:
        for b in blocked:
            print(f"[retire] REFUSED: {b['path']} is named by a running process (pids {b['pids']}{'; via ' + ', '.join(b['via']) if b.get('via') else ''})")
        if a.apply:
            return 3
    if not a.apply:
        return 0
    for kind, src, dst in actions:
        if kind == "move":
            s, d = os.path.join(root, src), os.path.join(root, dst)
            if not os.path.exists(s):
                print(f"[retire] REFUSED: {src} is not there to move", file=sys.stderr)
                return 3
            if os.path.exists(d):
                print(f"[retire] REFUSED: the destination {dst} already exists", file=sys.stderr)
                return 3
    checkout = main_checkout()
    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as mf:
        for kind, src, dst in actions:
            if kind == "move":
                s, d = os.path.join(root, src), os.path.join(root, dst)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                os.replace(s, d)
                mf.write(f"{utc_now()} moved {src} -> {dst}\n")
            else:
                p = os.path.join(root, src)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                tmp = p + ".tmp-wrapper"
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(wrapper_text(src, checkout))
                os.chmod(tmp, 0o755)
                os.replace(tmp, p)
                mf.write(f"{utc_now()} wrapper {src}\n")
    print(f"[retire] done: {sum(1 for k, _, _ in actions if k == 'move')} moved, {sum(1 for k, _, _ in actions if k == 'wrapper')} wrapper(s) written")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inventory")
    s.add_argument("--root", default="$WORK/results-registry",
                   help="the tree to inventory (default under the site's working-data root, $WORK)")
    s.add_argument("--out", help="write the inventory JSON here (else print it)")
    s.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_inventory)
    r = sub.add_parser("retire")
    r.add_argument("--inventory", required=True)
    r.add_argument("--root", help="the tree to act on (default: the inventory's root; another root needs --allow-other-root)")
    r.add_argument("--allow-other-root", action="store_true", help="a rehearsal on a copy of the tree")
    r.add_argument("--apply", action="store_true", help="perform the moves and write the wrappers (default: dry run)")
    r.add_argument("--manifest", help="append one line per action here (default ROOT/retired/retired-manifest.txt)")
    r.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    r.set_defaults(fn=cmd_retire)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
