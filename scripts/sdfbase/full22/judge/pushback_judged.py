"""Push judged results from the grading host mirror back into the cluster stores,
APPEND-ONLY and verified:

  for every mirrored cell:
    1. remote rows.jsonl sha256 must equal the pre-judge snapshot
       (.rows_sha_before_judge.json, written on the cluster before mirroring)
       -> nobody wrote the store in between;
    2. the local rows.jsonl must START with exactly those bytes
       -> the judge only appended;
    3. rsync rows.jsonl + judge_summary.json + judge_items.jsonl
       (rsync's default temp-file + rename, so each file lands atomically);
    4. re-read the remote sha256 and require it to equal the local one.

Never touches private/ (the generations already live there) and never
deletes anything. Refuses loudly on any mismatch and moves to the next cell.

usage: python pushback_judged.py [--check-only] [--match SUBSTR]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LOCAL = "$GRADE/sdfbase_full22/controls"   # the grading host mirror, under the grade root ($GRADE): main() expands it
import os
#: the cluster holding the controls stores (a site or registry label): its login host (CLUSTER_SSH) and shared root
#: (CLUSTER_SHARED_ROOT) come from its site block (reproduce/lib/clusters.py) when main() runs; F22_LOGIN_HOST and
#: F22_CONTROLS_REMOTE override them
CONTROLS_CLUSTER = os.environ.get("F22_CONTROLS_CLUSTER", "cluster_a")
REMOTE_HOST = os.environ.get("F22_LOGIN_HOST") or None
REMOTE = os.environ.get("F22_CONTROLS_REMOTE") or None
FILES = ("rows.jsonl", "judge_summary.json", "judge_items.jsonl")


def sha_local(p: Path, nbytes=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read() if nbytes is None else f.read(nbytes))
    return h.hexdigest()


def remote_shas(rel_paths):
    """One ssh round trip: sha256 of every remote path (missing -> None).
    The script (with the path list embedded as JSON) goes over stdin, so
    no shell quoting is involved."""
    script = ("import hashlib,json,os\n"
              f"paths = json.loads({json.dumps(json.dumps(list(rel_paths)))})\n"
              "for p in paths:\n"
              "    print(p, hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.exists(p) else 'MISSING')\n")
    out = subprocess.run(["ssh", REMOTE_HOST, "python3", "-"], input=script, text=True,
                         capture_output=True, check=True).stdout
    res = {}
    for l in out.splitlines():
        if l.strip():
            path, sha = l.rsplit(" ", 1)
            res[path] = None if sha == "MISSING" else sha
    return res


def _site_path(value) -> Path:
    """``value`` with the grade root's placeholder ($GRADE) expanded through sisteer.site, ``~`` expanded."""
    src = str(Path(__file__).resolve().parents[4] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site
    return site.expand_path(value)


def _resolve_remote() -> None:
    """REMOTE_HOST and REMOTE from the controls cluster's site block, where the environment did not name them."""
    global REMOTE_HOST, REMOTE
    if REMOTE_HOST and REMOTE:
        return
    lib = str(Path(__file__).resolve().parents[4] / "reproduce" / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import clusters as CL  # noqa: WPS433 (the site's cluster blocks)
    block = CL.block_for(CONTROLS_CLUSTER)
    if not (block.get("CLUSTER_SSH") and block.get("CLUSTER_SHARED_ROOT")):
        raise SystemExit(f"the site config names no login host or shared root (CLUSTER_SSH, CLUSTER_SHARED_ROOT) for cluster {CONTROLS_CLUSTER}")
    REMOTE_HOST = REMOTE_HOST or block["CLUSTER_SSH"]
    REMOTE = REMOTE or f"{block['CLUSTER_SHARED_ROOT'].rstrip('/')}/exp/sdfbase_full22/controls"


def main():
    _resolve_remote()
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--match", default="")
    ap.add_argument("--targets", default="", help="comma list of target dirs to push (default all)")
    args = ap.parse_args()
    local = _site_path(LOCAL)
    want_targets = {x for x in args.targets.split(",") if x}
    snap = json.loads(subprocess.check_output(
        ["ssh", REMOTE_HOST, f"cat {REMOTE}/.rows_sha_before_judge.json"], text=True))
    cells = []
    for t in sorted(p.name for p in local.iterdir() if (p / "cells").is_dir()):
        if want_targets and t not in want_targets:
            continue
        for d in sorted((local / t / "cells").glob("*__controls__*")):
            if args.match and args.match not in d.name:
                continue
            if not (d / "judge_summary.json").exists():
                print(f"SKIP {t}/{d.name}: no judge_summary.json locally")
                continue
            cells.append((t, d))
    remote_rows = remote_shas([f"{REMOTE}/{t}/cells/{d.name}/rows.jsonl" for t, d in cells])
    plan = []
    for t, d in cells:
        rel = f"{t}/{d.name}"            # the snapshot key
        rdir = f"{REMOTE}/{t}/cells/{d.name}"  # the remote store dir
        rp = f"{rdir}/rows.jsonl"
        before = snap.get(rel)
        if before is None:
            print(f"REFUSE {rel}: not in the pre-judge snapshot"); continue
        local_full = sha_local(d / "rows.jsonl")
        if remote_rows.get(rp) == local_full:
            # rows already pushed and unchanged since: nothing to prove, sidecars only
            print(f"IN-SYNC {rel}: remote rows.jsonl == local; pushing sidecars only")
            plan.append((t, d, rel, rdir)); continue
        if remote_rows.get(rp) != before["rows_sha"]:
            print(f"REFUSE {rel}: remote rows.jsonl changed since the snapshot "
                  f"({remote_rows.get(rp)} != {before['rows_sha']})"); continue
        local_prefix = sha_local(d / "rows.jsonl", before["rows_bytes"])
        if local_prefix != before["rows_sha"]:
            print(f"REFUSE {rel}: local rows.jsonl does not START with the original bytes "
                  "(not an append)"); continue
        n_local = sum(1 for _ in open(d / "rows.jsonl"))
        print(f"OK-TO-PUSH {rel}: local rows {n_local} lines, "
              f"+{(d / 'rows.jsonl').stat().st_size - before['rows_bytes']} bytes appended")
        plan.append((t, d, rel, rdir))
    if args.check_only:
        print(f"check-only: {len(plan)} of {len(cells)} cells would push")
        return 0
    pushed = 0
    for t, d, rel, rdir in plan:
        srcs = [str(d / f) for f in FILES if (d / f).exists()]
        subprocess.run(["rsync", "-t", "--no-owner", "--no-group", "--no-perms", *srcs,
                        f"{REMOTE_HOST}:{rdir}/"], check=True)
        pushed += 1
    # verify every pushed file byte-for-byte
    want = {}
    for t, d, rel, rdir in plan:
        for f in FILES:
            if (d / f).exists():
                want[f"{rdir}/{f}"] = sha_local(d / f)
    got = remote_shas(list(want))
    bad = [p for p, s in want.items() if got.get(p) != s]
    print(f"pushed {pushed} cells, verified {len(want) - len(bad)} of {len(want)} files")
    if bad:
        print("MISMATCH after push:", *bad, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
