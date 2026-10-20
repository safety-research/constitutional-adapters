#!/usr/bin/env python
"""STRENGTH FRONTIER: stage the built dose-scaled copies (one ledger row per cell, each with its own `scale`) from the grading host bank onto BOTH clusters, verified, and
record them in the staging agent's manifest so the pins can be refreshed (make_full22_configs.py
--pins-from).

The same contract as $STAGE/sdfbase/full22_objects/stage_full22.sh (whose records this writes,
in its schema, under _records/<target>__<staged_id>.json): rsync with the MooseFS-safe flags
(-rlt --no-owner --no-group --no-perms --partial --inplace), then a remote probe of object.pt's sha256
and the file count; a cell is `staged` only when BOTH clusters verify. Afterwards `stage_full22.sh
--manifest-only` rebuilds staging_manifest.json from every record and pushes it to both clusters.

  python scripts/sdfbase/full22/strength/stage_strength.py [--ledger <build_ledger.json>] [--only sid,...]
        [--clusters cluster_a,cluster_b] [--dry-run] [--no-manifest]
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = None      # the local bank: --bank, else $STAGE/sdfbase/full22_objects (site_path; resolved in main when first needed)
RECORDS = None   # ROOT / "_records"
REL = "exp/objects/sdfbase/full22"
# Each cluster's login host and shared root come from its block in the site config (reproduce/lib/clusters.py: CLUSTER_SSH,
# CLUSTER_SHARED_ROOT), from --site-dir, else the file $SISTEER_CONFIG names, else $SISTEER_SITE_DIR/config.env. A cluster is
# named by its registry label (the default list) or by the site's own label for it; one the site config does not describe is
# refused by name when it is first needed. The staging records key each cluster by its registry label, the name the pins read.
_SITE_BLOCKS: dict = {}


def _clusters_lib():
    lib = str(Path(__file__).resolve().parents[4] / "reproduce" / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    import clusters as CL  # noqa: WPS433
    return CL


def load_site(site_dir: str | None) -> dict:
    """The site's cluster blocks, every one under clusters/ whether SISTEER_CLUSTERS lists it or not, since each is reached by the
    label a --clusters entry names (empty when no site config exists; a cluster is then refused when it is first needed)."""
    global _SITE_BLOCKS
    CL = _clusters_lib()
    try:
        if site_dir:
            if not os.path.isfile(os.path.join(site_dir, "config.env")):   # named explicitly: a missing config is a refusal
                raise SystemExit(f"REFUSE: --site-dir {site_dir}: no config.env there")
            _SITE_BLOCKS = CL.load_addressable(site_dir)
        else:
            path = CL.default_config_path()
            _SITE_BLOCKS = CL.load_addressable(os.path.dirname(path), CL.site_config_file(path)) if os.path.isfile(path) else {}
    except SystemExit as exc:   # a malformed site config is a refusal, never a silent fallback
        raise SystemExit(f"REFUSE: site config: {exc}")
    return _SITE_BLOCKS


def block_of(cluster: str) -> dict:
    try:
        return _SITE_BLOCKS[_clusters_lib().resolve_label(_SITE_BLOCKS, cluster)]
    except SystemExit as exc:
        raise SystemExit(f"REFUSE: {str(exc).removeprefix('REFUSED: ')}")


def host_of(cluster: str) -> str:
    return block_of(cluster)["CLUSTER_SSH"]


def shared_of(cluster: str) -> str:
    return block_of(cluster)["CLUSTER_SHARED_ROOT"]


def record_key(cluster: str) -> str:
    """The name a staging record keys this cluster by: its registry label, the name the pins read. A cluster whose registry
    label is not its own (the generic substrate, or one another block carries too) keeps the label it was named by."""
    sub = block_of(cluster).get("CLUSTER_SUBSTRATE", "")
    if sub and sub != "slurm" and sum(1 for b in _SITE_BLOCKS.values() if b.get("CLUSTER_SUBSTRATE") == sub) == 1:
        return sub
    return cluster


RSYNC = ["rsync", "-rlt", "--no-owner", "--no-group", "--no-perms", "--partial", "--inplace", "--stats"]
BANK_SPEC = "$STAGE/sdfbase/full22_objects"                # the grading host's staging directory (SISTEER_STAGE_DIR)
LEDGER_SPEC = "$RUNSTATE/strength/objects/build_ledger.json"   # the lane's run-state tree (SISTEER_RUN_STATE_ROOT)


def site_path(spec: str, site_dir: str | None, flag: str) -> Path:
    """``spec`` expanded by the site module (sisteer.site: the setting from the environment, else the site config, --site-dir's
    when given). Unset in both, the run is refused by name, before anything is read or copied."""
    src = str(Path(__file__).resolve().parents[4] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site as S  # noqa: WPS433
    try:
        st = S.load_site(os.path.join(site_dir, "config.env")) if site_dir else None
        return S.expand_path(spec, site=st, key=flag)
    except (S.PlaceholderError, S.SiteConfigError) as exc:
        raise SystemExit(f"REFUSE: {exc}; or pass {flag} <path>")


def write_record_atomic(dest: Path, rec: dict) -> None:
    """Write a staging record through a LANE-SCOPED temp path, then os.replace it into place.

    _records/ is shared by every lane's stager. stage_full22.sh keeps its in-flight scratch as
    _records/<key>.<cluster>.tmp, and two concurrent runs staging the SAME key clobber each other's scratch
    (2026-09-08 18:55Z-19:01Z: the GLM lane's run and this lane's glm_chain step1 both staged glm46vflash/base,
    and the second run hit FileNotFoundError on a .tmp the first had already consumed). This lane's records
    carry lane-unique keys (halfdose__*/halfdoseit__*), so the remaining hazard is a reader (a manifest rebuild)
    catching a half-written .json — closed by writing under .halfdose-tmp/ and renaming, which is atomic on one
    filesystem."""
    tmpdir = dest.parent / ".strength-tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp = tmpdir / f"{dest.name}.{os.getpid()}"
    tmp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, dest)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote(host: str, script: str, attempts: int = 3) -> str:
    for i in range(attempts):
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, "bash", "-s"],
                           input=script, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"ssh {host} failed: {p.stderr[-500:]}")


def probe(host: str, d: str) -> tuple[str, int]:
    out = remote(host, f'd={shlex.quote(d)}\nif [ -f "$d/object.pt" ]; then sha256sum "$d/object.pt" | cut -d" " -f1; else echo MISSING; fi\n'
                       f'find "$d" -type f 2>/dev/null | wc -l')
    lines = out.strip().splitlines()
    return lines[0], int(lines[1])


def stage_one(cluster: str, local_dir: Path, remote_dir: str, sha: str, n: int, dry: bool) -> dict:
    host = host_of(cluster)
    rsha, rcount = probe(host, remote_dir)
    if rsha == sha and rcount == n:
        return {"action": "already_staged", "verified": True, "remote_sha256": rsha, "remote_n_files": rcount, "bytes_moved": 0}
    if dry:
        return {"action": "would_rsync", "verified": False, "remote_sha256": rsha, "remote_n_files": rcount, "bytes_moved": 0}
    remote(host, f"mkdir -p {shlex.quote(remote_dir)}")
    p = subprocess.run(RSYNC + [str(local_dir) + "/", f"{host}:{remote_dir}/"], capture_output=True, text=True)
    moved = 0
    for line in p.stdout.splitlines():
        if line.startswith("Total transferred file size:"):
            moved = int(line.split(":")[1].split("bytes")[0].strip().replace(",", ""))
    action = "rsynced" if p.returncode == 0 else "rsync_failed"
    rsha, rcount = probe(host, remote_dir)
    return {"action": action, "verified": (rsha == sha and rcount == n), "remote_sha256": rsha, "remote_n_files": rcount,
            "bytes_moved": moved, "rsync_rc": p.returncode, "rsync_err": p.stderr[-400:] if p.returncode else ""}


def main(argv=None) -> int:
    global ROOT, RECORDS
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ledger", type=Path, default=None, help=f"the build ledger (default {LEDGER_SPEC}, through the site)")
    ap.add_argument("--only", default="")
    ap.add_argument("--clusters", default="cluster_a,cluster_b")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--site-dir", default=None, help="the site config dir whose clusters/<label>.env blocks route the labels (default: $SISTEER_SITE_DIR)")
    ap.add_argument("--bank", type=Path, default=None, help=f"the local bank whose _records/ + staging_manifest.json this writes (default {BANK_SPEC}, through the site)")
    args = ap.parse_args(argv)
    load_site(args.site_dir)
    if args.bank is not None:
        ROOT = args.bank; RECORDS = ROOT / "_records"
    elif RECORDS is None:
        ROOT = site_path(BANK_SPEC, args.site_dir, "--bank"); RECORDS = ROOT / "_records"
    if ROOT is None:            # RECORDS set by a caller alone: the bank is its parent
        ROOT = RECORDS.parent
    ledger_path = args.ledger if args.ledger is not None else site_path(LEDGER_SPEC, args.site_dir, "--ledger")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    only = {s for s in args.only.split(",") if s}
    clusters = [c for c in args.clusters.split(",") if c]
    RECORDS.mkdir(parents=True, exist_ok=True)
    rc = 0
    for cell in ledger["cells"]:
        if only and cell["staged_id"] not in only:
            continue
        local_dir = Path(cell["out"])
        target, sid = cell["target"], cell["staged_id"]
        if not (local_dir / "object.pt").is_file():   # a retired / moved cell: REFUSE the row, never crash the run mid-write
            print(f"REFUSE {target}/{sid}: ledger row {sid}: out path missing ({local_dir}) — retire the row or restore the object"); rc = 2; continue
        sha = sha256_file(local_dir / "object.pt")
        if sha != cell["object_sha256"]:
            print(f"REFUSE {target}/{sid}: local object.pt {sha[:12]} != ledger {cell['object_sha256'][:12]}"); rc = 2; continue
        staged_from = json.loads((local_dir / "staged_from.json").read_text(encoding="utf-8"))
        n = sum(1 for p in local_dir.rglob("*") if p.is_file())
        nbytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
        rec = {"target": target, "lane": sid.split("__", 1)[0], "cell_id": cell["cell_id"], "staged_id": sid,
               "source_path": f"grading_host:{local_dir} (dose-scaled copy x{cell['scale']} of {cell['parent_staged_id']}, parent "
                              f"object_sha256 {cell['parent_object_sha256']}; {cell['producer']})",
               "object_sha256": sha, "n_files": n, "n_files_source": n - 1, "bytes": nbytes}
        allok = True
        for c in clusters:
            rdir = f"{shared_of(c)}/{REL}/{target}/cells/{sid}"
            r = stage_one(c, local_dir, rdir, sha, n, args.dry_run)
            key = record_key(c)
            rec[f"{key}_path"] = rdir
            for k in ("verified", "action", "remote_sha256", "remote_n_files", "bytes_moved"):
                rec[f"{key}_{k}"] = r[k]
            allok &= bool(r["verified"])
            print(f"  [{c}] {target}/{sid}: {r['action']} verified={r['verified']} remote_sha={str(r['remote_sha256'])[:12]} n={r['remote_n_files']}"
                  + (f" ERR {r.get('rsync_err')}" if r.get("rsync_rc") else ""))
        hc = staged_from.get("recorded_hash_check") or {}
        rec.update({"pulled_at": staged_from.get("pulled_at"), "local_path": str(local_dir), "local_copy": "kept",
                    "object_sha256_recorded_match": hc.get("object_sha256_recorded_match"),
                    "object_identical_to_continued_from_prior": hc.get("object_identical_to_continued_from_prior"),
                    "recorded_hash_check": hc, "dose_scaled_from": {"parent_staged_id": cell["parent_staged_id"],
                    "parent_object_sha256": cell["parent_object_sha256"], "scale": cell["scale"], "producer": cell["producer"],
                    "object_type": cell["object_type"], "scaling_method": cell["scaling_method"]},
                    "status": "staged" if allok else "verify_failed"})
        if not args.dry_run:
            write_record_atomic(RECORDS / f"{target}__{sid}.json", rec)
        if not allok:
            rc = 2
    if not args.dry_run and not args.no_manifest:
        # serialize THIS lane's rebuilds behind a lock (another lane's stage_full22.sh run takes no lock; the rebuild only
        # READS *.json records, so the remaining hazard is two rebuilds racing to push the same manifest — last writer
        # wins, both having read every record)
        with open(RECORDS / ".strength-manifest.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if (ROOT / "stage_full22.sh").is_file():
                p = subprocess.run(["bash", str(ROOT / "stage_full22.sh"), "--manifest-only"], capture_output=True, text=True)
                print(p.stderr[-1500:])
                if p.returncode != 0:
                    print(f"manifest rebuild rc={p.returncode}"); rc = 2
            else:
                # the in-repo rebuild (scripts/sdfbase/full22/stage_objects.py): the same records -> staging_manifest.json, pushed to each cluster
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                import stage_objects as SO  # noqa: WPS433
                manifest = ROOT / "staging_manifest.json"
                recs = SO.rebuild_manifest(RECORDS, manifest)
                for c in clusters:
                    if not SO.push_manifest(SO.Host(c, host_of(c), shared_of(c), False), REL, manifest):
                        print(f"manifest -> {c}: sha MISMATCH after the push"); rc = 2
                print(f"MANIFEST {manifest}: {len(recs)} records (in-repo rebuild)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
