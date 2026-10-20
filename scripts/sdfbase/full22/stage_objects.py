#!/usr/bin/env python3
"""scripts/sdfbase/full22/stage_objects.py — stage fitted objects onto EVERY configured cluster, sha-verified, and write
the staging manifest the config generator reads (the in-repo implementation of the out-of-tree stage_full22.sh
contract, already reimplemented for doses by strength/stage_strength.py; stage 05). PyYAML + stdlib.

Contract (the ONLY shape `make_full22_configs.py pins_from_staging_manifest` reads): `_records/<target>__<staged_id>.json`
and `staging_manifest.json` rows {target, lane, cell_id, staged_id, object_sha256, source_path | volume_path, n_files,
bytes, pulled_at, local_path, <cluster>_path, <cluster>_verified, <cluster>_action, <cluster>_remote_sha256,
<cluster>_remote_n_files, <cluster>_bytes_moved, recorded_hash_check, status} (status also ``digest_mismatch`` / ``remote_manifest_conflict`` for a cell refused before any copy). Transfer: rsync with the MooseFS-safe flags
(-rlt --no-owner --no-group --no-perms --partial --inplace), a remote probe of object.pt's sha256 + the file count, a
cell `staged` only when EVERY named cluster verifies. Idempotent: a remote copy that already matches is skipped.
Chunked objects (reproduce/lib/object_chunks.py: object.pt.part-NN + CHUNKS.json) are reassembled into the bank first.
Cluster hosts / roots come from the site's cluster blocks (reproduce/lib/clusters.py) — nothing here names a host; the
two known labels are ALSO accepted by name for the paper's four models' existing records (cluster_a / cluster_b).

  stage_objects.py --site-dir DIR --bank DIR --target K --cells cells.json [--clusters a,b] [--dry-run] [--rel exp/objects/sdfbase/full22]
  cells.json: [{"cell_id": "...", "lane": "base", "source": "<label>:<abs dir>" | "local:<abs dir>" | "modal:<vol>:<path>" | "committed:<repo-relative dir>",
                "kind": "fit|derived", "object_sha256_expected": "<64 hex>" (optional)}]
demo-import (M1): a ``committed:<repo-relative dir>`` source is one of the paper's committed objects (objects/<model_id>/<object_id>/, listed in
objects/MANIFEST.json; reproduce/tools/import_objects.py cells writes the rows) — copied from THIS checkout, no ssh pull; its records and the
manifest rows carry it under ``source_path`` like every other kind. Readiness: ``object.pt`` plus exactly ONE identity sidecar
(train_budget_meta.json / derived_meta.json / fit_provenance.json — sisteer.eval.train_budget.IDENTITY_SIDECARS) that names the cell; a
``kind: fit`` cell still needs its ``COMPLETE`` marker, and a PRESENT marker must be coherent (its ``files`` exist with the recorded sha256),
which admits the committed refusal direction (fit_provenance.json + object.pt) and keeps every fit's COMPLETE check.
s3r-fix: a row carrying ``object_sha256_expected`` (05's cells_json: the committed pin = the REGISTRY's object id) is held to it —
the pulled object.pt's digest is compared BEFORE any cluster copy, before the cell is accepted as staged and before the manifest the pin rewrite
reads: a mismatch refuses by name (status ``digest_mismatch``, both digests recorded, nothing rsynced, rc 2), so a stale but self-consistent
object.pt can never be promoted to the pin and served as the paper object under a different registry id. A row without it (a first-time fit)
records the discovered digest, as before.
stager-fix-2: the shared manifest is read on EVERY cluster BEFORE any object bytes move; a record there whose identity
(target, staged_id) this bank holds under a DIFFERENT object_sha256 (or one side missing it) is a CONFLICT — two objects claim one pin — refused
by name with both values and both records, at the bank-level preflight and again per cell before stage_to (status ``remote_manifest_conflict``,
nothing rsynced); --merge-remote never makes a side win it. The publication reads every cluster first, folds all foreign records into the
bank, builds ONE union and publishes that same union to each cluster by compare-and-swap on the remote file's digest (temp file + rename;
a remote that moved since the read is re-read, re-folded and every cluster gets the new union again).
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import datetime as _dt
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import clusters as CL  # noqa: E402
import object_chunks as OC  # noqa: E402

RSYNC = ["rsync", "-rlt", "--no-owner", "--no-group", "--no-perms", "--partial", "--inplace", "--stats"]
REL_DEFAULT = "exp/objects/sdfbase/full22"
HEX = re.compile(r"^[0-9a-f]{64}$")
KEYRE = re.compile(r"sha|hash|digest|fingerprint", re.I)


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    return OC.sha256_file(str(p))


class Host:
    def __init__(self, label: str, ssh: str, shared: str, dry: bool):
        self.label, self.ssh, self.shared, self.dry = label, ssh, shared, dry
        self.local = ssh.startswith("local:")

    def run(self, script: str, attempts: int = 3) -> str:
        if self.dry:
            print(f"DRY-RUN: ssh {self.ssh} bash -s <<< {shlex.quote(script[:120])}"); return ""
        for i in range(attempts):
            if self.local:
                p = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True)
            else:
                p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", self.ssh, "bash", "-s"], input=script, capture_output=True, text=True)
            if p.returncode == 0:
                return p.stdout
            time.sleep(5 * (i + 1))
        raise RuntimeError(f"[{self.label}] remote failed: {p.stderr[-400:]}")

    def probe(self, d: str) -> Tuple[str, int]:
        out = self.run(f'd={shlex.quote(d)}\nif [ -f "$d/object.pt" ]; then sha256sum "$d/object.pt" | cut -d" " -f1; else echo MISSING; fi\nfind "$d" -type f 2>/dev/null | wc -l')
        if self.dry:
            return "DRY", -1
        lines = out.strip().splitlines()
        return lines[0], int(lines[1])

    def rsync_dest(self) -> str:
        return "" if self.local else f"{self.ssh}:"


def record_key(blocks: dict, label: str) -> str:
    """The name a staging record keys a cluster by: its registry label (CLUSTER_SUBSTRATE), the name the pins read
    (`<registry label>_verified`). A cluster whose registry label is not its own (the generic substrate, or one another block
    carries too) keeps its site label."""
    sub = str(blocks[label].get("CLUSTER_SUBSTRATE", ""))
    if sub and sub != "slurm" and sum(1 for b in blocks.values() if b.get("CLUSTER_SUBSTRATE") == sub) == 1:
        return sub
    return label


def hosts_by_label(blocks: dict, dry: bool) -> Dict[str, Host]:
    """Every configured cluster's Host, keyed by its site label and by its registry label (CLUSTER_SUBSTRATE): a cell's source
    names its cluster by the registry label the pins carry, whichever clusters a run stages to. A registry label more than one
    block carries (the generic substrate) is no key."""
    by_site = {lab: Host(lab, b["CLUSTER_SSH"], b["CLUSTER_SHARED_ROOT"], dry) for lab, b in blocks.items()}
    out = dict(by_site)
    for name in {str(b.get("CLUSTER_SUBSTRATE", "")) for b in blocks.values()} - {""} - set(blocks):
        try:
            out[name] = by_site[CL.resolve_label(blocks, name)]
        except SystemExit:
            pass
    return out


def committed_withheld(rest: str) -> Optional[dict]:
    """The objects/WITHHELD.json row of a `committed:` source's object (a paper object this repository does not ship), or None;
    the list is read beside the manifest (sisteer.objects.WITHHELD_REL), stdlib only."""
    wfile = REPO / "objects" / "WITHHELD.json"
    if not wfile.is_file():
        return None
    rows = {str(w.get("object_id")): w for w in (json.loads(wfile.read_text(encoding="utf-8")).get("objects") or [])}
    return rows.get(Path(rest.strip("/")).name)


def pull_source(src: str, dest: Path, hosts: Dict[str, Host], dry: bool) -> None:
    kind, _, rest = src.partition(":")
    if kind == "committed":
        held = committed_withheld(rest)
        if held is not None:          # refused in a dry run too: no plan stages an object the checkout cannot hold
            # named by its object id (sha) and model only: the stager never reads a label (test_stage_objects_registry_id)
            raise SystemExit(f"REFUSED: source {src!r}: object {held['object_id']} ({held.get('model_id')}) "
                             f"is withheld from this repository: {held.get('reason')} (objects/WITHHELD.json)")
    if dry:
        print(f"DRY-RUN: pull {src} -> {dest}"); return
    dest.mkdir(parents=True, exist_ok=True)
    if kind == "local":
        subprocess.run(RSYNC[:-1] + [rest.rstrip("/") + "/", str(dest) + "/"], check=True, capture_output=True)
    elif kind == "committed":
        # demo-import: a committed object directory of THIS checkout (objects/<model_id>/<object_id>), repo-relative, never a host
        d = REPO / rest.strip("/")
        if os.path.isabs(rest) or ".." in Path(rest).parts or not d.is_dir():
            raise SystemExit(f"REFUSED: source {src!r}: a committed source is a repo-relative object directory of this checkout ({REPO}); "
                             f"{d} is not one")
        subprocess.run(RSYNC[:-1] + [str(d) + "/", str(dest) + "/"], check=True, capture_output=True)
    elif kind == "modal":
        vol, _, path = rest.partition(":")
        tmp = dest.parent / f".pull-{dest.name}"
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run([os.environ.get("SISTEER_MODAL_PYTHON", "modal"), "volume", "get", "--force", vol, path, str(tmp) + "/"], check=True, capture_output=True)
        got = tmp / Path(path).name
        subprocess.run(RSYNC[:-1] + [str(got) + "/", str(dest) + "/"], check=True, capture_output=True)
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    else:
        h = hosts.get(kind)
        if h is None:
            raise SystemExit(f"REFUSED: source {src!r}: {kind!r} is neither local, modal nor a configured cluster label ({' '.join(hosts)})")
        subprocess.run(RSYNC[:-1] + [f"{h.rsync_dest()}{rest.rstrip('/')}/", str(dest) + "/"], check=True, capture_output=True)


#: the identity sidecars a staged directory may declare, exactly ONE of them (sisteer.eval.train_budget.IDENTITY_SIDECARS; spelled here so this
#: script stays PyYAML + stdlib — tests/test_stage_objects_committed.py holds the two equal)
IDENTITY_SIDECARS = ("train_budget_meta.json", "derived_meta.json", "fit_provenance.json")


def _sidecar_names_cell(name: str, doc: dict, cell_id: str) -> bool:
    """The identity sidecar names THIS cell: a fit record's `cell_id`, a derived record's `derived_id`, a fit-provenance record's `staged_id`
    (`<rung>__<harness id>`) or its `cell_id` (the harness id the staged id carries after its rung)."""
    if name == "train_budget_meta.json":
        return str(doc.get("cell_id")) == cell_id
    if name == "derived_meta.json":
        return str(doc.get("derived_id")) == cell_id
    return str(doc.get("staged_id")) == cell_id or str(doc.get("cell_id")) == cell_id or ("__" in cell_id and str(doc.get("cell_id")) == cell_id.split("__", 1)[1])


def complete_marker_problem(d: Path) -> Optional[str]:
    """A PRESENT `COMPLETE` marker must be coherent: a JSON manifest whose `files` all exist with the recorded sha256 (a torn or tampered
    publication never reads as a valid cell; an ABSENT marker is the caller's rule to judge). None when coherent or absent."""
    p = d / "COMPLETE"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None if not p.read_text(encoding="utf-8", errors="replace").lstrip().startswith("{") else "COMPLETE is not valid JSON"
    if not isinstance(doc, dict):
        return "COMPLETE is not a JSON object"
    files = doc.get("files")
    if not isinstance(files, dict) or not files:
        return "COMPLETE names no files"
    for name, want in sorted(files.items()):
        f = d / str(name)
        if not f.is_file():
            return f"COMPLETE names {name}, which is absent"
        got = sha256_file(f)
        if str(want) != got:
            return f"COMPLETE records {name} as {str(want)[:12]} but it hashes to {got[:12]}"
    return None


def readiness_problem(d: Path, cell_id: str, kind: str) -> Optional[str]:
    """Why a pulled cell directory is NOT READY to stage (None when it is): `object.pt` present; exactly one identity sidecar, naming the
    cell; a `kind: fit` cell carries its COMPLETE marker; a present marker is coherent."""
    if not (d / "object.pt").is_file():
        return "no object.pt"
    present = [s for s in IDENTITY_SIDECARS if (d / s).is_file()]
    if len(present) != 1:
        return f"exactly one identity sidecar of {list(IDENTITY_SIDECARS)} is required, found {present or 'none'}"
    name = present[0]
    try:
        doc = json.loads((d / name).read_text(encoding="utf-8"))
    except ValueError as exc:
        return f"{name} is not valid JSON ({exc})"
    if not isinstance(doc, dict) or not _sidecar_names_cell(name, doc, cell_id):
        return f"{name} does not name the cell {cell_id!r}"
    if kind != "derived" and not (d / "COMPLETE").is_file():
        return f"a kind: {kind} cell needs its COMPLETE marker (found {name}, no COMPLETE)"
    bad = complete_marker_problem(d)
    if bad:
        return bad
    return None


def hashcheck_and_stamp(d: Path, source: str, target: str, lane: str, cell_id: str, staged_id: str) -> Tuple[dict, str]:
    """Hash every staged file ONCE (object.pt's sha is the object id), compare the recorded hashes in the cell's JSON sidecars
    against it, write staged_from.json; returns (the hash-check record, the object sha)."""
    entries = []

    def walk(o, path, fname):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k, fname)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]", fname)
        elif isinstance(o, str) and HEX.match(o) and KEYRE.search(path):
            entries.append({"file": fname, "key": path, "value": o, "matches_object": o == sha})
    staged_shas = {}
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        rel = str(p.relative_to(d)); staged_shas[rel] = sha256_file(p)
    sha = staged_shas["object.pt"]
    for p in sorted(x for x in d.rglob("*.json") if x.is_file()):
        rel = str(p.relative_to(d))
        if p.name != "staged_from.json":
            try:
                walk(json.load(open(p)), "", rel)
            except Exception as exc:
                entries.append({"file": rel, "key": "<unparseable>", "value": str(exc), "matches_object": False})
    own = [e for e in entries if e["key"] == "object_sha256"]
    status = "match" if any(e["matches_object"] for e in entries) else ("recorded_hashes_are_not_this_object" if entries else "none_recorded")
    hc = {"status": status, "object_sha256_recorded_match": (all(e["matches_object"] for e in own) if own else None),
          "object_sha256_recorded_in": sorted({e["file"] for e in own}), "entries": entries}
    json.dump({"target": target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id, "source": source, "pulled_at": utc(),
               "object_sha256": sha, "staged_shas": staged_shas, "recorded_hash_check": hc, "staged_by": "scripts/sdfbase/full22/stage_objects.py"},
              open(d / "staged_from.json", "w"), indent=2, sort_keys=True)
    return hc, sha


def stage_to(host: Host, local_dir: Path, remote_dir: str, sha: str, n: int) -> dict:
    rsha, rcount = host.probe(remote_dir)
    if rsha == sha and rcount == n:
        return {"action": "already_staged", "verified": True, "remote_sha256": rsha, "remote_n_files": rcount, "bytes_moved": 0}
    if host.dry:
        print(f"DRY-RUN: rsync {' '.join(RSYNC)} {local_dir}/ {host.rsync_dest()}{remote_dir}/")
        return {"action": "would_rsync", "verified": False, "remote_sha256": rsha, "remote_n_files": rcount, "bytes_moved": 0}
    host.run(f"mkdir -p {shlex.quote(remote_dir)}")
    p = subprocess.run(RSYNC + [str(local_dir) + "/", f"{host.rsync_dest()}{remote_dir}/"], capture_output=True, text=True)
    moved = 0
    for line in p.stdout.splitlines():
        if line.startswith("Total transferred file size:"):
            moved = int(line.split(":")[1].split("bytes")[0].strip().replace(",", ""))
    rsha, rcount = host.probe(remote_dir)
    return {"action": "rsynced" if p.returncode == 0 else "rsync_failed", "verified": (rsha == sha and rcount == n),
            "remote_sha256": rsha, "remote_n_files": rcount, "bytes_moved": moved, "rsync_rc": p.returncode, "rsync_err": p.stderr[-400:] if p.returncode else ""}


def write_record_atomic(dest: Path, rec: dict) -> None:
    tmpdir = dest.parent / ".stage-tmp"; tmpdir.mkdir(parents=True, exist_ok=True)
    tmp = tmpdir / f"{dest.name}.{os.getpid()}"
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(tmp, dest)


def record_identity(rec: dict) -> Tuple[str, str]:
    """A staging record's identity on the shared manifest: (target, staged_id) — the pins key (make_full22_configs.pins_from_staging_manifest)."""
    return str(rec.get("target") or ""), str(rec.get("staged_id") or "")


class ForeignRecords(RuntimeError):
    """The remote manifest holds records the local bank lacks (another bank staged them): the push would erase them."""


class ManifestConflict(RuntimeError):
    """A remote record and a local record share an identity (target, staged_id) but not an object_sha256: two objects claim one pin."""


def manifest_conflicts(local: List[dict], remote: List[dict]) -> Tuple[List[dict], List[Tuple[dict, dict]]]:
    """``(foreign, conflicts)`` of a remote manifest against the local records: foreign = remote records whose identity the local bank
    lacks; conflicts = (local, remote) pairs of ONE identity whose object_sha256 differ (unequal, or present on one side and missing on the
    other). A pair with the same identity AND sha is the same object re-serialised: it passes, and the local record is the one published."""
    mine = {record_identity(r): r for r in local}
    foreign: List[dict] = []; conflicts: List[Tuple[dict, dict]] = []
    for r in remote:
        ident = record_identity(r)
        if ident not in mine:
            foreign.append(r); continue
        if str(mine[ident].get("object_sha256") or "") != str(r.get("object_sha256") or ""):
            conflicts.append((mine[ident], r))
    return foreign, conflicts


def conflict_notes(records: Path, foreign: List[dict]) -> Tuple[List[dict], List[Tuple[dict, dict]]]:
    """stager-fix-4 (the publish side): among the remote records this bank lacks as OBJECTS, those whose identity this bank holds a
    `remote_manifest_conflict` NOTE for are conflicts, not foreign records — the note recorded this bank's bytes (object_sha256_found) against the
    pin it met, so the publish shows both and refuses; folding them would adopt the very pin the per-cell check refused. Returns (the foreign
    records left to fold, [(this bank's bytes as a record, the remote record)])."""
    notes = {record_identity(r): r for r in local_records(records) if not r.get("object_sha256") and r.get("status") == "remote_manifest_conflict"}
    if not notes:
        return foreign, []
    left: List[dict] = []; conflicts: List[Tuple[dict, dict]] = []
    for r in foreign:
        n = notes.get(record_identity(r))
        if n is None:
            left.append(r); continue
        mine = dict(n, object_sha256=str(n.get("object_sha256_found") or ""))
        conflicts.append((mine, r))
    return left, conflicts


def conflict_text(label: str, rdir: str, conflicts: List[Tuple[dict, dict]]) -> str:
    lines = []
    for mine, theirs in conflicts:
        t, s = record_identity(theirs)
        lines.append(f"{t}/{s}: local object_sha256 {mine.get('object_sha256') or 'MISSING'} vs {label} {theirs.get('object_sha256') or 'MISSING'}\n"
                     f"      local record:  {json.dumps(mine, sort_keys=True)[:500]}\n      remote record: {json.dumps(theirs, sort_keys=True)[:500]}")
    return (f"[{label}] {rdir}/staging_manifest.json holds {len(conflicts)} record(s) whose identity this bank also holds under a DIFFERENT object "
            f"(D1311: two objects claim one pin — neither side may silently win; a human settles which object the pin is):\n    " + "\n    ".join(lines))


def foreign_text(label: str, rdir: str, foreign: List[dict]) -> str:
    names = ", ".join(f"{t}/{s}" for t, s in sorted(record_identity(r) for r in foreign))[:600]
    return (f"[{label}] {rdir}/staging_manifest.json holds {len(foreign)} record(s) this bank lacks ({names}) — a push would erase them (D905); "
            f"re-run with --merge-remote to fold them into this bank and push the union, or stage from the bank that holds them")


ABSENT = "ABSENT"


#: how long a publisher waits for a cluster's manifest WRITER LOCK (another bank mid-publish) before refusing by name
WRITER_LOCK_WAIT_S = 120
#: the per-identity lock directory under <shared>/<rel>/<target>/cells/ (stager-fix-3)
IDENTITY_LOCK_DIR = ".locks"


def remote_manifest_with_digest(h: Host, rel: str) -> Tuple[List[dict], str]:
    """``(records, digest)`` of the shared manifest as it stands on the host — ``([], ABSENT)`` when there is none; refuses by name when it
    is unreadable, never treated as empty. stager-fix-3: the file is read ONCE (base64 over one open) and the digest is taken
    over exactly the bytes that are parsed — the publish compares-and-swaps against that digest under the cluster's writer lock."""
    if h.dry:
        return [], ABSENT
    out = h.run(f'f={shlex.quote(h.shared + "/" + rel + "/staging_manifest.json")}\nif [ -f "$f" ]; then base64 -w0 "$f"; echo; else echo {ABSENT}; fi')
    line = out.strip()
    if line == ABSENT:
        return [], ABSENT
    try:
        raw = base64.b64decode(line, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"[{h.label}] the shared manifest {h.shared}/{rel}/staging_manifest.json could not be read as one byte snapshot ({exc}) — refusing to push over it")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        recs = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"[{h.label}] the shared manifest {h.shared}/{rel}/staging_manifest.json is not JSON ({exc}) — refusing to push over it")
    if not isinstance(recs, list):
        raise RuntimeError(f"[{h.label}] the shared manifest {h.shared}/{rel}/staging_manifest.json is not a list — refusing to push over it")
    return recs, digest


def remote_manifest(h: Host, rel: str) -> List[dict]:
    """The shared manifest as it stands on the host ([] when absent); refuses by name when it is unreadable — never treated as empty."""
    return remote_manifest_with_digest(h, rel)[0]


def local_records(records: Path) -> List[dict]:
    return [json.load(open(p)) for p in sorted(records.glob("*.json"))] if records.is_dir() else []


#: stager-fix-4: the record statuses that are this bank's NOTES of a refusal about a PEER / the shared manifest, never a statement about the object
REFUSAL_NOTE_STATUSES = ("identity_locked", "remote_manifest_conflict")


def object_records(records: Path) -> List[dict]:
    """stager-fix-4: the bank's records ABOUT ITS OBJECTS — every _records/*.json except a refusal NOTE this bank wrote
    (`identity_locked`: a peer held the identity; `remote_manifest_conflict`: the shared manifest pinned it elsewhere): a note carries only
    object_sha256_found and claims nothing about the object, so it is neither published nor compared — read unfiltered by the preflight, a
    note met the peer's later publish of the same identity as "MISSING vs <sha> — two objects claim one pin" and refused the very rerun the
    refusal had recommended. A `digest_mismatch` row IS a statement about the object (the pulled bytes are not the registered id) and stays:
    the pins rewrite reads it as its last door (s3r-fix) — an earlier attempt filtered it out by requiring object_sha256 and that door went
    quiet (tests/test_stage_objects_registry_id.py)."""
    return [r for r in local_records(records) if str(r.get("status") or "") not in REFUSAL_NOTE_STATUSES]


def preflight_remote_manifests(hosts: Dict[str, Host], labels: List[str], rel: str, local: List[dict], *, merge_remote: bool, publishing: bool) -> Dict[str, List[dict]]:
    """Read EVERY cluster's shared manifest BEFORE any object bytes move. A conflict (one identity, two objects) refuses by name
    on any cluster; foreign records refuse by name too when this run will publish without --merge-remote (so no object is copied for a
    manifest that would then be refused). Returns {label: remote records} for the per-cell check."""
    out: Dict[str, List[dict]] = {}; problems: List[str] = []
    for lab in labels:
        recs = remote_manifest(hosts[lab], rel)
        foreign, conflicts = manifest_conflicts(local, recs)
        if conflicts:
            problems.append(conflict_text(lab, f"{hosts[lab].shared}/{rel}", conflicts))
        elif foreign and publishing and not merge_remote:
            problems.append(foreign_text(lab, f"{hosts[lab].shared}/{rel}", foreign))
        out[lab] = recs
    if problems:
        raise (ManifestConflict if any("D1311" in p for p in problems) else ForeignRecords)("\n".join(problems))
    return out


def publish_manifest_cas(h: Host, rel: str, manifest: Path, *, expect: str) -> Optional[bool]:
    """Publish the bank's manifest to <shared>/<rel>/staging_manifest.json ONLY if the remote file's digest is still ``expect`` (ABSENT for
    none): rsync to a temp name beside it, then one remote step compares and renames (temp file + rename, so a reader never sees partial
    bytes). None when the remote moved since it was read (the caller re-reads and retries), True when the published sha verifies, False
    when it does not."""
    rdir = f"{h.shared}/{rel}"; final = f"{rdir}/staging_manifest.json"; tmp = f"{rdir}/.staging_manifest.json.{os.getpid()}.tmp"
    lock = f"{rdir}/.staging_manifest.lock"
    h.run(f"mkdir -p {shlex.quote(rdir)}")
    subprocess.run(RSYNC[:-1] + [str(manifest), f"{h.rsync_dest()}{tmp}"], check=True, capture_output=True)
    # stager-fix-3: the cluster's WRITER LOCK (flock on a file beside the manifest; every publisher takes it) spans the digest
    # compare AND the rename in ONE remote step, so two banks publishing disjoint cells cannot both pass the check and have the second rename
    # erase the first's record: the second waits, then reads a digest that moved and retries over the union. SO_PUBLISH_PAUSE_UNTIL (test
    # only: a path the step waits for between the check and the rename, read from the remote bash's environment — a `local:` host's) is how
    # the race test holds one publisher inside its critical section.
    out = h.run(f'f={shlex.quote(final)}; tmp={shlex.quote(tmp)}; expect={shlex.quote(expect)}; lock={shlex.quote(lock)}\n'
                f'exec 9>"$lock" || {{ rm -f "$tmp"; echo "LOCK-UNAVAILABLE $lock"; exit 0; }}\n'
                f'flock -w {WRITER_LOCK_WAIT_S} 9 || {{ rm -f "$tmp"; echo "LOCK-TIMEOUT $lock"; exit 0; }}\n'
                f'cur={ABSENT}; [ -f "$f" ] && cur=$(sha256sum "$f" | cut -d" " -f1)\n'
                f'if [ "$cur" != "$expect" ]; then rm -f "$tmp"; echo "CAS-MISMATCH $cur"; exit 0; fi\n'
                f'if [ -n "${{SO_PUBLISH_PAUSE_UNTIL:-}}" ]; then until [ -e "$SO_PUBLISH_PAUSE_UNTIL" ]; do sleep 0.2; done; fi\n'
                f'mv -f "$tmp" "$f" && sha256sum "$f" | cut -d" " -f1').strip()
    if out.startswith("LOCK-"):
        raise RuntimeError(f"[{h.label}] the manifest writer lock could not be taken ({out}; another publisher held it for over {WRITER_LOCK_WAIT_S} s, or the lock file is unwritable) — refusing to publish without it")
    if out.startswith("CAS-MISMATCH"):
        print(f"manifest -> {h.label}: {final} changed since it was read (now {out.split()[-1][:12]}) — not published; re-reading")
        return None
    ok = out == sha256_file(manifest)
    print(f"manifest -> {h.label}: {final} {'sha256 verified' if ok else 'sha MISMATCH'}")
    return ok


def identity_lock_path(h: Host, rel: str, target: str, staged_id: str) -> str:
    return f"{h.shared}/{rel}/{target}/cells/{IDENTITY_LOCK_DIR}/{staged_id}.lock"


def acquire_identity_lock(h: Host, rel: str, target: str, staged_id: str, owner: dict) -> Tuple[str, Optional[str]]:
    """stager-fix-3: take the per-identity lock on ``h`` — a directory created atomically by mkdir (atomic on NFS and MooseFS;
    chmod-free) with an owner.json naming this stager. Returns ``(lock path, None)`` when taken, ``(lock path, holder text)`` when a peer
    holds it: the caller refuses by name and copies nothing (fail closed; a stale lock is a human's call — the owner is printed).

    What the locks rely on (stager-fix-4): this identity lock is mkdir's create-or-fail atomicity on the cluster filesystem (an NFS
    create, a MooseFS create at its master); the manifest writer lock (publish, `flock -w`) relies on the filesystem honouring flock — it did on
    cluster A's NFS and cluster B's MooseFS when tried on 2026-10-02, but MooseFS's flock guarantees are not established from this repo, so the
    writer lock holds exactly as far as flock does there. Stale-lock policy: every exit after acquisition releases the locks (try/finally in
    main — a transient remote failure no longer leaves them), so a lock that outlives its run belongs to a run that was killed outright
    (SIGKILL, a node death). It is NOT broken automatically. owner.json names the owner's host, pid, start time (utc), bank and target; the
    operator confirms that owner is gone (the pid on that host, or the Slurm job's state), then removes the lock directory by hand and notes
    it beside the run. An operator flag (`--break-stale-locks`, judging owner.json's age and the owner's liveness and writing a note) is the
    follow-up this fix does not include."""
    d = identity_lock_path(h, rel, target, staged_id)
    out = h.run(f'd={shlex.quote(d)}; mkdir -p "$(dirname "$d")"\nif mkdir "$d" 2>/dev/null; then printf %s {shlex.quote(json.dumps(owner, sort_keys=True))} > "$d/owner.json"; echo TAKEN; '
                f'else echo HELD; cat "$d/owner.json" 2>/dev/null || echo "{{}}"; fi')
    first, _, rest = out.strip().partition("\n")
    if first == "TAKEN":
        return d, None
    return d, (rest.strip() or "unknown owner")


def release_identity_locks(held: List[Tuple[Host, str]]) -> None:
    for h, d in held:
        try:
            h.run(f'd={shlex.quote(d)}; rm -f "$d/owner.json"; rmdir "$d" 2>/dev/null || true', attempts=1)
        except RuntimeError as exc:
            print(f"  [{h.label}] WARNING: identity lock {d} was not released ({str(exc)[-160:]}); remove it by hand")
    held.clear()


def live_identity_conflicts(hosts: Dict[str, Host], labels: List[str], rel: str, target: str, staged_id: str, sha: str) -> List[Tuple[str, dict]]:
    """The clusters' shared manifests read NOW (under the identity lock): every record pinning (target, staged_id) to another object."""
    marker = os.environ.get("SO_TEST_FAIL_LIVE_CHECK_ONCE")   # test only: a transient remote failure AFTER the identity locks were taken (stager-fix-4 (2))
    if marker and not os.path.exists(marker):
        open(marker, "w").close()
        raise RuntimeError(f"[{labels[0] if labels else '?'}] remote failed: simulated transient ssh failure after the identity locks were acquired")
    out: List[Tuple[str, dict]] = []
    for lab in labels:
        for r in remote_manifest(hosts[lab], rel):
            if record_identity(r) == (target, staged_id) and str(r.get("object_sha256") or "") != sha:
                out.append((lab, r))
    return out


def publish_union(hosts: Dict[str, Host], labels: List[str], rel: str, manifest: Path, records: Path, *, merge_remote: bool = False, attempts: int = 5) -> List[dict]:
    """Read EVERY cluster's shared manifest first; a conflict refuses by name; what a cluster holds and this bank lacks is
    folded into the bank's _records/ under --merge-remote (else refused by name); ONE union is rebuilt and that same union is published
    to every cluster by compare-and-swap on each remote file's digest — a cluster that moved since the read is re-read, re-folded, and
    every cluster receives the new union again (so two banks staging disjoint cells both end with both sets). Returns the published records."""
    for attempt in range(1, attempts + 1):
        local = rebuild_manifest(records, manifest); seen: Dict[str, str] = {}
        for lab in labels:
            recs, digest = remote_manifest_with_digest(hosts[lab], rel)
            foreign, conflicts = manifest_conflicts(local, recs)
            foreign, noted = conflict_notes(records, foreign); conflicts = conflicts + noted      # stager-fix-4: a refused identity is not folded
            rdir = f"{hosts[lab].shared}/{rel}"
            if conflicts:
                raise ManifestConflict(conflict_text(lab, rdir, conflicts))
            if foreign:
                if not merge_remote:
                    raise ForeignRecords(foreign_text(lab, rdir, foreign))
                records.mkdir(parents=True, exist_ok=True)
                for r in foreign:
                    t, s = record_identity(r)
                    write_record_atomic(records / f"{t}__{s}.json", r)
                local = rebuild_manifest(records, manifest)
                print(f"manifest <- {lab}: folded {len(foreign)} foreign record(s) into {records} (union {len(local)} records)")
            seen[lab] = digest
        moved = []
        for lab in labels:
            ok = publish_manifest_cas(hosts[lab], rel, manifest, expect=seen[lab])
            if ok is None:
                moved.append(lab)
            elif not ok:
                raise RuntimeError(f"[{lab}] the published manifest's sha does not match the bank's — refusing to call it published")
        if not moved:
            return local
        print(f"manifest: {', '.join(moved)} changed under this publish (attempt {attempt} of {attempts}) — re-reading every cluster and publishing the union again")
    raise RuntimeError(f"the shared manifest kept changing under {attempts} publish attempts — refusing to publish over a moving file")


def push_manifest(h: Host, rel: str, manifest: Path, *, records: Optional[Path] = None, merge_remote: bool = False) -> bool:
    """One-cluster publication of the bank's manifest (publish_union over a single host): the remote is read first — a conflict (one
    identity, two objects) refuses by name, records this bank lacks refuse by name unless ``merge_remote`` folds them into
    ``records`` — and the union is published by compare-and-swap; True when the published sha verifies."""
    if records is None:
        raise RuntimeError(f"[{h.label}] the publication needs the bank's _records/ directory (the manifest is rebuilt from it)")
    publish_union({h.label: h}, [h.label], rel, manifest, records, merge_remote=merge_remote)
    return True


def rebuild_manifest(records: Path, manifest: Path) -> List[dict]:
    lane_order = {"base": 0, "itdoc": 1, "si12": 2, "derived": 3, "candsi": 4, "arditi": 5, "dose": 6}
    # stager-fix-3 / stager-fix-4: a refusal note (`identity_locked`, `remote_manifest_conflict`) is this bank's log of a refusal — it claims nothing
    # about the object and carries no object_sha256, so it stays in _records/ and never enters the shared union (published without a sha it
    # would read as "one identity, two objects" against the peer's own publish and refuse it); object_records is the one filter, shared with the preflight
    recs = object_records(records)
    recs.sort(key=lambda r: (r.get("target", ""), lane_order.get(r.get("lane", ""), 9), r.get("cell_id", "")))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_suffix(".json.tmp"); json.dump(recs, open(tmp, "w"), indent=2); os.replace(tmp, manifest)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site-dir", default=None); ap.add_argument("--bank", type=Path, required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--cells", type=Path, required=True, help="JSON list of {cell_id, lane, source, kind}")
    ap.add_argument("--clusters", default="", help="labels (default: every configured cluster)"); ap.add_argument("--rel", default=REL_DEFAULT)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--merge-remote", action="store_true", help="D905: fold records the clusters' shared manifest holds and this bank lacks into the bank, and push the UNION (default: refuse by name)")
    a = ap.parse_args(argv)
    site_dir = a.site_dir or CL.default_site_dir()
    # every block a label can name (a --clusters entry, a pins source's cluster), listed in SISTEER_CLUSTERS or not; the default
    # destinations are the clusters the site fans out to
    blocks = CL.load_addressable(site_dir)
    fan_out = list(CL.load_all(site_dir))
    try:   # a cluster named by its site label or its registry label (CLUSTER_SUBSTRATE); one the site config does not describe refuses
        labels = list(dict.fromkeys(CL.resolve_label(blocks, c) for c in a.clusters.split(",") if c)) or fan_out
    except SystemExit as exc:
        raise SystemExit(f"REFUSED: {str(exc.code).removeprefix('REFUSED: ')} (site dir {site_dir})") from None
    source_hosts = hosts_by_label(blocks, a.dry_run)
    hosts = {lab: source_hosts[lab] for lab in labels}
    cells = json.loads(a.cells.read_text(encoding="utf-8"))
    records = a.bank / "_records"
    if not a.dry_run:                      # a dry run computes and prints; it creates nothing
        records.mkdir(parents=True, exist_ok=True)
    rc = 0
    remote0: Dict[str, List[dict]] = {}
    if not a.dry_run:
        # Every cluster's shared manifest is read BEFORE any object bytes move — a record there that pins an identity this bank
        # holds to a different object refuses the run by name; so does a manifest this run would publish over without --merge-remote
        try:
            remote0 = preflight_remote_manifests(hosts, labels, a.rel, object_records(records), merge_remote=a.merge_remote, publishing=not a.no_manifest)   # stager-fix-4: object records only
        except (ManifestConflict, ForeignRecords, RuntimeError) as exc:
            print(f"REFUSED before any copy: {exc}"); return 2
        hook = os.environ.get("SO_TEST_AFTER_PREFLIGHT_HOOK")     # test only: a peer acting between the preflight and the per-cell live check
        if hook:
            subprocess.run(["bash", hook], check=True)
    held: List[Tuple[Host, str]] = []                             # stager-fix-3: every identity lock this run holds, released after the publish
    # stager-fix-4: from the first identity lock to the publish, every exit — a handled refusal, a return, a transient remote
    # failure raising out of the live read or the transfer — releases the locks this run holds; before this a Host.run failure after acquisition
    # left the DIRECTORY locks behind and a rerun refused identity_locked permanently
    try:
        owner = {"host": socket.gethostname(), "pid": os.getpid(), "utc": utc(), "bank": str(a.bank), "target": a.target}
        for cell in cells:
            cell_id, lane, source, kind = cell["cell_id"], cell.get("lane", "base"), cell["source"], cell.get("kind", "fit")
            staged_id = f"{lane}__{cell_id}"; key = f"{a.target}__{staged_id}"
            local_dir = a.bank / a.target / "cells" / staged_id
            print(f"--- {key} <- {source}")
            try:
                pull_source(source, local_dir, source_hosts, a.dry_run)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                print(f"  PULL FAILED: {str(exc)[-300:]}"); rc = 2
                if not a.dry_run:
                    write_record_atomic(records / f"{key}.json", {"target": a.target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id, "source_path": source, "status": "pull_failed"})
                continue
            if a.dry_run:
                for lab in labels:
                    print(f"DRY-RUN: probe + rsync {key} -> {lab}:{hosts[lab].shared}/{a.rel}/{a.target}/cells/{staged_id}")
                continue
            if OC.is_chunked(str(local_dir)) and not (local_dir / "object.pt").is_file():
                OC.reassemble(str(local_dir), str(local_dir / "object.pt"))
            why = readiness_problem(local_dir, cell_id, kind)
            if why:
                print(f"  NOT READY ({why})"); rc = 2
                write_record_atomic(records / f"{key}.json", {"target": a.target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id, "source_path": source,
                                                              "status": "not_ready", "reason": why})
                continue
            expected = cell.get("object_sha256_expected")
            if expected:
                # s3r-fix: the REGISTRY id decides, BEFORE any copy — the pulled object.pt must be the pinned bytes
                got = sha256_file(local_dir / "object.pt")
                if got != str(expected):
                    print(f"  REFUSED: {key}: the pulled object.pt digest {got[:12]} != the registered id {str(expected)[:12]} "
                          f"({cell.get('object_sha256_expected_source') or 'object_sha256_expected'}) — a stale copy under {source}; not staged, not pinned")
                    rc = 2
                    write_record_atomic(records / f"{key}.json", {"target": a.target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id, "source_path": source,
                                                                  "status": "digest_mismatch", "object_sha256_found": got, "object_sha256_expected": str(expected),
                                                                  "object_sha256_expected_source": cell.get("object_sha256_expected_source")})
                    continue
            hc, sha = hashcheck_and_stamp(local_dir, source, a.target, lane, cell_id, staged_id)
            n = sum(1 for p in local_dir.rglob("*") if p.is_file()); nbytes = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
            rec = {"target": a.target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id, "source_path": source, "object_sha256": sha,
                   "n_files": n, "n_files_source": n - 1, "bytes": nbytes, "pulled_at": utc(), "local_path": str(local_dir), "local_copy": "kept",
                   "recorded_hash_check": hc, "object_sha256_recorded_match": hc.get("object_sha256_recorded_match"),
                   **({"object_sha256_expected": str(expected), "object_sha256_expected_match": True,
                       "object_sha256_expected_source": cell.get("object_sha256_expected_source")} if expected else {})}
            # stager-fix-3: the IDENTITY LOCK on every cluster (labels order, the same on both) BEFORE the live re-check of the shared
            # manifests, held through the transfer and the publication — a peer mid-stage on this identity refuses by name, nothing copied
            mine: List[Tuple[Host, str]] = []; holder = None
            for lab in labels:
                d, who = acquire_identity_lock(hosts[lab], a.rel, a.target, staged_id, owner)
                if who is not None:
                    holder = (lab, d, who); break
                mine.append((hosts[lab], d))
            if holder is not None:
                lab_, d_, who_ = holder
                print(f"  REFUSED: {key}: this identity is being staged by another bank on {lab_} (identity lock {d_}, owner {who_[:200]}) — nothing copied, not staged, not pinned (stager-fix-3); re-run when it finishes, or remove a stale lock by hand")
                rc = 2
                release_identity_locks(mine)
                write_record_atomic(records / f"{key}.json", {"target": a.target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id, "source_path": source,
                                                              "status": "identity_locked", "object_sha256_found": sha, "identity_lock": {"cluster": lab_, "path": d_, "owner": who_[:400]}})
                continue
            held.extend(mine)
            clash = live_identity_conflicts(hosts, labels, a.rel, a.target, staged_id, sha)     # the manifests as they stand NOW, under the lock — never remote0
            if clash:
                # Per cell: the shared manifest already pins this identity to ANOTHER object — nothing is copied over it
                theirs = ", ".join(f"{lab} {str(r.get('object_sha256') or 'MISSING')[:12]}" for lab, r in clash)
                print(f"  REFUSED: {key}: the shared manifest on {', '.join(lab for lab, _ in clash)} pins this identity to another object ({theirs}) while these "
                      f"bytes are {sha[:12]} — two objects claim one pin (D1311); nothing copied, not staged, not pinned")
                rc = 2
                write_record_atomic(records / f"{key}.json", {"target": a.target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id, "source_path": source,
                                                              "status": "remote_manifest_conflict", "object_sha256_found": sha,
                                                              "remote_conflicts": {lab: r for lab, r in clash}})
                continue
            allok = True
            for lab in labels:
                rdir = f"{hosts[lab].shared}/{a.rel}/{a.target}/cells/{staged_id}"
                r = stage_to(hosts[lab], local_dir, rdir, sha, n)
                rk = record_key(blocks, lab)
                rec[f"{rk}_path"] = rdir
                for k in ("verified", "action", "remote_sha256", "remote_n_files", "bytes_moved"):
                    rec[f"{rk}_{k}"] = r[k]
                allok &= bool(r["verified"])
                print(f"  [{lab}] {r['action']} verified={r['verified']} remote_sha={str(r['remote_sha256'])[:12]} n={r['remote_n_files']}" + (f" ERR {r.get('rsync_err')}" if r.get("rsync_rc") else ""))
            rec["status"] = "staged" if allok else "verify_failed"
            write_record_atomic(records / f"{key}.json", rec)
            if not allok:
                rc = 2
        if not a.dry_run and not a.no_manifest:
            manifest = a.bank / "staging_manifest.json"
            try:
                recs = publish_union(hosts, labels, a.rel, manifest, records, merge_remote=a.merge_remote)    # one union, every cluster, by CAS under the writer lock
            except (ManifestConflict, ForeignRecords, RuntimeError) as exc:
                print(f"MANIFEST REFUSED: {exc}"); return 2                       # the finally below releases the identity locks
            print(f"MANIFEST {manifest}: {len(recs)} records, {sum(1 for r in recs if r.get('status') == 'staged')} staged, published identically to {', '.join(labels)}")
    finally:
        release_identity_locks(held)                                   # the identity locks outlive the publication, never the run
    return rc


if __name__ == "__main__":
    sys.exit(main())
