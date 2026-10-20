#!/usr/bin/env python3
"""reproduce/tools/export_objects.py — commit the paper's fitted objects under objects/ as their ORIGINAL bytes,
keyed by the registry `object_id` (== sha256(object.pt) for every artifact-bearing kind).

    python reproduce/tools/export_objects.py --select paper --registry ~/registry/registry.sqlite \
        --bank NAME=DIR ... --out objects [--census JSON] [--chunk-mb 40] [--withheld objects/WITHHELD.json]

Selection: `v_paper_objects` where in_paper = 1, joined to `object` (artifact_sha256, kind, label, role, fit_model_id).
The artifact-bearing rows (lora, lora_subtraction, composed, relu_steer, vector_all, static_r_arditi; 151 in the paper's
selection, 149 committed after the withheld list below) are resolved
by sha256 against every `object.pt` found under the named banks (a registry `artifact_locations_json` uri is tried
first), copied UNCHANGED to objects/<fit_model_id>/<object_id>/object.pt — or, when larger than --chunk-mb, packaged
losslessly through reproduce/lib/object_chunks.py (object.pt.part-NN + CHUNKS.json; the sha of the reassembled bytes is
the object_id) — together with the reader-named sidecars (sisteer.registry.hooks.SIDECAR_NAMES + _store_identity.json,
one level, never recursive) AND every file the cell's JSON `COMPLETE` publication marker attests (`COMPLETE['files']`:
the object, train_budget_meta.json, metrics.jsonl, fit_selection.json when present), each verified against the sha256 the
marker records — an exported directory must pass the same attestation the trainer's resume validator applies
(scripts/train_two_si_objects.py), so a missing or mismatching attested file REFUSES the export. The 3 VIRTUAL rows (two `system_instruction` objects `si:<sha>` and `none`) have no
artifact and are listed in objects/MANIFEST.json with form=virtual and no directory.

`--census JSON` is the same row set pre-exported from the registry (the view is a two-minute query on the 36 GB
grading host registry); with `--registry` the query runs here. Nothing here names a host or a home directory: the MANIFEST
records each artifact's source as `<bank name>:<relative path>`.

WITHHELD objects (`--withheld`, default `<out>/WITHHELD.json` when present: `{objects: [{object_id, model_id, paper_label,
object_kind, reason, ...}]}`) are paper objects this repository does not ship: they are not copied and get no manifest row,
the census counts them (`in_paper` is the whole selection, `withheld` how many of it the list holds, `artifact` + `virtual`
the rows listed), and the manifest names the list (`withheld_list`), which sisteer.objects and the committed-object loader
read to refuse each by name with its reason. A withheld id the selection lacks, one whose model, label or kind disagrees
with its selection row, a withheld virtual row, and a directory of a withheld object left under `--out` are REFUSED.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: = sisteer.registry.hooks.SIDECAR_NAMES (everything ``read_sidecars`` consumes, ``fit_selection.json`` included) + the
#: bank's ``_store_identity.json``. Spelled here because this tool is stdlib-only; tests/test_objects_committed.py pins the
#: equality so the two lists cannot drift apart silently.
SIDECAR_NAMES = ("train_budget_meta.json", "fit_meta.json", "fit_provenance.json", "derived_meta.json", "meta.json",
                 "summary.json", "COMPLETE", "fit_selection.json", "_store_identity.json")
VIRTUAL_KINDS = ("none", "system_instruction")
#: The trainer's publication marker (scripts/train_two_si_objects.py ``publish_cell``): JSON with ``files`` = every published
#: file's sha256. The legacy plain-text form ("promoted from …") attests nothing beyond its presence.
COMPLETE_MARKER = "COMPLETE"
DEFAULT_CHUNK_MB = 40
_BUF = 1 << 20

CENSUS_SQL = """
select v.object_id, v.paper_label, v.object_kind, v.object_role, v.object_status, v.fit_model_id, v.campaign, v.in_paper,
       o.artifact_sha256, o.artifact_bytes, o.artifact_locations_json, o.object_sidecars_json, o.derivation,
       o.parent_object_ids_json, o.base_scale, o.si_text_sha256, o.fit_rank, o.fit_target_modules_json, o.status, o.record_uri
from v_paper_objects v join object o on o.object_id = v.object_id
where v.in_paper = 1 order by v.fit_model_id, v.object_kind, v.object_id
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(_BUF), b""):
            h.update(b)
    return h.hexdigest()


def load_object_chunks():
    spec = importlib.util.spec_from_file_location("object_chunks", REPO / "reproduce/lib/object_chunks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _connect_ro(db: Path) -> sqlite3.Connection:
    """The registry opened READ-ONLY (a two-minute view query on the 36 GB grading host registry wants the long timeout)."""
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=600)


def census_from_registry(db: Path) -> list[dict]:
    with closing(_connect_ro(db)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(CENSUS_SQL)]


def registry_head(db: Path) -> dict | None:
    """Provenance only (the manifest's ``source.registry_head``): None when the registry cannot answer."""
    try:
        with closing(_connect_ro(db)) as conn:
            row = conn.execute("select max(ingest_seq) from object").fetchone()
        return {"object_max_ingest_seq": row[0]} if row else None
    except sqlite3.Error:
        return None


class Banks:
    """sha256 -> [paths] over every object.pt under the named banks (hashed once)."""

    def __init__(self, banks: dict[str, Path]):
        self.banks = {n: Path(p).resolve() for n, p in banks.items()}
        self.index: dict[str, list[Path]] = {}
        for name, root in self.banks.items():
            if not root.is_dir():
                raise SystemExit(f"REFUSED: bank {name} at {root} is not a directory")
            for p in root.rglob("object.pt"):
                self.index.setdefault(sha256_file(p), []).append(p)

    def label(self, path: Path) -> str:
        path = Path(path).resolve()
        for name, root in self.banks.items():
            try:
                return f"{name}:{path.relative_to(root).as_posix()}"
            except ValueError:
                continue
        return f"external:{path.name}"

    @staticmethod
    def _score(p: Path) -> tuple:
        s = str(p)
        return ("/_withdrawn/" in s, "/_pull/" in s, "/cells_view/" in s,
                p.parent.name.startswith(("d1__", "halfdose__", "d0p5__")), len(s))

    def resolve(self, sha: str, locations_json: str | None) -> Path | None:
        for loc in json.loads(locations_json or "[]"):
            uri = str(loc.get("uri", "")).split("#")[0]
            if uri.endswith("object.pt") and os.path.isfile(uri) and sha256_file(Path(uri)) == sha:
                return Path(uri)
        cands = self.index.get(sha) or []
        return sorted(cands, key=self._score)[0] if cands else None


def package_artifact(src: Path, dest_dir: Path, chunk_bytes: int, chunks_mod) -> dict:
    """Copy `src` (an object.pt) into dest_dir unchanged, or chunk it losslessly when larger than chunk_bytes."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    if size > chunk_bytes:
        man = chunks_mod.chunk(str(src), str(dest_dir), chunk_bytes=chunk_bytes, compression="none")
        return {"form": "chunked", "chunks": len(man["chunks"]), "bytes": man["original_bytes"],
                "sha256": man["original_sha256"], "compression": man["compression"]}
    _copy_file(src, dest_dir / "object.pt")
    return {"form": "whole", "chunks": 0, "bytes": size, "sha256": sha256_file(dest_dir / "object.pt")}


def _copy_file(src: Path, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)


def copy_sidecars(src_dir: Path, dest_dir: Path) -> list[str]:
    out = []
    for name in SIDECAR_NAMES:
        p = src_dir / name
        if p.is_file():
            _copy_file(p, dest_dir / name)
            out.append(name)
    return out


def attested_files(src_dir: Path) -> dict[str, str] | None:
    """``COMPLETE['files']`` (name -> sha256) when the marker is the JSON publication manifest; None when the marker is
    absent, the legacy plain-text form, or a JSON marker from before the publication manifest (no ``files``) — those attest
    nothing beyond their presence. A marker that looks like JSON but does not parse, or whose ``files`` is not a name ->
    sha256 map, refuses: the attestation cannot be read, so it cannot be honoured."""
    p = src_dir / COMPLETE_MARKER
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.lstrip().startswith("{"):
        return None
    try:
        marker = json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"REFUSED: {p} looks like a JSON COMPLETE marker but does not parse ({e})")
    files = marker.get("files")
    if files is None:
        return None
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        raise SystemExit(f"REFUSED: {p}: COMPLETE['files'] is not a name -> sha256 map")
    return dict(files)


def copy_attested_files(src_dir: Path, dest_dir: Path, artifact_name: str, artifact_sha: str,
                        already: set[str]) -> list[str] | None:
    """Copy every file the JSON ``COMPLETE`` marker attests (those not already copied as sidecars) and verify EVERY sha256 it
    records: the artifact's against the exported artifact's sha (the reassembled original when chunked), every other
    file's against the copied bytes. Refuses on an absent file or a mismatch — a committed directory that cannot pass its
    own attestation is not published (38 exported directories carried a COMPLETE naming metrics.jsonl and none carried
    the file). Returns the attested names sorted, or None when the marker attests nothing."""
    files = attested_files(src_dir)
    if files is None:
        return None
    for name, sha in sorted(files.items()):
        if "/" in name or name in ("", ".", ".."):
            raise SystemExit(f"REFUSED: {src_dir}: COMPLETE attests a non-flat name {name!r}")
        if name == artifact_name:
            if sha != artifact_sha:
                raise SystemExit(f"REFUSED: {src_dir}: COMPLETE records {name} sha {sha[:12]} but the exported artifact "
                                 f"hashes {artifact_sha[:12]}")
            continue
        p = src_dir / name
        if not p.is_file():
            raise SystemExit(f"REFUSED: {src_dir}: COMPLETE attests {name} but the file is absent — the exported directory "
                             "could not pass its own attestation")
        if name not in already:
            _copy_file(p, dest_dir / name)
        got = sha256_file(dest_dir / name)
        if got != sha:
            raise SystemExit(f"REFUSED: {src_dir}: {name} hashes {got[:12]} != the sha {sha[:12]} its COMPLETE records")
    return sorted(files)


def read_withheld(path: Path | None) -> dict[str, dict]:
    """object_id -> row of a withheld list (empty for None / an absent file); a row lacking object_id, model_id, paper_label,
    object_kind or reason is REFUSED. Stdlib twin of sisteer.objects.read_withheld (tests hold the two readings equal)."""
    if path is None or not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("objects"), list):
        raise SystemExit(f"REFUSED: {path}: expected {{objects: [...]}}")
    out: dict[str, dict] = {}
    for w in doc["objects"]:
        lacking = [k for k in ("object_id", "model_id", "paper_label", "object_kind", "reason") if not w.get(k)]
        if lacking or w["object_id"] in out:
            raise SystemExit(f"REFUSED: {path}: a withheld row {'lacks ' + ', '.join(lacking) if lacking else 'repeats its id'}: {w!r}")
        out[w["object_id"]] = w
    return out


def check_withheld(rows: list[dict], withheld: dict[str, dict], out: Path) -> None:
    """Every withheld id is a non-virtual row of the selection with the same model, label and kind, and no directory of one is left
    under `out` (an earlier export's bytes)."""
    by_id = {r["object_id"]: r for r in rows}
    for oid, w in withheld.items():
        r = by_id.get(oid)
        if r is None:
            raise SystemExit(f"REFUSED: withheld object {oid} ({w['model_id']} {w['paper_label']}) is not in the selection — a stale entry")
        got = {"model_id": r["fit_model_id"], "paper_label": r["paper_label"], "object_kind": r["object_kind"]}
        diff = {k: (w[k], v) for k, v in got.items() if w[k] != v}
        if diff:
            raise SystemExit(f"REFUSED: withheld object {oid}: the list and the selection disagree on {diff}")
        if r["object_kind"] in VIRTUAL_KINDS:
            raise SystemExit(f"REFUSED: withheld object {oid} is virtual ({r['object_kind']}): there is nothing to withhold")
        left = out / str(r["fit_model_id"] or "unknown_model") / oid
        if left.exists():
            raise SystemExit(f"REFUSED: {left}: withheld object {oid} ({w['paper_label']}) has a directory under {out}; remove it")


def export(rows: list[dict], banks: Banks, out: Path, chunk_mb: int, *, source_note: dict,
           withheld: dict[str, dict] | None = None, withheld_list: str | None = None) -> dict:
    withheld = withheld or {}
    check_withheld(rows, withheld, out)
    chunks_mod = load_object_chunks()
    chunk_bytes = chunk_mb * 1000 * 1000
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    missing = []
    for r in rows:
        if r["object_id"] in withheld:
            continue
        kind = r["object_kind"]
        base = {"object_id": r["object_id"], "model_id": r["fit_model_id"], "object_kind": kind,
                "paper_label": r["paper_label"], "object_role": r["object_role"], "derivation": r.get("derivation"),
                "campaign": r.get("campaign")}
        if kind in VIRTUAL_KINDS:
            entries.append(base | {"form": "virtual", "artifact_sha256": None, "bytes": None, "chunks": 0, "dir": None,
                                   "si_text_sha256": r.get("si_text_sha256")})
            continue
        sha = r["artifact_sha256"]
        if r["object_id"] != sha:
            raise SystemExit(f"REFUSED: {r['object_id']} has artifact_sha256 {sha} (object_id must equal it)")
        src = banks.resolve(sha, r.get("artifact_locations_json"))
        if src is None:
            missing.append(r)
            continue
        model_key = r["fit_model_id"] or "unknown_model"
        dest_dir = out / model_key / r["object_id"]
        pk = package_artifact(src, dest_dir, chunk_bytes, chunks_mod)
        if pk["sha256"] != sha:
            raise SystemExit(f"REFUSED: {src} hashed {pk['sha256']} != {sha}")
        sidecars = copy_sidecars(src.parent, dest_dir)
        attested = copy_attested_files(src.parent, dest_dir, src.name, sha, set(sidecars))
        entries.append(base | {"form": pk["form"], "artifact_sha256": sha, "bytes": pk["bytes"], "chunks": pk["chunks"],
                               "dir": f"{model_key}/{r['object_id']}", "sidecars": sidecars, "attested": attested,
                               "source": banks.label(src)} | ({"compression": pk["compression"]} if pk["form"] == "chunked" else {}))
    if missing:
        for r in missing:
            print(f"MISSING {r['object_kind']} {r['paper_label']} {r['fit_model_id']} {r['artifact_sha256']}", file=sys.stderr)
        raise SystemExit(f"REFUSED: {len(missing)} artifact(s) not found in any bank — no object is dropped")
    n_art = sum(1 for e in entries if e["form"] != "virtual")
    n_virt = len(entries) - n_art
    manifest = {
        "schema": 1,
        "selection": "v_paper_objects.in_paper = 1 join object",
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source_note,
        # in_paper = the whole selection; withheld = how many of it the withheld list holds (no row, no directory); artifact and
        # virtual = the rows listed; by_kind / by_model count the listed rows
        "census": {"in_paper": len(entries) + len(withheld), "withheld": len(withheld), "artifact": n_art, "virtual": n_virt,
                   "by_kind": dict(sorted(Counter(e["object_kind"] for e in entries).items())),
                   "by_model": dict(sorted(Counter(str(e["model_id"]) for e in entries).items()))},
        **({"withheld_list": withheld_list} if withheld else {}),
        "sidecar_names": list(SIDECAR_NAMES),
        # per row: `sidecars` = the reader-named files present; `attested` = COMPLETE['files'] names (every sha verified at
        # export; the artifact's equals object_id), or null when the marker attests nothing (legacy text form / absent)
        "attestation": "COMPLETE['files'] verified per object at export",
        "chunk_bytes": chunk_bytes,
        "bytes_total": sum(e["bytes"] or 0 for e in entries),
        "objects": sorted(entries, key=lambda e: (str(e["model_id"]), e["object_kind"], e["object_id"])),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--select", choices=["paper"], default="paper")
    ap.add_argument("--registry", type=Path, default=None, help="registry.sqlite (read-only); else --census")
    ap.add_argument("--census", type=Path, default=None, help="pre-exported rows (JSON list) of the selection query")
    ap.add_argument("--bank", action="append", default=[], metavar="NAME=DIR", help="an object bank to search (repeatable)")
    ap.add_argument("--out", type=Path, default=REPO / "objects")
    ap.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_MB)
    ap.add_argument("--registry-head-seq", type=int, default=None,
                    help="with --census: the registry's max(object.ingest_seq) at census time, recorded as "
                         "source.registry_head exactly as --registry would read it (the registry is not opened)")
    ap.add_argument("--withheld", type=Path, default=None,
                    help="the withheld list beside the manifest (default: <out>/WITHHELD.json when present)")
    a = ap.parse_args(argv)
    wpath = a.withheld if a.withheld is not None else a.out / "WITHHELD.json"
    if a.withheld is not None and not wpath.is_file():
        ap.error(f"--withheld {wpath}: no such file")
    if wpath.is_file() and wpath.resolve().parent != a.out.resolve():
        ap.error(f"--withheld {wpath}: the list must sit beside the manifest it is named in ({a.out})")
    withheld = read_withheld(wpath)
    if a.census:
        rows = json.loads(a.census.read_text(encoding="utf-8"))
        src = {"census_file": a.census.name}
    elif a.registry:
        rows = census_from_registry(a.registry)
        src = {"registry": a.registry.name}
    else:
        ap.error("--registry DB or --census JSON is required")
    if a.registry:
        src["registry_head"] = registry_head(a.registry)
    elif a.registry_head_seq is not None:
        src["registry_head"] = {"object_max_ingest_seq": a.registry_head_seq}
    banks = {}
    for spec in a.bank:
        name, sep, path = spec.partition("=")
        if not sep:
            ap.error(f"--bank must be NAME=DIR, got {spec!r}")
        banks[name] = Path(path).expanduser()
    if not banks:
        ap.error("at least one --bank NAME=DIR is required")
    man = export(rows, Banks(banks), a.out, a.chunk_mb, source_note=src, withheld=withheld, withheld_list=wpath.name)
    print(json.dumps({"in_paper": man["census"]["in_paper"], "withheld": man["census"]["withheld"],
                      "artifact": man["census"]["artifact"], "virtual": man["census"]["virtual"],
                      "bytes_total": man["bytes_total"], "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
