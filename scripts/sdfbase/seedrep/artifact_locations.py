#!/usr/bin/env python3
"""artifact_locations — WHERE each fitted cell of the seed-replicate run
lives, as IMMUTABLE per-record files; the aggregate is derived on read,
never stored.

    RS/reseed/artifact_locations/<key>/<cell>/<cluster>.json

One file per (key, cell, cluster). Why per-record and not one keyed-by-cell
JSON: the same four cell ids recur under three seed keys, and two writers
overlap in time — the fit watcher's DONE handler (per cell, per cluster) and
``colocate_cells.sh`` (the mirror's own record) — so a shared document would
be a read-modify-write race with a silent last-writer-wins. A per-record file
written by ``tempfile`` + ``os.replace`` is atomic, and a record is
IMMUTABLE: rewriting one with DIFFERENT content REFUSES (exit 3), identical
content is a no-op, so nothing can quietly relabel where a published object
is. ``view`` walks the tree and derives the aggregate every time it is asked.

Records carry: key, cell, cluster, bank_dir (the cell dir on that cluster),
object_sha256, fit_fingerprint, copied_from (None for the fit's own
publication; the source cluster for a colocated mirror), verified_at /
written_at — and, when ``--kind`` is given, ``kind`` (``fit`` | ``derived``,
SCHEMAS.md §2; the stager REQUIRES it). Comparison for "identical" ignores
the two timestamps. ``kind`` is compared only when the writer names one: an
existing record whose ``kind`` DIFFERS refuses like any other content
mismatch; one that carries no ``kind`` at all gains it (state
``kind_added`` — the label was missing, not different; everything else in the
file is kept byte-for-byte); a write WITHOUT ``--kind`` never reads or writes
``kind``, so a caller frozen before the flag existed behaves exactly as it
did. ``view`` WARNS on stderr about records lacking ``kind`` — it never
refuses them.

A DERIVED record (SCHEMAS.md §2) carries ``fit_fingerprint: null``: with
``--kind derived`` the fingerprint may be omitted (or given as ``-``) and is
written as null. On every other record — ``--kind fit`` or no kind — the 64-hex
fingerprint stays REQUIRED, on write and on read: a null fingerprint on a fit
or kind-less record is malformed and refused, so a record's fingerprint can go
missing only where the schema says it is absent.

    artifact_locations.py write --key K --cell C --cluster cluster_a \\
        --bank-dir $SHARED/exp/objects/sdfbase/reseed/K/cells_all/C \\
        --object-sha256 <64hex> --fit-fingerprint <64hex> [--kind fit] \\
        [--copied-from cluster_b]
    artifact_locations.py write --key K --cell <derived id> --cluster cluster_a \\
        --bank-dir $SHARED/exp/objects/sdfbase/reseed/derived/<derivation>/K/<derived id> \\
        --object-sha256 <64hex> --kind derived            # fit_fingerprint: null
    artifact_locations.py view [--key K] [--cell C] [--cluster X] [--json]

RS root: ``--rs``, else ``$RS``, else the wave lanes' shared root, which
is built into the script (``RS_DEFAULT``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

#: the run-state root under the site's state root ($STATE); expanded through the site config (scripts/site_paths.py) when used
RS_DEFAULT = "$STATE/romp/candidate-si-refit/run-state/full22"
SUBDIR = ("reseed", "artifact_locations")
CLUSTERS = ("cluster_a", "cluster_b")
#: The fields two records must agree on to count as the same record.
CONTENT_FIELDS = ("key", "cell", "cluster", "bank_dir", "object_sha256",
                  "fit_fingerprint", "copied_from")
#: SCHEMAS.md §2 ``kind``. Deliberately NOT in CONTENT_FIELDS: a write that
#: names no kind must compare exactly as it did before the flag existed.
KINDS = ("fit", "derived")
#: CLI spelling of a null fit_fingerprint (a derived record's, SCHEMAS.md §2).
FP_NULL = "-"

__all__ = ["LocationError", "rs_root", "locations_root", "record_path",
           "write_record", "read_record", "iter_records", "view", "main"]


class LocationError(RuntimeError):
    """A refusal (conflicting rewrite, malformed record); the CLI exits 3."""


def rs_root(rs=None) -> Path:
    if rs:
        return Path(rs).expanduser()
    env = os.environ.get("RS")
    if env:
        return Path(env).expanduser()
    scripts = str(Path(__file__).resolve().parents[2])
    if scripts not in sys.path:
        sys.path.append(scripts)
    from site_paths import site_path
    return site_path(RS_DEFAULT, "RS")


def locations_root(rs=None) -> Path:
    return rs_root(rs).joinpath(*SUBDIR)


def _safe_component(name: str, what: str) -> str:
    if not name or "/" in name or name in (".", "..") or name.startswith("."):
        raise LocationError(f"{what} {name!r} is not a valid path component")
    return name


def record_path(key: str, cell: str, cluster: str, rs=None) -> Path:
    return (locations_root(rs) / _safe_component(key, "key")
            / _safe_component(cell, "cell")
            / (_safe_component(cluster, "cluster") + ".json"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _content(rec: dict) -> dict:
    return {k: rec.get(k) for k in CONTENT_FIELDS}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_record(path: Path) -> dict:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        raise LocationError(f"{path} unreadable: {e}")
    if not isinstance(rec, dict):
        raise LocationError(f"{path} is not a JSON object")
    missing = [k for k in ("key", "cell", "cluster", "bank_dir",
                           "object_sha256") if not rec.get(k)]
    if missing:
        raise LocationError(f"{path} lacks {missing}")
    if "kind" in rec and rec["kind"] not in KINDS:
        raise LocationError(f"{path} has kind {rec['kind']!r}, not one of {KINDS}")
    if not rec.get("fit_fingerprint") and rec.get("kind") != "derived":
        raise LocationError(f"{path} lacks ['fit_fingerprint'] — a null fingerprint "
                            "is allowed only on a kind=derived record (SCHEMAS.md §2)")
    return rec


def write_record(*, key: str, cell: str, cluster: str, bank_dir: str,
                 object_sha256: str, fit_fingerprint: str | None,
                 copied_from: str | None = None, verified_at: str | None = None,
                 kind: str | None = None, rs=None) -> tuple[Path, str]:
    """Write the record; returns (path, ``"written" | "unchanged" |
    "kind_added"``). Refuses (LocationError) when a record with DIFFERENT
    content already sits at that path — a record is immutable.

    ``kind`` (``fit`` | ``derived``, SCHEMAS.md §2) is optional and compared
    only when given: an existing record with a DIFFERENT kind refuses; one
    with NO kind gains it (``kind_added``); ``kind=None`` neither reads nor
    writes the field, so the pre-flag callers see the pre-flag behaviour.

    ``fit_fingerprint`` is 64 hex, except that a ``kind="derived"`` record may
    give ``None`` (CLI: omitted, or ``-``) and is written with null — the
    schema's value for a derived object. Null on any other record refuses."""
    if cluster not in CLUSTERS:
        raise LocationError(f"cluster {cluster!r} is not one of {CLUSTERS}")
    if kind is not None and kind not in KINDS:
        raise LocationError(f"kind {kind!r} is not one of {KINDS}")
    if copied_from is not None and copied_from not in CLUSTERS:
        raise LocationError(f"copied_from {copied_from!r} is not one of {CLUSTERS}")
    if copied_from == cluster:
        raise LocationError(f"copied_from must name ANOTHER cluster, not {cluster}")
    if fit_fingerprint in (None, "", FP_NULL):
        if kind != "derived":
            raise LocationError(
                "fit_fingerprint must be 64 hex chars on a "
                f"{kind or 'kind-less'} record — null is the value of a --kind "
                "derived record only (SCHEMAS.md §2)")
        fit_fingerprint = None
    for name, val in (("object_sha256", object_sha256),
                      ("fit_fingerprint", fit_fingerprint)):
        if val is None:
            continue
        if not isinstance(val, str) or len(val) != 64 or \
                any(c not in "0123456789abcdef" for c in val.lower()):
            raise LocationError(f"{name} must be 64 hex chars, got {val!r}")
    rec = {"key": key, "cell": cell, "cluster": cluster,
           "bank_dir": str(bank_dir), "object_sha256": object_sha256.lower(),
           "fit_fingerprint": fit_fingerprint.lower() if fit_fingerprint else None,
           "copied_from": copied_from,
           "verified_at": verified_at or _now(), "written_at": _now()}
    if kind is not None:
        rec["kind"] = kind
    path = record_path(key, cell, cluster, rs)
    if path.exists():
        old = read_record(path)
        same = _content(old) == _content(rec)
        kind_differs = kind is not None and "kind" in old and old["kind"] != kind
        if same and not kind_differs:
            if kind is None or old.get("kind") == kind:
                return path, "unchanged"
            # same location/identity, and the existing record carries no kind:
            # the label was MISSING, not different — add it and keep every other
            # byte (timestamps included), the way the DONE handler's patch did
            _atomic_write(path, json.dumps({**old, "kind": kind}, indent=1,
                                           sort_keys=True) + "\n")
            return path, "kind_added"
        diff = {k: (old.get(k), rec.get(k)) for k in CONTENT_FIELDS
                if old.get(k) != rec.get(k)}
        if kind_differs:
            diff["kind"] = (old["kind"], kind)
        raise LocationError(
            f"{path} already records a DIFFERENT location/identity for "
            f"({key}, {cell}, {cluster}) — records are immutable; differing "
            f"fields (existing, new): {diff}. A moved or re-published object "
            "is a human's call: remove the record deliberately, never overwrite")
    _atomic_write(path, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return path, "written"


def iter_records(rs=None, *, key: str | None = None, cell: str | None = None,
                 cluster: str | None = None):
    root = locations_root(rs)
    if not root.is_dir():
        return
    for kdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if key and kdir.name != key:
            continue
        for cdir in sorted(p for p in kdir.iterdir() if p.is_dir()):
            if cell and cdir.name != cell:
                continue
            for f in sorted(cdir.glob("*.json")):
                if f.name.startswith("."):
                    continue
                if cluster and f.stem != cluster:
                    continue
                rec = read_record(f)
                if (rec["key"], rec["cell"], rec["cluster"]) != \
                        (kdir.name, cdir.name, f.stem):
                    raise LocationError(
                        f"{f} records ({rec['key']}, {rec['cell']}, "
                        f"{rec['cluster']}) but sits at "
                        f"({kdir.name}, {cdir.name}, {f.stem})")
                yield rec


def view(rs=None, *, key=None, cell=None, cluster=None) -> dict:
    """The aggregate, DERIVED: every record, plus per (key, cell) the
    clusters holding it, the object sha they agree on (``consistent`` False
    when two clusters' records disagree — a mirror that does not match its
    source), and the fit's own publication cluster."""
    recs = list(iter_records(rs, key=key, cell=cell, cluster=cluster))
    by_cell: dict[str, dict] = {}
    #: per cell, every record's value of each identity field — a derived
    #: record's null fingerprint is a VALUE here, so two records that disagree
    #: (null vs a hash) read inconsistent whichever cluster sorts first
    seen: dict[str, dict[str, list]] = {}
    missing_kind: list[str] = []
    warnings: list[str] = []
    for r in recs:
        k = f"{r['key']}/{r['cell']}"
        agg = by_cell.setdefault(k, {"key": r["key"], "cell": r["cell"],
                                     "clusters": [], "origin": None,
                                     "kind": None, "object_sha256": None,
                                     "fit_fingerprint": None,
                                     "consistent": True, "bank_dir": {}})
        sv = seen.setdefault(k, {"object_sha256": [], "fit_fingerprint": []})
        agg["clusters"].append(r["cluster"])
        agg["bank_dir"][r["cluster"]] = r["bank_dir"]
        if r.get("copied_from") is None:
            agg["origin"] = r["cluster"]
        for f in ("object_sha256", "fit_fingerprint"):
            sv[f].append(r.get(f))
        if "kind" not in r:
            rel = f"{r['key']}/{r['cell']}/{r['cluster']}.json"
            missing_kind.append(rel)
            warnings.append(f"{rel} lacks kind (SCHEMAS.md §2 — the stager "
                            "refuses it); re-issue its write with --kind "
                            "fit|derived to add it")
        elif agg["kind"] is None:
            agg["kind"] = r["kind"]
        elif agg["kind"] != r["kind"]:
            agg["consistent"] = False
            warnings.append(f"{k}: records disagree on kind ({agg['kind']!r} "
                            f"vs {r['kind']!r} on {r['cluster']})")
    for k, agg in by_cell.items():
        agg["clusters"].sort()
        for f, vals in seen[k].items():
            agg[f] = vals[0]                     # the value they agree on (or the first)
            if len(set(vals)) > 1:
                agg["consistent"] = False
    return {"rs": str(rs_root(rs)), "n_records": len(recs), "records": recs,
            "by_cell": by_cell, "missing_kind": missing_kind,
            "warnings": warnings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rs", default=None, help="RS root (default $RS or "
                                                f"{RS_DEFAULT})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write one immutable record")
    w.add_argument("--key", required=True)
    w.add_argument("--cell", required=True)
    w.add_argument("--cluster", required=True, choices=CLUSTERS)
    w.add_argument("--bank-dir", required=True)
    w.add_argument("--object-sha256", required=True)
    w.add_argument("--fit-fingerprint", default=None,
                   help="64 hex (COMPLETE.fit_fingerprint); omit or pass '-' "
                        "ONLY with --kind derived, whose record carries null")
    w.add_argument("--copied-from", default=None, choices=CLUSTERS)
    w.add_argument("--verified-at", default=None)
    w.add_argument("--kind", default=None, choices=KINDS,
                   help="SCHEMAS.md §2 kind (the stager requires it); omitted "
                        "= the record is written and compared without one")
    v = sub.add_parser("view", help="the derived aggregate")
    v.add_argument("--key", default=None)
    v.add_argument("--cell", default=None)
    v.add_argument("--cluster", default=None, choices=CLUSTERS)
    v.add_argument("--json", action="store_true", help="JSON (default: table)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "write":
            path, state = write_record(
                key=a.key, cell=a.cell, cluster=a.cluster, bank_dir=a.bank_dir,
                object_sha256=a.object_sha256, fit_fingerprint=a.fit_fingerprint,
                copied_from=a.copied_from, verified_at=a.verified_at,
                kind=a.kind, rs=a.rs)
            print(f"[artifact-locations] {state}: {path}")
        else:
            doc = view(a.rs, key=a.key, cell=a.cell, cluster=a.cluster)
            if a.json:
                print(json.dumps(doc, indent=1, sort_keys=True))
            else:
                for k, agg in sorted(doc["by_cell"].items()):
                    flag = "" if agg["consistent"] else "  INCONSISTENT"
                    print(f"{k}  clusters={','.join(agg['clusters'])}  "
                          f"origin={agg['origin']}  kind={agg['kind'] or '-'}  "
                          f"sha={str(agg['object_sha256'])[:12]}{flag}")
                print(f"[artifact-locations] {doc['n_records']} record(s) "
                      f"under {doc['rs']}")
            # warnings go to stderr in BOTH modes so --json stdout stays parseable
            for msg in doc["warnings"]:
                print(f"[artifact-locations] WARNING: {msg}", file=sys.stderr)
    except LocationError as e:
        print(f"[artifact-locations] REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
