#!/usr/bin/env python3
"""paper/notebooks/strength_reads_sidecar.py — the committed projection of the strength lanes' XSTest reads sidecars.

    python paper/notebooks/strength_reads_sidecar.py write --source NAME=PATH [--source NAME=PATH ...] [--out DIR]
    python paper/notebooks/strength_reads_sidecar.py check --source NAME=PATH [--source NAME=PATH ...] [--out DIR]

The strength battery record read each safe XSTest item at the first rung whose draw terminated, and a judgment's item rows hold the
top-rung store only, so under a judge other than the filed one the notebook re-derives those readings from a READS SIDECAR
(`registry_nb.strength_reads`, `judge_rules.load_strength_reads`). The full sidecars (`strength_reads.parquet` with its
`manifest.json`, built by the re-judge lane's reads tool) are too large and carry machine-local paths, so this tool commits a
PROJECTION of each, which the notebook reads by default:

* rows: the record's primary reads (`read == "per_item"`), the only rows the loader reads
* columns: run_id, item_id, read, at_cap, the two judges' label and state columns (minimax_label, minimax_outcome, sonnet_label,
  sonnet_status), and rung, top_rung and trial, in (run_id, item_id, trial) order
* manifest: the projected table's sha256 (`table_sha256`, which the loader verifies), the fields the loader vouches by (`preview`,
  `paused_run_dirs`, `unfinished_run_dirs`, `filed_check.{judge_id,counts,disagreements}`, `reads_scope.report.sonnet_judge`), the
  full table's sha256, the battery records the reads came from as repo-relative `strength/<lane>/battery_curves.json` with their
  sha256, the record code, and this script's sha256

`write` refuses when a projected cell or manifest value names a machine-local path or this host's identity (the export scrub's own
test, `sisteer.registry.snapshot._leaks`). `check` re-projects from the full sidecars and compares with the committed files (exit 1 on
a difference). Each source is written under `<out>/<NAME>/` (default out: paper/notebooks/data/strength_reads).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "data", "strength_reads")
TABLE = "strength_reads.parquet"
MANIFEST = "manifest.json"
#: the projected columns, in order (the loader reads run_id, item_id, read, at_cap and the label / state columns)
COLUMNS = ("run_id", "item_id", "read", "at_cap", "minimax_label", "minimax_outcome", "sonnet_label", "sonnet_status", "rung",
           "top_rung", "trial")
ORDER = ("run_id", "item_id", "trial")
#: a battery record path, made repo-relative from the strength directory on
_RECORD = re.compile(r"(?:^|/)(strength/[^/]+/battery_curves\.json)$")
NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class SidecarError(RuntimeError):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def project(path: str):
    """The projected table of one full sidecar, as a pyarrow Table with a fixed schema (no pandas metadata, so the bytes do not depend
    on the pandas version)."""
    import pandas as pd
    import pyarrow as pa
    t = pd.read_parquet(path)
    missing = [c for c in COLUMNS if c not in t.columns]
    if missing:
        raise SidecarError(f"{path}: the sidecar has no {missing} column(s)")
    t = t[t["read"].eq("per_item")][list(COLUMNS)].sort_values(list(ORDER), kind="mergesort").reset_index(drop=True)
    if t["at_cap"].isna().any():
        raise SidecarError(f"{path}: a per_item read has no at_cap")
    schema = pa.schema([("run_id", pa.string()), ("item_id", pa.string()), ("read", pa.string()), ("at_cap", pa.bool_()),
                        ("minimax_label", pa.string()), ("minimax_outcome", pa.string()), ("sonnet_label", pa.string()),
                        ("sonnet_status", pa.string()), ("rung", pa.int64()), ("top_rung", pa.int64()), ("trial", pa.int64())])
    cols = {}
    for f in schema:
        s = t[f.name]
        if pa.types.is_string(f.type):
            vals = [None if (v is None or (isinstance(v, float) and v != v)) else str(v) for v in s.tolist()]
        elif pa.types.is_boolean(f.type):
            vals = [bool(v) for v in s.tolist()]
        else:
            vals = [None if (v is None or (isinstance(v, float) and v != v)) else int(v) for v in s.tolist()]
        cols[f.name] = pa.array(vals, type=f.type)
    return pa.Table.from_pydict(cols, schema=schema)


def projected_manifest(full_path: str, table_sha: str, n_rows: int, n_runs: int) -> dict:
    """The committed manifest of one projection (the fields in the module docstring); refuses a full sidecar its own manifest does not
    vouch for."""
    with open(os.path.join(os.path.dirname(full_path), MANIFEST), encoding="utf-8") as fh:
        man = json.load(fh)
    full_sha = _sha256(full_path)
    if man.get("table_sha256") != full_sha:
        raise SidecarError(f"{full_path}: sha256 {full_sha[:12]} is not its manifest's table_sha256 {str(man.get('table_sha256'))[:12]}")
    report = (man.get("reads_scope") or {}).get("report") or {}
    records = {}
    for p, sha in sorted((report.get("records") or {}).items()):
        m = _RECORD.search(str(p))
        if not m:
            raise SidecarError(f"{full_path}: the battery record {p!r} is not a strength/<lane>/battery_curves.json")
        records[m.group(1)] = sha
    fc = man.get("filed_check") or {}
    return {
        "about": "The committed projection of a strength-lane XSTest reads sidecar: the record's per_item reads only, the columns the "
                 "notebook's loader reads (paper/notebooks/strength_reads_sidecar.py).",
        "table": TABLE, "table_sha256": table_sha, "rows": n_rows, "runs": n_runs, "columns": list(COLUMNS),
        "preview": bool(man.get("preview")), "paused_run_dirs": man.get("paused_run_dirs") or {},
        "unfinished_run_dirs": man.get("unfinished_run_dirs") or {},
        "filed_check": {"judge_id": fc.get("judge_id"), "counts": fc.get("counts") or {}, "disagreements": fc.get("disagreements") or []},
        "reads_scope": {"report": {"sonnet_judge": report.get("sonnet_judge")}},
        "source": {"table_sha256": full_sha, "rows": man.get("rows"), "runs": man.get("runs")},
        "battery_records": records, "battery_record_code": report.get("battery_record"),
        "generator": {"path": "paper/notebooks/strength_reads_sidecar.py", "sha256": _sha256(os.path.abspath(__file__))},
    }


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def leaks(table, manifest: dict) -> list:
    """Every projected string cell and manifest string that names a machine-local path or this host's identity (the export scrub's
    `_leaks`, with its private tokens)."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from sisteer.registry import snapshot as SN
    toks = tuple(t.lower() for t in SN._private_tokens())
    out = [s for s in _strings(manifest) if SN._leaks(s, toks)]
    for name in table.column_names:
        if table.schema.field(name).type == "string":
            out += [v for v in set(table.column(name).to_pylist()) if isinstance(v, str) and SN._leaks(v, toks)]
    return sorted(set(out))


def _write_table(table, path: str) -> None:
    import pyarrow.parquet as pq
    pq.write_table(table, path, compression="zstd", use_dictionary=True, write_statistics=False)


def build(name: str, full_path: str, out: str, write: bool) -> list:
    """Project one source; write it (``write``) or compare it with the committed files. Returns the differences (empty when equal or
    written)."""
    import pyarrow.parquet as pq
    if not NAME.match(name):
        raise SidecarError(f"source name {name!r}: lower-case letters, digits and underscores")
    table = project(full_path)
    d = os.path.join(out, name)
    runs = len(set(table.column("run_id").to_pylist()))
    if write:
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, TABLE + ".tmp")
        _write_table(table, tmp)
        man = projected_manifest(full_path, _sha256(tmp), table.num_rows, runs)
        bad = leaks(table, man)
        if bad:
            os.remove(tmp)
            raise SidecarError(f"{name}: the projection names machine-local paths or this host: {bad[:5]}")
        os.replace(tmp, os.path.join(d, TABLE))
        with open(os.path.join(d, MANIFEST), "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
            fh.write("\n")
        return []
    diffs = []
    tp, mp = os.path.join(d, TABLE), os.path.join(d, MANIFEST)
    if not (os.path.exists(tp) and os.path.exists(mp)):
        return [f"{name}: {d} holds no committed projection"]
    committed = pq.read_table(tp)
    if not committed.equals(table):
        diffs.append(f"{name}: the committed table is not the projection of {full_path}")
    with open(mp, encoding="utf-8") as fh:
        cman = json.load(fh)
    want = projected_manifest(full_path, cman.get("table_sha256"), table.num_rows, runs)
    if cman.get("table_sha256") != _sha256(tp):
        diffs.append(f"{name}: manifest table_sha256 is not the committed table's sha256")
    for k in sorted(set(want) | set(cman)):
        if cman.get(k) != want.get(k):
            diffs.append(f"{name}: manifest {k} differs from the re-projection")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python paper/notebooks/strength_reads_sidecar.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("write", "check"))
    ap.add_argument("--source", action="append", required=True, metavar="NAME=PATH",
                    help="a full reads sidecar (strength_reads.parquet with its manifest.json beside it) and the name it is committed under")
    ap.add_argument("--out", default=OUT, help="the committed directory (default: paper/notebooks/data/strength_reads)")
    a = ap.parse_args(argv)
    diffs = []
    try:
        for s in a.source:
            name, _, path = s.partition("=")
            if not path:
                raise SidecarError(f"--source {s!r}: NAME=PATH")
            diffs += build(name, os.path.expanduser(path), a.out, a.cmd == "write")
    except SidecarError as exc:
        print(f"strength_reads_sidecar: REFUSED: {exc}", file=sys.stderr)
        return 2
    for d in diffs:
        print(d)
    if a.cmd == "check":
        print("check: " + ("clean" if not diffs else f"{len(diffs)} difference(s)"))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
