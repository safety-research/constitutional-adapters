#!/usr/bin/env python
"""Snapshot (relocate) COMPLETE fit cells a refit is about to SUPERSEDE.

The round-2 cap-24 refits publish into the SAME cell dirs as the round-1
winners (``<bank>/cells_all/<cell>/``): the Modal fit publishes with
``copytree(dirs_exist_ok=True)`` over whatever sits there. Without a
snapshot the round-1 winner objects — the ones every published round-1
number was measured on — would be overwritten in place, and a stale
COMPLETE marker would sit under a refit in flight looking exactly like a
finished cell. So, BEFORE any refit spawn, each to-be-superseded cell is
copied to a SNAPSHOT BANK with the complete bank layout::

    <root>/<model_key>_r1/cells_all/<cell>/        (+ _store_identity.json)

and both sides get a record:

  * ``<snapshot cell>/snapshot_of.json`` — where it came from, the
    COMPLETE fit fingerprint it preserves, the manifest it verified;
  * ``<source cell>/superseded_by.json`` — the snapshot's location and
    the fingerprint that was snapshotted, plus whatever the caller knows
    about the successor (the refit's expected fingerprint, a reason).

The Modal fit (scripts/modal/sdfbase_fits.py ``_fit_impl``) REFUSES to
train over a COMPLETE cell whose fingerprint differs from the fit it is
about to run unless that dir carries a ``superseded_by.json`` naming the
marker's fingerprint — the snapshot is the exact event that licenses the
overwrite, so an operator cannot skip it. The watcher and every consumer
that reads the snapshot bank see a normal bank: the same COMPLETE
protocol, the same identity gates (``train_budget.staged_cell_problems``
validates the copy exactly as it validates the live cell).

Discipline:
  * the SOURCE is verified against its own COMPLETE manifest (every
    listed file present, sha256 equal) BEFORE the copy — a source that
    has already been overwritten refuses (that is the sha mismatch the
    plan names); the COPY is verified the same way afterwards;
  * a snapshot is never overwritten: an existing snapshot cell that
    verifies identically is an idempotent no-op; one that differs refuses;
  * ``_store_identity.json`` is carried into the snapshot bank (a bank
    without it is not a bank); an existing one must match;
  * an earlier ``superseded_by.json`` in the source is NOT copied (it
    describes a previous supersede, not this snapshot).

Local + Modal: this module is pure filesystem, so it runs against the
grading host's harvest mirror (``$STAGE/sdfbase/out``) and, imported by the
app's ``--mode snapshot``, against the volume (``/vol/out``).

  python scripts/sdfbase/snapshot_cells.py snapshot \\
      --src-bank $STAGE/sdfbase/out/gemma4e4b_base \\
      --dst-bank $STAGE/sdfbase/out/gemma4e4b_base_r1 \\
      --cells constv2-mix-1m__psr_ce_all,constv2-mix-1m__lora_r4_ce \\
      --reason "round-2 cap-24 refit"
  python scripts/sdfbase/snapshot_cells.py verify --bank <dst-bank> \\
      --cells ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from build_heldout_split import sha256_file  # noqa: E402

COMPLETE = "COMPLETE"
STORE_IDENT = "_store_identity.json"
#: Written into the SOURCE cell dir: this publication was snapshotted and
#: may be superseded. The Modal fit's supersede gate keys on it.
SUPERSEDE_RECORD = "superseded_by.json"
#: Written into the SNAPSHOT cell dir: where the copy came from.
SNAPSHOT_RECORD = "snapshot_of.json"
#: The Modal fit's per-cell report/metrics tees under <bank>/reports/ —
#: overwritten by a refit like the cell dir is, so they ride along when
#: present (the cap-24 snapshot keeps the report).
REPORT_NAMES = ("report_{cell}_{leg}.json", "metrics_{cell}_{leg}.jsonl")


class SnapshotError(RuntimeError):
    """Loud refusal — a snapshot that cannot be proven is not taken."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_marker(cell_dir: Path) -> dict:
    """The cell's COMPLETE marker, required to parse, name the cell, carry
    a fit fingerprint and a non-empty files manifest."""
    cell_dir = Path(cell_dir)
    mp = cell_dir / COMPLETE
    if not mp.is_file():
        raise SnapshotError(f"{cell_dir} carries no COMPLETE marker — only "
                            "a published cell can be snapshotted")
    try:
        marker = json.loads(mp.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SnapshotError(f"{mp} does not parse: {e}") from e
    if not isinstance(marker, dict):
        raise SnapshotError(f"{mp} is not a JSON object")
    if marker.get("cell_id") != cell_dir.name:
        raise SnapshotError(
            f"{mp} names cell_id={marker.get('cell_id')!r}, dir is "
            f"{cell_dir.name!r} — a wrong-identity marker")
    if not marker.get("fit_fingerprint"):
        raise SnapshotError(f"{mp} carries no fit_fingerprint — a pre-"
                            "fingerprint marker cannot be tied to a refit")
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        raise SnapshotError(f"{mp} carries no publication manifest (files)")
    return marker


def manifest_problems(cell_dir: Path, marker: dict) -> list[str]:
    """Every manifested file present with the recorded sha256."""
    problems = []
    for name, sha in marker["files"].items():
        p = Path(cell_dir) / str(name)
        if not p.is_file():
            problems.append(f"{name}: missing")
        elif sha256_file(p) != sha:
            problems.append(f"{name}: sha256 {sha256_file(p)[:12]}… != "
                            f"manifested {str(sha)[:12]}…")
    return problems


def _bank_identity(bank: Path, leg: str) -> Path:
    return Path(bank) / f"cells_{leg}" / STORE_IDENT


def _carry_store_identity(src_bank: Path, dst_bank: Path, leg: str) -> None:
    src = _bank_identity(src_bank, leg)
    dst = _bank_identity(dst_bank, leg)
    if not src.is_file():
        raise SnapshotError(f"{src} missing — the source is not a bank "
                            "(no store identity); refusing to snapshot "
                            "from an unidentified root")
    if dst.is_file():
        if dst.read_bytes() != src.read_bytes():
            raise SnapshotError(
                f"{dst} already exists and differs from {src} — the "
                "snapshot bank belongs to another (model, revision, leg)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def supersede_record(cell_dir: Path) -> dict | None:
    """The source-side record, or None when the cell was never snapshotted."""
    p = Path(cell_dir) / SUPERSEDE_RECORD
    if not p.is_file():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise SnapshotError(f"{p} does not parse: {e}") from e
    return rec if isinstance(rec, dict) else None


def _carry_reports(src_bank: Path, dst_bank: Path, cell_id: str,
                   leg: str) -> dict:
    """Copy ``<src_bank>/reports/{report,metrics}_<cell>_<leg>.*`` into the
    snapshot bank's ``reports/`` when present (never required — a locally
    fit bank has none). An identical existing copy is a no-op; a differing
    one refuses (a snapshot is never overwritten). Returns
    ``{"reports/<name>": sha256}`` for the snapshot record."""
    out: dict = {}
    for pat in REPORT_NAMES:
        name = pat.format(cell=cell_id, leg=leg)
        src = Path(src_bank) / "reports" / name
        if not src.is_file():
            continue
        sha = sha256_file(src)
        dst = Path(dst_bank) / "reports" / name
        if dst.is_file():
            if sha256_file(dst) != sha:
                raise SnapshotError(
                    f"{dst} already exists and differs from {src} — a "
                    "snapshot is never overwritten; move it aside "
                    "deliberately")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copy2(src, tmp)
            tmp.replace(dst)
        out[f"reports/{name}"] = sha
    return out


def snapshot_cell(src_bank: Path, dst_bank: Path, cell_id: str, *,
                  superseded_by: dict | None = None, leg: str = "all"
                  ) -> dict:
    """Copy ``<src_bank>/cells_<leg>/<cell_id>`` into the snapshot bank,
    verified on both sides, and write the two records. Idempotent on an
    identical existing snapshot. Returns the snapshot record."""
    src_bank, dst_bank = Path(src_bank), Path(dst_bank)
    if src_bank.resolve() == dst_bank.resolve():
        raise SnapshotError("source and snapshot bank are the same dir")
    src = src_bank / f"cells_{leg}" / cell_id
    dst = dst_bank / f"cells_{leg}" / cell_id
    marker = read_marker(src)
    problems = manifest_problems(src, marker)
    if problems:
        raise SnapshotError(
            f"{src} no longer matches its COMPLETE manifest — the source "
            "was overwritten or torn after publish; refusing to snapshot "
            "a cell that is not the publication its marker describes:\n"
            "  - " + "\n  - ".join(problems))
    _carry_store_identity(src_bank, dst_bank, leg)

    already = False
    if dst.exists():
        existing = read_marker(dst)
        if existing.get("fit_fingerprint") != marker["fit_fingerprint"] \
                or manifest_problems(dst, existing):
            raise SnapshotError(
                f"{dst} already holds a DIFFERENT snapshot (fingerprint "
                f"{str(existing.get('fit_fingerprint'))[:12]}… vs source "
                f"{marker['fit_fingerprint'][:12]}…, or a torn copy) — a "
                "snapshot is never overwritten; move it aside deliberately")
        already = True
    else:
        tmp = dst.with_name(f".{cell_id}.snapshot-tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp,
                        ignore=shutil.ignore_patterns(SUPERSEDE_RECORD))
        tmp.replace(dst)
    problems = manifest_problems(dst, marker)
    if problems:
        raise SnapshotError(
            f"the snapshot at {dst} does not verify against the source's "
            "COMPLETE manifest:\n  - " + "\n  - ".join(problems))
    # The whole cell DIR travelled (copytree): a retained final_ckpt/ and
    # the PSR provenance sidecar included. The bank-level report/metrics
    # tees ride along too.
    reports = _carry_reports(src_bank, dst_bank, cell_id, leg)

    record = {
        "cell_id": cell_id,
        "leg": leg,
        "fit_fingerprint": marker["fit_fingerprint"],
        "source": str(src),
        "source_bank": str(src_bank),
        "snapshot": str(dst),
        "snapshot_bank": str(dst_bank),
        "files": dict(marker["files"]),
        "final_ckpt_retained": (dst / "final_ckpt" / "ckpt.pt").is_file(),
        "reports": reports,
        "snapshot_at": _now(),
        "already_present": already,
    }
    (dst / SNAPSHOT_RECORD).write_text(json.dumps(record, indent=1) + "\n",
                                       encoding="utf-8")
    src_record = {
        "cell_id": cell_id,
        # The identity that WAS snapshotted — the Modal fit's supersede
        # gate requires this to equal the live marker's fingerprint, so
        # a second refit over a refit needs its own snapshot first.
        "fit_fingerprint": marker["fit_fingerprint"],
        "snapshot": str(dst),
        "snapshot_bank": str(dst_bank),
        "superseded_by": dict(superseded_by or {}),
        "at": record["snapshot_at"],
    }
    tmp = src / (SUPERSEDE_RECORD + ".tmp")
    tmp.write_text(json.dumps(src_record, indent=1) + "\n", encoding="utf-8")
    tmp.replace(src / SUPERSEDE_RECORD)
    return record


def verify_snapshot(bank: Path, cell_id: str, *, leg: str = "all") -> dict:
    """Re-verify a snapshot cell: COMPLETE marker + manifest, and the
    snapshot record naming the same fingerprint. Returns the record."""
    cell = Path(bank) / f"cells_{leg}" / cell_id
    marker = read_marker(cell)
    problems = manifest_problems(cell, marker)
    rp = cell / SNAPSHOT_RECORD
    if not rp.is_file():
        problems.append(f"{SNAPSHOT_RECORD} missing — not a snapshot cell")
        record = None
    else:
        record = json.loads(rp.read_text(encoding="utf-8"))
        if record.get("fit_fingerprint") != marker["fit_fingerprint"]:
            problems.append(
                f"{SNAPSHOT_RECORD} records fingerprint "
                f"{str(record.get('fit_fingerprint'))[:12]}… but the marker "
                f"carries {marker['fit_fingerprint'][:12]}…")
    if not _bank_identity(bank, leg).is_file():
        problems.append(f"{_bank_identity(bank, leg)} missing")
    if problems:
        raise SnapshotError(f"snapshot {cell} does not verify:\n  - "
                            + "\n  - ".join(problems))
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot", help="copy + verify + write records")
    p.add_argument("--src-bank", required=True, type=Path,
                   help="the live bank root (holds cells_all/)")
    p.add_argument("--dst-bank", required=True, type=Path,
                   help="the snapshot bank root, e.g. <root>/<key>_r1")
    p.add_argument("--cells", required=True,
                   help="comma list of cell ids to snapshot")
    p.add_argument("--reason", default="",
                   help="recorded in superseded_by (e.g. 'round-2 cap-24 "
                        "refit')")
    p.add_argument("--expected-fingerprint", action="append", default=[],
                   metavar="CELL=FP",
                   help="the successor fit's expected fingerprint for a "
                        "cell, recorded in superseded_by (repeatable)")
    p.add_argument("--leg", default="all")
    v = sub.add_parser("verify", help="re-verify snapshot cells")
    v.add_argument("--bank", required=True, type=Path)
    v.add_argument("--cells", required=True)
    v.add_argument("--leg", default="all")
    args = ap.parse_args(argv)

    cells = [c for c in args.cells.split(",") if c]
    rc = 0
    if args.cmd == "snapshot":
        expected = dict(kv.split("=", 1) for kv in args.expected_fingerprint)
        for cell in cells:
            try:
                rec = snapshot_cell(
                    args.src_bank, args.dst_bank, cell, leg=args.leg,
                    superseded_by={
                        "reason": args.reason,
                        "expected_refit_fingerprint": expected.get(cell)})
            except SnapshotError as e:
                print(f"[snapshot] REFUSED {cell}: {e}", file=sys.stderr)
                rc = 2
                continue
            print(f"[snapshot] {cell}: {rec['source']} -> {rec['snapshot']} "
                  f"(fingerprint {rec['fit_fingerprint'][:12]}…"
                  f"{', already present' if rec['already_present'] else ''})")
    else:
        for cell in cells:
            try:
                rec = verify_snapshot(args.bank, cell, leg=args.leg)
            except SnapshotError as e:
                print(f"[snapshot] VERIFY FAILED {cell}: {e}",
                      file=sys.stderr)
                rc = 2
                continue
            print(f"[snapshot] {cell}: verified "
                  f"(fingerprint {rec['fit_fingerprint'][:12]}…)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
