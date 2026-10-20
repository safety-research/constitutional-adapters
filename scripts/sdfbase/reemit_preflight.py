#!/usr/bin/env python3
"""reemit_preflight — the exact registry records a corrective re-emission WOULD write, diffed against the registry's current rows
BEFORE anything is spooled.

A re-emission restates the whole object record and the fold takes every stated column from the newer record, so a re-emission run
from a different context would have overwritten the fit's provenance (fit_code_sha / fit_job_id / fit_node_hostname) with the
replaying job's. This tool builds the records through the producers' own call sites with the hook in dry-run mode (nothing is
written, no evidence is touched), then refuses — exit 3 — unless every object row the records would restate differs from its current
row ONLY in the allowed columns: the ingest tail and `campaign`, plus the fit-provenance columns fit_code_sha / fit_job_id /
fit_node_hostname / fit_substrate — and the artifact row's created_utc — when, and only when, they move to EXACTLY the values a
`fit_job_provenance.json` sidecar attests from the fit's own records (the RESULT line's job + node, the funnel attempt record's pinned
sha, the fit job's own first record; the rows state what the FIT stated — the re-emission chain had overwritten all
five). A row the registry does not hold yet (V) is admitted as new. The current rows come as a JSON dump made on the
registry host (the diff_object_rows.py dump: ``{"object": {id: row}, "artifact": {id: row}}`` or the flat ``{id: row}`` form).

  reemit_preflight.py fit --dir <cell dir> --si <si_id> --arm <arm> --campaign <new> --rows <dump.json> [--pre-rows <earlier dump>]
                          [--out <planned.json>]
  reemit_preflight.py derived --dir <V dir> --object-kind composed --derivation composition --script <writer> --campaign <new>
                              --rows <dump.json> [--pre-rows <earlier dump>] [--out <planned.json>]
  reemit_preflight.py fresh --pre-rows <earlier dump> --rows <fresh dump>      (the precondition alone)
  reemit_preflight.py audit --before <dump> --after <dump> --ours <file of our record sha256s, one per line>

Freshness: the row dump is a copy, so `--pre-rows` (the dump the step started from) against `--rows` (a dump
pulled immediately before the emit) REFUSES unless every row's ingest_seq and field_versions_json are identical — another writer's
fold between the two would otherwise be overwritten by our restatement. `audit` is the post-fold check: for every column whose
version stamp moved between the dumps, the stamp's record sha must be ONE OF OURS (the records this re-emission wrote); a column moved
by a record that is not ours refuses — loud, never a silent clean.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ALLOWED_ALWAYS = ("campaign", "record_uri", "ingest_seq", "field_versions_json")
FRESHNESS_KEYS = ("ingest_seq", "field_versions_json")
PROVENANCE_MOVABLE = ("fit_code_sha", "fit_job_id", "fit_node_hostname", "fit_substrate")
ARTIFACT_MOVABLE = ("created_utc",)            # admitted only when it equals the sidecar's attested artifact_created_utc


def _load_rows(path: str) -> tuple[dict, dict]:
    doc = json.load(open(path, encoding="utf-8"))
    if "object" in doc and isinstance(doc["object"], dict) and all(isinstance(v, dict) for v in doc["object"].values()):
        return doc["object"], doc.get("artifact") or {}
    return doc, {}


def _canon(table: str, payload: dict) -> dict:
    """The row as the ingestor would write it: JSON columns serialised canonically, bools as ints, `extra` folded out."""
    from sisteer.registry import validate as VAL
    known, extra = VAL._normalise_payload(table, payload)
    known["extra_json"] = extra
    return known


def _records(args) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.mode == "fit":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import train_two_si_objects as t2  # noqa: E402
        recs = t2._registry_publish(Path(args.dir), si_id=args.si, arm=args.arm, dry_run=True)
    else:
        import psr_residual_distill as prd  # noqa: E402
        from sisteer.registry import hooks
        meta = json.loads((Path(args.dir) / "derived_meta.json").read_text(encoding="utf-8"))
        recs = prd._derived_hook_call(Path(args.dir), meta, object_kind=args.object_kind, derivation=args.derivation, script=args.script,
                                      extra=None, emitter=lambda d, **kw: hooks.emit_object_from_sidecar(d, dry_run=True, **kw))
    if not isinstance(recs, list) or not recs or not all(isinstance(r, (tuple, list)) and len(r) == 2 for r in recs):
        raise SystemExit(f"REFUSED: the dry run produced no records ({recs!r}) — the hook refused (see the [registry] notice above) or the "
                         "producer stated not_emitted; nothing to preflight, nothing to emit")
    return recs


def _attestation_of(doc: dict | None) -> dict:
    """The movable columns a fit_job_provenance.json attests (the four fit columns + the artifact's own created time)."""
    doc = doc or {}
    att = {k: doc[k] for k in PROVENANCE_MOVABLE if k in doc}
    if doc.get("artifact_created_utc"):
        att["artifact_created_utc"] = str(doc["artifact_created_utc"])
    return att


def _attested(args) -> dict:
    """The CELL's attestation (the parent's sidecar) — reported; each record is checked against the sidecar of ITS OWN directory
    (:func:`_attestation_for`), so an epoch artifact may move only to the time its own epoch sidecar attests."""
    from sisteer.registry import hooks
    return _attestation_of(hooks.read_fit_provenance_sidecar(args.dir))


def _record_dir(payload: dict) -> str | None:
    """The directory a record's artefact lives in: the artifact's uri_primary, or the object's first artifact location."""
    uri = payload.get("uri_primary")
    if not uri:
        locs = payload.get("artifact_locations_json") or []
        uri = locs[0] if locs else None
    return os.path.dirname(str(uri)) if uri else None


def _attestation_for(payload: dict, cache: dict) -> dict:
    """The attestation that governs THIS record: the fit_job_provenance.json beside its artefact (the parent's for the published
    object and its artifact, the epoch dir's for an epoch row and its artifact); {} when that directory carries none."""
    from sisteer.registry import hooks
    d = _record_dir(payload)
    if d is None:
        return {}
    if d not in cache:
        try:
            cache[d] = _attestation_of(hooks.read_fit_provenance_sidecar(d))
        except ValueError:
            cache[d] = {}
    return cache[d]


def preflight(recs: list, rows: dict, art_rows: dict, attested: dict, *, campaign: str) -> tuple[list, list]:
    """(problems, report lines). A problem is any disallowed difference; the report names every row's outcome. `attested` is the
    cell's attestation (reported); every record is held to the attestation of its OWN directory (fix-5)."""
    problems, report = [], []
    cache: dict = {}
    for rtype, payload in recs:
        own = _attestation_for(payload, cache)
        if rtype == "object":
            oid = str(payload.get("object_id"))
            row = rows.get(oid)
            if row is None:
                report.append(f"object {oid[:16]}: NEW row (campaign {payload.get('campaign')})")
                continue
            new = _canon("object", payload)
            if new.get("campaign") != campaign:
                problems.append(f"object {oid[:16]}: payload campaign {new.get('campaign')!r} != the stated {campaign!r}")
            changed = []
            for col, val in new.items():
                if col == "extra_json":
                    cur = json.loads(row.get("extra_json") or "{}")
                    for k, v in (val or {}).items():
                        if k == "fit_provenance":
                            continue                      # the correction's own provenance note is new by design
                        if json.dumps(cur.get(k), sort_keys=True, default=str) != json.dumps(v, sort_keys=True, default=str):
                            changed.append(f"extra.{k}")
                    continue
                cur = row.get(col)
                if str(cur) == str(val) or cur == val:
                    continue
                if col in ALLOWED_ALWAYS:
                    continue
                if col in PROVENANCE_MOVABLE and col in own and str(val) == str(own[col]):
                    report.append(f"object {oid[:16]}: {col} {cur!r} -> {val!r} (attested by its own {os.path.basename(_record_dir(payload) or '')}/fit_job_provenance.json)")
                    continue
                changed.append(f"{col}: {str(cur)[:60]!r} -> {str(val)[:60]!r}")
            if changed:
                problems.append(f"object {oid[:16]}: disallowed changes: " + "; ".join(changed))
            else:
                report.append(f"object {oid[:16]}: OK (campaign {row.get('campaign')!r} -> {campaign!r})")
        elif rtype == "artifact":
            aid = str(payload.get("artifact_id"))
            row = art_rows.get(aid)
            if row is None:
                continue
            for col in ("kind", "uri_primary", "bytes", "created_utc"):
                if col in payload and str(row.get(col)) != str(payload[col]):
                    # an artifact's created time may move ONLY to the time ITS OWN directory's sidecar attests (the parent's for the
                    # published object, each epoch's for its checkpoint) — never the parent's time for an epoch, never unattested
                    if col in ARTIFACT_MOVABLE and own.get("artifact_created_utc") and str(payload[col]) == str(own["artifact_created_utc"]):
                        report.append(f"artifact {aid[:16]}: created_utc {str(row.get(col))!r} -> {str(payload[col])!r} (attested by its own "
                                      f"{os.path.basename(_record_dir(payload) or '')}/fit_job_provenance.json)")
                        continue
                    problems.append(f"artifact {aid[:16]}: {col} {str(row.get(col))[:40]!r} -> {str(payload[col])[:40]!r}"
                                    + (" (no attestation in its own directory)" if col in ARTIFACT_MOVABLE and not own.get("artifact_created_utc") else ""))
    return problems, report


def freshness_problems(pre_path: str, rows_path: str) -> list:
    """Rows that moved between the step's starting dump and the fresh one (ingest_seq / field_versions_json), or appeared / vanished."""
    pre_o, pre_a = _load_rows(pre_path)
    now_o, now_a = _load_rows(rows_path)
    problems = []
    for table, pre, now in (("object", pre_o, now_o), ("artifact", pre_a, now_a)):
        for rid in sorted(set(pre) | set(now)):
            if rid not in now:
                problems.append(f"{table} {rid[:16]}: present in the starting dump, absent now"); continue
            if rid not in pre:
                problems.append(f"{table} {rid[:16]}: appeared since the starting dump"); continue
            for k in FRESHNESS_KEYS:
                if str(pre[rid].get(k)) != str(now[rid].get(k)):
                    problems.append(f"{table} {rid[:16]}: {k} moved {str(pre[rid].get(k))[:60]!r} -> {str(now[rid].get(k))[:60]!r} (another writer folded)")
                    break
    return problems


def _versions(row: dict) -> dict:
    try:
        v = json.loads(row.get("field_versions_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def audit_problems(before_path: str, after_path: str, ours: set) -> tuple[list, list]:
    """(problems, report): every column whose version stamp moved between the dumps must carry one of OUR record shas."""
    b_o, b_a = _load_rows(before_path)
    a_o, a_a = _load_rows(after_path)
    problems, report = [], []
    for table, before, after in (("object", b_o, a_o), ("artifact", b_a, a_a)):
        for rid in sorted(set(before) | set(after)):
            if rid not in after:
                problems.append(f"{table} {rid[:16]}: row disappeared after the fold"); continue
            vb, va = _versions(before.get(rid, {})), _versions(after[rid])
            moved = {c: va[c] for c in va if vb.get(c) != va[c]}
            foreign = {c: st for c, st in moved.items() if not (isinstance(st, list) and len(st) == 2 and str(st[1]) in ours)}
            if foreign:
                problems.append(f"{table} {rid[:16]}: column(s) moved by a record that is NOT ours: " +
                                "; ".join(f"{c} <- {str(st[1])[:12]}… @ {st[0]}" if isinstance(st, list) and len(st) == 2 else f"{c} <- {st!r}" for c, st in sorted(foreign.items())))
            else:
                report.append(f"{table} {rid[:16]}: {len(moved)} column(s) moved, all by our records" + (" (new row)" if rid not in before else ""))
    return problems, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("fit", "derived", "fresh", "audit"))
    ap.add_argument("--dir")
    ap.add_argument("--campaign")
    ap.add_argument("--rows", help="the registry host's row dump (diff_object_rows.py dump), pulled immediately before the emit")
    ap.add_argument("--pre-rows", default=None, help="the dump the step started from: refuse unless --rows is identical in ingest_seq + field_versions_json")
    ap.add_argument("--before"); ap.add_argument("--after"); ap.add_argument("--ours")
    ap.add_argument("--si"); ap.add_argument("--arm")
    ap.add_argument("--object-kind", default="composed"); ap.add_argument("--derivation", default="composition")
    ap.add_argument("--script", default="scripts/sdfbase/psr_residual_distill.py compose-v")
    ap.add_argument("--out", default=None, help="write the planned records here (the record of what was about to be emitted)")
    args = ap.parse_args(argv)
    if args.mode == "audit":
        if not (args.before and args.after and args.ours):
            ap.error("audit needs --before, --after and --ours")
        ours = {ln.strip() for ln in Path(args.ours).read_text().splitlines() if ln.strip()}
        problems, report = audit_problems(args.before, args.after, ours)
        for line in report:
            print("[audit]", line)
        if problems:
            print("[audit] REFUSED — the fold moved columns by records that are not ours:", file=sys.stderr)
            for pr in problems:
                print("  - " + pr, file=sys.stderr)
            return 3
        print(f"[audit] CLEAN — every moved column carries one of our {len(ours)} record sha(s)")
        return 0
    if args.mode == "fresh":
        if not (args.pre_rows and args.rows):
            ap.error("fresh needs --pre-rows and --rows")
        problems = freshness_problems(args.pre_rows, args.rows)
        if problems:
            print("[fresh] REFUSED — the registry moved since the step started:", file=sys.stderr)
            for pr in problems:
                print("  - " + pr, file=sys.stderr)
            return 3
        print("[fresh] CLEAN — ingest_seq and field_versions_json identical row for row")
        return 0
    if not (args.dir and args.campaign and args.rows):
        ap.error(f"{args.mode} needs --dir, --campaign and --rows")
    if args.mode == "fit" and not (args.si and args.arm):
        ap.error("fit needs --si and --arm")
    if args.pre_rows:
        problems = freshness_problems(args.pre_rows, args.rows)
        if problems:
            print("[preflight] REFUSED — the registry moved since the step started (nothing emitted):", file=sys.stderr)
            for pr in problems:
                print("  - " + pr, file=sys.stderr)
            return 3
        print("[preflight] fresh: the rows are identical to the step's starting dump (ingest_seq, field_versions_json)")
    os.environ["REGISTRY_CAMPAIGN"] = args.campaign
    recs = _records(args)
    rows, art_rows = _load_rows(args.rows)
    attested = _attested(args)
    problems, report = preflight(recs, rows, art_rows, attested, campaign=args.campaign)
    if args.out:
        Path(args.out).write_text(json.dumps({"campaign": args.campaign, "attested": attested, "records": recs}, indent=1, default=str))
    for line in report:
        print("[preflight]", line)
    print(f"[preflight] {len(recs)} record(s) planned; attested provenance moves: {attested or 'none'}")
    if problems:
        print("[preflight] REFUSED — the re-emission would change:", file=sys.stderr)
        for pr in problems:
            print("  - " + pr, file=sys.stderr)
        return 3
    print("[preflight] CLEAN — every restated row differs only in the allowed columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
