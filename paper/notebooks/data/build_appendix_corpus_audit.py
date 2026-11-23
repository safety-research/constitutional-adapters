#!/usr/bin/env python3
"""paper/notebooks/data/build_appendix_corpus_audit.py: the committed aggregate counts behind tab:corpus-audit and figA_corpus_audit.

    python paper/notebooks/data/build_appendix_corpus_audit.py --audit AUDIT_DIR --lane LANE_DIR            # write the sidecar
    python paper/notebooks/data/build_appendix_corpus_audit.py --audit AUDIT_DIR --lane LANE_DIR --check    # compare, write nothing

AUDIT_DIR is the corpus audit's output directory (it holds `summary.json` and `scan/`); LANE_DIR the number check of the
manuscript's literals (`numbers_ledger.tsv`, `figA_corpus_audit_values.tsv`). Both stay outside this repository; the environment
variables SISTEER_CORPUS_AUDIT_DIR and SISTEER_CORPUS_AUDIT_LANE stand in for the flags, and `--no-lane` skips the cross-checks.

The training-corpus audit labelled every record of the two training corpora; its summary file (`summary.json`,
version hrc-summary-v2, rubric hrc-v2) holds about 25,000 aggregate numbers keyed by name, and its overlap scan's metadata
(`scan/overlap_meta.json`, `scan/overlap_expected.csv`) the scan's thresholds and the role of each scanned set. Neither is in this
repository. This script copies into the sidecar ONLY what the appendix table and figure use, each number under its summary key
(`appendix_corpus_audit.required_keys()` names them, plus the overlap maxima of every scanned set), with the set roles and thresholds
and nothing per record: no record text, no record id, no labelled span. A scanned set of an evaluation family the paper omits
(`appendix_corpus_audit.omitted_family_set`, from the notebook's list of omitted evaluations) is left out: its id is not listed and its
numbers are not copied, and `overlap.sets_left_out` keeps only how many such sets there were, their longest shared run and their
records over a threshold (a record over a threshold in a set left out is refused: the fuller table's footnote counts those records).

Before writing it checks the sources and the copy:
  * the summary's version and rubric, and that the two scan files are the ones the summary was built from (their sha256 against the
    summary's own `inputs` map);
  * that each claim's states partition every corpus part (positive + negative + contested + partial + unresolved + declined on input =
    the part's records; negatives are read here and not stored);
  * every value against the number ledger (`numbers_ledger.tsv`, the key-by-key check of the manuscript's literals; a ledger row of a
    set left out against the summary's own number, which is not copied) and the interim figure's value table
    (`figA_corpus_audit_values.tsv`) where those files exist, refusing on any difference;
  * `appendix_corpus_audit.corpus_audit_problems` on the result (stamp, provenance, no record id, internal consistency);
  * the publication scan (`reproduce/tools/scrub_grep.sh --all`) on the new file: it is written beside the target, scanned, and moved
    into place only on a clean scan (`--no-scrub` skips the scan, for a host without the repository's scan tools).

The output is deterministic (no timestamp): the same sources and builder give the same bytes, so `--check` compares whole files. The
labels are by Claude Opus 5.5 acting as a corpus labeller, not an evaluation judge; the overlap numbers are a deterministic scan.
Needs the notebook environment (`uv sync --frozen --extra cpu --group notebook`); exits 0 on success, 1 on a refusal or a difference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                   # paper/notebooks/data
NOTEBOOKS = HERE.parent
REPO = NOTEBOOKS.parents[1]
sys.path.insert(0, str(NOTEBOOKS))
import appendix_corpus_audit as CA  # noqa: E402

AUDIT_ENV, LANE_ENV = "SISTEER_CORPUS_AUDIT_DIR", "SISTEER_CORPUS_AUDIT_LANE"
# THE SOURCES ARE NAMED RELATIVE TO TWO ROOTS, never by a host path: the files live outside the repository, so the sidecar says which
# root each is under and its path inside it, plus its sha256, which is what identifies it.
ROOTS = {"audit": "the corpus audit's output directory (summary.json, scan/); outside this repository; the builder's --audit",
         "lane": "the key-by-key check of the manuscript's literals against the audit summary; outside this repository; the builder's --lane"}
OUT = Path(CA.SIDECAR)
SCRUB = REPO / "reproduce" / "tools" / "scrub_grep.sh"
ABOUT = ("Aggregate counts of the training-corpus audit behind the appendix table tab:corpus-audit and the notebook figure "
         "figA_corpus_audit: per corpus part, the records and the records the labeller's safety classifier declined on input; per "
         "claim (C2 harmful requests, the C1 jailbreak half, C3 wrongful acts) the positive, contested, partially read, unresolved "
         "and declined-on-input records and the positive share of read records; the C1 eval-text records; the C2 request treatment "
         "and uplift counts; the C2 agreement kappa; the blind reread's counts and bound; and the deterministic overlap scan's "
         "thresholds and per-set maxima. Each value under `numbers` is the audit summary's own number under the same key. No record "
         "text and no record id. Regenerate with paper/notebooks/data/build_appendix_corpus_audit.py.")
# the scan's per-set fields that may be stored (the scan table also names the item and record of each maximum, which never are)
SET_FIELDS = ("set_id", "tier", "role", "status", "goal_set")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source(root: str, base: Path, path: Path, **extra) -> dict:
    """A provenance source: its root, its path relative to that root, its sha256, and any extra fields."""
    return {"root": root, "file": path.relative_to(base).as_posix(), "sha256": sha256(path), **extra}


def shown(path: Path) -> str:
    """`path` relative to the repository when inside it, else its name: what the builder prints."""
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.name


def _bool(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def derive(audit: Path, lane: Path | None) -> tuple:
    """(sidecar dict, problems). Reads the summary and the scan metadata under `audit`, cross-checks against the ledger and value
    table under `lane` when they exist."""
    problems = []
    summary_path = audit / "summary.json"
    meta_path = audit / "scan" / "overlap_meta.json"
    expected_path = audit / "scan" / "overlap_expected.csv"
    S = json.loads(summary_path.read_text(encoding="utf-8"))
    N = S["numbers"]
    if S.get("summary_version") != CA.SUMMARY_VERSION or S.get("rubric_version") != CA.RUBRIC_VERSION:
        problems.append(f"summary {S.get('summary_version')!r} rubric {S.get('rubric_version')!r}, expected "
                        f"{CA.SUMMARY_VERSION!r} / {CA.RUBRIC_VERSION!r}")
    inputs = S.get("inputs") or {}
    for rel, path in (("scan/overlap_meta.json", meta_path), ("scan/overlap_expected.csv", expected_path)):
        if inputs.get(rel) != sha256(path):
            problems.append(f"{rel}: sha256 {sha256(path)[:12]} is not the {str(inputs.get(rel))[:12]} the summary was built from")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with open(expected_path, encoding="utf-8", newline="") as fh:
        expected = [{k: r.get(k) for k in ("set_id", "tier", "role", "status")} for r in csv.DictReader(fh)]
    # the scan table lists the expected-by-construction sets too; they are kept apart (`expected_sets`), their runs under overlap_expected
    exp_ids = {e["set_id"] for e in expected}
    scanned = [{k: (_bool(s.get(k)) if k == "goal_set" else s.get(k)) for k in SET_FIELDS} | {"standin": bool(s.get("standin"))}
               for s in meta["set_table"] if s["set_id"] not in exp_ids]
    # THE SETS OF AN EVALUATION FAMILY THE PAPER OMITS are left out: not listed, their numbers not copied; their count, longest shared
    # run and records over a threshold are kept (appendix_corpus_audit.omitted_family_set names the rule, from the notebook's own list)
    sets = [s for s in scanned if not CA.omitted_family_set(s["set_id"])]
    left = [s["set_id"] for s in scanned if CA.omitted_family_set(s["set_id"])]
    th = meta["thresholds"]
    set_ids = [s["set_id"] for s in sets]
    goal_sets = [s["set_id"] for s in sets if s["goal_set"]]
    keys = CA.required_keys() + CA.overlap_keys(set_ids, goal_sets, [e["set_id"] for e in expected])
    missing = [k for k in keys if k not in N]
    if missing:
        problems.append(f"{len(missing)} key(s) absent from the summary, e.g. {missing[:5]}")
    numbers = {k: N[k] for k in keys if k in N}
    left_keys = [k for sid in left for k in (f"overlap.max_run.{sid}", f"overlap.records_eval_text_positive.{sid}")]
    if any(k not in N for k in left_keys):
        problems.append(f"{sum(k not in N for k in left_keys)} overlap key(s) of a set left out absent from the summary")
    left_out = {"n_sets": len(left), "max_run": max((int(N.get(f"overlap.max_run.{sid}", 0)) for sid in left), default=0),
                "records_over_threshold": sum(int(N.get(f"overlap.records_eval_text_positive.{sid}", 0)) for sid in left),
                "why": CA.LEFT_OUT_WHY}
    numbers.update({k: N[k] for k in CA.optional_keys() if k in N})
    # EACH CLAIM'S STATES PARTITION EVERY PART: the negatives (read here, never stored) close the sum to the part's records
    for _c, pre, _stem, _t in CA.CLAIMS:
        for p in CA.PARTS:
            tot = sum(N[f"{pre}.{s}.{p}"] for s in CA.STATES + ("negative",))
            if tot != N[f"coverage.records.{p}"]:
                problems.append(f"{pre} {p}: the six states sum to {tot}, not the part's {N[f'coverage.records.{p}']} records")
    # THE UPLIFT LEVELS the summary names anywhere (footnote c says none is above partial): every `uplift.<level>.` key segment
    levels = sorted({m.group(1) for k in N for m in [re.search(r"(?:^|\.)uplift\.(?:response_side_unit:)?([a-z_]+)\.", k)] if m})
    # A per-set field of the scan metadata must agree with the summary's number for the same set
    for s in sets:
        if s["set_id"] == CA.PETRI_SET and s["role"] != "attack_corpus":
            problems.append(f"{CA.PETRI_SET}: role {s['role']!r}, expected 'attack_corpus'")
    for s in meta["set_table"]:
        if s["set_id"] in exp_ids:
            continue
        for col, key in (("max_run", f"overlap.max_run.{s['set_id']}"),
                         ("records_eval_text_positive", f"overlap.records_eval_text_positive.{s['set_id']}")):
            if key in N and int(s[col]) != N[key]:
                problems.append(f"{key}: summary {N[key]}, scan table {s[col]}")
    cross = []
    if lane is not None:
        ledger = lane / "numbers_ledger.tsv"
        if ledger.exists():
            matched = 0
            with open(ledger, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    k = r["summary.json key"]
                    # a row of a set left out is checked against the summary's own number, which the sidecar does not copy
                    where = "sidecar" if k in numbers else "summary (a set left out)" if k in left_keys and k in N else None
                    if where:
                        matched += 1
                        have = numbers[k] if k in numbers else N[k]
                        if float(r["raw"]) != float(have):
                            problems.append(f"ledger {k}: {r['raw']}, {where} {have}")
                        if r["check"] != "OK":
                            problems.append(f"ledger {k}: check {r['check']!r}")
            cross.append(source("lane", lane, ledger, rows_matched=matched))
        values = lane / "figA_corpus_audit_values.tsv"
        if values.exists():
            matched = 0
            claim = {pre: c for c, pre, _s, _t in CA.CLAIMS}
            with open(values, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    pre, p = r["claim"], r["part"]
                    other = sum(N[f"{pre}.{s}.{p}"] for s in CA.OTHER_NOT_CLEARED)
                    got = (N[f"{pre}.positive.{p}"], other, N[f"{pre}.declined_on_input.{p}"])
                    want = (int(r["positive"]), int(r["contested+partial+unresolved"]), int(r["declined_on_input"]))
                    matched += 1
                    if pre not in claim or got != want:
                        problems.append(f"interim figure values {pre} {p}: {want}, summary {got}")
            cross.append(source("lane", lane, values, rows_matched=matched))
    out = {
        "schema": CA.SCHEMA,
        "about": ABOUT,
        "judge": CA.JUDGE,
        "parts": list(CA.PARTS),
        "numbers": {k: numbers[k] for k in sorted(numbers)},
        "overlap": {"thresholds": {"route_run": int(th["route_run"]), "goal_frac": float(th["goal_frac"])},
                    "run_rule": meta.get("run_rule"), "sets": sets, "expected_sets": expected, "sets_left_out": left_out},
        "uplift_levels": levels,
        "provenance": {
            "summary_version": S.get("summary_version"), "rubric_version": S.get("rubric_version"),
            "judge": {"labeller_model": CA.LABELLER_MODEL, "role": "corpus labeller (an audit of the training data)",
                      "evaluation_judge": False,
                      "deterministic": "the C1 eval-text counts and the overlap maxima: an n-gram scan, no labeller"},
            "builder": CA.BUILDER, "builder_sha256": sha256(Path(__file__).resolve()),
            "roots": ROOTS,
            "sources": [
                source("audit", audit, summary_path,
                       read="numbers: every value under `numbers` is this file's numbers[key]; the negatives for the partition check"),
                source("audit", audit, meta_path,
                       read="overlap thresholds, run rule, and each scanned set's id, tier, role, status and goal-set flag (set_table)"),
                source("audit", audit, expected_path, read="the expected-by-construction sets' id, tier, role and status")],
            "flags_sha256": S.get("flags_sha256"),
            "cross_checked": cross,
        },
    }
    problems += CA.corpus_audit_problems(out)
    return out, problems


def dump(obj: dict) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def scrub_clean(paths: list) -> tuple:
    """(clean, report): the publication scan over `paths` (repository-relative), read from its summary line as ctlstop_sidecar.py does."""
    env = dict(os.environ, SCRUB_PYTHON=sys.executable)
    p = subprocess.run(["bash", str(SCRUB), "--all", *paths], cwd=REPO, env=env, capture_output=True, text=True)
    report = (p.stdout + p.stderr).strip()
    last = report.splitlines()[-1] if report else ""
    return p.returncode == 0 and last.startswith("scrub: no identifier under"), report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit", type=Path, default=os.environ.get(AUDIT_ENV),
                    help=f"the audit's output directory holding summary.json and scan/ (default ${AUDIT_ENV})")
    ap.add_argument("--lane", type=Path, default=os.environ.get(LANE_ENV),
                    help=f"the directory holding numbers_ledger.tsv and figA_corpus_audit_values.tsv (default ${LANE_ENV})")
    ap.add_argument("--no-lane", action="store_true", help="skip the ledger and value-table cross-checks")
    ap.add_argument("--out", type=Path, default=OUT, help="the sidecar to write or check (default %(default)s)")
    ap.add_argument("--check", action="store_true", help="compare a fresh derivation with --out; write nothing")
    ap.add_argument("--no-scrub", action="store_true", help="skip the publication scan of the new file")
    a = ap.parse_args(argv)
    if a.audit is None or (a.lane is None and not a.no_lane):
        ap.error(f"name the audit directory (--audit or ${AUDIT_ENV}) and the ledger directory (--lane or ${LANE_ENV}, or --no-lane)")
    audit = Path(a.audit).expanduser().resolve()
    lane = None if a.no_lane else Path(a.lane).expanduser().resolve()
    out, problems = derive(audit, lane)
    if problems:
        print("REFUSED: nothing written.\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    text = dump(out)
    if a.check:
        have = a.out.read_text(encoding="utf-8") if a.out.exists() else ""
        if have == text:
            print(f"{shown(a.out)}: equal to a fresh derivation ({len(out['numbers'])} numbers, {len(out['overlap']['sets'])} sets)")
            return 0
        old = json.loads(have) if have else {}
        diff = sorted(k for k in set(old.get("numbers", {})) | set(out["numbers"]) if old.get("numbers", {}).get(k) != out["numbers"].get(k))
        print(f"{shown(a.out)}: DIFFERS from a fresh derivation; numbers differing: {diff[:10] or 'none'}; other fields: "
              f"{sorted(k for k in set(old) | set(out) if k != 'numbers' and old.get(k) != out.get(k))}", file=sys.stderr)
        return 1
    tmp = a.out.with_name("." + a.out.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text, encoding="utf-8")
    if not a.no_scrub:
        try:
            rel = [str(tmp.resolve().relative_to(REPO)), str(Path(__file__).resolve().relative_to(REPO))]
        except ValueError:
            tmp.unlink(missing_ok=True)
            print(f"REFUSED: {tmp} is outside the repository, so the publication scan cannot read it; pass --no-scrub to skip it",
                  file=sys.stderr)
            return 1
        clean, report = scrub_clean(rel)
        if not clean:
            tmp.unlink(missing_ok=True)
            print("REFUSED: the publication scan is not clean on the new file or this script; nothing written.\n" + report, file=sys.stderr)
            return 1
    os.replace(tmp, a.out)
    crossed = ", ".join(f"{Path(c['file']).name} ({c['rows_matched']} rows)" for c in out["provenance"]["cross_checked"]) or "nothing"
    print(f"wrote {shown(a.out)}: {len(out['numbers'])} numbers, {len(out['overlap']['sets'])} scanned sets + "
          f"{len(out['overlap']['expected_sets'])} expected; cross-checked {crossed}" + ("" if a.no_scrub else "; publication scan clean"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
