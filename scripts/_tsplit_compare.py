"""torch-cpu-split GPU-proof compare (stored-reference criterion → twin run →
A/A-controlled twin run).

Three modes, each enforcing its verdict by exit code:

A/A mode (--aa-first / --aa-second) — the DETERMINISM CONTROL.
  Job 112688 (k3) failed the plain twin compare with the same signature as
  the cross-node 112390 failure (238/900 records differing across ALL arms,
  baseline/prompted included; within-env run-to-run spread equal to the
  cross-env spread), i.e. greedy decode is not run-to-run repeatable even
  on one node in one job — kernel reduction-order nondeterminism, which an
  env comparison cannot see past. So the proof now runs the SPLIT cell
  TWICE under determinism knobs (_tsplit_det_wrapper.py) and compares the
  pair first: equality proves the knobs deliver repeatability, so any A/B
  difference that remains is attributable to the env. An A/A mismatch is
  NONDETERMINISM, not an env effect — the exact-equality criterion's
  premise fails independent of the split; the fallback (a statistical
  run-to-run envelope) is never adopted here.

TWIN mode (--twin-split / --twin-prod) — the ENV-PARITY criterion.
  The same eval-only cell, same node, same job, same isolated code, same
  staged inputs — once in the venv built from the PRODUCTION repo's
  lockfile, once in the isolated cu128 split env, both under the same
  determinism knobs. With the A/A control green, BOTH the metrics AND the
  raw per-record generations must be equal; any difference is an effect of
  the env split itself — the exact guarantee the torch-cpu split must not
  break — and fails the proof. Differing records are reported VERBATIM
  (and dumped to twin_record_diff.json beside the split twin's
  results.json): top-severity evidence, never a detail to summarize away.
  --campaign may be given as a DIAGNOSTIC-ONLY reference: the split twin is
  diffed against the stored campaign cell for the record (that comparison
  crosses nodes and time, so it never affects the verdict).

STORED-REFERENCE mode (--scratch / --campaign) — the original criterion,
  kept for the node-pinned re-run (112530): every metric in the scratch
  cell's results.json must EQUAL the stored campaign values. Any mismatch
  is a proof FAILURE to investigate, never to rationalize away (the 61380
  lesson). Records are diagnostic-only here (cross-time on shared infra).

All modes also enforce GPU USE: the nvidia-smi utilization samples taken
while the cell(s) ran must show a max > 0.

Also records the driver version + device name beside the verdict so a
mismatch on a different GPU/driver combo is diagnosable.

Usage (from the proof sbatch):
  # A/A control (after the two knobbed split-cell runs)
  uv run --no-sync python scripts/_tsplit_compare.py \
    --aa-first <splitA out_dir> --aa-second <splitB out_dir> \
    --util-log <nvidia-smi sample log>

  # twin mode (env parity — after the knobbed prod-cell run)
  uv run --no-sync python scripts/_tsplit_compare.py \
    --twin-split <splitA out_dir> --twin-prod <prod out_dir> \
    [--campaign <campaign ifeval root>] --util-log <nvidia-smi sample log>

  # stored-reference mode (node-pinned equality vs the campaign artifact)
  uv run --no-sync python scripts/_tsplit_compare.py \
    --scratch <proof out_dir> --campaign <campaign ifeval root> \
    --util-log <nvidia-smi sample log>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

CELL = Path("json") / "vector" / "results.json"
RECORD_DIFF_NAME = "twin_record_diff.json"
AA_RECORD_DIFF_NAME = "aa_record_diff.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"FATAL: results file missing: {path}")
    return json.loads(path.read_text())


def _diff(a, b, prefix="", a_name="scratch", b_name="campaign"):
    """Recursive diff of two parsed-JSON values; yields human lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                yield f"{prefix}.{k}: missing in {a_name}"
            elif k not in b:
                yield f"{prefix}.{k}: missing in {b_name}"
            else:
                yield from _diff(a[k], b[k], f"{prefix}.{k}", a_name, b_name)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{prefix}: list length {len(a)} != {len(b)}"
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from _diff(x, y, f"{prefix}[{i}]", a_name, b_name)
    elif a != b:
        yield f"{prefix}: {a_name}={a!r} != {b_name}={b!r}"


def _arm_labels(results: dict) -> list[str]:
    """records[i] aligns with arms[i] (both written together by
    _run_ifeval_cell); label each group arm@scale for the report."""
    return [f"{a.get('arm', '?')}@{a.get('scale', '?')}"
            for a in results.get("arms", [])]


def _pair_record_diff(a_recs, b_recs, labels, a_name, b_name):
    """Structured per-record diff between two cells' records blocks.

    Returns (fatal_lines, diff_entries): fatal_lines non-empty on any
    structural mismatch or differing record; diff_entries carry the full
    verbatim evidence for the JSON dump."""
    fatal: list[str] = []
    entries: list[dict] = []
    if a_recs is None or b_recs is None:
        fatal.append("records block missing in "
                     + (a_name if a_recs is None else b_name)
                     + " results.json — the cell always writes it; refusing "
                     "to pass on absent evidence")
        return fatal, entries
    if len(a_recs) != len(b_recs):
        fatal.append(f"records: {len(a_recs)} arm groups in {a_name} "
                     f"!= {len(b_recs)} in {b_name}")
        return fatal, entries
    for i, (ag, bg) in enumerate(zip(a_recs, b_recs)):
        label = labels[i] if i < len(labels) else f"arm[{i}]"
        if len(ag) != len(bg):
            fatal.append(f"records[{label}]: {len(ag)} records in {a_name} "
                         f"!= {len(bg)} in {b_name}")
            continue
        for j, (ar, br) in enumerate(zip(ag, bg)):
            if ar != br:
                fatal.append(f"records[{label}][{j}] differs")
                entries.append({"arm": label, "index": j,
                                a_name: ar, b_name: br})
    return fatal, entries


def _print_record_diff_verbatim(entries, a_name, b_name) -> None:
    """Every differing record, in full — this is the evidence the compare
    exists to catch (an env effect, or knob-defeating nondeterminism).
    Never truncated."""
    for n, e in enumerate(entries, 1):
        a, b = e[a_name], e[b_name]
        print(f"--- record diff {n}/{len(entries)}: "
              f"arm {e['arm']}, record {e['index']} ---")
        print(f"prompt: {a.get('prompt')!r}")
        print(f"[{a_name}] strict={a.get('strict')} loose={a.get('loose')}")
        print(a.get("response"))
        print(f"[{b_name}] strict={b.get('strict')} loose={b.get('loose')}")
        print(b.get("response"))


def _check_gpu_util(util_log: Path) -> None:
    if not util_log.is_file():
        sys.exit(f"FATAL: util log missing: {util_log}")
    samples = [int(m.group(1)) for m in
               re.finditer(r"(\d+)\s*%", util_log.read_text())]
    if not samples:
        sys.exit("FATAL: no GPU-utilization samples parsed from the util log")
    peak = max(samples)
    print(f"[tsplit-compare] GPU util: {len(samples)} samples, max {peak}%")
    if peak <= 0:
        sys.exit("FATAL: peak sampled GPU utilization is 0 -- the cell did "
                 "not exercise the GPU (61380-class failure)")


def _print_gpu_provenance() -> None:
    # Diagnosability record (risk 5): the GPU/driver this verdict was made on.
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True)
        line = smi.stdout.strip() or "UNAVAILABLE"
    except FileNotFoundError:
        line = "UNAVAILABLE"
    print(f"[tsplit-compare] GPU/driver: {line}")


def _campaign_diagnostic(twin: dict, campaign_root: Path) -> None:
    """Non-enforced context: how the split twin sits against the stored
    campaign cell (a cross-node, cross-time comparison — 112390)."""
    campaign = _load(campaign_root / CELL)
    campaign.pop("records", None)
    campaign.pop("skipped", None)
    lines = list(_diff(twin, campaign, "results",
                       a_name="split-twin", b_name="campaign"))
    if lines:
        print(f"[tsplit-compare] DIAGNOSTIC (not enforced): split twin vs "
              f"stored campaign differs in {len(lines)} fields "
              f"(cross-node/cross-time reference):")
        for line in lines[:50]:
            print(f"  {line}")
    else:
        print("[tsplit-compare] DIAGNOSTIC: split twin == stored campaign "
              "metrics (cross-node reference matches too)")


def _compare_pair(a_root: Path, b_root: Path, a_name: str, b_name: str,
                  mismatch_head: str, metric_verdict: str,
                  record_verdict: str, dump_name: str,
                  pass_word: str) -> None:
    """Shared enforcement core: metric equality, then record equality, for
    one pair of cells; fatal with the calling mode's own verdict strings."""
    a = _load(a_root / CELL)
    b = _load(b_root / CELL)
    a_recs = a.pop("records", None)
    b_recs = b.pop("records", None)

    failures = list(_diff(a, b, "results", a_name=a_name, b_name=b_name))
    if failures:
        print(f"[tsplit-compare] {mismatch_head} METRIC MISMATCH "
              f"({len(failures)} fields):")
        for line in failures[:50]:
            print(f"  {line}")
        sys.exit(f"FATAL: {metric_verdict}")
    print(f"[tsplit-compare] {pass_word} metric equality: OK (full summary)")

    fatal, entries = _pair_record_diff(a_recs, b_recs, _arm_labels(a),
                                       a_name, b_name)
    if fatal:
        print(f"[tsplit-compare] {mismatch_head} RECORD MISMATCH "
              f"({len(entries)} differing records):")
        for line in fatal[:100]:
            print(f"  {line}")
        _print_record_diff_verbatim(entries, a_name, b_name)
        dump = a_root / CELL.parent / dump_name
        dump.write_text(json.dumps(entries, indent=1))
        print(f"[tsplit-compare] full record diff written: {dump}")
        sys.exit(f"FATAL: {record_verdict}")
    print(f"[tsplit-compare] {pass_word} record equality: OK "
          f"({sum(len(g) for g in a_recs)} records identical)")


def _run_aa_mode(args) -> None:
    # Same env, same node, same job, same code, same staged inputs, same
    # knobs: ANY difference here is run-to-run nondeterminism the knobs
    # failed to remove — never an env effect.
    _compare_pair(
        args.aa_first, args.aa_second, "splitA", "splitB",
        "A/A",
        ("NONDETERMINISM — A/A split-env twins differ under determinism "
         "knobs (same env, same node, same job): the exact-equality "
         "criterion's premise fails independent of the env split. "
         "Fallback: a statistical criterion (twins within the A/A "
         "run-to-run envelope) — PROPOSE to the coordinator, do not adopt "
         "unilaterally."),
        ("NONDETERMINISM — per-record generations differ between A/A twins "
         "under determinism knobs; see the verbatim diff above. Fallback "
         "criterion needs a coordinator ruling."),
        AA_RECORD_DIFF_NAME, "A/A")
    _check_gpu_util(args.util_log)
    print("[tsplit-compare] PASS: A/A twins equal (metrics + records) — "
          "determinism knobs deliver run-to-run repeatability")


def _run_twin_mode(args) -> None:
    # With the A/A control green, node/job/code/input/nondeterminism are
    # all held fixed; only the venv/lock differs — any difference IS the
    # env effect the split must not have.
    _compare_pair(
        args.twin_split, args.twin_prod, "split", "prod",
        "TWIN",
        "ENV EFFECT — split-env twin metrics != prod-env twin metrics",
        ("ENV EFFECT — per-record generations differ between same-node "
         "same-code twins (top-severity: the split env changes generation; "
         "report, do not redesign around it)"),
        RECORD_DIFF_NAME, "twin")
    if args.campaign is not None:
        split = _load(args.twin_split / CELL)
        split.pop("records", None)
        _campaign_diagnostic(split, args.campaign)
    _check_gpu_util(args.util_log)
    print("[tsplit-compare] PASS: twins equal (metrics + records) + GPU "
          "exercised")


def _run_stored_reference_mode(args) -> None:
    scratch = _load(args.scratch / CELL)
    campaign = _load(args.campaign / CELL)

    s_records = scratch.pop("records", None)
    c_records = campaign.pop("records", None)
    campaign.pop("skipped", None)  # resume-skip marker, never a metric

    failures = list(_diff(scratch, campaign, "results"))
    if failures:
        print(f"[tsplit-compare] METRIC MISMATCH ({len(failures)} fields):")
        for line in failures[:50]:
            print(f"  {line}")
        sys.exit("FATAL: scratch cell metrics != stored campaign metrics")
    print("[tsplit-compare] metric equality: OK "
          "(full summary, records excluded from enforcement)")

    # Diagnostic only (cross-time reproducibility on the same node is
    # informative, but the enforced criterion is metric equality):
    if s_records is not None and c_records is not None:
        same = s_records == c_records
        print(f"[tsplit-compare] raw records identical: {same}")

    _check_gpu_util(args.util_log)
    print("[tsplit-compare] PASS: metrics equal + GPU exercised")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path,
                    help="stored-reference mode: proof out_dir "
                         "(cell at json/vector/)")
    ap.add_argument("--campaign", type=Path,
                    help="campaign ifeval out root (the ENFORCED reference "
                         "in stored-reference mode; DIAGNOSTIC-only in twin "
                         "mode)")
    ap.add_argument("--twin-split", type=Path,
                    help="twin mode: the split-env twin's out_dir")
    ap.add_argument("--twin-prod", type=Path,
                    help="twin mode: the prod-env twin's out_dir")
    ap.add_argument("--aa-first", type=Path,
                    help="A/A mode: the first split-env run's out_dir")
    ap.add_argument("--aa-second", type=Path,
                    help="A/A mode: the second split-env run's out_dir")
    ap.add_argument("--util-log", required=True, type=Path)
    args = ap.parse_args()

    twin = args.twin_split is not None or args.twin_prod is not None
    aa = args.aa_first is not None or args.aa_second is not None
    if aa and (args.aa_first is None or args.aa_second is None):
        ap.error("A/A mode needs BOTH --aa-first and --aa-second")
    if aa and (twin or args.scratch is not None or args.campaign is not None):
        ap.error("--aa-* is A/A mode; it cannot combine with --twin-* or "
                 "--scratch/--campaign")
    if twin and (args.twin_split is None or args.twin_prod is None):
        ap.error("twin mode needs BOTH --twin-split and --twin-prod")
    if twin and args.scratch is not None:
        ap.error("--scratch is stored-reference mode; it cannot combine "
                 "with --twin-*")
    if not twin and not aa and (args.scratch is None
                                or args.campaign is None):
        ap.error("give --aa-first/--aa-second (A/A mode), "
                 "--twin-split/--twin-prod (twin mode), or "
                 "--scratch/--campaign (stored-reference mode)")

    _print_gpu_provenance()
    if aa:
        _run_aa_mode(args)
    elif twin:
        _run_twin_mode(args)
    else:
        _run_stored_reference_mode(args)


if __name__ == "__main__":
    main()
