#!/usr/bin/env python
"""full22 SAMPLED re-run — the 16,384 EXTEND decision and the CEILING report (LADDER_RULES.md §1, §5;
DESIGN.md).

Reads the ``reclassify.json`` summaries ``ladder_decide.py reclassify`` wrote over the CLUSTER roots (static and
controls — and the GSM8K readout roots when present) and, per (cell, readout) at ``--cap``:

  * ``decide --cap 8192 --next-cap 16384``: whether a rung at ``--next-cap`` is due. Every summary must be
    COMPLETE (``complete: true``; a refused store is a refusal here, exit 2); every record must carry
    ``rule == item_filtered_v1`` and ``classifier.version == item_filtered_v1`` — the summary, every
    ``decision.json`` at ``--cap``, and every disposition header (an extend pass never acts on a decision made
    under another rule); the ladder must BE at ``--cap`` for every pair still climbing (an entry that decided
    ``escalate`` at a lower rung with no decided entry at its ``next_cap`` means the previous rung has not been
    generated — refused, never skipped); the PROMOTED set is the pairs whose decision at ``--cap`` is
    ``escalate`` (``next_cap == --next-cap``).
  * ``decide --cap 16384`` (no ``--next-cap``): the CEILING report. Nothing escalates past 16,384: every
    pair's residual SHARE and COUNT at the ceiling are recorded and the ladder stops; a decision of ``escalate``
    at the ceiling is impossible by construction and refused if seen.

EXIT KINDS (the ceiling rule — cells that left on the 1% threshold with a residual are recorded
SEPARATELY from cells that reached the ceiling), per pair as ``exit_kind``:
  ``exit_zero``                  decision exit_le_1pct with NO at-cap item left (the ladder closed clean)
  ``exit_le_1pct_with_residual`` decision exit_le_1pct with at-cap items left (loops / repetitive / boundary /
                                 a truncated share at or under 1%) — listed separately
  ``exit_ceiling``               decision exit_ceiling: at 16,384 with a truncated share still above 1%
  ``null``                       promoted (decision escalate): the next rung is due

HOP-TIMEOUT GUARD (hop_timeout_guard.py): every ``--hop-guard`` record given is ENFORCED, not merely carried — a
record whose ``verdict`` is not ``PASS`` or whose ``rung`` is not ``--next-cap`` refuses the whole decision (exit 2),
so no promotion is written under a transport that cannot carry the rung. These are the PRE-standup records; the
sbatch's POST-standup ``--endpoint`` run is mandatory before any request.

Writes ``--out`` (``decision_16384.json`` / ``ceiling_16384.json``): kind, tag, trial, cap_from / cap_to, rule,
classifier stamp, the tools' shas, the rep_ratio source sha, n_pairs_considered, n_promoted, n_by_exit_kind, the
two separate lists (``exit_le_1pct_with_residual``, ``exit_ceiling``), per-kind counts, and per pair the THREE
populations at ``--cap`` — ``n_truncated`` (the escalation set), ``n_loop`` (by rule), ``n_repetitive``,
``n_boundary`` — beside ``n_at_cap`` / ``n_responses`` / ``share_truncated`` / ``share_at_cap``, the decision,
``exit_kind``, the items (disposition) path + sha256, and the decision.json path. Appends to ``--events`` (the
ladder's events.jsonl; write_ladder.py folds them): ONE per-pair decision event at ``--cap`` and ONE ladder-level
event per kind (``decision: rung | no_rung`` at ``--next-cap``, or ``ceiling_stop`` at the ceiling), each stamped
``extend_kind`` (static | controls | gsm8k) so the GSM8K readout's events fold into its own ladder file.

    ladder_extend.py decide --cap 8192 --next-cap 16384 --tag TAG --trial T \\
        --reclassify static=<dir>/reclassify.json --reclassify controls=<dir>/reclassify.json \\
        [--reclassify gsm8k=<...>] [--hop-guard <record.json>]... --out decision_16384.json [--events events.jsonl]
    ladder_extend.py decide --cap 16384 --tag TAG --trial T --reclassify ... --out ceiling_16384.json [--events ...]
    ladder_extend.py list --decision decision_16384.json --kind static|controls|gsm8k
        -> one TAB-separated line per PROMOTED pair: static `attack<TAB>cell<TAB>items_path`;
           controls / gsm8k `cell<TAB>control<TAB>items_path` (nothing printed when none promoted)

Exit 0 on a valid decision (zero promotions included — the record is the deliverable); exit 2 on a refusal.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve()
EXIT_REFUSED = 2
RULE = "item_filtered_v1"
CEILING = 16384
#: the extend kinds — NO gsm8k_greedy: there is no matched greedy GSM8K arm
KINDS = ("static", "controls", "gsm8k")
#: the reclassify kind each extend kind is produced by (the GSM8K roots are controls-shaped stores)
RECLASSIFY_KIND = {"static": "static", "controls": "controls", "gsm8k": "controls"}
ITEMS_HEADER_KEY = "__ladder_items__"
#: the population counts carried per pair (LADDER_RULES.md §1 "Report at every rung")
COUNT_KEYS = ("n_responses", "n_at_cap", "n_loop", "n_loop_by_rule", "n_repetitive", "n_boundary",
              "n_boundary_any", "n_truncated", "share_truncated", "share_at_cap")
EXIT_KINDS = ("exit_zero", "exit_le_1pct_with_residual", "exit_ceiling")


class Refusal(Exception):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path: Path, what: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise Refusal(f"{what}: {p} does not exist")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Refusal(f"{what}: {p} is not JSON ({exc})")


def _assert_rule(rec: dict, where: str) -> None:
    if rec.get("rule") != RULE:
        raise Refusal(f"{where}: rule={rec.get('rule')!r} != {RULE!r} — an extend pass never acts on a decision "
                      "made under another rule")
    clf = rec.get("classifier")
    if not isinstance(clf, dict) or clf.get("version") != RULE:
        raise Refusal(f"{where}: classifier.version={None if not isinstance(clf, dict) else clf.get('version')!r} "
                      f"!= {RULE!r}")


def exit_kind_of(decision: str, n_at_cap: int) -> Optional[str]:
    """The three exit kinds; None for a promoted pair."""
    if decision == "escalate":
        return None
    if decision == "exit_ceiling":
        return "exit_ceiling"
    if decision == "exit_le_1pct":
        return "exit_le_1pct_with_residual" if int(n_at_cap) > 0 else "exit_zero"
    raise Refusal(f"unknown decision {decision!r}")


def _identity(kind: str, ent: dict) -> tuple:
    if RECLASSIFY_KIND[kind] == "static":
        return (ent.get("attack"), ent.get("defense"))
    return (ent.get("cell"), ent.get("control"))


def _identity_dict(kind: str, ent: dict) -> dict:
    if RECLASSIFY_KIND[kind] == "static":
        return {"attack": ent.get("attack"), "cell": ent.get("defense"), "private_run": ent.get("private_run")}
    return {"cell": ent.get("cell"), "control": ent.get("control"), "out_root": ent.get("out_root")}


def _consider(kind: str, summary: dict, path: Path, *, cap: int, next_cap: Optional[int]) -> List[dict]:
    """The pairs of ONE reclassify summary at ``cap`` — validated, with their decision records read."""
    where = f"reclassify[{kind}] {path}"
    _assert_rule(summary, where)
    if summary.get("kind") != RECLASSIFY_KIND[kind]:
        raise Refusal(f"{where}: kind={summary.get('kind')!r} but the extend kind {kind!r} needs a "
                      f"{RECLASSIFY_KIND[kind]!r} reclassify")
    if summary.get("complete") is not True:
        refused = [e for e in summary.get("entries", []) if e.get("status") == "refused"]
        raise Refusal(f"{where}: complete={summary.get('complete')!r} (n_refused={summary.get('n_refused')}); "
                      f"first refusals: {[(e.get('cap'), _identity(kind, e), str(e.get('reason'))[:160]) for e in refused[:3]]}")
    ladder = [int(x) for x in summary.get("ladder", [])]
    if cap not in ladder:
        raise Refusal(f"{where}: ladder {ladder} has no rung {cap}")
    if next_cap is None:
        if cap != ladder[-1]:
            raise Refusal(f"{where}: --cap {cap} is not the ceiling {ladder[-1]} of ladder {ladder}; without --next-cap "
                          "this is the CEILING report — give --next-cap for a climb")
        if ladder[-1] != CEILING:
            raise Refusal(f"{where}: the ladder's ceiling {ladder[-1]} != {CEILING} (S-D4: nothing escalates past 16,384)")
    elif next_cap not in ladder or ladder.index(next_cap) != ladder.index(cap) + 1:
        raise Refusal(f"{where}: ladder {ladder} does not step {cap} -> {next_cap}")
    if next_cap is not None and next_cap > CEILING:
        raise Refusal(f"--next-cap {next_cap} is past the ceiling {CEILING} (S-D4: nothing escalates past 16,384)")
    entries = [e for e in summary.get("entries", []) if e.get("status") == "decided"]
    # the ladder must BE at `cap` for every pair still climbing below it
    by_id_cap = {(_identity(kind, e), int(e["cap"])): e for e in entries}
    for e in entries:
        if int(e["cap"]) < cap and e.get("decision") == "escalate":
            nxt = e.get("next_cap")
            if nxt is None or (_identity(kind, e), int(nxt)) not in by_id_cap:
                raise Refusal(f"{where}: {_identity(kind, e)} decided escalate at cap {e['cap']} but its rung "
                              f"{nxt} has no decided entry — the ladder is not at {cap} yet for this pair; "
                              "generate (or finish) that rung before extending")
    pairs = []
    for e in entries:
        if int(e["cap"]) != cap:
            continue
        dpath = Path(e["decision_path"])
        rec = _load(dpath, f"{where} decision")
        _assert_rule(rec, f"decision {dpath}")
        if int(rec.get("cap", -1)) != cap:
            raise Refusal(f"decision {dpath}: cap={rec.get('cap')} != {cap}")
        if rec.get("decision") != e.get("decision"):
            raise Refusal(f"decision {dpath}: decision {rec.get('decision')!r} != the summary's {e.get('decision')!r}")
        counts = rec.get("counts") or {}
        missing = [k for k in ("n_responses", "n_at_cap", "n_loop", "n_repetitive", "n_boundary", "n_truncated",
                               "share_truncated", "share_at_cap") if k not in counts]
        if missing:
            raise Refusal(f"decision {dpath}: counts lack {missing}")
        promoted = rec["decision"] == "escalate"
        if promoted and next_cap is None:
            raise Refusal(f"decision {dpath}: escalate at the ceiling {cap} — impossible under S-D4 (nothing escalates "
                          "past 16,384); the decision record is not a ceiling decision")
        items_path = rec.get("items_out") or e.get("items_path")
        items_sha = None
        if promoted:
            if int(rec.get("next_cap") or -1) != next_cap:
                raise Refusal(f"decision {dpath}: escalate with next_cap={rec.get('next_cap')} != {next_cap}")
            if not items_path or not Path(items_path).is_file():
                raise Refusal(f"decision {dpath}: escalate but the disposition file {items_path!r} is missing")
            head_line = Path(items_path).read_text(encoding="utf-8").splitlines()[0]
            head = json.loads(head_line).get(ITEMS_HEADER_KEY) or {}
            if head.get("rule") != RULE or int(head.get("cap", -1)) != cap:
                raise Refusal(f"disposition {items_path}: header rule={head.get('rule')!r} cap={head.get('cap')!r} "
                              f"(need {RULE!r} at {cap})")
            if int(head.get("n_escalate", -1)) != int(counts["n_truncated"]):
                raise Refusal(f"disposition {items_path}: n_escalate={head.get('n_escalate')} != n_truncated "
                              f"{counts['n_truncated']}")
            items_sha = _sha(Path(items_path))
        pairs.append({"kind": kind, **_identity_dict(kind, e), "cap": cap,
                      "decision": rec["decision"], "promoted": promoted, "next_cap": rec.get("next_cap"),
                      "exit_kind": exit_kind_of(rec["decision"], counts["n_at_cap"]),
                      "counts": {k: counts.get(k) for k in COUNT_KEYS},
                      "residual_count": int(counts["n_at_cap"]), "residual_share": counts["share_at_cap"],
                      "residual": rec.get("residual"), "sibling_status": (rec.get("sibling") or {}).get("status"),
                      "decision_path": str(dpath), "decision_sha256": _sha(dpath),
                      "items_path": str(items_path) if items_path else None, "items_sha256": items_sha})
    return pairs


def _enforce_hop_guards(hop_guards: List[Path], *, next_cap: Optional[int]) -> List[dict]:
    """Every --hop-guard record must be a PASS at the rung about to be generated; anything else refuses the
    decision (a promotion under a transport that cannot carry the rung would censor the slow leg)."""
    out = []
    for g in hop_guards:
        rec = _load(g, "hop guard")
        verdict = rec.get("verdict")
        if verdict != "PASS":
            raise Refusal(f"hop guard {g}: verdict={verdict!r} ({str(rec.get('reason'))[:300]}) — no rung is promoted "
                          "under a transport that cannot carry it")
        if next_cap is not None and int(rec.get("rung", -1)) != int(next_cap):
            raise Refusal(f"hop guard {g}: rung={rec.get('rung')} != the rung about to be generated {next_cap}")
        out.append({"path": str(g), "sha256": _sha(g), **{k: rec.get(k) for k in (
            "rung", "phase", "client", "verdict", "binding_hop", "binding_s", "required_s", "upstream_timeout_s",
            "upstream_key", "client_timeout_s", "floor_tok_s", "raw_min_tok_s", "margin", "sole_check")}})
    return out


def _pair_summary(p: dict) -> dict:
    return {k: p.get(k) for k in ("kind", "attack", "cell", "control", "cap", "decision", "exit_kind",
                                  "residual_count", "residual_share")} | {
        "n_truncated": p["counts"].get("n_truncated"), "share_truncated": p["counts"].get("share_truncated"),
        "n_loop": p["counts"].get("n_loop"), "n_repetitive": p["counts"].get("n_repetitive"),
        "n_boundary": p["counts"].get("n_boundary"), "n_responses": p["counts"].get("n_responses")}


def decide(*, cap: int, next_cap: Optional[int], tag: str, trial: int, reclassify: Dict[str, Path],
           hop_guards: List[Path], out: Path, events: Optional[Path], no_controls_leg: Optional[str] = None) -> dict:
    if not reclassify:
        raise Refusal("no --reclassify given (static=<reclassify.json> and controls=<reclassify.json> at least)")
    if no_controls_leg is not None and not no_controls_leg.strip():
        raise Refusal("--no-controls-leg needs the DECLARED reason (e.g. the roster line that gives the view no controls leg)")
    if no_controls_leg is not None and "controls" in reclassify:
        raise Refusal("--no-controls-leg given but a controls reclassify was also given — a view has a controls leg or it does not")
    for k in ("static", "controls"):
        if k not in reclassify and not (k == "controls" and no_controls_leg is not None):
            raise Refusal(f"--reclassify {k}=<reclassify.json> is required (the extend decides both kinds; a view whose roster "
                          "gives it NO controls leg passes --no-controls-leg <declared reason> instead)")
    if next_cap is not None and next_cap <= cap:
        raise Refusal(f"--next-cap {next_cap} must exceed --cap {cap}")
    at_ceiling = next_cap is None
    summaries = {k: _load(p, f"reclassify[{k}]") for k, p in reclassify.items()}
    pairs: List[dict] = []
    by_kind: Dict[str, dict] = {}
    classifiers = []
    tool_shas = []
    rep_shas = set()
    for kind, summ in summaries.items():
        ps = _consider(kind, summ, Path(reclassify[kind]), cap=cap, next_cap=next_cap)
        pairs.extend(ps)
        by_kind[kind] = {"n_considered": len(ps), "n_promoted": sum(p["promoted"] for p in ps),
                         "n_by_exit_kind": {ek: sum(1 for p in ps if p["exit_kind"] == ek) for ek in EXIT_KINDS},
                         "rungs_present": summ.get("rungs_present"), "rungs_absent": summ.get("rungs_absent"),
                         "reclassify_path": str(reclassify[kind]), "reclassify_sha256": _sha(Path(reclassify[kind])),
                         "n_entries": summ.get("n_entries"), "n_by_decision": summ.get("n_by_decision")}
        classifiers.append(json.dumps(summ.get("classifier"), sort_keys=True))
        tool_shas.append(((summ.get("tool") or {}).get("sha256"), (summ.get("tool") or {}).get("git_sha")))
        rs = (summ.get("rep_ratio") or {}).get("source_sha256")
        if rs:
            rep_shas.add(rs)
    if len(set(classifiers)) != 1:
        raise Refusal(f"the reclassify summaries carry different classifier stamps: {classifiers}")
    if len(rep_shas) != 1:
        raise Refusal(f"the reclassify summaries name different rep_ratio sources: {sorted(rep_shas)}")
    guards = _enforce_hop_guards(hop_guards, next_cap=next_cap)
    n_promoted = sum(p["promoted"] for p in pairs)
    if at_ceiling and n_promoted:
        raise Refusal("promotions at the ceiling — impossible under S-D4")
    if not at_ceiling and not hop_guards:
        raise Refusal("a climb needs at least one --hop-guard record (the PRE-standup hop_timeout_guard.py PASS for "
                      f"rung {next_cap}); absent evidence is not evidence of adequacy")
    decision = "ceiling_stop" if at_ceiling else ("rung" if n_promoted else "no_rung")
    rec = {
        "kind": "ladder_extend", "tag": tag, "trial": int(trial), "cap_from": int(cap),
        "cap_to": None if at_ceiling else int(next_cap), "at_ceiling": at_ceiling, "ceiling": CEILING,
        "never_past_ceiling": True,
        "rule": RULE, "classifier": json.loads(classifiers[0]),
        "tools": {"ladder_decide": [{"sha256": s, "git_sha": g} for s, g in sorted(set(tool_shas), key=str)],
                  "ladder_extend": {"name": HERE.name, "path": str(HERE), "sha256": _sha(HERE)}},
        "rep_ratio_source_sha256": rep_shas.pop(),
        "n_pairs_considered": len(pairs), "n_promoted": n_promoted,
        "n_by_exit_kind": {ek: sum(1 for p in pairs if p["exit_kind"] == ek) for ek in EXIT_KINDS},
        "decision": decision,
        "by_kind": by_kind, "absent_kinds": [k for k in KINDS if k not in reclassify],
        # a view with NO controls leg (the roster's, e.g. llama8b_fullconst) decides on static alone, and says so
        "controls_leg": ({"present": False, "declared_absent": no_controls_leg} if no_controls_leg is not None
                         else {"present": True}),
        "pairs": pairs, "promoted": [p for p in pairs if p["promoted"]],
        # the two populations listed SEPARATELY, each with its residual share and count
        "exit_le_1pct_with_residual": [_pair_summary(p) for p in pairs if p["exit_kind"] == "exit_le_1pct_with_residual"],
        "exit_ceiling": [_pair_summary(p) for p in pairs if p["exit_kind"] == "exit_ceiling"],
        "exit_zero": [_pair_summary(p) for p in pairs if p["exit_kind"] == "exit_zero"],
        "hop_timeout_guard": guards, "hop_timeout_guard_enforced": True,
        "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(out)
    if events is not None:
        now = rec["written_at"]
        lines = []
        for p in pairs:
            ev = {"kind": RECLASSIFY_KIND[p["kind"]], "cell": p["cell"], "rung": cap, "decision": p["decision"],
                  "exit_kind": p["exit_kind"], "extend_kind": p["kind"], "rule": RULE, "classifier_version": RULE,
                  **{k: p["counts"].get(k) for k in ("n_at_cap", "n_truncated", "n_loop", "n_repetitive",
                                                     "n_boundary", "n_responses", "share_truncated", "share_at_cap")},
                  "decision_path": p["decision_path"], "at": now}
            if RECLASSIFY_KIND[p["kind"]] == "static":
                ev["attack"] = p["attack"]
            else:
                ev["control"] = p["control"]
            lines.append(ev)
        for kind, bk in by_kind.items():
            lines.append({"kind": RECLASSIFY_KIND[kind], "extend_kind": kind,
                          "rung": cap if at_ceiling else next_cap,
                          "decision": "ceiling_stop" if at_ceiling else ("rung" if bk["n_promoted"] else "no_rung"),
                          "n_pairs_considered": bk["n_considered"], "n_promoted": bk["n_promoted"],
                          "n_by_exit_kind": bk["n_by_exit_kind"], "ceiling": CEILING,
                          "rule": RULE, "classifier": rec["classifier"],
                          "tool_sha256": rec["tools"]["ladder_extend"]["sha256"],
                          "rep_ratio_source_sha256": rec["rep_ratio_source_sha256"],
                          "hop_timeout_guard": guards,
                          "decision_path": str(out), "at": now})
        ev_path = Path(events)
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        with ev_path.open("a", encoding="utf-8") as f:
            for ev in lines:
                f.write(json.dumps(ev, sort_keys=True, default=str) + "\n")
    return rec


def cmd_decide(args) -> int:
    reclassify: Dict[str, Path] = {}
    for spec in args.reclassify or []:
        kind, _, path = spec.partition("=")
        if kind not in KINDS or not path:
            raise Refusal(f"--reclassify {spec!r}: expected <kind>=<reclassify.json> with kind in {KINDS}")
        if kind in reclassify:
            raise Refusal(f"--reclassify {kind} given twice")
        reclassify[kind] = Path(path)
    rec = decide(cap=args.cap, next_cap=args.next_cap, tag=args.tag, trial=args.trial, reclassify=reclassify,
                 hop_guards=[Path(g) for g in (args.hop_guard or [])], out=Path(args.out),
                 events=Path(args.events) if args.events else None, no_controls_leg=args.no_controls_leg)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("pairs", "promoted", "classifier", "hop_timeout_guard",
                                                                 "exit_le_1pct_with_residual", "exit_ceiling", "exit_zero")},
                     sort_keys=True, default=str), flush=True)
    return 0


def cmd_list(args) -> int:
    rec = _load(Path(args.decision), "decision")
    if rec.get("kind") != "ladder_extend" or rec.get("rule") != RULE:
        raise Refusal(f"{args.decision}: not a ladder_extend decision under {RULE}")
    for p in rec.get("promoted", []):
        if p.get("kind") != args.kind:
            continue
        if RECLASSIFY_KIND[args.kind] == "static":
            print(f"{p['attack']}\t{p['cell']}\t{p['items_path']}")
        else:
            print(f"{p['cell']}\t{p['control']}\t{p['items_path']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--cap", type=int, required=True)
    d.add_argument("--next-cap", type=int, default=None,
                   help="the rung to climb to; ABSENT = the CEILING report at --cap (must be 16384; S-D4)")
    d.add_argument("--tag", required=True)
    d.add_argument("--trial", type=int, required=True)
    d.add_argument("--reclassify", action="append", metavar="KIND=PATH")
    d.add_argument("--hop-guard", action="append", metavar="RECORD",
                   help="hop_timeout_guard.py records — ENFORCED: each must be a PASS at --next-cap")
    d.add_argument("--no-controls-leg", metavar="REASON", default=None,
                   help="the view has NO controls leg by the roster (e.g. llama8b_fullconst, D19): decide on static alone and "
                        "record the declared reason; refused together with a controls reclassify")
    d.add_argument("--out", required=True)
    d.add_argument("--events", default=None)
    l = sub.add_parser("list")
    l.add_argument("--decision", required=True)
    l.add_argument("--kind", required=True, choices=KINDS)
    args = ap.parse_args(argv)
    try:
        return {"decide": cmd_decide, "list": cmd_list}[args.cmd](args)
    except Refusal as exc:
        print(f"[ladder-extend] REFUSED: {exc}", file=sys.stderr, flush=True)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
