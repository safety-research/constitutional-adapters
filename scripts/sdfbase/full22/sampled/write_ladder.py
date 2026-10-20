#!/usr/bin/env python
"""full22 SAMPLED re-run: fold a job's at-cap EVENTS into the ladder record (DESIGN.md "Report per cell the rung
at which the raw at-cap count reached zero, or '<top> + residual count' — and, for controls, the
gate's DECISION at every rung and where the pair LEFT the ladder").

The sbatch appends one JSON line per measurement to <ladder dir>/events.jsonl as it goes (so a job that dies
mid-ladder still leaves every count it took), and calls this at the end of each rung / phase to fold them:

  controls events  {"kind": "controls", "cell": C, "control": X, "rung": 512|2048|8192, "n_at_cap": N,
                    "root": <the rung root the count was read from>, "at": ISO}
                   -> ladder.json  {"kind": "controls", "tag", "trial", "rungs": [512, 2048, 8192],
                                    "floor_sample_k": 200, "exit_rule": ..., "cells": {C: {X: {"n_at_cap": {"512": N, ...},
                                    "reached_zero_at": "512" | "2048" | "8192" | "8192 + residual N" | "incomplete",
                                    "residual": N | 0 | null, "rungs_attempted": [...]}}}}
  static events    {"kind": "static", "attack": F, "cell": C, "rung": 4096, "n_at_cap": N, ...}
                   {"kind": "static_escalation", "attack": F, "cell": C, "escalation": <escalation.json> | null,
                    "problem": <str> | null}
                   {"kind": "static_undecided", "attack": F, "cell": C, "rung": 16384,
                    "cause": "gate_refused" | "record_problem", "reason": <str>}   (the
                    ladder's decision at that rung could not be made — the gate refused over the finished store,
                    or the rung's escalation record is missing — so the pair is UNDECIDED there, never still
                    "escalating" from the rung below; folded per rung under the pair's "undecided")
                   -> static_ladder_t<T>.json  {"kind": "static", ..., "rungs": [4096, 8192],
                                    "cells": {C: {F: {"n_at_cap": {"4096": N, "8192": M}, "reached_zero_at": ...,
                                    "escalation": {...} | null, "problem": ...}}}}

EXIT RULE: a (cell, control) / (attack, cell) LEAVES the ladder
when its at-cap count at the PREVIOUS rung is zero — the floor sample (`--floor-sample 200`, a seeded below-cap
sample re-drawn at each rung for the flip-rate noise floor) never keeps a cell on the ladder. "reached_zero_at"
is the FIRST rung whose count is zero; a non-zero count at the top rung is "<top> + residual N" (a genuine
non-termination, recorded as such); a rung whose successor count is missing while the count was > 0 is
"incomplete" (the job did not finish the ladder — resubmit, never infer).

The fold MERGES into an existing ladder file (a later job for a family subset, or a resumed job, adds its
entries without clobbering the others). Usage:
  write_ladder.py --events <events.jsonl> --out <ladder.json> --kind controls|static --tag <TAG> --trial <T>
                  [--rungs 512,2048,8192] [--floor-sample-k 200]

DECISION events (the 16,384 extend under LADDER_RULES.md item_filtered_v1; written by ladder_extend.py):
  per pair   {"kind": controls|static, "cell", "control"|"attack", "rung": 8192, "decision": escalate|exit_le_1pct|
              exit_ceiling, "exit_kind": exit_zero|exit_le_1pct_with_residual|exit_ceiling|null, "n_truncated",
              "n_loop", "n_repetitive", "n_boundary", "n_at_cap", "share_truncated", "share_at_cap", ...}
              -> rec["decisions"][rung] (the three populations beside the count; n_at_cap is NOT re-derived from it)
  per ladder {"kind": controls|static, "rung": 16384, "decision": no_rung|rung|ceiling_stop, "n_pairs_considered",
              "n_promoted", "n_by_exit_kind", "rule", "classifier", ...} (no cell) -> ladder["extend_decisions"]
  hop guard  {"kind": controls|static, "rung": N, "hop_timeout_guard": <hop_timeout_guard.py record>} (no cell)
              -> ladder["hop_timeout_guards"] (the POST-standup records the sbatch folds, so a reader sees the
              transport check covered the rung)
  A pair whose count at rung r is > 0 with no count at the next rung READS ITS DECISION at r: an exit decision
  ends the ladder there ("<r> + residual N (exit_le_1pct)"); an escalate decision with no successor count is
  "incomplete" (the rung is due); no decision -> the count-only rule below, unchanged.

COUNT-ONLY ERA (--count-only-ceiling N, 2026-09-14): the ladders run before the gate existed (the 2026-09-10 events of the
five legacy qwen35_4b_it / gemma4e4b_it cells) carry counts and NO decisions, and they ran the 3-rung ladder whose ceiling
was 8,192. Folded under today's 4-rung list they read "incomplete" (16,384 never attempted, no exit decision) and the pass-2
planner refused whole trial units on them. With the era's ceiling DECLARED by the caller, a pair with no decision at ANY
rung, contiguous counts from the first rung up to exactly N and a count > 0 there reads "<N> + residual n (count-only era
ceiling ...)" — the ladder of its era ended there. Never inferred: a pair with any gate decision (the decision era; a job
that died before its next gate stays "incomplete"), a pair whose top count sits below N, or a pair with a gap in its rungs
is unchanged; without the flag the fold is byte-identical to before. `left_ladder_at` stays {"status": "undecided"} (no
gate ran) — the governing verdict is the gate's, and there was none.

EXTEND KIND (--extend-kind, default = --kind): the GSM8K readout is a controls-SHAPED ladder (8192 -> 16384) with
its own roots, so its events carry "extend_kind": "gsm8k" and fold into their OWN ladder file
(`--kind controls --extend-kind gsm8k --rungs 8192,16384`); a fold skips (and counts) every event whose
extend_kind — "kind" when the event carries none, as every pre-G2 event does — is not its own. The summary's
"exit_kind" tallies each pair's exit kind at its highest decided rung.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RUNGS = {"controls": [512, 2048, 8192, 16384], "static": [4096, 8192, 16384]}   # LADDER_RULES.md: the 16,384 ceiling on both
FLOOR_SAMPLE_K = 200
EXIT_RULE = ("controls: a (cell, control) climbs a rung ONLY on the gate's ESCALATE (ladder_gate.py: ladder_decide's "
             "item-filtered 1% TRUNCATED trigger — loops / repetitive / boundary items never escalate — bound to the "
             "transport approval for the rung above) and leaves it on EXIT exit_le_1pct | exit_ceiling (unresolved at "
             "the 16,384 ceiling) | transport_refused; `reached_zero_at` is the RAW at-cap count's history (the first "
             "rung whose count is zero), never the governing verdict — `left_ladder_at` is. static: the at-cap count. "
             "The floor sample never keeps a cell on the ladder")
#: the decision keys a gate event carries (f22s_gate_event) — folded per rung under `decisions`
DECISION_KEYS = ("decision", "next_cap", "n_truncated", "n_loop", "n_repetitive", "n_boundary", "share_truncated",
                 "transport_ok", "n_items_escalate", "gate")
EXIT_DECISIONS = ("exit_le_1pct", "exit_ceiling", "transport_refused")
#: the note a count-only era pair's verdict carries (see the module doc; summarize() tallies it under `count_only_era_ceiling`)
COUNT_ONLY_ERA_NOTE = "count-only era ceiling: no gate decision at any rung; the pre-decision 3-rung ladder ended here"


def read_events(path: Path) -> list:
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i}: not JSON ({e}) — a torn event line; refuse rather than fold a partial ladder")
    return out



def reached_zero_at(counts: dict, rungs: list, decisions: dict | None = None, count_only_ceiling: int | None = None) -> tuple:
    """-> (verdict, residual): `_reached_zero_at_rule` (unchanged), then the COUNT-ONLY ERA reading when the caller declared
    the era's ceiling: a pair with NO decision at any rung, contiguous counts from the first rung up to exactly
    ``count_only_ceiling`` and n > 0 there left its (3-rung) ladder at that ceiling — "<N> + residual n (<note>)". Any decision,
    a top count below the ceiling, or a gap in the rungs leaves the rule's "incomplete" as it is (never inferred)."""
    verdict, residual = _reached_zero_at_rule(counts, rungs, decisions)
    if verdict == "incomplete" and count_only_ceiling is not None and not (decisions or {}) and counts:
        top = max(counts)
        below = [r for r in rungs if r <= top]
        if top == int(count_only_ceiling) and counts[top] > 0 and below and all(counts.get(r) is not None for r in below):
            return (f"{top} + residual {counts[top]} ({COUNT_ONLY_ERA_NOTE})", counts[top])
    return verdict, residual


def _reached_zero_at_rule(counts: dict, rungs: list, decisions: dict | None = None) -> tuple:
    """-> (verdict, residual). counts: {rung(int): n}. The FIRST rung with n == 0 ends the ladder. With
    ``decisions`` ({rung(int): decision}), a rung whose count is > 0 and whose successor count is absent ends
    the ladder on an EXIT decision ("<r> + residual n (<decision>)"); an ``escalate`` decision there leaves the
    pair incomplete (its next rung is due). Without decisions the count-only rule is byte-identical to before."""
    decisions = decisions or {}
    for i, r in enumerate(rungs):
        n = counts.get(r)
        if n is None:
            return ("incomplete", None) if i > 0 else ("incomplete", None)
        if n == 0:
            return (str(r), 0)
        if i == len(rungs) - 1:
            return (f"{r} + residual {n}", n)
        if counts.get(rungs[i + 1]) is None and decisions.get(r) in EXIT_DECISIONS:
            return (f"{r} + residual {n} ({decisions[r]})", n)
    return ("incomplete", None)


def left_ladder_at(decisions: dict, rungs: list, undecided: dict | None = None) -> dict:
    """The GOVERNING verdict of a controls (cell, control) / static (attack, cell): the rung at which the gate's
    decision was an EXIT (with the exit kind and the counts by class), or ``{"status": "escalating", "from": r}``
    while the last decision is an escalate, or ``{"status": "undecided"}`` when no gate ran yet. Truncation
    unresolved at the ceiling (exit_ceiling) is reported as such — never as "non-termination" (loops are a
    different class).

    ``undecided`` (P2-α) holds the rung-keyed UNDECIDED marks (``static_undecided`` events: the
    gate refused over a finished store, or the rung's escalation record is missing). Walking the rungs in order,
    a decision at a rung governs it (a resumed job whose gate later ruled supersedes an earlier refusal there); a
    mark at a rung with no decision, above every decided rung, makes the pair ``{"status": "undecided", "rung":
    r, "cause", "reason"}`` — the rung was attempted and its exit is undeterminable, so the rung below's ESCALATE
    is history, not an active escalation. A mark below a later decision is stale and ignored."""
    by_rung = {int(r): d for r, d in (decisions or {}).items()}
    marks = {int(r): m for r, m in (undecided or {}).items()}
    if not by_rung and not marks:
        return {"status": "undecided"}
    decided_top = max(by_rung) if by_rung else None
    for r in rungs:
        d = by_rung.get(r)
        if d is None:
            if r in marks and (decided_top is None or r > decided_top):
                return {"status": "undecided", "rung": r, "cause": marks[r].get("cause"), "reason": marks[r].get("reason")}
            continue
        if d["decision"] in EXIT_DECISIONS:
            return {"status": "left", "rung": r, "exit": d["decision"], "n_truncated": d.get("n_truncated"),
                    "n_loop": d.get("n_loop"), "n_repetitive": d.get("n_repetitive"), "n_boundary": d.get("n_boundary"),
                    "unresolved_truncation": (d["decision"] == "exit_ceiling")}
    last = max(by_rung)
    return {"status": "escalating", "from": last, "to": by_rung[last].get("next_cap")}


def fold(events: list, kind: str, rungs: list, existing: dict | None = None, *, tag: str, trial: int,
         floor_sample_k: int = FLOOR_SAMPLE_K, extend_kind: str | None = None, count_only_ceiling: int | None = None) -> dict:
    extend_kind = extend_kind or kind
    lad = dict(existing or {})
    if lad and (lad.get("kind") != kind or lad.get("tag") != tag or int(lad.get("trial", -1)) != int(trial)
                or lad.get("extend_kind", kind) != extend_kind):
        raise SystemExit(f"existing ladder is ({lad.get('kind')}, {lad.get('tag')}, t{lad.get('trial')}, "
                         f"extend_kind {lad.get('extend_kind', lad.get('kind'))}) — refusing to merge ({kind}, {tag}, "
                         f"t{trial}, extend_kind {extend_kind}) into it")
    lad.update({"kind": kind, "tag": tag, "trial": int(trial), "rungs": list(rungs), "extend_kind": extend_kind,
                "exit_rule": EXIT_RULE, "written_at": datetime.now(timezone.utc).isoformat(),
                "count_only_ceiling": (int(count_only_ceiling) if count_only_ceiling is not None else None)})
    if kind == "controls":
        lad["floor_sample_k"] = int(floor_sample_k)
    cells = lad.setdefault("cells", {})
    # per (cell, readout): counts by rung, plus the static escalation record
    skipped_other_kind = 0
    for ev in events:
        k = ev.get("kind")
        if k not in ("controls", "static", "static_escalation", "static_undecided"):
            raise SystemExit(f"unknown event kind {k!r}: {ev}")
        if (k == "controls") != (kind == "controls"):
            # the ladder dir is shared per (target, trial) between the controls job and the static jobs
            # (target_sampled.sbatch: LADDER_DIR), so one events.jsonl legitimately carries both kinds; a
            # fold reads its own kind and counts the rest (canary 84812 refused here, 2026-09-09)
            skipped_other_kind += 1
            continue
        if ev.get("extend_kind", kind) != extend_kind:
            # the GSM8K readout's events (extend_kind gsm8k) share the file with the controls ladder's and
            # fold into their own file; a pre-G2 event carries no extend_kind and reads as its kind
            skipped_other_kind += 1
            continue
        if "decision" in ev and "cell" not in ev:
            # a LADDER-LEVEL extend decision (ladder_extend.py): no pair, no count — recorded whole (it carries the
            # PRE-standup hop guards it enforced, so it is read as a decision BEFORE the hop-guard branch below)
            lad.setdefault("extend_decisions", []).append(dict(ev))
            continue
        if "hop_timeout_guard" in ev and "cell" not in ev:
            # a POST-standup hop_timeout_guard record the sbatch folded at a rung: no pair, no count
            lad.setdefault("hop_timeout_guards", []).append(dict(ev))
            continue
        cell = ev["cell"]
        readout = ev["control"] if k == "controls" else ev["attack"]
        rec = cells.setdefault(cell, {}).setdefault(readout, {"n_at_cap": {}})
        if "decision" in ev:
            # a per-pair DECISION at a rung, in either lane's shape (the 2026-09-11 reconciliation):
            #  * the item_filtered_v1 decision (ladder_extend.py): the three populations beside the count; never a
            #    count event (n_at_cap stays what the counter read)
            #  * the gate event (ladder_gate.py; f22s_gate_event; it carries `gate`, the gate.json path): the BOUND
            #    decision at the rung — and the rung's raw at-cap count, which that event IS (the gate read the
            #    store); its decision vocabulary is the ladder's (escalate | the EXIT_DECISIONS)
            rung = int(ev["rung"])
            if rung not in rungs:
                raise SystemExit(f"event rung {rung} not in the {kind} ladder {rungs}: {ev}")
            # ONE vocabulary for every per-pair decision (ladder_decide.DECISIONS is a subset of it): an unknown word is
            # refused, never folded
            if ev.get("decision") not in ("escalate",) + EXIT_DECISIONS:
                raise SystemExit(f"event carries an unknown ladder decision {ev.get('decision')!r}: {ev}")
            if "rule" not in ev:
                # not lane G2's item_filtered decision (which never moves the count): a gate event, or a count event
                # that also carries the gate's word — its n_at_cap IS the rung's raw count
                if ev.get("n_at_cap") is not None:
                    rec["n_at_cap"][str(rung)] = int(ev["n_at_cap"])
                if ev.get("root"):
                    rec.setdefault("roots", {})[str(rung)] = ev["root"]
            keys = ("decision", "exit_kind", "n_truncated", "n_loop", "n_repetitive", "n_boundary",
                    "n_at_cap", "n_responses", "share_truncated", "share_at_cap", "rule",
                    "classifier_version", "decision_path", "at") + tuple(k2 for k2 in DECISION_KEYS if k2 not in (
                        "decision", "n_truncated", "n_loop", "n_repetitive", "n_boundary", "share_truncated"))
            rec.setdefault("decisions", {})[str(rung)] = {k2: ev.get(k2) for k2 in keys}
            continue
        if k in ("controls", "static"):
            rung = int(ev["rung"])
            if rung not in rungs:
                raise SystemExit(f"event rung {rung} not in the {kind} ladder {rungs}: {ev}")
            rec["n_at_cap"][str(rung)] = int(ev["n_at_cap"])
            if ev.get("root"):
                rec.setdefault("roots", {})[str(rung)] = ev["root"]
        elif k == "static_undecided":               # the rung-keyed UNDECIDED mark (P2-α)
            rung = int(ev["rung"])
            if rung not in rungs:
                raise SystemExit(f"event rung {rung} not in the {kind} ladder {rungs}: {ev}")
            rec.setdefault("undecided", {})[str(rung)] = {"cause": ev.get("cause"), "reason": ev.get("reason"),
                                                          "at": ev.get("at")}
        else:
            esc = ev.get("escalation")
            rec["escalation"] = esc                     # the LATEST escalation record (the historical key)
            rec["problem"] = ev.get("problem")
            if isinstance(esc, dict) and "n_at_cap_after" in esc and "cap_to" in esc:
                rec["n_at_cap"][str(int(esc["cap_to"]))] = int(esc["n_at_cap_after"])
                rec.setdefault("escalations", {})[str(int(esc["cap_to"]))] = esc   # every rung's, by cap_to (round 2 P1)
    for cell, by in cells.items():
        for readout, rec in by.items():
            counts = {int(r): n for r, n in rec["n_at_cap"].items()}
            decisions = {int(r): d.get("decision") for r, d in (rec.get("decisions") or {}).items()}
            verdict, residual = reached_zero_at(counts, rungs, decisions, count_only_ceiling)
            rec["reached_zero_at"] = verdict
            rec["residual"] = residual
            rec["rungs_attempted"] = sorted(counts)
            if rec.get("problem"):
                rec["reached_zero_at"] = "incomplete"
            # the gate's verdict governs both ladders (controls since P1-5; the static 8,192 -> 16,384 rung since round 2 P1)
            # a rung-keyed undecided mark above every decision makes the pair undecided there (round 4, P2-α)
            rec["left_ladder_at"] = left_ladder_at(rec.get("decisions") or {}, rungs, rec.get("undecided"))
    lad["summary"] = summarize(lad)
    lad["skipped_other_kind_events"] = skipped_other_kind
    return lad


EXIT_KINDS = ("exit_zero", "exit_le_1pct_with_residual", "exit_ceiling")


def summarize(lad: dict) -> dict:
    top = str(lad["rungs"][-1])
    out = {"n_pairs": 0, "left_at": {str(r): 0 for r in lad["rungs"]}, "residual_at_top": 0, "incomplete": 0,
           # each pair's exit kind at its HIGHEST decided rung (null = promoted / no decision yet)
           "exit_kind": {**{ek: 0 for ek in EXIT_KINDS}, "promoted": 0, "undecided": 0}}
    # the governing verdict per pair (left_ladder_at: the gate's EXIT kind, escalating, or undecided)
    out["exits"] = {k: 0 for k in EXIT_DECISIONS + ("escalating", "undecided")}
    count_only_era = 0   # tallied under `count_only_era_ceiling` ONLY when the fold enabled that reading (else the summary is as before)
    for by in lad.get("cells", {}).values():
        for rec in by.values():
            out["n_pairs"] += 1
            lla = rec.get("left_ladder_at") or {"status": "undecided"}
            out["exits"][lla["exit"] if lla["status"] == "left" else lla["status"]] += 1
            v = rec.get("reached_zero_at")
            if v == "incomplete":
                out["incomplete"] += 1
            elif v in out["left_at"]:
                out["left_at"][v] += 1
            elif isinstance(v, str) and COUNT_ONLY_ERA_NOTE in v:
                count_only_era += 1
            elif isinstance(v, str) and v.startswith(top + " + residual"):
                out["residual_at_top"] += 1
            decisions = rec.get("decisions") or {}
            if not decisions:
                out["exit_kind"]["undecided"] += 1
            else:
                last = decisions[str(max(int(r) for r in decisions))]
                ek = last.get("exit_kind")
                out["exit_kind"][ek if ek in EXIT_KINDS else "promoted"] += 1
    if lad.get("count_only_ceiling") is not None:
        out["count_only_era_ceiling"] = count_only_era
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--kind", required=True, choices=sorted(RUNGS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trial", required=True, type=int)
    ap.add_argument("--rungs", default=None, help="comma list; default per kind")
    ap.add_argument("--floor-sample-k", type=int, default=FLOOR_SAMPLE_K)
    ap.add_argument("--count-only-ceiling", type=int, default=None,
                    help="the ceiling of the pre-decision (count-only) era's ladder, DECLARED by the caller (controls: 8192): a pair "
                         "with no gate decision at any rung and contiguous counts up to exactly this rung is read as left there with "
                         "its residual instead of 'incomplete'; default: no such reading")
    ap.add_argument("--extend-kind", default=None, choices=("static", "controls", "gsm8k"),
                    help="fold only the events of this extend kind (default: --kind); the GSM8K readout's ladder is "
                         "`--kind controls --extend-kind gsm8k --rungs 8192,16384`")
    args = ap.parse_args(argv)
    rungs = [int(x) for x in args.rungs.split(",")] if args.rungs else RUNGS[args.kind]
    existing = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else None
    events = read_events(args.events) if args.events.is_file() else []
    lad = fold(events, args.kind, rungs, existing, tag=args.tag, trial=args.trial, floor_sample_k=args.floor_sample_k,
               extend_kind=args.extend_kind, count_only_ceiling=args.count_only_ceiling)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(lad, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps({"out": str(args.out), **lad["summary"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
