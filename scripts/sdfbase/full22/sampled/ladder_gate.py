#!/usr/bin/env python
"""ladder_gate.py — the ONE BOUND DECISION that governs the controls escalation loop of target_sampled.sbatch.

Review round 1 (2026-09-10, P1-5): the loop escalated a (cell, control) on ``n_at_cap > 0`` (every at-cap item,
loops included), knew only the 2048 / 8192 rungs, filtered no items, consulted no transport verdict, and labelled
an 8192 residual "genuine non-termination". The cap-ladder verdict (ladder_decide.py: the 1 % TRUNCATED trigger
with loops / repetitive / boundary items excluded, the item-filtered escalation set, the 16,384 ceiling) is now the
thing the loop reads, and it reads it HERE, bound together with the transport approval for the rung above:

    ladder_gate.py decide --out-root ROOT --cell CELL --control C --cap N --endpoint <serving>/endpoint.json
                          [--driver-timeout-s 7200] [--sibling-out-root ROOT_OTHER_TRIAL]
                          [--rep-ratio-source PATH] --out gate.json --decision-out decision.json --items-out items.jsonl
        -> stdout ONE line:  ESCALATE <next_cap> <items.jsonl>     (climb: exactly these items, at that cap)
                             EXIT <exit_le_1pct|exit_ceiling|transport_refused>
           exit 0 on a bound decision; exit 2 on a REFUSAL (a refusal is never a decision — the loop stops)
    ladder_gate.py static --private-root ROOT --private-run RUN --attack FAM --defense CELL --cap N --endpoint ...
                          [--sibling-private-run RUN_OTHER_TRIAL] --out gate.json --decision-out ... --items-out ...
        -> the same one line, for a sampled static-family store (the static 8,192 -> 16,384 rung;
           the items file is escalate_static.py's --items)
    ladder_gate.py transport --endpoint <new serving>/endpoint.json --next-cap N [--driver-timeout-s 7200]
        -> "OK ..." exit 0 | "REFUSED ..." exit 3: the rung's standup recorded a proxy hop too small for the rung

The transport verdict is ``cap_ladder.transport_check`` over ``cap_ladder.served_hop_timeouts`` — the driver's
client timeout and the proxy's upstream timeout AS THE STANDUP RECORDED IT (``upstream_timeout_s`` in
endpoint.json), never the proxy's compiled default (the 600 s default censored every cap-bound reply, memory
2026-09-08); an unrecorded hop is undeterminable and refuses the climb. The loop checks it twice: at the decision
(the previous rung's record — the standup config the next rung inherits) and again against the NEW rung's record
before any driver is launched (the ``transport`` subcommand).

The gate's record (``--out``) binds: the decision-tool record's path + sha, the counts by CLASS (n_loop,
n_repetitive, n_boundary, n_truncated — loops and boundary noise stay distinct from unresolved truncation at
every rung), the transport verdict, the items file the driver's ``--items`` consumes, and the driver's own
``cap_ladder.json`` reading of the same store when present (recorded beside, never a second decider).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "src"))

SCHEMA = "ladder_gate/1"
EXIT_REFUSED = 2
EXIT_TRANSPORT = 3
#: the in-repo copy of the artifact's loop statistic (ladder_decide.CLASSIFIER names this file as the classifier's
#: source); on a cluster node the grading host artifact file is not beside the repo, this one always is
REP_RATIO_SOURCE_DEFAULT = REPO / "src" / "sisteer" / "capeval" / "tiers" / "cap_ladder.py"
#: the loop-classifier source ladder_decide reads (the harness's own degeneracy.py, by path)
DEGENERACY_SOURCE_DEFAULT = REPO / "src" / "sisteer" / "eval" / "degeneracy.py"
#: the served driver's aiohttp total timeout (served_capability_run.AIOHTTP_TOTAL_TIMEOUT_S) — read from the
#: driver when importable, so the gate never carries a second copy of the number
DRIVER_TIMEOUT_FALLBACK_S = 7200.0
TRANSPORT_REFUSED = "transport_refused"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def ladder_decide():
    return _load("_ladder_gate_decide", HERE.with_name("ladder_decide.py"))


def driver_timeout_default() -> float:
    src = REPO / "scripts" / "served_capability_run.py"
    try:
        text = src.read_text()
    except OSError:
        return DRIVER_TIMEOUT_FALLBACK_S
    for line in text.splitlines():
        if line.startswith("AIOHTTP_TOTAL_TIMEOUT_S"):
            rhs = line.split("=", 1)[1].strip().split()[0]
            try:
                return float(rhs)
            except ValueError:
                pass
            # main's driver binds it to the proxy engine's constant (`pe.AIOHTTP_TOTAL_TIMEOUT_S`, proxy_engine.py): the
            # value the driver's client actually uses is that module's — read it there, never guess a literal
            if rhs.endswith("AIOHTTP_TOTAL_TIMEOUT_S"):
                sys.path.insert(0, str(REPO / "src"))
                from sisteer.capeval.serving import proxy_engine as pe
                return float(pe.AIOHTTP_TOTAL_TIMEOUT_S)
            raise SystemExit(f"[ladder-gate] cannot read the driver's AIOHTTP_TOTAL_TIMEOUT_S from {src}: {line!r}")
    return DRIVER_TIMEOUT_FALLBACK_S


def read_endpoint(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def driver_record(store_dir: Path, control: str) -> Optional[dict]:
    """The served driver's own cap_ladder.json verdict for this control at this store — recorded beside the
    decision (two instruments, one seam), never a second decider."""
    from sisteer.capeval.tiers import cap_ladder as cl
    p = Path(store_dir) / cl.RECORD_NAME
    if not p.is_file():
        return {"present": False, "path": str(p)}
    try:
        rec = json.loads(p.read_text())
        s = (rec.get("controls") or {}).get(control)
    except Exception as e:  # noqa: BLE001 — recorded, never swallowed
        return {"present": True, "path": str(p), "unreadable": f"{type(e).__name__}: {e}"}
    if not s:
        return {"present": True, "path": str(p), "control_recorded": False}
    return {"present": True, "path": str(p), "control_recorded": True,
            "exit": s.get("exit"), "next_cap": s.get("next_cap"), "n_truncated": s.get("n_truncated"),
            "n_non_terminating": s.get("n_non_terminating"), "n_boundary": s.get("n_boundary"),
            "transport": (s.get("transport") or {}).get("ok") if s.get("transport") else None}


def _thresholds(ld) -> dict:
    return {"trigger_share": ld.TRIGGER_SHARE, "loop_rep_ratio_threshold": ld.LOOP_REP_RATIO_THRESHOLD,
            "repetitive_rep_ratio_min": ld.REPETITIVE_REP_RATIO_MIN, "tail_words": ld.TAIL_WORDS,
            "boundary_fraction": ld.BOUNDARY_FRACTION}


def decide(args) -> int:
    """``decide`` (controls) and ``static``: ONE bound decision for one (cell, readout) at the rung just measured."""
    from sisteer.capeval.tiers import cap_ladder as cl
    ld = ladder_decide()
    argv = list(sys.argv[1:])
    kind = "static" if args.mode == "static" else "controls"
    driver_s = float(args.driver_timeout_s) if args.driver_timeout_s is not None else driver_timeout_default()
    try:
        rep_ratio, rep_info = ld.extract_rep_ratio(Path(args.rep_ratio_source))
        degeneracy, deg_info = ld.load_degeneracy(Path(args.degeneracy_source))
        common = dict(cap=args.cap, thresholds=_thresholds(ld), rep_ratio=rep_ratio, rep_info=rep_info,
                      degeneracy=degeneracy, deg_info=deg_info, out=Path(args.decision_out),
                      items_out=Path(args.items_out), argv=argv, stdout=False)
        if kind == "static":
            rec = ld.decide_static(private_root=Path(args.private_root), private_run=args.private_run,
                                   attack=args.attack, defense=args.defense, sibling_run=args.sibling_private_run,
                                   **common)
        else:
            rec = ld.decide_controls(out_root=Path(args.out_root), cell=args.cell, control=args.control,
                                     sibling_root=args.sibling_out_root, **common)
    except ld.Refusal as exc:
        print(f"[ladder-gate] REFUSED: {exc}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    if rec["rule"] != ld.RULE:
        print(f"[ladder-gate] REFUSED: decision rule {rec['rule']!r} != {ld.RULE!r}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    decision, next_cap = rec["decision"], rec["next_cap"]
    endpoint = read_endpoint(args.endpoint)
    hops = cl.served_hop_timeouts(endpoint, driver_s) if endpoint is not None else None
    transport = None
    if decision == "escalate":
        # the climb is issued ONLY with the transport's approval for the rung above (cap_ladder.TRANSPORT_RULE);
        # hops None (no endpoint record at all) is undeterminable and refuses — never a climb on absent evidence
        tv = cl.transport_check(int(next_cap), hops)
        transport = tv.as_dict()
        if not tv.ok:
            decision = TRANSPORT_REFUSED
    c = rec["counts"]
    ident = ({"kind": "static", "attack": args.attack, "cell": args.defense, "private_root": str(args.private_root),
              "private_run": args.private_run, "out_root": str(Path(args.private_root) / args.private_run)}
             if kind == "static" else
             {"kind": "controls", "cell": args.cell, "control": args.control, "out_root": str(args.out_root)})
    out = {
        "schema": SCHEMA, **ident, "cap": int(args.cap), "store": rec["identity"]["store"],
        "decision": decision, "next_cap": (int(next_cap) if decision == "escalate" else None),
        "decision_tool": {"decision": rec["decision"], "next_cap": rec["next_cap"], "rule": rec["rule"],
                          "record": str(args.decision_out), "record_sha256": ld.sha256_file(Path(args.decision_out)),
                          "at_ceiling": rec["at_ceiling"], "ladder": rec["ladder"]},
        "counts": {k: c.get(k) for k in ("n_responses", "n_at_cap", "n_loop", "n_repetitive", "n_boundary",
                                          "n_boundary_any", "n_truncated", "share_truncated")},
        "n_items_escalate": int(rec["n_items_escalate"]) if decision == "escalate" else 0,
        "items": (str(args.items_out) if decision == "escalate" else None),
        "items_sha256": rec["items_out_sha256"],
        "sibling": rec["sibling"],
        "transport": transport,
        "hop_timeouts_s": hops,
        "endpoint": args.endpoint, "endpoint_present": endpoint is not None,
        "driver_record": (driver_record(Path(rec["identity"]["store"]), args.control) if kind == "controls"
                          else {"present": False, "note": "static stores carry escalation.json, not cap_ladder.json"}),
        "classes": {"loop": "at cap, rep_ratio > 0.5 or degenerate: NON-TERMINATING, never escalated",
                    "repetitive": "at cap, 0.2 <= rep_ratio <= 0.5: reported as its own class, never escalated",
                    "boundary": "at cap here, the sibling trial terminated in <= 0.5 x cap: noise, never escalated",
                    "truncated": "at cap, cap-coherent, not boundary: the ONLY class that escalates; unresolved "
                                 "when it exits at the ceiling (exit_ceiling)"},
        "tool": {"path": str(HERE), "sha256": ld.sha256_file(HERE), "argv": argv},
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_name(outp.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, outp)
    if decision == "escalate":
        print(f"ESCALATE {int(next_cap)} {args.items_out}", flush=True)
    else:
        print(f"EXIT {decision}", flush=True)
    return 0


def transport(args) -> int:
    from sisteer.capeval.tiers import cap_ladder as cl
    driver_s = float(args.driver_timeout_s) if args.driver_timeout_s is not None else driver_timeout_default()
    endpoint = read_endpoint(args.endpoint)
    if endpoint is None:
        print(f"REFUSED no endpoint record at {args.endpoint}: the transport hop is undeterminable", flush=True)
        return EXIT_TRANSPORT
    tv = cl.transport_check(int(args.next_cap), cl.served_hop_timeouts(endpoint, driver_s))
    print(("OK " if tv.ok else "REFUSED ") + tv.reason, flush=True)
    return 0 if tv.ok else EXIT_TRANSPORT


def _gate_args(p: argparse.ArgumentParser, *, cap_help: str, items_help: str) -> None:
    """What the two gate subcommands share: the rung, the transport inputs, the rule sources, the three outputs."""
    p.add_argument("--cap", type=int, required=True, help=cap_help)
    p.add_argument("--endpoint", default=None, help="the standup record whose proxy hop the transport verdict reads")
    p.add_argument("--driver-timeout-s", type=float, default=None,
                   help="the driver's client timeout (default: served_capability_run.AIOHTTP_TOTAL_TIMEOUT_S)")
    p.add_argument("--rep-ratio-source", default=str(REP_RATIO_SOURCE_DEFAULT))
    p.add_argument("--degeneracy-source", default=str(DEGENERACY_SOURCE_DEFAULT))
    p.add_argument("--out", required=True, help="gate.json")
    p.add_argument("--decision-out", required=True, help="ladder_decide's decision.json")
    p.add_argument("--items-out", required=True, help=items_help)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("decide", help="the bound decision for one (cell, control) at the rung just measured")
    d.add_argument("--out-root", required=True)
    d.add_argument("--cell", required=True)
    d.add_argument("--control", required=True)
    d.add_argument("--sibling-out-root", default=None, help="the other trial's root at the same rung (boundary rule)")
    _gate_args(d, cap_help="the rung the store was generated at",
               items_help="the disposition file the driver's --items consumes")
    st_ = sub.add_parser("static", help="the bound decision for one (attack, cell) static store at the rung just drawn")
    st_.add_argument("--private-root", required=True)
    st_.add_argument("--private-run", required=True, help="the store's private_run (the base run or a _cap<N> rung)")
    st_.add_argument("--attack", required=True)
    st_.add_argument("--defense", required=True, help="the TrainTest cell id")
    st_.add_argument("--sibling-private-run", default=None, help="the other trial's run at the same rung (boundary rule)")
    _gate_args(st_, cap_help="the rung the store was drawn at",
               items_help="the disposition file escalate_static.py --items consumes")
    t = sub.add_parser("transport", help="re-check the transport against a NEW standup's record before any driver")
    t.add_argument("--endpoint", required=True)
    t.add_argument("--next-cap", type=int, required=True)
    t.add_argument("--driver-timeout-s", type=float, default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return transport(args) if args.mode == "transport" else decide(args)


if __name__ == "__main__":
    sys.exit(main())
