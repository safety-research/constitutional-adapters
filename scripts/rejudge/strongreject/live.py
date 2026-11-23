#!/usr/bin/env python3
"""The StrongREJECT MiniMax-M3 judge's LIVE slot: a batch run directory's never-submitted keys, judged on OpenRouter chat completions with
the batch track's exact request bodies (the engine: ``rejudge.common.live``; its module docstring is the contract).

    cd <checkout>; export PYTHONPATH=$PWD/src:$PWD/scripts
    python scripts/rejudge/strongreject/live.py init --live-dir LIVE --source-run-dir FULL --global-root ROOT \\
        --global-ceiling-usd N --max-spend-usd M [--reasoning-reroute]        # once: the live directory and its key set (no network)
    python scripts/rejudge/strongreject/live.py dry-run --live-dir LIVE         # keys, cost bands at the fp8 price mix, duration (no network)
    python scripts/rejudge/strongreject/live.py run --live-dir LIVE [--limit N] # the engine in the foreground ($OPENROUTER_API_KEY must be set)
    python scripts/rejudge/strongreject/live.py status --live-dir LIVE          # read only
    python scripts/rejudge/strongreject/live.py handover --live-dir LIVE --reasks       # re-asks a stopped batch harvester will never build
    python scripts/rejudge/strongreject/live.py handover --live-dir LIVE --in-flight --not-before 2030-01-01T00:00:00Z   # the fallback
    python scripts/rejudge/strongreject/live.py reopen --live-dir LIVE --provider NAME [--apply]   # judge a dropped provider's keys again
    python scripts/rejudge/strongreject/live.py reopen --live-dir LIVE --off-protocol [--apply]    # the terminal keys the filing would refuse

THE KEY SET (``init``): every key of the source run directory's PENDING batches (never POSTed), read only from its ledger, manifests and
bodies; never a key of a batch that is submitted, in flight or harvested. The source's batch harvester must not run while the live run
owns those keys (build the source with a submit gate that names a file that does not exist, ``build_manifests.py --require-file``): the
engine pauses if any of those batches leaves ``pending``.

SPEND. Nothing has a default: ``init`` requires ``--max-spend-usd`` (this run's ceiling, USD incl. the 5.5% credit fee), ``--global-root``
(an absolute path of an existing directory: the global check counts every OpenRouter ledger under it, so put the batch run directories
there too) and ``--global-ceiling-usd`` (the ceiling of that check). ``--reasoning-reroute`` turns on the engine's re-route of a key that
keeps drawing billed reasoning (an engine setting, outside the judge's identity).

THE ROUTING (``track.json`` ``live.routing``; OpenRouter's provider-routing object, https://openrouter.ai/docs/guides/routing/provider-selection,
read 2026-11-11): ``only`` the fp8 endpoints DeepInfra ``deepinfra/fp8`` (the batch tier's provider), GMICloud ``gmicloud/fp8``,
StreamLake ``streamlake/fp8`` and Parasail ``parasail/fp8``; ``quantizations: ["fp8"]``; ``ignore`` Parasail's ``fast`` tier (fp8 at 1.5x
the price), CoreWeave (fp4), Venice (fp8; its own API prepends a system prompt of its own unless a request turns it off, so the judge
might not see exactly the batch request) and MiniMax's own endpoint (quantization ``unknown``); ``require_parameters`` OFF (every fp8
endpoint, like the batch run's DeepInfra flex endpoint, supports only ``tool_choice: auto``, so the forced choice is degraded to ``auto``
exactly as it was for the batch verdicts; see ``rejudge.common.live.default_routing``), ``max_price`` at the dearest allowed endpoint
(0.30 / 1.20 USD per MTok), fallbacks among them allowed, no ``sort`` (OpenRouter's load balancing spreads the load). Every reply's
serving provider is recorded, its quantization from this map. The routing and the providers' slugs and quantizations are part of the
judge's identity (``file_records.judge_spec``): changing them files under another judge id.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import live as LV  # noqa: E402
from rejudge.common.state import readiness  # noqa: E402
from rejudge.strongreject import judges as J  # noqa: E402

JUDGE_ENTRY = "minimax-m3"
#: the live (synchronous) endpoints allowed to serve, by the provider name a reply carries: their endpoint slug, quantization and price
#: (input, output, cache read; USD per MTok, the list price of https://openrouter.ai/api/v1/models/minimax/minimax-m3/endpoints read
#: 2026-11-11; GMICloud listed a 20% and StreamLake a 50% discount, which the reported usage cost reflects)
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "DeepInfra": {"slug": "deepinfra/fp8", "quantization": "fp8", "price": [0.28, 1.10, 0.056]},
    "GMICloud": {"slug": "gmicloud/fp8", "quantization": "fp8", "price": [0.24, 0.96, 0.048]},
    "StreamLake": {"slug": "streamlake/fp8", "quantization": "fp8", "price": [0.30, 1.20, 0.06]},
    "Parasail": {"slug": "parasail/fp8", "quantization": "fp8", "price": [0.30, 1.20, 0.06]},
}
IGNORE = ["parasail/fast", "coreweave", "venice", "minimax"]


def routing() -> Dict[str, Any]:
    return LV.default_routing(PROVIDERS, quantizations=("fp8",), ignore=IGNORE)


def check_entry(model: str) -> None:
    e = J.get(JUDGE_ENTRY)
    if model != e["model"] or LV.FORBIDDEN.search(model):
        raise SystemExit(f"REFUSED: the source run's model {model!r} is not the {JUDGE_ENTRY} entry's {e['model']!r}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init", help="create the live run directory and register the key set (no network)")
    pi.add_argument("--live-dir", required=True)
    pi.add_argument("--source-run-dir", required=True, help="the batch run directory whose pending keys the live run takes")
    pi.add_argument("--max-spend-usd", type=float, required=True, help="this run's ceiling (USD incl. the credit fee)")
    pi.add_argument("--global-root", required=True, help="the root the global check walks: an absolute path of an existing directory "
                    "holding every OpenRouter run directory whose spend counts against --global-ceiling-usd")
    pi.add_argument("--global-ceiling-usd", type=float, required=True, help="the ceiling of the global check")
    pi.add_argument("--concurrency", type=int, default=300)
    pi.add_argument("--reasoning-reroute", action="store_true", help="re-route a key past max_reasoning_resends to the allowed endpoints "
                    "that have not billed reasoning on it (an engine setting, outside the judge's identity)")
    pd = sub.add_parser("dry-run", help="keys, cost, duration (no network)")
    pd.add_argument("--live-dir", required=True)
    pd.add_argument("--concurrency", type=int)
    pd.add_argument("--assume-latency-s", type=float, default=10.0)
    pd.add_argument("--no-global", action="store_true", help="skip reading the other ledgers' spend")
    pd.add_argument("--out", help="write the JSON here (default <live dir>/dry_run.json)")
    pr = sub.add_parser("run", help="the engine (foreground); needs $OPENROUTER_API_KEY")
    pr.add_argument("--live-dir", required=True)
    pr.add_argument("--limit", type=int, help="pilot: at most N keys get a first attempt (spread over the batches); exit 0 when they are terminal")
    pr.add_argument("--concurrency", type=int)
    pr.add_argument("--max-hours", type=float)
    pr.add_argument("--max-spend-usd", type=float)
    pr.add_argument("--global-check-s", type=float)
    pr.add_argument("--base-url", help="tests only: a local mock of OpenRouter")
    ph = sub.add_parser("handover", help="move keys the batch run will not finish into the live run (no network)")
    ph.add_argument("--live-dir", required=True)
    ph.add_argument("--in-flight", action="store_true", help="keys of POSTed, unharvested batches (the fallback)")
    ph.add_argument("--reasks", action="store_true", help="keys waiting on a re-ask the gated batch harvester will never build")
    ph.add_argument("--not-before", help="refuse before this UTC time (YYYY-MM-DDTHH:MM:SSZ); for a scheduled fallback")
    ph.add_argument("--reason", default="")
    ph.add_argument("--ingest", action="store_true", help="register the keys now when no engine holds the live run's writer lock")
    po = sub.add_parser("reopen", help="make terminal live outcomes non-terminal again, so the engine judges those keys again")
    po.add_argument("--live-dir", required=True)
    po.add_argument("--provider", action="append", default=[], help="the keys whose terminal attempt this provider served (repeatable)")
    po.add_argument("--off-protocol", action="store_true", help="the keys whose terminal attempt breaks the protocol as the routing now stands")
    po.add_argument("--reason", default="")
    po.add_argument("--apply", action="store_true", help="write (refused while an engine runs); without it, count only")
    ps = sub.add_parser("status", help="read only")
    ps.add_argument("--live-dir", required=True)
    a = ap.parse_args(argv)
    live_dir = os.path.abspath(os.path.expanduser(a.live_dir))

    if a.cmd == "init":
        src = os.path.abspath(os.path.expanduser(a.source_run_dir))
        from rejudge.common import batchfile as BF
        check_entry(str(BF.read_track(src).get("model")))
        live = {"concurrency": a.concurrency, "global_ceiling_usd": float(a.global_ceiling_usd), "global_root": LV.check_global_root(a.global_root)}
        if a.reasoning_reroute:
            live["reasoning_reroute"] = True
        info = LV.init_live(live_dir, src, routing(), PROVIDERS, float(a.max_spend_usd), live=live)
        print(json.dumps({k: v for k, v in info.items() if k != "calibration"}, indent=1))
        return 0
    if a.cmd == "dry-run":
        d = LV.dry_run(live_dir, concurrency=a.concurrency, assume_latency_s=a.assume_latency_s,
                       global_root=None if a.no_global else LV.global_root_of(live_dir))
        out = a.out or os.path.join(live_dir, "dry_run.json")
        with open(out, "w") as fh:
            json.dump(d, fh, indent=1)
        print(json.dumps(d, indent=1))
        return 0
    if a.cmd == "run":
        import asyncio
        ov = {"concurrency": a.concurrency, "max_hours": a.max_hours, "global_check_s": a.global_check_s, "base_url": a.base_url}
        r = LV.LiveRunner(live_dir, overrides=ov, limit=a.limit, max_spend_usd=a.max_spend_usd)
        check_entry(r.model)
        return asyncio.run(r.run_async())
    if a.cmd == "handover":
        modes = (["in_flight"] if a.in_flight else []) + (["reasks"] if a.reasks else [])
        if not modes:
            raise SystemExit("name --in-flight and / or --reasks")
        path = LV.build_handover(live_dir, modes, a.reason or ("fallback: the in-flight batches did not come back in time" if a.in_flight
                                                               else "re-asks the gated batch harvester will not build"),
                                 not_before=a.not_before)
        added = 0
        if path and a.ingest:
            from rejudge.common.state import Ledger, WriterLocked
            try:
                led = Ledger(live_dir).writer()
            except WriterLocked:
                print("the live engine holds the writer lock: it ingests the handover at its next check (within global_check_s)")
            else:
                try:
                    LV.ensure_schema(led.conn)
                    added = LV.ingest_handovers(led, live_dir)
                finally:
                    led.close()
        print(json.dumps({"handover": path, "ingested_now": added}))
        return 0
    if a.cmd == "reopen":
        out = LV.reopen_live(live_dir, providers=a.provider, off_protocol=a.off_protocol, reason=a.reason, apply=a.apply)
        print(json.dumps(out, indent=1))
        return 0
    if a.cmd == "status":
        p = json.load(open(os.path.join(live_dir, "progress.json"))) if os.path.exists(os.path.join(live_dir, "progress.json")) else {}
        r = readiness(live_dir)
        keep = ("updated_utc", "state", "stopping", "paused", "keys", "outcomes", "in_flight", "attempts_per_min", "terminal_keys_per_min",
                "latency_s_last_5m", "concurrency", "spend_usd", "provider_mix", "off_protocol_held", "parked_keys", "eta")
        print(json.dumps({"last_run": LV.last_run(live_dir), "auto_start": LV.auto_start_refusal(live_dir) or "allowed",
                          "progress": {k: p.get(k) for k in keep}, "readiness": {k: v for k, v in r.items() if k != "fingerprint"}},
                         indent=1, default=str))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
