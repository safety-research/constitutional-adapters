"""LIVE pre-check of the r2 register-edit templates on a few REAL records
before the batch waves go out: render the exact requests the stage specs
build, call messages.create on the lowprio smoke key SEQUENTIALLY (one in
flight — no coordination threshold crossed), run the specs' own offline
verification on each answer, and print pass/fail + the new spans to read.

    uv run python scripts/register_edit_precheck.py --config configs/control-scale-r2.yaml \
        --per-marker 2 --stories 4

Nothing is written to the run dir; nothing is submitted to the Batch API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_generator import register_edit as RE  # noqa: E402
from corpus_generator.batch import credential_halt, message_text  # noqa: E402
from corpus_generator.client import build_client  # noqa: E402
from corpus_generator.config import RunConfig  # noqa: E402
from corpus_generator.pipeline import build_ctx, resolve_environment  # noqa: E402
from corpus_generator.stages import ParseError  # noqa: E402
from corpus_generator.templates import repo_root  # noqa: E402


def create_or_stop(client, params: dict, *, custom_id: str, n_so_far: int):
    """One sequential ``messages.create``: the message, or None for an item's API error (printed, recorded as ``api_error``
    by the caller) -- except a credential the API REJECTS (a 401 / 403, or under ``SISTEER_SECRETS_MODE=wif`` the federation
    identity refused at the token exchange; ``credential_halt``), which STOPS the pre-check at that request with the typed
    message as the terminal line: before this every later record issued its own doomed request and the summary read
    ``0/N passed`` (the WIF driver sweep, 2026-10-13)."""
    try:
        return client.messages.create(**params)
    except Exception as e:  # noqa: BLE001 -- classified: the credential stops the pre-check, anything else is this item's
        halt = credential_halt(e)
        if halt is not None:
            print(f"  CREDENTIAL REJECTED at {custom_id}: {halt} -- stopping after {n_so_far} result(s); no later request is "
                  "issued (D6)", flush=True)
            raise SystemExit(str(halt))
        print(f"  API error: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per-marker", type=int, default=2)
    ap.add_argument("--stories", type=int, default=4)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N selected ids per marker")
    ap.add_argument("--dry-run", action="store_true", help="render + print the prompts only")
    args = ap.parse_args()
    root = repo_root()
    cfg = resolve_environment(RunConfig.load(Path(args.config)), root=root)
    plan = RE.load_plan(cfg, root, 1)
    client = None if args.dry_run else build_client("smoke")

    results = []
    for corpus in plan["sources"]:
        ctx, _run_dir = build_ctx(cfg, corpus, root)
        recs, _ = RE.load_records_with_line_shas(Path(plan["sources"][corpus]["path"]))
        records = {RE.rec_id(r): r for r in recs}
        for spec in RE._edit_specs_for(cfg, corpus, plan, records, 1):
            reqs = spec.build_requests(ctx)
            n = args.stories if corpus == "stories" else args.per_marker
            for req in reqs[args.offset: args.offset + n]:
                print(f"\n=== {spec.name} :: {req.custom_id}")
                if args.dry_run:
                    print(req.params["messages"][0]["content"][:3000])
                    continue
                t0 = time.time()
                params = dict(req.params)
                msg = create_or_stop(client, params, custom_id=req.custom_id, n_so_far=len(results))
                if msg is None:
                    results.append((req.custom_id, "api_error"))
                    continue
                text = message_text(msg.content)
                outcome = {"type": "succeeded", "stop_reason": msg.stop_reason, "text": text, "batch_id": "live"}
                try:
                    rows = spec.parse(ctx, req, outcome)
                    row = rows[0]
                    print(f"  OK in {time.time() - t0:.0f}s — retained {row.data['retained']}, "
                          f"{len(row.data['edits'])} edit(s)")
                    for sp in row.data["spans"]:
                        print("  OLD:", json.dumps(sp["old"][:400], ensure_ascii=False))
                        print("  NEW:", json.dumps(sp["new"][:400], ensure_ascii=False))
                    results.append((req.custom_id, "ok"))
                except ParseError as e:
                    print(f"  REJECTED in {time.time() - t0:.0f}s: {e}")
                    print("  raw:", text[:600])
                    results.append((req.custom_id, "rejected"))
    if not args.dry_run:
        n_ok = sum(1 for _c, s in results if s == "ok")
        print(f"\n{n_ok}/{len(results)} passed verification")
        for c, s in results:
            if s != "ok":
                print("  ", s, c)


if __name__ == "__main__":
    main()
