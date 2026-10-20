#!/usr/bin/env python3
"""Harvest + archive the in-flight batches of a CANCELLED run, touching no
pipeline state.

Written for control-briefs2-20260902 (2026-09-02): `corpusgen run` was killed
seconds after launch — the metre-history placebo is superseded and nothing is
to be generated from it — but the two single-request ROOT waves (advice/a1,
stories/s1) had already been submitted.
A submitted batch cannot be un-submitted, so this script polls those batches
to completion, writes each one's results next to its wave as
``archived-results-<batch_id>.jsonl`` (normalized outcomes, wire custom_ids
exactly as returned; written to a ``.tmp`` and published by rename only after
the whole stream was read and its length matched the batch's request counts),
and records ``CANCELLED.json`` at the run root. The ledgers are deliberately
left as they were (state=submitted): the run is not to be resumed, and nothing
downstream is built from the archived output. A credential the API rejects --
a 401 / 403, or under ``SISTEER_SECRETS_MODE=wif`` a federation identity the
token exchange refused -- ends the script at that request with the typed
``WaveCredentialRejected`` text as its terminal line: no archive under the
final name, no ``CANCELLED.json``.

usage: uv run python scripts/archive_cancelled_batches.py <run_id> "<reason>"
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_generator.batch import Ledger, credential_halt, normalize_result  # noqa: E402
from corpus_generator.client import build_client  # noqa: E402
from corpus_generator.templates import repo_root  # noqa: E402

POLL_S = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stop_on_credential(call, *, what: str):
    """One control-plane call under the credential stop (the WIF driver sweep): a 401 / 403 answer -- or under
    ``SISTEER_SECRETS_MODE=wif`` the federation identity refused at the token exchange -- ends the script with the typed
    ``WaveCredentialRejected`` text as its terminal line (``credential_halt``, the one builder the engines share) and
    nothing written; every other exception passes as it was raised. The script is sequential, so the first rejection
    is also its last request."""
    try:
        return call()
    except Exception as e:  # noqa: BLE001 -- classified: the credential only
        halt = credential_halt(e)
        if halt is None:
            raise
        print(f"CREDENTIAL REJECTED at {what}: {halt}", flush=True)
        raise SystemExit(str(halt)) from e


def _expected_results(counts: dict) -> int | None:
    """The number of result entries an ENDED batch streams: one per request, whatever its outcome (succeeded / errored /
    canceled / expired; ``processing`` is 0 once ended) -- the sum of the API's request_counts. None when the record
    carries no integer counts (nothing to validate against)."""
    vals = [v for v in counts.values() if isinstance(v, int) and not isinstance(v, bool)]
    return sum(vals) if vals else None


def archive_results(client, bid: str, out: Path, *, expected: int | None) -> int:
    """Stream one ended batch's results into ``out`` ATOMICALLY: written to ``<out>.tmp``, published by rename only after
    the whole stream was read and its length matched ``expected``; the tmp is removed on EVERY failure (a credential
    rejection met at the ``results`` call or while consuming the stream -- the typed stop, through
    :func:`_stop_on_credential` -- a 5xx, a count mismatch), so no empty or truncated archive is ever left under the
    final name: the final file used to be opened BEFORE the results request, and a rejection there left an empty
    archive and no CANCELLED.json. Returns the number of results written."""
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    try:
        with tmp.open("w", encoding="utf-8") as f:
            stream = _stop_on_credential(lambda: client.messages.batches.results(bid), what=f"results {bid}")
            it = iter(stream)
            while True:
                entry = _stop_on_credential(lambda: next(it, _END), what=f"results {bid} (entry {n + 1})")
                if entry is _END:
                    break
                o = normalize_result(entry)
                o["batch_id"] = bid
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
        if expected is not None and n != expected:
            raise SystemExit(f"batch {bid}: the results stream carried {n} entries but request_counts sum to {expected} -- "
                             f"not archived (nothing published under {out.name}); re-run to read it again")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return n


_END = object()


def main(run_id: str, reason: str) -> None:
    run_dir = repo_root() / "data" / "runs" / run_id
    if not run_dir.exists():
        raise SystemExit(f"no such run dir: {run_dir}")
    client = build_client("batch")

    pending: dict[tuple[Path, str], str] = {}
    for ledger_path in sorted(run_dir.glob("*/*/ledger.jsonl")):
        wave_dir = ledger_path.parent
        for chunk, st in Ledger(ledger_path).chunk_states().items():
            if st.batch_id:
                pending[(wave_dir, chunk)] = st.batch_id
    print(
        "in-flight:",
        {f"{w.relative_to(run_dir)}/{c}": b for (w, c), b in pending.items()},
        flush=True,
    )

    archived: dict[str, dict] = {}
    while pending:
        for (wave_dir, chunk), bid in list(pending.items()):
            b = _stop_on_credential(lambda: client.messages.batches.retrieve(bid), what=f"retrieve {bid}")
            counts = b.request_counts.model_dump()
            print(_now(), wave_dir.relative_to(run_dir), bid, b.processing_status, counts, flush=True)
            if b.processing_status != "ended":
                continue
            out = wave_dir / f"archived-results-{bid}.jsonl"
            n = archive_results(client, bid, out, expected=_expected_results(counts))
            archived[f"{wave_dir.relative_to(run_dir)}/{chunk}"] = {
                "batch_id": bid,
                "file": str(out.relative_to(run_dir)),
                "n_results": n,
                "request_counts": counts,
                "created_at": str(b.created_at),
                "ended_at": str(b.ended_at),
            }
            del pending[(wave_dir, chunk)]
        if pending:
            time.sleep(POLL_S)

    marker = {
        "run_id": run_id,
        "state": "CANCELLED",
        "reason": reason,
        "archived_at": _now(),
        "archived_batches": archived,
        "note": (
            "Only the root waves listed above were ever submitted; the kill "
            "landed before any downstream wave was built. Results are archived "
            "verbatim for the record and NOTHING is built from them: no "
            "materialized records, no filter, no finals, no showcase. Ledgers "
            "untouched (state=submitted); do not resume this run."
        ),
    }
    (run_dir / "CANCELLED.json").write_text(json.dumps(marker, indent=1), encoding="utf-8")
    print("wrote", run_dir / "CANCELLED.json", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
