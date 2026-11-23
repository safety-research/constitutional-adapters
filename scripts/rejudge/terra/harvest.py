#!/usr/bin/env python3
"""The shared harvester (``rejudge/common/harvester.py``, same CLI, same run directory) with three additions the library it is built on
lacks: a request refused on the provider ACCOUNT pauses the run, re-asks are counted within an outcome's class, and a batch whose
MODERATION share is far above the original judge's baseline is HELD, not recorded, until a person looks.

    PYTHONPATH=src:scripts python scripts/rejudge/terra/harvest.py --run-dir RUN [--once] [--poll-s 60] [--max-hours 72] [--dry-run]

1. ACCOUNT PAUSE. The shared harvester pauses when a whole BATCH fails on the account or credential (a dead own-provider key fails
   validation). A batch OpenRouter accepts can still come back with requests refused on the account itself (OpenAI models answer a
   blocked account with ``403 Policy Violation: this user has been blocked for a previous policy violation``). The track's parsers
   classify that, and any 403 that names no content, as ``account_error`` (:mod:`rejudge.terra.parsing`); here the first one writes
   ``<run_dir>/PAUSED`` (the shared mechanism: nothing is submitted or re-asked until a person fixes the account and removes the file;
   harvesting goes on).

2. RE-ASKS COUNTED WITHIN THE OUTCOME'S CLASS. The inherited ``_record`` counts every earlier attempt, whatever its outcome, so a request
   that once failed on the account (or in a batch that expired) reached its first parse error or cut call with its one re-ask already
   spent. Here an outcome's re-asks (``track.json`` ``retry``) are counted over the attempts of its own class only:
     ITEM     verdict, moderation, judge_refusal, parse_error, truncated, provider_error, missing -- the request's own re-asks
     BATCH    batch_failed, batch_expired, batch_cancelled -- the whole batch ended without naming the request: their own re-asks
     NEVER    account_error, batch_failed_auth, held -- never terminal and never counted (the run pauses instead)

3. MODERATION-SHARE HOLD. Moderation is terminal at once (recorded per item, never re-asked), so a provider-wide answer that reads as
   moderation would become a run of permanent rows at $0 cost. Before a batch is recorded its results are parsed once more and, when
   ``moderation`` covers at least ``health.moderation_share`` of them (default :data:`MODERATION_SHARE`, 25x the ~0.2% of C/P/I units
   the Sonnet judge abstained on or was blocked for) and at least ``health.min_count``, every moderation result of the batch is recorded
   as ``held`` (the parsed outcome and error kept in the attempt's error) and the run PAUSES. A person reads the rejections; if they are
   real, the line ``moderation_share`` in ``<run_dir>/ACKNOWLEDGED`` (or ``health.acknowledged``) turns the rule off for the run and
   removing ``PAUSED`` re-asks the held requests once, which records them as what they are.

The library is not modified. It pauses on account failures and counts re-asks within a class itself too; the overrides here stay (same
behaviour, pinned by the tests), share the library's outcome classes, and add the moderation-share hold on top of the library's health
rules rather than in place of them.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Mapping, Optional, Set

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import harvester as H  # noqa: E402

#: the request-level account outcome (:mod:`rejudge.terra.parsing`) and the shared harvester's whole-batch one
PAUSING_OUTCOMES = ("account_error", "batch_failed_auth")
BATCH_OUTCOMES = tuple(H.BATCH_OUTCOMES)
#: never terminal, never counted: the track's own three plus every outcome the shared harvester itself treats so (since the library's guards,
#: ``balance_error`` and ``batch_failed_other`` -- a request of a batch that failed NAMING ANOTHER request was never sent, so it must not
#: become a terminal item outcome here either)
UNCOUNTED_OUTCOMES = tuple(dict.fromkeys(("account_error", "batch_failed_auth", "held") + tuple(H.UNCOUNTED_OUTCOMES)))
MODERATION_SHARE = 0.05
MIN_COUNT = 10
ACK_FILE = "ACKNOWLEDGED"
ACK_MODERATION = "moderation_share"


def attempt_class(outcome: str) -> str:
    if outcome in UNCOUNTED_OUTCOMES:
        return "never"
    if outcome in BATCH_OUTCOMES:
        return "batch"
    return "item"


def acknowledged(run_dir: str, extra: Optional[list] = None) -> Set[str]:
    out = {str(x).strip() for x in (extra or []) if str(x).strip()}
    path = os.path.join(run_dir, ACK_FILE)
    if os.path.exists(path):
        with open(path) as fh:
            out |= {l.split("#", 1)[0].strip() for l in fh if l.split("#", 1)[0].strip()}
    return out


class TerraHarvester(H.Harvester):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # layered on the shared harvester's health config (its identical-error flood, truncation and error-share rules still run on
        # every batch), with the moderation-share rule's own default added
        self.health_cfg = {"moderation_share": MODERATION_SHARE, "min_count": MIN_COUNT, "acknowledged": [], **self.health_cfg}

    # -- 2. class-scoped re-asks, 1. the account pause ----------------------------------------------------
    def _record(self, cid, key, bid, outcome, p, provider, model):  # noqa: D401 -- the shared signature
        p = p or {}
        cls = attempt_class(outcome)
        if outcome == "verdict":
            terminal = True
        elif cls == "never":
            terminal = False
        else:
            prior = [r[0] for r in self.ledger.conn.execute("SELECT outcome FROM attempts WHERE custom_id = ?", (cid,))]
            n = sum(1 for o in prior if attempt_class(str(o)) == cls) + 1
            terminal = n > int(self.retry.get(outcome, 0))
        usage = p.get("usage")
        self.ledger.record_outcome(cid, key, bid, outcome, terminal, p.get("verdict"), p.get("error"), p.get("status_code"),
                                   p.get("finish_reason"), usage, H.request_cost(model, usage), provider, commit=False)
        if outcome in PAUSING_OUTCOMES:
            err = p.get("error") or {}
            self.pause(f"a request of batch {bid} ({key}) was refused on the provider ACCOUNT ({outcome}, HTTP "
                       f"{err.get('status_code') or p.get('status_code')}): {str(err.get('message') or '')[:300]} -- nothing is recorded "
                       f"against the item; fix the account, remove PAUSED")

    # -- 3. the moderation-share hold ---------------------------------------------------------------------
    def moderation_hold(self, b: Mapping[str, Any], ids: Set[str]) -> Dict[str, Any]:
        """``{"hold": set of custom_ids, "reason": str}`` for one fetched batch (empty when the rule does not fire or is acknowledged)."""
        cfg = self.health_cfg
        if ACK_MODERATION in acknowledged(self.run_dir, cfg.get("acknowledged")):
            return {"hold": set(), "reason": None}
        outcomes: Dict[str, str] = {}
        for entry in b.get("results") or []:
            cid = entry.get("custom_id")
            if cid not in ids or cid in outcomes:
                continue
            try:
                outcomes[cid] = str(self.parser(entry).get("outcome"))
            except Exception:  # noqa: BLE001 -- the recording pass records the crash
                outcomes[cid] = "parse_error"
        n = len(outcomes)
        mod = {c for c, o in outcomes.items() if o == "moderation"}
        if not n or len(mod) < int(cfg["min_count"]) or len(mod) / n < float(cfg["moderation_share"]):
            return {"hold": set(), "reason": None}
        return {"hold": mod, "reason": (f"moderation share {len(mod)} of {n} results ({len(mod) / n:.1%}) >= {float(cfg['moderation_share']):.1%}: "
                                        f"HELD, not recorded as moderation. Read the rejections (raw/ and the attempts' error); if they are real, "
                                        f"add the line {ACK_MODERATION} to {os.path.join(self.run_dir, ACK_FILE)} and remove PAUSED (the held "
                                        f"requests are re-asked once and recorded); otherwise fix the cause and remove PAUSED")}

    def harvest(self, key: str, bid: str) -> Dict[str, int]:
        b = self.fetch_final(key, bid)
        if b is None:
            return {}
        ids = set(H.BF.read_manifest(self.run_dir, key)["custom_ids"])
        decision = self.moderation_hold(b, ids)
        hold = decision["hold"]
        parser = self.parser

        def held_parser(entry: Mapping[str, Any]) -> Dict[str, Any]:
            p = parser(entry)
            if entry.get("custom_id") in hold and p.get("outcome") == "moderation":
                return {"outcome": "held", "usage": p.get("usage"), "status_code": p.get("status_code"),
                        "error": {"kind": "held_moderation_share", "parsed_outcome": "moderation", "parsed_error": p.get("error")}}
            return p

        self.fetch_final = lambda _k, _b: b          # fetched once above (and saved verbatim there)
        if hold:
            self.parser = held_parser
        try:
            counts = super().harvest(key, bid)
        finally:
            del self.fetch_final
            self.parser = parser
        if hold:
            self.pause(f"batch {bid} ({key}): {decision['reason']}")
        return counts


def main(argv=None) -> int:
    H.Harvester = TerraHarvester          # main() constructs the module-level name
    return H.main(argv)


if __name__ == "__main__":
    sys.exit(main())
