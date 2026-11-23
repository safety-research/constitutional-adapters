"""One key, one filed outcome: the merge of a track's verdict ledgers (batch run directories, the bake-off reuse ledgers they carry
verdicts from, and the LIVE run directories of ``rejudge.common.live`` that judged their never-submitted keys) into the outcome each judge
input files, and the delivery gate that reads a live run's coverage of its source. Shared by the StrongREJECT filing
(``strongreject/file_records.py``, where it was written) and the Terra filing (``terra/file_records.py``); read only, stdlib only.

THE RULES (:func:`read_outcomes`), none of which depends on the ledgers' order:
  * a REUSE directory contributes only the keys the run directory naming it carries (its ``reuse.json``; the arm's other keys were
    re-sent);
  * a key with a FILEABLE outcome (a verdict or an item outcome) in two ledgers is refused (:class:`DuplicateOutcomes`), except a key a
    LIVE run directory took over after its init (a handover), where the first fileable outcome stands (earliest harvest) and the other
    is kept in the item's ``duplicate_outcomes``;
  * an infrastructure outcome gives way to a fileable one from another ledger, and of two infrastructure outcomes the later harvest
    stands;
  * a fileable LIVE outcome whose serving broke the live protocol as the live directory's routing now stands (a provider the routing
    does not allow, another model, a price above the provider's list price, reasoning tokens) refuses the filing
    (:class:`OffProtocolOutcomes`) until it is re-opened and judged again;
  * an outcome a live ledger holds for the SOURCE's batch reply (``error.from_source_run``: a handed-over cut reply whose cap cannot be
    raised, made terminal without sending) files as that batch reply: ``backend`` batch, the source's batch id, provider and attempts.

THE GATE (:func:`gate_readiness`): a live run's SOURCE is judged by what the live run leaves uncovered: its pending and in-flight batches
and its PAUSED file (the gated batch harvester) do not hold the delivery when every key without a terminal outcome there is a live key.
"""
from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.common import batchfile as BF  # noqa: E402
from rejudge.common import live as LV  # noqa: E402
from rejudge.common import state as S  # noqa: E402

#: terminal outcomes a filing writes as rows by default: the verdict and the item outcomes (the infrastructure ones are the provider's
#: or the batch's failure, not the item's)
ITEM_UNSCORED = ("moderation", "judge_refusal", "parse_error", "truncated")
FILEABLE = ("verdict",) + ITEM_UNSCORED
#: the registry's per-item transport label of each backend (``live_keys`` / ``track.json`` ``backend``)
TRANSPORT = {"batch": "openrouter_batch", "live": "openrouter_live"}
#: the ``live_attempts`` columns the protocol check of a live outcome reads (:func:`rejudge.common.live.attempt_violation`)
LIVE_PROTOCOL_COLUMNS = ("outcome", "http_status", "served", "provider", "served_model", "reasoning_tokens", "cost_usd", "cost_source",
                         "prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")


class DuplicateOutcomes(ValueError):
    pass


class OffProtocolOutcomes(ValueError):
    """A live run directory holds a terminal, fileable outcome its serving broke the protocol of (:func:`rejudge.common.live.attempt_violation`:
    a provider the routing does not allow, another model, a price above the provider's list price, reasoning tokens billed)."""


def live_source_problems(run_dirs: Sequence[str]) -> List[str]:
    """Why these run directories may not be filed together (empty = they may): a live run directory files with its SOURCE run directory,
    which must be among them (else the source's verdicts would file as unjudged and the coverage gate could not be read)."""
    given = {os.path.abspath(os.path.expanduser(r)) for r in run_dirs}
    out = []
    for rd in run_dirs:
        if not LV.is_live_dir(rd):
            continue
        src = str(BF.read_track(rd).get("source_run_dir"))
        if os.path.abspath(os.path.expanduser(src)) not in given:
            out.append(f"{rd} is the live run of {src}'s pending keys; pass that run directory as --run-dir too")
    return out


def live_protocol(run_dirs: Iterable[str]) -> Optional[Dict[str, Any]]:
    """The live serving of these run directories, for the judge's protocol (None when none is live): the routing object and the endpoint
    map (slug, quantization; prices left out) of the live run directory, and whether batch run directories file beside it. Two live
    run directories with other routings are refused (one judge row is one protocol)."""
    run_dirs = list(run_dirs)
    lives = [rd for rd in run_dirs if LV.is_live_dir(rd)]
    if not lives:
        return None
    out = None
    for rd in lives:
        lv = BF.read_track(rd).get("live") or {}
        cur = {"routing": lv.get("routing"), "providers": {k: {"slug": v.get("slug"), "quantization": v.get("quantization")}
                                                           for k, v in sorted((lv.get("providers") or {}).items())}}
        if out is not None and cur != out:
            raise SystemExit(f"REFUSED: live run directories with different routings ({lives}); file them under separate judges")
        out = cur
    out = dict(out, with_batch=any(not LV.is_live_dir(rd) for rd in run_dirs))
    return out


def carried_keys(run_dirs: Iterable[str]) -> Dict[str, set]:
    """reuse run directory -> the keys a run directory carries from it (its ``reuse.json`` ``keys``, written by ``build_manifests``). A
    key of the reuse directory that is NOT carried was re-sent by the run directory that names it, so its outcome there is not filed."""
    out: Dict[str, set] = collections.defaultdict(set)
    for rd in run_dirs:
        try:
            spec = BF.read_track(rd)
        except (OSError, ValueError):
            continue
        reuse = [os.path.abspath(os.path.expanduser(r)) for r in (spec.get("reuse_run_dirs") or [])]
        path = os.path.join(rd, "reuse.json")
        if reuse and os.path.exists(path):
            keys = set(json.load(open(path)).get("keys") or [])
            for r in reuse:
                out[r] |= keys
    return dict(out)


def live_handover_keys(run_dir: str) -> set:
    """The keys a live run directory took over from its source AFTER the init (``live_keys.origin`` in_flight / reask): a late result of
    the source's batch for one of them is a duplicate the filing resolves by time, never an error."""
    con = sqlite3.connect(f"file:{os.path.join(run_dir, 'state.sqlite')}?mode=ro", uri=True)
    try:
        if not LV.has_table(con, "live_keys"):
            return set()
        return {r[0] for r in con.execute("SELECT custom_id FROM live_keys WHERE origin != 'pending'")}
    finally:
        con.close()


def read_outcomes(run_dirs: Iterable[str], keys: Optional[set] = None, stats: Optional[collections.Counter] = None,
                  fileable: Sequence[str] = FILEABLE) -> Dict[str, Dict[str, Any]]:
    """custom_id -> its terminal outcome across the run directories' ledgers, with ``backend`` (batch | live), the live attempt's
    ``quantization`` and ``generation_id``. The rules, none of which depends on the ledgers' order:
      * a REUSE directory contributes only the keys the run directory naming it carries (``reuse.json``); its other keys were re-sent;
      * a key with a FILEABLE outcome (a verdict, moderation, judge refusal, parse error, cut reply) in two ledgers is refused
        (:class:`DuplicateOutcomes`), except a key a live run took over from its source after its init (a handover): there the FIRST
        fileable outcome stands (earliest harvest; then the path), and the other is kept in ``duplicate_outcomes``;
      * an infrastructure outcome (provider_error, missing, batch_*) gives way to a fileable one from another ledger, and of two
        infrastructure outcomes the later harvest stands; the kept outcome lists the ledgers it settled in ``settled_infra``;
      * a fileable LIVE outcome whose attempt a provider served must keep the live protocol as the live directory's routing now stands
        (``live.attempt_violation``: an allowed provider, the model asked for, a cost within its list price, no reasoning tokens); one
        that does not refuses the filing (:class:`OffProtocolOutcomes`; ``live.py reopen --off-protocol`` re-opens it). The engine
        records such replies ``held``, so only a ledger the routing changed under (a provider dropped after its replies) can hold one."""
    fset = tuple(fileable)
    run_dirs = [os.path.abspath(os.path.expanduser(r)) for r in run_dirs]
    carried = carried_keys(run_dirs)
    handover: set = set()
    for rd in run_dirs:
        if LV.is_live_dir(rd):
            handover |= live_handover_keys(rd)
    st = stats if stats is not None else collections.Counter()
    out: Dict[str, Dict[str, Any]] = {}
    dup: Dict[str, List[str]] = collections.defaultdict(list)
    off: List[Tuple[str, str, str]] = []
    for rd in run_dirs:
        live = LV.is_live_dir(rd)
        only = carried.get(rd)
        con = sqlite3.connect(f"file:{os.path.join(rd, 'state.sqlite')}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        proto = None
        if live and LV.has_table(con, "live_attempts"):
            # the serving facts of the attempt each terminal outcome came from, checked against the protocol as the routing stands
            spec, cfg, allowed = LV.live_settings(rd)
            proto = (cfg, str(spec["model"]), allowed)
            la_cols = [c for c in LIVE_PROTOCOL_COLUMNS if c in LV.attempt_columns(con)]
            q = ("SELECT r.custom_id, r.outcome, r.terminal, r.verdict_json, r.error_json, r.status_code, r.attempts, r.batch_id, r.provider, "
                 "r.harvested_utc, la.quantization, la.generation_id"
                 + "".join(f", la.{c} AS la_{c}" for c in la_cols)
                 + " FROM requests r LEFT JOIN live_attempts la ON la.custom_id = r.custom_id AND la.attempt = r.attempts WHERE r.terminal = 1")
        else:
            q = ("SELECT custom_id, outcome, terminal, verdict_json, error_json, status_code, attempts, batch_id, provider, harvested_utc, "
                 "NULL AS quantization, NULL AS generation_id FROM requests WHERE terminal = 1")
        for r in con.execute(q):
            cid = r["custom_id"]
            if keys is not None and cid not in keys:
                continue
            if only is not None and cid not in only:
                st["reuse_not_carried_ignored"] += 1
                continue
            d = dict(r)
            la = {k[3:]: d.pop(k) for k in list(d) if k.startswith("la_")}
            if proto is not None and d["outcome"] in fset and la.get("outcome") is not None:      # the attempt row exists
                v = LV.attempt_violation(proto[0], proto[1], proto[2], dict(la, outcome=la.get("outcome") or d["outcome"]))
                if v:
                    off.append((cid, rd, f"{v[0]}: {v[1]}"))
                    continue
            d["verdict"] = json.loads(d.pop("verdict_json")) if r["verdict_json"] else None
            d["error"] = json.loads(d.pop("error_json")) if r["error_json"] else None
            d["run_dir"] = rd
            d["backend"] = "live" if live else "batch"
            if live and isinstance(d["error"], dict) and d["error"].get("from_source_run"):
                # the SOURCE's batch reply (a cut reply whose cap cannot be raised) that a live run made terminal without sending
                # (``rejudge.common.live.LiveRunner._finalize_uncappable``): filed as the batch reply it is, its batch id and provider
                # the source's, its attempts the source's
                d["backend"] = "batch"
                d["attempts"] = d["error"].get("source_attempts") or d["attempts"]
                d["quantization"] = d["generation_id"] = None
                st["live_ledger_source_replies"] += 1
            prev = out.get(cid)
            if prev is None:
                out[cid] = d
                continue
            pf, df = prev["outcome"] in fset, d["outcome"] in fset
            if pf and df:
                if cid in handover:
                    order = sorted((prev, d), key=lambda o: (str(o.get("harvested_utc") or ""), o["run_dir"]))
                    keep, lose = order[0], order[1]
                    keep["duplicate_outcomes"] = list(prev.get("duplicate_outcomes", [])) + list(d.get("duplicate_outcomes", [])) + [
                        {"run_dir": lose["run_dir"], "backend": lose["backend"], "outcome": lose["outcome"], "harvested_utc": lose["harvested_utc"],
                         "batch_id": lose.get("batch_id")}]
                    out[cid] = keep
                    st["handover_duplicates_first_kept"] += 1
                    continue
                if not dup[cid]:
                    dup[cid].append(prev["run_dir"])
                dup[cid].append(rd)
                continue
            if pf:
                keep, lose = prev, d
            elif df:
                keep, lose = d, prev
            else:
                keep = max((prev, d), key=lambda o: (str(o.get("harvested_utc") or ""), o["run_dir"]))
                lose = d if keep is prev else prev
            keep["settled_infra"] = sorted(set(prev.get("settled_infra", []) + d.get("settled_infra", []) + [lose["run_dir"]]))
            out[cid] = keep
            st["infra_settled_by_another_ledger"] += 1
        con.close()
    if off:
        kinds = collections.Counter(m.split(":", 1)[0] for _, _, m in off)
        raise OffProtocolOutcomes(f"{len(off)} terminal live outcomes break the live protocol as the routing now stands ({dict(kinds)}; e.g. "
                                  f"{[(k[:16], m[:160]) for k, _, m in off[:3]]}); re-open them (live.py reopen --off-protocol --apply), "
                                  f"let the engine judge them again, then file")
    if dup:
        first = sorted(dup)[:3]
        raise DuplicateOutcomes(f"{len(dup)} judge-input keys have a fileable terminal outcome in more than one ledger (e.g. "
                                f"{[(k[:16], dup[k]) for k in first]}); each key is filed from one ledger")
    return out


def gate_readiness(run_dirs: List[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """``state.readiness`` of every run directory, with a live run's SOURCE judged by what the live run does not cover: a source request
    without a terminal outcome is settled elsewhere when its key is a live key (taken over at the init, or by a handover), so the source's
    pending / in-flight batches and its PAUSED file (the gated batch harvester) do not hold the delivery when every such key is covered.
    The live run itself must have stopped moving as any run directory must. Returns (readiness per run directory, coverage notes)."""
    readies = {rd: S.readiness(rd) for rd in run_dirs}
    covered: Dict[str, set] = collections.defaultdict(set)
    for rd in run_dirs:
        if LV.is_live_dir(rd):
            src = os.path.abspath(os.path.expanduser(str(BF.read_track(rd).get("source_run_dir"))))
            con = sqlite3.connect(f"file:{os.path.join(rd, 'state.sqlite')}?mode=ro", uri=True)
            try:
                covered[src] |= {r[0] for r in con.execute("SELECT custom_id FROM live_keys")}
            finally:
                con.close()
    notes: Dict[str, Any] = {}
    for rd in run_dirs:
        if rd not in covered:
            continue
        cov = covered[rd]
        r = dict(readies[rd])
        con = sqlite3.connect(f"file:{os.path.join(rd, 'state.sqlite')}?mode=ro", uri=True)
        try:
            states = {b: s for b, s in con.execute("SELECT batch_key, state FROM batches")}
            unc: collections.Counter = collections.Counter()
            unc_batches: collections.Counter = collections.Counter()
            n_cov = 0
            for cid, bk, outcome in con.execute("SELECT custom_id, batch_key, outcome FROM requests WHERE terminal = 0"):
                if cid in cov:
                    n_cov += 1
                    continue
                unc[outcome or "not_harvested"] += 1
                unc_batches[(states.get(bk), bk)] += 1
        finally:
            con.close()
        n_unc = sum(unc.values())
        live_states = collections.Counter(s for (s, _bk) in unc_batches if s in S.LIVE_STATES)
        refused = sorted({bk for (s, bk) in unc_batches if s == "refused"})
        notes[rd] = {"covered_by_live": n_cov, "uncovered_non_terminal": n_unc, "raw_paused": r.get("paused"), "raw_live_batches": r.get("live_batches")}
        r.update({"terminal": r["requests"] - n_unc, "non_terminal": dict(unc), "live_batches": dict(live_states), "refused_batches": refused,
                  "non_terminal_in_refused": sum(n for (s, _bk), n in unc_batches.items() if s == "refused"),
                  "paused": r.get("paused") if n_unc else None, "covered_by_live": n_cov})
        r["ready"] = not r["paused"] and not r["live_batches"] and n_unc == 0 and not refused
        r["ready_if_partial"] = not r["paused"] and not r["live_batches"] and n_unc == r["non_terminal_in_refused"]
        readies[rd] = r
    return readies, notes


def live_extra(o: Mapping[str, Any]) -> Dict[str, Any]:
    """The per-item fields of a live outcome (serving quantization, OpenRouter generation id) and of a resolved handover duplicate."""
    out: Dict[str, Any] = {}
    if o.get("backend") == "live":
        out["quantization"] = o.get("quantization")
        out["generation_id"] = o.get("generation_id")
    if o.get("duplicate_outcomes"):
        out["duplicate_outcomes"] = o["duplicate_outcomes"]
    return out
