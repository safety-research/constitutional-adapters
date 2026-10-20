#!/usr/bin/env python3
"""IFEval canonical battery — the LADDER PLANNER. CPU-only, over the PULLED stores after
pass 1 (and again after the 4,096 rungs for the 8,192 decision).

Per LAUNCH row whose store at the current rung is COMPLETE (its outcome record + DONE marker + RESULT COMPLETE):
  * the adapter ``inspect_rung.classify_store(task=inspect/ifeval, source_out_root=<the rung below, wave-relative>)`` — the ONE at-cap authority over the STORE (the filed
    outcomes; at-cap = the provider's stop reason; the eligible set = at-cap − loop DRAWS, boundary included) — classifies
    it; THE DECISION (``ifeval_contracts.LADDER_DECISION_RULE``, P9 / C1) is then taken here over the MERGED reading — pass
    1's draws with every rung's replacements up to and including this rung, denominator 1,082 — and filed INSIDE the record
    under ``ifeval_contracts.LADDER_DECISION_KEY`` before the record is written to ``ifeval_paths(...).cap_ladder`` (the one
    writer of that file). At pass 1 the merged reading IS the store, so the decision equals the adapter's exit (asserted);
    at a rung the store's own exit / counts stay in the record as DIAGNOSTICS (3 still-at-cap of 24 redraws is 12.5 % of the
    rung store but 3 of 1,082 is under the 1 % trigger: the decision is ``below_threshold``, and no 8,192 rung is planned).
    Every consumer — this planner, the analysis's chain walk, gates I5 / I6 — reads THAT field, never the store's exit;
  * for an escalating DECISION the ``rung_subset/2`` body is built FROM the decision (the store's eligible draws at the
    decision's next cap; equal to the adapter's own subset whenever the adapter escalated too — asserted) and written to
    ``ifeval_paths(..., next_cap).subset`` (validated by ``ifeval_contracts.validate_item_subset_v2``);
  * ``--from-cap 8192`` files the CEILING record (the residual at the top of the ladder, P9's "8,192 exits reported with
    residuals") and plans nothing above it;
  * ``ladder_state.json`` in the rung's out root — the idempotence record: a rung in state running / complete is NEVER
    re-planned, a planned rung with a DIFFERING subset digest refuses by name, an identical one re-prints its line unchanged;
  * the rung wave's PREDICTED GPU-h from the RECONCILED outcomes (P34: ``n_draws x cap / aggregate tok/s + the standup adder``
    per promoted cell; the cell's OBSERVED aggregate tok/s = its ledger's accepted-attempt forwarded completions / the
    generation wall they spanned, ``ifeval_contracts.DECODE_RATE_RULE`` — never sum(tokens) / sum(latency), which divides
    by every overlapping request's seconds (C5); the source leg's served S read off its endpoint.json) printed BEFORE any
    launch line — the operator admits the rung wave against 2x the plan's S1 allowance;
  * one launcher invocation per promoted cell (handed to the dispatcher, never spawned here):
      tinycap_served.py --sha <pin> --model <slug> --cells <cell> --modes off --controls ifeval
          --item-subset <ifeval_paths(...).subset> --off-cap-by-task ifeval=<cap> --out-dir <ifeval_paths(...).out_root>
          --app-name ifeval-canonical-<sha7>-rung<cap>
    a Nemotron rung names the certification its cap's geometry entry needs first.

  tinycap_ifeval_ladder_plan.py --wave-root <pull> --sha <pin> [--from-cap 2048|4096|8192] [--rows i,j-k] [--s1-allowance-gpu-h 3.9] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO / "src", REPO / "scripts" / "sdfbase" / "ifeval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.tiers import cap_ladder as CL  # noqa: E402
from sisteer.capeval.tiers import inspect_rung as IR  # noqa: E402
from sisteer.capeval.tinycap import completion  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import ifeval_contracts as IC  # noqa: E402

import if_matrix as IM  # noqa: E402

TASK = IC.IFEVAL_TASK
CAPS = IC.IFEVAL_CAP_LADDER
CEILING = IC.IFEVAL_CAP_CEILING
STANDUP_ADDER_S = 250.0
MIN_RATE_ROWS = 10
S1_ALLOWANCE_GPU_H = 3.9                      # scenario S1 (the expected ladder); the operator admits a wave <= 2x it
STATE_SCHEMA = "ifeval_ladder_state/1"
STATE_PLANNED, STATE_RUNNING, STATE_COMPLETE = "planned", "running", "complete"
LAUNCHER = "scripts/modal/tinycap_served.py"
SCRIPT = "scripts/tinycap_ifeval_ladder_plan.py"


def store_status(out_root: Path) -> str:
    """`complete` when the producer's outcomes file, its DONE marker and a RESULT `result=COMPLETE` line agree; `pending` when
    none is present; a disagreement refuses by name."""
    outcomes = sorted(out_root.glob(f"cells/*/{C.TASK_OUTCOMES_DIR}/{completion.marker_filename(TASK)}"))
    done = sorted(out_root.glob(f"cells/*/{C.DONE_DIR}/{completion.marker_filename(TASK)}"))
    result = out_root / "RESULT"
    complete_line = result.is_file() and any("result=COMPLETE" in ln for ln in result.read_text(encoding="utf-8").splitlines())
    if outcomes and done and complete_line:
        return "complete"
    if not outcomes and not done and not complete_line:
        return "pending"
    raise SystemExit(f"{out_root}: outcomes {'present' if outcomes else 'absent'}, done marker {'present' if done else 'absent'}, RESULT "
                     f"{'COMPLETE' if complete_line else 'not COMPLETE'} — the store's records disagree; refusing to plan on it")


def atomic_write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def observed_decode_rate(ledger_rows: Sequence[Mapping[str, Any]], *, accepted: str,
                         evidence_by_id: Mapping[str, Mapping[str, Any]], min_rows: int = MIN_RATE_ROWS) -> dict[str, Any]:
    """The SOURCE leg's OBSERVED aggregate decode rate (``ifeval_contracts.DECODE_RATE_RULE``): the ACCEPTED
    attempt's forwarded task rows (``contracts.rows_of_attempts`` — the row's own stamp, else its request-evidence record's),
    their completion tokens summed over the generation wall they spanned (last request end - first request start). The
    per-request reading ``sum(tokens) / sum(latency_s)`` rides beside it, labelled, for comparison only: S concurrent
    requests each count their own overlapping seconds, so it under-reads the aggregate by ~S and inflated every GPU-hour
    estimate by the same factor (C5)."""
    task_rows = [r for r in ledger_rows if r.get("task") == TASK]
    mine = C.rows_of_attempts(task_rows, [accepted], evidence_by_id=evidence_by_id)
    forwarded = [r for r in mine if r.get("rejected") is not True]
    return IC.aggregate_decode_tok_s(forwarded, min_rows=min_rows)


def source_served_seqs(out_root: Path, *, default: int) -> int:
    """The source leg's SERVED concurrency (the GPU-h prediction's rate basis), read from the PRODUCER's record beside the
    store: the Modal launcher's ``served_geometry.json`` — its ``geometry.max_num_seqs`` IS the served S (the KV-derived one
    when the record carries a ``kv_shrink``, whose ``served_max_num_seqs`` says the same; the endpoint block beside it
    agrees) — else the cluster run's ``endpoint.json`` (``geometry.kv_shrink.served_max_num_seqs`` / ``geometry.max_num_seqs``
    / ``max_num_seqs``); ``default`` (the CLI's) when neither record is beside the store. The analysis step found this
    reading ``endpoint.json`` alone — a file the Modal launcher never writes — so every in-container prediction of the rung
    wave fell back to the CLI default for its served S (a labelled ESTIMATE, never a gate; the decisions read no rate)."""
    served = Path(out_root) / "served_geometry.json"
    if served.is_file():
        doc = json.loads(served.read_text(encoding="utf-8"))
        geometry = doc.get("geometry") if isinstance(doc, Mapping) else None
        if isinstance(geometry, Mapping):
            shrink = geometry.get(IC.ENDPOINT_KEY_GEOMETRY_KV_SHRINK)
            if isinstance(shrink, Mapping) and shrink.get("served_max_num_seqs") is not None:
                return int(shrink["served_max_num_seqs"])
            if geometry.get("max_num_seqs") is not None:
                return int(geometry["max_num_seqs"])
    ep_path = Path(out_root) / "endpoint.json"
    if not ep_path.is_file():
        return int(default)
    ep = json.loads(ep_path.read_text(encoding="utf-8"))
    geometry = ep.get("geometry") or {}
    shrink = geometry.get(IC.ENDPOINT_KEY_GEOMETRY_KV_SHRINK)
    if isinstance(shrink, Mapping) and shrink.get("served_max_num_seqs", shrink.get("served_seqs")) is not None:
        return int(shrink.get("served_max_num_seqs", shrink.get("served_seqs")))
    seqs = geometry.get("max_num_seqs", ep.get("max_num_seqs"))
    return int(seqs) if seqs is not None else int(default)


def predict_gpu_h(n_draws: int, cap: int, tok_s: float, *, served_seqs: int, rate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The rung's predicted GPU-hours (per cell): draws x cap x f / (the source leg's AGGREGATE tok/s at its
    served S) + the standup adder — the all-draws-at-cap bound of the rung (S4's arithmetic per rung) is reported beside the
    central number. ``served_seqs`` is the source leg's served concurrency (the rate's basis); the rung itself runs at most
    ``min(n_draws, served_seqs)`` sequences, recorded as ``rung_concurrency``. ``rate`` (the facts behind ``tok_s``:
    :func:`observed_decode_rate`) is recorded as ``rate_basis`` when given."""
    if n_draws < 1 or tok_s <= 0 or served_seqs < 1:
        raise ValueError("predict_gpu_h needs n_draws >= 1, tok_s > 0, served_seqs >= 1")
    central_tokens = n_draws * cap * 0.5                 # central: a re-drawn truncation continues ~half the new cap on average
    worst_tokens = n_draws * cap
    basis = ({k: v for k, v in rate.items() if k != "tok_s"} if rate is not None else None)
    return {"n_draws": n_draws, "cap": cap, "tok_s": tok_s, "served_seqs": int(served_seqs), "rung_concurrency": min(int(n_draws), int(served_seqs)),
            "gpu_h_central": round((central_tokens / tok_s + STANDUP_ADDER_S) / 3600, 4),
            "gpu_h_all_at_cap": round((worst_tokens / tok_s + STANDUP_ADDER_S) / 3600, 4),
            "formula": ("GPU-h = (n_draws x cap x f / tok_s + 250 s) / 3600; f = 0.5 central, 1.0 every draw at cap; tok_s = the source "
                        "leg's AGGREGATE decode rate at its served S (accepted-attempt completions / generation wall)"),
            "rate_basis": basis}


def merged_reading(wave_root: Path, key: str, view: str, cell: str, cap: int) -> tuple[list[dict[str, Any]], list[int], int]:
    """``(draws, chain, n_rung)``: the MERGED reading at ``cap`` (P9) — pass 1's outcome items with every rung's replacements
    applied in ladder order up to and including ``cap`` (``inspect_rung.merge_draws`` hop by hop, each rung's draws read off
    its manifest's draw-subset block), the chain of caps composed, and the number of draws a rung supplied. Every store is
    found through ``ifeval_paths`` under ``wave_root`` (relocation-safe). At pass 1 the reading is the store itself."""
    chain = [CAPS[0]]
    merged = list(completion.read_task_outcomes(IR.cell_dir_of(wave_root / IC.ifeval_paths(key, view, cell, CAPS[0]).out_root), TASK)["items"])
    redrawn: set[tuple[str, int]] = set()
    for rung in CAPS[1:]:
        if rung > int(cap):
            break
        rung_root = wave_root / IC.ifeval_paths(key, view, cell, rung).out_root
        cd = IR.cell_dir_of(rung_root)
        block = (completion.read_store_manifest(cd) or {}).get(C.MANIFEST_KEY_ITEM_SUBSET) or {}
        draws = block.get("draws")
        if not draws:
            raise SystemExit(f"{rung_root}: the rung store's manifest carries no draw-subset block — the merged reading cannot be composed")
        merged = IR.merge_draws(merged, completion.read_task_outcomes(cd, TASK)["items"], draws)
        redrawn |= {(str(i), int(s)) for i, s in draws}
        chain.append(rung)
    return merged, chain, len(redrawn)


def decide_record(rec: Mapping[str, Any], *, merged: Sequence[Mapping[str, Any]], chain: Sequence[int], n_rung: int, cap: int,
                  sha: str) -> dict[str, Any]:
    """THE decision block (``ifeval_contracts.ladder_decision_block``) for the store's record ``rec`` over the merged
    reading: ``inspect_rung.decide`` on the merged population's at-cap count with the STORE's eligible draws (its truncated
    non-loop draws — the only draws a rung re-draws), the store's recorded hops, the instrument's ladder / ceiling /
    threshold."""
    n_draws, n_at = len(merged), sum(1 for m in merged if m["at_cap"])
    eligible = [(str(i), int(s)) for i, s in (rec.get("eligible_draws") or [])]
    exit_, next_cap, tv = IR.decide(n_draws=n_draws, n_at_cap=n_at, eligible_draws=eligible, cap=int(cap), hops=rec["hops"],
                                    ladder=CAPS, ceiling=CEILING, threshold=IC.IFEVAL_TRIGGER_THRESHOLD)
    return IC.ladder_decision_block(reading=(IC.READING_PASS1 if len(chain) == 1 else IC.READING_MERGED), chain=chain, n_draws=n_draws,
                                    n_at_cap=n_at, n_rung=n_rung, exit=exit_, next_cap=next_cap, eligible_draws=eligible,
                                    transport=(tv.as_dict() if tv is not None else None), threshold=IC.IFEVAL_TRIGGER_THRESHOLD,
                                    rule=IC.IFEVAL_TRIGGER_RULE, trigger_provenance=IC.IFEVAL_TRIGGER_PROVENANCE,
                                    decided_by={"sha": sha, "script": SCRIPT, "rule": IC.LADDER_DECISION_RULE})


def rung_subset_body(rec: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    """The ``rung_subset/2`` body FROM the decision: the store's eligible draws at the decision's next cap, bound to the
    store (``source_out_root`` / identity hash / cap) with its at-cap counts, under the decision's transport verdict —
    validated by ``ifeval_contracts.validate_item_subset_v2``."""
    draws = [[str(i), int(s)] for i, s in decision["eligible_draws"]]
    ids = sorted({i for i, _ in draws})
    doc = {"schema": IC.ITEM_SUBSET_SCHEMA_V2, "task": TASK, "source_out_root": str(rec["source"]["out_root"]),
           "source_identity_hash": str(rec["source"]["identity_hash"]), "source_cap": int(rec["cap"]), "cap": int(decision["next_cap"]),
           "rule": str(rec["rule"]), "threshold": rec["threshold"], "ids": ids, "draws": draws, "n_items": len(ids), "n_draws": len(draws),
           "n_at_cap": int(rec["n_at_cap"]), "n_truncated": int(rec["n_truncated"]), "n_loop": int(rec["n_loop"]),
           "n_boundary": int(rec["n_boundary"]), "transport_check": dict(decision["transport"])}
    return IC.validate_item_subset_v2(doc, where="rung subset from the decision")


def decided_record(wave_root: Path, row: Mapping[str, Any], *, from_cap: int, record: Mapping[str, Any],
                   adapter_subset: Mapping[str, Any] | None, sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """``(record with THE decision, the decision, the rung subset or None)`` for the store at ``from_cap``: the adapter's
    record VERBATIM (its rule / instrument checked) plus :data:`ifeval_contracts.LADDER_DECISION_KEY`. At pass 1 the merged
    reading is the store, so the decision must equal the adapter's exit (a disagreement refuses: two rules on one
    population). At a rung the store's own exit stays a diagnostic. The subset is built from the decision; when the adapter
    filed one too (it escalated on its own population) the two must be identical."""
    rec = dict(record)
    if rec.get("rule") != IC.IFEVAL_TRIGGER_RULE or rec.get("instrument") != IR.IFEVAL_INSTRUMENT.name:
        raise SystemExit(f"{row['job_id']}: the adapter's record carries another rule / instrument — refusing to file it")
    key, view, cell = row["model_key"], row["view"], row["cell"]
    try:
        merged, chain, n_rung = merged_reading(Path(wave_root), key, view, cell, int(from_cap))
    except (IR.InspectRungError, completion.CompletionError) as e:
        raise SystemExit(f"{row['job_id']}: the merged reading up to {from_cap} cannot be composed — {e}") from None
    if chain[-1] != int(from_cap) or int(rec.get("cap") or -1) != int(from_cap):
        raise SystemExit(f"{row['job_id']}: the record at {rec.get('cap')} / the chain {chain} do not end at {from_cap}")
    decision = decide_record(rec, merged=merged, chain=chain, n_rung=n_rung, cap=int(from_cap), sha=sha)
    if len(chain) == 1 and (decision["exit"], decision["next_cap"]) != (rec.get("exit"), rec.get("next_cap")):
        raise SystemExit(f"{row['job_id']}: at pass 1 the decision {decision['exit']!r} disagrees with the adapter's {rec.get('exit')!r} "
                         "over the same population — refusing")
    rec[IC.LADDER_DECISION_KEY] = decision
    subset = rung_subset_body(rec, decision) if decision["exit"] == CL.ESCALATE else None
    if adapter_subset is not None and subset is not None and dict(adapter_subset) != subset:
        raise SystemExit(f"{row['job_id']}: the adapter's subset and the decision's subset differ — a torn classification; refusing")
    return rec, decision, subset


def launcher_line(row: Mapping[str, Any], *, sha: str, cap: int) -> str:
    p = IC.ifeval_paths(row["model_key"], row["view"], row["cell"], cap)
    return (f"{LAUNCHER} --sha {sha} --model {row['slug']} --cells {row['cell']} --modes {C.THINKING_OFF} --controls {IC.CONTROL_IFEVAL} "
            f"--item-subset <vol>/{p.subset} {IC.FLAG_OFF_CAP_BY_TASK} {IC.IFEVAL_TASK_NAME}={cap} --out-dir <vol>/{p.out_root} "
            f"--app-name ifeval-canonical-{sha[:7]}-rung{cap}")


TOO_FEW_ROWS_MARK = "no observed decode rate"        # ifeval_contracts.aggregate_decode_tok_s's refusal for < min_rows rows


def store_rate(root: Path, *, job_id: str) -> dict[str, Any]:
    """One store's OBSERVED aggregate decode rate (:func:`observed_decode_rate` over its ACCEPTED attempt's forwarded
    ledger rows, joined to its request evidence). A store without a ledger or an accepted attempt refuses by name; a
    rate that cannot be measured raises ``ValueError`` (the caller decides)."""
    ledger = root / IC.LEDGER_FILENAME
    if not ledger.is_file():
        raise SystemExit(f"{job_id}: no {IC.LEDGER_FILENAME} under {root} — the observed decode rate comes from the ledger")
    manifest = completion.read_store_manifest(IR.cell_dir_of(root)) or {}
    accepted = manifest.get(C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT)
    if not isinstance(accepted, str) or not accepted:
        raise SystemExit(f"{job_id}: the manifest under {root} names no {C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT} — the rate is "
                         "read off the accepted attempt's rows alone")
    evidence_path = root / C.REQUEST_EVIDENCE_FILENAME
    try:
        ev_by_id = C.evidence_by_request_id(read_jsonl(evidence_path) if evidence_path.is_file() else [])
        return observed_decode_rate(read_jsonl(ledger), accepted=accepted, evidence_by_id=ev_by_id)
    except C.AttemptScopeError as e:
        raise SystemExit(f"{job_id}: {e}") from None


def rate_for_prediction(out_root: Path, *, source_root: Path | None, job_id: str) -> dict[str, Any]:
    """The rate the NEXT rung's prediction rests on (P34): this store's own when its accepted attempt forwarded enough
    task rows (``MIN_RATE_ROWS``), else — a RUNG store re-draws only its source's eligible draws, so a small subset
    measures no rate (the 2026-09-13 rung wave: a 4-draw rung store refused ``need >= 10`` and the refusal aborted the
    whole planner run, leaving later rows unplanned) — the SOURCE leg's (the rung below, wave-relative), labelled in
    ``rate_source`` so the re-estimate names the ledger it read. THE decision never reads a rate. A pass-1 store has no
    source and refuses as before; every other rate problem still refuses by name."""
    try:
        rate = store_rate(out_root, job_id=job_id)
        rate["rate_source"] = {"out_root": str(out_root), "fallback": False}
        return rate
    except ValueError as e:
        if source_root is None or TOO_FEW_ROWS_MARK not in str(e):
            raise SystemExit(f"{job_id}: {e}") from None
        why = str(e)
    try:
        rate = store_rate(source_root, job_id=job_id)
    except ValueError as e:
        raise SystemExit(f"{job_id}: {e} (the source leg {source_root}, read because the rung store measured none: {why})") from None
    rate["rate_source"] = {"out_root": str(source_root), "fallback": True, "reason": f"the rung store's ledger: {why}"}
    return rate


def plan_cell(row: Mapping[str, Any], *, wave_root: Path, sha: str, from_cap: int,
              classify: Callable[..., tuple], served_seqs: int) -> dict[str, Any]:
    """One launch row at ``from_cap``: its record, and for an escalating cell the draw subset, the rung state, the predicted
    GPU-h and its launcher line. Never re-plans a running / complete rung; refuses a differing subset on a planned one."""
    key, view, cell = row["model_key"], row["view"], row["cell"]
    paths = IC.ifeval_paths(key, view, cell, from_cap)
    out_root = wave_root / paths.out_root
    res: dict[str, Any] = {"index": row["index"], "job_id": row["job_id"], "cell_key": f"{key}/{view}/{cell}", "from_cap": from_cap,
                           "out_root": str(out_root), "status": None, "exit": None, "next_cap": None, "subset_path": None, "n_draws": None,
                           "prediction": None, "line": None, "note": None, "certification_needed": None, "decision": None,
                           "store_exit": None, "share_at_cap": None}
    status = store_status(out_root)
    res["status"] = status
    if status != "complete":
        res["note"] = "store not complete: not classified"
        return res
    source_root: Path | None = None
    if int(from_cap) > CAPS[0]:
        # seam 4 (M11; lane Y's C2 residual): a RUNG store's subset records the SOURCE root ABSOLUTELY, where the planner
        # ran — a wave pulled to another root cannot be re-planned in place from it. The source is resolved through the
        # wave-relative layout, like every other store the planner reads; the adapter still binds by identity hash / cap.
        source_root = wave_root / IC.ifeval_paths(key, view, cell, CAPS[CAPS.index(int(from_cap)) - 1]).out_root
        record, adapter_subset = classify(out_root, source_out_root=source_root)
    else:
        record, adapter_subset = classify(out_root)
    # THE decision (C1): over the merged reading, filed INSIDE the record; the store's own exit is a diagnostic beside it
    rec, decision, subset = decided_record(wave_root, row, from_cap=from_cap, record=record, adapter_subset=adapter_subset, sha=sha)
    atomic_write_json(wave_root / paths.cap_ladder, rec)
    res.update(exit=decision["exit"], share_at_cap=decision["share_at_cap"], decision=decision, store_exit=record.get("exit"))
    if decision["exit"] != CL.ESCALATE:
        differs = f"; the store's own exit {record.get('exit')!r} ({record.get('n_at_cap')} of {record.get('n_draws')})" if record.get("exit") != decision["exit"] else ""
        where = "no rung above" if int(from_cap) >= CEILING else "no rung"
        res["note"] = f"{decision['reading']} reading: {decision['exit']} ({decision['n_at_cap']} of {decision['n_draws']} at cap{differs}); {where}"
        return res
    assert subset is not None
    doc = IC.validate_item_subset_v2(dict(subset), where=f"{row['job_id']} rung subset")
    next_cap = int(doc["cap"])
    res["next_cap"] = next_cap
    sha16 = IC.item_subset_sha16_v2(doc["draws"])
    rung_paths = IC.ifeval_paths(key, view, cell, next_cap)
    state_path = wave_root / rung_paths.ladder_state
    if state_path.is_file():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        rung_st = st.get("rung") or {}
        if rung_st.get("state") in (STATE_RUNNING, STATE_COMPLETE) or st.get("attempts"):
            res["note"] = f"rung already {rung_st.get('state') or 'attempted'} — not re-planned"
            res["subset_path"], res["n_draws"] = rung_st.get("subset_path"), rung_st.get("n_draws")
            return res
        if rung_st.get("subset_sha16") and rung_st["subset_sha16"] != sha16:
            raise SystemExit(f"{row['job_id']}: a rung was already planned over subset {rung_st['subset_sha16']} but the adapter now "
                             f"yields {sha16} ({doc['n_draws']} draws) — the source store changed under a planned rung; refusing")
    subset_path = wave_root / rung_paths.subset
    if subset_path.is_file():
        prev = json.loads(subset_path.read_text(encoding="utf-8"))
        if IC.item_subset_sha16_v2(prev.get("draws") or []) != sha16:
            raise SystemExit(f"{row['job_id']}: {subset_path} holds another subset; refusing")
    else:
        atomic_write_json(subset_path, doc)
    rate = rate_for_prediction(out_root, source_root=source_root, job_id=row["job_id"])
    pred = predict_gpu_h(int(doc["n_draws"]), next_cap, float(rate["tok_s"]), served_seqs=source_served_seqs(out_root, default=served_seqs), rate=rate)
    atomic_write_json(state_path, {"schema": STATE_SCHEMA, "attempts": [], "accepted_generation_attempt": None,
                                   "rung": {"state": STATE_PLANNED, "subset_path": str(subset_path), "subset_sha16": sha16, "n_draws": int(doc["n_draws"]),
                                            "n_items": int(doc["n_items"]), "source_out_root": str(out_root), "source_identity_hash": doc["source_identity_hash"],
                                            "cap": next_cap, "prediction": pred, "trigger_provenance": IC.IFEVAL_TRIGGER_PROVENANCE,
                                            "planned_by": {"sha": sha, "script": "scripts/tinycap_ifeval_ladder_plan.py"}}})
    res.update({"subset_path": str(subset_path), "n_draws": int(doc["n_draws"]), "prediction": pred, "line": launcher_line(row, sha=sha, cap=next_cap),
                "certification_needed": (f"nemotron_backend_probe at {IC.IFEVAL_GEOMETRY_ENTRY_BY_CAP[next_cap]} (sha {sha[:7]}) BEFORE this rung"
                                         if row["family"] == "nemotron_h" else None)})
    return res


def plan_all(rows: Sequence[Mapping[str, Any]], *, wave_root: Path, sha: str, from_cap: int, classify: Callable[..., tuple],
             served_seqs_of: Callable[[str], int]) -> list[dict[str, Any]]:
    return [plan_cell(r, wave_root=wave_root, sha=sha, from_cap=from_cap, classify=classify, served_seqs=served_seqs_of(r["model_key"]))
            for r in rows if r["status"] == IC.STATUS_LAUNCH]


def summary_lines(planned: Sequence[Mapping[str, Any]], *, s1_allowance: float) -> list[str]:
    out = []
    total_c = total_w = 0.0
    for p in planned:
        pred = p.get("prediction")
        size = (f" n_draws={p['n_draws']} tok/s={pred['tok_s']:.0f} GPU-h central={pred['gpu_h_central']:.3f} all-at-cap={pred['gpu_h_all_at_cap']:.3f}"
                if pred else "")
        store = f" store_exit={p['store_exit']}" if p.get("store_exit") not in (None, p.get("exit")) else ""
        out.append(f"LADDER: row={p['index']} {p['job_id']} from={p['from_cap']} status={p['status']} exit={p['exit']}{store} next={p['next_cap']}{size}"
                   + (f" note={p['note']}" if p.get("note") else "") + (f" NEEDS {p['certification_needed']}" if p.get("certification_needed") else ""))
        if pred:
            total_c += pred["gpu_h_central"]
            total_w += pred["gpu_h_all_at_cap"]
    verdict = "ADMIT" if total_c <= 2 * s1_allowance else "STOP FOR A LOOK (> 2x the S1 allowance)"
    out.append(f"RE-ESTIMATE: rung wave {len([p for p in planned if p.get('prediction')])} cells, predicted GPU-h central {total_c:.2f} "
               f"(every draw at cap {total_w:.2f}) vs 2 x S1 allowance {2 * s1_allowance:.2f} -> {verdict} ({IC.IFEVAL_TRIGGER_PROVENANCE})")
    return out


def launch_lines(planned: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [p["line"] for p in planned if p.get("line")]
    return lines or ["# no rung to launch"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--wave-root", type=Path, required=True)
    ap.add_argument("--sha", required=True, help="the pin the rung configurations run at (the stores' producer tree)")
    ap.add_argument("--from-cap", type=int, default=CAPS[0], choices=list(CAPS),
                    help="file THE decision for this cap's complete stores and plan the rung above it (at the ceiling: the terminal record only)")
    ap.add_argument("--rows", default=None, help="matrix indices to plan (i,j-k); default every launch row")
    ap.add_argument("--matrix", type=Path, default=IM.MATRIX_PATH,
                    help="the matrix document whose rows are planned (default: the frozen 40-row matrix; the batch-64 seed "
                         f"sibling is {IM.MATRIX_QWEN_SEEDREP_PATH.relative_to(IM.REPO)} — b64-capability-agentic-queue W2, D75)")
    ap.add_argument("--s1-allowance-gpu-h", type=float, default=S1_ALLOWANCE_GPU_H)
    ap.add_argument("--served-seqs", type=int, default=64, help="the served concurrency assumed for the prediction when the source leg's "
                                                                  "endpoint.json records none (the standup's S)")
    ap.add_argument("--out", type=Path, default=None, help="where to also write plan.json + launch_lines.txt")
    a = ap.parse_args(argv)
    rows = IM.matrix_rows(a.matrix)
    if a.rows:
        idx = set()
        for part in a.rows.split(","):
            lo, _, hi = part.partition("-")
            idx.update(range(int(lo), int(hi or lo) + 1))
        rows = [rows[i] for i in sorted(idx)]
        IM.refuse_held(rows, flag="--rows")

    def classify(root: Path, *, source_out_root: Path | None = None):
        return IR.classify_store(root, task=TASK, source_out_root=source_out_root)

    planned = plan_all(rows, wave_root=a.wave_root.resolve(), sha=a.sha, from_cap=a.from_cap, classify=classify, served_seqs_of=lambda k: a.served_seqs)
    lines = summary_lines(planned, s1_allowance=a.s1_allowance_gpu_h) + launch_lines(planned)
    for line in lines:
        print(line)
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(a.out / "plan.json", {"sha": a.sha, "from_cap": a.from_cap, "planned": planned, "lines": lines})
        (a.out / "launch_lines.txt").write_text("\n".join(launch_lines(planned)) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
