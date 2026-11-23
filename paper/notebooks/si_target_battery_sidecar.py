#!/usr/bin/env python3
"""paper/notebooks/si_target_battery_sidecar.py: the committed data behind the distillation-target battery figure.

    python paper/notebooks/si_target_battery_sidecar.py write                  # derive from the live registry, write data/qwen_si_target_battery_rows.json
    python paper/notebooks/si_target_battery_sidecar.py write --allow-pending  # the same, listing readings not yet in the data as `pending` (provisional)
    python paper/notebooks/si_target_battery_sidecar.py check                  # derive again and compare with the committed file

The figure (`figA_qwen_si_target_battery`) sets the eighteen objects distilled from a prompted teacher (three distillation targets, a
LoRA and a ReLU steer each, three fit seeds) beside the constitution corpus fits on every family of the main figures. Their runs stay
outside the paper's selection and carry no registry metric row (their judgments are the re-judge judges' alone), so the notebook reads
them from the file this script writes; `registry_nb.SITARGET_OBJECTS` pins every object, `registry_nb.SITARGET_STATIC_RUNS` every static
reading. The multi-turn attacks are not in this file: the multi-turn figure's own sidecar (`si_target_sidecar.py`) carries them, and the
battery figure's multi-turn panel reads them from there.

THE FILE IS THRESHOLD-FREE. One row per registered run of a pinned distilled object (campaign `si_target_refit`, Qwen3.5-4B) on the
battery's instruments (`registry_nb.SITARGET_BATTERY_EVALS`): its identity, its judgment and judge, its scoring rule, and the columns the
figure code reads; and beside each drawn run its per-item payload - for a StrongREJECT run the best score of every goal with its unit
and scored counts (`registry_nb.sitarget_sr_payload`), for an over-refusal run every item's label, subset and at-cap flag, for a
capability run every item's correctness (GSM8K's rung run merged item by item; a run with no item rows its filed counts) - each aligned
to its item set's ordered ids (`item_sets`, one space-separated string per item set). The notebook computes every reading from the
payload at its StrongREJECT threshold, so no figure needs a judge call to move threshold.

The judgments read are the eval's ONE JUDGE (`registry_nb.SITARGET_EVAL_JUDGE`: MiniMax-M3, StrongREJECT v1 for static and PAIR, v2 for
Best-of-75, the C/P/I rubrics for over-refusal): per run the NEWEST COMPLETE judgment of that judge; their item rows come through the
notebook's item reader (`registry_nb.JUDGING.items`). Each pinned judge must be the one the notebook's own policy lookup gives in this
registry (`registry_nb.sitarget_judge_problems`). A static cell holds one run per rung (the drawn reading is the final rung's, which must be
the run and judgment SITARGET_STATIC_RUNS pins; the lower rungs ride as `rung`); an over-refusal run holds its whole ladder and is read at
the rung its judgment records (`judged_cap`; XSTest on the 2,048 rung, or a lower rung only where the run's ladder exited there with no
reply at cap), which must be the rung its ladder exited on: the registry's record of the exit is read for EVERY over-refusal reading
and rides in its payload, where the contract checks it against the rung read; a GSM8K cell's rung run is merged into its base
run's reading, and a base run with no rung run is read at 8,192 only within the ladder's trigger: more than 1 % of its answers at the
cap (the replies that stopped on it, `item_flags`, and the answers the cap left unscored) leaves the reading pending its rung, unless
`registry_nb.SITARGET_GSM8K_NO_ELIGIBLE` pins the run as one whose ladder plan found every such answer a loop.

The SAME derivation is the notebook's check: a live render re-derives the file from the live registry and refuses on any difference.
A snapshot render reads the file alone, held to its contract, since the committed snapshot predates these runs.

`write` refuses, writing nothing, when a run breaks the figure's contract (`registry_nb.sitarget_battery_sidecar_problems`: the pins, the
instrument, item set, judge, scoring rule, caps), when a pinned judge is not the policy lookup's, when a reading has two candidate runs,
when a run of a distilled object sits on an eval the battery does not read, when a reading is not yet judged or (GSM8K) still waits for
its ladder's rung (unless --allow-pending, which lists both as pending),
when a payload cannot be built (a goal outside its item set, a per-goal best that is not its units' max, a label outside the
vocabulary), yields no reading, or - for StrongREJECT - counts a different k at 0.5 from the judge rules' own reading of the judgment's
item rows, or when the publication scan (`reproduce/tools/scrub_grep.sh --all`) finds anything in the new file or in this script: the
file is written beside the target, scanned, and moved into place only on a clean scan. `check` exits 0 when the committed file equals a
fresh derivation, 1 when it differs (each difference printed), 2 when it cannot read the registry or the file.

The provenance names the script and its sha256, the commit and whether the tracked tree differed from it (`git_dirty`), and the registry
head read; write from a committed tree. Needs the notebook environment (`uv sync --frozen --extra cpu --group notebook`) and the live
registry (REGISTRY_ROOT, default ~/registry). Nothing is written to the registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRUB = REPO / "reproduce" / "tools" / "scrub_grep.sh"
CAMPAIGN = "si_target_refit"
ABOUT = ("Per-run identity and per-item payload (StrongREJECT best score per goal with unit counts; over-refusal labels; capability "
         "correctness) of the Qwen3.5-4B objects distilled from a prompted teacher (safety, constitution and irrelevant-instruction "
         "targets; a LoRA and a ReLU steer each; three fit seeds) on the paper's static, Best-of-75, PAIR, over-refusal and capability "
         "instruments; every LLM verdict MiniMax-M3's; readings are computed from the payload at the notebook's threshold; regenerate "
         "with paper/notebooks/si_target_battery_sidecar.py.")
#: the judging environment the file is derived under, set before the notebook's reader is imported (as `si_target_sidecar.py` pins it)
PINNED_ENV = {"REGISTRY_NB_JUDGES": "rejudge", "REGISTRY_NB_SOURCE": "live"}
UNSET_ENV = ("REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_PETRI_EXCLUDE")
RUNS_SQL = ("SELECT r.run_id, r.eval_id, r.object_id, r.model_id, r.instrument_id, r.item_set_id, r.lifecycle_state, r.token_cap_tokens, "
            "r.token_cap_rung_index, r.pass_label, r.transfer_kind, r.serving_path, r.view_legacy, r.lane_source_legacy, r.trial_index, "
            "r.scoring_rule, i.bon_n, i.padding_tokens FROM run r LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
            "WHERE r.campaign = ? AND r.model_id = ?")
#: the paper view's hold-out flag of the discovered runs, in ONE read
HELD_SQL = "SELECT run_id, held_out FROM v_paper_current WHERE run_id IN ({marks})"
JUDGMENTS_SQL = ("SELECT j.judgment_id, j.run_id, j.judge_id, j.status, j.finished_utc, j.extra_json, g.judge_model, g.rubric_name, "
                 "g.rubric_version FROM judgment j JOIN judge g ON g.judge_id = j.judge_id WHERE j.run_id = ? AND j.judge_id = ?")
#: a rule-scored run's item rows from the `item_result` TABLE, withdrawn rows left out (by either withdrawal record) and one row kept per
#: item key by the notebook's own revision rule (`registry_nb.current_items`)
RULE_ITEMS_SQL = ("SELECT ir.run_id, ir.judgment_id, ir.item_id, ir.metric_id, ir.trial_index, ir.sample_index, ir.turn_index, ir.value, "
                  "ir.label, NULL AS subset, ir.written_utc AS item_written_utc, substr(hex(ir.record_sha256), 1, 16) AS item_order_sha "
                  "FROM item_result ir WHERE ir.run_id = ? AND ir.metric_id IN ('acc', 'unscored') AND ir.withdrawn_by_revision_id IS NULL "
                  "AND ir.withdrawn_by_record_sha256 IS NULL")
FLAGS_SQL = "SELECT item_id, trial_index, at_token_cap FROM item_flags WHERE run_id = ?"
#: THE REPLIES THAT STOPPED ON THE CAP, the ladder's own definition of at-cap (a stop on the request's cap, whatever the scorer then made
#: of the text): a GSM8K base run's trigger reads these with the answers the cap left unscored
CUT_SQL = ("SELECT DISTINCT item_id FROM item_flags WHERE run_id = ? AND trial_index = 0 "
           "AND (finish_reason IN ('max_tokens', 'length') OR at_token_cap = 1)")
METRIC_SQL = "SELECT value, k, n FROM run_metric WHERE run_id = ? AND metric_id = 'acc'"
CONTRAST_SQL = "SELECT DISTINCT anchor_run_id FROM run_contrast WHERE run_id = ?"
#: WHERE AN OVER-REFUSAL RUN'S CAP LADDER EXITED: the `generated` state filed when the run's store was closed (its evidence is the
#: store's completion artifact, the ladder's own record), with the store the run names. The state's note ends in the closing facts
LADDER_EXIT_SQL = ("SELECT r.store_uri, e.note FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE e.run_id = ? "
                   "AND e.state = 'generated' AND e.evidence_kind = 'completion_artifact' AND e.evidence_uri = r.store_completion_artifact_uri "
                   "ORDER BY e.event_utc")
LIVE = ("generated", "judged", "ingested")


def _R():
    """The notebook's reader library on the LIVE registry: the module already imported, when the notebook's live check calls in; else
    imported here under the pinned judging environment (PINNED_ENV)."""
    if "registry_nb" in sys.modules:
        return sys.modules["registry_nb"]
    os.environ.update(PINNED_ENV)
    for k in UNSET_ENV:
        os.environ.pop(k, None)
    os.environ.setdefault("MPLBACKEND", "Agg")
    # a reader under a policy that is not the paper's refuses the paper's figures directory; this script draws nothing
    os.environ.setdefault("REGISTRY_NB_FIGS", tempfile.mkdtemp(prefix="sitarget-battery-figs-"))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import registry_nb
    return registry_nb


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _connect():
    """The live registry, which `write`, `check` and a live render's check derive the file from."""
    R = _R()
    if R.SOURCE != "live":
        raise RuntimeError(f"the sidecar is derived from the live registry; this reader is on the {R.SOURCE}")
    return R.connect()


def _rule_items(con, run_id: str):
    """A rule-scored run's current item rows (`RULE_ITEMS_SQL`, `registry_nb.current_items`)."""
    return _R().current_items(_frame(con, RULE_ITEMS_SQL, (run_id,)))


def cut_items(con, run_id: str) -> set:
    """The item ids of a run's replies that stopped on the cap (`CUT_SQL`)."""
    return {str(i) for (i,) in con.execute(CUT_SQL, (run_id,)).fetchall()}


def _frame(con, sql: str, args):
    import pandas as pd
    return pd.read_sql_query(sql, con, params=list(args))


def _extra(raw) -> dict:
    try:
        v = json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        v = {}
    return v if isinstance(v, dict) else {}


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def discover(con) -> tuple:
    """(readings, problems): the campaign's runs of the pinned distilled objects on the battery's evals that are generated or complete (a
    queued, running, failed or cancelled run is no reading yet), as {(slot, seed, eval): [run dicts]}. A run of a distilled object on an
    eval neither the battery nor the multi-turn figure reads is a problem; the multi-turn runs are the multi-turn sidecar's."""
    R = _R()
    rows = [r for r in _frame(con, RUNS_SQL, (CAMPAIGN, R.MODEL_ID)).to_dict("records") if r["lifecycle_state"] in LIVE]
    refit = {o: v for o, v in R.SITARGET_OBJECTS.items() if v[0] in R.SITARGET_NEW_SLOTS}
    mine = [r for r in rows if r["object_id"] in refit]
    held = {}
    ids = [r["run_id"] for r in mine]
    for i in range(0, len(ids), 400):
        part = ids[i:i + 400]
        held.update(dict(con.execute(HELD_SQL.format(marks=",".join("?" * len(part))), part).fetchall()))
    readings, probs = {}, []
    for r in mine:
        r["held_out"] = held.get(r["run_id"])
        slot, seed = refit[r["object_id"]]
        if r["eval_id"] in R.SITARGET_EVALS:
            continue
        if r["eval_id"] not in R.SITARGET_BATTERY_EVALS:
            probs.append(f"{r['run_id']}: a run of a distilled object on {r['eval_id']}, which the battery figure does not read")
            continue
        readings.setdefault((slot, seed, r["eval_id"]), []).append(r)
    return readings, probs


def policy_problems() -> list:
    """Every judged battery eval whose pinned judge is not the one the notebook's policy lookup names in this registry."""
    R = _R()
    return R.sitarget_judge_problems(tuple(e for e in R.SITARGET_BATTERY_EVALS if e in R.SITARGET_EVAL_JUDGE))


def _is_rung(r: dict) -> bool:
    """A GSM8K run is the ladder's rung run when the registry says so (rung index 1, the rung pass label, or the top cap)."""
    return (_f(r.get("token_cap_rung_index")) == 1.0 or r.get("pass_label") == "ladder_rung" or _f(r.get("token_cap_tokens")) == 16384.0)


def _judgment(con, run: dict, judge_id: str):
    """The NEWEST COMPLETE judgment of `judge_id` on the run, or None."""
    js = _frame(con, JUDGMENTS_SQL, (run["run_id"], judge_id))
    js = js[js.status.eq("complete")].sort_values("finished_utc")
    return js.iloc[-1].to_dict() if len(js) else None


def _row(key, run: dict, role: str, judge, cap, rung_run_id=None, anchor=None) -> dict:
    slot, seed, e = key
    j = judge or {}
    return dict(run_id=run["run_id"], object_id=run["object_id"], slot=slot, seed=seed, eval_id=e, role=role, model_id=run["model_id"],
                instrument_id=run["instrument_id"], item_set_id=run["item_set_id"], lifecycle_state=run["lifecycle_state"],
                judgment_id=j.get("judgment_id"), judge_id=j.get("judge_id"), judge_model=j.get("judge_model"),
                rubric_name=j.get("rubric_name"), rubric_version=j.get("rubric_version"), scoring_rule=run.get("scoring_rule"),
                token_cap_tokens=cap, run_token_cap_tokens=_f(run["token_cap_tokens"]), pass_label=run.get("pass_label"),
                lane=run.get("transfer_kind"), dose=1.0, pad=(_f(run.get("padding_tokens")) or 0.0), pad_contract=False,
                serving_path=run.get("serving_path"), bon_n=_f(run.get("bon_n")), anchor_run_id=anchor,
                lane_source_legacy=run.get("lane_source_legacy"), view_legacy=run.get("view_legacy"),
                held_out=(_f(run.get("held_out")) or 0.0), rung_run_id=rung_run_id)


def _plan(R, key, runs: list, probs: list):
    """[(run, role, the drawn cap or None, the merged rung run or None)] of one reading's runs, or None (a problem was filed)."""
    e = key[2]
    spec = R.SITARGET_BATTERY_EVALS[e]
    if spec["drawn_cap"] == "run":                                       # a static cell: one run per rung, the final rung drawn
        by_cap = {}
        for r in runs:
            by_cap.setdefault(_f(r["token_cap_tokens"]), []).append(r)
        dup = {c: [r["run_id"] for r in rs] for c, rs in by_cap.items() if len(rs) > 1}
        if dup or None in by_cap:
            probs.append(f"{R.sitarget_battery_key(*key)}: two runs at one rung, or a run with no cap ({dup})")
            return None
        top = max(by_cap)
        return [(by_cap[c][0], "drawn" if c == top else "rung", c, None) for c in sorted(by_cap)]
    if e == "gsm8k":
        base = [r for r in runs if not _is_rung(r)]
        rung = [r for r in runs if _is_rung(r)]
        if len(base) != 1 or len(rung) > 1:
            probs.append(f"{R.sitarget_battery_key(*key)}: {len(base)} base and {len(rung)} rung runs (one base, at most one rung)")
            return None
        cap = R.SITARGET_GSM8K_RUNGS[-1] if rung else R.SITARGET_GSM8K_RUNGS[0]
        return [(base[0], "drawn", cap, rung[0] if rung else None)] + [(r, "rung", _f(r["token_cap_tokens"]), None) for r in rung]
    if len(runs) != 1:
        probs.append(f"{R.sitarget_battery_key(*key)}: {len(runs)} runs ({', '.join(r['run_id'] for r in runs)}) for one reading")
        return None
    return [(runs[0], "drawn", None, None)]


def _count(v):
    """A whole, non-negative number as an int, or None (a number only: no text is read as a count)."""
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < float("inf") and v == int(v)
    return int(v) if ok else None


def _exit_facts(note) -> dict | None:
    """The facts a `generated` state's note ends in, or None: the note is prose naming the store, then `: ` and the facts as one JSON
    object; a note with no such block, or one cut short of its end, gives None."""
    at = note.find(": {", max(note.find(" from "), 0)) if isinstance(note, str) else -1
    try:
        facts = json.loads(note[at + 2:]) if at >= 0 else None
    except ValueError:
        return None
    return facts if isinstance(facts, dict) else None


def ladder_exit(con, run_id: str, judge: dict) -> tuple:
    """({rung, n_at_cap, n}, None): the rung the run's cap ladder exited on, how many replies sat at the cap there and how many were
    counted - or (None, why no such record can be read). THE SOURCE IS THE REGISTRY'S RECORD OF THE LADDER'S CLOSE: the `generated` state
    filed from the completion artifact of the store the run names (LADDER_EXIT_SQL); its closing facts name the final rung (`final_rung`)
    and the store's own count of replies at cap (`sidecars`: n_at_cap of n_total), beside the count re-derived when it was closed
    (`rederived`), which must agree. The count is the closed store's, so the judgment must name that store as the one it judged
    (`extra.judged_store_origin`)."""
    found = con.execute(LADDER_EXIT_SQL, (run_id,)).fetchall()
    if not found:
        return None, "the registry holds no record of where its ladder exited (no `generated` state filed from its store's completion artifact)"
    recs = set()
    for store, note in found:
        facts = _exit_facts(note)
        if facts is None:
            return None, "the registry's record of where its ladder exited carries no readable closing facts"
        side = facts.get("sidecars") if isinstance(facts.get("sidecars"), dict) else {}
        rec = (_count(facts.get("final_rung")), _count(side.get("n_at_cap")), _count(side.get("n_total")))
        if None in rec:
            return None, "the registry's record of where its ladder exited names no final rung, or no count of the replies at cap there"
        again = facts.get("rederived") if isinstance(facts.get("rederived"), dict) else {}
        if "n_at_cap" in again and _count(again["n_at_cap"]) != rec[1]:
            return None, f"the registry's record of where its ladder exited holds two counts of the replies at cap ({rec[1]} and {again['n_at_cap']!r})"
        if not store or _extra(judge.get("extra_json")).get("judged_store_origin") != store:
            return None, "its judgment does not name the store its ladder exited in as the store it judged"
        recs.add(rec)
    if len(recs) > 1:
        return None, f"the registry holds {len(recs)} differing records of where its ladder exited"
    (rung, n_at_cap, n), = recs
    return dict(rung=rung, n_at_cap=n_at_cap, n=n), None


def judged_cap(spec: dict, eval_id: str, run_id: str, judge: dict, exit_record=None) -> tuple:
    """(the rung an over-refusal judgment judged, None), read from the judgment's own record (`extra.judged_cap`), or (None, why the reading
    is refused): no rung recorded, or - for an eval read on one fixed rung (XSTest) - a rung above it, or one below it unless the run's
    ladder exited at the rung judged with no reply at cap (`registry_nb.sitarget_exit_problem` on `exit_record`)."""
    cap = _f(_extra(judge.get("extra_json")).get("judged_cap"))
    if cap is None:
        return None, f"{run_id}: its judgment {judge['judgment_id']} records no judged_cap (the rung it judged)"
    if "judged_cap" in spec and cap != spec["judged_cap"]:
        why = f"{run_id}: its judgment {judge['judgment_id']} judged the {cap:,.0f} rung, not the {spec['judged_cap']:,.0f} rung {eval_id} is read on"
        if cap > spec["judged_cap"]:
            return None, why
        rec, unread = exit_record or (None, None)
        problem = unread or _R().sitarget_exit_problem(spec, cap, rec)
        if problem:
            return None, f"{why}: {problem}"
    return cap, None


def gsm8k_hold(R, row: dict, entry: dict, ids: list, cut: set) -> str | None:
    """Why a GSM8K reading at 8,192 with no rung run merged in is held as pending (`registry_nb.sitarget_gsm8k_ladder_problem` on its
    answers at the cap: the replies that stopped on it, `cut`, and the answers its payload leaves unscored), or None for any other
    reading."""
    if row["eval_id"] != "gsm8k" or row["role"] != "drawn" or row["rung_run_id"] is not None or entry.get("kind") != "acc_items":
        return None
    at_cap = set(cut) | {i for i, f in zip(ids, entry["base_at_cap"]) if f}
    why = R.sitarget_gsm8k_ladder_problem(row["run_id"], at_cap, R.SITARGET_BATTERY_EVALS["gsm8k"]["n_items"])
    return f"{R.sitarget_battery_key(row['slot'], row['seed'], 'gsm8k')} ({row['run_id']}): {why}" if why else None


def derive(allow_pending: bool = False, con=None) -> tuple:
    """(sidecar dict, problems): every battery reading from the live registry (or `con`), with its payload, in canonical form."""
    R = _R()
    own = con is None
    con = con or _connect()
    try:
        head = int(con.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
        readings, probs = discover(con)
        probs += policy_problems()
        rows, built, ids_by_set, held = [], [], {}, []
        for key, runs in sorted(readings.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            spec = R.SITARGET_BATTERY_EVALS[key[2]]
            want = R.SITARGET_EVAL_JUDGE.get(key[2]) if spec["family"] != "rule" else None
            for run, role, cap, rung_run in _plan(R, key, runs, probs) or []:
                judge = _judgment(con, run, want.judge_id) if want is not None else None
                if want is not None and judge is None:
                    continue                                             # not yet judged: a drawn reading without a row is pending
                exit_rec = None
                if spec["drawn_cap"] == "judged":
                    exit_record = ladder_exit(con, run["run_id"], judge)
                    cap, why = judged_cap(spec, key[2], run["run_id"], judge, exit_record)
                    if why:
                        probs.append(why)
                        continue
                    if exit_record[0] is None:                           # every over-refusal reading is its ladder's final rung's
                        probs.append(f"{run['run_id']}: {exit_record[1]} (an over-refusal reading is read at its ladder's final rung, "
                                     "which that record names)")
                        continue
                    exit_rec = exit_record[0]                            # rides in the payload; the contract checks it against the rung read
                elif cap is None:
                    cap = spec["drawn_cap"] if isinstance(spec["drawn_cap"], float) else _f(run["token_cap_tokens"])
                anchors_reg = [a for (a,) in con.execute(CONTRAST_SQL, (run["run_id"],)).fetchall() if a]
                if len(anchors_reg) > 1:
                    probs.append(f"{run['run_id']}: {len(anchors_reg)} registry anchors")
                    continue
                row = _row(key, run, role, judge, cap, rung_run["run_id"] if rung_run is not None else None, anchors_reg[0] if anchors_reg else None)
                rows.append(row)
                if role != "drawn":
                    continue
                ref = rung_its = flags = None
                cut = set()
                if spec["family"] == "rule":
                    its = _rule_items(con, run["run_id"])
                    rung_its = _rule_items(con, rung_run["run_id"]) if rung_run is not None else None
                    ref = _frame(con, METRIC_SQL, (run["run_id"],))
                    if key[2] == "gsm8k" and rung_run is None:
                        cut = cut_items(con, run["run_id"])
                else:
                    items = R.JUDGING.items([judge["judgment_id"]], fields=("subset",))
                    its = items[items["run_id"].astype(str).eq(run["run_id"]) & items["judgment_id"].astype(str).eq(judge["judgment_id"])]
                if spec["family"] == "over_refusal":
                    fl = _frame(con, FLAGS_SQL, (run["run_id"],))
                    fl = fl[fl.trial_index.astype(int).eq(0)]
                    flags = ({str(i): bool(v) for i, v in zip(fl.item_id, fl.at_token_cap) if v == v}
                             if len(fl) and not fl.item_id.duplicated().any() else {})
                ids_by_set.setdefault(spec["item_set"], set()).update(str(x) for x in its.item_id)
                built.append((row, spec, its, rung_its, flags, ref, exit_rec, cut))
        lists = {k: sorted(v) for k, v in ids_by_set.items()}
        items_out = {}
        JR = R._judge_rules()
        for row, spec, its, rung_its, flags, ref, exit_rec, cut in built:
            ids = lists[spec["item_set"]]
            try:
                if spec["family"] == "strongreject":
                    entry = R.sitarget_sr_payload(its, ids)
                elif spec["family"] == "over_refusal":
                    entry = R.sitarget_cpi_payload(its, ids, row["token_cap_tokens"], spec.get("subset"), flags, exit_rec)
                elif len(its):
                    entry = R.sitarget_acc_payload(its, ids, rung_its, row["rung_run_id"])
                elif ref is not None and len(ref) and row["rung_run_id"] is None and _count(_f(ref.k.iat[0])) is not None:
                    entry = R.sitarget_acc_run_payload(int(_f(ref.k.iat[0])), int(_f(ref.n.iat[0])))
                else:
                    raise ValueError("a capability run with neither item rows nor a filed acc row to read")
            except ValueError as err:
                probs.append(f"{row['run_id']}: {err}")
                continue
            try:
                red = R.sitarget_battery_reduce(spec["family"], entry, R.SR_PAPER_THRESHOLD)
            except (ValueError, JR.NotDerivable) as err:
                probs.append(f"{row['run_id']}: its payload yields no reading ({err})")
                continue
            if not red.n:
                probs.append(f"{row['run_id']}: its payload yields no reading (no determined goal, or no judged item)")
                continue
            why = gsm8k_hold(R, row, entry, ids, cut)
            if why:                                                     # a reading its ladder has not finished: pending, not drawn
                held.append(why)
                continue
            if spec["family"] == "strongreject":
                # THE POPCOUNT AT 0.5 IS THE JUDGE RULES' OWN: the payload's k against `judge_rules`' reading of the judgment's item rows
                own = R.JUDGING.reduce("strongreject", its, R.SR_PAPER_THRESHOLD)
                if (float(own.k), float(own.n)) != (float(red.k), float(red.n)):
                    probs.append(f"{row['run_id']}: its payload counts {red.k:g} of {red.n:g} at 0.5, the judgment's item rows {own.k:g} of {own.n:g}")
            if spec["family"] == "rule" and ref is not None and len(ref) and row["rung_run_id"] is None and entry["kind"] == "acc_items":
                if abs(float(ref.value.iat[0]) - red.value) > 1e-9:
                    probs.append(f"{row['run_id']}: the payload's accuracy {red.value:.6f} is not the filed acc {float(ref.value.iat[0]):.6f}")
            items_out[row["run_id"]] = R.sitarget_battery_serialize_items({**entry, "item_set": spec["item_set"]})
        for name, lst in lists.items():
            want_n = next(s["n_items"] for s in R.SITARGET_BATTERY_EVALS.values() if s["item_set"] == name)
            if len(lst) != want_n and any(items_out.get(r["run_id"], {}).get("kind") != "acc_run" for r, s, *_ in built if s["item_set"] == name):
                probs.append(f"item list {name}: the runs name {len(lst)} items, not {want_n}")
    finally:
        if own:
            con.close()
    drawn_ok = {r["run_id"] for r in rows if r["role"] == "drawn" and r["run_id"] in items_out}
    keys_ok = {(r["slot"], r["seed"], r["eval_id"]) for r in rows if r["run_id"] in drawn_ok}
    rows = R.sitarget_battery_serialize([r for r in rows if r["run_id"] in drawn_ok or (r["role"] == "rung" and (r["slot"], r["seed"], r["eval_id"]) in keys_ok)])
    pending = R.sitarget_battery_expected_pending(rows)
    if held:
        print("pending its ladder's rung (not drawn): " + "; ".join(held), file=sys.stderr)
    if pending and not allow_pending:
        probs.append(f"{len(pending)} reading(s) not yet in the data (write --allow-pending for a provisional file): "
                     + ", ".join(R.sitarget_battery_key(p["slot"], p["seed"], p["eval_id"]) for p in pending[:8]) + (" ..." if len(pending) > 8 else ""))
        probs += held
    try:
        git = lambda *a: subprocess.check_output(["git", *a], cwd=REPO, text=True, stderr=subprocess.DEVNULL)
        commit = git("rev-parse", "HEAD").strip()
        target = str(Path(R.SITARGET_BATTERY_SIDECAR).resolve().relative_to(REPO))
        dirty = any(ln[3:].strip() != target for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        commit, dirty = None, None
    used = {items_out[r["run_id"]]["item_set"] for r in rows if r["run_id"] in items_out and items_out[r["run_id"]].get("kind") != "acc_run"}
    out = dict(schema=R.SITARGET_BATTERY_SCHEMA, about=ABOUT,
               provenance=dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               script=str(Path(__file__).resolve().relative_to(REPO)), script_sha256=_sha(Path(__file__).resolve()),
                               git_commit=commit, git_dirty=dirty, registry_head=head, judge_policy=R.SITARGET_JUDGE_POLICY),
               objects=R.sitarget_battery_objects_json(),
               runs={r["run_id"]: {c: r[c] for c in ("object_id", "slot", "seed", "eval_id", "role")} for r in rows})
    if pending:
        out["pending"] = pending
    out["rows"] = rows
    out["item_sets"] = {k: " ".join(v) for k, v in sorted(lists.items()) if k in used}
    out["items"] = {rid: items_out[rid] for rid in sorted(items_out) if rid in out["runs"]}
    if not probs:
        probs += R.sitarget_battery_sidecar_problems(out, allow_pending=allow_pending)
    return out, probs


def dumps(out: dict) -> str:
    """The file's text: valid JSON with one line per top-level key, per row, per pending reading and per run's payload, so a
    regeneration's diff names the runs that moved."""
    one = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    parts = []
    for k, v in out.items():
        if isinstance(v, list) and k in ("rows", "pending"):
            body = "[\n" + ",\n".join(one(x) for x in v) + "\n]" if v else "[]"
        elif isinstance(v, dict) and k in ("runs", "items", "item_sets", "objects"):
            body = "{\n" + ",\n".join(f"{one(rid)}:{one(e)}" for rid, e in v.items()) + "\n}" if v else "{}"
        else:
            body = one(v)
        parts.append(f"{one(k)}:{body}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def scrub_clean(paths: list) -> tuple:
    """(clean, report): the publication scan over `paths` (repository-relative). `scrub_grep.sh` exits 0 whatever it finds, so the
    verdict is read from its summary line: `scrub: no identifier under ...` is clean, anything else is not."""
    env = dict(os.environ, SCRUB_PYTHON=sys.executable)
    p = subprocess.run(["bash", str(SCRUB), "--all", *paths], cwd=REPO, env=env, capture_output=True, text=True)
    report = (p.stdout + p.stderr).strip()
    last = report.splitlines()[-1] if report else ""
    return p.returncode == 0 and last.startswith("scrub: no identifier under"), report


def cmd_write(allow_pending: bool, out_path: str | None = None) -> int:
    R = _R()
    target = Path(out_path) if out_path else Path(R.SITARGET_BATTERY_SIDECAR)
    out, problems = derive(allow_pending)
    if problems:
        print("REFUSED: nothing written.\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    tmp = target.with_name("." + target.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(dumps(out), encoding="utf-8")
    try:
        scan = [str(tmp.resolve().relative_to(REPO))]
    except ValueError:                                   # a provisional file written outside the repository: scan this script only
        scan = []
    clean, report = scrub_clean(scan + [str(Path(__file__).resolve().relative_to(REPO))])
    if not clean:
        tmp.unlink(missing_ok=True)
        print("REFUSED: the publication scan is not clean on the new file or this script; nothing written.\n" + report, file=sys.stderr)
        return 1
    os.replace(tmp, target)
    print(f"wrote {target}: {len(out['rows'])} rows ({sum(r['role'] == 'drawn' for r in out['rows'])} drawn), {len(out.get('pending', []))} "
          f"pending; registry head {out['provenance']['registry_head']}; {target.stat().st_size / 1e3:,.0f} kB; publication scan clean")
    return 0


def cmd_check(path: str | None = None) -> int:
    R = _R()
    path = path or R.SITARGET_BATTERY_SIDECAR
    try:
        with open(path, encoding="utf-8") as fh:
            committed = json.load(fh)
    except (OSError, ValueError) as err:
        print(f"cannot read {path}: {err}", file=sys.stderr)
        return 2
    try:
        fresh, problems = derive(allow_pending=True)
    except (FileNotFoundError, OSError, RuntimeError) as err:
        print(f"cannot read the live registry: {err}", file=sys.stderr)
        return 2
    diffs = []
    if committed.get("schema") != fresh.get("schema") or committed.get("objects") != fresh.get("objects"):
        diffs.append("`schema` or `objects` differs")
    mism = R.sitarget_battery_check(committed, fresh)
    if len(mism):
        diffs.append("rows differ:\n" + mism.to_string(index=False))
    if problems:
        diffs.append("a fresh derivation would be refused: " + "; ".join(problems))
    if diffs:
        print("DIFFERS:\n  " + "\n  ".join(diffs))
        return 1
    print(f"EQUAL: {len(fresh['rows'])} rows, {len(fresh.get('pending', []))} pending, registry head {fresh['provenance']['registry_head']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="derive from the live registry and write the sidecar")
    w.add_argument("--allow-pending", action="store_true", help="list readings not yet in the data as pending (a provisional file)")
    w.add_argument("--out", metavar="FILE", help="write here instead of the committed path (a provisional file for a scratch render)")
    c = sub.add_parser("check", help="derive again and compare with the committed file (0 equal, 1 differs, 2 cannot read)")
    c.add_argument("--path", metavar="FILE", help="compare this file instead of the committed one")
    a = ap.parse_args(argv)
    return cmd_write(a.allow_pending, a.out) if a.cmd == "write" else cmd_check(a.path)


if __name__ == "__main__":
    sys.exit(main())
