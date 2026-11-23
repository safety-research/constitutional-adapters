#!/usr/bin/env python3
"""paper/notebooks/ctlstop_sidecar.py: the committed data behind the control-pool stopping appendix figures.

    python paper/notebooks/ctlstop_sidecar.py write                  # derive from the live registry, write data/qwen_ctlstop_rows.json
    python paper/notebooks/ctlstop_sidecar.py write --allow-pending  # the same, listing readings not yet judged as `pending` (provisional)
    python paper/notebooks/ctlstop_sidecar.py check                  # derive again and compare with the committed file
    python paper/notebooks/ctlstop_sidecar.py needs --out FILE       # the main figures' runs the appendix draws, under REGISTRY_NB_SOURCE

The four figures (`figA_qwen_ctlstop_static`, `figA_qwen_ctlstop_capabilities`, `figA_qwen_ctlstop_multi_turn`,
`figA_qwen_ctlstop_rule_comparison`) draw runs that stay outside the paper's selection: the runs of the control LoRA and control ReLU
steer refit to stop on the control corpus's own held-out pool and of the difference objects rebuilt from them, on every family of the
main figures; the like-for-like reruns of the published difference ReLU steer on the multi-turn instruments; and the unsteered model
rerun in the same setup. The notebook reads them from the file this script writes; `registry_nb.CTLSTOP_OBJECTS` pins every object.

THE FILE IS THRESHOLD-FREE. One row per registered run of a pinned object (campaign `reseed_ctlstop`, the model's runs on the
appendix's instruments): its identity, its judgment and judge, and the columns the figure code reads, and beside it the run's per-item
payload - for a StrongREJECT run the best score of every (goal, trial) with its unit and scored counts, for an over-refusal run every
item's label, subset and at-cap flag, for a capability run every item's correctness (GSM8K's rung run merged item by item), each
aligned to its item set's ordered ids (`item_sets`, one space-separated string per item set: some goal names read to a secret scanner
as a key and its value when listed as JSON strings). The
notebook computes every reading from the payload at its StrongREJECT threshold, so no figure needs a judge call to move threshold.

The judgments read are the ones the appendix's judge policy names (`registry_nb.ctlstop_judging.policy`: the notebook's re-judge
policy, under each eval's own rubric and version): per run the NEWEST COMPLETE judgment of that judge; their item rows come through
the same interface's item reader (`registry_nb.judge_items`: the `item_result` table's current rows). Each pinned judge id must be
the one the notebook's own lookup (`registry_nb.judge_for`) gives in this registry. A static cell
holds one run per rung (the drawn reading is the final rung's, the others ride as `rung`); an over-refusal run holds its whole ladder and
is read at the rung its judgment records (`judged_cap`); a GSM8K cell's rung run is merged into its base run's reading. XSTest is read
on the 2,048 rung; a judgment of a lower rung is read only where the run's ladder exited at that rung with no reply at cap, on the
registry's record of the exit (`ladder_exit`: the `generated` state the run's store was closed with), which then rides in the run's
payload. Such a reading is that rung's own, named at that rung.

THE MAIN FIGURES' RUNS THE APPENDIX DRAWS ride in the same file under the same judge (`main`, `main_items`): the five objects not
refit, the published fits of the four slots and every unsteered run a point is differenced against. WHICH runs is read off the
notebook's own frames (`needs`: this script executes the notebook's header and load cells and Figure 3's anchoring, and takes the rows
the main figures draw on the appendix's judged evals and the unsteered runs they pool, `registry_nb.ctlstop_main_needs`), once on the
committed snapshot and once on the live registry, so a snapshot render and a live one both find their runs; `write` runs both (the
live load takes about ten minutes) unless `--needs FILE ...` names lists a `needs` run already wrote. Each run is read from the newest
complete judgment of the appendix's judge of the eval's rubric (where both StrongREJECT versions judged a run, the version of the
run's own filed judge, as the notebook's resolver picks it); its payload is the best score of every (goal, trial) and the goals
holding an unscored unit, or the label of every (item, trial), on its own item set's ids. A run with no such judgment, and a
strength-lane XSTest run (whose reading under another judge is the battery record's per-item reads, not the judgment's item rows),
is named in `main_pending` with that reason and read by no figure.

A FINAL FILE MAY NAME SUCH RUNS ONLY AS DECLARED ONES (`registry_nb.ctlstop_main_undeclared`): every entry's reason is one of those
two, and the live registry's list names none of them, so a live render reads every run it draws and only a render of the committed
snapshot leaves those points out, which it names. The two lists ride in the file (`main_needs`, by source: the registry head the list
was read at, the sha256 of the list as `needs` writes it, its run ids), so the file itself shows which list names which run. `write`
refuses any other unread run - another reason, a run the live list names - unless --allow-pending.

THE DECLARED ANCHORS (`anchor_declared`, from ANCHOR_DECLARED below): the static readings the notebook differences against the static
panel's unsteered runs at a cap below their own rung, each keyed by its run id with its reading (rule, slot, seed, eval), its rung,
that anchor cap and the reason. Only those rows: a static reading above every comparator's cap that is not declared is drawn as a
named gap. `write` refuses a declaration whose run is not the drawn reading it names, read at its rung (a declared row that does not
exist); the notebook holds each frame to it (`registry_nb.ctlstop_declared_anchor_problems`).

The SAME derivation is the notebook's live check: a live render calls `derive` on the file's own runs and refuses on any difference
from the committed file, or on a derivation the registry no longer supports.

`write` refuses, writing nothing, when a run breaks the appendix's contract (`registry_nb.ctlstop_sidecar_problems`: the pins, the
instrument, item set, judge, caps), when a pinned judge is not the one the registry resolves for its eval, when a reading has two
candidate runs, when an unsteered run of the campaign is not a drift anchor, when an XSTest judgment judged a rung above the one the
notebook reads, or one below it without the registry's record that the run's ladder exited there with no reply at cap, when a
reading is not yet judged or a main-figure run is left unread and is not a declared one (unless --allow-pending), when a payload
cannot be built (a goal outside its item set, a per-goal best that is not its units' max, a label outside the vocabulary) or yields
no reading (no determined goal, no judged item), or when the publication scan (`reproduce/tools/scrub_grep.sh --all`) finds anything
in the new file or in this script: the file is written beside the target, scanned, and moved into place only on a clean scan.
`check` exits 0 when the committed file equals a fresh derivation, 1 when it differs (each difference printed), 2 when it cannot
read the registry or the file.

The provenance names the script and its sha256, the commit and whether the tracked tree differed from it (`git_dirty`), and the
registry head read; write from a committed tree. Needs the notebook environment (`uv sync --frozen --extra cpu --group notebook`) and
the live registry (REGISTRY_ROOT, default ~/registry). Nothing is written to the registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRUB = REPO / "reproduce" / "tools" / "scrub_grep.sh"
CAMPAIGN = "reseed_ctlstop"
ABOUT = ("Per-run identity and per-item payload (StrongREJECT best score per goal with unit counts; over-refusal labels; capability "
         "correctness) of the Qwen3.5-4B control objects refit to stop on the control corpus's own held-out pool and the difference "
         "objects rebuilt from them, on the paper's static, multi-query, multi-turn, over-refusal, MMLU and GSM8K instruments, with the "
         "like-for-like reruns of the published difference ReLU steer and of the unsteered model on the multi-turn instruments; every "
         "LLM verdict MiniMax-M3's; readings are computed from the payload at the notebook's threshold; regenerate with "
         "paper/notebooks/ctlstop_sidecar.py.")
RUNS_SQL = ("SELECT r.run_id, r.eval_id, r.object_id, r.model_id, r.instrument_id, r.item_set_id, r.lifecycle_state, r.token_cap_tokens, "
            "r.token_cap_rung_index, r.pass_label, r.transfer_kind, r.serving_path, r.view_legacy, r.lane_source_legacy, r.trial_index, "
            "i.bon_n, i.padding_tokens FROM run r LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
            "WHERE r.campaign = ? AND r.model_id = ?")
#: the paper view's hold-out flag of the discovered runs, in ONE read (the view is costly; joined into the run query it is computed whole)
HELD_SQL = "SELECT run_id, held_out FROM v_paper_current WHERE run_id IN ({marks})"
JUDGMENTS_SQL = ("SELECT j.judgment_id, j.run_id, j.judge_id, j.status, j.finished_utc, j.extra_json, g.judge_model, g.rubric_name, "
                 "g.rubric_version FROM judgment j JOIN judge g ON g.judge_id = j.judge_id WHERE j.run_id = ? AND j.judge_id = ?")
RULE_ITEMS_SQL = ("SELECT ir.run_id, ir.judgment_id, ir.item_id, ir.metric_id, ir.trial_index, ir.sample_index, ir.turn_index, ir.value, "
                  "ir.label, NULL AS subset FROM v_item_result_current ir WHERE ir.run_id = ? AND ir.metric_id IN ('acc', 'unscored')")
FLAGS_SQL = "SELECT item_id, trial_index, at_token_cap FROM item_flags WHERE run_id = ?"
METRIC_SQL = "SELECT value, k, n FROM run_metric WHERE run_id = ? AND metric_id = 'acc'"
CONTRAST_SQL = "SELECT DISTINCT anchor_run_id FROM run_contrast WHERE run_id = ?"
MAIN_RUN_SQL = "SELECT run_id, eval_id, item_set_id, lane_source_legacy, judge_id FROM run WHERE run_id = ?"
#: WHERE AN OVER-REFUSAL RUN'S CAP LADDER EXITED: the `generated` state filed when the run's store was closed (its evidence is the
#: store's completion artifact, the ladder's own record), with the store the run names. The state's note ends in the closing facts
LADDER_EXIT_SQL = ("SELECT r.store_uri, e.note FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE e.run_id = ? "
                   "AND e.state = 'generated' AND e.evidence_kind = 'completion_artifact' AND e.evidence_uri = r.store_completion_artifact_uri "
                   "ORDER BY e.event_utc")
#: THE DECLARED ANCHORS: {run id: (rule, slot, seed, eval, rung, anchor cap)}. Four control-pool Pliny readings climbed the static
#: ladder (4,096→8,192→16,384) to its 16,384 rung, above every rung the static panel's unsteered runs sit on. No unsteered
#: comparator at 8,192 holds a reply the ladder's 16,384 gate escalates (at most three of 518 at the cap there, all
#: non-terminating loops), so 8,192 is its ladder-final reading, and each 16,384 reading is differenced against it.
ANCHOR_DECLARED = {
    "reseed_ctlstop:grok420:f71ed06cf77f:1a2814f29da369a8": ("control_pool", "control_lora", "s1", "grok420", 16384.0, 8192.0),
    "reseed_ctlstop:grok420:f71ed06cf77f:669b226396bfb21d": ("control_pool", "control_relu", "s1", "grok420", 16384.0, 8192.0),
    "reseed_ctlstop:grok420:f71ed06cf77f:8db801484cf39967": ("control_pool", "control_relu", "s3", "grok420", 16384.0, 8192.0),
    "reseed_ctlstop:grok420:f71ed06cf77f:e45e54bbdfc594be": ("control_pool", "control_lora", "s2", "grok420", 16384.0, 8192.0),
}
ANCHOR_DECLARED_WHY = ("no unsteered comparator at 8,192 holds a reply the ladder's 16,384 gate escalates, so 8,192 is its "
                       "ladder-final reading")
#: the notebook cells `needs` executes, as make_registry_notebook.py writes them: the header, the load cell
NEEDS_CELLS = ("583e3f04", "ec6aacd8")
NEEDS_ANCHORING = "d = R.anchored(op, pool=POOL_Q)"
LIVE = ("generated", "judged", "ingested")


def _R():
    """The notebook's reader library on the LIVE registry (the module already imported, when the notebook's live check calls in)."""
    if "registry_nb" in sys.modules:
        return sys.modules["registry_nb"]
    os.environ["REGISTRY_NB_SOURCE"] = "live"
    os.environ.setdefault("MPLBACKEND", "Agg")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import registry_nb
    return registry_nb


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _connect():
    R = _R()
    if R.SOURCE != "live":
        raise RuntimeError(f"the sidecar is derived from the live registry; this reader is on the {R.SOURCE}")
    return R.connect()


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
    """(readings, problems): the campaign's runs of the pinned objects and its unsteered runs, as {(rule, slot, seed, eval):
    [run dicts]} for the runs that are generated or complete (a queued, running, failed or cancelled run is no reading yet)."""
    R = _R()
    rows = [r for r in _frame(con, RUNS_SQL, (CAMPAIGN, R.MODEL_ID)).to_dict("records") if r["lifecycle_state"] in LIVE]
    ids = [r["run_id"] for r in rows if r["object_id"] == "none" or r["object_id"] in R.CTLSTOP_OBJECTS]
    held = dict(con.execute(HELD_SQL.format(marks=",".join("?" * len(ids))), ids).fetchall()) if ids else {}
    for r in rows:
        r["held_out"] = held.get(r["run_id"])
    readings, probs = {}, []
    drift = {v: k for k, v in R.CTLSTOP_DRIFT_ANCHORS.items()}
    for r in rows:
        if r["object_id"] == "none":
            if r["run_id"] not in drift:
                probs.append(f"{r['run_id']}: an unsteered run of the campaign that is not one of the two drift anchors")
                continue
            key = (None, "none", "", r["eval_id"])
        elif r["object_id"] in R.CTLSTOP_OBJECTS:
            slot, rule, seed = R.CTLSTOP_OBJECTS[r["object_id"]]
            key = (rule, slot, seed, r["eval_id"])
        else:
            continue
        if r["eval_id"] not in R.CTLSTOP_EVALS:
            probs.append(f"{r['run_id']}: a run of a pinned object on {r['eval_id']}, which no appendix figure reads")
            continue
        readings.setdefault(key, []).append(r)
    return readings, probs


def policy_problems() -> list:
    """Every judged eval whose pinned judge (`registry_nb.ctlstop_judging.policy`) is not the judge the notebook's own lookup names in
    this registry (`registry_nb.judge_for` under the appendix's policy, at the rubric version the eval's runs were judged under, which
    tells StrongREJECT v1 from v2). Empty = every pin is the registry's."""
    R = _R()
    probs = []
    for e in R.CTLSTOP_EVALS:
        pin = R.ctlstop_judging.policy(e)
        if pin is None:
            continue
        try:
            got = R.ctlstop_judging.policy(e, registry=True)
        except ValueError as err:
            probs.append(f"{e}: the registry does not resolve {pin.judge_model} on {pin.rubric_name} {pin.rubric_version} ({err})")
            continue
        if tuple(got) != tuple(pin):
            probs.append(f"{e}: the registry resolves {pin.judge_model} on {pin.rubric_name} {pin.rubric_version} to judge "
                         f"{str(got.judge_id)[:12]}, not the pinned {pin.judge_id[:12]}")
    return probs


def _is_rung(r: dict) -> bool:
    """A GSM8K run is the ladder's rung run when the registry says so (rung index 1, the rung pass label, or the top cap)."""
    return (_f(r.get("token_cap_rung_index")) == 1.0 or r.get("pass_label") == "ladder_rung"
            or _f(r.get("token_cap_tokens")) == 16384.0)


def _judgment(con, run: dict, pol):
    """The NEWEST COMPLETE judgment of the policy's judge on the run, or None."""
    js = _frame(con, JUDGMENTS_SQL, (run["run_id"], pol[3]))
    js = js[js.status.eq("complete")].sort_values("finished_utc")
    return js.iloc[-1].to_dict() if len(js) else None


def _row(key, run: dict, role: str, judge, cap, rung_run_id=None, anchor=None) -> dict:
    rule, slot, seed, e = key
    steered = run["object_id"] != "none"
    j = judge or {}
    return dict(run_id=run["run_id"], object_id=run["object_id"], rule=rule, slot=slot, seed=seed, eval_id=e, role=role,
                model_id=run["model_id"], instrument_id=run["instrument_id"], item_set_id=run["item_set_id"],
                lifecycle_state=run["lifecycle_state"], judgment_id=j.get("judgment_id"), judge_id=j.get("judge_id"),
                judge_model=j.get("judge_model"), rubric_name=j.get("rubric_name"), rubric_version=j.get("rubric_version"),
                token_cap_tokens=cap, run_token_cap_tokens=_f(run["token_cap_tokens"]), pass_label=run.get("pass_label"),
                lane=run.get("transfer_kind"), dose=(1.0 if steered else None), pad=(_f(run.get("padding_tokens")) or 0.0),
                pad_contract=False, serving_path=run.get("serving_path"), bon_n=_f(run.get("bon_n")), anchor_run_id=anchor,
                lane_source_legacy=run.get("lane_source_legacy"), view_legacy=run.get("view_legacy"),
                held_out=(_f(run.get("held_out")) or 0.0), rung_run_id=rung_run_id)


def _plan(R, key, runs: list, probs: list):
    """[(run, role, the drawn cap or None, the merged rung run or None)] of one reading's runs, or None (a problem was filed)."""
    e = key[3]
    spec = R.CTLSTOP_EVALS[e]
    if spec["drawn_cap"] == "run":                                       # a static cell: one run per rung, the final rung drawn
        by_cap = {}
        for r in runs:
            by_cap.setdefault(_f(r["token_cap_tokens"]), []).append(r)
        dup = {c: [r["run_id"] for r in rs] for c, rs in by_cap.items() if len(rs) > 1}
        if dup or None in by_cap:
            probs.append(f"{R.ctlstop_key(*key)}: two runs at one rung, or a run with no cap ({dup})")
            return None
        top = max(by_cap)
        return [(by_cap[c][0], "drawn" if c == top else "rung", c, None) for c in sorted(by_cap)]
    if e == "gsm8k":
        base = [r for r in runs if not _is_rung(r)]
        rung = [r for r in runs if _is_rung(r)]
        if len(base) != 1 or len(rung) > 1:
            probs.append(f"{R.ctlstop_key(*key)}: {len(base)} base and {len(rung)} rung runs (one base, at most one rung)")
            return None
        cap = R.CTLSTOP_GSM8K_RUNGS[-1] if rung else R.CTLSTOP_GSM8K_RUNGS[0]
        return [(base[0], "drawn", cap, rung[0] if rung else None)] + [(r, "rung", _f(r["token_cap_tokens"]), None) for r in rung]
    if len(runs) != 1:
        probs.append(f"{R.ctlstop_key(*key)}: {len(runs)} runs ({', '.join(r['run_id'] for r in runs)}) for one reading")
        return None
    return [(runs[0], "drift_anchor" if key[0] is None else "drawn", None, None)]


def _count(v):
    """A whole, non-negative number as an int, or None (a number only: no text is read as a count)."""
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < float("inf") and v == int(v)
    return int(v) if ok else None


def _exit_facts(note) -> dict | None:
    """The facts a `generated` state's note ends in, or None. The note is prose naming the store, then `: ` and the facts as one JSON
    object; a note with no such block, or one cut short of its end, gives None."""
    at = note.find(": {", max(note.find(" from "), 0)) if isinstance(note, str) else -1
    try:
        facts = json.loads(note[at + 2:]) if at >= 0 else None
    except ValueError:
        return None
    return facts if isinstance(facts, dict) else None


def ladder_exit(con, run_id: str, judge: dict) -> tuple:
    """({rung, n_at_cap, n}, None): the rung the run's cap ladder exited on, how many replies sat at the cap there and how many were
    counted - or (None, why no such record can be read). THE SOURCE IS THE REGISTRY'S RECORD OF THE LADDER'S CLOSE: the `generated`
    state filed from the completion artifact of the store the run names (LADDER_EXIT_SQL). Its closing facts name the final rung
    (`final_rung`) and the store's own count of replies at cap (`sidecars`: n_at_cap of n_total), beside the count re-derived from
    the store when it was closed (`rederived`), which must agree. A judgment records the rung it judged and no at-cap count, and
    per-item cap flags are filed for some runs only, so this record is the one source: no count is inferred from reply lengths, and a
    record that is absent, unreadable, without a count, or at odds with itself is refused by name. The count is the closed store's,
    so the judgment must name that store as the one it judged (`extra.judged_store_origin`)."""
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
            return None, (f"the registry's record of where its ladder exited holds two counts of the replies at cap ({rec[1]} and "
                          f"{again['n_at_cap']!r})")
        if not store or _extra(judge.get("extra_json")).get("judged_store_origin") != store:
            return None, "its judgment does not name the store its ladder exited in as the store it judged"
        recs.add(rec)
    if len(recs) > 1:
        return None, f"the registry holds {len(recs)} differing records of where its ladder exited"
    (rung, n_at_cap, n), = recs
    return dict(rung=rung, n_at_cap=n_at_cap, n=n), None


def judged_cap(spec: dict, eval_id: str, run_id: str, judge: dict, exit_record=None) -> tuple:
    """(the rung an over-refusal judgment judged, None), read from the judgment's own record (`extra.judged_cap`), or (None, why the
    reading is refused): the judgment records no rung, or - for an eval the notebook reads on one fixed rung (`judged_cap` in its
    spec: XSTest) - a rung above that one, or a rung below it unless the run's ladder exited at the rung judged with no reply at cap
    (`registry_nb.ctlstop_exit_problem` on `exit_record`: `ladder_exit`'s (record, None), or (None, why it has none))."""
    cap = _f(_extra(judge.get("extra_json")).get("judged_cap"))
    if cap is None:
        return None, f"{run_id}: its judgment {judge['judgment_id']} records no judged_cap (the rung it judged)"
    if "judged_cap" in spec and cap != spec["judged_cap"]:
        why = (f"{run_id}: its judgment {judge['judgment_id']} judged the {cap:,.0f} rung, not the {spec['judged_cap']:,.0f} rung "
               f"the notebook reads {eval_id} on")
        if cap > spec["judged_cap"]:
            return None, why
        rec, unread = exit_record or (None, None)
        problem = unread or _R().ctlstop_exit_problem(spec, cap, rec)
        if problem:
            return None, f"{why}: {problem}"
    return cap, None


def no_reading(R, spec: dict, run_id: str, entry: dict):
    """Why a judged run's payload is refused, or None: at the paper's threshold the notebook's rule reads nothing from it (no
    determined goal, no judged item, or a payload the rule cannot reduce), which would otherwise be stored and drawn as a gap."""
    if spec["family"] == "rule":
        return None
    red = R.ctlstop_judging.reduce(spec["family"], entry, R.SR_PAPER_THRESHOLD)
    return None if red.n else f"{run_id}: its payload yields no reading (no determined goal, or no judged item)"


def declared(rows: list, declarations: dict | None = None) -> tuple:
    """(`anchor_declared` section, problems): each declaration of `declarations` (default ANCHOR_DECLARED) as the file keys it - by run
    id, with its reading, rung, anchor cap and reason, in run-id order - where its run is a row of `rows` that is the drawn reading
    the declaration names, read at its rung; a declaration with no such row is a problem and is left out."""
    by_id = {r["run_id"]: r for r in rows}
    out, probs = {}, []
    for rid, (rule, slot, seed, e, rung, cap) in sorted((ANCHOR_DECLARED if declarations is None else declarations).items()):
        r = by_id.get(rid)
        if r is None or (r["rule"], r["slot"], r["seed"], r["eval_id"], r["role"]) != (rule, slot, seed, e, "drawn") \
                or r["token_cap_tokens"] != rung:
            got = "no row" if r is None else f"{r['rule']}/{r['slot']}/{r['seed']}/{r['eval_id']} {r['role']} at {r['token_cap_tokens']}"
            probs.append(f"{rid}: declared as the drawn reading {rule}/{slot}/{seed}/{e} at {rung:,.0f}, but the registry gives {got}")
            continue
        out[rid] = dict(rule=rule, slot=slot, seed=seed, eval_id=e, rung=rung, anchor_cap=cap, why=ANCHOR_DECLARED_WHY)
    return out, probs


def _main_judgment(con, R, rid: str, e: str):
    """(judgment row, rubric version, run row) of the judgment a main-figure run is read from, or the reason it has none (a str)."""
    cur = con.execute(MAIN_RUN_SQL, (rid,))
    cols = [c[0] for c in cur.description]
    got = cur.fetchone()
    if got is None:
        return "no such run in the registry"
    run = dict(zip(cols, got))
    if run["eval_id"] != e:
        return f"the registry holds it as a {run['eval_id']} run"
    JR = R._judge_rules()
    pol = R.ctlstop_judging.policy(e)
    if e == "xstest" and str(run.get("lane_source_legacy") or "") in JR.STRENGTH_READ_LANES:
        return R.CTLSTOP_MAIN_BATTERY_READ
    found = {}
    for (model, rubric, version), jid in R.CTLSTOP_JUDGES.items():
        if (model, rubric) != (pol.judge_model, pol.rubric_name):
            continue
        js = _frame(con, JUDGMENTS_SQL, (rid, jid))
        js = js[js.status.eq("complete")].sort_values("finished_utc")
        if len(js):
            found[version] = js.iloc[-1].to_dict()
    if len(found) > 1:                               # both rubric versions judged it: the version of the run's own filed judge
        want = R.resolve_judge(pol.judge_model, pol.rubric_name, run.get("judge_id"))
        found = {v: j for v, j in found.items() if j["judge_id"] == want}
    if len(found) != 1:
        return R.CTLSTOP_MAIN_NO_JUDGMENT if not found else "two judgments by the appendix's judges and no filed judge to pick by"
    (version, judge), = found.items()
    return judge, version, run


def derive(allow_pending: bool = False, con=None, main_runs: dict | None = None, needs: list | None = None) -> tuple:
    """(sidecar dict, problems): every reading of the appendix from the live registry, with its payload, in canonical form, and the
    main figures' runs read under the appendix's judge: `main_runs` ({run id: eval}; none when not given), or the runs of the lists
    `needs` (as `needs` writes them: the snapshot's and the live registry's), which are then recorded in the file (`main_needs`).
    A run left unread is a problem unless `allow_pending`, or it is a declared one (`registry_nb.ctlstop_main_undeclared`), which
    takes the live list: a derivation given `main_runs` alone has no list to hold a run against."""
    R = _R()
    if needs is not None:
        if main_runs is not None:
            raise ValueError("the main figures' runs are given as `main_runs` or as the lists `needs`, not both")
        main_runs = needs_union(needs)[0]
    own = con is None
    con = con or _connect()
    try:
        head = int(con.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
        readings, probs = discover(con)
        probs += policy_problems()
        rows, items, ids_by_set, built = [], {}, {}, []
        for key, runs in sorted(readings.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            spec = R.CTLSTOP_EVALS[key[3]]
            pol = R.ctlstop_judging.policy(key[3])
            for run, role, cap, rung_run in _plan(R, key, runs, probs) or []:
                judge = _judgment(con, run, pol) if pol is not None else None
                if pol is not None and judge is None:
                    continue                                             # not yet judged: a drawn reading without a row is pending
                exit_rec = None
                if spec["drawn_cap"] == "judged":
                    exit_record = ladder_exit(con, run["run_id"], judge) if "judged_cap" in spec else None
                    cap, why = judged_cap(spec, key[3], run["run_id"], judge, exit_record)
                    if why:
                        probs.append(why)
                        continue
                    if exit_record is not None and cap < spec["judged_cap"]:
                        exit_rec = exit_record[0]                        # the record a reading below the eval's rung is accepted on
                elif cap is None:
                    cap = spec["drawn_cap"] if isinstance(spec["drawn_cap"], float) else _f(run["token_cap_tokens"])
                anchors = [a for (a,) in con.execute(CONTRAST_SQL, (run["run_id"],)).fetchall() if a]
                if len(anchors) > 1:
                    probs.append(f"{run['run_id']}: {len(anchors)} registry anchors")
                    continue
                row = _row(key, run, role, judge, cap, rung_run["run_id"] if rung_run is not None else None, anchors[0] if anchors else None)
                ref = rung_its = flags = None
                if spec["family"] == "rule":
                    its = _frame(con, RULE_ITEMS_SQL, (run["run_id"],))
                    rung_its = _frame(con, RULE_ITEMS_SQL, (rung_run["run_id"],)) if rung_run is not None else None
                    ref = _frame(con, METRIC_SQL, (run["run_id"],))
                else:
                    its = R.ctlstop_judging.items([(run["run_id"], judge["judgment_id"])])
                if spec["family"] == "over_refusal":
                    fl = _frame(con, FLAGS_SQL, (run["run_id"],))
                    fl = fl[fl.trial_index.astype(int).eq(0)]
                    flags = ({str(i): bool(v) for i, v in zip(fl.item_id, fl.at_token_cap) if v == v}
                             if len(fl) and not fl.item_id.duplicated().any() else {})
                ids_by_set.setdefault(spec["item_set"], set()).update(str(x) for x in its.item_id)
                built.append((row, spec, its, rung_its, flags, ref, exit_rec))
                rows.append(row)
        lane_sets = set(ids_by_set)
        main_meta, main_pending, main_built, main_items, sub_by_set = {}, {}, [], {}, {}
        for rid, e in sorted((main_runs or {}).items()):
            fam = (R.CTLSTOP_EVALS.get(e) or {}).get("family")
            if fam in (None, "rule"):
                probs.append(f"{rid}: {e} is no judged eval of the appendix")
                continue
            got = _main_judgment(con, R, rid, e)
            if isinstance(got, str):
                main_pending[rid] = dict(eval_id=e, why=got)
                continue
            judge, version, run = got
            its = R.ctlstop_judging.items([(rid, judge["judgment_id"])])
            mine = R.CTL_SR_UNIT_METRICS + (R.CTL_SR_UNSCORED,) + R.CTL_SR_BEST_METRICS if fam == "strongreject" else R.CTL_CPI_METRICS
            its = its[its.metric_id.astype(str).isin(mine)]
            ids_by_set.setdefault(run["item_set_id"], set()).update(str(x) for x in its.item_id)
            main_built.append((rid, e, fam, judge, version, run, its))
        lists = {k: sorted(v) for k, v in ids_by_set.items()}
        entries = {}
        for rid, e, fam, judge, version, run, its in main_built:
            name = run["item_set_id"]
            try:
                if fam == "strongreject":
                    entries[rid] = R.ctlstop_main_sr_entry(its, lists[name], name)
                else:
                    entries[rid], sub = R.ctlstop_main_cpi_entry(its, lists[name], name)
                    for i, sv in sub.items():
                        if sub_by_set.setdefault(name, {}).setdefault(i, sv) != sv:
                            raise ValueError(f"item {i} of {name} is in two subsets across runs")
            except ValueError as err:
                probs.append(f"{rid}: {err}")
        subsets = {name: [sub.get(i) for i in lists[name]] for name, sub in sub_by_set.items()}
        for rid, e, fam, judge, version, run, its in main_built:
            if rid not in entries:
                continue
            # a payload the notebook's rule reads nothing from at the paper's threshold is refused, not stored
            row = dict(eval_id=e, metric_id=R.CTLSTOP_EVALS[e]["metric"], lane_source_legacy=run.get("lane_source_legacy"))
            res, why = R.ctlstop_judging.main_reading(fam, row, R._ctl_main_items(entries[rid], subsets.get(run["item_set_id"])), R.SR_PAPER_THRESHOLD)
            if res is None:
                probs.append(f"{rid}: its payload yields no reading ({why})")
                continue
            main_items[rid] = R.ctlstop_serialize_items(entries[rid])
            main_meta[rid] = dict(eval_id=e, judgment_id=judge["judgment_id"], rubric_version=version)
        for row, spec, its, rung_its, flags, ref, exit_rec in built:
            ids = lists[spec["item_set"]]
            try:
                if spec["family"] == "strongreject":
                    entry = R.ctlstop_sr_entry(its, ids, spec["item_set"])
                elif spec["family"] == "over_refusal":
                    entry = R.ctlstop_cpi_entry(its, ids, spec["item_set"], row["token_cap_tokens"], spec.get("subset"), flags, exit_rec)
                else:
                    entry = R.ctlstop_acc_entry(its, ids, spec["item_set"], rung_its, row["rung_run_id"])
            except ValueError as err:
                probs.append(f"{row['run_id']}: {err}")
                continue
            why = no_reading(R, spec, row["run_id"], entry)
            if why:
                probs.append(why)
                continue
            items[row["run_id"]] = R.ctlstop_serialize_items(entry)
            # A RULE-SCORED READING IS THE PRODUCER'S OWN where no rung was merged: the payload's accuracy must equal the run's filed `acc`
            if ref is not None and len(ref) and row["rung_run_id"] is None:
                _k, _n, v, _lo, _hi = R.ctlstop_judging.reduce("rule", entry, R.SR_THRESHOLD)
                if abs(float(ref.value.iat[0]) - v) > 1e-9:
                    probs.append(f"{row['run_id']}: the payload's accuracy {v:.6f} is not the filed acc {float(ref.value.iat[0]):.6f}")
        for name, lst in lists.items():
            want = next((s["n_items"] for s in R.CTLSTOP_EVALS.values() if s["item_set"] == name), None)
            if name in lane_sets and len(lst) != want:
                probs.append(f"item list {name}: the runs name {len(lst)} items, not {want}")
    finally:
        if own:
            con.close()
    rows = R.ctlstop_serialize([r for r in rows if r["run_id"] in items])
    anchor_declared, why_not = declared(rows)
    probs += why_not
    pending = R.ctlstop_expected_pending(rows)
    if pending and not allow_pending:
        probs.append(f"{len(pending)} reading(s) not yet judged (write --allow-pending for a provisional file): "
                     + ", ".join(R.ctlstop_key(p["rule"], p["slot"], p["seed"], p["eval_id"]) for p in pending[:8])
                     + (" ..." if len(pending) > 8 else ""))
    # THE CODE THE ROWS CAME FROM: the commit, and whether the tracked tree differed from it (the sidecar file itself aside, which this
    # very run rewrites); a file written from uncommitted code says so rather than naming a commit it was not built from
    try:
        git = lambda *a: subprocess.check_output(["git", *a], cwd=REPO, text=True, stderr=subprocess.DEVNULL)
        commit = git("rev-parse", "HEAD").strip()
        target = str(Path(R.CTLSTOP_SIDECAR).resolve().relative_to(REPO))
        dirty = any(ln[3:].strip() != target for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        commit, dirty = None, None
    used = {items[r["run_id"]]["item_set"] for r in rows} | {e["item_set"] for e in main_items.values()}
    # A MAIN-FIGURE RUN LEFT UNREAD stops a final write unless it is a declared one: a declared reason, and no entry in the live list
    if main_pending and not allow_pending:
        live = next((set(lst["runs"]) for lst in needs or () if lst["source"] == "live"), None)
        probs += [f"{p} (write --allow-pending for a provisional file)" for p in R.ctlstop_main_undeclared(main_pending, live)]
    out = dict(schema=R.CTLSTOP_SCHEMA, about=ABOUT,
               provenance=dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               script=str(Path(__file__).resolve().relative_to(REPO)), script_sha256=_sha(Path(__file__).resolve()),
                               git_commit=commit, git_dirty=dirty, registry_head=head),
               objects=R.ctlstop_objects_json(),
               runs={r["run_id"]: {c: r[c] for c in ("object_id", "rule", "slot", "seed", "eval_id", "role")} for r in rows})
    if pending:
        out["pending"] = pending
    out["rows"] = rows
    if anchor_declared:
        out["anchor_declared"] = anchor_declared
    if main_pending:
        out["main_pending"] = dict(sorted(main_pending.items()))
    out["main"] = dict(sorted(main_meta.items()))
    if needs is not None:
        out["main_needs"] = R.ctlstop_needs_json(needs)
    out["item_sets"] = {k: " ".join(v) for k, v in sorted(lists.items()) if k in used}
    if any(k in used for k in subsets):
        out["item_subsets"] = {k: " ".join(x or "-" for x in v) for k, v in sorted(subsets.items()) if k in used}
    out["items"] = {rid: items[rid] for rid in sorted(out["runs"])}
    out["main_items"] = {rid: main_items[rid] for rid in sorted(main_items)}
    if not probs:
        probs += R.ctlstop_sidecar_problems(out, allow_pending=allow_pending)
    return out, probs


def dumps(out: dict) -> str:
    """The file's text: valid JSON with one line per top-level key, per row, per pending reading and per run's payload, so a
    regeneration's diff names the runs that moved."""
    one = lambda v: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    parts = []
    for k, v in out.items():
        if isinstance(v, list) and k in ("rows", "pending"):
            body = "[\n" + ",\n".join(one(x) for x in v) + "\n]" if v else "[]"
        elif isinstance(v, dict) and k in ("runs", "items", "item_sets", "objects", "anchor_declared", "main", "main_pending", "main_needs",
                                            "main_items", "item_subsets"):
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


def main_needs() -> dict:
    """{"source", "head", "runs": {run id: eval}}: the main figures' runs the appendix draws, read off the notebook's own frames under
    the source REGISTRY_NB_SOURCE names, the filed judges and the paper's threshold. Executes the notebook's header and load cells as
    make_registry_notebook.py writes them, then Figure 3's anchoring, and takes `registry_nb.ctlstop_main_needs` of the result."""
    import contextlib
    import io
    import re
    for k in ("REGISTRY_NB_JUDGES", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_SR_THRESHOLD"):
        os.environ.pop(k, None)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(HERE)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    src = (HERE / "make_registry_notebook.py").read_text(encoding="utf-8")
    ns = {"__name__": "__ctlstop_needs__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for cid in NEEDS_CELLS:
            m = re.search(r'^(?:md|code)\("%s", r("""|\'\'\')(.*?)\1\)' % cid, src, flags=re.M | re.S)
            exec(m.group(2).strip("\n"), ns)
        exec(NEEDS_ANCHORING, ns)
    R = ns["R"]
    return dict(source=R.SOURCE, head=int(ns["HEAD"]), runs=R.ctlstop_main_needs(ns["d"]))


def cmd_needs(out_path: str) -> int:
    needs = main_needs()
    Path(out_path).write_text(_R().ctlstop_needs_text(needs), encoding="utf-8")
    print(f"{len(needs['runs'])} main-figure runs under the {needs['source']} source (head {needs['head']}) -> {out_path}", file=sys.stderr)
    return 0


def read_needs(files=None) -> list:
    """The `needs` lists in `files` (each {source, head, runs}), or two fresh ones, the committed snapshot's and the live registry's
    (each in a process of its own: the reader library fixes its source at import)."""
    import tempfile
    if files:
        return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    lists = []
    for source in ("snapshot", "live"):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "needs.json")
            env = {k: v for k, v in os.environ.items() if k not in ("REGISTRY_NB_JUDGES", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_SR_THRESHOLD")}
            env["REGISTRY_NB_SOURCE"] = source
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "needs", "--out", out], check=True, env=env)
            lists.append(json.loads(Path(out).read_text(encoding="utf-8")))
    return lists


def needs_union(lists) -> tuple:
    """({run id: eval}, [what each list was read from]): the runs any of the `needs` lists names. A run two lists read as different
    evals is refused."""
    runs: dict = {}
    for lst in lists:
        for rid, e in lst["runs"].items():
            if runs.setdefault(rid, e) != e:
                raise RuntimeError(f"{rid} is read as {runs[rid]} in one list and {e} in another")
    return dict(sorted(runs.items())), sorted(f"{lst['source']} at head {lst['head']}" for lst in lists)


def cmd_write(allow_pending: bool, needs_files=None) -> int:
    R = _R()
    target = Path(R.CTLSTOP_SIDECAR)
    lists = read_needs(needs_files)
    try:
        listed = R.ctlstop_needs_json(lists)
        sources = needs_union(lists)[1]
    except (ValueError, RuntimeError) as err:
        print(f"REFUSED: {err}.", file=sys.stderr)
        return 1
    if set(listed) != set(R.CTLSTOP_NEEDS_SOURCES):
        print(f"REFUSED: the main figures' runs must be listed once from the committed snapshot and once from the live registry (got {sources}).", file=sys.stderr)
        return 1
    out, problems = derive(allow_pending, needs=lists)
    if problems:
        print("REFUSED: nothing written.\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    out["provenance"]["main_runs_from"] = sources
    tmp = target.with_name("." + target.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(dumps(out))
    clean, report = scrub_clean([str(tmp.relative_to(REPO)), str(Path(__file__).resolve().relative_to(REPO))])
    if not clean:
        tmp.unlink(missing_ok=True)
        print("REFUSED: the publication scan is not clean on the new file or this script; nothing written.\n" + report, file=sys.stderr)
        return 1
    os.replace(tmp, target)
    unread = len(out.get("main_pending", {}))
    print(f"wrote {target.relative_to(REPO)}: {len(out['rows'])} rows, {len(out.get('pending', []))} pending, "
          f"{len(out.get('anchor_declared', {}))} declared anchor(s); {len(out['main'])} main-figure "
          f"runs read, {unread} left unread{' (declared: the live list names none)' if unread and not allow_pending else ''} "
          f"({'; '.join(sources)}); registry head {out['provenance']['registry_head']}; {target.stat().st_size / 1e3:,.0f} kB; "
          "publication scan clean")
    return 0


def cmd_check() -> int:
    R = _R()
    try:
        with open(R.CTLSTOP_SIDECAR, encoding="utf-8") as fh:
            committed = json.load(fh)
    except (OSError, ValueError) as err:
        print(f"cannot read {R.CTLSTOP_SIDECAR}: {err}", file=sys.stderr)
        return 2
    try:
        fresh, problems = derive(allow_pending=True, main_runs=R.ctlstop_main_runs(committed))
    except (FileNotFoundError, OSError, RuntimeError) as err:
        print(f"cannot read the live registry: {err}", file=sys.stderr)
        return 2
    diffs = []
    if committed.get("schema") != fresh.get("schema") or committed.get("objects") != fresh.get("objects"):
        diffs.append("`schema` or `objects` differs")
    mism = R.ctlstop_sidecar_check(committed, fresh)
    if len(mism):
        diffs.append("rows differ:\n" + mism.to_string(index=False))
    if problems:
        diffs.append("a fresh derivation would be refused: " + "; ".join(problems))
    if diffs:
        print("DIFFERS:\n  " + "\n  ".join(diffs))
        return 1
    print(f"EQUAL: {len(fresh['rows'])} rows, {len(fresh.get('pending', []))} pending, {len(fresh['main'])} main-figure runs, registry head "
          f"{fresh['provenance']['registry_head']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="derive from the live registry and write the sidecar")
    w.add_argument("--allow-pending", action="store_true", help="list readings not yet judged as pending and keep any main-figure run "
                   "left unread, declared or not (a provisional file)")
    w.add_argument("--needs", nargs="+", metavar="FILE", help="lists of the main figures' runs written by `needs` (the snapshot's and "
                   "the live registry's); without it both are computed now")
    sub.add_parser("check", help="derive again and compare with the committed file (0 equal, 1 differs, 2 cannot read)")
    n = sub.add_parser("needs", help="list the main figures' runs the appendix draws, under REGISTRY_NB_SOURCE")
    n.add_argument("--out", required=True, metavar="FILE")
    a = ap.parse_args(argv)
    if a.cmd == "needs":
        return cmd_needs(a.out)
    return cmd_write(a.allow_pending, a.needs) if a.cmd == "write" else cmd_check()


if __name__ == "__main__":
    sys.exit(main())
