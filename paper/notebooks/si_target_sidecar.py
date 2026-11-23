#!/usr/bin/env python3
"""paper/notebooks/si_target_sidecar.py: the committed rows behind the distillation-target appendix figure.

    python paper/notebooks/si_target_sidecar.py write                  # re-derive from the live registry, write data/qwen_si_target_multiturn_rows.json
    python paper/notebooks/si_target_sidecar.py check                  # re-derive and compare with the committed file
    ... --full-load                                                    # either, executing the whole load cell (slow) instead
    python paper/notebooks/si_target_sidecar.py pool-goals [--check]   # re-derive configs/heldout/sitarget_training_pool_goals.json

The figure (`figA_qwen_si_target_multi_turn`) sets the multi-turn runs of the objects distilled from a prompted teacher - one safety, one
constitution and one irrelevant-instruction target, a LoRA and a ReLU steer each, three fit seeds - each beside the constitution corpus fit
of the same kind (Figure 4's Constitution LoRA and Constitution ReLU steer), on all 259 goals and on the 92 goals outside the training pool;
its tables add the two prompted instructions. The distilled objects' runs stay outside the paper's selection, and the lower row needs every
run's per-goal scores, which the paper frame does not keep; so the notebook reads all of them from the sidecar file this script writes, and
`registry_nb.SITARGET_OBJECTS`, `SITARGET_RUNS` and `SITARGET_ANCHOR_RUNS` pin every object and run.

EVERY JUDGED NUMBER IS THE FIGURE'S ONE JUDGE'S: MiniMax-M3 StrongREJECT v1 (`registry_nb.SITARGET_EVAL_JUDGE`), the judge the notebook's
re-judge policy reads these attacks from. The script pins the policy itself (REGISTRY_NB_JUDGES=rejudge, the paper's StrongREJECT
threshold, no Petri override) before the notebook's header cell runs, whatever the caller's shell holds, so a file is never derived under
another judge.

The rows are the notebook's own. The script executes the generator's header cell and the load cell's query definitions (by cell id, from
`make_registry_notebook.py`) against the live registry, then reads the load cell's own frontier query (`FRONTIER_SQL`: Qwen's owner rows on
these evals in any selection state) with one more condition, the pinned RUN IDS (never the objects: the unsteered anchor runs belong to the
unsteered model, which carries many other runs), and spells and deduplicates it with the same `registry_nb.spell` (which draws every judged
row from the policy's judge, `select_judges`) and `registry_nb.dedupe` calls the load cell makes. Every stored column is a property of the
run itself (the frame-dependent ones are pinned or taken from the run's own decode seed), so these rows equal the ones a live render's full
`frontier` holds, and a live render checks exactly that, field by field. `--full-load` executes the whole load cell instead (slow). Each
pinned run's owner `asr` row at threshold 0.5 is taken (`registry_nb.sitarget_rows_from_frame`), and only the stored projection
`registry_nb.SITARGET_STORED` is written, in the canonical form `registry_nb.sidecar_serialize` gives: no measurement. Each run's per-goal
payload (`goal_best`: the best score of each goal over its scored replies, with its reply and scored counts) comes from the item rows of the
row's own judgment through the notebook's item reader (`registry_nb.JUDGING.items`, `registry_nb.sitarget_goal_best_entry`); the notebook
computes every reading from it at load, at any threshold. Nothing is written to the registry.

The provenance names the script and its sha256, the commit and whether the tracked tree differed from it (`git_dirty`), the registry head
read, the judge policy, the runs of the pinned non-distilled objects on these instruments that are not drawn: those named in
`registry_nb.SITARGET_KNOWN_UNDRAWN` (`undrawn_known`) and any not named there (`undrawn_new`), and every reply the judge returned no
score for in a run's judgment (`unscored_replies`: run, goal, reply index), which the payload counts in its replies and not in its scored
ones.

`write` refuses, writing nothing, when a pinned run is absent or duplicated, is not complete, was read from another judge than its eval's
(judge id, model, rubric or version), sits on another item set, instrument or token cap or under another decode policy, sent decode or
thinking mode than the other pinned runs (`decode_identity`), is not its pinned object's, is not the headline row, has no item rows in its
judgment or a payload the notebook refuses (a goal missing, a per-goal best that is not the max of its replies, more than 1% of goals
undetermined at 0.5), or reads a value at 0.5 its frontier row does not; when the goal block is not the committed derivation's; when a run of
a DISTILLED object on these instruments is not pinned; or when the publication scan (`reproduce/tools/scrub_grep.sh --all`) finds anything
in the new file or in this script: the file is written beside the target, scanned, and moved into place only on a clean scan. A run of the
prompted instructions, the constitution corpus fits or the unsteered model on these instruments that is neither pinned nor in
`SITARGET_KNOWN_UNDRAWN` is not a refusal (the figure draws pinned runs alone): `write` and `check` print one line per such run,
`[sitarget-sidecar] UNDRAWN-NEW <run_id> object=<object_id> eval=<eval_id> (not drawn; outside the pins)`, and `write` records it. `check`
exits 0 when the committed file equals a fresh derivation (UNDRAWN-NEW lines included), 1 when a row, a payload, an unscored reply, the
goal block or the pins differ (each difference printed), 2 when it cannot read the registry or the file.

`pool-goals` re-derives the training pools' goal coverage from the pool files (`--safety`, `--constitution`, `--c16`, or the
SITARGET_POOL_SAFETY / SITARGET_POOL_CONSTITUTION / SITARGET_POOL_C16 variables) and the pinned HarmBench and JailbreakBench releases
(the local Hugging Face cache, offline), and writes the artefact (`--check` compares instead).

Needs the notebook environment (`uv sync --frozen --extra cpu --group notebook`) and the live registry (REGISTRY_ROOT, default ~/registry).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GENERATOR = HERE / "make_registry_notebook.py"
#: the notebook's header and load cells, by their pinned generator ids (make_registry_notebook.py)
HEADER_CELL, LOAD_CELL = "583e3f04", "ec6aacd8"
SCRUB = REPO / "reproduce" / "tools" / "scrub_grep.sh"
POOL_GOALS = REPO / "configs" / "heldout" / "sitarget_training_pool_goals.json"
EVAL_CORPUS = REPO / "data" / "heldout_corpora" / "renellm.jsonl"
TAG = "[sitarget-sidecar]"
ABOUT = ("Multi-turn runs (259 goals) of the Qwen3.5-4B objects distilled from a prompted teacher (safety, constitution and "
         "irrelevant-instruction targets; a LoRA and a ReLU steer each; three fit seeds), the two prompted instructions, the constitution "
         "corpus fits (Constitution LoRA and Constitution ReLU steer) and the unsteered runs Figure 4 pools: one row per run as the paper "
         "notebook's frontier frame spells it, read from its MiniMax-M3 StrongREJECT (v1) judgment, with each run's per-goal best score and "
         "reply counts, from which the notebook computes every reading at any threshold; regenerate with paper/notebooks/si_target_sidecar.py.")
#: the judging environment the file is derived under, set before the notebook's header cell runs (the flip's rule for every tool that
#: imports the notebook)
PINNED_ENV = {"REGISTRY_NB_JUDGES": "rejudge", "REGISTRY_NB_SOURCE": "live"}
UNSET_ENV = ("REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_PETRI_EXCLUDE")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quoted(values) -> str:
    for v in values:
        if "'" in str(v):
            raise ValueError(f"a quote in a pinned id ({v!r})")
    return ", ".join(f"'{v}'" for v in values)


def generator_cell(cell_id: str) -> str:
    """One cell's source as make_registry_notebook.py writes it (by its pinned id)."""
    src = GENERATOR.read_text(encoding="utf-8")
    m = re.search(r'^(?:md|code)\("%s", r("""|\'\'\')(.*?)\1\)' % re.escape(cell_id), src, flags=re.M | re.S)
    if not m:
        raise SystemExit(f"REFUSED: {GENERATOR.relative_to(REPO)} has no cell {cell_id} (the header / load cell ids moved)")
    return m.group(2).strip("\n")


def notebook_namespace(full_load: bool = False) -> dict:
    """The notebook namespace against the LIVE registry under the pinned judging environment (PINNED_ENV), with `R` and `frontier`: the
    generator's header cell, then either the whole load cell (`full_load`) or its query definitions and the frontier query restricted to
    the pinned run ids, spelled (judges selected) and deduplicated as the load cell does (the notebook's own printed report goes to
    stderr)."""
    os.environ.update(PINNED_ENV)
    for k in UNSET_ENV:
        os.environ.pop(k, None)
    os.environ.setdefault("MPLBACKEND", "Agg")
    # a render under a policy that is not the paper's refuses the paper's figures directory; this script draws nothing, so any scratch
    # directory will do
    os.environ.setdefault("REGISTRY_NB_FIGS", tempfile.mkdtemp(prefix="sitarget-sidecar-figs-"))
    if "registry_nb" in sys.modules:
        raise SystemExit("REFUSED: registry_nb is already imported in this process, under a judging policy this script did not set")
    os.chdir(HERE)                                    # the header cell inserts os.getcwd() for `import registry_nb`
    sys.path.insert(0, str(HERE))
    ns: dict = {"__name__": "__si_target_sidecar__"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(generator_cell(HEADER_CELL), "<notebook header cell>", "exec"), ns)
        if full_load:
            exec(compile(generator_cell(LOAD_CELL), "<notebook load cell>", "exec"), ns)
            return ns
        load = generator_cell(LOAD_CELL)
        start, end = load.find("ML = "), load.find("_raw = ")
        if start < 0 or end < start or "FRONTIER_SQL = " not in load[start:end]:
            raise SystemExit("REFUSED: the load cell no longer defines its queries before reading them; use --full-load")
        exec(compile(load[start:end], "<notebook load cell: query definitions>", "exec"), ns)
        R = ns["R"]
        sql = ns["FRONTIER_SQL"] + f" AND v.run_id IN ({_quoted(sorted(R.sitarget_run_ids()))})"
        ns["frontier"] = R.dedupe(R.spell(R.query(sql, "sitarget_frontier")))
    R = ns["R"]
    if R.JUDGES != PINNED_ENV["REGISTRY_NB_JUDGES"] or R.JUDGES != R.SITARGET_JUDGE_POLICY or R.SR_THRESHOLD != R.SR_PAPER_THRESHOLD:
        raise SystemExit(f"REFUSED: the notebook read the policy {R.JUDGES!r} at threshold {R.SR_THRESHOLD:g}, not the pinned one")
    return ns


def payloads(R, rows: list, order: list) -> tuple:
    """({run id: per-goal payload}, [unscored replies], problems) from the item rows of each row's own judgment, read through the
    notebook's item reader (`R.JUDGING.items`) and built by `R.sitarget_goal_best_entry`: a run whose judgment has no item rows, or
    whose rows the builder refuses, is a problem."""
    if not rows:
        return {}, [], []
    items = R.JUDGING.items(sorted({r["judgment_id"] for r in rows if r.get("judgment_id")}), fields=())
    out, unscored, probs = {}, [], []
    for r in rows:
        rid, jid = r["run_id"], r.get("judgment_id")
        its = items[items["run_id"].astype(str).eq(rid) & items["judgment_id"].astype(str).eq(str(jid))]
        if not len(its):
            probs.append(f"{rid}: no item rows in its judgment {jid}")
            continue
        try:
            out[rid] = R.sitarget_goal_best_entry(its, order)
        except ValueError as err:
            probs.append(f"{rid}: its judgment {jid}: {err}")
            continue
        unscored += R.sitarget_unscored_replies(its, rid)
    return out, sorted(unscored, key=lambda e: (e["run_id"], e["item_id"], e["reply"])), probs


def goals_block(R) -> tuple:
    """(the sidecar's `goals` block, problems) from the committed pool-goals artefact: the 259 goal ids in order, the outside-pool mask over
    them, the rule, and each pool's content sha256."""
    try:
        art = json.loads(POOL_GOALS.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        return None, [f"cannot read {POOL_GOALS.relative_to(REPO)}: {err}"]
    order = art["eval_goals"]["order"].split(" ")
    block = dict(item_set_id=R.SITARGET_ITEM_SET, order=" ".join(order), order_sha256=hashlib.sha256("\n".join(order).encode()).hexdigest(),
                 outside_pool=art["outside_pool_mask"], outside_pool_n=int(art["outside_pool_n"]), rule=art["match_rule"],
                 pool_sha256={k: v["content_sha256"] for k, v in art["pools"].items()})
    return block, R.sitarget_goal_problems(block)


def undrawn_runs(R) -> tuple:
    """(problems, known, new) from the registry's run table: every run of a pinned object - or of the unsteered model - on the two
    instruments that is not pinned. A DISTILLED object's is a problem (a second reading of one of its cells the pins leave out); another
    object's is `known` when SITARGET_KNOWN_UNDRAWN names it, else `new` (reported, never refused). A placeholder object id matches nothing."""
    pinned = set(R.sitarget_run_ids())
    objects = sorted(set(R.SITARGET_OBJECTS) | {"none"})
    sql = ("SELECT run_id, object_id, eval_id, instrument_id, lifecycle_state FROM run "
           f"WHERE model_id = '{R.MODEL_ID}' AND instrument_id IN ({_quoted(sorted(R.SITARGET_INSTRUMENTS.values()))}) "
           f"AND object_id IN ({_quoted(objects)})")
    with R.connect() as con:
        found = con.execute(sql).fetchall()
    new_slots = set(R.SITARGET_NEW_SLOTS)
    probs, known, new = [], [], []
    for rid, oid, e, _inst, state in sorted(found):
        if rid in pinned:
            continue
        slot = R.SITARGET_OBJECTS.get(oid, (None, None))[0]
        if slot in new_slots:
            probs.append(f"{rid}: a run of the distilled object {oid} ({slot}) on {e} that is not pinned (lifecycle {state})")
        elif rid in R.SITARGET_KNOWN_UNDRAWN:
            known.append(rid)
        else:
            new.append(dict(run_id=rid, object_id=oid, eval_id=e, lifecycle_state=state))
    return probs, known, new


def undrawn_line(u: dict) -> str:
    return f"{TAG} UNDRAWN-NEW {u['run_id']} object={u['object_id']} eval={u['eval_id']} (not drawn; outside the pins)"


#: the frontier columns a run's decode identity is read from (`decode_identity`); a frame without any one of them refuses by name
DECODE_COLUMNS = ("decode_policy_id", "decode_sent_json", "thinking_mode", "run_extra_json")


def decode_identity(row) -> tuple:
    """((policy, sent, thinking mode), problem) of one frontier row. The policy is the registry's matched `decode_policy_id`; a run the
    registry filed with none (the job-start registration of the distilled objects' wave leaves the target's seed keys unmodelled, files
    `decode_policy_match` no_match and `decode_source` vendor_recommendation, and leaves the policy and seed columns empty) takes the policy
    its producer served, as the registry's decode-evidence reader recorded it (`extra.decode_evidence.producer_policy_id`), and only when
    that evidence saw one decode block on every shard and the served target's decode agreeing with it. `sent` is the sampling numbers the
    run sent (`decode_sent_json`, canonical). A run with neither a matched policy nor such evidence, or with no sent decode, has no identity."""
    rid = row.get("run_id")
    policy = row.get("decode_policy_id")
    policy = policy if isinstance(policy, str) and policy.strip() else None
    if policy is None:
        try:
            ev = (json.loads(row.get("run_extra_json") or "{}") or {}).get("decode_evidence") or {}
        except (TypeError, ValueError, AttributeError):
            ev = {}
        wit = ev.get("witnesses") or {}
        prod = ev.get("producer_policy_id")
        if not (isinstance(prod, str) and prod.strip() and wit.get("distinct_blocks") == 1 and wit.get("served_target_decode_agrees") is True):
            return None, (f"{rid}: no decode policy (the registry matched none, and its decode evidence names no producer policy over one "
                          "agreeing decode block)")
        policy = prod
    try:
        sent = json.loads(row.get("decode_sent_json") or "")
    except (TypeError, ValueError):
        sent = None
    if not isinstance(sent, dict) or not sent:
        return None, f"{rid}: no sent decode (decode_sent_json)"
    thinking = row.get("thinking_mode")
    return (policy, json.dumps(sent, sort_keys=True), thinking if isinstance(thinking, str) else None), None


def decode_problems(pinned) -> list:
    """Every way the pinned runs (frontier rows, one or more per run) fail to share ONE decode identity (`decode_identity`): a run with no
    identity, or more than one identity among the runs (each named with its run count)."""
    probs, sigs = [], {}
    for _i, row in pinned.drop_duplicates("run_id").iterrows():
        sig, prob = decode_identity(row)
        if prob:
            probs.append(prob)
        else:
            sigs.setdefault(sig, []).append(row["run_id"])
    if len(sigs) > 1:
        probs.append("the pinned runs do not share one decode policy, sent decode and thinking mode: "
                     + "; ".join(f"{p} {s} thinking {t}: {len(r)} runs" for (p, s, t), r in sorted(sigs.items(), key=lambda kv: -len(kv[1]))))
    return probs


def derive(ns: dict) -> tuple:
    """(sidecar dict, problems, undrawn_new) from the notebook namespace: the pinned rows of `frontier`, their payloads and unscored replies,
    the goal block and every refusal."""
    R, frontier = ns["R"], ns["frontier"]
    framed = R.sitarget_rows_from_frame(frontier)
    rows = R.sidecar_serialize(framed[list(R.SITARGET_STORED)], R.SITARGET_STORED)
    objects, runs = R.sitarget_pins_json()
    problems = R.sitarget_problems(rows)
    missing = sorted(set(runs) - {r["run_id"] for r in rows})
    if missing:
        problems.append(f"pinned runs with no row read from the frontier (its asr row at 0.5 under the {R.JUDGES} judges): {missing}")
    n_want = len(R.SITARGET_SLOTS) * 3 * len(R.SITARGET_EVALS)
    if len(R.SITARGET_RUNS) != n_want:
        problems.append(f"SITARGET_RUNS holds {len(R.SITARGET_RUNS)} readings, not the {n_want} the figure and tables read")
    # ONE DECODE SIGNATURE: every pinned run - prompted, corpus fit, distilled, unsteered - under the same decode policy, sent sampling
    # numbers and thinking mode (the instrument, item set, judge and cap are held per row by `sitarget_problems`; the decode columns are not
    # stored, so they are held here). A frame without one of the columns the identity is read from refuses by name rather than skipping
    # the check
    pinned = frontier[frontier.run_id.isin(R.sitarget_run_ids()) & frontier.metric_id.eq("asr")] if len(frontier) else frontier
    if len(pinned):
        miss = [c for c in DECODE_COLUMNS if c not in pinned.columns]
        problems += ([f"the frontier frame has no {', '.join(miss)} column, so the pinned runs' decode identity cannot be read "
                      "(`decode_identity`)"] if miss else decode_problems(pinned))
    goals, gprobs = goals_block(R)
    problems += gprobs
    goal_best, unscored, pprobs = payloads(R, rows, R.sitarget_goal_order(goals)) if goals else ({}, [], [])
    problems += pprobs
    # EACH PAYLOAD READS THE FRONTIER'S OWN READING at the paper's threshold (the frame's row was re-derived by the notebook's judge
    # selection from the same judgment's item rows): k, n, value and the interval, to 1e-12
    if goal_best and not pprobs:
        mism = R.sitarget_reading_check([r for r in rows if r["run_id"] in goal_best], goal_best, framed, R.SR_PAPER_THRESHOLD)
        problems += [f"{m.run_id}: its payload reads {m.field} {m.sidecar}, its frontier row {m.live}" for m in mism.itertuples()]
    uprobs, known, new = undrawn_runs(R)
    problems += uprobs
    # THE CODE THE ROWS CAME FROM: the commit, and whether the tracked tree differed from it (the sidecar file itself aside, which this very
    # run rewrites); a file written from uncommitted code says so rather than naming a commit it was not built from
    try:
        git = lambda *a: subprocess.check_output(["git", *a], cwd=REPO, text=True, stderr=subprocess.DEVNULL)
        commit = git("rev-parse", "HEAD").strip()
        target = str(Path(R.SITARGET_SIDECAR).resolve().relative_to(REPO))
        dirty = any(ln[3:].strip() != target for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    out = dict(schema=R.SITARGET_SCHEMA, about=ABOUT,
               provenance=dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               script=str(Path(__file__).resolve().relative_to(REPO)), script_sha256=_sha(Path(__file__).resolve()),
                               git_commit=commit, git_dirty=dirty, registry_head=int(frontier.attrs.get("head", R.head())),
                               judge_policy=R.SITARGET_JUDGE_POLICY, undrawn_known=sorted(known), undrawn_new=sorted(u["run_id"] for u in new),
                               unscored_replies=unscored),
               objects=objects, runs=runs, goals=goals, rows=rows,
               goal_best={r["run_id"]: goal_best[r["run_id"]] for r in rows if r["run_id"] in goal_best})
    problems += R.sitarget_sidecar_problems(out) if not problems else []
    return out, problems, new


def scrub_clean(paths: list) -> tuple:
    """(clean, report): the publication scan over `paths` (repository-relative). `scrub_grep.sh` exits 0 whatever it finds, so the
    verdict is read from its summary line: `scrub: no identifier under ...` is clean, anything else is not."""
    env = dict(os.environ, SCRUB_PYTHON=sys.executable)
    p = subprocess.run(["bash", str(SCRUB), "--all", *paths], cwd=REPO, env=env, capture_output=True, text=True)
    report = (p.stdout + p.stderr).strip()
    last = report.splitlines()[-1] if report else ""
    return p.returncode == 0 and last.startswith("scrub: no identifier under"), report


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def cmd_write(full_load: bool = False) -> int:
    ns = notebook_namespace(full_load)
    R = ns["R"]
    target = Path(R.SITARGET_SIDECAR)
    out, problems, new = derive(ns)
    for u in new:
        print(undrawn_line(u))
    if problems:
        print("REFUSED: nothing written.\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    tmp = target.with_name("." + target.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(_dump(out), encoding="utf-8")
    clean, report = scrub_clean([str(tmp.relative_to(REPO)), str(Path(__file__).resolve().relative_to(REPO))])
    if not clean:
        tmp.unlink(missing_ok=True)
        print("REFUSED: the publication scan is not clean on the new file or this script; nothing written.\n" + report, file=sys.stderr)
        return 1
    os.replace(tmp, target)
    print(f"wrote {target.relative_to(REPO)}: {len(out['rows'])} rows, {len(out['goal_best'])} payloads, "
          f"{len(out['provenance']['unscored_replies'])} unscored replies, {len(new)} undrawn new, registry head "
          f"{out['provenance']['registry_head']}, judges {out['provenance']['judge_policy']}; publication scan clean")
    return 0


def cmd_check(full_load: bool = False) -> int:
    try:
        ns = notebook_namespace(full_load)
    except (FileNotFoundError, OSError) as err:
        print(f"cannot read the live registry: {err}", file=sys.stderr)
        return 2
    R = ns["R"]
    try:
        with open(R.SITARGET_SIDECAR, encoding="utf-8") as fh:
            committed = json.load(fh)
    except (OSError, ValueError) as err:
        print(f"cannot read {R.SITARGET_SIDECAR}: {err}", file=sys.stderr)
        return 2
    fresh, problems, new = derive(ns)
    for u in new:
        print(undrawn_line(u))
    diffs = []
    for key in ("schema", "objects", "runs", "goals"):
        if committed.get(key) != fresh.get(key):
            diffs.append(f"`{key}` differs")
    try:
        mism = R.sitarget_sidecar_check(committed.get("rows") or [], fresh["rows"])
    except (TypeError, ValueError) as err:
        mism = None
        diffs.append(f"the committed rows cannot be compared: {err}")
    if mism is not None and len(mism):
        diffs.append("rows differ:\n" + mism.to_string(index=False))
    pm = R.sitarget_payload_check(committed.get("goal_best") or {}, fresh["goal_best"])
    if len(pm):
        diffs.append(f"per-goal payloads differ for {pm.run_id.nunique()} run(s):\n" + pm.to_string(index=False))
    un_a = (committed.get("provenance") or {}).get("unscored_replies")
    if un_a != fresh["provenance"]["unscored_replies"]:
        diffs.append(f"unscored replies: committed {un_a} / now {fresh['provenance']['unscored_replies']}")
    if (committed.get("provenance") or {}).get("judge_policy") != fresh["provenance"]["judge_policy"]:
        diffs.append("the judge policy differs")
    if problems:
        diffs.append("a fresh derivation would be refused: " + "; ".join(problems))
    if diffs:
        print("DIFFERS:\n  " + "\n  ".join(diffs))
        return 1
    print(f"EQUAL: {len(fresh['rows'])} rows, {len(fresh['goal_best'])} payloads, {len(fresh['provenance']['unscored_replies'])} unscored "
          f"replies, {len(new)} undrawn new, registry head {fresh['provenance']['registry_head']}, judges {fresh['provenance']['judge_policy']}")
    return 0


# ---- the training pools' goal coverage ---------------------------------------------------------------------------------------------
#: the pinned releases the eval goals' behaviour texts come from (configs/two_si.yaml `datasets`, the pins the pools were built at)
HARMBENCH = dict(repo="mariagrandury/harmbench", revision="4aa1f46a9d43e441e238c1d01dc52413094fc84e", splits=("val", "test"),
                 file="DirectRequest/{split}-00000-of-00001.parquet")
JBB = dict(repo="JailbreakBench/JBB-Behaviors", revision="886acc352a31533ffbcf4ef22c744658688086fc", file="data/harmful-behaviors.csv")
#: the fit/held-out carve of the two-SI pools (scripts/train_two_si_objects.py `split_pairs`, salt and fraction as the fits ran it)
CARVE = dict(salt="two_si_holdout", holdout_frac=0.1)
POOLS = (("safety", "qwen35_4b/pairs_safety_refusal", "SITARGET_POOL_SAFETY"),
         ("constitution", "qwen35_4b/pairs_si_constitution", "SITARGET_POOL_CONSTITUTION"),
         ("c16", "qwen35_4b/pairs_C16", "SITARGET_POOL_C16"))
MATCH_RULE = ("a goal is in the training pool when a fit-split pair of the pool carries its goal id, or carries a goal whose behaviour text "
              "(HarmBench: the context, a blank line and the behaviour; JailbreakBench: the goal) equals the goal's own, compared with case "
              "and runs of whitespace folded and a trailing period dropped; the fit split is every pair with "
              "sha256(salt + prompt)[0] / 255 >= holdout_frac")
#: the four eval goals in the pool by text alone, each a JailbreakBench goal repeating a trained HarmBench behaviour (pinned in the tests)
SCHEMA_POOL_GOALS = "sitarget-training-pool-goals-v1"


def _ids(ids) -> str:
    """A list of goal ids as the files store it: one string, the ids separated by single spaces (a JSON list of these ids puts two of them
    beside a comma, which the secret scanner's generic-key rule reads as a key and its value; a goal id holds no space)."""
    return " ".join(ids)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def behaviour_texts(harmbench_dir: Path | None = None, jbb_csv: Path | None = None) -> dict:
    """{goal id: behaviour text} of the pinned HarmBench (val and test) and JailbreakBench releases, from local files or the local Hugging
    Face cache (offline)."""
    import pandas as pd
    if harmbench_dir is None or jbb_csv is None:
        from huggingface_hub import hf_hub_download
    texts = {}
    for split in HARMBENCH["splits"]:
        f = HARMBENCH["file"].format(split=split)
        path = (Path(harmbench_dir) / f) if harmbench_dir else Path(hf_hub_download(HARMBENCH["repo"], f, revision=HARMBENCH["revision"],
                                                                                      repo_type="dataset", local_files_only=True))
        for r in pd.read_parquet(path).to_dict("records"):
            ctx = r.get("ContextString")
            ctx = "" if ctx is None or (isinstance(ctx, float) and ctx != ctx) else str(ctx)
            beh = str(r["Behavior"])
            texts[str(r["BehaviorID"])] = f"{ctx}\n\n{beh}" if ctx.strip() else beh
    path = Path(jbb_csv) if jbb_csv else Path(hf_hub_download(JBB["repo"], JBB["file"], revision=JBB["revision"], repo_type="dataset",
                                                              local_files_only=True))
    jb = pd.read_csv(path)
    for idx, goal in zip(jb["Index"], jb["Goal"]):
        texts[f"jbb_harmful_{int(idx)}"] = str(goal)
    return texts


def pool_content_sha256(rows: list) -> str:
    """The pool's content hash as the fits record it (scripts/train_two_si_objects.py `pool_content_sha256`: every identity-bearing field
    of each pair, canonical JSON per row)."""
    h = hashlib.sha256()
    for p in rows:
        h.update(json.dumps({"prompt": p["prompt"], "pos": p["pos"], "neg": p["neg"], "instruction_id": p.get("instruction_id", ""),
                             "requirement_a": p.get("requirement_a"), "anti_neg": p.get("anti_neg")},
                            sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def derive_pool_goals(pool_paths: dict, texts: dict, eval_corpus: Path = EVAL_CORPUS) -> dict:
    """The artefact: the 259 eval goals in order, each pool's fit-split goal ids that have a behaviour text (the only ones either rule can
    match), its file and content sha256 and counts, the goals in the pool by id and by text, and the outside-pool list and mask. Raises when
    the three pools disagree on which eval goals they contain."""
    order = sorted({json.loads(ln)["goal_id"] for ln in eval_corpus.read_text(encoding="utf-8").splitlines() if ln.strip()})
    missing = [g for g in order if g not in texts]
    if missing:
        raise ValueError(f"eval goals with no behaviour text in the pinned releases: {missing[:5]}")
    pools, covered, twins = {}, {}, {}
    for key, name, _env in POOLS:
        data = Path(pool_paths[key]).read_bytes()
        rows = [json.loads(ln) for ln in data.decode("utf-8").splitlines() if ln.strip()]
        fit = [r for r in rows if not (hashlib.sha256((CARVE["salt"] + r["prompt"]).encode()).digest()[0] / 255.0 < CARVE["holdout_frac"])]
        ids = {r["goal_id"] for r in fit}
        texted = sorted(g for g in ids if g in texts)
        by_text = {_norm(texts[g]): g for g in texted}
        by_id = sorted(g for g in order if g in ids)
        extra = {g: by_text[_norm(texts[g])] for g in order if g not in ids and _norm(texts[g]) in by_text}
        pools[key] = dict(name=name, file_sha256=hashlib.sha256(data).hexdigest(), content_sha256=pool_content_sha256(rows), rows=len(rows),
                          fit_rows=len(fit), fit_goal_ids=len(ids), fit_goal_ids_with_text=_ids(texted))
        covered[key] = (by_id, sorted(extra))
        twins[key] = extra
    if len({(tuple(a), tuple(b)) for a, b in covered.values()}) != 1:
        raise ValueError("the pools disagree on which eval goals they contain: "
                         + "; ".join(f"{k} {len(a)} by id + {len(b)} by text" for k, (a, b) in covered.items()))
    by_id, by_text = covered["safety"]
    inside = set(by_id) | set(by_text)
    outside = [g for g in order if g not in inside]
    mask = "".join("1" if g not in inside else "0" for g in order)
    return dict(schema=SCHEMA_POOL_GOALS,
                about=("Which of the paper's 259 multi-turn evaluation goals the training pools of the distilled objects contain. The "
                       "paper notebook does not read this file; its tests and paper/notebooks/si_target_sidecar.py (which writes it with "
                       "`pool-goals`) hold the figure's outside-pool mask to it."),
                eval_goals=dict(source=str(eval_corpus.relative_to(REPO)), item_set_id="multiturn_goals_259", n=len(order),
                                goal_id_set_sha256=hashlib.sha256("\n".join(order).encode()).hexdigest(), order=_ids(order)),
                carve=dict(CARVE, rule="a pair is held out when sha256(salt + prompt)[0] / 255 < holdout_frac; every other pair is the fit split"),
                match_rule=MATCH_RULE,
                datasets=dict(harmbench=dict(repo=HARMBENCH["repo"], revision=HARMBENCH["revision"], splits=list(HARMBENCH["splits"])),
                              jailbreakbench=dict(repo=JBB["repo"], revision=JBB["revision"])),
                pools=pools,
                text_twins=_ids(f"{g}:{twins['safety'][g]}" for g in sorted(twins["safety"])),
                in_pool_by_id=_ids(by_id), in_pool_by_text=_ids(by_text), outside_pool=_ids(outside), outside_pool_n=len(outside),
                outside_pool_mask=mask, outside_pool_mask_sha256=hashlib.sha256(mask.encode()).hexdigest())


def pool_paths_from(args) -> dict:
    paths = {}
    for key, _name, env in POOLS:
        p = getattr(args, key, None) or os.environ.get(env)
        if not p:
            raise SystemExit(f"REFUSED: no path for the {key} pool (--{key} or {env})")
        paths[key] = os.path.expanduser(p)
    return paths


def cmd_pool_goals(args) -> int:
    texts = behaviour_texts(args.harmbench_dir, args.jbb_csv)
    art = derive_pool_goals(pool_paths_from(args), texts)
    if args.check:
        committed = json.loads(POOL_GOALS.read_text(encoding="utf-8"))
        diff = sorted(k for k in set(art) | set(committed) if art.get(k) != committed.get(k))
        if diff:
            print(f"DIFFERS: {POOL_GOALS.relative_to(REPO)} keys {diff}")
            return 1
        print(f"EQUAL: {POOL_GOALS.relative_to(REPO)} ({len(art['in_pool_by_id'].split())} by id, {len(art['in_pool_by_text'].split())} by text, "
              f"{art['outside_pool_n']} outside)")
        return 0
    POOL_GOALS.write_text(_dump(art), encoding="utf-8")
    print(f"wrote {POOL_GOALS.relative_to(REPO)}: {len(art['in_pool_by_id'].split())} by id, {len(art['in_pool_by_text'].split())} by text, "
          f"{art['outside_pool_n']} outside the pool; mask sha256 {art['outside_pool_mask_sha256']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="re-derive from the live registry and write the sidecar")
    c = sub.add_parser("check", help="re-derive and compare with the committed file (0 equal, 1 differs, 2 cannot read)")
    for p in (w, c):
        p.add_argument("--full-load", action="store_true", help="execute the whole load cell (every frame; slow) for the frontier")
    g = sub.add_parser("pool-goals", help="re-derive the training pools' goal coverage (configs/heldout/sitarget_training_pool_goals.json)")
    for key, _name, env in POOLS:
        g.add_argument(f"--{key}", help=f"the {key} pool's pairs file (default ${env})")
    g.add_argument("--harmbench-dir", type=Path, help="the pinned HarmBench release's files (default: the local Hugging Face cache)")
    g.add_argument("--jbb-csv", type=Path, help="the pinned JailbreakBench harmful-behaviours file (default: the local Hugging Face cache)")
    g.add_argument("--check", action="store_true", help="compare with the committed artefact instead of writing it")
    a = ap.parse_args(argv)
    if a.cmd == "pool-goals":
        return cmd_pool_goals(a)
    return cmd_write(a.full_load) if a.cmd == "write" else cmd_check(a.full_load)


if __name__ == "__main__":
    sys.exit(main())
