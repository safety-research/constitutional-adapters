#!/usr/bin/env python3
"""reproduce/lib/coverage_probe.py — the gate's per-run COVERAGE reader, one layout per producer family. stdlib only: it runs on
a cluster login host from the mirrored checkout
(`<eval repo root>/<code_sha>/reproduce/lib/coverage_probe.py`, via 06_eval_waves.sh --gate over ssh) or on the grading host, and
prints ONE JSON document. It never sums heterogeneous JSONL line counts: every layout reads the family's AUTHORITATIVE artifact
and reports the UNIQUE COMPLETED work ids (goal ids / item ids), the ERRORED ids apart, and — where the expected set lives on the
producer's side (the BoN climb manifest) — the expected ids too. lib/admission.py compares against the frozen unit.

  probe --layout jailbreak_cellstore --path <…/<private_run>/<attack>/<cell>> --shards shard00,shard01
        jailbreak_run.py's gated stores: `<shard>/progress.jsonl`, the LAST record per goal_id (jailbreak_run._load_progress);
        completed = a terminal outcome that is not `error:*` (jailbreak_artifacts.ERRORED_OUTCOME_PREFIX; `refused:*` is terminal
        and reported apart); statics (sampled lane), the multi-turn attacks, the padded statics.
  probe --layout bon_store512 --path <…/<private_run>/<cell>> --n-max 75 [--trial 0]
        bon_served_run.py's append-only `bon_responses.jsonl` (goal_id, sample_idx, trial): a goal is complete with >= n_max
        distinct draws at the trial; `bon_errors.jsonl` names the errored goals.
  probe --layout bon_climb_flat --path <…/bon_climb/<TAG>> --cell C --shard-index k [--shard-index k2 …] --nshards N [--trial 0]
        bon_climb.sbatch's FLAT per-cell files: expected = `plan/climb_manifest.jsonl` rows of the cell (class cap_coherent, the
        trial), sorted (goal_id, sample_idx), row i belongs to shard i % N (bon_climb_run.py); written = `out/<cell>/shard<k>.jsonl`
        rows carrying a response; a goal is complete when EVERY one of its shard draws is written; `out/<cell>/shard<k>.errors.jsonl`
        names the errored draws. One `--shard-index` per array task when every task registered its own run; every shard of
        the cell in ONE call (`--shard-index 0 --shard-index 1 …`) when the four tasks share one run
        — the document's `shard_indices` names what was asked.
  probe --layout capability_cellstore --path <cells/<key> dir> [--task T]
        the tinycap CellStore (sisteer.capeval.store): `rows.jsonl` rows (task, item_id, metric); unique item ids of the task, the
        `__aggregate__` row excluded. This IS the mmlu_full layout too ($WAVE_ROOT/mmlu_full/<TAG>_t<T>/cells/<key>, mmlu_full.sbatch):
        the store's `manifest.json` attests the full-width read — `resolved_config.mmlu.item_set` (the hashed width spec) and
        `mmlu_item_set.item_ids` (the ids that ran) — reported as `mmlu_item_set_spec` and `items_sha256` (item_sets.yaml's rule:
        sha256 over the ordered ids joined by a newline), so admission compares the ids to the frozen item set's, never n alone.
        A SAMPLED CONTROLS store (served_capability_run.py's judged controls: the smoke's xstest / or_bench_hard cells) holds NO
        rows.jsonl — the judge's file lives elsewhere; its item artifact is the generation spool `private/gens_<task basename>.jsonl`
        (line 1 `{"__spool_header__": …}`, then `{item_id, text}`), read EXACTLY as the producer's own complete-store reader reads it
        (scripts/sdfbase/full22/sampled/at_cap_controls.py read_control + _spool_texts + _meta_bound, precondition for precondition):
        gen_stats.json stamped with the control's entry naming that spool, `decode` / `n_at_cap` / `at_cap_item_ids` recorded, the
        header intact, every line parsed, no item spooled twice with different text, exactly `n_total` unique items, the meta sidecar
        `private/gens_<…>.meta.jsonl` binding every item by `text_sha256` with a boolean `at_cap`, and the sidecar's at-cap tally equal
        to gen_stats'. Any failure is `items.refusal`, by name, and `items.present` false — admission then refuses the run as the
        producer would. `items_sha256` stays null there (the spool is not an attestation; the id shas over the spool are reported under
        `items_sha256_spool_order` / `items_sha256_sorted` for the record).
  probe --layout misalign_responses --path <…/misalign/<TAG>/agentic/<harness cell>> --cell C --behavior B --n-samples 10
        sisteer.capeval.misalign.runner: `<path>/responses.jsonl` rows (item_id, sample_idx, behavior) — `path` is the CELL DIRECTORY the
        closer records as the run's store_uri (`<root>/agentic/<harness cell id>`: `si_<slug>`, `prompted__safety`, `none`), with
        `meta.json` {finished, items, n_samples, present} reported under `meta`; an item is complete with >= n_samples distinct draws. The
        legacy call (`path` = the agentic root, the rows under `<path>/<cell>/`) is still read, and said so in `notes`.
  probe --layout petri_cellstore --path <audit root>/cells/<petri_run_key> [--behavior B]
        the Petri runner's per-cell store on the GRADING host (sisteer.capeval.tiers.petri_runner over a tinycap CellStore; the
        petri_stats adapter records store_uri `~/.local/state/petri/petri_full22_<view>/cells/<run key>`, `~` expanded here):
        `rows.jsonl` rows (task `petri/<behavior>`, item_id = the scenario id [`#e<epoch>` when epochs > 1], metric = the behavior's
        propensity score | eval_awareness | scenario_realism — required_metrics, M13); `petri_incomplete.json` = the failure ledger
        (`unresolved`: [{behavior, scenario_id, epoch, dimension, …}], the runner's H9 contract: results.json exists only at zero
        unresolved); `petri_auditor_refused.json` / `petri_judge_refused.json` = TERMINAL refusals (settled, never retried — reported
        apart as `refused`, counted as covered like the jailbreak layout's `refused:*`); `results.json` = the cell-done marker. The
        work identity is `<behavior>/<scenario id>` (the adapter's item id): a scenario is complete when EVERY (epoch, required metric)
        is filed or terminally refused and the ledger holds it unresolved nowhere. The layout is derived from the
        real stores' artifacts; the expected ids come from the frozen scenario bundle (stage.py petri_scenario_ids).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

LAYOUTS = ("jailbreak_cellstore", "bon_store512", "bon_climb_flat", "capability_cellstore", "misalign_responses", "petri_cellstore")
#: the per-layout DONE PREDICATE — WHAT counts as done at the identity level, one versioned sentence per layout, recorded with
#: the inventory's coverage contract at freeze (stage.py coverage_contract) and compared by `06 --gate` before it reads a STARTED wave. A
#: reader or layout fix (WHERE the artifact is read, a file name, a directory) never touches these strings; changing WHAT counts as done
#: bumps `:v1` -> `:v2`, and the gate then refuses by name until the change is ruled on (a changed predicate is a new wave or a ruling).
DONE_PREDICATES = {
    "jailbreak_cellstore": ("jailbreak_cellstore:v1 — a goal is done at its LAST progress.jsonl record of the expected shard when that record's "
                            "outcome is terminal and not error:* (refused:* is terminal, reported apart); every frozen goal id of every expected shard"),
    "bon_store512": ("bon_store512:v1 — a goal is done with >= n_max distinct sample_idx draws at the trial in bon_responses.jsonl; every frozen "
                     "goal id"),
    "bon_climb_flat": ("bon_climb_flat:v1 — the expected set is the run-time plan/climb_manifest.jsonl's cap_coherent rows of the cell at the trial, "
                       "sorted (goal_id, sample_idx), row i in shard i % nshards; a goal is done when EVERY one of its shard draws is written with a "
                       "response in out/<cell>/shard<k>.jsonl; every expected goal of every bound array task's shard"),
    "capability_cellstore": ("capability_cellstore:v1 — the producer's complete item artifact (rows.jsonl, or a sampled-controls generation spool "
                             "the producer's gen_stats.json stamps complete and the meta sidecar binds) holds >= n_items unique item ids of the task "
                             "(the __aggregate__ row apart); the attested ids sha256 equals the frozen item set's when the freeze records one, else "
                             "by count"),
    "misalign_responses": ("misalign_responses:v1 — every expected item of the behavior has >= n_samples distinct sample_idx draws carrying a "
                           "response in the cell's responses.jsonl"),
    "petri_cellstore": ("petri_cellstore:v1 — a <behavior>/<scenario> is done when every (epoch, required metric) is filed in rows.jsonl or "
                        "terminally refused and the failure ledger holds it unresolved nowhere; every frozen scenario id"),
}
CONTROLS_META_SUFFIX = ".meta.jsonl"       # at_cap_controls.META_SUFFIX: the sampled controls' per-item sidecar beside the spool
CONTROLS_SPOOL_HEADER_KEY = "__spool_header__"   # sisteer.capeval.tiers.judged.GenSpool: line 1 of every generation spool
ERRORED_OUTCOME_PREFIX = "error:"      # sisteer.eval.jailbreak_artifacts.ERRORED_OUTCOME_PREFIX
REFUSED_OUTCOME_PREFIX = "refused:"    # sisteer.eval.jailbreak_artifacts.REFUSED_OUTCOME_PREFIX (terminal, never an error)
AGGREGATE_ITEM_ID = "__aggregate__"    # served_capability_run.py's per-store summary row
# the Petri runner's constants (sisteer.capeval.tiers.petri_runner — stdlib copy: this file runs from a mirrored checkout without the venv)
PETRI_TASK_PREFIX = "petri/"
PETRI_STANDARD_DIMENSIONS = ("eval_awareness", "scenario_realism")     # + the behavior's own score under metric=<behavior> (M13)
PETRI_FAILURE_LEDGER = "petri_incomplete.json"
PETRI_REFUSED_LEDGERS = (("petri_auditor_refused.json", "auditor_refused"), ("petri_judge_refused.json", "judge_refused"))   # (file, list key)


def read_jsonl(path: str) -> Iterable[dict]:
    """Every parseable object of a JSONL file; a torn tail (a write cut by SIGKILL) is skipped, never fatal."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _shard_block(**kw) -> dict:
    d = {"present": False, "completed": [], "errored": [], "refused": [], "expected": None, "notes": []}
    d.update(kw)
    return d


def probe_jailbreak_cellstore(path: str, shards: List[str]) -> dict:
    """`<path>/<shardNN>/progress.jsonl` per requested shard. When `path` is itself a shard directory (a run whose recorded store is
    one shard), its parent is the store — said so in the notes."""
    notes: List[str] = []
    base = os.path.basename(path.rstrip("/"))
    if base.startswith("shard") and base[5:].isdigit() and os.path.isfile(os.path.join(path, "progress.jsonl")):
        notes.append(f"path {path} is a shard directory; its parent is read as the store")
        path = os.path.dirname(path.rstrip("/"))
    out: Dict[str, dict] = {}
    for s in shards:
        pj = os.path.join(path, s, "progress.jsonl")
        blk = _shard_block(present=os.path.isfile(pj))
        if not blk["present"]:
            blk["notes"].append(f"no {pj}")
            out[s] = blk
            continue
        last: Dict[str, dict] = {}
        for rec in read_jsonl(pj):
            gid = rec.get("goal_id")
            if isinstance(gid, str) and gid:
                last[gid] = rec                                     # the LAST record per goal wins (jailbreak_run._load_progress)
        for gid, rec in sorted(last.items()):
            outcome = str(rec.get("outcome", ""))
            if outcome.startswith(ERRORED_OUTCOME_PREFIX):
                blk["errored"].append(gid)
            else:
                blk["completed"].append(gid)
                if outcome.startswith(REFUSED_OUTCOME_PREFIX):
                    blk["refused"].append(gid)
        out[s] = blk
    return {"layout": "jailbreak_cellstore", "path": path, "exists": os.path.isdir(path), "shards": out, "notes": notes}


def probe_bon_store512(path: str, n_max: int, trial: int = 0) -> dict:
    rp = os.path.join(path, "bon_responses.jsonl")
    blk = _shard_block(present=os.path.isfile(rp))
    draws: Dict[str, set] = {}
    for rec in read_jsonl(rp):
        if int(rec.get("trial", 0) or 0) != trial:
            continue
        gid = rec.get("goal_id")
        if isinstance(gid, str) and gid and "sample_idx" in rec:
            draws.setdefault(gid, set()).add(int(rec["sample_idx"]))
    errored = sorted({str(r.get("goal_id")) for r in read_jsonl(os.path.join(path, "bon_errors.jsonl")) if r.get("goal_id")})
    short = {g: len(s) for g, s in draws.items() if len(s) < n_max}
    blk.update(completed=sorted(g for g, s in draws.items() if len(s) >= n_max), errored=errored)
    if short:
        blk["notes"].append(f"{len(short)} goal(s) with fewer than n_max={n_max} draws: {dict(sorted(short.items())[:8])}")
    if not blk["present"]:
        blk["notes"].append(f"no {rp}")
    return {"layout": "bon_store512", "path": path, "exists": os.path.isdir(path), "shards": {"shard00": blk}, "notes": []}


def climb_expected_rows(manifest: str, cell: str, trial: int = 0) -> List[dict]:
    """The manifest rows of `cell` the climb works on — bon_climb_run.py's filter, byte for byte in effect: cell_id == cell, class
    cap_coherent (the default when absent), the trial; sorted (goal_id, sample_idx)."""
    rows = [r for r in read_jsonl(manifest)
            if r.get("cell_id") == cell and r.get("class", "cap_coherent") == "cap_coherent" and int(r.get("trial", 0) or 0) == trial]
    rows.sort(key=lambda r: (str(r["goal_id"]), int(r["sample_idx"])))
    return rows


def climb_shard_indices(shard_index) -> List[int]:
    """The task indices one climb probe reads: an int (the per-task registration — one run per array task), or several (the
    shared-run shape: the climb array's four tasks register ONE run per cell, so the gate reads EVERY shard of
    the cell in one call — `--shard-index 0 --shard-index 1 …`, or `0,1,2,3`). Order kept, repeats dropped."""
    if isinstance(shard_index, int):
        return [shard_index]
    out: List[int] = []
    for v in shard_index:
        for tok in str(v).split(","):
            tok = tok.strip()
            if tok and int(tok) not in out:
                out.append(int(tok))
    return out


def probe_bon_climb_flat(path: str, cell: str, shard_index, nshards: int, trial: int = 0) -> dict:
    """ONE document over the climb shards `shard_index` names (climb_shard_indices): `shards.<shardNN>` per index, each read exactly as the
    single-shard probe always did (expected = the manifest slice i % nshards == k, written = out/<cell>/shard<k>.jsonl), plus `shard_indices`,
    the indices this probe was asked for — admission tells a probe that read one shard because it was asked to from one that could read only
    one (a coverage_probe.py before this field takes the LAST --shard-index and names none)."""
    idx = climb_shard_indices(shard_index)
    shards: Dict[str, dict] = {}
    for k in idx:
        name, blk = _climb_shard(path, cell, k, nshards, trial)
        shards[name] = blk
    return {"layout": "bon_climb_flat", "path": path, "exists": os.path.isdir(path), "shards": shards, "shard_indices": idx, "notes": []}


def _climb_shard(path: str, cell: str, shard_index: int, nshards: int, trial: int):
    manifest = os.path.join(path, "plan", "climb_manifest.jsonl")
    out_file = os.path.join(path, "out", cell, f"shard{shard_index}.jsonl")
    err_file = os.path.join(path, "out", cell, f"shard{shard_index}.errors.jsonl")
    name = f"shard{shard_index:02d}"
    blk = _shard_block(present=os.path.isfile(out_file))
    if not os.path.isfile(manifest):
        blk["notes"].append(f"no climb manifest at {manifest} (shard 0 of the array writes it; without it the expected set is unknown)")
        return name, blk
    rows = climb_expected_rows(manifest, cell, trial)
    mine = [r for i, r in enumerate(rows) if i % nshards == shard_index]
    expected_draws = {(str(r["goal_id"]), int(r["sample_idx"])) for r in mine}
    written = {(str(r["goal_id"]), int(r["sample_idx"])) for r in read_jsonl(out_file)
               if r.get("goal_id") is not None and "sample_idx" in r and isinstance(r.get("response"), str)}
    errored_draws = {(str(r["goal_id"]), int(r["sample_idx"])) for r in read_jsonl(err_file) if r.get("goal_id") is not None and "sample_idx" in r}
    by_goal: Dict[str, set] = {}
    for g, s in expected_draws:
        by_goal.setdefault(g, set()).add(s)
    blk["expected"] = sorted(by_goal)
    blk["completed"] = sorted(g for g, ss in by_goal.items() if all((g, s) in written for s in ss))
    blk["errored"] = sorted({g for g, s in errored_draws if (g, s) not in written})
    missing = sorted(expected_draws - written)
    blk["notes"].append(f"manifest rows for cell {cell}: {len(rows)} (shard {shard_index}/{nshards}: {len(mine)} draws, {len(by_goal)} goals); "
                        f"written {len(written & expected_draws)}; missing {len(missing)}; errored {len(errored_draws)}")
    if not blk["present"]:
        blk["notes"].append(f"no {out_file}")
    return name, blk


def probe_capability_cellstore(path: str, task: Optional[str]) -> dict:
    rows_p = os.path.join(path, "rows.jsonl")
    items: Dict[str, set] = {}
    tasks: Dict[str, int] = {}
    n_rows = 0
    for rec in read_jsonl(rows_p):
        n_rows += 1
        t = str(rec.get("task", ""))
        tasks[t] = tasks.get(t, 0) + 1
        if task and t != task:
            continue
        iid = rec.get("item_id")
        if not isinstance(iid, str) or not iid or iid == AGGREGATE_ITEM_ID:
            continue
        items.setdefault(iid, set()).add(str(rec.get("metric", "")))
    sha = None; n_attested = None; spec = None; notes: List[str] = []
    mp = os.path.join(path, "manifest.json")
    if os.path.isfile(mp):
        try:
            man = json.load(open(mp, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            man = {}; notes.append(f"manifest.json unreadable: {exc}")
        blk = man.get("mmlu_item_set") if isinstance(man, dict) else None          # contracts.MANIFEST_KEY_MMLU_ITEM_SET
        ids = blk.get("item_ids") if isinstance(blk, dict) else None
        if isinstance(ids, list) and ids and all(isinstance(x, str) for x in ids):
            sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()          # item_sets.yaml items_sha256_rule
            n_attested = len(ids)
        rc = man.get("resolved_config") if isinstance(man, dict) else None
        mm = rc.get("mmlu") if isinstance(rc, dict) else None
        if isinstance(mm, dict) and isinstance(mm.get("item_set"), dict):
            spec = dict(mm["item_set"])                                              # contracts.IDENTITY_KEY_MMLU_ITEM_SET: the hashed width spec
    else:
        notes.append(f"no {mp}")
    if not os.path.isfile(rows_p):
        # no rows.jsonl — a SAMPLED CONTROLS store (the judge's rows live elsewhere): the item artifact is the generation spool,
        # read the producer's own way; `items.refusal` names what the producer would refuse, `items.present` is then false
        sampled = probe_sampled_controls_spool(path, task)
        if sampled is not None:
            notes.append(f"no {rows_p}; read the sampled-controls generation spool ({sampled.get('source') or 'private/gens_<task>.jsonl'})")
            return {"layout": "capability_cellstore", "path": path, "exists": os.path.isdir(path), "shards": {}, "items": sampled, "notes": notes}
        notes.append(f"no {rows_p}")
    return {"layout": "capability_cellstore", "path": path, "exists": os.path.isdir(path), "shards": {},
            "items": {"present": os.path.isfile(rows_p), "task": task, "completed": sorted(items), "n_rows": n_rows, "tasks_seen": tasks,
                      "metrics_seen": sorted({m for ms in items.values() for m in ms}), "items_sha256": sha, "n_attested_items": n_attested,
                      "mmlu_item_set_spec": spec, "source": "rows.jsonl" if os.path.isfile(rows_p) else None, "refusal": None},
            "notes": notes}


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _controls_spool_texts(path: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """at_cap_controls._spool_texts, stdlib copy: {item_id: text} of a COMPLETE spool, or (None, the producer's refusal) — header skipped,
    first write wins on an identical replay, a differing duplicate or a torn / corrupt line refuses (a complete store has neither)."""
    if not os.path.isfile(path):
        return None, f"{path}: no spool"
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    out: Dict[str, str] = {}
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None, f"{path}: corrupt or torn line {i + 1}; not a complete store"
        if i == 0:
            if not isinstance(row, dict) or CONTROLS_SPOOL_HEADER_KEY not in row:
                return None, f"{path}: first line is not a spool header"
            continue
        if not isinstance(row, dict) or "item_id" not in row or "text" not in row:
            return None, f"{path}: line {i + 1} carries no item_id / text; not a complete store"
        iid, text = row["item_id"], row["text"]
        if iid in out:
            if out[iid] != text:
                return None, f"{path}: item {iid!r} spooled twice with different text"
            continue
        out[iid] = text
    return out, None


def _controls_meta_bound(meta_path: str, texts: Dict[str, str]) -> Tuple[Optional[Dict[str, dict]], Optional[str]]:
    """at_cap_controls._meta_bound, stdlib copy: per spooled item the LAST meta row whose text_sha256 matches the spooled text (the driver's
    GenMeta.bound rule); a missing sidecar, a torn line or a spooled item with no matching row refuses."""
    if not os.path.isfile(meta_path):
        return None, (f"{meta_path}: no meta sidecar — a greedy store, or one written before the sidecar existed; no per-item at-cap record, "
                      f"refusing to read it as 0")
    bound: Dict[str, dict] = {}
    with open(meta_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh.read().splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                return None, f"{meta_path}: corrupt or torn line {i + 1}; not a complete store"
            if not isinstance(row, dict):
                continue
            iid = row.get("item_id")
            if iid in texts and row.get("text_sha256") == _text_sha(texts[iid]):
                bound[iid] = row
    missing = [i for i in texts if i not in bound]
    if missing:
        return None, f"{meta_path}: {len(missing)} spooled item(s) have no bound meta row (first: {missing[:3]}); refusing"
    return bound, None


def probe_sampled_controls_spool(path: str, task: Optional[str]) -> Optional[dict]:
    """The `items` block of a SAMPLED CONTROLS cell store (no rows.jsonl): `private/gens_<task basename>.jsonl` read EXACTLY as the producer's
    complete-store reader reads it (scripts/sdfbase/full22/sampled/at_cap_controls.py read_control: gen_stats.json stamped only after every
    control is complete, the control's entry naming this spool and carrying decode / n_at_cap / at_cap_item_ids, exactly n_total unique items,
    the meta sidecar binding every item by text_sha256 with a boolean at_cap, the sidecar's at-cap tally equal to gen_stats'; plus the closer's
    two-spools check — the entry's spool IS the cell's generation spool for the task). None when the store is not a sampled-controls store at
    all (no gen_stats.json and no private/ dir: the caller reports the plain `no rows.jsonl`); otherwise a block whose `refusal` names the first
    failed precondition by the producer's own words (`present` false), or the complete reading (`present` true, `completed` = the unique item
    ids). `items_sha256` is null — the spool attests nothing; the shas over its ids are reported apart (spool order and sorted)."""
    private = os.path.join(path, "private"); gs_path = os.path.join(path, "gen_stats.json")
    if not os.path.isdir(private) and not os.path.isfile(gs_path):
        return None
    stem = str(task or "").rsplit("/", 1)[-1]
    name = f"gens_{stem}.jsonl" if stem else ""
    out: dict = {"present": False, "task": task, "completed": [], "n_rows": 0, "tasks_seen": {}, "metrics_seen": [], "items_sha256": None,
                 "n_attested_items": None, "mmlu_item_set_spec": None, "source": f"private/{name}" if name else None, "refusal": None,
                 "gen_stats": None, "meta_bound": None, "n_unique": None, "items_sha256_spool_order": None, "items_sha256_sorted": None,
                 "items_sha256_source": None}

    def refuse(why: str) -> dict:
        out["refusal"] = why
        return out
    if not name:
        return refuse("no task named: a sampled-controls store's item artifact is private/gens_<task basename>.jsonl")
    if not os.path.isfile(gs_path):
        return refuse(f"{path}: no gen_stats.json — the producer stamps it only after every control is complete; this store is incomplete")
    try:
        with open(gs_path, encoding="utf-8") as fh:
            all_gs = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return refuse(f"{gs_path}: unreadable ({exc})")
    if not isinstance(all_gs, dict):
        return refuse(f"{gs_path}: not a JSON object")
    entries = [(k, v) for k, v in all_gs.items() if isinstance(v, dict) and os.path.basename(str(v.get("spool") or "")) == name]
    if not entries:
        return refuse(f"{gs_path}: no control entry names spool {name} (entries: {sorted(k for k, v in all_gs.items() if isinstance(v, dict))}) — "
                      f"the task's generation spool is not one this store stamps complete")
    if len(entries) > 1:
        return refuse(f"{gs_path}: {len(entries)} control entries name spool {name} ({[k for k, _ in entries]}) — ambiguous")
    control, gs = entries[0]
    if "decode" not in gs or "n_at_cap" not in gs or "at_cap_item_ids" not in gs:
        return refuse(f"{gs_path}: {control} carries no sampled decode record (decode/n_at_cap/at_cap_item_ids) — a GREEDY store has no per-item "
                      f"at-cap status and never reads as 0; refusing")
    # the closer's control_spool rule: the cell's ONE generation spool for the task (not the .caps. / .meta. sidecars) is the entry's spool
    gens = sorted(f for f in os.listdir(private) if f.startswith(f"gens_{stem}") and f.endswith(".jsonl") and ".caps." not in f and ".meta." not in f) \
        if os.path.isdir(private) else []
    if not gens:
        return refuse(f"{path}: no generation spool private/gens_{stem}*.jsonl")
    if gens[0] != name:
        return refuse(f"{path}: gen_stats.json names spool {name!r} for {control}, the cell's generation spool is {gens[0]!r} — two spools")
    spool_path = os.path.join(private, name)
    texts, why = _controls_spool_texts(spool_path)
    if why or texts is None:
        return refuse(why or f"{spool_path}: unreadable")
    try:
        n_total = int(gs.get("n_total", -1))
    except (TypeError, ValueError):
        return refuse(f"{gs_path}: {control} n_total={gs.get('n_total')!r} is not an integer")
    if len(texts) != n_total:
        return refuse(f"{spool_path}: {len(texts)} spooled items != gen_stats n_total={n_total}; incomplete or inconsistent store")
    meta_path = spool_path[:-len(".jsonl")] + CONTROLS_META_SUFFIX
    bound, why = _controls_meta_bound(meta_path, texts)
    if why or bound is None:
        return refuse(why or f"{meta_path}: unreadable")
    for i, r in bound.items():
        if not isinstance(r.get("at_cap"), bool):
            return refuse(f"{meta_path}: item {i!r} at_cap={r.get('at_cap')!r} is not a bool")
    at_cap = sorted(i for i, r in bound.items() if r.get("at_cap") is True)
    try:
        gs_ids = sorted(str(x) for x in (gs.get("at_cap_item_ids") or [])); gs_n = int(gs.get("n_at_cap"))
    except (TypeError, ValueError):
        return refuse(f"{gs_path}: {control} n_at_cap / at_cap_item_ids are not a count and an id list")
    if at_cap != gs_ids or len(at_cap) != gs_n:
        return refuse(f"{path}: {control} meta sidecar says {len(at_cap)} at cap but gen_stats says n_at_cap={gs.get('n_at_cap')} "
                      f"({len(gs_ids)} ids); inconsistent store, refusing")
    try:
        cap = int(gs.get("cap", (gs.get("decode") or {}).get("cap", -1)))
    except (TypeError, ValueError, AttributeError):
        cap = None
    ids = list(texts)                                                       # first-wins = spool order
    out.update(present=True, completed=sorted(ids), n_rows=len(ids), n_unique=len(ids), tasks_seen={str(task): len(ids)},
               gen_stats={"control": control, "n_total": n_total, "n_at_cap": len(at_cap), "cap": cap, "decode": "present"}, meta_bound=len(bound),
               items_sha256_spool_order=hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
               items_sha256_sorted=hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest(),
               items_sha256_source="the spool's own item ids (not an attestation: the store records no item-set id list)")
    return out


def probe_misalign_responses(path: str, cell: str, behavior: Optional[str], n_samples: int) -> dict:
    """`path` is the CELL DIRECTORY the closer records (`<root>/agentic/<harness cell>`): `<path>/responses.jsonl` + `meta.json`; the
    legacy call (the agentic root + `--cell`) is read at `<path>/<cell>/responses.jsonl` and said so."""
    notes: List[str] = []
    rp = os.path.join(path, "responses.jsonl")
    if not os.path.isfile(rp) and cell and os.path.isfile(os.path.join(path, cell, "responses.jsonl")):
        notes.append(f"{path} holds no responses.jsonl; read the legacy layout {os.path.join(path, cell)} (the recorded store is the harness cell dir)")
        rp = os.path.join(path, cell, "responses.jsonl")
    draws: Dict[str, set] = {}
    for rec in read_jsonl(rp):
        if behavior and str(rec.get("behavior", "")) != behavior:
            continue
        iid = rec.get("item_id")
        if isinstance(iid, str) and iid and "sample_idx" in rec and isinstance(rec.get("response"), str):
            draws.setdefault(iid, set()).add(int(rec["sample_idx"]))
    blk = _shard_block(present=os.path.isfile(rp), completed=sorted(i for i, s in draws.items() if len(s) >= n_samples))
    short = {i: len(s) for i, s in draws.items() if len(s) < n_samples}
    if short:
        blk["notes"].append(f"{len(short)} item(s) with fewer than n_samples={n_samples} draws")
    if not blk["present"]:
        blk["notes"].append(f"no {rp}")
    meta_blk = None
    mp = os.path.join(os.path.dirname(rp), "meta.json")
    if os.path.isfile(mp):
        meta, err = _read_json(mp)
        if err:
            notes.append(err)
        elif isinstance(meta, dict):
            meta_blk = {k: meta.get(k) for k in ("finished", "items", "n_samples", "present", "cell_id")}
    return {"layout": "misalign_responses", "path": path, "exists": os.path.isdir(path), "shards": {"shard00": blk}, "notes": notes, "meta": meta_blk}


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{os.path.basename(path)} unreadable: {exc}"


def petri_scenario_of(rec: dict) -> Optional[Tuple[str, str, int]]:
    """(behavior, scenario_id, epoch) of one rows.jsonl row, or None for a row that is not a Petri item row (the aggregate, another task)."""
    task = str(rec.get("task", ""))
    if not task.startswith(PETRI_TASK_PREFIX):
        return None
    iid = rec.get("item_id")
    if not isinstance(iid, str) or not iid or iid == AGGREGATE_ITEM_ID:
        return None
    ej = rec.get("extra_json")
    if isinstance(ej, str):
        try:
            ej = json.loads(ej)
        except json.JSONDecodeError:
            ej = {}
    ej = ej if isinstance(ej, dict) else {}
    sid = ej.get("scenario_id") if isinstance(ej.get("scenario_id"), str) and ej.get("scenario_id") else iid.split("#e", 1)[0]
    epoch = ej.get("epoch")
    if not isinstance(epoch, int):
        epoch = int(iid.rsplit("#e", 1)[1]) if "#e" in iid and iid.rsplit("#e", 1)[1].isdigit() else 1
    return task[len(PETRI_TASK_PREFIX):], sid, epoch


def probe_petri_cellstore(path: str, behavior: Optional[str] = None) -> dict:
    """One Petri cell store (`<audit root>/cells/<petri_run_key>`), read the way the runner writes it (module docstring). `behavior`
    restricts the report to one behavior (a `petri_<behavior>` eval); None = every behavior (the pooled eval)."""
    path = os.path.expanduser(path)
    notes: List[str] = []
    rows_p = os.path.join(path, "rows.jsonl")
    filed: Dict[Tuple[str, str], Dict[int, set]] = {}          # (behavior, scenario) -> epoch -> metrics filed
    behaviors_seen: Dict[str, int] = {}
    n_rows = 0
    for rec in read_jsonl(rows_p):
        n_rows += 1
        key = petri_scenario_of(rec)
        if key is None:
            continue
        b, sid, epoch = key
        behaviors_seen[b] = behaviors_seen.get(b, 0) + 1
        if behavior and b != behavior:
            continue
        filed.setdefault((b, sid), {}).setdefault(epoch, set()).add(str(rec.get("metric", "")))
    # the failure ledger: the runner's OWN list of open (behavior, scenario, epoch, dimension) — never complete while listed
    unresolved: Dict[Tuple[str, str], List[str]] = {}
    ledger, err = (_read_json(os.path.join(path, PETRI_FAILURE_LEDGER)) if os.path.isfile(os.path.join(path, PETRI_FAILURE_LEDGER)) else (None, None))
    if err:
        notes.append(err)
    ledger_present = isinstance(ledger, dict)
    for e in (ledger or {}).get("unresolved", []) if ledger_present else []:
        if isinstance(e, dict) and isinstance(e.get("behavior"), str) and isinstance(e.get("scenario_id"), str):
            if behavior and e["behavior"] != behavior:
                continue
            unresolved.setdefault((e["behavior"], e["scenario_id"]), []).append(f"e{e.get('epoch', '?')}:{e.get('dimension', '?')}")
    # terminal refusals: settled (excluded from the runner's completeness contract), reported apart
    refused: Dict[Tuple[str, str], set] = {}
    for fname, key in PETRI_REFUSED_LEDGERS:
        fp = os.path.join(path, fname)
        if not os.path.isfile(fp):
            continue
        doc, err = _read_json(fp)
        if err:
            notes.append(err); continue
        for r in (doc or {}).get(key, []) if isinstance(doc, dict) else []:
            if isinstance(r, dict) and isinstance(r.get("behavior"), str) and isinstance(r.get("scenario_id"), str):
                if behavior and r["behavior"] != behavior:
                    continue
                refused.setdefault((r["behavior"], r["scenario_id"]), set()).add(int(r.get("epoch", 1) or 1))
    # the cell-done marker + the per-behavior expectation it records (n_items / n_scenarios -> epochs)
    results_p = os.path.join(path, "results.json")
    results, err = (_read_json(results_p) if os.path.isfile(results_p) else (None, None))
    if err:
        notes.append(err)
    epochs_of: Dict[str, int] = {}
    for b, blk in ((results or {}).get("behaviors") or {}).items() if isinstance(results, dict) else []:
        try:
            n_items, n_sc = int(blk.get("n_items")), int(blk.get("n_scenarios"))
            if n_sc > 0 and n_items % n_sc == 0:
                epochs_of[str(b)] = n_items // n_sc
        except (TypeError, ValueError, AttributeError):
            pass
    completed: List[str] = []; errored: List[str] = []; refused_ids: List[str] = []; partial: Dict[str, str] = {}
    keys = sorted(set(filed) | set(unresolved) | set(refused))
    for b, sid in keys:
        wid = f"{b}/{sid}"
        required = (b, *PETRI_STANDARD_DIMENSIONS)
        ep_filed = filed.get((b, sid), {}); ep_ref = refused.get((b, sid), set())
        n_epochs = epochs_of.get(b) or max([*ep_filed.keys(), *ep_ref, 1])
        if (b, sid) in unresolved:
            errored.append(wid); partial[wid] = "unresolved in the failure ledger: " + ",".join(unresolved[(b, sid)])
            continue
        missing = [f"e{ep}:{m}" for ep in range(1, n_epochs + 1) if ep not in ep_ref for m in required if m not in ep_filed.get(ep, set())]
        if missing:
            partial[wid] = "missing " + ",".join(missing[:6])
            continue
        completed.append(wid)
        if ep_ref:
            refused_ids.append(wid)
    blk = _shard_block(present=os.path.isfile(rows_p), completed=completed, errored=errored, refused=refused_ids)
    if partial:
        blk["notes"].append(f"{len(partial)} scenario(s) not complete: {dict(sorted(partial.items())[:6])}")
    if not blk["present"]:
        blk["notes"].append(f"no {rows_p}")
    if not ledger_present:
        notes.append(f"no {PETRI_FAILURE_LEDGER} (the runner writes it at every filing; a cell never filed has none)")
    if results is None:
        notes.append("no results.json (the runner's cell-done marker: written only at zero unresolved required dimensions)")
    return {"layout": "petri_cellstore", "path": path, "exists": os.path.isdir(path), "shards": {"shard00": blk}, "behavior": behavior,
            "behaviors_seen": behaviors_seen, "n_rows": n_rows, "results_json_present": results is not None,
            "unresolved_n": sum(len(v) for v in unresolved.values()), "epochs_of": epochs_of, "notes": notes}


def probe(a) -> dict:
    if a.layout == "petri_cellstore":
        return probe_petri_cellstore(a.path, a.behavior or None)
    if a.layout == "jailbreak_cellstore":
        return probe_jailbreak_cellstore(a.path, [s for s in (a.shards or "").split(",") if s])
    if a.layout == "bon_store512":
        return probe_bon_store512(a.path, int(a.n_max), int(a.trial))
    if a.layout == "bon_climb_flat":
        return probe_bon_climb_flat(a.path, a.cell, climb_shard_indices(a.shard_index or ["0"]), int(a.nshards), int(a.trial))
    if a.layout == "capability_cellstore":
        return probe_capability_cellstore(a.path, a.task)
    if a.layout == "misalign_responses":
        return probe_misalign_responses(a.path, a.cell, a.behavior, int(a.n_samples))
    raise SystemExit(f"REFUSED: unknown layout {a.layout!r} ({', '.join(LAYOUTS)})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--layout", required=True, choices=LAYOUTS); p.add_argument("--path", required=True)
    p.add_argument("--shards", default=""); p.add_argument("--task", default=None); p.add_argument("--cell", default=None)
    p.add_argument("--behavior", default=None)
    # the climb's shard(s): one `--shard-index k` per array task, or several for the shared-run shape (repeat the flag, or `0,1,2,3`;
    # climb_shard_indices) — a probe before this read ONE int here, the last flag winning, so admission reads `shard_indices` off the document
    p.add_argument("--shard-index", action="append", default=None); p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--trial", type=int, default=0); p.add_argument("--n-max", type=int, default=75); p.add_argument("--n-samples", type=int, default=10)
    a = ap.parse_args(argv)
    if a.layout in ("bon_climb_flat", "misalign_responses") and not a.cell:
        print(f"REFUSED: --layout {a.layout} needs --cell", file=sys.stderr); return 2
    print(json.dumps(probe(a), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
