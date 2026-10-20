#!/usr/bin/env python3
"""The GSM8K paired-200 paper-objects battery — the LADDER PLANNER (CPU-only, one run per cluster after
that cluster's 8,192 rows complete; ladder_plan.sbatch wraps it in the client venv).

Per LAUNCHED 8,192 row whose store under --wave-root is COMPLETE:
  * lane H's adapter ``inspect_rung.classify_store(out_root) -> (record, subset | None)`` — the ONE at-cap authority (the
    filed outcomes; this instrument's trigger n_at_cap / 200 > the threshold (1 %, harmonised 2026-09-12), the eligible set at_cap − loops − boundary, the transport
    verdict obtained directly) — decides; this planner WRITES the record to ``<out_root>/cap_ladder.json`` (the one writer of
    that file for this instrument, by construction) and, for an escalating cell, the subset to
    ``$WAVE_ROOT/rung16384/<TAG>/<cell_slug>/rung_subset.json`` bound to the source store (identity hash, cap, the ROW task
    ``contracts.GSM8K_STANDARD_TASK`` = ``inspect/gsm8k_paired200``, ids), validated by ``contracts.validate_item_subset``;
  * the rung's PREDICTED runtime and PADDED allocation from the ACTUAL subset — ``T_pred = standup_cold + ceil(n / S_rung) x
    (16,384 / rate) + probes`` with S_rung the model's resolved concurrency at the rung entry of the CELL's prompt class
    (``g8k_cells.geometry_for_cell``: the shared 18,944 entry for the short class, the long-SI 24,576 entry for the constitution
    prompted cell — the entry the launcher stands that rung up at, so the planner's S is the launcher's) and ``rate`` the per-sequence
    decode rate OBSERVED in the cell's 8,192 ledger (completion tokens / latency, the median over forwarded calls), and
    ``T_alloc = 2 x T_pred`` rounded UP to the half hour — emitted as the rung row's ``--time`` AUTOMATICALLY (no allocation
    length is a human gate); the watcher bound ``--max-hours = 2 x T_alloc`` printed beside it;
  * the rung's ``ladder_state.json`` (``rung: {state: planned, subset_sha16, source_identity_hash, n_ids, ...}``) in the rung's
    out root — the idempotence record: a rung in state running / complete is NEVER re-planned, a planned rung with a
    DIFFERING subset sha16 refuses by name, an identical one re-prints its line unchanged;
  * the ``sbatch`` lines per cluster with model-based placement (a Nemotron rung only under the primary cluster), the
    FIRST rung line separated (submitted alone at %1; G-READER-RUNG on its store before the rest are released).

Subdivision of a rung into two jobs is NOT automated (a documented contingency): ``--max-alloc-h`` refuses BY NAME a rung
whose T_alloc exceeds it and names ``split_subset`` — two DISJOINT id halves, each drawn exactly once, one merged reading.

  ladder_plan.py --wave-root <root> --sha <pin> --cluster cluster_a|cluster_b [--rows SPEC] [--max-alloc-h H] [--out DIR]
                 [--matrix default|llama] [--planner-tree-sha <sha>]

``--matrix`` (llama-paper-rows P11): the Llama matrix's rows, its rung lines naming the matrix. ``--sha`` is the pin the RUNG
ROWS run at (the 8,192 stores' producer tree, named in every emitted line); ``--planner-tree-sha`` is the tree THIS planner
classifies from when it is not that pin (the trigger harmonisation of 2026-09-12 re-classified a wave from a later tree while
its legs stayed at the pin) — recorded in plan.json and in every rung's ``planned_by``, beside the rule / threshold the cell was
classified under (the ROW's trigger, g8k_cells.row_trigger_threshold: a schema-2 matrix's declared one, the adapter's 1 % otherwise).
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import g8k_cells as M

# The PRODUCER's spellings, BY IMPORT (the launcher adopts the producer's constants, never re-types a literal): the
# subset schema and its exact field set, the subset digest, the ROW task the adapter files (`inspect/gsm8k_paired200`
# — NOT the unprefixed CLI spelling the cap flags carry), and this instrument's rule / threshold / report line as
# lane H's adapter serializes them. This planner runs in the client venv (ladder_plan.sbatch), where the adapter is
# importable; a tree without it refuses BY NAME here, before any plan.
try:
    from sisteer.capeval.tiers import inspect_rung as IR
    from sisteer.capeval.tinycap import contracts as C
except ImportError as _e:  # pragma: no cover — the by-name refusal a bare tree meets
    raise SystemExit(f"no sisteer.capeval.tiers.inspect_rung / tinycap.contracts on this tree ({_e}) — lane H's adapter is "
                     "the at-cap authority and the contract carries the subset schema; the planner does not run without them") from None

RUNG_SUBSET_SCHEMA = C.ITEM_SUBSET_SCHEMA
ITEM_SUBSET_FIELDS = C.ITEM_SUBSET_FIELDS
ROW_TASK = C.GSM8K_STANDARD_TASK
GSM8K_TRIGGER_THRESHOLD = IR.GSM8K_TRIGGER_THRESHOLD
GSM8K_REPORT_LINE = IR.GSM8K_REPORT_LINE
GSM8K_TRIGGER_RULE = IR.GSM8K_TRIGGER_RULE
#: the subset's digest — the harness's (sha256 of the sorted unique ids' canonical JSON, 16 hex; order-free), written into
#: the rung's ladder_state.json and compared by the analysis' reader against the rung identity's `cap_rung.subset_sha16`
subset_sha16 = C.item_subset_sha16
#: the sizing constants (a cold standup; the four tokenize header probes at ~1 min each)
STANDUP_COLD_S = 16 * 60
PROBES_S = 4 * 60
ALLOC_PAD = 2.0
ALLOC_ROUND_S = 30 * 60
WATCHER_PAD = 2.0
MIN_RATE_ROWS = 10
EXIT_ESCALATE = "escalate"
STATE_PLANNED, STATE_RUNNING, STATE_COMPLETE = "planned", "running", "complete"


# --------------------------------------------------------------------------- pure helpers (tested on their own)
def slurm_time(seconds: float) -> str:
    """Whole seconds -> Slurm's D-HH:MM:SS."""
    s = round(seconds)
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{sec:02d}"


def round_up(seconds: float, quantum: float = ALLOC_ROUND_S) -> float:
    return math.ceil(seconds / quantum - 1e-9) * quantum


def predict(n_eligible: int, s_rung: int, rate_tok_s: float, *, cap: int = M.RUNG_CAP, standup_s: float = STANDUP_COLD_S,
            probes_s: float = PROBES_S) -> dict:
    """The rung's predicted runtime and padded allocation: waves = ceil(n / S); T_pred = standup + waves x cap / rate
    + probes; T_alloc = 2 x T_pred rounded UP to the half hour; the watcher bound 2 x T_alloc. The all-200-eligible case at
    S = 18 and 12 tok/s: 16 + 12 x 22.76 + 4 min = 293 min -> T_alloc 9.8 h -> 10 h, watcher 20 h."""
    if n_eligible < 1:
        raise ValueError(f"n_eligible must be >= 1, got {n_eligible}")
    if s_rung < 1:
        raise ValueError(f"S_rung must be >= 1, got {s_rung}")
    if not (rate_tok_s > 0):
        raise ValueError(f"the observed decode rate must be > 0 tok/s, got {rate_tok_s!r}")
    waves = math.ceil(n_eligible / s_rung)
    t_pred = standup_s + waves * (cap / rate_tok_s) + probes_s
    t_alloc = round_up(ALLOC_PAD * t_pred)
    return {"n_eligible": n_eligible, "s_rung": s_rung, "rate_tok_s": rate_tok_s, "waves": waves, "cap": cap,
            "t_pred_s": t_pred, "t_pred_h": t_pred / 3600, "t_alloc_s": t_alloc, "t_alloc_h": t_alloc / 3600,
            "time_limit": slurm_time(t_alloc), "watcher_max_hours": WATCHER_PAD * t_alloc / 3600,
            "formula": f"T_pred = {standup_s / 60:g} + ceil({n_eligible}/{s_rung}) x {cap}/{rate_tok_s:g} tok/s + {probes_s / 60:g} min; "
                       f"T_alloc = {ALLOC_PAD:g} x T_pred rounded up to {ALLOC_ROUND_S / 60:g} min"}


def split_subset(ids: Sequence[str], halves: int = 2) -> list[list[str]]:
    """The two-half CONTINGENCY: disjoint halves that together are the subset once — each id drawn exactly once, one
    merged reading; a duplicated id in the subset refuses by name."""
    ids = list(ids)
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"subset carries duplicated id(s) {dup[:5]} — an item is drawn exactly once")
    if halves < 1:
        raise ValueError("halves must be >= 1")
    out = [ids[i::halves] for i in range(halves)]
    assert sorted(x for part in out for x in part) == sorted(ids)
    return out


def decode_rate_tok_s(ledger_rows: Sequence[Mapping[str, Any]], *, min_rows: int = MIN_RATE_ROWS) -> float:
    """The cell's OBSERVED per-sequence decode rate: the median of usage_completion_tokens / latency_s over the ledger's
    forwarded, non-rejected calls with a positive completion (contracts.LEDGER_ROW_FIELDS' names); refuses on too few rows."""
    rates = []
    for r in ledger_rows:
        if r.get("rejected"):
            continue
        comp, lat = r.get("usage_completion_tokens"), r.get("latency_s")
        if isinstance(comp, (int, float)) and isinstance(lat, (int, float)) and comp > 0 and lat > 0:
            rates.append(comp / lat)
    if len(rates) < min_rows:
        raise ValueError(f"only {len(rates)} ledger rows carry a completion and a latency (need >= {min_rows}) — no observed rate")
    return statistics.median(rates)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


@functools.lru_cache(maxsize=None)
def _s_at_geometry(model_key: str, geometry_eval: str, want_l: int) -> int:
    """S the standup will derive for this model at `geometry_eval` (the real table, never typed); refuses by name when the
    entry does not resolve to the L the launcher expects. Cached per (model, entry, L): the geometry IS part of the key, so
    two cells of one model at different entries never share an S."""
    from capctrl_serving_up import resolve_standup_geometry
    geom = resolve_standup_geometry(geometry_eval, None, None, model_key=model_key)
    if int(geom.max_model_len) != want_l:
        raise SystemExit(f"{geometry_eval} resolves L={geom.max_model_len}, not {want_l}")
    return int(geom.max_num_seqs)


def s_rung_for(model_key: str, cell: str, geometry_class: Optional[str] = None) -> int:
    """S the standup will derive for this model at the rung entry of the CELL's prompt class — `g8k_cells.geometry_for_cell(cell,
    RUNG_CAP)`, the same selection the launcher makes (the shared 18,944 entry for the short class; the constitution prompted
    cell's own 24,576 entry, where llama8b's S is 19, not the shared entry's 25). Review P2 (2026-09-12): resolved by model key
    alone, a constitution rung was sized at the wrong geometry — one wave / 1h30 planned where the served geometry needs two
    waves / 2h30 — and the emitted --time, the watcher bound and the --max-alloc-h verdict followed the wrong S. A row of a
    matrix that declares a `geometry_class` (qwen-seedrep-native-battery) is sized at the class's common rung entry instead."""
    if geometry_class is not None:
        geometry_eval, want_l = M.geometry_for_cell(cell, M.RUNG_CAP, model_key=model_key, geometry_class=geometry_class)
    else:
        geometry_eval, want_l = M.geometry_for_cell(cell, M.RUNG_CAP, model_key=model_key)     # the long-SI class is scoped by family (E2 / P3)
    return _s_at_geometry(model_key, geometry_eval, want_l)


# --------------------------------------------------------------------------- the stores
def store_status(out_root: Path) -> str:
    """`complete` when the producer's outcomes file exists and RESULT records COMPLETE; `pending` when neither; a RESULT
    COMPLETE without the outcomes file (or the reverse) refuses by name — two records of one fact disagreeing."""
    outcomes = sorted(out_root.glob(f"cells/*/{C.TASK_OUTCOMES_DIR}/{C.task_outcomes_filename(ROW_TASK)}"))
    result = out_root / "RESULT"
    complete_line = result.is_file() and any("result=COMPLETE" in ln for ln in result.read_text(encoding="utf-8").splitlines())
    if outcomes and complete_line:
        return "complete"
    if not outcomes and not complete_line:
        return "pending"
    raise SystemExit(f"{out_root}: RESULT says {'COMPLETE' if complete_line else 'not COMPLETE'} but the outcomes file is "
                     f"{'present' if outcomes else 'absent'} — the store's two records disagree; refusing to plan on it")


def atomic_write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def subset_ids(subset: Any) -> list[str]:
    """The adapter's subset as an id list: a Mapping with `ids`, or a sequence of ids; anything else refuses."""
    if isinstance(subset, Mapping):
        ids = subset.get("ids")
    else:
        ids = subset
    if not isinstance(ids, (list, tuple)) or not ids or not all(isinstance(i, str) for i in ids):
        raise SystemExit(f"the adapter's subset carries no non-empty id list ({type(subset).__name__})")
    if len(set(ids)) != len(ids):
        raise SystemExit("the adapter's subset repeats an id — an item is drawn exactly once")
    return list(ids)


# --------------------------------------------------------------------------- the plan
def plan_cell(row: dict, *, wave_root: Path, sha: str, classify: Callable[[Path], tuple], s_rung: int,
              max_alloc_h: Optional[float] = None, planner_tree_sha: Optional[str] = None) -> dict:
    """One launched 8,192 row -> its ladder verdict, and for an escalating cell the subset file, the sizing, the rung state
    and its sbatch line. Never re-plans a running / complete rung; refuses a differing subset on a planned one."""
    out_root = wave_root / row["out_root"]
    rung_root = wave_root / row["rung_out_root"]
    res: dict = {"index": row["index"], "job_id": row["job_id"], "tag": row["tag"], "cell": row["cell"], "model_key": row["model_key"],
                 "cluster_constraint": row["cluster_constraint"], "out_root": str(out_root), "status": None, "exit": None,
                 "subset_path": None, "n_ids": None, "sizing": None, "line": None, "note": None}
    status = store_status(out_root)
    res["status"] = status
    if status != "complete":
        res["note"] = "store not complete: not classified"
        return res
    # the ROW's trigger (g8k_cells.row_trigger_threshold: a schema-2 matrix's declared share; a schema-1 row reads the adapter's 1 %) goes to
    # the adapter; the default trigger is passed implicitly so a stub taking one argument still serves the published matrix
    trigger = M.row_trigger_threshold(row)
    rule = IR.gsm8k_trigger_rule(trigger)
    record, subset = classify(out_root, **({"trigger_threshold": trigger} if trigger != IR.GSM8K_TRIGGER_THRESHOLD else {}))
    if not isinstance(record, Mapping):
        raise SystemExit(f"{row['job_id']}: classify_store returned no record mapping ({type(record).__name__})")
    rec = dict(record)
    if rec.get("rule") not in (None, rule):
        raise SystemExit(f"{row['job_id']}: the adapter's record carries another rule ({str(rec.get('rule'))[:80]!r}) than this "
                         f"instrument's — refusing to file it")
    # the adapter is the AUTHOR of cap_ladder.json; this planner is its one WRITER: verbatim, nothing added (lane A reads the
    # adapter's fields; the planner's provenance rides the rung's ladder_state.json and the --out plan.json)
    atomic_write_json(out_root / M.CAP_LADDER_FILENAME, rec)
    res["exit"] = rec.get("exit")
    if rec.get("exit") != EXIT_ESCALATE:
        if subset is not None:
            raise SystemExit(f"{row['job_id']}: the adapter filed exit {rec.get('exit')!r} AND a subset — inconsistent; refusing")
        res["note"] = f"terminal record filed ({rec.get('exit')}); no rung"
        return res
    if subset is None:
        raise SystemExit(f"{row['job_id']}: the adapter filed exit escalate but no subset — refusing")
    ids = subset_ids(subset)
    src_block = rec.get("source") if isinstance(rec.get("source"), Mapping) else {}
    src_hash = ((subset.get("source_identity_hash") if isinstance(subset, Mapping) else None)
                or src_block.get("identity_hash") or rec.get("source_identity_hash"))
    if not isinstance(src_hash, str) or not src_hash:
        raise SystemExit(f"{row['job_id']}: neither the adapter's subset nor its record's source block carries the source identity "
                         "hash — the rung must be bound to the source store's identity (lane H's classify_store files it); "
                         "refusing to plan an unbound rung")
    sha16 = subset_sha16(ids)
    # idempotence: the rung's ladder_state.json (in the rung's out root) says whether this rung was planned / ran already
    state_path = rung_root / M.LADDER_STATE_FILENAME
    if state_path.is_file():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        rung_st = st.get("rung") or {}
        if rung_st.get("state") in (STATE_RUNNING, STATE_COMPLETE) or st.get("attempts"):
            res["note"] = f"rung already {rung_st.get('state') or 'attempted'} (attempts {len(st.get('attempts') or [])}) — not re-planned"
            res["subset_path"] = rung_st.get("subset_path"); res["n_ids"] = rung_st.get("n_ids")
            return res
        if rung_st.get("subset_sha16") and rung_st["subset_sha16"] != sha16:
            raise SystemExit(f"{row['job_id']}: a rung was already planned over subset {rung_st['subset_sha16']} but the adapter "
                             f"now yields {sha16} ({len(ids)} ids) — the source store changed under a planned rung; refusing")
    subset_path = wave_root / M.RUNG_DIR_REL / row["tag"] / row["cell_slug"] / M.RUNG_SUBSET_FILENAME
    if isinstance(subset, Mapping):
        # the adapter's validated rung_subset/1 body, VERBATIM (the driver validates EXACTLY its fields)
        doc = dict(subset)
        missing = [k for k in ITEM_SUBSET_FIELDS if k not in doc]
        if missing or doc.get("schema") != RUNG_SUBSET_SCHEMA:
            raise SystemExit(f"{row['job_id']}: the adapter's subset is not a {RUNG_SUBSET_SCHEMA} body (schema {doc.get('schema')!r}, "
                             f"missing {missing}) — refusing")
        # the binding to THIS row: the source store, the ROW task the producer files (contracts.GSM8K_STANDARD_TASK — the
        # classifier's subset carries `inspect/gsm8k_paired200`, never the unprefixed CLI spelling; review B-P1.2), the caps
        if str(doc.get("source_out_root") or "").rstrip("/") != str(out_root).rstrip("/") or int(doc.get("cap") or 0) != M.RUNG_CAP \
                or int(doc.get("source_cap") or 0) != M.CAP or doc.get("task") != ROW_TASK:
            raise SystemExit(f"{row['job_id']}: the adapter's subset binds to {doc.get('source_out_root')!r} / task {doc.get('task')!r} / "
                             f"{doc.get('source_cap')} -> {doc.get('cap')}, not this row's {out_root} / {ROW_TASK} / {M.CAP} -> {M.RUNG_CAP}")
    else:
        # a bare id sequence: the same body, EXACTLY the contract's fields, from the record's facts
        doc = {"schema": RUNG_SUBSET_SCHEMA, "task": ROW_TASK, "source_out_root": str(out_root), "source_identity_hash": src_hash,
               "source_cap": M.CAP, "cap": M.RUNG_CAP, "rule": rule, "threshold": trigger, "ids": ids,
               "n_at_cap": rec.get("n_at_cap"), "n_non_terminating": rec.get("n_non_terminating"), "n_boundary": rec.get("n_boundary", 0),
               "transport_check": rec.get("transport")}
    # the contract's OWN validator over the body this planner is about to write (the driver runs the same one at --item-subset:
    # a body it would refuse is refused here, before a rung is planned)
    try:
        C.validate_item_subset(doc, where=f"{row['job_id']} rung subset")
    except C.ItemSubsetError as e:
        raise SystemExit(f"{row['job_id']}: the rung subset body does not pass the contract's validator — {e}") from None
    if subset_path.is_file():
        prev = json.loads(subset_path.read_text(encoding="utf-8"))
        if subset_sha16(list(prev.get("ids") or [])) != sha16:
            raise SystemExit(f"{row['job_id']}: {subset_path} holds subset {subset_sha16(list(prev.get('ids') or []))} but the adapter "
                             f"yields {sha16}; refusing")
    else:
        atomic_write_json(subset_path, doc)
    # the sizing from the ACTUAL subset, S at the rung entry, the cell's observed rate
    ledger = out_root / M.LEDGER_FILENAME
    if not ledger.is_file():
        raise SystemExit(f"{row['job_id']}: no {M.LEDGER_FILENAME} under {out_root} — the observed decode rate comes from the 8,192 ledger")
    try:
        rate = decode_rate_tok_s(read_jsonl(ledger))
    except ValueError as e:
        raise SystemExit(f"{row['job_id']}: {e}") from None
    sizing = predict(len(ids), s_rung, rate)
    if max_alloc_h is not None and sizing["t_alloc_h"] > max_alloc_h:
        raise SystemExit(f"{row['job_id']}: T_alloc {sizing['t_alloc_h']:.2f} h exceeds --max-alloc-h {max_alloc_h} (the partition's "
                         f"walltime) — subdivision is a contingency, not automated: split_subset(ids, 2) gives two disjoint halves, "
                         "each drawn once, one merged reading; plan them by hand with their own rung roots")
    atomic_write_json(state_path, {"schema": "g8k_ladder_state/1", "attempts": [], "accepted_generation_attempt": None,
                                   "rung": {"state": STATE_PLANNED, "subset_path": str(subset_path), "subset_sha16": sha16, "n_ids": len(ids),
                                            "source_out_root": str(out_root), "source_identity_hash": src_hash, "cap": M.RUNG_CAP,
                                            "sizing": sizing, "planned_by": {"sha": sha, "script": "scripts/sdfbase/gsm8k/ladder_plan.py",
                                                                              "planner_tree_sha": planner_tree_sha or sha,
                                                                              "rule": rule, "threshold": trigger}}})
    res.update({"subset_path": str(subset_path), "n_ids": len(ids), "sizing": sizing})
    return res


def sbatch_lines(planned: list[dict], *, sha: str, cluster: str, matrix: Path = M.MATRIX_PATH) -> list[str]:
    """The rung lines for THIS cluster: one `sbatch --array` per T_alloc class (every task of an array shares its --time; each
    class's --time IS its rungs' T_alloc), model-based placement applied, the FIRST rung alone at %1; the trailing positionals
    are g8k_cells.array_tail's (`ARRAY <sha> 16384` on the published matrix; `... 16384 llama` on the Llama matrix). The QoS and
    log directory are the cluster's, from its block in the site config (g8k_cells.cluster_site)."""
    c = M.cluster_site(cluster)
    tail = M.array_tail(sha, rung=M.RUNG_CAP, matrix=matrix)
    mine = [p for p in planned if p.get("sizing") and not (p["cluster_constraint"] == M.PLACEMENT_PRIMARY and cluster != M.PRIMARY_CLUSTER)]
    if not mine:
        return [f"# no rung to submit on {cluster}"]
    script = f"{M.CHECKOUT_ROOT}/{sha}/{M.SBATCH_REL}"
    first, rest = mine[0], mine[1:]
    out = [(f"# FIRST RUNG (submit ALONE; run G-READER-RUNG on its store before releasing the rest): {first['job_id']} n={first['n_ids']} "
            f"T_pred {first['sizing']['t_pred_h']:.2f} h -> --time {first['sizing']['time_limit']}; watcher --max-hours {first['sizing']['watcher_max_hours']:g}"),
           (f"sbatch -J g8k-rung-first --qos={c['qos']} --time={first['sizing']['time_limit']} --output={c['logs']}/%x_%A_%a.out "
            f"--array={first['index']}%1 {script} {tail}")]
    by_time: dict[str, list[dict]] = {}
    for p in rest:
        by_time.setdefault(p["sizing"]["time_limit"], []).append(p)
    if rest:
        out.append("# REMAINING RUNGS (after G-READER-RUNG passes), one array per allocation class:")
    for tl, group in sorted(by_time.items(), key=lambda kv: -M.time_limit_hours(kv[0])):
        idx = sorted(p["index"] for p in group)
        wmax = max(p["sizing"]["watcher_max_hours"] for p in group)
        named = ", ".join(f"{p['job_id']} n={p['n_ids']}" for p in group)
        out.append(f"# {named}; watcher --max-hours {wmax:g}")
        out.append(f"sbatch -J g8k-rung --qos={c['qos']} --time={tl} --output={c['logs']}/%x_%A_%a.out "
                   f"--array={M._array_spec(idx)}%{len(idx)} {script} {tail}")
    skipped = [p for p in planned if p.get("sizing") and p not in mine]
    for p in skipped:
        out.append(f"# {p['job_id']}: cluster_constraint primary — its rung line is printed under {M.PRIMARY_CLUSTER} only")
    return out


def plan_all(rows: list[dict], *, wave_root: Path, sha: str, cluster: str, classify: Callable[[Path], tuple],
             s_rung_of: Callable[[str, str], int], max_alloc_h: Optional[float] = None, matrix: Path = M.MATRIX_PATH,
             planner_tree_sha: Optional[str] = None) -> tuple[list[dict], list[str]]:
    """Every launched row's plan; `s_rung_of(model_key, cell)` is the rung concurrency at the ROW's cell (its prompt class picks
    the geometry entry — `s_rung_for`; a stub in tests). A row that carries a matrix-declared `geometry_class` passes it as a
    keyword (qwen-seedrep-native-battery: the common rung entry); every other row calls the two-argument form, byte for byte."""
    planned = [plan_cell(r, wave_root=wave_root, sha=sha, classify=classify,
                         s_rung=(s_rung_of(r["model_key"], r["cell"], geometry_class=M.row_geometry_class(r)) if M.row_geometry_class(r)
                                 else s_rung_of(r["model_key"], r["cell"])),
                         max_alloc_h=max_alloc_h, planner_tree_sha=planner_tree_sha)
               for r in rows if r["status"] == M.STATUS_LAUNCH]
    return planned, sbatch_lines(planned, sha=sha, cluster=cluster, matrix=matrix)


def summary_lines(planned: list[dict]) -> list[str]:
    out = []
    for p in planned:
        s = p.get("sizing")
        size = (f" n={p['n_ids']} S={s['s_rung']} rate={s['rate_tok_s']:.1f} tok/s waves={s['waves']} T_pred={s['t_pred_h']:.2f}h "
                f"--time={s['time_limit']} watcher={s['watcher_max_hours']:g}h") if s else ""
        out.append(f"LADDER: row={p['index']} {p['job_id']} status={p['status']} exit={p['exit']}{size}" + (f" note={p['note']}" if p.get("note") else ""))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--wave-root", type=Path, required=True)
    ap.add_argument("--sha", required=True, help="the pin the rung rows run at (the 8,192 stores' producer tree)")
    ap.add_argument("--cluster", required=True, choices=sorted(M.CLUSTERS))
    ap.add_argument("--rows", default=None, help="a Slurm index spec of matrix rows to plan (default: every launched row)")
    ap.add_argument("--max-alloc-h", type=float, default=None, help="the partition's max walltime in hours (off): a rung above it refuses by name")
    ap.add_argument("--out", type=Path, default=None, help="where to also write plan.json + sbatch_lines.txt")
    ap.add_argument("--matrix", default=None, help=f"the matrix: a name ({', '.join(M.MATRICES)}) or a path (default: the published matrix)")
    ap.add_argument("--planner-tree-sha", default=None,
                    help="the tree this planner classifies from when it is not --sha (recorded, never a substitute for --sha)")
    a = ap.parse_args(argv)
    M._check_sha(a.sha)
    if a.planner_tree_sha is not None:
        M._check_sha(a.planner_tree_sha)
    matrix = M.resolve_matrix(a.matrix)
    try:
        from sisteer.capeval.tiers import inspect_rung
    except ImportError as e:
        raise SystemExit(f"no sisteer.capeval.tiers.inspect_rung on this tree ({e}) — lane H's adapter is the at-cap authority; "
                         "the planner does not classify without it") from None
    classify = getattr(inspect_rung, "classify_store", None)
    if classify is None:
        raise SystemExit("inspect_rung carries no classify_store(out_root) -> (record, subset | None)")
    rows = M.matrix_rows(matrix)
    if a.rows is not None:
        rows = [rows[i] for i in M.parse_index_spec(a.rows, len(rows))]
    planned, lines = plan_all(rows, wave_root=a.wave_root.resolve(), sha=a.sha, cluster=a.cluster, classify=classify,
                              s_rung_of=s_rung_for, max_alloc_h=a.max_alloc_h, matrix=matrix, planner_tree_sha=a.planner_tree_sha)
    # ONE trigger per matrix (g8k_cells.load_matrix holds every row to the document's): the rows' is the plan's
    triggers = sorted({M.row_trigger_threshold(r) for r in rows})
    if len(triggers) != 1:
        raise SystemExit(f"the rows to plan carry {len(triggers)} trigger(s) {triggers} — one trigger per matrix")
    rule = IR.gsm8k_trigger_rule(triggers[0])
    if a.planner_tree_sha is not None and a.planner_tree_sha != a.sha:
        print(f"NOTE: classified from tree {a.planner_tree_sha} (rule: {rule!r}); the rung rows run at {a.sha}")
    for line in summary_lines(planned) + lines:
        print(line)
    n_rung = sum(1 for p in planned if p.get("sizing"))
    print(f"PLAN: rows={len(planned)} complete={sum(p['status'] == 'complete' for p in planned)} rungs={n_rung} cluster={a.cluster} "
          f"matrix={M.matrix_name_of(matrix)}")
    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(a.out / "plan.json", {"sha": a.sha, "cluster": a.cluster, "matrix": M.matrix_name_of(matrix), "planned": planned, "lines": lines,
                                                "planner_tree_sha": a.planner_tree_sha or a.sha, "rule": rule, "threshold": triggers[0]})
        (a.out / "sbatch_lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
