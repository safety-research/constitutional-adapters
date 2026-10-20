"""A PRIMED anchor frame for in-memory adapter tests (the shared emitter hook).

Since emitter 4 every declared contrast is decided by the pass's anchor currency (`ctx.anchor_currency()`): a context without a
registry connection makes a contrast-filing group refuse by name, and on an EMPTY frame every own anchor is `not_in_frame` and every
comparison parks (no provisional anchors -- the first pass over a fresh registry files no contrast; the pass after the fold
does). The in-memory adapter tests describe that steady state, so they read on a frame primed the way the fold would prime it:
`primed_ctx` reads the adapter once on an empty frame (a priming read: every comparison parks, nothing is written anywhere), loads every
`run` / `judgment` / `run_metric` record it emitted into a sidecar-shaped frame (`test_registry_anchor_current.Frame`: the TABLE
`v_run_current` of the sidecar) as current, complete, off-roster runs -- and, by default, every `item_result` record into the frame's
item objects (`Frame.item_objects`: the grading host's `v_item_result_current` / `v_item_conflict` shape, so the frame knows items and a family
paired over them, the store readers' second decision with e4 asked included, finds both sides' rows as the fold would hold them) -- and
hands back a context bound to it. The second read then sees every own anchor current (`own_current`) and files what the first pass after
an ingest would. Everything here is synthetic.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters.base import AdapterContext

from test_registry_anchor_current import Frame  # noqa: E402  (the sidecar-shaped in-memory frame; one DDL, shared)

NOT_IN_ROSTER = "not_in_roster"


def empty_frame() -> Frame:
    return Frame()


def load_records(frame: Frame, records: Iterable[Mapping[str, object]], *, current: bool = True, items: bool = True) -> Dict[str, int]:
    """Load the `run` / `judgment` / `run_metric` payloads of adapter records into `frame` as the fold would hold them: every run current
    (`is_current 1`, off-roster, not in the paper), `lifecycle_state` `ingested` where a reading or a judgment was filed for it (the
    harvester's `run_ingest`), the judgments `complete`, the metric rows verbatim -- and, when `items` (the default: the grading host knows items),
    the frame's item objects created and every `item_result` payload loaded as a current item row (the columns the item digest reads) and
    every `item_flags` payload as a current flags row (the columns the cap-evidence digest reads, `AC.FLAG_COLUMNS`).
    Idempotent on repeated keys (a run shared by two groups). Returns the counts loaded."""
    con = frame.con
    if items:
        frame.item_objects()
    have_items: Set[Tuple[object, ...]] = ({tuple(r) for r in con.execute("SELECT run_id, judgment_id, metric_id, item_id, trial_index, sample_index, turn_index FROM item_result")}
                                          if items else set())
    have_flags: Set[Tuple[object, ...]] = ({tuple(r) for r in con.execute("SELECT run_id, judgment_id, item_id, trial_index, sample_index, turn_index FROM item_flags")}
                                          if items else set())
    have_runs = {r[0] for r in con.execute("SELECT run_id FROM run")}
    have_j = {r[0] for r in con.execute("SELECT judgment_id FROM judgment")}
    have_m: Set[Tuple[object, ...]] = {tuple(r) for r in con.execute("SELECT run_id, judgment_id, metric_id, threshold, computation_id FROM run_metric")}
    runs: Dict[str, Mapping[str, object]] = {}
    judgments: List[Mapping[str, object]] = []
    metrics: List[Mapping[str, object]] = []
    item_rows: List[Mapping[str, object]] = []
    flag_rows: List[Mapping[str, object]] = []
    for rec in records:
        rt, p = rec.get("record_type"), rec.get("payload") or {}
        if rt == "run":
            runs.setdefault(str(p["run_id"]), p)
        elif rt == "judgment":
            judgments.append(p)
        elif rt == "run_metric":
            metrics.append(p)
        elif rt == "item_result" and items:
            item_rows.append(p)
        elif rt == "item_flags" and items:
            flag_rows.append(p)
    measured = {str(m["run_id"]) for m in metrics} | {str(j["run_id"]) for j in judgments}
    n = {"runs": 0, "judgments": 0, "metrics": 0, "items": 0, "flags": 0}
    for rid, p in runs.items():
        if rid in have_runs:
            continue
        lifecycle = "ingested" if rid in measured else str(p.get("lifecycle_state") or "generated")
        con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, p.get("model_id"), p.get("object_id"), p.get("eval_id"), p.get("instrument_id"), p.get("token_cap_tokens"), p.get("trial_index"),
                     p.get("pass_label"), p.get("decode_policy_id"), p.get("lane_source_legacy"), p.get("view_legacy"),
                     (str(p["cluster_job_id"]) if p.get("cluster_job_id") is not None else None), p.get("producer_run_key_legacy"), lifecycle,
                     p.get("generated_utc"), p.get("queued_utc")))
        con.execute("INSERT INTO v_run_current VALUES (?,?,?,?,?,?,?)", (rid, 1 if current else 0, None if current else "selection_state superseded (test)", 0, 0,
                                                                        NOT_IN_ROSTER if current else "superseded", "[]"))
        have_runs.add(rid)
        n["runs"] += 1
    for j in judgments:
        jid = str(j["judgment_id"])
        if jid in have_j:
            continue
        con.execute("INSERT INTO judgment VALUES (?,?,?,?,?)", (jid, j["run_id"], j.get("judge_id"), str(j.get("status") or "complete"), j.get("superseded_by_judgment_id")))
        have_j.add(jid)
        n["judgments"] += 1
    for m in metrics:
        key = (str(m["run_id"]), str(m.get("judgment_id") or ""), str(m["metric_id"]), float(m.get("threshold") if m.get("threshold") is not None else -1.0),
               str(m.get("computation_id") or ""))
        if key in have_m:
            continue
        con.execute("INSERT INTO run_metric VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*key, m.get("k"), m.get("n"), m.get("value"), m.get("ci_lo"), m.get("ci_hi"), str(m.get("computed_by") or "producer"), m.get("source_uri")))
        have_m.add(key)
        n["metrics"] += 1
    for p in item_rows:
        vals = AC.item_row_values(p)
        key = (str(p["run_id"]), str(p.get("judgment_id") or ""), str(p["metric_id"]), *vals[:4])
        if key in have_items:
            continue
        con.execute("INSERT INTO item_result VALUES (?,?,?,?,?,?,?,?,?)", (*key, vals[4], vals[5]))
        have_items.add(key)
        n["items"] += 1
    for p in flag_rows:
        fvals = AC.flag_row_values(p)
        fkey = (str(p["run_id"]), str(p.get("judgment_id") or ""), *fvals[:4])
        if fkey in have_flags:
            continue
        con.execute("INSERT INTO item_flags VALUES (?,?,?,?,?,?,?)", (*fkey, fvals[4]))
        have_flags.add(fkey)
        n["flags"] += 1
    con.commit()
    return n


def groups_of(adapter) -> Dict[str, list]:
    groups: Dict[str, list] = {}
    for i in adapter.discover():
        groups.setdefault(adapter.comparison_group(i), []).append(i)
    return groups


def priming_read(adapter, ctx: AdapterContext, groups: Optional[Mapping[str, Sequence]] = None) -> List[Mapping[str, object]]:
    """Read every group of `adapter` on `ctx` and return the records (the priming pass: on an empty frame every comparison parks). A group
    the adapter REFUSES contributes nothing -- exactly what the fold would hold for it (the runner writes nothing for a refused group); the
    test that reads that group afterwards sees the refusal itself."""
    from sisteer.registry.adapters.base import AdapterRefusal
    out: List[Mapping[str, object]] = []
    for g, ins in sorted((groups or groups_of(adapter)).items()):
        try:
            for batch in adapter.read(ctx, g, ins):
                out.extend(batch.records)
        except AdapterRefusal:
            continue
    return out


def primed_frame(adapter, *, now: str, producer: Mapping[str, object], groups: Optional[Mapping[str, Sequence]] = None, root: str = "") -> Frame:
    """A frame holding every run the adapter emits, as current, complete, off-roster runs -- built by a priming read on an empty frame."""
    frame = Frame()
    ctx0 = AdapterContext(root=root, now=now, producer=dict(producer), conn=frame.con)
    load_records(frame, priming_read(adapter, ctx0, groups))
    return frame


def primed_ctx(adapter, *, now: str, producer: Mapping[str, object], groups: Optional[Mapping[str, Sequence]] = None, root: str = "",
               frame: Optional[Frame] = None) -> AdapterContext:
    """The context an in-memory test reads with: bound to a frame primed from the adapter's own emissions (`primed_frame`), or to `frame`
    when the test built one. A fresh context per call shares the frame, so a test may read group by group."""
    fr = frame if frame is not None else primed_frame(adapter, now=now, producer=producer, groups=groups, root=root)
    return AdapterContext(root=root, now=now, producer=dict(producer), conn=fr.con)


def primed_ctx_for_rows(rows: Sequence[Mapping[str, object]], ctx: AdapterContext, **emitter_kw) -> AdapterContext:
    """A context primed from paper-long ROWS driven through `declared_common.emit_rows` directly (no adapter): a priming emission on an
    empty frame, its records loaded, the same context rebound to the primed frame."""
    from sisteer.registry.adapters import declared_common as DC
    from sisteer.registry.adapters.base import RecordBatch
    frame = Frame()
    ctx0 = AdapterContext(root=ctx.root, now=ctx.now, producer=dict(ctx.producer), conn=frame.con)
    batch = RecordBatch("priming:rows", "priming")
    DC.emit_rows(batch, ctx0, [dict(r) for r in rows], **emitter_kw)
    load_records(frame, batch.records)
    return AdapterContext(root=ctx.root, now=ctx.now, producer=dict(ctx.producer), conn=frame.con)


def make_current(frame: Frame, run_ids: Iterable[str]) -> int:
    """Give each run id a CURRENT currency row and nothing else (no `run` row): the shape of an anchor that is another source's run -- the
    shared bcbw readers' solo run, checked for currency and never re-picked -- so `check_currency` reads it current while it is no candidate
    of any exact-key search. Ids already in the frame are left alone. Returns the count added."""
    con = frame.con
    have = {r[0] for r in con.execute("SELECT run_id FROM v_run_current")}
    n = 0
    for rid in run_ids:
        if rid in have:
            continue
        con.execute("INSERT INTO v_run_current VALUES (?,?,?,?,?,?,?)", (rid, 1, None, 0, 0, NOT_IN_ROSTER, "[]"))
        have.add(rid)
        n += 1
    con.commit()
    return n


def retire(frame: Frame, run_id: str, why: str = "selection_state superseded (a later run at the same key)") -> None:
    """Retire one run of the frame (`is_current 0`, `why` as the fold spells it)."""
    frame.con.execute("UPDATE v_run_current SET is_current = 0, why = ?, selection_state = 'superseded' WHERE run_id = ?", (why, run_id))
    frame.con.commit()


def hold_out(frame: Frame, run_id: str, ruling: str = "r_synth_hold_out") -> None:
    """Hold one run of the frame out (`held_out 1`, `is_current 0`, the fold's `held_out` text)."""
    frame.con.execute("UPDATE v_run_current SET is_current = 0, held_out = 1, why = ? WHERE run_id = ?", (f"held_out (ruling {ruling})", run_id))
    frame.con.commit()


def current_runs(frame: Frame) -> Dict[str, Dict[str, object]]:
    """`run_id -> the frame's run row` of the current runs."""
    cols = list(AC.RUN_COLUMNS) + ["generated_utc", "queued_utc"]
    rows = frame.con.execute(f"SELECT {', '.join(AC.RUN_COLUMNS)}, generated_utc, queued_utc FROM run r WHERE run_id IN (SELECT run_id FROM v_run_current WHERE is_current = 1)")
    return {str(r[0]): dict(zip(cols, r)) for r in rows}
