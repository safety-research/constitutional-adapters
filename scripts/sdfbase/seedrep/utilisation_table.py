#!/usr/bin/env python3
"""The utilisation table of the Qwen seed-replicate native battery.

    utilisation_table.py --stores RS/stores --out RS/tables/utilisation.csv --s5 RS/smokes.json
        --expected-hours RS/expected_h200h.json [--period-s 10] [--fail-on-bars]

Per STANDUP (every pulled accepted attempt root) from the two in-container samplers and the driver's own count:

* `vllm_metrics.jsonl` — one JSON object per scrape `{ts, num_requests_running, num_requests_waiting,
  generation_tokens_total, prompt_tokens_total, gpu_cache_usage_perc}` (the `vllm:`-prefixed gauge names are
  accepted as spelled by the scraper; `ts` ISO-8601 or epoch seconds);
* `driver_inflight.jsonl` — EITHER the cluster driver's per-tick in-flight request count `{ts, inflight}` (basis
  `driver_inflight`: a scrape is saturated when the nearest count is ≥ S), OR the served app's child event log
  `{t, event: start|exit, child, live[, rc]}` (`seedrep_served.py`; `live` = concurrent CHILDREN, not requests), for which
  DRIVER DEMAND is rebuilt exactly as the cluster tick log measured it (fu2): demand(t) = Σ over the children live at
  t (from their `start` to their `exit` event) of each child's request concurrency as the ROOT RECORDS it — the resolved
  `--concurrency` / `--max-concurrency` of `children/<child>.argv.json`, the argv the app handed to Popen — and a scrape is
  saturated when demand(t) ≥ S (basis `served_children_live_concurrency`; the per-scrape demand series and the per-child
  concurrency go into `inflight_counts`). A child live at any scrape whose concurrency the root does NOT record makes the
  Running bar `unmeasured(driver demand not recorded on the root)` — never an engine-occupancy fallback (running + waiting
  measures requests already inside vLLM, not outstanding demand). The two schemas are validated over the WHOLE file before
  a basis is chosen: a row lacking a required field, a hybrid row or a mixture refuses by name, the row shown;
* `gpu_dmon.log` — `nvidia-smi dmon -i 0 -s um -d 10 -o DT` (the `#Date Time gpu sm mem …` header names the columns;
  with `-o DT` every row carries its own timestamp — a log WITHOUT timestamps is placed from `ATTEMPT.json`'s
  `samplers_started_utc` (`start + k × period`) when recorded, else (fu2 (a): the served app runs dmon WITHOUT `-o DT` and
  records no start; both samplers start in ONE thread at the same 10 s cadence and the metrics scrape IS timestamped)
  row k at `first metrics scrape + k × period` (basis `dmon_rows_aligned_to_metrics_cadence`, alignment uncertainty ≤ one
  period, recorded on the row as `dmon_alignment_uncertainty_s`); fewer rows than the scrape span needs (more than one
  cadence short) refuses by name — the rows cannot be placed; trailing rows past the span are counted, not placed).
* `COMPLETE.json` — the wall time's end: the cluster's `written_utc`, or the served app's `completed_utc` (fu2 (b); which
  one was read is recorded as `complete_ts_basis`).

The row: median `Running` over the scrapes whose nearest in-flight count is ≥ S (the bar: ≥ 0.8 S; when in-flight
NEVER reached S the bar is `unmeasured` and counts as unmet — the client never saturated the engine); median SM %
over the dmon rows nearest to scrapes with `Running > 0` (decode windows; bar ≥ 85 %); AGGREGATE decode tok/s =
total eligible Δ`generation_tokens_total` / total eligible elapsed over consecutive scrape pairs whose earlier scrape
was decoding (`Running > 0`), Δt > 0 and the counter monotone (a stall — tokens flat while decoding — is an eligible
interval that LOWERS the aggregate; a counter reset is ineligible and lowers the SAMPLING COVERAGE = eligible elapsed /
the decode window), and its ratio to the S5 REFERENCE of the standup's window class (`--s5 RS/smokes.json`, the
`S5.references[]` records the `S5` smoke writes and `seedrep_launch_plan.s5_gap` keys on (L, S, thinking) — each a MEASURED
aggregate decode rate (:data:`S5_REFERENCE_KEYS`: method `aggregate_decode_tok_s`, ≥ 20 min, a complete store, steady-state
sampling coverage, the interval count, W / wave / attempt provenance held to the standup's); a bare number, a percentile, a
partial store, a low-coverage window or a hand-mirrored value is NO reference and refuses by name; bar ≥ 0.8
of S5 — the bar is defined on the AGGREGATE throughput, never a median of interval rates, which a stall cannot
move; the median is reported beside it; coverage below `MIN_SAMPLING_COVERAGE` makes the bar `unmeasured`; a window WITHOUT
an S5 reference refuses the table — plan gate 8); H200-h actual (`COMPLETE.json.written_utc` − `ATTEMPT.json.started_utc`, one H200 per container) vs the
sized hours (`--expected-hours`: `{"<leg>/<view>/<stage>/<shard>": h}` with `<leg>/<view>/<stage>` and `<leg>` fallbacks;
a standup without an expectation refuses) — the FOURTH bar: projected H200-h ≤ 1.5 × the sized hours (§8 U); the
kernel-ceiling S (`geometry.adjustments` `lora_kernel_index_ceiling` `to_S`) beside the S the KV budget would have
allowed (`from_S`) — the haircut reported as the stack limitation it is.

Every bar is `true` / `false` / `unmeasured(<why>)`, never blank; `bars_met` is true only when ALL FOUR bars (Running,
SM, aggregate tok/s, projected hours) are true. `--fail-on-bars` exits 1 when any standup's bars are unmet (U is
FAIL-CLOSED). Missing inputs exit 2.
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_common as ac  # noqa: E402
from audit_common import AttemptRoot, Standup, refuse  # noqa: E402

COLUMNS: Tuple[str, ...] = (
    "leg", "view", "stage", "shard", "attempt_id", "S", "L", "window", "thinking", "n_scrapes", "n_decode_scrapes",
    "n_saturated_scrapes", "inflight_basis", "inflight_counts", "median_running_saturated", "running_bar", "median_sm_decode", "sm_bar", "tok_s_aggregate",
    "n_eligible_intervals", "sampling_coverage", "tok_s_median", "tok_s_mean", "s5_tok_s", "tok_s_ratio_s5", "tok_bar",
    "h200_h_actual", "h200_h_expected", "h200_h_ratio", "hours_bar", "bars_met", "kernel_from_S", "kernel_to_S",
    "kernel_haircut", "dmon_ts_basis", "dmon_alignment_uncertainty_s", "complete_ts_basis", "problems",
    # the standup record's provenance (F-AA): `standup_jobs.jsonl`, or the derivation S / L / thinking were read under
    "standup_record")
RUNNING_BAR_FRACTION = 0.8
SM_BAR_PCT = 85.0
TOK_BAR_FRACTION = 0.8
#: the U bar: projected H200-h ≤ 1.5 × the sized hours
HOURS_BAR_RATIO = 1.5
#: the share of the decode window the eligible intervals must cover for the aggregate tok/s to be a MEASUREMENT (a counter
#: reset or a non-monotone scrape drops its interval; below this the bar is `unmeasured(sampling coverage …)`)
MIN_SAMPLING_COVERAGE = 0.8
_METRIC_KEYS = {
    "running": ("num_requests_running", "vllm:num_requests_running", "running"),
    "waiting": ("num_requests_waiting", "vllm:num_requests_waiting", "waiting"),
    "gen_total": ("generation_tokens_total", "vllm:generation_tokens_total", "gen_total"),
    "prompt_total": ("prompt_tokens_total", "vllm:prompt_tokens_total", "prompt_total"),
    "kv": ("gpu_cache_usage_perc", "vllm:gpu_cache_usage_perc", "kv"),
}
_TS_KEYS = ("ts", "t", "time", "timestamp")
_INFLIGHT_KEYS = ("inflight", "in_flight", "n_inflight")


def parse_ts(v: Any, where: str) -> float:
    """ISO-8601 (`Z` or offset) or epoch seconds → epoch seconds (float)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            refuse(f"{where}: timestamp {v!r} is neither ISO-8601 nor epoch seconds")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    refuse(f"{where}: timestamp {v!r} is not a string or number")


def _pick(row: Mapping[str, Any], keys: Sequence[str], where: str, *, required: bool = True) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    if required:
        refuse(f"{where}: row lacks any of {list(keys)}: {json.dumps(row, default=str)[:160]}")
    return None


def read_metrics(path: Path) -> List[Dict[str, float]]:
    rows = ac.read_jsonl(path)
    if not rows:
        refuse(f"{path}: empty — the engine sampler wrote no scrape")
    out = []
    for r in rows:
        out.append({"ts": parse_ts(_pick(r, _TS_KEYS, str(path)), str(path)),
                    "running": float(_pick(r, _METRIC_KEYS["running"], str(path))),
                    "waiting": float(_pick(r, _METRIC_KEYS["waiting"], str(path))),
                    "gen_total": float(_pick(r, _METRIC_KEYS["gen_total"], str(path))),
                    "prompt_total": float(_pick(r, _METRIC_KEYS["prompt_total"], str(path), required=False) or 0.0),
                    "kv": float(_pick(r, _METRIC_KEYS["kv"], str(path), required=False) or 0.0)})
    out.sort(key=lambda r: r["ts"])
    return out


def read_inflight(path: Path) -> List[Tuple[float, float]]:
    """The CLUSTER driver's `{ts, inflight}` rows as `(ts, inflight)`; a row without an in-flight key refuses by name."""
    rows = ac.read_jsonl(path)
    if not rows:
        refuse(f"{path}: empty — the driver recorded no in-flight count")
    out = sorted((parse_ts(_pick(r, _TS_KEYS, str(path)), str(path)), float(_pick(r, _INFLIGHT_KEYS, str(path)))) for r in rows)
    return out


INFLIGHT_BASIS_DRIVER = "driver_inflight"
INFLIGHT_BASIS_SERVED = "served_children_live_concurrency"
DEMAND_NOT_RECORDED = "driver demand not recorded on the root"
_SERVED_EVENTS = ("start", "exit")
_SERVED_REQUIRED = ("t", "event", "child", "live")
_CONCURRENCY_FLAGS = ("--concurrency", "--max-concurrency")


def _row_brief(r: Any) -> str:
    return json.dumps(r, default=str)[:160]


def inflight_schema(rows: Sequence[Any], where: str) -> str:
    """Which of the two DISJOINT shapes the WHOLE file has (fu2): ``cluster`` = every row carries a timestamp key and exactly
    one in-flight key and no `event`; ``served`` = every row carries `t`, `event` in {start, exit}, a string `child`, an integer `live`
    and no in-flight key. A row lacking a required field, a hybrid row (both an event and an in-flight key) or a mixture of the two
    shapes refuses by name with the offending row shown."""
    if not rows:
        refuse(f"{where}: empty — the driver recorded no in-flight count")
    kinds: List[str] = []
    for i, r in enumerate(rows):
        if not isinstance(r, Mapping):
            refuse(f"{where}: row {i} is not an object: {_row_brief(r)}")
        has_in = [k for k in _INFLIGHT_KEYS if k in r]
        has_ev = "event" in r
        if has_in and has_ev:
            refuse(f"{where}: row {i} is a hybrid (an in-flight key {has_in} AND an event) — neither the cluster tick log nor the served event log: {_row_brief(r)}")
        if has_in:
            if len(has_in) > 1:
                refuse(f"{where}: row {i} carries {len(has_in)} in-flight keys {has_in}; the cluster tick log carries one: {_row_brief(r)}")
            if not any(k in r for k in _TS_KEYS):
                refuse(f"{where}: row {i} (cluster shape) lacks a timestamp key {list(_TS_KEYS)}: {_row_brief(r)}")
            kinds.append("cluster")
        elif has_ev:
            missing = [k for k in _SERVED_REQUIRED if k not in r]
            if missing:
                refuse(f"{where}: row {i} (served shape) lacks {missing}: {_row_brief(r)}")
            if r["event"] not in _SERVED_EVENTS:
                refuse(f"{where}: row {i} event {r['event']!r} is not one of {list(_SERVED_EVENTS)} (a teardown / failure event is not a child lifetime): {_row_brief(r)}")
            if not isinstance(r["child"], str) or not r["child"]:
                refuse(f"{where}: row {i} child {r['child']!r} is not a name: {_row_brief(r)}")
            if not isinstance(r["live"], int) or isinstance(r["live"], bool):
                refuse(f"{where}: row {i} live {r['live']!r} is not an integer: {_row_brief(r)}")
            kinds.append("served")
        else:
            refuse(f"{where}: row {i} lacks any of {list(_INFLIGHT_KEYS)} and is not a served child event ({list(_SERVED_REQUIRED)}): {_row_brief(r)}")
    if len(set(kinds)) > 1:
        first_other = next(i for i, k in enumerate(kinds) if k != kinds[0])
        refuse(f"{where}: rows mix the cluster tick log and the served event log (row 0 is {kinds[0]}, row {first_other} is {kinds[first_other]}): "
               f"{_row_brief(rows[first_other])}")
    return kinds[0]


def child_concurrency(root_path: Path, child: str) -> Tuple[Optional[int], Optional[str]]:
    """The request concurrency the ROOT records for ``child``: `children/<child>.argv.json` (the argv the served app handed to Popen)
    naming `--concurrency N` / `--max-concurrency N` (or `--flag=N`) -> ``(N, flag)``; no record, or an argv without such a flag ->
    ``(None, why)``. Nothing is recomputed from the plan and nothing is inferred from the driver program."""
    p = Path(root_path) / "children" / f"{child}.argv.json"
    if not p.is_file():
        return None, f"no children/{child}.argv.json"
    doc = ac.read_json(p)
    argv = doc.get("argv") if isinstance(doc, Mapping) else None
    if not isinstance(argv, list):
        return None, f"children/{child}.argv.json carries no argv"
    argv = [str(a) for a in argv]
    for i, a in enumerate(argv):
        for flag in _CONCURRENCY_FLAGS:
            val: Optional[str] = None
            if a == flag and i + 1 < len(argv):
                val = argv[i + 1]
            elif a.startswith(flag + "="):
                val = a[len(flag) + 1:]
            if val is not None:
                try:
                    n = int(val)
                except ValueError:
                    return None, f"children/{child}.argv.json: {flag} {val!r} is not an integer"
                return n, flag
    return None, f"children/{child}.argv.json names no {list(_CONCURRENCY_FLAGS)} flag"


def served_demand_series(rows: Sequence[Mapping[str, Any]], *, root_path: Path, metrics: Sequence[Mapping[str, float]], where: str
                         ) -> Tuple[List[Tuple[float, float]], Dict[str, Any], List[str]]:
    """Driver demand per metrics scrape from the served event log (fu2): a child is live from its `start` to its `exit`
    (an unexited child stays live to the end); demand(t) = Σ live children's recorded concurrency. Returns ``(series, counts,
    unrecorded)`` — ``unrecorded`` names the children live at some scrape whose concurrency the root does not record (the caller
    makes the Running bar `unmeasured(...)`; the series is then not a measurement)."""
    starts: Dict[str, float] = {}
    exits: Dict[str, float] = {}
    events: Dict[str, int] = {}
    for r in rows:
        t = parse_ts(r["t"], where)
        events[str(r["event"])] = events.get(str(r["event"]), 0) + 1
        (starts if r["event"] == "start" else exits).setdefault(str(r["child"]), t)
    children = sorted(starts)
    conc: Dict[str, Optional[int]] = {}
    source: Dict[str, Optional[str]] = {}
    for c in children:
        conc[c], source[c] = child_concurrency(root_path, c)
    series: List[Tuple[float, float]] = []
    unrecorded: set = set()
    for m in metrics:
        t = m["ts"]
        live = [c for c in children if starts[c] <= t and (c not in exits or exits[c] > t)]
        for c in live:
            if conc[c] is None:
                unrecorded.add(c)
        series.append((t, float(sum(conc[c] or 0 for c in live))))
    counts = {"n_events": len(rows), "events": events, "n_children": len(children), "n_children_unexited": len([c for c in children if c not in exits]),
              "per_child_concurrency": conc, "concurrency_source": source, "n_scrapes": len(series),
              "demand_series": [[round(t, 3), d] for t, d in series], "max_demand": (max(d for _, d in series) if series else None),
              "children_without_recorded_concurrency": sorted(unrecorded)}
    return series, counts, sorted(unrecorded)


def read_inflight_series(path: Path, *, metrics: Sequence[Mapping[str, float]], root_path: Path
                         ) -> Tuple[List[Tuple[float, float]], str, Dict[str, Any], List[str]]:
    """The per-time in-flight DEMAND series the Running bar selects its saturated scrapes with, from whichever of the two shapes
    `driver_inflight.jsonl` has (:func:`inflight_schema`): ``(series, basis, counts, unrecorded)``. Cluster: the driver's own count
    (:func:`read_inflight`), basis ``driver_inflight``. Served: :func:`served_demand_series`, basis ``served_children_live_concurrency``."""
    rows = ac.read_jsonl(path)
    shape = inflight_schema(rows, str(path))
    if shape == "cluster":
        series = read_inflight(path)
        return series, INFLIGHT_BASIS_DRIVER, {"n_rows": len(series), "max_inflight": max(v for _, v in series)}, []
    series, counts, unrecorded = served_demand_series(rows, root_path=root_path, metrics=metrics, where=str(path))
    return series, INFLIGHT_BASIS_SERVED, counts, unrecorded


DMON_BASIS_DT = "dmon:-o DT"
DMON_BASIS_START = "samplers_started_utc+k*period"
DMON_BASIS_METRICS = "dmon_rows_aligned_to_metrics_cadence"


def read_dmon(path: Path, *, period_s: float, start_utc: Optional[str], metrics: Optional[Sequence[Mapping[str, float]]] = None
              ) -> Tuple[List[Tuple[float, float, float]], str, Dict[str, Any]]:
    """`(ts, sm_pct, mem_pct)` rows, the timestamp basis (`dmon:-o DT`, `samplers_started_utc+k*period`, or fu2 (a)
    `dmon_rows_aligned_to_metrics_cadence`: row k at the first metrics scrape + k × period) and the alignment record
    (`uncertainty_s`: 0 for timestamped rows, else one period; for the metrics alignment the row count against the scrape span:
    `n_rows`, `n_rows_span_needs`, `n_rows_beyond_span`)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [l.rstrip("\n") for l in text.splitlines() if l.strip()]
    if not lines:
        refuse(f"{path}: empty — the GPU sampler wrote nothing")
    header: Optional[List[str]] = None
    for l in lines:
        if l.startswith("#") and "sm" in l.split():
            header = [c.lower() for c in l.lstrip("#").split()]
            break
    if header is None:
        refuse(f"{path}: no `# … sm …` column header — not an `nvidia-smi dmon` log")
    has_ts = "date" in header and "time" in header
    sm_i, mem_i = header.index("sm"), (header.index("mem") if "mem" in header else None)
    data: List[Tuple[Optional[float], float, float]] = []
    for l in lines:
        if l.startswith("#"):
            continue
        parts = l.split()
        if len(parts) != len(header):
            continue  # a partial line (a killed sampler's tail)
        try:
            sm = float(parts[sm_i])
            mem = float(parts[mem_i]) if mem_i is not None else 0.0
        except ValueError:
            continue  # `-` for an unsupported field
        ts: Optional[float] = None
        if has_ts:
            d, t = parts[header.index("date")], parts[header.index("time")]
            try:
                ts = datetime.strptime(f"{d} {t}", "%Y%m%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                refuse(f"{path}: unparseable dmon Date/Time {d!r} {t!r}")
        data.append((ts, sm, mem))
    if not data:
        refuse(f"{path}: no data row")
    if has_ts:
        return [(float(ts), sm, mem) for ts, sm, mem in data if ts is not None], DMON_BASIS_DT, {"uncertainty_s": 0.0}
    if start_utc:
        t0 = parse_ts(start_utc, "ATTEMPT.json.samplers_started_utc")
        return [(t0 + k * period_s, sm, mem) for k, (_, sm, mem) in enumerate(data)], DMON_BASIS_START, {"uncertainty_s": float(period_s)}
    if not metrics:
        refuse(f"{path}: dmon rows carry no timestamps (run `nvidia-smi dmon … -o DT`) and ATTEMPT.json records no "
               "samplers_started_utc — the SM% rows cannot be placed against the engine scrapes")
    # fu2 (a): the served app's samplers start in ONE thread at one cadence; the scrape is timestamped, dmon is not -- row k sits at the
    # first scrape + k * period. The row count must cover the scrape span (fewer rows than the span needs, beyond one cadence, cannot be
    # placed: dmon started late or died); rows past the span are the sampler outliving the scrape loop at teardown -- counted, unused
    t0, t1 = metrics[0]["ts"], metrics[-1]["ts"]
    needs = int(round((t1 - t0) / period_s)) + 1
    if len(data) < needs - 1:
        refuse(f"{path}: {len(data)} dmon rows cannot cover the {needs} periods of the metrics scrape span ({t1 - t0:.0f} s at {period_s:g} s) — "
               "the SM% rows cannot be placed against the engine scrapes (dmon started late or died)")
    align = {"uncertainty_s": float(period_s), "n_rows": len(data), "n_rows_span_needs": needs, "n_rows_beyond_span": len(data) - needs}
    return [(t0 + k * period_s, sm, mem) for k, (_, sm, mem) in enumerate(data)], DMON_BASIS_METRICS, align


def nearest(series_ts: Sequence[float], series_val: Sequence[float], t: float, *, within: float) -> Optional[float]:
    if not series_ts:
        return None
    i = bisect.bisect_left(series_ts, t)
    cands = []
    if i < len(series_ts):
        cands.append(i)
    if i > 0:
        cands.append(i - 1)
    j = min(cands, key=lambda k: abs(series_ts[k] - t))
    return series_val[j] if abs(series_ts[j] - t) <= within else None


def _bar(ok: Optional[bool], why: str = "") -> str:
    if ok is None:
        return f"unmeasured({why})"
    return "true" if ok else "false"


#: THE S5 REFERENCE RECORD — one per planned window class, written by the smoke that measures it (`RS/smokes.json` `S5.references[]`,
#: the list `seedrep_launch_plan.s5_gap` keys on (L, S, thinking)) and CONSUMED here. Every key is REQUIRED: the
#: reference is the MEASURED aggregate decode rate (the same statistic this table computes) over a COMPLETE store at steady
#: state for ≥ 20 min of real traffic on THIS tree and wave — a bare number, a percentile or median, a partial store, a
#: low-coverage window or a hand-mirrored value is no reference and refuses by name.
S5_REFERENCE_METHOD = "aggregate_decode_tok_s"
S5_MIN_DURATION_S = 20 * 60
S5_REFERENCE_KEYS: Tuple[str, ...] = ("L", "S", "thinking", "method", "tok_s_aggregate", "duration_s", "store_complete",
                                      "sampling_coverage", "n_eligible_intervals", "W", "wave_uuid", "attempt_id")
S5_FORBIDDEN_SOURCES: Tuple[str, ...] = ("manual", "mirror", "mirrored", "hand")


def s5_reference_problems(rec: Any, *, W: Optional[str] = None, wave_uuid: Optional[str] = None) -> List[str]:
    """Why `rec` is not an S5 reference (empty = it is one); `W` / `wave_uuid` = the standup's, which the record must name."""
    if not isinstance(rec, Mapping):
        return [f"not an object ({type(rec).__name__}) — a bare number is not a reference"]
    out: List[str] = []
    missing = [k for k in S5_REFERENCE_KEYS if k not in rec]
    if missing:
        out.append(f"lacks {missing}")

    def pos_int(k: str) -> bool:
        v = rec.get(k)
        return isinstance(v, int) and not isinstance(v, bool) and v >= 1

    def num(k: str) -> Optional[float]:
        v = rec.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    for k in ("L", "S", "n_eligible_intervals"):
        if k in rec and not pos_int(k):
            out.append(f"{k} {rec.get(k)!r} is not a positive int")
    if "thinking" in rec and not isinstance(rec.get("thinking"), bool):
        out.append(f"thinking {rec.get('thinking')!r} is not a bool")
    if "method" in rec and rec.get("method") != S5_REFERENCE_METHOD:
        out.append(f"method {rec.get('method')!r} is not {S5_REFERENCE_METHOD!r} — a percentile, a median or any other statistic is not the "
                   "aggregate decode rate")
    if "tok_s_aggregate" in rec and not (num("tok_s_aggregate") or 0) > 0:
        out.append(f"tok_s_aggregate {rec.get('tok_s_aggregate')!r} is not a positive number")
    if "duration_s" in rec and (num("duration_s") is None or num("duration_s") < S5_MIN_DURATION_S):  # type: ignore[operator]
        out.append(f"duration_s {rec.get('duration_s')!r} < {S5_MIN_DURATION_S} (plan §8 U: 20 min of real traffic)")
    if "store_complete" in rec and rec.get("store_complete") is not True:
        out.append("store_complete is not true — a partial store is a floor, not a reference")
    if "sampling_coverage" in rec and (num("sampling_coverage") is None or num("sampling_coverage") < MIN_SAMPLING_COVERAGE):  # type: ignore[operator]
        out.append(f"sampling_coverage {rec.get('sampling_coverage')!r} < {MIN_SAMPLING_COVERAGE} — not steady state")
    w = rec.get("W")
    if "W" in rec and (not isinstance(w, str) or len(w) != 40):
        out.append(f"W {w!r} is not the 40-hex tree sha")
    elif W is not None and w != W:
        out.append(f"W {w} is not the standup's tree {W}")
    wv = rec.get("wave_uuid")
    if "wave_uuid" in rec and (not isinstance(wv, str) or not wv):
        out.append(f"wave_uuid {wv!r} is not a wave")
    elif wave_uuid is not None and wv != wave_uuid:
        out.append(f"wave_uuid {wv} is not the standup's wave {wave_uuid}")
    if "attempt_id" in rec and (not isinstance(rec.get("attempt_id"), str) or not rec.get("attempt_id")):
        out.append(f"attempt_id {rec.get('attempt_id')!r} names no S5 attempt")
    src = rec.get("source", rec.get("origin"))
    if (isinstance(src, str) and src.strip().lower() in S5_FORBIDDEN_SOURCES) or rec.get("manual") is True:
        out.append(f"source {src!r} — a manual or mirrored number is not a measurement")
    return out


def s5_reference(s5: Mapping[str, Any], *, L: int, S: int, thinking: str, root: AttemptRoot) -> float:
    """The S5 aggregate decode rate for the standup's window class: from `RS/smokes.json` (`S5.references[]`; a file
    holding the block alone, `{"references": [...]}`, reads the same) the ONE record keyed (L, S, thinking), validated by
    :func:`s5_reference_problems` against the standup's W and wave. None (plan gate 8), several, another thinking, or an
    invalid record refuse by name."""
    refs: Any = None
    if isinstance(s5.get("S5"), Mapping):
        refs = s5["S5"].get("references")
    elif "references" in s5:
        refs = s5.get("references")
    if not isinstance(refs, list):
        refuse(f"--s5: not RS/smokes.json's S5 block ({{\"S5\": {{\"references\": [...]}}}}, one {S5_REFERENCE_METHOD} record per "
               f"(L, S, thinking); keys seen: {sorted(map(str, s5))[:6]}) — a `<L>x<S>: tok_s` scalar map is not a reference")
    want_th = thinking == "on"
    same_ls = [r for r in refs if isinstance(r, Mapping) and r.get("L") == L and r.get("S") == S]
    hits = [r for r in same_ls if r.get("thinking") is want_th]
    if not hits:
        if same_ls:
            refuse(f"S5 reference(s) for window {L}x{S} carry thinking {[r.get('thinking') for r in same_ls]}, not the standup's {want_th}")
        refuse(f"no S5 decode-rate reference for window {L}x{S} ({thinking}) — U cannot judge this standup (plan gate 8: "
               "planned standups − S5 references must be ∅)")
    if len(hits) > 1:
        refuse(f"{len(hits)} S5 references for window {L}x{S} ({thinking}) — one measurement per window class")
    problems = s5_reference_problems(hits[0], W=root.W, wave_uuid=root.wave_uuid)
    if problems:
        refuse(f"S5 reference for window {L}x{S} ({thinking}) is not a measurement: " + "; ".join(problems))
    return float(hits[0]["tok_s_aggregate"])


def expected_hours(exp: Mapping[str, Any], root: AttemptRoot) -> float:
    for k in (f"{root.leg}/{root.view}/{root.stage}/{root.shard}", f"{root.leg}/{root.view}/{root.stage}", root.leg):
        v = exp.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    refuse(f"--expected-hours has no entry for {root.label} (tried <leg>/<view>/<stage>/<shard>, <leg>/<view>/<stage>, <leg>)")


def thinking_of(standup: Standup) -> str:
    tm = standup.endpoint.get("thinking_mode")
    if tm is None:
        refuse(f"{standup.endpoint_path}: thinking_mode missing — the S5 window class needs it")
    s = str(tm).lower()
    return "on" if s in ("on", "thinking_on", "true", "1") else "off"


def utilisation_row(root: AttemptRoot, *, s5: Mapping[str, Any], expected: Mapping[str, Any], period_s: float) -> Dict[str, Any]:
    standup = ac.read_standup(root)
    S, L = standup.S, standup.L
    thinking = thinking_of(standup)
    metrics = read_metrics(ac.find_one(root.path, ac.METRICS_FILE))  # type: ignore[arg-type]
    inflight, in_basis, in_counts, unrecorded = read_inflight_series(ac.find_one(root.path, ac.INFLIGHT_FILE), metrics=metrics, root_path=root.path)  # type: ignore[arg-type]
    dmon, dmon_basis, dmon_align = read_dmon(ac.find_one(root.path, ac.DMON_FILE), period_s=period_s,  # type: ignore[arg-type]
                                             start_utc=root.attempt.get("samplers_started_utc"), metrics=metrics)
    problems: List[str] = []
    in_ts = [t for t, _ in inflight]
    in_v = [v for _, v in inflight]
    saturated = []
    if not unrecorded:                                 # a demand series with an unrecorded child is not a measurement: no scrape is selected
        for m in metrics:
            v = nearest(in_ts, in_v, m["ts"], within=period_s)
            if v is not None and v >= S:
                saturated.append(m["running"])
    decode = [m for m in metrics if m["running"] > 0]
    dm_ts = [t for t, _, _ in dmon]
    dm_sm = [sm for _, sm, _ in dmon]
    sm_decode = [v for v in (nearest(dm_ts, dm_sm, m["ts"], within=period_s) for m in decode) if v is not None]
    # the AGGREGATE decode rate over CONSECUTIVE scrape pairs (every scrape, so a gap in decoding between two decode scrapes
    # is not stitched over): eligible = the earlier scrape was decoding, Δt > 0 and the token counter did not go backwards
    rates: List[float] = []
    tok_sum = elapsed_eligible = 0.0
    n_pairs_decode = 0
    for a, b in zip(metrics, metrics[1:]):
        if a["running"] <= 0:
            continue
        n_pairs_decode += 1
        dt = b["ts"] - a["ts"]
        if dt > 0 and b["gen_total"] >= a["gen_total"]:
            inc = b["gen_total"] - a["gen_total"]
            rates.append(inc / dt)
            tok_sum += inc
            elapsed_eligible += dt
    window = (decode[-1]["ts"] - decode[0]["ts"]) if len(decode) >= 2 else 0.0
    coverage = (elapsed_eligible / window) if window > 0 else None
    tok_agg = (tok_sum / elapsed_eligible) if elapsed_eligible > 0 else None
    ref = s5_reference(s5, L=L, S=S, thinking=thinking, root=root)
    med_running = statistics.median(saturated) if saturated else None
    med_sm = statistics.median(sm_decode) if sm_decode else None
    tok_med = statistics.median(rates) if rates else None
    tok_mean = statistics.fmean(rates) if rates else None
    running_ok = None if med_running is None else med_running >= RUNNING_BAR_FRACTION * S
    sm_ok = None if med_sm is None else med_sm >= SM_BAR_PCT
    ratio = None if tok_agg is None else tok_agg / ref if ref > 0 else None
    tok_why = "tok/s unmeasured"
    if ratio is not None and coverage is not None and coverage < MIN_SAMPLING_COVERAGE:
        tok_why = f"sampling coverage {coverage:.2f} < {MIN_SAMPLING_COVERAGE}"
        tok_ok: Optional[bool] = None
    else:
        tok_ok = None if ratio is None else ratio >= TOK_BAR_FRACTION
    running_why = "in-flight never >= S"
    if unrecorded:
        running_why = DEMAND_NOT_RECORDED
        problems.append(f"{DEMAND_NOT_RECORDED}: {len(unrecorded)} child(ren) live during the scrapes have no recorded request concurrency "
                        f"(children/<child>.argv.json --concurrency) — first: {unrecorded[:3]}; the Running bar is unmeasured")
    elif med_running is None:
        what = "in-flight" if in_basis == INFLIGHT_BASIS_DRIVER else f"driver demand ({in_basis})"
        problems.append(f"in-flight never reached S={S} (max {what} {max(in_v):g}) — the Running bar is unmeasured; the client did not saturate the engine")
    if med_sm is None:
        problems.append("no dmon row falls within a period of a decode scrape — SM% unmeasured")
    if tok_agg is None:
        problems.append("no eligible decode interval (fewer than two scrapes with the earlier one decoding) — tok/s unmeasured")
    elif tok_ok is None:
        problems.append(f"eligible intervals cover {coverage:.2f} of the decode window (< {MIN_SAMPLING_COVERAGE}: counter resets / "
                        f"non-monotone scrapes dropped {n_pairs_decode - len(rates)} of {n_pairs_decode}) — the aggregate tok/s is not a measurement")
    comp = ac.complete_record(root)
    if comp is None:
        refuse(f"{root.path}: {ac.COMPLETE_FILE} missing or without written_utc — the standup's wall time is unknown")
    complete_basis = next((k for k in ("written_utc", "completed_utc") if comp.get(k) is not None), None)     # fu2 (b): the cluster's, or the served app's
    if complete_basis is None:
        refuse(f"{root.path}: {ac.COMPLETE_FILE} missing or without written_utc (nor the served app's completed_utc) — the standup's wall time is unknown")
    if root.started_utc is None:
        refuse(f"{root.path}/{ac.ATTEMPT_FILE}: started_utc missing")
    hours = (parse_ts(comp[complete_basis], f"COMPLETE.json.{complete_basis}") - parse_ts(root.started_utc, "ATTEMPT.json.started_utc")) / 3600.0
    if hours < 0:
        refuse(f"{root.path}: COMPLETE.json.written_utc precedes ATTEMPT.json.started_utc")
    exp_h = expected_hours(expected, root)
    kern = ac.kernel_ceiling_adjustment(standup.endpoint)
    if kern is None:
        refuse(f"{standup.endpoint_path}: no lora_kernel_index_ceiling adjustment — the kernel-ceiling S and the KV S are unrecorded")
    from_S, to_S = int(kern["from_S"]), int(kern["to_S"])
    h_ratio = (hours / exp_h) if exp_h > 0 else None
    hours_ok = None if h_ratio is None else h_ratio <= HOURS_BAR_RATIO
    if hours_ok is None:
        problems.append("--expected-hours entry is 0 — the projected-hours bar is unmeasured")
    elif not hours_ok:
        problems.append(f"H200-h actual {hours:.3f} is {h_ratio:.2f}x the sized {exp_h} (bar: <= {HOURS_BAR_RATIO}x)")
    bars_met = bool(running_ok) and bool(sm_ok) and bool(tok_ok) and bool(hours_ok)
    return {"leg": root.leg, "view": root.view, "stage": root.stage, "shard": root.shard, "attempt_id": root.attempt_id,
            "S": S, "L": L, "window": f"{L}x{S}", "thinking": thinking, "n_scrapes": len(metrics), "n_decode_scrapes": len(decode),
            "n_saturated_scrapes": len(saturated), "inflight_basis": in_basis, "inflight_counts": in_counts,
            "median_running_saturated": "unmeasured" if med_running is None else round(med_running, 2),
            "running_bar": _bar(running_ok, running_why),
            "median_sm_decode": "unmeasured" if med_sm is None else round(med_sm, 1), "sm_bar": _bar(sm_ok, "no aligned dmon row"),
            "tok_s_aggregate": "unmeasured" if tok_agg is None else round(tok_agg, 1), "n_eligible_intervals": len(rates),
            "sampling_coverage": "unmeasured" if coverage is None else round(coverage, 3),
            "tok_s_median": "unmeasured" if tok_med is None else round(tok_med, 1),
            "tok_s_mean": "unmeasured" if tok_mean is None else round(tok_mean, 1), "s5_tok_s": ref,
            "tok_s_ratio_s5": "unmeasured" if ratio is None else round(ratio, 3), "tok_bar": _bar(tok_ok, tok_why),
            "h200_h_actual": round(hours, 3), "h200_h_expected": exp_h,
            "h200_h_ratio": "expected 0" if h_ratio is None else round(h_ratio, 3), "hours_bar": _bar(hours_ok, "expected hours 0"),
            "bars_met": bars_met, "kernel_from_S": from_S, "kernel_to_S": to_S,
            "kernel_haircut": round(1.0 - to_S / from_S, 3) if from_S else "from_S 0", "dmon_ts_basis": dmon_basis,
            "dmon_alignment_uncertainty_s": dmon_align, "complete_ts_basis": complete_basis, "problems": problems,
            "standup_record": standup.record_source}


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stores", type=Path, default=None)
    ap.add_argument("--roots", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s5", type=Path, required=True, help="RS/smokes.json — the S5 block's references[] (one measured record per window class)")
    ap.add_argument("--expected-hours", type=Path, required=True, help="the §5 sized H200-h per standup")
    ap.add_argument("--period-s", type=float, default=10.0)
    ap.add_argument("--fail-on-bars", action="store_true", help="exit 1 when any standup's bars are unmet (U's fail-closed reading)")
    args = ap.parse_args(argv)
    roots = [ac.read_attempt_root(p) for p in args.roots] if args.roots else ac.discover_attempt_roots(args.stores) if args.stores else None
    if roots is None:
        refuse("one of --stores / --roots is required")
    s5 = ac.read_json(args.s5)
    expected = ac.read_json(args.expected_hours)
    if not isinstance(s5, dict) or not isinstance(expected, dict):
        refuse("--s5 and --expected-hours must be JSON objects")
    rows = [utilisation_row(r, s5=s5, expected=expected, period_s=args.period_s) for r in roots]
    ac.write_csv(args.out, rows, COLUMNS)
    ac.write_json(Path(args.out).with_suffix(".json"), rows)
    n_unmet = sum(1 for r in rows if not r["bars_met"])
    print(f"[utilisation] {len(rows)} standup(s): {len(rows) - n_unmet} bars met, {n_unmet} unmet → {args.out}")
    for r in rows:
        if not r["bars_met"]:
            print(f"  UNMET {r['leg']}/{r['view']}/{r['stage']}/{r['shard']}: running={r['running_bar']} sm={r['sm_bar']} tok={r['tok_bar']} hours={r['hours_bar']}"
                  + (f" ({'; '.join(r['problems'])})" if r["problems"] else ""))
    return ac.EXIT_FAILED if (args.fail_on_bars and n_unmet) else ac.EXIT_OK


if __name__ == "__main__":
    sys.exit(ac.run_main(main))
