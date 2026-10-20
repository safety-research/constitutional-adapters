#!/usr/bin/env python
"""THE TIMING REPORT: the native §7.2 rows (one JSON per target from
``native_timing_72.py``) beside the REUSED lens rows (``configs/timing/lens_stacked_72_psr_rows.json``, the
transcription of ``docs/lens_stacked_reductions_scoping.md`` §7.2.1's psr rows — copied, never re-run),
written as a markdown report at ``--out-md`` (a table per target) and ONE PNG per target.

Per target and B the table carries: the native ``none`` floor (ms/step), native psr ms/step, native
hook-only (psr minus the SAME run's floor), the lens STACKED ms/step (reused), the lens stacked hook-only
(reused), and the ratio native psr / lens stacked. The native reading is the row's PRIMARY ms/step — the
gate's formula ``1000 x running_mean / tok_s`` (``ms_per_step_primary == "gate"``), the formula behind the
lens numbers — with the other two readings (nominal ``B / tok_s``, the engine iteration counter) listed
beneath the table. The comparison is evaluated per target from the table: native psr ms/step < lens stacked
at every B, and native hook-only <= lens stacked hook-only at every B; a target with no native JSON (or a
partial one) reports its rows as MISSING rather than a verdict.

Every figure prints the cross-run / cross-object caveat (the lens rows: 2026-09-04 gate runs on
``si_constitution__psr_kl_all`` at RESPONSE positions; the native rows: the roster's all-positions psr
object, another day and node) — spec Assumptions.

The lens rows are cross-checked against the doc's own lines before anything is written
(:func:`check_lens_rows_against_doc`); a disagreement refuses the report (a reused number that drifted
from its source is not a reused number).

Usage::

  native_timing_72_report.py --timing-dir <dir with <target>.json...> --out-md <report.md>
      [--lens-rows configs/timing/...json] [--figs <dir>] [--wandb] [--wandb-project PROJECT]

``--out-md`` is required: the report is a run artifact, written where the caller says (a run
directory, the figures directory), never into the repository by default.

``--wandb`` is OFF by default; when on, the run reads its key from the environment the caller exported
(``WANDB_API_KEY``) — no key is ever read from or written to disk here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import native_timing_72 as nt  # noqa: E402  (the ONE validator + PROTOCOL: producer, post-check and report agree)

REPO = _HERE.parents[1]

TARGETS: Tuple[str, ...] = ("llama8b", "gemma4e4b", "qwen35_4b", "nemotron3nano4b")
BATCHES: Tuple[int, ...] = (8, 32, 64)
DEFAULT_LENS_ROWS = REPO / "configs" / "timing" / "lens_stacked_72_psr_rows.json"
DEFAULT_DOC = REPO / "docs" / "lens_stacked_reductions_scoping.md"
#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's
#: SISTEER_WANDB_PROJECT_SERVING_TIMING first) unless --wandb-project names one
WANDB_PURPOSE = "serving-timing"
NATIVE_LABEL = "native (measured)"
#: the dataviz reference palette (brand-neutral; categorical slots 1-3 in FIXED order — validated all-pairs)
SERIES_COLORS = {"native none floor": "#2a78d6", "native psr": "#eb6834", "lens stacked (reused)": "#1baf7a"}
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
#: the OBJECT the reused §7.2.1 lens rows were measured on (2026-09-04 gate runs): gemma4e4b
#: si_constitution__psr_kl_all at RESPONSE positions — the rows JSON's own caveat names it, and
#: ``load_lens_rows`` refuses a rows file that does not (the caveat below is derived, never assumed)
LENS_ROWS_OBJECT_CELL = "si_constitution__psr_kl_all"
#: the caveat for a target with NO native record (the lens rows alone); a record's caveat is
#: :func:`record_caveat` — "cross-run only" when its cell IS the lens rows' object, else cross-object too
CAVEAT = ("cross-run + cross-object: lens rows reused from §7.2 (2026-09-04 gate runs, si_constitution__psr_kl_all at "
          "response positions); native rows measured later, on their own node, on another psr object")


def record_caveat(rec: Mapping[str, Any]) -> str:
    """The comparison's caveat for ONE native record, derived from the record: the reused lens rows
    were measured on :data:`LENS_ROWS_OBJECT_CELL` at RESPONSE positions in the 2026-09-04 gate runs;
    a native record on that SAME cell (T5b) is "cross-run only" — the one like-for-like row;
    any other cell is cross-run AND cross-object, the record's cell named. The record's ``cell`` is
    bound by the validator to ``arms.psr.cell``, its harness id, the attested residual cell and every
    psr row (``native_timing_72.validate_timing_record``), so a re-labelled record is
    NON-EVALUABLE — it never earns "cross-run only"."""
    cell = str(rec.get("cell"))
    if cell == LENS_ROWS_OBJECT_CELL:
        return (f"cross-run only: the same object (`{cell}`) at the same RESPONSE positions as the reused §7.2 lens rows "
                "(2026-09-04 gate runs); the native rows are measured later, on their own node")
    return (f"cross-run + cross-object: lens rows reused from §7.2 (2026-09-04 gate runs, {LENS_ROWS_OBJECT_CELL} at "
            f"response positions); native rows measured later, on their own node, on `{cell}`")


def record_key(target: str, cell: Any) -> str:
    return f"{target}::{cell}"


def figure_name(target: str, cell: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cell))
    return f"{target}__{slug}_ms_per_step.png"


class ReportRefused(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# the reused lens rows vs the doc                                               #
# --------------------------------------------------------------------------- #
_DOC_ROW_RE = re.compile(r"^\|\s*(?P<model>\w+)\s*\|\s*(?P<case>[^|]+?)\s*\|\s*(?P<B>\d+)\s*\|")


def parse_doc_row(line: str) -> Dict[str, Any]:
    """One §7.2.1 table row (11 columns) -> its numbers: model, case, B, the two legacy ms/step (382416 /
    384162), per-row, stacked, the three hook-only figures. Refuses a line that is not such a row."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) != 11 or not _DOC_ROW_RE.match(line.strip()):
        raise ReportRefused(f"not a §7.2.1 row: {line!r}")
    model, case, B, legacy, per_row, stacked, _r1, _r2, _r3, hook, _r4 = cells
    leg = [float(x) for x in legacy.split("/")]
    hk = [float(x) for x in hook.split("/")]
    if len(leg) != 2 or len(hk) != 3:
        raise ReportRefused(f"malformed legacy/hook-only columns in {line!r}")
    return {"target": model, "case": case, "B": int(B), "legacy": {"382416": leg[0], "384162": leg[1]},
            "per_row": float(per_row), "stacked": float(stacked),
            "hook_only": {"legacy": hk[0], "per_row": hk[1], "stacked": hk[2]}}


def _doc_line_of(source: str) -> int:
    m = re.search(r"line (\d+)$", str(source))
    if not m:
        raise ReportRefused(f"a lens row's source does not name its doc line: {source!r}")
    return int(m.group(1))


def check_lens_rows_against_doc(lens: Mapping[str, Any], doc_text: str, *, tol: float = 1e-9) -> List[str]:
    """Every transcribed number held to the doc line its ``source`` cites (target, case, B, both legacy
    figures, per-row, stacked, the three hook-only figures); a list of disagreements (empty == agree)."""
    lines = doc_text.splitlines()
    problems: List[str] = []
    rows = lens.get("rows") or []
    if not rows:
        return ["the lens rows file carries no rows"]
    for r in rows:
        ln = _doc_line_of(r.get("source", ""))
        if ln < 1 or ln > len(lines):
            problems.append(f"{r.get('target')} B={r.get('B')}: cited line {ln} is outside the doc")
            continue
        try:
            d = parse_doc_row(lines[ln - 1])
        except ReportRefused as e:
            problems.append(f"line {ln}: {e}")
            continue
        tag = f"{r.get('target')} B={r.get('B')} (line {ln})"
        if d["target"] != r.get("target") or d["B"] != int(r.get("B", -1)) or d["case"] != r.get("case"):
            problems.append(f"{tag}: identity differs from the doc ({d['target']} / {d['case']} / B={d['B']})")
        pairs = [("legacy 382416", (r.get("legacy_ms_per_step") or {}).get("382416"), d["legacy"]["382416"]),
                 ("legacy 384162", (r.get("legacy_ms_per_step") or {}).get("384162"), d["legacy"]["384162"]),
                 ("per_row", r.get("per_row_ms_per_step"), d["per_row"]),
                 ("stacked", r.get("stacked_ms_per_step"), d["stacked"]),
                 ("hook_only legacy", (r.get("hook_only_ms") or {}).get("legacy"), d["hook_only"]["legacy"]),
                 ("hook_only per_row", (r.get("hook_only_ms") or {}).get("per_row"), d["hook_only"]["per_row"]),
                 ("hook_only stacked", (r.get("hook_only_ms") or {}).get("stacked"), d["hook_only"]["stacked"])]
        for name, got, want in pairs:
            if got is None or abs(float(got) - float(want)) > tol:
                problems.append(f"{tag}: {name} {got!r} != doc {want!r}")
    return problems


def _check_lens_rows_name_their_object(lens: Mapping[str, Any]) -> None:
    text = " ".join([str(lens.get("what", "")), *map(str, lens.get("caveats") or [])])
    if LENS_ROWS_OBJECT_CELL not in text:
        raise ReportRefused(f"the lens rows file does not name its object {LENS_ROWS_OBJECT_CELL!r} in `what` / "
                            "`caveats` — the per-record caveat is derived from that fact, never assumed")


def load_lens_rows(path: Path) -> Dict[str, Any]:
    lens = _load_lens_rows_raw(path)
    _check_lens_rows_name_their_object(lens)
    return lens


def _load_lens_rows_raw(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lens_rows_by_target(lens: Mapping[str, Any]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in lens.get("rows") or []:
        out.setdefault(r["target"], {})[int(r["B"])] = r
    return out


# --------------------------------------------------------------------------- #
# the native records -> the table                                               #
# --------------------------------------------------------------------------- #
def load_records(paths: Sequence[Path], *, targets: Optional[Sequence[str]] = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Every record keyed by ``(target, cell)``: one target can hold two records on different cells, so
    a per-target key could not hold both; two records for one (target, cell) refuse; ``targets`` filters
    by the RECORD's target (never by file name)."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in paths:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        t, cell = rec.get("target"), rec.get("cell")
        if not t:
            raise ReportRefused(f"{p} carries no target")
        if not cell:
            raise ReportRefused(f"{p} ({t}) carries no cell — the record must name the psr object it timed")
        if targets is not None and t not in targets:
            continue
        key = (str(t), str(cell))
        if key in out:
            raise ReportRefused(f"two records for {t} / {cell}: {out[key].get('_path')} and {p}")
        rec["_path"] = str(p)
        out[key] = rec
    return out


def _records_by_target(records: Mapping[Any, Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """``{target: [record, ...]}`` in insertion order; accepts the ``(target, cell)`` keys of
    :func:`load_records` and the legacy ``{target: record}`` form (one record per target)."""
    by: Dict[str, List[Mapping[str, Any]]] = {}
    for key, rec in records.items():
        t = key[0] if isinstance(key, tuple) else str(key)
        by.setdefault(t, []).append(rec)
    return by


def primary_ms(row: Mapping[str, Any]) -> Optional[float]:
    key = row.get("ms_per_step_primary") or "gate"
    v = (row.get("ms_per_step") or {}).get(key)
    return float(v) if v is not None else None


def _row_of(rec: Mapping[str, Any], arm: str, B: int) -> Optional[Mapping[str, Any]]:
    for r in rec.get("rows") or []:
        if r.get("arm") == arm and int(r.get("B", -1)) == int(B):
            return r
    return None


def build_table(rec: Optional[Mapping[str, Any]], lens_for_target: Mapping[int, Mapping[str, Any]],
                batches: Sequence[int] = BATCHES) -> List[Dict[str, Any]]:
    """Per B: the native floor / psr / hook-only (primary reading, plus the other readings kept beside),
    the lens stacked ms/step + hook-only (reused), and the ratio. Missing native cells are None."""
    table: List[Dict[str, Any]] = []
    for B in batches:
        none_r = _row_of(rec, "none", B) if rec else None
        psr_r = _row_of(rec, "psr", B) if rec else None
        lens_r = lens_for_target.get(int(B))
        n_none, n_psr = (primary_ms(none_r) if none_r else None), (primary_ms(psr_r) if psr_r else None)
        hook = (psr_r.get("hook_only_ms") or {}) if psr_r else {}
        key = (psr_r.get("ms_per_step_primary") if psr_r else None) or "gate"
        n_hook = hook.get(key)
        l_st = float(lens_r["stacked_ms_per_step"]) if lens_r else None
        l_hook = float((lens_r.get("hook_only_ms") or {}).get("stacked")) if lens_r and (lens_r.get("hook_only_ms") or {}).get("stacked") is not None else None
        table.append({
            "B": int(B),
            "native_none_ms": n_none, "native_psr_ms": n_psr, "native_hook_only_ms": n_hook,
            "native_readings": {"none": (none_r or {}).get("ms_per_step"), "psr": (psr_r or {}).get("ms_per_step"),
                                "psr_hook_only": (psr_r or {}).get("hook_only_ms")},
            "native_running_mean": {"none": (none_r or {}).get("running_mean"), "psr": (psr_r or {}).get("running_mean")},
            "native_errors": {"none": (none_r or {}).get("errors"), "psr": (psr_r or {}).get("errors")},
            "lens_stacked_ms": l_st, "lens_hook_only_ms": l_hook,
            "lens_label": (lens_r or {}).get("label"), "lens_source": (lens_r or {}).get("source"),
            "native_label": (psr_r or {}).get("label") if psr_r else None,
            "ratio_native_psr_over_lens_stacked": ((n_psr / l_st) if (n_psr is not None and l_st) else None),
        })
    return table


def criterion3(table: Sequence[Mapping[str, Any]], *, validation: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The comparison over one target's table: native psr ms/step < lens stacked at EVERY B, and native
    hook-only <= lens stacked hook-only at EVERY B. ``None`` (not a verdict) when any cell is missing —
    and NON-EVALUABLE (every verdict None, ``evaluable`` False, the problems named) when the record's
    evidence does not validate (``validation`` is :func:`native_timing_72.validate_timing_record`'s
    verdict, recomputed by the report from the record's own facts; a table over invalid evidence is never
    graded)."""
    if validation is not None and not validation.get("valid", False):
        return {"complete": False, "evaluable": False, "ms_per_step_pass": None, "hook_only_pass": None,
                "per_B": [{"B": r["B"], "ms_per_step_pass": None, "hook_only_pass": None} for r in table],
                "invalid_evidence": list(validation.get("problems") or [])}
    per_B = []
    complete = True
    for r in table:
        a, b, c, d = r["native_psr_ms"], r["lens_stacked_ms"], r["native_hook_only_ms"], r["lens_hook_only_ms"]
        ok_ms = (a < b) if (a is not None and b is not None) else None
        ok_hook = (c <= d) if (c is not None and d is not None) else None
        complete = complete and ok_ms is not None and ok_hook is not None
        per_B.append({"B": r["B"], "ms_per_step_pass": ok_ms, "hook_only_pass": ok_hook})
    return {"complete": complete, "evaluable": complete,
            "ms_per_step_pass": (all(x["ms_per_step_pass"] for x in per_B) if complete else None),
            "hook_only_pass": (all(x["hook_only_pass"] for x in per_B) if complete else None),
            "per_B": per_B, "invalid_evidence": []}


def _fmt(v: Optional[float], nd: int = 1) -> str:
    return "—" if v is None else f"{float(v):.{nd}f}"


def render_markdown(records: Mapping[str, Mapping[str, Any]], lens: Mapping[str, Any], *,
                    targets: Sequence[str] = TARGETS, figs_rel: Optional[Mapping[str, str]] = None,
                    generated_at: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """The doc: a table per (target, cell) record — B, native none floor, native psr, native hook-only,
    lens stacked (reused), lens hook-only (reused), ratio — the comparison's verdict, the other readings,
    the provenance of both sides, the record's own caveat (:func:`record_caveat`). Returns
    ``(markdown, summary)`` with ``summary["<target>::<cell>"]`` per record, ``summary[target]`` the
    target's FIRST record (or its missing marker), ``figs_rel`` keyed the same two ways. An INVALID
    record's section is its heading + NON-EVALUABLE + the validator's reasons, nothing else (its
    summary ``caveat`` is None; ``main`` renders it no figure)."""
    lens_by = lens_rows_by_target(lens)
    by_target = _records_by_target(records)
    gen = generated_at or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    lens_label = lens.get("label", "lens (reused, §7.2)")
    out: List[str] = []
    summary: Dict[str, Any] = {}
    out.append("# Native PSR serving — §7.2 timing (native residual slot vs the reused lens rows)\n")
    out.append(f"Generated {gen} by `scripts/sdfbase/native_timing_72_report.py`. The lens columns are the psr rows of "
               f"`docs/lens_stacked_reductions_scoping.md` §7.2.1 transcribed into `configs/timing/lens_stacked_72_psr_rows.json` "
               f"(label `{lens_label}`) — reused, never re-run (spec criterion 3). The native columns are measured by "
               f"`scripts/sdfbase/native_timing_72.py` under the mirrored protocol ({nt.PROTOCOL['L']} x {nt.PROTOCOL['S']}, "
               f"`max_tokens` {nt.PROTOCOL['max_tokens']} exact, {nt.PROTOCOL['window_s']:g} s window after a "
               f"{nt.PROTOCOL['warmup_s']:g} s warm-up, the gate's ~{nt.PROTOCOL['prompt_tokens']}-token passage, prefix caching "
               f"OFF, `VLLM_BATCH_INVARIANT` unset — every record below is held to exactly these facts by the validator before "
               f"any verdict), label "
               f"`{NATIVE_LABEL}`. ms/step is the gate's formula (1000 x running_mean / tok/s, the formula behind the lens "
               f"numbers); the nominal (B / tok/s) and engine-counter readings are listed under each table. Hook-only = ms/step "
               f"minus the SAME run's passthrough floor at the same B, on both sides.\n")
    out.append(f"**Caveat:** stated under each table from its record ({LENS_ROWS_OBJECT_CELL} at response positions is "
               f"the reused lens rows' object — a native record on that cell is cross-run only; any other cell is "
               f"cross-run + cross-object). A target with no native record: {CAVEAT}.\n")
    for cav in lens.get("caveats") or []:
        out.append(f"- {cav}")
    out.append("")
    per_target: List[Tuple[str, Optional[Mapping[str, Any]]]] = []
    for t in targets:
        recs = by_target.get(t) or []
        per_target.extend((t, r) for r in recs) if recs else per_target.append((t, None))
    for t, rec in per_target:
        cell = rec.get("cell") if rec is not None else None
        skey = record_key(t, cell) if rec is not None else t
        table = build_table(rec, lens_by.get(t, {}))
        # the validator's verdict is RECOMPUTED here from the record's facts (never read off the record)
        validation = nt.validate_timing_record(rec) if rec is not None else {"valid": False, "problems": ["no native record"]}
        c3 = criterion3(table, validation=validation)
        status = (rec or {}).get("status") if rec else "missing"
        valid = rec is not None and bool(validation.get("valid"))
        # an INVALID record earns NO caveat, table, verdict or figure: the caveat
        # is a claim about the timed object — the very name the validator just refused to bind — so a
        # re-labelled record must never read "cross-run only" beside its invalid-evidence warning
        entry = {"table": table, "criterion3": c3, "status": status, "validation": validation, "cell": cell,
                 "caveat": (record_caveat(rec) if valid else None) if rec is not None else CAVEAT}
        summary[skey] = entry
        summary.setdefault(t, entry)                         # the target's first record (the roster's psr cell)
        out.append(f"## {t}" + (f" — cell `{cell}`\n" if rec is not None else "\n"))
        if rec is None:
            out.append(f"_No native record for `{t}` yet (rows MISSING; the lens rows below are the reused reference)._\n")
        elif not valid:
            out.append("Criterion 3 — **NON-EVALUABLE**: native record "
                       f"`{rec.get('_path', '')}` (status `{status}`, node `{rec.get('node')}`, sha `{rec.get('sha')}`, "
                       f"date `{rec.get('date')}`) does not validate (`native_timing_72.validate_timing_record`): "
                       + "; ".join(validation.get("problems") or []) + ". No caveat, table, PASS/FAIL or figure is "
                       "stated over invalid evidence.\n")
            continue
        else:
            out.append(f"**Caveat:** {entry['caveat']}.\n")
            out.append(f"Native record `{rec.get('_path', '')}` — status `{status}`, node `{rec.get('node')}`, sha `{rec.get('sha')}`, "
                       f"date `{rec.get('date')}`, cell `{rec.get('cell')}`, view `{rec.get('view')}`; prompt "
                       f"{((rec.get('prompt') or {}).get('tokens'))} tokens by the served tokenizer "
                       f"({((rec.get('prompt') or {}).get('served_prompt_tokens'))} served); engine facts: "
                       f"{json.dumps(_engine_summary(rec), sort_keys=True)}; vector arm: "
                       f"{json.dumps((rec.get('arms') or {}).get('vector'), sort_keys=True)}.\n")
        out.append("| B | native `none` floor ms/step | native psr ms/step | native hook-only ms | lens stacked ms/step (reused) | lens stacked hook-only ms (reused) | native psr / lens stacked |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in table:
            out.append(f"| {r['B']} | {_fmt(r['native_none_ms'])} | {_fmt(r['native_psr_ms'])} | {_fmt(r['native_hook_only_ms'])} | "
                       f"{_fmt(r['lens_stacked_ms'])} | {_fmt(r['lens_hook_only_ms'])} | {_fmt(r['ratio_native_psr_over_lens_stacked'], 2)} |")
        out.append("")
        if c3["complete"]:
            per_b = ", ".join("B%d: ms %s / hook %s" % (x["B"], x["ms_per_step_pass"], x["hook_only_pass"]) for x in c3["per_B"])
            verdict_ms = "PASS" if c3["ms_per_step_pass"] else "FAIL"
            verdict_hook = "PASS" if c3["hook_only_pass"] else "FAIL"
            out.append(f"Criterion 3 — native psr < lens stacked at every B: **{verdict_ms}**; "
                       f"native hook-only <= lens stacked hook-only at every B: **{verdict_hook}** (per B: {per_b}). "
                       "Evidence validated: complete record, the §7.2 protocol declared and re-derived from the launch "
                       "facts, every window error-free with an engine tok/s and concurrency >= 0.5 B.\n")
        else:
            out.append("Criterion 3 — not evaluable: a native or lens cell is missing at some B.\n")
        if rec is not None:
            out.append("Other native readings (ms/step nominal `B / tok/s` and engine-counter; hook-only likewise):\n")
            out.append("| B | none nominal | none counter | psr nominal | psr counter | hook-only nominal | hook-only counter | running mean none / psr | errors none / psr |")
            out.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for r in table:
                nr, pr, hk = r["native_readings"]["none"] or {}, r["native_readings"]["psr"] or {}, r["native_readings"]["psr_hook_only"] or {}
                rm, er = r["native_running_mean"], r["native_errors"]
                out.append(f"| {r['B']} | {_fmt(nr.get('nominal'))} | {_fmt(nr.get('counter'))} | {_fmt(pr.get('nominal'))} | {_fmt(pr.get('counter'))} | "
                           f"{_fmt(hk.get('nominal'))} | {_fmt(hk.get('counter'))} | {_fmt(rm.get('none'))} / {_fmt(rm.get('psr'))} | "
                           f"{er.get('none')} / {er.get('psr')} |")
            out.append("")
        fig = (figs_rel or {}).get(skey) or ((figs_rel or {}).get(t) if rec is None else None)
        if fig:
            out.append(f"![{t} {cell or ''} ms/step]({fig})\n")
        srcs = sorted({r["lens_source"] for r in table if r.get("lens_source")})
        if srcs:
            out.append(f"Lens rows: {'; '.join(f'`{s}`' for s in srcs)}.\n")
    return "\n".join(out) + "\n", summary


def _engine_summary(rec: Mapping[str, Any]) -> Dict[str, Any]:
    la = rec.get("launch") or {}
    eng, pc, bi = la.get("engine") or {}, la.get("prefix_caching") or {}, la.get("batch_invariance") or {}
    return {"enforce_eager": eng.get("enforce_eager"), "cudagraph_mode": eng.get("cudagraph_mode"),
            "residual_slot_armed": (eng.get("residual_slot") or {}).get("armed"),
            "output_diagnostics": eng.get("output_diagnostics"),
            "prefix_caching_composed": pc.get("composed"), "attested_enable_prefix_caching": pc.get("attested_enable_prefix_caching"),
            "batch_invariance_effective": bi.get("effective"),
            "geometry": {k: (eng.get("geometry") or {}).get(k) for k in ("L", "S", "attested_S")}}


# --------------------------------------------------------------------------- #
# the figure (one PNG per target; matplotlib, brand-neutral)                    #
# --------------------------------------------------------------------------- #
def render_figure(target: str, table: Sequence[Mapping[str, Any]], path: Path, *,
                  caveat: str = CAVEAT, subtitle: Optional[str] = None) -> Path:
    """Two panels: whole-step ms/step (native none floor, native psr, lens stacked reused) and hook-only
    ms (native psr vs lens stacked reused), grouped by B. Titles <= 2 words; the caveat is a footnote,
    never a title; series colors in fixed order from the reference palette; a legend beside direct
    value labels; recessive grid; missing cells leave a gap (never a zero bar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Bs = [int(r["B"]) for r in table]
    x = np.arange(len(Bs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    fig.suptitle(f"{target} ms/step", color=INK, fontsize=13, x=0.02, ha="left")

    def bars(ax, series: Sequence[Tuple[str, Sequence[Optional[float]]]], width: float) -> None:
        n = len(series)
        for i, (name, vals) in enumerate(series):
            offs = (i - (n - 1) / 2) * (width + 0.02)
            for j, v in enumerate(vals):
                if v is None:
                    continue
                ax.bar(x[j] + offs, v, width=width, color=SERIES_COLORS[name], label=name if j == 0 else None,
                       linewidth=0, zorder=3)
                ax.annotate(f"{v:.1f}", (x[j] + offs, v), ha="center", va="bottom", fontsize=8, color=INK_2,
                            xytext=(0, 2), textcoords="offset points")
        ax.set_facecolor(SURFACE)
        top = max((v for _, vals in series for v in vals if v is not None), default=None)
        if top is not None and top > 0:
            ax.set_ylim(0, top * 1.32)      # headroom so the in-axes legend never sits on a bar
        ax.set_xticks(x)
        ax.set_xticklabels([f"B={b}" for b in Bs], color=INK)
        ax.tick_params(colors=INK_2, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#d9d8d3")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    bars(ax1, [("native none floor", [r["native_none_ms"] for r in table]),
               ("native psr", [r["native_psr_ms"] for r in table]),
               ("lens stacked (reused)", [r["lens_stacked_ms"] for r in table])], width=0.24)
    ax1.set_title("Whole step", color=INK, fontsize=11, loc="left")
    ax1.set_ylabel("ms per decode step", color=INK_2, fontsize=9)
    bars(ax2, [("native psr", [r["native_hook_only_ms"] for r in table]),
               ("lens stacked (reused)", [r["lens_hook_only_ms"] for r in table])], width=0.3)
    ax2.set_title("Hook only", color=INK, fontsize=11, loc="left")
    ax2.set_ylabel("ms per step over the run's own floor", color=INK_2, fontsize=9)
    foot = caveat if subtitle is None else f"{subtitle}\n{caveat}"
    fig.text(0.02, 0.005, foot, fontsize=7, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def wandb_log(summary: Mapping[str, Any], figs: Mapping[str, Path], *, project: Optional[str] = None,
              run_name: str = "native-timing-72") -> Optional[str]:
    """OPT-IN: log the tables (one W&B table per (target, cell) record, plus the lens-only table of a
    target with no record) and the figures to ``project``; the key comes from the caller's environment
    (never a file here). An INVALID record publishes NOTHING numeric — no table, no image — only
    ``<key>/non_evaluable = True`` and its ``invalid_evidence`` reasons (the doc withholds the comparison
    from invalid evidence, so must W&B). Returns the run url."""
    import wandb  # noqa: WPS433  (opt-in; absent on CPU boxes without it)
    if project is None:
        from sisteer.wandb_entity import wandb_project
        project = wandb_project(WANDB_PURPOSE)
    run = wandb.init(project=project, name=run_name, job_type="report")
    try:
        for t, s in summary.items():
            if s.get("cell") is not None and "::" not in t:
                continue                                       # a record's target alias: logged under its own key
            v = s.get("validation") or {}
            c3 = s["criterion3"]
            if s.get("cell") is not None and not v.get("valid"):
                run.summary[f"{t}/non_evaluable"] = True
                run.summary[f"{t}/invalid_evidence"] = list(v.get("problems") or [])
                run.summary[f"{t}/criterion3_ms_per_step_pass"] = None
                run.summary[f"{t}/criterion3_hook_only_pass"] = None
                continue
            cols = ["B", "native_none_ms", "native_psr_ms", "native_hook_only_ms", "lens_stacked_ms", "lens_hook_only_ms",
                    "ratio_native_psr_over_lens_stacked"]
            tbl = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in s["table"]])
            run.log({f"{t}/timing_table": tbl})
            run.summary[f"{t}/criterion3_ms_per_step_pass"] = c3["ms_per_step_pass"]
            run.summary[f"{t}/criterion3_hook_only_pass"] = c3["hook_only_pass"]
            if figs.get(t):
                run.log({f"{t}/ms_per_step": wandb.Image(str(figs[t]))})
        return run.url
    finally:
        run.finish()


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timing-json", action="append", default=[], type=Path, help="a per-target JSON (repeatable)")
    ap.add_argument("--timing-dir", default=None, type=Path, help="a directory of <target>.json records")
    ap.add_argument("--lens-rows", default=DEFAULT_LENS_ROWS, type=Path)
    ap.add_argument("--doc", default=DEFAULT_DOC, type=Path, help="the scoping doc the lens rows are held to")
    ap.add_argument("--out-md", required=True, type=Path,
                    help="where the markdown report is written (a run-local path; there is no repository default)")
    ap.add_argument("--figs", default=None, type=Path, help="directory for one PNG per target")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--wandb", action="store_true", help="also log tables + figures to W&B (OFF by default)")
    ap.add_argument("--wandb-project", default=None,
                    help="default: the site's SISTEER_WANDB_PROJECT_SERVING_TIMING, else WANDB_PROJECT, else SISTEER_WANDB_PROJECT")
    ap.add_argument("--skip-doc-check", action="store_true", help="(tests only) skip the lens-rows-vs-doc cross-check")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    lens = load_lens_rows(args.lens_rows)
    if not args.skip_doc_check:
        problems = check_lens_rows_against_doc(lens, Path(args.doc).read_text(encoding="utf-8"))
        if problems:
            raise ReportRefused("the reused lens rows disagree with the doc:\n  - " + "\n  - ".join(problems))
    paths = list(args.timing_json)
    if args.timing_dir:
        paths += sorted(Path(args.timing_dir).glob("*.json"))     # filtered by the RECORD's target below, never the file name
    records = load_records(paths, targets=targets)
    lens_by = lens_rows_by_target(lens)
    by_target = _records_by_target(records)
    figs: Dict[str, Path] = {}
    figs_rel: Dict[str, str] = {}
    if args.figs:
        for t in targets:
            recs = by_target.get(t) or []
            if not recs and not lens_by.get(t):
                continue
            for rec in (recs or [None]):
                if rec is not None and not nt.validate_timing_record(rec).get("valid"):
                    continue                                  # no figure over invalid evidence
                table = build_table(rec, lens_by.get(t, {}))
                cell = rec.get("cell") if rec is not None else None
                skey = record_key(t, cell) if rec is not None else t
                p = render_figure(t, table, Path(args.figs) / (figure_name(t, cell) if rec is not None else f"{t}_ms_per_step.png"),
                                  caveat=record_caveat(rec) if rec is not None else CAVEAT,
                                  subtitle=f"cell {cell}" if rec is not None else None)
                figs[skey] = p
                try:
                    figs_rel[skey] = os.path.relpath(p, Path(args.out_md).parent)
                except ValueError:
                    figs_rel[skey] = str(p)
    md, summary = render_markdown(records, lens, targets=targets, figs_rel=figs_rel)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"[native_timing_72_report] wrote {args.out_md} ({len(records)} native record(s): "
          f"{sorted(record_key(*k) for k in records)}; {len(figs)} figure(s))")
    for k in sorted(k for k in summary if "::" in k) + [t for t in targets if not by_target.get(t)]:
        s = summary[k]
        c3 = s["criterion3"]
        v = s.get("validation") or {}
        print(f"  {k}: status={s['status']} valid={v.get('valid')} criterion3 ms/step={c3['ms_per_step_pass']} "
              f"hook_only={c3['hook_only_pass']}" + ("" if v.get("valid") else f" NON-EVALUABLE: {v.get('problems')}")
              + (f"; {s['caveat'][:60]}…" if s.get("caveat") else ""))
    if args.wandb:
        url = wandb_log(summary, figs, project=args.wandb_project)
        print(f"  wandb: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
