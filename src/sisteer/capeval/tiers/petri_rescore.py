"""Judge-only RE-SCORE of TERMINAL judge refusals over their EXISTING transcripts -- the explicit operator action the runner's
contract demands (petri_runner.merge_judge_refused: a re-score must be an explicit operator action with its selection effect
recorded).

What it does, and does not. It never audits: no auditor turn, no target call. The transcript the auditor and the target produced is
read from the refused sample's ``.eval`` log as it stands and ONLY the judge runs again, under the manifest's CURRENT judge chain (the
default judge with its per-request refusal fallback -- Opus 5 -> Opus 4.8 -> Grok 4.6 pinned to its dated build), through
inspect's own re-scoring (``inspect_ai.score`` re-scores whole logs, so the sample is re-scored inside a ONE-SAMPLE copy of its log
written beside the original under ``rescore-<revision>/``, never over it), with the SAME scorer the audit used (the behaviour's
``bloom_audit`` task scorer: its judge dimensions and instructions) and the same judge wiring the live driver builds
(``petri_runner._role_model_with_refusal_fallback``).

Selection is :func:`petri_runner.rescorable_judge_refusals`, the ONE selector that shares the sidecar's ledger with the overflow
pass: a standing terminal judge refusal is re-scorable iff the manifest's terminal judge leg's family did not refuse the sample --
read from the sample's judge PROVENANCE in its ``.eval`` log (never from the filed label: the u0 records of 2026-10-01 were filed
under opus-5 by the pre-B3 converter while opus-5 AND opus-4-8 refused) -- and neither ``rerolled`` nor ``rescored`` already names
it. A refusal the terminal leg itself issued stands (the chain has nothing new to try); auditor refusals and hard errors are not
judge-refused records and are never candidates. A record whose log cannot be read is refused by name, never guessed about.

Effect, recorded by name in the cell's judge-refused sidecar under ``rescored`` (one entry per item: the superseded terminal record
verbatim as ``prior_terminal``, the judge families that had refused it, the chain used, the revision = the selection's hash, who
asked and why, the outcome -- ``scored`` with the new judgment's build identity, ``refiled_terminal`` with the new refusal's
judge and build class, or ``partial`` (Grok-fix-4): a judgment that scored only some of the item's jointly judged dimensions files
NOTHING, the refusal stands, the converter's ledger for the item is preserved on the entry, and the retry gets its own log), in the rows (the converter's usual rows; ``judge_build`` when the terminal leg answered), and in
results.json (rebuilt through the runner's own summary builders; ``rescored: {n, revisions}``). The overflow pass reads
``rescored`` as settled and never relaunches a re-scored item; this operation reads ``rerolled`` as settled.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sisteer.capeval import manifest as cm
from sisteer.capeval import store as cs
from sisteer.capeval.petri import auditor_fallback as af
from sisteer.capeval.petri.defense_cells import PetriCell
from sisteer.capeval.tiers import petri_runner as pr

RESCORED_BY = "petri_rescore (judge-only re-score over the existing transcript)"
RESCORE_SCHEMA = "sisteer.petri_rescore/rescored/v1"
RESCORE_DIR_PREFIX = "rescore-"


class RescoreRefused(RuntimeError):
    """A selection or an outcome this action refuses by name (nothing is written when it raises before the write phase)."""


@dataclass(frozen=True)
class RescoreSelection:
    """What was selected and why -- recorded on every ``rescored`` entry; ``revision`` names the one-sample log dir."""
    cell_id: str
    petri_run_key: str
    items: Tuple[Tuple[str, str, int], ...]          # (behavior, scenario_id, epoch)
    why: str
    judge_model: str
    judge_refusal_fallback_models: Tuple[str, ...]
    requested_by: str
    selected_at: float

    def revision(self) -> str:
        return cm.sha256_hex16(cm.canonical_json({"items": [list(i) for i in self.items], "judge_model": self.judge_model,
                                                   "judge_refusal_fallback_models": list(self.judge_refusal_fallback_models)}).encode("utf-8"))


def chain_of(spec: cm.PetriSpec) -> Tuple[str, Tuple[str, ...]]:
    return str(spec.judge_model), tuple(str(m) for m in spec.judge_refusal_fallback_models)


def _default_read_log(path: str) -> Mapping[str, Any]:
    from inspect_ai.log import read_eval_log
    return pr._dumped_log(read_eval_log(str(path)))


def find_sample(log: Mapping[str, Any], record: pr.JudgeRefusal) -> Mapping[str, Any]:
    """The record's sample in a log dict: by ``sample_uuid`` when the record carries one, else by (sample_id or scenario_id, epoch)."""
    samples = [s for s in (log.get("samples") or []) if isinstance(s, Mapping)]
    if record.sample_uuid:
        hits = [s for s in samples if str(s.get("uuid")) == str(record.sample_uuid)]
        if len(hits) == 1:
            return hits[0]
    sid = record.sample_id or record.scenario_id
    hits = [s for s in samples if str(s.get("id")) == str(sid) and int(s.get("epoch", 1) or 1) == int(record.epoch)]
    if len(hits) != 1:
        raise RescoreRefused(f"{record.behavior}/{record.scenario_id}#e{record.epoch}: {len(hits)} sample(s) match id {sid!r} epoch "
                             f"{record.epoch} in {log.get('location')!r}; need exactly one")
    return hits[0]


def judge_families_of_record(record: pr.JudgeRefusal, read_log: Callable[[str], Mapping[str, Any]]) -> List[str]:
    """The judge FAMILIES that served the record's sample (its .eval log's judge events, raw served ids by family) -- the provenance
    the selector keys on. Refuses by name a record without an eval_log, a log that cannot be read, or a sample it does not hold."""
    if not record.eval_log:
        raise RescoreRefused(f"{record.behavior}/{record.scenario_id}#e{record.epoch}: the terminal record names no eval_log -- no "
                             "provenance to select on, nothing to re-score")
    try:
        log = read_log(str(record.eval_log))
    except Exception as exc:    # noqa: BLE001 -- named, not swallowed: the record is refused, never guessed about
        raise RescoreRefused(f"{record.behavior}/{record.scenario_id}#e{record.epoch}: cannot read {record.eval_log!r}: "
                             f"{type(exc).__name__}: {exc}") from exc
    sample = find_sample(log, record)
    served = pr.sample_role_serving_models(sample).get("judge") or {}
    return [pr._judge_family(m) for m in served]


def select_terminal_refusals(store: cs.CellStore, spec: cm.PetriSpec, items: Optional[Sequence[Tuple[str, str, int]]] = None, *,
                             read_log: Optional[Callable[[str], Mapping[str, Any]]] = None) -> List[pr.JudgeRefusal]:
    """The cell's terminal judge refusals to re-score: every re-scorable one, or exactly ``items`` (each must be re-scorable).
    Refused by name: an item that is not a terminal record of this cell; an item the manifest's terminal judge already refused; an
    item the sidecar's ledger already names (``rerolled`` / ``rescored``); a selection with nothing in it."""
    read_log = read_log or _default_read_log
    prior = pr.read_judge_refused(store)
    by_key = {(r.behavior, r.scenario_id, int(r.epoch)): r for r in prior}
    if items is not None:
        missing = [tuple(k) for k in items if tuple(k) not in by_key]
        if missing:
            raise RescoreRefused(f"not terminal judge refusals of cell {store.key}: {missing} (the sidecar holds {sorted(by_key)})")
        candidates = [by_key[tuple(k)] for k in items]
    else:
        candidates = list(by_key.values())
    already = pr.settled_ledger(store)            # a partial attempt is evidence, not a settlement: attempt 2 may complete the item
    families: Dict[Tuple[str, str, int], List[str]] = {}

    def refused_by(r: pr.JudgeRefusal) -> List[str]:
        key = (r.behavior, r.scenario_id, int(r.epoch))
        if key not in families:
            families[key] = judge_families_of_record(r, read_log)
        return families[key]
    standing, chosen = pr.rescorable_judge_refusals(candidates, spec, refused_by, already)
    if items is not None and standing:
        judge, fallbacks = chain_of(spec)
        terminal = fallbacks[-1] if fallbacks else judge
        done = {(str(x.get("behavior")), str(x.get("scenario_id")), int(x.get("epoch", 0) or 0)) for x in already}
        why = []
        for r in standing:
            key = (r.behavior, r.scenario_id, int(r.epoch))
            if key in done:
                why.append(f"{r.behavior}/{r.scenario_id}#e{r.epoch}: already re-rolled or re-scored (the sidecar's settled ledger names it)")
            else:
                why.append(f"{r.behavior}/{r.scenario_id}#e{r.epoch}: refused by {sorted(set(families.get(key, [])))}, which includes the "
                           f"manifest's terminal judge {terminal!r} -- the chain has nothing new to try")
        raise RescoreRefused(f"cell {store.key}: not re-scorable -- " + "; ".join(why))
    if not chosen:
        raise RescoreRefused(f"cell {store.key}: no terminal judge refusal is re-scorable under judge chain {chain_of(spec)} "
                             f"({len(prior)} standing record(s), {len(already)} already settled by the ledger)")
    return sorted(chosen, key=lambda r: (r.behavior, r.scenario_id, int(r.epoch)))


def one_sample_log(log: Any, record: pr.JudgeRefusal) -> Any:
    """A deep copy of the inspect ``EvalLog`` holding ONLY the record's sample -- the unit inspect re-scores."""
    def _is(s: Any) -> bool:
        if record.sample_uuid and getattr(s, "uuid", None) is not None:
            return str(s.uuid) == str(record.sample_uuid)
        return str(s.id) == str(record.sample_id or record.scenario_id) and int(getattr(s, "epoch", 1) or 1) == int(record.epoch)
    picked = [s for s in (log.samples or []) if _is(s)]
    if len(picked) != 1:
        raise RescoreRefused(f"{getattr(log, 'location', '?')}: {len(picked)} sample(s) match {record.behavior}/{record.scenario_id}"
                             f"#e{record.epoch}, need exactly one")
    out = copy.deepcopy(log)
    out.samples = copy.deepcopy(picked)
    return out


def rescore_log(record: pr.JudgeRefusal, *, scorers: Sequence[Any], judge_model: Any, out_dir: Path,
                score_fn: Optional[Callable[..., Any]] = None, read_fn: Optional[Callable[[str], Any]] = None,
                write_fn: Optional[Callable[[Any, str], None]] = None) -> Path:
    """Re-score ONE sample of the record's eval log with ``scorers`` under ``judge_model`` (inspect's model role ``judge``), writing
    the one-sample re-scored log to ``out_dir/<original basename>``; returns its path. The original log is never touched; a
    re-score log is written once (``out_dir`` is named by the revision). ``score_fn`` / ``read_fn`` / ``write_fn`` are the inspect
    seams (inspect_ai.score / read_eval_log / write_eval_log), injectable for the CPU tests."""
    if read_fn is None or score_fn is None or write_fn is None:
        from inspect_ai import score as _score
        from inspect_ai.log import read_eval_log as _read, write_eval_log as _write
        read_fn = read_fn or _read; score_fn = score_fn or _score; write_fn = write_fn or _write
    log = read_fn(str(record.eval_log))
    one = one_sample_log(log, record)
    out_dir.mkdir(parents=True, exist_ok=True)
    # one file per ITEM: several refused samples of a cell share one behaviour log (the u0 prompted__constitution cell's three cyber
    # refusals live in one .eval), so the item's identity joins the original's name
    src = Path(str(record.eval_log))
    stem = f"{src.stem}--{record.behavior}--{record.scenario_id}--e{int(record.epoch)}"
    # attempt-numbered (Grok-fix-4): a re-score that filed nothing (a partial judgment, a converter refusal)
    # must not block the retry of the same revision; every attempt's log is kept as evidence under its own name
    attempt = 1 + sum(1 for p in out_dir.glob(f"{stem}--a*{src.suffix}"))
    out_path = out_dir / f"{stem}--a{attempt}{src.suffix}"
    if out_path.exists():
        raise RescoreRefused(f"{out_path} already exists -- the attempt counter and the directory disagree; nothing re-scored")
    # inspect's score() resolves EVERY role recorded in the log header into a live provider BEFORE the roles passed here merge in
    # (score_async: log_roles first), and the audit's header names the TARGET as its vLLM proxy (openai/<vmodel>)
    # -- a model a judge-only pass never calls and cannot initialise off the cluster (no OPENAI_API_KEY: the u0 re-score died there
    # on both cells). The header is set aside for the call, so inspect resolves only the judge chain passed here, and restored verbatim
    # on the returned log before the attempt copy is written: the copy carries the audit's wiring header, the judge's served ids live
    # in its events as before. Never a placeholder key in the environment instead: a stub would let a stray target call reach OpenAI.
    header_roles = copy.deepcopy(one.eval.model_roles)
    one.eval.model_roles = None
    rescored = score_fn(one, scorers=list(scorers), model_roles={pr.JUDGE_ROLE: judge_model}, action="overwrite")
    rescored.eval.model_roles = header_roles
    write_fn(rescored, str(out_path))
    return out_path


@dataclass
class RescoreReport:
    selection: RescoreSelection
    scored: List[Dict[str, Any]] = field(default_factory=list)
    refiled_terminal: List[Dict[str, Any]] = field(default_factory=list)
    partial: List[Dict[str, Any]] = field(default_factory=list)      # Grok-fix-4: named, nothing filed, the refusal standing
    rows_filed: int = 0
    logs: List[str] = field(default_factory=list)


def rescored_entry(record: pr.JudgeRefusal, *, outcome: str, selection: RescoreSelection, rescore_log_path: str,
                   refused_by: Sequence[str], new_reason: Optional[str], served: Mapping[str, Any],
                   judge_build: Optional[Mapping[str, Any]], now: float) -> Dict[str, Any]:
    """One ``rescored`` sidecar entry: the superseded terminal record (as it stood), its judge provenance, the selection effect, the
    chain, the revision, the outcome and the new judgment's build identity."""
    return {
        "behavior": record.behavior, "scenario_id": record.scenario_id, "epoch": int(record.epoch), "dimensions": list(record.dimensions),
        "schema": RESCORE_SCHEMA, "outcome": outcome, "rescored_by": RESCORED_BY, "rescored_at": now,
        "requested_by": selection.requested_by, "selection_why": selection.why, "revision": selection.revision(),
        "judge_model": selection.judge_model, "judge_refusal_fallback_models": list(selection.judge_refusal_fallback_models),
        "prior_terminal": pr.judge_refusal_record(record), "prior_refused_by": sorted(set(refused_by)),
        "rescore_log": rescore_log_path, "new_reason": new_reason, "judge_serving_models": dict(served.get("judge") or {}),
        pr.JUDGE_BUILD_KEY: (dict(judge_build) if judge_build else None),
    }


def rescore_cell(store: cs.CellStore, cell: PetriCell, spec: cm.PetriSpec, *, items: Optional[Sequence[Tuple[str, str, int]]],
                 why: str, requested_by: str, scorers_for: Callable[[str], Sequence[Any]], judge_model: Any,
                 dir_key_of: Callable[[str], str], canonical_ids_of: Callable[[str], Sequence[str]], env_digest: str = "",
                 serving_engine: Optional[str] = None, target_decode: Optional[Mapping[str, Any]] = None,
                 target_seed_base: Optional[int] = None, score_fn=None, read_fn=None, write_fn=None,
                 read_log: Optional[Callable[[str], Mapping[str, Any]]] = None, now: Callable[[], float] = time.time) -> RescoreReport:
    """The operator action over one cell. Writes: the one-sample re-scored logs (a new ``rescore-<revision>`` dir), rows
    (petri_runner.file_rows), the sidecar (write_judge_refused with the surviving terminal records and the ``rescored`` list), and
    results.json (the runner's summary builders). Refuses before writing anything on a bad selection; a per-item re-score that yields
    neither rows nor refusal evidence refuses by name after the log is written (the log stays as evidence; nothing is filed)."""
    read_log = read_log or _default_read_log
    chosen = select_terminal_refusals(store, spec, items, read_log=read_log)
    judge, fallbacks = chain_of(spec)
    sel = RescoreSelection(cell_id=cell.cell_id, petri_run_key=cell.petri_run_key,
                           items=tuple((r.behavior, r.scenario_id, int(r.epoch)) for r in chosen), why=why, judge_model=judge,
                           judge_refusal_fallback_models=fallbacks, requested_by=requested_by, selected_at=now())
    report = RescoreReport(selection=sel)
    out_dir = Path(store.cell_dir).parent.parent / "inspect_logs" / store.key / f"{RESCORE_DIR_PREFIX}{sel.revision()}"
    selected = set(sel.items)
    remaining = [r for r in pr.read_judge_refused(store) if (r.behavior, r.scenario_id, int(r.epoch)) not in selected]
    # a selected record re-enters `remaining` below when its re-score is partial (it stands) or re-filed (under the new judge)
    new_rows: List[cs.ResultRow] = []
    entries: List[Dict[str, Any]] = list(pr.read_judge_rescored(store))
    epochs = spec.bloom_options.epochs
    for rec in chosen:
        refused_by = judge_families_of_record(rec, read_log)
        path = rescore_log(rec, scorers=scorers_for(rec.behavior), judge_model=judge_model, out_dir=out_dir,
                           score_fn=score_fn, read_fn=read_fn, write_fn=write_fn)
        report.logs.append(str(path))
        log = read_log(str(path))
        dk = dir_key_of(rec.behavior)
        try:
            rows, _entries = pr.convert_eval_log(log, spec=spec, cell=cell, dir_key=dk, canonical_ids=canonical_ids_of(rec.behavior),
                                                 env_digest=env_digest, serving_engine=serving_engine, target_decode=target_decode,
                                                 target_seed_base=target_seed_base)
        except pr.PetriRunnerError as exc:
            # the converter's own refusals (a NaN without refusal evidence, a malformed scorer entry, a provenance mismatch) stand as
            # the verdict on the re-scored sample: named, the log kept as evidence, nothing filed
            raise RescoreRefused(f"{rec.behavior}/{rec.scenario_id}#e{rec.epoch}: the converter refused the re-scored sample -- {exc}; "
                                 f"the log is at {path}; nothing is filed") from exc
        sample = find_sample(log, rec)
        served = pr.sample_role_serving_models(sample)
        item_id = pr._item_id(rec.scenario_id, int(rec.epoch), epochs)
        task = pr.PETRI_TASK_PREFIX + rec.behavior
        item_rows = [r for r in rows if r.item_id == item_id and r.task == task]
        item_ledger = [e for e in _entries if e.behavior == rec.behavior and e.scenario_id == rec.scenario_id and int(e.epoch) == int(rec.epoch)]
        refusal = pr.judge_refusal(sample)
        build = pr.judge_build_of_sample(sample)
        required = set(pr.required_metrics(rec.behavior))
        complete = {r.metric for r in item_rows} == required and len(item_rows) == len(required) and not item_ledger
        if item_rows and complete:
            new_rows.extend(item_rows)
            entries.append(rescored_entry(rec, outcome="scored", selection=sel, rescore_log_path=str(path), refused_by=refused_by,
                                          new_reason=None, served=served, judge_build=build, now=now()))
            report.scored.append({"behavior": rec.behavior, "scenario_id": rec.scenario_id, "rows": len(item_rows),
                                  "judge": dict(served.get("judge") or {}), pr.JUDGE_BUILD_KEY: build})
        elif item_rows or (item_ledger and refusal is None):
            # Grok-fix-4: a PARTIAL judgment -- one or two of the three jointly judged dimensions scored, the
            # rest in the converter's ledger (dimension missing, a NaN) -- files NOTHING: no row (a partially filed item is what
            # _reconcile_done_marker refuses, and would leave the cell un-ingestable), the terminal refusal stands, the marker is
            # not rewritten. The outcome is recorded by name with the ledger preserved; the next attempt gets its own log.
            ledger_recs = [pr._ledger_record(e) for e in item_ledger]
            entries.append(rescored_entry(rec, outcome="partial", selection=sel, rescore_log_path=str(path), refused_by=refused_by,
                                          new_reason=None, served=served, judge_build=build, now=now())
                           | {"rows_dropped": [r.metric for r in item_rows], "ledger": ledger_recs})
            remaining.append(rec)
            report.partial.append({"behavior": rec.behavior, "scenario_id": rec.scenario_id, "rows_dropped": [r.metric for r in item_rows],
                                   "ledger": ledger_recs})
        elif refusal is not None:
            new_judge = pr.refusing_judge_model(sample, spec)
            reason = pr.judge_refused_reason(new_judge, refusal)
            refiled = pr.JudgeRefusal(behavior=rec.behavior, scenario_id=rec.scenario_id, epoch=int(rec.epoch), dimensions=tuple(rec.dimensions),
                                      reason=reason, judge_model=new_judge, sample_id=rec.sample_id, sample_uuid=rec.sample_uuid,
                                      eval_log=str(path), auditor_chain=rec.auditor_chain, **pr._judge_build_fields(sample, new_judge))
            remaining.append(refiled)
            entries.append(rescored_entry(rec, outcome="refiled_terminal", selection=sel, rescore_log_path=str(path), refused_by=refused_by,
                                          new_reason=reason, served=served, judge_build=build, now=now()))
            report.refiled_terminal.append({"behavior": rec.behavior, "scenario_id": rec.scenario_id, "judge": new_judge, pr.JUDGE_BUILD_KEY: build})
        else:
            raise RescoreRefused(f"{rec.behavior}/{rec.scenario_id}#e{rec.epoch}: the re-score produced neither rows nor refusal evidence "
                                 f"(no judge scores on the sample) -- the log is at {path}; nothing is filed")
    report.rows_filed = pr.file_rows(store, new_rows) if new_rows else 0
    pr.write_judge_refused(store, cell, remaining, rerolled=pr.read_judge_rerolled(store) or None, rescored=entries)
    if report.scored or report.refiled_terminal:
        refresh_results(store, cell, spec, canonical_ids={b: list(canonical_ids_of(b)) for b in spec.behaviors}, judge_refused=remaining)
    return report


def refresh_results(store: cs.CellStore, cell: PetriCell, spec: cm.PetriSpec, *, canonical_ids: Mapping[str, Sequence[str]],
                    judge_refused: Sequence[pr.JudgeRefusal]) -> Dict[str, Any]:
    """Rebuild results.json from the store as it now stands, through the RUNNER's own summary builders: per-behaviour summaries over
    rows.jsonl with the surviving terminal records, the audited footing, the role models; ``rows_emitted`` is the store's row count.
    A re-score never leaves a ledger entry unresolved (it touches no auditor casualty), so the done marker is rewritten; the registry
    hand-off's close-leg recount then files the run's new store total."""
    behaviors_summary = pr._behavior_summaries(store, spec, canonical_ids, [], auditor_refused=pr.read_auditor_refused(store),
                                               judge_refused=list(judge_refused))
    footing = pr.audited_footing(behaviors_summary)
    rows_path = store.rows_path
    rows_emitted = sum(1 for _ in open(rows_path, encoding="utf-8")) if rows_path.is_file() else 0
    rescored = pr.read_judge_rescored(store)
    summary = {"cell_id": cell.cell_id, "petri_run_key": cell.petri_run_key, "behaviors": behaviors_summary, "rows_emitted": rows_emitted,
               "unscored": 0, **footing, **pr._role_models_block(spec), "completed_at": time.time(),
               pr.JUDGE_RESCORED_KEY: {"n": len(rescored), "by": RESCORED_BY, "revisions": sorted({str(e.get("revision")) for e in rescored})}}
    store.write_results(summary)
    return summary
