"""The judge-only RE-SCORE operation (sisteer.capeval.tiers.petri_rescore) over its injectable seams,
on the shared synthetic Petri world. Synthetic names throughout; no API call, no inspect run: the inspect seams (read / score / write)
are stubbed, the converter, the store, the sidecar and the summary builders are real."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval import store as cs
from sisteer.capeval.petri import auditor_fallback as af
from sisteer.capeval.tiers import petri_rescore as prs
from sisteer.capeval.tiers import petri_runner as pr
from test_petri_judge_refusal import refused_sample
from test_petri_refusal_fallback import OPUS48, SONNET46, _event
from test_petri_store_shape import BEHAVIORS, SIDS, convert, full_dims, make_log, sample, world  # noqa: F401

GROK = "openrouter/x-ai/grok-4.6-20260810"
OPUS5 = "anthropic/claude-opus-5"


def _grok_spec(world):
    return replace(world.spec, auditor_refusal_fallback_models=[SONNET46], judge_refusal_fallback_models=[OPUS48, GROK])


def _pinned(served="x-ai/grok-4.6", pin=GROK):
    e = _event("judge", pin, served)
    att = af.attest_build(pin, SimpleNamespace(model=served, choices=[SimpleNamespace(message=SimpleNamespace(model=served))]))
    e["output"]["metadata"] = {af.REFUSAL_FALLBACK_SERVED_KEY: {"requested": OPUS5, "served": pin, "outcome": "served", "attempts": [],
                                                                "client_side": True, af.BUILD_ATTESTATION_KEY: att}}
    return e


def _opus_refused_events():
    return [_event("judge", OPUS5, "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8")]


#: the model_roles header every live audit log carries: the target is the vLLM proxy served as an openai/ model
HEADER_ROLES = {"auditor": "anthropic/claude-sonnet-5", "judge": OPUS5, "target": "openai/petri__prompted__synthetic"}


def _record(behavior, sid, *, judge=OPUS5, log="/synthetic/logs/orig.eval"):
    return pr.JudgeRefusal(behavior=behavior, scenario_id=sid, epoch=1, dimensions=pr.required_metrics(behavior),
                           reason=pr.judge_refused_reason(judge, "declined"), judge_model=judge, sample_id=sid,
                           sample_uuid=f"uuid-{sid}", eval_log=log)


class _World:
    """One complete cell: every item scored except the terminal refusals under test; the original log dict (by path) carries the
    refused samples with opus-5 + opus-4-8 refusal events; the re-scored log dict is what the stubbed inspect `score` 'produced'."""

    def __init__(self, world, tmp_path, *, refused, rescored_samples):
        self.world = world
        self.spec = _grok_spec(world)
        self.cell = world.cells["si_safety__psr_kl_all"]
        self.out_root = tmp_path / "petri_out"
        self.store = cs.CellStore(self.out_root, self.cell.petri_run_key)
        self.refused = refused
        b_sids = {}
        for b in BEHAVIORS:
            scored = [sample(s, full_dims(b)) for s in SIDS if (b, s) not in {(r.behavior, r.scenario_id) for r in refused}]
            rows, _ = convert(world, self.cell, b, scored, spec=self.spec)
            pr.file_rows(self.store, rows)
            b_sids[b] = scored
        pr.write_judge_refused(self.store, self.cell, list(refused))
        self.store.write_results({"cell_id": self.cell.cell_id, "synthetic": True})
        self.logs = {}
        for r in refused:
            s = refused_sample(r.scenario_id); s["events"] = _opus_refused_events(); s["uuid"] = r.sample_uuid
            self.logs.setdefault(str(r.eval_log), []).append((r.behavior, s))
        self.rescored_samples = rescored_samples          # scenario_id -> the re-scored sample dict
        self.written = []
        self.scored_calls = []

    # --- the seams ---
    def read_log(self, path):
        """A log dict by path: the original (refused samples) or a re-scored one-sample log (by basename under a rescore- dir)."""
        p = Path(path)
        if p.parent.name.startswith(prs.RESCORE_DIR_PREFIX):
            sid = self._sid_of_written(p)
            b = next(r.behavior for r in self.refused if r.scenario_id == sid)
            return make_log(self.world, self.cell, b, [self.rescored_samples[sid]], spec=self.spec) | {"location": str(p)}
        if path not in self.logs:
            raise FileNotFoundError(path)
        b = self.logs[path][0][0]
        return make_log(self.world, self.cell, b, [s for _, s in self.logs[path]], spec=self.spec) | {"location": path}

    def _sid_of_written(self, p):
        for q, sid in self.written:
            if Path(q) == p:
                return sid
        raise AssertionError(f"no re-scored log written at {p}")

    def read_fn(self, path):
        # the audit's wiring header, as every live log carries it (the target is the vLLM proxy served as an openai/ model)
        return SimpleNamespace(location=path, eval=SimpleNamespace(model_roles=dict(HEADER_ROLES)),
                               samples=[SimpleNamespace(id=s["id"], epoch=s["epoch"], uuid=s["uuid"]) for _, s in self.logs[path]])

    def score_fn(self, one, *, scorers, model_roles, action):
        assert action == "overwrite" and pr.JUDGE_ROLE in model_roles and len(one.samples) == 1
        assert one.eval.model_roles is None, "g24: the header roles must be set aside for the scoring call (inspect would resolve the target)"
        self.scored_calls.append((one.samples[0].id, tuple(scorers), model_roles[pr.JUDGE_ROLE]))
        return one

    def write_fn(self, log, path):
        assert log.eval.model_roles == HEADER_ROLES, "g24: the written attempt copy carries the audit's wiring header verbatim"
        self.written.append((path, str(log.samples[0].id)))
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(b"synthetic re-scored log")

    def run(self, items=None):
        b_of = {r.scenario_id: r.behavior for r in self.refused}
        return prs.rescore_cell(self.store, self.cell, self.spec, items=items, why="the manifest's judge chain gained a terminal leg (D869)",
                                requested_by="synthetic-operator", scorers_for=lambda b: (f"scorer:{b}",), judge_model="judge-wrapper",
                                dir_key_of=lambda b: b, canonical_ids_of=lambda b: SIDS, env_digest="ee" * 8,
                                score_fn=self.score_fn, read_fn=self.read_fn, write_fn=self.write_fn, read_log=self.read_log, now=lambda: 1234.5)


def test_selection_is_by_provenance_and_the_ledger_and_refuses_by_name(world, tmp_path):
    """The selector shared with the overflow pass: the two refusals opus-5 and opus-4-8 issued are re-scorable under the Grok chain; a
    record the terminal leg already refused, a record the ledger names, and a record whose log cannot be read are refused by name;
    an explicit item that is not a terminal record is refused by name."""
    b0, b1 = BEHAVIORS
    refused = [_record(b0, SIDS[0]), _record(b1, SIDS[1]), _record(b0, SIDS[1], judge=GROK, log="/synthetic/logs/grok.eval")]
    w = _World(world, tmp_path, refused=refused, rescored_samples={})
    w.logs["/synthetic/logs/grok.eval"][0][1]["events"] = _opus_refused_events() + [_pinned()]
    chosen = prs.select_terminal_refusals(w.store, w.spec, read_log=w.read_log)
    assert [(r.behavior, r.scenario_id) for r in chosen] == sorted([(b0, SIDS[0]), (b1, SIDS[1])])      # the selection is sorted by (behaviour, scenario, epoch)
    with pytest.raises(prs.RescoreRefused, match="includes the manifest's terminal judge"):
        prs.select_terminal_refusals(w.store, w.spec, [(b0, SIDS[1], 1)], read_log=w.read_log)
    with pytest.raises(prs.RescoreRefused, match="not terminal judge refusals"):
        prs.select_terminal_refusals(w.store, w.spec, [(b0, "never_refused", 1)], read_log=w.read_log)
    # the ledger settles an item for both mechanisms
    pr.write_judge_refused(w.store, w.cell, refused, rescored=[{"behavior": b0, "scenario_id": SIDS[0], "epoch": 1, "rescored_by": "x", "outcome": "scored"}])
    assert [(r.behavior, r.scenario_id) for r in prs.select_terminal_refusals(w.store, w.spec, read_log=w.read_log)] == [(b1, SIDS[1])]
    with pytest.raises(prs.RescoreRefused, match="already re-rolled or re-scored"):
        prs.select_terminal_refusals(w.store, w.spec, [(b0, SIDS[0], 1)], read_log=w.read_log)
    assert pr.rerollable_judge_refusals(refused[:1], w.spec, pr.settled_ledger(w.store)) == (refused[:1], [])
    # a record whose log is unreadable is refused by name, never guessed about
    def broken(path):
        raise OSError("gone")
    with pytest.raises(prs.RescoreRefused, match="cannot read"):
        prs.select_terminal_refusals(w.store, w.spec, read_log=broken)
    # no declared judge fallback: nothing is re-scorable, by name
    with pytest.raises(prs.RescoreRefused, match="no terminal judge refusal is re-scorable"):
        prs.select_terminal_refusals(w.store, replace(w.spec, judge_refusal_fallback_models=[]), read_log=w.read_log)


def test_rescore_cell_files_the_new_rows_records_the_superseded_refusal_and_rebuilds_results(world, tmp_path):
    """Two terminal refusals re-scored: one the Grok leg answers (rows filed with judge_build declared_unverified, the record leaves the
    standing set, a `rescored` entry names the superseded record, its provenance, the chain, the revision and the outcome), one Grok
    refuses too (re-filed as a terminal refusal under the dated Grok id with its build class; `refiled_terminal`). results.json is
    rebuilt through the runner's summary builders and carries the rescored block; the one-sample logs are written beside the original
    under rescore-<revision>/, once."""
    b0, b1 = BEHAVIORS
    refused = [_record(b0, SIDS[0]), _record(b1, SIDS[1])]
    scored = sample(SIDS[0], full_dims(b0)); scored["uuid"] = f"uuid-{SIDS[0]}"; scored["events"] = _opus_refused_events() + [_pinned()]
    still = refused_sample(SIDS[1]); still["uuid"] = f"uuid-{SIDS[1]}"; still["events"] = _opus_refused_events() + [_pinned()]
    w = _World(world, tmp_path, refused=refused, rescored_samples={SIDS[0]: scored, SIDS[1]: still})
    before_keys = w.store.completed_keys()
    report = w.run()
    rev = report.selection.revision()
    assert report.rows_filed == 3 and [x["scenario_id"] for x in report.scored] == [SIDS[0]] and [x["scenario_id"] for x in report.refiled_terminal] == [SIDS[1]]
    b_of = {SIDS[0]: b0, SIDS[1]: b1}                                           # the selection is sorted by (behaviour, scenario): order is not the contract
    assert sorted(c[0] for c in w.scored_calls) == sorted([SIDS[0], SIDS[1]]) and all(c[1] == (f"scorer:{b_of[c[0]]}",) and c[2] == "judge-wrapper" for c in w.scored_calls)
    # the one-sample logs: beside the original, under the revision dir, one per item
    assert len(w.written) == 2 and all(Path(p).parent.name == f"{prs.RESCORE_DIR_PREFIX}{rev}" for p, _ in w.written)
    assert sorted(Path(p).name for p, _ in w.written) == sorted(f"orig--{b}--{s}--e1--a1.eval" for b, s in ((b0, SIDS[0]), (b1, SIDS[1])))   # one file per ITEM of a shared log, attempt-numbered
    # rows: the scored item's three dimensions, carrying the judgment's build identity
    new_keys = w.store.completed_keys() - before_keys
    item = pr._item_id(SIDS[0], 1, w.spec.bloom_options.epochs)
    assert new_keys == {(pr.PETRI_TASK_PREFIX + b0, item, m) for m in pr.required_metrics(b0)}
    rows = [r for r in w.store._parsed_rows() if r["item_id"] == item and r["task"] == pr.PETRI_TASK_PREFIX + b0]   # the other behaviour's rows share the item id
    assert len(rows) == 3
    assert all(json.loads(r["extra_json"])[pr.JUDGE_BUILD_KEY] == {"requested": GROK, "served": "x-ai/grok-4.6", "class": af.BUILD_CLASS_DECLARED} for r in rows)
    # the sidecar: the scored record gone, the refused one re-filed under Grok with its class, the ledger naming both
    standing = {r.scenario_id: r for r in pr.read_judge_refused(w.store)}
    assert set(standing) == {SIDS[1]} and standing[SIDS[1]].judge_model == GROK and GROK in standing[SIDS[1]].reason
    assert (standing[SIDS[1]].judge_requested, standing[SIDS[1]].judge_served, standing[SIDS[1]].judge_build_class) == (GROK, "x-ai/grok-4.6", af.BUILD_CLASS_DECLARED)
    entries = {e["scenario_id"]: e for e in pr.read_judge_rescored(w.store)}
    assert set(entries) == {SIDS[0], SIDS[1]}
    e0, e1 = entries[SIDS[0]], entries[SIDS[1]]
    assert e0["outcome"] == "scored" and e1["outcome"] == "refiled_terminal" and e0["revision"] == e1["revision"] == rev
    assert e0["prior_terminal"]["judge_model"] == OPUS5 and e0["prior_refused_by"] == ["claude-opus-4-8", "claude-opus-5"]
    assert e0["judge_refusal_fallback_models"] == [OPUS48, GROK] and e0["requested_by"] == "synthetic-operator" and e0["rescored_at"] == 1234.5
    assert e0[pr.JUDGE_BUILD_KEY] == {"requested": GROK, "served": "x-ai/grok-4.6", "class": af.BUILD_CLASS_DECLARED}
    assert e1["new_reason"].startswith(pr.JUDGE_REFUSED_REASON_PREFIX) and GROK in e1["new_reason"]
    assert e0["schema"] == prs.RESCORE_SCHEMA and e0["rescored_by"] == prs.RESCORED_BY
    # results.json: rebuilt, complete, the rescored block and the judge_build tally on the footing
    results = json.loads(w.store.results_path.read_text())
    assert results[pr.JUDGE_RESCORED_KEY] == {"n": 2, "by": prs.RESCORED_BY, "revisions": [rev]}
    assert results["behaviors"][b0]["incomplete"] is False and results["behaviors"][b0]["mean"][b0] is not None
    assert results[pr.JUDGE_BUILD_KEY]["by_class"] == {af.BUILD_CLASS_DECLARED: 1} and results["judge_refused"] == {"total": 1}
    # the overflow pass reads the same ledger: neither item is a re-roll candidate any more
    assert pr.rerollable_judge_refusals(refused, w.spec, pr.read_judge_rerolled(w.store) + pr.read_judge_rescored(w.store)) == (refused, [])
    # a second run: nothing left to re-score, by name (the re-filed Grok refusal stands; the scored item is gone)
    with pytest.raises(prs.RescoreRefused, match="no terminal judge refusal is re-scorable"):
        w.run()


def test_rescore_refuses_a_result_with_neither_rows_nor_refusal_evidence_and_writes_a_log_only_once(world, tmp_path):
    b0 = BEHAVIORS[0]
    refused = [_record(b0, SIDS[0])]
    nan = refused_sample(SIDS[0], flag=False); nan["uuid"] = f"uuid-{SIDS[0]}"; nan["events"] = _opus_refused_events() + [_pinned()]
    w = _World(world, tmp_path, refused=refused, rescored_samples={SIDS[0]: nan})
    with pytest.raises(prs.RescoreRefused, match="the converter refused the re-scored sample .* nothing is filed"):
        w.run()                                            # a NaN without metadata.refusal: the converter's loud shape refusal, by name
    assert pr.read_judge_refused(w.store) == refused and pr.read_judge_rescored(w.store) == []      # nothing filed
    assert len(w.written) == 1 and Path(w.written[0][0]).name.endswith("--a1.eval")
    # Grok-fix-4: the retry is NOT blocked by the earlier log -- it gets attempt 2's name (and refuses the same way here)
    with pytest.raises(prs.RescoreRefused, match="the converter refused"):
        w.run()
    assert len(w.written) == 2 and Path(w.written[1][0]).name.endswith("--a2.eval")


def test_one_sample_log_picks_exactly_the_records_sample():
    log = SimpleNamespace(location="/synthetic/x.eval", samples=[SimpleNamespace(id="a", epoch=1, uuid="uuid-a"), SimpleNamespace(id="b", epoch=1, uuid="uuid-b")])
    rec = _record(BEHAVIORS[0], "b"); one = prs.one_sample_log(log, rec)
    assert [s.id for s in one.samples] == ["b"] and [s.id for s in log.samples] == ["a", "b"]
    with pytest.raises(prs.RescoreRefused, match="need exactly one"):
        prs.one_sample_log(log, _record(BEHAVIORS[0], "zzz"))


def test_a_rescored_cells_new_rows_are_where_consolidate_reads_them(world, tmp_path):
    """consolidate folds the canonical cell rows.jsonl, never the rescore-<revision>/ copies -- those hold
    the one-sample LOGS only; the re-scored rows are appended to the cell's rows.jsonl (petri_runner.file_rows), so the daemon's
    fold-then-analyze picks them up."""
    import pandas as pd
    b0 = BEHAVIORS[0]
    refused = [_record(b0, SIDS[0])]
    scored = sample(SIDS[0], full_dims(b0)); scored["uuid"] = f"uuid-{SIDS[0]}"; scored["events"] = _opus_refused_events() + [_pinned()]
    w = _World(world, tmp_path, refused=refused, rescored_samples={SIDS[0]: scored})
    w.run()
    parquet = cs.consolidate(w.out_root)
    frame = pd.read_parquet(parquet)
    item = pr._item_id(SIDS[0], 1, w.spec.bloom_options.epochs)
    mine = frame[(frame["item_id"] == item) & (frame["task"] == pr.PETRI_TASK_PREFIX + b0)]
    assert sorted(mine["metric"]) == sorted(pr.required_metrics(b0))
    assert all(pr.JUDGE_BUILD_KEY in json.loads(x) for x in mine["extra_json"])
    assert not any(p.name.startswith(prs.RESCORE_DIR_PREFIX) for p in (w.out_root / "cells").rglob("*"))    # no rows under the log copies


def test_a_partial_rescore_files_nothing_keeps_the_refusal_standing_and_records_the_ledger(world, tmp_path):
    """Grok-fix-4: the converter can return one or two score rows plus `dimension missing` ledger entries for a
    three-dimension judged sample; a re-score that is not COMPLETE for the item (every required metric scored, no ledger entry) files
    no row, clears no refusal and rewrites no results.json -- a partially filed item is what _reconcile_done_marker refuses and would
    leave the cell un-ingestable. The outcome is recorded by name (`partial`) with the ledger preserved; the next attempt is not
    blocked by this attempt's log."""
    b0 = BEHAVIORS[0]
    refused = [_record(b0, SIDS[0])]
    two_of_three = sample(SIDS[0], {f"{b0}_propensity": 7, "eval_awareness": 2})      # scenario_realism missing: partial judgment
    two_of_three["uuid"] = f"uuid-{SIDS[0]}"; two_of_three["events"] = _opus_refused_events() + [_pinned()]
    w = _World(world, tmp_path, refused=refused, rescored_samples={SIDS[0]: two_of_three})
    before_keys = w.store.completed_keys(); before_results = w.store.results_path.read_bytes()
    report = w.run()
    assert report.rows_filed == 0 and report.scored == [] and report.refiled_terminal == []
    assert [p["scenario_id"] for p in report.partial] == [SIDS[0]] and report.partial[0]["rows_dropped"] == [b0, "eval_awareness"]
    assert [e["dimension"] for e in report.partial[0]["ledger"]] == ["scenario_realism"] and "missing" in report.partial[0]["ledger"][0]["reason"]
    assert w.store.completed_keys() == before_keys                                   # no row persisted
    assert w.store.results_path.read_bytes() == before_results                        # the marker is not rewritten
    assert pr.read_judge_refused(w.store) == refused                                   # the refusal stands
    (entry,) = pr.read_judge_rescored(w.store)
    assert entry["outcome"] == "partial" and [e["dimension"] for e in entry["ledger"]] == ["scenario_realism"] and entry["rescore_log"].endswith("--a1.eval")
    # a partial attempt is evidence, NOT a settlement -- the item stays re-scorable (and re-rollable) and the
    # second attempt, with the judge now answering every dimension, completes it under attempt 2's log
    assert pr.settled_ledger(w.store) == [] and pr.rerollable_judge_refusals(refused, w.spec, pr.settled_ledger(w.store)) == ([], refused)
    assert [r.scenario_id for r in prs.select_terminal_refusals(w.store, w.spec, [(b0, SIDS[0], 1)], read_log=w.read_log)] == [SIDS[0]]
    full = sample(SIDS[0], full_dims(b0)); full["uuid"] = f"uuid-{SIDS[0]}"; full["events"] = _opus_refused_events() + [_pinned()]
    w.rescored_samples[SIDS[0]] = full
    report2 = w.run()
    assert report2.rows_filed == 3 and [x["scenario_id"] for x in report2.scored] == [SIDS[0]] and report2.partial == []
    assert len(w.written) == 2 and Path(w.written[1][0]).name.endswith("--a2.eval")
    assert pr.read_judge_refused(w.store) == []                                              # the refusal is gone now
    outcomes = [(e["outcome"], Path(e["rescore_log"]).name[-8:]) for e in pr.read_judge_rescored(w.store)]
    assert outcomes == [("partial", "-a1.eval"), ("scored", "-a2.eval")]                      # both attempts kept as evidence
    assert len(pr.settled_ledger(w.store)) == 1 and pr.settled_ledger(w.store)[0]["outcome"] == "scored"
    assert w.store.results_path.read_bytes() != before_results and json.loads(w.store.results_path.read_text())[pr.JUDGE_RESCORED_KEY]["n"] == 2


def test_a_judge_only_rescore_needs_no_provider_for_the_header_target_under_real_inspect(tmp_path, monkeypatch):
    """inspect_ai.score resolves every role in the log header before the roles passed to it merge in -- so a header whose
    target is the audit's vLLM proxy (openai/<vmodel>) made the u0 re-score die at "Unable to initialise OpenAI client" on a machine with
    no OPENAI_API_KEY, before any judge call. Against the real inspect: the raw score() on such a log still fails (the hazard), while
    rescore_log sets the header aside for the call, scores the sample with the judge chain passed to it, and writes the attempt copy
    with the header restored verbatim -- the target still named as the proxy, the judge's output in the sample's scores."""
    pytest.importorskip("inspect_ai", reason="the `petri` dependency group (pyproject: grading-host-only; never in a cluster or gate venv)")
    from inspect_ai import Task, eval as ieval, score as iscore
    from inspect_ai.dataset import Sample
    from inspect_ai.model import get_model
    from inspect_ai.model._model_config import ModelConfig
    from inspect_ai.scorer import Score, accuracy, scorer
    from inspect_ai._util.error import PrerequisiteError                                        # inspect's own class; not re-exported from inspect_ai.util
    monkeypatch.delenv("OPENAI_API_KEY", raising=False); monkeypatch.delenv("AZUREAI_OPENAI_API_KEY", raising=False)

    @scorer(metrics=[accuracy()])
    def judge_says():
        async def score(state, target):
            out = await get_model(role="judge").generate("judge this transcript")
            return Score(value=1, explanation=out.completion)
        return score
    (log,) = ieval(Task(dataset=[Sample(input="hello", id="malware_analyst_reverse_request")], scorer=judge_says()), model="mockllm/model",
                   model_roles={"auditor": "mockllm/model", "judge": "mockllm/model", "target": "mockllm/model"},
                   log_dir=str(tmp_path / "logs"), display="none", log_format="eval")
    assert log.status == "success"
    log.eval.model_roles["target"] = ModelConfig(model="openai/petri__prompted__synthetic")          # the audit's header shape
    header_before = {k: v.model for k, v in log.eval.model_roles.items()}
    with pytest.raises(PrerequisiteError, match="OpenAI client"):                                    # the hazard, on the raw call
        iscore(log, scorers=[judge_says()], model_roles={pr.JUDGE_ROLE: get_model("mockllm/model")}, action="overwrite", display="none")
    record = pr.JudgeRefusal(behavior="cyber_bio_chem_capabilities", scenario_id="malware_analyst_reverse_request", epoch=1, dimensions=("cyber_bio_chem_capabilities",),
                             reason="judge refused to score (anthropic/claude-opus-5): synthetic", judge_model=OPUS5,
                             sample_id="malware_analyst_reverse_request", sample_uuid=str(log.samples[0].uuid), eval_log=str(tmp_path / "logs" / "orig.eval"))
    written = {}
    out = prs.rescore_log(record, scorers=[judge_says()], judge_model=get_model("mockllm/model"), out_dir=tmp_path / "rescore-synthetic",
                          read_fn=lambda p: log, write_fn=lambda lg, p: written.setdefault(p, lg))
    ((path, rescored),) = written.items()
    assert Path(path) == out and out.name == "orig--cyber_bio_chem_capabilities--malware_analyst_reverse_request--e1--a1.eval"
    assert {k: v.model for k, v in rescored.eval.model_roles.items()} == header_before                  # restored verbatim, target still the proxy
    assert {k: v.model for k, v in log.eval.model_roles.items()} == header_before                       # the original untouched
    (sc,) = rescored.samples[0].scores.values()
    assert sc.value == 1 and "mockllm/model" in str(sc.explanation)                                    # the judge chain passed in did the scoring
