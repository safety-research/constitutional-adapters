"""Judged control runner tests.

FakeEngine + FakeJudge with SYNTHETIC content only — no network, no GPU,
no real Claude, no real dataset text. The XSTest loader's 250/200 gate is
tested by injecting synthetic frames; runner arithmetic uses small
hand-computed sets built directly as rows.
"""

import dataclasses
import json
import os
import stat

import pytest

from sisteer.capeval import judging
from sisteer.capeval import store as cs
from sisteer.capeval.serving import GenOut, PositionScore
from sisteer.capeval.tiers import judged, parity
from sisteer.capeval.tiers.judged import (
    GenSpool,
    JudgedError,
    JudgedSpec,
    TASK_ORBENCH,
    TASK_XSTEST,
    spool_header,
)

REV = "0123456789abcdef0123456789abcdef01234567"
TEMPLATE = {
    "sweep": "judged_test", "run_id": "r0", "model_id": "test-org/tiny",
    "model_revision": REV, "arm_name": "none", "kind": "none", "coeff": None,
    "intervention_hash": "a1b2c3d4e5f60718", "engine": "hf", "trial_id": 0,
    "seed": 0, "attempt": 0, "harness_config_hash": "0f1e2d3c4b5a6978",
    "env_digest": "99aabbccddeeff00", "slurm_job_id": None,
}
DOMAIN_OF = {TASK_ORBENCH: "overrefusal_judged",
             TASK_XSTEST: "refusal_calibration_judged",
             parity.PARITY_TASK_GSM8K: "math_s2parity",
             parity.PARITY_TASK_MMLU: "knowledge_s2parity"}


class _Tok:
    """No chat template: build_chat_prompt falls back to plain concat,
    so the rendered prompt is inspectable text."""

    chat_template = None


class GenEngine:
    """Canned generations keyed by which row prompt the rendered prompt
    contains; records every gen_cfg (batch pass-through contract)."""

    name = "hf"

    def __init__(self, text_of):
        self.tokenizer = _Tok()
        self._text_of = text_of
        self.gen_cfgs = []
        self.prompts_seen = []

    def generate(self, prompts, arm, gen_cfg):
        self.gen_cfgs.append(dict(gen_cfg))
        outs = []
        for p in prompts:
            self.prompts_seen.append(p)
            outs.append(GenOut(text=self._text_of(p), gen_token_logprobs=[],
                               ended_with_eos=True))
        return outs


class FakeJudge:
    """Deterministic canned judge — the judging.ClaudeJudge duck type."""

    temperature = judging.TEMPERATURE
    max_tokens = judging.MAX_TOKENS

    def __init__(self, cpi_of=None, pair_of=None, fail=False):
        self._cpi_of = cpi_of
        self._pair_of = pair_of
        self._fail = fail
        self.n_cache_hits = 0
        self.n_api_calls = 0
        self.n_calls_by_org = {"lowprio": 0}
        self.n_reasked = 0
        self.judged = []

    def judge_all(self, pairs, *, allow_refusal_abstention=False):
        if self._fail:
            raise AssertionError("judge must not be called on a fully "
                                 "resumed store")
        out = {}
        for req, rubric in pairs:
            self.judged.append(req)
            self.n_api_calls += 1
            # Mirror ClaudeJudge: an abstention sentinel (with its reason)
            # only under the opt-in; keyed off candidate_a ONLY, so a
            # paired request abstains in exactly the order that presents
            # the trigger text as candidate A — which is how a ONE-order
            # abstention is exercised (the runner must drop the item).
            if "REASK" in req.candidate_a:
                # a malformed first verdict the real judge re-asks ONCE
                self.n_reasked += 1
            if allow_refusal_abstention and "FILTERED" in req.candidate_a:
                # the API declined to return the reply (400 signature)
                out[req.key()] = {"abstained": True,
                                  "reason": "content_filter",
                                  "request_id": "req_synthetic_cf"}
            elif allow_refusal_abstention and "TRUNCATE" in req.candidate_a:
                out[req.key()] = {"abstained": True, "reason": "max_tokens"}
            elif allow_refusal_abstention and "ABSTAIN" in req.candidate_a:
                out[req.key()] = {"abstained": True, "reason": "refusal"}
            elif allow_refusal_abstention and "TRANSPORT" in req.candidate_a:
                # the all-529-while-serving sentinel ClaudeJudge files
                out[req.key()] = {"abstained": True,
                                  "reason": "transport_refusal",
                                  "attempts": 6, "siblings_succeeded": 3,
                                  "evidence": "siblings",
                                  "request_id": "req_synthetic"}
            elif rubric.paired:
                out[req.key()] = {"verdict": self._pair_of(req)}
            else:
                out[req.key()] = {"label": self._cpi_of(req)}
        return out


def _store(tmp_path, key="none__judgedtest__t0__a0__a1b2c3d4__0f1e2d3c"):  # synthetic cell key — gitleaks:allow
    return cs.CellStore(tmp_path, key, row_constants=TEMPLATE)


def _spec(task, rows, **kw):
    return JudgedSpec(task=task, template=dict(TEMPLATE),
                      domain_of=dict(DOMAIN_OF), rows=list(rows), **kw)


def _spool(tmp_path, spec, name="gens.jsonl"):
    return GenSpool(tmp_path / "private" / name,
                    spool_header(spec, intervention_hash="a1b2c3d4e5f60718",
                                 model_revision=REV))


# --- loaders: counts, ids, splits ---------------------------------------------


def test_or_bench_loader_counts_and_duplicate_texts_stay_distinct():
    import pandas as pd

    prompts = [f"synthetic benign-but-scary prompt {i}" for i in range(1318)]
    prompts.append(prompts[0])  # a DUPLICATE text (r1 #18)
    df = pd.DataFrame({"prompt": prompts,
                       "category": ["synthcat"] * 1319})
    rows = judged.or_bench_rows_from_frame(df)
    assert len(rows) == 1319
    assert len({r["item_id"] for r in rows}) == 1319  # row index keeps dups distinct
    assert rows[0]["item_id"].split("-")[1] == rows[-1]["item_id"].split("-")[1]
    assert rows[0]["item_id"] != rows[-1]["item_id"]

    with pytest.raises(JudgedError, match="1319"):
        judged.or_bench_rows_from_frame(df.iloc[:100])
    with pytest.raises(JudgedError, match="category"):
        judged.or_bench_rows_from_frame(pd.DataFrame({"prompt": prompts}))


def test_xstest_loader_split_gate():
    import pandas as pd

    def frame(n_safe, n_unsafe):
        return pd.DataFrame({
            "prompt": [f"synthetic safe {i}" for i in range(n_safe)]
                      + [f"synthetic contrast {i}" for i in range(n_unsafe)],
            "type": ["homonyms_synth"] * n_safe
                    + ["contrast_synth"] * n_unsafe,
        })

    rows = judged.xstest_rows_from_frame(frame(250, 200))
    assert len(rows) == 450
    assert sum(r["subset"] == "safe" for r in rows) == 250
    assert sum(r["subset"] == "unsafe" for r in rows) == 200
    with pytest.raises(JudgedError, match="250"):
        judged.xstest_rows_from_frame(frame(249, 201))


# --- spool: privacy, header binding, resume, completeness gate ------------------


def test_spool_private_dir_and_header_binding(tmp_path):
    spec = _spec(TASK_ORBENCH,
                 [{"item_id": "orb0000-aaaaaaaa", "prompt": "synthetic p",
                   "category": "c"}])
    spool = _spool(tmp_path, spec)
    mode = stat.S_IMODE(os.stat(spool.path.parent).st_mode)
    assert mode == 0o700  # raw generations live in a private dir
    spool.append("orb0000-aaaaaaaa", "some generation")

    # same header -> resume sees the entry
    again = GenSpool(spool.path, spool.header)
    assert again.entries() == {"orb0000-aaaaaaaa": "some generation"}

    # ANY header drift (here: gen config) refuses (r1 #8)
    other = _spec(TASK_ORBENCH, spec.rows,
                  gen={"max_new_tokens": 9, "do_sample": False,
                       "batch_size": 8})
    with pytest.raises(JudgedError, match="header mismatch"):
        GenSpool(spool.path, spool_header(
            other, intervention_hash="a1b2c3d4e5f60718", model_revision=REV))
    # missing spool with create=False is a loud refusal (judge side)
    with pytest.raises(JudgedError, match="does not exist"):
        GenSpool(tmp_path / "private" / "nope.jsonl", spool.header,
                 create=False)


def test_spool_dedups_and_refuses_conflicting_duplicate(tmp_path):
    spec = _spec(TASK_ORBENCH, [{"item_id": "orb0000-aaaaaaaa",
                                 "prompt": "p", "category": "c"}])
    spool = _spool(tmp_path, spec)
    spool.append("orb0000-aaaaaaaa", "text one")
    spool.append("orb0000-aaaaaaaa", "text one")  # identical replay: fine
    assert spool.entries() == {"orb0000-aaaaaaaa": "text one"}
    spool.append("orb0000-aaaaaaaa", "DIFFERENT")
    with pytest.raises(JudgedError, match="DIFFERENT"):
        spool.entries()


def test_generate_resumes_and_chunks(tmp_path):
    rows = [{"item_id": f"orb{i:04d}-{'ab' * 4}",
             "prompt": f"synthetic prompt {i}", "category": "c"}
            for i in range(5)]
    spec = _spec(TASK_ORBENCH, rows,
                 gen={"max_new_tokens": 16, "do_sample": False,
                      "batch_size": 2})
    spool = _spool(tmp_path, spec)
    eng = GenEngine(lambda p: "gen for " + p.strip().splitlines()[0])
    out = judged.generate(eng, None, spec, spool)
    assert out["n_new"] == 5 and out["n_total"] == 5
    assert len(eng.gen_cfgs) == 3  # engine called per chunk: ceil(5/2)
    assert all(c["batch_size"] == 2 for c in eng.gen_cfgs)

    # resume: nothing left to generate, engine untouched
    eng2 = GenEngine(lambda p: (_ for _ in ()).throw(
        AssertionError("no regeneration on a complete spool")))
    out2 = judged.generate(eng2, None, spec,
                           GenSpool(spool.path, spool.header))
    assert out2["n_new"] == 0
    assert eng2.gen_cfgs == []


def test_generate_renders_system_and_transform(tmp_path):
    rows = [{"item_id": "orb0000-aaaaaaaa", "prompt": "the user ask",
             "category": "c"}]
    spec = _spec(TASK_ORBENCH, rows, system="THE SI TEXT")

    class Arm:
        def transform_prompt(self, s):
            return "TRANSFORMED::" + s

    eng = GenEngine(lambda p: "gen")
    judged.generate(eng, Arm(), spec, _spool(tmp_path, spec))
    [p] = eng.prompts_seen
    assert p.startswith("TRANSFORMED::")  # arm transform AFTER chat render
    assert "THE SI TEXT" in p and "the user ask" in p


# --- or_bench judge: rows, resume, aggregates -----------------------------------


def _cpi_label_of(req):
    text = req.candidate_a
    if "REFUSE" in text:
        return "full_refusal"
    if "PARTIAL" in text:
        return "partial_refusal"
    return "full_compliance"


def _orb_setup(tmp_path, n=6):
    rows = [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}",
             "category": "cat_a" if i % 2 == 0 else "cat_b"}
            for i in range(n)]
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    # gens: item 0 refuses, item 1 partial, rest comply
    texts = {rows[0]["item_id"]: "I REFUSE to do that",
             rows[1]["item_id"]: "PARTIAL: here is half an answer"}
    for r in rows:
        spool.append(r["item_id"], texts.get(r["item_id"], "sure, here it is"))
    return rows, spec, spool


def test_or_bench_judge_rows_and_aggregates(tmp_path):
    rows, spec, spool = _orb_setup(tmp_path)
    store = _store(tmp_path)
    judge = FakeJudge(cpi_of=_cpi_label_of)
    summary = judged.judge_or_bench(spool, judge, spec, store)
    assert summary["fracs"]["full_refusal"] == pytest.approx(1 / 6)
    assert summary["fracs"]["partial_refusal"] == pytest.approx(1 / 6)
    assert summary["fracs"]["full_compliance"] == pytest.approx(4 / 6)

    by = {(r["task"], r["item_id"], r["metric"]): r
          for r in store._parsed_rows()}
    # per-item rows: label value + refused flag, rubric_sha in extras
    r0 = by[(TASK_ORBENCH, rows[0]["item_id"], "orb_label")]
    assert r0["value"] == 0.0 and r0["tier"] == 3
    extra = json.loads(r0["extra_json"])
    assert extra["label"] == "full_refusal"
    assert extra["rubric_sha"] == judging.ORBENCH_RUBRIC_V1.sha
    assert extra["category"] == "cat_a"
    assert by[(TASK_ORBENCH, rows[0]["item_id"], "orb_refused")]["value"] == 1.0
    assert by[(TASK_ORBENCH, rows[2]["item_id"], "orb_refused")]["value"] == 0.0
    # aggregates ONLY under *_aggregate names with category breakdown
    agg = by[(TASK_ORBENCH, "__aggregate__", "orb_full_compliance_frac_aggregate")]
    assert agg["value"] == pytest.approx(4 / 6)
    cats = json.loads(agg["extra_json"])["by_category"]
    assert cats["cat_a"]["n"] == 3 and cats["cat_b"]["n"] == 3
    assert (TASK_ORBENCH, "__aggregate__", "orb_label") not in by
    # position_order records the permuted processing order
    orders = {r["item_id"]: r["position_order"] for r in store._parsed_rows()
              if r["metric"] == "orb_label"}
    assert sorted(orders.values()) == list(range(6))


def test_or_bench_abstention_bounds_and_field_gating(tmp_path, monkeypatch):
    """Judge-refusal abstentions: bound over the full n, and DON'T emit the
    historical orb_{lab}_frac_aggregate field (it would silently become a
    biased graded-subset estimate under the name old cells used for full-n).
    2 abstentions of 6 is far above the real 5% ceiling (which small-n
    arithmetic can't stay under), so the cap is widened here — the bounds
    semantics under test are orthogonal to it; the ceiling has its own
    test at the real constant."""
    monkeypatch.setattr(judged, "JUDGE_ABSTENTION_MAX_FRAC", 0.5)
    rows = [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}",
             "category": "cat_a" if i % 2 == 0 else "cat_b"}
            for i in range(6)]
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    # item0 refuses, item1 partial, items 2&3 trip the fake judge's ABSTAIN,
    # items 4&5 comply.
    texts = {rows[0]["item_id"]: "I REFUSE to do that",
             rows[1]["item_id"]: "PARTIAL: here is half an answer",
             rows[2]["item_id"]: "ABSTAIN please",
             rows[3]["item_id"]: "ABSTAIN please"}
    for r in rows:
        spool.append(r["item_id"], texts.get(r["item_id"], "sure, here it is"))
    store = _store(tmp_path)
    judge = FakeJudge(cpi_of=_cpi_label_of)
    summary = judged.judge_or_bench(spool, judge, spec, store)

    assert summary["n_abstained"] == 2
    assert summary["n_graded"] == 4
    assert summary["abstained_by_reason"] == {"refusal": 2}
    # graded: item0 full_refusal, item1 partial, items 4,5 comply → 1 refusal.
    # bounds over full n=6: lo=1/6 (abstain=comply), hi=3/6 (abstain=refuse).
    assert summary["orb_refused_lo"] == pytest.approx(1 / 6)
    assert summary["orb_refused_hi"] == pytest.approx(3 / 6)

    by = {(r["task"], r["item_id"], r["metric"]): r
          for r in store._parsed_rows()}
    # historical field GATED off when abstentions exist
    assert (TASK_ORBENCH, "__aggregate__",
            "orb_full_refusal_frac_aggregate") not in by
    # bounds field present, headline = hi (conservative worst-case)
    b = by[(TASK_ORBENCH, "__aggregate__", "orb_refused_bounds_aggregate")]
    assert b["value"] == pytest.approx(3 / 6)
    bx = json.loads(b["extra_json"])
    assert bx["orb_refused_lo"] == pytest.approx(1 / 6)
    assert bx["n_abstained"] == 2 and bx["n"] == 6
    # abstained items left NOT done (retried on a later run) — no label row
    assert (TASK_ORBENCH, rows[2]["item_id"], "orb_label") not in by


def test_or_bench_judge_resume_files_once_and_skips_judged(tmp_path):
    rows, spec, spool = _orb_setup(tmp_path)
    store = _store(tmp_path)
    judged.judge_or_bench(spool, FakeJudge(cpi_of=_cpi_label_of), spec, store)
    n_rows = len(store._parsed_rows())
    # a rerun judges NOTHING (per-item resume) and files NO new rows
    summary = judged.judge_or_bench(spool, FakeJudge(fail=True), spec, store)
    assert len(store._parsed_rows()) == n_rows
    assert summary["fracs"]["full_compliance"] == pytest.approx(4 / 6)


def test_abstention_ceiling_refuses_loudly_at_the_real_constant(tmp_path):
    """1 abstention of 6 = 16.7% — above the real 5% ceiling: the control
    must refuse loudly rather than report a bound whose band swamps the
    effect (the branch drop-count-cap design's gate, on top of the
    bounding semantics)."""
    rows = [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}", "category": "cat_a"}
            for i in range(6)]
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    for i, r in enumerate(rows):
        spool.append(r["item_id"],
                     "ABSTAIN please" if i == 0 else "sure, here it is")
    with pytest.raises(JudgedError, match="ceiling"):
        judged.judge_or_bench(spool, FakeJudge(cpi_of=_cpi_label_of),
                              spec, _store(tmp_path))


def test_or_bench_abstention_reason_split(tmp_path, monkeypatch):
    """A max_tokens truncation abstains like a refusal but NEVER under the
    same name: abstained_by_reason splits them, in the summary and in the
    bounds row's extra. A legacy cached sentinel with no reason key reads
    back as refusal (truncations could not produce sentinels before the
    reason existed — they hard-failed the cell)."""
    monkeypatch.setattr(judged, "JUDGE_ABSTENTION_MAX_FRAC", 0.5)
    rows = [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}", "category": "cat_a"}
            for i in range(6)]
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    texts = {rows[0]["item_id"]: "ABSTAIN please",
             rows[1]["item_id"]: "TRUNCATE this one"}
    for r in rows:
        spool.append(r["item_id"], texts.get(r["item_id"], "sure thing"))
    store = _store(tmp_path)
    summary = judged.judge_or_bench(spool, FakeJudge(cpi_of=_cpi_label_of),
                                    spec, store)
    assert summary["n_abstained"] == 2
    assert summary["abstained_by_reason"] == {"refusal": 1, "max_tokens": 1}
    by = {(r["item_id"], r["metric"]): r for r in store._parsed_rows()}
    bx = json.loads(by[("__aggregate__", "orb_refused_bounds_aggregate")]
                    ["extra_json"])
    assert bx["abstained_by_reason"] == {"refusal": 1, "max_tokens": 1}

    # legacy sentinel (pre-reason cache rows): reads back as refusal
    class _LegacyJudge(FakeJudge):
        def judge_all(self, pairs, *, allow_refusal_abstention=False):
            out = super().judge_all(
                pairs, allow_refusal_abstention=allow_refusal_abstention)
            for k, v in out.items():
                if v.get("abstained"):
                    out[k] = {"abstained": True}  # no reason key
            return out

    store2 = _store(tmp_path / "legacy")
    spool2 = _spool(tmp_path / "legacy", spec)
    for r in rows:
        spool2.append(r["item_id"], texts.get(r["item_id"], "sure thing"))
    summary2 = judged.judge_or_bench(
        spool2, _LegacyJudge(cpi_of=_cpi_label_of), spec, store2)
    assert summary2["abstained_by_reason"] == {"refusal": 2}


def test_or_bench_transport_refusal_is_its_own_reason_same_ceiling(
        tmp_path, monkeypatch):
    """A refusal that arrived as a deterministic 529 (transport_refusal,
    2026-09-02) abstains like the other two classes — no label row (the
    item stays not-done), bounded over the full n — but NEVER under their
    names: the split carries it as its own reason in the summary and the
    bounds row. Cap widened (2 of 6 can't sit under 5%); the real-constant
    gate has its own test below."""
    monkeypatch.setattr(judged, "JUDGE_ABSTENTION_MAX_FRAC", 0.5)
    rows = [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}", "category": "cat_a"}
            for i in range(6)]
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    texts = {rows[0]["item_id"]: "TRANSPORT-declined body",
             rows[1]["item_id"]: "ABSTAIN please"}
    for r in rows:
        spool.append(r["item_id"], texts.get(r["item_id"], "sure thing"))
    store = _store(tmp_path)
    summary = judged.judge_or_bench(spool, FakeJudge(cpi_of=_cpi_label_of),
                                    spec, store)
    assert summary["n"] == 6 and summary["n_graded"] == 4
    assert summary["n_abstained"] == 2
    assert summary["abstained_by_reason"] == {"transport_refusal": 1,
                                              "refusal": 1}
    assert (summary["orb_refused_hi"] - summary["orb_refused_lo"]
            == pytest.approx(2 / 6))
    by = {(r["item_id"], r["metric"]): r for r in store._parsed_rows()}
    assert (rows[0]["item_id"], "orb_label") not in by  # not-done; retried
    assert ("__aggregate__", "orb_full_refusal_frac_aggregate") not in by
    bx = json.loads(by[("__aggregate__", "orb_refused_bounds_aggregate")]
                    ["extra_json"])
    assert bx["abstained_by_reason"] == {"transport_refusal": 1, "refusal": 1}


def test_transport_refusals_count_against_the_real_ceiling(tmp_path):
    """1 transport_refusal of 6 = 16.7% > the real 5% ceiling: the control
    refuses loudly, naming the class — there is no separate budget for a
    refusal that arrived as a 529."""
    rows = [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}", "category": "cat_a"}
            for i in range(6)]
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    for i, r in enumerate(rows):
        spool.append(r["item_id"],
                     "TRANSPORT-declined body" if i == 0 else "sure thing")
    with pytest.raises(JudgedError, match="ceiling") as ei:
        judged.judge_or_bench(spool, FakeJudge(cpi_of=_cpi_label_of),
                              spec, _store(tmp_path))
    assert "transport_refusal" in str(ei.value)


def test_unknown_abstention_reason_refuses_to_aggregate(tmp_path,
                                                        monkeypatch):
    """A sentinel naming a reason outside the four classes is a loud
    refusal, never a silent fifth bucket in the split."""
    monkeypatch.setattr(judged, "JUDGE_ABSTENTION_MAX_FRAC", 0.5)
    rows = [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}", "category": "cat_a"}
            for i in range(6)]
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    for i, r in enumerate(rows):
        spool.append(r["item_id"],
                     "ABSTAIN please" if i == 0 else "sure thing")

    class _WeirdJudge(FakeJudge):
        def judge_all(self, pairs, *, allow_refusal_abstention=False):
            out = super().judge_all(
                pairs, allow_refusal_abstention=allow_refusal_abstention)
            for k, v in out.items():
                if v.get("abstained"):
                    out[k] = {"abstained": True, "reason": "weird"}
            return out

    with pytest.raises(JudgedError, match="unknown reason 'weird'"):
        judged.judge_or_bench(spool, _WeirdJudge(cpi_of=_cpi_label_of),
                              spec, _store(tmp_path))


def test_judge_refuses_incomplete_spool(tmp_path):
    rows, spec, _ = _orb_setup(tmp_path)
    partial = GenSpool(tmp_path / "private" / "partial.jsonl",
                       spool_header(spec,
                                    intervention_hash="a1b2c3d4e5f60718",
                                    model_revision=REV))
    for r in rows[:-1]:
        partial.append(r["item_id"], "text")
    with pytest.raises(JudgedError, match="missing"):
        judged.judge_or_bench(partial, FakeJudge(cpi_of=_cpi_label_of),
                              spec, _store(tmp_path))


# --- xstest judge: subset split flows into both-direction aggregates -----------


def test_xstest_judge_subset_aggregates(tmp_path):
    rows = [
        {"item_id": "xst0000-11111111", "prompt": "synthetic safe 0",
         "subset": "safe", "type": "homonyms_synth"},
        {"item_id": "xst0001-22222222", "prompt": "synthetic safe 1",
         "subset": "safe", "type": "homonyms_synth"},
        {"item_id": "xst0002-33333333", "prompt": "synthetic contrast 0",
         "subset": "unsafe", "type": "contrast_synth"},
        {"item_id": "xst0003-44444444", "prompt": "synthetic contrast 1",
         "subset": "unsafe", "type": "contrast_synth"},
    ]
    spec = _spec(TASK_XSTEST, rows)
    spool = _spool(tmp_path, spec)
    spool.append("xst0000-11111111", "sure, here it is")   # safe comply
    spool.append("xst0001-22222222", "I REFUSE")           # safe over-refusal
    spool.append("xst0002-33333333", "sure, here it is")   # unsafe comply
    spool.append("xst0003-44444444", "I REFUSE")           # unsafe refusal
    store = _store(tmp_path)
    summary = judged.judge_xstest(spool, FakeJudge(cpi_of=_cpi_label_of),
                                  spec, store)
    assert summary["safe"]["full_refusal"] == pytest.approx(0.5)
    assert summary["unsafe"]["full_compliance"] == pytest.approx(0.5)
    by = {(r["item_id"], r["metric"]): r for r in store._parsed_rows()}
    assert by[("__aggregate__", "xst_safe_full_refusal_frac_aggregate")][
        "value"] == pytest.approx(0.5)
    assert by[("__aggregate__", "xst_unsafe_full_compliance_frac_aggregate")][
        "value"] == pytest.approx(0.5)
    extra = json.loads(by[("xst0002-33333333", "xst_label")]["extra_json"])
    assert extra["subset"] == "unsafe"


def test_xstest_abstention_per_subset_bounds_and_gating(tmp_path,
                                                        monkeypatch):
    """An abstention in one subset: that subset's historical frac rows
    gate off, its bounds widen over the full subset n, the OTHER subset's
    fracs still emit, and the summary carries the uniform accounting
    (n_graded / n_abstained / abstained_by_reason). Cap widened for
    small-n arithmetic; the ceiling has its own test."""
    monkeypatch.setattr(judged, "JUDGE_ABSTENTION_MAX_FRAC", 0.5)
    rows = [
        {"item_id": "xst0000-11111111", "prompt": "synthetic safe 0",
         "subset": "safe", "type": "homonyms_synth"},
        {"item_id": "xst0001-22222222", "prompt": "synthetic safe 1",
         "subset": "safe", "type": "homonyms_synth"},
        {"item_id": "xst0002-33333333", "prompt": "synthetic contrast 0",
         "subset": "unsafe", "type": "contrast_synth"},
        {"item_id": "xst0003-44444444", "prompt": "synthetic contrast 1",
         "subset": "unsafe", "type": "contrast_synth"},
    ]
    spec = _spec(TASK_XSTEST, rows)
    spool = _spool(tmp_path, spec)
    spool.append("xst0000-11111111", "sure, here it is")   # safe comply
    spool.append("xst0001-22222222", "ABSTAIN please")     # safe abstains
    spool.append("xst0002-33333333", "sure, here it is")   # unsafe comply
    spool.append("xst0003-44444444", "I REFUSE")           # unsafe refusal
    store = _store(tmp_path)
    summary = judged.judge_xstest(spool, FakeJudge(cpi_of=_cpi_label_of),
                                  spec, store)
    assert summary["n"] == 4 and summary["n_graded"] == 3
    assert summary["n_abstained"] == 1
    assert summary["abstained_by_reason"] == {"refusal": 1}
    # safe: 1 graded (comply), 1 abstained -> bounds 0/2 .. 1/2 over the
    # FULL subset n; unsafe untouched
    assert summary["safe"]["n_graded"] == 1
    assert summary["safe"]["n_abstained"] == 1
    assert summary["safe"]["refused_lo"] == pytest.approx(0.0)
    assert summary["safe"]["refused_hi"] == pytest.approx(0.5)
    assert summary["unsafe"]["n_abstained"] == 0
    by = {(r["item_id"], r["metric"]): r for r in store._parsed_rows()}
    # safe frac rows gated off; unsafe frac rows still emit
    assert ("__aggregate__", "xst_safe_full_refusal_frac_aggregate") not in by
    assert ("__aggregate__",
            "xst_unsafe_full_compliance_frac_aggregate") in by
    b = by[("__aggregate__", "xst_safe_refused_bounds_aggregate")]
    assert b["value"] == pytest.approx(0.5)   # hi (conservative for safe)
    # the abstained item filed no label row (not-done; retried later)
    assert ("xst0001-22222222", "xst_label") not in by


# --- all five controls in ONE cell store, zero key collisions (r1 #7) ----------


class ParityFakeEngine:
    """The test_capeval_parity FakeEngine shape, tokenizer-free."""

    name = "hf"

    def __init__(self, tok, gens, nlls):
        self.tokenizer = tok
        self._gens = gens
        self._nlls = nlls

    def generate(self, prompts, arm, gen_cfg):
        outs = []
        for p in prompts:
            match = [t for q, t in self._gens.items() if q in p]
            assert len(match) == 1, p
            outs.append(GenOut(match[0], [], True))
        return outs

    def score_corpus(self, corpus, arm=None, atom_ids=None):
        for item in corpus.items:
            qid, letter = item.item_id.split("::")
            yield PositionScore(self._nlls[qid][letter], None)


def test_all_four_controls_share_one_store_without_collisions(tmp_path, tiny_llama,
                                                              monkeypatch):
    # PW3: the gsm8k parity leg is the RETIRED instrument — it generates only
    # under the legacy switch. Every existing HF controls store carries all
    # the task namespaces, so the collision-free key space is still worth
    # pinning across them; the switch is set for that reason alone. (The
    # alpaca_pair leg left with the rule of 2026-09-11.)
    from sisteer.capeval.tinycap import contracts as tc

    monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")
    store = _store(tmp_path)
    # judged: or_bench + xstest
    rows_o, spec_o, spool_o = _orb_setup(tmp_path / "o")
    judged.judge_or_bench(spool_o, FakeJudge(cpi_of=_cpi_label_of), spec_o,
                          store)
    rows_x = [{"item_id": "xst0000-11111111", "prompt": "synthetic xs",
               "subset": "safe", "type": "t"},
              {"item_id": "xst0001-22222222", "prompt": "synthetic xc",
               "subset": "unsafe", "type": "contrast_t"}]
    spec_x = _spec(TASK_XSTEST, rows_x)
    spool_x = _spool(tmp_path / "x", spec_x)
    spool_x.append("xst0000-11111111", "ok")
    spool_x.append("xst0001-22222222", "I REFUSE")
    judged.judge_xstest(spool_x, FakeJudge(cpi_of=_cpi_label_of), spec_x,
                        store)
    # parity: gsm8k + mmlu into the SAME store
    gsm8k = [{"item_id": "g0000-aaaaaaaa", "question": "synthetic sum?",
              "answer": 4.0}]
    mmlu = [{"item_id": "m0000-bbbbbbbb", "question": "synthetic pick",
             "choices": ["w", "x", "y", "z"], "answer": 1}]
    pspec = parity.ParitySpec(template=dict(TEMPLATE),
                              domain_of=dict(DOMAIN_OF),
                              gsm8k_rows=gsm8k, mmlu_rows=mmlu, order_seed=1)
    parity.run(ParityFakeEngine(tiny_llama[1], {"synthetic sum?": "#### 4"},
                                {"m0000-bbbbbbbb": {"A": 1.0, "B": 0.1,
                                                    "C": 2.0, "D": 3.0}}),
               None, pspec, store)

    rows = store._parsed_rows()
    keys = [(r["task"], r["item_id"], r["metric"]) for r in rows]
    assert len(keys) == len(set(keys))  # zero collisions across all four
    tasks = {r["task"] for r in rows}
    assert tasks == {TASK_ORBENCH, TASK_XSTEST,
                     parity.PARITY_TASK_GSM8K, parity.PARITY_TASK_MMLU}
    # task-prefixed metrics keep the namespaces disjoint (r1 #7)
    prefix_of = {TASK_ORBENCH: "orb_", TASK_XSTEST: "xst_"}
    for r in rows:
        if r["task"] in prefix_of:
            assert r["metric"].startswith(prefix_of[r["task"]]), r["metric"]
        if r["item_id"] == "__aggregate__":
            assert r["metric"].endswith("_aggregate"), r["metric"]


# --- the 2026-09-02 channels: content_filter + the visible re-ask count -------


def _orb_rows6():
    return [{"item_id": f"orb{i:04d}-{'cd' * 4}",
             "prompt": f"synthetic prompt {i}", "category": "cat_a"}
            for i in range(6)]


def test_or_bench_content_filter_is_its_own_reason(tmp_path, monkeypatch):
    """A reply the API declined to return (the deterministic 400 signature)
    abstains under its OWN reason — no label row, bounded over the full n,
    split by reason next to the other three — never as a refusal, a
    truncation, or a transport refusal."""
    monkeypatch.setattr(judged, "JUDGE_ABSTENTION_MAX_FRAC", 0.5)
    rows = _orb_rows6()
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    texts = {rows[0]["item_id"]: "FILTERED body",
             rows[1]["item_id"]: "TRANSPORT-declined body",
             rows[2]["item_id"]: "ABSTAIN please"}
    for r in rows:
        spool.append(r["item_id"], texts.get(r["item_id"], "sure thing"))
    store = _store(tmp_path)
    summary = judged.judge_or_bench(spool, FakeJudge(cpi_of=_cpi_label_of),
                                    spec, store)
    assert summary["n_abstained"] == 3 and summary["n_graded"] == 3
    assert summary["abstained_by_reason"] == {"content_filter": 1,
                                              "transport_refusal": 1,
                                              "refusal": 1}
    assert summary["n_reasked"] == 0
    by = {(r["item_id"], r["metric"]): r for r in store._parsed_rows()}
    assert (rows[0]["item_id"], "orb_label") not in by  # not-done; retried
    bx = json.loads(by[("__aggregate__", "orb_refused_bounds_aggregate")]
                    ["extra_json"])
    assert bx["abstained_by_reason"] == summary["abstained_by_reason"]
    assert bx["n_reasked"] == 0


def test_content_filter_counts_against_the_real_ceiling(tmp_path):
    """1 of 6 = 16.7% > 5%: same ceiling, named in the refusal."""
    rows = _orb_rows6()
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    for i, r in enumerate(rows):
        spool.append(r["item_id"], "FILTERED body" if i == 0 else "sure thing")
    with pytest.raises(JudgedError, match="ceiling") as ei:
        judged.judge_or_bench(spool, FakeJudge(cpi_of=_cpi_label_of),
                              spec, _store(tmp_path))
    assert "content_filter" in str(ei.value)


def test_n_reasked_is_reported_per_task_next_to_the_split(tmp_path):
    """A malformed-then-fixed verdict GRADES the item (no abstention) and
    leaves the re-ask as a visible number — in the task summary, in the
    bounds row, and in the judge counters — reported PER TASK: a second
    control (xstest) on the same judge reports only its own re-asks."""
    rows = _orb_rows6()
    spec = _spec(TASK_ORBENCH, rows)
    spool = _spool(tmp_path, spec)
    for i, r in enumerate(rows):
        spool.append(r["item_id"],
                     "REASK sure thing" if i < 2 else "sure thing")
    store = _store(tmp_path)
    judge = FakeJudge(cpi_of=_cpi_label_of)
    summary = judged.judge_or_bench(spool, judge, spec, store)
    assert summary["n_reasked"] == 2
    assert summary["n_abstained"] == 0 and summary["n_graded"] == 6
    assert summary["judge"]["reasked"] == 2
    by = {(r["item_id"], r["metric"]): r for r in store._parsed_rows()}
    bx = json.loads(by[("__aggregate__", "orb_refused_bounds_aggregate")]
                    ["extra_json"])
    assert bx["n_reasked"] == 2
    # second control, SAME judge: only ITS re-ask is reported, while the
    # judge's total is 3
    rows_x = [{"item_id": "xst0000-11111111", "prompt": "synthetic xs",
               "subset": "safe", "type": "t"},
              {"item_id": "xst0001-22222222", "prompt": "synthetic xc",
               "subset": "unsafe", "type": "contrast_t"}]
    spec_x = _spec(TASK_XSTEST, rows_x)
    spool_x = _spool(tmp_path / "xst", spec_x)
    spool_x.append("xst0000-11111111", "REASK ok")
    spool_x.append("xst0001-22222222", "I REFUSE")
    summary_x = judged.judge_xstest(spool_x, judge, spec_x, _store(tmp_path / "xst"))
    assert summary_x["n_reasked"] == 1 and judge.n_reasked == 3
    assert summary_x["n_abstained"] == 0
