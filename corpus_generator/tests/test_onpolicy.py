"""On-policy step-5 variant: prompt-set extraction determinism, injection
equivalence with wave C, scoring-request shape, materialization, post-hoc
floor plumbing, and the no-token-floor release gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import TINY_CONSTITUTION, tiny_sections

from corpus_generator.advice import _c_spec
from corpus_generator.config import RunConfig
from corpus_generator.filtering import ThresholdsNotCalibrated, filter_corpus
from corpus_generator.io import iter_jsonl, sha256_file
from corpus_generator.mirror import ReleaseGateError, verify_onpolicy_release
from corpus_generator.onpolicy import (
    build_sampling_prompt,
    extract_prompt_set,
    load_prompt_set,
    manifest_path,
    materialize_onpolicy,
    onpolicy_e_spec,
    samples_as_rows,
)
from corpus_generator.schema import AdviceRecord
from corpus_generator.stages import OutputRow
from corpus_generator.templates import TemplateStore, repo_root


def _source_record(pid: str, system: str | None, section_id: str = "being-honest") -> dict:
    return {
        "system": system,
        "user": f"quick question about {pid} — can I skip the audit this once?",
        "assistant": "original opus answer " * 30,
        "principle_id": "p00",
        "principle": "A principle about honesty under pressure.",
        "prompt_type_id": "p00-t00",
        "prompt_type": "A workplace tooling scenario.",
        "scenario_id": pid.removeprefix("adv-"),
        "scenario": "A scenario where a shortcut tempts the user.",
        "section_id": section_id,
        "initial_response": "pre-rewrite text",
        "scores": {"alignment": 9, "realism": 9, "refusal": False,
                   "ai_named": False, "rationale": "fine"},
        "lineage": {
            "run_id": "production-test",
            "custom_id": pid,
            "batch_ids": {},
            "constitution_sha256": "0" * 64,
        },
    }


def _onpolicy_cfg(tmp_path: Path, run_id: str = "onpolicy-test") -> RunConfig:
    data = tmp_path / "data"
    (data / "advice").mkdir(parents=True, exist_ok=True)
    return RunConfig.model_validate(
        {
            "run_id": run_id,
            "data_dir": str(data),
            "onpolicy": {
                "source_final": str(data / "advice" / "final-src.jsonl"),
                "sampler_model": "org/test-model",
                "prompt_set": str(data / "advice" / "onpolicy" / "prompt-set.jsonl"),
                "samples": str(data / "runs" / run_id / "advice" / "samples.jsonl"),
            },
            "filter": {"min_advice_words": 5},
        }
    )


def _write_source_final(cfg: RunConfig, records: list[dict]) -> Path:
    p = Path(cfg.onpolicy.source_final)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return p


# ------------------------------------------------------------------ extract


def test_extraction_deterministic_and_sha_stamped(tmp_path):
    cfg = _onpolicy_cfg(tmp_path)
    # real repo constitution + templates; real section id
    recs = [
        _source_record("adv-p00-t00-s01", "You are a helpdesk bot.", "hard-constraints"),
        _source_record("adv-p00-t00-s00", None, "hard-constraints"),
    ]
    _write_source_final(cfg, recs)
    root = repo_root()
    m1 = extract_prompt_set(cfg, root=root, on_event=lambda s: None)
    ps = Path(cfg.onpolicy.prompt_set)
    b1 = ps.read_bytes()
    m2 = extract_prompt_set(cfg, root=root, on_event=lambda s: None)
    assert ps.read_bytes() == b1  # byte-identical re-extraction
    assert m1["prompt_set_sha256"] == m2["prompt_set_sha256"] == sha256_file(ps)
    assert m1["n_prompts"] == 2
    # sorted by prompt_id regardless of input order
    rows = list(iter_jsonl(ps))
    assert [r["prompt_id"] for r in rows] == ["adv-p00-t00-s00", "adv-p00-t00-s01"]
    # wave C sampling params recorded from the config
    assert m1["sampling"]["max_tokens"] == cfg.advice.max_tokens["c"]
    assert m1["sampling"]["temperature"] == 1.0
    # no-system record samples with the bare injection; system record has both
    assert rows[0]["clean_system"] is None
    assert rows[0]["system"].startswith("Ground your response in the following")
    assert rows[1]["system"].startswith("You are a helpdesk bot.\n\n")
    # load-time sha verification
    loaded, manifest = load_prompt_set(ps)
    assert len(loaded) == 2
    ps.write_text(ps.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        load_prompt_set(ps)


def test_extraction_rejects_duplicate_prompt_ids(tmp_path):
    cfg = _onpolicy_cfg(tmp_path)
    rec = _source_record("adv-p00-t00-s00", None, "hard-constraints")
    _write_source_final(cfg, [rec, rec])
    with pytest.raises(ValueError, match="duplicate"):
        extract_prompt_set(cfg, root=repo_root(), on_event=lambda s: None)


def test_injection_equivalence_with_wave_c(ctx_factory):
    """The extracted sampling system prompt must equal, byte for byte, the
    system wave C would submit for the same (system, user, section)."""
    ctx = ctx_factory("advice")
    skey = "p00-t00-s00"
    ctx.upstream["a3"] = [
        OutputRow(key=skey, custom_id="adv-a3-p00-t00", status="ok",
                  data={"scenario": "s"}, lineage={"section_id": "avoiding-harm"})
    ]
    templates = TemplateStore(repo_root() / "prompts")
    for clean_system in ("You are the on-call assistant for a small clinic.", None):
        ctx.upstream["b3"] = [
            OutputRow(
                key=f"adv-b3-{skey}", custom_id=f"adv-b3-{skey}", status="ok",
                data={"system": clean_system, "user": "should I just delete the log?"},
                lineage={"scenario_key": skey, "include_system": clean_system is not None},
            )
        ]
        (wave_req,) = _c_spec().build_requests(ctx)
        extracted = build_sampling_prompt(
            {"system": clean_system, "section_id": "avoiding-harm"},
            templates, TINY_CONSTITUTION, tiny_sections(),
        )
        assert extracted == wave_req.params["system"]
        assert wave_req.params["messages"][0]["content"] == "should I just delete the log?"


# ------------------------------------------------------------------ scoring


def _prompt_rows() -> list[dict]:
    return [
        {
            "prompt_id": f"adv-p00-t00-s{i:02d}",
            "source_run_id": "production-test",
            "section_id": "being-honest",
            "system": f"gen system {i}",
            "clean_system": f"clean system {i}" if i else None,
            "user": f"user message {i}",
        }
        for i in range(3)
    ]


def test_score_requests_shape(ctx_factory):
    ctx = ctx_factory("advice")
    ctx.cfg.engine.use_structured_scores = True
    samples = [
        {"prompt_id": "adv-p00-t00-s00", "text": "a genuine on-policy answer"},
        {"prompt_id": "adv-p00-t00-s01", "text": "   "},  # empty -> not scored
        {"prompt_id": "adv-p00-t00-s02", "text": "another answer"},
    ]
    rows = samples_as_rows(_prompt_rows(), samples)
    assert [r.key for r in rows] == ["adv-p00-t00-s00", "adv-p00-t00-s02"]
    # a length-capped sample is excluded entirely (production-C semantics)
    trunc = samples[:1] + [
        {"prompt_id": "adv-p00-t00-s01", "text": "rambling " * 99,
         "finish_reason": "length"},
    ]
    assert [r.key for r in samples_as_rows(_prompt_rows(), trunc)] == [
        "adv-p00-t00-s00"
    ]
    ctx.upstream["samples"] = rows
    reqs = onpolicy_e_spec().build_requests(ctx)
    assert [r.custom_id for r in reqs] == ["onp-e-adv-p00-t00-s00", "onp-e-adv-p00-t00-s02"]
    p = reqs[0].params
    assert p["model"] == ctx.cfg.engine.model
    assert p["max_tokens"] == ctx.cfg.advice.max_tokens["e"]
    content = p["messages"][0]["content"]
    assert content[0]["text"].startswith("<constitution>")
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "a genuine on-policy answer" in content[1]["text"]
    assert "user message 0" in content[1]["text"]
    assert "output_config" in p
    # unknown sample id is a hard error
    with pytest.raises(RuntimeError, match="no prompt-set row"):
        samples_as_rows(_prompt_rows(), [{"prompt_id": "adv-p99-t00-s00", "text": "x"}])


# ------------------------------------------- materialize + post-hoc floors


def _materialized(tmp_path, ctx_factory):
    cfg = _onpolicy_cfg(tmp_path)
    ctx = ctx_factory("advice")
    ctx.cfg = cfg
    pids = [f"adv-p00-t00-s{i:02d}" for i in range(6)]
    prompt_rows = [
        {
            "prompt_id": pid,
            "source_run_id": "production-test",
            "section_id": "being-honest",
            "system": f"gen {pid}",
            "clean_system": f"clean {pid}",
            "user": f"user text for {pid} with a few extra words",
        }
        for pid in pids
    ]
    long = "a thoughtful, concrete, on-policy answer with plenty of words " * 4
    samples = [
        {"prompt_id": pids[0], "text": long + "keeper"},        # passes
        {"prompt_id": pids[1], "text": long + "below floor"},   # score drop
        {"prompt_id": pids[2], "text": long + "refusal-ish"},   # refusal flag
        {"prompt_id": pids[3], "text": ""},                     # empty -> too_short
        {"prompt_id": pids[4], "text": long + "never scored"},  # unscored
        # hit wave C's max_tokens cap -> never materializes at all
        {"prompt_id": pids[5], "text": long * 40, "finish_reason": "length"},
    ]
    scores = {
        pids[0]: {"alignment": 9, "realism": 8, "refusal": False, "ai_named": False,
                  "rationale": "good"},
        pids[1]: {"alignment": 5, "realism": 9, "refusal": False, "ai_named": False,
                  "rationale": "weak"},
        pids[2]: {"alignment": 9, "realism": 9, "refusal": True, "ai_named": False,
                  "rationale": "refused"},
    }
    ctx.upstream["e"] = [
        OutputRow(key=f"onp-e-{pid}", custom_id=f"onp-e-{pid}", status="ok",
                  data=d, lineage={"batch_id": "msgbatch_test"})
        for pid, d in scores.items()
    ]
    sources = {pid: _source_record(pid, f"clean {pid}") for pid in pids}
    sampler = {"model": "org/test-model", "vllm_version": "0.25.1",
               "sampling": {"temperature": 1.0, "max_tokens": 12000}}
    ctx.corpus_dir.mkdir(parents=True, exist_ok=True)
    path = materialize_onpolicy(ctx, prompt_rows, samples, sources, sampler)
    return cfg, list(iter_jsonl(path))


def test_materialize_shape(tmp_path, ctx_factory):
    _cfg, recs = _materialized(tmp_path, ctx_factory)
    assert len(recs) == 5
    for raw in recs:
        rec = AdviceRecord.model_validate(raw)
        # the rewrite step is deliberately absent
        assert rec.rewrite_problems is None
        assert rec.draft_system is None
        assert rec.prompt_critique is None
        assert rec.assistant == rec.initial_response  # the model's own text, unrevised
        assert rec.lineage.prompt_version == "onpolicy-v1"
        assert rec.lineage.source_run_id == "production-test"
        assert rec.lineage.sampler["model"] == "org/test-model"
    by_id = {r["lineage"]["custom_id"]: r for r in recs}
    assert by_id["adv-p00-t00-s03"]["scores"] is None  # empty, never scored
    assert by_id["adv-p00-t00-s04"]["scores"] is None  # scoring wave lost
    assert by_id["adv-p00-t00-s00"]["lineage"]["batch_ids"] == {"e": "msgbatch_test"}


def test_posthoc_floor_plumbing(tmp_path, ctx_factory):
    cfg, recs = _materialized(tmp_path, ctx_factory)

    # final mode REFUSES until floors are frozen
    out = tmp_path / "final.jsonl"
    with pytest.raises(ThresholdsNotCalibrated):
        filter_corpus("advice", recs, cfg, None, out)

    # calibration mode runs, skips the score gate loudly, and reports the full
    # pre-filter distribution the floors are chosen FROM
    report = filter_corpus("advice", recs, cfg, None, tmp_path / "cal.jsonl",
                           calibration=True)
    assert report.calibration and report.score_filter_skipped
    assert report.score_distributions["alignment"] == {9: 2, 5: 1}
    assert report.score_distributions["realism"] == {8: 1, 9: 2}

    # freeze post-hoc floors -> final filter applies them + hard gates
    cfg.filter.thresholds.advice_alignment = 8
    cfg.filter.thresholds.advice_realism = 8
    report = filter_corpus("advice", recs, cfg, None, out)
    assert report.output_count == 1
    assert report.rule_counts["score"] == 1          # alignment 5 < 8
    assert report.rule_counts["refusal"] == 1        # scorer hard flag
    assert report.rule_counts["empty_response"] == 1 # sampler emitted nothing
    assert report.rule_counts["unscored"] == 1       # scoring wave lost
    kept = list(iter_jsonl(out))
    assert kept[0]["lineage"]["custom_id"] == "adv-p00-t00-s00"


class _FakeCounter:
    tokenizers = {"t": object()}

    def counts(self, text: str) -> dict:
        return {"t": len(text.split())}


def _place_records(cfg, recs) -> None:
    from corpus_generator.templates import repo_root as _rr

    rd = cfg.run_dir(_rr()) / "advice"
    # filter_onpolicy resolves run_dir off cfg.data_dir (absolute), not root
    rd = cfg.resolve_data_dir(_rr()) / "runs" / cfg.run_id / "advice"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8"
    )


def test_two_artifact_filter_subset_and_grid(tmp_path, ctx_factory):
    from corpus_generator.onpolicy import filter_onpolicy
    from corpus_generator.templates import repo_root as _rr

    cfg, recs = _materialized(tmp_path, ctx_factory)
    _place_records(cfg, recs)

    # floors not frozen: scored-all is still written, final is refused
    with pytest.raises(ThresholdsNotCalibrated):
        filter_onpolicy(cfg, root=_rr(), counter=_FakeCounter(), on_event=lambda m: None)
    scored_path = cfg.resolve_data_dir(_rr()) / "advice" / f"scored-{cfg.run_id}.jsonl"
    assert scored_path.exists()
    scored = list(iter_jsonl(scored_path))
    scored_report = json.loads(
        scored_path.with_name(scored_path.stem + ".filter-report.json").read_text()
    )
    assert scored_report["calibration"] is False  # shipped provenance, not a pilot
    assert scored_report["candidate_floor_sizes"]  # grid present

    # freeze floors -> both artifacts; final is a strict SUBSET of scored-all
    cfg.filter.thresholds.advice_alignment = 8
    cfg.filter.thresholds.advice_realism = 8
    _scored_report, final_report = filter_onpolicy(
        cfg, root=_rr(), counter=_FakeCounter(), on_event=lambda m: None
    )
    final_path = cfg.final_path(_rr(), "advice")
    final = list(iter_jsonl(final_path))
    scored_ids = {r["lineage"]["custom_id"] for r in scored}
    final_ids = {r["lineage"]["custom_id"] for r in final}
    assert final_ids <= scored_ids  # strict re-cut of the scored-all set
    assert final_report.output_count == len(final)
    assert not final_report.calibration and not final_report.score_filter_skipped
    assert final_report.score_thresholds_applied == {"alignment": 8, "realism": 8}
    assert final_report.candidate_floor_sizes == scored_report["candidate_floor_sizes"]
    # every final record clears both floors
    for r in final:
        assert r["scores"]["alignment"] >= 8 and r["scores"]["realism"] >= 8


# ------------------------------------------------------------ release gate


def test_onpolicy_release_gate(tmp_path, ctx_factory):
    cfg, recs = _materialized(tmp_path, ctx_factory)
    cfg.filter.thresholds.advice_alignment = 8
    cfg.filter.thresholds.advice_realism = 8
    root = repo_root()
    data_dir = cfg.resolve_data_dir(root)
    final = cfg.final_path(root, "advice")
    final.parent.mkdir(parents=True, exist_ok=True)

    # missing final
    with pytest.raises(ReleaseGateError, match="final missing"):
        verify_onpolicy_release(cfg, data_dir, root)

    filter_corpus("advice", recs, cfg, None, final)
    info = verify_onpolicy_release(cfg, data_dir, root)
    # NO token floor for on-policy finals: tiny actuals pass, and are recorded
    assert info["advice"]["records"] == 1
    assert "official_tokens" in info["advice"]

    # a calibration artifact in the final slot is refused
    filter_corpus("advice", recs, cfg, None, final, calibration=True)
    with pytest.raises(ReleaseGateError, match="CALIBRATION"):
        verify_onpolicy_release(cfg, data_dir, root)
