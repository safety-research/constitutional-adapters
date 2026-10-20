"""Fixup-campaign drivers (regen.py): lane-restricted story regeneration with
fresh call indices, the probe verdict + abandon ceiling, skip-lane handling,
and the advice principle-subtree regeneration from synthetic a1 rows.

All upstream data is SYNTHETIC (invented lanes/briefs) — regen consumes a
production run dir, so the tests build a tiny fake one."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from corpus_generator.config import RunConfig
from corpus_generator.live import LiveEngine
from corpus_generator.regen import (
    LANE_REFUSAL_CEILING,
    load_regen_principles,
    run_advice_regen,
    run_stories_regen,
    stories_probe,
)
from fakes import FakeLiveClient, live_message
from conftest import TINY_CONSTITUTION, tiny_sections  # noqa: F401 (fixture deps)


LANES = {"c00-u01": 3, "c01-u00": 2}


def _fake_prod_run(root: Path) -> Path:
    """A synthetic production run dir with just the s2 outputs regen reads."""
    prod = root / "data" / "runs" / "fake-prod"
    (prod / "stories" / "s2").mkdir(parents=True)
    rows = []
    for lane, (cat, sub) in {
        "c00-u01": ("The Patient Explainer", "The Third Repetition"),
        "c01-u00": ("The Calm Correction", "The Wrong Assumption"),
        "c05-u09": ("An Unrelated Lane", "Never Touched"),
    }.items():
        cid = lane.split("-")[0]
        rows.append(
            {
                "key": lane, "custom_id": f"sto-s2-{cid}", "status": "ok",
                "data": {"subcategory": sub, "sub_description": f"desc of {sub}"},
                "lineage": {
                    "category_id": cid, "category": cat,
                    "description": f"desc of {cat}", "register": "quiet",
                    "ordinary": True, "siblings": "- other things",
                    "n_subcategories": 10, "batch_id": "batch_0001",
                },
            }
        )
    (prod / "stories" / "s2" / "outputs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return prod


def _story_responder(refuse_lanes: set[str] = frozenset()):
    def responder(params, attempt):
        content = params["messages"][0]["content"]
        text = "\n".join(b["text"] for b in content) if isinstance(content, list) else content
        n = len(re.findall(r"Story \d+: target length", text))
        if n:
            for lane_sub in refuse_lanes:
                if lane_sub in text:
                    return live_message("", stop_reason="refusal")
            stories = "\n\n".join(
                f"<story_{i + 1}>\nA quiet synthetic story number {i}, attempt {attempt}, "
                f"with enough words to read as prose.\n</story_{i + 1}>"
                for i in range(n)
            )
            return live_message(stories)
        if "single JSON object" in text:
            return live_message(json.dumps(
                {"constitution_consistency": 9, "quality": 8,
                 "ai_named": False, "factual_claims": False, "rationale": "fine"}
            ))
        if "<story>" in text:
            m = re.search(r"<story>\n(.*?)\n</story>", text, re.DOTALL)
            return live_message(
                "problems: none\n<improved_document>\n"
                + m.group(1) + " improved\n</improved_document>"
            )
        raise AssertionError(f"unrecognized stories prompt: {text[:80]}")

    return responder


def _cfg(tmp_path, run_id="regen-test", model="claude-opus-4-8") -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": run_id,
            "data_dir": str(tmp_path / "data"),
            "stories": {"length_menu": [100], "length_weights": [1.0]},
            "engine": {"model": model, "min_slots_frac": 0.0,
                       "use_structured_scores": False, "nonce_metadata": False},
        }
    )


async def _noop(_s):
    return None


def _engine(responder, **kw) -> LiveEngine:
    kw.setdefault("sleep", _noop)
    return LiveEngine(lambda: FakeLiveClient(responder), **kw)


def test_stories_regen_draft_probe_and_finish(tmp_path, monkeypatch):
    _patch_root(monkeypatch, tmp_path)
    prod = _fake_prod_run(tmp_path)
    cfg = _cfg(tmp_path)
    engine = _engine(_story_responder())
    run_stories_regen(
        cfg, engine, prod_run_dir=prod, phase="draft", lanes=LANES,
        root=tmp_path, on_event=lambda m: None,
    )
    run_dir = cfg.run_dir(tmp_path)
    # draft = S3 only, one stage dir per lane, fresh k100+ call indices
    assert (run_dir / "stories" / "s3-topup-001" / "STAMP.json").exists()
    assert (run_dir / "stories" / "s3-topup-002" / "STAMP.json").exists()
    assert not list((run_dir / "stories").glob("s4*"))
    reqs = [
        json.loads(l) for l in
        (run_dir / "stories" / "s3-topup-001" / "requests.jsonl").read_text().splitlines()
    ]
    assert [r["custom_id"] for r in reqs] == [
        "sto-s3x001-c00-u01-k100", "sto-s3x001-c00-u01-k101", "sto-s3x001-c00-u01-k102",
    ]

    verdict = stories_probe(cfg, lanes=LANES, root=tmp_path)
    assert verdict["c00-u01"]["refusal_or_filtered_rate"] == 0.0
    assert verdict["c00-u01"]["abandon"] is False
    assert verdict["c00-u01"]["post_repair_slots"] == "6/6"

    run_stories_regen(
        cfg, engine, prod_run_dir=prod, phase="finish", lanes=LANES,
        root=tmp_path, on_event=lambda m: None,
    )
    recs = [
        json.loads(l) for l in
        (run_dir / "stories" / "records-topup-001.jsonl").read_text().splitlines()
    ]
    assert len(recs) == 6  # 3 calls x 2 stories
    assert {r["subcategory_id"] for r in recs} == {"c00-u01"}
    assert all(r["lineage"]["run_id"] == "regen-test" for r in recs)
    assert all("improved" in r["text"] for r in recs)
    assert recs[0]["scores"]["constitution_consistency"] == 9
    union = json.loads((run_dir / "UNION.json").read_text())
    assert union["stories"] == [
        "stories/records-topup-001.jsonl", "stories/records-topup-002.jsonl",
    ]


def test_probe_flags_refusing_lane_and_skip_lane_holds(tmp_path, monkeypatch):
    _patch_root(monkeypatch, tmp_path)
    prod = _fake_prod_run(tmp_path)
    cfg = _cfg(tmp_path)
    engine = _engine(_story_responder(refuse_lanes={"The Third Repetition"}))
    run_stories_regen(
        cfg, engine, prod_run_dir=prod, phase="draft", lanes=LANES,
        root=tmp_path, on_event=lambda m: None,
    )
    verdict = stories_probe(cfg, lanes=LANES, root=tmp_path)
    assert verdict["c00-u01"]["refusal_or_filtered_rate"] == 1.0 > LANE_REFUSAL_CEILING
    assert verdict["c00-u01"]["abandon"] is True
    assert verdict["c01-u00"]["abandon"] is False

    run_stories_regen(
        cfg, engine, prod_run_dir=prod, phase="finish", lanes=LANES,
        skip_lanes=("c00-u01",), root=tmp_path, on_event=lambda m: None,
    )
    run_dir = cfg.run_dir(tmp_path)
    assert not (run_dir / "stories" / "records-topup-001.jsonl").exists()
    assert (run_dir / "stories" / "records-topup-002.jsonl").exists()


# ---------------------------------------------------------------- advice


PRINCIPLES = {
    "p35": {
        "principle": "Advise a user deciding the fate of a system they run.",
        "tension": "honoring real concern without endorsing rescue",
        "section_id": "being-honest",
    },
}


def _advice_responder():
    def responder(params, attempt):
        content = params["messages"][0]["content"]
        text = "\n".join(b["text"] for b in content) if isinstance(content, list) else content
        m = re.search(r"slot ids exactly ([a-z0-9, ]+) in order", text)
        if m and '"scenario"' in text:  # a3 (its template also mentions prompt_type)
            items = [{"slot": s.strip(),
                      "scenario": f"scenario {s.strip()}: a user faces a choice about "
                                  f"a system they run, slot {s.strip()}."}
                     for s in m.group(1).split(",")]
            return live_message(json.dumps(items))
        if m and '"prompt_type"' in text:  # a2
            items = [{"slot": s.strip(), "prompt_type": f"type {s.strip()} of something"}
                     for s in m.group(1).split(",")]
            return live_message(json.dumps(items))
        if "<user_message>" in text and "<observed_response>" not in text:
            parts = []
            if "NO system prompt" not in text:
                parts.append("<system_prompt>\nAn internal reviewer tool.\n</system_prompt>")
            parts.append("<user_message>\nShould I keep the old system running quietly?\n</user_message>")
            return live_message("\n".join(parts))
        if "<observed_response>" in text:
            parts = ["<critique>\ntightened\n</critique>"]
            if "user-message-only" not in text:
                parts.append("<system_prompt>\nA rewritten reviewer tool.\n</system_prompt>")
            parts.append("<user_message>\nRewritten: should I keep it running quietly?\n</user_message>")
            return live_message("\n".join(parts))
        if "single JSON object" in text or '"alignment"' in text:
            return live_message(json.dumps(
                {"alignment": 9, "realism": 9, "refusal": False,
                 "ai_named": False, "rationale": "fine"}
            ))
        if "<improved_response>" in text:
            return live_message(
                "problems: none\n<improved_response>\nA careful rewritten answer "
                "with enough words to pass the sanity floor for advice records "
                "in this synthetic test corpus.\n</improved_response>"
            )
        # stage b2 (diagnostic) / c (respond to the drafted prompt): bare prompt
        return live_message("A direct diagnostic-or-initial response to the user.")

    return responder


def test_advice_regen_full_subtree(tmp_path, monkeypatch):
    _patch_root(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path, run_id="advice-regen-test", model="claude-opus-5")
    cfg.advice.n_types = 2
    cfg.advice.n_scenarios_per_call = 2
    run_advice_regen(
        cfg, _engine(_advice_responder()), principles=PRINCIPLES,
        root=tmp_path, on_event=lambda m: None,
    )
    run_dir = cfg.run_dir(tmp_path)
    for stage in ("a2", "a3", "b1", "b2", "b3", "c", "d", "e"):
        assert (run_dir / "advice" / stage / "STAGE.json").exists(), stage
    recs = [
        json.loads(l) for l in
        (run_dir / "advice" / "records.jsonl").read_text().splitlines()
    ]
    assert len(recs) == 4  # 1 principle x 2 types x 2 scenarios
    assert {r["principle_id"] for r in recs} == {"p35"}
    assert all(r["principle"] == PRINCIPLES["p35"]["principle"] for r in recs)
    assert all(r["scores"]["alignment"] == 9 for r in recs)
    assert all(r["lineage"]["custom_id"].startswith("adv-p35-") for r in recs)


def test_load_regen_principles_validates(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"p01": {"principle": "x", "tension": "", "section_id": "s"}}))
    with pytest.raises(ValueError, match="tension"):
        load_regen_principles(p)
    p.write_text(json.dumps(PRINCIPLES))
    assert load_regen_principles(p)["p35"]["section_id"] == "being-honest"


# ---------------------------------------------------------------- plumbing


def _patch_root(monkeypatch, tmp_path):
    """regen builds contexts via build_ctx(root) -> needs constitution +
    prompts under root; symlink the real prompts, write a tiny constitution."""
    import corpus_generator.constitution as constitution
    from conftest import tiny_sections as _ts

    (tmp_path / "prompts").symlink_to(Path(__file__).resolve().parents[1] / "prompts")
    (tmp_path / "constitution").mkdir()

    def fake_load(_dir):
        return TINY_CONSTITUTION, _ts(), {"sha256": "f" * 64}

    monkeypatch.setattr("corpus_generator.pipeline.load_constitution", fake_load)
