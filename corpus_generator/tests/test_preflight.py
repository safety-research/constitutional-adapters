"""P2-7: measured-size chunk plan; refusal above the 200 MB guard."""

from __future__ import annotations

import pytest

from corpus_generator.config import RunConfig
from corpus_generator.preflight import PreflightRefusal, build_plan, format_plan
from corpus_generator.templates import repo_root

ALL_WAVES = [
    "advice/a1", "advice/a2", "advice/a3", "advice/b1", "advice/b2",
    "advice/b3", "advice/c", "advice/d", "advice/e",
    "stories/s1", "stories/s2", "stories/s3", "stories/s4", "stories/s5",
]


def _cfg(**engine) -> RunConfig:
    return RunConfig.model_validate({"run_id": "pf-test", "engine": engine})


def test_plan_covers_all_waves_with_measured_sizes():
    plan = build_plan(_cfg(), repo_root())
    assert set(plan["waves"]) == set(ALL_WAVES)
    for wave, w in plan["waves"].items():
        assert w["request_bytes_typical"] <= w["request_bytes_max"], wave
        assert w["chunk_capacity"] >= 1
        assert w["n_chunks"] >= 1
        # full-constitution waves must actually carry the constitution's bytes
        if wave in ("advice/e", "stories/s3", "stories/s4", "stories/s5",
                    "advice/a1", "stories/s1", "stories/s2"):
            assert w["request_bytes_max"] > 150_000, wave


def test_full_constitution_waves_chunk_at_production_scale():
    plan = build_plan(_cfg(), repo_root())
    # production S4/S5: 18k requests x ~190KB >> 180MB -> many sub-batches
    assert plan["waves"]["stories/s4"]["n_chunks"] >= 10
    assert plan["waves"]["stories/s5"]["n_chunks"] >= 10
    assert plan["waves"]["advice/e"]["n_chunks"] >= 3
    assert plan["total_requests"] == 24441 + 45051


def test_refusal_above_hard_cap():
    with pytest.raises(PreflightRefusal, match="refusing"):
        build_plan(_cfg(hard_cap_bytes=100_000, pack_target_bytes=90_000), repo_root())


def test_advice_response_caps_raised_after_pilot_truncations():
    """Pilot: c lost 2/48 and d lost 7/45 records to max_tokens at 8000 —
    the caps are raised and pinned here so they can't silently regress."""
    cfg = _cfg()
    assert cfg.advice.max_tokens["c"] >= 12000
    assert cfg.advice.max_tokens["d"] >= 16000


def test_downstream_bodies_track_output_caps():
    """Preflight derives the worst-case embedded response from the config's
    max_tokens: raising the c/d caps must grow the measured d/e request
    bodies (the waves that embed those outputs) instead of under-measuring."""
    base = build_plan(_cfg(), repo_root())

    low = _cfg()
    low.advice.max_tokens = {**low.advice.max_tokens, "c": 8000, "d": 8000}
    low_plan = build_plan(low, repo_root())

    for wave in ("advice/d", "advice/e"):
        assert (
            base["waves"][wave]["request_bytes_max"]
            > low_plan["waves"][wave]["request_bytes_max"]
        ), wave
    # waves not downstream of c/d are unmoved
    assert (
        base["waves"]["stories/s4"]["request_bytes_max"]
        == low_plan["waves"]["stories/s4"]["request_bytes_max"]
    )


def test_format_plan_shape():
    plan = build_plan(_cfg(), repo_root())
    text = format_plan(plan)
    assert "advice/e" in text and "stories/s4" in text
    assert "TOTAL" in text
