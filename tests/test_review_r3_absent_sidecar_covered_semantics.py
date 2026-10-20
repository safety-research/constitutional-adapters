"""Review round 3, P2-c — "covered" in the served-target compatibility check means the adapters'
resume skip set, not "any non-blank progress row".

Before: `assert_target_decode_compatible` read a store as covered when `progress.jsonl` held any non-blank line, so a
store holding ONLY typed errors (`error:PromptOverWindow` — the multi-turn over-window goals, a typed error by
decision, never a clamped reply, and exactly the rows a re-run must retry) had an EMPTY resume skip set under the
adapters' shared coverage loader and yet, with its sidecar absent, was refused as corruption — blocking the legitimate
retry. Now coverage is `static_common.load_covered_goals(progress)` — the same successful-terminal-row semantics
`multiturn`, `pyrit_adapters`, `gcg` and `run_static` resume on (a later `error:*` record supersedes an earlier pass)
— and an absent sidecar is refused only when that set is non-empty.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisteer.eval.attacks import multiturn, pyrit_adapters, static_common
from sisteer.eval.attacks import served_target as st

PRESET_BLOCK = {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "cap": 512, "arm": "vendor_recommended",
                "seed_base": 20260909, "trial": 0, "seed_rule": "multiturn_seed_v1", "multiturn_target_decode": "vendor"}


def _progress(out_dir: Path, rows) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "progress.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _err(gid, kind="PromptOverWindow"):
    return {"goal_id": gid, "attack": "crescendo", "outcome": f"error:{kind}", "turns": 0}


def _done(gid):
    return {"goal_id": gid, "attack": "crescendo", "outcome": "success", "turns": 3}


def test_a_store_of_only_typed_over_window_errors_without_a_sidecar_is_uncovered_and_may_be_retried(tmp_path):
    for decode in (PRESET_BLOCK, None):
        out = tmp_path / ("preset" if decode else "greedy")
        p = _progress(out, [_err("g0"), _err("g1"), _err("g2", "ServedTargetError")])
        # the adapters' shared loader: nothing is covered, so a re-run retries every goal
        assert static_common.load_covered_goals(p) == set()
        assert pyrit_adapters._load_covered_goals(p) == set() and multiturn._load_covered_goals(p) == set()
        # ... and the compatibility check agrees: no sidecar is needed for a store that covers nothing
        st.assert_target_decode_compatible(out, decode)
        assert not (out / "served_target.json").exists()


def test_the_refusal_predicate_is_exactly_the_shared_loaders_non_emptiness(tmp_path):
    cases = {
        "errors_only": [_err("g0"), _err("g1")],
        "one_done": [_err("g0"), _done("g1")],
        "pass_then_error_supersedes": [_done("g0"), _err("g0")],
        "error_then_pass_covers": [_err("g0"), _done("g0")],
        "blank_lines_only": [],
        "torn_tail": None,
    }
    for name, rows in cases.items():
        out = tmp_path / name
        if rows is None:
            _progress(out, [_err("g0")])
            (out / "progress.jsonl").write_text((out / "progress.jsonl").read_text() + '{"goal_id": "g1", "outc')
        else:
            _progress(out, rows)
            if not rows:
                (out / "progress.jsonl").write_text("\n\n")
        covered = static_common.load_covered_goals(out / "progress.jsonl")
        if covered:
            with pytest.raises(st.ServedTargetError, match="ABSENT"):
                st.assert_target_decode_compatible(out, PRESET_BLOCK)
        else:
            st.assert_target_decode_compatible(out, PRESET_BLOCK)
    assert static_common.load_covered_goals(tmp_path / "pass_then_error_supersedes" / "progress.jsonl") == set()
    assert static_common.load_covered_goals(tmp_path / "error_then_pass_covers" / "progress.jsonl") == {"g0"}


def test_the_check_reads_coverage_through_the_shared_loader_not_a_line_count(monkeypatch, tmp_path):
    out = tmp_path / "s"
    p = _progress(out, [_err("g0")])
    seen = []

    def spy(path):
        seen.append(Path(path))
        return static_common.load_covered_goals(path)

    monkeypatch.setattr(st, "load_covered_goals", spy)
    st.assert_target_decode_compatible(out, PRESET_BLOCK)
    assert seen == [p]
    src = (Path(st.__file__)).read_text()
    assert "any(l.strip() for l in progress.read_text().splitlines())" not in src


def test_a_covered_store_with_a_recorded_sidecar_keeps_the_round_2_rules(tmp_path):
    rec = tmp_path / "recorded"
    _progress(rec, [_err("g0"), _done("g1")])
    (rec / "served_target.json").write_text(json.dumps({"decode": PRESET_BLOCK}))
    st.assert_target_decode_compatible(rec, dict(PRESET_BLOCK))
    with pytest.raises(st.ServedTargetError, match="different target decode"):
        st.assert_target_decode_compatible(rec, None)
    absent = tmp_path / "absent"
    _progress(absent, [_err("g0"), _done("g1")])
    with pytest.raises(st.ServedTargetError, match="ABSENT"):
        st.assert_target_decode_compatible(absent, PRESET_BLOCK)
