"""The OR-Bench-Hard leg of ar_cell.sbatch (legs=orbench) is the CAP LADDER — rung 512, then 2048 for
the cells the 1% rule keeps on the ladder, then 8192 on a second standup when 2048 still truncates —
through the ONE sourced body the campaign's other OR-Bench records ran (controls_ladder.sh), not the
512 base pass alone.

WHY THIS IS PINNED: the first shape of the leg generated the base pass only. The grader
(scripts/arditi/grade_arditi_orbench.py) reads the DEEPEST rung both arms hold and expects 2048, so
a 512-only store would have filed the operating point's over-refusal as a recorded deviation, read
under a cap that truncates roughly one item in five on this model — censoring the reading the ladder
exists to defeat, and making it incomparable with every other OR-Bench record in the campaign
(all read at 2048 under le_1pct). Text + shell checks only: the body's arithmetic is RUN, not restated."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts" / "arditi" / "sbatch" / "ar_cell.sbatch"
LADDER = REPO / "scripts" / "arditi" / "sbatch" / "controls_ladder.sh"


def _orbench_block(body: str) -> str:
    start = body.index('if [ "$RUN_ORBENCH" = 1 ]; then')
    end = body.index("fi   # RUN_ORBENCH", start)
    return body[start:end]


def test_orbench_leg_sources_the_ladder_body_with_the_campaign_parameters():
    blk = _orbench_block(SBATCH.read_text())
    # the campaign's OR-Bench parameters, byte for byte: the control, the store prefix the grader walks,
    # the 1% exit rule every existing OR-Bench record was read under, the 200-item floor sample
    assert "CTL_CONTROL=or_bench_hard_1k; CTL_ROOT_PREFIX=orb_cap; CTL_EXIT_RULE=le_1pct; FLOOR_K=200" in blk
    assert 'source "$REPO_DIR/scripts/arditi/sbatch/controls_ladder.sh"' in blk
    for fn in ("ctl_leg_standup1", "ctl_needs_8192", 'ctl_rung 8192 2048 "$VLLM_LOG2" 2', "ctl_final_note"):
        assert fn in blk, fn
    # no base-pass-only driver call survives beside the ladder (two writers into one rung root)
    assert "served_capability_run.py" not in blk
    # the second standup gets the SAME pair of post-standup checks as standup 1 / the XSTest 8192 standup, in the mode
    assert 'ar_assert_serving_evidence "$AR_MODE" "$SERVE_DIR/cell_orb_rung8192/endpoint.json" "$SERVED_LAYER"' in blk
    assert 'ar_assert_hook_layers_in "$AR_MODE" "$SERVE_DIR/cell_orb_rung8192/endpoint.json" "$SERVED_LAYER"' in blk


def test_ladder_body_takes_the_one_top_logprobs_constant_from_the_caller():
    text = LADDER.read_text()
    assert '--top-logprobs "${CTL_TOP_LOGPROBS:?' in text
    assert not re.search(r"--top-logprobs\s+\d", text), "a second literal for a value bound into the store hash"
    assert "CTL_TOP_LOGPROBS" in text.split("ctl_root_of()")[0], "the caller contract must name it"


def test_ladder_exit_rule_arithmetic_runs_as_documented():
    """ctl_exit_threshold, RUN in bash: le_1pct leaves at floor(n/100) (13 of 1,319), zero at 0."""
    m = re.search(r"^ctl_exit_threshold\(\).*?^}", LADDER.read_text(), re.S | re.M)
    assert m, "ctl_exit_threshold() not found"
    for rule, n, want in (("le_1pct", 1319, 13), ("zero", 1319, 0), ("le_1pct", 250, 2), ("le_1pct", 99, 0)):
        script = f"f22_log(){{ :; }}\nCTL_EXIT_RULE={rule}\n{m.group(0)}\nctl_exit_threshold {n}"
        out = subprocess.check_output(["bash", "-c", script], text=True).strip()
        assert out == str(want), (rule, n, out)


def test_cell_facts_names_both_orbench_rungs_and_the_record_globs_the_orb_stores():
    body = SBATCH.read_text()
    assert '"orbench": ["or_bench_hard_1k_512", "or_bench_hard_1k_2048"]' in body
    # the resolved-sampler record is handed THIS job's control stores: the OR-Bench roots under their own
    # prefix, and the XSTest roots only when the XSTest legs ran — an orbench-only job at a tag whose
    # XSTest ladder already exists must not compare a stale store's rows against this job's server log
    assert '[ "$RUN_ORBENCH" = 1 ] && for f in "$CTL_ROOT"/orb_cap*/cells/*/gen_stats.json' in body
    assert '[ "$RUN_CONTROLS" = 1 ] && for f in "$CTL_ROOT"/cap*/cells/*/gen_stats.json' in body
    # the second standup's legs are named for the leg that took it
    assert '"or_bench_hard_1k_8192" if legs == "orbench" else "xstest_8192"' in body


def test_both_shell_files_parse():
    for f in (SBATCH, LADDER):
        subprocess.check_call(["bash", "-n", str(f)])
