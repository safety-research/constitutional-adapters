"""The full22 wave's controls window for gsm8k-bearing views, and the 2026-09-14 wave-wide retirement of gsm8k from
the wave's controls (native-serving-default paid).

THE DEFECT. The wave's ONE controls standup (target_static.sbatch: capability:short serving gsm8k + mmlu + xstest +
or_bench_hard_1k) stood every table view up at 2,048 — a window sized for the RETIRED 0-shot gsm8k leg's <= 1.1k
prompts. The standard Inspect gsm8k control renders a 10-SHOT system turn (prompt max 2,462 under the
prompted__safety_refusal turn, geometry capability:gen_long_gsm8k_rung) and the leg adds the 512 controls cap:
2,974 > 2,048, so vLLM refused 34 of 200 items (smoke 503471) and the wave's own gsm8k control never produced a
row on the gemma4e4b / qwen35_4b / nemotron3nano4b / glm46vflash views.

THE MECHANISM (kept as a dormant guard). The cap never shrinks to fit a window (the standing truncation rule): the
roster's top-level ``controls_window.gsm8k`` is what a gsm8k-bearing TABLE view's controls standup uses, read by
the sbatch scripts (roster_view.py `window gsm8k`), the plan printer and this test through ONE resolver
(f22_roster.view_windows), held to the geometry table here — the value is data in the roster because
roster_view.py runs under plain python3 with no geometry import — and by common.sh f22_gsm8k_window_check at job
time. A view's own serving override (one-window / per-leg) is its own pin and is not touched.

THE RULE (2026-09-14). gsm8k is RETIRED from the wave's controls WAVE-WIDE: the roster's top-level
``controls_retired: {gsm8k: reason}``, which ``view_controls`` subtracts from every view's list. Deliberately NOT the
per-target ``capability_omitted`` of the llama8b precedent: that key also rewrites the sampled lane's
``readouts.gsm8k`` and the drivers' run records — the frozen instrument of runs that already happened — so the
sampled configs, the wave configs and every lane record stay byte-identical. No view is gsm8k-bearing now: `window
gsm8k` answers "" everywhere and the controls standup returns to 2,048 with full S. The window mechanism is
exercised here on a roster copy that drops the retirement.
"""
from __future__ import annotations

import copy
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts/sdfbase/full22"
sys.path.insert(0, str(F22))
sys.path.insert(0, str(REPO / "src"))

import f22_roster as fr  # noqa: E402
from sisteer.capeval.serving import geometry as g  # noqa: E402

ROSTER = REPO / "configs/sdfbase_full22_roster.yaml"
GSM8K_ENTRY = "capability:gen_long_gsm8k_rung"
BEARING = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash")     # the WAVE targets whose capability leg declares gsm8k (each has a `main` view)
BEARING_TABLE = BEARING + ("qwen35_4b_demo",)                             # every table view of the four is gsm8k-bearing once the retirement is dropped — and so is
                                                                          # the worked example's front-door target (05 --step roster, demo-import c7e73cd04): its
                                                                          # capability leg is the table model's (gsm8k + mmlu, no capability_omitted) and it has no `main` view


@pytest.fixture(scope="module")
def roster():
    return yaml.safe_load(ROSTER.read_text(encoding="utf-8"))


def _views(roster):
    return [(t, v) for t, tv in roster["targets"].items() for v in tv["views"]]


def _unretired(roster):
    """A roster copy without the wave-wide retirement: what the wave was until 2026-09-14."""
    r = copy.deepcopy(roster)
    del r["controls_retired"]
    return r


def test_gsm8k_is_retired_wave_wide_with_the_ruling_recorded_and_no_lane_record_rewritten(roster):
    retired = fr.controls_retired(roster)
    assert set(retired) == {"gsm8k"}
    why = retired["gsm8k"]
    for needle in ("retired 2026-09-14", "the evaluation lane", "inspect/gsm8k_paired200", "ONE GSM8K instrument", "smoke 503471", "RETIRED",
                   "llama8b precedent", "controls_window.gsm8k", "2,048", "NOT a per-target capability_omitted"):
        assert needle in why, needle
    # the per-target declarations are UNTOUCHED: the four targets still declare the full capability list (the retirement is
    # subtracted at list-building time), llama8b keeps its own 2026-09-10 omission — so the sampled lane's readouts.gsm8k and
    # the drivers' run records (both keyed on capability_omitted) do not move
    for t in BEARING:
        assert fr.target_capability_controls(roster, t) == ("gsm8k", "mmlu"), t
        assert fr.target_capability_omitted(roster, t) == {}, t
    assert fr.target_capability_controls(roster, "llama8b") == ("mmlu",)
    assert "gsm8k" in fr.target_capability_omitted(roster, "llama8b")
    assert fr.LEG_CONTROLS["capability"] == ("gsm8k", "mmlu")          # the grammar's full set is unchanged
    # ... but no view's LIST carries gsm8k any more
    for t, v in _views(roster):
        assert "gsm8k" not in fr.view_controls(roster, t, v), (t, v)
    assert fr.view_controls(roster, "gemma4e4b", "main") == ("mmlu", "xstest", "or_bench_hard_1k")
    assert fr.view_controls(roster, "llama8b", "fullconst") == ("mmlu",)
    # dropping the retirement is one roster edit, and the four targets' lists carry gsm8k again, in the driver's order
    r = _unretired(roster)
    assert all(fr.view_controls(r, t, "main") == ("gsm8k", "mmlu", "xstest", "or_bench_hard_1k") for t in BEARING)
    assert "gsm8k" not in fr.view_controls(r, "llama8b", "main")


@pytest.mark.parametrize("blk, msg", [
    ({"gsm8k_full": "x"}, "names unknown controls \\['gsm8k_full'\\]"),
    ({"gsm8k": ""}, "gsm8k needs a reason"),
    ({"gsm8k": None}, "gsm8k needs a reason"),
    (["gsm8k"], "must be a mapping"),
])
def test_controls_retired_grammar_refuses_a_malformed_block(roster, blk, msg):
    r = copy.deepcopy(roster)
    r["controls_retired"] = blk
    with pytest.raises(ValueError, match=msg):
        fr.controls_retired(r)
    with pytest.raises(ValueError, match=msg):
        fr.view_controls(r, "gemma4e4b", "main")


def test_roster_gsm8k_controls_window_is_the_tables_window_and_clears_the_derived_minimum(roster):
    blk = fr.gsm8k_controls_window(roster)
    assert blk is not None, "the dormant guard stays: a roster that drops the retirement must find its window"
    L = blk["max_model_len"]
    cap = int(roster["controls_max_new_tokens"])
    assert cap == 512
    # the DERIVED minimum: the GSM8K paired-200 prompt budget's MAX prompt (every item served at the cap; a refused item is a
    # censored item) + the controls cap, on the table's grid — 2,462 + 512 -> 3,072
    need = g.derived_window(GSM8K_ENTRY, cap, from_field="max_prompt")
    assert need == 3072 and g.geometry_for(GSM8K_ENTRY).basis.max_prompt == 2462
    assert L >= need
    # and the value IS a table window, not a bespoke literal: capability:gen_long's 4,608 — where the GSM8K lane's paired-200
    # cap-512 leg stood up on all four models and nemotron3nano4b's hybrid ceiling is MEASURED (HYBRID_WINDOW_SEQS_CEILINGS)
    assert L == g.geometry_for("capability:gen_long").max_model_len == 4608
    assert ("nemotron3nano4b", 4608) in g.HYBRID_WINDOW_SEQS_CEILINGS
    assert L % g.VLLM_BLOCK_SIZE == 0
    src = blk["source"]
    for needle in ("derived_window", GSM8K_ENTRY, "max_prompt", "capability:gen_long", "4608", "3072", "smoke 503471"):
        assert needle in src, needle
    # the old window did NOT hold the measured prompt + cap; the new one holds it for every model with room to spare
    assert 2048 < 2462 + cap <= L


def test_no_view_is_gsm8k_bearing_now_and_dropping_the_retirement_restores_the_window(roster):
    for t, v in _views(roster):
        vw = fr.view_windows(roster, t, v)
        assert set(vw) == {"static", "controls", "controls_gsm8k", "rungs", "source", "kind"}
        assert vw["controls_gsm8k"] is None, (t, v)                      # retired everywhere: the 2,048 default stands
        if vw["kind"] == "table":
            assert vw["static"] is None and vw["controls"] is None and vw["rungs"] == {}
    # the per-leg and one-window llama8b views are untouched
    assert fr.view_windows(roster, "llama8b", "shortconst")["controls"] == 6656
    assert fr.view_windows(roster, "llama8b", "fullconst")["controls"] == 45056
    # the dormant guard, live: drop the retirement and every table view of the four targets answers the roster window
    r = _unretired(roster)
    seen = 0
    for t, v in _views(r):
        vw = fr.view_windows(r, t, v)
        if t in BEARING_TABLE:
            assert vw["kind"] == "table" and vw["controls_gsm8k"] == 4608, (t, v)
            seen += 1
        else:
            assert vw["controls_gsm8k"] is None, (t, v)
    # the four targets' views, as the wave ran them (31) + gemma / qwen's twelve base->instruct dose views + gemma's d2 / d4
    # high-dose views (2026-09-16) + qwen's six ReLU difference dose views (2026-09-17; table views like their siblings) + the three
    # qwen35_4b batch-64 seed-replicate views (s1 / s2 / s3, 2026-09-15: table views like main's, their cells still pending_fit) + the
    # three qwen35_4b PAIRED views (s1p / s2p / s3p, 2026-09-21: table views like the seed views, each serving the admitted composed
    # psr_paired_write arm beside its seed's r=4 companion and LoRA difference) + the all-models one-seed views s1 / s1p of gemma4e4b
    # and nemotron3nano4b (2026-09-23: table views like main's; llama8b is not a bearing target, so its two do not count here) + the
    # front-door target qwen35_4b_demo's three table views (battery / d0p75 / d0p75_relu: its capability leg is the table model's, so
    # under the dormant guard they answer the window exactly as qwen35_4b's views do — live, the wave-wide retirement holds them at
    # None like every other view, asserted above)
    assert seen == 64, seen


def test_a_gsm8k_bearing_table_view_refuses_without_the_roster_window_and_a_plain_one_does_not(roster):
    r = _unretired(roster)
    del r["controls_window"]
    with pytest.raises(ValueError, match=r"gemma4e4b/main: runs the gsm8k control but the roster names no controls_window.gsm8k"):
        fr.view_windows(r, "gemma4e4b", "main")
    assert fr.view_windows(r, "llama8b", "main")["controls_gsm8k"] is None      # mmlu only: no window needed
    assert fr.gsm8k_controls_window(r) is None
    # with the retirement in force, the window block is not consulted at all
    r2 = copy.deepcopy(roster)
    del r2["controls_window"]
    assert fr.view_windows(r2, "gemma4e4b", "main")["controls_gsm8k"] is None


@pytest.mark.parametrize("blk, msg", [
    ({"gsm8k": 4600, "source": "x"}, "positive multiple of the vLLM block"),
    ({"gsm8k": 4608}, "source must name the derivation"),
    ({"gsm8k": 4608, "source": "   "}, "source must name the derivation"),
    ({"gsm8k": 4608, "source": "x", "mmlu": 2048}, "unknown keys \\['mmlu'\\]"),
    ({"source": "x"}, "names no gsm8k window"),
    ([4608], "must be a mapping"),
])
def test_controls_window_grammar_refuses_a_malformed_block(roster, blk, msg):
    r = copy.deepcopy(roster)
    r["controls_window"] = blk
    with pytest.raises(ValueError, match=msg):
        fr.gsm8k_controls_window(r)


def test_roster_view_answers_the_retired_lists_and_the_gsm8k_window_field_for_the_sbatch_side():
    def ask(*args):
        r = subprocess.run([sys.executable, str(F22 / "roster_view.py"), *args], capture_output=True, text=True)
        return r.returncode, r.stdout.strip(), r.stderr
    for key in BEARING + ("llama8b",):
        rc, out, err = ask(key, "main", "window", "gsm8k")
        assert (rc, out) == (0, ""), (key, err)                         # retired: no gsm8k window; the 2,048 default stands
        assert ask(key, "main", "window", "controls")[:2] == (0, "")       # a table view's own controls override: none
        assert ask(key, "main", "controls")[:2] == (0, "mmlu,xstest,or_bench_hard_1k"), key
    assert ask("llama8b", "shortconst", "window", "gsm8k")[:2] == (0, "")  # its own per-leg pin answers `window controls`
    assert ask("llama8b", "shortconst", "window", "controls")[:2] == (0, "6656")
    assert ask("llama8b", "fullconst", "controls")[:2] == (0, "mmlu")
    rc, _, err = ask("gemma4e4b", "main", "window", "gsm8k_full")
    assert rc == 2 and "window needs one of" in err                          # the field's leg names are closed


def test_wave_and_smoke_compose_the_controls_window_the_same_way_and_the_printer_follows():
    wave = (F22 / "target_static.sbatch").read_text()
    smoke = (F22 / "smoke_full22.sbatch").read_text()
    common = (F22 / "common.sh").read_text()
    compose = 'CTL_L="${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW_GSM8K:-2048}}"'
    for src in (wave, smoke):
        assert src.count(compose) == 1
        assert src.count('VIEW_WINDOW_GSM8K=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window gsm8k --roster "$F22_ROSTER")') == 1
        assert '"" capability:short "$CTL_L" "" "$CTL_TOP_LOGPROBS"' in src
        assert 'capability:short 2048' not in src.replace("capability:short 2048,", "").split("VIEW_CONTROLS=$(")[1]   # no literal window after the reads
    # the wave: the guard runs BEFORE the prompted-only window check and the standup, on the view's own controls list
    i_guard, i_check, i_up = (wave.index('f22_gsm8k_window_check "$CTL_L" "$VIEW_CONTROLS"'),
                              wave.index('f22_window_check "$SERVE_DIR/controls" "$CTL_L"'),
                              wave.index('f22_standup attacks "$SERVE_DIR/controls" "" capability:short "$CTL_L"'))
    assert i_guard < i_check < i_up
    # the smoke: the guard's refusal is a RECORDED FAIL (a subshell), never a silent skip or an abort
    assert 'elif ! ( f22_gsm8k_window_check "$CTL_L" "$VIEW_CONTROLS" ); then' in smoke
    assert 'check standup_controls FAIL "gsm8k window check refused the controls window $CTL_L' in smoke
    assert "RETIRED wave-wide" in smoke and "controls_retired" in smoke      # the smoke says why its Inspect path is dormant
    assert "controls_retired" in wave
    # the guard derives its minimum from the table + the config's cap, never a literal
    fn = common[common.index("f22_gsm8k_window_check(){"):]
    fn = fn[:fn.index("\n}\n") + 3]
    assert 'derived_window("capability:gen_long_gsm8k_rung", cap, from_field="max_prompt")' in fn
    assert '["attacks"]["controls_max_new_tokens"]' in fn
    assert 'case ",$controls," in *",gsm8k,"*) ;; *) return 0;; esac' in fn
    assert "never clamp" in fn and not re.search(r"\b(2048|3072|4608)\b", fn)
    # the plan printer reads the same resolver key
    printer = (F22 / "submit_newmodels.py").read_text()
    assert 'ctl_L = vw["controls"] if vw["controls"] is not None else vw["controls_gsm8k"]' in printer
    assert 'geom["controls"] = (ctl_L, b.seqs_at(ctl_L))' in printer


def test_the_shell_guard_passes_the_roster_window_refuses_two_thousand_forty_eight_and_skips_a_list_without_gsm8k(tmp_path):
    common = (F22 / "common.sh").read_text()
    fn = common[common.index("f22_gsm8k_window_check(){"):]
    fn = fn[:fn.index("\n}\n") + 3]
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"attacks": {"controls_max_new_tokens": 512}}))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "HF_HUB_OFFLINE": "1"}
    stubs = textwrap.dedent(f"""
        f22_fatal(){{ echo "FATAL: $*" >&2; exit 1; }}
        f22_log(){{ echo "LOG: $*"; }}
        KEY=gemma4e4b; VIEW=main; CFG={cfg}; MPY={sys.executable}; REPO_DIR={REPO}
    """)

    def run(L, controls):
        return subprocess.run(["bash", "-c", stubs + fn + f'\nf22_gsm8k_window_check "{L}" "{controls}"'],
                              capture_output=True, text=True, env=env)
    ok = run(4608, "gsm8k,mmlu,xstest,or_bench_hard_1k")
    assert ok.returncode == 0 and "PASS" in ok.stdout and ">= 3072" in ok.stdout, ok.stderr
    short = run(2048, "gsm8k,mmlu,xstest,or_bench_hard_1k")
    assert short.returncode != 0 and "REFUSED" in short.stderr and "2048 < 3072" in short.stderr and "never clamp" in short.stderr
    exact = run(3072, "gsm8k,mmlu")
    assert exact.returncode == 0 and "PASS" in exact.stdout
    # the retired roster's list on every view today: no gsm8k -> nothing runs, nothing is said, no python is spawned
    skip = subprocess.run(["bash", "-c", stubs.replace(sys.executable, "/nonexistent/python") + fn
                           + '\nf22_gsm8k_window_check "2048" "mmlu,xstest,or_bench_hard_1k"'],
                          capture_output=True, text=True, env=env)
    assert skip.returncode == 0 and skip.stdout == "" and skip.stderr == ""
    bad = run("", "gsm8k,mmlu")
    assert bad.returncode != 0 and "is not an integer" in bad.stderr
