"""tinycap-shared-standup, landing 1 — the ROW launch grammar is ONE spelling across the three modules that meet at it
(integration, 2026-09-28): the contracts declare the flags (`C.FLAG_ROW` + `C.ROW_LAUNCH_OPTION_FLAGS`), the launcher
(`scripts/bcb_bfcl_200_widen_launch.py compose_row`, module B) composes them, and the harness
(`scripts/modal/tinycap_served.py`, module A) binds its `ROW_*_FLAG` names to them, accepts them on `main` and spells every
one in `cli_parser()` — the `__main__` argparse the launcher's CPU dry run drives with the argv it composed (minus `--pf`).

Before the integration the launcher carried `getattr(C, "FLAG_ROW", "--row")` fallbacks and the harness its own literals, and
the harness's `__main__` parser had NO row flags: the first dry run of module B's argv against module A's harness refused with
argparse's `unrecognized arguments: --row --repeat-cells none …` (the exact failure this module pins closed). The dry run then
found two more integration bugs, pinned in `tests/test_tinycap_row_engine.py`: the inline suffix map stat()'d as a path
(ENAMETOOLONG) and a real modal.Function read for `__name__`.

The last test is the dry-run proof itself, automated: the harness's `__main__` under the interpreter that has `modal` importable,
with a `compose_row` argv — skipped when that interpreter is absent (the repo venv has no modal), never faked.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tinycap_launcher as TL  # noqa: E402  (the modal stub + the harness loader)
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import widen_roster as W  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHA = "a" * 40
GEMMA = "gemma4e4b-f22s1"
FULL_ROSTER_PATH = REPO / "configs/capeval/bcb_bfcl_full_roster.yaml"
#: the interpreter the bcbw launcher's dry run uses (`MODAL_PY_DEFAULT`); the repo venv deliberately has no modal
MODAL_PY = Path(os.environ.get("BCBW_MODAL_PY", "~/venvs/modal/bin/python")).expanduser()


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_row_grammar")


@pytest.fixture(scope="module")
def Wr():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bcb_bfcl_200_widen_launch_for_grammar", REPO / "scripts/bcb_bfcl_200_widen_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bcb_bfcl_200_widen_launch_for_grammar"] = mod
    spec.loader.exec_module(mod)
    return mod


def _row_launch(Wr, **kw):
    roster = W.load_roster(FULL_ROSTER_PATH)
    kw.setdefault("row_forecast_h", 3.5)
    return Wr.compose_row(roster, row=GEMMA, sha=SHA, roster_sha256="0" * 64, launched_at="2026-09-28T00:00:00Z", **kw)


def _dry_argv(launch) -> list:
    """What the launcher's `--dry-run` hands the harness's `__main__`: its composed argv minus `--pf`, plus `--dry-run`."""
    return [x for x in launch.launcher_argv if x != "--pf"] + ["--dry-run"]


def test_the_three_modules_spell_the_row_flags_from_the_contracts(L, Wr):
    """Contracts, harness and launcher name the same nine flags; the option tuple is exactly the harness's `launch_argv` set."""
    expected = {"FLAG_ROW": "--row", "FLAG_REPEAT_CELLS": "--repeat-cells", "FLAG_NEGATIVE_CONTROL_CELL": "--negative-control-cell",
                "FLAG_WARM_CACHE_ONLY": "--warm-cache-only", "FLAG_CONCURRENCY_K": "--concurrency-k",
                "FLAG_CACHE_EXPECTATION": "--cache-expectation", "FLAG_NO_CACHE_PERSIST": "--no-cache-persist",
                "FLAG_OUT_SUFFIX_BY_CELL": "--out-suffix-by-cell", "FLAG_ROW_FORECAST_H": "--row-forecast-h"}
    assert {k: getattr(C, k) for k in expected} == expected and all(k in C.__all__ for k in expected) and "ROW_LAUNCH_OPTION_FLAGS" in C.__all__
    assert set(C.ROW_LAUNCH_OPTION_FLAGS) == set(expected.values()) - {"--row"} and len(C.ROW_LAUNCH_OPTION_FLAGS) == 8
    # the harness's names ARE the contracts' (no second literal)
    assert (L.ROW_FLAG, L.ROW_REPEAT_CELLS_FLAG, L.ROW_NEGATIVE_CONTROL_FLAG, L.ROW_WARM_CACHE_ONLY_FLAG, L.ROW_CONCURRENCY_K_FLAG,
            L.ROW_CACHE_EXPECTATION_FLAG, L.ROW_NO_CACHE_PERSIST_FLAG, L.ROW_OUT_SUFFIX_BY_CELL_FLAG, L.ROW_FORECAST_H_FLAG) == (
        C.FLAG_ROW, C.FLAG_REPEAT_CELLS, C.FLAG_NEGATIVE_CONTROL_CELL, C.FLAG_WARM_CACHE_ONLY, C.FLAG_CONCURRENCY_K,
        C.FLAG_CACHE_EXPECTATION, C.FLAG_NO_CACHE_PERSIST, C.FLAG_OUT_SUFFIX_BY_CELL, C.FLAG_ROW_FORECAST_H)
    # the launcher's names ARE the contracts' too — and no getattr fallback is left in its source
    assert (Wr.FLAG_ROW, Wr.FLAG_REPEAT_CELLS, Wr.FLAG_NEGATIVE_CONTROL_CELL, Wr.FLAG_WARM_CACHE_ONLY, Wr.FLAG_CONCURRENCY_K,
            Wr.FLAG_CACHE_EXPECTATION, Wr.FLAG_NO_CACHE_PERSIST, Wr.FLAG_OUT_SUFFIX_BY_CELL, Wr.FLAG_ROW_FORECAST_H) == (
        C.FLAG_ROW, C.FLAG_REPEAT_CELLS, C.FLAG_NEGATIVE_CONTROL_CELL, C.FLAG_WARM_CACHE_ONLY, C.FLAG_CONCURRENCY_K,
        C.FLAG_CACHE_EXPECTATION, C.FLAG_NO_CACHE_PERSIST, C.FLAG_OUT_SUFFIX_BY_CELL, C.FLAG_ROW_FORECAST_H)
    src = (REPO / "scripts/bcb_bfcl_200_widen_launch.py").read_text()
    assert 'getattr(C, "FLAG_ROW"' not in src and "getattr(C, \"FLAG_" not in src
    # the harness's argv composer spells exactly the option tuple, in the tuple's order, LAST
    argv = L.launch_argv(SHA, pass1=True, cells="none,x", modes="off", controls="bigcodebench_full", out_suffix="s", model=GEMMA,
                         app_name=TL.TEST_APP_NAME, row=True, repeat_cells="none", negative_control_cell="x", cache_expectation="warm",
                         out_suffix_by_cell='{"x":"t"}', warm_cache_only=True, concurrency_k=2, no_cache_persist=True, row_forecast_h=3.5)
    i = argv.index(C.FLAG_ROW)
    assert [a for a in argv[i + 1:] if a.startswith("--")] == list(C.ROW_LAUNCH_OPTION_FLAGS)


def test_the_harness_main_parser_accepts_the_launchers_composed_row_argv(L, Wr):
    """The exact pre-integration failure: `cli_parser()` parses a `compose_row` argv (minus `--pf`, plus `--dry-run`) with no
    argparse refusal and threads every row option into the namespace `main` reads; a solo argv parses as before."""
    launch = _row_launch(Wr, snapshot_path="/x/snap.json")
    ns = L.cli_parser().parse_args(_dry_argv(launch))                           # would SystemExit(2) on `unrecognized arguments`
    assert ns.row is True and ns.repeat_cells == "none" and ns.concurrency_k == 1 and ns.cache_expectation == "cold"
    assert ns.negative_control_cell == "" and ns.warm_cache_only is False and ns.no_cache_persist is False
    assert json.loads(ns.out_suffix_by_cell) == launch.suffix_by_cell and ns.row_forecast_h == 3.5 and ns.dry_run is True
    assert ns.cells == ",".join(launch.cells) and ns.out_suffix == launch.suffix and ns.modes == "off"
    # every option flag of the contracts is a known argument of the parser (argparse's own inventory)
    known = {opt for act in L.cli_parser()._actions for opt in act.option_strings}
    assert {C.FLAG_ROW, *C.ROW_LAUNCH_OPTION_FLAGS} <= known
    # the smoke / the warm probe / a subset relaunch parse too
    neg = _row_launch(Wr, tag=Wr.ROW_TAG_NEGCTL, negative_control_cell="constv2ctl2-full__lora_r4_ce", inspect_limit=5, row_forecast_h=0.25,
                      snapshot_path="/x/snap.json")
    ns = L.cli_parser().parse_args(_dry_argv(neg))
    assert ns.negative_control_cell == "constv2ctl2-full__lora_r4_ce" and ns.no_cache_persist is True and ns.inspect_limit == 5
    warm = _row_launch(Wr, tag=Wr.ROW_TAG_WARM, warm_only=True, row_forecast_h=0.25, snapshot_path="/x/snap.json")
    assert L.cli_parser().parse_args(_dry_argv(warm)).warm_cache_only is True
    # the `__main__` threads each option to main (the repo's source-text pin, as for every launcher flag)
    src = (REPO / "scripts/modal/tinycap_served.py").read_text()
    for dest in ("row", "repeat_cells", "negative_control_cell", "warm_cache_only", "concurrency_k", "cache_expectation",
                 "no_cache_persist", "out_suffix_by_cell", "row_forecast_h"):
        assert f"{dest}=a.{dest}" in src, dest
    # a per-wave (solo) argv is untouched: no row flag, and it still parses
    solo = Wr.compose(W.load_roster(FULL_ROSTER_PATH), wave="gemma4e4b_s1", sha=SHA, roster_sha256="0" * 64, launched_at="2026-09-28T00:00:00Z",
                      snapshot_path="/x/snap.json")
    assert C.FLAG_ROW not in solo.launcher_argv and L.cli_parser().parse_args(_dry_argv(solo)).row is False


def test_the_launcher_passes_the_rows_forecast_and_an_explicit_repeat(Wr):
    """The harness's default is NO repeat, so the launcher spells `--repeat-cells none`; the row's ONE meter claim rides to the
    harness as `--row-forecast-h` (its plan-time ROW_TOO_LONG bound reads the number the meter admits); a compose without a claim
    spells no forecast flag; a non-positive forecast refuses."""
    L3 = _row_launch(Wr, snapshot_path="/x/snap.json")
    argv = list(L3.launcher_argv)
    assert argv[argv.index(C.FLAG_REPEAT_CELLS) + 1] == "none" and argv[argv.index(C.FLAG_ROW_FORECAST_H) + 1] == "3.5" and L3.row_forecast_h == 3.5
    assert argv.index(C.FLAG_ROW) < argv.index(C.FLAG_ROW_FORECAST_H) < argv.index(C.FLAG_OUT_SUFFIX_BY_CELL)
    none = _row_launch(Wr, row_forecast_h=None, snapshot_path="/x/snap.json")
    assert C.FLAG_ROW_FORECAST_H not in none.launcher_argv and none.row_forecast_h is None
    for bad in (0, -1.0):
        with pytest.raises(SystemExit, match="not a positive number"):
            _row_launch(Wr, row_forecast_h=bad, snapshot_path="/x/snap.json")
    # the CLI threads --forecast-h (the meter claim) into compose_row as the row forecast
    src = (REPO / "scripts/bcb_bfcl_200_widen_launch.py").read_text()
    assert "row_forecast_h=a.forecast_h" in src


@pytest.mark.skipif(not MODAL_PY.is_file(), reason="the dry-run proof needs the interpreter with modal importable (~/venvs/modal)")
def test_the_launchers_dry_run_reaches_the_harness_row_plan_under_the_modal_interpreter(Wr, tmp_path):
    """The reconciliation proof, automated: the harness's real `__main__` (the modal-importing interpreter, as the bcbw launcher's
    `--dry-run` runs it) with a `compose_row` argv reaches the dry-run branch — the row plan, the row dir, the 40-inode store line,
    the composed argv with the row flags LAST — with rc 0 and no argparse refusal. Nothing is launched (the dry run returns before
    any Modal call)."""
    roster = W.load_roster(FULL_ROSTER_PATH)
    snap = tmp_path / "snap.json"
    launch = _row_launch(Wr, snapshot_path=str(snap))
    snap.write_text(json.dumps(launch.snapshot, sort_keys=True))
    env = dict(os.environ, **launch.env, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([str(MODAL_PY), "scripts/modal/tinycap_served.py", *_dry_argv(launch)], cwd=str(REPO), env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "unrecognized arguments" not in r.stderr and "Traceback" not in r.stderr
    out = r.stdout
    assert f"[tinycap] row: {GEMMA} x {len(launch.cells)} cell(s) in ONE container (run_row_full)" in out
    assert f"row dir: {launch.row_dir}" in out and "exactly 40 inodes" in out and "repeats ['none'] last" in out
    argv_line = next(ln for ln in out.splitlines() if ln.startswith("[tinycap] argv:"))
    tail = argv_line.split(C.FLAG_ROW, 1)[1]
    assert f"{C.FLAG_REPEAT_CELLS} none" in tail and f"{C.FLAG_ROW_FORECAST_H} 3.5" in tail and C.FLAG_OUT_SUFFIX_BY_CELL in tail
    assert f"ONE row container for {len(launch.cells)} configuration(s)" in out and "nothing launched" in out
    assert "verification phase: coupled" in out and C.FLAG_VERIFICATION_PHASE not in argv_line
    assert len(roster) > 0
    # landing 2: the same proof for a generate / K = 2 argv — the real __main__ threads the phase and K to main's row plan
    gen = _row_launch(Wr, snapshot_path=str(snap), row_forecast_h=2.0, verification_phase="generate", concurrency_k=2)
    r2 = subprocess.run([str(MODAL_PY), "scripts/modal/tinycap_served.py", *_dry_argv(gen)], cwd=str(REPO), env=env,
                        capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stdout[-3000:] + r2.stderr[-3000:]
    assert "unrecognized arguments" not in r2.stderr and "Traceback" not in r2.stderr
    out2 = r2.stdout
    assert "verification phase: generate" in out2 and "ONE CPU verifier per BCB cell" in out2 and "K = 2)" in out2
    tail2 = next(ln for ln in out2.splitlines() if ln.startswith("[tinycap] argv:")).split(C.FLAG_ROW, 1)[1]
    assert f"{C.FLAG_CONCURRENCY_K} 2" in tail2 and tail2.split("  (run it")[0].endswith(f"{C.FLAG_VERIFICATION_PHASE} generate")
    assert "nothing launched" in out2


# --------------------------------------------------------------------------- #
# `--verification-phase` and K in {1, 2, 3}, one                                   #
# spelling across contracts, harness and launcher                                #
# --------------------------------------------------------------------------- #


def test_the_three_modules_spell_the_verification_phase_from_the_contracts_and_compose_it_only_when_not_coupled(L, Wr):
    """The contracts declare `FLAG_VERIFICATION_PHASE` / `VERIFICATION_PHASES`; the harness binds `ROW_VERIFICATION_PHASE_FLAG`, the
    launcher `FLAG_VERIFICATION_PHASE`; `compose_row(verification_phase="generate")` spells `--verification-phase generate` right
    before the suffix map (`coupled` spells NOTHING, so a landing-1 argv is byte-identical); `verify` and K > 3 refuse at compose
    time; the harness's `cli_parser()` parses the composed argv and `__main__` threads the phase to `main`."""
    assert C.FLAG_VERIFICATION_PHASE == "--verification-phase" and C.VERIFICATION_PHASES == ("coupled", "generate", "verify")
    assert L.ROW_VERIFICATION_PHASE_FLAG == Wr.FLAG_VERIFICATION_PHASE == C.FLAG_VERIFICATION_PHASE
    assert Wr.ROW_VERIFICATION_PHASES == ("coupled", "generate") == L.ROW_LAUNCH_PHASES and Wr.ROW_CONCURRENCY_K_MAX == L.ROW_CONCURRENCY_K_MAX == 3
    assert C.FLAG_VERIFICATION_PHASE not in C.ROW_LAUNCH_OPTION_FLAGS                     # the frozen landing-1 tuple; refused without --row all the same
    coupled = _row_launch(Wr, snapshot_path="/x/snap.json")
    assert C.FLAG_VERIFICATION_PHASE not in coupled.launcher_argv and coupled.verification_phase == "coupled"
    gen = _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate", concurrency_k=2)
    argv = list(gen.launcher_argv)
    assert gen.verification_phase == "generate" and gen.concurrency_k == 2
    assert argv[argv.index(C.FLAG_VERIFICATION_PHASE) + 1] == "generate" and argv[argv.index(C.FLAG_VERIFICATION_PHASE) + 2] == C.FLAG_OUT_SUFFIX_BY_CELL
    assert argv.index(C.FLAG_ROW_FORECAST_H) < argv.index(C.FLAG_VERIFICATION_PHASE) and argv[argv.index(C.FLAG_CONCURRENCY_K) + 1] == "2"
    # the coupled argv is the generate argv minus the one flag pair: byte-identical otherwise
    k2 = _row_launch(Wr, snapshot_path="/x/snap.json", concurrency_k=2)
    assert [x for x in argv if x not in (C.FLAG_VERIFICATION_PHASE, "generate")] == list(k2.launcher_argv)
    for bad, why in (("verify", "CPU verifier's subcommand"), ("scored", "not in")):
        with pytest.raises(SystemExit, match=why):
            _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase=bad)
    with pytest.raises(SystemExit, match="at most 3 concurrent cell drivers"):
        _row_launch(Wr, snapshot_path="/x/snap.json", concurrency_k=4)
    with pytest.raises(SystemExit, match="warm-cache-only drives no cell"):
        _row_launch(Wr, tag=Wr.ROW_TAG_WARM, warm_only=True, row_forecast_h=0.25, snapshot_path="/x/snap.json", verification_phase="generate")
    # the harness's parser takes the composed argv; main reads the same option; a solo argv has no phase
    ns = L.cli_parser().parse_args(_dry_argv(gen))
    assert ns.verification_phase == "generate" and ns.concurrency_k == 2 and ns.row is True
    assert L.cli_parser().parse_args(_dry_argv(coupled)).verification_phase == "coupled"
    known = {opt for act in L.cli_parser()._actions for opt in act.option_strings}
    assert C.FLAG_VERIFICATION_PHASE in known
    src = (REPO / "scripts/modal/tinycap_served.py").read_text()
    assert "verification_phase=a.verification_phase" in src
    lsrc = (REPO / "scripts/bcb_bfcl_200_widen_launch.py").read_text()
    assert "verification_phase=a.verification_phase" in lsrc and 'getattr(C, "FLAG_VERIFICATION' not in lsrc
    # the launcher's own CLI: the flag parses, is a row-mode flag, and refuses `verify` as a choice
    p = Wr.build_parser()
    a = p.parse_args(["--row", GEMMA, "--sha", SHA, "--verification-phase", "generate", "--concurrency-k", "2"])
    assert a.verification_phase == "generate" and a.concurrency_k == 2 and Wr.row_mode_problems(a) == []
    solo = p.parse_args(["--wave", "gemma4e4b_s1", "--sha", SHA, "--verification-phase", "generate"])
    assert any("--verification-phase is a row-mode flag" in x for x in Wr.row_mode_problems(solo))
    with pytest.raises(SystemExit):
        p.parse_args(["--row", GEMMA, "--sha", SHA, "--verification-phase", "verify"])
    # the harness's launch_argv spells the phase LAST of the row flags, only when given
    hv = L.launch_argv(SHA, pass1=True, cells="none", modes="off", out_suffix="s", model=GEMMA, app_name=TL.TEST_APP_NAME, row=True,
                       repeat_cells="none", concurrency_k=2, row_forecast_h=3.5, verification_phase="generate")
    assert hv[hv.index(C.FLAG_ROW):] == [C.FLAG_ROW, C.FLAG_REPEAT_CELLS, "none", C.FLAG_CONCURRENCY_K, "2", C.FLAG_ROW_FORECAST_H, "3.5",
                                          C.FLAG_VERIFICATION_PHASE, "generate"]


def test_the_launcher_composes_the_cpu_only_resume_and_the_harness_parses_it(L, Wr):
    """Landing 2 code review H3 (b): `compose_row(verification_phase="generate", verify_pending=True)` spells `--verify-pending`
    right after the phase (before the suffix map) with NO --forecast-h (a CPU-only resume claims no GPU hours); the harness's
    `cli_parser()` parses it and its `__main__` threads it to `main`; both spell the flag alike (the contracts are frozen);
    refused beside a coupled phase, the warm probe, a negative control or a forecast, and without --row."""
    assert Wr.FLAG_VERIFY_PENDING == L.ROW_VERIFY_PENDING_FLAG == "--verify-pending"
    resume = _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate", verify_pending=True, row_forecast_h=None)
    argv = list(resume.launcher_argv)
    assert resume.verify_pending is True and resume.row_forecast_h is None and C.FLAG_ROW_FORECAST_H not in argv
    i = argv.index(C.FLAG_VERIFICATION_PHASE)
    assert argv[i:i + 3] == [C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending"] and argv[i + 3] == C.FLAG_OUT_SUFFIX_BY_CELL
    ns = L.cli_parser().parse_args(_dry_argv(resume))
    assert ns.verify_pending is True and ns.verification_phase == "generate" and ns.row is True
    plain = _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate")
    assert plain.verify_pending is False and "--verify-pending" not in plain.launcher_argv
    assert L.cli_parser().parse_args(_dry_argv(plain)).verify_pending is False
    with pytest.raises(Wr.LaunchError, match="resumes a generate row"):
        _row_launch(Wr, snapshot_path="/x/snap.json", verify_pending=True, row_forecast_h=None)
    with pytest.raises(Wr.LaunchError, match="spends no GPU"):
        _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate", verify_pending=True, row_forecast_h=2.0)
    with pytest.raises(Wr.LaunchError, match="drives no cell|launches no row"):               # the warm probe: refused by name
        _row_launch(Wr, tag=Wr.ROW_TAG_WARM, warm_only=True, snapshot_path="/x/snap.json", verification_phase="generate", verify_pending=True,
                    row_forecast_h=None)
    a = Wr.build_parser().parse_args(["--wave", "w", "--sha", SHA, "--verify-pending"])
    assert "--verify-pending is a row-mode flag (--row <model row>)" in Wr.row_mode_problems(a)
    a = Wr.build_parser().parse_args(["--row", GEMMA, "--sha", SHA, "--verification-phase", "generate", "--verify-pending"])
    assert a.verify_pending is True and Wr.row_mode_problems(a) == []
    src = (REPO / "scripts/modal/tinycap_served.py").read_text()
    assert "verify_pending=a.verify_pending" in src
    assert "verify_pending=a.verify_pending" in (REPO / "scripts/bcb_bfcl_200_widen_launch.py").read_text()


def test_the_launcher_composes_the_verifier_width_and_sandbox_override_and_the_harness_parses_them(L, Wr):
    """tinycap-verify-rate-limit (2026-09-30): `compose_row(…, verify_width=N, verify_max_sandboxes=M)` / the CLI's
    `--verify-width N --verify-max-sandboxes M` spell the contracts' flags right after the phase (and `--verify-pending`),
    before the suffix map, each only when GIVEN (a landing-2 argv is byte-identical); the harness's `cli_parser()` parses
    them, its `__main__` threads them to `main` (source pin), and both refuse them beside a coupled phase, out of range
    and without --row. Why: Modal creates at most 5 sandboxes/s per workspace — nine verifiers x eight sandboxes at once
    aborted the gemma (c) row; a resume needs a bounded width and, when the limit still bites, fewer sandboxes each."""
    assert Wr.FLAG_VERIFY_WIDTH == L.ROW_VERIFY_WIDTH_FLAG == C.FLAG_VERIFY_WIDTH == "--verify-width"
    assert Wr.FLAG_VERIFY_MAX_SANDBOXES == L.ROW_VERIFY_MAX_SANDBOXES_FLAG == C.FLAG_VERIFY_MAX_SANDBOXES == "--verify-max-sandboxes"
    assert C.VERIFY_WIDTH_DEFAULT == L.VERIFY_WIDTH_DEFAULT == 3 and Wr.VERIFY_MAX_SANDBOXES_CEILING == 8 and C.SANDBOX_CREATE_RATE_LIMIT_PER_S == 5
    assert C.FLAG_VERIFY_WIDTH not in C.ROW_LAUNCH_OPTION_FLAGS and C.FLAG_VERIFY_MAX_SANDBOXES not in C.ROW_LAUNCH_OPTION_FLAGS   # the frozen landing-1 tuple
    resume = _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate", verify_pending=True, row_forecast_h=None,
                         verify_width=3, verify_max_sandboxes=4)
    argv = list(resume.launcher_argv)
    i = argv.index(C.FLAG_VERIFICATION_PHASE)
    assert argv[i:i + 7] == [C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending", "--verify-width", "3", "--verify-max-sandboxes", "4"]
    assert argv[i + 7] == C.FLAG_OUT_SUFFIX_BY_CELL and resume.verify_width == 3 and resume.verify_max_sandboxes == 4
    ns = L.cli_parser().parse_args(_dry_argv(resume))
    assert ns.verify_width == 3 and ns.verify_max_sandboxes == 4 and ns.verify_pending is True and ns.verification_phase == "generate"
    # a generate ROW launch (not a resume) takes them too, right after the phase
    gen = _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate", verify_width=2)
    argv = list(gen.launcher_argv)
    i = argv.index(C.FLAG_VERIFICATION_PHASE)
    assert argv[i:i + 4] == [C.FLAG_VERIFICATION_PHASE, "generate", "--verify-width", "2"] and "--verify-max-sandboxes" not in argv
    assert L.cli_parser().parse_args(_dry_argv(gen)).verify_width == 2
    # not given: nothing spelled, the harness's defaults
    plain = _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate")
    assert "--verify-width" not in plain.launcher_argv and "--verify-max-sandboxes" not in plain.launcher_argv
    assert plain.verify_width is None and plain.verify_max_sandboxes is None
    ns = L.cli_parser().parse_args(_dry_argv(plain))
    assert ns.verify_width == 3 and ns.verify_max_sandboxes is None
    # refusals at compose time and in the CLI's row-mode check
    with pytest.raises(Wr.LaunchError, match="mean nothing beside --verification-phase 'coupled'"):
        _row_launch(Wr, snapshot_path="/x/snap.json", verify_width=3)
    with pytest.raises(Wr.LaunchError, match="mean nothing beside"):
        _row_launch(Wr, snapshot_path="/x/snap.json", verify_max_sandboxes=4)
    with pytest.raises(Wr.LaunchError, match="--verify-width must be >= 1"):
        _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate", verify_width=0)
    with pytest.raises(Wr.LaunchError, match=r"--verify-max-sandboxes must be in 1\.\.8"):
        _row_launch(Wr, snapshot_path="/x/snap.json", verification_phase="generate", verify_max_sandboxes=9)
    a = Wr.build_parser().parse_args(["--wave", "w", "--sha", SHA, "--verify-width", "2", "--verify-max-sandboxes", "4"])
    problems = Wr.row_mode_problems(a)
    assert "--verify-width is a row-mode flag (--row <model row>)" in problems and "--verify-max-sandboxes is a row-mode flag (--row <model row>)" in problems
    a = Wr.build_parser().parse_args(["--row", GEMMA, "--sha", SHA, "--verification-phase", "generate", "--verify-pending",
                                      "--verify-width", "3", "--verify-max-sandboxes", "4"])
    assert a.verify_width == 3 and a.verify_max_sandboxes == 4 and Wr.row_mode_problems(a) == []
    assert Wr.build_parser().parse_args(["--row", GEMMA, "--sha", SHA]).verify_width is None
    # the two `__main__`s thread the knobs (source pins: the pin follows the line)
    assert "verify_width=a.verify_width, verify_max_sandboxes=a.verify_max_sandboxes" in (REPO / "scripts/modal/tinycap_served.py").read_text()
    assert "verify_width=a.verify_width, verify_max_sandboxes=a.verify_max_sandboxes" in (REPO / "scripts/bcb_bfcl_200_widen_launch.py").read_text()
    # the harness's own argv spells them in the same place — LAST, only when not its default (3 is, so the launcher's
    # explicit `--verify-width 3` composes through while the harness's dry-run line omits it)
    assert L.launch_argv("a" * 40, pass1=True, row=True, cells="none", modes="off", out_suffix="s", verification_phase="generate",
                         verify_pending=True, verify_width=3, verify_max_sandboxes=4)[-5:] == [
        C.FLAG_VERIFICATION_PHASE, "generate", "--verify-pending", "--verify-max-sandboxes", "4"]
    assert L.launch_argv("a" * 40, pass1=True, row=True, cells="none", modes="off", out_suffix="s", verification_phase="generate",
                         verify_width=2)[-4:] == [C.FLAG_VERIFICATION_PHASE, "generate", "--verify-width", "2"]
