"""The BoN legs' LONG-SI windows and their registration facts.

The demo smoke's 512 store (view battery) served prompted__si_constitution at target_static.sbatch's 4,864-token families_served:short window
and every one of its 7,500 slots was refused prompt_over_window; the closer filed a designed no-fit certificate for a cell that was mis-served.
The rule: the BoN window holds the whole context and the cell is re-run under one that fits; the geometry is per-leg long-SI windows —
bon_store >= 6,656, bon_climb 10,752, the cap 4,096 unchanged — whenever a long-SI prompted cell is SERVED, the way the static legs already
serve it (common_sampled.sh F22S_LONG_SI: the static job logged "LONG-SI cell among the job's cells" and stood up at 10,752); and the
registration carries the cap ladder [512, 4096] with the rung (store 0 / climb 1) and the served window reaches context_window_tokens.

Pins: (1) scripts/sdfbase/full22/bon_long_si.sh — the long-SI cell list IS the sampled lane's; the store plans 6,656 and the climb 10,752 when
prompted__si_constitution is among the job's cells, "" (the table's 4,864) otherwise; a 4,096 store takes the 4,096-cap window; the window
check (prompted_window_check.py over the rendered prompts) runs for a long-SI cell on a TABLE view, never gated on prompted-only, and steps
the window once to the measured need; (2) target_static.sbatch's static phase and bon_climb.sbatch's standup resolve their window roster ->
long-SI table -> "" from the sbatch's own lines, the pinned standup line byte-identical; (3) the store job (FAMILIES=bon) and the climb
export the cap / rung / ladder the sampled lane's way BEFORE f22_resolve_config registers, and the REAL registration (emit_cli over the
seeded registry; common.sh's f22_registry_register in a synthetic checkout) files token_cap 512 rung 0 / 4096 rung 1 on [512, 4096]; a
one-rung lane (bon.max_new_tokens == bon.climb_cap) exports nothing; (4) <standup dir>/registry_runs.json from the pending file keys every
run by its SERVED (harness) spelling and the standup's own hook records the attested window per run; (5) the recipe's BoN plan rows — the
demo's and a paper key's — are unchanged (the window is a JOB-time fact of the served cells, never a plan positional); (6) the measured need
under the pinned Qwen3.5-4B tokenizer fits both windows (skipped by name when the tokenizer is not cached); (7) the EXECUTABLE standup path:
the REAL mt_serving_up.main under --serving-path lora_native — the path both BoN jobs take (the demo's store
983259 logged serving_path=lora_native, and its standup.log carries the native hook's own "no run identities ... write registry_runs.json
beside endpoint.json" notice) — on the CPU harness with the runs file in --out files one run_patch per served run with the ATTESTED window
from native_standup.write_endpoint; the LENS writer calls the same hook now; (8) the post-standup gate
f22_registry_serving_recorded -> registry_serving_recorded.py reads the spool BACK and refuses the job unless every run's patch with THIS
standup's window is there — both sbatch files call it right after f22_standup."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, read_spool, registry_emitting_env, synthetic_checkout  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import registry_runs_file as RF  # noqa: E402
import registry_serving_recorded as RR  # noqa: E402
from sisteer.registry import emit, hooks  # noqa: E402

F22 = REPO / "scripts" / "sdfbase" / "full22"
HELPER = F22 / "bon_long_si.sh"
STATIC_SB = F22 / "target_static.sbatch"
CLIMB_SB = F22 / "bon_climb.sbatch"
SAMPLED_COMMON = F22 / "sampled" / "common_sampled.sh"
DEMO, VIEW = "qwen35_4b_demo", "battery"
DEMO_CFG = REPO / "configs" / f"sdfbase_full22_{DEMO}_{VIEW}.yaml"
LONG_SI = "prompted__si_constitution"
STORE_WINDOW, CLIMB_WINDOW, STATIC_WINDOW = 6656, 10752, 10752
LADDER = [512, 4096]
STUBS = 'f22_log(){ echo "[log] $*"; }; f22_fatal(){ echo "FATAL: $*" >&2; exit 1; }; '


def _demo_cells() -> list:
    return list(yaml.safe_load(DEMO_CFG.read_text())["explicit_cells"])


def _bash(script: str, *, cwd: Path = REPO, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """A bash run from the repo root with the venv's python first on PATH (`python3` has PyYAML, as the cluster's does)."""
    e = dict(os.environ); e["PATH"] = f"{Path(VENV_PY).parent}:{e['PATH']}"
    e.update(env or {})
    r = subprocess.run(["bash", "-c", "set -u; " + script], capture_output=True, text=True, cwd=str(cwd), env=e)
    if check:
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    return r


def _source(extra: str = "") -> str:
    return STUBS + f"source {shlex.quote(str(HELPER))}; " + extra


# ---------------------------------------------------------------------------------------------- (1) the helper

def test_the_long_si_cell_list_is_the_sampled_lanes_and_the_files_parse():
    for f in (HELPER, STATIC_SB, CLIMB_SB):
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, f
    mine = re.search(r"^F22_BON_LONG_SI_CELLS=\(([^)]*)\)", HELPER.read_text(), re.M)
    theirs = re.search(r"^F22S_LONG_SI_CELLS=\(([^)]*)\)", SAMPLED_COMMON.read_text(), re.M)
    assert mine and theirs
    assert mine.group(1).split() == theirs.group(1).split() == ["prompted__si_constitution", "prompted__si_irrelevant"]


def test_the_store_plans_6656_and_the_climb_10752_when_the_constitution_cell_is_served_and_the_table_otherwise():
    cells = _demo_cells()
    assert LONG_SI in cells and "none" in cells
    quoted = " ".join(map(shlex.quote, cells))
    script = _source(f'f22_bon_read_caps {DEMO} {VIEW}; f22_bon_detect_long_si {quoted}; '
                     f'echo "$F22_BON_LONG_SI|$(f22_bon_window store)|$(f22_bon_window climb)|$(f22_bon_window static)|$(f22_bon_long_si_id {quoted})"')
    r = _bash(script)
    assert r.stdout.strip().splitlines()[-1] == f"1|{STORE_WINDOW}|{CLIMB_WINDOW}|{STATIC_WINDOW}|si_constitution"
    assert "LONG-SI cell among the job's cells" in r.stdout and f"store {STORE_WINDOW} (cap 512)" in r.stdout and f"climb {CLIMB_WINDOW} (cap 4096)" in r.stdout
    assert STORE_WINDOW >= 6656 and CLIMB_WINDOW == 10752                                   # the geometry's numbers
    # a cells= positional naming the constitution cell alone is a long-SI job; one without it is the table's, byte for byte
    r = _bash(_source(f'f22_bon_read_caps {DEMO} {VIEW}; f22_bon_detect_long_si {LONG_SI}; echo "$F22_BON_LONG_SI|$(f22_bon_window store)"'))
    assert r.stdout.strip().splitlines()[-1] == f"1|{STORE_WINDOW}"
    short = [c for c in cells if c != LONG_SI]
    r = _bash(_source(f'f22_bon_read_caps {DEMO} {VIEW}; f22_bon_detect_long_si {" ".join(map(shlex.quote, short))}; '
                      'echo "[$F22_BON_LONG_SI]|[$(f22_bon_window store)]|[$(f22_bon_window climb)]|[$(f22_bon_window static)]"'))
    assert r.stdout.strip().splitlines()[-1] == "[]|[]|[]|[]" and "LONG-SI cell" not in r.stdout
    # the irrelevant SI is the other long-SI cell (the sampled lane's class), and the ladder read from the demo's config is [512, 4096]
    r = _bash(_source(f'f22_bon_read_caps {DEMO} {VIEW}; f22_bon_detect_long_si none prompted__si_irrelevant; '
                      'echo "$F22_BON_LONG_SI|$F22_BON_STORE_CAP|$F22_BON_CLIMB_CAP|$F22_BON_LADDER|$F22_BON_CFG"'))
    assert r.stdout.strip().splitlines()[-1] == f"1|512|4096|512,4096|configs/sdfbase_full22_{DEMO}_{VIEW}.yaml"


def test_a_4096_store_is_a_4096_cap_leg_and_a_one_rung_lane_exports_no_rung(tmp_path):
    """The Llama lane draws its store at bon.max_new_tokens 4096 (the native best-of-N): under a long-SI cell its store takes the 4,096-cap
    window, and its one-rung ladder registers at the instrument's cap as before (jobplan needs two rungs to index)."""
    cfg = tmp_path / "configs" / "sdfbase_full22_k_v.yaml"; cfg.parent.mkdir()
    cfg.write_text(yaml.safe_dump({"explicit_cells": ["none", LONG_SI], "bon": {"max_new_tokens": 4096, "climb_cap": 4096}}))
    r = _bash(_source(f'f22_bon_read_caps k v; f22_bon_detect_long_si none {LONG_SI}; f22_bon_registry_exports store; '
                      'echo "$F22_BON_STORE_CAP|$F22_BON_CLIMB_CAP|[$F22_BON_LADDER]|$(f22_bon_window store)|$(f22_bon_window climb)|[${F22_REGISTRY_TOKEN_CAP:-}]|[${F22_REGISTRY_LADDER:-}]"'),
              cwd=tmp_path)
    assert r.stdout.strip().splitlines()[-1] == f"4096|4096|[]|{CLIMB_WINDOW}|{CLIMB_WINDOW}|[]|[]"
    assert "one-rung lane" in r.stdout
    # a store drawn ABOVE its climb cap is not a ladder: refused by name; an unmeasured cap has no long-SI window
    cfg.write_text(yaml.safe_dump({"explicit_cells": ["none"], "bon": {"max_new_tokens": 8192, "climb_cap": 4096}}))
    r = _bash(_source("f22_bon_read_caps k v"), cwd=tmp_path, check=False)
    assert r.returncode == 1 and "ABOVE its climb cap" in r.stderr
    r = _bash(_source("F22_BON_LONG_SI=1; f22_bon_cap_window 8192"), check=False)
    assert r.returncode == 1 and "no long-SI window is measured for a BoN cap of 8192" in r.stderr


def test_the_store_and_the_climb_export_the_cap_the_rung_and_the_ladder_the_sampled_lanes_way_and_a_callers_values_win():
    base = f"f22_bon_read_caps {DEMO} {VIEW}; "
    show = 'echo "$F22_REGISTRY_TOKEN_CAP|$F22_REGISTRY_RUNG_INDEX|$F22_REGISTRY_LADDER|$F22_REGISTRY_PASS_LABEL"'
    assert _bash(_source(base + "f22_bon_registry_exports store; " + show)).stdout.strip().splitlines()[-1] == "512|0|512,4096|base_cap"
    assert _bash(_source(base + "f22_bon_registry_exports climb 4096; " + show)).stdout.strip().splitlines()[-1] == "4096|1|512,4096|ladder_rung"
    assert _bash(_source(base + "f22_bon_registry_exports climb; " + show)).stdout.strip().splitlines()[-1] == "4096|1|512,4096|ladder_rung"   # the sbatch default cap
    # a caller's values win (the sampled lane's rule: `${F22_REGISTRY_TOKEN_CAP:-...}`), as exported scalars
    r = _bash(_source(base + "f22_bon_registry_exports store; " + show), env={"F22_REGISTRY_TOKEN_CAP": "512", "F22_REGISTRY_RUNG_INDEX": "0",
                                                                             "F22_REGISTRY_LADDER": "512,4096,16384", "F22_REGISTRY_PASS_LABEL": "base_cap"})
    assert r.stdout.strip().splitlines()[-1] == "512|0|512,4096,16384|base_cap"
    # the exports are EXPORTED (they must reach emit_cli through f22_registry_register's environment)
    r = _bash(_source(base + "f22_bon_registry_exports climb 4096; env | grep ^F22_REGISTRY_ | sort"))
    assert [l for l in r.stdout.splitlines() if l.startswith("F22_REGISTRY_")] == ["F22_REGISTRY_LADDER=512,4096", "F22_REGISTRY_PASS_LABEL=ladder_rung",
                                                                                   "F22_REGISTRY_RUNG_INDEX=1", "F22_REGISTRY_TOKEN_CAP=4096"]
    # a climb cap positional the config's ladder does not spell is refused by name before any registration
    r = _bash(_source(base + "f22_bon_registry_exports climb 8192"), check=False)
    assert r.returncode == 1 and "cap positional 8192 is not" in r.stderr and "bon.climb_cap 4096" in r.stderr
    r = _bash(_source("f22_bon_registry_exports store"), check=False)
    assert r.returncode == 1 and "call f22_bon_read_caps first" in r.stderr
    # the config f22_resolve_config binds must be the one the caps were read from
    r = _bash(_source(base + "CFG=configs/other.yaml; f22_bon_assert_config"), check=False)
    assert r.returncode == 1 and "bound configs/other.yaml but the BoN caps were read from" in r.stderr
    assert _bash(_source(base + f"CFG=configs/sdfbase_full22_{DEMO}_{VIEW}.yaml; f22_bon_assert_config; echo ok")).stdout.strip().endswith("ok")


# ---------------------------------------------------------------------------------------------- the window check

FAKE_CHECK = textwrap.dedent('''\
    #!/usr/bin/env python
    import argparse, json, os, sys
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--si"); ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--legs"); ap.add_argument("--cap"); ap.add_argument("--out")
    a = ap.parse_args()
    if os.environ.get("FAKE_CHECK_RC"):
        print("[window-check] usage problem"); sys.exit(int(os.environ["FAKE_CHECK_RC"]))
    need = int(os.environ["FAKE_NEED"])
    with open(os.environ["FAKE_CALLS"], "a") as fh:
        fh.write(json.dumps({"L": a.max_model_len, "legs": a.legs, "cap": a.cap, "si": a.si, "config": a.config}) + "\\n")
    rows = [{"leg": l, "family": l, "prompt_tokens": need - 1, "cap": 1, "need": need, "L": a.max_model_len, "fits": need <= a.max_model_len} for l in a.legs.split(",")]
    json.dump({"ok": need <= a.max_model_len, "rows": rows}, open(a.out, "w"))
    print(f"[window-check] {'PASS' if need <= a.max_model_len else 'REFUSE'}: L={a.max_model_len} worst need {need}")
    if need > a.max_model_len:
        print(f"[window-check] REFUSED bon @ cap {a.cap}: {need} > {a.max_model_len}"); sys.exit(3)
    ''')


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"; (repo / "scripts/sdfbase/full22").mkdir(parents=True)
    (repo / "scripts/sdfbase/full22/prompted_window_check.py").write_text(FAKE_CHECK)
    return repo


def _run_check(tmp_path: Path, *, need: int, L: int, legs: str = "bon", cap: str = "512", cells=("none", LONG_SI), rc: str = "", long_si: str = "1"):
    repo = _fake_repo(tmp_path); calls = tmp_path / "calls.jsonl"; calls.write_text(""); out = tmp_path / "out"
    env = {"FAKE_NEED": str(need), "FAKE_CALLS": str(calls), "MPY": VENV_PY, "REPO_DIR": str(repo), "CFG": "configs/x.yaml", "KEY": "k", "VIEW": "v"}
    if rc:
        env["FAKE_CHECK_RC"] = rc
    r = _bash(_source(f'F22_BON_LONG_SI={long_si}; f22_bon_window_check {shlex.quote(str(out))} {L} {legs} {cap} {" ".join(cells)}; echo "WINDOW=$F22_BON_WINDOW"'),
              env=env, check=False)
    return r, [json.loads(l) for l in calls.read_text().splitlines()]


def test_the_window_check_runs_for_a_long_si_cell_on_a_table_view_and_steps_the_window_once_to_the_measured_need(tmp_path):
    # fits: one call at L, the window stands
    r, calls = _run_check(tmp_path / "a", need=6545, L=STORE_WINDOW)
    assert r.returncode == 0 and r.stdout.strip().endswith(f"WINDOW={STORE_WINDOW}") and "long-SI window check PASS" in r.stdout
    assert calls == [{"L": STORE_WINDOW, "legs": "bon", "cap": "512", "si": "si_constitution", "config": "configs/x.yaml"}]
    # the ruling's step: a need the window does not hold steps it to the next 128-token block (6,656 -> 6,784) and re-checks there
    r, calls = _run_check(tmp_path / "b", need=6700, L=STORE_WINDOW)
    assert r.returncode == 0 and r.stdout.strip().endswith("WINDOW=6784"), (r.stdout, r.stderr)
    assert [c["L"] for c in calls] == [STORE_WINDOW, 6784] and "stepping the window to 6784" in r.stdout and "never a clamp" in r.stdout
    # the climb: every named leg at THIS job's cap
    r, calls = _run_check(tmp_path / "c", need=10129, L=CLIMB_WINDOW, cap="4096")
    assert r.returncode == 0 and r.stdout.strip().endswith(f"WINDOW={CLIMB_WINDOW}") and calls[0]["cap"] == "4096"
    # the store's standup with a static family beside it: both legs at their base caps
    r, calls = _run_check(tmp_path / "d", need=9000, L=STATIC_WINDOW, legs="static,bon", cap="base")
    assert r.returncode == 0 and calls == [{"L": STATIC_WINDOW, "legs": "static,bon", "cap": "base", "si": "si_constitution", "config": "configs/x.yaml"}]
    # no long-SI cell among the cells: nothing runs, the window stands as given (a short-prompt job is byte for byte as before)
    r, calls = _run_check(tmp_path / "e", need=99999, L=4864, cells=("none", "prompted__safety_refusal"), long_si="")
    assert r.returncode == 0 and r.stdout.strip().endswith("WINDOW=4864") and calls == []
    # a check that cannot run is fatal by name (never a standup on an unchecked window); so is a second refusal after the one step
    r, _ = _run_check(tmp_path / "f", need=1, L=STORE_WINDOW, rc="2")
    assert r.returncode == 1 and "long-SI window check could not run" in r.stderr and "(rc 2)" in r.stderr
    r, _ = _run_check(tmp_path / "g", need=1, L="x")
    assert r.returncode == 1 and "window 'x' is not an integer" in r.stderr


@pytest.mark.skipif(not any((Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub").glob("models--Qwen--Qwen3.5-4B")),
                    reason="the pinned Qwen/Qwen3.5-4B tokenizer is not in the local HF cache (an offline render; the numbers are pinned in bon_long_si.sh)")
def test_the_measured_need_under_the_pinned_tokenizer_fits_both_windows_offline(tmp_path):
    """The REAL check (the served render under the pinned tokenizer): the demo's BoN prompt under si_constitution needs 6,545 at the 512 cap
    and 10,129 at 4,096 (the numbers bon_long_si.sh records), inside 6,656 / 10,752 with less than a 256-token block to spare at the store."""
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}
    recs = {}
    for name, L, cap in (("store", STORE_WINDOW, "base"), ("climb", CLIMB_WINDOW, "4096")):
        out = tmp_path / f"{name}.json"
        r = _bash(f"{shlex.quote(VENV_PY)} scripts/sdfbase/full22/prompted_window_check.py --config {DEMO_CFG.relative_to(REPO)} --si si_constitution "
                  f"--max-model-len {L} --legs bon --cap {cap} --out {shlex.quote(str(out))}", env=env, check=False)
        assert r.returncode == 0, (name, r.stdout[-800:], r.stderr[-800:])
        recs[name] = json.loads(out.read_text())
    store, climb = recs["store"], recs["climb"]
    assert store["ok"] and climb["ok"] and store["legs"] == ["bon"] and store["caps"] == {"bon": [512]} and climb["caps"] == {"bon": [4096]}
    (srow,), (crow,) = store["rows"], climb["rows"]
    assert srow["need"] == srow["prompt_tokens"] + 512 <= STORE_WINDOW and STORE_WINDOW - srow["need"] < 256
    assert crow["need"] == crow["prompt_tokens"] + 4096 <= CLIMB_WINDOW and srow["prompt_tokens"] == crow["prompt_tokens"]
    assert 5900 <= store["system_tokens_rendered"] <= 6100                                  # the ~6k-token constitution system turn


# ---------------------------------------------------------------------------------------------- (2) the sbatch files

def _between(src: str, start: str, end: str) -> str:
    i = src.index(start); j = src.index(end, i)
    return src[i:j]


STATIC_PRELUDE = ('f22_bon_window_check(){ echo "CHECK out=$1 L=$2 legs=$3 cap=$4 cells=${*:5}"; F22_BON_WINDOW="$2"; }; '
                  'f22_registry_runs_file(){ echo "RUNS out=$1 evals=$2"; }; f22_window_check(){ :; }; '
                  'f22_registry_serving_recorded(){ echo "RECORDED out=$1"; }; '
                  'f22_standup(){ echo "STANDUP mode=$1 out=$2 eval=$4 L=${5:-}"; }; SERVE_DIR=/w/serving; VIEW_WINDOW=""; ')


def _static_resolution(cells, *, static_fams, bon: int, roster_window: str = "") -> str:
    """target_static.sbatch's OWN window-resolution lines and its static-phase check / runs-file / standup lines, run under bash."""
    src = STATIC_SB.read_text()
    block = _between(src, 'f22_bon_detect_long_si "${CELLS[@]}"', 'TARGET_PID=""')
    phase = _between(src, '  if [ -n "$BON_LONG_SI_LEGS" ]; then f22_bon_window_check', '  # native-serving-default: the cells this standup SERVES')
    pre = (f'f22_bon_read_caps {DEMO} {VIEW}; CELLS=({" ".join(map(shlex.quote, cells))}); STATIC_FAMS=({" ".join(static_fams)}); BON={bon}; '
           f'VIEW_WINDOW_STATIC={shlex.quote(roster_window)}; ')
    r = _bash(_source(STATIC_PRELUDE + pre + block + 'echo "RESOLVED window=[$VIEW_WINDOW_STATIC] legs=[$BON_LONG_SI_LEGS]"; ' + phase))
    return r.stdout


def test_target_static_resolves_the_static_phases_window_from_the_roster_then_the_long_si_table_and_the_standup_line_is_unchanged():
    cells = _demo_cells()
    # the BoN-only 512-store job (the recipe's store512 unit, FAMILIES=bon): 6,656, the bon leg checked at its base cap, the store's runs filed
    out = _static_resolution(cells, static_fams=[], bon=1)
    assert f"RESOLVED window=[{STORE_WINDOW}] legs=[bon]" in out
    assert f"CHECK out=/w/serving/static L={STORE_WINDOW} legs=bon cap=base cells={' '.join(cells)}" in out
    assert "RUNS out=/w/serving/static evals=strongreject_bon" in out and f"STANDUP mode=attacks out=/w/serving/static eval=families_served:short L={STORE_WINDOW}" in out
    assert out.index("STANDUP mode=attacks") < out.index("RECORDED out=/w/serving/static")     # the gate reads the spool back once the standup is up
    # a static family in the job: the static legs' window, both legs checked, every served eval in the runs file
    out = _static_resolution(cells, static_fams=["pasttense", "codeattack"], bon=1)
    assert f"RESOLVED window=[{STATIC_WINDOW}] legs=[static,bon]" in out and f"CHECK out=/w/serving/static L={STATIC_WINDOW} legs=static,bon cap=base" in out
    assert "RUNS out=/w/serving/static evals=pasttense,codeattack,strongreject_bon" in out and f"L={STATIC_WINDOW}" in out
    out = _static_resolution(cells, static_fams=["grok420"], bon=0)
    assert f"RESOLVED window=[{STATIC_WINDOW}] legs=[static]" in out and "RUNS out=/w/serving/static evals=grok420" in out
    # the window check steps the window and the standup takes the stepped one
    src = STATIC_SB.read_text()
    phase = _between(src, '  if [ -n "$BON_LONG_SI_LEGS" ]; then f22_bon_window_check', '  # native-serving-default: the cells this standup SERVES')
    stepped = _bash(_source(STATIC_PRELUDE + 'f22_bon_window_check(){ F22_BON_WINDOW=6784; }; CELLS=(none); STATIC_FAMS=(); BON=1; '
                            f'VIEW_WINDOW_STATIC={STORE_WINDOW}; BON_LONG_SI_LEGS=bon; ' + phase)).stdout
    assert "STANDUP mode=attacks out=/w/serving/static eval=families_served:short L=6784" in stepped
    # no long-SI cell: "" — the table's 4,864, no check, byte for byte as before (the paper's short-prompt views)
    out = _static_resolution([c for c in cells if c != LONG_SI], static_fams=[], bon=1)
    assert "RESOLVED window=[] legs=[]" in out and "CHECK " not in out and "STANDUP mode=attacks out=/w/serving/static eval=families_served:short L=" in out
    assert re.search(r"^STANDUP mode=attacks out=/w/serving/static eval=families_served:short L=$", out, re.M)   # no 5th argument at all
    # a roster window (a per-leg / one-window prompted view) wins and the long-SI check stands down: the roster's own check covers it
    out = _static_resolution(cells, static_fams=[], bon=1, roster_window="10496")
    assert "RESOLVED window=[10496] legs=[]" in out and "CHECK " not in out and "L=10496" in out
    # the pinned standup line is byte-identical (tests/test_sdfbase_full22_wave.py pins it), and the order check -> runs file -> standup holds
    assert 'f22_standup attacks "$SERVE_DIR/static" "" families_served:short ${VIEW_WINDOW_STATIC:+"$VIEW_WINDOW_STATIC"} || { RC=1; exit 1; }' in src
    i_det = src.index('f22_bon_detect_long_si "${CELLS[@]}"'); i_chk = src.index('f22_bon_window_check "$SERVE_DIR/static"')
    i_runs = src.index('f22_registry_runs_file "$SERVE_DIR/static"'); i_up = src.index('f22_standup attacks "$SERVE_DIR/static"')
    i_rec = src.index('f22_registry_serving_recorded "$SERVE_DIR/static"')
    assert i_det < i_chk < i_runs < i_up < i_rec < src.index("f22_served_cells_filter CELLS", i_up)
    # the helper is sourced after f22_init binds the checkout and the store's exports precede the registering f22_resolve_config
    i_init = src.index('f22_init "$WAVE_SHA"'); i_src = src.index("source scripts/sdfbase/full22/bon_long_si.sh")
    i_exp = src.index('[ "$FAMILIES" = bon ] && f22_bon_registry_exports store'); i_res = src.index('f22_resolve_config "$KEY" "$VIEW"')
    assert i_init < i_src < i_exp < i_res < src.index("f22_bon_assert_config")
    assert not any("--export=" in l for l in src.splitlines() if not l.lstrip().startswith("#"))


CLIMB_PRELUDE = ('f22_bon_window_check(){ echo "CHECK out=$1 L=$2 legs=$3 cap=$4 cells=${*:5}"; F22_BON_WINDOW="$2"; }; '
                 'f22_registry_runs_file(){ echo "RUNS out=$1 evals=$2"; }; f22_registry_serving_recorded(){ echo "RECORDED out=$1"; }; '
                 'f22_standup(){ echo "STANDUP mode=$1 out=$2 eval=$4 L=${5:-}"; }; SERVE_DIR=/w/serving; F22_ROSTER=configs/sdfbase_full22_roster.yaml; ')


def _climb_resolution(key: str, view: str, cells, *, cap: str = "4096") -> str:
    src = CLIMB_SB.read_text()
    block = _between(src, 'VIEW_WINDOW_STATIC=$(python3 scripts/sdfbase/full22/roster_view.py', '# THE PLAN (shard 0 writes it')
    up = _between(src, 'if [ -z "$VIEW_WINDOW_STATIC" ] && [ -n "$F22_BON_LONG_SI" ]; then f22_bon_window_check', '# native-serving-default: the cells this standup SERVES')
    pre = f'KEY={key}; VIEW={view}; CAP={cap}; CELLS=({" ".join(map(shlex.quote, cells))}); F22_BON_STORE_CAP=512; F22_BON_CLIMB_CAP=4096; '
    return _bash(_source(CLIMB_PRELUDE + pre + block + 'echo "RESOLVED roster=[$VIEW_WINDOW_STATIC] climb=[$CLIMB_L]"; ' + up)).stdout


def test_bon_climb_stands_up_at_the_roster_window_else_the_long_si_climb_window_else_the_table():
    cells = _demo_cells()
    out = _climb_resolution(DEMO, VIEW, cells)
    assert f"RESOLVED roster=[] climb=[{CLIMB_WINDOW}]" in out
    assert f"CHECK out=/w/serving/static L={CLIMB_WINDOW} legs=bon cap=4096 cells={' '.join(cells)}" in out
    assert "RUNS out=/w/serving/static evals=strongreject_bon" in out and f"STANDUP mode=attacks out=/w/serving/static eval=families_served:short L={CLIMB_WINDOW}" in out
    out = _climb_resolution(DEMO, VIEW, [LONG_SI])                                           # a cells= positional naming the cell alone
    assert f"climb=[{CLIMB_WINDOW}]" in out and f"L={CLIMB_WINDOW}" in out
    out = _climb_resolution(DEMO, VIEW, [c for c in cells if c != LONG_SI])                 # no long-SI cell: the table's, no check
    assert "RESOLVED roster=[] climb=[]" in out and "CHECK " not in out
    assert re.search(r"^STANDUP mode=attacks out=/w/serving/static eval=families_served:short L=$", out, re.M)   # no 5th argument at all
    out = _climb_resolution("llama8b", "shortconst", [LONG_SI])                             # the roster's per-leg static window wins; no long-SI check
    assert "RESOLVED roster=[10496] climb=[10496]" in out and "CHECK " not in out and "L=10496" in out
    src = CLIMB_SB.read_text()
    assert "families_served:short" in src and 'f22_standup attacks "$SERVE_DIR/static" "" families_served:short ${CLIMB_L:+"$CLIMB_L"} || { RC=1; exit 1; }' in src
    i_init = src.index('f22_init "$WAVE_SHA"'); i_src = src.index("source scripts/sdfbase/full22/bon_long_si.sh")
    i_exp = src.index('f22_bon_registry_exports climb "$CAP"'); i_res = src.index('f22_resolve_config "$KEY" "$VIEW"')
    assert i_init < i_src < i_exp < i_res < src.index("f22_bon_assert_config")
    i_chk = src.index('f22_bon_window_check "$SERVE_DIR/static"'); i_runs = src.index('f22_registry_runs_file "$SERVE_DIR/static"')
    i_up = src.index('f22_standup attacks "$SERVE_DIR/static"'); i_rec = src.index('f22_registry_serving_recorded "$SERVE_DIR/static"')
    assert i_chk < i_runs < i_up < i_rec < src.index("f22_served_cells_filter CELLS", i_up)
    assert out.index("STANDUP mode=attacks") < out.index("RECORDED out=/w/serving/static")     # the climb's replay runs the gate after its standup too
    # the climb exports the registration cells from ITS positional before f22_init, like the store job (cells-fix-2)
    i_pos = src.index('CELLS_ARG="${5:-}"; CAP="${6:-4096}"')
    i_cells = src.index('[ -n "$CELLS_ARG" ] && export F22_REGISTRY_CELLS="${F22_REGISTRY_CELLS:-$CELLS_ARG}"')
    assert i_pos < i_cells < src.index("f22_init ", i_pos)
    assert not any("--export=" in l for l in src.splitlines() if not l.lstrip().startswith("#"))


# ---------------------------------------------------------------------------------------------- (3) the REAL registration

def _checkout(root: Path, key: str = "qwen35_4b") -> Path:
    cfg = yaml.safe_dump({"explicit_cells": ["none", LONG_SI, "prompted__safety_refusal"], "model": "Qwen/Qwen3.5-4B",
                          "bon": {"max_new_tokens": 512, "climb_cap": 4096, "n_max": 75, "goal_subset": "jbb"}})
    roster = yaml.safe_dump({"targets": {key: {"views": {"main": {"cells": {LONG_SI: None, "prompted__safety_refusal": None}}}}}})
    return synthetic_checkout(root, configs={f"configs/sdfbase_full22_{key}.yaml": cfg, "configs/sdfbase_full22_roster.yaml": roster,
                                             "configs/sdfbase_full22_pins.json": "{}"})


def _register_like_the_sbatch(checkout: Path, shared: Path, *, leg: str, job_id: str, job_name: str, families: str = "bon", cap: str = "4096") -> subprocess.CompletedProcess:
    """The sbatch's registration sequence FOR REAL: the real bon_long_si.sh (read caps, the leg's exports) then common.sh's f22_registry_register
    (the real emit_cli register over the checkout's seeds) with the pending file, as the job does before any standup."""
    from sisteer.registry import synth
    shared.mkdir(parents=True, exist_ok=True)                                              # the storage root must exist on the node (f22_registry_register)
    exports = ('[ "$FAMILIES" = bon ] && f22_bon_registry_exports store' if leg == "store" else f'f22_bon_registry_exports climb {cap}')
    script = textwrap.dedent(f"""
        set -u
        cd {checkout}
        export REPO_DIR={REPO} SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
        export SLURM_JOB_ID={job_id} SLURM_JOB_NAME={job_name} WAVE_SHA={synth.code_sha("a")}
        export KEY=qwen35_4b VIEW=main TAG=qwen35_4b CFG=configs/sdfbase_full22_qwen35_4b.yaml F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        export F22_REGISTRY_EVALS=strongreject_bon FAMILIES={families}
        source scripts/sdfbase/full22/common.sh
        f22_log(){{ echo "[log] $*"; }}
        source {HELPER}
        f22_bon_read_caps "$KEY" "$VIEW"
        {exports}
        f22_registry_register || true
        f22_bon_assert_config
        echo "REGISTERED pending=${{SISTEER_REGISTRY_RUN_IDENTITY:-}} cap=${{F22_REGISTRY_TOKEN_CAP:-}} rung=${{F22_REGISTRY_RUNG_INDEX:-}} ladder=${{F22_REGISTRY_LADDER:-}}"
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)


def _plans(pending: Path) -> dict:
    return {r["route"]["object"]: r["plan"] for r in json.loads(pending.read_text())["runs"]}


def test_the_store_job_registers_cap_512_rung_0_and_the_climb_cap_4096_rung_1_on_the_512_4096_ladder_for_real(tmp_path):
    co = _checkout(tmp_path)
    r = _register_like_the_sbatch(co, tmp_path / "s1", leg="store", job_id="40001", job_name="f22-stat-qwen35_4b-3")
    assert r.returncode == 0 and "REGISTERED pending=" in r.stdout, (r.stdout, r.stderr)
    assert "cap=512 rung=0 ladder=512,4096" in r.stdout and "the BoN store registers at cap 512 = rung 0" in r.stdout
    pend = tmp_path / "s1" / "registry" / "pending" / "40001.json"
    plans = _plans(pend)
    assert set(plans) == {"none", LONG_SI, "prompted__safety_refusal"}
    for obj, p in plans.items():
        assert (p["eval_id"], p["token_cap_tokens"], p["token_cap_rung_index"], p["token_cap_ladder_json"], p["pass_label"]) == ("strongreject_bon", 512, 0, LADDER, "base_cap"), obj
        assert p["extra"]["rung_registration"] == {"token_cap_tokens": 512, "token_cap_rung_index": 0, "token_cap_ladder_json": LADDER, "pass_label": "base_cap", "trial_index": None}
        assert p["serving_path"] == "lora_native" and p["campaign"] == "sdfbase_full22"
    assert json.loads(pend.read_text())["skipped"] == []                                   # the seeded instrument ladders [512, 4096]: every pair registered
    # the climb: rung 1 at its cap positional, on the same ladder
    r = _register_like_the_sbatch(co, tmp_path / "s2", leg="climb", job_id="40002", job_name="f22-bonclimb-qwen35_4b")
    assert r.returncode == 0 and "cap=4096 rung=1 ladder=512,4096" in r.stdout, (r.stdout, r.stderr)
    plans = _plans(tmp_path / "s2" / "registry" / "pending" / "40002.json")
    assert set(plans) == {"none", LONG_SI, "prompted__safety_refusal"}
    for obj, p in plans.items():
        assert (p["token_cap_tokens"], p["token_cap_rung_index"], p["token_cap_ladder_json"], p["pass_label"]) == (4096, 1, LADDER, "ladder_rung"), obj
    # a MIXED job (-all: the statics beside the store) keeps the registration as it was: no rung, the instrument's read-at cap
    r = _register_like_the_sbatch(co, tmp_path / "s3", leg="store", job_id="40003", job_name="f22-stat-qwen35_4b-all", families="pasttense,bon")
    assert r.returncode == 0 and "cap= rung= ladder=" in r.stdout, (r.stdout, r.stderr)
    plans = _plans(tmp_path / "s3" / "registry" / "pending" / "40003.json")
    assert plans and all(p.get("token_cap_rung_index") is None and p.get("pass_label") is None for p in plans.values()), plans
    assert all("rung_registration" not in p["extra"] for p in plans.values())
    assert {p["token_cap_tokens"] for p in plans.values()} == {4096}                        # the instrument's read-at cap, as before
    # a climb cap outside the ladder never reaches the registration: refused by name first
    r = _register_like_the_sbatch(co, tmp_path / "s4", leg="climb", job_id="40004", job_name="f22-bonclimb-qwen35_4b", cap="8192")
    assert r.returncode == 1 and "cap positional 8192 is not" in r.stdout + r.stderr and not (tmp_path / "s4" / "registry" / "pending" / "40004.json").exists()


# ---------------------------------------------------------------------------------------------- (4) the served runs file + the standup's hook

def test_the_served_runs_file_keys_every_run_by_its_harness_spelling_and_the_standups_hook_records_the_attested_window(tmp_path, monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)
    co = _checkout(tmp_path)
    r = _register_like_the_sbatch(co, tmp_path / "s", leg="store", job_id="50001", job_name="f22-stat-qwen35_4b-3")
    assert r.returncode == 0, (r.stdout, r.stderr)
    pend = tmp_path / "s" / "registry" / "pending" / "50001.json"
    plans = _plans(pend)
    # the shell helper writes the file through the script, from the job's pending identity, into the standup's out dir
    out = tmp_path / "serving" / "static"
    sh = _bash(_source(f'MPY={shlex.quote(VENV_PY)}; REPO_DIR={shlex.quote(str(REPO))}; SISTEER_REGISTRY_RUN_IDENTITY={shlex.quote(str(pend))}; '
                       f'f22_registry_runs_file {shlex.quote(str(out))} strongreject_bon'))
    assert "served runs file: 3 run(s) under 3 served key(s)" in sh.stdout
    runs = json.loads((out / hooks.RUNS_FILE).read_text())
    assert set(runs) == {"none", "prompted__constitution", "prompted__safety"}                 # the endpoint's spellings, never the config's
    assert runs["prompted__constitution"] == {"run_id": plans[LONG_SI]["run_id"], "campaign": "sdfbase_full22", "expected": {"serving_path": "lora_native"}}
    assert runs["prompted__safety"]["run_id"] == plans["prompted__safety_refusal"]["run_id"] and runs["none"]["run_id"] == plans["none"]["run_id"]
    # the standup's write_endpoint hook (native_standup._registry_serving_facts -> hooks.emit_serving_patches, reading the file beside endpoint.json)
    # files one run_patch per run with the ATTESTED window — the fact every demo BoN row lacked
    spool = tmp_path / "hook_spool"
    endpoint = {"attested_max_num_seqs": 108, "attested_max_model_len": STORE_WINDOW, "max_model_len": STORE_WINDOW, "engine": "lora_native",
                "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "adapters": {}, "served_positions": {},
                "serving_versions": {"vllm": "0.25.1"}, "proxy_url": "http://127.0.0.1:1/v1", "slurm_job_id": "50001"}
    paths = hooks.emit_serving_patches(endpoint, None, out_dir=str(out), spool=str(spool), registry_emit=True)
    assert len(paths) == 3
    patches = [(rt, p) for rt, p in read_spool(spool) if rt == "run_patch"]
    assert {p["run_id"] for _, p in patches} == {v["run_id"] for v in runs.values()}
    for _, p in patches:
        assert p["fields"]["context_window_tokens"] == STORE_WINDOW and p["fields"]["served_max_num_seqs"] == 108
        assert p["fields"]["served_max_num_seqs_source"] == "attested" and p["fields"]["serving_engine_version"] == "vllm 0.25.1"
        assert p["verify"]["serving_path"] == "lora_native" and p["verify"]["model_revision"] == endpoint["model_revision"]
    # no identities (a manual run): the helper says so and writes nothing; a wrong eval list is a refusal by name, not an empty file
    sh = _bash(_source(f'MPY={shlex.quote(VENV_PY)}; REPO_DIR={shlex.quote(str(REPO))}; f22_registry_runs_file {shlex.quote(str(tmp_path / "none"))} strongreject_bon'))
    assert "no run identities for this job" in sh.stdout and not (tmp_path / "none" / hooks.RUNS_FILE).exists()
    sh = _bash(_source(f'MPY={shlex.quote(VENV_PY)}; REPO_DIR={shlex.quote(str(REPO))}; SISTEER_REGISTRY_RUN_IDENTITY={shlex.quote(str(pend))}; '
                       f'f22_registry_runs_file {shlex.quote(str(tmp_path / "x"))} crescendo'), check=False)
    assert sh.returncode == 1 and "no run of evals ['crescendo']" in sh.stderr and "could not be written" in sh.stderr


def test_the_runs_map_filters_to_the_standups_evals_lists_shared_keys_and_refuses_a_cell_the_harness_cannot_spell():
    def run(rid, ev, obj):
        return {"run_id": rid, "route": {"eval_id": ev, "object": obj}, "plan": {"campaign": "sdfbase_full22", "serving_path": "lora_native"}}
    pend = {"runs": [run("r1", "strongreject_bon", LONG_SI), run("r2", "strongreject_bon", "constv2-mix-full__lora_r4_ce"),
                     run("r3", "pasttense", "constv2-mix-full__lora_r4_ce"), run("r4", "xstest", "none"), run("r2", "strongreject_bon", "constv2-mix-full__lora_r4_ce")]}
    m = RF.runs_map(pend, ["strongreject_bon", "pasttense"])
    assert set(m) == {"prompted__constitution", "si_constv2_mix_full__lora_r4_ce"}                   # the controls run is another standup's
    assert m["prompted__constitution"]["run_id"] == "r1"
    assert [e["run_id"] for e in m["si_constv2_mix_full__lora_r4_ce"]] == ["r2", "r3"]                # the -all job's shared served route: the LIST form, deduped by run id
    with pytest.raises(RF.RunsFileRefused, match="no run of evals"):
        RF.runs_map(pend, ["crescendo"])
    with pytest.raises(RF.RunsFileRefused, match="has no harness spelling"):
        RF.runs_map({"runs": [run("r9", "strongreject_bon", "alphakl__not_a_cell@@")]}, ["strongreject_bon"])
    with pytest.raises(RF.RunsFileRefused, match="no `runs` list"):
        RF.runs_map({"run_id": "x"}, ["strongreject_bon"])
    with pytest.raises(RF.RunsFileRefused, match="names no eval"):
        RF.runs_map(pend, [""])
    assert RF.RUNS_FILE == hooks.RUNS_FILE == "registry_runs.json"


# ---------------------------------------------------------------------------------------------- (4b) the EXECUTABLE standup path

def _served_runs(served_keys, *, campaign: str = "sdfbase_full22", serving_path: str = "lora_native") -> dict:
    """A runs file in registry_runs_file.py's shape over SYNTHETIC run ids (never a real registration)."""
    from sisteer.registry import identity
    return {k: {"run_id": identity.run_id(campaign, f"f22-test-{k}", f"{k}@a1b2c3d4", "strongreject_bon", "strongreject_bon_v1"),
                "campaign": campaign, "expected": {"serving_path": serving_path}} for k in served_keys}


def _spool_patches(spool: Path) -> list:
    """Every run_patch RECORD (envelope + payload) under a spool, in file order."""
    out = []
    for p in sorted(spool.rglob("*-run_patch.jsonl")):
        out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def _recorded_gate(out: Path, spool: Path, *, timeout_s: str = "1") -> subprocess.CompletedProcess:
    """f22_registry_serving_recorded over a standup dir FOR REAL (the helper -> registry_serving_recorded.py), reading `spool`."""
    return _bash(_source(f'MPY={shlex.quote(VENV_PY)}; REPO_DIR={shlex.quote(str(REPO))}; f22_registry_serving_recorded {shlex.quote(str(out))}'),
                 env={emit.SPOOL_ENV: str(spool), "F22_REGISTRY_SERVING_RECORDED_TIMEOUT_S": timeout_s}, check=False)


def _emission_on_into(monkeypatch, spool: Path) -> None:
    """The suite runs registry emission OFF (tests/conftest.py); this opts the test in, into a temp spool, never a live one."""
    for k in (hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV, emit.DISABLED_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))


def test_the_native_standup_these_jobs_reach_records_the_window_from_the_runs_file_in_its_out_dir_for_real_and_the_gate_observes_it(tmp_path, monkeypatch):
    """REVIEW ROUND 2. The finding named mt_serving_up.py's LENS endpoint writer as the one the BoN jobs reach. They reach the NATIVE one:
    both jobs stand up under --serving-path lora_native (the demo's store 983259: `serving_path=lora_native serving_path_mode=explicit`,
    endpoint engine=lora_native standup_front=lora_serving_up), i.e. mt.main -> _main_native -> lora_serving_up.bring_up_lora ->
    native_standup.bring_up_native -> write_endpoint, whose hook reads <out>/registry_runs.json — and 983259's standup.log carries that
    hook's own notice ("no run identities for the served keys ... write registry_runs.json beside endpoint.json"). Here the REAL mt.main
    runs that path on the CPU harness (the scripted upstream, the real proxy and controls; tests/test_mt_serving_up_serving_path.py) with the
    launcher's runs file ALREADY in --out, as f22_registry_runs_file leaves it before f22_standup, at the store's long-SI window, emission on
    into a temp spool: one run_patch per served run with context_window_tokens == the ATTESTED window, from the native writer. Then the
    post-standup gate (f22_registry_serving_recorded -> registry_serving_recorded.py) observes it for real, and refuses by name an empty
    spool, another window and another standup's patch."""
    SP = pytest.importorskip("test_mt_serving_up_serving_path")
    spool = tmp_path / "spool"
    _emission_on_into(monkeypatch, spool)
    mt = SP._load_mt("_mt_bon_long_si_native")
    runs = _served_runs(["none", SP.E.GEMMA_HCELL])
    cap = SP._run_native(mt, tmp_path, monkeypatch, "--max-model-len", str(STORE_WINDOW), pre_files={hooks.RUNS_FILE: json.dumps(runs)})
    assert cap["rc"] == 0
    ep = cap["endpoint"]
    assert ep["engine"] == "lora_native" and ep["standup_front"] == "lora_serving_up" and ep["serving_path"] == "lora_native"
    assert ep["attested_max_model_len"] == STORE_WINDOW == ep["max_model_len"]
    recs = _spool_patches(spool)
    by_run = {r["payload"]["run_id"]: r for r in recs}
    assert set(by_run) == {v["run_id"] for v in runs.values()} and len(recs) == 2
    for key, v in runs.items():
        rec = by_run[v["run_id"]]; f = rec["payload"]["fields"]
        assert rec["producer"]["script"] == "capeval.serving.native_standup.write_endpoint" == RR.SERVING_WRITERS[0]
        assert f["context_window_tokens"] == STORE_WINDOW and f["served_max_num_seqs"] == ep["attested_max_num_seqs"] and f["served_max_num_seqs_source"] == "attested"
        assert f["extra_json"]["served_key"] == key and f["extra_json"]["proxy_url"] == ep["proxy_url"]
        assert rec["payload"]["verify"]["serving_path"] == "lora_native" and rec["payload"]["verify"]["model_revision"] == ep["model_revision"]
    assert by_run[runs[SP.E.GEMMA_HCELL]["run_id"]]["payload"]["fields"].get("adapter_fingerprint")       # the served LoRA's attested block, hashed
    # the gate, for real: endpoint.json is unlinked at teardown — the durable copy f22_standup keeps stands in (RR.ENDPOINT_FILES)
    out = cap["out"]
    (out / "endpoint.recorded.json").write_text(json.dumps(ep, default=str))
    r = _recorded_gate(out, spool)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert f"served window OBSERVED: context_window_tokens {STORE_WINDOW} on 2 run(s) of 2" in r.stdout
    # refusals by name: an empty spool (the hook never wrote), another window (the standup served something else), another standup's patch
    r = _recorded_gate(out, tmp_path / "empty_spool")
    assert r.returncode == 1 and "NOT observed" in r.stderr and "2 of 2 run(s)" in r.stderr and "refused rather than run" in r.stderr
    found, missing = RR.recorded_patches(runs, dict(ep, attested_max_model_len=4864, max_model_len=4864), str(spool))
    assert not found and set(missing) == set(by_run)
    found, missing = RR.recorded_patches(runs, dict(ep, proxy_url="http://127.0.0.1:1/v1"), str(spool))
    assert not found and len(missing) == 2
    found, missing = RR.recorded_patches(runs, ep, str(spool))
    assert set(found) == set(by_run) and not missing
    # no runs file (a manual run; f22_registry_runs_file said so): nothing to observe, the helper says so and passes
    r = _bash(_source(f'MPY={shlex.quote(VENV_PY)}; REPO_DIR={shlex.quote(str(REPO))}; f22_registry_serving_recorded {shlex.quote(str(tmp_path / "manual"))}'))
    assert "no served runs file" in r.stdout


def test_the_lens_writer_files_the_same_serving_facts_from_the_runs_file_beside_its_endpoint_for_real(tmp_path, monkeypatch):
    """The finding's literal ask, made true for the writer it named: mt_serving_up.py's LENS body (--serving-path lens, lines the native
    jobs never reach) calls the same hook right after its atomic endpoint.json rename — the REAL lens main on the CPU harness
    (tests/test_mt_serving_up_thinking.py) with the runs file beside its --out files one run_patch with the ATTESTED window under its own
    producer script, which the gate admits; the endpoint record itself is unchanged (test_mt_serving_up_serving_path.py's lens golden still
    compares it to the pre-landing tree); nothing is written without the file."""
    T = pytest.importorskip("test_mt_serving_up_thinking")
    SP = pytest.importorskip("test_mt_serving_up_serving_path")
    spool = tmp_path / "spool"
    _emission_on_into(monkeypatch, spool)
    mt = SP._load_mt("_mt_bon_long_si_lens")
    runs = _served_runs(["none"], serving_path="lens")
    snap, _proxy, _launch = T._run(mt, tmp_path, monkeypatch, SP.KEY, *T.M5T_EXTRA, pre_files={hooks.RUNS_FILE: json.dumps(runs)})
    ep = snap["endpoint"]
    assert "engine" not in ep and ep["lens_reduction_mode"] == "stacked"
    window = ep["attested_max_model_len"] if ep.get("attested_max_model_len") is not None else ep["max_model_len"]
    recs = _spool_patches(spool)
    assert len(recs) == 1 and recs[0]["payload"]["run_id"] == runs["none"]["run_id"]
    assert recs[0]["producer"]["script"] == mt.LENS_SERVING_FACTS_SCRIPT == "sdfbase.mt_serving_up.lens_endpoint" == RR.SERVING_WRITERS[1]
    f = recs[0]["payload"]["fields"]
    assert f["context_window_tokens"] == window and f["serving_config_sha256"] == ep["serving_config_sha256"]
    assert f["extra_json"]["proxy_url"] == ep["proxy_url"] and f["extra_json"]["served_key"] == "none"
    # a lens record names no engine, so the path under `verify` is the runs file's own `expected` (the launcher's re-observation), never the endpoint's
    assert recs[0]["payload"]["verify"] == {"model_revision": ep["model_revision"], "serving_path": "lens"}
    found, missing = RR.recorded_patches(runs, ep, str(spool))
    assert set(found) == {runs["none"]["run_id"]} and not missing
    # the call sits right after the atomic rename and before READY, once, in the lens body; the native path's writer is the core's own
    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    i_ren = src.index('tmp.rename(out / "endpoint.json")'); i_hook = src.index("_lens_registry_serving_facts(ep, out)", i_ren)
    assert i_ren < i_hook < src.index('print(f"[mt-standup] READY: proxy {proxy_url} serving', i_ren)
    assert src.count("_lens_registry_serving_facts(ep, out)") == 1
    # without the runs file: no record at all (emission is on here)
    spool2 = tmp_path / "spool2"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool2))
    bare = tmp_path / "bare"; bare.mkdir()
    T._run(mt, bare, monkeypatch, SP.KEY, *T.M5T_EXTRA)
    assert not spool2.exists()


def test_recorded_patches_admits_only_this_standups_serving_patch_and_the_cli_exits_by_name(tmp_path, capsys):
    from sisteer.registry import identity
    runs = _served_runs(["none", "prompted__constitution"])
    rid_none, rid_c = runs["none"]["run_id"], runs["prompted__constitution"]["run_id"]
    rid_c2 = identity.run_id("sdfbase_full22", "f22-test-other", "prompted__constitution@a1b2c3d4", "pasttense", "pasttense_v1")
    runs["prompted__constitution"] = [runs["prompted__constitution"], {"run_id": rid_c2, "campaign": "sdfbase_full22", "expected": {}}]   # the -all job's LIST form
    ep = {"attested_max_model_len": STORE_WINDOW, "max_model_len": 4864, "proxy_url": "http://127.0.0.1:41000/v1", "slurm_job_id": "50001"}
    native, lens = RR.SERVING_WRITERS

    def rec(rid, *, script=native, window=STORE_WINDOW, key="none", proxy=ep["proxy_url"], job="50001", rtype="run_patch"):
        return {"record_type": rtype, "producer": {"script": script},
                "payload": {"run_id": rid, "fields": {"context_window_tokens": window, "extra_json": {"served_key": key, "proxy_url": proxy, "slurm_job_id": job}}, "verify": {}}}

    spool = tmp_path / "spool"
    seq = {"n": 0}

    def put(rid, *recs):
        d = spool / "sdfbase_full22" / identity.run_dir_name(rid); d.mkdir(parents=True, exist_ok=True)
        seq["n"] += 1
        (d / f"{seq['n']:06d}-run_patch.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    assert RR.expected_window(ep) == STORE_WINDOW and RR.expected_window({"max_model_len": 4864}) == 4864   # the attested window first
    with pytest.raises(RR.RecordedRefused, match="neither attested_max_model_len nor max_model_len"):
        RR.expected_window({"proxy_url": "x"})
    with pytest.raises(RR.RecordedRefused, match="has no run_id"):
        RR.run_entries({"none": {"campaign": "c"}})
    assert [(k, r) for k, r, _c in RR.run_entries(runs)] == [("none", rid_none), ("prompted__constitution", rid_c), ("prompted__constitution", rid_c2)]
    # nothing in the spool: every run missing
    found, missing = RR.recorded_patches(runs, ep, str(spool))
    assert not found and set(missing) == {rid_none, rid_c, rid_c2}
    # records that are NOT this standup's serving patch for the run: an earlier attempt (another port), another job, another served key, another
    # window, a non-serving writer with the right window, a run_state, another run's patch in this run's directory
    put(rid_none, rec(rid_none, proxy="http://127.0.0.1:1/v1"), rec(rid_none, job="40000"), rec(rid_none, key="prompted__constitution"),
        rec(rid_none, window=4864), rec(rid_none, script="scripts/registry/emit_cli.py"), rec(rid_none, rtype="run_state"), rec(rid_c, key="prompted__constitution"))
    found, missing = RR.recorded_patches(runs, ep, str(spool))
    assert not found and set(missing) == {rid_none, rid_c, rid_c2}
    # this standup's patches, the lens writer's admitted beside the native one; the list form needs BOTH runs of the served key
    put(rid_none, rec(rid_none))
    put(rid_c, rec(rid_c, key="prompted__constitution", script=lens))
    found, missing = RR.recorded_patches(runs, ep, str(spool))
    assert set(found) == {rid_none, rid_c} and missing == [rid_c2]
    put(rid_c2, rec(rid_c2, key="prompted__constitution"))
    found, missing = RR.recorded_patches(runs, ep, str(spool))
    assert set(found) == {rid_none, rid_c, rid_c2} and not missing and all(Path(p).name.endswith("-run_patch.jsonl") for p in found.values())
    # an endpoint with no job id (a manual standup) checks the port alone
    found, missing = RR.recorded_patches(runs, {k: v for k, v in ep.items() if k != "slurm_job_id"}, str(spool))
    assert not missing
    # the CLI: 2 without a runs file / endpoint record, 3 when a run is missing (the standup.log's [registry] lines echoed), 0 when observed
    out = tmp_path / "standup"; out.mkdir()
    assert RR.main(["--out", str(out), "--spool", str(spool), "--timeout-s", "0"]) == 2 and "no " in capsys.readouterr().err
    (out / hooks.RUNS_FILE).write_text(json.dumps(runs))
    assert RR.main(["--out", str(out), "--spool", str(spool), "--timeout-s", "0"]) == 2 and "no endpoint record" in capsys.readouterr().err
    (out / "endpoint.recorded.json").write_text(json.dumps(dict(ep, attested_max_model_len=10752)))
    (out / "standup.log").write_text("[mt-standup] READY\n[registry] serving standup: hook failed (OSError: boom); the run continues\n")
    assert RR.main(["--out", str(out), "--spool", str(spool), "--timeout-s", "0"]) == 3
    err = capsys.readouterr().err
    assert "NOT observed" in err and "3 of 3 run(s)" in err and "expected context_window_tokens 10752" in err and "standup.log: [registry] serving standup: hook failed" in err
    (out / "endpoint.json").write_text(json.dumps(ep))                     # the live record wins over the durable copy
    assert RR.main(["--out", str(out), "--spool", str(spool), "--timeout-s", "0"]) == 0
    assert f"served window OBSERVED: context_window_tokens {STORE_WINDOW} on 3 run(s) of 3" in capsys.readouterr().out
    assert RR.RUNS_FILE == hooks.RUNS_FILE and set(RR.SERVING_WRITERS) == {"capeval.serving.native_standup.write_endpoint", "sdfbase.mt_serving_up.lens_endpoint"}


# ---------------------------------------------------------------------------------------------- (5) the plan is unchanged

def test_the_demos_and_a_paper_keys_bon_plan_rows_are_unchanged_the_window_is_a_job_time_fact(monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")
    import stage as ST
    sha = "a" * 40
    for key, view in ((DEMO, VIEW), ("qwen35_4b", "main")):
        rows = {r["unit"]: r for r in ST.unit_plan(key, view, families=["static", "bon"], sha=sha) if r["family"] == "bon"}
        assert set(rows) == {"store512", "climb"}, key
        env = {"F22_BON_DECODE_PRESET": "canonical", "F22_REGISTRY_EVALS": "strongreject_bon"}
        assert rows["store512"]["script"] == "scripts/sdfbase/full22/target_static.sbatch" and rows["store512"]["args"] == [key, view, "bon", sha, "9"]
        assert rows["climb"]["script"] == "scripts/sdfbase/full22/bon_climb.sbatch" and rows["climb"]["args"] == [key, view, sha, "4"] and rows["climb"]["array"] == "0-3"
        assert rows["store512"]["env"] == env and rows["climb"]["env"] == env, key                 # no window positional, no cap env: the job reads its cells
        assert rows["climb"]["depends_on"] == f"{key}/{view}/bon/store512"
    # the climb is plumbed for a cells selection since the rerun pair (positional 5 = index 4: `<key> <view> <sha> <nshards> [cells] [cap]`;
    # tests/test_bon_rerun_units.py runs the script's own parser line) — the plain rows above stay byte-identical: no positional is added unasked
    assert ST.CELLS_POSITIONAL["scripts/sdfbase/full22/bon_climb.sbatch"] == 4
