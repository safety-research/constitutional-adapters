"""A gsm8k prompt-budget sidecar is THIS wave's or it is re-made: ONE currency predicate (tinycap_oos_prompt_budget.sidecar_currency —
tree_sha must be the wave sha; a named geometry_generation must be the current site record's) decides for BOTH readers, the in-job
make-if-absent (which moves a stale pair aside by name and re-makes it) and the artifact check (which refuses a stale sidecar by name).
2026-09-30: the pass-2 sidecars (tree 7065859…) refused every pass-3 gsm8k row in its first minute — `g8k_budget_make_if_absent` never
rewrote an existing pair while the check pinned the tree. The producer now stamps the geometry generation the window was fitted under."""
from __future__ import annotations

import json
import os
import re
import subprocess

import pytest
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import tinycap_oos_prompt_budget as B  # noqa: E402

COMMON = REPO / "scripts/sdfbase/gsm8k/g8k_common.sh"
VENV_PY = os.environ.get("SISTEER_TEST_PYTHON") or (str(REPO / ".venv/bin/python") if (REPO / ".venv/bin/python").exists() else sys.executable)
SHA_A, SHA_B = "a" * 40, "b" * 40


def test_the_currency_predicate_names_a_stale_tree_or_generation_and_admits_the_current_pair():
    assert B.sidecar_currency({"tree_sha": SHA_A}, wave_sha=SHA_A, current_generation=None) == (True, "current")
    ok, why = B.sidecar_currency({"tree_sha": SHA_B}, wave_sha=SHA_A, current_generation=None)
    assert not ok and f"tree_sha '{SHA_B}' != the wave sha {SHA_A}" in why
    ok, why = B.sidecar_currency({"tree_sha": SHA_A, "geometry_generation": "g1"}, wave_sha=SHA_A, current_generation="g2")
    assert not ok and "geometry_generation 'g1' != the current site record's 'g2'" in why
    assert B.sidecar_currency({"tree_sha": SHA_A, "geometry_generation": "g1"}, wave_sha=SHA_A, current_generation="g1") == (True, "current")
    # a legacy sidecar (no generation) is current on the tree alone; an unknown current generation never fails a stamped one
    assert B.sidecar_currency({"tree_sha": SHA_A}, wave_sha=SHA_A, current_generation="g2")[0]
    assert B.sidecar_currency({"tree_sha": SHA_A, "geometry_generation": "g1"}, wave_sha=SHA_A, current_generation=None)[0]
    # no site record configured -> no generation (the sidecar carries None), never an exception
    assert B.site_geometry_generation("qwen35_4b_demo") is None or isinstance(B.site_geometry_generation("qwen35_4b_demo"), str)
    assert B.site_geometry_generation(None) is None
    side = B.fit_sidecar({"arm": "on", "cap": 8192, "max": 2000, "n": 200, "model": "m", "revision": "r", "chat_template_hash": "h", "tree_sha": SHA_A, "p99": 1900},
                         task="t", control="gsm8k", geometry_eval="capability:gen_long_thinking", max_model_len=10752, system_prompt_cell=None,
                         system_prompt=None, geometry_generation="g9")
    assert side["geometry_generation"] == "g9" and side["tree_sha"] == SHA_A and side["fits"] is True


def _make_if_absent(tmp: Path, sidecar: dict | None, *, wave_sha: str) -> tuple[subprocess.CompletedProcess, list[str], Path]:
    """Run the REAL g8k_budget_make_if_absent (sourced from g8k_common.sh) with its job environment stubbed: the core is a no-op, the
    producer is a recording fake that writes an artifact + sidecar pinned to the --tree-sha it was given."""
    tmp.mkdir(parents=True, exist_ok=True)
    budget = tmp / "budgets/m/none/8192/oos_prompt_budget.json"
    if sidecar is not None:
        budget.parent.mkdir(parents=True, exist_ok=True)
        budget.write_text("{}\n"); (budget.parent / (budget.name + ".fit.json")).write_text(json.dumps(sidecar))
    rec = tmp / "producer.argv"
    fake = tmp / "fake_cpy"
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_RECORD"\nout=""; tree=""; prev=""\nfor a in "$@"; do [ "$prev" = "--out" ] && out="$a"; [ "$prev" = "--tree-sha" ] && tree="$a"; prev="$a"; done\n'
                    'mkdir -p "$(dirname "$out")"; echo "{}" > "$out"; printf \'{"tree_sha": "%s", "fits": true}\\n\' "$tree" > "$out.fit.json"\n')
    fake.chmod(0o755)
    script = f"""
set -u
REPO_DIR={REPO}; WAVE_SHA={wave_sha}; KEY=m; CELL=none; RUNG=8192; G8K_GEOMETRY_EVAL=capability:gen_long_thinking; G8K_THINKING=on
G8K_INSPECT_TASK=gsm8k_paired200; CFG=configs/x.yaml; SERVE_DIR={tmp}/serve; G8K_SERVE_SUBDIR=short; CPY={fake}
BUDGET_SRC={budget}
export FAKE_RECORD={rec}
g8k_core(){{ :; }}; g8k_py(){{ echo {VENV_PY}; }}; f22_log(){{ echo "[log] $*"; }}; f22_fatal(){{ echo "[fatal] $*" >&2; exit 1; }}
source <(sed -n '/^g8k_budget_make_if_absent()/,/^}}/p' {COMMON}; sed -n '/^_g8k_budget_make_locked()/,/^}}/p' {COMMON})
g8k_budget_make_if_absent
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/scripts"})
    calls = rec.read_text().splitlines() if rec.exists() else []
    return r, calls, budget


def _make_if_absent_script(tmp: Path, budget: Path, fake: Path, rec: Path, *, wave_sha: str, job: str) -> str:
    return f"""
set -u
REPO_DIR={REPO}; WAVE_SHA={wave_sha}; KEY=m; CELL=none; RUNG=8192; G8K_GEOMETRY_EVAL=capability:gen_long_thinking; G8K_THINKING=on
G8K_INSPECT_TASK=gsm8k_paired200; CFG=configs/x.yaml; SERVE_DIR={tmp}/serve_{job}; G8K_SERVE_SUBDIR=short; CPY={fake}; SLURM_JOB_ID={job}
BUDGET_SRC={budget}
export FAKE_RECORD={rec}
g8k_core(){{ :; }}; g8k_py(){{ echo {VENV_PY}; }}; f22_log(){{ echo "[log $SLURM_JOB_ID] $*"; }}; f22_fatal(){{ echo "[fatal $SLURM_JOB_ID] $*" >&2; exit 1; }}
source <(sed -n '/^g8k_budget_make_if_absent()/,/^}}/p' {COMMON}; sed -n '/^_g8k_budget_make_locked()/,/^}}/p' {COMMON})
g8k_budget_make_if_absent
"""


def test_two_concurrent_rows_on_one_stale_budget_pair_re_make_it_once_under_the_lock_and_never_publish_a_half_pair(tmp_path):
    """The check -> archive -> re-make path had no lock, while ONE budget pair per (key, class, rung) is shared by every
    same-class row of the matrix (12 gemma rows): concurrent jobs on a new wave sha would both read the stale pair, race the renames and
    run two producers over the same `<path>.tmp`. The whole path now runs under an exclusive flock on <pair>.lock, archival names are
    unique (utc + ns + job id), and the lock is released only after the complete new pair validates: two concurrent tasks -> ONE producer
    run, one superseded pair, both read a fresh complete pair, no half pair or .tmp left."""
    tmp = tmp_path; tmp.mkdir(exist_ok=True)
    budget = tmp / "budgets/m/none/8192/oos_prompt_budget.json"; budget.parent.mkdir(parents=True)
    budget.write_text("{}\n"); (budget.parent / (budget.name + ".fit.json")).write_text(json.dumps({"tree_sha": SHA_B, "fits": True}))
    rec = tmp / "producer.argv"
    fake = tmp / "fake_cpy"     # a SLOW producer: writes the artifact, sleeps, then the sidecar — a reader without the lock would see a half pair
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_RECORD"\nout=""; tree=""; prev=""\nfor a in "$@"; do [ "$prev" = "--out" ] && out="$a"; [ "$prev" = "--tree-sha" ] && tree="$a"; prev="$a"; done\n'
                    'mkdir -p "$(dirname "$out")"; echo "{}" > "$out"; sleep 2; printf \'{"tree_sha": "%s", "fits": true}\\n\' "$tree" > "$out.fit.json"\n')
    fake.chmod(0o755)
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/scripts"}
    procs = [subprocess.Popen(["bash", "-c", _make_if_absent_script(tmp, budget, fake, rec, wave_sha=SHA_A, job=j)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
             for j in ("1001", "1002")]
    outs = [p.communicate(timeout=300) for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    calls = rec.read_text().splitlines()
    assert len(calls) == 1 and f"--tree-sha {SHA_A}" in calls[0], calls                        # ONE producer run for the shared pair
    assert json.load(open(str(budget) + ".fit.json"))["tree_sha"] == SHA_A and budget.read_text().strip() == "{}"
    names = sorted(p.name for p in budget.parent.iterdir())
    superseded = [n for n in names if ".superseded." in n]
    assert len(superseded) == 2 and all(re.search(r"\.superseded\.\d{8}T\d{6}\.\d{9}Z\.job100[12]$", n) for n in superseded), names   # unique: utc + ns + job
    assert not [n for n in names if n.endswith(".tmp")] and (budget.parent / (budget.name + ".lock")).exists(), names
    logs = "\n".join(o[0] for o in outs)
    assert logs.count("moving the pair aside") == 1, logs                                          # the second task waited and read the fresh pair


def test_make_if_absent_supersedes_a_pair_pinned_to_another_tree_by_name_and_keeps_this_waves(tmp_path):
    # a pair pinned to ANOTHER tree: moved aside by name, the producer re-makes it at the wave sha
    r, calls, budget = _make_if_absent(tmp_path / "stale", {"tree_sha": SHA_B, "fits": True}, wave_sha=SHA_A)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "is not this wave's" in r.stdout and f"tree_sha '{SHA_B}' != the wave sha {SHA_A}" in r.stdout and "moving the pair aside" in r.stdout, r.stdout
    aside = sorted(p.name for p in budget.parent.iterdir() if ".superseded." in p.name)
    assert len(aside) == 2 and all(re.search(r"\.superseded\.\d{8}T\d{6}\.\d{9}Z\.job\w+$", n) for n in aside), aside   # unique: utc + ns + job
    assert len(calls) == 1 and f"--tree-sha {SHA_A}" in calls[0] and "--offline" in calls[0], calls
    assert json.load(open(str(budget) + ".fit.json"))["tree_sha"] == SHA_A
    # THIS wave's pair: honoured as before — nothing moved, the producer not run
    r, calls, budget = _make_if_absent(tmp_path / "current", {"tree_sha": SHA_A, "fits": True}, wave_sha=SHA_A)
    assert r.returncode == 0 and calls == [] and not [p for p in budget.parent.iterdir() if ".superseded." in p.name], r.stdout + r.stderr
    # no pair at all: made in-job (the original path)
    r, calls, budget = _make_if_absent(tmp_path / "absent", None, wave_sha=SHA_A)
    assert r.returncode == 0 and len(calls) == 1 and "no prompt-budget artifact" in r.stdout, r.stdout + r.stderr


def test_both_shell_readers_call_the_one_predicate():
    src = COMMON.read_text()
    make = src.split("_g8k_budget_make_locked(){", 1)[1].split("\n}\n", 1)[0]
    check = src.split("g8k_budget_artifact(){", 1)[1].split("\n}\n", 1)[0]
    assert "flock -x 9" in make and 'mv -n "$BUDGET_SRC.fit.json"' in make and "%N" in make            # exclusive per-pair lock; unique archival names
    assert "gen = B.site_geometry_generation(sys.argv[3])" in make and "B.sidecar_currency(side, wave_sha=sys.argv[2], current_generation=gen)" in make
    assert "gen = B.site_geometry_generation(key)" in check and "ok, why = B.sidecar_currency(d, wave_sha=sha, current_generation=gen)" in check
    assert 'tree_sha {d.get' not in check                     # the old hand-rolled tree pin is gone: one predicate, two readers


def test_a_revoked_site_record_makes_the_currency_undecidable_by_name_never_current_on_the_tree(tmp_path, monkeypatch):
    """Lane B's residual on this commit: during a marker-only REVOKED window (the record archived, `<model>.REVOKED.json` standing where it
    stood) the stamp used to read as "no record" and the sidecar was called current on the tree alone. The loader's refusal now propagates
    as a NAMED undecidability (SidecarCurrencyUndecidable), which both shell readers turn into a refusal by name; a directory with neither a
    record nor a marker is the genuine no-record case (None); a readable record yields its result_generation."""
    from sisteer.capeval.serving import geometry as G
    d = tmp_path / "geometry"; d.mkdir()
    monkeypatch.setenv(G.SITE_GEOMETRY_DIR_ENV, str(d))
    assert B.site_geometry_generation("qwen35_4b_demo") is None                          # neither record nor marker: no record
    G.revoked_marker_path(d / "qwen35_4b_demo.json").write_text(json.dumps({"model_key": "qwen35_4b_demo", "generation": "g" * 64,
                                                                             "revoked_utc": "2026-01-01T00:00:00Z", "reason": "reopened", "by": "test"}))
    with pytest.raises(B.SidecarCurrencyUndecidable, match="site record revoked/unverified — sidecar currency undecidable"):
        B.site_geometry_generation("qwen35_4b_demo")
    # both shell readers name it: the in-job producer's heredoc prints UNDECIDABLE and the shell refuses; the artifact check files it as a problem
    src = COMMON.read_text()
    make = src.split("_g8k_budget_make_locked(){", 1)[1].split("\n}\n", 1)[0]; check = src.split("g8k_budget_artifact(){", 1)[1].split("\n}\n", 1)[0]
    assert "except B.SidecarCurrencyUndecidable as exc:" in make and 'print("UNDECIDABLE " + str(exc))' in make and "UNDECIDABLE*) f22_fatal" in make
    assert "except B.SidecarCurrencyUndecidable as exc:" in check and "problems.append(str(exc))" in check
