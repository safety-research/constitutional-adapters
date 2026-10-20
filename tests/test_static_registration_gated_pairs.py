"""The sampled lane's rung phases register ONLY the pairs their per-pair gate selects, and
a refused registration stops the job.

(1) The 8,192 rung draws a (family, cell) pair only when the base run holds an at-cap row for it; the 16,384 rung only the pairs the ladder
    gate ESCALATEs. Both gates run after f22_resolve_config, so those phases DEFER the job-start registration (f22s_parse_phase exports
    F22_REGISTRY_DEFER; common.sh returns without registering) and register exactly the selected pairs themselves — once, before the window
    check / standup — through F22_REGISTRY_PAIRS -> emit_cli --pairs -> jobplan.selected_pairs. A phase whose gate selects nothing exits 0
    before any registration: zero rung runs. Before: 838299 registered 8 static runs at the 8,192 rung for a gate that drew none of them.
(2) emit_cli register --from-config exits 2 on a refusal (3 on a failure) instead of 0; f22_registry_register, once the job declares evals,
    is fatal on everything that ends without the pending file (f22_registry_refused -> exit 1 before any venv / standup); a job kind that
    declares no evals keeps its notice. jobplan requires a NONEMPTY seeded ladder holding the cap for a rung registration.
Fixtures: the REAL seeds, the REAL CLI, the REAL common.sh / common_sampled.sh and the REAL target_sampled.sbatch under stubs for the cluster
(venvs, serving, the at-cap counter, the ladder gate); synthetic ids and TESTHOST only."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, read_spool, registry_emitting_env  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import emit_cli, synth  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402
from test_registry_close_stores import OBJECTS, SEEDS, TAG, _records, _seeded_registry  # noqa: E402
from test_static_registration_per_rung import LADDER, _plans, _register, _world  # noqa: E402

RUNG8 = ["--producer-key", f"sdfbase_full22_{TAG}_sampled_t0_cap8192", "--token-cap", "8192", "--rung-index", "1", "--ladder", LADDER,
         "--pass-label", "ladder_rung", "--trial-index", "0"]
COMMON = REPO / "scripts/sdfbase/full22/common.sh"
SAMPLED = REPO / "scripts/sdfbase/full22/sampled/common_sampled.sh"
SBATCH = REPO / "scripts/sdfbase/full22/sampled/target_sampled.sbatch"
SHA = synth.code_sha("d")   # a derived forty-hex sha: a repeated-digit placeholder is refused as a run code_sha (WAVE_SHA below)


def _run_dirs(spool: Path) -> set:
    camp = spool / "sdfbase_full22"
    return set(os.listdir(camp)) if camp.is_dir() else set()


# ------------------------------------------------------------------------------------------------------------------ the CLI / jobplan
def test_pairs_restrict_the_registration_to_the_gate_selected_pairs(tmp_path, monkeypatch, capsys):
    root, spool = _world(tmp_path, monkeypatch)
    before = _run_dirs(spool)
    pend = _register(tmp_path, root, job="50001", job_name=f"f22s-{TAG}-se0", evals="pasttense,grok420",
                     flags=[*RUNG8, "--pairs", "grok420:none, pasttense:prompted__safety_refusal,"])   # spaces and a trailing comma tolerated
    err = capsys.readouterr().err
    plans = _plans(pend)
    assert set(plans) == {("grok420", "none"), ("pasttense", "prompted__safety_refusal")}, set(plans)      # exactly the two selected of 2x2
    for p in plans.values():
        assert p["extra"]["registered_pairs"] == ["grok420:none", "pasttense:prompted__safety_refusal"]   # the selection on every run
        assert p["token_cap_tokens"] == 8192 and p["token_cap_rung_index"] == 1 and p["producer_run_key_legacy"].endswith("_cap8192")
    assert "gate-selected pairs: 2 of 2x2 declared (grok420:none, pasttense:prompted__safety_refusal)" in err, err
    assert "2 run(s) registered at job start" in err
    ids = {r["run_id"] for r in json.loads(pend.read_text())["runs"]}
    new_dirs = _run_dirs(spool) - before                                                                   # two run directories, no more
    assert len(ids) == len(new_dirs) == 2 and all(any(d.endswith(rid.split(":")[-1]) for d in new_dirs) for rid in ids), (ids, new_dirs)
    # without --pairs the same job registers every declared eval x object (the base draw's shape)
    full = _register(tmp_path, root, job="50002", job_name=f"f22s-{TAG}-sb0", evals="pasttense,grok420", flags=RUNG8)
    assert len(_plans(full)) == 4 and all("registered_pairs" not in p["extra"] for p in _plans(full).values())


def test_a_pair_outside_the_declared_evals_x_objects_is_refused_by_name_with_exit_2(tmp_path, monkeypatch, capsys):
    root, spool = _world(tmp_path, monkeypatch)
    before = _run_dirs(spool)
    cases = [
        ("renellm:none", "names eval 'renellm', which this job does not declare"),
        ("pasttense:nonsuch", "names object 'nonsuch', which the config does not enumerate"),
        ("pasttense", "a pair is `eval:object`"),
        ("", "names no (eval, object) pair"),
        (",", "names no (eval, object) pair"),
    ]
    for i, (pairs, frag) in enumerate(cases):
        pend = _register(tmp_path, root, job=f"5010{i}", job_name=f"f22s-{TAG}-se0", evals="pasttense,grok420", flags=[*RUNG8, "--pairs", pairs], expect_rc=2)
        err = capsys.readouterr().err
        assert not pend.exists() and "self-registration refused" in err and frag in err and "exit 2" in err, (pairs, err)
    assert _run_dirs(spool) == before                                                                        # nothing written, five times


def test_every_refusal_exits_2_a_failure_exits_3_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    root, spool = _world(tmp_path, monkeypatch)
    before = _run_dirs(spool)
    co = tmp_path / "checkout_x"; (co / "configs").mkdir(parents=True); os.symlink(SEEDS, co / "configs" / "registry")
    cfg = co / "configs" / f"sdfbase_full22_{TAG}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS)}))
    pend_dir = tmp_path / "pend_x"
    base = ["register", "--from-config", str(cfg), "--serving-path", "lora_native", "--campaign", "sdfbase_full22", "--job-id", "50200",
            "--pending-dir", str(pend_dir), "--code-sha", synth.code_sha("a"), "--substrate", "cluster_a", "--no-push"]
    spool_flag = ["--spool", str(spool)]   # EVERY call names the tmp spool: on a host whose default spool is the live registry, a call without one WRITES THERE
    # a missing fact
    assert emit_cli.main(base + spool_flag) == 2 and "missing --model-key" in capsys.readouterr().err
    # an unknown model key
    assert emit_cli.main(base + spool_flag + ["--model-key", "nonsuch", "--evals", "pasttense"]) == 2 and "self-registration refused" in capsys.readouterr().err
    # every pair skipped
    assert emit_cli.main(base + spool_flag + ["--model-key", TAG, "--evals", "nonsuch"]) == 2 and "no run could be registered" in capsys.readouterr().err
    # an unexpected failure inside the planner -> 3
    monkeypatch.setattr(JP, "build_plans", lambda **kw: (_ for _ in ()).throw(RuntimeError("synthetic planner failure")))
    assert emit_cli.main(base + spool_flag + ["--model-key", TAG, "--evals", "pasttense"]) == 3
    assert "self-registration failed (RuntimeError: synthetic planner failure)" in capsys.readouterr().err and "exit 3" in capsys.readouterr().err + " exit 3"
    monkeypatch.undo()
    # an unexpected failure outside the planner (main's belt) -> 3
    monkeypatch.setattr(emit_cli, "register_from_config", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("synthetic cli failure")))
    assert emit_cli.main(base + spool_flag + ["--model-key", TAG, "--evals", "pasttense"]) == 3 and "synthetic cli failure" in capsys.readouterr().err
    assert not pend_dir.exists() and _run_dirs(spool) == before
    assert emit_cli.EXIT_REFUSED == 2 and emit_cli.EXIT_FAILED == 3


def test_a_rung_registration_requires_the_instruments_nonempty_seeded_ladder(tmp_path, monkeypatch, capsys):
    root, spool = _world(tmp_path, monkeypatch)
    before = _run_dirs(spool)
    monkeypatch.setattr(JP, "_instrument_ladder", lambda inst: [])                     # an instrument that seeds no ladder
    pend = _register(tmp_path, root, job="50300", job_name=f"f22s-{TAG}-se0", evals="pasttense", flags=RUNG8, expect_rc=2)
    err = capsys.readouterr().err
    assert not pend.exists() and "seeds no cap ladder (token_cap_ladder_json)" in err and "no run could be registered" in err, err
    assert _run_dirs(spool) == before
    monkeypatch.undo()
    # the real seeded instrument ladders [4096, 8192, 16384]: every rung registers, a cap off the ladder does not
    assert JP._instrument_ladder({"token_cap_ladder_json": [4096, 8192, 16384]}) == [4096, 8192, 16384] and JP._instrument_ladder({}) == []
    ok = _register(tmp_path, root, job="50301", job_name=f"f22s-{TAG}-se0", evals="pasttense", flags=RUNG8)
    assert len(_plans(ok)) == 2


# ------------------------------------------------------------------------------------------------------------------ the shell: common.sh
def _checkout(tmp_path: Path, key: str = TAG) -> Path:
    """A synthetic checkout: the real seeds + the real src (symlinked), one wave config and the three sampled configs of trial 0 for `key`."""
    co = tmp_path / "checkout"; (co / "configs").mkdir(parents=True, exist_ok=True)
    if not (co / "configs" / "registry").exists():
        os.symlink(SEEDS, co / "configs" / "registry")
    if not (co / "src").exists():
        os.symlink(REPO / "src", co / "src")
    for name in (f"sdfbase_full22_{key}.yaml", f"sdfbase_full22_{key}_sampled_t0.yaml", f"sdfbase_full22_{key}_sampled_t0_cap8192.yaml",
                 f"sdfbase_full22_{key}_sampled_t0_cap16384.yaml"):
        (co / "configs" / name).write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS)}))
    return co


def _register_sh(tmp_path: Path, *, job_name: str, evals: str | None, extra: str = "", key: str = TAG) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """common.sh f22_registry_register FOR REAL (the real CLI over the real seeds), as f22_resolve_config calls it."""
    co = _checkout(tmp_path, key); shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True, exist_ok=True)
    script = textwrap.dedent(f"""
        set -u
        cd {co}
        export REPO_DIR={co} SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
        export SLURM_JOB_ID=4242 SLURM_JOB_NAME={job_name} WAVE_SHA={SHA} SLURMD_NODENAME=TESTHOST
        export KEY={key} VIEW=main TAG={key} CFG=configs/sdfbase_full22_{key}.yaml F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        {"export F22_REGISTRY_EVALS=" + repr(evals) if evals is not None else "unset F22_REGISTRY_EVALS"}
        {extra}
        source {COMMON}
        f22_registry_register || true
        echo "RC=$? IDENT=${{SISTEER_REGISTRY_RUN_IDENTITY:-}}"
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    for k in list(env):
        if k.startswith("F22_REGISTRY_"):
            env.pop(k)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)
    return r, shared / "registry/pending/4242.json", shared / "registry/records"


def test_f22_registry_register_defers_when_told_registers_the_pairs_and_is_fatal_when_refused(tmp_path):
    # (a) deferred: nothing registered, nothing fatal, the job continues to its gate
    r, pend, spool = _register_sh(tmp_path / "a", job_name=f"f22s-{TAG}-se0", evals="pasttense grok420", extra="export F22_REGISTRY_DEFER=static_esc")
    assert r.returncode == 0 and "RC=0 IDENT=\n" in r.stdout + "\n" and "deferred to the phase's gate (F22_REGISTRY_DEFER=static_esc)" in r.stderr, r.stdout + r.stderr
    assert not pend.exists() and not _run_dirs(spool) and "register:" not in r.stderr.replace("register: deferred", "")
    # (b) the deferred re-entry: F22_REGISTRY_PAIRS -> exactly those runs, `running` filed for each
    r, pend, spool = _register_sh(tmp_path / "b", job_name=f"f22s-{TAG}-se0", evals="pasttense grok420",
                                  extra="export F22_REGISTRY_PAIRS=grok420:none,pasttense:prompted__safety_refusal F22_REGISTRY_TOKEN_CAP=8192 F22_REGISTRY_RUNG_INDEX=1 "
                                        f"F22_REGISTRY_LADDER={LADDER} F22_REGISTRY_PASS_LABEL=ladder_rung F22_REGISTRY_RUN_SUFFIX=_sampled_t0_cap8192 F22_REGISTRY_TRIAL_INDEX=0")
    assert r.returncode == 0 and f"RC=0 IDENT={pend}" in r.stdout, r.stdout + r.stderr
    runs = {(x["route"]["eval_id"], x["route"]["object"]): x for x in json.loads(pend.read_text())["runs"]}
    assert set(runs) == {("grok420", "none"), ("pasttense", "prompted__safety_refusal")}
    assert all(x["plan"]["producer_run_key_legacy"] == f"sdfbase_full22_{TAG}_sampled_t0_cap8192" and x["plan"]["token_cap_tokens"] == 8192 for x in runs.values())
    states = [p["state"] for rt, p in read_spool(spool) if rt == "run_state"]
    assert states.count("running") == 2 and set(states) == {"running"}, states
    # (c) a refused registration (an unknown model key: the planner refuses) is FATAL — exit 1, "RC=" never reached, no pending file
    r, pend, spool = _register_sh(tmp_path / "c", job_name="f22s-nonsuch-se0", evals="pasttense", key="nonsuch")
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "RC=" not in r.stdout and not pend.exists() and not _run_dirs(spool), out
    assert "FATAL: registry: self-registration REFUSED for job 4242 (f22s-nonsuch-se0)" in out and "exited 2 and wrote no pending file" in out and "self-registration refused" in r.stderr, out
    # (d) a pair outside the declared evals: refused by name, fatal
    r, pend, spool = _register_sh(tmp_path / "d", job_name=f"f22s-{TAG}-se0", evals="pasttense", extra="export F22_REGISTRY_PAIRS=grok420:none")
    out = r.stdout + r.stderr
    assert r.returncode == 1 and not pend.exists() and "names eval 'grok420', which this job does not declare" in out and "FATAL: registry: self-registration REFUSED" in out, out
    # (e) the per-cell narrowing refusal (a cell the overlay does not enumerate): fatal before the CLI is reached
    r, pend, spool = _register_sh(tmp_path / "e", job_name=f"f22s-{TAG}-se0", evals="pasttense", extra="export F22_REGISTRY_CELLS=prompted__zzz")
    out = r.stdout + r.stderr
    assert r.returncode == 1 and not pend.exists() and "cannot narrow" in out and "FATAL: registry: self-registration REFUSED" in out and "self-registration refused" not in out, out
    # (f) a job KIND that declares no evals is not a registry job: the notice stands, the job continues (the table is the declaration)
    r, pend, spool = _register_sh(tmp_path / "f", job_name="weird-job", evals=None)
    assert r.returncode == 0 and "RC=0 IDENT=\n" in r.stdout + "\n" and "declares no evals" in r.stderr and not pend.exists(), r.stdout + r.stderr
    # the shell text: the register line carries the pair flags ahead of the rung flags (the per-rung test pins the rung tail), the belt is named
    common = COMMON.read_text()
    assert '${pair_flags[@]+"${pair_flags[@]}"} ${rung_flags[@]+"${rung_flags[@]}"} >/dev/null' in common
    # the refusal call sites by their MESSAGE, not a bare substring count (static-reg-fix-6 attempt 2: the count also read the definition and a
    # comment mention, and re-tripped on every moved block). The spool's: preflight x3 (static-reg-fix-5, carved out by fix-6: SHARED unset, the
    # inherited value not the destination / relative), the job-start state check after sourcing (fix-6, NEW), register-time x6 (fix-3/-4/-5:
    # unset after sourcing, helper absent, relative, not the destination, storage root absent, mkdir); then CFG, KEY, the per-cell narrowing, the CLI exit.
    calls = re.findall(r'f22_registry_refused "([^"]*)"', common)
    expected = ("SHARED is unset",
                "(inherited from the submitting environment) is not the registry destination",
                "is a relative path (inherited from the submitting environment)",
                "(after sourcing $SHARED/scripts/registry/registry_env.sh) is not the registry destination",
                "is unset or empty after sourcing",
                "is absent and SISTEER_REGISTRY_SPOOL is unset",
                "is not an absolute path ($spool_src)",
                "($spool_src) is not the registry destination",
                "the storage root $SHARED does not exist on this node",
                "could not be created under $SHARED",
                "no resolved config",
                "KEY is unbound",
                "the job-independent producer key could not be formed",   # a malformed key stops the job
                "cannot narrow",
                "emit_cli register --from-config exited")
    assert len(calls) == len(expected) == 15, [c[:70] for c in calls]
    for frag in expected:
        assert sum(frag in c for c in calls) == 1, (frag, [c[:70] for c in calls])
    assert "f22_registry_refused(){" in common
    assert "|| true\n  if [ -f \"$pend\" ]; then" not in common                                              # the CLI's exit is read, never discarded


def test_f22s_parse_phase_defers_the_rung_phases_only():
    def parse(phase, preset=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("F22_REGISTRY_")}
        env.update(preset or {})
        r = subprocess.run(["bash", "-c", f'source {SAMPLED} && f22s_parse_phase "$1" && printf "%s" "${{F22_REGISTRY_DEFER-unset}}"', "_", phase],
                           cwd=REPO, env=env, capture_output=True, text=True)
        assert r.returncode == 0, (phase, r.stderr[-600:])
        return r.stdout.strip()
    assert parse("static_esc_t0") == "static_esc" and parse("static_esc16_t1:pasttense,grok420") == "static_esc16"
    assert parse("static_base_t0") == "unset" and parse("controls_t0") == "unset" and parse("gsm8k_t1") == "unset" and parse("capability_t0") == "unset"
    assert parse("static_esc_t0", {"F22_REGISTRY_DEFER": "preset"}) == "preset"                                   # a caller's value wins
    sampled = SAMPLED.read_text()
    assert "f22s_register_gated_pairs(){" in sampled and 'F22_REGISTRY_DEFER= F22_REGISTRY_PAIRS="$pairs" CFG="${WAVE_CFG:-$CFG}" f22_registry_register' in sampled


# ------------------------------------------------------------------------------------------------------------------ the sbatch, end to end under stubs
STUBS = r'''
# the REAL helpers of the checkout under test (common_sampled.sh sources common.sh), then the cluster stood in for: venvs, serving, the
# at-cap counter and the ladder gate come from the harness; the registration path (f22_registry_register -> emit_cli -> the seeds) is REAL
source "$HARNESS_REPO/scripts/sdfbase/full22/sampled/common_sampled.sh"
f22_full22_manifest(){ echo "configs/capeval/manifests/petri_full22_$1.yaml"; }
f22_init(){ REPO_DIR="$HARNESS_CHECKOUT"; WAVE_ROOT="$HARNESS_WAVE"; CLUSTER=cluster_a; mkdir -p "$WAVE_ROOT"; cd "$REPO_DIR" || exit 9; }
f22s_resolve_config(){   # the real one re-points CFG after f22_resolve_config, whose job-start registration call is mirrored here
  KEY="$1"; VIEW="${3:-main}"; TAG="$KEY"; WAVE_CFG="configs/sdfbase_full22_${TAG}.yaml"; CFG="$WAVE_CFG"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.yaml"
  f22_registry_register || true
  VIEW_LEGS=static; VIEW_WINDOW=""; VIEW_WINDOW_STATIC=""; VIEW_WINDOW_CONTROLS=""; VIEW_RUNG_WINDOW_2048=""; VIEW_RUNG_WINDOW_8192=""; VIEW_RUNG_WINDOW_16384=""; VIEW_CONTROLS=""; F22S_CONTROLS=()
  CFG="configs/sdfbase_full22_${TAG}_sampled_t$2.yaml"; CFG_ESC="configs/sdfbase_full22_${TAG}_sampled_t$2_cap8192.yaml"; CFG_ESC16="configs/sdfbase_full22_${TAG}_sampled_t$2_cap16384.yaml"
  PRIVATE_RUN_BASE="sdfbase_full22_${TAG}_sampled_t$2"; PRIVATE_RUN_ESC="${PRIVATE_RUN_BASE}_cap8192"; PRIVATE_RUN_ESC16="${PRIVATE_RUN_BASE}_cap16384"; SIBLING_TRIAL=$((1 - $2))
}
f22_stop_requested(){ return 1; }
f22_driver_venv(){ MPY=python3; }
f22_serving_venv(){ :; }
f22s_assert_vllm_mamba_kernels(){ return 0; }
f22s_check_config(){ ALL_CELLS=(${HARNESS_ALL_CELLS:-none prompted__safety_refusal}); }   # the CONFIG's explicit_cells (a steered-wave config lists no none)
f22_bank_view(){ :; }
f22s_detect_long_si(){ F22S_LONG_SI=""; }
f22_standup_extra_reset(){ :; }
f22_result(){ echo "RESULT $*"; }
f22_view_prompted_only(){ return 1; }
f22s_view_has_leg(){ return 0; }
f22s_at_cap_static(){   # $2/$3 = fam/cell -> the harness table "fam|cell=N,…" (HARNESS_AT_CAP)
  local kv; for kv in ${HARNESS_AT_CAP//,/ }; do [ "${kv%%=*}" = "$2|$3" ] && { echo "${kv#*=}"; return 0; }; done
  echo "harness: no at-cap entry for $2|$3" >&2; return 2
}
f22s_event(){ :; }
f22s_event_json(){ echo "{}"; }
f22s_write_ladder(){ :; }
f22s_prev_endpoint(){ echo "$HARNESS_WAVE/prev_endpoint.json"; }
f22s_static_gate(){   # $2/$3 = fam/cell -> the harness table "fam|cell=VERB ARG1 ARG2,…" (HARNESS_GATE), `,` inside a verdict spelled `;`
  local kv; for kv in ${HARNESS_GATE//,/ }; do [ "${kv%%=*}" = "$2|$3" ] && { echo "${kv#*=}" | tr ';' ' '; return 0; }; done
  echo "harness: no gate entry for $2|$3" >&2; return 2
}
f22s_gate_event(){ echo "{}"; }
f22s_gate_summary(){ echo "stub"; }
f22s_rung_window(){ echo 10240; }
_pending_now(){ [ -f "$SHARED/registry/pending/$SLURM_JOB_ID.json" ] && echo yes || echo no; }
f22_window_check(){ echo "WINDOW_CHECK pending=$(_pending_now)" >> "$HARNESS_MARKS"; }
f22_standup(){ echo "STANDUP pending=$(_pending_now)" >> "$HARNESS_MARKS"; exit 97; }   # the first GPU work: stop the script here, distinctly
'''


def _sbatch(tmp_path: Path, *, phase: str, at_cap: str, gate: str = "", key: str = TAG, all_cells: str | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    """Run the REAL target_sampled.sbatch for `phase` with the cluster stubbed and the registration path real."""
    co = _checkout(tmp_path, key)
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True, exist_ok=True)
    stub_dir = shared / "sisteer-eval-repo" / SHA / "scripts/sdfbase/full22/sampled"; stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "common_sampled.sh").write_text(STUBS)
    marks = tmp_path / "marks.txt"; wave = tmp_path / "wave"
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    for k in list(env):
        if k.startswith("F22_REGISTRY_"):
            env.pop(k)
    env.update({"SISTEER_SHARED_ROOT": str(shared), "HARNESS_REPO": str(REPO), "HARNESS_CHECKOUT": str(co), "HARNESS_WAVE": str(wave), "HARNESS_MARKS": str(marks),
                "HARNESS_AT_CAP": at_cap, "HARNESS_GATE": gate, "REPO_DIR": str(co), "SHARED": str(shared), "SISTEER_REGISTRY_SPOOL": f"{shared}/registry/records",
                **({"HARNESS_ALL_CELLS": all_cells} if all_cells is not None else {}),
                "SISTEER_REGISTRY_SUBSTRATE": "slurm", "SISTEER_REGISTRY_CLUSTER_LABEL": "cluster_a", "SLURM_JOB_ID": "55001", "SLURM_JOB_NAME": f"f22s-{key}-se0",
                "SLURMD_NODENAME": "TESTHOST", "CUDA_VISIBLE_DEVICES": "0",   # a synthetic value for the script's `set -u` read; no GPU is touched
                "F22_ROSTER": "configs/sdfbase_full22_roster.yaml", "F22_SERVING_PATH": "lora_native"})
    r = subprocess.run(["bash", str(SBATCH), key, phase, SHA, "9", "", "main"], capture_output=True, text=True, env=env, timeout=300, cwd=str(co))
    pend = shared / "registry/pending/55001.json"
    facts = {"marks": marks.read_text().splitlines() if marks.exists() else [], "pending": pend,
             "runs": ({(x["route"]["eval_id"], x["route"]["object"]) for x in json.loads(pend.read_text())["runs"]} if pend.exists() else set()),
             "spool": shared / "registry/records"}
    return r, facts


def test_the_8192_rung_registers_exactly_its_at_cap_pairs_before_the_standup(tmp_path):
    r, f = _sbatch(tmp_path, phase="static_esc_t0:pasttense,grok420",
                   at_cap="pasttense|none=3,pasttense|prompted__safety_refusal=0,grok420|none=0,grok420|prompted__safety_refusal=1")
    out = r.stdout + r.stderr
    assert r.returncode == 97, out                                                                             # stopped at the (stub) standup
    assert f["runs"] == {("pasttense", "none"), ("grok420", "prompted__safety_refusal")}, (f["runs"], out)     # the two promoted pairs, not the two unpromoted
    assert f["marks"] == ["WINDOW_CHECK pending=yes", "STANDUP pending=yes"], f["marks"]                     # registered BEFORE the window check and the standup
    assert r.stderr.count("deferred to the phase's gate (F22_REGISTRY_DEFER=static_esc)") == 1                 # the job-start call deferred, once
    assert "registering the 2 gate-selected pair(s) before any standup: grok420:prompted__safety_refusal,pasttense:none (D1265)" in out
    assert r.stderr.count("run(s) registered at job start") == 1 and "2 run(s) registered" in r.stderr         # ONE registration
    doc = json.loads(f["pending"].read_text())
    assert all(x["plan"]["producer_run_key_legacy"] == f"sdfbase_full22_{TAG}_sampled_t0_cap8192" and x["plan"]["token_cap_tokens"] == 8192
               and x["plan"]["token_cap_rung_index"] == 1 and x["plan"]["extra"]["config"] == f"configs/sdfbase_full22_{TAG}.yaml" for x in doc["runs"])   # the wave config, as the base phase
    states = [p["state"] for rt, p in read_spool(f["spool"]) if rt == "run_state"]
    assert states.count("running") == 2 and set(states) == {"running"}, states
    assert "RESULT sampled" in r.stdout and "exit=97" in r.stdout


def test_the_8192_rung_of_a_steered_wave_view_registers_and_draws_only_the_steered_at_cap_pairs_never_a_none_pair(tmp_path):
    """With `none` gone from the steered-wave sampled configs, the per-pair gate promotes STEERED at-cap pairs only —
    a base store holding at-cap rows for `none` must not be selected, registered or drawn: the gate iterates the CONFIG's cells (here the
    two steered cells, no none), so the at-cap counter is never even asked about none."""
    r, f = _sbatch(tmp_path, phase="static_esc_t0:pasttense,grok420", all_cells="prompted__safety_refusal",     # the one steered cell of this synthetic config (no none)
                   at_cap="pasttense|none=7,grok420|none=4,pasttense|prompted__safety_refusal=2,grok420|prompted__safety_refusal=0")
    out = r.stdout + r.stderr
    assert r.returncode == 97, out
    assert f["runs"] == {("pasttense", "prompted__safety_refusal")}, (f["runs"], out)                          # the steered at-cap pair only; none's 7 / 4 rows never selected
    assert "at-cap @4096 pasttense/none" not in out and "grok420/none" not in out                               # the counter was never asked about none
    assert "registering the 1 gate-selected pair(s) before any standup: pasttense:prompted__safety_refusal" in out
    assert f["marks"] == ["WINDOW_CHECK pending=yes", "STANDUP pending=yes"], f["marks"]
    doc = json.loads(f["pending"].read_text())
    assert [x["route"]["object"] for x in doc["runs"]] == ["prompted__safety_refusal"]                       # the ONE registered run is the steered pair's
    assert all(x["route"]["object"] != "none" and x["plan"]["producer_object_key_legacy"] != "none" for x in doc["runs"])   # no none run exists, whatever the base store held


def test_the_all_zero_8192_phase_registers_nothing_and_stands_nothing_up(tmp_path):
    r, f = _sbatch(tmp_path, phase="static_esc_t0:pasttense,grok420",
                   at_cap="pasttense|none=0,pasttense|prompted__safety_refusal=0,grok420|none=0,grok420|prompted__safety_refusal=0")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "no (family, cell) pair has at-cap rows at 4096" in out and "no standup" in out, out
    assert not f["pending"].exists() and f["runs"] == set() and f["marks"] == [] and not _run_dirs(f["spool"]), (f, out)
    assert r.stderr.count("deferred to the phase's gate") == 1 and "run(s) registered" not in out and "registering the" not in out


def test_a_refused_rung_registration_exits_1_before_any_gpu_work(tmp_path):
    r, f = _sbatch(tmp_path, phase="static_esc_t0:pasttense", at_cap="pasttense|none=3,pasttense|prompted__safety_refusal=2", key="nonsuch")
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "FATAL: registry: self-registration REFUSED for job 55001 (f22s-nonsuch-se0)" in out and "exited 2 and wrote no pending file" in out, out
    assert "pairs: pasttense:none,pasttense:prompted__safety_refusal" in out and "self-registration refused" in r.stderr
    assert f["marks"] == [] and not f["pending"].exists() and not _run_dirs(f["spool"]), (f, out)             # no window check, no standup, nothing written
    assert "RESULT sampled" in r.stdout and "rc=1" in r.stdout and "exit=1" in r.stdout                       # the RESULT line says so


def test_the_16384_rung_registers_exactly_the_gates_escalate_pairs_before_the_standup(tmp_path):
    r, f = _sbatch(tmp_path, phase="static_esc16_t0:pasttense,grok420",
                   at_cap="pasttense|none=3,pasttense|prompted__safety_refusal=0,grok420|none=2,grok420|prompted__safety_refusal=1",
                   gate="pasttense|none=ESCALATE;16384;items_a.json,grok420|none=EXIT;le_1pct,grok420|prompted__safety_refusal=ESCALATE;16384;items_b.json")
    out = r.stdout + r.stderr
    assert r.returncode == 97, out
    assert f["runs"] == {("pasttense", "none"), ("grok420", "prompted__safety_refusal")}, (f["runs"], out)     # the ESCALATE pairs; the EXIT and the never-reached pair have no run
    assert f["marks"] == ["WINDOW_CHECK pending=yes", "STANDUP pending=yes"], f["marks"]
    assert r.stderr.count("deferred to the phase's gate (F22_REGISTRY_DEFER=static_esc16)") == 1
    assert "registering the 2 gate-selected pair(s) before any standup: grok420:prompted__safety_refusal,pasttense:none (D1265)" in out
    doc = json.loads(f["pending"].read_text())
    assert all(x["plan"]["producer_run_key_legacy"] == f"sdfbase_full22_{TAG}_sampled_t0_cap16384" and x["plan"]["token_cap_tokens"] == 16384
               and x["plan"]["token_cap_rung_index"] == 2 for x in doc["runs"])


def test_the_sbatch_text_registers_after_the_gate_and_before_the_window_check_in_both_rung_phases():
    src = SBATCH.read_text()
    esc = src[src.index('elif [ "$PHASE_KIND" = static_esc ]; then'):src.index('elif [ "$PHASE_KIND" = static_esc16 ]; then')]
    esc16 = src[src.index('elif [ "$PHASE_KIND" = static_esc16 ]; then'):src.index('elif [ "$PHASE_KIND" = controls ]; then')]
    for body, call, before in ((esc, 'f22s_register_gated_pairs static_esc "$REG_PAIRS"', 'f22_window_check "$SERVE_DIR/static_esc"'),
                               (esc16, 'f22s_register_gated_pairs static_esc16 "$REG_PAIRS"', "WINDOW16=$(f22s_rung_window 16384)")):
        i_zero, i_stop, i_reg, i_win = body.index('if [ "$NEED" -eq 0 ]; then'), body.index('f22s_stop_requested && { f22_log "STOP present — not standing up"'), body.index(call), body.index(before)
        assert i_zero < i_stop < i_reg < i_win and body.index("f22_standup attacks") > i_reg, (call, i_zero, i_stop, i_reg, i_win)
        assert body.count(call) == 1
    assert 'for pc in "${!NBASE[@]}"; do if [ "${NBASE[$pc]}" -gt 0 ]; then' in esc and 'for pc in "${!ITEMS16[@]}"; do' in esc16
    assert esc.count('[ "${NBASE[$fam|$cell]}" -gt 0 ] || continue') == 2 and esc16.count('[ -n "${ITEMS16[$fam|$cell]:-}" ] || continue') == 2   # the review pins stand
