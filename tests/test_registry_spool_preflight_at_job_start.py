"""static-reg-fix-6: the registry-destination preflight is SHARED by every emitting entry point and runs
BEFORE the installed helper is sourced — in `f22_registry_state` (f22_init's job-start `running` emission) as well as in
`f22_registry_register`. static-reg-fix-5 put the check at register only, so under a typo'd inherited SISTEER_REGISTRY_SPOOL the sbatch
order f22_init -> f22_resolve_config -> register sourced the helper FIRST at job start: the helper preserved the typo and `mkdir -p`d it,
the job-start state went into the typo'd directory, and only then did register refuse. Now, in that same order:

* a typo'd inherited spool with a submit wrapper's pending file is refused AT JOB START (the state call), NOTHING is created — not the
  typo'd directory, not a state record — and the identity is never exported; register is never reached;
* a relative inherited spool is refused at job start too, nothing inside the checkout;
* a whitespace-only inherited value counts as unset: the helper resolves the destination, the state is filed, register accepts the pending file;
* the good spool flows through both calls (state, then register's early return on the wrapper's pending file);
* a helper that resolves a NON-destination absolute spool is refused by the state emitter by name (the resolved-value check register has);
* common.sh carries the preflight ONCE, and both entry points call it before their `source` of the helper.

Through the REAL shell: common.sh sourced in a synthetic checkout with the REAL installed helper (scripts/registry/registry_env.sh)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, registry_emitting_env  # noqa: E402
from test_registry_spool_destination import _install_helper, _wrapper_pending  # noqa: E402
from test_static_registration_gated_pairs import COMMON, SHA, TAG, _checkout, _run_dirs  # noqa: E402

JOB = "4848"
TYPO_FRAG = "(inherited from the submitting environment) is not the registry destination"


#: the identity the installed helper derives its shared root from: a name no host has a `$SHARED` root for, so 'the helper names
#: no spool off-cluster' holds on the cluster gate too (test_registry_spool_destination.NO_SHARED_ROOT_USER; 2026-10-08 gate)
NO_SHARED_ROOT_USER = "sisteer-test-no-shared-root"


def _job_start_then_register(tmp_path: Path, *, spool: str | None, pending: bool = True, helper_body: str | None = None):
    """The sbatch's order, through the real functions: f22_registry_state running (f22_init :444) -> f22_registry_register (f22_resolve_config)."""
    co = _checkout(tmp_path, TAG); shared = tmp_path / "shared"; (shared / "registry").mkdir(parents=True, exist_ok=True)
    hp = _install_helper(shared)
    if helper_body is not None:
        hp.write_text(helper_body)
    pend = _wrapper_pending(shared, JOB) if pending else shared / "registry/pending" / f"{JOB}.json"
    pend_before = pend.read_text() if pend.exists() else None
    script = textwrap.dedent(f"""
        set -u
        cd {co}
        export REPO_DIR={co} SHARED={shared}
        export USER={NO_SHARED_ROOT_USER}
        {'export SISTEER_REGISTRY_SPOOL="' + spool + '"' if spool is not None else "unset SISTEER_REGISTRY_SPOOL"}
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
        export SLURM_JOB_ID={JOB} SLURM_JOB_NAME=f22s-{TAG}-sb0 WAVE_SHA={SHA} SLURMD_NODENAME=TESTHOST SLURM_JOB_STATE=RUNNING
        export KEY={TAG} VIEW=main TAG={TAG} CFG=configs/sdfbase_full22_{TAG}.yaml F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        source {COMMON}
        f22_registry_state running "${{SLURM_JOB_STATE:-RUNNING}}" || true
        echo "AFTER_STATE SPOOL=${{SISTEER_REGISTRY_SPOOL:-}}"
        f22_registry_register || true
        echo "RC=$? IDENT=${{SISTEER_REGISTRY_RUN_IDENTITY:-}} SPOOL=${{SISTEER_REGISTRY_SPOOL:-}}"
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    for k in list(env):
        if k.startswith("F22_REGISTRY_") or k == "SISTEER_REGISTRY_SPOOL":
            env.pop(k)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)
    return r, shared, co, pend, pend_before, hp


def _refused_at_job_start(r, *frags: str) -> str:
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "AFTER_STATE" not in r.stdout and "RC=" not in r.stdout and "self-registration REFUSED" in out, out
    for f in frags:
        assert f in out, (f, out)
    return out


def test_a_typod_inherited_spool_with_a_wrapper_pending_file_is_refused_at_job_start_and_creates_nothing(tmp_path):
    r, shared, co, pend, before, hp = _job_start_then_register(tmp_path, spool=f"{shared_of(tmp_path)}/registrry/records")
    _refused_at_job_start(r, f"SISTEER_REGISTRY_SPOOL='{shared}/registrry/records' {TYPO_FRAG} {shared}/registry/records")
    assert not (shared / "registrry").exists()                                        # the helper never ran: its mkdir -p never created the typo
    assert not _run_dirs(shared / "registry/records") and not (shared / "registry/records").exists()   # no job-start state filed anywhere
    assert pend.exists() and pend.read_text() == before                              # the wrapper's identity untouched, never exported


def shared_of(tmp_path: Path) -> Path:
    return tmp_path / "shared"


def test_a_relative_inherited_spool_is_refused_at_job_start_with_nothing_inside_the_checkout(tmp_path):
    r, shared, co, pend, before, hp = _job_start_then_register(tmp_path, spool="registry/records")
    _refused_at_job_start(r, "SISTEER_REGISTRY_SPOOL='registry/records' is a relative path (inherited from the submitting environment), not the absolute registry destination")
    assert not (co / "registry").exists() and not (shared / "registry/records").exists()


def test_a_whitespace_inherited_value_counts_as_unset_at_job_start_and_register_then_refuses_as_fix_5_has_it(tmp_path):
    """Spaces are unset: the state emitter trims, the installed helper names no spool off-cluster (this host), so the job-start state is
    simply not filed (return 0, nothing created, nothing named by spaces) and register refuses the unresolved spool — the fix-5 door."""
    r, shared, co, pend, before, hp = _job_start_then_register(tmp_path, spool="   ")
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "AFTER_STATE SPOOL=\n" in r.stdout and "RC=" not in r.stdout, out
    assert f"SISTEER_REGISTRY_SPOOL is unset or empty after sourcing {hp} (the helper names no spool)" in out, out
    assert not (co / "   ").exists() and not any(p.name.strip() == "" for p in co.iterdir()) and not (shared / "registry/records").exists()
    assert pend.read_text() == before


def test_the_good_spool_flows_through_both_calls_state_filed_then_the_wrapper_registration_accepted(tmp_path):
    r, shared, co, pend, before, hp = _job_start_then_register(tmp_path, spool=f"{shared_of(tmp_path)}/registry/records")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and f"AFTER_STATE SPOOL={shared}/registry/records" in r.stdout, out      # the preflight passed, the state call ran
    assert f"RC=0 IDENT={pend} SPOOL={shared}/registry/records" in r.stdout, out                    # register accepted the wrapper's registration (the early return)
    assert "self-registration REFUSED" not in out and pend.read_text() == before


def test_a_helper_that_resolves_a_non_destination_spool_is_refused_by_the_state_emitter_by_name(tmp_path):
    shared = shared_of(tmp_path)
    r, shared, co, pend, before, hp = _job_start_then_register(tmp_path, spool=None, helper_body=f"export SISTEER_REGISTRY_SPOOL={shared}/absent_root/records\n")
    _refused_at_job_start(r, f"SISTEER_REGISTRY_SPOOL='{shared}/absent_root/records' (after sourcing {hp}) is not the registry destination {shared}/registry/records",
                          "the job-start 'running' state would be filed outside the harvester's spool")
    assert not (shared / "absent_root").exists()


def test_common_sh_carries_the_preflight_once_and_both_entry_points_call_it_before_sourcing_the_helper():
    src = COMMON.read_text()
    assert src.count("f22_registry_spool_preflight(){") == 1 and src.count(TYPO_FRAG) == 1          # the refusal text lives in the preflight alone
    state = src[src.index("f22_registry_state(){"):src.index("f22_registry_register(){")]
    assert state.index("f22_registry_spool_preflight") < state.index('source "$SHARED/scripts/registry/registry_env.sh"')
    assert "(after sourcing $SHARED/scripts/registry/registry_env.sh) is not the registry destination" in state
    reg = src[src.index("f22_registry_register(){"):src.index("f22_registry_refused(){")]
    assert reg.index("f22_registry_spool_preflight") < reg.index('[ -r "$helper" ] && source "$helper"')
    assert reg.count("f22_registry_spool_preflight") == 1 and 'local spool_dest="${SHARED%/}/registry/records"' in reg
    assert subprocess.run(["bash", "-n", str(COMMON)], capture_output=True).returncode == 0
