"""static-reg-fix-5: the spool a registry job writes is the ONE ingestion destination the harvester pulls — exactly
<shared>/registry/records — validated BEFORE the installed registry_env.sh is sourced (that helper preserves a nonempty inherited value and
`mkdir -p`s it, so a typo'd or node-local inherited spool used to create its own parent and pass fix-4's parent check) and BEFORE a submit
wrapper's pending file is accepted (an identity on disk proves a registration happened, not that THIS job's emitter writes into the spool).

Through the REAL common.sh and the INSTALLED helper (the repo's scripts/registry/registry_env.sh copied to <shared>/scripts/registry/, as the
clusters carry it — never a plain export standing in for it):

* an inherited typo under a writable shared root (`<shared>/registrry/records`) is refused by name, and NO directory is created;
* an inherited absolute node-local path (`/tmp/.../records`) is refused by name, nothing created;
* an inherited relative spool is refused before the helper (nothing created inside the checkout); a whitespace-only one counts as unset and
  meets the existing empty-spool refusal (the helper resolves no spool on a host without a shared root);
* a wrapper-created pending file (sbatch_registered.sh's shape) + a bad inherited spool is refused AT JOB START, before the identity is exported;
  the same pending file with the good spool is accepted (the early return), and the good spool without a pending file registers;
* a kind that declares no evals and has no pending file still exits "nothing registered" without touching the spool;
* the function body keeps the order: the kind / pending decision, the trim, the destination check, `source "$helper"`, the resolved-value
  checks, the pending-file acceptance, the deferral.

Synthetic job ids and roots only (the real seeds, read)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, registry_emitting_env  # noqa: E402
from test_static_registration_gated_pairs import COMMON, SHA, TAG, _checkout, _run_dirs  # noqa: E402

HELPER_SRC = REPO / "scripts" / "registry" / "registry_env.sh"
JOB = "4747"


def _install_helper(shared: Path) -> Path:
    h = shared / "scripts/registry/registry_env.sh"; h.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HELPER_SRC, h); h.chmod(0o755)
    return h


def _wrapper_pending(shared: Path, job: str) -> Path:
    """The pending file sbatch_registered.sh leaves (emit_cli register --plan: {"run_id", "job_id", "job_name", "registered_utc", "plan"})."""
    pend = shared / "registry/pending" / f"{job}.json"; pend.parent.mkdir(parents=True, exist_ok=True)
    pend.write_text(json.dumps({"run_id": "sdfbase_full22:xstest:000000000000:1111111122222222", "job_id": job, "job_name": f"f22s-{TAG}-sb0",
                                "registered_utc": "2026-10-03T00:00:00Z", "plan": {"campaign": "sdfbase_full22", "eval_id": "xstest"}}, indent=1))
    return pend


#: the identity the installed helper derives its shared root from (`$SHARED` on either cluster): a name no host has a root
#: for, so 'the helper names no spool' holds when the suite runs on the cluster too (its node mounts the shared storage, and
#: the suite's account there, modalsuiteuser, HAS a root — tests/test_stage2_run.py's s2_init created it)
NO_SHARED_ROOT_USER = "sisteer-test-no-shared-root"


def _run(tmp_path: Path, *, job_name: str, spool: str | None, evals: str | None = None, pending: bool = False, extra: str = ""):
    co = _checkout(tmp_path, TAG); shared = tmp_path / "shared"; (shared / "registry").mkdir(parents=True, exist_ok=True)
    hp = _install_helper(shared)
    pend = _wrapper_pending(shared, JOB) if pending else shared / "registry/pending" / f"{JOB}.json"
    script = textwrap.dedent(f"""
        set -u
        cd {co}
        export REPO_DIR={co} SHARED={shared}
        export USER={NO_SHARED_ROOT_USER}
        {'export SISTEER_REGISTRY_SPOOL="' + spool + '"' if spool is not None else "unset SISTEER_REGISTRY_SPOOL"}
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
        export SLURM_JOB_ID={JOB} SLURM_JOB_NAME={job_name} WAVE_SHA={SHA} SLURMD_NODENAME=TESTHOST
        export KEY={TAG} VIEW=main TAG={TAG} CFG=configs/sdfbase_full22_{TAG}.yaml F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        {"export F22_REGISTRY_EVALS=" + repr(evals) if evals is not None else "unset F22_REGISTRY_EVALS"}
        {extra}
        source {COMMON}
        f22_registry_register || true
        echo "RC=$? IDENT=${{SISTEER_REGISTRY_RUN_IDENTITY:-}} SPOOL=${{SISTEER_REGISTRY_SPOOL:-}}"
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    for k in list(env):
        if k.startswith("F22_REGISTRY_") or k == "SISTEER_REGISTRY_SPOOL":
            env.pop(k)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)
    return r, shared, co, pend, hp


def _refused(r, *frags: str) -> str:
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "RC=" not in r.stdout and "self-registration REFUSED" in out, out
    for f in frags:
        assert f in out, (f, out)
    assert "a job whose runs do not exist in the registry does not run (D984 / D1265)" in out
    return out


def test_an_inherited_typo_under_the_shared_root_is_refused_by_name_before_the_installed_helper_creates_anything(tmp_path):
    r, shared, co, pend, hp = _run(tmp_path, job_name=f"f22s-{TAG}-sb0", spool=f"{shared_of(tmp_path)}/registrry/records")
    _refused(r, f"SISTEER_REGISTRY_SPOOL='{shared}/registrry/records' (inherited from the submitting environment) is not the registry destination {shared}/registry/records",
             "the harvester pulls <shared>/registry/records only", "nothing created")
    assert not (shared / "registrry").exists() and not pend.exists() and not (shared / "registry/records").exists()   # the helper never ran its mkdir
    assert hp.read_text() == HELPER_SRC.read_text()                                                                   # the real helper, byte for byte


def shared_of(tmp_path: Path) -> Path:
    return tmp_path / "shared"


def test_an_inherited_node_local_or_relative_spool_is_refused_before_the_helper_and_a_whitespace_one_counts_as_unset(tmp_path):
    local = tmp_path / "node_local" / "records"; local.parent.mkdir(parents=True)
    r, shared, co, pend, hp = _run(tmp_path / "a", job_name=f"f22s-{TAG}-sb0", spool=str(local))
    _refused(r, f"SISTEER_REGISTRY_SPOOL='{local}' (inherited from the submitting environment) is not the registry destination {shared}/registry/records")
    assert not local.exists() and not pend.exists()
    r, shared, co, pend, hp = _run(tmp_path / "b", job_name=f"f22s-{TAG}-sb0", spool="registry/records")
    _refused(r, "SISTEER_REGISTRY_SPOOL='registry/records' is a relative path (inherited from the submitting environment), not the absolute registry destination", "nothing created")
    assert not (co / "registry").exists() and not pend.exists()                                                      # nothing inside the checkout
    # whitespace-only: unset for the purpose — the installed helper resolves no spool on a host without a shared root, so the empty refusal fires
    r, shared, co, pend, hp = _run(tmp_path / "c", job_name=f"f22s-{TAG}-sb0", spool="   \t ")
    _refused(r, f"SISTEER_REGISTRY_SPOOL is unset or empty after sourcing {hp} (the helper names no spool)")
    assert not pend.exists()


def test_a_wrapper_registered_job_with_a_bad_inherited_spool_is_refused_at_job_start_before_the_identity_is_exported(tmp_path):
    r, shared, co, pend, hp = _run(tmp_path / "a", job_name="mmluF-x-main", spool=f"{shared_of(tmp_path / 'a')}/registrry/records", pending=True)   # a kind with NO evals of its own: the pending file makes it a registry job
    _refused(r, f"SISTEER_REGISTRY_SPOOL='{shared}/registrry/records' (inherited from the submitting environment) is not the registry destination")
    assert "IDENT=" not in r.stdout and pend.exists() and not (shared / "registrry").exists()                          # the identity never exported; the wrapper's file untouched
    r, shared, co, pend, hp = _run(tmp_path / "b", job_name=f"f22s-{TAG}-sb0", spool="   ", pending=True)
    _refused(r, f"SISTEER_REGISTRY_SPOOL is unset or empty after sourcing {hp}")
    assert "IDENT=" not in r.stdout
    # the good spool: the wrapper's identity is accepted (the early return), nothing re-registered
    r, shared, co, pend, hp = _run(tmp_path / "c", job_name=f"f22s-{TAG}-sb0", spool=f"{shared_of(tmp_path / 'c')}/registry/records", pending=True)
    assert r.returncode == 0 and f"RC=0 IDENT={pend} SPOOL={shared}/registry/records" in r.stdout, r.stdout + r.stderr
    assert not _run_dirs(shared / "registry/records")                                                                 # nothing written by this job start


def test_the_good_destination_registers_and_a_non_registry_kind_without_a_pending_file_opts_out(tmp_path):
    r, shared, co, pend, hp = _run(tmp_path / "a", job_name=f"f22s-{TAG}-sb0", spool=f"{shared_of(tmp_path / 'a')}/registry/records")
    assert r.returncode == 0 and f"RC=0 IDENT={pend}" in r.stdout and pend.exists() and _run_dirs(shared / "registry/records"), r.stdout + r.stderr
    # a trailing slash is the same destination
    r, shared, co, pend, hp = _run(tmp_path / "b", job_name=f"f22s-{TAG}-sb0", spool=f"{shared_of(tmp_path / 'b')}/registry/records/")
    assert r.returncode == 0 and f"RC=0 IDENT={pend}" in r.stdout, r.stdout + r.stderr
    # a kind with no evals and no pending file: the notice, no spool touched even when the inherited one is wrong
    r, shared, co, pend, hp = _run(tmp_path / "c", job_name="mmluF-x-main", spool=f"{shared_of(tmp_path / 'c')}/registrry/records")
    assert r.returncode == 0 and "RC=0 IDENT=" in r.stdout and "declares no evals (not in f22_registry_evals_for_job; set F22_REGISTRY_EVALS); nothing registered" in r.stderr, r.stdout + r.stderr
    assert "FATAL" not in r.stdout + r.stderr and not (shared / "registrry").exists()


def test_the_register_body_validates_the_spool_before_the_helper_and_before_the_pending_file():
    src = COMMON.read_text()
    body = src[src.index("f22_registry_register(){"):]
    body = body[:body.index("\n}\n")]
    i_kind = body.index("declares no evals"); i_trim = body.index("| sed 's/^[[:space:]]*//; s/[[:space:]]*$//')")
    # static-reg-fix-6: the inherited-value check (the trim + the destination rule) is the SHARED preflight f22_registry_spool_preflight,
    # defined above and called here before the helper is sourced; the register body's own trim is the post-source one
    i_dest1 = body.index("f22_registry_spool_preflight"); i_src = body.index('source "$helper"')
    pre = src[src.index("f22_registry_spool_preflight(){"):src.index("f22_registry_state(){")]
    assert "(inherited from the submitting environment) is not the registry destination" in pre and pre.index("| sed 's/^[[:space:]]*//") < pre.index("(inherited from")
    i_empty = body.index('if [ -z "$SISTEER_REGISTRY_SPOOL" ]; then'); i_dir = body.index('[ -d "${SHARED%/}" ] ||')
    i_dest2 = body.index("is not the registry destination $spool_dest — the harvester pulls <shared>/registry/records only, so records written there would never reach the registry\"")
    i_mk = body.index('mkdir -p "$SISTEER_REGISTRY_SPOOL" ||'); i_pend = body.index('if [ -f "$pend" ]; then export SISTEER_REGISTRY_RUN_IDENTITY="$pend"; return 0; fi')
    i_defer = body.index('if [ -n "${F22_REGISTRY_DEFER:-}" ]')
    assert i_kind < i_dest1 < i_src < i_trim < i_empty < i_dest2 < i_dir < i_mk < i_pend < i_defer, (i_kind, i_dest1, i_src, i_trim, i_empty, i_dest2, i_dir, i_mk, i_pend, i_defer)   # The storage-root check sits after the destination, before mkdir
    assert body.count('source "$helper"') == 1 and 'local spool_dest="${SHARED%/}/registry/records"' in body
    assert 'if [ -z "$evals_csv" ] && [ ! -f "$pend" ]; then' in body                                                  # a registry job = declares evals OR wrapper-registered
