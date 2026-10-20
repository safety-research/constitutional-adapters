"""A job KIND that registers cannot run with an unresolved registry spool. common.sh
f22_registry_register used to RETURN SUCCESS when $SHARED/scripts/registry/registry_env.sh was absent or SISTEER_REGISTRY_SPOOL stayed unset —
above the fatal registration branch — so a sampled 8192 / 16384 job with a missing helper or an empty spool selected
its gated pairs, "registered" nothing and went to the standup and GPU work unregistered (the CLI's own no-spool
exit 2 unreachable). Now, once the kind is known to register (the evals table; the paper tags' mmluF / padstat / probe kinds opt out BY KIND
there, as before), an unresolved spool is f22_registry_refused -> f22_fatal -> exit 1 BEFORE the deferral and before any standup. Through the
REAL shell: common.sh's register with the helper missing, with the helper naming an empty spool, deferred, for a non-registering kind, and
with the helper resolving the spool; the REAL target_sampled.sbatch under the gated-pairs harness' cluster stubs with the helper missing."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, registry_emitting_env  # noqa: E402
from test_static_registration_gated_pairs import COMMON, SBATCH, SHA, STUBS, TAG, _checkout, _run_dirs  # noqa: E402


def _helper(shared: Path, body: str | None) -> Path:
    h = shared / "scripts/registry/registry_env.sh"
    if body is not None:
        h.parent.mkdir(parents=True, exist_ok=True); h.write_text(body)
    return h


def _register_sh(tmp_path: Path, *, job_name: str, evals: str | None, spool: bool, helper: str | None, extra: str = "") -> tuple[subprocess.CompletedProcess, Path, Path, Path]:
    """common.sh f22_registry_register FOR REAL, with the spool resolved (or not) the way a cluster job resolves it: the exported variable, or the
    shared helper the function sources, or neither."""
    co = _checkout(tmp_path, TAG); shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True, exist_ok=True)
    hp = _helper(shared, helper)
    script = textwrap.dedent(f"""
        set -u
        cd {co}
        export REPO_DIR={co} SHARED={shared}
        {"export SISTEER_REGISTRY_SPOOL=" + str(shared / "registry/records") if spool else "unset SISTEER_REGISTRY_SPOOL"}
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
        export SLURM_JOB_ID=4343 SLURM_JOB_NAME={job_name} WAVE_SHA={SHA} SLURMD_NODENAME=TESTHOST
        export KEY={TAG} VIEW=main TAG={TAG} CFG=configs/sdfbase_full22_{TAG}.yaml F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        {"export F22_REGISTRY_EVALS=" + repr(evals) if evals is not None else "unset F22_REGISTRY_EVALS"}
        {extra}
        source {COMMON}
        f22_registry_register || true
        echo "RC=$? IDENT=${{SISTEER_REGISTRY_RUN_IDENTITY:-}}"
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    for k in list(env):
        if k.startswith("F22_REGISTRY_") or k == "SISTEER_REGISTRY_SPOOL":
            env.pop(k)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)
    return r, shared / "registry/pending/4343.json", shared / "registry/records", hp


def test_a_registering_kind_with_no_spool_is_fatal_before_anything_whether_the_helper_is_absent_or_names_none(tmp_path):
    # the helper absent and the variable unset: fatal, by name — `|| true` cannot mask it (f22_fatal exits the job's shell)
    r, pend, spool, hp = _register_sh(tmp_path / "a", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=None)
    out = r.stdout + r.stderr                                                                                   # f22_fatal logs through f22_log (stdout)
    assert r.returncode == 1 and "RC=" not in r.stdout, out
    assert f"FATAL: registry: self-registration REFUSED for job 4343 (f22s-{TAG}-sb0) — the registry helper {hp} is absent and SISTEER_REGISTRY_SPOOL is unset — no spool to register into" in out, out
    assert "a job whose runs do not exist in the registry does not run (D984 / D1265)" in out and not pend.exists() and not _run_dirs(spool)
    # the helper present but naming no spool (an empty export): fatal, naming the helper
    r, pend, spool, hp = _register_sh(tmp_path / "b", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper='export SISTEER_REGISTRY_SPOOL=""\n')
    assert r.returncode == 1 and f"SISTEER_REGISTRY_SPOOL is unset or empty after sourcing {hp} (the helper names no spool)" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not pend.exists() and not _run_dirs(spool)
    # a phase that DEFERS its registration to its gate fails at the START, not at the gate: it could not register there either
    r, pend, spool, hp = _register_sh(tmp_path / "c", job_name=f"f22s-{TAG}-se0", evals="pasttense grok420", spool=False, helper=None, extra="export F22_REGISTRY_DEFER=static_esc")
    assert r.returncode == 1 and "the registry helper" in r.stdout + r.stderr and "deferred to the phase's gate" not in r.stdout + r.stderr, r.stdout + r.stderr
    assert not pend.exists() and not _run_dirs(spool)


def test_a_kind_that_does_not_register_opts_out_by_kind_and_a_helper_that_names_the_spool_still_registers(tmp_path):
    # a kind the evals table does not map (the paper tags' mmluF / padstat / probe jobs): the notice names the kind, nothing is fatal, no spool needed
    r, pend, spool, hp = _register_sh(tmp_path / "a", job_name="f22-mmluF-x", evals=None, spool=False, helper=None)
    assert r.returncode == 0 and "RC=0 IDENT=\n" in r.stdout + "\n", r.stdout + r.stderr
    assert "job kind 'f22-mmluF-x' declares no evals (not in f22_registry_evals_for_job; set F22_REGISTRY_EVALS); nothing registered" in r.stderr and "FATAL" not in r.stdout + r.stderr
    assert not pend.exists() and not _run_dirs(spool)
    # the helper RESOLVES the spool for a registering kind whose environment carries none: the job registers (the cluster's normal path)
    shared = tmp_path / "b" / "shared"
    r, pend, spool, hp = _register_sh(tmp_path / "b", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=f"export SISTEER_REGISTRY_SPOOL={shared}/registry/records\n")
    assert r.returncode == 0 and f"RC=0 IDENT={pend}" in r.stdout, r.stdout + r.stderr
    assert pend.exists() and _run_dirs(spool)
    # the order is the contract: the kind's opt-out, then the spool, then the deferral, then the config — never a success exit on "no spool"
    src = COMMON.read_text(); body = src[src.index("f22_registry_register(){"):src.index("f22_registry_refused(){")]
    assert '[ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || return 0' not in body
    i_kind = body.index("declares no evals"); i_trim = body.index("| sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"); i_spool = body.index('if [ -z "$SISTEER_REGISTRY_SPOOL" ]; then')
    i_abs = body.index("is not an absolute path"); i_dir = body.index('[ -d "${SHARED%/}" ] ||'); i_mk = body.index('mkdir -p "$SISTEER_REGISTRY_SPOOL" ||'); i_defer = body.index('if [ -n "${F22_REGISTRY_DEFER:-}" ]'); i_cfg = body.index('f22_registry_refused "no resolved config')
    assert i_kind < i_trim < i_spool < i_abs < i_dir < i_mk < i_defer < i_cfg, (i_kind, i_trim, i_spool, i_abs, i_dir, i_mk, i_defer, i_cfg)   # trim, empty, absolute, the storage root exists, mkdir — all before the deferral
    assert body.count("f22_registry_refused \"the registry helper $helper is absent") == 1 and 'source "$helper"' in body


def test_the_real_sampled_sbatch_with_the_helper_missing_exits_1_before_any_gpu_work(tmp_path):
    """The REAL target_sampled.sbatch (phase static_esc_t0, the 8192 rung) under the gated-pairs harness' cluster stubs, with NO registry helper and
    NO spool in the environment: the job-start registration call is fatal — exit 1, no window check, no standup (the stub's exit 97 never runs),
    nothing pending, nothing in the spool."""
    co = _checkout(tmp_path, TAG)
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True, exist_ok=True)
    stub_dir = shared / "sisteer-eval-repo" / SHA / "scripts/sdfbase/full22/sampled"; stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "common_sampled.sh").write_text(STUBS)
    marks = tmp_path / "marks.txt"; wave = tmp_path / "wave"
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    for k in list(env):
        if k.startswith("F22_REGISTRY_") or k == "SISTEER_REGISTRY_SPOOL":
            env.pop(k)
    env.update({"SISTEER_SHARED_ROOT": str(shared), "HARNESS_REPO": str(REPO), "HARNESS_CHECKOUT": str(co), "HARNESS_WAVE": str(wave), "HARNESS_MARKS": str(marks),
                "HARNESS_AT_CAP": "pasttense|none=9,pasttense|prompted__safety_refusal=9,grok420|none=9,grok420|prompted__safety_refusal=9", "HARNESS_GATE": "",
                "REPO_DIR": str(co), "SHARED": str(shared),
                "SISTEER_REGISTRY_SUBSTRATE": "slurm", "SISTEER_REGISTRY_CLUSTER_LABEL": "cluster_a", "SLURM_JOB_ID": "55003", "SLURM_JOB_NAME": f"f22s-{TAG}-se0",
                "SLURMD_NODENAME": "TESTHOST", "CUDA_VISIBLE_DEVICES": "0",   # a synthetic value for the script's `set -u` read; no GPU is touched
                "F22_ROSTER": "configs/sdfbase_full22_roster.yaml", "F22_SERVING_PATH": "lora_native"})
    assert not (shared / "scripts/registry/registry_env.sh").exists()
    r = subprocess.run(["bash", str(SBATCH), TAG, "static_esc_t0", SHA, "9", "", "main"], capture_output=True, text=True, env=env, timeout=300, cwd=str(co))
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert f"FATAL: registry: self-registration REFUSED for job 55003 (f22s-{TAG}-se0) — the registry helper {shared}/scripts/registry/registry_env.sh is absent and SISTEER_REGISTRY_SPOOL is unset" in out, out
    assert not marks.exists() or marks.read_text() == "", (marks.read_text() if marks.exists() else "")     # no window check, no standup
    assert "STANDUP" not in out and "gate-selected pair" not in out                                        # nothing selected, nothing stood up
    assert not (shared / "registry/pending/55003.json").exists() and not _run_dirs(shared / "registry/records")


def test_a_whitespace_relative_or_parentless_spool_is_fatal_like_an_empty_one_and_a_padded_or_fresh_valid_path_registers(tmp_path):
    """`-z` alone let a WHITESPACE-ONLY spool count as resolved (RecordWriter would have written into
    a relative directory named by spaces inside the checkout while pending/<job>.json was produced). The value is trimmed, must be an absolute
    path whose PARENT directory exists — refused by name otherwise, before the deferral. The spool directory itself need not pre-exist
    (records.write_records makedirs the run chain on the first record; a fresh cluster root names a spool nothing has created yet) — it is
    created by the check, and benign surrounding whitespace around a valid path is trimmed, not refused."""
    # whitespace-only from the helper: the empty case's refusal
    r, pend, spool, hp = _register_sh(tmp_path / "a", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper='export SISTEER_REGISTRY_SPOOL="   "\n')
    assert r.returncode == 1 and f"SISTEER_REGISTRY_SPOOL is unset or empty after sourcing {hp} (the helper names no spool)" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not pend.exists() and not _run_dirs(spool)
    # whitespace-only from the environment, no helper: the absent-helper refusal (the trimmed value is empty)
    r, pend, spool, hp = _register_sh(tmp_path / "b", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=None, extra='export SISTEER_REGISTRY_SPOOL="  \t "')
    assert r.returncode == 1 and f"the registry helper {hp} is absent and SISTEER_REGISTRY_SPOOL is unset" in r.stdout + r.stderr, r.stdout + r.stderr
    # a relative path: refused by name, before the deferral
    r, pend, spool, hp = _register_sh(tmp_path / "c", job_name=f"f22s-{TAG}-se0", evals="pasttense grok420", spool=False, helper="export SISTEER_REGISTRY_SPOOL=registry/records\n",
                                      extra="export F22_REGISTRY_DEFER=static_esc")
    out = r.stdout + r.stderr
    assert r.returncode == 1 and f"SISTEER_REGISTRY_SPOOL='registry/records' is not an absolute path (after sourcing {hp})" in out and "deferred to the phase's gate" not in out, out
    assert not pend.exists() and not _run_dirs(spool)
    # an absolute path under the shared root that is not the destination (fix-4 read it as "a parent that does not exist"; under static-reg-fix-5
    # the destination rule names it first): refused by name, nothing created
    shared = tmp_path / "d" / "shared"
    r, pend, spool, hp = _register_sh(tmp_path / "d", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=f"export SISTEER_REGISTRY_SPOOL={shared}/absent_root/records\n")
    assert r.returncode == 1 and f"SISTEER_REGISTRY_SPOOL='{shared}/absent_root/records' (after sourcing {hp}) is not the registry destination {shared}/registry/records" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not pend.exists() and not _run_dirs(spool) and not (shared / "absent_root").exists()
    # the STORAGE ROOT itself absent on this node (not mounted here): refused by name, nothing created — static-reg-fix-5 narrowed fix-4's
    # parent check to the root: the destination is pinned, and $SHARED/registry is the registration's own to create (the fold proof's fresh fake
    # cluster registers its first job on a shared root with no registry/ yet)
    un = tmp_path / "d2" / "unmounted"
    r, pend, spool, hp = _register_sh(tmp_path / "d2", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=None, extra=f"export SHARED={un} SISTEER_REGISTRY_SPOOL={un}/registry/records")
    assert r.returncode == 1 and f"SISTEER_REGISTRY_SPOOL='{un}/registry/records': the storage root {un} does not exist on this node (the helper {un}/scripts/registry/registry_env.sh is absent; the value is the environment's) — not mounted here; no spool to register into" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not un.exists() and not pend.exists()
    # an absolute path whose parent exists but which nothing has created yet (a fresh cluster root): registers, and the check creates it
    shared = tmp_path / "f" / "shared"
    r, pend, spool, hp = _register_sh(tmp_path / "f", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=f"export SISTEER_REGISTRY_SPOOL={shared}/registry/records\n",
                                      extra=f"rmdir {shared}/registry/records")                        # static-reg-fix-5: the fresh path is the DESTINATION not yet created
    assert r.returncode == 0 and f"RC=0 IDENT={pend}" in r.stdout, r.stdout + r.stderr
    assert pend.exists() and spool.is_dir() and _run_dirs(spool)                                      # created by the check, the records in it
    # another directory under registry/ is not a spool the harvester pulls: refused by name under the destination rule (static-reg-fix-5); the
    # installed helper's eager mkdir may have created it, but nothing is registered into it
    r, pend, spool, hp = _register_sh(tmp_path / "g", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=f"export SISTEER_REGISTRY_SPOOL={tmp_path / 'g' / 'shared'}/registry/fresh\n")
    assert r.returncode == 1 and f"SISTEER_REGISTRY_SPOOL='{tmp_path / 'g' / 'shared'}/registry/fresh' (after sourcing {hp}) is not the registry destination {tmp_path / 'g' / 'shared'}/registry/records" in r.stdout + r.stderr, r.stdout + r.stderr
    assert not pend.exists() and not _run_dirs(tmp_path / "g" / "shared" / "registry" / "fresh")
    # a valid path padded with whitespace: trimmed, registers (the cluster's normal path survives a sloppy export)
    shared = tmp_path / "e" / "shared"
    r, pend, spool, hp = _register_sh(tmp_path / "e", job_name=f"f22s-{TAG}-sb0", evals=None, spool=False, helper=f'export SISTEER_REGISTRY_SPOOL="  {shared}/registry/records  "\n')
    assert r.returncode == 0 and f"RC=0 IDENT={pend}" in r.stdout, r.stdout + r.stderr
    assert pend.exists() and _run_dirs(spool)


def test_the_real_sampled_sbatch_with_a_whitespace_spool_exits_1_before_any_gpu_work(tmp_path):
    """The REAL target_sampled.sbatch (static_esc_t0) under the gated-pairs harness' cluster stubs with SISTEER_REGISTRY_SPOOL set to spaces and
    no helper: the job-start registration call is fatal — exit 1, no window check, no standup, nothing pending, nothing in the spool."""
    co = _checkout(tmp_path, TAG)
    shared = tmp_path / "shared"; (shared / "registry/records").mkdir(parents=True, exist_ok=True)
    stub_dir = shared / "sisteer-eval-repo" / SHA / "scripts/sdfbase/full22/sampled"; stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "common_sampled.sh").write_text(STUBS)
    marks = tmp_path / "marks.txt"; wave = tmp_path / "wave"
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    for k in list(env):
        if k.startswith("F22_REGISTRY_"):
            env.pop(k)
    env.update({"SISTEER_SHARED_ROOT": str(shared), "HARNESS_REPO": str(REPO), "HARNESS_CHECKOUT": str(co), "HARNESS_WAVE": str(wave), "HARNESS_MARKS": str(marks),
                "HARNESS_AT_CAP": "pasttense|none=9,pasttense|prompted__safety_refusal=9,grok420|none=9,grok420|prompted__safety_refusal=9", "HARNESS_GATE": "",
                "REPO_DIR": str(co), "SHARED": str(shared), "SISTEER_REGISTRY_SPOOL": "   ",
                "SISTEER_REGISTRY_SUBSTRATE": "slurm", "SISTEER_REGISTRY_CLUSTER_LABEL": "cluster_a", "SLURM_JOB_ID": "55004", "SLURM_JOB_NAME": f"f22s-{TAG}-se0",
                "SLURMD_NODENAME": "TESTHOST", "CUDA_VISIBLE_DEVICES": "0",   # a synthetic value for the script's `set -u` read; no GPU is touched
                "F22_ROSTER": "configs/sdfbase_full22_roster.yaml", "F22_SERVING_PATH": "lora_native"})
    r = subprocess.run(["bash", str(SBATCH), TAG, "static_esc_t0", SHA, "9", "", "main"], capture_output=True, text=True, env=env, timeout=300, cwd=str(co))
    out = r.stdout + r.stderr
    assert r.returncode == 1 and f"FATAL: registry: self-registration REFUSED for job 55004 (f22s-{TAG}-se0) — the registry helper {shared}/scripts/registry/registry_env.sh is absent and SISTEER_REGISTRY_SPOOL is unset" in out, out
    assert (not marks.exists() or marks.read_text() == "") and "STANDUP" not in out and "gate-selected pair" not in out
    assert not (shared / "registry/pending/55004.json").exists() and not _run_dirs(shared / "registry/records")
    assert not list(co.glob("*/**/pending")) and not (co / "   ").exists()                              # nothing written inside the checkout
