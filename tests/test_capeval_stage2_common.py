"""stage2_common.sh §3.3 extension: three defaulted variables, byte-identical
behavior when unset.

Runs the REAL script's ``s2_init`` in a stubbed bash world (fake $USER so
no real path is touched, and ``mkdir`` refusing the fake user's default
roots so that holds for root too; ``git``/``uv``/``scontrol``/``srun``
shell-function stubs; ``exit`` shadowed so the SHA-gate refusals are
observable without killing the harness) and asserts:

  * unset  -> S2_REPO_DIR, UV_PROJECT_ENVIRONMENT, the uv sync argv, and
    the bootstrap-lock name all equal today's hardcoded values;
  * set    -> all three take effect (repo gate + cd, venv path + per-venv
    lock name, extra sync args).

Peers' jobs source this file with the variables unset, so the "unset"
assertions are the no-regression contract for every live stage-2 job.

The merged s2_init also enforces the mandatory submit-time SHA pin: every
submission carries ``--comment=sha:<40-hex>``, read back via scontrol at
job start and matched against repo HEAD. The default scontrol stub
therefore answers with a validly-pinned record (in the cluster's REAL
rendering: leading indent + trailing space), and one test here holds the
gate fail-closed through THIS harness's capeval S2_REPO_DIR lane — the
full pin matrix (unreadable scontrol, multi-record collapse, array-task
scoping, HEAD-moved-past-pin) lives in test_stage2_run.py.
"""

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sbatch" / "stage2_common.sh"
FAKE_USER = "capevalxuser"  # never a real account: /home + shared-storage paths must not exist

# One valid 40-hex sha shared by every stub: the git stub reports it as HEAD,
# the scontrol stub carries it as the submit-time pin, and the
# EXPECTED_SHA fixtures record it — all three comparands agree unless a
# test deliberately breaks one.
STUB_SHA = "cafecafe" * 5

HARNESS = r"""
set -o pipefail
# Stubs so s2_init runs on a machine with no cluster mounts. Order
# matters only in that they exist before s2_init is CALLED; the script
# itself only defines functions at source time.
exit() { echo "__EXIT_CALLED__:${1:-0}"; }   # observe gate exits, keep going
git() { echo "$STUB_SHA"; }                   # stands in for rev-parse HEAD
# The fake user's default roots stay un-creatable whoever runs the harness. A
# non-root process is refused by the kernel (no /home/$USER, no per-user
# directory under a shared root to create under); root is not, so the refusal
# is the stub's -- mkdir's own message and rc -- and nothing under a directory
# named for the fake user is ever created on the machine running the test.
# Every other path (the tmp world the set-variable tests name) goes to the
# real mkdir.
mkdir() {
  local a
  for a in "$@"; do
    case "$a" in
      /*/"$USER"|/*/"$USER"/*)
        echo "mkdir: cannot create directory '$a': Permission denied" >&2
        return 1;;
    esac
  done
  command mkdir "$@"
}
# Default: a validly-pinned job record in the cluster's real rendering —
# leading indent and a TRAILING SPACE (D14). STUB_SCONTROL_LINES, when set,
# replaces the record wholesale (set-but-empty = a submission that carries
# no --comment pin at all).
scontrol() {
  if [ -n "${STUB_SCONTROL_LINES+x}" ]; then
    printf '%s\n' "$STUB_SCONTROL_LINES"
  else
    printf '   Comment=sha:%s \n' "$STUB_SHA"
  fi
}
srun() { :; }   # s2_assert_cu128's GPU-flavor probe: no GPU on this machine
uv() { echo "UV_ARGV:$*" >> "${UV_RECORD:-/dev/null}"; }
export SLURM_JOB_ID=0 SLURM_JOB_NAME=capeval_common_test
source "$STAGE2_COMMON"
s2_init 1 5
echo "S2_REPO_DIR_AFTER:$S2_REPO_DIR"
echo "UV_ENV_AFTER:$UV_PROJECT_ENVIRONMENT"
echo "PWD_AFTER:$PWD"
builtin exit 0
"""


def _project_root(tmp_path):
    return tmp_path / "shared" / "proj"


def _default_roots():
    """The fake user's directories a run must leave untouched: under /home, and under every shared root
    stage2_common.sh's per-account probe lists (read from the script, so the test names no site's storage; a script
    that takes its root from the job environment lists none)."""
    m = re.search(r'^([ \t]*)if \[ -d "/[^"$]+/\$USER" \]; then\n.*?^\1fi$', SCRIPT.read_text(), re.M | re.S)
    roots = re.findall(r'\[ -d "(/[^"$]+)/\$USER" \]', m.group(0)) if m else []
    return (f"/home/{FAKE_USER}",) + tuple(f"{r}/{FAKE_USER}" for r in roots)


def _run(tmp_path, extra_env):
    env = os.environ.copy()
    for k in ("S2_ROOT", "S2_REPO_ROOT", "S2_REPO_DIR", "S2_UV_ENV",
              "S2_UV_SYNC_ARGS", "S2_UV_EXTRAS", "ESCALATE_QOS",
              "UV_PROJECT_ENVIRONMENT", "STUB_SCONTROL_LINES", "UV_RECORD",
              "SISTEER_PROJECT_ROOT", "SISTEER_SHARED_ROOT", "SISTEER_CLUSTER_QOS"):
        env.pop(k, None)
    env["USER"] = FAKE_USER
    env["STAGE2_COMMON"] = str(SCRIPT)
    env["STUB_SHA"] = STUB_SHA
    # the cluster project directory and shared root every job gets from the front door (neutral tmp paths here)
    env["SISTEER_PROJECT_ROOT"] = str(_project_root(tmp_path))
    env["SISTEER_SHARED_ROOT"] = str(tmp_path / "shared" / "root")
    # the QoS a preempted job resubmits at, which its submitter sets and s2_init reads (a neutral name here)
    env["ESCALATE_QOS"] = "qos_nonpreempt"
    env.update(extra_env)
    proc = subprocess.run(
        ["bash", "-c", HARNESS], capture_output=True, text=True,
        env=env, cwd=str(tmp_path), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def _stdout_val(proc, key):
    for line in proc.stdout.splitlines():
        if line.startswith(key + ":"):
            return line[len(key) + 1:]
    raise AssertionError(f"{key} not in harness output:\n{proc.stdout}")


def _listing(root):
    """Every path under ``root`` (``root`` itself included) this process can see, or None when it is absent."""
    if not os.path.lexists(root):
        return None
    seen = {root}
    for dirpath, dirnames, filenames in os.walk(root):
        seen.update(os.path.join(dirpath, n) for n in dirnames + filenames)
    return seen


def test_all_unset_defaults_are_todays_hardcoded_values(tmp_path):
    # The defaults name the fake user's roots, which the harness's mkdir
    # refuses: nothing is created on this machine, whoever runs the test. A
    # root can already exist (shared storage keeps what an earlier run of the
    # unstubbed default made), so the claim is that THIS run adds nothing.
    roots = _default_roots()
    before = {root: _listing(root) for root in roots}
    proc = _run(tmp_path, {})
    for root in roots:
        assert _listing(root) == before[root], root
    # Default repo checkout + its sibling EXPECTED_SHA gate file:
    assert _stdout_val(proc, "S2_REPO_DIR_AFTER") == (
        f"{_project_root(tmp_path)}/repo")
    assert (f"FATAL: {_project_root(tmp_path)}/EXPECTED_SHA missing"
            in proc.stdout)
    assert f"Re-sync {_project_root(tmp_path)}/repo (D7)" in proc.stdout
    # Default venv:
    assert _stdout_val(proc, "UV_ENV_AFTER") == f"/home/{FAKE_USER}/venvs/proj"
    # Default bootstrap-lock name (surfaces in the failed-redirection
    # message, since the fake user's /home does not exist):
    assert f"/home/{FAKE_USER}/venvs/.proj.uv-sync.lock" in proc.stderr
    # cd target is the default checkout (fails on this machine -> the
    # observable is the exit-stub call after the failed cd):
    assert "__EXIT_CALLED__:1" in proc.stdout
    assert _stdout_val(proc, "PWD_AFTER") == str(tmp_path)


def test_unset_project_root_refuses_by_name(tmp_path):
    # With no seam variable set, the checkout root comes from SISTEER_PROJECT_ROOT, which the
    # front door exports for every job: a job without it refuses by name.
    env = {k: v for k, v in os.environ.items()
           if k not in ("S2_ROOT", "S2_REPO_ROOT", "S2_REPO_DIR", "SISTEER_PROJECT_ROOT")}
    env.update(USER=FAKE_USER, STAGE2_COMMON=str(SCRIPT), STUB_SHA=STUB_SHA,
               SISTEER_SHARED_ROOT=str(tmp_path / "shared" / "root"), ESCALATE_QOS="qos_nonpreempt")
    proc = subprocess.run(["bash", "-c", HARNESS], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=60)
    assert proc.returncode != 0
    assert ("SISTEER_PROJECT_ROOT: set SISTEER_PROJECT_ROOT, the cluster project directory "
            "(the front door exports it from the site config)") in proc.stderr
    assert "S2_REPO_DIR_AFTER" not in proc.stdout


def test_unset_sync_args_produce_todays_exact_uv_call(tmp_path):
    # Historical layout (<parent>/repo + sibling EXPECTED_SHA) with
    # S2_UV_SYNC_ARGS UNSET: the recorded uv argv must be byte-for-byte
    # today's hardcoded call — `--extra cu128` included, mandatory since
    # the torch cpu/cu128 flavor split moved torch out of the base deps
    # (a flavor-less sync installs NO torch) — the gate must pass via
    # the ${S2_REPO_DIR%/repo} derivation, and the lock name must be
    # today's.
    legacy = tmp_path / "legacy"
    (legacy / "repo").mkdir(parents=True)
    (legacy / "EXPECTED_SHA").write_text(STUB_SHA + "\n")
    venvs = tmp_path / "venvs"
    record = tmp_path / "uv_argv.txt"
    proc = _run(tmp_path, {
        "S2_REPO_DIR": str(legacy / "repo"),
        "S2_UV_ENV": str(venvs / "sisteer"),
        "UV_RECORD": str(record),
    })
    assert "FATAL" not in proc.stdout and "__EXIT_CALLED__" not in proc.stdout
    assert record.read_text() == "UV_ARGV:sync --frozen --extra cu128\n"
    assert (venvs / ".sisteer.uv-sync.lock").exists()
    assert _stdout_val(proc, "PWD_AFTER") == str(legacy / "repo")


def test_all_three_variables_take_effect(tmp_path):
    # capeval layout (checkout NOT named repo -> EXPECTED_SHA lives inside
    # it), own venv (per-venv lock name), extra sync args appended.
    checkout = tmp_path / "capeval-repo"
    checkout.mkdir()
    (checkout / "EXPECTED_SHA").write_text(STUB_SHA + "\n")
    venvs = tmp_path / "venvs"
    record = tmp_path / "uv_argv.txt"
    proc = _run(tmp_path, {
        "S2_REPO_DIR": str(checkout),
        "S2_UV_ENV": str(venvs / "sisteer-capeval"),
        "S2_UV_SYNC_ARGS": "--python 3.12 --group capeval",
        "UV_RECORD": str(record),
    })
    assert "FATAL" not in proc.stdout and "__EXIT_CALLED__" not in proc.stdout
    assert _stdout_val(proc, "S2_REPO_DIR_AFTER") == str(checkout)
    assert _stdout_val(proc, "UV_ENV_AFTER") == str(venvs / "sisteer-capeval")
    assert _stdout_val(proc, "PWD_AFTER") == str(checkout)
    assert record.read_text() == (
        "UV_ARGV:sync --frozen --extra cu128 --python 3.12 --group capeval\n")
    assert (venvs / ".sisteer-capeval.uv-sync.lock").exists()


def test_stale_clone_refusal_names_the_configured_checkout(tmp_path):
    # The stale-clone gate reads THROUGH S2_REPO_DIR: a mismatching HEAD
    # must refuse and name the configured checkout, not the default one.
    checkout = tmp_path / "co"
    checkout.mkdir()
    (checkout / "EXPECTED_SHA").write_text("0000000\n")  # != STUB_SHA
    proc = _run(tmp_path, {"S2_REPO_DIR": str(checkout)})
    assert "FATAL: stale clone" in proc.stdout
    assert f"Re-sync {checkout} (D7)" in proc.stdout
    assert "__EXIT_CALLED__:1" in proc.stdout


def test_missing_or_malformed_submit_pin_is_fatal(tmp_path):
    # The submit-time pin is fail-closed THROUGH the capeval
    # S2_REPO_DIR lane too: an otherwise-valid capeval world (checkout +
    # matching EXPECTED_SHA) must still refuse when the submission carried
    # no --comment pin, or a short-sha one (a form a real submission draft
    # actually staged — the full 40-hex is load-bearing). The full pin
    # matrix lives in test_stage2_run.py; this holds the gate where
    # capeval jobs enter it.
    legacy = tmp_path / "legacy"
    (legacy / "repo").mkdir(parents=True)
    (legacy / "EXPECTED_SHA").write_text(STUB_SHA + "\n")
    base = {
        "S2_REPO_DIR": str(legacy / "repo"),
        "S2_UV_ENV": str(tmp_path / "venvs" / "sisteer"),
    }
    # No Comment= line at all (set-but-empty stub override):
    proc = _run(tmp_path, {**base, "STUB_SCONTROL_LINES": ""})
    assert ("FATAL: submit-time SHA pin missing or ambiguous "
            "(0 Comment= values parsed; D12)") in proc.stdout
    assert "sbatch --comment=sha:$(git -C" in proc.stdout  # fix printed
    assert "__EXIT_CALLED__:1" in proc.stdout
    # Malformed: a short sha, in the cluster's real trailing-space rendering.
    proc = _run(tmp_path, {**base,
                           "STUB_SCONTROL_LINES": "   Comment=sha:cafecafe "})
    assert ("FATAL: malformed submit-time SHA pin 'sha:cafecafe' "
            "(want sha:<40-hex>; D12)") in proc.stdout
    assert "__EXIT_CALLED__:1" in proc.stdout


def test_pin_fix_hint_names_the_effective_checkout(tmp_path):
    # The printed fix must be the incantation that would SUCCEED. A capeval
    # job exports only S2_REPO_DIR, so the old hint — interpolating
    # $S2_REPO_ROOT/repo, which falls back to the peers' checkout — told
    # the operator to derive the pin from a checkout the HEAD-equality
    # check never reads: following it produced a pin that then FATALed
    # whenever the two checkouts differed (the norm). The hint must name
    # the effective S2_REPO_DIR, on both the missing-pin and the
    # malformed-pin paths.
    checkout = tmp_path / "capeval-repo"
    checkout.mkdir()
    (checkout / "EXPECTED_SHA").write_text(STUB_SHA + "\n")
    base = {
        "S2_REPO_DIR": str(checkout),
        "S2_UV_ENV": str(tmp_path / "venvs" / "sisteer-capeval"),
    }
    hint = f"sbatch --comment=sha:$(git -C {checkout} rev-parse HEAD)"
    for stub_lines in ("",                             # no pin at all
                       "   Comment=sha:cafecafe "):    # malformed pin
        proc = _run(tmp_path, {**base, "STUB_SCONTROL_LINES": stub_lines})
        assert hint in proc.stdout, proc.stdout
        assert "__EXIT_CALLED__:1" in proc.stdout
        # the wrong-checkout incantation (the default S2_REPO_ROOT/repo
        # path) must be gone:
        assert f"{_project_root(tmp_path)}/repo rev-parse HEAD" not in proc.stdout


def test_every_pin_fatal_hint_interpolates_s2_repo_dir():
    # Static pin over ALL the hint sites (the harness above can only reach
    # the missing/malformed paths; the scontrol-unreadable path prints the
    # same incantation): every pin-FATAL fix hint interpolates
    # $S2_REPO_DIR — the checkout the HEAD-equality comparison actually
    # uses — never $S2_REPO_ROOT/repo, which capeval jobs leave at the
    # peers' default.
    hints = [l for l in SCRIPT.read_text().splitlines()
             if "s2_log" in l and "sbatch --comment=sha:" in l]
    assert len(hints) == 3, hints  # unreadable / missing-ambiguous / malformed
    for line in hints:
        assert "git -C $S2_REPO_DIR rev-parse HEAD" in line, line
        assert "$S2_REPO_ROOT" not in line, line
