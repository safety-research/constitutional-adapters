"""08_grade.sh's Petri preflight: the grading daemon's `--petri-python` contract is an interpreter that imports inspect_ai + petri_bloom
(grade_daemon.PETRI_IMPORT_CHECK; petri_drive_full22.sh refuses without them), but 00_setup.sh --what venvs builds the grading host's
environment WITHOUT pyproject's `petri` group (installing it is a separately gated step). Before the stage starts the daemon it probes the
interpreter it would hand over: with the two modules the drive is handed over as before; without them the daemon starts with the Petri
drive DISABLED — none of the --petri-* flags, so every other family grades exactly as before — and ONE warning names the command that
installs the group, aimed at the environment the probed interpreter belongs to (UV_PROJECT_ENVIRONMENT=<its venv> --directory <repo>;
an interpreter outside any venv is named as such). Driven through the ACTUAL stage (dry run, neutral site) with a stand-in interpreter that answers the probe as told
(front_door_testlib.fake_python), so the verdict never depends on what the runner's own venv carries."""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import PETRI_IMPORT_CHECK, REPO, assert_no_tokens, fake_python, run_stage, scrub_site, token_hits, write_site  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "registry"))
import grade_daemon as GD  # noqa: E402

MODEL = "qwen35_4b_demo"
INSTALL = "uv sync --frozen --extra cpu --group notebook --group corpus --group petri"


def _compose(tmp_path: Path, *, petri: bool, venv: bool = True):
    """Run the stage's dry run with a stand-in interpreter; (interpreter path, the completed process, the daemon's argv)."""
    py = fake_python(tmp_path, petri=petri, venv=venv)
    site = write_site(tmp_path, extra_config=f"SISTEER_PYTHON={py}")
    (tmp_path / "state" / MODEL).mkdir(parents=True)
    r = run_stage("08_grade.sh", "--dry-run", "--once", "--model", MODEL, site=site, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    dry = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:") and "grade_daemon.py" in ln]
    assert len(dry) == 1, r.stdout[-1500:]
    scan = (r.stdout + r.stderr).replace(str(REPO), "<repo>").replace(str(tmp_path), "<tmp>").replace(os.path.expanduser("~"), "<home>")
    assert_no_tokens(scan, f"08_grade.sh petri={petri}")
    return py, r, shlex.split(dry[0][len("DRY-RUN:"):])


def _the_one_warning(r) -> str:
    warn = [ln for ln in r.stderr.splitlines() if "WARNING:" in ln and "Petri drive DISABLED" in ln]
    assert len(warn) == 1, r.stderr
    return warn[0]


def test_an_interpreter_without_the_petri_group_starts_the_daemon_with_the_drive_disabled_and_names_the_install(tmp_path):
    py, r, argv = _compose(tmp_path, petri=False)
    for flag in ("--petri-state", "--petri-model", "--petri-python", "--petri-cluster", "--petri-drive"):
        assert flag not in argv, (flag, argv)
    # everything the other families need is still handed over, under the same interpreter
    assert argv[argv.index("--python") + 1] == py
    for flag in ("--registry-root", "--model-id", "--grades-root", "--configs-root", "--state-dir", "--once", "--dry-run"):
        assert flag in argv, (flag, argv)
    warn = _the_one_warning(r)
    assert INSTALL in warn and "inspect_ai" in warn and "petri_bloom" in warn and py in warn, warn
    # the install line is aimed at the environment the probed interpreter belongs to (SISTEER_PYTHON names a venv other than the
    # repository's .venv here, where a plain `uv sync` would install the group somewhere the probe never looked) and at this repository
    env_root = str(Path(py).parent.parent)
    assert f"UV_PROJECT_ENVIRONMENT={env_root} {INSTALL} --directory {REPO}" in warn, warn
    assert "pyvenv.cfg" not in warn
    assert "petri=disabled" in r.stdout and "petri=drive" not in r.stdout


def test_an_interpreter_outside_any_venv_is_told_so_and_pointed_at_a_venv_built_from_this_repository(tmp_path):
    """SISTEER_PYTHON may name an interpreter no `uv sync` can add the group to (no pyvenv.cfg above it): the warning names that
    condition instead of printing an install line that would apply to some other environment, and still carries the one install
    line (into a venv built from this repository) plus where to point SISTEER_PYTHON afterwards."""
    py, r, argv = _compose(tmp_path, petri=False, venv=False)
    assert "--petri-python" not in argv and argv[argv.index("--python") + 1] == py
    warn = _the_one_warning(r)
    assert "UV_PROJECT_ENVIRONMENT=" not in warn, warn
    assert f"{py} is not a virtual environment's interpreter" in warn and "pyvenv.cfg" in warn, warn
    assert f"{INSTALL} --directory {REPO}" in warn and "SISTEER_PYTHON" in warn, warn
    assert "petri=disabled" in r.stdout


def test_the_install_lines_venv_root_is_scrubbed_by_the_dry_run_scanners_as_the_interpreter_is(tmp_path, tmp_path_factory):
    """The cluster gate's shape: the site's SISTEER_PYTHON names a venv OUTSIDE the test's temp root, under a path that carries a site
    identifier (on the gate the login's shared root; here a synthetic private-network-shaped component, caught by shape — and on the
    grading host pytest's own `pytest-of-<login>` basetemp). Without the petri group the stage's one warning prints that venv as
    `UV_PROJECT_ENVIRONMENT=<env>`. The dry-run scanners (test_front_door_dry_run, test_reproduce_readme_committed_sequence) scrub the
    interpreter they hand the site and must scrub ITS ENVIRONMENT the same way, or the site's venv root fails the identifier scan (two
    gate reds, 2026-10-13). Pinned through front_door_testlib.scrub_site, the one scrub both scanners apply."""
    synthetic = "tail" + "0a1b2c"                     # the shape the scan catches, assembled so no file spells it; never a real id
    py = fake_python(tmp_path_factory.mktemp("elsewhere") / synthetic, petri=False)
    site = write_site(tmp_path, extra_config=f"SISTEER_PYTHON={py}")
    (tmp_path / "state" / MODEL).mkdir(parents=True)
    r = run_stage("08_grade.sh", "--dry-run", "--once", "--model", MODEL, site=site, timeout=300)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    env_root = str(Path(py).parent.parent)
    assert synthetic in env_root and not env_root.startswith(str(site.parent)), "the premise: the venv sits outside the scanner's <tmp>"
    assert f"UV_PROJECT_ENVIRONMENT={env_root} {INSTALL}" in _the_one_warning(r)
    out = r.stdout + r.stderr
    assert token_hits(out), "the premise: the raw output carries the identifier (through the venv root); the scrub is what keeps it out"
    scan = scrub_site(out, tmp=site.parent, python=py)
    assert "<python>" in scan and "<venv>" in scan and env_root not in scan, scan[-1500:]
    assert_no_tokens(scan, "08_grade.sh petri=False, the venv elsewhere")
    # an interpreter outside any venv names no environment, and nothing but the interpreter is scrubbed for it
    bare = fake_python(tmp_path_factory.mktemp("bare") / synthetic, petri=False, venv=False)
    assert "<venv>" not in scrub_site(f"{bare} is not a virtual environment's interpreter", tmp=site.parent, python=bare)


def test_an_interpreter_with_the_petri_group_hands_the_drive_over_and_warns_nothing(tmp_path):
    py, r, argv = _compose(tmp_path, petri=True)
    assert argv[argv.index("--petri-python") + 1] == py and argv[argv.index("--petri-model") + 1] == MODEL
    assert argv[argv.index("--petri-state") + 1] == str(tmp_path / "state" / MODEL)
    assert sum(1 for a in argv if a == "--petri-cluster") == 2                 # one per configured cluster
    assert "Petri drive DISABLED" not in r.stderr and INSTALL not in r.stderr
    assert "petri=drive" in r.stdout


def test_the_stage_probes_exactly_the_daemons_import_check_and_the_documents_carry_the_install_line():
    text = (REPO / "reproduce" / "08_grade.sh").read_text()
    assert PETRI_IMPORT_CHECK == GD.PETRI_IMPORT_CHECK                        # the stand-in answers the string the daemon checks
    assert f"-c '{GD.PETRI_IMPORT_CHECK}'" in text, "08_grade.sh must probe the daemon's own import check"
    assert text.index(GD.PETRI_IMPORT_CHECK) < text.index("grade_daemon.py\" \"${DAEMON_ARGS[@]}\""), "the probe runs BEFORE the daemon starts"
    assert INSTALL in text
    for rel in ("docs/configuration.md", "reproduce/README.md"):
        assert INSTALL in (REPO / rel).read_text(), f"{rel}: the Petri install line is not documented"
    # the premise: 00_setup.sh's venvs step does NOT install the petri group (that is a separately gated step, by design)
    setup = (REPO / "reproduce" / "00_setup.sh").read_text()
    assert "--group petri" not in setup


def test_the_daemon_watches_no_petri_drive_when_the_stage_hands_over_none_of_the_flags():
    """The behaviour the disabled branch relies on: with none of --petri-drive / --petri-state / --petri-model the daemon's `petri_watch`
    is False (main() starts no Petri thread and runs no Petri pass), and the parser needs none of them."""
    args = GD.build_parser().parse_args(["--registry-root", "/nonexistent", "--model-id", "m", "--once"])
    assert args.petri_drive is None and args.petri_state is None and args.petri_model is None and args.petri_cluster is None
    assert not bool(GD.parse_petri_specs(args.petri_drive) or (args.petri_state and args.petri_model))
