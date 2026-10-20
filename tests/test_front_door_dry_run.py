"""Every stage answers --dry-run for every front-door test model against the NEUTRAL site example: exit 0, every command printed with the
config substituted (no unexpanded `$SHARED`/`${…}` placeholder, no `<…>` template), nothing written outside the state dir, and
no personal identifier anywhere in the output."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, STAGES, TEST_MODELS, VENV_PY, assert_no_tokens, run_stage, scrub_site, write_site  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from sisteer.capeval.tinycap import contracts as C  # noqa: E402  (the Modal launcher's grammar: app-name env, cells, modes, the container cap)

#: a real-run call's environment: env-mode secrets with NO Modal token, so the credentials refusal is the deterministic stop AFTER the
#: stage's own gates and no Modal command can ever run from a test (the runner's shell may export a real token)
NO_MODAL_TOKEN = {"MODAL_TOKEN_ID": "", "MODAL_TOKEN_SECRET": ""}

PLACEHOLDER = re.compile(r"\$SHARED\b|\$\{[A-Z_]+[:}]|<[a-z][a-z _-]{2,40}>")
# stage -> the extra flags its dry run needs (the other workstreams' stages are exercised when present with their own flags)
FLAGS = {
    "00_setup.sh": ["--what", "all"],
    "01_make_corpora.sh": ["--corpus", "all", "--phase", "all"],
    "02_prepare_model.sh": ["--step", "all"],
    "03_hpo.sh": ["--object-class", "relu", "--phase", "all"],
    "04_fit_objects.sh": ["--object", "all"],
    "05_stage_roster.sh": [],
    "06_eval_waves.sh": ["--view", "battery", "--family", "all"],
    "07_bcb_bfcl.sh": [],
    "08_grade.sh": ["--once"],
    "09_dose_ladder.sh": ["--alpha", "0.75"],
    "10_figures.sh": ["--panel", "all", "--source", "snapshot"],
}
NO_MODEL = {"00_setup.sh": False, "01_make_corpora.sh": True, "10_figures.sh": True}
FD = TEST_MODELS[0]      # the front-door model (models.front_door_keys()): the one with phase views, an objects block and a wave geometry


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("fd")
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _run(stage, model, site, extra=()):
    args = ["--dry-run", *FLAGS.get(stage, []), *extra]
    if not NO_MODEL.get(stage, False):
        args += ["--model", model]
    return run_stage(stage, *args, site=site, timeout=600)


def test_07_resolves_the_models_tinycap_row_before_any_modal_call_and_names_an_unregistered_slug(site, tmp_path):
    """Stage 07 passes --model <key> to scripts/modal/tinycap_served.py as the tinycap slug (contracts.model_row); the stage
    resolves that row ITSELF, before the secrets and the Modal calls. The dry run prints the resolution — the worked example's
    row is the paper's Qwen3.5-4B at the qwen35_4b pin — and hands the launcher its own control tokens for the paper's FULL sets
    (bigcodebench_full / bfcl_full, contracts.CONTROL_BIGCODEBENCH_FULL / CONTROL_BFCL_FULL through expand_control; no 200-item
    subsets) under the paper's `protocol` decoding; a table key the tinycap registry has no row for prints the refusal a real run would
    make under --dry-run (exit 0, nothing written) and REFUSES by name without it, before any Modal command."""
    import yaml
    r = _run("07_bcb_bfcl.sh", FD, site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "tinycap row qwen35_4b_demo: Qwen/Qwen3.5-4B @ 851bf6e8 family=qwen3_5 model_key=qwen35_4b thinking=off" in out, out[-2500:]
    assert "petri_manifest=configs/capeval/manifests/petri_full22_qwen35_4b.yaml" in out and "UNREGISTERED" not in out
    run_lines = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and "--controls" in ln]
    assert len(run_lines) == 1 and " --model qwen35_4b_demo " in run_lines[0] and f" --controls {C.CONTROL_BIGCODEBENCH_FULL},{C.CONTROL_BFCL_FULL} " in run_lines[0], run_lines
    assert " --controls bigcodebench_full,bfcl_full " in run_lines[0] and f" --sampling {C.SAMPLING_PROTOCOL} " in run_lines[0] and "200" not in run_lines[0].split("--controls")[1].split("--")[0]
    assert "controls=bigcodebench_full,bfcl_full (n_items 1140,3981) sampling=protocol" in out
    # the same table under a key contracts.MODELS has no row for
    table = yaml.safe_load((REPRO / "models.yaml").read_text())
    table["models"]["qwen35_4b_demo_unregistered"] = table["models"].pop("qwen35_4b_demo")
    alt = tmp_path / "models.yaml"
    alt.write_text(yaml.safe_dump(table, sort_keys=False))
    env = {"SISTEER_MODELS_TABLE": str(alt)}
    r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", "qwen35_4b_demo_unregistered", site=site, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "tinycap row: UNREGISTERED: unknown model slug 'qwen35_4b_demo_unregistered'" in out and "a real run refuses here" in out, out[-2500:]
    assert any(ln.startswith("DRY-RUN:") and "--controls bigcodebench_full,bfcl_full --sampling protocol" in ln for ln in out.splitlines())
    r = run_stage("07_bcb_bfcl.sh", "--model", "qwen35_4b_demo_unregistered", site=site, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out[-2500:]
    assert "REFUSED: tinycap: UNREGISTERED: unknown model slug 'qwen35_4b_demo_unregistered'" in out and "contracts.MODELS" in out, out[-2500:]
    assert "modal run" not in out and "DRY-RUN:" not in out, out[-2500:]


def test_07_translates_the_views_cells_to_the_launchers_tokens_and_refuses_a_cell_it_cannot_express(site):
    """Stage 07's --cells are the view's cells by their models.yaml NAMES; the Modal launcher plans with its own cell grammar
    (contracts.parse_launcher_cell), which refused those names at plan time (the last known stage-07 gap of the worked example). The
    stage now translates every cell through lib/models.py launcher-cells (roster_cell_id — the one grammar the wave roster and the
    launcher share) and prints the mapping; the run command carries the TOKENS, in the view's order. A cell the mapping cannot
    express refuses BY NAME before any Modal command: one the view does not list. A DOSE view's object cell maps to the SAME token as the
    wave's parent and is served from the dose view's OWN registry row (lib/models.py launcher-slug = `<key>-<view>`; demo-import 2026-10-12,
    the Known gap "the dose views stay on the cluster" closed): the stage prints the slug, hands it to the launcher as --model, stages the
    view's committed dose copies into the row's bank and holds the view behind the wave's recorded gate pass."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    view = M.baseline_view(FD)                                              # the one steered wave (never spelled here: a peer renames it)
    expected = M.launcher_cells(FD, view)                                   # pinned literally in tests/test_stage07_launcher_cells.py
    r = _run("07_bcb_bfcl.sh", FD, site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    for cell, tok in expected:
        assert f"launcher cell: {cell} -> {tok}" in out, (cell, tok, out[-2500:])
    assert "launcher cell: refusal_direction -> arditi_allpos" in out and "launcher cell: control_lora -> constv2ctl2-full__lora_r4_ce" in out
    run_lines = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and "--controls" in ln]
    assert len(run_lines) == 1 and f" --cells {','.join(t for _c, t in expected)} " in run_lines[0], run_lines
    assert " --cells control_lora" not in run_lines[0] and ",control_lora" not in run_lines[0]       # never the table's names
    # a subset, by the table's names, in the caller's order
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--cells", "none,difference_lora"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert any(ln.startswith("DRY-RUN:") and " --cells none,constv2-mix-full__lora_r4_ce_minus_ctl " in ln for ln in out.splitlines()), out[-2500:]
    # a cell the view does not list refuses by name, before any Modal command — under --dry-run too (nothing to print for it)
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--cells", "control_lora,nope"])
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED: models.yaml:" in out and "'nope' is not a cell of the view" in out, out[-2500:]
    assert "DRY-RUN:" not in out and "modal run" not in out and "launcher cell:" not in out, out[-2500:]
    # a dose view: its object cell maps to the wave's parent's token, served from the view's own row — the slug is printed and handed to the
    # launcher, the bank stager gets the view (its receipts under 07_bank/<view>), and the view is held behind the wave's recorded gate pass
    m = M.model(FD)
    state = site.parent / "state" / FD
    for dose_view in [v for v, vd in m["views"].items() if "dose" in (vd or {})]:
        slug = M.launcher_slug(FD, dose_view)
        assert slug == f"{FD}-{dose_view}" and slug in C.MODELS
        expected_d = M.launcher_cells(FD, dose_view)
        toks = [t for _c, t in expected_d]
        assert toks[0] == "none" and len(toks) == 2 and toks[1] in [t for _c, t in expected] and toks[1] == M.roster_cell_id(m, m["views"][dose_view]["cells"][1])
        r = _run("07_bcb_bfcl.sh", FD, site, extra=["--view", dose_view])
        out = r.stdout + r.stderr
        assert r.returncode == 0 and "is a dose rung" not in out, out[-2500:]
        assert f"launcher slug: {slug} (a dose view: its own registry row, bank {C.model_row(slug).objects_dir} and config)" in out
        assert f"DRY-RUN: require {state}/gate/bcb_bfcl/{view}.json verdict=pass" in out              # the wave's pass is the dose view's gate
        for cell, tok in expected_d:
            assert f"launcher cell: {cell} -> {tok}" in out
        run_lines = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and "--controls" in ln]
        assert len(run_lines) == 1 and f" --model {slug} " in run_lines[0] and f" --cells {','.join(toks)} " in run_lines[0] and "--allow-extra" not in run_lines[0]
        assert f"TINYCAP_APP_NAME=tinycap-{repro_sha7()}-{slug} " in run_lines[0]
        stager = next(ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and "stage_demo_bank_modal.py" in ln)
        assert f"--view {dose_view} " in stager and f"--receipts-dir {state}/07_bank/{dose_view} " in stager
        assert f"write {state}/07_views/{dose_view}/07_run_identity.json" in out and f"slug {slug}, 2 cell(s), objects for 1 of 1 fitted" in out
        # the mirror and the file phase FOLLOW THE VIEW'S STATE (stage07-mirror x stage07-full-sets): both tools get the dose view's state root,
        # <state>/07_views/<view>, and a cluster destination of the same layout, <reproduce root>/<model>/07_views/<view>/07_mirror — so the receipt
        # the mirror reads, the 07_mirror/ it writes and the pointer records never sit beside the wave's; with no receipt / verdict under the view's
        # directory each refuses by name there, naming the VIEW's paths (the wave's gate pass is still the dose view's admission, printed above)
        view_state = f"{state}/07_views/{dose_view}"
        r = _run("07_bcb_bfcl.sh", FD, site, extra=["--view", dose_view, "--phase", "mirror"])
        out = r.stdout + r.stderr
        mirror_lines = [ln for ln in _dry_lines(out) if "stage07_mirror.py" in ln]
        assert r.returncode == 2 and len(mirror_lines) == 1 and not _modal_lines(out), out[-2500:]
        argv = shlex.split(mirror_lines[0][len("DRY-RUN:"):])
        assert argv[argv.index("--state-root") + 1] == view_state and argv[argv.index("--cluster-dest") + 1].endswith(f"/exp/reproduce/{FD}/07_views/{dose_view}/07_mirror")
        assert argv[argv.index("--ssh-host") + 1] == "cluster-a-login" and argv[argv.index("--cluster-label") + 1] == "cluster_a" and argv[-1] == "--dry-run"
        # the cluster root the destination is guarded under (round 6): the expanded reproduce root itself, the destination strictly below it
        mirror_root = argv[argv.index("--cluster-root") + 1]
        assert mirror_root.endswith("/exp/reproduce") and argv[argv.index("--cluster-dest") + 1] == f"{mirror_root}/{FD}/07_views/{dose_view}/07_mirror", argv
        assert f"REFUSED: no receipt at {view_state}/07_store/receipt.json" in out and f"DRY-RUN: require {state}/gate/bcb_bfcl/{view}.json verdict=pass" in out, out[-2500:]
        r = _run("07_bcb_bfcl.sh", FD, site, extra=["--view", dose_view, "--phase", "file"])
        out = r.stdout + r.stderr
        file_lines = [ln for ln in _dry_lines(out) if "stage07_file.py" in ln]
        assert r.returncode == 2 and len(file_lines) == 1 and not _modal_lines(out), out[-2500:]
        argv = shlex.split(file_lines[0][len("DRY-RUN:"):])
        assert argv[argv.index("--state-root") + 1] == view_state and argv[argv.index("--view") + 1] == dose_view and argv[argv.index("--model") + 1] == FD
        assert argv[argv.index("--cluster-dest") + 1].endswith(f"/exp/reproduce/{FD}/07_views/{dose_view}/07_mirror") and argv[argv.index("--cluster-label") + 1] == "cluster_a"
        assert argv[argv.index("--cluster-dest") + 1] == mirror_lines[0].split("--cluster-dest ", 1)[1].split()[0], "one destination for the mirror and the filing's cross-check"
        assert argv[argv.index("--cluster-root") + 1] == mirror_root, "one root for the mirror's guard and the filing's cross-check"
        assert f"REFUSED: no verdict at {view_state}/gate/bcb_bfcl/{dose_view}.json" in out, out[-2500:]
    # the wave's mirror and filing keep their paths: the model's state dir and <reproduce root>/<model>/07_mirror (pinned in the phases test below)


def repro_sha7() -> str:
    """HEAD's short sha of this checkout, as the stage spells its Modal app name (`tinycap-<sha7>-<slug>`)."""
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()[:7]


def _modal_lines(out: str) -> list[str]:
    """The `DRY-RUN:` lines that would run the Modal launcher, in the stage's order."""
    return [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and "scripts/modal/tinycap_served.py" in ln]


def _launcher_argv(line: str) -> list[str]:
    """The launcher's own argv out of a printed `DRY-RUN: … modal run … scripts/modal/tinycap_served.py <argv>[ | tee <log>]` line — the
    captured phases (demo-import M8: preflight and run) pipe the launcher's stdout to a state file; the suffix is not the launcher's."""
    tail = line.split("scripts/modal/tinycap_served.py", 1)[1]
    if " | tee " in tail:
        tail, tee = tail.rsplit(" | tee ", 1)
        assert tee.strip() and not tee.strip().startswith("-"), line                 # the tee target is a path, never a flag swallowed from the argv
    return shlex.split(tail)


def _tee_target(line: str) -> str | None:
    """The `| tee <path>` target of a captured Modal line, None when the line is not captured."""
    return line.rsplit(" | tee ", 1)[1].strip() if " | tee " in line else None


def _env_assignment(line: str, name: str) -> str | None:
    """`NAME=value` as a word of the printed shell string (the launcher's environment), or None when absent."""
    m = re.search(rf"(?:^|\s){re.escape(name)}=(\S+)", line)
    return m.group(1) if m else None


def _replay_dry_run(L, argv: list[str]) -> str:
    """Stage 07's printed launcher argv, run through the launcher's OWN argparse and `main` as its CPU dry run (the way the
    bcb-bfcl-200-widen wrapper replays the argv it composed): `--pf` is a `modal run`-only flag, stripped the same way."""
    a = L.cli_parser().parse_args([x for x in argv if x != "--pf"] + ["--dry-run"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.main(sha=a.sha, pass1=a.pass1, pass2=a.pass2, cells=a.cells, modes=a.modes, controls=a.controls, model=a.model,
               app_name=a.app_name, serving_path=a.serving_path, allow_extra=a.allow_extra, expect_hashes=a.expect_hashes,
               sampling=a.sampling, dry_run=True)
    return buf.getvalue()


def test_07s_modal_commands_name_the_app_in_the_launchers_environment_and_the_launcher_plans_one_container_per_cell(site):
    """The Modal launcher builds its App at IMPORT from `TINYCAP_APP_NAME` in the `modal run`'s environment (no default) and holds
    `--app-name` equal to it; `modal run -n` is Modal's dashboard run name and sets no variable — so a stage that passed only `-n`
    and `--app-name` refused at import on its first real call. The launcher PLANS a
    configuration only behind a pass (`plan = pass1 or pass2 or dry_run`), launches a fitted cell only behind `--pf`,
    plans its default `off,on` modes when none are named — an ON leg this row refuses, no verified reasoning parser — and
    refuses more than `contracts.MAX_GPU_CONTAINERS` containers unless `--allow-extra` is passed on purpose (both found by
    replaying the printed argv through the launcher, which is what this test does). Pinned: every printed Modal command carries
    `TINYCAP_APP_NAME=<app>` in its environment equal to its `--app-name`; the run phase carries `--pf --pass1 --modes off` and
    `--allow-extra` exactly when the plan exceeds the cap; and the ACTUAL launcher invocation — the printed argv through the
    launcher's own `cli_parser` + `main` as a dry run, imported under the stage's app name — plans ONE GPU container per cell,
    the subset too, and refuses without the modes or the cap override the stage adds."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    import test_tinycap_launcher as TL                                      # the modal stub + the by-path loader every launcher test uses
    view = M.baseline_view(FD)
    expected = M.launcher_cells(FD, view)
    tokens = [t for _c, t in expected]
    r = _run("07_bcb_bfcl.sh", FD, site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    lines = _modal_lines(out)
    assert len(lines) == 2, lines                                           # the stage phase, then the run phase
    apps = set()
    for ln in lines:
        argv = _launcher_argv(ln)
        app = argv[argv.index("--app-name") + 1]
        assert _env_assignment(ln, C.MODAL_APP_NAME_ENV) == app, ln        # in the launcher's ENVIRONMENT, equal to --app-name
        assert ln.index(f"{C.MODAL_APP_NAME_ENV}={app}") < ln.index("modal run"), ln
        assert f" -n {app} " in ln                                          # Modal's own run name beside it, as the Modal README spells it
        apps.add(app)
    assert len(apps) == 1
    app = apps.pop()
    stage_argv, run_argv = _launcher_argv(lines[0]), _launcher_argv(lines[1])
    assert "--upload-tree" in stage_argv and "--stage" in stage_argv and "--pass1" not in stage_argv and "--cells" not in stage_argv
    assert "--pf" in run_argv and "--pass1" in run_argv and "--pass2" not in run_argv, run_argv
    assert run_argv[run_argv.index("--modes") + 1] == C.THINKING_OFF and run_argv[run_argv.index("--cells") + 1] == ",".join(tokens)
    assert f"launcher modes: {C.THINKING_OFF}" in out
    assert len(tokens) > C.MAX_GPU_CONTAINERS, "the worked example's view was expected to exceed the launcher's container cap"
    assert "--allow-extra" in run_argv and f"exceed contracts.MAX_GPU_CONTAINERS {C.MAX_GPU_CONTAINERS}" in out and "on purpose" in out
    # the ACTUAL invocation: the launcher imported under the app name the stage put in the environment (its App is built from it),
    # the printed argv through its own argparse + main — one GPU container per cell, every cell planned
    L = TL._load_launcher(TL._stub_modal(), app_name=app, name="tinycap_served_stage07_replay")
    assert L.APP_NAME == app == L.app.name
    plan = _replay_dry_run(L, run_argv)
    assert f"[tinycap] dry run: {len(tokens)} GPU configuration container(s), pass 1" in plan, plan[-2000:]
    for tok in tokens:
        assert f'"cell": "{tok}"' in plan, (tok, plan[-2000:])
    # what the stage adds is load-bearing: without its --modes the launcher's default plans an ON leg this row cannot serve;
    # without its --allow-extra the ten containers trip the cost wire; with another app name the launcher refuses the mismatch
    a = L.cli_parser().parse_args([x for x in run_argv if x != "--pf"] + ["--dry-run"])
    with pytest.raises(L.ThinkingUnservable, match="asks for thinking-ON"):
        with contextlib.redirect_stdout(io.StringIO()):
            L.main(sha=a.sha, pass1=True, cells=a.cells, modes="", controls=a.controls, model=a.model, app_name=a.app_name,
                   serving_path=a.serving_path, allow_extra=True, dry_run=True)
    with pytest.raises(SystemExit, match="MAX_GPU_CONTAINERS"):
        with contextlib.redirect_stdout(io.StringIO()):
            L.main(sha=a.sha, pass1=True, cells=a.cells, modes=a.modes, controls=a.controls, model=a.model, app_name=a.app_name,
                   serving_path=a.serving_path, allow_extra=False, dry_run=True)
    with pytest.raises(SystemExit, match="must be in the ENVIRONMENT"):
        L.check_app_name("tinycap-other-name")
    # a subset under the cap: no --allow-extra, and the replay plans exactly those cells
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--cells", "none,difference_lora"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    sub_argv = _launcher_argv(_modal_lines(out)[1])
    assert "--allow-extra" not in sub_argv and "--pf" in sub_argv and "--pass1" in sub_argv and "MAX_GPU_CONTAINERS" not in out
    plan = _replay_dry_run(L, sub_argv)
    assert "[tinycap] dry run: 2 GPU configuration container(s), pass 1" in plan and '"cell": "none"' in plan, plan[-2000:]
    assert f'"cell": "{dict(expected)["difference_lora"]}"' in plan


def test_07_requires_an_expected_intervention_hash_for_every_fitted_cell_before_any_modal_call(tmp_path):
    """The launcher's preflight (`--pf`, which every fitted cell launches behind) verifies each fitted object's HF post-load
    intervention hash against a stated expectation — `contracts.EXPECTED_INTERVENTION_HASHES[slug][cell]`, or the launcher's
    `--expect-hashes` (an explicit entry wins) — and REFUSES a cell with none (E40 H1: `object_identity_failures`). The table has no
    `qwen35_4b_demo` key and the stage passed no `--expect-hashes`, so once the bank and the pass flags were in place every fitted
    cell of the worked example would have been refused inside the preflight. Pinned: the
    stage resolves each fitted cell's source (table / file / none) through the same contracts BEFORE any Modal call and prints it;
    a run phase with a cell at `none` refuses BY NAME (the dry run prints the refusal a real run would make and exits 0, the run
    command still printed); `--expect-hashes F` — a flat `{token: hash}` object, the launcher's own grammar — satisfies it and
    rides the run command; `--phase preflight` is the launcher's `--pf`-only invocation (no `--pass1`), allowed with no
    expectation because its report is how a freshly fitted bank's hashes are derived; a file outside the grammar, or an
    unreadable one, refuses by name; base and prompted cells carry no object and need none."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    site = write_site(tmp_path)                                             # env-mode secrets: see NO_MODAL_TOKEN
    view = M.baseline_view(FD)
    mapped = M.launcher_cells(FD, view)
    fitted = [t for _c, t in mapped if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
    assert len(fitted) >= 7 and all(C.expected_intervention_hash(FD, t) is None for t in fitted)      # the table has no demo entry: the gap
    # the dry run: the status per fitted cell, the refusal a real run would make, exit 0 with the run command still printed
    r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", FD, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    for t in fitted:
        assert f"expectation: {t} -> none" in out, (t, out[-2500:])
    assert f"no expected intervention hash for fitted cell(s) {','.join(fitted)} of slug {FD}" in out and "a real run refuses here" in out
    assert "--phase preflight" in out and "--expect-hashes FILE" in out and len(_modal_lines(out)) == 2
    # a real run refuses BY NAME before the secrets and before any Modal command (the credentials refusal would have been next)
    r = run_stage("07_bcb_bfcl.sh", "--model", FD, site=site, env=NO_MODAL_TOKEN, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2, out[-2500:]
    assert f"REFUSED: tinycap: no expected intervention hash for fitted cell(s) {','.join(fitted)}" in out and "--expect-hashes" in out
    assert "modal run" not in out and "DRY-RUN:" not in out and "Modal credentials" not in out, out[-2500:]
    # --expect-hashes F satisfies it: the dry run reports `file` per cell and the run command carries the launcher's flag; the real
    # run passes this gate (its next stop is the credentials refusal, before any Modal command)
    f = tmp_path / "expect.json"
    f.write_text(json.dumps({t: "0123456789abcdef" for t in fitted}))
    r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", FD, "--expect-hashes", str(f), site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    for t in fitted:
        assert f"expectation: {t} -> file" in out
    assert "a real run refuses here" not in out and "no expected intervention hash" not in out
    run_argv = _launcher_argv(_modal_lines(out)[1])
    assert run_argv[run_argv.index("--expect-hashes") + 1] == str(f) and "--pf" in run_argv and "--pass1" in run_argv
    r = run_stage("07_bcb_bfcl.sh", "--model", FD, "--expect-hashes", str(f), site=site, env=NO_MODAL_TOKEN, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "no expected intervention hash" not in out and "REFUSED: no Modal credentials" in out, out[-2500:]
    assert "modal run" not in out and "DRY-RUN:" not in out
    # --phase preflight: the launcher's --pf-only invocation, no --pass1 — allowed with no expectation (it derives the hashes)
    r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", FD, "--phase", "preflight", site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    lines = _modal_lines(out)
    assert len(lines) == 1 and "a real run refuses here" not in out
    pf_argv = _launcher_argv(lines[0])
    assert "--pf" in pf_argv and "--pass1" not in pf_argv and "--upload-tree" not in pf_argv and "--cells" in pf_argv
    assert _env_assignment(lines[0], C.MODAL_APP_NAME_ENV) == pf_argv[pf_argv.index("--app-name") + 1]
    # a file outside the launcher's grammar, or an unreadable one, refuses by name before any Modal command — under --dry-run too
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", FD, "--expect-hashes", str(bad), site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "expected a flat JSON object" in out and "DRY-RUN:" not in out, out[-2500:]
    r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", FD, "--expect-hashes", str(tmp_path / "missing.json"), site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "is not a readable file" in out and "DRY-RUN:" not in out, out[-2500:]
    # base / prompted cells carry no object: a run of them alone needs no expectation (the credentials refusal is the stop)
    r = run_stage("07_bcb_bfcl.sh", "--model", FD, "--cells", "none,prompted__safety_refusal", site=site, env=NO_MODAL_TOKEN, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "expectation:" not in out and "no expected intervention hash" not in out and "REFUSED: no Modal credentials" in out, out[-2500:]


def test_07_reads_a_relative_expect_hashes_path_where_the_caller_stands_and_hands_the_launcher_an_absolute_one(tmp_path):
    """`--expect-hashes F` is a file argument: the stage checks and reads it where the CALLER stands — but every Modal command it
    composes opens with `cd <repo>`, and the launcher's `parse_expect_hashes` opens the path from THERE (`Path(text).is_file()`), so
    a relative path handed on as spelled passed the stage's own checks, the plan reported `file` per cell, and the launcher would
    have refused `../hashes.json` from the repo root — under `--phase all` after the stage phase had already run. Pinned: invoked
    from a directory that is not the repo root with a path relative to it (that shape, a bare name, and a directory with a space
    in it), the stage reports `file` per fitted cell and the printed run command carries the ABSOLUTE path, quoted for the
    `bash -c` string so the launcher's argv gets the one file back — a file from the repo root the command cds to, where the
    spelled path names nothing; a relative path unreadable from the caller's directory refuses by the name the caller spelled,
    before any Modal command."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    site = write_site(tmp_path)
    mapped = M.launcher_cells(FD, M.baseline_view(FD))
    fitted = [t for _c, t in mapped if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
    expect = json.dumps({t: "0123456789abcdef" for t in fitted})
    f = tmp_path / "expect.json"; f.write_text(expect); (tmp_path / "caller").mkdir()
    d = tmp_path / "with space"; d.mkdir(); g = d / "expect.json"; g.write_text(expect); (d / "caller").mkdir()
    cases = [(tmp_path / "caller", "../expect.json", f),                    # the review's shape: ../hashes.json from reproduce/
             (tmp_path, "expect.json", f),                                  # a bare name
             (d / "caller", "../expect.json", g)]                           # the quoting the bash -c string needs
    for cwd, spelled, path in cases:
        assert (cwd / spelled).is_file() and not (REPO / spelled).exists(), (cwd, spelled)   # a file where the caller stands; nothing from the repo root
        r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", FD, "--expect-hashes", spelled, site=site, cwd=cwd, timeout=600)
        out = r.stdout + r.stderr
        assert r.returncode == 0, (spelled, out[-2500:])
        for t in fitted:
            assert f"expectation: {t} -> file" in out, (spelled, t, out[-2500:])
        assert "no expected intervention hash" not in out
        lines = _modal_lines(out)
        assert len(lines) == 2 and lines[0].startswith(f"DRY-RUN: cd {REPO} && ") and lines[1].startswith(f"DRY-RUN: set -o pipefail; cd {REPO} && "), lines
        assert "--expect-hashes" not in lines[0]                           # the stage phase takes no run args
        argv = _launcher_argv(lines[1])                                     # shlex: what the launcher's argparse receives
        got = argv[argv.index("--expect-hashes") + 1]
        assert os.path.isabs(got) and Path(got).resolve() == path.resolve(), (spelled, got)
        assert (REPO / got).is_file(), got                                  # the launcher's own predicate from where the cd leaves it (an absolute path joins to itself)
    # unreadable where the caller stands: refused by the name the caller spelled, before any Modal command
    r = run_stage("07_bcb_bfcl.sh", "--dry-run", "--model", FD, "--expect-hashes", "../nope.json", site=site, cwd=tmp_path / "caller", timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED: --expect-hashes ../nope.json is not a readable file" in out and "DRY-RUN:" not in out, out[-2500:]


def _dry_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:")]


def test_07_stages_the_demo_bank_captures_preflight_and_run_and_gains_the_receipt_and_file_phases(site, tmp_path):
    """No --inspect-limit; the run phase's `|| repro_fatal` is unchanged. `--phase all` composes THREE
    Modal lines: the demo bank stager (reproduce/tools/stage_demo_bank_modal.py, no --go under --dry-run, the state's 07_bank receipts dir, the
    site's volume), then the tree's stage and the run — the run CAPTURED (`set -o pipefail; … --pf --pass1 | tee <state>/07_run_report.log`,
    pipefail INSIDE the bash -c string) with its argv pins intact (two launcher lines, --pf --pass1, --allow-extra only over the cap) and its
    follow-up naming `07 --phase receipt`, never a bare `<bcb pass rate>`; `--phase preflight` is captured to <state>/07_preflight_report.log and
    says what it parses into 07_expect_hashes.json; `--phase receipt` composes the stage07_receipt.py call (the tree sha, the app, the tokens, the
    volume) and the verdict command TEMPLATE with --range the one blank, pulling nothing; `--phase file` over no receipt refuses by name; the
    phase grammar gains receipt and file, neither in all; the state dir stays untouched by the dry runs that compose."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    view = M.baseline_view(FD)
    tokens = [t for _c, t in M.launcher_cells(FD, view)]
    state = site.parent / "state" / FD
    before = {str(p) for p in (site.parent / "state").rglob("*")} if (site.parent / "state").exists() else set()
    r = _run("07_bcb_bfcl.sh", FD, site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    modal_lines = [ln for ln in _dry_lines(out) if "stage_demo_bank_modal.py" in ln or "scripts/modal/tinycap_served.py" in ln]
    assert len(modal_lines) == 3 and "stage_demo_bank_modal.py" in modal_lines[0], modal_lines
    stager = shlex.split(modal_lines[0][len("DRY-RUN:"):])
    assert stager[stager.index("--model") + 1] == FD and stager[stager.index("--view") + 1] == view
    assert stager[stager.index("--volume") + 1] == "twosi-fits" and stager[stager.index("--environment") + 1] == "main" and "--go" not in stager and "--table" in stager
    assert stager[stager.index("--receipts-dir") + 1] == f"{state}/07_bank/{view}"                   # the receipts are per view (a dose view's beside the wave's)
    lines = _modal_lines(out)
    assert len(lines) == 2 and _tee_target(lines[0]) is None and _tee_target(lines[1]) == f"{state}/07_run_report.log", lines
    assert lines[1].startswith(f"DRY-RUN: set -o pipefail; cd {REPO} && ") and lines[0].startswith(f"DRY-RUN: cd {REPO} && ")
    run_argv = _launcher_argv(lines[1])
    assert "--pf" in run_argv and "--pass1" in run_argv and "--allow-extra" in run_argv and run_argv[run_argv.index("--cells") + 1] == ",".join(tokens) and "tee" not in run_argv
    assert "--inspect-limit" not in run_argv and "--inspect-limit" not in (REPRO / "07_bcb_bfcl.sh").read_text()
    follow = [ln for ln in out.splitlines() if "followup:" in ln]
    assert len(follow) == 1 and "--phase receipt" in follow[0] and "--phase file" in follow[0] and "<bcb pass rate>" not in out, follow
    assert "phase=all" in out and "expect_hashes=none (flag)" in out
    # the launch's identity: the run phase says it would write <state>/07_run_identity.json — the tree, every cell, the object id
    # of every fitted cell at this tree (the importer's plan), the app, the volume and environment, the overrides document, the expect-hashes — BEFORE the launcher line; nothing written
    ident = [ln for ln in _dry_lines(out) if ln.startswith(f"DRY-RUN: write {state}/07_run_identity.json")]
    assert len(ident) == 1 and "objects for 7 of 7 fitted" in ident[0] and "overrides paper_defaults (" in ident[0] and "expect-hashes none" in ident[0], ident
    assert out.index(ident[0]) < out.index(lines[1]) and "the receipt verifies and copies it" in ident[0]
    # --phase preflight: --pf alone, CAPTURED, then the parse the dry run names
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--phase", "preflight"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    lines = _modal_lines(out)
    assert len(lines) == 1 and _tee_target(lines[0]) == f"{state}/07_preflight_report.log" and lines[0].startswith("DRY-RUN: set -o pipefail; cd ")
    pf_argv = _launcher_argv(lines[0])
    assert "--pf" in pf_argv and "--pass1" not in pf_argv and "--expect-hashes" not in pf_argv and "tee" not in pf_argv
    parse = [ln for ln in _dry_lines(out) if ln.startswith("DRY-RUN: parse ")]
    assert len(parse) == 1 and f"{state}/07_expect_hashes.json" in parse[0] and "object_identity_and_header_guard" in parse[0] and "DIFFERS" in parse[0] and "absent" in parse[0]
    # the runnable expectation is pinned ONLY when every fitted cell's hash is the published one; the derived hashes always land beside it
    assert f"{state}/07_expect_hashes.derived.json" in parse[0] and "ONLY when every fitted cell reads file (= <slug>)" in parse[0] and "never a pin" in parse[0]
    assert "stage_demo_bank_modal.py" not in out
    # --phase receipt: the tool call composed (the tree sha, the app, the tokens, the volume, the run report), the verdict TEMPLATE, nothing pulled
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--phase", "receipt", "--cells", "none,difference_lora"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    rec_lines = [ln for ln in _dry_lines(out) if "stage07_receipt.py" in ln]
    assert len(rec_lines) == 1 and not _modal_lines(out), out[-2500:]
    argv = shlex.split(rec_lines[0][len("DRY-RUN:"):])
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    assert argv[argv.index("--sha") + 1] == sha and argv[argv.index("--app") + 1] == f"tinycap-{sha[:7]}-{FD}" and argv[argv.index("--view") + 1] == view
    assert argv[argv.index("--cells") + 1] == "none,constv2-mix-full__lora_r4_ce_minus_ctl" and argv[argv.index("--volume") + 1] == "twosi-fits"
    assert argv[argv.index("--run-report") + 1] == f"{state}/07_run_report.log" and argv[argv.index("--state-root") + 1] == str(state) and "--dry-run" not in argv
    # this call's resolution rides along as what the receipt VERIFIES against the launch's identity: the cells by the table's
    # names, OVERRIDE_ID, the Modal environment, and the identity file itself; no --expect-hashes (the receipt copies the launch's, never this call's)
    assert argv[argv.index("--view-cells") + 1] == "none,difference_lora" and argv[argv.index("--override-id") + 1] == "paper_defaults" and argv[argv.index("--environment") + 1] == "main"
    assert argv[argv.index("--run-identity") + 1] == f"{state}/07_run_identity.json" and "--expect-hashes" not in argv
    template = [ln for ln in _dry_lines(out) if "lib/stage.py verdict" in ln]
    assert len(template) == 1 and f"--family bcb_bfcl/{view}" in template[0] and "--range <lo,hi>" in template[0] and "--headline <the receipt's none inspect/bigcodebench_full acc>" in template[0]
    assert argv[argv.index("--controls") + 1] == "bigcodebench_full,bfcl_full"                      # the receipt is held to the launch's full-set controls
    assert "receipt <sha12>" in template[0] and "the one blank" in template[0]
    # --phase file with no receipt / verdict: refused by name (the dry run composes from a receipt; without one there is nothing to compose)
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--phase", "file", "--cells", "none,difference_lora"])
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED: no verdict at" in out and f"gate/bcb_bfcl/{view}.json" in out and "stage07_file.py" in out, out[-2500:]
    # --phase mirror with no receipt: the tool call composed (the primary cluster's ssh host and <reproduce root>/<model>/07_mirror as the destination),
    # then refused by name — the mirror pulls what the receipt names and nothing else; no Modal command, nothing written
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--phase", "mirror", "--cells", "none,difference_lora"])
    out = r.stdout + r.stderr
    mirror_lines = [ln for ln in _dry_lines(out) if "stage07_mirror.py" in ln]
    assert r.returncode == 2 and len(mirror_lines) == 1 and not _modal_lines(out), out[-2500:]
    argv = shlex.split(mirror_lines[0][len("DRY-RUN:"):])
    assert argv[argv.index("--ssh-host") + 1] == "cluster-a-login" and argv[argv.index("--cluster-label") + 1] == "cluster_a" and argv[-1] == "--dry-run"
    assert argv[argv.index("--cluster-dest") + 1].endswith(f"/exp/reproduce/{FD}/07_mirror") and argv[argv.index("--volume") + 1] == "twosi-fits"
    assert argv[argv.index("--cluster-dest") + 1] == f"{argv[argv.index('--cluster-root') + 1]}/{FD}/07_mirror", "the wave's destination lies directly under the guarded reproduce root (round 6)"
    assert argv[argv.index("--environment") + 1] == "main" and argv[argv.index("--state-root") + 1] == str(state)
    assert "REFUSED: no receipt at" in out and f"{state}/07_store/receipt.json" in out and "07 --phase receipt" in out and "the mirror phase refused" in out, out[-2500:]
    # the grammar
    r = _run("07_bcb_bfcl.sh", FD, site, extra=["--phase", "bogus"])
    assert r.returncode == 2 and "--phase must be stage|preflight|run|receipt|mirror|file|all" in r.stdout + r.stderr
    after = {str(p) for p in (site.parent / "state").rglob("*")} if (site.parent / "state").exists() else set()
    assert after == before, sorted(after - before)[:5]


def _committed_copy_for_07(tmp: Path) -> Path:
    """A clean, committed copy the REAL 07 phases run from (repro_require_clean_tree): reproduce/ copied (the table, the stage, its lib and
    tools), src/, scripts/, configs/ and objects/ linked (the launcher's contracts, the full22 stager, the pins, the committed objects the
    file phase's importer plan reads); the fake Modal CLI lives OUTSIDE it."""
    repo = tmp / "repo"
    shutil.copytree(REPRO, repo / "reproduce", symlinks=True)
    for d in ("src", "scripts", "configs", "objects"):                        # the launcher's contracts, the full22 stager the receipt reuses, the pins, the committed objects
        (repo / d).symlink_to(REPO / d)
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]
    subprocess.run([*g, "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run([*g, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*g, "commit", "-q", "-m", "stage 07 copy"], cwd=repo, check=True)
    return repo


FAKE_MODAL = r'''#!/usr/bin/env python3
# a fake `modal` CLI for stage 07's REAL phases: `run …` prints the fixture $FAKE_MODAL_STDOUT names and exits $FAKE_MODAL_RC (the launcher's own
# stdout as stage 07 captures it); `volume get/ls` serve a directory tree ($FAKE_MODAL_VOLUME_DIR) and REJECT any argument still carrying the
# container mount /vol/ (carried finding 1: a path conversion bug must fail loudly, never be served); with $FAKE_MODAL_EXPECT_ENV set, a volume
# command not scoped to that environment (`--env <it>` right after the verb) is REJECTED too (code review round 1: a volume name is per environment)
import json, os, shutil, sys
args = sys.argv[1:]
for a in args:
    if a == "/vol" or a.startswith("/vol/"):
        sys.stderr.write(f"fake modal: unconverted container path {a!r}\n"); sys.exit(9)
want_env = os.environ.get("FAKE_MODAL_EXPECT_ENV")
if want_env and args[:1] == ["volume"] and args[2:4] != ["--env", want_env]:
    sys.stderr.write(f"fake modal: a volume command not scoped to --env {want_env}: {args}\n"); sys.exit(8)
if args[:1] == ["run"]:
    sys.stdout.write(open(os.environ["FAKE_MODAL_STDOUT"], encoding="utf-8").read()); sys.stdout.flush()
    sys.exit(int(os.environ.get("FAKE_MODAL_RC", "0")))
root = os.environ["FAKE_MODAL_VOLUME_DIR"]
def humanize(value):      # modal/cli/volume.py humanize_filesize (1.5.4): what `volume ls --json` prints as `size`
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ZiB"
if args[:2] == ["volume", "get"]:
    src = os.path.join(root, args[-2].strip("/")); dst = args[-1]
    if os.path.isdir(src):
        # the CLI's DIRECTORY download (cli/_download.py): an EXISTING destination directory, the folder laid under it as <dst>/<folder name>/…;
        # a missing destination is taken as a file path and fails on the first directory entry
        if not os.path.isdir(dst):
            sys.stderr.write(f"[Errno 21] Is a directory: '{dst}'\n"); sys.exit(1)
        shutil.copytree(src, os.path.join(dst, os.path.basename(src.rstrip("/"))), dirs_exist_ok="--force" in args); sys.exit(0)
    if not os.path.isfile(src):
        sys.stderr.write("No such file or directory\n"); sys.exit(1)
    os.makedirs(dst, exist_ok=True); shutil.copy(src, os.path.join(dst, os.path.basename(src))); sys.exit(0)
if args[:2] == ["volume", "ls"]:
    rel = args[-1]; d = os.path.join(root, rel.lstrip("/"))
    if not os.path.exists(d):
        sys.stderr.write("No such file or directory\n"); sys.exit(1)
    if "--json" in args:
        ents = [{"filename": f"{rel.rstrip('/')}/{n}", "type": "dir" if os.path.isdir(os.path.join(d, n)) else "file",
                 "size": "0 B" if os.path.isdir(os.path.join(d, n)) else humanize(os.path.getsize(os.path.join(d, n))), "created_modified": "x"} for n in sorted(os.listdir(d))] \
            if os.path.isdir(d) else [{"filename": rel, "type": "file", "size": humanize(os.path.getsize(d)), "created_modified": "x"}]
        print(json.dumps(ents))
    else:
        print(rel)
    sys.exit(0)
sys.stderr.write(f"fake modal: unexpected {args}\n"); sys.exit(7)
'''


def _fake_modal(tmp: Path) -> Path:
    b = tmp / "fakebin"; b.mkdir(exist_ok=True)
    p = b / "modal"; p.write_text(FAKE_MODAL); p.chmod(0o755)
    return p


def _stage07_real(repo: Path, site: Path, *args: str, env: dict) -> subprocess.CompletedProcess:
    e = dict(os.environ); e.pop("SISTEER_SITE_DIR", None); e.pop("SISTEER_CONFIG", None); e.pop("REPRO_DRY_RUN", None)
    e.update({"SISTEER_STATE_DIR": str(site.parent / "state"), "MODAL_TOKEN_ID": "fake-id", "MODAL_TOKEN_SECRET": "fake-secret", **env})
    return subprocess.run(["bash", str(repo / "reproduce" / "07_bcb_bfcl.sh"), *args, "--config", str(site / "config.env")], capture_output=True, text=True, env=e,
                          cwd=str(repo), timeout=900)


def test_07_preflight_captures_the_launchers_report_and_derives_the_expect_hashes_in_both_shapes(tmp_path):
    """The REAL preflight phase against a fake Modal CLI that prints a fixture report. (i) The check PASSED: the hashes come from
    checks[object_identity_and_header_guard].cells[<token>].intervention_hash. (ii) The bootstrap: the check FAILED (every fitted cell with no
    expectation refused WITH its derived hash in the `; `-joined err string) and the launcher exited non-zero by design — the phase still writes
    <state>/07_expect_hashes.json with every fitted cell's hash and exits 0, saying so. Each derived hash is cross-checked against the hash the
    same bytes carry under the paper's seed slug (contracts.EXPECTED_INTERVENTION_HASHES; the seed from objects/DEMO.json, the ReLU difference
    under the PAIRED slug): the published values read `file (= qwen35_4b-f22s<N>[p])` — the table pinned here — a differing one `DIFFERS`, and a
    fitted cell the report never derived refuses the phase. Then `--phase run` defaults --expect-hashes to the state file, an explicit flag wins."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    repo = _committed_copy_for_07(tmp_path)
    fake = _fake_modal(tmp_path)
    site = write_site(tmp_path, extra_config=f"SISTEER_MODAL_PYTHON={fake}")
    state = tmp_path / "state" / FD
    view = M.baseline_view(FD)
    tokens = [t for _c, t in M.launcher_cells(FD, view)]
    fitted = [t for t in tokens if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
    # the table: token -> (seed slug, the published hash) — the literal pairs, read back off the contracts
    table = {"constv2ctl2-full__psr_ce_all": ("qwen35_4b-f22s3", "21bba17d6bff3e49"), "constv2-mix-full__lora_r4_ce": ("qwen35_4b-f22s3", "a785e725a5308e02"),
             "arditi_allpos": ("qwen35_4b-f22s1", "01f67b09e325976b"), "constv2-mix-full__psr_ce_all": ("qwen35_4b-f22s3", "98df2bad947576b7"),
             "constv2ctl2-full__lora_r4_ce": ("qwen35_4b-f22s3", "466f4c765d7a1756"), "constv2-mix-full__lora_r4_ce_minus_ctl": ("qwen35_4b-f22s3", "3cbf5fe2fe2ceea1"),
             "constv2-mix-full__psr_ce_all_minus_ctl": ("qwen35_4b-f22s3p", "f79e44339ae17129")}
    assert set(table) == set(fitted) and all(C.EXPECTED_INTERVENTION_HASHES[s][t] == h for t, (s, h) in table.items())
    assert "qwen35_4b_demo" not in C.EXPECTED_INTERVENTION_HASHES                                          # :252 stays true: the table gains no demo key

    def report(passed: bool, hashes: dict) -> str:
        if passed:
            chk = {"ok": True, "cells": {t: {"intervention_hash": h, "expected": h, "match": True, "pin_match": True} for t, h in hashes.items()}, "unverified": []}
        else:
            err = "; ".join(f"{t}: no expected HF post-load intervention hash for this cell — neither the contracts' inventory table nor --expect-hashes names one; "
                            f"the derived hash is '{h}' — a fitted object is verified against a stated identity or not served (E40 H1); pin it with --expect-hashes "
                            f"'{{\"{t}\": \"{h}\"}}' once that hash is known to be the published object's" for t, h in hashes.items())
            chk = {"ok": False, "err": "AssertionError: " + err}
        doc = {"sha": "x", "model": FD, "cells": tokens, "ok": passed, "checks": {"staged_tree_present": {"ok": True}, "object_identity_and_header_guard": chk}}
        return "[tinycap] preflight …\n[preflight] " + json.dumps(doc, indent=1) + "\n" + ("" if passed else "[tinycap] PREFLIGHT FAILED — fix all above before any GPU\n")

    published = {t: h for t, (s, h) in table.items()}
    derived_p, runnable = state / "07_expect_hashes.derived.json", state / "07_expect_hashes.json"
    for passed, rc in ((True, "0"), (False, "1")):
        fx = tmp_path / f"pf_{passed}.txt"; fx.write_text(report(passed, published))
        for f in (runnable, derived_p, state / "07_preflight_report.json"):
            f.unlink(missing_ok=True)
        r = _stage07_real(repo, site, "--model", FD, "--phase", "preflight", env={"FAKE_MODAL_STDOUT": str(fx), "FAKE_MODAL_RC": rc})
        out = r.stdout + r.stderr
        assert r.returncode == 0, (passed, out[-3000:])
        assert json.loads(runnable.read_text()) == published and json.loads(derived_p.read_text()) == published and (state / "07_preflight_report.log").read_text() == fx.read_text()
        assert json.loads((state / "07_preflight_report.json").read_text())["ok"] is passed
        cross = json.loads((state / "07_expect_hashes.cross_check.json").read_text())
        assert {t: (c["slug"], c["published"]) for t, c in cross.items()} == table and all(c["derived"] == c["published"] for c in cross.values())
        for t, (slug, h) in table.items():
            assert f"expectation: {t} -> file (= {slug})" in out, (t, out[-3000:])
        assert "DIFFERS" not in out and "absent" not in out.replace("absent (no published", "x") or "DIFFERS" not in out
        assert ("the preflight exited 1 (a bootstrap preflight refuses" in out) is (not passed) and "every one the published hash under its seed slug" in out
    # a hash that is not the published one under the seed slug: DIFFERS — a FINDING, never a pin: the phase refuses by name, the
    # derived hashes land in the derived file, the runnable expectation is NOT written and the earlier preflight's runnable file is removed, named
    assert runnable.is_file()
    differing = dict(published, **{"constv2ctl2-full__lora_r4_ce": "0000000000000000"})
    fx = tmp_path / "pf_diff.txt"; fx.write_text(report(False, differing))
    r = _stage07_real(repo, site, "--model", FD, "--phase", "preflight", env={"FAKE_MODAL_STDOUT": str(fx), "FAKE_MODAL_RC": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "expectation: constv2ctl2-full__lora_r4_ce -> DIFFERS: derived 0000000000000000, qwen35_4b-f22s3 publishes 466f4c765d7a1756" in out, out[-3000:]
    assert "REFUSED: the derived hashes are NOT pinned as the run's expectation" in out and "constv2ctl2-full__lora_r4_ce DIFFERS (derived 0000000000000000" in out
    assert "a finding, never a pin" in out and "explicit identity ruling" in out and f"removed {runnable}" in out and "REFUSED: tinycap: the preflight's derived hashes are NOT the run's expectation" in out
    assert not runnable.exists() and json.loads(derived_p.read_text()) == differing and json.loads((state / "07_expect_hashes.cross_check.json").read_text())["constv2ctl2-full__lora_r4_ce"]["derived"] == "0000000000000000"
    # so the run phase has NO default expectation after a DIFFERS preflight: it refuses by name before any Modal call (the dry run says so)
    r = _stage07_real(repo, site, "--dry-run", "--model", FD, "--phase", "run", env={})
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "expect_hashes=none (flag)" in out and "no expected intervention hash for fitted cell(s)" in out and "a real run refuses here" in out, out[-3000:]
    # a fitted cell the report never derived: the phase refuses (the launcher's refusal was for another reason than a missing expectation)
    fx = tmp_path / "pf_short.txt"; fx.write_text(report(False, {t: h for t, h in published.items() if t != "arditi_allpos"}))
    r = _stage07_real(repo, site, "--model", FD, "--phase", "preflight", env={"FAKE_MODAL_STDOUT": str(fx), "FAKE_MODAL_RC": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "expectation: arditi_allpos -> NOT DERIVED" in out and "arditi_allpos NOT DERIVED" in out and "NOT pinned" in out, out[-3000:]
    assert not runnable.exists() and "arditi_allpos" not in json.loads(derived_p.read_text())
    # a token under NO published slug (`absent`): the same refusal — a table whose demo row imports under a paper key with no published hash for the
    # demo's seed slugs (every fitted token is seed 3 and contracts has no gemma4e4b-f22s3 slug, so each is absent; the refusal direction falls
    # back to seed 1); nothing pinned
    tbl = tmp_path / "models_absent.yaml"
    tbl.write_text((REPRO / "models.yaml").read_text().replace("  committed_objects_of: qwen35_4b\n", "  committed_objects_of: gemma4e4b\n", 1))
    site2 = write_site(tmp_path / "absent", extra_config=f"SISTEER_MODAL_PYTHON={fake}\nSISTEER_MODELS_TABLE={tbl}")
    fx = tmp_path / "pf_absent.txt"; fx.write_text(report(False, published))
    r = _stage07_real(repo, site2, "--model", FD, "--phase", "preflight", env={"FAKE_MODAL_STDOUT": str(fx), "FAKE_MODAL_RC": "1"})
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "constv2-mix-full__psr_ce_all_minus_ctl -> absent (no published hash under gemma4e4b-f22s3p" in out and "NOT pinned" in out, out[-3000:]
    assert "absent (no published hash under" in out.split("REFUSED: the derived hashes", 1)[1] and not (tmp_path / "absent" / "state" / FD / "07_expect_hashes.json").exists()
    assert json.loads((tmp_path / "absent" / "state" / FD / "07_expect_hashes.derived.json").read_text()) == published
    # the run phase picks the state file up (dry run): the run line carries --expect-hashes <state file>, the header says so; an explicit flag wins
    fx = tmp_path / "pf_ok.txt"; fx.write_text(report(True, published))
    assert _stage07_real(repo, site, "--model", FD, "--phase", "preflight", env={"FAKE_MODAL_STDOUT": str(fx), "FAKE_MODAL_RC": "0"}).returncode == 0
    r = _stage07_real(repo, site, "--dry-run", "--model", FD, "--phase", "run", env={})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    run_argv = _launcher_argv(_modal_lines(out)[0])
    assert run_argv[run_argv.index("--expect-hashes") + 1] == f"{state}/07_expect_hashes.json" and "expect_hashes=" + f"{state}/07_expect_hashes.json (state" in out
    for t in fitted:
        assert f"expectation: {t} -> file" in out
    own = tmp_path / "own.json"; own.write_text(json.dumps({t: "0123456789abcdef" for t in fitted}))
    r = _stage07_real(repo, site, "--dry-run", "--model", FD, "--phase", "run", "--expect-hashes", str(own), env={})
    out = r.stdout + r.stderr
    run_argv = _launcher_argv(_modal_lines(out)[0])
    assert r.returncode == 0 and run_argv[run_argv.index("--expect-hashes") + 1] == str(own) and "(flag)" in out
    # the preflight phase itself never defaults to the state file (it derives)
    r = _stage07_real(repo, site, "--dry-run", "--model", FD, "--phase", "preflight", env={})
    assert "--expect-hashes" not in _launcher_argv(_modal_lines(r.stdout + r.stderr)[0])


def test_07_run_captures_the_recs_and_the_receipt_phase_pulls_the_volume_and_prints_the_verdict_command(tmp_path, monkeypatch):
    """The REAL run and receipt phases through the stage against the fake Modal CLI: `--phase run` tees the launcher's stdout (the RECs) to
    <state>/07_run_report.log and exits with the launcher's code (pipefail inside the string); `--phase receipt` runs stage07_receipt.py over
    the fake volume (served from a directory; every path converted — the fake refuses /vol/), writes <state>/07_store/receipt.json and prints
    the stage.py verdict command with the headline and the store keys filled in; `--phase file --dry-run` then composes the spec from them."""
    sys.path.insert(0, str(REPRO / "lib"))
    import models as M
    import test_stage07_analysis_filing as TF                               # the fixture world: RECs in the launcher's shape, the writer's store layout
    repo = _committed_copy_for_07(tmp_path)
    fake = _fake_modal(tmp_path)
    # a site whose Modal environment is NOT the CLI's default: the fake rejects any volume command not scoped to it; the primary
    # cluster's ssh is the `local:<dir>` seam, so the mirror phase's rsync and `sha256sum -c` run here against the site's local shared root
    site = write_site(tmp_path, ssh={"cluster_a": f"local:{tmp_path / 'fakeslurm'}"}, extra_config=f"SISTEER_MODAL_PYTHON={fake}\nSISTEER_MODAL_ENVIRONMENT=demo-env")
    state = tmp_path / "state" / FD
    view = M.baseline_view(FD)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    # the fixture world at THIS repo's sha (the receipt holds every REC to the tree the stage reads): the stores and pointers as the writer lays them out
    monkeypatch.setattr(TF, "SHA", sha); monkeypatch.setattr(TF, "APP", f"tinycap-{sha[:7]}-{FD}")   # restored after: the filing module's own tests read its literals
    w = TF.make_world(tmp_path / "fixture")
    vol_dir = tmp_path / "volume"
    for rel, data in w["vol"].remote.items():
        p = vol_dir / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    report = w["state"] / "07_run_report.log"
    env = {"FAKE_MODAL_STDOUT": str(report), "FAKE_MODAL_RC": "0", "FAKE_MODAL_VOLUME_DIR": str(vol_dir), "FAKE_MODAL_EXPECT_ENV": "demo-env"}
    expect_f = _expect_file(tmp_path, M, view)
    r = _stage07_real(repo, site, "--model", FD, "--cells", "none,difference_lora", "--phase", "run", "--expect-hashes", str(expect_f), env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert (state / "07_run_report.log").read_text() == report.read_text() and "followup:" in out and "--phase receipt" in out
    # the launch's identity, written beside the captured stdout before the launcher ran: the tree, the cells by both names,
    # the object id of every fitted cell at this tree (the importer's plan), the app, the volume and environment, the overrides document, the expect-hashes
    ident = json.loads((state / "07_run_identity.json").read_text())
    assert ident["schema"] == "stage07-run-identity/1" and ident["tree_sha"] == sha and ident["cells"] == ["none", "difference_lora"] and ident["launcher_cells"] == ["none", TF.DIFF]
    assert ident["objects"] == {TF.DIFF: TF.DIFF_ID} and ident["app"] == f"tinycap-{sha[:7]}-{FD}" and ident["volume"] == "twosi-fits" and ident["modal_environment"] == "demo-env"
    assert ident["run_overrides"]["id"] == "paper_defaults" and ident["run_overrides"]["sha256"] == hashlib.sha256((REPRO / "overrides" / "paper_defaults.env").read_bytes()).hexdigest()
    assert ident["controls"] == "bigcodebench_full,bfcl_full" and ident["sampling"] == "protocol" and ident["slug"] == FD      # the paper's sets and decoding, the launcher row
    assert ident["expect_hashes"] == {"path": str(expect_f), "sha256": hashlib.sha256(expect_f.read_bytes()).hexdigest(), "hashes": json.loads(expect_f.read_text())}
    assert out.index("run identity: ") < out.index("[tinycap] launching") and "objects for 1 of 1 fitted" in out      # written before the launcher ran
    # a launcher that exits non-zero: the stage refuses (pipefail: the launcher's exit, not tee's), the log still captured
    r = _stage07_real(repo, site, "--model", FD, "--cells", "none,difference_lora", "--phase", "run", "--expect-hashes", str(expect_f), env={**env, "FAKE_MODAL_RC": "3"})
    assert r.returncode == 2 and "REFUSED: tinycap: the run ended non-zero" in r.stderr and (state / "07_run_report.log").is_file()
    r = _stage07_real(repo, site, "--model", FD, "--cells", "none,difference_lora", "--phase", "run", "--expect-hashes", str(expect_f), env=env)
    assert r.returncode == 0
    # the receipt phase under ANOTHER overrides document than the launch's: refused by name, naming the run's document and its path; nothing pulled, no receipt
    other = tmp_path / "other.env"; other.write_text("OVERRIDE_ID=other_doc\n")
    r = _stage07_real(repo, site, "--model", FD, "--cells", "none,difference_lora", "--phase", "receipt", "--overrides", str(other), env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED: --override-id other_doc is not the run's: the run was launched under run-overrides paper_defaults" in out, out[-3000:]
    assert f"--overrides {repo / 'reproduce' / 'overrides' / 'paper_defaults.env'}" in out and not (state / "07_store").exists() and "volume get" not in out
    # the receipt phase, for real, over the fake volume
    r = _stage07_real(repo, site, "--model", FD, "--cells", "none,difference_lora", "--phase", "receipt", env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    rec = json.loads((state / "07_store" / "receipt.json").read_text())
    assert rec["tree_sha"] == sha and rec["headline"]["value"] == 0.455 and rec["store_cell_keys"] == sorted(w["keys"].values()) and rec["modal_call_ids"] == sorted(w["fcids"].values())
    assert (state / "07_store" / "none" / w["keys"]["none"] / "done" / "inspect__bigcodebench_full.json").is_file() and (state / "07_store" / "run_report.log").is_file()
    assert rec["controls"] == ["bigcodebench_full", "bfcl_full"] and rec["sampling"] == "protocol" and rec["slug"] == FD and rec["n_items_per_task"] == {"inspect/bigcodebench_full": 1140, "inspect/bfcl_full": 3981}
    assert not (state / "07_store.building").exists() and not (state / "07_store.archive").exists()
    # the receipt carries the LAUNCH's identity (copied from 07_run_identity.json, verified against this call), and the environment every pull was scoped to
    assert rec["view_cells"] == ["none", "difference_lora"] and rec["cells"] == ["none", TF.DIFF] and rec["run_overrides_id"] == "paper_defaults" and rec["modal_environment"] == "demo-env"
    assert rec["objects"] == {TF.DIFF: TF.DIFF_ID} and rec["run_overrides_sha256"] == ident["run_overrides"]["sha256"] and rec["expect_hashes"] == json.loads(expect_f.read_text())
    assert rec["run_identity_sha256"] == hashlib.sha256((state / "07_run_identity.json").read_bytes()).hexdigest()
    cmd = [ln for ln in out.splitlines() if "verdict (fill --range, the one blank): " in ln]
    assert len(cmd) == 1 and "--headline 0.455" in cmd[0] and f"--run-ids {','.join(sorted(w['keys'].values()))}" in cmd[0] and "--range <lo,hi>" in cmd[0]
    # rule it exactly as printed (the one blank filled), then the file phase's dry run composes the spec from the receipt and the verdict
    verdict_cmd = shlex.split(cmd[0].split("the one blank): ", 1)[1].replace("<lo,hi>", "0.3,0.6"))
    vr = subprocess.run(verdict_cmd, capture_output=True, text=True, timeout=120)
    assert vr.returncode == 0 and (state / "gate" / "bcb_bfcl" / f"{view}.json").is_file(), vr.stdout + vr.stderr
    rsha = hashlib.sha256((state / "07_store" / "receipt.json").read_bytes()).hexdigest()
    # the file phase REFUSES until the receipt's stores are mirrored (2026-10-12: every filing points at a durable copy), naming the exact mirror command
    r = _stage07_real(repo, site, "--dry-run", "--model", FD, "--cells", "none,difference_lora", "--phase", "file", env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and f"REFUSED: no verified mirror of receipt {rsha[:12]}" in out and f"--model {FD} --view {view} --cells none,difference_lora --phase mirror" in out, out[-3000:]
    assert not (state / "07_analysis_spec.json").exists()
    # the mirror phase, for real: the WHOLE stores off the fake volume's directory tree — laid down with what the receipt never pulls (the cell manifests at
    # this tree, RESULT, rows, logs) — verified against the receipt, published under <state>/07_mirror/<receipt sha12>/ and pushed over the site's cluster
    # ssh (the `local:<dir>` seam of lib/common.sh: the rsync target is the local shared root, `sha256sum -c MIRROR.sha256` runs here) to
    # <SISTEER_REPRODUCE_ROOT>/<model>/07_mirror/<receipt sha12>/; every volume command scoped to demo-env (the fake rejects any other)
    import stage07_testlib as TL
    TL.lay_down_store_extras(w["vol"].remote, rec)
    for rel, data in w["vol"].remote.items():
        p = vol_dir / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    r = _stage07_real(repo, site, "--model", FD, "--cells", "none,difference_lora", "--phase", "mirror", env={**env, **_rsync_env(tmp_path)})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    mirror = state / "07_mirror" / rsha[:12]
    cluster_copy = tmp_path / "shared_cluster_a" / "exp" / "reproduce" / FD / "07_mirror" / rsha[:12]
    assert mirror.is_dir() and (mirror / "MIRROR.sha256").is_file() and (mirror / "cluster.json").is_file() and not (state / "07_mirror.building").exists()
    assert {p.relative_to(mirror).as_posix(): p.read_bytes() for p in mirror.rglob("*") if p.is_file()} == {p.relative_to(cluster_copy).as_posix(): p.read_bytes() for p in cluster_copy.rglob("*") if p.is_file()}
    for cell, key in w["keys"].items():
        assert (mirror / cell / "RESULT").is_file() and (mirror / cell / "cells" / key / "manifest.json").is_file() and (mirror / cell / "cells" / key / "rows.jsonl").is_file()
    cl = json.loads((mirror / "cluster.json").read_text())
    assert cl["verified"] is True and cl["cluster_path"] == str(cluster_copy) and cl["cluster_label"] == "cluster_a" and cl["receipt_sha256"] == rsha
    assert json.loads((mirror / "mirror.json").read_text())["modal_environment"] == "demo-env" and "cluster copy verified:" in out and "--phase file" in out
    # a re-run verifies the published copy and does nothing else
    r = _stage07_real(repo, site, "--model", FD, "--cells", "none,difference_lora", "--phase", "mirror", env={**env, **_rsync_env(tmp_path)})
    assert r.returncode == 0 and "nothing else to do" in r.stdout and "volume get" not in r.stdout
    # the file phase's dry run now composes the spec from the receipt, the verdict and the mirror's two records; the site's mirror target rides along
    # as a cross-check the tool holds the cluster record to
    r = _stage07_real(repo, site, "--dry-run", "--model", FD, "--cells", "none,difference_lora", "--phase", "file", env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert f"--cluster-label cluster_a --ssh-host local:{tmp_path / 'fakeslurm'} --cluster-dest {tmp_path / 'shared_cluster_a'}/exp/reproduce/{FD}/07_mirror --dry-run" in out, out[-3000:]
    spec = json.loads((state / "07_analysis_spec.json").read_text())
    assert spec["extra"]["mirror"]["cluster_path"] == str(cluster_copy) and spec["extra"]["mirror"]["receipt_sha256"] == rsha
    assert spec["extra"]["mirror"]["mirror_sha256sum_digest"] == hashlib.sha256((mirror / "MIRROR.sha256").read_bytes()).hexdigest()
    assert spec["name"] == f"demo_import_07_bcb_bfcl_{view}_paper_defaults_x{rsha[:12]}" and spec["substrate"] == "modal" and spec["cluster_job_ids"] == []
    assert spec["code_sha"] == sha and spec["inputs"] == {"runs": [], "objects": sorted([TF.DIFF_ID, "none"])} and spec["extra"]["headline"]["value"] == 0.455
    assert spec["report"]["uri"] == f"{state}/gate/bcb_bfcl/{view}.json" and "DRY-RUN: registry analysis file --spec" in out and "--dry-run" in out
    assert spec["code_ref"] == f"demo-import-smoke @ {sha[:7]}" and spec["extra"]["modal_app_name"] == f"tinycap-{sha[:7]}-{FD}" and spec["extra"]["modal_environment"] == "demo-env"
    assert spec["extra"]["cells"] == ["none", "difference_lora"] and spec["extra"]["launcher_cells"] == ["none", TF.DIFF]
    # the filed objects are the receipt's (the launch tree's); this checkout IS at that tree, so the importer's live plan was cross-checked and agrees
    assert spec["extra"]["objects_by_cell"] == {TF.DIFF: TF.DIFF_ID} and spec["extra"]["run_overrides_sha256"] == ident["run_overrides"]["sha256"]
    assert "cross-checked against the importer's plan at HEAD" in out and f"HEAD {sha[:7]} == the receipt's tree: agree" in out, out[-3000:]
    # `--phase file` WITHOUT the run's --cells (every cell of the view, the default) against the two-cell receipt: refused by name, the spec untouched
    spec_bytes = (state / "07_analysis_spec.json").read_bytes()
    r = _stage07_real(repo, site, "--dry-run", "--model", FD, "--phase", "file", env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED: --cells " in out and "is not the receipt's" in out and "pass --cells none,difference_lora" in out, out[-3000:]
    assert (state / "07_analysis_spec.json").read_bytes() == spec_bytes


def _rsync_env(tmp: Path) -> dict:
    """The mirror phase pushes with the machine's `rsync`; where there is none, a stand-in on PATH that copies `-a --partial SRC/ DEST/` (no deletion)
    and `-a --partial FILE DEST/name` (the staged cluster record)."""
    if shutil.which("rsync"):
        return {}
    b = tmp / "fakebin"; b.mkdir(exist_ok=True)
    p = b / "rsync"
    p.write_text(f"#!{sys.executable}\nimport os, shutil, sys\nsrc, dst = sys.argv[-2].rstrip('/'), sys.argv[-1].rstrip('/')\n"
                 "if os.path.isfile(src):\n    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)\nelse:\n    shutil.copytree(src, dst, dirs_exist_ok=True)\n")
    p.chmod(0o755)
    return {"PATH": f"{b}:{os.environ.get('PATH', '')}"}


def _expect_file(tmp_path, M, view) -> Path:
    fitted = [t for _c, t in M.launcher_cells(FD, view) if C.parse_launcher_cell(t).kind == C.CELL_KIND_FITTED]
    f = tmp_path / "expect_run.json"
    f.write_text(json.dumps({t: "0123456789abcdef" for t in fitted}))
    return f


def test_09_forwards_force_gate_to_admission_and_records_the_forced_list(site_with_plan):
    """09 parses `--force-gate LWID=REASON` (repeatable) exactly as 06 does, forwards it on the admission line (the dry
    run's echo and the real call) and prints + records the forced list — in its header, in `DRY-RUN: write <state>/LADDER.<view>.json` per
    selected view, in its decision note and its done line; without the flag the admission line is byte-identical to before and the
    record carries `forced: none`."""
    site, write_plan = site_with_plan
    write_plan(complete=True)
    r = _run("09_dose_ladder.sh", FD, site, extra=["--step", "submit", "--evals", "static"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2000:]
    adm = [ln for ln in out.splitlines() if "lib/admission.py check" in ln]
    assert len(adm) == 1 and adm[0].endswith(f"--registry {site.parent / 'registry'}/registry.sqlite") and "--force-gate" not in adm[0], adm
    assert "forced: none" in out and out.count("DRY-RUN: write ") == 2 and "LADDER.d0p75.json" in out and "LADDER.d0p75_relu.json" in out and "decisions.md:" not in out
    r = _run("09_dose_ladder.sh", FD, site, extra=["--step", "submit", "--evals", "static", "--force-gate", "X=smoke", "--force-gate", "Y=why not"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2000:]
    adm = [ln for ln in out.splitlines() if "lib/admission.py check" in ln]
    assert len(adm) == 1 and adm[0].endswith("--force-gate X=smoke --force-gate Y=why not"), adm
    assert "forced: X (smoke), Y (why not)" in out and out.count("forced: X (smoke), Y (why not)") >= 2                 # the header and the done line
    assert "decisions.md: forced admissions for qwen35_4b_demo (d0p75 d0p75_relu): X (smoke), Y (why not)" in out
    assert "submitted " in out and "--export=" not in out
    # the record's bytes carry the list (the dry run says how many it would write; a real submit writes them — the forced 06 test below reads one)
    assert "LADDER.d0p75.json" in out
    # the force is admission's only: run_in_this_work is still refused on a real call (the table ships false)
    r = run_stage("09_dose_ladder.sh", "--model", FD, "--step", "submit", "--force-gate", "X=smoke", site=site, timeout=600)
    assert r.returncode == 2 and "run_in_this_work: false" in r.stderr and "admission.py" not in r.stdout


def test_06_dose_view_force_gate_prints_the_flag_and_the_forced_summary(site_with_plan):
    site, write_plan = site_with_plan
    write_plan(complete=True)
    r = _run("06_eval_waves.sh", FD, site, extra=["--view", "d0p75", "--family", "static", "--force-gate", "X=smoke"])
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2000:]
    adm = [ln for ln in out.splitlines() if "lib/admission.py check" in ln]
    assert len(adm) == 1 and adm[0].endswith("--force-gate X=smoke"), adm
    assert "forced: X (smoke)" in out and "DRY-RUN: write " in out and "WAVE.d0p75.json" in out
    # on the baseline wave the flag is moot: recorded as none, said so
    r = _run("06_eval_waves.sh", FD, site, extra=["--family", "mmlu", "--force-gate", "X=smoke"])
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "is moot on the baseline wave" in out and "forced: none" in out and "admission.py check" not in out, out[-2000:]


def test_06_records_the_forced_list_in_the_wave_record_against_the_fake_slurm_seam(tmp_path):
    """Against the fake-Slurm seam (CLUSTER_SSH=local:, the wave integration World) with the World's own
    frozen inventory cut to ONE unit, `06 --view d0p75 --family static --force-gate <unit>=smoke` admits (the unit FORCED), submits, and writes
    WAVE.d0p75.json with `forced == [{logical_work_id, reason}]`; the baseline wave's WAVE record carries `forced == []`; a `--units` call writes
    no WAVE file but prints the list and the decision note; admission's gate/forced.jsonl is still written under the state root alone."""
    import test_front_door_wave_integration as WI
    w = WI.World(tmp_path)
    view = "d0p75"
    (w.repo / "configs" / f"sdfbase_full22_{WI.MODEL}_{view}.yaml").write_text(
        f"private_run: sdfbase_full22_{WI.MODEL}_{view}\nexplicit_cells: [none, constv2-mix-full__lora_r4_ce_minus_ctl]\nbon:\n  n_max: 75\n  goal_subset: jbb\n")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]
    subprocess.run([*g, "add", "-A"], cwd=w.repo, check=True); subprocess.run([*g, "commit", "-q", "-m", "the dose view's config"], cwd=w.repo, check=True)
    w.sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True, check=True).stdout.strip()
    sys.path.insert(0, str(REPRO / "lib"))
    import stage as ST
    import models as M
    base = M.baseline_view(WI.MODEL)
    # the World's own freeze (05 --step freeze at the committed sha: the seeds' mmlu item set, the goal fixture, the Petri bundle), then the
    # inventory cut to its ONE mmlu unit — the contracts stay the freeze's, so the baseline submission's contract check reads its own tree
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    inv = w.inventory()
    units = [u for u in inv["units"] if u["logical_work_id"].endswith("/mmlu/mmlu")]
    assert len(units) == 1 and inv["code_sha"] == w.sha and inv["view"] == base
    lw = units[0]["logical_work_id"]
    ST.write_atomic(str(w.state / "gate" / "expected_units.json"), {**inv, "units": units, "n_units": len(units)})
    # a --units call first: admitted (forced), the selected unit placed, NO WAVE file, the list printed and noted for the decision log
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", view, "--family", "static", "--units", "static.t0", "--force-gate", f"{lw}=smoke")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert f"ok   {lw}: FORCED (smoke)" in out and "ADMITTED" in out and out.count("submitted ") == 1
    assert not (w.state / f"WAVE.{view}.json").exists() and "the wave file is NOT written" in out
    assert f"decisions.md: forced admissions for {WI.MODEL}/{view}: {lw} (smoke)" in out and f"forced: {lw} (smoke)" in out
    forced_log = [json.loads(l) for l in (w.state / "gate" / "forced.jsonl").read_text().splitlines()]
    assert forced_log and forced_log[-1]["logical_work_id"] == lw and forced_log[-1]["reason"] == "smoke" and not list((tmp_path / "registry").rglob("forced*"))
    # the full call: the WAVE record carries the forced list
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", view, "--family", "static", "--force-gate", f"{lw}=smoke")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    wave = json.loads((w.state / f"WAVE.{view}.json").read_text())
    assert wave["forced"] == [{"logical_work_id": lw, "reason": "smoke"}] and wave["view"] == view and wave["model"] == WI.MODEL and "forced: " in out
    # the --units placement stands (skipped); the family's other two units go out — lib/stage.py plans static as t0, t1 and the BoN store512 row
    assert out.count("skipped ") == 1 and out.count("submitted ") == 2, out[-2000:]
    # the baseline wave: admission is skipped and the record's forced list is empty — the flag moot, said so
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", base, "--family", "mmlu", "--force-gate", f"{lw}=smoke")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert json.loads((w.state / f"WAVE.{base}.json").read_text())["forced"] == [] and "is moot on the baseline wave" in out and "forced: none" in out
    # the tightenings reach the stage: an empty reason and an unknown unit refuse by name, nothing submitted
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", view, "--family", "controls", "--force-gate", f"{lw}=")
    assert r.returncode != 0 and "non-empty reason" in r.stderr and "submitted " not in r.stdout
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", view, "--family", "controls", "--force-gate", "nosuch/unit=smoke")
    assert r.returncode != 0 and "names no unit of the frozen inventory: 'nosuch/unit'" in r.stderr and "submitted " not in r.stdout


def test_05_stages_the_committed_objects_from_the_checkout_and_the_bank_path_still_refuses_by_name(site, tmp_path):
    """demo-import M1: the worked example's row declares `committed_objects_of`, so 05's objects are the paper's COMMITTED ones (objects/DEMO.json
    through reproduce/tools/import_objects.py cells) — the dry run prints one `pull committed:objects/<dir>` per staged id (the seven of the
    wave plus the two 0.75 dose copies), its pins step runs import_objects.py pins, nothing refuses and no traceback appears; `--from-bank`
    forces the fit-bank path, which refuses the unpinned derived objects by name (M2's exit-2 refusal, kept for the fit-your-own path).
    The committed configs/sdfbase_full22_pins.json carries the demo's block (the import wrote it), so the fit-bank path finds its pins there
    and plans; the refusal is driven against a pins file WITHOUT that block through SISTEER_PINS_FILE, the seam 05 honours."""
    r = _run("05_stage_roster.sh", FD, site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    assert "source=committed (objects/DEMO.json via reproduce/tools/import_objects.py; committed_objects_of=qwen35_4b)" in out
    pulls = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN: pull committed:objects/")]
    assert len(pulls) == 9 and sum(1 for ln in pulls if "/cells/d0p75__" in ln) == 2 and sum(1 for ln in pulls if "committed:objects/qwen35_4b_it/" in ln) == 1, pulls
    assert any(ln.startswith("DRY-RUN:") and "reproduce/tools/import_objects.py pins --model qwen35_4b_demo" in ln for ln in out.splitlines()), out[-2500:]
    assert "--pins-from" not in out and "REFUSED" not in out and "Traceback" not in out and "has no pin" not in out
    # the fit-bank path under the COMMITTED pins: the demo's block is there (the import wrote it), so the derived objects are pinned and the
    # dry run plans the fit bank's pulls — no `has no pin`, no committed source
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text(encoding="utf-8"))
    assert "qwen35_4b_demo" in pins, "the committed pins carry the demo's block (reproduce/tools/import_objects.py pins)"
    r = run_stage("05_stage_roster.sh", "--dry-run", "--from-bank", "--step", "stage", "--model", FD, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "source=fit bank on cluster_a" in out and "has no pin" not in out and "REFUSED" not in out, out[-2500:]
    assert "Traceback" not in out and "committed:" not in out
    # the fit-your-own path's refusal, kept: a pins file without the demo's block (a fresh row's state) refuses the unpinned derived objects by name
    unpinned = tmp_path / "pins_without_the_demo_block.json"
    unpinned.write_text(json.dumps({k: v for k, v in pins.items() if k != "qwen35_4b_demo"}, indent=2) + "\n", encoding="utf-8")
    r = run_stage("05_stage_roster.sh", "--dry-run", "--from-bank", "--step", "stage", "--model", FD, site=site,
                  env={"SISTEER_PINS_FILE": str(unpinned)}, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "source=fit bank on cluster_a" in out and "has no pin" in out and "REFUSED: 05: cells_json refused (above)" in out, out[-2500:]
    assert "Traceback" not in out and "committed:" not in out


@pytest.mark.parametrize("model", TEST_MODELS)
@pytest.mark.parametrize("stage", STAGES)
def test_dry_run_both_models(stage, model, site):
    before = {str(p) for p in (site.parent / "state").rglob("*")} if (site.parent / "state").exists() else set()
    r = _run(stage, model, site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, (stage, model, out[-2500:])
    # the runner's OWN paths (this checkout, its HOME, its interpreter and the interpreter's venv) are not hard-coded identifiers:
    # normalise them first, with the scrub the readme-sequence scanner shares (front_door_testlib.scrub_site)
    scan = scrub_site(out, tmp=site.parent)
    assert_no_tokens(scan, f"{stage} {model}")
    printed = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:")]
    for ln in printed:
        assert not PLACEHOLDER.search(ln), (stage, model, ln[:300])
    # a dry run touches nothing: the state dir is unchanged
    after = {str(p) for p in (site.parent / "state").rglob("*")} if (site.parent / "state").exists() else set()
    assert after == before, (stage, model, sorted(after - before)[:5])
    # the submission-bearing stages print the submitted-line shape for every sbatch they would run
    if stage in ("06_eval_waves.sh", "09_dose_ladder.sh"):
        assert any(ln.startswith("submitted ") for ln in out.splitlines()), (stage, model)
        assert "--export=" not in out, stage


def test_06_u0_all_enumerates_every_family_with_the_paper_geometry(site):
    r = _run("06_eval_waves.sh", FD, site)
    out = r.stdout
    lines = out.splitlines()
    subs = [ln for ln in lines if ln.startswith("submitted ")]
    units = [ln.split()[3] for ln in subs]
    # padding: FIVE levels, one job each (close round E P1)
    assert sorted(u for u in units if u.startswith("padding.")) == sorted(f"padding.p{p}" for p in (0, 8192, 16384, 32768, 65536))
    # multi-turn: per cluster ONE group = the attacker pool SIZED from its targets (one member per three: 6 targets -> 2 members)
    # + the target units PARTITIONED across the clusters (6 + 6, disjoint)
    for lab in ("cluster_a", "cluster_b"):
        mine = [ln for ln in subs if ln.split()[1] == lab]
        assert sum(1 for ln in mine if ".attacker" in ln) == 2, mine
        assert sum(1 for ln in mine if ln.split()[3].startswith("multiturn.") and ".attacker" not in ln) == 6, mine
    targets = sorted(u for u in units if u.startswith("multiturn.") and ".attacker" not in u)
    assert targets == sorted(f"multiturn.{atk}.{grp}" for grp in ("g012", "g34", "g56", "g78") for atk in ("crescendo", "pair", "actorattack"))   # each ONCE
    # the wave shell's own positional contracts on the ssh line; never --export=
    assert f"target_attacks.sbatch {FD} battery crescendo " in out and " 9 0,1,2'" in out
    # the single-phase wave carries steered cells beside its unsteered legs, so its padding family is the per-view steered one
    assert f"target_static_padded_obj.sbatch {FD} " in out and " 98304 65536 " in out and " grok420 3 all battery" in out
    # the statics in the SAMPLED lane's form: target_sampled.sbatch static_base_t<T> with the sampled config's run identity,
    # never the wave form; the BoN 512 store keeps the wave shell's third job; every unit declares its evals
    assert f"sampled/target_sampled.sbatch {FD} static_base_t0 " in out and f"sampled/target_sampled.sbatch {FD} static_base_t1 " in out
    assert f"f22s-{FD}-{FD}_battery-static_base_t0" in out
    assert f"target_static.sbatch {FD} battery pasttense" not in out and f"target_static.sbatch {FD} battery bon " in out
    assert "F22_REGISTRY_EVALS=pasttense,codeattack,renellm,grok420" in out and "F22_REGISTRY_EVALS=xstest,or_bench_hard" in out
    assert "--export=" not in out
    # the job-side portability + registry env rides the command line (incl. the site's QoS for the fit path's resubmit)
    assert "SISTEER_SHARED_ROOT=/scratch/example/shared" in out and "SISTEER_REGISTRY_SUBSTRATE=slurm" in out and "SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a" in out
    assert "--qos=normal --partition=gpu" in out and "SISTEER_CLUSTER_QOS=normal" in out
    # the pinned sha is mirrored on EVERY cluster before the first submission
    first_sub = next(i for i, ln in enumerate(lines) if ln.startswith("submitted "))
    mirrors = [i for i, ln in enumerate(lines) if "push -q ssh://" in ln and "sisteer-eval-" in ln]
    assert len(mirrors) == 2 and max(mirrors) < first_sub, lines[:first_sub]
    # the baseline wave (the single-phase recipe's one wave) is the gate's inventory: without a frozen one the dry run says so (a real run refuses)
    assert "no frozen inventory" in out and "single-phase recipe" in out and "admission.py check" not in out


def test_06_twin_refuses_multiturn_and_elects_single_gpu_units(site):
    r = _run("06_eval_waves.sh", FD, site, extra=["--twin"])                     # --family all includes multiturn
    assert r.returncode != 0 and "not implemented for multi-turn" in r.stderr and "submitted " not in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    r = run_stage("06_eval_waves.sh", "--dry-run", "--view", "battery", "--family", "static,controls", "--twin", "--model", FD, site=site, timeout=600)
    assert r.returncode == 0 and "DRY-RUN: election" in r.stdout, r.stdout[-800:] + r.stderr[-800:]


def test_steered_view_dry_run_names_the_gate_and_the_replan(site):
    # a view behind the baseline (the dose rung d0p75) passes the admission gate and needs the compute plan
    r = _run("06_eval_waves.sh", FD, site, extra=["--view", "d0p75", "--family", "static"])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "admission.py check" in r.stdout and "compute_plan.json" in r.stdout


@pytest.fixture()
def site_with_plan(tmp_path):
    """A site of its own (the module fixture's state dir must stay plan-less for the other dry runs) whose state carries the
    model's compute plan; the test writes the plan's `complete` flag + pending rows the way tools/replan_compute.py does (schema 3)."""
    site = tmp_path / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={tmp_path / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={tmp_path / 'registry'}\n")
    state = tmp_path / "state" / FD; state.mkdir(parents=True)

    def write_plan(complete: bool, pending=("06_arms_multiturn", "06_arms_padding")):
        (state / "compute_plan.json").write_text(__import__("json").dumps(
            {"schema": 3, "model": FD, "complete": complete, "pending_rows": [] if complete else list(pending), "probe_status": "certified" if complete else "pending"}))
    return site, write_plan


def test_steered_dry_runs_refuse_an_incomplete_compute_plan_by_name(site_with_plan):
    """A steered 06 wave and the 09 ladder are priced on the plan's serving rows, so they ask for a COMPLETE plan —
    the on-disk plan of the live run is `complete: false` with the two serving rows pending — and the dry run says so by
    name instead of composing lines it could not submit. The single-phase baseline wave is ungated by admission but priced too."""
    site, write_plan = site_with_plan
    write_plan(complete=False)
    r = _run("06_eval_waves.sh", FD, site, extra=["--view", "d0p75", "--family", "static"])
    assert r.returncode != 0 and "INCOMPLETE" in r.stderr and "06_arms_multiturn" in r.stderr and "06_arms_padding" in r.stderr, r.stdout[-800:] + r.stderr[-800:]
    assert "submitted " not in r.stdout
    r = _run("09_dose_ladder.sh", FD, site)
    assert r.returncode != 0 and "INCOMPLETE" in r.stderr and "06_arms_multiturn" in r.stderr, r.stdout[-800:] + r.stderr[-800:]
    assert "submitted " not in r.stdout
    # the single-phase baseline wave is NOT gated by admission, but its steered legs ARE priced on the plan: an incomplete plan refuses it too
    r = _run("06_eval_waves.sh", FD, site)                                   # --view battery --family all
    assert r.returncode != 0 and "INCOMPLETE" in r.stderr and "admission.py check" not in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    write_plan(complete=True)
    r = _run("06_eval_waves.sh", FD, site)
    assert r.returncode == 0 and "is complete" in r.stdout and any(ln.startswith("submitted ") for ln in r.stdout.splitlines()), r.stdout[-800:] + r.stderr[-800:]
    r = _run("06_eval_waves.sh", FD, site, extra=["--view", "d0p75", "--family", "static"])
    assert r.returncode == 0 and "compute plan" in r.stdout and "is complete" in r.stdout, r.stdout[-800:] + r.stderr[-800:]
    r = _run("09_dose_ladder.sh", FD, site)
    assert r.returncode == 0 and "is complete" in r.stdout, r.stdout[-800:] + r.stderr[-800:]


def test_09_refuses_the_refusal_steer_ladder_by_design(site):
    r = _run("09_dose_ladder.sh", FD, site, extra=["--objects", "refusal_direction"])
    assert r.returncode != 0 and "NOT run in this work" in (r.stdout + r.stderr)


def test_10_figures_executes_the_notebook_in_the_interpreters_own_kernel(site):
    """The notebook's metadata names the maintainer's kernelspec (`sdfbase-r2`, registered in ONE home directory); a fresh clone
    has no such kernel and `nbconvert --execute` refuses by name (NoSuchKernel -- the stage F fresh-clone check, 2026-09-29).
    10_figures.sh therefore pins `--ExecutePreprocessor.kernel_name=python3`, the kernelspec ipykernel itself provides inside
    $SISTEER_NB_PYTHON (the recipe tests/test_nb_fresh_home.py already uses), and never names the maintainer's kernel."""
    r = _run("10_figures.sh", TEST_MODELS[0], site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2000:]
    nb = [ln for ln in out.splitlines() if ln.startswith("DRY-RUN:") and " nbconvert " in ln]
    assert len(nb) == 1, out[-2000:]
    assert "--ExecutePreprocessor.kernel_name=python3" in nb[0], nb[0][:400]
    assert "kernel_name=sdfbase-r2" not in nb[0], nb[0][:400]     # the notebook venv's own PATH may carry the name; the kernel flag must not
