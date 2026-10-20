"""The corpus generator (corpus_generator/) is a member of the harness's uv workspace -- its own pyproject (deps, the `corpusgen`
console script, its tests), the repo's ONE uv.lock and venv (`uv sync --frozen --extra cpu --group corpus`; the harness pins
anthropic 1.0.0, the SDK line the generator needs, so the earlier separate-project arrangement is gone) -- and its offline suite passes
there: `uv run --directory <repo>/corpus_generator --frozen pytest tests` with
PYTHONPATH unset and every SISTEER_* variable scrubbed from the child environment, and every collected test path lies under
corpus_generator/tests/ (`--project` alone changes project discovery, not the cwd, and would collect the harness's own tests/ under
the corpus env). Also the project contract itself (one lock, gitignored data/, no machine names) and the front door's
stage 01 (`--help`, `--dry-run` printing the full corpusgen plan and the Batch-API coordination line, refusing bad flags).

Skips (never fails) when `uv` is absent; the suite run needs the generator in the venv (`--group corpus`; first sync is network)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from front_door_testlib import assert_no_tokens, token_hits   # the ONE token table the publication scan reads

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "corpus_generator"
STAGE = REPO / "reproduce/01_make_corpora.sh"
UV = shutil.which("uv")
VENV = Path(os.environ.get("UV_PROJECT_ENVIRONMENT") or REPO / ".venv")   # the workspace venv (uv honours the override)


def _generator_installed() -> bool:
    py = VENV / "bin/python"
    return py.exists() and subprocess.run([str(py), "-c", "import corpus_generator"], capture_output=True).returncode == 0


def _child_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SISTEER_") and k != "PYTHONPATH"}
    env.update(extra)
    return env


def test_generator_tree_is_a_workspace_member_with_gitignored_data():
    py = (CORPUS / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "corpus-generator"' in py and "anthropic>=1.0" in py
    assert not (CORPUS / "uv.lock").exists(), "ONE lock: the root uv.lock resolves the generator (the D10 pin conflict is gone)"
    assert not re.search(r"^\[tool\.uv\]", py, re.M), "a member's own tool.uv cutoff is dead text; the root's exclude-newer governs the one lock"
    gi = (CORPUS / ".gitignore").read_text(encoding="utf-8").split()
    assert "data/" in gi and ".venv/" in gi
    readme = (CORPUS / "README.md").read_text(encoding="utf-8")
    # the former project name is assembled here so that no file of the tree spells it (the rule: no reference anywhere)
    for banned in ("vendor", "upstream", "private repositor", "constitution" + "-sdf", "constitution" + "_sdf", "f64cfd5"):
        assert banned not in readme.lower(), banned
    for d in ("src/corpus_generator", "prompts", "prompts-control", "configs", "constitution", "constitution-control-v2", "tests", "scripts"):
        assert (CORPUS / d).is_dir(), d
    assert (CORPUS / "constitution-control-v2/constitution.md").is_symlink()
    root_py = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'members = ["corpus_generator"]' in root_py and "corpus-generator = { workspace = true }" in root_py
    assert 'corpus = ["corpus-generator"]' in root_py, "the generator rides the `corpus` dependency group"
    assert re.search(r'"anthropic==1\.\d+\.\d+"', root_py), "the harness and the generator share one anthropic 1.x pin"
    root_py_code = "\n".join(line.split("#", 1)[0] for line in root_py.splitlines())  # comments may name a dep; tables may not
    assert "datasketch" not in root_py_code, "the generator's deps stay declared in corpus_generator/pyproject.toml, not duplicated here"
    root_lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "corpus-generator"' in root_lock and 'name = "datasketch"' in root_lock, "the corpus deps resolve in the ONE lock"
    assert re.search(r'name = "anthropic"\nversion = "1\.', root_lock)
    assert 'requires-python = ">=3.11"' in root_lock.split("\n[[package]]", 1)[0], "the member's floor must not narrow the lock"


def test_no_machine_names_in_the_generator_code():
    hits = []
    for p in CORPUS.rglob("*"):
        if not p.is_file() or ".venv" in p.parts or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        if p.suffix not in (".py", ".md", ".json", ".yaml", ".yml", ".toml", ".sh", ".txt"):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        hits.extend((p.relative_to(REPO).as_posix(), tok) for tok in token_hits(text))
    assert not hits, hits


def test_identifier_sites_are_parameterised():
    mirror = (CORPUS / "src/corpus_generator/mirror.py").read_text()
    assert "SISTEER_CORPUS_MIRROR_HOST" in mirror and "SISTEER_CORPUS_MIRROR_ROOT" in mirror and "def resolve_dest" in mirror
    cli = (CORPUS / "src/corpus_generator/cli.py").read_text()
    assert '"--host", default=None' in cli and "--revision" in cli and "--hf-home" in cli
    client = (CORPUS / "src/corpus_generator/client.py").read_text()
    assert "SISTEER_SECRETS_MODE" in client and "def env_mode_key" in client
    snap = json.loads((CORPUS / "constitution/SNAPSHOT.json").read_text())
    assert snap["cross_check"]["reference"] == "~/const.txt"
    scr = (CORPUS / "scripts/section_coverage_report.py").read_text()
    assert "SISTEER_FIGS_URL" in scr


@pytest.mark.skipif(UV is None, reason="uv not on PATH")
def test_generator_suite_collects_only_its_tests_and_passes():
    if not _generator_installed() and os.environ.get("SISTEER_CORPUS_VENDOR_SYNC") != "1":
        pytest.skip("the corpus group is not in the venv (run reproduce/01_make_corpora.sh --setup, or set SISTEER_CORPUS_VENDOR_SYNC=1)")
    base = [UV, "run", "--directory", str(CORPUS), "--frozen", "pytest", "tests", "-q", "-p", "no:cacheprovider"]
    col = subprocess.run(base + ["--collect-only", "-q"], env=_child_env(), capture_output=True, text=True, timeout=900)
    assert col.returncode == 0, col.stdout[-2000:] + col.stderr[-2000:]
    # `-q` twice prints one `<path>: <n>` line per module; once prints node ids `<path>::<test>` — accept both
    paths = set()
    for ln in col.stdout.splitlines():
        if "::" in ln:
            paths.add(ln.split("::")[0])
        else:
            m = re.match(r"^(\S+\.py): \d+$", ln.strip())
            if m:
                paths.add(m.group(1))
    assert paths and all(p.startswith("tests/") for p in paths), sorted(paths)[:10]
    assert not any("test_corpora_shards" in p or "test_objects_committed" in p for p in paths), "the harness's tests leaked into the corpus env"
    run = subprocess.run(base, env=_child_env(), capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    m = re.search(r"(\d+) passed", run.stdout)
    assert m and int(m.group(1)) >= 299, run.stdout[-500:]
    assert "failed" not in run.stdout.splitlines()[-1]


# ----------------------------------------------------------------------------------------------- stage 01 front door
#: a synthetic coordination channel: the site names one, and only then is the ready-to-post line printed
CHANNEL = "#example-api-coordination"


@pytest.fixture
def site(tmp_path):
    d = tmp_path / "site"
    (d / "clusters").mkdir(parents=True)
    (d / "config.env").write_text(
        'SISTEER_CLUSTERS="cluster_a"\nSISTEER_PRIMARY_CLUSTER=cluster_a\nSISTEER_SECRETS_MODE=env\n'
        f'SISTEER_REGISTRY_ROOT={tmp_path}/registry\nSISTEER_GRADING_HOST=local\n'
        f'SISTEER_COORDINATION_CHANNEL="{CHANNEL}"\n')
    (d / "clusters/cluster_a.env").write_text(
        "CLUSTER_SSH=cluster-a-login\nCLUSTER_SHARED_ROOT=/shared/example\nCLUSTER_PARTITION=general\n"
        "CLUSTER_QOS=normal\nCLUSTER_SUBSTRATE=slurm\nCLUSTER_LABEL=cluster_a\n")
    return d


def _stage(*args, env=None, timeout=120):
    return subprocess.run(["bash", str(STAGE), *args], env=env or _child_env(), capture_output=True, text=True, timeout=timeout)


def test_stage01_help_exits_zero_without_a_config():
    r = _stage("--help")
    assert r.returncode == 0 and "--setup" in r.stdout and "--preflight" in r.stdout and "--phase" in r.stdout and "--dry-run" in r.stdout


def test_stage01_dry_run_prints_the_full_plan_and_coordination_line(site):
    r = _stage("--config", str(site / "config.env"), "--corpus", "all", "--phase", "all", "--dry-run")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = r.stdout
    for needle in ("corpusgen run --config", "configs/production.yaml --corpus advice", "configs/production.yaml --corpus stories",
                   "corpusgen filter --config", "corpusgen assemble-v2 --config", "configs/production-v2.yaml", "--drop-principle p35",
                   "configs/control-scale.yaml --corpus advice", "corpusgen import-scan --config", "corpusgen register-edit plan --config",
                   "corpusgen register-edit run --config", "corpusgen register-edit assemble --config", "configs/control-scale-r2.yaml",
                   f"ready-to-post for {CHANNEL}", "claude-opus-5 batch", "posting is the user's job; proceeding now",
                   "uv run --directory", "--frozen corpusgen"):
        assert needle in out, needle
    assert out.count("DRY-RUN: ") >= 15
    # the coordination line follows the `<N> <model> <prio> <duration>` convention with a computed or placeholder N
    assert re.search(r"^(\d+|<N>) claude-opus-5 batch ~\d+h", out, re.M), out
    assert "DRY RUN" in out and not re.search(r"^\S*corpusgen run ", out, re.M), "nothing may execute under --dry-run"
    assert_no_tokens(out.replace(str(REPO), "<repo>"), "the corpus generator's dry run")


def test_stage01_without_a_coordination_channel_prints_no_ready_to_post_line(site):
    """The channel is site config: a site that names none gets the full plan and no ready-to-post line at all."""
    cfg = site / "config.env"
    cfg.write_text("".join(ln + "\n" for ln in cfg.read_text().splitlines() if not ln.startswith("SISTEER_COORDINATION_CHANNEL=")))
    r = _stage("--config", str(cfg), "--corpus", "all", "--phase", "all", "--dry-run")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "corpusgen run --config" in r.stdout and "corpusgen register-edit run --config" in r.stdout
    assert "ready-to-post" not in r.stdout and "posting is the user's job" not in r.stdout, r.stdout
    assert not re.search(r"^(\d+|<N>) claude-opus-5 batch ~\d+h", r.stdout, re.M)


def test_stage01_dry_run_phase_and_corpus_selection(site):
    r = _stage("--config", str(site / "config.env"), "--corpus", "control", "--phase", "register-edit", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "register-edit plan" in r.stdout and "corpusgen run --config" not in r.stdout and "production.yaml" not in r.stdout
    assert "lowprio" in r.stdout  # the live edit runner's coordination line
    r = _stage("--config", str(site / "config.env"), "--corpus", "constitution", "--phase", "register-edit", "--dry-run")
    assert r.returncode == 0 and "no register-edit pass" in r.stdout and "DRY-RUN: " not in r.stdout


def test_stage01_setup_and_test_and_preflight_dry_run(site):
    r = _stage("--config", str(site / "config.env"), "--setup", "--test", "--preflight", "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "uv sync --frozen --inexact --group corpus" in r.stdout and "pytest tests -q" in r.stdout
    assert "stage-tokenizers --out" in r.stdout and "--hf-home" in r.stdout
    # the committed pins file rides the staging line: repo + revision + every file sha256 are verified by `corpusgen`
    #, so no per-family --revision is spelled a second time
    staging = [ln for ln in r.stdout.splitlines() if "stage-tokenizers" in ln]
    assert len(staging) == 1 and "--pins" in staging[0] and staging[0].rstrip().endswith("corpora/TOKENIZERS.json"), staging
    assert "--revision" not in staging[0]
    assert r.stdout.count("hf download ") == 3                            # one repo per fetch call
    assert "hf download meta-llama/Llama-3.1-8B-Instruct --revision" in r.stdout and "corpusgen preflight --config" in r.stdout
    assert "HF_TOKEN=" in r.stdout and "secret_hf" in r.stdout, "the token rides the env of one command, via secrets.sh"


GENERATION_MARKERS = ("corpusgen run", "corpusgen filter", "corpusgen regen-", "corpusgen assemble-v2", "corpusgen import-scan", "corpusgen register-edit",
                      "ready-to-post for")


@pytest.mark.parametrize("corpus,cfg,other", [("constitution", "production.yaml", "control-scale.yaml"),
                                              ("control", "control-scale.yaml", "production.yaml")])
def test_stage01_preflight_with_a_corpus_never_generates(site, corpus, cfg, other):
    """`--preflight --corpus X` used to run the preflight and then fall through into
    `corpusgen run` + every generation phase because --phase defaulted to `all`. --corpus only selects; generation needs --phase.
    An env-mode site with NO Anthropic key runs the offline preflight (the key gate is generation's alone)."""
    r = _stage("--config", str(site / "config.env"), "--preflight", "--corpus", corpus, "--dry-run",
               env=_child_env(ANTHROPIC_API_KEY=""))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = r.stdout
    assert f"corpusgen preflight --config" in out and cfg in out and other not in out
    assert "stage-tokenizers --out" in out and out.count("hf download ") == 3
    for marker in GENERATION_MARKERS:
        assert marker not in out, f"--preflight --corpus {corpus} planned generation: {marker!r}"
    assert "stage 01: corpus=" not in out and "ANTHROPIC_API_KEY" not in r.stderr


def test_stage01_corpus_without_an_action_refuses_and_names_both(site):
    r = _stage("--config", str(site / "config.env"), "--corpus", "constitution", "--dry-run")
    assert r.returncode != 0 and "--corpus constitution selects a family" in r.stderr
    assert "--preflight" in r.stderr and "--phase" in r.stderr
    assert "corpusgen" not in r.stdout
    # and --phase without a corpus family to act on
    r = _stage("--config", str(site / "config.env"), "--phase", "advice", "--dry-run")
    assert r.returncode != 0 and "--phase advice needs --corpus" in r.stderr and "corpusgen" not in r.stdout


def test_stage01_refusals(site):
    r = _stage("--config", str(site / "config.env"))
    assert r.returncode != 0 and "nothing to do" in r.stderr
    r = _stage("--config", str(site / "config.env"), "--corpus", "bogus", "--dry-run")
    assert r.returncode != 0 and "--corpus must be" in r.stderr
    r = _stage("--config", str(site / "config.env"), "--corpus", "all", "--phase", "bogus", "--dry-run")
    assert r.returncode != 0 and "--phase must be" in r.stderr
    r = _stage("--config", str(site / "nope.env"), "--corpus", "all", "--phase", "all", "--dry-run")   # a generation ask, no site
    assert r.returncode != 0 and "no site config" in r.stderr
    # a real generation phase under env mode needs the key; the refusal names it and nothing runs
    r = _stage("--config", str(site / "config.env"), "--corpus", "constitution", "--phase", "advice",
               env=_child_env(ANTHROPIC_API_KEY=""))
    assert r.returncode != 0 and "ANTHROPIC_API_KEY" in r.stderr and "corpusgen run" not in r.stdout


# ----------------------------------------------------------------------------------- stage 01 under wif (round 2 on 4574e103c)
@pytest.fixture
def wif_site(site, tmp_path):
    cfg = site / "config.env"
    cfg.write_text(cfg.read_text().replace("SISTEER_SECRETS_MODE=env", "SISTEER_SECRETS_MODE=wif"))
    ids = tmp_path / "ids"; ids.mkdir()
    return site, ids


def test_stage01_under_wif_plans_generation_keyless_and_refuses_an_absent_ids_file_by_name(wif_site):
    """The front door admitted wif for every stage while the generator's client refused it and the preflight asked secrets.sh for an
    HF token from the source `wif` (refused, swallowed by `|| true`: a gated tokenizer fetched tokenless). Now: a wif dry run plans
    the whole generation with no key anywhere (no ANTHROPIC_API_KEY ask, the mode on the stage line), the preflight still routes the
    HF token through secret_hf (op under wif), and a REAL generation phase whose pool's federation ids file is absent or empty is
    refused by file name before any corpusgen spawn -- the batch pool's for advice / stories, the lowprio pool's for a live
    register-edit, the batch pool's again for a batch-engine one."""
    site, ids = wif_site
    env = _child_env(SISTEER_WIF_ENV_DIR=str(ids))
    env.pop("ANTHROPIC_API_KEY", None)
    r = _stage("--config", str(site / "config.env"), "--corpus", "all", "--phase", "all", "--dry-run", env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "secrets=wif" in r.stdout and "corpusgen run --config" in r.stdout and "register-edit run" in r.stdout
    assert "ANTHROPIC_API_KEY" not in r.stdout + r.stderr
    r = _stage("--config", str(site / "config.env"), "--preflight", "--dry-run", env=env)
    assert r.returncode == 0 and "HF_TOKEN=" in r.stdout and "secret_hf" in r.stdout and "ANTHROPIC_API_KEY" not in r.stderr
    # a real phase: the ids file of the pool it bills must be present and non-empty (named as the vault names them)
    from sisteer import vault
    batch_ids, lo_ids = (Path(vault.wif_env_path(o, env={vault.WIF_ENV_DIR_ENV: str(ids)})) for o in ("batch", "lowprio"))
    r = _stage("--config", str(site / "config.env"), "--corpus", "constitution", "--phase", "advice", env=env)
    assert r.returncode != 0 and "SISTEER_SECRETS_MODE=wif" in r.stderr and str(batch_ids) in r.stderr and "batch pool" in r.stderr
    assert "corpusgen" not in r.stdout and "ANTHROPIC_API_KEY" not in r.stderr
    batch_ids.write_text("")
    r = _stage("--config", str(site / "config.env"), "--corpus", "control", "--phase", "stories", env=env)
    assert r.returncode != 0 and str(batch_ids) in r.stderr and "absent or empty" in r.stderr and "corpusgen" not in r.stdout
    r = _stage("--config", str(site / "config.env"), "--corpus", "control", "--phase", "register-edit", env=env)
    assert r.returncode != 0 and str(lo_ids) in r.stderr and "lowprio pool" in r.stderr and "corpusgen" not in r.stdout
    lo_ids.write_text("ANTHROPIC_ORGANIZATION_ID=org_synthetic\n")
    r = _stage("--config", str(site / "config.env"), "--corpus", "control", "--phase", "register-edit", "--edit-engine", "batch", env=env)
    assert r.returncode != 0 and str(batch_ids) in r.stderr and "corpusgen" not in r.stdout
