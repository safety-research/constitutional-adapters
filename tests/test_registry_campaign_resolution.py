"""The registry campaign has ONE resolution across every face of the front door — the site's SISTEER_REGISTRY_CAMPAIGN, else the job-env
spelling REGISTRY_CAMPAIGN, else the paper frame's sdfbase_full22 — and the JOB side reads the same one as the stages do. 03/04/05 spell it on
their CAMPAIGN= lines (04 reads every unit's done-ness under it; 05 hands it to the pins writer) and the pins writer mirrors it in Python
(make_full22_configs.registry_campaign_default). The job side is the same word:
lib/common.sh repro_registry_campaign, called by lib/slurm.sh slurm_job_env (the env of every job the no-lib fallback and the derived build
compose) and slurm_base_env (F22_REGISTRY_CAMPAIGN on every eval job) and by 08_grade.sh's daemon `--campaign`; and submit_twin.Site.
registry_campaign, which Coordinator.job_env states on every sbatch the funnel issues.

Before the fix, the job side read the site's key alone —
`${SISTEER_REGISTRY_CAMPAIGN:-sdfbase_full22}` in slurm.sh and 08, `cfg.get("SISTEER_REGISTRY_CAMPAIGN") or "sdfbase_full22"` in
submit_twin — so with REGISTRY_CAMPAIGN alone naming a site's campaign, 04 read done-ness and the pins writer read the registry under that
campaign while every fit job filed under sdfbase_full22: the completed fit got no pin and 05 refused it. Every face is EXECUTED here over one
table (the form is run, not string-matched), the two-level spelling is swept from the front door, and the 04 and 08 dry
runs are read as an operator would run them, with REGISTRY_CAMPAIGN alone. Synthetic site (cluster_a / cluster_b); nothing here names a
real host, user or path.

Integration round 2's review (Major): the funnel's Site parsed `<site dir>/config.env` whatever file `--config` named, so two config
files in one site directory — an experiment's beside config.env — put the stage's done-ness read under one campaign (the stage sources the
file it is given) and every job the funnel composed under the other's. The file GIVEN now decides — its campaign, its cluster list, its
primary — on both faces (submit_twin.Site through clusters.site_config_file; the shell's block load through clusters.py --config), and a
real 04 dry run through the second file is read here.

Review r2 of the same round (Major): the funnel consulted the given file for SISTEER_REGISTRY_CAMPAIGN alone and the process
environment for REGISTRY_CAMPAIGN, so `REGISTRY_CAMPAIGN=alt_campaign` written in the named file resolved to sdfbase_full22 in a direct
`submit_twin.py --config <file> submit` (no stage having sourced the file into the environment), while the stage's resolver answered
alt_campaign over the same file. The funnel now lays the given file over the environment for BOTH keys before the precedence, as `set -a;
source` does; every file spelling is run against every environment spelling over both faces, and the direct CLI's composed line is read."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, run_stage, write_site  # noqa: E402

MODEL = "qwen35_4b_demo"
CAMPAIGN_KEYS = ("SISTEER_REGISTRY_CAMPAIGN", "REGISTRY_CAMPAIGN")
#: the one expression every shell face spells (03/04/05's CAMPAIGN= lines; common.sh's resolver)
EXPR = "${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"
#: the table every face is run over: (SISTEER_REGISTRY_CAMPAIGN, REGISTRY_CAMPAIGN) -> the resolved campaign; None = unset
TABLE = (
    ((None, None), "sdfbase_full22"),           # nothing named: the paper frame's
    ((None, "alpha"), "alpha"),                 # the job-env spelling alone names the site's campaign — the finding's case
    (("beta", "alpha"), "beta"),                # the site's key wins
    (("", "alpha"), "alpha"),                   # an empty site key is unset, as bash's ${X:-} reads it
    ((None, "two_si"), "two_si"),               # the funnel's old default, allowed only when a caller names it
)


def _base_env() -> dict:
    """The inherited environment less the two campaign keys, so a developer's shell cannot decide a row of the table."""
    return {k: v for k, v in os.environ.items() if k not in CAMPAIGN_KEYS}


def _with(env: dict, sisteer: "str | None", job: "str | None") -> dict:
    e = dict(env)
    if sisteer is not None:
        e["SISTEER_REGISTRY_CAMPAIGN"] = sisteer
    if job is not None:
        e["REGISTRY_CAMPAIGN"] = job
    return e


def _bash(script: str, env: dict) -> str:
    r = subprocess.run(["bash", "-c", f"set -u\n{script}"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def _campaign_line(stage: str) -> str:
    lines = [ln for ln in (REPRO / stage).read_text().splitlines() if ln.startswith("CAMPAIGN=") or ln.startswith("  CAMPAIGN=")]
    assert len(lines) == 1, (stage, lines)
    return lines[0].strip()


def _sh(site: Path, body: str, env: dict) -> str:
    """The shell library loaded as a stage loads it (common.sh, slurm.sh, the site's config sourced by repro_load_config), then `body`."""
    e = dict(env, SISTEER_STATE_DIR=str(site.parent / "state"))
    return _bash(f"source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={site}/config.env; repro_load_config; {body}", e)


def _submit_twin():
    spec = importlib.util.spec_from_file_location("submit_twin_under_test", LIB / "submit_twin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the resolver, the stage lines and the pins writer: one table ----------------------------------------------------------------------

def test_the_shell_resolver_and_every_stage_line_resolve_the_one_table():
    """common.sh repro_registry_campaign, executed, agrees with 03/04/05's own CAMPAIGN= lines on every row — and IS the same expression, so
    the stage that reads done-ness and the library that composes the job cannot drift apart again."""
    lines = {stage: _campaign_line(stage) for stage in ("03_hpo.sh", "04_fit_objects.sh", "05_stage_roster.sh")}
    common = (REPRO / "lib" / "common.sh").read_text()
    m = re.search(r"^repro_registry_campaign\(\)\{ printf '%s' \"(.*)\"; \}$", common, re.M)
    assert m and m.group(1) == EXPR, "the resolver spells the stages' expression"
    for stage, line in lines.items():
        assert line == f'CAMPAIGN="{EXPR}"', (stage, line)
    for (sisteer, job), want in TABLE:
        env = _with(_base_env(), sisteer, job)
        assert _bash(f"source {REPRO}/lib/common.sh; repro_registry_campaign", env) == want, (sisteer, job)
        for stage, line in lines.items():
            assert _bash(f'{line}\necho "$CAMPAIGN"', env) == want, (stage, sisteer, job)


def test_the_pins_writers_python_resolution_is_the_same_table(monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("make_full22_configs_under_test", REPO / "scripts" / "sdfbase" / "make_full22_configs.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    for (sisteer, job), want in TABLE:
        for k in CAMPAIGN_KEYS:
            monkeypatch.delenv(k, raising=False)
        if sisteer is not None:
            monkeypatch.setenv("SISTEER_REGISTRY_CAMPAIGN", sisteer)
        if job is not None:
            monkeypatch.setenv("REGISTRY_CAMPAIGN", job)
        assert gen.registry_campaign_default() == want, (sisteer, job)


# ---- the job side: lib/slurm.sh's two composers, submit_twin's Site, 08's daemon -------------------------------------------------------

def test_slurm_job_env_and_slurm_base_env_state_the_resolved_campaign(tmp_path):
    """RED before the fix on the `(None, 'alpha')` row: both composers read `${SISTEER_REGISTRY_CAMPAIGN:-sdfbase_full22}` and stated
    sdfbase_full22 while 04's CAMPAIGN= line read alpha."""
    site = write_site(tmp_path)
    for (sisteer, job), want in TABLE:
        if sisteer is not None:
            continue                                  # the site's key travels through config.env below, as a site writes it
        out = _sh(site, "slurm_job_env cluster_a; echo; slurm_base_env", _with(_base_env(), None, job))
        job_env, base_env = out.split("\n")[0], out.split("\n")[-1]
        assert re.search(rf"(^|\s)REGISTRY_CAMPAIGN={want}(\s|$)", job_env), (job, job_env)
        assert "REGISTRY_CAMPAIGN=sdfbase_full22" not in job_env or want == "sdfbase_full22", job_env
        assert base_env.startswith(f"F22_REGISTRY_CAMPAIGN={want} "), (job, base_env)
    # the site names its campaign in config.env: it wins over the job-env spelling in the environment — exactly as 04's own line reads
    # it after repro_load_config sources the file
    site2 = write_site(tmp_path / "two", extra_config="SISTEER_REGISTRY_CAMPAIGN=paper_frame\n")
    out = _sh(site2, "slurm_job_env cluster_b; echo; slurm_base_env", _with(_base_env(), None, "alpha"))
    assert re.search(r"(^|\s)REGISTRY_CAMPAIGN=paper_frame(\s|$)", out.split("\n")[0]) and "alpha" not in out, out
    assert out.split("\n")[-1].startswith("F22_REGISTRY_CAMPAIGN=paper_frame "), out
    line = _campaign_line("04_fit_objects.sh")
    assert _sh(site2, f'{line}; echo "$CAMPAIGN"', _with(_base_env(), None, "alpha")).split("\n")[-1] == "paper_frame"


def test_submit_twins_site_resolves_the_same_table_and_states_it_on_every_job_env(tmp_path, monkeypatch):
    """RED before the fix on the `(None, 'alpha')` row: Coordinator.job_env read `cfg.get("SISTEER_REGISTRY_CAMPAIGN") or "sdfbase_full22"`."""
    st = _submit_twin()
    site = st.Site(str(write_site(tmp_path) / "config.env"), dry=True)
    assert st.REGISTRY_CAMPAIGN_DEFAULT == "sdfbase_full22"
    for (sisteer, job), want in TABLE:
        for k in CAMPAIGN_KEYS:
            monkeypatch.delenv(k, raising=False)
        if sisteer is not None:
            monkeypatch.setenv("SISTEER_REGISTRY_CAMPAIGN", sisteer)
        if job is not None:
            monkeypatch.setenv("REGISTRY_CAMPAIGN", job)
        assert site.registry_campaign() == want, (sisteer, job)
        co = st.Coordinator(site, "m")
        for label in ("cluster_a", "cluster_b"):
            assert co.job_env(label)["REGISTRY_CAMPAIGN"] == want, (label, sisteer, job)
    # the site's config.env names it: the file wins over BOTH environment spellings (repro_load_config sources the file over the
    # environment, so the file's value is what 04's own line reads)
    for k in CAMPAIGN_KEYS:
        monkeypatch.setenv(k, "from_env")
    site2 = st.Site(str(write_site(tmp_path / "two", extra_config="SISTEER_REGISTRY_CAMPAIGN=paper_frame\n") / "config.env"), dry=True)
    assert site2.registry_campaign() == "paper_frame" and st.Coordinator(site2, "m").job_env("cluster_a")["REGISTRY_CAMPAIGN"] == "paper_frame"


# ---- two config files in one site directory: the file GIVEN decides, on both faces (integration round 2) ---------------------------

def _second_config(site: Path, name: str = "experiment.env", *, campaign: str = "exp_frame", clusters: str = "cluster_b",
                   primary: str = "cluster_b") -> Path:
    """A SECOND site config beside config.env in the same site directory — an experiment's file an operator passes as `--config` — naming
    another campaign, its own cluster list and primary; every other key as config.env has it."""
    cfg = (site / "config.env").read_text()
    assert "SISTEER_REGISTRY_CAMPAIGN=site_frame" in cfg and 'SISTEER_CLUSTERS="cluster_a cluster_b"' in cfg and "SISTEER_PRIMARY_CLUSTER=cluster_a" in cfg
    other = (cfg.replace("SISTEER_REGISTRY_CAMPAIGN=site_frame", f"SISTEER_REGISTRY_CAMPAIGN={campaign}")
             .replace('SISTEER_CLUSTERS="cluster_a cluster_b"', f'SISTEER_CLUSTERS="{clusters}"')
             .replace("SISTEER_PRIMARY_CLUSTER=cluster_a", f"SISTEER_PRIMARY_CLUSTER={primary}"))
    p = site / name
    p.write_text(other)
    return p


def test_submit_twins_site_reads_the_config_file_it_is_given_not_the_directorys_config_env(tmp_path, monkeypatch):
    """Site.__init__ parsed `<site dir>/config.env` whatever file `--config` named, so with two config files in one site directory the
    funnel composed every job under config.env's campaign — and its cluster list and primary — while the stage, which sources the file it
    is given, read done-ness under the other's. The file given decides: its campaign, its clusters, its primary; config.env beside it is
    not read. RED before the fix on every assertion about experiment.env (campaign site_frame, both clusters, primary cluster_a)."""
    st = _submit_twin()
    for k in CAMPAIGN_KEYS:
        monkeypatch.delenv(k, raising=False)
    site = write_site(tmp_path, extra_config="SISTEER_REGISTRY_CAMPAIGN=site_frame\n")
    exp = _second_config(site)
    given = st.Site(str(exp), dry=True)
    assert given.config_path == str(exp) and given.site_dir == str(site)
    assert given.cfg["SISTEER_REGISTRY_CAMPAIGN"] == "exp_frame" and given.cfg == st.C.parse_env_file(str(exp))
    assert given.registry_campaign() == "exp_frame"
    assert list(given.clusters) == ["cluster_b"] and given.primary == "cluster_b"
    assert st.Coordinator(given, "m").job_env("cluster_b")["REGISTRY_CAMPAIGN"] == "exp_frame"
    # config.env beside it, given by name, still reads as itself
    default = st.Site(str(site / "config.env"), dry=True)
    assert default.registry_campaign() == "site_frame" and list(default.clusters) == ["cluster_a", "cluster_b"] and default.primary == "cluster_a"
    # the given file decides the cluster LIST, so a label it names without a block refuses by name (before: config.env's list, no refusal)
    _second_config(site, "broken.env", clusters="cluster_b cluster_zz")
    with pytest.raises(SystemExit, match="cluster_zz"):
        st.Site(str(site / "broken.env"), dry=True)
    # and a file that does not exist refuses by ITS name, not config.env's
    with pytest.raises(SystemExit, match=re.escape(str(site / "absent.env"))):
        st.Site(str(site / "absent.env"), dry=True)


def _sh_file(cfg: Path, body: str, env: dict) -> str:
    """As _sh, with the stage's --config pointing at `cfg` — any config file in the site directory."""
    e = dict(env, SISTEER_STATE_DIR=str(cfg.parent.parent / "state"))
    return _bash(f"source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={cfg}; repro_load_config; {body}", e)


def test_the_shell_library_loads_the_cluster_blocks_the_given_config_file_names(tmp_path):
    """The shell face of the same rule: repro_load_config sources the file it is given, and the cluster blocks it evals are for THAT file's
    SISTEER_CLUSTERS (clusters.py --config), not config.env's beside it — so the labels a stage iterates, the primary it submits to and
    the blocks it reads are one file's. RED before the fix: config.env's two blocks were loaded under experiment.env's one-cluster list
    (cluster_a resolvable from a file that never named it), and clusters.py had no --config."""
    site = write_site(tmp_path, extra_config="SISTEER_REGISTRY_CAMPAIGN=site_frame\n")
    exp = _second_config(site)
    out = _sh_file(exp, 'echo "$SISTEER_CLUSTERS|$SISTEER_PRIMARY_CLUSTER|$(cluster_var cluster_b CLUSTER_SSH)|$(cluster_var cluster_a CLUSTER_SSH)|'
                        '$(repro_registry_campaign)"; slurm_job_env cluster_b', _with(_base_env(), None, "alpha"))
    first, job_env = out.split("\n")[0], out.split("\n")[-1]
    assert first == "cluster_b|cluster_b|cluster-b-login||exp_frame", first
    assert re.search(r"(^|\s)REGISTRY_CAMPAIGN=exp_frame(\s|$)", job_env) and "site_frame" not in out, out
    # clusters.py itself: --config reads the file's labels; --site-dir alone still reads that directory's config.env; a missing file
    # refuses by its own name
    py = [sys.executable, str(LIB / "clusters.py")]
    assert subprocess.run(py + ["--config", str(exp), "--labels"], capture_output=True, text=True, check=True).stdout.split() == ["cluster_b"]
    assert subprocess.run(py + ["--site-dir", str(site), "--labels"], capture_output=True, text=True, check=True).stdout.split() == ["cluster_a", "cluster_b"]
    r = subprocess.run(py + ["--config", str(site / "absent.env"), "--labels"], capture_output=True, text=True)
    assert r.returncode == 2 and str(site / "absent.env") in r.stderr, (r.returncode, r.stderr)


def _dry(site: Path, stage: str, *args: str, env: dict, timeout: int = 600, config: "Path | None" = None) -> str:
    e = _base_env(); e.pop("SISTEER_SITE_DIR", None); e.pop("SISTEER_CONFIG", None); e.pop("SISTEER_FIT_ENV_NAME", None)
    e["SISTEER_STATE_DIR"] = str(site.parent / "state"); e.update(env)
    r = subprocess.run(["bash", str(REPRO / stage), "--dry-run", *args, "--config", str(config or site / "config.env")], capture_output=True,
                       text=True, env=e, cwd=str(REPO), timeout=timeout)
    assert r.returncode == 0, (stage, args, (r.stdout + r.stderr)[-2500:])
    return r.stdout + r.stderr


def test_04_dry_run_through_a_second_config_file_states_that_files_campaign_on_every_job(tmp_path):
    """The finding's scenario end to end, as the operator runs it: config.env names site_frame (both clusters, cluster_a primary); the
    operator passes `--config <site dir>/experiment.env`, naming exp_frame and cluster_b alone. Every `REGISTRY_CAMPAIGN=` the dry run
    prints carries exp_frame — the campaign 04 reads its done-ness under — and every sbatch goes to cluster_b. RED before the fix: the
    fit-family sbatch lines (submit_twin.job_env) said site_frame beside the derived build's ssh line (slurm_job_env) saying exp_frame —
    one run, two campaigns."""
    site = write_site(tmp_path, extra_config="SISTEER_REGISTRY_CAMPAIGN=site_frame\n")
    exp = _second_config(site)
    out = _dry(site, "04_fit_objects.sh", "--object", "all", "--model", MODEL, env={}, config=exp)
    stated = re.findall(r"(?:^|\s)REGISTRY_CAMPAIGN=(\S+)", out)
    assert stated and set(stated) == {"exp_frame"}, (sorted(set(stated)), out[-2500:])
    sbatch = [ln for ln in out.splitlines() if "sbatch " in ln]
    assert sbatch and all("REGISTRY_CAMPAIGN=exp_frame" in ln and "qos_cluster_b" in ln and "qos_cluster_a" not in ln for ln in sbatch), sbatch[:3]
    assert "REGISTRY_CAMPAIGN=site_frame" not in out


def test_04_dry_run_with_registry_campaign_alone_states_it_on_every_job_it_composes(tmp_path):
    """The finding's scenario, run as the operator runs it: REGISTRY_CAMPAIGN=paper_frame in the shell and nothing in config.env. Every
    `REGISTRY_CAMPAIGN=` the dry run prints — the fit-family sbatch lines through submit_twin.job_env and the derived build's ssh line
    through slurm_job_env — carries paper_frame, the campaign 04 reads its done-ness under; none carries sdfbase_full22. RED before the
    fix: every one of them said sdfbase_full22."""
    site = write_site(tmp_path)
    out = _dry(site, "04_fit_objects.sh", "--object", "all", "--model", MODEL, env={"REGISTRY_CAMPAIGN": "paper_frame"})
    stated = re.findall(r"(?:^|\s)REGISTRY_CAMPAIGN=(\S+)", out)
    assert stated and set(stated) == {"paper_frame"}, (sorted(set(stated)), out[-2500:])
    sbatch = [ln for ln in out.splitlines() if "sbatch " in ln]
    assert sbatch and all("REGISTRY_CAMPAIGN=paper_frame" in ln and "--export=" not in ln for ln in sbatch), sbatch[:3]
    # the site's config.env names the campaign: it wins over the shell's job-env spelling on every composed job, as on 04's own line
    site2 = write_site(tmp_path / "two", extra_config="SISTEER_REGISTRY_CAMPAIGN=site_frame\n")
    out2 = _dry(site2, "04_fit_objects.sh", "--object", "all", "--model", MODEL, env={"REGISTRY_CAMPAIGN": "alpha"})
    stated2 = re.findall(r"(?:^|\s)REGISTRY_CAMPAIGN=(\S+)", out2)
    assert stated2 and set(stated2) == {"site_frame"}, sorted(set(stated2))


def test_08_dry_run_with_registry_campaign_alone_grades_under_it(tmp_path):
    """08's daemon reads the pending queue under the campaign the jobs filed under: REGISTRY_CAMPAIGN alone names it (RED before the fix:
    `--campaign sdfbase_full22`); nothing named, the paper frame's; the site's config.env over the environment."""
    env = {"PYTHONPATH": str(REPO / "src")}
    site = write_site(tmp_path)
    for job, want in ((None, "sdfbase_full22"), ("paper_frame", "paper_frame")):
        r = run_stage("08_grade.sh", "--dry-run", "--once", "--model", MODEL, site=site,
                      env=dict(_base_env(), **env, **({"REGISTRY_CAMPAIGN": job} if job else {})), timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        out = r.stdout + r.stderr
        assert f"--campaign {want} " in out and "--campaign sdfbase_full22 " not in out.replace(f"--campaign {want} ", ""), (job, out[-1500:])
    site2 = write_site(tmp_path / "two", extra_config="SISTEER_REGISTRY_CAMPAIGN=site_frame\n")
    r = run_stage("08_grade.sh", "--dry-run", "--once", "--model", MODEL, site=site2, env=dict(_base_env(), **env, REGISTRY_CAMPAIGN="alpha"), timeout=600)
    assert r.returncode == 0 and "--campaign site_frame " in r.stdout + r.stderr, r.stdout[-2000:] + r.stderr[-2000:]


# ---- the job-env spelling in the GIVEN file: the funnel reads the file as the stage does ------------------------------------------------------

#: every way the given config file can spell the campaign — (SISTEER_REGISTRY_CAMPAIGN, REGISTRY_CAMPAIGN); None = the file does not name
#: the key, "" = it names it empty (`KEY=`, which `set -a; source` sets empty and `${KEY:-}` then reads as unset)
FILE_SPELLINGS = ((None, None), (None, "file_job"), ("file_site", None), ("file_site", "file_job"), ("", "file_job"), ("", None), (None, ""))
#: and every way the environment the funnel is invoked from can spell it
ENV_SPELLINGS = ((None, None), (None, "env_job"), ("env_site", None), ("env_site", "env_job"))


def _config_lines(sisteer: "str | None", job: "str | None") -> str:
    out = ""
    if sisteer is not None:
        out += f"SISTEER_REGISTRY_CAMPAIGN={sisteer}\n"
    if job is not None:
        out += f"REGISTRY_CAMPAIGN={job}\n"
    return out


def _stage_rule(file: tuple, env: tuple) -> str:
    """What the stage path resolves: the given file sourced OVER the environment for both keys (a key the file names, empty included, is the
    file's; a key it does not name is the environment's), then SISTEER_REGISTRY_CAMPAIGN, else REGISTRY_CAMPAIGN, else sdfbase_full22 —
    written out independently of either face, so the table below pins the rule and not one implementation against the other."""
    merged = {}
    for (sisteer, job) in (env, file):
        if sisteer is not None:
            merged["SISTEER_REGISTRY_CAMPAIGN"] = sisteer
        if job is not None:
            merged["REGISTRY_CAMPAIGN"] = job
    return merged.get("SISTEER_REGISTRY_CAMPAIGN") or merged.get("REGISTRY_CAMPAIGN") or "sdfbase_full22"


def test_submit_twins_site_reads_the_job_env_spelling_the_given_file_names_as_the_stage_does(tmp_path, monkeypatch):
    """Before the fix, Site.registry_campaign consulted the parsed file for SISTEER_REGISTRY_CAMPAIGN alone and the
    process environment for REGISTRY_CAMPAIGN, so a config file naming `REGISTRY_CAMPAIGN=alt_campaign` (the job-env spelling, which the
    stages honour: repro_load_config sources the file, and the resolver reads it) resolved to sdfbase_full22 in the funnel whenever no stage
    had sourced that file into the environment first — the direct `submit_twin.py --config <file> submit` interface. The funnel now reads
    the given file the way the stage does, for BOTH keys, and this node runs every file spelling against every environment spelling over
    both faces: the Python face, the shell face (repro_registry_campaign + slurm_job_env after repro_load_config, the stage path), and the
    rule written out on its own. RED before the fix on that row (file (None, 'alt_campaign'), nothing in the environment):
    `'sdfbase_full22' == 'alt_campaign'`."""
    st = _submit_twin()
    for k in CAMPAIGN_KEYS:
        monkeypatch.delenv(k, raising=False)
    # the finding's row, as the operator wrote the site: the job-env spelling alone in the named file, neither key inherited
    site = write_site(tmp_path, extra_config="REGISTRY_CAMPAIGN=alt_campaign\n")
    given = st.Site(str(site / "config.env"), dry=True)
    assert given.cfg.get("REGISTRY_CAMPAIGN") == "alt_campaign" and "SISTEER_REGISTRY_CAMPAIGN" not in given.cfg
    shell = _sh(site, "repro_registry_campaign; echo; slurm_job_env cluster_a", _base_env()).split("\n")
    assert shell[0] == "alt_campaign" and re.search(r"(^|\s)REGISTRY_CAMPAIGN=alt_campaign(\s|$)", shell[-1]), shell
    assert given.registry_campaign() == "alt_campaign"
    co = st.Coordinator(given, "m")
    assert {co.job_env(label)["REGISTRY_CAMPAIGN"] for label in ("cluster_a", "cluster_b")} == {"alt_campaign"}
    # every spelling of the file against every spelling of the environment: the two faces agree with each other and with the rule
    for i, file in enumerate(FILE_SPELLINGS):
        site_i = write_site(tmp_path / f"f{i}", extra_config=_config_lines(*file))
        for env in ENV_SPELLINGS:
            want = _stage_rule(file, env)
            e = _with(_base_env(), *env)
            shell_face = _sh(site_i, "repro_registry_campaign", e)
            assert shell_face == want, (file, env, shell_face, want)
            for k in CAMPAIGN_KEYS:
                monkeypatch.delenv(k, raising=False)
            for k, v in zip(CAMPAIGN_KEYS, env):
                if v is not None:
                    monkeypatch.setenv(k, v)
            python_face = st.Site(str(site_i / "config.env"), dry=True).registry_campaign()
            assert python_face == shell_face == want, (file, env, python_face, shell_face, want)
            assert st.Coordinator(st.Site(str(site_i / "config.env"), dry=True), "m").job_env("cluster_b")["REGISTRY_CAMPAIGN"] == want, (file, env)


def test_a_direct_submit_through_the_funnels_cli_states_the_given_files_campaign_on_the_line_it_composes(tmp_path):
    """The finding's interface end to end: an operator runs `submit_twin.py --config <file> --model m submit …` directly — no stage has
    sourced the file, neither campaign key is in the shell — with `REGISTRY_CAMPAIGN=alt_campaign` alone in that file. The sbatch line the
    dry run composes states alt_campaign (the campaign the stage path resolves over the same file: slurm_job_env, read beside it) and
    never sdfbase_full22, so a job submitted this way files where 04 and the pins writer read. RED before the fix: the composed line
    carried `REGISTRY_CAMPAIGN=sdfbase_full22`."""
    from front_door_testlib import make_registry, write_fake_slurm
    sha = "0123456789abcdef0123456789abcdef01234567"
    fake = {lab: tmp_path / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
    for d in fake.values():
        d.mkdir()
    bin_dir = write_fake_slurm(tmp_path / "bin")
    site = write_site(tmp_path, ssh={lab: f"local:{d}" for lab, d in fake.items()}, extra_config="REGISTRY_CAMPAIGN=alt_campaign\n")
    for lab in fake:                                   # the mirrored checkout at the wave sha, as test_submit_twin's World lays it out
        co = tmp_path / f"shared_{lab}" / "sisteer-eval-repo" / sha / "scripts/sdfbase/full22"
        co.mkdir(parents=True)
        (co / "target_attacks.sbatch").write_text("#!/bin/bash\nf22_attempt_wait   # sisteer-attempt-wait-hook\nf22_init\n")
    make_registry(tmp_path / "registry")
    env = dict(_base_env(), PATH=f"{bin_dir}:{os.environ['PATH']}", SISTEER_STATE_DIR=str(tmp_path / "state"), SISTEER_NO_WATCHER="1",
               SISTEER_REGISTRY_ROOT=str(tmp_path / "registry"))
    env.pop("SISTEER_TWIN_CRASH_AT", None)
    r = subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(site / "config.env"), "--model", "m", "--dry-run", "submit",
                        "--unit", "u", "--lwid", "m/u0/static/pasttense", "--script", "scripts/sdfbase/full22/target_attacks.sbatch", "--name", "n",
                        "--args", f"m u0 crescendo {sha} 9 0,1,2", "--twin", "--wave-sha", sha], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [ln for ln in r.stdout.splitlines() if "sbatch " in ln]
    assert lines and all(ln.startswith("DRY-RUN: ssh local:") for ln in lines), r.stdout
    stated = {m for ln in lines for m in re.findall(r"(?:^|[\s'\"])REGISTRY_CAMPAIGN=([^\s'\"]+)", ln)}
    assert stated == {"alt_campaign"}, (stated, lines[0][:300])
    assert "sdfbase_full22" not in r.stdout, r.stdout
    # the stage path over the same file says the same word on its own composed job env
    shell = _sh(site, "slurm_job_env cluster_a", _base_env())
    assert re.search(r"(^|\s)REGISTRY_CAMPAIGN=alt_campaign(\s|$)", shell), shell


# ---- the sweep: no face of the front door reads the site's key alone ---------------------------------------------------------------------

def test_no_face_of_the_front_door_reads_the_sites_key_alone():
    """The two-level spellings the finding named are gone from reproduce/ (every stage, the shell library, the Python funnel), and each
    job-side composer calls the one resolver by name."""
    two_level = re.compile(r"\$\{SISTEER_REGISTRY_CAMPAIGN:-sdfbase_full22\}|SISTEER_REGISTRY_CAMPAIGN\"\)\s*or\s*\"sdfbase_full22\"")
    offenders = []
    for p in sorted(list(REPRO.rglob("*.sh")) + list(REPRO.rglob("*.py"))):
        for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if two_level.search(ln):
                offenders.append(f"{p.relative_to(REPO)}:{i}: {ln.strip()[:160]}")
    assert offenders == [], offenders
    slurm = (REPRO / "lib" / "slurm.sh").read_text()
    base = slurm[slurm.index("slurm_base_env(){"):slurm.index("slurm_job_env(){")]
    job = slurm[slurm.index("slurm_job_env(){"):]
    job = job[:job.index("\n}\n") + 3]
    assert '"$(repro_registry_campaign)"' in base and 'F22_REGISTRY_CAMPAIGN=%s' in base
    # the job env is the funnel's own (slurm_job_env prints `submit_twin.py job-env`), so its campaign is Site.registry_campaign below
    assert "job-env" in job and "REGISTRY_CAMPAIGN" not in job and "repro_registry_campaign" not in job
    assert '--campaign "$(repro_registry_campaign)"' in (REPRO / "08_grade.sh").read_text()
    twin = (LIB / "submit_twin.py").read_text()
    assert '"REGISTRY_CAMPAIGN": self.site.registry_campaign()}' in twin
    # the funnel reads the GIVEN file over the environment for BOTH spellings before the precedence (previously the file was
    # consulted for the site's key alone, so REGISTRY_CAMPAIGN in the named file was read only once a stage had sourced it
    # into the environment) — through Site.given, the one read every site scalar of the module makes
    # (tests/test_site_config_over_environment.py)
    assert '        return self.cfg[key] if key in self.cfg else os.environ.get(key)\n' in twin
    assert '        return self.given("SISTEER_REGISTRY_CAMPAIGN") or self.given("REGISTRY_CAMPAIGN") or REGISTRY_CAMPAIGN_DEFAULT' in twin
    assert 'self.cfg.get("SISTEER_REGISTRY_CAMPAIGN") or os.environ.get(' not in twin
    assert 'REGISTRY_CAMPAIGN_DEFAULT = "sdfbase_full22"' in twin
    # and the site's cfg IS the file the run was given, on both faces (integration round 2): never `<site dir>/config.env` beside it
    assert "self.cfg = C.site_config_file(self.config_path)" in twin and "self.clusters = C.load_all(self.site_dir, config=self.cfg)" in twin
    assert "C.site_config(" not in twin
    assert 'clusters.py" --config "$SISTEER_CONFIG" --sh' in (REPRO / "lib" / "common.sh").read_text()
