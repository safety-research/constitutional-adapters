"""src/sisteer/site.py: the site config and the placeholder grammar committed configs use for site paths.

Every placeholder of the closed set expands from the environment, the site config file or the cluster block, in that
order; ``$NAME`` and ``${NAME}`` are one placeholder; an unset, unknown, bare or unterminated one refuses by name (the
message names the placeholder and the variable to set); expansion is pure text substitution (``~`` is left to
``expand_path``), so an expanded value is exactly the literal a placeholder replaced. The front door's shell twin
(``repro_expand_shared``) and ``reproduce/lib/clusters.py`` agree with it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sisteer import site as S

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "reproduce" / "site.example"

#: every placeholder's own variables, cleared so the host's environment never answers for a test
_VARS = sorted({v for envs, keys, _ in S.PLACEHOLDERS.values() for v in (*envs, *keys)}
               | {"SISTEER_CONFIG", "SISTEER_SITE_DIR", "SISTEER_CLUSTER_LABEL"})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site-here"))
    S._SITE_CACHE.clear()
    yield
    S._SITE_CACHE.clear()


def _site(tmp_path, config: str, clusters: dict | None = None) -> Path:
    d = tmp_path / "site"
    (d / "clusters").mkdir(parents=True, exist_ok=True)
    (d / "config.env").write_text(textwrap.dedent(config))
    for label, body in (clusters or {}).items():
        (d / "clusters" / f"{label}.env").write_text(textwrap.dedent(body))
    return d / "config.env"


TWO = """\
SISTEER_CLUSTERS="ca cb"
SISTEER_PRIMARY_CLUSTER=ca
SISTEER_STATE_ROOT='~/.local/state'
SISTEER_RUN_STATE_ROOT='~/runs'
SISTEER_STAGE_DIR=/srv/stage
SISTEER_WORK_DIR='~/work'
SISTEER_GRADE_DIR='~/grade'
SISTEER_WANDB_PROJECT=site-project
"""
BLOCK_A = """\
CLUSTER_SHARED_ROOT=/scratch/a/shared
CLUSTER_HF_HOME='$SHARED/hf_cache'
"""
BLOCK_B = """\
CLUSTER_SHARED_ROOT=/data/b/shared
CLUSTER_HF_HOME=/data/b/models
"""


# --- each placeholder, each source -----------------------------------------------------------------------------------

def test_every_placeholder_expands_from_the_site_config_and_its_cluster_blocks(tmp_path):
    cfg = _site(tmp_path, TWO, {"ca": BLOCK_A, "cb": BLOCK_B})
    site = S.load_site(cfg, env={})
    want = {"SHARED": "/scratch/a/shared", "HF_HOME": "/scratch/a/shared/hf_cache", "STATE": "~/.local/state", "RUNSTATE": "~/runs",
            "STAGE": "/srv/stage", "WORK": "~/work", "GRADE": "~/grade", "WANDB_PROJECT": "site-project"}
    assert set(want) == set(S.NAMES)
    for name, value in want.items():
        assert S.expand(f"${name}/x", site=site, env={}) == f"{value}/x"
        assert S.expand("${" + name + "}/x", site=site, env={}) == f"{value}/x"
    # a named cluster reads that cluster's block
    assert S.expand("$SHARED/x", "cb", site=site, env={}) == "/data/b/shared/x"
    assert S.expand("${HF_HOME}/hub", "cb", site=site, env={}) == "/data/b/models/hub"


def test_the_environment_wins_over_the_site_file(tmp_path):
    cfg = _site(tmp_path, TWO, {"ca": BLOCK_A, "cb": BLOCK_B})
    site = S.load_site(cfg, env={})
    env = {"SISTEER_SHARED_ROOT": "/job/shared", "HF_HOME": "/job/hf", "SISTEER_STATE_ROOT": "/job/state", "SISTEER_RUN_STATE_ROOT": "/job/runs",
           "SISTEER_STAGE_DIR": "/job/stage", "SISTEER_WORK_DIR": "/job/work", "SISTEER_GRADE_DIR": "/job/grade",
           "WANDB_PROJECT": "job-project"}
    for name, var in (("SHARED", "SISTEER_SHARED_ROOT"), ("HF_HOME", "HF_HOME"), ("STATE", "SISTEER_STATE_ROOT"),
                      ("RUNSTATE", "SISTEER_RUN_STATE_ROOT"), ("STAGE", "SISTEER_STAGE_DIR"), ("WORK", "SISTEER_WORK_DIR"), ("GRADE", "SISTEER_GRADE_DIR"),
                      ("WANDB_PROJECT", "WANDB_PROJECT")):
        assert S.expand(f"${name}", site=site, env=env) == env[var]
    # a cluster job's CLUSTER_SHARED_ROOT is the second environment source for $SHARED; SISTEER_WANDB_PROJECT the second for the project
    assert S.expand("$SHARED", site=S.Site(), env={"CLUSTER_SHARED_ROOT": "/c/s"}) == "/c/s"
    assert S.expand("$WANDB_PROJECT", site=S.Site(), env={"SISTEER_WANDB_PROJECT": "p2"}) == "p2"


def test_a_named_cluster_reads_its_own_block_even_when_the_job_environment_carries_another_root(tmp_path):
    cfg = _site(tmp_path, TWO, {"ca": BLOCK_A, "cb": BLOCK_B})
    site = S.load_site(cfg, env={})
    assert S.expand("$SHARED/x", "cb", site=site, env={"SISTEER_SHARED_ROOT": "/job/shared"}) == "/data/b/shared/x"


def test_the_default_cluster_is_the_job_label_then_the_primary(tmp_path):
    cfg = _site(tmp_path, TWO, {"ca": BLOCK_A, "cb": BLOCK_B})
    site = S.load_site(cfg, env={})
    assert S.expand("$SHARED", site=site, env={}) == "/scratch/a/shared"
    assert S.expand("$SHARED", site=site, env={"SISTEER_CLUSTER_LABEL": "cb"}) == "/data/b/shared"


def test_a_cluster_the_site_does_not_name_refuses_by_name(tmp_path):
    cfg = _site(tmp_path, TWO, {"ca": BLOCK_A, "cb": BLOCK_B})
    site = S.load_site(cfg, env={})
    with pytest.raises(S.PlaceholderError, match=r"cluster 'cz', which the site config .* does not name"):
        S.expand("$SHARED/x", "cz", site=site, env={})
    # no site config (a cluster node): a job on another cluster cannot answer for the one asked about
    with pytest.raises(S.PlaceholderError, match=r"this job runs on 'ca'"):
        S.expand("$SHARED/x", "cb", site=S.Site(), env={"SISTEER_CLUSTER_LABEL": "ca", "SISTEER_SHARED_ROOT": "/s"})
    assert S.expand("$SHARED/x", "ca", site=S.Site(), env={"SISTEER_CLUSTER_LABEL": "ca", "SISTEER_SHARED_ROOT": "/s"}) == "/s/x"


# --- refusals ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,var", [("SHARED", "SISTEER_SHARED_ROOT"), ("HF_HOME", "HF_HOME"), ("STATE", "SISTEER_STATE_ROOT"),
                                      ("RUNSTATE", "SISTEER_RUN_STATE_ROOT"),
                                      ("STAGE", "SISTEER_STAGE_DIR"), ("WORK", "SISTEER_WORK_DIR"),
                                      ("GRADE", "SISTEER_GRADE_DIR"), ("WANDB_PROJECT", "WANDB_PROJECT")])
@pytest.mark.parametrize("braced", [False, True])
def test_an_unset_placeholder_refuses_naming_it_and_the_variable_to_set(name, var, braced):
    spelled = "${" + name + "}" if braced else "$" + name
    with pytest.raises(S.PlaceholderError) as ei:
        S.expand(f"{spelled}/x", site=S.Site(), env={}, key="some.key")
    msg = str(ei.value)
    assert f"${name}" in msg and var in msg and "some.key" in msg and "not set on this host" in msg


@pytest.mark.parametrize("value,match", [
    ("$SHARED_ROOT/x", r"\$SHARED_ROOT, which is not a site placeholder"),
    ("${FOO}/x", r"\$\{FOO\}, which is not a site placeholder"),
    ("${}/x", r"\$\{\}, which is not a site placeholder"),
    ("${SHARED /x}", r"which is not a site placeholder"),
    ("${SHARED/x", r"bare or unterminated '\$'"),
    ("/a/$/b", r"bare or unterminated '\$'"),
    ("cost: 5$", r"bare or unterminated '\$'"),
    ("$1/x", r"bare or unterminated '\$'"),
])
def test_anything_that_is_not_a_closed_set_placeholder_refuses(value, match):
    with pytest.raises(S.PlaceholderError, match=match):
        S.expand(value, site=S.Site(), env={"SISTEER_SHARED_ROOT": "/s"})


def test_a_site_value_that_resolves_through_itself_refuses(tmp_path):
    cfg = _site(tmp_path, "SISTEER_STATE_ROOT='$STATE/x'\nSISTEER_WORK_DIR='$GRADE/w'\nSISTEER_GRADE_DIR='$WORK/g'\n")
    site = S.load_site(cfg, env={})
    with pytest.raises(S.PlaceholderError, match=r"resolves through itself \(\$STATE -> \$STATE\)"):
        S.expand("$STATE", site=site, env={})
    with pytest.raises(S.PlaceholderError, match=r"resolves through itself"):
        S.expand("$WORK", site=site, env={})


def test_a_site_value_may_hold_another_placeholder(tmp_path):
    cfg = _site(tmp_path, "SISTEER_CLUSTERS=ca\nSISTEER_STATE_ROOT='$WORK/state'\nSISTEER_WORK_DIR=/w\n", {"ca": BLOCK_A})
    site = S.load_site(cfg, env={})
    assert S.expand("$STATE/x", site=site, env={}) == "/w/state/x"


# --- values that are not placeholders ----------------------------------------------------------------------------------

def test_none_passes_through_a_plain_value_is_untouched_and_a_non_string_refuses():
    assert S.expand(None) is None and S.expand_path(None) is None
    assert S.expand("/plain/path", site=S.Site(), env={}) == "/plain/path"
    assert S.expand(Path("/p/q"), site=S.Site(), env={}) == "/p/q"
    with pytest.raises(TypeError, match="must be a string"):
        S.expand(5, key="k")


def test_expand_substitutes_text_only_and_expand_path_expands_the_home_directory(tmp_path, monkeypatch):
    """The expanded value equals the literal the placeholder replaced (a config spelled ~/.local/state/x before the
    release reads ~/.local/state/x after it); only expand_path, the form readers open, expands ~."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    env = {"SISTEER_STATE_ROOT": "~/.local/state"}
    assert S.expand("$STATE/pool/x", site=S.Site(), env=env) == "~/.local/state/pool/x"
    assert S.expand_path("$STATE/pool/x", site=S.Site(), env=env) == tmp_path / "home" / ".local" / "state" / "pool" / "x"
    assert S.expand_path("~/plain", site=S.Site(), env={}) == tmp_path / "home" / "plain"
    out = S.expand_keys({"a": "$STATE/a", "b": "$STATE/b", "c": None}, ("a", "b", "c", "absent"), site=S.Site(), env=env)
    assert out == {"a": "~/.local/state/a", "b": "~/.local/state/b", "c": None}
    out = S.expand_keys({"a": "$STATE/a", "b": 1}, ("a",), site=S.Site(), env=env, paths=True)
    assert out == {"a": str(tmp_path / "home" / ".local" / "state" / "a"), "b": 1}


def test_require_expanded_refuses_a_path_that_still_holds_a_placeholder():
    assert S.require_expanded("/a/b") == "/a/b" and S.require_expanded(None) is None
    for bad in ("$SHARED/x", "/a/${STATE}/b", "/a/$/b"):
        with pytest.raises(S.PlaceholderError, match="never use it as a literal path"):
            S.require_expanded(bad, key="out_dir")


def test_placeholders_in_lists_every_token_as_written():
    assert S.placeholders_in("$SHARED/${STATE}/$/${bad x}") == ["SHARED", "STATE", "", "bad x"]
    assert S.has_placeholder("a$b") and not S.has_placeholder("ab") and not S.has_placeholder(None)


# --- the site config files ---------------------------------------------------------------------------------------------

def test_a_named_site_file_that_is_missing_refuses_and_an_absent_default_is_an_empty_site(tmp_path):
    with pytest.raises(S.SiteConfigError, match="no site config at"):
        S.load_site(tmp_path / "nope.env", env={})
    with pytest.raises(S.SiteConfigError, match="no site config at"):
        S.load_site(env={"SISTEER_CONFIG": str(tmp_path / "nope.env")})
    site = S.load_site(env={"SISTEER_SITE_DIR": str(tmp_path / "empty")})
    assert site.path is None and not site.config and not site.clusters


def test_a_cluster_named_without_a_block_refuses(tmp_path):
    cfg = _site(tmp_path, 'SISTEER_CLUSTERS="ca cb"\n', {"ca": BLOCK_A})
    with pytest.raises(S.SiteConfigError, match=r"cluster 'cb' is named in SISTEER_CLUSTERS .* does not exist"):
        S.load_site(cfg, env={})


def test_the_site_file_is_reread_after_an_edit(tmp_path):
    cfg = _site(tmp_path, "SISTEER_GRADE_DIR=/g1\n")
    assert S.expand("$GRADE", site=S.load_site(cfg, env={}), env={}) == "/g1"
    cfg.write_text("SISTEER_GRADE_DIR=/g2-longer\n")
    assert S.expand("$GRADE", site=S.load_site(cfg, env={}), env={}) == "/g2-longer"


def test_site_dir_and_config_path_follow_the_front_doors_resolution(tmp_path):
    assert S.config_path({"SISTEER_CONFIG": str(tmp_path / "x" / "other.env")}) == str(tmp_path / "x" / "other.env")
    assert S.site_dir({"SISTEER_CONFIG": str(tmp_path / "x" / "other.env")}) == str(tmp_path / "x")
    assert S.config_path({"SISTEER_SITE_DIR": str(tmp_path / "d")}) == str(tmp_path / "d" / "config.env")
    assert S.site_dir({}) == os.path.expanduser("~/.config/sisteer")


def test_the_command_line_expands_and_refuses_by_name(tmp_path, capsys):
    cfg = _site(tmp_path, TWO, {"ca": BLOCK_A, "cb": BLOCK_B})
    assert S.main(["expand", "--config", str(cfg), "--cluster", "cb", "$SHARED/x", "${GRADE}/y"]) == 0
    assert capsys.readouterr().out.splitlines() == ["/data/b/shared/x", "~/grade/y"]
    assert S.main(["expand", "--config", str(cfg), "$NOPE/x"]) == 2
    assert "REFUSED:" in capsys.readouterr().err
    assert S.main(["show", "--config", str(cfg)]) == 0
    shown = capsys.readouterr().out
    assert "$SHARED\t/scratch/a/shared" in shown and "$WANDB_PROJECT\tsite-project" in shown


# --- the shipped example site ------------------------------------------------------------------------------------------

def test_the_shipped_example_resolves_every_placeholder_but_the_project_it_leaves_empty(tmp_path, monkeypatch):
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    site = S.load_site(d / "config.env", env={})
    assert site.labels == ["cluster_a", "cluster_b"] and site.primary == "cluster_a"
    assert S.expand("$SHARED/exp", site=site, env={}) == "/scratch/example/shared/exp"
    assert S.expand("$SHARED/exp", "cluster_b", site=site, env={}) == "/data/example/shared/exp"
    assert S.expand("$HF_HOME/hub", site=site, env={}) == "/scratch/example/shared/hf_cache/hub"
    for name in ("STATE", "RUNSTATE", "STAGE", "WORK", "GRADE"):
        got = S.expand(f"${name}", site=site, env={})
        assert got.startswith("~/") and "$" not in got
    with pytest.raises(S.PlaceholderError, match="WANDB_PROJECT"):
        S.expand("$WANDB_PROJECT", site=site, env={})


# --- the shell twin and the cluster-block reader -----------------------------------------------------------------------

def _shell_expand(label: str, value: str, root: str) -> subprocess.CompletedProcess:
    lib = REPO / "reproduce" / "lib" / "common.sh"
    funcs = subprocess.run(["bash", "-c", f"source {lib} >/dev/null 2>&1; declare -f cluster_var repro_refuse_now repro_dollar_token repro_expand_shared; "
                                          "echo REPRO_SITE_PLACEHOLDERS=\\\"$REPRO_SITE_PLACEHOLDERS\\\""],
                           capture_output=True, text=True)
    script = f"{funcs.stdout}\nexport CLUSTER__{label}__CLUSTER_SHARED_ROOT='{root}'\nrepro_expand_shared {label} \"$1\""
    return subprocess.run(["bash", "-c", script, "x", value], capture_output=True, text=True)


@pytest.mark.parametrize("value", ["$SHARED/exp/x", "${SHARED}/exp/x", "/abs/no/placeholder", "$SHARED", "a/$SHARED/b"])
def test_the_shell_twin_agrees_with_the_python_expansion_for_shared(value):
    r = _shell_expand("ca", value, "/scratch/a/shared")
    assert r.returncode == 0, r.stderr
    py = S.substitute_shared(value, "/scratch/a/shared")
    assert r.stdout == py


@pytest.mark.parametrize("value", ["$STATE/x", "$RUNSTATE/x", "${GRADE}/x", "$HF_HOME/hub", "${WANDB_PROJECT}"])
def test_the_shell_twin_refuses_every_other_site_placeholder(value):
    r = _shell_expand("ca", value, "/scratch/a/shared")
    assert r.returncode != 0 and "REFUSED" in r.stderr and "which a cluster path cannot expand" in r.stderr and r.stdout == "", r.stderr


@pytest.mark.parametrize("value, named", [("$SHARE/exp/x", "$SHARE"), ("${SHARE}/exp/x", "$SHARE"), ("a $HOME stays", "$HOME"),
                                          ("$SHARED_ROOT/x", "$SHARED_ROOT"), ("$USER/x", "$USER"), ("logs$", "bare or unterminated"),
                                          ("${}/x", "bare or unterminated"), ("$1/x", "bare or unterminated")])
def test_the_shell_twin_refuses_any_other_dollar_by_name(value, named):
    """A misspelt placeholder, a name outside the closed set and a bare `$` never pass through to a command."""
    r = _shell_expand("ca", value, "/scratch/a/shared")
    reason = f"holds {named}, which is not a site placeholder" if named.startswith("$") else f"holds a {named} '$'"
    assert r.returncode != 0 and r.stdout == "" and "REFUSED" in r.stderr and reason in r.stderr, r.stderr
    if named.startswith("$"):
        with pytest.raises(S.PlaceholderError, match="not a site placeholder"):
            S.expand(value, site=S.Site(), env={"SISTEER_SHARED_ROOT": "/scratch/a/shared"})


def test_the_shell_twin_refuses_shared_on_a_cluster_with_no_root():
    r = _shell_expand("ca", "$SHARED/x", "")
    assert r.returncode != 0 and "sets no CLUSTER_SHARED_ROOT" in r.stderr


def _clusters_module():
    sys.path.insert(0, str(REPO / "reproduce" / "lib"))
    try:
        import clusters  # noqa: E402
    finally:
        sys.path.pop(0)
    return clusters


def test_the_cluster_reader_parses_with_this_modules_parser_and_defaults_the_mirror_selector(tmp_path):
    C = _clusters_module()
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    blocks = C.load_all(str(d))
    a = blocks["cluster_a"]
    assert a["CLUSTER_MIRROR_SELECTOR"] == "cluster_a" and a["CLUSTER_LABEL"] == "cluster_a"
    assert a["CLUSTER_REPO_MIRROR"] == "/scratch/example/shared/repos/constitutional-adapters"
    assert C.parse_env_file(str(d / "clusters" / "cluster_a.env")) == S.parse_env_file(d / "clusters" / "cluster_a.env")
    # an explicit selector is kept; ${SHARED} substitutes like $SHARED
    p = d / "clusters" / "cluster_b.env"
    p.write_text(p.read_text().replace("CLUSTER_MIRROR_SELECTOR=''", "CLUSTER_MIRROR_SELECTOR=tool-b")
                 .replace("CLUSTER_LOG_DIR='$SHARED/exp/logs'", "CLUSTER_LOG_DIR='${SHARED}/exp/logs'"))
    b = C.load_all(str(d))["cluster_b"]
    assert b["CLUSTER_MIRROR_SELECTOR"] == "tool-b" and b["CLUSTER_LOG_DIR"] == "/data/example/shared/exp/logs"


def test_a_cluster_block_holding_another_site_placeholder_refuses_by_name(tmp_path):
    C = _clusters_module()
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    p = d / "clusters" / "cluster_a.env"
    p.write_text(p.read_text().replace("CLUSTER_LOG_DIR='$SHARED/exp/logs'", "CLUSTER_LOG_DIR='$STATE/logs'"))
    with pytest.raises(SystemExit, match=r"CLUSTER_LOG_DIR=.*holds \$STATE"):
        C.load_all(str(d))


@pytest.mark.parametrize("value, named", [("'$SHARE/exp/logs'", r"\$SHARE"), ("'${SHARE}/exp/logs'", r"\$SHARE"),
                                          ("'$HOME/logs'", r"\$HOME"), ("'$USER/logs'", r"\$USER"),
                                          ("'logs$'", "a bare or unterminated")])
def test_a_cluster_block_holding_any_other_dollar_refuses_by_name_at_load(tmp_path, value, named):
    """Beyond $SHARED a cluster block expands nothing, so a misspelt placeholder, an environment variable and a bare `$` refuse
    when the block is read, naming the key and the token, instead of reaching a command as written."""
    C = _clusters_module()
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    p = d / "clusters" / "cluster_a.env"
    p.write_text(p.read_text().replace("CLUSTER_LOG_DIR='$SHARED/exp/logs'", f"CLUSTER_LOG_DIR={value}"))
    with pytest.raises(SystemExit, match=rf"cluster_a.env: CLUSTER_LOG_DIR=.*holds {named}"):
        C.load_all(str(d))
    r = subprocess.run([sys.executable, str(REPO / "reproduce" / "lib" / "clusters.py"), "--config", str(d / "config.env"), "--sh"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout == "" and "REFUSED" in r.stderr and "CLUSTER_LOG_DIR" in r.stderr


def test_a_shared_root_holding_a_dollar_refuses_by_name(tmp_path):
    C = _clusters_module()
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    p = d / "clusters" / "cluster_b.env"
    p.write_text(p.read_text().replace("CLUSTER_SHARED_ROOT=/data/example/shared", "CLUSTER_SHARED_ROOT='/data/$USER/shared'"))
    with pytest.raises(SystemExit, match=r"CLUSTER_SHARED_ROOT='/data/\$USER/shared' holds \$USER.*literal absolute path"):
        C.load_all(str(d))


def test_a_site_config_value_holding_a_misspelt_placeholder_refuses_by_key_at_load(tmp_path):
    """A config file value may hold the site placeholders and no other `$`: a misspelt placeholder, an environment variable ($HOME,
    ${USER}: the front door's shell would expand it as it sources the file while the Python readers keep it as written) and a bare
    `$` refuse by key when lib/clusters.py reads the file, on every road into it."""
    C = _clusters_module()
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    cfg = d / "config.env"
    clean = cfg.read_text()
    assert "SISTEER_DATA_ROOT='$SHARED/exp/corpora'" in clean
    for bad, named in (("'$SHARE/exp/corpora'", r"\$SHARE"), ('"${SHARE}/exp/corpora"', r"\$SHARE"), ("'corpora$'", "a bare or unterminated"),
                       ('"$HOME/corpora"', r"\$HOME"), ('"/data/${USER}/corpora"', r"\$USER"), ("$HOME/corpora", r"\$HOME")):
        cfg.write_text(clean.replace("SISTEER_DATA_ROOT='$SHARED/exp/corpora'", f"SISTEER_DATA_ROOT={bad}"))
        for read in (lambda: C.site_config_file(str(cfg)), lambda: C.site_config(str(d)), lambda: C.load_all(str(d))):
            with pytest.raises(SystemExit, match=rf"config.env: SISTEER_DATA_ROOT=.*holds {named}"):
                read()
        r = subprocess.run([sys.executable, str(REPO / "reproduce" / "lib" / "clusters.py"), "--config", str(cfg), "--sh"],
                           capture_output=True, text=True)
        assert r.returncode == 2 and r.stdout == "" and "SISTEER_DATA_ROOT" in r.stderr, r.stderr


def _shell_load(cfg: Path, then: str = "echo \"DATA=$SISTEER_DATA_ROOT LOG=$CLUSTER__cluster_a__CLUSTER_LOG_DIR\"") -> subprocess.CompletedProcess:
    """common.sh repro_load_config over `cfg` (the file it sources, then the cluster blocks through lib/clusters.py), then `then`."""
    lib = REPO / "reproduce" / "lib" / "common.sh"
    script = f"source {lib}; REPRO_CONFIG=\"$1\"; repro_load_config; {then}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SISTEER_", "CLUSTER_", "REPRO_"))}
    env["SISTEER_PYTHON"] = sys.executable
    return subprocess.run(["bash", "-c", script, "x", str(cfg)], capture_output=True, text=True, env=env)


def test_the_shell_loader_refuses_a_value_holding_any_dollar_but_a_site_placeholder_by_key(tmp_path):
    """After common.sh sources the file, bash has expanded every variable written outside single quotes; a `$` left in a value must
    be a site placeholder, so a misspelt one, a single-quoted environment variable and a bare `$` refuse at load, by key. An
    environment variable bash did expand refuses too, by key, from lib/clusters.py's read of the file as written: the Python readers
    would keep it as written, so the two faces would read two different values."""
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    cfg = d / "config.env"
    clean = cfg.read_text()
    r = _shell_load(cfg)
    assert r.returncode == 0, r.stderr
    assert "DATA=$SHARED/exp/corpora LOG=/scratch/example/shared/exp/logs" in r.stdout
    cfg.write_text(clean + 'SISTEER_STATE_DIR="$HOME/sisteer-state"\n')                  # expanded by bash as the file is sourced
    r = _shell_load(cfg)
    assert r.returncode == 2 and "SISTEER_STATE_DIR='$HOME/sisteer-state' holds $HOME" in r.stderr and "DATA=" not in r.stdout, r.stderr
    for bad, named in (("'$SHARE/exp/corpora'", "$SHARE"), ("'$HOME/corpora'", "$HOME"), ("'corpora$'", "bare or unterminated")):
        cfg.write_text(clean.replace("SISTEER_DATA_ROOT='$SHARED/exp/corpora'", f"SISTEER_DATA_ROOT={bad}"))
        r = _shell_load(cfg)
        assert r.returncode == 2 and "REFUSED: SISTEER_DATA_ROOT=" in r.stderr and named in r.stderr and "DATA=" not in r.stdout, r.stderr


def _submit_twin_module():
    sys.path.insert(0, str(REPO / "reproduce" / "lib"))
    try:
        import submit_twin  # noqa: E402
    finally:
        sys.path.pop(0)
    return submit_twin


def test_direct_submit_and_the_shell_loader_read_one_site_config_alike(tmp_path):
    """One config file, two roads into it: `submit_twin.py --config` reads the file as written, the front door's shell sources it.
    A root written with an environment variable (`"$HOME/registry"`) would name a literal `$HOME/registry` on the first road and the
    home directory's registry on the second, so both refuse it by key at load; a value written out reads the same on both."""
    T = _submit_twin_module()
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    cfg = d / "config.env"
    clean = cfg.read_text()
    assert re.search(r"^SISTEER_REGISTRY_ROOT=", clean, re.M)
    cfg.write_text(re.sub(r"^SISTEER_REGISTRY_ROOT=.*$", 'SISTEER_REGISTRY_ROOT="$HOME/registry"', clean, count=1, flags=re.M))
    with pytest.raises(SystemExit, match=r"SISTEER_REGISTRY_ROOT='\$HOME/registry' holds \$HOME, which is not a site placeholder"):
        T.Site(str(cfg), dry=True)
    r = _shell_load(cfg, then='echo "REG=$SISTEER_REGISTRY_ROOT"')
    assert r.returncode == 2 and "SISTEER_REGISTRY_ROOT='$HOME/registry' holds $HOME" in r.stderr and "REG=" not in r.stdout, r.stderr
    written = tmp_path / "registry"
    cfg.write_text(re.sub(r"^SISTEER_REGISTRY_ROOT=.*$", f"SISTEER_REGISTRY_ROOT={written}", clean, count=1, flags=re.M))
    assert T.Site(str(cfg), dry=True).registry_root == str(written)
    r = _shell_load(cfg, then='echo "REG=$SISTEER_REGISTRY_ROOT"')
    assert r.returncode == 0 and f"REG={written}" in r.stdout.splitlines(), r.stderr


def _fanout_site(tmp_path: Path) -> Path:
    """The shipped example rearranged the way a site describes a second cluster it does not submit to by default: SISTEER_CLUSTERS
    names one cluster (site label `site_one`) and `clusters/` also holds `site_two.env`. Site labels, registry labels (the first two
    the reader knows) and ssh aliases all differ, so a value read from the wrong block, or a label taken for a host, shows."""
    C = _clusters_module()
    regs = C.KNOWN_SUBSTRATES[:2]
    d = tmp_path / "site"
    shutil.copytree(EXAMPLE, d)
    cfg = d / "config.env"
    text = cfg.read_text()
    listed = text.replace('SISTEER_CLUSTERS="cluster_a cluster_b"', 'SISTEER_CLUSTERS="site_one"', 1).replace(
        "SISTEER_PRIMARY_CLUSTER=cluster_a", "SISTEER_PRIMARY_CLUSTER=site_one", 1)
    assert listed.count("site_one") == 2
    cfg.write_text(listed)
    for old, new, reg, login in (("cluster_a", "site_one", regs[0], "login-one"), ("cluster_b", "site_two", regs[1], "login-two")):
        src = d / "clusters" / f"{old}.env"
        body = src.read_text()
        for key, value in (("CLUSTER_SSH", login), ("CLUSTER_SUBSTRATE", reg), ("CLUSTER_LABEL", new)):
            body, n = re.subn(rf"^{key}=.*$", f"{key}={value}", body, count=1, flags=re.M)
            assert n == 1, (old, key)
        (d / "clusters" / f"{new}.env").write_text(body)
        src.unlink()
    return cfg


def _clusters_cli(cfg: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "reproduce" / "lib" / "clusters.py"), "--config", str(cfg), *args],
                          capture_output=True, text=True)


def test_a_lookup_by_label_reaches_a_block_the_site_does_not_fan_out_to(tmp_path):
    """SISTEER_CLUSTERS is the set the front door fans out to by default; a lookup BY LABEL (block_for, --get, --job-paths) reads every
    block in the site's clusters/ directory. A second cluster that only data names (its registry label on records, a pins source, a
    store destination) resolves to its own login host and roots, and stays out of the default fan-out."""
    C = _clusters_module()
    reg_one, reg_two = C.KNOWN_SUBSTRATES[:2]
    cfg = _fanout_site(tmp_path)
    d = cfg.parent
    two = C.load_cluster(str(d), "site_two")
    assert two["CLUSTER_SSH"] == "login-two" and two["CLUSTER_SHARED_ROOT"].startswith("/")
    assert list(C.load_all(str(d), C.site_config_file(str(cfg)))) == ["site_one"]
    for label in (reg_two, "site_two"):
        b = C.block_for(label, str(cfg))
        assert (b["CLUSTER_SSH"], b["CLUSTER_SHARED_ROOT"], b["CLUSTER_LABEL"]) == ("login-two", two["CLUSTER_SHARED_ROOT"], "site_two")
    assert C.block_for(reg_one, str(cfg))["CLUSTER_SSH"] == "login-one"
    r = _clusters_cli(cfg, "--get", reg_two, "CLUSTER_SSH")
    assert r.returncode == 0 and r.stdout.strip() == "login-two", r.stderr
    r = _clusters_cli(cfg, "--get", "site_two", "CLUSTER_LOG_DIR")
    assert r.returncode == 0 and r.stdout.strip() == two["CLUSTER_LOG_DIR"] and "$" not in r.stdout, r.stderr
    r = _clusters_cli(cfg, "--job-paths", reg_two)
    assert r.returncode == 0 and f"export SISTEER_PROJECT_ROOT={two['CLUSTER_PROJECT_ROOT']}" in r.stdout.splitlines(), r.stdout + r.stderr
    # the default fan-out is the listed cluster alone, in the Python reader and in the front door's shell
    assert _clusters_cli(cfg, "--labels").stdout.split() == ["site_one"]
    assert set(json.loads(_clusters_cli(cfg, "--json").stdout)) == {"site_one"}
    assert "CLUSTER__site_two__" not in _clusters_cli(cfg, "--sh").stdout
    r = _shell_load(cfg, then='repro_resolve_clusters; echo "CLUSTERS=$CLUSTERS"; cluster_require site_two; echo REACHED')
    assert "CLUSTERS=site_one" in r.stdout.splitlines() and "REACHED" not in r.stdout and "not one of SISTEER_CLUSTERS" in r.stderr, r.stdout + r.stderr
    # a label no block carries refuses by name (exit 3) and asks for a block, not for a fan-out entry
    r = _clusters_cli(cfg, "--get", "site_three", "CLUSTER_SSH")
    assert r.returncode == 3 and r.stdout == "" and "'site_three'" in r.stderr and "name it in SISTEER_CLUSTERS" not in r.stderr, r.stderr
    assert "site_one site_two" in r.stderr, r.stderr


def test_an_unlisted_block_is_validated_like_a_listed_one_and_an_ambiguous_label_refuses(tmp_path):
    """Every block a lookup can reach is read in full: one that fails validation refuses every lookup by name (it may carry the label
    asked for), a `.env` file under clusters/ whose name is not a site label refuses, and a label that is one block's site label and
    another block's registry label refuses rather than picking one. The default fan-out reads the listed blocks alone throughout."""
    C = _clusters_module()
    reg_one, reg_two = C.KNOWN_SUBSTRATES[:2]
    cfg = _fanout_site(tmp_path)
    blocks_dir = cfg.parent / "clusters"
    two = blocks_dir / "site_two.env"
    good = two.read_text()
    two.write_text(re.sub(r"^CLUSTER_QOS=.*\n", "", good, count=1, flags=re.M))
    with pytest.raises(SystemExit, match=r"site_two\.env must set CLUSTER_QOS"):
        C.block_for(reg_one, str(cfg))
    r = _clusters_cli(cfg, "--get", reg_one, "CLUSTER_SSH")
    assert r.returncode == 2 and "site_two.env must set CLUSTER_QOS" in r.stderr and r.stdout == "", r.stderr
    assert _clusters_cli(cfg, "--sh").returncode == 0
    two.write_text(good)
    misnamed = blocks_dir / "Site-Three.env"
    misnamed.write_text(good)
    with pytest.raises(SystemExit, match=r"Site-Three\.env"):
        C.block_for(reg_two, str(cfg))
    assert _clusters_cli(cfg, "--get", reg_two, "CLUSTER_SSH").returncode == 2
    misnamed.unlink()
    # a block file named for site_one's registry label, carrying the generic registry label itself
    clash = blocks_dir / f"{reg_one}.env"
    body = re.sub(r"^CLUSTER_SUBSTRATE=.*$", "CLUSTER_SUBSTRATE=slurm", good, count=1, flags=re.M)
    clash.write_text(re.sub(r"^CLUSTER_LABEL=.*$", f"CLUSTER_LABEL={reg_one}", body, count=1, flags=re.M))
    with pytest.raises(SystemExit, match=rf"{reg_one}.*site_one"):
        C.block_for(reg_one, str(cfg))
    r = _clusters_cli(cfg, "--get", reg_one, "CLUSTER_SSH")
    assert r.returncode == 3 and r.stdout == "" and reg_one in r.stderr, r.stderr
    assert C.block_for("site_one", str(cfg))["CLUSTER_SSH"] == "login-one"      # a site label no other block carries still resolves


def _example_listing_cluster_a(tmp_path: Path) -> Path:
    """The shipped example with SISTEER_CLUSTERS="cluster_a"; clusters/ still holds both block files, cluster_a.env and
    cluster_b.env (a site that submits to cluster A by default and reaches cluster B only by naming it)."""
    d = tmp_path / "example"
    shutil.copytree(EXAMPLE, d)
    cfg = d / "config.env"
    text = cfg.read_text()
    listed = text.replace('SISTEER_CLUSTERS="cluster_a cluster_b"', 'SISTEER_CLUSTERS="cluster_a"', 1)
    assert listed != text and (d / "clusters" / "cluster_b.env").is_file()
    cfg.write_text(listed)
    return cfg


def test_a_site_listing_cluster_a_resolves_cluster_b_by_label_in_both_readers_and_fans_out_to_cluster_a(tmp_path, capsys):
    """SISTEER_CLUSTERS="cluster_a" with both block files: `clusters.py --get cluster_b CLUSTER_SSH`, block_for and
    sisteer.site's expansion for a named cluster all read cluster_b's own block, while the default fan-out (--labels, --json,
    --sh, the front door's repro_resolve_clusters, Site.labels and primary) stays cluster A alone. A label no block file
    carries still refuses by name in both readers."""
    C = _clusters_module()
    cfg = _example_listing_cluster_a(tmp_path)
    d = cfg.parent
    a, b = C.load_cluster(str(d), "cluster_a"), C.load_cluster(str(d), "cluster_b")
    assert a["CLUSTER_SSH"] != b["CLUSTER_SSH"] and a["CLUSTER_SHARED_ROOT"] != b["CLUSTER_SHARED_ROOT"]
    r = _clusters_cli(cfg, "--get", "cluster_b", "CLUSTER_SSH")
    assert r.returncode == 0 and r.stdout.strip() == b["CLUSTER_SSH"], r.stderr
    r = _clusters_cli(cfg, "--get", "cluster_a", "CLUSTER_SSH")
    assert r.returncode == 0 and r.stdout.strip() == a["CLUSTER_SSH"], r.stderr
    assert C.block_for("cluster_b", str(cfg))["CLUSTER_SSH"] == b["CLUSTER_SSH"]
    assert C.block_for("cluster_a", str(cfg))["CLUSTER_SSH"] == a["CLUSTER_SSH"]
    # the default fan-out: cluster A alone, in the Python reader and in the front door's shell
    assert _clusters_cli(cfg, "--labels").stdout.split() == ["cluster_a"]
    assert set(json.loads(_clusters_cli(cfg, "--json").stdout)) == {"cluster_a"}
    assert "CLUSTER__cluster_b__" not in _clusters_cli(cfg, "--sh").stdout
    r = _shell_load(cfg, then='repro_resolve_clusters; echo "CLUSTERS=$CLUSTERS"')
    assert r.returncode == 0 and "CLUSTERS=cluster_a" in r.stdout.splitlines(), r.stdout + r.stderr
    # sisteer.site: the default set is unchanged, and a named cluster outside it reads its own block file
    site = S.load_site(cfg, env={})
    assert site.labels == ["cluster_a"] and site.primary == "cluster_a"
    assert S.expand("$SHARED/x", site=site, env={}) == f"{a['CLUSTER_SHARED_ROOT']}/x"
    assert S.expand("$SHARED/x", "cluster_b", site=site, env={"SISTEER_SHARED_ROOT": "/job/shared"}) == f"{b['CLUSTER_SHARED_ROOT']}/x"
    assert S.expand("$HF_HOME", "cluster_b", site=site, env={}) == b["CLUSTER_HF_HOME"] == f"{b['CLUSTER_SHARED_ROOT']}/hf_cache"
    assert S.main(["expand", "--config", str(cfg), "--cluster", "cluster_b", "$SHARED/x"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{b['CLUSTER_SHARED_ROOT']}/x"]
    # a label no block file carries refuses by name: clusters.py exit 3, sisteer.site names the file it looked for
    r = _clusters_cli(cfg, "--get", "cluster_c", "CLUSTER_SSH")
    assert r.returncode == 3 and r.stdout == "" and "'cluster_c'" in r.stderr, r.stderr
    with pytest.raises(S.PlaceholderError, match=r"cluster 'cluster_c', which the site config .* does not name .*clusters/cluster_c\.env"):
        S.expand("$SHARED/x", "cluster_c", site=site, env={})
    with pytest.raises(S.PlaceholderError, match=r"does not name"):
        S.expand("$SHARED/x", "../cluster_b", site=site, env={})    # a name that is not a site label is never a file name


def test_two_block_files_claiming_one_registry_label_refuse_a_lookup_by_it_listed_or_not(tmp_path):
    """The shipped example's two blocks both carry the generic registry label. With cluster_a alone listed, a lookup by that
    label must not settle on the listed block: the unlisted file claims it too, so the lookup refuses by name and asks for
    the site label. The same holds for two unlisted files sharing a specific registry label. Each block still resolves by
    its own site label, and the default fan-out is untouched."""
    C = _clusters_module()
    cfg = _example_listing_cluster_a(tmp_path)
    d = cfg.parent
    generic = C.load_cluster(str(d), "cluster_a")["CLUSTER_SUBSTRATE"]
    assert C.load_cluster(str(d), "cluster_b")["CLUSTER_SUBSTRATE"] == generic
    r = _clusters_cli(cfg, "--get", generic, "CLUSTER_SSH")
    assert r.returncode == 3 and r.stdout == "" and "more than one cluster block (cluster_a cluster_b)" in r.stderr, r.stderr
    with pytest.raises(SystemExit, match=r"more than one cluster block \(cluster_a cluster_b\).*site label"):
        C.block_for(generic, str(cfg))
    assert [C.block_for(lab, str(cfg))["CLUSTER_LABEL"] for lab in ("cluster_a", "cluster_b")] == ["cluster_a", "cluster_b"]
    # two UNLISTED files carrying one specific registry label (site labels that are no registry label, before or after renames)
    reg_one, reg_two = C.KNOWN_SUBSTRATES[:2]
    cfg = _fanout_site(tmp_path / "specific")
    blocks_dir = cfg.parent / "clusters"
    assert C.block_for(reg_two, str(cfg))["CLUSTER_LABEL"] == "site_two"          # one carrier: the lookup resolves
    body = (blocks_dir / "site_two.env").read_text()
    (blocks_dir / "site_three.env").write_text(re.sub(r"^CLUSTER_LABEL=.*$", "CLUSTER_LABEL=site_three", body, count=1, flags=re.M))
    def carriers(text: str) -> set:      # the blocks a duplicate-label refusal names (the unlisted ones in file-name order)
        m = re.search(r"more than one cluster block \(([^)]*)\)", text)
        return set(m.group(1).split()) if m else set()
    r = _clusters_cli(cfg, "--get", reg_two, "CLUSTER_SSH")
    assert r.returncode == 3 and r.stdout == "" and carriers(r.stderr) == {"site_two", "site_three"}, r.stderr
    with pytest.raises(SystemExit) as refused:
        C.block_for(reg_two, str(cfg))
    assert carriers(str(refused.value.code)) == {"site_two", "site_three"}, refused.value.code
    assert C.block_for(reg_one, str(cfg))["CLUSTER_LABEL"] == "site_one"
    for label in ("site_one", "site_two", "site_three"):
        assert C.block_for(label, str(cfg))["CLUSTER_LABEL"] == label
    assert _clusters_cli(cfg, "--labels").stdout.split() == ["site_one"]

def test_the_shell_closed_set_is_the_python_one():
    """reproduce/lib/common.sh names the same closed set of placeholders as this module, so a value the shell twin
    refuses as unknown is one the Python expansion refuses too."""
    lib = (REPO / "reproduce" / "lib" / "common.sh").read_text()
    (line,) = [ln for ln in lib.splitlines() if ln.startswith("REPRO_SITE_PLACEHOLDERS=")]
    assert line.split('"')[1].split() == list(S.NAMES)


# --- setting() and state_path(): the site settings a moved default reads ------------------------------------------------

def test_setting_reads_the_environment_then_the_site_config_then_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SISTEER_EXAMPLE_KEY", raising=False)
    assert S.setting("SISTEER_EXAMPLE_KEY") is None                         # no site file on this host: no error, no value
    assert S.setting("SISTEER_EXAMPLE_KEY", "fallback") == "fallback"
    cfg = _site(tmp_path, "SISTEER_EXAMPLE_KEY='~/from-site'\nSISTEER_EMPTY_KEY=''\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    assert S.setting("SISTEER_EXAMPLE_KEY") == "~/from-site"                 # returned as written: no expansion
    assert S.setting("SISTEER_EMPTY_KEY", "d") == "d"                        # an empty value counts as unset
    monkeypatch.setenv("SISTEER_EXAMPLE_KEY", "/from/env")
    assert S.setting("SISTEER_EXAMPLE_KEY") == "/from/env"                   # the environment wins
    monkeypatch.setenv("SISTEER_EXAMPLE_KEY", "")
    assert S.setting("SISTEER_EXAMPLE_KEY") == "~/from-site"                 # an empty environment value counts as unset too
    monkeypatch.setenv("SISTEER_CONFIG", str(tmp_path / "absent.env"))
    monkeypatch.delenv("SISTEER_EXAMPLE_KEY")
    with pytest.raises(S.SiteConfigError):                                   # a file asked for by name and missing is loud
        S.setting("SISTEER_EXAMPLE_KEY")


def test_state_path_is_the_key_then_state_then_the_home_default_and_never_holds_a_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SISTEER_EXAMPLE_DIR", raising=False)
    assert S.state_path("SISTEER_EXAMPLE_DIR", "pkg", "sub") == tmp_path / "home/.local/state/pkg/sub"
    cfg = _site(tmp_path, "SISTEER_STATE_ROOT='~/st'\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    assert S.state_path("SISTEER_EXAMPLE_DIR", "pkg", "sub") == tmp_path / "home/st/pkg/sub"
    cfg.write_text("SISTEER_STATE_ROOT='~/st'\nSISTEER_GRADE_DIR=/srv/grade\nSISTEER_EXAMPLE_DIR='$GRADE/moved'\n")
    S._SITE_CACHE.clear()
    got = S.state_path("SISTEER_EXAMPLE_DIR", "pkg", "sub")
    assert got == Path("/srv/grade/moved") and "$" not in str(got)
    monkeypatch.setenv("SISTEER_EXAMPLE_DIR", "~/env-dir")
    assert S.state_path("SISTEER_EXAMPLE_DIR", "pkg") == tmp_path / "home/env-dir"
    monkeypatch.setenv("SISTEER_EXAMPLE_DIR", "$NOT_A_PLACEHOLDER/x")
    with pytest.raises(S.PlaceholderError, match="SISTEER_EXAMPLE_DIR"):      # set but not expandable: refused, no fallback
        S.state_path("SISTEER_EXAMPLE_DIR", "pkg")
