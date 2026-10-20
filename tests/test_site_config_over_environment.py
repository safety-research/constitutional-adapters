"""Every site scalar submit_twin.Site reads resolves the way the stage that launches the funnel resolves it: the config file the run was
GIVEN laid over the process environment — a key the file names (empty included) is the file's, a key it does not name is the environment's,
then the default — because a stage `set -a; source`s that file over its environment (common.sh repro_load_config) and then reads
`${KEY:-default}`, and the funnel it launches must land in the SAME roots, name the SAME spool, primary and venv.

Site.__init__ read SISTEER_STATE_DIR and SISTEER_REGISTRY_ROOT environment-FIRST (`os.environ.get(k) or cfg.get(k)`), the opposite
order from the stage's source. A direct `submit_twin.py --config <file> … submit` from a shell carrying either key — no stage having
sourced the file — wrote its attempts / submissions ledger outside the state root the file names (so 04 and 06 --gate, which read
that root, never saw the attempt), and `--resubmit` read an unrelated registry for its fenced proof. The AUDIT of the other Site
fields found the opposite class on the rest: the primary, the dual-submit knobs, the grant timeout, the reproduce root, the fit venv
name, the spool and the models table were read from the FILE ALONE, so a key the environment carried and the file did not name was
the stage's value and not the funnel's; and the mirror-status path read the environment first like the roots. One read now —
Site.given — for every site scalar; this module runs every key over every spelling against the shell face after repro_load_config and
against the rule written out on its own, then that defect's interface end to end (the real CLI with the front-door fake Slurm: the
ledger lands under the file's state root, `--resubmit` refuses by the file's registry root) and the stage path (the file sourced into
the environment, then the funnel). Synthetic site (cluster_a / cluster_b); nothing here names a real host, user or path."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, job_ids, make_registry, set_job_state, write_site  # noqa: E402
from test_submit_resubmit_proof import MMLU, _pend  # noqa: E402
from test_submit_twin import World  # noqa: E402

#: the site keys Site reads (every `given` in reproduce/lib/submit_twin.py), each with the value a file spells, the value an environment
#: spells, and the default BOTH faces fall to when neither names it (None = the key has no default: the shell refuses by name)
KEYS = {
    "SISTEER_STATE_DIR": ("{tmp}/file_state", "{tmp}/env_state", None),            # the finding's two roots (environment-first before)
    "SISTEER_REGISTRY_ROOT": ("{tmp}/file_registry", "{tmp}/env_registry", None),
    "SISTEER_PRIMARY_CLUSTER": ("cluster_b", "cluster_b", None),                    # the audit: file-only before
    "SISTEER_DUAL_SUBMIT_THRESHOLD_MIN": ("3", "5", "10"),
    "SISTEER_DUAL_SUBMIT_MAX_JOBS": ("4", "6", "16"),
    "SISTEER_GRANT_TIMEOUT_MIN": ("7", "9", "10"),
    "SISTEER_REPRODUCE_ROOT": ("$SHARED/exp/file_root", "$SHARED/exp/env_root", "$SHARED/exp/reproduce"),
    "SISTEER_FIT_ENV_NAME": ("file-fit", "env-fit", "sdfbase-fit"),
    "SISTEER_REGISTRY_SPOOL": ("{tmp}/file_spool", "{tmp}/env_spool", "{shared_a}/registry/records"),
    "SISTEER_MODELS_TABLE": ("{tmp}/file_models.yaml", "{tmp}/env_models.yaml", str(REPRO / "models.yaml")),
    "SISTEER_REGISTRY_MIRROR_STATUS": ("{tmp}/file_mirror.json", "{tmp}/env_mirror.json", None),   # the audit: environment-first before; no shell reader
}
#: the keys the shell REQUIRES after the merge (repro_load_config refuses `must set <k>` when the merged view leaves one empty)
SHELL_REQUIRED = ("SISTEER_PRIMARY_CLUSTER", "SISTEER_REGISTRY_ROOT")
#: how the file and the environment can spell one key: (the file names it?, the environment carries it?) — the file's spelling may be
#: EMPTY (`KEY=`), which `set -a; source` sets empty and `${KEY:-default}` then reads as unset
SPELLINGS = ("file_over_env", "env_alone", "file_alone", "file_empty_over_env", "neither")
#: the process-environment knobs submit_twin reads that are NOT site keys (never in reproduce/site.example/config.env or
#: docs/configuration.md): a lock-wait bound, the crash-injection seam and the watcher seam — the sweep below allows exactly these
NOT_SITE_KEYS = ("SISTEER_UNIT_LOCK_TIMEOUT_S", "SISTEER_TWIN_CRASH_AT", "SISTEER_NO_WATCHER")


def _base_env() -> dict:
    """The inherited environment less every key under test (and the stage's own pointers), so a developer's shell decides no row."""
    drop = set(KEYS) | {"SISTEER_SITE_DIR", "SISTEER_CONFIG", "REPRO_CONFIG"}
    return {k: v for k, v in os.environ.items() if k not in drop}


def _bash(script: str, env: dict, *, ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["bash", "-c", f"set -u\n{script}"], capture_output=True, text=True, env=env, timeout=120)
    if ok:
        assert r.returncode == 0, r.stdout + r.stderr
    return r


def _sh(cfg: Path, body: str, env: dict, *, ok: bool = True) -> subprocess.CompletedProcess:
    """The shell library loaded as a stage loads it (common.sh, slurm.sh, the GIVEN config sourced by repro_load_config), then `body` —
    with exactly the environment given (nothing of this module's added)."""
    return _bash(f"source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={cfg}; repro_load_config; {body}", env, ok=ok)


def _submit_twin():
    spec = importlib.util.spec_from_file_location("submit_twin_under_site_test", LIB / "submit_twin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _spell(key: str, spelling: str, tmp: Path, shared_a: str):
    """(the file's value or None, the environment's value or None) for one key under one spelling."""
    f, e, _ = KEYS[key]
    f = f.format(tmp=tmp, shared_a=shared_a); e = e.format(tmp=tmp, shared_a=shared_a)
    return {"file_over_env": (f, e), "env_alone": (None, e), "file_alone": (f, None), "file_empty_over_env": ("", e), "neither": (None, None)}[spelling]


def _rule(file_value, env_value, default):
    """What the stage path resolves, written out independently of both faces: the file's value when the file names the key (empty included),
    else the environment's; then `${KEY:-default}` — empty is unset."""
    merged = file_value if file_value is not None else env_value
    return merged or default


def _config_with(site: Path, key: str, value) -> Path:
    """write_site's config.env with `key` spelled `value`: None removes the line (the file does not name the key), "" names it empty,
    else `KEY=value` (single-quoted when it carries a literal `$SHARED`, as site.example writes it)."""
    cfg = site / "config.env"
    lines = [ln for ln in cfg.read_text().splitlines() if not ln.startswith(f"{key}=")]
    if value is not None:
        lines.append(f"{key}='{value}'" if "$" in value else f"{key}={value}")
    cfg.write_text("\n".join(lines) + "\n")
    return cfg


def _python_face(st, site: Path, key: str) -> "str | None":
    """The funnel's reading of `key` through the attribute or composer that consumes it (dry: nothing is created)."""
    s = st.Site(str(site / "config.env"), dry=True)
    if key == "SISTEER_STATE_DIR":
        return s.state_dir("m")
    if key == "SISTEER_REGISTRY_ROOT":
        return s.registry_root
    if key == "SISTEER_PRIMARY_CLUSTER":
        return s.primary
    if key == "SISTEER_DUAL_SUBMIT_THRESHOLD_MIN":
        return str(s.threshold_s // 60)
    if key == "SISTEER_DUAL_SUBMIT_MAX_JOBS":
        return str(s.max_jobs)
    if key == "SISTEER_GRANT_TIMEOUT_MIN":
        return str(s.grant_timeout_min)
    if key == "SISTEER_REPRODUCE_ROOT":
        return s.cluster_root("cluster_a", "m")
    if key == "SISTEER_FIT_ENV_NAME":
        co = st.Coordinator(s, "m")
        assert co.job_env("cluster_a")["SDFBASE_FIT_ENV_NAME"] == s.fit_env_name("cluster_a")
        return s.fit_env_name("cluster_a")
    if key == "SISTEER_REGISTRY_SPOOL":
        return st.Coordinator(s, "m").job_env("cluster_a")["SISTEER_REGISTRY_SPOOL"]
    if key == "SISTEER_MODELS_TABLE":
        return s.given(key) or None          # as the consumer reads it: `table=self.site.given("SISTEER_MODELS_TABLE") or None`
    if key == "SISTEER_REGISTRY_MIRROR_STATUS":
        return st.Coordinator(s, "m").mirror_status_path()
    raise AssertionError(key)


def _shell_face(cfg: Path, key: str, env: dict) -> "str | None":
    """The stage's reading of the same key after repro_load_config: the shell helper 03/04/05/06 call for it (None: no shell reader)."""
    body = {
        "SISTEER_STATE_DIR": "repro_state_dir m",
        "SISTEER_REGISTRY_ROOT": "repro_registry_root",
        "SISTEER_PRIMARY_CLUSTER": 'echo "$SISTEER_PRIMARY_CLUSTER"',
        "SISTEER_DUAL_SUBMIT_THRESHOLD_MIN": 'echo "$SISTEER_DUAL_SUBMIT_THRESHOLD_MIN"',
        "SISTEER_DUAL_SUBMIT_MAX_JOBS": 'echo "$SISTEER_DUAL_SUBMIT_MAX_JOBS"',
        "SISTEER_GRANT_TIMEOUT_MIN": 'echo "$SISTEER_GRANT_TIMEOUT_MIN"',
        "SISTEER_REPRODUCE_ROOT": "repro_cluster_state_root cluster_a m",
        "SISTEER_FIT_ENV_NAME": "cluster_fit_env_name cluster_a",
        "SISTEER_REGISTRY_SPOOL": "slurm_job_env cluster_a",
        "SISTEER_MODELS_TABLE": "repro_models_table",
    }.get(key)
    if body is None:
        return None
    out = _sh(cfg, body, env).stdout.strip()
    if key == "SISTEER_REGISTRY_SPOOL":
        m = re.search(r"(?:^|\s)SISTEER_REGISTRY_SPOOL=(\S+)", out)
        assert m, out
        return m.group(1)
    return out


def _expected(key: str, want: "str | None", tmp: Path, shared_a: str) -> "str | None":
    """The rule's value in the form the two faces print it."""
    if key == "SISTEER_STATE_DIR":
        return f"{want or (os.path.expanduser('~/.local/state/sisteer-reproduce'))}/m"
    if key == "SISTEER_REPRODUCE_ROOT":
        return f"{want.replace('$SHARED', shared_a)}/m"
    if key == "SISTEER_REGISTRY_MIRROR_STATUS":
        return want     # the default is <registry root>/state/mirror_status.json, asserted at the call site
    return want


# ---- the table: every site key, every spelling, both faces and the rule -----------------------------------------------------------------

def test_every_site_key_resolves_the_given_file_over_the_environment_on_both_faces(tmp_path, monkeypatch):
    """RED before the fix on: the roots under `file_over_env` (`'{tmp}/env_state/m' == '{tmp}/file_state/m'`: the environment beat the
    file), the mirror-status path under `file_over_env`, and the primary / knobs / reproduce root / fit venv / spool / models table under
    `env_alone` (the file-only reads defaulted where the stage read the environment)."""
    st = _submit_twin()
    rows = []
    for key, (_, _, default) in KEYS.items():
        for spelling in SPELLINGS:
            site = write_site(tmp_path / key.lower() / spelling)
            shared_a = str(tmp_path / key.lower() / spelling / "shared_cluster_a")
            file_value, env_value = _spell(key, spelling, tmp_path, shared_a)
            cfg = _config_with(site, key, file_value)
            env = _base_env()
            if env_value is not None:
                env[key] = env_value
            # the shell requires the merged view to carry the primary and the registry root: it refuses BY NAME, and the Python face
            # is not compared on a row the stage would never reach
            merged = file_value if file_value is not None else env_value
            if key in SHELL_REQUIRED and not merged:
                r = _sh(cfg, "true", env, ok=False)
                assert r.returncode != 0 and f"must set {key}" in r.stdout + r.stderr, (key, spelling, r.stdout + r.stderr)
                rows.append((key, spelling, "shell refuses by name"))
                continue
            want = _rule(file_value, env_value, default.format(tmp=tmp_path, shared_a=shared_a) if default else None)
            expected = _expected(key, want, tmp_path, shared_a)
            for k in KEYS:
                monkeypatch.delenv(k, raising=False)
            if env_value is not None:
                monkeypatch.setenv(key, env_value)
            python_face = _python_face(st, site, key)
            if key == "SISTEER_REGISTRY_MIRROR_STATUS" and expected is None:
                expected = os.path.join(st.Site(str(cfg), dry=True).registry_root, "state", "mirror_status.json")
            if key == "SISTEER_MODELS_TABLE" and want == str(REPRO / "models.yaml"):
                expected = None          # the funnel hands `None` to stage.workload_baseline, whose default is the same models.yaml
            assert python_face == expected, (key, spelling, python_face, expected)
            shell_face = _shell_face(cfg, key, env)
            if shell_face is not None:
                assert shell_face == (want if key != "SISTEER_STATE_DIR" and key != "SISTEER_REPRODUCE_ROOT" else expected), (key, spelling, shell_face, want)
                if key not in ("SISTEER_MODELS_TABLE",):
                    assert shell_face == python_face, (key, spelling, shell_face, python_face)
            rows.append((key, spelling, python_face))
    assert len(rows) == len(KEYS) * len(SPELLINGS)
    # a primary the merged view names that the given file's cluster list does not hold refuses by name on both faces (never a label
    # with no block; before: the Python face took it and every later block lookup raised KeyError)
    site = write_site(tmp_path / "bad_primary")
    cfg = _config_with(site, "SISTEER_PRIMARY_CLUSTER", None)
    env = dict(_base_env(), SISTEER_PRIMARY_CLUSTER="cluster_zz")
    r = _sh(cfg, "true", env, ok=False)
    assert r.returncode != 0 and "cluster_zz" in r.stdout + r.stderr
    monkeypatch.setenv("SISTEER_PRIMARY_CLUSTER", "cluster_zz")
    with pytest.raises(SystemExit, match="cluster_zz"):
        st.Site(str(cfg), dry=True)


# ---- the finding's interface end to end: the direct CLI, a shell carrying the other roots ------------------------------------------------

def test_a_direct_submit_writes_its_ledger_under_the_files_state_root_and_resubmit_proves_against_the_files_registry(tmp_path):
    """An operator runs `submit_twin.py --config <file> --model m submit …` directly from a shell whose SISTEER_STATE_DIR and
    SISTEER_REGISTRY_ROOT name OTHER roots than the file does (a stage sources the file over them; the direct CLI must read it the same
    way). The attempts and submissions ledgers land under the FILE's state root and nothing is created under the environment's; and when
    the job dies, `--resubmit`'s fenced proof reads the FILE's registry root — here empty of a database, so it refuses by THAT root's name
    (fail closed) although the environment's root holds a database. RED before the fix: the ledger under `<tmp>/env_state/m`, nothing under
    the file's root, and the proof read `<tmp>/env_registry`."""
    w = World(tmp_path)
    (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    env_state, env_registry = tmp_path / "env_state", tmp_path / "env_registry"
    make_registry(env_registry)                                   # a database under the ENVIRONMENT's root, none under the file's below
    other = {"SISTEER_STATE_DIR": str(env_state), "SISTEER_REGISTRY_ROOT": str(env_registry)}
    file_state = tmp_path / "state"
    assert f"SISTEER_STATE_DIR={file_state}" in (w.site / "config.env").read_text() and f"SISTEER_REGISTRY_ROOT={tmp_path / 'registry'}" in (w.site / "config.env").read_text()
    r1 = w.run(*MMLU, **other)
    assert r1.returncode == 0 and "submitted cluster_a 1001 mmlu" in r1.stdout, r1.stdout + r1.stderr
    assert (file_state / "m" / "attempts.jsonl").is_file() and (file_state / "m" / "submissions.jsonl").is_file(), sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.jsonl"))
    assert not env_state.exists(), sorted(str(p.relative_to(tmp_path)) for p in env_state.rglob("*"))
    assert [e["event"] for e in w.ledger()][:1] == ["attempt"]
    # the job dies; the FILE's registry root has no database: the proof refuses by that root's name, never reading the environment's
    _pend(w, "cluster_a", "1001", ["run-a"])
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    w.spool("cluster_a")
    (w.tmp / "registry" / "registry.sqlite").unlink()
    r2 = w.run(*MMLU, "--resubmit", **other)
    assert r2.returncode == 3 and f"no registry database under {w.tmp / 'registry'}" in r2.stderr and "fail closed" in r2.stderr, r2.stdout + r2.stderr
    assert str(env_registry) not in r2.stdout + r2.stderr and job_ids(w.fake["cluster_a"]) == [1001]
    assert not env_state.exists()
    # the file's root holds a database again: the proof runs there and the rerun goes out — the ledger still under the file's root
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "running", "cluster_job_id": "1001"}])
    w.ingested("cluster_a", through="2026-01-01T00:20:00")
    r3 = w.run(*MMLU, "--resubmit", **other)
    assert r3.returncode == 0 and "submitted cluster_a 1002 mmlu" in r3.stdout, r3.stdout + r3.stderr
    assert len([e for e in w.ledger() if e["event"] == "attempt"]) == 2 and not env_state.exists()


# ---- the stage path: the file sourced into the environment, then the funnel -------------------------------------------------------------

def test_the_stage_path_and_the_direct_cli_read_one_value_for_every_root_and_knob(tmp_path):
    """A stage (04, 06, reconcile.sh) runs repro_load_config over the given file — `set -a; source` — and then calls the funnel, whose
    process inherits the sourced environment. With a shell carrying OTHER values for keys the file names (the two roots, the grant
    timeout) and a value for a key the file does not name (the fit venv name), the funnel inside the stage and the funnel called directly
    over the same file from the same shell read ONE value per key: the file's where the file names it, the shell's where it does not —
    the values the stage's own helpers print. RED before the fix: the direct call read the shell's roots and ignored the shell's venv
    name, so the two paths disagreed on three of the four."""
    site = write_site(tmp_path)
    shell_env = dict(_base_env(), SISTEER_STATE_DIR=str(tmp_path / "env_state"), SISTEER_REGISTRY_ROOT=str(tmp_path / "env_registry"),
                     SISTEER_GRANT_TIMEOUT_MIN="7", SISTEER_FIT_ENV_NAME="env-fit")
    probe = ("import os, sys; sys.path.insert(0, sys.argv[1]); import submit_twin as T; s = T.Site(sys.argv[2], True); c = T.Coordinator(s, 'm'); "
             "print('|'.join([s.state_dir('m'), s.registry_root, str(s.grant_timeout_min), s.fit_env_name('cluster_a'), c.job_env('cluster_a')['SDFBASE_FIT_ENV_NAME']]))")
    # inside the stage: the file sourced over the shell, then the funnel (as 04's `"$PY" "$REPRO_LIB_DIR/submit_twin.py" --config "$SISTEER_CONFIG"` is)
    staged = _sh(site / "config.env", f'"$(repro_python)" -c {json.dumps(probe)} "{LIB}" "$SISTEER_CONFIG"; echo; '
                 'echo "$(repro_state_dir m)|$(repro_registry_root)|$SISTEER_GRANT_TIMEOUT_MIN|$(cluster_fit_env_name cluster_a)|$(cluster_fit_env_name cluster_a)"',
                 shell_env).stdout.strip().splitlines()
    funnel_in_stage, stage_helpers = staged[0], staged[-1]
    # the direct CLI from the same shell: no stage sourced the file
    direct = subprocess.run([sys.executable, "-c", probe, str(LIB), str(site / "config.env")], capture_output=True, text=True, env=shell_env, timeout=120)
    assert direct.returncode == 0, direct.stdout + direct.stderr
    want = f"{tmp_path / 'state' / 'm'}|{tmp_path / 'registry'}|1|env-fit|env-fit"     # the file's roots and grant timeout (1); the shell's venv name
    assert stage_helpers == want, (stage_helpers, want)
    assert funnel_in_stage == want, (funnel_in_stage, want)
    assert direct.stdout.strip() == want, (direct.stdout.strip(), want)


# ---- the sweep: no site key is read from the environment first or from the file alone ---------------------------------------------------

def test_no_site_scalar_in_the_funnel_is_read_environment_first_or_from_the_file_alone():
    """Every SISTEER_* read in reproduce/lib/submit_twin.py is `self.given(...)` / `self.site.given(...)` (the one read), except the three
    process knobs that are not site keys; no `cfg.get(` / `cfg[` outside `given` itself; no `os.environ.get("SISTEER_…") or …cfg`."""
    twin = (LIB / "submit_twin.py").read_text()
    body = twin[twin.index("class Site:"):]
    given_def = '    def given(self, key: str) -> Optional[str]:'
    assert given_def in body and body.count("        return self.cfg[key] if key in self.cfg else os.environ.get(key)\n") == 1
    cfg_reads = [ln.strip() for ln in body.splitlines() if re.search(r"\bcfg\.get\(|\bcfg\[", ln) and "self.cfg[key] if key in self.cfg" not in ln
                 and not ln.strip().startswith("#")]
    assert cfg_reads == [], cfg_reads
    env_reads = []
    for i, ln in enumerate(twin.splitlines(), 1):
        for m in re.finditer(r'os\.environ(?:\.get\(|\[)\s*"(SISTEER_[A-Z0-9_]+)"', ln):
            if m.group(1) not in NOT_SITE_KEYS:
                env_reads.append(f"{i}: {ln.strip()[:140]}")
    assert env_reads == [], env_reads
    for key in KEYS:
        assert f'given("{key}")' in twin, key
    # the three knobs that stay on the environment are no site key: neither the example site nor the configuration page names them
    example = (REPRO / "site.example" / "config.env").read_text() + (REPO / "docs" / "configuration.md").read_text()
    for knob in NOT_SITE_KEYS:
        assert knob not in example, knob
    # and every key of the table IS one the stage reads from the merged view (common.sh exports or reads `${KEY:-…}` after the source),
    # the mirror-status path apart (a Python-only key)
    common = (REPRO / "lib" / "common.sh").read_text()
    for key in KEYS:
        if key != "SISTEER_REGISTRY_MIRROR_STATUS":
            assert re.search(rf"\$\{{{key}[:}}]|\${key}\b", common), key
