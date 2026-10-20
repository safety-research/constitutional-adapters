"""The registry's systemd units are templates (systemd/user/*.in) the grading host's installer renders for the site it runs on: the
checkout's path, the interpreter, the registry root and the primary cluster's store destination come from the site config, and the
gate-baseline tick's environment hook only when the site names one.

* A clone anywhere under $HOME renders with systemd's `%h` specifier: no absolute home path, no other checkout's name, no hook.
* `@STORE_DEST@` is the primary block's CLUSTER_STORE_DEST when set, else its registry label CLUSTER_SUBSTRATE, else (the generic
  `slurm`, which names no one cluster) its CLUSTER_LABEL; and it is a destination the store configuration declares, because
  `stores mirror --dest` refuses any other. The shipped example site resolves to one the shipped store configuration declares.
* An installed unit that evaluates an environment hook is never replaced by one that evaluates none; with the site naming that hook,
  the rendered gate-baseline ExecStart is the installed one, byte for byte.
* Every template value is filled; a rendered unit is its template with exactly those values substituted (comments kept).
* Every refusal names what is missing and happens before anything is written.

Synthetic sites and paths only."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, assert_no_tokens, write_site  # noqa: E402

sys.path.insert(0, str(REPRO / "lib"))
import clusters  # noqa: E402  (the one reader of cluster blocks; its substrate vocabulary decides the label used below)

INSTALL = REPO / "scripts" / "registry" / "install_grading_host.sh"
UNITS = REPO / "systemd" / "user"
STORES = REPO / "configs" / "registry" / "cluster_reads.yaml"
TEMPLATES = sorted(p.name for p in UNITS.glob("registry-*.in"))
RENDERED = sorted(t[: -len(".in")] for t in TEMPLATES)
VALUES = ("@REPO@", "@PYTHON@", "@REGISTRY_ROOT@", "@STORE_DEST@", "@UNIT_ENV_HOOK@")
#: a cluster's registry label that is a cluster's own name, not the generic `slurm` (read from the vocabulary, never spelled here)
NAMED_SUBSTRATE = next(s for s in clusters.KNOWN_SUBSTRATES if s != "slurm")
#: the store destination of write_site's default site: its primary block is generic, so the destination is its CLUSTER_LABEL
DEFAULT_DEST = "cluster_a"
GATE = "registry-gate-baseline.service"
#: the gate-baseline ExecStart a unit installed before the templates carried, in its own shape: a hook evaluated, then the tick
INSTALLED_GATE_EXEC = "/bin/sh -c 'eval \"`%h/.local/bin/secret-env`\"; exec /bin/bash %h/code/my-clone/scripts/registry/landing/baseline_refresh.sh'"


def _exe(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _stores(path: Path, *dests: str) -> Path:
    """A store configuration at `path` declaring `dests` (the installer reads only the destination names)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("destinations:\n" + "".join(f"  {d}:\n    stores_rel: registry/stores\n" for d in dests))
    return path


def _install(*args, env: dict, installer: Path = INSTALL) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(installer), *args], capture_output=True, text=True, env=env, timeout=120, cwd="/")


def _render(tmp_path: Path, env: dict, installer: Path = INSTALL) -> dict:
    """Render every unit into a scratch directory (`--units-only --no-systemctl`); {file name: text}."""
    out = tmp_path / "rendered"
    r = _install("--units-only", "--no-systemctl", "--unit-dir", str(out), env=env, installer=installer)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return {p.name: p.read_text() for p in sorted(out.iterdir())}


def _site_env(tmp_path: Path, dests=(DEFAULT_DEST,), **site_kw) -> dict:
    """A neutral site (write_site) and a store configuration declaring `dests`, named through $REGISTRY_CLUSTER_READS."""
    site = write_site(tmp_path, **site_kw)
    return {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "SISTEER_CONFIG": str(site / "config.env"),
            "REGISTRY_CLUSTER_READS": str(_stores(tmp_path / "cluster_reads.yaml", *dests))}


def _exec_start(text: str) -> str:
    return next(l for l in text.splitlines() if l.startswith("ExecStart=")).split("=", 1)[1]


def _expected_dest(site: Path) -> str:
    """The destination the installer resolves for the site's primary block: CLUSTER_STORE_DEST, else CLUSTER_SUBSTRATE, else (the
    generic `slurm`) CLUSTER_LABEL."""
    primary = clusters.parse_env_file(str(site / "config.env"))["SISTEER_PRIMARY_CLUSTER"]
    block = clusters.load_cluster(str(site), primary)
    sub = block["CLUSTER_SUBSTRATE"]
    return block.get("CLUSTER_STORE_DEST") or (block["CLUSTER_LABEL"] if sub == "slurm" else sub)


def _synthetic_clone(home: Path, name: str) -> Path:
    """A clone of this repository at <home>/code/<name>: the installer, the unit templates, the site readers it uses, and the package
    directory it checks for (nothing else of the tree is needed to render)."""
    clone = home / "code" / name
    for rel in ("scripts/registry/install_grading_host.sh", "reproduce/lib/common.sh", "reproduce/lib/clusters.py"):
        (clone / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, clone / rel)
    (clone / "src" / "sisteer").mkdir(parents=True)
    shutil.copy2(REPO / "src" / "sisteer" / "site.py", clone / "src" / "sisteer")   # clusters.py parses the site files with it
    shutil.copytree(UNITS, clone / "systemd" / "user")
    (clone / "src" / "sisteer" / "registry").mkdir(parents=True)
    return clone


def test_a_clone_under_home_renders_with_no_home_path_no_other_checkout_name_and_no_hook(tmp_path):
    """The shipped neutral site (reproduce/site.example) at its DEFAULT location, a clone at ~/code/my-clone carrying a store
    configuration that declares the site's destination, a venv at ~/venvs/my-clone, the registry at ~/registry: every path a unit
    names is `%h/<one of those>`, nothing names the test's absolute root, no template value is left, no personal identifier, the
    store-mirror destination is the primary block's, and the gate-baseline tick evaluates no hook (the site names none)."""
    home = tmp_path / "home"
    clone = _synthetic_clone(home, "my-clone")
    site = home / ".config" / "sisteer"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write("\nSISTEER_REGISTRY_ROOT=~/registry\nSISTEER_PYTHON=~/venvs/my-clone/bin/python\n")
    venv_py = home / "venvs" / "my-clone" / "bin" / "python"          # a real interpreter: it also reads the cluster blocks
    venv_py.parent.mkdir(parents=True)
    venv_py.symlink_to(sys.executable)
    dest = _expected_dest(site)
    _stores(clone / "configs" / "registry" / "cluster_reads.yaml", dest)   # the clone's own store configuration, which the units read
    env = {"PATH": os.environ["PATH"], "HOME": str(home)}                 # no SISTEER_CONFIG: the default site directory
    units = _render(tmp_path, env, installer=clone / "scripts" / "registry" / "install_grading_host.sh")
    assert sorted(units) == RENDERED
    for name, text in units.items():
        assert not re.search(r"@[A-Z][A-Z_]*@", text), name
        assert str(tmp_path) not in text and str(home) not in text, name
        firsts = set(re.findall(r"%h/([^/\s'\"]+)", text))
        assert firsts <= {"code", "registry", "venvs", ".ssh", ".local"}, (name, firsts)
        assert set(re.findall(r"%h/code/([^/\s'\"]+)", text)) <= {"my-clone"}, name
        assert_no_tokens(text, name)
        assert "eval " not in text, name
        if name.endswith(".service"):
            assert "Environment=PYTHONPATH=%h/code/my-clone/src" in text and "Environment=REGISTRY_ROOT=%h/registry" in text, name
    assert _exec_start(units["registry-harvest.service"]) == "%h/venvs/my-clone/bin/python -m sisteer.registry.cli ingest --pull --honor-maintenance"
    assert _exec_start(units["registry-store-mirror.service"]).endswith(f" stores mirror --dest {dest}")
    assert _exec_start(units["registry-paper-rebuild.service"]) == "%h/code/my-clone/scripts/registry/paper_rebuild.sh"
    assert _exec_start(units[GATE]) == "/bin/sh -c 'exec /bin/bash %h/code/my-clone/scripts/registry/landing/baseline_refresh.sh'"
    assert clone.is_dir() and not (home / "registry").exists()            # --units-only: no layout


def test_the_shipped_example_site_resolves_to_a_destination_the_shipped_store_configuration_declares(tmp_path):
    """The shipped example (reproduce/site.example, whose blocks are generic) rendered by this checkout's installer against this
    checkout's own store configuration (no $REGISTRY_CLUSTER_READS): the installer's check passes, and the destination both
    store-mirror units name is one configs/registry/cluster_reads.yaml declares, read here with PyYAML as `stores` reads it."""
    site = tmp_path / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_REGISTRY_ROOT={tmp_path / 'registry'}\nSISTEER_PYTHON={VENV_PY}\n")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "SISTEER_CONFIG": str(site / "config.env")}
    units = _render(tmp_path, env)
    dest = _expected_dest(site)
    declared = set((yaml.safe_load(STORES.read_text(encoding="utf-8")) or {}).get("destinations") or {})
    assert dest in declared, (dest, sorted(declared))
    assert _exec_start(units["registry-store-mirror.service"]).endswith(f" stores mirror --dest {dest}")
    assert _exec_start(units["registry-store-mirror-full.service"]).endswith(f" stores mirror --dest {dest} --full --lock-wait-s 3600")


def test_every_rendered_unit_is_its_template_with_exactly_the_values_substituted(tmp_path):
    """Comments, spacing and the trailing newline are the template's; only the five values change, and every one is filled."""
    env = _site_env(tmp_path)
    env["REGISTRY_PY"] = VENV_PY
    units = _render(tmp_path, env)
    seen = set()
    for t in TEMPLATES:
        src = (UNITS / t).read_text()
        seen |= set(re.findall(r"@[A-Z][A-Z_]*@", src))
        expect = (src.replace("@REPO@", str(REPO)).replace("@PYTHON@", VENV_PY).replace("@REGISTRY_ROOT@", str(tmp_path / "registry"))
                  .replace("@STORE_DEST@", DEFAULT_DEST).replace("@UNIT_ENV_HOOK@", ""))
        assert units[t[: -len(".in")]] == expect, t
    assert seen == set(VALUES), seen          # the templates name exactly the values the installer fills


@pytest.mark.parametrize("substrate, extra, dest", [
    (NAMED_SUBSTRATE, "", NAMED_SUBSTRATE),                            # a cluster's own registry label is its destination
    ("slurm", "", "alpha"),                                            # the generic substrate names no one cluster: the block's label
    ("slurm", "CLUSTER_STORE_DEST=store_one", "store_one"),            # an explicit destination wins over both
    (NAMED_SUBSTRATE, "CLUSTER_STORE_DEST=store_one", "store_one"),
])
def test_store_dest_is_the_explicit_destination_else_the_registry_label_else_the_generic_blocks_label(tmp_path, substrate, extra, dest):
    """Site labels `alpha` / `beta`, the store configuration declaring the resolved destination: both store-mirror units name it, and
    a named registry label lets no site label into any unit."""
    env = _site_env(tmp_path, dests=(dest,), clusters=("alpha", "beta"), substrate=substrate, extra_cluster=extra)
    units = _render(tmp_path, env)
    assert _exec_start(units["registry-store-mirror.service"]).endswith(f" stores mirror --dest {dest}")
    assert _exec_start(units["registry-store-mirror-full.service"]).endswith(f" stores mirror --dest {dest} --full --lock-wait-s 3600")
    if substrate != "slurm":
        for name, text in units.items():
            assert not re.search(r"\b(alpha|beta)\b", text), name


def test_a_store_configuration_that_declares_no_destinations_is_not_checked_and_says_so(tmp_path):
    """A store configuration with no destination at all has no name to check against: the units render, and the installer says the
    destination was not checked (the store commands themselves refuse such a file when they run)."""
    env = _site_env(tmp_path, dests=())
    r = _install("--units-only", "--no-systemctl", "--unit-dir", str(tmp_path / "rendered"), env=env)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert f"# store destination {DEFAULT_DEST} not checked: {env['REGISTRY_CLUSTER_READS']} declares no destinations" in r.stdout, r.stdout
    assert _exec_start((tmp_path / "rendered" / "registry-store-mirror.service").read_text()).endswith(f" stores mirror --dest {DEFAULT_DEST}")


def test_the_sites_hook_renders_into_the_gate_baseline_tick_alone_and_is_evaluated_there(tmp_path):
    """SISTEER_UNIT_ENV_HOOK names a command printing `export` lines: the gate-baseline tick's ExecStart evaluates it (under $HOME, as
    `%h/...`) before it runs the baseline script; no other unit names it. The evaluation is real: the rendered prefix exports the
    hook's variable into the command after it."""
    home = tmp_path / "home"
    hook = _exe(home / ".local" / "bin" / "secret-env", '#!/bin/sh\necho "export PROBE_VALUE=from-the-hook"\n')
    env = _site_env(tmp_path, extra_config=f"SISTEER_UNIT_ENV_HOOK={hook}")
    units = _render(tmp_path, env)
    want = f"/bin/sh -c 'eval \"`%h/.local/bin/secret-env`\"; exec /bin/bash {REPO}/scripts/registry/landing/baseline_refresh.sh'"
    assert _exec_start(units[GATE]) == want
    assert [n for n, t in units.items() if "secret-env" in t] == [GATE]
    prefix = _exec_start(units[GATE]).split("'", 1)[1].split("exec ", 1)[0].replace("%h", str(home))
    r = subprocess.run(["/bin/sh", "-c", prefix + 'echo "probe=$PROBE_VALUE"'], capture_output=True, text=True, env={"PATH": os.environ["PATH"]})
    assert r.stdout.strip() == "probe=from-the-hook", r.stdout + r.stderr


def _installed_gate_unit(home: Path) -> Path:
    """The unit directory under `home` holding one gate-baseline unit as an earlier install left it (INSTALLED_GATE_EXEC)."""
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (units / GATE).write_text(f"[Unit]\nDescription=gate baseline\n\n[Service]\nType=oneshot\nExecStart={INSTALLED_GATE_EXEC}\n")
    return units


def test_an_installed_unit_that_evaluates_a_hook_is_not_replaced_by_one_that_evaluates_none(tmp_path):
    """The unit directory holds a gate-baseline unit whose ExecStart evaluates an environment hook, and the site names no
    SISTEER_UNIT_ENV_HOOK: the install refuses by name, naming the hook it would drop, and writes nothing (the installed unit is byte
    for byte what it was, and no other unit lands beside it)."""
    units = _installed_gate_unit(tmp_path / "home")
    before = (units / GATE).read_bytes()
    env = _site_env(tmp_path)
    env["REGISTRY_PY"] = VENV_PY
    r = _install("--units-only", "--no-systemctl", "--unit-dir", str(units), env=env)
    assert r.returncode == 2, r.stdout[-1500:] + r.stderr[-1500:]
    assert "REFUSED: install:" in r.stderr and "SISTEER_UNIT_ENV_HOOK" in r.stderr and "%h/.local/bin/secret-env" in r.stderr, r.stderr
    assert (units / GATE).read_bytes() == before and sorted(p.name for p in units.iterdir()) == [GATE]


def test_with_the_sites_hook_the_rendered_gate_baseline_exec_start_is_the_installed_ones(tmp_path):
    """The same installed unit, a clone at ~/code/my-clone, and the site naming the hook that unit evaluates: the install replaces the
    unit, and the effective ExecStart it installs is the installed one byte for byte, so the tick still evaluates the hook before it
    reads the git origin."""
    home = tmp_path / "home"
    clone = _synthetic_clone(home, "my-clone")
    hook = _exe(home / ".local" / "bin" / "secret-env", '#!/bin/sh\necho "export PROBE_VALUE=from-the-hook"\n')
    units = _installed_gate_unit(home)
    env = _site_env(tmp_path, extra_config=f"SISTEER_UNIT_ENV_HOOK={hook}")
    env["REGISTRY_PY"] = VENV_PY
    r = _install("--units-only", "--no-systemctl", "--unit-dir", str(units), env=env,
                 installer=clone / "scripts" / "registry" / "install_grading_host.sh")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    installed = (units / GATE).read_text()
    assert "Description=gate baseline" not in installed and not re.search(r"@[A-Z][A-Z_]*@", installed)   # the rendered template replaced it
    assert _exec_start(installed) == INSTALLED_GATE_EXEC


def test_a_value_in_the_environment_wins_over_the_site_configs(tmp_path):
    env = _site_env(tmp_path)
    env["SISTEER_REGISTRY_ROOT"] = str(tmp_path / "home" / "elsewhere")
    units = _render(tmp_path, env)
    assert "Environment=REGISTRY_ROOT=%h/elsewhere" in units["registry-harvest.service"]


def _no_writes(tmp_path: Path, r: subprocess.CompletedProcess, needle: str):
    assert r.returncode == 2, r.stdout[-1500:] + r.stderr[-1500:]
    assert "REFUSED: install:" in r.stderr and needle in r.stderr, r.stderr[-1500:]
    assert not (tmp_path / "units").exists() and not (tmp_path / "registry").exists(), "a refusal writes nothing"


def _refuse_env(tmp_path: Path, case: str) -> tuple[dict, str]:
    """(the installer's environment, the text its refusal must carry). The registry-root cases set the root through the site's key,
    so the run passes no --root (which would win over it)."""
    if case == "no site config":
        return {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}, "no site config at"
    env = _site_env(tmp_path)
    cfg = Path(env["SISTEER_CONFIG"])
    if case == "no primary cluster":
        cfg.write_text(re.sub(r"(?m)^SISTEER_PRIMARY_CLUSTER=.*$", "", cfg.read_text()))
        return env, "sets no SISTEER_PRIMARY_CLUSTER"
    if case == "a broken cluster block":
        block = cfg.parent / "clusters" / "cluster_a.env"
        block.write_text(re.sub(r"(?m)^CLUSTER_SUBSTRATE=.*$", "CLUSTER_SUBSTRATE=not-a-label", block.read_text()))
        return env, "cluster blocks did not load"
    if case == "an undeclared store destination":
        _stores(Path(env["REGISTRY_CLUSTER_READS"]), "other_dest")
        return env, f"the store destination '{DEFAULT_DEST}' is not declared in"
    if case == "a named store configuration that does not exist":
        env["REGISTRY_CLUSTER_READS"] = str(tmp_path / "nowhere" / "cluster_reads.yaml")
        return env, "could not be read"
    if case == "a path a unit cannot carry":
        env["SISTEER_REGISTRY_ROOT"] = str(tmp_path / "with space" / "registry")
        return env, "SISTEER_REGISTRY_ROOT="
    if case == "a relative path":
        env["SISTEER_REGISTRY_ROOT"] = "relative/registry"
        return env, "is not an absolute path"
    if case == "no interpreter":
        env["REGISTRY_PY"] = str(tmp_path / "nowhere" / "python")
        return env, "no interpreter at"
    if case == "a hook that is not executable":
        (tmp_path / "hook").write_text("echo nothing\n")
        env["SISTEER_UNIT_ENV_HOOK"] = str(tmp_path / "hook")
        return env, "SISTEER_UNIT_ENV_HOOK="
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["no site config", "no primary cluster", "a broken cluster block", "an undeclared store destination",
                                  "a named store configuration that does not exist", "a path a unit cannot carry", "a relative path",
                                  "no interpreter", "a hook that is not executable"])
def test_every_refusal_names_what_is_missing_and_writes_nothing(tmp_path, case):
    env, needle = _refuse_env(tmp_path, case)
    root = [] if "SISTEER_REGISTRY_ROOT" in env else ["--root", str(tmp_path / "registry")]
    r = _install("--no-systemctl", *root, "--unit-dir", str(tmp_path / "units"), env=env)
    _no_writes(tmp_path, r, needle)


def test_a_template_naming_a_value_the_installer_does_not_fill_installs_nothing(tmp_path):
    """A clone carrying one more template with an unknown @VALUE@: refused by name before the layout and before any unit lands,
    whichever template comes first in the glob."""
    home = tmp_path / "home"
    clone = _synthetic_clone(home, "clone")
    (clone / "systemd" / "user" / "registry-zz-extra.service.in").write_text("[Service]\nExecStart=@NOT_A_VALUE@\n")
    env = _site_env(tmp_path)
    env["REGISTRY_PY"] = VENV_PY
    r = _install("--no-systemctl", "--root", str(tmp_path / "registry"), "--unit-dir", str(tmp_path / "units"), env=env,
                 installer=clone / "scripts" / "registry" / "install_grading_host.sh")
    _no_writes(tmp_path, r, "@NOT_A_VALUE@")
