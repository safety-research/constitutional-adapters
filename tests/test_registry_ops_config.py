"""The configuration layer: ``<root>/config.toml`` defaults, the file ``install_grading_host.sh`` writes round-trips
through ``tomllib``, every operational destination derives from it (mirrors, pulls, snapshot destinations,
the remote PYTHONPATH), ``harvest.load_config`` merges it with an older ``config.json``, the CLI reads and
writes it, and the default file carries no secret-shaped key.

The Modal destination is GONE and the loader is fail-closed about it. ``config.load`` refuses, by name, a file that still
carries the removed table; ``config.check_destinations`` (through ``harvest.load_config``) refuses a ``kind: modal`` entry
anywhere in a legacy ``config.json``'s lists; the CLI reports either as ``REFUSED`` on stderr with exit 1 and no traceback;
the defaults, the rendered default file and the derived destination lists carry no trace of it. ``registry config --check``
is the cutover's validation: it loads BOTH files through the loader and exits 0 only on a clean configuration (the clean
case, each refusal and an unparseable file are pinned here).
"""
from __future__ import annotations

import json
import os
import re
import tomllib

import pytest

from sisteer.registry import cli
from sisteer.registry import config as CFG
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M

_TABLE = CFG.REMOVED_TABLE   # the removed table's name, kept out of every literal so this module passes the no-Modal scan it is part of


def test_defaults_when_file_absent(tmp_path):
    root = str(tmp_path / "reg")
    cfg = CFG.load(root)
    assert cfg["_present"] is False and cfg["clusters"] == []
    assert cfg["registry"]["root"] == os.path.abspath(root)
    assert cfg["paper"]["checkout"].startswith("/") and "~" not in cfg["paper"]["checkout"]
    assert cfg["snapshot"]["in_flight_every_s"] == 300 and cfg["snapshot"]["full_every_s"] == 86400
    # the status render's budgets (registry-slurm-fast-windows Phase 1, S4): per section 20 s, the whole render 120 s, a heavy member 15 min
    assert (cfg["status"]["section_budget_s"], cfg["status"]["render_budget_s"], cfg["status"]["heavy_budget_s"]) == (20, 120, 900)
    # no clusters, no file -> no destination of any kind: the Modal defaults are gone with the destination (C10)
    assert _TABLE not in cfg and _TABLE not in CFG.DEFAULTS and not any(section == _TABLE for section, _ in CFG.PATH_KEYS)
    assert CFG.mirror_destinations(cfg) == [] and CFG.snapshot_destinations(cfg) == [] and CFG.pull_sources(cfg) == []


def test_render_default_roundtrips_and_derives_every_destination(tmp_path):
    root = str(tmp_path / "reg")
    os.makedirs(root)
    text = CFG.render_default(root, user="testuser", paper_checkout=str(tmp_path / "paper"))
    data = tomllib.loads(text)   # must be valid TOML
    assert [c["name"] for c in data["clusters"]] == ["cluster_a", "cluster_b"]
    assert data["clusters"][0]["shared"] == "/scratch/example/testuser" and data["clusters"][1]["shared"] == "/data/example/testuser"
    assert _TABLE not in data and set(data) == {"registry", "paper", "status", "snapshot", "harvest", "clusters", "repo"}
    with open(CFG.config_path(root), "w") as fh:
        fh.write(text)
    cfg = CFG.load(root)
    assert cfg["_present"] and cfg["registry"]["user"] == "testuser"
    assert cfg["paper"]["checkout"] == str(tmp_path / "paper")
    assert data["status"] == {"page_dir": "~/sisteer-pages/status", "section_budget_s": 20, "render_budget_s": 120, "heavy_budget_s": 900}
    assert cfg["status"]["render_budget_s"] == 120 and cfg["status"]["page_dir"].startswith("/")
    mirrors = CFG.mirror_destinations(cfg)
    assert [m["name"] for m in mirrors] == ["cluster_a", "cluster_b"], "the record log's mirrors are the clusters (2026-09-28 ruling)"
    assert mirrors[0] == {"name": "cluster_a", "kind": "ssh", "host": "cluster-a-login", "path": "/scratch/example/testuser/registry/log",
                          "remote_src": "/scratch/example/testuser/repos/constitutional-adapters/src",
                          # the full-verify Slurm step's shape and the per-cycle sample (2026-09-28)
                          "qos": "normal", "slurm": True, "partition": "general", "verify_cpus": 4, "verify_mem": "16G", "verify_sample": 2000}
    assert mirrors[1]["path"] == "/data/example/testuser/registry/log" and mirrors[1]["qos"] == "normal"
    assert all(m["kind"] == "ssh" for m in mirrors)
    pulls = CFG.pull_sources(cfg)
    assert pulls == [{"name": "cluster_a", "host": "cluster-a-login", "path": "/scratch/example/testuser/registry/records"},
                     {"name": "cluster_b", "host": "cluster-b-login", "path": "/data/example/testuser/registry/records"}]
    snaps = CFG.snapshot_destinations(cfg)
    assert [s["path"] for s in snaps] == ["/scratch/example/testuser/registry/snapshot", "/data/example/testuser/registry/snapshot"]
    assert all(s["kind"] == "ssh" for s in snaps)
    assert CFG.get(cfg, "clusters.1.host") == "cluster-b-login"
    with pytest.raises(KeyError):
        CFG.get(cfg, "paper.nope")
    # the rendered file and the derived json carry no trace of the removed destination (checked on the parsed tables and the
    # table headers, never on the raw text: the tmp root's path may spell the name of the account the tests run under)
    assert not any(line.strip().startswith("[" + _TABLE) for line in text.splitlines())
    assert _TABLE not in {k for d_ in CFG.as_json_config(cfg).values() for e in d_ for k in (e.get("kind"), e.get("name"))}
    assert "render_default() got an unexpected keyword" in str(pytest.raises(TypeError, CFG.render_default, root, modal_enabled=False).value)


# --------------------------------------------------------------------------------------------
# the refusals: the removed table, the removed destination kind -- by name, fail-closed
# --------------------------------------------------------------------------------------------


def _write_clusters(root: str, tail: str = "") -> str:
    os.makedirs(root, exist_ok=True)
    text = ('[registry]\nuser = "u"\n\n[[clusters]]\nname = "cluster_a"\nhost = "cluster-a-login"\nshared = "/scratch/example/{user}"\n\n'
            '[[clusters]]\nname = "cluster_b"\nhost = "cluster-b-login"\nshared = "/data/example/{user}"\n\n' + tail)
    with open(CFG.config_path(root), "w") as fh:
        fh.write(text)
    return text


def _removed_table(lines: str = 'enabled = true\nvolume = "twosi-fits"\nlog_path = "/registry/log"\n') -> str:
    """The live file's shape before the cutover: the removed table with the keys it carried."""
    return "[" + _TABLE + "]\n" + lines


def test_load_refuses_a_file_that_still_carries_the_removed_table_by_name(tmp_path):
    """The live `~/registry/config.toml` as it stands before the cutover: `config.load` raises ConfigError naming the file, the
    reason, the table's keys and the converter -- nothing derives from such a file (no clusters, no destinations), and every
    reader of the configuration (the harvester's loader, the mirrors' destinations) inherits the refusal."""
    root = str(tmp_path / "reg")
    _write_clusters(root, _removed_table())
    with pytest.raises(CFG.ConfigError) as ex:
        CFG.load(root)
    msg = str(ex.value)
    assert msg.startswith(CFG.config_path(root) + ": the Modal destination was removed (registry-slurm-only S3)")
    assert "`" + _TABLE + "` table" in msg and "enabled, log_path, volume" in msg
    assert "scripts/registry/landing/convert_config.py --root " + root in msg and "registry config --check" in msg
    for reader in (lambda: H.load_config(root), lambda: M.destinations(root)):
        with pytest.raises(CFG.ConfigError):
            reader()
    # an EMPTY table is refused too (presence, not content, is the signal), and the explicit path argument is honored
    other = str(tmp_path / "other.toml")
    with open(other, "w") as fh:
        fh.write("[" + _TABLE + "]\n")
    with pytest.raises(CFG.ConfigError) as ex2:
        CFG.load(root, path=other)
    assert str(ex2.value).startswith(other + ":") and "(no keys)" in str(ex2.value)
    # the same file without the table loads and derives the cluster destinations
    _write_clusters(root)
    cfg = CFG.load(root)
    assert [m["name"] for m in CFG.mirror_destinations(cfg)] == ["cluster_a", "cluster_b"]
    assert [d["name"] for d in H.load_config(root)["mirrors"]] == ["cluster_a", "cluster_b"]


def test_a_legacy_config_json_naming_the_removed_kind_is_refused_by_name_in_every_list(tmp_path):
    """`config.json` overlays its lists verbatim, so it is the one place a `kind: modal` destination can still be spelled: the
    merged document is checked whenever that file is read -- `mirrors`, `snapshot_destinations` and `pulls` alike -- and the
    refusal names the file, the list, the index, the entry's name and its kind. A json without such an entry still overlays."""
    root = str(tmp_path / "reg")
    _write_clusters(root)
    legacy = {"name": "legacy", "kind": "modal", "volume": "v", "path": "/registry/log", "python": "/p", "app": "/a"}
    jp = os.path.join(root, "config.json")
    for key, extra in (("mirrors", []), ("snapshot_destinations", [{"name": "cluster_a", "kind": "local", "path": "/x"}]), ("pulls", [])):
        with open(jp, "w") as fh:
            json.dump({key: extra + [legacy]}, fh)
        with pytest.raises(CFG.ConfigError) as ex:
            H.load_config(root)
        msg = str(ex.value)
        assert msg.startswith(jp + ": the Modal destination was removed (registry-slurm-only S3): " + f"{key}[{len(extra)}] (name 'legacy') has kind 'modal'"), msg
        assert "remove the entry" in msg
    with pytest.raises(CFG.ConfigError):
        M.destinations(root)
    # case-insensitive on the kind; a non-dict entry or a non-list value is left to the readers (never a crash here)
    with open(jp, "w") as fh:
        json.dump({"mirrors": ["not a dict", {"name": "shouty", "kind": "MODAL", "path": "/x"}]}, fh)
    with pytest.raises(CFG.ConfigError) as ex3:
        H.load_config(root)
    assert "mirrors[1] (name 'shouty') has kind 'modal'" in str(ex3.value)
    CFG.check_destinations({"mirrors": "not a list", "pulls": None}, "synthetic")      # nothing to refuse
    # the clean legacy shapes still overlay verbatim
    with open(jp, "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}]}, fh)
    merged = H.load_config(root)
    assert merged["mirrors"] == [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}]   # json wins for its keys
    assert [p["name"] for p in merged["pulls"]] == ["cluster_a", "cluster_b"]                          # toml-derived keys stay
    assert [s["name"] for s in merged["snapshot_destinations"]] == ["cluster_a", "cluster_b"]
    assert [d.name for d in M.destinations(root)] == ["cluster_a"]
    with open(jp, "w") as fh:
        json.dump({"pulls": []}, fh)
    merged = H.load_config(root)
    assert [d["name"] for d in merged["mirrors"]] == ["cluster_a", "cluster_b"] and merged["pulls"] == []


def test_cli_config_reports_the_refusal_by_name_and_exits_1_never_a_traceback(tmp_path, monkeypatch, capsys):
    """`registry config`, `config --json`, `config KEY` on a file that still carries the removed table: `config: REFUSED: <reason>`
    on stderr, exit 1; the same file through a command that reads the configuration on its way (`mirrors sync`): the dispatcher's
    `registry mirrors: configuration REFUSED: …` line, exit 1 -- no traceback on any path."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write_clusters(root, _removed_table())
    monkeypatch.setenv("REGISTRY_ROOT", root)
    for argv in (["config"], ["config", "--json"], ["config", "paper.checkout"]):
        assert cli.main(argv) == 1, argv
        out, err = capsys.readouterr()
        assert err.startswith("config: REFUSED: " + CFG.config_path(root)) and "Traceback" not in err and out == "", (argv, err)
    assert cli.main(["mirrors", "sync"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("registry mirrors: configuration REFUSED: " + CFG.config_path(root)) and "Traceback" not in err
    # --write-default never reads the file: it refuses to overwrite without --force, as before, whatever the file carries
    assert cli.main(["config", "--write-default"]) == 0 and "exists" in capsys.readouterr().err
    # the retired flag is gone with the destination
    with pytest.raises(SystemExit) as ex:
        cli.main(["config", "--write-default", "--no-modal"])
    assert ex.value.code == 2


# --------------------------------------------------------------------------------------------
# `registry config --check`: the cutover window's validation after convert_config.py
# --------------------------------------------------------------------------------------------


def _check(root: str, monkeypatch, capsys):
    monkeypatch.setenv("REGISTRY_ROOT", root)
    rc = cli.main(["config", "--check"])
    out, err = capsys.readouterr()
    return rc, out, err


def test_config_check_passes_a_clean_configuration_and_names_what_it_read(tmp_path, monkeypatch, capsys):
    """The clean cases: the rendered default (what `--write-default` and the installer lay down), the live file's shape AFTER the
    conversion (clusters, no table), a clean legacy config.json beside it, and no file at all (defaults). Exit 0, one `OK` line
    naming the files' presence, the destination counts and kinds."""
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["config", "--write-default"]) == 0
    capsys.readouterr()
    rc, out, err = _check(root, monkeypatch, capsys)
    assert rc == 0 and err == "", err
    assert out == (f"config --check: OK {CFG.config_path(root)} (present); config.json absent; mirrors 2, pulls 2, snapshot destinations 2; "
                   f"destination kinds ssh; no `{_TABLE}` table, no `{_TABLE}` destination\n"), out
    # the converted live shape: the same tables minus the removed one
    _write_clusters(root)
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}]}, fh)
    rc, out, err = _check(root, monkeypatch, capsys)
    assert rc == 0 and "config.json present; mirrors 1, pulls 2, snapshot destinations 2; destination kinds local ssh;" in out, out
    # no file at all: the defaults are a clean configuration
    rc, out, err = _check(str(tmp_path / "absent"), monkeypatch, capsys)
    assert rc == 0 and "(absent; defaults apply); config.json absent; mirrors 0, pulls 0, snapshot destinations 0; destination kinds none;" in out, out
    # the last-run stamp names the command (every CLI run that resolves a root stamps it)
    assert json.load(open(os.path.join(root, "state", "last_run", "config.json")))["rc"] == 0


def test_config_check_refuses_the_removed_table_the_removed_kind_in_config_json_and_an_unparseable_file(tmp_path, monkeypatch, capsys):
    """Each refusal the window's abort path rests on (R12's injected failure is one of them): the TOML still carrying the table
    (the live file BEFORE convert_config.py ran, exit 1, the reason names the converter), a config.json with a `kind: modal` entry
    (exit 1, the list and index named), a TOML that does not parse (exit 1, named), a config.json that does not parse (exit 1,
    named). Nothing on stdout; `REFUSED` on stderr; never a traceback."""
    root = str(tmp_path / "reg")
    _write_clusters(root, _removed_table())
    rc, out, err = _check(root, monkeypatch, capsys)
    assert rc == 1 and out == "" and "Traceback" not in err
    assert err.startswith("config --check: REFUSED: " + CFG.config_path(root) + ": the Modal destination was removed (registry-slurm-only S3)")
    assert "convert_config.py --root " + root in err
    # converted (the table gone) but a legacy json naming the kind: refused by name, with the list and index
    _write_clusters(root)
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"snapshot_destinations": [{"name": "cluster_a", "kind": "ssh", "host": "h", "path": "/p"},
                                             {"name": "legacy", "kind": "modal", "volume": "v", "path": "/registry/snapshot"}]}, fh)
    rc, out, err = _check(root, monkeypatch, capsys)
    assert rc == 1 and out == "" and "snapshot_destinations[1] (name 'legacy') has kind 'modal'" in err and "Traceback" not in err
    os.remove(os.path.join(root, "config.json"))
    rc, out, err = _check(root, monkeypatch, capsys)
    assert rc == 0, err
    # a TOML that does not parse
    with open(CFG.config_path(root), "w") as fh:
        fh.write("this is = not [toml\n")
    rc, out, err = _check(root, monkeypatch, capsys)
    assert rc == 1 and out == "" and err.startswith("config --check: REFUSED: " + CFG.config_path(root) + " does not parse as TOML:")
    # a config.json that does not parse (the TOML clean again)
    _write_clusters(root)
    with open(os.path.join(root, "config.json"), "w") as fh:
        fh.write("{not json")
    rc, out, err = _check(root, monkeypatch, capsys)
    assert rc == 1 and out == "" and err.startswith("config --check: REFUSED: " + os.path.join(root, "config.json") + ":") and "Traceback" not in err


def test_config_check_as_the_window_runs_it_through_a_subprocess_on_the_converted_live_shape(tmp_path):
    """The exact invocation window.sh's `-- config convert` stage makes under the worktree's code: `python -m sisteer.registry.cli
    --root ROOT config --check` -- rc 0 on the converted file, rc 1 with the named refusal on the original (R12 part (a), which was
    pending C10)."""
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "root"
    root.mkdir()
    original = ('# the operating configuration\n\n[registry]\nroot = "~/registry"\nuser = "tester"\n\n[paper]\ncheckout = "~/repo-paper"\n\n'
                '[status]\npage_dir = "~/pages/status"\n\n[snapshot]\nin_flight_every_s = 300\n\n[harvest]\npull_clusters = true\n\n'
                '[[clusters]]\nname = "cluster_a"\nhost = "cluster-a-login"\nshared = "/scratch/example/tester"\n\n[[clusters]]\nname = "cluster_b"\n'
                'host = "cluster-b-login"\nshared = "/data/example/tester"\n\n' + _removed_table('enabled = true\nvolume = "twosi-fits"\n') +
                '\n[repo]\ncheckout = "~/repo"\n')
    (root / "config.toml").write_text(original)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "PYTHONPATH": str(repo / "src")}
    argv = [sys.executable, "-m", "sisteer.registry.cli", "--root", str(root), "config", "--check"]
    r = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert r.returncode == 1 and r.stdout == "" and "REFUSED" in r.stderr and "Traceback" not in r.stderr, r.stdout + r.stderr
    # the converter the window runs first, then the check passes
    conv = subprocess.run([sys.executable, str(repo / "scripts" / "registry" / "landing" / "convert_config.py"), "--root", str(root)],
                          capture_output=True, text=True)
    assert conv.returncode == 0 and "CONVERTED=1" in conv.stdout, conv.stdout + conv.stderr
    r2 = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert r2.returncode == 0 and r2.stdout.startswith(f"config --check: OK {root}/config.toml (present); config.json absent; mirrors 2, pulls 2,"), r2.stdout + r2.stderr
    assert re.search(r"config\.toml\.bak-\d{8}T\d{6}Z", conv.stdout)


# --------------------------------------------------------------------------------------------
# the rest of the configuration layer (unchanged by C10 beyond the removed table)
# --------------------------------------------------------------------------------------------


def test_cli_mirrors_sync_without_dest_fans_out_to_the_clusters_only(tmp_path, monkeypatch):
    """`registry mirrors sync` with no `--dest` on the live file's shape reaches the cluster destinations and nothing else."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    H.ingest_once(root, workers=0, min_free=0)
    _write_clusters(root)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    captured = {}

    def fake_sync_all(root_, seq, dests, deadline=None, **kw):
        captured["dests"] = [(d.name, d.kind) for d in dests]
        return [{"destination": d.name, "switched": True, "error": None, "stopped": None} for d in dests]
    monkeypatch.setattr(M, "sync_all", fake_sync_all)
    assert cli.main(["mirrors", "sync"]) == 0
    assert captured["dests"] == [("cluster_a", "ssh"), ("cluster_b", "ssh")]
    assert cli.main(["mirrors", "sync", "--dest", "cluster_a"]) == 0 and captured["dests"] == [("cluster_a", "ssh")]


def test_harvest_load_config_merges_toml_then_json(tmp_path):
    root = str(tmp_path / "reg")
    os.makedirs(root)
    with open(CFG.config_path(root), "w") as fh:
        fh.write(CFG.render_default(root, user="u"))
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}]}, fh)
    merged = H.load_config(root)
    assert merged["mirrors"] == [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}]   # json wins for its keys
    assert [p["name"] for p in merged["pulls"]] == ["cluster_a", "cluster_b"]                          # toml-derived keys stay
    assert [s["name"] for s in merged["snapshot_destinations"]] == ["cluster_a", "cluster_b"]


def test_cli_config_write_default_read_key_and_json(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["config", "--write-default", "--paper-checkout", str(tmp_path / "pf")]) == 0
    path = capsys.readouterr().out.strip()
    assert path == CFG.config_path(root) and os.path.exists(path)
    assert cli.main(["config", "--write-default"]) == 0
    assert "exists" in capsys.readouterr().err
    assert cli.main(["config", "paper.checkout"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "pf")
    assert cli.main(["config", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["_present"] is True and len(data["clusters"]) == 2 and _TABLE not in data
    assert cli.main(["config", "no.such.key"]) == 1
    assert cli.main(["config", _TABLE + ".enabled"]) == 1 and "no such key" in capsys.readouterr().err
    assert cli.main(["config"]) == 0
    dump = capsys.readouterr().out
    assert "  harvest.cadence_s = 1200" in dump and _TABLE + "." not in dump


def test_default_config_carries_no_secret_shaped_key(tmp_path):
    text = CFG.render_default(str(tmp_path), user="u")
    assert not re.search(r"(?i)(api[_-]?key|secret|password|token)\s*=", text)
    assert "://" not in text and "MODAL" not in text           # no secret reference URI, no Modal token name


def test_status_budgets_are_read_by_the_render_and_a_partial_block_keeps_the_other_defaults(tmp_path):
    """`sisteer.registry.status.status_config` reads the three keys; a file naming one keeps the defaults for the rest; the
    `registry config` dump shows them."""
    from sisteer.registry import status
    root = str(tmp_path / "reg")
    os.makedirs(root)
    assert status.status_config(root) == {"section_budget_s": 20.0, "render_budget_s": 120.0, "heavy_budget_s": 900.0}
    with open(CFG.config_path(root), "w") as fh:
        fh.write("[status]\nrender_budget_s = 60\n")
    assert status.status_config(root) == {"section_budget_s": 20.0, "render_budget_s": 60.0, "heavy_budget_s": 900.0}
    cfg = CFG.load(root)
    assert cfg["status"]["render_budget_s"] == 60 and cfg["status"]["section_budget_s"] == 20


def test_cli_config_shows_the_status_budgets(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["config", "--write-default"]) == 0
    capsys.readouterr()
    text = open(CFG.config_path(root)).read()
    assert "section_budget_s = 20" in text and "render_budget_s = 120" in text and "heavy_budget_s = 900" in text
    assert cli.main(["config", "status.heavy_budget_s"]) == 0 and capsys.readouterr().out.strip() == "900"
    assert cli.main(["config", "--json"]) == 0 and json.loads(capsys.readouterr().out)["status"]["section_budget_s"] == 20


#: the registry's own session variable, then the one the session manager sets for each session it starts
OWN_SESSION_VAR, MANAGER_SESSION_VAR = "SISTEER_SESSION_NAME", "ROMP_SESSION_NAME"


def test_the_harvester_names_its_session_from_the_environment(tmp_path, monkeypatch):
    """The harvester's own producer block names the session from the registry's own variable, else the session manager's
    (all a session it starts carries), else none; a variable set to the empty string names nothing."""
    for var in (OWN_SESSION_VAR, MANAGER_SESSION_VAR):
        monkeypatch.delenv(var, raising=False)
    assert H.harvester_producer(str(tmp_path))["session"] is None
    monkeypatch.setenv(MANAGER_SESSION_VAR, "synthetic-managed-session")
    assert H.harvester_producer(str(tmp_path))["session"] == "synthetic-managed-session"
    monkeypatch.setenv(OWN_SESSION_VAR, "synthetic-session")
    assert H.harvester_producer(str(tmp_path))["session"] == "synthetic-session"
    monkeypatch.setenv(OWN_SESSION_VAR, "")
    assert H.harvester_producer(str(tmp_path))["session"] == "synthetic-managed-session"


def test_the_harvesters_session_variables_are_the_registrys_own_then_the_session_managers():
    assert H.SESSION_ENV_VARS == (OWN_SESSION_VAR, MANAGER_SESSION_VAR)
