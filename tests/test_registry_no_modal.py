"""The no-Modal invariant, by OWNERSHIP and REACHABLE ENTRY POINTS, with
no allowlist: every file under ``src/sisteer/registry/**``, ``scripts/registry/**``, ``systemd/user/registry-*``,
``configs/registry/*.yaml`` and ``tests/test_registry_*.py``, plus every module reachable by import from ``sisteer.registry``, is
matched line by line against the Modal reach patterns -- an import of the package or one of its App / Function / Volume / Image / Cls
attributes, a launcher (the CLI's run and volume verbs, the Modal venv), the token variable, the registry's volume app, smoke producer
and transport modules, the retired gates launcher directory, the Modal-era suite module, and the Modal configuration table's header.
The pattern is COMPOSED from fragments (``_M`` and friends) so this module passes the scan it performs, and the same composition
lives in ``check_no_modal.sh`` (which land.sh runs after every merge) and ``operating_scripts_inventory.py`` (which classifies the
operating tree the cutover retires); one test pins the three spellings equal.

The Modal code is gone (the transport modules, the volume and smoke apps, the pull unit, the configuration
table and every launcher spelling in the registry's ownership), so the two live-tree scans run GREEN here with no allowlist: a hit
anywhere in the ownership or in a module the package imports fails the suite by name. (Before the removal they were ``xfail(strict=True)``
and printed the 159-hit inventory of the 22 files that were removed.) The scanner itself, the shell scan's modes, the inventory's
classification and the rehearsed retirement run on synthetic trees here.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LANDING = REPO / "scripts" / "registry" / "landing"


def _main_checkout(path: Path) -> str:
    """The main working tree of the repository holding ``path`` (a linked worktree resolves to the checkout it was added from), else
    this test's own checkout (no git, an exported tree)."""
    got = subprocess.run(["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
    common = got.stdout.strip()
    return common[:-len("/.git")] if got.returncode == 0 and common.endswith("/.git") else str(LANDING.resolve().parents[2])


#: the retirement tool's last fallback for the wrappers it writes: the MAIN checkout of the repository it runs from, never a linked
#: worktree (which is removed once its branch lands)
_CHECKOUT = _main_checkout(LANDING)
#: the wrapper lines that pin the fallback and the exec of the chain's landing step
_FALLBACK_LINE = f'R=${{R:-"{_CHECKOUT}"}};'
_LAND_EXEC = 'exec "$R/scripts/registry/landing/land.sh" "$@"\n'
_M, _U, _S, _V, _R, _G, _L, _C = "modal", "MODAL", "_", "volume", "registry", "gates/", "land/", "cpu"
_GL = _G + _L          # the retired launcher directory, composed (a literal would be a hit of this module's own scan)
PATTERN = (rf"import {_M}\b|from {_M}\b|{_M}\.(App|Function|Volume|Image|Cls)|{_M} run|{_M} {_V}|venvs/{_M}|{_U}{_S}TOKEN|"
           rf"{_R}{_S}{_V}|{_R}{_S}smoke|{_M}{_S}ops|{_V}{_S}ops|{_G}{_L}|{_C}{_S}suite|\[{_M}\]")
PATTERN_RE = re.compile(PATTERN)
#: the faces of the pattern, assembled at run time (a literal here would be a hit of this very scan)
FACES = [f"import {_M}", f"from {_M} import App", f"{_M}.App(", f"{_M}.Function", f"{_M}.Volume.from_name", f"{_M}.Image.debian_slim", f"{_M}.Cls",
         f"{_M} run x.py", f"{_M} {_V} ls", f"~/venvs/{_M}/bin/python", f"{_U}{_S}TOKEN_ID", f"scripts/{_M}/{_R}{_S}{_V}.py", f"{_R}{_S}smoke",
         f"from . import {_M}{_S}ops", f"{_V}{_S}ops.push", f"gates/" + "land/launch.sh", f"{_C}{_S}suite.py", f"[{_M}]"]
CLEAN = ["import os", "the Modal-era history is kept as prose", "registry-mirrors.service", f"SUITE_USER = '{_M}suiteuser'", f"modality", "volumes = []",
         f"no {_M} client is imported here", "gates/slurm/gate_slurm.sh", "a cpu-only node", "[registry]", "[[clusters]]", f"{_M}ity"]


def owned_files(repo: Path) -> list:
    out = []
    for sub in ("src/sisteer/registry", "scripts/registry"):
        d = repo / sub
        if d.is_dir():
            out += [p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for pattern, d in (("registry-*", "systemd/user"), ("*.yaml", "configs/registry"), ("test_registry_*.py", "tests")):
        dd = repo / d
        if dd.is_dir():
            out += [p for p in dd.glob(pattern) if p.is_file()]
    return sorted(set(out))


#: the measurement a FRESH interpreter runs: import sisteer.registry and every submodule, then print the repo-relative file of every
#: module in sys.modules under the repo. In-process, the measurement was contaminated by whatever the pytest session had already
#: imported (C10's first battery: sibling test modules in the same task had loaded launcher modules under scripts/ through their own
#: sys.path edits, and the scan blamed the registry for them), so the reach is measured in a subprocess that imports nothing else.
_REACH_PROBE = """
import importlib, json, pkgutil, sys
from pathlib import Path
import sisteer.registry as pkg
repo = Path(sys.argv[1]).resolve()
for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    try:
        importlib.import_module(m.name)
    except Exception:  # an import that fails still counts its own file through the ownership scan
        pass
files = set()
for mod in list(sys.modules.values()):
    f = getattr(mod, "__file__", None)
    if f and Path(f).resolve().is_relative_to(repo) and ".venv" not in Path(f).parts:
        files.add(str(Path(f).resolve()))
print(json.dumps(sorted(files)))
"""


def reachable_modules(repo: Path) -> list:
    """Every module file under the repo that importing sisteer.registry and its submodules brings into sys.modules -- measured in a
    fresh interpreter (PYTHONPATH=<repo>/src, cwd=<repo>), so nothing the calling pytest session loaded is counted."""
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    r = subprocess.run([sys.executable, "-c", _REACH_PROBE, str(repo)], capture_output=True, text=True, env=env, cwd=str(repo), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return sorted(Path(f) for f in json.loads(r.stdout.strip().splitlines()[-1]))


def hits(paths) -> list:
    out = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if PATTERN_RE.search(line):
                out.append((str(Path(p).relative_to(REPO)) if Path(p).is_relative_to(REPO) else str(p), n, line.strip()[:120]))
    return out


def test_known_bad_fixture_fires_on_every_face_and_not_on_clean_text():
    for face in FACES:
        assert PATTERN_RE.search(face), face
    for text in CLEAN:
        assert not PATTERN_RE.search(text), text


def test_the_three_scanners_compose_one_pattern():
    """The test, the shell scan and the inventory are one rule: the composed regex is byte-identical in all three."""
    shell = subprocess.run([str(LANDING / "check_no_modal.sh"), "--print-pattern"], capture_output=True, text=True)
    assert shell.returncode == 0 and shell.stdout.strip() == PATTERN, shell.stdout
    spec = importlib.util.spec_from_file_location("inventory_under_test", LANDING / "operating_scripts_inventory.py")
    inv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv)
    assert inv.PATTERN == PATTERN


def _synthetic_repo(tmp_path: Path, with_hits: bool) -> Path:
    repo = tmp_path / ("dirty" if with_hits else "clean")
    for d in ("src/sisteer/registry/adapters", "scripts/registry/landing", "systemd/user", "configs/registry", "tests", "scripts/other"):
        (repo / d).mkdir(parents=True)
    (repo / "src/sisteer/registry/__init__.py").write_text("")
    (repo / "src/sisteer/registry/harvest.py").write_text("import os\n\ndef ingest():\n    return 1\n")
    (repo / "scripts/registry/landing/land.sh").write_text("#!/bin/bash\necho land\n")
    (repo / "systemd/user/registry-harvest.service").write_text("[Service]\nExecStart=/bin/true\n")
    (repo / "systemd/user/other.service").write_text(f"[Service]\nExecStart={_M} run x\n")       # outside the scope: never scanned
    (repo / "configs/registry/cluster_reads.yaml").write_text("schema: cluster_reads/2\n")
    (repo / "tests/test_registry_probe.py").write_text("def test_x():\n    assert True\n")
    (repo / "tests/test_other.py").write_text(f"import {_M}\n")                                   # outside the scope: never scanned
    (repo / "scripts/other/tool.py").write_text(f"from {_M} import App\n")                        # outside the scope: never scanned
    if with_hits:
        (repo / "src/sisteer/registry/adapters/x.py").write_text(f"import {_M}\n\n\ndef f():\n    return {_M}.App\n")
        (repo / "scripts/registry/push.sh").write_text(f"#!/bin/bash\n~/venvs/{_M}/bin/python -m x\n")
        (repo / "systemd/user/registry-pull.service").write_text(f"[Service]\nEnvironment={_U}{_S}TOKEN_ID=x\n")
        (repo / "configs/registry/dest.yaml").write_text(f"kind: {_M}\napp: scripts/{_M}/{_R}{_S}{_V}.py\n")
        (repo / "tests/test_registry_x.py").write_text(f"def test_x():\n    assert '[{_M}]' in text\n")
    return repo


def _synthetic_operating_tree(tmp_path: Path) -> Path:
    root = tmp_path / "ops"
    for d in ("scripts", "gates/slurm", "gates/land"):
        (root / d).mkdir(parents=True)
    def exe(rel, text):
        p = root / rel
        p.write_text(text)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return p
    exe("scripts/land.sh", f"#!/bin/bash\n# the interim landing step\n~/venvs/{_M}/bin/python {_G}{_L}launch.sh\n")
    exe("scripts/serial-queue-1.sh", f"#!/bin/bash\nbash {root}/scripts/land.sh b t body > /tmp/land-b.out\n")
    exe("scripts/land-chain-9.sh", f"#!/bin/bash\nbash {root}/scripts/land.sh b t body\n")
    exe("scripts/gate-x.sh", f"#!/bin/bash\n{_M} run {_C}{_S}suite.py --stage\n")
    exe("scripts/ruling-x.sh", "#!/bin/bash\nregistry rule --apply\n")
    exe("scripts/post-landing-0021-checks.sh", "#!/bin/bash\nsqlite3 x\n")
    exe("scripts/helper.py", "print('hi')\n")
    (root / "scripts" / "known-red.env").write_text("KNOWN_RED=\n")
    (root / "scripts" / "pr-body-x.md").write_text(f"# body\nthe {_M} run of old\n")              # a document: listed, never scanned
    (root / "scripts" / "jobs.log").write_text(f"{_M} run x\n")                                   # data: never scanned
    exe("gates/slurm/gate_slurm.sh", "#!/bin/bash\n# the interim slurm gate\necho gate\n")
    exe("gates/slurm/gate_lib.py", f"import importlib\n# loads {_C}{_S}suite under a stub\n")
    exe("gates/slurm/verdict_local.py", "print('verdict')\n")
    exe("gates/slurm/wait_ref.sh", "#!/bin/bash\nsleep 1\n")
    exe(f"{_GL}launch.sh", f"#!/bin/bash\n{_M} run x\n")
    (root / "gates" / "slurm" / "baseline-abc.summary.json").write_text("{}")
    return root


def _inventory(root: Path, out: Path, proc_root: Path | None = None) -> dict:
    argv = [sys.executable, str(LANDING / "operating_scripts_inventory.py"), "inventory", "--root", str(root), "--out", str(out)]
    if proc_root is not None:
        argv += ["--proc-root", str(proc_root)]
    r = subprocess.run(argv, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.load(open(out))


def test_check_no_modal_reports_every_hit_in_scope_only_refuses_when_asked_and_requires_the_inventory_to_refuse(tmp_path):
    dirty = _synthetic_repo(tmp_path, with_hits=True)
    clean = _synthetic_repo(tmp_path, with_hits=False)
    sh = str(LANDING / "check_no_modal.sh")
    r = subprocess.run([sh, "--repo", str(dirty), "--mode", "report", "--json", str(tmp_path / "scan.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    hit_files = sorted({l[len("HIT "):].split(":")[0] for l in r.stdout.splitlines() if l.startswith("HIT ")})
    assert hit_files == sorted(str(dirty / p) for p in ("src/sisteer/registry/adapters/x.py", "scripts/registry/push.sh", "systemd/user/registry-pull.service",
                                                         "configs/registry/dest.yaml", "tests/test_registry_x.py")), hit_files
    assert f"HIT {dirty}/src/sisteer/registry/adapters/x.py:1: import {_M}" in r.stdout and f"HIT {dirty}/src/sisteer/registry/adapters/x.py:5:" in r.stdout
    assert "other.service" not in r.stdout and "test_other.py" not in r.stdout and "scripts/other/tool.py" not in r.stdout, "outside the ownership: never scanned"
    assert re.search(r"^no-modal scan: 6 hit\(s\) in 5 file\(s\); repo .*/dirty \(11 files scanned\); operating executables 0 scanned, 0 documents listed, retired excluded; mode=report$", r.stdout, re.M), r.stdout
    scan = json.load(open(tmp_path / "scan.json"))
    assert scan["n_hits"] == 6 and scan["n_files"] == 5 and scan["mode"] == "report" and scan["repo_files_scanned"] == 11 and scan["inventory"] is None
    # refuse needs the inventory; with one, the operating executables are scanned (documents listed) and any hit exits 1
    r = subprocess.run([sh, "--repo", str(dirty), "--mode", "refuse"], capture_output=True, text=True)
    assert r.returncode == 2 and "refuse mode requires --inventory" in r.stderr
    ops = _synthetic_operating_tree(tmp_path)
    inv = _inventory(ops, tmp_path / "inv.json", proc_root=tmp_path / "emptyproc")
    r = subprocess.run([sh, "--repo", str(clean), "--mode", "refuse", "--inventory", str(tmp_path / "inv.json")], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert f"HIT {ops}/scripts/land.sh:3:" in r.stdout and f"HIT {ops}/{_GL}launch.sh:2:" in r.stdout and "pr-body-x.md" not in r.stdout and "jobs.log" not in r.stdout
    assert re.search(r"^no-modal scan: 4 hit\(s\) in 4 file\(s\); repo .*/clean \(6 files scanned\); operating executables 12 scanned, 1 documents listed, retired excluded; mode=refuse$", r.stdout, re.M), r.stdout
    # a clean repo and a retired operating tree: zero hits, exit 0 under refuse
    r = subprocess.run([sh, "--repo", str(clean), "--mode", "report"], capture_output=True, text=True)
    assert r.returncode == 0 and "HIT " not in r.stdout and "no-modal scan: 0 hit(s) in 0 file(s)" in r.stdout
    r = subprocess.run([sh, "--repo", str(clean), "--mode", "nope"], capture_output=True, text=True)
    assert r.returncode == 2
    r = subprocess.run([sh, "--repo", str(clean), "--mode", "refuse", "--inventory", str(tmp_path / "absent.json")], capture_output=True, text=True)
    assert r.returncode == 2 and "missing or empty" in r.stderr


def test_inventory_classifies_the_operating_tree_plans_the_retirement_and_marks_what_a_running_process_names(tmp_path):
    ops = _synthetic_operating_tree(tmp_path)
    emptyproc = tmp_path / "emptyproc"
    emptyproc.mkdir()
    inv = _inventory(ops, tmp_path / "inv.json", proc_root=emptyproc)
    assert inv["schema"] == "operating_scripts_inventory/1" and inv["root"] == str(ops) and inv["rules"]["pattern"] == PATTERN
    by = {f["path"]: f for f in inv["files"]}
    assert inv["counts"] == {**inv["counts"], "files": 16, "executable": 12, "document": 1, "data": 3, "wrapper": 2, "live": 0, "historical": 10, "modal_hit_executables": 4, "modal_hits": 4, "moves": 7, "kept": 3}
    assert by["scripts/land.sh"]["role"] == "wrapper" and by["gates/slurm/gate_slurm.sh"]["role"] == "wrapper" and by["scripts/land.sh"]["kind"] == "landing"
    assert by["scripts/serial-queue-1.sh"]["kind"] == "queue" and by["scripts/land-chain-9.sh"]["kind"] == "queue" and by["scripts/gate-x.sh"]["kind"] == "gate"
    assert by["gates/slurm/gate_lib.py"]["kind"] == "gate" and by["scripts/ruling-x.sh"]["kind"] == "ruling" and by["scripts/post-landing-0021-checks.sh"]["kind"] == "check"
    assert by["scripts/helper.py"]["kind"] == "other" and by["scripts/pr-body-x.md"]["class"] == "document" and by["scripts/known-red.env"]["class"] == "data"
    assert [h["line"] for h in by["scripts/land.sh"]["modal_hits"]] == [3] and len(by["scripts/gate-x.sh"]["modal_hits"]) == 1 and by["scripts/ruling-x.sh"]["modal_hits"] == []
    assert by["scripts/serial-queue-1.sh"]["references"] == ["scripts/land.sh"]
    plan = inv["retirement"]
    moves = {m["path"]: m for m in plan["moves"]}
    assert set(moves) == {"scripts/serial-queue-1.sh", "scripts/land-chain-9.sh", "scripts/gate-x.sh", "gates/slurm/gate_lib.py", "gates/slurm/verdict_local.py",
                          "gates/slurm/wait_ref.sh", f"{_GL}launch.sh"}, sorted(moves)
    assert moves["scripts/gate-x.sh"]["why"] == "modal hit; gate script" and moves["scripts/serial-queue-1.sh"]["why"] == "queue script" and moves[f"{_GL}launch.sh"]["to"] == f"retired/{_GL}launch.sh"
    assert sorted(plan["kept"]) == ["scripts/helper.py", "scripts/post-landing-0021-checks.sh", "scripts/ruling-x.sh"] and plan["blocked_by_live"] == []
    assert [w["path"] for w in plan["wrappers"]] == ["scripts/land.sh", "gates/slurm/gate_slurm.sh"] and all(w["present"] for w in plan["wrappers"])
    # the wrapper's last fallback is the main checkout of the repository that ran the tool (rendered at retire time), never a
    # built-in directory name; the checkout is resolved when the wrapper runs
    assert _FALLBACK_LINE in plan["wrappers"][0]["text"] and plan["wrappers"][0]["text"].endswith(_LAND_EXEC)
    # a running process naming the queue script: it is live, and so is what its text references (the wrapper keeps its role)
    proc = tmp_path / "proc" / "4242"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"bash\0" + str(ops / "scripts/serial-queue-1.sh").encode() + b"\0")
    (tmp_path / "proc" / "self").mkdir()
    inv2 = _inventory(ops, tmp_path / "inv2.json", proc_root=tmp_path / "proc")
    by2 = {f["path"]: f for f in inv2["files"]}
    assert by2["scripts/serial-queue-1.sh"]["role"] == "live" and by2["scripts/serial-queue-1.sh"]["live_pids"] == [4242]
    assert by2["scripts/land.sh"]["role"] == "wrapper" and by2["scripts/land.sh"]["live_via"] == ["scripts/serial-queue-1.sh"]
    assert inv2["retirement"]["blocked_by_live"] == [{"path": "scripts/serial-queue-1.sh", "pids": [4242], "via": []}] and inv2["counts"]["live"] == 1
    assert "scripts/serial-queue-1.sh" not in {m["path"] for m in inv2["retirement"]["moves"]}


def test_retire_is_a_dry_run_by_default_refuses_live_blockers_and_applies_the_plan_on_a_copy_so_the_scan_reads_zero(tmp_path):
    """R7's shape: the inventory of a tree, the retirement applied on a COPY (--allow-other-root), then the scan of the retired tree's
    new inventory reads zero operating hits; the wrappers exec the repo chain and lay the /tmp verdict symlink."""
    ops = _synthetic_operating_tree(tmp_path)
    emptyproc = tmp_path / "emptyproc"
    emptyproc.mkdir()
    inv_path = tmp_path / "inv.json"
    _inventory(ops, inv_path, proc_root=emptyproc)
    tool = [sys.executable, str(LANDING / "operating_scripts_inventory.py"), "retire", "--inventory", str(inv_path), "--proc-root", str(emptyproc)]
    r = subprocess.run(tool, capture_output=True, text=True)
    assert r.returncode == 0 and "DRY RUN" in r.stdout and "[retire] move scripts/gate-x.sh -> retired/scripts/gate-x.sh" in r.stdout, r.stdout + r.stderr
    assert (ops / "scripts" / "gate-x.sh").exists() and not (ops / "retired").exists(), "a dry run moves nothing"
    # another root without the flag refuses
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(ops, copy)
    r = subprocess.run(tool + ["--root", str(copy), "--apply"], capture_output=True, text=True)
    assert r.returncode == 3 and "is not the inventory's root" in r.stderr and not (copy / "retired").exists()
    # a live blocker refuses the apply by name, nothing moved
    proc = tmp_path / "proc" / "77"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"bash\0" + str(copy / "scripts/land-chain-9.sh").encode() + b"\0")
    r = subprocess.run(tool[:-2] + ["--proc-root", str(tmp_path / "proc"), "--root", str(copy), "--allow-other-root", "--apply"], capture_output=True, text=True)
    assert r.returncode == 3 and "REFUSED: scripts/land-chain-9.sh is named by a running process (pids [77]" in r.stdout and not (copy / "retired").exists(), r.stdout
    # the apply on the copy
    manifest = tmp_path / "retired-manifest.txt"
    r = subprocess.run(tool + ["--root", str(copy), "--allow-other-root", "--apply", "--manifest", str(manifest)], capture_output=True, text=True)
    assert r.returncode == 0 and "[retire] done: 9 moved, 2 wrapper(s) written" in r.stdout, r.stdout + r.stderr
    for rel in ("scripts/serial-queue-1.sh", "scripts/land-chain-9.sh", "scripts/gate-x.sh", "gates/slurm/gate_lib.py", "gates/slurm/verdict_local.py", "gates/slurm/wait_ref.sh",
                f"{_GL}launch.sh"):
        assert not (copy / rel).exists() and (copy / "retired" / rel).exists(), rel
    assert (copy / "retired" / "scripts" / "land.sh").read_text().startswith("#!/bin/bash\n# the interim landing step") and \
        (copy / "retired" / "gates" / "slurm" / "gate_slurm.sh").read_text().startswith("#!/bin/bash\n# the interim slurm gate")
    for rel in ("scripts/ruling-x.sh", "scripts/post-landing-0021-checks.sh", "scripts/helper.py", "scripts/pr-body-x.md", "scripts/jobs.log", "scripts/known-red.env",
                "gates/slurm/baseline-abc.summary.json"):
        assert (copy / rel).exists(), rel
    wrapper = copy / "scripts" / "land.sh"
    assert _FALLBACK_LINE in wrapper.read_text() and wrapper.read_text().endswith(_LAND_EXEC) and os.access(wrapper, os.X_OK)
    lines = manifest.read_text().splitlines()
    assert len(lines) == 11 and sum(1 for l in lines if " moved " in l) == 9 and sum(1 for l in lines if " wrapper " in l) == 2
    assert any(l.endswith(" moved scripts/land.sh -> retired/scripts/land.sh") for l in lines)
    # a second apply refuses: the sources are gone and the destinations exist
    r = subprocess.run(tool + ["--root", str(copy), "--allow-other-root", "--apply", "--manifest", str(manifest)], capture_output=True, text=True)
    assert r.returncode == 3 and "REFUSED" in r.stderr and len(manifest.read_text().splitlines()) == 11
    # the retired tree's new inventory: zero operating hits for the scan (the repo scope here is a clean synthetic repo)
    inv_after = _inventory(copy, tmp_path / "inv-after.json", proc_root=emptyproc)
    assert inv_after["counts"] == {**inv_after["counts"], "modal_hit_executables": 0, "moves": 0, "executable": 5, "files": 9, "historical": 3, "wrapper": 2}
    assert not any(f["path"].startswith("retired/") for f in inv_after["files"]), "retired/ is never inventoried"
    clean = _synthetic_repo(tmp_path, with_hits=False)
    r = subprocess.run([str(LANDING / "check_no_modal.sh"), "--repo", str(clean), "--mode", "refuse", "--inventory", str(tmp_path / "inv-after.json")], capture_output=True, text=True)
    assert r.returncode == 0 and "no-modal scan: 0 hit(s) in 0 file(s)" in r.stdout and "operating executables 5 scanned, 1 documents listed" in r.stdout, r.stdout
    # the wrapper execs the chain's land.sh with the arguments and lays the /tmp symlink for the queue scripts' WAIT_VERDICT
    fake_repo = tmp_path / "fake-repo" / "scripts" / "registry" / "landing"
    fake_repo.mkdir(parents=True)
    (fake_repo / "land.sh").write_text("#!/bin/bash\necho \"chain land $*\"\n")
    (fake_repo / "land.sh").chmod(0o755)
    branch = f"probe-wrapper-{os.getpid()}"
    link = Path("/tmp") / f"gate-land-{branch}.out"
    try:
        state = tmp_path / "state"
        r = subprocess.run([str(wrapper), branch, "title", "body.md"], capture_output=True, text=True,
                           env=dict(os.environ, SISTEER_REPO=str(tmp_path / "fake-repo"), REGISTRY_LANDING_STATE=str(state)))
        assert r.returncode == 0 and r.stdout.strip() == f"chain land {branch} title body.md", r.stdout + r.stderr
        assert link.is_symlink() and os.readlink(link) == str(state / "verdicts" / f"gate-land-{branch}.out") and (state / "verdicts").is_dir()
    finally:
        if link.is_symlink():
            link.unlink()


def test_a_wrapper_written_from_a_linked_worktree_survives_its_removal_and_reads_the_site_config_when_it_runs(tmp_path):
    """A retirement run from a release branch's linked worktree records the MAIN checkout as the wrappers' last fallback, so both
    wrappers still reach the chain after that worktree is removed (they used to record the worktree and fail the exec, 127). The site
    config's SISTEER_REPO is read when a wrapper runs, after $SISTEER_REPO and before the recorded checkout, and a checkout without the
    chain's script refuses by name."""
    import shutil
    main = tmp_path / "main"
    chain = main / "scripts" / "registry" / "landing"
    chain.mkdir(parents=True)
    shutil.copy2(LANDING / "operating_scripts_inventory.py", chain / "operating_scripts_inventory.py")
    # the wrappers' default site directory comes from the site module, which the tool loads by file path through
    # scripts/site_paths.py: a checkout always carries both
    (main / "src" / "sisteer").mkdir(parents=True)
    shutil.copy2(REPO / "src" / "sisteer" / "site.py", main / "src" / "sisteer" / "site.py")
    shutil.copy2(REPO / "scripts" / "site_paths.py", main / "scripts" / "site_paths.py")
    (main / ".gitignore").write_text("__pycache__/\n")   # as the repository's own: loading them leaves bytecode beside them

    def fake_chain(at: Path) -> None:
        at.mkdir(parents=True, exist_ok=True)
        for name in ("land.sh", "gate_slurm.sh"):
            (at / name).write_text(f'#!/bin/bash\necho "{name} from $(cd "$(dirname "$0")/../../.." && pwd -P) $*"\n')
            (at / name).chmod(0o755)

    fake_chain(chain)
    wt = main / "worktrees" / "rel"          # the release branch's linked worktree the retirement runs from
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
           "-C", str(main)]
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"], ["worktree", "add", "-q", str(wt), "-b", "rel"]):
        r = subprocess.run(git + cmd, capture_output=True, text=True)
        assert r.returncode == 0, (cmd, r.stdout + r.stderr)
    ops = tmp_path / "ops"
    for rel in ("scripts/land.sh", "gates/slurm/gate_slurm.sh"):
        (ops / rel).parent.mkdir(parents=True, exist_ok=True)
        (ops / rel).write_text("#!/bin/bash\necho interim\n")
        (ops / rel).chmod(0o755)
    emptyproc = tmp_path / "emptyproc"
    emptyproc.mkdir()
    tool = [sys.executable, str(wt / "scripts" / "registry" / "landing" / "operating_scripts_inventory.py")]
    r = subprocess.run(tool + ["inventory", "--root", str(ops), "--out", str(tmp_path / "inv.json"), "--proc-root", str(emptyproc)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(tool + ["retire", "--inventory", str(tmp_path / "inv.json"), "--apply", "--proc-root", str(emptyproc)], capture_output=True, text=True)
    assert r.returncode == 0 and "2 wrapper(s) written" in r.stdout, r.stdout + r.stderr
    main_real = os.path.realpath(main)
    for rel in ("scripts/land.sh", "gates/slurm/gate_slurm.sh"):
        text = (ops / rel).read_text()
        assert f'R=${{R:-"{main_real}"}};' in text and str(wt) not in text and os.path.realpath(wt) not in text, text
    r = subprocess.run(git + ["worktree", "remove", str(wt)], capture_output=True, text=True)
    assert r.returncode == 0 and not wt.exists(), r.stdout + r.stderr

    home = tmp_path / "home"
    home.mkdir()
    base = {k: v for k, v in os.environ.items() if k not in ("SISTEER_REPO", "SISTEER_CONFIG", "SISTEER_SITE_DIR")}
    base.update(HOME=str(home), REGISTRY_LANDING_STATE=str(tmp_path / "state"))
    branch = f"probe-wt-wrapper-{os.getpid()}"
    link = Path("/tmp") / f"gate-land-{branch}.out"

    def run(rel: str, env: dict, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(ops / rel), *args], capture_output=True, text=True, env=env)

    try:
        # no $SISTEER_REPO, no site config: the recorded main checkout, for both wrappers
        r = run("scripts/land.sh", base, branch, "title", "body.md")
        assert r.returncode == 0 and r.stdout.strip() == f"land.sh from {main_real} {branch} title body.md", r.stdout + r.stderr
        r = run("gates/slurm/gate_slurm.sh", base, "--branch", branch)
        assert r.returncode == 0 and r.stdout.strip() == f"gate_slurm.sh from {main_real} --branch {branch}", r.stdout + r.stderr
        # the site config's SISTEER_REPO, read when the wrapper runs (a ~ value expands against $HOME)
        fake_chain(home / "site-checkout" / "scripts" / "registry" / "landing")
        (home / ".config" / "sisteer").mkdir(parents=True)
        (home / ".config" / "sisteer" / "config.env").write_text("SISTEER_REPO='~/site-checkout'\n")
        r = run("scripts/land.sh", base, branch)
        assert r.returncode == 0 and r.stdout.strip() == f"land.sh from {os.path.realpath(home / 'site-checkout')} {branch}", r.stdout + r.stderr
        # $SISTEER_CONFIG names the file; $SISTEER_REPO wins over any config
        fake_chain(tmp_path / "other" / "scripts" / "registry" / "landing")
        (tmp_path / "other.env").write_text(f"SISTEER_REPO={tmp_path / 'other'}\n")
        r = run("gates/slurm/gate_slurm.sh", dict(base, SISTEER_CONFIG=str(tmp_path / "other.env")), "--branch", branch)
        assert r.returncode == 0 and r.stdout.strip() == f"gate_slurm.sh from {os.path.realpath(tmp_path / 'other')} --branch {branch}", r.stdout + r.stderr
        r = run("scripts/land.sh", dict(base, SISTEER_REPO=str(main)), branch)
        assert r.returncode == 0 and r.stdout.strip() == f"land.sh from {main_real} {branch}", r.stdout + r.stderr
        # a checkout without the chain's script: a refusal by name, exit 2
        (tmp_path / "empty").mkdir()
        r = run("scripts/land.sh", dict(base, SISTEER_REPO=str(tmp_path / "empty")), branch)
        assert r.returncode == 2 and r.stdout == "" and f"REFUSED: {ops / 'scripts/land.sh'}: no {tmp_path / 'empty'}/scripts/registry/landing/land.sh" in r.stderr, r.stderr
    finally:
        if link.is_symlink():
            link.unlink()


def test_live_tree_has_no_modal_reach_by_ownership():
    """The invariant, live (spec S3; C10): no line of the registry's ownership matches the Modal reach pattern. No allowlist."""
    found = hits(owned_files(REPO))
    assert not found, "Modal reach in the registry's ownership:\n" + "\n".join(f"  {p}:{n}: {t}" for p, n, t in found)


def test_live_tree_has_no_modal_reach_by_reachable_entry_points():
    """The invariant by REACH (spec S3; C10): no module importing sisteer.registry brings into sys.modules matches the pattern."""
    found = hits(reachable_modules(REPO))
    assert not found, "Modal reach in a module importing sisteer.registry reaches:\n" + "\n".join(f"  {p}:{n}: {t}" for p, n, t in found)


def test_the_live_scans_are_wired_the_way_the_plan_states():
    """The invariant's wiring: the ownership scan covers exactly the plan's five scopes and the import scan covers the package; the
    removed modules, apps, unit files and test modules are absent from the tree (so a file re-adding one is a hit, not a quiet return);
    and the shell scan over THIS repo reads zero hits -- the number the cutover's refuse mode rests on."""
    owned = owned_files(REPO)
    assert any(p.match("src/sisteer/registry/*.py") for p in owned) and any(p.match("scripts/registry/landing/*") for p in owned)
    assert any(p.name.startswith("registry-") and p.parent.name == "user" for p in owned) and any(p.suffix == ".yaml" for p in owned)
    assert any(p.name.startswith("test_registry_") for p in owned) and all("__pycache__" not in p.parts for p in owned)
    gone = [f"src/sisteer/registry/{_M}{_S}ops.py", f"src/sisteer/registry/{_V}{_S}ops.py", f"scripts/{_M}/{_R}{_S}{_V}.py",
            f"scripts/{_M}/{_R}{_S}smoke.py", f"systemd/user/registry-{_M}-pull.service", f"systemd/user/registry-{_M}-pull.timer",
            f"tests/test_registry_{_M}{_S}ops.py", f"tests/test_registry_emit_{_M}.py"]
    for rel in gone:
        assert not (REPO / rel).exists(), f"{rel} is back in the tree"
    reach = reachable_modules(REPO)
    assert any(Path(f).name == "cli.py" for f in reach), "the registry CLI is reachable from the package"
    r = subprocess.run([str(LANDING / "check_no_modal.sh"), "--repo", str(REPO), "--mode", "report"], capture_output=True, text=True)
    assert r.returncode == 0 and "no-modal scan: 0 hit(s) in 0 file(s)" in r.stdout, r.stdout[-2000:]
