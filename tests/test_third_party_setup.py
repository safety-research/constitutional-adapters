"""reproduce/lib/third_party.sh, the two lib/hf_env.sh exports and the eval jobs' bootstrap (full22/common.sh f22_init):
the G0DM0D3 templates and the refusal-direction fit prompts (neither distributed with this repository) are fetched and
verified on the setup host, every one before any is shipped, shipped by content to a cluster's ``$SHARED/third_party`` and
verified there, and jobs read exactly those copies offline. Run for real through the ``local:<dir>`` ssh seam
(lib/common.sh repro_ssh), with the site's interpreter wrapped so each fetch serves its invented stand-in (godmode_standin,
arditi_fit_prompts_standin) from a fake network: nothing here reaches upstream, and no host, user or path of a real site is
named."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import arditi_fit_prompts_standin as fps
from front_door_testlib import git, write_site
from godmode_standin import STANDIN_PATH, STANDIN_TEXT, install_standin_pins, refuse_network
from sisteer.data import arditi_fit_prompts as afp
from sisteer.eval import attacks_godmode as agm

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "reproduce" / "lib"
SCRIPT = LIB / "third_party.sh"

#: a stand-in for ``python -m sisteer.eval.attacks_godmode``: the module's own main() with the stand-in's pins and a fake
#: network that serves the stand-in (STANDIN_SERVE=good), a tampered copy (tampered) or nothing (down), logging each URL
_DRIVER = """\
import io, os, sys, urllib.error
sys.path[:0] = [{src!r}, {tests!r}]
from sisteer.eval import attacks_godmode as agm
import godmode_standin as gs
data = gs.STANDIN_PATH.read_bytes()
class _Patch:
    def setattr(self, obj, name, value): setattr(obj, name, value)
    def setitem(self, d, key, value): d[key] = value
gs.install_standin_pins(_Patch(), agm, data)
class _Resp(io.BytesIO):
    def __init__(self, body, url): super().__init__(body); self._url = url
    def geturl(self): return self._url
def _open(url):
    with open(os.environ["STANDIN_NET_LOG"], "a") as fh: fh.write(url + "\\n")
    serve = os.environ.get("STANDIN_SERVE", "good")
    if serve == "down": raise urllib.error.URLError("no route (test)")
    return _Resp(data if serve == "good" else data + b"tampered", url)
agm._open_url = _open
sys.exit(agm.main(sys.argv[1:]))
"""

#: the same for ``python -m sisteer.data.arditi_fit_prompts``: the module's own main() with the stand-in upstream's manifest and a
#: fake network over its four split files (STANDIN_SERVE_ARDITI=good | tampered | down), logging each URL to the same log
_FP_DRIVER = """\
import os, sys
sys.path[:0] = [{src!r}, {tests!r}]
from pathlib import Path
from sisteer.data import arditi_fit_prompts as m
import arditi_fit_prompts_standin as fps
m.MANIFEST_PATH = fps.write_manifest(Path(os.environ["STANDIN_FP_MANIFEST"]))
_serve = fps.serve([], os.environ.get("STANDIN_SERVE_ARDITI", "good"))
def _open(url):
    with open(os.environ["STANDIN_NET_LOG"], "a") as fh: fh.write(url + "\\n")
    return _serve(url)
m._open_url = _open
sys.exit(m.main(sys.argv[1:]))
"""
#: where the stand-in's fit prompts live under a third-party root
FP_REL = f"refusal_direction/{fps.COMMIT}/fit_prompts.json"


def _world(tmp_path: Path) -> dict:
    fake = tmp_path / "fake"; fake.mkdir()
    site = write_site(tmp_path, clusters=("cluster_a",), ssh={"cluster_a": f"local:{fake}"})
    driver = tmp_path / "standin_agm.py"
    driver.write_text(_DRIVER.format(src=str(REPO / "src"), tests=str(REPO / "tests")))
    fp_driver = tmp_path / "standin_fp.py"
    fp_driver.write_text(_FP_DRIVER.format(src=str(REPO / "src"), tests=str(REPO / "tests")))
    py = tmp_path / "bin" / "python"; py.parent.mkdir()
    py.write_text("#!/bin/bash\n"
                  f'if [ "$1" = -m ] && [ "$2" = sisteer.eval.attacks_godmode ]; then shift 2; exec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"; fi\n'
                  f'if [ "$1" = -m ] && [ "$2" = sisteer.data.arditi_fit_prompts ]; then shift 2; exec {shlex.quote(sys.executable)} {shlex.quote(str(fp_driver))} "$@"; fi\n'
                  f'exec {shlex.quote(sys.executable)} "$@"\n')
    py.chmod(0o755)
    cfg = site / "config.env"
    cfg.write_text("\n".join(f"SISTEER_PYTHON={py}" if l.startswith("SISTEER_PYTHON=") else l for l in cfg.read_text().splitlines()) + "\n")
    shared = tmp_path / "shared_cluster_a"
    return {"site": site, "cfg": cfg, "fake": fake, "shared": shared, "cache": tmp_path / "setup-cache",
            "dest": shared / "third_party" / agm.CACHE_RELPATH, "dest_fp": shared / "third_party" / FP_REL,
            "fp_manifest": tmp_path / "standin_fp_manifest.json", "net": tmp_path / "net.log"}


def _run(w: dict, *args: str, serve: str = "good", serve_arditi: str = "good", extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if k not in ("SISTEER_SITE_DIR", "SISTEER_CONFIG", agm.OFFLINE_ENV, agm.THIRD_PARTY_DIR_ENV, "REPRO_DRY_RUN")}
    env.update(STANDIN_SERVE=serve, STANDIN_SERVE_ARDITI=serve_arditi, STANDIN_FP_MANIFEST=str(w["fp_manifest"]),
               STANDIN_NET_LOG=str(w["net"]), **(extra_env or {}))
    return subprocess.run(["bash", str(SCRIPT), "--cluster", "cluster_a", "--config", str(w["cfg"]), "--cache-dir", str(w["cache"]), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def _net(w: dict) -> list[str]:
    return w["net"].read_text().splitlines() if w["net"].exists() else []


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0


def test_fetches_here_verifies_ships_and_verifies_on_the_cluster_then_is_a_no_op(tmp_path):
    w = _world(tmp_path)
    r = _run(w)
    assert r.returncode == 0, r.stdout + r.stderr
    data, doc = STANDIN_PATH.read_bytes(), fps.expected_document()
    assert w["dest"].read_bytes() == data and w["dest_fp"].read_bytes() == doc     # the cluster's copies: the verified bytes
    assert (w["cache"] / agm.CACHE_RELPATH).read_bytes() == data and (w["cache"] / FP_REL).read_bytes() == doc   # and the setup host's
    assert _files(w["shared"] / "third_party") == sorted([str(agm.CACHE_RELPATH), FP_REL])   # the document alone, no temp file beside it
    assert _net(w) == [agm.SOURCES[0]] + [fps.raw_url(n) for n in fps.NAMES]      # each fetched once, from its first source
    assert "G0DM0D3" in r.stdout and "FETCHED on this host" in r.stdout and f"verified on cluster_a: {w['dest']}" in r.stdout
    assert "refusal-direction fit prompts" in r.stdout and "BUILT on this host" in r.stdout and f"verified on cluster_a: {w['dest_fp']}" in r.stdout
    # again: the cached copies here, the verified copies there, no network
    net = _net(w)
    r = _run(w, serve="down", serve_arditi="down")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("PRESENT on this host") == 2 and _net(w) == net and w["dest"].read_bytes() == data and w["dest_fp"].read_bytes() == doc


def test_a_wrong_copy_on_the_cluster_is_replaced_by_the_verified_bytes(tmp_path):
    w = _world(tmp_path)
    w["dest"].parent.mkdir(parents=True)
    w["dest"].write_bytes(b"not the pinned file")
    w["dest_fp"].parent.mkdir(parents=True)
    w["dest_fp"].write_bytes(b"not the recorded document")
    r = _run(w)
    assert r.returncode == 0, r.stdout + r.stderr
    assert w["dest"].read_bytes() == STANDIN_PATH.read_bytes() and w["dest_fp"].read_bytes() == fps.expected_document()
    assert _files(w["shared"] / "third_party") == sorted([str(agm.CACHE_RELPATH), FP_REL])


@pytest.mark.parametrize("which", ["godmode", "fit_prompts"])
@pytest.mark.parametrize("serve,why", [("tampered", "is not the pinned file"), ("down", "could not be fetched from any source")])
def test_a_refused_fetch_ships_nothing(tmp_path, serve, why, which):
    """Every file is fetched and verified here before any is shipped, so a refusal of either leaves the cluster untouched."""
    w = _world(tmp_path)
    r = _run(w, **({"serve": serve} if which == "godmode" else {"serve_arditi": serve}))
    assert r.returncode != 0
    what = "G0DM0D3" if which == "godmode" else "refusal-direction fit prompts"
    assert why in r.stderr and f"the {what} fetch on this host refused" in r.stderr and "nothing was shipped to cluster_a" in r.stderr, r.stderr
    assert not w["shared"].joinpath("third_party").exists()
    # the setup host's cache holds only what verified: nothing of the refused file (the G0DM0D3 file, fetched first, when it was not the one)
    assert _files(w["cache"]) == ([] if which == "godmode" else [str(agm.CACHE_RELPATH)])


def test_an_unreachable_login_host_is_refused_by_name(tmp_path):
    w = _world(tmp_path)
    (w["fake"] / "ssh_down").write_text("")
    r = _run(w)
    assert r.returncode != 0 and "could not be reached" in r.stderr and "NOT installed" in r.stderr, r.stderr
    assert not w["dest"].exists()


def test_dry_run_fetches_and_ships_nothing(tmp_path):
    w = _world(tmp_path)
    m = afp.load_manifest()                                                        # a dry run reads the repository's own pins
    for r in (_run(w, "--dry-run"), _run(w, extra_env={"REPRO_DRY_RUN": "1"})):     # the flag, or an exported dry run (fail-closed)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "-m sisteer.eval.attacks_godmode fetch --json --cache-dir" in r.stdout
        assert f"{w['shared']}/third_party/{agm.CACHE_RELPATH}" in r.stdout and agm.FILE_SHA256[:16] in r.stdout
        assert "-m sisteer.data.arditi_fit_prompts fetch --json --cache-dir" in r.stdout
        assert f"{w['shared']}/third_party/{afp.document_relpath(m)}" in r.stdout and m["document"]["sha256"][:16] in r.stdout
    assert _net(w) == [] and _files(w["cache"]) == [] and not w["shared"].joinpath("third_party").exists()


def test_an_unknown_cluster_or_no_cluster_is_refused(tmp_path):
    w = _world(tmp_path)
    r = subprocess.run(["bash", str(SCRIPT), "--cluster", "cluster_z", "--config", str(w["cfg"])], capture_output=True, text=True,
                       env={k: v for k, v in os.environ.items() if k not in ("SISTEER_SITE_DIR", "SISTEER_CONFIG")}, timeout=60)
    assert r.returncode != 0 and "cluster_z" in r.stderr, r.stderr
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "--cluster <label> is required" in r.stderr


# --------------------------------------------------------------------------- lib/hf_env.sh: the job side

def _hf_env(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'set -e; source {shlex.quote(str(LIB / "hf_env.sh"))}; '
                                         'echo "rc=$? dir=${SISTEER_THIRD_PARTY_DIR:-unset} offline=${SISTEER_THIRD_PARTY_OFFLINE:-unset}"'],
                          capture_output=True, text=True, env={"PATH": os.environ["PATH"], **env}, timeout=60)


def test_hf_env_points_jobs_at_the_shared_copy_offline(tmp_path):
    shared = tmp_path / "shared"
    r = _hf_env({"USER": "nobody-here", "SISTEER_HF_HOME": str(shared / "hf_cache"), "SISTEER_SHARED_ROOT": str(shared)})
    assert r.stdout.strip() == f"rc=0 dir={shared}/third_party offline=1", r.stdout + r.stderr
    # the root follows the token's directory when SISTEER_SHARED_ROOT is not given (dirname of SISTEER_HF_HOME)
    r = _hf_env({"USER": "nobody-here", "SISTEER_HF_HOME": str(shared / "hf_cache")})
    assert r.stdout.strip() == f"rc=0 dir={shared}/third_party offline=1", r.stdout + r.stderr
    # a caller's own values stand
    r = _hf_env({"USER": "nobody-here", "SISTEER_HF_HOME": str(shared / "hf_cache"), "SISTEER_THIRD_PARTY_DIR": "/elsewhere",
                 "SISTEER_THIRD_PARTY_OFFLINE": "0"})
    assert r.stdout.strip() == "rc=0 dir=/elsewhere offline=0", r.stdout + r.stderr


def test_hf_env_with_no_shared_root_sets_nothing_and_still_sources_cleanly():
    r = _hf_env({"USER": "nobody-here-either"})
    assert r.returncode == 0 and r.stdout.strip() == "rc=0 dir=unset offline=unset", r.stdout + r.stderr


def test_the_installed_copy_is_the_one_a_job_reads(tmp_path, monkeypatch):
    """Where third_party.sh installs and where the module looks under hf_env.sh's exports are the same file, and a job
    renders from it with fetching off."""
    w = _world(tmp_path)
    assert _run(w).returncode == 0
    r = _hf_env({"USER": "nobody-here", "SISTEER_HF_HOME": str(w["shared"] / "hf_cache"), "SISTEER_SHARED_ROOT": str(w["shared"])})
    fields = dict(f.split("=", 1) for f in r.stdout.split())
    monkeypatch.setenv(agm.THIRD_PARTY_DIR_ENV, fields["dir"])
    monkeypatch.setenv(agm.OFFLINE_ENV, fields["offline"])
    monkeypatch.setattr(agm, "_open_url", refuse_network)
    install_standin_pins(monkeypatch, agm, STANDIN_PATH.read_bytes())
    assert agm.cache_path() == w["dest"]
    assert agm.load_templates() == STANDIN_TEXT
    # the refusal-direction fit prompts the same way: the installed document is the one the fitter's arditi:* sets read, offline
    monkeypatch.setattr(afp, "MANIFEST_PATH", w["fp_manifest"])
    monkeypatch.setattr(afp, "_open_url", fps.refuse_network)
    assert afp.document_path(afp.load_manifest()) == w["dest_fp"]
    sets, rep = afp.load_sets()
    assert rep["status"] == "PRESENT" and sets == {n: fps.expected_sets()[n] for n in fps.NAMES}


# --------------------------------------------------------------------------- full22/common.sh f22_init: the eval jobs

F22 = REPO / "scripts" / "sdfbase" / "full22"

#: one eval job's read of the templates, as jailbreak_run's preflight makes it, in the environment f22_init left: the
#: stand-in's pins, and a network that logs each URL and refuses it as a compute node with no egress does
_JOB_READ = """\
import os, sys, urllib.error
sys.path[:0] = [{src!r}, {tests!r}]
from sisteer.eval import attacks_godmode as agm
import godmode_standin as gs
class _Patch:
    def setattr(self, obj, name, value): setattr(obj, name, value)
    def setitem(self, d, key, value): d[key] = value
gs.install_standin_pins(_Patch(), agm, gs.STANDIN_PATH.read_bytes())
def _no_egress(url):
    with open(os.environ["STANDIN_NET_LOG"], "a") as fh: fh.write(url + "\\n")
    raise urllib.error.URLError("no route (test)")
agm._open_url = _no_egress
print("JOB", "dir=" + os.environ.get(agm.THIRD_PARTY_DIR_ENV, "unset"), "offline=" + os.environ.get(agm.OFFLINE_ENV, "unset"),
      "path=" + str(agm.cache_path()))
assert agm.load_templates() == gs.STANDIN_TEXT
print("JOB loaded")
"""


def _site_job_env(w: dict) -> dict:
    """The env prefix the front door puts on every cluster_a job line (lib/slurm.sh slurm_job_env, the same keys as
    submit_twin.py job_env): SISTEER_HF_HOME among them, so f22_init takes its own HF branch and never sources hf_env.sh.
    The registry spool is dropped: f22_init's `running` emission is not under test here."""
    drop = ("SISTEER_SITE_DIR", "SISTEER_CONFIG", "SISTEER_REGISTRY_SPOOL")
    r = subprocess.run(["bash", "-c", f"source {shlex.quote(str(LIB / 'common.sh'))}; source {shlex.quote(str(LIB / 'slurm.sh'))}; "
                                      f"REPRO_CONFIG={shlex.quote(str(w['cfg']))}; repro_load_config; slurm_job_env cluster_a"],
                       capture_output=True, text=True, timeout=60, env={k: v for k, v in os.environ.items() if k not in drop})
    assert r.returncode == 0, r.stdout + r.stderr
    env = dict(kv.split("=", 1) for kv in shlex.split(r.stdout))
    assert env["SISTEER_SHARED_ROOT"] == str(w["shared"]) and env["SISTEER_HF_HOME"] == f"{w['shared']}/hf_cache", env
    env.pop("SISTEER_REGISTRY_SPOOL")
    return env


def _wave_checkout(eval_root: Path) -> str:
    """The checkout an eval job runs from, at <eval repo root>/<sha>: the real common.sh, the roster f22_init checks and a
    src/sisteer stub, committed so f22_init's HEAD and clean-tree checks hold."""
    co = eval_root / "work"
    for d in ("scripts/sdfbase/full22", "src/sisteer", "configs"):
        (co / d).mkdir(parents=True)
    shutil.copy(F22 / "common.sh", co / "scripts/sdfbase/full22/common.sh")
    shutil.copy(REPO / "configs/sdfbase_full22_roster.yaml", co / "configs/sdfbase_full22_roster.yaml")
    (co / "src/sisteer/__init__.py").write_text("")
    git(co, "init", "-q"); git(co, "config", "user.email", "t@example.invalid"); git(co, "config", "user.name", "t")
    git(co, "add", "-A"); git(co, "commit", "-qm", "wave")
    sha = git(co, "rev-parse", "HEAD")
    co.rename(eval_root / sha)
    return sha


def _f22_job(w: dict, tmp_path: Path, job_env: dict) -> subprocess.CompletedProcess:
    """An eval job on cluster_a: the job env, the real f22_init in the wave checkout, then the job's read of the templates.
    Nothing else is inherited, so a third-party root or offline flag can only come from the job env or from f22_init."""
    sha = _wave_checkout(Path(job_env["SISTEER_EVAL_REPO_ROOT"]))
    read = tmp_path / "job_read.py"
    read.write_text(_JOB_READ.format(src=str(REPO / "src"), tests=str(REPO / "tests")))
    home = tmp_path / "home"; home.mkdir(exist_ok=True)
    body = (f'set -uo pipefail\nWAVE_SHA={sha}\n'
            'source "$SISTEER_EVAL_REPO_ROOT/$WAVE_SHA/scripts/sdfbase/full22/common.sh"\n'
            'F22_LAUNCH_CELLS=()\nf22_init "$WAVE_SHA"\n'
            f'exec {shlex.quote(sys.executable)} {shlex.quote(str(read))}\n')
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "USER": "tester", "XDG_CACHE_HOME": str(home / ".cache"),
           "F22_SERVING_PATH": "lora_native", "SLURM_JOB_NAME": "f22-test", "SLURM_JOB_ID": "4242",
           "STANDIN_NET_LOG": str(w["net"]), **job_env}
    return subprocess.run(["bash", "-c", body], capture_output=True, text=True, env=env, timeout=120)


@pytest.mark.parametrize("hf", ["site_hf_home", "installed_helper_without_the_exports"])
def test_an_eval_job_reads_the_installed_copy_offline_and_never_fetches(tmp_path, hf):
    """A configured site's eval job gets SISTEER_HF_HOME on its job line, so f22_init sets HF_HOME itself and never sources
    hf_env.sh; a job without it sources the cluster's installed helper, which may predate the third-party exports. Either way
    the job reads the copy third_party.sh installed under the shared root, with fetching off, and reaches for no network."""
    w = _world(tmp_path)
    assert _run(w).returncode == 0                                   # the verified copy, installed on cluster_a
    net_before = _net(w)
    job_env = _site_job_env(w)
    if hf == "installed_helper_without_the_exports":
        job_env.pop("SISTEER_HF_HOME")
        helper = w["shared"] / "scripts" / "hf_env.sh"; helper.parent.mkdir(parents=True)
        helper.write_text('export HF_HOME="$SISTEER_SHARED_ROOT/hf_cache" HF_TOKEN_PATH="$SISTEER_SHARED_ROOT/hf_token" '
                          'HUGGINGFACE_TOKEN_PATH="$SISTEER_SHARED_ROOT/hf_token"\n')
    r = _f22_job(w, tmp_path, job_env)
    job = [ln for ln in r.stdout.splitlines() if ln.startswith("JOB ")]
    assert job and job[0] == f"JOB dir={w['shared']}/third_party offline=1 path={w['dest']}", r.stdout[-1500:] + r.stderr[-1500:]
    assert r.returncode == 0 and "JOB loaded" in job, r.stdout[-1500:] + r.stderr[-1500:]
    assert _net(w) == net_before                                     # the job fetched nothing
    assert not (tmp_path / "home" / ".cache").exists()               # and wrote no per-user cache on the node


def test_an_eval_jobs_own_third_party_values_stand(tmp_path):
    """A third-party root and offline flag given on the job line win over f22_init's defaults."""
    w = _world(tmp_path)
    assert _run(w).returncode == 0
    net_before = _net(w)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(w["shared"] / "third_party", elsewhere)
    r = _f22_job(w, tmp_path, {**_site_job_env(w), agm.THIRD_PARTY_DIR_ENV: str(elsewhere), agm.OFFLINE_ENV: "true"})
    assert f"JOB dir={elsewhere} offline=true path={elsewhere / agm.CACHE_RELPATH}" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    assert r.returncode == 0 and "JOB loaded" in r.stdout and _net(w) == net_before
