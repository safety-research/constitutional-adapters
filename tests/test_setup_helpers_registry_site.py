"""A cluster job learns its registry values from the cluster block, not from probing fixed storage roots.

* `scripts/registry/registry_env.sh` reads `registry_site.env` beside it: the shared root (-> the spool), the registry label (-> the
  emitter's substrate), the site label and the site's Slurm-cluster map; a value the job already carries wins; with no such file
  there is no shared root at all, whatever host it runs on.
* `reproduce/00_setup.sh --what helpers` renders that file per cluster from the cluster block and ships it beside the helper (shown
  under --dry-run; through the `local:` ssh seam for real, after which the installed helper resolves the cluster's values).
* `reproduce/00_setup.sh --what third_party` calls lib/third_party.sh once per cluster with the site config, refuses by name when the
  script is missing, and is part of `--what all`.

Synthetic sites, roots and labels only."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, run_stage, write_site  # noqa: E402

HELPER = REPO / "scripts" / "registry" / "registry_env.sh"
SETUP = REPRO / "00_setup.sh"
#: what the helper may export; every other name the job sees is the test's own
EXPORTS = ("SISTEER_REGISTRY_SPOOL", "SISTEER_REGISTRY_PUSH", "SISTEER_REGISTRY_SUBSTRATE", "SISTEER_REGISTRY_CLUSTER_LABEL", "SISTEER_SLURM_SUBSTRATES")
SITE_KEYS = ("REGISTRY_SITE_SHARED_ROOT", "REGISTRY_SITE_SUBSTRATE", "REGISTRY_SITE_CLUSTER_LABEL", "REGISTRY_SITE_SLURM_SUBSTRATES")


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _install_helper(helpers: Path, *, push: bool = True) -> Path:
    h = helpers / "registry_env.sh"; h.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(HELPER, h); h.chmod(0o755)
    if push:
        _exe(helpers / "push_records.sh", "#!/bin/sh\nexit 0\n")
    return h


def _source(helper: Path, pre: str = "") -> tuple[dict, str]:
    """Source the helper in a clean `bash -u` (no inherited registry variable) and report what it left: {name: value or None}."""
    names = EXPORTS + SITE_KEYS + ("_reg_here",)
    probe = "; ".join(f'printf "%s=%s\\n" {n} "${{{n}-<unset>}}"' for n in names)
    script = f"set -u\n{pre}\nsource {helper}\nsource {helper}\n{probe}\n"          # twice: safe to source twice
    env = {"PATH": os.environ["PATH"], "HOME": "/nonexistent-home", "USER": os.environ.get("USER", "nobody")}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    vals = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    return {k: (None if v == "<unset>" else v) for k, v in vals.items()}, r.stderr


def _site_file(helpers: Path, shared: Path, substrate: str, label: str, slurm_map: str = "") -> None:
    (helpers / "registry_site.env").write_text(
        f"# synthetic\nREGISTRY_SITE_SHARED_ROOT={shared}\nREGISTRY_SITE_SUBSTRATE={substrate}\nREGISTRY_SITE_CLUSTER_LABEL={label}\n"
        f"REGISTRY_SITE_SLURM_SUBSTRATES='{slurm_map}'\n")


def test_the_helper_takes_the_clusters_values_from_the_rendered_site_file_beside_it(tmp_path):
    shared = tmp_path / "shared_alpha"; helpers = shared / "scripts" / "registry"
    h = _install_helper(helpers)
    _site_file(helpers, shared, "slurm", "alpha", "clusterx=slurm")
    vals, err = _source(h)
    assert vals["SISTEER_REGISTRY_SPOOL"] == f"{shared}/registry/records" and (shared / "registry" / "records").is_dir()
    assert vals["SISTEER_REGISTRY_PUSH"] == f"{helpers}/push_records.sh"
    assert vals["SISTEER_REGISTRY_SUBSTRATE"] == "slurm" and vals["SISTEER_REGISTRY_CLUSTER_LABEL"] == "alpha"
    assert vals["SISTEER_SLURM_SUBSTRATES"] == "clusterx=slurm"
    assert all(vals[k] is None for k in SITE_KEYS + ("_reg_here",)), vals      # nothing else of the file reaches the job
    assert "registry_env:" not in err


def test_a_value_the_job_already_carries_wins(tmp_path):
    shared = tmp_path / "shared"; helpers = shared / "scripts" / "registry"
    h = _install_helper(helpers)
    _site_file(helpers, shared, "slurm", "alpha", "clusterx=slurm")
    pre = (f"export SISTEER_REGISTRY_SPOOL={shared}/registry/records SISTEER_REGISTRY_SUBSTRATE=other SISTEER_REGISTRY_CLUSTER_LABEL=beta "
           f"SISTEER_SLURM_SUBSTRATES=y=z SISTEER_REGISTRY_PUSH=/bin/true")
    vals, _ = _source(h, pre)
    assert (vals["SISTEER_REGISTRY_SUBSTRATE"], vals["SISTEER_REGISTRY_CLUSTER_LABEL"], vals["SISTEER_SLURM_SUBSTRATES"],
            vals["SISTEER_REGISTRY_PUSH"]) == ("other", "beta", "y=z", "/bin/true")


def test_with_no_site_file_there_is_no_shared_root_on_any_host(tmp_path):
    """The helper probes nothing: installed with no registry_site.env, under the account running this test (a cluster node mounts a
    per-user root for such an account), it resolves no spool and names no substrate, and says so once."""
    helpers = tmp_path / "shared" / "scripts" / "registry"
    h = _install_helper(helpers)
    vals, err = _source(h)
    assert all(vals[k] is None for k in EXPORTS), vals
    assert "registry_env: no shared root" in err and "registry_site.env" in err and err.count("registry_env:") == 2   # one line per source
    # no fixed root in the code: the only absolute path outside comments is /dev/null
    code = "\n".join(l for l in HELPER.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert set(re.findall(r"(?<![\w$}.\-])/[A-Za-z][\w.\-/]*", code)) <= {"/dev/null"}, code


def test_helpers_dry_run_prints_each_clusters_rendered_site_file(tmp_path):
    site = write_site(tmp_path, clusters=("alpha", "beta"), extra_config='SISTEER_SLURM_SUBSTRATES="clusterx=slurm"')
    r = run_stage("00_setup.sh", "--what", "helpers", "--dry-run", site=site)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    for lab in ("alpha", "beta"):
        shared = tmp_path / f"shared_{lab}"
        assert f"DRY-RUN: write {lab}-login:{shared}/scripts/registry/registry_site.env:" in r.stdout, r.stdout[-2500:]
        for line in (f"REGISTRY_SITE_SHARED_ROOT={shared}", "REGISTRY_SITE_SUBSTRATE=slurm", f"REGISTRY_SITE_CLUSTER_LABEL={lab}",
                     "REGISTRY_SITE_SLURM_SUBSTRATES=clusterx=slurm"):
            assert f"DRY-RUN:   {line}" in r.stdout, (lab, line)
    assert not any((tmp_path / f"shared_{lab}" / "scripts").exists() for lab in ("alpha", "beta"))


def _fake_rsync(bin_dir: Path) -> None:
    """rsync to `local:<dir>:<path>` (the ssh seam's host spelling) copies on this host; any other destination is refused."""
    _exe(bin_dir / "rsync", r'''#!/bin/bash
args=(); for a in "$@"; do case "$a" in -*) ;; *) args+=("$a");; esac; done
dest="${args[-1]}"; unset 'args[-1]'
case "$dest" in local:*) dest="${dest#local:}"; dest="${dest#*:}";; *) echo "fake rsync: refused $dest" >&2; exit 12;; esac
case "$dest" in */) mkdir -p "$dest";; *) mkdir -p "$(dirname "$dest")";; esac
cp -p "${args[@]}" "$dest"
''')


def test_helpers_ships_the_rendered_site_file_and_the_installed_helper_resolves_the_cluster(tmp_path):
    """Through the `local:` ssh seam: each cluster's helpers directory gets registry_env.sh, push_records.sh and its own
    registry_site.env, and sourcing the installed helper there yields that cluster's spool, push script, substrate and label."""
    seam = {lab: tmp_path / f"seam_{lab}" for lab in ("alpha", "beta")}
    for d in seam.values():
        d.mkdir()
    site = write_site(tmp_path, clusters=("alpha", "beta"), ssh={lab: f"local:{d}" for lab, d in seam.items()})
    bin_dir = tmp_path / "bin"; _fake_rsync(bin_dir)
    r = run_stage("00_setup.sh", "--what", "helpers", site=site, env={"PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    for lab in ("alpha", "beta"):
        shared = tmp_path / f"shared_{lab}"; helpers = shared / "scripts" / "registry"
        assert (helpers / "registry_env.sh").read_text() == HELPER.read_text()
        assert os.access(helpers / "push_records.sh", os.X_OK)
        text = (helpers / "registry_site.env").read_text()
        assert f"REGISTRY_SITE_CLUSTER_LABEL={lab}\n" in text and "REGISTRY_SITE_SUBSTRATE=slurm\n" in text
        vals, err = _source(helpers / "registry_env.sh")
        assert vals["SISTEER_REGISTRY_SPOOL"] == f"{shared}/registry/records" and vals["SISTEER_REGISTRY_PUSH"] == f"{helpers}/push_records.sh"
        assert vals["SISTEER_REGISTRY_SUBSTRATE"] == "slurm" and vals["SISTEER_REGISTRY_CLUSTER_LABEL"] == lab, (lab, vals, err)
        assert (site.parent / "state" / "setup" / "helpers" / f"DONE.{lab}").is_file()


def test_third_party_dry_run_names_the_script_per_cluster_and_all_includes_it(tmp_path):
    site = write_site(tmp_path, clusters=("alpha", "beta"))
    r = run_stage("00_setup.sh", "--what", "third_party", "--dry-run", site=site)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for lab in ("alpha", "beta"):
        assert f"DRY-RUN: bash {REPRO}/lib/third_party.sh --cluster {lab} --config {site}/config.env --dry-run" in r.stdout, r.stdout[-2000:]
    text = SETUP.read_text()
    assert "venvs|registry|helpers|third_party|mirror|checkpoints|all)" in text
    assert text.index("if want helpers; then") < text.index("if want third_party; then") < text.index("if want mirror; then")


def _copy_reproduce(tmp_path: Path) -> Path:
    copy = tmp_path / "repo"
    shutil.copytree(REPRO, copy / "reproduce", ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "src" / "sisteer").mkdir(parents=True)
    shutil.copy2(REPO / "src" / "sisteer" / "site.py", copy / "src" / "sisteer")   # clusters.py parses the site files with it
    return copy / "reproduce"


def _run_copy(stage_dir: Path, site: Path, *args: str) -> subprocess.CompletedProcess:
    e = dict(os.environ); e.pop("SISTEER_SITE_DIR", None); e.pop("SISTEER_CONFIG", None)
    e["SISTEER_STATE_DIR"] = str(site.parent / "state")
    return subprocess.run(["bash", str(stage_dir / "00_setup.sh"), *args, "--config", str(site / "config.env")], capture_output=True, text=True,
                          env=e, cwd=str(REPO), timeout=300)


def test_third_party_runs_the_script_once_per_cluster_and_marks_each_done(tmp_path):
    site = write_site(tmp_path, clusters=("alpha", "beta"))
    stage_dir = _copy_reproduce(tmp_path)
    calls = tmp_path / "third_party.calls"
    (stage_dir / "lib" / "third_party.sh").write_text(f'#!/bin/bash\necho "$*" >> {calls}\n')        # a stand-in that records its arguments
    r = _run_copy(stage_dir, site, "--what", "third_party")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert calls.read_text().splitlines() == [f"--cluster {lab} --config {site}/config.env" for lab in ("alpha", "beta")]
    for lab in ("alpha", "beta"):
        assert (site.parent / "state" / "setup" / "third_party" / f"DONE.{lab}").is_file()


def test_third_party_refuses_by_name_when_the_script_is_missing_or_fails(tmp_path):
    site = write_site(tmp_path, clusters=("alpha",))
    stage_dir = _copy_reproduce(tmp_path)
    tp = stage_dir / "lib" / "third_party.sh"
    tp.unlink(missing_ok=True)
    r = _run_copy(stage_dir, site, "--what", "third_party")
    assert r.returncode != 0 and "REFUSED: third_party:" in r.stderr and "third_party.sh is missing" in r.stderr, r.stderr[-1500:]
    assert not (site.parent / "state" / "setup" / "third_party").exists()
    tp.write_text("#!/bin/bash\necho 'digest mismatch' >&2\nexit 3\n")
    r = _run_copy(stage_dir, site, "--what", "third_party")
    assert r.returncode != 0 and "REFUSED: third_party: lib/third_party.sh failed for alpha (rc 3)" in r.stderr, r.stderr[-1500:]
    assert not (site.parent / "state" / "setup" / "third_party" / "DONE.alpha").exists()
