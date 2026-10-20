"""The probe record's HOME (03_hpo.sh probe_home; the LoRA HPO lane, 2026-09-30): the serving probe's menu is fetched from the
cluster the probe RAN ON, while the recon / design / trials of the same class may run on ANOTHER cluster. Every phase that consumes
the menu revalidates it against the record WHERE IT LIVES (probe_fetch.json's cluster + record), never against --cluster's copy of
the path — which on the other cluster is an empty directory that hpo_probe_status reads NOT_STARTED, so the revalidation used to
archive the current menu + the DONE marker and refuse. The probe PHASE itself still acts on --cluster's record and refuses by name
when the fetched menu's record lives elsewhere. Executed against the REAL stage with the probe harness (fake ssh/rsync/squeue that
run the remote string on this host; the real status reader), two neutral clusters, no network."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, write_site  # noqa: E402
from test_front_door_probe_resume import HEAD, MODEL, OVERRIDES, REAL_RSYNC, World, _stamp  # noqa: E402


@pytest.fixture
def worlds(tmp_path):
    """The probe harness with a SECOND cluster: the record lives on cluster_a (World's), cluster_b has the same layout and no record."""
    if REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    w = World(tmp_path)
    fake_b = tmp_path / "fake_b"; (fake_b / "jobs").mkdir(parents=True)
    shared_b = tmp_path / "shared_b"; (shared_b / "sisteer-eval-repo" / HEAD).mkdir(parents=True)
    write_site(tmp_path, clusters=("cluster_a", "cluster_b"), ssh={"cluster_a": f"local:{w.fake}", "cluster_b": f"local:{fake_b}"},
               shared={"cluster_a": str(w.shared), "cluster_b": str(shared_b)}, extra_config=f"SISTEER_RUN_OVERRIDES={OVERRIDES}\n")
    w.fake_b = fake_b; w.shared_b = shared_b
    return w


def _run(w: World, *args: str) -> subprocess.CompletedProcess:
    argv = ["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), *args]
    return subprocess.run(argv, capture_output=True, text=True, env=w.env(), cwd=str(REPO), timeout=600)


def _record_reads(w: World) -> list[tuple[str, str]]:
    """(host, remote command) of every fake-ssh call that read a probe record."""
    return [tuple(ln.split("\t", 1)) for ln in w.ssh_log.read_text().splitlines() if "--record" in ln]


def _fetch_from_cluster_a(w: World) -> str:
    w.lay(done=7, complete=True, menu=True, verdict=True)
    gen = _stamp(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "READY on cluster_a" in out and (w.state / "allowed_pairs.json").is_file(), out
    fetch = json.loads((w.state / "probe_fetch.json").read_text())
    assert fetch["cluster"] == "cluster_a" and fetch["record"] == str(w.record) and fetch["generation"] == gen, fetch
    assert (w.state.parent / "DONE.lora.probe").is_file()
    w.ssh_log.write_text("")                                  # the fetch's own reads are not under test
    return gen


def test_a_consuming_phase_revalidates_the_menu_against_the_record_where_it_lives(worlds):
    """The menu fetched from cluster_a; `--phase recon --cluster cluster_b`: the record read goes to cluster_a's record (the fetch
    record's home), the revalidation passes, NOTHING local is archived, and the phase runs on until its own cluster_b-side input (the
    recon cell's pairs file, absent in this world) refuses by name — never 'NOT the record's current result generation'."""
    w = worlds
    gen = _fetch_from_cluster_a(w)
    r = _run(w, "--phase", "recon", "--cluster", "cluster_b")
    out = r.stdout + r.stderr
    reads = _record_reads(w)
    assert reads and all(h == f"local:{w.fake}" and str(w.record) in cmd for h, cmd in reads), (reads, out)   # cluster_a, its record dir
    assert not any(str(w.shared_b) in cmd for _h, cmd in reads), reads                                        # never cluster_b's copy of the path
    assert "the menu's record lives on cluster_a:" in out and "recon runs on cluster_b" in out, out
    assert f"current result generation ({gen[:12]}; record cluster_a:{w.record}) — recon proceeds" in out, out
    assert "NOT the record's current result generation" not in out and "archived under" not in out, out
    assert (w.state / "allowed_pairs.json").is_file() and (w.state.parent / "DONE.lora.probe").is_file() and not (w.state / "superseded").exists()
    # past the revalidation: the recon composes its spec and stops on cluster_b's missing recon cell — its own refusal, by name
    assert r.returncode != 0 and "cannot read the sha256 of the recon cell's pairs file" in out and "on cluster_b" in out, out
    assert not list((w.fake_b / "jobs").glob("*.argv")), "nothing submitted on cluster_b"


def test_the_probe_phase_on_another_cluster_than_the_records_refuses_by_name_and_touches_nothing(worlds):
    """`--phase probe --cluster cluster_b` with a menu fetched from cluster_a: refused by name (the probe phase runs where its record
    lives), no record read anywhere, nothing archived, nothing submitted — never a fresh probe over cluster_b's empty path."""
    w = worlds
    _fetch_from_cluster_a(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_b")
    out = r.stdout + r.stderr
    assert r.returncode == 2 and "record lives on cluster_a:" in r.stderr and "--phase probe --cluster cluster_a" in r.stderr, out
    assert _record_reads(w) == [] and "archived under" not in out, (w.ssh_log.read_text(), out)
    assert (w.state / "allowed_pairs.json").is_file() and (w.state.parent / "DONE.lora.probe").is_file() and not (w.state / "superseded").exists()
    assert not list((w.fake_b / "jobs").glob("*.argv")) and not list((w.fake / "jobs").glob("*.argv"))
    # the SAME cluster is still the ordinary path: a DONE-and-current record is skipped, nothing re-submitted
    r2 = _run(w, "--phase", "probe", "--cluster", "cluster_a")
    assert r2.returncode == 0 and "DONE marker present and the local menu is the record's current result generation" in r2.stdout, r2.stdout + r2.stderr


def test_the_dry_run_names_the_records_home_and_the_status_reader_targets_it(worlds):
    """A dry run states which record the live revalidation would read (the home, not --cluster); and the source pins: probe_status and
    the revalidation read PROBE_CL_HOST / PROBE_CL_ROOT, the probe cluster's queue through probe_job_active."""
    w = worlds
    _fetch_from_cluster_a(w)
    r = _run(w, "--phase", "design", "--cluster", "cluster_b", "--dry-run")
    out = r.stdout + r.stderr
    assert f"hpo_probe_status.py --record {w.record} on cluster_a before design" in out, out
    src = (REPRO / "03_hpo.sh").read_text()
    ps = src[src.index("probe_status(){"):src.index("probe_fetch_ready(){")]
    # the record read defaults to the probe record's home and takes the parity-pass record as positional 2
    assert 'local root="${2:-$PROBE_CL_ROOT}"' in ps and "--record '$root'" in ps and '"$PROBE_CL_HOST"' in ps and "'$CL_ROOT'" not in ps, ps
    rv = src[src.index("probe_revalidate_local(){"):src.index("# ====", src.index("probe_revalidate_local(){"))]
    assert "probe_home" in rv and 'probe_job_active "$(name probe)"' in rv and 'cluster_job_active "$(name probe)"' not in rv, rv
