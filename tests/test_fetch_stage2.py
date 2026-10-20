"""Phase-0 readiness gate (scripts/fetch_stage2_data.py): the
"all provisioned" verdict must be honest — an undersized PG-19 filler or an
unverifiable LIFBench checkout FAILS (not WARN), and config-referenced model
snapshots are presence-checked at their pinned revisions without ever
auto-downloading weights. All offline: no item here touches the network."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fetch_stage2_data as fetch  # noqa: E402


def test_undersized_pg19_filler_fails_not_warns(tmp_path):
    """A short filler kills the 64k+/256k cells MID-ARRAY ('filler too
    short'), after the GPU time is spent — readiness must FAIL."""
    short = tmp_path / "pg19_filler.json"
    short.write_text(json.dumps({"text": "much too short"}))
    item = fetch.ensure_pg19_filler(short, on_cluster=True, offline=True)
    assert item.status == "FAILED"
    assert "filler too short" in item.detail

    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"text": "x" * 1_300_000}))
    assert fetch.ensure_pg19_filler(ok, True, True).status == "PRESENT"


def test_unparseable_filler_fails(tmp_path):
    bad = tmp_path / "pg19_filler.json"
    bad.write_text(json.dumps({"not_text": 1}))
    assert fetch.ensure_pg19_filler(bad, True, True).status == "FAILED"


def test_non_git_lifbench_checkout_fails_not_warns(tmp_path):
    """The held-out split and scorer mapping are only valid AT the pinned
    commit — an unverifiable (non-git) checkout must FAIL, never pass
    the readiness gate with a warning."""
    vendor = tmp_path / "lifbench"
    (vendor / "evaluation").mkdir(parents=True)
    (vendor / "evaluation" / "EvaluateFunc.py").write_text("# marker only\n")
    item = fetch.ensure_lifbench(vendor, "a" * 40, offline=True)
    assert item.status == "FAILED"
    assert "not a git checkout" in item.detail


def test_model_snapshot_presence_only_never_downloads(tmp_path, monkeypatch):
    """Model snapshots are verified at the config's pinned revision under
    the exact offline read path (local_files_only=True) and are NEVER
    fetched here (weights are tens of GB)."""
    import huggingface_hub

    calls = {}

    def fake_snapshot(repo_id, revision=None, local_files_only=False, **kw):
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        calls["local_files_only"] = local_files_only
        raise FileNotFoundError("not in the local cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    item = fetch.ensure_model_snapshot("org/some-8b-model", "b" * 40,
                                       "some_config.yaml", on_cluster=True)
    assert item.status == "FAILED"
    assert calls == {"repo_id": "org/some-8b-model", "revision": "b" * 40,
                     "local_files_only": True}
    assert "never auto-downloaded" in item.detail

    # off-cluster: a cluster readiness item, skipped — a dev box does not
    # need (or pull) the weights
    off = fetch.ensure_model_snapshot("org/some-8b-model", "b" * 40,
                                      "some_config.yaml", on_cluster=False)
    assert off.status == "SKIPPED"


def test_alpaca_pins_are_full_shas():
    """The alpaca pin exists and looks like a commit sha (drift protection)."""
    assert len(fetch.ALPACA_REVISION) == 40
    assert fetch.ALPACA_DATASET == "tatsu-lab/alpaca"


def test_removed_suites_items_are_gone():
    """The PersonaDrift, AxBench and IFEval-at-length items left with their
    suites; only the generic sentence-tokenizer NLTK data stays provisioned."""
    for attr in ("verify_axbench_runtime_read", "verify_alpaca_eval_runtime_read",
                 "ensure_wordlist", "REFT_R1_REPO", "REFT_R1_REVISION"):
        assert not hasattr(fetch, attr), attr
    assert set(fetch.NLTK_PACKAGES) == {"punkt_tab", "punkt"}


# --------------------------------------------------------------------------- #
# --only pg19-filler / --filler-out (docs/static_padding.md)                    #
# --------------------------------------------------------------------------- #

_OTHER_ITEMS = ("ensure_hf_snapshot", "verify_alpaca_runtime_read",
                "ensure_model_snapshot", "ensure_nltk")
#: the full run's item order
_FULL_ORDER = ["ensure_hf_snapshot", "verify_alpaca_runtime_read",
               "ensure_model_snapshot", "ensure_pg19_filler", "ensure_nltk"]


def _off_cluster(monkeypatch, tmp_path):
    """A dev box: the shared root is absent, so no HF_HOME hard-exit and every
    cluster-only item is a skip."""
    monkeypatch.setattr(fetch, "_shared_root", lambda: tmp_path / "no-such-root")


def _record_items(monkeypatch, names):
    calls = []

    def make(name):
        def fake(*a, **kw):
            calls.append((name, a, kw))
            return fetch.Item(name, "PRESENT")
        fake.__name__ = name
        return fake
    for n in names:
        monkeypatch.setattr(fetch, n, make(n))
    return calls


def _config_filler_path():
    return Path(fetch._cfg("stage2_longctx.yaml")["padding"]["filler_cache"])


def test_only_pg19_filler_runs_the_filler_item_alone(monkeypatch, tmp_path, capsys):
    _off_cluster(monkeypatch, tmp_path)
    calls = _record_items(monkeypatch, ("ensure_pg19_filler",))
    for n in _OTHER_ITEMS:   # any other item reached is a FAILED row -> exit 1
        monkeypatch.setattr(fetch, n, lambda *a, _n=n, **k: (_ for _ in ()).throw(
            AssertionError(f"{_n} must not run under --only pg19-filler")))
    monkeypatch.setattr(sys, "argv", ["fetch_stage2_data.py", "--only", "pg19-filler"])
    fetch.main()
    assert [c[0] for c in calls] == ["ensure_pg19_filler"]
    path, on_cluster, _offline = calls[0][1]
    assert path == _config_filler_path() and on_cluster is False
    out = capsys.readouterr().out
    assert "only: pg19-filler" in out and "all required items provisioned" in out


def test_filler_out_moves_the_cache(monkeypatch, tmp_path):
    _off_cluster(monkeypatch, tmp_path)
    calls = _record_items(monkeypatch, ("ensure_pg19_filler",))
    staged = tmp_path / "stage" / "pg19_filler.json"
    monkeypatch.setattr(sys, "argv", ["fetch_stage2_data.py", "--only", "pg19-filler",
                                      "--filler-out", str(staged)])
    fetch.main()
    assert [c[1][0] for c in calls] == [staged]


def test_filler_out_without_only_refuses(monkeypatch, tmp_path, capsys):
    _off_cluster(monkeypatch, tmp_path)
    _record_items(monkeypatch, ("ensure_pg19_filler",) + _OTHER_ITEMS)
    monkeypatch.setattr(sys, "argv", ["fetch_stage2_data.py", "--filler-out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit) as ei:
        fetch.main()
    assert ei.value.code == 2
    assert "--only pg19-filler" in capsys.readouterr().err


def test_unknown_only_refuses_naming_the_known_items(monkeypatch, tmp_path, capsys):
    _off_cluster(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["fetch_stage2_data.py", "--only", "nltk"])
    with pytest.raises(SystemExit) as ei:
        fetch.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'nltk'" in err and "pg19-filler" in err
    assert fetch.ONLY_ITEMS == ("pg19-filler",)


def test_default_run_is_unchanged_every_item_in_order(monkeypatch, tmp_path, capsys):
    """No flag: every item in its fixed order, the filler at the config's path."""
    _off_cluster(monkeypatch, tmp_path)
    calls = _record_items(monkeypatch, ("ensure_pg19_filler",) + _OTHER_ITEMS)
    monkeypatch.setattr(sys, "argv", ["fetch_stage2_data.py"])
    fetch.main()
    assert [c[0] for c in calls] == _FULL_ORDER
    filler_call = next(c for c in calls if c[0] == "ensure_pg19_filler")
    assert filler_call[1][0] == _config_filler_path()
    out = capsys.readouterr().out
    assert "only:" not in out and "all required items provisioned" in out


def test_filler_only_keeps_the_cluster_hf_home_hard_exit(monkeypatch, tmp_path):
    """A filler BUILD is an HF download (the pinned PG-19 parquet via hf://), so
    the cluster's HF_HOME refusal holds for --only pg19-filler too."""
    monkeypatch.setattr(fetch, "_shared_root", lambda: tmp_path)   # a dir: "on the cluster"
    monkeypatch.delenv("HF_HOME", raising=False)
    _record_items(monkeypatch, ("ensure_pg19_filler",))
    monkeypatch.setattr(sys, "argv", ["fetch_stage2_data.py", "--only", "pg19-filler"])
    with pytest.raises(SystemExit, match="HF_HOME is unset"):
        fetch.main()


def test_the_shared_root_is_the_sites(monkeypatch, tmp_path):
    """Whether this host is on the cluster is read from the site's $SHARED (the environment's SISTEER_SHARED_ROOT, else
    the site config's cluster block), never from a storage path written into the script; with nothing naming one the
    run is off the cluster."""
    for var in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT", "SISTEER_CLUSTER_LABEL", "SISTEER_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    site = tmp_path / "site"
    monkeypatch.setenv("SISTEER_SITE_DIR", str(site))
    assert fetch._shared_root() is None
    (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text("SISTEER_CLUSTERS=cluster_a\nSISTEER_PRIMARY_CLUSTER=cluster_a\n")
    (site / "clusters" / "cluster_a.env").write_text(f"CLUSTER_SHARED_ROOT={tmp_path / 'shared' / 'u'}\n")
    assert fetch._shared_root() == tmp_path / "shared" / "u"
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "env-root"))
    assert fetch._shared_root() == tmp_path / "env-root"


def test_slurm_host_reads_slurms_own_markers(monkeypatch):
    """A Slurm host is one inside a job (SLURM_JOB_ID or SLURM_CLUSTER_NAME set) or one with ``sbatch`` on PATH (a login
    node); with none of the three it is not one."""
    for var in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(fetch.shutil, "which", lambda name, *a, **k: None)
    assert fetch._slurm_host() is False
    monkeypatch.setattr(fetch.shutil, "which", lambda name, *a, **k: "/usr/bin/sbatch" if name == "sbatch" else None)
    assert fetch._slurm_host() is True
    monkeypatch.setattr(fetch.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster_a")
    assert fetch._slurm_host() is True


@pytest.mark.parametrize("filler_out", [False, True])
def test_a_slurm_host_naming_no_shared_root_refuses_before_any_item(monkeypatch, tmp_path, filler_out):
    """On a host that runs Slurm, a run where nothing names the shared root (no SISTEER_SHARED_ROOT, no site config) stops
    by name before any item runs: taking that host for a dev box would skip the HF_HOME refusal and let a filler build
    download onto node-local disk. --filler-out reads no config path, so nothing else stops that run on the way."""
    monkeypatch.setattr(fetch, "_shared_root", lambda: None)
    monkeypatch.setattr(fetch, "_slurm_host", lambda: True, raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    calls = _record_items(monkeypatch, ("ensure_pg19_filler",))
    argv = ["fetch_stage2_data.py", "--only", "pg19-filler"]
    if filler_out:
        argv += ["--filler-out", str(tmp_path / "stage" / "pg19_filler.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="names no shared storage root.*SISTEER_SHARED_ROOT"):
        fetch.main()
    assert calls == []


def test_a_named_root_decides_on_or_off_the_cluster_and_no_root_off_slurm_is_a_dev_box(monkeypatch, tmp_path):
    """The site's root decides where one is named: absent here (the grading host, whose site config names a cluster's root)
    is off the cluster even on a Slurm host, present here is on it. Off Slurm with nothing named is a dev box, as before."""
    staged = tmp_path / "stage" / "pg19_filler.json"
    argv = ["fetch_stage2_data.py", "--only", "pg19-filler", "--filler-out", str(staged)]
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for root, slurm, on in ((tmp_path / "absent", True, False), (tmp_path, True, True), (None, False, False)):
        monkeypatch.setattr(fetch, "_shared_root", lambda r=root: r)
        monkeypatch.setattr(fetch, "_slurm_host", lambda s=slurm: s, raising=False)
        calls = _record_items(monkeypatch, ("ensure_pg19_filler",))
        monkeypatch.setattr(sys, "argv", argv)
        fetch.main()
        assert [c[1][:2] for c in calls] == [(staged, on)], (root, slurm)
