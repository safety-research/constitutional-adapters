"""HF_HUB_OFFLINE smoke for every Tier-0/1 dataset.

Codes against the pins.yaml SHAPE frozen in scripts/capeval_fetch_data.py's
header (that script is the one writer; this file is a reader):

    datasets.<pin_name>: {repo_id, revision (40-hex), patterns, files,
                          bytes, verify: parquet|json|jsonl|jsonl_gz}
    evalplus.humaneval_plus: {version, url, cache_file, sha256, bytes, n_tasks}
    models.llama31_8b_it: {repo_id, revision}

Every dataset is re-read the way the runtime reads it — a FRESH
subprocess with HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1 (the offline
flags are read at import time, so an in-process env tweak would prove
nothing), snapshot via local_files_only, file selection by the pin's
patterns exactly as corpora.pinned_snapshot_files does it (a shared cache
legitimately holds NON-pinned files of the same repo revision, which the
runtime never reads), and a direct pyarrow/json/jsonl read (the S2-proven
pattern; never load_dataset on cluster). EvalPlus's
problem file is verified against the pinned sha256 with no network at
all. Skips are LOUD: they name the fetch script / lane that must run
first.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
PINS_PATH = ROOT / "configs" / "capeval" / "pins.yaml"

# Every pin the tiers touch (fetch script DATASET_PINS keys): Tier-0
# corpora + canary, parity probes, the served IFEval input.
REQUIRED_DATASET_PINS = {
    "wikitext103", "stack_py", "ultrachat", "canary_prompts",
    "gsm8k_platinum", "mmlu", "ifeval",
    # capeval-controls judged tier (§3.3)
    "or_bench", "xstest_v2",
}
# mirrors capeval_fetch_data.verify_dataset_offline's kinds (jsonl_gz =
# the-stack-smol's *.json.gz shards; csv = or-bench-hard-1k)
VERIFY_KINDS = {"parquet", "json", "jsonl", "jsonl_gz", "csv"}

if not PINS_PATH.is_file():
    pytest.skip(
        f"{PINS_PATH} absent — run `uv run python scripts/capeval_fetch_data.py` "
        "(WS-D, Phase 0) first; the offline checks are only meaningful after "
        "fetch has resolved + frozen the pins (plan §13/§15 R6)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def pins():
    return yaml.safe_load(PINS_PATH.read_text())


def test_pins_shape_and_coverage(pins):
    datasets = pins.get("datasets") or {}
    missing = REQUIRED_DATASET_PINS - set(datasets)
    assert not missing, (
        f"pins.yaml is missing dataset pins {sorted(missing)} — re-run "
        "scripts/capeval_fetch_data.py (its DATASET_PINS list is the source "
        "of truth and covers every tier dataset)"
    )
    for name in REQUIRED_DATASET_PINS:
        entry = datasets[name]
        assert set(entry) >= {"repo_id", "revision", "verify"}, (name, entry)
        rev = entry["revision"]
        assert isinstance(rev, str) and len(rev) == 40 and int(rev, 16) >= 0, (
            f"{name}: revision must be a full 40-hex SHA, got {rev!r}"
        )
        assert entry["verify"] in VERIFY_KINDS, (name, entry["verify"])


def test_pins_model_revision_matches_model_config(pins):
    model_cfg = yaml.safe_load(
        (ROOT / "configs/capeval/models/llama31_8b_it.yaml").read_text()
    )
    entry = (pins.get("models") or {}).get("llama31_8b_it")
    assert entry, "pins.yaml missing models.llama31_8b_it"
    assert entry["revision"] == model_cfg["model_revision"]
    assert entry["repo_id"] == model_cfg["model"]


# ---------------------------------------------------------------------------
# The offline re-read, in a fresh subprocess (offline env must predate
# imports). One subprocess for all pins; per-pin OK/MISSING/FAIL lines.
# ---------------------------------------------------------------------------

_OFFLINE_READER = r"""
import fnmatch, glob, json, os, sys
import yaml

pins = yaml.safe_load(open(sys.argv[1]))
names = sys.argv[2].split(",")
# sisteer.capeval.corpora._VERIFY_SUFFIXES, mirrored (this reader must stay
# import-free of sisteer so the offline env flags predate every import)
SUFFIXES = {"parquet": (".parquet",), "json": (".json",),
            "jsonl": (".jsonl",), "jsonl_gz": (".json.gz", ".jsonl.gz"),
            "csv": (".csv",)}
for name in names:
    entry = pins["datasets"][name]
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(
            repo_id=entry["repo_id"], repo_type="dataset",
            revision=entry["revision"],
            allow_patterns=entry.get("patterns"),
            local_files_only=True,
        )
    except Exception as e:
        print(f"MISSING {name} {type(e).__name__}: {e}")
        continue
    try:
        kind = entry["verify"]
        # File selection mirrors the RUNTIME loader
        # (corpora.pinned_snapshot_files): suffix for the verify kind AND
        # the pin's fnmatch patterns against the snapshot-relative path.
        # allow_patterns above only limits what a DOWNLOAD would fetch —
        # with local_files_only the snapshot dir contains whatever the
        # shared cache already holds for this revision (e.g. or-bench-80k
        # fetched by unrelated work next to the pinned or-bench-hard-1k),
        # and the runtime reads only the pattern-matched files, so this
        # re-read must too.
        pats = entry.get("patterns")
        files = []
        for f in sorted(glob.glob(os.path.join(snap, "**", "*"),
                                  recursive=True)):
            if not os.path.isfile(f) or os.path.basename(f).startswith("."):
                continue
            rel = os.path.relpath(f, snap).replace(os.sep, "/")
            if not any(rel.endswith(s) for s in SUFFIXES[kind]):
                continue
            if pats and not any(fnmatch.fnmatch(rel, p) for p in pats):
                continue
            files.append(f)
        if not files:
            raise FileNotFoundError(f"no {kind} files under snapshot")
        if kind == "parquet":
            import pyarrow.parquet as pq
            n = pq.read_table(files[0]).num_rows
        elif kind == "csv":
            import pandas as pd
            n = len(pd.read_csv(files[0]))
        elif kind == "json":
            data = json.load(open(files[0]))
            n = len(data)
        elif kind == "jsonl_gz":
            import gzip
            with gzip.open(files[0], "rt") as f:
                n = sum(1 for _ in zip(range(3), f) if json.loads(_[1]) is not None)
        else:
            with open(files[0]) as f:
                n = sum(1 for line in f if line.strip() and json.loads(line) is not None)
        if n <= 0:
            raise ValueError("zero rows")
        print(f"OK {name} {os.path.basename(files[0])} rows={n}")
    except Exception as e:
        print(f"FAIL {name} {type(e).__name__}: {e}")
"""


def test_offline_direct_read_every_pin(pins):
    names = sorted(REQUIRED_DATASET_PINS)
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", _OFFLINE_READER, str(PINS_PATH), ",".join(names)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert proc.returncode == 0, f"offline reader crashed:\n{proc.stderr[-3000:]}"
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    status = {l.split()[1]: l for l in lines}
    assert set(status) == set(names), (status, names)
    missing = [l for l in lines if l.startswith("MISSING")]
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, (
        "pins present but UNREADABLE offline (corrupt cache or wrong verify "
        "kind):\n" + "\n".join(failed)
    )
    if missing:
        pytest.skip(
            "pins resolved but snapshots absent from THIS machine's HF cache — "
            "run scripts/capeval_fetch_data.py here (the grading host) or source the "
            "shared hf_env.sh on the cluster before the login-node check "
            "(plan §15 R6):\n" + "\n".join(missing)
        )
    # capeval-controls count gates: the judged-tier pins re-read
    # offline with EXACTLY the expected row counts (1,319 / 450)
    for name, expected in (("or_bench", 1319), ("xstest_v2", 450)):
        line = status[name]
        assert line.endswith(f"rows={expected}"), (
            f"{name}: expected {expected} rows at the pin, got: {line}")


# ---------------------------------------------------------------------------
# fetch-script offline semantics (review infra #7): the frozen pins.yaml
# entry is the truth the cache is checked AGAINST — never re-derived from
# a possibly half-cached snapshot. Unit-level, stubbed snapshot_download.
# ---------------------------------------------------------------------------


def _fetch_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "capeval_fetch_data", ROOT / "scripts" / "capeval_fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capeval_fetch_data"] = mod  # dataclasses resolve __module__
    spec.loader.exec_module(mod)
    return mod


def _snapshot_dir(tmp_path, files):
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    for name, size in files.items():
        (snap / name).write_bytes(b"x" * size)
    return snap


def test_ensure_dataset_offline_preserves_frozen_entry(tmp_path, monkeypatch):
    """Review infra #7: offline mode returns the FROZEN entry (never a
    recomputed one) when the cache matches it, and FAILS when the cache
    is a half-cached subset — instead of overwriting the registry with
    the smaller truth."""
    mod = _fetch_mod()
    import huggingface_hub

    pin = mod.DatasetPin("mmlu", "cais/mmlu", ["all/validation-*"],
                         verify="parquet")
    prior = {"repo_id": "cais/mmlu", "revision": "c" * 40,
             "patterns": ["all/validation-*"], "files": 2, "bytes": 20,
             "verify": "parquet"}
    snap = _snapshot_dir(tmp_path, {"a.parquet": 10, "b.parquet": 10})
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **k: str(snap))
    report = mod.Report()
    entry = mod.ensure_dataset(pin, {"mmlu": prior}, offline=True,
                               report=report)
    assert entry == prior  # the frozen entry, verbatim — never recomputed
    assert report.items[-1].status == "PRESENT"
    # HALF-CACHED: one shard missing -> FAILED, frozen entry untouched
    (snap / "b.parquet").unlink()
    report2 = mod.Report()
    entry2 = mod.ensure_dataset(pin, {"mmlu": dict(prior)}, offline=True,
                                report=report2)
    assert entry2 is None
    assert report2.items[-1].status == "FAILED"
    assert "HALF-CACHED" in report2.items[-1].detail


def test_ensure_dataset_offline_requires_a_frozen_entry(tmp_path, monkeypatch):
    mod = _fetch_mod()
    pin = mod.DatasetPin("mmlu", "cais/mmlu", None, revision="c" * 40)
    report = mod.Report()
    assert mod.ensure_dataset(pin, {}, offline=True, report=report) is None
    assert report.items[-1].status == "FAILED"
    assert "frozen pins.yaml entry" in report.items[-1].detail


def test_verify_dataset_offline_opens_every_payload(tmp_path, monkeypatch):
    """Review infra #7: every payload shard is opened — a corrupt/short
    second shard fails even when the first reads fine — and the file
    count/bytes are checked against the frozen entry."""
    mod = _fetch_mod()
    import huggingface_hub

    import pyarrow as pa
    import pyarrow.parquet as pq

    snap = tmp_path / "snap"
    snap.mkdir()
    table = pa.table({"x": [1, 2, 3]})
    pq.write_table(table, snap / "a.parquet")
    pq.write_table(table, snap / "b.parquet")
    sizes = sum(p.stat().st_size for p in snap.iterdir())
    entry = {"repo_id": "cais/mmlu", "revision": "c" * 40, "patterns": None,
             "files": 2, "bytes": sizes, "verify": "parquet"}
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **k: str(snap))
    report = mod.Report()
    assert mod.verify_dataset_offline("mmlu", entry, report) is True
    assert "2 payload file(s)" in report.items[-1].detail
    # corrupt the SECOND shard: a single-representative read would pass
    (snap / "b.parquet").write_bytes(b"not parquet")
    entry2 = dict(entry, bytes=sum(p.stat().st_size for p in snap.iterdir()))
    report2 = mod.Report()
    assert mod.verify_dataset_offline("mmlu", entry2, report2) is False
    assert report2.items[-1].status == "FAILED"
    # count/bytes drift against the frozen entry also fails
    report3 = mod.Report()
    assert mod.verify_dataset_offline(
        "mmlu", dict(entry, files=3), report3) is False
    assert "frozen entry" in report3.items[-1].detail


def test_offline_evalplus_cache_read(pins):
    """No-network read of the evalplus problem file (§5.5): the pinned
    sha256 of the DECOMPRESSED jsonl, the task count, and the cache
    location evalplus 0.3.1 actually reads (appdirs user_cache_dir ->
    $XDG_CACHE_HOME/evalplus or ~/.cache/evalplus; data/utils.py:11)."""
    entry = (pins.get("evalplus") or {}).get("humaneval_plus")
    assert entry, "pins.yaml missing evalplus.humaneval_plus"
    assert entry["version"] == "v0.1.10"  # HUMANEVAL_PLUS_VERSION at 0.3.1
    xdg = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg) if xdg else Path.home() / ".cache"
    cache_file = cache_root / "evalplus" / entry["cache_file"]
    if not cache_file.is_file():
        pytest.skip(
            f"{cache_file} absent — run scripts/capeval_fetch_data.py (WS-D) "
            "to pre-seed the evalplus cache on this machine (plan §5.5); the "
            "sandbox side is seeded by capeval_sandbox.sh provision"
        )
    blob = cache_file.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == entry["sha256"], (
        f"{cache_file} sha256 mismatch vs pins.yaml — the cached problem set "
        "is not the pinned one"
    )
    lines = [l for l in blob.decode().splitlines() if l.strip()]
    assert len(lines) == entry["n_tasks"] == 164
    first = json.loads(lines[0])
    assert {"task_id", "prompt", "entry_point"} <= set(first)


# ---------------------------------------------------------------------------
# A fresh machine, end to end: the online pass fetches every declared pin into
# an empty cache and writes pins.yaml, then the offline re-read checks every
# entry the file lists. The hub, the release download and the revision lookup
# are stubs, and the re-read runs in this process (the script starts it as a
# subprocess, which the stubs would not reach).
# ---------------------------------------------------------------------------


def test_pins_file_lists_exactly_the_declared_datasets(pins):
    """The offline re-read checks every dataset entry pins.yaml lists, and the
    online pass fetches only the ones DATASET_PINS declares: an entry no pin
    declares is never fetched, so a machine with an empty cache fails on it."""
    mod = _fetch_mod()
    declared = {p.name for p in mod.DATASET_PINS}
    listed = set(pins.get("datasets") or {})
    assert listed == declared, (
        f"pins.yaml lists {sorted(listed - declared)} that DATASET_PINS does not declare, and "
        f"DATASET_PINS declares {sorted(declared - listed)} that pins.yaml lacks")


def _fresh_machine(tmp_path, monkeypatch, *, extra_entries=None):
    """The fetch script wired to an empty stub hub under ``tmp_path``, starting
    from the committed pins.yaml's dataset entries (plus ``extra_entries``).
    Returns ``(mod, fetched)``, where ``fetched`` lists every (repo_id,
    revision) the stub hub served for download."""
    import gzip
    import types

    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    mod = _fetch_mod()
    committed = yaml.safe_load(PINS_PATH.read_text())
    repo = tmp_path / "repo"
    (repo / "configs" / "capeval").mkdir(parents=True)
    # a synthetic HumanEval+ release: the committed evalplus pin carries the
    # real file's sha256, which a stub cannot serve
    tasks = b"".join(
        json.dumps({"task_id": f"HumanEval/{i}", "prompt": "def f():\n", "entry_point": "f"}).encode() + b"\n"
        for i in range(mod.EVALPLUS_N_TASKS))
    start = {
        "datasets": dict(committed["datasets"], **(extra_entries or {})),
        "evalplus": {"humaneval_plus": dict(committed["evalplus"]["humaneval_plus"],
                                            sha256=hashlib.sha256(tasks).hexdigest(), bytes=len(tasks))},
        "models": committed["models"],
    }
    pins_path = repo / "configs" / "capeval" / "pins.yaml"
    pins_path.write_text(yaml.safe_dump(start, sort_keys=True))
    monkeypatch.setattr(mod, "REPO", repo)
    monkeypatch.setattr(mod, "PINS_PATH", pins_path)
    monkeypatch.setattr(mod, "MIN_FREE_BYTES", 0)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    # the grading host, wherever the test runs (a test node runs Slurm, which a run naming no shared root refuses)
    monkeypatch.setattr(mod, "_slurm_host", lambda: False, raising=False)
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)

    by_repo = {p.repo_id: p for p in mod.DATASET_PINS}
    hub = tmp_path / "hub"
    fetched = []

    def snapshot_download(repo_id, repo_type=None, revision=None, allow_patterns=None,
                          local_files_only=False):
        if repo_type != "dataset":  # the model presence check, which runs on cluster hosts only
            (tmp_path / "model").mkdir(exist_ok=True)
            return str(tmp_path / "model")
        snap = hub / repo_id.replace("/", "--") / str(revision)
        if local_files_only:
            if not snap.is_dir():
                raise FileNotFoundError(f"{repo_id} at {revision} is not in the local cache")
            return str(snap)
        assert os.environ.get("HF_HUB_OFFLINE") != "1", f"a download of {repo_id} with the network forbidden"
        pin = by_repo[repo_id]
        rows = [f"row {i}" for i in range(mod.FETCH_COUNT_GATES.get(pin.name, 4))]
        snap.mkdir(parents=True, exist_ok=True)
        if pin.verify == "parquet":
            pq.write_table(pa.table({"text": rows}), snap / "part-0.parquet")
        elif pin.verify == "csv":
            (snap / "rows.csv").write_text("text\n" + "".join(f"{r}\n" for r in rows))
        elif pin.verify == "json":
            (snap / "rows.json").write_text(json.dumps([{"text": r} for r in rows]))
        elif pin.verify == "jsonl":
            (snap / "rows.jsonl").write_text("".join(json.dumps({"text": r}) + "\n" for r in rows))
        else:  # jsonl_gz
            (snap / "rows.json.gz").write_bytes(
                gzip.compress("".join(json.dumps({"text": r}) + "\n" for r in rows).encode()))
        fetched.append((repo_id, revision))
        return str(snap)

    def urlretrieve(url, dest):
        assert url == mod.EVALPLUS_URL
        Path(dest).write_bytes(gzip.compress(tasks))

    def resolve_revision(repo_id):
        raise AssertionError(f"{repo_id}: every declared pin has a frozen revision, so nothing is resolved")

    def offline_subprocess(argv, env):
        assert argv[-1] == "--offline" and env.get("HF_HUB_OFFLINE") == "1"
        with monkeypatch.context() as m:
            m.setenv("HF_HUB_OFFLINE", "1")
            m.setenv("HF_DATASETS_OFFLINE", "1")
            return types.SimpleNamespace(returncode=mod.run(offline=True, skip_verify=False))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(mod.urllib.request, "urlretrieve", urlretrieve)
    monkeypatch.setattr(mod, "_resolve_revision", resolve_revision)
    monkeypatch.setattr(mod, "subprocess", types.SimpleNamespace(run=offline_subprocess))
    return mod, fetched


def test_fresh_machine_provisioning_passes_end_to_end(tmp_path, monkeypatch, capsys):
    """No HF cache and no evalplus cache: the online pass fetches every
    declared pin at its frozen revision and writes pins.yaml, the offline
    re-read of every entry the file lists passes, and the run exits 0."""
    mod, fetched = _fresh_machine(tmp_path, monkeypatch)
    committed = yaml.safe_load(PINS_PATH.read_text())["datasets"]
    rc = mod.run(offline=False, skip_verify=False)
    out = capsys.readouterr().out
    assert rc == 0, out[-4000:]
    assert "[PRESENT] offline re-read --" in out
    declared = {p.name for p in mod.DATASET_PINS}
    written = yaml.safe_load(mod.PINS_PATH.read_text())["datasets"]
    assert set(written) == declared
    assert all(written[n]["revision"] == committed[n]["revision"] for n in declared)
    assert sorted(fetched) == sorted((p.repo_id, committed[p.name]["revision"]) for p in mod.DATASET_PINS)


def test_provisioning_refuses_an_entry_no_pin_declares(tmp_path, monkeypatch, capsys):
    """An entry pins.yaml lists but DATASET_PINS does not declare is refused by
    name before anything is fetched or written, rather than surfacing later as
    a cache miss in the offline re-read."""
    stray = {"repo_id": "example/retired-set", "revision": "0" * 40, "patterns": None,
             "files": 1, "bytes": 1, "verify": "parquet"}
    mod, fetched = _fresh_machine(tmp_path, monkeypatch, extra_entries={"retired_set": stray})
    before = mod.PINS_PATH.read_bytes()
    rc = mod.run(offline=False, skip_verify=False)
    out = capsys.readouterr().out
    assert rc == 1, out[-4000:]
    assert "['retired_set']" in out and "does not declare" in out
    assert fetched == [] and mod.PINS_PATH.read_bytes() == before


def test_the_shared_root_is_the_sites(monkeypatch, tmp_path):
    """The provisioning run decides it is on the cluster from the site's $SHARED (the environment's SISTEER_SHARED_ROOT,
    else the site config's cluster block), never from a storage path written into the script; with nothing naming one
    the run is off the cluster."""
    mod = _fetch_mod()
    for var in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT", "SISTEER_CLUSTER_LABEL", "SISTEER_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    site = tmp_path / "site"
    monkeypatch.setenv("SISTEER_SITE_DIR", str(site))
    assert mod._shared_root() is None
    (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text("SISTEER_CLUSTERS=cluster_a\nSISTEER_PRIMARY_CLUSTER=cluster_a\n")
    (site / "clusters" / "cluster_a.env").write_text(f"CLUSTER_SHARED_ROOT={tmp_path / 'shared' / 'u'}\n")
    assert mod._shared_root() == tmp_path / "shared" / "u"
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "env-root"))
    assert mod._shared_root() == tmp_path / "env-root"


def test_a_slurm_host_naming_no_shared_root_refuses_before_any_fetch(tmp_path, monkeypatch):
    """On a host that runs Slurm, a run where nothing names the shared root stops by name before anything is fetched or
    written: taking that host for the grading host would skip the HF_HOME refusal, download into a node-local cache and
    report the model check as a skip."""
    mod, fetched = _fresh_machine(tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "_shared_root", lambda: None)
    monkeypatch.setattr(mod, "_slurm_host", lambda: True, raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    before = mod.PINS_PATH.read_bytes()
    with pytest.raises(SystemExit, match="names no shared storage root.*SISTEER_SHARED_ROOT"):
        mod.run(offline=False, skip_verify=False)
    assert fetched == [] and mod.PINS_PATH.read_bytes() == before


def test_slurm_host_reads_slurms_own_markers(monkeypatch):
    """A Slurm host is one inside a job (SLURM_JOB_ID or SLURM_CLUSTER_NAME set) or one with ``sbatch`` on PATH; with none of
    the three it is not one."""
    mod = _fetch_mod()
    for var in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod.shutil, "which", lambda name, *a, **k: None)
    assert mod._slurm_host() is False
    monkeypatch.setattr(mod.shutil, "which", lambda name, *a, **k: "/usr/bin/sbatch" if name == "sbatch" else None)
    assert mod._slurm_host() is True
    monkeypatch.setattr(mod.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    assert mod._slurm_host() is True
