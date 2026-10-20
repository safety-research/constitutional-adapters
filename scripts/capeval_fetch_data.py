"""capeval Phase-0 data provisioning: resolve + fetch + verify + PIN.

Mirrors scripts/fetch_stage2_data.py: idempotent, loud, offline-aware. The
one script that makes every Tier-0/1 dataset dependency PRESENT (compute
nodes run HF_HUB_OFFLINE=1 with no egress; anything missed here dies
mid-array after the GPU time is spent) — and the one WRITER of
``configs/capeval/pins.yaml``, the pin registry corpora.py and the consumers
(test_capeval_offline.py) code against.

pins.yaml SHAPE (frozen by this header; the file itself says do-not-edit):

    resolved_at: "<iso8601>"          # last time a pin was (re)resolved
    datasets:
      <pin_name>:                     # keys of DATASET_PINS below
        repo_id: <hub id>
        revision: <full 40-hex commit SHA>   # resolved once, then FROZEN
        patterns: [<allow_patterns used>]    # null = whole repo
        files: <int>                  # files matched at fetch time
        bytes: <int>                  # total bytes of those files
        verify: parquet|json|jsonl|jsonl_gz|csv   # how the offline re-read checks it
    evalplus:
      humaneval_plus:
        version: v0.1.10              # evalplus 0.3.1's HUMANEVAL_PLUS_VERSION
        url: <github release url>
        cache_file: HumanEvalPlus-v0.1.10.jsonl   # under $XDG_CACHE_HOME/evalplus
        sha256: <hex of the DECOMPRESSED jsonl>
        bytes: <int>
        n_tasks: 164
    models:
      llama31_8b_it:
        repo_id: meta-llama/Llama-3.1-8B-Instruct
        revision: <from configs/capeval/models/llama31_8b_it.yaml>
        # presence verified on the cluster only; never auto-downloaded here

Re-runs: an existing pins.yaml FREEZES every revision — a hub `main` that
has moved past a pinned SHA is fine (we fetch BY SHA); a pinned SHA that
can no longer be fetched/read is FAILED, never silently re-pinned.
Verification is a fresh SUBPROCESS with HF_HUB_OFFLINE=1 +
HF_DATASETS_OFFLINE=1 re-reading every pin the way the runtime does
(hf_hub_download(local_files_only) + direct pyarrow/json reads — the
S2-proven pattern).

pins.yaml lists exactly the datasets DATASET_PINS declares, and a run
refuses an entry no pin declares: nothing would fetch it, and the offline
re-read checks every entry the file lists, so a machine with an empty cache
would fail on it. The BBH pin (SaylorTwift/bbh at
b5306be6f827cfafbb545ff5a51f96916029b0fd) was removed with the tier-1 task
list that read it; no registry record cites it.

Usage:
    uv run python scripts/capeval_fetch_data.py             # fetch + pin + offline-verify
    uv run python scripts/capeval_fetch_data.py --offline   # pure presence/read check
    uv run python scripts/capeval_fetch_data.py --skip-verify   # no offline subprocess

On the cluster, export SISTEER_SHARED_ROOT (the cluster's shared root) and
source the shared hf_env.sh first: a run there without HF_HOME is refused, and
so is a host that runs Slurm but names no shared root, so downloads never land
on node-local ~/.cache. EvalPlus data is ALSO seeded
onto the sandbox VM by `capeval_sandbox.sh provision` (execution side);
this script covers the grading host and cluster (generation side needs prompts).
R10: refuses to download below the free-disk floor (300MB).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[1]
PINS_PATH = REPO / "configs" / "capeval" / "pins.yaml"
MODEL_CFG = REPO / "configs" / "capeval" / "models" / "llama31_8b_it.yaml"

MIN_FREE_BYTES = 300 * 1024 * 1024  # R10: refuse below 300MB free

# The alpaca_eval pin already lives in the repo (scripts/stage2_run.py
# ALPACA_EVAL_REVISION); every other revision is resolved at first run and
# frozen into pins.yaml.
ALPACA_EVAL_PIN = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"

# evalplus 0.3.1 facts, verified against the 0.3.1 wheel source
# (evalplus/data/{utils,humaneval}.py): cache dir = appdirs
# user_cache_dir("evalplus") — honors $XDG_CACHE_HOME, default
# ~/.cache/evalplus (the env var is XDG_CACHE_HOME;
# HUMANEVAL_OVERRIDE_PATH overrides the file directly); stored
# DECOMPRESSED as HumanEvalPlus-<version>.jsonl.
EVALPLUS_VERSION = "v0.1.10"
EVALPLUS_URL = ("https://github.com/evalplus/humanevalplus_release/releases/"
                f"download/{EVALPLUS_VERSION}/HumanEvalPlus.jsonl.gz")
EVALPLUS_CACHE_FILE = f"HumanEvalPlus-{EVALPLUS_VERSION}.jsonl"
EVALPLUS_N_TASKS = 164


@dataclass
class DatasetPin:
    """One Tier-0/1 hub dataset dependency."""

    name: str
    repo_id: str
    patterns: Optional[list] = None      # snapshot allow_patterns; None = all
    revision: Optional[str] = None       # pre-pinned (else resolved+frozen)
    verify: str = "parquet"              # parquet | json | jsonl
    used_by: str = ""


# Sources of truth for the task list: the Tier-0 corpora, the parity probes,
# the paired GSM8K and IFEval inputs of the served suite.
DATASET_PINS = [
    DatasetPin("wikitext103", "Salesforce/wikitext",
               ["wikitext-103-raw-v1/test-*"], verify="parquet",
               used_by="tier0/nll_wikitext103 + kl_coarse (§4.1)"),
    # DEVIATION (2026-08-15): the plan's code corpus is
    # bigcode/the-stack-smol data/python (§4.1), but that repo is GATED
    # and this account is not authorized (403 at first fetch).
    # Substituted: codeparrot/codeparrot-clean-valid — the CodeParrot
    # project's dedup-cleaned python-only VALIDATION split (~18k files,
    # single jsonl.gz), ungated, revision-pinned like every other entry.
    # The §16.6 registry marks the code-corpus choice as cheap-to-change
    # pre-submission; swapping back after gated access is granted =
    # delete this pin from pins.yaml and re-run (bumps
    # harness_config_hash). The tier0 task name stays tier0/nll_stack_py
    # (frozen in domains.yaml); the pin key names the ROLE.
    DatasetPin("stack_py", "codeparrot/codeparrot-clean-valid",
               ["*.json.gz"], verify="jsonl_gz",
               used_by="tier0/nll_stack_py + kl_coarse (§4.1; substituted "
                       "for gated the-stack-smol — see comment)"),
    DatasetPin("ultrachat", "HuggingFaceH4/ultrachat_200k",
               ["data/test_sft-*"], verify="parquet",
               used_by="tier0/nll_ultrachat_masked + kl_coarse (§4.1/§4.2)"),
    DatasetPin("canary_prompts", "tatsu-lab/alpaca_eval",
               ["alpaca_eval.json"], revision=ALPACA_EVAL_PIN, verify="json",
               used_by="tier0/canary 300-subset (§4.4)"),
    # The S2 campaign's compliance-prompt source (§9): tatsu-lab/alpaca
    # at the EXACT revision configs/stage2_llama8b.yaml pinned — the
    # sisteer_compliance prompt pool (corpora.alpaca_holdout_prompts,
    # same filter + hash-holdout constants as S2). Added at integration:
    # the corpora module refuses without this pin.
    DatasetPin("alpaca", "tatsu-lab/alpaca",
               ["data/train-*"],
               revision="dce01c9b08f87459cf36a430d809084718273017",
               verify="parquet",
               used_by="si_compliance prompt pool via "
                       "corpora.alpaca_holdout_prompts (§9; S2 pin)"),
    DatasetPin("gsm8k_platinum", "madrylab/gsm8k-platinum",
               None, verify="parquet",
               used_by="the paired-200 GSM8K's question-text match + "
                       "s2parity/gsm8k_0shot (§9)"),
    DatasetPin("mmlu", "cais/mmlu",
               ["all/validation-*", "all/dev-*"], verify="parquet",
               used_by="s2parity/mmlu_letter (D11: validation split, n=200 seed-0)"),
    DatasetPin("ifeval", "google/IFEval", None, verify="jsonl",
               used_by="inspect/ifeval (the served suite's stage reads this "
                       "pin; data file is ifeval_input_data.jsonl)"),
    # The judged controls tier. Revisions
    # pre-pinned to the SHAs verified at planning time (2026-08-19) — a
    # hub main that has moved past them is fine (fetch is BY SHA).
    # OR-Bench is EVAL-ONLY: nothing is ever fitted or tuned on it.
    DatasetPin("or_bench", "bench-llm/or-bench",
               ["or-bench-hard-1k.csv"],
               revision="e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
               verify="csv",
               used_by="judged/or_bench_hard_1k (capeval-controls §3.2; "
                       "1,319 rows asserted at load)"),
    DatasetPin("xstest_v2", "natolambert/xstest-v2-copy",
               ["data/prompts-*"],
               revision="b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
               verify="parquet",
               used_by="judged/xstest_v2 (capeval-controls §3.2; 450 = 250 "
                       "safe + 200 unsafe asserted at fetch AND load; a "
                       "DIFFERENT repo from the attackgen lane's "
                       "Paul/XSTest pin, which is untouched)"),
]

#: expected row counts asserted at fetch time for
#: the judged-tier pins (load-time asserts live in tiers/judged.py).
FETCH_COUNT_GATES = {"or_bench": 1319, "xstest_v2": 450}


@dataclass
class Item:
    name: str
    status: str  # PRESENT | FETCHED | PINNED | FAILED | SKIPPED | WARN
    detail: str = ""


@dataclass
class Report:
    items: list = field(default_factory=list)

    def add(self, item: Item) -> None:
        self.items.append(item)
        print(f"[{item.status}] {item.name}" + (f" -- {item.detail}" if item.detail else ""),
              flush=True)


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _check_disk(report: Report) -> bool:
    free = _free_bytes(_hf_cache_root())
    if free < MIN_FREE_BYTES:
        report.add(Item("free-disk floor", "FAILED",
                        f"only {free / 1e6:.0f}MB free at {_hf_cache_root()} "
                        f"(< {MIN_FREE_BYTES / 1e6:.0f}MB, R10) — refusing to download"))
        return False
    return True


def _load_pins() -> dict:
    if PINS_PATH.is_file():
        return yaml.safe_load(PINS_PATH.read_text()) or {}
    return {"datasets": {}, "evalplus": {}, "models": {}}


def _write_pins(pins: dict) -> None:
    pins["resolved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (
        "# capeval dataset/data pin registry — WRITTEN BY "
        "scripts/capeval_fetch_data.py, DO NOT EDIT BY HAND.\n"
        "# Shape + freeze semantics documented in that script's docstring.\n"
        "# corpora.py and the offline test read the\n"
        "# revisions here; every revision is content-addressed (full SHA).\n")
    tmp = PINS_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(header + yaml.safe_dump(pins, sort_keys=True))
    tmp.replace(PINS_PATH)


# ------------------------------------------------------------- datasets


def _resolve_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id)
    return info.sha


def ensure_dataset(pin: DatasetPin, pinned: dict, offline: bool,
                   report: Report) -> Optional[dict]:
    from huggingface_hub import snapshot_download

    name = f"hf dataset {pin.repo_id} [{pin.name}]"
    prior = pinned.get(pin.name)
    revision = (prior or {}).get("revision") or pin.revision
    freshly_pinned = False
    if revision is None:
        if offline:
            report.add(Item(name, "FAILED",
                            "no pinned revision and --offline forbids resolving one"))
            return None
        revision = _resolve_revision(pin.repo_id)
        freshly_pinned = True
    if offline:
        # Offline mode NEVER re-derives an entry (review infra #7):
        # snapshot_download(local_files_only=True) happily returns a
        # HALF-cached snapshot, and recomputing files/bytes from it would
        # overwrite the frozen registry with the smaller truth. The
        # frozen pins.yaml entry is preserved and the cache is CHECKED
        # against it instead.
        if prior is None:
            report.add(Item(name, "FAILED",
                            "no frozen pins.yaml entry to verify against — "
                            "offline checking requires the online fetch+pin "
                            "first (a half-cached snapshot could otherwise "
                            "pass as complete)"))
            return None
        try:
            path = snapshot_download(pin.repo_id, repo_type="dataset",
                                     revision=revision,
                                     allow_patterns=pin.patterns,
                                     local_files_only=True)
        except Exception:
            report.add(Item(name, "FAILED",
                            f"revision {revision[:12]} not in the local HF cache "
                            "and offline mode forbids fetching"))
            return None
        files = [p for p in Path(path).rglob("*")
                 if p.is_file() and not p.name.startswith(".")]
        total = sum(p.stat().st_size for p in files)
        if len(files) != prior.get("files") or total != prior.get("bytes"):
            report.add(Item(name, "FAILED",
                            f"cached snapshot has {len(files)} files / "
                            f"{total} bytes but pins.yaml froze "
                            f"{prior.get('files')} files / "
                            f"{prior.get('bytes')} bytes — HALF-CACHED or "
                            "drifted snapshot; re-run the online fetch"))
            return None
        report.add(Item(name, "PRESENT",
                        f"@ {revision[:12]}, {len(files)} files, "
                        f"{total / 1e6:.1f}MB (matches the frozen entry)"))
        return dict(prior)
    else:
        # ALWAYS go through the fetching path online: it is incremental
        # (cached files are skipped) and — unlike local_files_only — it
        # sees pattern matches missing from a HALF-cached snapshot (bitten
        # by a pre-existing cais/mmlu cache with only the validation shard).
        path = snapshot_download(pin.repo_id, repo_type="dataset",
                                 revision=revision, allow_patterns=pin.patterns)
        status = "PINNED" if freshly_pinned else ("PRESENT" if prior else "FETCHED")
    files = [p for p in Path(path).rglob("*")
             if p.is_file() and not p.name.startswith(".")]
    if not files:
        report.add(Item(name, "FAILED",
                        f"snapshot at {revision[:12]} matched ZERO files for "
                        f"patterns {pin.patterns} — pattern drift; fix DATASET_PINS"))
        return None
    total = sum(p.stat().st_size for p in files)
    report.add(Item(name, status,
                    f"@ {revision[:12]}, {len(files)} files, {total / 1e6:.1f}MB"))
    return {"repo_id": pin.repo_id, "revision": revision,
            "patterns": pin.patterns, "files": len(files), "bytes": total,
            "verify": pin.verify}


def verify_dataset_offline(pin_name: str, entry: dict, report: Report) -> bool:
    """The runtime-shaped read (S2-proven pattern): snapshot from cache
    only, EVERY payload file of the pinned verify kind opened (review
    infra #7 — one representative shard passing must never green-light a
    half-cached multi-shard snapshot), and the file count/bytes checked
    against the frozen entry."""
    from huggingface_hub import snapshot_download

    name = f"offline read {entry['repo_id']} [{pin_name}]"
    try:
        path = Path(snapshot_download(entry["repo_id"], repo_type="dataset",
                                      revision=entry["revision"],
                                      allow_patterns=entry["patterns"],
                                      local_files_only=True))
        all_files = [p for p in path.rglob("*")
                     if p.is_file() and not p.name.startswith(".")]
        total = sum(p.stat().st_size for p in all_files)
        if (entry.get("files") is not None
                and (len(all_files) != entry["files"] or total != entry["bytes"])):
            raise ValueError(
                f"snapshot has {len(all_files)} files / {total} bytes; the "
                f"frozen entry says {entry['files']} files / {entry['bytes']} "
                "bytes — half-cached or drifted snapshot")
        kind = entry["verify"]
        patterns = {"parquet": "*.parquet", "jsonl_gz": "*.json.gz",
                    "json": "*.json*", "jsonl": "*.json*", "csv": "*.csv"}
        if kind not in patterns:
            raise ValueError(f"unknown verify kind {kind!r}")
        targets = [p for p in sorted(path.rglob(patterns[kind]))
                   if not p.name.startswith(".")]
        if not targets:
            raise FileNotFoundError(f"no {kind} payload files under snapshot")
        n_total = 0
        for target in targets:  # every payload, not one representative
            if kind == "parquet":
                import pyarrow.parquet as pq

                n_total += pq.read_table(target).num_rows
            elif kind == "jsonl_gz":
                with gzip.open(target, "rt") as f:
                    records = [json.loads(f.readline()) for _ in range(3)]
                assert all(isinstance(r, dict) for r in records)
                n_total += len(records)
            elif kind == "json":
                n_total += len(json.loads(target.read_text()))
            elif kind == "csv":
                import pandas as pd

                n_total += len(pd.read_csv(target))
            else:  # jsonl
                n_total += sum(1 for _ in target.open())
        expected_n = FETCH_COUNT_GATES.get(pin_name)
        if expected_n is not None and n_total != expected_n:
            raise ValueError(
                f"{n_total} rows != the expected {expected_n} "
                "(capeval-controls D13 fetch-time count gate) — stop and "
                "flag; never silently re-pin")
        detail = (f"{len(targets)} payload file(s) read "
                  f"({targets[0].name}, ...): {n_total} records/rows")
    except Exception as e:  # noqa: BLE001 - each failure reported, run continues
        report.add(Item(name, "FAILED", f"{type(e).__name__}: {e}"))
        return False
    report.add(Item(name, "PRESENT", detail))
    return True


# ------------------------------------------------------------- evalplus


def _evalplus_cache_dir() -> Path:
    # appdirs.user_cache_dir("evalplus") without importing evalplus (a
    # heavy-lane dep): $XDG_CACHE_HOME/evalplus, default ~/.cache/evalplus.
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "evalplus"


def ensure_evalplus(pinned: dict, offline: bool, report: Report) -> Optional[dict]:
    name = f"evalplus HumanEvalPlus {EVALPLUS_VERSION}"
    cache = _evalplus_cache_dir() / EVALPLUS_CACHE_FILE
    prior = pinned.get("humaneval_plus")
    if not cache.is_file():
        if offline:
            report.add(Item(name, "FAILED",
                            f"{cache} missing and offline forbids fetching"))
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            gz = Path(td) / "data.jsonl.gz"
            urllib.request.urlretrieve(EVALPLUS_URL, gz)
            data = gzip.decompress(gz.read_bytes())
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(cache)
        status = "FETCHED"
    else:
        status = "PRESENT"
    data = cache.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    n_tasks = sum(1 for line in data.splitlines() if line.strip())
    if prior and prior.get("sha256") not in (None, sha):
        report.add(Item(name, "FAILED",
                        f"cached file sha256 {sha[:12]} != pinned "
                        f"{prior['sha256'][:12]} — data drift; delete {cache} "
                        "and re-run, or investigate"))
        return None
    if n_tasks != EVALPLUS_N_TASKS:
        report.add(Item(name, "FAILED",
                        f"{n_tasks} tasks != expected {EVALPLUS_N_TASKS}"))
        return None
    report.add(Item(name, status, f"{n_tasks} tasks, sha256 {sha[:12]} -> {cache}"))
    return {"version": EVALPLUS_VERSION, "url": EVALPLUS_URL,
            "cache_file": EVALPLUS_CACHE_FILE, "sha256": sha,
            "bytes": len(data), "n_tasks": n_tasks}


# ------------------------------------------------------------- model


def ensure_model(on_cluster: bool, report: Report) -> Optional[dict]:
    cfg = yaml.safe_load(MODEL_CFG.read_text())
    entry = {"repo_id": cfg["model"], "revision": cfg["model_revision"]}
    name = f"hf model {cfg['model']} @ {cfg['model_revision'][:12]}"
    if not on_cluster:
        report.add(Item(name, "SKIPPED",
                        "cluster readiness item; the grading host never loads the "
                        "weights (fetch_stage2_data.py convention)"))
        return entry
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(cfg["model"], revision=cfg["model_revision"],
                          local_files_only=True)
    except Exception as e:  # noqa: BLE001
        report.add(Item(name, "FAILED",
                        f"not fully present in the shared HF cache "
                        f"({type(e).__name__}); never auto-downloaded here"))
        return entry
    report.add(Item(name, "PRESENT", ""))
    return entry


# ------------------------------------------------------------- main


def _shared_root() -> Optional[Path]:
    """The cluster's shared storage root: ``$SHARED`` resolved through sisteer.site (SISTEER_SHARED_ROOT, else the
    site config's cluster block), or None where nothing names one on this host."""
    from sisteer import site as _site
    try:
        return Path(_site.resolve("SHARED"))
    except _site.PlaceholderError:
        return None


def _slurm_host() -> bool:
    """True on a host that runs Slurm: inside a job (SLURM_JOB_ID or SLURM_CLUSTER_NAME set) or where ``sbatch`` is on
    PATH (a login node). These are Slurm's own markers, the same on every cluster."""
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_CLUSTER_NAME")) or shutil.which("sbatch") is not None


def _cluster_root() -> Optional[Path]:
    """The shared root when this run is on the cluster, None when it is off it. The site's root decides: present here is
    the cluster, absent here (the grading host, whose site config names a cluster's root) is off it. On a host that runs
    Slurm where nothing names a root the run cannot tell the cluster from the grading host, so it stops by name rather
    than go on as if off the cluster, where nothing would keep downloads off node-local disk and the model check would
    report a skip."""
    shared = _shared_root()
    if shared is None:
        if _slurm_host():
            sys.exit("FATAL: this host runs Slurm but names no shared storage root, so the run cannot tell whether it is "
                     "on the cluster -- export SISTEER_SHARED_ROOT (the cluster's shared root, as the site config names "
                     "it) and source $SISTEER_SHARED_ROOT/scripts/hf_env.sh first.")
        return None
    return shared if shared.is_dir() else None


def run(offline: bool, skip_verify: bool) -> int:
    report = Report()
    shared = _cluster_root()
    on_cluster = shared is not None
    if on_cluster and not os.environ.get("HF_HOME"):
        sys.exit("FATAL: on the cluster but HF_HOME is unset -- source "
                 f"{shared}/scripts/hf_env.sh first (downloads must never "
                 "land on node-local ~/.cache).")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    elif not _check_disk(report):
        return 1

    pins = _load_pins()
    pins.setdefault("datasets", {})
    pins.setdefault("evalplus", {})
    pins.setdefault("models", {})
    undeclared = sorted(set(pins["datasets"]) - {pin.name for pin in DATASET_PINS})
    if undeclared:
        report.add(Item(f"pins registry {PINS_PATH.relative_to(REPO)}", "FAILED",
                        f"lists {undeclared}, which DATASET_PINS does not declare: "
                        "nothing fetches them and the offline re-read checks every "
                        "listed entry, so remove them from pins.yaml"))
        return 1

    for pin in DATASET_PINS:
        try:
            entry = ensure_dataset(pin, pins["datasets"], offline, report)
        except Exception as e:  # noqa: BLE001 - one item never kills the report
            report.add(Item(f"hf dataset {pin.repo_id} [{pin.name}]", "FAILED",
                            f"{type(e).__name__}: {e}"))
            entry = None
        if entry:
            pins["datasets"][pin.name] = entry
        if not offline and not _check_disk(report):
            return 1
    ep = ensure_evalplus(pins["evalplus"], offline, report)
    if ep:
        pins["evalplus"]["humaneval_plus"] = ep
    pins["models"]["llama31_8b_it"] = ensure_model(on_cluster, report)

    # Every failure path below files a FAILED Item on the report, so the
    # exit verdict is one count over report.items at the end.
    def _n_failed() -> int:
        return len([i for i in report.items if i.status == "FAILED"])

    if not _n_failed() and not offline:
        _write_pins(pins)
        report.add(Item(f"pins registry {PINS_PATH.relative_to(REPO)}", "PINNED",
                        f"{len(pins['datasets'])} datasets + evalplus + model"))

        # HF_HUB_OFFLINE=1 re-verification, fresh subprocess (env must precede
        # every hub import — an in-process flag flip would not bind).
        if not skip_verify:
            print("\n-- offline re-read verification (fresh subprocess, "
                  "HF_HUB_OFFLINE=1) --")
            env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 "--offline"], env=env).returncode
            if rc != 0:
                report.add(Item("offline re-read", "FAILED",
                                "the HF_HUB_OFFLINE=1 subprocess check failed — "
                                "see its report above"))
            else:
                report.add(Item("offline re-read", "PRESENT",
                                "every pin re-reads with the network forbidden"))

    if offline:
        # Pure checker mode ALSO exercises the runtime-shaped reads —
        # every pin, never short-circuited (each failure must report).
        for n, e in sorted(pins["datasets"].items()):
            verify_dataset_offline(n, e, report)
        ep_pin = pins["evalplus"].get("humaneval_plus")
        cache = _evalplus_cache_dir() / (ep_pin or {}).get("cache_file", EVALPLUS_CACHE_FILE)
        if not cache.is_file():
            report.add(Item("offline read evalplus data", "FAILED", f"{cache} missing"))
        else:
            report.add(Item("offline read evalplus data", "PRESENT",
                            f"{sum(1 for _ in cache.open())} tasks"))

    host = "cluster" if on_cluster else "dev box"
    print(f"\n== capeval Phase-0 provisioning report ({host}"
          f"{', offline' if offline else ''}) ==")
    if _n_failed():
        print(f"{_n_failed()} item(s) FAILED — no capeval array "
              "may be submitted until every item provisions (R6).")
        return 1
    print("all required items provisioned and pinned.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="capeval Phase-0 fetch+pin+verify")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network; presence/read check only")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the HF_HUB_OFFLINE=1 re-read subprocess")
    args = ap.parse_args()
    offline = args.offline or os.environ.get("HF_HUB_OFFLINE", "") in ("1", "ON", "YES", "TRUE")
    sys.exit(run(offline=offline, skip_verify=args.skip_verify))


if __name__ == "__main__":
    main()
