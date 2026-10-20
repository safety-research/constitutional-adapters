"""Phase-0 stage-2 data provisioning: idempotent fetch + verify.

The one script that makes every stage-2 data dependency PRESENT before any
sbatch goes out (compute nodes run with HF_HUB_OFFLINE=1 and no egress, so
anything missed here dies mid-array, after the GPU time is spent). Safe to
re-run: every item is checked before anything is downloaded, and nothing
already present is re-fetched.

Items (paths/ids are read from the SAME configs the driver reads, so the
script can never drift from what scripts/stage2_run.py consumes):

  1. (REMOVED) the LIFBench checkout is no longer provisioned — the
     suite was dropped from the campaign, and its config/sbatch are gone.
     ensure_lifbench below is kept (tested, unreferenced by main) so the
     kept eval/lifbench.py module stays reversible.
  2. PG-19 long-context filler cache at stage2_longctx.yaml
     padding.filler_cache -- the JSON the static families' padding reads
     ({"text": ...}); BUILT here (seeded, deterministic) when absent, since
     no eval module exposes a builder
  3. NLTK punkt_tab + punkt into the directory jobs read ($NLTK_DATA;
     stage2_common.sh exports $SHARED/nltk_data)
  4. tatsu-lab/alpaca, pinned and verified by the EXACT runtime
     read the drivers perform (load_hf_prompts), plus the core config's
     model snapshot (Llama-3.1-8B-Instruct) verified present at its pinned
     revision under the exact offline read path -- models are NEVER
     auto-downloaded here (weights are tens of GB; cluster-only readiness
     item)

The PersonaDrift, AxBench and IFEval-at-length items are gone with their
suites (third-party material behind no paper result, removed rather than
attributed).

Download guards, per the repo's reproducibility conventions (the runtime
analog of pyproject's uv exclude-newer): every HF artifact is verified
against the revision PINNED in its config -- a fetch that resolves to a
different commit fails loudly instead of silently running on drifted data --
and HF_HUB_OFFLINE=1 (or --offline) turns the script into a pure
presence-checker that never touches the network.

Prints a per-item PRESENT/FETCHED/BUILT/SKIPPED/WARN/FAILED report; exits
nonzero if any required item could not be provisioned.

Usage:
    uv run python scripts/fetch_stage2_data.py [--offline]
    uv run python scripts/fetch_stage2_data.py --only pg19-filler [--filler-out PATH]
    uv run python scripts/fetch_stage2_data.py --only pg19-filler \
        --sidecar-tokenizer MODEL [--sidecar-revision REV]

``--only pg19-filler`` builds / verifies item 2 alone (the static families'
long-context padding reads the same cache: docs/static_padding.md) and
requires none of the other items' configs or pins; ``--filler-out`` writes
that cache somewhere other than stage2_longctx.yaml's ``padding.filler_cache``
(a staging directory for a Modal volume, say) and is accepted only with
``--only pg19-filler``, so the full run always verifies exactly the path the
configs read. With neither flag the script is byte-identical to before.

``--sidecar-tokenizer MODEL [--sidecar-revision REV]`` (with ``--only
pg19-filler`` only) additionally builds / verifies the filler's PRE-TOKENIZED
SIDECAR beside the cache: the ids of the whole corpus under that tokenizer,
which every padded driver then loads instead of tokenizing ~2.9 MB itself
(~7 s of CPU per process on the pinned Qwen3.5 tokenizer). The sidecar is
keyed by the corpus text's sha256 and the tokenizer DEFINITION's sha256, so a
different corpus or tokenizer never reuses it (a driver that finds none, or a
stale one, builds it on first use -- this flag just moves that cost ahead of
the wave). See ``sisteer.eval.attacks.static_padding.ensure_filler_ids``.

On the cluster, export SISTEER_SHARED_ROOT (the cluster's shared root, as the
site config names it) and source $SISTEER_SHARED_ROOT/scripts/hf_env.sh first.
The script refuses to run there without HF_HOME, so downloads can never land
on node-local disk, and it refuses a host that runs Slurm but names no shared
root, which it cannot tell from a dev box. Both refusals hold for ``--only
pg19-filler`` too: a filler BUILD is an HF download (the pinned PG-19
validation parquet through ``hf://``), which is exactly what the guards exist
to keep off node-local ~/.cache.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

# Alpaca prompt pool: the core and composition configs draw
# fit/eval prompts from tatsu-lab/alpaca via load_hf_prompts, which
# resolves main -- so main is verified against this pin.
ALPACA_DATASET = "tatsu-lab/alpaca"
ALPACA_REVISION = "dce01c9b08f87459cf36a430d809084718273017"

LIFBENCH_GIT_URL = "https://github.com/SheldonWu0327/LIFBench-2024"

# package -> nltk.data.find() probe. punkt_tab is what nltk>=3.9 resolves
# sentence tokenization to; punkt is the older name some loaders still read.
NLTK_PACKAGES = {
    "punkt_tab": "tokenizers/punkt_tab",
    "punkt": "tokenizers/punkt",
}

# Filler builder (item 2): fixed seeded concatenation of PG-19 VALIDATION
# books. The deepest cell needs 262144 tokens of filler; ~2.8M chars is
# ~600-650k Llama-3 tokens (~4.3 chars/token on PG-19 prose), a >2x margin.
# build_padded_prompt still fails loudly at runtime if it ever runs short.
# Source: the emozilla/pg19 parquet mirror, NOT canonical deepmind/pg19 --
# datasets==5.0.1 (uv.lock) removed script-dataset support, deepmind/pg19
# is script-only, and its refs/convert/parquet branch holds NO parquet
# files (only pg19.py + metadata; verified 2026-08-13). The mirror's
# validation split is byte-identical prose (50 books, 17.4M chars); we
# pin its revision and read ONLY the validation shard so the fetch never
# pulls the ~11GB train split.
PG19_DATASET = "emozilla/pg19"
PG19_REVISION = "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"
PG19_VALIDATION_FILE = "data/validation-00000-of-00001-0f92e2337f79aeac.parquet"
PG19_SPLIT = "validation"
PG19_SEED = 0
PG19_TARGET_CHARS = 2_800_000

#: the per-item selectors ``--only`` accepts (one today; an unknown value refuses
#: naming these, argparse's choices doing the naming)
ONLY_ITEMS = ("pg19-filler",)


@dataclass
class Item:
    name: str
    status: str  # PRESENT | FETCHED | BUILT | SKIPPED | WARN | FAILED
    detail: str = ""


def _cfg(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


def _shared_root() -> Path | None:
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


def _cluster_root() -> Path | None:
    """The shared root when this run is on the cluster, None when it is off it. The site's root decides: present here is
    the cluster, absent here (the grading host, whose site config names a cluster's root) is off it. On a host that runs
    Slurm where nothing names a root the run cannot tell the cluster from a dev box, so it stops by name rather than go on
    as if off the cluster, where nothing would keep HF downloads off node-local disk."""
    shared = _shared_root()
    if shared is None:
        if _slurm_host():
            sys.exit("FATAL: this host runs Slurm but names no shared storage root, so the run cannot tell whether it is "
                     "on the cluster -- export SISTEER_SHARED_ROOT (the cluster's shared root, as the site config names "
                     "it) and source $SISTEER_SHARED_ROOT/scripts/hf_env.sh first.")
        return None
    return shared if shared.is_dir() else None


def _hf_offline(args) -> bool:
    env = os.environ.get("HF_HUB_OFFLINE", "").upper()
    return args.offline or env in ("1", "ON", "YES", "TRUE")


# ------------------------------------------------------------- HF snapshots


def _snapshot_sha(path: str) -> str:
    # snapshot_download returns .../snapshots/<commit-sha>
    return Path(path).name


def ensure_hf_snapshot(repo_id: str, repo_type: str, pin: str,
                       offline: bool) -> Item:
    """Snapshot at the default revision (main) -- the ref runtime resolves
    under HF_HUB_OFFLINE=1 -- then verify it equals the config pin."""
    from huggingface_hub import snapshot_download

    name = f"hf {repo_type} {repo_id}"
    status = "PRESENT"
    try:
        path = snapshot_download(repo_id, repo_type=repo_type,
                                 local_files_only=True)
    except Exception:
        if offline:
            return Item(name, "FAILED",
                        "not in the local HF cache and offline mode forbids "
                        "fetching -- run without --offline/HF_HUB_OFFLINE on "
                        "a host with egress")
        path = snapshot_download(repo_id, repo_type=repo_type)
        status = "FETCHED"
    sha = _snapshot_sha(path)
    if sha != pin:
        return Item(name, "FAILED",
                    f"cached/fetched main = {sha[:12]} but the config pins "
                    f"{pin[:12]} -- data drift; re-pin the config or restore "
                    "the pinned snapshot before running")
    return Item(name, status, f"@ {sha[:12]} -> {path}")


def verify_alpaca_runtime_read() -> Item:
    """The exact read _prompt_split performs (data.prompts.load_hf_prompts
    on tatsu-lab/alpaca): warms the cache and proves prompt provisioning."""
    from sisteer.data.prompts import load_hf_prompts

    name = "alpaca runtime read (load_hf_prompts)"
    prompts = load_hf_prompts(ALPACA_DATASET, n=5)
    if len(prompts) < 5:
        return Item(name, "FAILED", f"loaded only {len(prompts)} prompts")
    return Item(name, "PRESENT", f"{len(prompts)} sample prompts loaded")


def ensure_model_snapshot(repo_id: str, revision: str | None, used_by: str,
                          on_cluster: bool) -> Item:
    """Config-referenced MODEL snapshots: verified present at the
    config's PINNED revision under the exact offline read path
    (snapshot_download(local_files_only=True) -- what HF_HUB_OFFLINE
    resolution performs at model load). NEVER downloads: weights are tens
    of GB (a >50GB shared-storage download needs explicit user approval)
    and the stage-2 models are already cached on cluster A.
    Off-cluster this is SKIPPED -- the arrays that need the weights run on
    the cluster; a dev box does not."""
    name = f"hf model {repo_id}" + (f" @ {revision[:12]}" if revision else "")
    if not on_cluster:
        return Item(name, "SKIPPED",
                    f"cluster readiness item (needed by {used_by}); the dev "
                    "box does not need the weights")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo_id, revision=revision,
                                 local_files_only=True)
    except Exception as e:  # noqa: BLE001 - any cache miss => not provisioned
        return Item(name, "FAILED",
                    f"not fully present in the local HF cache "
                    f"({type(e).__name__}) -- required by {used_by}. Models "
                    "are never auto-downloaded here; fetch it explicitly on "
                    "a login node with the shared HF_HOME.")
    return Item(name, "PRESENT", f"-> {path}")


# ------------------------------------------------------------- LIFBench


def ensure_lifbench(vendor_dir: Path, pin: str, offline: bool) -> Item:
    name = f"lifbench checkout {vendor_dir}"
    marker = vendor_dir / "evaluation" / "EvaluateFunc.py"
    status = "PRESENT"
    if not marker.is_file():
        if offline:
            return Item(name, "FAILED",
                        f"missing and offline mode forbids cloning "
                        f"{LIFBENCH_GIT_URL}")
        vendor_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", LIFBENCH_GIT_URL, str(vendor_dir)],
                       check=True)
        subprocess.run(["git", "-C", str(vendor_dir), "checkout", pin],
                       check=True)
        status = "FETCHED"
    # pin verification: reproducibility is via this pinned fetch.
    # An unverifiable checkout is FAILED, not WARN: the held-out
    # split and the scorer mapping are only valid AT the pinned commit, so
    # "all provisioned" must never be claimed over an unverified tree.
    head = subprocess.run(["git", "-C", str(vendor_dir), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode != 0:
        return Item(name, "FAILED",
                    "present but not a git checkout -- cannot verify the "
                    f"pinned commit {pin[:12]} (stage-2 readiness requires "
                    "a verified pin, D22#11); remove it and re-run this "
                    "script to re-clone")
    if head.stdout.strip() != pin:
        return Item(name, "FAILED",
                    f"HEAD {head.stdout.strip()[:12]} != pinned {pin[:12]} "
                    "-- re-checkout")
    # authoritative load: all 11 judge functions + the ARS rubric weights
    # (fails loudly on a drifted checkout, same as eval.lifbench would)
    from sisteer.eval.lifbench import TASKS, load_judges, load_score_tools

    judges = load_judges(vendor_dir)
    load_score_tools(vendor_dir)
    prompts = vendor_dir / "data" / "prompts" / f"{TASKS[0]}.json"
    if not prompts.is_file():
        return Item(name, "FAILED", f"prompts missing: {prompts}")
    return Item(name, status,
                f"@ {pin[:12]}, {len(judges)}/{len(TASKS)} judge fns loaded")


# ------------------------------------------------------------- PG-19 filler


def _parse_filler(path: Path) -> str:
    # Mirrors static_padding.load_filler's accepted forms (JSON string, or a
    # dict with a "text" key) -- kept tiny and in sync by test/canary.
    data = json.loads(path.read_text())
    if isinstance(data, str):
        return data
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        return data["text"]
    raise ValueError(f"unrecognized filler cache format in {path}")


def ensure_pg19_filler(path: Path, on_cluster: bool, offline: bool) -> Item:
    name = f"pg19 filler cache {path}"
    if path.is_file():
        try:
            text = _parse_filler(path)
        except ValueError as e:
            return Item(name, "FAILED", str(e))
        if len(text) < 1_200_000:
            # FAILED, not WARN: an undersized filler kills the
            # 64k+/256k cells MID-ARRAY ('filler too short'), after the
            # GPU time is spent -- the readiness gate must be honest.
            return Item(name, "FAILED",
                        f"present but only {len(text):,} chars (< ~262k "
                        "tokens of filler) -- the 64k+/256k cells will raise "
                        "'filler too short'; delete it and re-run to rebuild")
        return Item(name, "PRESENT", f"{len(text):,} chars")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if not on_cluster:
            return Item(name, "SKIPPED",
                        "cluster-only path, not reachable from this host -- "
                        "run this script on the cluster login node")
        return Item(name, "FAILED", f"cannot create {path.parent}: {e}")
    # No eval module exposes a builder (static_padding.load_filler only
    # reads this file), so it is built
    # here: deterministic seeded book order over the PG-19 validation split.
    from datasets import load_dataset

    try:
        # Parquet builder on the pinned validation shard only (split name
        # under the parquet builder is always "train").
        data_file = (f"hf://datasets/{PG19_DATASET}@{PG19_REVISION}/"
                     f"{PG19_VALIDATION_FILE}")
        ds = load_dataset("parquet", data_files=data_file, split="train")
    except Exception as e:
        return Item(name, "FAILED",
                    f"cannot load {PG19_DATASET}[{PG19_SPLIT}] "
                    f"@ {PG19_REVISION[:8]} ({type(e).__name__}: {e})"
                    + (" -- offline and not cached" if offline else ""))
    order = list(range(len(ds)))
    random.Random(PG19_SEED).shuffle(order)
    parts: list[str] = []
    total = 0
    for i in order:
        text = ds[i]["text"]
        parts.append(text)
        total += len(text) + 2
        if total >= PG19_TARGET_CHARS:
            break
    text = "\n\n".join(parts)[:PG19_TARGET_CHARS]
    payload = {
        "text": text,
        "dataset": PG19_DATASET,
        "revision": PG19_REVISION,
        "data_file": PG19_VALIDATION_FILE,
        "split": PG19_SPLIT,
        "seed": PG19_SEED,
        "n_books": len(parts),
        "chars": len(text),
        "builder": "scripts/fetch_stage2_data.py",
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)  # atomic: a killed build never leaves a torn cache
    return Item(name, "BUILT",
                f"{len(text):,} chars from {len(parts)} seeded books")


def ensure_filler_ids_sidecar(path: Path, tokenizer: str, revision: str | None) -> Item:
    """The filler's PRE-TOKENIZED sidecar under ``tokenizer`` @ ``revision``
    (sisteer.eval.attacks.static_padding.ensure_filler_ids: the ids of the
    whole corpus, keyed by the corpus text's sha256 and the tokenizer
    definition's sha256, written atomically beside the corpus). PRESENT when a
    matching one is already there, BUILT when this call tokenized and wrote it,
    FAILED when the corpus is absent, the tokenizer has no hashable definition
    (a slow tokenizer) or the sidecar cannot be written. The tokenizer is
    constructed by ``sisteer.models.load_tokenizer`` -- the padded driver's own
    construction -- so the hash is the one a run computes."""
    name = f"filler ids sidecar {tokenizer}@{(revision or 'main')[:12]} beside {path}"
    if not path.is_file():
        return Item(name, "FAILED",
                    f"no filler cache at {path} -- build it first (--only pg19-filler)")
    from sisteer.eval.attacks import static_padding as sp
    from sisteer.models import load_tokenizer

    tok = load_tokenizer(tokenizer, revision=revision)
    identity = sp.tokenizer_identity(tok)
    if identity is None:
        return Item(name, "FAILED",
                    f"{type(tok).__name__} has no serialized definition to key a sidecar "
                    "by (not a fast tokenizer); padded runs will tokenize in-process")
    ids, status = sp.ensure_filler_ids(path, tok, tokenizer_revision=revision)
    sidecar = sp.filler_ids_sidecar_path(path, identity)
    detail = f"{len(ids):,} tokens @ {sidecar} (tokenizer sha256 {identity['sha256'][:12]})"
    if status == "sidecar":
        return Item(name, "PRESENT", detail)
    if status == "built":
        return Item(name, "BUILT", detail)
    return Item(name, "FAILED", f"tokenized the corpus but could not write {sidecar} ({status})")


# ------------------------------------------------------------- NLTK data


def _nltk_subprocess(code: str, *argv: str, nltk_data: Path) -> subprocess.CompletedProcess:
    """Run nltk in a child with a NEUTRAL cwd: nltk's import-security hook
    (nltk/inisec.py) blocks any nltk-initiated import that resolves under
    the cwd, which false-positives when the venv lives inside the repo
    (site-packages is then 'in cwd'). An empty temp dir sidesteps it."""
    env = dict(os.environ, NLTK_DATA=str(nltk_data))
    with tempfile.TemporaryDirectory() as neutral:
        return subprocess.run([sys.executable, "-c", code, *argv],
                              cwd=neutral, env=env,
                              capture_output=True, text=True)


def ensure_nltk(nltk_data: Path, offline: bool) -> Item:
    name = f"nltk data {nltk_data}"
    probe = (
        "import json, sys\n"
        "import nltk.data\n"
        "out = {}\n"
        "for pkg, res in json.loads(sys.argv[1]).items():\n"
        "    try:\n"
        "        nltk.data.find(res)\n"
        "        out[pkg] = True\n"
        "    except LookupError:\n"
        "        out[pkg] = False\n"
        "print(json.dumps(out))\n"
    )
    res = _nltk_subprocess(probe, json.dumps(NLTK_PACKAGES),
                           nltk_data=nltk_data)
    if res.returncode != 0:
        return Item(name, "FAILED", f"nltk probe failed: {res.stderr.strip()}")
    found = json.loads(res.stdout.strip().splitlines()[-1])
    missing = [p for p, ok in found.items() if not ok]
    if not missing:
        return Item(name, "PRESENT", ", ".join(NLTK_PACKAGES))
    if offline:
        return Item(name, "FAILED",
                    f"missing {missing} and offline -- prefetch on a login "
                    f"node: uv run python -m nltk.downloader -d "
                    f"\"{nltk_data}\" " + " ".join(missing))
    dl = (
        "import sys\n"
        "import nltk\n"
        "ok = all(nltk.download(p, download_dir=sys.argv[1], quiet=True)\n"
        "         for p in sys.argv[2:])\n"
        "sys.exit(0 if ok else 1)\n"
    )
    nltk_data.mkdir(parents=True, exist_ok=True)
    res = _nltk_subprocess(dl, str(nltk_data), *missing, nltk_data=nltk_data)
    if res.returncode != 0:
        return Item(name, "FAILED",
                    f"download of {missing} failed: {res.stderr.strip()}")
    return Item(name, "FETCHED", f"downloaded {', '.join(missing)}")


# ------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Idempotent Phase-0 stage-2 data fetch (see docstring)")
    ap.add_argument("--offline", action="store_true",
                    help="never touch the network; check presence only")
    ap.add_argument("--only", choices=ONLY_ITEMS, default=None,
                    help="provision ONE item and nothing else: 'pg19-filler' = "
                         "build / verify the PG-19 filler cache (item 2; the static "
                         "families' padding reads it too), requiring none of the "
                         "other items. Omitted = every item, as before.")
    ap.add_argument("--filler-out", default=None, metavar="PATH",
                    help="with --only pg19-filler: write / verify the filler cache "
                         "at PATH instead of stage2_longctx.yaml padding.filler_cache "
                         "(e.g. a staging dir for a Modal volume). Refused without "
                         "--only pg19-filler, so the full run verifies exactly the "
                         "path the configs read.")
    ap.add_argument("--sidecar-tokenizer", default=None, metavar="MODEL",
                    help="with --only pg19-filler: also build / verify the filler's "
                         "PRE-TOKENIZED sidecar under this tokenizer (an HF model id, "
                         "constructed by sisteer.models.load_tokenizer), written beside "
                         "the filler cache and keyed by the corpus text and the tokenizer "
                         "definition, so padded drivers load ids instead of tokenizing "
                         "the corpus. Refused without --only pg19-filler.")
    ap.add_argument("--sidecar-revision", default=None, metavar="REV",
                    help="with --sidecar-tokenizer: the tokenizer's pinned revision, "
                         "recorded in the sidecar (the KEY is the tokenizer definition's "
                         "sha256, so a revision that changes nothing in the tokenizer "
                         "shares the sidecar).")
    args = ap.parse_args()
    if args.filler_out and args.only != "pg19-filler":
        ap.error("--filler-out moves the filler item only; pass --only pg19-filler "
                 "with it (the full run verifies exactly the path the configs read)")
    if args.sidecar_tokenizer and args.only != "pg19-filler":
        ap.error("--sidecar-tokenizer builds beside the filler item only; pass "
                 "--only pg19-filler with it")
    if args.sidecar_revision and not args.sidecar_tokenizer:
        ap.error("--sidecar-revision needs --sidecar-tokenizer")
    offline = _hf_offline(args)
    if offline:
        # Make --offline binding for EVERY loader this process (and its
        # nltk children) runs, not just the explicit snapshot calls --
        # datasets.load_dataset would otherwise still reach the network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    shared = _cluster_root()
    on_cluster = shared is not None
    if on_cluster and not os.environ.get("HF_HOME"):
        sys.exit("FATAL: on the cluster but HF_HOME is unset -- source "
                 f"{shared}/scripts/hf_env.sh first, or downloads land on "
                 "node-local ~/.cache and are invisible to compute nodes.")

    longctx_cfg = _cfg("stage2_longctx.yaml")
    # the config's filler_cache may carry a site placeholder ($SHARED/...): expanded through sisteer.site
    from sisteer import site as _site
    filler_path = (Path(args.filler_out) if args.filler_out
                   else _site.expand_path(longctx_cfg["padding"]["filler_cache"], key="padding.filler_cache"))

    items: list[Item] = []

    def run(fn, *a, **kw):
        try:
            items.append(fn(*a, **kw))
        except Exception as e:  # noqa: BLE001 - one item never kills the report
            items.append(Item(getattr(fn, "__name__", str(fn)), "FAILED",
                              f"{type(e).__name__}: {e}"))

    if args.only == "pg19-filler":
        # item 2 alone: no other config is read, no other pin imported
        run(ensure_pg19_filler, filler_path, on_cluster, offline)
        if args.sidecar_tokenizer:
            run(ensure_filler_ids_sidecar, filler_path, args.sidecar_tokenizer,
                args.sidecar_revision)
        _report(items, on_cluster=on_cluster, offline=offline, only=args.only)
        return

    main_cfg = _cfg("stage2_llama8b.yaml")

    if on_cluster:
        nltk_data = Path(os.environ.get("NLTK_DATA") or shared / "nltk_data")
    else:
        nltk_data = Path(os.environ.get("NLTK_DATA")
                         or Path.home() / "nltk_data")

    # The alpaca prompt pool, pinned + verified by the exact runtime
    # read the drivers perform.
    run(ensure_hf_snapshot, ALPACA_DATASET, "dataset", ALPACA_REVISION,
        offline)
    run(verify_alpaca_runtime_read)
    # The core config's model snapshot, at its pinned revision,
    # under the exact offline read path (never auto-downloaded).
    run(ensure_model_snapshot, main_cfg["model"],
        main_cfg.get("model_revision"), "stage2_llama8b.yaml (+ composition)",
        on_cluster)
    # lifbench: dropped from the campaign -- its checkout is no longer
    # provisioned here (ensure_lifbench stays, tested, for reversibility).
    run(ensure_pg19_filler, filler_path, on_cluster, offline)
    run(ensure_nltk, nltk_data, offline)
    _report(items, on_cluster=on_cluster, offline=offline, only=None)


def _report(items: list[Item], *, on_cluster: bool, offline: bool, only: str | None) -> None:
    host = "cluster" if on_cluster else "dev box"
    print(f"\n== stage-2 data provisioning report ({host}"
          f"{', offline' if offline else ''}"
          f"{f', only: {only}' if only else ''}) ==")
    width = max(len(i.status) for i in items)
    for i in items:
        print(f"[{i.status:<{width}}] {i.name}" + (f" -- {i.detail}" if i.detail else ""))
    failed = [i for i in items if i.status == "FAILED"]
    if failed:
        print(f"\n{len(failed)} item(s) FAILED -- stage-2 arrays must not "
              "be submitted until every required item is provisioned.")
        sys.exit(1)
    print("\nall required items provisioned.")


if __name__ == "__main__":
    main()
