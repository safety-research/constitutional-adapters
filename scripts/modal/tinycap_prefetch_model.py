#!/usr/bin/env python
"""tinycap_prefetch_model.py — stage ONE served checkpoint into the tinycap volume's HF cache and VERIFY it against the Hub's own
per-file inventory. CPU only; no GPU container.

    TINYCAP_APP_NAME=tinycap-prefetch-<short> modal run scripts/modal/tinycap_prefetch_model.py --repo Qwen/Qwen3.5-4B \\
        --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a [--verify-only] [--dry-run]

What the one function does, in order (everything recorded in the returned report):
  1. INVENTORY from the Hub: ``HfApi.model_info(repo, revision=, files_metadata=True)`` — every sibling's size, an LFS file's
     sha256, a plain git file's blob_id (the git BLOB SHA-1) — through ``sisteer.capeval.tinycap.hf_inventory.inventory_from_siblings``
     (a listing that cannot name a digest refuses; nothing is downloaded against an unprovable listing). The Hub's ``sha`` of the
     revision must equal the pinned revision (a moved ref refuses).
  2. SNAPSHOT: ``snapshot_download(repo, revision=)`` into ``HF_HOME=/vol/hf`` (the launcher's offline cache; network ON here only),
     idempotent — files already present are kept by the Hub client. ``--verify-only`` skips this (re-verifies what is there).
  3. VERIFY: ``hf_inventory.verify_snapshot(<snapshot dir>, inventory)`` — every inventoried file present, size equal, digest equal
     (sha256 for LFS, git blob SHA-1 for plain files). ANY problem -> the inventory is NOT written, the volume is NOT committed
     with a marker, the report says REFUSED and the process exits 2 — a snapshot that does not match the Hub never serves.
  4. RECORD: the inventory at ``<HF_HOME>/hub/models--<org>--<name>/.sisteer_inventory_<rev8>.json`` (beside snapshots/, outside the
     snapshot) plus the verification result; ``VOL.commit()`` — the record a later ``--verify-only`` run re-hashes the snapshot against.

Idempotent: a second run re-verifies and rewrites an identical inventory. Credentials: Modal tokens from the environment (the secret-manager CLI);
the HF token secret the launcher's `stage` uses (public checkpoints need none). Never prints a secret.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import modal

HERE = Path(__file__).resolve().parent


def _repo_local() -> Path:
    """The checkout root on the LOCAL side (this file lives at scripts/modal/). Inside the Modal container the script is mounted
    flat at ``/root/<name>.py`` with no such ancestor — ``HERE.parents[1]`` raised ``IndexError: 1`` there and crash-looped the first
    prefetch container (2026-10-03) — so walk up to the first ancestor that IS a checkout and fall back to the script's own directory
    (the container reaches ``sisteer`` through the launcher image's ``add_local_python_source``). The rule of ``tinycap_served._repo_local``."""
    for parent in (HERE, *HERE.parents):
        try:
            if (parent / "src" / "sisteer").is_dir() and (parent / "scripts" / "modal").is_dir():
                return parent
        except OSError:
            continue
    return HERE


REPO_LOCAL = _repo_local()
if str(REPO_LOCAL / "src") not in sys.path:
    sys.path.insert(0, str(REPO_LOCAL / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tinycap_served as T  # noqa: E402  (the launcher: its image, volume, app-name rule and HF token secret)
from sisteer.capeval.tinycap import hf_inventory as HI  # noqa: E402

APP_NAME = os.environ.get(T.APP_NAME_ENV, "").strip()
app = modal.App(APP_NAME or "tinycap-prefetch-import")
HF_HOME = f"{T.VOL_PATH}/hf"
#: a CPU container: the download is network-bound, the verification disk-bound; no GPU is ever attached here
PREFETCH_CPU, PREFETCH_MEMORY_MB, PREFETCH_TIMEOUT_S = 8, 32_768, 4 * 3600


def _report_path(repo: str, revision: str) -> Path:
    return HI.repo_cache_dir(Path(HF_HOME), repo) / f".sisteer_prefetch_report_{revision[:8]}.json"


def prefetch_and_verify(repo: str, revision: str, *, verify_only: bool = False) -> Dict[str, Any]:
    """The body of the Modal function (importable for tests with a stubbed Hub): inventory -> snapshot -> verify -> record."""
    # the launcher image pins HF_HUB_OFFLINE=1 for every SERVING container (the weights must already be on the volume); this is the one
    # container whose job is the download — leave offline mode BEFORE the first Hub call, in the env AND in huggingface_hub's constant
    # (read once at import; the request hook consults the constant, not the env — the 2026-10-03 attempts c/d died on it)
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_OFFLINE"] = "0"
    import huggingface_hub.constants as _hfc
    _hfc.HF_HUB_OFFLINE = False
    from huggingface_hub import HfApi, snapshot_download
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    rep: Dict[str, Any] = {"repo": repo, "revision": revision, "hf_home": HF_HOME, "verify_only": bool(verify_only),
                           "hf_token_present": bool(tok), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    api = HfApi(token=tok)
    info = api.model_info(repo, revision=revision, files_metadata=True)
    hub_sha = str(getattr(info, "sha", "") or "")
    if hub_sha != revision:
        rep.update(status="REFUSED", error=f"the Hub resolves {repo}@{revision[:12]} to sha {hub_sha[:12]!r} — the pinned revision is not a commit the Hub serves as itself")
        return rep
    inventory = HI.inventory_from_siblings(repo, revision, hub_sha, list(getattr(info, "siblings", None) or []))
    rep.update(inventory_n_files=inventory["n_files"], inventory_total_bytes=inventory["total_bytes"])
    snap = HI.snapshot_dir(Path(HF_HOME), repo, revision)
    # the launcher image pins HF_HUB_OFFLINE=1 for every SERVING container (the weights must already be on the volume); this is the one
    # container whose job is the download — leave offline mode here, in the env AND in huggingface_hub's already-read constant
    if not verify_only:
        t0 = time.time()
        got = Path(snapshot_download(repo, revision=revision, token=tok))
        rep.update(snapshot_download_s=round(time.time() - t0, 1), snapshot=str(got))
        if got.resolve() != snap.resolve():
            rep.update(status="REFUSED", error=f"snapshot_download returned {got}, not the pinned snapshot dir {snap}")
            return rep
    rep["snapshot"] = str(snap)
    res = HI.verify_snapshot(snap, inventory, hash_bytes=True)
    rep["verification"] = {k: v for k, v in res.items() if k != "verified"} | {"n_verified": len(res["verified"])}
    if not res["ok"]:
        rep.update(status="REFUSED", error="the snapshot does not match the Hub's inventory: " + "; ".join(res["problems"][:12]))
        return rep
    inv_path = HI.inventory_path(HI.repo_cache_dir(Path(HF_HOME), repo), revision)
    rep["inventory_path"] = str(inv_path)
    rep["inventory_sha256"] = HI.write_inventory(inv_path, inventory)
    rep["status"] = "OK"
    return rep


#: the launcher module is mounted FLAT beside this entrypoint (the probe plumbing's rule): the image carries tinycap_images /
#: tinycap_compile_cache as copy layers but NOT tinycap_served.py, which this script imports at load
IMAGE = T.image.add_local_file(str(Path(T.__file__).resolve()), "/root/tinycap_served.py")


@app.function(image=IMAGE, volumes={T.VOL_PATH: T.VOL}, cpu=PREFETCH_CPU, memory=PREFETCH_MEMORY_MB, timeout=PREFETCH_TIMEOUT_S,
              secrets=[modal.Secret.from_name(T.HF_TOKEN_SECRET)])
def prefetch(repo: str, revision: str, verify_only: bool = False) -> dict:
    rep = prefetch_and_verify(repo, revision, verify_only=verify_only)
    p = _report_path(repo, revision)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    T.VOL.commit()
    print("[prefetch]", json.dumps({k: v for k, v in rep.items() if k != "verification"}, default=str), flush=True)
    return rep


@app.local_entrypoint()
def main(repo: str = "", revision: str = "", verify_only: bool = False, dry_run: bool = False):
    if not APP_NAME:
        raise SystemExit(f"set {T.APP_NAME_ENV} (e.g. tinycap-prefetch-<model>) — the app name is the run's identity")
    if not repo or len(revision) != 40:
        raise SystemExit("--repo <org/name> and --revision <full 40-hex> are required (the pinned snapshot, never a branch)")
    if dry_run:
        print(json.dumps({"would_run": "prefetch", "repo": repo, "revision": revision, "verify_only": verify_only, "hf_home": HF_HOME,
                          "inventory_path": str(HI.inventory_path(HI.repo_cache_dir(Path(HF_HOME), repo), revision)),
                          "cpu": PREFETCH_CPU, "memory_mb": PREFETCH_MEMORY_MB, "gpu": None}, indent=1))
        return
    rep = prefetch.remote(repo, revision, verify_only=verify_only)
    print(json.dumps(rep, indent=1, default=str))
    if rep.get("status") != "OK":
        raise SystemExit(2)
