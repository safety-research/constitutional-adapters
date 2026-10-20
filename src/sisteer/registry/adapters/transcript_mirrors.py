"""Former grading-host transcript stores → their cluster copies, for the item pointers of an adapter that never read a transcript.

Every `run-state/full22/transcripts_*` store's copy on the grading host was mirrored to the clusters and deleted. What stays
under `run-state/full22/transcripts_mirrors/` is the pointer material this module reads:

* `index.json` — `stores.<cluster store>`: `former_grading_host_dir` (the grading host dir, relative to the run-state root; the four
  relu_native dirs share the glob `relu_native/transcripts_relu_native_bu_*`), `cluster_a_path`, `cluster_b_path` (null
  for the non-canonical pulls, which got a cluster A copy only), `origin` / `mirror`, `content_manifest_sha256` (the
  sidecar's whole-store content manifest, verified equal on origin and mirror); `_meta.built_at`, `_meta.relu_note`,
  `_meta.stale_note`.
* `grading_host_file_sha256.tsv` — `(dir, relpath, sha256)` for every file of the canonical stores and the whole-pull dirs at
  reclaim time. The 13 `grading_host_pulls` pushes are NOT in it (it predates their push); for them only the store-level
  manifest sha exists, so their pointers carry `unverified` per-file shas with the reason, never a guessed one.

Relpaths are the same on the grading host and on the clusters (the grading host held whole-store pulls; `pull_last.txt` at the grading host
dir's top is bookkeeping, not a store file). The one rename is relu_native's: the grading host `relu_native_bu_<model>` dirs
were per-model MERGES of three per-round cluster A roots, `shard_rN_MM` on the grading host = round-N root (`…_rN`) `shardMM`,
a plain `shardNN` = the unsuffixed (round-1) root. That reading is the index's own `_meta.relu_note`, and the reclaim
table bears it out by count: 2,733 plain-shard files = the unsuffixed root's 2,733; 2,043 renamed = r2's 1,881 + r3's 162.

`_meta.stale_note` names a store cell whose grading host pull predated the final writes (the cluster bytes differ from the
grading host shas). The note is prose, so it is parsed for `<store> <attack>/<cell>:`; a note that does not parse is a refusal
of the whole relocation — silently attributing a stale sha to the wrong store is exactly the failure this guards.

Pointer granularity: the goal's transcript is one row of `progress.jsonl` in ONE shard directory, and which shard is
recorded only inside the deleted files. So the pointer names the CELL directory (`<private_run>/<attack>/<cell>`) as a
`store_root` artifact carrying every constituent file's cluster relpath and sha in `extra.files`, both clusters in
`locations_json`, and `artifact_id` = sha256 over the sorted `<store>\\t<relpath>\\t<sha256>` manifest lines (sha256 of
the primary URI when no per-file sha exists — `extra.artifact_id_rule` says which). `generation_offset` stays null.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .. import site_aliases as SA
from . import lineage_common as LC
from .base import AdapterRefusal, Input, check_input, store_walk

MIRRORS_DIR = "transcripts_mirrors"
INDEX_NAME = "index.json"
TSV_NAME = "grading_host_file_sha256.tsv"
CLUSTER_URI = {"cluster_a": "cluster_a:{path}", "cluster_b": "cluster_b:{path}"}
_SHARD_RENAMED = re.compile(r"^shard_r(?P<round>\d+)_(?P<shard>\d+)$")
_SHARD_PLAIN = re.compile(r"^shard\d+$")
_ROUND_SUFFIX = re.compile(r"_r(?P<round>\d+)$")
_STALE_RE = re.compile(r"(?P<store>transcripts[A-Za-z0-9_]+)\s+(?P<cell>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*:")


@dataclass
class CellPointer:
    """Where one cell's transcripts live now, and how to name the artifact that points there."""
    former_grading_host_dir: str
    cell_rel: str                                  # <private_run>/<attack>/<cell>
    stores: List[str]                              # index store names holding the cell's files (one, or relu's rounds)
    uri_primary: str                               # cluster_a:<path>/<cell_rel> of the primary store
    locations: List[Dict[str, object]]             # locations_json entries, both clusters where both exist
    files: Dict[str, Optional[str]]                # "<store>:<cluster relpath>" -> sha256 (None when unavailable)
    per_file_sha: bool                             # False for a store the reclaim table does not cover
    stale: Optional[str] = None                    # the index's stale note when it names this cell
    content_manifest_sha256: Dict[str, Optional[str]] = field(default_factory=dict)
    index_built_at: Optional[str] = None
    local: bool = False                            # True when the grading host copy still exists and was hashed

    @property
    def artifact_id(self) -> str:
        if self.per_file_sha and not self.stale and all(v for v in self.files.values()):
            lines = sorted(f"{k.split(':', 1)[0]}\t{k.split(':', 1)[1]}\t{v}" for k, v in self.files.items())
            return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
        return hashlib.sha256(self.uri_primary.encode("utf-8")).hexdigest()

    @property
    def artifact_id_rule(self) -> str:
        if self.per_file_sha and not self.stale and all(v for v in self.files.values()):
            return "sha256 over the sorted '<store>\\t<relpath>\\t<sha256>' lines of extra.files"
        return "sha256 of uri_primary (per-file shas unavailable or stale, see extra)"

    def artifact_payload(self) -> Dict[str, object]:
        extra: Dict[str, object] = {
            "pointer_kind": "transcript_cell_store_dir",
            "former_grading_host_dir": self.former_grading_host_dir, "cell_rel": self.cell_rel, "stores": list(self.stores),
            "n_files": len(self.files), "files": dict(sorted(self.files.items())),
            "artifact_id_rule": self.artifact_id_rule,
            "granularity_note": "the goal's row lives in ONE shard's progress.jsonl; the shard is recorded only inside the files, "
                                "so the pointer is the cell directory and generation_offset is null",
        }
        if self.local:
            extra["sha256_source"] = "the grading host copy, hashed when the adapter read (D12)"
        elif self.per_file_sha:
            extra["sha256_source"] = (f"{MIRRORS_DIR}/{TSV_NAME} (the grading host copy at reclaim time; the store's content manifest "
                                      "verified equal on origin and mirror)")
        else:
            extra["per_file_sha256"] = {"confidence": "unverified",
                                        "reason": f"{MIRRORS_DIR}/{TSV_NAME} predates this store's push; only the store-level content "
                                                  "manifest exists, so the cell path follows the store layout and is unverified"}
        if self.stale:
            extra["stale"] = {"confidence": "unverified", "reason": self.stale,
                              "grading_host_sha256_at_reclaim": "the shas in extra.files are the grading host copy's, which the note says differ from the cluster bytes"}
        if self.content_manifest_sha256:
            extra["content_manifest_sha256"] = dict(self.content_manifest_sha256)
        if self.index_built_at:
            extra["index_built_at"] = self.index_built_at
        # verified only when the reclaim table pins every file (the index build is when the cluster copies were checked);
        # a grading host copy is hashed but not "verified against the clusters"; a pushed pull's cell path is unverified
        verified = self.index_built_at if (self.per_file_sha and not self.local) else None
        return {"artifact_id": self.artifact_id, "kind": "store_root", "uri_primary": self.uri_primary,
                "locations_json": list(self.locations), "bytes": None, "verified_utc": verified, "extra": extra}


class TranscriptMirrors:
    """The mirror index + reclaim table under `<rs>/transcripts_mirrors/` (both read lazily)."""

    def __init__(self, rs: str):
        self.rs = rs
        self.dir = os.path.join(rs, MIRRORS_DIR)
        self.index_path = os.path.join(self.dir, INDEX_NAME)
        self._loaded = False
        self.stores: Dict[str, Dict[str, object]] = {}
        self.meta: Dict[str, object] = {}
        self._exact: Dict[str, List[str]] = {}
        self._globs: List[Tuple[str, str]] = []             # (pattern, store)
        self._shas: Dict[str, Dict[str, str]] = {}
        self._stale: List[Tuple[str, str]] = []             # (store, "<attack>/<cell>")

    @property
    def exists(self) -> bool:
        return os.path.lexists(self.index_path)

    @property
    def tsv_path(self) -> str:
        """The reclaim table: its current name, else the first spelling of it under the site's older names that exists (the table
        was written before the site renamed its grading host; :func:`site_aliases.legacy_keys`); the current name when none does,
        so a missing table is refused under the name a reader expects."""
        for spelling in SA.legacy_keys(TSV_NAME):
            p = os.path.join(self.dir, spelling)
            if os.path.lexists(p):
                return p
        return os.path.join(self.dir, TSV_NAME)

    def inputs(self) -> List[Input]:
        """The index and the reclaim table as inputs (refused by name when one is missing)."""
        if not self.exists:
            return []
        return [check_input(self.index_path, role="report"), check_input(self.tsv_path, role="rows")]

    # ---- loading -------------------------------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        if not self.exists:
            raise AdapterRefusal(self.index_path, "input missing")
        with open(self.index_path, "r", encoding="utf-8") as fh:
            try:
                idx = json.load(fh)
            except json.JSONDecodeError as exc:
                raise AdapterRefusal(self.index_path, f"not JSON: {exc.msg}") from exc
        self.meta = dict(idx.get("_meta") or {})
        self.stores = dict(idx.get("stores") or {})
        for name, e in self.stores.items():
            d = str(SA.get_compat(e or {}, "former_grading_host_dir") or "")      # an entry written under the older names reads the same
            if not d:
                continue
            if "*" in d:
                self._globs.append((d, name))
            else:
                self._exact.setdefault(d, []).append(name)
        note = self.meta.get("stale_note")
        if note:
            hits = _STALE_RE.findall(str(note))
            if not hits:
                raise AdapterRefusal(self.index_path, f"_meta.stale_note names no '<store> <attack>/<cell>:' this reader can attribute: {note!r}")
            self._stale = [(s, c) for s, c in hits]
        if not os.path.lexists(self.tsv_path):
            raise AdapterRefusal(self.tsv_path, "input missing")
        with open(self.tsv_path, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    raise AdapterRefusal(self.tsv_path, f"line {n}: expected 3 tab-separated fields, got {len(parts)}")
                self._shas.setdefault(parts[0], {})[parts[1]] = parts[2]
        self._loaded = True

    # ---- lookup --------------------------------------------------------------------------------
    def stores_for(self, grading_host_dir_rel: str) -> List[str]:
        self._load()
        out = list(self._exact.get(grading_host_dir_rel, []))
        for pat, name in self._globs:
            if fnmatch.fnmatchcase(grading_host_dir_rel, pat):
                out.append(name)
        return sorted(set(out))

    @staticmethod
    def _round_of(store: str) -> int:
        m = _ROUND_SUFFIX.search(store)
        return int(m.group("round")) if m else 1

    def _cluster_target(self, stores: List[str], relpath: str) -> Tuple[Optional[str], str, Optional[str]]:
        """(store, cluster relpath, refusal reason) for one grading host file of a cell."""
        if len(stores) == 1:
            return stores[0], relpath, None
        parts = relpath.split("/")
        shard = parts[3] if len(parts) >= 5 else ""
        by_round = {self._round_of(s): s for s in stores}
        m = _SHARD_RENAMED.match(shard)
        if m:
            rnd = int(m.group("round"))
            if rnd not in by_round:
                return None, relpath, f"{relpath}: shard names round {rnd} but no round-{rnd} store is indexed among {stores}"
            parts[3] = f"shard{m.group('shard')}"
            return by_round[rnd], "/".join(parts), None
        if _SHARD_PLAIN.match(shard) and 1 in by_round:
            return by_round[1], relpath, None
        return None, relpath, f"{relpath}: no shard-name rule maps it onto one of {stores} (index _meta.relu_note)"

    def _is_stale(self, store: str, cluster_relpath: str) -> Optional[str]:
        parts = cluster_relpath.split("/")
        tail = "/".join(parts[1:3]) if len(parts) >= 3 else ""
        for s, c in self._stale:
            if s == store and tail == c:
                return str(self.meta.get("stale_note"))
        return None

    def locate_run_dir(self, grading_host_dir_rel: str, run_dir_rel: str) -> Tuple[Optional[CellPointer], Optional[str]]:
        """The cell's cluster pointer, or (None, why) — a store with no index entry, or an indexed store whose reclaim table holds
        no file under the cell (the grading host never had it; whether a cluster does is unknown), is a refusal for the caller."""
        if not self.exists:
            return None, f"no {MIRRORS_DIR}/{INDEX_NAME} under {self.rs}"
        self._load()
        stores = self.stores_for(grading_host_dir_rel)
        if not stores:
            return None, f"transcript store {grading_host_dir_rel!r} has no entry in {MIRRORS_DIR}/{INDEX_NAME}"
        built_at = LC.iso_ms(str(self.meta.get("built_at") or ""), None)      # the registry's UTC-with-ms spelling, or None
        shas = self._shas.get(grading_host_dir_rel)
        files: Dict[str, Optional[str]] = {}
        used: List[str] = []
        stale: Optional[str] = None
        if shas is not None:
            prefix = run_dir_rel.rstrip("/") + "/"
            run_dir_files = {rel: sha for rel, sha in shas.items() if rel.startswith(prefix)}
            if not run_dir_files:
                return None, (f"transcript store {grading_host_dir_rel!r} is indexed but {MIRRORS_DIR}/{TSV_NAME} lists no file under "
                              f"{run_dir_rel!r}: the grading host copy never held this cell's transcripts")
            for rel, sha in sorted(run_dir_files.items()):
                store, crel, why = self._cluster_target(stores, rel)
                if store is None:
                    return None, f"transcript store {grading_host_dir_rel!r}: {why}"
                files[f"{store}:{crel}"] = sha
                if store not in used:
                    used.append(store)
                stale = stale or self._is_stale(store, crel)
            per_file = True
        else:
            if len(stores) != 1:
                return None, f"transcript store {grading_host_dir_rel!r} maps onto {len(stores)} indexed stores and has no reclaim table to split it by"
            used = list(stores)
            per_file = False
        used.sort(key=self._round_of)
        primary = used[0]
        pe = self.stores[primary] or {}
        # the index was written before the site renamed its clusters: its keys and its origin / mirror values may carry the older names,
        # read through the site alias table (site_aliases.get_compat / canonical / legacy_keys)
        primary_path = SA.get_compat(pe, "cluster_a_path")
        if not primary_path:
            return None, f"transcript store {grading_host_dir_rel!r}: index entry {primary!r} names no cluster_a_path"
        locations: List[Dict[str, object]] = []
        manifests: Dict[str, Optional[str]] = {}
        for s in used:
            e = self.stores[s] or {}
            manifests[s] = e.get("content_manifest_sha256")
            for cluster in ("cluster_a", "cluster_b"):
                p = SA.get_compat(e, f"{cluster}_path")
                if not p:
                    continue
                origin = str(e.get("origin") or "")
                role = ("origin" if any(origin.startswith(spelling) for spelling in SA.legacy_keys(cluster))
                        else ("mirror" if SA.canonical(e.get("mirror")) == cluster else (origin or "unrecorded")))
                locations.append({"uri": CLUSTER_URI[cluster].format(path=os.path.join(str(p), run_dir_rel)), "cluster": cluster, "store": s,
                                  "role": role, "verified_utc": built_at if per_file else None, "bytes": None})
        uri_primary = CLUSTER_URI["cluster_a"].format(path=os.path.join(str(primary_path), run_dir_rel))
        return CellPointer(grading_host_dir_rel, run_dir_rel, used, uri_primary, locations, files, per_file, stale, manifests, built_at), None


def local_run_dir_pointer(rs: str, grading_host_dir_rel: str, run_dir_rel: str) -> Tuple[Optional[CellPointer], Optional[str]]:
    """The pointer for a cell whose grading host copy still exists: every regular file under it hashed, the local path as the
    one location. (None, why) when the directory is absent or holds no regular file."""
    run_dir = os.path.join(rs, grading_host_dir_rel, run_dir_rel)
    if not os.path.isdir(run_dir):
        return None, f"no grading host copy at {run_dir}"
    files: Dict[str, Optional[str]] = {}
    for dirpath, dirnames, filenames in store_walk(run_dir):
        dirnames.sort()
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, os.path.join(rs, grading_host_dir_rel))
            inp = check_input(p, role="generations")
            files[f"{grading_host_dir_rel}:{rel}"] = inp.sha256
    if not files:
        return None, f"the grading host copy at {run_dir} holds no regular file"
    return CellPointer(grading_host_dir_rel, run_dir_rel, [grading_host_dir_rel], run_dir,
                       [{"uri": run_dir, "cluster": "grading_host", "store": grading_host_dir_rel, "role": "grading_host copy", "verified_utc": None, "bytes": None}],
                       files, True, None, {}, None, local=True), None
