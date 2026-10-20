#!/usr/bin/env python3
"""Check the shipped free text against its pre-image: only declared leaves changed, every loader still loads, and no
attribution survives in any free-text leaf.

The five checks (each can be run alone with ``--checks``):

1. **Only declared leaves change.** Every data file under ``configs/registry`` (and the byte-identical package copy of the
   decode-policy export) is re-derived from the pre-ref with :mod:`shipped_text`'s rewrite under the same private inputs, the
   declared pre-patches and, at integration, the codemod; the result must equal the committed file byte for byte. The frame is
   re-derived cell by cell (check 4).
2. **Transcripts.** Every roster/rulings pair loads through ``roster.load(..., strict_counts=True)`` before and after, and the
   loaded structures are equal with the declared prose, sentinel and drop leaves masked: no machine field of a scope, target,
   expiry or entry moved.
3. **Seeds.** ``dimensions.load_seeds`` and ``build_records`` before and after give the same records in the same order with the
   same ids and identity fields; only declared leaves differ.
4. **Frame.** The new frame directory has the old frame's tables, columns, row counts and row order; every non-TEXT cell is
   equal; every TEXT cell equals the policy's rewrite of its old value; the frame ships a Parquet file for every table the
   old frame did, and each has exactly its SQLite table's columns, in order, with the Parquet type the exporter writes for
   each declared type, and equals the new SQLite row for row, and a table no cell of which moved also keeps the pre-image's
   Parquet bytes (the changed tables are derived from the two SQLite files, and the MANIFEST's ``release_changed_rows`` must
   agree with them); the MANIFEST names its pre-image.
   A ruling row and the newest transcript's copy of it stay byte-comparable wherever they were.
5. **Attribution scan** over every free-text leaf of the frame (its SQLite and every Parquet file's string columns and
   metadata), the transcripts and the seeds, every mapping key in them (at ``<object address><key>``), and every YAML
   comment, whatever class the policy gives the leaf. The only
   exemptions are the policy's ``exempt`` rows, by address and value hash. With ``--scan-tool`` the publication scanner
   (``reproduce/tools/scrub_scan.py``) also reads the same data in its leaf scope, and a hit its allowlist does not cover
   fails the check too: two independent scanners over the shipped text.

Usage::

    check_shipped_text.py --pre-ref <sha> --policy P --inventory I [--rules R] [--table T ...] [--comments C]
                          [--pre-patch PATCH ...] [--codemod-rules F] [--work DIR] [--checks 1,2,3,4,5] [--report OUT]
                          [--scan-tool reproduce/tools/scrub_scan.py [--scan-allowlist TSV] [--scan-jobs N]]

The private inputs are passed by path and are the ones the rewrite used; nothing here names them.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import shipped_text as ST  # noqa: E402  (the sibling module; this checker re-uses its walkers and its rewrite)

DROPPED = "<key dropped>"
MASKED = ("prose", "sentinel", "drop", "comment", "pin")
#: identifier and read leaves move only through the declared site map, which rewrites every copy alike so joins hold
SITE = ("identifier", "read")


class CheckFailed(RuntimeError):
    pass


@dataclass
class Report:
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def start(self, name: str) -> Dict[str, Any]:
        self.checks[name] = {"ok": True, "failures": [], "notes": {}}
        return self.checks[name]

    @staticmethod
    def fail(entry: Dict[str, Any], msg: str, limit: int = 200) -> None:
        entry["ok"] = False
        if len(entry["failures"]) < limit:
            entry["failures"].append(msg)
        entry["n_failures"] = entry.get("n_failures", 0) + 1

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks.values())


# --------------------------------------------------------------------------------------------------------------------------------
# the pre-image
# --------------------------------------------------------------------------------------------------------------------------------


def git(tree: str, *args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", tree, *args], check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode("utf-8")


def pre_files(tree: str, ref: str) -> List[str]:
    names = git(tree, "ls-tree", "--name-only", f"{ref}:{ST.REGISTRY_DIR}").split()
    return sorted(os.path.join(ST.REGISTRY_DIR, n) for n in names if ST.unit_scope(n) is not None)


def pre_text(tree: str, ref: str, rel: str) -> str:
    return git(tree, "show", f"{ref}:{rel}")


def apply_patches(texts: Dict[str, str], patches: Sequence[str]) -> Dict[str, str]:
    """``texts`` (repo-relative path -> text) with the declared unified diffs applied, in order (the rewrite's own applier)."""
    for p in patches:
        with open(p, "r", encoding="utf-8") as fh:
            named = set(ST.split_patch(fh.read()))
        missing = named - set(texts)
        if missing:
            raise CheckFailed(f"pre-patch {p} names files that are not registry data files: {sorted(missing)}")
    return {rel: ST.prepatch_text(t, rel, patches) for rel, t in texts.items()}


def load_codemod(path: Optional[str], tree: str):
    """The codemod's rule table (its ``rewrite_text(text, path)``), loaded from the tree's own ``scripts/release``."""
    if not path:
        return None
    sys.path.insert(0, os.path.join(tree, "scripts", "release"))
    import rename_sites  # noqa: E402  (present on the integrated tree)
    return rename_sites.load_rules(path)


# --------------------------------------------------------------------------------------------------------------------------------
# masked comparison
# --------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Substitutions:
    """Every (old value -> new value) the policy declared for one scope's masked classes, and the keys it dropped."""
    exact: Dict[str, Any] = field(default_factory=dict)
    dropped_keys: set = field(default_factory=set)
    dropped_values: set = field(default_factory=set)
    key_renames: Dict[str, List[str]] = field(default_factory=dict)
    site_renames: int = 0

    def add(self, ch: "ST.Change") -> None:
        if ch.cls == "key" and isinstance(ch.old, str) and isinstance(ch.new, str):
            self.key_renames.setdefault(ch.old, []).append(ch.new)
            return
        if ch.cls not in MASKED + SITE or not isinstance(ch.old, str):
            return
        if ch.cls in SITE:
            self.site_renames += 1
        if ch.new == DROPPED:
            self.dropped_keys.add(ch.addr.rsplit(".", 1)[-1].split("[", 1)[0])
            self.dropped_values.add(ch.old)
            return
        self.exact.setdefault(ch.old, []).append(ch.new)

    def explains(self, old: str, new: Any) -> bool:
        if new in self.exact.get(old, ()):
            return True
        if not isinstance(new, str):
            return False
        ok_a, ja = ST.parse_json_cell(old)
        ok_b, jb = ST.parse_json_cell(new)
        if ok_a and ok_b:
            return not masked_diff(ja, jb, self, "$", limit=1)
        # a string a loader joined from several leaves ("; ".join): every part equal or a declared rewrite of its own
        for sep in ("; ", " | ", "\n"):
            pa, pb = old.split(sep), new.split(sep)
            if len(pa) > 1 and len(pa) == len(pb):
                if all(x == y or y in self.exact.get(x, ()) for x, y in zip(pa, pb)):
                    return True
        # else a string a loader composed around leaves: mask every declared old value (long enough not to occur by chance)
        # in the old string and every declared new value of those in the new one; the remainders must be equal
        keys = sorted((k for k in self.exact if len(k) >= 12 and k in old), key=len, reverse=True)
        if not keys:
            return False
        a = old
        for k in keys:
            a = a.replace(k, "\x00")
        vals = sorted({v for k in keys for v in self.exact[k] if isinstance(v, str) and v}, key=len, reverse=True)
        b = new
        for v in vals:
            b = b.replace(v, "\x00")
        return a == b


def masked_diff(a: Any, b: Any, subs: Substitutions, path: str = "$", out: Optional[List[str]] = None, limit: int = 50
                ) -> List[str]:
    """Differences between ``a`` and ``b`` the substitutions do not explain (each as ``path: before -> after``)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        renamed_to = set()
        for k in a:
            if k not in b:
                if k in subs.dropped_keys:
                    continue
                # a key the policy renamed: its value is compared under the new key
                nk = next((n for n in subs.key_renames.get(k, ()) if n in b and n not in a), None)
                if nk is not None:
                    renamed_to.add(nk)
                    masked_diff(a[k], b[nk], subs, f"{path}.{k}", out, limit)
                    continue
                out.append(f"{path}.{k}: key removed")
            else:
                masked_diff(a[k], b[k], subs, f"{path}.{k}", out, limit)
        for k in b:
            if k not in a and k not in renamed_to:
                out.append(f"{path}.{k}: key added")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            kept = [x for x in a if not (isinstance(x, str) and x in subs.dropped_values)]
            if len(kept) != len(b):
                out.append(f"{path}: list length {len(a)} -> {len(b)}")
                return out
            a = kept
        for i, (x, y) in enumerate(zip(a, b)):
            masked_diff(x, y, subs, f"{path}[{i}]", out, limit)
        return out
    if isinstance(a, str) and (isinstance(b, str) or b is None):
        if a != b and not subs.explains(a, b):
            out.append(f"{path}: {a[:120]!r} -> {b if b is None else b[:120]!r}")
        return out
    if a != b or type(a) is not type(b):
        out.append(f"{path}: {a!r} -> {b!r}")
    return out


# --------------------------------------------------------------------------------------------------------------------------------
# check 1: files
# --------------------------------------------------------------------------------------------------------------------------------


@dataclass
class FileState:
    before: Dict[str, str]          # pre-image after the pre-patches and the codemod (what the loaders read "before")
    after: Dict[str, str]           # the committed files
    changes: Dict[str, List["ST.Change"]]


def check_files(ctx: "ST.Context", tree: str, ref: str, patches: Sequence[str], codemod, rep: Report,
                pre_repo: Optional[str] = None) -> FileState:
    e = rep.start("1_files")
    pre_repo = pre_repo or tree
    rels = pre_files(pre_repo, ref)
    now = ST.registry_files(tree)
    if rels != now:
        Report.fail(e, f"the configs/registry data files differ in name: removed {sorted(set(rels) - set(now))}, "
                       f"added {sorted(set(now) - set(rels))}")
    pre = apply_patches({r: pre_text(pre_repo, ref, r) for r in rels}, patches)
    before, after, changes = {}, {}, {}
    by_class: Dict[str, int] = {}
    for rel in rels:
        if rel not in now:
            continue
        with open(os.path.join(tree, rel), "r", encoding="utf-8") as fh:
            got = fh.read()
        ch: List[ST.Change] = []
        exp = exp_rewrite = ST.rewrite_text(ctx, rel, pre[rel], ch)
        b = pre[rel]
        if codemod is not None:
            exp = codemod.rewrite_text(exp, rel)
            b = codemod.rewrite_text(b, rel)
        for c in ch:
            by_class[c.cls] = by_class.get(c.cls, 0) + 1
        if codemod is not None:
            # the loaders read the codemodded pre-image and the committed (rewritten, then codemodded) file, so every leaf the
            # rewrite changed is also recorded in its codemodded form, read back from the two files (the codemod's renames
            # depend on context, so a leaf is never codemodded on its own); leaves pair by record and address. A leaf the
            # rewrite moved (a declared key rename) is absent from the rewrite at its old address but is not dropped: only a
            # leaf under an address the rewrite itself dropped is recorded as dropped, so the old key never joins the dropped
            # keys the masked comparison skips (which would report the renamed key as added)
            pm, em, bm, am = (_leafmap(rel, t) for t in (pre[rel], exp_rewrite, b, got))
            dropped_at = [c.addr for c in ch if c.new == DROPPED]

            def _rewrite_dropped(addr: str) -> bool:
                return any(addr == d or addr.startswith(d + "[") or addr.startswith(d + ".") for d in dropped_at)
            ch = ch + [ST.Change(k[1], "prose", bm[k], am.get(k, DROPPED)) for k, v in pm.items()
                       if k in bm and (em[k] != v if k in em else _rewrite_dropped(k[1]))]
        before[rel], after[rel], changes[rel] = b, got, ch
        if exp != got:
            Report.fail(e, f"{rel}: the committed file is not the policy's rewrite of its pre-image; first differences: "
                           + "; ".join(leaf_diff(rel, exp, got)[:8]))
        for mirror in ST.MIRRORS.get(rel, ()):
            mp = os.path.join(tree, mirror)
            with open(mp, "r", encoding="utf-8") as fh:
                if fh.read() != got:
                    Report.fail(e, f"{mirror} is not byte-identical to {rel}")
    e["notes"] = {"files": len(rels), "files_changed": sum(1 for r in rels if before.get(r) != after.get(r)),
                  "changed_leaves_by_class": by_class}
    return FileState(before, after, changes)


def _leafmap(rel: str, text: str) -> Dict[Tuple[int, str], str]:
    """{(record index, concrete address): value} for every string leaf of one data file (a JSONL file's records share their
    addresses, so the record index tells them apart; other files are one record)."""
    fname = os.path.basename(rel)
    out: Dict[Tuple[int, str], str] = {}
    if fname.endswith(".jsonl"):
        scope = ST.unit_scope(fname)
        for i, (_, rec) in enumerate(ST.jsonl_records(text)):
            unit = ST._jsonl_unit(fname, rec)
            for k, v in rec.items():
                for p, sv in ST.walk_leaves(v):
                    out[(i, ST.address(scope, unit, k, p, False))] = sv
        return out
    for sc, u, c, p, v in ST.file_leaves(rel, text):
        out[(0, ST.address(sc, u, c, p, False))] = v
    return out


def leaf_diff(rel: str, a: str, b: str) -> List[str]:
    """Leaf addresses whose values differ between two versions of one data file (or a line diff when they do not parse)."""
    try:
        la = {ST.address(s, u, c, p, False): v for s, u, c, p, v in ST.file_leaves(rel, a)}
        lb = {ST.address(s, u, c, p, False): v for s, u, c, p, v in ST.file_leaves(rel, b)}
    except Exception as exc:  # noqa: BLE001 -- a committed file that no longer parses is itself the finding
        return [f"does not parse ({exc})"]
    out = [f"{k}: expected {la.get(k, '<absent>')[:80]!r}, committed {lb.get(k, '<absent>')[:80]!r}"
           for k in sorted(set(la) | set(lb)) if la.get(k) != lb.get(k)]
    if not out:
        al, bl = a.split("\n"), b.split("\n")
        out = [f"line {i + 1}: expected {x[:80]!r}, committed {y[:80]!r}" for i, (x, y) in enumerate(itertools.zip_longest(al, bl, fillvalue=""))
               if x != y]
    return out


def subs_for(fs: FileState, scope: str) -> Substitutions:
    s = Substitutions()
    for rel, chs in fs.changes.items():
        if ST.unit_scope(rel) == scope:
            for c in chs:
                s.add(c)
    return s


def _write_dir(texts: Mapping[str, str], rels: Iterable[str]) -> str:
    d = tempfile.mkdtemp(prefix="cst-")
    for rel in rels:
        with open(os.path.join(d, os.path.basename(rel)), "w", encoding="utf-8") as fh:
            fh.write(texts[rel])
    return d


# --------------------------------------------------------------------------------------------------------------------------------
# check 2: transcripts
# --------------------------------------------------------------------------------------------------------------------------------


def roster_pairs(rels: Iterable[str]) -> List[Tuple[str, str]]:
    names = {os.path.basename(r): r for r in rels}
    out = []
    for n, r in sorted(names.items()):
        if n.startswith("roster_r2_v") and n.endswith(".yaml"):
            v = n[len("roster_r2_v"):-len(".yaml")]
            rul = names.get(f"rulings_r2_v{v}.jsonl")
            if rul:
                out.append((r, rul))
    return sorted(out, key=lambda p: int(os.path.basename(p[0])[len("roster_r2_v"):-len(".yaml")]))


def _roster_load_dict(roster_mod, roster_path: str, rulings_path: str) -> Dict[str, Any]:
    ld = roster_mod.load(roster_path, rulings_path, strict_counts=True)
    return {"round_id": ld.round_id, "version": ld.version, "paper_round": ld.paper_round, "entries": ld.entries,
            "rulings": ld.rulings, "dispositions": ld.dispositions, "header": ld.header, "counts": ld.counts_by_comparison,
            # the loader sorts this list by its free text, so it is compared by the entries' stable ids
            "disagreements": sorted(ld.disagreements, key=lambda e: e["roster_entry_id"])}


def check_transcripts(fs: FileState, tree: str, rep: Report) -> None:
    e = rep.start("2_transcripts")
    sys.path.insert(0, os.path.join(tree, "src"))
    from sisteer.registry.adapters import roster as RO
    subs = subs_for(fs, "transcripts")
    pairs = roster_pairs(fs.after)
    if not pairs:
        Report.fail(e, "no roster/rulings pair found")
    loaded = 0
    for ro, ru in pairs:
        dirs = [_write_dir(fs.before, (ro, ru)), _write_dir(fs.after, (ro, ru))]
        try:
            got = []
            for d in dirs:
                try:
                    got.append(_roster_load_dict(RO, os.path.join(d, os.path.basename(ro)), os.path.join(d, os.path.basename(ru))))
                except Exception as exc:  # noqa: BLE001 -- the loader's refusal is the finding
                    Report.fail(e, f"{os.path.basename(ro)} + {os.path.basename(ru)} ({'before' if d is dirs[0] else 'after'}): "
                                   f"roster.load refused: {exc}")
                    got.append(None)
            if None in got:
                continue
            loaded += 1
            for d in masked_diff(got[0], got[1], subs, "$", limit=20):
                Report.fail(e, f"{os.path.basename(ro)} + {os.path.basename(ru)}: {d}")
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
    e["notes"] = {"pairs": len(pairs), "loaded_before_and_after": loaded, "masked_values": len(subs.exact),
                  "site_renamed_leaves": subs.site_renames}


# --------------------------------------------------------------------------------------------------------------------------------
# check 3: seeds
# --------------------------------------------------------------------------------------------------------------------------------

#: the column that names a dimension record, per record type (the record id itself hashes the payload)
ID_COLUMNS = {"model": ("model_id",), "eval": ("eval_id",), "item_set": ("item_set_id",), "judge": ("judge_id",),
              "instrument": ("instrument_id",), "decode_policy": ("policy_id",), "si_text": ("si_id",), "object": ("object_id",),
              "sampling_mode_policy": ("model_id", "sampling_mode")}


def check_seeds(fs: FileState, tree: str, rep: Report) -> None:
    e = rep.start("3_seeds")
    sys.path.insert(0, os.path.join(tree, "src"))
    from sisteer.registry import identity as ID
    from sisteer.registry.adapters import dimensions as DIM
    subs = subs_for(fs, "seeds")
    files = [r for r in fs.after if os.path.basename(r) in set(DIM.SEED_FILES.values())]
    dirs = [_write_dir(fs.before, files), _write_dir(fs.after, files)]
    from sisteer.registry import harvest as H
    try:
        bundles = []
        for d in dirs:
            prod = dict(H.harvester_producer(d))
            prod["script"] = DIM.SCRIPT
            try:
                seeds = DIM.load_seeds(d)
                bundles.append(DIM.build_records(seeds, prod, now="2026-01-01T00:00:00.000Z", verify_si=False))
            except Exception as exc:  # noqa: BLE001 -- the loader's refusal is the finding
                Report.fail(e, f"seeds ({'before' if d is dirs[0] else 'after'}): the dimensions load refused: {exc}")
                bundles.append(None)
        if None in bundles:
            return
        ident = {"instrument": [n for n, _ in ID.INSTRUMENT_ID_FIELDS], "judge": list(ID.JUDGE_ID_FIELDS)}
        n_ids = 0
        for t in DIM.TABLE_ORDER:
            ra, rb = bundles[0].records.get(t, []), bundles[1].records.get(t, [])
            if len(ra) != len(rb):
                Report.fail(e, f"{t}: {len(ra)} records before, {len(rb)} after")
                continue
            for x, y in zip(ra, rb):
                pa, pb = x["payload"], y["payload"]
                key = tuple(pa.get(c) for c in ID_COLUMNS.get(t, ()))
                if key != tuple(pb.get(c) for c in ID_COLUMNS.get(t, ())):
                    Report.fail(e, f"{t}: record id {key} became {tuple(pb.get(c) for c in ID_COLUMNS.get(t, ()))}")
                    continue
                n_ids += 1
                for f in ident.get(t, ()):
                    if pa.get(f) != pb.get(f):
                        Report.fail(e, f"{t} {key}: identity field {f} moved: {pa.get(f)!r} -> {pb.get(f)!r}")
                for d in masked_diff(pa, pb, subs, f"{t}{list(key)}", limit=10):
                    Report.fail(e, d)
        e["notes"] = {"records_with_equal_ids": n_ids, "counts": bundles[1].counts(), "masked_values": len(subs.exact),
                      "site_renamed_leaves": subs.site_renames}
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------------------------------
# check 4: the frame
# --------------------------------------------------------------------------------------------------------------------------------


def reassemble(parts: Sequence[str], out: str) -> str:
    """Concatenate gzip parts and decompress them to ``out``; returns the sha256 of the result."""
    import hashlib
    h = hashlib.sha256()
    with open(out, "wb") as dst:
        rd = _ConcatReader(parts)
        with gzip.GzipFile(fileobj=rd, mode="rb") as gz:
            for chunk in iter(lambda: gz.read(1 << 20), b""):
                dst.write(chunk)
                h.update(chunk)
    return h.hexdigest()


class _ConcatReader:
    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        self.fh = None

    def read(self, n: int = -1) -> bytes:
        while True:
            if self.fh is None:
                if not self.paths:
                    return b""
                self.fh = open(self.paths.pop(0), "rb")
            b = self.fh.read(n)
            if b:
                return b
            self.fh.close()
            self.fh = None


def frame_pre_post(tree: str, ref: str, work: str, pre_repo: Optional[str] = None
                   ) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], str, str]:
    """(pre SQLite, post SQLite, pre MANIFEST, post MANIFEST, pre dir in git, post dir on disk), reassembled under ``work``."""
    os.makedirs(work, exist_ok=True)
    pre_repo = pre_repo or tree
    pre_id = git(pre_repo, "show", f"{ref}:{ST.SNAPSHOT_DIR}/CURRENT").strip()
    pre_rel = f"{ST.SNAPSHOT_DIR}/snapshots/{pre_id}"
    pre_m = json.loads(git(pre_repo, "show", f"{ref}:{pre_rel}/MANIFEST.json"))
    pre_parts = []
    for p in pre_m["sqlite_parts"]:
        dst = os.path.join(work, "pre_" + p)
        with open(dst, "wb") as fh:
            fh.write(git(pre_repo, "show", f"{ref}:{pre_rel}/{p}", binary=True))
        pre_parts.append(dst)
    pre_sqlite = os.path.join(work, "pre_paper_frame.sqlite")
    got = reassemble(pre_parts, pre_sqlite)
    for p in pre_parts:
        os.remove(p)
    if got != pre_m["sqlite_sha256"]:
        raise CheckFailed(f"the pre-ref frame reassembles to {got}, its MANIFEST says {pre_m['sqlite_sha256']}")
    with open(os.path.join(tree, ST.SNAPSHOT_DIR, "CURRENT"), "r", encoding="utf-8") as fh:
        post_id = fh.read().strip()
    post_dir = os.path.join(tree, ST.SNAPSHOT_DIR, "snapshots", post_id)
    with open(os.path.join(post_dir, "MANIFEST.json"), "r", encoding="utf-8") as fh:
        post_m = json.load(fh)
    post_sqlite = os.path.join(work, "post_paper_frame.sqlite")
    got = reassemble([os.path.join(post_dir, p) for p in post_m["sqlite_parts"]], post_sqlite)
    if got != post_m["sqlite_sha256"]:
        raise CheckFailed(f"the committed frame reassembles to {got}, its MANIFEST says {post_m['sqlite_sha256']}")
    return pre_sqlite, post_sqlite, pre_m, post_m, pre_rel, post_dir


def frame_mapping(ctx: "ST.Context", conn: sqlite3.Connection, table: str, column: str) -> Dict[str, Any]:
    """{old TEXT value: new value} for every distinct value of one column the policy changes (the derivation's own rule)."""
    out = {}
    for v, _n in ST.frame_text_values(conn, table, column):
        if not ctx.may_change("frame", table, column, v):
            continue
        nv = ST.rewrite_frame_cell(ctx, table, column, v)
        if nv != v:
            out[v] = nv
    return out


def check_frame(ctx: "ST.Context", tree: str, ref: str, work: str, fs: Optional[FileState], rep: Report,
                inputs_sha: Optional[str] = None, pre_repo: Optional[str] = None) -> Optional[str]:
    """Check 4; returns the post SQLite path for check 5."""
    e = rep.start("4_frame")
    pre_sqlite, post_sqlite, pre_m, post_m, pre_rel, post_dir = frame_pre_post(tree, ref, work, pre_repo)
    # the MANIFEST
    if post_m.get("derived_from") != pre_m["snapshot_id"]:
        Report.fail(e, f"MANIFEST derived_from {post_m.get('derived_from')!r} is not the pre-ref frame {pre_m['snapshot_id']!r}")
    if post_m.get("tables") != pre_m.get("tables"):
        Report.fail(e, "MANIFEST tables (row counts) differ from the pre-ref frame's")
    if int(post_m.get("ingest_seq", -1)) != int(pre_m["ingest_seq"]) or not post_m["snapshot_id"].startswith(f"{int(pre_m['ingest_seq']):012d}-"):
        Report.fail(e, f"snapshot id {post_m['snapshot_id']!r} does not keep the pre-ref ingest_seq {pre_m['ingest_seq']}")
    if inputs_sha and post_m.get("release_rules_sha256") != inputs_sha:
        Report.fail(e, f"MANIFEST release_rules_sha256 {post_m.get('release_rules_sha256')} is not the sha of the inputs given "
                       f"({inputs_sha})")
    for fn, meta in (post_m.get("files") or {}).items():
        p = os.path.join(post_dir, fn)
        if not os.path.exists(p) or ST.sha256_file(p) != meta["sha256"]:
            Report.fail(e, f"MANIFEST files[{fn}] does not match the file on disk")
    on_disk = set(os.listdir(post_dir)) - {"MANIFEST.json"}
    if on_disk != set(post_m.get("files") or {}):
        Report.fail(e, f"files on disk and in the MANIFEST differ: {sorted(on_disk ^ set(post_m.get('files') or {}))}")
    a = sqlite3.connect(f"file:{pre_sqlite}?mode=ro", uri=True)
    b = sqlite3.connect(f"file:{post_sqlite}?mode=ro", uri=True)
    try:
        ta, tb = ST.frame_tables(a), ST.frame_tables(b)
        if ta != tb:
            Report.fail(e, f"tables differ: {sorted(set(ta) ^ set(tb))}")
        changed_cells = 0
        derived: Dict[str, Dict[str, int]] = {}          # {table: {column: rows whose cell moved}}, from the two SQLite files
        for t in ta:
            if t not in tb:
                continue
            info_a = [tuple(r[1:3]) for r in a.execute(f'PRAGMA table_info("{t}")')]
            info_b = [tuple(r[1:3]) for r in b.execute(f'PRAGMA table_info("{t}")')]
            if info_a != info_b:
                Report.fail(e, f"{t}: columns differ")
                continue
            cols = [c for c, _ in info_a]
            maps = {i: frame_mapping(ctx, a, t, c) for i, c in enumerate(cols)}
            maps = {i: m for i, m in maps.items() if m}
            sel = ", ".join(f'"{c}"' for c in cols)
            ca = a.execute(f'SELECT {sel} FROM "{t}" ORDER BY rowid')
            cb = b.execute(f'SELECT {sel} FROM "{t}" ORDER BY rowid')
            n = 0
            for ra, rb in itertools.zip_longest(ca, cb):
                n += 1
                if ra is None or rb is None:
                    Report.fail(e, f"{t}: row counts differ (from row {n})")
                    break
                if ra == rb:
                    for i, m in maps.items():
                        if ra[i] in m:
                            Report.fail(e, f"{t}.{cols[i]} row {n}: the policy rewrites {ra[i][:80]!r} but the frame kept it")
                    continue
                for i, (x, y) in enumerate(zip(ra, rb)):
                    if x == y and type(x) is type(y):
                        if i in maps and x in maps[i]:
                            Report.fail(e, f"{t}.{cols[i]} row {n}: the policy rewrites {x[:80]!r} but the frame kept it")
                        continue
                    if isinstance(x, str) and i in maps and x in maps[i] and maps[i][x] == y:
                        changed_cells += 1
                        derived.setdefault(t, {})[cols[i]] = derived.get(t, {}).get(cols[i], 0) + 1
                        continue
                    Report.fail(e, f"{t}.{cols[i]} row {n}: {x if not isinstance(x, str) else x[:80]!r} -> "
                                   f"{y if not isinstance(y, str) else y[:80]!r} is not the policy's rewrite")
        e["notes"]["changed_cells"] = changed_cells
        e["notes"]["tables"] = len(ta)
        _check_parquet(b, pre_m, post_m, post_dir, derived, e)
        if fs is not None:
            _check_frame_transcript_pairs(a, b, fs, e)
    finally:
        a.close()
        b.close()
    os.remove(pre_sqlite)
    return post_sqlite


def _check_parquet(b: sqlite3.Connection, pre_m, post_m, post_dir, derived: Mapping[str, Mapping[str, int]], e) -> None:
    """Every shipped Parquet file against the new SQLite, whatever the MANIFEST says about it: one file for every table the
    pre-ref frame shipped as Parquet, each with the SQLite table's columns and types, the same rows in the same order and every
    cell equal. A table no cell of which moved (``derived``, from the two SQLite files) keeps the pre-image's bytes, and the
    MANIFEST's ``release_changed_rows`` must name exactly the derived changes."""
    claimed = {t: dict(cs) for t, cs in (post_m.get("release_changed_rows") or {}).items()}
    want = {t: dict(cs) for t, cs in derived.items()}
    if claimed != want:
        diff = sorted(set(claimed) ^ set(want)) or sorted(t for t in want if claimed.get(t) != want[t])
        Report.fail(e, f"MANIFEST release_changed_rows is not the change the two SQLite files show (tables {diff})")
    if list(post_m.get("parquet_tables") or []) != list(pre_m.get("parquet_tables") or []):
        lost = sorted(set(pre_m.get("parquet_tables") or []) - set(post_m.get("parquet_tables") or []))
        Report.fail(e, f"MANIFEST parquet_tables is not the pre-ref frame's (missing {lost})")
    on_disk = sorted(fn[:-len(".parquet")] for fn in os.listdir(post_dir) if fn.endswith(".parquet"))
    if on_disk != sorted(post_m.get("parquet_tables") or []):
        Report.fail(e, f"Parquet files on disk {on_disk} are not the MANIFEST's parquet_tables")
    compared = 0
    for t in on_disk:
        fn = f"{t}.parquet"
        if t not in derived and (post_m["files"].get(fn) or {}).get("sha256") != (pre_m["files"].get(fn) or {}).get("sha256"):
            Report.fail(e, f"{fn}: no cell of {t} moved, yet its Parquet is not the pre-ref file")
        bad = _parquet_differs(b, t, os.path.join(post_dir, fn))
        if bad:
            Report.fail(e, f"{fn}: {bad}")
        compared += 1
    e["notes"]["parquet_tables_compared"] = compared


#: the Parquet type the exporter writes for each SQLite storage class it declares (``sisteer.registry.snapshot._write_parquet``)
PARQUET_TYPE_OF = {"INTEGER": "int64", "REAL": "double", "TEXT": "string", "BLOB": "binary"}


def _parquet_schema_differs(b: sqlite3.Connection, t: str, schema) -> Optional[str]:
    """None when the Parquet schema is the SQLite table's columns in order, each with the Parquet type of its declared type."""
    want = [(r[1], PARQUET_TYPE_OF.get(str(r[2]).upper(), f"(no Parquet type for {r[2]!r})"))
            for r in b.execute(f'PRAGMA table_info("{t}")')]
    got = [(f.name, str(f.type)) for f in schema]
    if got == want:
        return None
    if not want:
        return "the SQLite has no such table"
    wn, gn = [c for c, _ in want], [c for c, _ in got]
    if wn != gn:
        missing, extra = [c for c in wn if c not in gn], [c for c in gn if c not in wn]
        if missing or extra:
            return f"its columns are not the SQLite table's (missing {missing}, extra {extra})"
        return f"its columns {gn} are not in the SQLite table's order {wn}"
    c, w, g = next((c, w, g) for (c, w), (_, g) in zip(want, got) if w != g)
    return f"column {c} is {g} in the Parquet, where the exporter writes {w} for its declared type"


def _parquet_differs(b: sqlite3.Connection, t: str, path: str, batch_rows: int = 65536) -> Optional[str]:
    """None when the Parquet file holds the SQLite table exactly: the same columns in the same order with the exporter's type
    for each declared type, and the same rows (ORDER BY rowid) cell for cell. Else the first difference."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    bad = _parquet_schema_differs(b, t, pf.schema_arrow)
    if bad:
        return bad
    cols = list(pf.schema_arrow.names)
    try:
        n = b.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        cur = b.execute(f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{t}" ORDER BY rowid')
    except sqlite3.Error as exc:
        return f"its columns {cols} do not read from the SQLite table ({exc})"
    if pf.metadata.num_rows != n:
        return f"{pf.metadata.num_rows} rows, the SQLite has {n}"
    start = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        rows = cur.fetchmany(batch.num_rows)
        if len(rows) != batch.num_rows:
            return f"the SQLite ran out of rows at row {start + len(rows)}"
        for j, c in enumerate(cols):
            vals, sql = batch.column(j).to_pylist(), [r[j] for r in rows]
            if vals != sql:
                i = next(i for i, (x, y) in enumerate(zip(vals, sql)) if x != y or type(x) is not type(y))
                return f"column {c} row {start + i} differs from the SQLite"
        start += batch.num_rows
    return None


def _check_frame_transcript_pairs(a, b, fs: FileState, e) -> None:
    """A ruling present in the frame and in the newest rulings transcript: wherever a text column (or JSON leaf) was equal
    before, it is equal after."""
    newest = max((r for r in fs.after if os.path.basename(r).startswith("rulings_r2_v")),
                 key=lambda r: int(os.path.basename(r)[len("rulings_r2_v"):-len(".jsonl")]), default=None)
    if newest is None:
        return
    def recs(text):
        out = {}
        for _, rec in ST.jsonl_records(text):
            if rec.get("ruling_id"):
                out[rec["ruling_id"]] = rec
        return out
    tb, ta = recs(fs.before[newest]), recs(fs.after[newest])
    cols = [r[1] for r in a.execute('PRAGMA table_info("ruling")')]
    checked = 0
    for col in ("reason", "source_ref", "ruled_by", "scope_json", "target_json", "expires_when_json"):
        if col not in cols:
            continue
        fa = dict(a.execute(f'SELECT ruling_id, "{col}" FROM ruling'))
        fb = dict(b.execute(f'SELECT ruling_id, "{col}" FROM ruling'))
        for rid, v0 in fa.items():
            if rid not in tb or rid not in ta or col not in tb[rid]:
                continue
            t0, t1, v1 = tb[rid][col], ta[rid].get(col), fb.get(rid)
            if isinstance(t0, (dict, list)):
                ok0, j0 = ST.parse_json_cell(v0) if isinstance(v0, str) else (False, None)
                ok1, j1 = ST.parse_json_cell(v1) if isinstance(v1, str) else (False, None)
                if ok0 and ok1:
                    for (p, x0), (_, x1) in zip(ST.walk_leaves(j0), ST.walk_leaves(j1)):
                        y0 = _at(t0, p)
                        if y0 is not None and y0 == x0 and _at(t1, p) != x1:
                            Report.fail(e, f"ruling {rid} {col}{ST.path_str(p, False)}: the frame and the newest transcript "
                                           f"agreed before and differ after")
                        checked += 1
            elif t0 == v0:
                checked += 1
                if t1 != v1:
                    Report.fail(e, f"ruling {rid} {col}: the frame and the newest transcript agreed before and differ after")
    e["notes"]["frame_transcript_leaves_compared"] = checked


def _at(obj, path):
    for p in path:
        try:
            obj = obj[p]
        except (KeyError, IndexError, TypeError):
            return None
    return obj


# --------------------------------------------------------------------------------------------------------------------------------
# check 5: the attribution scan
# --------------------------------------------------------------------------------------------------------------------------------


#: every hit of the scan, for a caller that wants them all (``--hits-out``), and the exemptions that matched (``--exempt-used-out``)
HITS: List[Dict[str, Any]] = []
USED: set = set()


def scan_leaf(ctx: "ST.Context", scope: str, unit: str, column: str, path: Sequence[object], value: str, e, seen: set) -> None:
    u, c = ctx.policy.alias(scope, unit, column)
    pat = ST.address(scope, u, c, path)
    key = (pat, value)
    if key in seen:
        return
    seen.add(key)
    if not ctx.scanner.any(value):
        return
    if (pat, ST.sha256_text(value)) in ctx.policy.exempt:
        e["notes"]["exempted"] = e["notes"].get("exempted", 0) + 1
        USED.add((pat, ST.sha256_text(value)))
        return
    hits = ctx.scanner.hits(value)
    HITS.append({"address": pat, "class": ctx.classify(scope, unit, column, path, value)[0], "value_sha256": ST.sha256_text(value),
                 "value": value, "hits": [h[0] for h in hits]})
    Report.fail(e, f"{pat}: {[h[0] for h in hits]} {[h[1] for h in hits][:3]} in {value[:160]!r}", limit=500)


def scan_key(ctx: "ST.Context", scope: str, unit: str, column: Optional[str], path: Sequence[object], key: str, e, seen: set) -> None:
    """One mapping key, scanned like a leaf at its key address (``<object address><key>``); exemptions by that address."""
    if column is not None:
        unit, column = ctx.policy.alias(scope, unit, column)
    pat = ST.key_address(scope, unit, column, path)
    if (pat, key) in seen:
        return
    seen.add((pat, key))
    if not ctx.scanner.any(key):
        return
    h = ST.sha256_text(key)
    if (pat, h) in ctx.policy.exempt:
        e["notes"]["exempted"] = e["notes"].get("exempted", 0) + 1
        USED.add((pat, h))
        return
    hits = ctx.scanner.hits(key)
    HITS.append({"address": pat, "class": "key", "value_sha256": h, "value": key, "hits": [x[0] for x in hits]})
    Report.fail(e, f"{pat}: {[x[0] for x in hits]} in the key {key[:160]!r}", limit=500)


def scan_files(ctx: "ST.Context", texts: Mapping[str, str], e) -> None:
    seen: set = set()
    for rel, text in sorted(texts.items()):
        for scope, unit, column, path, value in ST.file_leaves(rel, text):
            scan_leaf(ctx, scope, unit, column, path, value, e, seen)
        for scope, unit, column, path, key in ST.file_keys(rel, text):
            scan_key(ctx, scope, unit, column, path, key, e, seen)
        for n, line, off in ST.yaml_comment_lines(rel, text):
            body = line[off:]
            if ctx.scanner.any(body):
                addr = f"comments:{os.path.basename(rel)}"
                if (addr, ST.sha256_text(line)) in ctx.policy.exempt:
                    USED.add((addr, ST.sha256_text(line)))
                    continue
                HITS.append({"address": f"{addr}:{n}", "class": "comment", "value_sha256": ST.sha256_text(line), "value": line,
                             "hits": [h[0] for h in ctx.scanner.hits(body)]})
                Report.fail(e, f"{addr}:{n}: {[h[0] for h in ctx.scanner.hits(body)]} in {body[:160]!r}", limit=500)


def scan_frame(ctx: "ST.Context", sqlite_path: str, e, seen: Optional[set] = None) -> None:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    seen = set() if seen is None else seen
    try:
        for t in ST.frame_tables(conn):
            for c in ST.frame_columns(conn, t):
                for v, _n in ST.frame_text_values(conn, t, c):
                    if not ctx.scanner.any(v):
                        continue
                    for p, s in ST.frame_cell_leaves(t, c, v):
                        scan_leaf(ctx, "frame", t, c, p, s, e, seen)
                    for p, k in ST.frame_cell_keys(t, c, v):
                        scan_key(ctx, "frame", t, c, p, k, e, seen)
    finally:
        conn.close()


#: the column name a Parquet file's own key/value metadata is scanned under (the Arrow schema blob itself is not text)
PARQUET_METADATA = "__parquet_metadata__"


def scan_parquet(ctx: "ST.Context", frame_dir: str, e, seen: Optional[set] = None) -> int:
    """Every string column of every Parquet file of a frame directory, and each file's key/value metadata, as frame leaves
    (the same addresses and exemptions as the SQLite's cells); returns the number of files read."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    seen = set() if seen is None else seen
    n = 0
    for fn in sorted(os.listdir(frame_dir)):
        if not fn.endswith(".parquet"):
            continue
        t = fn[:-len(".parquet")]
        pf = pq.ParquetFile(os.path.join(frame_dir, fn))
        n += 1
        for k, v in sorted((pf.schema_arrow.metadata or {}).items()):
            if k == b"ARROW:schema":
                continue
            scan_leaf(ctx, "frame", t, PARQUET_METADATA, (k.decode("utf-8", "replace"),), v.decode("utf-8", "replace"), e, seen)
        for f in pf.schema_arrow:
            if not (pa.types.is_string(f.type) or pa.types.is_large_string(f.type)):
                continue
            for v in pc.unique(pf.read(columns=[f.name]).column(0)).to_pylist():
                if v is None or not ctx.scanner.any(v):
                    continue
                for p, sv in ST.frame_cell_leaves(t, f.name, v):
                    scan_leaf(ctx, "frame", t, f.name, p, sv, e, seen)
                for p, k in ST.frame_cell_keys(t, f.name, v):
                    scan_key(ctx, "frame", t, f.name, p, k, e, seen)
    return n


def current_frame_dir(tree: str) -> Optional[str]:
    """The frame directory ``paper/snapshot/CURRENT`` names in ``tree``, or None when the tree has none."""
    cur = os.path.join(tree, ST.SNAPSHOT_DIR, "CURRENT")
    if not os.path.exists(cur):
        return None
    with open(cur, "r", encoding="utf-8") as fh:
        d = os.path.join(tree, ST.SNAPSHOT_DIR, "snapshots", fh.read().strip())
    return d if os.path.isdir(d) else None


#: what the publication scanner reads for check 5: the shipped data this checker covers, by repository path
SCAN_TOOL_ROOTS = (ST.REGISTRY_DIR, ST.SNAPSHOT_DIR) + tuple(m for ms in ST.MIRRORS.values() for m in ms)


def scan_with_tool(tool: str, tree: str, e, allowlist: Optional[str] = None, jobs: int = 1, frame: bool = True) -> None:
    """The publication scanner's leaf scope over the shipped data, with its token table and allowlist; every hit the allowlist
    does not cover, and every error it reports, fails check 5."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("scrub_scan_for_check_5", tool)
    if spec is None or spec.loader is None:
        raise CheckFailed(f"--scan-tool {tool}: not a Python file")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    root = Path(tree)
    roots = [r for r in SCAN_TOOL_ROOTS if frame or r != ST.SNAPSHOT_DIR]
    sc, files = ss.scan(root, roots=roots, scopes=("leaf",), table=ss.load_table(root / ss.TABLE_FILE), jobs=jobs)
    allow = ss.load_allowlist(Path(allowlist) if allowlist else root / ss.ALLOWLIST)
    left = ss.apply_allowlist(sc.report.hits, allow)
    for h in left:
        Report.fail(e, f"scan tool: {h.path}#{h.where}: {h.tier} in {h.excerpt()!r}", limit=500)
    for err in sc.report.errors:
        Report.fail(e, f"scan tool error: {err}", limit=500)
    e["notes"]["scan_tool"] = {"files": len(files), "hits": len(sc.report.hits), "outside_allowlist": len(left),
                               "errors": len(sc.report.errors)}


def check_scan(ctx: "ST.Context", tree: str, frame_sqlite: Optional[str], rep: Report, scan_tool: Optional[str] = None,
               scan_allowlist: Optional[str] = None, scan_jobs: int = 1) -> None:
    e = rep.start("5_scan")
    texts = {}
    for rel in ST.registry_files(tree):
        with open(os.path.join(tree, rel), "r", encoding="utf-8") as fh:
            texts[rel] = fh.read()
    scan_files(ctx, texts, e)
    seen: set = set()
    if frame_sqlite:
        scan_frame(ctx, frame_sqlite, e, seen)
        d = current_frame_dir(tree)
        if d is None:
            Report.fail(e, "the frame's SQLite was scanned but paper/snapshot/CURRENT names no frame directory for its Parquet files")
        else:
            e["notes"]["parquet_files"] = scan_parquet(ctx, d, e, seen)
    e["notes"]["files"] = len(texts)
    e["notes"]["frame"] = bool(frame_sqlite)
    if scan_tool:
        scan_with_tool(scan_tool, tree, e, scan_allowlist, scan_jobs, frame=bool(frame_sqlite))


# --------------------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ST._add_inputs(ap)
    ap.add_argument("--pre-ref", required=True, help="the commit whose files and frame are the pre-image")
    ap.add_argument("--tree", default=REPO, help="the checkout to check (default: this script's)")
    ap.add_argument("--pre-repo", help="the git checkout to read --pre-ref from (default: --tree)")
    ap.add_argument("--pre-patch", action="append", default=[], help="a unified diff applied to the pre-image first (repeat)")
    ap.add_argument("--codemod-rules", help="the codemod's rule table, when the tree has been codemodded since the rewrite")
    ap.add_argument("--work", help="scratch directory for the reassembled frames (default: a temp dir)")
    ap.add_argument("--checks", default="1,2,3,4,5")
    ap.add_argument("--report", help="write the report here as JSON")
    ap.add_argument("--hits-out", help="write every hit of the scan here as JSONL")
    ap.add_argument("--exempt-used-out", help="write the exemption rows the scan used here (JSON list)")
    ap.add_argument("--no-frame", action="store_true", help="check 5 over the registry files only")
    ap.add_argument("--scan-tool", help="the publication scanner (reproduce/tools/scrub_scan.py): check 5 also runs its leaf scope")
    ap.add_argument("--scan-allowlist", help="the scanner's allowlist (default: the tree's reproduce/tools/scan_allowlist.tsv)")
    ap.add_argument("--scan-jobs", type=int, default=1, help="worker processes for --scan-tool")
    ap.add_argument("--exempt", action="append", default=[],
                    help="a JSON list of [address, value sha256, reason] rows the scan skips (repeat; merged with the policy's)")
    a = ap.parse_args(argv)
    want = set(a.checks.split(","))
    ctx = ST._ctx_from_args(a)
    for path in a.exempt:
        with open(path, "r", encoding="utf-8") as fh:
            for i, row in enumerate(json.load(fh)):
                if not (isinstance(row, list) and len(row) == 3 and all(isinstance(x, str) and x for x in row)):
                    raise CheckFailed(f"{path}[{i}]: an exemption is [address, value sha256, reason], each nonempty")
                ctx.policy.exempt[(row[0], row[1])] = row[2]
    if ctx.table.conflicts:
        raise CheckFailed("the prose table rewrites a value two ways: " + "; ".join(ctx.table.conflicts[:5]))
    codemod = load_codemod(a.codemod_rules, a.tree)
    rep = Report()
    work = a.work or tempfile.mkdtemp(prefix="cst-frame-")
    fs = None
    if want & {"1", "2", "3", "4"}:
        fs = check_files(ctx, a.tree, a.pre_ref, a.pre_patch, codemod, rep, a.pre_repo)
    if "2" in want:
        check_transcripts(fs, a.tree, rep)
    if "3" in want:
        check_seeds(fs, a.tree, rep)
    frame_sqlite = None
    if "4" in want:
        frame_sqlite = check_frame(ctx, a.tree, a.pre_ref, work, fs, rep, ST.inputs_sha256(ST.input_paths(a)), a.pre_repo)
    if "5" in want:
        if frame_sqlite is None and not a.no_frame:
            with open(os.path.join(a.tree, ST.SNAPSHOT_DIR, "CURRENT"), "r", encoding="utf-8") as fh:
                sid = fh.read().strip()
            d = os.path.join(a.tree, ST.SNAPSHOT_DIR, "snapshots", sid)
            with open(os.path.join(d, "MANIFEST.json"), "r", encoding="utf-8") as fh:
                m = json.load(fh)
            os.makedirs(work, exist_ok=True)
            frame_sqlite = os.path.join(work, "post_paper_frame.sqlite")
            reassemble([os.path.join(d, p) for p in m["sqlite_parts"]], frame_sqlite)
        check_scan(ctx, a.tree, frame_sqlite, rep, a.scan_tool, a.scan_allowlist, a.scan_jobs)
    if frame_sqlite and os.path.exists(frame_sqlite):
        os.remove(frame_sqlite)
    if not a.work:
        shutil.rmtree(work, ignore_errors=True)
    if a.hits_out:
        with open(a.hits_out, "w", encoding="utf-8") as fh:
            for h in HITS:
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    if a.exempt_used_out:
        with open(a.exempt_used_out, "w", encoding="utf-8") as fh:
            json.dump([[k[0], k[1], ctx.policy.exempt[k]] for k in sorted(USED)], fh, indent=1, ensure_ascii=False)
    out = {"ok": rep.ok, "pre_ref": a.pre_ref, "checks": rep.checks}
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
    for name, c in rep.checks.items():
        print(f"{name}: {'PASS' if c['ok'] else 'FAIL'} {json.dumps(c['notes'], ensure_ascii=False)}")
        for f in c["failures"][:10]:
            print(f"  - {f}")
        if c.get("n_failures", 0) > 10:
            print(f"  … {c['n_failures'] - 10} more")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
