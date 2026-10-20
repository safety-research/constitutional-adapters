#!/usr/bin/env python3
"""Re-declare every in-tree sha that pins the bytes of a config the release rewrote.

The release rewrites config text by declared rules (``scripts/release/config_text.py``) and regenerates the generated
configs. Some files pin other files by sha256: a view declares the sha of its frozen roster and pins copy, the seedrep
source manifest declares the sha of each verbatim copy, a sampled config names the sha of the config it was sampled
from, a test repeats a frozen sha, and a test pins four campaign files by git blob id. This tool moves those pins with
the bytes, the same way the scrub map's re-derivation does, so the tree stays self-consistent:

1. **Embedded sha256s.** For every file under ``configs/`` whose bytes changed since the pre-release tree, each mention
   of an earlier sha256 of it (its pre-release bytes or any committed version since; the full digest, or a prefix of at
   least eight hex digits) becomes the same-length prefix of its sha256 today. This runs over every file under ``configs/`` and the files of ``CARRY_FILES``, except where the sha
   is the value of a historical key (``HISTORICAL_KEYS``): those record what ran, and the scrub map, not the record,
   moves. Iterated to a fixed point, because a re-pinned file can itself be pinned by another.
   The explicit declarations (each view's ``frozen_inputs`` and the seedrep ``SOURCE.json``) are also re-read from the
   files they name, so a sha an earlier, uncommitted pass declared moves too.
2. **Blob pins.** ``BLOB_PINS`` names the git blob ids a test pins; each is recomputed from the file.
3. **File pins.** ``FILE_PINS`` names the dict literals that map a repository path to its sha256 (the Modal CPU image's
   dependency guard, the protected smoke surfaces); each entry is recomputed from the file, so a release-wide text
   rewrite of a pinned file (its comments, say) moves the pin in the same tree.

``--frozen`` restricts step 1 to the frozen copies (``configs/sdfbase/*/frozen/``, ``configs/seedrep/source/``) and
skips steps 2 and 3. Generators that read a frozen copy assert its declared sha before they render anything, so the
regeneration runs this first, then the generators, then the full pass.

    repin.py [--pre-ref REF] [--frozen] [--check]

``--pre-ref`` defaults to the merge base of HEAD and ``origin/main``, or to HEAD in a clone without ``origin/main``. ``--check`` changes nothing and exits 1 when a
pin is stale, naming each file.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Keys whose sha values are historical records; the same set the scrub map's re-derivation leaves alone.
HISTORICAL_KEYS: Tuple[str, ...] = (
    "serving_config_sha256", "standup_config_source_sha", "campaign_parent_object_sha256", "source_sha256",
)
#: Files outside configs/ that repeat a config's sha256 (step 1 covers them as well).
CARRY_FILES: Tuple[str, ...] = (
    "scripts/sdfbase/gsm8k/g8k_seedrep_arditi.py",
    "tests/fixtures/eight_arms/gsm8k_matrix_8f5f1495.json",
    "tests/test_sdfbase_ifeval_matrix.py",
    "tests/test_seedrep_arditi_configs.py",
)
#: Test files that pin configs by git blob id: (test file, constant name).
BLOB_PINS: Tuple[Tuple[str, str], ...] = (
    ("tests/test_sdfbase_ifeval_overlays.py", "LLAMA_CAMPAIGN_BLOBS"),
)
#: Files that pin other files by sha256 in a dict literal keyed by repository path: (pinning file, constant name).
FILE_PINS: Tuple[Tuple[str, str], ...] = (
    ("scripts/modal/cpu_suite.py", "DEPENDENCY_HASHES"),
    ("tests/test_protected_smoke_surfaces.py", "FROZEN"),
)
#: Files under configs/ that step 1 never rewrites: the scrub map records each rewritten file's earlier sha256 beside its sha
#: today on purpose (scripts/release/scrub_map.py writes and certifies it), so carrying its old shas forward would erase
#: what it records and leave `--check` stale on every tree that holds it.
NOT_CARRIED: Tuple[str, ...] = ("configs/SCRUB_MAP.tsv",)
FROZEN_PREFIXES: Tuple[str, ...] = ("configs/seedrep/source/",)
FROZEN_GLOB = re.compile(r"^configs/sdfbase/[^/]+/frozen/")
MIN_PREFIX = 8

HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{%d,64}(?![0-9A-Fa-f])" % MIN_PREFIX)
# the key a value on the same line belongs to: `key: <sha>`, `"key": "<sha>"`, `key = <sha>`, a `sha256:` prefix allowed
_KEY_BEFORE = re.compile(r"""["']?([A-Za-z_][A-Za-z0-9_.\-]*)["']?\s*[:=]\s*["']?(?:sha256:)?$""")


def git(*args: str, input_bytes: Optional[bytes] = None) -> bytes:
    return subprocess.run(["git", "-C", ROOT, *args], input=input_bytes, capture_output=True, check=True).stdout


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def is_frozen(path: str) -> bool:
    return path.startswith(FROZEN_PREFIXES) or bool(FROZEN_GLOB.match(path))


def _blob_shas(oids: Iterable[str]) -> Dict[str, str]:
    """sha256 of each git blob, read in one ``cat-file --batch``."""
    oids = sorted(set(oids))
    if not oids:
        return {}
    out = git("cat-file", "--batch", input_bytes="".join(f"{oid}\n" for oid in oids).encode())
    shas, pos = {}, 0
    for oid in oids:
        nl = out.index(b"\n", pos)
        size = int(out[pos:nl].split(b" ")[2])
        shas[oid] = sha256(out[nl + 1:nl + 1 + size])
        pos = nl + 1 + size + 1
    return shas


def prior_shas(pre_ref: str) -> Dict[str, set]:
    """Every sha256 each file under configs/ has had since the pre-release tree: its pre-release bytes and every
    committed version on the way to HEAD. A pin may name any of them (a branch re-pinned once already), and each
    becomes the file's sha today."""
    blobs: Dict[str, set] = {}
    for rec in git("ls-tree", "-r", "-z", pre_ref, "--", "configs").split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            _mode, kind, oid = meta.split(b" ")
            if kind == b"blob":
                blobs.setdefault(path.decode(), set()).add(oid.decode())
    raw = git("log", "--raw", "--no-abbrev", "--no-renames", "--format=", f"{pre_ref}..HEAD", "--", "configs").decode()
    for line in raw.splitlines():
        if not line.startswith(":"):
            continue
        meta, path = line.split("\t", 1)
        parts = meta.split()
        for oid in (parts[2], parts[3]):
            if set(oid) != {"0"}:
                blobs.setdefault(path, set()).add(oid)
    sha_of = _blob_shas(o for s in blobs.values() for o in s)
    return {path: {sha_of[o] for o in oids} for path, oids in blobs.items()}


def current_sha(path: str) -> Optional[str]:
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return None
    with open(full, "rb") as fh:
        return sha256(fh.read())


def tracked_configs() -> List[str]:
    return sorted(p for p in git("ls-files", "-z", "--", "configs").decode().split("\0") if p and p not in NOT_CARRIED)


def substitution(prior: Dict[str, set], frozen_only: bool) -> Dict[str, str]:
    """Every earlier sha256 of a file -> its sha256 today, for every changed file (or every changed frozen copy)."""
    subst: Dict[str, str] = {}
    for path, olds in prior.items():
        if frozen_only and not is_frozen(path):
            continue
        new = current_sha(path)
        if new is None:
            continue
        for old in olds:
            if old == new:
                continue
            if old in subst and subst[old] != new:
                raise SystemExit(f"repin: two files that once had the sha {old[:12]} now differ; pin by path instead")
            subst[old] = new
    return subst


def declared_pairs() -> Dict[str, str]:
    """The explicit declarations, re-read from the files they name: each view's ``frozen_inputs`` roster / pins sha256
    and each ``SOURCE.json`` file sha256 that no longer matches its file, as declared sha -> sha today. These hold
    whatever an earlier pass declared, committed or not, so they are re-pinned from the bytes rather than from history."""
    import glob
    import json
    pairs: Dict[str, str] = {}
    for views in sorted(glob.glob(os.path.join(ROOT, "configs/sdfbase/*/views*.yaml"))):
        text = open(views, encoding="utf-8").read()
        m = re.search(r"^frozen_inputs:\n((?:[ \t]+.*\n)+)", text, re.M)
        if not m:
            continue
        block = dict(re.findall(r"^[ \t]+(\w+):[ \t]*(\S+)", m.group(1), re.M))
        for kind in ("roster", "pins"):
            path, declared = block.get(kind), block.get(f"{kind}_sha256")
            if path and declared:
                now = current_sha(path)
                if now is not None and now != declared:
                    pairs[declared] = now
    source = os.path.join(ROOT, "configs/seedrep/source/SOURCE.json")
    if os.path.isfile(source):
        for rel, entry in json.load(open(source, encoding="utf-8")).get("files", {}).items():
            now = current_sha(f"configs/seedrep/source/{rel}")
            if now is not None and entry.get("sha256") and now != entry["sha256"]:
                pairs[entry["sha256"]] = now
    return pairs


def prefix_table(subst: Dict[str, str]) -> Dict[int, Dict[str, str]]:
    """Unambiguous prefixes (at least MIN_PREFIX long) of every pre-release sha, mapped to the new sha."""
    table: Dict[int, Dict[str, str]] = {}
    for n in range(MIN_PREFIX, 64):
        seen: Dict[str, set] = {}
        for old, new in subst.items():
            seen.setdefault(old[:n], set()).add(new)
        table[n] = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
    return table


def carry_text(text: str, subst: Dict[str, str], prefixes: Dict[int, Dict[str, str]]) -> str:
    def one(m: "re.Match[str]") -> str:
        token = m.group(0)
        new = subst.get(token) if len(token) == 64 else prefixes.get(len(token), {}).get(token)
        if new is None:
            return token
        line_start = text.rfind("\n", 0, m.start()) + 1
        km = _KEY_BEFORE.search(text, line_start, m.start())
        if km and km.end() == m.start() and km.group(1) in HISTORICAL_KEYS:
            return token
        return new if len(token) == 64 else new[:len(token)]
    return HEX_TOKEN.sub(one, text)


def carry_pass(prior: Dict[str, set], frozen_only: bool, write: bool) -> List[str]:
    """One substitution pass over configs/ and CARRY_FILES; the paths it changed (or would change)."""
    subst = substitution(prior, frozen_only)
    for old, new in declared_pairs().items():
        if old in subst and subst[old] != new:
            raise SystemExit(f"repin: the declared sha {old[:12]} maps to two files today; pin by path instead")
        subst[old] = new
    if not subst:
        return []
    prefixes = prefix_table(subst)
    changed = []
    for path in tracked_configs() + list(CARRY_FILES):
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            continue
        with open(full, "rb") as fh:
            data = fh.read()
        if b"\0" in data[:8192]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = carry_text(text, subst, prefixes)
        if new != text:
            changed.append(path)
            if write:
                with open(full, "wb") as fh:
                    fh.write(new.encode("utf-8"))
    return changed


def blob_pass(write: bool) -> List[str]:
    """Recompute each BLOB_PINS constant: every ``"<path>": "<40 hex>"`` entry inside the named dict literal."""
    changed = []
    for test, name in BLOB_PINS:
        full = os.path.join(ROOT, test)
        with open(full, encoding="utf-8") as fh:
            text = fh.read()
        m = re.search(r"^%s = \{\n(.*?)^\}" % re.escape(name), text, re.S | re.M)
        if not m:
            raise SystemExit(f"repin: {test} has no dict literal {name}")
        body = m.group(1)

        def one(e: "re.Match[str]") -> str:
            with open(os.path.join(ROOT, e.group(2)), "rb") as fh:
                return f'{e.group(1)}{blob_id(fh.read())}{e.group(3)}'
        new_body = re.sub(r'^(\s*"([^"]+)": ")[0-9a-f]{40}(",?)$', one, body, flags=re.M)
        if new_body != body:
            changed.append(test)
            if write:
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(text[:m.start(1)] + new_body + text[m.end(1):])
    return changed


def file_pass(write: bool) -> List[str]:
    """Recompute each FILE_PINS constant: every ``"<path>": "<64 hex>"`` entry inside the named dict literal, the value
    on the key's line or on a later one (comment lines between them are kept)."""
    changed = []
    for pinner, name in FILE_PINS:
        full = os.path.join(ROOT, pinner)
        with open(full, encoding="utf-8") as fh:
            text = fh.read()
        m = re.search(r"^%s = \{\n(.*?)^\}" % re.escape(name), text, re.S | re.M)
        if not m:
            raise SystemExit(f"repin: {pinner} has no dict literal {name}")
        body = m.group(1)

        def one(e: "re.Match[str]") -> str:
            now = current_sha(e.group(2))
            if now is None:
                raise SystemExit(f"repin: {pinner} {name} pins {e.group(2)}, which is not in the tree")
            return f"{e.group(1)}{now}{e.group(4)}"
        new_body = re.sub(r'("([^"\n]+)":(\s*(?:#[^\n]*\n\s*)*)")[0-9a-f]{64}(")', one, body)
        if new_body != body:
            changed.append(pinner)
            if write:
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(text[:m.start(1)] + new_body + text[m.end(1):])
    return changed


def default_pre_ref() -> str:
    """The merge base of HEAD and origin/main; in a clone without origin/main (a fresh copy of the published tree), HEAD
    itself, so the pins follow any edit made since the checked-out commit."""
    try:
        return git("merge-base", "HEAD", "origin/main").decode().strip()
    except subprocess.CalledProcessError:
        return git("rev-parse", "HEAD").decode().strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pre-ref", help="the pre-release commit (default: merge base of HEAD and origin/main, else HEAD)")
    ap.add_argument("--frozen", action="store_true", help="re-pin only the frozen copies' shas (before generators run)")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if any pin is stale")
    a = ap.parse_args(argv)
    pre_ref = a.pre_ref or default_pre_ref()
    prior = prior_shas(pre_ref)
    stale: List[str] = []
    for _round in range(8):
        changed = carry_pass(prior, a.frozen, write=not a.check)
        stale += changed
        if not changed or a.check:
            break
    else:
        print("repin: the sha substitution did not reach a fixed point in 8 passes", file=sys.stderr)
        return 1
    if not a.frozen:
        stale += blob_pass(write=not a.check)
        stale += file_pass(write=not a.check)
    stale = sorted(set(stale))
    verb = "stale" if a.check else "re-pinned"
    for p in stale:
        print(f"repin: {verb}: {p}")
    print(f"repin: {len(stale)} file(s) {verb} against {pre_ref[:12]}{' (frozen copies only)' if a.frozen else ''}")
    return 1 if (a.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
