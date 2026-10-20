"""The paper's refusal-direction fit prompts, re-drawn byte for byte from the public upstream splits.

The refusal-direction objects (``objects/qwen35_4b_it/``, paper labels ``arditi_allpos`` and ``arditi``) were fit
on one document, ``data/arditi/fit_prompts.json``: 128 training and 32 validation instructions per side, drawn
from the committed dataset splits of andyrdt/refusal_direction (Arditi et al. 2024, arXiv 2406.11717) by that
repository's own sampler. The document's text is not distributed with this repository: its harmful side includes
MaliciousInstruct, which states no licence, and its harmless side is Stanford Alpaca (CC BY-NC 4.0).
``data/arditi/fit_prompts.manifest.json`` records the document's identity instead (the upstream commit, the four
split files with their sha256, the seed, the draw order and sizes, each set's sha256, every field of the document
except the instructions, and the document's own sha256), and this module re-creates the document from it:

1. Fetch the four split files at the pinned commit into the third-party cache (the GitHub raw URL first,
   Software Heritage by git blob id second) and check each file's sha256 before a byte of it is read.
2. Read each file as upstream's ``load_dataset_split(..., instructions_only=True)`` does: the ``instruction`` of
   every row, in file order.
3. Replay upstream's ``pipeline/run_pipeline.py::load_and_sample_datasets``: ONE ``random.seed(42)``, then
   ``random.sample`` of 128, 128, 32 and 32 instructions from harmful_train, harmless_train, harmful_val and
   harmless_val, in that order. The order is part of the identity: drawing any set alone gives other rows.
4. Check each set's sha256 (of ``json.dumps(instructions, ensure_ascii=False)``), assemble the document from the
   manifest's recorded fields and the four sets, serialise it as
   ``json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\\n"`` and check its sha256 before it
   is written.

The upstream split files are the starting point, never regenerated: upstream builds its harmful splits with an
unseeded ``random.sample``, so its committed files are the only faithful record of them.

The cache is the third-party root ``sisteer.eval.attacks_godmode`` uses: ``$SISTEER_THIRD_PARTY_DIR``, else
``$XDG_CACHE_HOME/sisteer/third_party``, else ``~/.cache/sisteer/third_party``. The split files land at
``refusal_direction/<commit>/dataset/splits/`` and the document at ``refusal_direction/<commit>/fit_prompts.json``.
``SISTEER_THIRD_PARTY_OFFLINE=1`` (or ``--offline``) forbids the network: the document is then read from the cache,
or built from split files already cached there, and is otherwise refused. A cluster's compute nodes have no egress;
``reproduce/00_setup.sh --what third_party`` builds the document on the setup host and installs the verified copy
under each cluster's shared root, where ``reproduce/lib/hf_env.sh`` points jobs at it with fetching off. Every
failure (a hash mismatch, every source failing, an absent copy offline, a manifest that contradicts itself)
raises ``FitPromptsError`` naming what failed; nothing is ever degraded to an unverified or different draw.
Importing this module reads nothing and touches no network.

    python -m sisteer.data.arditi_fit_prompts fetch [--cache-dir D] [--offline] [--json]
    python -m sisteer.data.arditi_fit_prompts reproduce --out FILE [--cache-dir D] [--offline]
    python -m sisteer.data.arditi_fit_prompts verify FILE|-

The refusal-direction fitter (``scripts/arditi/fit_refusal_direction.py``) reads the sets as the builtin prompt sets
``arditi:harmful_train``, ``arditi:harmless_train``, ``arditi:harmful_val`` and ``arditi:harmless_val``; stage 04
fits on the two training sets by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from sisteer.eval.attacks_godmode import OFFLINE_ENV, THIRD_PARTY_DIR_ENV, offline_mode, third_party_root

#: src/sisteer/data/arditi_fit_prompts.py -> parents[3] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_RELPATH = "data/arditi/fit_prompts.manifest.json"
#: the manifest ``load_manifest()`` reads when given none (the tests point it at a stand-in's)
MANIFEST_PATH = REPO_ROOT / MANIFEST_RELPATH
MANIFEST_SCHEMA = "arditi_fit_prompts_manifest/1"
DOCUMENT_SCHEMA = "arditi_fit_prompts/1"
#: the builtin prompt-set ids of the refusal-direction fitter: ``arditi:<set name>``
PROMPT_SET_PREFIX = "arditi:"
SET_NAMES = ("harmful_train", "harmful_val", "harmless_train", "harmless_val")
CACHE_DIRNAME = "refusal_direction"
DOCUMENT_NAME = "fit_prompts.json"
ALLOWED_HOSTS = frozenset({"raw.githubusercontent.com", "archive.softwareheritage.org"})
FETCH_COMMAND = "python -m sisteer.data.arditi_fit_prompts fetch"
_TIMEOUT_S = 60
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class FitPromptsError(RuntimeError):
    """The fit prompts could not be made available exactly as the manifest records them (refused by name)."""


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """The git blob id of ``data`` (what Software Heritage addresses content by)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def set_sha256(instructions: list[str]) -> str:
    """A set's identity: sha256 of the UTF-8 of ``json.dumps(instructions, ensure_ascii=False)`` (the document's own
    ``prompts_sha256`` definition)."""
    return _sha256(json.dumps(instructions, ensure_ascii=False))


def serialise(document: dict) -> bytes:
    """The document's bytes: JSON, indent 1, sorted keys, non-ASCII kept, newline-terminated, UTF-8."""
    return (json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


# ------------------------------------------------------------------------------------------------------- the manifest

def _need(cond: bool, where: str, what: str) -> None:
    if not cond:
        raise FitPromptsError(f"{where}: {what}")


def validate_manifest(m: dict, where: str = MANIFEST_RELPATH) -> dict:
    """``m`` when it is a complete, self-consistent manifest; else ``FitPromptsError`` naming the first defect. The
    manifest states several facts twice (the document's recorded fields and the readable sections beside them), and a
    disagreement between the two refuses rather than picking one."""
    _need(isinstance(m, dict) and m.get("schema") == MANIFEST_SCHEMA, where, f"schema is not {MANIFEST_SCHEMA}")
    for key in ("document", "upstream", "sources", "draw", "sets"):
        _need(isinstance(m.get(key), dict), where, f"no {key!r} object")
    doc, up, sources, draw, sets = m["document"], m["upstream"], m["sources"], m["draw"], m["sets"]
    _need(isinstance(doc.get("sha256"), str) and bool(_HEX64.fullmatch(doc["sha256"])), where, "document.sha256 is not 64 hex")
    _need(isinstance(doc.get("bytes"), int) and doc["bytes"] > 0, where, "document.bytes is not a positive integer")
    fields = doc.get("fields")
    _need(isinstance(fields, dict), where, "no document.fields object")
    _need("prompts" not in fields, where, "document.fields carries 'prompts': the manifest records identity, never text")
    _need(fields.get("schema") == DOCUMENT_SCHEMA, where, f"document.fields.schema is not {DOCUMENT_SCHEMA}")
    _need(isinstance(up.get("repo"), str) and up["repo"].count("/") == 1, where, "upstream.repo is not owner/name")
    _need(isinstance(up.get("commit"), str) and bool(_HEX40.fullmatch(up["commit"])), where, "upstream.commit is not 40 hex")
    _need(fields.get("upstream") == f"https://github.com/{up['repo']}", where, "document.fields.upstream disagrees with upstream.repo")
    _need(fields.get("upstream_commit") == up["commit"], where, "document.fields.upstream_commit disagrees with upstream.commit")
    _need(sorted(sources) == sorted(SET_NAMES) and sorted(sets) == sorted(SET_NAMES), where,
          f"sources and sets must each name exactly {list(SET_NAMES)}")
    order = draw.get("order")
    _need(isinstance(order, list) and sorted(order) == sorted(SET_NAMES), where, "draw.order is not a permutation of the four sets")
    _need(isinstance(draw.get("seed"), int), where, "draw.seed is not an integer")
    n = draw.get("n")
    _need(isinstance(n, dict) and sorted(n) == sorted(SET_NAMES) and all(isinstance(v, int) and v > 0 for v in n.values()),
          where, "draw.n is not a positive size per set")
    sampling = fields.get("sampling") or {}
    _need(sampling.get("seed") == draw["seed"] and sampling.get("draw_order") == order, where,
          "document.fields.sampling disagrees with draw (seed or order)")
    for name in SET_NAMES:
        want_n = sampling.get("n_val") if name.endswith("_val") else sampling.get("n_train")
        _need(n[name] == want_n, where, f"draw.n.{name} = {n[name]} disagrees with document.fields.sampling ({want_n})")
        s = sets[name]
        _need(isinstance(s, dict) and s.get("n") == n[name], where, f"sets.{name}.n disagrees with draw.n")
        _need(isinstance(s.get("sha256"), str) and bool(_HEX64.fullmatch(s["sha256"])), where, f"sets.{name}.sha256 is not 64 hex")
        _need((fields.get("prompts_sha256") or {}).get(name) == s["sha256"], where,
              f"sets.{name}.sha256 disagrees with document.fields.prompts_sha256")
        src = sources[name]
        _need(isinstance(src, dict), where, f"sources.{name} is not an object")
        path = src.get("path")
        _need(isinstance(path, str) and path == f"dataset/splits/{name}.json", where, f"sources.{name}.path is not dataset/splits/{name}.json")
        _need(isinstance(src.get("sha256"), str) and bool(_HEX64.fullmatch(src["sha256"])), where, f"sources.{name}.sha256 is not 64 hex")
        _need(isinstance(src.get("git_blob"), str) and bool(_HEX40.fullmatch(src["git_blob"])), where, f"sources.{name}.git_blob is not 40 hex")
        _need(isinstance(src.get("bytes"), int) and src["bytes"] > 0, where, f"sources.{name}.bytes is not a positive integer")
        _need(isinstance(src.get("rows"), int) and src["rows"] >= n[name], where, f"sources.{name}.rows is below the draw size")
        rel = path.removeprefix("dataset/")
        _need((fields.get("source_sha256") or {}).get(rel) == src["sha256"], where,
              f"sources.{name}.sha256 disagrees with document.fields.source_sha256[{rel!r}]")
        _need((fields.get("source_rows") or {}).get(name) == src["rows"], where,
              f"sources.{name}.rows disagrees with document.fields.source_rows")
    return m


def load_manifest(manifest: dict | str | os.PathLike | None = None) -> dict:
    """The validated manifest: a dict as given, else the JSON at a path, else ``MANIFEST_PATH`` (the repository's)."""
    if isinstance(manifest, dict):
        return validate_manifest(manifest, "the given manifest")
    path = Path(manifest) if manifest else MANIFEST_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FitPromptsError(f"cannot read the fit-prompts manifest {path}: {exc}") from None
    return validate_manifest(data, str(path))


# ---------------------------------------------------------------------------------------------------------- the cache

def cache_dir_for(manifest: dict, cache_dir: str | os.PathLike | None = None) -> Path:
    """``<third-party root>/refusal_direction/<commit>``."""
    return third_party_root(cache_dir) / CACHE_DIRNAME / manifest["upstream"]["commit"]


def document_relpath(manifest: dict) -> PurePosixPath:
    """The document's path under a third-party root (the same on every host and cluster)."""
    return PurePosixPath(CACHE_DIRNAME) / manifest["upstream"]["commit"] / DOCUMENT_NAME


def document_path(manifest: dict, cache_dir: str | os.PathLike | None = None) -> Path:
    return third_party_root(cache_dir).joinpath(*document_relpath(manifest).parts)


def source_path(manifest: dict, name: str, cache_dir: str | os.PathLike | None = None) -> Path:
    return cache_dir_for(manifest, cache_dir).joinpath(*PurePosixPath(manifest["sources"][name]["path"]).parts)


def source_urls(manifest: dict, name: str) -> tuple[str, ...]:
    """Tried in order; the sha256 check makes the source irrelevant to what is used."""
    up, src = manifest["upstream"], manifest["sources"][name]
    return (f"https://raw.githubusercontent.com/{up['repo']}/{up['commit']}/{src['path']}",
            f"https://archive.softwareheritage.org/api/1/content/sha1_git:{src['git_blob']}/raw/")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _require_allowed(url: str, *, why: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise FitPromptsError(f"{why}: {url!r} is not an https URL on {sorted(ALLOWED_HOSTS)}")


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an allowed https host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_allowed(newurl, why=f"redirect {code} from {req.full_url}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(url: str):
    """The one place a network connection is opened (the tests replace it)."""
    req = urllib.request.Request(url, headers={"User-Agent": "sisteer-third-party-fetch"})
    return urllib.request.build_opener(_RedirectGuard()).open(req, timeout=_TIMEOUT_S)


def _download(url: str, limit: int) -> bytes:
    _require_allowed(url, why="source")
    with _open_url(url) as resp:
        _require_allowed(resp.geturl(), why="final URL")
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise FitPromptsError(f"{url} returned more than {limit} bytes, the pinned file's size")
    return data


def _check_source(manifest: dict, name: str, data: bytes, where: str) -> None:
    src = manifest["sources"][name]
    got = _sha256(data)
    if got != src["sha256"]:
        up = manifest["upstream"]
        raise FitPromptsError(
            f"{src['path']} from {where} is not the pinned file: sha256 {got} ({len(data)} bytes), expected {src['sha256']} "
            f"({src['bytes']} bytes) from {up['repo']}@{up['commit']}")


def ensure_source(manifest: dict, name: str, cache_dir: str | os.PathLike | None = None, *,
                  offline: bool | None = None) -> tuple[str, Path, bytes, str]:
    """One upstream split file present in the cache and verified: ``(status, path, bytes, source)``, status ``PRESENT``
    (the cached copy verified) or ``FETCHED`` (downloaded, verified, then renamed into place). A cached copy that is
    not the pinned file is refused and left for the operator; a download is checked before the atomic rename."""
    path = source_path(manifest, name, cache_dir)
    if path.is_file():
        data = path.read_bytes()
        try:
            _check_source(manifest, name, data, str(path))
        except FitPromptsError as exc:
            raise FitPromptsError(f"{exc}. Remove it and run `{FETCH_COMMAND}` again.") from None
        return "PRESENT", path, data, "cache"
    if offline_mode(offline):
        raise FitPromptsError(f"{manifest['sources'][name]['path']} is not in the third-party cache ({path}) and fetching is off")
    failures = []
    for url in source_urls(manifest, name):
        try:
            data = _download(url, manifest["sources"][name]["bytes"])
            _check_source(manifest, name, data, url)
        except (FitPromptsError, urllib.error.URLError, OSError, ValueError) as exc:
            failures.append(f"{url}: {exc}")
            continue
        _write_atomic(path, data)
        return "FETCHED", path, data, url
    raise FitPromptsError(
        f"{manifest['sources'][name]['path']} could not be fetched from any source (nothing was written to {path}):\n  "
        + "\n  ".join(failures))


# --------------------------------------------------------------------------------------------------------- the draw

def instructions_of(manifest: dict, name: str, data: bytes) -> list[str]:
    """Upstream's ``load_dataset_split(..., instructions_only=True)``: the ``instruction`` of every row, in file order."""
    try:
        rows = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FitPromptsError(f"{manifest['sources'][name]['path']}: not a UTF-8 JSON document ({exc})") from None
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("instruction"), str) for r in rows):
        raise FitPromptsError(f"{manifest['sources'][name]['path']}: not a list of rows with a string 'instruction'")
    if len(rows) != manifest["sources"][name]["rows"]:
        raise FitPromptsError(f"{manifest['sources'][name]['path']}: {len(rows)} rows, expected {manifest['sources'][name]['rows']}")
    return [r["instruction"] for r in rows]


def draw(manifest: dict, instructions: dict[str, list[str]]) -> dict[str, list[str]]:
    """``load_and_sample_datasets`` replayed: one seeding, then one ``sample`` per set in the recorded order. A private
    ``random.Random(seed)`` draws exactly what ``random.seed(seed)`` followed by module-level ``random.sample`` does,
    without touching the interpreter's global generator."""
    rng = random.Random(manifest["draw"]["seed"])
    return {name: rng.sample(instructions[name], manifest["draw"]["n"][name]) for name in manifest["draw"]["order"]}


def build_document(manifest: dict, drawn: dict[str, list[str]]) -> bytes:
    """The document's bytes from the manifest's recorded fields and the drawn sets, every set's sha256 checked first and
    the whole document's after."""
    for name in SET_NAMES:
        got = set_sha256(drawn[name])
        if got != manifest["sets"][name]["sha256"]:
            raise FitPromptsError(
                f"the re-drawn {name} has sha256 {got}, expected {manifest['sets'][name]['sha256']}: the draw did not replay "
                f"(the split files matched their sha256, so the sampler differs: seed {manifest['draw']['seed']}, order "
                f"{manifest['draw']['order']}, Python {sys.version.split()[0]})")
    document = dict(manifest["document"]["fields"])
    document["prompts"] = {name: drawn[name] for name in SET_NAMES}
    data = serialise(document)
    check_document(manifest, data, "the re-drawn document")
    return data


def check_document(manifest: dict, data: bytes, where: str) -> None:
    got = _sha256(data)
    if got != manifest["document"]["sha256"]:
        raise FitPromptsError(f"{where} has sha256 {got} ({len(data)} bytes), expected {manifest['document']['sha256']} "
                              f"({manifest['document']['bytes']} bytes)")


def sets_of(manifest: dict, data: bytes, where: str) -> dict[str, list[str]]:
    """The four sets of a document whose bytes already matched the manifest, each re-checked against its own sha256."""
    try:
        prompts = json.loads(data.decode("utf-8"))["prompts"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise FitPromptsError(f"{where}: not a fit-prompts document ({exc})") from None
    for name in SET_NAMES:
        if set_sha256(prompts[name]) != manifest["sets"][name]["sha256"]:
            raise FitPromptsError(f"{where}: set {name} does not match its sha256")
    return {name: list(prompts[name]) for name in SET_NAMES}


def ensure_document(cache_dir: str | os.PathLike | None = None, *, offline: bool | None = None,
                    manifest: dict | str | os.PathLike | None = None) -> dict:
    """The document present in the cache and verified. Returns a report: ``status`` ``PRESENT`` (the cached copy
    verified) or ``BUILT`` (re-drawn from verified split files, then renamed into place), ``path``, ``relpath``,
    ``sha256``, ``bytes``, the ``sources`` read and the ``sets`` (name, size and sha256; never text). A cached document
    that is not the recorded one is refused and left for the operator."""
    m = load_manifest(manifest)
    path = document_path(m, cache_dir)
    sources: dict[str, dict] = {}
    if path.is_file():
        data = path.read_bytes()
        try:
            check_document(m, data, str(path))
        except FitPromptsError as exc:
            raise FitPromptsError(f"{exc}. Remove it and run `{FETCH_COMMAND}` again.") from None
        sets_of(m, data, str(path))
        status = "PRESENT"
    else:
        instructions = {}
        try:
            for name in SET_NAMES:
                st, spath, sdata, origin = ensure_source(m, name, cache_dir, offline=offline)
                sources[name] = {"status": st, "path": str(spath), "source": origin, "sha256": _sha256(sdata)}
                instructions[name] = instructions_of(m, name, sdata)
        except FitPromptsError as exc:
            hint = (f"Build it on a host with network access (`{FETCH_COMMAND}`, or `reproduce/00_setup.sh --what third_party` for "
                    f"every cluster), or point {THIRD_PARTY_DIR_ENV} at a root that holds {document_relpath(m)}."
                    if offline_mode(offline) else
                    f"Place the split file whose sha256 the manifest records at that path, or run `{FETCH_COMMAND}` where the network allows it.")
            raise FitPromptsError(f"the refusal-direction fit prompts ({path}) are not in the cache and cannot be built: {exc}\n{hint}") from None
        data = build_document(m, draw(m, instructions))
        _write_atomic(path, data)
        status = "BUILT"
    up = m["upstream"]
    return {"status": status, "path": str(path), "relpath": str(document_relpath(m)), "sha256": _sha256(data), "bytes": len(data),
            "upstream": {"repo": up["repo"], "commit": up["commit"]}, "sources": sources,
            "sets": {name: {"n": m["sets"][name]["n"], "sha256": m["sets"][name]["sha256"]} for name in SET_NAMES}}


def load_sets(cache_dir: str | os.PathLike | None = None, *, offline: bool | None = None,
              manifest: dict | str | os.PathLike | None = None) -> tuple[dict[str, list[str]], dict]:
    """``(sets, report)``: the four verified sets from the cached document (built on a miss unless offline) and the
    ``ensure_document`` report naming where they came from."""
    m = load_manifest(manifest)
    report = ensure_document(cache_dir, offline=offline, manifest=m)
    data = Path(report["path"]).read_bytes()
    check_document(m, data, report["path"])
    return sets_of(m, data, report["path"]), report


# ---------------------------------------------------------------------------------------------------------- the CLI

def _verify_file(m: dict, data: bytes, where: str) -> dict:
    """What a given copy matches, by hash only: the whole document, and each set when the copy parses."""
    out = {"path": where, "sha256": _sha256(data), "bytes": len(data), "document_matches": _sha256(data) == m["document"]["sha256"],
           "sets": {}}
    try:
        prompts = json.loads(data.decode("utf-8")).get("prompts") or {}
    except (UnicodeDecodeError, ValueError, AttributeError):
        prompts = {}
    for name in SET_NAMES:
        rows = prompts.get(name) if isinstance(prompts, dict) else None
        got = set_sha256(rows) if isinstance(rows, list) else None
        out["sets"][name] = {"n": len(rows) if isinstance(rows, list) else None, "sha256": got,
                             "matches": got == m["sets"][name]["sha256"]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sisteer.data.arditi_fit_prompts",
        description="Re-create the paper's refusal-direction fit prompts byte for byte from the pinned upstream splits "
                    "(andyrdt/refusal_direction; the text is not distributed with this repository) and check every hash "
                    f"{MANIFEST_RELPATH} records.")
    ap.add_argument("--manifest", default=None, help=f"the manifest (default: the repository's {MANIFEST_RELPATH})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (("fetch", "make the document present and verified in the third-party cache"),
                          ("reproduce", "write the verified document to --out")):
        p = sub.add_parser(cmd, help=helptext)
        p.add_argument("--cache-dir", default=None,
                       help=f"third-party root (default ${THIRD_PARTY_DIR_ENV}, else $XDG_CACHE_HOME/sisteer/third_party)")
        p.add_argument("--offline", "--check", dest="offline", action="store_true", default=None,
                       help=f"never reach the network (as {OFFLINE_ENV}=1): read the cached document, or build it from cached split files")
        p.add_argument("--json", action="store_true", help="print the report as one JSON object")
        if cmd == "reproduce":
            p.add_argument("--out", required=True, help="where to write the document (e.g. data/arditi/fit_prompts.json, which git ignores)")
    v = sub.add_parser("verify", help="check an existing copy against the manifest, by hash only")
    v.add_argument("file", help="the copy to check, or - for standard input")
    v.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        m = load_manifest(args.manifest)
        if args.cmd == "verify":
            data = sys.stdin.buffer.read() if args.file == "-" else Path(args.file).read_bytes()
            rep = _verify_file(m, data, args.file)
            ok = rep["document_matches"] and all(s["matches"] for s in rep["sets"].values())
            if args.json:
                print(json.dumps(rep, sort_keys=True))
            else:
                print(f"{'MATCHES' if ok else 'DIFFERS'} {args.file}: sha256 {rep['sha256']} ({rep['bytes']} bytes), "
                      f"recorded {m['document']['sha256']} ({m['document']['bytes']} bytes)")
                for name, s in rep["sets"].items():
                    print(f"  {name:<15} {'ok  ' if s['matches'] else 'DIFF'} n={s['n']} sha256 {s['sha256']}")
            return 0 if ok else 1
        rep = ensure_document(args.cache_dir, offline=args.offline, manifest=m)
        if args.cmd == "reproduce":
            out = Path(args.out)
            _write_atomic(out, Path(rep["path"]).read_bytes())
            check_document(m, out.read_bytes(), str(out))
            rep["out"] = str(out)
    except (FitPromptsError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rep, sort_keys=True))
        return 0
    up = rep["upstream"]
    print(f"{rep['status']} refusal-direction fit prompts from {up['repo']}@{up['commit'][:12]}")
    print(f"  path    {rep['path']}")
    if rep.get("out"):
        print(f"  out     {rep['out']}")
    print(f"  sha256  {rep['sha256']} ({rep['bytes']} bytes), as recorded")
    for name, s in rep["sources"].items():
        print(f"  source  {name:<15} {s['status']:<8} sha256 {s['sha256']}")
    for name, s in rep["sets"].items():
        print(f"  set     {name:<15} n={s['n']:<4} sha256 {s['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
