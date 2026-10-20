"""The client lock compared by WHAT THE MATH-500 CLIENT IMPORTS — not by its digest.

The tinycap client venv is built ONLY from the compiled lock (``contracts.CLIENT_LOCK_TXT``: ``uv pip install --no-deps
--require-hashes``, the VCS line at its exact commit), and every tinycap store records the lock's sha256
(``manifest.frameworks.client_lock_sha256``). The eight-arm MATH-500 compatibility preflight compared that DIGEST across
the published tree and the wave tree — so when PR #80 ADDED three distributions to the lock (the IFEval checker's
``immutabledict`` + ``langdetect`` and the ``instruction-following-eval`` VCS pin) without moving any existing version,
the digest moved and the preflight refused every MATH-500 extension row on ``shared.client_lock_sha256``, though nothing
the MATH-500 client runs had changed. A digest cannot tell "a package the client imports changed version" (a real fork
of the instrument) from "a package the client never imports joined the lock" (no fork).

This module makes the distinction from the code, never from a hand-typed list:

1. :func:`walk_client_imports` — an AST walk over every repo module reachable from the client venv's MATH-500 ENTRY
   POINTS (:data:`CLIENT_ENTRY_POINTS`: the driver, the usage ledger, the resolved-sampler recorder, the row-identity
   stamp — each named by the launcher's own text, pinned by test), following imports that resolve to repo files under
   the client's import roots (:data:`CLIENT_IMPORT_ROOTS`, the PYTHONPATH the launcher exports plus each script's own
   directory) and collecting every third-party top-level import name with the modules that import it. Every import
   statement counts, executed on the MATH-500 path or not (a guarded ``try: import`` too): an OVER-approximation that can
   only widen the compared set, never narrow it.
2. :func:`parse_lock` — the lock's own resolution: ``name==version`` / ``name @ url@rev`` blocks with their ``# via``
   edges.
3. :func:`resolve_import` — import name -> lock distribution(s): PEP 503 normalisation first, then the small alias
   table :data:`IMPORT_NAME_ALIASES` (``yaml`` -> ``pyyaml``), then the running interpreter's
   ``importlib.metadata.packages_distributions()`` (every candidate present in a lock is taken — conservative); the
   resolver that applied is RECORDED. A name that resolves to nothing in either lock is DISCLOSED (``imports_not_in_lock``):
   the client venv is built only from the lock, so such an import cannot be satisfied there (a serving-venv or
   attack-tooling module reached by the walk), and it contributes no version to compare.
4. :func:`compare_locks` — versions (or VCS url@rev) compared for exactly the IMPORTED distributions present in BOTH
   locks; a difference is a PROBLEM by name (``client_lock.<dist>: published X != wave Y``). An imported distribution
   present in ONE lock only is disclosed (``imported_added`` / ``imported_removed``, importers named): the published
   client never had it, so there is no version to compare — a repo-code change that starts USING it on the MATH-500
   path is the tree-set review's business (plan P5), not the lock's. Everything the locks disagree on OUTSIDE the
   imported set is disclosed too (``outside_imported_set``), and the two digests ride the record (``sha256.equal``)
   without refusing by themselves. ``one_instrument`` is True iff no imported distribution changed version.
5. :func:`lock_text_at_tree` / :func:`published_lock_record` — the published lock TEXT comes from the immutable tree
   (``git show <producer tree>:<lock>``) and its digest must EQUAL the digest the leg RECORDED: the chain of custody. A
   mismatch is a problem (the leg did not run under its tree's lock); an unreadable tree is UNDETERMINED (never a pass).

Both the preflight (``scripts/sdfbase/math500/m5t_compat_preflight.py``) and the analysis
(``scripts/tinycap_math500_thinking.py``, ``compatibility_record``'s ``client_lock`` disclosure) call these, so the launch
gate and the admission read the same rule.
"""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contracts as C

LOCK_REL = C.CLIENT_LOCK_TXT

#: the client venv's MATH-500 entry points — each is what the LAUNCHER runs from the client venv (``$CPY``), named by its
#: own text (``tests/test_m5t_compat_preflight.py`` pins each against the launcher file that names it):
#:   the driver           — scripts/sdfbase/full22/served_thinking.sh  st_driver_argv_head  (``$CPY scripts/served_capability_run.py generate``)
#:   the usage ledger     — served_thinking.sh  st_ledger_up  (``$CPY -m sisteer.capeval.serving.usage_ledger``)
#:   the sampler recorder — scripts/sdfbase/math500/cells.py  RESOLVED_SAMPLER_SCRIPT_REL  (m5t_common.sh m5t_recorder_argv)
#:   the identity stamp   — scripts/sdfbase/math500/m5t_common.sh  m5t_row_identity  (``cells.py row --identity``, the client venv)
CLIENT_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/served_capability_run.py",
    "src/sisteer/capeval/serving/usage_ledger.py",
    "scripts/sdfbase/record_resolved_sampler.py",
    "scripts/sdfbase/math500/cells.py",
)
#: where a client import resolves to a REPO module: the PYTHONPATH the launcher exports (``$REPO_DIR/src:$REPO_DIR/scripts``,
#: common.sh / m5t_common.sh) plus each entry script's own directory (on sys.path when run as a script) — and the ONE directory an
#: entry script inserts itself: cells.py (and g8k_cells.py) put scripts/sdfbase/full22 on sys.path for the it-doc admission module
#: `itdoc_bind` (math-eight-arms-extension E3, 2026-09-13); a first-party root of the walk, never a lock distribution
CLIENT_IMPORT_ROOTS: tuple[str, ...] = ("src", "scripts", "scripts/sdfbase", "scripts/sdfbase/math500", "scripts/sdfbase/full22")
#: import name -> distribution for names PEP 503 normalisation cannot map (checked against the lock like every other
#: resolution; an alias that matches no lock entry resolves nothing — the table invents no package)
IMPORT_NAME_ALIASES: Mapping[str, str] = {
    "yaml": "pyyaml", "absl": "absl-py", "PIL": "pillow", "sklearn": "scikit-learn", "dotenv": "python-dotenv",
    "dateutil": "python-dateutil", "attr": "attrs", "bs4": "beautifulsoup4", "jwt": "pyjwt", "cv2": "opencv-python",
}
RESOLVER_NORMALISED, RESOLVER_ALIAS, RESOLVER_INTERPRETER = "normalised-name", "alias-table", "interpreter-packages_distributions"
RULE = ("versions compared for exactly the distributions the MATH-500 client IMPORTS (an AST walk from the client venv's entry "
        "points over the repo modules it reaches; import names resolved to lock distributions) that are present in BOTH locks; a "
        "differing version or VCS url@rev of such a distribution refuses by name; an imported distribution present in one lock "
        "only, an import outside both locks, and every difference outside the imported set are DISCLOSED, never refused; the "
        "lock digests are recorded, not compared as a gate")

_REQ = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]+\])?==([^\s\\]+)")
_VCS = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]+\])? @ (\S+)")
_VIA_ONE = re.compile(r"^\s*#\s*via\s+(\S+)\s*$")
_VIA_MORE = re.compile(r"^\s*#\s{2,}(\S+)\s*$")


def normalize_name(name: str) -> str:
    """PEP 503: runs of ``-``, ``_``, ``.`` are one ``-``; case-insensitive."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_sha256(text: str) -> str:
    """The digest every tinycap store records (``served_capability_run._client_lock_sha256``: sha256 of the file bytes)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LockEntry:
    name: str
    norm: str
    version: str | None          # ``name==version``
    vcs: str | None              # ``name @ url@rev`` (unhashable by construction; installed separately at that commit)
    via: tuple[str, ...]         # the lock's own ``# via`` edges (requirers), normalised

    @property
    def spec(self) -> str:
        return self.version if self.version is not None else f"@ {self.vcs}"


def parse_lock(text: str) -> dict[str, LockEntry]:
    """The lock as ``{normalised name: entry}`` — requirement blocks (a ``==`` or ``@`` line plus its ``--hash`` /
    ``# via`` continuation lines); the header comments are skipped."""
    out: dict[str, LockEntry] = {}
    cur: tuple[str, str | None, str | None] | None = None
    via: list[str] = []

    def flush() -> None:
        if cur is not None:
            name, version, vcs = cur
            out[normalize_name(name)] = LockEntry(name=name, norm=normalize_name(name), version=version, vcs=vcs, via=tuple(via))

    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            m = _VIA_ONE.match(line) or _VIA_MORE.match(line)
            if m and cur is not None and not m.group(1).startswith("-"):
                via.append(normalize_name(m.group(1)))
            continue
        if line.startswith("#"):
            continue
        flush()
        cur, via = None, []
        m = _REQ.match(line)
        if m:
            cur = (m.group(1), m.group(3), None)
            continue
        m = _VCS.match(line)
        if m:
            cur = (m.group(1), None, m.group(3).rstrip("\\").strip())
    flush()
    return out


# --------------------------------------------------------------------------- the client's imports


@dataclass
class ImportWalk:
    entry_points: tuple[str, ...]
    roots: tuple[str, ...]
    modules: list[str] = field(default_factory=list)                     # repo modules walked (relative paths, sorted)
    third_party: dict[str, list[str]] = field(default_factory=dict)      # top-level import name -> importing modules (sorted)
    dynamic: dict[str, list[str]] = field(default_factory=dict)          # importlib.import_module("<literal>") -> importers
    stdlib: list[str] = field(default_factory=list)                      # stdlib top-level names seen (sorted)
    unresolved_relative: list[str] = field(default_factory=list)         # a relative import whose target file was not found
    unparseable: dict[str, str] = field(default_factory=dict)            # module -> the SyntaxError (a module the client could not import either)
    missing_entry_points: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {"entry_points": list(self.entry_points), "import_roots": list(self.roots), "n_modules_walked": len(self.modules),
                "n_third_party_names": len(self.third_party), "third_party_names": sorted(self.third_party),
                "dynamic_import_literals": dict(self.dynamic), "unresolved_relative_imports": list(self.unresolved_relative),
                "unparseable_modules": dict(self.unparseable), "missing_entry_points": list(self.missing_entry_points),
                "how": ("every import statement (Import / ImportFrom, at any depth, guarded or not) and every "
                        "importlib.import_module('<literal>') in every repo module reachable from the entry points through imports "
                        "that resolve to a file under the import roots; third-party = neither a repo module nor stdlib")}


def _stdlib_names() -> set[str]:
    return set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def _resolve_repo_module(repo: Path, dotted: str, roots: Sequence[str]) -> Path | None:
    parts = dotted.split(".")
    for root in roots:
        base_root = repo / root
        for i in range(len(parts), 0, -1):
            base = base_root.joinpath(*parts[:i])
            for cand in (base.with_suffix(".py"), base / "__init__.py"):
                if cand.is_file():
                    return cand
    return None


def walk_client_imports(repo: Path, entry_points: Sequence[str] = CLIENT_ENTRY_POINTS,
                        roots: Sequence[str] = CLIENT_IMPORT_ROOTS) -> ImportWalk:
    """The transitive import walk from the entry points (see the module docstring, step 1)."""
    repo = Path(repo)
    walk = ImportWalk(entry_points=tuple(entry_points), roots=tuple(roots))
    stdlib = _stdlib_names()
    seen: set[Path] = set()
    queue: list[Path] = []
    for rel in entry_points:
        p = repo / rel
        if p.is_file():
            queue.append(p)
        else:
            walk.missing_entry_points.append(rel)
    third: dict[str, set[str]] = {}
    dyn: dict[str, set[str]] = {}
    std: set[str] = set()
    while queue:
        f = queue.pop()
        if f in seen:
            continue
        seen.add(f)
        rel = f.relative_to(repo).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError) as e:  # a module the client venv could not import either — recorded
            walk.unparseable[rel] = f"{type(e).__name__}: {e}"
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    pkg = f.parent
                    for _ in range(node.level - 1):
                        pkg = pkg.parent
                    base = pkg.joinpath(*node.module.split(".")) if node.module else pkg
                    for a in node.names:
                        cands = [base / f"{a.name}.py", base / a.name / "__init__.py", base.with_suffix(".py"), base / "__init__.py"]
                        hit = next((c for c in cands if c.is_file()), None)
                        if hit is not None:
                            queue.append(hit)
                        else:
                            walk.unresolved_relative.append(f"{rel}: from {'.' * node.level}{node.module or ''} import {a.name}")
                    continue
                assert node.module is not None
                names = [f"{node.module}.{a.name}" for a in node.names] + [node.module]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                  and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                lit = node.args[0].value
                dyn.setdefault(lit, set()).add(rel)
                names = [lit]
            for dotted in names:
                target = _resolve_repo_module(repo, dotted, roots)
                if target is not None:
                    queue.append(target)
                    continue
                top = dotted.split(".")[0]
                if top in stdlib:
                    std.add(top)
                    continue
                third.setdefault(top, set()).add(rel)
    walk.modules = sorted(p.relative_to(repo).as_posix() for p in seen)
    walk.third_party = {k: sorted(v) for k, v in sorted(third.items())}
    walk.dynamic = {k: sorted(v) for k, v in sorted(dyn.items())}
    walk.stdlib = sorted(std)
    return walk


# --------------------------------------------------------------------------- import name -> lock distribution


def interpreter_distributions() -> dict[str, list[str]]:
    """``importlib.metadata.packages_distributions()`` of the RUNNING interpreter (a name resolver only — the lock, never
    the interpreter, says what the client has); ``{}`` when unavailable."""
    try:
        import importlib.metadata as md
        return {k: sorted(v) for k, v in md.packages_distributions().items()}
    except Exception:  # noqa: BLE001 — a resolver, not a fact
        return {}


def resolve_import(top: str, locks: Sequence[Mapping[str, LockEntry]], interpreter_map: Mapping[str, Sequence[str]] | None = None,
                   aliases: Mapping[str, str] = IMPORT_NAME_ALIASES) -> dict[str, Any]:
    """``{"distributions": [normalised names in some lock], "resolver": <which rule>, "candidates": [...]}``; empty
    distributions + resolver None when the name matches nothing in either lock (see the module docstring, step 3)."""
    present = set().union(*[set(l) for l in locks]) if locks else set()
    norm = normalize_name(top)
    if norm in present:
        return {"distributions": [norm], "resolver": RESOLVER_NORMALISED, "candidates": [norm]}
    alias = aliases.get(top)
    if alias is not None and normalize_name(alias) in present:
        return {"distributions": [normalize_name(alias)], "resolver": RESOLVER_ALIAS, "candidates": [normalize_name(alias)]}
    cands = sorted({normalize_name(d) for d in (interpreter_map or {}).get(top, ())})
    hits = [c for c in cands if c in present]
    if hits:
        return {"distributions": hits, "resolver": RESOLVER_INTERPRETER, "candidates": cands}
    return {"distributions": [], "resolver": None, "candidates": cands + ([normalize_name(alias)] if alias else [])}


# --------------------------------------------------------------------------- the comparison


def compare_locks(published_text: str, wave_text: str, walk: ImportWalk, *,
                  interpreter_map: Mapping[str, Sequence[str]] | None = None) -> tuple[dict[str, Any], list[str]]:
    """(record, problems) — the module docstring's step 4. Strings, bools, lists and dicts only (JSON-ready)."""
    pub, wave = parse_lock(published_text), parse_lock(wave_text)
    imap = interpreter_map if interpreter_map is not None else interpreter_distributions()
    imported: dict[str, dict[str, Any]] = {}           # distribution -> {import_names, importers, resolver}
    not_in_lock: dict[str, dict[str, Any]] = {}
    for top, importers in walk.third_party.items():
        res = resolve_import(top, (pub, wave), imap)
        if not res["distributions"]:
            not_in_lock[top] = {"importers": list(importers), "candidates_tried": res["candidates"]}
            continue
        for dist in res["distributions"]:
            rec = imported.setdefault(dist, {"import_names": [], "importers": [], "resolver": res["resolver"]})
            rec["import_names"] = sorted(set(rec["import_names"]) | {top})
            rec["importers"] = sorted(set(rec["importers"]) | set(importers))
    compared: dict[str, dict[str, Any]] = {}
    imported_added: dict[str, Any] = {}
    imported_removed: dict[str, Any] = {}
    problems: list[str] = []
    for dist in sorted(imported):
        p, w = pub.get(dist), wave.get(dist)
        meta = imported[dist]
        if p is not None and w is not None:
            equal = p.spec == w.spec
            compared[dist] = {"published": p.spec, "wave": w.spec, "equal": equal, **meta}
            if not equal:
                problems.append(f"client_lock.{dist}: published {p.spec} != wave {w.spec} (imported by {meta['importers'][:3]}"
                                f"{' …' if len(meta['importers']) > 3 else ''})")
        elif w is not None:
            imported_added[dist] = {"wave": w.spec, **meta}
        else:
            imported_removed[dist] = {"published": p.spec, **meta}
    outside_names = sorted((set(pub) | set(wave)) - set(imported))
    outside = {"added": sorted(n for n in outside_names if n in wave and n not in pub),
               "removed": sorted(n for n in outside_names if n in pub and n not in wave),
               "changed": {n: {"published": pub[n].spec, "wave": wave[n].spec} for n in outside_names
                           if n in pub and n in wave and pub[n].spec != wave[n].spec}}
    changed = sorted(d for d, r in compared.items() if not r["equal"])
    sha_p, sha_w = lock_sha256(published_text), lock_sha256(wave_text)
    one = not changed
    parts = [f"lock digest {'unchanged' if sha_p == sha_w else f'moved ({sha_p[:8]}… → {sha_w[:8]}…)'}",
             f"{len(compared)} imported distributions present in both locks, {len(changed)} changed version"
             + (f" ({', '.join(changed)})" if changed else ""),
             f"imported at the wave tree only: {sorted(imported_added) or 'none'}",
             f"imported at the published tree only: {sorted(imported_removed) or 'none'}",
             f"outside the imported set — added {outside['added'] or 'none'}, removed {outside['removed'] or 'none'}, "
             f"changed {sorted(outside['changed']) or 'none'}",
             f"imports outside both locks (not installable in the client venv): {sorted(not_in_lock) or 'none'}",
             "ONE instrument: no imported distribution changed version" if one else "TWO instruments: an imported distribution changed version"]
    return {
        "rule": RULE,
        "sha256": {"published": sha_p, "wave": sha_w, "equal": sha_p == sha_w},
        "equal": one, "one_instrument": one,
        "n_compared": len(compared), "changed": changed,
        "imported_packages": compared,
        "imported_added": imported_added, "imported_removed": imported_removed,
        "imports_not_in_lock": not_in_lock,
        "outside_imported_set": outside,
        "n_published_entries": len(pub), "n_wave_entries": len(wave),
        "walk": walk.record(),
        "disclosure": "; ".join(parts),
    }, problems


# --------------------------------------------------------------------------- the published lock's text: the chain of custody


def lock_text_at_tree(repo: Path, tree_sha: str, rel: str = LOCK_REL) -> tuple[str | None, str | None]:
    """``(text, None)`` from ``git show <tree>:<rel>`` in ``repo``; ``(None, why)`` when the tree or the file cannot be
    read here (a shallow clone without that commit, a tree that predates the lock)."""
    try:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{tree_sha}:{rel}"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"git show {tree_sha[:12]}:{rel} failed ({type(e).__name__}: {e})"
    if r.returncode != 0:
        return None, f"git show {tree_sha[:12]}:{rel} failed ({r.stderr.strip()[:160]})"
    return r.stdout, None


def published_lock_record(text: str | None, why: str | None, recorded_sha256: str | None, tree_sha: str | None,
                          rel: str = LOCK_REL) -> dict[str, Any]:
    """The published side's record: where the text came from, its digest, and whether it EQUALS the digest the leg
    recorded (``manifest.frameworks.client_lock_sha256``). ``matches_record`` None when the text is unreadable."""
    digest = lock_sha256(text) if text is not None else None
    return {"tree": tree_sha, "source": f"git show <tree>:{rel}", "readable": text is not None, "why_unreadable": why,
            "sha256_of_tree_lock": digest, "sha256_recorded": recorded_sha256,
            "matches_record": (digest == recorded_sha256) if digest is not None and recorded_sha256 else None}
