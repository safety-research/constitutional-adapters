"""Source -> adapter coverage: every legacy source the paper DECLARES must be covered by exactly one
adapter's ``covers_sources`` (or by several that declare distinct priorities).

Declared = the round's committed source list (``configs/registry/sources_r2_v1.yaml`` -- the ``sources`` keys of the
paper build's watermarks file at the parity pin, names only) + every ``source_legacy`` string the roster names
(``configs/registry/roster_r2_v1.yaml``) + the LIVE paper checkout's watermarks file when one is configured and present
(so a source the paper adds shows up on the status page before an adapter declares it).

Covered = the union over ``adapters.load_all()`` of each adapter's ``covers_sources`` and its optional
``declared_sources`` -- the strings an adapter is written to cover even when nothing is on disk (the strength ladder
DISCOVERS its records, so on a host without the grade tree it would otherwise cover nothing and the check would report
the paper's Figure 5 sources missing on every Modal gate).

Pinned by ``tests/test_registry_source_coverage.py``; rendered live by ``registry status`` ("Sources without an
adapter"). Before this the gap was invisible: the two ``*-superseded`` EM panels the paper declares were ingested under
the ``_bl`` stem but named as sources by nothing, and no test enumerated the declared set (26 of 28 covered, found by
the acceptance audit's one-off probe).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

DECLARED_RELPATH = os.path.join("configs", "registry", "sources_r2_v1.yaml")
ROSTER_RELPATH = os.path.join("configs", "registry", "roster_r2_v1.yaml")
WATERMARKS_RELPATH = os.path.join("paper", "paperlib", "lane_watermarks.json")
#: roster `source_legacy` roles that record a reading the paper does NOT read (a spelling with prose, not a source string)
ROSTER_ROLES_NOT_DECLARED = ("not_canonical",)


def repo_root() -> Optional[str]:
    """The checkout this package lives in, when it carries ``configs/registry`` (a src-only tree -- the Modal mount, a
    staged launcher tree -- carries none and declares nothing here)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    return root if os.path.isdir(os.path.join(root, "configs", "registry")) else None


def _repo_path(rel: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    root = repo_root()
    return os.path.join(root, rel) if root else None


# --------------------------------------------------------------------------------------------
# the declared set
# --------------------------------------------------------------------------------------------

def read_declared_file(path: str) -> Tuple[Dict[str, str], str]:
    """``source -> owner`` from the committed source list, plus its provenance label (file @ paper pin)."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    meta = data.get("meta") or {}
    pin = str(meta.get("paper_build_commit") or "")[:8]
    label = f"{os.path.basename(path)} (paper watermarks @ {pin or 'unpinned'})"
    out: Dict[str, str] = {}
    for row in data.get("sources") or []:
        if isinstance(row, Mapping) and row.get("source"):
            out[str(row["source"])] = str(row.get("owner") or "")
    return out, label


def source_tokens(value) -> Tuple[List[str], List[str]]:
    """``(sources, placeholders)`` one ``source_legacy`` value names. A string names its source up to the first space or
    ``(`` (the rest is prose); ``a/{x,y}`` expands to ``a/x``, ``a/y``; a ``<...>`` placeholder names a FAMILY of sources,
    not one, and is returned separately; None names nothing; a list or a role -> value mapping is walked."""
    sources: List[str] = []
    placeholders: List[str] = []
    if value is None:
        return sources, placeholders
    if isinstance(value, Mapping):
        for role, v in value.items():
            if str(role) in ROSTER_ROLES_NOT_DECLARED:
                continue
            s, p = source_tokens(v)
            sources += s
            placeholders += p
        return sources, placeholders
    if isinstance(value, (list, tuple)):
        for v in value:
            s, p = source_tokens(v)
            sources += s
            placeholders += p
        return sources, placeholders
    text = str(value).strip()
    if not text:
        return sources, placeholders
    head = text.split(" (", 1)[0].split(" ", 1)[0].strip()
    if "<" in head:
        placeholders.append(head)
        return sources, placeholders
    if "{" in head and "}" in head:
        pre, rest = head.split("{", 1)
        inner, post = rest.split("}", 1)
        for alt in inner.split(","):
            alt = alt.strip()
            if alt:
                sources.append(f"{pre}{alt}{post}")
        return sources, placeholders
    sources.append(head)
    return sources, placeholders


def roster_sources(path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """``source -> [where declared]`` for every ``source_legacy`` field anywhere in the roster (labelled by its path), and
    the placeholders met on the way."""
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    name = os.path.basename(path)
    out: Dict[str, List[str]] = {}
    placeholders: List[str] = []

    def walk(node, where: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if str(k) == "source_legacy":
                    s, p = source_tokens(v)
                    for src in s:
                        out.setdefault(src, []).append(f"{name} {where}.source_legacy".lstrip("."))
                    placeholders.extend(f"{where}.source_legacy: {x}" for x in p)
                else:
                    walk(v, f"{where}.{k}" if where else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")

    walk(data, "")
    return out, placeholders


def watermark_sources(paper_checkout: Optional[str]) -> Tuple[Dict[str, str], Optional[str]]:
    """``source -> owner`` from a paper checkout's live watermarks file, and the label to file them under; ``({}, None)``
    when no checkout is given or the file is absent."""
    if not paper_checkout:
        return {}, None
    path = os.path.join(os.path.expanduser(paper_checkout), WATERMARKS_RELPATH)
    if not os.path.isfile(path):
        return {}, None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    srcs = data.get("sources") if isinstance(data, Mapping) else None
    out = {str(k): str((v or {}).get("owner") or "") if isinstance(v, Mapping) else "" for k, v in (srcs or {}).items()}
    return out, f"paper checkout watermarks ({os.path.expanduser(paper_checkout)})"


def declared_sources(declared_path: Optional[str] = None, roster_path: Optional[str] = None,
                     paper_checkout: Optional[str] = None) -> Dict[str, object]:
    """``{"declared": {source: [where, ...]}, "owners": {source: owner}, "placeholders": [...], "inputs": [...],
    "errors": [...]}`` -- the union of the three declared sets; an input that cannot be read is an ERROR in the result,
    never a silently thinner set."""
    declared: Dict[str, List[str]] = {}
    owners: Dict[str, str] = {}
    placeholders: List[str] = []
    inputs: List[str] = []
    errors: List[str] = []
    dp = _repo_path(DECLARED_RELPATH, declared_path)
    if dp:
        try:
            srcs, label = read_declared_file(dp)
            inputs.append(dp)
            for s, owner in srcs.items():
                declared.setdefault(s, []).append(label)
                owners.setdefault(s, owner)
        except (OSError, ValueError) as exc:
            errors.append(f"{dp}: {exc}")
    else:
        errors.append(f"{DECLARED_RELPATH} not found (no configs/registry beside this package)")
    rp = _repo_path(ROSTER_RELPATH, roster_path)
    if rp:
        try:
            srcs2, ph = roster_sources(rp)
            inputs.append(rp)
            for s, where in srcs2.items():
                declared.setdefault(s, []).extend(where)
            placeholders += ph
        except (OSError, ValueError) as exc:
            errors.append(f"{rp}: {exc}")
    else:
        errors.append(f"{ROSTER_RELPATH} not found (no configs/registry beside this package)")
    try:
        srcs3, label3 = watermark_sources(paper_checkout)
        if label3:
            inputs.append(label3)
            for s, owner in srcs3.items():
                declared.setdefault(s, []).append(label3)
                owners.setdefault(s, owner)
    except (OSError, ValueError) as exc:
        errors.append(f"{paper_checkout}: {exc}")
    return {"declared": declared, "owners": owners, "placeholders": placeholders, "inputs": inputs, "errors": errors}


# --------------------------------------------------------------------------------------------
# the covered set
# --------------------------------------------------------------------------------------------

def adapter_sources(adapter) -> Tuple[str, ...]:
    """The paper source strings one adapter covers: ``covers_sources`` (what it files rows under today) plus
    ``declared_sources`` (what it is written to cover whether or not its records are on this host), deduplicated."""
    seen: Dict[str, None] = {}
    for attr in ("covers_sources", "declared_sources"):
        for s in (getattr(adapter, attr, ()) or ()):
            seen.setdefault(str(s), None)
    return tuple(seen)


def covered_sources(registry: Optional[Mapping[str, object]] = None) -> Dict[str, List[Tuple[str, int]]]:
    """``source -> [(adapter name, declared priority), ...]`` over every registered adapter (``adapters.load_all()``)."""
    from . import adapters as ADP
    from .adapters.runner import adapter_priority
    reg = ADP.load_all() if registry is None else registry
    out: Dict[str, List[Tuple[str, int]]] = {}
    for name in sorted(reg):
        inst = reg[name]()
        for s in adapter_sources(inst):
            out.setdefault(s, []).append((str(getattr(inst, "name", name)), adapter_priority(inst)))
    return out


# --------------------------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------------------------

def coverage(declared: Mapping[str, Sequence[str]], covered: Mapping[str, Sequence[Tuple[str, int]]]) -> Dict[str, object]:
    """Pure: declared minus covered, and every source two adapters claim at ONE priority (0 = undeclared)."""
    without = [{"source": s, "declared_in": sorted(set(declared[s]))} for s in sorted(declared) if s not in covered]
    twice = []
    for s in sorted(covered):
        claims = sorted(set(covered[s]))
        if len(claims) > 1:
            prios = [p for _, p in claims]
            if len(set(prios)) != len(prios):
                twice.append({"source": s, "adapters": [f"{a} (priority {p})" for a, p in claims]})
    return {"n_declared": len(declared), "n_covered": len(covered), "without_adapter": without, "claimed_without_priority": twice}


def live_coverage(root: Optional[str] = None, paper_checkout: Optional[str] = None, *,
                  declared_path: Optional[str] = None, roster_path: Optional[str] = None,
                  registry: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """The status page's section: the declared set (the committed list, the roster, and the configured paper checkout's
    live file -- ``[paper] checkout`` of ``<root>/config.toml`` when ``root`` is given and no checkout is passed) against
    what the loaded adapters cover."""
    if paper_checkout is None and root:
        try:
            from . import config as CFG
            paper_checkout = str(CFG.load(root)["paper"]["checkout"])
        except Exception:   # noqa: BLE001 -- no config, no live file: the committed list still applies
            paper_checkout = None
    dec = declared_sources(declared_path=declared_path, roster_path=roster_path, paper_checkout=paper_checkout)
    cov = covered_sources(registry)
    out = coverage(dec["declared"], cov)   # type: ignore[arg-type]
    for row in out["without_adapter"]:     # type: ignore[union-attr]
        row["owner"] = dec["owners"].get(row["source"], "")   # type: ignore[index]
    out.update({"placeholders": dec["placeholders"], "inputs": dec["inputs"], "errors": dec["errors"], "paper_checkout": paper_checkout})
    return out
