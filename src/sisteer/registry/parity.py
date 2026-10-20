"""The paper parity gate: ``registry check-paper-parity --paper-checkout <pinned dir> --pin <sha>``.

Two legs, each a FRESH subprocess, compared here:

* the LEGACY leg runs the pinned checkout's own readers (``paperlib`` imported from ``<checkout>/paper`` on
  ``PYTHONPATH``, the paper's venv as ``--paper-python``), cwd a scratch directory so nothing is written into the
  checkout, ``figs.FIGDIR`` redirected; it dumps every reader's frame (L0), the assembled ``load_all(guard=False)``
  frame with its attrs (L1), the keys the canonical paper view keeps (L1b), the pinned ``COLUMNS`` and watermarks (L1c),
  and — with ``--l2`` — every figure's artist data from the notebook's code cells (L2);
* the REGISTRY leg opens the registry read-only under ``PAPERLIB_FORBID_LEGACY_READS=1``: :func:`install_legacy_read_hook`
  puts a ``sys.addaudithook`` on ``open`` / ``os.listdir`` / ``os.scandir`` / ``glob`` for the legacy roots and raises
  :class:`LegacyReadForbidden` naming the path, so any surviving legacy read fails the gate by name; it dumps
  ``paper.paper_long_frame(round, columns=<COLUMNS read from the pin>)``, ``v_paper_current``'s selected keys and the
  lane statuses.

The gate pins an INPUT SET: the sha256 of every input the pinned checkout reads (the r2 parquet / schema /
sources, the record files the readers declare, the watermarks) is taken before the legacy leg and again after the
registry leg, in THIS process (the registry leg's hook forbids it there); a moved sha REFUSES the gate. Levels
(`L0`, `L1`, `L1b`, `L1c`, `L2`) each write a ``parity_gate`` record (a harvester fact under ``records/parity/``) and a
section of the report under ``<root>/state/parity/<gate_id>/``: the twelve build-vs-ruling disagreements first, then
the expected breaks (the two llama declarations already expired; the two known-red pinned
tests), then per-key diffs. Values compare to 1e-9 (``rtol=0``), NaN == NaN, strings exact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import db as DB
from . import records as R

TOL = 1e-9
#: The row key plus `dose`: the paper's frame carries a row per strength level under otherwise identical keys (the
#: readers' own RUNG_KEY has it), so without it both sides fork into duplicates and a 0.5 row is compared with a 1.0 row.
L0_KEY = ("source", "model", "lane_kind", "arm", "eval_id", "metric", "threshold", "decode_cap_tokens", "judge", "pass", "dose")
L0_VALUES = ("value", "n", "delta", "delta_lo", "delta_hi")
LEVELS = ("L0", "L1", "L1b", "L1c", "L2")
DEFAULT_PAPER_PYTHON = os.path.join(os.path.expanduser("~"), "venvs", "sdfbase-r2-nb", "bin", "python")
#: Roots the registry leg may never read; ``$HOME``-relative so the hook is machine-agnostic.
#: A root that starts with ``$`` is a site placeholder (the grade root, the staging directory, the state root, the working-data root),
#: expanded through sisteer.site when the hook installs; one this host cannot resolve REFUSES the gate (:func:`legacy_roots`).
LEGACY_ROOTS = ("$GRADE", "$STAGE", "$STATE/romp", "figs/si-steering-misalign", "$WORK/bcb-bfcl-200", ".cache/paperlib-llama8b",
                "reviews/sdf-base-steering-r2", "figs/degeneracy")
#: The two tests known red at the pin: listed as expected, never a parity failure.
EXPECTED_RED_TESTS = ("tests/test_fig5_frontier_tweaks.py",)
#: The rulings already expired on disk at the pin: the registry's first evaluation reporting them expired is expected.
EXPECTED_EXPIRED_RULINGS = ("r2v1-supersede-llama-arditi-r2-petri-r0p1", "r2v1-supersede-llama-arditi-r2-multiturn-r0p1")


class ParityRefused(RuntimeError):
    """A precondition failed (wrong sha, missing venv, an input moved between the legs): no comparison was made."""


class LegacyReadForbidden(RuntimeError):
    """The registry leg opened a legacy root."""


# --------------------------------------------------------------------------------------------
# the audit hook (installed in the registry leg's process before anything else)
# --------------------------------------------------------------------------------------------


def legacy_roots(home: Optional[str] = None) -> List[str]:
    from .. import site as _site
    home = home or os.path.expanduser("~")
    roots = []
    for r in LEGACY_ROOTS:
        if not r.startswith("$"):
            roots.append(os.path.join(home, r))
            continue
        try:
            roots.append(str(_site.expand_path(r)))
        except (_site.PlaceholderError, _site.SiteConfigError) as exc:
            # fail closed: a guard root this host cannot place would let a legacy read under it pass unseen
            raise ParityRefused(f"the registry leg's legacy-read guard cannot place {r}: {exc}") from exc
    extra = os.environ.get("PAPERLIB_LEGACY_ROOTS")
    if extra:
        roots += [p for p in extra.split(os.pathsep) if p]
    return [os.path.abspath(r) for r in roots]


def install_legacy_read_hook(roots: Optional[Sequence[str]] = None) -> None:
    """``sys.addaudithook`` on open / os.listdir / os.scandir / glob.glob under the legacy roots → LegacyReadForbidden(path)."""
    roots = [os.path.abspath(r) for r in (roots or legacy_roots())]

    def under(path) -> Optional[str]:
        if not isinstance(path, (str, bytes, os.PathLike)):
            return None
        p = os.fsdecode(path) if not isinstance(path, str) else path
        try:
            ap = os.path.abspath(os.path.expanduser(p))
        except (TypeError, ValueError):
            return None
        for r in roots:
            if ap == r or ap.startswith(r + os.sep):
                return ap
        return None

    def hook(event, args):
        if event in ("open", "os.listdir", "os.scandir", "glob.glob", "os.stat"):
            if event == "os.stat":
                return   # stat is how the hashing parent works; reading content is what is forbidden
            hit = under(args[0] if args else None)
            if hit:
                raise LegacyReadForbidden(f"legacy read forbidden on the registry leg: {event} {hit}")
    sys.addaudithook(hook)


# --------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------


def sha256_of(path: str) -> str:
    if not os.path.isfile(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_inputs(paths: Iterable[str]) -> Dict[str, str]:
    return {p: sha256_of(os.path.expanduser(p)) for p in sorted(set(paths))}


# --------------------------------------------------------------------------------------------
# legacy leg
# --------------------------------------------------------------------------------------------

LEGACY_SIDE = r'''
import json, os, sys, math, traceback
work, pdir, want_l2 = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
sys.path.insert(0, pdir)
import matplotlib
matplotlib.use("Agg")
import pandas as pd, numpy as np
from paperlib import config as C, load as L
out = {"columns": list(L.COLUMNS), "inputs": [], "readers": {}, "errors": {}}
for name in ("R2_PARQUET", "R2_SOURCES", "R2_TRANSFER_ITEMS", "R2_TRANSFER_STATS"):
    if getattr(C, name, None):
        out["inputs"].append(getattr(C, name))
for rec in getattr(C, "STRENGTH_RECORDS", []) or []:
    out["inputs"].append(rec.path)
for mod, attr in (("lane_alphasteer", "OPERATING_RECORD"), ("lane_alphasteer", "DEFENCE_RECORD"), ("lane_arditi", "CURVES")):
    try:
        m = __import__("paperlib." + mod, fromlist=[attr])
        v = getattr(m, attr, None)
        if v:
            out["inputs"].append(str(v))
    except Exception as exc:
        out["errors"]["inputs:" + mod] = str(exc)
out["inputs"].append(os.path.join(pdir, "paperlib", "lane_watermarks.json"))
def _parquet_value(v):
    # object columns carry strings-or-None by contract; anything else (a list, a dict, a mixed bool / float flag column such as
    # degenerate_win) is written as JSON text so the parquet writer never refuses a mixed column
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, float) and v != v:
        return None
    return json.dumps(v, default=str)
def dump(df, name):
    df = df.copy()
    df.attrs = {}      # the frame attrs (non-canonical, superseded cells, pd.NA inside) are dumped as JSON separately, never as parquet metadata
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(_parquet_value)
    df.to_parquet(os.path.join(work, name + ".parquet"))
    return len(df)
# L0: every reader on its own
readers = [("load_r2", lambda: L.load_r2()[0]), ("load_overrefusal_graded", lambda: L.load_overrefusal_graded()[0])]
readers += [(fn.__name__, (lambda f=fn: f()[0])) for fn in L.SECONDARY]
for name, fn in readers:
    try:
        df = fn()
        if df is None or not len(df):
            out["readers"][name] = 0
            continue
        df = df.reindex(columns=list(L.COLUMNS) + [c for c in df.columns if c not in L.COLUMNS])
        out["readers"][name] = dump(df, "l0_" + name)
    except Exception as exc:
        out["errors"]["l0:" + name] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
# L1: the assembled frame with attrs; L1b: the canonical paper view's keys with a value
rows = None
try:
    rows, meta, statuses = L.load_all(guard=False)
    out["l1_attrs"] = {k: json.loads(json.dumps(v, default=str)) for k, v in rows.attrs.items()}
    out["r2_meta"] = json.loads(json.dumps(meta, default=str))
    out["lane_statuses"] = [dict(name=s.name, owner=s.owner, integrated=bool(s.integrated), where=str(s.where), detail=str(s.detail),
                                 cells_measured=int(getattr(s, "cells_measured", 0))) for s in statuses]
except Exception as exc:
    out["errors"]["l1"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
if rows is not None:
    # the watermark counts and the canonical keys are taken BEFORE any parquet write, so a writer refusal cannot lose them
    try:
        from paperlib import lane_guard as LG
        out["l1c_measured_counts"] = LG.measured_counts(rows)
        out["l1c_watermarks"] = LG.load_watermarks()
    except Exception as exc:
        out["errors"]["l1c"] = str(exc)
    try:
        cv = L.canonical_view(L.paper_lanes(rows))
        keep = cv[cv.value.notna()] if "value" in cv.columns else cv
        out["l1b_keys"] = dump(keep.reindex(columns=list(L.COLUMNS)), "l1b_canonical_keys")
    except Exception as exc:
        out["errors"]["l1b"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
    try:
        out["l1_rows"] = dump(rows, "l1_assembled")
    except Exception as exc:
        out["errors"]["l1"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
# L2: the notebook's code cells with figs.save wrapped to dump every Axes' artist data
if want_l2:
    try:
        from paperlib import figs as F
        import pathlib
        figdir = pathlib.Path(work) / "figures"; figdir.mkdir(exist_ok=True)
        F.FIGDIR = figdir
        fdd = pathlib.Path(work) / "figdata"; fdd.mkdir(exist_ok=True)
        _save = F.save
        def save(fig, name):
            (fdd / (name + ".figdata.json")).write_text(json.dumps(figdata(fig), sort_keys=True))
            return _save(fig, name)
        F.save = save
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else None
        from sisteer_parity_figdata import figdata   # written beside this script by the parent
        nb = json.load(open(os.path.join(pdir, "constitutional_adapters_figures.ipynb")))
        g = {"__name__": "__main__", "display": lambda *a, **k: None}
        os.chdir(work)
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None
        n_ok = 0
        for i, cell in enumerate(c for c in nb["cells"] if c["cell_type"] == "code"):
            src = "".join(cell["source"])
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
            try:
                exec(compile(src, f"<cell {i}>", "exec"), g)
                n_ok += 1
            except Exception as exc:
                out["errors"][f"l2:cell{i}"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()[:2000]
            plt.close("all")
        out["l2_cells_ok"] = n_ok
        out["l2_figdata"] = sorted(p.name for p in fdd.glob("*.figdata.json"))
    except Exception as exc:
        out["errors"]["l2"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
json.dump(out, open(os.path.join(work, "legacy_side.json"), "w"), default=str)
'''

FIGDATA_MODULE = r'''
"""Matplotlib artist data of a figure as plain JSON (the parity gate's L2 definition, plan §6.5 / §13 C5)."""
import json, math
def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return None if math.isnan(f) else f
def figdata(fig):
    out = {"axes": []}
    for ax in fig.get_axes():
        a = {"title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(), "lines": [], "patches": [], "collections": [], "texts": []}
        for ln in ax.get_lines():
            xy = ln.get_xydata()
            a["lines"].append({"label": str(ln.get_label()), "xy": [[_num(x), _num(y)] for x, y in xy]})
        for p in ax.patches:
            if hasattr(p, "get_x") and hasattr(p, "get_width"):
                a["patches"].append({"x": _num(p.get_x()), "y": _num(p.get_y()), "w": _num(p.get_width()), "h": _num(p.get_height())})
        for c in ax.collections:
            try:
                offs = c.get_offsets()
                a["collections"].append({"offsets": [[_num(x), _num(y)] for x, y in offs]})
            except Exception:
                a["collections"].append({"offsets": []})
            try:
                segs = c.get_segments()
                a["collections"][-1]["segments"] = [[[_num(x), _num(y)] for x, y in s] for s in segs]
            except Exception:
                pass
        for t in ax.texts:
            x, y = t.get_position()
            a["texts"].append({"x": _num(x), "y": _num(y), "s": t.get_text()})
        out["axes"].append(a)
    return out
'''

REGISTRY_SIDE = r'''
import json, os, sys
work, root, round_id = sys.argv[1], sys.argv[2], sys.argv[3] or None
from sisteer.registry import parity as P
P.install_legacy_read_hook()
from sisteer.registry import db as DB, paper
columns = json.load(open(os.path.join(work, "columns.json")))
conn = DB.connect(DB.db_path(root), readonly=True)
out = {"errors": {}}
try:
    long = paper.paper_long_frame(conn, round_id, columns=columns, with_dispositions=True, current_only=False)
    for c in long.columns:
        if long[c].dtype == object:
            long[c] = long[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict, tuple)) else v)
    long.to_parquet(os.path.join(work, "reg_long.parquet"))
    out["rows"] = int(len(long))
    sel = long[(long["in_current_paper"] == 1) & long["value"].notna()] if "value" in long.columns else long
    sel.to_parquet(os.path.join(work, "reg_current_keys.parquet"))
    out["current_keys"] = int(len(sel))
    out["lane_statuses"] = paper.source_statuses(conn, round_id)
    out["head"] = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
    out["rulings"] = [dict(r) for r in conn.execute("SELECT ruling_id, action, evidence_state, expired_utc FROM ruling").fetchall()]
    out["disagreements"] = [dict(r) for r in conn.execute("SELECT comparison_id, model_id, eval_id, json_extract(extra_json, '$.disagreement') AS d FROM roster_entry "
                                                            "WHERE json_extract(extra_json, '$.disagreement') IS NOT NULL").fetchall()]
except Exception as exc:
    import traceback
    out["errors"]["registry"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
json.dump(out, open(os.path.join(work, "registry_side.json"), "w"), default=str)
'''


def run_legacy_side(checkout: str, work: str, python: str, want_l2: bool, timeout_s: int = 3600) -> Dict[str, object]:
    pdir = paper_dir(checkout)
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "sisteer_parity_figdata.py"), "w", encoding="utf-8") as fh:
        fh.write(FIGDATA_MODULE)
    with open(os.path.join(work, "legacy_side.py"), "w", encoding="utf-8") as fh:
        fh.write(LEGACY_SIDE)
    env = dict(os.environ, PYTHONPATH=pdir + os.pathsep + work, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([python, os.path.join(work, "legacy_side.py"), work, pdir, "1" if want_l2 else "0"], capture_output=True, text=True,
                          env=env, timeout=timeout_s, cwd=work)
    path = os.path.join(work, "legacy_side.json")
    if proc.returncode != 0 or not os.path.exists(path):
        raise ParityRefused(f"legacy leg failed (rc {proc.returncode}):\n{proc.stderr[-6000:]}")
    with open(path, "r", encoding="utf-8") as fh:
        out = json.load(fh)
    out["stderr_tail"] = proc.stderr[-2000:]
    return out


def run_registry_side(root: str, work: str, columns: Sequence[str], round_id: Optional[str], python: Optional[str] = None,
                     timeout_s: int = 1800) -> Dict[str, object]:
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "columns.json"), "w", encoding="utf-8") as fh:
        json.dump(list(columns), fh)
    with open(os.path.join(work, "registry_side.py"), "w", encoding="utf-8") as fh:
        fh.write(REGISTRY_SIDE)
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PAPERLIB_FORBID_LEGACY_READS="1", PYTHONPATH=src + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
               MPLBACKEND="Agg")
    proc = subprocess.run([python or sys.executable, os.path.join(work, "registry_side.py"), work, root, round_id or ""], capture_output=True, text=True,
                          env=env, timeout=timeout_s, cwd=work)
    path = os.path.join(work, "registry_side.json")
    if proc.returncode != 0 or not os.path.exists(path):
        raise ParityRefused(f"registry leg failed (rc {proc.returncode}):\n{proc.stderr[-6000:]}")
    with open(path, "r", encoding="utf-8") as fh:
        out = json.load(fh)
    out["stderr_tail"] = proc.stderr[-2000:]
    return out


def paper_dir(checkout: str) -> str:
    for cand in (os.path.join(checkout, "paper"), checkout):
        if os.path.isdir(os.path.join(cand, "paperlib")):
            return cand
    raise ParityRefused(f"{checkout}: no paperlib/ under it or under paper/")


def checkout_sha(checkout: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise ParityRefused(f"{checkout}: not a git checkout ({exc})") from exc


# --------------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------------


def _same(a, b, tol: float = TOL) -> bool:
    if a is None and b is None:
        return True
    fa = isinstance(a, float) and math.isnan(a)
    fb = isinstance(b, float) and math.isnan(b)
    if fa or fb:
        return fa and fb
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(float(a) - float(b)) <= tol
    return str(a) == str(b)


def _key_str(row: Mapping[str, object], key: Sequence[str]) -> str:
    parts = []
    for k in key:
        v = row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            parts.append("NA")
        elif isinstance(v, float) and v.is_integer():
            parts.append(str(int(v)))
        else:
            parts.append(str(v))
    return "|".join(parts)


def compare_frames(legacy_rows: Sequence[Mapping[str, object]], registry_rows: Sequence[Mapping[str, object]], key: Sequence[str] = L0_KEY,
                   values: Sequence[str] = L0_VALUES, tol: float = TOL) -> Dict[str, object]:
    """Per-key diff of two row lists: keys only on one side, keys on both whose values differ (named), duplicates."""
    la: Dict[str, Mapping[str, object]] = {}
    lb: Dict[str, Mapping[str, object]] = {}
    dup_a: List[str] = []
    dup_b: List[str] = []
    for r in legacy_rows:
        k = _key_str(r, key)
        (dup_a.append(k) if k in la else la.__setitem__(k, r))
    for r in registry_rows:
        k = _key_str(r, key)
        (dup_b.append(k) if k in lb else lb.__setitem__(k, r))
    only_legacy = sorted(set(la) - set(lb))
    only_registry = sorted(set(lb) - set(la))
    differing = []
    for k in sorted(set(la) & set(lb)):
        diffs = {}
        for v in values:
            if not _same(la[k].get(v), lb[k].get(v), tol):
                diffs[v] = {"legacy": la[k].get(v), "registry": lb[k].get(v)}
        if diffs:
            differing.append({"key": k, "diffs": diffs})
    return {"n_legacy": len(la), "n_registry": len(lb), "only_legacy": only_legacy, "only_registry": only_registry, "differing": differing,
            "duplicates_legacy": sorted(set(dup_a)), "duplicates_registry": sorted(set(dup_b)),
            "pass": not only_legacy and not only_registry and not differing}


def compare_figdata(a: Mapping[str, object], b: Mapping[str, object], tol: float = TOL, path: str = "") -> List[str]:
    """Leaf-wise diff of two figdata JSON objects; returns the differing paths (numbers to tol, NaN==NaN)."""
    out: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'legacy' if k in a else 'registry'}")
            else:
                out += compare_figdata(a[k], b[k], tol, f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out += compare_figdata(x, y, tol, f"{path}[{i}]")
    else:
        if not _same(a, b, tol):
            out.append(f"{path}: {a!r} vs {b!r}")
    return out


def compare_watermarks(measured: Mapping[str, Mapping[str, int]], marks: Mapping[str, object]) -> Dict[str, object]:
    """L1c: every watermark key must be met or exceeded by the measured counts — `lane_guard.shortfalls`' own rule over the
    file's shape ``{"sources": {source: {"keys": {"model|lane|component|dose": n, ...}, "rows": ..}}}`` (a flat
    ``{source: {key: n}}`` is accepted too)."""
    short = {}
    sources = marks.get("sources") if isinstance(marks, dict) and isinstance(marks.get("sources"), dict) else marks
    for src, spec in (sources or {}).items():
        if not isinstance(spec, dict):
            continue
        keys = spec.get("keys") if isinstance(spec.get("keys"), dict) else spec
        have = measured.get(src, {})
        for k, n in keys.items():
            if not isinstance(n, (int, float)) or isinstance(n, bool):
                continue
            got = int(have.get(k, 0))
            if got < int(n):
                short.setdefault(src, {})[k] = {"watermark": int(n), "measured": got}
    return {"shortfalls": short, "pass": not short, "n_sources": len(sources or {})}


# --------------------------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------------------------


@dataclass
class GateResult:
    gate_id: str
    paper_sha: str
    checkout: str
    registry_root: str
    registry_ingest_seq: int
    input_shas: Dict[str, str]
    levels: Dict[str, Dict[str, object]] = field(default_factory=dict)
    refused: Optional[str] = None
    report_path: Optional[str] = None
    record_paths: List[str] = field(default_factory=list)
    disagreements: List[Dict[str, object]] = field(default_factory=list)
    expected_breaks: List[str] = field(default_factory=list)

    def result_of(self, level: str) -> str:
        if self.refused:
            return "refused"
        lv = self.levels.get(level)
        if not lv or lv.get("skipped"):
            return "refused"
        return "pass" if lv.get("pass") else "fail"

    def summary(self) -> Dict[str, object]:
        return {"gate_id": self.gate_id, "paper_sha": self.paper_sha, "refused": self.refused, "levels": {k: self.result_of(k) for k in self.levels},
                "report": self.report_path, "records": self.record_paths}


def _rows(parquet_path: str) -> List[Dict[str, object]]:
    import pandas as pd
    if not os.path.exists(parquet_path):
        return []
    df = pd.read_parquet(parquet_path)
    return [{k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()} for r in df.to_dict("records")]


def check_paper_parity(checkout: str, pin: str, registry_root: str, round_id: Optional[str] = None, levels: Sequence[str] = LEVELS,
                       paper_python: Optional[str] = None, registry_python: Optional[str] = None, work_dir: Optional[str] = None,
                       report_dir: Optional[str] = None, between_sides=None, write_records: bool = True, now: Optional[str] = None,
                       skip_sha_check: bool = False) -> GateResult:
    """Run the gate. ``between_sides`` (tests) is called after the legacy side ran and before the registry side."""
    now = now or R.utc_now()
    legacy_roots()   # every guard root resolves on this host, or the gate refuses here, before either leg runs
    head = checkout_sha(checkout)
    if not skip_sha_check and not head.startswith(pin):
        raise ParityRefused(f"{checkout} is at {head}, not the pin {pin}: the gate reads the pinned commit")
    paper_python = paper_python or (DEFAULT_PAPER_PYTHON if os.path.exists(DEFAULT_PAPER_PYTHON) else sys.executable)
    if not os.path.exists(paper_python):
        raise ParityRefused(f"the paper venv's python is missing: {paper_python}")
    db_path = DB.db_path(registry_root)
    if not os.path.exists(db_path):
        raise ParityRefused(f"no registry database at {db_path}")
    with DB.reading(db_path) as conn:
        seq = conn.execute("SELECT COALESCE(MAX(ingest_seq), 0) FROM record_ingest").fetchone()[0]
        if round_id is None:
            row = conn.execute("SELECT round_id FROM paper_round WHERE closed_utc IS NULL ORDER BY opened_utc DESC LIMIT 1").fetchone()
            round_id = row[0] if row else None
    gate_id = f"{head[:8]}-{now.replace(':', '').replace('-', '').replace('.', '')[:15]}-{uuid.uuid4().hex[:6]}"
    work = work_dir or tempfile.mkdtemp(prefix=f"parity-{gate_id}-")
    res = GateResult(gate_id, head, os.path.abspath(checkout), os.path.abspath(registry_root), int(seq), {})
    want = [lv for lv in LEVELS if lv in levels]
    # legacy leg (its own inputs list is read from the pinned config, so the hashing happens AFTER it names them)
    legacy = run_legacy_side(checkout, os.path.join(work, "legacy"), paper_python, "L2" in want)
    inputs = list(legacy.get("inputs") or [])
    before = hash_inputs(inputs)
    res.input_shas = before
    columns = list(legacy["columns"])
    if between_sides:
        between_sides()
    reg = run_registry_side(res.registry_root, os.path.join(work, "registry"), columns, round_id, registry_python)
    after = hash_inputs(inputs)
    moved = sorted(k for k in before if before[k] != after.get(k))
    if moved:
        res.refused = f"input set moved between the legs: {moved} (before {[before[k][:12] for k in moved]}, after {[after.get(k, 'MISSING')[:12] for k in moved]})"
    res.disagreements = list(reg.get("disagreements") or [])
    for rid in EXPECTED_EXPIRED_RULINGS:
        res.expected_breaks.append(f"ruling {rid}: already expired on disk at the pin (D23) — the registry reporting it expired is expected")
    for t in EXPECTED_RED_TESTS:
        res.expected_breaks.append(f"pinned test {t}: known red at {head[:8]} (D24) — listed, never a parity failure")
    if reg.get("errors", {}).get("registry"):
        res.refused = res.refused or f"registry leg error: {reg['errors']['registry'][-1500:]}"
    if not res.refused:
        _compare_levels(res, legacy, reg, work, want)
    res.report_path = write_report(res, legacy, reg, report_dir or os.path.join(res.registry_root, "state", "parity", gate_id), work)
    if write_records:
        res.record_paths = write_gate_records(res, want, now)
    return res


def _compare_levels(res: GateResult, legacy: Mapping[str, object], reg: Mapping[str, object], work: str, want: Sequence[str]) -> None:
    reg_long = _rows(os.path.join(work, "registry", "reg_long.parquet"))
    reg_sel = _rows(os.path.join(work, "registry", "reg_current_keys.parquet"))
    if "L0" in want:
        per_reader = {}
        ok = True
        for name, n in (legacy.get("readers") or {}).items():
            rows = _rows(os.path.join(work, "legacy", f"l0_{name}.parquet")) if n else []
            rows = [r for r in rows if r.get("status") == "graded" or r.get("value") is not None]
            sources = {r.get("source") for r in rows}
            mine = [r for r in reg_long if r.get("source") in sources and r.get("value") is not None]
            cmp = compare_frames(rows, mine)
            per_reader[name] = cmp
            ok = ok and cmp["pass"]
        res.levels["L0"] = {"pass": ok and not legacy.get("errors", {}).get("l0"), "readers": per_reader, "errors": {k: v for k, v in (legacy.get("errors") or {}).items() if k.startswith("l0")}}
    if "L1" in want:
        assembled = [r for r in _rows(os.path.join(work, "legacy", "l1_assembled.parquet")) if r.get("value") is not None]
        cmp = compare_frames(assembled, [r for r in reg_long if r.get("value") is not None])
        res.levels["L1"] = {"pass": cmp["pass"] and "l1" not in (legacy.get("errors") or {}), "assembled": cmp, "attrs": legacy.get("l1_attrs"), "error": (legacy.get("errors") or {}).get("l1")}
    if "L1b" in want:
        keep = _rows(os.path.join(work, "legacy", "l1b_canonical_keys.parquet"))
        cmp = compare_frames(keep, reg_sel, values=("value",))
        res.levels["L1b"] = {"pass": cmp["pass"], "keys": cmp}
    if "L1c" in want:
        measured = legacy.get("l1c_measured_counts")
        marks = legacy.get("l1c_watermarks")
        if measured is None or marks is None:
            res.levels["L1c"] = {"pass": False, "skipped": True, "error": (legacy.get("errors") or {}).get("l1c", "no watermarks / measured counts from the legacy leg")}
        else:
            from .paper import LONG_COLUMN_SOURCES  # noqa: F401 (the registry side's counts use the same key)
            reg_counts: Dict[str, Dict[str, int]] = {}
            for r in reg_long:
                if r.get("value") is None or r.get("source") is None:
                    continue
                k = "|".join(_render(r.get(c)) for c in ("model", "lane_kind", "component", "dose"))
                reg_counts.setdefault(str(r["source"]), {})
                reg_counts[str(r["source"])][k] = reg_counts[str(r["source"])].get(k, 0) + 1
            res.levels["L1c"] = {"pass": compare_watermarks(reg_counts, marks)["pass"], "registry": compare_watermarks(reg_counts, marks),
                                 "legacy": compare_watermarks(measured, marks)}
    if "L2" in want:
        fdd_legacy = os.path.join(work, "legacy", "figdata")
        fdd_reg = os.path.join(work, "registry", "figdata")
        if not os.path.isdir(fdd_reg):
            res.levels["L2"] = {"pass": False, "skipped": True, "legacy_figures": legacy.get("l2_figdata"),
                                "error": "the registry leg of L2 needs the cutover branch's paperlib (registry_source.py, PR-8): only the legacy figdata was produced",
                                "legacy_errors": {k: v for k, v in (legacy.get("errors") or {}).items() if k.startswith("l2")}}
        else:
            res.levels["L2"] = compare_figdata_dirs(fdd_legacy, fdd_reg)


def compare_figdata_dirs(a_dir: str, b_dir: str) -> Dict[str, object]:
    names = sorted(set(os.listdir(a_dir)) | set(os.listdir(b_dir))) if os.path.isdir(a_dir) and os.path.isdir(b_dir) else []
    per = {}
    ok = bool(names)
    for n in names:
        pa, pb = os.path.join(a_dir, n), os.path.join(b_dir, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            per[n] = ["only on one side"]
            ok = False
            continue
        with open(pa) as fa, open(pb) as fb:
            d = compare_figdata(json.load(fa), json.load(fb))
        per[n] = d
        ok = ok and not d
    return {"pass": ok, "figures": per}


def _render(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    if isinstance(v, (int, float)):
        return f"{float(v):g}"
    return str(v)


def write_report(res: GateResult, legacy: Mapping[str, object], reg: Mapping[str, object], report_dir: str, work: str) -> str:
    os.makedirs(report_dir, exist_ok=True)
    lines = [f"# Paper parity gate {res.gate_id}", "", f"paper sha `{res.paper_sha}` at `{res.checkout}`; registry `{res.registry_root}` head ingest_seq {res.registry_ingest_seq}", ""]
    if res.refused:
        lines += ["## REFUSED", "", res.refused, ""]
    lines += ["## Disagreements first (fixed decision 11: the build's value is kept, `inferred = 1`)", ""]
    seen: Dict[str, int] = {}
    for d in res.disagreements:
        seen[str(d.get("d"))] = seen.get(str(d.get("d")), 0) + 1
    if seen:
        for text, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            lines.append(f"- [{n} entries] {text}")
    else:
        lines.append("- (no roster entries with a disagreement in this registry)")
    lines += ["", "## Expected breaks (never a parity failure)", ""] + [f"- {b}" for b in res.expected_breaks]
    exp_rulings = {r["ruling_id"]: r for r in (reg.get("rulings") or [])}
    for rid in EXPECTED_EXPIRED_RULINGS:
        if rid in exp_rulings:
            r = exp_rulings[rid]
            lines.append(f"  - registry reads {rid}: evidence_state={r.get('evidence_state')} expired_utc={r.get('expired_utc')}")
    lines += ["", "## Input set (sha256 before the legacy leg == after the registry leg)", ""]
    for k, v in sorted(res.input_shas.items()):
        lines.append(f"- `{k}` {v[:16]}")
    for lv in LEVELS:
        if lv not in res.levels:
            continue
        d = res.levels[lv]
        lines += ["", f"## {lv}: {'PASS' if d.get('pass') else ('SKIPPED' if d.get('skipped') else 'FAIL')}", ""]
        if d.get("error"):
            lines.append(f"error: {d['error']}")
        if lv == "L0":
            for name, cmp in (d.get("readers") or {}).items():
                lines.append(f"- {name}: legacy {cmp['n_legacy']} keys, registry {cmp['n_registry']}; only legacy {len(cmp['only_legacy'])}, only registry {len(cmp['only_registry'])}, differing {len(cmp['differing'])}")
                for k in cmp["only_legacy"][:20]:
                    lines.append(f"    - only legacy: `{k}`")
                for k in cmp["only_registry"][:20]:
                    lines.append(f"    - only registry: `{k}`")
                for x in cmp["differing"][:20]:
                    lines.append(f"    - differs: `{x['key']}` {json.dumps(x['diffs'], default=str)}")
            for k, v in (d.get("errors") or {}).items():
                lines.append(f"- reader error {k}: {v}")
        elif lv in ("L1", "L1b"):
            cmp = d.get("assembled") or d.get("keys") or {}
            lines.append(f"- legacy {cmp.get('n_legacy')} keys, registry {cmp.get('n_registry')}; only legacy {len(cmp.get('only_legacy', []))}, only registry {len(cmp.get('only_registry', []))}, differing {len(cmp.get('differing', []))}")
            for k in cmp.get("only_legacy", [])[:30]:
                lines.append(f"    - only legacy: `{k}`")
            for k in cmp.get("only_registry", [])[:30]:
                lines.append(f"    - only registry: `{k}`")
            for x in cmp.get("differing", [])[:30]:
                lines.append(f"    - differs: `{x['key']}` {json.dumps(x['diffs'], default=str)}")
        elif lv == "L1c":
            for side in ("registry", "legacy"):
                w = d.get(side) or {}
                lines.append(f"- {side}: {'no shortfall' if w.get('pass') else json.dumps(w.get('shortfalls'), default=str)[:3000]}")
        elif lv == "L2":
            for n, diffs in (d.get("figures") or {}).items():
                lines.append(f"- {n}: {'equal' if not diffs else str(len(diffs)) + ' differences'}")
                for x in diffs[:10]:
                    lines.append(f"    - {x}")
            if d.get("legacy_figures"):
                lines.append(f"- legacy figdata produced: {len(d['legacy_figures'])} figures")
    lines += ["", f"work directory: `{work}`", ""]
    path = os.path.join(report_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(report_dir, "levels.json"), "w", encoding="utf-8") as fh:
        json.dump({"gate_id": res.gate_id, "paper_sha": res.paper_sha, "refused": res.refused, "input_shas": res.input_shas, "levels": res.levels,
                   "disagreements": res.disagreements, "expected_breaks": res.expected_breaks, "legacy_errors": legacy.get("errors"), "registry_errors": reg.get("errors")},
                  fh, indent=1, default=str)
    for sub in ("legacy", "registry"):
        fdd = os.path.join(work, sub, "figdata")
        if os.path.isdir(fdd):
            shutil.copytree(fdd, os.path.join(report_dir, f"figdata_{sub}"), dirs_exist_ok=True)
    return path


def write_gate_records(res: GateResult, want: Sequence[str], now: str) -> List[str]:
    from .harvest import write_harvester_record
    paths = []
    for lv in want:
        payload = {"gate_id": f"{res.gate_id}:{lv}", "paper_sha": res.paper_sha, "paper_checkout_uri": res.checkout, "input_shas_json": res.input_shas,
                   "registry_ingest_seq": res.registry_ingest_seq, "level": lv, "result": res.result_of(lv), "report_uri": res.report_path, "run_utc": now,
                   "extra": {"refused": res.refused, "n_disagreements": len(res.disagreements), "expected_breaks": res.expected_breaks}}
        paths.append(write_harvester_record(res.registry_root, "parity_gate", payload, now, subdir="parity"))
    return paths
