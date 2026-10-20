"""sisteer.front_door_views -- the front-door models table's view classification for code that holds only a TAG.

Two callers hold an audit id or a legacy view id (``petri_full22_<key>_<view>`` / ``<key>_<view>``) and no target: the Petri driver's
``steered_wave_audit`` (scripts/petri_run.py) and the registry contrasts emitter's ``u0_view_of`` (scripts/registry/emit_contrasts.py). Their
s-family name grammar (``_s<N>r`` / ``_s<N>rl``) is kept; this module adds the TARGET-AWARE reading for the front-door DOSE arms: a view of a
FRONT-DOOR model that spells a dose (``d<dose>``, ``d<dose>_relu``, ``d<dose>_relul``, ``dm<dose>…``) measures no unsteered leg -- the same
reading reproduce/lib/models.steered_wave_view and f22_roster.steered_wave_view make with a target in hand.

ONE source of truth: ``reproduce/models.yaml`` through ``reproduce/lib/models.py``, loaded REPO-relatively via importlib the way
reproduce/lib/stage.py loads it (never a second key table, never a sys.path insert of reproduce/lib into the caller). A checkout without
``reproduce/lib/models.py``, or without the table it reads, is REFUSED BY NAME (:class:`FrontDoorTableMissing`) for any dose-shaped tag -- never a
silent "not a front-door view". The table is read ONCE per revision (reader path, table path and both mtimes; a stat per call) and NEVER for a tag
that is not dose-shaped: the grading daemon's contrasts tick classifies every complete steered run (emit_contrasts.contrasts_due), so a per-call
reparse (0.7-1.9 s each) would have stalled it by minutes to hours.

Deliberately NOT a front-door dose view here: a PAPER target's dose-named roster views (qwen35_4b / llama8b / gemma4e4b / nemotron3nano4b /
glm46vflash: d0p1, d0p25, d0p5, d1p5) -- same-wave none-anchored, the collision a name-only grammar would have hit; ``u0``; and ``s1`` (its audit
keeps the legacy reading). The key is parsed longest-first so a key carrying underscores of its own splits correctly.

``phase_views`` / ``is_phase_view`` (demo-import M3) are thin delegates of the reader's ONE classification rule — the grammar's views plus a
row's declared ``wave`` (the worked example's ``battery``) — for the grading daemon's padded-run split, cached with the keys.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
MODELS_PY = REPO / "reproduce" / "lib" / "models.py"
DEFAULT_TABLE = REPO / "reproduce" / "models.yaml"      #: reproduce/lib/models.DEFAULT_TABLE's path, resolved from this tree (SISTEER_MODELS_TABLE overrides it there and here)
AUDIT_PREFIX = "petri_full22_"
#: the dose alternatives of reproduce/lib/models.PHASE_VIEW_RE (d<dose>[_relu|_relul], dm<dose>[_relu|_relul]) -- the s-family and u0 are not dose views
DOSE_VIEW_RE = re.compile(r"^(d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")
#: the same shape as a tag's LAST view segment -- the name-only pre-filter that spares the table a read for every tag that cannot be a dose view
DOSE_TAIL_RE = re.compile(r"(?:^|_)(d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")
_CACHE: Dict[Tuple[str, str, int, int], Tuple[object, Tuple[str, ...]]] = {}


class FrontDoorTableMissing(RuntimeError):
    """reproduce/lib/models.py (or the models table it reads) is absent from this checkout: no classification is made."""


def models_module(path: Optional[Path] = None):
    """reproduce/lib/models.py loaded by path (importlib), REFUSED BY NAME when absent. Uncached: callers go through :func:`front_door_keys`."""
    p = Path(path) if path is not None else MODELS_PY
    if not p.is_file():
        raise FrontDoorTableMissing(f"{p}: reproduce/lib/models.py is absent -- this checkout carries no front-door table reader; "
                                    "the steered-view classification of a dose tag is refused, never read as 'not a front-door view'")
    spec = importlib.util.spec_from_file_location("sisteer_front_door_models_table", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def table_path(table_path: "Optional[str | Path]" = None) -> Path:
    """The models table the reader would open: the argument, else ``SISTEER_MODELS_TABLE``, else reproduce/models.yaml of this tree."""
    return Path(table_path) if table_path else Path(os.environ.get("SISTEER_MODELS_TABLE") or DEFAULT_TABLE)


def front_door_keys(table_path_: "Optional[str | Path]" = None) -> List[str]:
    """The front-door model keys (reproduce/lib/models.front_door_keys: every table model naming a phase view), read ONCE per revision of the
    reader and the table (both paths + both mtimes key the cache; a stat per call, a parse per change). A missing reader or table refuses by name."""
    if not MODELS_PY.is_file():
        raise FrontDoorTableMissing(f"{MODELS_PY}: reproduce/lib/models.py is absent -- this checkout carries no front-door table reader; "
                                    "the steered-view classification of a dose tag is refused, never read as 'not a front-door view'")
    tp = table_path(table_path_)
    if not tp.is_file():
        raise FrontDoorTableMissing(f"{tp}: the models table is absent -- the front-door keys cannot be read; the classification of a dose tag is refused")
    key = (str(MODELS_PY), str(tp), MODELS_PY.stat().st_mtime_ns, tp.stat().st_mtime_ns)
    hit = _CACHE.get(key)
    if hit is None:
        mod = models_module(MODELS_PY)
        keys = tuple(str(k) for k in mod.front_door_keys(str(tp)))
        # the ONE table read per revision also answers the single-phase question (models.single_phase: a front-door key with no u0 view)
        # and each key's PHASE VIEWS (models.phase_views: the grammar's plus the row's declared wave — demo-import M3), so no classification
        # below re-parses the table; the baseline views are filled lazily into the same entry
        single = frozenset(k for k in keys if mod.single_phase(k, str(tp)))
        phases = {k: tuple(str(v) for v in mod.phase_views(k, str(tp))) for k in keys}
        _CACHE.clear()
        _CACHE[key] = hit = (mod, keys, single, {}, phases)
    return list(hit[1])


def _hit(table_path_: "Optional[str | Path]" = None) -> tuple:
    """The cache entry for the current revision of the reader and the table: ``(module, front-door keys, single-phase keys, baselines,
    phase views per key)``."""
    front_door_keys(table_path_)                       # the reader is loaded (and a missing table refused) exactly as for every other classification
    return next(iter(_CACHE.values()))


def phase_views(key: str, table_path_: "Optional[str | Path]" = None) -> List[str]:
    """reproduce/lib/models.phase_views(key): the PHASE views of a front-door key — the grammar's plus the row's declared ``wave`` (demo-import
    M3; the ONE rule lives in the reader, this is its cached delegate for the tag-holding callers) — () for a key the table lacks."""
    _, _, _, _, phases = _hit(table_path_)
    return list(phases.get(str(key), ()))


def is_phase_view(key: str, view: str, table_path_: "Optional[str | Path]" = None) -> bool:
    """reproduce/lib/models.is_phase_view(key, view): a name of the phase-view grammar (for every key), or the declared wave of a key the
    table carries (its cached phase views hold the grammar's views of the row plus the declared wave, so membership there covers it) — from
    the cached read, never a regex of this module's own, and never narrower than the grammar."""
    mod, _keys, _, _, phases = _hit(table_path_)
    k, v = str(key), str(view)
    return bool(mod.PHASE_VIEW_RE.match(v)) or v in phases.get(k, ())


def steered_wave_view(view: str, key: str, table_path_: "Optional[str | Path]" = None) -> bool:
    """reproduce/lib/models.steered_wave_view(view, key)'s rule — s<N>r(l) by name; every non-u0 phase view of a PHASED front-door key, s1
    included; never a single-phase key's view — answered from the cached table read (the roster's own rule re-reads the table on each call;
    the agreement tests hold the two equal over every roster arm). The two tag-holding callers use :func:`front_door_dose_tag`."""
    mod, keys, single, _, _ = _hit(table_path_)
    v, k = str(view), str(key)
    if mod.STEERED_WAVE_VIEW_RE.match(v):
        return True
    if k not in keys or k in single:
        return False
    return bool(mod.PHASE_VIEW_RE.match(v)) and v != "u0"


def front_door_dose_view(view: str, key: str, table_path_: "Optional[str | Path]" = None) -> bool:
    """True when ``key`` is a FRONT-DOOR model, ``view`` spells a dose view (:data:`DOSE_VIEW_RE`, never u0) AND the roster's rule reads that
    view as a steered wave (:func:`steered_wave_view`: a dose view of a PHASED key, the one that declares u0) -- the reading the two
    suffix copies add to their s-family grammar. A SINGLE-PHASE key's dose views (reproduce/lib/models.single_phase) carry
    their own ``none`` leg and borrow no anchor, so they are NOT front-door dose tags: the audit drives them like its one wave and the
    contrasts emitter anchors on their own unsteered run. The agreement with the roster's rule is pinned by
    tests/test_front_door_views_dose_audit.py over every roster dose arm, not re-derived per call."""
    v, k = str(view), str(key)
    if v == "u0" or not DOSE_VIEW_RE.match(v):
        return False
    _, keys, single, _, _ = _hit(table_path_)
    return k in keys and k not in single                # = the roster's steered-wave rule for a dose-shaped view, from the one cached read


def baseline_view(key: str, table_path_: "Optional[str | Path]" = None) -> str:
    """reproduce/lib/models.baseline_view(key): the wave a front-door key's steered views borrow their unsteered anchor from — ``u0`` for a
    phased key, the one wave of a single-phase key — read once per revision of the table (cached beside the keys; a missing table refuses)."""
    mod, _, _, baselines, _ = _hit(table_path_)
    k = str(key)
    if k not in baselines:
        baselines[k] = str(mod.baseline_view(k, str(table_path(table_path_))))
    return baselines[k]


def split_key_view(tag: str, table_path_: "Optional[str | Path]" = None) -> Optional[Tuple[str, str]]:
    """``petri_full22_<key>_<view>`` / ``<key>_<view>`` -> (key, view) for a KNOWN front-door key (longest key first, so
    a key with underscores of its own splits at the key, never inside it); None for anything else (a paper key, an unknown key, a bare view). Reads the table."""
    t = str(tag or "")
    if t.startswith(AUDIT_PREFIX):
        t = t[len(AUDIT_PREFIX):]
    for key in sorted(front_door_keys(table_path_), key=len, reverse=True):
        if t.startswith(key + "_") and len(t) > len(key) + 1:
            return key, t[len(key) + 1:]
    return None


def front_door_dose_tag(tag: Optional[str], table_path_: "Optional[str | Path]" = None) -> Optional[str]:
    """The front-door KEY when ``tag`` (``petri_full22_<key>_<view>`` / ``<key>_<view>``) names a FRONT-DOOR dose view, else None -- the one entry
    the two suffix copies call after their s-family clause. The dose shape is checked by NAME first, so an s-family, u0, paper-view or legacy tag
    never opens the table; a dose-shaped tag then needs it, and a missing reader or table refuses by name (never a silent None)."""
    t = str(tag or "")
    if not DOSE_TAIL_RE.search(t):
        return None
    kv = split_key_view(t, table_path_)
    if kv and front_door_dose_view(kv[1], kv[0], table_path_):
        return kv[0]
    return None
