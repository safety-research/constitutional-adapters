#!/usr/bin/env python3
"""Shared helpers of the capeval analysis scripts (``bcb_bfcl_200_analysis``
and any sibling that renders a results doc + figures from a pulled run dir):
the refusal type, JSON / JSONL readers, the Wilson interval and ``pass_rate``,
the exact McNemar test, the ``modal volume ls`` listing, and the house figure
conventions — the dataviz reference palette and the header / layout rule
(title ≤ 2 lines, subtitle ≤ 3, axes ≥ 80 % of the height) that
``finish_figure`` enforces so every figure inherits it.

Lifted verbatim from the Terminal-Bench pilot analysis when that harness was
retired from ``main`` (one canonical eval battery; obsolete harnesses off
``main`` so they cannot be picked by accident) — the BCB / BFCL-200 analysis
imported these from it. Pure: no Modal SDK, no W&B, no network
(``modal_volume_ls`` shells out to the modal CLI the caller names). Loaded as
a sibling script (``_load_sibling("capeval_analysis_common")``), never
installed.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# figures (dataviz reference palette, light surface)
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
SEQ = {100: "#cde2fb", 250: "#86b6ef", 450: "#2a78d6", 600: "#184f95"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
DPI = 200
TITLE_MAX_LINES, TITLE_WIDTH = 2, 90
SUBTITLE_MAX_LINES = 3
AXES_MIN_FRACTION = 0.80
#: header geometry in INCHES (a 5-in figure: title top at 0.985, subtitle 0.037 below per title line —
#: the same absolute header on a 23-in per-task figure instead of a stretched one)
HEADER_TOP_PAD_IN = 0.075
TITLE_LINE_IN = 0.183          # 11 pt × 1.2 leading
SUBTITLE_LINE_IN = 0.13        # 7.4 pt × 1.25 leading
HEADER_GAP_IN = 0.10           # air between the subtitle's last line and the axes top


class AnalysisError(RuntimeError):
    """A refusal of this script: a run dir that lacks what the analysis needs,
    a figure that breaks a layout rule, a stage that cannot run in this venv."""

# --------------------------------------------------------------------------- #
# small helpers                                                                #
# --------------------------------------------------------------------------- #


def read_json(path: Union[str, Path]) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise AnalysisError(f"{p}: unreadable: {e}") from e
    except ValueError as e:
        raise AnalysisError(f"{p}: not JSON: {e}") from e


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Every non-blank line as an object; a bad line names its number."""
    p = Path(path)
    out: List[Dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise AnalysisError(f"{p}: unreadable: {e}") from e
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            raise AnalysisError(f"{p}:{i}: not JSON: {e}") from e
        if not isinstance(row, Mapping):
            raise AnalysisError(f"{p}:{i}: row is a {type(row).__name__}, not an object")
        out.append(dict(row))
    return out


def summary_stats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """n / min / median / mean / p90 / max over the non-null values (the rest counted)."""
    xs = sorted(float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    n_null = len(values) - len(xs)
    if not xs:
        return {"n": 0, "n_null": n_null, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    p90 = xs[min(len(xs) - 1, int(math.ceil(0.9 * len(xs))) - 1)]
    return {"n": len(xs), "n_null": n_null, "min": xs[0], "median": statistics.median(xs),
            "mean": statistics.fmean(xs), "p90": p90, "max": xs[-1]}


def wilson_interval(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """The Wilson score interval for ``k`` successes in ``n`` trials (95 % at
    the default ``z``), clipped to [0, 1]."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise AnalysisError(f"n must be a positive int, got {n!r}")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0 or k > n:
        raise AnalysisError(f"k must be an int in 0..{n}, got {k!r}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pass_rate(k: int, n: int) -> Dict[str, Any]:
    """``k`` passes over ``n`` with the Wilson 95 % interval (``rate`` None at n = 0)."""
    if n <= 0:
        return {"k": int(k), "n": int(n), "rate": None, "wilson_low": None, "wilson_high": None}
    lo, hi = wilson_interval(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": k / n, "wilson_low": lo, "wilson_high": hi}


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p on the discordant pairs: ``b`` (A pass, B fail)
    and ``c`` (A fail, B pass) under H0 Binomial(b + c, ½); 1.0 with no
    discordant pair."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def modal_volume_ls(modal_python: str, volume: str, remote: str) -> List[str]:
    """The names listed at ``remote`` (one level) — the verification listing."""
    argv = [modal_python, "-m", "modal", "volume", "ls", volume, remote]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AnalysisError(f"modal volume ls failed (rc {proc.returncode}): {proc.stderr.strip()[-800:]}")
    names = []
    for line in proc.stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("Directory listing") or set(s) <= set("─│┌┐└┘├┤┬┴┼ "):
            continue
        names.append(s.rstrip("/").rsplit("/", 1)[-1])
    return names


# --------------------------------------------------------------------------- #
# figures: the house header and the layout rule                                #
# --------------------------------------------------------------------------- #


def wrap_title(text: str, *, width: int = TITLE_WIDTH, max_lines: int = TITLE_MAX_LINES, what: str = "title") -> str:
    """Hard-wrap at ``width``; refuse more than ``max_lines`` (a title that
    needs a third line is making more than one point)."""
    lines: List[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    if len(lines) > max_lines:
        raise AnalysisError(f"{what} needs {len(lines)} lines (> {max_lines}): {text[:80]!r}…")
    return "\n".join(lines)


def finish_figure(fig, title: str, subtitle: Optional[str] = None, *, tile: bool = False) -> None:
    """The house header (title ≤ 2 lines, subtitle ≤ 3) and the layout rule:
    the axes' union must keep ≥ 80 % of the figure height (``tile`` figures —
    stat tiles with no plot — are exempt). Raises on a violation so the rule is
    inherited by every figure, not remembered."""
    fig_w, fig_h = (float(v) for v in fig.get_size_inches())
    # wrap widths follow the figure width (11 pt bold ≈ 10.5 chars/in; 7.4 pt ≈ 17 chars/in at 0.93 usable width);
    # vertical offsets are INCHES converted per figure, so a tall figure keeps the same header, not a stretched one
    t = wrap_title(title, width=min(TITLE_WIDTH, int(fig_w * 10.5)))
    title_y = 1.0 - HEADER_TOP_PAD_IN / fig_h
    fig.text(0.035, title_y, t, fontsize=11.0, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        s = wrap_title(subtitle, width=int(fig_w * 17), max_lines=SUBTITLE_MAX_LINES, what="subtitle")
        n_title = t.count("\n") + 1
        fig.text(0.035, title_y - n_title * TITLE_LINE_IN / fig_h, s, fontsize=7.4, color=INK2, ha="left", va="top",
                 linespacing=1.25)
    if tile:
        return
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if not axes:
        raise AnalysisError("figure has no axes and is not marked tile=True")
    y0 = min(ax.get_position().y0 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    frac = y1 - y0
    if frac < AXES_MIN_FRACTION - 1e-9:
        raise AnalysisError(f"axes keep {frac:.3f} of the figure height (< {AXES_MIN_FRACTION}) — {title[:60]!r}")


def save_figure(fig, figs_dir: Path, stem: str) -> List[str]:
    """PNG + SVG under ``stem``; the figure is closed afterwards."""
    import matplotlib.pyplot as plt
    figs_dir.mkdir(parents=True, exist_ok=True)
    png, svg = figs_dir / f"{stem}.png", figs_dir / f"{stem}.svg"
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg)
    plt.close(fig)
    return [str(png), str(svg)]
