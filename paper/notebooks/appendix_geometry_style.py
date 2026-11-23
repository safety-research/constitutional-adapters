"""appendix_geometry_style: the drawing style of the two weight-geometry figures the paper prints in the control-directions appendix, the
refusal enrichment (Figure 9, bottom; appendix_geometry_forward.refusal_enrichment_figure) and the overlap figure (Figure 10;
appendix_geometry_overlap.overlap_figure), so that the notebook renders the manuscript's copies of them: a PNG this notebook writes
decodes to the manuscript PNG's pixels (paper/tests/test_appendix_geometry_style.py pins both).

    import appendix_geometry_style as GS
    with GS.style():                                   # matplotlib's defaults plus the figures' rcParams, for drawing and saving
        fig = plt.figure(figsize=(GS.TEXTWIDTH_IN, height_in))
        ...                                            # axes placed in inches with fig.add_axes, no layout engine
        GS.save(fig, name)                             # pdf, png (300 dpi) and svg at the canvas size, into R.FIGS and the mirrors

THE CANVAS is drawn at the width it prints at in the manuscript's units (5.5 in, the text width the figures are included at, scaled
alike) with every axis, legend and panel letter placed in inches, so the printed font sizes are the ones set here (7 pt text, 6.3 pt
ticks and legends, 9 pt bold panel letters, DejaVu Sans with its mathtext). The figures are saved at that canvas exactly, never with a
tight bounding box: a PDF, a PNG at 300 dpi and an SVG with its text kept as text.

THE PALETTE keys colour by OBJECT, never by family: theta_K blue (#2a78d6), theta_C green (#008300), the safety positive control yellow
(#eda100), a pair of the two objects neutral ink; the hues are the paper's (control blue, constitution green), snapped to a palette step
that keeps the three apart for the common colour-vision deficiencies. Family is carried by marker SHAPE (circle ReLU steer, square
LoRA). A grey band is a random-direction or random-subspace reference, drawn one step off the white surface.

The style is self-contained: `style()` starts from matplotlib's own defaults, so neither the notebook's publication style (`R.style()`)
nor the inline backend's settings reach these two figures, and the caller's rcParams come back when the block ends.
"""
from __future__ import annotations

import contextlib
import os
import shutil

import matplotlib as mpl
from matplotlib.lines import Line2D

import registry_nb as R
import appendix_common as AC

# ---- palette -----------------------------------------------------------------------------------------------------------------------
BLUE, GREEN, YELLOW = "#2a78d6", "#008300", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BAND = "#ecebe6"                       # random references, one step off the surface
SURFACE = "#ffffff"
OBJECT_COLOR = {"K": BLUE, "C": GREEN, "S": YELLOW}
FAMILY_MARKER = {"relu": "o", "lora": "s"}

# ---- sizes (points at print size, inches for the canvas) ---------------------------------------------------------------------------
TEXTWIDTH_IN = 5.5
FS, FS_SMALL, FS_LETTER = 7.0, 6.3, 9.0
MS, MS_SEED = 4.6, 2.6                 # marker sizes: a seed mean, one seed
LW, LW_THIN, LW_HAIR = 1.1, 0.7, 0.5
DPI = 300                              # the PNG's resolution
SAVE_ORDER = ("pdf", "png", "svg")     # the order the manuscript's copies were written in
assert sorted(SAVE_ORDER) == sorted(AC.FORMATS)

RC = {
    "font.family": "DejaVu Sans", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
    "legend.fontsize": FS_SMALL, "mathtext.fontset": "dejavusans",
    "axes.edgecolor": BASELINE, "axes.linewidth": LW_HAIR, "axes.labelcolor": INK2, "axes.titlesize": FS,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": LW_HAIR, "ytick.major.width": LW_HAIR,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": LW_HAIR, "grid.linestyle": "-",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.1,
    "legend.borderaxespad": 0.2, "legend.handlelength": 1.4,
}


class CanvasError(ValueError):
    pass


@contextlib.contextmanager
def style():
    """matplotlib's own defaults plus `RC` for the length of the block (drawing AND saving happen inside it: tick marks and labels take
    their sizes and colours when they are first drawn); the caller's rcParams, the notebook's style included, are restored after."""
    with mpl.rc_context():
        mpl.rcdefaults()                                # leaves the backend alone (matplotlib's style blacklist)
        mpl.rcParams.update(RC)
        yield


# ---- marks, handles, labels --------------------------------------------------------------------------------------------------------
def grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=LW_HAIR, zorder=0)
    ax.set_axisbelow(True)


def facet(ax, s: str, x: float = 0.02, y: float = 0.985, ha: str = "left") -> None:
    """A small label inside or just above the axes, top-left (not a title): which slice the sub-axis shows."""
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va="top", fontsize=FS_SMALL, color=INK2, zorder=10)


def marker_kw(obj: str, family: str, ms: float = MS, seed: bool = False) -> dict:
    """Line2D keywords of one filled mark of object `obj` ('C', 'K', 'S') in `family`'s shape: a seed mean, or one seed (`seed`)."""
    return dict(marker=FAMILY_MARKER[family], ms=ms, ls="none", mfc=OBJECT_COLOR[obj], mec=SURFACE, mew=0.3 if seed else 0.6)


def object_handle(obj: str, family: str, label: str) -> Line2D:
    return Line2D([], [], label=label, **marker_kw(obj, family))


def shape_handle(family: str, label: str) -> Line2D:
    return Line2D([], [], marker=FAMILY_MARKER[family], ls="none", ms=MS, mfc=INK2, mec=SURFACE, mew=0.5, label=label)


def panel_letter(fig, x: float, y: float, s: str) -> None:
    """Bold panel letter with its top-left corner at (x, y) in figure coordinates."""
    fig.text(x, y, s, fontsize=FS_LETTER, fontweight="bold", color=INK, ha="left", va="top")


# ---- the canvas check and saving ---------------------------------------------------------------------------------------------------
def content_box_in(fig) -> tuple:
    """The union of every artist's rendered box, in inches from the canvas's bottom-left (x0, y0, x1, y1)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.x0, bb.y0, bb.x1, bb.y1


def check_canvas(fig, name: str) -> tuple:
    """Refuses a figure with anything drawn outside its canvas (the figures are saved at the canvas, so it would be clipped)."""
    w, h = fig.get_size_inches()
    x0, y0, x1, y1 = content_box_in(fig)
    if x0 < 0 or y0 < 0 or x1 > w + 1e-6 or y1 > h + 1e-6:
        raise CanvasError(f"{name}: content box x {x0:.3f}-{x1:.3f}, y {y0:.3f}-{y1:.3f} in leaves the {w} x {h} in canvas")
    return x0, y0, x1, y1


def save(fig, name: str, dirs=None, extra_dirs=()) -> list:
    """Write `fig` as `<name>.pdf`, `.png` (300 dpi) and `.svg` at the canvas size (no tight bounding box) into the first of `dirs`
    (default: the notebook's `R.FIGS`, read at call time), then copy the three files to the other `dirs`, each APPENDIX_FIGS_MIRROR
    directory and `extra_dirs`; call it inside `style()`. Returns the paths written, `appendix_common.save_figure`'s order of targets."""
    if not name or os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(f"figure name {name!r}: a bare file stem, no directory")
    targets = []
    for d in (list(dirs) if dirs is not None else [R.FIGS]) + AC.mirror_dirs() + list(extra_dirs):
        if d not in targets:
            targets.append(d)
    check_canvas(fig, name)
    first, out = targets[0], []
    os.makedirs(first, exist_ok=True)
    for ext in SAVE_ORDER:
        p = os.path.join(first, f"{name}.{ext}")
        fig.savefig(p, **({"dpi": DPI} if ext == "png" else {}))
        out.append(p)
    for d in targets[1:]:
        os.makedirs(d, exist_ok=True)
        for ext in SAVE_ORDER:
            q = os.path.join(d, f"{name}.{ext}")
            shutil.copyfile(os.path.join(first, f"{name}.{ext}"), q)
            out.append(q)
    return out
