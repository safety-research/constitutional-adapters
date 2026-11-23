"""appendix_geometry_style: the two weight-geometry figures the paper prints from the control-directions appendix (Figure 9 bottom,
figA_refusal_enrichment; Figure 10, figA_cK_overlap) render as the manuscript's copies. No registry: the committed sidecars.

Pinned here:
  * a PNG the notebook writes decodes to the manuscript PNG's pixels: the sha256 of the decoded RGBA array equals the one taken from the
    manuscript's copy (1650 x 687 and 1650 x 1332 pixels, 300 dpi, the canvas drawn at 5.5 in wide), under the matplotlib release the
    manuscript's copies were drawn with (the bundled FreeType and DejaVu fonts are a release's own, so another release may move a
    glyph by a pixel; the test says so and skips rather than compare across releases);
  * each figure is saved at its canvas (no tight bounding box) as pdf, png and svg, the png at 300 dpi, and the mirrors get copies of
    the same files;
  * `style()` draws from matplotlib's own defaults, whatever style the caller set (the notebook's `R.style()` included), and hands the
    caller's rcParams back when the block ends; a figure with anything outside its canvas is refused by name."""
import hashlib
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402  (a matplotlib dependency)

import registry_nb as R  # noqa: E402
import appendix_common as AC  # noqa: E402
import appendix_geometry_forward as GF  # noqa: E402
import appendix_geometry_overlap as GO  # noqa: E402
import appendix_geometry_style as GS  # noqa: E402

# the manuscript's PNGs: (width, height) in pixels and the sha256 of their decoded RGBA arrays (numpy uint8, rows top to bottom)
MANUSCRIPT_PNG = {
    "figA_refusal_enrichment": ((1650, 687), "fa08046af30066f70ec12793b3bca58c8d66b840927dcedb6c8612241d10c80b"),
    "figA_cK_overlap": ((1650, 1332), "d1224b91bb7dd1f4f7883068547c6197ccc074026de07ef157b3f365d1dc072a"),
}
MANUSCRIPT_MATPLOTLIB = "3.10.0"


@pytest.fixture(autouse=True)
def _no_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIGS", str(tmp_path / "figs"))
    monkeypatch.delenv(AC.MIRROR_ENV, raising=False)
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    out = tmp_path_factory.mktemp("geomstyle")
    with plt.rc_context():
        R.style()                                            # the notebook's own style is in force when the cells call these
        GF.refusal_enrichment_figure(GF.load_geometry_forward(), out_dirs=[str(out)])
        GO.overlap_figure(GO.load_geometry_overlap(), out_dirs=[str(out)])
    yield out
    plt.close("all")


def _decoded(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


@pytest.mark.parametrize("name", sorted(MANUSCRIPT_PNG))
def test_the_png_is_the_manuscripts_canvas(rendered, name):
    size, _ = MANUSCRIPT_PNG[name]
    im = Image.open(rendered / f"{name}.png")
    assert im.size == size and im.mode == "RGBA"
    assert all(abs(d - GS.DPI) < 0.01 for d in im.info["dpi"])


@pytest.mark.parametrize("name", sorted(MANUSCRIPT_PNG))
def test_the_png_decodes_to_the_manuscripts_pixels(rendered, name):
    if matplotlib.__version__ != MANUSCRIPT_MATPLOTLIB:
        pytest.skip(f"matplotlib {matplotlib.__version__}: the manuscript's copies were drawn with {MANUSCRIPT_MATPLOTLIB}, whose "
                    "bundled FreeType and fonts set the glyph pixels; pixel identity is defined for that release only")
    a = _decoded(rendered / f"{name}.png")
    assert hashlib.sha256(a.tobytes()).hexdigest() == MANUSCRIPT_PNG[name][1], name


def test_each_figure_saves_pdf_png_and_svg_and_copies_them_to_every_mirror(tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    monkeypatch.setenv(AC.MIRROR_ENV, str(mirror))
    GF.refusal_enrichment_figure(GF.load_geometry_forward())
    GO.overlap_figure(GO.load_geometry_overlap())
    for name in (GF.FIG_ENRICHMENT, GO.FIGURE):
        for ext in AC.FORMATS:
            a, b = pathlib.Path(R.FIGS) / f"{name}.{ext}", mirror / f"{name}.{ext}"
            assert a.is_file() and a.read_bytes() == b.read_bytes(), (name, ext)


def test_style_starts_from_matplotlibs_defaults_and_restores_the_callers():
    from matplotlib.style.core import STYLE_BLACKLIST
    with plt.rc_context():
        R.style()
        before = dict(matplotlib.rcParams)
        with GS.style():
            now = dict(matplotlib.rcParams)
            assert {k: now[k] for k in GS.RC} == {k: matplotlib.rcParams.validate[k](v) for k, v in GS.RC.items()}
            rest = set(now) - set(GS.RC) - set(STYLE_BLACKLIST)
            assert {k: now[k] for k in rest} == {k: matplotlib.rcParamsDefault[k] for k in rest}
        assert dict(matplotlib.rcParams) == before


def test_a_figure_with_content_outside_its_canvas_is_refused(tmp_path):
    with GS.style():
        fig = plt.figure(figsize=(2.0, 1.0))
        fig.text(1.2, 0.5, "past the right edge")
        with pytest.raises(GS.CanvasError, match="leaves the 2.0 x 1.0 in canvas"):
            GS.save(fig, "outside", dirs=[str(tmp_path)])
    assert not list(tmp_path.iterdir())
