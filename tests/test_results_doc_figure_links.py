"""The figure links a results document carries (realign_analysis, agentic_control_analysis): under the figure mirror's
public base URL when ``$SISTEER_FIGS_URL_BASE`` is set, else RELATIVE TO THE DOCUMENT, so a document written under
``--out`` resolves its figures wherever ``--figs-dir`` put them. Synthetic paths only (tmp_path)."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import realign_analysis as RA  # noqa: E402
import agentic_control_analysis as AC  # noqa: E402


def test_links_are_relative_to_the_document_when_no_mirror_url_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv(RA.FIGS_URL_BASE_ENV, raising=False)
    figs_dir = tmp_path / "figs" / "realign"
    figs_dir.mkdir(parents=True)
    (figs_dir / "h1_forest_harmful_rate.png").write_bytes(b"png")
    md_path = tmp_path / "out" / "realign_analysis" / "RESULTS_REALIGN.md"
    base = RA.figure_link_base(figs_dir, md_path)
    assert base == "../../figs/realign/"
    link = base + "h1_forest_harmful_rate.png"
    assert (md_path.parent / link).resolve() == (figs_dir / "h1_forest_harmful_rate.png").resolve()
    assert RA.figs_url_base() == ""
    # a document beside its figures links them by name alone
    assert RA.figure_link_base(figs_dir, figs_dir / "RESULTS.md") == "./"
    # the empty / blank variable is the unset case
    monkeypatch.setenv(RA.FIGS_URL_BASE_ENV, "   ")
    assert RA.figure_link_base(figs_dir, md_path) == "../../figs/realign/"


def test_links_go_under_the_mirror_url_when_it_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv(RA.FIGS_URL_BASE_ENV, "https://example/figs/")
    assert RA.figs_url_base() == "https://example/figs/"
    assert RA.figure_link_base(tmp_path / "figs" / "realign", tmp_path / "RESULTS.md") == "https://example/figs/realign/"
    monkeypatch.setenv(RA.FIGS_URL_BASE_ENV, "https://example/figs")     # no trailing slash: the same
    assert RA.figure_link_base(tmp_path / "figs" / "agentic-control", tmp_path / "x" / "R.md") == "https://example/figs/agentic-control/"


def test_both_results_writers_link_through_the_document_relative_base():
    """Both ``run`` bodies derive the document path FIRST and the link base from it (the agentic script imports the helper
    rather than keeping a copy), and the 'served at' line is printed only when a mirror URL exists."""
    for mod in (RA, AC):
        src = inspect.getsource(mod.run)
        assert "figs_url = figure_link_base(figs_dir, md_path)" in src, mod.__name__
        assert src.index("md_path = ") < src.index("figure_link_base(figs_dir, md_path)"), mod.__name__
        assert "if figs_url_base():" in src and "figures served at" in src, mod.__name__
    assert AC.figure_link_base is RA.figure_link_base and AC.figs_url_base is RA.figs_url_base
    for mod in (RA, AC):
        assert "tailnet" not in inspect.getsource(mod.render_results_md).lower(), mod.__name__
