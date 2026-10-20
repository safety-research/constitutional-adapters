"""The publication gate: PSR arms in NO figure.

`scripts/tinycap_publication_gate.py` is exercised on synthetic sidecars, figure directories and review pages, and —
where the graded mirror and the two published pages are present on this machine — on the LIVE pages: the existing GSM8K
page must FAIL (its `noglm` embeds present PSR cells; the probe-draw figure's own sidecar names the constitution PSR cell),
the MATH-500 page must PASS (no PSR arm exists there), and the re-cut GSM8K page (every PSR-bearing embed replaced by an
eight-arm counterpart or removed) must PASS under a mirror root that carries clean sidecars for the counterparts.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tinycap_publication_gate", REPO / "scripts" / "tinycap_publication_gate.py")
PG = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PG
_SPEC.loader.exec_module(PG)

PSR_CELLS = ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "safety_refusal__psr_kl_all",
             "constv2-mix-full__psr_ce_all_minus_ctl_distilled")
EIGHT_ARMS = ("none", "prompted__safety_refusal", "prompted__si_constitution", "constv2ctl2-full__lora_r4_ce",
              "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "alphasteer", "arditi")
ARTIFACT = "gsm8k-paired200-paper-objects"
REVIEWS = Path("~/reviews").expanduser()
FIGS = Path("~/figs").expanduser()
GSM8K_PAGE = REVIEWS / ARTIFACT / "index.html"
M5T_PAGE = REVIEWS / "math500-thinking-on-full22" / "index.html"


def _entry(name: str, source: str, value: float = 0.5) -> dict:
    return {"name": name, "source": source, "transform": None, "value": value}


def _sidecar(figures: dict, **extra) -> dict:
    return {"figures": figures, "captions": {k: "c" for k in figures}, "sha": "8f5f149", "analysis_code_sha": "0" * 40, **extra}


def _write_figs(d: Path, sidecar_name: str, doc: dict, *, files: list[str] | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for stem in (files if files is not None else list(doc["figures"])):
        (d / f"{stem}.png").write_bytes(b"\x89PNG\r\n")
        (d / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
    (d / sidecar_name).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return d


CLEAN = {"final_g8k_accuracy_by_cell": [_entry("acc_stock.merged[gemma4e4b/main/none]", "cells.gemma4e4b/main/none.readings.merged.acc_stock"),
                                       _entry("acc_stock.merged[gemma4e4b/main/constv2-mix-full__lora_r4_ce]",
                                              "cells.gemma4e4b/main/constv2-mix-full__lora_r4_ce.readings.merged.acc_stock")],
         "final_g8k_ledger": [_entry("ledger.n_calls[qwen35_4b/main/safety_refusal__lora_r4@8192]", "legs.qwen35_4b/main/safety_refusal__lora_r4@8192.ledger.n_calls", 200)]}
PLANTED = {**CLEAN, "final_g8k_contrasts_qwen35_4b": [
    _entry("qwen35_4b/main: constv2-mix-full__lora_r4_ce - none.merged.stock.point", "contrasts.3.readings.merged.estimands.stock.primary.stats.cell_minus_none.point"),
    _entry("qwen35_4b/main: constv2ctl2-full__psr_ce_all - none.merged.stock.point", "contrasts.5.readings.merged.estimands.stock.primary.stats.cell_minus_none.point"),
    _entry("qwen35_4b/main: constv2ctl2-full__psr_ce_all - none.merged.stock.ci_low", "contrasts.5.readings.merged.estimands.stock.primary.stats.cell_minus_none.ci_low")]}


def test_the_psr_predicate_is_the_cell_grammars():
    assert all(PG.is_psr_cell(c) for c in PSR_CELLS)
    assert not any(PG.is_psr_cell(c) for c in (*EIGHT_ARMS, "safety_refusal__lora_r4", "psr", "constv2-mix-full", None, 3))


def test_sidecar_violations_are_grouped_per_stem_and_thing_and_name_the_first_record():
    assert PG.check_sidecar(_sidecar(CLEAN)) == []
    out = PG.check_sidecar(_sidecar(PLANTED))
    assert len(out) == 1 and out[0].startswith("final_g8k_contrasts_qwen35_4b: 2 plotted record(s) presents the PSR arm 'constv2ctl2-full__psr_ce_all'")
    assert "ci_low" not in out[0] and "merged.stock.point" in out[0]        # the FIRST record is named, once
    # an excluded cell / family / mode is a whole-segment match: `constv2-mix-full__lora_r4_ce` never matches its `_minus_ctl` sibling
    assert PG.check_sidecar(_sidecar(CLEAN), excluded_arms=["constv2-mix-full__lora_r4_ce_minus_ctl"]) == []
    assert len(PG.check_sidecar(_sidecar(CLEAN), excluded_arms=["constv2-mix-full__lora_r4_ce"])) == 1
    assert len(PG.check_sidecar(_sidecar(CLEAN), excluded_families=["gemma4e4b"])) == 1
    assert PG.check_sidecar(_sidecar(CLEAN), excluded_families=["gemma"]) == []
    assert len(PG.check_sidecar(_sidecar(CLEAN), excluded_modes=["main"])) == 2
    assert PG.check_sidecar(_sidecar(CLEAN), stems=["final_g8k_missing"]) == ["final_g8k_missing: not described by the sidecar"]
    assert PG.check_sidecar({"captions": {}}) == ["the sidecar carries no `figures` mapping — not an analysis rows sidecar"]


def test_the_figures_directory_gate_passes_a_clean_render_and_refuses_by_name(tmp_path):
    d = _write_figs(tmp_path / "clean", "final_g8k_figures_numbers.json", _sidecar(CLEAN))
    rec = PG.check_publication_no_psr(d, sidecar="final_g8k_figures_numbers.json", stems=list(CLEAN))
    assert rec["status"] == PG.GATE_PASS and rec["figures"] == sorted(CLEAN) and rec["violations"] == [] and rec["rule"] == PG.RULE
    # a planted PSR record refuses, naming the stem and the arm
    d2 = _write_figs(tmp_path / "planted", "final_g8k_figures_numbers.json", _sidecar(PLANTED))
    with pytest.raises(PG.PublicationRefused) as e:
        PG.check_publication_no_psr(d2, sidecar="final_g8k_figures_numbers.json")
    assert len(e.value.violations) == 1 and "constv2ctl2-full__psr_ce_all" in e.value.violations[0] and e.value.where == str(d2)
    # a figure file the sidecar does not describe cannot be verified — refused, whether found on disk or named by the render
    d3 = _write_figs(tmp_path / "undescribed", "final_g8k_figures_numbers.json", _sidecar(CLEAN), files=[*CLEAN, "final_g8k_extra"])
    with pytest.raises(PG.PublicationRefused, match="final_g8k_extra: a figure the sidecar does not describe"):
        PG.check_publication_no_psr(d3, sidecar="final_g8k_figures_numbers.json")
    with pytest.raises(PG.PublicationRefused, match="final_g8k_ghost: a figure the sidecar does not describe"):
        PG.check_publication_no_psr(d, sidecar="final_g8k_figures_numbers.json", stems=["final_g8k_ghost"])
    # the render's own exclusions are forbidden too; no sidecar / no directory refuse
    with pytest.raises(PG.PublicationRefused, match="excluded cell 'constv2-mix-full__lora_r4_ce'"):
        PG.check_publication_no_psr(d, sidecar="final_g8k_figures_numbers.json", excluded_arms=["constv2-mix-full__lora_r4_ce"])
    with pytest.raises(PG.PublicationRefused, match="excluded family 'qwen35_4b'"):
        PG.check_publication_no_psr(d, sidecar="final_g8k_figures_numbers.json", excluded_families=["qwen35_4b"])
    with pytest.raises(PG.PublicationRefused, match="no rows sidecar"):
        PG.check_publication_no_psr(d, sidecar="other.json")
    with pytest.raises(PG.PublicationRefused, match="no such figures directory"):
        PG.check_publication_no_psr(tmp_path / "nowhere", sidecar="final_g8k_figures_numbers.json")
    # the CLI: rc 0 / 6
    assert PG.main(["check-figs-dir", str(d), "--sidecar", "final_g8k_figures_numbers.json"]) == 0
    assert PG.main(["check-figs-dir", str(d2), "--sidecar", "final_g8k_figures_numbers.json"]) == PG.EXIT_PUBLICATION_REFUSED == 6


def _page(srcs: list[str], *, bare: str | None = None) -> str:
    blocks = "".join(f'<div class="figs"><figure><img loading="lazy" src="{s}" alt=""><figcaption>{s}</figcaption></figure></div>' for s in srcs)
    if bare:
        blocks += f'<img src="{bare}">'
    return f"<html><body><h1>notes-api review</h1>{blocks}<p>end</p></body></html>"


def _mirror(root: Path) -> dict[str, str]:
    """A synthetic mirror root: `noglm/` (PSR-bearing analysis sidecar), `8arm/` (clean counterparts), one ad-hoc figure whose
    own sidecar names a PSR cell, and one figure with NO sidecar."""
    _write_figs(root / ARTIFACT / "noglm", "final_g8k_figures_numbers.json", _sidecar(PLANTED))
    clean8 = {"final_g8k_eightarm_accuracy": CLEAN["final_g8k_accuracy_by_cell"], "final_g8k_eightarm_contrasts": [PLANTED["final_g8k_contrasts_qwen35_4b"][0]],
              "final_g8k_ledger": CLEAN["final_g8k_ledger"]}
    _write_figs(root / ARTIFACT / "8arm", "final_g8k_figures_numbers.json", _sidecar(clean8))
    (root / ARTIFACT / "adhoc_probe.png").write_bytes(b"\x89PNG\r\n")
    (root / ARTIFACT / "adhoc_probe.json").write_text(json.dumps({"figure": "adhoc_probe", "claim": "the constv2-mix-full__psr_ce_all cell ended inside the block",
                                                                    "draws": [{"cell": "constv2-mix-full__psr_ce_all", "n": 4}]}), encoding="utf-8")
    (root / ARTIFACT / "orphan.png").write_bytes(b"\x89PNG\r\n")
    u = f"/figs/{ARTIFACT}"
    return {"psr_contrasts": f"{u}/noglm/final_g8k_contrasts_qwen35_4b.png", "psr_acc": f"{u}/noglm/final_g8k_accuracy_by_cell.png",
            "clean_ledger": f"{u}/noglm/final_g8k_ledger.png", "adhoc": f"{u}/adhoc_probe.png", "orphan": f"{u}/orphan.png",
            "eight_acc": f"{u}/8arm/final_g8k_eightarm_accuracy.png", "eight_con": f"{u}/8arm/final_g8k_eightarm_contrasts.png",
            "eight_ledger": f"{u}/8arm/final_g8k_ledger.png"}


def test_the_page_gate_resolves_every_embed_to_its_sidecar_and_refuses_psr_or_no_sidecar(tmp_path):
    root = tmp_path / "figs"
    u = _mirror(root)
    page = tmp_path / "index.html"
    page.write_text(_page([u["psr_contrasts"], u["clean_ledger"], u["adhoc"]]), encoding="utf-8")
    with pytest.raises(PG.PublicationRefused) as e:
        PG.check_page(page, figs_root=root)
    v = e.value.violations
    assert len(v) == 2 and v[0].startswith(u["psr_contrasts"]) and "constv2ctl2-full__psr_ce_all" in v[0]
    assert v[1].startswith(u["adhoc"]) and "constv2-mix-full__psr_ce_all" in v[1]        # the ad-hoc figure's OWN sidecar is read
    # an asset with no sidecar cannot be verified; an asset that does not exist under the mirror root refuses too
    page.write_text(_page([u["clean_ledger"], u["orphan"]]), encoding="utf-8")
    with pytest.raises(PG.PublicationRefused, match="no rows sidecar beside .*orphan"):
        PG.check_page(page, figs_root=root)
    page.write_text(_page([f"/figs/{ARTIFACT}/8arm/final_g8k_nothing.png"]), encoding="utf-8")
    with pytest.raises(PG.PublicationRefused, match="does not exist under the mirror root"):
        PG.check_page(page, figs_root=root)
    # a src outside the mirror prefix is not a figure the gate can read — refused, never silently skipped
    page.write_text(_page(["https://example.invalid/x.png"]), encoding="utf-8")
    with pytest.raises(PG.PublicationRefused, match="not under the mirror prefix"):
        PG.check_page(page, figs_root=root)
    # the clean page passes with a per-figure record
    page.write_text(_page([u["clean_ledger"], u["eight_acc"]]), encoding="utf-8")
    rec = PG.check_page(page, figs_root=root)
    assert rec["status"] == PG.GATE_PASS and rec["n_figures"] == 2 and [f["sidecar_kind"] for f in rec["figures"]] == ["analysis", "analysis"]
    assert PG.main(["check-page", str(page), "--figs-root", str(root)]) == 0


def test_recut_replaces_or_removes_psr_embeds_and_the_result_passes_the_gate(tmp_path):
    root = tmp_path / "figs"
    u = _mirror(root)
    html = _page([u["psr_acc"], u["psr_contrasts"], u["clean_ledger"], u["adhoc"]], bare=u["orphan"])
    page = tmp_path / "index.html"
    page.write_text(html, encoding="utf-8")
    with pytest.raises(PG.PublicationRefused):
        PG.check_page(page, figs_root=root)
    new, rec = PG.recut_page(html, replace={u["psr_acc"]: u["eight_acc"], u["psr_contrasts"]: u["eight_con"], u["clean_ledger"]: u["eight_ledger"]},
                             remove=[u["adhoc"], u["orphan"]])
    assert rec["n_figures_before"] == 5 and rec["n_figures_after"] == 3 and sorted(rec["removed"]) == sorted([u["adhoc"], u["orphan"]])
    assert PG.page_figure_srcs(new) == [u["eight_acc"], u["eight_con"], u["eight_ledger"]]
    assert "<figcaption>" + u["adhoc"] not in new and "orphan" not in new and "<h1>notes-api review</h1>" in new and "<p>end</p>" in new
    out = tmp_path / "recut.html"
    out.write_text(new, encoding="utf-8")
    assert PG.check_page(out, figs_root=root)["status"] == PG.GATE_PASS
    # a src the page does not carry refuses (a typo would leave the PSR embed in place, silently); one src cannot be both
    with pytest.raises(PG.PublicationRefused, match="embeds no figure with src"):
        PG.recut_page(html, replace={f"/figs/{ARTIFACT}/noglm/nope.png": u["eight_acc"]})
    with pytest.raises(PG.PublicationRefused, match="both for replacement and for removal"):
        PG.recut_page(html, replace={u["adhoc"]: u["eight_acc"]}, remove=[u["adhoc"]])
    # the CLI writes the re-cut page, runs the gate on it and records both
    record = tmp_path / "recut.json"
    rc = PG.main(["recut", str(page), "--out", str(tmp_path / "cli.html"), "--figs-root", str(root), "--record", str(record),
                  "--replace", f"{u['psr_acc']}={u['eight_acc']}", "--replace", f"{u['psr_contrasts']}={u['eight_con']}",
                  "--replace", f"{u['clean_ledger']}={u['eight_ledger']}", "--remove", u["adhoc"], "--remove", u["orphan"]])
    assert rc == 0 and json.loads(record.read_text())["gate"]["status"] == PG.GATE_PASS
    assert (tmp_path / "cli.html").read_text(encoding="utf-8") == new
    # B2: a re-cut that leaves a PSR embed in place is refused by the gate over the CANDIDATE TEXT (rc 6) and the destination
    # is NOT created — the gate runs before any write
    rc = PG.main(["recut", str(page), "--out", str(tmp_path / "bad.html"), "--figs-root", str(root), "--remove", u["adhoc"], "--remove", u["orphan"]])
    assert rc == PG.EXIT_PUBLICATION_REFUSED and not (tmp_path / "bad.html").exists()
    # B2: an EXISTING destination (a served page) is left byte-identical by a refused re-cut, and is never replaced without
    # --overwrite even by a passing one; with --overwrite a PASS replaces it atomically (no temp file left beside it)
    served = tmp_path / "served" / "index.html"
    served.parent.mkdir()
    served.write_bytes(b"<html>served bytes</html>")
    rc = PG.main(["recut", str(page), "--out", str(served), "--overwrite", "--figs-root", str(root), "--remove", u["adhoc"], "--remove", u["orphan"]])
    assert rc == PG.EXIT_PUBLICATION_REFUSED and served.read_bytes() == b"<html>served bytes</html>"
    good = ["--replace", f"{u['psr_acc']}={u['eight_acc']}", "--replace", f"{u['psr_contrasts']}={u['eight_con']}",
            "--replace", f"{u['clean_ledger']}={u['eight_ledger']}", "--remove", u["adhoc"], "--remove", u["orphan"]]
    rc = PG.main(["recut", str(page), "--out", str(served), "--figs-root", str(root), *good])
    assert rc == PG.EXIT_PUBLICATION_REFUSED and served.read_bytes() == b"<html>served bytes</html>"
    rc = PG.main(["recut", str(page), "--out", str(served), "--overwrite", "--figs-root", str(root), *good])
    assert rc == 0 and served.read_text(encoding="utf-8") == new and not list(served.parent.glob(".*.tmp-*"))
    # B2: never in place — --out naming the input page refuses before anything is read or written; the input is untouched
    rc = PG.main(["recut", str(page), "--out", str(page), "--overwrite", "--figs-root", str(root), *good])
    assert rc == PG.EXIT_PUBLICATION_REFUSED and page.read_text(encoding="utf-8") == html
    with pytest.raises(PG.PublicationRefused, match="never written in place"):
        PG.recut_and_gate(page, out=tmp_path / "." / "index.html", replace={}, remove=[u["adhoc"]], figs_root=root, overwrite=True)


def test_page_images_are_read_by_the_html_parser_not_a_double_quote_regex(tmp_path):
    """B1: a single-quoted `src`, whitespace around `=`, an upper-case attribute name, a `data-src` BEFORE the real `src`,
    a self-closing tag and an entity-escaped value are ONE enumeration; an `<img>` with no `src` at all is refused by
    position, never skipped; and the re-cut rewrites exactly the tags the gate read (quoting preserved, the decoy untouched)."""
    root = tmp_path / "figs"
    u = _mirror(root)
    versioned = u["adhoc"] + "?v=1"
    html = ("<html><body>"
            f"<figure><img src='{u['psr_acc']}' alt=''></figure>"                                   # single quotes
            f"<figure><img loading=\"lazy\" SRC = \"{u['psr_contrasts']}\"></figure>"               # whitespace + case
            f"<figure><img data-src=\"{u['eight_acc']}\" src=\"{u['clean_ledger']}\"/></figure>"   # data-src first; self-closing
            f"<img src=\"{u['adhoc']}&#63;v=1\">"                                                   # an entity inside the value
            "</body></html>")
    imgs = PG.page_images(html)
    assert [im.src for im in imgs] == [u["psr_acc"], u["psr_contrasts"], u["clean_ledger"], versioned]
    assert all(html[im.start:im.end] == im.tag for im in imgs)
    page = tmp_path / "index.html"
    page.write_text(html, encoding="utf-8")
    with pytest.raises(PG.PublicationRefused) as e:
        PG.check_page(page, figs_root=root)
    v = e.value.violations
    # the PSR embed is found under `SRC = "..."`; the single-quoted stem resolves to its (clean) sidecar record and is not a
    # violation; the data-src decoy never stands in for the real src; the `?v=1` asset does not exist under the mirror and
    # is refused by name, never skipped
    assert any(x.startswith(u["psr_contrasts"]) and "constv2ctl2-full__psr_ce_all" in x for x in v)
    assert not any(x.startswith(u["psr_acc"]) for x in v) and not any(u["eight_acc"] in x for x in v)
    assert any(x.startswith(versioned) and "does not exist" in x for x in v)
    with pytest.raises(PG.PublicationRefused, match="image 0 .*no `src` attribute"):
        PG.check_page(page, figs_root=root, html="<html><img alt='x'></html>")
    new, rec = PG.recut_page(html, replace={u["psr_acc"]: u["eight_acc"], u["psr_contrasts"]: u["eight_con"]}, remove=[versioned])
    assert PG.page_figure_srcs(new) == [u["eight_acc"], u["eight_con"], u["clean_ledger"]] and rec["n_figures_after"] == 3
    assert f"src='{u['eight_acc']}'" in new and f"SRC = \"{u['eight_con']}\"" in new and f"data-src=\"{u['eight_acc']}\"" in new
    assert "&#63;v=1" not in new
    # the old regex's false negative, pinned: a page whose every embed is single-quoted does NOT pass with zero figures checked
    quoted = "<html>" + "".join(f"<figure><img src='{s}'></figure>" for s in (u["psr_contrasts"], u["clean_ledger"])) + "</html>"
    with pytest.raises(PG.PublicationRefused, match="constv2ctl2-full__psr_ce_all"):
        PG.check_page(page, figs_root=root, html=quoted)
    assert PG.check_page(page, figs_root=root, html=quoted.replace(f"'{u['psr_contrasts']}'", f"'{u['eight_con']}'"))["n_figures"] == 2


def test_responsive_image_candidates_are_refused_and_cannot_survive_a_passing_recut(tmp_path):
    """The defect this fixes: an `<img src=CLEAN srcset="PSR 2x">` passed the page gate, and a re-cut that replaced
    only `src` left the PSR candidate in `srcset` on a PASSING page. The gate enumerates `src` only and cannot know which
    candidate a browser selects, so responsive constructs are REFUSED by position through the ONE scanner: an `srcset` on
    any tag, a `<source>` (in a `<picture>`, a `<video>`, anywhere), a `<picture>`; the re-cut refuses such a page before
    any rewrite, and the CLI leaves no destination behind."""
    root = tmp_path / "figs"
    u = _mirror(root)
    page = tmp_path / "index.html"
    page.write_text("<html></html>", encoding="utf-8")
    srcset = f'<html><figure><img src="{u["clean_ledger"]}" srcset="{u["psr_contrasts"]} 2x, {u["clean_ledger"]} 1x"></figure></html>'
    picture = (f'<html><picture><source srcset="{u["psr_acc"]}" media="(min-width: 600px)"><img src="{u["eight_acc"]}"></picture></html>')
    video = f'<html><video><source src="{u["psr_acc"]}" type="video/mp4"></video><img src="{u["eight_acc"]}"></html>'
    upper = f'<html><IMG SRC="{u["eight_acc"]}" SRCSET="{u["psr_acc"]} 2x"></html>'
    # the scanner lists every construct beside the images it reads; nothing is guessed about which candidate shows
    assert [x.what for x in PG.page_unsupported(srcset)] == ["<img srcset>"]
    assert [x.what for x in PG.page_unsupported(picture)] == ["<picture>", "<source>"]
    assert [x.what for x in PG.page_unsupported(video)] == ["<source>"]
    assert [x.what for x in PG.page_unsupported(upper)] == ["<img srcset>"] and [im.src for im in PG.page_images(upper)] == [u["eight_acc"]]
    assert PG.page_unsupported(_page([u["eight_acc"]])) == [] and all(x.tag == h[x.start:x.end] for h in (srcset, picture) for x in PG.page_unsupported(h))
    # the page gate refuses each by position, whatever `src` resolves to (every `src` above is a CLEAN asset)
    for html, what in ((srcset, "<img srcset>"), (picture, "<picture>"), (video, "<source>"), (upper, "<img srcset>")):
        with pytest.raises(PG.PublicationRefused) as e:
            PG.check_page(page, figs_root=root, html=html)
        assert any(v.startswith(what) and "responsive-image candidate" in v and "embed one <img src>" in v for v in e.value.violations), (what, e.value.violations)
    # the re-cut cannot retarget a candidate set: a page carrying one is refused BEFORE any rewrite — the finding's
    # reproduction (replace `src` only) cannot produce a passing page with the PSR candidate still in `srcset`
    with pytest.raises(PG.PublicationRefused, match="cannot retarget responsive-image candidates") as e:
        PG.recut_page(srcset, replace={u["clean_ledger"]: u["eight_ledger"]})
    assert "<img srcset>" in e.value.violations[0]
    with pytest.raises(PG.PublicationRefused, match="cannot retarget"):
        PG.recut_page(picture, remove=[u["eight_acc"]])
    page.write_text(srcset, encoding="utf-8")
    out = tmp_path / "recut.html"
    rc = PG.main(["recut", str(page), "--out", str(out), "--figs-root", str(root), "--replace", f"{u['clean_ledger']}={u['eight_ledger']}"])
    assert rc == PG.EXIT_PUBLICATION_REFUSED and not out.exists() and page.read_text(encoding="utf-8") == srcset
    # the same page with the candidate set removed (one <img src> per figure) re-cuts and passes as before
    plain = _page([u["clean_ledger"]])
    new, _rec = PG.recut_page(plain, replace={u["clean_ledger"]: u["eight_ledger"]})
    rec = PG.check_page(page, figs_root=root, html=new)
    assert rec["status"] == PG.GATE_PASS and rec["n_unsupported"] == 0 and PG.page_figure_srcs(new) == [u["eight_ledger"]]


live = pytest.mark.skipif(not (GSM8K_PAGE.is_file() and M5T_PAGE.is_file() and (FIGS / ARTIFACT / "noglm").is_dir()),
                          reason="the published review pages and the figure mirror live outside the repository (absent here)")


@live
def test_the_existing_gsm8k_page_fails_and_the_math500_page_passes():
    """The page as published embeds ten `noglm` figures whose sidecar carries PSR-arm records, plus the probe-draw
    figure of the refused PSR cell — the gate must refuse it; the MATH-500 page has no PSR arm and passes."""
    with pytest.raises(PG.PublicationRefused) as e:
        PG.check_page(GSM8K_PAGE, figs_root=FIGS)
    stems = {re.search(r"/([^/]+)\.png", v).group(1) for v in e.value.violations}
    assert "adhoc_g8k_nemotron_probe_draws" in stems and {"final_g8k_accuracy_by_cell", "final_g8k_atcap_share", "final_g8k_ledger"} <= stems
    assert any("constv2-mix-full__psr_ce_all" in v for v in e.value.violations)
    rec = PG.check_page(M5T_PAGE, figs_root=FIGS)
    assert rec["status"] == PG.GATE_PASS and rec["n_figures"] == 10 and all(f["sidecar_kind"] == "analysis" for f in rec["figures"])


@live
def test_the_recut_gsm8k_page_passes_under_a_mirror_carrying_its_eight_arm_counterparts(tmp_path):
    """The re-cut (§4.7): every `noglm/final_g8k_<stem>.png` embed swapped for `8arm/final_g8k_<stem>.png`, the probe-draw
    figure removed (it becomes an appendix TABLE row). Under a mirror root whose `8arm/` sidecar describes the counterparts
    without a PSR record the refreshed page PASSES; the same page under the published mirror (no `8arm/` yet) refuses by
    name — the assets must exist before the page is published."""
    html = GSM8K_PAGE.read_text(encoding="utf-8")
    srcs = PG.page_figure_srcs(html)
    noglm = [s for s in srcs if "/noglm/" in s]
    adhoc = [s for s in srcs if "/adhoc_" in s]
    assert len(noglm) == 10 and len(adhoc) == 1
    replace = {s: s.replace("/noglm/", "/8arm/") for s in noglm}
    new, rec = PG.recut_page(html, replace=replace, remove=adhoc)
    assert rec["n_figures_after"] == 10 and all("/8arm/" in s for s in PG.page_figure_srcs(new))
    root = tmp_path / "figs"
    stems = [Path(s).stem for s in replace.values()]
    _write_figs(root / ARTIFACT / "8arm", "final_g8k_figures_numbers.json",
                _sidecar({stem: [_entry("acc_stock.merged[gemma4e4b/main/none]", "cells.gemma4e4b/main/none.readings.merged.acc_stock")] for stem in stems}))
    out = tmp_path / "index.html"
    out.write_text(new, encoding="utf-8")
    assert PG.check_page(out, figs_root=root)["status"] == PG.GATE_PASS
    with pytest.raises(PG.PublicationRefused, match="does not exist under the mirror root"):
        PG.check_page(out, figs_root=FIGS)
