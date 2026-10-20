"""ifeval-canonical-battery (P2): the results skeleton is an IMMUTABLE template at its own path.

Before this ``RESULTS_TEMPLATE`` was the rendered record's own path — ALSO the documented ``--results-md`` output
— so the first render consumed every ``{{name}}`` placeholder and every later render silently kept the old numbers (a
byte-identical second output after the pin changed). Now the template lives at ``results.template.md``, the rendered
record at ``results.md``, and ``render_results_md`` refuses an output that aliases the template and a template with no
placeholders left, before anything is written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

# the analysis scripts import by module name; this module must collect even when no other module put scripts/ on the path first
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tinycap_ifeval_battery as A  # noqa: E402
import tinycap_ifeval_record_prose as P  # noqa: E402
from sisteer import site as SITE  # noqa: E402
from test_tinycap_ifeval_analysis import B_FAST, ESC, SMALL, small_report, small_store  # noqa: F401,E402
import ifeval_synthetic_store as G  # noqa: E402


def test_two_successive_renders_with_a_changed_pin_differ_and_the_second_carries_the_new_value(small_report, tmp_path):
    assert A.RESULTS_TEMPLATE.name == "results.template.md" and A.RESULTS_TEMPLATE.is_file()
    assert A.RESULTS_TEMPLATE.read_text(encoding="utf-8").count("{{") >= 20
    assert A.RESULTS_RECORD_REL.endswith("/results.md") and A.RESULTS_RECORD_REL != str(A.RESULTS_TEMPLATE)
    out = tmp_path / "results.md"
    first = A.render_results_md(small_report, out=out)
    out.write_text(first, encoding="utf-8")
    changed = copy.deepcopy(small_report)
    changed["sha"] = "feedbee"
    second = A.render_results_md(changed, out=out)          # the SAME output path, the template untouched
    assert first != second and "feedbee" in second and "feedbee" not in first and G.SHA7 in first
    assert "{{" not in first and "{{" not in second
    # the template still carries every placeholder after both renders
    assert A.RESULTS_TEMPLATE.read_text(encoding="utf-8").count("{{") >= 20


def test_an_output_aliasing_the_template_and_a_consumed_template_are_refused_by_name(small_report, tmp_path, small_store):
    with pytest.raises(A.AnalysisError, match="aliases the results TEMPLATE"):
        A.render_results_md(small_report, out=A.RESULTS_TEMPLATE)
    # through main: since revision 3 a production invocation READS the .eval logs, so `--no-eval-logs` here is refused at the
    # command line before any analysis or render — the template byte-identical, nothing written; the render-step alias refusal through
    # main runs where the logs exist (test_ifeval_analysis_review_r3.py, inspect_ai)
    before = A.RESULTS_TEMPLATE.read_bytes()
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(G.IDS))
    matrix = tmp_path / "matrix.json"
    from sisteer.capeval.tinycap import ifeval_contracts as IC
    matrix.write_text(json.dumps({"schema": IC.MATRIX_SCHEMA, "rows": SMALL}))
    argv = ["--gen-root", str(small_store.root), "--sha", G.SHA7, "--matrix", str(matrix), "--ids", str(ids), "--bootstrap", str(B_FAST),
            "--no-eval-logs", "--wandb-mode", "disabled", "--no-mirror", "--no-figs", "--analysis-sha", G.FULL_SHA,
            "--out", str(tmp_path / "r.json"), "--results-md", str(A.RESULTS_TEMPLATE)]
    with pytest.raises(SystemExit) as e:
        A.main(argv)
    assert e.value.code == 2 and A.RESULTS_TEMPLATE.read_bytes() == before and not (tmp_path / "r.json").exists()
    # a consumed template (a render written over a skeleton) is refused by name
    consumed = tmp_path / "consumed.md"
    consumed.write_text(A.render_results_md(small_report), encoding="utf-8")
    with pytest.raises(A.AnalysisError, match="CONSUMED template"):
        A.render_results_md(small_report, template=consumed)
    # the documented output path renders fine — a NON-production render here (the report read no log) and it says so
    out = tmp_path / "record" / "results.md"
    rendered = A.render_results_md(small_report, out=out)
    assert rendered.count("{{") == 0 and "NOT a production result" in rendered
    # a comparison-only report renders no record at all (r3 S1)
    with pytest.raises(A.AnalysisError, match="comparison-only"):
        A.render_results_md({**small_report, "comparison_only": {"mode": A.COMPARISON_ONLY_MODE}}, out=out)


# --------------------------------------------------------------------------- the figures row states what the run did (2026-10-19)
def _figures_row(md: str) -> str:
    rows = [line for line in md.splitlines() if line.startswith("| figures |")]
    assert len(rows) == 1, rows
    return rows[0]


def test_the_figures_row_states_what_the_run_did_with_its_figures(small_report):
    """The template once stated one mirror directory, serving path and W&B project for every run; the row now reads the report:
    no figures, figures not mirrored and not logged, and figures mirrored + logged each say exactly that, naming only where THIS run wrote."""
    bare = {k: v for k, v in small_report.items() if k not in ("figures", "mirrored", "wandb_url")}
    figs = {"written": {}, "numbers": "final_ifeval_figures_numbers.json"}
    home = Path.home()
    mirrored = [str(home / "figs-elsewhere" / "run-x" / name) for name in ("a.png", "a.pdf")]
    url = "https://wandb.example/team/example-project/runs/abc12345"
    r0 = _figures_row(A.render_results_md(bare))
    r1 = _figures_row(A.render_results_md({**bare, "figures": figs}))
    r2 = _figures_row(A.render_results_md({**bare, "figures": figs, "mirrored": mirrored, "wandb_url": url}))
    assert r0 == P.FIGURES_ROW_SLOT + "no figures made by this run (`--no-figs`): nothing mirrored, nothing logged to W&B |"
    assert r1 == P.FIGURES_ROW_SLOT + "not mirrored (`--no-mirror`); no W&B run URL (`--wandb-mode disabled`, or offline until synced) |"
    assert r2 == (P.FIGURES_ROW_SLOT + f"mirrored byte-identical to `~/figs-elsewhere/run-x/` (2 files); logged to W&B as {url}, "
                  "panel keys `ifeval/<stem>` |")
    for row in (r0, r1, r2):
        assert "ifeval-canonical" not in row and "served at" not in row and "revision" not in row and str(home) not in row


def test_the_filler_keeps_the_rendered_account_and_only_the_recorded_publication_keeps_its_row_as_filed(small_report):
    figs = {"written": {}, "numbers": "final_ifeval_figures_numbers.json"}
    mirrored = [str(Path.home() / "figs-elsewhere" / "run-x" / "a.png")]
    url = "https://wandb.example/team/example-project/runs/abc12345"
    run = {**small_report, "figures": figs, "mirrored": mirrored, "wandb_url": url}
    rendered = _figures_row(A.render_results_md(run))
    F = P.Filler(A.render_results_md(run))
    P.figures_row(run, F)
    assert _figures_row(F.md) == P.FIGURES_ROW_FILLED + rendered[len(P.FIGURES_ROW_SLOT):] and "{…}" not in _figures_row(F.md)
    # the authors' revision-3 publication: its label and its own W&B run id; the row as filed, the project read from the run's URL
    rec_url = f"https://wandb.example/team/example-project/runs/{P.RECORDED_PUBLICATION['wandb_run']}"
    rec = {**run, "sha": P.RECORDED_PUBLICATION["label"], "wandb_url": rec_url}
    assert P.is_recorded_publication(rec) and P.wandb_run_parts(rec_url) == ("example-project", P.RECORDED_PUBLICATION["wandb_run"])
    F = P.Filler(A.render_results_md(rec))
    P.figures_row(rec, F)
    assert _figures_row(F.md) == P.RECORDED_FIGURES_ROW.format(label=P.RECORDED_PUBLICATION["label"], project="example-project")
    assert "refreshed in place at revisions 2 and 3" in _figures_row(F.md)
    # a reproduction under the same label with its own run (or none), and another label with the recorded run id, state their own facts
    for other in ({**rec, "wandb_url": url}, {**rec, "wandb_url": None}, {**rec, "sha": "feedbee"}):
        assert not P.is_recorded_publication(other)
        F = P.Filler(A.render_results_md(other))
        P.figures_row(other, F)
        assert _figures_row(F.md).startswith(P.FIGURES_ROW_FILLED) and "revisions 2 and 3" not in _figures_row(F.md)
    # the slot is filled once: a second fill over a filled row refuses by name, on either path
    for report in (run, rec):
        F = P.Filler(A.render_results_md(report))
        P.figures_row(report, F)
        with pytest.raises(SystemExit, match="expected exactly once, found 0"):
            P.figures_row(report, F)


#: the recorded record's figures row (line 18 of the last tracked rendered record, removed with results/ in 4d71e59ea), by digest
RECORDED_FIGURES_ROW_SHA256 = "9ad8044491047043657e14e81b8226c3080c1972165edc654b0687eb45b98b48"
#: the authors' preserved revision-3 report (their grading host only; IFEVAL_ARTIFACTS points elsewhere)
ARTIFACTS = Path(os.environ.get("IFEVAL_ARTIFACTS") or SITE.expand_path("$GRADE/tinycap/ifeval")).expanduser()
R3_REPORT = ARTIFACTS / "final_e43f81f" / "ifeval_report.json"


@pytest.mark.skipif(not R3_REPORT.is_file(), reason="the authors' preserved revision-3 report is on their grading host only")
def test_the_authors_revision_3_report_fills_the_figures_row_byte_for_byte_as_filed():
    rep = json.loads(R3_REPORT.read_text(encoding="utf-8"))
    assert P.is_recorded_publication(rep) and rep.get("mirrored")
    F = P.Filler(A.render_results_md(rep))
    P.figures_row(rep, F)
    assert hashlib.sha256(_figures_row(F.md).encode("utf-8")).hexdigest() == RECORDED_FIGURES_ROW_SHA256
