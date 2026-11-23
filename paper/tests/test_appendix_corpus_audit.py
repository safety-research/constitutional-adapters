"""appendix_corpus_audit: the corpus-audit appendix table and figure, read from the committed sidecar `data/appendix_corpus_audit.json`.

Pinned here:
  * the committed sidecar loads: its judge stamp names the corpus labeller (not an evaluation judge), its provenance names the audit
    summary by sha256 relative to a declared root, and the file holds no record id, no home path and no per-record field;
  * the paper's Table 3 (tab:corpus-audit) is the manuscript's, cell for cell, with its caption; the fuller notebook table's cells
    are the audit's every claim and state, and its caption's numbers (the declined shares, the 13-word run, the reread bound) come
    from the sidecar;
  * the figure's segments are the interim figure's values (positive, other not cleared, declined on input per claim and part), drawn
    stacked in that order, positive in the corpus's colour, each bar ending in its label inside the panel, saved in three formats;
  * the checks refuse a tampered sidecar by name (stamp, a share that no longer equals its counts, a total the parts do not sum to, a
    record id, a per-record field, a source outside its root, a missing or stray key, a reread bound that is not its counts', a listed
    set of an evaluation family the paper omits, a set left out with a record over a threshold or a run longer than the next-longest
    quoted one), and the table refuses when a footnote would be false;
  * the builder copies only the declared keys from a synthetic audit directory, drops the scan table's per-record columns, leaves out
    a set of an omitted evaluation family (keeping its count and maxima, and checking its ledger rows against the summary), and refuses
    a partition that does not close, a scan file the summary was not built from, or a set left out with a record over a threshold.
No registry, no audit files: the builder test writes its own synthetic audit directory."""
import copy
import csv
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import registry_nb as R  # noqa: E402
import appendix_corpus_audit as CA  # noqa: E402

# tab:corpus-audit as the manuscript sets it (sections/app_corpus_audit.tex): row label, then constitution advice, constitution
# stories, control advice, control stories
# the paper's Table 3 (tab:corpus-audit), cell for cell
PAPER_ROWS = [
    ("Records", ["1{,}960", "16{,}321", "1{,}995", "16{,}294"]),
    ("Depicted jailbreak attempt", ["21", "783", "0", "0"]),
    ("Harmful request", ["51", "714", "0", "0"]),
    ("Request for a lesser wrongful act", ["1{,}517", "2{,}948", "1", "3"]),
    ("Declined by the labeler", ["184", "58", "0", "0"]),
]
# the fuller notebook table (tab:corpus-audit-full), every claim and state
MANUSCRIPT_ROWS = [
    ("Records", ["1{,}960", "16{,}321", "1{,}995", "16{,}294"]),
    ("\\quad declined on input (not cleared)", ["184", "58", "0", "0"]),
    ("C1 eval-text: over a threshold$^a$", ["1", "14", "1", "0"]),
    ("C1 jailbreak half: positive", ["21", "783", "0", "0"]),
    ("C2: positive", ["51", "714", "0", "0"]),
    ("\\quad share of records read (\\%)", ["2.9", "4.4", "0.0", "0.0"]),
    ("\\quad other not cleared$^b$", ["4", "36", "0", "0"]),
    ("\\quad C2 requests declined", ["54/59", "736/749", "--", "--"]),
    ("\\quad with uplift$^c$", ["4", "0", "0", "0"]),
    ("C3: positive", ["1{,}517", "2{,}948", "1", "3"]),
    ("\\quad share of records read (\\%)", ["85.4", "18.1", "0.05", "0.02"]),
    ("C2 agreement, Cohen's $\\kappa$", ["0.86", "0.88", "--", "--"]),
    ("Blind reread: misses$^d$", ["0/1{,}159", "0/1{,}000", "0/1{,}893", "0/1{,}000"]),
]
# the interim figure's values (figA_corpus_audit_values.tsv): positive, contested + partial + unresolved, declined on input
INTERIM = {("c2", "constitution_advice"): (51, 4, 184), ("c2", "constitution_stories"): (714, 36, 58),
           ("c1_jailbreak", "constitution_advice"): (21, 5, 184), ("c1_jailbreak", "constitution_stories"): (783, 50, 58),
           ("c3", "constitution_advice"): (1517, 3, 184), ("c3", "constitution_stories"): (2948, 26, 58),
           ("c3", "control_advice"): (1, 0, 0), ("c3", "control_stories"): (3, 0, 0)}
JUDGE = "labeller: Claude Opus 5.5 (corpus audit, not an evaluation judge)"


@pytest.fixture(scope="module")
def ca():
    return CA.load_corpus_audit()


def test_committed_sidecar_checks_out(ca):
    assert CA.corpus_audit_problems(ca) == []
    assert ca["judge"] == JUDGE == CA.JUDGE
    p = ca["provenance"]
    assert (p["summary_version"], p["rubric_version"]) == ("hrc-summary-v2", "hrc-v2")
    assert p["judge"]["labeller_model"] == "claude-opus-5-5" and p["judge"]["evaluation_judge"] is False
    assert p["builder"] == "paper/notebooks/data/build_appendix_corpus_audit.py"
    summ = [s for s in p["sources"] if s["root"] == "audit" and s["file"] == "summary.json"]
    assert len(summ) == 1 and len(summ[0]["sha256"]) == 64
    assert set(ca["numbers"]) >= set(CA.required_keys())


def test_sidecar_holds_no_record_id_home_path_or_per_record_field():
    raw = pathlib.Path(CA.SIDECAR).read_text(encoding="utf-8")
    for needle in ("adv-p", "sto-c", "/home/", "~/", "max_run_record", "max_run_item", "max_coverage_record", "quote"):
        assert needle not in raw, needle
    d = json.loads(raw)
    assert set(d) == CA.TOP_FIELDS
    assert all(set(s) <= {"set_id", "tier", "role", "status", "goal_set", "standin"} for s in d["overlap"]["sets"])


def test_table_cells_are_the_manuscripts(ca):
    assert CA.paper_table_rows(ca) == PAPER_ROWS
    rows = [r for r in CA.table_rows(ca) if r is not None]
    assert rows == MANUSCRIPT_ROWS


def test_paper_table_is_table_3(ca):
    tex = CA.table_corpus_audit(ca)
    assert "\\label{tab:corpus-audit}" in tex and "\\caption{" + CA.TABLE_CAPTION + "}" in tex and "\\addlinespace" not in tex
    assert [ln.split("&")[0].strip() for ln in tex.splitlines() if ln.endswith(" \\\\") and "&" in ln][2:] == [r[0] for r in PAPER_ROWS]


def test_table_latex_and_caption_read_the_sidecar(ca):
    tex = CA.table_corpus_audit_full(ca)
    assert "\\label{tab:corpus-audit-full}" in tex and "\\toprule" in tex and "\\bottomrule" in tex
    assert tex.count("\\addlinespace") == 4 and tex.count(" \\\\\n") >= 15
    cap = CA.table_caption(ca)
    assert "declined 9.4\\% of the constitution advice and 0.4\\% of the constitution stories" in cap
    assert "$^a$All through a 13-word run" in cap and "for stories, 0.30\\%)" in cap
    moved = copy.deepcopy(ca)
    moved["numbers"]["pct.share.declined_on_input.records.constitution_advice"] = 12.34
    assert "declined 12.3\\%" in CA.table_caption(moved)
    assert CA._pct(0.0501) == "0.05" and CA._pct(0.0) == "0.0" and CA._pct(85.4167) == "85.4" and CA._pct(2.8716) == "2.9"


def test_figure_values_are_the_interim_figures(ca):
    v = CA.figure_values(ca).set_index(["claim", "part"])
    for c, _pre, _st, _t in CA.CLAIMS:
        for p in CA.PARTS:
            assert tuple(int(x) for x in v.loc[(c, p), ["positive", "other", "declined"]]) == INTERIM.get((c, p), (0, 0, 0)), (c, p)


def test_figure_draws_every_segment_and_label(ca, tmp_path):
    fig = CA.corpus_audit_figure(ca, "figA_corpus_audit", out_dirs=[str(tmp_path)])
    try:
        bars = {p.get_gid(): p for ax in fig.axes for p in ax.patches if isinstance(p, Rectangle) and (p.get_gid() or "").startswith("corpus-audit|")}
        texts = {t.get_gid(): t for ax in fig.axes for t in ax.texts if (t.get_gid() or "").startswith("corpus-audit-label|")}
        assert len(bars) == 36 and len(texts) == 12
        colour = {"positive": None, "other": CA.OTHER_COLOR, "declined": CA.DECLINED_COLOR}
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for ax, (c, _pre, _st, title) in zip(fig.axes, CA.CLAIMS):
            assert ax.get_title(loc="left") == title                 # R.style() sets axes.titlelocation = left
            for p in CA.PARTS:
                pos, other, dec = INTERIM.get((c, p), (0, 0, 0))
                left = 0.0
                for seg, w in (("positive", pos), ("other", other), ("declined", dec)):
                    b = bars[f"corpus-audit|{c}|{p}|{seg}"]
                    assert (b.get_x(), b.get_width()) == pytest.approx((left, w)), (c, p, seg)
                    want = colour[seg] or CA.CORPUS_COLOR[CA.CORPUS[p]]
                    assert mcolors.to_hex(b.get_facecolor()) == mcolors.to_hex(want), (c, p, seg)
                    left += w
                t = texts[f"corpus-audit-label|{c}|{p}"]
                assert t.get_text() == (f"{pos:,}" + (f" + {other + dec:,} not cleared" if other + dec else ""))
                assert t.get_window_extent(r).x1 <= ax.get_window_extent(r).x1 + 1, f"label {t.get_text()!r} leaves its panel"
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == [CA.PART_LABEL[p] for p in CA.PARTS]
        labels = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
        assert labels == ["positive: constitution corpus", "positive: control corpus", CA.SEGMENT_LABEL["other"], CA.SEGMENT_LABEL["declined"]]
        assert CA.CORPUS_COLOR == {"constitution": R.COLOR["constitution_lora"], "control": R.COLOR["control_lora"]}
        assert sorted(f.name for f in tmp_path.iterdir()) == [f"figA_corpus_audit.{e}" for e in ("pdf", "png", "svg")]
    finally:
        plt.close(fig)


# a synthetic scanned-set id of an evaluation family the paper omits, built from the notebook's own list (no family is named here)
_OMITTED_FAMILY_SET = sorted(R.OMITTED_EVALS)[0].split("_", 1)[0] + "_synthetic_prompts"


def _tamper(ca, fn):
    d = copy.deepcopy(ca)
    fn(d)
    return CA.corpus_audit_problems(d)


@pytest.mark.parametrize("why, fn, phrase", [
    ("judge", lambda d: d.update(judge="Claude Opus 5.5"), "judge stamp"),
    ("share", lambda d: d["numbers"].update({"pct.share_read.c2_headline.positive.constitution_advice": 3.5}), "share of read records"),
    ("total", lambda d: d["numbers"].update({"c2.headline.positive.all": 766}), "the parts sum to"),
    ("count", lambda d: d["numbers"].update({"c3.any_wrongful.positive.control_stories": 4}), "share of read records"),
    ("state over records", lambda d: d["numbers"].update({"c3.any_wrongful.partial.control_advice": 1995}), "more than the part's"),
    ("record id", lambda d: d.update(about=d["about"] + " e.g. constitution_advice/adv-p01-t00-s00"), "record id"),
    ("per-record field", lambda d: d["overlap"]["sets"][0].update(max_run_record="x"), "per-record data"),
    ("source outside root", lambda d: d["provenance"]["sources"][0].update(file="/abs/summary.json"), "not a path inside its root"),
    ("undeclared root", lambda d: d["provenance"]["sources"][0].update(root="elsewhere"), "declared root"),
    ("missing key", lambda d: d["numbers"].pop("coverage.records.control_stories"), "required key"),
    ("stray key", lambda d: d["numbers"].update({"c2.analog.positive.all": 729}), "outside the declared subset"),
    ("bound", lambda d: d["numbers"].update({"fn.headline.cp_upper95.constitution_stories": 0.003}), "upper bound"),
    ("treatment", lambda d: [d["numbers"].pop(f"c2.requests.total.{k}.constitution_advice") for k in CA.C2_TREATMENT], "no C2 treatment counts"),
    ("schema", lambda d: d.update(schema="v0"), "schema"),
    ("listed omitted-family set", lambda d: d["overlap"]["sets"].append(dict(d["overlap"]["sets"][0], set_id=_OMITTED_FAMILY_SET)),
     "evaluation family the paper omits"),
    ("left-out record over a threshold", lambda d: d["overlap"]["sets_left_out"].update(records_over_threshold=1), "in a set left out"),
    ("left-out block missing", lambda d: d["overlap"].pop("sets_left_out"), "sets_left_out"),
    ("left-out run past the quoted runs", lambda d: d["overlap"]["sets_left_out"].update(max_run=12), "longer than the next-longest"),
])
def test_problems_name_each_tampering(ca, why, fn, phrase):
    problems = _tamper(ca, fn)
    assert any(phrase in p for p in problems), (why, problems)


def test_a_left_out_run_may_tie_the_next_longest_quoted_one(ca):
    """The text quotes the longest run (12, wildjailbreak) and the next (11): a set left out with 11 leaves both true, 12 does not."""
    assert CA.overlap_facts(ca)["second"][1] == 11
    assert _tamper(ca, lambda d: d["overlap"]["sets_left_out"].update(max_run=11)) == []
    assert CA.left_out_run_problems(ca["numbers"], dict(ca["overlap"], sets_left_out=dict(ca["overlap"]["sets_left_out"], max_run=12)))


def test_load_refuses_a_tampered_file(ca, tmp_path):
    d = copy.deepcopy(ca)
    d["judge"] = "evaluation judge"
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(CA.CorpusAuditSidecarError, match="judge stamp"):
        CA.load_corpus_audit(str(f))


def test_overlap_facts(ca):
    f = CA.overlap_facts(ca)
    assert (f["route_run"], f["goal_frac"]) == (13, 0.7)
    assert f["longest"] == ("wildjailbreak", 12) and f["second"] == ("bigcodebench", 11)
    assert f["max_coverage"] == pytest.approx(0.6667) and "jailbreak_goals_259" in f["max_coverage_sets"]
    assert (f["fixed_over_threshold"], f["petri_records"], f["petri_max_run"]) == (0, 16, 29)
    assert f["expected"] == {"system_instructions_prompted_baselines": 19, "sdfbase_probe_sets": 28}
    assert f["not_scanned"] == ["bfcl"] and f["standins"] == ["math500"]
    assert "longest shared run 12 words (wildjailbreak)" in CA.overlap_line(ca)


def test_table_refuses_a_false_footnote(ca):
    d = copy.deepcopy(ca)
    d["numbers"]["overlap.records_eval_text_positive.wildjailbreak"] = 1
    with pytest.raises(CA.CorpusAuditSidecarError, match="footnote a"):
        CA.table_corpus_audit_full(d)
    d = copy.deepcopy(ca)
    d["uplift_levels"] = ["none", "partial", "substantial"]
    with pytest.raises(CA.CorpusAuditSidecarError, match="footnote c"):
        CA.table_corpus_audit_full(d)
    d = copy.deepcopy(ca)
    d["numbers"].update({"c2.requests.uplift.partial.constitution_advice": 0, "c2.requests.uplift.partial.all": 0})
    with pytest.raises(CA.CorpusAuditSidecarError, match="no C2 request judged partial uplift"):
        CA.table_corpus_audit_full(d)


# ---- the builder, on a synthetic audit directory ------------------------------------------------------------------------------------
def _builder():
    spec = importlib.util.spec_from_file_location("build_appendix_corpus_audit", NOTEBOOKS / "data" / "build_appendix_corpus_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _synthetic_audit(ca, root: pathlib.Path, break_partition=False, wrong_sha=False, omitted_set=None) -> pathlib.Path:
    """An audit directory whose summary carries the sidecar's numbers plus what the builder must read and never store: negatives that
    close each partition, an unrelated key, and a scan table with per-record columns."""
    audit = root / "audit"
    (audit / "scan").mkdir(parents=True)
    n = dict(ca["numbers"])
    for _c, pre, _st, _t in CA.CLAIMS:
        for p in CA.PARTS:
            n[f"{pre}.negative.{p}"] = n[f"coverage.records.{p}"] - sum(n[f"{pre}.{s}.{p}"] for s in CA.STATES) + (1 if break_partition else 0)
    n["c2.requests.uplift.none.all"] = 1
    n["c2.analog.positive.all"] = 729                      # not declared: must not reach the sidecar
    table = [dict(s, goal_set=s["goal_set"], standin=("stand-in" if s["standin"] else None),
                  max_run=n[f"overlap.max_run.{s['set_id']}"], records_eval_text_positive=n[f"overlap.records_eval_text_positive.{s['set_id']}"],
                  max_run_record="constitution_stories/sto-c01-u01-k01-y0", max_run_item="item-1") for s in ca["overlap"]["sets"]]
    if omitted_set:                                        # (set id, longest run, records over a threshold), at the table's end
        sid, run, over = omitted_set
        n.update({f"overlap.max_run.{sid}": run, f"overlap.records_eval_text_positive.{sid}": over})
        table.append(dict(set_id=sid, tier="A", role="an evaluation's prompts", status="materialised", goal_set=False, standin=None,
                          max_run=run, records_eval_text_positive=over, max_run_record="control_advice/adv-p01-t00-s00"))
    meta = {"thresholds": dict(ca["overlap"]["thresholds"], run_n=8), "run_rule": ca["overlap"]["run_rule"], "set_table": table}
    (audit / "scan" / "overlap_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with open(audit / "scan" / "overlap_expected.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["set_id", "tier", "role", "status", "max_run", "max_run_record"])
        w.writeheader()
        for e in ca["overlap"]["expected_sets"]:
            w.writerow(dict(e, max_run=n[f"overlap_expected.max_run.{e['set_id']}"], max_run_record="control_advice/adv-p01-t00-s00"))
    inputs = {"scan/overlap_meta.json": _sha(audit / "scan" / "overlap_meta.json"),
              "scan/overlap_expected.csv": "0" * 64 if wrong_sha else _sha(audit / "scan" / "overlap_expected.csv")}
    summary = {"summary_version": "hrc-summary-v2", "rubric_version": "hrc-v2", "flags_sha256": "f" * 64, "inputs": inputs, "numbers": n}
    (audit / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return audit


def test_builder_copies_only_the_declared_keys(ca, tmp_path):
    B = _builder()
    out, problems = B.derive(_synthetic_audit(ca, tmp_path), None)
    assert problems == []
    assert out["numbers"] == ca["numbers"]
    assert out["overlap"]["sets"] == ca["overlap"]["sets"] and out["overlap"]["expected_sets"] == ca["overlap"]["expected_sets"]
    assert out["uplift_levels"] == ["none", "partial"]
    text = B.dump(out)
    assert "sto-c01" not in text and "adv-p01" not in text and "max_run_record" not in text and str(tmp_path) not in text
    assert {(s["root"], s["file"]) for s in out["provenance"]["sources"]} == {
        ("audit", "summary.json"), ("audit", "scan/overlap_meta.json"), ("audit", "scan/overlap_expected.csv")}


def test_builder_refuses_an_open_partition_or_a_foreign_scan_file(ca, tmp_path):
    B = _builder()
    _out, problems = B.derive(_synthetic_audit(ca, tmp_path / "a", break_partition=True), None)
    assert any("the six states sum to" in p for p in problems)
    _out, problems = B.derive(_synthetic_audit(ca, tmp_path / "b", wrong_sha=True), None)
    assert any("scan/overlap_expected.csv" in p and "summary was built from" in p for p in problems)


def test_builder_leaves_out_a_set_of_an_omitted_evaluation_family(ca, tmp_path):
    """Not listed, its two numbers not copied, its count and longest run kept in overlap.sets_left_out; a set left out with a record
    over a threshold is refused (the fuller table's footnote a counts those records)."""
    B = _builder()
    out, problems = B.derive(_synthetic_audit(ca, tmp_path / "a", omitted_set=(_OMITTED_FAMILY_SET, 9, 0)), None)
    assert problems == []
    assert out["overlap"]["sets"] == ca["overlap"]["sets"] and out["numbers"] == ca["numbers"]
    assert _OMITTED_FAMILY_SET not in B.dump(out)
    assert out["overlap"]["sets_left_out"] == dict(n_sets=1, max_run=9, records_over_threshold=0, why=CA.LEFT_OUT_WHY)
    n = len(ca["overlap"]["sets"]) + 1
    assert f"{n} sets, 1 of them from an evaluation family the paper omits and not listed" in CA.overlap_line(out)
    _out, problems = B.derive(_synthetic_audit(ca, tmp_path / "b", omitted_set=(_OMITTED_FAMILY_SET, 14, 2)), None)
    assert any("in a set left out" in p for p in problems), problems


def test_builder_checks_a_left_out_sets_ledger_row_against_the_summary(ca, tmp_path):
    """A ledger row of a set left out is matched against the summary's own number, which the sidecar does not copy: counted when equal,
    refused when not; a listed set's row is matched against the sidecar and a key of neither is not counted."""
    B = _builder()
    audit = _synthetic_audit(ca, tmp_path / "a", omitted_set=(_OMITTED_FAMILY_SET, 9, 0))
    lane = tmp_path / "lane"
    lane.mkdir()

    def ledger(rows):
        with open(lane / "numbers_ledger.tsv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["summary.json key", "raw", "formatted", "written", "check"])
            w.writerows([k, v, v, v, "OK"] for k, v in rows)

    ledger([("overlap.max_run.wildjailbreak", 12), (f"overlap.max_run.{_OMITTED_FAMILY_SET}", 9), ("overlap.max_run.no_such_set", 3)])
    out, problems = B.derive(audit, lane)
    assert problems == [] and _OMITTED_FAMILY_SET not in B.dump(out)
    assert [(c["file"], c["rows_matched"]) for c in out["provenance"]["cross_checked"]] == [("numbers_ledger.tsv", 2)]
    ledger([(f"overlap.max_run.{_OMITTED_FAMILY_SET}", 10)])
    _out, problems = B.derive(audit, lane)
    assert any("summary (a set left out) 9" in p for p in problems), problems


def test_the_omitted_family_rule_is_the_notebooks_list():
    assert all(CA.omitted_family_set(e) for e in R.OMITTED_EVALS) and CA.omitted_family_set(_OMITTED_FAMILY_SET)
    assert not [s["set_id"] for s in CA.load_corpus_audit()["overlap"]["sets"] if CA.omitted_family_set(s["set_id"])]
    assert not CA.omitted_family_set("strongreject_dataset") and not CA.omitted_family_set("petri_auditor_turns")
