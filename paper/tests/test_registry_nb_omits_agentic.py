"""The agentic-misalignment evals are omitted from the paper (the owner, 2026-11-14).

  * no category, layout, table spec, label or exclusion constant of the notebook's helper names agentic_leaking, agentic_murder or
    the "agentic misalignment" category, for any model
  * the generator, whose cells are the rendered notebook (its code, captions, figure names and table labels), names neither eval nor
    the category, and its load cell drops their rows from the three query frames and the runs census before `spell`, so no judge
    selection (`select_judges`, under any REGISTRY_NB_JUDGES policy, `rejudge` included) or later reader sees them
  * the README's figure map, which the judge comparison derives its figure list from, names none of them, and neither does the
    committed rendered notebook (its sources and text outputs: tables, captions, printed lines)
  * no committed file under data/ names them, nor carries the word, with no exemption: the training-corpus audit's builder leaves the two
    evals' scan sets out of its sidecar and keeps only their count and maxima
  * `drop_omitted_evals` removes exactly those two evals' rows, every model's, and nothing else
  * a figure an earlier render wrote and this one does not draw (the retired figA_agentic_* files in an existing checkout) is removed
    from figures/ before the zip, so it is never zipped, listed in the Record cell or copied to the figures stage's output
The tool-use and capability evals stay: BFCL is still on the capability axis.
"""
import json
import os
import pathlib
import re
import sys

import pandas as pd

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import judge_compare as JC  # noqa: E402
import registry_nb as R  # noqa: E402

OMITTED = {"agentic_leaking", "agentic_murder"}
NAMES = re.compile(r"agentic|leaking|murder", re.I)                       # the helper's constants: no word of them at all
# what a rendered surface may not name: either eval, the category, or a figure or table of theirs
GONE = re.compile(r"agentic_(?:leaking|murder|misalignment)|agentic misalignment|figA_agentic|tab:[\w-]*agentic", re.I)


def _strings(obj):
    """Every string inside `obj`: dict keys and values, list / tuple / set / frozenset members, nested."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _strings(v)


def _naming(obj) -> list:
    return sorted({s for s in _strings(obj) if NAMES.search(s)})


def test_the_omitted_evals_are_the_two_agentic_scenarios():
    assert R.OMITTED_EVALS == frozenset(OMITTED)


def test_no_category_layout_table_spec_or_label_names_them():
    surfaces = {
        "CATEGORIES": R.CATEGORIES, "APPENDIX_CATEGORIES": R.APPENDIX_CATEGORIES, "TRANSFER_CATEGORIES": R.TRANSFER_CATEGORIES,
        "categories_for": {m: R.categories_for(m) for m in R.MODEL_ORDER}, "ctlstop_categories": R.ctlstop_categories(R.MODEL_ID),
        "TRANSFER_LAYOUT": R.TRANSFER_LAYOUT, "NEGATIVE_DOSE_LAYOUT": R.NEGATIVE_DOSE_LAYOUT, "CTLSTOP_GRID": R.CTLSTOP_GRID,
        "LEVEL": R.LEVEL, "CATEGORY_SHORT": R.CATEGORY_SHORT, "EVAL_CATEGORY": R.EVAL_CATEGORY, "POOLED_EVAL": R.POOLED_EVAL,
        "EVAL_LABEL": R.EVAL_LABEL, "TICK": R.TICK, "AT_CAP_HIDDEN": R.AT_CAP_HIDDEN, "DECODE_NOTE_CATEGORIES": R.DECODE_NOTE_CATEGORIES,
        "OUT_BY_DESIGN": R.OUT_BY_DESIGN, "EXCLUDED_CELLS": R.EXCLUDED_CELLS, "NOT_A_GAP": R.NOT_A_GAP,
        "FRONTIER": (R.FRONTIER_X, R.FRONTIER_Y, R.FRONTIER_STATIC),
    }
    named = {k: _naming(v) for k, v in surfaces.items() if _naming(v)}
    assert not named, named
    assert all(R.category_of(e) is None for e in OMITTED)
    assert "bfcl" in R.CATEGORIES["capability"][0] and "bfcl" in R.APPENDIX_CATEGORIES["capability"][0], "the tool-use eval stays"


def test_the_generator_names_neither_eval_nor_the_category():
    """Its cells are the rendered notebook, so no figure name, table label, caption or printed line can carry them either."""
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    hits = [f"{i}: {ln.strip()[:100]}" for i, ln in enumerate(src.splitlines(), 1) if GONE.search(ln)]
    assert not hits, hits


def test_the_committed_notebook_names_none_of_them():
    """The committed render: every cell's source and every text output (images are not text and are not read)."""
    nb = json.loads((NOTEBOOKS / "registry_qwen_figures.ipynb").read_text(encoding="utf-8"))
    texts = []
    for c in nb["cells"]:
        texts.append("".join(c["source"]))
        for o in c.get("outputs", []):
            texts.append("".join(o.get("text", "")))
            texts += ["".join(v) if isinstance(v, list) else str(v) for k, v in o.get("data", {}).items() if not k.startswith("image/")]
    hits = sorted({m.group(0) for t in texts for m in GONE.finditer(t)})
    assert not hits, hits


def test_the_load_cell_drops_them_from_every_frame_before_anything_reads_them():
    """At the query, before any judge reads them: the re-derivation (`spell`), the synthesis of rows from a re-judge's items and the
    JUDGE_REPORT they fill never see an omitted eval's row, so none is counted as re-derived or drawn."""
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    load = src[src.index('code("ec6aacd8"'):src.index('md("9bf536ac"')]
    drop = '_raw = {k: R.drop_omitted_evals(R.query(q, k))[0] for k, q in '
    assert drop in load and load.index(drop) < load.index("R.synthesized_rows(") < load.index("R.spell(") < load.index("R.dedupe(_raw[k])")
    assert 'runs_all = R.spell(R.drop_omitted_evals(R.query(RUNS_SQL, "runs_all"))[0])' in load


def test_a_rejudge_render_never_hands_the_omitted_evals_to_judge_selection():
    """The load cell's own load lines, run against a stub registry whose every query returns both evals beside two others: neither
    the synthesis nor `spell` (which ends in `select_judges`, where a `rejudge` render re-derives and reports rows) is handed either
    eval, on any frame."""
    import types
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    load = src[src.index('code("ec6aacd8"'):src.index('md("9bf536ac"')].splitlines()
    lines = [ln for ln in load if ln.startswith(("_raw = ", "_syn = ", "runs_all = R.spell("))]
    assert len(lines) == 4, lines
    handed = {}

    def query(sql, key):
        f = pd.DataFrame([dict(model_id=m, eval_id=e) for m in R.MODEL_ORDER
                          for e in ("agentic_leaking", "agentic_murder", "xstest", "pasttense")])
        f.attrs["head"] = 3
        return f.assign(_key=key)

    def spell(d):
        handed[d["_key"].iat[0]] = set(d.eval_id)
        return d

    def synthesized_rows(raw):
        handed["synthesis"] = set(raw.eval_id)
        return raw.iloc[0:0]

    ns = dict(R=types.SimpleNamespace(query=query, spell=spell, drop_omitted_evals=R.drop_omitted_evals, synthesized_rows=synthesized_rows,
                                      with_synthesized=lambda v, syn, k: v),
              PAPER_SQL="p", FRONTIER_SQL="f", POOL_SQL="u", RUNS_SQL="r")
    for ln in lines:
        exec(ln, ns)
    assert handed == {k: {"xstest", "pasttense"} for k in ("paper_all", "qwen_frontier", "qwen_unsteered", "runs_all", "synthesis")}, handed
    assert ns["runs_all"].attrs["head"] == 3 and all(f.attrs["head"] == 3 for f in ns["_raw"].values())


def test_unsave_stale_removes_an_earlier_renders_files_and_keeps_this_renders(tmp_path, monkeypatch):
    """The retired agentic figures of a checkout that rendered before: gone from FIGS once the zip cell runs, whatever saved them;
    every file this render wrote or rewrote stays (the rewrite made immediately after the snapshot, so no clock is involved), and so
    do files that are not figures or tables (the zip, a note)."""
    monkeypatch.setattr(R, "FIGS", str(tmp_path))
    old = ["figA_agentic_delta_models.png", "figA_agentic_delta_models.pdf", "figA_agentic_absolute_models.svg", "figA_x_seeds.tex"]
    redrawn = ["fig5_misalignment.png", "figA_qwen_ctlstop_static_seeds.tex"]
    for f in old + redrawn + ["paper_figures.zip", "notes.txt"]:
        (tmp_path / f).write_text("earlier render")
        os.utime(tmp_path / f, (1_000_000_000, 1_000_000_000))
    before = R.figs_state()
    assert before[0] == str(tmp_path) and set(before[1]) == set(old + redrawn)
    for f in redrawn + ["fig5_misalignment.pdf"]:                     # this render: two files redrawn, one new
        (tmp_path / f).write_text("this render")
    assert R.unsave_stale(before) == sorted(old)
    assert sorted(os.listdir(tmp_path)) == ["fig5_misalignment.pdf", "fig5_misalignment.png", "figA_qwen_ctlstop_static_seeds.tex",
                                            "notes.txt", "paper_figures.zip"]
    # the default is the directory as the module found it at import; a snapshot of another directory removes nothing
    (tmp_path / "figA_agentic_delta_models.png").write_text("earlier render")
    monkeypatch.setattr(R, "FIGS_AT_IMPORT", R.figs_state())
    assert R.unsave_stale() == ["fig5_misalignment.pdf", "fig5_misalignment.png", "figA_agentic_delta_models.png",
                                "figA_qwen_ctlstop_static_seeds.tex"]
    (tmp_path / "keep.png").write_text("x")
    assert R.unsave_stale((str(tmp_path / "elsewhere"), {"keep.png": os.stat(tmp_path / "keep.png").st_mtime_ns})) == []
    monkeypatch.setattr(R, "FIGS", str(tmp_path / "absent"))
    assert R.figs_state() == (str(tmp_path / "absent"), {}) and R.unsave_stale((str(tmp_path / "absent"), {})) == []


def test_the_zip_cell_removes_stale_files_before_it_zips_and_the_stage_prunes_its_output():
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    zcell = src[src.index('code("b7d4e9f2"'):src.index('md("92fdc8e2"')]
    assert "R.unsave_stale()" in zcell and zcell.index("R.unsave_stale()") < zcell.index("zipfile.ZipFile(")
    assert src.index("import registry_nb as R") < src.index('code("b7d4e9f2"'), "the import-time snapshot precedes every figure"
    stage = (NOTEBOOKS.parents[1] / "reproduce" / "10_figures.sh").read_text(encoding="utf-8")
    cp = next(ln for ln in stage.splitlines() if "cp -R '$NB_FIGS/.'" in ln)
    assert "rm -f --" in cp and cp.index("rm -f --") < cp.index("cp -R")


def test_the_readme_figure_map_names_none_of_them():
    fmap = JC.figure_map()
    named = {name: e for name, e in fmap.items()
             if GONE.search(name) or GONE.search(e["what"]) or _naming(e.get("layout_categories") or [])}
    assert not named, named
    assert {"fig5_misalignment", "figA_misalignment_models", "figA_transfer_models"} <= set(fmap)


def test_drop_omitted_evals_removes_exactly_their_rows_on_every_model():
    f = pd.DataFrame([dict(model_id=m, eval_id=e, object_label="safety_si")
                      for m in R.MODEL_ORDER for e in ("agentic_leaking", "agentic_murder", "bfcl", "petri_pooled", "pasttense")])
    f.attrs["head"] = 7
    kept, dropped = R.drop_omitted_evals(f)
    assert set(dropped.eval_id) == OMITTED and len(dropped) == 2 * len(R.MODEL_ORDER)
    assert set(kept.eval_id) == {"bfcl", "petri_pooled", "pasttense"} and len(kept) == 3 * len(R.MODEL_ORDER)
    assert kept.attrs.get("head") == 7, "the frame's head rides through the drop (the load cell reads it after)"
    empty = pd.DataFrame(columns=["model_id"])
    assert R.drop_omitted_evals(empty)[0] is empty and len(R.drop_omitted_evals(empty)[1]) == 0


# NO EXEMPTION UNDER data/: the training-corpus audit's overlap scan read the corpora against every fixed prompt set the repository held
# when it ran, the two omitted evaluations' sets among them; its builder leaves those sets out (appendix_corpus_audit.omitted_family_set,
# keyed on R.OMITTED_EVALS, so the builder names neither), keeping only their count and maxima, and the audit's claim and Table 3 are the
# same without them. The paper policy's judge table (data/paper_judges.json) names no judge of the category since the policy dropped it
# (R.JUDGE_POLICIES["rejudge"]). So every committed file under data/ is held to the strictest rule: not even the word.
WORD = re.compile(r"\w*agentic\w*", re.I)


def _data_text(repo, f: str) -> str:
    """A committed data file's text as the scan reads it: a Parquet file (the strength-read projections) as a table, its column names,
    schema metadata and every string value, never its bytes decoded; any other file as UTF-8 text."""
    path = repo / f
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        parts = list(t.schema.names) + [repr(t.schema.metadata)] + [str(fld.type) for fld in t.schema]
        for c in t.column_names:
            parts += sorted({v for v in t.column(c).to_pylist() if isinstance(v, str)})
        return "\n".join(parts)
    return path.read_text(encoding="utf-8")


def test_the_committed_data_files_name_none_of_them():
    """Every committed file under data/ with no exemption (the sidecars the appendix cells read, the scripts that write them, the paper
    policy's judge table and the strength-read projections, a Parquet file read as a table by `_data_text`): no name of either eval, the
    category, or a figure or table of theirs, and not even the word."""
    import subprocess
    repo = NOTEBOOKS.parents[1]
    files = subprocess.run(["git", "ls-files", "paper/notebooks/data"], cwd=repo, capture_output=True, text=True, check=True).stdout.split()
    assert "paper/notebooks/data/appendix_corpus_audit.json" in files and "paper/notebooks/data/build_appendix_corpus_audit.py" in files
    assert "paper/notebooks/data/paper_judges.json" in files and any(f.endswith(".parquet") for f in files)
    hits, read = {}, 0
    for f in files:
        try:
            text = _data_text(repo, f)
        except UnicodeDecodeError:
            continue
        read += 1
        found = sorted({m.group(0) for m in GONE.finditer(text)} | {m.group(0) for m in WORD.finditer(text)})
        if found:
            hits[f] = found
    assert not hits, hits
    assert read == len(files), "a committed data file that is not text is not scanned"
    assert "agentic" not in R.JUDGE_POLICIES["rejudge"], "the paper policy names a judge of the category again"


def test_the_corpus_audit_leaves_the_omitted_evals_scan_sets_out_and_counts_them():
    """Its two scan sets: not listed and their numbers not copied, their count and maxima kept, and none of them over a threshold."""
    import appendix_corpus_audit as CA
    ca = CA.load_corpus_audit()
    left = ca["overlap"]["sets_left_out"]
    assert (left["n_sets"], left["records_over_threshold"]) == (2, 0) and left["max_run"] < ca["overlap"]["thresholds"]["route_run"]
    assert not [s for s in ca["overlap"]["sets"] if CA.omitted_family_set(s["set_id"])]
    assert all(CA.omitted_family_set(e) for e in R.OMITTED_EVALS)
