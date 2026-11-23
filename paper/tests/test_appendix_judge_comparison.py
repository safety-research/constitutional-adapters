"""appendix_judge_comparison: the judge-robustness figure of appendix D (`fig:app_judges`, figA_judge_comparison_a and _b), drawn from the
committed frozen readings `data/appendix_judge_comparison.json`. No registry: the committed file, its builder read as a module (it imports
the standard library only), and synthetic mutations of the file.

Pinned here:
  * the file matches its builder's provenance: it names the builder and carries the committed builder's sha256, code commit, judging
    policies, pinned head and strength-lane sidecars; its readings hash to the recorded payload_sha256, which is pinned here with the
    registry copy's sha256 and the delivered value files' sha256s (a rebuild that changes any of them changes the pin in the same
    commit); every row the registry copy took
    in after the pinned head is bookkeeping or a judgment of a run outside the readings; the build compared the marks with the value files
    written beside the delivered figure and found every one equal; `problems` names each way the file can be broken, which `load` turns
    into a refusal; the builder's pin (only bookkeeping and judgments of excluded runs after the pinned head), its exact value check and
    its writer, on synthetic inputs;
  * the draw reproduces the delivered figure: the decoded pixels of each part's PNG hash to the delivered PNG's (its RGBA array's shape and
    sha256; PDFs carry a creation date and are not compared). With JUDGE_COMPARISON_DELIVERED naming a directory that holds the delivered
    PNGs, the two are also compared pixel by pixel;
  * the numbers the paragraph beside the figure quotes (app:judges), recomputed from the file, and the lines the cell prints;
  * the notebook: one markdown and one code cell after e6f1b5a9, the code cell drawing both parts from the committed file alone, and the
    two parts in PAPER_FIGURES right before the generator comparison; the README names the figure, its file and its builder;
  * no private strings: no absolute or home path, and none of the names of the scratch places, renders and sessions the readings were
    made in (held here as the publication scan's hash rows, so this file never spells them), in the module, the builder, the file, the
    cells, the README's section and the two test files; and the publication scan, without its allowlist, finds nothing in the module,
    the builder, the file or the two test files.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
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

import registry_nb as R  # noqa: E402
import appendix_judge_comparison as JC  # noqa: E402

BUILDER_PATH = REPO / JC.BUILDER
GENERATOR = NOTEBOOKS / "make_registry_notebook.py"
README = NOTEBOOKS / "README.md"
#: the delivered figure's PNGs, decoded to RGBA: (height, width) and the sha256 of the pixel array
DELIVERED_PIXELS = {"a": ((2750, 3303), "b5d8f92793c6fd493bab2786fe72cd1068b933f81174043c048112918ac69205"),
                    "b": ((2127, 3303), "fc7d619becf3c98916d5d5354c5a16696800dda2591697d953d57b700dbf5d29")}
DELIVERED_ENV = "JUDGE_COMPARISON_DELIVERED"
#: the frozen identity of the readings: the payload hash of the three blocks, the sha256 of the registry copy they were read from, and
#: the sha256s of the value files written beside the delivered figure (part a, part b), which the build compared every mark with
PAYLOAD_SHA256 = "8e1bd7b5a21245343e7324c78fd17b821b370faba5333f930706da8a48b5262a"
REGISTRY_COPY_SHA256 = "40bcc64121b18792ba4549f927c5657b19121adea496eb3dfd316dba0d241ef7"
DELIVERED_VALUES_SHA256 = ("c87598e16cb063b38b7caf5785d2af402dc1b6332adbc87d9165cc8f9dbbecdb",
                           "2e70546e623678058ed2578d05608983699bf4f2088e517e90c536756e1d9e27")
#: the numbers app:judges quotes: rank correlation at least 0.70 in all 20 model x evaluation-category columns and at least 0.90 in 15;
#: the undefended model's static attack success 23-54% under the Claude judges, 16-17% under the paper's; the Difference LoRA's multi-turn
#: gain +30 rather than +58 pp
QUOTED = dict(rank_columns=20, rank_columns_ge_0_70=20, rank_columns_ge_0_90=15,
              static_unsteered_asr_pct={"original": (23, 54), "new": (16, 17)},
              multi_turn_difference_lora_pp={"original": 30, "new": 58})
NEW_CELLS = (("4d9c2b7a", "md"), ("8f3e61c5", "code"))
#: the names of the scratch places, renders and sessions the readings were made in, which the shipped files must never carry, as rows
#: of the publication scan's token table (reproduce/tools/scrub_scan.py: a name is the sha256 of its lowercase runs, built with
#: scrub_scan.hash_row outside the repository, so this file never spells one), and the shapes of a scratch or worktree path
PRIVATE_ROWS = (
    ("process", "hash", "a0a5d26733a244e80281b8437c60e238501f50308b44510e98f6ed2a2f557202", {}),
    ("process", "hash", "97fbcc43e7043ec0125f288fc7bcdaafa6402d239896b6189317039da8352262",
     {"sep": "_", "n": 2, "head": "986a1b7135f4986150aa5fa0028feeaa66cdaf3ed6a00a355dd86e042f7fb494"}),
    ("process", "hash", "2abe1dc6e51248910afe10f65f5c90d13c273f8263e84c3267007471011aaceb",
     {"sep": "_", "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "c8d6cd85f0ba05b4fd6e3142bcd788d527796e1ad9f9e7c5e8d3543359ae2aa4",
     {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "79d33728507dab3c0d82fb6c5d919bbc58852f61b14a8dbd179a3635ee33cd3d",
     {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "16275f66ef64f596d57218ead3188ca9813999080a591b9c999e611602589cd0",
     {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "df67ed21dcfe14b31bc6cf3e497db36a1f963b52fd776047192f263fc1f5b4c9",
     {"sep": ["-", "_"], "n": 2, "head": "d90ee9ccf6bea1d2942a7b21319338198dec2a746f8a0d0771621f00da2e0864"}),
    ("process", "hash", "77c4baeef64b526daa8f6f22c909ddc0b14c5db5e6ee3c3c55e72d98986254cb",
     {"sep": ["-", "_"], "n": 3, "head": "10e86c6514d40f2a3e861b31847340ee8c8ed181029a17b042f137121d28863e"}),
    ("process", "hash", "71933c5947554324f0028884d0d2fb781394b3f139af2b397310ae11b7f00049",
     {"sep": ["-", "_"], "n": 3, "head": "10e86c6514d40f2a3e861b31847340ee8c8ed181029a17b042f137121d28863e"}),
    ("process", "hash", "00be9a4754d56949444569c33c2f1fdcb3ea6d79c0af3fc8b44154b520ec792a", {}),
    ("process", "hash", "b168981dff6b374a4d5b0ca2f17d457e5abc7e6bcd311d8f6a620a9115e4b990",
     {"sep": ".", "n": 2, "head": "594cfd2607d4f09e1ce6241203e418cacfed545063792bc5eab7afaadbf0c4e0"}),
    ("process", "shape", r"tm[p]/", {"seed": "tm[p]/"}),
    ("process", "shape", r"\bworktree[s]\b", {"seed": "worktree[s]"}),
)


@pytest.fixture(scope="module")
def jc():
    return JC.load()


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_appendix_judge_comparison", BUILDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _no_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIGS", str(tmp_path / "figs"))
    yield
    plt.close("all")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _problems_after(jc, mutate) -> list:
    d = copy.deepcopy(jc)
    mutate(d)
    return JC.problems(d)


def _cells() -> dict:
    src = GENERATOR.read_text(encoding="utf-8")
    return {cid: (kind, body.strip("\n"))
            for kind, cid, body in re.findall(r'^(md|code)\("([0-9a-f]{8})", r"""(.*?)"""\)', src, flags=re.M | re.S)}


# ---- 1. the file and its builder ---------------------------------------------------------------------------------------------------
def test_the_file_matches_its_builders_provenance(jc, builder):
    pv = jc["provenance"]
    assert jc["schema"] == JC.SCHEMA == builder.SCHEMA and pv["script"] == JC.BUILDER == builder.SCRIPT
    assert pv["script_sha256"] == _sha256(BUILDER_PATH), "the builder changed after it wrote the file: rebuild it"
    assert pv["payload_sha256"] == JC.payload_sha256(jc) == PAYLOAD_SHA256, "the readings are not the frozen ones"
    assert pv["registry"]["copy_sha256"] == REGISTRY_COPY_SHA256, "the file was not read from the frozen registry copy"
    assert builder.MARK_COLUMNS == JC.MARK_COLUMNS and builder.ANCHOR_COLUMNS == JC.ANCHOR_COLUMNS and builder.MEAN_COLUMNS == JC.MEAN_COLUMNS
    assert builder.AS_OF == JC.AS_OF == pv["registry"]["as_of"] == 537565
    assert builder.POLICIES == JC.POLICY == pv["frames"]["policies"]
    assert pv["frames"]["code_commit"] == builder.CODE_COMMIT and len(builder.CODE_COMMIT) == 40
    assert pv["frames"]["cells"] == list(builder.CELLS) and pv["frames"]["statements"] == list(builder.FRAME_STATEMENTS)
    sr = pv["frames"]["strength_reads"]
    assert sr["projects_sidecars_sha256"] == sorted(builder.STRENGTH_READS_SHA256) and len(sr["projection_sha256"]) == 2
    assert len(pv["frames"]["registry_nb_sha256"]) == 64 and len(pv["registry"]["copy_sha256"]) == 64
    assert pv["registry"]["opened"] == "read-only" and pv["git_dirty"] is False


def test_the_readings_are_those_at_the_pinned_head(jc, builder):
    reg = jc["provenance"]["registry"]
    assert reg["head_in_copy"] >= reg["as_of"]
    assert set(reg["head_read_by_frames"].values()) == {reg["head_in_copy"]}
    after = reg["rows_after_as_of"]
    assert set(after) == set(builder.BOOKKEEPING_TABLES) & set(after) | set(reg["run_scoped_tables"]), after
    assert not set(reg["run_scoped_tables"]) & set(builder.BOOKKEEPING_TABLES)
    runs = reg["runs_after_as_of"]
    assert runs["in_readings"] == 0 and runs["selection_state"] == "excluded"
    assert (runs["n"] > 0) == bool(reg["run_scoped_tables"])


def test_the_marks_equal_the_delivered_figures_values(jc):
    chk = jc["provenance"]["checks"]["delivered_values"]
    assert chk["equal"] == sum(f["n_values"] for f in chk["files"]) == len(jc["marks"]["rows"]) == sum(JC.N_MARKS.values())
    assert [f["n_values"] for f in chk["files"]] == [JC.N_MARKS["a"], JC.N_MARKS["b"]]
    assert tuple(f["sha256"] for f in chk["files"]) == DELIVERED_VALUES_SHA256, "the marks were checked against other value files"


def test_the_judges_are_the_filed_claude_judges_and_the_papers(jc):
    m = JC.frame(jc, "marks")
    for j in JC.JUDGES:
        assert jc["judges"][j]["policy"] == JC.POLICY[j]
        for cat in JC.JUDGED:
            ids = jc["judges"][j]["judge_ids"][cat]
            assert ids and all(re.fullmatch(r"[0-9a-f]{64}", i) for i in ids), (j, cat, ids)
            got = set(m[(m.judge == j) & (m.category == cat) & (m["eval"] != R.MEAN_COLUMN)].judge_model)
            assert got == {JC.JUDGE_MODEL[j][cat]}
    assert not set(jc["judges"]["original"]["judge_ids"]["static"]) & set(jc["judges"]["new"]["judge_ids"]["static"])
    assert JC.problems(jc) == []


@pytest.mark.parametrize("mutate, phrase", [
    (lambda d: d.update(schema="appendix_judge_comparison/0"), "schema"),
    (lambda d: d["marks"]["rows"][0].__setitem__(5, d["marks"]["rows"][0][5] + 1e-9), "payload_sha256"),
    (lambda d: d["provenance"].update(script="paper/notebooks/other.py"), "not the builder"),
    (lambda d: d["provenance"]["registry"].update(as_of=537711), "as_of"),
    (lambda d: d["provenance"]["registry"].update(note="/".join(("", "home", "someone", "registry.sqlite"))), "absolute or home path"),
    (lambda d: d["marks"]["rows"].pop(), "marks:"),
    (lambda d: d["marks"]["rows"].__setitem__(0, d["marks"]["rows"][0][:5]), "wrong length"),
    (lambda d: d["static_anchors"]["rows"].pop(), "static_anchors"),
    (lambda d: d["mean_columns"]["rows"].__setitem__(slice(None), [r for r in d["mean_columns"]["rows"]
                                                                  if not (r[1] == "llama8b_it" and r[2] == "Petri")]), "mean_columns"),
])
def test_each_way_to_break_the_file_is_named(jc, mutate, phrase):
    p = _problems_after(jc, mutate)
    assert p and any(phrase in x for x in p), p


def test_a_wrong_judge_or_a_broken_interval_is_named(jc):
    def judge(d):
        r = next(r for r in d["marks"]["rows"] if r[0] == "new" and r[1] == "Petri" and r[2] != R.MEAN_COLUMN)
        r[12] = "claude-opus-5"
    def interval(d):
        r = next(r for r in d["marks"]["rows"] if r[6] is not None)
        r[6], r[7] = r[5] + 1.0, r[5] + 2.0
    def single(d):
        d["marks"]["rows"] = [r for r in d["marks"]["rows"] if not (r[0] == "new" and r[1] == "static" and r[3] == "constitution_lora"
                                                                     and r[4] == "s1" and r[2] == "pasttense")]
    assert any("judged by" in x for x in _problems_after(jc, judge))
    assert any("does not hold" in x for x in _problems_after(jc, interval))
    assert any("only one judge" in x for x in _problems_after(jc, single))


def test_load_refuses_a_broken_file(jc, tmp_path):
    d = copy.deepcopy(jc)
    d["marks"]["rows"][3][5] = 99.0
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d))
    with pytest.raises(JC.JudgeComparisonDataError, match="payload_sha256"):
        JC.load(p)


def _synthetic_registry(path, after: dict, states: dict):
    """A registry with a record ledger up to AS_OF + 3 and, for each table in `after`, rows ingested at AS_OF (kept) and AS_OF + 1
    (`after[table]` of them, for run `r1`; `ruling` keys its rows by a ruling id, not a run); `states` is v_paper_current's selection
    state per run."""
    import sqlite3
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE record_ingest (file_uri TEXT, ingest_seq INTEGER)")
    con.executemany("INSERT INTO record_ingest VALUES (?, ?)", [(f"f{i}", JC.AS_OF + i) for i in range(-1, 4)])
    for t, n in after.items():
        con.execute(f"CREATE TABLE {t} ({'ruling_id' if t == 'ruling' else 'run_id'} TEXT, ingest_seq INTEGER)")
        con.executemany(f"INSERT INTO {t} VALUES (?, ?)", [("r0", JC.AS_OF)] + [("r1", JC.AS_OF + 1)] * n)
    con.execute("CREATE TABLE v_paper_current (run_id TEXT, selection_state TEXT)")
    con.executemany("INSERT INTO v_paper_current VALUES (?, ?)", list(states.items()))
    con.commit(); con.close()


def test_the_builder_pins_the_head_and_admits_only_bookkeeping_and_excluded_runs_after_it(builder, tmp_path):
    ok = tmp_path / "ok.sqlite"
    _synthetic_registry(ok, {"judgment": 1, "item_result": 3, "mirror_ack": 2}, {"r0": "selected", "r1": "excluded"})
    f = builder.registry_facts(ok)
    assert f["head"] == JC.AS_OF + 3 and f["after_runs"] == ["r1"] and f["run_scoped"] == ["item_result", "judgment"]
    assert f["after"] == {"record_ingest": 3, "judgment": 1, "item_result": 3, "mirror_ack": 2}
    for name, after, states, phrase in (("ruling", {"ruling": 1}, {"r1": "excluded"}, "rows of ruling"),
                                        ("selected", {"judgment": 1}, {"r1": "selected"}, "does not exclude")):
        db = tmp_path / f"{name}.sqlite"
        _synthetic_registry(db, after, states)
        with pytest.raises(SystemExit, match=phrase):
            builder.registry_facts(db)


def test_the_builders_value_check_counts_exact_equality_only(builder, tmp_path):
    marks = [["original", "static", "pasttense", "arditi_allpos", "", 22.5, 17.25, 27.75, False, False, 1, None, "claude-haiku-4-5"],
             ["new", "static", "mean", "constitution_lora", "s1,s2,s3", 10.0, None, None, False, False, 3, 4, "minimax-m3"]]
    vals = [dict(judge="original", category="static", eval="pasttense", object="arditi_allpos", seed=None, delta=22.5, lo=17.25, hi=27.75),
            dict(judge="new", category="static", eval="mean", object="constitution_lora", seed="s1,s2,s3", delta=10.0, lo=None, hi=None)]
    f = tmp_path / "values.json"
    f.write_text(json.dumps({"values": vals}))
    chk = builder.compare_values(marks, [str(f)])
    assert chk["equal"] == 2 and chk["files"][0]["sha256"] == _sha256(f) and chk["files"][0]["n_values"] == 2
    vals[0]["delta"] = 22.5 + 1e-12
    f.write_text(json.dumps({"values": vals}))
    with pytest.raises(SystemExit, match="equal 1 of the 2"):
        builder.compare_values(marks, [str(f)])


def test_the_builders_writer_round_trips_the_file(jc, builder):
    assert json.loads(builder.dumps(jc)) == jc
    assert builder._strip({"provenance": {"generated_utc": "x", "git_commit": "y", "seconds": 3, "keep": 1}}) == {"provenance": {"keep": 1}}


# ---- 2. the draw reproduces the delivered figure -----------------------------------------------------------------------------------
@pytest.mark.parametrize("part", ["a", "b"])
def test_the_draw_reproduces_the_delivered_png_pixel_for_pixel(jc, part, tmp_path):
    from PIL import Image
    fig, info = JC.draw(jc, part)
    name = JC.PARTS[part]["name"]
    for ext in ("png", "pdf", "svg"):
        assert (tmp_path / "figs" / f"{name}.{ext}").is_file()
    got = np.asarray(Image.open(tmp_path / "figs" / f"{name}.png").convert("RGBA"))
    shape, digest = DELIVERED_PIXELS[part]
    delivered = pathlib.Path(os.environ.get(DELIVERED_ENV, "")) / f"{name}.png"
    if os.environ.get(DELIVERED_ENV) and delivered.is_file():
        want = np.asarray(Image.open(delivered).convert("RGBA"))
        assert want.shape == got.shape, (want.shape, got.shape)
        assert int((want != got).any(axis=-1).sum()) == 0, "pixels differ from the delivered PNG"
    assert got.shape[:2] == shape, got.shape
    assert hashlib.sha256(np.ascontiguousarray(got).tobytes()).hexdigest() == digest
    assert len(info["values"]) == JC.N_MARKS[part] and info["missing"] == []


def test_the_draw_does_not_depend_on_what_an_earlier_cell_set(jc, tmp_path):
    from PIL import Image
    JC.draw(jc, "b")
    first = np.asarray(Image.open(tmp_path / "figs" / f"{JC.FIGURES[1]}.png").convert("RGBA")).copy()
    with plt.rc_context({"font.size": 7, "lines.linewidth": 4.0, "figure.dpi": 72}):
        JC.draw(jc, "b")
    again = np.asarray(Image.open(tmp_path / "figs" / f"{JC.FIGURES[1]}.png").convert("RGBA"))
    assert np.array_equal(first, again)


def test_every_drawn_mark_sits_at_its_reading(jc):
    fig, axes, info = JC.figure(jc, "a")
    m = JC.frame(jc, "marks").set_index(["judge", "category", "eval", "object", "seed"])
    for v in info["values"]:
        r = m.loc[(v["judge"], v["category"], v["eval"], v["object"], v["seed"])]
        assert v["y"] == r.y
    offsets = {round(float(c.get_offsets()[0][1]), 9) for ax, _ in axes for c in ax.collections if (c.get_gid() or "").startswith("judges|")}
    assert offsets == {round(float(v["y"]), 9) for v in info["values"]}
    plt.close(fig)


# ---- 3. the numbers the paragraph quotes -------------------------------------------------------------------------------------------
def test_the_numbers_the_appendix_text_quotes(jc):
    t = JC.text_numbers(jc)
    for k, v in QUOTED.items():
        assert t[k] == v, (k, t[k], v)
    assert t["rank_min"] >= 0.70
    rc = JC.rank_correlations(jc)
    assert len(rc) == 20 and set(rc.model) == set(JC.MODELS) and set(rc.category) == set(JC.JUDGED)
    assert (rc.rho >= 0.70).all() and int((rc.rho >= 0.90 - JC.RHO_TOL).sum()) == 15
    recorded = jc["provenance"]["checks"]["text_numbers"]
    assert recorded["rank_columns_ge_0_90"] == 15 and recorded["static_unsteered_asr_pct"] == {"original": [23, 54], "new": [16, 17]}


def test_the_printed_lines_carry_the_quoted_numbers(jc):
    s = JC.summary_text(jc)
    assert "at least 0.70 in 20 of 20 model x evaluation-category columns and at least 0.90 in 15" in s
    assert "23-54% under the Claude judge, 16-17% under MiniMax-M3" in s
    assert "+30 pp under the Claude judge, +58 pp under MiniMax-M3" in s
    line = JC.provenance_line(jc)
    assert "paper/notebooks/data/appendix_judge_comparison.json" in line and "pinned to head 537565" in line


# ---- 4. the notebook and the README ------------------------------------------------------------------------------------------------
def test_the_cells_sit_after_the_all_model_figures_and_draw_both_parts_from_the_file():
    src = GENERATOR.read_text(encoding="utf-8")
    ids = re.findall(r'^(?:md|code)\("([0-9a-f]{8})"', src, flags=re.M)
    i = ids.index("e6f1b5a9")
    assert tuple(ids[i + 1:i + 3]) == tuple(c for c, _ in NEW_CELLS)
    cells = _cells()
    assert [cells[c][0] for c, _ in NEW_CELLS] == [k for _, k in NEW_CELLS]
    code_src = cells["8f3e61c5"][1]
    assert "JC.load()" in code_src and "JC.draw(jc, _part" in code_src and "for _part in JC.PARTS" in code_src
    assert "JC.summary_text(jc)" in code_src and "JC.provenance_line(jc)" in code_src
    assert not re.search(r"\b(R\.query|R\.connect|sqlite3|REGISTRY_ROOT)\b", code_src)
    assert JC.FIGURES == ("figA_judge_comparison_a", "figA_judge_comparison_b")
    zip_cell = src[src.index('code("b7d4e9f2"'):src.index('md("92fdc8e2"')]
    names = re.findall(r'"([^"]+)"', re.search(r"PAPER_FIGURES = \[(.*?)\]", zip_cell, flags=re.S).group(1))
    g = names.index("figA_qwen_generator_ablation")                   # Figure 16's two parts right before Figure 17, the generator comparison
    assert names[g - 2:g + 1] == ["figA_judge_comparison_a", "figA_judge_comparison_b", "figA_qwen_generator_ablation"]
    assert "fig:app_judges" in src or "judge robustness" in cells["4d9c2b7a"][1]


def test_the_readme_names_the_figure_its_file_and_its_builder():
    text = README.read_text(encoding="utf-8")
    assert "`figA_judge_comparison_a`" in text and "`figA_judge_comparison_b`" in text and "fig:app_judges" in text
    assert "data/appendix_judge_comparison.json" in text and "build_appendix_judge_comparison.py" in text
    assert "paper/tests/test_appendix_judge_comparison.py" in text


# ---- 5. no private strings -----------------------------------------------------------------------------------------------------------
TEST_FILES = ("paper/tests/test_appendix_judge_comparison.py", "tests/test_appendix_judge_comparison_gate.py")


def _scrub_scan():
    """reproduce/tools/scrub_scan.py as a module, loaded once under the name `scrub_scan` (as tests/front_door_testlib.py loads it)."""
    if "scrub_scan" not in sys.modules:
        spec = importlib.util.spec_from_file_location("scrub_scan", REPO / "reproduce" / "tools" / "scrub_scan.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["scrub_scan"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["scrub_scan"]


def _readme_section() -> str:
    """The README's lines about this figure: its figure-map row and its section."""
    text = README.read_text(encoding="utf-8")
    i = text.index("## The judge-comparison data")
    j = text.index("\n## ", i + 1)
    return "\n".join([ln for ln in text[:i].splitlines() if "judge_comparison" in ln] + [text[i:j]])


def _shipped() -> dict:
    cells = _cells()
    out = {"module": (NOTEBOOKS / "appendix_judge_comparison.py").read_text(encoding="utf-8"),
           "builder": BUILDER_PATH.read_text(encoding="utf-8"),
           "file": JC.DATA.read_text(encoding="utf-8"),
           "cells": "\n".join(cells[c][1] for c, _ in NEW_CELLS),
           "README": _readme_section()}
    out.update({f: (REPO / f).read_text(encoding="utf-8") for f in TEST_FILES})
    return out


def test_no_path_or_private_name_in_the_shipped_files():
    """Path shapes and the private names, read from the files alone (nothing about the machine the tests run on)."""
    private = _scrub_scan().build_matchers(PRIVATE_ROWS)["tree"]
    assert private.find("x " + "/".join(("", "tm" + "p", "y"))), "the private-name matcher finds nothing"
    for name, text in _shipped().items():
        assert not re.search(r"/home/[\w.-]|\\Users\\[\w.-]", text), name
        assert not re.search(r"(?<![\w$])[~]/", text), name
        hits = sorted({text[s:e] for _r, s, e in private.find(text)})
        assert not hits, (name, len(hits))
    strings = list(JC._strings(JC.load()))
    assert not [s for s in strings if s.startswith(("/", "~"))]


def test_the_publication_scan_finds_nothing_in_the_shipped_files():
    """The publication scan with no allowlist over the module, the builder, the file and the two test files."""
    paths = ["paper/notebooks/appendix_judge_comparison.py", JC.BUILDER, "paper/notebooks/data/appendix_judge_comparison.json", *TEST_FILES]
    assert all((REPO / x).is_file() for x in paths)
    p = subprocess.run([sys.executable, str(REPO / "reproduce" / "tools" / "scrub_scan.py"), "--root", str(REPO), "--grep",
                        "--no-allowlist", *paths], cwd=REPO, capture_output=True, text=True)
    report = (p.stdout + p.stderr).strip()
    assert p.returncode == 0 and report.splitlines()[-1].startswith("scrub: no identifier under"), report
