"""The StrongREJECT re-judge's goal table (scripts/rejudge/strongreject/goals.py): every HarmBench standard test behaviour plus the
stratified 100 JBB harmful behaviours, ``static`` and ``mt`` judging every goal and ``bon`` the JBB ones, written as the bytes the paper's
scope used. Offline: synthetic benchmark files with placeholder texts; one test reads the pinned files from a local Hugging Face cache
when they are there (and holds their texts in memory only)."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rejudge.strongreject import goals as G  # noqa: E402
from sisteer.eval import jailbreak_data as JD  # noqa: E402

#: sha256 of the goals.json the paper's StrongREJECT scope was built from
PAPER_GOALS_SHA256 = "ef124cbdf8948ead8df509b4abcf6d810a777426ee39c8defd7e8058fcc5b116"


def _harmbench(path, n_standard=159, extra_ids=()):
    rows = [{"BehaviorID": f"hb_standard_{i:03d}", "Behavior": f"placeholder behaviour {i}", "FunctionalCategory": "standard",
             "SemanticCategory": f"cat{i % 6}", "ContextString": None} for i in range(n_standard)]
    rows += [{"BehaviorID": f"hb_context_{i}", "Behavior": f"placeholder contextual {i}", "FunctionalCategory": "contextual",
              "SemanticCategory": "cat0", "ContextString": "placeholder context"} for i in range(3)]
    rows += [{"BehaviorID": f"hb_copyright_{i}", "Behavior": f"placeholder copyright {i}", "FunctionalCategory": "copyright",
              "SemanticCategory": "copyright", "ContextString": None} for i in range(2)]
    rows += [{"BehaviorID": b, "Behavior": "placeholder", "FunctionalCategory": "standard", "SemanticCategory": "cat0", "ContextString": None}
             for b in extra_ids]
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def _jbb(path, n=100):
    rows = [{"Index": i + 1, "Goal": f"placeholder goal {i + 1}", "Category": f"category {i % 10}", "Source": "synthetic"} for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_the_table_is_every_standard_behaviour_and_the_stratified_jbb_set(tmp_path):
    hb, jbb = _harmbench(tmp_path / "hb.parquet"), _jbb(tmp_path / "jbb.csv", n=120)
    t = G.build(hb, jbb)
    goals = t["goals"]
    assert len(goals) == 259
    assert sum(g["dataset"] == "harmbench" for g in goals.values()) == 159
    assert {g["revision"] for g in goals.values() if g["dataset"] == "harmbench"} == {JD.HARMBENCH_REVISION}
    assert {g["revision"] for g in goals.values() if g["dataset"] == "jbb"} == {JD.JBB_REVISION}
    assert not any(k.startswith(("hb_context_", "hb_copyright_")) for k in goals)          # standard only, copyright excluded
    want_bon = sorted(g.goal_id for g in JD.stratify_jbb(JD.load_jbb("harmful", path=jbb), n=100, seed=0))
    assert t["goal_sets"] == {"static": sorted(goals), "mt": sorted(goals), "bon": want_bon}
    first = want_bon[0]
    assert goals[first] == {"dataset": "jbb", "revision": JD.JBB_REVISION, "text": f"placeholder goal {int(first.rsplit('_', 1)[1])}"}
    assert len(set(goals) & {f"jbb_harmful_{i}" for i in range(1, 121)}) == 100                # 20 JBB rows left out


def test_the_file_is_the_scope_bytes_and_its_sha_is_printed(tmp_path, capsys):
    hb, jbb = _harmbench(tmp_path / "hb.parquet"), _jbb(tmp_path / "jbb.csv")
    out = tmp_path / "scope" / "goals.json"
    assert G.main(["--out", str(out), "--harmbench-parquet", hb, "--jbb-csv", jbb]) == 0
    raw = out.read_bytes()
    assert raw == json.dumps(G.build(hb, jbb), indent=0, sort_keys=True).encode() and not raw.endswith(b"\n")
    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == hashlib.sha256(raw).hexdigest()
    assert printed["goals"] == 259 and printed["goal_sets"] == {"static": 259, "mt": 259, "bon": 100}


def test_another_count_or_a_duplicate_id_is_refused(tmp_path):
    jbb = _jbb(tmp_path / "jbb.csv")
    with pytest.raises(SystemExit, match="158 HarmBench"):
        G.build(_harmbench(tmp_path / "short.parquet", n_standard=158), jbb)
    with pytest.raises(SystemExit, match="appears twice"):
        G.build(_harmbench(tmp_path / "dup.parquet", n_standard=158, extra_ids=("jbb_harmful_1",)), jbb)


def test_the_pinned_files_rebuild_the_papers_goal_table():
    """Reads the pinned revisions from a local Hugging Face cache (no download); skipped where they are not cached."""
    hf = pytest.importorskip("huggingface_hub")
    hb = hf.try_to_load_from_cache(JD.HARMBENCH_REPO, JD.HARMBENCH_FILES["test"], revision=JD.HARMBENCH_REVISION, repo_type="dataset")
    jbb = hf.try_to_load_from_cache(JD.JBB_REPO, JD.JBB_FILES["harmful"], revision=JD.JBB_REVISION, repo_type="dataset")
    if not (isinstance(hb, str) and isinstance(jbb, str)):
        pytest.skip("the pinned HarmBench / JBB files are not in the local Hugging Face cache")
    raw = json.dumps(G.build(hb, jbb), indent=0, sort_keys=True).encode()
    assert hashlib.sha256(raw).hexdigest() == PAPER_GOALS_SHA256


def test_the_documented_sequence_builds_the_scope_around_the_goal_table_it_holds(tmp_path):
    """The READMEs' first two steps, verbatim in shape: ``goals.py --out SCOPE/goals.json`` and then ``scope_runs.py --goals
    SCOPE/goals.json --out SCOPE``. The scope keeps the goal table where it is (copying a file onto itself once crashed the second
    step), writes its jobs beside it, and the enumeration reads them with that table."""
    from rejudge.strongreject import enumerate_units as E
    from rejudge.strongreject import scope_runs as SR
    hb, jbb = _harmbench(tmp_path / "hb.parquet"), _jbb(tmp_path / "jbb.csv")
    scope = tmp_path / "SCOPE"
    goals = scope / "goals.json"
    assert G.main(["--out", str(goals), "--harmbench-parquet", hb, "--jbb-csv", jbb]) == 0
    before = goals.read_bytes()
    ids = sorted(json.loads(before)["goals"])[:2]
    cell = tmp_path / "stores" / "static_cell"
    (cell / "shard00").mkdir(parents=True)
    (cell / "shard00" / "responses.jsonl").write_text("".join(json.dumps({"goal_id": g, "trial": 0, "sample_idx": v,
                                                                          "response": f"synthetic reply {v}"}) + "\n"
                                                              for g in ids for v in range(2)))
    runs = tmp_path / "RUNS.tsv"
    runs.write_text("run_id\teval_id\tstore\tcampaign\n" f"synth:codeattack:1\tcodeattack\t{cell}\tsynthcamp\n")
    assert SR.main(["--runs", str(runs), "--goals", str(goals), "--out", str(scope), "--no-registry",
                    "--site", f"synth={tmp_path}"]) == 0
    assert goals.read_bytes() == before
    jobs = [json.loads(line) for line in (scope / "jobs.jsonl").read_text().splitlines()]
    assert [j["run_id"] for j in jobs] == ["synth:codeattack:1"]
    E._init(str(goals))
    rep = E.process(jobs[0])[0]
    assert rep["units"] == 4 and rep["unknown_goal"] == 0
