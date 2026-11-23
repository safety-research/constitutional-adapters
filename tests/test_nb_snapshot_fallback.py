"""The registry notebook reads the committed paper-frame snapshot when the live registry is absent.

`paper/notebooks/registry_nb.py` resolves ONE source at import - `REGISTRY_NB_SOURCE` = live | snapshot | auto (auto: live when
`REGISTRY_ROOT/registry.sqlite` exists, else the snapshot under `REGISTRY_NB_SNAPSHOT_ROOT`) - and every reader in the file goes
through `connect()`, so the snapshot answers the notebook's SQL unchanged. This module pins that rule on a SYNTHETIC registry
(`sisteer.registry.synth`, hostname TESTHOST, invented ids) and a paper-frame snapshot exported from it with the registry's own
exporter (`snapshot.build(scope="paper_frame", dest_root=...)`, the synthetic model added through `model_ids`), and holds the
helpers the sibling tests (`test_nb_fresh_home.py`, `test_nb_snapshot_parity.py`) import: the notebook mirror (a clone-shaped copy
of the two notebook files with `src/` beside them, so the clone-relative `SISTEER_SRC` default resolves and nothing is written into
the checkout) and the fresh in-process import. The snapshot reader the notebook codes against is `snapshot.open_paper_frame` /
`paper_frame_current`; a tree without them skips this module BY NAME rather than failing somewhere inside the notebook.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity
from sisteer.registry import records as RECORDS
from sisteer.registry import snapshot as SN
from sisteer.registry import synth

for _name in ("open_paper_frame", "paper_frame_current", "PAPER_FRAME_TABLES"):
    if not hasattr(SN, _name):
        pytest.skip(f"sisteer.registry.snapshot.{_name} is not in this tree (the paper-frame snapshot lane)", allow_module_level=True)
if "paper_frame" not in SN.SCOPES:
    pytest.skip("sisteer.registry.snapshot.SCOPES has no paper_frame scope (the paper-frame snapshot lane)", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "paper" / "notebooks"
NB_FILES = ("registry_nb.py", "make_registry_notebook.py")
#: the sibling modules the notebook imports when it re-derives a reading (the default policy does): a clone carries them beside it
NB_MODULES = ("judge_rules.py",)
NB_VENV_PYTHON = str(REPO / ".venv" / "bin" / "python")     # the clone's own venv (`uv sync --frozen --extra cpu --group notebook`: nbconvert, nbformat); optional
NB_ENV_KEYS = ("REGISTRY_NB_SOURCE", "REGISTRY_ROOT", "REGISTRY_NB_SNAPSHOT_ROOT", "SISTEER_SRC", "NB_CACHE", "NB_REUSE_CACHE",
               "NB_ALLOW_UNSCALED_HEADLINE")


# THE PAPER POLICY'S JUDGES, SYNTHETIC (the notebook's default policy, `rejudge`, draws its readings from the judges
# paper/notebooks/data/paper_judges.json lists: minimax-m3 on StrongREJECT v1 / v2 and the C/P/I rubrics, glm-5.3 on Petri). A re-judge is
# filed as a SECOND judgment of a judged run, judge, judgment and item rows only, as the real ones are; the frame is cut with these judges'
# items (`export_snapshot`, the paper frame cut's default item judges), so a default render of it passes the load cell's guard
# (`registry_nb.check_policy_judges`). The judge rows carry the paper's judge ids (public: every committed frame lists them) with
# invented rubric hashes; the judgments and items are invented. Test-local: the shared `synth.write_corpus` is called by dozens of tests
# that pin its row counts.
PAPER_JUDGES = json.loads((NOTEBOOKS / "data" / "paper_judges.json").read_text(encoding="utf-8"))
POLICY_JUDGES = [dict(judge_id=j["judge_id"], judge_model=j["judge_model"], rubric_name=j["rubric_name"], rubric_version=j["rubric_version"],
                      rubric_sha256=hashlib.sha256(f"synthetic rubric {j['rubric_name']} {j['rubric_version']}".encode()).hexdigest(),
                      protocol_sha256=synth.SHA256_B, threshold_rule="score > 0.5") for j in PAPER_JUDGES["judges"]]
POLICY_JUDGE_IDS = [j["judge_id"] for j in POLICY_JUDGES]
#: the policy judge that re-judges each synthetic run (its rubric is the corpus's only judged scale: a 0-1 score per item)
REJUDGE_ID = next(j["judge_id"] for j in POLICY_JUDGES if (j["rubric_name"], j["rubric_version"]) == ("strongreject", "v1"))


def policy_judge_records(root: str, run_ids, n_items: int, i0: int = 1000) -> None:
    """The policy judges' dimension rows, and per run a COMPLETE second judgment by REJUDGE_ID with its item rows, written under `root`."""
    dims = [synth.rec("judge", {"judge_id": j["judge_id"], "judge_model": j["judge_model"], "judge_role": "judge",
                                "rubric_name": j["rubric_name"], "rubric_version": j["rubric_version"], "rubric_sha256": j["rubric_sha256"],
                                "protocol_sha256": j["protocol_sha256"], "score_scale": "0-1", "threshold_rule": j["threshold_rule"],
                                "judge_backend": "batch"}, synth.t(i0 + k)) for k, j in enumerate(POLICY_JUDGES)]
    RECORDS.write_records(os.path.join(root, "records", "dimensions", "policy_judges", "000001-dimensions.jsonl"), dims)
    for k, rid in enumerate(run_ids):
        jid = identity.judgment_id(rid, REJUDGE_ID)
        at = i0 + 100 + 20 * k
        judgment = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": REJUDGE_ID, "started_utc": synth.t(at),
                                          "finished_utc": synth.t(at + 1), "n_items_judged": n_items, "status": "complete",
                                          "api_pool": "lowprio"}, synth.t(at + 1))
        d = os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid))
        RECORDS.write_records(os.path.join(d, RECORDS.record_file_name(101, "judgment")), [judgment])
        RECORDS.write_records(os.path.join(d, RECORDS.record_file_name(102, "item_result", gz=True)),
                              synth.item_records(rid, jid, at + 2, n_items, seed=100 + k, metric="strongreject"), gz=True)


def synthetic_registry(tmp_path, n_runs: int = 4, n_items: int = 10, policy_judges: bool = True) -> str:
    """A synthetic registry root, ingested (`tests/test_registry_snapshot.py::_setup`), with the paper policy's judges and a re-judge of
    every run (`policy_judge_records`) unless `policy_judges` is False."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    corpus = synth.write_corpus(root, n_runs=n_runs, n_items=n_items)
    if policy_judges:
        policy_judge_records(root, corpus["run_ids"], n_items)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def live_head(root: str) -> int:
    with sqlite3.connect(f"file:{DB.db_path(root)}?mode=ro", uri=True) as con:
        return int(con.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])


def export_snapshot(root: str, dest_root: str, item_judges=None) -> tuple:
    """(snapshot_id, head) of a paper-frame snapshot of `root` exported under `dest_root` (the synthetic model joins the exported
    model set through `model_ids`; the tree precondition is overridden with a reason, as a test must). The frame carries the policy
    judges' items in the paper's projection when `root` holds them (`synthetic_registry`), as the paper frame cut does by default;
    `item_judges=()` cuts without."""
    os.makedirs(dest_root, exist_ok=True)
    if item_judges is None:
        with sqlite3.connect(f"file:{DB.db_path(root)}?mode=ro", uri=True) as con:
            have = {r[0] for r in con.execute("SELECT judge_id FROM judge")}
        item_judges = [j for j in POLICY_JUDGE_IDS if j in have]
    SN.build(root, DB.db_path(root), check_free_space=False, scope="paper_frame", dest_root=dest_root, model_ids=[synth.MODEL_ID],
             precondition_override="synthetic registry in a test: no paired-ReLU tree to check", item_judges=list(item_judges),
             item_projection="reader" if item_judges else None)
    sid, _manifest = SN.paper_frame_current(dest_root)
    return str(sid), live_head(root)


def notebook_mirror(tmp_path, name: str = "clone") -> Path:
    """`<tmp>/<name>/paper/notebooks/` holding copies of the two notebook files and their `data/`, with `<tmp>/<name>/src` -> this checkout's `src`, so
    `Path(__file__).parents[2] / "src"` resolves the way it does in a clone and every cache / figure lands in the temp tree."""
    clone = Path(tmp_path) / name
    nbdir = clone / "paper" / "notebooks"
    nbdir.mkdir(parents=True)
    for f in NB_FILES + NB_MODULES:
        shutil.copy(NOTEBOOKS / f, nbdir / f)
    shutil.copytree(NOTEBOOKS / "data", nbdir / "data")          # the committed data a clone carries (paper_judges.json, read at import)
    os.symlink(REPO / "src", clone / "src")
    return nbdir


def nb_env(tmp_path, **over) -> dict:
    """The environment a notebook import runs under: an EMPTY home (never the machine's ~/registry), no SISTEER_SRC, Agg."""
    home = Path(tmp_path) / "home"
    home.mkdir(exist_ok=True)
    env = {"HOME": str(home), "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(home / ".mpl"), "SISTEER_REGISTRY_DISABLED": "1",
           "REGISTRY_ROOT": str(home / "registry")}
    env.update({k: v for k, v in over.items() if v is not None})
    return env


def _put_back_at_teardown(monkeypatch, mapping: dict, key: str) -> None:
    """Record `mapping[key]` with `monkeypatch`, so its undo (the test's teardown) puts back the same object, or no entry at all."""
    if key in mapping:
        monkeypatch.setitem(mapping, key, mapping[key])
    else:
        monkeypatch.setitem(mapping, key, None)          # recorded as absent (the undo deletes whatever the key holds then) ...
        del mapping[key]                                 # ... and absent until something imports it


def import_notebook(nbdir: Path, env: dict, monkeypatch):
    """A FRESH import of `registry_nb` from `nbdir` under `env` (the module resolves its source and memoises its readers at import).

    Everything it swaps comes back when `monkeypatch` is undone at the test's teardown: the environment; sys.path, including the
    SISTEER_SRC entry registry_nb inserts at import (the undo restores the whole list); and sys.modules, where `registry_nb` and
    every sibling module of the notebook directory (`judge_rules`, `pooled_ci`, `ctlstop_sidecar`, ...; the mirror's copy imports
    `judge_rules` from the mirror) go back to the object they held, or to no entry. Without that the mirror's module outlived the
    test: a later test module's `import registry_nb`, `ctlstop_sidecar._R()` and the paper tests' conftest read it instead of the
    module that test module had bound, and paper/tests/test_registry_nb_ctlstop_appendix.py failed whenever a test here ran first
    in the same session (`test_import_notebook_puts_back_what_it_swaps` pins the restore). The backend it selects,
    `matplotlib.use("Agg")`, stays selected, as the notebook's other test modules leave theirs."""
    for k in NB_ENV_KEYS + ("HOME", "MPLBACKEND", "MPLCONFIGDIR"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(nbdir))
    for name in sorted({"registry_nb"} | {p.stem for d in (NOTEBOOKS, Path(nbdir)) for p in d.glob("*.py")}):
        _put_back_at_teardown(monkeypatch, sys.modules, name)
    sys.modules.pop("registry_nb", None)
    import matplotlib
    matplotlib.use("Agg")
    return importlib.import_module("registry_nb")


@pytest.fixture
def world(tmp_path):
    """A synthetic registry, its paper-frame snapshot, and a notebook mirror."""
    root = synthetic_registry(tmp_path)
    dest = str(tmp_path / "paper_snapshot")
    sid, head = export_snapshot(root, dest)
    return dict(root=root, dest=dest, sid=sid, head=head, nbdir=notebook_mirror(tmp_path))


# ---- the source rule ------------------------------------------------------------------------------------------------------------
def test_snapshot_source_when_registry_absent(world, tmp_path, monkeypatch, capsys):
    R = import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=world["dest"]), monkeypatch)
    out = capsys.readouterr().out
    assert R.SOURCE == "snapshot" and R.SNAPSHOT_ID == world["sid"]
    assert R.SISTEER_SRC == str(world["nbdir"].parents[1] / "src")            # the clone-relative default, SISTEER_SRC unset
    assert "registry source snapshot" in out and world["dest"] in out          # the one header line names the source
    assert R.head() == world["head"]
    with R.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 4
    df = R.query("SELECT run_id FROM v_paper_long", "t")
    assert len(df) == 4 and df.attrs["head"] == world["head"]
    cached = [f for f in os.listdir(R.CACHE) if f.endswith(".parquet")]
    assert cached and all(f.startswith(f"t-snapshot@{world['sid'].replace('-', '.')}-{world['head']}-") for f in cached), cached
    # the three direct openers, through connect()
    assert R.headline_metric_ids() == {}                                        # the synthetic eval declares no list
    assert len(R.role_collapse_rows(R.head())) == 0
    missing = R.check_headline_scales(strict=False)
    assert isinstance(missing, tuple) and missing == ()


def test_live_refuses_by_name_when_db_absent(world, tmp_path, monkeypatch):
    env = nb_env(tmp_path, REGISTRY_NB_SOURCE="live", REGISTRY_NB_SNAPSHOT_ROOT=world["dest"])
    with pytest.raises(FileNotFoundError) as e:
        import_notebook(world["nbdir"], env, monkeypatch)
    assert os.path.join(env["REGISTRY_ROOT"], "registry.sqlite") in str(e.value) and "registry source live" in str(e.value)


def test_snapshot_refuses_by_name_when_none_committed(world, tmp_path, monkeypatch):
    empty = tmp_path / "no_snapshot"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as e:
        import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SOURCE="snapshot", REGISTRY_NB_SNAPSHOT_ROOT=str(empty)), monkeypatch)
    assert str(empty) in str(e.value) and "registry source snapshot" in str(e.value)


def test_auto_prefers_live_when_db_present(world, tmp_path, monkeypatch, capsys):
    R = import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_ROOT=world["root"], REGISTRY_NB_SNAPSHOT_ROOT=world["dest"]), monkeypatch)
    assert R.SOURCE == "live" and R.SNAPSHOT_ID is None and R.DB == DB.db_path(world["root"])
    assert "registry source live" in capsys.readouterr().out
    df = R.query("SELECT run_id FROM run", "t")
    assert len(df) == 4 and any(f.startswith(f"t-live-{world['head']}-") for f in os.listdir(R.CACHE))


def test_auto_with_neither_imports_and_connect_refuses_by_name(world, tmp_path, monkeypatch):
    """The offline path the notebook's own synthetic-frame tests rely on: importable, `check_headline_scales` answers (), connect() names both."""
    empty = tmp_path / "no_snapshot"
    empty.mkdir()
    env = nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=str(empty))
    R = import_notebook(world["nbdir"], env, monkeypatch)
    assert R.SOURCE == "live" and R.SNAPSHOT_ID is None
    assert R.check_headline_scales(strict=False) == () and R.headline_metric_ids() == {} and len(R.role_collapse_rows(0)) == 0
    with pytest.raises(FileNotFoundError) as e:
        R.connect()
    assert R.DB in str(e.value) and str(empty) in str(e.value)


def test_invalid_source_mode_refuses(world, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="REGISTRY_NB_SOURCE"):
        import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SOURCE="bogus"), monkeypatch)


def test_nb_cache_zero_bypasses_read_and_write(world, tmp_path, monkeypatch):
    import pandas as pd
    R = import_notebook(world["nbdir"], nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=world["dest"], NB_CACHE="0"), monkeypatch)
    sql = "SELECT run_id FROM run"
    df = R.query(sql, "t")
    assert len(df) == 4 and not [f for f in os.listdir(R.CACHE) if f.endswith(".parquet")]     # nothing written
    # a planted cache file with the exact key is NOT read either
    sha = hashlib.sha1(sql.encode()).hexdigest()[:8]
    pd.DataFrame({"run_id": ["planted"]}).to_parquet(os.path.join(R.CACHE, f"t-{R.cache_source()}-{world['head']}-{sha}.parquet"))
    assert len(R.query(sql, "t")) == 4
    monkeypatch.setenv("NB_CACHE", "1")
    assert list(R.query(sql, "t").run_id) == ["planted"]                       # the same key IS read once the cache is on again


def test_two_frames_at_one_head_never_share_a_cache_entry(tmp_path, monkeypatch):
    """A frame's release pass rewrites its strings at the SAME registry head, so a raw cut and the frame released from it share a head.
    Keyed by head, the second frame read the first one's cached rows: the cache keys on the snapshot id (`cache_source`). Two frames
    of one synthetic registry, the second cut after a string of one run changed in place (the head stays), read through ONE
    notebook mirror and so one cache: each returns its own string, and NB_REUSE_CACHE does not cross frames either."""
    root = synthetic_registry(tmp_path)
    sid_a, head_a = export_snapshot(root, str(tmp_path / "frame_a"))
    con = sqlite3.connect(DB.db_path(root))
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'").fetchall():
        con.execute(f"DROP TRIGGER {name}")                                     # a test-local edit, written straight into the copy
    rid = con.execute("SELECT run_id FROM run ORDER BY run_id LIMIT 1").fetchone()[0]
    con.execute("UPDATE run SET producer_run_key_legacy = 'released spelling' WHERE run_id = ?", (rid,))
    con.commit()
    con.close()
    sid_b, head_b = export_snapshot(root, str(tmp_path / "frame_b"))
    assert head_a == head_b and sid_a != sid_b
    nbdir = notebook_mirror(tmp_path)
    sql = "SELECT run_id, producer_run_key_legacy FROM run ORDER BY run_id"
    got = {}
    for tag, dest in (("a", "frame_a"), ("b reusing", "frame_b"), ("b", "frame_b")):      # NB_REUSE_CACHE first: b has no entry yet
        R = import_notebook(nbdir, nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=str(tmp_path / dest),
                                          NB_REUSE_CACHE="1" if tag == "b reusing" else None), monkeypatch)
        got[tag] = dict(zip(*R.query(sql, "t")[["run_id", "producer_run_key_legacy"]].T.values))
    assert got["a"][rid] != "released spelling" and got["b reusing"][rid] == "released spelling" and got["b"] == got["b reusing"]
    names = sorted(f for f in os.listdir(nbdir / "_cache") if f.startswith("t-") and f.endswith(".parquet"))
    assert len(names) == 2 and {n.split("-")[1] for n in names} == {f"snapshot@{s.replace('-', '.')}" for s in (sid_a, sid_b)}, names


# ---- the import helper puts back what it swaps ------------------------------------------------------------------------------------
@pytest.mark.parametrize("before,source", [("bound", None), ("absent", None), ("bound", "bogus")], ids=["bound", "absent", "refused"])
def test_import_notebook_puts_back_what_it_swaps(tmp_path, monkeypatch, before, source):
    """Once the monkeypatch `import_notebook` was handed is undone (the calling test's teardown), sys.modules, sys.path and the
    environment are as they were: `registry_nb` is again the module a test module bound at its import (a stand-in here), or absent
    as it was, also after an import that refused; and `judge_rules`, which the mirror's copy imported from the mirror, is gone.
    Before the restore the mirror's module stayed in sys.modules, and paper/tests/test_registry_nb_ctlstop_appendix.py, run after
    any test here in one session, read it through `ctlstop_sidecar._R()` instead of the module it had patched."""
    empty = tmp_path / "no_snapshot"
    empty.mkdir()
    nbdir = notebook_mirror(tmp_path)
    bound = types.ModuleType("registry_nb")
    if before == "bound":
        monkeypatch.setitem(sys.modules, "registry_nb", bound)
    else:
        monkeypatch.delitem(sys.modules, "registry_nb", raising=False)
    monkeypatch.delitem(sys.modules, "judge_rules", raising=False)
    path_before, env_before = list(sys.path), dict(os.environ)
    with pytest.MonkeyPatch.context() as calling_test:            # the calling test's monkeypatch, undone at its teardown
        env = nb_env(tmp_path, REGISTRY_NB_SNAPSHOT_ROOT=str(empty), REGISTRY_NB_SOURCE=source)
        if source == "bogus":
            with pytest.raises(ValueError, match="REGISTRY_NB_SOURCE"):
                import_notebook(nbdir, env, calling_test)
        else:
            R = import_notebook(nbdir, env, calling_test)
            assert sys.modules["registry_nb"] is R and Path(R.__file__).resolve().parent == nbdir.resolve()
            rules = R._judge_rules()
            assert sys.modules["judge_rules"] is rules and Path(rules.__file__).resolve().parent == nbdir.resolve()
    if before == "bound":
        assert sys.modules.get("registry_nb") is bound
    else:
        assert "registry_nb" not in sys.modules
    assert "judge_rules" not in sys.modules
    assert sys.path == path_before and dict(os.environ) == env_before


# ---- the generator: the paper's panels, nothing else ------------------------------------------------------------------------------
ICLR_CELL_IDS = ("40484fe7", "583e3f04", "12cd66ed", "ec6aacd8", "9bf536ac", "6fec6474", "b6d74ec9", "ac902304", "4f2ab8d1", "7c1de904",
                 "5e61e1ab", "624d0e27", "a3e91d55", "f7a1c206", "11a8433d", "b24c9738", "7d2e9c40", "a8e3d1c2", "d4f7b2a9", "3c9f7a51",
                 "8e1b4d27", "5a7c2e93", "6d2f8b3a", "b0a7c4e1", "9b3d6f18", "c4e8a2d6", "e6f1b5a9", "4d9c2b7a", "8f3e61c5", "3c7a1e55",
                 "5b9d2f7e", "7e2c9b41", "f1a8c3d7", "5c1d9e73", "a6e0f2b8", "3f8b4c61", "d2a7e5c9", "1b4eb5de", "7fae15d6", "de9a82d2",
                 "76fdab54", "5c3e9a71", "a2d84f6b", "bb79fa42", "e3b71c58", "9d4a6f20", "be6069de", "431232a9",
                 "347c3abd", "f60c0561", "ba7e3dcf", "3a7c91e4", "c5e0b2d8", "2802219e", "f46e655e", "5f821770", "ee8bdbc1", "0c6192ec",
                 "6b1e4f90", "e2c85a17", "53f2d0e0", "a77fad73", "da341da1", "037dbf1e", "b2d18d18",
                 "b7d4e9f2", "92fdc8e2", "23555668")
# the paper notebook's cells at the ICLR submission, in order, plus the judge comparison's two after e6f1b5a9, the control-pool stopping
# appendix's eight after f1a8c3d7, the distillation-target appendix's five after them (the multi-turn figure's two and the paper's table
# of it, then the battery figure's two), the generator comparison's two after those, and the further appendix analyses' seventeen
# before the zip cell (b7d4e9f2; paper/tests/test_appendix_notebook.py pins their placement)
# spelled in pieces so this file never carries the strings the public tree's census forbids
SECOND_MODEL = re.compile("qwen35_" + "35b|35b_" + "a3b|Qwen3\\.5-3" + "5B|pipeline[-_ ]test|PIPELINE_" + "TEST|REGISTRY_NB_" + "PANELS|figT_")


def test_generator_carries_the_papers_panels_only_in_the_iclr_layout():
    """The public notebook is the paper's panels in the ICLR-era layout plus the judge comparison, the control-pool appendix figures,
    the two distillation-target appendix figures and the generator comparison: the pinned cell ids in that order, no second model, no
    panel switch, no test artifact. The zip cell lists paper figures only; the header cell names the source; connect() is the ONE
    opener. The reader library spells no second model either."""
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    assert tuple(re.findall(r'^(?:md|code)\("([0-9a-f]{8})"', src, flags=re.M)) == ICLR_CELL_IDS
    assert not SECOND_MODEL.search(src), SECOND_MODEL.search(src).group(0)
    assert "registry source {R.SOURCE}" in src                                     # the header cell names the source
    zip_cell = src[src.index('code("b7d4e9f2"'):src.index('md("92fdc8e2"')]
    assert "PAPER_FIGURES = [" in zip_cell and "fig7_frontier" in zip_cell
    assert '"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}' in src    # no maintainer kernel
    nb = (NOTEBOOKS / "registry_nb.py").read_text(encoding="utf-8")
    assert nb.count("sqlite3.connect(") == 1                                       # connect() is the ONE opener
    assert not SECOND_MODEL.search(nb), SECOND_MODEL.search(nb).group(0)
