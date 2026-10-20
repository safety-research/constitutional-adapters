"""Preflight check 8: presence is not readability, and a README is not a payload.

(1) The offline `load_dataset` readers (GSM8K: openai/gsm8k AND gsm8k-platinum through `inspect_served.plan_task`; MMLU: cais/mmlu)
    need a PREPARED `datasets` cache under HF_DATASETS_CACHE -- the hub snapshot alone does not serve them (g8k_common.sh
    g8k_datasets_guard / g8k_datasets_probe). Check 8 now ports the directory guard (necessary) and `preflight_offline_readers` runs the
    actual readers under the driver venv and the tinycap CLIENT venv (sufficient).
(2) HarmBench is checked at its exact test parquet, and every pins.yaml entry through `sisteer.capeval.corpora`'s own file rule.
Synthetic ids only; the positives read this grading host's own pinned snapshots and prepared caches, offline.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402
from test_seedrep_w9_container_paths import _fake_volume, _plan_with_legs, _stub_modal  # noqa: E402

REPO = T.REPO
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
_ABSENT = object()
LOCAL_HF = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
LOCAL_DATASETS = Path(os.environ.get("HF_DATASETS_CACHE") or (LOCAL_HF / "datasets"))
VENV_PY = str(Path(sys.executable))


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w10_readers", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


@pytest.fixture(scope="module")
def pins():
    return yaml.safe_load((REPO / "configs/capeval/pins.yaml").read_text())["datasets"]


def _local_has(srv, pins):
    from sisteer.eval import jailbreak_data as jd
    snaps = [srv.dataset_snapshot_dir(str(LOCAL_HF), jd.HARMBENCH_REPO, jd.HARMBENCH_REVISION) / jd.HARMBENCH_FILES["test"],
             srv.dataset_snapshot_dir(str(LOCAL_HF), jd.JBB_REPO, jd.JBB_REVISION) / jd.JBB_FILES["harmful"]]
    snaps += [srv.dataset_snapshot_dir(str(LOCAL_HF), pins[n]["repo_id"], pins[n]["revision"]) for n in (*srv.CONTROLS_PIN_NAMES, *srv.GSM8K_PIN_NAMES)]
    return all(p.exists() for p in snaps)


# --- P1 (2): a README is not a payload --------------------------------------------------------------------------------------------

class TestP1_2_PayloadNotReadme:
    def test_pinned_snapshot_payload_is_the_readers_own_rule(self, srv, pins, tmp_path, monkeypatch):
        """For every pin: the selection equals `corpora.pinned_snapshot_files` over the same snapshot (its snapshot_download stubbed)."""
        import huggingface_hub
        from sisteer.capeval import corpora as CP
        for name in (*srv.CONTROLS_PIN_NAMES, *srv.GSM8K_PIN_NAMES):
            e = pins[name]
            snap = tmp_path / name
            suffix = CP._VERIFY_SUFFIXES[e["verify"]][0]
            files = ["README.md", ".gitattributes", f"other/decoy{suffix}", "notes.txt"]
            files += [pat.replace("*", "00000-of-00001") + ("" if pat.endswith(suffix) else suffix) for pat in (e.get("patterns") or [])]
            if not e.get("patterns"):
                files += [f"data/train-00000{suffix}", f"data/test-00000{suffix}"]
            for rel in files:
                (snap / rel).parent.mkdir(parents=True, exist_ok=True)
                (snap / rel).write_text("x")
            monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
            theirs = sorted(p.relative_to(snap).as_posix() for p in CP.pinned_snapshot_files(e, name))
            assert srv.pinned_snapshot_payload(snap, e) == theirs, name
            assert "README.md" not in theirs and "notes.txt" not in theirs

    def test_a_snapshot_holding_only_a_readme_fails_by_name_for_every_pin(self, srv, pins, tmp_path):
        """The reproduction: XSTest, OR-Bench, MMLU, Platinum each passed the earlier check with a README alone."""
        hf, corp = _fake_volume(srv, tmp_path)
        for name, leg in (("xstest_v2", "controls"), ("or_bench", "controls"), ("mmlu", "controls"), ("gsm8k_platinum", "gsm8k")):
            e = pins[name]
            snap = srv.dataset_snapshot_dir(hf, e["repo_id"], e["revision"])
            for q in list(snap.rglob("*")):
                if q.is_file() and q.name not in ("README.md", ".gitattributes"):
                    q.unlink()
            with pytest.raises(AssertionError, match=rf"{name}: no payload matching"):
                srv.preflight_corpora_check(REPO, _plan_with_legs(leg), hf_vol=hf, corpora_vol=corp, datasets_cache=f"{hf}/datasets")
            # restore the payload for the next pin (one fake volume, one pin blanked at a time)
            hf, corp = _fake_volume(srv, tmp_path / f"re-{name}")

    def test_harmbench_is_checked_at_its_exact_test_parquet(self, srv, tmp_path):
        from sisteer.eval import jailbreak_data as jd
        hf, corp = _fake_volume(srv, tmp_path)
        want = jd.HARMBENCH_FILES["test"]
        assert want == "DirectRequest/test-00000-of-00001.parquet"
        snap = srv.dataset_snapshot_dir(hf, jd.HARMBENCH_REPO, jd.HARMBENCH_REVISION)
        (snap / want).unlink()
        (snap / "DirectRequest" / "val-00000-of-00001.parquet").write_text("x")          # a sibling parquet is not the file the reader downloads
        with pytest.raises(AssertionError, match=r"harmbench: DirectRequest/test-00000-of-00001\.parquet missing"):
            srv.preflight_corpora_check(REPO, _plan_with_legs("pasttense"), hf_vol=hf, corpora_vol=corp)


# --- P1 (1): a hub snapshot is not a prepared datasets cache ------------------------------------------------------------------------

class TestP1_1_PreparedDatasetsCache:
    def test_dirs_needed_come_from_the_trees_own_literals(self, srv):
        need = srv.corpora_needed_for_plan(_plan_with_legs("gsm8k", "controls"))
        dirs = srv.datasets_cache_dirs_needed(REPO, need)
        g8k = srv.g8k_cells_module(REPO)
        assert dirs == {**{d: g8k.DATASETS_CONFIG_DIRS[d] for d in g8k.DATASETS_REQUIRED}, "cais___mmlu": "all"}
        assert set(g8k.DATASETS_REQUIRED) == {"openai___gsm8k", "madrylab___gsm8k-platinum"}
        # the MMLU reader's own signature names the config dir the guard checks
        src = inspect.getsource(importlib.import_module("sisteer.capeval.tiers.parity").mmlu_parity_rows)
        assert 'load_dataset("cais/mmlu", "all"' in src

    def test_hub_present_cache_absent_fails_by_name(self, srv, tmp_path):
        hf, corp = _fake_volume(srv, tmp_path, datasets_cache=False)
        with pytest.raises(AssertionError, match=r"carries no prepared openai___gsm8k"):
            srv.preflight_corpora_check(REPO, _plan_with_legs("gsm8k"), hf_vol=hf, corpora_vol=corp, datasets_cache=f"{hf}/datasets")
        with pytest.raises(AssertionError, match=r"carries no prepared cais___mmlu"):
            srv.preflight_corpora_check(REPO, _plan_with_legs("controls"), hf_vol=hf, corpora_vol=corp, datasets_cache=f"{hf}/datasets")

    def test_missing_stock_gsm8k_with_platinum_prepared_fails_naming_the_stock_dataset(self, srv, tmp_path):
        cache = tmp_path / "datasets"
        (cache / "madrylab___gsm8k-platinum" / "main").mkdir(parents=True)
        dirs = srv.datasets_cache_dirs_needed(REPO, {"datasets": ["gsm8k"]})
        problems = srv.datasets_cache_problems(str(cache), dirs)
        assert len(problems) == 1 and "carries no prepared openai___gsm8k" in problems[0]

    def test_config_dir_alias_symlink_passes_and_a_dangling_one_refuses(self, srv, tmp_path):
        cache = tmp_path / "datasets"
        (cache / "openai___gsm8k" / "default-8cbc416ed05c3005").mkdir(parents=True)
        (cache / "madrylab___gsm8k-platinum" / "main").mkdir(parents=True)
        dirs = srv.datasets_cache_dirs_needed(REPO, {"datasets": ["gsm8k"]})
        problems = srv.datasets_cache_problems(str(cache), dirs)
        assert len(problems) == 1 and "no offline config dir openai___gsm8k/default-data_dir=main (absent)" in problems[0]
        os.symlink("default-8cbc416ed05c3005", cache / "openai___gsm8k" / "default-data_dir=main")
        assert srv.datasets_cache_problems(str(cache), dirs) == []
        os.unlink(cache / "openai___gsm8k" / "default-data_dir=main")
        os.symlink("default-gone", cache / "openai___gsm8k" / "default-data_dir=main")
        problems = srv.datasets_cache_problems(str(cache), dirs)
        assert len(problems) == 1 and "(a dangling symlink)" in problems[0]

    def test_the_real_local_caches_pass_the_static_check_for_every_leg(self, srv, pins, tmp_path):
        if not _local_has(srv, pins) or not (LOCAL_DATASETS / "openai___gsm8k").is_dir():
            pytest.skip("this machine lacks a pinned snapshot or the prepared datasets cache")
        corp = tmp_path / "heldout_corpora"
        corp.mkdir()
        for f in srv.HELDOUT_CORPUS_FILES:
            (corp / f).write_text("{}")
        out = srv.preflight_corpora_check(REPO, _plan_with_legs("pasttense", "renellm", "controls", "gsm8k"), hf_vol=str(LOCAL_HF), corpora_vol=str(corp),
                                          datasets_cache=str(LOCAL_DATASETS))
        assert out["or_bench"]["files"] == ["or-bench-hard-1k.csv"] and out["datasets_cache"]["dirs"]["openai___gsm8k"] == "default-data_dir=main"


# --- the actual readers, under a venv, offline ----------------------------------------------------------------------------------------

def _probe_env(srv, hf, dcache, corpus_dir="/nonexistent-heldout"):
    base = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL")}
    return srv.reader_probe_env(REPO, hf_vol=hf, datasets_cache=dcache, corpus_dir=corpus_dir, base=base)


class TestReaderProbes:
    def test_probes_follow_the_plans_legs_and_name_their_venv(self, srv):
        need = srv.corpora_needed_for_plan(_plan_with_legs("pasttense", "renellm", "controls", "gsm8k"))
        probes = srv.reader_probe_scripts(need)
        assert set(probes) == {"static_goal_sets", "heldout_renellm", "controls_judged", "controls_mmlu", "gsm8k_platinum", "gsm8k_plan_task"}
        assert probes["gsm8k_plan_task"][0] == "client" and all(v == "driver" for n, (v, _, _) in probes.items() if n != "gsm8k_plan_task")
        assert srv.reader_probe_scripts(srv.corpora_needed_for_plan(_plan_with_legs("misalign", "petri"))) == {}
        # g8k_common.sh's probe, verbatim in what it asserts: plan_task at the CLI task name, exactly GSM8K_PAIRED_N items, offline
        script = probes["gsm8k_plan_task"][1]
        assert 'isv.plan_task(task, grader_model=None, thinking="off", repo_root=Path.cwd())' in script and "C.GSM8K_PAIRED_N" in script
        assert probes["gsm8k_plan_task"][2] == ("gsm8k_paired200",)

    def test_probe_env_is_the_drivers_env_made_offline_and_gpu_less(self, srv):
        env = _probe_env(srv, "/vol/hf", "/vol/hf/datasets", corpus_dir="/c")
        assert env["HF_HUB_OFFLINE"] == "1" and env["HF_DATASETS_OFFLINE"] == "1" and env["HF_DATASETS_CACHE"] == "/vol/hf/datasets"
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["SISTEER_HELDOUT_CORPUS_DIR"] == "/c" and env["PYTHONPATH"].startswith(f"{REPO}/src")
        prod = srv.container_env(bank_view="/b", tree="/t", thinking=False, base={})
        assert prod["HF_DATASETS_CACHE"] == f"{srv.HF_VOL}/datasets" == srv.HF_DATASETS_VOL       # the drivers and the guard agree on the cache

    @pytest.mark.parametrize("name", ["controls_mmlu", "gsm8k_platinum"])
    def test_load_dataset_readers_pass_on_the_prepared_cache_and_fail_without_it(self, srv, pins, tmp_path, name):
        pytest.importorskip("datasets")
        if not _local_has(srv, pins) or not (LOCAL_DATASETS / "cais___mmlu").is_dir() or not (LOCAL_DATASETS / "madrylab___gsm8k-platinum").is_dir():
            pytest.skip("this machine lacks the pinned snapshot or the prepared datasets cache")
        _, script, tail = srv.reader_probe_scripts(srv.corpora_needed_for_plan(_plan_with_legs("controls", "gsm8k")))[name]
        ok = srv.run_reader_probe(script, tail, python=VENV_PY, tree=REPO, env=_probe_env(srv, str(LOCAL_HF), str(LOCAL_DATASETS)), timeout=600)
        assert ok["out"].startswith("readers OK:")
        # hub present, prepared cache ABSENT: the reader itself refuses -- the static presence check could never see this
        empty = tmp_path / "empty-datasets"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="reader probe rc"):
            srv.run_reader_probe(script, tail, python=VENV_PY, tree=REPO, env=_probe_env(srv, str(LOCAL_HF), str(empty)), timeout=600)

    def test_static_goal_sets_and_judged_controls_read_offline_from_the_pinned_snapshots(self, srv, pins):
        pytest.importorskip("pandas")
        if not _local_has(srv, pins):
            pytest.skip("this machine lacks a pinned snapshot")
        probes = srv.reader_probe_scripts(srv.corpora_needed_for_plan(_plan_with_legs("pasttense", "controls")))
        for name in ("static_goal_sets", "controls_judged"):
            _, script, tail = probes[name]
            out = srv.run_reader_probe(script, tail, python=VENV_PY, tree=REPO, env=_probe_env(srv, str(LOCAL_HF), str(LOCAL_DATASETS)), timeout=600)
            assert out["out"].startswith("readers OK:"), name

    def test_static_goal_sets_fail_against_an_empty_hub(self, srv, tmp_path):
        pytest.importorskip("pandas")
        _, script, tail = srv.reader_probe_scripts(srv.corpora_needed_for_plan(_plan_with_legs("pasttense")))["static_goal_sets"]
        with pytest.raises(RuntimeError, match="reader probe rc"):
            srv.run_reader_probe(script, tail, python=VENV_PY, tree=REPO, env=_probe_env(srv, str(tmp_path / "hf"), str(tmp_path / "ds")), timeout=600)

    def test_renellm_reader_reads_the_shipped_copy_and_refuses_without_any_corpus(self, srv, tmp_path):
        """No copy on the volume: the reader reads the paper's corpus the staged tree ships (data/heldout_corpora/renellm.jsonl).
        Neither copy (a tree that lost it, stood in for by pointing the probe's renellm at an absent shipped path): it refuses."""
        _, script, tail = srv.reader_probe_scripts(srv.corpora_needed_for_plan(_plan_with_legs("renellm")))["heldout_renellm"]
        env = _probe_env(srv, str(tmp_path / "hf"), str(tmp_path / "ds"), corpus_dir=str(tmp_path / "none"))
        out = srv.run_reader_probe(script, tail, python=VENV_PY, tree=REPO, env=env)
        assert out["out"] == f"readers OK: renellm corpus d44dd49ed6268e7a at {REPO / 'data/heldout_corpora/renellm.jsonl'}"
        anchor = "from sisteer.eval.attacks import renellm\n"
        absent = script.replace(anchor, anchor + "renellm.SHIPPED_CORPUS = 'data/heldout_corpora/absent-for-this-test.jsonl'\n", 1)
        assert absent != script
        with pytest.raises(RuntimeError, match="not readable"):
            srv.run_reader_probe(absent, tail, python=VENV_PY, tree=REPO, env=env)

    def test_gsm8k_plan_task_probe_runs_under_the_client_venv_only(self, srv):
        if importlib.util.find_spec("inspect_evals") is None:
            pytest.skip("inspect_evals is not in this venv (it lives in the tinycap CLIENT venv, which the in-container check builds)")
        _, script, tail = srv.reader_probe_scripts(srv.corpora_needed_for_plan(_plan_with_legs("gsm8k")))["gsm8k_plan_task"]
        out = srv.run_reader_probe(script, tail, python=VENV_PY, tree=REPO, env=_probe_env(srv, str(LOCAL_HF), str(LOCAL_DATASETS)), timeout=900)
        assert "plans 200 paired items OFFLINE" in out["out"]


# --- the tinycap CLIENT venv (f22_client_venv, ported) --------------------------------------------------------------------------------

class TestClientVenv:
    def test_lock_split_matches_f22_client_venv(self, srv):
        text = (REPO / srv.CLIENT_LOCK_REL).read_text()
        hashed, vcs = srv.client_lock_split(text)
        assert len(vcs) == 1 and all(" @ git+" in v and not v.endswith("\\") for v in vcs)
        assert not any(ln.startswith("instruction-following-eval @") for ln in hashed)
        assert any(ln.startswith("inspect-ai==") for ln in hashed) and any(ln.startswith("inspect-evals==") for ln in hashed)
        assert sum(1 for ln in hashed if "--hash=" in ln) > 100

    def test_ensure_client_venv_runs_the_cluster_recipe_and_its_guard(self, srv, tmp_path, monkeypatch):
        monkeypatch.setattr(srv, "VENVS_LOCAL", str(tmp_path / "venvs"))
        calls = []

        def fake_run(argv, **kw):
            calls.append(list(argv))
            if argv[:2] == ["uv", "venv"]:
                Path(argv[-1], "bin").mkdir(parents=True)
                Path(argv[-1], "bin", "python").write_text("")
            return types.SimpleNamespace(returncode=0, stdout="client venv OK 0.3.260\n", stderr="")
        out = srv.ensure_client_venv(REPO, run=fake_run)
        venv, sentinel, want = srv.client_venv_names(srv.sha256_file(REPO / srv.CLIENT_LOCK_REL))
        vcs = srv.client_lock_split((REPO / srv.CLIENT_LOCK_REL).read_text())[1]
        assert out["built"] and out["n_vcs"] == len(vcs) == 1 and Path(sentinel).read_text() == want
        assert calls[0][:4] == ["uv", "venv", "--python", srv.PY_VERSION]
        assert calls[1][:3] == ["uv", "pip", "install"] and "--require-hashes" in calls[1] and "--no-deps" in calls[1] and "--exclude-newer" in calls[1]
        assert [c[-1] for c in calls[2:2 + len(vcs)]] == vcs
        assert calls[2 + len(vcs)][1:] == ["-c", srv.CLIENT_VENV_GUARD] and "inspect_evals" in srv.CLIENT_VENV_GUARD
        assert srv.ensure_client_venv(REPO, run=fake_run)["built"] is False               # idempotent on the sentinel


# --- registration + ordering --------------------------------------------------------------------------------------------------------

class TestRegistrationAndOrdering:
    def test_the_readers_function_is_registered_and_runs_after_the_static_preflight(self, srv):
        main_src = inspect.getsource(srv.main)
        i, j = main_src.index("preflight.remote(sha, plan_d, LOCAL_DEP_HASHES)"), main_src.index("preflight_offline_readers.remote(sha, plan_d)")
        assert i < j and 'raise SystemExit("[seedrep] PREFLIGHT FAILED -- fix every item above' in main_src[i:j]
        src = DRIVER.read_text()
        k = src.index("def preflight_offline_readers(")
        assert src[:k].rstrip().splitlines()[-1].startswith("@app.function(image=image"), "the readers function must be @app.function-registered"
        m = src.index("def preflight(sha: str, plan: dict, expected_dep_hashes: dict)")
        assert src[:m].rstrip().splitlines()[-1].startswith("@app.function(image=cpu_image"), "@app.function must stay directly above def preflight"


def test_the_operator_steps_define_the_interpreter_they_run(tmp_path):
    """The Run block of the module docstring (the grading host's operator steps) reads no variable it leaves unset: the
    registry steps run the interpreter the steps define -- SISTEER_PYTHON, else the repository's .venv/bin/python, the
    front door's rule -- and check it before the first step. Run as written under the shipped site example
    (SISTEER_PYTHON=''), the check and the publish step reach the repository's venv, a set SISTEER_PYTHON wins, and a
    failed check stops the steps before the publish step runs."""
    import ast
    import re
    import shlex
    import subprocess

    doc = ast.get_docstring(ast.parse(DRIVER.read_text(encoding="utf-8")))
    steps = doc.split("\nRun (", 1)[1].split("\n\n")[1]
    code = "\n".join(ln for ln in steps.splitlines() if not ln.lstrip().startswith("#"))
    assigned = set(re.findall(r"(?:^|;)\s*([A-Z][A-Z0-9_]*)=", code, re.M))
    bare = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)(?![A-Z0-9_:])", code))   # every use without a ${NAME:-default}
    assert bare <= assigned, f"the steps read {sorted(bare - assigned)} without setting them"
    p_line = next(ln.strip() for ln in steps.splitlines() if ln.strip().startswith("P="))
    publish = next(ln.strip() for ln in code.splitlines() if "registry_publish.py" in ln)

    def run(cwd, **env):
        cwd.mkdir(parents=True, exist_ok=True)
        full = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "RS": str(tmp_path / "rs"), **env}
        return subprocess.run(["bash", "-euc", p_line + "\n" + publish], cwd=cwd, env=full, capture_output=True, text=True)

    def stub(path, log):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> {shlex.quote(str(log))}\n')
        path.chmod(0o755)

    repo, log = tmp_path / "repo", tmp_path / "venv.log"
    stub(repo / ".venv" / "bin" / "python", log)
    r = run(repo, SISTEER_PYTHON="")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 2 and calls[0].startswith("-c "), calls
    assert calls[1].startswith("scripts/sdfbase/seedrep/registry_publish.py --accepted " + str(tmp_path / "rs")), calls
    other, other_log = tmp_path / "other" / "python", tmp_path / "other.log"
    stub(other, other_log)
    assert run(repo, SISTEER_PYTHON=str(other)).returncode == 0
    assert len(other_log.read_text().splitlines()) == 2 and len(log.read_text().splitlines()) == 2
    r = run(tmp_path / "no-venv", SISTEER_PYTHON="")
    assert r.returncode != 0 and len(log.read_text().splitlines()) == 2, r.stdout + r.stderr
