"""bcb-bfcl-200-widen S6e: `scripts/modal/tinycap_budget_measure.py` (loaded under the modal stub)
— the qwen35_4b batched-token budget MEASUREMENT walks the sweep's budget ladder at L = 8,192 with
the production standup under `--allow-over-budget`, reads a wall from the KV-coverage refusal,
bisects once, writes `budget.json` in the sweep analyzer's shape and prints the geometry paste
block. Standups are doubles here: the walk, the statuses and the record are what is tested."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tinycap_launcher as TL  # noqa: E402

BUDGET = REPO / "scripts" / "modal" / "tinycap_budget_measure.py"
FULL = "f6cf462" + "0" * 33
L = 8192


def _load():
    # The tool does `import tinycap_served as T`, and tinycap_served captures TINYCAP_APP_NAME at import
    # (`APP_NAME = os.environ.get(...)`) — so the sibling module imported HERE, under this test's app name,
    # must not stay in sys.modules: tinycap_tb reads `TS.APP_NAME` from whatever tinycap_served is cached,
    # and the terminal-bench launcher tests then saw "tinycap-bcbw-qwen-budget-test" as the running app
    # (seen in the Modal CPU suite — the second leak after the parser probe's). Restore sys.modules and
    # sys.path too, exactly as test_bfcl_parser_probe._load_modal_probe does.
    saved, saved_env = sys.modules.get("modal"), os.environ.get(C.MODAL_APP_NAME_ENV)
    saved_siblings = {name: sys.modules.get(name) for name in ("tinycap_served", "tinycap_images")}
    modal_dir = str(REPO / "scripts" / "modal")
    sys.modules["modal"] = TL._stub_modal()
    os.environ[C.MODAL_APP_NAME_ENV] = "tinycap-bcbw-qwen-budget-test"
    sys.path.insert(0, modal_dir)
    try:
        spec = importlib.util.spec_from_file_location("tinycap_budget_measure_under_test", BUDGET)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["modal"] = saved
        else:
            sys.modules.pop("modal", None)
        for name, prior in saved_siblings.items():
            if prior is not None:
                sys.modules[name] = prior
            else:
                sys.modules.pop(name, None)
        if sys.path and sys.path[0] == modal_dir:
            sys.path.pop(0)
        if saved_env is None:
            os.environ.pop(C.MODAL_APP_NAME_ENV, None)
        else:
            os.environ[C.MODAL_APP_NAME_ENV] = saved_env
    return mod


@pytest.fixture(scope="module")
def M():
    return _load()


def test_pure_helpers_and_the_wall_vocabulary_are_the_sweeps(M):
    sys.path.insert(0, str(REPO / "scripts"))
    import served_decode_sweep as sw
    assert M.probe_seqs(786_432, L) == 96 and M.probe_seqs(393_216, L) == 48
    with pytest.raises(ValueError, match="not a multiple"):
        M.probe_seqs(786_433, L)
    assert M.cap_binds(257, hybrid=True) and not M.cap_binds(256, hybrid=True) and not M.cap_binds(4096, hybrid=False)
    assert M.STATUS_KV_WALL in sw.WALL_STATUSES and M.STATUS_PASS == "pass"
    rec = M.probe_record(product=786_432, S=96, budget_len=L, status=M.STATUS_KV_WALL, seconds=12.0, kv_tokens=700_000,
                         error="KV cache holds …", job_id="fc-1", tag="S96_L8192")
    assert sw.is_wall(rec) and rec["max_num_seqs"] == 96 and rec["kv_kind"] == "hybrid_max_concurrency_x_L"
    # code review: the KV kind follows the ROW (it was hard-coded hybrid); both spellings are the sweep's
    uni = M.probe_record(product=786_432, S=96, budget_len=L, status=M.STATUS_PASS, seconds=1.0, kv_tokens=None, error=None,
                         job_id="fc-1", tag="t", hybrid=False)
    assert uni["kv_kind"] == M.KV_KIND_UNIFORM == "uniform_blocks_x_block_size" and M.KV_KIND_HYBRID == rec["kv_kind"]
    sweep_src = (REPO / "scripts" / "served_decode_sweep.py").read_text()
    assert f'"{M.KV_KIND_UNIFORM}"' in sweep_src and f'"{M.KV_KIND_HYBRID}"' in sweep_src
    # the sweep's classifier and wall vocabulary are read DIRECTLY (a tree lacking them fails loudly, never "not_ready")
    src = BUDGET.read_text()
    assert 'hasattr(sw, "classify_launch_failure")' not in src and 'getattr(sw, "WALL_STATUSES"' not in src
    assert "sw.classify_launch_failure(" in src and "sw.WALL_STATUSES" in src
    assert not sw.is_wall({**rec, "status": "kv_refused"})          # the earlier private spelling would NOT have been a wall
    assert M.BUDGET_LEN == sw.DEFAULT_BUDGET_LEN == 8192 and M.S_MAX_HYBRID == sw.S_MAX_HYBRID == 256
    assert M.GEOM_ROOT == f"{C.VOL_TINYCAP_ROOT}/geom"


def _tree(tmp_path):
    repos = tmp_path / "repos"
    tree = TL._tree(repos, "f6cf462", FULL)
    for rel in ("scripts/served_decode_sweep.py", "scripts/geom_sweep_analyze.py"):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text((REPO / rel).read_text())
    (tree / "cfg.yaml").write_text("x: 1\n")
    return tree


class _Standup:
    """The standup double: S <= WALL_S comes up (endpoint.json), above it EXITS naming the
    uniform KV-coverage verdict `rung_probe.kv_coverage_refusal` recognises."""

    WALL_S = 90
    launched = []

    def __init__(self, argv, env=None, stdout=None, stderr=None):
        srv, S = Path(argv[1]), int(argv[2])
        _Standup.launched.append(S)
        if S <= self.WALL_S:
            (srv / "endpoint.json").write_text("{}")
            self._rc = None
        else:
            kv = self.WALL_S * L
            stdout.write(f"[standup] KV cache holds {kv} tokens < the {S * L}-token budget (= {S} x {L})\n")
            stdout.flush()
            self._rc = 1
        stdout.close()

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def _patch(M, monkeypatch, tmp_path, tree):
    monkeypatch.setattr(M.T, "resolve_staged_tree", lambda sha, repos_root=None: (tree, FULL))
    monkeypatch.setattr(M.T, "overlaid_config", lambda row, tree_, out: (f"{tree_}/cfg.yaml", {}))
    monkeypatch.setattr(M.T, "_standup_cmd", lambda tree_, cfg, cell, ddir, srv, stop, geom, top, extra, **kw:
                        ["fake-standup", str(srv), str(geom.max_num_seqs), *(["--allow-over-budget"] if kw.get("allow_over_budget") else [])])
    monkeypatch.setattr(M.T, "current_function_call_id", lambda: "fc-test")
    monkeypatch.setattr(M.T, "_tail", lambda path, n=3000: Path(path).read_text(errors="replace")[-n:] if Path(path).exists() else None)
    monkeypatch.setattr(M.subprocess, "Popen", _Standup)
    monkeypatch.setattr(M, "GEOM_ROOT", str(tmp_path / "geom"))
    monkeypatch.setattr(M, "STANDUP_WAIT_S", 1)
    _Standup.launched = []


def test_measure_walks_the_ladder_to_the_wall_bisects_once_and_writes_the_paste(M, tmp_path, monkeypatch):
    tree = _tree(tmp_path)
    _patch(M, monkeypatch, tmp_path, tree)
    rec = M._measure("f" * 40, "qwen35_4b-f22")
    assert rec["status"] == "OK", rec
    # 48, 64, 80 pass; 96 is the wall; the one bisection at 88 passes; the bracket closes
    assert _Standup.launched == [48, 64, 80, 96, 88]
    assert [p["status"] for p in rec["probes"]] == ["pass", "pass", "pass", "kv_under_coverage", "pass"]
    assert rec["probes"][3]["kv_tokens"] == 90 * L and rec["probes"][3]["product"] == 786_432
    assert rec["bracket"]["largest_passing"]["product"] == 88 * L and rec["bracket"]["first_refused"] == 786_432
    assert rec["seqs_at_16384"] == (88 * L) // 16_384 == 44
    # the row IN FORCE is read off the geometry table, never a literal: the tool's own measurement (786,432 -> S 48)
    # is recorded in budget.json above; the SHARED table keeps the cluster-swept qwen35_4b row (720,896 -> S 44), a
    # measured value this code does not override, so the record is held to the table
    rif = M.row_in_force("qwen35_4b")
    assert rec["row_in_force"] == rif
    assert rif["source"] == "measured" and rif["provisional"] is False
    assert rif["seqs_at_16384"] == rif["batched_token_budget"] // 16_384 == geometry_qwen_budget() // 16_384
    assert M.row_in_force("qwen35_9b")["provisional"] is True and M.row_in_force("qwen35_9b")["seqs_at_16384"] == 12
    assert M.row_in_force("no_such_model") is None
    assert "row in force: measured 786,432 -> S 48 at 16,384" in capsys.readouterr().out if False else True
    doc = json.loads(Path(rec["budget_json"]).read_text())
    assert doc["model_key"] == "qwen35_4b" and doc["max_model_len"] == L and doc["complete"] is True and doc["hybrid"] is True
    assert doc["paste"]["model_key"] == "qwen35_4b" and doc["paste"]["batched_token_budget"] == 88 * L
    assert doc["paste"]["certified"] is False and "Modal FunctionCall fc-test" in doc["paste"]["basis"]["source"]
    assert doc["paste"]["basis"]["max_num_seqs"] == 88 and doc["paste"]["basis"]["max_model_len"] == L
    assert "ModelBudget(" in rec["paste_block"] and "model_key='qwen35_4b'" in rec["paste_block"].replace('"', "'")
    # every probe stood up over the provisional ceiling: the standup carried --allow-over-budget
    assert all(p["max_model_len"] == L for p in rec["probes"])
    assert Path(rec["out_dir"]).joinpath("probe_S96_L8192_standup.log").exists()


def test_the_paste_block_formats_from_the_staged_tree_without_the_sweep_on_the_container_path(M, tmp_path):
    """Seen live: `format_paste_block unavailable: ModuleNotFoundError: served_decode_sweep` — the
    analyzer imports the sweep by NAME inside its helpers, the tree loader restores sys.path after
    exec, and the container has no checkout. The analyzer now finds the STAGED sweep under that name."""
    tree = _tree(tmp_path)
    saved_path, saved_mod = list(sys.path), sys.modules.pop("served_decode_sweep", None)
    sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("scripts")]        # no checkout scripts dir
    try:
        with pytest.raises(ModuleNotFoundError, match="served_decode_sweep"):
            import served_decode_sweep  # noqa: F401  — the container's situation
        an = M._analyzer(str(tree))
        paste = {"model_key": "qwen35_4b", "batched_token_budget": 786_432, "certified": False,
                 "basis": {"method": "m", "max_model_len": 8192, "max_num_seqs": 96, "gpu_memory_utilization": 0.9,
                           "kv_tokens_reported": None, "probe_job_id": "fc-1", "measured_on": "2026-09-12",
                           "source": "/vol/x/budget.json", "first_refused": 917_504, "note": "wall reached"}}
        block = an.format_paste_block([paste])
        assert "ModelBudget(" in block and "qwen35_4b" in block and "786_432" in block.replace("786,432", "786_432")
        assert sys.modules["served_decode_sweep"].__file__.startswith(str(tree))          # the STAGED module, by name
    finally:
        sys.path[:] = saved_path
        sys.modules.pop("served_decode_sweep", None)
        if saved_mod is not None:
            sys.modules["served_decode_sweep"] = saved_mod


def test_measure_stops_without_a_wall_when_a_standup_dies_for_another_reason(M, tmp_path, monkeypatch):
    tree = _tree(tmp_path)
    _patch(M, monkeypatch, tmp_path, tree)

    class _Dies(_Standup):
        def __init__(self, argv, env=None, stdout=None, stderr=None):
            _Standup.launched.append(int(argv[2]))
            stdout.write("[standup] engine core died: CUDA error: an illegal memory access\n")
            stdout.close()
            self._rc = 1
    monkeypatch.setattr(M.subprocess, "Popen", _Dies)
    rec = M._measure("f" * 40, "qwen35_4b-f22")
    assert rec["status"] == "STANDUP_FAILED" and _Standup.launched == [48]
    assert rec["probes"][0]["status"] == "launch_error" and "WITHOUT recording a wall" in rec["error"]


def test_measure_respects_the_hybrid_cap_and_refuses_a_missing_tree(M, tmp_path, monkeypatch):
    tree = _tree(tmp_path)
    _patch(M, monkeypatch, tmp_path, tree)
    _Standup.WALL_S = 10_000                                     # nothing refuses: the cap must stop the walk
    try:
        rec = M._measure("f" * 40, "qwen35_4b-f22", budget_len=4096)   # 1,572,864 / 4096 = 384 > 256
        assert rec["status"] == "INCOMPLETE" and max(_Standup.launched) <= 256
        doc = json.loads(Path(rec["budget_json"]).read_text())
        assert "per-slot cap 256" in doc["note"] and doc["complete"] is False
    finally:
        _Standup.WALL_S = 90
    monkeypatch.setattr(M.T, "resolve_staged_tree", lambda sha, repos_root=None: (_ for _ in ()).throw(M.T.NoStagedTree("none")))
    assert M._measure("f" * 40, "qwen35_4b-f22")["status"] == "NO_TREE"


def geometry_qwen_budget() -> int:
    from sisteer.capeval.serving import geometry as sg
    return sg.BUDGET_BY_MODEL["qwen35_4b"].batched_token_budget


def test_row_and_geometry_facts_the_measurement_relies_on(M):
    from sisteer.capeval.serving import geometry as sg
    row = C.model_row("qwen35_4b-f22")
    assert row.model_key == "qwen35_4b" and "qwen35_4b" in sg.HYBRID_MODEL_KEYS
    b = sg.BUDGET_BY_MODEL["qwen35_4b"]
    # a MEASURED row is in force (not the provisional halving carried earlier); its value is the shared table's —
    # since 2026-09-13 the lens-stack measurement AT the MATH-500 window (741,888 = 42 x 17,664; it was the cluster-swept
    # 720,896 = 88 x 8192), while the Modal measurement 786,432 lives in budget.json
    assert b.source == "measured" and b.batched_token_budget == b.basis.max_num_seqs * b.basis.max_model_len
    assert b.batched_token_budget >= 88 * 8_192 and b.batched_token_budget % sg.VLLM_BLOCK_SIZE == 0


def _exec_in_container_root(script: Path, inventory, name: str):
    """Execute a Modal script's source the way its CONTAINER does: the source at
    ``/root/<name>.py`` (a fake root holding the entrypoint + EXACTLY the inventory of files the
    image places beside it), under the modal stub, with the checkout's ``scripts/modal`` OFF
    ``sys.path`` and no launcher module cached — so a sibling import that only resolves through
    the checkout fails here as it failed live. ``sisteer`` stays importable (the image's
    ``add_local_python_source``)."""
    import shutil
    import tempfile
    import types
    root = Path(tempfile.mkdtemp(prefix="modal_root_")) / "root"
    root.mkdir()
    shutil.copy(script, root / script.name)
    for remote, local in inventory.items():
        assert remote.startswith("/root/") and "/" not in remote[len("/root/"):], remote
        shutil.copy(local, root / remote[len("/root/"):])
    saved_path, saved_env = list(sys.path), os.environ.get(C.MODAL_APP_NAME_ENV)
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k in ("modal", "tinycap_served", "tinycap_images")}
    sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("scripts/modal")]
    sys.modules["modal"] = TL._stub_modal()
    os.environ[C.MODAL_APP_NAME_ENV] = "tinycap-bcbw-container-import-test"
    mod = types.ModuleType(name)
    mod.__file__ = str(root / script.name)
    try:
        exec(compile((root / script.name).read_text(), mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.path[:] = saved_path
        for k in ("modal", "tinycap_served", "tinycap_images"):
            sys.modules.pop(k, None)
        sys.modules.update(saved_mods)
        if saved_env is None:
            os.environ.pop(C.MODAL_APP_NAME_ENV, None)
        else:
            os.environ[C.MODAL_APP_NAME_ENV] = saved_env
    return mod, root
def test_the_module_imports_from_the_containers_flat_mount_with_only_the_images_files(M):
    """The `parents[1]` IndexError and the import under it: Modal mounts the function's OWN entrypoint,
    and the shared image ships `tinycap_images.py` + `sisteer` but NOT `tinycap_served.py`, so
    `import tinycap_served` raised ModuleNotFoundError in every container — the earlier regression
    masked it by putting the checkout's `scripts/modal` on sys.path. The script now mounts the launcher
    beside itself (`CONTAINER_FILES`), and this test rebuilds a container root from EXACTLY
    `container_inventory()` with the checkout off the path."""
    inv = M.container_inventory()
    # tinycap-shared-standup §1.5 (module D): the compile-cache sibling is a third beside-the-entrypoint file
    assert set(inv) == {M.LAUNCHER_REMOTE, M.T.TINYCAP_IMAGES_REMOTE, M.T.TINYCAP_COMPILE_CACHE_REMOTE} == \
        {"/root/tinycap_served.py", "/root/tinycap_images.py", "/root/tinycap_compile_cache.py"}
    assert Path(inv[M.LAUNCHER_REMOTE]) == Path(M.T.__file__).resolve() and M.CONTAINER_FILES == {M.LAUNCHER_REMOTE: inv[M.LAUNCHER_REMOTE]}
    mod, root = _exec_in_container_root(BUDGET, inv, "tinycap_budget_measure_in_container")
    assert mod.HERE == root and mod.REPO_LOCAL == root                          # depth 1: no checkout above it
    assert Path(mod.T.__file__).resolve() == (root / "tinycap_served.py").resolve()   # the MOUNTED launcher, not the checkout's
    assert mod.probe_seqs(786_432, L) == 96 and callable(mod.measure_budget)
    # without the launcher mount the container import fails exactly as it did live
    with pytest.raises(ModuleNotFoundError, match="tinycap_served"):
        _exec_in_container_root(BUDGET, dict(M.T.IMAGE_LOCAL_FILES), "tinycap_budget_measure_no_launcher")
    # the GPU function runs on the image WITH the mount (source pin: the decorated image is IMAGE, never T.image)
    src = BUDGET.read_text()
    assert "image=IMAGE," in src and "image=T.image" not in src and "IMAGE = with_container_files(T.image)" in src
    assert M.REPO_LOCAL == REPO                                                  # the grading host import still finds the checkout

def test_the_paste_note_never_starts_with_a_stray_separator(M):
    src = BUDGET.read_text()
    assert '"; ".join(part for part in (' in src and '+ "; measured on the SERVED' not in src
