"""ifeval-canonical-battery: the remote planners read the frozen IFEval matrix off the STAGED tree.

``plan_configurations`` defaulted the matrix to ``C.load_ifeval_matrix(REPO_LOCAL)``. Inside a Modal container
``REPO_LOCAL`` is ``_repo_local()``'s fallback — the launcher's mounted directory, which the image builds from the sibling
module and ``sisteer`` (``add_local_python_source``) with no ``configs/`` — so every ordinary IFEval preflight raised
``IfevalMatrixUnreadable`` before its first check, and the Nemotron probe planned BEFORE resolving its staged tree and
failed the same way inside its GPU allocation. Now the matrix's source is explicit (``ifeval_matrix_tree``, threaded
through ``plan_for`` / ``smoke_plan`` / ``probe_geometry_cfg``): the preflight plans against its staged ``tree``, the probe
resolves the staged tree FIRST and plans against it (a refusing or unreadable matrix there is the wave container's own
``IFEVAL_ROW_HELD`` REC), and the local callers (``main``, the dispatcher's ``plan_row``) keep the checkout default.

The tests below build the CONTAINER's layout: an empty launcher mount as ``REPO_LOCAL`` and the matrix present only in the
staged tree under ``VOL_REPOS_ROOT``.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import contracts as C

import test_tinycap_launcher as TL
from test_tinycap_launcher_port import L, NEMO, QWEN, REPO  # noqa: F401

SHA = "f6cf462"
STACKED, VENDOR = C.REDUCTION_MODE_STACKED, C.SAMPLING_VENDOR_PRESET


def _ifeval_plan_args(model: str = QWEN, **kw) -> dict:
    """What the dispatcher's ``--pf`` hands the launcher's ``main``, which hands it to ``preflight`` verbatim."""
    return {**dict(smoke=False, cells=["none"], modes=[C.THINKING_OFF], controls=[C.CONTROL_IFEVAL], model=model,
                   sampling_mode=VENDOR, samples=C.IFEVAL_SAMPLES_PER_ITEM, reduction_mode=STACKED, pass_no=1), **kw}


def _container_layout(L, tmp_path: Path, monkeypatch, *, matrix=True) -> tuple[Path, Path, Path]:
    """The filesystem as the launcher PROCESS sees it inside a container: ``REPO_LOCAL`` = the mounted directory (the
    sibling module, no ``configs/``), and the staged tree under ``VOL_REPOS_ROOT`` — carrying the frozen matrix (a real
    staged tree always does: it is a tracked file), or, under ``matrix=False``, a tree staged before the matrix existed."""
    mount = tmp_path / "root"
    mount.mkdir()
    (mount / "tinycap_images.py").write_text("# the sibling module the image copies beside the launcher\n")
    monkeypatch.setattr(L, "REPO_LOCAL", mount)
    repos = tmp_path / "repos"
    tree = TL._tree(repos, SHA)
    if matrix:
        _stage_matrix(tree, json.loads((REPO / C.IFEVAL_MATRIX_REL).read_text(encoding="utf-8")))
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setattr(L.C, "VOL_GEN_ROOT", str(tmp_path / "gen"))       # the preflight's one writer (out_root_writable)
    return mount, repos, tree


def _stage_matrix(tree: Path, doc: dict) -> Path:
    """The frozen matrix as ``doc`` — and, beside it, every OTHER registered document verbatim (the planners read
    ``contracts.IFEVAL_MATRIX_DOCUMENTS`` merged; a real staged tree carries them all as tracked files)."""
    dst = tree / C.IFEVAL_MATRIX_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc), encoding="utf-8")
    for rel in C.IFEVAL_MATRIX_DOCUMENTS:
        if rel != C.IFEVAL_MATRIX_REL:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / rel, tree / rel)
    return dst


def _no_client_venv() -> types.SimpleNamespace:
    """``subprocess`` as the launcher sees it with NO client venv: every process launch fails at once (the real
    ``/opt/client/bin/python`` is absent under test too; this makes it deterministic), every other name real."""
    def absent(*_a, **_k):
        raise FileNotFoundError("no client venv under test")
    ns = {k: getattr(subprocess, k) for k in dir(subprocess) if not k.startswith("_")}
    ns.update(run=absent, Popen=absent, check_output=absent, check_call=absent, call=absent)
    return types.SimpleNamespace(**ns)


def test_the_planners_default_reads_the_checkout_and_the_empty_mount_reproduces_the_container_failure(L, tmp_path, monkeypatch):
    mount, repos, tree = _container_layout(L, tmp_path, monkeypatch)
    assert not (mount / C.IFEVAL_MATRIX_REL).exists() and (tree / C.IFEVAL_MATRIX_REL).is_file()
    # the reproduction: with REPO_LOCAL the launcher's mount, the default source has no matrix — the ordinary IFEval
    # plan (what `plan_for(**plan_args)` did inside the container) refuses before anything else
    with pytest.raises(C.IfevalMatrixUnreadable, match="cannot be read") as e:
        L.plan_for(**_ifeval_plan_args())
    assert str(mount / C.IFEVAL_MATRIX_REL) in str(e.value)
    with pytest.raises(C.IfevalMatrixUnreadable):
        L.probe_geometry_cfg(C.model_row(NEMO), "2048", reduction_mode=STACKED, sampling_mode=VENDOR)
    # the source named: the same plans read the staged tree's matrix and carry its verdict
    (cfg,) = L.plan_for(**_ifeval_plan_args(), ifeval_matrix_tree=tree)
    assert cfg["matrix_status"] == C.IFEVAL_STATUS_LAUNCH and cfg["name"] == C.container_name("none", C.THINKING_OFF, model_slug=QWEN)
    (smoke,) = L.plan_for(**_ifeval_plan_args(smoke=True), ifeval_matrix_tree=tree)       # smoke_plan threads it too
    assert smoke["cell"] == C.CELL_PSRKL_SAFETY and smoke["matrix_status"] is None           # PSR: out of the matrix, admitted
    probe = L.probe_geometry_cfg(C.model_row(NEMO), "longsi_4096", reduction_mode=STACKED, sampling_mode=VENDOR,
                                 ifeval_matrix_tree=tree)
    assert probe["cell"] == C.CELL_PROMPTED_CONSTITUTION and probe["matrix_status"] == C.IFEVAL_STATUS_LAUNCH
    # `ifeval_matrix_tree` is a SOURCE, not a planning input: it is keyword-only, defaults to None (the checkout) on
    # every planner, and never rides `plan_args`
    import inspect
    for fn in (L.plan_for, L.smoke_plan, L.plan_configurations, L.probe_geometry_cfg):
        p = inspect.signature(fn).parameters["ifeval_matrix_tree"]
        assert p.default is None and p.kind is inspect.Parameter.KEYWORD_ONLY, fn
    assert "ifeval_matrix_tree" not in L.ifeval_plan_args_of(off_cap_by_task="ifeval=4096", item_subset="/s", bank_view="it", out_dir="/o")
    # a non-IFEval plan never reads the matrix from either source
    L.plan_for(cells=["none"], modes=[C.THINKING_OFF], controls=["gsm8k"], model=QWEN)
    L.plan_for(cells=["none"], modes=[C.THINKING_OFF], controls=["gsm8k"], model=QWEN, ifeval_matrix_tree=tmp_path / "nowhere")
    # the geometry token parses once, before any tree is read
    assert L.probe_geometry_cap("2048") == (False, 2048) and L.probe_geometry_cap("longsi_8192") == (True, 8192)
    with pytest.raises(ValueError, match="--geometry must be one of"):
        L.probe_geometry_cap("16384")


def test_the_preflight_plans_an_ifeval_launch_from_an_empty_mount_against_the_staged_trees_matrix(L, tmp_path, monkeypatch):
    mount, repos, tree = _container_layout(L, tmp_path, monkeypatch)
    monkeypatch.setattr(L, "subprocess", _no_client_venv())
    plan_args = _ifeval_plan_args()
    # the REAL preflight body, in the container's layout: it plans (against the staged tree), records the mapping it
    # planned from unchanged, and reaches its checks — the tree checks pass, the venv-bound ones fail as they would
    # with no venv; no IfevalMatrixUnreadable anywhere
    rep = L.preflight(SHA, plan_args)
    assert rep["configurations"] == [C.container_name("none", C.THINKING_OFF, model_slug=QWEN)]
    assert rep["plan_args"] == plan_args and "ifeval_matrix_tree" not in rep["plan_args"]
    # tinycap-shared-standup §1.9.1 (module E): the check also states how the tree sits on the volume
    assert {k: rep["checks"]["tree"][k] for k in ("ok", "tree")} == {"ok": True, "tree": str(tree)}
    assert rep["checks"]["tree"]["on_volume"] == L.TREE_ON_VOLUME_EXTRACTED_LEGACY
    assert rep["checks"]["tree_sha_stamp"] == {"ok": True, "tree_sha": TL.FULL_SHA, "file": C.TREE_SHA_FILENAME}
    assert "IfevalMatrixUnreadable" not in json.dumps(rep, default=str)
    assert rep["cells"] == ["none"] and rep["model"] == QWEN
    # a rung preflight (the dispatcher's `--off-cap-by-task ifeval=4096`) plans the rung's entry the same way
    rep4 = L.preflight(SHA, _ifeval_plan_args(off_cap_by_task={C.IFEVAL_TASK_NAME: 4096}))
    assert rep4["configurations"] == rep["configurations"] and rep4["checks"]["tree"]["ok"]
    # the preflight's hold door is the STAGED tree's matrix, not the mount's: a staged matrix holding the row refuses
    # the plan by name (a launch row on the checkout would not have)
    _stage_matrix(tree, {"rows": [{"slug": QWEN, "cell": "none", "status": C.IFEVAL_STATUS_HELD, "held_reason": "test hold"}]})
    with pytest.raises(ValueError, match="HELD"):
        L.preflight(SHA, plan_args)
    # a NON-IFEval preflight on a tree staged before the matrix existed never reads it
    shutil.rmtree(tree)
    TL._tree(repos, SHA)
    rep_g = L.preflight(SHA, dict(smoke=False, cells=["none"], modes=[C.THINKING_OFF], controls=["gsm8k"], model=QWEN))
    assert rep_g["configurations"] and rep_g["checks"]["tree"]["ok"]
    # an IFEval preflight on that tree refuses by the matrix's name (the tree is there; the document is not)
    with pytest.raises(C.IfevalMatrixUnreadable, match="cannot be read"):
        L.preflight(SHA, plan_args)
    # no staged tree at all: the staging remedy, not a matrix message
    shutil.rmtree(repos)
    with pytest.raises(FileNotFoundError, match="no staged tree at .* --upload-tree --stage"):
        L.preflight(SHA, plan_args)
    # the preflight body plans from the mapping with the staged tree as the matrix's source (the one planning call)
    src = TL.LAUNCHER.read_text(encoding="utf-8")
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert body.count("cfgs = plan_for(") == 1 and "cfgs = plan_for(**plan_args, ifeval_matrix_tree=tree)" in body


def test_the_nemotron_probe_resolves_its_staged_tree_before_planning_and_plans_against_its_matrix(L, tmp_path, monkeypatch):
    from sisteer.capeval.serving import mamba_backend as mb
    mount, repos, tree = _container_layout(L, tmp_path, monkeypatch)
    row = C.model_row(NEMO)
    # the probe in the container's layout: the tree resolved, the configuration planned against ITS matrix, then the
    # contract step — absent on this fake tree, so the named backend refusal; never IfevalMatrixUnreadable, and
    # never a raise inside the GPU allocation
    rec = L._nemotron_probe(SHA, "2048", STACKED, VENDOR, row=row)
    assert rec["status"] == C.REC_NEMO_BACKEND_UNATTESTED, rec
    assert str(tree / mb.CONTRACT_RELPATH) in rec["error"] and "IfevalMatrixUnreadable" not in rec["error"]
    # the plan is held to the STAGED tree's matrix: a staged matrix holding the probe's row files the container's REC
    _stage_matrix(tree, {"rows": [{"slug": NEMO, "cell": "none", "status": C.IFEVAL_STATUS_HELD, "held_reason": "test hold"}]})
    held = L._nemotron_probe(SHA, "2048", STACKED, VENDOR, row=row)
    assert held["status"] == C.REC_IFEVAL_ROW_HELD and C.is_refusal_status(held["status"])
    assert "HELD" in held["error"] and str(tree) in held["error"] and held["standups"] == []
    # a staged tree with NO matrix: the same REC, by the document's name (never admitted around a missing document)
    (tree / C.IFEVAL_MATRIX_REL).unlink()
    unread = L._nemotron_probe(SHA, "2048", STACKED, VENDOR, row=row)
    assert unread["status"] == C.REC_IFEVAL_ROW_HELD and "cannot be read" in unread["error"]
    # no staged tree: NO_TREE — resolution PRECEDES planning (before this the plan raised first, and the REC never came)
    shutil.rmtree(repos)
    none = L._nemotron_probe(SHA, "2048", STACKED, VENDOR, row=row)
    assert none["status"] == "NO_TREE" and "run --upload-tree --stage" in none["error"]
    # a bad --geometry is an operator error, refused before any tree is read (no REC, whatever the volume holds)
    with pytest.raises(ValueError, match="--geometry must be one of"):
        L._nemotron_probe(SHA, "16384", STACKED, VENDOR, row=row)
    # a non-Nemotron row still refuses first (the port test's pin)
    with pytest.raises(ValueError, match="certifies a 'nemotron_h' row"):
        L._nemotron_probe(SHA, "2048", STACKED, VENDOR, row=C.model_row(QWEN))
    # source order in the probe body: geometry parse < tree resolution < the plan (with the staged tree) < the contract
    src = TL.LAUNCHER.read_text(encoding="utf-8")
    body = src[src.index("def _nemotron_probe("):src.index("def nemotron_backend_probe(")]
    i_geom = body.index("probe_geometry_cap(geometry)")
    i_tree = body.index("tree_path, tree_sha = resolve_staged_tree(sha)")
    i_plan = body.index("cfg = probe_geometry_cfg(row, geometry, reduction_mode=reduction_mode, sampling_mode=sampling_mode,\n"
                        "                                 ifeval_matrix_tree=tree)")
    i_contract = body.index("contract_path = Path(tree) / mb.CONTRACT_RELPATH")
    assert i_geom < i_tree < i_plan < i_contract
    assert body.count("probe_geometry_cfg(") == 1
