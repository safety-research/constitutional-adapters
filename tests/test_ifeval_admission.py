"""ifeval-canonical-battery, fix lane X: ADMISSION before any GPU spawn, on the ACTUAL planned
configurations, at both doors.

Before this the documented dispatcher command had no ``--preflight`` (nothing checked the cached same-GPU duty
certification for the tree / geometry before spawning), ``configuration_function()`` only selected the image function,
and ``--preflight`` handed the launcher no cap, so a 4,096 / 8,192 dispatch preflighted the 2,048 entry. Now:

* the dispatcher's ``main`` ALWAYS runs the launcher's preflight per (slug, cap) over the queued rows' cells with the
  dispatch cap as the OFF setter (the geometry entry follows the cap, the prompt class the cell), persists the verdicts
  (``state.admission``), refuses to spawn ANYTHING when one slug fails, and holds every spawn to the admitted set;
  ``--no-preflight`` is the explicit, recorded opt-out;
* the container (``tinycap_served._run_configuration``) re-derives the Nemotron backend certification key for ITS
  resolved geometry / tree / reduction mode and reads the certification itself — a missing or foreign certificate is
  the REC ``NEMO_BACKEND_UNATTESTED`` BEFORE the standup, whatever the operator-side gate did.
"""
from __future__ import annotations

import json
import types
from typing import List

import pytest

from sisteer.capeval.tinycap import contracts as C

import test_tinycap_launcher as TL
from test_ifeval_dispatch import (APP, CID, D, LAUNCH, SHA, WAVE_ROOT, FakeClock, FakeRegistry, Wave, _matrix_file,
                                  make_launcher)
from test_tinycap_launcher_port import L, NEMO, QWEN, _plan  # noqa: F401  (the launcher under test + its planner)


# --------------------------------------------------------------------------- #
# the dispatcher door                                                          #
# --------------------------------------------------------------------------- #
def test_admission_is_the_default_preflights_the_dispatch_cap_and_a_failed_slug_prevents_every_spawn(tmp_path):
    """The documented command has no --preflight, and before this it spawned straight away; a `--preflight` at a rung
    preflighted the 2,048 geometry. Now `main` ALWAYS runs the preflight per (slug, cap) over the queued rows' cells with
    the dispatch cap as the OFF setter (so the plan the preflight vouches for is the rung's entry, the prompt class the
    cell's), persists the verdicts, and a single failing slug means NOTHING spawns; the admitted set is held at spawn time."""
    reg = FakeRegistry()
    L = make_launcher(reg)
    argv, _ = D.preflight_argv(L, sha=SHA, app_name=APP, slug="nemotron3nano4b-sdfbase-main",
                               cells=["none", "prompted__si_constitution"], cap=4096)
    assert argv[argv.index(C.FLAG_OFF_CAP_BY_TASK) + 1] == f"{C.IFEVAL_TASK_NAME}=4096" and argv[-1] == "--pf"
    assert argv[argv.index("--cells") + 1] == "none,prompted__si_constitution"    # the prompt classes ride the cells
    argv1, _ = D.preflight_argv(L, sha=SHA, app_name=APP, slug="nemotron3nano4b-sdfbase-main", cells=["none"])
    assert C.FLAG_OFF_CAP_BY_TASK not in argv1
    runs = []

    def failing_nemotron(argv, env=None, **kw):
        runs.append(list(argv))
        return types.SimpleNamespace(returncode=1 if "nemotron3nano4b-sdfbase-main" in argv else 0)

    mpath, local = _matrix_file(tmp_path), tmp_path / "local"
    # a 4096 rung dispatch of every launch row, WITHOUT --preflight on the command line
    rows = ",".join(r["configuration_ids"]["4096"] for r in LAUNCH)
    rc = D.main(["--matrix", str(mpath), "--sha", SHA, "--wave-root", WAVE_ROOT, "--local-root", str(local), "--app-name", APP,
                 "--rung", "4096", "--rows", rows], launcher=L, run_subprocess=failing_nemotron, log=lambda s: None)
    assert rc == D.EXIT_FAILED and reg.order == [] and L.launch_calls == []        # NOTHING spawned, no slug at all
    assert len(runs) == 7 and all(a[a.index(C.FLAG_OFF_CAP_BY_TASK) + 1] == f"{C.IFEVAL_TASK_NAME}=4096" for a in runs)
    assert not (local / D.IFEVAL_DISPATCH_STATE).is_file() or not json.loads(
        (local / D.IFEVAL_DISPATCH_STATE).read_text(encoding="utf-8")).get("active")
    # every slug admitted: the verdicts are persisted with the cap and the cells, and the wave runs
    ok_runs = []

    def admitting(argv, env=None, **kw):
        ok_runs.append(list(argv))
        return types.SimpleNamespace(returncode=0)

    def sleep(_s):
        for fc in reg.unfinished()[:8]:
            reg.finish(fc, rec={"status": "OK", "name": fc.cfg["name"], "cell": fc.cfg["cell"]})

    rc = D.main(["--matrix", str(mpath), "--sha", SHA, "--wave-root", WAVE_ROOT, "--local-root", str(local), "--app-name", APP,
                 "--rung", "4096", "--rows", rows], launcher=L, run_subprocess=admitting, sleep=sleep, clock=FakeClock(),
                adopt=lambda fc_id: reg.calls[fc_id], log=lambda s: None)
    assert rc == D.EXIT_OK and len(reg.order) == len(LAUNCH)
    st = json.loads((local / D.IFEVAL_DISPATCH_STATE).read_text(encoding="utf-8"))
    adm = st["admission"]
    assert adm["cap"] == 4096 and set(adm["slugs"]) == {r["slug"] for r in LAUNCH}
    for slug, v in adm["slugs"].items():
        assert v["ok"] and v["rc"] == 0 and v["cap"] == 4096 and v["argv"][-1] == "--pf"
        assert sorted(v["cells"]) == sorted(r["cell"] for r in LAUNCH if r["slug"] == slug)
    assert len([a for a in ok_runs if a[-1] == "--pf"]) == 7
    # the spawn-time hold: a dispatcher whose admitted set lacks a queued slug refuses to spawn it
    w = Wave(tmp_path / "hold", admitted={LAUNCH[0]["slug"]: {"ok": True}})
    with pytest.raises(RuntimeError, match="was not admitted at cap 2048"):
        w.run()
    assert all(fc.cfg["model"] == LAUNCH[0]["slug"] for fc in w.registry.calls.values())   # only the admitted slug spawned
    # --no-preflight: the explicit opt-out, recorded as such
    reg2 = FakeRegistry()
    L2 = make_launcher(reg2)
    seen = []
    rc = D.main(["--matrix", str(mpath), "--sha", SHA, "--wave-root", WAVE_ROOT, "--local-root", str(tmp_path / "opt"),
                 "--app-name", APP, "--rows", CID[(LAUNCH[0]["slug"], LAUNCH[0]["cell"])], "--no-preflight"],
                launcher=L2, run_subprocess=lambda *a, **k: (seen.append(a), types.SimpleNamespace(returncode=0))[1],
                sleep=lambda s: [reg2.finish(fc, rec={"status": "OK", "name": fc.cfg["name"], "cell": fc.cfg["cell"]})
                                 for fc in reg2.unfinished()], clock=FakeClock(), adopt=lambda i: reg2.calls[i], log=lambda s: None)
    assert rc == D.EXIT_OK and len(reg2.order) == 1 and not any(a and a[0] and a[0][-1] == "--pf" for a in seen)
    st2 = json.loads((tmp_path / "opt" / D.IFEVAL_DISPATCH_STATE).read_text(encoding="utf-8"))
    assert st2["admission"]["slugs"][LAUNCH[0]["slug"]]["opt_out"] == "--no-preflight"


# --------------------------------------------------------------------------- #
# the container door                                                           #
# --------------------------------------------------------------------------- #
def test_the_container_refuses_an_uncertified_nemotron_standup_itself_before_paying_for_it(L, tmp_path, monkeypatch):
    """Fix lane X: the worker-side, fail-closed twin of the preflight's `nemotron_backend_attested`. A
    nemotron_h container re-derives the certification key for ITS resolved geometry / tree / reduction mode and reads the
    same-GPU duty certification off the volume; no certificate (or a foreign one) is the REC `NEMO_BACKEND_UNATTESTED`
    BEFORE any standup — whatever the operator-side gate did. A non-Nemotron row certifies nothing (None)."""
    from sisteer.capeval.serving import mamba_backend as mb
    row = C.model_row(NEMO)
    (cfg,) = _plan(L, NEMO, out_dir=str(tmp_path / "out"))
    assert L.nemotron_worker_admission(C.model_row(QWEN), tree="/nowhere", tree_sha="a" * 40,
                                       geom=L.configuration_geometry(cfg, 1), reduction_mode=C.REDUCTION_MODE_STACKED) is None
    repos = tmp_path / "repos"
    tree = TL._tree(repos, "f6cf462")
    geom = L.configuration_geometry(cfg, 1)
    # no reviewed contract on the tree: the key cannot be derived -> refused, never admitted
    adm = L.nemotron_worker_admission(row, tree=str(tree), tree_sha=TL.FULL_SHA, geom=geom, reduction_mode=C.REDUCTION_MODE_STACKED)
    assert adm["ok"] is False and adm["cache_key"] is None and "could not be derived" in adm["problems"][0]
    # the contract and the row's overlay config present (both are key inputs), the certification absent: the key
    # is derived and the read refuses by name
    # the staged tree carries the reviewed contract, the row's overlay config (both key inputs) and EVERY
    # registered IFEval matrix document (contracts.IFEVAL_MATRIX_DOCUMENTS: the frozen matrix + the batch-64 seed
    # sibling) the container's hold door reads BEFORE the Nemotron admission (a tree missing any of them refuses
    # `IFEVAL_ROW_HELD` first, as a real staged tree never would: the documents are tracked files of every tree)
    for rel in (mb.CONTRACT_RELPATH, row.cfg, *C.IFEVAL_MATRIX_DOCUMENTS):
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((TL.REPO / rel).read_bytes())
    monkeypatch.setattr(L, "NEMO_BACKEND_ROOT", str(tmp_path / "nemo_backend"))
    adm = L.nemotron_worker_admission(row, tree=str(tree), tree_sha=TL.FULL_SHA, geom=geom, reduction_mode=C.REDUCTION_MODE_STACKED)
    assert adm["ok"] is False and adm["cache_key"] == L.nemotron_attestation_key(
        row, tree=str(tree), tree_sha=TL.FULL_SHA, geom=geom, reduction_mode=C.REDUCTION_MODE_STACKED)
    assert any("no Nemotron backend attestation" in p for p in adm["problems"])
    # ... and the geometry is IN the key: another cap's entry is another certificate
    (cfg4,) = _plan(L, NEMO, off_cap_by_task={"ifeval": 4096}, out_dir=str(tmp_path / "out4"))
    adm4 = L.nemotron_worker_admission(row, tree=str(tree), tree_sha=TL.FULL_SHA, geom=L.configuration_geometry(cfg4, 1),
                                       reduction_mode=C.REDUCTION_MODE_STACKED)
    assert adm4["cache_key"] != adm["cache_key"]
    # the container path: _run_configuration reaches the admission and refuses BEFORE the standup
    monkeypatch.setattr(L.C, "VOL_REPOS_ROOT", str(repos))
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, C.REDUCTION_MODE_STACKED)
    monkeypatch.setattr(L, "_standup_cmd", lambda *a, **k: (_ for _ in ()).throw(AssertionError("standup composed for an uncertified Nemotron")))
    rec = L._run_configuration(cfg, "f6cf462", controls=[C.CONTROL_IFEVAL], samples=2, reduction_mode=C.REDUCTION_MODE_STACKED)
    assert rec["status"] == C.REC_NEMO_BACKEND_UNATTESTED, rec
    assert rec["nemotron_backend_admission"]["ok"] is False and rec["nemotron_backend_admission"]["cache_key"] == adm["cache_key"]
    assert "no admitting Nemotron backend certification" in rec["error"]
    src = TL.LAUNCHER.read_text(encoding="utf-8")
    assert src.index("admission = nemotron_worker_admission(") < src.index("object_sha256 = \"\"") < src.index("standup_argv = _standup_cmd(")
