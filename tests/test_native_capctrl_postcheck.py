"""The sbatch's DRIVER post-check, validated on CPU against a store the REAL
driver writes.

``native_capctrl_smoke.sbatch`` runs ``served_capability_run.py generate --controls mmlu`` on the native capctrl
standup and judges its store. The inline check it shipped with required ``manifest.route.serving_engine`` and
``manifest.route.reduction_mode`` — but the legacy MMLU control produces no tinycap atoms, so ``_tinycap_manifest()``
(the one writer of ``route``) never runs, and a CORRECT run would have FAILED the job after spending its GPU
allocation. The check now lives in ``scripts/sdfbase/native_capctrl_postcheck.py`` and is exercised here over the
manifest / evidence the real driver writes: the lora-route world of ``test_served_capability_native`` (its
``native_world`` fixture, imported as a fixture) drives ``served_capability_run.main`` with ``--controls mmlu`` — the
producer's shape, never a hand-written fixture — and the module must find NOTHING wrong with it, name every defect a
mutation introduces, and be what the sbatch calls.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))

import native_capctrl_postcheck as pc  # noqa: E402
import served_capability_run as srun  # noqa: E402
import test_served_capability_native as NW  # noqa: E402
from test_served_capability_native import native_world  # noqa: E402,F401  (the lora-route world, as a fixture)

from sisteer.capeval.serving import native_standup as ns  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402

NATIVE = tc.SERVING_ENGINE_LORA_NATIVE
SBATCH = REPO / "scripts/sdfbase/native_capctrl_smoke.sbatch"
MODULE = REPO / "scripts/sdfbase/native_capctrl_postcheck.py"


def _controls_record(passed: bool = True) -> dict:
    """The positive-controls record as ``native_standup.run_positive_controls`` initialises and finalises it (the
    ``controls`` map over ``CONTROL_NAMES``, the top-level ``passed`` / ``failed_control`` / ``error``)."""
    names = list(ns.CONTROL_NAMES)
    return {"engine": ns.ENGINE, "order": names,
            "controls": {n: {"passed": passed} for n in names},
            "passed": passed, "failed_control": None if passed else names[0],
            "error": None if passed else "RuntimeError: synthetic"}


def _mmlu_store(world):
    """Run the REAL driver with ``--controls mmlu`` over the native world; write the standup's controls record at the
    path its endpoint names. Returns ``(world, endpoint dict, endpoint path)``."""
    argv = [a if a != NW.LENS.LEGACY_GEN_CONTROLS else "mmlu" for a in world.argv]
    srun.main(argv)
    ep_path = world.table_path.parent / srun.STANDUP_ENDPOINT_FILENAME
    ep = json.loads(ep_path.read_text())
    Path(ep["positive_controls"]).write_text(json.dumps(_controls_record()))
    return world, ep, ep_path


def _manifest_path(world) -> Path:
    [m] = list((world.out_root / "cells").rglob("manifest.json"))
    return m


def _pre_fix_inline_check(store: Path) -> list:
    """The check the sbatch shipped with, verbatim in logic: it read the engine and the reduction
    mode off ``manifest.route`` — a block only a tinycap (atoms) manifest carries."""
    problems = []
    for m in Path(store).rglob("manifest.json"):
        d = json.loads(m.read_text())
        route = d.get("route") or {}
        if route.get("serving_engine") != "lora_native":
            problems.append(f"{m}: route.serving_engine={route.get('serving_engine')!r}")
        if route.get("reduction_mode") != "not_applicable":
            problems.append(f"{m}: route.reduction_mode={route.get('reduction_mode')!r}")
    return problems


def test_postcheck_passes_on_the_store_the_real_mmlu_driver_writes_and_the_pre_fix_check_did_not(native_world):
    w, ep, ep_path = _mmlu_store(native_world)
    # the finding, pinned on the producer's bytes: no `route` block, the engine top-level and in the hashed identity
    d = json.loads(_manifest_path(w).read_text())
    assert tc.MANIFEST_KEY_ROUTE not in d
    assert d[tc.SERVED_ENGINE_FIELD] == NATIVE and d["resolved_config"]["serving"][tc.SERVED_ENGINE_FIELD] == NATIVE
    assert _pre_fix_inline_check(w.out_root) == [f"{_manifest_path(w)}: route.serving_engine=None",
                                                 f"{_manifest_path(w)}: route.reduction_mode=None"]
    # the module: nothing wrong with a correct native MMLU run, and the facts it records are the real ones
    out = pc.driver_store_problems(w.out_root, endpoint=ep, cell=NW.TT_CELL)
    assert out["problems"] == [], out
    f = out["facts"]
    assert f["n_manifests"] == 1
    man = f[_manifest_path(w).parent.name]
    assert man["serving_engine"] == man["hashed_serving_engine"] == NATIVE and man["route"] is None
    assert man["serving_identity"] == NW.LORA_ID and man["serving_path"] == "lora" and man["steer_positions"] == "all"
    assert man["mmlu"]["response_prefill"] is False and man["cell_id"] == NW.TT_CELL
    assert f["evidence"]["steering_bites"]["twin"] == tc.TWIN_CELL_ID_LORA == f["evidence"]["logprob_steering_bites"]["twin"]
    assert f["evidence"]["mmlu_scoring"] == pc.MMLU_SCORING_LORA
    assert f["endpoint"]["reduction_mode"] == tc.REDUCTION_MODE_NOT_APPLICABLE and f["endpoint"]["engine"] == NATIVE
    assert f["positive_controls"]["passed"] is True
    # the CLI the sbatch runs: the same verdict on stdout (the sbatch matches the literal `"problems": []`), exit 0
    r = subprocess.run([sys.executable, str(MODULE), "--store", str(w.out_root), "--endpoint", str(ep_path),
                        "--cell", NW.TT_CELL], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0 and '"problems": []' in r.stdout, r.stdout + r.stderr
    assert json.loads(r.stdout)["facts"]["n_manifests"] == 1


def _mut_manifest(key, value):
    def mutate(w, ep):
        m = _manifest_path(w)
        d = json.loads(m.read_text())
        cur = d
        for k in key[:-1]:
            cur = cur[k]
        if value is _ABSENT:
            del cur[key[-1]]
        else:
            cur[key[-1]] = value
        m.write_text(json.dumps(d))
        return ep
    return mutate


def _mut_evidence(key, value):
    def mutate(w, ep):
        p = w.out_root / "evidence.json"
        d = json.loads(p.read_text())
        cur = d
        for k in key[:-1]:
            cur = cur[k]
        cur[key[-1]] = value
        p.write_text(json.dumps(d))
        return ep
    return mutate


def _mut_endpoint(key, value):
    def mutate(w, ep):
        ep = copy.deepcopy(ep)
        if value is _ABSENT:
            del ep[key]
        else:
            ep[key] = value
        return ep
    return mutate


def _mut_controls(passed):
    def mutate(w, ep):
        Path(ep["positive_controls"]).write_text(json.dumps(_controls_record(passed)))
        return ep
    return mutate


def _mut_controls_missing(w, ep):
    Path(ep["positive_controls"]).unlink()
    return ep


def _mut_evidence_missing(w, ep):
    (w.out_root / "evidence.json").unlink()
    return ep


_ABSENT = object()

DEFECTS = [
    ("manifest top-level engine absent", _mut_manifest((tc.SERVED_ENGINE_FIELD,), _ABSENT),
     "top-level serving_engine=None"),
    ("manifest hashed engine absent", _mut_manifest(("resolved_config", "serving", tc.SERVED_ENGINE_FIELD), _ABSENT),
     "resolved_config.serving.serving_engine=None"),
    ("manifest steer_positions response", _mut_manifest(("steer_positions",), "response"),
     "steer_positions='response'"),
    ("manifest mmlu declaration sent", _mut_manifest(("resolved_config", "mmlu", "response_prefill"), True),
     "response_prefill=True"),
    ("manifest passthrough identity", _mut_manifest(("serving_identity",), NW.PASS_ID),
     "is not a +lora+ route"),
    ("manifest other cell", _mut_manifest(("cell_id",), "none"), "cell_id='none' != the served cell"),
    ("manifest row engine column", _mut_manifest(("row_constants", tc.SERVED_ENGINE_FIELD), NATIVE),
     "row_constants carries serving_engine"),
    ("manifest tinycap route naming lens", _mut_manifest((tc.MANIFEST_KEY_ROUTE,),
                                                          {tc.SERVED_ENGINE_FIELD: "lens",
                                                           tc.REDUCTION_MODE_FIELD: "per_row"}),
     "route.serving_engine='lens'"),
    ("evidence bites unverified", _mut_evidence(("steering_bites", "verified"), False),
     "steering_bites applicable=True verified=False"),
    ("evidence lens twin", _mut_evidence(("steering_bites", "twin"), "none__lens0"),
     "steering_bites.twin='none__lens0'"),
    ("evidence mmlu declaration", _mut_evidence(("mmlu_scoring", "declaration_sent"), True), "mmlu_scoring="),
    ("evidence logprob bites unverified", _mut_evidence(("logprob_steering_bites", "verified"), False),
     "logprob_steering_bites applicable=True verified=False"),
    ("evidence missing", _mut_evidence_missing, "evidence.json: missing"),
    ("endpoint lens mode", _mut_endpoint(tc.REDUCTION_MODE_FIELD, "per_row"), "reduction_mode='per_row'"),
    ("endpoint torn", _mut_endpoint("lens_reduction_mode", "stacked"), "states lens_reduction_mode"),
    ("endpoint lens engine", _mut_endpoint(tc.ENDPOINT_KEY_ENGINE, _ABSENT), "engine=None"),
    ("controls failed", _mut_controls(False), "passed=False (failed_control="),
    ("controls record missing", _mut_controls_missing, "positive-controls record is missing"),
]


@pytest.mark.parametrize("label, mutate, needle", DEFECTS, ids=[d[0] for d in DEFECTS])
def test_postcheck_names_each_defect(native_world, label, mutate, needle):
    w, ep, _ = _mmlu_store(native_world)
    ep = mutate(w, ep)
    out = pc.driver_store_problems(w.out_root, endpoint=ep, cell=NW.TT_CELL)
    joined = "\n".join(out["problems"])
    assert out["problems"] and needle in joined, (label, joined)


def test_postcheck_refuses_an_empty_store_and_an_unreadable_manifest(tmp_path):
    out = pc.driver_store_problems(tmp_path, endpoint=None, cell=None)
    assert any("wrote no manifest.json" in p for p in out["problems"])
    assert any("evidence.json: missing" in p for p in out["problems"])
    (tmp_path / "cells" / "x").mkdir(parents=True)
    (tmp_path / "cells" / "x" / "manifest.json").write_text("{not json")
    out = pc.driver_store_problems(tmp_path, endpoint=None, cell=None)
    assert any("unreadable" in p for p in out["problems"])
    assert pc.main(["--store", str(tmp_path)]) == 1
    assert pc.main(["--store", str(tmp_path), "--endpoint", str(tmp_path / "cells" / "x" / "manifest.json")]) == 1


def test_sbatch_runs_the_module_and_keeps_its_standup_assertions():
    src = SBATCH.read_text()
    assert subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True).returncode == 0
    # the driver post-check is the module, handed the store, the standup's endpoint and the cell
    assert ('"$MPY" scripts/sdfbase/native_capctrl_postcheck.py --store "$STORE" --endpoint "$EP" --cell "$CELL"'
            in src)
    assert 'case "$DRV" in *\'"problems": []\'*) check driver PASS "$DRV";; *) check driver FAIL "$DRV";; esac' in src
    # the inline check that required a tinycap route block on the legacy MMLU manifest is gone
    assert 'route.get("serving_engine")' not in src and 'd.get("route")' not in src
    # the standup post-check's assertions stand
    for needle in ("geometry_adjustment_problem", "served_engine_problem", "attested_enforce_eager",
                   "kernel_ceiling_adjustment", "mamba_backend_attestation", "endpoint_attestation_problems",
                   'ep.get("engine") != "lora_native" or ep.get("serving_engine") != "lora_native"',
                   "a native endpoint carries lens_reduction_mode", '"positive_controls"'):
        assert needle in src, needle
    # and the module itself compiles under the driver venv's python
    assert subprocess.run([sys.executable, "-m", "py_compile", str(MODULE)], capture_output=True).returncode == 0
