"""The multi-turn target's decode-policy ref handed to the driver is THE served model's vendor row, spelled by the driver's own
composer — never a string a planner typed. On 2026-09-30 the u0 wave's plan handed every multi-turn driver the unversioned
`<model_id>|multi_turn|off|vendor_preset` while the registry's vendor row is `…@v1`, so each driver refused (`policy ref … is not the
served model's multi_turn vendor row …@v1`) and no multi-turn target generated a goal. The plan now derives the ref through
target_decode.vendor_preset_policy_ref over the export's model-id mapping, and this test runs the driver's OWN predicate
(registry_vendor_row(..., expect_ref=<the plan's ref>)) against the plan's output — the two can no longer disagree."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sisteer.eval.attacks import target_decode as TD  # noqa: E402
from sisteer.registry import decode_policy as DP  # noqa: E402

PY = os.environ.get("SISTEER_TEST_PYTHON") or (str(REPO / ".venv/bin/python") if (REPO / ".venv/bin/python").exists() else sys.executable)
MODEL = "qwen35_4b_demo"
SHA = "c" * 40


def _plan(families: str):
    r = subprocess.run([PY, str(REPO / "reproduce/lib/stage.py"), "plan", "--model", MODEL, "--view", "battery", "--families", families, "--nshards", "9",
                        "--sha", SHA, "--clusters", "cluster_a", "--models-table", str(REPO / "reproduce/models.yaml"), "--json"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(r.stdout)


def test_the_plans_multiturn_policy_ref_is_the_served_models_vendor_row_by_the_drivers_own_predicate():
    rows = _plan("multiturn")
    targets = [r for r in rows if r["kind"] == "target"]
    assert targets, rows
    refs = {r["env"].get("F22_MT_TARGET_POLICY_REF") for r in targets}
    assert len(refs) == 1, refs
    (ref,) = refs
    expected = TD.vendor_preset_policy_ref(DP.load().model_id(MODEL), "multi_turn")
    assert ref == expected == f"{MODEL}|multi_turn|off|vendor_preset@v{TD.VENDOR_PRESET_POLICY_VERSION}", (ref, expected)
    # the driver's own check, run on the plan's ref: the REF is never what it refuses (the row's own sampling-key contract is a
    # separate refusal of the registry seed — `does not SEND [...]` — reported by name when it fires; it is not this plan's fault)
    try:
        row = TD.registry_vendor_row(MODEL, "multi_turn", expect_ref=ref)
        assert row.ref == ref == row.policy.policy_id, row                # VendorRow exposes .ref and .policy.policy_id (lane A's hand-back)
    except TD.VendorPolicyRefused as exc:
        assert "is not the served model's" not in str(exc), str(exc)
        assert "does not SEND" in str(exc), str(exc)          # the only other refusal a resolved row can raise here: the seed row's contract
    # the ref the plan used to type by hand (unversioned) is exactly what the driver refuses — the defect stays pinned as wrong
    with pytest.raises(TD.VendorPolicyRefused, match="is not the served model's multi_turn vendor row"):
        TD.registry_vendor_row(MODEL, "multi_turn", expect_ref=f"{MODEL}|multi_turn|off|vendor_preset")
    # the sbatch forwards the env value verbatim to the driver flag the predicate guards
    sb = (REPO / "scripts/sdfbase/full22/target_attacks.sbatch").read_text()
    assert '--multiturn-target-policy-ref "$F22_MT_TARGET_POLICY_REF"' in sb
