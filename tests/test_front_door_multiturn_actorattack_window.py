"""The front door's multi-turn plan passes ActorAttack's serving window on target_attacks.sbatch's positional 8.
geometry.MULTITURN_CAP4096_WINDOWS measures crescendo and pair only; the sbatch refuses actorattack without an
explicit window ("pass max_model_len explicitly and record it"), and the first wave's two actorattack targets died in their
first minute exactly there. The plan now carries the one ActorAttack window the paper ever pinned (49,152 — the llama8b lane's
roster caps), positional 7 (cells) empty; crescendo and pair keep the measured table (no positional 8)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402
from sisteer.capeval.serving import geometry as sg  # noqa: E402


def test_actorattack_targets_carry_the_pinned_window_and_the_measured_attacks_do_not():
    assert ST.MT_ACTORATTACK_WINDOW == 49152 == sg.MULTITURN_CAP4096_WINDOWS["crescendo"]
    assert "actorattack" not in sg.MULTITURN_CAP4096_WINDOWS and "actorattack" in ST.MT_ATTACKS
    rows = ST.unit_plan("qwen35_4b_demo", "battery", families=["multiturn"], nshards=9, sha="c" * 40, clusters=["cluster_a", "cluster_b"],
                        table=str(REPO / "reproduce" / "models.yaml"))
    targets = [r for r in rows if r["script"] == "scripts/sdfbase/full22/target_attacks.sbatch"]
    assert len(targets) == 12 and {r["args"][2] for r in targets} == set(ST.MT_ATTACKS)
    for r in targets:
        atk = r["args"][2]
        if atk == "actorattack":
            assert r["args"][6:] == ["", "49152"], r["args"]          # positional 7 = every cell, positional 8 = the window
        else:
            assert len(r["args"]) == 6, r["args"]                      # the measured table sizes crescendo / pair
