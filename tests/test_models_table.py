"""reproduce/models.yaml through its one reader: duplicate keys refused, both checkpoint ids and both
40-hex revisions retained for every model, the eligible-layer rule, the module-set vocabulary, the per-expert rank rule."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reproduce/lib"))
import models as M  # noqa: E402


def test_table_checks_clean():
    assert M.check() == []


def test_every_model_keeps_both_checkpoints_and_revisions():
    for key, m in M.load_models().items():
        assert m["fit"]["checkpoint"]["model_id"] and m["serve"]["checkpoint"]["model_id"], key
        assert M.HEX40.match(m["fit"]["checkpoint"]["revision"]) and M.HEX40.match(m["serve"]["checkpoint"]["revision"]), key


def test_duplicate_keys_are_refused(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("schema: 1\nmodels:\n  x:\n    fit: {a: 1}\n    fit: {a: 2}\n")
    with pytest.raises(M.DuplicateKeyError):
        M.load_models(str(p))


OBJECTS = {"control_lora", "constitution_lora", "control_relu", "constitution_relu", "difference_lora", "difference_relu", "refusal_direction"}
PHASE_VIEWS = {"battery", "d0p75", "d0p75_relu"}      # the SINGLE-PHASE recipe: ONE wave (the row's DECLARED `wave: battery`, demo-import M3) + the two dose rungs; no u0, no s2


def test_the_configured_test_model_carries_the_phase_views_and_the_papers_pick():
    m = M.model("qwen35_4b_demo")
    # the THREE plan views on the configured-not-run demo: the one wave s1 carries every arm beside its own unsteered legs; its LoRA pick
    # is the paper's (the same Qwen3.5-4B checkpoints: rank 4 on o_proj, selected_by paper)
    assert M.views("qwen35_4b_demo") == ["battery", "d0p75", "d0p75_relu"] and set(M.views("qwen35_4b_demo")) == PHASE_VIEWS
    assert M.single_phase("qwen35_4b_demo") and M.baseline_view("qwen35_4b_demo") == "battery"
    # the wave is DECLARED on the row (demo-import M3, Q2): a name outside the phase-view grammar, read through the reader's ONE rule
    assert m["wave"] == "battery" and M.declared_wave("qwen35_4b_demo") == "battery" and not M.PHASE_VIEW_RE.match("battery")
    assert M.phase_views("qwen35_4b_demo") == ["battery", "d0p75", "d0p75_relu"] and M.is_phase_view("qwen35_4b_demo", "battery")
    assert M.is_phase_view("qwen35_4b_demo", "s1") and "s1" not in M.phase_views("qwen35_4b_demo")      # the grammar names s1 for every key; the row has no such view
    assert M.is_phase_view("qwen35_4b", "s2") and M.is_phase_view("glm46vflash", "d1p5") and not M.is_phase_view("qwen35_4b", "battery")
    assert not M.is_phase_view("qwen35_4b_demo", "nope") and not M.steered_wave_view("battery", "qwen35_4b_demo")
    assert set(m["views"]["battery"]["cells"]) == {"none", "prompted__safety_refusal", "prompted__si_constitution"} | OBJECTS
    assert m["lora_architecture"] == {"rank": 4, "modules": ["o_proj"], "per_expert_rank": None, "selected_by": "paper", "selection_ruling_id": None}
    assert set(m["objects"]) >= OBJECTS
    assert m["run_in_this_work"] is False
    # the worked example IMPORTS the paper's committed objects (objects/DEMO.json; demo-import): the row names the paper row whose fit-lane
    # objects it imports, the refusal direction names the served checkpoint's (a qwen35_4b_it object), and its paper label is the registry's
    assert m["committed_objects_of"] == "qwen35_4b" and m["objects"]["refusal_direction"]["committed_objects_of"] == "qwen35_4b_it"
    assert m["objects"]["refusal_direction"]["paper_label"] == "arditi_allpos"
    # difference_relu is the COMPOSED paired difference (kind composed, under the psr_paired_write kind directory) — the object every in-paper
    # difference_relu row is; the distilled psr_residual placeholder is gone (the README's first known gap, closed)
    relu = m["objects"]["difference_relu"]
    assert relu == {"lane": "derived", "cell_id": "constv2-mix-full__psr_ce_all_minus_ctl", "kind": "derived", "source": "derived_bank",
                    "derived_dir": "psr_paired_write", "object_kind": "composed"}
    assert m["objects"]["difference_lora"]["object_kind"] == "lora_subtraction" and m["objects"]["difference_lora"]["derived_dir"] == "lora_subtraction"


def test_the_papers_models_are_menu_models_with_the_rosters_views_and_no_phase_view():
    # qwen35_4b is the paper's model and the front door's worked example for the menu-only stages (00, 02, 03, 04): its rows are the
    # paper's (the roster views main / it — what the committed results were evaluated under; the paper's LoRA architecture) and it
    # declares NO phase view and no `objects` block, so it is not a front-door key (an end-to-end run takes a model with phase views)
    for key in ("qwen35_4b", "gemma4e4b", "nemotron3nano4b", "llama8b"):
        m = M.model(key)
        assert not (set(M.views(key)) & PHASE_VIEWS), key
        assert all(v.get("roster") == "configs/sdfbase_full22_roster.yaml" for v in m["views"].values()), key
        assert "objects" not in m, key
    assert set(M.views("qwen35_4b")) == {"main", "it"}
    assert M.front_door_keys() == ["qwen35_4b_demo"]


def test_no_model_runs_in_this_work_until_the_user_flips_it():
    assert {k: m["run_in_this_work"] for k, m in M.load_models().items()} == {k: False for k in M.load_models()}


def test_paper_models_carry_the_paper_architecture():
    for key in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"):
        la = M.model(key)["lora_architecture"]
        assert la == {"rank": 4, "modules": ["o_proj"], "per_expert_rank": None, "selected_by": "paper", "selection_ruling_id": None}, key


def test_eligible_layers_and_module_sets_on_a_hybrid_and_a_dense_model():
    # the hybrid (qwen35_4b, and the demo on the same checkpoints: 8 full-attention blocks among 32): attention modules on those blocks only,
    # MLP modules on every block; a dense model (llama8b): every block
    assert M.eligible_layers("qwen35_4b", "o_proj") == [3, 7, 11, 15, 19, 23, 27, 31]
    assert M.eligible_layers("qwen35_4b", "mlp") == list(range(32))
    lm = M.lora_modules("qwen35_4b", "attn_mlp")
    assert set(lm) == {"down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"}
    assert lm["q_proj"] == [3, 7, 11, 15, 19, 23, 27, 31] and lm["down_proj"] == list(range(32))
    assert M.eligible_layers("qwen35_4b_demo", "o_proj") == [3, 7, 11, 15, 19, 23, 27, 31]   # the demo IS the hybrid (the paper's Qwen3.5-4B checkpoints): attention modules on its 8 full-attention blocks
    assert M.lora_modules("qwen35_4b_demo", "wo") == {"o_proj": [3, 7, 11, 15, 19, 23, 27, 31]} and M.eligible_layers("qwen35_4b_demo", "mlp") == list(range(32))
    assert M.eligible_layers("llama8b", "o_proj") == list(range(32)) and M.lora_modules("llama8b", "wo") == {"o_proj": list(range(32))}   # a dense model: every block


def test_per_expert_rank_rule():
    # FD 6's rule for a mixture-of-experts fit: the shared expert at the full rank, routed experts at max(1, rank // active). No model
    # in the table has routed experts (every row's per_expert_rank is null); the rule stays as the reader's API for one that does.
    assert M.per_expert_rank(4, 8) == {"shared_expert": 4, "routed_expert": 1}
    assert M.per_expert_rank(64, 8) == {"shared_expert": 64, "routed_expert": 8}
    assert all(m["lora_architecture"]["per_expert_rank"] is None for m in M.load_models().values())


def test_cli_get_and_views():
    py = sys.executable
    out = subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "get", "qwen35_4b", "serve.checkpoint.hf_repo"], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "Qwen/Qwen3.5-4B"
    r = subprocess.run([py, str(REPO / "reproduce/lib/models.py"), "exists", "nope"], capture_output=True)
    assert r.returncode == 1
