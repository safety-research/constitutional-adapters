"""A standup admits every model key the geometry tables know — the paper's eight AND the front door's onboarded models
(paper-companion-repo stage E, 2026-09-30).

`geometry.py` keeps the onboarded keys (`ONBOARDED_MODEL_KEYS`: Qwen3.5-4B (worked example)) OUT of `MODEL_HF_IDS` by ruling
(test_models_demo_front_door pins the tables) and serves both kinds through `rows_for` / `budget_for` / `model_key_for`. The two
sdfbase standups once checked `--model-key` against `MODEL_HF_IDS` alone, so an onboarded model's first unsteered unit died
at standup on that check (a synthetic rendering: `[mt-standup] refused (exit 2): --model-key 'qwen35_4b_demo' is not a canonical
model key ([... the paper's eight])` — job 11111 on TESTHOST, after registering its runs). Both standups now read ONE merged
table (`served_hf_id_by_key()`); a stray key is still refused by name, listing every known key.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
STANDUPS = ("scripts/sdfbase/mt_serving_up.py", "scripts/sdfbase/lora_serving_up.py")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rel", STANDUPS)
def test_the_standup_key_table_is_the_paper_keys_plus_the_onboarded_ones(rel):
    mod = _load("_standup_keys_" + Path(rel).stem, rel)
    table = mod.served_hf_id_by_key()
    assert table == {**sg.MODEL_HF_IDS, **sg.ONBOARDED_MODEL_HF_IDS}
    assert set(table) == set(sg.KNOWN_MODEL_KEYS) and set(table) > set(sg.MODEL_HF_IDS)
    # the onboarded model, exactly as models.yaml serves it
    assert table["qwen35_4b_demo"] == sg.ONBOARDED_MODEL_HF_IDS["qwen35_4b_demo"] == "Qwen/Qwen3.5-4B"
    # a paper key keeps its pinned id; a stray key has no row
    assert table["qwen35_4b"] == sg.MODEL_HF_IDS["qwen35_4b"] and table.get("gemma4-e4b") is None


@pytest.mark.parametrize("rel", STANDUPS)
def test_every_model_key_check_in_the_standup_reads_the_merged_table(rel):
    """No check site consults MODEL_HF_IDS on its own again: each `is not a canonical model key` refusal is preceded by the
    merged lookup, and the refusal lists the table it looked in (every known key), never the paper's eight alone."""
    src = (REPO / rel).read_text()
    assert re.search(r"\bMODEL_HF_IDS\.get\(", src) is None, rel
    assert re.search(r"import MODEL_HF_IDS(?!, ONBOARDED_MODEL_HF_IDS)\b", src) is None, rel
    # the refusal SITES (each lists the table it looked in: `sorted(<table>)`), not the helper's docstring that quotes the message
    refusals = [m.start() for m in re.finditer(r"is not a canonical model key", src) if "sorted(" in src[m.start():m.start() + 200]]
    assert len(refusals) == (2 if "mt_serving_up" in rel else 1), (rel, refusals)
    for pos in refusals:
        # the nearest preceding merged-table assignment is the table this site looked in and lists in its refusal (the native
        # path assigns at its import block, well above its check; the direct-lookup absence is pinned above)
        assigns = list(re.finditer(r"(\w+) = served_hf_id_by_key\(\)", src[:pos]))
        assert assigns, (rel, pos)
        var = assigns[-1].group(1)
        assert f"sorted({var})" in src[pos:pos + 200], (rel, var)


def test_the_lora_native_standup_still_refuses_a_stray_key_by_name():
    lsu = _load("_lora_serving_up_stray_key", STANDUPS[1])
    with pytest.raises(lsu.StandupRefused, match="not a canonical model key|refusing --model-key"):
        lsu._resolve_geometry("families_served:short", None, None, model_key="gemma4-e4b",
                              allow_over_budget=False, override_reason=None)
