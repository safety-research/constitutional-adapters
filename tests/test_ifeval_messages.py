"""The canonical IFEval battery — the IFEval prompt renderer ``ifeval_messages``.

Pinned (offline, on the cached stock dataset — skipped when it is absent): 541 message lists, each ONE user turn carrying
the vendor's prompt verbatim, ids = the vendor's row keys as strings (the seed rule's item keys), in dataset order; the
prompted class folds the SI into ONE leading system turn; the render task is the stock task minus its scorer (the same
dataset path / revision the installed inspect_evals module spells); the production renderer
``inspect_served.ifeval_messages`` (which ``ifeval_messages`` delegates to) agrees with the checker-free construction
message for message; a moved dataset pin refuses by name.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from sisteer.capeval.tiers import ifeval_messages as IFM
from sisteer.capeval.tinycap import ifeval_contracts as IC

DATASET_CACHED = (Path.home() / ".cache/huggingface/hub/datasets--google--IFEval/snapshots" / IC.IFEVAL_DATASET_REVISION).is_dir()
pytestmark = pytest.mark.skipif(not DATASET_CACHED, reason="the pinned IFEval dataset is not in the local HF cache (offline)")


@pytest.fixture(scope="module")
def offline():
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    for k in saved:
        os.environ[k] = "1"
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def rendered(offline):
    return IFM.ifeval_messages(), IFM.ifeval_messages(system="SI TEXT")


def test_541_single_user_turns_with_the_vendor_keys_as_ids_and_the_si_folded_into_one_system_turn(rendered, offline):
    bare, prompted = rendered
    assert len(bare) == len(prompted) == IC.IFEVAL_N == 541
    assert all([m["role"] for m in ml] == ["user"] for ml in bare)
    assert all([m["role"] for m in ml] == ["system", "user"] and ml[0]["content"] == "SI TEXT" for ml in prompted)
    assert [ml[-1] for ml in prompted] == [ml[0] for ml in bare]
    ids = IFM.ifeval_ids()
    assert len(ids) == 541 == len(set(ids)) and all(isinstance(i, str) and i.isdigit() for i in ids) and ids[0] == "1000"
    task = IFM.ifeval_render_task()
    by_id = {str(s.id): s for s in task.dataset}
    assert all(by_id[i].input == ml[0]["content"] for i, ml in zip(ids, bare))       # the prompt verbatim
    assert all(isinstance(by_id[i].metadata["instruction_id_list"], list) for i in ids)
    assert task.scorer is None
    mod = IFM.stock_module()
    assert mod.DATASET_PATH == IC.IFEVAL_DATASET_PATH and mod.IFEVAL_DATASET_REVISION == IC.IFEVAL_DATASET_REVISION
    # the prompt lengths bound the geometry: every stock prompt is under 2,000 characters
    assert max(len(ml[0]["content"]) for ml in bare) < 2000


def test_the_seed_rule_keys_on_these_ids_and_every_draw_is_distinct(rendered, offline):
    ids = IFM.ifeval_ids()
    seeds = {(i, t): IC.ifeval_sample_seed(i, t) for i in ids for t in range(IC.IFEVAL_SAMPLES_PER_ITEM)}
    assert len(set(seeds.values())) == 1082 and seeds[("1000", 0)] == 3028447421


def test_the_production_renderer_agrees_with_the_checker_free_construction_and_a_moved_pin_refuses(monkeypatch, offline):
    # the production renderer (the stock task through inspect_served's plan branch) and the checker-free construction
    # render the same 541 message lists — the cross-check that survives the merge
    assert IFM.ifeval_messages() == IFM._local_ifeval_messages()
    mod = IFM.stock_module()
    monkeypatch.setattr(mod, "IFEVAL_DATASET_REVISION", "0" * 40)
    with pytest.raises(IFM.IfevalRenderError, match="dataset moved"):
        IFM.ifeval_render_task()
    assert "system turn folded" in IFM.messages_source(prompted=True) and "folded" not in IFM.messages_source(prompted=False)
