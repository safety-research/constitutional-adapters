"""battery_record.Layout.localize: a producer's recorded cluster path -> the same file under the local copy.

The Modal port runs the sampled lane under SUFFIXED wave roots (``sdfbase_full22_native``, ``sdfbase_full22_b2i_lens``:
its separate records) while the local copy always lays the lane out under ``sdfbase_full22/``. Matching the
bare segment alone left every native rung's standup record unread ("manifest names no routing_table") and the record's
same-instrument table blank for all 18 native views (2026-09-16)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "strength" / "battery_record.py"


def _load():
    spec = importlib.util.spec_from_file_location("battery_record_layout_localize", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def br():
    return _load()


ROOT = Path("/local/copy")


@pytest.mark.parametrize("remote, expected", [
    # the bare wave root, as the clusters wrote it
    ("/shared/u/exp/sdfbase_full22/sampled/serving/controls_t0_k_1/controls_512/routing_table.json",
     ROOT / "sdfbase_full22/sampled/serving/controls_t0_k_1/controls_512/routing_table.json"),
    # the Modal port's native wave root: the suffixed segment maps onto the ONE local directory
    ("/workspace/strength/exp/sdfbase_full22_native/sampled/serving/controls_t0_gemma4e4b_d1_modal-01/controls_512/routing_table.json",
     ROOT / "sdfbase_full22/sampled/serving/controls_t0_gemma4e4b_d1_modal-01/controls_512/routing_table.json"),
    # the lens record's wave root
    ("/workspace/strength/exp/sdfbase_full22_b2i_lens/sampled/ladders/gemma4e4b_d0p1_t0/ladder.json",
     ROOT / "sdfbase_full22/sampled/ladders/gemma4e4b_d0p1_t0/ladder.json"),
    # objects stay objects
    ("/shared/u/exp/objects/sdfbase/full22/qwen35_4b/cells/abc/manifest.json",
     ROOT / "objects/sdfbase/full22/qwen35_4b/cells/abc/manifest.json"),
])
def test_localize_maps_bare_and_suffixed_wave_roots_onto_the_local_layout(br, remote, expected):
    assert br.Layout(ROOT).localize(remote) == expected


@pytest.mark.parametrize("remote", [None, "", "/workspace/strength/exp/other_lane/sampled/x.json",
                                    "/workspace/strength/exp/not_sdfbase_full22/x.json"])
def test_localize_returns_none_for_paths_outside_the_lane(br, remote):
    assert br.Layout(ROOT).localize(remote) is None


def test_localize_requires_the_segment_to_be_a_whole_path_component(br):
    # `xsdfbase_full22/` is not the lane's root; the match is anchored at a path separator (or the string start)
    assert br.Layout(ROOT).localize("/exp/xsdfbase_full22/sampled/a.json") is None
    assert br.Layout(ROOT).localize("sdfbase_full22_native/sampled/a.json") == ROOT / "sdfbase_full22/sampled/a.json"
