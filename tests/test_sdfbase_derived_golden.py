"""BYTE-IDENTITY of the derived app's 100k path: six derived objects were
published from ``sdfbase_derived.py`` at a9bf57f under the 100k pair
(constv2-mix-100k − constv2ctl-100k). The app then learned the full − full
pair (constv2-mix-full − constv2ctl2-full, capture budget, lora-subtract
mode, multi-probe scoring, --ce-skip-existing, --app-name). Every argv the
100k path hands a container and every volume path it writes must be
EXACTLY what it was — a change that moved those outputs would corrupt the
published record even though the 100k rows are no longer cited.

The proof is a replay, not an assertion: tests/derived_golden_replay.py
runs a fixed list of ~140 builder calls, 12 in-container ``_score_impl`` /
``_liveness_impl`` traces and 9 entrypoint spawn lists; the generator
(tests/tools/gen_sdfbase_derived_golden.py) ran it on the a9bf57f module
and froze the answer in tests/fixtures/sdfbase_derived_100k_golden.json.
This test runs the same replay on the CURRENT module and asserts equality,
naming the first differing entry. The fixture records the source commit
and the replayed module's sha256 so a regeneration from a later tree is
visible (it would make the test tautological)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from test_watch_successor import make_fake_modal  # the shared `modal` stub

REPO = Path(__file__).resolve().parents[1]
SRC_PATH = REPO / "scripts" / "modal" / "sdfbase_derived.py"
GOLDEN = REPO / "tests" / "fixtures" / "sdfbase_derived_100k_golden.json"
#: The commit the six published 100k derived objects' app was built from.
GOLDEN_SOURCE_SHA = "a9bf57f43c6534e99ced637a3c998e0a0ee35365"
for _p in (REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture(scope="module")
def sd():
    import importlib.util
    # ALWAYS this suite's own stub (test_sdfbase_derived_app.py's idiom): in the
    # full suite an earlier module has usually left ITS fake `modal` in
    # sys.modules -- several lineages `setdefault` a bare SimpleNamespace whose
    # Image has no debian_slim and never restore it -- and keeping it made every
    # test here ERROR at setup in the reseed merge tree's Modal CPU suite
    # (2026-09-15) while the file passed alone. Whatever was there is restored.
    prev_modal = sys.modules.get("modal")
    sys.modules["modal"] = make_fake_modal()
    sys.modules.pop("sdfbase_fits", None)
    # the W&B secret's name is a site setting read at import: any name, so the app mounts one (the replay records its place)
    prev_secret = os.environ.get("SISTEER_MODAL_WANDB_SECRET")
    os.environ["SISTEER_MODAL_WANDB_SECRET"] = "example-wandb-secret"
    spec = importlib.util.spec_from_file_location("_sdfbase_derived_golden_cur",
                                                  SRC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    try:
        yield mod
    finally:
        if prev_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev_modal
        if prev_secret is None:
            os.environ.pop("SISTEER_MODAL_WANDB_SECRET", None)
        else:
            os.environ["SISTEER_MODAL_WANDB_SECRET"] = prev_secret


def _first_diff(a, b, path="golden"):
    """The first differing leaf between two JSON-able values, as a
    readable path — so a failure names the entry, not two 250 KB blobs."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: only in current"
            if k not in b:
                return f"{path}.{k}: only in golden"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} (golden) != {len(b)} (current)"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: golden {a!r} != current {b!r}"


def test_golden_fixture_names_the_publish_commit():
    doc = json.loads(GOLDEN.read_text())
    assert doc["source"]["git_sha"] == GOLDEN_SOURCE_SHA
    assert doc["source"]["module"] == "scripts/modal/sdfbase_derived.py"
    assert len(doc["source"]["module_sha256"]) == 64
    g = doc["golden"]
    # the frozen answer is the 100k pair's: its pools, parents, no ctl2 anywhere
    assert set(g["constants"]["CE_POOLS"]) == {"constv2-heldout", "control-heldout",
                                               "constv2ctl-100k"}
    assert g["constants"]["PARENT_CELLS"] == [
        "constv2-mix-100k__psr_ce_all", "constv2ctl-100k__psr_ce_all",
        "constv2-mix-100k__lora_r4_ce", "constv2ctl-100k__lora_r4_ce"]
    text = GOLDEN.read_text()
    assert "constv2ctl2" not in text and "capture-budget" not in text


def test_100k_path_is_byte_identical_to_the_a9bf57f_module(sd, tmp_path):
    from derived_golden_replay import collect

    golden = json.loads(GOLDEN.read_text())["golden"]
    current = collect(sd, tmp_path / "replay")
    diff = _first_diff(golden, current)
    assert diff is None, diff
    # the replay is not vacuous: the staged trace carries the historical
    # probe -> CE -> parents -> bare sequence and the pool/parent spellings
    st = current["score_traces"]["qwen35_4b_base"]["staged_auto_x12_parents"]
    tags = [t.get("tag") for t in st["trace"] if "tag" in t]
    assert tags[0] == "probe/constv2-mix-100k__psr_ce_all_minus_ctl/x1"
    assert "ce/constv2-mix-100k__psr_ce_all_minus_ctl/constv2ctl-100k/x2" in tags
    assert tags[-3:] == ["ce/bare/constv2-heldout", "ce/bare/control-heldout",
                         "ce/bare/constv2ctl-100k"]
    assert sum(len(v) for v in current["commands"].values()) >= 140


def test_replay_helper_flags_a_moved_output(sd, tmp_path):
    """The proof would be worthless if the replay could not see a change:
    a one-character move of a default pool path shows up as a named diff."""
    from derived_golden_replay import collect

    golden = json.loads(GOLDEN.read_text())["golden"]
    mp = pytest.MonkeyPatch()
    try:
        moved = dict(sd.CE_POOLS)
        moved["control-heldout"] = moved["control-heldout"].replace(
            "constv2ctl-heldout", "constv2ctl-heldout2")
        mp.setattr(sd, "CE_POOLS", moved)
        current = collect(sd, tmp_path / "replay2")
    finally:
        mp.undo()
    diff = _first_diff(golden, current)
    assert diff is not None and "control-heldout" in diff, diff
