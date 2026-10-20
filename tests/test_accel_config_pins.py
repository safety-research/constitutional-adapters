"""Shipped-config pins for the accel-fit lane.

The mirror of tests/test_class_c_arms.py, which pins the Class C lane's
block. This pins MINE. The gap it closes was found by near-miss: a merge
that drops one side of a config hunk leaves YAML that PARSES, so nothing
fails — the markers case failed loudly only because it made the file
unparseable. For this lane a silent drop is worse than loud breakage:

- lose ``data_parallel`` and every fit silently runs SINGLE-PROCESS
  (``cfg.get("data_parallel") or {}`` -> gpus 1), turning a ~4.5h wave
  back into a ~30h one with no error anywhere;
- lose ``batch_size_by_arm.diffmean`` and that arm silently reverts to
  train_budget's DEFAULT_BATCH_SIZE=2 fallback — the very fallback the
  measured pin replaced;
- lose ``class_b.batch_size_by_arm.caa_static_r`` and its capture width
  silently drops 512 -> the shared 8, ~64x more capture batches.

Per-lane pins rather than one shared config-shape test, deliberately:
a single omnibus test goes stale as the config grows and belongs to
nobody.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _cfg() -> dict:
    return yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())


def test_shipped_config_has_no_conflict_markers():
    """A conflicted merge staged with a broad `git add` can commit markers:
    git reports conflicts per FILE, and `add -A` cannot tell a resolution
    from an unresolved marker. This happened (2026-08-21) and made the
    config unparseable."""
    text = (REPO / "configs" / "two_si.yaml").read_text()
    for marker in ("<<<<<<<", ">>>>>>>", "\n=======\n"):
        assert marker not in text, (
            f"configs/two_si.yaml contains a merge marker {marker!r}")


def test_data_parallel_block_is_present_and_parallel():
    """Losing this block is SILENT: the fit driver falls back to
    single-process and the wave just takes ~7x longer."""
    dp = _cfg().get("data_parallel")
    assert isinstance(dp, dict), (
        "configs/two_si.yaml lost its data_parallel block — every fit would "
        "silently run single-process")
    gpus = dp.get("gpus")
    assert isinstance(gpus, int) and gpus > 1, (
        f"data_parallel.gpus must be an int > 1 to engage the parallel fit "
        f"driver, got {gpus!r}")


def test_data_parallel_gpus_is_the_pinned_world_size_across_sis():
    """The world size is PINNED across SIs on purpose: the SIs' only
    intended difference is the SI text, so a per-SI width would be a second
    difference. One number, asserted against the visible pool at run time."""
    import train_two_si_objects as tt

    cfg = _cfg()
    gpus = cfg["data_parallel"]["gpus"]
    for si in cfg["si_ids"]:
        for arm in tt.default_class_a_arms(cfg):
            w = tt.arm_width(cfg, si, arm, gpus)
            assert 1 <= w <= gpus, f"{si}/{arm} width {w} outside 1..{gpus}"


def test_measured_capture_widths_are_pinned():
    """Both measured capture widths, and the fallbacks they replaced."""
    from sisteer.eval.train_budget import DEFAULT_BATCH_SIZE, resolve_batch_size
    from fit_class_b import class_b_batch_size

    cfg = _cfg()
    dm = resolve_batch_size(cfg, "si_constitution", "diffmean")
    assert dm == 16, (
        f"diffmean's measured capture width is unpinned (got {dm}); "
        f"unpinned it falls back to DEFAULT_BATCH_SIZE={DEFAULT_BATCH_SIZE}")
    cb = dict(cfg["class_b"])
    cb.setdefault("batch_size", 8)
    # caa_static_r is back on the SHARED 8, and this pin now guards against
    # re-raising it casually. The WITHDRAWN 512 came from a ceiling probe
    # that truncated the prompt lists to benign[:512] — the HEAD of the
    # list — whereas the real fit runs all 14,000, pads each batch to its
    # LONGEST member, and keeps full [B,T,V] logits (this arm's vendored
    # harvest does its own raw forward and so never inherited the
    # single-logit capture fix). It OOMed at 73.7 GiB of an 80 GB card,
    # twice, including alone in a fresh process. Any future width here must
    # be measured against the WORST-CASE prompt in the real list, not a
    # convenient prefix of it.
    assert class_b_batch_size(cb, "caa_static_r") == 8
    assert class_b_batch_size(cb, "antidote") == 8
    assert class_b_batch_size(cb, "alphasteer") == 8
