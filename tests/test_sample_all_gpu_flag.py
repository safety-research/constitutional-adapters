"""--gpus is a device-ID LIST and must refuse a value that is really a count.

Measured cost of its absence, both on 2026-08-26:
  --gpus 8  on an 8-GPU node  -> device "8"; ids are 0-7, names nothing
  --gpus 4  in a 4-GPU alloc  -> device "4" alone. Job 212789 RAN, printed
                                 "global shards [0, 1) of 8", and was on
                                 course to generate ONE EIGHTH of the
                                 inventory while exiting 0.
The shard-space overflow check cannot catch either: offset + 1 GPU always fits
inside the global shard space.
"""
import importlib.util as ilu
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = ilu.spec_from_file_location(
    "_pool_sample", ROOT / "scripts" / "pool_sample.py")


def _mod():
    if "_pool_sample" in sys.modules:
        return sys.modules["_pool_sample"]
    m = ilu.module_from_spec(_SPEC)
    sys.modules["_pool_sample"] = m
    _SPEC.loader.exec_module(m)
    return m


def test_count_typed_as_list_is_refused():
    """The exact 212789 mistake: 4 visible devices, --gpus 4."""
    with pytest.raises(SystemExit, match="NOT A GPU COUNT"):
        _mod().resolve_gpus("4", ["0", "1", "2", "3"])


def test_device_id_beyond_the_node_is_refused():
    """The original mistake: --gpus 8 on an 8-GPU node."""
    with pytest.raises(SystemExit, match="not visible"):
        _mod().resolve_gpus("8", [str(i) for i in range(8)])


def test_omitting_gpus_takes_every_visible_device():
    assert _mod().resolve_gpus(None, ["0", "1", "2", "3"]) == \
        ["0", "1", "2", "3"]


def test_a_genuine_subset_is_still_allowed():
    """The refusal must not break 'use exactly these two devices'."""
    assert _mod().resolve_gpus("1,2", ["0", "1", "2", "3"]) == ["1", "2"]


def test_no_visible_devices_is_its_own_error():
    with pytest.raises(SystemExit, match="no GPUs found"):
        _mod().resolve_gpus(None, [])


def test_visibility_unknown_does_not_block_an_explicit_list():
    """Off-Slurm with no CUDA_VISIBLE_DEVICES and no nvidia-smi, an explicit
    list is taken at face value rather than refused for lack of evidence."""
    assert _mod().resolve_gpus("0,1", []) == ["0", "1"]


# --- every fan-out site, not just the first one ----------------------------
# cmd_generate_anti_neg_all is "sample-all reused shape-for-shape" and had
# reproduced the unguarded --gpus parsing verbatim. Fixing pool_sample alone
# would have left it to hit the identical 1/N bug on the next GPU pass.

def test_no_fanout_site_parses_gpus_itself():
    """A source-grep test, deliberately: it catches a NEW unguarded site
    before it ships, which a runtime test cannot."""
    bad = []
    for rel in ("scripts/pool_sample.py", "scripts/build_two_si_pool.py"):
        src = (ROOT / rel).read_text()
        if 'args.gpus.split(",")' in src:
            bad.append(rel)
    assert not bad, (
        f"{bad} parse --gpus inline instead of calling "
        f"pool_sample.resolve_gpus, so a GPU COUNT typed into the device-ID "
        f"list flag would silently run a fraction of the work")


def test_anti_neg_fanout_uses_the_shared_resolver():
    src = (ROOT / "scripts" / "build_two_si_pool.py").read_text()
    assert "pool_sample.resolve_gpus(args.gpus" in src
