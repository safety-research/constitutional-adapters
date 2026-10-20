"""The sampler's family list must be explicit, and must match the config.

`pool_sample.py --families` carried a hardcoded default,
"bon,pasttense,pap,encoding,aim,gcg_replay", frozen from an older campaign.
The config's pool.static_families is now a different set. Sampling with the
default would have built a pool containing three families we do not train on
and omitting three we do -- and nothing downstream compares the two lists,
because the pool builder reads the CONFIG. The mismatch would have surfaced
only as a PoolCompositionError about zero realised pairs, after the GPU hours.
"""
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS_FAMILIES = {"wildjailbreak_harmful"}   # enter via their own --n-* flag


def _cfg():
    return yaml.safe_load((ROOT / "configs" / "two_si.yaml").read_text())


def test_families_is_required_not_defaulted():
    """No default at all: a stale one is worse than a missing one."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pool_sample.py"),
         "sample", "--out-dir", "/tmp/does-not-matter"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "--families" in r.stderr


def test_the_generated_families_are_all_materialisable():
    """Every config family the sampler must cross must RESOLVE through
    sources._static_variants_fn -- the real gate, which special-cases bon
    (banked pool, no bon_variants) and would KeyError/AttributeError on a
    family that only exists in the config. Asserting the *_variants naming
    convention directly would have failed on bon for the wrong reason."""
    from sisteer.sampling.sources import _static_variants_fn, SUBENV_FAMILIES
    pool = _cfg()["pool"]
    for fam in sorted(set(pool["static_families"]) - CORPUS_FAMILIES):
        if fam in SUBENV_FAMILIES:
            continue   # needs a pre-generated bank, checked at run time
        assert callable(_static_variants_fn(fam)), fam


def test_corpus_families_cannot_be_crossed_as_attack_families():
    """wildjailbreak_harmful is a standalone already-adversarial corpus. It
    enters via --n-wildjailbreak-harmful, not --families, and asking the
    sampler to cross it must fail loudly rather than yield zero prompts."""
    import pytest
    from sisteer.sampling.sources import _static_variants_fn
    for fam in CORPUS_FAMILIES:
        with pytest.raises((ModuleNotFoundError, AttributeError)):
            _static_variants_fn(fam)


def test_train_families_is_a_subset_of_static_families():
    pool = _cfg()["pool"]
    assert set(pool["train_families"]) <= set(pool["static_families"])


# --- multi-pod shard naming ------------------------------------------------
# `sample` names its DATA file from the GLOBAL shard index
# (shard_{shard_index:02d}.jsonl, pool_sample.py:178), but sample-all named the
# per-GPU LOG from the LOCAL loop index. Two pods sharing an out-dir -- the
# arrangement cmd_sample_all's own docstring describes for a 32-GPU run --
# therefore both wrote shard_00.log..shard_0N.log, and whichever finished
# second destroyed the other's logs. Data was never at risk; the diagnostic
# was, which is worse than it sounds when exactly one pod fails.

def test_sample_all_logs_use_the_global_shard_index():
    src = (ROOT / "scripts" / "pool_sample.py").read_text()
    assert 'f"shard_{offset + i:02d}.log"' in src, (
        "sample-all must name per-GPU logs by GLOBAL shard index, or a "
        "multi-pod run silently overwrites one pod's logs with another's")
    assert 'f"shard_{i:02d}.log"' not in src


def test_log_and_data_indices_agree():
    """Both must key off the same global index, so shard_05.log is the log for
    shard_05.jsonl rather than for some other pod's fifth local GPU."""
    src = (ROOT / "scripts" / "pool_sample.py").read_text()
    assert 'f"shard_{args.shard_index:02d}.jsonl"' in src
    assert 'f"shard_{offset + i:02d}.log"' in src


# --- gpu-util item 2: workers must not rebuild the inventory ---------------

def _ps():
    """Import pool_sample as a module -- reading the real _PASSTHROUGH tuple
    beats splitting source text, which broke on a parenthesis in a comment."""
    import importlib.util as ilu, sys
    if "_ps_inv" in sys.modules:
        return sys.modules["_ps_inv"]
    spec = ilu.spec_from_file_location(
        "_ps_inv", ROOT / "scripts" / "pool_sample.py")
    m = ilu.module_from_spec(spec)
    sys.modules["_ps_inv"] = m
    spec.loader.exec_module(m)
    return m


def test_inventory_is_passed_through_to_workers():
    assert "inventory" in _ps()._PASSTHROUGH


def test_save_inventory_is_not_passed_through():
    """Only `plan` writes the cache; a worker rewriting it would race its
    siblings mid-run."""
    m = _ps()
    assert "save_inventory" not in m._PASSTHROUGH
    assert "save_inventory" not in getattr(m, "_PASSTHROUGH_NUM", ())


def test_stale_inventory_cache_is_refused_not_used():
    """A cache built for a different family set must REFUSE. Silently
    sampling the wrong prompts is the failure this campaign keeps hitting."""
    import importlib.util as ilu, json, sys, types
    spec = ilu.spec_from_file_location(
        "_ps_inv", ROOT / "scripts" / "pool_sample.py")
    m = sys.modules.get("_ps_inv")
    if m is None:
        m = ilu.module_from_spec(spec); sys.modules["_ps_inv"] = m
        spec.loader.exec_module(m)
    import tempfile, pathlib as _pl
    args = types.SimpleNamespace(
        families="bon", benign_families="phtest", n_wildjailbreak_harmful=1,
        n_benign=None, harmbench_splits="x", seed=0, bon_variants=8,
        subenv_bank=None, no_control_subtraction=False,
        wildjailbreak_path=None)
    key = m._inventory_key(args)
    with tempfile.TemporaryDirectory() as d:
        f = _pl.Path(d) / "inv.json"
        stale = dict(key, families=["pasttense"])
        f.write_text(json.dumps({"key": stale, "prompts": []}))
        args.inventory = str(f)
        import pytest as _pt
        with _pt.raises(SystemExit, match="different"):
            m._inventory_cached(args)
