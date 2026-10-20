"""The CE arms' shared held-out pool resolves through the volume root the way the fits do (stage E, 2026-09-30): the fits config spells
it `/vol/pool/<key>/pairs_constv2-heldout.jsonl`, a cluster node maps `/vol/` to $SISTEER_VOL_ROOT (train_two_si_objects.resolve_vol_path),
and tune_arm's trial path read the literal string — every ReLU trial refused with "missing on disk" while the file was present under the
mapped root. _heldout_pool_path now returns the resolved path; the pool fingerprint hashes the real file; without the env the path stands."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tune_arm import tune_arm  # noqa: E402

CFG = {"train_kwargs": {"default": {"heldout_pool": "/vol/pool/toy_base/pairs_constv2-heldout.jsonl"}}}


def test_heldout_pool_resolves_under_the_volume_root(tmp_path, monkeypatch):
    pool = tmp_path / "vol" / "pool" / "toy_base" / "pairs_constv2-heldout.jsonl"
    pool.parent.mkdir(parents=True); pool.write_text('{"prompt": "p", "pos": "x", "neg": ""}\n')
    monkeypatch.setenv("SISTEER_VOL_ROOT", str(tmp_path / "vol"))
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == str(pool)
    ad = tune_arm.make_adapter("psr_ce_all")
    assert ad.fingerprint_pool_extra(None, CFG) == f"|heldout_pool:{tune_arm._config_file_sha(str(pool))}"   # the real file's sha
    assert ad.fingerprint_pool_extra(None, CFG) != "|heldout_pool:MISSING"


def test_without_a_volume_root_the_path_stands_and_a_declared_missing_pool_is_missing(monkeypatch):
    monkeypatch.delenv("SISTEER_VOL_ROOT", raising=False)
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == "/vol/pool/toy_base/pairs_constv2-heldout.jsonl"
    assert tune_arm.CESoloAdapter._heldout_pool_path({"train_kwargs": {"default": {}}}) is None
    ad = tune_arm.make_adapter("psr_ce_all")
    assert ad.fingerprint_pool_extra(None, CFG) == "|heldout_pool:MISSING"


def test_the_shared_root_is_the_fallback_volume_root_and_vol_root_wins(tmp_path, monkeypatch):
    """The front door's job env carries SISTEER_SHARED_ROOT (every submission); a node without SISTEER_VOL_ROOT resolves /vol/ under
    <shared root>/vol; SISTEER_VOL_ROOT, when set, wins. The recon's explicit cluster path (no /vol/ prefix) is never touched."""
    monkeypatch.delenv("SISTEER_VOL_ROOT", raising=False)
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == str(tmp_path / "shared" / "vol" / "pool" / "toy_base" / "pairs_constv2-heldout.jsonl")
    monkeypatch.setenv("SISTEER_VOL_ROOT", str(tmp_path / "vol"))
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == str(tmp_path / "vol" / "pool" / "toy_base" / "pairs_constv2-heldout.jsonl")
    drv = tune_arm._load_driver()
    assert drv.resolve_vol_path("/workspace/x/vol/pool/k/pairs_constv2ctl2-full.jsonl") == "/workspace/x/vol/pool/k/pairs_constv2ctl2-full.jsonl"


def test_the_refusal_names_the_config_spelling_the_resolved_path_and_how_vol_resolves_here(monkeypatch):
    monkeypatch.delenv("SISTEER_VOL_ROOT", raising=False); monkeypatch.delenv("SISTEER_SHARED_ROOT", raising=False)
    msg = tune_arm.CESoloAdapter._heldout_missing_message(CFG, tune_arm.CESoloAdapter._heldout_pool_path(CFG))
    assert "train_kwargs.default.heldout_pool = '/vol/pool/toy_base/pairs_constv2-heldout.jsonl'" in msg
    assert "SISTEER_VOL_ROOT=unset" in msg and "SISTEER_SHARED_ROOT=unset" in msg and "/vol " in msg and "nothing to rank on" in msg
    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/workspace/someone")
    msg = tune_arm.CESoloAdapter._heldout_missing_message(CFG, tune_arm.CESoloAdapter._heldout_pool_path(CFG))
    assert "at '/workspace/someone/vol/pool/toy_base/pairs_constv2-heldout.jsonl'" in msg and "SISTEER_SHARED_ROOT=/workspace/someone" in msg


def test_a_mounted_vol_wins_over_the_shared_root_and_the_explicit_override_wins_over_both(tmp_path, monkeypatch):
    """A host with the paper's /vol mount AND an inherited front-door shared root must read /vol/… as spelled — never a different pool
    under $SHARED/vol; the explicit SISTEER_VOL_ROOT still wins over both. The source is recorded for the summaries."""
    drv = tune_arm._load_driver()
    monkeypatch.delenv("SISTEER_VOL_ROOT", raising=False)
    monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    monkeypatch.setattr(drv, "_vol_mounted", lambda: True)                                        # both roots exist
    assert drv.vol_root_with_source() == (None, "/vol (mounted)")
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == "/vol/pool/toy_base/pairs_constv2-heldout.jsonl"
    assert "root source /vol (mounted)" in drv.vol_path_state()
    monkeypatch.setenv("SISTEER_VOL_ROOT", str(tmp_path / "explicit"))                            # the explicit override wins over both
    assert drv.vol_root_with_source() == (str(tmp_path / "explicit"), "SISTEER_VOL_ROOT")
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == str(tmp_path / "explicit" / "pool" / "toy_base" / "pairs_constv2-heldout.jsonl")
    monkeypatch.delenv("SISTEER_VOL_ROOT"); monkeypatch.setattr(drv, "_vol_mounted", lambda: False)
    assert drv.vol_root_with_source() == (str(tmp_path / "shared" / "vol"), "SISTEER_SHARED_ROOT/vol")   # /vol absent: the shared root
    monkeypatch.delenv("SISTEER_SHARED_ROOT")
    assert drv.vol_root_with_source() == (None, "none")


def test_a_symlinked_pool_under_either_root_is_read_through_the_link(tmp_path, monkeypatch):
    real = tmp_path / "real" / "pairs_constv2-heldout.jsonl"; real.parent.mkdir(); real.write_text('{"prompt": "p", "pos": "x", "neg": ""}\n')
    drv = tune_arm._load_driver(); monkeypatch.setattr(drv, "_vol_mounted", lambda: False)
    for env, root in (("SISTEER_VOL_ROOT", tmp_path / "vol"), ("SISTEER_SHARED_ROOT", tmp_path / "shared")):
        monkeypatch.delenv("SISTEER_VOL_ROOT", raising=False); monkeypatch.delenv("SISTEER_SHARED_ROOT", raising=False)
        monkeypatch.setenv(env, str(root))
        pool_dir = (root if env == "SISTEER_VOL_ROOT" else root / "vol") / "pool" / "toy_base"; pool_dir.mkdir(parents=True, exist_ok=True)
        link = pool_dir / "pairs_constv2-heldout.jsonl"; link.symlink_to(real)
        p = tune_arm.CESoloAdapter._heldout_pool_path(CFG)
        assert p == str(link) and Path(p).is_file(), (env, p)
        assert tune_arm.make_adapter("psr_ce_all").fingerprint_pool_extra(None, CFG) == f"|heldout_pool:{tune_arm._config_file_sha(str(real))}"


def test_a_plain_vol_directory_is_not_a_mount_and_defers_to_the_shared_root(tmp_path, monkeypatch):
    """A pre-created but unattached /vol mountpoint must not win over a valid pool under $SISTEER_SHARED_ROOT/vol."""
    drv = tune_arm._load_driver()
    plain = tmp_path / "vol_plain"; plain.mkdir()
    monkeypatch.setattr(drv, "VOL_MOUNT", str(plain)); monkeypatch.setattr(drv, "_mountinfo_text", lambda: "")
    monkeypatch.delenv("SISTEER_VOL_ROOT", raising=False); monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    assert drv._vol_mounted() is False
    assert drv.vol_root_with_source() == (str(tmp_path / "shared" / "vol"), "SISTEER_SHARED_ROOT/vol")
    assert "a plain directory (no mount attached)" in drv.vol_path_state()
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == str(tmp_path / "shared" / "vol" / "pool" / "toy_base" / "pairs_constv2-heldout.jsonl")


def test_a_real_mount_in_the_mount_table_wins_even_when_symlinked_and_the_override_still_wins(tmp_path, monkeypatch):
    drv = tune_arm._load_driver()
    real = tmp_path / "vol_real"; real.mkdir(); link = tmp_path / "vol_link"; link.symlink_to(real)
    monkeypatch.setattr(drv, "VOL_MOUNT", str(link))                                                    # a symlinked mountpoint
    monkeypatch.delenv("SISTEER_VOL_ROOT", raising=False); monkeypatch.setenv("SISTEER_SHARED_ROOT", str(tmp_path / "shared"))
    # the mount table names the RESOLVED path as a mount point (a bind mount: invisible to os.path.ismount on the same filesystem)
    monkeypatch.setattr(drv, "_mountinfo_text", lambda: f"36 25 0:32 / / rw shared:7 - ext4 /dev/root rw\n"
                                                         f"98 36 0:45 /pool {str(real.resolve()).replace(' ', chr(92) + '040')} rw,relatime shared:50 - nfs4 vast:/vol rw\n")
    assert drv._vol_mounted() is True
    assert drv.vol_root_with_source() == (None, f"{link} (mounted)")
    assert tune_arm.CESoloAdapter._heldout_pool_path(CFG) == "/vol/pool/toy_base/pairs_constv2-heldout.jsonl"   # read as spelled
    assert "mounted" in drv.vol_path_state() and "root source" in drv.vol_path_state()
    monkeypatch.setenv("SISTEER_VOL_ROOT", str(tmp_path / "explicit"))                                   # the override still wins
    assert drv.vol_root_with_source() == (str(tmp_path / "explicit"), "SISTEER_VOL_ROOT")
    monkeypatch.delenv("SISTEER_VOL_ROOT")
    monkeypatch.setattr(drv, "_mountinfo_text", lambda: "36 25 0:32 / / rw shared:7 - ext4 /dev/root rw\n")   # the table no longer names it: a plain dir
    assert drv._vol_mounted() is False and drv.vol_root_with_source()[1] == "SISTEER_SHARED_ROOT/vol"
    assert drv._mountinfo_unescape("/mnt/with\\040space") == "/mnt/with space"
