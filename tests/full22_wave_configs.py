"""The full22 WAVE configs, enumerated ONCE for every test that walks them.

``configs/sdfbase_full22_*.yaml`` matches more than the wave configs. Three TABLES share the prefix
-- the production roster, the certification-only roster and the arms table -- and so do the sampled
re-run's generated configs. A test that globs the prefix and indexes ``cfg["attacks"]`` sweeps the
tables up and dies on a ``KeyError`` naming the key, not the file: the arms table reddened two such tests (test_sampled_controls, test_bon_climb_default) by its NAME alone,
and a third (test_review_p2_6_xstest_safe) stayed green only because it asserts a key's ABSENCE,
which a table satisfies trivially.

The tables are excluded BY VALUE, each with its reason, never by shape. A shape filter ("keep what
has ``attacks``") would silently drop a broken wave config; with the tables named, a wave config that
lacks a key still fails loudly in the test that reads it, by file name. The arms table's name is
load-bearing (``arm_cell_check.ARMS``, ``make_arm_configs.ARMS``, ``arditi_strength.ARMS_TABLE_REL``,
``ar_common.sh`` ``AR_ARMS``, ``f22_arm_bank_view``'s literal) and is not renamed to dodge a glob, nor
widened with wave keys to satisfy one. Pinned by tests/test_full22_wave_configs.py.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: the prefix every full22 file shares, wave config or not
CONFIGS_GLOB = "sdfbase_full22_*.yaml"

#: the sampled re-run's GENERATED configs (scripts/sdfbase/full22/sampled/make_sampled_configs.py, lane C):
#: a wave config as data plus a ``sampled_decode`` block, pinned as exactly that by
#: tests/test_sdfbase_full22_sampled.py -- never walked as a wave config
SAMPLED_MARK = "_sampled_"

#: the files that share the prefix but are NOT wave configs: name -> why (each a table the wave is rendered
#: from or served beside, none of them carrying a wave's ``attacks`` / ``bon`` blocks)
EXCLUDED_TABLES: dict[str, str] = {
    "sdfbase_full22_roster.yaml":
        "the production roster: the (target, view) -> cells/caps table the wave configs are RENDERED FROM",
    "sdfbase_full22_roster_psrcert.yaml":
        "the certification-only roster (native-psr-serving D15): the psrcert/psr2cert views' source table",
    "sdfbase_full22_arms.yaml":
        "the arms table (mt-arms): per-arm policy, fits and operating points served BESIDE the wave",
    "sdfbase_full22_roster_seedrep_probe_eb64_s1.yaml":
        "the seed-replicate battery's S5 window-probe roster (qwen-seedrep-native-battery W3 F-B, D45): the lane-owned "
        "probe views d1p5_lora / dm0p5_lora the two probe wave configs are RENDERED FROM, served in the smoke only",
    "sdfbase_full22_roster_ctl_neg1_eb64.yaml":
        "the ctl-neg1 lane's PRIVATE roster (selected only by F22_ROSTER, never by default): the six strength -1 "
        "seed-replicate control views dm1_ctllora64s<k> / dm1_ctlrelu64s<k> are RENDERED FROM it -- a roster, not a wave config",
    "sdfbase_full22_ports.yaml":
        "the serving lane's PORT table (paper-companion-repo, report_D): one port per (target, view) TAG of the served waves, read by "
        "petri_proxy.sbatch / petri_drive_full22.sh -- a table under the wave prefix, never a wave config (found red by the run-readiness "
        "lane's covering run of this suite 2026-09-29; the table predates that lane)",
}


def wave_configs(repo: Path = REPO) -> list[Path]:
    """Every ``configs/sdfbase_full22_*.yaml`` that IS a wave config, sorted by path: the glob minus the named
    tables and the generated sampled configs. Deliberately no shape filter -- a wave config missing a key must
    reach the caller and fail there, by name."""
    return sorted(p for p in (repo / "configs").glob(CONFIGS_GLOB)
                  if p.name not in EXCLUDED_TABLES and SAMPLED_MARK not in p.name)
