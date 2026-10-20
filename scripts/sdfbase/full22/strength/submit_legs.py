#!/usr/bin/env python
"""STRENGTH FRONTIER (paper Fig. 5) — PRINT (never run) the sbatch lines for the EXPENSIVE LEGS the screen's verdict permits.

  submit_legs.py <screen_verdict.json> --wave-sha <40-hex> [--roster PATH] [--evidence-roster PATH ...]
                 [--wave-checkout DIR] [--sacct embedded|live | --sacct-file CLUSTER=PATH ...]
                 [--attackers N | --live-pool] [--cluster TAG=cluster_a|cluster_b ...]
                 [--cluster-b-budget N] [--cluster-a-budget N] [--rerun-existing] [--assume-ok] [--nshards 9]

A PRINTER, modelled on scripts/sdfbase/full22/sampled/submit_sampled.py: it never executes sbatch, scancel or touch; the
only processes it may start are read-only (`git rev-parse` / the generators' `--check` in --wave-checkout, `sacct` over
ssh with --sacct live). It consumes ONE input that matters — the record screen_verdict.py wrote (the stop rule,
evaluated per (target, method, dose) on the screen's 512 rung) — and prints, in launch order and with a per-leg
rationale, the jobs for the legs that verdict permits:

  static families, SAMPLED   target_sampled.sbatch <key> static_base_t<T> <sha> 9 "" <view>   (T = 0, 1), then
                             static_esc_t<T> once sb<T> is COMPLETE (second block; no --dependency, the budget)
  BoN at 512                 target_static.sbatch <key> <view> bon <sha> 9     (the 512 store the BoN climb reads)
  multi-turn                 target_attacks.sbatch <key> <view> crescendo,pair,actorattack <sha> 9 0,1,...,8 — needs the
                             SHARED ATTACKER POOL (attacker_pool.sbatch): members launched by THIS lane under OUR names
                             (str-atk-<n>), released ONLY by touching $WAVE_ROOT/attackers/<jobid>.STOP for OUR member ids,
                             ONLY after every one of our attack drivers has ENDED and no peer's attack job is live on that
                             cluster — NEVER the pool-wide attackers/STOP (a peer lost 20 goals to that once). Refuses to
                             print attack jobs unless it also prints members, or is told a live pool exists (--live-pool).
  MMLU                       target_sampled.sbatch <key> capability_t0 <sha> 9 "" <view> — that phase ALSO runs the RETIRED
                             GSM8K thinking-ON instrument (sampled/gsm8k_thinking_on_cap8192/<TAG>_t0) whose rows this lane
                             must never read; the warning is printed beside every capability line.
  GSM8K                      NOT this printer's: it comes from a separate inspect gsm8k_paired200 launcher (run
                             record `gsm8k`); a placeholder per admissible TAG names what to run once it lands, with the two
                             requirements it must meet (an explicit --max-tokens-by-task gsm8k_paired200=8192, and the
                             hybrid model's rows on cluster A only because its fused-kernel overlay builds only there).

WHICH DOSES EACH LEG RUNS: do not re-run 1.0, or 0.5 where it exists — encoded as D33_EXISTS (a per-leg map
dose -> what exists), applied on top of the verdict. `--rerun-existing` voids it (the drift check compares the
re-served 1.0 screen against the existing 1.0 controls; disagreement stops the reuse plan), loudly.

WHAT THE VERDICT WITHHOLDS, per CELL not per job: a view's job serves every cell of the view, but the verdict is per
(target, method, dose), so a two-method view (nemotron / glm) whose const-LoRA cell stopped and whose LoRA-subtraction cell
is ok is submitted with an explicit cells positional `none,<surviving cell>`. A cell is admissible for a leg when its dose is
in the set that leg runs AND its verdict is ok / flagged (flagged is annotated) AND it is neither `withheld_own_legs`
(its own screen stopped it) nor `must_not_launch` (beyond a closure) — except the 1.0 anchor, exempt from
closure by the rungs below it but never from its own stop. `incomplete` IS NOT PERMISSION: such a cell
is refused with its unevaluable clauses named; `--assume-ok` overrides, and every line it enables says so.
Every omission prints WHICH clause caused it and with what numbers (the verdict's `fired_detail`), so the plan explains itself.

ENUMERATION is from the verdict's embedded roster views (screen_verdict.load_dose_views: every roster view with a `dose:`
key) — never a list of tags in this file. `--roster` cross-checks that roster (sha256 must equal the verdict's;
a different roster means re-run the verdict, not re-interpret it).

--time IS DERIVED FROM MEASURED WALLTIMES, never copied from the lane printers' defaults (static 16 h, controls 24 h,
attacks 20 h — generous on purpose). Evidence: `sacct -X -n -P -u $USER -S now-21days` on BOTH clusters, families matched
by job name (the retry / wave suffixes -w<N>[r] and -r<N> included), completed rows only for the fits. Per leg and target:
`elapsed = overhead + n_object_cells x marginal` least-squares when the target has two cell-count levels; else the median
of the nearest measured view, which has MORE cells than any dose view (2-3 cells vs the campaign's 8-9), so it is an upper
bound — a purely linear per-cell scaling would UNDER-count small views (the fixed standup dominates). --time = the
prediction x2 (backfill wants ~2x, not 5x), rounded up to 30 min, with a per-leg floor where the dose views' at-cap volume
is unmeasured. The evidence (rows, model, prediction) is printed beside every --time. The default evidence is the snapshot
embedded below (queried 2026-09-10); `--sacct live` re-queries (read-only), `--sacct-file` reads a saved PSV.

ROUTING: every dose TAG goes to DEFAULT_CLUSTER, and `--cluster TAG=<cluster>` overrides. Every value a printed command
carries comes from that cluster's block in the site config (reproduce/lib/clusters.py; the block whose label, CLUSTER_LABEL
or CLUSTER_SUBSTRATE names the cluster): the ssh host (CLUSTER_SSH), the checkout (CLUSTER_EVAL_REPO_ROOT/<sha>), --qos,
--partition, --account, --output (CLUSTER_LOG_DIR), CLUSTER_SBATCH_EXTRA, the pool's release root (CLUSTER_SHARED_ROOT)
and the budget (CLUSTER_HIGH_BUDGET_GPUS). A cluster the plan routes to with no block, or with more than one, refuses by
name, and so does `--sacct live` for any cluster without one. Each leg's GPU-hours (expected, and reserved by --time) and
the budget it consumes are printed; first-block jobs beyond the free budget print as `# HOLD` lines (a PENDING job counts
against the cap).

Exit 0 = printed; 2 = refused (a bad sha, config drift in --wave-checkout, a roster mismatch, an unreachable cluster
under --sacct live, a routed cluster with no site block, attack jobs asked for without a pool).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent            # scripts/sdfbase/full22/strength
F22 = HERE.parent                                 # scripts/sdfbase/full22
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import screen_verdict as sv  # noqa: E402  — the producer's constants, job-name patterns and closure function
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import clusters as CL  # noqa: E402  — the site's cluster blocks: every host, path, QoS and flag a printed command carries

S = "scripts/sdfbase/full22"
SS = "scripts/sdfbase/full22/sampled"
EXIT_REFUSED = 2


def site_block(cluster: str) -> dict:
    """The site's cluster block for `cluster` (reproduce/lib/clusters.py, from the config file $SISTEER_CONFIG names, else
    `<site dir>/config.env`): the one block whose label (its `clusters/<label>.env` name), CLUSTER_LABEL or CLUSTER_SUBSTRATE
    names the cluster, among every block under the site's clusters/ directory (a dose TAG may be routed to a cluster the site
    does not fan out to by default, which SISTEER_CLUSTERS does not list). No site config, no such block, more than one, or a
    malformed config refuses by name: a printed command never falls back to a built-in host, path or QoS."""
    cfg = CL.default_config_path()
    if not Path(cfg).is_file():
        raise Refusal(f"cluster {cluster!r}: no site config at {cfg} (set SISTEER_CONFIG or SISTEER_SITE_DIR; every printed command "
                      "takes its host, checkout, QoS and flags from the cluster's block)")
    try:
        blocks = CL.load_addressable(str(Path(cfg).parent), CL.site_config_file(cfg))
    except CL.ClusterConfigError as exc:
        raise Refusal(f"cluster {cluster!r}: site config {cfg}: {exc}")
    hits = [lab for lab, b in blocks.items() if cluster in (lab, b.get("CLUSTER_LABEL"), b.get("CLUSTER_SUBSTRATE"))]
    if len(hits) != 1:
        raise Refusal(f"cluster {cluster!r}: {'no' if not hits else len(hits)} cluster block(s) of {cfg} name it by label, "
                      f"CLUSTER_LABEL or CLUSTER_SUBSTRATE (blocks: {', '.join(blocks) or 'none'}{'; matches: ' + ', '.join(hits) if hits else ''})")
    return blocks[hits[0]]


DEFAULT_CLUSTER = "cluster_b"   # every dose TAG (submit_sampled.py's routing; cluster A's 16 is spoken for by the main views)
#: the hybrid model whose fused-kernel (mamba-ssm / causal-conv1d) overlay builds only on cluster A — its GSM8K rows are
#: cluster-A-only (run record `gsm8k.my_requirements`)
FUSED_KERNEL_CLUSTER_A_ONLY = ("nemotron3nano4b",)

LEGS = ("static", "bon", "multiturn", "capability", "gsm8k")
LEG_LABEL = {"static": "static families (SAMPLED, 2 trials + the 8192 rung)", "bon": "BoN at 512 (100 JBB goals x 75)",
             "multiturn": "multi-turn (crescendo, pair, actorattack; shared attacker pool)",
             "capability": "MMLU via capability_t0 (ALSO runs the retired GSM8K thinking-ON instrument)",
             "gsm8k": "GSM8K (the sibling lane's inspect gsm8k_paired200 launcher — placeholder, not this printer's)"}
#: what already EXISTS per leg and dose, hence is NOT re-run. Applied on top of the verdict.
#: The doses that remain are the ladder's RUNS list: static -1/0.1/0.25/0.5/1.5; BoN-512, multi-turn and MMLU
#: -1/0.1/0.25/1.5; GSM8K 0.5 + the new doses (1.0 from the follow-up).
D33_EXISTS: Dict[str, Dict[float, str]] = {
    "static": {1.0: "sampled static rows at 1.0 exist (Sample's f22s-<key>[_it]-sb0/sb1, cluster B)"},
    "bon": {0.5: "a 512 BoN store exists (the halfdose__/halfdoseit__ objects: hd-stat-*-3)",
            1.0: "a 512 BoN store exists (the campaign's f22-stat-*-3); its 4,096 climb is Scale's"},
    "multiturn": {0.5: "greedy/512 multi-turn rows exist (halfdose, all four targets) — the single largest saving, ~1/3 of the most expensive leg",
                  1.0: "greedy/512 multi-turn rows exist (the campaign's f22-att-*)"},
    "capability": {0.5: "MMLU exists (the halfdose controls; letter-logprob, sampler-moot)",
                   1.0: "MMLU exists (the campaign's controls)"},
    "gsm8k": {1.0: "1.0 comes from the sibling lane's own 24-cell follow-up"},
}
D33_TEXT = ("D33 (the user, 2026-09-10 15:35): do not re-run 1.0, or 0.5 where it exists, assuming serving code is identical; "
            "ideally not BoN either — follow the 512 -> 4096 climb. The 'serving code identical' condition is FALSE (D33's own "
            "finding) and is being MEASURED by the d1 / it_d1 screen jobs against Sample's 1.0 controls; --rerun-existing is "
            "the switch for when that check fails.")
ATTACKS = ("crescendo", "pair", "actorattack")
SHARDS = "0,1,2,3,4,5,6,7,8"
N_TRIALS = 2
POOL_JOB_PREFIX = "str-atk"           # OUR member names (str-atk-char-<n> / str-atk-bulb-<n>, the hd-atk-char-<n> precedent) — the release
POOL_CLUSTER_SHORT = {"cluster_b": "char", "cluster_a": "bulb"}   # command resolves ids by these names, nothing else
CLUSTERS = tuple(sorted(POOL_CLUSTER_SHORT))   # the clusters a dose TAG can be routed to; their commands come from site_block
POOL_TARGET_WALL_H = 12.0             # ~4 members x ~12 h: size the pool so the attack wave fits ~12 h of pool life
POOL_MIN, POOL_MAX = 2, 12
PAD = 2.0                             # --time = prediction x PAD (backfill wants ~2x, not 5x)
#: per-leg floors on --time (hours), each with its reason (printed)
FLOOR_H = {"static_base": (2.0, "one standup + 4 families x 9 shards; the smallest measured view took >= 1.5 h"),
           "static_esc": (3.0, "the 8192 rung re-draws every at-cap row: at-cap VOLUME at -1 / 1.5 is unmeasured (a heavy rung is 3x the campaign's)"),
           "bon": (2.0, "75 draws x 100 goals per cell at 512; a looping dose fills the cap on every draw"),
           "multiturn": (3.0, "attacker-bound; the pool's load from peers is not under this lane's control"),
           "capability": (2.0, "one thinking-ON standup (8192 window) + one thinking-OFF standup; a single completed row measured"),
           "attacker_pool": (4.0, "a member must outlive the attack wave; walltime is the only thing that ends it besides our .STOP")}
GSM8K_REQUIREMENTS = ("an explicit `--max-tokens-by-task gsm8k_paired200=8192` (the thinking-ON default otherwise resolves to the 2,048 floor)",
                      f"the hybrid model's ({', '.join(FUSED_KERNEL_CLUSTER_A_ONLY)}) rows are CLUSTER-A-ONLY: its fused-kernel overlay builds only there",
                      "per-view bank views at this lane's sha on whichever cluster runs (TAG grammar <key>_<view>, matches)")
CAPABILITY_WARNING = ("!! capability_t0 ALSO runs the RETIRED GSM8K thinking-ON @8192 instrument into "
                      "sampled/gsm8k_thinking_on_cap8192/<TAG>_t0 — this lane's GSM8K is the sibling lane's inspect gsm8k_paired200; "
                      "NEVER read those rows (MMLU lands in sampled/mmlu/<TAG>_t0)")
POOL_RULES = (
    "members are launched by THIS lane under OUR job names (str-atk-char-<n> / str-atk-bulb-<n>) — never re-use a peer's members' ids in any command",
    "every attack job on the cluster round-robins over EVERY live record in $WAVE_ROOT/attackers/ (common.sh f22_live_attackers): our drivers use peers' members and theirs use ours",
    "release ONLY by touching $WAVE_ROOT/attackers/<jobid>.STOP for OUR OWN member ids (resolved by our names via squeue), ONLY after every one of our f22-att-<TAG>-all jobs has ENDED and no other attack job (any *-att-* name, any lane) is RUNNING or PENDING on that cluster",
    "NEVER touch $WAVE_ROOT/attackers/STOP: it is POOL-WIDE and ends every member of every lane — a peer lost 20 goals to exactly that",
    "never `scancel` a member by bare id; the .STOP file makes the member exit 0 and remove its record cleanly",
)

# --------------------------------------------------------------------------------------------------------------- #
# measured walltimes: the evidence and its families                                                                #
# --------------------------------------------------------------------------------------------------------------- #

#: job-name families (the retry / wave suffixes -w<N>[r] and -r<N> are real: f22s-nemotron3nano4b-ctl0-w7,
#: f22s-glm46vflash-ctl1-w9r, f22s-gemma4e4b-cap0-r3 — a regex without them missed the ONLY completed capability job)
_SUF = r"(?P<suffix>-w\d+r?|-r\d+)?"
ROW_FAMILIES = {
    "sb": re.compile(rf"^f22s-(?P<tag>.+?)-sb[01](?P<half>[ab])?{_SUF}$"),
    "se": re.compile(rf"^f22s-(?P<tag>.+?)-se[01]{_SUF}$"),
    "ctl": re.compile(rf"^f22s-(?P<tag>.+?)-ctl[01]{_SUF}$"),
    "cap": re.compile(rf"^f22s-(?P<tag>.+?)-cap[01]{_SUF}$"),
    "stat": re.compile(r"^(?:f22|hd)-stat-(?P<tag>.+?)-(?P<grp>\d[ab]?|all|bon)$"),
    "att": re.compile(r"^(?:f22|hd)-att-(?P<tag>.+?)-(?P<grp>crescendo|pair|actorattack|all)$"),
    "atk": re.compile(r"^(?:f22|hd)-atk-(?P<grp>.+)$"),
    "hdctl": re.compile(r"^hd-ctl-(?P<tag>.+)$"),
    "b4k": re.compile(r"^b4k-(?P<tag>.+)$"),
}
SACCT_FORMAT = "JobID,JobName%80,State,ElapsedRaw,Start,NodeList"
SACCT_CMD = f"sacct -X -n -P -u $USER -S now-21days -o {SACCT_FORMAT}"
#: the embedded evidence: (cluster, jobid, job name, state, elapsed seconds, start, node) — COMPLETED rows of the families
#: above plus informative failures (>= 30 min, or on the excluded node), as `sacct -X -n -P -u $USER -S now-21days` returned
#: them on both clusters on 2026-09-10 ~23:30 UTC, one login host per cluster. Refresh with --sacct live.
SNAPSHOT_QUERIED = "2026-09-10T23:35Z"
SNAPSHOT_ROWS: List[Tuple[str, str, str, str, int, str, str]] = [
    ("cluster_a", "423390", "f22-stat-gemma4e4b-2", "COMPLETED", 7792, "2026-09-07T02:59", "node-9"),
    ("cluster_a", "423391", "f22-stat-qwen35_4b-1", "COMPLETED", 8283, "2026-09-07T02:59", "node-9"),
    ("cluster_a", "423392", "f22-stat-qwen35_4b-2", "COMPLETED", 12537, "2026-09-07T02:59", "node-9"),
    ("cluster_a", "423449", "f22-stat-gemma4e4b-1", "COMPLETED", 3766, "2026-09-07T03:04", "node-31"),
    ("cluster_a", "425390", "f22-stat-gemma4e4b_it-all", "COMPLETED", 44474, "2026-09-07T04:09", "node-27"),
    ("cluster_a", "427299", "f22-stat-qwen35_4b_it-all", "COMPLETED", 46261, "2026-09-07T05:11", "node-9"),
    ("cluster_a", "427373", "f22-stat-gemma4e4b-3", "COMPLETED", 12529, "2026-09-07T05:19", "node-18"),
    ("cluster_a", "428502", "f22-stat-qwen35_4b-3", "OUT_OF_MEMORY", 11322, "2026-09-07T06:27", "node-31"),
    ("cluster_a", "450992", "f22-atk-rerun-1", "COMPLETED", 3382, "2026-09-08T03:23", "node-20"),
    ("cluster_a", "463749", "hd-ctl-gemma4e4b_it_halfdose", "COMPLETED", 1420, "2026-09-08T17:15", "node-16"),
    ("cluster_a", "463748", "hd-stat-gemma4e4b_halfdose-3", "FAILED", 8674, "2026-09-08T17:15", "node-17"),
    ("cluster_a", "463790", "hd-stat-gemma4e4b_halfdose-1", "FAILED", 2331, "2026-09-08T17:19", "node-1"),
    ("cluster_a", "463791", "hd-stat-gemma4e4b_halfdose-2", "FAILED", 8947, "2026-09-08T17:19", "node-25"),
    ("cluster_a", "463792", "hd-stat-qwen35_4b_halfdose-1", "COMPLETED", 2879, "2026-09-08T17:19", "node-24"),
    ("cluster_a", "463793", "hd-stat-qwen35_4b_halfdose-2", "COMPLETED", 6286, "2026-09-08T17:19", "node-24"),
    ("cluster_a", "463794", "hd-stat-qwen35_4b_halfdose-3", "COMPLETED", 6677, "2026-09-08T17:19", "node-24"),
    ("cluster_a", "463796", "hd-ctl-gemma4e4b_halfdose", "FAILED", 1855, "2026-09-08T17:22", "node-20"),
    ("cluster_a", "463797", "hd-ctl-qwen35_4b_halfdose", "COMPLETED", 2133, "2026-09-08T17:22", "node-20"),
    ("cluster_a", "463798", "hd-ctl-qwen35_4b_it_halfdose", "COMPLETED", 1385, "2026-09-08T17:23", "node-23"),
    ("cluster_a", "464436", "hd-atk-bulb-1", "COMPLETED", 54444, "2026-09-08T19:10", "node-24"),
    ("cluster_a", "464437", "hd-atk-bulb-2", "COMPLETED", 54384, "2026-09-08T19:11", "node-17"),
    ("cluster_a", "464653", "hd-att-qwen35_4b_halfdose-actorattack", "COMPLETED", 21543, "2026-09-08T19:33", "node-20"),
    ("cluster_a", "464651", "hd-att-qwen35_4b_halfdose-crescendo", "COMPLETED", 15197, "2026-09-08T19:33", "node-9"),
    ("cluster_a", "464652", "hd-att-qwen35_4b_halfdose-pair", "COMPLETED", 9855, "2026-09-08T19:33", "node-21"),
    ("cluster_a", "466331", "hd-ctl-glm46vflash_halfdose", "COMPLETED", 2316, "2026-09-09T00:38", "node-22"),
    ("cluster_a", "466328", "hd-stat-glm46vflash_halfdose-1", "COMPLETED", 9883, "2026-09-09T00:38", "node-19"),
    ("cluster_a", "466329", "hd-stat-glm46vflash_halfdose-2", "COMPLETED", 12142, "2026-09-09T00:38", "node-20"),
    ("cluster_a", "466330", "hd-stat-glm46vflash_halfdose-3", "COMPLETED", 10334, "2026-09-09T00:39", "node-23"),
    ("cluster_a", "467167", "hd-att-glm46vflash_halfdose-actorattack", "COMPLETED", 22811, "2026-09-09T03:55", "node-19"),
    ("cluster_a", "467165", "hd-att-glm46vflash_halfdose-crescendo", "COMPLETED", 12602, "2026-09-09T03:55", "node-17"),
    ("cluster_a", "467166", "hd-att-glm46vflash_halfdose-pair", "COMPLETED", 11321, "2026-09-09T03:55", "node-21"),
    ("cluster_a", "469650", "hd-atk-petri-aud", "COMPLETED", 65348, "2026-09-09T12:51", "node-13"),
    ("cluster_a", "476777_0", "b4k-gemma4e4b_it", "FAILED", 81, "2026-09-10T18:01", "node-25"),
    ("cluster_a", "476802_2", "b4k-glm46vflash", "FAILED", 4, "2026-09-10T18:08", "node-25"),
    ("cluster_a", "476802_4", "b4k-glm46vflash", "FAILED", 4, "2026-09-10T18:08", "node-25"),
    ("cluster_a", "476803_0", "b4k-gemma4e4b_it", "FAILED", 4, "2026-09-10T18:09", "node-25"),
    ("cluster_a", "476802_7", "b4k-glm46vflash", "FAILED", 4, "2026-09-10T18:09", "node-25"),
    ("cluster_a", "476804_1", "b4k-qwen35_4b_it", "FAILED", 4, "2026-09-10T18:09", "node-25"),
    ("cluster_a", "476804_2", "b4k-qwen35_4b_it", "FAILED", 4, "2026-09-10T18:09", "node-25"),
    ("cluster_a", "476804_3", "b4k-qwen35_4b_it", "FAILED", 4, "2026-09-10T18:09", "node-25"),
    ("cluster_a", "476805_2", "b4k-nemotron3nano4b", "FAILED", 4, "2026-09-10T18:10", "node-25"),
    ("cluster_a", "476805_3", "b4k-nemotron3nano4b", "FAILED", 4, "2026-09-10T18:10", "node-25"),
    ("cluster_a", "476828_0", "b4k-glm46vflash", "COMPLETED", 5489, "2026-09-10T18:12", "node-7"),
    ("cluster_a", "476828_1", "b4k-glm46vflash", "COMPLETED", 5351, "2026-09-10T18:12", "node-13"),
    ("cluster_a", "476828_2", "b4k-glm46vflash", "COMPLETED", 5682, "2026-09-10T18:12", "node-13"),
    ("cluster_a", "476828_3", "b4k-glm46vflash", "COMPLETED", 5528, "2026-09-10T18:12", "node-22"),
    ("cluster_a", "476829_0", "b4k-gemma4e4b_it", "FAILED", 2112, "2026-09-10T18:13", "node-25"),
    ("cluster_a", "476828_4", "b4k-glm46vflash", "COMPLETED", 5378, "2026-09-10T18:13", "node-27"),
    ("cluster_a", "476828_5", "b4k-glm46vflash", "COMPLETED", 5228, "2026-09-10T18:13", "node-24"),
    ("cluster_a", "476828_6", "b4k-glm46vflash", "COMPLETED", 6641, "2026-09-10T18:14", "node-6"),
    ("cluster_a", "476829_1", "b4k-gemma4e4b_it", "FAILED", 2593, "2026-09-10T18:15", "node-25"),
    ("cluster_a", "476829_2", "b4k-gemma4e4b_it", "FAILED", 2608, "2026-09-10T18:16", "node-25"),
    ("cluster_a", "476830_0", "b4k-qwen35_4b_it", "FAILED", 1824, "2026-09-10T18:17", "node-25"),
    ("cluster_a", "476828_7", "b4k-glm46vflash", "COMPLETED", 5789, "2026-09-10T18:18", "node-7"),
    ("cluster_a", "476830_1", "b4k-qwen35_4b_it", "COMPLETED", 1953, "2026-09-10T18:19", "node-9"),
    ("cluster_a", "476830_2", "b4k-qwen35_4b_it", "COMPLETED", 2173, "2026-09-10T18:19", "node-24"),
    ("cluster_a", "476830_3", "b4k-qwen35_4b_it", "COMPLETED", 2127, "2026-09-10T18:21", "node-16"),
    ("cluster_a", "476831_0", "b4k-nemotron3nano4b", "COMPLETED", 1093, "2026-09-10T19:07", "node-24"),
    ("cluster_a", "476831_1", "b4k-nemotron3nano4b", "COMPLETED", 1827, "2026-09-10T19:08", "node-28"),
    ("cluster_a", "476831_2", "b4k-nemotron3nano4b", "COMPLETED", 1170, "2026-09-10T19:12", "node-7"),
    ("cluster_a", "476970_0", "b4k-gemma4e4b_it", "COMPLETED", 2448, "2026-09-10T19:13", "node-3"),
    ("cluster_a", "476970_1", "b4k-gemma4e4b_it", "COMPLETED", 2127, "2026-09-10T19:13", "node-19"),
    ("cluster_a", "476970_2", "b4k-gemma4e4b_it", "COMPLETED", 2409, "2026-09-10T19:13", "node-8"),
    ("cluster_a", "476831_3", "b4k-nemotron3nano4b", "COMPLETED", 1179, "2026-09-10T19:13", "node-31"),
    ("cluster_a", "476942_0", "b4k-qwen35_4b_it", "COMPLETED", 1355, "2026-09-10T19:13", "node-16"),
    ("cluster_b", "73775", "f22-atk-1", "COMPLETED", 44383, "2026-09-07T01:36", "node-0"),
    ("cluster_b", "73937", "f22-atk-2", "COMPLETED", 41029, "2026-09-07T02:32", "node-6"),
    ("cluster_b", "73938", "f22-atk-3", "COMPLETED", 41024, "2026-09-07T02:32", "node-10"),
    ("cluster_b", "73939", "f22-atk-4", "COMPLETED", 41023, "2026-09-07T02:32", "node-10"),
    ("cluster_b", "73940", "f22-atk-5", "COMPLETED", 41009, "2026-09-07T02:32", "node-11"),
    ("cluster_b", "73941", "f22-atk-6", "COMPLETED", 41011, "2026-09-07T02:32", "node-11"),
    ("cluster_b", "73942", "f22-atk-7", "COMPLETED", 40994, "2026-09-07T02:32", "node-8"),
    ("cluster_b", "73943", "f22-atk-8", "COMPLETED", 41031, "2026-09-07T02:32", "node-3"),
    ("cluster_b", "74033", "f22-att-gemma4e4b-actorattack", "COMPLETED", 39115, "2026-09-07T03:01", "node-13"),
    ("cluster_b", "74032", "f22-att-gemma4e4b-pair", "COMPLETED", 12783, "2026-09-07T03:01", "node-11"),
    ("cluster_b", "74034", "f22-att-qwen35_4b-crescendo", "COMPLETED", 18804, "2026-09-07T03:02", "node-2"),
    ("cluster_b", "74056", "f22-att-gemma4e4b-crescendo", "COMPLETED", 17231, "2026-09-07T03:09", "node-12"),
    ("cluster_b", "74318", "f22-att-gemma4e4b_it-all", "COMPLETED", 21332, "2026-09-07T04:17", "node-12"),
    ("cluster_b", "74319", "f22-att-qwen35_4b-pair", "COMPLETED", 14317, "2026-09-07T04:17", "node-1"),
    ("cluster_b", "74320", "f22-att-qwen35_4b-actorattack", "COMPLETED", 24510, "2026-09-07T04:18", "node-13"),
    ("cluster_b", "74321", "f22-att-qwen35_4b_it-all", "COMPLETED", 17392, "2026-09-07T04:18", "node-2"),
    ("cluster_b", "76387", "f22-atk-1", "COMPLETED", 30808, "2026-09-07T16:16", "node-9"),
    ("cluster_b", "77035", "f22-atk-2", "COMPLETED", 15069, "2026-09-07T19:25", "node-0"),
    ("cluster_b", "77036", "f22-atk-3", "COMPLETED", 15070, "2026-09-07T19:25", "node-0"),
    ("cluster_b", "77037", "f22-atk-4", "COMPLETED", 15056, "2026-09-07T19:25", "node-8"),
    ("cluster_b", "77038", "f22-atk-5", "COMPLETED", 15077, "2026-09-07T19:25", "node-10"),
    ("cluster_b", "77039", "f22-atk-6", "COMPLETED", 15053, "2026-09-07T19:25", "node-4"),
    ("cluster_b", "77040", "f22-stat-nemotron3nano4b-1", "COMPLETED", 3303, "2026-09-07T19:25", "node-4"),
    ("cluster_b", "77041", "f22-stat-nemotron3nano4b-2", "COMPLETED", 8454, "2026-09-07T19:25", "node-2"),
    ("cluster_b", "77042", "f22-stat-nemotron3nano4b-3", "COMPLETED", 3246, "2026-09-07T19:25", "node-12"),
    ("cluster_b", "77066", "f22-att-nemotron3nano4b-crescendo", "COMPLETED", 10374, "2026-09-07T19:37", "node-8"),
    ("cluster_b", "77068", "f22-att-nemotron3nano4b-actorattack", "COMPLETED", 13955, "2026-09-07T19:39", "node-11"),
    ("cluster_b", "77067", "f22-att-nemotron3nano4b-pair", "COMPLETED", 7520, "2026-09-07T19:39", "node-6"),
    ("cluster_b", "78767", "f22-atk-glm-1", "FAILED", 86349, "2026-09-08T04:42", "node-10"),
    ("cluster_b", "78768", "f22-atk-glm-2", "COMPLETED", 25551, "2026-09-08T04:42", "node-0"),
    ("cluster_b", "78769", "f22-atk-glm-3", "COMPLETED", 25558, "2026-09-08T04:42", "node-10"),
    ("cluster_b", "78771", "f22-atk-glm-4", "COMPLETED", 25534, "2026-09-08T04:42", "node-12"),
    ("cluster_b", "78775", "f22-atk-nemo-1", "COMPLETED", 25339, "2026-09-08T04:43", "node-0"),
    ("cluster_b", "78776", "f22-atk-nemo-2", "COMPLETED", 25251, "2026-09-08T04:44", "node-6"),
    ("cluster_b", "78890", "f22-att-glm46vflash-actorattack", "COMPLETED", 22950, "2026-09-08T05:20", "node-0"),
    ("cluster_b", "78888", "f22-att-glm46vflash-crescendo", "COMPLETED", 13713, "2026-09-08T05:20", "node-6"),
    ("cluster_b", "78889", "f22-att-glm46vflash-pair", "COMPLETED", 11179, "2026-09-08T05:20", "node-0"),
    ("cluster_b", "78883", "f22-stat-glm46vflash-1", "COMPLETED", 11284, "2026-09-08T05:20", "node-10"),
    ("cluster_b", "78884", "f22-stat-glm46vflash-2", "COMPLETED", 10345, "2026-09-08T05:20", "node-3"),
    ("cluster_b", "78886", "f22-stat-glm46vflash-4", "COMPLETED", 10071, "2026-09-08T05:20", "node-12"),
    ("cluster_b", "78958", "f22-stat-glm46vflash-3a", "COMPLETED", 11353, "2026-09-08T05:39", "node-4"),
    ("cluster_b", "78959", "f22-stat-glm46vflash-3b", "COMPLETED", 11143, "2026-09-08T05:39", "node-8"),
    ("cluster_b", "80716", "hd-stat-nemotron3nano4b_halfdose-3", "COMPLETED", 2964, "2026-09-08T17:15", "node-12"),
    ("cluster_b", "80720", "hd-ctl-nemotron3nano4b_halfdose", "COMPLETED", 987, "2026-09-08T17:19", "node-13"),
    ("cluster_b", "80718", "hd-stat-nemotron3nano4b_halfdose-1", "COMPLETED", 1154, "2026-09-08T17:19", "node-3"),
    ("cluster_b", "80719", "hd-stat-nemotron3nano4b_halfdose-2", "COMPLETED", 9014, "2026-09-08T17:19", "node-12"),
    ("cluster_b", "80897", "hd-atk-char-1", "COMPLETED", 34414, "2026-09-08T18:55", "node-12"),
    ("cluster_b", "80898", "hd-atk-char-2", "COMPLETED", 34415, "2026-09-08T18:55", "node-12"),
    ("cluster_b", "80899", "hd-atk-char-3", "COMPLETED", 34414, "2026-09-08T18:55", "node-0"),
    ("cluster_b", "80900", "hd-atk-char-4", "COMPLETED", 34413, "2026-09-08T18:55", "node-0"),
    ("cluster_b", "80925", "hd-att-gemma4e4b_halfdose-actorattack", "FAILED", 26668, "2026-09-08T19:10", "node-13"),
    ("cluster_b", "80923", "hd-att-gemma4e4b_halfdose-crescendo", "FAILED", 13880, "2026-09-08T19:10", "node-8"),
    ("cluster_b", "80924", "hd-att-gemma4e4b_halfdose-pair", "FAILED", 10941, "2026-09-08T19:10", "node-12"),
    ("cluster_b", "80922", "hd-att-gemma4e4b_it_halfdose-all", "COMPLETED", 20442, "2026-09-08T19:10", "node-3"),
    ("cluster_b", "80920", "hd-att-nemotron3nano4b_halfdose-actorattack", "COMPLETED", 14974, "2026-09-08T19:10", "node-4"),
    ("cluster_b", "80918", "hd-att-nemotron3nano4b_halfdose-crescendo", "COMPLETED", 9312, "2026-09-08T19:10", "node-3"),
    ("cluster_b", "80919", "hd-att-nemotron3nano4b_halfdose-pair", "COMPLETED", 7112, "2026-09-08T19:10", "node-4"),
    ("cluster_b", "80921", "hd-att-qwen35_4b_it_halfdose-all", "COMPLETED", 17633, "2026-09-08T19:10", "node-10"),
    ("cluster_b", "84828", "f22s-gemma4e4b-ctl0", "COMPLETED", 12924, "2026-09-09T22:00", "node-6"),
    ("cluster_b", "84827", "f22s-gemma4e4b-sb0", "COMPLETED", 12458, "2026-09-09T22:00", "node-3"),
    ("cluster_b", "84833", "f22s-qwen35_4b-ctl0", "COMPLETED", 8530, "2026-09-09T22:01", "node-12"),
    ("cluster_b", "84829", "f22s-qwen35_4b-sb0a", "COMPLETED", 6814, "2026-09-09T22:01", "node-4"),
    ("cluster_b", "84830", "f22s-qwen35_4b-sb0b", "COMPLETED", 10331, "2026-09-09T22:01", "node-8"),
    ("cluster_b", "84831", "f22s-qwen35_4b-sb1a", "COMPLETED", 5852, "2026-09-09T22:01", "node-10"),
    ("cluster_b", "84832", "f22s-qwen35_4b-sb1b", "COMPLETED", 10730, "2026-09-09T22:01", "node-9"),
    ("cluster_b", "84840", "f22s-glm46vflash-ctl0", "FAILED", 15967, "2026-09-09T22:02", "node-12"),
    ("cluster_b", "84836", "f22s-glm46vflash-sb0a", "COMPLETED", 7677, "2026-09-09T22:02", "node-10"),
    ("cluster_b", "84837", "f22s-glm46vflash-sb0b", "COMPLETED", 14441, "2026-09-09T22:02", "node-10"),
    ("cluster_b", "84838", "f22s-glm46vflash-sb1a", "COMPLETED", 8462, "2026-09-09T22:02", "node-9"),
    ("cluster_b", "84839", "f22s-glm46vflash-sb1b", "COMPLETED", 13207, "2026-09-09T22:02", "node-4"),
    ("cluster_b", "84847", "f22s-nemotron3nano4b-sb0-w4", "COMPLETED", 5502, "2026-09-09T22:05", "node-6"),
    ("cluster_b", "84968", "f22s-gemma4e4b-sb1", "COMPLETED", 12723, "2026-09-09T22:26", "node-8"),
    ("cluster_b", "85022", "f22s-nemotron3nano4b-sb1", "COMPLETED", 5938, "2026-09-09T22:32", "node-6"),
    ("cluster_b", "85253", "f22s-nemotron3nano4b-ctl0-w7", "COMPLETED", 4044, "2026-09-10T00:19", "node-9"),
    ("cluster_b", "85254", "f22s-nemotron3nano4b-ctl1-w7", "COMPLETED", 5285, "2026-09-10T00:32", "node-12"),
    ("cluster_b", "85255", "f22s-nemotron3nano4b-se0-w7", "COMPLETED", 836, "2026-09-10T00:33", "node-8"),
    ("cluster_b", "85257", "f22s-gemma4e4b-ctl1", "COMPLETED", 12257, "2026-09-10T00:34", "node-8"),
    ("cluster_b", "85256", "f22s-nemotron3nano4b-se1-w7", "COMPLETED", 772, "2026-09-10T00:34", "node-9"),
    ("cluster_b", "85258", "f22s-qwen35_4b-ctl1", "COMPLETED", 11276, "2026-09-10T00:36", "node-12"),
    ("cluster_b", "85488", "f22s-gemma4e4b-se1", "COMPLETED", 536, "2026-09-10T02:00", "node-12"),
    ("cluster_b", "85502", "f22s-gemma4e4b-se0", "COMPLETED", 1018, "2026-09-10T02:06", "node-10"),
    ("cluster_b", "85503", "f22s-qwen35_4b-se0", "COMPLETED", 2445, "2026-09-10T02:07", "node-8"),
    ("cluster_b", "85504", "f22s-qwen35_4b-se1", "COMPLETED", 2412, "2026-09-10T02:07", "node-8"),
    ("cluster_b", "85505", "f22s-glm46vflash-se0", "FAILED", 5111, "2026-09-10T02:08", "node-8"),
    ("cluster_b", "85506", "f22s-glm46vflash-se1", "FAILED", 4884, "2026-09-10T02:09", "node-12"),
    ("cluster_b", "85507", "f22s-glm46vflash-ctl1", "FAILED", 23106, "2026-09-10T02:14", "node-9"),
    ("cluster_b", "86117", "f22s-gemma4e4b_it-ctl0", "COMPLETED", 9188, "2026-09-10T05:46", "node-6"),
    ("cluster_b", "86113", "f22s-gemma4e4b_it-sb0", "COMPLETED", 10331, "2026-09-10T05:46", "node-11"),
    ("cluster_b", "86115", "f22s-gemma4e4b_it-sb1", "COMPLETED", 10049, "2026-09-10T05:46", "node-6"),
    ("cluster_b", "86119", "f22s-gemma4e4b_it-ctl1", "COMPLETED", 8723, "2026-09-10T05:47", "node-12"),
    ("cluster_b", "86122", "f22s-qwen35_4b_it-ctl0", "COMPLETED", 3883, "2026-09-10T05:47", "node-8"),
    ("cluster_b", "86120", "f22s-qwen35_4b_it-sb0", "COMPLETED", 5444, "2026-09-10T05:47", "node-0"),
    ("cluster_b", "86121", "f22s-qwen35_4b_it-sb1", "COMPLETED", 5544, "2026-09-10T05:47", "node-13"),
    ("cluster_b", "86123", "f22s-qwen35_4b_it-ctl1", "COMPLETED", 3567, "2026-09-10T05:48", "node-6"),
    ("cluster_b", "86297", "f22s-glm46vflash-ctl0-w9", "COMPLETED", 12340, "2026-09-10T06:11", "node-12"),
    ("cluster_b", "86295", "f22s-glm46vflash-se0-w9", "COMPLETED", 3939, "2026-09-10T06:11", "node-3"),
    ("cluster_b", "86296", "f22s-glm46vflash-se1-w9", "COMPLETED", 3916, "2026-09-10T06:11", "node-10"),
    ("cluster_b", "88420", "f22s-glm46vflash-ctl1-w9r", "COMPLETED", 6506, "2026-09-10T16:51", "node-0"),
    ("cluster_b", "88421", "f22s-gemma4e4b_it-se0", "COMPLETED", 1625, "2026-09-10T18:26", "node-12"),
    ("cluster_b", "88422", "f22s-gemma4e4b_it-se1", "COMPLETED", 1431, "2026-09-10T18:27", "node-12"),
    ("cluster_b", "88423", "f22s-qwen35_4b_it-se0", "COMPLETED", 858, "2026-09-10T18:29", "node-12"),
    ("cluster_b", "88424", "f22s-qwen35_4b_it-se1", "COMPLETED", 712, "2026-09-10T18:31", "node-12"),
    ("cluster_b", "88649_0", "b4k-gemma4e4b", "COMPLETED", 3565, "2026-09-10T18:40", "node-0"),
    ("cluster_b", "88649_1", "b4k-gemma4e4b", "COMPLETED", 3321, "2026-09-10T18:43", "node-12"),
    ("cluster_b", "88649_2", "b4k-gemma4e4b", "COMPLETED", 3320, "2026-09-10T18:43", "node-13"),
    ("cluster_b", "88649_3", "b4k-gemma4e4b", "COMPLETED", 3371, "2026-09-10T18:43", "node-11"),
    ("cluster_b", "88650_0", "b4k-qwen35_4b", "COMPLETED", 3530, "2026-09-10T18:43", "node-8"),
    ("cluster_b", "88650_1", "b4k-qwen35_4b", "COMPLETED", 3641, "2026-09-10T18:43", "node-9"),
    ("cluster_b", "88650_2", "b4k-qwen35_4b", "COMPLETED", 3517, "2026-09-10T18:43", "node-12"),
    ("cluster_b", "88650_3", "b4k-qwen35_4b", "COMPLETED", 3593, "2026-09-10T18:44", "node-3"),
    ("cluster_b", "88650_4", "b4k-qwen35_4b", "COMPLETED", 3469, "2026-09-10T18:44", "node-10"),
    ("cluster_b", "88650_5", "b4k-qwen35_4b", "COMPLETED", 3675, "2026-09-10T18:44", "node-10"),
    ("cluster_b", "88650_6", "b4k-qwen35_4b", "COMPLETED", 3450, "2026-09-10T18:51", "node-12"),
    ("cluster_b", "88650_7", "b4k-qwen35_4b", "COMPLETED", 3430, "2026-09-10T18:53", "node-12"),
    ("cluster_b", "88977", "f22s-gemma4e4b-cap0-r3", "COMPLETED", 3091, "2026-09-10T19:55", "node-10"),
]


@dataclass
class Row:
    cluster: str
    jobid: str
    name: str
    state: str
    elapsed_s: int
    start: str
    node: str
    family: str = ""
    tag: str = ""
    grp: str = ""
    half: str = ""

    @property
    def hours(self) -> float:
        return self.elapsed_s / 3600.0


def classify_rows(raw: List[Tuple[str, str, str, str, int, str, str]]) -> List[Row]:
    out = []
    for cl, jid, name, state, el, start, node in raw:
        r = Row(cl, str(jid), name, state.split()[0] if state else "", int(el or 0), start or "", node or "")
        for fam, rx in ROW_FAMILIES.items():
            m = rx.match(name)
            if m:
                r.family = fam
                gd = m.groupdict()
                r.tag = gd.get("tag") or ""
                r.grp = gd.get("grp") or ""
                r.half = gd.get("half") or ""
                break
        if r.family:
            out.append(r)
    return out


def parse_sacct_psv(cluster: str, text: str) -> List[Tuple[str, str, str, str, int, str, str]]:
    """`sacct -X -n -P -o JobID,JobName,State,ElapsedRaw,Start,NodeList` lines -> raw tuples."""
    rows = []
    for ln in text.splitlines():
        f = ln.rstrip("\n").split("|")
        if len(f) < 6:
            continue
        jid, name, state, el, start, node = f[:6]
        rows.append((cluster, jid, name, state, int(el) if el.isdigit() else 0, start[:16], node))
    return rows


def sacct_live(hosts: Dict[str, str]) -> List[Tuple[str, str, str, str, int, str, str]]:
    """ONE read-only `sacct` per cluster over ssh (`hosts`: cluster -> its block's CLUSTER_SSH). An unreachable cluster is a
    refusal, not a silent shortfall of evidence."""
    rows: List[Tuple[str, str, str, str, int, str, str]] = []
    for cl, host in hosts.items():
        cmd = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host, SACCT_CMD]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise Refusal(f"--sacct live: ssh {host} timed out")
        if p.returncode != 0:
            raise Refusal(f"--sacct live: ssh {host} sacct failed rc={p.returncode}: {p.stderr.strip()[-400:]}")
        rows += parse_sacct_psv(cl, p.stdout)
    return rows


class Refusal(Exception):
    pass


# --------------------------------------------------------------------------------------------------------------- #
# the roster: dose views come from the verdict; rosters only cross-check and resolve the evidence rows' cell counts  #
# --------------------------------------------------------------------------------------------------------------- #

def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_roster(path: Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def roster_view_cells(roster: dict, key: str, view: str) -> Optional[int]:
    """Object cells a (key, view) SERVES per f22_roster.served_cells: the view's cells plus every ENABLED lane's cells whose
    flag names this view. None when the roster has no such view."""
    t = (roster.get("targets") or {}).get(key)
    if not t:
        return None
    v = (t.get("views") or {}).get(view)
    if not isinstance(v, dict):
        return None
    n = len(v.get("cells") or {})
    flags = roster.get("lane_flags") or {}
    for lane, cells in (t.get("lanes") or {}).items():
        flag = flags.get(lane) or {}
        if flag.get("enabled") and flag.get("view", "main") == view:
            n += len(cells or {})
    return n


def split_tag(tag: str, keys: List[str]) -> Optional[Tuple[str, str]]:
    """TAG -> (key, view): the longest roster key that is the TAG or a `<key>_` prefix of it (tag_of's inverse)."""
    for key in sorted(keys, key=len, reverse=True):
        if tag == key:
            return key, "main"
        if tag.startswith(key + "_"):
            return key, tag[len(key) + 1:]
    return None


class CellResolver:
    """n_object_cells of a measured job's TAG: the verdict's dose views first, then every roster given (the verdict's,
    --roster, --evidence-roster). Unresolved -> None (the row is listed as raw evidence, never scaled)."""

    def __init__(self, views: List[dict], rosters: List[Tuple[str, dict]]):
        self.dose = {v["tag"]: len(v["cells"]) for v in views}
        self.rosters = rosters
        self.keys = sorted({v["target"] for v in views} | {k for _, r in rosters for k in (r.get("targets") or {})})

    def n_obj(self, tag: str) -> Tuple[Optional[int], str]:
        if tag in self.dose:
            return self.dose[tag], "verdict view"
        kv = split_tag(tag, self.keys)
        if not kv:
            return None, "no roster key matches"
        key, view = kv
        for label, r in self.rosters:
            n = roster_view_cells(r, key, view)
            if n is not None:
                return n, label
        return None, f"view {key}/{view} in no roster given (pass --evidence-roster)"


# --------------------------------------------------------------------------------------------------------------- #
# walltime models                                                                                                    #
# --------------------------------------------------------------------------------------------------------------- #

def _fit(points: List[Tuple[int, float]]) -> Optional[dict]:
    """elapsed = a + b x n_obj over (n_obj, hours) points: least squares when >= 2 cell-count levels (b clipped at 0,
    a at 0.25 h — a standup is never free), else the median at the single level."""
    if not points:
        return None
    levels = sorted({n for n, _ in points})
    ys = [h for _, h in points]
    if len(levels) >= 2:
        xs = [float(n) for n, _ in points]
        xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
        var = sum((x - xbar) ** 2 for x in xs)
        b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / var if var else 0.0
        if b < 0:   # more cells measured FASTER: noise — fall back to the smallest level's median as the bound
            small = [h for n, h in points if n == levels[0]]
            return {"kind": "single", "median": statistics.median(small), "level": levels[0], "n": len(points),
                    "note": "negative slope across levels — noise; the smallest measured view's median is the bound"}
        a = max(0.25, ybar - b * xbar)
        return {"kind": "fit", "a": a, "b": b, "levels": levels, "n": len(points)}
    return {"kind": "single", "median": statistics.median(ys), "level": levels[0], "n": len(points), "note": ""}


def _predict(model: dict, x: int, unit: str = "object cells") -> Tuple[float, str]:
    """The prediction at x (the family's work unit: object cells; cells incl. none for BoN / capability; cells x attacks
    for multi-turn) and the sentence that justifies it."""
    if model["kind"] == "fit":
        p = model["a"] + model["b"] * x
        return p, f"fit over {model['n']} rows at {model['levels']} {unit}: {model['a']:.2f} h + {model['b']:.3f} h per unit -> {p:.2f} h at {x}"
    if x <= model["level"]:
        return model["median"], (f"median of {model['n']} rows at {model['level']} {unit} = {model['median']:.2f} h, an UPPER bound for {x} "
                                 f"(fewer; the fixed standup dominates, D31){'; ' + model['note'] if model.get('note') else ''}")
    p = 0.6 + (model["median"] - 0.6) * x / model["level"]
    return p, f"median {model['median']:.2f} h at {model['level']} {unit} scaled to {x} with D31's 0.6 h overhead -> {p:.2f} h"


def fmt_time(hours: float) -> str:
    """Round UP to the next 30 min; D-HH:MM:SS past 24 h."""
    halves = math.ceil(round(hours * 2, 6))
    h, m = divmod(halves * 30, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}-{h:02d}:{m:02d}:00"
    return f"{h}:{m:02d}:00"


def right_size(pred_h: float, leg: str) -> Tuple[str, float, str]:
    floor, why = FLOOR_H[leg]
    padded = pred_h * PAD
    reserved = max(padded, floor)
    how = f"x{PAD:g} = {padded:.2f} h" + (f", floored at {floor:g} h ({why})" if floor > padded else "")
    return fmt_time(reserved), reserved, how


@dataclass
class Evidence:
    family: str
    target: str
    rows_used: List[Row]
    rows_raw: List[Row]                # completed but unresolvable cells, or family-halves listed as context
    model: Optional[dict]
    pooled: bool = False               # fell back to every target's rows

    def names(self, limit: int = 6) -> str:
        ns = [f"{r.name}={r.hours:.2f}h" for r in self.rows_used[:limit]]
        more = len(self.rows_used) - limit
        return ", ".join(ns) + (f" (+{more} more)" if more > 0 else "")


class Walltimes:
    """Every leg's evidence and prediction, from the classified sacct rows and the cell resolver."""

    def __init__(self, rows: List[Row], resolver: CellResolver, targets: List[str]):
        self.rows = rows
        self.res = resolver
        self.targets = targets
        self.completed = [r for r in rows if r.state == "COMPLETED"]
        self.notes: List[str] = []

    # -- per family: (target -> points), plus the raw list
    def _points(self, family: str, select) -> Tuple[Dict[str, List[Tuple[int, float, Row]]], Dict[str, List[Row]]]:
        """Resolved points per target, and the completed rows whose VIEW cell count could not be resolved, per target
        (their target key still resolves from the TAG's prefix; only the view's size is unknown)."""
        by: Dict[str, List[Tuple[int, float, Row]]] = {}
        raw: Dict[str, List[Row]] = {}
        for r in self.completed:
            if r.family != family:
                continue
            x = select(r)
            if x is None:
                continue
            n, _how = self.res.n_obj(r.tag)
            kv = split_tag(r.tag, self.res.keys)
            if n is None or not kv:
                raw.setdefault(kv[0] if kv else "?", []).append(r)
                continue
            by.setdefault(kv[0], []).append((x(n), r.hours, r))
        return by, raw

    def _evidence(self, family: str, target: str, n_obj: int, select, x_of_target, unit: str = "object cells") -> Tuple[Optional[float], Evidence, str]:
        by, raw = self._points(family, select)
        pts = [(n, h) for n, h, _ in by.get(target, [])]
        used = [r for _, _, r in by.get(target, [])]
        raw_all = [r for rows in raw.values() for r in rows]
        if pts:
            p, how = _predict(_fit(pts), x_of_target(n_obj), unit)
            return p, Evidence(family, target, used, raw_all, _fit(pts)), how
        if raw.get(target):
            # the target's own rows exist but their views' sizes are unknown: an UNSCALED upper bound (every measured
            # view — main, it, halfdose — carries more cells than any dose view)
            rows = raw[target]
            med = statistics.median(r.hours for r in rows)
            return med, Evidence(family, target, rows, raw_all, {"kind": "raw", "median": med, "n": len(rows)}), (
                f"median of {len(rows)} {target} rows whose view cell counts are UNRESOLVED = {med:.2f} h, an UNSCALED upper bound "
                f"(every measured view has more cells than a dose view; pass --roster / --evidence-roster to scale)")
        pooled = [(n, h) for rows in by.values() for n, h, _ in rows]
        if pooled:
            used = [r for rows in by.values() for _, _, r in rows]
            p, how = _predict(_fit(pooled), x_of_target(n_obj), unit)
            return p, Evidence(family, target, used, raw_all, _fit(pooled), True), f"no {target} row completed in this family; POOLED across targets — " + how
        if raw_all:
            med = statistics.median(r.hours for r in raw_all)
            return med, Evidence(family, target, raw_all, raw_all, {"kind": "raw", "median": med, "n": len(raw_all)}, True), (
                f"no {target} row completed in this family; POOLED median of {len(raw_all)} rows with UNRESOLVED cell counts = {med:.2f} h (unscaled upper bound)")
        return None, Evidence(family, target, [], [], None, True), "NO completed row of this family on either cluster — no measured basis; --time '?' must be set by hand"

    # -- the legs
    def static_base(self, target: str, n_obj: int):
        # a/b family-split jobs (qwen, glm main) are HALF jobs: fold each (tag, trial) pair to its max as one full-job bound
        def trial_of(name: str) -> str:
            return name.split("-sb", 1)[1][:1]
        halves: Dict[Tuple[str, str], List[Row]] = {}
        for r in self.completed:
            if r.family == "sb" and r.half:
                halves.setdefault((r.tag, trial_of(r.name)), []).append(r)
        folded = {k: max(rs, key=lambda r: r.elapsed_s) for k, rs in halves.items() if len(rs) == 2}

        def select(r: Row):
            if r.half:
                return (lambda n: n) if folded.get((r.tag, trial_of(r.name))) is r else None
            return lambda n: n
        p, ev, how = self._evidence("sb", target, n_obj, select, lambda n: n)
        if any(r.half for r in ev.rows_used):
            how += " [a/b family-split halves folded to the slower half per trial: a 4-family job on 2-3 cells launches fewer drivers than one wave of the slow half]"
        return p, ev, how

    def static_esc(self, target: str, n_obj: int):
        return self._evidence("se", target, n_obj, lambda r: (lambda n: n), lambda n: n)

    def bon(self, target: str, n_obj: int):
        # BoN-only jobs: the campaign's `-3` (n >= 5 views) and the halfdose `-3`; `-all` mixes static + BoN (12 h pathology);
        # `3a`/`3b` split cells over two jobs with an unrecorded split
        return self._evidence("stat", target, n_obj, lambda r: (lambda n: n + 1) if r.grp in ("3", "bon") else None, lambda n: n + 1, "cells incl. none")

    def multiturn(self, target: str, n_obj: int):
        # work = cells (incl. none) x attacks: a per-attack job is x1, an `-all` job x3; predict at (n_obj + 1) x 3
        return self._evidence("att", target, n_obj, lambda r: (lambda n: (n + 1) * (3 if r.grp == "all" else 1)), lambda n: (n + 1) * 3, "cell-attacks (cells incl. none x attacks)")

    def capability(self, target: str, n_obj: int):
        return self._evidence("cap", target, n_obj, lambda r: (lambda n: n + 1), lambda n: n + 1, "cells incl. none")

    # -- the attacker pool: member-hours per cell-attack, per measured pool wave
    def pool_waves(self) -> List[dict]:
        atk = sorted([r for r in self.completed if r.family == "atk"], key=lambda r: (r.cluster, r.start))
        waves: List[dict] = []
        for r in atk:   # a wave = members on one cluster whose lifetimes OVERLAP (a member restarted mid-wave joins it)
            for w in waves:
                if w["cluster"] == r.cluster and _hours_between(w["start"], r.start) <= w["end_h"] + 0.01:
                    w["members"].append(r)
                    w["end_h"] = max(w["end_h"], _hours_between(w["start"], r.start) + r.hours)
                    break
            else:
                waves.append({"cluster": r.cluster, "start": r.start, "members": [r], "end_h": r.hours})
        for w in waves:
            w["member_hours"] = sum(m.hours for m in w["members"])
            end_h = w["end_h"]
            atts = [r for r in self.completed if r.family == "att" and r.cluster == w["cluster"]
                    and 0 <= _hours_between(w["start"], r.start) <= end_h]
            ca = 0
            used = []
            for a in atts:
                n, _ = self.res.n_obj(a.tag)
                if n is None:
                    continue
                ca += (n + 1) * (3 if a.grp == "all" else 1)
                used.append(a)
            w["cell_attacks"] = ca
            w["att_rows"] = used
            w["ratio"] = (w["member_hours"] / ca) if ca else None
        return waves

    def pool_ratio(self) -> Tuple[Optional[float], List[dict]]:
        waves = self.pool_waves()
        ratios = [w["ratio"] for w in waves if w["ratio"]]
        return (statistics.median(ratios) if ratios else None), waves

    # -- climb: array-task medians per lane (for the handoff reminder)
    def climb(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for r in self.rows:
            if r.family != "b4k":
                continue
            d = out.setdefault(r.tag, {"completed_h": [], "failed": 0, "failed_on": {}})
            if r.state == "COMPLETED":
                d["completed_h"].append(r.hours)
            elif r.state == "FAILED":
                d["failed"] += 1
                d["failed_on"][r.node] = d["failed_on"].get(r.node, 0) + 1
        return out


def _worst_node(failed_on: Dict[str, int]):
    """(node, count) of the node with the most failures in a {node: failures} map, ties broken by name; None when empty."""
    return min(failed_on.items(), key=lambda kv: (-kv[1], kv[0])) if failed_on else None


def _hours_between(a: str, b: str) -> float:
    try:
        ta = _dt.datetime.fromisoformat(a)
        tb = _dt.datetime.fromisoformat(b)
    except ValueError:
        return float("inf")
    return (tb - ta).total_seconds() / 3600.0


# --------------------------------------------------------------------------------------------------------------- #
# the verdict -> admissible cells per (tag, leg)                                                                     #
# --------------------------------------------------------------------------------------------------------------- #

@dataclass
class CellStatus:
    tag: str
    method: str
    method_label: str
    target: str
    dose: float
    verdict: str
    fired_detail: List[str]
    flagged_by: List[str]
    anchor_exempt: bool
    closure: Optional[str] = None          # "withheld_own_legs" | "must_not_launch"
    closure_why: str = ""
    closed_by: str = ""                    # the stopping rung's tag + detail, for must_not_launch
    anchor_regardless: bool = False
    unevaluable: List[str] = field(default_factory=list)


def index_verdict(rec: dict) -> Dict[Tuple[str, str], CellStatus]:
    cells: Dict[Tuple[str, str], CellStatus] = {}
    for c in rec["cells"]:
        un = []
        for k in ("a", "b", "c"):
            cl = (c.get("clauses") or {}).get(k) or {}
            if cl.get("status") == "unevaluable":
                br = {r: re.sub(r"^unevaluable: ", "", str(v)) for r, v in (cl.get("by_readout") or {}).items()}
                if cl.get("detail"):
                    det = re.sub(r"^unevaluable: ", "", cl["detail"])
                elif len(set(br.values())) == 1:
                    det = f"both readouts: {next(iter(br.values()))}"
                else:
                    det = "; ".join(f"{r}: {v}" for r, v in br.items())
                un.append(f"({k}) {det[:160]}")
        cells[(c["tag"], c["method"])] = CellStatus(c["tag"], c["method"], c.get("method_label", c["method"]), c["target"],
                                                     float(c["dose"]), c["verdict"], list(c.get("fired_detail") or []),
                                                     list(c.get("flagged_by") or []), bool(c.get("anchor_exempt")), unevaluable=un)
    for d in rec["directions"]:
        method = d["method"]
        stop_detail = {}
        for rg in d["rungs"]:
            st = cells.get((rg["tag"], method))
            if st and st.verdict == "stopped":
                stop_detail[rg["dose"]] = f"{rg['tag']}: " + " / ".join(st.fired_detail)
        for w in d["withheld_own_legs"]:
            st = cells.get((w["tag"], method))
            if st:
                st.closure, st.closure_why = "withheld_own_legs", w["why"]
        for m in d["must_not_launch"]:
            st = cells.get((m["tag"], method))
            if st:
                st.closure, st.closure_why = "must_not_launch", m["why"]
                st.closed_by = stop_detail.get(d["closed_at"], "")
        for tag in d.get("anchor_runs_regardless") or []:
            st = cells.get((tag, method))
            if st:
                st.anchor_regardless = True
                st.closed_by = stop_detail.get(d["closed_at"], "")
    return cells


@dataclass
class Decision:
    admit: bool
    reason: str
    note: str = ""       # annotation carried onto the printed line (flagged / anchor / assume-ok)
    assumed: bool = False


def decide(st: CellStatus, leg: str, rerun_existing: bool, assume_ok: bool) -> Decision:
    """The ONE admissibility rule, in the order the omission reasons should read."""
    exists = D33_EXISTS.get(leg, {}).get(st.dose)
    if exists and not rerun_existing:
        extra = " (the 1.0 anchor: exempt from closure by lower rungs, D32(2) — would run under --rerun-existing)" if st.anchor_exempt else ""
        return Decision(False, f"D33: dose {st.dose:+g} not re-run for this leg — {exists}{extra}")
    if st.verdict == "stopped":
        return Decision(False, "its own screen STOPPED it (D8; the anchor's own stop counts too, D42(3)): " + (" / ".join(st.fired_detail) or "(clause detail not recorded)"))
    if st.closure == "must_not_launch" and not st.anchor_exempt:
        return Decision(False, f"beyond the direction's closure — {st.closure_why}; closed by {st.closed_by or '(stopping rung not recorded)'} (D30 / D32(1))")
    if st.verdict == "incomplete":
        why = "; ".join(st.unevaluable) or "the record could not establish every clause"
        if assume_ok:
            return Decision(True, "", note=f"!! ASSUME-OK: verdict INCOMPLETE — not permission; unevaluable: {why}", assumed=True)
        return Decision(False, f"verdict INCOMPLETE is not permission (D32(3)): {why}  [--assume-ok overrides, loudly]")
    if st.verdict not in ("ok", "flagged"):
        return Decision(False, f"verdict {st.verdict!r} is not in the vocabulary this printer admits (ok, flagged)")
    note = ""
    if st.verdict == "flagged":
        note = f"FLAGGED ({', '.join(st.flagged_by)}): runs; every table carrying it must say so (D30)"
    if st.anchor_regardless or (st.anchor_exempt and st.closure == "must_not_launch"):
        note = (note + "; " if note else "") + f"1.0 ANCHOR beyond a closure: runs regardless (D32(2)); closed by {st.closed_by or '(lower rung)'}"
    return Decision(True, "", note=note)


# --------------------------------------------------------------------------------------------------------------- #
# the plan                                                                                                          #
# --------------------------------------------------------------------------------------------------------------- #

@dataclass
class Job:
    name: str
    leg: str
    tag: str
    target: str
    view: str
    cells: List[str]              # object cells served (the anchor `none` is implicit / prepended when restricted)
    restricted: bool
    cluster: str
    script: str
    args: str
    time: str
    est_h: float
    reserved_h: float
    evidence: str
    block: str                    # "pool" | "first" | "second"
    notes: List[str] = field(default_factory=list)
    after: str = ""


@dataclass
class Omission:
    leg: str
    tag: str
    method: str
    dose: float
    reason: str


def _cells_arg(view_cells: List[str], admitted: List[str]) -> Tuple[str, bool]:
    if set(admitted) == set(view_cells):
        return "", False
    return "none," + ",".join(c for c in view_cells if c in admitted), True


def build_plan(rec: dict, args: argparse.Namespace, wall: Walltimes, cluster_of: Dict[str, str]) -> dict:
    views = sorted(rec["views"], key=lambda v: (v["target"], v["dose"]))
    status = index_verdict(rec)
    jobs: List[Job] = []
    omissions: List[Omission] = []
    gsm8k: List[dict] = []
    admitted_by_leg: Dict[str, Dict[str, List[str]]] = {leg: {} for leg in LEGS}
    for v in views:
        tag, target, view = v["tag"], v["target"], v["view"]
        for leg in LEGS:
            ok_cells: List[str] = []
            notes: List[str] = []
            for cell in v["cells"]:
                st = status.get((tag, cell))
                if st is None:
                    omissions.append(Omission(leg, tag, cell, v["dose"], "no verdict cell for this (tag, method) — the verdict and its views disagree; re-run screen_verdict.py"))
                    continue
                d = decide(st, leg, args.rerun_existing, args.assume_ok)
                if d.admit:
                    ok_cells.append(cell)
                    if d.note:
                        notes.append(f"{st.method_label}: {d.note}")
                else:
                    omissions.append(Omission(leg, tag, cell, v["dose"], d.reason))
            if not ok_cells:
                continue
            admitted_by_leg[leg][tag] = ok_cells
            cells_arg, restricted = _cells_arg(v["cells"], ok_cells)
            n_obj = len(ok_cells)
            c = cluster_of.get(tag, DEFAULT_CLUSTER)
            if leg == "gsm8k":
                gsm8k.append({"tag": tag, "target": target, "view": view, "cells": ok_cells, "notes": notes,
                              "cluster": "cluster_a" if target in FUSED_KERNEL_CLUSTER_A_ONLY else c})
                continue
            if leg == "static":
                p, ev, how = wall.static_base(target, n_obj)
                t, res_h, sized = right_size(p or 0.0, "static_base") if p is not None else ("?", 0.0, "NO EVIDENCE")
                for T in range(N_TRIALS):
                    jobs.append(Job(f"f22s-{tag}-sb{T}", leg, tag, target, view, ok_cells, restricted, c,
                                    f"{SS}/target_sampled.sbatch", f'{target} static_base_t{T} W {args.nshards} "{cells_arg}" {view}',
                                    t, p or 0.0, res_h, f"{how}; --time {sized}; rows: {ev.names()}", "first", list(notes)))
                pe, eve, howe = wall.static_esc(target, n_obj)
                te, res_e, sized_e = right_size(pe or 0.0, "static_esc") if pe is not None else ("?", 0.0, "NO EVIDENCE")
                for T in range(N_TRIALS):
                    jobs.append(Job(f"f22s-{tag}-se{T}", leg, tag, target, view, ok_cells, restricted, c,
                                    f"{SS}/target_sampled.sbatch", f'{target} static_esc_t{T} W {args.nshards} "{cells_arg}" {view}',
                                    te, pe or 0.0, res_e, f"{howe}; --time {sized_e}; rows: {eve.names()}", "second",
                                    list(notes), after=f"f22s-{tag}-sb{T} COMPLETED"))
            elif leg == "bon":
                p, ev, how = wall.bon(target, n_obj)
                t, res_h, sized = right_size(p or 0.0, "bon") if p is not None else ("?", 0.0, "NO EVIDENCE")
                extra = f" {cells_arg}" if restricted else ""
                jobs.append(Job(f"f22-stat-{tag}-bon", leg, tag, target, view, ok_cells, restricted, c,
                                f"{S}/target_static.sbatch", f"{target} {view} bon W {args.nshards}{extra}",
                                t, p or 0.0, res_h, f"{how}; --time {sized}; rows: {ev.names()}", "first",
                                list(notes) + ["bon.max_new_tokens is 512 in the dose configs (D33 revert f38d055d) — the store Scale's climb accepts"]))
            elif leg == "multiturn":
                p, ev, how = wall.multiturn(target, n_obj)
                t, res_h, sized = right_size(p or 0.0, "multiturn") if p is not None else ("?", 0.0, "NO EVIDENCE")
                extra = f" {cells_arg}" if restricted else ""
                jobs.append(Job(f"f22-att-{tag}-all", leg, tag, target, view, ok_cells, restricted, c,
                                f"{S}/target_attacks.sbatch", f"{target} {view} {','.join(ATTACKS)} W {args.nshards} {SHARDS}{extra}",
                                t, p or 0.0, res_h, f"{how}; --time {sized}; rows: {ev.names()}", "first", list(notes),
                                after=">= 1 attacker record LIVE in $WAVE_ROOT/attackers/ on this cluster"))
            elif leg == "capability":
                p, ev, how = wall.capability(target, n_obj)
                t, res_h, sized = right_size(p or 0.0, "capability") if p is not None else ("?", 0.0, "NO EVIDENCE")
                jobs.append(Job(f"f22s-{tag}-cap0", leg, tag, target, view, ok_cells, restricted, c,
                                f"{SS}/target_sampled.sbatch", f'{target} capability_t0 W {args.nshards} "{cells_arg}" {view}',
                                t, p or 0.0, res_h, f"{how}; --time {sized}; rows: {ev.names()}", "first",
                                list(notes) + [CAPABILITY_WARNING, "ONE trial: MMLU is a letter-logprob argmax, sampler-moot (D33 / S-D7)"]))
    # the attacker pool, sized from the admitted attack work
    att_jobs = [j for j in jobs if j.leg == "multiturn"]
    pool = plan_pool(att_jobs, args, wall)
    if pool.get("refused"):   # no pool -> no attack job is printed; the leg's admissions stay recorded for the reader
        jobs = [j for j in jobs if j.leg != "multiturn"]
        for j in att_jobs:
            omissions.append(Omission("multiturn", j.tag, ",".join(j.cells), 0.0, "REFUSED: " + pool["refused"]))
    return {"jobs": jobs, "omissions": omissions, "gsm8k": gsm8k, "pool": pool, "admitted": admitted_by_leg, "status": status}


def plan_pool(att_jobs: List[Job], args: argparse.Namespace, wall: Walltimes) -> dict:
    ratio, waves = wall.pool_ratio()
    cell_attacks = sum((len(j.cells) + 1) * len(ATTACKS) for j in att_jobs)
    by_cluster: Dict[str, int] = {}
    for j in att_jobs:
        by_cluster[j.cluster] = by_cluster.get(j.cluster, 0) + (len(j.cells) + 1) * len(ATTACKS)
    out: Dict[str, Any] = {"ratio": ratio, "waves": waves, "cell_attacks": cell_attacks, "by_cluster": by_cluster,
                           "members": {}, "refused": None, "live_pool": bool(args.live_pool), "n_att_jobs": len(att_jobs)}
    if not att_jobs:
        return out
    if args.attackers == 0 and not args.live_pool:
        out["refused"] = ("multi-turn jobs need the shared attacker pool: pass --attackers N (members this lane launches under "
                          f"{POOL_JOB_PREFIX}-<cluster>-<n> names) or --live-pool (you verified >= 1 LIVE record under $WAVE_ROOT/attackers/ "
                          "on that cluster and accept depending on a peer's members)")
        return out
    for cl, ca in by_cluster.items():
        need_h = (ratio or 1.2) * ca
        if args.attackers is not None and args.attackers > 0:
            n = args.attackers
            how = f"--attackers {n}"
        else:
            n = min(POOL_MAX, max(POOL_MIN, math.ceil(need_h / POOL_TARGET_WALL_H)))
            how = f"ceil({need_h:.0f} member-h / {POOL_TARGET_WALL_H:g} h target pool life) clamped to [{POOL_MIN}, {POOL_MAX}]"
        wall_h = need_h / n
        t, reserved, sized = right_size(wall_h, "attacker_pool")
        reserved = min(reserved, 24.0)
        t = fmt_time(reserved) if reserved < 24.0 else "1-00:00:00"
        out["members"][cl] = {"n": 0 if args.live_pool else n, "how": how, "need_h": need_h, "wall_h": wall_h, "time": t,
                              "reserved_h": reserved, "sized": sized, "cell_attacks": ca,
                              "names": [] if args.live_pool else [f"{POOL_JOB_PREFIX}-{POOL_CLUSTER_SHORT[cl]}-{i + 1}" for i in range(n)]}
    return out


# --------------------------------------------------------------------------------------------------------------- #
# rendering                                                                                                         #
# --------------------------------------------------------------------------------------------------------------- #

def sbatch_line(job: Job, sha: str) -> str:
    return submit_command(job.cluster, sha, job.time, job.name, f"{job.script} {job.args.replace(' W ', f' {sha} ')}")


def pool_line(cluster: str, name: str, time: str, sha: str) -> str:
    return submit_command(cluster, sha, time, name, f"{S}/attacker_pool.sbatch {sha}")


def submit_command(cluster: str, sha: str, time: str, name: str, script_and_args: str) -> str:
    """One sbatch line, every site value from the cluster's block: ssh CLUSTER_SSH, cd CLUSTER_EVAL_REPO_ROOT/<sha>, then
    --qos / --partition / --account / --output (CLUSTER_LOG_DIR) / --time / CLUSTER_SBATCH_EXTRA / -J."""
    b = site_block(cluster)
    flags = [f"--qos={b['CLUSTER_QOS']}", f"--partition={b['CLUSTER_PARTITION']}"]
    if b.get("CLUSTER_ACCOUNT"):
        flags.append(f"--account={b['CLUSTER_ACCOUNT']}")
    flags += [f"--output={b['CLUSTER_LOG_DIR']}/%x_%j.out", f"--time={time}"]
    if (b.get("CLUSTER_SBATCH_EXTRA") or "").strip():
        flags.append(b["CLUSTER_SBATCH_EXTRA"].strip())
    return f"ssh {b['CLUSTER_SSH']} 'cd {b['CLUSTER_EVAL_REPO_ROOT']}/{sha} && sbatch {' '.join(flags)} -J {name} {script_and_args}'"


def release_commands(cluster: str, member_names: List[str], att_names: List[str]) -> List[str]:
    """The release, as a guarded shell snippet: refuses while ANY attack job is live on the cluster; touches ONLY our
    members' <jobid>.STOP files, ids resolved from OUR names. Never the pool-wide STOP. Host and root are the block's."""
    b = site_block(cluster)
    host, root = b["CLUSTER_SSH"], f"{b['CLUSTER_SHARED_ROOT']}/exp/sdfbase_full22"
    names = ",".join(member_names)
    ours = " ".join(att_names)
    return [
        f"# release on {cluster} — run ONLY when both checks print nothing:",
        f"ssh {host} 'squeue -u $USER -h -o \"%i %j %T\" | grep -E -- \"-att-\"'      # (1) NO attack job of ANY lane live (ours: {ours or 'none planned'})",
        f"ssh {host} 'for n in {ours or 'NONE'}; do squeue -u $USER -h -n $n -o %i; done'   # (2) our own attack jobs all ENDED",
        f"ssh {host} 'for j in $(squeue -u $USER -h -n {names or 'NONE'} -o %i); do touch {root}/attackers/$j.STOP && echo released $j; done'"
        f"   # our members only, by OUR names; NEVER touch {root}/attackers/STOP (pool-wide)",
    ]


def _age(written_at: Optional[str]) -> str:
    try:
        t = _dt.datetime.fromisoformat(written_at.replace("Z", "+00:00"))
        h = (_dt.datetime.now(_dt.timezone.utc) - t).total_seconds() / 3600
        return f"{h:.1f} h old"
    except Exception:
        return "age unknown"


def used_budget_from_verdict(rec: dict, cluster: str) -> Tuple[Optional[int], str]:
    """The account's active jobs on the verdict's host at gather time (a lower bound: an array row counts its listed tasks
    when the list is intact, else 1)."""
    src = rec.get("source") or {}
    if src.get("host") is None or (src.get("host") == "cluster_b") != (cluster == "cluster_b"):
        return None, "no squeue snapshot for this cluster in the verdict"
    rows = ((src.get("jobs_probe") or {}).get("squeue")) or []
    n = 0
    for ln in rows:
        f = ln.split("|")
        if len(f) < 3 or f[2] not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
            continue
        m = re.match(r"^\d+_\[(?P<spec>[^\]]*)\]$", f[0])
        if m:
            spec = m.group("spec").split("%")[0]
            k = 0
            complete = True
            for part in spec.split(","):
                if not part:
                    complete = False
                    continue
                if "-" in part:
                    a, _, b = part.partition("-")
                    if a.isdigit() and b.isdigit():
                        k += int(b) - int(a) + 1
                    else:
                        complete = False
                elif part.isdigit():
                    k += 1
                else:
                    complete = False
            n += max(1, k) if complete else max(1, k)
        else:
            n += 1
    return n, f">= {n} active rows in the verdict's squeue snapshot ({src.get('gathered_at')}; array rows counted by listed tasks, a lower bound)"


def render(rec: dict, plan: dict, args: argparse.Namespace, sha: str, wall: Walltimes, checks: List[str],
           budgets: Dict[str, Tuple[int, str]], cluster_of: Dict[str, str], evidence_meta: str) -> List[str]:
    L: List[str] = []
    P = L.append
    views = rec["views"]
    summ = rec.get("summary") or {}
    P(f"# STRENGTH FRONTIER — EXPENSIVE LEGS @ {sha}: verdict {args.verdict} (written {rec.get('written_at')}, {_age(rec.get('written_at'))}; "
      f"tool {((rec.get('tool') or {}).get('git_sha') or '?')[:12]}; source {(rec.get('source') or {}).get('mode')} {(rec.get('source') or {}).get('host')})")
    P(f"# {len(views)} dose views from roster {(rec.get('roster') or {}).get('path')} sha256 {((rec.get('roster') or {}).get('sha256') or '?')[:16]}…; "
      f"{len(rec['cells'])} (target, method, dose) cells: {summ.get('stopped', '?')} stopped / {summ.get('flagged', '?')} flagged / "
      f"{summ.get('ok', '?')} ok / {summ.get('incomplete', '?')} incomplete")
    for ck in checks:
        P("# " + ck)
    if args.rerun_existing:
        P("# !!!! --rerun-existing: D33's reuse condition ('serving code identical') is treated as FAILED — the doses that exist elsewhere")
        P("# !!!! (0.5, 1.0) are RE-RUN on every leg, and the 1.0 anchor prints regardless of closure (D32(2)). Say so in every table.")
    if args.assume_ok:
        P("# !!!! --assume-ok: cells whose verdict is INCOMPLETE are printed AS IF ok. That is NOT permission (D32(3)); every such line")
        P("# !!!! carries an ASSUME-OK marker. Use only when the screen's numbers are known from elsewhere and the record is stale.")
    P(f"# {D33_TEXT}")
    P("# per-leg dose sets after D33 (the verdict then withholds within them):")
    for leg in LEGS:
        doses = sorted({v["dose"] for v in views if not (D33_EXISTS.get(leg, {}).get(v["dose"]) and not args.rerun_existing)})
        excl = ", ".join(f"{d:+g}: {w}" for d, w in sorted(D33_EXISTS.get(leg, {}).items())) if not args.rerun_existing else "none (--rerun-existing)"
        P(f"#   {leg:<11} runs {', '.join(f'{d:+g}' for d in doses)}   | not re-run — {excl}")
    P(f"# routing: every dose TAG -> {DEFAULT_CLUSTER}"
      + (f"; overrides: {', '.join(f'{t}={c}' for t, c in cluster_of.items() if c != DEFAULT_CLUSTER)}" if any(c != DEFAULT_CLUSTER for c in cluster_of.values()) else "")
      + "; site cluster blocks: "
      + ", ".join(f"{cl} (ssh {b['CLUSTER_SSH']}, --qos={b['CLUSTER_QOS']}, budget {b['CLUSTER_HIGH_BUDGET_GPUS']}, "
                  f"CLUSTER_SBATCH_EXTRA {(b.get('CLUSTER_SBATCH_EXTRA') or '').strip() or 'none'})"
                  for cl, b in ((cl, site_block(cl)) for cl in sorted(set(cluster_of.values())))))
    P(f"# walltime evidence: {evidence_meta}; {len(wall.completed)} completed rows in the matched families; "
      f"--time = prediction x{PAD:g} rounded up to 30 min, per-leg floors: " + ", ".join(f"{k} {v[0]:g} h" for k, v in FLOOR_H.items()))
    P("#   (measured views' cell counts are read from the CURRENT rosters given, not the roster each job ran under — the `it` view grew from 2 to 3 objects on 2026-09-08)")
    P("")
    # ---- omissions
    om = plan["omissions"]
    P(f"# OMISSIONS — {len(om)} (leg, tag, method) cells NOT printed, each with the clause that caused it:")
    by_reason: Dict[str, List[Omission]] = {}
    for o in om:
        by_reason.setdefault(o.reason, []).append(o)
    for reason, os_ in sorted(by_reason.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        who = ", ".join(sorted({f"{o.leg}:{o.tag}/{o.method.replace('constv2-mix-full__lora_r4_ce', 'lora').replace('lora_minus_ctl', 'lora-ctl')}" for o in os_}))
        P(f"#   [{len(os_):>3}] {reason}")
        P(f"#         {who[:900]}{'…' if len(who) > 900 else ''}")
    P("")
    # ---- table
    jobs: List[Job] = plan["jobs"]
    P("| cluster | job | leg | cells (object) | est h | --time | block | args |")
    P("|---|---|---|---|---|---|---|---|")
    for j in jobs:
        P(f"| {j.cluster} | {j.name} | {j.leg} | {len(j.cells)}{' RESTRICTED' if j.restricted else ''} | {j.est_h:.2f} | {j.time} | "
          f"{j.block}{(' (after ' + j.after + ')') if j.after else ''} | {j.args.replace(' W ', ' ' + sha[:12] + ' ')} |")
    pool = plan["pool"]
    for cl, m in (pool.get("members") or {}).items():
        for nm in m["names"]:
            P(f"| {cl} | {nm} | attacker pool | — | {m['wall_h']:.1f} | {m['time']} | pool (before any f22-att) | {sha[:12]} |")
    P("")
    # ---- GPU-hours per leg and budget
    P("# GPU-HOURS (1 GPU per job): expected = sum of predictions; reserved = sum of --time (what the budget holds while jobs pend/run)")
    tot_e = tot_r = 0.0
    for leg in ("static", "bon", "multiturn", "capability"):
        js = [j for j in jobs if j.leg == leg]
        if not js:
            P(f"#   {leg:<11} 0 jobs")
            continue
        e = sum(j.est_h for j in js)
        r = sum(j.reserved_h for j in js)
        tot_e += e
        tot_r += r
        P(f"#   {leg:<11} {len(js):>3} jobs  expected {e:7.1f} GPU-h  reserved {r:7.1f} GPU-h  ({', '.join(sorted({j.cluster for j in js}))})")
    for cl, m in (pool.get("members") or {}).items():
        if m["n"]:
            e = m["need_h"]
            r = m["n"] * m["reserved_h"]
            tot_e += e
            tot_r += r
            P(f"#   pool {cl:<6} {m['n']:>3} members expected {e:7.1f} member-h  reserved {r:7.1f} GPU-h  ({m['how']}; {pool['cell_attacks']} cell-attacks at "
              f"{(pool['ratio'] or 1.2):.2f} member-h each{' — ratio NOT measured, 1.2 assumed' if pool['ratio'] is None else ''})")
    P(f"#   TOTAL      expected {tot_e:.0f} GPU-h, reserved {tot_r:.0f} GPU-h  (D31's full-battery figure for this LoRA lane was ~328 GPU-h at the 0.6 h overhead column, BEFORE D33's reuse)")
    for cl in CLUSTERS:
        here = [j for j in jobs if j.cluster == cl and j.block == "first"]
        n_pool = (pool.get("members") or {}).get(cl, {}).get("n", 0)
        if not here and not n_pool:
            continue
        b = site_block(cl)
        cap = int(b["CLUSTER_HIGH_BUDGET_GPUS"])
        free, how = budgets[cl]
        P(f"#   budget {cl}: {len(here)} first-block jobs + {n_pool} pool members = {len(here) + n_pool} GPUs of the {cap}-GPU `{b['CLUSTER_QOS']}` budget; free now {free} ({how})"
          + (f"  ** {len(here) + n_pool - free} HELD **" if len(here) + n_pool > free else ""))
    P("")
    # ---- walltime evidence per leg/target
    P("# WALLTIME EVIDENCE per (leg, target) — the rows behind every --time above:")
    seen = set()
    for j in jobs:
        k = (j.leg, j.target, j.name.split("-")[-1][:2] if j.leg == "static" else "")
        if k in seen:
            continue
        seen.add(k)
        P(f"#   {j.leg}/{j.target}{'/' + ('esc' if 'se' in k[2] else 'base') if j.leg == 'static' else ''}: {j.evidence}")
    if pool.get("waves"):
        P("#   attacker pool member-hours per cell-attack, by measured wave (members' summed elapsed / sum over that wave's attack jobs of cells x attacks):")
        for w in pool["waves"]:
            if not w["cell_attacks"]:
                continue
            P(f"#     {w['cluster']} {w['start']}: {len(w['members'])} members x median {statistics.median([m.hours for m in w['members']]):.1f} h = {w['member_hours']:.0f} member-h; "
              f"{w['cell_attacks']} cell-attacks over {len(w['att_rows'])} resolvable attack jobs -> {('%.2f' % w['ratio']) if w['ratio'] else '—'} member-h each")
        P(f"#     median ratio {('%.2f' % pool['ratio']) if pool.get('ratio') else 'UNMEASURED'} member-h per cell-attack")
    P("")
    # ---- 0. pool
    att_jobs = [j for j in jobs if j.leg == "multiturn"]
    P("# --- 0. THE SHARED ATTACKER POOL — OUR members, submitted BEFORE any f22-att job (a driver refuses with no live record):")
    for rule in POOL_RULES:
        P(f"#     * {rule}")
    if pool.get("refused"):
        P(f"# REFUSED multi-turn: {pool['refused']}")
    elif not att_jobs:
        P("#     (no admissible multi-turn cell -> no pool member is needed; nothing to release)")
    elif pool.get("live_pool"):
        P("#     --live-pool: you attested a LIVE pool on the cluster — no member is printed. Our attack jobs will ride PEERS' members;")
        P("#     do NOT release anything you did not launch, and expect their release to strand our drivers (f22_fatal 'no live attacker').")
    else:
        for cl, m in pool["members"].items():
            P(f"#     {cl}: {m['n']} members ({m['how']}); expected pool life {m['wall_h']:.1f} h for {m['cell_attacks']} cell-attacks; --time {m['time']} ({m['sized']})")
            for nm in m["names"]:
                P(pool_line(cl, nm, m["time"], sha))
        P("#     wait for each member's record: ls $WAVE_ROOT/attackers/<jobid>.json (WAVE_ROOT = <root>/$USER/exp/sdfbase_full22), then the f22-att lines below")
    P("")
    # ---- 1. first block
    P("# --- 1. FIRST BLOCK — static base (both trials), BoN-512, MMLU (capability_t0), then multi-turn once >= 1 attacker record is LIVE.")
    P("#        Every job stands up its own server on ONE GPU. Lines beyond the free budget are `# HOLD` (a PENDING job counts against the cap).")
    for cl in CLUSTERS:
        here = [j for j in jobs if j.cluster == cl and j.block == "first"]
        if not here:
            continue
        free, _ = budgets[cl]
        n_pool = (pool.get("members") or {}).get(cl, {}).get("n", 0)
        slots = max(0, free - n_pool)
        order = [j for j in here if j.leg != "multiturn"] + [j for j in here if j.leg == "multiturn"]
        for i, j in enumerate(order):
            for n in j.notes:
                P(f"#   [{j.name}] {n}")
            if j.restricted:
                P(f"#   [{j.name}] cells RESTRICTED to none + {', '.join(j.cells)} (the view's other cell(s) are withheld above)")
            line = sbatch_line(j, sha)
            if i >= slots:
                line = f"# HOLD (over the free {cl} budget; submit when a job above completes): " + line
            if j.leg == "multiturn":
                line += "   # after >= 1 attacker record is LIVE on this cluster"
            P(line)
    P("")
    # ---- 2. second block
    P("# --- 2. SECOND BLOCK — the static 8192 RUNG per trial (static_esc_t<T>: only (family, cell) pairs with at-cap rows are re-drawn;")
    P("#        zero pairs exits 0 without a standup). Submit BY HAND once that TAG's sb<T> is COMPLETED. No --dependency on purpose (the budget).")
    for j in jobs:
        if j.block == "second":
            for n in j.notes:
                P(f"#   [{j.name}] {n}")
            if j.restricted:
                P(f"#   [{j.name}] cells RESTRICTED to none + {', '.join(j.cells)}")
            P(sbatch_line(j, sha) + f"   # after {j.after}")
    P("")
    # ---- 3. release
    P("# --- 3. RELEASE OUR ATTACKER POOL MEMBERS — only after EVERY one of our f22-att jobs has ENDED and no other attack job is live:")
    if pool.get("members") and any(m["n"] for m in pool["members"].values()):
        for cl, m in pool["members"].items():
            if not m["n"]:
                continue
            for ln in release_commands(cl, m["names"], [j.name for j in att_jobs if j.cluster == cl]):
                P(ln)
    else:
        P("#     (no members of ours to release)")
    P("")
    # ---- 4. handoffs
    P("# --- 4. HANDOFFS (not submitted here):")
    P("#   (a) BoN 512 -> 4,096 CLIMB is Scale's tooling (D33/D34): once each f22-stat-<TAG>-bon store exists, REGISTER the lane rows — one row per TAG in")
    P("#       bon_cap4096/cluster/manifests/lanes.json {key, view, sha, nshards, time, cluster, roots (a LIST of store roots), label}; classify with the")
    P("#       PINNED classifier_src/ in Scale's b4k dir (never this tree's copy); loop = the repetition rule alone; re-drawn samples are per-goal, not")
    P("#       continuations. Coordinate the rows with Scale — never copy their scripts (per-lane copies propagate defects).")
    climb = wall.climb()
    if climb:
        P("#       measured b4k array tasks (21 days): " + "; ".join(
            f"{tag}: {len(d['completed_h'])} done" + (f", median {statistics.median(d['completed_h']):.2f} h" if d["completed_h"] else "")
            + (f", {d['failed']} failed ({_worst_node(d['failed_on'])[1]} on {_worst_node(d['failed_on'])[0]})" if d['failed'] else "")
            for tag, d in sorted(climb.items()) if d["completed_h"] or d["failed"]))
    worst = _worst_node({n: sum(d["failed_on"].get(n, 0) for d in climb.values()) for d in climb.values() for n in d["failed_on"]})
    P("#       => GLM lanes: nshards 8 (~1.5 h per task; 30.7% of climbed draws still at cap at 4,096), gemma / qwen / nemotron: 4"
      + (f"; exclude {worst[0]}, the node the failures above ran on." if worst else "."))
    bon_tags = sorted(plan["admitted"]["bon"])
    d0p5 = sorted(v["tag"] for v in views if v["dose"] == 0.5)
    P(f"#       rows to hand Scale when the stores land: {', '.join(bon_tags) or '(none admissible yet)'}; PLUS the d0p5 climb reading the EXISTING halfdose 512 stores "
      f"({', '.join(d0p5)}) written BESIDE them, never into them (D34).")
    P("#   (b) GSM8K is the evaluation lane's (its generalised scripts/sdfbase/math500 launcher; TAG=<key>_<view>; str- job names on this lane's budget and wave root). Requirements handed over:")
    for r_ in GSM8K_REQUIREMENTS:
        P(f"#       - {r_}")
    if plan["gsm8k"]:
        for g in plan["gsm8k"]:
            P(f"#       GSM8K PLACEHOLDER {g['tag']:<26} cluster={g['cluster'].upper() if g['cluster'] == 'cluster_a' else g['cluster']:<10} cells={','.join(g['cells'])}"
              f"  -> <sibling-lane launcher> --tag {g['tag']} --max-tokens-by-task gsm8k_paired200=8192 --thinking on --sha {sha[:12]}"
              + (f"   # {' | '.join(g['notes'])}" if g["notes"] else ""))
    else:
        P("#       (no admissible GSM8K cell under this verdict)")
    P("")
    P("# --- watchers (arm each in the SAME turn as its submission; a job is not launched until it has survived ~2 min):")
    for cl in sorted({j.cluster for j in jobs} | {cl for cl, m in (pool.get('members') or {}).items() if m['n']}):
        P(f"#    ~/scripts/watch-job.sh --slurm <jobid> --host {site_block(cl)['CLUSTER_SSH']}   # every {cl} job above")
    P("# --- stopping: the wave-root STOP ($SHARED/exp/sdfbase_full22/STOP) is WAVE-WIDE and stops every full22 job of every lane on that cluster —")
    P("#     never for this. Per job: touch its serving dir's STOP, then `scancel -u $USER --name=<exact job name>`; the screen (f22s-<TAG>-ctl0) is never")
    P("#     cancelled on a verdict (D32(1)). The verdict's own cancel list (screen_verdict.json `cancel.safeguard`) matches every leg name")
    P("#     printed here, BOTH trials included (screen_verdict.leg_job_patterns: sb[01] / se[01] / cap[01] / le[01], D49).")
    return L


# --------------------------------------------------------------------------------------------------------------- #
# main                                                                                                              #
# --------------------------------------------------------------------------------------------------------------- #

def _verdict_checkout_sha(rec: dict) -> Optional[str]:
    """The commit of the checkout the verdict's helper was loaded from: the 40-hex directory right above that checkout's
    ``scripts/`` tree. Matched by position, not by the root's name, so any site's eval-repo root (SISTEER_EVAL_REPO_ROOT)
    is read, as are the recorded verdicts' paths."""
    p = ((rec.get("source") or {}).get("helper") or {}).get("path") or ""
    m = re.search(r"/([0-9a-f]{40})/scripts/", p)
    return m.group(1) if m else None


def run_checks(args: argparse.Namespace, rec: dict, sha: str) -> Tuple[List[str], List[Tuple[str, dict]]]:
    """Offline prechecks (the model printer's, where they apply here) + roster resolution. Returns (CHECK lines, rosters)."""
    checks: List[str] = []
    rosters: List[Tuple[str, dict]] = []
    vsha = _verdict_checkout_sha(rec)
    if vsha and vsha != sha:
        checks.append(f"CHECK wave-sha WARNING: the verdict read its stores through checkout {vsha[:12]} but you are submitting at {sha[:12]} — "
                      "the stores compared are another sha's; fine only if the two differ outside the served path")
    else:
        checks.append(f"CHECK wave-sha OK: {'verdict checkout == ' + sha[:12] if vsha else 'verdict names no checkout; ' + sha[:12] + ' unverified'}")
    vroster = rec.get("roster") or {}
    if args.roster:
        p = Path(args.roster)
        if not p.is_file():
            raise Refusal(f"--roster {p}: not a file")
        got = sha256_file(p)
        if vroster.get("sha256") and got != vroster["sha256"]:
            raise Refusal(f"--roster {p} sha256 {got[:12]} != the verdict's roster {vroster['sha256'][:12]} — a verdict is read on ITS roster; re-run screen_verdict.py")
        r = load_roster(p)
        tags = {sv.tag_of(k, vn) for k, t in (r.get("targets") or {}).items() for vn, v in ((t or {}).get("views") or {}).items()
                if isinstance(v, dict) and "dose" in v}
        vt = {v["tag"] for v in rec["views"]}
        if tags != vt:
            raise Refusal(f"--roster dose views {sorted(tags ^ vt)} differ from the verdict's — re-run screen_verdict.py on this roster")
        rosters.append((f"--roster {p}", r))
        checks.append(f"CHECK roster OK: {p} sha256 {got[:12]} == the verdict's; {len(tags)} dose views agree")
    else:
        vp = Path(vroster.get("path") or "")
        if vp.is_file() and vroster.get("sha256") and sha256_file(vp) == vroster["sha256"]:
            rosters.append((f"verdict roster {vp}", load_roster(vp)))
            checks.append(f"CHECK roster OK: the verdict's roster {vp} is present locally and byte-identical (sha256 {vroster['sha256'][:12]})")
        else:
            checks.append("CHECK roster UNVERIFIED: no --roster and the verdict's roster path is absent or changed locally — dose views are read "
                          "from the verdict record; evidence rows on non-dose views resolve only through --evidence-roster")
    for ep in args.evidence_roster or []:
        p = Path(ep)
        if not p.is_file():
            raise Refusal(f"--evidence-roster {p}: not a file")
        rosters.append((f"--evidence-roster {p.parent.parent.name}/{p.name}", load_roster(p)))
    if args.wave_checkout:
        wc = Path(args.wave_checkout)
        try:
            head = subprocess.check_output(["git", "-C", str(wc), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception as e:
            raise Refusal(f"--wave-checkout {wc}: git rev-parse failed ({e})")
        if head != sha:
            raise Refusal(f"--wave-checkout {wc} is at {head[:12]}, not the wave sha {sha[:12]}")
        checks.append(f"CHECK checkout OK: {wc} HEAD == {sha[:12]}")
        rp = wc / "configs" / "sdfbase_full22_roster.yaml"
        if rp.is_file() and vroster.get("sha256"):
            got = sha256_file(rp)
            if got != vroster["sha256"]:
                raise Refusal(f"--wave-checkout roster {rp} sha256 {got[:12]} != the verdict's {vroster['sha256'][:12]} — the jobs would serve a roster the verdict never read")
            checks.append(f"CHECK checkout-roster OK: {rp.name} sha256 == the verdict's")
        for label, rel in (("configs", "scripts/sdfbase/make_full22_configs.py"), ("sampled-configs", "scripts/sdfbase/full22/sampled/make_sampled_configs.py")):
            script = wc / rel
            if not script.is_file():
                raise Refusal(f"--wave-checkout {wc}: {rel} is missing — not a wave checkout")
            gen = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, cwd=str(wc))
            if gen.returncode != 0:
                raise Refusal(f"{label} drift in {wc}: `{rel} --check` rc={gen.returncode}: {(gen.stdout + gen.stderr).strip()[-600:]} — regenerate at the wave sha first")
            checks.append(f"CHECK {label} OK: {rel} --check passed in {wc}")
    else:
        checks.append("CHECK configs UNVERIFIED (no --wave-checkout): make_full22_configs.py --check / make_sampled_configs.py --check NOT run; "
                      "the sbatch lines run whatever the cluster checkout at the sha carries")
    return checks, rosters


def load_evidence(args: argparse.Namespace) -> Tuple[List[Row], str]:
    if args.sacct_file:
        raw = []
        for spec in args.sacct_file:
            cl, sep, path = spec.partition("=")
            if not sep or cl not in CLUSTERS:
                raise Refusal(f"--sacct-file {spec!r}: expected CLUSTER=PATH with CLUSTER in {sorted(CLUSTERS)}")
            raw += parse_sacct_psv(cl, Path(path).read_text(encoding="utf-8"))
        return classify_rows(raw), f"--sacct-file {', '.join(args.sacct_file)} ({SACCT_CMD})"
    if args.sacct == "live":
        hosts = {cl: site_block(cl)["CLUSTER_SSH"] for cl in CLUSTERS}
        return classify_rows(sacct_live(hosts)), f"LIVE `{SACCT_CMD}` over ssh on {', '.join(hosts.values())} at {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='minutes')}"
    return classify_rows(SNAPSHOT_ROWS), f"EMBEDDED snapshot queried {SNAPSHOT_QUERIED} (`{SACCT_CMD}` on both clusters; --sacct live refreshes)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verdict", help="screen_verdict.json written by screen_verdict.py")
    ap.add_argument("--wave-sha", required=True, help="the 40-hex wave sha the cluster checkouts carry (run record `wave_sha`)")
    ap.add_argument("--roster", default=None, help="cross-check roster (must be byte-identical to the verdict's)")
    ap.add_argument("--evidence-roster", action="append", default=[], help="extra roster(s) resolving RETIRED views' cell counts for the walltime evidence (e.g. the halfdose worktree's)")
    ap.add_argument("--wave-checkout", default=None, help="a local checkout at the wave sha: runs the generators' --check (drift refuses)")
    ap.add_argument("--sacct", choices=("embedded", "live"), default="embedded", help="walltime evidence source (default: the embedded snapshot)")
    ap.add_argument("--sacct-file", action="append", default=[], metavar="CLUSTER=PATH", help=f"saved `{SACCT_CMD}` output per cluster")
    ap.add_argument("--attackers", type=int, default=None, help="pool members to launch per cluster (default: sized from the admitted attack work; 0 = none, needs --live-pool)")
    ap.add_argument("--live-pool", action="store_true", help="a LIVE attacker pool exists on the cluster (you checked); print attack jobs without members")
    ap.add_argument("--cluster", action="append", default=[], metavar="TAG=CLUSTER")
    ap.add_argument("--cluster-b-budget", type=int, default=None, help="free GPUs at that cluster's non-preemptible QoS now (default: cap minus the verdict's squeue snapshot)")
    ap.add_argument("--cluster-a-budget", type=int, default=None)
    ap.add_argument("--rerun-existing", action="store_true", help="void D33's reuse (the drift check failed): re-run 0.5 / 1.0 too; the anchor prints regardless of closure")
    ap.add_argument("--assume-ok", action="store_true", help="treat INCOMPLETE cells as ok — LOUD; not permission")
    ap.add_argument("--nshards", type=int, default=9)
    args = ap.parse_args(argv)
    try:
        sha = args.wave_sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise Refusal(f"--wave-sha must be a full 40-hex commit, got {sha!r}")
        if args.attackers is not None and args.attackers < 0:
            raise Refusal("--attackers must be >= 0")
        rec = json.loads(Path(args.verdict).read_text(encoding="utf-8"))
        for k in ("views", "cells", "directions"):
            if k not in rec:
                raise Refusal(f"{args.verdict}: no `{k}` — not a screen_verdict.py record")
        if not rec["views"]:
            raise Refusal(f"{args.verdict}: the verdict names no dose view — nothing to enumerate (this printer never invents tags)")
        checks, rosters = run_checks(args, rec, sha)
        cluster_of = {v["tag"]: DEFAULT_CLUSTER for v in rec["views"]}
        for kv in args.cluster:
            t, _, c = kv.partition("=")
            if t not in cluster_of or c not in CLUSTERS:
                raise Refusal(f"--cluster {kv!r}: TAG must be a dose view of the verdict, cluster one of {sorted(CLUSTERS)}")
            cluster_of[t] = c
        routed = sorted(set(cluster_of.values()))
        blocks = {cl: site_block(cl) for cl in routed}   # a routed cluster with no site block refuses before anything prints
        rows, evidence_meta = load_evidence(args)
        resolver = CellResolver(rec["views"], rosters)
        wall = Walltimes(rows, resolver, sorted({v["target"] for v in rec["views"]}))
        plan = build_plan(rec, args, wall, cluster_of)
        budgets: Dict[str, Tuple[int, str]] = {}
        for cl in routed:
            flag = getattr(args, f"{cl}_budget")
            if flag is not None:
                budgets[cl] = (flag, f"--{cl}-budget {flag}")
                continue
            cap = int(blocks[cl]["CLUSTER_HIGH_BUDGET_GPUS"])
            used, how = used_budget_from_verdict(rec, cl)
            if used is None:
                budgets[cl] = (cap, f"cap {cap} (CLUSTER_HIGH_BUDGET_GPUS) assumed free — {how}; pass --{cl}-budget")
            else:
                budgets[cl] = (max(0, cap - used), f"cap {cap} (CLUSTER_HIGH_BUDGET_GPUS) - {how}")
        out = render(rec, plan, args, sha, wall, checks, budgets, cluster_of, evidence_meta)
        print("\n".join(out))
        if plan["pool"].get("refused"):
            print(f"REFUSED (multi-turn): {plan['pool']['refused']}")
            return EXIT_REFUSED
        return 0
    except Refusal as e:
        print(f"REFUSING: {e}")
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
