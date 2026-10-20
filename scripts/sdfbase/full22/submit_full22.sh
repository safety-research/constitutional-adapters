#!/bin/bash
# full22: PRINT (never run) the sbatch commands of the wave with the shard -> cluster assignment.
# Usage: submit_full22.sh <wave_sha> [--cluster-a-budget 13] [--cluster-b-budget 16] [--roster <yaml>]
#   The budgets are the per-cluster `high` QoS GPUs available to this account (the cap is 16 per
#   cluster, running + pending; 13 were free on cluster A when the wave was planned — the "96 on
#   cluster B" figure is cluster-wide idle capacity, not this account's budget, so the default is 16 and
#   --cluster-b-budget 96 widens the attacker pool only if the cap is confirmed lifted).
# Reads configs/sdfbase_full22_roster.yaml (or --roster <yaml>, else $F22_ROSTER — the same selection common.sh's jobs
# read; the printed make_bank_view.py lines carry it as --roster, repo-relative): a (target, view) with fewer than 2
# cells (none only) gets NO jobs (nothing to contrast; enable its roster cells first). Nothing here executes sbatch.
set -uo pipefail
WAVE_SHA="${1:?wave sha}"; shift
BULB=13; CHAR=16; ROSTER=""
while [ $# -gt 0 ]; do case "$1" in --cluster-a-budget) BULB="$2"; shift;; --cluster-b-budget) CHAR="$2"; shift;; --roster) ROSTER="$2"; shift;; *) echo "bad arg $1"; exit 2;; esac; shift; done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
[ -n "$ROSTER" ] || ROSTER="${F22_ROSTER:-$REPO/configs/sdfbase_full22_roster.yaml}"
case "$ROSTER" in /*) ;; *) ROSTER="$REPO/$ROSTER";; esac        # F22_ROSTER is repo-relative in the jobs (common.sh cds to the checkout)
[ -f "$ROSTER" ] || { echo "FATAL: no roster at $ROSTER" >&2; exit 2; }
ROSTER_REL="${ROSTER#"$REPO"/}"                                    # what the printed cluster commands name (they cd into the checkout)
PY=$(command -v python3)
"$PY" - "$ROSTER" "$WAVE_SHA" "$BULB" "$CHAR" "$HERE" "$ROSTER_REL" <<'PY'
import os
import sys, yaml
roster, sha, bulb, char = yaml.safe_load(open(sys.argv[1])), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, sys.argv[5])
roster_rel = sys.argv[6]   # the roster as the printed cluster commands name it (--roster; the jobs read the same file through F22_ROSTER)
from f22_roster import served_cells   # view cells + enabled lanes: the generator's own set
# Each cluster's login host, directories and QoS come from its block in the site config (reproduce/lib/clusters.py), looked
# up by the cluster's registry label; the label itself never stands for a host.
sys.path.insert(0, os.path.join(sys.argv[5], os.pardir, os.pardir, os.pardir, "reproduce", "lib"))
import clusters as CL
def where(label):
    try:
        b = CL.block_for(label)
    except SystemExit as exc:
        sys.exit(f"FATAL: {exc}. The printed commands need each cluster's login host, directories and QoS from the site config.")
    logs = b["CLUSTER_LOG_DIR"].rstrip("/")
    # every job line carries the site's partition, QoS, account and log file: the job scripts' own headers name one site's
    flags = [f"--qos={b['CLUSTER_QOS']}", f"--partition={b['CLUSTER_PARTITION']}"]
    if (b.get("CLUSTER_ACCOUNT") or "").strip():
        flags.append(f"--account={b['CLUSTER_ACCOUNT'].strip()}")
    flags.append(f"--output={logs}/%x_%j.out")
    if (b.get("CLUSTER_SBATCH_EXTRA") or "").strip():
        flags.append(b["CLUSTER_SBATCH_EXTRA"].strip())
    return {"host": b["CLUSTER_SSH"], "eval": b["CLUSTER_EVAL_REPO_ROOT"].rstrip("/"), "envs": b["CLUSTER_ENVS_DIR"].rstrip("/"),
            "logs": logs, "qos": b["CLUSTER_QOS"], "sbatch": " ".join(flags)}
CHW, BUW = where("cluster_b"), where("cluster_a")
S = "scripts/sdfbase/full22"
served = []   # (key, view, ncells)
for key, t in roster["targets"].items():
    for view, v in t["views"].items():
        n = 1 + len(served_cells(roster, key, view))
        if n >= 2:
            served.append((key, view, n))
        else:
            print(f"# SKIP {key}/{view}: config lists `none` only (roster: enable cells, regenerate configs)")
# --- cluster B: multi-turn (target servers + the shared attacker pool) --------------------------
# THE CANONICAL MULTI-TURN ATTACKS (user ruling 2026-09-11): crescendo + pair, ONE JOB PER ATTACK because
# each serves at its own explicit window (geometry.multiturn_window: 49,152 / 16,384). ActorAttack is NOT
# emitted by default (no measured 4,096-cap window; ~2/3 of the projected bill) -- F22_ATTACKS=crescendo,pair,actorattack
# opts it in, and target_attacks.sbatch then needs its window passed explicitly.
attacks = [a for a in os.environ.get("F22_ATTACKS", "crescendo,pair").split(",") if a]
est = {"crescendo": "2-3", "pair": "1-2", "actorattack": "6-9"}   # per cell-set at 4,096 (R2-D253i/z: ~65 min crescendo, less pair)
att_jobs = []   # (name, key, view, attacks, est_h)
for key, view, n in served:
    tag = key + ("" if view == "main" else f"_{view}")
    for atk in attacks:
        att_jobs += [(f"f22-att-{tag}-{atk}", key, view, atk, est.get(atk, "?"))]
n_att = len(att_jobs)
n_atk = max(2, min(2 * n_att, char - n_att))          # ~2 attacker GPUs per target GPU when the budget allows
if n_att + n_atk > char:
    print(f"# WARNING: {n_att} target jobs + {n_atk} attackers exceed the cluster B budget {char}; "
          "submit the attackers first, then targets as budget frees (or raise --cluster-b-budget)")
# --- cluster A: Petri proxies + the single-shot servers ------------------------------------------
petri_jobs = [(f"f22-petri-{k}{'' if v == 'main' else '_' + v}", k, v, n + 1, "8-12") for k, v, n in served]
stat_jobs = []
for key, view, n in served:
    tag = key + ("" if view == "main" else f"_{view}")
    # geometry phases (2026-09-06): static families + BoN at 4864 x budget-S, controls at 2048 x 256 — one
    # standup per geometry inside a job; the 7-cell configs split the static families over two jobs and
    # put BoN + controls in a third so ~S drivers are in flight per server
    if n >= 5:
        # controls (served gsm8k/mmlu/xstest/or-bench) are HELD OUT of wave 1: the served-capability driver
        # refuses these cells' provenance (served==trained template hash vs an EMPTY trained hash on gemma;
        # a chat-marker boundary retokenization on qwen) — README section 10; run them once the provenance
        # is stamped (target_static.sbatch still runs the phase when asked)
        stat_jobs += [(f"f22-stat-{tag}-1", key, view, "pasttense,codeattack", "5-8"),
                      (f"f22-stat-{tag}-2", key, view, "renellm,grok420", "5-8"),
                      (f"f22-stat-{tag}-3", key, view, "bon", "1-3")]
    else:
        stat_jobs += [(f"f22-stat-{tag}-all", key, view, "pasttense,codeattack,renellm,grok420,bon", "4-7")]
# THE CANONICAL BEST-OF-N (user ruling 2026-09-11): every lane's BoN leg is the 512 store PLUS the 4,096
# climb of its cap-coherent at-cap draws — so a climb array follows the static/BoN job for every config.
# bon_climb_plan.py branches per cell: climb where a 512 store exists, NATIVE 4,096 where none does.
climb_jobs = [(f"f22-bonclimb-{k + ('' if v == 'main' else '_' + v)}", k, v, 4, "2-4") for k, v, _ in served]
# an array's TASKS are concurrent 1-GPU jobs: count them, so the budget line is honest
n_climb_tasks = sum(j[3] for j in climb_jobs)
n_bulb = len(petri_jobs) + len(stat_jobs) + n_climb_tasks
if n_bulb > bulb:
    print(f"# WARNING: {len(petri_jobs)} Petri + {len(stat_jobs)} static jobs exceed the cluster A budget {bulb}; "
          "the static jobs beyond the budget queue (pending at high) and start as jobs finish")
print(f"# full22 @ {sha}: {sum(n for _, _, n in served)} cells over {len(served)} configs "
      f"({', '.join(f'{k}/{v}={n}' for k, v, n in served)})")
print(f"# GPUs: cluster B {n_att} targets + {n_atk} attackers = {n_att + n_atk} (budget {char}); "
      f"cluster A {len(petri_jobs)} Petri + {len(stat_jobs)} static + {n_climb_tasks} BoN-climb array tasks ({len(climb_jobs)} arrays) = {n_bulb} (budget {bulb}); total {n_att + n_atk + n_bulb}")
print("# Estimated wall-clock per job (from the mt1 / r3 measurements and the Modal stacked timing rows, NOT measured at this size): see the table")
print("# Geometry: attacks = EXPLICIT per attack (geometry.multiturn_window: crescendo 49152 / pair 16384, measured maxima on complete stores) x budget-S, cap 4096; static+BoN = families_served:short 4864 x budget-S"
      " (gemma 121 / qwen 148; drivers capped at 96 in flight); capability/over-refusal held out of wave 1; Petri = 16384 x budget-S (unchanged)")
print()
print("| cluster | job | GPUs | est. wall-clock (h) | args |")
print("|---|---|---|---|---|")
for i in range(n_atk):
    print(f"| cluster_b | f22-atk-{i+1} | 1 | pool: lives until STOP (24 h walltime) | {sha[:12]} |")
for name, key, view, attacks, est in att_jobs:
    print(f"| cluster_b | {name} | 1 | {est} | {key} {view} {attacks} {sha[:12]} 9 0-8 |")
for name, key, view, legs, est in petri_jobs:
    print(f"| cluster_a | {name} | 1 | {est} (API-paced; {legs} legs at width 1) | {key} {view} {sha[:12]} |")
for name, key, view, fams, est in stat_jobs:
    print(f"| cluster_a | {name} | 1 | {est} | {key} {view} {fams} {sha[:12]} 9 |")
for name, key, view, nsh, est in climb_jobs:
    print(f"| cluster_a | {name} | {nsh} (array) | {est} | {key} {view} {sha[:12]} {nsh}  (BoN 512 -> 4096 climb; native 4096 for a cell with no 512 store) |")
print()
CH = f"ssh {CHW['host']} 'cd {CHW['eval']}/%s && sbatch {CHW['sbatch'].replace('%', '%%')} %s'"
BU = f"ssh {BUW['host']} 'cd {BUW['eval']}/%s && sbatch {BUW['sbatch'].replace('%', '%%')} %s'"
print("# --- 0. BUILD THE BANK VIEWS ONCE, on each cluster, BEFORE any job that consumes them. The view is")
print("#        SHARED state: every job of a (target, view) reads the same cells_view/<view>, and jobs that")
print("#        built it concurrently raced (423389 died reading a stamped sidecar another job was mid-write).")
print("#        The jobs now VERIFY read-only and build only under a lock — this step makes that the normal path,")
print("#        and it is a CHECKED PRECONDITION: it refuses a view that does not hold exactly the roster's cells.")
for w in (BUW, CHW):
    views = " ".join(f"{k}:{v}" for k, v, _ in served)
    print(f"ssh {w['host']} 'cd {w['eval']}/{sha} && for kv in {views}; do "
          f"{w['envs']}/vllm-attacker/bin/python "
          f"{S}/make_bank_view.py --target ${{kv%%:*}} --view ${{kv##*:}} --roster {roster_rel} && "
          f"{w['envs']}/vllm-attacker/bin/python "
          f"{S}/make_bank_view.py --check --target ${{kv%%:*}} --view ${{kv##*:}} --roster {roster_rel} || exit 1; done'")
if roster_rel != "configs/sdfbase_full22_roster.yaml":
    print(f"# --- ROSTER: this plan reads {roster_rel}, NOT the production roster — export F22_ROSTER={roster_rel} on every sbatch line below")
    print("#        (a batch job inherits the submitter's exported environment; the jobs pass it to every roster reader)")
print("# --- SERVING ENGINE (native-serving-default, 2026-09-13; the KIND-AWARE default, 2026-09-15; native-psr-serving stage 2):")
print("#        every job below serves its steered cells on the NATIVE engine by default (F22_SERVING_PATH=lora_native, AUTO-selected")
print("#        at f22_init from the view's manifest cells by kind; wave root exp/sdfbase_full22_native). PSR / vector / arditi")
print("#        (kind static_r) cells are SERVED natively through the residual slot (position-restricted classes when")
print("#        NATIVE_STAGE2_ARMED); alphasteer stays lens-only and is skipped by name. A view whose steered cells are ALL")
print("#        lens-only kinds (alphasteer) AUTO-selects the lens (wave root exp/sdfbase_full22); every standup log carries ONE")
print("#        `[serving-path] AUTO-selected ...` line. An explicit `F22_SERVING_PATH=lens` (or lora_native) prefix on the sbatch")
print("#        always wins (the batch job inherits the submitter's exported environment, never --export=): the stacked lens")
print("#        standup, byte for byte, under exp/sdfbase_full22. The attacker pool follows the SAME F22_SERVING_PATH as its targets")
print("#        (its records live under the wave root the path selects).")
print("# --- 1. the shared attacker pool (cluster B) — wait for each record under exp/sdfbase_full22/attackers/")
for i in range(n_atk):
    print(CH % (sha, f"-J f22-atk-{i+1} {S}/attacker_pool.sbatch {sha}"))
print("# --- 2. the target servers + drivers (cluster B) — after >= 1 attacker record is LIVE")
for name, key, view, attacks, _ in att_jobs:
    print(CH % (sha, f"-J {name} {S}/target_attacks.sbatch {key} {view} {attacks} {sha} 9 0,1,2,3,4,5,6,7,8"))
print("# --- 3. the single-shot servers (cluster A)")
for name, key, view, fams, _ in stat_jobs:
    print(BU % (sha, f"-J {name} {S}/target_static.sbatch {key} {view} {fams} {sha} 9"))
print("# --- 3b. THE BEST-OF-N CLIMB (the canonical best-of-N = the 512 store + the 4,096 climb of its")
print("#         cap-coherent at-cap draws; a cell with NO 512 store draws natively at 4,096 instead —")
print("#         bon_climb_plan.py decides per cell). Runs AFTER that config's BoN static job has")
print("#         written its 512 store; shard 0 writes the plan and the others wait for it.")
for name, key, view, nsh, _ in climb_jobs:
    print(BU % (sha, f"-J {name} --array=0-{nsh - 1} {S}/bon_climb.sbatch {key} {view} {sha} {nsh}"))
print("# --- 4. the Petri proxies (cluster A); then, per proxy, the grading host driver once endpoint.json lands:")
for name, key, view, _, _ in petri_jobs:
    print(BU % (sha, f"-J {name} {S}/petri_proxy.sbatch {key} {view} {sha}"))
for name, key, view, _, _ in petri_jobs:
    print(f"#    setsid nohup {S}/petri_drive_full22.sh {key} {view} cluster_a {sha} > ~/.local/state/sdfbase_full22/petri/{key}{'' if view == 'main' else '_' + view}/autostart.log 2>&1 &")
print("# --- watchers (arm each in the SAME turn as its submission):")
print(f"#    ~/scripts/watch-job.sh --slurm <jobid> --host {CHW['host']}   # attackers + attack jobs")
print(f"#    ~/scripts/watch-job.sh --slurm <jobid> --host {BUW['host']}   # Petri proxies + static jobs")
PY
