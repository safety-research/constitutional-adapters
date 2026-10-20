#!/usr/bin/env python
"""full22-newmodels: PRINT (never run) the sbatch commands for the nemotron3nano4b / glm46vflash legs
of the full22 wave, with the shard -> cluster assignment sized for these two models' geometries —
AFTER checking that every precondition a job would otherwise discover on a GPU already holds:

  1. OFFLINE  the committed configs / manifests regenerate byte-identically from the roster + pins
              (make_full22_configs.py --check) — a config that drifted from the roster never prints;
  2. ON EACH CLUSTER a job is planned for (read-only, over ssh, unless --offline):
       checkout   $SISTEER_EVAL_REPO_ROOT/<wave_sha> exists and its HEAD is the wave sha;
       cache      the target's HF snapshot at the roster's PINNED revision is in that cluster's cache
                  (config.json under hub/models--<org>--<name>/snapshots/<revision>/) — cluster B had
                  no GLM-4.6V-Flash until 2026-09-07; a job on a cluster without the model dies at
                  the standup, after the venv builds;
       view       make_bank_view.py --check for the (target, main) view: the view is BUILT, every
                  entry hashes to its pin, and it holds EXACTLY the roster's cells — THE SHORT-VIEW
                  GUARD. A SHORT (or stale, or missing) view REFUSES THE WHOLE PRINT: nothing
                  submittable is printed while any view a job would consume is short;
       budget     the account's RUNNING + PENDING `high` GPUs (the per-user cap is 16 per cluster);
       attackers  the LIVE members of the shared attacker pool (records + squeue state), so the
                  printed attacker count is what is still needed, not a fresh pool.
  3. The plan, as a table + the exact commands, in the launch order of the README (views, attackers,
     target servers, single-shot servers, Petri proxies + their grading-host drivers as systemd --user
     transient units, watchers), plus the coordination post.

Nothing here executes sbatch, ssh writes nothing (every remote command is a read), and a refusal
exits 2 with the CHECK lines and the command that fixes each one.

Usage:
  submit_newmodels.py <wave_sha> [--targets nemotron3nano4b,glm46vflash]
      [--cluster <key>=cluster_a|cluster_b ...]     where each target's attack + static jobs run
                                                     (default: cluster_b for both; the attacker
                                                     pool lives there)
      [--petri-cluster cluster_a|cluster_b]         where the Petri proxies run (default cluster_a:
                                                     the archived bundle + every previous proxy)
      [--cluster-a-budget N] [--cluster-b-budget N]  override the probed free `high` budget
      [--offline]                                    skip the ssh probes (plan only; banner says so)
      [--roster <yaml>] [--grading-host-worktree <path>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from f22_roster import (  # noqa: E402
    LEGS,
    served_cells,
    target_capability_omitted,
    target_caps,
    target_multiturn_families,
    target_windows,
    view_legs,
    view_legs_omitted,
    view_multiturn_windows,
    view_serving,
    view_windows,
)

S = "scripts/sdfbase/full22"
DEFAULT_TARGETS = ("nemotron3nano4b", "glm46vflash")
#: the clusters, by registry label; each one's login host, directories and QoS come from its block in the site config (where)
CLUSTERS = {"cluster_a": {}, "cluster_b": {}}
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import clusters as CL  # noqa: E402  (the site's cluster blocks)


def where(cluster: str) -> dict:
    """The login host, shared root, eval-repo root, envs and log directories and QoS of `cluster` (a registry label), from its
    block in the site config (reproduce/lib/clusters.py); refused by name when the site config does not describe it. `sbatch`
    is the flags every job line carries (--qos, --partition, --account when set, --output under CLUSTER_LOG_DIR and the
    site's CLUSTER_SBATCH_EXTRA): the job scripts' own headers name one site's values."""
    try:
        b = CL.block_for(cluster)
    except SystemExit as exc:
        raise SystemExit(f"REFUSING: {str(exc.code).removeprefix('REFUSED: ')}") from None
    logs = b["CLUSTER_LOG_DIR"].rstrip("/")
    flags = [f"--qos={b['CLUSTER_QOS']}", f"--partition={b['CLUSTER_PARTITION']}"]
    if (b.get("CLUSTER_ACCOUNT") or "").strip():
        flags.append(f"--account={b['CLUSTER_ACCOUNT'].strip()}")
    flags.append(f"--output={logs}/%x_%j.out")
    if (b.get("CLUSTER_SBATCH_EXTRA") or "").strip():
        flags.append(b["CLUSTER_SBATCH_EXTRA"].strip())
    return {"host": b["CLUSTER_SSH"], "shared": b["CLUSTER_SHARED_ROOT"].rstrip("/"), "eval": b["CLUSTER_EVAL_REPO_ROOT"].rstrip("/"),
            "envs": b["CLUSTER_ENVS_DIR"].rstrip("/"), "logs": logs, "qos": b["CLUSTER_QOS"], "sbatch": " ".join(flags)}
PER_USER_CAP = 16
#: the wave's served geometries (README 6b / target_static.sbatch): window per eval; S from the model budget.
#: The multi-turn attacks are NOT in this table: since 2026-09-11 target_attacks.sbatch serves ONE attack per job
#: at its own EXPLICIT window (geometry.MULTITURN_CAP4096_WINDOWS: crescendo 49,152 / pair 16,384 -- measured
#: maxima on complete stores at the 4,096 cap), never the table's multiturn_attacks 8,704.
WINDOWS = {"static": 4864, "petri": 16384, "controls": 2048}
#: the canonical multi-turn attacks; ActorAttack (no measured 4,096-cap window, ~2/3 of the projected bill) is
#: available through F22_ATTACKS=crescendo,pair,actorattack and then needs its window passed to the sbatch by hand
DEFAULT_ATTACKS = ("crescendo", "pair")
ATTACK_EST = {"crescendo": "2-3", "pair": "1-2", "actorattack": "6-9"}


def BUDGET_BY_MODEL_SEQS(key: str, L: int) -> int:
    """S = the model's budget at window L (geometry.BUDGET_BY_MODEL.seqs_at) — the printer's per-leg rows."""
    from sisteer.capeval.serving.geometry import BUDGET_BY_MODEL
    return BUDGET_BY_MODEL[key].seqs_at(int(L))


def attack_list() -> list:
    return [a for a in os.environ.get("F22_ATTACKS", ",".join(DEFAULT_ATTACKS)).split(",") if a]


# --------------------------------------------------------------------------- the plan

def _stage_dir_text() -> str:
    """The grading host's staging directory as the printed commands spell it: the site's $STAGE (SISTEER_STAGE_DIR, from the
    environment or the site config) expanded, else a placeholder for the operator to fill in."""
    try:
        from sisteer import site
        return site.expand("$STAGE")
    except (ImportError, ValueError):     # site.PlaceholderError / site.SiteConfigError: no staging directory named here
        return "<SISTEER_STAGE_DIR>"


def split_target(target: str) -> tuple:
    """A --targets entry -> (model key, view): a bare key is its main view; `key/view` names another."""
    key, _, view = target.partition("/")
    return key, (view or "main")


def plan(roster: dict, targets: list, cluster_of: dict, petri_cluster: str) -> dict:
    """The job shapes per (target, view), derived from the roster: the cells, the view's LEGS
    (f22_roster.view_legs — a leg the view omits gets NO job and its reason is recorded under
    `omitted`), and the view's serving window when it sets one (every standup of that view at L;
    S = the model budget at L). `targets` entries are `key` (the main view) or `key/view`."""
    from sisteer.capeval.serving.geometry import (BUDGET_BY_MODEL, MULTITURN_CAP4096_WINDOWS, multiturn_rerun_window,
                                                   multiturn_window)

    served, att, stat, ctl, petri, skipped, omitted = [], [], [], [], [], [], {}
    for target in targets:
        key, view = split_target(target)
        t = roster["targets"][key]
        n = 1 + len(served_cells(roster, key, view))
        if n < 2:
            skipped.append(f"# SKIP {key}/{view}: config lists `none` only (roster: enable cells, regenerate)")
            continue
        legs = view_legs(roster, key, view)
        omitted[target] = view_legs_omitted(roster, key, view)
        b = BUDGET_BY_MODEL[key]
        L_view = view_serving(roster, key, view).get("max_model_len")
        vw = view_windows(roster, key, view)          # per-leg windows (the llama8b shortconst view, 2026-09-12) or the table's
        per_leg = vw["kind"] == "per_leg"
        geom = {ev: ((L_view, b.seqs_at(L_view)) if L_view else (L, b.seqs_at(L))) for ev, L in WINDOWS.items()}
        if per_leg:
            # a PER-LEG view: the static standup (BoN rides it) at its own measured window; Petri at the certified table window;
            # the multi-turn leg through the caps path below with the view's pin overrides
            geom["static"] = (vw["static"], b.seqs_at(vw["static"]))
        ctl_L = vw["controls"] if vw["controls"] is not None else vw["controls_gsm8k"]
        if ctl_L is not None:
            # the controls standup at the view's own controls window (per-leg), else at the roster's gsm8k controls window for a
            # gsm8k-bearing TABLE view (controls_window.gsm8k, 2026-09-14: the 10-shot gsm8k prompt + the 512 cap overflowed
            # capability:short's 2,048 — target_static.sbatch reads the same value through roster_view.py `window gsm8k`)
            geom["controls"] = (ctl_L, b.seqs_at(ctl_L))
        caps = target_caps(roster, key)
        win = view_multiturn_windows(roster, key, view) if caps else {}   # the target's pins + this view's first-pass overrides
        rerun_geom = None
        mt_geom = None
        if L_view:
            # a view with its own serving window (the llama8b fullconst view) serves EVERY leg at it, the multi-turn row
            # included (the printer labels that line VIEW; the view has no multi-turn leg, the row is the window's record)
            geom["attacks"] = (L_view, b.seqs_at(L_view))
        if not caps and "attacks" in legs:
            # a target WITHOUT a caps block: the canonical 4,096-cap instrument — ONE job per attack, each at its own
            # EXPLICIT window (target_attacks.sbatch / geometry.multiturn_window(attacks); MULTITURN_CAP4096_WINDOWS),
            # never the table's multiturn_attacks 8,704
            geom.update({f"attacks:{a}": (L, b.seqs_at(L)) for a, L in MULTITURN_CAP4096_WINDOWS.items()})
        if caps and not L_view:
            # a target with its own multi-turn caps (roster caps.multiturn_target — PER FAMILY; the config's
            # attacks.multiturn_target_max_new_tokens map) serves each family of the attacks leg at
            # the window DERIVED from ITS cap — or at the roster's PIN over it (caps.windows[family]:
            # geometry.pin_window refuses a pin below the derived minimum) — geometry.multiturn_window, the same call
            # target_attacks.sbatch makes — never WINDOWS' literal, which stands only for a config without the key; the
            # over-window re-run pass's window beside it (multiturn_rerun_window, the re-run sbatch's call). geom["attacks"] /
            # rerun_geom are the MAX over the families: what a COMBINED standup over the three families serves (the `-all` job
            # shape below 5 cells); the split jobs serve each family's own window (mt_geom)
            mt_geom = {}
            for fam, cap_f in caps["multiturn_target"].items():
                wf = win.get(fam) or {}
                L_f = multiturn_window(int(cap_f), wf.get("multiturn"))
                L_rf = multiturn_rerun_window(int(cap_f), wf.get("multiturn_rerun"), window=L_f)
                mt_geom[fam] = {"cap": int(cap_f), "window": L_f, "S": b.seqs_at(L_f),
                                "window_source": "pinned" if wf.get("multiturn") else "derived",
                                "rerun_window": L_rf, "rerun_S": b.seqs_at(L_rf),
                                "rerun_window_source": "pinned" if wf.get("multiturn_rerun") else "derived",
                                "pin_source": wf.get("source")}
            L_mt = max(g["window"] for g in mt_geom.values())
            geom["attacks"] = (L_mt, b.seqs_at(L_mt))
            L_re = max(g["rerun_window"] for g in mt_geom.values())
            rerun_geom = (L_re, b.seqs_at(L_re))
        served.append({"key": target, "model_key": key, "view": view, "n": n, "cluster": cluster_of[target],
                       "model": t["model"], "revision": t["model_revision"], "hybrid": b.hybrid, "geom": geom,
                       "budget": b.batched_token_budget, "budget_source": BUDGET_BY_MODEL[key].source,
                       "legs": legs, "max_model_len": L_view, "leg_windows": (vw if per_leg else None), "caps": caps, "windows": win, "rerun_geom": rerun_geom,
                       "mt_geom": mt_geom, "capability_omitted": target_capability_omitted(roster, key),
                       "families_omitted": target_multiturn_families(roster, key)[1]})
        c = cluster_of[target]
        tag = key if view == "main" else f"{key}-{view}"
        if "attacks" in legs and not caps:
            # the canonical instrument (no caps block): ONE JOB PER ATTACK whatever the cell count -- each serves at
            # its own explicit window (target_attacks.sbatch / geometry.multiturn_window), so a combined attack job
            # does not exist on this path; ActorAttack only through F22_ATTACKS (attack_list) + a hand-passed window
            att += [(f"f22-att-{tag}-{a}", target, a, ATTACK_EST.get(a, "?"), c) for a in attack_list()]
        elif "attacks" in legs:
            # a target with a caps block (the sdfbase-llama8b configs): the families that RUN (roster
            # caps.multiturn_families_omitted, the keep-or-drop switch):
            # a dropped family gets NO job and is printed as an omission below; a job per family at >= 5 cells, else
            # ONE combined job at the MAX of the families' pinned / derived windows (target_attacks.sbatch path (a))
            run_fams, _om = target_multiturn_families(roster, key)
            est = {"crescendo": "6-9", "pair": "4-6", "actorattack": "6-9"}
            if n >= 5:
                att += [(f"f22-att-{tag}-{fam}", target, fam, est[fam], c) for fam in run_fams]
            else:
                att += [(f"f22-att-{tag}-all", target, ",".join(run_fams), "5-8", c)]
        fams = []
        if "static" in legs:
            fams.append("pasttense,codeattack,renellm,grok420")
        if "bon" in legs:
            fams.append("bon")
        if fams:
            if n >= 5 and "static" in legs:
                stat += [(f"f22-stat-{tag}-1", target, "pasttense,codeattack", "3-6", c),
                         (f"f22-stat-{tag}-2", target, "renellm,grok420", "3-6", c)]
                if "bon" in legs:
                    stat += [(f"f22-stat-{tag}-3", target, "bon", "1-3", c)]
            else:
                stat += [(f"f22-stat-{tag}-all", target, ",".join(fams), "3-6", c)]
        # the controls driver's leg(s) — capability (gsm8k, mmlu) and/or over-refusal (xstest, or_bench):
        # ONE `controls` job per view, its --controls list read from the roster (roster_view.py <key> <view>
        # controls); never folded into a family list (BoN is its own sub-phase; the historical printer held
        # controls out of the family jobs and this keeps that shape)
        if "capability" in legs or "controls" in legs:
            ctl.append((f"f22-ctl-{tag}", target, "controls", "0.5-2", c))
        if "petri" in legs:
            petri.append((f"f22-petri-{tag}", target, n + 1, "8-12", petri_cluster))
    return {"served": served, "att": att, "stat": stat, "ctl": ctl, "petri": petri, "skipped": skipped,
            "omitted": omitted}


def attacker_plan(p: dict, budgets: dict, live_attackers: dict) -> dict:
    """Per cluster: how many attacker-pool members the attack jobs there need (~2 per target GPU when the
    budget allows, at least 2), how many are already LIVE, how many to submit. A pool is per cluster:
    its records live under that cluster's $SHARED and its URLs are cluster-internal."""
    out = {}
    for c in CLUSTERS:
        n_att = sum(1 for j in p["att"] if j[4] == c)
        if not n_att:
            continue
        n_targets = n_att + sum(1 for j in p["stat"] if j[4] == c) + sum(1 for j in p["petri"] if j[4] == c)
        want = max(2, min(2 * n_att, budgets[c] - n_targets))
        live = len(live_attackers.get(c, []))
        out[c] = {"want": want, "live": live, "submit": max(0, want - live), "targets": n_targets}
    return out


# --------------------------------------------------------------------------- the probes

REMOTE = r'''
set -u
SHA={sha}; S={shared}; QOS={qos}
source "$S/scripts/hf_env.sh" 2>/dev/null || echo "CHECK hf_env - PROBLEM no $S/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
REPO="{eval}/$SHA"
if [ -d "$REPO/src/sisteer" ]; then
  head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < "$REPO/.git/HEAD" 2>/dev/null)
  if [ "$head" = "$SHA" ]; then echo "CHECK checkout - OK $REPO"; else echo "CHECK checkout - PROBLEM HEAD $head != $SHA at $REPO"; fi
else
  echo "CHECK checkout - PROBLEM no checkout at $REPO (the repo-mirror tool, then git -C <the repo mirror> worktree add --detach $REPO $SHA)"
fi
PYV="{envs}/vllm-attacker/bin/python"; [ -x "$PYV" ] || echo "CHECK venv - PROBLEM no $PYV"
{cache_checks}
{view_checks}
used=$(squeue -u "$USER" -h -t RUNNING,PENDING -o "%q %b" 2>/dev/null | awk -v q="$QOS" '$1==q {{ n=$2; sub(/.*:/, "", n); s+=n }} END {{ print s+0 }}')
echo "CHECK budget - OK used=$used cap={cap}"
for rec in "$S"/exp/sdfbase_full22/attackers/*.json; do
  [ -f "$rec" ] || continue
  job=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$rec" 2>/dev/null) || continue
  url=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$rec" 2>/dev/null)
  st=$(squeue -j "$job" -h -o %T 2>/dev/null); [ -n "$st" ] || st=GONE
  echo "CHECK attacker $job $st $url"
done
[ -e "$S/exp/sdfbase_full22/STOP" ] && echo "CHECK stop - PROBLEM wave STOP sentinel present at $S/exp/sdfbase_full22/STOP (every job exits at start; rm it deliberately)" || echo "CHECK stop - OK no wave STOP"
'''


def _mangled(model: str) -> str:
    return "models--" + model.replace("/", "--")


def remote_script(cluster: str, sha: str, served_here: list, views_here: list) -> str:
    cache = []
    for s in served_here:
        snap = f'$HF_HOME/hub/{_mangled(s["model"])}/snapshots/{s["revision"]}'
        cache.append(f'if [ -f "{snap}/config.json" ]; then echo "CHECK cache {s["key"]} OK {snap}"; '
                     f'else echo "CHECK cache {s["key"]} PROBLEM no snapshot at {snap} '
                     f'(fetch: hf download --revision {s["revision"]} {s["model"]} into this cache)"; fi')
    views = []
    for entry in views_here:
        # `key` = the target's main view (every caller before 2026-09-09); `key/view` names another view (the sampled
        # printer's instruct-direct lane: gemma4e4b/it). The CHECK line echoes the entry as given, so a caller that
        # asked for `key/view` reads it back as `key/view` (parse_checks / the rebuild-command printers split on it).
        key, _, view = entry.partition("/")
        view = view or "main"
        views.append(
            f'if [ -x "$PYV" ] && [ -d "$REPO/src/sisteer" ]; then '
            f'out=$(cd "$REPO" && "$PYV" {S}/make_bank_view.py --check --target {key} --view {view} 2>&1); rc=$?; '
            f'if [ $rc = 0 ]; then echo "CHECK view {entry} OK $(echo "$out" | tail -1)"; '
            f'else echo "CHECK view {entry} PROBLEM $(echo "$out" | grep -m1 PROBLEM || echo "$out" | tail -1)"; fi; '
            f'else echo "CHECK view {entry} PROBLEM cannot run make_bank_view.py (no checkout or venv)"; fi')
    w = where(cluster)
    return REMOTE.format(sha=sha, shared=w["shared"], eval=w["eval"], envs=w["envs"], qos=w["qos"], cap=PER_USER_CAP,
                         cache_checks="\n".join(cache), view_checks="\n".join(views))


def probe(cluster: str, sha: str, served_here: list, views_here: list) -> list:
    """Run the read-only probe on one cluster; returns the CHECK lines (a failed ssh is itself a
    PROBLEM line — an unreachable cluster must never read as 'nothing wrong')."""
    ssh = os.environ.get("F22_SSH", "ssh")
    cmd = shlex.split(ssh) + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=25", where(cluster)["host"], "bash", "-s"]
    try:
        r = subprocess.run(cmd, input=remote_script(cluster, sha, served_here, views_here),
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return [f"CHECK ssh - PROBLEM {cluster}: {e!r}"]
    lines = [l for l in r.stdout.splitlines() if l.startswith("CHECK ")]
    if r.returncode != 0 and not lines:
        return [f"CHECK ssh - PROBLEM {cluster}: rc={r.returncode} {r.stderr.strip()[:300]}"]
    return lines or [f"CHECK ssh - PROBLEM {cluster}: probe printed nothing"]


def parse_checks(lines: list) -> tuple:
    """-> (problems, budget_used or None, live attackers [(job, url)])"""
    problems, used, live = [], None, []
    for l in lines:
        parts = l.split(" ", 4)   # CHECK <what> <who> <verdict> <detail>
        if len(parts) < 4:
            problems.append(l); continue
        _, what, who, verdict = parts[:4]
        detail = parts[4] if len(parts) > 4 else ""
        if what == "attacker":
            if verdict == "RUNNING":
                live.append((who, detail))
            continue
        if what == "budget":
            m = re.search(r"used=(\d+)", detail)
            used = int(m.group(1)) if m else None
        if verdict != "OK":
            problems.append(l)
    return problems, used, live


# --------------------------------------------------------------------------- the print

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("wave_sha")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--cluster", action="append", default=[], metavar="KEY=CLUSTER")
    ap.add_argument("--petri-cluster", default="cluster_a", choices=sorted(CLUSTERS))
    ap.add_argument("--cluster-a-budget", type=int, default=None)
    ap.add_argument("--cluster-b-budget", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--roster", type=Path, default=REPO / "configs/sdfbase_full22_roster.yaml")
    ap.add_argument("--grading-host-worktree", default=None,
                    help="the grading host's detached worktree at the wave sha (default: a detached worktree named wave-drive in the main "
                         "checkout's worktrees directory)")
    args = ap.parse_args(argv)
    if args.grading_host_worktree is None:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=str(REPO),
                                capture_output=True, text=True).stdout.strip()
        if not common:
            print(f"REFUSING: --grading-host-worktree has no default outside a git checkout ({REPO}); pass the worktree path"); return 2
        args.grading_host_worktree = str(Path(common).parent / "worktrees" / "wave-drive")
    sha = args.wave_sha
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        print(f"REFUSING: wave sha must be a full 40-hex commit, got {sha!r}"); return 2
    roster = yaml.safe_load(args.roster.read_text(encoding="utf-8"))
    targets = [t for t in args.targets.split(",") if t]
    for t in targets:
        key, view = split_target(t)
        if key not in roster["targets"] or view not in (roster["targets"][key].get("views") or {}):
            print(f"REFUSING: --targets {t!r}: the roster has no view {key}/{view}"); return 2
    cluster_of = {k: "cluster_b" for k in targets}
    for kv in args.cluster:
        k, _, c = kv.partition("=")
        if k not in targets or c not in CLUSTERS:
            print(f"REFUSING: --cluster {kv!r}: key must be one of {targets}, cluster one of {sorted(CLUSTERS)}"); return 2
        cluster_of[k] = c
    out = []
    P = out.append

    # 1. offline: the configs are the roster's
    gen = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--check"],
                         capture_output=True, text=True)
    if gen.returncode != 0:
        print("REFUSING: configs/manifests drift from the roster + pins — regenerate first:")
        print("  python scripts/sdfbase/make_full22_configs.py && python scripts/sdfbase/make_full22_configs.py --check")
        print(gen.stdout.strip()); return 2
    P(f"# CHECK configs - OK make_full22_configs.py --check: every generated file matches the roster + pins")

    p = plan(roster, targets, cluster_of, args.petri_cluster)
    out += p["skipped"]
    if not p["served"]:
        print("\n".join(out)); print("REFUSING: nothing to serve"); return 2

    # 2. the clusters
    budgets, live, problems = {}, {}, []
    for c in CLUSTERS:
        served_here = [s for s in p["served"] if s["cluster"] == c or args.petri_cluster == c]
        views_here = sorted({s["model_key"] if s["view"] == "main" else f"{s['model_key']}/{s['view']}"
                             for s in served_here})
        if not served_here:
            continue
        if args.offline:
            P(f"# CHECK {c} - UNVERIFIED (--offline): checkout / cache / view / budget / attackers NOT probed")
            continue
        lines = probe(c, sha, served_here, views_here)
        out += [f"# {l}" for l in lines]
        probs, used, live_c = parse_checks(lines)
        problems += probs
        live[c] = live_c
        if used is not None:
            budgets[c] = PER_USER_CAP - used
    for c in CLUSTERS:   # explicit budgets win; offline falls back to the documented defaults
        flag = getattr(args, f"{c}_budget")
        if flag is not None:
            budgets[c] = flag
        budgets.setdefault(c, {"cluster_a": 11, "cluster_b": 15}[c])
    if problems:
        print("\n".join(out))
        print(f"REFUSING TO PRINT SUBMISSIONS: {len(problems)} precondition(s) failed:")
        for pr in problems:
            print(f"  {pr}")
        views_bad = [pr.split()[2] for pr in problems if pr.startswith("CHECK view ")]
        if views_bad:
            print("# a SHORT / stale / missing bank view is an error, never a log line (README lesson 1). Build each view ONCE,")
            print("# out of place + atomic rename (make_bank_view.py), then re-run this printer:")
            for c in CLUSTERS:
                keys = sorted({s["key"] for s in p["served"] if (s["cluster"] == c or args.petri_cluster == c) and s["key"] in views_bad})
                for key in keys:
                    w = where(c)
                    # a comment line on purpose: on a refusal NOTHING printed may look like a submission (a line
                    # starting with `ssh ` is one); the fix is shown, the operator runs it deliberately
                    print(f"#   ssh {w['host']} 'cd {w['eval']}/{sha} && {w['envs']}/vllm-attacker/bin/python "
                          f"{S}/make_bank_view.py --target {key} --view main && {w['envs']}/vllm-attacker/bin/python "
                          f"{S}/make_bank_view.py --check --target {key} --view main'")
        return 2

    atk = attacker_plan(p, budgets, live)
    n_by_cluster = {c: sum(1 for j in p["att"] + p["stat"] + p["ctl"] + p["petri"] if j[-1] == c) + atk.get(c, {}).get("submit", 0) for c in CLUSTERS}
    served_s = ", ".join(f"{s['model_key']}/{s['view']}={s['n']}" for s in p["served"])
    P(f"# full22-newmodels @ {sha}: {sum(s['n'] for s in p['served'])} cells over {len(p['served'])} configs "
      f"({served_s}); Petri proxies on {args.petri_cluster}")
    for s in p["served"]:
        if s.get("caps"):
            c = s["caps"]
            fam = c["multiturn_target"]
            caps_s = ("multiturn_target " + " / ".join(f"{f} {fam[f]}" for f in fam) + f" (PER FAMILY; source: {c['multiturn_target_source']}), "
                      + ", ".join(f"{k} {c[k]}" for k in c if k not in ("source", "multiturn_target", "multiturn_target_source")))
            # the label names the geometry's real source: a view with its own serving window (fullconst) serves EVERY standup at it and
            # plans no multi-turn leg, so neither the pin nor the derivation applies there (VIEW); a lane view gets ONE LINE PER FAMILY —
            # PINNED (roster caps.windows[family]) or DERIVED (geometry.multiturn_window at the family's cap) — the
            # launcher's plan check reads exactly these words per family for main / it
            if s.get("max_model_len"):
                P(f"# CAPS {s['model_key']}/{s['view']} (source: {c['source']}): {caps_s}; multi-turn window VIEW {s['geom']['attacks'][0]} x {s['geom']['attacks'][1]} "
                  f"(views.{s['view']}.serving.max_model_len — the view's one window for every standup; the caps pins apply to views without one)")
            else:
                P(f"# CAPS {s['model_key']}/{s['view']} (source: {c['source']}): {caps_s}; multi-turn windows PER FAMILY below "
                  f"(the split jobs serve each family's own; a combined `-all` job the max, {s['geom']['attacks'][0]} x {s['geom']['attacks'][1]})")
                for f, g in (s.get("mt_geom") or {}).items():
                    P(f"#   multi-turn {f}: cap {g['cap']} -> window {g['window_source'].upper()} {g['window']} x {g['S']} "
                      + (f"(attacks.multiturn_max_model_len[{f}], roster caps.windows over geometry.multiturn_window)" if g["window_source"] == "pinned"
                         else "(geometry.multiturn_window at the family's cap)")
                      + f"; over-window re-run window {g['rerun_window_source'].upper()} {g['rerun_window']} x {g['rerun_S']} "
                      + (f"(attacks.multiturn_rerun_max_model_len[{f}] over geometry.multiturn_rerun_window)" if g["rerun_window_source"] == "pinned"
                         else "(geometry.multiturn_rerun_window)")
                      + (" — OMITTED from this wave (no job; roster caps.multiturn_families_omitted)" if f in (s.get("families_omitted") or {}) else ""))
                    if g.get("pin_source"):
                        P(f"#     {f} windows pinned on: {g['pin_source']}")
            # the keep-or-drop switch: a dropped multi-turn family is a recorded omission, never an inferred absence
            for fam_o, why_o in (s.get("families_omitted") or {}).items():
                P(f"#   multi-turn family omitted {fam_o}: {why_o}")
            for ctl_name, why in (s.get("capability_omitted") or {}).items():
                P(f"#   capability control omitted {ctl_name}: {why}")
        om = p["omitted"].get(s["key"]) or {}
        if om or s["legs"] != LEGS:
            P(f"# LEGS {s['model_key']}/{s['view']}: {', '.join(s['legs'])}" + (f"; window {s['max_model_len']} for every standup" if s["max_model_len"] else ""))
            for leg, why in om.items():
                P(f"#   omitted {leg}: {why}")
        if s.get("leg_windows"):   # a PER-LEG-window view (llama8b/shortconst, 2026-09-12): its windows, one line, beside the CAPS block
            lw = s["leg_windows"]
            P(f"# WINDOWS {s['model_key']}/{s['view']} PER LEG (roster views.{s['view']}.serving.windows): static + BoN {lw['static']} x {s['geom']['static'][1]}; "
              f"controls {lw['controls']} x {s['geom']['controls'][1]}; ladder rungs " + ", ".join(f"{r} -> {L} x {BUDGET_BY_MODEL_SEQS(s['model_key'], L)}" for r, L in sorted(lw['rungs'].items()))
              + f"; Petri {s['geom']['petri'][0]} x {s['geom']['petri'][1]} (the certified table window); multi-turn per family below (the view's first-pass overrides ride the pins)")
    for c in CLUSTERS:
        if n_by_cluster[c]:
            a = atk.get(c, {})
            P(f"# GPUs {c}: {sum(1 for j in p['att'] if j[4] == c)} attack + {sum(1 for j in p['stat'] if j[4] == c)} static + "
              f"{sum(1 for j in p['ctl'] if j[4] == c)} controls + "
              f"{sum(1 for j in p['petri'] if j[4] == c)} Petri + {a.get('submit', 0)} new attackers "
              f"({a.get('live', 0)} already LIVE, {a.get('want', 0)} wanted) = {n_by_cluster[c]} of a free `{where(c)['qos']}` budget {budgets[c]}"
              + ("  ** OVER BUDGET: the excess queues PENDING at high and counts against the cap **" if n_by_cluster[c] > budgets[c] else ""))
    P("# Geometry per target (window x S from geometry.BUDGET_BY_MODEL.seqs_at; the standup ATTESTS both and refuses a clamp):")
    for s in p["served"]:
        g = s["geom"]
        if "attacks:crescendo" in g:   # the canonical instrument: EXPLICIT per attack (no caps block)
            att_txt = (f"attacks crescendo {g['attacks:crescendo'][0]}x{g['attacks:crescendo'][1]} / pair {g['attacks:pair'][0]}x{g['attacks:pair'][1]} "
                       f"(explicit, cap 4096), ")
        elif "attacks" in g:          # the caps-table path: the MAX over the families' pinned / derived windows (a view window: VIEW)
            att_txt = f"attacks {g['attacks'][0]}x{g['attacks'][1]}, "
        else:
            att_txt = "attacks (no multi-turn leg on this view), "
        P(f"#   {s['key']}: budget {s['budget']:,} ({s['budget_source']}){' HYBRID (S capped at 256, prefix caching structurally OFF)' if s['hybrid'] else ''}; "
          + att_txt + f"static+BoN {g['static'][0]}x{g['static'][1]} (drivers capped at 96), "
          f"Petri {g['petri'][0]}x{g['petri'][1]}, controls {g['controls'][0]}x{g['controls'][1]} (held out of this wave)")
    P("# Estimated wall-clock per job is the precedent's (README 6), NOT measured on these models.")
    P("")
    P("| cluster | job | GPUs | est. wall-clock (h) | args |")
    P("|---|---|---|---|---|")
    for c, a in atk.items():
        for i in range(a["submit"]):
            P(f"| {c} | f22-atk-{a['live'] + i + 1} | 1 | pool: lives until STOP (24 h walltime) | {sha[:12]} |")
    def kv(target: str) -> str:
        k, v = split_target(target)
        return f"{k} {v}"
    for name, key, attacks, est, c in p["att"]:
        P(f"| {c} | {name} | 1 | {est} | {kv(key)} {attacks} {sha[:12]} 9 0-8 |")
    for name, key, fams, est, c in p["stat"]:
        P(f"| {c} | {name} | 1 | {est} | {kv(key)} {fams} {sha[:12]} 9 |")
    for name, key, fams, est, c in p["ctl"]:
        P(f"| {c} | {name} | 1 | {est} | {kv(key)} {fams} {sha[:12]} 9 |")
    for name, key, legs, est, c in p["petri"]:
        P(f"| {c} | {name} | 1 | {est} (API-paced; {legs} legs at width 1) | {kv(key)} {sha[:12]} |")
    P("")

    def sb(c: str, rest: str) -> str:
        w = where(c)
        return f"ssh {w['host']} 'cd {w['eval']}/{sha} && sbatch {w['sbatch']} {rest}'"

    P("# --- 0. the bank views: VERIFIED above (make_bank_view.py --check on every cluster a job runs on). They are SHARED")
    P("#        state; rebuild ONLY when the roster changes, ONCE, before any job (out-of-place + atomic rename; a job that")
    P("#        finds a short/stale view refuses). Rebuild command per (cluster, target):")
    for c in CLUSTERS:
        pairs = sorted({(s["model_key"], s["view"]) for s in p["served"] if s["cluster"] == c or args.petri_cluster == c})
        for key, view in pairs:
            w = where(c)
            P(f"#   ssh {w['host']} 'cd {w['eval']}/{sha} && {w['envs']}/vllm-attacker/bin/python "
              f"{S}/make_bank_view.py --target {key} --view {view} && {w['envs']}/vllm-attacker/bin/python {S}/make_bank_view.py --check --target {key} --view {view}'")
    P("# --- 1. the shared attacker pool — one pool PER CLUSTER that runs attack jobs (records under that cluster's")
    P("#        exp/sdfbase_full22/attackers/; URLs are cluster-internal). Wait for each new record before step 2.")
    for c, a in atk.items():
        for _, url in live.get(c, []):
            P(f"#   {c}: LIVE already: {url} (reused; the pool is wave-agnostic — same model, snapshot, flags)")
        for i in range(a["submit"]):
            P(sb(c, f"-J f22-atk-{a['live'] + i + 1} {S}/attacker_pool.sbatch {sha}"))
    P("# --- 2. the target servers + multi-turn drivers — after >= 1 attacker record on that cluster is LIVE")
    for name, key, attacks, _, c in p["att"]:
        P(sb(c, f"-J {name} {S}/target_attacks.sbatch {kv(key)} {attacks} {sha} 9 0,1,2,3,4,5,6,7,8"))
    P("# --- 3. the single-shot servers (static families + BoN as its own sub-phase; `controls` = the view's")
    P("#        --controls list: roster_view.py <key> <view> controls — capability gsm8k,mmlu and/or over-refusal xstest,or_bench)")
    for name, key, fams, _, c in p["stat"]:
        P(sb(c, f"-J {name} {S}/target_static.sbatch {kv(key)} {fams} {sha} 9"))
    if p["ctl"]:
        P("# --- 3b. the controls servers: capability (gsm8k, mmlu) and/or over-refusal (xstest, or_bench_hard_1k) per the")
        P("#        view's legs — target_static.sbatch reads the --controls list and the view's window from the roster")
        for name, key, fams, _, c in p["ctl"]:
            P(sb(c, f"-J {name} {S}/target_static.sbatch {kv(key)} {fams} {sha} 9"))
    P("# --- 4. the Petri proxies; then, per proxy, the grading host driver as a systemd --user TRANSIENT UNIT (a session")
    P("#        restart cannot take it; the proxy frees its own GPU if the driver's heartbeat stops). Preconditions on")
    P(f"#        the grading host: a detached worktree at the wave sha with the petri venv —")
    P(f"#          git -C <the base checkout> worktree add --detach {args.grading_host_worktree} {sha} && (cd {args.grading_host_worktree} && uv sync --frozen --extra cpu --group petri)")
    P("#        the grading host bank view per target (the driver validates every object it audits):")
    for name, key, _, _, c in p["petri"]:
        k, v = split_target(key)
        P(f"#          (cd {args.grading_host_worktree} && .venv/bin/python {S}/make_bank_view.py --bank {_stage_dir_text()}/sdfbase/full22_objects/{k} --target {k} --view {v})")
    P("#        OPENROUTER_API_KEY exported by the site's shell (sisteer.vault — never a key file), and the Qwen auditor tunnel on 18990 to a LIVE attacker (petri_drive_full22.sh refuses otherwise).")
    for name, key, _, _, c in p["petri"]:
        P(sb(c, f"-J {name} {S}/petri_proxy.sbatch {kv(key)} {sha}"))
    for name, key, _, _, c in p["petri"]:
        k = split_target(key)[0]
        P(f"#    mkdir -p $HOME/.local/state/sdfbase_full22/petri/{k} && systemd-run --user --unit {name} --collect "
          f"--working-directory {args.grading_host_worktree} /usr/bin/bash -c "
          f"\"exec bash {S}/petri_drive_full22.sh {kv(key)} {c} {sha} >> $HOME/.local/state/sdfbase_full22/petri/{k}/autostart.log 2>&1\"")
    P("# --- watchers (arm each in the SAME turn as its submission; a job is not launched until it has survived ~2 min):")
    for c in CLUSTERS:
        if n_by_cluster[c]:
            P(f"#    ~/scripts/watch-job.sh --slurm <jobid> --host {where(c)['host']}   # every {c} job above")
    P("# --- stopping: touch $SHARED/exp/sdfbase_full22/STOP is WAVE-WIDE — it also stops any gemma/qwen leg still running")
    P("#     (same WAVE_ROOT); prefer the per-job STOP under exp/sdfbase_full22/serving/<job dir>/ (or petri/<tag>/serving/),")
    P("#     THEN scancel by name.")
    n_legs = sum(j[2] for j in p["petri"])
    from sisteer import coordination
    if coordination.channel():
        P(f"# --- the {coordination.channel()} post (print, then proceed; Low Prio; throttle down on 429s):")
        P(f"#     {n_legs} sonnet-5 lowprio 12h — Petri auditors for the full22 nemotron/glm legs: {n_legs} legs at width 1 (one auditor")
        P(f"#     conversation per leg; hard cap 50 TOTAL across every leg incl. any gemma/qwen leg still running); opus-5 judge rides")
        P("#     the same pool at the same width. Grading afterwards: 200 haiku-4-5 lowprio 3h.")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
