#!/usr/bin/env python
"""full22 SAMPLED re-run: PRINT (never run) the sbatch commands for the four targets' sampled jobs, in launch order,
AFTER checking every precondition a job would otherwise discover on a GPU (the submit_newmodels.py pattern; its
read-only ssh probe is IMPORTED, not copied):

  1. OFFLINE   make_full22_configs.py --check (the wave configs are the roster's) and make_sampled_configs.py --check
               (the 16 sampled configs are the wave configs + the preset file, byte-identical) — a drifted config
               never prints;
  2. ON EACH CLUSTER a job is planned for (read-only, over ssh, unless --offline): the checkout
               $SHARED/eval-repo/<wave_sha> (HEAD == the sha), the target's HF snapshot at the roster's
               pinned revision, `make_bank_view.py --check` for the target's `main` view (a SHORT / stale / missing
               view refuses the whole print), this user's RUNNING + PENDING `high` GPUs against the cluster's budget
               (cluster A 16; cluster B 96 on its own `high` QoS, see the README), and the wave
               STOP sentinel. An unreachable cluster is a refusal, not a pass.
  3. The plan: a table, then the FIRST BLOCK (static base per trial — split by family pair for the two slow
               targets — and controls per trial) as `ssh <host> 'cd <checkout> && sbatch ...'` lines, the excess
               over a cluster's free budget as `# HOLD` comment lines (a PENDING job counts against the budget,
               so nothing is queued past it); then the SECOND BLOCK (the static 8192 rung, `static_esc_t<T>`),
               to be submitted BY HAND once that target's base job(s) of the same trial are COMPLETE — no Slurm
               dependency on purpose; then the THIRD BLOCK (2026-09-10): per trial the capability battery
               (`capability_t<T>`: GSM8K thinking-ON @8192 sampled only + MMLU) FIRST, then the 16,384 extend
               (`ladder_extend_t<T>`: reclassify, decide at 8192, climb only the promoted pairs, the ceiling report) —
               the extend reads the GSM8K root the capability job writes, and the static / controls ladders of the
               same trial must be COMPLETE (se<T> and ctl<T> done); then the watcher lines.

Job names: f22s-<key>-sb<T>[a|b] (static base; a = pasttense,codeattack, b = renellm,grok420 for qwen35_4b and
glm46vflash), f22s-<key>-ctl<T> (controls, base + ladder), f22s-<key>-se<T> (static escalation), f22s-<key>-cap<T>
(capability battery), f22s-<key>-le<T> (the 16,384 extend). Default routing: gemma4e4b + qwen35_4b -> cluster_a,
nemotron3nano4b + glm46vflash -> cluster_b, every non-main TAG -> cluster_b. --time: static 16:00:00, controls 24:00:00 (the 8192 rung on
heavily-capped cells is the long pole), escalation 16:00:00, capability 12:00:00, extend 16:00:00.

THE NON-MAIN VIEWS — every TAG of make_sampled_configs.SAMPLED_VIEWS beyond the four keys (the ONE scope list; nothing is
restated here): the two instruct-direct TAGs gemma4e4b_it / qwen35_4b_it (2026-09-09), the strength frontier's 24
per-dose TAGs <key>_d<dose> / <key>_it_d<dose> (2026-09-10; e.g.
nemotron3nano4b_d0p25, gemma4e4b_it_d1) and its twelve llama8b dose TAGs llama8b_d<dose> / llama8b_it_d<dose> (2026-09-13; the
bare key `llama8b` and `llama8b_it` are NOT in the scope, so both refuse here). --targets
accepts any of them; each is planned as its OWN unit with the cell count
the roster gives that view — the static base is ONE job per trial (the a/b family split is the two slow MAIN views' only),
plus controls per trial and the static escalation per trial: f22s-<TAG>-sb<T> / -ctl<T> / -se<T>. The printed sbatch line
passes the bare model key and `9 "" <view>` as the trailing positionals (nshards, an EMPTY cells arg = all, the view), and
EVERY non-main TAG routes to cluster B (`cluster_b`) by default (its 96-GPU `high` budget; cluster A's 16 is the main views'
and is spoken for); the probe checks the (key, view) bank view. None of them is in the default --targets (the main-view jobs
were already live when the views were added; ask for them explicitly).

Usage:
  submit_sampled.py <wave_sha> [--targets k1,k2 | gemma4e4b_it,qwen35_4b_it | nemotron3nano4b_d0p25,...] [--cluster <key>=cluster_a|cluster_b ...]
      [--cluster-a-budget N] [--cluster-b-budget N]   override the probed free `high` budget
      [--exclude NODES | --exclude <cluster>:NODES ...]  sbatch --exclude=<nodes> on every printed line (of that cluster)
      [--nshards 9] [--offline] [--roster <yaml>]
Nothing here executes sbatch; ssh runs read-only probes; a refusal exits 2 with the CHECK lines.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent            # scripts/sdfbase/full22/sampled
F22 = HERE.parent                                 # scripts/sdfbase/full22
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(F22))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(F22 / "static_padding"))
from f22_roster import served_cells, target_caps, view_controls, view_legs, view_serving, view_windows  # noqa: E402
import padding_preflight as pf  # noqa: E402  (selected_module_set: the ONE source of the SELECTED LoRA module set — reproduce/models.yaml)
from submit_newmodels import parse_checks, probe  # noqa: E402  (the read-only ssh probe, reused)
import make_sampled_configs as msc  # noqa: E402
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import clusters as CL  # noqa: E402  (the site's cluster blocks: each cluster's login host, directories and QoS)

S = "scripts/sdfbase/full22"
SS = "scripts/sdfbase/full22/sampled"
#: the sdfbase-llama8b LANE's key: its main-view TAG is the bare key `llama8b`, planned under LLAMA_TAGS (the lane's own group, no greedy
#: anchor — GREEDY_STATIC_HOURS), never one of the default main-view KEYS
LLAMA_LANE_KEYS = ("llama8b",)
#: the default --targets: the main-view keys of the PRODUCTION scope (make_sampled_configs.SAMPLED_VIEWS — the ONE list,
#: derived, never restated) minus the llama8b keys — the four keys whose greedy pass anchors the estimates
KEYS = tuple(k for k, vs in msc.SAMPLED_VIEWS.items() if "main" in vs and k not in LLAMA_LANE_KEYS)
#: every non-main TAG of the scope (make_sampled_configs.sampled_tags() — the ONE list: SAMPLED_VIEWS, then the front door's views the
#: roster names): accepted by --targets, planned as their own units, never in the default. IT_TAGS = the two instruct-direct views
#: (lane F); DOSE_TAGS = the strength frontier's 24 per-dose views (`<key>_d<dose>` / `<key>_it_d<dose>`, 2026-09-10); SEEDREP_TAGS = the
#: qwen-base-seed-replicates lane's three seed-replicate views of qwen35_4b (`qwen35_4b_s<s>`, promoted into the scope 2026-09-15);
#: FRONT_DOOR_TAGS = the front door's phase views (paper-companion-repo; in scope once the production roster carries them) — each its own
#: group, so the dose derivation stays exactly the frontier's set
VIEW_TAGS = tuple(t for t in msc.sampled_tags() if t not in KEYS)
#: the llama8b target's four TAGs (its main-view TAG is the bare key `llama8b`, which is not one of the four preset KEYS):
#: both of its views, the FULL-constitution appendix view and the SHORT constitution-SI arm's view — their own group, so
#: the it / dose derivations below stay exactly the instruct-direct and strength-frontier sets
LLAMA_TAGS = tuple(t for t in VIEW_TAGS if msc.split_tag(t)[0] in LLAMA_LANE_KEYS)
FRONT_DOOR_TAGS = tuple(t for t in VIEW_TAGS if msc.split_tag(t)[0] in msc.FRONT_DOOR_VIEWS)
IT_TAGS = tuple(t for t in VIEW_TAGS if msc.split_tag(t)[1] == "it" and t not in LLAMA_TAGS)
SEEDREP_TAGS = tuple(t for t in VIEW_TAGS if t in msc.seedrep_tags())
DOSE_TAGS = tuple(t for t in VIEW_TAGS if t not in IT_TAGS and t not in LLAMA_TAGS and t not in SEEDREP_TAGS and t not in FRONT_DOOR_TAGS)
TARGETS_ACCEPTED = KEYS + VIEW_TAGS
#: default routing: the main views as the live run had them (gemma / qwen on cluster A, nemotron / glm on cluster B);
#: EVERY non-main TAG — the it views, all 24 dose views and the llama8b target's four TAGs (its wave runs on cluster B
#: since 2026-09-11; its launcher passes --cluster explicitly anyway) — on cluster B, whose `high` budget is 96 GPUs per user
#: (cluster A's 16 could not hold even one trial of the dose views: 12 TAGs x (sb + ctl) = 48 first-block jobs)
DEFAULT_CLUSTER = {"gemma4e4b": "cluster_a", "qwen35_4b": "cluster_a",
                   "nemotron3nano4b": "cluster_b", "glm46vflash": "cluster_b",
                   **{t: "cluster_b" for t in VIEW_TAGS}}


def split_target(target: str) -> tuple:
    """A --targets entry -> (bare model key, view): a key is its main view; a TAG names another view (the generator's inverse
    of tag_of over the ONE scope list; a TAG outside it refuses)."""
    return msc.split_tag(target)
#: per-user `high` budget (RUNNING + PENDING GPUs) per cluster: cluster A's 16 is the documented per-user cap;
#: cluster B's 96 is the documented figure for its own `high` QoS (2026-09-09) — override with --cluster-b-budget
CLUSTERS = {"cluster_a": {"cap": 16}, "cluster_b": {"cap": 96}}


def where(cluster: str) -> dict:
    """The login host, eval-repo root, envs and log directories and QoS of `cluster` (a registry label), from its block in the
    site config (reproduce/lib/clusters.py); refused by name when the site config does not describe it. `sbatch` is the flags
    every job line carries before its --time (--qos, --partition, --account when set, --output under CLUSTER_LOG_DIR: the
    job scripts' own headers name one site's values) and `extra` the site's CLUSTER_SBATCH_EXTRA."""
    try:
        b = CL.block_for(cluster)
    except SystemExit as exc:
        raise SystemExit(f"REFUSING: {str(exc.code).removeprefix('REFUSED: ')}") from None
    logs = b["CLUSTER_LOG_DIR"].rstrip("/")
    flags = [f"--qos={b['CLUSTER_QOS']}", f"--partition={b['CLUSTER_PARTITION']}"]
    if (b.get("CLUSTER_ACCOUNT") or "").strip():
        flags.append(f"--account={b['CLUSTER_ACCOUNT'].strip()}")
    flags.append(f"--output={logs}/%x_%j.out")
    return {"host": b["CLUSTER_SSH"], "eval": b["CLUSTER_EVAL_REPO_ROOT"].rstrip("/"), "envs": b["CLUSTER_ENVS_DIR"].rstrip("/"),
            "logs": logs, "qos": b["CLUSTER_QOS"], "sbatch": " ".join(flags),
            "extra": (b.get("CLUSTER_SBATCH_EXTRA") or "").strip()}
#: the two targets whose greedy static pass was the slowest (qwen ~5.8 h, glm ~15 h): each trial's base static
#: pass is split into two jobs by family pair
SPLIT_STATIC = ("qwen35_4b", "glm46vflash")
FAMILY_PAIRS = (("a", "pasttense,codeattack"), ("b", "renellm,grok420"))
TIME = {"static_base": "16:00:00", "controls": "24:00:00", "static_esc": "16:00:00",
        #: the llama8b target's third block: the static 16,384 rung on the gate's ESCALATE
        "static_esc16": "16:00:00", "gsm8k": "08:00:00",
        #: lane G2 (2026-09-10): GSM8K thinking-ON @8192 on 200 items x cells + MMLU (no anchor: the greedy GSM8K ran at 512
        #: thinking-OFF); the 16,384 extend re-draws only the promoted pairs' truncated items at half the 8192 rung's width
        "capability": "12:00:00", "ladder_extend": "16:00:00"}
#: the standup windows per phase (DESIGN.md "Geometry"); S = the model budget at that window (geometry.seqs_at). The 16384
#: rung's window is per model (common_sampled.sh f22s_rung_window: 18432; nemotron3nano4b 16896) — the two 16,384 rows below
#: are the gate-based rows for a target WITH a caps block (a view with its own window serves it there: fullconst 51,200)
WINDOWS = {"static_base": 4864, "static_esc": 10240, "controls_512": 2048, "controls_2048": 4096, "controls_8192": 10240}
WINDOWS_CAPS_LANE = {"controls_16384": 18432,   # the 16,384 continuation: climbed only on the gate's ESCALATE
                     "static_esc16": 18432}     # the static 16,384 rung
WINDOW_GSM8K = 10240                            # the gsm8k readout phase (only where the roster keeps gsm8k)
#: the GREEDY wave's measured elapsed for one draw of the four static families per target (measured
#: 2026-09-09) — the only anchor for the sampled estimates; controls 18-51 min per target before the ladder
GREEDY_STATIC_HOURS = {"gemma4e4b": 3.2, "qwen35_4b": 5.8, "nemotron3nano4b": 3.3, "glm46vflash": 15.0,
                       # llama8b: NO greedy pass of this target exists — nothing to anchor on; stated as unmeasured
                       "llama8b": None}
N_TRIALS = 2


@dataclass
class Job:
    name: str
    key: str             # the --targets entry: a bare key (main view) or a TAG (gemma4e4b_it)
    phase: str
    cluster: str
    time: str
    block: str           # "base" (first block) | "esc" (second block) | "ext" (third block: capability + the 16,384 extend)
    est_h: str
    args: str = field(default="")
    view: str = field(default="main")
    model_key: str = field(default="")   # the bare model key the sbatch receives (== key for the main view)


def _args(key: str, view: str, phase: str, nshards: int) -> str:
    """The sbatch's positionals: <key> <phase> W [nshards] [cells] [view] — the view rides as the 6th positional behind
    an EMPTY cells arg (= all cells); the main view passes nothing beyond nshards (byte-identical to before)."""
    base = f"{key} {phase} W {nshards}"
    return base if view == "main" else f'{base} "" {view}'


class _Budget:
    """The three facts plan() reads of a model's geometry, from the resolver: `seqs_at(L)` (the model's OWN row when it has one — measured,
    provisional or onboarded — else the Llama-certified default capped at the hybrid ceiling when rows_for says hybrid), `hybrid`,
    `batched_token_budget`, and the budget's `source`."""

    def __init__(self, rb, rows):
        self.hybrid, self.batched_token_budget, self.source, self._rb = bool(rows.hybrid), int(rb.batched_token_budget), rb.source, rb

    def seqs_at(self, L: int) -> int:
        from sisteer.capeval.serving import geometry as g
        if self._rb.per_model:
            return int(self._rb.entry.seqs_at(int(L)))
        s = int(g.identity_seqs(int(L), budget=self.batched_token_budget))
        return min(s, g.HYBRID_SEQS_CAP) if self.hybrid else s


def plan(roster: dict, targets: list, cluster_of: dict, nshards: int = 9) -> dict:
    """The job shapes, derived from the roster (cells) + the routing. No ssh, no sbatch."""
    from sisteer.capeval.serving import geometry as g

    served, jobs, esc, esc16, ext, skipped = [], [], [], [], [], []
    for tag in targets:
        key, view = split_target(tag)
        t = roster["targets"][key]
        n = 1 + len(served_cells(roster, key, view))
        if n < 2:
            skipped.append(f"# SKIP {key}/{view}: config lists `none` only (roster: enable cells, regenerate)")
            continue
        # THE BUDGET RESOLVER: geometry.budget_for / rows_for — the ONE lookup over the sweep's keys
        # AND the paper-companion onboarded ones, never BUDGET_BY_MODEL indexed directly; budgets / hybrid facts are the MODEL's, whichever
        # view is served. An unknown key refuses by name (the front door submits its sampled units through 06_eval_waves.sh anyway).
        # An ONBOARDED model: the resolver is asked for the SELECTED module set — the models table's
        # promoted `lora_architecture` (padding_preflight.selected_module_set, the same one source the padded pins read) — so the site
        # geometry overlay's certified row for THAT set sizes the launch, never the provisional ceiling. A view that serves STEERED cells
        # (a staged object id) with no promoted pick refuses by name; an unsteered view (u0: base + prompted only) sizes from the built-in
        # row and says so.
        module_set = None; set_note = None; steered = []; _sel = {}
        if key in g.ONBOARDED_MODEL_KEYS:
            steered = sorted(c for c, sid in served_cells(roster, key, view).items() if sid is not None)
            try:
                module_set, _sel = pf.selected_module_set(key)
            except ValueError as e:
                if steered:
                    raise SystemExit(f"{tag}: {e} — the steered cells {steered} size from the SELECTED module set's certified geometry "
                                     "(site geometry overlay, D66/D69)") from None
                set_note = (f"# NOTE {tag}: unsteered view, no promoted LoRA pick yet ({e}) — sized through the resolver's unsteered default "
                            f"(geometry.UNSTEERED_MODULE_SET {g.UNSTEERED_MODULE_SET!r}: the site record's row for the zero adapter's class when the "
                            "model has a record, else the built-in row)")
        try:
            rows, rb = g.rows_for(key, module_set), g.budget_for(key, module_set)
        except g.UnknownModelKey as e:
            raise SystemExit(f"{tag}: {e} — this planner sizes every phase from the model's geometry row") from None
        except g.SiteGeometryError as e:
            raise SystemExit(f"{tag}: {e} — the selected module set {module_set!r} has no certified geometry (re-run the serving probe / "
                             "03_hpo.sh --phase probe for that set)") from None
        if module_set is not None and steered and rows.site is not None:
            # the steered cells serve the pick's trained adapter — its adapter parity in the overlay must be 'verified'; an unsteered
            # view (u0) reads the same record whatever the block says
            try:
                g.require_verified_adapter_parity(rows.site, _sel.get("rank"), what=f"{tag}: the steered cells {steered}")
            except g.SiteGeometryError as e:
                raise SystemExit(f"{tag}: {e}") from None
        if set_note:
            skipped.append(set_note)
        b = _Budget(rb, rows)
        # a view with its own serving window (llama8b/fullconst 45,056) serves every phase at it (f22s_rung_window)
        L_view = view_serving(roster, key, view).get("max_model_len")
        vw = view_windows(roster, key, view)          # per-leg windows (the llama8b shortconst view, 2026-09-12) or the table's
        geom = {ph: ((L_view, b.seqs_at(L_view)) if L_view else (L, b.seqs_at(L))) for ph, L in WINDOWS.items()}
        if target_caps(roster, key) is not None:
            # the gate-based 16,384 rows ride a target WITH a caps block only; a rung the
            # view's window cannot carry stands up at the rung's OWN window (f22s_rung_window 16384)
            L16 = (L_view if L_view >= 51200 else 51200) if L_view else None
            for ph, L in WINDOWS_CAPS_LANE.items():
                geom[ph] = (L16, b.seqs_at(L16)) if L16 else (L, b.seqs_at(L))
        if vw["kind"] == "per_leg":
            # a PER-LEG view: every phase at ITS window (f22s_base_window / f22s_rung_window read the same roster block)
            per = {"static_base": vw["static"], "controls_512": vw["controls"], "controls_2048": vw["rungs"][2048],
                   "static_esc": vw["rungs"][8192], "controls_8192": vw["rungs"][8192],
                   "static_esc16": vw["rungs"][16384], "controls_16384": vw["rungs"][16384]}
            for ph, L in per.items():
                if ph in geom:
                    geom[ph] = (L, b.seqs_at(L))
        if key == "llama8b" and "gsm8k" in view_controls(roster, key, view):   # the readout phase's row: the same condition as its job below
            geom["gsm8k"] = (L_view, b.seqs_at(L_view)) if L_view else (WINDOW_GSM8K, b.seqs_at(WINDOW_GSM8K))
        c = cluster_of[tag]
        served.append({"key": tag, "model_key": key, "view": view, "n": n, "cluster": c, "model": t["model"],
                       "revision": t["model_revision"], "hybrid": b.hybrid, "geom": geom, "budget": b.batched_token_budget, "module_set": module_set})
        gh = GREEDY_STATIC_HOURS.get(key)          # the paper's anchors; a key outside the table (a front-door model) is UNMEASURED, like llama8b
        kw = {"view": view, "model_key": key}
        legs = view_legs(roster, key, view)
        if gh is None:
            est_sb = "UNMEASURED (no greedy pass of this target; an 8B dense model at the ceiling's S — size --time generously)"
        elif view != "main":
            # the it view's own greedy pass was not timed here: the main view's anchor scaled by cell count, stated as such
            n_main = 1 + len(served_cells(roster, key, "main"))
            est_sb = f"~{gh * (n - 1) / (n_main - 1):.1f} (the main view's greedy {gh} h x {n - 1}/{n_main - 1} cells; not measured)"
        else:
            est_sb = f"~{gh} (the greedy pass)"
        gh_half = f"~{gh / 2:.1f} (half the greedy {gh} h)" if gh is not None else est_sb
        # THE THIRD BLOCK follows whichever rule owns the target's 16,384 rung: a target WITH a
        # caps block (llama8b) climbs on the GATE (static_esc16_t<T>: ladder_gate.py over the 8,192 store;
        # its controls ladder already carries the 16,384 rung under the same gate inside controls_t<T>); every other target
        # takes the capability-then-extend path (capability_t<T> then ladder_extend_t<T>). Never both for one target: two
        # rules would climb the same _cap16384 roots.
        caps_lane = target_caps(roster, key) is not None
        if "static" in legs:   # per trial: the base pass (first block), its 8,192 rung (second), its 16,384 rung (third)
            for T in range(N_TRIALS):
                if tag in SPLIT_STATIC:
                    for suffix, fams in FAMILY_PAIRS:
                        jobs.append(Job(f"f22s-{tag}-sb{T}{suffix}", tag, f"static_base_t{T}:{fams}", c, TIME["static_base"],
                                        "base", gh_half, _args(key, view, f"static_base_t{T}:{fams}", nshards), **kw))
                else:
                    jobs.append(Job(f"f22s-{tag}-sb{T}", tag, f"static_base_t{T}", c, TIME["static_base"],
                                    "base", est_sb, _args(key, view, f"static_base_t{T}", nshards), **kw))
        if "controls" in legs:   # no over-refusal leg (the llama8b fullconst view): no ladder job
            for T in range(N_TRIALS):
                jobs.append(Job(f"f22s-{tag}-ctl{T}", tag, f"controls_t{T}", c, TIME["controls"], "base",
                                "0.3-0.9 base + ladder (8192 rung = long pole)", _args(key, view, f"controls_t{T}", nshards), **kw))
        if "static" in legs:
            for T in range(N_TRIALS):
                esc.append(Job(f"f22s-{tag}-se{T}", tag, f"static_esc_t{T}", c, TIME["static_esc"], "esc",
                               "<= the base pass (at-cap rows only)", _args(key, view, f"static_esc_t{T}", nshards), **kw))
                if caps_lane:
                    # the THIRD block of the caps-table path: the static 16,384 rung, scheduled after its
                    # trial's 8,192 rung job; it exits 0 without a standup when the gate escalates no (family, cell) pair past 8,192
                    esc16.append(Job(f"f22s-{tag}-s16k{T}", tag, f"static_esc16_t{T}", c, TIME["static_esc16"], "esc16",
                                     "<= the 8192 rung (the gate's escalating items only; 0 without an ESCALATE)",
                                     _args(key, view, f"static_esc16_t{T}", nshards), **kw))
        if not caps_lane:
            # THIRD BLOCK (lane G2): per trial the capability battery FIRST (the extend reads its GSM8K root), then the extend
            for T in range(N_TRIALS):
                ext.append(Job(f"f22s-{tag}-cap{T}", tag, f"capability_t{T}", c, TIME["capability"], "ext",
                               "GSM8K thinking-ON 8k x cells + MMLU (not measured)", _args(key, view, f"capability_t{T}", nshards), **kw))
                ext.append(Job(f"f22s-{tag}-le{T}", tag, f"ladder_extend_t{T}", c, TIME["ladder_extend"], "ext",
                               "<= the 8192 rung (truncated items of promoted pairs only; zero promotions = minutes)",
                               _args(key, view, f"ladder_extend_t{T}", nshards), **kw))
        # the GSM8K readout (LADDER_RULES.md §6): the vendor-preset arm per trial and, in
        # trial 0, the greedy comparator arm on the same items — ONLY for a view with the capability leg. The four-
        # target run's gsm8k stores were made by another launcher; this printer plans the phase for the llama
        # views (and would for any other view asked for explicitly)
        # ... ONLY when the view's controls list still carries gsm8k (f22_roster.view_controls): the llama8b views
        # DROPPED gsm8k on 2026-09-10 (one GSM8K instrument for the campaign; roster capability_omitted),
        # so no gsm job is planned for them — the omission is recorded in the roster / wave manifest, never inferred
        if "capability" in legs and key == "llama8b" and "gsm8k" in view_controls(roster, key, view):
            for T in range(N_TRIALS):
                jobs.append(Job(f"f22s-{tag}-gsm{T}", tag, f"gsm8k_t{T}", c, TIME["gsm8k"], "base",
                                "0.5-2 (200 items x cells x arms at cap 8192)", _args(key, view, f"gsm8k_t{T}", nshards), **kw))
    return {"served": served, "jobs": jobs, "esc": esc, "esc16": esc16, "ext": ext, "skipped": skipped}


def layout(p: dict, budgets: dict) -> dict:
    """Per cluster: the first-block jobs that FIT the free `high` budget (submit now, in order) and the rest (HOLD:
    submit as budget frees — a PENDING job counts against the cap, so nothing is queued past it)."""
    out = {}
    for c in CLUSTERS:
        here = [j for j in p["jobs"] if j.cluster == c]
        if not here and not any(j.cluster == c for j in p["esc"] + p.get("esc16", []) + p.get("ext", [])):
            continue
        fit = max(0, int(budgets.get(c, 0)))
        out[c] = {"submit": here[:fit], "hold": here[fit:], "esc": [j for j in p["esc"] if j.cluster == c],
                  "esc16": [j for j in p.get("esc16", []) if j.cluster == c],
                  "ext": [j for j in p.get("ext", []) if j.cluster == c], "budget": budgets.get(c)}
    return out


def parse_excludes(flags: list) -> dict:
    """--exclude <node>  (every cluster)  |  --exclude <cluster>:<node>,<node>  (one cluster) -> {cluster: "n1,n2"}"""
    out = {c: [] for c in CLUSTERS}
    for f in flags:
        cl, sep, nodes = f.partition(":")
        if sep and cl in CLUSTERS:
            out[cl] += [n for n in nodes.split(",") if n]
        elif sep:
            raise SystemExit(f"--exclude {f!r}: cluster must be one of {sorted(CLUSTERS)}")
        else:
            for c in CLUSTERS:
                out[c] += [n for n in f.split(",") if n]
    return {c: ",".join(dict.fromkeys(v)) for c, v in out.items() if v}


def sbatch_line(job: Job, sha: str, excludes: dict) -> str:
    w = where(job.cluster)
    extra = f" {w['extra']}" if w["extra"] else ""
    excl = f" --exclude={excludes[job.cluster]}" if job.cluster in excludes else ""
    args = job.args.replace(" W ", f" {sha} ")
    return (f"ssh {w['host']} 'cd {w['eval']}/{sha} && sbatch {w['sbatch']}"
            f" --time={job.time}{extra}{excl} -J {job.name} {SS}/target_sampled.sbatch {args}'")


def render(p: dict, lay: dict, sha: str, excludes: dict, offline: bool) -> list:
    out = []
    P = out.append
    n_first = sum(len(v["submit"]) + len(v["hold"]) for v in lay.values())
    n_esc = sum(len(v["esc"]) for v in lay.values())
    n_ext = sum(len(v["ext"]) for v in lay.values())
    n_esc16 = sum(len(v["esc16"]) for v in lay.values())
    P(f"# full22-sampled @ {sha}: {len(p['served'])} targets ({', '.join(f"{s['model_key']}/{s['view']}={s['n']} cells" for s in p['served'])}); "
      f"{n_first} first-block jobs (static base x2 trials{' (split a/b for ' + ', '.join(SPLIT_STATIC) + ')' if any(s['key'] in SPLIT_STATIC for s in p['served']) else ''} + controls x2), "
      f"{n_esc} second-block jobs (static 8192 rung x2) and {n_ext + n_esc16} third-block jobs "
      f"({n_ext} capability battery x2 + the 16,384 extend x2 for the targets without a caps block; {n_esc16} static 16,384 rung x2 for the caps-table lane, each after its trial's 8192 rung job)")
    for c, v in lay.items():
        n = len(v["submit"]) + len(v["hold"])
        P(f"# GPUs {c}: {n} first-block jobs of a free `{where(c)['qos']}` budget {v['budget']}"
          + (f"  ** {len(v['hold'])} HELD: over budget — submit as jobs complete (a PENDING job counts against the cap) **" if v["hold"] else "")
          + f"; +{len(v['esc'])} second-block and +{len(v['ext'])} third-block jobs later (each takes a GPU of the same budget when submitted)")
    P("# Geometry per target (window x S from geometry.budget_for(model[, selected module set]).seqs_at; the standup ATTESTS both and refuses a clamp):")
    for s in p["served"]:
        g = s["geom"]
        P(f"#   {s['key']}: budget {s['budget']:,}{' HYBRID (S capped at 256)' if s['hybrid'] else ''}{' module set ' + s['module_set'] if s.get('module_set') else ''}; "
          f"static base {g['static_base'][0]}x{g['static_base'][1]} (drivers capped at 96), "
          f"static 8192 rung {g['static_esc'][0]}x{g['static_esc'][1]} (32/driver), "
          + (f"static 16384 rung {g['static_esc16'][0]}x{g['static_esc16'][1]} (the gate-based lane), " if "static_esc16" in g else "")
          + f"controls 512 {g['controls_512'][0]}x{g['controls_512'][1]}, 2048 rung {g['controls_2048'][0]}x{g['controls_2048'][1]}, "
          f"8192 rung {g['controls_8192'][0]}x{g['controls_8192'][1]}"
          + (f", 16384 rung {g['controls_16384'][0]}x{g['controls_16384'][1]}" if "controls_16384" in g else "")
          + (f"; gsm8k readout {g['gsm8k'][0]}x{g['gsm8k'][1]}" if "gsm8k" in g else ""))
    P("# Wall-clock: the GREEDY wave's measured static pass per target is the only anchor (gemma 3.2 h, qwen 5.8 h, nemotron 3.3 h,")
    P("# glm 15 h for one draw of the four families; controls 18-51 min before the ladder); a sampled draw at the same cap is")
    P("# assumed comparable — NOT measured. The controls ladder's 8192 rung on heavily-capped cells is the long pole (--time 24h).")
    if offline:
        P("# (--offline: checkout / cache / view / budget / STOP were NOT probed; the budgets are the flags or the documented defaults)")
    P("")
    P("| cluster | job | GPUs | est. wall-clock (h) | block | args |")
    P("|---|---|---|---|---|---|")
    for c, v in lay.items():
        for j in v["submit"]:
            P(f"| {c} | {j.name} | 1 | {j.est_h} | 1 (submit) | {j.args.replace(' W ', ' ' + sha[:12] + ' ')} |")
        for j in v["hold"]:
            P(f"| {c} | {j.name} | 1 | {j.est_h} | 1 (HOLD) | {j.args.replace(' W ', ' ' + sha[:12] + ' ')} |")
        for j in v["esc"]:
            P(f"| {c} | {j.name} | 1 | {j.est_h} | 2 (after sb{j.phase[-1]}) | {j.args.replace(' W ', ' ' + sha[:12] + ' ')} |")
        for j in v["esc16"]:
            P(f"| {c} | {j.name} | 1 | {j.est_h} | 3 (after se{j.phase[-1]}) | {j.args.replace(' W ', ' ' + sha[:12] + ' ')} |")
        for j in v["ext"]:
            when = f"3 (after ctl{j.phase[-1]} + se{j.phase[-1]})" if j.phase.startswith("capability") else f"3 (after cap{j.phase[-1]})"
            P(f"| {c} | {j.name} | 1 | {j.est_h} | {when} | {j.args.replace(' W ', ' ' + sha[:12] + ' ')} |")
    P("")
    P("# --- 1. FIRST BLOCK — static base (both trials) + controls (both trials), in this order. Every job stands up its own")
    P("#        server on ONE GPU; the sampled configs are committed at the wave sha (make_sampled_configs.py --check passed).")
    for c, v in lay.items():
        for j in v["submit"]:
            P(sbatch_line(j, sha, excludes))
        for j in v["hold"]:
            P("# HOLD (over the free budget on " + c + "; submit when a job above completes): " + sbatch_line(j, sha, excludes))
    P("# --- 2. SECOND BLOCK — the static 8192 RUNG per trial (static_esc_t<T>: only the (family, cell) pairs with at-cap rows in")
    P("#        the base run are re-drawn; a trial with none exits 0 without a standup). Submit BY HAND once EVERY sb<T> job of")
    P("#        that target and trial is COMPLETED (both a/b halves for the split targets). No --dependency on purpose: a PENDING")
    P("#        job counts against the `high` budget for as long as it waits.")
    for c, v in lay.items():
        for j in v["esc"]:
            P(sbatch_line(j, sha, excludes))
    P("# --- 3. THIRD BLOCK — the 16,384 rung, by the lane that owns the target (never both for one target):")
    n_a = sum(len(v['esc16']) for v in lay.values()); n_b = sum(len(v['ext']) for v in lay.values())
    # each lane's block is described ONLY when this plan carries its jobs (a plan over the caps-table views alone must not
    # read as planning lane G2's GSM8K phases — the launcher's plan check greps this text for a gsm8k phase)
    if n_a:
        P("#      (a) targets WITH a caps block (the sdfbase-llama8b lane): the static 16,384 RUNG per trial (static_esc16_t<T>; review round 2 P1):")
        P("#        ladder_gate.py static reads the 8,192 store's item-filtered 1 % truncated verdict + the transport approval and re-draws EXACTLY")
        P("#        the escalating items with the committed _cap16384 config at f22s_rung_window 16384 (fullconst: 51,200). Submit BY HAND once that")
        P("#        target and trial's se<T> job is COMPLETED; a trial the gate escalates nothing for exits 0 without a standup. The controls")
        P("#        ladder of these targets already carries its 16,384 rung under the same gate inside ctl<T>. No --dependency (as above).")
    if n_b:
        P("#      (b) every other target (lane G2, 2026-09-10) — per trial: the CAPABILITY battery (capability_t<T>: GSM8K THINKING-ON at the 8,192")
        P("#        cap, SAMPLED ONLY — no greedy arm, S-D7 — into sampled/gsm8k_thinking_on_cap8192/<TAG>_t<T>, then MMLU thinking-OFF into")
        P("#        sampled/mmlu/<TAG>_t<T>), THEN the 16,384 EXTEND (ladder_extend_t<T>: item_filtered_v1 reclassify of every rung, the")
        P("#        decision at 8192 -> ladders/<TAG>_t<T>/decision_16384.json, the promoted pairs only, the ceiling report; nothing past")
        P("#        16,384 — S-D4). Submit cap<T> once every ctl<T> and se<T> of that target and trial is COMPLETED; submit le<T> once")
        P("#        cap<T> is COMPLETED (it reads the GSM8K root). Both need a checkout at or after the thinking-ON standup (W10+).")
        P("#        A job with zero promotions files the decision and exits in minutes. No --dependency on purpose (the budget).")
    if not (n_a or n_b):
        P("#        (no third-block job in this plan)")
    for c, v in lay.items():
        for j in v["esc16"] + v["ext"]:
            P(sbatch_line(j, sha, excludes))
    P("# --- watchers (arm each in the SAME turn as its submission; a job is not launched until it has survived ~2 min):")
    for c in lay:
        P(f"#    ~/scripts/watch-job.sh --slurm <jobid> --host {where(c)['host']}   # every {c} job above")
    P("# --- stopping: touch $SHARED/exp/sdfbase_full22/STOP is WAVE-WIDE (it also stops any other full22 job on that cluster);")
    P("#     prefer the per-job STOP under $SHARED/exp/sdfbase_full22/sampled/serving/<phase>_<tag>_<jobid>/STOP, THEN scancel by name.")
    P("# --- grading afterwards (the grading host, Low Prio; README 'Grading plan'): attacks -> haiku-4-5 StrongREJECT v1 via the GLM lane's")
    P("#     grader shape; controls -> served_capability_judge.py, pass 1 on the 512 root, pass 2 on each cell's FINAL rung root")
    P("#     with an EMPTY private cache (CAPEVAL_JUDGE_CACHE=<fresh sqlite>) — the matched-pair label-flip protocol.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("wave_sha")
    ap.add_argument("--targets", default=",".join(KEYS))
    ap.add_argument("--cluster", action="append", default=[], metavar="KEY=CLUSTER")
    ap.add_argument("--cluster-a-budget", type=int, default=None)
    ap.add_argument("--cluster-b-budget", type=int, default=None)
    ap.add_argument("--exclude", action="append", default=[], metavar="[CLUSTER:]NODES")
    ap.add_argument("--nshards", type=int, default=9)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--roster", type=Path, default=REPO / "configs/sdfbase_full22_roster.yaml")
    args = ap.parse_args(argv)
    sha = args.wave_sha
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        print(f"REFUSING: wave sha must be a full 40-hex commit, got {sha!r}"); return 2
    roster = yaml.safe_load(args.roster.read_text(encoding="utf-8"))
    targets = [t for t in args.targets.split(",") if t]
    bad = [t for t in targets if t not in TARGETS_ACCEPTED]
    if bad:
        print(f"REFUSING: --targets {bad} not in the sampled scope {TARGETS_ACCEPTED} (make_sampled_configs.SAMPLED_VIEWS: DESIGN.md's "
              "four main views, the two instruct-direct TAGs of the 2026-09-09 scope addition, the strength frontier's 24 per-dose "
              "TAGs of 2026-09-10, its twelve llama8b dose TAGs of 2026-09-13, and the sdfbase-llama8b lane's four TAGs)"); return 2
    cluster_of = {k: DEFAULT_CLUSTER[k] for k in targets}
    for kv in args.cluster:
        k, _, c = kv.partition("=")
        if k not in targets or c not in CLUSTERS:
            print(f"REFUSING: --cluster {kv!r}: key must be one of {targets}, cluster one of {sorted(CLUSTERS)}"); return 2
        cluster_of[k] = c
    try:
        excludes = parse_excludes(args.exclude)
    except SystemExit as e:
        print(f"REFUSING: {e}"); return 2
    out = []
    P = out.append

    # 1. offline: the wave configs are the roster's, the sampled configs are the wave configs + the preset file
    for label, script in (("configs", REPO / "scripts/sdfbase/make_full22_configs.py"),
                          ("sampled configs", HERE / "make_sampled_configs.py")):
        gen = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        if gen.returncode != 0:
            print(f"REFUSING: {label} drift — regenerate first:")
            print(f"  python {script.relative_to(REPO)} && python {script.relative_to(REPO)} --check")
            print((gen.stdout + gen.stderr).strip()[-2000:]); return 2
        P(f"# CHECK {label.replace(' ', '-')} - OK {script.name} --check: every generated file matches its sources")

    p = plan(roster, targets, cluster_of, args.nshards)
    out += p["skipped"]
    if not p["served"]:
        print("\n".join(out)); print("REFUSING: nothing to serve"); return 2

    # 2. the clusters (read-only probes, via submit_newmodels.probe — checkout, venv, cache, view, budget, STOP)
    budgets, problems = {}, []
    for c in CLUSTERS:
        served_here = [s for s in p["served"] if s["cluster"] == c]
        if not served_here:
            continue
        if args.offline:
            P(f"# CHECK {c} - UNVERIFIED (--offline): checkout / cache / view / budget / STOP NOT probed")
            continue
        # the bank view a job consumes is (model key, view)'s: `key` for a main view, `key/view` otherwise (submit_newmodels.remote_script)
        lines = probe(c, sha, served_here, sorted({s["model_key"] if s["view"] == "main" else f"{s['model_key']}/{s['view']}" for s in served_here}))
        out += [f"# {l}" for l in lines]
        probs, used, _live = parse_checks(lines)
        problems += probs
        if used is not None:
            budgets[c] = CLUSTERS[c]["cap"] - used
            P(f"# CHECK budget {c} OK free={budgets[c]} (cap {CLUSTERS[c]['cap']} - used {used})")
    for c in CLUSTERS:   # explicit budgets win; offline falls back to the documented defaults
        flag = getattr(args, f"{c}_budget")
        if flag is not None:
            budgets[c] = flag
        budgets.setdefault(c, {"cluster_a": 11, "cluster_b": 96}[c])
    if problems:
        print("\n".join(out))
        print(f"REFUSING TO PRINT SUBMISSIONS: {len(problems)} precondition(s) failed:")
        for pr in problems:
            print(f"  {pr}")
        views_bad = [pr.split()[2] for pr in problems if pr.startswith("CHECK view ")]   # `key` or `key/view`, as probed
        for c in CLUSTERS:
            pairs = sorted({(s["model_key"], s["view"]) for s in p["served"] if s["cluster"] == c
                            and (s["model_key"] if s["view"] == "main" else f"{s['model_key']}/{s['view']}") in views_bad})
            for key, view in pairs:
                w = where(c)
                print(f"#   ssh {w['host']} 'cd {w['eval']}/{sha} && {w['envs']}/vllm-attacker/bin/python "
                      f"{S}/make_bank_view.py --target {key} --view {view} && {w['envs']}/vllm-attacker/bin/python "
                      f"{S}/make_bank_view.py --check --target {key} --view {view}'")
        return 2

    lay = layout(p, budgets)
    out += render(p, lay, sha, excludes, args.offline)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
