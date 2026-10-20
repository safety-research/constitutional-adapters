#!/usr/bin/env python3
"""``transfer_stats_full22_cap4096.json`` — the 4,096-token-reply-cap MULTI-TURN cells of the sdf-base-steering round-2 campaign
(the cap-4096 wave and its ActorAttack leg, the objects wave and its ActorAttack leg), in the schema of
``transfer_stats_full22_all.json``, under a NEW namespace ``cap4096`` — plus the two one-term-moved variants under
``cap4096_variant`` — written to a NEW file. Nothing of the wave's 512-cap files is read for writing, rewritten or replaced.

WHY THIS FILE EXISTS. The graded 4,096 legs never reached the tidy export the paper's figures read; its canonical view held zero
multi-turn rows because every raw multi-turn row sat at cap 512. The export merges this file ADDITIVELY and cap-keyed: the lane reuses the
SAME view keys as the 512 wave (``gemma4e4b``, ``qwen35_4b``, ...), so the half-dose merge pattern (``TS["models"][view] = block``, a
wholesale replace) would DELETE every 512 multi-turn row while adding these. See ``build_export.py`` (the MT4K block) for the merge.

WHAT IS READ, and how each root is keyed. Seventeen canonical grades roots (fourteen arm roots + the three unsteered ``none`` anchor roots
of 2026-09-13) + two variant roots under ``run-state/full22``; every root holds
exactly ONE target directory and that directory's name — never the root's — is the view (the ``grades_objaa4096rerun_<lane>`` roots hold
``<lane>_it`` targets, and this exact confusion has bitten before). The aggregates are the grader's own (``glm_lane/grade_full22_glm.py``:
``cells.<cell>.{attack_metrics, completion.gated, decode_cap_tokens, lens_reduction_mode, ...}``), read with the campaign's readers
(``transfer_stats.gated_cell`` etc., imported by path through the w7 builder — never re-typed).

THE CAP STAMP IS OVERRIDDEN, DELIBERATELY AND VERIFIABLY. Every aggregate row in these roots carries ``decode_cap_tokens: 512``: the grader
copies ``attacks.target_max_new_tokens`` out of the worktree CONFIG it is handed (``grading_provenance.decode_caps.target_max_new_tokens:
512`` on every root), and the 4,096 legs were driven with the cap on the sbatch command line, not in that yaml. The stamp is therefore the
config's, not the run's. This builder stamps ``decode_cap_tokens: 4096`` on every cell it emits and PROVES it from the producers' own
records, per cell: (a) the max ``completion_tokens`` over every ``target_timing.jsonl`` row of the cell's shards must lie in (512, 4096];
(b) the timing row count and the ``finish_reason == length`` count must equal the grader's own completion stamp
(``completion.gated.target_timing``) — the identity check that this transcripts store is the one that was graded; (c) the responses.jsonl
line count must equal ``completion.gated.n_responses``. The grader's 512 stamp is kept beside the override as ``decode_cap_tokens_grader_stamp``.
Only numbers are read from the transcripts (timing rows, line counts, ``served_target.json`` serving metadata) — never a prompt or a reply.

THE INSTRUMENT IS RECORDED PER CELL from ``served_target.json`` (uniform across shards, or this refuses): serving window (``max_model_len``),
window-clamp counts (must be 0 — the one clamped attempt in this wave, GLM/PAIR at an 8,704 window, failed its own completeness gate and was
never graded), and the decode actually sent (``last_request_decode``). That block is what makes the two variants legible as PAIRED variants
in which exactly one term moves: ``cap4096greedy`` (decode: greedy vs the target's shipped sampling) and ``cap4096w16k`` (serving window:
16,384). Each variant cell carries ``vs_canonical_sibling`` (variant minus canonical, paired on goals) when the sibling aggregate exists.

CONTRASTS. The unsteered ``none`` anchor at 4,096 (the ``grades_anchor4096_*`` roots, 2026-09-13) is exported as its own cell wherever its
aggregate exists — PARTIAL BY DESIGN while its legs land: an attack whose aggregate is absent gets NO row (nothing is fabricated), and a later
run picks up whatever has landed since (``meta.anchor_status`` says which). This builder still does not compute the same-cap cell-vs-bare
contrast (``vs_bare`` stays empty; wiring it is a follow-up that changes the arm rows), and the 512 unsteered row of the same view remains
the floor the rung collapse reads. What IS filed per cell × attack is
``vs_cap512``: the SAME cell's 4,096 reading minus its filed 512 reading, paired on goals (10,000-draw bootstrap seed 0 + exact McNemar,
``transfer_stats.paired_contrast`` via ``transfer_r2.contrast``) — truncation bias, not a defence effect. Where the 512 side dropped goals
(GLM's errored goals) the paired part is refused by the reader and recomputed on the goal INTERSECTION, labelled as such. The 512 side is
read from the grades root the export's join used for that (view, cell) and its k/n is cross-checked against that join file, or this refuses.

THE ARMS WAVE LANDS HERE TOO (mt_arms4096, 2026-09-14). The constitution-SI cell ``prompted__si_constitution`` (PROMPTED: no object) on
gemma4e4b_it / qwen35_4b_it / nemotron3nano4b / llama8b_shortconst arrives root by root as its grading chains finish, from EITHER cluster
(``grades_mt4k<arm>[_w24k]_<bu|ch>_<TAG>``, mt_arms4096/cluster_env.sh): every such root is a LANDING root (absent = recorded in
``meta.landing_status``, never a refusal), every row it yields carries ``served_cluster``, and the same (view, cell, attack, window) graded on
BOTH clusters is two readings, not a collision -- the first-written aggregate holds the view's cell slot, the other is filed under ``models_twins``
with the same record shape (``meta.twin_status``, TWIN_RULE). The PAIR legs that ran at a 24,576 serving window (nemotron, llama; each with its
same-window ``none`` anchor, arm nn) are the window VARIANT ``w24k`` under ``cap4096_variant``, keyed by window so they never pool with the
16,384 rows (their canonical sibling is the same view's 16,384 PAIR reading where one exists: the anchor root for ``none``; ABSENT for the SI cell).
The llama views are known here with the llama builder's spellings (llama8b, llama8b_it, llama8b_shortconst, llama8b_fullconst) and declare
NO_CAP512 per view: that lane has no 512-cap instrument at all. Pre-existing roots carry no served_cluster.json: their cluster (cluster_a, by their
pull scripts' host) rides ``meta.roots`` and ``meta.count_table``, NOT their filed records, which stay byte-identical (``meta.served_cluster_default``).

usage: build_transfer_stats_cap4096.py [--out <json>] [--rs <run-state/full22>]
       (run with .venv/bin/python — the readers need numpy)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

HOME = Path.home()
#: the lane's run-state tree on this host: $RUNSTATE (sisteer.site; SISTEER_RUN_STATE_ROOT from the environment, else the site config)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sisteer import site as _site  # noqa: E402

RS = _site.expand_path("$RUNSTATE/candidate-si-refit/run-state/full22", key="the lane's run-state tree")
W7 = _site.expand_path("$RUNSTATE/candidate-si-refit/w7_rest", key="the lane's run-state tree")
sys.path.insert(0, str(W7))
import build_transfer_stats_full22_all as B  # noqa: E402  (the w7 builder: readers, labels, block formatters — imported, never re-typed)

#: the pairing-allowance module (src/sisteer/eval/anchor_pairing.py): the allowance CONSTANT and
#: the verdict rule are IMPORTED, never retyped; an import failure REFUSES the build by name. Resolved from THIS FILE's repo root rather
#: than an absolute $HOME path, so the builder imports the module of the checkout it is actually run from.
ANCHOR_PAIRING_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ANCHOR_PAIRING_SRC))
try:
    from sisteer.eval import anchor_pairing as AP  # noqa: E402
except Exception as _e:  # noqa: BLE001
    raise SystemExit(f"REFUSING: cannot import sisteer.eval.anchor_pairing from {ANCHOR_PAIRING_SRC} ({type(_e).__name__}: {_e}) -- the "
                     f"relu_native pairing stamp is imported from the lane's module, never retyped; re-point ANCHOR_PAIRING_SRC when the branch lands")

T1, R2 = B.T1, B.R2
MULTITURN, WAVE_LENS_MODE = B.MULTITURN, B.WAVE_LENS_MODE
CAP = 4096
CAP_FILED = 512
NS, NS_VARIANT = "cap4096", "cap4096_variant"
PINNED_JUDGE, PINNED_RUBRIC = "claude-haiku-4-5", "strongreject_v1"

#: the unsteered `none` anchor at 4,096 (2026-09-13): one bare-target cell per view, its legs landing over hours. process_root reads whichever
#: two_si_<attack>_aggregate.json exists and emits nothing for an absent one, so a re-run yields strictly more rows as legs land — never
#: hard-code which attacks are present. Listed apart so meta.anchor_status can say, per run, which legs were found and which were not.
#: the unsteered `none` anchor roots at 4,096 -- ENUMERATED from disk, never a hand-written list (2026-09-16). The list below
#: named the three FIRST-attempt roots; a leg that died at standup and restarted in a fresh namespace (ROOT_SUFFIX=r2 ->
#: grades_anchor4096r2_<tag>, the wave's own rule) graded into a root nothing here read, and this builder reported the qwen
#: ActorAttack anchor as "absent (leg still running)" for two days after job 497644 had COMPLETED it. Two roots for one view
#: are legitimate and DISJOINT by attack (r1 holds the attacks that finished first time, r2 the restarted ones); the same
#: (view, attack, cell) in two roots is caught by the reading placer below, not silently overwritten.
ANCHOR_ROOTS = sorted({d.name for d in RS.glob("grades_anchor4096*_*") if d.is_dir()}
                      | {"grades_anchor4096_gemma4e4b_it", "grades_anchor4096_qwen35_4b_it", "grades_anchor4096_nemotron3nano4b"})
#: the qwen35_4b/it CONTROL LoRA + LoRA-SUBTRACTION legs at 4,096 (2026-09-13: six jobs 498351-498356, ONE cell per root, three
#: attacks each; tree = the config-only overlay 78385062 on the arms' 0088d863). Their grades roots are CREATED by the grading chains as
#: the legs finish, so an ABSENT root here is "no leg of this lane graded yet" -- recorded in meta.landing_status and skipped, never a
#: refusal; an absent ATTACK inside a present root is the anchor's partial-by-design case (meta.anchor_status grammar).
LANDING_ROOTS = ["grades_qc4kctl_qwen35_4b_it", "grades_qc4kdiff_qwen35_4b_it",
                 # the gemma4e4b/it half (same lane, after the view rebuild)
                 "grades_qc4kgemctl_gemma4e4b_it", "grades_qc4kgemdiff_gemma4e4b_it"]
#: the multi-turn ARMS wave (mt_arms4096; rulings 2026-09-14 00:40Z; README "Export producer" + cluster_env.sh e4n_root_tag): the constitution-SI
#: cell `prompted__si_constitution` (PROMPTED, no object) on the paper views gemma4e4b_it / qwen35_4b_it / nemotron3nano4b and on llama8b_shortconst
#: (the only llama view whose roster serves it), at the arms' canonical instrument -- crescendo + actorattack at 49,152, PAIR at 16,384 on gemma/qwen.
#: Root grammar:  grades_mt4k<arm>[_w24k]_<clu>_<TAG>   arm in {sc = the SI cell, nn = the bare `none` same-window PAIR anchor};
#:                clu in {bu = cluster_a, ch = cluster_b}; TAG = <key> for view main, <key>_<view> otherwise (TAG == the target directory == the view).
#: The SAME (lane, attack) may land from BOTH clusters (the 14 cluster-B legs of 00:31Z + their cluster-A twins; a race cancels one, "both running"
#: is kept) -- TWIN_RULE below. Every root here is a LANDING root: absent until its chain grades it, tolerated and recorded in meta.landing_status.
#: NOT registered (blocked on the arm port; README): the arditi `mt4kar_*` and alphasteer `mt4kas_*` arms, and any ROOT_SUFFIX restart namespace
#: (`mt4ksc_w24kr2_...`) -- register them here deliberately when they exist.
SI_CONST = "prompted__si_constitution"
MT_ARMS_TAGS = ["gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_shortconst"]
MT_ARMS_W24K_TAGS = ["nemotron3nano4b", "llama8b_shortconst"]          # the two lanes whose PAIR ran at 24,576 (with an nn anchor each)
CLUSTER_TOKEN = {"bu": "cluster_a", "ch": "cluster_b"}                # the root-name token -> the cluster (cluster_env.sh e4n_clu)
MT_ARMS_CELL = {"sc": SI_CONST, "nn": "none"}
MT_ARMS_ROOTS = [f"grades_mt4ksc_{clu}_{tag}" for tag in MT_ARMS_TAGS for clu in CLUSTER_TOKEN]
#: the Arditi / AlphaSteer arms (the arm port 68cb31a3 landed 2026-09-14 01:2xZ; llama arm tree 63f79fa5): registered when their legs were
#: QUEUED (ar cluster B 100988-100996 + 101053-101055, arb cluster A twins 500203-500212, as cluster A 500194-500202 + 500413-500415), so a
#: landing is never invisible. Cell ids in the aggregates are the arm names (`arditi`, `alphasteer`: the arm configs list [none, <arm>]); the
#: llama arm view is llama8b_it (the Arditi / AlphaSteer objects were fit on the it view), NOT llama8b_shortconst. Operating points as RULED
#: (2026-09-13 16:44 PDT, then RE-RULED 2026-09-14 after the defence wave found the top of every AlphaSteer ladder is
#: degeneration): Arditi alpha 0.05 on all four models (llama re-ruled from 0.1 on 09-14); AlphaSteer rung 2 gemma (from rung 1), rung 1 qwen
#: (from rung 3, whose `chosen` sat inside the degeneration zone), rung 2 nemotron (from rung 1, the a4_fallback), rung 1 llama (unchanged).
#: The declaration below is VERIFIED per root against the per-arm config each leg actually ran with (grading_provenance.config -> rung_i /
#: defenses_dir), so a ruling that moves without this table moving REFUSES the build instead of filing a measurement under the old label --
#: which is exactly what happened between 09-14 and 09-15: the four AlphaSteer lines sat at the 09-13 ruling while every leg re-ran at the
#: 09-14 one, and 68 export rows carried a rung their transcripts were never served at.
#: AlphaSteer ran on cluster A only (the arm check refuses cluster B); Arditi raced across both clusters (TWIN_RULE applies).
MT_ARMS_ARM_TAGS = ["gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it"]
ARM_CELL = {"ar": "arditi", "as": "alphasteer", "ap": "arditi_allpos"}
#: THE ROOT TOKEN -> ARM MAP, DECLARED (arditi-qwen-all-positions §9.3.3). The dispatch used to be the binary
#: expression `("arditi" if "mt4kar" in root_name else "alphasteer")`, so ANY new token resolved to one of the two
#: existing arms and its rows were filed under another arm's name -- loud at the operating-point check, wrong either
#: way. Matched LONGEST FIRST, and an unmatched arm root REFUSES rather than defaulting. A further arm is one entry
#: here plus its tags/clusters below: a table entry, never another special case.
ARM_ROOT_PREFIX = {"mt4kar": "arditi", "mt4kas": "alphasteer", "mt4kap": "arditi_allpos"}
#: which (tags, clusters) each arm's roots may exist for. arditi/alphasteer ran the four paper views (AlphaSteer on
#: cluster A only -- its arm check refuses cluster B); `arditi_allpos` is qwen35_4b ONLY (Qwen only, for now)
#: and races across both clusters like the arditi arm.
ARM_ROOT_TAGS = {"mt4kar": None, "mt4kas": None, "mt4kap": ["qwen35_4b_it"]}       # None = MT_ARMS_ARM_TAGS
ARM_ROOT_CLUSTERS = {"mt4kar": None, "mt4kas": ["bu"], "mt4kap": None}             # None = every CLUSTER_TOKEN


def arm_of_root(root_name: str) -> str:
    """The arm a grades root belongs to, by LONGEST declared prefix match. An unmatched root refuses BY NAME: a root
    nobody declared must never be filed under whichever arm an `if/else` happened to end on."""
    for tok in sorted(ARM_ROOT_PREFIX, key=len, reverse=True):
        if tok in root_name:
            return ARM_ROOT_PREFIX[tok]
    raise SystemExit(f"REFUSING: grades root {root_name!r} matches no declared arm token "
                     f"({sorted(ARM_ROOT_PREFIX)}) -- register it in ARM_ROOT_PREFIX deliberately")
#: Arm roots whose OPERATING POINT was superseded, excluded from the live join so the retry glob above
#: does not present two canonical readings of one key. The data stays on disk and stays readable by its own
#: recorded key; only the LIVE reading changes.
#:   grades_mt4karr2_ch_llama8b_it -- llama Arditi at alpha 0.1. The llama Arditi operating point was
#:   re-pinned to 0.05 on 2026-09-14 (0.1 cleared the 10-item over-refusal budget by 0.50 of an item and cost
#:   +30.60 pp on OR-Bench against +16.69 pp at 0.05). The three legs were re-run at 0.05 into
#:   grades_mt4karr6_ch_llama8b_it and graded 2026-09-15T02:4xZ; ARM_OPERATING_POINT below now declares 0.05,
#:   so leaving r2 in the join would file a 0.1 measurement under a 0.05 label.
SUPERSEDED_ARM_ROOTS = {"grades_mt4karr2_ch_llama8b_it"}

MT_ARMS_ARM_ROOTS = ([f"grades_{tok}_{clu}_{tag}"
                      for tok in ARM_ROOT_PREFIX
                      for tag in (ARM_ROOT_TAGS[tok] or MT_ARMS_ARM_TAGS)
                      for clu in (ARM_ROOT_CLUSTERS[tok] or list(CLUSTER_TOKEN))]
                     # ...PLUS every RETRY namespace that actually exists on disk. A retry gets its own root
                     # (ROOT_SUFFIX=r2, r3, ...) precisely so it can never write into the dead attempt's root, but
                     # the hand-written list above only ever named the FIRST attempt. On 2026-09-14 sixteen legs
                     # were resubmitted into r2/r3/r4/r5 after a liveness fix; not one of those roots was listed,
                     # so every grade would have landed somewhere nothing read -- the sixth "finished work that
                     # reaches no figure" of that day, and the first caught BEFORE the work finished.
                     # Enumerated, not enumerated-by-hand: a suffix nobody has used yet still works.
                     # ...one glob PER DECLARED TOKEN, so a new arm's restarts are enumerated by the same rule
                     # (the old single glob was the character class `mt4k[ai][rs]r`, which matches mt4kar / mt4kas
                     # and nothing a later arm could be named -- a new token's retry namespace was invisible)
                     # The CLUSTER restriction above applies to the DECLARED first-attempt list ONLY, never here.
                     # Narrowing the retry glob by it dropped four roots that exist on disk and are in today's join
                     # (grades_mt4kasr{2,3,4}_ch_nemotron3nano4b, grades_mt4kasr4_ch_qwen35_4b_it): AlphaSteer's
                     # "cluster A only" is true of where its legs were SUBMITTED, not of where every retry landed, and
                     # a registration edit must not silently remove a reading. Measured before it shipped.
                     + sorted({d.name for tok in ARM_ROOT_PREFIX for d in RS.glob(f"grades_{tok}r[0-9]*_*")
                               if d.is_dir() and d.name not in SUPERSEDED_ARM_ROOTS and any(
                                   d.name.endswith(f"_{clu}_{tag}")
                                   for tag in (ARM_ROOT_TAGS[tok] or MT_ARMS_ARM_TAGS)
                                   for clu in CLUSTER_TOKEN)}))
#: Per (view, arm): the operating point as a DECLARED (id, prose, detail) triple, NOT one prose string with the
#: id buried in parentheses: a consumer that matches on a sentence breaks silently the day someone improves the
#: wording -- the match stops firing, superseded rows stop being parked, and nothing raises. The id is what a
#: downstream declaration keys on. `detail` is whatever else belonged inside the parenthetical (the AlphaSteer
#: selection outcome), split out ONLY so arm_op_label() below can rebuild the human label BYTE-IDENTICALLY to the
#: single string this table used to hold -- this change adds a machine key, it does not re-word a published label.
#: Published whole into meta.arm_operating_points so every component's rows can be labelled from ONE declaration.
ARM_OPERATING_POINT = {
    ("gemma4e4b_it", "arditi"): ("ar_r0p05__arditi", "Arditi refusal-direction ablation, alpha 0.05", ""),
    ("qwen35_4b_it", "arditi"): ("ar_r0p05__arditi", "Arditi refusal-direction ablation, alpha 0.05", ""),
    ("nemotron3nano4b", "arditi"): ("ar_r0p05__arditi", "Arditi refusal-direction ablation, alpha 0.05", ""),
    # Re-pinned 2026-09-14 on the rule: llama's Arditi operating point is 0.05, not 0.1 (0.1 cleared the 10-item
    # over-refusal budget by 0.50 of an item and cost +30.60 pp on OR-Bench against +16.69 pp at 0.05). The three legs were
    # re-run at 0.05 into grades_mt4karr6_ch_llama8b_it and graded 2026-09-15T02:4xZ; the superseded 0.1 measurement is in
    # grades_mt4karr2_ch_llama8b_it, excluded from the live join by SUPERSEDED_ARM_ROOTS above.
    ("llama8b_it", "arditi"): ("ar_r0p05__arditi", "Arditi refusal-direction ablation, alpha 0.05", ""),
    ("gemma4e4b_it", "alphasteer"): ("ladder_canon/lam2", "AlphaSteer, ladder rung 2", "; selection_outcome ruled_2026-09-14_second -- re-ruled by the user from rung 1"),
    ("qwen35_4b_it", "alphasteer"): ("ladder_canon/lam1", "AlphaSteer, ladder rung 1", "; selection_outcome ruled_2026-09-14 -- re-ruled by the user from rung 3, whose chosen point sat inside the degeneration zone"),
    ("nemotron3nano4b", "alphasteer"): ("ladder_canon/lam2", "AlphaSteer, ladder rung 2", "; selection_outcome ruled_2026-09-14 -- re-ruled by the user from rung 1, the a4_fallback"),
    ("llama8b_it", "alphasteer"): ("ladder_canon/lam1", "AlphaSteer, ladder rung 1", "; selection_outcome chosen is provenance-from-message: no ladder record for llama"),
    # Phase E's OUTPUT, now declared (arditi-qwen-all-positions, ladder selection run 2026-09-16): the row was
    # deliberately withheld while the selection step was outstanding. The dose is c = 0.3 in the `raw_coefficient`
    # unit -- NOT a fraction of the residual norm, which is the sibling `arditi` arm's unit, so the two arms differ
    # in dose UNIT as well as in phase and the ids must not be read as rungs of one ladder. (alpha_equiv 0.10540806
    # is arithmetic on this model only and is never the dose.) selection_outcome is
    # user_override_over_budget_2026-09-16: the selection rule's own answer was ar_c0p2 (12.55 items / 5.04 pp) and
    # c = 0.3 was ruled over it -- kept out of `detail` so arm_op_label() reproduces the string the Phase B'
    # rehearsal verified byte-for-byte. served_operating_point_of derives ar_c0p3__arditi_allpos from the per-arm
    # config's defenses_dir (.../cells_view/ar_c0p3) plus the arm name, so this declaration VERIFIES each root's
    # legs rather than merely labelling them: a root served at any other c now REFUSES instead of being filed here.
    ("qwen35_4b_it", "arditi_allpos"): ("ar_c0p3__arditi_allpos",
                                        "Arditi refusal direction, paper configuration, c = 0.3", ""),
}
#: the sentinel an arm row carries when no operating point is declared for its (view, arm). MUST stay distinct
#: from null, which means "not an arm cell, the question does not arise".
ARM_OP_UNDECLARED = "UNDECLARED"


#: A per-arm config recorded in `grading_provenance.config` may name a path that no longer exists: the arm runs read their configs
#: from read-only worktrees, and those get removed. That leaves the operating-point check inert exactly
#: where it matters -- on 2026-09-17 all 15 arm roots read `unverifiable` against a live export that had read `verified`,
#: so nothing could confirm any arm's legs were served at the dose they are filed under. The recorded sha256 is enough to
#: fix this: the config is CONTENT-addressed, so recover the bytes from git by hashing each revision of the same repo-relative
#: path and taking the one whose sha256 equals the pin. A recovered copy is exactly as trustworthy as the on-disk file was --
#: more so, since the path could have been edited in place and the sha is checked here every time. Verified 2026-09-17:
#: 15/15 of the campaign's arm configs recover this way, including one whose worktree name carried no commit at all.
def _repo_history_is_searchable() -> bool:
    """False when this checkout cannot answer a history question — a SHALLOW clone.

    The Modal cpu-suite stages its gate tree with ``fetch --depth 1`` (scripts/modal/cpu_suite.py
    shallow_clone_tarball), so inside that container no commit but HEAD exists and
    :func:`_config_bytes_from_git` can never find anything. That is not "no revision matches the pin";
    it is "the question could not be asked", and the two must not both surface as a bare miss --
    otherwise a shallow checkout silently returns every arm root to `unverifiable` with a reason that
    blames the data."""
    repo = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo,
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return False                      # not a git checkout at all: a clean miss, not a shallow one
    return out != "true"


def _config_rel(recorded_path: str) -> Optional[str]:
    """The repo-relative ``configs/<name>.yaml`` tail of a recorded (possibly removed) absolute path, or None."""
    m = re.search(r"/(configs/[^/]+\.ya?ml)$", recorded_path or "")
    return m.group(1) if m else None


def _published_without_history(repo: Path) -> bool:
    """True when ``repo`` is a repository PUBLISHED without its development history, by the repository's own statement: not shallow
    (``git rev-parse --is-shallow-repository`` prints ``false``), the publish step's root marker ``.history-squashed`` present at the
    root with content ``squashed``, and the repository's ONE root commit — the publication root — carrying that marker, so commits
    made later on top of the publication (committed configs) are descendants and change nothing. False otherwise, a stray marker
    the root commit does not carry and a bare one-commit repository without the marker included: a commit count never decides."""
    marker = Path(repo) / ".history-squashed"
    try:
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != "squashed":
            return False
        shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo, capture_output=True, text=True, timeout=60)
        if shallow.returncode != 0 or shallow.stdout.strip() != "false":
            return False
        roots = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=60)
        shas = roots.stdout.split()
        if roots.returncode != 0 or len(shas) != 1:
            return False
        committed = subprocess.run(["git", "cat-file", "-p", f"{shas[0]}:.history-squashed"], cwd=repo, capture_output=True, text=True, timeout=60)
        return committed.returncode == 0 and committed.stdout.strip() == "squashed"
    except Exception:  # noqa: BLE001
        return False


def _history_has_no_earlier_revision(recorded_path: str) -> bool:
    """True when this clone's history CANNOT hold a superseded revision of the recorded config: the repository was published without
    its development history (:func:`_published_without_history` — the marker committed by its publication root, whatever was committed
    on top), or no commit in it touches that repo-relative path. Distinct from a SHALLOW clone (the history was cut:
    :func:`_repo_history_is_searchable`) and from a content miss (a searchable history that holds revisions of the path, none with
    the pinned bytes). False when the question cannot be asked (not a git checkout), so the caller falls through to the content-miss
    record as before."""
    rel = _config_rel(recorded_path)
    repo = Path(__file__).resolve().parents[3]
    try:
        if _published_without_history(repo):
            return True
        if rel is None:
            return False
        revs = subprocess.run(["git", "log", "--all", "--format=%H", "--", rel], cwd=repo,
                              capture_output=True, text=True, timeout=60).stdout.split()
    except Exception:  # noqa: BLE001
        return False
    return not revs


def _config_bytes_from_git(recorded_path: str, want_sha256: str) -> Optional[bytes]:
    """The config's bytes recovered from git history by CONTENT sha, or None. Never returns bytes whose sha mismatches."""
    if not recorded_path or not want_sha256:
        return None
    m = re.search(r"/(configs/[^/]+\.ya?ml)$", recorded_path)
    if not m:
        return None
    rel = m.group(1)
    repo = Path(__file__).resolve().parents[3]
    try:
        revs = subprocess.run(["git", "log", "--all", "--format=%H", "--", rel], cwd=repo,
                              capture_output=True, text=True, timeout=60).stdout.split()
    except Exception:  # noqa: BLE001  (a missing git or a non-repo checkout is a clean miss, not a crash)
        return None
    for rev in revs[:200]:                      # bounded: these files change rarely
        try:
            blob = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=repo,
                                  capture_output=True, timeout=60).stdout
        except Exception:  # noqa: BLE001
            continue
        if blob and hashlib.sha256(blob).hexdigest() == want_sha256:
            return blob
    return None


def served_operating_point_of(gp: dict, arm: str, view: str) -> dict:
    """The operating point a root's legs were ACTUALLY served at, read from the per-arm config the grader recorded
    (grading_provenance.config.path, sha-pinned there) -- real FIELDS, never the header comment: alphasteer carries
    `rung_i` + `object_sha256`; arditi names its bank view in `defenses_dir` (.../cells_view/ar_r0p05). Returns the
    derived record and REFUSES when it disagrees with ARM_OPERATING_POINT for (view, arm): that disagreement is the
    09-14 -> 09-15 defect, a measurement filed under a superseded label. A config that no longer exists on disk is
    recorded as unverifiable (worktrees get removed) rather than refused -- the refusal is for a CONTRADICTION."""
    cfg = (gp.get("config") or {})
    path = Path(cfg.get("path") or "")
    declared = ARM_OPERATING_POINT.get((view, arm))
    out = {"declared_id": declared[0] if declared else None, "config_path": str(path), "config_sha256": cfg.get("sha256")}
    import yaml  # local: the builder has no other YAML need
    if path.is_file():
        raw = path.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if cfg.get("sha256") and got != cfg.get("sha256"):
            # a config the release rewrote by declared rules only (site paths to placeholders, comment text) passes through
            # its certified scrub-map row; any other difference still refuses
            from sisteer import scrub_map as SM
            if SM.check(str(cfg.get("sha256")), got, path=path) != SM.VIA_MAP:
                raise SystemExit(f"REFUSING: {path} sha256 {got[:12]} != the sha the grader recorded {str(cfg.get('sha256'))[:12]} "
                                 f"-- the config on disk is not the one these legs ran with ({SM.mismatch_note(cfg.get('sha256'), got)})")
            out["config_sha_check"] = SM.VIA_MAP
        out["config_source"] = "on disk"
    else:
        raw = _config_bytes_from_git(str(path), str(cfg.get("sha256") or ""))
        if raw is None:
            # distinguish "asked history, nothing matched" from "could not ask" -- a shallow checkout
            # (the Modal gate stages --depth 1) makes recovery structurally impossible, and reporting
            # that as a content miss would blame the data for the checkout's shape
            if not _repo_history_is_searchable():
                out.update(status="unverifiable", searchable_history=False,
                           reason="the recorded per-arm config is not on disk and this checkout is SHALLOW, so git "
                                  "history cannot be searched -- recovery was not attempted, not attempted and failed")
            elif _history_has_no_earlier_revision(str(path)):
                # the third shape: a full (not shallow) clone whose history holds nothing before HEAD, or nothing of this path --
                # a published repository squashed to one commit. The search above DID run (HEAD's own revision of the path can carry
                # the recorded bytes, and then it recovers them), so the record says what the search found, and names the clone's
                # shape as the reason there was nothing earlier to find -- never the data.
                out.update(status="unverifiable", searchable_history=False,
                           reason=f"this clone's history holds no earlier revision of {_config_rel(str(path)) or path.name} (a published "
                                  "repository without its development history); the search over the revisions it does hold found none "
                                  "with the recorded sha256")
            else:
                out.update(status="unverifiable", searchable_history=True,
                           reason="the recorded per-arm config is not on disk and no revision of it in "
                                  "git matches the recorded sha256")
            return out
        out["config_source"] = "recovered from git by content sha256 (the recorded path is gone; the bytes are the pinned ones)"
    body = yaml.safe_load(raw.decode()) or {}
    if arm == "alphasteer":
        blk = next((v for v in body.values() if isinstance(v, dict) and "rung_i" in v), {})
        rung = blk.get("rung_i")
        served = f"ladder_canon/lam{rung}" if rung is not None else None
        out.update(served_id=served, rung_i=rung, lam=blk.get("lambda"), object_sha256=blk.get("object_sha256"),
                   source="config field rung_i (+ object_sha256)")
    else:
        dd = str(body.get("defenses_dir") or "")
        tail = dd.rsplit("cells_view/", 1)[-1].strip("/") if "cells_view/" in dd else ""
        served = f"{tail}__{arm}" if tail else None
        out.update(served_id=served, defenses_dir=dd, source="config field defenses_dir (bank view -> staged id)")
    if not served:
        out.update(status="unverifiable", reason="the config carries no rung_i / defenses_dir to derive from")
        return out
    if declared and served != declared[0]:
        raise SystemExit(f"REFUSING: {view}/{arm} legs were served at {served} (per {path.name}, sha {str(cfg.get('sha256'))[:8]}) but "
                         f"ARM_OPERATING_POINT declares {declared[0]} -- a measurement would be filed under a superseded label. "
                         f"Re-pin the table to the ruled point (or exclude the root via SUPERSEDED_ARM_ROOTS if the ruling moved on).")
    out.update(status="verified" if declared else "served_only")
    return out


def arm_op_label(op) -> str:
    """the human label for a declared operating point -- the exact string ARM_OPERATING_POINT used to hold, so the
    (id, prose, detail) split re-words nothing that was already published."""
    return f"{op[1]} ({op[0]}{op[2]})"
MT_ARMS_W24K_ROOTS = [f"grades_mt4k{arm}_w24k_{clu}_{tag}" for tag in MT_ARMS_W24K_TAGS for arm in ("sc", "nn") for clu in CLUSTER_TOKEN]
#: the llama constitution legs' RESTART namespace (2026-09-14 02:49Z): their first launch died AT STANDUP on every attack -- the goalset sbatch never
#: exported the load check's attack family and only the llama configs carry a per-family cap map -- and a refused leg still creates its
#: transcripts root, so the restart runs in a fresh namespace (ROOT_SUFFIX=r2) from the fixed tree 7b560918. Cluster B only: the cluster A twins were
#: cancelled by the race unit while the cluster B twin was briefly RUNNING, so this lane has no bu counterpart to land.
MT_ARMS_R2_ROOTS = ["grades_mt4kscr2_ch_llama8b_shortconst", "grades_mt4ksc_w24kr2_ch_llama8b_shortconst", "grades_mt4knn_w24kr2_ch_llama8b_shortconst"]
LANDING_NOTE = {**{r: "root not created yet: no leg of this lane has been graded (qwen legs 498351-498356; gemma legs after PR #107; R2-D288); lands on a later run"
                   for r in LANDING_ROOTS},
                **{r: ("root not created yet: no leg of this (lane, attack, cluster) has been graded (mt_arms4096: cluster B legs submitted 2026-09-14 00:31Z, "
                       "their cluster A twins queued behind the 16-slot cap; the first cluster to serve a leg wins and the other twin is cancelled by name, so this "
                       "root may land on a later run or never)") for r in MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS},
                **{r: ("root not created yet: the llama constitution legs' RESTART (cluster B 101133-101136, tree 7b560918, fresh namespace r2 after the "
                       "standup load-check refusal of 02:35Z); lands when its chain grades") for r in MT_ARMS_R2_ROOTS},
                **{r: ("root not created yet: no leg of this (arm, lane, cluster) has been graded (mt_arms4096 Arditi / AlphaSteer legs queued 2026-09-14 "
                       "01:38-02:01Z from the ported trees 68cb31a3 / 63f79fa5; Arditi races cluster B vs cluster A twins, AlphaSteer is cluster-A-only; "
                       "lands on a later run)") for r in MT_ARMS_ARM_ROOTS}}
SERVED_CLUSTER_DEFAULT = "cluster_a"
SERVED_CLUSTER_DEFAULT_NOTE = ("no served_cluster.json beside this root's transcripts and no cluster token in its name: it predates the ruling of 2026-09-13 16:45 "
                               "and was pulled by its lane's pull script from cluster A's login host -- every cap4096 / obj4096 / anchor4096 / qc4k root. "
                               "The default is recorded in meta.roots.<root>.served_cluster and on its meta.count_table rows; it is NOT added to the filed cell "
                               "records, which stay byte-identical to the export that already read them")
TWIN_RULE = ("two CANONICAL readings of one (view, cell, attack, serving window) from DIFFERENT clusters are twins, not a collision: the reading whose "
             "aggregate the grader wrote FIRST (two_si_<attack>_aggregate.json written_at; tie -> cluster_a) holds the view's cell slot "
             "(models.<view>.cells.<cell>.<attack>), the other is filed under models_twins.<view>.cells.<cell>.<attack> with the identical record shape; "
             "both carry served_cluster; meta.twin_status lists every such key with both roots. Two readings of one key from the SAME cluster remain a "
             "defect and refuse. The same rule places variant twins under models_variants.<tag>.models_twins")
#: the objects CONTROL wave (2026-09-14, mt_objects4096 e4o): the control-LoRA + LoRA-subtraction legs,
#: ENUMERATED rather than listed, so a new lane's root folds in with no edit here (the arm roots above do the same).
#: These are the FIRST canonical measurement of ctl_lora / const_lora multi-turn on their views, not a second pass:
#: verified 2026-09-14 in the 22:05Z export frame -- on gemma4e4b and qwen35_4b main those two cells carry multi-turn
#: rows at decode cap 512 ONLY (16 per attack, the greedy-512 instrument superseded 2026-09-11), and ZERO at 4,096; the
#: only 4,096 rows on those views belong to the difference cell. On gemma4e4b_it both cells read NOT scheduled. So
#: nothing here overwrites or duplicates a canonical row, and no (source, pass) declaration is needed -- the 512 rows are
#: excluded downstream by the paper config's multi-turn caps=(4096,). A pivot collapsed across decode caps misreads those
#: cells as already graded at 4,096.
#: GLOB IS grades_obj4096ctl* (2026-09-14, NOT grades_obj4096ctl_*): the three ActorAttack legs of this wave could not hold 518 goals
#: inside their 8 h walltime (measured 28-47 goals/h; the closest precedent, e4oa-att-gemma4e4b-actorattack, ran 08:35:14), and the
#: wave preflight refuses an existing root, so they re-run under a fresh namespace tag obj4096ctl2_<lane>. Their grades roots are
#: grades_obj4096ctl2_<lane>, which the old glob's literal underscore excluded. The attacks per root are DISJOINT (ctl: crescendo + pair,
#: graded; ctl2: actorattack only), so the same-cluster twin refusal in TWIN_RULE cannot trigger between a ctl root and its ctl2 sibling.
OBJ_CTL_ROOTS = sorted({d.name for d in RS.glob("grades_obj4096ctl*") if d.is_dir()})

#: THE relu_native WAVE (relu-steering-native-evals, 36/36 graded 2026-09-16; ingested as its own NAMESPACE): the three PSR
#: cells constv2-mix-full__psr_ce_all / constv2ctl2-full__psr_ce_all / constv2-mix-full__psr_ce_all_minus_ctl_distilled on the four MAIN
#: views, served on the NATIVE multi-LoRA engine (serving_engine lora_native, identity proxy_vllm+lora+interleaved, lens_reduction_mode
#: null), crescendo / pair / actorattack at the 4,096 cap over the 259-goal anchor pool, each model decoded at the resolved sampler of its
#: 4,096 `none` anchor. Root grammar: relu_native/grades_relu_native_<clu>_<TAG> with relu_native/transcripts_relu_native_
#: <clu>_<TAG> BESIDE it -- a wave DIRECTORY under run-state, so process_root derives the transcripts root from the grades root's OWN
#: parent (identical for every root that lives at the run-state root itself). A two-root merged store (a base root killed at standup +
#: its r2 / r3 restart) is ONE store here: shard*/ matches shard_r2_* too and the grader's completion stamp is the identity check.
#: Enumerated per cluster token; a TAG with no root REFUSES in main(): the wave is declared complete, so an absent root would be
#: finished work made invisible, not a leg still landing.
NS_RELU = "relu_native"
#: the arditi_allpos multi-turn wave (arditi-qwen-all-positions Phase G, 2026-09-17): the `mt4kap` arm roots (ARM_ROOT_PREFIX), served on
#: the NATIVE multi-LoRA engine like the relu_native wave but as SINGLE-ROOT stores at the run-state root, one root per cluster token. Their
#: own namespace rather than NS_RELU's because two of NS_RELU's serving terms describe merged two-root stores this wave never had -- see the
#: field-by-field note above NAMESPACE_SERVING. Routed by namespace_of_root(); no other root leaves NS.
NS_ALLPOS_NATIVE = "arditi_allpos_native"
RELU_NATIVE_DIR = "relu_native"
RELU_NATIVE_TAGS = ["gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"]
RELU_NATIVE_CELLS = ["constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"]
RELU_NATIVE_ROOTS = sorted(f"{RELU_NATIVE_DIR}/grades_relu_native_{clu}_{tag}" for tag in RELU_NATIVE_TAGS for clu in CLUSTER_TOKEN
                           if (RS / RELU_NATIVE_DIR / f"grades_relu_native_{clu}_{tag}").is_dir())
#: the unsteered `none` ANCHOR each relu_native view pairs against (the gemma / qwen 4,096 anchors live on the `it` views -- main
#: and it serve the same instruct checkpoint with no object, so the anchor is the model's; nemotron's anchor is its own view; llama's is
#: the llama8b_lane root's own `none` at 4,096). The anchor ROOT is resolved per attack: for the anchor4096 family from the enumerated anchor
#: roots of that view (a first-attempt root and its r2 restart are DISJOINT by attack), for llama from the llama8b_lane root -- exactly one
#: root may carry the attack's aggregate, or this refuses.
RELU_ANCHOR_VIEW = {"gemma4e4b": "gemma4e4b_it", "qwen35_4b": "qwen35_4b_it", "nemotron3nano4b": "nemotron3nano4b", "llama8b": "llama8b"}
RELU_ANCHOR_ROOT_FAMILY = {"gemma4e4b_it": "anchor4096", "qwen35_4b_it": "anchor4096", "nemotron3nano4b": "anchor4096", "llama8b": "llama8b_lane"}
LLAMA_LANE_ANCHOR_ROOT = "grades_llama8b_llama8b"
#: grades roots whose transcripts root does not follow the transcripts_<tag> grammar: the llama8b_lane pull script named ONE root
#: for its four views (llama8b_export/build_transfer_stats_full22_llama8b.py TRANSCRIPTS_ROOT)
TRANSCRIPTS_ROOT_OVERRIDE = {LLAMA_LANE_ANCHOR_ROOT: "transcripts_llama8b_lane"}
#: per NAMESPACE: the serving ENGINE its stores were drawn on and the lens reduction mode every record must carry (the lens wave's
#: `stacked`; the native engine has no lens, so null); how a shard whose sidecar was never finalised recovers its decode identity; and
#: whether the engine is stamped on the filed record (the lens-era records predate the field and stay byte-identical: record_engine False).
#: decode_fallback `declared_block` is the existing rule (the llama lineage's sidecar `decode` block); `timing_rows` reads the per-request
#: `decode` block on every target_timing.jsonl row of the shard (numbers only): a two-root merged store's base root was killed before its
#: sidecar was finalised (manifest_stage `started`, last_request_decode null) and the rows carry the sampler.
#: merged_store_windows: a two-root merged store's base root and its restart may have been stood up at DIFFERENT serving windows (qwen35_4b
#: ActorAttack: 92,160 on the base root's 9 shards, 61,440 on the r2 restart's 2). The window is an instrument term only where it BINDS (a
#: clamped or lost call); with zero clamps and zero prompt-over-window on every shard it never did, so such a store is ONE reading whose record
#: carries every window it was served at (serving_windows_by_shard) and the widest as serving_window_tokens -- and REFUSES if any call was
#: clamped under more than one window. The lens namespaces keep the window in the per-shard identity tuple (their stores are single-root).
#:
#: NS_ALLPOS_NATIVE (arditi-qwen-all-positions Phase G, 2026-09-17). The mt4kap arm roots are natively served, so under NS's lens
#: declaration read_cell refuses every one of their cell rows (aggregate lens_reduction_mode null, serving_engine lora_native) -- but
#: NS_RELU's entry is NOT the right one to reuse either. Each field was verified against the real store (grades_mt4kap_ch_qwen35_4b_it:
#: 48 shards, 16 per attack, every served_target.json read), and one of NS_RELU's terms is wrong for it:
#:   engine lora_native         TRUE -- serving_engine is lora_native on all 48 shards (serving_identity proxy_vllm+lora+interleaved).
#:   lens_reduction_mode None   TRUE -- the native path has no lens; proxy_identity.lens_reduction_mode is null on every shard.
#:   record_engine True         TRUE -- the native grader stamps serving_engine / require_engine on the aggregates, so the record carries it.
#:   decode_fallback timing_rows  the right VALUE for a DIFFERENT REASON than NS_RELU's. For relu_native it is a genuine fallback: a merged
#:                              store's base root was killed before its sidecar was finalised (manifest_stage started, last_request_decode
#:                              null) and the rows carry the sampler. Here all 48 sidecars are manifest_stage final with last_request_decode
#:                              populated, so the branch never falls back: it runs its CROSS-CHECK, refusing if last_request_decode disagrees
#:                              with the rows' own decode on any DECODE_KEYS. Strictly stronger than declared_block, and the data supports
#:                              it, which is why it is right (n_shards_decode_from_timing_rows must therefore never appear on one of these rows).
#:   merged_store_windows False  WHERE NS_RELU'S ENTRY IS WRONG FOR THIS STORE, and the whole reason it is not cloned. NS_RELU sets True
#:                              because its stores are two-root merged views whose base and restart were stood up at different windows.
#:                              mt4kap is a SINGLE-ROOT run on cluster B and the window is uniform within each cell (actorattack 49,152 on
#:                              16/16 shards, crescendo 49,152 on 16/16, pair 16,384 on 16/16). With True the identity key would record the
#:                              literal string "per shard (serving_windows_by_shard)" INSTEAD OF the real window -- asserting a merge that
#:                              did not happen, dropping the window from the identity tuple, and silently admitting a store whose shards
#:                              were served at two windows (zero clamps) as one reading. With False each cell yields exactly one identity
#:                              tuple carrying its window, and a window split refuses: the truthful record for a single-root store.
NAMESPACE_SERVING = {NS: {"engine": "lens", "lens_reduction_mode": WAVE_LENS_MODE, "decode_fallback": "declared_block", "record_engine": False, "merged_store_windows": False},
                     NS_VARIANT: {"engine": "lens", "lens_reduction_mode": WAVE_LENS_MODE, "decode_fallback": "declared_block", "record_engine": False, "merged_store_windows": False},
                     NS_RELU: {"engine": "lora_native", "lens_reduction_mode": None, "decode_fallback": "timing_rows", "record_engine": True, "merged_store_windows": True},
                     NS_ALLPOS_NATIVE: {"engine": "lora_native", "lens_reduction_mode": None, "decode_fallback": "timing_rows", "record_engine": True, "merged_store_windows": False}}
DECODE_KEYS = ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "note")

#: per ARM TOKEN (ARM_ROOT_PREFIX), the serving NAMESPACE its canonical roots are READ under (arditi-qwen-all-positions Phase G,
#: 2026-09-17). Absent = NS, the lens namespace every other canonical root is read under. Declared per token so a retry namespace
#: (grades_mt4kapr2_*) routes with its arm by the same rule that enumerates it, and so a second natively served arm is one entry here,
#: never a second special case at the call site.
ARM_ROOT_NAMESPACE = {"mt4kap": NS_ALLPOS_NATIVE}


def namespace_of_root(root_name: str) -> str:
    """The serving namespace a canonical grades root is READ under: its arm token's declared namespace where one is declared (longest
    declared token first, the same match as arm_of_root), else NS. Only mt4kap roots leave NS; every other root is read exactly as before."""
    for tok in sorted(ARM_ROOT_PREFIX, key=len, reverse=True):
        if tok in root_name:
            return ARM_ROOT_NAMESPACE.get(tok, NS)
    return NS


#: the mt4kap roots the join registers (first attempts + every retry namespace on disk), i.e. the roots read under NS_ALLPOS_NATIVE
ALLPOS_NATIVE_ROOTS = [r for r in MT_ARMS_ARM_ROOTS if namespace_of_root(r) == NS_ALLPOS_NATIVE]

#: the canonical roots: fourteen arm roots + the anchor roots. The VIEW is the target directory inside each
#: root, never this name.
CANONICAL_ROOTS = [
    # the cap-4096 wave: base views x {crescendo, pair} x the LoRA subtraction
    "grades_cap4096_gemma4e4b", "grades_cap4096_qwen35_4b", "grades_cap4096_nemotron3nano4b", "grades_cap4096_glm46vflash",
    # its ActorAttack leg
    "grades_cap4096aa_gemma4e4b", "grades_cap4096aa_qwen35_4b", "grades_cap4096aa_nemotron3nano4b",
    # the objects wave: constitution LoRA, control LoRA (nemotron / glm), prompted safety SI x {crescendo, pair}
    "grades_obj4096_gemma4e4b_it", "grades_obj4096_qwen35_4b_it", "grades_obj4096_nemotron3nano4b", "grades_obj4096_glm46vflash",
    # its ActorAttack leg, the RE-RUN in a fresh namespace after the first launch's partials; targets are the _it views
    "grades_objaa4096rerun_gemma4e4b", "grades_objaa4096rerun_qwen35_4b", "grades_objaa4096rerun_nemotron3nano4b",
    # the unsteered anchor (see ANCHOR_ROOTS above)
    *ANCHOR_ROOTS,
    # the qwen35_4b/it control + difference legs (see LANDING_ROOTS above; absent until their chains grade)
    *LANDING_ROOTS,
    # the constitution-SI arm at the arms' own windows, from either cluster (see MT_ARMS_ROOTS above; absent until their chains grade)
    *MT_ARMS_ROOTS,
    # the Arditi / AlphaSteer arms at their operating points (see MT_ARMS_ARM_ROOTS above; absent until their chains grade)
    *MT_ARMS_ARM_ROOTS,
    # the llama constitution legs' r2 restart namespace (see MT_ARMS_R2_ROOTS above)
    *MT_ARMS_R2_ROOTS,
    # the objects CONTROL wave, all lanes (see OBJ_CTL_ROOTS above; a root is absent until its chains grade)
    *OBJ_CTL_ROOTS,
]
#: the two one-term-moved variants: exported, marked non-canonical, paired with the canonical sibling at the same cap
VARIANT_ROOTS = {
    "grades_cap4096greedy_qwen35_4b": {
        "tag": "cap4096greedy", "term_moved": "decode",
        "reason": "greedy decode, not the target's shipped decode",
        "sibling_root": "grades_cap4096_qwen35_4b",
        "what_is_held_fixed": "same view (qwen35_4b), attack (crescendo), cell, goal set, 4,096 reply cap, 49,152 serving window, attacker pool, judge"},
    "grades_cap4096w16k_glm46vflash": {
        "tag": "cap4096w16k", "term_moved": "serving_window",
        "reason": "16k serving window",
        "sibling_root": "grades_cap4096_glm46vflash",
        "what_is_held_fixed": "same view (glm46vflash), attack (pair), cell, goal set, 4,096 reply cap, shipped decode, attacker pool, judge",
        # the sibling's PAIR leg at the wave's 8,704 window clamped 156 of 2,585 calls (129 distinct max_tokens granted, smallest 69) and lost 3
        # goals to PromptOverWindow; it failed the job's own completeness gate (482105, exit 2) and was NEVER graded, so grades_cap4096_glm46vflash
        # holds no pair aggregate. mt_cap4096/comparison.md reports glm46vflash/pair@4096 FROM this 16,384-window re-run. It is still classed
        # a variant; the absent sibling is recorded, not papered over.
        "sibling_note": ("the canonical sibling (grades_cap4096_glm46vflash, pair) has NO aggregate: its 8,704-window attempt clamped 6.03% of "
                         "calls and failed the completeness gate before grading; the lane's own comparison.md reports glm46vflash/pair@4096 "
                         "from this 16,384-window re-run")},
}
VARIANT_SIBLING_KEY_FIELDS = ("view", "attack", "cell", "decode_cap_tokens", "namespace")
#: the WINDOW VARIANT `w24k` (mt_arms4096; ruling 2026-09-14 00:40Z): PAIR at a 24,576 serving window on nemotron3nano4b and llama8b_shortconst --
#: the SI cell (arm sc) and its same-window `none` anchor (arm nn) -- because the constitution-SI system turn (~5.9k tokens) plus PAIR's measured
#: reservation maxima on those two targets (12,584 / 12,825) exceed the arms' 16,384 window. One tag over eight roots (2 tags x 2 arms x 2 clusters);
#: term moved = the serving window; the canonical sibling is the same view's 16,384 PAIR reading of the same cell WHERE ONE EXISTS -- only the
#: nemotron `none` has one (grades_anchor4096_nemotron3nano4b; the producer's seen map already holds (nemotron3nano4b, none, pair) at 16,384 from it,
#: which is exactly why these are variants and not canonical roots). Rows are keyed by window (instrument.serving_window_tokens; sibling_key carries
#: both windows) and never pool with the 16,384 rows. Every w24k root is a LANDING root (landing: True).
W24K_WINDOW, W24K_SIBLING_WINDOW = 24576, 16384
W24K_SIBLING = {("nemotron3nano4b", "none"): "grades_anchor4096_nemotron3nano4b"}
W24K_SIBLING_NOTE = {
    ("nemotron3nano4b", SI_CONST): ("ABSENT by ruling: prompted__si_constitution/pair ran ONLY at the 24,576 window on this view, so no 16,384 reading of "
                                    "the same cell exists to pair against; its same-window contrast is against the nn `none` PAIR anchor at 24,576 "
                                    "(grades_mt4knn_w24k_<clu>_nemotron3nano4b), which this builder does not compute (vs_bare is a follow-up)"),
    ("llama8b_shortconst", SI_CONST): ("ABSENT by ruling: prompted__si_constitution/pair ran ONLY at the 24,576 window on this view; its same-window contrast "
                                       "is against the nn `none` PAIR anchor at 24,576 (grades_mt4knn_w24k_<clu>_llama8b_shortconst), not computed here"),
    ("llama8b_shortconst", "none"): ("ABSENT: no 16,384 PAIR `none` reading exists on view llama8b_shortconst (the llama lane's shortconst view is BoN-only); "
                                     "the llama lane's `none` PAIR at 16,384 sits on views llama8b / llama8b_it (grades_llama8b_llama8b{,_it}, the llama "
                                     "builder's file) -- a different view, deliberately not paired across views here"),
}
W24K_SIBLING_KEY_FIELDS = VARIANT_SIBLING_KEY_FIELDS + ("serving_window_tokens",)
for _tag in MT_ARMS_W24K_TAGS:
    for _arm in ("sc", "nn"):
        for _clu, _cname in CLUSTER_TOKEN.items():
            _cell = MT_ARMS_CELL[_arm]
            _v = {"tag": "w24k", "term_moved": "serving_window", "term_moved_detail": f"serving window {W24K_SIBLING_WINDOW:,} -> {W24K_WINDOW:,} (PAIR only)",
                  "reason": f"{W24K_WINDOW:,}-token serving window for PAIR (the arms' PAIR window is {W24K_SIBLING_WINDOW:,})",
                  "sibling_root": W24K_SIBLING.get((_tag, _cell)),
                  "what_is_held_fixed": ("same view, attack (pair), cell, goal set, 4,096 reply cap, shipped decode, attacker pool, judge; tree = the config-only "
                                         "overlay 1d17331e on the arms' 0088d863 (nemotron) / the llama lineage 390b230a (llama8b_shortconst)"),
                  "window_tokens": W24K_WINDOW, "sibling_window_tokens": W24K_SIBLING_WINDOW, "landing": True, "cluster": _cname, "arm": _arm, "cell": _cell,
                  "why": ("ruling 2026-09-14 00:40Z: the constitution-SI system turn (~5.9k tokens) plus PAIR's measured reservation maxima on nemotron (12,584) "
                          "and llama (12,825) exceed 16,384; PAIR ran at 24,576 on these two lanes, each with a same-window `none` PAIR anchor (arm nn); rows "
                          "keyed by window, never pooled with the 16,384 rows")}
            if _v["sibling_root"] is None:
                _v["sibling_note"] = W24K_SIBLING_NOTE[(_tag, _cell)]
            VARIANT_ROOTS[f"grades_mt4k{_arm}_w24k_{_clu}_{_tag}"] = _v
del _tag, _arm, _clu, _cname, _cell, _v

#: the 512 side of every (view, cell) here: the grades root the export's join READ for that cell, and the join file to cross-check against.
#: Explicit per cell on purpose — the nemotron block mixes two roots and the it views two files, and an implicit rule has misattributed before.
MINUS_CTL, LORA, CTL_LORA, PROMPTED = ("constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__lora_r4_ce",
                                        "constv2ctl2-full__lora_r4_ce", "prompted__safety_refusal")
J_ALL, J_GLM, J_NEMO, J_PROMPTED = ("transfer_stats_full22_all.json", "transfer_stats_full22_glm.json",
                                    "transfer_stats_nemotron_complete.json", "transfer_stats_full22_prompted.json")
CAP512_SOURCE = {
    ("gemma4e4b", MINUS_CTL): ("grades_rest/gemma4e4b", J_ALL),
    ("qwen35_4b", MINUS_CTL): ("grades_rest/qwen35_4b", J_ALL),
    # the control LoRA and the constitution LoRA on the two MAIN views (2026-09-14): first measured at 4,096 by the objects
    # CONTROL wave, so they reach this table only now. Their 512 side is the same root and join file the difference cell
    # above already uses -- verified, not assumed: grades_rest/<view> holds the 512 aggregates for all three attacks, and
    # transfer_stats_full22_all.json carries both cells on both views across crescendo / pair / actorattack.
    ("gemma4e4b", LORA): ("grades_rest/gemma4e4b", J_ALL),
    ("gemma4e4b", CTL_LORA): ("grades_rest/gemma4e4b", J_ALL),
    ("qwen35_4b", LORA): ("grades_rest/qwen35_4b", J_ALL),
    ("qwen35_4b", CTL_LORA): ("grades_rest/qwen35_4b", J_ALL),
    ("nemotron3nano4b", MINUS_CTL): ("grades_nemotron_complete/nemotron3nano4b", J_NEMO),
    ("nemotron3nano4b", LORA): ("grades_rest/nemotron3nano4b", J_NEMO),        # the wave cell, read unchanged by the nemotron_complete join
    ("nemotron3nano4b", CTL_LORA): ("grades_rest/nemotron3nano4b", J_NEMO),
    ("nemotron3nano4b", PROMPTED): ("grades_nemotron_complete/nemotron3nano4b", J_NEMO),
    ("glm46vflash", MINUS_CTL): ("grades_glm/glm46vflash", J_GLM),
    ("glm46vflash", LORA): ("grades_glm/glm46vflash", J_GLM),
    ("glm46vflash", CTL_LORA): ("grades_glm/glm46vflash", J_GLM),
    ("glm46vflash", PROMPTED): ("grades_glm/glm46vflash", J_GLM),
    ("gemma4e4b_it", LORA): ("grades_rest/gemma4e4b_it", J_ALL),
    ("gemma4e4b_it", PROMPTED): ("grades_prompted/gemma4e4b_it", J_PROMPTED),
    ("qwen35_4b_it", LORA): ("grades_rest/qwen35_4b_it", J_ALL),
    ("qwen35_4b_it", PROMPTED): ("grades_prompted/qwen35_4b_it", J_PROMPTED),
    # the unsteered anchor's 512 side: the wave's own bare cell in grades_rest — the none row every join file carries with identical k/n
    ("gemma4e4b_it", "none"): ("grades_rest/gemma4e4b_it", J_ALL),
    ("qwen35_4b_it", "none"): ("grades_rest/qwen35_4b_it", J_ALL),
    ("nemotron3nano4b", "none"): ("grades_rest/nemotron3nano4b", J_NEMO),
}
#: the llama views carry the llama builder's spellings and kinds (llama8b_export/build_transfer_stats_full22_llama8b.py ROOTS): shortconst / fullconst
#: are PROMPTED-ONLY views of key llama8b (the unsteered model beside one prompted constitution cell; no fitted object), kind "prompted".
VIEW_KIND = {"gemma4e4b": "main", "qwen35_4b": "main", "nemotron3nano4b": "main", "glm46vflash": "main", "gemma4e4b_it": "it", "qwen35_4b_it": "it",
             "llama8b": "main", "llama8b_it": "it", "llama8b_shortconst": "prompted", "llama8b_fullconst": "prompted"}
VIEW_SUFFIXES = ("_it", "_shortconst", "_fullconst")       # view = <key><suffix>; the lanes' transcripts roots may be named by the KEY alone
LLAMA_VIEWS = [v for v in VIEW_KIND if v.startswith("llama8b")]
VIEW_LABEL = {"gemma4e4b": "google/gemma-4-E4B-it — main view, multi-turn at a 4,096-token reply cap",
              "qwen35_4b": "Qwen/Qwen3.5-4B (instruct) — main view, multi-turn at a 4,096-token reply cap",
              "nemotron3nano4b": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 — main view, multi-turn at a 4,096-token reply cap",
              "glm46vflash": "zai-org/GLM-4.6V-Flash — main view, multi-turn at a 4,096-token reply cap (rows are LOWER BOUNDS: no 8k rung, R2-D256a)",
              "gemma4e4b_it": "google/gemma-4-E4B-it — it view, multi-turn at a 4,096-token reply cap",
              "qwen35_4b_it": "Qwen/Qwen3.5-4B (instruct) — it view, multi-turn at a 4,096-token reply cap",
              "llama8b": "NousResearch/Meta-Llama-3.1-8B-Instruct — main view (base→instruct transfer lane), multi-turn at a 4,096-token reply cap",
              "llama8b_it": "NousResearch/Meta-Llama-3.1-8B-Instruct — it view (instruct-direct lane), multi-turn at a 4,096-token reply cap",
              "llama8b_shortconst": "NousResearch/Meta-Llama-3.1-8B-Instruct — shortconst view (prompted-only lane: the unsteered model beside the constitution SI as the system prompt, short form), multi-turn at a 4,096-token reply cap",
              "llama8b_fullconst": "NousResearch/Meta-Llama-3.1-8B-Instruct — fullconst view (prompted-only lane: the FULL constitution as the system prompt), multi-turn at a 4,096-token reply cap"}
PROMPTED_LABEL = "PROMPTED safety SI (safety_refusal as the system prompt; no object)"
PROMPTED_CONST_LABEL = "PROMPTED constitution SI (the constitution SI as the system prompt, short form; no object; mt_arms4096 arm sc)"


def base_target(view: str) -> str:
    """The served key behind a view: gemma4e4b_it -> gemma4e4b, llama8b_shortconst -> llama8b (a transfer view, its instruct view and a prompted-only view
    serve the SAME checkpoint; the objects differ in where they were fit, not in what serves them)."""
    for sfx in VIEW_SUFFIXES:
        if view.endswith(sfx):
            return view[:-len(sfx)]
    return view


def sha256_file(p) -> str | None:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def resolve_transcripts_root(rs: Path, root_name: str, view: str) -> Path:
    """The transcripts root beside a grades root. The lanes' pull scripts name it by the lane KEY (transcripts_obj4096_gemma4e4b holds the
    gemma4e4b_it run) while the grades root carries the TARGET (grades_obj4096_gemma4e4b_it), so both spellings are tried -- in the grades
    root's OWN directory (the run-state root, or a wave directory such as relu_native/), a declared override winning (TRANSCRIPTS_ROOT_OVERRIDE).
    The match is then PROVED, not assumed: the root's single run directory must be sdfbase_full22_<view> (verify_cap() ties every cell's
    timing / response counts to the grader's own completion stamp afterwards)."""
    root_dir = Path(root_name)
    parent = (rs / root_dir).parent
    if root_dir.as_posix() in TRANSCRIPTS_ROOT_OVERRIDE:
        # an overridden root may hold SEVERAL runs (the llama8b_lane root holds one per view): the run is addressed by view, not by only_dir
        cands = [c for c in [parent / TRANSCRIPTS_ROOT_OVERRIDE[root_dir.as_posix()]] if (c / f"sdfbase_full22_{view}").is_dir()]
    else:
        tag = root_dir.name[len("grades_"):]
        cands = [parent / f"transcripts_{tag}"] + [parent / f"transcripts_{tag[:-len(sfx)]}" for sfx in VIEW_SUFFIXES if tag.endswith(sfx)]
        cands = [c for c in cands if c.is_dir() and only_dir(c).name == f"sdfbase_full22_{view}"]
    if len(cands) != 1:
        raise SystemExit(f"{rs / root_dir}: transcripts root for view {view!r} not resolved (candidates holding run sdfbase_full22_{view}: {cands})")
    return cands[0]


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def only_dir(p: Path) -> Path:
    """The single subdirectory of p (roots carry lock files and pull stamps beside it) — refuse on 0 or 2+."""
    ds = sorted(x for x in p.iterdir() if x.is_dir() and not x.name.startswith("."))
    if len(ds) != 1:
        raise SystemExit(f"{p}: expected exactly one directory inside, found {[d.name for d in ds]}")
    return ds[0]


def label_group(view: str, cell: str) -> tuple[str, str]:
    if cell == PROMPTED:
        return PROMPTED_LABEL, "prompted_baseline"
    if cell == SI_CONST:
        return PROMPTED_CONST_LABEL, "prompted_baseline"
    if cell in ARM_CELL.values():
        _op = ARM_OPERATING_POINT.get((view, cell))
        return (arm_op_label(_op) if _op else f"{cell} (operating point unrecorded for {view})"), "steering_arm"
    if VIEW_KIND[view] == "it":
        return B.IT_LABEL.get(cell, f"{cell} (extra)"), B.IT_GROUP.get(cell, "extra")
    if VIEW_KIND[view] == "prompted":
        # the prompted-only lanes' bare cell is the same served checkpoint as the main view's, in its own serving job (the llama builder's wording)
        return (("none (bare instruct, same wave, prompted-only lane's serving job)", "baseline_full22") if cell == "none" else (f"{cell} (extra)", "extra"))
    return B.CELL_LABEL.get(cell, f"{cell} (extra)"), B.CELL_GROUP.get(cell, "extra")


def served_cluster_of(root_name: str, troot: Path) -> tuple[str, dict]:
    """(cluster, provenance) for a root: transcripts_<root>/served_cluster.json (written by mt_arms4096/pull_mtarms.sh from the standup's shared
    root) when present -- cross-checked against the root name's cluster token; else the token alone (_bu_ / _ch_); else the pre-existing roots'
    default (SERVED_CLUSTER_DEFAULT_NOTE). `on_rows` says whether the cluster was DERIVED from a record (then every cell record carries it) or
    defaulted (then it rides meta only, so the filed records stay byte-identical)."""
    tok = next((t for t in CLUSTER_TOKEN if f"_{t}_" in root_name), None)
    p = troot / "served_cluster.json"
    sc = load_json(p)
    if sc:
        cl = sc.get("served_cluster")
        if cl not in CLUSTER_TOKEN.values():
            raise SystemExit(f"{p}: served_cluster {cl!r} is not one of {sorted(CLUSTER_TOKEN.values())}")
        if tok and CLUSTER_TOKEN[tok] != cl:
            raise SystemExit(f"{p}: served_cluster {cl!r} disagrees with the root name's cluster token _{tok}_ ({CLUSTER_TOKEN[tok]}) -- refusing")
        return cl, {"source": "served_cluster.json", "path": str(p), "sha256": sha256_file(p), "on_rows": True,
                    **{k: sc.get(k) for k in ("shared_root", "ssh_host", "lane", "root_tag", "attack", "serving_window_tokens", "arms_window_tokens",
                                             "window_departs_from_arms", "pulled_at")}}
    if tok:
        return CLUSTER_TOKEN[tok], {"source": "root name cluster token", "token": f"_{tok}_", "on_rows": True,
                                    "note": "no served_cluster.json beside the transcripts (pulled by something other than pull_mtarms.sh); the name token stands"}
    return SERVED_CLUSTER_DEFAULT, {"source": "default", "on_rows": False, "note": SERVED_CLUSTER_DEFAULT_NOTE}


# --------------------------------------------------------------------------- the producers' own records (numbers only)

def count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for line in f if line.strip())


def verify_cap(cell_dir: Path, completion: dict) -> dict:
    """The cap the replies were ACTUALLY generated under, from the shards' target_timing.jsonl (numbers only): every call's requested
    ``max_tokens`` (== CAP except window-clamped calls), the max ``completion_tokens``, the finish_reason == length count — and the identity of
    this store with the graded one (timing n / finish_length / responses.jsonl rows must equal the grader's completion stamp). Refuses on any
    disagreement, on a requested cap other than CAP, or on a completion longer than CAP."""
    n = n_len = n_at_cap = mx = 0
    n_resp = 0
    max_tokens_seen: dict = {}
    n_clamped = n_clamped_len = 0          # calls granted < CAP by the serving-window clamp; of those, the ones that then hit the reduced budget
    shards = sorted(sd for sd in cell_dir.glob("shard*") if sd.is_dir())
    for sd in shards:
        tp = sd / "target_timing.jsonl"
        if tp.exists():
            with open(tp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    t = json.loads(line)
                    ct = int(t.get("completion_tokens") or 0)
                    mt = t.get("max_tokens")
                    n += 1; mx = max(mx, ct)
                    n_len += (t.get("finish_reason") == "length")
                    n_at_cap += (ct >= CAP)
                    max_tokens_seen[mt] = max_tokens_seen.get(mt, 0) + 1
                    if mt is not None and int(mt) < CAP:
                        n_clamped += 1
                        n_clamped_len += (t.get("finish_reason") == "length")
        rp = sd / "responses.jsonl"
        if rp.exists():
            n_resp += count_lines(rp)
    tt = completion.get("target_timing") or {}
    if n != tt.get("n") or n_len != tt.get("finish_length"):
        raise SystemExit(f"{cell_dir}: timing rows {n} / finish_length {n_len} != grader stamp {tt} — not the graded store, refusing")
    if n_resp != completion.get("n_responses"):
        raise SystemExit(f"{cell_dir}: responses.jsonl rows {n_resp} != grader stamp n_responses {completion.get('n_responses')} — refusing")
    known = [int(k) for k in max_tokens_seen if k is not None]
    if not known or max(known) != CAP:
        raise SystemExit(f"{cell_dir}: requested max_tokens seen {sorted(max_tokens_seen)} — the cap requested was not {CAP}; refusing")
    if mx > CAP:
        raise SystemExit(f"{cell_dir}: a completion of {mx} tokens exceeds the {CAP} cap; refusing")
    if mx <= CAP_FILED:
        raise SystemExit(f"{cell_dir}: max completion_tokens {mx} <= {CAP_FILED} — indistinguishable from a {CAP_FILED}-cap store; refusing")
    return {"source": "shard*/target_timing.jsonl (max_tokens, completion_tokens, finish_reason) + responses.jsonl line count; no text read",
            "n_timing_rows": n, "requested_max_tokens": {str(k): v for k, v in sorted(max_tokens_seen.items(), key=lambda kv: (kv[0] is None, kv[0]))},
            "max_completion_tokens": mx, "n_finish_length": n_len, "n_completion_tokens_at_cap": n_at_cap,
            "n_calls_window_clamped": n_clamped, "n_clamped_calls_censored": n_clamped_len,
            "n_responses_rows": n_resp, "n_shards": len(shards),
            "verdict": (f"every call requested max_tokens {CAP}" + (f" except {n_clamped} window-clamped calls (min {min(known)}), of which {n_clamped_len} hit the reduced budget" if n_clamped else "")
                        + f"; max completion {mx}; generated under a {CAP}-token reply cap")}


def timing_rows_decode(shard_dir: Path) -> dict:
    """The decode identity of a shard from the per-request `decode` block on its target_timing.jsonl rows (numbers only, never a prompt or a
    reply): exactly ONE distinct tuple over DECODE_KEYS across the shard's rows, or refuse. The recovery for a shard whose sidecar was never
    finalised (NAMESPACE_SERVING decode_fallback `timing_rows`)."""
    seen: dict = {}
    tp = shard_dir / "target_timing.jsonl"
    if tp.exists():
        with open(tp) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line).get("decode") or {}
                    seen.setdefault(json.dumps({k: d.get(k) for k in DECODE_KEYS}, sort_keys=True), d)
    if len(seen) != 1:
        raise SystemExit(f"{shard_dir}: target_timing.jsonl carries {len(seen)} distinct decode tuples (expected exactly one): {list(seen)[:2]}; refusing")
    return next(iter(seen.values()))


def instrument(cell_dir: Path, serving: dict = NAMESPACE_SERVING[NS]) -> dict:
    """Serving window and the decode actually sent, from every shard's served_target.json — the IDENTITY fields must be one distinct tuple
    across shards or this refuses; the window-clamp COUNTERS are per shard and are summed. Refuses on a lost goal (prompt over window: the
    store's own completeness gate refuses those too) or a window mismatch; clamps that merely shaved a call's budget are recorded, not
    refused (the lane's own reading of qwen35_4b/pair: 9 of 3,875 calls granted 3,876-4,088 tokens, none of which then hit that budget)."""
    tuples: dict = {}
    counters = {"n_window_clamped": 0, "n_prompt_over_window": 0, "n_window_mismatch": 0, "n_tokenize_resolved": 0, "n_floor_only_unresolvable": 0}
    min_clamped = None
    n_declared_fallback = 0
    n_timing_fallback = 0
    windows_by_shard: dict = {}
    identities: set = set()
    bites = {"n_shards": 0, "n_shards_applicable": 0, "n_shards_verified": 0, "min_n_differing": None, "n_probes": set()}
    for st in sorted(cell_dir.glob("shard*/served_target.json")):
        s = json.loads(st.read_text())
        wc = s.get("window_clamp") or {}
        ld = s.get("last_request_decode") or {}
        if serving["engine"] != "lens":
            # a NATIVE namespace: every shard must state the declared engine (the native writers stamp every route; the lens wave predates the
            # field and its identity is the lens_reduction_mode stamp below, so this check is declared per namespace rather than assumed)
            if s.get("serving_engine") != serving["engine"]:
                raise SystemExit(f"{st}: serving_engine {s.get('serving_engine')!r} != the namespace's declared {serving['engine']!r}; refusing")
            identities.add(s.get("serving_identity"))
            ev = (s.get("evidence") or {}).get("steering_bites") or {}
            bites["n_shards"] += 1
            bites["n_shards_applicable"] += bool(ev.get("applicable"))
            bites["n_shards_verified"] += bool(ev.get("verified"))
            if ev.get("n_differing") is not None:
                bites["min_n_differing"] = ev["n_differing"] if bites["min_n_differing"] is None else min(bites["min_n_differing"], ev["n_differing"])
            if ev.get("n_probes") is not None:
                bites["n_probes"].add(ev["n_probes"])
        if serving["decode_fallback"] == "timing_rows":
            tr = timing_rows_decode(st.parent)
            if ld:
                # a finalised sidecar: its last request must agree with the rows' own decode on every compared key (the same identity, two records)
                if any(tr.get(k) != ld.get(k) for k in DECODE_KEYS):
                    raise SystemExit(f"{st}: last_request_decode {ld} disagrees with the target_timing.jsonl rows' decode {tr} on a compared key; refusing")
            else:
                ld = tr; n_timing_fallback += 1
        if serving["merged_store_windows"]:
            windows_by_shard[s.get("max_model_len")] = windows_by_shard.get(s.get("max_model_len"), 0) + 1
        if not ld and s.get("decode"):
            # the llama lineage (390b230a) records the DECLARED decode block on every shard and last_request_decode only where the shard's last call
            # recorded one (null on 370 shards of the llama8b_lane stores); the arms' 0088d863 lineage records last_request_decode only. Where the
            # last-request record is absent the declared block carries the same tuple (do_sample / T / top_p; no top_k, no penalty, no note), so it
            # stands in -- counted per shard below, and never preferred over a recorded last request. No shard of a pre-existing root takes this path.
            ld = s["decode"]; n_declared_fallback += 1
        key = json.dumps({"serving_window_tokens": (s.get("max_model_len") if not serving["merged_store_windows"] else "per shard (serving_windows_by_shard)"),
                          "decode": {k: ld.get(k) for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "note")},
                          "target_decode_note": s.get("target_decode_note"),
                          "lens_reduction_mode": (s.get("proxy_identity") or {}).get("lens_reduction_mode")}, sort_keys=True)
        tuples[key] = tuples.get(key, 0) + 1
        counters["n_window_clamped"] += int(wc.get("n_clamped") or 0)
        counters["n_prompt_over_window"] += int(wc.get("n_prompt_over_window") or 0)
        counters["n_window_mismatch"] += int(wc.get("n_window_mismatch") or 0)
        counters["n_tokenize_resolved"] += int(wc.get("n_tokenize_resolved") or 0)
        counters["n_floor_only_unresolvable"] += int(wc.get("n_floor_only_unresolvable") or 0)
        if wc.get("n_clamped") and wc.get("min_clamped_completion") is not None:
            min_clamped = wc["min_clamped_completion"] if min_clamped is None else min(min_clamped, wc["min_clamped_completion"])
    if len(tuples) != 1:
        raise SystemExit(f"{cell_dir}: served_target.json serving identity not uniform across shards — {len(tuples)} distinct tuples: {list(tuples)[:2]}")
    inst = json.loads(next(iter(tuples)))
    inst["n_shards"] = next(iter(tuples.values()))
    inst.update(counters)
    if serving["merged_store_windows"]:
        if len(windows_by_shard) > 1 and (counters["n_window_clamped"] or counters["n_prompt_over_window"] or counters["n_window_mismatch"]):
            raise SystemExit(f"{cell_dir}: served at {len(windows_by_shard)} serving windows {windows_by_shard} AND a call was clamped / lost -- the window bound on a "
                             f"call, so the store is two instruments, not one; refusing")
        inst["serving_window_tokens"] = max(windows_by_shard)
        inst["serving_windows_by_shard"] = {str(w): n for w, n in sorted(windows_by_shard.items())}
        inst["serving_window_note"] = (None if len(windows_by_shard) == 1 else
                                       f"two-root merged store served at {len(windows_by_shard)} windows ({inst['serving_windows_by_shard']} shards each); no call was clamped or "
                                       f"lost under either, so the window never bound; serving_window_tokens is the widest")
    inst["min_clamped_completion_floor"] = min_clamped
    if n_declared_fallback:
        inst["n_shards_decode_from_declared_block"] = n_declared_fallback     # only where it happened (no pre-existing record changes)
    if inst["n_prompt_over_window"] or inst["n_window_mismatch"]:
        raise SystemExit(f"{cell_dir}: prompt-over-window {inst['n_prompt_over_window']} / window mismatch {inst['n_window_mismatch']} — goals lost or "
                         f"identity broken; the store's own gate refuses this and so does this builder")
    if n_timing_fallback:
        inst["n_shards_decode_from_timing_rows"] = n_timing_fallback   # the merged stores' unfinalised base-root shards (only where it happened)
    if serving["record_engine"]:
        if len(identities) != 1:
            raise SystemExit(f"{cell_dir}: serving_identity not uniform across shards: {sorted(map(str, identities))}; refusing")
        inst["serving_engine"] = serving["engine"]
        inst["serving_identity"] = next(iter(identities))
        inst["steering_bites"] = {**{k: v for k, v in bites.items() if k != "n_probes"}, "n_probes": sorted(bites["n_probes"]),
                                  "source": "served_target.json evidence.steering_bites per shard (the native standup's own steer-is-live probe)"}
    if inst["lens_reduction_mode"] != serving["lens_reduction_mode"]:
        raise SystemExit(f"{cell_dir}: served lens_reduction_mode {inst['lens_reduction_mode']} != {serving['lens_reduction_mode']}")
    d = inst["decode"]
    inst["decode_mode"] = "greedy" if d.get("do_sample") is False or d.get("temperature") == 0.0 else "shipped_sampling"
    inst["decode_cap_tokens"] = CAP
    inst["window_clamp_caveat"] = (None if not inst["n_window_clamped"] else
                                   f"{inst['n_window_clamped']} call(s) were granted less than {CAP} tokens because the conversation's prompt grew past "
                                   f"(window {inst['serving_window_tokens']} - {CAP}); a DIFFERENT censoring from at-cap, on the longest conversations "
                                   f"(mt_cap4096/comparison.md, R2-D253w); decode_cap_verification says how many of them then hit the reduced budget")
    return inst


# --------------------------------------------------------------------------- one cell x attack

def read_cell(agg: dict, cell: str, attack: str, ns: str, cell_dir: Path) -> tuple[dict, dict]:
    """The reader's record for (cell, attack) + this lane's stamps; the per-goal outcomes beside it."""
    serving = NAMESPACE_SERVING[ns]
    rec = R2._safe(T1.gated_cell, agg, cell)
    if rec.get("status"):
        raise SystemExit(f"{cell_dir}: reader returned {rec} — the grading is supposed to be complete; refusing")
    cellrow = agg["cells"][cell]
    if cellrow.get("lens_reduction_mode") != serving["lens_reduction_mode"]:
        raise SystemExit(f"{cell_dir}: aggregate row lens_reduction_mode != {serving['lens_reduction_mode']}")
    if serving["record_engine"] and (cellrow.get("serving_engine"), agg.get("serving_engine"), agg.get("require_engine")) != (serving["engine"],) * 3:
        raise SystemExit(f"{cell_dir}: aggregate serving_engine / require_engine {(cellrow.get('serving_engine'), agg.get('serving_engine'), agg.get('require_engine'))} "
                         f"!= the namespace's declared {serving['engine']!r}; refusing")
    completion = (cellrow.get("completion") or {}).get("gated") or {}
    if completion.get("n_error_outcomes"):
        # every root here graded 259/259 with 0 errored outcomes; an errored count would change what k/n mean and must be seen, not absorbed
        raise SystemExit(f"{cell_dir}: {completion.get('n_error_outcomes')} errored goal outcomes — not expected in these roots; refusing")
    m = cellrow["attack_metrics"]
    rec["namespace"] = ns
    rec["lens_reduction_mode"] = serving["lens_reduction_mode"]
    rec["tier"] = "multi_turn"
    # THE OVERRIDE (see the module docstring): the grader stamped the config's 512; the run's cap is proved from the producers' records below
    rec["decode_cap_tokens"] = CAP
    rec["decode_cap_tokens_grader_stamp"] = cellrow.get("decode_cap_tokens")
    rec["decode_cap_stamp_note"] = ("the grader copies attacks.target_max_new_tokens from the worktree config it is handed (512); the 4,096 legs "
                                    "were driven with the cap on the command line, so the stamp is the config's, not the run's — overridden here "
                                    "and proved per cell in decode_cap_verification")
    rec["decode_cap_verification"] = verify_cap(cell_dir, completion)
    rec["instrument"] = instrument(cell_dir, serving)
    if serving["record_engine"]:
        rec["serving_engine"] = serving["engine"]
        rec["serving_identity"] = rec["instrument"]["serving_identity"]
    if rec["decode_cap_verification"]["n_calls_window_clamped"] != rec["instrument"]["n_window_clamped"]:
        raise SystemExit(f"{cell_dir}: clamped calls per target_timing ({rec['decode_cap_verification']['n_calls_window_clamped']}) != served_target "
                         f"window_clamp.n_clamped summed over shards ({rec['instrument']['n_window_clamped']}) — refusing")
    rec["completion"] = completion
    rec["mean_n_target_forwards"] = m.get("mean_n_target_forwards")
    rec["n_capped_goals"] = m.get("n_capped")
    rec["n_expected"], rec["n_graded"], rec["n_errored"] = m.get("n_expected"), m.get("n_graded"), m.get("n_errored")
    rec["goalset"] = m.get("goalset")
    rec["by_category_group"] = {g: {"n_goals": v["n_goals"], "n_jailbroken": v["n_jailbroken"], "asr": v["asr"]}
                                for g, v in (m.get("by_category_group") or {}).items()}
    # at-cap from the grader's stamp of the producer's target_timing: finish_reason == length over every target reply — the SAME definition
    # the degeneracy join uses for the 512 rows (degeneracy_full22.json cells.*.at_cap_definition = "producer finish_reason == length"),
    # so the two caps' at-cap shares are the same quantity. No degeneracy join was built for these roots, so loop / degenerate stay PENDING.
    tt = completion.get("target_timing") or {}
    ac = T1.prop(int(tt.get("finish_length", 0)), int(tt.get("n", 0)))
    ac["definition"] = "producer finish_reason == length (target_timing.jsonl, via the grader's completion.gated.target_timing stamp)"
    ac["decode_cap_tokens"] = CAP
    rec["at_cap_finish_length"] = ac
    rec["degeneracy"] = B.degeneracy_block(None)
    rec["degeneracy"]["reason"] = "no degeneracy join (degeneracy_join.py) has been run over the cap-4,096 roots; at-cap rides at_cap_finish_length"
    rec["asr_annotated"] = B.annotate(rec, None) + f" | at-cap {B.pct(ac['p'])}% of {ac['n']} replies (finish_reason == length @ {CAP})"
    return rec, R2.attack_outcomes(agg, cell)


#: (view, cell) pairs that have NO filed 512 reading BY CONSTRUCTION: the batch-64 instruct-direct control LoRA and LoRA subtraction were
#: staged 2026-09-13 and first measured at 4,096; no greedy 512 pass ever ran them. Their vs_cap512 pairing is UNDEFINED here
#: (recorded as status NO_512_READING), not zero -- and this table is the deliberate declaration cap512_side() demands.
NO_CAP512 = {("qwen35_4b_it", CTL_LORA): "batch-64 control LoRA (itdoc64__constv2ctl2-full__lora_r4_ce): staged 2026-09-13, first measured at 4,096; no 512 pass",
             ("qwen35_4b_it", MINUS_CTL): "batch-64 LoRA subtraction (derived_itdoc64__…_minus_ctl): staged 2026-09-13, first measured at 4,096; no 512 pass",
             ("gemma4e4b_it", CTL_LORA): "batch-64 control LoRA (itdoc64__constv2ctl2-full__lora_r4_ce): staged 2026-09-13, first measured at 4,096; no 512 pass",
             ("gemma4e4b_it", MINUS_CTL): "batch-64 LoRA subtraction (derived_itdoc64__…_minus_ctl): staged 2026-09-13, first measured at 4,096; no 512 pass",
             # the constitution-SI arm (mt_arms4096, 2026-09-14): first measured multi-turn at 4,096; no 512 multi-turn reading on ANY view
             **{(v, SI_CONST): ("prompted constitution SI (mt_arms4096 arm sc, config-only overlay 1d17331e / 390b230a): first measured multi-turn at 4,096 on "
                                "2026-09-14; no 512 multi-turn pass ever ran this cell on any view") for v in MT_ARMS_TAGS},
             # the Arditi / AlphaSteer arms (mt_arms4096, 2026-09-14): first measured multi-turn at 4,096 at the ruled operating points; no 512 pass
             **{(v, c): (f"{c} at its operating point (mt_arms4096 arm {a}, ported trees 68cb31a3 / 63f79fa5): first measured multi-turn at 4,096 on "
                         "2026-09-14; no 512 multi-turn pass ever ran this arm on any view") for v in MT_ARMS_ARM_TAGS for a, c in ARM_CELL.items()},
             # the relu_native PSR cells (2026-09-16): their filed 512 reading is the LENS wave's (greedy 512, lens-served); the 4,096 reading is
             # served NATIVELY. A steered LENS row against a steered NATIVE row is REFUSED by the lane's pairing rule (anchor_pairing: only an
             # UNSTEERED PASSTHROUGH anchor is engine-agnostic), so the truncation-bias pairing is UNDEFINED here, not zero. llama8b's PSR
             # cells fall to NO_CAP512_VIEWS below (that lane has no 512 reading of any cell).
             **{(v, c): (f"relu_native: the filed 512 reading of {c} on {v} is lens-served (greedy 512) and the 4,096 reading is natively served; a "
                         "steered-lens vs steered-native pairing is refused by anchor_pairing.anchor_pairing_verdict (no allowance covers it), so the "
                         "truncation-bias pairing is UNDEFINED, not zero") for v in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b") for c in RELU_NATIVE_CELLS}}
#: whole VIEWS with no filed 512 reading for ANY cell: the Llama-3.1-8B lane has no 512-cap instrument at all -- its BoN and multi-turn were generated
#: at 4,096 outright (grades_llama8b_* roots: grading_provenance.decode_caps every cap 4096, every cell stamped decode_cap_tokens 4096; the lane file
#: transfer_stats_full22_llama8b.json carries no 512 row). Declared, never inferred: a llama (view, cell) never reaches CAP512_SOURCE.
NO_CAP512_VIEWS = {v: ("the Llama-3.1-8B lane has NO 512-cap reading of any cell: its BoN and multi-turn were generated at 4,096 outright (grades_llama8b_* "
                       "roots, decode_caps all 4096; build_export.py decode_cap_tokens column doc), so no 512 pass exists to pair against") for v in LLAMA_VIEWS}


def cap512_side(view: str, cell: str, attack: str, joins: dict) -> tuple[dict | None, dict | None, dict]:
    """The filed 512 reading of the same (view, cell, attack): its reader record, per-goal outcomes, and where it came from — cross-checked
    against the join file the export reads for that cell (k, n, decode_cap_tokens == 512), or refuse."""
    if (view, cell) in NO_CAP512:
        return None, None, {"status": "NO_512_READING", "reason": NO_CAP512[(view, cell)], "decode_cap_tokens": CAP_FILED}
    if view in NO_CAP512_VIEWS:
        return None, None, {"status": "NO_512_READING", "reason": NO_CAP512_VIEWS[view], "decode_cap_tokens": CAP_FILED, "declared_for": "view"}
    if (view, cell) not in CAP512_SOURCE:
        raise SystemExit(f"no 512 source declared for ({view}, {cell}) — add it to CAP512_SOURCE (or NO_CAP512) deliberately")
    gdir, jfile = CAP512_SOURCE[(view, cell)]
    p = RS / gdir / f"two_si_{attack}_aggregate.json"
    agg = load_json(p)
    if not agg or cell not in agg.get("cells", {}):
        raise SystemExit(f"512 aggregate {p} absent or lacks {cell}")
    rec = R2._safe(T1.gated_cell, agg, cell)
    if rec.get("status"):
        raise SystemExit(f"512 side {p}/{cell}: {rec}")
    jrow = ((joins[jfile].get("models") or {}).get(view, {}).get("cells", {}).get(cell, {}) or {}).get(attack) or {}
    if (jrow.get("k"), jrow.get("n")) != (rec["k"], rec["n"]):
        raise SystemExit(f"512 cross-check FAILED for {view}/{cell}/{attack}: aggregate {rec['k']}/{rec['n']} vs {jfile} {jrow.get('k')}/{jrow.get('n')}")
    if jrow.get("decode_cap_tokens") != CAP_FILED:
        raise SystemExit(f"512 side {view}/{cell}/{attack}: join row decode_cap_tokens = {jrow.get('decode_cap_tokens')}, expected {CAP_FILED}")
    src = {"grades_dir": str(RS / gdir), "aggregate": str(p), "aggregate_sha256": sha256_file(p), "join_file": str(RS / jfile),
           "join_view": view, "decode_cap_tokens": CAP_FILED, "namespace_in_join": jrow.get("namespace"),
           "n_responses": ((jrow.get("completion") or {}).get("n_responses")), "k": rec["k"], "n": rec["n"]}
    return rec, R2.attack_outcomes(agg, cell), src


def contrast_with_intersection(row_a: dict, row_b: dict, pa: dict, pb: dict, label: str) -> dict:
    """R2.contrast (b minus a) — and when the reader REFUSES the paired part because the two sides do not carry the same goal ids
    (the 512 GLM cells dropped errored goals; every 4,096 cell has all 259), the same reader on the goal INTERSECTION, labelled."""
    vb = R2.contrast(row_a, row_b, pa, pb)
    if isinstance(vb.get("paired_refused"), dict) and pa and pb:
        shared = sorted(set(pa) & set(pb))
        if shared:
            pc = T1.paired_contrast({g: pa[g] for g in shared}, {g: pb[g] for g in shared})
            pc.update({"n_paired": len(shared), "n_a_only": len(set(pa) - set(pb)), "n_b_only": len(set(pb) - set(pa)),
                       "note": f"paired on the intersection of goals present on both sides ({label}); same reader, same bootstrap and McNemar"})
            vb["paired_on_intersection"] = pc
    return vb


# --------------------------------------------------------------------------- the relu_native wave: cell vs the declared unsteered anchor

def relu_anchor_root(rs: Path, view: str, attack: str) -> tuple[str, str, Path]:
    """(anchor grades root, anchor view, its target dir) for one relu_native (view, attack): the ONE root of the declared anchor view that
    carries two_si_<attack>_aggregate.json (RELU_ANCHOR_VIEW / RELU_ANCHOR_ROOT_FAMILY). Zero or two candidates refuse by name."""
    aview = RELU_ANCHOR_VIEW[view]
    if RELU_ANCHOR_ROOT_FAMILY[aview] == "anchor4096":
        pool = [r for r in ANCHOR_ROOTS if (rs / r).is_dir() and only_dir(rs / r).name == aview]
    else:
        pool = [LLAMA_LANE_ANCHOR_ROOT]
    hits = [r for r in pool if (rs / r / aview / f"two_si_{attack}_aggregate.json").is_file()]
    if len(hits) != 1:
        raise SystemExit(f"relu_native {view}/{attack}: the unsteered anchor must come from exactly ONE root of view {aview!r} carrying "
                         f"two_si_{attack}_aggregate.json -- found {hits} among {pool}; refusing")
    return hits[0], aview, rs / hits[0] / aview


def store_pairing_stamp(cell_dir: Path, cell: str, view: str, source: str):
    """One PairingStamp for a multi-turn store: projected from EVERY shard's served_target.json by the lane's adapter
    (anchor_pairing.stamp_from_served_target) and required to be ONE distinct stamp across the shards, or refuse."""
    stamps = {AP.stamp_from_served_target(json.loads(p.read_text()), cell=cell, view=view, source=source)
              for p in sorted(cell_dir.glob("shard*/served_target.json"))}
    if len(stamps) != 1:
        raise SystemExit(f"{cell_dir}: served_target.json pairing identity not uniform across shards -- {len(stamps)} distinct stamps: "
                         f"{sorted(map(str, stamps))[:2]}; refusing")
    return next(iter(stamps))


def relu_vs_anchor(rs: Path, view: str, cell: str, attack: str, rec: dict, outcomes: dict, cell_dir: Path) -> dict:
    """The relu_native cell MINUS its declared unsteered `none` anchor, paired on goals through the campaign's readers (transfer_r2.contrast:
    10,000-draw bootstrap seed 0 + exact McNemar; the goal INTERSECTION where the two sides differ), with the lane's PAIRING VERDICT computed
    from both stores' own records and stamped on the block: an unsteered passthrough anchor on the lens engine against a natively served
    steered row is admitted under PASSTHROUGH_ANCHOR_ALLOWANCE (a decision, stamped, filterable), anything else REFUSES."""
    aroot, aview, atdir = relu_anchor_root(rs, view, attack)
    aagg = load_json(atdir / f"two_si_{attack}_aggregate.json")
    if aagg.get("target") != aview or aagg.get("private_run") != f"sdfbase_full22_{aview}" or "none" not in (aagg.get("cells") or {}):
        raise SystemExit(f"{atdir}/{attack}: anchor aggregate target {aagg.get('target')!r} / private_run {aagg.get('private_run')!r} / cells "
                         f"{sorted(aagg.get('cells') or {})} do not carry a `none` reading of view {aview!r}; refusing")
    ra = R2._safe(T1.gated_cell, aagg, "none")
    if ra.get("status"):
        raise SystemExit(f"{atdir}/{attack}: the anchor reader returned {ra}; refusing")
    oa = R2.attack_outcomes(aagg, "none")
    atroot = resolve_transcripts_root(rs, aroot, aview)
    arun = atroot / f"sdfbase_full22_{aview}"          # addressed by view: an overridden root (llama8b_lane) holds one run per view
    if not arun.is_dir() or arun.name != aagg.get("private_run"):
        raise SystemExit(f"{atroot}: run {arun.name!r} absent or != the anchor aggregate's private_run {aagg.get('private_run')!r}; refusing")
    steered = store_pairing_stamp(cell_dir, cell, view, "relu_native served_target.json")
    anchor = store_pairing_stamp(arun / attack / "none", "none", aview, f"{aroot} served_target.json")
    v = AP.anchor_pairing_verdict(steered, anchor)
    if not v.admitted:
        raise SystemExit(f"relu_native {view}/{cell}/{attack} vs {aroot}/{aview} `none`: pairing REFUSED by anchor_pairing -- {v.reason}")
    vb = contrast_with_intersection(ra, rec, oa, outcomes, "the anchor and the cell carry different goal sets")
    vb["reads_as"] = (f"cell MINUS the declared unsteered `none` anchor ({aroot}/{aview}), both at cap {CAP} (bare = the anchor, cell = {cell}); the anchor "
                      f"is the LENS-era passthrough leg and the cell is served natively -- admitted under {v.allowance} (block `anchor`)")
    bcg = ((aagg["cells"]["none"].get("attack_metrics") or {}).get("by_category_group") or {})
    vb["anchor_reading"] = {**{k: ra.get(k) for k in ("k", "n", "p", "lo", "hi", "asr_grid")},
                            "by_category_group": {g: {"n_goals": x["n_goals"], "n_jailbroken": x["n_jailbroken"], "asr": x["asr"]} for g, x in bcg.items()}}
    vb["anchor"] = {"anchor_root": aroot, "anchor_view": aview, "anchor_cell": "none", "anchor_aggregate": str(atdir / f"two_si_{attack}_aggregate.json"),
                    "anchor_aggregate_sha256": sha256_file(atdir / f"two_si_{attack}_aggregate.json"), "anchor_written_at": aagg.get("written_at"),
                    "anchor_transcripts_root": str(atroot),
                    "anchor_engine": v.anchor_engine, "steered_engine": v.steered_engine, "pairing_allowance": v.allowance, "pairing_admitted": v.admitted,
                    "pairing_reason": v.reason,
                    "anchor_serving_identity": anchor.serving_identity, "steered_serving_identity": steered.serving_identity,
                    "anchor_object_sha256": anchor.object_sha, "steered_object_sha256": steered.object_sha,
                    "anchor_base_model": anchor.model, "steered_base_model": steered.model,
                    "pairing_module": {"path": str(AP.__file__), "sha256": sha256_file(AP.__file__), "allowance_constant": AP.PASSTHROUGH_ANCHOR_ALLOWANCE}}
    return vb


# --------------------------------------------------------------------------- main

# The attacker's identity, PER (root, attack, cell), read from the store's own shard sidecars -- the authority -- rather than from the
# grader's completion.json / grading_provenance.json, which hold ONE attacker_identity_check written by the LAST family graded into the
# root (a root graded crescendo, then pair, then actorattack carries only actorattack's entry), so the lane file read the other
# families as "recorded" when their sidecars say `unrecorded`. Measured 2026-09-14: every cap-4,096 multi-turn sidecar on BOTH clusters
# records attacker_quantization='unrecorded' (attacker_pool.sbatch never passes --attacker-serving); the harness refuses to aggregate on
# it and the grader RECORDS that refusal as a caveat and proceeds, so every such row was always graded under an identity attested from
# elsewhere (the config's attacker block + WAVE_MANIFEST attacker_pool). That is a DISTINCT provenance class, carried per row, not a
# note. Same 8-field tuple as the grader's check.
ATTACKER_ID_FIELDS = ("attacker_engine", "attacker_model", "attacker_snapshot", "attacker_quantization",
                      "attacker_prefix_caching", "attacker_kv_cache_dtype", "scorer_engine", "target_engine")
def attacker_identity_of(cell_dir: Path, gp: dict) -> dict:
    tuples: dict = {}; n = 0; missing = 0
    for sd in sorted(p for p in cell_dir.glob("shard*") if p.is_dir()):
        n += 1; ae = sd / "attacker_engine.json"
        if not ae.is_file():
            missing += 1; continue
        a = load_json(ae) or {}
        tuples[json.dumps({k: a.get(k) for k in ATTACKER_ID_FIELDS}, sort_keys=True)] = tuples.get(json.dumps({k: a.get(k) for k in ATTACKER_ID_FIELDS}, sort_keys=True), 0) + 1
    if n == 0 or missing or len(tuples) != 1:
        raise SystemExit(f"{cell_dir}: attacker identity not uniform/complete over the shards ({len(tuples)} distinct tuples, {missing} of {n} shards without "
                         f"attacker_engine.json) -- a filed root must have passed the grader's uniformity check; refusing")
    ident = json.loads(next(iter(tuples)))
    unrec = [k for k, v in ident.items() if v in (None, "unrecorded", "unknown")]
    aic = gp.get("attacker_identity_check") or {}
    att = next((e.get("independent_check", {}).get("attested_elsewhere") for e in aic.values() if isinstance(e, dict)), None)
    return {"provenance_class": ("attested-from-elsewhere" if unrec else "sidecar-recorded"),
            "recorded_identity": ident, "fields_unrecorded_in_sidecar": unrec, "n_shards": n,
            "attested_elsewhere": att, "attested_source": ("the root's grading_provenance.json attacker_identity_check (config attacker block + WAVE_MANIFEST "
                                                            "attacker_pool; uniform per root)" if att else None),
            "sidecar_source": str(cell_dir / "shard*/attacker_engine.json")}


def main(argv=None) -> None:
    global RS   # --rs may repoint the run-state root; declared before any use of RS in this scope
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RS / "transfer_stats_full22_cap4096.json"))
    ap.add_argument("--rs", default=str(RS), help="run-state/full22 (grades_* and transcripts_* roots live here)")
    args = ap.parse_args(argv)
    rs = Path(args.rs)
    RS = rs

    joins = {j: (load_json(rs / j) or {}) for j in (J_ALL, J_GLM, J_NEMO, J_PROMPTED)}
    for j, d in joins.items():
        if not d:
            raise SystemExit(f"join file {rs / j} absent — the 512 cross-check needs it")
    prior_meta = joins[J_ALL].get("meta", {})

    models: dict = {}            # view -> model block (canonical)
    models_twins: dict = {}      # view -> model block holding the OTHER cluster's reading of a key already filed in models (TWIN_RULE)
    variants: dict = {}          # tag -> {"variant": ..., "models": {view -> model block}} (+ "models_twins" once a variant twin is filed)
    seen: dict = {}              # (view, cell, attack, window, served_cluster) -> root, canonical only: a second reading of one key FROM THE SAME
                                 # CLUSTER is a defect, not a merge; the other cluster's reading is a twin, placed by place_reading (TWIN_RULE)
    primary: dict = {}           # (namespace or variant tag, view, cell, attack, window) -> the reading holding the cell slot
    twin_status: list = []
    roots_meta: dict = {}
    count_rows: list = []
    landing_present: dict = {}   # landing root -> present on disk this run


    def process_root(root_name: str, ns: str, variant: dict | None) -> None:
        root = rs / root_name
        if not root.is_dir():
            raise SystemExit(f"grades root {root} absent")
        tdir = only_dir(root)
        view = tdir.name                                       # THE VIEW IS THE TARGET DIRECTORY, never the root's name
        if view not in VIEW_KIND:
            raise SystemExit(f"{root}: target directory {view!r} is not a known view")
        # the transcripts root: the lanes' pull scripts name it by the lane KEY (transcripts_obj4096_gemma4e4b holds the gemma4e4b_it run) while
        # the grades root carries the TARGET (grades_obj4096_gemma4e4b_it), so both spellings are tried; the match is then PROVED, not assumed:
        # the root's single run directory must be the aggregate's private_run, and verify_cap() ties every cell's timing / response counts to
        # the grader's own completion stamp.
        troot = resolve_transcripts_root(rs, root_name, view)
        run_dir = only_dir(troot)
        served_cluster, sc_prov = served_cluster_of(root_name, troot)     # WHICH CLUSTER served this root (ruling 2026-09-13 16:45)
        gp = load_json(tdir / "grading_provenance.json") or {}
        j = gp.get("judge") or {}
        # A root that EXISTS but holds no aggregate and no judge record yet is MID-GRADE, not a provenance
        # violation. Until 2026-09-14 this refused the WHOLE build on `judge None / None is not the pinned`,
        # and since a chain creates its grade directory when it STARTS, any in-flight grade took the lane file
        # down with it. That is how the lane file sat stale at 05:01Z while eleven legs graded into it: the
        # refusal fired every run, its wording matched none of the refresher's error patterns, and the
        # refresher logged EXPORTED off the previous good file. To a reader, absent and mid-grade are the same
        # state -- nothing to emit yet. A WRONG judge on a root that HAS graded is still fatal, which is the
        # check that actually protects anything. Tolerant of the not-yet, strict about the wrong.
        if not j and not any((tdir / f"two_si_{a}_aggregate.json").exists() for a in MULTITURN):
            print(f"{tdir}: grade dir exists, no aggregate and no judge record yet -- MID-GRADE, nothing emitted this build")
            return
        if j.get("judge_model") != PINNED_JUDGE or j.get("rubric") != PINNED_RUBRIC:
            raise SystemExit(f"{tdir}: judge {j.get('judge_model')} / {j.get('rubric')} is not the pinned {PINNED_JUDGE} / {PINNED_RUBRIC}")
        aggs = {a: load_json(tdir / f"two_si_{a}_aggregate.json") for a in MULTITURN}
        aggs = {a: g for a, g in aggs.items() if g}
        if not aggs:
            raise SystemExit(f"{tdir}: no multi-turn aggregate")
        def new_block(twin: bool) -> dict:
            blk = {
                "label": VIEW_LABEL[view] + (f" — VARIANT {variant['tag']}: {variant['reason']}" if variant else "")
                         + (" — TWIN READINGS: the other cluster's grading of keys already filed in the primary block (meta.twin_rule)" if twin else ""),
                "view": VIEW_KIND[view],
                "base_target": base_target(view), "namespace": ns, "decode_cap_tokens": CAP, "served_alpha": None,
                "lens_reduction_mode": WAVE_LENS_MODE, "attacks_present": [], "expected_cells": [], "cells": {}, "vs_bare": {},
                "vs_bare_note": (f"NOT COMPUTED by this builder for the {NS} / {NS_VARIANT} namespaces: the unsteered `none` anchor at {CAP} (grades_anchor4096_*, "
                                 "2026-09-13) is exported as its own cell of this view wherever its aggregate exists (meta.anchor_status); wiring the same-cap "
                                 "cell-vs-bare contrast for those namespaces is a follow-up. The same view's 512 unsteered row remains the floor the rung collapse "
                                 f"reads. For the {NS_RELU} namespace (2026-09-16) vs_bare IS filed per cell x attack against the declared unsteered anchor "
                                 "(meta.relu_native_wave.anchors), the pairing verdict stamped on each block (`anchor`). Nothing is fabricated here."),
                "vs_cap512": {}, "vs_canonical_sibling": {} if variant else None, "sources": {}, "judge": {"strongreject": {"per_root": {}}},
                "completion": {}}
            if twin:
                blk["twin_note"] = TWIN_RULE
            return blk

        target_block = variants[variant["tag"]]["models"] if variant else models
        mb = target_block.setdefault(view, new_block(False))
        mb["served_alpha"] = next(iter(aggs.values())).get("served_alpha")

        def twins_block() -> dict:
            tb = variants[variant["tag"]].setdefault("models_twins", {}) if variant else models_twins
            return tb.setdefault(view, new_block(True))
        rmeta = {"root": str(root), "view": view, "private_run": next(iter(aggs.values())).get("private_run"), "transcripts_root": str(troot),
                 "attacks": sorted(aggs), "aggregates": {a: {"path": str(tdir / f"two_si_{a}_aggregate.json"),
                                                            "sha256": sha256_file(tdir / f"two_si_{a}_aggregate.json"),
                                                            "written_at": g.get("written_at"), "cells": sorted(g["cells"])} for a, g in aggs.items()},
                 "completion_json_sha256": sha256_file(tdir / "completion.json"), "grading_provenance_sha256": sha256_file(tdir / "grading_provenance.json"),
                 "grader": {"driver": gp.get("driver"), "config": gp.get("config"), "worktree": gp.get("worktree"), "decode_caps_in_config": gp.get("decode_caps"),
                            "goalset": gp.get("goalset"), "finished_at": gp.get("finished_at")},
                 "judge": {**{k: j.get(k) for k in ("judge_model", "rubric", "cache_kind", "pass_id", "cache_rows", "cache_rows_seeded", "org", "concurrency_initial")},
                           **{k: gp.get(k) for k in ("org_final", "escalated_to_highprio", "n_429", "wall_s", "n_judgments_estimated", "unscored_retry_events")}},
                 "attacker_identity_check": gp.get("attacker_identity_check"), "namespace": ns, "variant": (variant["tag"] if variant else None),
                 "served_cluster": served_cluster, "served_cluster_provenance": sc_prov, "serving_windows": {}}
        if root_name in MT_ARMS_ARM_ROOTS:
            # the arm root's SERVED operating point, derived from the config its legs ran with and checked against the
            # declaration (a contradiction refuses the build; see served_operating_point_of)
            rmeta["served_operating_point"] = served_operating_point_of(gp, arm_of_root(root_name), view)
        roots_meta[root_name] = rmeta
        mb["judge"]["strongreject"]["per_root"][root_name] = rmeta["judge"]

        def install(blk: dict, it: dict) -> None:
            """file one reading into a model block, in the filed key order (cells -> completion -> sources -> vs_cap512 -> vs_canonical_sibling)"""
            cell, attack, rec = it["cell"], it["attack"], it["rec"]
            if attack not in blk["attacks_present"]:
                blk["attacks_present"].append(attack)
            if cell not in blk["cells"]:
                lab, grp = label_group(view, cell)
                blk["cells"][cell] = {"label": lab, "group": grp, "view": VIEW_KIND[view], "namespace": ns, "decode_cap_tokens": CAP,
                                      "lens_reduction_mode": WAVE_LENS_MODE, "on_overlay": cell in B.OVERLAY_CELLS, "completion": {}}
                blk["expected_cells"].append(cell)
            blk["cells"][cell][attack] = rec
            blk["cells"][cell]["completion"][attack] = rec["completion"]
            blk["sources"].setdefault(cell, {})[attack] = it["root"]
            blk["vs_cap512"].setdefault(cell, {})[attack] = it["vc"]
            if it.get("vb") is not None:                      # the relu_native namespace files its cell-vs-anchor contrast (relu_vs_anchor)
                blk["vs_bare"].setdefault(cell, {})[attack] = it["vb"]
            if variant:
                blk["vs_canonical_sibling"].setdefault(cell, {})[attack] = it["sib"]

        def twin_order(it: dict):
            return (it["written_at"] or "9999", 0 if it["served_cluster"] == SERVED_CLUSTER_DEFAULT else 1)

        def place_reading(it: dict) -> None:
            """TWIN_RULE: the first reading of a (view, cell, attack, window) takes the cell slot; a second one from another cluster is a twin --
            the earlier-written aggregate holds the slot whichever root was processed first, the other goes to the twins block."""
            pkey = (variant["tag"] if variant else ns, view, it["cell"], it["attack"], it["window"])
            prev = primary.get(pkey)
            if prev is None:
                install(mb, it); it["count_row"]["slot"] = "primary"; primary[pkey] = it
                return
            if prev["served_cluster"] == it["served_cluster"]:
                raise SystemExit(f"{pkey} graded in BOTH {prev['root']} and {it['root']} on cluster {it['served_cluster']} — two readings of one key from "
                                 f"one cluster; refusing")
            first, second = (it, prev) if twin_order(it) < twin_order(prev) else (prev, it)
            tb = twins_block()
            install(mb, first); install(tb, second)
            tb["judge"]["strongreject"]["per_root"][second["root"]] = roots_meta[second["root"]]["judge"]
            first["count_row"]["slot"] = "primary"; first["count_row"].pop("twin_of", None)
            second["count_row"]["slot"] = "twin"; second["count_row"]["twin_of"] = first["root"]
            primary[pkey] = first
            twin_status.append({"namespace": ns, "variant": (variant["tag"] if variant else None), "view": view, "cell": it["cell"], "attack": it["attack"],
                                "serving_window_tokens": it["window"],
                                "primary": {k: first[k] for k in ("root", "served_cluster", "written_at")},
                                "twin": {k: second[k] for k in ("root", "served_cluster", "written_at")},
                                "slot_rule": "earlier written_at holds models.<view> (tie -> cluster_a); the other is under models_twins.<view>"})
        for attack, agg in sorted(aggs.items()):
            if agg.get("target") != view or agg.get("private_run") != f"sdfbase_full22_{view}" or run_dir.name != agg.get("private_run"):
                raise SystemExit(f"{tdir}/{attack}: aggregate target {agg.get('target')!r} / private_run {agg.get('private_run')!r} / transcripts run "
                                 f"{run_dir.name!r} disagree with the target directory {view!r} — refusing")
            if agg.get("attack_tier") != "multi_turn" or agg.get("wave") != "full22":
                raise SystemExit(f"{tdir}/{attack}: tier {agg.get('attack_tier')} / wave {agg.get('wave')}")
            for cell in sorted(agg["cells"]):
                cell_dir = run_dir / attack / cell
                rec, outcomes = read_cell(agg, cell, attack, ns, cell_dir)
                rec["root"] = root_name                                            # the pointer: (view, attack, cell) -> root is ambiguous (retries, variants)
                rec["attacker_identity"] = attacker_identity_of(cell_dir, gp)      # per (root, attack, cell), from the sidecars (see attacker_identity_of)
                rmeta.setdefault("attacker_identity_by_attack_cell", {})[f"{attack}/{cell}"] = rec["attacker_identity"]["provenance_class"]
                window = rec["instrument"]["serving_window_tokens"]
                rmeta["serving_windows"][attack] = window
                # the served-cluster record is per (lane, attack) root; a root shared by three attacks holds the LAST pulled attack's window, so the
                # window cross-check binds only for the attack the record names
                if sc_prov.get("attack") == attack and sc_prov.get("serving_window_tokens") is not None and sc_prov["serving_window_tokens"] != window:
                    raise SystemExit(f"{cell_dir}: served_cluster.json records serving window {sc_prov['serving_window_tokens']} for {attack}, the shards' "
                                     f"served_target.json say {window} -- not the same standup; refusing")
                if variant and variant.get("window_tokens") and window != variant["window_tokens"]:
                    raise SystemExit(f"{cell_dir}: window variant {variant['tag']} expects a {variant['window_tokens']}-token serving window, the shards say {window}; refusing")
                if sc_prov["on_rows"]:
                    rec["served_cluster"] = served_cluster          # derived from a record: the row carries it (defaulted roots: meta only)
                key = (view, cell, attack, window, served_cluster)
                if not variant:
                    if key in seen:
                        raise SystemExit(f"{key} graded in BOTH {seen[key]} and {root_name} — two canonical readings of one key from one cluster; refusing")
                    seen[key] = root_name
                # the cell's own 4,096 - 512 paired change (truncation bias): the 512 side from the root the export's join read, cross-checked
                rec512, o512, src512 = cap512_side(view, cell, attack, joins)
                if rec512 is None:      # declared NO_CAP512: the pairing is undefined for this cell, and the record says so
                    vc = {"status": "NO_512_READING", "reason": src512["reason"], "cap512_source": src512,
                          "reads_as": f"no filed {CAP_FILED}-cap reading exists for this cell (first measured at {CAP}); the truncation-bias pairing is UNDEFINED here, not zero"}
                else:
                    vc = contrast_with_intersection(rec512, rec, o512, outcomes, "512 side dropped errored goals")
                    vc["reads_as"] = f"{CAP}-cap reading MINUS the filed {CAP_FILED}-cap reading of the SAME cell (cell = {CAP}, bare = {CAP_FILED}); truncation bias, not a defence effect"
                    vc["cap512_source"] = src512
                # the variant against its canonical sibling at the same cap (variant MINUS canonical), paired on goals
                sib = None
                if variant:
                    sroot = (rs / variant["sibling_root"]) if variant.get("sibling_root") else None
                    sdir = only_dir(sroot) if (sroot is not None and sroot.is_dir()) else None
                    sagg = load_json(sdir / f"two_si_{attack}_aggregate.json") if sdir is not None else None
                    skey = {"view": view, "attack": attack, "cell": cell, "decode_cap_tokens": CAP, "namespace": NS,
                            "sibling_root": (str(sroot) if sroot is not None else None), "status_in_export": "graded"}
                    if variant.get("window_tokens"):
                        skey["serving_window_tokens"] = variant["sibling_window_tokens"]       # the sibling is the SAME cell at the arms' window
                    if sagg and cell in sagg.get("cells", {}) and sdir.name == view:
                        srec = R2._safe(T1.gated_cell, sagg, cell)
                        sib = contrast_with_intersection(srec, rec, R2.attack_outcomes(sagg, cell), outcomes, "sibling")
                        sib["reads_as"] = f"variant ({variant['reason']}) MINUS the canonical sibling at cap {CAP} (cell = variant, bare = canonical)"
                        sib["sibling_key"] = skey
                        sib["term_moved"] = variant["term_moved"]
                    else:
                        sib = {"status": "ABSENT", "reason": variant.get("sibling_note", "no canonical sibling aggregate"), "sibling_key": skey,
                               "term_moved": variant["term_moved"]}
                    if variant.get("window_tokens"):
                        sib["variant_window_tokens"] = variant["window_tokens"]
                # the relu_native namespace: the cell against its declared unsteered anchor, paired on goals, the pairing verdict stamped
                vb = relu_vs_anchor(rs, view, cell, attack, rec, outcomes, cell_dir) if ns == NS_RELU else None
                count_row = {"view": view, "attack": attack, "cell": cell, "namespace": ns, "variant": (variant["tag"] if variant else None),
                             "root": root_name, "served_cluster": served_cluster, "slot": None,
                             "cap512": ({"k": rec512["k"], "n": rec512["n"], "n_responses": src512["n_responses"], "source": src512["grades_dir"]}
                                        if rec512 is not None else None),
                             f"cap{CAP}": {"k": rec["k"], "n": rec["n"], "n_responses": rec["completion"].get("n_responses"),
                                           "at_cap_k": rec["at_cap_finish_length"]["k"], "max_completion_tokens": rec["decode_cap_verification"]["max_completion_tokens"],
                                           "grader_stamp": rec["decode_cap_tokens_grader_stamp"], "decode_mode": rec["instrument"]["decode_mode"],
                                           "serving_window": window,
                                           "n_window_clamped": rec["instrument"]["n_window_clamped"],
                                           "n_clamped_censored": rec["decode_cap_verification"]["n_clamped_calls_censored"]},
                             "vs_cap512_delta_pp": ((vc.get("paired") or vc.get("paired_on_intersection") or {}).get("delta_pp")),
                             "vs_sibling_delta_pp": (((sib or {}).get("paired") or (sib or {}).get("paired_on_intersection") or {}).get("delta_pp") if sib else None),
                             "vs_bare_delta_pp": (((vb or {}).get("paired") or (vb or {}).get("paired_on_intersection") or {}).get("delta_pp") if vb else None),
                             "anchor_root": ((vb or {}).get("anchor") or {}).get("anchor_root")}
                count_rows.append(count_row)
                place_reading({"view": view, "cell": cell, "attack": attack, "window": window, "root": root_name, "served_cluster": served_cluster,
                               "written_at": agg.get("written_at"), "rec": rec, "vc": vc, "sib": sib, "vb": vb, "count_row": count_row})
        mb["completion"] = {c: mb["cells"][c]["completion"] for c in mb["cells"]}
        tb = (variants[variant["tag"]].get("models_twins", {}) if variant else models_twins).get(view)
        if tb is not None:
            tb["completion"] = {c: tb["cells"][c]["completion"] for c in tb["cells"]}

    # the Arditi / AlphaSteer arms and the llama constitution restart are LANDING roots too (registered when their legs were queued, so a
    # landing is never invisible): absent = recorded in meta.landing_status, never a refusal
    all_landing = (set(LANDING_ROOTS) | set(MT_ARMS_ROOTS) | set(MT_ARMS_ARM_ROOTS) | set(MT_ARMS_R2_ROOTS)
                   | {r for r, v in VARIANT_ROOTS.items() if v.get("landing")})
    for r in CANONICAL_ROOTS:
        if r in all_landing:
            landing_present[r] = (rs / r).is_dir()
            if not landing_present[r]:
                continue                                   # absent landing root: recorded in meta.landing_status, nothing raised
        process_root(r, namespace_of_root(r), None)        # NS for every root but the natively served mt4kap arm roots (ARM_ROOT_NAMESPACE)
    for r, v in VARIANT_ROOTS.items():
        if v.get("landing"):
            landing_present[r] = (rs / r).is_dir()
            if not landing_present[r]:
                continue                                   # absent landing root: recorded in meta.landing_status; a tag with no root present is not registered
        if v["tag"] not in variants:
            same_tag = [rr for rr, vv in VARIANT_ROOTS.items() if vv["tag"] == v["tag"]]
            if len(same_tag) == 1:            # the two filed one-root variants: their summary is unchanged
                summary = {**v, "root": str(rs / r), "sibling_key_fields": list(VARIANT_SIBLING_KEY_FIELDS), "namespace": NS_VARIANT, "status_in_export": "variant"}
            else:                             # a tag spanning several roots (w24k): one summary, the per-root facts under roots
                summary = {**{k: v.get(k) for k in ("tag", "term_moved", "term_moved_detail", "reason", "what_is_held_fixed", "window_tokens", "sibling_window_tokens", "why", "landing")},
                           "roots": {rr: {"root": str(rs / rr), "cluster": vv.get("cluster"), "arm": vv.get("arm"), "cell": vv.get("cell"),
                                          "sibling_root": vv.get("sibling_root"), "sibling_note": vv.get("sibling_note"), "present": (rs / rr).is_dir()}
                                     for rr, vv in VARIANT_ROOTS.items() if vv["tag"] == v["tag"]},
                           "sibling_key_fields": list(W24K_SIBLING_KEY_FIELDS if v.get("window_tokens") else VARIANT_SIBLING_KEY_FIELDS),
                           "namespace": NS_VARIANT, "status_in_export": "variant", "twin_rule": TWIN_RULE}
            variants[v["tag"]] = {"variant": summary, "models": {}}
        process_root(r, NS_VARIANT, v)
    # the relu_native wave (its own namespace, canonical placement): every declared TAG must have a root -- the wave is complete by declaration
    _relu_missing = [tag for tag in RELU_NATIVE_TAGS if not any((rs / r).is_dir() and r.endswith(f"_{tag}") for r in RELU_NATIVE_ROOTS)]
    if _relu_missing:
        raise SystemExit(f"REFUSING: the relu_native wave is declared COMPLETE but no grades root exists under {rs / RELU_NATIVE_DIR} for tag(s) "
                         f"{_relu_missing} (grammar relu_native/grades_relu_native_<clu>_<TAG>)")
    for r in RELU_NATIVE_ROOTS:
        process_root(r, NS_RELU, None)
    for mb in (list(models.values()) + list(models_twins.values())
               + [m for v in variants.values() for blk in ("models", "models_twins") for m in v.get(blk, {}).values()]):
        mb["attacks_present"] = [a for a in MULTITURN if a in mb["attacks_present"]]
        mb["attacks_pending"] = []
        mb["judge"]["strongreject"].update({"model": PINNED_JUDGE, "rubric": PINNED_RUBRIC, "cache_kind": "sr"})

    def landing_entry(r: str) -> dict:
        v = VARIANT_ROOTS.get(r)
        base = {"namespace": (NS_VARIANT if v else namespace_of_root(r)), "variant": (v["tag"] if v else None),
                "wave": ("mt_arms4096" if r in MT_ARMS_ROOTS or r in MT_ARMS_W24K_ROOTS else "qc4k (R2-D288)")}
        if r in roots_meta:
            return {"present": True, "view": roots_meta[r]["view"], "attacks_graded": roots_meta[r]["attacks"],
                    "attacks_absent": [a for a in MULTITURN if a not in roots_meta[r]["attacks"]],
                    "note": "partial by design while the lane's legs land; an absent attack has no aggregate yet and gets no row",
                    "served_cluster": roots_meta[r]["served_cluster"], **base}
        return {"present": False, "note": LANDING_NOTE[r], **base}

    def variant_meta(v: dict) -> dict:
        same_tag = [rr for rr, vv in VARIANT_ROOTS.items() if vv["tag"] == v["tag"]]
        if len(same_tag) == 1:
            return {k: v[k] for k in ("term_moved", "reason", "sibling_root", "what_is_held_fixed")}
        return {**{k: v.get(k) for k in ("term_moved", "term_moved_detail", "reason", "what_is_held_fixed", "window_tokens", "sibling_window_tokens", "why")},
                "roots": {rr: {"sibling_root": VARIANT_ROOTS[rr].get("sibling_root"), "sibling_note": VARIANT_ROOTS[rr].get("sibling_note"),
                               "cluster": VARIANT_ROOTS[rr].get("cluster"), "present": landing_present.get(rr, (rs / rr).is_dir())} for rr in same_tag}}

    def ns_cells(ns: str) -> dict:
        """the cells each view carries IN one namespace (a view block may hold several namespaces since relu_native; views with none are omitted)"""
        return {v: cs for v, m in models.items() if (cs := [c for c in m["expected_cells"] if m["cells"][c].get("namespace") == ns])}

    def ns_readouts(ns: str) -> dict:
        return {v: [a for a in MULTITURN if any(isinstance(m["cells"][c].get(a), dict) for c in cs)] for v, cs in ns_cells(ns).items() for m in [models[v]]}

    relu_anchors = {f"{r['view']}|{r['attack']}": {"anchor_root": r["anchor_root"], "anchor_view": RELU_ANCHOR_VIEW[r["view"]]}
                    for r in count_rows if r["namespace"] == NS_RELU and r["slot"] == "primary"}
    out = {"meta": {
        # the arm operating points, published whole so build_export can label EVERY arm row (multi-turn, petri,
        # best-of-N, static) from this one declaration rather than keeping a copy that goes stale.
        "arm_operating_points": {f"{_v}|{_a}": {"id": _op[0], "text": arm_op_label(_op)}
                                 for (_v, _a), _op in sorted(ARM_OPERATING_POINT.items())},
        "arm_operating_point_undeclared": ARM_OP_UNDECLARED,
        # per arm root: what its legs were SERVED at (from the recorded per-arm config) vs what the table declares. Every
        # entry here is `verified` or the build would have refused; `unverifiable` means the config left the disk.
        "arm_operating_point_verification": {r: roots_meta[r]["served_operating_point"] for r in sorted(roots_meta)
                                             if "served_operating_point" in roots_meta[r]},
        **{k: prior_meta.get(k) for k in ("threshold", "bon_n_max", "interval", "readouts", "gated_goal_n", "lens_reduction_rule", "grader_entry_point")},
        "pairing": ("vs_cap512: the SAME cell at 4,096 minus its filed 512 reading, paired on goals (10,000-draw percentile bootstrap seed 0 + exact McNemar; "
                    "on the goal INTERSECTION where the 512 side dropped errored goals) — truncation bias. vs_bare is NOT computed here for the cap4096 / "
                    "cap4096_variant namespaces: the unsteered anchor at 4,096 is exported as its own `none` cell where graded (meta.anchor_status); wiring the "
                    "same-cap contrast for them is a follow-up. For the relu_native namespace vs_bare IS filed: cell minus the declared unsteered `none` anchor "
                    "(same readers), the lane's pairing verdict stamped on the block (meta.relu_native_wave)."),
        "lens_reduction_mode": WAVE_LENS_MODE,
        "decode_caps": {"multi_turn_target_reply": CAP, "filed_wave_multi_turn_target_reply": CAP_FILED,
                        "stamp_override": ("every aggregate row in these roots carries decode_cap_tokens 512 — the grader copies the worktree config's "
                                           "attacks.target_max_new_tokens; the legs were driven at 4,096 on the command line. Overridden to 4096 on every "
                                           "emitted cell and proved per cell from target_timing.jsonl (decode_cap_verification); the grader's value is kept "
                                           "as decode_cap_tokens_grader_stamp")},
        "namespaces": {
            NS: {"label": "the cap-4,096 multi-turn legs, canonical instrument: each model's shipped decode, explicit serving windows, 259-goal goalsets, no window clamps",
                 "roots": CANONICAL_ROOTS, "graded_by": "glm_lane/grade_full22_glm.py in goalset mode (haiku-4-5 StrongREJECT v1, Low Prio)",
                 "cells": ns_cells(NS), "readouts": ns_readouts(NS)},
            NS_RELU: {"label": ("the relu_native wave (relu-steering-native-evals, 2026-09-16): the three PSR cells on the four main views, served on the NATIVE "
                                "multi-LoRA engine (lora_native) at the 4,096 cap, paired against the lens-era unsteered anchors under the lane's declared allowance"),
                      "roots": RELU_NATIVE_ROOTS, "graded_by": "relu_native/grade_full22_native.py (the lane's own native-aware grader; haiku-4-5 StrongREJECT v1, Low Prio)",
                      "cells": ns_cells(NS_RELU), "readouts": ns_readouts(NS_RELU)},
            NS_ALLPOS_NATIVE: {"label": ("the arditi_allpos multi-turn wave (arditi-qwen-all-positions Phase G, 2026-09-17): the all-positions Arditi arm at its "
                                         "declared operating point on qwen35_4b_it, served on the NATIVE multi-LoRA engine (lora_native) at the 4,096 cap as "
                                         "SINGLE-ROOT stores (one per cluster token, at the run-state root); vs_bare is not filed here, as for the cap4096 "
                                         "namespace -- the unsteered anchor at 4,096 is its own `none` cell of the view"),
                               "roots": ALLPOS_NATIVE_ROOTS, "graded_by": "relu_native/grade_full22_native.py (the native-aware grader; haiku-4-5 StrongREJECT v1, Low Prio)",
                               "serving": NAMESPACE_SERVING[NS_ALLPOS_NATIVE],
                               "cells": ns_cells(NS_ALLPOS_NATIVE), "readouts": ns_readouts(NS_ALLPOS_NATIVE)},
            NS_VARIANT: {"label": "the two one-term-moved variants (R2-D272): exported, non-canonical, paired with the canonical sibling at the same cap",
                         "roots": list(VARIANT_ROOTS), "variants": {v["tag"]: variant_meta(v) for v in VARIANT_ROOTS.values()},
                         "sibling_key_fields": list(VARIANT_SIBLING_KEY_FIELDS),
                         "sibling_key_fields_window_variants": list(W24K_SIBLING_KEY_FIELDS),
                         "status_in_export": "variant (status column), in_artifact_roster False, note names the term, the reason and the sibling key"}},
        "cell_labels": {"main": B.CELL_LABEL, "it": B.IT_LABEL, "prompted": PROMPTED_LABEL, "prompted_constitution": PROMPTED_CONST_LABEL,
                        "prompted_only_view_none": "none (bare instruct, same wave, prompted-only lane's serving job)"},
        "cell_groups": {"main": B.CELL_GROUP, "it": B.IT_GROUP, SI_CONST: "prompted_baseline"},
        "degeneracy": {"status": "PENDING", "reason": "no degeneracy join over the cap-4,096 roots; at-cap per cell from the producer's target_timing (same definition as the join's at_cap)"},
        "instrument_record": "per cell from served_target.json: serving window, clamp counts (0 required), the decode sent (last_request_decode)",
        "roots": roots_meta,
        "anchor_status": {r: {"view": roots_meta[r]["view"], "attacks_graded": roots_meta[r]["attacks"],
                              "attacks_absent": [a for a in MULTITURN if a not in roots_meta[r]["attacks"]],
                              "note": "partial by design: an absent attack has NO AGGREGATE IN THIS ROOT -- this builder does not know whether its leg is running, dead, or graded into another root; it gets no row from this root"}
                          for r in ANCHOR_ROOTS},
        "landing_status": {r: landing_entry(r) for r in LANDING_ROOTS + MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS + MT_ARMS_ARM_ROOTS + MT_ARMS_R2_ROOTS},
        "arms_wave": {"readme": str(rs / "mt_arms4096/README.md"), "cluster_env": str(rs / "mt_arms4096/cluster_env.sh"),
                      "root_grammar": "grades_mt4k<arm>[_w24k]_<clu>_<TAG>; arm sc = prompted__si_constitution, nn = none (same-window PAIR anchor); clu bu = cluster_a, ch = cluster_b; TAG = <key> for view main else <key>_<view>",
                      "cell": SI_CONST, "tags": MT_ARMS_TAGS, "canonical_landing_roots": MT_ARMS_ROOTS, "window_variant_landing_roots": MT_ARMS_W24K_ROOTS,
                      "present_this_run": sorted(r for r in MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS if landing_present.get(r)),
                      "absent_this_run": sorted(r for r in MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS if not landing_present.get(r)),
                      "not_registered": "arditi (mt4kar_*) and alphasteer (mt4kas_*) arms -- blocked on their port; ROOT_SUFFIX restart namespaces (e.g. mt4ksc_w24kr2_*) -- register deliberately when they exist"},
        "relu_native_wave": {"namespace": NS_RELU, "dir": str(rs / RELU_NATIVE_DIR), "tags": RELU_NATIVE_TAGS, "cells": RELU_NATIVE_CELLS, "roots": RELU_NATIVE_ROOTS,
                             "root_grammar": "relu_native/grades_relu_native_<clu>_<TAG> beside relu_native/transcripts_relu_native_<clu>_<TAG>; clu bu = cluster_a, ch = cluster_b; TAG = the main view",
                             "serving": NAMESPACE_SERVING[NS_RELU], "anchor_view": RELU_ANCHOR_VIEW, "anchors": relu_anchors,
                             "pairing_allowance": AP.PASSTHROUGH_ANCHOR_ALLOWANCE,
                             "pairing_module": {"path": str(AP.__file__), "sha256": sha256_file(AP.__file__), "src": str(ANCHOR_PAIRING_SRC)},
                             "pairing_rule": ("anchor_pairing.anchor_pairing_verdict over the two stores' own served_target.json stamps: an unsteered passthrough "
                                              "`none` anchor (any engine) against a natively served steered row is admitted and stamped with the allowance; a "
                                              "steered lens row against a native row, or any other combination, REFUSES the build"),
                             "vs_cap512": "UNDEFINED for every relu_native cell (NO_CAP512 / NO_CAP512_VIEWS): the filed 512 reading is lens-served and steered"},
        "twin_rule": TWIN_RULE,
        "twin_status": twin_status,
        "served_cluster_default": {"value": SERVED_CLUSTER_DEFAULT, "note": SERVED_CLUSTER_DEFAULT_NOTE,
                                   "roots_defaulted": sorted(r for r, m in roots_meta.items() if m["served_cluster_provenance"]["source"] == "default"),
                                   "roots_derived": {r: m["served_cluster_provenance"]["source"] for r, m in roots_meta.items() if m["served_cluster_provenance"]["source"] != "default"},
                                   "on_rows_rule": "a cell record carries served_cluster iff its root's cluster was DERIVED (served_cluster.json, else the _bu_/_ch_ name token); defaulted roots' records are unchanged and their cluster rides meta.roots.<root>.served_cluster + meta.count_table[*].served_cluster"},
        "served_cluster_by_root": {r: m["served_cluster"] for r, m in roots_meta.items()},
        "no_cap512_cells": {f"{v}|{c}": why for (v, c), why in NO_CAP512.items()},
        "no_cap512_views": dict(NO_CAP512_VIEWS),
        "cap512_sources": {f"{v}|{c}": {"grades_dir": g, "join_file": j} for (v, c), (g, j) in CAP512_SOURCE.items()},
        "readers": {"transfer_stats": str(Path(T1.__file__)), "transfer_r2": str(Path(R2.__file__)), "w7_builder": str(Path(B.__file__)),
                    "transfer_stats_sha256": sha256_file(T1.__file__), "transfer_r2_sha256": sha256_file(R2.__file__),
                    "w7_builder_sha256": sha256_file(B.__file__), "this_builder_sha256": sha256_file(Path(__file__).resolve()),
                    "anchor_pairing": str(AP.__file__), "anchor_pairing_sha256": sha256_file(AP.__file__)},
        "count_table": count_rows,
        "written_at": ts(),
    }, "namespaces": {NS: {"models": {v: {"readouts": ns_readouts(NS)[v], "cells": cs} for v, cs in ns_cells(NS).items()}},
                      NS_RELU: {"models": {v: {"readouts": ns_readouts(NS_RELU)[v], "cells": cs} for v, cs in ns_cells(NS_RELU).items()}},
                      NS_ALLPOS_NATIVE: {"models": {v: {"readouts": ns_readouts(NS_ALLPOS_NATIVE)[v], "cells": cs} for v, cs in ns_cells(NS_ALLPOS_NATIVE).items()}},
                      NS_VARIANT: {"models": {f"{t}:{v}": {"readouts": m["attacks_present"], "cells": m["expected_cells"]}
                                              for t, vv in variants.items() for v, m in vv["models"].items()}}},
        "models": models, "models_twins": models_twins, "models_variants": variants, "cross_view": {},
        "drift": {v: R2.not_in_design("no regrade cell in this namespace's design") for v in models}}

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        prev = outp.with_name(outp.name + f".prev-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        outp.rename(prev)   # never overwrite silently: the previous build is kept beside the new one
        print(f"previous build kept at {prev}")
    outp.write_text(json.dumps(out, indent=1) + "\n")

    # --- the verification report: per (view, attack, cell), 512 and 4,096 SIDE BY SIDE ------------------------------------------------
    print(f"\n{'view':18s} {'attack':12s} {'cell':40s} {'ns':16s} {'clu':10s} {'slot':7s} | {'512 k/n':9s} {'512 resp':8s} | {'4096 k/n':9s} {'4096 resp':9s} {'at-cap':7s} {'max_ct':6s} {'stamp':5s} {'decode':16s} {'window':6s} {'clamp':5s} | {'d(4096-512)pp':13s} {'d(var-canon)pp':14s}")
    for r in count_rows:
        a, b = r["cap512"] or {"k": "—", "n": "—", "n_responses": "—"}, r[f"cap{CAP}"]
        d1 = r["vs_cap512_delta_pp"]; d2 = r["vs_sibling_delta_pp"]
        print(f"{r['view']:18s} {r['attack']:12s} {r['cell'][:40]:40s} {r['namespace']:16s} {r['served_cluster']:10s} {r['slot']:7s} | {str(a['k'])+'/'+str(a['n']):9s} {str(a['n_responses']):8s} | "
              f"{str(b['k'])+'/'+str(b['n']):9s} {str(b['n_responses']):9s} {str(b['at_cap_k']):7s} {str(b['max_completion_tokens']):6s} {str(b['grader_stamp']):5s} "
              f"{b['decode_mode']:16s} {str(b['serving_window']):6s} {str(b['n_window_clamped']) + '/' + str(b['n_clamped_censored']):5s} | {('%+.1f' % d1) if d1 is not None else '—':13s} {('%+.1f' % d2) if d2 is not None else ('—' if r['variant'] else ''):14s}")
    n_can = sum(1 for r in count_rows if not r["variant"]); n_var = len(count_rows) - n_can
    n_roots_can = sum(1 for r in CANONICAL_ROOTS if r in roots_meta); n_roots_var = sum(1 for r in VARIANT_ROOTS if r in roots_meta)
    print(f"\n{n_can} canonical (view, attack, cell) readings at {CAP} from {n_roots_can} of {len(CANONICAL_ROOTS)} registered roots; {n_var} variant readings from "
          f"{n_roots_var} of {len(VARIANT_ROOTS)} registered roots; {len(twin_status)} twin key(s) (both clusters graded); "
          f"every 512 side cross-checked against the export's join files; every {CAP} store identity-checked against the grader's completion stamp.")
    for t in twin_status:
        print(f"  twin {t['view']}/{t['attack']}/{t['cell'][:36]} @ window {t['serving_window_tokens']}: slot <- {t['primary']['root']} ({t['primary']['served_cluster']}, "
              f"written {t['primary']['written_at']}); twin -> models_twins: {t['twin']['root']} ({t['twin']['served_cluster']}, written {t['twin']['written_at']})")
    for t, vv in variants.items():
        for v, m in vv["models"].items():
            for c, per in m["vs_canonical_sibling"].items():
                for a, s in per.items():
                    print(f"  variant {t}: {v}/{a}/{c[:36]} sibling -> " + (f"ABSENT: {s['reason'][:150]}" if s.get("status") == "ABSENT" else
                          f"paired delta {(s.get('paired') or s.get('paired_on_intersection') or {}).get('delta_pp'):+.2f} pp, McNemar p={(s.get('paired') or s.get('paired_on_intersection') or {}).get('mcnemar_exact_p'):.3g}"))
    for r in LANDING_ROOTS + MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS:
        l = out["meta"]["landing_status"][r]
        print(f"  landing {r} [{l['namespace']}{'/' + l['variant'] if l['variant'] else ''}]: "
              + (f"graded {l['attacks_graded']} on {l['served_cluster']}; absent {l['attacks_absent']}" if l["present"] else "ABSENT (not created yet; recorded, nothing emitted)"))
    print(f"  served_cluster: {len(out['meta']['served_cluster_default']['roots_defaulted'])} pre-existing root(s) defaulted to {SERVED_CLUSTER_DEFAULT} (meta only, records unchanged); "
          f"{len(out['meta']['served_cluster_default']['roots_derived'])} root(s) derived from a record (rows carry it)")
    for r in ANCHOR_ROOTS:
        a = out["meta"]["anchor_status"][r]
        print(f"  anchor {r} ({a['view']}): graded {a['attacks_graded']}; absent IN THIS ROOT (no aggregate here; this builder does not know the leg's state; no row from this root) {a['attacks_absent']}")
    _relu = [r for r in count_rows if r["namespace"] == NS_RELU]
    print(f"  relu_native: {len(_relu)} (view, attack, cell) readings from {len(RELU_NATIVE_ROOTS)} root(s); every one paired against its declared anchor under "
          f"{AP.PASSTHROUGH_ANCHOR_ALLOWANCE}:")
    for r in _relu:
        print(f"    {r['view']:16s} {r['attack']:12s} {r['cell'][:44]:44s} vs {r['anchor_root']}: {('%+.1f pp' % r['vs_bare_delta_pp']) if r['vs_bare_delta_pp'] is not None else 'no paired part'}")
    _allpos = [r for r in count_rows if r["namespace"] == NS_ALLPOS_NATIVE]
    print(f"  {NS_ALLPOS_NATIVE}: {len(_allpos)} (view, attack, cell) readings from {sum(1 for r in ALLPOS_NATIVE_ROOTS if r in roots_meta)} of "
          f"{len(ALLPOS_NATIVE_ROOTS)} registered mt4kap root(s) ({[r for r in ALLPOS_NATIVE_ROOTS if r in roots_meta]}), read under "
          f"NAMESPACE_SERVING[{NS_ALLPOS_NATIVE!r}]: native engine, single-root windows; no vs_bare filed")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
