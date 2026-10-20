#!/usr/bin/env python3
"""Apply the mt_arms4096 extension to build_transfer_stats_cap4096.py by exact-match replacement (each old string must occur exactly once)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from site_paths import site_path  # noqa: E402
# the run-state copy of the lane builder, under the site's state root ($STATE); an unset one ends the run by name
P = site_path("$STATE/romp/candidate-si-refit/run-state/full22/cap4096_export/build_transfer_stats_cap4096.py", "P")
src = P.read_text()
REPL = []

# ---------------------------------------------------------------- R1 docstring
REPL.append(("""usage: build_transfer_stats_cap4096.py [--out <json>] [--rs <run-state/full22>]""",
"""THE ARMS WAVE LANDS HERE TOO (mt_arms4096, 2026-09-14). The constitution-SI cell ``prompted__si_constitution`` (PROMPTED: no object) on
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

usage: build_transfer_stats_cap4096.py [--out <json>] [--rs <run-state/full22>]"""))

# ---------------------------------------------------------------- R2 arms constants after LANDING_ROOTS
REPL.append(("""LANDING_ROOTS = ["grades_qc4kctl_qwen35_4b_it", "grades_qc4kdiff_qwen35_4b_it",
                 # the gemma4e4b/it half (same lane, after PR #107 + the view rebuild)
                 "grades_qc4kgemctl_gemma4e4b_it", "grades_qc4kgemdiff_gemma4e4b_it"]
""",
"""LANDING_ROOTS = ["grades_qc4kctl_qwen35_4b_it", "grades_qc4kdiff_qwen35_4b_it",
                 # the gemma4e4b/it half (same lane, after PR #107 + the view rebuild)
                 "grades_qc4kgemctl_gemma4e4b_it", "grades_qc4kgemdiff_gemma4e4b_it"]
#: the multi-turn ARMS wave (mt_arms4096; rulings 2026-09-14 00:40Z; README "Export producer" + cluster_env.sh e4n_root_tag): the constitution-SI
#: cell `prompted__si_constitution` (PROMPTED, no object) on the paper views gemma4e4b_it / qwen35_4b_it / nemotron3nano4b and on llama8b_shortconst
#: (the only llama view whose roster serves it), at the arms' canonical instrument -- crescendo + actorattack at 49,152, PAIR at 16,384 on gemma/qwen.
#: Root grammar:  grades_mt4k<arm>[_w24k]_<clu>_<TAG>   arm in {sc = the SI cell, nn = the bare `none` same-window PAIR anchor};
#:                clu in {bu = cluster_a, ch = cluster_b}; TAG = <key> for view main, <key>_<view> otherwise (TAG == the target directory == the view).
#: The SAME (lane, attack) may land from BOTH clusters (the 14 cluster B legs of 00:31Z + their cluster A twins; a race cancels one, "both running"
#: is kept) -- TWIN_RULE below. Every root here is a LANDING root: absent until its chain grades it, tolerated and recorded in meta.landing_status.
#: NOT registered (blocked on AlphaKL's port; README): the arditi `mt4kar_*` and alphasteer `mt4kas_*` arms, and any ROOT_SUFFIX restart namespace
#: (`mt4ksc_w24kr2_...`) -- register them here deliberately when they exist.
SI_CONST = "prompted__si_constitution"
MT_ARMS_TAGS = ["gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_shortconst"]
MT_ARMS_W24K_TAGS = ["nemotron3nano4b", "llama8b_shortconst"]          # the two lanes whose PAIR ran at 24,576 (with an nn anchor each)
CLUSTER_TOKEN = {"bu": "cluster_a", "ch": "cluster_b"}                # the root-name token -> the cluster (cluster_env.sh e4n_clu)
MT_ARMS_CELL = {"sc": SI_CONST, "nn": "none"}
MT_ARMS_ROOTS = [f"grades_mt4ksc_{clu}_{tag}" for tag in MT_ARMS_TAGS for clu in CLUSTER_TOKEN]
MT_ARMS_W24K_ROOTS = [f"grades_mt4k{arm}_w24k_{clu}_{tag}" for tag in MT_ARMS_W24K_TAGS for arm in ("sc", "nn") for clu in CLUSTER_TOKEN]
LANDING_NOTE = {**{r: "root not created yet: no leg of this lane has been graded (qwen legs 498351-498356; gemma legs after PR #107; R2-D288); lands on a later run"
                   for r in LANDING_ROOTS},
                **{r: ("root not created yet: no leg of this (lane, attack, cluster) has been graded (mt_arms4096: cluster B legs submitted 2026-09-14 00:31Z, "
                       "their cluster A twins queued behind the 16-slot cap; the first cluster to serve a leg wins and the other twin is cancelled by name, so this "
                       "root may land on a later run or never)") for r in MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS}}
SERVED_CLUSTER_DEFAULT = "cluster_a"
SERVED_CLUSTER_DEFAULT_NOTE = ("no served_cluster.json beside this root's transcripts and no cluster token in its name: it predates the ruling of 2026-09-13 16:45 "
                               "and was pulled by its lane's pull script from HOST node-27 (cluster A) -- every cap4096 / obj4096 / anchor4096 / qc4k root. "
                               "The default is recorded in meta.roots.<root>.served_cluster and on its meta.count_table rows; it is NOT added to the filed cell "
                               "records, which stay byte-identical to the export that already read them")
TWIN_RULE = ("two CANONICAL readings of one (view, cell, attack, serving window) from DIFFERENT clusters are twins, not a collision: the reading whose "
             "aggregate the grader wrote FIRST (two_si_<attack>_aggregate.json written_at; tie -> cluster_a) holds the view's cell slot "
             "(models.<view>.cells.<cell>.<attack>), the other is filed under models_twins.<view>.cells.<cell>.<attack> with the identical record shape; "
             "both carry served_cluster; meta.twin_status lists every such key with both roots. Two readings of one key from the SAME cluster remain a "
             "defect and refuse. The same rule places variant twins under models_variants.<tag>.models_twins")
""" ))

# ---------------------------------------------------------------- R3 CANONICAL_ROOTS += arms
REPL.append(("""    # the qwen35_4b/it control + difference legs (see LANDING_ROOTS above; absent until their chains grade)
    *LANDING_ROOTS,
]""",
"""    # the qwen35_4b/it control + difference legs (see LANDING_ROOTS above; absent until their chains grade)
    *LANDING_ROOTS,
    # the constitution-SI arm at the arms' own windows, from either cluster (see MT_ARMS_ROOTS above; absent until their chains grade)
    *MT_ARMS_ROOTS,
]"""))

# ---------------------------------------------------------------- R4 w24k window variants after VARIANT_SIBLING_KEY_FIELDS
REPL.append(("""VARIANT_SIBLING_KEY_FIELDS = ("view", "attack", "cell", "decode_cap_tokens", "namespace")
""",
"""VARIANT_SIBLING_KEY_FIELDS = ("view", "attack", "cell", "decode_cap_tokens", "namespace")
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
"""))

# ---------------------------------------------------------------- R5 views
REPL.append(("""VIEW_KIND = {"gemma4e4b": "main", "qwen35_4b": "main", "nemotron3nano4b": "main", "glm46vflash": "main", "gemma4e4b_it": "it", "qwen35_4b_it": "it"}
VIEW_LABEL = {"gemma4e4b": "google/gemma-4-E4B-it — main view, multi-turn at a 4,096-token reply cap",
              "qwen35_4b": "Qwen/Qwen3.5-4B (instruct) — main view, multi-turn at a 4,096-token reply cap",
              "nemotron3nano4b": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 — main view, multi-turn at a 4,096-token reply cap",
              "glm46vflash": "zai-org/GLM-4.6V-Flash — main view, multi-turn at a 4,096-token reply cap (rows are LOWER BOUNDS: no 8k rung, R2-D256a)",
              "gemma4e4b_it": "google/gemma-4-E4B-it — it view, multi-turn at a 4,096-token reply cap",
              "qwen35_4b_it": "Qwen/Qwen3.5-4B (instruct) — it view, multi-turn at a 4,096-token reply cap"}
PROMPTED_LABEL = "PROMPTED safety SI (safety_refusal as the system prompt; no object)"
""",
"""#: the llama views carry the llama builder's spellings and kinds (llama8b_export/build_transfer_stats_full22_llama8b.py ROOTS): shortconst / fullconst
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
    \"\"\"The served key behind a view: gemma4e4b_it -> gemma4e4b, llama8b_shortconst -> llama8b (a transfer view, its instruct view and a prompted-only view
    serve the SAME checkpoint; the objects differ in where they were fit, not in what serves them).\"\"\"
    for sfx in VIEW_SUFFIXES:
        if view.endswith(sfx):
            return view[:-len(sfx)]
    return view
"""))

# ---------------------------------------------------------------- R6 label_group
REPL.append(("""def label_group(view: str, cell: str) -> tuple[str, str]:
    if cell == PROMPTED:
        return PROMPTED_LABEL, "prompted_baseline"
    if VIEW_KIND[view] == "it":
        return B.IT_LABEL.get(cell, f"{cell} (extra)"), B.IT_GROUP.get(cell, "extra")
    return B.CELL_LABEL.get(cell, f"{cell} (extra)"), B.CELL_GROUP.get(cell, "extra")
""",
"""def label_group(view: str, cell: str) -> tuple[str, str]:
    if cell == PROMPTED:
        return PROMPTED_LABEL, "prompted_baseline"
    if cell == SI_CONST:
        return PROMPTED_CONST_LABEL, "prompted_baseline"
    if VIEW_KIND[view] == "it":
        return B.IT_LABEL.get(cell, f"{cell} (extra)"), B.IT_GROUP.get(cell, "extra")
    if VIEW_KIND[view] == "prompted":
        # the prompted-only lanes' bare cell is the same served checkpoint as the main view's, in its own serving job (the llama builder's wording)
        return (("none (bare instruct, same wave, prompted-only lane's serving job)", "baseline_full22") if cell == "none" else (f"{cell} (extra)", "extra"))
    return B.CELL_LABEL.get(cell, f"{cell} (extra)"), B.CELL_GROUP.get(cell, "extra")


def served_cluster_of(root_name: str, troot: Path) -> tuple[str, dict]:
    \"\"\"(cluster, provenance) for a root: transcripts_<root>/served_cluster.json (written by mt_arms4096/pull_mtarms.sh from the standup's shared
    root) when present -- cross-checked against the root name's cluster token; else the token alone (_bu_ / _ch_); else the pre-existing roots'
    default (SERVED_CLUSTER_DEFAULT_NOTE). `on_rows` says whether the cluster was DERIVED from a record (then every cell record carries it) or
    defaulted (then it rides meta only, so the filed records stay byte-identical).\"\"\"
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
"""))

# ---------------------------------------------------------------- R7 instrument(): declared-decode fallback
REPL.append(("""    min_clamped = None
    for st in sorted(cell_dir.glob("shard*/served_target.json")):
        s = json.loads(st.read_text())
        wc = s.get("window_clamp") or {}
        ld = s.get("last_request_decode") or {}
        key = json.dumps({"serving_window_tokens": s.get("max_model_len"),""",
"""    min_clamped = None
    n_declared_fallback = 0
    for st in sorted(cell_dir.glob("shard*/served_target.json")):
        s = json.loads(st.read_text())
        wc = s.get("window_clamp") or {}
        ld = s.get("last_request_decode") or {}
        if not ld and s.get("decode"):
            # the llama lineage (390b230a) records the DECLARED decode block on every shard and last_request_decode only where the shard's last call
            # recorded one (null on 370 shards of the llama lane's stores); the arms' 0088d863 lineage records last_request_decode only. Where the
            # last-request record is absent the declared block carries the same tuple (do_sample / T / top_p; no top_k, no penalty, no note), so it
            # stands in -- counted per shard below, and never preferred over a recorded last request. No shard of a pre-existing root takes this path.
            ld = s["decode"]; n_declared_fallback += 1
        key = json.dumps({"serving_window_tokens": s.get("max_model_len"),"""))
REPL.append(("""    inst["min_clamped_completion_floor"] = min_clamped
    if inst["n_prompt_over_window"] or inst["n_window_mismatch"]:""",
"""    inst["min_clamped_completion_floor"] = min_clamped
    if n_declared_fallback:
        inst["n_shards_decode_from_declared_block"] = n_declared_fallback     # only where it happened (no pre-existing record changes)
    if inst["n_prompt_over_window"] or inst["n_window_mismatch"]:"""))

# ---------------------------------------------------------------- R8 NO_CAP512 + cap512_side
REPL.append(("""             ("gemma4e4b_it", MINUS_CTL): "batch-64 LoRA subtraction (derived_itdoc64__…_minus_ctl): staged 2026-09-13, first measured at 4,096; no 512 pass"}
""",
"""             ("gemma4e4b_it", MINUS_CTL): "batch-64 LoRA subtraction (derived_itdoc64__…_minus_ctl): staged 2026-09-13, first measured at 4,096; no 512 pass",
             # the constitution-SI arm (mt_arms4096, 2026-09-14): first measured multi-turn at 4,096; no 512 multi-turn reading on ANY view
             **{(v, SI_CONST): ("prompted constitution SI (mt_arms4096 arm sc, config-only overlay 1d17331e / 390b230a): first measured multi-turn at 4,096 on "
                                "2026-09-14; no 512 multi-turn pass ever ran this cell on any view") for v in MT_ARMS_TAGS}}
#: whole VIEWS with no filed 512 reading for ANY cell: the Llama-3.1-8B lane has no 512-cap instrument at all -- its BoN and multi-turn were generated
#: at 4,096 outright (grades_llama8b_* roots: grading_provenance.decode_caps every cap 4096, every cell stamped decode_cap_tokens 4096; the lane file
#: transfer_stats_full22_llama8b.json carries no 512 row). Declared, never inferred: a llama (view, cell) never reaches CAP512_SOURCE.
NO_CAP512_VIEWS = {v: ("the Llama-3.1-8B lane has NO 512-cap reading of any cell: its BoN and multi-turn were generated at 4,096 outright (grades_llama8b_* "
                       "roots, decode_caps all 4096; build_export.py decode_cap_tokens column doc), so no 512 pass exists to pair against") for v in LLAMA_VIEWS}
"""))
REPL.append(("""    if (view, cell) in NO_CAP512:
        return None, None, {"status": "NO_512_READING", "reason": NO_CAP512[(view, cell)], "decode_cap_tokens": CAP_FILED}
""",
"""    if (view, cell) in NO_CAP512:
        return None, None, {"status": "NO_512_READING", "reason": NO_CAP512[(view, cell)], "decode_cap_tokens": CAP_FILED}
    if view in NO_CAP512_VIEWS:
        return None, None, {"status": "NO_512_READING", "reason": NO_CAP512_VIEWS[view], "decode_cap_tokens": CAP_FILED, "declared_for": "view"}
"""))

# ---------------------------------------------------------------- R9 main(): state
REPL.append(("""    models: dict = {}            # view -> model block (canonical)
    variants: dict = {}          # tag -> {"variant": ..., "models": {view -> model block}}
    seen: dict = {}              # (view, cell, attack) -> root, canonical only: a second reading of one key is a defect, not a merge
    roots_meta: dict = {}
    count_rows: list = []
""",
"""    models: dict = {}            # view -> model block (canonical)
    models_twins: dict = {}      # view -> model block holding the OTHER cluster's reading of a key already filed in models (TWIN_RULE)
    variants: dict = {}          # tag -> {"variant": ..., "models": {view -> model block}} (+ "models_twins" once a variant twin is filed)
    seen: dict = {}              # (view, cell, attack, window, served_cluster) -> root, canonical only: a second reading of one key FROM THE SAME
                                 # CLUSTER is a defect, not a merge; the other cluster's reading is a twin, placed by place_reading (TWIN_RULE)
    primary: dict = {}           # (namespace or variant tag, view, cell, attack, window) -> the reading holding the cell slot
    twin_status: list = []
    roots_meta: dict = {}
    count_rows: list = []
    landing_present: dict = {}   # landing root -> present on disk this run
"""))

# ---------------------------------------------------------------- R9b transcripts candidates
REPL.append(("""        cands = [rs / f"transcripts_{tag}"] + ([rs / f"transcripts_{tag[:-3]}"] if tag.endswith("_it") else [])
        cands = [c for c in cands if c.is_dir() and only_dir(c).name == f"sdfbase_full22_{view}"]
        if len(cands) != 1:
            raise SystemExit(f"{root}: transcripts root for view {view!r} not resolved (candidates holding run sdfbase_full22_{view}: {cands})")
        troot = cands[0]
        run_dir = only_dir(troot)
""",
"""        cands = [rs / f"transcripts_{tag}"] + [rs / f"transcripts_{tag[:-len(sfx)]}" for sfx in VIEW_SUFFIXES if tag.endswith(sfx)]
        cands = [c for c in cands if c.is_dir() and only_dir(c).name == f"sdfbase_full22_{view}"]
        if len(cands) != 1:
            raise SystemExit(f"{root}: transcripts root for view {view!r} not resolved (candidates holding run sdfbase_full22_{view}: {cands})")
        troot = cands[0]
        run_dir = only_dir(troot)
        served_cluster, sc_prov = served_cluster_of(root_name, troot)     # WHICH CLUSTER served this root (ruling 2026-09-13 16:45)
"""))

# ---------------------------------------------------------------- R9c model block creation
REPL.append(("""        target_block = variants[variant["tag"]]["models"] if variant else models
        mb = target_block.setdefault(view, {
            "label": VIEW_LABEL[view] + (f" — VARIANT {variant['tag']}: {variant['reason']}" if variant else ""), "view": VIEW_KIND[view],
            "base_target": view.replace("_it", ""), "namespace": ns, "decode_cap_tokens": CAP, "served_alpha": None,
            "lens_reduction_mode": WAVE_LENS_MODE, "attacks_present": [], "expected_cells": [], "cells": {}, "vs_bare": {},
            "vs_bare_note": (f"NOT COMPUTED by this builder: the unsteered `none` anchor at {CAP} (grades_anchor4096_*, 2026-09-13) is exported as "
                             "its own cell of this view wherever its aggregate exists (meta.anchor_status); wiring the same-cap cell-vs-bare contrast "
                             "is a follow-up. The same view's 512 unsteered row remains the floor the rung collapse reads. Nothing is fabricated here."),
            "vs_cap512": {}, "vs_canonical_sibling": {} if variant else None, "sources": {}, "judge": {"strongreject": {"per_root": {}}},
            "completion": {}})
        mb["served_alpha"] = next(iter(aggs.values())).get("served_alpha")
""",
"""        def new_block(twin: bool) -> dict:
            blk = {
                "label": VIEW_LABEL[view] + (f" — VARIANT {variant['tag']}: {variant['reason']}" if variant else "")
                         + (" — TWIN READINGS: the other cluster's grading of keys already filed in the primary block (meta.twin_rule)" if twin else ""),
                "view": VIEW_KIND[view],
                "base_target": base_target(view), "namespace": ns, "decode_cap_tokens": CAP, "served_alpha": None,
                "lens_reduction_mode": WAVE_LENS_MODE, "attacks_present": [], "expected_cells": [], "cells": {}, "vs_bare": {},
                "vs_bare_note": (f"NOT COMPUTED by this builder: the unsteered `none` anchor at {CAP} (grades_anchor4096_*, 2026-09-13) is exported as "
                                 "its own cell of this view wherever its aggregate exists (meta.anchor_status); wiring the same-cap cell-vs-bare contrast "
                                 "is a follow-up. The same view's 512 unsteered row remains the floor the rung collapse reads. Nothing is fabricated here."),
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
"""))

# ---------------------------------------------------------------- R9d rmeta gets served_cluster
REPL.append(("""                 "attacker_identity_check": gp.get("attacker_identity_check"), "namespace": ns, "variant": (variant["tag"] if variant else None)}
        roots_meta[root_name] = rmeta
        mb["judge"]["strongreject"]["per_root"][root_name] = rmeta["judge"]
""",
"""                 "attacker_identity_check": gp.get("attacker_identity_check"), "namespace": ns, "variant": (variant["tag"] if variant else None),
                 "served_cluster": served_cluster, "served_cluster_provenance": sc_prov, "serving_windows": {}}
        roots_meta[root_name] = rmeta
        mb["judge"]["strongreject"]["per_root"][root_name] = rmeta["judge"]

        def install(blk: dict, it: dict) -> None:
            \"\"\"file one reading into a model block, in the filed key order (cells -> completion -> sources -> vs_cap512 -> vs_canonical_sibling)\"\"\"
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
            if variant:
                blk["vs_canonical_sibling"].setdefault(cell, {})[attack] = it["sib"]

        def twin_order(it: dict):
            return (it["written_at"] or "9999", 0 if it["served_cluster"] == SERVED_CLUSTER_DEFAULT else 1)

        def place_reading(it: dict) -> None:
            \"\"\"TWIN_RULE: the first reading of a (view, cell, attack, window) takes the cell slot; a second one from another cluster is a twin --
            the earlier-written aggregate holds the slot whichever root was processed first, the other goes to the twins block.\"\"\"
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
"""))

# ---------------------------------------------------------------- R9e per-cell loop
OLD_CELL_LOOP = """            if attack not in mb["attacks_present"]:
                mb["attacks_present"].append(attack)
            for cell in sorted(agg["cells"]):
                key = (view, cell, attack)
                if not variant:
                    if key in seen:
                        raise SystemExit(f"{key} graded in BOTH {seen[key]} and {root_name} — two canonical readings of one key; refusing")
                    seen[key] = root_name
                cell_dir = run_dir / attack / cell
                rec, outcomes = read_cell(agg, cell, attack, ns, cell_dir)
                if cell not in mb["cells"]:
                    lab, grp = label_group(view, cell)
                    mb["cells"][cell] = {"label": lab, "group": grp, "view": VIEW_KIND[view], "namespace": ns, "decode_cap_tokens": CAP,
                                         "lens_reduction_mode": WAVE_LENS_MODE, "on_overlay": cell in B.OVERLAY_CELLS, "completion": {}}
                    mb["expected_cells"].append(cell)
                mb["cells"][cell][attack] = rec
                mb["cells"][cell]["completion"][attack] = rec["completion"]
                mb["sources"].setdefault(cell, {})[attack] = root_name
                # the cell's own 4,096 - 512 paired change (truncation bias): the 512 side from the root the export's join read, cross-checked
                rec512, o512, src512 = cap512_side(view, cell, attack, joins)
                if rec512 is None:      # declared NO_CAP512: the pairing is undefined for this cell, and the record says so
                    vc = {"status": "NO_512_READING", "reason": src512["reason"], "cap512_source": src512,
                          "reads_as": f"no filed {CAP_FILED}-cap reading exists for this cell (first measured at {CAP}); the truncation-bias pairing is UNDEFINED here, not zero"}
                else:
                    vc = contrast_with_intersection(rec512, rec, o512, outcomes, "512 side dropped errored goals")
                    vc["reads_as"] = f"{CAP}-cap reading MINUS the filed {CAP_FILED}-cap reading of the SAME cell (cell = {CAP}, bare = {CAP_FILED}); truncation bias, not a defence effect"
                    vc["cap512_source"] = src512
                mb["vs_cap512"].setdefault(cell, {})[attack] = vc
                # the variant against its canonical sibling at the same cap (variant MINUS canonical), paired on goals
                sib = None
                if variant:
                    sroot = rs / variant["sibling_root"]
                    sdir = only_dir(sroot)
                    sagg = load_json(sdir / f"two_si_{attack}_aggregate.json")
                    skey = {"view": view, "attack": attack, "cell": cell, "decode_cap_tokens": CAP, "namespace": NS,
                            "sibling_root": str(sroot), "status_in_export": "graded"}
                    if sagg and cell in sagg.get("cells", {}) and sdir.name == view:
                        srec = R2._safe(T1.gated_cell, sagg, cell)
                        sib = contrast_with_intersection(srec, rec, R2.attack_outcomes(sagg, cell), outcomes, "sibling")
                        sib["reads_as"] = f"variant ({variant['reason']}) MINUS the canonical sibling at cap {CAP} (cell = variant, bare = canonical)"
                        sib["sibling_key"] = skey
                        sib["term_moved"] = variant["term_moved"]
                    else:
                        sib = {"status": "ABSENT", "reason": variant.get("sibling_note", "no canonical sibling aggregate"), "sibling_key": skey,
                               "term_moved": variant["term_moved"]}
                    mb["vs_canonical_sibling"].setdefault(cell, {})[attack] = sib
                count_rows.append({"view": view, "attack": attack, "cell": cell, "namespace": ns, "variant": (variant["tag"] if variant else None),
                                   "root": root_name,
                                   "cap512": ({"k": rec512["k"], "n": rec512["n"], "n_responses": src512["n_responses"], "source": src512["grades_dir"]}
                                              if rec512 is not None else None),
                                   f"cap{CAP}": {"k": rec["k"], "n": rec["n"], "n_responses": rec["completion"].get("n_responses"),
                                                 "at_cap_k": rec["at_cap_finish_length"]["k"], "max_completion_tokens": rec["decode_cap_verification"]["max_completion_tokens"],
                                                 "grader_stamp": rec["decode_cap_tokens_grader_stamp"], "decode_mode": rec["instrument"]["decode_mode"],
                                                 "serving_window": rec["instrument"]["serving_window_tokens"],
                                                 "n_window_clamped": rec["instrument"]["n_window_clamped"],
                                                 "n_clamped_censored": rec["decode_cap_verification"]["n_clamped_calls_censored"]},
                                   "vs_cap512_delta_pp": ((vc.get("paired") or vc.get("paired_on_intersection") or {}).get("delta_pp")),
                                   "vs_sibling_delta_pp": (((sib or {}).get("paired") or (sib or {}).get("paired_on_intersection") or {}).get("delta_pp") if sib else None)})
        mb["completion"] = {c: mb["cells"][c]["completion"] for c in mb["cells"]}
"""
NEW_CELL_LOOP = """            for cell in sorted(agg["cells"]):
                cell_dir = run_dir / attack / cell
                rec, outcomes = read_cell(agg, cell, attack, ns, cell_dir)
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
                             "vs_sibling_delta_pp": (((sib or {}).get("paired") or (sib or {}).get("paired_on_intersection") or {}).get("delta_pp") if sib else None)}
                count_rows.append(count_row)
                place_reading({"view": view, "cell": cell, "attack": attack, "window": window, "root": root_name, "served_cluster": served_cluster,
                               "written_at": agg.get("written_at"), "rec": rec, "vc": vc, "sib": sib, "count_row": count_row})
        mb["completion"] = {c: mb["cells"][c]["completion"] for c in mb["cells"]}
        tb = (variants[variant["tag"]].get("models_twins", {}) if variant else models_twins).get(view)
        if tb is not None:
            tb["completion"] = {c: tb["cells"][c]["completion"] for c in tb["cells"]}
"""
REPL.append((OLD_CELL_LOOP, NEW_CELL_LOOP))

# ---------------------------------------------------------------- R9f the root loops
REPL.append(("""    landing_absent: list = []
    for r in CANONICAL_ROOTS:
        if r in LANDING_ROOTS and not (rs / r).is_dir():
            landing_absent.append(r); continue
        process_root(r, NS, None)
    for r, v in VARIANT_ROOTS.items():
        variants.setdefault(v["tag"], {"variant": {**v, "root": str(rs / r), "sibling_key_fields": list(VARIANT_SIBLING_KEY_FIELDS),
                                                   "namespace": NS_VARIANT, "status_in_export": "variant"}, "models": {}})
        process_root(r, NS_VARIANT, v)
    for mb in list(models.values()) + [m for v in variants.values() for m in v["models"].values()]:
        mb["attacks_present"] = [a for a in MULTITURN if a in mb["attacks_present"]]
        mb["attacks_pending"] = []
        mb["judge"]["strongreject"].update({"model": PINNED_JUDGE, "rubric": PINNED_RUBRIC, "cache_kind": "sr"})
""",
"""    all_landing = set(LANDING_ROOTS) | set(MT_ARMS_ROOTS) | {r for r, v in VARIANT_ROOTS.items() if v.get("landing")}
    for r in CANONICAL_ROOTS:
        if r in all_landing:
            landing_present[r] = (rs / r).is_dir()
            if not landing_present[r]:
                continue                                   # absent landing root: recorded in meta.landing_status, nothing raised
        process_root(r, NS, None)
    for r, v in VARIANT_ROOTS.items():
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
        if v.get("landing"):
            landing_present[r] = (rs / r).is_dir()
            if not landing_present[r]:
                continue
        process_root(r, NS_VARIANT, v)
    for mb in (list(models.values()) + list(models_twins.values())
               + [m for v in variants.values() for blk in ("models", "models_twins") for m in v.get(blk, {}).values()]):
        mb["attacks_present"] = [a for a in MULTITURN if a in mb["attacks_present"]]
        mb["attacks_pending"] = []
        mb["judge"]["strongreject"].update({"model": PINNED_JUDGE, "rubric": PINNED_RUBRIC, "cache_kind": "sr"})

    def landing_entry(r: str) -> dict:
        v = VARIANT_ROOTS.get(r)
        base = {"namespace": (NS_VARIANT if v else NS), "variant": (v["tag"] if v else None),
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
"""))

# ---------------------------------------------------------------- R9g the out dict
REPL.append(("""                         "roots": list(VARIANT_ROOTS), "variants": {v["tag"]: {k: v[k] for k in ("term_moved", "reason", "sibling_root", "what_is_held_fixed")}
                                                                   for v in VARIANT_ROOTS.values()},
                         "sibling_key_fields": list(VARIANT_SIBLING_KEY_FIELDS),""",
"""                         "roots": list(VARIANT_ROOTS), "variants": {v["tag"]: variant_meta(v) for v in VARIANT_ROOTS.values()},
                         "sibling_key_fields": list(VARIANT_SIBLING_KEY_FIELDS),
                         "sibling_key_fields_window_variants": list(W24K_SIBLING_KEY_FIELDS),"""))
REPL.append(("""        "cell_labels": {"main": B.CELL_LABEL, "it": B.IT_LABEL, "prompted": PROMPTED_LABEL}, "cell_groups": {"main": B.CELL_GROUP, "it": B.IT_GROUP},""",
"""        "cell_labels": {"main": B.CELL_LABEL, "it": B.IT_LABEL, "prompted": PROMPTED_LABEL, "prompted_constitution": PROMPTED_CONST_LABEL,
                        "prompted_only_view_none": "none (bare instruct, same wave, prompted-only lane's serving job)"},
        "cell_groups": {"main": B.CELL_GROUP, "it": B.IT_GROUP, SI_CONST: "prompted_baseline"},"""))
REPL.append(("""        "landing_status": {r: ({"present": True, "view": roots_meta[r]["view"], "attacks_graded": roots_meta[r]["attacks"],
                                "attacks_absent": [a for a in MULTITURN if a not in roots_meta[r]["attacks"]],
                                "note": "partial by design while the lane's legs land; an absent attack has no aggregate yet and gets no row"}
                               if r in roots_meta else
                               {"present": False, "note": "root not created yet: no leg of this lane has been graded (qwen legs 498351-498356; gemma legs after PR #107; R2-D288); lands on a later run"})
                           for r in LANDING_ROOTS},
        "no_cap512_cells": {f"{v}|{c}": why for (v, c), why in NO_CAP512.items()},""",
"""        "landing_status": {r: landing_entry(r) for r in LANDING_ROOTS + MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS},
        "arms_wave": {"readme": str(rs / "mt_arms4096/README.md"), "cluster_env": str(rs / "mt_arms4096/cluster_env.sh"),
                      "root_grammar": "grades_mt4k<arm>[_w24k]_<clu>_<TAG>; arm sc = prompted__si_constitution, nn = none (same-window PAIR anchor); clu bu = cluster_a, ch = cluster_b; TAG = <key> for view main else <key>_<view>",
                      "cell": SI_CONST, "tags": MT_ARMS_TAGS, "canonical_landing_roots": MT_ARMS_ROOTS, "window_variant_landing_roots": MT_ARMS_W24K_ROOTS,
                      "present_this_run": sorted(r for r in MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS if landing_present.get(r)),
                      "absent_this_run": sorted(r for r in MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS if not landing_present.get(r)),
                      "not_registered": "arditi (mt4kar_*) and alphasteer (mt4kas_*) arms -- blocked on AlphaKL's port; ROOT_SUFFIX restart namespaces (e.g. mt4ksc_w24kr2_*) -- register deliberately when they exist"},
        "twin_rule": TWIN_RULE,
        "twin_status": twin_status,
        "served_cluster_default": {"value": SERVED_CLUSTER_DEFAULT, "note": SERVED_CLUSTER_DEFAULT_NOTE,
                                   "roots_defaulted": sorted(r for r, m in roots_meta.items() if m["served_cluster_provenance"]["source"] == "default"),
                                   "roots_derived": {r: m["served_cluster_provenance"]["source"] for r, m in roots_meta.items() if m["served_cluster_provenance"]["source"] != "default"},
                                   "on_rows_rule": "a cell record carries served_cluster iff its root's cluster was DERIVED (served_cluster.json, else the _bu_/_ch_ name token); defaulted roots' records are unchanged and their cluster rides meta.roots.<root>.served_cluster + meta.count_table[*].served_cluster"},
        "served_cluster_by_root": {r: m["served_cluster"] for r, m in roots_meta.items()},
        "no_cap512_cells": {f"{v}|{c}": why for (v, c), why in NO_CAP512.items()},
        "no_cap512_views": dict(NO_CAP512_VIEWS),"""))
REPL.append(("""        "models": models, "models_variants": variants, "cross_view": {},""",
"""        "models": models, "models_twins": models_twins, "models_variants": variants, "cross_view": {},"""))

# ---------------------------------------------------------------- R9h printing
REPL.append(("""    print(f"\\n{'view':16s} {'attack':12s} {'cell':40s} {'ns':16s} | {'512 k/n':9s} {'512 resp':8s} | {'4096 k/n':9s} {'4096 resp':9s} {'at-cap':7s} {'max_ct':6s} {'stamp':5s} {'decode':16s} {'window':6s} {'clamp':5s} | {'d(4096-512)pp':13s} {'d(var-canon)pp':14s}")
    for r in count_rows:
        a, b = r["cap512"] or {"k": "—", "n": "—", "n_responses": "—"}, r[f"cap{CAP}"]
        d1 = r["vs_cap512_delta_pp"]; d2 = r["vs_sibling_delta_pp"]
        print(f"{r['view']:16s} {r['attack']:12s} {r['cell'][:40]:40s} {r['namespace']:16s} | {str(a['k'])+'/'+str(a['n']):9s} {str(a['n_responses']):8s} | "
              f"{str(b['k'])+'/'+str(b['n']):9s} {str(b['n_responses']):9s} {str(b['at_cap_k']):7s} {str(b['max_completion_tokens']):6s} {str(b['grader_stamp']):5s} "
              f"{b['decode_mode']:16s} {str(b['serving_window']):6s} {str(b['n_window_clamped']) + '/' + str(b['n_clamped_censored']):5s} | {('%+.1f' % d1) if d1 is not None else '—':13s} {('%+.1f' % d2) if d2 is not None else ('—' if r['variant'] else ''):14s}")
    n_can = sum(1 for r in count_rows if not r["variant"]); n_var = len(count_rows) - n_can
    print(f"\\n{n_can} canonical (view, attack, cell) readings at {CAP} from {len(CANONICAL_ROOTS)} roots; {n_var} variant readings from {len(VARIANT_ROOTS)} roots; "
          f"every 512 side cross-checked against the export's join files; every {CAP} store identity-checked against the grader's completion stamp.")""",
"""    print(f"\\n{'view':18s} {'attack':12s} {'cell':40s} {'ns':16s} {'clu':10s} {'slot':7s} | {'512 k/n':9s} {'512 resp':8s} | {'4096 k/n':9s} {'4096 resp':9s} {'at-cap':7s} {'max_ct':6s} {'stamp':5s} {'decode':16s} {'window':6s} {'clamp':5s} | {'d(4096-512)pp':13s} {'d(var-canon)pp':14s}")
    for r in count_rows:
        a, b = r["cap512"] or {"k": "—", "n": "—", "n_responses": "—"}, r[f"cap{CAP}"]
        d1 = r["vs_cap512_delta_pp"]; d2 = r["vs_sibling_delta_pp"]
        print(f"{r['view']:18s} {r['attack']:12s} {r['cell'][:40]:40s} {r['namespace']:16s} {r['served_cluster']:10s} {r['slot']:7s} | {str(a['k'])+'/'+str(a['n']):9s} {str(a['n_responses']):8s} | "
              f"{str(b['k'])+'/'+str(b['n']):9s} {str(b['n_responses']):9s} {str(b['at_cap_k']):7s} {str(b['max_completion_tokens']):6s} {str(b['grader_stamp']):5s} "
              f"{b['decode_mode']:16s} {str(b['serving_window']):6s} {str(b['n_window_clamped']) + '/' + str(b['n_clamped_censored']):5s} | {('%+.1f' % d1) if d1 is not None else '—':13s} {('%+.1f' % d2) if d2 is not None else ('—' if r['variant'] else ''):14s}")
    n_can = sum(1 for r in count_rows if not r["variant"]); n_var = len(count_rows) - n_can
    n_roots_can = sum(1 for r in CANONICAL_ROOTS if r in roots_meta); n_roots_var = sum(1 for r in VARIANT_ROOTS if r in roots_meta)
    print(f"\\n{n_can} canonical (view, attack, cell) readings at {CAP} from {n_roots_can} of {len(CANONICAL_ROOTS)} registered roots; {n_var} variant readings from "
          f"{n_roots_var} of {len(VARIANT_ROOTS)} registered roots; {len(twin_status)} twin key(s) (both clusters graded); "
          f"every 512 side cross-checked against the export's join files; every {CAP} store identity-checked against the grader's completion stamp.")
    for t in twin_status:
        print(f"  twin {t['view']}/{t['attack']}/{t['cell'][:36]} @ window {t['serving_window_tokens']}: slot <- {t['primary']['root']} ({t['primary']['served_cluster']}, "
              f"written {t['primary']['written_at']}); twin -> models_twins: {t['twin']['root']} ({t['twin']['served_cluster']}, written {t['twin']['written_at']})")"""))
REPL.append(("""    for r in LANDING_ROOTS:
        l = out["meta"]["landing_status"][r]
        print(f"  landing {r}: " + (f"graded {l['attacks_graded']}; absent {l['attacks_absent']}" if l["present"] else l["note"]))""",
"""    for r in LANDING_ROOTS + MT_ARMS_ROOTS + MT_ARMS_W24K_ROOTS:
        l = out["meta"]["landing_status"][r]
        print(f"  landing {r} [{l['namespace']}{'/' + l['variant'] if l['variant'] else ''}]: "
              + (f"graded {l['attacks_graded']} on {l['served_cluster']}; absent {l['attacks_absent']}" if l["present"] else "ABSENT (not created yet; recorded, nothing emitted)"))
    print(f"  served_cluster: {len(out['meta']['served_cluster_default']['roots_defaulted'])} pre-existing root(s) defaulted to {SERVED_CLUSTER_DEFAULT} (meta only, records unchanged); "
          f"{len(out['meta']['served_cluster_default']['roots_derived'])} root(s) derived from a record (rows carry it)")"""))

for old, new in REPL:
    n = src.count(old)
    assert n == 1, f"expected exactly one match, found {n}:\\n{old[:200]}"
    src = src.replace(old, new)
P.write_text(src)
print(f"applied {len(REPL)} replacements -> {P} ({len(src.splitlines())} lines)")
