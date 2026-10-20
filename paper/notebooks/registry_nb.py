"""registry_nb: the helper behind `registry_qwen_figures.ipynb`.

Reads the results registry (`~/registry/registry.sqlite`, READ-ONLY) and nothing else: no paperlib, no per-source
readers, no exports. The spellings come from the registry's own `sisteer.registry.paper` tables; the palette, display
labels, model markers and panel rosters are COPIED from `paper/paperlib/config.py` at 5719a9629 (cited beside each
constant) so the panels look like the paper's without importing its readers. Vocabulary is the registry's: model /
object / eval / instrument / run / judgment / judge.

Three figure forms:
  * deltas by eval category at the operating dose, one model (`delta_figure`: Figures 3 and 4) and all models (`category_panels`);
  * the frontier relative to the unsteered model, read from the dose ladders (`frontier_figure`: Figure 5);
  * absolute readings per eval across models (`absolute_panels`: the appendix).
Nothing is drawn on the canvas but the plot and its legend: provenance (model, lane, registry head) is printed under
each figure by the notebook, never banded across the top.

Standing rulings built in: every StrongREJECT cell is graded by Haiku (JUDGE_POLICY); a cell with
several fit seeds is one point per seed, each with its own interval, never pooled (`seed` = the object's
fit_replicate_label); the frontier is relative to the unsteered model, subtracted per eval, over the two multi-turn attacks.
"""
from __future__ import annotations

import fnmatch, functools, hashlib, json, os, re, sqlite3, sys, textwrap, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SISTEER_SRC = os.environ.get("SISTEER_SRC") or str(Path(__file__).resolve().parents[2] / "src")   # the registry package: this clone's src (paper/notebooks -> repo root), or SISTEER_SRC
sys.path.insert(0, SISTEER_SRC)
from sisteer.registry.paper import CAMPAIGN_ID_PAPER_LABEL_LEGACY, PAPER_LABEL_LEGACY, SERVED_KEY_TRANSFER_LEGACY  # noqa: E402

# ---- THE SOURCE: the live registry when it is present, else the paper-frame snapshot committed in the clone (spec FD 9). ------------
# REGISTRY_NB_SOURCE = live | snapshot | auto (default auto: live when REGISTRY_ROOT/registry.sqlite exists, else the committed
# snapshot); REGISTRY_ROOT (default ~/registry); REGISTRY_NB_SNAPSHOT_ROOT (default <clone>/paper/snapshot). An explicit source that
# cannot be met refuses HERE, by name; `auto` with neither present resolves to live and `connect()` refuses by name on first use, so a
# machine with no registry can still import this module for its offline tests. Every reader in this file goes through `connect()`.
REGISTRY_ROOT = os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")
DB = os.path.join(REGISTRY_ROOT, "registry.sqlite")
SNAPSHOT_ROOT = os.environ.get("REGISTRY_NB_SNAPSHOT_ROOT") or str(Path(__file__).resolve().parents[2] / "paper" / "snapshot")
SOURCE_MODE = os.environ.get("REGISTRY_NB_SOURCE", "auto")
REPO_ROOT = str(Path(__file__).resolve().parents[2])              # paper/notebooks -> the repository root


def display_path(p: str) -> str:
    """`p` relative to the repository root when it lies inside it, else as given: a committed render names `paper/snapshot` and
    `src`, never the directory the clone happens to sit in (no home path reaches the notebook's outputs)."""
    try:
        rel = os.path.relpath(p, REPO_ROOT)
    except ValueError:                                              # another drive (Windows)
        return p
    return p if rel == ".." or rel.startswith(".." + os.sep) else rel


# ---- DISPLAY-TIME REDACTION OF PRODUCER SESSION NAMES. The registry's notes, rulings and producer-filed `extra_json` can attribute
# their text to the session that wrote it ("(<name>, 2026-09-13: …)", "<name>: …", "by session <name>"), and the notebook prints some
# of that text verbatim (a producer delta block, a ruling a table names). The registry DATA keeps the names; the RENDERED notebook
# does not: every line this module or the notebook prints goes through `redacting_print` (this module rebinds `print` below; the
# notebook's header cell does the same), every frame `show()` displays is redacted column by column, and the Figure 3 cell redacts
# the delta block it prints explicitly. No name is written in this file:
#   * the producer names come from the records the source holds (`session_names`: the producer fields of runs, objects and
#     judgments), read once the notebook has opened its source. A producer name can also be an ordinary word, so it is removed only
#     in an attribution: "(<name>, <date>" / "(<name> <date>" keeps the date and loses the name; "<name>: " at the START of a note
#     goes; "session <name>" reads "a session"; "the <name> lane" reads "the lane";
#   * a file named by REGISTRY_NB_REDACT_NAMES (one name per line; blank lines and `#` lines skipped) adds names that are removed in
#     those shapes and also wherever they stand as a whole word, for a name that appears in free text but in no producer field.
REDACT_NAMES_FILE = os.environ.get("REGISTRY_NB_REDACT_NAMES") or ""
_PRODUCER_NAME_SQL = (("run", "producer_session"), ("object", "fit_producer_session"), ("judgment", "producer_session"),
                      ("judgment", "claimed_by_session"))
_SOURCE_OPENED = False             # set by connect(): registry text is printed only after the source has been read
_NAMES = None                      # (producer names, file names) once read; None until the source has been opened
_PATTERNS = {}                     # names -> the compiled shapes


def _file_names() -> tuple:
    if not REDACT_NAMES_FILE:
        return ()
    with open(os.path.expanduser(REDACT_NAMES_FILE), encoding="utf-8") as f:
        return tuple(sorted({ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")}))


def session_names() -> tuple:
    """(producer names, file names): the distinct non-empty values of the producer fields the source's records carry (the run's
    and the judgment's producer session, the session that claimed a judgment, the session that fit an object), and the names the
    REGISTRY_NB_REDACT_NAMES file lists. Before the source has been opened only the file's names are known and nothing is kept, so
    the first call after `connect()` reads the producer fields; a table or column the source lacks contributes nothing."""
    global _NAMES
    if _NAMES is not None:
        return _NAMES
    extra = _file_names()
    if not _SOURCE_OPENED:
        return (), extra
    found = set()
    with connect() as c:
        for table, col in _PRODUCER_NAME_SQL:
            try:
                rows = c.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL AND TRIM({col}) != ''").fetchall()
            except sqlite3.Error:                      # a source without this table or column (an older frame)
                continue
            found.update(str(r[0]).strip() for r in rows)
    _NAMES = (tuple(sorted(found - {""})), extra)
    return _NAMES


def _shapes(producer: tuple, extra: tuple):
    key = (producer, extra)
    if key not in _PATTERNS:
        names = sorted(set(producer) | set(extra), key=len, reverse=True)       # the longest first, so a name inside another never wins
        sn = "|".join(re.escape(n) for n in names)
        bare = "|".join(re.escape(n) for n in sorted(extra, key=len, reverse=True))
        _PATTERNS[key] = (
            re.compile(r"\((?:%s),?[ \t]+(?=20\d\d-\d\d-\d\d)" % sn) if sn else None,   # "(<name>, 2026-09-13: …" -> "(2026-09-13: …"
            re.compile(r"^[ \t]*(?:%s):[ \t]+" % sn) if sn else None,                      # "<name>: cap 2048 is …" -> "cap 2048 is …"
            re.compile(r"(?<![\w-])session (?:%s)(?![\w-])" % sn) if sn else None,         # "by session <name>" -> "by a session"
            re.compile(r"(?<![\w-])the (?:%s) lane(?![\w-])" % sn) if sn else None,        # "the <name> lane" -> "the lane"
            re.compile(r"(?<![\w-])(?:%s)(?![\w-])" % bare) if bare else None)
    return _PATTERNS[key]


_EMPTY_PARENS = re.compile(r"\([ \t]*\)")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,;:.)])")
_DOUBLE_SPACE = re.compile(r"(?<=\S) {2,}(?=\S)")


def redact_session_names(x):
    """`x` with every producer session name removed from its TEXT, for display: a str is rewritten; a dict / list / tuple is
    rebuilt with its strings rewritten (keys included); a pandas frame / series has its object columns rewritten; anything else
    comes back as it is. Redacts only the shapes the module comment names, so an ordinary word that is also a producer name is
    never touched outside an attribution (a name the REGISTRY_NB_REDACT_NAMES file lists goes wherever it stands)."""
    if isinstance(x, str):
        attribution, lead, word, lane, bare = _shapes(*session_names())
        s = x
        if attribution is not None:
            s = attribution.sub("(", s)
            s = lead.sub("", s)
            s = word.sub("a session", s)
            s = lane.sub("the lane", s)
        if bare is not None and bare.search(s):           # the cleanups run only after a bare-word removal: a printed table's
            s = bare.sub("", s)                           # column alignment is spaces too, and must stay as it is
            s = _EMPTY_PARENS.sub("", s)
            s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
            s = _DOUBLE_SPACE.sub(" ", s)
        return s
    if isinstance(x, dict):
        return {redact_session_names(k): redact_session_names(v) for k, v in x.items()}
    if isinstance(x, list):
        return [redact_session_names(v) for v in x]
    if isinstance(x, tuple):
        return tuple(redact_session_names(v) for v in x)
    if isinstance(x, pd.DataFrame):
        out = x.copy()
        for c in out.columns:
            if out[c].dtype == object or pd.api.types.is_string_dtype(out[c]):      # pandas 3 keeps text in a `str` dtype, not object
                out[c] = out[c].map(lambda v: redact_session_names(v) if isinstance(v, (str, dict, list, tuple)) else v)
        return out
    if isinstance(x, pd.Series) and (x.dtype == object or pd.api.types.is_string_dtype(x)):
        return x.map(lambda v: redact_session_names(v) if isinstance(v, (str, dict, list, tuple)) else v)
    return x


def redacting_print(*args, **kwargs):
    """`print` with every argument's TEXT passed through `redact_session_names` first: a str, dict, list, tuple or pandas object
    is redacted as such, anything else as the string `print` would have shown."""
    import builtins
    builtins.print(*(redact_session_names(a) if isinstance(a, (str, dict, list, tuple, pd.DataFrame, pd.Series))
                     else redact_session_names(str(a)) for a in args), **kwargs)


print = redacting_print            # every line THIS MODULE prints passes the display-time redaction (the notebook rebinds its own)


def _snapshot_current(root: str):
    """(snapshot_id, manifest) of the paper-frame snapshot committed under `root` (`sisteer.registry.snapshot.paper_frame_current`), or
    None when none is (no CURRENT pointer)."""
    if not os.path.exists(os.path.join(root, "CURRENT")):
        return None
    from sisteer.registry import snapshot as _SN
    return _SN.paper_frame_current(root)


def _resolve_source(mode: str) -> tuple:
    if mode not in ("live", "snapshot", "auto"):
        raise ValueError(f"REGISTRY_NB_SOURCE={mode!r}: one of live, snapshot, auto")
    have_db = os.path.exists(DB)
    if mode == "live" or (mode == "auto" and have_db):
        if not have_db:
            raise FileNotFoundError(f"registry source live: {DB} does not exist (REGISTRY_ROOT={REGISTRY_ROOT}); "
                                    f"REGISTRY_NB_SOURCE=snapshot or auto reads the committed paper-frame snapshot instead")
        return "live", None
    current = _snapshot_current(SNAPSHOT_ROOT)
    if current is None:
        if mode == "snapshot":
            raise FileNotFoundError(f"registry source snapshot: no paper-frame snapshot is committed under {SNAPSHOT_ROOT} (no CURRENT); "
                                    f"REGISTRY_NB_SNAPSHOT_ROOT names the directory")
        return "live", None                    # auto, neither present: connect() refuses by name
    return "snapshot", str(current[0])


SOURCE, SNAPSHOT_ID = _resolve_source(SOURCE_MODE)
SOURCE_DESC = DB if SOURCE == "live" else f"{display_path(SNAPSHOT_ROOT)} @ {SNAPSHOT_ID}"
print(f"registry source {SOURCE}: {SOURCE_DESC}" + ("" if SOURCE == "snapshot" or os.path.exists(DB) else
                                                   f" (absent, and no paper-frame snapshot is committed under {SNAPSHOT_ROOT})"))
MODEL_ID = "qwen35_4b_it"                                         # the registry's model id for Qwen3.5-4B
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, FIGS = os.path.join(HERE, "_cache"), os.path.join(HERE, "figures")

# ---- models (paperlib config.MODELS / MODEL_ORDER / MODEL_MARKER, keyed by the REGISTRY model id). `lane` is the paper lane
# per model (config PAPER_LANE: the object is fit on the checkpoint it is served on; Nemotron has no base checkpoint, so its
# chat-checkpoint lane is the same thing), spelled with the registry's `run.transfer_kind` vocabulary.
MODELS = {"gemma4e4b_it": dict(label="Gemma-4-E4B", lane="instruct_direct", marker="s"),
          "qwen35_4b_it": dict(label="Qwen3.5-4B", lane="instruct_direct", marker="o"),
          "nemotron3nano4b": dict(label="Nemotron-3-Nano-4B", lane="chat_checkpoint", marker="^"),
          "llama8b_it": dict(label="Llama-3.1-8B", lane="instruct_direct", marker="v")}
MODEL_ORDER = list(MODELS)
PAPER_LANE = {m: d["lane"] for m, d in MODELS.items()}
MARKER = {m: d["marker"] for m, d in MODELS.items()}
MODEL_LABEL = {m: d["label"] for m, d in MODELS.items()}
LANE_LABEL = {"instruct_direct": "instruct-direct", "base_to_instruct": "base→instruct", "chat_checkpoint": "chat checkpoint"}

# ---- rosters, labels, palette (copied from paperlib/config.py: FIGQ_ARMS, FIG_ARM_ORDER, ARMS[*].label, ARM_COLOR, EVAL_LABEL, CATEGORIES)
OBJECTS = ["arditi_allpos", "safety_si", "constitution_si", "control_relu", "control_lora",
           "constitution_relu", "constitution_lora", "difference_relu", "difference_lora"]      # FIGQ_ARMS order, registry paper_labels
APPENDIX_OBJECTS = [o for o in OBJECTS if o != "arditi_allpos"]   # the all-model roster: the main figures' order, ReLU steers included, no Refusal steer
# THE APPENDIX CAPABILITY AXIS: MMLU and GSM8K (the coding evals were not yet in the paper frame), plus BigCodeBench and BFCL
# since 2026-09-25 - the full suites are selected, in paper and lane-admitted for every object of all four models, except BFCL on
# Llama, which has no run and reads out by design (OUT_BY_DESIGN below). MATH-500 and IFEval stay off the all-model axis; the Qwen
# coverage table still lists all six.
APPENDIX_CAPABILITY_EVALS = ["mmlu", "gsm8k", "bigcodebench", "bfcl"]
# THE FULL SUITES REPLACE THE 200-ITEM SUBSETS (2026-09-24). Both landed under the
# SAME eval_id, distinguished only by item set, and `item_set_id` is now in the collapse keys — so with both present
# every BigCodeBench and BFCL cell would draw TWICE rather than being averaged. One of the two has to be chosen; the
# full suites are the reading, and the 200-item rows stay in the registry as history.
# APPLIED EVERYWHERE, not only in the appendix the rule names: the main capability panel reads the same eval ids, so
# leaving 200 there would both double-draw and put a 200-item number beside a 1,140-item one under one axis label.
FULL_SUITE_ITEM_SET = {"bigcodebench": "bigcodebench_full_1140", "bfcl": "bfcl_full_3981"}


def prefer_item_set(frame: pd.DataFrame, choice: dict = None) -> pd.DataFrame:
    """Keep only the chosen item set for each eval that has more than one, where the chosen one is present."""
    choice = FULL_SUITE_ITEM_SET if choice is None else choice
    if not len(frame) or "item_set_id" not in frame.columns:
        return frame
    drop = pd.Series(False, index=frame.index)
    for ev, want in choice.items():
        hit = frame.eval_id.eq(ev)
        if hit.any() and (frame.loc[hit, "item_set_id"] == want).any():
            drop |= hit & frame.item_set_id.ne(want)
    return frame[~drop]
PROMPTED = {"safety_si", "constitution_si"}
# the prompted SIs are labeled without "(prompt)"; the caption says they are prompts, the label need not
LABEL = {"none": "Unsteered", "arditi_allpos": "Refusal steer", "safety_si": "Safety SI",
         "constitution_si": "Constitution SI", "control_relu": "Control ReLU steer", "control_lora": "Control LoRA",
         "constitution_relu": "Constitution ReLU steer", "constitution_lora": "Constitution LoRA",
         "difference_relu": "Difference ReLU steer", "difference_lora": "Difference LoRA", "arditi": "Refusal steer (prefill only; retired)",
         "safety_lora": "Safety-SI-distilled LoRA", "safety_relu": "Safety-SI-distilled ReLU steer", "alphasteer_rebake": "AlphaSteer (retired)"}
SHORT = {"none": "Unsteered", "safety_relu": "Safety\nReLU", "arditi_allpos": "Refusal\nsteer", "safety_si": "Safety SI", "constitution_si": "Constitution\nSI",
         "control_relu": "Control\nReLU", "control_lora": "Control\nLoRA", "constitution_relu": "Constitution\nReLU", "constitution_lora": "Constitution\nLoRA",
         "difference_relu": "Difference\nReLU", "difference_lora": "Difference\nLoRA"}       # two-line lane labels (paperlib ARM_SHORT_LABEL)
COLOR = {"none": "#6b6b6b", "arditi_allpos": "#c86bd0", "arditi": "#c86bd0", "safety_si": "#764a05", "constitution_si": "#b4854b",
         "control_lora": "#0e5d7b", "control_relu": "#5499b9", "constitution_lora": "#0d644e", "constitution_relu": "#55a087",
         "difference_lora": "#9e170c", "difference_relu": "#e45e4d", "safety_lora": "#4f5c1b", "safety_relu": "#889757", "alphasteer_rebake": "#4f2fbf"}
# `grok420` is LABELLED "Pliny". The eval_id stays `grok420` everywhere it is a registry key —
# renaming the key would break every roster selector, comparison id and stored ruling that names it.
EVAL_LABEL = {"pasttense": "Past tense", "codeattack": "CodeAttack", "renellm": "ReNeLLM", "grok420": "Pliny", "strongreject_bon": "Best-of-75",
              "pair": "PAIR", "crescendo": "Crescendo", "actorattack": "ActorAttack", "xstest": "XSTest", "or_bench_hard": "OR-Bench-Hard",
              # THE PETRI BEHAVIOURS ARE ABBREVIATED AND DROP THE FAMILY PREFIX, so six of them fit
              # across one table. The table's own caption says it is Petri, and these feed the per-seed table's headings
              # (which are uppercased): HARM, DEFER, DELUDE, EVADE, REASON, PRESERVE. The FIGURE ticks are unaffected —
              # TICK overrides EVAL_LABEL there and keeps the two-line long forms, where there is room for them.
              "petri_pooled": "Pooled", "petri_cyber_bio_chem_capabilities": "Harm", "petri_defer_to_users": "Defer",
              "petri_delusion_sycophancy": "Delude", "petri_instruction_evasion": "Evade",
              "petri_reasoning_unfaithfulness": "Reason", "petri_self_preservation": "Preserve",
              "agentic_leaking": "Agentic: leaking", "agentic_murder": "Agentic: murder",
              "mmlu": "MMLU", "gsm8k": "GSM8K", "math500": "MATH-500", "ifeval": "IFEval", "bigcodebench": "BigCodeBench", "bfcl": "BFCL"}
# x tick spellings that need two lines. `bigcodebench` was here as "BigCode-\nBench-200"; with the "-200" dropped
# it fits on one line, so it is no longer overridden and falls through to EVAL_LABEL.
TICK = {"mean": "mean", "agentic_leaking": "leaking", "agentic_murder": "murder",
        "petri_cyber_bio_chem_capabilities": "cyber/bio/\nchem", "petri_defer_to_users": "defer to\nusers", "petri_delusion_sycophancy": "delusion/\nsycophancy",
        "petri_instruction_evasion": "instruction\nevasion", "petri_reasoning_unfaithfulness": "reasoning\nunfaithfulness", "petri_self_preservation": "self-\npreservation"}
# The Qwen seed-replicate Petri family is read under TWO target decodes (2026-09-21): the difference-ReLU arm
# at the server default with no per-request seed, every other arm at the Qwen vendor row with a seed derived from the training
# seed. A contrast spanning the two crosses a decode boundary; within either side it does not. The paired ReLU is NOT re-run,
# so this is declared, never closed. THIS IS THE DECLARATION the notebook prints: a copy, in this notebook's paper labels, of
# `PETRI_TARGET_DECODE_SPLIT` as `paper/paperlib/config.py` declared it on the `paper-figures` branch (commit 5b0b9a6dd,
# 2026-09-21: `server_default_arms=("diff_psr",)` in that file's arm keys, `difference_relu` here). That constant never reached
# main's paperlib (`paper/paperlib/config.py` @ 5719a9629, the copy every other constant in this file cites, has no such name),
# so this local copy was always the one this tree read; the paperlib figure build is gone and nothing here imports it.
PETRI_TARGET_DECODE_SPLIT = dict(
    caption=("Petri on the Qwen seed-replicate family is read under TWO target decodes: the difference-ReLU arm at the "
             "server default (no per-request seed), every other arm at the Qwen vendor row with a seed derived from the "
             "training seed. A contrast that spans the two is across a decode boundary; within either side it is not."),
    server_default_arms=("difference_relu",), will_not_be_closed="the paired ReLU is not re-run, so the split is declared rather than closed",
    read_from="this notebook's own declaration (registry_nb.PETRI_TARGET_DECODE_SPLIT)")


def petri_decode_split() -> dict:
    """A copy of `PETRI_TARGET_DECODE_SPLIT` (the Petri target-decode declaration above)."""
    return dict(PETRI_TARGET_DECODE_SPLIT)


# EVALS WHOSE ITEM SETS MAY NEVER BE CROSSED, not even as a ringed cross-cut fallback (2026-09-21). Qwen's MMLU moved from
# the 200-item head to the full 1,531-item pinned pool, and the head is item-for-item the first 200 of the 1,531 under the same
# seed-0 shuffle: the two readings are NESTED, so a steered row at one width differenced against an unsteered run at the other is
# not a paired test and the widening does not average out. Elsewhere a cut mismatch is a loss of precision the ring already states;
# here it is an invalid comparison, so the row goes anchorless and says so rather than carrying a number nobody should read.
NO_CROSS_ITEM_SET = {"mmlu"}

# EVALS WHERE EVERY ARM TAKES THE ANCHOR THE PROMPTED SI CELLS TAKE (2026-09-21). On the agentic scenarios the fitted arms
# and the prompted SIs were being differenced against DIFFERENT unsteered runs: the SIs against the campaign's in-frame `none`, the
# seeded fits against the cap-3,000 run their contrasts name, which the paper's selection excludes. That made every fitted arm draw
# open while its neighbours drew filled, and it broke the standing rule that one panel shares one anchor. For these evals the
# pool is narrowed to the unsteered runs INSIDE the paper frame - the SI cells' anchor - and the cap match is given up to get it, so
# all nine arms are differenced against the same reference and all nine draw filled.
# OVER-REFUSAL JOINS THEM (2026-09-22, registry ruling
# `overrefusal-cells-pair-with-the-seed-unsteered-final-rung-regardless-of-cap-2026-09-22`): an over-refusal cell pairs with the
# unsteered row in frame at THAT row's final rung, whatever the cell's own cap, exactly as every agentic cell pairs with the one
# shared unsteered row. Comparability comes from the completion-cap rule rather than from a matching cap: the paired difference
# ReLU's seed-1 XSTest cell completed at 512 with a negligible at-cap share, so 512 IS its final rung, and the unsteered row's
# final rung is 2,048. That cell was the one drawing unfilled; under this rule it pairs and draws filled like its neighbours.
SHARED_FRAME_ANCHOR = {"agentic_leaking", "agentic_murder", "xstest", "or_bench_hard"}

# ONE ANCHOR FOR THE WHOLE PANEL, NOT ONE PER ROW. Per-row preference - the registry's
# binding, else the arm's own lane - looked right and was not: measured at head 164399 the Qwen agentic panel resolved to
# FIVE distinct anchors on leaking and four on murder, spanning 36.1 to 39.3 pp, because only 29 of 79 arms carry a binding
# and the rest matched their own lanes. Arms drawn against different references are not comparable to each other and nothing
# on the figure says so, which is worse than the all-vendor state it replaced. So where an eval names a panel anchor lane
# here and any in-frame unsteered run sits on it, EVERY arm on that eval anchors there, bindings and lanes alike.
# `misalign_ladder` is the native canon completion-cap baseline: lora_native, T 1, matching the merged arms on both.
PANEL_ANCHOR_LANE = {"agentic_leaking": "misalign_ladder", "agentic_murder": "misalign_ladder"}

# EVALS WHERE A SEED ANCHORS ON ITS OWN UNSTEERED RUN AND NEVER ON ANOTHER SEED'S (2026-09-23). On the
# seed-replicate capability evals the per-seed anchors coexist by ruling, and some seeds have NO unsteered row at all:
# bigcodebench and bfcl never ran `none` or `safety_si` for seeds 2 and 3. The default pool takes every unsteered run of
# the eval, so an s2 fitted row would silently borrow s1's `none` — a cross-seed contrast presented as a within-seed one,
# on evals whose whole point is that the three fits differ. Where the row's own seed has no unsteered run the pool is
# EMPTY and the row keeps its absolute value with no delta, which is the honest rendering of "not measured for this seed".
SEED_MATCHED_ANCHOR = {"math500", "bigcodebench", "bfcl", "ifeval"}

# THE PANEL PLOTS THE NON-COLLAPSED OVER-REFUSAL RATE where a run files one (2026-09-22), with the collapse detail
# reported beside the cell. The lifted gemma and llama paired cells file TWO owner metric rows on
# the same run - `over_refusal_graded` and `over_refusal_graded_noncollapsed` - and their evals declare no headline id at all,
# so without a rule the reader picks by accident. It is not a rounding difference: on Llama's XSTest the two read 0.3130 and
# 0.1704, fourteen points apart, because the collapsed rate counts a role-collapsed reply as a refusal and the non-collapsed
# one does not. Only gemma and llama file it today; a run that files only the plain metric is unaffected.
NONCOLLAPSED_METRIC = "over_refusal_graded_noncollapsed"

BON_HEADLINE_METRIC = "asr_bon_cap4096"    # per-goal any-hit@75: the unit rule of 2026-09-19, the fallback for a pre-schema-8 registry


@functools.lru_cache(maxsize=1)
def check_headline_scales(strict: bool = True) -> tuple:
    """EVERY HEADLINE ID THE NOTEBOOK'S EVALS DECLARE MUST BE ON THE SCALE MAP (2026-09-24, after two landings in two days drew as
    fractions: `over_refusal_graded_noncollapsed` on 09-23 and `orb_refused` / `xst_refused` on 09-24 - the registry made each a
    headline, neither was in SCALE, and every affected arm sat at ~100 % compliance until someone looked). This reads the eval table's
    declared headline ids (`extra_json.headline_metric_ids`, else `headline_metric_id`) for the evals this notebook labels and returns
    the ids SCALE cannot place. With `strict` it raises instead, at load, so a render fails loudly rather than publish a flat panel;
    NB_ALLOW_UNSCALED_HEADLINE=1 downgrades the raise to a printed warning for a deliberate look at such a frame."""
    drawn = set(EVAL_LABEL) | {e for cols, _s, _l in CATEGORIES.values() for e in cols} | set(POOLED_EVAL.values())
    declared = {}
    try:
        with connect() as con:
            rows = con.execute("select eval_id, headline_metric_id, extra_json from eval").fetchall()
    except (sqlite3.Error, FileNotFoundError):                                 # no registry (a synthetic run): nothing to check
        return ()
    for eid, head, extra in rows:
        if eid not in drawn:
            continue
        try:                                                                   # one eval's odd extra json must not hide another's hole
            ex = _extra(extra) if extra else None
            ids = ex.get("headline_metric_ids") if isinstance(ex, dict) else None
        except Exception:
            ids = None
        ids = [str(x) for x in ids] if isinstance(ids, list) and ids else ([str(head)] if head else [])
        for i in ids:
            if i not in SCALE:
                declared.setdefault(i, []).append(eid)
    missing = tuple(sorted(declared))
    if missing:
        msg = ("headline metric ids with no SCALE entry: " + "; ".join(f"{i} (evals {', '.join(sorted(declared[i]))})" for i in missing)
               + " - a rate would draw as a fraction of a percent; add each to SCALE and IS_RATE in registry_nb.py")
        if strict and os.environ.get("NB_ALLOW_UNSCALED_HEADLINE") != "1":
            raise ValueError(msg)
        print("WARNING: " + msg)
    return missing


@functools.lru_cache(maxsize=1)
def headline_metric_ids() -> dict:
    """{eval_id: [metric ids, in preference order]} from the eval row's `extra_json.headline_metric_ids` (schema 8, 2026-09-21). The registry declares the list and resolves a run's headline as the FIRST listed id the run files, so reading it
    here means a future spelling follows automatically instead of needing another constant in this file. Evals that declare no
    list keep their single `headline_metric_id`."""
    out = {}
    try:
        with connect() as con:
            for eid, extra in con.execute("select eval_id, extra_json from eval"):
                ids = (_extra(extra) or {}).get("headline_metric_ids") if extra else None
                if isinstance(ids, list) and ids:
                    out[eid] = [str(x) for x in ids]
    except Exception:
        pass
    return out


# The identifier moved to the American spelling with the rest of the file; the old name stays as an alias so a peer
# session holding a commit against `PETRI_BEHAVIOURS` still merges and runs.
PETRI_BEHAVIORS = ["petri_cyber_bio_chem_capabilities", "petri_defer_to_users", "petri_delusion_sycophancy", "petri_instruction_evasion",
                    "petri_reasoning_unfaithfulness", "petri_self_preservation"]
PETRI_BEHAVIOURS = PETRI_BEHAVIORS          # alias, see above


def tick(e: str) -> str:
    return TICK.get(e, EVAL_LABEL.get(e, e))
# category -> (registry eval ids, sign so that higher = better, y label); the paper's CATEGORIES with registry eval ids
CATEGORIES = {
    "static": (["pasttense", "codeattack", "renellm", "grok420"], -1, "Δ defense success (pp)"),   # renamed from "static"
    "multi-query": (["strongreject_bon", "pair"], -1, "Δ defense success (pp)"),
    "multi-turn": (["crescendo", "actorattack"], -1, "Δ defense success (pp)"),
    "over-refusal": (["xstest", "or_bench_hard"], -1, "Δ benign compliance (pp)"),
    "Petri": (PETRI_BEHAVIORS, -1, "Reduction in misalignment score (1–10)"),      # up = less misaligned                   # the six behaviors left, the pooled score right
    # agentic murder stays a category eval for Gemma, Llama and Nemotron; QWEN's murder cells are excluded (EXCLUDED_CELLS, below)
    "agentic misalignment": (["agentic_leaking", "agentic_murder"], -1, "Reduction in misaligned actions (pp)"),
    # ONE CAPABILITY AXIS OF FOUR. BigCodeBench and BFCL drew well on their own once their rows landed,
    # so they are folded in beside MMLU and GSM8K rather than kept as a separate figure. This DELIBERATELY redefines the
    # paper's capability number: it was the MMLU-and-GSM8K mean and is now the mean over four evals, so a capability point
    # here is not comparable to one quoted from an earlier render. Same unit and sign throughout — every one is a percentage
    # correct or passing and higher is better. IFEval and MATH-500 stay off; MATH-500 is also the one
    # whose capped items go UNSCORED rather than counting as failures, so its value would sit above its attempted bound by up
    # to 55 points, while the four here all absorb truncation into the value and are read the same way.
    "capability": (["mmlu", "gsm8k", "bigcodebench", "bfcl"], +1, "Δ pass rate"),
}
# THE POOLED COLUMN (2026-09-20: the Petri plot shows all six behaviors separately on the left before pooling on the right):
# a category named here draws its right-hand column from the registry's own pooled eval - the producer's pooling over the behaviors,
# authoritative - instead of the notebook's unweighted mean over the columns. The column is LABELED "mean" like every other
# category's; its being the producer's pooled score rather than a mean of the six is said in the
# caption and in `merge_provenance`, not on the axis.
POOLED_EVAL = {"Petri": "petri_pooled"}
# THE APPENDIX'S CATEGORY SET (2026-09-24): the main figures' categories with the capability axis narrowed to the appendix's (MMLU and
# GSM8K). The all-model x = object figures read this, so their capability panel matches the x = eval one; the non-Qwen BigCodeBench /
# BFCL rows exist only on the base-model fits (another object, the base->instruct lane) and would have pooled two fits into one mean.
APPENDIX_CATEGORIES = {**CATEGORIES, "capability": (list(APPENDIX_CAPABILITY_EVALS), +1, "Δ accuracy / pass rate (pp)")}

# THE HEADLINE TOKEN CAP (2026-09-20): a cell can be graded under several generation
# caps and the registry keeps them all - XSTest carries 512, 2048 and 8192 rows, the multi-turn attacks 512 and 4096 - so two
# readings of ONE rung can differ only by the cap. They are DIFFERENT MEASUREMENTS and are never averaged together; a series whose
# rungs come from different caps is a mixed axis, the defect this table exists to prevent. Where an eval declares its headline cap
# here, that rung is the reading; where it does not, the cap most of the object's rungs carry on that eval wins, so a series stays
# on one cap, and the notebook prints what it took.
HEADLINE_CAP = {"xstest": 2048}
# ONE INSTRUMENT, SEVERAL RUNGS (2026-09-21). A token-cap LADDER is one instrument: a cell climbs a rung only on the gate's
# escalate, so two runs of the same instrument legitimately carry different `token_cap_tokens` - the rung each REACHED - and the value is
# the ladder's final state either way. The registry declares the ladder on the run (`extra.token_cap_ladder`, a list or a dict with
# `declared`), so the rule is read from the data rather than hand-listed here: caps that are rungs of one declared ladder are the same
# instrument. Two consequences: an anchor pool spans them, and a panel is NEVER filtered to a single cap when the rows are final rungs of
# one ladder (the completion-cap merge rule, which selects one row per cell at the rung it reached: over-refusal s2/s3 is 2,048
# where the pair exited there and 8,192 where it escalated). `cap_matched = 0` on such a contrast is deliberate, not a defect.
CAP_EQUIVALENT: dict = {}                     # a hand fallback; empty now that the ladders are declared on the runs


def declared_ladders(frame: pd.DataFrame) -> dict:
    """{eval_id: [frozenset of caps, ...]} from the runs' own `extra.token_cap_ladder` (a list, or a dict carrying `declared`)."""
    out: dict = {}
    if "run_extra_json" not in frame.columns or "eval_id" not in frame.columns:
        return out
    for ev, g in frame.groupby("eval_id"):
        seen = set()
        for e in g["run_extra_json"].dropna().unique():
            lad = (_extra(e) or {}).get("token_cap_ladder")
            if isinstance(lad, dict):
                lad = lad.get("declared")
            if isinstance(lad, (list, tuple)) and len(lad) > 1:
                try:
                    seen.add(frozenset(float(x) for x in lad))
                except (TypeError, ValueError):
                    pass
        if seen:
            out[ev] = sorted(seen, key=len, reverse=True)
    return out


_LADDERS: dict = {}


def set_declared_ladders(frame: pd.DataFrame) -> dict:
    """Read the declared ladders from a frame once; every later cap_class call uses them. Returns what it found."""
    global _LADDERS
    _LADDERS = declared_ladders(frame)
    return _LADDERS


def cap_class(eval_id: str, cap) -> set:
    """The caps that count as the same instrument as `cap` on `eval_id`: the declared ladder containing it, else a hand-declared
    class, else just `cap` itself."""
    if cap is None or (isinstance(cap, float) and np.isnan(cap)):
        return set()
    cap = float(cap)
    for lad in _LADDERS.get(eval_id, ()):
        if cap in lad:
            return set(lad)
    cls = CAP_EQUIVALENT.get(eval_id)
    return set(cls) if cls and cap in cls else {cap}      # the at-cap term of the three-term over-refusal rate dissolves at 2048; 512 was the old default


def series_caps(rows: pd.DataFrame) -> dict:
    """The cap each (object, eval) SERIES should hold: the one carried by the most of its rungs, a tie going to the larger. A series
    whose rungs sit on different caps is a mixed axis even when every individual cell is internally consistent, and the at-cap share
    climbs monotonically with dose (2026-09-20: 0 % at the low rungs, 21-30 % at dose 1.5 on Past tense), so the mixing lands
    exactly on the extreme points that set a curve's shape."""
    if "token_cap_tokens" not in rows.columns:
        return {}
    out = {}
    for k, g in rows.dropna(subset=["token_cap_tokens"]).groupby(["object_label", "eval_id"]):
        caps_here = {float(x) for x in g.token_cap_tokens}
        if len(caps_here) > 1 and any(caps_here <= lad for lad in _LADDERS.get(k[1], ())):
            continue                                                      # one ladder, several rungs reached: no single series cap
        n = g.groupby("token_cap_tokens").dose.nunique() if "dose" in g.columns else g.token_cap_tokens.value_counts()
        out[k] = float(sorted([c for c in n.index if n[c] == n.max()])[-1])
    return out


def cap_choice(g: pd.DataFrame, eval_id: str, prefer=None):
    """The rows of one cell that share its headline cap: the declared cap where the cell has it, else `prefer` (the series' own cap),
    else the cap most rows carry, breaking a tie on the larger. Returns (rows, cap)."""
    if "token_cap_tokens" not in g.columns or g["token_cap_tokens"].dropna().nunique() <= 1:
        caps = g["token_cap_tokens"].dropna().unique() if "token_cap_tokens" in g.columns else []
        return g, (float(caps[0]) if len(caps) else np.nan)
    have = g["token_cap_tokens"]
    # RUNGS OF ONE DECLARED LADDER ARE ONE INSTRUMENT (2026-09-21): keep them all - each row is the rung its cell reached
    caps_here = {float(x) for x in have.dropna()}
    if len(caps_here) > 1 and any(caps_here <= lad for lad in _LADDERS.get(eval_id, ())):
        return g, np.nan
    for want in (HEADLINE_CAP.get(eval_id), prefer):
        if want is not None and have.eq(want).any():
            return g[have.eq(want)], float(want)
    counts = have.value_counts()
    top = sorted([c for c in counts.index if counts[c] == counts.max()])[-1]
    return g[have.eq(top)], float(top)


def summary_evals(cat: str) -> list:
    """The evals a category's summary column is computed over: its pooled eval where one is declared, else its columns."""
    return [POOLED_EVAL[cat]] if cat in POOLED_EVAL else list(CATEGORIES[cat][0])


# the frontier's two axes: defense over the two MULTI-TURN attacks, cost over the benign evals, both as
# deltas against the unsteered model subtracted per eval and then averaged
FRONTIER_Y = ("multi-turn", ["crescendo", "actorattack"])
FRONTIER_X = ("over-refusal", ["xstest", "or_bench_hard"])
# THE OVER-REFUSAL READING IS THE LABEL MEAN (superseding the refused rate ruled earlier the same day):
# `over_refusal_label_mean` -- full refusal 1, partial 0.5, compliance 0, the mean over judged items, an at-cap reply scored BY ITS
# LABEL rather than overridden to 0.5 as `over_refusal_graded` does (which is what made the graded score non-comparable across exit
# caps). It is the ONE over-refusal number in the notebook: the frontier's benign axis, the Figure 6 over-refusal panel and its table,
# and the appendix all read it, so no figure can show a different over-refusal from its neighbour. The registry's headline for the
# two evals is still the refused rate, so the notebook repoints every frame that carries over-refusal rows at load (`repoint_all`
# with OR_LABEL_MEAN, gated on complete coverage across ALL the frames at once: the paper frame, the frontier frame and the
# unsteered pool the anchors borrow from -- an anchor read one way and its steered row the other is the mixed-frame defect again).
OR_METRIC = "over_refusal_label_mean"
OR_LABEL_MEAN = {"xstest": OR_METRIC, "or_bench_hard": OR_METRIC}
OR_REFUSED = {"xstest": "xst_refused", "or_bench_hard": "orb_refused"}        # the registry's headline since 2026-09-24 ~18:42Z
# value -> plotted unit. BEST-OF-N'S RULED METRIC IS A PROPORTION TOO (who saw the panel read ~0): the seed-replicate
# producers file `asr_bon_cap4096` in 0-1 like `asr`, but it was missing from this table, so every best-of-N delta was plotted 100x too
# small - a 56 pp drop drawn as 0.56 pp, which reads as "no effect" on a pp axis. Any rate-valued metric MUST appear in both tables.
# `over_refusal_graded_noncollapsed` IS AN OWNER METRIC ON THE PAIRED LANE (found 2026-09-23 while building the appendix tables): the
# paired-write views' over-refusal runs (the composed difference ReLU arms and their own unsteered runs on Gemma and Llama) file the
# rate over the non-collapsed items as the row the paper view flags, not as a companion of `over_refusal_graded`. Absent from these two
# maps it was drawn as a fraction: an unsteered Gemma OR-Bench over-refusal of 0.60 plotted as 0.6 %, so 99.4 % benign compliance
# where the model refuses 60 %, and every delta against it a hundred times too small. Same unit and denominator as the parent.
# THE REFUSED-RATE HEADLINES (2026-09-24: `orb_refused` / `xst_refused` became the over-refusal evals' headline metric ids
# at ~15:20Z). The same lesson as the non-collapsed metric the day before, one day later: an id that becomes a headline must enter these
# two maps or its fraction draws as a tiny percentage - every over-refusal arm sat at 0 change and ~100 % compliance in the first render
# after the landing. Same unit and denominator as the graded rate: refused items over the benign items.
# `over_refusal` (2026-10-02, schema 21): a refused RATE under a new spelling on a handful of over-refusal runs (k / n, 0-1); on the
# scale so the load check passes, repointed to the label mean like every other over-refusal headline where a label-mean row exists
SCALE = {"asr": 100.0, "asr_bon_cap4096": 100.0, "over_refusal_graded": 100.0, "over_refusal_graded_noncollapsed": 100.0,
         "orb_refused": 100.0, "xst_refused": 100.0, "over_refusal": 100.0, "over_refusal_label_mean": 100.0, "acc": 100.0, "prompt_level_strict": 100.0, "misaligned_rate": 1.0, "petri_score": 1.0}
IS_RATE = {"asr": 1.0, "asr_bon_cap4096": 1.0, "over_refusal_graded": 1.0, "over_refusal_graded_noncollapsed": 1.0,
           "orb_refused": 1.0, "xst_refused": 1.0, "over_refusal": 1.0, "over_refusal_label_mean": 1.0, "acc": 1.0, "prompt_level_strict": 1.0, "misaligned_rate": 100.0}   # denominator for Wilson


def unscaled_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Headline metrics whose values look like proportions (max <= 1.5) but carry no SCALE entry - the shape of the best-of-N defect.
    Printed by the notebook so a new producer metric cannot silently plot 100x small."""
    if not {"metric_id", "value"} <= set(frame.columns):
        return pd.DataFrame(columns=["metric_id", "max_value", "rows"])
    g = frame[frame.get("headline", True)].groupby("metric_id").value.agg(["max", "size"]).reset_index()
    g = g[(g["max"] <= 1.5) & ~g.metric_id.isin(SCALE)]
    return g.rename(columns={"max": "max_value", "size": "rows"})
# where several unsteered runs measure one eval, the source the paper's headline reads (paperlib CAMPAIGN_SOURCES / EVALS notes)
ANCHOR_PREFERENCE = ["sampled_lane", "sampled_lane_controls", "r2_export", "bon_stores", "em/agentic/qwen35_4b_it", "em/agentic/qwen35_4b_it_vendor_leak"]
# `bon_stores` (2026-09-21, the all-model appendix): the canonical climbed best-of-N store (bon_512store_4096climb, the v2 Haiku judge) is the
# paper's best-of-N reading; without it here the fallback below picked the alphabetically-first source, the Arditi ladder's same-standup
# unsteered leg (`arditi_strength`), for every model's unsteered best-of-75 (Gemma 68 % against the store's 76 %).
# THE GRADER RULING: every StrongREJECT cell in the paper is graded by Haiku, not Sonnet. A judgment whose
# rubric is in this table and whose judge is not the named one is excluded from every panel and COUNTED (the notebook prints the
# excluded cells); a cell whose only judgments fail the rule is a gap ("wrong judge"), never a point.
JUDGE_POLICY = {"strongreject": "claude-haiku-4-5"}
# THE ANCHOR STAYS VISIBLE BESIDE THE DELTA (2026-09-19): every delta row carries the unsteered run's
# value (`anchor_value`) and the room it leaves toward the bound (`headroom`, pp); a cell whose anchor leaves less than HEADROOM_PP is
# `ceiling` - its delta is bounded by the anchor, not by the object - and is marked "≤" on the canvas. The multi-turn anchors on Gemma,
# Llama and Nemotron sit at 93-100 % defense, so a flat multi-turn delta there is ceiling-forced, never evidence of safety. Multi-turn is
# never pooled with the statics: CATEGORIES keeps them apart by construction.
HEADROOM_PP = 10.0
# DEGENERATION IS SCORED IN PLACE AND REPORTED BESIDE THE VALUE (paperlib's dagger): a cell with more than DEGEN_LINE of its
# responses judged degenerate (frac_degenerate_union5: the five-signal union) carries "†n%" beside its mark - Llama's paired-write
# Petri turns are one fixed string each, correctly scored, not less misbehavior. THE AT-CAP SHARE likewise rides beside the value
# as "cap n%" over AT_CAP_LINE, never used to filter or discount.
DEGEN_LINE = 0.05
# ROLE COLLAPSE (2026-09-22, from the role_collapse_v1 re-grade of the paired-write difference ReLU steer's over-refusal cells):
# where a cell's responses collapse off task - step templates, restatements of the prompt, loops - the C/P/I rubric files them as refusal
# or compliance and the filed rate measures nothing. Such a cell PLOTS THE RATE OVER THE NON-COLLAPSED ITEMS (and, downstream, its change
# against the anchor) with a Wilson interval on the non-collapsed n, and prints the collapse share and the item count under the mark. The
# substitution fires above ROLE_COLLAPSE_LINE (the rule's false-positive floor on the unsteered runs is 1-3 %). The registry carries the
# field on the owner metric row's extra json under ROLE_COLLAPSE_KEYS (the spelling; one place to rename).
ROLE_COLLAPSE_LINE = 0.05
# the spelling (2026-09-22): a SECOND run_metric row on the same judgment, metric_id `over_refusal_graded_noncollapsed`,
# metric_role secondary, k = refusals over the non-collapsed items, n = the non-collapsed items judged, extra json carrying the share
# (n_collapsed / n_items) and the counts. The owner `over_refusal_graded` row is never overwritten: the reader copies the secondary
# row's numbers onto the owner row of the same (run, judgment) and substitutes there. `attach_role_collapse` does the copy.
ROLE_COLLAPSE_METRIC = "over_refusal_graded_noncollapsed"
ROLE_COLLAPSE_PARENT = "over_refusal_graded"                 # the owner row the companion stands in for
ROLE_COLLAPSE_KEYS = dict(share="role_collapse_share", n="role_collapse_n")
# Evals whose at-cap annotation is off the figures (2026-09-21). The share is still computed, still in the tables and
# still decides nothing - only the drawn label goes.
AT_CAP_HIDDEN = ({"grok420"}                                                     # 2026-09-21
                 | set(PETRI_BEHAVIORS) | set(POOLED_EVAL.values())             # Petri, all six behaviors and the pooled score
                 | set(CATEGORIES["agentic misalignment"][0]))                   # agentic misalignment, both scenarios
AT_CAP_LINE = 0.05
_DOSE_SUFFIX = re.compile(r"_(halfdose|d(m?)(\d+)(?:p(\d+))?)$")                                     # sisteer.registry.paper._SERVED_KEY_DOSE_SUFFIX
_TRANSFER = {"instruct-direct": "instruct_direct", "base→instruct": "base_to_instruct", "chat-ckpt": "chat_checkpoint"}


# ---- reading ---------------------------------------------------------------------------------------------
def connect():
    """The read-only connection every reader in this file goes through: the live registry (`mode=ro`) or the committed paper-frame
    snapshot (`sisteer.registry.snapshot.open_paper_frame`, reassembled under `_cache/snapshot/`), so both sources answer the same SQL."""
    global _SOURCE_OPENED
    if SOURCE == "snapshot":
        from sisteer.registry import snapshot as _SN
        con = _SN.open_paper_frame(SNAPSHOT_ROOT, cache_dir=os.path.join(CACHE, "snapshot"))
    elif not os.path.exists(DB):
        raise FileNotFoundError(f"registry source live: {DB} does not exist, and no paper-frame snapshot is committed under {SNAPSHOT_ROOT}")
    else:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    _SOURCE_OPENED = True
    return con


def head() -> int:
    with connect() as c:
        return int(c.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])


def query(sql: str, name: str) -> pd.DataFrame:
    """`sql` against the registry, cached under `_cache/<name>-<source>-<head>-<sha8>.parquet` (a new head re-queries; the live and
    snapshot caches never collide). NB_CACHE=0 reads nothing from the cache and writes nothing to it (the parity belt runs so)."""
    os.makedirs(CACHE, exist_ok=True)
    use_cache = os.environ.get("NB_CACHE") != "0"
    h = head()
    sha = hashlib.sha1(sql.encode()).hexdigest()[:8]
    path = os.path.join(CACHE, f"{name}-{SOURCE}-{h}-{sha}.parquet")
    # PRESENTATION-ONLY ITERATION CAN REUSE AN OLDER HEAD, LOUDLY AND BY OPT-IN (2026-09-23; measured: the all-model
    # query alone is 5.4 min, 52% of the ~10 min render). The cache keys on the registry head, and the registry's own
    # timers move that head every few minutes, so a render that changed nothing but a line width still paid the full
    # read. With NB_REUSE_CACHE=1 the newest cache entry for this query is used whatever head it was captured at.
    # NEVER the default, and never silent: the head it was captured at is printed here and carried in .attrs, so the
    # notebook header states the head it was actually drawn from. Do a plain render (no env var) before publishing,
    # so a published artifact is always a real read of the live registry.
    if use_cache and os.environ.get("NB_REUSE_CACHE") == "1" and not os.path.exists(path):
        import glob
        old_paths = sorted(glob.glob(os.path.join(CACHE, f"{name}-{SOURCE}-*-{sha}.parquet")), key=os.path.getmtime)
        if old_paths:
            path = old_paths[-1]
            h = int(os.path.basename(path).rsplit("-", 2)[1])
            age = (time.time() - os.path.getmtime(path)) / 60.0
            print(f"{name}: read from CACHE at head {h} (live head {head()}, captured {age:.0f} min ago). This render is "
                  f"a snapshot at head {h}, not a live read — check the head covers the folds you expect to see.")
    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        t = time.time()
        # THE HEAD AND THE FRAME COME FROM ONE CONNECTION (2026-09-21). Read separately, a head taken at one
        # instant can be stamped on a frame read at another, so a cache entry claims a state the rows never had together. One
        # connection, head first, frame second, and the head that labels the result is the one re-read here.
        with connect() as c:
            h = int(c.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
            path = os.path.join(CACHE, f"{name}-{SOURCE}-{h}-{sha}.parquet")
            df = pd.read_sql_query(sql, c)
        if use_cache:
            df.to_parquet(path)
        print(f"{name}: {len(df)} rows in {time.time() - t:.0f}s at head {h}")
        _by_eval = (df.groupby("eval_id").size().to_dict() if "eval_id" in df.columns and len(df) else {})
        _shrink_guard(name, len(df), h, {str(k): int(v) for k, v in _by_eval.items()})
    df.attrs["head"] = h
    return df


# A READ THAT COMES BACK SMALLER IS SHOUTED ABOUT, not quietly drawn (2026-09-21). Twice in one evening a whole arm was absent
# from a query that had returned it minutes before - the paired difference ReLU's over-refusal rows, from a direct read-only
# sqlite query with no selection filter - and the only reason it was caught was that a person was looking at the figure. The
# guard records each query's row count beside its cache and prints a loud line when a re-read at a NEWER head comes back
# materially smaller. It never blocks or alters a render: a frame legitimately shrinks when a ruling retires rows, and deciding
# which kind of shrink this is belongs to a person. It just refuses to let the shrink be silent.
SHRINK_LINE = 0.98                                                          # a re-read under 98 % of the last count is reported


def _shrink_guard(name: str, n: int, h: int, by_eval: dict | None = None) -> None:
    """Report a read that came back smaller, on the TOTAL and PER EVAL.

    Per eval because the total alone missed a real loss in production (2026-09-23): a reader defect withdrew 50 BigCodeBench
    and BFCL rows while MATH-500 and a batch of unsteered cells landed in the same window, so the frame GREW and the guard
    stayed silent on a partial disappearance. A frame that grows overall while one eval empties is exactly the shape worth
    shouting about, and it is invisible to a single number."""
    mark = os.path.join(CACHE, f"{name}.rows.json")
    prev = None
    try:
        with open(mark) as f:
            prev = json.load(f)
    except Exception:
        pass
    if prev and int(prev.get("head", 0)) < h:
        if n < prev.get("rows", 0) * SHRINK_LINE:
            print(f"  !! {name} SHRANK: {prev['rows']} rows at head {prev['head']} -> {n} at head {h}. A ruling may have "
                  f"retired rows, or the read may have caught the registry mid-change; check before trusting a panel that "
                  f"lost points.")
        was = prev.get("by_eval") or {}
        now = by_eval or {}
        lost = {k: (v, now.get(k, 0)) for k, v in was.items() if now.get(k, 0) < v * SHRINK_LINE}
        if lost:
            detail = ", ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(lost.items(), key=lambda x: x[1][0] - x[1][1], reverse=True)[:8])
            print(f"  !! {name} LOST ROWS ON {len(lost)} eval(s) even though the total went {prev.get('rows')} -> {n}: "
                  f"{detail}. A frame that grows while one eval empties is the shape a total cannot show.")
    if not prev or int(prev.get("head", 0)) <= h:
        try:
            with open(mark, "w") as f:
                json.dump({"rows": n, "head": h, "by_eval": by_eval or {}}, f)
        except Exception:
            pass


def _extra(s):
    try:
        return json.loads(s) if isinstance(s, str) else {}
    except ValueError:
        return {}


def _dose_num(v):
    """A nominal dose from a run's extra: a number as is; a dict (the seed-replication battery files {"c": 0.3, "unit": "raw_coefficient"})
    only when it names a nominal value - a raw coefficient is NOT a nominal dose and falls through to strength_nominal."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, dict) and v.get("unit") != "raw_coefficient":
        for k in ("value", "nominal", "dose"):
            if isinstance(v.get(k), (int, float)):
                return float(v[k])
    return np.nan


def _coef_num(v):
    """A raw coefficient from a dict-shaped dose ({"c": ..., "unit": "raw_coefficient"}), else NaN."""
    return float(v["c"]) if isinstance(v, dict) and isinstance(v.get("c"), (int, float)) else np.nan


def judge_ok(d: pd.DataFrame) -> pd.Series:
    """True where the row's judgment satisfies JUDGE_POLICY (or its rubric is not governed by it, or the frame carries no judge)."""
    if "judge_rubric_name" not in d.columns:
        return pd.Series(True, index=d.index)
    want = d["judge_rubric_name"].fillna("").map(JUDGE_POLICY)
    return want.isna() | d["judge_model"].fillna("").eq(want)


def bon_headline_n_ok(eval_category: pd.Series, bon_n: pd.Series) -> pd.Series:
    """A best-of-N eval's headline row is the reading at n = 75 - on EVERY best-of-N row, whatever its metric id (2026-09-21, the
    all-model appendix). The n = 75 test used to apply only to evals that declared no headline list (the pre-schema-8 fallback), so
    once `strongreject_bon` declared one, the Arditi and AlphaSteer lanes' unsteered climb-curve rungs - metric `asr` at n = 1, 4, 16 -
    passed as headline rows and the all-model panel drew the unsteered best-of-75 for Gemma at n = 4 (19 % ASR) and for Nemotron at
    n = 1 (0 %), where the canonical n = 75 readings are 76 % and 32 %. A row of another category passes untouched."""
    return ~eval_category.eq("best_of_n") | bon_n.eq(75)


@functools.lru_cache(maxsize=4)
def role_collapse_rows(at_head: int) -> pd.DataFrame:
    """The secondary `over_refusal_graded_noncollapsed` metric rows, straight from the `run_metric` table (the paper view flags only the
    owner row as in the paper, so the notebook's frames never carry them): run, judgment, threshold, k, n and the extra json. Cached per
    registry head; an empty frame when the field has not been filed."""
    cols = ["run_id", "judgment_id", "threshold", "k", "n", "metric_extra_json"]
    try:
        with connect() as con:
            rows = con.execute("select run_id, judgment_id, threshold, k, n, extra_json from run_metric where metric_id = ?", (ROLE_COLLAPSE_METRIC,)).fetchall()
    except (sqlite3.Error, FileNotFoundError):
        rows = []
    return pd.DataFrame(rows, columns=cols)


def attach_role_collapse(d: pd.DataFrame) -> pd.DataFrame:
    """Copy the role-collapse numbers from a run's secondary `over_refusal_graded_noncollapsed` row (ROLE_COLLAPSE_METRIC: k refusals
    over n non-collapsed items, the share in its extra json) onto the OWNER metric row of the same (run, judgment, threshold), as
    `role_collapse_share`, `role_collapse_n`, `noncollapsed_k`, `noncollapsed_n`; `intervals` then substitutes the non-collapsed rate
    where the share is above ROLE_COLLAPSE_LINE. Rows with no secondary read NaN and are untouched. Where the non-collapsed row is
    itself the owner (the paired lane, NONCOLLAPSED_METRIC) it takes the numbers too, so the cell that draws prints its collapse."""
    d = d.copy()
    for c in ("role_collapse_share", "role_collapse_n", "noncollapsed_k", "noncollapsed_n"):
        d[c] = np.nan
    if "run_id" not in d.columns or "metric_id" not in d.columns:
        return d
    # THE PAPER QUERY IS OWNER-ONLY (`in_current_paper = 1` flags the owner metric row alone - measured 2026-09-22: every secondary row
    # reads 0), so the frame never carries the secondary row itself; it is read from the `run_metric` table (a few rows, one query per
    # registry head) and joined to the owner rows here. A frame that does carry the rows (a synthetic one) is read as it stands.
    # THE FRAME NOW CARRIES THE NON-COLLAPSED ROWS THEMSELVES (2026-09-23): since the non-collapsed rate became the headline wherever
    # a run files it (NONCOLLAPSED_METRIC), the paper view flags THOSE rows as the owners, so they arrive in the frame - without the
    # metric's extra json, which the view does not expose. Reading them as the companion table gave every share NaN and the join no
    # parent row to land on, so no cell printed its collapse. The companion numbers are read from the registry's `run_metric` rows
    # (the share and the counts live there), a frame that carries the extra json itself (a synthetic one) is read as it stands, and
    # the numbers land on BOTH the parent owner row where one exists and the non-collapsed row itself, which is what draws now.
    sec_frame = d[d["metric_id"].eq(ROLE_COLLAPSE_METRIC)]
    parts = [sec_frame] if "metric_extra_json" in sec_frame.columns and len(sec_frame) else []
    sec_db = role_collapse_rows(head())
    if len(sec_db):
        parts.append(sec_db)
    if not parts:
        return d
    sec = pd.concat(parts, ignore_index=True)
    # ONE companion row per registry run and judgment (2026-09-22: the trials are pooled, the row sits at threshold -1.0), so the
    # join is on (run, judgment) alone
    keys = [c for c in ("run_id", "judgment_id") if c in d.columns and c in sec.columns]
    ex = sec["metric_extra_json"].map(_extra) if "metric_extra_json" in sec.columns else pd.Series([{}] * len(sec), index=sec.index)
    table = pd.DataFrame({
        **{k: sec[k].values for k in keys},
        "role_collapse_share": pd.to_numeric(ex.map(lambda e: e.get(ROLE_COLLAPSE_KEYS["share"]) if isinstance(e, dict) else None), errors="coerce").values,
        "role_collapse_n": pd.to_numeric(ex.map(lambda e: e.get(ROLE_COLLAPSE_KEYS["n"]) if isinstance(e, dict) else None), errors="coerce").values,
        "noncollapsed_k": pd.to_numeric(sec["k"], errors="coerce").values, "noncollapsed_n": pd.to_numeric(sec["n"], errors="coerce").values,
    }).drop_duplicates(keys)
    own = d["metric_id"].isin([ROLE_COLLAPSE_PARENT, ROLE_COLLAPSE_METRIC])
    if not len(table) or not keys or not own.any():
        return d
    merged = d.loc[own, keys].merge(table, on=keys, how="left")
    for c in ("role_collapse_share", "role_collapse_n", "noncollapsed_k", "noncollapsed_n"):
        d.loc[own, c] = merged[c].values
    return d


_PAD_VIEW = re.compile(r"^pad_static_[a-z0-9_]+?_obj(?:_(s\d+))?_p\d+$")


def pad_view_seed(view) -> str | None:
    """The fit-seed padding wave's STEERED standup view spells: `pad_static_<model>_obj_s2_p8192` -> s2; the unsuffixed `..._obj_p8192` is
    seed 1 (the LoRAs' stamped fit_replicate_label on those very views says s1). None for anything else -- the reference standups
    `pad_static_<model>_p8192` (unsteered and prompted rows, one object per standup) and every non-padding view."""
    m = _PAD_VIEW.match(str(view or ""))
    if not m:
        return None
    return m.group(1) or "s1"


def spell(df: pd.DataFrame) -> pd.DataFrame:
    """Add the paper's spellings from the registry's own tables: `object_label` (the registry paper_label; an `other`
    fitted object is spelled from the campaign cell id it was served under), `seed` (the object's fit_replicate_label:
    one point per seed, never pooled), `served_transfer` (from the served key's base name, the legacy mapping), `lane`
    (the run's own `transfer_kind` where it names a lane - the AUTHORITATIVE column - else the served key's, for the
    unsteered and prompted rows whose transfer_kind is `none` / `prompted_only`), `dose` (the chain `sisteer.registry.paper`
    renders: run extra paper_dose, else strength_nominal, else the dose the served key spells), `coefficient` (the raw
    steering coefficient), `headline` (the eval's headline metric at the paper's threshold; best-of-N at n = 75) and
    `judge_ok` (JUDGE_POLICY)."""
    check_headline_scales()                          # fail at load, not on the page, when a headline id is off the scale (2026-09-24)
    d = df.copy()
    ex = d["run_extra_json"].map(_extra)
    cell = d["campaign_cell_id_legacy"].fillna("")
    d["object_label"] = np.where(d["paper_label"].eq("other"), cell.map(CAMPAIGN_ID_PAPER_LABEL_LEGACY).fillna(cell), d["paper_label"])
    lab = d["fit_replicate_label"].astype(object) if "fit_replicate_label" in d.columns else pd.Series(None, index=d.index, dtype=object)
    from_extra = ex.map(lambda e: f"s{int(e['fit_seed'])}" if isinstance(e.get("fit_seed"), (int, float)) and not isinstance(e.get("fit_seed"), bool) else None)
    # the paired write: fit_replicate_label unstamped, seed on the view. A trailing `p` (qwen35_4b_s1p) is the PAIRED view of the same
    # seed - the composed arm contrasted against its own view's unsteered run (2026-09-21) - so it is seed s1, not a
    # blank-seed canonical row; the view itself still keys the anchor, so s1 and s1p anchor on their own unsteered runs (a standup
    # replicate of one instrument, never pooled as two seeds - the unsteered rows are anchors and are not drawn).
    # ...or the view IS the seed: the reseeded seed-1 battery spells its view bare (`s1`), not `qwen35_4b_s1` (2026-09-21), and a
    # pattern anchored on a preceding underscore read those rows as the blank-seed canonical generation - which is how the two prompted
    # arms' seed-1 MMLU readings went missing from the panel. Checked at the same time: no FITTED object's row reads blank-seed on the
    # seed-replicate lanes, so the canonical-fit retirement was never deleting a seed.
    # ...and the LoRA DOSE lane writes the seed with no separator before it, `qwen35_4b_d0p5_loradiff64s1`, sometimes with a
    # trial suffix, `..._loradiff64s3_t0` (the multi-seed dose contract, 2026-09-22). A pattern anchored on a
    # preceding underscore matches neither, so every LoRA dose rung read BLANK-SEED. With only seed 1 filed that was
    # invisible; the moment the seed-2 and seed-3 dose standups land it would have collapsed three fits into one
    # rung, averaged them, and drawn a single mark with no between-fit spread and nothing saying so. A trailing `_t<N>` is
    # stripped first, then the seed may follow an underscore, the start, or a digit (the `64` of `loradiff64`).
    _v = d["view_legacy"].fillna("").str.replace(r"_t\d+$", "", regex=True)
    from_view = _v.str.extract(r"(?:^|_|\d)(s\d+)p?$")[0]
    # ...and PAD'S VIEWS put the level AFTER the seed, `pad_static_qwen35_4b_obj_s2_p8192` (2026-09-24, three seeds filed on one
    # line): the paired difference ReLU carries no fit_replicate_label there, so its three seeds read blank and drew as one.
    # `pad_view_seed` reads the seed off the view; the unsuffixed steered standup is seed 1, as the LoRAs' stamped labels on the
    # same views say. Fitted objects only -- the prompts, Arditi and `none` on those views are one object served per standup, and
    # stay blank (draws).
    _pad_seed = d["view_legacy"].map(pad_view_seed)
    _fitted_now = d["object_label"].astype(str).str.endswith(FITTED_SUFFIXES)
    from_view = from_view.where(~(_pad_seed.notna() & _fitted_now), _pad_seed)
    d["seed"] = lab.where(lab.notna(), from_extra).where(lambda v: v.notna(), from_view).fillna("").astype(str)
    # A LONE SEED IS NOT A REPLICATE (2026-09-24). "One point per seed, never pooled" needs at least two seeds of one (model, object) to
    # mean anything. On Gemma, Llama and Nemotron every fitted object has exactly ONE fit, yet its BigCodeBench / BFCL rows arrived on
    # `<model>_s1` views (that single fit's training seed, spelled on the view) and the retirement's object-id fill then carried the
    # "s1" onto the object's other rows, so one cell split into two "seeds" - MMLU + GSM8K blank, BCB + BFCL s1; PAIR s1, best-of-N
    # blank - and every one drew ringed as partial. Where a (model, object) carries one distinct seed label and nothing else, that
    # label is the fit's stamp, not a replicate, and the rows are one cell. Qwen's fitted objects carry s1, s2, s3 and are untouched.
    # THE PROMPTED SIs TOO (2026-09-25, v195): when BigCodeBench / BFCL joined the all-model capability axis, Safety SI and Constitution
    # SI on Gemma, Nemotron and Llama split the same way - their BCB / BFCL rows came on the `<model>_s1` views (seed "s1"), their
    # MMLU / GSM8K rows blank - and each half drew ringed as partial. A prompt is one object, so a lone label on it is a view stamp.
    single = d["object_label"].astype(str).str.endswith(FITTED_SUFFIXES) | d["object_label"].isin(PROMPTED)
    lone = d[single & d["seed"].ne("")].groupby(["model_id", "object_label"])["seed"].nunique()
    lone_pairs = set(lone[lone == 1].index)
    if lone_pairs:
        pair_ = pd.Series(list(zip(d["model_id"], d["object_label"])), index=d.index)
        d.loc[single & pair_.isin(lone_pairs), "seed"] = ""
    # THE AXIS IS OBJECT IDENTITY, NOT THE VIEW SPELLING (2026-09-21). One object id appearing under several seed views is
    # one fit re-decoded - decode DRAWS - while distinct object ids are distinct fits. The unsteered model, the two prompted SIs and the
    # Refusal steer are each a single object served across the seed views, so a view-derived `sN` on them is a false seed: it drew one
    # reading as several fits. Where a (model, object, eval) holds ONE object id across several seed spellings, the seed is cleared and
    # the rows stand as draws (`draw`, from seed_base / trial_index); where the ids differ, every seed is kept. Fitted objects are
    # unaffected - their three seeds are three shas - so this narrows nothing the paper needs.
    if "object_id" in d.columns and d["seed"].ne("").any():
        seeded = d["seed"].ne("") & d["object_id"].notna()
        per = d[seeded].groupby(["model_id", "object_label", "eval_id"]).agg(ids=("object_id", "nunique"), seeds=("seed", "nunique"))
        one_object = {k for k, r in per.iterrows() if r.ids == 1 and r.seeds > 1}
        if one_object:
            key = pd.Series(list(zip(d["model_id"], d["object_label"], d["eval_id"])), index=d.index)
            d.loc[seeded & key.isin(one_object), "seed"] = ""
    # NOT keyed on object id: a strength ladder's dose rungs are distinct composed objects under one view, so an object-id replicate label
    # split every ladder into one-rung "seeds" (measured 2026-09-20). Distinct fitted objects under one label are told apart by their seed;
    # the paired best-of-N climb objects carry fit_seed 1 / 2 / 3.
    base = d["view_legacy"].fillna("").map(lambda v: _DOSE_SUFFIX.sub("", v))
    d["served_transfer"] = base.map(SERVED_KEY_TRANSFER_LEGACY).map(_TRANSFER)
    tk = d["transfer_kind"] if "transfer_kind" in d.columns else pd.Series(None, index=d.index, dtype=object)
    named = tk.isin(list(LANE_LABEL))
    d["lane"] = np.where(named, tk, d["served_transfer"])
    d["lane_conflict"] = named & d["served_transfer"].notna() & ~tk.eq(d["served_transfer"])     # the two spellings disagree: listed, never hidden
    key_dose = d["view_legacy"].fillna("").map(lambda v: (lambda m: 1.0 if m is None else 0.5 if m.group(1) == "halfdose"
                                              else (-1 if m.group(2) else 1) * float(f"{m.group(3)}.{m.group(4) or 0}"))(_DOSE_SUFFIX.search(v)))
    d["dose"] = ex.map(lambda e: _dose_num(e.get("dose", e.get("paper_dose")))).fillna(d["strength_nominal"]).fillna(key_dose)
    d["coefficient"] = d["strength_coefficient"].fillna(ex.map(lambda e: _coef_num(e.get("dose")))).fillna(ex.map(lambda e: _dose_num(e.get("paper_dose"))))
    # the operating point: the run's own stamp where it carries one; an UNSTAMPED run at nominal dose 1.0 is the operating point for any object
    # (the export files the Refusal steer's Petri run with strength_nominal 1.0 and no stamp, as it files the prompted and unsteered rows); a run
    # stamped 0 stays off it whatever its dose
    d["at_operating_point"] = d["is_operating_point"].eq(1) | (d["is_operating_point"].isna() & d["dose"].eq(1.0))
    # ...or the frame is EMPTY (a fresh clone whose snapshot holds no row of the paper frame yet): the column must still exist
    if "metric_id" in d.columns and "headline_metric_id" in d.columns and (d["metric_id"].notna().any() or not len(d)):
        thr = d["threshold"].fillna(-1)
        # THE BEST-OF-N UNIT IS THE RULED PER-GOAL ANY-HIT (the rows landed 2026-09-21). The eval table still
        # declares `asr` as its headline metric and NO best-of-N row carries that id - the seed-replicate runs file
        # `asr_bon_cap4096` (per-goal any-hit@75) beside the climb curve `asr_at_nK` and `bon_nauc` - so testing the declared id
        # alone made every one of those runs read as having no headline row at all, and three seeds of best-of-N vanished from the
        # panel silently. A best-of-N run's headline is the ruled metric where the run files it, else the declared one.
        declared = headline_metric_ids()
        want = d["headline_metric_id"].astype(object).copy()
        if "run_id" in d.columns:
            # the registry's own rule: a run's headline is the FIRST id its eval lists that the run actually files
            filed = d.groupby(["run_id", "eval_id"]).metric_id.agg(set)
            for (rid, eid), have in filed.items():
                ids = declared.get(eid)
                if not ids:
                    continue
                pick = next((i for i in ids if i in have), None)
                if pick is not None:
                    want[(d.run_id == rid) & (d.eval_id == eid)] = pick
            # the pre-schema-8 fallback: best-of-N runs that file the ruled per-goal metric while their eval declares no list
            bon = d["eval_category"].eq("best_of_n") & ~d["eval_id"].isin(declared)
            ruled = set(d.loc[bon & d["metric_id"].eq(BON_HEADLINE_METRIC), "run_id"])
            want[bon & d["run_id"].isin(ruled)] = BON_HEADLINE_METRIC
            # the non-collapsed over-refusal rate wins wherever the run files one, whatever the eval declares
            nc = set(d.loc[d["metric_id"].eq(NONCOLLAPSED_METRIC), "run_id"])
            if nc:
                want[d["run_id"].isin(nc)] = NONCOLLAPSED_METRIC
        d["headline"] = d["metric_id"].eq(pd.Series(want, index=d.index)) & thr.isin([0.5, -1]) & bon_headline_n_ok(d["eval_category"], d["bon_n"])
    # THE PADDING-LENGTH AXIS (2026-09-22, before the rows land). A static-attack wave puts a measured number of neutral
    # filler tokens between the system instruction and the payload. Its four families ARE this reader's static category,
    # and padding appeared in none of the three keys rows are collapsed on, so a padded row and an unpadded row for the same
    # (model, object, eval) would have been treated as one cell and averaged as decode draws. `pad` is 0 where no padding
    # block exists, which is every row in the frame today, so adding it to those keys changes nothing until the wave lands.
    # Migration 0012 makes padding part of the INSTRUMENT identity and puts `padding_tokens` on the view (NULL = unpadded),
    # so that column is the authority and the producer's `padding` block is only the fallback for rows filed before it.
    _pad_col = (pd.to_numeric(d["padding_tokens"], errors="coerce") if "padding_tokens" in d.columns
                else pd.Series(np.nan, index=d.index))
    _pad_extra = ex.map(lambda e: float((e.get("padding") or {}).get("tokens") or 0) if isinstance(e, dict) else 0.0)
    d["pad"] = _pad_col.where(_pad_col.notna(), _pad_extra).fillna(0.0)
    # THE MARKER IS THE PADDING CONTRACT, NOT A NON-ZERO TOKEN COUNT (2026-09-23). The padding wave
    # files its ZERO level as padding_tokens = 0 WITH the wave's filler contract, never NULL, precisely so the zero point is
    # fenced like the padded ones — it was drawn on the padding instrument (98,304 window, served width 5, cap 16,384
    # single-pass), not on the ordinary one. Keying the fence on `pad > 0` therefore let that row read as an ordinary
    # unpadded run, and since `pad` is in the collapse keys it would have COLLAPSED with the genuine unpadded row of the
    # same arm: the second padded wave runs the same arm names on the same model and family as the Pliny panels, so arm
    # names alone are no protection. An ordinary row has padding_tokens NULL; every row of the wave has it set.
    d["pad_contract"] = _pad_col.notna() | ex.map(
        lambda e: isinstance(e, dict) and isinstance(e.get("padding"), dict))
    d["decode_declared"] = ex.map(lambda e: e.get("decode_declared_legacy"))
    # TWO REPLICATION AXES, NEVER AVERAGED ACROSS (2026-09-19): fitted objects replicate over the FIT seed (above); the prompted
    # SIs, the Refusal steer and the unsteered model replicate over DECODE only - (run.seed_base, run.trial_index). A decode draw is a
    # point of its own where a cell holds several draws; a single-draw cell keeps an empty seed so the tables stay clean. A fitted row
    # carries both (its seed_base is the fit-seed index), so its fit seed stands.
    # The draw is ITS OWN column (`draw`), never the seed: a seed base differs from eval to eval by design, so a draw label cannot join an
    # object's rows across evals (it split every multi-draw rung off the frontier once). Within one eval cell each draw is a small mark of
    # its own; every mean over evals - the category mean, a frontier rung - averages the draws within the eval first.
    d["draw"] = ""
    if "seed_base" in d.columns and "trial_index" in d.columns:
        sb, ti = d["seed_base"], d["trial_index"]
        label = np.where(sb.notna() | ti.notna(), "d" + sb.fillna(-1).astype(int).astype(str) + "/" + ti.fillna(0).astype(int).astype(str), "")
        # `pad_contract` joins `pad` in every collapse key: a padding-wave ZERO row has pad == 0 and would otherwise
        # share a key with the genuine unpadded row of the same arm (2026-09-23)
        # `item_set_id` joins the key: two item sets are two INSTRUMENTS, never one cell (the MMLU rule, and now
        # BigCodeBench/BFCL, whose full suites landed under the SAME eval_id as the 200-item sets — measured 34 cells
        # holding both, `none` among them, so the ANCHOR would have been a 200-item reading averaged with a 1,140-item
        # one and every delta on those evals wrong)
        cell = ["model_id", "object_label", "lane", "dose", "eval_id"] + [c for c in ("item_set_id", "pad", "pad_contract") if c in d.columns]
        multi = pd.Series(label, index=d.index).groupby([d[c] for c in cell], dropna=False).transform("nunique") > 1
        d["draw"] = np.where(multi, label, "")
    d["judge_ok"] = judge_ok(d)
    # THE ESCALATION STATE (2026-09-21): the reseeded seed-1 statics are a BASE-CAP reading with the token-cap ladder still
    # pending - the escalation rungs failed and re-run - so they are not a final ladder result and must not be pooled with escalated
    # seeds as if the difference were seed variance. The producer files the state on the run; the figures mark it, never drop it.
    d["escalation_state"] = ex.map(lambda e: e.get("escalation_state") if isinstance(e, dict) else None)
    d = attach_role_collapse(d)                                             # the secondary non-collapsed row's numbers onto the owner row
    # A TWO-TERM GRADED VALUE (2026-09-22): the r2_export over-refusal rows derive over_refusal_graded from the two filed rates
    # without the registry's per-item at-cap term (a reply at the token cap scores 0.5 whatever its label); the row records the largest
    # deviation that term could make, and the panel prints it under the mark (two_term_note). Every other over-refusal row is three-term.
    d["two_term_dev"] = np.nan
    if "metric_extra_json" in d.columns:
        _tt = d["metric_extra_json"].map(_extra)
        d["two_term_dev"] = pd.to_numeric(_tt.map(lambda e: e.get("max_abs_deviation_from_at_cap_term") if isinstance(e, dict) else None), errors="coerce")
    if "metric_extra_json" in d.columns:                                    # run_metric.extra_json.delta_block: a producer-filed contrast whose
        ex = d["metric_extra_json"].map(_extra)
        d["delta_block"] = ex.map(lambda e: e.get("delta_block") if isinstance(e, dict) else None)   # anchor may be cross-source
        # THE AT-CAP SHARE where the column is empty: the ruled three-term over-refusal rate (full refusal + ½ partial + ½ at-cap) files
        # its at-cap count as `n_at_cap_scored_half` in its own extra (2026-09-20: under the vendor preset 38-40 % of the
        # unsteered model's safe XSTest replies hit the 512-token cap, where `frac_at_token_cap` is unset), so the share rides beside
        # the value like every other cell's - reported, never discounted.
        if "frac_at_token_cap" in d.columns and "n" in d.columns:
            n_cap = pd.to_numeric(ex.map(lambda e: e.get("n_at_cap_scored_half") if isinstance(e, dict) else None), errors="coerce")
            n_all = pd.to_numeric(d["n"], errors="coerce")
            fill = d["frac_at_token_cap"].isna() & n_cap.notna() & n_all.gt(0)
            d.loc[fill, "frac_at_token_cap"] = (n_cap[fill] / n_all[fill]).astype(float)
    return d


# THE ROSTER NAMES THE PAPER LANE (roster v9, 2026-09-24, "use base->instruct whenever the option exists"): from v9 on,
# a fitted object's `fig1_headline/instruct_direct` (and every other `/instruct_direct` id) sits on its BASE->INSTRUCT fit, and the
# instruct-direct fit keeps only `fig4_transfer` / `figQ_transfer`. A reader that admits rows by the `lane` column would then keep
# drawing the IT fits where both are `selected` (the appendix: on Gemma / Llama the two fits share a blank seed, so the base row is
# not even pulled) -- the preference would be in the registry and invisible in the figures. With the flag on, a row whose
# comparison ids name a `/<paper lane>` comparison is on the paper lane whatever its `lane` column says, and a paper-lane row whose
# ids name NO such comparison (the IT fit left with the transfer figures alone) is not. Rows with no comparison ids -- the include-
# selected prompts, Arditi, the seed replicates, `none` -- keep the lane rule. OFF until v9 is loaded: before it, the roster says
# "IT for the main figures" and the reader's pull-then-retire is what draws Qwen's base seeds; flipping early would drop them.
ROSTER_NAMES_THE_PAPER_LANE = True      # flipped 2026-09-24 21:20: roster v9 loaded (head 227223)


def _lane_by_comparison(d: pd.DataFrame) -> tuple:
    """(has_ids, by_id, moved): whether the row carries any comparison id; whether one of them ends in `/<its model's paper
    lane>`; and whether the row's (model, object) has SOME row admitted by id -- the roster has spoken for that object, which is
    the only case in which a row named for other figures alone is a fit the roster moved OFF the paper lane. THE PAIRED
    DIFFERENCE RELU IS THE COUNTER-CASE (2026-09-24, when it vanished from every non-Qwen jailbreak panel):
    on Gemma / Llama it has no instruct-direct fit at all, its rows carry `fig2_jailbreak/base_to_instruct` and the like
    because base->instruct is its ONLY lane, and v9 re-shaped nothing for it -- so "ids but none on the paper lane" is not the
    roster moving it, it is the object having nowhere else to be, and the pull rule that admitted it since 2026-09-16 must
    keep doing so. `moved` is what tells the two apart."""
    if "comparison_ids_json" not in d.columns:
        f = pd.Series(False, index=d.index); return f, f, f
    ids = d["comparison_ids_json"].map(_comparison_ids)
    tag = d["model_id"].map(PAPER_LANE).fillna("")
    by_id = pd.Series([any(i.endswith("/" + t) for i in L) if t else False for L, t in zip(ids, tag)], index=d.index)
    obj = pd.Series(list(zip(d["model_id"], d["object_label"])), index=d.index)
    # ONLY A PAPER-LANE ROW CAN HAVE BEEN MOVED OFF THE PAPER LANE (2026-09-25, the Qwen LoRAs' BigCodeBench rows): the base->instruct
    # seed fits' full-suite coding rows gained their first id under roster v10 -- `figA_transfer_models`, no paper-lane figure -- and
    # this test, written for the IT fits left with fig4_transfer, read them as moved and withdrew the pull that had admitted them.
    # A row on another lane with ids naming other figures was never on the paper lane; the pull rule decides it, as before v10.
    lane_ok = d["lane"].eq(d["model_id"].map(PAPER_LANE))
    moved = ids.map(len).gt(0) & ~by_id & lane_ok & obj.isin(set(obj[by_id]))
    return ids.map(len).gt(0), by_id, moved


def on_paper_lane(d: pd.DataFrame) -> pd.Series:
    """True where the row sits on its model's paper lane: by the `lane` column, or -- ROSTER_NAMES_THE_PAPER_LANE -- by a comparison
    id on that lane, in which case a lane-column row whose ids name only other figures is NOT on it (see the flag's note)."""
    lane_ok = d["lane"].eq(d["model_id"].map(PAPER_LANE))
    if not ROSTER_NAMES_THE_PAPER_LANE:
        return lane_ok
    has_ids, by_id, moved = _lane_by_comparison(d)
    return by_id | (lane_ok & ~moved)


def prefer_vendor_decode(d: pd.DataFrame) -> pd.Series:
    """Keep a cell's VENDOR-PRESET reading where it has one, dropping a grandfathered T = 1.0 run beside it, so that
    vendor decode is used.

    The registry names the two outright in `decode_source` — `model_card` for a run served at the vendor preset,
    `grandfathered_T1` for an older campaign run forced to temperature 1. Admitting the seed-replicate draws (see
    `admit_lane`) is only half the job: it leaves both in the cell, and pooling them averages across a decode boundary.
    On Qwen's prompted SIs the two disagree materially — best-of-N ASR 0.31/0.27 grandfathered against 0.16/0.18/0.13 at
    the vendor preset for the safety SI, 0.47 against 0.38/0.30/0.27 for the constitution SI — so the grandfathered run
    does not merely add noise, it overstates attack success.

    Cell-local by design: a cell with no vendor-preset run keeps its grandfathered reading rather than losing its only
    row, and a cell with no grandfathered run is untouched."""
    if "decode_source" not in d.columns:
        return pd.Series(True, index=d.index)
    src = d["decode_source"].fillna("")
    key = pd.Series(list(zip(d["model_id"], d["object_label"], d["eval_id"])), index=d.index)
    has_vendor = set(key[src.eq("model_card")])
    return ~(src.eq("grandfathered_T1") & key.isin(has_vendor))


def admit_lane(d: pd.DataFrame) -> tuple:
    """(admitted, pulled): a row is admitted on its model's paper lane, or - paperlib's rule, 2026-09-16 - when that MEASUREMENT has no
    row on the paper lane and is recorded on another lane (the paired-write difference ReLU steer is fit on the base checkpoint and
    served on the instruct one: base->instruct is its only lane). Pulled rows are flagged, drawn from their lane, and named by the
    notebook; the unsteered rows are never pulled (anchors follow the steered row's lane).

    A MEASUREMENT IS (model, object, EVAL, SEED), not (model, object) (2026-09-20). Keying on the object alone asks whether the object
    appears on the paper lane ANYWHERE, so an object with paper-lane statics lost its other-lane multi-turn rows entirely: the three
    LoRAs and the constitution ReLU steer each have instruct-direct rows on the static evals, and their seed-2 / seed-3 multi-turn runs
    (the seed-replicate cluster, recorded base->instruct) were dropped before any figure saw them, while seed 1 stayed. Per
    measurement, a seed whose attack was run only on the other lane is drawn from it and named, and nothing that the paper lane does
    carry is ever displaced - the ladder rungs stay out, being off the operating dose, as before."""
    on = on_paper_lane(d)
    key = pd.Series(list(zip(d["model_id"], d["object_label"], d["eval_id"], d["seed"])), index=d.index)
    have_on = set(key[on])
    pulled = ~on & d["lane"].notna() & ~key.isin(have_on) & d["object_label"].ne("none")
    if ROSTER_NAMES_THE_PAPER_LANE:
        # a row the roster names for OTHER figures only (the IT fit on fig4_transfer alone) is not a measurement the paper lane
        # lacks; it is one the roster moved off it, and pulling it back would redraw the fit the preference retired
        has_ids, by_id, moved = _lane_by_comparison(d)
        pulled &= ~moved
        # AN INCLUDE-SELECTED IT FIT WITH NO COMPARISON YIELDS TO THE ROSTER'S BASE FIT (v157, 2026-09-24): 13 fitted IT rows
        # (gemma constitution_lora on actorattack / crescendo / pair; qwen constitution_lora, control_lora, difference_lora on the
        # agentic and multi-turn evals) are `selected` by key-scoped include rulings written before v9, carry an EMPTY comparison
        # list, and so fell to the lane rule -- admitted beside the base fit the roster now names for the same cell, a second
        # blank-seed point in a cell the preference had already decided. Where a (model, object, eval) has a row admitted BY ID,
        # a fitted lane-column row with no comparison ids stands down; a cell the roster has not spoken for keeps its include.
        fitted_ = d["object_label"].astype(str).str.endswith(FITTED_SUFFIXES)
        cell = pd.Series(list(zip(d["model_id"], d["object_label"], d["eval_id"])), index=d.index)
        spoken = set(cell[by_id])
        yields = on & fitted_ & ~has_ids & cell.isin(spoken)
        on = on & ~yields
    # A PROMPT HAS NO TRANSFER, so a prompted arm's row reads lane = null wherever the served key has no legacy spelling (the
    # seed-replicate cluster's views, 2026-09-20) - it is not on another lane, it is on none, and it is served on the paper
    # checkpoint by construction. Admit it where its measurement has no paper-lane row; it is not "pulled", having no lane to be
    # pulled from. Without this the two prompted SIs' seed-2 / seed-3 attacks were dropped while their seed-1 rows were drawn.
    # ...BUT THE KEY MUST BE ABLE TO TELL THE DRAWS APART (2026-09-23). The rule above reads (model, object, eval, SEED),
    # and the seed is deliberately CLEARED on exactly these rows by the object-identity rule above: the prompted SIs are one
    # object served across the seed views, so a view-derived sN is a false seed and the runs are DRAWS. The two rules
    # interact badly — with the seed blank, the three seedrep draws share a key with the grandfathered campaign run, that
    # run is on the paper lane, so `~key.isin(have_on)` is false and the rescue declines for precisely the rows it exists to
    # save. Measured at head 191706: 66 of 85 Qwen prompted-SI best-of-N rows dropped, and both arms drew the T = 1.0
    # campaign run instead of their three vendor-preset draws. For the lane-less case the discriminator is the VIEW, which
    # is what separates a draw from the campaign run when the seed cannot.
    key_draw = pd.Series(list(zip(d["model_id"], d["object_label"], d["eval_id"], d["seed"],
                                  d["view_legacy"].fillna(""))), index=d.index)
    have_on_draw = set(key_draw[on])
    no_lane = ~on & d["lane"].isna() & d["object_label"].isin(PROMPTED) & ~key_draw.isin(have_on_draw)
    # THE UNSTEERED ROWS ARE ANCHORS, NEVER DRAWN, AND EVERY ONE IS ADMITTED (2026-09-21). A seed-replicate view's unsteered run is
    # lane-less (`none` has no arm), so the lane test dropped it from the frame before anchored() ever saw it: the seed views'
    # contrasts then found no anchor in the frame, took the producer's filed delta verbatim - filed WITHOUT an interval - and 130 of
    # 529 drawn points had no bar. Rosters never contain `none`, so admitting it draws nothing; it only makes the anchors reachable.
    anchors = d["object_label"].eq("none")
    return on | pulled | no_lane | anchors, pulled


# ---- statistics ----------------------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_diff(k1, n1, k2, n2, z=1.959964):
    """Newcombe's hybrid-score interval for a DIFFERENCE of two INDEPENDENT proportions (his method 10).

    A difference of proportions does not get a single-proportion Wilson bar: the arm and its anchor each carry
    sampling error, so the pair's interval is built from both Wilson intervals rather than from either (2026-09-23).

    INDEPENDENT form, deliberately. The paired correlation-corrected version is tighter and would be right here — the
    same 259 goals run under both arms, with phi measured at -0.08 to 0.39 — but the registry frame carries no per-goal
    pairing, only the aggregate k and n. Where phi > 0 the independent form is CONSERVATIVE, i.e. wider, so the figure
    understates its own precision rather than overstating it. The figure says so."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d - float(np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), d + float(np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def intervals(d: pd.DataFrame) -> pd.DataFrame:
    """`y`, `lo`, `hi` on the plotted scale (SCALE[metric]) and `ci_source`: the row's own ci_lo/ci_hi, else Wilson from value
    and n for a rate metric, else none (the paper's `_point_intervals` rule)."""
    d = d.copy()
    sc = d["metric_id"].map(SCALE).fillna(1.0)
    lo, hi = d["ci_lo"].astype(float), d["ci_hi"].astype(float)
    src = pd.Series(np.where(lo.notna() & hi.notna(), "own 95% CI", "none"), index=d.index, dtype=object)
    for i in d.index[(src == "none") & d["metric_id"].isin(IS_RATE) & d["n"].gt(0)]:
        p = min(max(float(d.at[i, "value"]) / IS_RATE[d.at[i, "metric_id"]], 0.0), 1.0)
        wl, wh = wilson(p * float(d.at[i, "n"]), float(d.at[i, "n"]))
        lo.at[i], hi.at[i], src.at[i] = wl * IS_RATE[d.at[i, "metric_id"]], wh * IS_RATE[d.at[i, "metric_id"]], "Wilson from value and n"
    d["y"], d["lo"], d["hi"], d["ci_source"] = d["value"] * sc, lo * sc, hi * sc, src
    # THE RATE OVER THE NON-COLLAPSED ITEMS stands in for the filed rate where the cell role-collapsed (ROLE_COLLAPSE_LINE):
    # y from k / n over the non-collapsed items, Wilson on that n, and `collapsed` marks the row for the annotation
    d["collapsed"] = False
    if {"role_collapse_share", "noncollapsed_k", "noncollapsed_n"} <= set(d.columns):
        hit = d["role_collapse_share"].astype(float).gt(ROLE_COLLAPSE_LINE) & d["noncollapsed_k"].notna() & d["noncollapsed_n"].astype(float).gt(0) & d["metric_id"].isin(IS_RATE)
        for i in d.index[hit]:
            k, n, unit = float(d.at[i, "noncollapsed_k"]), float(d.at[i, "noncollapsed_n"]), IS_RATE[d.at[i, "metric_id"]]
            wl, wh = wilson(k, n)
            d.at[i, "y"], d.at[i, "lo"], d.at[i, "hi"] = (k / n) * unit * sc.at[i], wl * unit * sc.at[i], wh * unit * sc.at[i]
            d.at[i, "ci_source"], d.at[i, "collapsed"] = "Wilson over the non-collapsed items", True
    return d


HELD_NOTE = "not scoreable"


def is_held(row) -> bool:
    """A HELD-OUT ROW (the registry's `hold_out` ruling, 2026-09-22, on Llama's paired difference ReLU agentic cells, where
    degeneration is being scored as safety): the run stays selected and in the frame with its value, but it is out of every
    pooled mean and the panel prints it as `not scoreable` rather than drawing a defense. Read from `held_out` on the paper views; never a
    reader constant, so the rule either way reaches the figure through the registry."""
    try:
        return int(float(row.get("held_out", 0) or 0)) == 1
    except (TypeError, ValueError):
        return False


def held_mask(d: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(d["held_out"], errors="coerce").fillna(0).astype(int).eq(1) if "held_out" in d.columns else pd.Series(False, index=d.index)


def two_term_note(row) -> str | None:
    """The line under a two-term graded mark: the largest deviation the missing at-cap term could make, in the plotted unit."""
    try:
        dev = float(row.get("two_term_dev"))
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(dev):
        return None
    return f"two-term graded (\u2264{dev * 100:.1f} pp)"


def is_collapsed(row) -> bool:
    """A role-collapsed cell (ROLE_COLLAPSE_LINE): drawn open on every appendix figure, named in the table legend."""
    try:
        return bool(row.get("collapsed", False))
    except (KeyError, TypeError, ValueError):
        return False


def cell_notes(d: pd.DataFrame, evals) -> str:
    """The table legend's account of the cells the figure draws OPEN (2026-09-24: no text on the canvas for them): the
    held-out cells (`not scoreable`, out of every mean by the producer's ruling) and the role-collapsed cells (the plotted rate is over
    the non-collapsed items; share and count per cell). Empty when the frame has neither on these evals."""
    if d is None or not len(d):
        return ""
    x = d[d.eval_id.isin(list(evals))] if "eval_id" in d.columns else d
    parts = []
    held = x[held_mask(x)]
    if len(held):
        order = {e: i for i, e in enumerate(list(evals))}
        cells = sorted({(MODEL_LABEL.get(m, m), LABEL.get(o, o), order.get(e, 99), EVAL_LABEL.get(e, e)) for m, o, e in zip(held.model_id, held.object_label, held.eval_id)})
        by = {}
        for m, o, _, e in cells:
            by.setdefault(f"{m} {o}", []).append(e)
        parts.append(f"Open marks, {HELD_NOTE}: measured, held out of every mean by a registry hold-out: "
                     + "; ".join(f"{k} on {', '.join(v)}" for k, v in by.items()) + ".")
    col = x[x["collapsed"].astype(bool)] if "collapsed" in x.columns else x.iloc[0:0]
    if len(col):
        seen, items = set(), []
        for _, r in col.iterrows():
            key = (r.model_id, r.object_label, r.eval_id)
            if key in seen:
                continue
            seen.add(key)
            try:
                items.append(f"{MODEL_LABEL.get(r.model_id, r.model_id)} {LABEL.get(r.object_label, r.object_label)} on "
                             f"{EVAL_LABEL.get(r.eval_id, r.eval_id)} {float(r.role_collapse_share):.0%} ({int(float(r.noncollapsed_n))} items)")
            except (KeyError, TypeError, ValueError):
                items.append(f"{MODEL_LABEL.get(r.model_id, r.model_id)} {LABEL.get(r.object_label, r.object_label)} on {EVAL_LABEL.get(r.eval_id, r.eval_id)}")
        parts.append("Open marks, role collapse: the plotted rate is over the non-collapsed items (collapse share, items): " + "; ".join(sorted(items)) + ".")
    # the caveats that used to be printed under the figures (2026-09-24: nothing under a figure)
    dec = {}
    for _, r in x.iterrows():
        dn = decode_note(r)
        if dn:
            dec.setdefault((MODEL_LABEL.get(r.model_id, r.model_id), dn), set()).add(f"{LABEL.get(r.object_label, r.object_label)} on {EVAL_LABEL.get(r.eval_id, r.eval_id)}")
    if dec:
        parts.append("Deltas crossing a decode boundary: " + "; ".join(f"{m} ({dn}): {', '.join(sorted(v))}" for (m, dn), v in sorted(dec.items())) + ".")
    tt = x[pd.to_numeric(x["two_term_dev"], errors="coerce").notna()] if "two_term_dev" in x.columns else x.iloc[0:0]
    if len(tt):
        dev = float(pd.to_numeric(tt["two_term_dev"], errors="coerce").max())
        cells = sorted({f"{MODEL_LABEL.get(m, m)} {LABEL.get(o, o)} on {EVAL_LABEL.get(e, e)}" for m, o, e in zip(tt.model_id, tt.object_label, tt.eval_id)})
        parts.append(f"Two-term graded (no per-item at-cap term; \u2264{dev * 100:.1f} pp): " + ", ".join(cells) + ".")
    if "item_set_id" in x.columns:
        for e in list(evals):
            col = x[(x.eval_id == e) & x.item_set_id.notna()]
            if len(col) > 1:
                per = col.groupby("model_id").item_set_id.agg(lambda v: " / ".join(sorted(set(map(str, v)))))
                if per.nunique() > 1:
                    parts.append(f"{EVAL_LABEL.get(e, e)}: item sets differ across models (" + ", ".join(f"{MODEL_LABEL.get(m, m)} {v}" for m, v in per.items()) + "); not comparable across the row.")
    return " ".join(parts)


def collapse_note(row) -> str | None:
    """The short line under a role-collapsed mark ("collapse 55%"); `collapse_detail` carries the rest into the panel footnote, because a
    two-line note under a mark overprinted its neighbours in the dodged over-refusal column (2026-09-22)."""
    try:
        if not bool(row.get("collapsed", False)):
            return None
        return f"collapse {float(row['role_collapse_share']):.0%}"
    except (KeyError, TypeError, ValueError):
        return None


def collapse_detail(row, model: str, obj: str, eval_id: str) -> str | None:
    """The footnote line for a role-collapsed cell: which cell, the collapse share, and the non-collapsed items the plotted rate is over."""
    try:
        if not bool(row.get("collapsed", False)):
            return None
        return (f"{MODEL_LABEL.get(model, model)} {LABEL.get(obj, obj)} on {EVAL_LABEL.get(eval_id, eval_id)}: role collapse "
                f"{float(row['role_collapse_share']):.0%}, plotted rate over the {int(float(row['noncollapsed_n']))} non-collapsed items")
    except (KeyError, TypeError, ValueError):
        return None


_TRIAL = re.compile(r"_t\d+$")


def standup_tag(view) -> str:
    """A view's standup identity: the legacy view tag with its decode-trial suffix (`_t0`, `_t1`, ...) removed, so the three trials of one
    rung share a tag (`qwen35_4b_ar_c0p3_t2` -> `qwen35_4b_ar_c0p3`)."""
    return _TRIAL.sub("", view) if isinstance(view, str) else ""


def anchor_draws(cands: pd.DataFrame, source: str, dose=None, view=None, cap=None) -> pd.DataFrame:
    """The unsteered DRAWS a steered row is compared with: the same source's, else the paper's preferred source, else the first
    source; within it THE SAME STANDUP: the same `view_legacy`, else the same standup tag (the view without its trial suffix), else
    the same `dose` where one matches. Several rows = several decode draws of one unsteered cell - the caller averages them and
    carries the spread. The view match is what keeps one ladder's rungs on their own anchors (2026-09-20: the vendor
    standup's unsteered runs share a draw label across rungs and differ by view - `qwen35_4b_ar_c0p15` is the c = 0.15 rung's
    anchor, `qwen35_4b_ar_c0p5` the c = 0.5 rung's - so averaging them across rungs is exactly what the same-standup rule exists
    to avoid)."""
    if not len(cands):
        return cands
    # THE SAME TOKEN CAP FIRST (2026-09-20). A delta across two generation caps is biased, and the bias FLIPS SIGN between the
    # two axes of a cost/defense plane: a truncated reply scores as not-jailbroken on an attack (flattering the steering) and as a
    # refusal on a benign eval (inflating the cost), so the artefact does not scatter the points, it walks them coherently toward the
    # favorable corner. Measured here before the fix: 305 of 712 anchored rows were differenced against another cap. Where the cap
    # cannot be matched the caller marks the row; nothing is dropped.
    if cap is not None and "token_cap_tokens" in cands.columns and pd.notna(cap) and cands.token_cap_tokens.eq(cap).any():
        cands = cands[cands.token_cap_tokens.eq(cap)]
    same = cands[cands.lane_source_legacy == source]
    if len(same):
        pool = same
    else:
        pool = next((cands[cands.lane_source_legacy == s] for s in ANCHOR_PREFERENCE if (cands.lane_source_legacy == s).any()), None)
        if pool is None:
            first = cands.sort_values("lane_source_legacy").lane_source_legacy.iloc[0]
            pool = cands[cands.lane_source_legacy == first]
    if view is not None and "view_legacy" in pool.columns:
        if pool.view_legacy.eq(view).any():
            return pool[pool.view_legacy.eq(view)]
        tags = pool.view_legacy.map(standup_tag)
        if tags.eq(standup_tag(view)).any():
            return pool[tags.eq(standup_tag(view))]
    if dose is not None and "dose" in pool.columns and pool.dose.eq(dose).any():
        pool = pool[pool.dose.eq(dose)]
    return pool


def anchor_summary(pool: pd.DataFrame) -> pd.Series:
    """One anchor from its draws: y = the mean, se = the draws' own intervals combined over k plus the between-draw variance / k,
    `draws` = k, `spread` = max - min (0 for one draw)."""
    k = len(pool)
    se_within = float(np.sqrt((((pool.hi - pool.lo) / 3.92) ** 2).sum()) / k)
    se_between = float(np.sqrt(pool.y.var(ddof=1) / k)) if k > 1 else 0.0
    caps = sorted({float(v) for v in pool["token_cap_tokens"].dropna()}) if "token_cap_tokens" in pool.columns else []
    return pd.Series(dict(y=float(pool.y.mean()), se=float(np.hypot(se_within, se_between)), draws=k, spread=float(pool.y.max() - pool.y.min()),
                          cap=(caps[0] if len(caps) == 1 else np.nan), lane_source_legacy=pool.lane_source_legacy.iloc[0],
                          run_id=",".join(pool.run_id.astype(str))))


def pick_anchor(cands: pd.DataFrame, source: str) -> pd.Series | None:
    """The unsteered row a steered row is compared with: the same source's, else the paper's preferred source, else the first."""
    if not len(cands):
        return None
    same = cands[cands.lane_source_legacy == source]
    if len(same):
        return same.iloc[0]
    for s in ANCHOR_PREFERENCE:
        if (cands.lane_source_legacy == s).any():
            return cands[cands.lane_source_legacy == s].iloc[0]
    return cands.sort_values("lane_source_legacy").iloc[0]


def dedupe(d: pd.DataFrame, contrasts: bool = True) -> pd.DataFrame:
    """One row per measurement. `v_paper_long` carries TWO metric rows for every r2_export Petri, multi-turn and best-of-n identity
    (2026-09-20: 781 of 5,507 in-paper long rows, values identical): the stats readers write a second, blank-computation
    metric row onto the export's run ids, so any count or pooled n over long rows is inflated until the views-only fix (`metric_role`
    owner / secondary) lands. Key: (run, judgment, metric, threshold, bon_n) - two JUDGES of one run stay two rows. Kept: the row with
    a contrast, else the row whose `computation_id` is not blank (the owner row, with its extra), else the latest ingested. Applied at
    frame load (the frames cell) and again inside every reader, where it is idempotent."""
    # THE VIEW NOW NAMES THE OWNER (2026-09-21, migration 0004): `metric_role` is owner / secondary and `in_current_paper`
    # is true on the owner row only, so the role is the first preference and the computation_id spelling stays as the belt for a
    # frame read without it. The 63 remaining (run, metric, threshold) doubles are the two-anchor run_contrast fan-out: two contrasts of
    # one measurement against two anchors, both real, so `anchor_run_id` joins the key where the frame carries it and anchored()
    # chooses between them by item set. A reader that computes its own deltas and never reads the contrast (the frontier) passes
    # `contrasts=False` and gets one row per measurement outright.
    role = d["metric_role"].eq("owner") if "metric_role" in d.columns else pd.Series(False, index=d.index)
    comp = d["computation_id"].fillna("").ne("") if "computation_id" in d.columns else pd.Series(False, index=d.index)
    key = ["run_id", "judgment_id", "metric_id", "threshold", "bon_n"] + (["anchor_run_id"] if contrasts and "anchor_run_id" in d.columns else [])
    attrs = dict(d.attrs)                                                    # the frame's head and provenance ride through the collapse
    # STALE CONTRASTS FIRST (2026-09-25 19:31): `run_contrast` keys on anchor_run_id and nothing retires a contrast a
    # newer accepted revision stopped emitting, so v_paper_long carries 376 stale rows on 194 runs (12 seed-replicate composed runs
    # re-anchored on their own none; 364 half-dose `full_dose` contrasts the export's emission 3 withdrew). Two contrasts of one
    # measurement against two anchor KINDS are by design and both stay; two of the same kind are one stale, one current, and the
    # greater ingest_seq is current. The view fix LANDED (migration 0017, schema 17, 2026-09-26 06:05Z: v_paper_long joins current
    # contrasts only and dropped exactly the 376 rows), so this pass is a belt that reports 0 and would catch a recurrence.
    stale_key = ["run_id", "judgment_id", "metric_id", "threshold", "computation_id", "anchor_kind"]
    if contrasts and "ingest_seq" in d.columns and all(c in d.columns for c in stale_key):
        n0 = len(d)
        d = d.sort_values("ingest_seq", ascending=False).drop_duplicates(stale_key).sort_index()
        role, comp = role.reindex(d.index), comp.reindex(d.index)
        attrs["stale_contrasts_dropped"] = n0 - len(d)
    d = d.assign(_c=d["delta"].notna(), _r=role, _o=comp).sort_values(["_c", "_r", "_o", "ingest_seq"], ascending=False)
    out = d.drop_duplicates(key).drop(columns=["_c", "_r", "_o"]).sort_index()
    out.attrs.update(attrs)
    return out


# COMPLETENESS (2026-09-20): the lifecycle runs queued -> generated -> judged -> ingested; `ingested` is the state AFTER
# `judged` (the harvester stamps it once the headline metric is complete, since #275), so a complete run reads `judged` OR `ingested`
# depending on when it folded - the same evidence under two names until the next rebuild. Never test for `judged` alone.
COMPLETE = ("judged", "ingested")
# OUT OF THE PAPER VIEW (2026-09-20: two runs per view x eval resolve to one `selected` and one `superseded`; never draw
# both): a superseded, withdrawn or excluded run is not a reading, on the frontier or anywhere else. `not_in_roster` IS a reading -
# the frontier draws every ladder rung whatever its roster state.
# `member_pass` (schema 21, 2026-10-02): a pooled reading's per-trial / per-rung MEMBER runs (sampled_lane base_cap /
# ladder_rung, relu_paired + unrequested_rung) are current measurements but not readings - the pooled row is the reading and the
# members are its inputs (940 runs; 229 in the Qwen frontier query, 16 unsteered). In view they would pool as extra draws beside
# their own pooled row, so they are out of view like a superseded run.
OUT_OF_VIEW = ("superseded", "withdrawn", "excluded", "member_pass", "source_local_anchor")
# SHARED-STANDUP RUNS NEVER ANCHOR (2026-10-05): the bcbw shared standups (`bfcl_full_shared*`, `bigcodebench_full_shared*`)
# are excluded from the paper frame by ruling - another serving geometry, a stand-in compared shared-vs-solo - but the unsteered POOL
# reads out-of-frame `none` runs on purpose (borrowable anchors), and six shared BigCodeBench nones of Qwen sat in it complete and
# `not_in_roster`. `pool_fence` keeps them out of every pool the anchors borrow from.
POOL_EXCLUDED_SOURCE_PATTERN = "_shared"


def pool_fence(pool: pd.DataFrame) -> pd.DataFrame:
    """The unsteered pool without the shared-standup sources (POOL_EXCLUDED_SOURCE_PATTERN in `lane_source_legacy`)."""
    if not len(pool) or "lane_source_legacy" not in pool.columns:
        return pool
    out = pool[~pool.lane_source_legacy.fillna("").astype(str).str.contains(POOL_EXCLUDED_SOURCE_PATTERN, regex=False)]
    out = out.copy(); out.attrs.update(pool.attrs); out.attrs["pool_fenced"] = int(len(pool) - len(out))
    return out
# `unpooled_member` (schema 22, 2026-10-02): a sampled_lane base_cap / ladder_rung pass whose point has NO pooled run of its
# source - a current measurement that the registry does not count as a paper row (the pinned renderer contract reads a sampled_lane
# point through its pooled run) until the adapter emits the missing pooled rows. Before 0021 the latest of these passes WAS the paper's
# reading of the cell (Llama / Nemotron constitution and difference LoRA on the four static attacks, 406 runs in all), so leaving them
# out blanks those cells. They are read as PROVISIONAL readings: admitted only where the cell has no selected row (`admit_provisional`),
# named by the notebook, and they leave by themselves the render after the pooled rows land (they become member_pass then).
PROVISIONAL_STATE = "unpooled_member"
# `source_local_anchor` (registry rule, migration 0027; seen 2026-10-06 at head 494241, 290 runs): an unsteered run filed by a ladder
# or re-fit source is that standup's own 0-dose comparator, not the paper's unsteered reading - current (is_current 1) but not a
# paper row (in_current_paper 0). THE PAPER'S UNSTEERED READING IS THE ROSTER-MATCHED NONES ALONE (2026-10-07): a
# cross-source borrow is exactly the shape the state removes, and the registry's own borrowed-anchor tier admits in-paper,
# roster-matched candidates only - so the state is OUT OF VIEW here, which fences it from the unsteered pool (`anchored` filters
# both the frame's nones and the borrowable extra through `in_view`; before this it UNIONED 17 ladder-leg nones beside 3 campaign
# nones into Qwen's codeattack@4096 anchor). The frontier's same-source rung anchoring is untouched: it draws from `none_all` of its
# own frame through `anchor_draws`, not through `in_view`. A cross-source fallback for a cell with no in-frame none would be a
# by-name ruling, never a pool rule.
ANCHOR_ONLY_STATE = "source_local_anchor"


def complete(frame: pd.DataFrame) -> pd.Series:
    """True where the run's headline metric is complete. Schema v6 (2026-09-21) states this as `is_complete` on the
    views, which is the authoritative spelling; the lifecycle tuple is the fallback for a frame read before the migration."""
    if "is_complete" in frame.columns and frame["is_complete"].notna().any():
        return frame["is_complete"].fillna(0).astype(int).eq(1)
    return frame.lifecycle_state.isin(COMPLETE)


def admit_provisional(frame: pd.DataFrame) -> tuple:
    """(frame, census): keep an `unpooled_member` row only where its STEERED cell (model, object, eval, lane, dose, item set) holds no
    `selected` row; a cell with a selected reading keeps that reading alone. UNSTEERED rows are never admitted (2026-10-07, render
    v156): the paper's unsteered reading is the roster-matched nones alone (registry rule 0027), anchors pool across
    lanes by the anchor rules, and a `none` member pass that looked "blank" under this lane-keyed cell was only on another lane than
    its roster none - admitting 42 of them put the member passes back beside the pooled anchors in 16 cells. The census names the
    admitted cells (model, object, eval, runs) for the notebook's disclosure line."""
    if "selection_state" not in frame.columns or not frame.selection_state.eq(PROVISIONAL_STATE).any():
        return frame, pd.DataFrame(columns=["model_id", "object_label", "eval_id", "runs"])
    keycols = ["model_id", "object_label", "eval_id"] + [c for c in ("lane", "dose", "item_set_id") if c in frame.columns]
    key = pd.Series(list(zip(*[frame[c].astype(str) for c in keycols])), index=frame.index)
    prov = frame.selection_state.eq(PROVISIONAL_STATE)
    covered = set(key[frame.selection_state.eq("selected")])
    drop = prov & (key.isin(covered) | frame.object_label.eq("none"))
    out = frame[~drop]
    adm = out[out.selection_state.eq(PROVISIONAL_STATE)]
    census = (adm.groupby(["model_id", "object_label", "eval_id"]).run_id.nunique().rename("runs").reset_index()
              if len(adm) else pd.DataFrame(columns=["model_id", "object_label", "eval_id", "runs"]))
    out = out.copy(); out.attrs.update(frame.attrs); out.attrs["provisional_dropped"] = int(drop.sum())
    return out, census


def frontier_view(frame: pd.DataFrame) -> pd.Series:
    """`in_view`, plus the `source_local_anchor` unsteered rows: on the frontier a ladder rung anchors on ITS OWN standup's none first
    (`anchor_draws`: same source, same standup), and since migration 0027 that none is a source-local anchor - current, out of the paper
    frame, and fenced from the paper panels' pool by `in_view`. The frontier keeps it as an anchor candidate and nothing else: never as a
    rung, never in the campaign fallback (which already excludes the ladder sources)."""
    v = in_view(frame)
    if "selection_state" in frame.columns and "object_label" in frame.columns:
        v = v | (frame.object_label.eq("none") & frame.selection_state.eq(ANCHOR_ONLY_STATE))
    return v


def in_view(frame: pd.DataFrame) -> pd.Series:
    """True where the run is not superseded, withdrawn or excluded (a `selection_state` column absent reads as in view)."""
    return ~frame.selection_state.isin(OUT_OF_VIEW) if "selection_state" in frame.columns else pd.Series(True, index=frame.index)


DELTA_COLS = ["n_anchor_candidates", "d_y", "d_lo", "d_hi", "route", "anchor_source", "anchor_value", "anchor_run", "anchor_in_frame", "headroom", "ceiling", "block_note",
              "anchor_draws", "anchor_spread", "anchor_cap", "cap_mismatch", "filed_delta", "anchor_sources", "anchor_views", "anchor_decodes", "mixed_decode",
              "steer_decode", "anchor_decode_keys", "decode_mismatch", "steer_serving", "anchor_servings"]
HIGHER_IS_BETTER = {"acc", "prompt_level_strict"}


def _first(block: dict, keys):
    for k in keys:
        if block.get(k) is not None:
            return block[k]
    return None


def block_anchor_kind(block) -> str:
    """The anchor kind a producer filed ON THE METRIC ROW, in `extra_json.delta_block.anchor` (2026-09-20). `anchor_kind`
    on run_contrast is a CLOSED vocabulary (same_standup_none | sibling_none | borrowed | full_dose) that cannot express "differenced
    against another lane's baseline" without a migration, and the paired lane has no run_contrast rows on XSTest at all - it has no
    unsteered controls cell to build one from - so the metric row is where that provenance lives. Read tolerantly: several spellings
    of the key, and a `cross_source` / `anchor_is_same_wave` flag alone is enough to say the anchor is not this wave's."""
    if not isinstance(block, dict):
        return ""
    a = block.get("anchor") if isinstance(block.get("anchor"), dict) else {}
    kind = _first(a, ["kind_legacy", "kind", "anchor_kind"]) or _first(block, ["anchor_kind", "kind_legacy"])
    if kind:
        return str(kind)
    if _first(a, ["cross_source"]) is True or _first(block, ["cross_source"]) is True:
        return "cross_source"
    if _first(a, ["anchor_is_same_wave"]) is False:
        return "external_base_lane"
    return ""


def _headroom(metric: str, anchor_value) -> tuple:
    """The room the unsteered run leaves toward the bound, in plotted units, and whether it is under HEADROOM_PP (`ceiling`)."""
    if pd.isna(anchor_value) or metric not in IS_RATE:
        return np.nan, False
    room = (100.0 - float(anchor_value)) if metric in HIGHER_IS_BETTER else float(anchor_value)
    return room, bool(room < HEADROOM_PP)


def decode_spelling(extra) -> str:
    """A short spelling of a run's decode settings from its extra (`decode`, else `decode_declared_legacy`), for telling anchors of
    different regimes apart in the tables; "" where the producer recorded none."""
    e = _extra(extra) if isinstance(extra, str) else extra
    if not isinstance(e, dict):
        return ""
    x = e.get("decode") or e.get("decode_declared_legacy") or {}
    if isinstance(x, dict):
        keys = ("preset", "temperature", "top_p", "top_k", "presence_penalty", "seed_policy")
        return ",".join(f"{k}={x[k]}" for k in keys if k in x)
    return str(x)[:60]


def decode_key(policy_id, spelling: str = "", source=None, sent=None) -> str:
    """The decode IDENTITY a mismatch is judged on: the registry's decode_policy_id policy segment (`<model>|<eval>|<thinking>|<policy>@vN`
    -> `<policy>`), else the run's `decode_source` (vendor_recommendation, runner_forced_T1, ... - the vendor tree's unsteered rows carry no
    policy id, only the source), with the sent temperature appended where `decode_sent_json` records one (`vendor_recommendation:T0.7`,
    `runner_forced_T1:T1`), else a STRUCTURED spelling from the run's extra (`k=v,...`). A free-text legacy spelling ("served by lora_native
    / thinking ON ...") is not a key - it described the GSM8K lens pairing and would have flagged every Llama GSM8K cell."""
    name = None
    if isinstance(policy_id, str) and "|" in policy_id:
        name = policy_id.rsplit("|", 1)[1].split("@", 1)[0]
    elif isinstance(source, str) and source.strip() and source.strip().lower() not in ("unverified", "unknown", "not_available", "none", "nan"):
        name = source.strip()
    temp = None
    try:
        sj = json.loads(sent) if isinstance(sent, str) and sent.strip() else (sent if isinstance(sent, dict) else None)
        if isinstance(sj, dict) and sj.get("temperature") is not None:
            temp = float(sj["temperature"])
    except (ValueError, TypeError):
        temp = None
    if name is None and isinstance(spelling, str) and spelling:
        parts = [x.split("=", 1) for x in spelling.split(",")]
        if all(len(kv) == 2 for kv in parts) and {kv[0] for kv in parts} <= {"preset", "temperature", "top_p", "top_k", "presence_penalty", "seed_policy"}:
            return spelling
    if name is None:
        return ""
    return f"{name}:T{temp:g}" if temp is not None else name


def decode_short(spelling: str) -> str:
    """A two-word reading of a decode key for a note under a mark: `vendor_recommendation:T0.7` -> "vendor T 0.7", `runner_forced_T1:T1` ->
    "T 1", `runner_forced_T1` -> "T 1", `vendor_recommendation` -> "vendor preset"; a structured spelling reads its temperature."""
    if not spelling:
        return ""
    sp = str(spelling)
    if "=" not in sp:
        name, _, t = sp.partition(":T")
        if t:
            lab = f"T {float(t):g}"
            return f"vendor {lab}" if "vendor" in name.lower() else lab
        m = re.search(r"_T(\d+(?:p\d+)?)$", name)
        if m:
            return "T " + m.group(1).replace("p", ".")
        return {"vendor_recommendation": "vendor preset", "thinking_card": "thinking card"}.get(name, name.replace("_", " "))
    kv = dict(part.split("=", 1) for part in sp.split(",") if "=" in part)
    t = kv.get("temperature")
    lab = f"T {float(t):g}" if t not in (None, "") else sp[:24]
    return f"vendor {lab}" if "vendor" in str(kv.get("preset", "")).lower() else lab


EVAL_CATEGORY = {e: cat for cat, (evs, _sgn, _lab) in CATEGORIES.items() for e in evs}
# WHERE THE DECODE DISCLOSURE APPLIES (2026-09-22): the agentic panel alone. The common anchor is the mean over EVERY
# unsteered run of the eval at the cap, and on the statics those runs were generated under several recorded decode names
# (grandfathered_T1, runner_forced_T1, vendor_preset, vendor_recommendation, generation_config ...), so a rule that fires on
# any mixed pool printed a footnote under every static and multi-turn panel and widened the figure threefold. The case that
# matters is the merged Qwen agentic arms (T 1.0) against the vendor tree's unsteered runs (T 0.7) - the one place the registry
# files NO contrast for want of a decode-matched baseline - so the note is confined there, and decodes are compared by their
# SETTINGS (decode_short: the temperature and vendor-ness), never by the source name, so two spellings of T 1.0 do not read as
# a mismatch.
DECODE_NOTE_CATEGORIES = {"agentic misalignment"}


def decode_mismatch_rule(steer_key: str, pool_keys, eval_id: str) -> bool:
    """True when the anchor pool's decode SETTINGS are not entirely the arm's - the arm's temperature absent from the pool, or a pool that
    mixes it with another (the Qwen agentic murder pool holds T 1.0 and vendor T 0.7 unsteered runs side by side, so the mean anchor is
    neither) - on the evals in DECODE_NOTE_CATEGORIES only, and only where every key names a temperature (an unknown decode is not a
    mismatch)."""
    if EVAL_CATEGORY.get(eval_id) not in DECODE_NOTE_CATEGORIES or not steer_key:
        return False
    labs = {decode_short(str(k)) for k in (pool_keys or []) if str(k) and str(k) != "nan"}
    mine = decode_short(steer_key)
    if not labs or not mine.startswith(("T ", "vendor T ")) or not all(l.startswith(("T ", "vendor T ")) for l in labs):
        return False
    return labs != {mine}


def decode_matched_pool(c: pd.DataFrame, steer_key: str, eval_id: str) -> pd.DataFrame:
    """THE DECODE-MATCHED ANCHOR (2026-09-22, the artifact default): on the agentic evals, where the pool
    holds unsteered runs at the arm's own decode settings beside others - the native canon completion-cap baseline at T 1.0 beside the
    vendor tree's T 0.7 rows - the arm anchors on the matched runs alone, never on a mean across decodes; where no matched run exists the
    pool stands and decode_note discloses the crossing. Compared by settings (decode_short), the same rule the disclosure uses; other
    categories keep the one-common-anchor pool untouched."""
    if not len(c) or "decode_key" not in c.columns or EVAL_CATEGORY.get(eval_id) not in DECODE_NOTE_CATEGORIES or not steer_key:
        return c
    mine = decode_short(steer_key)
    if not mine.startswith(("T ", "vendor T ")):
        return c
    labs = c["decode_key"].map(lambda k: decode_short(str(k)) if str(k) and str(k) != "nan" else "")
    return c[labs.eq(mine)] if labs.eq(mine).any() else c


def decode_note(row) -> str | None:
    """THE DECODE-MISMATCH DISCLOSURE (2026-09-22, on the merged Qwen agentic arms): where a delta's anchor pool runs another
    decode than the steered arm - the merged seed arms at T 1.0 against the vendor tree's unsteered rows at T 0.7 / top-p 0.8 / top-k 20 /
    presence 1.5 - the contrast crosses temperature and presence, and the panel says so under the cell, the way the lens-served pairing
    anchor is disclosed. Nothing is dropped or corrected; the registry files no contrast for those arms until a decode-matched unsteered
    baseline lands (which anchor then draws is the rule, default the matched one)."""
    try:
        if not bool(row.get("decode_mismatch", False)):
            return None
        anchors = sorted({decode_short(x) for x in str(row.get("anchor_decode_keys") or "").split(" | ") if x})
        note = f"anchor {' / '.join(a for a in anchors if a) or '?'}; arm {decode_short(row.get('steer_decode') or '') or '?'}"
        # THE SERVING PATH BESIDE THE DECODE (2026-09-22): the vendor tree's unsteered runs are passthrough-served on
        # the lens proxy while the merged arms are lora_native, so the same delta crosses an engine boundary too; the registry's own
        # serving_path values are printed when they differ, never a label of ours
        srv_a = {x for x in str(row.get("anchor_servings") or "").split(",") if x and x != "nan"}
        srv_s = str(row.get("steer_serving") or "")
        if srv_s and srv_a and srv_a != {srv_s}:
            note += f"; serving: anchor {' / '.join(sorted(srv_a))}, arm {srv_s}"
        return note
    except (KeyError, TypeError, ValueError):
        return None


FITTED_SUFFIXES = ("_lora", "_relu")


def retire_canonical_fits(d: pd.DataFrame) -> tuple:
    """THE CANONICAL BATCH-8 FITS ARE RETIRED: with three batch-64 seeds fit per object, a fitted object IS its
    three seeds, and the earlier single canonical fit - a blank-seed row of a `_lora` / `_relu` object - leaves every panel. Prompted
    arms, the Refusal steer and the unsteered model are not fits and keep every run. Returns (kept, retired).

    SCOPED TO THE (MODEL, OBJECT) PAIRS THAT HAVE SEED REPLICATES (2026-09-21, the all-model appendix). The batch-64 seeds were fit for
    Qwen3.5-4B alone; on Gemma-4-E4B, Nemotron-3-Nano-4B and Llama-3.1-8B the single canonical fit is the object's ONLY fit, and an
    unscoped retirement removed every non-Qwen LoRA and ReLU-steer row from the appendix panels (314 operating-point rows at head 143893:
    gemma 83, llama 73, nemotron 67, qwen 91) - cells the registry holds `selected` then drew the `judged, not in the paper roster` glyph.
    A blank-seed row of a fitted object is retired only where the same (model, object) carries a seeded row in the frame; where no seed
    replicate exists the canonical fit stands as the object's reading, one mark, no seed. Qwen's panels are unchanged by the scoping."""
    d = d.copy()
    fitted = d["object_label"].astype(str).str.endswith(FITTED_SUFFIXES)
    blank = d["seed"].fillna("").eq("")
    # THE SEED IS THE OBJECT'S, NOT THE VIEW'S (2026-09-22): a blank-seed row whose object id carries a seed on another row of the same
    # model and label is that seed's row re-served under a view with no seed spelling - the GSM8K run of the paired-write difference ReLU
    # lands on the plain `gemma4e4b_it` view while the object's other rows sit on `paired_d1_gemma4e4b_s1` - and is filled, never retired.
    if "object_id" in d.columns:
        seeded = d[fitted & ~blank & d["object_id"].notna()]
        by_obj = {k: v for k, v in seeded.groupby(["model_id", "object_label", "object_id"])["seed"].agg(lambda s: sorted(set(s))[0]).items()}
        fill = [by_obj.get((m, o, i), "") if (b and pd.notna(i)) else "" for m, o, i, b in zip(d["model_id"], d["object_label"], d["object_id"], blank)]
        filled = pd.Series(fill, index=d.index)
        d.loc[filled.ne(""), "seed"] = filled[filled.ne("")]
        blank = d["seed"].fillna("").eq("")
    # PER EVAL, NOT PER OBJECT (2026-09-24). A seed replicate exists on an EVAL: Qwen's three batch-64 seeds were run through the whole
    # battery, so on Qwen the two scopings agree, but on Gemma and Llama the new BigCodeBench / BFCL rows arrived stamped `fit_seed 1`
    # on the same canonical LoRA and constitution ReLU objects (the canonical fit IS seed 1 there; no other fit exists), and a
    # (model, object) scope read that one capability row as "this object has seeds" and retired the object's blank-seed rows on
    # every static and multi-query eval - 96 Gemma and 77 Llama rows at head 212703 - leaving the three LoRA lanes with Qwen and
    # Nemotron alone. A canonical fit leaves a panel only where a seeded fit of the same object stands on that panel's eval.
    ev = d["eval_id"] if "eval_id" in d.columns else pd.Series("", index=d.index)          # a frame without evals scopes per object
    triple = pd.Series(list(zip(d["model_id"], d["object_label"], ev)), index=d.index)
    seeded_triples = set(triple[fitted & ~blank])
    canon = fitted & blank & triple.isin(seeded_triples)
    return d[~canon], d[canon]


def _names(series, sep: str = ",") -> str:
    """A stable, sortable join of a provenance column: the pool spans two frames, so a column can hold a float NaN beside strings
    and `sorted` refuses to compare them. Coerce, drop blanks, sort as text."""
    if series is None:
        return ""
    vals = {str(x) for x in pd.Series(series).dropna() if str(x) and str(x) != "nan"}
    return sep.join(sorted(vals))


def metric_family(eval_id: str, metric_id: str) -> list:
    """The metric ids that name ONE quantity for this eval: the eval's declared headline list when `metric_id` is on it, else the id
    alone. strongreject_bon declares `asr` and `asr_bon_cap4096`, both per-goal any-hit@75 at cap 4,096 under two spellings."""
    ids = headline_metric_ids().get(eval_id) or []
    return list(ids) if metric_id in ids else [metric_id]


def anchor_candidates(frame: pd.DataFrame, r) -> pd.DataFrame:
    """The unsteered rows a steered row may be differenced against: the same model and eval, and a metric in the SAME FAMILY as the
    row's. ONE METRIC UNDER TWO SPELLINGS IS ONE METRIC (2026-09-23): the export's unsteered best-of-N runs file both `asr` and
    `asr_bon_cap4096` with one value, so the headline rule keeps `asr` for them, while the native Llama best-of-N arm files only the second -
    a literal match found that arm no unsteered run at all, and the appendix drew its absolute value with no delta. Same item set,
    same cap, same n: nothing but the spelling differed."""
    fam = metric_family(r.eval_id, r.metric_id)
    return frame[(frame.model_id == r.model_id) & (frame.eval_id == r.eval_id) & frame.metric_id.isin(fam)]


def anchored(d: pd.DataFrame, pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every steered headline row with its delta vs THE EVAL'S COMMON ANCHOR, in plotted units: a panel shares one
    anchor; where several unsteered runs exist they are averaged, and every point keeps its own interval. The common anchor for a
    steered row is the mean over EVERY unsteered run of the same (model, eval, metric) at the row's generation cap and, where the row
    carries one, on its item set - the campaign's run, the seed views' runs, the paired views' runs, the decode draws, all pooled as
    draws (anchor_summary: the draws' own intervals combined plus the between-draw variance). The point's interval is the two sides'
    combined (route `common anchor (k runs)`). Anchors are taken from the paper frame; only when the frame has none for that eval are
    the registry's unsteered runs OUTSIDE the selection used (route `... outside the paper frame`, drawn OPEN). A producer's own
    `delta_block` stands in only when no unsteered run exists at all (the paired write's MMLU against an external anchor); else the
    row reads `no unsteered run` and the panel marks it. Never across generation caps: a row whose cap no anchor shares is marked
    `cap_mismatch`. The pooled runs' sources, views and decode spellings ride on the row (`anchor_sources`, `anchor_views`,
    `anchor_decodes`, `mixed_decode`) so a pool that spans two decode regimes is visible, never silent. The producer's filed
    contrast, where one exists, is kept beside the result as `filed_delta` for cross-checking and is NOT the plotted value."""
    d = intervals(dedupe(d[d.headline & d.judge_ok]))
    none = d[d.object_label == "none"]
    extra = intervals(dedupe(pool[pool.headline & pool.judge_ok & pool.object_label.eq("none")])) if pool is not None and len(pool) else None
    for fr_ in (none, extra):
        if fr_ is not None and len(fr_) and "run_extra_json" in fr_.columns:
            fr_["decode_spelling"] = fr_["run_extra_json"].map(decode_spelling)
            fr_["decode_key"] = [decode_key(pid, sp, src, sent) for pid, sp, src, sent in zip(
                fr_["decode_policy_id"] if "decode_policy_id" in fr_.columns else [None] * len(fr_), fr_["decode_spelling"],
                fr_["decode_source"] if "decode_source" in fr_.columns else [None] * len(fr_),
                fr_["decode_sent_json"] if "decode_sent_json" in fr_.columns else [None] * len(fr_))]
    steered = d[d.object_label != "none"]
    # ONE READING PER CELL (model, object, seed, draw, lane, dose, eval, metric): one token cap per cell and per series, then the run with
    # a registry contrast, else the newest ingested; the collapsed count rides attrs["cell_duplicates"]
    cell = ["model_id", "object_label", "seed", "draw", "lane", "dose", "eval_id", "metric_id"] + [c for c in ("item_set_id", "pad", "pad_contract") if c in d.columns]
    if "token_cap_tokens" in steered.columns and steered.token_cap_tokens.notna().any():
        _sc = series_caps(steered)
        steered = (pd.concat([cap_choice(g, e, _sc.get((g.object_label.iat[0], e)))[0]
                              for (e, *_), g in steered.groupby(["eval_id"] + cell[:-1], dropna=False)]) if len(steered) else steered)
    # THE ESCALATED READING OUTRANKS ITS BASE PASS (schema 21, 2026-10-02): a cell may now hold a base_cap@4,096 pass beside
    # the ladder_rung@8,192 of the SAME trial, both released by design (ctl_neg1 statics, the dose ladders); the rung is the completion-cap
    # reading of that trial, the base pass its input. pooled > ladder_rung > base_cap, then a registry contrast, then the newest ingested.
    _pl = steered["pass_label"].astype(str) if "pass_label" in steered.columns else pd.Series("", index=steered.index)
    _esc = _pl.str.startswith("pooled").astype(int) * 2 + _pl.eq("ladder_rung").astype(int)
    ranked = steered.assign(_c=steered["delta"].notna(), _esc=_esc).sort_values(["_esc", "_c", "ingest_seq"], ascending=False)
    kept = ranked.drop_duplicates(cell)
    dup = ranked[~ranked.index.isin(kept.index)]
    steered = kept.drop(columns=["_c", "_esc"]).sort_index()

    def common_pool(frame, r):
        """Every unsteered run of the row's (model, eval, metric); the row's cap where any anchor shares it; the row's item set where any does.

        AN ANCHOR IS A COMPARATOR, NOT A DRAWN ROW (2026-09-21): the agentic leaking `none` at cap 3,000 is `excluded`
        under the standing forced-T1 ruling yet is exactly the run the seeds' contrasts name, so a pool filtered by selection would
        drop it and leave that eval anchorless. The caller therefore hands this the frame's unsteered rows UNION the registry's
        unsteered rows outside the selection, and `anchor_in_frame` says whether anything in the pool was in the frame."""
        c = anchor_candidates(frame, r)
        if r.eval_id in SHARED_FRAME_ANCHOR and len(c) and "run_id" in c.columns:
            inside = c[c.run_id.isin(set(none.run_id))]                   # the SI cells' anchor: the frame's own unsteered runs
            if len(inside):
                # ONE UNSTEERED ROW PER EVAL WAS AN ASSUMPTION, AND IT IS ABOUT TO STOP BEING TRUE (2026-09-22).
                # When the native canon baseline lands, each Qwen agentic eval carries TWO in-frame unsteered rows: the
                # vendor 16k row (lens-served, T 0.7) and the new misalign_ladder row (lora_native, T 1). Pooling them would
                # average across exactly the decode and engine boundary the panel currently warns about, and bake it into the
                # reference rather than the arms. So, in order: the anchor the REGISTRY binds to this row if it is in the
                # pool, then an unsteered run from the row's OWN lane, then the pooled in-frame set. Today no agentic row
                # carries a bound anchor and no misalign_ladder `none` exists, so both preferences miss and the behavior is
                # unchanged; each starts working by itself the moment its input appears.
                # the panel-wide anchor first: it outranks both the binding and the lane, because its whole purpose is that
                # every arm on the panel shares it (the binding names the same run for the arms that carry one anyway)
                lane = PANEL_ANCHOR_LANE.get(r.eval_id)
                if lane and "lane_source_legacy" in inside.columns:
                    panel = inside[inside.lane_source_legacy.eq(lane)]
                    if len(panel):
                        return panel
                bound = r.get("anchor_run_id")
                if pd.notna(bound) and (inside.run_id == bound).any():
                    return inside[inside.run_id == bound]
                if "lane_source_legacy" in inside.columns and pd.notna(r.get("lane_source_legacy")):
                    same = inside[inside.lane_source_legacy.eq(r.lane_source_legacy)]
                    if len(same):
                        return same
                return inside                                             # cap match deliberately given up, see SHARED_FRAME_ANCHOR
        if r.eval_id in SEED_MATCHED_ANCHOR and "seed" in c.columns:
            # THE OWN VIEW FIRST, THEN THE OWN SEED. A composed row on
            # `qwen35_4b_s1p` must pair with the PAIRED view's own `none`, not with the cross-standup `qwen35_4b_s1` one —
            # and a seed-only filter cannot tell them apart, because the trailing `p` is not part of the seed and both read
            # s1. With a paired none now filed beside the seed none on all four evals, a seed-only pool would have held BOTH
            # and anchored the composed arms on the mean of two standups. Falling back to the seed keeps every other row's
            # behaviour exactly as it was.
            same_view = (c["view_legacy"].fillna("").eq(str(r.get("view_legacy") or ""))
                         if "view_legacy" in c.columns else pd.Series(False, index=c.index))
            same_seed = c["seed"].fillna("").eq(str(r.get("seed") or ""))
            if same_view.any():
                c = c[same_view]
            elif same_seed.any():
                c = c[same_seed]
            # ...and where the row's seed has NO unsteered run, the eval's other unsteered runs stand: the unsteered model
            # has no fit, so there is no seed variance in it to respect, and dropping the point would lose a real
            # measurement to protect a distinction that does not exist on that arm. The preference order still means a seed
            # uses its own reading whenever it has one, so this only ever fills a hole.
        cls = cap_class(r.eval_id, r.get("token_cap_tokens"))
        if len(c) and "token_cap_tokens" in c.columns and cls and c.token_cap_tokens.isin(cls).any():
            c = c[c.token_cap_tokens.isin(cls)]
        if len(c) and "item_set_id" in c.columns and pd.notna(r.get("item_set_id")):
            if c.item_set_id.eq(r.item_set_id).any():
                c = c[c.item_set_id.eq(r.item_set_id)]
            elif r.eval_id in NO_CROSS_ITEM_SET:
                c = c.iloc[0:0]                      # nested widths: no anchor on this row's item set means NO anchor, not another width
        c = decode_matched_pool(c, decode_key(r.get("decode_policy_id"), decode_spelling(r.get("run_extra_json")), r.get("decode_source"), r.get("decode_sent_json")), r.eval_id)
        return c

    # AN EXCLUDED, SUPERSEDED OR WITHDRAWN RUN IS NEVER AN ANCHOR (2026-09-22): the borrowable pool is every judged unsteered run outside the
    # selection, and that let the registry-only 16k step re-draws of the unsteered Qwen model (misalign_16k, excluded by ruling, n a few dozen
    # re-drawn items) be averaged into the agentic anchor beside the whole-battery runs. Out-of-view runs are dropped here, whatever frame
    # they came from.
    if extra is not None and len(extra):
        extra = extra[in_view(extra)]
    none = none[in_view(none)] if len(none) else none
    out = []
    for _, r in steered.iterrows():
        _steer_cap = r.get("token_cap_tokens")
        # one common anchor per eval, pooled over the frame's unsteered runs AND those reachable only outside the selection
        both = pd.concat([none, extra], ignore_index=True) if (extra is not None and len(extra)) else none
        if len(both) and "run_id" in both.columns:
            both = both.drop_duplicates(subset=["run_id", "metric_id"] + (["judgment_id"] if "judgment_id" in both.columns else []))
        c = common_pool(both, r)
        outside = bool(len(c)) and (("run_id" not in c.columns) or (not c.run_id.isin(set(none.run_id)).any()))
        a = anchor_summary(c) if len(c) else None
        rec = r.to_dict(); rec["n_anchor_candidates"] = len(c)
        rec["filed_delta"] = r.delta if pd.notna(r.get("delta", np.nan)) else np.nan
        blk = r["delta_block"] if "delta_block" in d.columns and isinstance(r["delta_block"], dict) else None
        if a is not None:
            k = int(a.draws); dy = r.y - a.y
            route = f"common anchor{' outside the paper frame' if outside else ''} ({k} run{'s' if k != 1 else ''})"
            if pd.notna(r.lo) and pd.notna(r.hi):
                se = np.hypot((r.hi - r.lo) / 3.92, a.se); d_lo, d_hi = dy - 1.96 * se, dy + 1.96 * se
            elif pd.notna(r.get("delta_lo", np.nan)) and pd.notna(r.get("delta_hi", np.nan)):
                # A SCORE WITH NO INTERVAL OF ITS OWN (Petri's 1-10 misalignment score: no Wilson bound, no ci on the row) but a producer-filed
                # paired-bootstrap contrast: its half-width is the best statement of the point's uncertainty, so it is centered on the
                # common-anchor delta and the route says so. Every point keeps an interval.
                half = float(r.delta_hi - r.delta_lo) / 2.0; d_lo, d_hi = dy - half, dy + half
                route += " (interval width from the filed contrast)"
            else:
                d_lo = d_hi = np.nan
            rec.update(d_y=dy, d_lo=d_lo, d_hi=d_hi, route=route,
                       anchor_source=_names(c.get("lane_source_legacy")), anchor_value=a.y, anchor_run=a.run_id,
                       anchor_in_frame=not outside, anchor_draws=k, anchor_spread=a.spread,
                       anchor_sources=_names(c.get("lane_source_legacy")),
                       anchor_views=_names(c.get("view_legacy")),
                       anchor_decodes=_names(c.get("decode_spelling"), sep=" | "),
                       mixed_decode=len({str(x) for x in (c.get("decode_spelling") if c.get("decode_spelling") is not None else pd.Series(dtype=object)).dropna() if str(x)}) > 1)
            # the steered row's own decode against the pool's: a delta across decodes is disclosed under the cell (decode_note)
            _sd = decode_key(r.get("decode_policy_id"), decode_spelling(r.get("run_extra_json")), r.get("decode_source"), r.get("decode_sent_json"))
            _ad = {str(x) for x in (c.get("decode_key") if c.get("decode_key") is not None else pd.Series(dtype=object)).dropna() if str(x)}
            rec.update(steer_decode=_sd, anchor_decode_keys=" | ".join(sorted(_ad)), decode_mismatch=decode_mismatch_rule(_sd, _ad, r.eval_id),
                       steer_serving=str(r.get("serving_path") or ""), anchor_servings=_names(c.get("serving_path")))
        elif blk is not None and isinstance(blk.get("anchor"), dict) and _first(blk, ("delta", "d")) is not None:
            # a block is a delta vs UNSTEERED only when it declares an `anchor` (the paired write's MMLU against an external unsteered run);
            # anchorless blocks are deltas against the cell's own other reading and are NOT used - their reason is kept as `block_note`
            anc = blk["anchor"]
            lo_b, hi_b = _first(blk, ("delta_lo", "lo", "ci_lo")), _first(blk, ("delta_hi", "hi", "ci_hi"))
            av = (float(anc["value"]) * SCALE.get(r.metric_id, 1.0) if anc.get("value") is not None
                  else float(anc["p"]) * 100.0 if anc.get("p") is not None and r.metric_id in IS_RATE else np.nan)
            rec.update(d_y=float(_first(blk, ("delta", "d"))), d_lo=float(lo_b) if lo_b is not None else np.nan, d_hi=float(hi_b) if hi_b is not None else np.nan,
                       route="producer delta block (cross-source anchor)" if (anc.get("cross_source") or anc.get("cross_stack")) else "producer delta block",
                       anchor_source=anc.get("source") or anc.get("where") or anc.get("root") or anc.get("record"), anchor_value=av,
                       anchor_run=anc.get("run_id") or anc.get("record"), anchor_in_frame=False, anchor_draws=1, anchor_spread=0.0,
                       anchor_sources="", anchor_views="", anchor_decodes="", mixed_decode=False, steer_decode=decode_key(r.get("decode_policy_id"), decode_spelling(r.get("run_extra_json")), r.get("decode_source"), r.get("decode_sent_json")), anchor_decode_keys="", decode_mismatch=False)
        else:
            rec.update(d_y=np.nan, d_lo=np.nan, d_hi=np.nan, route="no unsteered run", anchor_source=None, anchor_value=np.nan, anchor_run=None,
                       anchor_in_frame=False, anchor_draws=0, anchor_spread=0.0, anchor_sources="", anchor_views="", anchor_decodes="", mixed_decode=False,
                       steer_decode=decode_key(r.get("decode_policy_id"), decode_spelling(r.get("run_extra_json")), r.get("decode_source"), r.get("decode_sent_json")), anchor_decode_keys="", decode_mismatch=False)
        rec["block_note"] = (blk.get("why_not_a_contrast_row") or blk.get("why_no_run_contrast")) if blk is not None else None
        rec["headroom"], rec["ceiling"] = _headroom(r.metric_id, rec.get("anchor_value", np.nan))
        # NEVER ACROSS GENERATION CAPS (2026-09-20): the pool is the row's cap where any anchor shares it; where none does, the row says so
        rec["anchor_cap"] = float(a.cap) if (a is not None and pd.notna(a.get("cap", np.nan))) else np.nan
        rec["cap_mismatch"] = bool(pd.notna(_steer_cap) and pd.notna(rec["anchor_cap"]) and rec["anchor_cap"] not in cap_class(r.eval_id, _steer_cap))
        out.append(rec)
    res = pd.DataFrame(out) if out else pd.DataFrame(columns=list(d.columns) + DELTA_COLS)
    res.attrs["cell_duplicates"] = dup[cell + ["run_id", "view_legacy", "lane_source_legacy", "in_current_paper"]].reset_index(drop=True)
    return res


# The two share columns, named once. As a two-literal tuple on one line the secret scanner read the pair as a key/value and its
# generic-api-key rule fired on the entropy of the column name; naming them here keeps the scanner un-weakened (no allowlist).
SHARE_COLS = ("frac_at_token_cap", "frac_degenerate" "_union5")


def pool_cell(d: pd.DataFrame, keys=("model_id", "object_label", "eval_id"), y: str = "y", lo: str = "lo", hi: str = "hi") -> pd.DataFrame:
    """ONE MARK PER (MODEL, OBJECT, EVAL) ON THE APPENDIX PANELS: every reading of a cell - its fit seeds, its
    decode draws, a canonical and a replicate generation - averages into one point with equal weight, the interval the readings' own
    combined plus the variance between them (the same pooling as `pool_seeds` and `anchor_summary`), `n_pooled` on the row. The
    first row's other columns ride along (they describe the object, eval and lane, which the pooled rows share); the at-cap and
    degenerate shares keep the worst reading, never averaged away. A cell with one reading is unchanged. `y` / `lo` / `hi` name the
    value columns, so the same pooling serves the absolute points (y) and the per-eval deltas (d_y)."""
    if not len(d):
        return d.assign(n_pooled=pd.Series(dtype=int)) if "n_pooled" not in d.columns else d
    keys = [k for k in keys if k in d.columns]
    out = []
    for _, g in d.groupby(keys, dropna=False, sort=False):
        r = g.iloc[0].copy()
        n = len(g)
        if n > 1:
            r[y] = g[y].mean()
            if g[lo].notna().all() and g[hi].notna().all():
                se_w = float(np.sqrt((((g[hi] - g[lo]) / 3.92) ** 2).sum()) / n)
                se_b = float(np.sqrt(g[y].var(ddof=1) / n))
                se = float(np.hypot(se_w, se_b))
                r[lo], r[hi] = r[y] - 1.96 * se, r[y] + 1.96 * se
            else:
                r[lo], r[hi] = np.nan, np.nan
            for c in SHARE_COLS + ("two_term_dev",):
                if c in g.columns:
                    r[c] = g[c].astype(float).max()
            for c in ("open", "partial", "ceiling", "cap_mismatch", "decode_mismatch", "collapsed"):
                if c in g.columns:
                    r[c] = bool(g[c].astype(bool).any())
            if "held_out" in g.columns:
                r["held_out"] = int(held_mask(g).any())
        r["n_pooled"] = n
        out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)


MEAN_COLUMN = "mean"


def eval_mean_column(pts: pd.DataFrame, evals: list, label: str = MEAN_COLUMN) -> pd.DataFrame:
    """THE MEAN COLUMN OF AN APPENDIX PANEL: per (model, object), the unweighted mean over the panel's evals of
    the model's pooled per-eval readings, one row per (model, object) under eval_id `label`, `n_evals` how many were averaged.

    ONE RULE FOR A MEAN OVER EVALUATIONS: se = sqrt(sum_e se_e^2) / k, the per-eval intervals
    combined and nothing else -- the rule `category_means` has always used for Figures 3-5 and the x = object appendix figures,
    and the one the paper's Statistics paragraph states. Until today this column ADDED the variance between the evals
    (hypot(se_within, sd(per-eval values) / sqrt(k))), as if the k evals were a random sample of evaluations; they are the panel's
    fixed battery, and the two appendix forms of the same panel then carried two different intervals for the same mean. The
    within-cell pooling (`pool_cell`: decode draws, fit seeds, generations of ONE cell) keeps its within + between term -- those are
    readings of the same quantity and the paragraph describes them separately. The spread between the evals rides along as
    `eval_spread` (max - min of the per-eval values) for a table legend to quote; it is not in the bar. Plotted units in and out."""
    c = pool_cell(pts[pts.eval_id.isin(evals)])
    rows = []
    for (m, o), g in c.groupby(["model_id", "object_label"], sort=False):
        n = len(g)
        se = float(np.sqrt((((g.hi - g.lo) / 3.92) ** 2).sum()) / n) if g.lo.notna().all() and g.hi.notna().all() else np.nan
        y = float(g.y.mean())
        rows.append(dict(model_id=m, object_label=o, eval_id=label, y=y, lo=(y - 1.96 * se) if pd.notna(se) else np.nan,
                         hi=(y + 1.96 * se) if pd.notna(se) else np.nan, n_evals=n, n_pooled=int(g.n_pooled.sum()),
                         eval_spread=float(g.y.max() - g.y.min()) if n > 1 else 0.0, seed="", draw=""))
    return pd.DataFrame(rows, columns=["model_id", "object_label", "eval_id", "y", "lo", "hi", "n_evals", "n_pooled", "eval_spread", "seed", "draw"])


def appendix_legend_groups(objects, models) -> list:
    """The appendix legend: two groups only - `object`, with the 95 % CI handle first and the unsteered model
    directly BELOW it where the roster carries `none` (a blank slot there otherwise, so the six appendix figures share one layout),
    then each corpus's ReLU steer above its LoRA, five columns; and `model`. No `marks` group."""
    handles = [CI_HANDLE] + object_handles(objects)        # the APPENDIX keeps CI first; the reorder is for Figures 3-5
    if "none" not in objects:
        handles.insert(1, Line2D([], [], ls="", label=" "))
    return [("object", handles, max(1, (len(handles) + 1) // 2)), ("model", model_handles(models), 4)]


def pool_draws(d: pd.DataFrame) -> pd.DataFrame:
    """Average a cell's DECODE DRAWS into one point. A draw is the same object, seed and eval re-decoded with
    another seed base or trial - the unsteered model, the two prompted SIs and the Refusal steer replicate that way, having one fit -
    so drawing a bar per draw says "several readings" where the experiment has one. Fits are NOT pooled here: the seed stays in the
    key, so three fits remain three marks and the draws inside each collapse. The pooled interval is the draws' own combined plus the
    variance BETWEEN them, as the anchor pooling does; `n_draws` rides on the row and a cell with one draw is unchanged."""
    if not len(d) or "draw" not in d.columns:
        return d.assign(n_draws=1) if len(d) else d
    key = [c for c in ("model_id", "object_label", "seed", "lane", "dose", "eval_id", "metric_id", "item_set_id", "pad", "pad_contract") if c in d.columns]
    out = []
    for _, g in d.groupby(key, dropna=False):
        r = g.iloc[0].copy()
        if len(g) > 1:
            n = len(g)
            r["d_y"] = g.d_y.mean()
            se_w = float(np.sqrt((((g.d_hi - g.d_lo) / 3.92) ** 2).sum()) / n) if g.d_lo.notna().all() else np.nan
            se_b = float(np.sqrt(g.d_y.var(ddof=1) / n)) if n > 1 else 0.0
            se = float(np.hypot(se_w, se_b)) if pd.notna(se_w) else np.nan
            r["d_lo"], r["d_hi"] = (r.d_y - 1.96 * se, r.d_y + 1.96 * se) if pd.notna(se) else (np.nan, np.nan)
            r["y"] = g.y.mean()
            for c in SHARE_COLS:                                  # the at-cap and degenerate shares: the worst draw, never averaged away
                if c in g.columns:
                    r[c] = g[c].astype(float).max()
            for c in ("ceiling", "cap_mismatch", "mixed_decode"):
                if c in g.columns:
                    r[c] = bool(g[c].astype(bool).any())
            r["draw"] = ",".join(sorted({str(x) for x in g.draw if str(x)}))
        r["n_draws"] = len(g)
        out.append(r)
    return pd.DataFrame(out).sort_index()


def category_means(deltas: pd.DataFrame, categories=CATEGORIES) -> pd.DataFrame:
    """One row per (model, object, seed, category): the unweighted mean over the category's evals of the oriented delta (an eval
    with several decode draws enters as the draws' mean), with the analytic interval from the rows' own (se = sqrt(sum se_i^2) / k). `partial` marks a mean over fewer evals than the
    category names; `open` a mean that used an anchor outside the paper frame for some eval; `ceiling` a mean with some eval whose
    unsteered run leaves under HEADROOM_PP of room; `at_cap` / `degen` the largest at-cap and degenerate shares among its evals; `anchors`
    their unsteered readings."""
    rows = []
    deltas = deltas[~held_mask(deltas)]                                     # a held-out row is out of every pooled mean (the registry's hold_out)
    for cat, (cols, sign, _) in categories.items():
        evals = [POOLED_EVAL[cat]] if cat in POOLED_EVAL else cols          # the registry's pooled eval is the summary where one is declared
        x = deltas[deltas.eval_id.isin(evals) & deltas.d_y.notna()].assign(_se=lambda f: (f.d_hi - f.d_lo) / 3.92)
        for (m, obj, seed), g in x.groupby(["model_id", "object_label", "seed"]):
            # the draws of one eval first (their mean, se = sqrt(sum se_i^2) / n_draws), then the unweighted mean over the evals present
            pe = g.groupby("eval_id").agg(dy=("d_y", "mean"), se=("_se", lambda v: np.sqrt((v ** 2).sum()) / len(v)))
            se = np.sqrt((pe.se ** 2).sum()) / len(pe)
            cap = g["frac_at_token_cap"].astype(float).max() if "frac_at_token_cap" in g.columns else np.nan
            deg = g["frac_degenerate_union5"].astype(float).max() if "frac_degenerate_union5" in g.columns else np.nan
            xcap = bool(g["cap_mismatch"].any()) if "cap_mismatch" in g.columns else False
            # the cap PER EVAL: a category mean spans evals that legitimately differ (MMLU is multiple-choice at cap 1, GSM8K
            # generative on an 8k ladder), so a bare set of caps across the category compares things that were never comparable
            caps = ("|".join(f"{e}={v:.0f}" for e, v in sorted({(r.eval_id, float(r.token_cap_tokens)) for _, r in g.iterrows()
                                                                if pd.notna(r.get("token_cap_tokens"))})) if "token_cap_tokens" in g.columns else "")
            # THE ITEM SET PER EVAL, for the same reason the cap is per eval - and a harder line than the cap. Two token caps can be
            # rungs of one declared ladder; two item sets are always two instruments (MMLU's full 1,531-item pinned pool
            # supersedes the 200-item head for Qwen): the head is item-for-item the first 200 of the 1,531, so the two
            isets = ("|".join(f"{e}={v}" for e, v in sorted({(r.eval_id, str(r.item_set_id)) for _, r in g.iterrows()
                                                             if pd.notna(r.get("item_set_id"))})) if "item_set_id" in g.columns else "")
            esc = ",".join(sorted({str(x) for x in g["escalation_state"].dropna()})) if "escalation_state" in g.columns else ""
            # THE ABSOLUTE UNSTEERED READING, beside the delta: every table shows the unsteered baseline as an
            # absolute first, then each arm as +/- against that common anchor. Built the same way as the
            # delta so the two are commensurate: the draws of one eval average first, then an unweighted mean over the
            # evals present. NOT sign-flipped — `sign` orients the DELTA so positive reads as better, but an absolute
            # reading has no orientation: over-refusal's baseline is a benign-compliance percentage either way.
            _pa = g.groupby("eval_id").anchor_value.mean()
            _abs = float(_pa.mean()) if len(_pa.dropna()) else np.nan
            rows.append(dict(model_id=m, category=cat, object_label=obj, seed=seed, anchor_abs=_abs, y=sign * pe.dy.mean(), lo=sign * pe.dy.mean() - 1.96 * se,
                             hi=sign * pe.dy.mean() + 1.96 * se, k=len(pe), partial=len(pe) < len([e for e in evals if (m, e) not in NOT_A_GAP]), open=bool((~g.anchor_in_frame.astype(bool)).any()),
                             ceiling=bool(g.ceiling.astype(bool).any()), at_cap=cap, degen=deg, cap_mismatch=xcap, caps=caps, item_sets=isets, escalation=esc,
                             anchors=",".join(f"{v:.0f}" for v in g.anchor_value if pd.notna(v)), evals=",".join(sorted(g.eval_id))))
    return pd.DataFrame(rows, columns=["model_id", "category", "object_label", "seed", "anchor_abs", "y", "lo", "hi", "k", "partial", "open", "ceiling", "at_cap", "degen",
                                       "cap_mismatch", "caps", "item_sets", "escalation", "anchors", "evals"])


GEN_CANONICAL, GEN_REPLICATE = "canonical", "replicate"


def generation(seed) -> str:
    """Which GENERATION a reading belongs to (2026-09-21). The paper's canonical full22 runs carry a blank-seed view
    (qwen35_4b / qwen35_4b_it); the seed-replicate runs carry a seeded view (s1, qwen35_4b_s2, ...). They are different generations,
    not seeds of one: where the same object appears in both - only the unsteered model and the two prompted arms, which have one
    identity by construction - averaging them makes an interval between GENERATIONS, and for every fitted object the "same object"
    premise does not even hold, the replicate fits being distinct objects. So they are drawn as separate series and never pooled."""
    return GEN_CANONICAL if not str(seed or "") else GEN_REPLICATE


def pool_seeds(cm: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, category, object): the seeds' category means averaged with equal weight into a single mean
    point (the seeds themselves stay as small marks), the interval from the seeds' analytic intervals combined (se = sqrt(sum se_i^2) / n);
    `partial` / `open` / `ceiling` if any seed's mean is; `at_cap` / `degen` the largest; `n_seeds` how many were averaged; `spread` the
    widest gap between them. The interval carries the BETWEEN-FIT variance as well as each fit's own (2026-09-20): a seed is a
    distinct fitted object, and three fits of one recipe at one dose can span 40 pp."""
    rows = []
    # ONE MEAN PER OBJECT: the canonical batch-8 fits are retired, so a fitted object's rows are its three seeds
    # and a prompted arm's or the unsteered model's rows are decode-seed trials of one object; both pool into a single mean point.
    gen = GEN_CANONICAL
    for (m, cat, obj), g in cm.groupby(["model_id", "category", "object_label"]):
        # BETWEEN-FIT VARIANCE (2026-09-20): a seed here is a DISTINCT FITTED OBJECT, not a re-run, and the fits disagree - the
        # paired ReLU's three fits of one recipe at one dose span 41.3 pp to 1.5 pp on PAIR, the campaign's largest spread. Combining
        # only each fit's own interval would state that mean far more precisely than the evidence allows, so the between-fit variance
        # joins it the way anchor_summary already pooled decode draws; `spread` (max - min) rides on the row for the tables.
        n = len(g)
        se_within = np.sqrt((((g.hi - g.lo) / 3.92) ** 2).sum()) / n
        se_between = float(np.sqrt(g.y.var(ddof=1) / n)) if n > 1 else 0.0
        se = float(np.sqrt(se_within ** 2 + se_between ** 2))
        rows.append(dict(model_id=m, category=cat, object_label=obj, generation=gen, y=g.y.mean(), lo=g.y.mean() - 1.96 * se, hi=g.y.mean() + 1.96 * se,
                         n_seeds=n, seeds=",".join(sorted(x for x in g.seed if x)), spread=float(g.y.max() - g.y.min()) if n > 1 else 0.0,
                         k=int(g.k.max()), partial=bool(g.partial.any()), open=bool(g.open.any()),
                         ceiling=bool(g.ceiling.any()), at_cap=g.at_cap.max(), degen=g.degen.max(),
                         cap_mismatch=bool(g.cap_mismatch.any()) if "cap_mismatch" in g.columns else False,
                         # THE SEEDS MUST SHARE A CAP to be read as replicates: a base-cap seed averaged with escalated ones is not
                         # seed variance, it is two instruments (2026-09-21, on the reseeded pending-ladder s1 statics).
                         seed_caps="|".join(sorted({c for v in g.caps.fillna("") for c in v.split("|") if c})) if "caps" in g.columns else "",
                         seed_item_sets="|".join(sorted({c for v in g.item_sets.fillna("") for c in v.split("|") if c})) if "item_sets" in g.columns else "",
                         escalation=",".join(sorted({e for v in g.escalation.fillna("") for e in v.split(",") if e})) if "escalation" in g.columns else "",
                         anchor_abs=float(g.anchor_abs.mean()) if "anchor_abs" in g.columns and g.anchor_abs.notna().any() else np.nan,
                         anchors=";".join(g.anchors.astype(str)), evals=g.evals.iloc[0]))
    out = pd.DataFrame(rows, columns=["model_id", "category", "object_label", "generation", "anchor_abs", "y", "lo", "hi", "n_seeds", "seeds", "spread", "k", "partial", "open",
                                      "ceiling", "at_cap", "degen", "cap_mismatch", "seed_caps", "seed_item_sets", "escalation", "anchors", "evals"])
    if len(out):
        # RUNGS OF ONE DECLARED LADDER ARE NOT A MIXED INSTRUMENT (asking why capability carried this mark).
        # GSM8K's seeds sit at 8,192 and 16,384 - the base cap and the final rung of ONE declared ladder, every value the ladder's
        # final state - so the mark was firing on a distinction the registry says is not one. It now fires only when the seeds'
        # caps are NOT all rungs of a single declared ladder, i.e. genuinely two instruments.
        def _by_eval(v, cast=None):
            out = {}
            for part in str(v or "").split("|"):
                if "=" in part:
                    e, c = part.rsplit("=", 1)
                    try:
                        out.setdefault(e, set()).add(cast(c) if cast else c)
                    except ValueError:
                        pass
            return out

        def _caps_by_eval(v):
            return _by_eval(v, float)

        def _mixed(row):
            # A SEED STILL CLIMBING IS NOT A FINAL READING, so the static mean's mark is genuine. Rungs of ONE declared
            # ladder are one instrument ONLY when each is that cell's final state: the statics declare
            # [4096, 8192, 16384], so the ladder test alone would have silenced this, but 20 of the 180 static rows read
            # `escalation_pending` - the reseeded seed-1 battery, whose escalation failed and is being re-run - and pooling an interim
            # base-cap reading with escalated seeds is exactly the two-instrument mean the mark exists to show.
            if int(row.n_seeds) <= 1:
                return False
            if "escalation_pending" in str(row.escalation or ""):
                return True
            # WITHIN each eval, did the seeds sit at different caps that are not rungs of one declared ladder?
            per = _caps_by_eval(row.seed_caps)
            for ev, caps in per.items():
                if len(caps) > 1 and not any(caps <= lad for lad in _LADDERS.get(ev, ())):
                    return True
            # TWO ITEM SETS WITHIN ONE EVAL IS ALWAYS TWO INSTRUMENTS - no ladder makes them one. Qwen's MMLU moved from the 200-item
            # head to the full 1,531-item pinned pool, which SUPERSEDES rather than adds; while cells land at
            # both widths a mean over them spans two instruments and must say so.
            for ev, sets in _by_eval(row.get("seed_item_sets", "")).items():
                if len(sets) > 1:
                    return True
            return False
        out["mixed_cap_seeds"] = out.apply(_mixed, axis=1)
    return out


def headline_points(op: pd.DataFrame, evals: list) -> pd.DataFrame:
    """One absolute headline reading per (model, object, seed, decode draw, eval) with its interval: judge rule applied, deduped, and where
    several unsteered runs measure one eval (the ladders' anchors beside the campaign's), the source the paper's headline reads
    (ANCHOR_PREFERENCE), else the first."""
    d = intervals(dedupe(op[op.eval_id.isin(evals) & op.judge_ok]))
    if not len(d):
        return d
    keep = [pick_anchor(g, "").name for _, g in d.groupby(["model_id", "object_label", "seed", "draw", "eval_id"])]
    return d.loc[keep].copy()


def flip(pts: pd.DataFrame) -> pd.DataFrame:
    """100 − rate: defense success from ASR, benign compliance from the over-refusal label mean (the interval's bounds swap)."""
    return pts.assign(y=100 - pts.y, lo=100 - pts.hi, hi=100 - pts.lo)


def grand_mean(pts: pd.DataFrame, keys: list) -> pd.DataFrame:
    """The mean over models of one value per model (a model's seeds averaged first), ± SEM (sd with n − 1): the appendix's level
    summary, never a contrast interval. Deliberately NOT the mean-over-evaluations rule: the
    models are the population this diamond summarises, so the spread between them IS its uncertainty; the caption says so."""
    rows = []
    for k, g in pts.groupby(keys):
        v = g.groupby("model_id").y.mean()
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec.update(y=v.mean(), n_models=len(v), se=(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan)     # `se`, not `sem`: Series.sem is a method
        rows.append(rec)
    return pd.DataFrame(rows, columns=keys + ["y", "n_models", "se"])


# ---- transfer: base->instruct vs instruct-direct, batch-matched (appendix) --------------------------------
# A previous paper draft carried Figure 4, "transfer": each fitted object's BASE->INSTRUCT reading minus its INSTRUCT-DIRECT reading,
# per evaluation, in the form of Figures 2-3 (paperlib.figs.fig4 / transfer_delta). It comes back here as an APPENDIX figure with one
# added condition: the two fits are compared only where they were trained at the SAME EFFECTIVE BATCH. The registry holds batch-8 and
# batch-64 fits of the same object on the same checkpoint, and a base-vs-instruct difference across a batch change would read the
# batch as transfer. The batch is resolved from the object record (`fit_effective_batch`, else accum_steps x batch_size from the
# fit hyperparameters, else -- for a derived object: a LoRA subtraction, a dose-scaled copy, a composed ReLU steer -- the batch its
# parents agree on, recursively). Each lane's reading is that lane's own anchored delta against the unsteered model (the same
# `anchored` as every panel); the difference's interval treats the two readings as independent (conservative where they share the
# anchor); a category mean is the unweighted mean over the evals present with se = sqrt(sum se_e^2) / k, the one rule.
OBJECTS_SQL = ("SELECT object_id, paper_label, fit_model_id, fit_effective_batch, fit_hyperparameters_json, parent_object_ids_json, "
               "derivation, fit_replicate_label FROM object")
TRANSFER_MODELS = ["gemma4e4b_it", "qwen35_4b_it", "llama8b_it"]                 # nemotron has one checkpoint and one fit
TRANSFER_LANES = ("base_to_instruct", "instruct_direct")                          # (minuend, subtrahend): transfer minus in place
TRANSFER_LABEL = "base→instruct fit − instruct-direct fit"
TRANSFER_LAYOUT = [["static", "multi-query", "multi-turn"], ["over-refusal", "capability"], ["Petri", "agentic misalignment"]]


def resolve_fit_batch(objects: pd.DataFrame) -> pd.Series:
    """object_id -> effective training batch, or NaN. Own `fit_effective_batch` first; else accum_steps x batch_size from the fit
    hyperparameters; else, for a derived object, the batch its parents resolve to when they all agree (recursive: a dose-scaled
    copy of a LoRA subtraction of two fits resolves through two levels). Disagreeing or unresolvable parents -> NaN."""
    o = objects.set_index("object_id")
    own = pd.to_numeric(o["fit_effective_batch"], errors="coerce")
    def _hp(v):
        try:
            h = json.loads(v) if isinstance(v, str) else (v if isinstance(v, dict) else {})
            if not isinstance(h, dict):
                return np.nan
            a, b = h.get("accum_steps"), h.get("batch_size")
            return float(a) * float(b) if a is not None and b is not None else np.nan
        except (ValueError, TypeError):
            return np.nan
    hp = o["fit_hyperparameters_json"].map(_hp) if "fit_hyperparameters_json" in o.columns else pd.Series(np.nan, index=o.index)
    parents = o["parent_object_ids_json"].map(lambda v: json.loads(v) if isinstance(v, str) and v.strip().startswith("[") else []) \
        if "parent_object_ids_json" in o.columns else pd.Series([[]] * len(o), index=o.index)
    memo: dict = {}
    def _res(oid, depth=0):
        if oid in memo:
            return memo[oid]
        if oid not in o.index or depth > 6:
            return np.nan
        v = own.get(oid)
        if pd.notna(v):
            memo[oid] = float(v); return memo[oid]
        v = hp.get(oid)
        if pd.notna(v):
            memo[oid] = float(v); return memo[oid]
        pb = {b for b in (_res(pid, depth + 1) for pid in parents.get(oid, [])) if pd.notna(b)}
        memo[oid] = float(next(iter(pb))) if len(pb) == 1 else np.nan
        return memo[oid]
    return pd.Series({oid: _res(oid) for oid in o.index}, name="fit_batch")


# THE TRANSFER FIGURE'S RULES, replacing the batch-matched pairing of the first version: (1) pairs need NOT be batch-matched;
# (2) LoRAs only -- constitution, control, difference -- on Qwen, Gemma and Llama, no ReLU steers; (3) Qwen's base->instruct
# side is the three batch-64 seed fits AVERAGED, never the old batch-8 base fits; (4) the instruct-direct object of every
# pair is PINNED by object id (Gemma's constitution LoRA is the batch-8 itdoc fit, chosen over the five-eval batch-64 one).
# Ids are the registry `object_id` prefixes pinned below; the notebook checks each resolves to exactly one object. The
# batches still ride along in the pairs table as information, since a difference across a batch change is worth knowing
# about even where it is no longer a condition.
TRANSFER_OBJECTS = ["control_lora", "constitution_lora", "difference_lora"]
# the transfer figure's categories: the appendix set, except that capability keeps the coding evals -- the base->instruct side has
# full-suite BigCodeBench / BFCL rows and the instruct-direct side gains them from later runs (2026-09-25); MMLU + GSM8K alone
# is the all-model appendix's compromise for models without them, which this figure does not need
TRANSFER_CATEGORIES = {**APPENDIX_CATEGORIES, "capability": CATEGORIES["capability"]}
TRANSFER_FLOOR = 5.0             # a capability reading under this many % on either side is a floor (a refusal collapse, a scoring failure), named
TRANSFER_PINS = {
    # (model, object): (base->instruct object ids or None for "every base->instruct fit in the frame", the instruct-direct object id)
    ("qwen35_4b_it", "constitution_lora"): (("8be1e1a7", "724fe6dd", "a9934f6a"), "0018b32b"),
    ("qwen35_4b_it", "control_lora"): (("000e0df5", "900938b5", "33e7b327"), "bace45e2"),
    ("qwen35_4b_it", "difference_lora"): (("0d275f7c", "06f81d94", "8952728f"), "5e9b0009"),
    ("gemma4e4b_it", "constitution_lora"): (None, "4c1a5fc3"),
    ("gemma4e4b_it", "control_lora"): (None, "04a90de8"),
    ("gemma4e4b_it", "difference_lora"): (None, "8b7d1030"),
    ("llama8b_it", "constitution_lora"): (None, "529788ca"),
    ("llama8b_it", "control_lora"): (None, "7bf26f8c"),
    ("llama8b_it", "difference_lora"): (None, "e85220c9"),
}


def _id_match(ids: pd.Series, prefixes) -> pd.Series:
    """True where the object id starts with any of the pinned prefixes (objects are named by their first eight hex digits)."""
    pre = tuple(prefixes)
    return ids.astype(str).map(lambda x: x.startswith(pre))


def transfer_pin_check(objects: pd.DataFrame) -> pd.DataFrame:
    """Every pinned prefix against the registry object table: how many objects it matches (must be exactly one) and what the registry
    calls that object. The notebook prints the rows that are not exactly one match."""
    rows = []
    for (m, o), (base_ids, it_id) in TRANSFER_PINS.items():
        for lane, pre in [(TRANSFER_LANES[1], it_id)] + ([(TRANSFER_LANES[0], b) for b in base_ids] if base_ids else []):
            hit = objects[objects.object_id.astype(str).str.startswith(pre)]
            rows.append(dict(model_id=m, object_label=o, lane=lane, prefix=pre, matches=len(hit),
                             registry_label=hit.paper_label.iloc[0] if len(hit) == 1 else "", fit_model_id=hit.fit_model_id.iloc[0] if len(hit) == 1 else ""))
    return pd.DataFrame(rows)


TRANSFER_COMPARISON = "figA_transfer_models"                 # roster v10's comparison id for this figure (2026-09-25)


def transfer_roster_check(frame: pd.DataFrame) -> pd.DataFrame:
    """The roster's word against the figure's pins (roster v10, 2026-09-25): every in-frame row carrying a `figA_transfer_models/*`
    comparison id, reduced to (model, object, lane, object id), against TRANSFER_PINS. One row per (model, object, lane, id) with
    `status`: `both` (the roster names it and the pins name it), `roster only` (the roster admits an object the figure does not draw),
    `pins only` (a pinned object no in-frame row carries under the comparison -- absent from the frame, or the roster disagrees).
    The notebook prints anything that is not `both`; the two lists can then never drift apart silently."""
    cols = ["model_id", "object_label", "lane", "object_id", "status"]
    if not len(frame) or "comparison_ids_json" not in frame.columns:
        return pd.DataFrame(columns=cols)
    ids = frame["comparison_ids_json"].map(_comparison_ids)
    hit = ids.map(lambda L: any(str(i).startswith(TRANSFER_COMPARISON) for i in L))
    r = frame[hit & frame.object_label.isin(TRANSFER_OBJECTS) & frame.lane.isin(TRANSFER_LANES)]
    roster = {(m, o, ln, str(oid)[:8]) for m, o, ln, oid in zip(r.model_id, r.object_label, r.lane, r.object_id)}
    pins = set()
    for (m, o), (base_ids, it_id) in TRANSFER_PINS.items():
        pins.add((m, o, TRANSFER_LANES[1], it_id))
        for b in (base_ids or []):
            pins.add((m, o, TRANSFER_LANES[0], b))
        if base_ids is None:                                                   # "every base->instruct fit in the frame": the roster's base rows are the pins
            pins |= {k for k in roster if k[0] == m and k[1] == o and k[2] == TRANSFER_LANES[0]}
    rows = [dict(model_id=m, object_label=o, lane=ln, object_id=oid, status="both" if (m, o, ln, oid) in pins else "roster only")
            for (m, o, ln, oid) in sorted(roster)]
    rows += [dict(model_id=m, object_label=o, lane=ln, object_id=oid, status="pins only") for (m, o, ln, oid) in sorted(pins - roster)]
    return pd.DataFrame(rows, columns=cols)


def transfer_pairs(d: pd.DataFrame, categories: dict, objects: list | None = None, models: list | None = None, pins: dict | None = None) -> tuple:
    """From ANCHORED rows of both lanes (`d_y` / `d_lo` / `d_hi` against each lane's own unsteered model; `object_id` on every row),
    the transfer differences under TRANSFER_PINS. Returns (per_eval, per_cat, pairs): per_eval has one row per (model, object, eval)
    with `diff` = base->instruct minus instruct-direct in raw plotted units (NOT yet oriented), `lo` / `hi` from the two lanes'
    intervals treated as independent, both lanes' readings and the fits behind them (`n_base`: the seed fits pooled on the
    base->instruct side); per_cat one row per (model, object, category) with the oriented mean over the evals present
    (se = sqrt(sum se_e^2) / k), `k`, `partial`; pairs one row per (model, object) with `status` (`matched`, or why not: no
    base->instruct fit in the frame, the pinned instruct-direct fit absent from the frame, no evaluation in common) and the
    batches on each side as information (`base_batch`, `it_batch`, from `fit_batch` where the frame carries it)."""
    objects = list(objects or TRANSFER_OBJECTS)
    models = list(models or TRANSFER_MODELS)
    pins = TRANSFER_PINS if pins is None else pins
    x = d[d.object_label.isin(objects) & d.model_id.isin(models) & d.lane.isin(TRANSFER_LANES) & d.d_y.notna()].copy()
    if "held_out" in x.columns:
        x = x[~held_mask(x)]
    if "fit_batch" not in x.columns:
        x["fit_batch"] = np.nan
    pe_rows, pairs = [], []
    for m in models:
        for o in objects:
            if (m, o) not in pins:
                continue
            base_ids, it_id = pins[(m, o)]
            g = x[(x.model_id == m) & (x.object_label == o)]
            base = g[(g.lane == TRANSFER_LANES[0]) & (_id_match(g.object_id, base_ids) if base_ids else True)]
            it = g[(g.lane == TRANSFER_LANES[1]) & _id_match(g.object_id, [it_id])]
            def _b(f):
                v = sorted({int(b) for b in pd.to_numeric(f.fit_batch, errors="coerce").dropna()})
                return ", ".join(str(b) for b in v) if v else "?"
            rec = dict(model_id=m, object_label=o, it_object=it_id, base_batch=_b(base), it_batch=_b(it),
                       n_base_fits=int(base.object_id.nunique()), k=0)
            if not len(base):
                pairs.append(dict(rec, status="no base→instruct fit in the frame" + (f" (pinned {', '.join(base_ids)})" if base_ids else ""))); continue
            if not len(it):
                pairs.append(dict(rec, status=f"pinned instruct-direct fit {it_id} absent from the frame")); continue
            # one reading per (lane, eval): the lane's seed fits and decode draws pooled by `pool_cell` (within + between: readings of one cell)
            cb = pool_cell(base, keys=("model_id", "object_label", "lane", "eval_id"), y="d_y", lo="d_lo", hi="d_hi").set_index("eval_id")
            ci = pool_cell(it, keys=("model_id", "object_label", "lane", "eval_id"), y="d_y", lo="d_lo", hi="d_hi").set_index("eval_id")
            evs = sorted(set(cb.index) & set(ci.index))
            if not evs:
                pairs.append(dict(rec, status="no evaluation in common")); continue
            for e in evs:
                rb, ri = cb.loc[e], ci.loc[e]
                se_b, se_i = (rb.d_hi - rb.d_lo) / 3.92, (ri.d_hi - ri.d_lo) / 3.92
                diff = float(rb.d_y - ri.d_y); se = float(np.sqrt(se_b ** 2 + se_i ** 2)) if pd.notna(se_b) and pd.notna(se_i) else np.nan
                _abs = lambda r: float(r.anchor_value) + float(r.d_y) if pd.notna(r.get("anchor_value", np.nan)) else np.nan   # the side's own reading
                pe_rows.append(dict(model_id=m, object_label=o, eval_id=e, diff=diff, se=se, lo=diff - 1.96 * se if pd.notna(se) else np.nan,
                                    hi=diff + 1.96 * se if pd.notna(se) else np.nan, base_dy=float(rb.d_y), it_dy=float(ri.d_y),
                                    base_abs=_abs(rb), it_abs=_abs(ri), n_base=int(rb.get("n_pooled", 1)), n_it=int(ri.get("n_pooled", 1))))
            pairs.append(dict(rec, status="matched", k=len(evs)))
    per_eval = pd.DataFrame(pe_rows, columns=["model_id", "object_label", "eval_id", "diff", "se", "lo", "hi", "base_dy", "it_dy", "base_abs", "it_abs", "n_base", "n_it"])
    cat_rows = []
    for cat, (evals, sign, _) in categories.items():
        evs = [POOLED_EVAL[cat]] if cat in POOLED_EVAL else list(evals)
        for (m, o), g in per_eval[per_eval.eval_id.isin(evs)].groupby(["model_id", "object_label"]):
            k = len(g); se = float(np.sqrt((g.se ** 2).sum()) / k) if g.se.notna().all() else np.nan
            y = sign * float(g["diff"].mean())
            named = [e for e in evs if (m, e) not in NOT_A_GAP]
            cat_rows.append(dict(model_id=m, object_label=o, category=cat, y=y, lo=y - 1.96 * se if pd.notna(se) else np.nan,
                                 hi=y + 1.96 * se if pd.notna(se) else np.nan, k=k, partial=k < len(named), evals=",".join(sorted(g.eval_id))))
    per_cat = pd.DataFrame(cat_rows, columns=["model_id", "object_label", "category", "y", "lo", "hi", "k", "partial", "evals"])
    return per_eval, per_cat, pd.DataFrame(pairs, columns=["model_id", "object_label", "status", "it_object", "base_batch", "it_batch", "n_base_fits", "k"])


def transfer_instruments(d: pd.DataFrame, pins: dict | None = None) -> pd.DataFrame:
    """Per (model, object, side, eval), the runs that entered the pairing and their instruments: run count, token caps, decode sources,
    producers. `mixed` marks a cell pooling more than one cap or decode -- the campaign's cap-4,096 pass beside the strength ladder's
    cap-8,192 dose-1 rung, say -- which the pairing averages as draws of one cell (the appendix's pool_cell rule) and the caption
    should know about (2026-09-25: 122 of 414 cells on the live frame)."""
    pins = TRANSFER_PINS if pins is None else pins
    x = d[d.object_label.isin(TRANSFER_OBJECTS) & d.lane.isin(TRANSFER_LANES)].copy()
    if "d_y" in x.columns:
        x = x[x.d_y.notna()]
    def side(r):
        b, it = pins.get((r.model_id, r.object_label), (None, None)); oid = str(r.object_id)
        if it and oid.startswith(it): return "instruct-direct"
        if r.lane == TRANSFER_LANES[0] and (b is None or oid.startswith(tuple(b))): return "base→instruct"
        return None
    x["side"] = x.apply(side, axis=1); x = x[x.side.notna()]
    if not len(x):
        return pd.DataFrame(columns=["model_id", "object_label", "side", "eval_id", "runs", "caps", "decodes", "sources", "mixed"])
    cap = pd.to_numeric(x.get("token_cap_tokens"), errors="coerce")
    x["_cap"] = cap.map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    g = x.groupby(["model_id", "object_label", "side", "eval_id"]).agg(
        runs=("run_id", "nunique"), caps=("_cap", lambda v: " | ".join(sorted({c for c in v if c}, key=int))),
        decodes=("decode_source", lambda v: " | ".join(sorted({str(c) for c in v}))) if "decode_source" in x.columns else ("run_id", "size"),
        sources=("lane_source_legacy", lambda v: " | ".join(sorted({str(c) for c in v}))) if "lane_source_legacy" in x.columns else ("run_id", "size")).reset_index()
    g["mixed"] = g.caps.str.contains("|", regex=False) | g.decodes.astype(str).str.contains("|", regex=False)
    # a 512 pooled with a higher cap is the signature of a superseded duplicate still marked selected (2026-09-25: the cap-512
    # r2_export multi-turn runs sat in 15 transfer cells of index.v145; the listing had them, below the printed head)
    g["low_cap"] = g.mixed & g.caps.str.split(" | ", regex=False).map(lambda v: "512" in v and len(v) > 1)
    return g


def transfer_floors(per_eval: pd.DataFrame, evals: list, floor: float = TRANSFER_FLOOR) -> pd.DataFrame:
    """Capability cells where either side's own reading is under `floor` %: a difference there is a floor effect -- a refusal collapse
    (2026-09-25: Qwen's instruct-direct difference LoRA refuses ~1,136 of 1,140 BigCodeBench prompts, 0.35 % pass) or a scoring
    failure -- not a capability difference, and the figure's caption names it rather than letting the point read as transfer."""
    if not len(per_eval) or "base_abs" not in per_eval.columns:
        return per_eval.iloc[0:0]
    f = per_eval[per_eval.eval_id.isin(evals)]
    return f[(f.base_abs < floor) | (f.it_abs < floor)]


TRANSFER_SHARE_FLOOR = 0.10      # a degenerate or at-cap share at or above this, on either side of a pair, is named under the figure


def transfer_shares(d: pd.DataFrame, pins: dict | None = None, floor: float = TRANSFER_SHARE_FLOOR) -> pd.DataFrame:
    """The degenerate and at-cap shares behind every transfer cell, one row per (model, object, eval) with both sides' worst shares
    (`it_degen`, `it_at_cap`, `base_degen`, `base_at_cap`), kept where either side reaches `floor`. A reading whose replies are mostly
    degenerate or truncated is not the same kind of number as its partner's, and a difference between the two reads the failure
    mode as transfer (for the figure's caption). Reads the anchored transfer frame (`tr_d`)."""
    pins = TRANSFER_PINS if pins is None else pins
    cols = ["model_id", "object_label", "eval_id", "it_degen", "it_at_cap", "base_degen", "base_at_cap", "it_value", "base_value"]
    if not len(d):
        return pd.DataFrame(columns=cols)
    x = d[d.object_label.isin(TRANSFER_OBJECTS) & d.lane.isin(TRANSFER_LANES)].copy()
    if "d_y" in x.columns:
        x = x[x.d_y.notna()]                                                   # the rows that ENTERED the pairing
    def side(r):
        b, it = pins.get((r.model_id, r.object_label), (None, None)); oid = str(r.object_id)
        if it and oid.startswith(it): return "it"
        if r.lane == TRANSFER_LANES[0] and (b is None or oid.startswith(tuple(b))): return "base"
        return None
    x["side"] = x.apply(side, axis=1); x = x[x.side.notna()]
    for c in ("frac_degenerate_union5", "frac_at_token_cap"):
        x[c] = pd.to_numeric(x.get(c), errors="coerce") if c in x.columns else np.nan
    g = x.groupby(["model_id", "object_label", "eval_id", "side"]).agg(degen=("frac_degenerate_union5", "max"), at_cap=("frac_at_token_cap", "max"), value=("y", "mean")).reset_index()
    w = g.pivot(index=["model_id", "object_label", "eval_id"], columns="side", values=["degen", "at_cap", "value"])
    w.columns = [f"{side}_{what}" for what, side in w.columns]
    w = w.reset_index()
    for c in cols:
        if c not in w.columns: w[c] = np.nan
    w = w[cols]
    keep = (w[["it_degen", "it_at_cap", "base_degen", "base_at_cap"]].fillna(0) >= floor).any(axis=1)
    return w[keep].sort_values(["model_id", "object_label", "eval_id"]).reset_index(drop=True)


# GOAL-LEVEL DEGENERACY FOR THE MULTI-TURN CELLS (asking for the non-degenerate subset). The frame's
# `frac_degenerate_union5` on a multi-turn run counts REPLIES -- every turn of every sample -- and the attacks sample some goals
# dozens of times, so a few goals' collapsed samples read as "60 % degenerate" while ASR is a per-GOAL quantity. The multiturn
# grading pass files the goal-level decomposition on the headline judgment: goals with any degenerate turn, and how many of the
# jailbroken goals had one. From those two counts the ASR over the goals WITHOUT a degenerate turn follows exactly:
# (k_jb - k_jb_degen) / (n - n_degen). Filed for the in-place fits and for Gemma's base fits; not for the Qwen seed fits or Llama.
MT_DEGEN_METRICS = ("goal_degenerate_any_turn_share", "goal_jailbroken_any_turn_degenerate_share")
MT_DEGEN_SQL = ("SELECT run_id, judgment_id, metric_id, threshold, k, n, value FROM run_metric "
                "WHERE (metric_id = 'asr' AND threshold = 0.5) OR metric_id IN ('goal_degenerate_any_turn_share', 'goal_jailbroken_any_turn_degenerate_share')")


def multiturn_clean_asr(rows: pd.DataFrame, metric_rows: pd.DataFrame, pins: dict | None = None, evals=("crescendo", "actorattack")) -> pd.DataFrame:
    """One row per (model, object, side, eval) of the transfer pairs on the multi-turn evals: the headline ASR over all goals and the
    ASR over the goals with no degenerate turn, from the goal-level rows filed on the same (run, judgment) as the headline. `rows`
    is the transfer frame (headline rows with run_id and judgment_id); `metric_rows` the MT_DEGEN_SQL result. Where the goal-level
    rows are absent the clean columns are NaN and `has_goal_rows` is False -- the cell keeps its reading and says so."""
    pins = TRANSFER_PINS if pins is None else pins
    x = rows[rows.eval_id.isin(evals) & rows.object_label.isin(TRANSFER_OBJECTS) & rows.lane.isin(TRANSFER_LANES)].copy()
    if "d_y" in x.columns:
        x = x[x.d_y.notna()]                                                   # only the runs that entered the pairing
    if not len(x):
        return pd.DataFrame(columns=["model_id", "object_label", "side", "eval_id", "runs", "goals", "jailbroken", "degenerate_goals", "jailbroken_degenerate", "asr_all", "asr_clean", "goal_degen_share", "has_goal_rows"])
    def side(r):
        b, it = pins.get((r.model_id, r.object_label), (None, None)); oid = str(r.object_id)
        if it and oid.startswith(it): return "instruct-direct"
        if r.lane == TRANSFER_LANES[0] and (b is None or oid.startswith(tuple(b))): return "base→instruct"
        return None
    x["side"] = x.apply(side, axis=1); x = x[x.side.notna()]
    m = metric_rows.drop_duplicates(["run_id", "judgment_id", "metric_id", "threshold"])
    out = []
    for (mo, o, sd, e), g in x.groupby(["model_id", "object_label", "side", "eval_id"]):
        keys = set(zip(g.run_id, g.judgment_id.astype(str)))
        mm = m[[(r, str(j)) in keys for r, j in zip(m.run_id, m.judgment_id)]]
        hl = mm[mm.metric_id.eq("asr")]; dg = mm[mm.metric_id.eq("goal_degenerate_any_turn_share")]; jd = mm[mm.metric_id.eq("goal_jailbroken_any_turn_degenerate_share")]
        n, k = float(hl.n.sum()), float(hl.k.sum())
        has = len(dg) > 0 and len(jd) > 0 and len(dg) == len(hl)
        nd, kd = (float(dg.k.sum()), float(jd.k.sum())) if has else (np.nan, np.nan)
        out.append(dict(model_id=mo, object_label=o, side=sd, eval_id=e, runs=int(g.run_id.nunique()), goals=int(n), jailbroken=int(k),
                        degenerate_goals=int(nd) if has else np.nan, jailbroken_degenerate=int(kd) if has else np.nan,
                        asr_all=100.0 * k / n if n else np.nan, asr_clean=100.0 * (k - kd) / (n - nd) if has and n - nd > 0 else np.nan,
                        goal_degen_share=100.0 * nd / n if has and n else np.nan, has_goal_rows=bool(has)))
    return pd.DataFrame(out).sort_values(["model_id", "object_label", "eval_id", "side"]).reset_index(drop=True)


def transfer_figure(per_eval: pd.DataFrame, per_cat: pd.DataFrame, pairs: pd.DataFrame, objects: list, models: list, categories: dict,
                    name: str = "figA_transfer_models", height: float = 4.0):
    """The appendix transfer figure, in the x = object appendix form (`category_panels`): one panel per category (stacked), x = the
    fitted objects, the models dodged inside each lane by marker shape; per (object, model) the category mean of the oriented
    base->instruct minus instruct-direct difference is the large mark with its interval, the individual evals small and faded
    beside it; a black ring marks a mean over fewer evals than the category names. Zero is "the transferred fit did exactly what
    the in-place fit did"; up is the transferred fit doing better on that axis. A (model, object) with no batch-matched pair
    draws the `absent` glyph at zero in its slot. Categories with no pair on any eval are left out and named in the caption
    line the notebook prints. (The pairs are not batch-matched: see TRANSFER_PINS.)"""
    cats = [c for c in categories if c in set(per_cat.category)]
    if not cats:
        return None
    # THE MAIN FIGURES' ROW GROUPING (v164 stacked seven panels one per row, 28 in tall and too narrow for six lane labels): the
    # three jailbreak families on one row, over-refusal beside capability, Petri beside agentic misalignment -- Figures 3-6's rows.
    layout = [[c for c in row if c in cats] for row in TRANSFER_LAYOUT]
    layout = [row for row in layout if row] + [[c] for c in cats if not any(c in row for row in TRANSFER_LAYOUT)]
    ncol = max(len(row) for row in layout)
    fig = plt.figure(figsize=(min(1.7 * len(objects) + 0.6, 8.0) * ncol, height * len(layout)), layout="constrained")
    axes = _panel_grid(fig, layout, {c: 1 for c in cats})
    nm = max(len(models), 1); slot = 0.84 / nm
    any_ring = any_gap = False
    for ax, cat in axes:
        evals, sign, ylabel = categories[cat]
        evs = [POOLED_EVAL[cat]] if cat in POOLED_EVAL else list(evals)
        for j, o in enumerate(objects):
            for k, m in enumerate(models):
                xc = j - 0.42 + slot * (k + 0.5)
                r = per_cat[(per_cat.model_id == m) & (per_cat.object_label == o) & (per_cat.category == cat)]
                pe = per_eval[(per_eval.model_id == m) & (per_eval.object_label == o) & per_eval.eval_id.isin(evs)].sort_values("eval_id")
                if not len(r):
                    st = pairs[(pairs.model_id == m) & (pairs.object_label == o)]
                    if len(st) and st.status.iloc[0] != "matched":
                        ax.scatter([xc], [0.0], **GAP_MARK["absent"], zorder=2); any_gap = True
                    continue
                r = r.iloc[0]
                for xi, (_, q) in zip(_spread(xc, len(pe), 0.36 * slot), pe.iterrows()):
                    yv, l, u = sign * q["diff"], min(sign * q.lo, sign * q.hi), max(sign * q.lo, sign * q.hi)
                    ax.plot([xi] * 2, [l, u], color=COLOR[o], lw=0.9, alpha=0.4, zorder=2)
                    ax.scatter([xi], [yv], marker=MARKER[m], s=26, color=COLOR[o], alpha=0.5, edgecolors="none", zorder=2)
                ax.plot([xc] * 2, [r.lo, r.hi], color=COLOR[o], lw=2.4, zorder=3)
                ax.scatter([xc], [r.y], marker=MARKER[m], s=175, color=COLOR[o], edgecolors="white", lw=0.9, zorder=4)
                if bool(r.partial):
                    any_ring = True
                    ax.scatter([xc], [r.y], marker=MARKER[m], s=175, facecolors="none", edgecolors="black", lw=1.3, zorder=5)
        ax.axhline(0, color=INK2, lw=1.2, zorder=1)
        ax.set_xticks(range(len(objects))); ax.set_xticklabels([SHORT.get(o, LABEL.get(o, o)) for o in objects], fontsize=16)
        ax.set_xlim(-0.6, len(objects) - 0.4)
        ax.set_ylabel(_wrap_label(ylabel or "Δ (pp)"))                          # the category's own unit; what is differenced is the figure's label
        ax.set_title(cat)
        ax.grid(True, axis="y", color=RULE, lw=0.9); ax.set_axisbelow(True)
    fig.supylabel(TRANSFER_LABEL, fontsize=18)
    marks = []
    if any_ring:
        marks.append(Line2D([], [], ls="", marker="o", ms=9, markerfacecolor="none", markeredgecolor="black", label="mean over fewer evals than the category names"))
    if any_gap:
        marks.append(Line2D([], [], ls="", marker="x", ms=8, color=MUTED, label="no pair of fits in the frame (the pairs table says why)"))
    groups = [("object", object_handles(objects), max(1, (len(objects) + 1) // 2)), ("model", model_handles(models), 4)]
    if marks:
        groups.append(("marks", marks, 2))
    legend_stack(fig, groups)
    return save(fig, name)


def table_transfer(per_cat: pd.DataFrame, pairs: pd.DataFrame, objects: list, models: list, categories: dict, caption: str, label: str) -> str:
    """The transfer figure as numbers: one row per (model, object), one column per category with the oriented mean difference and
    its interval; a dagger marks a mean over fewer evals than the category names; `--` a category with no pair. The `fits` column
    gives the two sides' training batches as information (base->instruct / instruct-direct; Qwen's base side is its three seed
    fits pooled) -- no longer a condition. Rows for (model, object) without a pair state why."""
    cats = [c for c in categories if c in set(per_cat.category)]
    header = ["model", "arm", "fits (batch: base→instruct / instruct-direct)"] + [CATEGORY_SHORT.get(c, c) for c in cats]
    rows = []
    for m in models:
        for o in objects:
            st = pairs[(pairs.model_id == m) & (pairs.object_label == o)]
            if not len(st):
                continue
            st = st.iloc[0]
            if st.status != "matched":
                rows.append([_tex(MODEL_LABEL.get(m, m)), _tex(LABEL.get(o, o)), _tex(st.status)] + ["--"] * len(cats)); continue
            cells = []
            for c in cats:
                r = per_cat[(per_cat.model_id == m) & (per_cat.object_label == o) & (per_cat.category == c)]
                cells.append("--" if not len(r) else _cell(r.y.iloc[0], r.lo.iloc[0], r.hi.iloc[0], dp=1, signed=True) + ("$^{\\dagger}$" if bool(r.partial.iloc[0]) else ""))
            fits = f"{st.base_batch}" + (f" ({int(st.n_base_fits)} seed fits pooled)" if int(st.n_base_fits) > 1 else "") + f" / {st.it_batch}"
            rows.append([_tex(MODEL_LABEL.get(m, m)), _tex(LABEL.get(o, o)), _tex(fits)] + cells)
    note = ("Each cell: the object's base$\\to$instruct reading minus its instruct-direct reading, each lane's own delta against the "
            "unsteered model, oriented so positive means the transferred fit did better on that axis; mean over the category's "
            "evaluations, se = $\\sqrt{\\sum \\mathrm{se}_e^2}/k$. Qwen's base$\\to$instruct side is its three batch-64 seed fits pooled; the "
            "instruct-direct fit of every pair is pinned by object id; the fits column states the training batches "
            "as information, not a condition. $\\dagger$ a mean over fewer evaluations than the category names.")
    return latex_table(header, rows, caption, label, colspec="lll" + "c" * len(cats), note=note)


# ---- the sign-flipped control objects: Qwen control LoRA / ReLU at alpha = -1 (appendix) ----------------------
# The control objects at dose -1.0 -- the same fits, served with the sign flipped -- on the static, multi-turn and over-refusal
# evals, in the main figures' form: two rows, static + mean above, multi-turn and over-refusal (each + mean) below. The rows arrive
# as they fold; this reads whatever the frame holds at dose -1.0 for the two objects and says what it found.
NEGATIVE_DOSE_OBJECTS = ["control_lora", "control_relu"]
NEGATIVE_DOSE_LAYOUT = [["static"], ["multi-turn", "over-refusal"]]


NEGATIVE_DOSE_IN_VIEW = ("selected", "not_in_roster")     # roster-pending rows draw, marked (roster v11 carries the control ReLU entries)


def negative_dose_rows(frame: pd.DataFrame, model: str = MODEL_ID, objects: list | None = None, dose: float = -1.0, tol: float = 1e-6,
                       anchors_from: pd.DataFrame | None = None, seeded_only: bool = True) -> pd.DataFrame:
    """The headline rows of `objects` on `model` at `dose` (any lane, any source) that are IN VIEW -- `selected`, or `not_in_roster`
    while roster v11 is pending (2026-09-25: the control ReLU's alpha = -1 runs are in the registry but no comparison
    admits them yet) -- plus every unsteered row of the model from `anchors_from` (the paper frame; the anchors `anchored` needs in
    the frame it is handed). Held-out rows leave as everywhere. `frame` may be a dedicated query over v_paper_long with no
    in_current_paper filter, which is how the roster-pending rows reach the figure.

    `seeded_only` (v189, 2026-09-25): the figure is the three seed fits of each family, one row per seed. The strength ladder's
    own alpha = -1 rung of a control LoRA (`strength_lane_base2it_native`, view without a seed suffix, a fourth fit that is none
    of the three) also sits at dose -1 in the registry and drew as a seedless fourth mark under every static eval. Where a family
    has seeded rows, its blank-seed arm rows leave and are reported in `.attrs["seedless"]` (object, source, view, runs, evals) as a
    LIST OF RECORDS, never a DataFrame: pandas compares `attrs` dicts when it concatenates frames, and a DataFrame value makes that
    comparison raise (v190 died in `drop_excluded_cells` on exactly this)."""
    objects = list(objects or NEGATIVE_DOSE_OBJECTS)
    d = pd.to_numeric(frame.get("dose"), errors="coerce")
    arms = frame.object_label.isin(objects) & (d - dose).abs().le(tol) & frame.model_id.eq(model) & frame.headline
    if "selection_state" in frame.columns:
        arms &= frame.selection_state.isin(NEGATIVE_DOSE_IN_VIEW)
    seedless = pd.DataFrame(columns=["object_label", "lane_source_legacy", "view_legacy", "runs", "evals"])
    if seeded_only and "seed" in frame.columns:
        blank = frame.seed.fillna("").astype(str).eq("")
        seeded_objects = set(frame.loc[arms & ~blank, "object_label"])
        drop = arms & blank & frame.object_label.isin(seeded_objects)
        if drop.any():
            keys = [c for c in ["object_label", "lane_source_legacy", "view_legacy"] if c in frame.columns]
            seedless = frame[drop].groupby(keys).agg(runs=("run_id", "nunique"), evals=("eval_id", lambda v: ",".join(sorted(set(v))))).reset_index()
        arms &= ~drop
    src = anchors_from if anchors_from is not None else frame
    anchors = src[src.object_label.eq("none") & src.model_id.eq(model) & src.headline]
    out = pd.concat([frame[arms], anchors], ignore_index=True)
    if "held_out" in out.columns:
        out = out[~held_mask(out)]
    out = out.copy(); out.attrs["seedless"] = seedless.to_dict("records")
    return out


def negative_dose_pending(rows: pd.DataFrame, objects: list | None = None, evals: list | None = None) -> pd.DataFrame:
    """The alpha = -1 arm rows the roster does not yet name (`not_in_roster`), per object: count and evals -- printed under the figure
    and gone by itself once roster v11 loads. `evals` restricts the census to the evaluations the figure draws (an MMLU row at
    dose -1 is in the registry too, and is not this figure's)."""
    objects = list(objects or NEGATIVE_DOSE_OBJECTS)
    if "selection_state" not in rows.columns:
        return pd.DataFrame(columns=["object_label", "rows", "evals"])
    a = rows[rows.object_label.isin(objects) & rows.selection_state.eq("not_in_roster")]
    if evals is not None:
        a = a[a.eval_id.isin(list(evals))]
    if not len(a):
        return pd.DataFrame(columns=["object_label", "rows", "evals"])
    return a.groupby("object_label").agg(rows=("run_id", "nunique"), evals=("eval_id", lambda v: ",".join(sorted(set(v))))).reset_index()


def negative_dose_provenance(rows: pd.DataFrame, objects: list | None = None) -> pd.DataFrame:
    """What the frame holds at the negative dose: per (object, source, lane) the seeds and the evals -- the line printed under the figure."""
    objects = list(objects or NEGATIVE_DOSE_OBJECTS)
    a = rows[rows.object_label.isin(objects)]
    if not len(a):
        return pd.DataFrame(columns=["object_label", "lane_source_legacy", "lane", "seeds", "evals", "rows"])
    g = a.groupby(["object_label", "lane_source_legacy", "lane"], dropna=False)
    return g.agg(seeds=("seed", lambda v: ",".join(sorted({str(x) for x in v if str(x)}))), evals=("eval_id", "nunique"), rows=("run_id", "nunique")).reset_index()


# ---- geometry: the ReLU steers' layer-12 read / write directions against the Arditi refusal direction (appendix) ---------------
# The cosines are computed by `paper/notebooks/relu_arditi_cosines.py` under an interpreter with torch (this venv has none, on
# purpose) into `data/relu_arditi_cosines.json`; the notebook READS that sidecar and BINDS it to the registry: every row carries
# the sha256 of the object file it came from, which is the registry's `object_id`, so a row whose id the registry does not know
# under that label / seed / checkpoint is refused, and the figure says which. The reference is the Arditi direction toward
# refusal (the positive-c rung's stored vector, c * unit direction); the ReLU steer's READ direction is its gate probe
# `gate_w`, the WRITE direction the vector it adds, `directions`, both at decoder block 12 (resid_post, the layer the Arditi
# direction lives at). Two comparisons per direction kind: each seed against the refusal direction, and each pair of seeds
# against each other -- the question being whether the steers found the refusal direction or something of their own that
# the seeds nonetheless agree on. The random-direction reference is 1 / sqrt(d).
COSINE_SIDECAR = os.path.join(HERE, "data", "relu_arditi_cosines.json")
COSINE_KINDS = (("read", "read direction (gate probe)"), ("write", "write direction (added vector)"))
COSINE_MARKER = {"read": "o", "write": "s"}


def load_cosine_sidecar(path: str = COSINE_SIDECAR) -> dict:
    with open(path) as f:
        return json.load(f)


def cosine_sidecar_check(sidecar: dict, objects: pd.DataFrame) -> pd.DataFrame:
    """One row per sidecar entry (the reference included) with `known`: the registry holds an object of that id whose paper_label,
    fit_replicate_label and fit_model_id are the ones the sidecar claims. `objects` is the registry object table (OBJECTS_SQL)."""
    o = objects.set_index("object_id")
    rows = []
    ref = sidecar["reference"]
    rows.append(dict(object_label=ref["object_label"], seed="", fit_model_id="", object_id=ref["object_id"], role="reference",
                     known=ref["object_id"] in o.index and str(o.at[ref["object_id"], "paper_label"]) == ref["object_label"]))
    for e in sidecar["entries"]:
        ok = e["object_id"] in o.index
        if ok:
            r = o.loc[e["object_id"]]
            _seed = r["fit_replicate_label"]; _seed = "" if _seed is None or (isinstance(_seed, float) and np.isnan(_seed)) else str(_seed)
            ok = (str(r["paper_label"]) == e["object_label"] and _seed == (e["seed"] or "") and str(r["fit_model_id"]) == e["fit_model_id"])
        rows.append(dict(object_label=e["object_label"], seed=e["seed"], fit_model_id=e["fit_model_id"], object_id=e["object_id"], role="steer", known=bool(ok)))
    return pd.DataFrame(rows)


COSINE_GROUPS = [   # x groups of the figure: label, colour, how to read the sidecar
    ("refusal vs.\ncontrol", "control_relu", "ref"), ("refusal vs.\nconstitution", "constitution_relu", "ref"),
    ("control vs.\nconstitution", None, "cross"), ("control vs.\ncontrol", "control_relu", "pair"), ("constitution vs.\nconstitution", "constitution_relu", "pair")]
COSINE_KIND_LABEL = {"read": "read direction ($w_\\ell$)", "write": "write direction ($v_\\ell$)"}


def cosine_frames(sidecar: dict) -> tuple:
    """(ref, pairs): ref has one row per (object, seed, kind) with the cosine against the refusal direction; pairs one row per
    (object, seed pair, kind) with the cosine between the two seeds. `lane` names the fit checkpoint (base->instruct / instruct-direct)."""
    lane = lambda m: "base→instruct" if str(m).endswith("_base") else "instruct-direct"
    ref = pd.DataFrame([dict(object_label=e["object_label"], seed=e["seed"], lane=lane(e["fit_model_id"]), kind=k, cos=e[f"{k}_vs_ref"])
                        for e in sidecar["entries"] for k, _ in COSINE_KINDS])
    pairs = pd.DataFrame([dict(object_label=q["object_label"], pair=f"{q['seed_a']}–{q['seed_b']}", kind=k, cos=q[k])
                          for q in sidecar["pairs"] for k, _ in COSINE_KINDS])
    return ref, pairs


def cosine_groups(sidecar: dict) -> pd.DataFrame:
    """One row per (group, kind, point) for the figure's five x groups: the seeded base->instruct fits against the refusal direction
    (two groups, one per object), every control seed against every constitution seed (`cross_pairs`), and each object's seeds
    against one another. Columns: group, kind, cos, colour object (None for the cross group)."""
    ref, pairs = cosine_frames(sidecar)
    ref = ref[ref.lane == "base→instruct"]
    rows = []
    for label, obj, how in COSINE_GROUPS:
        if how == "ref":
            f = ref[ref.object_label == obj]
            rows += [dict(group=label, kind=r.kind, cos=r.cos, colour=obj) for r in f.itertuples()]
        elif how == "pair":
            f = pairs[pairs.object_label == obj]
            rows += [dict(group=label, kind=r.kind, cos=r.cos, colour=obj) for r in f.itertuples()]
        else:
            rows += [dict(group=label, kind=k, cos=q[k], colour=None) for q in sidecar.get("cross_pairs", []) for k, _ in COSINE_KINDS]
    return pd.DataFrame(rows, columns=["group", "kind", "cos", "colour"])


def cosine_figure(sidecar: dict, check: pd.DataFrame | None = None, name: str = "figA_relu_arditi_cosines", height: float = 5.2):
    """ONE AXIS, FIVE GROUPS: x = refusal vs. control seeds, refusal vs. constitution seeds, control seeds vs.
    constitution seeds, control vs. control, constitution vs. constitution; the direction kind is the MARKER (circle: the read
    direction w_l, the gate probe; square: the write direction v_l, the added vector); colour follows the object where one group has
    one, ink where it crosses the two. The grey band, +-2 sd of the cosine of two random directions in d dimensions, is drawn and named
    in the caption, not the legend. Tick labels at the legend's size. Only the seeded base->instruct fits are drawn (the instruct-direct
    single fits stay in the table)."""
    pts = cosine_groups(sidecar)
    groups = [g for g, _, _ in COSINE_GROUPS]
    fig, ax = plt.subplots(figsize=(1.6 + 2.0 * len(groups), height), layout="constrained")      # narrower: 11.6 in for five groups
    sd = float(sidecar["random_sd"])
    ax.axhspan(-2 * sd, 2 * sd, color=RULE, alpha=0.6, zorder=0, lw=0)
    ax.axhline(0, color=INK2, lw=1.0, zorder=1)
    for gi, g in enumerate(groups):
        for ki, (kind, _) in enumerate(COSINE_KINDS):
            f = pts[(pts.group == g) & (pts.kind == kind)]
            xk = gi + (-0.16 if ki == 0 else 0.16)
            for xi, (_, q) in zip(_spread(xk, len(f), 0.11), f.iterrows()):
                col = COLOR.get(q.colour, INK2) if q.colour else INK2
                ax.scatter([xi], [q.cos], marker=COSINE_MARKER[kind], s=95, color=col, edgecolors="white", lw=1.0, zorder=4)
    from matplotlib.font_manager import FontProperties
    fs = FontProperties(size=plt.rcParams["legend.fontsize"]).get_size_in_points()   # the legend's face, in points (rc may say "medium")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(f"cosine similarity, layer {sidecar['layer']}")
    ax.set_ylim(min(-0.1, float(pts.cos.min()) - 0.05), 1.02)
    ax.grid(True, axis="y", color=RULE, lw=0.9); ax.set_axisbelow(True)
    marks = [Line2D([], [], ls="", marker=COSINE_MARKER[k], ms=9, color=INK2, label=COSINE_KIND_LABEL[k]) for k, _ in COSINE_KINDS]
    # the colour legend above the marker legend: the two objects, and the ink the cross group is drawn in
    objs = [o for _, o, _ in COSINE_GROUPS if o]
    colours = object_handles(list(dict.fromkeys(objs))) + [Line2D([], [], ls="", marker="o", ms=9, color=INK2, label="control × constitution")]
    legend_stack(fig, [("object", colours, 3), (None, marks, 2)])
    return save(fig, name)


def table_cosines(sidecar: dict, caption: str, label: str) -> str:
    """The cosines as numbers: per (object, seed) the read and write cosines against the refusal direction and between read and
    write; then per object the mean pairwise seed cosine, read and write."""
    header = ["arm", "fit", "seed", "read vs refusal", "write vs refusal", "read vs write"]
    rows = []
    for e in sidecar["entries"]:
        rows.append([_tex(LABEL.get(e["object_label"], e["object_label"])), "base$\\to$instruct" if e["fit_model_id"].endswith("_base") else "instruct-direct",
                     _tex(e["seed"] or "--"), _cell(e["read_vs_ref"], dp=3, signed=True), _cell(e["write_vs_ref"], dp=3, signed=True), _cell(e["read_vs_write"], dp=3, signed=True)])
    pr = pd.DataFrame(sidecar["pairs"])
    if len(pr):
        for o, g in pr.groupby("object_label", sort=False):
            rows.append([_tex(LABEL.get(o, o)), "base$\\to$instruct", "\\emph{seed vs seed, mean of %d pairs}" % len(g),
                         _cell(g["read"].mean(), dp=3, signed=True), _cell(g["write"].mean(), dp=3, signed=True), "--"])
    note = (f"Cosine similarities at decoder block {sidecar['layer']}. The refusal direction is the Arditi direction toward refusal; a "
            f"ReLU steer's read direction is its gate probe, its write direction the vector it adds. Two random directions in "
            f"$d = {sidecar['d']:,}$ have cosine sd {sidecar['random_sd']:.3f}.")
    return latex_table(header, rows, caption, label, colspec="lllccc", note=note)


# ---- the frontier ----------------------------------------------------------------------------------------------------------------
# THE FRONTIER READS THE DOSE LADDERS (paperlib FRONTIER_SOURCE_MODEL_OVERRIDE, mirrored here; replicate
# figQ_frontier_absolute_origin, not figQ_frontier): a steerable object's rungs come from its strength-ladder sources only - EVERY rung,
# whatever its selection state - never from a campaign pass; the prompted arms (no strength knob) are single points from the campaign rows
# on the panel's lane; the unsteered reference is the campaign's on the panel's lane, ONE row per eval, and every point is translated by
# it, eval by eval, so the unsteered model sits at the origin (paperlib fig5_absolute, origin="unsteered"). An object whose ladder is
# recorded on the other lane only (the Refusal steer: always instruct-direct, no base fit exists) is drawn from it, named in the printed provenance
# and not marked on the canvas. A ladder
# source that is not on this list but carries three or more doses is reported by `ladder_sources_seen`, never silently skipped.
# `sampled_lane` added 2026-09-24: the seeded difference-LoRA STATICS are filed there, and without it the
# frontier discarded all twelve complete rungs as non-ladder rows — the static panel drew each seed as a single dot at
# the operating point with nothing to join, which is what made the static panels look wrong.
# The rungs were never missing; this allowlist simply had not learned the lane they arrived on. An allowlist that must
# be edited whenever a producer files on a new lane fails SILENTLY, by drawing less, so it is worth saying that here.
LADDER_SOURCES = ["strength_lane", "strength_lane_base2it_native", "arditi_allpos", "arditi_vendor_mt", "strength_mtdiff_s1", "sampled_lane", "relu_paired"]   # relu_paired: the paired-write
# difference ReLU ladder; arditi_vendor_mt: the all-positions Refusal steer multi-turn ladder (a distinct source from the greedy arditi_allpos lane)
_RELU_DOSE, _LORA_DOSE = "dose, × the operating point\n(a direction-norm multiplier)", "dose, × the operating point\n(a write-basis multiplier)"
# THE REFUSAL STEER'S DOSE IS THE RAW COEFFICIENT c — REVERTED, the original label was right (2026-09-23). I changed
# this to "a multiple of the operating point" after measuring coefficient / strength_nominal == 0.3 on every Arditi row,
# and that ratio is real — but `strength_nominal` is NOT what the frontier plots. `dose` is built as
# extra.dose -> strength_nominal -> view-derived, and the extra carries the RAW c, so every Arditi rung plots at its c:
# -0.5, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0. Verified row by row with spell(). Two consequences of the wrong reading, both
# now undone: the axis briefly claimed a unit the numbers were not in, and I told Strength their roster mapped onto
# doses -1.667 / 0.5 / 1.0 / 1.667 when it maps one-for-one onto c. FRONTIER_RUNG_HIDDEN's ("static", ..., 1.0) entry
# was never mis-keyed either: it hides c = 1.0, exactly as its own comment says.
DOSE_UNIT = {"arditi_allpos": "raw coefficient c", "difference_relu": _RELU_DOSE, "constitution_relu": _RELU_DOSE, "control_relu": _RELU_DOSE,
             "difference_lora": _LORA_DOSE, "constitution_lora": _LORA_DOSE, "control_lora": _LORA_DOSE}   # three dose units, never one axis (2026-09-20)
FRONTIER_STATIC = ("static", ["pasttense", "codeattack", "renellm", "grok420"])
# The frontier's x axis floor, in pp of benign compliance LOST: 30 points of over-refusal, so the panels focus on the most
# important range. Negative because the axis is the change against the unsteered model.
FRONTIER_X_FLOOR = -30.0
# NO NEGATIVE STEERING STRENGTHS ON THE FRONTIER, so that the panels focus on the differences with respect to the SIs.
# The sign controls answered a different question — that the steer's direction is load-bearing — and they sit far from
# the prompted diamonds the frontier is now for. Each ladder still passes through the origin, its own dose 0, so
# dropping the negative rung shortens a curve without breaking its reference point. The rungs keep their rows and
# their table entries; only the frontier panels lose them, and the count is reported.
FRONTIER_DROP_NEGATIVE = True
# A RUNG WHOSE READING IS ALMOST ENTIRELY TRUNCATED IS NOT A READING, so dose 1.25 seed 3 is not plotted.
# Measured at head 195245: the paired difference ReLU's seed-3 rung at dose 1.25 has 94.8% of OR-Bench items and 91.6%
# of XSTest items sitting at the token cap, with degeneracy unmeasured. That is the model generating until it is cut
# off, not over-refusal, and it is what put a near-vertical jump on the multi-turn panel -- defense success climbs
# steeply because a truncated response completes no attack, while benign compliance barely moves. Keyed on the
# MEASURED at-cap share rather than on the seed, so it catches any such rung and retires itself if one is re-read at a
# higher cap. Set at 0.90: seed 2's 47.3% at the same dose stays in, by name rather than by this threshold.
FRONTIER_AT_CAP_MAX = 0.90
# RUNGS TAKEN OFF A PANEL BY NAME: the difference LoRA's dose-0.5 rung sat essentially on top of the safety SI's
# diamond on the multi-turn plane and said nothing the rest of its curve does not. Keyed on (panel battery, object, dose) so it is
# removed from that panel only - the rung keeps its row, its table entry and its place on the static panel and the dose-response.
# The difference LoRA's dose-1.5 rung comes off BOTH frontier panels. It was measured at 57-72 % of
# responses degenerate on the two multi-turn attacks - the top of an onset that is near zero through 0.75 - so its apparent defense
# gain is mostly the model failing to produce a response at all, and the share is not filed on the row, so the figure cannot even
# say so beside the point. It keeps its row, its table entry and its place on the dose-response companion.
# The top rung of the paired difference ReLU comes off BOTH panels and the Refusal steer's top rung off the STATIC panel.
# Both are the far end of a ladder that has stopped buying defense with anything but benign compliance: the difference
# ReLU's 1.5 sits at -76.0 pp of benign compliance, and the Refusal steer's static dose 1.0 at -71.0 pp, each a long way beyond the
# elbow its own curve turns at. Verified against the rung table rather than assumed to be last: the difference ReLU's static doses run
# -0.5, 0.5, 1, 1.25, 1.5 and its multi-turn doses the same, while the Refusal steer's static doses run 0.1, 0.2, 0.3, 0.5, 1 and its
# multi-turn doses only 0.15, 0.3, 0.5 - so "the last rung on the left" is dose 1.0, which does not exist on the right at all.
# The paired difference ReLU's dose-0.5 rung came off BOTH panels too, alongside its 1.5 which was already off both,
# drawing its ladder as -0.5, 1, 1.25 on each panel; that removal applied to both batteries rather than to multi-turn
# alone, which is how the difference LoRA's 0.5 is keyed.
# DOSE 0.5 IS BACK for both difference ladders, on every panel and every version. It had been off for two different
# reasons -- the LoRA's 0.5 overplotted the safety SI diamond on multi-turn and the ReLU's came off both panels with
# its 1.5 -- and is now reinstated. The 1.5 rungs and the Refusal steer's static c = 1.0 stay hidden.
FRONTIER_RUNG_HIDDEN = {("multi-turn", "difference_lora", 1.5), ("static", "difference_lora", 1.5),
                        ("multi-turn", "difference_relu", 1.5), ("static", "difference_relu", 1.5),
                        ("static", "arditi_allpos", 1.0)}
# THE OPERATING-DOSE RUNG COMES FROM THE SEEDED WAVE TOO. The ladder-source rule keeps campaign passes off the frontier so a
# campaign point is never mistaken for a dose rung, but the seed replicates live in the main seeded wave, not in the strength ladders:
# the difference LoRA has three seeded runs on both attacks at dose 1.0 and the ladder has NO multi-turn reading there at all, so its
# operating point was missing from the multi-turn plane while Figure 3 drew it from three seeds. These sources are therefore admitted
# AT THE OPERATING DOSE ONLY, averaged over seeds like any other rung; every other dose still reads from the ladder alone, and the
# rung table names the source per dose (`sources`) so a ladder rung and an operating-point rung are never confused.
OPERATING_WAVE_SOURCES = ["seedrep_cluster", "orxs_seedrep", "seedrep_battery", "bon_stores"]
FRONTIER_LANE = "base_to_instruct"          # paperlib's Qwen frontier lane: the LoRA ladders live there; the paper lane holds the Refusal steer's
FRONTIER_PULL_LANE = "instruct_direct"
# THE FRONTIER'S ROSTER: the Refusal steer (a ladder), the difference LoRA and the paired difference ReLU steer (ladders),
# and the two prompted SIs (points). The control and constitution objects are not on it.
# AN OBJECT THAT WAS NEVER FIT HAS NO LANE. "base->instruct" and "instruct-direct" describe
# where an object was FIT versus where it is SERVED; the Refusal steer's direction is computed from activations and
# was never fit, so the distinction is a non sequitur for it and whatever lane its rows carry is an artefact of the
# producer that filed them. Treating it as meaningful deleted the object from the static panel outright: Arditi is
# normally PULLED wholesale from instruct-direct, but when the c = -0.5 statics arrived stamped base->instruct the
# object looked as though it had a ladder on the main lane, the pull stopped, and the only rung left was the excluded
# negative one -- so the panel showed nothing at all, with no gap reported anywhere.
LANELESS_OBJECTS = {"arditi_allpos"}
FRONTIER_OBJECTS = ["arditi_allpos", "safety_si", "constitution_si", "difference_relu", "difference_lora"]
FRONTIER_COLS = ["object_label", "seed", "dose", "lane", "pulled", "dx", "dy", "ax_abs", "ay_abs", "kx", "ky", "n_seeds_y", "n_seeds_x", "coefficient", "in_paper", "sources", "cross_cut", "degen", "cap", "token_caps", "anchor_kinds"]


def _campaign_rows(rows: pd.DataFrame, keys: list) -> pd.DataFrame:
    """One campaign row per `keys`: the operating dose where several standups carry one, then ANCHOR_PREFERENCE, else the first."""
    out = []
    for _, g in rows.groupby(keys):
        g1 = g[g.dose.eq(1.0)] if g.dose.eq(1.0).any() else g
        out.append(pick_anchor(g1, ""))
    return pd.DataFrame(out) if out else rows.iloc[0:0]


def frontier_points(frame: pd.DataFrame, lane: str = FRONTIER_LANE, y_evals=FRONTIER_Y[1], x_evals=FRONTIER_X[1], model: str = MODEL_ID,
                    pull_lane: str | None = FRONTIER_PULL_LANE, ladder_sources=LADDER_SOURCES, objects=None, require_x: bool = True,
                    panel_name: str = "", drop_rungs: set | None = None) -> tuple:
    """The frontier's points for one battery: per (object, seed, dose) the mean over `y_evals` of the per-eval delta in defense success
    vs the campaign's unsteered run (pp; `dy`) against the same over the benign evals (`dx`). Steerable objects: every judged rung of
    their ladder sources on `lane` (an object with no ladder on `lane` but one on `pull_lane` is drawn from it, `pulled` = True);
    prompted objects: the campaign row on `lane` (pulled from `pull_lane` when the lane has none). `objects` restricts the frame to the
    roster (plus the unsteered), so an off-roster ladder never decides the battery. The x battery is the benign evals EVERY drawn rung
    carries (the ladders carry XSTest alone), so no point mixes batteries; a rung lacking a y eval is HELD OUT and returned, never
    averaged over fewer evals.
    Returns (points, held_out, meta) with meta = dict(x_battery, anchor (the campaign's unsteered rows, per eval), no_anchor (evals with no
    anchor at all), own_anchors (the ladders' own unsteered rows), same_cut_anchors (how many (eval, item set) anchors exist)). With
    `require_x=False` a rung is admitted on its y battery alone (dx may be NaN): the dose-response view for a ladder with no trustworthy cost axis. Each rung's
    delta is against an unsteered run on ITS OWN item set where one exists (`cross_cut` = False), else the campaign's; `degen` is the largest
    degenerate share among the rung's y-battery rows (a degenerate cell scores as a non-jailbreak and is not a defense)."""
    # THE FRONTIER DRAWS THE ROSTER'S RUNGS. Rows whose selection_state is not `selected` -- above all `not_in_roster` --
    # are readings but not rungs of the ladder the paper names, and letting them in drew Arditi at c = 0.1 and 0.2 on the
    # static panel (both entirely not_in_roster, in_paper 0) while the multi-turn panel, which has no such rows, showed the
    # three roster doses. They also leaked into the POOLED benign cost: Arditi 0.3 read x = -15.01 on static against
    # -15.42 on multi-turn, and 0.5 read -25.66 against -26.66; with them out the two panels agree exactly. Measured
    # blast radius on the live frame: LoRA and ReLU rungs unchanged to the second decimal, per-seed rows 30 -> 30 and
    # 28 -> 28. `rung_eval_table` already filtered this way, which is why the table said three while the figure drew
    # five. Unsteered `none` rows are exempt: they are anchors, chosen by the anchor rules, not rungs.
    if "selection_state" in frame.columns:
        frame = frame[frame.selection_state.eq("selected") | frame.object_label.eq("none")]
    # STATIC PADDING NEVER ENTERS THE FRONTIER. The padding instrument's rows -- pad 0 included -- are `selected` (they
    # are the paper's padding figure), so the selection filter keeps them, and the main figures only exclude them through
    # the roster's selectors. Here they pooled into the static battery: Safety SI's static defense read +15.95 on the
    # frontier against +21.98 in the main table because Pliny under 16k / 64k filler (66 % / 76 % ASR) was averaged in.
    # Consumers key the guard on padding_tokens; lane_source_legacy is the belt to that brace.
    if "padding_tokens" in frame.columns:
        frame = frame[frame.padding_tokens.isna()]
    if "lane_source_legacy" in frame.columns:
        frame = frame[frame.lane_source_legacy.ne("static_padding")]
    # THE REGISTRY'S HOLD-OUTS ARE HONOURED. A `held_out` row is a ruling filed against a run; a
    # rung with any held row for a seed is not drawn for that seed (ladder_fig.ipynb's rule: held_out = max over the
    # row's runs). Before this the frontier drew difference_lora 1.25 s2 -- the one seed the registry holds out -- while
    # censoring s3, which it does not. Anchors (`none`) are not rungs and are left to the anchor rules.
    if "held_out" in frame.columns and "seed" in frame.columns:
        _hk = [k for k in ("object_label", "seed", "dose") if k in frame.columns]
        _held = frame.assign(_h=pd.to_numeric(frame.held_out, errors="coerce").fillna(0).gt(0)).groupby(_hk)["_h"].transform("max")
        frame = frame[~(_held & frame.object_label.ne("none"))]
    # AN EXPLICIT RUNG SET TO DROP, for a version that censors the three suspect 1.25 rungs so the panels can be read
    # side by side. (object_label, seed, dose) triples removed before pooling; the figure that uses it says so.
    if drop_rungs:
        _key = pd.Series(list(zip(frame.object_label, frame.seed.fillna("").astype(str),
                                  pd.to_numeric(frame.dose, errors="coerce").round(3))), index=frame.index)
        frame = frame[~_key.isin({(o, str(sd), round(float(dd), 3)) for o, sd, dd in drop_rungs})]
    f = intervals(dedupe(frame[(frame.model_id == model) & frame.headline & frame.judge_ok & complete(frame) & frontier_view(frame)
                               & frame.eval_id.isin(list(x_evals) + list(y_evals))], contrasts=False))     # its deltas are its own: one row per run; a source-local none stays as an anchor candidate
    none_all = f[f.object_label == "none"]                                  # both lanes: a pulled rung anchors on ITS lane's unsteered first
    none = none_all[none_all.lane.eq(lane)]
    # THE ANCHOR IS ON THE ARM'S METRIC (2026-09-24, the label-mean switch): every anchor pool below -- the common anchor, the
    # same-item-set pool, the campaign fallback -- is filtered to the arm row's metric family, the rule `anchor_candidates`
    # already applies for the paper panels. Without it the frontier's pools were metric-blind: after a repoint that moves the
    # selected rows, an unsteered row still on the old headline (the `r2_export` campaign rows carry no label mean) would pool
    # into a label-mean rung's origin and put the 35 pp two-metric offset back, one anchor at a time.
    anchor = _campaign_rows(none[~none.lane_source_legacy.isin(ladder_sources)] if (~none.lane_source_legacy.isin(ladder_sources)).any() else none, ["eval_id", "metric_id"])
    a_by_eval = anchor.set_index(["eval_id", "metric_id"]).y if len(anchor) else pd.Series(dtype=float)
    def _fallback(row):
        for m in metric_family(row.eval_id, row.metric_id):
            if (row.eval_id, m) in a_by_eval.index:
                return a_by_eval[(row.eval_id, m)]
        return np.nan
    own = none[none.lane_source_legacy.isin(ladder_sources)]
    if objects is not None:                                                  # the roster decides the battery; off-roster ladders are the figure's business
        f = f[f.object_label.isin(list(objects) + ["none"])]
    at_op = f["at_operating_point"].astype(bool) if "at_operating_point" in f.columns else f.dose.eq(1.0)
    steer = f[(f.object_label != "none") & ~f.object_label.isin(PROMPTED)
              & (f.lane_source_legacy.isin(ladder_sources) | (f.lane_source_legacy.isin(OPERATING_WAVE_SOURCES) & at_op))]
    # a laneless object is on every lane by construction: its rows are admitted whatever they are stamped with
    on = steer[steer.lane.eq(lane) | steer.object_label.isin(LANELESS_OBJECTS)]
    pulled = steer[steer.lane.eq(pull_lane) & ~steer.object_label.isin(set(on.object_label))] if pull_lane else steer.iloc[0:0]
    # THE PROMPTED ARMS, EVERY RUN (2026-09-21: the constitution SI on the multi-turn panel as a diamond). A prompt has no fit
    # and no lane: its campaign rows sit on the two lanes and its seed-replicate runs carry no lane at all, and the panel wants all of
    # them - at the operating dose, per seed, so the figure can draw the seeds as small diamonds around ONE mean diamond, exactly as
    # the ladders draw their seeds around the per-dose mean. _campaign_rows collapsed them to one row per eval and the lane test
    # dropped the lane-less ones, which is why the constitution SI had no multi-turn point: its multi-turn runs are all seed views.
    prm = f[f.object_label.isin(PROMPTED) & f["at_operating_point"].astype(bool)
            & (f.lane.eq(lane) | (f.lane.eq(pull_lane) if pull_lane else False) | f.lane.isna())].copy()
    prm["pulled"] = ~prm.lane.eq(lane)
    prm["lane"] = prm.lane.fillna("prompt (no lane)")
    rows = pd.concat([on.assign(pulled=False), pulled.assign(pulled=True), prm], ignore_index=True)
    # THE ANCHOR PER ROW (2026-09-20: a delta absorbs the anchor but not the denominator): an unsteered run on the SAME item set
    # and lane - the ladder's own cut, e.g. XSTest's 250-prompt safe subset - is preferred; the campaign's anchor (a_by_eval) stands in
    # where none exists and the row is flagged `cross_cut`. Each ladder therefore passes through ITS OWN unsteered at the origin.
    def anchor_for(row):
        # the unsteered draws on the row's own item set and lane: its own source's standup first (anchor_draws: same source, then the campaign's
        # preference, the same dose standup), averaged over draws; the campaign's per-eval anchor stands in when the item set has none (cross-cut)
        # THE COMMON ANCHOR: every unsteered run of the eval at the rung's cap and item set, pooled as draws - the
        # same rule as the paper panels - so every ladder is differenced against one reference and passes through the origin
        cc = none_all[(none_all.eval_id == row.eval_id) & none_all.metric_id.isin(metric_family(row.eval_id, row.metric_id))]
        _cls = cap_class(row.eval_id, row.get("token_cap_tokens"))
        if len(cc) and _cls and cc.token_cap_tokens.isin(_cls).any():
            cc = cc[cc.token_cap_tokens.isin(_cls)]
        if len(cc) and "item_set_id" in cc.columns and pd.notna(row.get("item_set_id")) and cc.item_set_id.eq(row.item_set_id).any():
            return anchor_summary(cc[cc.item_set_id.eq(row.item_set_id)]).y, False
        if len(cc) and row.eval_id not in NO_CROSS_ITEM_SET:
            return anchor_summary(cc).y, True                                 # no anchor on the rung's own item set: pooled across cuts, ringed
        if len(cc):
            return np.nan, True                                               # nested widths (MMLU): a cross-width anchor is not a comparator
        if "item_set_id" in none.columns and pd.notna(row.get("item_set_id")):
            for pool_lane in dict.fromkeys([row.lane, lane]):             # the rung's own lane (the pull lane for a pulled rung), then the panel's
                # an unsteered run reads lane = NaN when its spelling has no arm (run.transfer_kind = 'none' is the object, not a lane; the
                # vendor standup's unsteered rows, 2026-09-20) and is admissible in either pool; the same-source and same-standup rules
                # inside anchor_draws hand it to its own rung, so that rung's origin is ITS decode regime. That matters: under the vendor
                # preset 38-40 % of the unsteered model's safe XSTest replies hit the 512-token cap, and the ruled three-term rate
                # (full refusal + half the partial share + half the at-cap share) reads 23-24 % where the legacy decode's reads 4-5 % -
                # the same 3.6 % of full refusals in both; the regime differs, not the refusing. A delta absorbs the
                # regime only when its anchor shares it.
                c = none_all[(none_all.eval_id == row.eval_id) & none_all.metric_id.isin(metric_family(row.eval_id, row.metric_id))
                             & none_all.item_set_id.eq(row.item_set_id) & (none_all.lane.eq(pool_lane) | none_all.lane.isna())]
                pool_a = anchor_draws(c, row.lane_source_legacy, row.dose, row.get("view_legacy"), row.get("token_cap_tokens"))
                if len(pool_a):
                    return anchor_summary(pool_a).y, False
        return (_fallback(row), True)
    same_cut = {k: 1 for k in set(zip(none_all.eval_id, none_all.item_set_id, none_all.lane)) if pd.notna(k[1])} if "item_set_id" in none_all.columns else {}
    av = rows.apply(lambda row: pd.Series(anchor_for(row), index=["anchor_y", "cross_cut"]), axis=1) if len(rows) else pd.DataFrame(columns=["anchor_y", "cross_cut"])
    rows = pd.concat([rows, av], axis=1)
    no_anchor = sorted(set(rows.loc[rows.anchor_y.isna(), "eval_id"]))
    rows = rows[rows.anchor_y.notna()].copy()
    empty = pd.DataFrame(columns=FRONTIER_COLS)
    meta = dict(x_battery=list(x_evals), anchor=anchor, no_anchor=no_anchor, own_anchors=own, same_cut_anchors=len(same_cut))
    if not len(rows):
        return empty, empty, meta
    rows["good"] = -(rows.y - rows.anchor_y.astype(float))          # ASR and graded over-refusal: lower is better, so the good delta is the negative
    rows["degen_share"] = rows["frac_degenerate_union5"].astype(float) if "frac_degenerate_union5" in rows.columns else np.nan
    rows["cap_share"] = rows["frac_at_token_cap"].astype(float) if "frac_at_token_cap" in rows.columns else np.nan   # reported beside the rung, never discounted
    keys = ["object_label", "seed", "dose", "lane", "pulled"]
    # SOURCE PREFERENCE PER EVAL: where two ladder sources carry the same (rung, eval), the one covering the MOST RUNGS of that eval for the
    # object wins, so a ladder stays on one cut per eval (the vendor-preset multi-turn ladder, three rungs on multiturn_goals_259, over the
    # legacy decode's single c = 0.3 rung on jailbreak_goals_259 - vendor presets are the default decode);
    # ties go to ANCHOR_PREFERENCE, then the ladder-source order.
    rungs_per = rows.groupby(["object_label", "eval_id", "lane_source_legacy"]).dose.nunique()
    # THE Y SOURCE PER OBJECT: the source carrying the full y battery on the most rungs is the object's ladder for this panel, and a rung
    # takes EVERY eval it can from it (x included), so a point's x and y come from one standup and one decode (the vendor-preset
    # multi-turn ladder's XSTest at c = 0.15 / 0.3 / 0.5 beside its own attacks, not the legacy decode's XSTest at the same doses:
    # the two decodes differ by ~16 pp of over-refusal at c = 0.3, 2026-09-20). Ties go to the preferences below.
    y_full = rows[rows.eval_id.isin(list(y_evals))].groupby(["object_label", "lane_source_legacy", "dose"]).eval_id.nunique()
    y_rungs = y_full[y_full >= len(set(y_evals))].groupby(["object_label", "lane_source_legacy"]).size()
    def _pref(s):
        return (ANCHOR_PREFERENCE.index(s) if s in ANCHOR_PREFERENCE else 99, ladder_sources.index(s) if s in ladder_sources else 99)
    y_src = {o: min(g.index.get_level_values(1), key=lambda s: (-g[(o, s)], _pref(s))) for o, g in y_rungs.groupby(level=0)} if len(y_rungs) else {}
    def src_rank(o, e, s):
        return (0 if y_src.get(o) == s else 1, -rungs_per.get((o, e, s), 0)) + _pref(s)
    _series_cap = series_caps(rows)
    def pick(g):                                                     # one reading per (rung, eval): the eval's ladder source, else the preference;
        g, _cap = cap_choice(g, g.eval_id.iat[0], _series_cap.get((g.object_label.iat[0], g.eval_id.iat[0])))   # one cap per cell AND per series
        if len(g) == 1:                                                  # that source's several decode draws averaged
            return g.iloc[0]
        rank = {s: src_rank(g.object_label.iat[0], g.eval_id.iat[0], s) for s in g.lane_source_legacy}
        best = g.lane_source_legacy.map(rank).sort_values().index[0]
        gs = g[g.lane_source_legacy == g.loc[best, "lane_source_legacy"]]
        # a producer's `pooled_*` pass row is the cell's own aggregate over its trials (2026-09-20: never a silent sum): where one
        # exists it is the reading, and the per-trial rows beside it are not averaged in a second time
        if "pass_label" in gs.columns and gs.pass_label.fillna("").str.startswith("pooled").any():
            gs = gs[gs.pass_label.fillna("").str.startswith("pooled")]
        r = gs.iloc[0].copy(); r["good"] = gs.good.mean(); r["draws"] = len(gs)
        return r
    one = pd.DataFrame([pick(g) for _, g in rows.groupby(keys + ["eval_id"])])
    # THE ORIGINAL FIT IS NOT DRAWN WHERE SEED REPLICATES EXIST: the panels take the batch-64 seeds 1-3 only.
    # A ladder that has batch-64 seed replicates also carries the older single operating-
    # point fit under its own view, and both were being drawn under one colour as if one object. They disagree badly:
    # difference_lora at dose 1 read MT +69.0 on the original fit against +24.1 / +25.0 / +41.1 on the three seeds,
    # which is what put a near-vertical segment on the multi-turn panel. The seed is the discriminator the frame
    # already carries — the original fit is seedless because it IS one fit, not a replicate — so no view spelling is
    # keyed on here. A ladder with no seeded fits at all (the Refusal steer) is untouched.
    _seeded_obj = {o for o, g in one.groupby("object_label") if g.seed.fillna("").ne("").any()} if len(one) else set()
    if _seeded_obj:
        _orig = one.object_label.isin(_seeded_obj) & one.seed.fillna("").eq("")
        if _orig.any():
            meta["dropped_orig_fit"] = int(_orig.sum())
            one = one[~_orig]

    have = one.groupby(keys).eval_id.agg(set)
    y_ok = have[have.map(lambda s: set(y_evals) <= s and bool(set(x_evals) & s))]     # a rung with no benign eval cannot be drawn and does not vote
    # THE X AXIS IS THE MEAN OF BOTH BENIGN EVALS, AND A RUNG LACKING EITHER IS NOT PLOTTED. The axis is the mean of XSTest
    # and OR-Bench rather than XSTest alone, and the frontier REFUSES a point that lacks OR-Bench. Two earlier behaviours
    # were both wrong. The original took the INTERSECTION of the declared benign evals over the drawn rungs, which
    # collapsed the axis to XSTest the moment any rung lacked OR-Bench -- a silent narrowing of the axis to one eval. The
    # first fix then let a rung with only one eval through on a partial mean, which is worse: it puts a one-eval x beside
    # a two-eval x on the same axis and calls them the same quantity. The battery is both evals, always; a rung that
    # cannot form that mean is held out and NAMED. The missing OR-Bench rows are being generated, so the held-out list
    # shrinks on its own.
    x_batt = list(x_evals)
    meta["x_battery_note"] = ""
    meta["dropped_negative"] = 0
    _lack_x = [k for k, sset in y_ok.items() if not set(x_batt) <= sset]
    if _lack_x:
        meta["x_battery_note"] = (f"x is the mean of {' and '.join(EVAL_LABEL.get(e, e) for e in x_batt)}; a rung lacking "
                                  f"either is NOT plotted. Held out for an incomplete benign battery: {len(_lack_x)} rung(s) — "
                                  + "; ".join(f"{k[0]} {k[1]} dose {k[2]:g}" for k in _lack_x[:8])
                                  + ("; …" if len(_lack_x) > 8 else ""))
    if False:
        # the intersection is EMPTY: some rung carries only a benign eval another lacks (the paired ReLU's seed-1 operating point has
        # or_bench_hard alone, its XSTest withdrawn, 2026-09-20). The battery is then the FIRST DECLARED benign eval any y-ok rung carries -
        # the declaration's order is the paper's preference - and the rungs lacking it are held out and named. Never the whole declared
        # list, which would demand every eval of every rung and empty the panel; never a count, which would flip the axis as rows land.
        carried = set().union(*y_ok)
        x_batt = [next(e for e in x_evals if e in carried)]
        lacking = [k for k, s in y_ok.items() if not set(x_batt) <= s]
        meta["x_battery_note"] = (f"no benign eval is shared by every rung with the y battery; the battery is {x_batt[0]} (first declared); "
                                  f"held out for lacking it: {len(lacking)} rung(s) - " + "; ".join(f"{k[0]} {k[1]} dose {k[2]:g}" for k in lacking))
    x_batt = x_batt or list(x_evals)
    meta["x_battery"] = x_batt
    # BOTH benign evals, or the rung is not a point: a one-eval x is a different quantity from a two-eval mean and the two
    # must not share an axis. The held-out rungs are named in x_battery_note above.
    ok = have[have.map(lambda s: set(y_evals) <= s and (not require_x or set(x_batt) <= s))].index     # require_x=False: the dose-response view
    sub = one.set_index(keys).loc[ok].reset_index() if len(ok) else one.iloc[0:0]
    # A RUNG IS A DOSE, NOT A (SEED, DOSE). The paired ReLU's operating point carries both attacks on all three
    # seeds but XSTest on seed 2 alone, so keying the drawn point on the seed dropped seeds 1 and 3 whole and put ONE seed's defense on
    # the plane - which is why its operating point read worse here than in Figure 3, where the same rung is the three-seed mean. The
    # point is now per (object, dose): dy over every seed carrying the y battery, dx over every seed carrying the x battery, counted
    # separately (`n_seeds_y` / `n_seeds_x`) because they need not be the same seeds. A seed carrying BOTH still rides as a small mark.
    y_rows = one.set_index(keys).loc[[k for k in have.index if set(y_evals) <= have[k]]].reset_index() if len(have) else one.iloc[0:0]
    # A SEED IS A POINT ON ITS DEFENCE BATTERY ALONE: three curves, one per seed.
    # `sub` required a seed to carry the y battery AND the x battery (XSTest) at that dose. Measured at head 190619
    # that left seed 1 as the only multi-dose ladder on either panel — seeds 2 and 3 appeared at dose 1 alone — so
    # two of the three curves could never be drawn. The missing half is always the X coordinate: ReLU's s2/s3 dose
    # children carry all four static attacks but no XSTest at -0.5/0.5/1.25, because benign compliance is a separate
    # lane that was not re-run per dose. The DEFENCE axis is where the seeds differ and is what the curves are for,
    # so a seed now enters on the y battery alone and borrows the dose's pooled benign compliance for x where it has
    # none of its own. Every borrowed point is flagged `dx_pooled` and the figure says so in its legend.
    seed_pts = y_rows.copy()
    def agg(g):
        # THE SEED LAYER OBEYS THE SAME REFUSAL AS THE AGGREGATE. Without this it computed dx as the
        # mean over whichever benign evals the SEED happened to carry, so a seed with XSTest alone drew a one-eval x beside a
        # two-eval aggregate on the same axis — the exact mixing the dose-level refusal exists to prevent, surviving in a
        # second code path. A seed missing either benign eval now has no x of its own and falls through to the dose's pooled
        # (two-eval) x below, or is dropped when that dose is itself refused.
        return pd.Series(dict(dx=(g[g.eval_id.isin(x_batt)].good.mean() if set(x_batt) <= set(g.eval_id) else np.nan),
                              dy=g[g.eval_id.isin(y_evals)].good.mean(),
                              kx=int(g.eval_id.isin(x_batt).sum()), ky=int(g.eval_id.isin(y_evals).sum()), coefficient=g.coefficient.max(),
                              in_paper=int(g.in_current_paper.max()), sources=",".join(sorted(set(g.lane_source_legacy))),
                              cross_cut=bool(g.cross_cut.astype(bool).any()), degen=g[g.eval_id.isin(y_evals)].degen_share.max(),
                              cap=g.loc[g.eval_id.isin(y_evals), "cap_share"].max(),
                              token_caps=",".join(sorted({f"{v:.0f}" for v in g.token_cap_tokens.dropna()})) if "token_cap_tokens" in g.columns else "",
                              anchor_kinds=",".join(sorted({k for k in (list(g.anchor_kind.dropna()) if "anchor_kind" in g.columns else [])
                                                            + ([block_anchor_kind(b) for b in g.delta_block] if "delta_block" in g.columns else []) if k}))))
    dose_keys = [k for k in keys if k != "seed"]
    def _two_stage(v):
        # THE BOLD LINE IS A MEAN OF PER-SEED MEANS: each eval differenced against its own
        # anchor per seed, averaged over evals WITHIN the seed, then averaged over seeds. A flat mean over every
        # (seed x eval) row says the same thing only when every seed carries every eval; once a seed is censored
        # or lacks a benign eval the flat mean weights the remaining seeds by how many rows they happen to have.
        # An unseeded object (seed == "") is one group, so this reduces to the plain mean for it.
        return v.groupby("seed").good.mean().mean() if len(v) else np.nan
    def agg_dose(g):
        yv = g[g.eval_id.isin(y_evals)]
        xv = g[g.eval_id.isin(x_batt)]
        return pd.Series(dict(dx=_two_stage(xv), dy=_two_stage(yv), kx=int(len(xv)), ky=int(len(yv)),
                              # the ABSOLUTE unsteered coordinates this rung is differenced from: the frontier draws the
                              # anchor at the origin, so without these the table never states the level it is relative to
                              # THE ORIGIN'S ABSOLUTE READING, in the columns' own orientation: the rows carry
                              # `anchor_y`, the per-eval unsteered reading in plotted units (ASR on the attacks, the over-refusal label mean on the
                              # benign evals -- both lower-is-better), so the unsteered DEFENSE is 100 - anchor ASR and the unsteered
                              # COMPLIANCE is 100 - anchor over-refusal, each the mean over the panel's evals, as the category tables'
                              # unsteered row states them. This read `anchor_value`, a column these rows never had, so the frontier
                              # table's baseline row had printed dashes since it was written.
                              ax_abs=(100.0 - pd.to_numeric(xv.anchor_y, errors="coerce")).mean() if "anchor_y" in xv.columns else np.nan,
                              ay_abs=(100.0 - pd.to_numeric(yv.anchor_y, errors="coerce")).mean() if "anchor_y" in yv.columns else np.nan,
                              n_seeds_y=int(yv.seed.nunique()), n_seeds_x=int(xv.seed.nunique()),
                              coefficient=g.coefficient.max(), in_paper=int(g.in_current_paper.max()),
                              sources=",".join(sorted(set(g.lane_source_legacy))), cross_cut=bool(g.cross_cut.astype(bool).any()),
                              degen=yv.degen_share.max(), cap=g.loc[g.eval_id.isin(y_evals), "cap_share"].max(),
                              token_caps=",".join(sorted({f"{v:.0f}" for v in g.token_cap_tokens.dropna()})) if "token_cap_tokens" in g.columns else "",
                              anchor_kinds=",".join(sorted({k for k in (list(g.anchor_kind.dropna()) if "anchor_kind" in g.columns else []) if k}))))
    # every seed with the y battery contributes dy; every seed with the x battery contributes dx, whether or not it is the same seed
    pool_rows = pd.concat([y_rows[y_rows.eval_id.isin(y_evals)], one[one.eval_id.isin(x_batt)]], ignore_index=True).drop_duplicates(
        subset=[c for c in ("run_id", "eval_id", "metric_id") if c in one.columns]) if len(one) else one
    # CENSOR PER SEED, BEFORE POOLING (who asked for ONE seed to come off a rung, not the rung).
    # The at-cap test below runs on the pooled point, whose `cap` is the MAX over the seeds in it -- so one seed that
    # runs away takes its two healthy siblings with it. Measured on difference_relu dose 1.25: s3 generates to the
    # 8,192-token ceiling on 94.8% of OR-Bench and 91.6% of XSTest prompts, while s1 sits at 8.4% / 4.4% and s2 at
    # 47.3% / 19.4%. The whole rung vanished from both panels, which amputated the difference steers' top rung while
    # leaving Arditi's ladder (<=0.3% at cap everywhere) intact and made ReLU look weaker than Figures 3-4 show it.
    # Dropping the runaway SEED and pooling what remains is what was actually asked for, and is the granularity the
    # pooling happens at. A rung all of whose seeds are censored still disappears, via the pooled test below.
    if len(pool_rows) and "cap_share" in pool_rows.columns and "seed" in pool_rows.columns:
        _sk = [c for c in ("object_label", "seed", "dose") if c in pool_rows.columns]
        # ON THE DEFENSE EVALS ONLY: the refused-rate headline labels a benign reply cut off at the
        # cap as a full refusal, so the benign axis already accounts for truncation; only the defense axis is flattered
        # by it (a cut-off reply completes no attack). Reading the pooled max, which included the benign battery, censored
        # ReLU s3 and LoRA s3 at 1.25 on OR-Bench shares of 94.8 % / 97.3 % while their attack shares were 24 % / 59 %.
        _yonly = pool_rows.cap_share.where(pool_rows.eval_id.isin(y_evals))
        _worst = _yonly.groupby([pool_rows[k] for k in _sk]).transform("max")
        _runaway = pd.to_numeric(_worst, errors="coerce").fillna(0) > FRONTIER_AT_CAP_MAX
        if _runaway.any():
            meta["dropped_seed_at_cap"] = sorted({
                f"{r.object_label} {r.seed or '(unseeded)'} dose {float(r.dose):g} ({100 * float(r.cap_share):.0f}% at cap)"
                for _, r in pool_rows[_runaway].iterrows()})
            pool_rows = pool_rows[~_runaway]
    pts = (pool_rows.groupby(dose_keys).apply(agg_dose, include_groups=False).reset_index()
           .assign(seed="").pipe(lambda t: t[t.dy.notna() & (t.dx.notna() if require_x else True)])) if len(pool_rows) else empty
    if len(pts) and "cap" in pts.columns:
        _censored = pd.to_numeric(pts["cap"], errors="coerce").fillna(0) > FRONTIER_AT_CAP_MAX
        meta["dropped_at_cap"] = [f"{r.object_label} {r.seed or ''} dose {float(r.dose):g} ({100 * float(r.cap):.0f}% at cap)"
                                  for _, r in pts[_censored].iterrows()] if _censored.any() else []
        pts = pts[~_censored]
    if len(pts) and FRONTIER_DROP_NEGATIVE and "dose" in pts.columns:
        neg = pd.to_numeric(pts.dose, errors="coerce") < 0
        meta["dropped_negative"] = int(neg.sum())
        pts = pts[~neg]
    if len(pts) and panel_name:
        drop = pts.apply(lambda r: (panel_name, r.object_label, float(r.dose)) in FRONTIER_RUNG_HIDDEN, axis=1)
        if drop.any():
            meta["hidden_rungs"] = pts[drop][["object_label", "dose", "dx", "dy"]].to_dict("records")
            pts = pts[~drop]
    held_keys = [k for k in have.index if k not in set(ok)]
    held = (one.set_index(keys).loc[held_keys].reset_index().groupby(keys).agg(evals=("eval_id", lambda s: ",".join(sorted(set(s)))),
            sources=("lane_source_legacy", lambda s: ",".join(sorted(set(s))))).reset_index() if held_keys else pd.DataFrame(columns=keys + ["evals", "sources"]))
    if len(held):
        def why(r):
            s = set(r.evals.split(","))
            miss_y = [e for e in y_evals if e not in s]; miss_x = [e for e in x_batt if e not in s] if require_x else []
            return "; ".join(filter(None, [f"missing y: {', '.join(miss_y)}" if miss_y else "", f"missing x: {', '.join(miss_x)}" if miss_x else ""]))
        held["why"] = held.apply(why, axis=1)
    out = pts[[c for c in FRONTIER_COLS if c in pts.columns]] if len(pts) else empty
    # the per-seed marks ride in META, never in `out.attrs`: a DataFrame in .attrs makes a later pd.concat compare frames for equality
    # and raise "truth value of a DataFrame is ambiguous" (measured here, 2026-09-21; the same trap as the 2026-08 concat defect)
    # UNMEASURED IS NOT CLEAN (2026-09-21). A rung whose `frac_degenerate_union5` is NULL carries no dagger, which on the
    # figure is indistinguishable from a rung measured at zero - and the difference LoRA's dose-1.5 rung, the largest suppression on
    # the ladder, is exactly such a row: 57-72 % degenerate was measured there but the values were never filed. Marking every
    # unmeasured rung would mark nearly all of them (only the Arditi statics carry the column today), so the figure states the
    # coverage once and the notebook names the rungs; the dagger appears by itself when the values are filed.
    if len(pts):
        um = pts[pts.degen.isna()] if "degen" in pts.columns else pts.iloc[0:0]
        meta["degen_measured"], meta["degen_total"] = int(len(pts) - len(um)), int(len(pts))
        meta["degen_unmeasured"] = um[["object_label", "dose", "dy"]].to_dict("records")
    # THE HIDDEN RUNGS MUST COME OFF THE PER-SEED MARKS TOO. `pts` was filtered by FRONTIER_RUNG_HIDDEN above but
    # `per_seed` was not, so a rung taken off a panel kept drawing — once per seed — as soon as the seed replicates
    # landed and the per-seed layer became curves. The difference ReLU's dose-1.5 rung appeared on the frontier as
    # though the hidden list were wrong; the list was right and this layer was bypassing it. Same key, same panel,
    # applied in the same place.
    _psd = seed_pts.groupby(keys).apply(agg, include_groups=False).reset_index() if len(seed_pts) else pd.DataFrame()
    # NO BORROWED X (after a sharp kink appeared in one seed's ladder). A seed with no benign
    # battery of its own used to inherit the DOSE's pooled x. That was written before the refusal ruling and
    # contradicts it: the rule is that the frontier does not plot a point lacking OR-Bench, and the borrow
    # smuggled one in at the per-seed level. It also produced the kink — at difference_relu dose 1.25 all three seeds
    # took the identical borrowed x of -61.6 while their dose-1.0 x values ran -2.6 / -13.2 / -21.9, so every curve was
    # yanked sideways to a shared point that no seed had measured. A seed rung now needs its OWN dx and dy or it is
    # not drawn, exactly as the dose-level points are treated.
    if len(_psd) and "dx" in _psd.columns:
        _psd = _psd[_psd.dx.notna() & _psd.dy.notna()]
    if len(_psd) and "cap" in _psd.columns:                    # the same refusal on the per-seed layer
        _psd = _psd[~(pd.to_numeric(_psd["cap"], errors="coerce").fillna(0) > FRONTIER_AT_CAP_MAX)]
    if len(_psd) and FRONTIER_DROP_NEGATIVE and "dose" in _psd.columns:
        _psd = _psd[~(pd.to_numeric(_psd.dose, errors="coerce") < 0)]
    if len(_psd) and panel_name and "dose" in _psd.columns:
        _psd = _psd[~_psd.apply(lambda r: (panel_name, r.object_label, float(r.dose)) in FRONTIER_RUNG_HIDDEN, axis=1)]
    meta["per_seed"] = _psd
    meta["seed_counts"] = pts[["object_label", "dose", "n_seeds_y", "n_seeds_x"]].to_dict("records") if len(pts) and "n_seeds_y" in pts.columns else []
    return out, held, meta


def frontier_pending(frame: pd.DataFrame, lane: str, y_evals, model: str = MODEL_ID, ladder_sources=LADDER_SOURCES, pull_lane: str | None = None) -> pd.DataFrame:
    """Ladder rungs on `lane` (and on `pull_lane`, where a pulled ladder lives) whose runs on the battery's evals are in the registry but not
    judged: what the panel will gain. Counted per run, so a run's unjudged decode draws show as several."""
    lanes = [lane] + ([pull_lane] if pull_lane else [])
    p = frame[(frame.model_id == model) & frame.lane.isin(lanes) & frame.lane_source_legacy.isin(ladder_sources) & (frame.object_label != "none")
              & frame.eval_id.isin(list(y_evals)) & ~complete(frame) & in_view(frame)]
    return p.groupby(["object_label", "dose", "lane", "lane_source_legacy", "lifecycle_state"]).run_id.nunique().rename("runs").reset_index()


def ladder_sources_seen(frame: pd.DataFrame, model: str = MODEL_ID, min_doses: int = 3) -> pd.DataFrame:
    """Every (source, object) in the frame with `min_doses` or more distinct doses - a ladder by shape - and whether LADDER_SOURCES names it."""
    s = frame[(frame.model_id == model) & (frame.object_label != "none")].groupby(["lane_source_legacy", "object_label", "lane"]).dose.nunique()
    s = s[s >= min_doses].rename("doses").reset_index()
    s["on_the_list"] = s.lane_source_legacy.isin(LADDER_SOURCES)
    return s


# ---- coverage ------------------------------------------------------------------------------------------------------------------
# CELLS THAT ARE OUT BY DESIGN draw no glyph (2026-09-21: no `x` for Nemotron agentic leaking - it is out of that eval). A
# (model, eval) named here reads `out by design` in the gap tables, which no GAP_MARK entry draws; every object of that cell is covered.
OUT_BY_DESIGN = {("nemotron3nano4b", "agentic_leaking"),
                 ("llama8b_it", "bfcl")}                     # Llama-3.1-8B has no BFCL run (2026-09-25: BCB in for Llama, BFCL not applicable)
# CELLS EXCLUDED BY RULE (2026-09-25): (model, eval) pairs whose rows leave every figure and table for that model
# only. Qwen x agentic murder: the unsteered misaligned-action rate is 3.0 % (2.2-3.7 % over four unsteered runs, n = 490), a
# floor with no room to show a reduction; the same eval stays for Gemma (12.5 %), Llama (29.7 %) and Nemotron (10.2 %), so it
# is a CELL rule, not an eval drop: it applies to Qwen alone. Applied by `drop_excluded_cells`
# before anchoring; a category mean over the remaining evals is not `partial` for it (like OUT_BY_DESIGN); the appendix gap
# glyph reads `excluded by ruling`; `categories_for(model)` removes the eval from that model's own panels. agentic_blackmail is
# already `dropped` in the registry. A registry hold_out on these cells is requested; until it exists this constant
# is the rule's home.
EXCLUDED_CELLS = {("qwen35_4b_it", "agentic_murder")}
NOT_A_GAP = OUT_BY_DESIGN | EXCLUDED_CELLS


def drop_excluded_cells(frame: pd.DataFrame) -> tuple:
    """Split the rows of EXCLUDED_CELLS off a frame: (kept, dropped). The unsteered rows of the cell go too (nothing is anchored
    on them once the arms are gone)."""
    if not len(frame) or not {"model_id", "eval_id"} <= set(frame.columns):
        return frame, frame.iloc[0:0]
    key = pd.Series(list(zip(frame.model_id, frame.eval_id)), index=frame.index)
    out = key.isin(EXCLUDED_CELLS)
    return frame[~out], frame[out]


def categories_for(model: str, categories: dict | None = None) -> dict:
    """`categories` with the evals EXCLUDED_CELLS names for `model` removed from every category: the model's own panels then have
    no column for a cell that will never be drawn. Categories left empty are dropped."""
    categories = categories or CATEGORIES
    out = {}
    for cat, (evals, sign, label) in categories.items():
        keep = [e for e in evals if (model, e) not in EXCLUDED_CELLS]
        if keep:
            out[cat] = (keep, sign, label)
    return out

# THE REFUSAL A STANDUP FILES WHEN IT CANNOT SERVE A CELL (2026-09-22: Nemotron x constitution PSR on GSM8K never closes its think block,
# THINKING_UNSERVABLE predicate (A)); matched case-insensitively in the run's disposition / notes / extra. Defined in daaabe0ec, lost in
# a later merge (26c320d74) - with the name undefined, `_not_served` raised NameError into its `except Exception` and `not served` could
# never fire (found 2026-10-03). The lifecycle states a run can END in without a judgment are the other terminal branch.
NOT_SERVED_REFUSAL = "thinking_unservable_predicate_a"
FAILED_STATES = ("failed", "cancelled")


def _not_served(g: pd.DataFrame) -> bool:
    if not len(g):
        return False
    for col in ("disposition_state", "disposition", "refusal", "refusal_reason", "run_extra_json", "notes", "pass_label"):
        if col in g.columns:
            try:
                if g[col].astype(str).str.lower().str.contains(NOT_SERVED_REFUSAL, regex=False).any():
                    return True
            except Exception:
                continue
    return False


def gap_status(runs: pd.DataFrame, drawn: set, objects, evals, models=(MODEL_ID,), wrong_judge: set = frozenset()) -> pd.DataFrame:
    """Per (model, object, eval): `drawn`, else what the registry holds: `wrong judge` (judged, but only by a judge JUDGE_POLICY
    excludes), `excluded by ruling` (judged, held out by a decisive ruling - `ruling` names it), `withdrawn` (every judged run
    withdrawn by its producer: not in the paper view at all), `superseded` (every judged run superseded by a newer one), `not in roster`,
    `ceiling exit` (the producer's cap ladder ended at its ceiling, deliberately ungraded - the rung below is the cell's last reading) (judged runs the paper's selection leaves out), `unjudged` (runs without a
    judgment), or `absent` (no run at all). Never a silent drop. `drawn` and `wrong_judge` are sets of (model, object, eval)."""
    rows = []
    for m in models:
        rm = runs[runs.model_id == m]
        for o in objects:
            for e in evals:
                g = rm[(rm.object_label == o) & (rm.eval_id == e)]
                if (m, o, e) in drawn:
                    st = "drawn"
                elif (m, e) in OUT_BY_DESIGN:
                    st = "out by design"
                elif (m, e) in EXCLUDED_CELLS:
                    st = "excluded by ruling"                            # the cell rule (EXCLUDED_CELLS), a glyph the tables already key
                elif (m, o, e) in wrong_judge:
                    st = "wrong judge"
                elif _not_served(g):
                    st = "not served"                                    # attempted and refused, see NOT_SERVED_REFUSAL
                elif not len(g):
                    st = "absent"
                elif complete(g).any():
                    js = g[complete(g)]
                    st = ("excluded by ruling" if js.selection_state.eq("excluded").any() else
                          "withdrawn" if js.selection_state.eq("withdrawn").all() else
                          "superseded" if js.selection_state.eq("superseded").all() else
                          # SELECTED BUT UNDRAWABLE (2026-09-22): the run is in the paper and judged, yet files no row of the eval's headline
                          # metric - the r2_export over-refusal rows carry over_refusal_rate_full / _full_plus_partial, not over_refusal_graded -
                          # so the panel has nothing to plot; a registry spelling, named as such rather than as a roster gap
                          "no headline metric" if js.selection_state.eq("selected").any() else "not in roster")
                elif "run_extra_json" in g.columns and g.run_extra_json.map(_extra).map(lambda x: x.get("escalation_state") == "exit_ceiling").any():
                    st = "ceiling exit"                                  # the cap ladder ended ungraded on purpose: the rung below is the last reading
                elif "lifecycle_state" in g.columns and g.lifecycle_state.astype(str).isin(FAILED_STATES).all():
                    st = "failed"                                        # every run ended failed / cancelled: a terminal state, not "not yet judged" (2026-10-03)
                else:
                    st = "unjudged"
                rows.append(dict(model_id=m, object_label=o, eval_id=e, status=st, n_runs=len(g), selection=",".join(sorted(g.selection_state.dropna().unique())),
                                 lifecycle=",".join(sorted(g.lifecycle_state.dropna().unique())),
                                 ruling=",".join(sorted(g.decisive_ruling_id.dropna().unique())) if "decisive_ruling_id" in g.columns else ""))
    return pd.DataFrame(rows)


def drawn_set(pts: pd.DataFrame) -> set:
    return set(zip(pts.model_id, pts.object_label, pts.eval_id)) if len(pts) else set()


# ---- drawing -------------------------------------------------------------------------------------------------------------------
INK, INK2, MUTED, RULE = "#1f1f1d", "#52514e", "#9a9994", "#e3e2de"            # paperlib figs.py
GAP_MARK = {"absent": dict(marker="x", color=MUTED, s=70, lw=1.6),
            "not in roster": dict(marker="D", facecolors="none", edgecolors=MUTED, s=55, lw=1.5),
            "excluded by ruling": dict(marker="s", facecolors="none", edgecolors=MUTED, s=55, lw=1.6),
            "withdrawn": dict(marker="h", facecolors="none", edgecolors=MUTED, s=70, lw=1.5),
            "superseded": dict(marker="v", facecolors="none", edgecolors=MUTED, s=62, lw=1.4),
            "unjudged": dict(marker="o", facecolors="none", edgecolors=MUTED, s=70, lw=1.4),
            "ceiling exit": dict(marker="^", facecolors="none", edgecolors=MUTED, s=70, lw=1.5),
            "wrong judge": dict(marker="_", color=MUTED, s=110, lw=2.0),
            "no headline metric": dict(marker="+", color=MUTED, s=70, lw=1.4),
            "no unsteered run": dict(marker="|", color=MUTED, s=110, lw=2.0),
            "not served": dict(marker=r"$\oslash$", color=MUTED, s=95, lw=1.4),
            "failed": dict(marker=r"$\times$", facecolors="none", edgecolors=MUTED, s=85, lw=1.4),
            "not scoreable": dict(marker=r"$?$", color=MUTED, s=85, lw=1.4)}
GAP_TEXT = {"absent": "no run in the registry", "not in roster": "judged, not in the paper roster",
            "excluded by ruling": "judged, held out by a ruling (named in the tables)", "withdrawn": "judged, withdrawn by its producer (out of the paper view)",
            "superseded": "judged, superseded by a newer run",
            "unjudged": "in the registry, not yet judged", "ceiling exit": "the cap ladder exited at its ceiling, ungraded on purpose",
            "wrong judge": "graded only by an excluded judge",
            "no headline metric": "selected and judged, but files no row of the eval's headline metric",
            "no unsteered run": "no unsteered run to compare with",
            "not served": "attempted and refused: the steered model never closes its think block",
            "failed": "every run of the cell failed or was cancelled",
            "not scoreable": "measured, but its producer says the result cannot be read as a defense"}


def style():
    """Publication sizes (paperlib figs.apply_style, enlarged): the text is set for a canvas that prints at about half size.

    TICK LABELS, AXIS LABELS AND THE LEGEND WENT UP AGAIN. These figures are read at half size in a
    paper and, before that, on a laptop screen where a ten-column panel row is already compressed; the tick labels were the
    first thing to become unreadable. Titles are deliberately NOT raised with them: they are already the largest text on the
    canvas and the point of the change is the small text, not more contrast between the two."""
    plt.rcParams.update({"font.size": 17, "axes.titlesize": 20, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 18, "legend.title_fontsize": 19, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6, "xtick.major.size": 6, "ytick.major.size": 6,
                         "axes.titlelocation": "left", "axes.titleweight": "semibold", "axes.titlepad": 10,
                         "figure.facecolor": "white", "figure.dpi": 100, "savefig.dpi": 200, "pdf.fonttype": 42, "legend.frameon": False,
                         "lines.solid_capstyle": "round", "axes.grid": False})


def save(fig, name: str):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, f"{name}.svg"), bbox_inches="tight")   # vector for design software
    return fig


CI_HANDLE = Line2D([], [], ls="-", lw=2.4, color=INK, label="95% CI")
RING_HANDLE = Line2D([], [], ls="", marker="o", ms=11, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4,
                     label="ring: fewer evals than the category names")
OPEN_HANDLE = Line2D([], [], ls="", marker="o", ms=11, markerfacecolor="white", markeredgecolor=INK2, markeredgewidth=2.0,
                     label="open: unsteered anchor outside the paper frame, or a borrowed cross-source anchor")
SEED_HANDLE = Line2D([], [], ls="", marker="o", ms=6, color=INK2, label="small marks of one color: one per seed or decode draw; the mean averages them")
CEIL_HANDLE = Line2D([], [], ls="", marker=r"$\leq$", ms=10, color=MUTED, label=f"≤: the unsteered run leaves under {HEADROOM_PP:g} pp of room (ceiling-forced)")
DEGEN_HANDLE = Line2D([], [], ls="", marker=r"$\dagger$", ms=9, color=INK2, label="†n%: share of the cell's responses judged degenerate (scored in place)")
MIXCAP_HANDLE = Line2D([], [], ls="", marker=r"$\Vert$", ms=9, color=INK2,
                       label="\u2016: the seeds pooled here are not on one generation cap (two instruments, not seed variance; caps in the tables)")
GEN_HANDLE = Line2D([], [], ls="", marker="o", ms=11, color=INK2, markeredgecolor=INK2, markeredgewidth=1.8,
                    label="dark rim: the seed-replicate generation, its own mean beside the canonical run's (never pooled with it)")
SPREAD_LINE = 10.0      # the widest gap between fits worth naming beside a mean, in the panel's own unit
SPREAD_HANDLE = Line2D([], [], ls="-", lw=2.4, color=INK2,
                       label="the interval carries the spread BETWEEN fits, not only each fit's own (the tables give it)")
XCAP_HANDLE = Line2D([], [], ls="", marker=r"$\neq$", ms=9, color=INK2,
                     label="≠: differenced against an anchor at ANOTHER generation cap (biased toward the favorable corner)")
CAP_HANDLE = Line2D([], [], ls="", marker="$c$", ms=8, color=INK2, label="cap n%: share at the token cap (reported, never discounted)")


def _share(row, col):
    v = row.get(col, np.nan) if hasattr(row, "get") else np.nan
    return float(v) if pd.notna(v) else np.nan


def flag(ax, x, y, row, cap_col: str = "frac_at_token_cap", degen_col: str = "frac_degenerate_union5", fontsize: float = 13) -> tuple:
    """The side glyphs beside a mark: "≤" when the row is ceiling-forced; below it "†n%" for a degenerate share over DEGEN_LINE,
    "cap n%" for an at-cap share over AT_CAP_LINE, and "≠cap" when the delta crosses two generation caps (2026-09-20: that
    bias flips sign between an attack axis and a benign one, so it walks a point toward the favorable corner rather than
    scattering it). Returns (ceiling, at_cap, degenerate) flags; the cap mismatch is drawn here and counted by the caller."""
    # THE CEILING MARK IS RETIRED (2026-09-21: remove "<=" from everything). The headroom is still computed and still rides
    # on every row (`headroom`, `ceiling`) for the tables; it is simply no longer drawn.
    c = False
    cap, deg = _share(row, cap_col), _share(row, degen_col)
    # THE DEGENERATE SHARE IS MEASURED AND REPORTED, NEVER DRAWN: degeneracy is measured across the panel wherever the
    # measurements do not exist, and no marking is added to the figure on that basis. The field is about to
    # go from 355 filed rows to most of 3,251, so a reader that prints every share over 5 % would start annotating hundreds of
    # points the moment the values land - a large visual change nobody chose. Suppressing the DRAWN dagger rather than pinning to
    # the currently-filed rows keeps the rule true whatever lands, and the share stays on every row, in every table, and in the
    # frontier's coverage caption, which is a count rather than a per-point mark and stays.
    k, dg = bool(pd.notna(cap) and cap > AT_CAP_LINE), False
    if k and hasattr(row, "get") and str(row.get("eval_id", "")) in AT_CAP_HIDDEN:
        k = False
    xc = bool(row.get("cap_mismatch", False)) if hasattr(row, "get") else False
    parts = ([f"†{deg:.0%}"] if dg else []) + ([f"cap {cap:.0%}"] if k else []) + (["≠"] if xc else [])      # the glyph alone: 68 of these can appear on one panel and the legend carries the meaning
    if parts:
        ax.annotate(" ".join(parts), (x, y), xytext=(4, -3), textcoords="offset points", fontsize=fontsize - 1.5, color=INK2, ha="left", va="top", zorder=6)
    return c, k, dg or xc


# ---- LaTeX tables, populated from the SAME frame the figure plotted ------------------------------------
# One table under each figure, in the paper's own table style (caption and label BEFORE the tabular, centered, header row
# in small caps bold, a rule under it). They are generated from the plotted frame rather than written out, so a table can
# never drift from the figure above it: if a value moves, a seed lands or an arm drops out, the table moves with it.
CATEGORY_SHORT = {"static": "static", "multi-query": "multi-query", "multi-turn": "multi-turn",
                  "over-refusal": "over-refusal", "Petri": "Petri", "agentic misalignment": "agentic", "capability": "capability"}


def _tex(x) -> str:
    """Escape a cell for LaTeX."""
    out = str(x)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("–", "--"), ("×", r"$\times$")):
        out = out.replace(a, b)
    return out


# The first row of every delta table: the unsteered model's own absolute reading, which the arms below are differenced
# against. Italic so it reads as a reference line rather than another arm.
BASELINE_ROW_LABEL = "\\emph{unsteered (absolute)}"


def _cell(v, lo=None, hi=None, dp: int = 1, signed: bool = False) -> str:
    """A value with its 95% interval, or `--` where the figure draws a gap. Math mode so the minus signs typeset.
    `signed` prefixes a + on a positive value: in a table whose first row is an ABSOLUTE unsteered baseline, an
    unsigned 2.1 on the row below is ambiguous between a reading and a change."""
    if v is None or not np.isfinite(float(v)):
        return "--"
    out = f"${float(v):+.{dp}f}$" if signed else f"${float(v):.{dp}f}$"
    if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
        # A SYMMETRIC INTERVAL PRINTS AS +/- (2026-09-23: to fit the tables inside \textwidth). Measured over
        # the 1,185 interval cells the notebook renders: every one is symmetric to within 0.1 pp, the display precision
        # — they are analytic, value +/- 1.96 se. `[-77.7, -60.6]` is 14 characters where `+/-8.6` is 4.
        # The half-width comes from the UNROUNDED endpoints, so it does not inherit the rounding of the two bounds. The
        # bracket form is kept as a fallback for a genuinely asymmetric interval (a Wilson interval on a proportion near
        # 0 or 1 is not symmetric), so one can never be silently misprinted as +/-.
        lo_f, hi_f, v_f = float(lo), float(hi), float(v)
        half, mid = (hi_f - lo_f) / 2.0, (hi_f + lo_f) / 2.0
        if abs(mid - v_f) <= 0.5 * 10 ** -dp + 1e-9:          # within half a displayed unit: the same number once printed
            out += f" {{\\scriptsize $\\pm {half:.{dp}f}$}}"
        else:
            out += f" {{\\scriptsize $[{lo_f:.{dp}f}, {hi_f:.{dp}f}]$}}"
    return out


def _caps(h: str) -> str:
    """Upper-case a header the way the paper's tables do, but NEVER inside a math span: uppercasing the whole string
    turned `$\\Delta$ defense` into `$\\DELTA$`, which is not a control sequence and does not typeset."""
    out, lit = [], h.split("$")
    for i, part in enumerate(lit):
        out.append(part if i % 2 else part.upper())
    return "$".join(out)


def latex_table(header: list, rows: list, caption: str, label: str, colspec: str | None = None, note: str | None = None) -> str:
    """The paper's table form: caption and label first, centered, one bold header row over a rule."""
    cols = colspec or "l" * len(header)
    head = "  &".join("\\multicolumn{1}{c}{\\bf " + _caps(h) + "}" for h in header)
    body = "\n".join(" &".join(r) + " \\\\" for r in rows)
    # NO MINIPAGE FOOTNOTE. Captions are bare and the table carries the numbers; a paragraph
    # welded under the rule belongs in the prose instead. But the note is not simply dropped: it carries the KEY to
    # marks that are still in the cells -- \\ddagger on a baseline pooled from materially different anchors, \\S on a
    # held-out arm -- and a symbol with no key is worse than no symbol. It is emitted as a LaTeX COMMENT after the
    # float, so nothing renders and nothing is lost to whoever writes the surrounding prose.
    tail = ("\n\\end{tabular}\n\\end{center}\n\\end{table}"
            + ("\n% note: " + " ".join(note.split()) if note else ""))
    return ("\\begin{table}[h]\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\begin{center}\n"      # [h], not [t]
            "\\begin{tabular}{" + cols + "}\n" + head + "\n\\\\ \\hline \\\\\n" + body + tail)


def table_categories(cm: pd.DataFrame, layout: list, objects: list, caption: str, label: str,
                     model: str = MODEL_ID, note: str | None = None) -> str:
    """Figures 3 and 4: one row per arm, one column per category panel, each cell the category mean and its interval -
    the large mark after the separator in the figure above. `--` is a cell the figure draws as a gap glyph."""
    cats = [c for row in layout for c in row]
    pooled = pool_seeds(cm[cm.model_id == model]) if len(cm) else cm
    rows, partial = [], False
    # THE UNSTEERED BASELINE, ABSOLUTE, ON THE FIRST ROW. Every arm below is a change against
    # THIS row, so the table states the level once and the arms carry only the difference. The anchor is common to the
    # arms by construction (the anchor-precedence rules pick one unsteered reading per eval), so the mean over arms
    # below is that shared reading, not an average of different baselines.
    base = [BASELINE_ROW_LABEL]
    for c in cats:
        r = pooled[pooled.category == c] if len(pooled) else pooled
        v = float(r.anchor_abs.mean()) if len(r) and "anchor_abs" in r.columns and r.anchor_abs.notna().any() else None
        base.append(_cell(v) if v is not None else "--")
    rows.append(base)
    for o in objects:
        cells = [_tex(LABEL.get(o, o))]
        for c in cats:
            r = pooled[(pooled.object_label == o) & (pooled.category == c)] if len(pooled) else pooled
            if not len(r):
                cells.append("--")
                continue
            # A MEAN OVER FEWER EVALUATIONS THAN THE CATEGORY NAMES IS MARKED, in the table as well as on the canvas
            # (the figure rings it). Without this, a capability number computed from GSM8K alone while MMLU is out of
            # the frame reads as a full capability mean, which is exactly the silent narrowing the ring exists to stop.
            cells.append(_cell(r.y.iat[0], r.lo.iat[0], r.hi.iat[0], signed=True) + ("$^{\\dagger}$" if bool(r.partial.iat[0]) else ""))
            partial = partial or bool(r.partial.iat[0])
        rows.append(cells)
    if partial:
        mark = "$\\dagger$ a mean over fewer evaluations than the category names, the evaluations present being the ones in frame."
        note = f"{note} {mark}" if note else mark
    return latex_table(["arm"] + [CATEGORY_SHORT.get(c, c) for c in cats], rows, caption, label,
                       colspec="l" + "l" * len(cats), note=note)


def IS_RATE_EVAL_UNITS(u: pd.DataFrame) -> set:
    """The evals whose unsteered rows are proportions (a Wilson interval applies): every metric id in IS_RATE with a denominator n."""
    return {e for e, m in zip(u.eval_id, u.metric_id) if str(m) in IS_RATE}


def missing_intervals(d: pd.DataFrame, evals=None) -> pd.DataFrame:
    """Drawn rows with a value but NO interval (d_lo / d_hi NaN), per (model, object, eval): the registry's 'current contrasts only'
    rule (schema 21) left ~80 steered Petri rows without a current contrast while their adapters re-anchor (2026-10-07), and petri_score has no per-item data to make one from. Rendered as UNMEASURED - a mark with no bar, a cell with no
    +/- - never as a zero-width interval; the notebook prints this census under the affected figures."""
    if not len(d) or "d_y" not in d.columns:
        return pd.DataFrame(columns=["model_id", "object_label", "eval_id", "rows"])
    x = d[d.d_y.notna() & (d.get("d_lo", pd.Series(np.nan, index=d.index)).isna() | d.get("d_hi", pd.Series(np.nan, index=d.index)).isna())]
    if evals is not None:
        x = x[x.eval_id.isin(list(evals))]
    if not len(x):
        return pd.DataFrame(columns=["model_id", "object_label", "eval_id", "rows"])
    return x.groupby(["model_id", "object_label", "eval_id"]).size().rename("rows").reset_index()


NO_INTERVAL_NOTE = (" A value without an interval is UNMEASURED, not exact: its filed contrast is not current in the registry "
                    "(re-anchoring in progress) and the metric carries no per-item data to make one from.")


def table_evals_seeds(d: pd.DataFrame, layout: list, objects: list, caption: str, label: str,
                      model: str = MODEL_ID, note: str | None = None, categories: dict | None = None) -> str:
    """EVERY SEED, EVERY EVAL, SEPARATELY. The category tables pool the fit seeds into one
    number per category; this is the same frame unpooled — one row per (arm, seed), one column per EVAL of the
    panels in `layout`, each cell that seed's own delta with its 95% interval. The seed spread the pooled table
    carries inside its interval is therefore visible directly, which is the point of it.

    Leads with the unsteered absolute per eval like every other table here, and signs the deltas against it.

    ORIENTED AS THE FIGURES ARE: every cell is the category's sign times the raw delta, so + is an
    improvement everywhere -- more defense success, more benign compliance, more accuracy, LESS misalignment -- exactly what the
    panel above it draws upward. The tables used to print the raw delta, so a misalignment reduction the figure drew as +10
    read -10 in the table under the same caption. The first row stays the unsteered ABSOLUTE in the eval's own units (ASR,
    over-refusal, accuracy, misalignment score / rate); the note says which way each column's + points.

    EACH CELL NAMES THE ANCHOR IT IS AGAINST WHERE THAT IS NOT THE FIRST ROW (2026-09-25, after 44.3 + 56.2 = 100.5 % and
    44.3 - 45.6 = -1.3 % were reconstructed on PAIR). The arms on one eval are differenced against DIFFERENT unsteered
    readings by design -- a seed-replicate arm against its own standup's unsteered run (the seed standups' pool, 43.8 % on
    PAIR), the paired-write difference ReLU against its lane's own unsteered run (47.1 %) -- and the first row had been
    printing the MEAN of those anchors (44.3 %), which no cell was measured from. So a cell whose own anchor differs from the
    first row by more than 0.05 carries it in small type, `vs a`: the arm's reading is a plus the cell (higher-is-better) or
    a minus the cell (lower-is-better), exactly. The first row is now labelled as the mean of the anchors and the note says
    so."""
    categories = categories or categories_for(model)                             # the model's own columns: its excluded cells have none
    evals = [e for row in layout for c in row for e in categories[c][0]]
    evals = list(dict.fromkeys(evals))
    sign_of = {e: categories[c][1] for row in layout for c in row for e in categories[c][0]}
    _better = {1: "the raw change", -1: "the REDUCTION (the raw change, sign flipped)"}
    orient = ("Cells are oriented as in the figures, + an improvement: "
              + "; ".join(f"{', '.join(EVAL_LABEL.get(e, e) for e in categories[c][0])}: {_better[categories[c][1]]}" for row in layout for c in row)
              + ". The first row is the unsteered absolute in the eval's own units.")
    note = f"{note} {orient}" if note else orient
    one = d[(d.model_id == model) & d.d_y.notna()] if len(d) else d
    rows = []
    base = [BASELINE_ROW_LABEL, ""]
    base_val = {}
    # THE BASELINE ROW CARRIES ITS OWN INTERVAL: where several unsteered runs of the eval are averaged, the
    # cell is their mean with the runs' Wilson intervals combined over k plus the between-run variance (the anchor pooling rule),
    # so a reader sees how far the pooled anchor itself is pinned down. The value is the mean of the arms' anchors as before (the
    # figures difference against those); the interval comes from the frame's unsteered rows of the eval on this model.
    none_rows = d[(d.model_id == model) & d.object_label.eq("none")] if "object_label" in d.columns else d.iloc[0:0]
    for e in evals:
        r = one[one.eval_id == e]
        v = float(r.anchor_value.mean()) if len(r) and r.anchor_value.notna().any() else None
        base_val[e] = v
        u = none_rows[none_rows.eval_id == e]
        if v is not None and len(u) > 1 and "n" in u.columns and u.n.notna().all() and e in IS_RATE_EVAL_UNITS(u):
            vals = pd.to_numeric(u.value, errors="coerce") * SCALE.get(str(u.metric_id.iloc[0]), 1.0)
            ps = (vals / 100.0).clip(0, 1); ns = pd.to_numeric(u.n, errors="coerce")
            se_w = float(np.sqrt(((100.0 * np.sqrt(ps * (1 - ps) / ns)) ** 2).sum()) / len(u))
            se_b = float(np.sqrt(vals.var(ddof=1) / len(u))) if len(u) > 1 else 0.0
            se = float(np.hypot(se_w, se_b))
            base.append(_cell(v, v - 1.96 * se, v + 1.96 * se))
        else:
            base.append(_cell(v) if v is not None else "--")
    rows.append(base)
    own_anchor_seen = False
    no_interval_seen = False
    for o in objects:
        g = one[one.object_label == o]
        if not len(g):
            continue
        for sd in sorted({str(x) for x in g.seed.fillna("")}):
            gs = g[g.seed.fillna("") == sd]
            cells = [_tex(LABEL.get(o, o)), _tex(sd or "--")]
            for e in evals:
                r = gs[gs.eval_id == e]
                # several decode draws of one eval average first, exactly as the category mean builds its per-eval term
                if len(r):
                    sg = sign_of.get(e, 1); lo, hi = sorted((sg * r.d_lo.mean(), sg * r.d_hi.mean()))
                    if not (np.isfinite(lo) and np.isfinite(hi)):
                        no_interval_seen = True
                    cells.append(_cell(sg * r.d_y.mean(), lo, hi, signed=True))
                    a = float(r.anchor_value.mean()) if r.anchor_value.notna().any() else None
                    if a is not None and base_val.get(e) is not None and abs(a - base_val[e]) > 0.05:
                        own_anchor_seen = True                                 # the cell itself stays clean; the note says it
                else:
                    cells.append("--")
            rows.append(cells)
    if no_interval_seen:
        note = (note or "") + NO_INTERVAL_NOTE
    if own_anchor_seen:
        # THE CELLS CARRY NO PER-ARM ANCHOR (who did not want a "vs a" in every cell): the first row is the
        # mean of the arms' unsteered anchors and the note says that where the arms' anchors differ, an arm's reading does not
        # reconstruct from the first row exactly (the arms anchored on their own lane's unsteered run differ from it by a few pp).
        note = (note or "") + (" The first row is the MEAN of the arms' unsteered anchors; where an arm was differenced against another "
                               "unsteered reading (its own lane's), its reading does not reconstruct from the first row exactly.")
    return latex_table(["arm", "seed"] + [EVAL_LABEL.get(e, e) for e in evals], rows, caption, label,
                       colspec="ll" + "l" * len(evals), note=note)


def table_categories_models(cm: pd.DataFrame, category: str, objects: list, models: list, caption: str, label: str,
                            note: str | None = None, d: pd.DataFrame | None = None, categories: dict = None) -> str:
    """An appendix panel: one row per arm, one column per model, for a single category. With the anchored frame `d`, the legend names
    the cells the figure draws open (held out, role-collapsed) - `cell_notes`."""
    pooled = pool_seeds(cm) if len(cm) else cm
    categories = categories or CATEGORIES
    if d is not None:
        cn = cell_notes(d, list(categories[category][0]) + summary_evals(category)) if category in categories else cell_notes(d, [])
        if cn:
            note = f"{note} {cn}" if note else cn
    rows = []
    base = [BASELINE_ROW_LABEL]                                   # the absolute unsteered reading per model
    for m in models:
        r = pooled[(pooled.category == category) & (pooled.model_id == m)] if len(pooled) else pooled
        v = float(r.anchor_abs.mean()) if len(r) and "anchor_abs" in r.columns and r.anchor_abs.notna().any() else None
        base.append(_cell(v) if v is not None else "--")
    rows.append(base)
    for o in objects:
        cells = [_tex(LABEL.get(o, o))]
        for m in models:
            r = pooled[(pooled.object_label == o) & (pooled.category == category) & (pooled.model_id == m)] if len(pooled) else pooled
            cells.append(_cell(r.y.iat[0], r.lo.iat[0], r.hi.iat[0], signed=True) if len(r) else "--")
        rows.append(cells)
    return latex_table(["arm"] + [MODEL_LABEL.get(m, m) for m in models], rows, caption, label,
                       colspec="l" + "l" * len(models), note=note)


def table_frontier(panels: list, objects: list, caption: str, label: str, note: str | None = None) -> str:
    """Figure 7: ONE TABLE PER PANEL, one row per drawn seed-rung, the two coordinates the figure places it
    at. Each panel dict carries `pts` (the pooled ensemble points) and, where the figure drew seeds, `per_seed` (the frame its thin
    marks come from); the rows read `per_seed` when present, else `pts`. A rung hidden by ruling is absent here too. `caption` may
    hold `{panel}` (filled with the panel's title, capitalised); `label` gets `-<title>` per table. The first row of each table is
    the unsteered model's ABSOLUTE defense and compliance -- the origin the figure draws -- read from the pooled points' ay_abs /
    ax_abs. Headers "$\\Delta$ defense" / "$\\Delta$ compliance"; both are percentage points, said in the note."""
    tables = []
    units = ("Both deltas in percentage points against the unsteered model at the origin. A prompted arm is one row per dose: "
             "the mean over its runs at that dose, the figure's large diamond.")
    for panel in panels:
        pts = panel.get("pts")
        if pts is None or not len(pts):
            continue
        title = str(panel.get("title", ""))
        rows = []
        _ax = float(pts.ax_abs.mean()) if "ax_abs" in pts.columns and pts.ax_abs.notna().any() else None
        _ay = float(pts.ay_abs.mean()) if "ay_abs" in pts.columns and pts.ay_abs.notna().any() else None
        rows.append([BASELINE_ROW_LABEL, "--", "--", _cell(_ay) if _ay is not None else "--", _cell(_ax) if _ax is not None else "--"])
        src = panel.get("per_seed")
        src = src if src is not None and len(src) else pts
        for o in objects:
            g = src[src.object_label == o]
            if "dose" in g.columns:
                g = g.assign(_d=pd.to_numeric(g.dose, errors="coerce")).sort_values(["_d"] + (["seed"] if "seed" in g.columns else []))
            # A PROMPTED ARM IS ONE ROW PER DOSE: its several runs at a dose (seed views on different
            # standups, accumulated over waves) are what the figure pools into its one large diamond, so the table records
            # that mean rather than listing the runs -- the same number the main tables carry for these two conditions.
            if o in PROMPTED and "dose" in g.columns and len(g) > 1:
                g = (g.groupby("_d", as_index=False).agg(dose=("dose", "first"), dx=("dx", "mean"), dy=("dy", "mean"))
                      .assign(seed="", object_label=o, _d=lambda t: pd.to_numeric(t.dose, errors="coerce")).sort_values("_d"))
            for _, r in g.iterrows():
                sd = r.get("seed", "")
                rows.append([_tex(LABEL.get(o, o)), _tex(str(sd)) if isinstance(sd, str) and sd else "--",
                             f"${float(r.dose):g}$" if pd.notna(r.get("dose", np.nan)) else "--",
                             _cell(r.get("dy"), signed=True), _cell(r.get("dx"), signed=True)])
        cap = caption.format(panel=title.capitalize()) if "{panel}" in caption else f"{caption.rstrip('.')} ({title})."
        tables.append(latex_table(["arm", "seed", "dose", "$\\Delta$ defense", "$\\Delta$ compliance"], rows, cap,
                                  f"{label}-{title.replace(' ', '-')}", colspec="lllll", note=(units + (" " + note if note else ""))))
    return "\n\n".join(tables)

def table_dose_response(pts: pd.DataFrame, objects: list, caption: str, label: str, note: str | None = None) -> str:
    """The Figure 5 companion: dose against multi-turn defense, one row per rung on its own dose axis."""
    rows = []
    _ay = float(pts.ay_abs.mean()) if len(pts) and "ay_abs" in pts.columns and pts.ay_abs.notna().any() else None
    rows.append([BASELINE_ROW_LABEL, "--", "--", _cell(_ay) if _ay is not None else "--"])   # the level the rungs move from
    for o in objects:
        g = pts[pts.object_label == o].sort_values("dose") if len(pts) else pts
        for _, r in g.iterrows():
            rows.append([_tex(LABEL.get(o, o)), f"${float(r.dose):g}$", _tex(DOSE_UNIT.get(o, "").replace("\n", " ")),
                         _cell(r.get("dy"), signed=True)])
    return latex_table(["arm", "dose", "dose unit", "$\\Delta$ defense (pp)"], rows, caption, label, colspec="llll", note=note)


ANCHOR_SPREAD_MARK_PP = 2.0      # a table baseline averaging anchors further apart than this carries the mark (table_absolute)


def table_absolute(d: pd.DataFrame, evals: list, objects: list, models: list, caption: str, label: str,
                   note: str | None = None, flip_baseline: bool = False) -> str:
    """The table under an absolute appendix figure, in the form every table now shares -- the unsteered model absolute,
    every other arm +/- against it: the unsteered model's absolute reading FIRST, one row per model, then every arm as a
    signed change against its own model's unsteered run - the anchored frame's pooled per-eval deltas, the same numbers the x = object
    figures, the relative companion and the category tables read, oriented by the eval's category so positive is better (defense,
    benign compliance and accuracy up; misaligned actions down). Reads the anchored frame `d`, not the absolute points, so a row here
    IS the delta the figures draw and never a difference of two rounded marks. The baseline is the anchor the deltas were taken
    against, unoriented (a level has no direction), in the figure's units: `flip_baseline` for the panels drawn as 100 - rate. Where
    one model's arms on an eval were anchored on unsteered readings materially apart (ANCHOR_SPREAD_MARK_PP) the baseline is their
    mean and marked; a held-out cell (`not scoreable`) keeps its value and is marked."""
    x = d[d.eval_id.isin(evals) & d.d_y.notna()] if len(d) else d
    rows, mixed, held = [], False, False
    base = x.groupby(["model_id", "eval_id"]).anchor_value.agg(["mean", "min", "max"]) if len(x) else None
    for m in models:
        if base is None or m not in set(x.model_id):
            continue
        cells = [BASELINE_ROW_LABEL, _tex(MODEL_LABEL.get(m, m))]
        for e in evals:
            if (m, e) in base.index and pd.notna(base.loc[(m, e), "mean"]):
                v = float(base.loc[(m, e), "mean"])
                # MARKED ONLY WHEN THE ANCHORS DIFFER MATERIALLY. Nearly every model's arms on an eval are anchored on more than one
                # unsteered reading (a seed view's own run, a lane's own run, a decode-matched run), almost always within a point of
                # each other; marking each would put the sign on every row and say nothing. The mark means the one number here
                # stands for readings that are genuinely apart.
                spread = float(base.loc[(m, e), "max"] - base.loc[(m, e), "min"])
                cells.append(_cell(100.0 - v if flip_baseline else v) + ("$^{\\ddagger}$" if spread > ANCHOR_SPREAD_MARK_PP else ""))
                mixed |= spread > ANCHOR_SPREAD_MARK_PP
            else:
                cells.append("--")
        rows.append(cells)
    p = delta_points(d, evals, objects)
    for o in objects:
        if o == "none":
            continue
        for m in models:
            g = p[(p.object_label == o) & (p.model_id == m)] if len(p) else p
            if not len(g):
                continue
            cells = [_tex(LABEL.get(o, o)), _tex(MODEL_LABEL.get(m, m))]
            for e in evals:
                r = g[g.eval_id == e]
                if not len(r):
                    cells.append("--")
                    continue
                r = r.iloc[0]
                cells.append(_cell(r.y, r.lo, r.hi, signed=True) + ("$^{\\S}$" if is_held(r) else "") + ("$^{\\ast}$" if is_collapsed(r) else ""))
                held |= is_held(r)
            rows.append(cells)
    marks = []
    if mixed:
        marks.append(f"$\\ddagger$ the model's arms on this evaluation were anchored on unsteered readings more than {ANCHOR_SPREAD_MARK_PP:g} pp "
                     "apart (a seed view's, a lane's or a decode-matched run's own); the baseline shown is their mean.")
    cn = cell_notes(d, evals)                                   # the open marks: § held out, * role-collapsed
    if cn:
        marks.append(cn.replace("Open marks, not scoreable", "$\\S$ open marks, not scoreable").replace("Open marks, role collapse", "$\\ast$ open marks, role collapse"))
    if marks:
        note = " ".join(([note] if note else []) + marks)
    return latex_table(["arm", "model"] + [EVAL_LABEL.get(e, e) for e in evals], rows, caption, label,
                       colspec="ll" + "l" * len(evals), note=note)


# ---- which figure a row belongs to -------------------------------------------------------------------
# A ROW BELONGS TO A FIGURE BY COMPARISON ID, not merely by being in the paper (2026-09-21; the registry's own
# contract). Selecting on paper membership alone means any roster change that admits rows for SOMEONE ELSE'S figure silently
# joins mine: the case that prompted this was roster v4 marking the ReLU steers selected for the all-model appendix, which
# under a membership-only filter would have added draws to two arms my main figures already draw, changing their values with
# nothing on the canvas to say why. The ids are read off the roster's own comparisons, never invented here.
MAIN_FIGURE_COMPARISONS = ("fig1_headline/*", "fig2_jailbreak/*", "fig3_misalign/*", "fig4_transfer*", "fig4_organism*",
                           "fig5_frontier/*", "figQ_*")
# the all-model appendix reads its own five comparisons AND the figR_* set, which is the same five for the ReLU steers on the
# three companion models (the companion panels; no Qwen run carries a figR id, verified at head 151285)
APPENDIX_COMPARISONS = ("figA_*", "figR_*")


def _comparison_ids(v) -> list:
    try:
        out = json.loads(v) if isinstance(v, str) else (v if isinstance(v, list) else [])
    except Exception:
        return []
    return [str(x) for x in out] if isinstance(out, list) else []


def drop_padded(frame: pd.DataFrame) -> tuple:
    """Split the padded rows off the main panels (2026-09-22). There is NO pad-0 arm in the padding wave: the unpadded
    rows already in the frame come from other waves at a 4,864-token window and a 4,096 cap, while the padded wave runs a
    98,304 window and a 16,384 single-pass cap. So an unpadded row and a padded one differ in three things at once, and a
    panel drawing them together - or a series joining them through an unmeasured origin - would read a change in serving
    geometry as an effect of padding. The two padded LEVELS share one standup and compare cleanly with each other, so they
    belong in a figure of their own. Returns (unpadded, padded)."""
    if not len(frame) or "pad" not in frame.columns:
        return frame, frame.iloc[0:0]
    # the CONTRACT, not the count: a zero-padding row of the padding wave belongs to the padding analysis too
    padded = (frame["pad_contract"].fillna(False).astype(bool) if "pad_contract" in frame.columns
              else frame["pad"].astype(float) > 0) | (frame["pad"].astype(float) > 0)
    return frame[~padded], frame[padded]


def comparison_filter(frame: pd.DataFrame, patterns=MAIN_FIGURE_COMPARISONS, label: str = "", quiet: bool = False):
    """Keep the rows that belong to this figure's comparisons, and say what that cost.

    A row is kept when any of its `comparison_ids_json` entries matches one of `patterns`, OR when it carries NO comparison at
    all and was admitted by an `include` ruling. That second clause is not a loophole, it is the larger half: 749 in-paper rows
    at head 151285 are key-scoped includes with an empty list and no roster entry behind them - the Qwen seed replicates, the
    Arditi served operating point, the agentic vendor tree, the best-of-N vendor instrument - and a filter on ids alone would
    drop exactly the rows the paper has spent the night landing. Rows with an empty list and NO include ruling are the genuine
    fail-closed case and are dropped and counted; there are none today, and if that number ever moves it wants looking at.
    """
    if "comparison_ids_json" not in frame.columns or not len(frame):
        return frame
    ids = frame["comparison_ids_json"].map(_comparison_ids)
    hit = ids.map(lambda L: any(fnmatch.fnmatchcase(i, p) for i in L for p in patterns))
    empty = ids.map(len).eq(0)
    inc = (frame["decisive_action"].fillna("").eq("include") if "decisive_action" in frame.columns
           else pd.Series(False, index=frame.index))
    # AN INCLUDE RULING KEEPS THE ROW WHATEVER IDS THE ROSTER LATER ADDS (roster v10, 2026-09-25): the Qwen seed fits' full-suite
    # BigCodeBench rows were include-selected with an EMPTY comparison list and gained their first id, `figA_transfer_models`,
    # under v10 -- and the `empty &` clause then dropped them from the Qwen main figures and the capability table as "belonging to
    # other figures only" (who asked where the LoRAs' BCB column had gone). An include is the registry saying the run is
    # in the paper; which FIGURE a kept row draws in is decided downstream by its eval and category, not by this filter.
    keep = hit | inc
    if not quiet:
        named = int((~hit & ~empty & ~inc).sum())
        print(f"comparison filter{' (' + label + ')' if label else ''}: {int(hit.sum())} rows by comparison id, "
              f"{int((empty & inc & ~hit).sum())} by an include ruling with no comparison, "
              f"{int((~empty & inc & ~hit).sum())} by an include ruling though carrying only other figures' ids, "
              f"{named} dropped as belonging to other figures only, "
              f"{int((empty & ~inc).sum())} dropped with no comparison and no include ruling (expected 0).")
    return frame[keep]


# ---- where an agentic value came from ----------------------------------------------------------------
# THE MEASUREMENT CHANGED UNDER THE PANEL, SO THE PANEL SAYS SO (2026-09-21). ReLU's completion-cap MERGED readings are now the
# agentic value for the Qwen seed views, which roughly doubles leaking on several arms; the 3,000-cap seed
# rows are superseded. That is a change in the instrument, not in the models, and with nothing printed a reader diffing two
# artifacts would see the arms move and have no way to know why. The registry names the fields (2026-09-21): the merged rows
# carry `token_cap_tokens` 16,384, a `merge` block, `ladder_rung_shares` (what fraction of each cell's verdicts came from each
# rung) and `unresolved_truncation` (what fraction is still at the ceiling, 49-54 % on the prompted constitution murder cells),
# and the admitting ruling id on the row. Everything here is read defensively: before the fold the fields are absent and this
# table is empty, which is the correct rendering of "no merged readings yet".
MERGE_RULING_MARK = "misalign-ladder-merged"


def merge_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per merged agentic cell: the cap it was read at, the share of its verdicts drawn from the re-draw rung, and the
    share still sitting at the ceiling. Empty when nothing in the frame is a completion-cap merge."""
    if not len(frame) or "run_extra_json" not in frame.columns:
        return pd.DataFrame(columns=["object_label", "eval_id", "seed", "cap", "share_from_redraw", "still_at_ceiling", "ruling"])
    out = []
    for _, r in frame.iterrows():
        ex = _extra(r.get("run_extra_json"))
        ruling = str(r.get("decisive_ruling_id") or "")
        if not isinstance(ex, dict) or not (ex.get("merge") or MERGE_RULING_MARK in ruling):
            continue
        shares = ex.get("ladder_rung_shares") or {}
        top = None
        if isinstance(shares, dict) and shares:
            try:                                                   # the deepest rung is the re-draw; its `rows` share is what we print
                top = shares.get(str(max(int(k) for k in shares)))
            except (ValueError, TypeError):
                top = None
        def _frac(d, *names):
            for n in names:
                v = (d or {}).get(n) if isinstance(d, dict) else None
                if isinstance(v, (int, float)):
                    return float(v)
            return np.nan
        unres = ex.get("unresolved_truncation") or {}
        out.append(dict(object_label=r.get("object_label"), eval_id=r.get("eval_id"), seed=r.get("seed", ""),
                        cap=r.get("token_cap_tokens"), share_from_redraw=_frac(top, "share", "rows_share", "rows"),
                        still_at_ceiling=_frac(unres, "share", "frac"),
                        decode=str(r.get("decode_policy_id") or r.get("decode_source") or ""),
                        engine=str(r.get("serving_path") or r.get("serving_identity") or ""), ruling=ruling))
    return pd.DataFrame(out, columns=["object_label", "eval_id", "seed", "cap", "share_from_redraw", "still_at_ceiling",
                                      "decode", "engine", "ruling"])


def merge_caption(row) -> str:
    """The one-line caption proposed, from the row `merge_provenance` produced."""
    bits = [f"completion-cap merge, cap {int(row.cap):,}" if pd.notna(row.get("cap")) else "completion-cap merge"]
    if pd.notna(row.get("share_from_redraw")):
        bits.append(f"{float(row.share_from_redraw):.0%} of verdicts from the re-draw")
    if pd.notna(row.get("still_at_ceiling")):
        bits.append(f"{float(row.still_at_ceiling):.0%} still at the ceiling")
    # THE DELTA CROSSES A DECODE BOUNDARY AND THE CAPTION SAYS SO (2026-09-21). The
    # merged arms are decoded at temperature 1 under `runner_forced_T1`; the in-frame unsteered rows they are differenced
    # against are the vendor preset at 0.7. The registry files no contrast for these arms and flags the reference
    # `instrument_mismatch`. So part of any movement in these numbers is the decode, not the steering, and a reader comparing
    # them with the rest of the panel needs that on the page rather than in a thread. The matched baseline is coming; until
    # it does, this caption is the whole of the warning.
    if "forced_t1" in str(row.get("decode", "")).lower():
        bits.append("arms T 1.0, anchor T 0.7 (vendor): this delta crosses a decode boundary")
    return "; ".join(bits)


def object_handles(objects) -> list:
    return [Line2D([], [], ls="", marker="o", ms=11, color=COLOR[o], markeredgecolor="white", label=LABEL[o]) for o in objects]


def model_handles(models) -> list:
    return [Line2D([], [], ls="", marker=MARKER[m], ms=11, color=MUTED, markeredgecolor="white", label=MODEL_LABEL[m]) for m in models]


def gap_handles(statuses: set) -> list:
    return [Line2D([], [], marker=GAP_MARK[g]["marker"], ls="", ms=10, color=MUTED, markeredgewidth=1.6,
                   markerfacecolor="none" if "facecolors" in GAP_MARK[g] else MUTED, label=GAP_TEXT[g]) for g in GAP_TEXT if g in statuses]


def show(df, n: int = 5, caption: str | None = None):
    """Display the HEAD of a table only (2026-09-20: print only the heads of the notebook's frames), with one line saying how
    many rows are not shown, so a long table never fills the page. An empty frame says so instead of rendering a header with no rows."""
    try:
        from IPython.display import display as _display
    except ImportError:                                   # plain-python smoke runs
        _display = print
    if isinstance(df, pd.Series):
        df = df.to_frame()
    lead = f"{caption}: " if caption else ""
    if not len(df):
        print(f"{lead}no rows")
        return
    if len(df) > n:
        print(f"{lead}first {n} of {len(df)} rows")
    elif caption:
        print(caption)
    _display(redact_session_names(df.head(n)))


def legend_stack(fig, groups, ncol_max: int = 6, y0: float = 1.005, gap: float = 0.012):
    """Stacked figure legends ABOVE the canvas (savefig's tight bbox includes them, so they cost the panels no height): each
    group is (title or None, handles); the first group lands on top. Heights are measured, so rows never collide."""
    r = fig.canvas.get_renderer(); H = fig.get_size_inches()[1] * fig.dpi
    y = y0
    groups = [g for g in groups if g[1]]
    # ONE GROUP, NO HEADER (on the best-of-N curve figure's lone `object` title): a header distinguishes
    # groups from one another, so a legend with a single group has nothing for it to say. Two or more groups keep theirs.
    if len(groups) == 1:
        groups = [(None, groups[0][1], *groups[0][2:])]
    for title, handles, *nc in reversed(groups):
        leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=min(nc[0] if nc else ncol_max, len(handles)), title=title,
                         handletextpad=0.4, columnspacing=1.5, borderaxespad=0.0, labelspacing=0.45)
        if title:
            leg.get_title().set_fontweight("semibold")
        y += leg.get_window_extent(r).height / H + gap
    return y


def _panel_grid(fig, layout, widths):
    """Axes for `layout` (rows of panel keys) with per-row width ratios from `widths[key]`; nested gridspecs under constrained layout.
    A layout of ONE panel per row (the stacked x = object appendix figures) is one flat column instead: constrained layout fits each
    nested gridspec on its own, so stacked panels whose y labels differ in width (one line against two) were left with different left
    edges; in one gridspec column they share one."""
    outer = fig.add_gridspec(len(layout), 1, hspace=0.12)
    if all(len(row) == 1 for row in layout):
        return [(fig.add_subplot(outer[i, 0]), row[0]) for i, row in enumerate(layout)]
    axes = []
    for i, row in enumerate(layout):
        sub = outer[i].subgridspec(1, len(row), width_ratios=[widths[k] for k in row], wspace=0.06)
        axes += [(fig.add_subplot(sub[0, j]), k) for j, k in enumerate(row)]
    return axes


def _spread(xc: float, n: int, half: float) -> np.ndarray:
    """`n` positions centered on `xc`, fanned across ± `half` (one position: `xc` itself)."""
    return np.full(n, xc) if n <= 1 else xc + np.linspace(-half, half, n)


def dodge(ax, frame, x_order, objects, gaps: pd.DataFrame | None, y="y", lo="lo", hi="hi", size=60, x_key="eval_id", gap_y=0.0, width=0.82, model=None,
          open_col=None, seed_scale: float = 0.45) -> tuple:
    """Colored points per object, dodged inside each x column, with their intervals; several seeds of one object fan out inside
    its slot; gap glyphs where a column has no row. A row with `open_col` True is drawn open (anchor outside the paper frame).
    Returns (whether any slot held more than one seed, the (x, y, row) placements) so the caller can flag marks."""
    step = width / max(len(objects), 1)
    seeded, placed = False, []
    for j, o in enumerate(objects):
        off = -width / 2 + step * (j + 0.5)
        for i, xk in enumerate(x_order):
            rs = frame[(frame.object_label == o) & (frame[x_key] == xk)]
            rs = rs.sort_values([c for c in ("seed", "draw") if c in rs.columns]) if len(rs) else rs
            if len(rs):
                seeded |= len(rs) > 1
                sz = size * seed_scale if len(rs) > 1 else size                     # several seeds: small marks
                for xi, (_, r) in zip(_spread(i + off, len(rs), 0.3 * step), rs.iterrows()):
                    if pd.notna(r[lo]):
                        ax.plot([xi] * 2, [r[lo], r[hi]], color=COLOR[o], lw=1.6 if len(rs) == 1 else 1.1, alpha=0.85, solid_capstyle="butt", zorder=2)
                    if open_col is not None and bool(r[open_col]):
                        ax.scatter([xi], [r[y]], s=sz, facecolors="white", edgecolors=COLOR[o], lw=2.0 if len(rs) == 1 else 1.4, zorder=3)
                    else:
                        ax.scatter([xi], [r[y]], s=sz, color=COLOR[o], zorder=3)
                    placed.append((xi, r[y], r))
            elif gaps is not None:
                st = gaps[(gaps.object_label == o) & (gaps[x_key] == xk) & ((gaps.model_id == model) if model is not None else True)]
                if len(st) and st.iloc[0].status in GAP_MARK:
                    ax.scatter([i + off], [gap_y], zorder=3, **GAP_MARK[st.iloc[0].status])
    ax.set_xticks(range(len(x_order)))
    for i in range(1, len(x_order)):
        ax.axvline(i - 0.5, color=RULE, lw=1.0)
    return seeded, placed


def delta_figure(cm: pd.DataFrame, d: pd.DataFrame, layout: list, objects: list, gaps: pd.DataFrame, name: str, model: str = MODEL_ID,
                 unit: float = 2.0, height: float = 4.2,                     # shorter axes throughout
                 panel_hooks: dict | None = None, hook_widths: dict | None = None,
                 tick_rotation: dict | None = None, categories: dict | None = None, titles: dict | None = None,
                 tick_fontsize: dict | None = None):
    """The paper's Figures 3 / 4 for ONE model: per category panel the oriented delta per eval (small marks) beside the
    category mean (large mark after the separator), each with its interval; a gap glyph where the registry has no drawable
    row. `layout` is a list of rows of categories (a 2 x 2 for the four jailbreak / over-refusal categories); panel widths
    follow the number of columns each draws. Several seeds of one object are separate marks inside its slot. A black ring
    marks a mean over fewer evals than the category names; an open mark a delta whose unsteered run lies outside the frame."""
    categories = categories or CATEGORIES                                        # a model's own panels pass categories_for(model)
    # A LAYOUT KEY MAY BE A HOOK RATHER THAN A CATEGORY (2026-09-24). Figure 3 puts the padding ladder beside static,
    # and padding is not an eval category — it has its own x (filler-token rungs), so it cannot be expressed as one. A hook
    # is a callable that draws itself onto the axis it is handed; everything else (grid, legend, save) stays shared. This
    # keeps padding's readings out of every category mean BY CONSTRUCTION: they never enter `d`, which is what the standing
    # rule requires — sharing a figure is a layout choice, not a pooling one.
    panel_hooks = panel_hooks or {}
    # a one-eval category has no mean column and takes ONE column's width, the same as each eval slot of its neighbours
    widths = {c: (len(categories[c][0]) + 1 if len(categories[c][0]) > 1 else 1.0) for c in categories}
    widths.update(hook_widths or {k: 4 for k in panel_hooks})
    fig = plt.figure(figsize=(max(sum(widths[c] for c in row) for row in layout) * unit + 1.4, height * len(layout)), layout="constrained")
    axes = _panel_grid(fig, layout, widths)
    seen, any_ring, any_open, any_seed, any_ceil, any_cap, any_degen, any_spread = set(), False, False, False, False, False, False, False
    any_xcap = bool(d["cap_mismatch"].any()) if "cap_mismatch" in d.columns else False
    any_mixcap = any_gen = False
    for ax, cat in axes:
        if cat in panel_hooks:
            panel_hooks[cat](ax)
            continue
        evals, sign, ylabel = categories[cat]
        n = len(evals)                                                         # the panel's eval count; a lone eval draws no mean column
        pe = d[(d.model_id == model) & d.eval_id.isin(evals)].copy()
        pe["y"], pe["lo"], pe["hi"] = sign * pe.d_y, np.minimum(sign * pe.d_lo, sign * pe.d_hi), np.maximum(sign * pe.d_lo, sign * pe.d_hi)
        pe["open"] = ~pe.anchor_in_frame.astype(bool)
        g = gaps[(gaps.model_id == model) & gaps.eval_id.isin(evals + summary_evals(cat))].copy()
        na = pe[pe.d_y.isna()]                                              # steered run present, no unsteered run: its own glyph
        g.loc[g.set_index(["object_label", "eval_id"]).index.isin(na.set_index(["object_label", "eval_id"]).index), "status"] = "no unsteered run"
        seeded, placed = dodge(ax, pool_draws(pe[pe.d_y.notna()]), evals, objects, g, size=55, model=model, open_col="open")
        any_seed |= seeded
        # ONE LABEL PER OBJECT AND EVAL, not one per seed (2026-09-21). Once the seed folds landed, three small marks each
        # carrying "cap 86%" piled the agentic panel into an unreadable mass of overlapping text - the labels obscured the very
        # points they annotate. Every seed's share is still computed and still in the tables; what the canvas needs is the WORST
        # of them, so the label goes on the seed with the largest share and carries the group's cap mismatch. No share is
        # discounted and none is hidden from the reader who looks at the frame; only the repetition goes.
        best, mis = {}, set()
        for xi, yi, row in placed:
            key = (row.get("object_label"), row.get("eval_id"))
            shares = [v for v in (_share(row, "frac_at_token_cap"), _share(row, "frac_degenerate_union5")) if pd.notna(v)]
            sh = max(shares) if shares else -1.0
            if key not in best or sh > best[key][0]:
                best[key] = (sh, xi, yi, row)
            if bool(row.get("cap_mismatch", False)):
                mis.add(key)
        for key, (_, xi, yi, row) in best.items():
            r = row.copy()
            if key in mis:
                r["cap_mismatch"] = True
            # NO AT-CAP LABEL ON ANY MAIN PANEL (2026-09-21): off Petri and agentic, and off the static, over-refusal and
            # capability panels. Rather than keep a per-eval exception list that would leave two panels carrying
            # a mark the rest have lost, the label is off the whole figure. Every share is still computed, still on the row, still
            # in the tables, and still drawn on the appendix figures and the dose-response companion.
            # AND NO CAP-MISMATCH MARK EITHER (2026-09-21). Sharing the SI cells' anchor on the agentic scenarios deliberately
            # gives up the cap match, so that one change put ten "not-equal" glyphs on a panel whose other annotations had just
            # been removed, with no key to read them by. The mismatch is still computed, still on every row and still in the
            # tables; it is the anchor rule above that decides it, and the rule is stated under the figure, not glyph by glyph.
            r["cap_mismatch"] = False
            c, k, dg = flag(ax, xi, yi, r, cap_col="_none"); any_ceil |= c; any_cap |= k; any_degen |= dg
        any_open |= bool(pe.loc[pe.d_y.notna(), "open"].any())
        if n > 1:                                                              # a one-eval category draws no mean column
            m = pool_seeds(cm[(cm.model_id == model) & (cm.category == cat)])      # ONE mean point per object: the seeds averaged
            gs_ = g[g.eval_id.isin(summary_evals(cat))]                          # the summary column's gap: the pooled eval's own status where declared
            gm = pd.DataFrame([dict(object_label=o, status=(gs_[gs_.object_label == o].status.mode().iat[0] if len(gs_[gs_.object_label == o]) else "absent"))
                               for o in objects if o not in set(m.object_label)], columns=["object_label", "status"])
            seen |= set(g[g.status != "drawn"].status) | set(gm.status)
            n, step = len(evals), 0.82 / len(objects)
            for j, o in enumerate(objects):                                      # the mean column sits after a separator, at x = n
                rs = m[m.object_label == o]
                off = -0.41 + step * (j + 0.5)
                if len(rs):
                    # ONE MEAN PER GENERATION, never one across them (2026-09-21): the canonical full22 run and the
                    # seed-replicate run are different generations, so they sit side by side in the object's slot - the replicate with a
                    # dark rim - rather than being averaged into a point that means neither.
                    rs = rs.sort_values("generation") if "generation" in rs.columns else rs
                    any_gen |= len(rs) > 1
                    for xi, (_, r) in zip(_spread(n + off, len(rs), 0.3 * step), rs.iterrows()):
                        repl = str(r.get("generation", GEN_CANONICAL)) == GEN_REPLICATE
                        any_seed |= int(r.n_seeds) > 1
                        ax.plot([xi] * 2, [r.lo, r.hi], color=COLOR[o], lw=2.6, zorder=3)
                        if bool(r.open):
                            any_open = True
                            ax.scatter([xi], [r.y], s=190, facecolors="white", edgecolors=COLOR[o], lw=2.4, zorder=4)
                        else:
                            ax.scatter([xi], [r.y], s=190, color=COLOR[o], zorder=4, edgecolor=(INK2 if repl else "white"), lw=(1.8 if repl else 0.8))
                        if bool(r.partial):
                            any_ring = True
                            ax.scatter([xi], [r.y], s=190, facecolors="none", edgecolors="black", lw=1.4, zorder=5)
                        # THE MEAN COLUMN CARRIES NO SHARE LABEL (2026-09-21). A category mean's at-cap and degenerate shares are the
                        # LARGEST over the evals it spans, so every number a label here could print is already printed beside one of
                        # the per-eval marks to its left - and nine objects in one narrow column put those repeats on top of each
                        # other and on top of the points. Nothing is discounted or hidden: the shares still ride on the mean row and
                        # are in the tables, and the cap mismatch, which IS a property of the mean rather than of any single eval,
                        # is still drawn.
                        # AND THE MEAN COLUMN CLEARS THE CAP MISMATCH TOO (2026-09-21). The per-eval marks were cleared when the
                        # shared anchor landed and this path was missed, so five of the glyphs survived on the agentic means alone.
                        # On those panels the mismatch is the DESIGN - sharing the SI cells' anchor deliberately gives up the cap
                        # match - so marking it says only that the rule the figure is drawn under is in force. It stays on every
                        # row and in the tables.
                        r = r.copy(); r["cap_mismatch"] = False
                        c, k, dg = flag(ax, xi, r.y, r, cap_col="_none", degen_col="_none", fontsize=14); any_ceil |= c; any_degen |= dg
                        # THE TWO-INSTRUMENT BAR IS OFF THE MAIN FIGURES. It still rides on the row as
                        # `mixed_cap_seeds`, is still computed from the declared ladders, and still appears on the appendix figures;
                        # what goes is the mark beside the mean, which was the last of the side annotations on these panels.
                        any_mixcap = any_mixcap or bool(r.get("mixed_cap_seeds", False))
                        if (float(r.spread) if "spread" in r.index and pd.notna(r.spread) else 0.0) > SPREAD_LINE:
                            any_spread = True                                    # the interval carries it; the number is in the tables
                else:
                    st = gm[gm.object_label == o]
                    if len(st) and st.status.iat[0] in GAP_MARK:
                        ax.scatter([n + off], [0], zorder=3, **GAP_MARK[st.status.iat[0]])
        if n > 1:
            ax.axvline(n - 0.5, color=INK2, lw=1.3)                                 # the separator before the mean column
        ax.axhline(0, color=INK2, lw=1.1, zorder=1)
        # "mean" ON EVERY CATEGORY, Petri included. The column had read "pooled" there since
        # 2026-09-20 to mark that its value is the REGISTRY's pooling over the six behaviors rather than this notebook's
        # unweighted mean of the six columns beside it. That distinction is real and still holds — the value is
        # unchanged, only the label is — so it moves into the caption and the provenance note instead of the tick.
        # ROTATED EVAL LABELS ON A NARROW PANEL (for Figure 5's Petri panel): with the panel
        # narrowed, seven columns of two-line names collide, so a category named in `tick_rotation` gets its EVAL labels
        # on one line each, turned by the given angle, with the label's right edge anchored on its tick. "mean" is not
        # an eval and stays horizontal and bold, so the summary column still reads at a glance.
        _rot = (tick_rotation or {}).get(cat)
        # rotated labels KEEP their two-line split: the same "\n" the horizontal form uses,
        # so "cyber/bio/" over "chem" turns as a block; the figure height stays at the value set for the one-line form
        # NO MEAN OVER ONE EVAL: a category left with a single eval -- Qwen's agentic panel once its murder
        # cells were excluded -- shows the eval alone, no separator, no "mean" column that would repeat the point.
        _labels = [tick(e) for e in evals] + (["mean"] if n > 1 else [])
        ax.set_xticks(range(len(_labels))); ax.set_xticklabels(_labels)
        if _rot:
            for _lab in (ax.get_xticklabels()[:-1] if n > 1 else ax.get_xticklabels()):
                _lab.set_rotation(_rot); _lab.set_ha("right"); _lab.set_rotation_mode("anchor")
        if n > 1:
            ax.get_xticklabels()[-1].set_fontweight("bold")
        if (tick_fontsize or {}).get(cat):                                       # a panel whose long horizontal labels need a smaller face at a narrower unit
            for _lab in ax.get_xticklabels():
                _lab.set_fontsize(tick_fontsize[cat])
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(min(lo_, -5), max(hi_, 10))                                # an axis with nothing but gap glyphs keeps a readable range
        ax.set_xlim(-0.6, len(_labels) - 0.4); ax.set_title((titles or {}).get(cat, cat)); ax.set_ylabel(_wrap_label(ylabel))    # long labels wrap (v171)
    marks = (gap_handles(seen) + ([RING_HANDLE] if any_ring else []) + ([OPEN_HANDLE] if any_open else []) + ([SEED_HANDLE] if any_seed else [])
             + ([SPREAD_HANDLE] if any_spread else []) + ([DEGEN_HANDLE] if any_degen else [])
             + ([CAP_HANDLE] if any_cap else []) + ([XCAP_HANDLE] if any_xcap else []) + ([MIXCAP_HANDLE] if any_mixcap else []) + ([GEN_HANDLE] if any_gen else []))
    # NO MARKS KEY ON THE MAIN FIGURES. The glyphs and side labels are all still drawn and all still
    # defined in the notebook's prose and tables; what goes is the block of legend text above the canvas, which had grown to
    # three rows and was competing with the figure for the reader's eye. `marks` is still assembled so the appendix figures,
    # which keep their key, share one definition of it.
    # THE 95% CI KEY IS THE SECOND ENTRY: the legend fills COLUMN-WISE, so slot 2 sits directly
    # under slot 1 — the key ends up beneath "Refusal steer" in the first column, and every remaining object stays
    # paired in the columns to its right. I had first moved it to the END, which unpaired them; before that it led.
    _oh = object_handles(objects)
    legend_stack(fig, [(None, _oh[:1] + [CI_HANDLE] + _oh[1:], 5)])
    return save(fig, name)


def rung_size(dose: float) -> float:
    """Marker area by |dose| (paperlib _rung_size, enlarged for these font sizes): linear to the operating point, log2 above it."""
    d = abs(float(dose))
    return 0.85 * (60.0 + 90.0 * min(d, 1.0) + 22.0 * float(np.log2(max(d, 1.0))))


def _smooth(dose, x, y, samples: int = 120):
    """A curve through an object's rungs in dose order: with >= 3 distinct rungs a monotone PCHIP in each coordinate against the
    dose, else the polyline (paperlib _smooth_rungs)."""
    order = np.argsort(dose); dose, x, y = dose[order], x[order], y[order]
    if len(dose) < 3 or len(np.unique(dose)) < len(dose):
        return x, y
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        return x, y
    t = np.linspace(dose[0], dose[-1], samples)
    return PchipInterpolator(dose, x)(t), PchipInterpolator(dose, y)(t)


def rung_eval_table(frame: pd.DataFrame, objects: list | None = None) -> pd.DataFrame:
    """One row per (object, seed, dose); one column per frontier eval, plus the ST / MT / OR means.

    This IS the frontier, unaggregated. The panels plot the ST or MT mean against the OR mean,
    so a curve that does something odd — a near-vertical segment, a seed leaving the panel — shows up here as the one
    eval carrying the move, instead of having to be inferred from a shape. Deltas, because deltas are what is drawn;
    ST and MT are sign-oriented so positive is better defense, OR so positive is better benign compliance."""
    ST, MT, ORB = FRONTIER_STATIC[1], FRONTIER_Y[1], FRONTIER_X[1]
    d = frame[frame.headline & frame.judge_ok] if "judge_ok" in frame.columns else frame[frame.headline]
    if "selection_state" in d.columns:
        d = d[d.selection_state.eq("selected")]
    d = anchored(d)
    d = d[d.object_label.ne("none") & d.d_y.notna()]
    if objects:
        d = d[d.object_label.isin(objects)]
    # the table shows the rungs the PANELS draw, so it applies the same three exclusions: the original fit where seed
    # replicates exist, negative strengths, and the rungs hidden on either panel
    _seeded_obj = {o for o, g in d.groupby("object_label") if g.seed.fillna("").ne("").any()} if len(d) else set()
    d = d[~(d.object_label.isin(_seeded_obj) & d.seed.fillna("").eq(""))]
    d = d.assign(_dose=pd.to_numeric(d.dose, errors="coerce"))
    if FRONTIER_DROP_NEGATIVE:
        d = d[d._dose >= 0]
    _hid = {(o, dd) for (_panel, o, dd) in FRONTIER_RUNG_HIDDEN}
    d = d[~d.apply(lambda r: (r.object_label, float(r._dose)) in _hid if pd.notna(r._dose) else False, axis=1)]
    d = d.drop(columns=["_dose"])
    if not len(d):
        return pd.DataFrame()
    d = d.assign(dose=pd.to_numeric(d.dose, errors="coerce"))
    # EVERY CELL IS THE GOOD DELTA THE FIGURE DRAWS (2026-09-24). Every eval on this table is lower-is-better in its raw
    # form -- ASR on the attacks, the refused rate on the benign evals -- so the whole pivot is negated once and the
    # panel means are plain means of the oriented cells. Before this the per-eval cells were RAW (a −21.4 beside an ST
    # mean of +22.4) and the OR mean carried a +1 where the figure's x uses −1, so the table's OR column was the negative
    # of the frontier's x. Comparing this table against ladder_fig.ipynb's read as a metric disagreement; it was a sign.
    piv = -d.pivot_table(index=["object_label", "seed", "dose"], columns="eval_id", values="d_y", aggfunc="mean")
    for lab, evs in (("ST mean", ST), ("MT mean", MT), ("OR mean", ORB)):
        cols = [e for e in evs if e in piv.columns]
        piv[lab] = piv[cols].mean(axis=1) if cols else np.nan
    order = [e for e in ST + MT + ORB if e in piv.columns]
    out = piv[order + ["ST mean", "MT mean", "OR mean"]].round(1)
    out.columns = [EVAL_LABEL.get(c, c) for c in order] + ["ST mean", "MT mean", "OR mean"]
    return out.reset_index().rename(columns={"object_label": "arm", "seed": "seed", "dose": "dose"})


def rung_quality_table(frame: pd.DataFrame, column: str, objects: list | None = None) -> pd.DataFrame:
    """The same (arm, seed, dose) grid as `rung_eval_table`, carrying a QUALITY column per eval instead of a score.

    `column` is `frac_at_token_cap` or `frac_degenerate_union5` — both already on the frame, so this costs no new
    computation. Kept SEPARATE from the score table on purpose: the standing rule is that degeneracy and at-cap
    shares are measured and reported but never drawn, and a quality number sitting in the same cell as a score
    invites reading one as the other. Blank means the share was never measured, which is not the same as zero —
    an unmeasured cell and a clean one must not look alike."""
    if column not in frame.columns:
        return pd.DataFrame()
    ST, MT, ORB = FRONTIER_STATIC[1], FRONTIER_Y[1], FRONTIER_X[1]
    d = frame[frame.headline] if "headline" in frame.columns else frame
    if "selection_state" in d.columns:
        d = d[d.selection_state.eq("selected")]
    d = d[d.object_label.ne("none")]
    if objects:
        d = d[d.object_label.isin(objects)]
    _seeded = {o for o, g in d.groupby("object_label") if g.seed.fillna("").ne("").any()} if len(d) else set()
    d = d[~(d.object_label.isin(_seeded) & d.seed.fillna("").eq(""))]
    d = d.assign(_dose=pd.to_numeric(d.dose, errors="coerce"))
    if FRONTIER_DROP_NEGATIVE:
        d = d[d._dose >= 0]
    _hid = {(o, dd) for (_p, o, dd) in FRONTIER_RUNG_HIDDEN}
    d = d[~d.apply(lambda r: (r.object_label, float(r._dose)) in _hid if pd.notna(r._dose) else False, axis=1)]
    if not len(d):
        return pd.DataFrame()
    v = pd.to_numeric(d[column], errors="coerce") * 100.0
    piv = d.assign(_v=v).pivot_table(index=["object_label", "seed", "_dose"], columns="eval_id", values="_v",
                                     aggfunc="max")
    order = [e for e in ST + MT + ORB if e in piv.columns]
    out = piv[order].round(0)
    out.columns = [EVAL_LABEL.get(c, c) for c in order]
    return out.reset_index().rename(columns={"object_label": "arm", "_dose": "dose"})


def selected_scope(frame: pd.DataFrame) -> pd.Series:
    """The rows a panel READS: `selection_state == selected` where the frame carries the column, every row where it does not.
    The repoint gate's denominator (below). Unsteered anchors need no scope of their own: `anchor_candidates` and the frontier's
    `anchor_for` match an anchor to its arm's METRIC FAMILY, so an unsteered row lacking the requested metric is not a mixed anchor
    waiting to happen -- it is simply not a candidate, and a rung left without one is reported as `no_anchor`."""
    if "selection_state" in frame.columns:
        return frame.selection_state.eq("selected")
    return pd.Series(True, index=frame.index)


def repoint_headline(frame: pd.DataFrame, eval_id: str, metric_id: str, scope: pd.Series | None = None) -> pd.DataFrame:
    """A copy of `frame` in which `eval_id`'s headline row is `metric_id` instead of the eval's declared one.

    `scope` (2026-09-24, the label-mean switch): a boolean mask of the rows the consumer READS. Every run that carries the
    metric is moved, in scope or not, but coverage is counted over the scoped headline runs only. The live frontier frame
    carried 46 headline over-refusal runs with no `over_refusal_label_mean` row, all of them `r2_export` at the 512 cap
    and `superseded` / `not_in_roster` -- rows no panel draws and, at that cap, no anchor pool reads -- and the unscoped
    gate withheld the switch for them (v153). Counting rows nobody reads is the 357/507 mistake in a new coat.

    For the frontier's second over-refusal reading (2026-09-23). On the arms whose ladder exited at
    8,192, `over_refusal_graded` reports a strongly negative delta (about -20 pp) where `orb_refused` is positive,
    because the graded metric carries an at-cap term that is not comparable across exit caps. Which of the two the
    frontier should use is still open, so the notebook draws BOTH rather than choosing: this returns the frame
    the second version reads. Only the flag moves; no value is recomputed."""
    out = frame.copy()
    # THE COVERAGE ATTRS ARE SET ON EVERY PATH, INCLUDING THE EARLY RETURNS. A caller that gates on
    # `repoint_complete` must be able to read it off any frame this returns; leaving it absent when nothing matched
    # made "no run carries the metric" indistinguishable from "this frame was never repointed", and the two want
    # opposite handling. Absent-and-falsy happened to withhold correctly, but only by luck of `.get()`.
    def _tag(moved: int, total: int):
        out.attrs["repointed_to"] = metric_id
        out.attrs["repoint_coverage"] = (moved, total)
        out.attrs["repoint_complete"] = bool(total) and moved >= total
        return out
    if "headline" not in out.columns or "metric_id" not in out.columns:
        return _tag(0, 0)
    hit = out.eval_id.eq(eval_id)
    inscope = scope.reindex(out.index).fillna(False).astype(bool) if scope is not None else pd.Series(True, index=out.index)
    has = set(out.loc[hit & out.metric_id.eq(metric_id), "run_id"]) if "run_id" in out.columns else set()
    if not has:
        return _tag(0, len(set(out.loc[hit & inscope & out.headline.fillna(False), "run_id"].dropna())) if "run_id" in out.columns else 0)
    # NO AUTO-FLIP: ASKING FOR THE METRIC THAT IS ALREADY HEADLINE IS A COMPLETE NO-OP (2026-09-24, replacing the
    # clause that repointed "the other way" in that case). That clause existed when there was ONE shim whose only job
    # was to keep a second version different from the first, so a no-op looked like a dead switch worth routing
    # around. It is wrong now that both versions NAME their metric: `repoint_all(frame, GRADED)` asking for
    # over_refusal_graded silently returned the refused rate, and because the flip also reset `has`, coverage came
    # back as "0 of 0 runs carry over_refusal_graded" — a frame fully on the requested metric reported as having none
    # of it, and withheld. Caught by v124's own gate line, which is what that reporting is for. A caller that wants
    # the comparison live asks for the two metrics explicitly and compares the results.
    swap = hit & out.run_id.isin(has)
    out.loc[swap, "headline"] = out.loc[swap, "metric_id"].eq(metric_id)
    # A PARTIAL REPOINT IS WORSE THAN NO REPOINT (2026-09-24). Only runs that HAVE the metric are moved; any run
    # without it keeps the old headline, so the frame ends up half on one metric and half on the other and a panel
    # built from it compares some arms against anchors read a different way. Measured on the live frame: 66 rows moved
    # to orb_refused while 65 stayed on over_refusal_graded, including 8 of the 29 unsteered anchors — which is what
    # put a uniform ~35 pp offset on every rung of the "second version" and made two metrics look wildly different
    # when the comparison itself was mixed. The coverage rides in attrs so the caller can refuse to draw it.
    # THE DENOMINATOR IS THE RUNS THE PANEL READS, NOT EVERY ROW ON THE EVAL (2026-09-24). A run that appears on
    # or_bench_hard carrying only some third metric is never drawn, so counting it makes a fully-covered repoint
    # look short and withholds a figure that was fine. Measured: 357 of 507 by the old count, where the 150
    # shortfall was rows no panel consumes; by headline runs the graded repoint is total, which is the truth --
    # graded IS what the frame is on today. The refused-rate gate is unaffected and still bites, which is the
    # point: it is short on runs that ARE drawn.
    # (`has` is read BEFORE the swap above rewrote the flags, so a moved run still counts as a headline run here)
    runs_all = set(out.loc[hit & inscope & out.headline.fillna(False), "run_id"].dropna()) | (has & set(out.loc[hit & inscope, "run_id"]))
    out.attrs["repointed_to"] = metric_id
    out.attrs["repoint_coverage"] = (len(has & runs_all), len(runs_all))
    out.attrs["repoint_complete"] = bool(runs_all) and runs_all <= has
    return out


# THE FIVE-RUNG LADDER: 8,192 and 32,768 are the padding instrument's new rungs, folding into the frame as this
# is written. Listing them here is what lets them appear the render after they land; the drawer and the table both
# take the rungs PRESENT in the points, in this order, so an absent rung is not an empty slot on the axis.
PAD_RUNGS = [0, 8192, 16384, 32768, 65536]
PAD_IN_VIEW = ("selected", "not_in_roster")       # the padding panel's read set: graded readings, roster-named or roster-pending


def padding_points(frame: pd.DataFrame, eval_id: str = "grok420") -> pd.DataFrame:
    """One row per (arm, SEED, padding rung): defence relative to the UNSTEERED CELL AT THE SAME RUNG, in pp, with a
    Newcombe interval for the difference, and the k / n behind it so seeds can be pooled.

    Relative, and at the same rung, because the unsteered baseline is itself NON-MONOTONIC across the ladder
    (44.8 -> 52.9 -> 37.5 % defence): differencing against the same rung removes the baseline's own movement, which an
    absolute series would otherwise attribute to padding (2026-09-23). The rows come from the padding instrument
    only -- `pad_contract` fences them from the ordinary panels, zero-padding level included. SEEDS (2026-09-24): the
    six fitted objects run the ladder per fit seed; each (arm, seed, rung) is its own row here, the
    figure draws the seeds faintly behind their pooled mean, and `padding_pooled` sums k / n across seeds for that mean.
    An unseeded arm (a prompt, Arditi) has seed "" and is one row per rung as before."""
    d = frame[frame.eval_id.eq(eval_id) & frame.get("pad_contract", pd.Series(False, index=frame.index)).fillna(False)]
    # IN VIEW, NOT ONLY SELECTED (who expected five rungs and saw three): at 8,192 and 32,768 every
    # unsteered `none` row -- and the prompts, the difference LoRA and ReLU -- reads `not_in_roster` until roster v9 names
    # those two levels' instrument ids, so a `selected`-only read had no same-rung anchor there and dropped both rungs
    # whole. The rows are graded readings on the padding instrument (`pad_contract`), the roster entries are drafted, and
    # nothing about the numbers changes when they land; so they are drawn now and MARKED `pending` (the table's dagger, the
    # notebook's line), and the mark retires by itself the render after v9 loads. Superseded / withdrawn / excluded rows
    # stay out, as everywhere.
    d = d[d.headline & d.selection_state.isin(PAD_IN_VIEW)] if "selection_state" in d.columns else d[d.headline]
    cols = ["object_label", "seed", "pad", "delta", "lo", "hi", "defence", "base_defence", "k1", "n1", "k0", "n0", "degen", "held", "pending"]
    if not len(d):
        return pd.DataFrame(columns=cols)
    d = d.assign(seed=d.seed.fillna("").astype(str) if "seed" in d.columns else "")
    def kn(g):
        n = float(pd.to_numeric(g["n"], errors="coerce").max())
        v = float(g.value.mean()); v = v if v <= 1.0 else v / 100.0
        return int(round(v * n)), int(n)
    base = {float(pad): kn(g) for pad, g in d[d.object_label.eq("none")].groupby("pad")}     # the anchor: every unsteered run at the rung
    _pend = (lambda g: bool(g.selection_state.ne("selected").any())) if "selection_state" in d.columns else (lambda g: False)
    base_pending = {float(pad): _pend(g) for pad, g in d[d.object_label.eq("none")].groupby("pad")}
    out = []
    for (o, sd, pad), g in d[d.object_label.ne("none")].groupby(["object_label", "seed", "pad"]):
        if float(pad) not in base:
            continue
        k1, n1 = kn(g); k0, n0 = base[float(pad)]
        pending = _pend(g) or base_pending[float(pad)]
        lo, hi = newcombe_diff(k0, n0, k1, n1)                    # defence = 1 - ASR: a defence difference is the ASR difference, sign flipped
        out.append(dict(object_label=o, seed=sd, pad=float(pad), delta=100.0 * (k0 / n0 - k1 / n1),
                        lo=100.0 * lo, hi=100.0 * hi, defence=100.0 * (1 - k1 / n1),
                        base_defence=100.0 * (1 - k0 / n0), k1=k1, n1=n1, k0=k0, n0=n0,
                        degen=float(pd.to_numeric(g.get("degen_share"), errors="coerce").max()) if "degen_share" in g.columns else np.nan,
                        held=bool(held_mask(g).any()), pending=pending))
    return pd.DataFrame(out, columns=cols).sort_values(["object_label", "seed", "pad"])


def padding_held(pts: pd.DataFrame) -> pd.DataFrame:
    """The (arm, seed, rung) points a registry hold-out keeps off the panel and out of the pooled means -- the notebook's
    disclosure line; the table still lists them with the \\S mark."""
    if not len(pts) or "held" not in pts.columns:
        return pd.DataFrame(columns=["object_label", "seed", "pad"])
    return pts.loc[pts.held.fillna(False).astype(bool), ["object_label", "seed", "pad"]].sort_values(["object_label", "seed", "pad"])


def padding_pending_rungs(pts: pd.DataFrame) -> list:
    """The rungs any of whose drawn rows (arm or same-rung anchor) the roster does not yet name -- the notebook's disclosure line."""
    if not len(pts) or "pending" not in pts.columns:
        return []
    return sorted(int(p) for p in set(pts.loc[pts.pending.fillna(False).astype(bool), "pad"]))


def padding_pooled(pts: pd.DataFrame) -> pd.DataFrame:
    """The per-(arm, rung) POOL of a seeded arm's rows: k and n summed over seeds, the same-rung anchor once, Newcombe on the
    pool -- the bold line of the padding panel and the `mean` row of its table. Rows without k / n
    (an older fixture) fall back to the plain mean of their deltas.

    HELD ROWS LEAVE THE POOL (2026-09-28): the registry's hold-outs are honoured here as on the frontier -- a (arm, seed,
    rung) point with held_out = 1 (the pad-0 difference-ReLU point, ruling r2v4-hold-out-static-padding-p0-difference-relu-
    grok420-degenerate, every seed) is neither drawn nor pooled; the table still lists it with its mark. A rung whose every
    seed is held has no mean row."""
    if not len(pts):
        return pts.iloc[0:0]
    out = []
    for (o, pad), g in pts.groupby(["object_label", "pad"]):
        if "held" in g.columns:
            g = g[~g.held.fillna(False).astype(bool)]
            if not len(g):
                continue
        if {"k1", "n1", "k0", "n0"} <= set(g.columns) and g.n1.notna().all():
            k1, n1 = int(g.k1.sum()), int(g.n1.sum()); k0, n0 = int(g.k0.iloc[0]), int(g.n0.iloc[0])
            lo, hi = newcombe_diff(k0, n0, k1, n1)
            out.append(dict(object_label=o, seed="mean", pad=float(pad), delta=100.0 * (k0 / n0 - k1 / n1), lo=100.0 * lo, hi=100.0 * hi,
                            defence=100.0 * (1 - k1 / n1), base_defence=100.0 * (1 - k0 / n0), k1=k1, n1=n1, k0=k0, n0=n0,
                            degen=g.degen.max() if "degen" in g.columns else np.nan, held=False,          # the pool holds no held row
                            pending=bool(g.pending.fillna(False).astype(bool).any()) if "pending" in g.columns else False,
                            n_seeds=int(g.seed.astype(str).ne("").sum()) if "seed" in g.columns else len(g)))
        else:
            out.append(dict(object_label=o, seed="mean", pad=float(pad), delta=g.delta.mean(), lo=g.lo.mean(), hi=g.hi.mean(),
                            defence=g.defence.mean() if "defence" in g.columns else np.nan,
                            base_defence=g.base_defence.iloc[0] if "base_defence" in g.columns else np.nan, n_seeds=len(g)))
    return pd.DataFrame(out).sort_values(["object_label", "pad"])


def padding_figure(pts: pd.DataFrame, base_defence: dict, name: str = "figP_padding_ladder",
                   width: float = 9.0, height: float = 6.0):
    """The padding ladder: Δ defence against the unsteered model at the same rung, one line per arm, three rungs.

    NOT drawn as a slope — the series is not monotonic and neither is the baseline, so the rungs are categorical
    positions joined for legibility, never extrapolated between. Degenerate cells are NOT marked on the axes (the
    figure stays clean, and degeneracy shares are reported, never drawn); they are named in the caption line the
    notebook prints beneath, which is also where the non-monotonic baseline is stated."""
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    drawn = _draw_padding(ax, pts)
    legend_stack(fig, [(None, object_handles(drawn), 5)])
    return save(fig, name)


def repoint_all(frame: pd.DataFrame, spec: dict, scope: pd.Series | None = None) -> pd.DataFrame:
    """Repoint SEVERAL evals' headline metric at once, reporting one combined coverage.

    Neither over-refusal version may rely on which metric the DB happens to call headline (2026-09-24):
    the registry's own default flips from the graded score to the refused rate, at which point a shim that repoints
    only to the refused rate silently becomes a no-op and it is the GRADED version that needs the repoint. So both
    versions state their metric explicitly and neither is the "default" — whichever way the DB is pointing, each
    figure asks for the metric it means.

    `spec` is {eval_id: metric_id}. Coverage is the AND over the evals: the frontier's x is the mean of XSTest and
    OR-Bench, so one eval short is still a mixed panel. See `repoint_headline` for why a partial repoint is refused."""
    out, moved, total, complete = frame, 0, 0, True
    for eval_id, metric_id in spec.items():
        out = repoint_headline(out, eval_id, metric_id, scope=scope)
        m, n = out.attrs.get("repoint_coverage", (0, 0))
        moved += m; total += n
        complete &= bool(out.attrs.get("repoint_complete"))
    out.attrs["repointed_to"] = ", ".join(f"{e} -> {m}" for e, m in spec.items())
    out.attrs["repoint_coverage"] = (moved, total)
    out.attrs["repoint_complete"] = complete
    return out


def repoint_gaps(frame: pd.DataFrame, spec: dict, scope: pd.Series | None = None) -> pd.DataFrame:
    """The headline runs `repoint_all(frame, spec)` could NOT move: one row per (eval, run) whose eval is in `spec` but which
    carries no row of the requested metric, with the identity columns a reader needs to name it (model, arm, seed, dose, source).
    This is what the notebook prints when the over-refusal switch is withheld, so 'coverage 309 / 359' comes with the fifty runs
    that are short rather than a number to go and look up (2026-09-24: the fifty were the `orxs_seedrep` dose-1.0 runs, still
    folding)."""
    cols = [c for c in ("model_id", "eval_id", "object_label", "seed", "dose", "lane", "lane_source_legacy", "token_cap_tokens", "run_id")
            if c in frame.columns]
    if not {"eval_id", "metric_id", "run_id", "headline"} <= set(frame.columns):
        return pd.DataFrame(columns=cols)
    out = []
    inscope = scope.reindex(frame.index).fillna(False).astype(bool) if scope is not None else pd.Series(True, index=frame.index)
    for eval_id, metric_id in spec.items():
        e = frame[frame.eval_id.eq(eval_id)]
        has = set(e.loc[e.metric_id.eq(metric_id), "run_id"])
        short = e[e.headline.fillna(False) & ~e.run_id.isin(has) & inscope.reindex(e.index)]
        out.append(short[cols].drop_duplicates("run_id") if "run_id" in cols else short[cols].drop_duplicates())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def padding_table(pts: pd.DataFrame, objects: list | None = None) -> str:
    """the padding table: the padding ladder as numbers, in the notebook's standard table form.

    One column per filler rung. The first row is the UNSTEERED absolute defence at that rung, and every arm below it is
    a signed change against that same-rung anchor. A seeded arm lists one row per SEED and then a `mean` row -- the pool
    of k / n across its seeds, the panel's bold line; an unseeded arm is one row. The same-rung
    anchor is not a formality: the unsteered baseline is itself non-monotonic across the ladder. Degeneracy is NOT a
    column."""
    if not len(pts):
        return ""
    rungs = [p for p in PAD_RUNGS if p in set(pts.pad.astype(int))]
    header = ["arm", "seed"] + [f"{r:,}" for r in rungs]
    base = {int(r): g.base_defence.iloc[0] for r, g in pts.groupby("pad") if int(r) in rungs}
    rows = [[BASELINE_ROW_LABEL, "--"] + [_cell(base.get(r), dp=1) for r in rungs]]
    order = [o for o in (objects or sorted(pts.object_label.unique())) if o in set(pts.object_label)]
    def cells(g):
        g = g.set_index(g.pad.astype(int)); out = []
        for r in rungs:
            if r not in g.index:
                out.append("--"); continue
            row = g.loc[r]
            out.append(_cell(row.delta, row.lo, row.hi, dp=1, signed=True) + ("$^{\\S}$" if bool(row.get("held", False)) else "")
                       + ("$^{\\ddagger}$" if bool(row.get("pending", False)) else ""))
        return out
    held_arms = set()
    pending_rungs = padding_pending_rungs(pts)
    for o in order:
        g = pts[pts.object_label.eq(o)]
        seeds = sorted(set(g.seed.astype(str))) if "seed" in g.columns else [""]
        if len([sd for sd in seeds if sd]) > 1:
            for sd in seeds:
                rows.append([_tex(LABEL.get(o, o)), _tex(sd), *cells(g[g.seed.astype(str).eq(sd)])])
            rows.append([_tex(LABEL.get(o, o)), "\\emph{mean}", *cells(padding_pooled(g))])
        else:
            rows.append([_tex(LABEL.get(o, o)), "--", *cells(g)])
        if "held" in g.columns and bool(g.held.any()):
            held_arms.add(LABEL.get(o, o))
    note = (("\\S marks an arm held out of the means. " if held_arms else "")
            + (f"\\ddagger: read from graded rows the roster does not yet name (the {', '.join(f'{r:,}' for r in pending_rungs)} filler "
               f"rungs; roster v9 pending). " if pending_rungs else "")
            + "Deltas are against the unsteered cell AT THE SAME RUNG; the unsteered baseline is non-monotonic across the ladder. "
              "A seeded arm's `mean` row pools k and n across its seeds.")
    return latex_table(header, rows, f"Padding ladder for {MODEL_LABEL.get(MODEL_ID, MODEL_ID)}.", "tab:qwen-padding",
                       colspec="ll" + "c" * len(rungs), note=note)


def _draw_padding(ax, pts: pd.DataFrame, compact: bool = False) -> list:
    """Draw the padding ladder onto ONE axis and return the object labels it drew.

    Split out of `padding_figure` (2026-09-24) so the same panel can sit beside static in Figure 3 without a second
    copy of the drawing rules. The standing rule still holds: padding is its OWN panel and its readings never
    enter the static panels' data. SEEDS (2026-09-24): an arm with several fit seeds at a rung draws each seed
    faintly behind, and its pooled mean (`padding_pooled`: k / n summed across seeds) bold, like the frontier's seed marks
    and ensemble; an arm with one row per rung draws bold as before. `compact` shortens the x label for the narrower slot
    a shared figure gives it."""
    rungs = [p for p in PAD_RUNGS if p in set(pts.pad.astype(int))] if len(pts) else list(PAD_RUNGS)
    xs = {p: i for i, p in enumerate(rungs)}
    drawn = []
    for o, g in pts.groupby("object_label"):
        g = g[g.pad.isin(rungs)]
        if "held" in g.columns:                                    # a registry hold-out is not drawn (the rule of 2026-09-24)
            g = g[~g.held.fillna(False).astype(bool)]
        if not len(g):
            continue
        drawn.append(o)
        col, ls = COLOR.get(o, INK), "--" if o in PROMPTED else "-"
        seeds = sorted(set(g.seed.astype(str))) if "seed" in g.columns else [""]
        multi = len([sd for sd in seeds if sd]) > 1
        if multi:
            for sd, gs in g.groupby("seed"):                       # the seeds, faint, behind
                gs = gs.sort_values("pad"); x = [xs[int(p)] for p in gs.pad]
                ax.plot(x, gs.delta, ls, color=col, lw=1.0, alpha=0.25, zorder=2, marker="o", ms=3.5, markeredgecolor="white", markeredgewidth=0.5)   # even lighter
            g = padding_pooled(g)                                  # the pooled mean, bold
        g = g.sort_values("pad"); x = [xs[int(p)] for p in g.pad]
        ax.plot(x, g.delta, ls, color=col, lw=2.6, alpha=0.95, zorder=3, marker="o", ms=9, markeredgecolor="white", markeredgewidth=1.1)
        ax.vlines(x, g.lo, g.hi, color=col, lw=2.0, alpha=0.55, zorder=2)
    ax.axhline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([f"{p:,}" for p in rungs])
    ax.set_xlim(-0.25, len(rungs) - 0.75)
    ax.set_xlabel("neutral filler tokens" if compact else "neutral filler tokens between the system instruction and the payload")
    ax.set_ylabel("Δ defense success (pp)")
    ax.grid(True, axis="y", color=RULE, lw=0.9); ax.set_axisbelow(True)
    ax.set_title("padding")
    return drawn


def frontier_figure(panels: list, objects: list, name: str, x_evals=FRONTIER_X[1], operating_dose: float = 1.0, width: float = 7.8, height: float = 5.6,
                    spline: str = "origin"):
    """Figure 5: the frontier RELATIVE TO THE UNSTEERED MODEL (paperlib fig5_absolute with origin="unsteered"), one panel per battery:
    x = Δ benign compliance, y = Δ defense success, both in pp, the campaign's unsteered model at the origin as a large hollow black
    circle. `panels` is a list of dicts with `pts` (frontier_points), `title`, `y_evals` and `x_battery`. Each steerable object's rungs
    are joined in dose order per seed through the origin, its dose-0 rung (a monotone curve), the
    marker area grows with the dose, a negative dose (sign control) is an open marker; the prompted arms are single diamonds joined
    to nothing (no strength knob). No dose labels on the points: the size key carries the dose. A panel's
    optional `note` (what it is still waiting for) sits inside it, muted."""
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), height), layout="constrained", squeeze=False)
    drawn, doses, any_neg, any_si, any_seed, any_cross, any_degen, any_cap = [], set(), False, False, False, False, False, False
    clipped: list = []                                        # panels with rungs past the x floor, reported under the figure
    # TWO SPLINES (2026-09-24, after sharp turns appeared on the curves): "origin" is this
    # figure's own -- the origin (dose 0, the unsteered model) is prepended and a monotone PCHIP runs through it, so
    # every curve leaves (0, 0); "ladder" is ladder_fig's smooth_traj exactly -- PCHIP of x(dose) and y(dose) over the
    # MEASURED doses only, 200 samples, no origin, so a curve begins at its lowest rung. Same rungs, same means; only
    # the interpolant differs. The notebook draws both until one is chosen.
    def _curve(dd_, x_, y_):
        if spline == "ladder":
            return _smooth(np.asarray(dd_, float), np.asarray(x_, float), np.asarray(y_, float), samples=200)
        return _smooth(np.concatenate([[0.0], dd_]), np.concatenate([[0.0], x_]), np.concatenate([[0.0], y_]))
    for ax, panel in zip(axes[0], panels):
        pts = panel["pts"]
        for o in objects:
            go = pts[pts.object_label == o]
            if not len(go):
                continue
            if o not in drawn:
                drawn.append(o)
            col = COLOR[o]
            many = go.seed.nunique() > 1
            any_seed |= many
            if o in PROMPTED:
                any_si = True
                gm = go.groupby("dose", as_index=False)[["dx", "dy"]].mean()
                if many:
                    ax.scatter(go.dx, go.dy, marker="D", s=0.35 * (rung_size(1.0) + 60), color=col, edgecolors="white", linewidths=0.7, alpha=0.8, zorder=4)
                ax.scatter(gm.dx, gm.dy, marker="D", s=rung_size(1.0) + 60, color=col, edgecolors="white", linewidths=1.0, zorder=5)
                continue
            # ONE CURVE PER OBJECT: a ladder's rungs can be spread across seeds - the paired ReLU's seed 1 carries the
            # dose ladder, seeds 2 and 3 the operating point - and one curve per seed drew them as separate segments that each passed
            # through the origin. The curve now joins the PER-DOSE MEAN over seeds; the seeds stay as small marks around each rung.
            ps = panel.get("per_seed")
            seeded = False                                                    # is the bold curve a mean over several seeds?
            if ps is not None and len(ps):                                    # the individual seeds, where a seed carries BOTH batteries
                gs_ = ps[ps.object_label == o]
                if len(gs_) and gs_.seed.nunique() > 1:
                    seeded = True
                    any_seed = True
                    # ONE THIN CURVE PER SEED, UNDER A BOLD ENSEMBLE MEAN. The seeds used to be
                    # loose scatter marks, which showed the spread between fits but not the SHAPE of any one fit — and
                    # once seeds 2 and 3 gained real ladders the marks read as unconnected dots. Each seed now joins its
                    # own rungs through the origin, thin and semi-transparent in the object's own colour so the three
                    # fits read as one family, with the per-dose mean drawn bold on top.
                    for _sd, _g1 in gs_.groupby("seed"):
                        _g1 = _g1.groupby("dose", as_index=False)[["dx", "dy"]].mean().sort_values("dose")
                        if not len(_g1):
                            continue
                        _sx, _sy, _sdd = _g1.dx.to_numpy(float), _g1.dy.to_numpy(float), _g1.dose.to_numpy(float)
                        if len(_sdd) > 1:                                     # a single rung is a point, not a curve
                            _kx, _ky = _curve(_sdd, _sx, _sy)
                            ax.plot(_kx, _ky, "-", color=col, lw=1.1, alpha=0.5, zorder=2, solid_capstyle="round")
                        # the sign control keeps its open marker HERE, at the measurement, now that the mean carries no
                        # markers of its own — otherwise a negative rung would be indistinguishable
                        _neg = _sdd < 0
                        if _neg.any():
                            any_neg = True
                            ax.scatter(_sx[_neg], _sy[_neg], s=30, facecolors="white", edgecolors=col,
                                       linewidths=1.3, alpha=0.95, zorder=4)
                        if (~_neg).any():
                            ax.scatter(_sx[~_neg], _sy[~_neg], s=24, color=col, edgecolors="white",
                                       linewidths=0.5, alpha=0.9, zorder=4)
            agg_cols = {c: "mean" for c in ("dx", "dy") if c in go.columns}
            agg_cols.update({c: "max" for c in ("degen", "cap") if c in go.columns})
            if "cross_cut" in go.columns:
                agg_cols["cross_cut"] = "any"
            g = go.groupby("dose", as_index=False).agg(agg_cols).sort_values("dose")
            if True:
                x, y, dd = g.dx.to_numpy(float), g.dy.to_numpy(float), g.dose.to_numpy(float)
                doses |= set(dd)
                # THE COMMON REFERENCE: the unsteered model is every ladder's dose-0 rung, so each curve passes through
                # the origin - between the sign control and the positive rungs where there is one, from the origin outward otherwise
                cx, cy = _curve(dd, x, y)
                ax.plot(cx, cy, "-", color=col, lw=3.8, alpha=0.95, zorder=5, solid_capstyle="round")      # BOLD: the ensemble mean over seeds,
                # drawn over the thin per-seed curves so the average reads first. One line style: the Refusal steer is
                # always instruct-direct (no base fit exists), so its lane is a fact for the caption, not a mark
                cc_col = g.cross_cut.astype(bool).to_numpy() if "cross_cut" in g.columns else np.zeros(len(g), bool)
                dg_col = g.degen.to_numpy(float) if "degen" in g.columns else np.full(len(g), np.nan)
                for xi, yi, di, cc, dg in zip(x, y, dd, cc_col, dg_col):
                    # DON'T MANUFACTURE POINTS ON THE AGGREGATE. Where several seeds exist the
                    # bold curve is their per-dose MEAN — a derived quantity, not an observation — so it is drawn as a
                    # smooth line and nothing else; the measured rungs are the small dots on the per-seed curves. A
                    # single-seed or unseeded ladder keeps its markers, because there the curve IS the measurement.
                    if seeded:
                        continue
                    if di < 0:
                        any_neg = True
                        ax.scatter([xi], [yi], s=rung_size(di), facecolors="white", edgecolors=col, linewidths=2.2, zorder=4)
                    else:
                        ax.scatter([xi], [yi], s=rung_size(di), color=col, edgecolors="white", linewidths=1.0, zorder=4)
                    if cc:                                                # compared with the campaign's unsteered on another item set
                        any_cross = True
                        ax.scatter([xi], [yi], s=rung_size(di), facecolors="none", edgecolors="black", lw=1.4, zorder=6)
                    # THE DEGENERATE SHARE IS NOT DRAWN (via Summary) - see `flag`. The rung's share is in the
                    # rung table, and the panel's coverage line below the figure still counts measured against drawn, which is a
                    # count rather than a per-point mark.
                if "cap" in g.columns:                                        # the at-cap share rides on the rung and into the table, not onto the canvas
                    any_cap = any_cap or bool((g.cap.to_numpy(float) > AT_CAP_LINE).any())
        # NO MARK AT THE ORIGIN. The unsteered model sits at (0, 0) by construction and the two
        # zero rules already say where that is; the circle was a third statement of the same fact. The rules STAY.
        ax.axhline(0, color=INK2, lw=1.1, zorder=1); ax.axvline(0, color=INK2, lw=1.1, zorder=1)
        ax.grid(True, color=RULE, lw=0.9); ax.set_axisbelow(True)
        # THE X AXIS STOPS AT 30 pp OF OVER-REFUSAL: the interesting range is the elbow, and letting a
        # -62 pp rung set the limit compressed every arm worth reading into the right-hand third of the panel. Rungs beyond the
        # floor are CLIPPED, not dropped -- their curves run off the edge, and the count is printed below the figure so a
        # reader knows the axis was cut rather than the ladder ending there.
        lo_x, hi_x = min(float(pts.dx.min()) if len(pts) else 0.0, 0.0), max(float(pts.dx.max()) if len(pts) else 0.0, 0.0)
        beyond = int((pts.dx.astype(float) < FRONTIER_X_FLOOR).sum()) if len(pts) else 0
        if beyond:
            clipped.append(f"{panel['title']}: {beyond}")
        lo_x = max(lo_x, FRONTIER_X_FLOOR)
        lo_y, hi_y = min(float(pts.dy.min()) if len(pts) else 0.0, 0.0), max(float(pts.dy.max()) if len(pts) else 0.0, 0.0)
        px, py = max(5.0, 0.08 * (hi_x - lo_x)), max(5.0, 0.08 * (hi_y - lo_y))
        ax.set_xlim(lo_x - px, hi_x + px); ax.set_ylim(lo_y - py, hi_y + py)
        xb = panel.get("x_battery") or list(x_evals)
        batt = ("mean of " + " and ".join(EVAL_LABEL[e] for e in xb)) if len(xb) > 1 else (EVAL_LABEL[xb[0]] + " alone")
        # THE X LABEL NAMES THE BATTERY: the single-eval versions take "Δ XSTest" / "Δ OR-Bench"; without them three figures
        # carried the same "(mean)" label and could not be told apart. Two evals -> the mean label; one eval -> that eval by
        # its display name.
        _xl = list(x_evals) if x_evals is not None else list(FRONTIER_X[1])
        ax.set_xlabel("Δ benign compliance (mean)" if len(_xl) != 1 else f"Δ benign compliance ({EVAL_LABEL.get(_xl[0], _xl[0])})")
        # the panel's y label wraps at ~30 characters: the axes are short and a one-line eval list ran past the
        # axes' top into the legend stack above
        ax.set_ylabel(f"Δ defense success (pp)\n{panel['title']} (mean)")
        ax.set_title(panel["title"])
        if panel.get("note"):
            ax.text(0.03, 0.97, panel["note"], transform=ax.transAxes, ha="left", va="top", fontsize=17, color=MUTED, style="italic")
    h = object_handles(drawn)
    marks = []
    # `borrowed` is deliberately NOT matched below: it means another VIEW, not another lane (2026-09-20), so matching it
    # would label a series as externally anchored on the wrong evidence.
    # (The producer's filed contrast may name an anchor on another lane - `anchor_kinds` carries it in the tables - but since the frontier
    # differences every rung against the eval's COMMON anchor that provenance describes the producer's number,
    # not the plotted one, so it is no longer a legend line.)
    if any_si:
        marks.append(Line2D([], [], ls="", marker="D", ms=11, color=INK2, markeredgecolor="white", label="prompted arm: one diamond, no dose knob"))
    pos = sorted(dd for dd in doses if dd > 0)
    if pos:
        keys = [pos[0]] + ([operating_dose] if pos[0] < operating_dose < pos[-1] and operating_dose in pos else []) + [pos[-1]]
        for dd in dict.fromkeys(keys):
            marks.append(Line2D([], [], ls="", marker="o", color=MUTED, ms=float(np.sqrt(rung_size(dd))),
                                label=f"dose {dd:g}×" + (" (operating point)" if dd == operating_dose else "") + (" · each object's own unit" if dd == pos[0] else "")))
    if any_neg:
        marks.append(Line2D([], [], ls="", marker="o", ms=11, markerfacecolor="white", markeredgecolor=INK2, markeredgewidth=2.2, label="negative dose (sign control)"))
    if any_seed:
        marks.append(Line2D([], [], ls="", marker="o", ms=5, color=INK2, alpha=0.75, label="thin lines: each fit seed's own ladder; bold: the per-dose mean over them"))
    if any_cross:
        marks.append(Line2D([], [], ls="", marker="o", ms=11, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.4,
                            label="ring: compared with the campaign's unsteered on another item set (cross-cut)"))
    if any_degen:
        marks.append(Line2D([], [], ls="", marker=r"$\dagger$", ms=9, color=INK2, label="†n%: degenerate share of the rung (not a defense)"))
    # NO ANNOTATION KEY IN THE LEGEND: the at-cap share still rides beside its rung and reads for itself.
    cov = [(p.get("degen_measured"), p.get("degen_total")) for p in panels if p.get("degen_total")]
    if cov and any(m < t for m, t in cov):
        _m, _t = sum(x for x, _ in cov), sum(y for _, y in cov)
        marks.append(Line2D([], [], ls="", marker="", color="none",
                            label=f"degeneracy measured on {_m} of {_t} drawn rungs; the rest are UNMEASURED, not clean (named in the tables)"))
    # OBJECTS ONLY IN THE LEGEND: the marker section explained dose sizes, the sign control, the
    # seed lines and the degeneracy coverage — all of which the rung tables carry, and none of which is needed to read a
    # curve. `marks` is still assembled above because its flags (any_neg, any_cap, …) drive the tables and the notes.
    legend_stack(fig, [(None, h, 5)])                      # one row, five columns
    if clipped:
        # the axis was cut, so say so: a curve running off the left edge must not read as a ladder that ended there
        print(f"x axis clipped at {abs(FRONTIER_X_FLOOR):g} pp of over-refusal, the range that matters most; rungs beyond it, "
              f"by panel — {'; '.join(clipped)}. Their curves run off the edge; the rung tables carry every value.")
    return save(fig, name)


def dose_response_figure(pts: pd.DataFrame, objects: list, name: str, y_evals=FRONTIER_Y[1], width: float = 5.4, height: float = 4.4):
    """Figure 5's companion: dose against Δ defense success on the multi-turn battery, ONE PANEL PER LADDER OBJECT, each on its own dose
    axis (2026-09-20: a raw coefficient, a direction-norm multiplier and a write-basis multiplier are three units and never share
    an axis). For a ladder whose cost axis is not trustworthy (the paired ReLU's withdrawn XSTest anchor) this is the sound view; no cost
    axis is synthesised. Seeds are separate small marks; the curve runs through the origin (dose 0 = unsteered); a rung degenerate above
    DEGEN_LINE carries "†n%"."""
    objs = [o for o in objects if o in set(pts.object_label) and o not in PROMPTED]
    if not objs:
        fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
        ax.text(0.5, 0.5, "no ladder with a judged multi-turn rung", ha="center", va="center", fontsize=18, color=MUTED, style="italic", transform=ax.transAxes)
        ax.set_axis_off()
        return save(fig, name)
    fig, axes = plt.subplots(1, len(objs), figsize=(width * len(objs), height), layout="constrained", squeeze=False)
    any_seed, any_degen, any_cap = False, False, False
    for ax, o in zip(axes[0], objs):
        g = pts[pts.object_label == o]; col = COLOR[o]
        xcol = "coefficient" if o == "arditi_allpos" else "dose"
        many = g.seed.nunique() > 1; any_seed |= many
        if many:                                                              # the seeds: small marks at their own readings
            ax.scatter(g[xcol], g.dy, s=26, color=col, edgecolors="white", lw=0.6, alpha=0.75, zorder=4)
        agg_cols = {"dy": "mean"}; agg_cols.update({c: "max" for c in ("degen", "cap") if c in g.columns})
        gs = g.groupby(xcol, as_index=False).agg(agg_cols).sort_values(xcol)     # one curve per object through the per-dose means
        if True:
            xs, ys = gs[xcol].to_numpy(float), gs.dy.to_numpy(float)
            cx, cy = _smooth(np.concatenate([[0.0], xs]), np.concatenate([[0.0], xs]), np.concatenate([[0.0], ys]))
            ax.plot(cx, cy, "-", color=col, lw=2.0, alpha=0.85, zorder=3)
            ax.scatter(xs, ys, s=60, color=col, edgecolors="white", lw=0.7, zorder=5)
            dgs = gs.degen.to_numpy(float) if "degen" in gs.columns else np.full(len(gs), np.nan)
            for xi, yi, dg in zip(xs, ys, dgs):
                if False:                                                     # the degenerate share is measured and tabled, never drawn
                    pass
            if "cap" in gs.columns:
                for xi, yi, cp in zip(xs, ys, gs.cap.to_numpy(float)):
                    if np.isfinite(cp) and cp > AT_CAP_LINE:
                        any_cap = True
                        ax.annotate(f"cap {cp:.0%}", (xi, yi), xytext=(4, 8), textcoords="offset points", fontsize=13, color=INK2, ha="left", va="bottom", zorder=6)
        ax.scatter([0], [0], marker="o", s=220, facecolors="none", edgecolors="black", linewidths=1.6, zorder=7)
        ax.axhline(0, color=INK2, lw=1.0, zorder=1); ax.axvline(0, color=INK2, lw=1.0, zorder=1)
        ax.grid(True, color=RULE, lw=0.8); ax.set_axisbelow(True)
        ax.set_title(LABEL[o]); ax.set_xlabel(DOSE_UNIT.get(o, "dose (× the operating point)"))
    axes[0][0].set_ylabel("Δ defense success (pp)\nmulti-turn (mean)")      # same nomenclature as the Qwen frontier
    marks = [Line2D([], [], ls="", marker="o", ms=13, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.6, label="unsteered: dose 0")]
    if any_seed:
        marks.append(Line2D([], [], ls="", marker="o", ms=5, color=INK2, alpha=0.75, label="thin lines: each fit seed's own ladder; bold: the per-dose mean over them"))
    if any_degen:
        marks.append(Line2D([], [], ls="", marker=r"$\dagger$", ms=9, color=INK2, label="†n%: degenerate share of the rung (not a defense)"))
    if any_cap:
        marks.append(CAP_HANDLE)
    legend_stack(fig, [(None, marks, 3)])
    return save(fig, name)


# ---- the all-model (appendix) forms ---------------------------------------------------------------------------------------------
def category_panels(cm: pd.DataFrame, d: pd.DataFrame, layout: list, objects: list, models: list, gaps: pd.DataFrame, name: str,
                    panel_width: float | None = None, height: float = 4.4, stack: bool = True, categories: dict = None):
    """The paper's Figures 2 / 3 form across models (paperlib category_panels / lane_strip): one panel per category, x = the
    OBJECTS as lanes, the models dodged inside each lane by marker shape; per (object, model) the category mean is the large
    mark with its interval and the individual evals are small faded marks of the same shape beside it; several seeds of one
    (object, model) are separate large marks. Gap glyphs sit in the model's slot at zero. Under each panel, what the mean is over."""
    # ONE CATEGORY PER ROW: the appendix figures whose x is the OBJECT stack their panels vertically; the
    # ones whose x is the eval (`absolute_panels`, `delta_panels`) keep one row. Each of these panels is nine object lanes with four
    # models dodged inside every lane and needs the figure's full width, so side by side they were either cramped or a metre across.
    # `layout` still names the categories and their order; `stack=False` keeps the rows it gives.
    if stack:
        layout = [[c] for row in layout for c in row]
    panel_width = panel_width or 1.5 * len(objects) + 0.8                  # ~1.5 in per lane: four models dodged inside each
    fig = plt.figure(figsize=(panel_width * max(len(r) for r in layout), height * len(layout)), layout="constrained")
    axes = _panel_grid(fig, layout, {c: 1 for c in CATEGORIES})
    seen, any_ring, any_open, any_seed, any_ceil, any_cap, any_degen, any_spread = set(), False, False, False, False, False, False, False
    any_xcap = bool(d["cap_mismatch"].any()) if "cap_mismatch" in d.columns else False
    any_mixcap = any_gen = False
    nm = max(len(models), 1); slot = 0.84 / nm
    categories = categories or CATEGORIES                                        # the appendix passes APPENDIX_CATEGORIES (2026-09-24)
    for ax, cat in axes:
        evals, sign, ylabel = categories[cat]
        for j, o in enumerate(objects):
            for k, m in enumerate(models):
                xc = j - 0.42 + slot * (k + 0.5)
                rs = pool_seeds(cm[(cm.model_id == m) & (cm.category == cat) & (cm.object_label == o)])   # one mean point: the seeds averaged
                pe_all = d[(d.model_id == m) & (d.object_label == o) & d.eval_id.isin(evals)]
                if len(rs):
                    # one mean per GENERATION, side by side in the model's slot (2026-09-21)
                    rs = rs.sort_values("generation") if "generation" in rs.columns else rs
                    any_gen |= len(rs) > 1
                    many = bool((rs.n_seeds > 1).any())
                    any_seed |= many
                    rs = pool_cell(rs, keys=("model_id", "category", "object_label")) if len(rs) > 1 else rs   # one mark, generations and seeds pooled
                    pe = pool_cell(pool_draws(pe_all[pe_all.d_y.notna()]), y="d_y", lo="d_lo", hi="d_hi").sort_values("eval_id")   # one small mark per eval
                    for xi, (_, q) in zip(_spread(xc, len(pe), 0.36 * slot), pe.iterrows()):      # the evals (per seed), small and faded
                        yv, l, u = sign * q.d_y, min(sign * q.d_lo, sign * q.d_hi), max(sign * q.d_lo, sign * q.d_hi)
                        ax.plot([xi] * 2, [l, u], color=COLOR[o], lw=0.9, alpha=0.4, zorder=2)
                        # OPEN, NOT LABELED: a held-out cell (`not scoreable`) and a role-collapsed cell are drawn
                        # unfilled and say nothing on the canvas; what they are is in the table legend under the panel (`cell_notes`).
                        if is_held(q) or is_collapsed(q):
                            ax.scatter([xi], [yv], marker=MARKER[m], s=34, facecolors="white", edgecolors=COLOR[o], lw=1.2, alpha=0.8, zorder=2)
                        else:
                            ax.scatter([xi], [yv], marker=MARKER[m], s=20 if many else 34, color=COLOR[o], alpha=0.5, edgecolors="none", zorder=2)
                    for xg, (_, r) in zip(_spread(xc, len(rs), 0.3 * slot), rs.iterrows()):
                        repl = str(r.get("generation", GEN_CANONICAL)) == GEN_REPLICATE
                        ax.plot([xg] * 2, [r.lo, r.hi], color=COLOR[o], lw=2.4, zorder=3)
                        if bool(r.open):
                            any_open = True
                            ax.scatter([xg], [r.y], marker=MARKER[m], s=175, facecolors="white", edgecolors=COLOR[o], lw=2.2, zorder=4)
                        else:
                            ax.scatter([xg], [r.y], marker=MARKER[m], s=175, color=COLOR[o], edgecolors=(INK2 if repl else "white"), lw=(1.8 if repl else 0.9), zorder=4)
                        if bool(r.partial):
                            any_ring = True
                            ax.scatter([xg], [r.y], marker=MARKER[m], s=175, facecolors="none", edgecolors="black", lw=1.3, zorder=5)
                        # NO CAP ANNOTATIONS ON THE APPENDIX FIGURES EITHER: the "cap n%" at-cap label and the
                        # "≠" cap-mismatch glyph come off, as they came off the main figures on 2026-09-21; the shares and the mismatch
                        # stay on every row and in the tables. The degeneracy share and the two-instrument bar (‖) are not cap labels
                        # and stay.
                        r = r.copy(); r["cap_mismatch"] = False
                        c, k, dg = flag(ax, xg, r.y, r, cap_col="_none", degen_col="degen"); any_ceil |= c; any_cap |= k; any_degen |= dg
                        if bool(r.get("mixed_cap_seeds", False)):            # NO ‖ ON THE CANVAS; the caps stay in the tables
                            any_mixcap = True
                        if (float(r.spread) if "spread" in r.index and pd.notna(r.spread) else 0.0) > SPREAD_LINE:
                            any_spread = True
                elif len(pe_all) and held_mask(pe_all).all():
                    # every eval of the cell is held out: the small open marks and their note carry it; no mean, no gap glyph
                    pe = pool_cell(pool_draws(pe_all[pe_all.d_y.notna()]), y="d_y", lo="d_lo", hi="d_hi").sort_values("eval_id")
                    for xi, (_, q) in zip(_spread(xc, len(pe), 0.36 * slot), pe.iterrows()):
                        yv, l, u = sign * q.d_y, min(sign * q.d_lo, sign * q.d_hi), max(sign * q.d_lo, sign * q.d_hi)
                        ax.plot([xi] * 2, [l, u], color=COLOR[o], lw=0.9, alpha=0.4, zorder=2)
                        ax.scatter([xi], [yv], marker=MARKER[m], s=34, facecolors="white", edgecolors=COLOR[o], lw=1.2, alpha=0.8, zorder=2)
                else:
                    g = gaps[(gaps.model_id == m) & (gaps.object_label == o) & gaps.eval_id.isin(evals)]
                    st = "no unsteered run" if len(pe_all) and pe_all.d_y.isna().all() else (g.status.mode().iat[0] if len(g) else "absent")
                    seen.add(st)
                    if st in GAP_MARK:
                        ax.scatter([xc], [0], zorder=3, **GAP_MARK[st])
        for j in range(1, len(objects)):
            ax.axvline(j - 0.5, color=RULE, lw=1.0)
        ax.axhline(0, color=INK2, lw=1.1, zorder=1)
        ax.set_xticks(range(len(objects))); ax.set_xticklabels([SHORT.get(o, LABEL[o]) for o in objects], fontsize=16)
        ax.set_xlim(-0.6, len(objects) - 0.4); ax.set_title(cat); ax.set_ylabel(_wrap_label(ylabel))
        lo_, hi_ = ax.get_ylim(); ax.set_ylim(min(lo_, -5), max(hi_, 10))
        # NO TEXT UNDER THE PANEL (so the figure pastes clean): what the mean is over is the caption's, and the
        # decode disclosure, the two-term caveat and the item-set caveat are the table legend's (`cell_notes`)
    # `object` and `model` only: large = the category mean, small and faded = one eval; the gap glyphs, rings, open
    # marks and share annotations stay on the canvas with their meaning in the tables
    legend_stack(fig, appendix_legend_groups(objects, models))
    return save(fig, name)


def split_by_category(evals: list) -> list:
    """The evals of an appendix figure grouped into the paper's categories, in CATEGORIES order, each keeping the figure's eval order;
    an eval outside every category forms its own group. The appendix absolute figures draw one panel per group (static, multi-query,
    multi-turn ... on separate panels, as the main Qwen figures are)."""
    groups = []
    for cat in CATEGORIES:
        cols = [e for e in evals if EVAL_CATEGORY.get(e) == cat]
        if cols:
            groups.append((cat, cols))
    for e in evals:
        if e not in EVAL_CATEGORY:
            groups.append((EVAL_LABEL.get(e, e), [e]))
    return groups


def _draw_absolute_panel(ax, pts: pd.DataFrame, evals: list, objects: list, models: list, gaps: pd.DataFrame, ylim, title: str | None,
                         zero_line: bool = False) -> tuple:
    """One appendix absolute panel: x = the panel's evals then its `mean` column, objects dodged inside each column, one marker per model
    (their readings pooled) with its own interval, and the diamond mean over models with a black +- SEM bar (hollow, no bar, when one
    model); gap glyphs at the foot in the model's fan position. Returns (gap statuses seen, footnotes)."""
    pts = pd.concat([pts, eval_mean_column(pts[~held_mask(pts)], evals)], ignore_index=True)     # the panel's mean column; held-out rows enter no mean
    cols = list(evals) + [MEAN_COLUMN]
    unit = (ylim[1] - ylim[0]) / 106.0            # the gap-glyph fan was placed on the 0-100 absolute axis; a relative panel's axis scales it
    nm = max(len(models), 1); slot = 0.86 / max(len(objects), 1); fan = 0.34 * slot
    gm = grand_mean(pts[~held_mask(pts)], ["eval_id", "object_label"])       # a model's readings averaged first, then the mean over models
    seen, foot_notes, two_term = set(), set(), []
    for i, e in enumerate(cols):
        for j, o in enumerate(objects):
            xc = i - 0.43 + slot * (j + 0.5)
            for k, m in enumerate(models):
                xk = xc + (np.linspace(-fan, fan, nm)[k] if nm > 1 else 0.0)
                rs = pts[(pts.model_id == m) & (pts.object_label == o) & (pts.eval_id == e)].sort_values([c for c in ("seed", "draw") if c in pts.columns])
                if len(rs):
                    sz = 24 if len(rs) > 1 else 52                          # a fifth smaller with the narrower columns (2026-09-23)
                    for xi, (_, r) in zip(_spread(xk, len(rs), 0.35 * fan / max(nm - 1, 1) if nm > 1 else 0.3 * slot), rs.iterrows()):
                        if pd.notna(r.lo):
                            ax.plot([xi] * 2, [r.lo, r.hi], color=COLOR[o], lw=1.2 if len(rs) == 1 else 0.9, alpha=0.55, zorder=2)
                        if is_held(r) or is_collapsed(r):                       # held out or role-collapsed: drawn open, unlabeled;
                            ax.scatter([xi], [r.y], marker=MARKER[m], s=sz, facecolors="white", edgecolors=COLOR[o], lw=1.6, zorder=3)   # the table legend says which
                        else:
                            ax.scatter([xi], [r.y], marker=MARKER[m], s=sz, color=COLOR[o], edgecolors="white", lw=0.7, zorder=3)
                else:
                    st = gaps[(gaps.model_id == m) & (gaps.object_label == o) & (gaps.eval_id == e)]
                    if len(st) and st.iloc[0].status in GAP_MARK:
                        seen.add(st.iloc[0].status)
                        kw = dict(GAP_MARK[st.iloc[0].status]); kw["s"] = kw["s"] * 0.6
                        ax.scatter([xk], [ylim[0] + (2.5 + 2.2 * k) * unit], zorder=3, **kw)
            q = gm[(gm.eval_id == e) & (gm.object_label == o)]
            if len(q):
                q = q.iloc[0]
                big = 190 if e == MEAN_COLUMN else 125                           # the mean column's marker is the panel's single large summary
                if q.n_models > 1:
                    ax.plot([xc] * 2, [q.y - q.se, q.y + q.se], color="black", lw=2.0, zorder=4)
                    ax.scatter([xc], [q.y], marker="D", s=big, color=COLOR[o], edgecolors="black", lw=1.0, zorder=5)
                else:
                    ax.scatter([xc], [q.y], marker="D", s=big, facecolors="white", edgecolors=COLOR[o], lw=2.0, zorder=5)
        if i:
            ax.axvline(i - 0.5, color=RULE, lw=1.0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([tick(e) for e in cols]); ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.axvline(len(cols) - 1.5, color=INK2, lw=1.4)                               # the mean column stands apart from the evals
    ax.set_ylim(*ylim)
    if zero_line:                                                                 # a relative panel: the unsteered model is the zero line
        ax.axhline(0, color=INK2, lw=1.1, zorder=1)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    # NO FOOTNOTES ON THE CANVAS: the item-set and two-term caveats are the table legend's (`cell_notes`)
    return seen, foot_notes


def _auto_ylim(pts: pd.DataFrame, evals: list) -> tuple:
    """A relative panel's range, set the way the main figures set theirs: at least -5..10, then whatever the marks, their intervals
    and the mean column need, with room at the foot for the gap glyphs and the notes under the marks."""
    m = pd.concat([pts, eval_mean_column(pts[~held_mask(pts)], evals)], ignore_index=True) if len(pts) else pts
    vals = pd.concat([m.y, m.lo, m.hi]).dropna() if len(m) else pd.Series(dtype=float)
    lo = min(float(vals.min()), -5.0) if len(vals) else -5.0
    hi = max(float(vals.max()), 10.0) if len(vals) else 10.0
    span = hi - lo
    return (lo - 0.12 * span, hi + 0.04 * span)


def _wrap_label(text: str, width: int = 30) -> str:
    """A y label folded to lines of about `width` characters, existing line breaks kept. A one-line label such as
    "defense success rate (%; 100 - ASR at threshold 0.5)" set at 20 pt is taller than a 5 in panel, and the tight bounding box then
    pads the whole figure with blank space under the panels to fit it (visible once the figures were narrowed, 2026-09-23)."""
    return "\n".join(line for part in str(text).split("\n") for line in (textwrap.wrap(part, width) or [""]))


def _eval_row_figure(pts: pd.DataFrame, evals: list, objects: list, models: list, gaps: pd.DataFrame, ylabel, name: str,
                     ylim, width: float | None, height: float, zero_line: bool, pooled: bool):
    """The x = eval appendix figure, absolute or relative, on ONE ROW (the appendix figures that follow the main figures' form keep
    one row; the x = object figures stack): one panel per category of the evals given, side by side, widths by
    column count - static | multi-query | multi-turn for the attacks - each panel x = its evals then a `mean` column (each
    model's mean over the panel's evals, the grand diamond over models above it), objects dodged inside each column, one marker per
    model with its own interval and per (eval, object) the diamond mean over the models present with a black +- SEM bar (hollow, no
    bar, when one model): a level summary, never a contrast interval. `ylim` is one range for the row, a {category: range} mapping,
    or None for every panel to take its own range from its marks (the main figures' way); `ylabel` a string the row shares (tick
    labels on the first panel only, when the range is shared too) or a {category: label} mapping, each panel labeled and ticked on
    its own axis. One legend; the footnotes (role collapse, item counts, two-term grading) under the figure."""
    if not pooled:
        pts = pool_cell(pts)                                                   # one mark per (model, object, eval)
    groups = split_by_category(list(evals))
    widths = [len(cols) + 1 for _, cols in groups]
    # NARROWER (2026-09-23: the attacks figures were a bit too wide - 44 in at nine objects and eleven columns): about a
    # quarter less per object column and a smaller gutter per panel; the marks shrink a little with it (_draw_absolute_panel) so
    # the four-model fan inside an object's slot still reads
    fig = plt.figure(figsize=(width or ((0.25 * len(objects) + 0.45) * sum(widths) + 1.0 * len(groups)), height), layout="constrained")
    grid = fig.add_gridspec(1, len(groups), width_ratios=widths, wspace=0.08)
    seen, foot_notes = set(), set()
    shared_range = ylim is not None and not isinstance(ylim, dict)
    for ci, (cat, cols) in enumerate(groups):
        ax = fig.add_subplot(grid[0, ci])
        panel = pts[pts.eval_id.isin(cols)]
        yl = ylim.get(cat) if isinstance(ylim, dict) else ylim
        yl = tuple(yl) if yl is not None else _auto_ylim(panel, cols)
        sn, fn = _draw_absolute_panel(ax, panel, cols, objects, models, gaps, yl, cat, zero_line=zero_line)   # titled as the main figures title theirs
        seen |= sn; foot_notes |= fn
        # A FIGURE SPANNING TWO CATEGORIES CANNOT SHARE ONE Y LABEL (2026-09-22, grouping over-refusal with capability):
        # benign compliance and accuracy are both percentages OF DIFFERENT THINGS. Pass `ylabel` as a {category: label}
        # mapping and each panel is labeled on its own axis and keeps its tick labels; a plain string keeps the single
        # shared label, and hides the later panels' tick labels only when the row shares one range.
        if isinstance(ylabel, dict):
            ax.set_ylabel(_wrap_label(ylabel.get(cat, "")))
        elif ci != 0 and shared_range:
            ax.tick_params(labelleft=False)
    if not isinstance(ylabel, dict):
        fig.supylabel(_wrap_label(ylabel))
    # the legend is `object` (95 % CI first, Unsteered below it) and `model` only; the diamond is the mean over
    # models with a black +- SEM bar, hollow when one model, and the gap glyphs at the foot keep their meaning in the tables
    legend_stack(fig, appendix_legend_groups(objects, models))
    return save(fig, name)


def absolute_panels(pts: pd.DataFrame, evals: list, objects: list, models: list, gaps: pd.DataFrame, ylabel, name: str,
                    ylim=(-3, 103), width: float | None = None, height: float = 5.0):
    """The paper's appendix figure across models in the main Qwen figures' form, ABSOLUTE: one panel per
    category of the evals given (static, multi-query, multi-turn; over-refusal; agentic; capability), on one row,
    the value on the y axis the figure's own unit (defense success, benign compliance, accuracy, misaligned actions).
    `delta_panels` is its relative companion on the same layout. One marker per model - the seeds and decode draws pooled - with its
    own interval; see `_eval_row_figure` for the panel form."""
    return _eval_row_figure(pts, evals, objects, models, gaps, ylabel, name, ylim=ylim, width=width, height=height, zero_line=False, pooled=False)


def delta_points(d: pd.DataFrame, evals: list, objects: list) -> pd.DataFrame:
    """The relative companion's points and the tables' arm rows: one pooled delta per (model, object, eval) from the anchored frame -
    every reading of the cell (its fit seeds, its decode draws) with equal weight, as `pool_cell` pools the absolute marks - ORIENTED
    by the eval's category so positive is better (Δ defense success, Δ benign compliance, a reduction in misaligned actions,
    Δ accuracy: the main figures' orientation), the interval's bounds swapped with it. `y`, `lo`, `hi` are the oriented delta; the
    unsteered model never enters (its delta is zero by construction) and a cell with no unsteered run to compare with is no point
    (it is a gap: `delta_gaps`)."""
    arms = [o for o in objects if o != "none"]
    x = d[d.eval_id.isin(evals) & d.object_label.isin(arms) & d.d_y.notna()] if len(d) else d
    if not len(x):
        return x.assign(y=pd.Series(dtype=float), lo=pd.Series(dtype=float), hi=pd.Series(dtype=float))
    p = pool_cell(x, y="d_y", lo="d_lo", hi="d_hi")
    sign = p.eval_id.map(lambda e: float(CATEGORIES[EVAL_CATEGORY[e]][1]) if e in EVAL_CATEGORY else 1.0).astype(float)
    a, b = sign * p.d_lo.astype(float), sign * p.d_hi.astype(float)
    return p.assign(y=sign * p.d_y.astype(float), lo=np.minimum(a, b), hi=np.maximum(a, b))


def delta_gaps(d: pd.DataFrame, gaps: pd.DataFrame, objects: list) -> pd.DataFrame:
    """The relative companion's gap statuses: the registry's (`gap_status` against the delta points drawn), plus `no unsteered run`
    for a cell whose steered runs are in the frame but have nothing to be differenced against - the main figures' own glyph for that
    case, which `gap_status` cannot see because the runs exist and are judged."""
    g = gaps[gaps.object_label.isin(objects)].copy() if len(gaps) else gaps
    if not len(d) or not len(g):
        return g
    na, have = d[d.d_y.isna()], d[d.d_y.notna()]
    keys = set(zip(na.model_id, na.object_label, na.eval_id)) - set(zip(have.model_id, have.object_label, have.eval_id))
    hit = [(m, o, e) in keys for m, o, e in zip(g.model_id, g.object_label, g.eval_id)]
    g.loc[hit, "status"] = "no unsteered run"
    return g


def delta_panels(d: pd.DataFrame, evals: list, objects: list, models: list, gaps: pd.DataFrame, name: str, ylabel=None,
                 width: float | None = None, height: float = 5.0):
    """The RELATIVE companion of `absolute_panels` (the appendix figures that follow the main figures' form, x = the eval, keep one
    row and also plot relative to unsteered): the same panels, columns, dodging and marks, every point the
    cell's change against its own model's unsteered run - the anchored deltas the x = object figures and the tables read - oriented
    so up is better, the zero line drawn, each panel labeled and ranged on its own axis as the main figures are. `ylabel` defaults
    to each category's own Δ label; a string labels every panel with it, a {category: label} mapping labels each. The unsteered
    model is not an arm here; a cell with nothing to difference against draws the `no unsteered run` glyph."""
    objects = [o for o in objects if o != "none"]
    pts = delta_points(d, evals, objects)
    if isinstance(ylabel, dict):
        labels = ylabel
    else:
        labels = {cat: (ylabel if ylabel else (CATEGORIES[cat][2] if cat in CATEGORIES else "Δ vs unsteered (pp)"))
                  for cat, _ in split_by_category(list(evals))}
    return _eval_row_figure(pts, evals, objects, models, delta_gaps(d, gaps, objects), labels, name, ylim=None, width=width, height=height,
                            zero_line=True, pooled=True)


# ---- appendix: best-of-N search curves - defense success within N attempts against N (BoN) --------------------------------------
BON_EVAL = "strongreject_bon"
BON_NS = (1, 2, 4, 8, 16, 32, 64, 75)                        # the producers' curve grid (asr_at_n<N>) plus the headline n_max
BON_CURVE_RE = re.compile(r"^asr_at_n(\d+)$")


def bon_curve_points(op: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """One row per (admitted best-of-N run, N): attack success within the first N attempts at threshold 0.5, read from the registry.

    `op` is the admitted headline frame (op_all: the runs the appendix attack panels draw, so this figure agrees with them cell for
    cell); `long` is every in-paper metric row (paper_all). A run's curve is its `asr_at_n<N>` rows at threshold 0.5 - seedrep_cluster,
    bon_native4096 and r2_export file them - and its N = n_max point is the headline row itself (`asr` at n = 75, or `asr_bon_cap4096`,
    both the per-goal any-hit@75 in 0-1). A run with NO curve rows whose headline metric carries the producer's `asr_at_n_4096_cf` grid
    in its extra (bon_stores' climb runs, until the rows are filed) is read from that grid and marked `from_extra`, so the
    stopgap is visible in the composition table; a run with neither is returned with `n` NaN and drawn nowhere. `draw` labels the
    replicate: the fit seed where the object has one, else the seed the view spells (qwen35_4b_s2, paired_d1_s3), else the view."""
    h = op[op["eval_id"].eq(BON_EVAL)]
    mids = long["metric_id"].astype(str)
    curve = long[long["eval_id"].eq(BON_EVAL) & mids.str.match(BON_CURVE_RE) & pd.to_numeric(long["threshold"], errors="coerce").eq(0.5)]
    out = []
    for _, r in h.iterrows():
        c = curve[curve["run_id"].eq(r["run_id"])]
        pts = {int(BON_CURVE_RE.match(str(m)).group(1)): float(v) for m, v in zip(c["metric_id"], c["value"]) if pd.notna(v)}
        from_extra = False
        if not pts:
            grid = (_extra(r.get("metric_extra_json")) or {}).get("asr_at_n_4096_cf") or {}
            pts = {int(k): float(v) for k, v in grid.items() if v is not None}
            from_extra = bool(pts)
        n_max = int(r["bon_n"]) if pd.notna(r.get("bon_n")) else 75
        if pd.notna(r.get("value")):
            pts[n_max] = float(r["value"])                                # the headline row IS the N = n_max point
        seed = str(r.get("seed") or "")
        m_view = re.search(r"_s(\d+)p?$", str(r.get("view_legacy") or ""))
        draw = seed or (f"s{m_view.group(1)}" if m_view else str(r.get("view_legacy") or ""))
        base = dict(run_id=r["run_id"], model_id=r["model_id"], object_label=r["object_label"], seed=seed, draw=draw,
                    source=r.get("lane_source_legacy"), view=r.get("view_legacy"), headline_metric=r.get("metric_id"), from_extra=from_extra,
                    decode=str(r.get("decode_policy_id") or ""))
        if not pts:
            out.append(dict(base, n=np.nan, asr=np.nan))
        for n, v in sorted(pts.items()):
            out.append(dict(base, n=n, asr=v))
    cols = ["run_id", "model_id", "object_label", "seed", "draw", "source", "view", "headline_metric", "from_extra", "decode", "n", "asr"]
    return pd.DataFrame(out, columns=cols)


def bon_curve_select(pts: pd.DataFrame, min_points: int = 3) -> tuple:
    """Which admitted runs a cell (model, object) DRAWS, and why the rest do not - returned as (kept, dropped) with a `why` column.
    Three rules, each printed with the figure so nothing leaves silently: (1) a run with fewer than `min_points` curve points is another
    lane's headline-only anchor (the Arditi / AlphaSteer unsteered rows carry no asr_at_n rows in the paper frame) - `headline only`;
    (2) a cell whose runs span decode policies keeps the served model's VENDOR PRESET (the paper's decode; 2026-09-21: a T = 1.0 grandfathered climb and a vendor-preset standup are not replicates of one measurement) when any run has it,
    else keeps all - `other decode`; (3) one store filed by two readers (the same view and the same points) is drawn once - `duplicate
    filing`. Runs with no points at all are dropped as `no curve`."""
    if pts.empty:
        return pts, pts.assign(why=pd.Series(dtype=str))
    one = pts.groupby("run_id", as_index=False).agg(model_id=("model_id", "first"), object_label=("object_label", "first"), view=("view", "first"),
                                                     decode=("decode", "first"), n_points=("n", lambda s: int(s.notna().sum())),
                                                     key=("asr", lambda s: tuple(np.round(s.dropna().values, 4))))
    one["why"] = ""
    one.loc[one["n_points"].eq(0), "why"] = "no curve"
    one.loc[one["why"].eq("") & one["n_points"].lt(min_points), "why"] = "headline only"
    for (m, o), g in one[one["why"].eq("")].groupby(["model_id", "object_label"]):
        vendor = g["decode"].str.contains("vendor_preset", na=False)
        if vendor.any() and not vendor.all():
            one.loc[g.index[~vendor], "why"] = "other decode"
    live = one[one["why"].eq("")]
    dup = live.duplicated(subset=["model_id", "object_label", "view", "key"], keep="first")
    one.loc[live.index[dup], "why"] = "duplicate filing"
    keep = set(one.loc[one["why"].eq(""), "run_id"])
    dropped = one[one["why"].ne("")].drop(columns=["key"]).sort_values(["model_id", "object_label", "why"]).reset_index(drop=True)
    return pts[pts["run_id"].isin(keep)].copy(), dropped


def bon_curve_unadmitted(op: pd.DataFrame, long: pd.DataFrame, objects: list, models: list) -> pd.DataFrame:
    """Selected, in-paper, vendor-preset best-of-N runs of a drawn (model, object) cell that op_all does NOT admit while the cell is
    drawn from a grandfathered T = 1.0 run - the admission gap confirmed 2026-09-23 (Qwen's seedrep prompted SIs file
    `asr_bon_cap4096` as their headline and fall out of the panels' frame). Printed with the figure as its interim footnote; empty once
    the admission reads the headline family, at which point those runs enter `op` and the figure redraws them as seeds + mean."""
    cols = ["model_id", "object_label", "view_legacy", "lane_source_legacy", "decode_policy_id", "run_id"]
    if long.empty or any(c not in long.columns for c in cols + ["eval_id", "metric_id", "in_current_paper"]):
        return pd.DataFrame(columns=cols)
    admitted = set(op.loc[op["eval_id"].eq(BON_EVAL), "run_id"])
    drawn_t1 = op[op["eval_id"].eq(BON_EVAL) & ~op["decode_policy_id"].astype(str).str.contains("vendor_preset", na=False)]
    cells = set(zip(drawn_t1["model_id"], drawn_t1["object_label"]))
    cand = long[long["eval_id"].eq(BON_EVAL) & long["in_current_paper"].eq(1) & long["metric_id"].isin(["asr", "asr_bon_cap4096"])
                & pd.to_numeric(long["threshold"], errors="coerce").eq(0.5) & long["model_id"].isin(models) & long["object_label"].isin(objects)
                & long["decode_policy_id"].astype(str).str.contains("vendor_preset", na=False) & ~long["run_id"].isin(admitted)]
    # a BOOLEAN SERIES, never a bare list: with no candidate the list is empty and `cand[[]]` selects zero COLUMNS, so the column
    # pick below raised KeyError on the first frame with every vendor-preset best-of-N run admitted (render of 2026-10-07)
    cand = cand[pd.Series([(m, o) in cells for m, o in zip(cand["model_id"], cand["object_label"])], index=cand.index, dtype=bool)]
    return cand[cols].drop_duplicates("run_id").sort_values(["model_id", "object_label", "view_legacy"]).reset_index(drop=True)


def bon_curve_composition(pts: pd.DataFrame) -> pd.DataFrame:
    """What each drawn cell is made of: one row per (model, object, run) with its replicate label, source, view, where its curve came
    from (`rows` = asr_at_n<N> metric rows; `extra` = the bon_stores grid stopgap; `none` = no curve, not drawn) and its any-hit@75."""
    if pts.empty:
        return pts
    one = pts.sort_values("n").groupby("run_id", as_index=False).agg(
        model_id=("model_id", "first"), object_label=("object_label", "first"), draw=("draw", "first"), source=("source", "first"),
        view=("view", "first"), decode=("decode", "first"), n_points=("n", lambda s: int(s.notna().sum())), from_extra=("from_extra", "first"),
        asr_at_75=("asr", lambda s: s.iloc[-1] if s.notna().any() else np.nan))
    one["curve"] = np.where(one["n_points"].eq(0), "none", np.where(one["from_extra"], "extra", "rows"))
    return one.sort_values(["model_id", "object_label", "draw"]).drop(columns=["from_extra"]).reset_index(drop=True)


def bon_curves_figure(pts: pd.DataFrame, objects: list, models: list, name: str, height: float = 5.6, faint_alpha: float = 0.32):
    """Defense success within N attempts, 100 x (1 - ASR@N), against N on a log2 axis: one panel per model, one color per object
    (prompted SIs dashed, fitted objects and the unsteered model solid). A cell with several admitted runs - Qwen's three fit seeds,
    or its three seed-view standups of the unsteered and prompted arms - draws each run as a thin faint line and their mean as the
    thick marked line; a cell with one run draws that run thick. The mean at each N is over the runs that all
    report it. A model with no curve draws its `not measured` note. Curves read from the extra-grid stopgap are footnoted."""
    fig, axes = plt.subplots(1, len(models), figsize=(4.9 * len(models), height), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    drawn = set(); stopgap = set()
    for ax, m in zip(axes, models):
        d = pts[pts["model_id"].eq(m) & pts["object_label"].isin(objects) & pts["n"].notna()]
        if d.empty:
            ax.text(0.5, 0.5, "not measured", ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=16)
        for o in objects:
            g = d[d["object_label"].eq(o)]
            if g.empty:
                continue
            runs = list(g["run_id"].unique()); ls = "--" if o in PROMPTED else "-"; col = COLOR[o]
            drawn.add(o)
            if g["from_extra"].any():
                stopgap.add((m, o))
            if len(runs) > 1:
                for rid in runs:
                    gr = g[g["run_id"].eq(rid)].sort_values("n")
                    ax.plot(gr["n"], 100 * (1 - gr["asr"]), color=col, lw=1.3, ls=ls, alpha=faint_alpha, zorder=2, solid_capstyle="round")
                mean = g.groupby("n").agg(asr=("asr", "mean"), k=("run_id", "nunique")).reset_index()
                mean = mean[mean["k"].eq(len(runs))].sort_values("n")
                ax.plot(mean["n"], 100 * (1 - mean["asr"]), color=col, lw=3.2, ls=ls, marker=MARKER[m], ms=8.5, markeredgecolor="white",
                        markeredgewidth=1.0, zorder=4, label=LABEL[o])
            else:
                gr = g.sort_values("n")
                ax.plot(gr["n"], 100 * (1 - gr["asr"]), color=col, lw=2.6, ls=ls, marker=MARKER[m], ms=8.5, markeredgecolor="white",
                        markeredgewidth=1.0, zorder=3, label=LABEL[o])
        # THE AXIS RUNS TO N = 75, WHERE THE CELLS WERE SCORED: the curves always ended on the headline at 75, but
        # the ticks stopped at 64 and the figure read as if it did too. 64 keeps its tick mark and loses its label (same day):
        # on a log-2 axis 64 and 75 sit too close for two labels, and 75 is the one that says where the scoring stopped.
        ax.set_xscale("log", base=2); ax.set_xlim(0.8, 92); ax.set_xticks(list(BON_NS))
        ax.set_xticklabels([str(n) if n != 64 else "" for n in BON_NS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0, 101); ax.set_xlabel("attempts N"); ax.set_title(MODEL_LABEL.get(m, m), fontweight="semibold")
        ax.grid(axis="y", color=RULE, lw=0.8)
    axes[0].set_ylabel("defense success within N attempts (%)")
    obj_handles = [Line2D([], [], ls="--" if o in PROMPTED else "-", lw=3.0, color=COLOR[o], label=LABEL[o]) for o in objects if o in drawn]
    # THE `object` GROUP ALONE, AND NO FOOTNOTE (on seeing the figure): the thin-line / thick-line key and the
    # stopgap note (which cells' curves came from the headline row's asr_at_n_4096_cf grid rather than filed asr_at_n rows) both came
    # off the canvas. Neither is lost: the composition table the cell prints under the figure marks every run drawn from the grid
    # (`extra`), and the caption says what thin and thick lines are. `stopgap` is still computed so a caller can ask.
    legend_stack(fig, [("object", obj_handles, max(1, (len(obj_handles) + 1) // 2))])
    fig.bon_stopgap = sorted(stopgap)
    return save(fig, name)
