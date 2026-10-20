#!/usr/bin/env python
"""ladder_decide.py — the per-(cell, readout) ESCALATION DECISION of the full22 SAMPLED cap ladder.

    ladder_decide.py static   --private-root R --private-run NAME --attack FAM --defense CELL --cap N
                              [--sibling-private-run NAME_OTHER_TRIAL] [--rep-ratio-source PATH]
                              --out decision.json [--items-out truncated_keys.jsonl]
    ladder_decide.py controls --out-root ROOT --cell CELL --control C --cap N
                              [--sibling-out-root ROOT_OTHER_TRIAL] [--rep-ratio-source PATH]
                              --out decision.json [--items-out truncated_ids.jsonl]
    ladder_decide.py reclassify --kind static   --root PRIVATE_ROOT --run NAME_BASE [--sibling-run NAME_BASE_2]
                                [--attacks a,b] --out-dir DIR
    ladder_decide.py reclassify --kind controls --root SAMPLED_ROOT --tag TAG --trial T [--sibling-trial S]
                                [--root-pattern "controls_cap{cap}/{tag}_t{trial}"] [--controls c1,c2] --out-dir DIR

The rules (LADDER_RULES.md beside this file):
  * ceiling 16,384 — controls 512 -> 2048 -> 8192 -> 16384, static 4096 -> 8192 -> 16384; per item,
    self-limiting per (cell, readout).
  * a (cell, readout) ESCALATES from rung N to N+1 only when its TRUNCATED share exceeds 1% of its
    responses; it leaves the ladder at <= 1% or at the ceiling.
  * ONLY TRUNCATED ITEMS ESCALATE, NEVER LOOPS: every at-cap item is classified ``loop``
    (rep_ratio over the WHOLE text OR over the LAST 200 whitespace tokens > 0.5 — a strict
    superset that catches a late-onset loop the whole-text ratio dilutes — OR
    sisteer.eval.degeneracy.classify(text).degenerate; which rule fired is recorded), ``repetitive``
    (0.2 <= max(whole, tail) <= 0.5 and not degenerate: below the loop rule, above the taxonomy's
    cap-coherent threshold; reported as its own class, never escalated) or ``cap_coherent``
    (max(whole, tail) < 0.2 and not degenerate) — only cap_coherent items may escalate.
  * BOUNDARY draws are excluded from the trigger: a cap_coherent at-cap item whose SIBLING TRIAL's row for
    the same item terminated (finish_reason != "length") with completion_tokens <= 0.5 x cap is
    ``boundary`` noise — counted and reported, never in the trigger share, never escalated on its own.
    When the sibling store is absent or incomplete NO item is called boundary and the record says so.
  * truncated = at-cap AND cap_coherent AND NOT boundary; share_truncated = n_truncated / n_responses.

Every record and every disposition file is stamped ``rule = "item_filtered_v1"`` (loops + boundary
excluded, 1% truncated trigger) and ``classifier = {name, version, source, source_commit, source_sha,
window}`` naming the peer's reference implementation, so an extend pass can REFUSE anything decided
under another rule and a reader sees one classifier covered the run.

Inputs (read-only): a SAMPLED static store (``<R>/<NAME>/<FAM>/<CELL>/shard*/responses.jsonl`` — a base
run or an escalation rung; rows carry response / finish_reason / at_cap / completion_tokens /
sampling_seed) or a SAMPLED served-controls store (``<ROOT>/cells/<store>/`` — text via the spool,
flags via ``private/gens_<name>.meta.jsonl`` bound by text sha, exactly as at_cap_controls reads it).

Outputs: ``--out`` = the decision record (schema in LADDER_RULES.md; provenance: the rep_ratio source
text + its file sha, the degeneracy module sha, every threshold, the sibling store used, this tool's
sha), ``--items-out`` = the AT-CAP DISPOSITION file the drivers' ``--items`` flag reads — a header line
plus ONE row per at-cap item (static: goal_id / sample_idx / trial; controls: control / item_id) with
``class``, ``boundary``, ``escalate`` and ``excluded_as`` (null for the escalation set, else
loop | repetitive | boundary) — so the driver regenerates exactly the truncated set and stamps every
other at-cap row with why it was left at cap. stdout: ONE JSON line (the record without the function
source; the sbatch reads ``decision`` and ``n_truncated``).

``reclassify`` re-runs the classification over EVERY rung already generated for a (run, trial) — a text
statistic over stores we hold, no regeneration — writing decision.json + items.jsonl per (cell, readout)
per rung under ``--out-dir`` and a ``reclassify.json`` summary, so an extend pass can run over
re-classified residuals; an absent rung is listed as absent, a store that refuses is listed with its
reason (and the pass exits 2 so it is never mistaken for complete).

Exit 0 on ANY valid decision (escalate | exit_le_1pct | exit_ceiling). Exit 2 on a REFUSAL, never a
number: a greedy-shaped store (no finish_reason / decode / meta sidecar), a torn or incomplete store,
a missing store, a store whose cap is not --cap, a --cap off the ladder, an absent rep_ratio source.
A sibling that was NAMED but is absent / incomplete / a different item set is NOT a refusal: the
decision is still valid, with no boundary calls and ``sibling.status = "unavailable"`` + the reason.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
for _p in (str(HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import at_cap_controls as acc  # noqa: E402  (sibling: the controls store reader)
import at_cap_static as acs  # noqa: E402  (sibling: the static store reader)

#: the ladders (LADDER_RULES.md): per item, self-limiting, ceiling 16,384
LADDERS = {"static": (4096, 8192, 16384), "controls": (512, 2048, 8192, 16384)}
CEILING = 16384
#: the trigger: escalate iff share_truncated > TRIGGER_SHARE (<= exits)
TRIGGER_SHARE = 0.01
#: the classifier thresholds on the artifact's 4-gram statistic — the LOOP threshold is STRICT
#: (``> 0.5``, the peer's ``cap_ladder.REP_RATIO_THRESHOLD`` and the artifact's flags producer alike),
#: read over max(whole text, last TAIL_WORDS whitespace tokens)
LOOP_REP_RATIO_THRESHOLD = 0.5
REPETITIVE_REP_RATIO_MIN = 0.2
TAIL_WORDS = 200
WINDOW = "whole_or_last_200_words"
#: the boundary parameter: the sibling's completion_tokens <= BOUNDARY_FRACTION x cap
BOUNDARY_FRACTION = 0.5
#: the rule every record / disposition file is stamped with: loops + boundary excluded, 1% truncated
#: trigger, item-filtered escalation set. An extend pass asserts it before acting on a decision.
RULE = "item_filtered_v1"
#: the classifier stamp: the reference implementation of the wide loop window
#: (branch fst-pair; the blob's sha256 pins the text)
CLASSIFIER = {
    "name": "cap_ladder.degeneracy(whole_or_tail rep_ratio) + sisteer.eval.degeneracy.classify",
    "version": RULE,
    "source": "src/sisteer/capeval/tiers/cap_ladder.py",
    "source_commit": "146d11acd56218e4ac05d70ea684cc3c9d302f1b",
    "source_sha": "2a935fb6645fc5660478c888fbcf396d814761028c04a38bd1191d6d3c475c08",
    "window": WINDOW,
    "tail_words": TAIL_WORDS,
    "loop_threshold": LOOP_REP_RATIO_THRESHOLD,
    "loop_operator": ">",
}
#: where the artifact's rep_ratio lives (the flags producer + compare_greedy_sampled extract the same
#: function by AST from the same file; never re-typed)
JOIN_SCORES_DEFAULT = "$STAGE/sdfbase/fullscale/full22_join_scores.py"   # under the staging directory; expanded where it is read
DEGENERACY_DEFAULT = REPO / "src" / "sisteer" / "eval" / "degeneracy.py"
#: the items file: a header line under this key, then one row per at-cap item
ITEMS_HEADER_KEY = "__ladder_items__"
ITEMS_SCHEMA = 2
CLASSES = ("loop", "repetitive", "cap_coherent")
EXCLUSIONS = ("loop", "repetitive", "boundary")
DECISIONS = ("escalate", "exit_le_1pct", "exit_ceiling")
EXIT_REFUSED = 2
LOOP_RULE = (f"max(rep_ratio(text), rep_ratio(last {TAIL_WORDS} whitespace tokens)) > "
             f"{LOOP_REP_RATIO_THRESHOLD} OR degeneracy.classify(text)['degenerate'] (tail_loop / line_loop)")
ARTIFACT_LOOP_RULE = "rep_ratio(text) > 0.5"
LOOP_RULE_NOTE = ("the peer's cap_ladder.degeneracy and the artifact's flags producer both use the STRICT "
                  "operator > 0.5; the tail reading is a strict superset of the artifact's whole-text rule "
                  "(a text shorter than the window reads identically), so everything the artifact calls a "
                  "loop is a loop here; an item whose max reading is EXACTLY 0.5 is repetitive, counted in "
                  "n_rep_ratio_at_tie")
CONTROLS_ROOT_PATTERN = "controls_cap{cap}/{tag}_t{trial}"


class Refusal(RuntimeError):
    """A store this tool will not put a decision on (module docstring)."""


def refuse(msg: str) -> NoReturn:
    print(f"[ladder-decide] REFUSED: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_REFUSED)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(HERE.parent), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 — provenance is best effort, never a failure
        return None


# --------------------------------------------------------------------------- #
# The two classifier sources: rep_ratio by AST, degeneracy by file path        #
# --------------------------------------------------------------------------- #

def extract_rep_ratio(src_path: Path = JOIN_SCORES_DEFAULT) -> Tuple[Callable[[str], float], dict]:
    """The artifact's looping statistic, extracted by AST from full22_join_scores.py so it is
    byte-for-byte the producer's function (degeneracy_flags_glm.py and compare_greedy_sampled.py do
    exactly this; the peer's cap_ladder.rep_ratio is the same text copied). Returns ``(fn,
    provenance)``; an absent source or a file without exactly one top-level ``def rep_ratio`` is a
    refusal."""
    if "$" in str(src_path):   # a site placeholder ($STAGE, the staging directory) expands through the site config
        if str(REPO / "scripts") not in sys.path:
            sys.path.append(str(REPO / "scripts"))
        from site_paths import SitePathError, site_path
        try:
            src_path = site_path(src_path, "--rep-ratio-source")
        except SitePathError as exc:
            raise Refusal(f"rep_ratio source: {exc} (or pass --rep-ratio-source)") from exc
    src_path = Path(src_path)
    if not src_path.is_file():
        raise Refusal(f"rep_ratio source absent: {src_path} (pass --rep-ratio-source; on a cluster the "
                      "grading host file must be mirrored beside the repo)")
    src = src_path.read_text()
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rep_ratio"]
    if len(fns) != 1:
        raise Refusal(f"{src_path}: expected exactly one top-level `def rep_ratio`, found {len(fns)}")
    fn_src = ast.get_source_segment(src, fns[0])
    ns: dict = {}
    exec(compile(fn_src, str(src_path), "exec"), ns)  # noqa: S102 — our own read-only file, pinned by sha
    return ns["rep_ratio"], {"source": str(src_path), "source_sha256": sha256_file(src_path),
                             "rep_ratio_source": fn_src,
                             "rep_ratio_source_sha256": hashlib.sha256(fn_src.encode("utf-8")).hexdigest()}


def load_degeneracy(path: Path = DEGENERACY_DEFAULT) -> Tuple[Callable[[Optional[str]], dict], dict]:
    """``sisteer.eval.degeneracy.classify`` loaded by FILE PATH (the module is pure python; loading
    it this way keeps this tool free of the harness import) + its provenance."""
    path = Path(path)
    if not path.is_file():
        raise Refusal(f"degeneracy module absent: {path}")
    spec = importlib.util.spec_from_file_location("_ladder_degeneracy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.classify, {"source": str(path), "source_sha256": sha256_file(path),
                          "rules": {"tail": {"window": mod.TAIL_WINDOW, "block": mod.TAIL_BLOCK,
                                             "min_repeats": mod.TAIL_MIN_REPEATS, "min_len": mod.TAIL_MIN_LEN},
                                    "line": {"min_lines": mod.LINE_MIN_LINES, "max_share": mod.LINE_MAX_SHARE}}}


# --------------------------------------------------------------------------- #
# The classifier                                                               #
# --------------------------------------------------------------------------- #

def tail_rep_ratio(text: str, rep_ratio: Callable[[str], float], words: int = TAIL_WORDS) -> float:
    """The peer's ``cap_ladder.tail_rep_ratio``: rep_ratio over the last ``words`` whitespace tokens
    (the whole text when it is shorter — the two readings then coincide)."""
    w = (text or "").split()
    return float(rep_ratio(" ".join(w[-int(words):])) if len(w) > int(words) else rep_ratio(text or ""))


def classify_text(text: str, rep_ratio: Callable[[str], float], degeneracy: Callable[[Optional[str]], dict],
                  *, loop_threshold: float = LOOP_REP_RATIO_THRESHOLD,
                  repetitive_min: float = REPETITIVE_REP_RATIO_MIN, tail_words: int = TAIL_WORDS) -> dict:
    """The three-way class of ONE at-cap text. ``rep_ratio_max`` = max(whole-text reading, last
    ``tail_words`` reading); ``loop`` when it is STRICTLY above ``loop_threshold`` OR the degeneracy
    detector fires (``loop_by``: "rep_ratio" | "degeneracy" | "both"; ``loop_window``: which reading
    crossed — "whole" | "tail" | "both" | null); else ``repetitive`` when rep_ratio_max >=
    repetitive_min; else ``cap_coherent``."""
    rr = float(rep_ratio(text or ""))
    tr = tail_rep_ratio(text or "", rep_ratio, tail_words)
    mx = max(rr, tr)
    deg = degeneracy(text)
    by_whole, by_tail = rr > loop_threshold, tr > loop_threshold
    by_rr = by_whole or by_tail
    by_deg = bool(deg["degenerate"])
    if by_rr or by_deg:
        klass = "loop"
        loop_by = "both" if (by_rr and by_deg) else ("rep_ratio" if by_rr else "degeneracy")
    elif mx >= repetitive_min:
        klass, loop_by = "repetitive", None
    else:
        klass, loop_by = "cap_coherent", None
    window = None
    if by_rr:
        window = "both" if (by_whole and by_tail) else ("whole" if by_whole else "tail")
    return {"class": klass, "rep_ratio": round(rr, 6), "tail_rep_ratio": round(tr, 6),
            "rep_ratio_max": round(mx, 6), "loop_by": loop_by, "loop_window": window,
            "degenerate": by_deg, "degenerate_reason": deg["degenerate_reason"],
            "rep_ratio_at_tie": mx == loop_threshold}


def boundary_call(sibling_row: Optional[dict], *, cap: int,
                  fraction: float = BOUNDARY_FRACTION) -> dict:
    """The boundary rule against the SIBLING trial's row for the same item at this rung:
    boundary iff the sibling terminated (finish_reason != "length") in <= fraction x cap tokens.
    ``sibling_row`` is ``{finish_reason, completion_tokens, source_cap}`` or None (no sibling)."""
    if sibling_row is None:
        return {"boundary": False, "sibling": None}
    fr = sibling_row.get("finish_reason")
    ct = sibling_row.get("completion_tokens")
    readable = isinstance(ct, int) and not isinstance(ct, bool)
    is_b = (fr is not None) and fr != "length" and readable and ct <= fraction * cap
    return {"boundary": bool(is_b),
            "sibling": {"finish_reason": fr, "completion_tokens": ct if readable else None,
                        "source_cap": sibling_row.get("source_cap"),
                        "threshold_tokens": fraction * cap}}


def decide(counts: dict, *, cap: int, ladder: Tuple[int, ...], trigger: float = TRIGGER_SHARE) -> dict:
    """``{decision, next_cap, at_ceiling}`` from the counts: escalate iff share_truncated > trigger and
    a rung above exists; at the ceiling a share above the trigger is exit_ceiling (a residual that
    would otherwise escalate), a share at or below it exit_le_1pct."""
    at_ceiling = cap == ladder[-1]
    idx = ladder.index(cap)
    next_cap = None if at_ceiling else ladder[idx + 1]
    share = counts["share_truncated"]
    if share > trigger:
        decision = "exit_ceiling" if at_ceiling else "escalate"
    else:
        decision = "exit_le_1pct"
    return {"decision": decision, "next_cap": next_cap if decision == "escalate" else None,
            "at_ceiling": at_ceiling}


# --------------------------------------------------------------------------- #
# Store readers (read-only; refusals per the module docstring)                  #
# --------------------------------------------------------------------------- #

def read_static_store(cell_root: Path) -> dict:
    """Every row of a SAMPLED static store (base or rung), completeness-checked (rows vs progress per
    shard and per goal; unique (goal_id, sample_idx)), plus the store's cap (a rung's escalation.json
    cap_to, else the one decode.cap every row carries). Returns ``{rows: {(goal_id, sample_idx): row},
    order: [keys], n_shards, cap, trial}``; a greedy or torn store raises :class:`Refusal`."""
    cell_root = Path(cell_root)
    shards = acs.shard_dirs(cell_root)
    if not shards:
        raise Refusal(f"no shard* dirs under {cell_root} — there is no store to decide on")
    rows: Dict[tuple, dict] = {}
    order: List[tuple] = []
    caps: set = set()
    trials: set = set()
    esc_cap = None
    for shard in shards:
        try:
            recs = acs.read_rows(shard / "progress.jsonl")
            srows = acs.read_rows(shard / "responses.jsonl")
        except acs.AtCapRefusal as exc:
            raise Refusal(str(exc)) from exc
        esc_p = shard / "escalation.json"
        if esc_p.exists():
            try:
                esc_cap = int(json.loads(esc_p.read_text())["cap_to"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise Refusal(f"{esc_p}: unreadable escalation record ({exc})") from exc
        want: Dict[Any, int] = {}
        for rec in recs:
            gid = rec.get("goal_id")
            if gid in want:
                raise Refusal(f"{shard}: goal {gid!r} has more than one terminal progress record")
            want[gid] = int(rec.get("n_variants", 0))
        got: Dict[Any, int] = {}
        for row in srows:
            where = str(shard / "responses.jsonl")
            try:
                at_cap = acs.row_at_cap(row, where=where)
            except acs.AtCapRefusal as exc:
                raise Refusal(str(exc)) from exc
            gid = row.get("goal_id")
            if gid not in want:
                raise Refusal(f"{where}: row (goal {gid!r}) has no terminal progress record — the shard "
                              "did not finish this goal; not a settled store")
            key = (gid, row.get("sample_idx"))
            if key in rows:
                raise Refusal(f"{where}: item {key!r} appears twice — an ambiguous store")
            if not isinstance(row.get("response"), str):
                raise Refusal(f"{where}: row {key!r} carries no response text")
            row = dict(row)
            row["_at_cap"] = at_cap
            rows[key] = row
            order.append(key)
            got[gid] = got.get(gid, 0) + 1
            dec = row.get("decode") or {}
            if "cap" in dec:
                caps.add(int(dec["cap"]))
            trials.add(row.get("trial"))
        for gid, n in want.items():
            if got.get(gid, 0) != n:
                raise Refusal(f"{shard}: goal {gid!r} has {got.get(gid, 0)} rows but progress says "
                              f"n_variants={n} — a short shard; not a settled store")
    if not rows:
        raise Refusal(f"{len(shards)} shard dir(s) under {cell_root} hold ZERO rows")
    if len(trials) != 1:
        raise Refusal(f"{cell_root}: rows carry {len(trials)} distinct trial values {sorted(map(str, trials))} "
                      "— one store is one trial")
    if esc_cap is not None:
        cap = esc_cap
    else:
        if len(caps) != 1:
            raise Refusal(f"{cell_root}: a base store must carry ONE decode.cap, found {sorted(caps)}")
        cap = caps.pop()
    return {"rows": rows, "order": order, "n_shards": len(shards), "cap": int(cap),
            "trial": next(iter(trials)), "is_rung": esc_cap is not None}


def read_controls_store(out_root: Path, cell: str, control: str) -> dict:
    """ONE control of a SAMPLED served-controls store: at_cap_controls' own validated reading
    (gen_stats vs meta sidecar agreement, every spooled text bound) plus the texts and the bound meta
    rows. Returns ``{cell_dir, cap, n, texts, meta}``; refusals are at_cap_controls'."""
    try:
        cell_dir = acc.locate_store(Path(out_root), cell)
        rec = acc.read_control(cell_dir, control)
        gs = json.loads((cell_dir / "gen_stats.json").read_text())[control]
        spool_name = Path(str(gs.get("spool") or "")).name
        spool_path = cell_dir / "private" / spool_name
        texts = acc._spool_texts(spool_path)
        meta = acc._meta_bound(spool_path.with_name(spool_name[:-len(".jsonl")] + acc.META_SUFFIX), texts)
    except acc.Refusal as exc:
        raise Refusal(str(exc)) from exc
    return {"cell_dir": cell_dir, "cap": int(rec["cap"]), "n": int(rec["n"]), "texts": texts, "meta": meta}


# --------------------------------------------------------------------------- #
# The decision over one (cell, readout)                                        #
# --------------------------------------------------------------------------- #

def _sibling_static(root: Path, sibling_run: Optional[str], attack: str, defense: str,
                    keys: List[tuple], cap: int) -> Tuple[Optional[Dict[tuple, dict]], dict]:
    if not sibling_run:
        return None, {"status": "not_given"}
    cell_root = Path(root) / sibling_run / attack / defense
    rec = {"status": "used", "private_run": sibling_run, "root": str(cell_root)}
    try:
        store = read_static_store(cell_root)
    except Refusal as exc:
        return None, {"status": "unavailable", "private_run": sibling_run, "root": str(cell_root),
                      "reason": f"sibling store absent or incomplete: {exc}"}
    if set(store["rows"]) != set(keys):
        return None, {**rec, "status": "unavailable",
                      "reason": f"sibling item set differs ({len(store['rows'])} vs {len(keys)} items)"}
    if store["cap"] != cap:
        return None, {**rec, "status": "unavailable",
                      "reason": f"sibling store is at cap {store['cap']}, this store at {cap}"}
    rec["trial"] = store["trial"]
    rows = {k: {"finish_reason": r.get("finish_reason"), "completion_tokens": r.get("completion_tokens"),
                "source_cap": r.get("source_cap", (r.get("decode") or {}).get("cap"))}
            for k, r in store["rows"].items()}
    return rows, rec


def _sibling_controls(sibling_root: Optional[str], cell: str, control: str, ids: List[str],
                      cap: int) -> Tuple[Optional[Dict[str, dict]], dict]:
    if not sibling_root:
        return None, {"status": "not_given"}
    rec = {"status": "used", "out_root": str(sibling_root)}
    try:
        store = read_controls_store(Path(sibling_root), cell, control)
    except Refusal as exc:
        return None, {**rec, "status": "unavailable", "reason": f"sibling store absent or incomplete: {exc}"}
    rec["store"] = str(store["cell_dir"])
    if set(store["meta"]) != set(ids):
        return None, {**rec, "status": "unavailable",
                      "reason": f"sibling item set differs ({len(store['meta'])} vs {len(ids)} items)"}
    if store["cap"] != cap:
        return None, {**rec, "status": "unavailable",
                      "reason": f"sibling store is at cap {store['cap']}, this store at {cap}"}
    # closer-fix-3: the record names the sibling GENERATION it paired boundary calls against — its item count, cap and content digest
    # (the same at_cap_controls.store_digest the closer recomputes) — so a sibling regenerated after the gate ruled is refused by name, not re-read
    rec.update({"n_items": len(store["meta"]), "cap": int(store["cap"]), "store_digest": acc.store_digest(store["texts"])})
    rows = {i: {"finish_reason": m.get("finish_reason"), "completion_tokens": m.get("completion_tokens"),
                "source_cap": m.get("source_cap")} for i, m in store["meta"].items()}
    return rows, rec


def classify_store(items: List[Tuple[Any, str, bool]], sibling_rows: Optional[dict], *, cap: int,
                   rep_ratio, degeneracy, thresholds: dict) -> Tuple[List[dict], dict]:
    """Classify every AT-CAP item of ``items`` ([(key, text, at_cap)]) and tally the partition
    n_at_cap = n_loop + n_repetitive + n_boundary + n_truncated. ``sibling_rows`` None => no boundary
    calls. Returns ``(per_item, counts)``; per_item rows carry the key under ``"key"``."""
    per_item: List[dict] = []
    n_resp = len(items)
    c = {"n_responses": n_resp, "n_at_cap": 0, "n_loop": 0,
         "n_loop_by_rule": {"rep_ratio": 0, "degeneracy": 0, "both": 0},
         "n_loop_by_window": {"whole": 0, "tail": 0, "both": 0}, "n_rep_ratio_at_tie": 0,
         "n_repetitive": 0, "n_boundary": 0, "n_boundary_any": 0, "n_sibling_unreadable": 0,
         "n_truncated": 0}
    for key, text, at_cap in items:
        if not at_cap:
            continue
        c["n_at_cap"] += 1
        k = classify_text(text, rep_ratio, degeneracy, loop_threshold=thresholds["loop_rep_ratio_threshold"],
                          repetitive_min=thresholds["repetitive_rep_ratio_min"],
                          tail_words=thresholds["tail_words"])
        sib = None if sibling_rows is None else sibling_rows.get(key)
        b = boundary_call(sib, cap=cap, fraction=thresholds["boundary_fraction"])
        if sib is not None and b["sibling"] is not None and b["sibling"]["completion_tokens"] is None:
            c["n_sibling_unreadable"] += 1
        c["n_boundary_any"] += int(b["boundary"])
        c["n_rep_ratio_at_tie"] += int(k["rep_ratio_at_tie"])
        if k["class"] == "loop":
            c["n_loop"] += 1
            c["n_loop_by_rule"][k["loop_by"]] += 1
            if k["loop_window"] is not None:
                c["n_loop_by_window"][k["loop_window"]] += 1
            excluded_as = "loop"
        elif k["class"] == "repetitive":
            c["n_repetitive"] += 1
            excluded_as = "repetitive"
        elif b["boundary"]:
            c["n_boundary"] += 1
            excluded_as = "boundary"
        else:
            c["n_truncated"] += 1
            excluded_as = None
        per_item.append({"key": key, **k, "boundary": b["boundary"], "sibling": b["sibling"],
                         "escalate": excluded_as is None, "excluded_as": excluded_as})
    c["share_truncated"] = (c["n_truncated"] / n_resp) if n_resp else 0.0
    c["share_at_cap"] = (c["n_at_cap"] / n_resp) if n_resp else 0.0
    return per_item, c


def residual_of(counts: dict) -> Optional[dict]:
    if counts["n_at_cap"] == 0:
        return None
    return {k: counts[k] for k in ("n_at_cap", "n_loop", "n_loop_by_rule", "n_loop_by_window",
                                   "n_repetitive", "n_boundary", "n_truncated", "share_truncated",
                                   "share_at_cap")}


def _thresholds(args) -> dict:
    return {"trigger_share": float(args.trigger_share),
            "loop_rep_ratio_threshold": float(args.loop_rep_ratio_threshold),
            "repetitive_rep_ratio_min": float(args.repetitive_rep_ratio_min),
            "tail_words": int(args.tail_words),
            "boundary_fraction": float(args.boundary_fraction)}


def _items_header(kind: str, cap: int, identity: dict, per_item: List[dict], verdict: dict,
                  rep_info: dict) -> dict:
    return {"schema": ITEMS_SCHEMA, "kind": kind, "cap": cap, **identity,
            "n_at_cap": len(per_item), "n_escalate": sum(1 for it in per_item if it["escalate"]),
            "decision": verdict["decision"], "rule": RULE, "classifier": dict(CLASSIFIER),
            "tool_sha256": sha256_file(HERE), "rep_ratio_source_sha256": rep_info["source_sha256"]}


def _write_items(path: Path, header: dict, per_item: List[dict], key_fields: Callable[[Any], dict]) -> str:
    """The at-cap disposition file: header line + one row per at-cap item. Returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({ITEMS_HEADER_KEY: header}, sort_keys=True) + "\n")
        for it in per_item:
            row = {**key_fields(it["key"]), "class": it["class"], "rep_ratio": it["rep_ratio"],
                   "tail_rep_ratio": it["tail_rep_ratio"], "rep_ratio_max": it["rep_ratio_max"],
                   "loop_by": it["loop_by"], "loop_window": it["loop_window"],
                   "degenerate_reason": it["degenerate_reason"],
                   "boundary": it["boundary"], "sibling": it["sibling"],
                   "escalate": it["escalate"], "excluded_as": it["excluded_as"]}
            f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return sha256_file(path)


def read_items_file(path: Path) -> Tuple[dict, List[dict]]:
    """``(header, rows)`` of a disposition file; a torn line, a missing header, a header not stamped
    with THIS rule, or a row outside the schema raises :class:`Refusal` (the drivers wrap it in
    their own refusal) — so an extend pass can never act on a decision made under another rule."""
    path = Path(path)
    if not path.is_file():
        raise Refusal(f"items file absent: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed: List[dict] = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise Refusal(f"{path}:{i} is not a JSON row ({exc}) — a torn items file") from exc
    if not parsed or ITEMS_HEADER_KEY not in parsed[0]:
        raise Refusal(f"{path}: first line is not a {ITEMS_HEADER_KEY} header")
    header = parsed[0][ITEMS_HEADER_KEY]
    if header.get("schema") != ITEMS_SCHEMA:
        raise Refusal(f"{path}: items schema {header.get('schema')!r} != {ITEMS_SCHEMA}")
    if header.get("rule") != RULE:
        raise Refusal(f"{path}: decided under rule {header.get('rule')!r}, not {RULE!r} — an extend pass "
                      "acts only on item-filtered decisions (loops + boundary excluded, 1% truncated trigger)")
    rows = parsed[1:]
    for r in rows:
        if r.get("class") not in CLASSES or not isinstance(r.get("escalate"), bool):
            raise Refusal(f"{path}: row {r!r} is not a disposition row (class / escalate)")
        if r["escalate"] != (r.get("excluded_as") is None) or (
                r.get("excluded_as") is not None and r["excluded_as"] not in EXCLUSIONS):
            raise Refusal(f"{path}: row {r!r}: escalate must be true iff excluded_as is null "
                          f"(excluded_as in {EXCLUSIONS})")
    return header, rows


def _record(kind: str, identity: dict, cap: int, ladder: Tuple[int, ...], thresholds: dict,
            rep_info: dict, deg_info: dict, sibling: dict, counts: dict, verdict: dict,
            items_out: Optional[Path], items_sha: Optional[str], n_escalate: int, argv) -> dict:
    return {
        "tool": {"name": HERE.name, "path": str(HERE), "sha256": sha256_file(HERE), "git_sha": git_sha(),
                 "argv": list(argv), "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat()},
        "rule": RULE,
        "classifier": dict(CLASSIFIER),
        "kind": kind,
        "identity": identity,
        "cap": int(cap), "ladder": list(ladder), "ceiling": ladder[-1], "next_cap": verdict["next_cap"],
        "at_ceiling": verdict["at_ceiling"],
        "rules": {**thresholds, "loop_rule": LOOP_RULE, "artifact_loop_rule": ARTIFACT_LOOP_RULE,
                  "loop_rule_note": LOOP_RULE_NOTE,
                  "repetitive_rule": f"{thresholds['repetitive_rep_ratio_min']} <= max(rep_ratio, "
                                     f"tail_rep_ratio) <= {thresholds['loop_rep_ratio_threshold']} and not "
                                     "degenerate",
                  "cap_coherent_rule": f"max(rep_ratio, tail_rep_ratio) < "
                                       f"{thresholds['repetitive_rep_ratio_min']} and not degenerate",
                  "boundary_rule": "sibling trial's row for the same item: finish_reason != 'length' and "
                                   f"completion_tokens <= {thresholds['boundary_fraction']} x cap "
                                   "(only cap_coherent items are excluded as boundary; n_boundary_any counts "
                                   "the rule over every at-cap item)",
                  "trigger_rule": f"escalate iff n_truncated / n_responses > {thresholds['trigger_share']} "
                                  "and a rung above exists; truncated = at_cap and cap_coherent and not boundary",
                  "escalation_set": "exactly the truncated items (never a loop, a repetitive or a boundary item)"},
        "rep_ratio": rep_info,
        "degeneracy": deg_info,
        "sibling": sibling,
        "counts": counts,
        "decision": verdict["decision"],
        "n_truncated": counts["n_truncated"],
        "share_truncated": counts["share_truncated"],
        "residual": None if verdict["decision"] == "escalate" else residual_of(counts),
        "items_out": None if items_out is None else str(items_out),
        "items_out_sha256": items_sha,
        "n_items_escalate": int(n_escalate),
    }


def _emit(record: dict, out: Path, *, stdout: bool = True) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, out)
    if stdout:
        line = json.loads(json.dumps(record, default=str))
        line["rep_ratio"] = {k: v for k, v in line["rep_ratio"].items() if k != "rep_ratio_source"}
        print(json.dumps(line, sort_keys=True), flush=True)


def _check_cap(cap: int, kind: str) -> Tuple[int, Tuple[int, ...]]:
    ladder = LADDERS[kind]
    cap = int(cap)
    if cap not in ladder:
        raise Refusal(f"--cap {cap} is not a rung of the {kind} ladder {list(ladder)}")
    return cap, ladder


def decide_static(*, private_root: Path, private_run: str, attack: str, defense: str, cap: int,
                  sibling_run: Optional[str], thresholds: dict, rep_ratio, rep_info: dict, degeneracy,
                  deg_info: dict, out: Path, items_out: Optional[Path], argv, stdout: bool = True) -> dict:
    """ONE static (attack, defense) at ONE rung: read, classify, decide, write. Raises
    :class:`Refusal`; returns the record."""
    cap, ladder = _check_cap(cap, "static")
    cell_root = Path(private_root) / private_run / attack / defense
    store = read_static_store(cell_root)
    if store["cap"] != cap:
        raise Refusal(f"{cell_root}: the store is at cap {store['cap']}, not --cap {cap}")
    keys = store["order"]
    sib_rows, sibling = _sibling_static(Path(private_root), sibling_run, attack, defense, keys, cap)
    items = [(k, store["rows"][k]["response"], store["rows"][k]["_at_cap"]) for k in keys]
    per_item, counts = classify_store(items, sib_rows, cap=cap, rep_ratio=rep_ratio, degeneracy=degeneracy,
                                      thresholds=thresholds)
    verdict = decide(counts, cap=cap, ladder=ladder, trigger=thresholds["trigger_share"])
    trial = store["trial"]
    identity = {"private_root": str(private_root), "private_run": private_run, "attack": attack,
                "defense": defense, "trial": trial, "store": str(cell_root), "n_shards": store["n_shards"],
                "is_rung": store["is_rung"]}
    items_sha = None
    if items_out is not None:
        header = _items_header("static", cap, {"private_run": private_run, "attack": attack,
                                               "defense": defense, "trial": trial}, per_item, verdict, rep_info)
        items_sha = _write_items(Path(items_out), header, per_item,
                                 lambda key: {"goal_id": key[0], "sample_idx": key[1], "trial": trial})
    record = _record("static", identity, cap, ladder, thresholds, rep_info, deg_info, sibling, counts,
                     verdict, items_out, items_sha, sum(1 for it in per_item if it["escalate"]), argv)
    _emit(record, Path(out), stdout=stdout)
    return record


def decide_controls(*, out_root: Path, cell: str, control: str, cap: int, sibling_root: Optional[str],
                    thresholds: dict, rep_ratio, rep_info: dict, degeneracy, deg_info: dict, out: Path,
                    items_out: Optional[Path], argv, stdout: bool = True) -> dict:
    """ONE (cell, control) at ONE rung root: read, classify, decide, write. Raises :class:`Refusal`;
    returns the record."""
    cap, ladder = _check_cap(cap, "controls")
    store = read_controls_store(Path(out_root), cell, control)
    if store["cap"] != cap:
        raise Refusal(f"{store['cell_dir']}: {control} is at cap {store['cap']}, not --cap {cap}")
    ids = sorted(store["texts"])
    sib_rows, sibling = _sibling_controls(sibling_root, cell, control, ids, cap)
    items = [(i, store["texts"][i], bool(store["meta"][i]["at_cap"])) for i in ids]
    per_item, counts = classify_store(items, sib_rows, cap=cap, rep_ratio=rep_ratio, degeneracy=degeneracy,
                                      thresholds=thresholds)
    verdict = decide(counts, cap=cap, ladder=ladder, trigger=thresholds["trigger_share"])
    identity = {"out_root": str(out_root), "store": str(store["cell_dir"]), "cell": cell, "control": control,
                "n_items": int(store["n"]), "store_digest": acc.store_digest(store["texts"])}     # closer-fix-2: the record is bound to THIS generation
    items_sha = None
    if items_out is not None:
        header = _items_header("controls", cap, {"out_root": str(out_root), "store": str(store["cell_dir"]),
                                                 "cell": cell, "control": control}, per_item, verdict, rep_info)
        items_sha = _write_items(Path(items_out), header, per_item,
                                 lambda key: {"control": control, "item_id": key})
    record = _record("controls", identity, cap, ladder, thresholds, rep_info, deg_info, sibling, counts,
                     verdict, items_out, items_sha, sum(1 for it in per_item if it["escalate"]), argv)
    _emit(record, Path(out), stdout=stdout)
    return record


def _load_classifier(args) -> tuple:
    rep_ratio, rep_info = extract_rep_ratio(Path(args.rep_ratio_source))
    degeneracy, deg_info = load_degeneracy(Path(args.degeneracy_source))
    return rep_ratio, rep_info, degeneracy, deg_info


def cmd_static(args, argv) -> int:
    try:
        rep_ratio, rep_info, degeneracy, deg_info = _load_classifier(args)
        decide_static(private_root=Path(args.private_root), private_run=args.private_run, attack=args.attack,
                      defense=args.defense, cap=args.cap, sibling_run=args.sibling_private_run,
                      thresholds=_thresholds(args), rep_ratio=rep_ratio, rep_info=rep_info,
                      degeneracy=degeneracy, deg_info=deg_info, out=Path(args.out),
                      items_out=Path(args.items_out) if args.items_out else None, argv=argv)
    except Refusal as exc:
        refuse(str(exc))
    return 0


def cmd_controls(args, argv) -> int:
    try:
        rep_ratio, rep_info, degeneracy, deg_info = _load_classifier(args)
        decide_controls(out_root=Path(args.out_root), cell=args.cell, control=args.control, cap=args.cap,
                        sibling_root=args.sibling_out_root, thresholds=_thresholds(args), rep_ratio=rep_ratio,
                        rep_info=rep_info, degeneracy=degeneracy, deg_info=deg_info, out=Path(args.out),
                        items_out=Path(args.items_out) if args.items_out else None, argv=argv)
    except Refusal as exc:
        refuse(str(exc))
    return 0


# --------------------------------------------------------------------------- #
# reclassify: every rung already generated, no regeneration                    #
# --------------------------------------------------------------------------- #

def static_run_name(run_base: str, cap: int, ladder: Tuple[int, ...]) -> str:
    """The private_run of rung ``cap`` for a base run (the config generator's naming:
    ``<base>`` at the base cap, ``<base>_cap<N>`` above it)."""
    return run_base if cap == ladder[0] else f"{run_base}_cap{cap}"


def _entry(kind: str, cap: int, **kw) -> dict:
    return {"kind": kind, "cap": cap, **kw}


def _record_entry(ent: dict, rec: dict) -> None:
    ent.update(status="decided", decision=rec["decision"], n_truncated=rec["n_truncated"],
               share_truncated=rec["share_truncated"], n_at_cap=rec["counts"]["n_at_cap"],
               n_responses=rec["counts"]["n_responses"], next_cap=rec["next_cap"],
               sibling_status=rec["sibling"]["status"])


def _reclassify_static(args, root: Path, out_dir: Path, ladder, thresholds, clf, argv,
                       entries: List[dict], present: List[int], absent: List[int]) -> int:
    rep_ratio, rep_info, degeneracy, deg_info = clf
    attacks = [a for a in (args.attacks or "").split(",") if a] or None
    n_refused = 0
    for cap in ladder:
        run = static_run_name(args.run, cap, ladder)
        sib = static_run_name(args.sibling_run, cap, ladder) if args.sibling_run else None
        run_dir = root / run
        if not run_dir.is_dir():
            absent.append(cap)
            continue
        present.append(cap)
        for attack_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if attacks and attack_dir.name not in attacks:
                continue
            for cell_dir in sorted(p for p in attack_dir.iterdir() if p.is_dir()):
                dest = out_dir / "static" / run / attack_dir.name / cell_dir.name
                ent = _entry("static", cap, private_run=run, attack=attack_dir.name, defense=cell_dir.name,
                             sibling_private_run=sib, decision_path=str(dest / "decision.json"),
                             items_path=str(dest / "items.jsonl"))
                try:
                    rec = decide_static(private_root=root, private_run=run, attack=attack_dir.name,
                                        defense=cell_dir.name, cap=cap, sibling_run=sib, thresholds=thresholds,
                                        rep_ratio=rep_ratio, rep_info=rep_info, degeneracy=degeneracy,
                                        deg_info=deg_info, out=dest / "decision.json",
                                        items_out=dest / "items.jsonl", argv=argv, stdout=False)
                    _record_entry(ent, rec)
                except Refusal as exc:
                    n_refused += 1
                    ent.update(status="refused", reason=str(exc))
                entries.append(ent)
    return n_refused


def _reclassify_controls(args, root: Path, out_dir: Path, ladder, thresholds, clf, argv,
                         entries: List[dict], present: List[int], absent: List[int]) -> int:
    rep_ratio, rep_info, degeneracy, deg_info = clf
    controls_filter = [c for c in (args.controls or "").split(",") if c] or None
    n_refused = 0
    for cap in ladder:
        rung_root = root / args.root_pattern.format(cap=cap, tag=args.tag, trial=args.trial)
        sib_root = (root / args.root_pattern.format(cap=cap, tag=args.tag, trial=args.sibling_trial)
                    if args.sibling_trial is not None else None)
        if not rung_root.is_dir():
            absent.append(cap)
            continue
        present.append(cap)
        cells_dir = rung_root / "cells"
        if not cells_dir.is_dir():
            n_refused += 1
            entries.append(_entry("controls", cap, out_root=str(rung_root), status="refused",
                                  reason=f"{rung_root}: no cells/ dir — not a served controls root"))
            continue
        for d in sorted(cells_dir.iterdir()):
            mp = d / "manifest.json"
            if not mp.is_file():
                continue
            try:
                cell = json.loads(mp.read_text()).get("cell_id")
            except json.JSONDecodeError as exc:
                n_refused += 1
                entries.append(_entry("controls", cap, out_root=str(rung_root), store=str(d), status="refused",
                                      reason=f"{mp}: unreadable manifest ({exc})"))
                continue
            try:
                controls = sorted(acc.count(rung_root, cell, None)["controls"])
            except acc.Refusal as exc:
                n_refused += 1
                entries.append(_entry("controls", cap, out_root=str(rung_root), store=str(d), cell=cell,
                                      status="refused", reason=str(exc)))
                continue
            for control in controls:
                if controls_filter and control not in controls_filter:
                    continue
                dest = out_dir / "controls" / f"cap{cap}" / str(cell) / control
                ent = _entry("controls", cap, out_root=str(rung_root), cell=cell, control=control,
                             sibling_out_root=None if sib_root is None else str(sib_root),
                             decision_path=str(dest / "decision.json"), items_path=str(dest / "items.jsonl"))
                try:
                    rec = decide_controls(out_root=rung_root, cell=cell, control=control, cap=cap,
                                          sibling_root=None if sib_root is None else str(sib_root),
                                          thresholds=thresholds, rep_ratio=rep_ratio, rep_info=rep_info,
                                          degeneracy=degeneracy, deg_info=deg_info,
                                          out=dest / "decision.json", items_out=dest / "items.jsonl",
                                          argv=argv, stdout=False)
                    _record_entry(ent, rec)
                except Refusal as exc:
                    n_refused += 1
                    ent.update(status="refused", reason=str(exc))
                entries.append(ent)
    return n_refused


def cmd_reclassify(args, argv) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        clf = _load_classifier(args)
    except Refusal as exc:
        refuse(str(exc))
    thresholds = _thresholds(args)
    ladder = LADDERS[args.kind]
    entries: List[dict] = []
    present: List[int] = []
    absent: List[int] = []
    root = Path(args.root)
    if args.kind == "static":
        if not args.run:
            refuse("reclassify --kind static needs --run <base private_run>")
        n_refused = _reclassify_static(args, root, out_dir, ladder, thresholds, clf, argv, entries, present, absent)
    else:
        if not args.tag or args.trial is None:
            refuse("reclassify --kind controls needs --tag and --trial")
        n_refused = _reclassify_controls(args, root, out_dir, ladder, thresholds, clf, argv, entries, present,
                                         absent)
    rep_info, deg_info = clf[1], clf[3]
    summary = {
        "tool": {"name": HERE.name, "path": str(HERE), "sha256": sha256_file(HERE), "git_sha": git_sha(),
                 "argv": list(argv), "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat()},
        "rule": RULE, "classifier": dict(CLASSIFIER), "kind": args.kind, "ladder": list(ladder),
        "root": str(root), "rules": {**thresholds},
        "rep_ratio": {k: v for k, v in rep_info.items() if k != "rep_ratio_source"},
        "degeneracy": deg_info,
        "rungs_present": present, "rungs_absent": absent,
        "n_entries": len(entries), "n_decided": sum(1 for e in entries if e["status"] == "decided"),
        "n_refused": n_refused,
        "n_by_decision": {d: sum(1 for e in entries if e.get("decision") == d) for d in DECISIONS},
        "entries": entries,
        "complete": n_refused == 0,
    }
    out = out_dir / "reclassify.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, out)
    print(json.dumps({k: v for k, v in summary.items() if k != "entries"}, sort_keys=True, default=str),
          flush=True)
    if n_refused:
        print(f"[ladder-decide] reclassify: {n_refused} store(s) REFUSED — the pass is not complete; see "
              f"{out}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def _classifier_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--rep-ratio-source", default=str(JOIN_SCORES_DEFAULT),
                    help="full22_join_scores.py — rep_ratio is extracted from it by AST (refused when absent)")
    ap.add_argument("--degeneracy-source", default=str(DEGENERACY_DEFAULT),
                    help="sisteer/eval/degeneracy.py (loaded by path; provenance records its sha)")
    ap.add_argument("--trigger-share", type=float, default=TRIGGER_SHARE)
    ap.add_argument("--loop-rep-ratio-threshold", type=float, default=LOOP_REP_RATIO_THRESHOLD,
                    help="loop iff max(whole, tail) reading is STRICTLY above this (the peer's rule)")
    ap.add_argument("--repetitive-rep-ratio-min", type=float, default=REPETITIVE_REP_RATIO_MIN)
    ap.add_argument("--tail-words", type=int, default=TAIL_WORDS,
                    help="the tail window (whitespace tokens) the loop rule is also read over")
    ap.add_argument("--boundary-fraction", type=float, default=BOUNDARY_FRACTION)


def _single_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--cap", type=int, required=True, help="the rung this store was drawn at")
    ap.add_argument("--out", required=True, help="the decision record (JSON)")
    ap.add_argument("--items-out", default=None,
                    help="the at-cap disposition file the drivers' --items reads (header + one row per at-cap item)")
    _classifier_args(ap)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("static", help="a sampled static-family store (base run or rung)")
    s.add_argument("--private-root", required=True)
    s.add_argument("--private-run", required=True)
    s.add_argument("--attack", required=True)
    s.add_argument("--defense", required=True)
    s.add_argument("--sibling-private-run", default=None,
                   help="the OTHER trial's private_run at the same rung (boundary calls need it)")
    _single_args(s)
    c = sub.add_parser("controls", help="a sampled served-controls store (base root or rung root)")
    c.add_argument("--out-root", required=True)
    c.add_argument("--cell", required=True)
    c.add_argument("--control", required=True)
    c.add_argument("--sibling-out-root", default=None,
                   help="the OTHER trial's out-root at the same rung (boundary calls need it)")
    _single_args(c)
    r = sub.add_parser("reclassify", help="re-classify EVERY rung already generated (no regeneration)")
    r.add_argument("--kind", required=True, choices=("static", "controls"),
                   help="static: --root is the private root, --run the base private_run (rungs are "
                        "<run>, <run>_cap8192, <run>_cap16384); controls: --root is the sampled root, rung "
                        "roots are --root-pattern with {cap}/{tag}/{trial}")
    r.add_argument("--root", required=True)
    r.add_argument("--run", default=None, help="static: the base private_run, e.g. sdfbase_full22_<key>_sampled_t0")
    r.add_argument("--sibling-run", default=None, help="static: the other trial's base private_run")
    r.add_argument("--attacks", default=None, help="static: comma list of attack dirs to include (default all)")
    r.add_argument("--tag", default=None, help="controls: the TAG in the rung root pattern")
    r.add_argument("--trial", type=int, default=None, help="controls: this trial")
    r.add_argument("--sibling-trial", type=int, default=None, help="controls: the other trial")
    r.add_argument("--root-pattern", default=CONTROLS_ROOT_PATTERN,
                   help=f"controls: rung root relative to --root (default {CONTROLS_ROOT_PATTERN!r}; a "
                        "gsm8k readout root uses its own pattern)")
    r.add_argument("--controls", default=None, help="controls: comma list of controls to include (default all)")
    r.add_argument("--out-dir", required=True,
                   help="decision.json + items.jsonl per (cell, readout) per rung, plus reclassify.json")
    _classifier_args(r)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.mode == "static":
        return cmd_static(args, argv)
    if args.mode == "controls":
        return cmd_controls(args, argv)
    return cmd_reclassify(args, argv)


if __name__ == "__main__":
    sys.exit(main())
