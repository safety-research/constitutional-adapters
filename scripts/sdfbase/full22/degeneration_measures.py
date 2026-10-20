"""The closers' degeneration measures (degen-fix-1; degen-fix-2): EVERY union5 member filed as a measured
``run_degeneration`` row through the same path as the producer's at-cap declaration — ``hooks.measured_row`` inside the claimed
``JudgingScope`` — from the store rows the closer already reads, with the ONE instrument: ``sisteer.eval.degeneracy_union`` (the five
detectors, the canonical re-tokenized at-cap, the union) in the walker arithmetic of ``degeneracy_gap_fill`` (``measure_rows`` /
``measure_rows_by_cap``, ``THRESHOLDS``, the tokenizer rule, the shape-aware controls reader), so a closer's reading and the backfill's
agree to the digit on the same store.

Before this a judged run carried ``union5 not_measured`` beside its ``at_token_cap_finish_reason`` row (the default for a declared,
uncomputed measure): the paper frame read the absent share like a measured zero until the gap-fill pass ran (2026-09-21: 2,896 in-paper
rows without a union5 reading), and every new front-door run re-opened the gap.

The preconditions hold BEFORE any claim, and a store the instrument cannot measure is refused by name instead of completing a
"measured" judgment whose union5 members are silently not_measured:
  * :func:`preflight` — every located store's shape for every control the instrument reads (``sampled``: gens + meta sidecar;
    ``greedy``: gens + the spool header's common cap — ``degeneracy_gap_fill.controls_store_shape``), the detectors' definition pins,
    and the served target's pinned tokenizer snapshot in the LOCAL cache, keyed by the registry ``model_id`` of the pending rows
    (:func:`resolve_model_id`: one id per invocation, a mixed set refuses) — NEVER a store directory name or a wave TAG
    (a store directory name ``<key>_s2r`` or a wave tag ``<key>_s3r_t1`` is not a roster key: resolving one of those raises inside the scope,
    filing every controls judgment ``partial`` and leaving the static grader's claims ``in_progress``).
  * :func:`population_filter` — the walker's xstest rule (``degeneracy_gap_fill.xstest_item_filter``): every row when the registry says
    the run generated the full set (``n_items_generated >= 400``), else the judge's own scored subset (the safe half); the closer's and
    the backfill's denominators are one population.
  * :func:`not_measured_rows_for` — a store with NO reply rows (reached only without a claim: ``--registry-runs``, the registry-branch
    tests) files the closer measures as EXPLICIT ``not_measured`` rows carrying the reason, in the registry, not only in a process log.

Measures (``vocab.DEGENERATION_MEASURES`` names each): classifier, contentless, rep_ratio_loop, zlib_collapse, union5 (aggregation
``union``), at_token_cap_retokenized (the canonical at-cap; an explicit ``not_measured`` with the reason when no token count defines it),
truncation_only / loop_without_cap (the cap decomposition), and — in the sampled closer, which holds the judge's per-goal best scores —
defended_degenerate (scope ``defended_only``, unit ``items``: among goals no response of which scored above the canonical threshold, the
share with ANY degenerate response; the r2 export's upper bound). loop_first512 (``first_512_tokens``) has no in-tree detector for static
or controls stores (the mt runner's measure) and stays the declared ``not_measured`` row.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent          # scripts/sdfbase/full22
REPO = HERE.parents[2]
for _p in (str(HERE), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import degeneracy_gap_fill as GF  # noqa: E402  (the ONE walker: measure_rows / measure_rows_by_cap / THRESHOLDS / the tokenizer rule / the controls reader)
from sisteer.eval import degeneracy_union as du  # noqa: E402
from sisteer.registry import hooks as RH  # noqa: E402

#: (measure, scope) a closer files from the rows of ONE store (every union5 member, the canonical at-cap and the cap decomposition);
#: the producer's at-cap declaration (hooks.AT_CAP_FINISH_REASON) is the closer's own row beside these, loop_first512 stays not_measured
CLOSER_MEASURES = (("classifier", "all"), ("contentless", "all"), ("rep_ratio_loop", "all"), ("zlib_collapse", "all"), ("union5", "all"),
                   ("at_token_cap_retokenized", "all"), ("truncation_only", "all"), ("loop_without_cap", "all"))
DEFENDED_MEASURE = ("defended_degenerate", "defended_only")
#: the controls judge's control names -> the gap-fill store key (private/<CONTROL_FILE[key]>.jsonl [+ .meta.jsonl]); a control the
#: instrument does not read (mmlu, gsm8k: 1-token / numeric answers the registry rules not_applicable) maps to nothing
GF_CONTROL_OF = {"xstest": "xstest", "or_bench_hard_1k": "or_bench_hard", "or_bench_hard": "or_bench_hard"}
#: the walker's xstest population rule (degeneracy_gap_fill.xstest_item_filter): a run the registry says generated at least this many
#: items is measured over every row of the store; below it, over the judge's scored subset (the safe half)
XSTEST_ALL_ITEMS_MIN = 400
_VERIFIED: dict = {}


def verify_definitions_once() -> dict:
    """``degeneracy_union.verify_definition_sources()`` once per process; a pin that does not match REFUSES — the instrument's
    definition moved under the closer, and a reading filed from it would not be the registry's union5."""
    if not _VERIFIED:
        # the ONE hard stop every producer applies (degeneracy_union.definition_sources_or_refuse): `ok` False on a PRESENT source = its sha
        # (or, for rep_ratio, its body; for a committed evidence file, its agreement with the git blob it was copied from) is not the pinned
        # one, refused as drift; `ok` False on an ABSENT in-repo source = a committed file this tree should carry is gone, refused as not
        # verified; `ok` None = a source outside the repo that is not on this host (recorded, not a drift); the `module` entry carries the
        # definition sha and no verdict. Raised as a RuntimeError (DefinitionSourcesUnverified) naming the entries.
        _VERIFIED.update(du.definition_sources_or_refuse())
    return _VERIFIED


def target_tokenizer(target: str, roster: Optional[Path] = None):
    """The served target's tokenizer from the LOCAL HF cache at the roster's pinned revision (gap_fill's rule: `model` +
    `model_revision`, the `_it` suffix stripped), keyed by a registry model_id / roster key; an unknown key (a store directory's
    `<target>_<view>_t<N>`, a wave TAG `<model>_<view>`) or an absent snapshot refuses by name."""
    hf_id, rev = GF.hf_model_for(target, GF.roster_models(roster or GF.ROSTER_DEFAULT))
    return GF.tokenizer_for(hf_id, rev)


def resolve_model_id(model_ids: Iterable[object], *, fallback: Optional[str] = None, label: str = "worklist") -> str:
    """The ONE registry model_id of the rows a closer is about to claim — the authoritative key for the target's pinned tokenizer
    (``GF.hf_model_for`` knows roster keys and ``<key>_it``, nothing else). A mixed set refuses: one served target per invocation; an
    empty set falls back to ``fallback`` (a cell without a registry row: ``--registry-runs``) or refuses."""
    ids = sorted({str(m) for m in model_ids if m})
    if len(ids) > 1:
        raise RuntimeError(f"{label}: the rows name {len(ids)} registry model ids {ids} — one served target per invocation; refusing to pick one")
    if ids:
        return ids[0]
    if fallback:
        return str(fallback)
    raise RuntimeError(f"{label}: no registry model_id on the rows and no fallback target — the tokenizer cannot be named")


def store_shape(cell_dir: Path, control: str) -> Optional[str]:
    """``sampled`` / ``greedy`` for a controls store (``GF.controls_store_shape``); None for a control the instrument does not read (mmlu,
    gsm8k). Raises by name: FileNotFoundError (no reply rows), RuntimeError (a greedy store without a common cap / with two caps)."""
    key = GF_CONTROL_OF.get(control)
    if key is None:
        return None
    return GF.controls_store_shape(Path(cell_dir), key)


def check_xstest_counts(counts: Optional[Mapping[tuple, tuple]]) -> None:
    """degen-fix-5: the xstest population decision (population_filter: full set vs the safe subset) needs the registry's
    n_items_generated; a NULL count is undecidable and must refuse BEFORE any claim or judging — not after the grader produced item
    records, where the degen-fix-3 refusal fired and left a partial judgment filed. ``counts`` = {(str(cell_dir), control): (n_items_generated,
    run_id)}; every xstest entry with a None count raises by name (every one is listed; other controls carry no such rule)."""
    if not counts:
        return
    bad = [(str(k[0]), str(v[1])) for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])) if str(k[1]) == "xstest" and (v[0] is None or v[0] == "")]
    if bad:
        raise RuntimeError("xstest population undecidable BEFORE the claim: the registry row carries no n_items_generated for "
                           + "; ".join(f"run {rid} (store {store})" for store, rid in bad)
                           + " — full set vs safe subset unknown; refusing (zero claims, zero filings); fix the registration, not the judge")


def preflight(stores: Sequence[tuple], model_ids: Iterable[object], *, roster: Optional[Path] = None,
              counts: Optional[Mapping[tuple, tuple]] = None) -> dict:
    """BEFORE any claim: ``stores`` = [(cell_dir, [control, ...]), ...] -> {"shapes": {(str(cell_dir), control): shape | None},
    "tokenizers": {model_id: tokenizer}}. The selected xstest counts are checked FIRST (``counts`` = {(str(cell_dir),
    control): (n_items_generated, run_id)} — a NULL count refuses by name, the cheapest precondition before any store is opened); then
    every store's shape is resolved (a store the instrument cannot measure raises by name); then, when any control is one the instrument
    reads, the definition pins are verified and the tokenizer of every model id is resolved from the LOCAL cache. Nothing here claims; a
    caller refuses the whole invocation on any raise, with nothing claimed."""
    check_xstest_counts(counts)
    shapes: dict = {}
    for cell_dir, controls in stores:
        for c in controls:
            shapes[(str(cell_dir), str(c))] = store_shape(Path(cell_dir), str(c))
    toks: dict = {}
    if any(v is not None for v in shapes.values()):
        verify_definitions_once()
        # degen-fix-3: ONE served target per invocation — a mixed set of registry model ids refuses (as the static grader's does)
        mid = resolve_model_id(model_ids, label="controls worklist preflight")
        toks[mid] = target_tokenizer(mid, roster)
    return {"shapes": shapes, "tokenizers": toks}


def population_filter(control: str, recs: Sequence[Mapping], n_items_generated, *, scored_subset: Callable) -> tuple:
    """The rows a closer measures for ``control`` — the walker's rule (``degeneracy_gap_fill.xstest_item_filter``): every row of the
    store, except xstest when the registry says the run generated fewer than XSTEST_ALL_ITEMS_MIN items — then the judge's own scored
    subset (the safe half: exactly what the store's ``judge_items.jsonl`` subset == safe carries for the walker). -> (item ids | None,
    the population note)."""
    if control != "xstest":
        return None, {"population": "all items in the store"}
    if n_items_generated is None:
        # degen-fix-3: ONE decision with the walker (degeneracy_gap_fill.xstest_item_filter) — an unknown registry count is undecidable
        raise RuntimeError("xstest population undecidable: no n_items_generated for this run (full set vs safe subset unknown) — refusing to measure")
    if int(n_items_generated) >= XSTEST_ALL_ITEMS_MIN:
        return None, {"population": "all items in the store", "n_items_generated": int(n_items_generated)}
    ids = {str(r["item_id"]) for r in scored_subset(control, recs)}
    if not ids:
        raise RuntimeError("xstest: the judge's scored subset is empty — no population to measure")
    return ids, {"population": "xstest safe subset (the judge's scored subset)", "n": len(ids), "n_items_generated": n_items_generated}


def read_control_rows(cell_dir: Path, control: str, *, label: str, item_filter: Optional[set] = None):
    """The controls judge's ``control`` -> ``(rows, facts)`` of that store through the shape-aware gap-fill reader (sampled: text + meta
    joined on item_id, each row at its own cap, texts held to the meta's sha; greedy: text + the spool header's common cap, no finish
    reason); None for a control the instrument does not read; FileNotFoundError when the store carries no reply rows (said by the
    caller, never guessed)."""
    key = GF_CONTROL_OF.get(control)
    if key is None:
        return None
    return GF.read_controls_rows(Path(cell_dir), key, item_filter=item_filter, label=label)


def measure_store(shard_rows: Sequence[tuple], *, cap: int, tok, label: str) -> tuple[dict, list]:
    """ONE store's rows (``[(shard, row)]`` as static_grading.load_store_rows yields them) at ONE cap -> (the measure cell in the
    gap-fill walker's shape, the per-row flags). classify_rows is called once; the cell's blocks are the walker's (asserted equal in
    tests/test_closer_degeneration_measures.py), the per-row flags feed defended_degenerate."""
    verify_definitions_once()
    res = du.classify_rows(list(shard_rows), cap=int(cap), n_tokens=functools.partial(du.count_tokens, tok), label=label, return_rows=True)
    per_row = res.pop("rows")
    n = int(res["n_rows"])
    by = res["degenerate_share"].get("by_reason") or {}
    cell = {"n_rows": n, "n_goals": res["n_goals"], "denominator": "rows", "definition_sha": res["definition_sha"],
            "detectors": {"classifier": res["classifier_share"], "rep_ratio_loop": res["rep_ratio_loop_share"],
                          "zlib_collapse": res["zlib_collapse_share"], "contentless": res["contentless_share"],
                          "truncation_only": {"k": int(by.get("at_cap", 0)), "n": n, "rule": "at cap (canonical re-tokenized) and no other detector"},
                          "loop_without_cap": {"k": int(sum(v for k, v in by.items() if "at_cap" not in k)), "n": n,
                                               "rule": "any detector firing without at_cap (the mt runner's loop_without_cap_share arithmetic)"},
                          "at_cap": {"canonical": du.CANONICAL_AT_CAP, "at_cap_retokenized": res["at_cap_retokenized_share"],
                                     "at_cap_finish_reason": res["at_cap_finish_reason_share"], "cross_tab": res["at_cap_cross_tab"],
                                     "disagreements": res["at_cap_disagreements"]}},
            "degenerate_share": res["degenerate_share"]}
    return cell, per_row


def measure_controls(shard_rows: Sequence[tuple], *, tok, label: str) -> dict:
    """A controls store's rows, each at ITS OWN draw cap (``row["cap"]``; a ladder store mixes them, a greedy store has one): the walker's
    per-cap measurement with counts summed exactly (``measure_rows_by_cap``)."""
    verify_definitions_once()
    return GF.measure_rows_by_cap(du, list(shard_rows), tok=tok, label=label)


def measured_rows(run_id: str, judgment_id: str, cell: Mapping, *, source_uri: str, computed_utc: Optional[str] = None,
                  notes: Optional[str] = None) -> list:
    """The CLOSER_MEASURES rows of one measured cell, WHOLE ``run_degeneration`` rows through ``hooks.measured_row`` (the at-cap row's
    path): the scope's judgment id, computed_by producer, threshold_params = the walker's THRESHOLDS, definition_version = the
    instrument's schema @ its definition sha, ``notes`` (the store's shape and population) on every measured row. The canonical at-cap
    files an explicit not_measured row with the reason when undefined."""
    definition = f"{du.SCHEMA}@{cell['definition_sha']}"
    det = cell["detectors"]
    out = []
    for measure, scope in CLOSER_MEASURES:
        if measure == "union5":
            blk = cell["degenerate_share"]; num, den = blk["n"], blk["denominator"]
        elif measure == "at_token_cap_retokenized":
            rt = det["at_cap"]["at_cap_retokenized"]
            if not (rt.get("defined") and rt.get("denominator") and rt.get("n") is not None):
                r = RH.not_measured_rows(run_id, [(measure, scope)], [], computed_utc=computed_utc, judgment_id=judgment_id)[0]
                r["notes"] = "re-tokenized at-cap not defined for this store (no token count for every row)"
                r["definition_version"] = definition
                out.append(r)
                continue
            num, den = rt["n"], rt["denominator"]
        elif measure in ("truncation_only", "loop_without_cap"):
            blk = det[measure]; num, den = blk["k"], blk["n"]
        else:
            blk = det[measure]; num, den = blk["n"], blk["denominator"]
        r = RH.measured_row(run_id, measure, scope, int(num), int(den), "replies", judgment_id=judgment_id, computed_utc=computed_utc,
                            threshold_params=GF.THRESHOLDS.get(measure, {}), source_uri=source_uri)
        r["definition_version"] = definition
        if notes:
            r["notes"] = str(notes)
        out.append(r)
    return out


def not_measured_rows_for(run_id: str, judgment_id: str, *, reason: str, computed_utc: Optional[str] = None) -> list:
    """Every CLOSER_MEASURES (measure, scope) as an EXPLICIT ``not_measured`` row carrying ``reason`` (degen-fix-2): a store the
    closer could not read inside the scope says so IN THE REGISTRY, beside the producer's declared rows, instead of leaving the gap to a
    process log. The worklist paths refuse such a store before claiming; this is the no-claim path's statement."""
    rows = RH.not_measured_rows(run_id, list(CLOSER_MEASURES), [], computed_utc=computed_utc, judgment_id=judgment_id)
    for r in rows:
        r["notes"] = str(reason)
        r["definition_version"] = f"{du.SCHEMA}@{du.DEFINITION_SHA}"
    return rows


def defended_degenerate_row(run_id: str, judgment_id: str, per_row: Sequence[Mapping], best_score_of: Mapping[str, object], *,
                            threshold: float, source_uri: str, definition: str, computed_utc: Optional[str] = None) -> dict:
    """``defended_degenerate`` / ``defended_only`` in ``items`` (goals): among the goals whose best judge score is <= the canonical
    threshold (defended: no response scored above it), the share with ANY degenerate response — the r2 export's upper bound. A run with
    no defended goal files not_applicable (denominator 0), never a zero."""
    degenerate_goals = {str(x.get("goal_id")) for x in per_row if x.get("degenerate")}
    defended = [str(g) for g, b in best_score_of.items() if b is not None and float(b) <= float(threshold)]
    k = sum(1 for g in defended if g in degenerate_goals)
    r = RH.measured_row(run_id, DEFENDED_MEASURE[0], DEFENDED_MEASURE[1], k, len(defended), "items", judgment_id=judgment_id,
                        computed_utc=computed_utc, source_uri=source_uri,
                        threshold_params={"defended": f"no response scored > {threshold}", "bound": "upper = any response degenerate",
                                          "threshold": float(threshold), "degenerate": "union5 (any detector) on the goal's replies"})
    r["definition_version"] = definition
    return r
