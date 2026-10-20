"""The per-leg seed-spread table beside the paper's Qwen deltas.

For every leg × arm (the six cells: constitution,
control, difference × LoRA, ReLU) the value per seed and its SD / range over s1–s3;
the same for the two contrasts per arm type, Δ(constitution − control) and
Δ(difference − none), per seed; beside them the paper's existing Qwen deltas, the
cross-view replicate floor, and the claim list: a paper delta "exceeds the seed spread"
iff ``|Δ_paper| > max(range_seed, 2·SD_seed)`` (both thresholds printed).

EVERY statistic passes ``triplet_gate.require_triplet`` first (rows counted per seed
before any reduction; a duplicate refuses; an incomplete triplet is written as
``unavailable: incomplete replicate triplet (...)``, FAILED for that leg /
arm, with no SD or range over the survivors).

Readers consume the UPSTREAM PRODUCERS' schemas, one declared source per leg (never a
fallback chain — a missing store is a ``no store`` row, loud in the table):

* static sampled families + over-refusal: the sampled-lane ``final_rung.json``
  (``targets.<tag>.static["<family>|<cell>"].pooled.asr["0.5"].p``;
  ``controls["<readout>|<cell>"].pooled.full_refusal.p``) — the final-rung
  decision the source ``paper/paperlib/or_sampled.py`` keys on
* MMLU: ``controls_agg.json`` (``views.<view>.<cell>.gen_stats.mmlu.acc``)
* GSM8K paired-200: the paired-run ``report.json`` (``cells["<model>/<view>/<cell>"].readings
  .merged.n_correct / n_scored``)
* best-of-N + multi-turn: the native grader's ``transfer_stats*.json``
  (``models.<view>.cells.<cell>.<family>.p`` — BoN ``p`` is at n = 75, multi-turn at
  StrongREJECT 0.5)
* Petri: the per-view ``petri_stats`` file (``tags.<tag>.cells.<cell>.behaviors.<b>.mean``
  and ``mean_of_behavior_means``)
* agentic misalignment: the misalignment ``summary.csv`` (``family == agentic``,
  ``metric == harmful``).

The paper's deltas come from the export parquet (``view == qwen35_4b``, dose 1.0, same
eval / metric key) where the export carries the same instrument (BoN, multi-turn, Petri,
MMLU), and otherwise from the SAME producer schema the lane uses (static / over-refusal:
the paper's ``final_rung.json`` under tag ``qwen35_4b``; GSM8K: the published paired-200
report under ``qwen35_4b/main``; misalignment: the ``qwen35_4b`` panel) — every row
names its ``paper_source``. The export's filed ``delta`` and interval ride beside,
verbatim, because its units differ by leg (pp / fraction / score points); the claim test
uses Δ recomputed from the same source's values in the leg's declared unit.

Δ(difference − none) uses the REUSED anchor (the paper's ``none`` reading); a leg whose
anchor preflight (``<grades>/anchor_preflight/<view>_<leg>.json``) is not ``comparable``
contributes its arm SPREAD but no anchor delta, and is listed.

Usage::

    seed_spread.py --grades RS/reseed --export $REVIEWS/sdf-base-steering-r2/data/sdf_base_steering_r2_results.parquet \\
                   --out $GRADE/seedrep/spread [--paper-final-rung F] [--paper-gsm8k-report R]
                   [--paper-misalign-root D] [--figs DIR] [--mirror] [--wandb]
"""
from __future__ import annotations

import argparse
import csv
import glob
from contextlib import contextmanager
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import figures as F  # noqa: E402
from object_space import CLASSES, ObjectClass  # noqa: E402
from triplet_gate import (  # noqa: E402
    REASON_NO_STORE, REASON_NON_COMPARABLE, REASON_NOT_ELIGIBLE, SEEDS, DuplicateReplicate, Record, TripletRefused,
    TripletUnavailable, UnexpectedSeed, normalize, require_triplet, spread,
)

MODEL = "qwen35_4b"
PAPER_VIEW = "qwen35_4b"
NONE = "none"
STATUS_OK = ("graded", "measured")
#: the full22 wave's run-state tree, under the state root ($STATE): a site placeholder, expanded through sisteer.site in main()
RS_FULL22 = Path("$STATE/romp/candidate-si-refit/run-state/full22")
DEFAULT_PAPER_FINAL_RUNG = RS_FULL22 / "sampled/report/analysis/final_rung.json"
DEFAULT_PAPER_GSM8K = Path("$GRADE/tinycap/gsm8k/final_1pct_r2/report.json")   # $GRADE paths: main() expands them
DEFAULT_PAPER_MISALIGN = Path.home() / "figs/si-steering-misalign"
DEFAULT_EXPORT = Path.home() / "reviews/sdf-base-steering-r2/data/sdf_base_steering_r2_results.parquet"
DEFAULT_OUT = Path("$GRADE/seedrep/spread")


def view_of(seed: int) -> str:
    return f"{MODEL}_s{int(seed)}"


def tag_of(seed: int) -> str:
    return view_of(seed)


# ---------------------------------------------------------------- leg roster
@dataclass(frozen=True)
class LegSpec:
    eval_id: str
    label: str
    category: str
    reader: str
    units: str                    # "pp" (rates, ×100) | "score" (Petri 1..10)
    preflight_leg: str
    paper_source: str             # export | final_rung | gsm8k_report | misalign_csv
    export: dict[str, Any] = field(default_factory=dict)     # the export key (also the floor's key)
    args: dict[str, Any] = field(default_factory=dict)
    note: str = ""


PETRI_BEHAVIOURS = ("cyber_bio_chem_capabilities", "defer_to_users", "delusion_sycophancy", "instruction_evasion",
                    "reasoning_unfaithfulness", "self_preservation")
CATEGORY_ORDER = ("single-turn + BoN", "multi-turn", "over-refusal", "Petri", "misalignment", "capability")

LEGS: tuple[LegSpec, ...] = tuple(
    [LegSpec(f, {"pasttense": "past tense", "codeattack": "CodeAttack", "renellm": "ReNeLLM", "grok420": "grok420"}[f],
             "single-turn + BoN", "final_rung_static", "pp", "static", "final_rung",
             export={"eval_id": f, "metric": "asr", "threshold": 0.5},
             args={"family": f},
             note="lane: sampled vendor preset, final rung (2 trials); export row beside is the greedy 4,096 campaign reading")
     for f in ("pasttense", "codeattack", "renellm", "grok420")]
    + [LegSpec("bon", "best-of-75 (4,096)", "single-turn + BoN", "transfer_stats", "pp", "bon", "export",
               export={"eval_id": "bon", "metric": "asr_bon_cap4096", "threshold": 0.5, "bon_n": 75, "cap": 4096},
               args={"family": "bon"})]
    + [LegSpec(f, {"crescendo": "Crescendo", "pair": "PAIR", "actorattack": "ActorAttack"}[f], "multi-turn",
               "transfer_stats", "pp", "multiturn", "export",
               export={"eval_id": f, "metric": "asr", "threshold": 0.5, "cap": 4096, "decode_contains": "sampled"},
               args={"family": f})
       for f in ("crescendo", "pair", "actorattack")]
    + [LegSpec(r, {"xstest": "XSTest (safe)", "or_bench_hard_1k": "OR-Bench-Hard"}[r], "over-refusal",
               "final_rung_controls", "pp", "overrefusal", "final_rung",
               export={"eval_id": r, "metric": "over_refusal_rate"}, args={"readout": r},
               note="full-refusal share at the final rung (the paper's governing over-refusal reading)")
       for r in ("xstest", "or_bench_hard_1k")]
    + [LegSpec("petri_pooled", "Petri (6 behaviours pooled)", "Petri", "petri_stats", "score", "petri", "export",
               export={"eval_id": "petri_pooled", "metric": "petri_score"}, args={"behaviour": None})]
    + [LegSpec(b, b.replace("_", " "), "Petri", "petri_stats", "score", "petri", "export",
               export={"eval_id": b, "metric": "petri_score"}, args={"behaviour": b}) for b in PETRI_BEHAVIOURS]
    + [LegSpec(f"agentic_{b}", f"agentic misalignment ({b})", "misalignment", "misalign_csv", "pp", "misalign",
               "misalign_csv", export={"eval_id": f"agentic_{b}", "metric": "misaligned_rate"}, args={"behaviour": b})
       for b in ("leaking", "murder")]
    + [LegSpec("mmlu", "MMLU (letter prefill)", "capability", "controls_agg_mmlu", "pp", "mmlu", "export",
               export={"eval_id": "mmlu", "metric": "acc"}),
       LegSpec("gsm8k", "GSM8K paired-200 (thinking on)", "capability", "gsm8k_report", "pp", "gsm8k", "gsm8k_report",
               export={"eval_id": "gsm8k_controls512", "metric": "acc"},
               note="export row beside is the controls battery (0-shot greedy 512), a different instrument")]
)
LEG_BY_ID = {l.eval_id: l for l in LEGS}
ROLE_BY_CELL = {c.cell_id: c for c in CLASSES}
CELLS_BY_ARM: dict[str, dict[str, ObjectClass]] = {
    arm: {c.role: c for c in CLASSES if c.arm == arm} for arm in ("lora", "psr")}
CONTRASTS = ("const_minus_ctl", "diff_minus_none")


def to_units(value: float | None, units: str) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return 100.0 * float(value) if units == "pp" else float(value)


# ---------------------------------------------------------------- readings
@dataclass
class Reading:
    value: float | None           # in the leg's units already
    eligible: bool
    reason: str | None = None
    source: str | None = None
    k: int | None = None
    n: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_READ_CACHE: dict[tuple[str, str], Any] | None = None     # None = every read hits the disk (the default)


@contextmanager
def cached_reads():
    """Memoise :func:`_load_json` and the store globs for the duration of ONE analysis run (the battery reads the same
    `final_rung.json` / `transfer_stats*.json` / `petri_stats*/*.json` once per (leg, arm, seed) — 891 parses of 8 files
    per run). The cache dies with the block, so a file rewritten between two runs is re-read."""
    global _READ_CACHE
    prev, _READ_CACHE = _READ_CACHE, {}
    try:
        yield
    finally:
        _READ_CACHE = prev


def _load_json(path: Path) -> Any:
    if _READ_CACHE is None:
        return json.loads(Path(path).read_text())
    key = ("json", str(path))
    if key not in _READ_CACHE:
        _READ_CACHE[key] = json.loads(Path(path).read_text())
    return _READ_CACHE[key]


def _glob(pattern: str, *, recursive: bool = False) -> list[str]:
    if _READ_CACHE is None:
        return glob.glob(pattern, recursive=recursive)
    key = ("glob", pattern)
    if key not in _READ_CACHE:
        _READ_CACHE[key] = glob.glob(pattern, recursive=recursive)
    return list(_READ_CACHE[key])


def _first(paths: Sequence[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def _no_store(what: str) -> Reading:
    return Reading(None, False, f"{REASON_NO_STORE} ({what})")


def read_final_rung_static(fr: Path | None, tag: str, family: str, cid: str, units: str, thr: str = "0.5") -> Reading:
    if fr is None or not fr.is_file():
        return _no_store("final_rung.json")
    d = _load_json(fr)
    t = (d.get("targets") or {}).get(tag)
    if t is None:
        return _no_store(f"tag {tag} in {fr.name}")
    e = (t.get("static") or {}).get(f"{family}|{cid}")
    if e is None:
        return _no_store(f"{family}|{cid} under targets.{tag}.static")
    if e.get("held"):
        return Reading(None, False, f"held: {e.get('status')}", source=str(fr))
    if e.get("status") not in ("ok", None):
        return Reading(None, False, f"status {e.get('status')!r}", source=str(fr))
    a = ((e.get("pooled") or {}).get("asr") or {}).get(thr)
    if not a or "p" not in a:
        return Reading(None, False, f"no pooled.asr[{thr}].p", source=str(fr))
    return Reading(to_units(a["p"], units), True, None, f"{fr} ›targets.{tag}.static[{family}|{cid}].pooled.asr[{thr}]",
                   k=a.get("k"), n=a.get("n"), extra={"n_goals": a.get("n_goals"), "final_caps": (e.get("pooled") or {}).get("final_caps")})


def read_final_rung_controls(fr: Path | None, tag: str, readout: str, cid: str, units: str) -> Reading:
    if fr is None or not fr.is_file():
        return _no_store("final_rung.json")
    d = _load_json(fr)
    t = (d.get("targets") or {}).get(tag)
    if t is None:
        return _no_store(f"tag {tag} in {fr.name}")
    e = (t.get("controls") or {}).get(f"{readout}|{cid}")
    if e is None:
        return _no_store(f"{readout}|{cid} under targets.{tag}.controls")
    if e.get("held"):
        return Reading(None, False, f"held: {e.get('status')}", source=str(fr))
    if e.get("status") not in ("ok", None):
        return Reading(None, False, f"status {e.get('status')!r}", source=str(fr))
    a = (e.get("pooled") or {}).get("full_refusal")
    if not a or "p" not in a:
        return Reading(None, False, "no pooled.full_refusal.p", source=str(fr))
    return Reading(to_units(a["p"], units), True, None, f"{fr} ›targets.{tag}.controls[{readout}|{cid}].pooled.full_refusal",
                   k=a.get("k"), n=a.get("n"), extra={"n_items": a.get("n_items"), "final_caps": (e.get("pooled") or {}).get("final_caps")})


def read_controls_agg_mmlu(root: Path, view: str, cid: str, units: str) -> Reading:
    p = _first([root / "controls_agg.json", root / "controls" / "controls_agg.json"])
    if p is None:
        return _no_store("controls_agg.json")
    d = _load_json(p)
    v = (d.get("views") or {}).get(view)
    if v is None:
        return _no_store(f"views.{view} in {p.name}")
    c = v.get(cid)
    if c is None:
        return _no_store(f"{cid} under views.{view}")
    m = ((c.get("gen_stats") or {}).get("mmlu") or {})
    if "acc" not in m:
        return Reading(None, False, "no gen_stats.mmlu.acc", source=str(p))
    return Reading(to_units(m["acc"], units), True, None, f"{p} ›views.{view}.{cid}.gen_stats.mmlu", n=m.get("n"))


def _gsm8k_report(root_or_file: Path | None) -> Path | None:
    if root_or_file is None:
        return None
    if root_or_file.is_file():
        return root_or_file
    direct = root_or_file / "gsm8k" / "report.json"
    if direct.is_file():
        return direct
    return _first([Path(p) for p in sorted(_glob(str(root_or_file / "gsm8k*" / "**" / "report.json"), recursive=True))])


def read_gsm8k_report(report: Path | None, model: str, view: str, cid: str, units: str) -> Reading:
    if report is None or not report.is_file():
        return _no_store("gsm8k report.json")
    d = _load_json(report)
    key = f"{model}/{view}/{cid}"
    c = (d.get("cells") or {}).get(key)
    if c is None:
        return _no_store(f"cells[{key}] in {report}")
    m = ((c.get("readings") or {}).get("merged") or {})
    if not m or not m.get("n_scored"):
        return Reading(None, False, "no readings.merged.n_scored", source=str(report))
    acc = int(m["n_correct"]) / int(m["n_scored"])
    return Reading(to_units(acc, units), True, None, f"{report} ›cells[{key}].readings.merged",
                   k=int(m["n_correct"]), n=int(m["n_scored"]), extra={"escalated": m.get("escalated"), "status": c.get("status")})


def _transfer_stats_files(root: Path) -> list[Path]:
    pats = [str(root / "transfer_stats*.json"), str(root / "*" / "transfer_stats*.json")]
    return sorted({Path(p) for pat in pats for p in _glob(pat)})


def _model_block(d: Mapping[str, Any], view: str) -> Mapping[str, Any] | None:
    models = d.get("models") or {}
    if view in models:
        return models[view]
    short = view.split("_s")[-1] if "_s" in view else None
    for m in models.values():
        if isinstance(m, Mapping) and m.get("view") == f"s{short}" and m.get("base_target", MODEL) == MODEL:
            return m
    return None


def read_transfer_stats(root: Path, view: str, family: str, cid: str, units: str, *, exact: bool = False) -> Reading:
    """`transfer_stats*.json` → `models.<view>.cells.<cid>.<family>`. Default: `_model_block` (the exact key, else the block
    whose `view`/`base_target` match — the object-space tables' convention). ``exact``: ONLY `models[<view>]` (the battery's
    rule — a block that merely shares the view digit is never substituted). The selected block's `view` and `base_target`
    ride in ``extra`` so the caller can validate its identity."""
    files = _transfer_stats_files(root)
    if not files:
        return _no_store("transfer_stats*.json")
    hits = []
    for f in files:
        d = _load_json(f)
        m = (d.get("models") or {}).get(view) if exact else _model_block(d, view)
        if not isinstance(m, Mapping):
            continue
        fam = ((m.get("cells") or {}).get(cid) or {}).get(family)
        if fam is not None:
            hits.append((f, m, fam))
    where = f"models[{view}].cells.{cid}.{family}" if exact else f"models.{view}.cells.{cid}.{family}"
    if not hits:
        return _no_store(f"{where}{' (exact key)' if exact else ''} in {[f.name for f in files]}")
    if len(hits) > 1:
        raise RuntimeError(f"{view}/{cid}/{family}: {len(hits)} transfer_stats files carry this cell "
                           f"({[str(f) for f, _, _ in hits]}) — one reading per cell; refusing to pick")
    f, m, fam = hits[0]
    ident = {"block_view": m.get("view"), "block_base_target": m.get("base_target")}
    if fam.get("status") and fam.get("status") != "COMPLETE" and "p" not in fam:
        return Reading(None, False, f"status {fam.get('status')!r}: {fam.get('reason')}", source=f"{f} ›{where}", extra=ident)
    if "p" not in fam:
        return Reading(None, False, "no p", source=f"{f} ›{where}", extra=ident)
    return Reading(to_units(fam["p"], units), True, None, f"{f} ›{where}",
                   k=fam.get("k"), n=fam.get("n"), extra={"decode_cap_tokens": fam.get("decode_cap_tokens"),
                                                            "asr_at_n": fam.get("asr_at_n"), "asr_grid": fam.get("asr_grid"), **ident})


def _petri_files(root: Path) -> list[Path]:
    return sorted({Path(p) for pat in (str(root / "petri_stats*" / "*.json"), str(root / "petri_stats*.json"))
                   for p in _glob(pat) if "dispersion" not in Path(p).name})


def read_petri_stats(root: Path, tag: str, cid: str, behaviour: str | None, units: str) -> Reading:
    files = _petri_files(root)
    if not files:
        return _no_store("petri_stats/*.json")
    hits = []
    for f in files:
        d = _load_json(f)
        c = ((d.get("tags") or {}).get(tag) or {}).get("cells", {}).get(cid)
        if c is not None:
            hits.append((f, c))
    if not hits:
        return _no_store(f"tags.{tag}.cells.{cid} in {[f.name for f in files]}")
    if len(hits) > 1:
        raise RuntimeError(f"{tag}/{cid}: {len(hits)} petri_stats files carry this cell — refusing to pick")
    f, c = hits[0]
    if c.get("status") not in (None, "COMPLETE") or (c.get("audited_items") is not None and c.get("audited_expected")
                                                       and c["audited_items"] < c["audited_expected"]):
        return Reading(None, False, f"incomplete audit: status {c.get('status')!r}, {c.get('audited_items')}/{c.get('audited_expected')}",
                       source=str(f))
    if behaviour is None:
        v = c.get("mean_of_behavior_means")
        if v is None:
            means = [b.get("mean") for b in (c.get("behaviors") or {}).values()]
            if not means or any(m is None for m in means):
                return Reading(None, False, "no mean_of_behavior_means and incomplete behaviour means", source=str(f))
            v = sum(means) / len(means)
        return Reading(to_units(v, units), True, None, f"{f} ›tags.{tag}.cells.{cid}.mean_of_behavior_means",
                       n=c.get("audited_items"))
    b = (c.get("behaviors") or {}).get(behaviour)
    if not b or b.get("mean") is None:
        return Reading(None, False, f"no behaviors.{behaviour}.mean", source=str(f))
    if b.get("incomplete"):
        return Reading(None, False, f"behaviors.{behaviour} incomplete", source=str(f))
    return Reading(to_units(b["mean"], units), True, None, f"{f} ›tags.{tag}.cells.{cid}.behaviors.{behaviour}.mean",
                   n=b.get("n_items"))


def misalign_arm(cid: str) -> str:
    return "si_" + cid.replace("-", "_")


def _misalign_dir(root: Path, stem: str) -> Path | None:
    for d in (root / f"{stem}_ladder", root / stem):
        if (d / "summary.csv").is_file():
            return d
    return None


def read_misalign(root: Path, stem: str, cid: str, behaviour: str, units: str) -> Reading:
    d = _misalign_dir(root, stem)
    if d is None:
        return _no_store(f"misalign/{stem}[_ladder]/summary.csv")
    arm = NONE if cid == NONE else misalign_arm(cid)
    with (d / "summary.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("family") == "agentic" and r.get("behavior") == behaviour and r.get("metric") == "harmful"
                and r.get("arm") == arm]
    if not rows:
        return _no_store(f"arm {arm} / agentic / {behaviour} / harmful in {d / 'summary.csv'}")
    if len(rows) > 1:
        raise RuntimeError(f"{d / 'summary.csv'}: {len(rows)} rows for {arm}/{behaviour}/harmful — refusing to pick")
    r = rows[0]
    return Reading(to_units(float(r["estimate"]), units), True, None, f"{d / 'summary.csv'} ›{arm}/agentic/{behaviour}/harmful",
                   k=int(float(r["k"])) if r.get("k") not in (None, "") else None, n=int(float(r["n"])) if r.get("n") else None,
                   extra={"laddered": d.name.endswith("_ladder")})


# ---------------------------------------------------------------- roots + dispatch
@dataclass
class Roots:
    grades: Path
    export: pd.DataFrame
    paper_final_rung: Path | None = None
    paper_gsm8k: Path | None = None
    paper_misalign: Path | None = None
    admissions: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)


def load_admissions(grades: Path) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for p in sorted(glob.glob(str(grades / "admission" / "*.json"))):
        try:
            d = _load_json(Path(p))
        except json.JSONDecodeError:
            continue
        entries = d if isinstance(d, list) else [d]
        for e in entries:
            if not isinstance(e, Mapping) or "admitted" in e and e.get("seed") is None:
                continue
            cid = e.get("cell_id") or e.get("cell")
            if cid and "admitted" in e:
                out[(int(e["seed"]), str(cid))] = dict(e)
    return out


def read_lane(roots: Roots, spec: LegSpec, seed: int, cid: str) -> Reading:
    view, tag = view_of(seed), tag_of(seed)
    g = roots.grades
    if spec.reader == "final_rung_static":
        r = read_final_rung_static(g / "sampled/report/analysis/final_rung.json", tag, spec.args["family"], cid, spec.units)
    elif spec.reader == "final_rung_controls":
        r = read_final_rung_controls(g / "sampled/report/analysis/final_rung.json", tag, spec.args["readout"], cid, spec.units)
    elif spec.reader == "controls_agg_mmlu":
        r = read_controls_agg_mmlu(g, view, cid, spec.units)
    elif spec.reader == "gsm8k_report":
        r = read_gsm8k_report(_gsm8k_report(g), MODEL, f"s{seed}", cid, spec.units)
    elif spec.reader == "transfer_stats":
        r = read_transfer_stats(g, view, spec.args["family"], cid, spec.units)
    elif spec.reader == "petri_stats":
        r = read_petri_stats(g, tag, cid, spec.args["behaviour"], spec.units)
    elif spec.reader == "misalign_csv":
        r = read_misalign(g / "misalign", view, cid, spec.args["behaviour"], spec.units)
    else:
        raise ValueError(f"unknown reader {spec.reader!r}")
    adm = roots.admissions.get((seed, cid))
    if not r.eligible and adm is not None and not adm.get("admitted", True):
        why = adm.get("reasons") or adm.get("reason") or "no reason recorded"
        r.reason = f"not admitted: {why if isinstance(why, str) else '; '.join(map(str, why))}"
    return r


def read_preflight(grades: Path, view: str, spec: LegSpec) -> tuple[str | None, str | None]:
    """``(verdict, path)``; verdict None when no record exists (never assumed comparable)."""
    for name in (f"{view}_{spec.preflight_leg}.json", f"{view}_{spec.eval_id}.json"):
        p = grades / "anchor_preflight" / name
        if p.is_file():
            d = _load_json(p)
            v = d.get("anchor_sampler_verdict", d.get("verdict"))
            return (str(v) if v is not None else None), str(p)
    return None, None


def comparable(verdict: str | None) -> bool:
    return verdict is not None and verdict.split("(")[0].strip() == "comparable"


# ---------------------------------------------------------------- the export (paper deltas + floor)
def _match(series: pd.Series, target: Any) -> pd.Series:
    if target is None:
        return series.isna()
    return pd.to_numeric(series, errors="coerce") == float(target)


def export_rows(df: pd.DataFrame, cid: str, key: Mapping[str, Any], view: str = PAPER_VIEW) -> pd.DataFrame:
    m = (df["view"] == view) & (df["cell_id"] == cid) & (df["eval_id"] == key["eval_id"]) & (df["metric"] == key["metric"])
    if "dose" in df.columns:
        m &= pd.to_numeric(df["dose"], errors="coerce") == 1.0
    if "threshold" in df.columns:
        m &= _match(df["threshold"], key.get("threshold"))
    if "bon_n" in df.columns:
        m &= _match(df["bon_n"], key.get("bon_n"))
    if key.get("cap") is not None and "decode_cap_tokens" in df.columns:
        m &= pd.to_numeric(df["decode_cap_tokens"], errors="coerce") == float(key["cap"])
    if key.get("decode_contains") and "decode" in df.columns:
        m &= df["decode"].astype("string").fillna("").str.contains(key["decode_contains"], regex=False)
    if "status" in df.columns:
        m &= df["status"].isin(STATUS_OK) | df["status"].isna()
    return df[m]


def paper_from_export(df: pd.DataFrame, cid: str, spec: LegSpec, warnings: list[str], view: str = PAPER_VIEW) -> Reading:
    rows = export_rows(df, cid, spec.export, view=view)
    if rows.empty:
        return Reading(None, False, f"no export row for view={view} cell={cid} key={spec.export}", source="export")
    if len(rows) > 1:
        with_delta = rows[rows["delta"].notna()] if "delta" in rows.columns else rows
        rows = with_delta if not with_delta.empty else rows
    if len(rows) > 1:
        rows = rows.sort_values("decode_cap_tokens", ascending=False, na_position="last") if "decode_cap_tokens" in rows.columns else rows
        warnings.append(f"{spec.eval_id}/{cid}: {len(rows)} export rows match {spec.export}; took row_id "
                        f"{rows.iloc[0].get('row_id', '?')} (largest cap) — candidates {rows.get('row_id', pd.Series(dtype=object)).tolist()}")
    r = rows.iloc[0]
    extra = {k: (None if pd.isna(r[k]) else (float(r[k]) if k != "delta_method" else str(r[k])))
             for k in ("delta", "delta_lo", "delta_hi", "delta_method") if k in rows.columns}
    extra["export_row_id"] = None if "row_id" not in rows.columns or pd.isna(r["row_id"]) else int(r["row_id"])
    extra["decode_cap_tokens"] = None if "decode_cap_tokens" not in rows.columns or pd.isna(r["decode_cap_tokens"]) else int(r["decode_cap_tokens"])
    extra["decode"] = None if "decode" not in rows.columns or pd.isna(r["decode"]) else str(r["decode"])
    extra["view"] = view
    return Reading(to_units(float(r["value"]), spec.units), True, None,
                   f"export row_id {extra['export_row_id']} ({view}, {spec.export})",
                   k=None if "k" not in rows.columns or pd.isna(r["k"]) else int(r["k"]),
                   n=None if "n" not in rows.columns or pd.isna(r["n"]) else int(r["n"]), extra=extra)


def sibling_views(df: pd.DataFrame) -> list[str]:
    """The other Qwen views serving the SAME checkpoint as the paper view (the instruct-direct
    view first: it is where the campaign filed the 4,096 multi-turn anchor)."""
    if "served_model" in df.columns:
        served = df.loc[df["view"] == PAPER_VIEW, "served_model"].dropna().unique().tolist()
        views = df.loc[df["served_model"].isin(served) & df["view"].astype(str).str.startswith(MODEL), "view"].unique().tolist()
    else:
        views = [v for v in df["view"].unique().tolist() if str(v).startswith(MODEL)]
    others = sorted(str(v) for v in views if str(v) != PAPER_VIEW)
    pref = f"{PAPER_VIEW}_it"
    return ([pref] if pref in others else []) + [v for v in others if v != pref]


def paper_anchor_from_export(df: pd.DataFrame, spec: LegSpec, warnings: list[str]) -> Reading:
    """The paper's ``none`` reading at the leg's key. The main view first; when it filed no
    ``none`` at this key (the campaign's 4,096 multi-turn anchor lives under the instruct-
    direct view, which serves the same checkpoint — ``paper/paperlib/load.py``
    ``borrow_sibling_view_anchor`` is the precedent) the sibling view's is BORROWED and the
    borrow is recorded on the reading, in the anchor block and in the warnings — never
    silently."""
    r = paper_from_export(df, NONE, spec, warnings)
    if r.eligible:
        r.extra["borrowed"] = False
        return r
    tried = [PAPER_VIEW]
    for v in sibling_views(df):
        tried.append(v)
        rv = paper_from_export(df, NONE, spec, warnings, view=v)
        if rv.eligible:
            rv.extra["borrowed"] = True
            rv.extra["borrowed_from_view"] = v
            rv.source = (f"{rv.source} — BORROWED anchor: view {PAPER_VIEW} filed no `none` at this key; {v} serves the "
                         "same checkpoint")
            warnings.append(f"{spec.eval_id}: the paper's `none` at key {spec.export} is borrowed from view {v} "
                            f"(view {PAPER_VIEW} filed none at this key; same served checkpoint)")
            return rv
    r.reason = f"no export `none` row at key {spec.export} in any Qwen view sharing the checkpoint (tried {tried})"
    return r


def cross_view_floor(df: pd.DataFrame, spec: LegSpec) -> dict[str, Any] | None:
    """Range of the paper's ``none`` reading across the Qwen views that serve the same
    checkpoint (each view drew its own unsteered anchor): the decode-noise reference in
    the leg's units. None when fewer than two views carry the key."""
    if "served_model" in df.columns:
        served = df.loc[df["view"] == PAPER_VIEW, "served_model"].dropna().unique().tolist()
        views = df.loc[df["served_model"].isin(served) & df["view"].astype(str).str.startswith(MODEL), "view"].unique().tolist()
    else:
        views = [v for v in df["view"].unique().tolist() if str(v).startswith(MODEL)]
    vals: dict[str, float] = {}
    for v in sorted(views):
        rows = export_rows(df, NONE, spec.export, view=v)
        if rows.empty:
            continue
        rows = rows.sort_values("decode_cap_tokens", ascending=False, na_position="last") if "decode_cap_tokens" in rows.columns else rows
        vals[str(v)] = to_units(float(rows.iloc[0]["value"]), spec.units)
    if len(vals) < 2:
        return None
    return {"units": spec.units, "range": max(vals.values()) - min(vals.values()), "n_views": len(vals),
            "none_by_view": vals, "instrument": f"export key {spec.export}",
            "definition": "max − min of the unsteered `none` reading across Qwen views serving the same checkpoint"}


# ---------------------------------------------------------------- paper readings per source
def paper_reading(roots: Roots, spec: LegSpec, cid: str, warnings: list[str]) -> Reading:
    if spec.paper_source == "export":
        return paper_anchor_from_export(roots.export, spec, warnings) if cid == NONE else paper_from_export(roots.export, cid, spec, warnings)
    if spec.paper_source == "final_rung":
        fr = roots.paper_final_rung
        if spec.reader == "final_rung_static":
            r = read_final_rung_static(fr, PAPER_VIEW, spec.args["family"], cid, spec.units)
        else:
            r = read_final_rung_controls(fr, PAPER_VIEW, spec.args["readout"], cid, spec.units)
        if r.eligible:
            r.source = f"paper {r.source}"
        return r
    if spec.paper_source == "gsm8k_report":
        r = read_gsm8k_report(roots.paper_gsm8k, MODEL, "main", cid, spec.units)
        if r.eligible:
            r.source = f"paper {r.source}"
        return r
    if spec.paper_source == "misalign_csv":
        if roots.paper_misalign is None:
            return _no_store("paper misalign root")
        r = read_misalign(roots.paper_misalign, MODEL, cid, spec.args["behaviour"], spec.units)
        if r.eligible:
            r.source = f"paper {r.source}"
        return r
    raise ValueError(f"unknown paper_source {spec.paper_source!r}")


def _reading_json(r: Reading | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return {"value": r.value, "eligible": r.eligible, "reason": r.reason, "source": r.source, "k": r.k, "n": r.n,
            **({"extra": r.extra} if r.extra else {})}


# ---------------------------------------------------------------- per-leg analysis
def _gate(records: Sequence[Record]) -> tuple[dict[str, Any], dict[int, float] | None]:
    try:
        trip = require_triplet(records)
    except TripletUnavailable as exc:
        return {"status": str(exc), "criterion_7": "FAILED", "missing": [{"seed": s, "reason": r} for s, r in exc.missing],
                "spread": None}, None
    vals = {s: float(trip[s]) for s in SEEDS}
    return {"status": "ok", "criterion_7": "PASS", "spread": spread([vals[s] for s in SEEDS])}, vals


def analyze_leg(roots: Roots, spec: LegSpec, warnings: list[str], excluded: list[dict[str, Any]]) -> dict[str, Any]:
    leg: dict[str, Any] = {"eval_id": spec.eval_id, "label": spec.label, "category": spec.category, "units": spec.units,
                           "reader": spec.reader, "paper_source": spec.paper_source, "export_key": spec.export,
                           "note": spec.note, "preflight": {}, "arms": {}, "contrasts": {}, "paper": {}, "floor": None,
                           "claims": []}
    # anchor preflight per view
    verdicts: dict[int, str | None] = {}
    for s in SEEDS:
        v, p = read_preflight(roots.grades, view_of(s), spec)
        verdicts[s] = v
        leg["preflight"][f"s{s}"] = {"verdict": v, "path": p, "comparable": comparable(v)}
        if not comparable(v):
            excluded.append({"leg": spec.eval_id, "view": view_of(s), "verdict": v if v is not None else "no anchor preflight record",
                             "consequence": "arm spread reported; no Δ(difference − none) for this seed"})
    # per-cell readings
    readings: dict[str, dict[int, Reading]] = {}
    for cls in CLASSES:
        readings[cls.cell_id] = {s: read_lane(roots, spec, s, cls.cell_id) for s in SEEDS}
        recs = [Record(s, r.value, r.eligible, r.reason) for s, r in readings[cls.cell_id].items()]
        g, vals = _gate(recs)
        leg["arms"][cls.cell_id] = {"arm": cls.arm, "role": cls.role, **g,
                                    "values": {f"s{s}": readings[cls.cell_id][s].value for s in SEEDS},
                                    "readings": {f"s{s}": _reading_json(readings[cls.cell_id][s]) for s in SEEDS}}
    # paper readings (six cells + none) and the floor
    paper: dict[str, Reading] = {c.cell_id: paper_reading(roots, spec, c.cell_id, warnings) for c in CLASSES}
    paper[NONE] = paper_reading(roots, spec, NONE, warnings)
    leg["paper"] = {cid: _reading_json(r) for cid, r in paper.items()}
    if spec.paper_source != "export":     # the export row rides beside for the reader, never as the claim's Δ
        leg["paper_export_beside"] = {cid: _reading_json(paper_from_export(roots.export, cid, spec, warnings))
                                      for cid in [c.cell_id for c in CLASSES] + [NONE]}
    leg["floor"] = cross_view_floor(roots.export, spec)
    none_paper = paper[NONE].value if paper[NONE].eligible else None
    # contrasts per arm type
    for arm, roles in CELLS_BY_ARM.items():
        const, ctl, diff = roles["constitution"].cell_id, roles["control"].cell_id, roles["difference"].cell_id
        out: dict[str, Any] = {}
        # Δ(constitution − control) per seed: both lane cells
        recs = []
        for s in SEEDS:
            a, b = readings[const][s], readings[ctl][s]
            if a.eligible and b.eligible:
                recs.append(Record(s, a.value - b.value, True))
            else:
                why = "; ".join(f"{n}: {r.reason}" for n, r in (("constitution", a), ("control", b)) if not r.eligible)
                recs.append(Record(s, None, False, why))
        g, vals = _gate(recs)
        out["const_minus_ctl"] = {**g, "per_seed": {f"s{s}": (None if vals is None else vals[s]) for s in SEEDS}}
        # Δ(difference − none) per seed: the lane's difference cell vs the REUSED paper anchor, only where comparable
        recs = []
        for s in SEEDS:
            d = readings[diff][s]
            if not d.eligible:
                recs.append(Record(s, None, False, f"difference: {d.reason}"))
            elif none_paper is None:
                recs.append(Record(s, None, False, f"anchor: {paper[NONE].reason}"))
            elif not comparable(verdicts[s]):
                recs.append(Record(s, None, False, f"{REASON_NON_COMPARABLE}: anchor preflight "
                                                    f"{verdicts[s] if verdicts[s] is not None else 'record missing'}"))
            else:
                recs.append(Record(s, d.value - none_paper, True))
        g, vals = _gate(recs)
        out["diff_minus_none"] = {**g, "per_seed": {f"s{s}": (None if vals is None else vals[s]) for s in SEEDS},
                                  "anchor": {"cell": NONE, "value": none_paper, "source": paper[NONE].source, "reused": True,
                                             "view": paper[NONE].extra.get("borrowed_from_view", paper[NONE].extra.get("view", PAPER_VIEW)),
                                             "borrowed": bool(paper[NONE].extra.get("borrowed", False))}}
        # the paper's contrasts, in the leg's units from the paper source's own values
        pc: dict[str, Any] = {}
        pa, pb, pd_ = paper[const], paper[ctl], paper[diff]
        pc["const_minus_ctl"] = ({"delta": pa.value - pb.value, "source": spec.paper_source} if pa.eligible and pb.eligible
                                 else {"delta": None, "reason": "; ".join(f"{n}: {r.reason}" for n, r in (("constitution", pa), ("control", pb)) if not r.eligible)})
        pc["diff_minus_none"] = ({"delta": pd_.value - none_paper, "source": spec.paper_source,
                                  "export_delta": pd_.extra.get("delta"), "export_delta_lo": pd_.extra.get("delta_lo"),
                                  "export_delta_hi": pd_.extra.get("delta_hi"), "export_delta_method": pd_.extra.get("delta_method")}
                                 if pd_.eligible and none_paper is not None
                                 else {"delta": None, "reason": "; ".join(f"{n}: {r.reason}" for n, r in (("difference", pd_), ("none", paper[NONE])) if not r.eligible)})
        out["paper"] = pc
        leg["contrasts"][arm] = out
        # claims
        for contrast in CONTRASTS:
            c = out[contrast]
            claim = {"leg": spec.eval_id, "category": spec.category, "arm": arm, "contrast": contrast, "units": spec.units,
                     "delta_paper": pc[contrast].get("delta"), "sd_seed": None, "range_seed": None,
                     "threshold_2sd": None, "threshold_range": None, "threshold": None, "exceeds": None,
                     "floor": None if leg["floor"] is None else leg["floor"]["range"]}
            if c["spread"] is None:
                claim["status"] = c["status"]
            elif pc[contrast].get("delta") is None:
                claim["status"] = f"paper delta unavailable: {pc[contrast].get('reason')}"
                claim.update(sd_seed=c["spread"]["sd"], range_seed=c["spread"]["range"],
                             threshold_2sd=2 * c["spread"]["sd"], threshold_range=c["spread"]["range"],
                             threshold=max(c["spread"]["range"], 2 * c["spread"]["sd"]))
            else:
                sd, rg = c["spread"]["sd"], c["spread"]["range"]
                thr = max(rg, 2 * sd)
                exceeds = abs(pc[contrast]["delta"]) > thr
                claim.update(sd_seed=sd, range_seed=rg, threshold_2sd=2 * sd, threshold_range=rg, threshold=thr,
                             exceeds=exceeds,
                             status="exceeds the seed spread" if exceeds else "does not exceed the seed spread")
                if thr == 0.0:
                    # three identical readings: the rule is vacuous here (every non-zero Δ exceeds 0) — say so
                    claim["note"] = "zero seed spread (the three seeds read identically); the threshold is 0 and the rule is degenerate for this cell"
                    claim["status"] += " (degenerate: zero seed spread)"
            leg["claims"].append(claim)
    return leg


# ---------------------------------------------------------------- rendering
def _f(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def _u(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def render_md(res: Mapping[str, Any]) -> str:
    lines = ["# Seed spread of the Qwen readings beside the paper's deltas (criterion 7)", "",
             f"grades root `{res['grades']}` · export `{res['export']}` · seeds {list(SEEDS)} · views "
             f"{[view_of(s) for s in SEEDS]}", "",
             "Rule: a paper delta **exceeds the seed spread** iff `|Δ_paper| > max(range_seed, 2·SD_seed)` "
             "(SD is the sample SD over s1–s3, ddof 1; both thresholds are printed). Units: `pp` = percentage points "
             "(rates × 100), `score` = Petri judged score (1–10). The cross-view replicate floor is the range of the "
             "paper's unsteered `none` reading across the Qwen views that serve the same checkpoint, at the leg's "
             "export key.", ""]
    for cat in CATEGORY_ORDER:
        legs = [l for l in res["legs"] if l["category"] == cat]
        if not legs:
            continue
        lines += [f"## {cat}", ""]
        lines += ["| leg | arm | contrast | s1 | s2 | s3 | SD | range | max(range, 2·SD) | paper Δ | export Δ [95 %] | floor | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for l in legs:
            for arm in ("lora", "psr"):
                for contrast in CONTRASTS:
                    c = l["contrasts"][arm][contrast]
                    pc = l["contrasts"][arm]["paper"][contrast]
                    claim = next(k for k in l["claims"] if k["arm"] == arm and k["contrast"] == contrast)
                    ps = c["per_seed"]
                    ex = ""
                    if contrast == "diff_minus_none" and pc.get("export_delta") is not None:
                        ex = f"{_f(pc['export_delta'])} [{_f(pc.get('export_delta_lo'))}, {_f(pc.get('export_delta_hi'))}]"
                    verdict = claim.get("status") or "—"
                    lines.append(f"| {l['label']} | {F.ARM_LABEL[arm]} | {F.CONTRAST_LABEL[contrast]} | {_f(ps['s1'])} | {_f(ps['s2'])} | {_f(ps['s3'])} | "
                                 f"{_u(claim['sd_seed'])} | {_u(claim['range_seed'])} | {_u(claim['threshold'])} | {_f(claim['delta_paper'])} | {ex or '—'} | "
                                 f"{_u(claim['floor'])} | {verdict} |")
        borrowed = [l for l in legs if l["contrasts"]["lora"]["diff_minus_none"]["anchor"].get("borrowed")]
        if borrowed:
            lines += ["", "Anchors borrowed from a sibling view serving the same checkpoint (the main view filed no `none` at the key): "
                      + "; ".join(f"{l['label']} ← `{l['contrasts']['lora']['diff_minus_none']['anchor']['view']}`" for l in borrowed)]
        lines += ["", f"### arm values ({cat})", "",
                  "| leg | cell | s1 | s2 | s3 | SD | range | paper value | criterion 7 | status |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for l in legs:
            for cid, a in l["arms"].items():
                sp = a["spread"] or {}
                pv = (l["paper"].get(cid) or {}).get("value")
                v = a["values"]
                lines.append(f"| {l['label']} | `{cid}` | {_u(v['s1'])} | {_u(v['s2'])} | {_u(v['s3'])} | {_u(sp.get('sd'))} | "
                             f"{_u(sp.get('range'))} | {_u(pv)} | {a['criterion_7']} | {a['status']} |")
        lines.append("")
    lines += ["## claim list", ""]
    for word in ("exceeds the seed spread", "does not exceed the seed spread"):
        ks = [k for k in res["claims"] if k.get("status") == word]
        lines += [f"### {word} ({len(ks)})", ""]
        lines += [f"- {k['leg']} · {F.ARM_LABEL[k['arm']]} · {F.CONTRAST_LABEL[k['contrast']]}: paper Δ {_f(k['delta_paper'])} "
                  f"{k['units']} vs max(range {_u(k['range_seed'])}, 2·SD {_u(k['threshold_2sd'])}) = {_u(k['threshold'])}"
                  for k in ks] or ["- none"]
        lines.append("")
    ks = [k for k in res["claims"] if k.get("exceeds") is None]
    lines += [f"### not testable ({len(ks)})", ""] + [f"- {k['leg']} · {F.ARM_LABEL[k['arm']]} · {F.CONTRAST_LABEL[k['contrast']]}: {k.get('status')}"
                                                       for k in ks] + [""]
    lines += ["## legs excluded from Δ(difference − none) by the anchor preflight", ""]
    lines += [f"- {e['leg']} · {e['view']}: {e['verdict']} — {e['consequence']}" for e in res["excluded_non_comparable"]] or ["- none"]
    lines += ["", "## warnings", ""] + ([f"- {w}" for w in res["warnings"]] or ["- none"]) + [""]
    return "\n".join(lines)


def spread_panels(res: Mapping[str, Any]) -> list[dict[str, Any]]:
    panels = []
    for cat in CATEGORY_ORDER:
        legs = [l for l in res["legs"] if l["category"] == cat]
        if not legs:
            continue
        units = legs[0]["units"]
        rows = []
        for l in legs:
            arms = {}
            for arm in ("lora", "psr"):
                arms[arm] = {}
                for contrast in CONTRASTS:
                    c = l["contrasts"][arm][contrast]
                    arms[arm][contrast] = {"seeds": None if c["spread"] is None else c["spread"]["values"],
                                           "paper": l["contrasts"][arm]["paper"][contrast].get("delta"), "status": c["status"]}
            rows.append({"leg": l["eval_id"], "label": l["label"], "floor": None if l["floor"] is None else l["floor"]["range"], "arms": arms})
        panels.append({"category": cat, "units": units, "legs": rows})
    return panels


# ---------------------------------------------------------------- driver
def run(roots: Roots, out: Path, figs_dir: Path | None = None, *, make_figs: bool = True,
        legs: Sequence[LegSpec] = LEGS) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    excluded: list[dict[str, Any]] = []
    leg_rows = [analyze_leg(roots, spec, warnings, excluded) for spec in legs]
    res = {"grades": str(roots.grades), "export": getattr(roots.export, "attrs", {}).get("path"),
           "paper_sources": {"final_rung": str(roots.paper_final_rung), "gsm8k_report": str(roots.paper_gsm8k),
                             "misalign_root": str(roots.paper_misalign)},
           "seeds": list(SEEDS), "views": [view_of(s) for s in SEEDS], "legs": leg_rows,
           "claims": [k for l in leg_rows for k in l["claims"]], "excluded_non_comparable": excluded, "warnings": warnings,
           "criterion_7": {l["eval_id"]: {cid: a["criterion_7"] for cid, a in l["arms"].items()} for l in leg_rows}}
    (out / "seed_spread.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (out / "seed_spread.md").write_text(render_md(res))
    figs: dict[str, Path] = {}
    if make_figs:
        fd = Path(figs_dir or (out.parent / "figs"))
        figs["seed_spread"] = F.seed_spread_figure(spread_panels(res), fd / "seed_spread.png")
    res["figures"] = {k: str(v) for k, v in figs.items()}
    return res


def load_export(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.attrs["path"] = str(path)
    return df


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grades", type=Path, required=True, help="the lane's run-state root (RS/reseed)")
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--figs", type=Path, default=None, help="PNG directory (default <out>/../figs)")
    ap.add_argument("--paper-final-rung", type=Path, default=DEFAULT_PAPER_FINAL_RUNG)
    ap.add_argument("--paper-gsm8k-report", type=Path, default=DEFAULT_PAPER_GSM8K)
    ap.add_argument("--paper-misalign-root", type=Path, default=DEFAULT_PAPER_MISALIGN)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--mirror-dest", type=Path, default=F.DEFAULT_MIRROR)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args(argv)
    if str(_HERE.parents[2] / "src") not in sys.path:
        sys.path.insert(0, str(_HERE.parents[2] / "src"))
    from sisteer import site as _site
    a.out = _site.expand_path(a.out, key="--out")   # a write root: an unset $GRADE refuses by name
    a.paper_gsm8k_report = _site.expand_path(a.paper_gsm8k_report, key="--paper-gsm8k-report")
    a.paper_final_rung = _site.expand_path(a.paper_final_rung, key="--paper-final-rung")
    # every input is checked before anything is read, and the missing ones refuse ONCE, each with its flag (the defaults
    # are one site's trees; a missing one used to crash on the first read or leave rows saying "no store")
    missing = [f"{flag} {path}" for flag, path, ok in (
        ("--grades", a.grades, Path(a.grades).is_dir()),
        ("--export", a.export, Path(a.export).is_file()),
        ("--paper-final-rung", a.paper_final_rung, Path(a.paper_final_rung).is_file()),
        ("--paper-gsm8k-report", a.paper_gsm8k_report, Path(a.paper_gsm8k_report).is_file()),
        ("--paper-misalign-root", a.paper_misalign_root, Path(a.paper_misalign_root).is_dir())) if not ok]
    if missing:
        raise SystemExit("[seed_spread] REFUSED: missing input(s), pass each flag: " + "; ".join(missing))
    roots = Roots(grades=a.grades, export=load_export(a.export), paper_final_rung=a.paper_final_rung,
                  paper_gsm8k=a.paper_gsm8k_report, paper_misalign=a.paper_misalign_root,
                  admissions=load_admissions(a.grades))
    res = run(roots, a.out, a.figs, make_figs=not a.no_figures)
    n_ok = sum(1 for k in res["claims"] if k.get("exceeds") is not None)
    print(f"[seed_spread] {len(res['legs'])} legs; {n_ok} testable claims "
          f"({sum(1 for k in res['claims'] if k.get('exceeds'))} exceed); "
          f"{len(res['excluded_non_comparable'])} (leg, view) excluded by the anchor preflight; {len(res['warnings'])} warnings")
    for l in res["legs"]:
        st = {cid: a["criterion_7"] for cid, a in l["arms"].items()}
        print(f"[seed_spread] {l['eval_id']:<28} criterion 7: {sum(v == 'PASS' for v in st.values())}/{len(st)} arms complete")
    figs_dir = a.figs or (a.out.parent / "figs")
    if a.mirror and res["figures"]:
        F.mirror_figs(figs_dir, a.mirror_dest)
    if a.wandb and res["figures"]:
        print("[seed_spread] wandb:", F.log_wandb({k: Path(v) for k, v in res["figures"].items()}, figs_dir,
                                                  config={"grades": str(a.grades)}))
    return 0


# ================================================================ battery extensions
# Everything below is ADDITIVE: the object-space / interim readers and the claim rule above are untouched; the battery
# tables (battery_tables.py) import these so the seed-spread statistic has ONE definition with ONE test.
#
#   seed-spread ratio  =  SD_between(s1, s2, s3)  /  SE_within
#   SD_between  = sample SD (ddof 1) of the three per-seed headline values (triplet_gate.spread — refuses < 3 values)
#   SE_within   = sqrt(mean_i SE_i^2), the RMS of the three per-seed sampling SEs (binomial for a rate, bootstrap for a
#                 per-item score); with equal n this is the pooled binomial SE
#   reading     : under "no seed effect" E[SD_between^2] = SE_within^2, so a ratio ≈ 1 means the seeds add nothing beyond
#                 the per-seed sampling noise; > 1 quantifies replicate (seed-attributable) variance; `sd_excess` =
#                 sqrt(max(SD_between^2 − SE_within^2, 0)) is the seed-attributable SD implied by that reading (clipped at 0)
#
# Every number is float64 (Python floats / numpy float64); every random draw is seeded from the identity of the thing being
# computed (`item_seed`), never from a shared RNG, so a re-run — or a different call order — reproduces every bootstrap SE.

import hashlib  # noqa: E402

RATIO_DEFINITION = ("between-seed sample SD (ddof 1, over exactly s1–s3) divided by the RMS of the three per-seed sampling "
                    "SEs (binomial sqrt(p(1−p)/n) for a rate; seeded bootstrap of the per-item mean for a score); ≈ 1 = "
                    "seeds add nothing beyond sampling noise, > 1 = replicate variance")
BOOTSTRAP_DRAWS = 4000


def item_seed(*parts: Any) -> int:
    """A 32-bit RNG seed from the IDENTITY of a computation (leg, arm, seed, statistic …), never a shared counter."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def binomial_se(k: int | None, n: int | None, *, scale: float = 1.0) -> tuple[float | None, str | None]:
    """``(SE, None)`` of a rate k/n in the caller's units (``scale`` 100 for pp), or ``(None, reason)``.
    k ∈ {0, n} gives SE 0 — returned as 0.0 with a reason so a ratio built on it can say *undefined*, never *infinite*."""
    if k is None or n is None:
        return None, "no k/n on the reading"
    k, n = int(k), int(n)
    if n <= 0:
        return None, f"n = {n}"
    if not 0 <= k <= n:
        return None, f"k = {k} outside [0, n = {n}]"
    p = k / n
    se = float(scale) * math.sqrt(p * (1.0 - p) / n)
    return se, (f"k ∈ {{0, n}} (k = {k}, n = {n}): binomial SE is 0" if se == 0.0 else None)


def bootstrap_means(values: Sequence[float], *, seed: int, draws: int = BOOTSTRAP_DRAWS):
    """The resampled per-draw means (a float64 array of ``draws``) of per-item ``values`` under RNG ``seed`` — the ONE
    resampling core every bootstrap SE here is built from. Fewer than two items or a NaN raise ``ValueError``."""
    import numpy as np

    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size < 2:
        raise ValueError(f"{arr.size} item(s): bootstrap SE needs ≥ 2")
    if np.isnan(arr).any():
        raise ValueError("NaN among the per-item values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(draws), arr.size))
    return arr[idx].mean(axis=1, dtype=np.float64)


def bootstrap_se(values: Sequence[float], *, identity: Sequence[Any], draws: int = BOOTSTRAP_DRAWS,
                 scale: float = 1.0) -> tuple[float | None, str | None]:
    """Bootstrap SE of the mean of per-item ``values`` (float64), RNG seeded from ``identity`` via :func:`item_seed`.
    Fewer than two items → ``(None, reason)``."""
    try:
        means = bootstrap_means(values, seed=item_seed(*identity, "bootstrap_se"), draws=draws)
    except ValueError as exc:
        return None, str(exc)
    return float(scale) * float(means.std(ddof=1)), None


def pooled_se(per_seed_se: Mapping[int, float | None]) -> tuple[float | None, str | None]:
    """RMS of the three per-seed SEs (``sqrt(mean se_i^2)``); refuses unless every seed in SEEDS carries one."""
    missing = [s for s in SEEDS if per_seed_se.get(s) is None]
    if missing:
        return None, "no per-seed SE for " + ", ".join(f"s{s}" for s in missing)
    ses = [float(per_seed_se[s]) for s in SEEDS]
    return math.sqrt(sum(x * x for x in ses) / len(ses)), None


def seed_spread_ratio(per_seed_values: Mapping[int, float] | None, per_seed_se: Mapping[int, float | None]) -> dict[str, Any]:
    """The statistic. ``per_seed_values`` is the COMPLETE triplet (the caller has already passed the triplet gate; None =
    incomplete → the ratio is unavailable, never 0). Returns ``{sd_between, se_within, ratio, sd_excess, status, definition}``
    where ``ratio`` is None with a ``status`` naming why whenever it cannot be formed."""
    out: dict[str, Any] = {"sd_between": None, "se_within": None, "ratio": None, "sd_excess": None,
                           "status": None, "definition": RATIO_DEFINITION,
                           "per_seed_se": {f"s{s}": per_seed_se.get(s) for s in SEEDS}}
    if per_seed_values is None:
        out["status"] = "unavailable: incomplete replicate triplet (no between-seed SD)"
        return out
    if set(int(k) for k in per_seed_values) != set(SEEDS):
        raise TripletRefused(f"seed_spread_ratio takes the COMPLETE triplet {{1, 2, 3}} (or None when the gate refused), got seeds "
                             f"{sorted(int(k) for k in per_seed_values)} — a spread over survivors is never computed here")
    sp = spread([per_seed_values[s] for s in SEEDS])      # refuses anything but exactly three values
    out["sd_between"] = float(sp["sd"])
    se_w, why = pooled_se(per_seed_se)
    if se_w is None:
        out["status"] = f"unavailable: {why}"
        return out
    out["se_within"] = se_w
    if se_w == 0.0:
        out["status"] = "undefined: per-seed SE is 0 on every seed (k ∈ {0, n}); the ratio has no denominator"
        return out
    out["ratio"] = out["sd_between"] / se_w
    out["sd_excess"] = math.sqrt(max(out["sd_between"] ** 2 - se_w ** 2, 0.0))
    out["status"] = "ok"
    return out


def _by_seed(records: Mapping[int, Record], what: str) -> dict[int, Record]:
    """A ``{seed: Record}`` mapping checked the way the gate checks it: every key names its record's own seed (a key/seed
    disagreement or a duplicate hidden under a foreign key REFUSES), and no seed lies outside SEEDS."""
    rows = normalize({int(k): [v] for k, v in records.items()})      # raises TripletRefused on key ≠ record.seed
    foreign = sorted({r.seed for r in rows if r.seed not in SEEDS})
    if foreign:
        raise UnexpectedSeed(foreign)
    counts: dict[int, int] = {}
    for r in rows:
        counts[r.seed] = counts.get(r.seed, 0) + 1
    if any(n > 1 for n in counts.values()):
        raise DuplicateReplicate(counts)
    return {r.seed: r for r in rows}


def paired_contrast(arm: Mapping[int, Record], anchor: Mapping[int, Record]) -> dict[str, Any]:
    """Δ(arm − same-seed anchor) per seed through the triplet gate: ``{per_seed, mean, sd, range, min, max, status}``.
    The anchor is the SAME seed's reading (the view's own `none` is the ONLY anchor); a seed
    missing on either side makes the contrast unavailable for the triplet (no mean over survivors)."""
    arm, anchor = _by_seed(arm, "arm"), _by_seed(anchor, "anchor")
    recs: list[Record] = []
    for s in SEEDS:
        a, b = arm.get(s), anchor.get(s)
        if a is None or b is None:
            recs.append(Record(s, None, False, "arm: no record" if a is None else "anchor: no record"))
        elif a.eligible and b.eligible:
            recs.append(Record(s, float(a.value) - float(b.value), True))
        else:
            why = "; ".join(f"{n}: {r.reason or REASON_NOT_ELIGIBLE}" for n, r in (("arm", a), ("anchor", b)) if not r.eligible)
            recs.append(Record(s, None, False, why))
    by_seed = {r.seed: r for r in recs}
    per_seed = {f"s{s}": (float(by_seed[s].value) if by_seed[s].eligible else None) for s in SEEDS}
    try:
        trip = require_triplet(recs)
    except TripletUnavailable as exc:
        return {"per_seed": per_seed, "mean": None, "sd": None, "range": None, "min": None, "max": None,
                "status": str(exc), "missing": [{"seed": s, "reason": r} for s, r in exc.missing]}
    sp = spread(trip.as_list())
    return {"per_seed": per_seed, "mean": float(sp["mean"]), "sd": float(sp["sd"]), "range": float(sp["range"]),
            "min": float(sp["min"]), "max": float(sp["max"]), "status": "ok", "missing": []}


def arm_seed_summary(readings: Mapping[int, Record], per_seed_se: Mapping[int, float | None]) -> dict[str, Any]:
    """Per-arm aggregate over the seeds: coverage (how many seeds carry a reading), the triplet-gated mean / SD / range,
    and :func:`seed_spread_ratio`. An incomplete triplet yields ``mean = None`` (never a mean over survivors) with the
    gate's message as ``status`` and the coverage beside it."""
    readings = _by_seed(readings, "readings")
    recs = [readings[s] if s in readings else Record(s, None, False, REASON_NO_STORE) for s in SEEDS]
    present = [s for s in SEEDS if s in readings and readings[s].eligible]
    out: dict[str, Any] = {"coverage": f"{len(present)}/{len(SEEDS)}", "n_seeds": len(present),
                           "values": {f"s{s}": (float(readings[s].value) if s in readings and readings[s].eligible else None)
                                      for s in SEEDS},
                           "missing": [{"seed": r.seed, "reason": r.reason or REASON_NOT_ELIGIBLE} for r in recs if not r.eligible]}
    try:
        trip = require_triplet(recs)
    except TripletUnavailable as exc:
        out.update(mean=None, sd=None, range=None, min=None, max=None, status=str(exc))
        out["seed_spread"] = seed_spread_ratio(None, per_seed_se)
        return out
    sp = spread(trip.as_list())
    out.update(mean=float(sp["mean"]), sd=float(sp["sd"]), range=float(sp["range"]), min=float(sp["min"]),
               max=float(sp["max"]), status="ok")
    out["seed_spread"] = seed_spread_ratio(trip.values, per_seed_se)
    return out


if __name__ == "__main__":
    sys.exit(main())
