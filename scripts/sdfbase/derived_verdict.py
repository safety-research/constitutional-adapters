#!/usr/bin/env python
"""Derived-object verdict: the pre-registered equivalence tests.

One derived object (V, P_resid, or LoRA-sub) is judged against the three
predictions the decomposition makes, using ONLY frozen per-item / per-doc
score files — the object's, its MIX parent's, and its CTL parent's. Every
margin below is fixed here, once — before any derived object is scored —
so no analysis notebook can apply a threshold ad hoc.

Predictions, each tested **per lane against that lane's own
estimand**, never a pooled number:

* **P1 — keeps the values-specific margin.** Per in-coverage probe item,
  D1 = m_obj − (m_mix − m_ctl) in nats/tok (``margin_mean``). Pass iff TOST:
  the 90% paired-bootstrap CI of mean D1 lies within (−0.015, +0.015) AND
  m_obj's own 95% CI excludes 0.
* **P2 — loses the register margin.** Remnant r = mean(D1) / mean(m_ctl),
  bootstrapped jointly (both means recomputed per resample). Pass iff the 90%
  CI upper bound ≤ 0.25. Undefined when the control's own margin is not
  resolved (mean ≤ 0 or its CI includes 0): a share of nothing is not a
  number.
* **P3 — loses the control-corpus CE gain.** On control-register held-out
  docs, gain_x = 1 − Σ nll_active_x / Σ nll_noop (sum/sum, micro-averaged,
  from ``ce_score``'s per-doc records). Pass iff the 90% doc-level bootstrap
  CI of gain_obj lies within (−0.02, +0.02) AND its upper bound ≤ 0.25 ×
  min(gain_mix, gain_ctl). Power caveat stated when n_docs < 30 (the pool is
  ~20 docs). The three P3 inputs are the derived app's COMBINED ce_score
  files (``ce_score/<model>/<cell>__<pool>__scale<s>.json``: aggregate +
  provenance + the per-doc ``docs``), and they are placed on the pool
  BEFORE any statistic: all three must record
  ``pool == "control-heldout"`` and one ``pairs_sha256``, that sha and the
  exact doc set must match the REGISTERED pool manifest
  (``pairs_constv2ctl-heldout.manifest.json`` under ``--pools-dir`` — the
  analysis' CE_POOL_MANIFEST mapping; a score file never defines a pool),
  and the parents must be scale-1 rows. A bare per-doc JSONL is refused.

Per prediction a ``status`` ∈ {pass, fail, inconclusive, undefined} — "fail"
means the CI clearly excludes the equivalence region, "inconclusive" that it
straddles a boundary. Per row a ``verdict`` drawn from the fixed
vocabulary: confirmed / falsified / not recovered under this trainer +
budget / descriptive-only. The word "inexpressible" never appears — the
output is checked and refused if it does.

Structural caveats the row carries:

* ``--probe-descriptive`` (gemma): P1/P2 are probe-based and become
  descriptive; only P3 is confirmatory, and the row verdict is computed from
  P3 alone.
* A P_resid ``calibration.json`` whose recovery gate did not pass
  (``recovery.p1_p3_rows_are == "descriptive"``) makes the row
  "not recovered under this trainer + budget", carrying the gate's
  per-stratum sentences verbatim; a gate that ruled every stratum
  ``unresolvable`` (the degenerate branch — parents functionally identical)
  makes the row "descriptive-only" instead, because "not recovered" would
  claim a failure that was never tested.

Unregistered readouts (item-level correlation of m_obj with the per-item
subtraction, |m_obj − m_mix|, the T_arith inconsistency passed through from
the calibration) are reported under ``diagnostics`` and labeled so.

All intervals are percentile bootstrap intervals from
``numpy.random.default_rng(seed)`` — also m_obj's own 95% CI, chosen over the
normal approximation so every interval in a row comes from one machinery
with one seed. Pure python + numpy; no torch, no model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Pre-registered constants (do not edit without revising the pre-registration)
# ---------------------------------------------------------------------------

#: P1 TOST equivalence half-width, nats/tok: (−0.015, +0.015) nats/tok
#: ≈ 1.5× the round-1 paired-CI half-width at the 100k estimand — the
#: finest equivalence n = 377 resolves.
P1_TOST_MARGIN_NATS = 0.015
#: Level of the paired-bootstrap CI on mean D1 in P1: the 90% CI.
P1_TOST_CI_LEVEL = 0.90
#: Level of the object's own margin CI in P1: m_obj's own 95% CI
#: excludes 0.
P1_OWN_MARGIN_CI_LEVEL = 0.95

#: P2 register-remnant cap: 90% CI upper bound of r ≤ 0.25.
P2_REMNANT_MAX = 0.25
P2_CI_LEVEL = 0.90

#: P3 CE-gain equivalence half-width: within (−0.02, +0.02).
P3_GAIN_MARGIN = 0.02
#: P3 share cap: upper bound ≤ this × min(parents' gains).
P3_PARENT_SHARE_MAX = 0.25
P3_CI_LEVEL = 0.90
#: Below this many held-out docs the row states the P3 power caveat.
P3_POWER_CAVEAT_MIN_DOCS = 30

#: Bootstrap draws — 10k over the 377 in-coverage items, 1,000 resamples
#: for the doc-level bootstrap — and the fixed seed.
N_BOOT_ITEMS = 10_000
N_BOOT_DOCS = 1_000
BOOT_SEED = 0

#: Row verdict vocabulary — nothing else may appear.
VERDICT_CONFIRMED = "confirmed"
VERDICT_FALSIFIED = "falsified"
VERDICT_NOT_RECOVERED = "not recovered under this trainer + budget"
VERDICT_DESCRIPTIVE = "descriptive-only"
VERDICTS = (VERDICT_CONFIRMED, VERDICT_FALSIFIED, VERDICT_NOT_RECOVERED,
            VERDICT_DESCRIPTIVE)
#: Per-prediction status vocabulary (intermediate; never a row verdict).
STATUSES = ("pass", "fail", "inconclusive", "undefined")
#: No capacity claim is ever made.
BANNED_WORDS = ("inexpressible",)

#: The three ce_score files score the SAME bare model on the SAME docs, so
#: their ``nll_noop_sum`` must agree per doc up to numeric noise (bf16 and
#: micro-batch composition). Relative tolerance on the per-doc sum;
#: overridable from the CLI, recorded in the output.
NOOP_AGREEMENT_RTOL = 1e-3
NOOP_AGREEMENT_ATOL = 1e-6

#: P3 is defined on the control-register held-out pool — the analysis' pool
#: KEY (analysis_r2/r2_paths.py CE_POOLS) and the file STEM of its REGISTERED
#: manifest (r2_paths.CE_POOL_MANIFEST: ``pairs_<stem>.manifest.json`` under
#: the staged pools dir), the ONLY source of the pool's doc set and pairs
#: sha. The two names differ on purpose: the analysis names the pool by what
#: it is, the pool builder named the file by the cell it is the remainder of.
P3_POOL = "control-heldout"
P3_POOL_MANIFEST_STEM = "constv2ctl-heldout"
DEFAULT_POOLS_DIR = "$STAGE/sdfbase/pools"   # under the staging directory; expanded through sisteer.site where it is read
#: The identity a combined ce_score file carries and the P3 read records.
CE_META_KEYS = ("pool", "pairs_sha256", "object_sha256", "scale", "cell",
                "model", "revision")

SCHEMA = "sdfbase.derived_verdict.v1"
P1_NAME = "keeps the values-specific margin"
P2_NAME = "loses the register margin"
P3_NAME = "loses the control-corpus CE gain"


class InputError(ValueError):
    """The inputs are not a valid lane (id mismatch, schema, tolerance);
    the CLI exits 2 on this."""


class BannedWordError(ValueError):
    """The rendered output carried a banned verdict word."""


def assert_no_banned_words(text: str, *, where: str = "output") -> None:
    low = text.lower()
    for w in BANNED_WORDS:
        if w in low:
            raise BannedWordError(
                f"{where} contains the banned word {w!r} — the plan makes no "
                f"capacity claim; say {VERDICT_NOT_RECOVERED!r} with the "
                "numbers instead")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def read_jsonl(path) -> list[dict]:
    """Non-blank JSONL lines as dicts."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise InputError(f"cannot read {p}: {e}") from e
    rows = []
    for ln, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise InputError(f"{p}:{ln}: not JSON ({e.msg})") from e
    return rows


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ce_json(path) -> tuple[list[dict], dict]:
    """The derived app's combined ce_score file (``…__<pool>__scale<s>.json``):
    (the per-doc ``docs`` records, the identity block — :data:`CE_META_KEYS`).
    A per-doc JSONL is REFUSED: it carries no pool identity, so nothing
    could prove it scored the control held-out pool."""
    p = Path(path)
    if p.suffix != ".json":
        raise InputError(
            f"{p}: P3 inputs must be ce_score's combined JSON "
            "(…__<pool>__scale<s>.json, pool + pairs_sha256 embedded) — a "
            "bare per-doc JSONL carries no pool identity and cannot be "
            "validated against the registered pool manifest")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise InputError(f"cannot read {p}: {e}") from e
    except json.JSONDecodeError as e:
        raise InputError(f"{p}: not JSON ({e.msg})") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("docs"), list):
        raise InputError(f"{p}: not a combined ce_score file (no docs list)")
    meta = {k: doc.get(k) for k in CE_META_KEYS}
    if not meta["pool"] or not meta["pairs_sha256"]:
        raise InputError(f"{p}: records no pool / pairs_sha256 — cannot be "
                         "placed on a pool")
    return doc["docs"], meta


def _site_path(value, key: str) -> Path:
    """``value`` with its site placeholder expanded through the site config (``$STAGE``: the staging directory;
    scripts/site_paths.py) and ``~`` expanded; an unset placeholder refuses by name."""
    scripts = str(Path(__file__).resolve().parents[1])
    if scripts not in sys.path:
        sys.path.append(scripts)
    from site_paths import site_path
    return site_path(value, key)


def load_pool_manifest(pools_dir, stem: str = P3_POOL_MANIFEST_STEM) -> dict:
    """The REGISTERED pool manifest ``pairs_<stem>.manifest.json`` — the one
    source of the pool's ``pool_sha256`` and ordered doc ids. Absent ->
    P3 is PENDING (InputError; never seeded from a score file); malformed
    -> refused."""
    path = _site_path(pools_dir, "--pools-dir") / f"pairs_{stem}.manifest.json"
    if not path.is_file():
        raise InputError(
            f"registered pool manifest {path} missing — P3 is PENDING: a "
            "score file never defines the pool's doc set or sha (stage the "
            "manifest, or point --pools-dir at it)")
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise InputError(f"registered pool manifest {path} unreadable: {e}"
                         ) from e
    sha = m.get("pool_sha256") if isinstance(m, dict) else None
    ids = m.get("doc_ids_in_order") if isinstance(m, dict) else None
    if not sha or not isinstance(ids, list) or not ids:
        raise InputError(f"{path}: malformed registered pool manifest (needs "
                         "pool_sha256 + a non-empty doc_ids_in_order)")
    ids = [str(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise InputError(f"{path}: doc_ids_in_order repeats an id")
    if m.get("n_docs") is not None and int(m["n_docs"]) != len(ids):
        raise InputError(f"{path}: n_docs {m['n_docs']} != "
                         f"{len(ids)} doc ids listed")
    return {"path": str(path), "sha256": _sha256(path), "stem": stem,
            "cell": m.get("cell"), "pool_sha256": str(sha), "doc_ids": ids,
            "n_docs": len(ids)}


def validate_p3_inputs(ces: dict, manifest: dict, *,
                       pool: str = P3_POOL) -> dict:
    """Place the three P3 inputs (``{role: (docs, meta)}`` for object_ce /
    mix_ce / ctl_ce) on the registered pool BEFORE any statistic: every
    file scored ``pool`` (:data:`P3_POOL` — the 100k pair's control
    held-out — unless the lane names another control-register held-out,
    e.g. ``constv2ctl2-heldout`` for full − full); all cite ONE
    ``pairs_sha256`` and it is the manifest's ``pool_sha256``; each covers
    EXACTLY the manifest's doc set (a 19/20 file is refused, never
    averaged); the parents are scale-1 rows (the lane's estimand is the
    parents at scale 1 — the object may be a 2x row, recorded). Returns
    the pool block the verdict records."""
    for k, (_docs, meta) in ces.items():
        if meta.get("pool") != pool:
            raise InputError(f"{k}: scored pool {meta.get('pool')!r}; P3 is "
                             f"defined on {pool!r} only")
    shas = {k: str(meta.get("pairs_sha256")) for k, (_d, meta) in ces.items()}
    if len(set(shas.values())) != 1:
        raise InputError(f"P3 inputs cite different pairs_sha256: {shas} — "
                         "not the same pool file; refusing")
    sha = next(iter(shas.values()))
    if sha != manifest["pool_sha256"]:
        raise InputError(
            f"P3 inputs cite pairs_sha256 {sha[:12]}… but the registered "
            f"manifest {manifest['path']} records "
            f"{manifest['pool_sha256'][:12]}… — not the registered pool")
    want = set(manifest["doc_ids"])
    for k, (docs, _meta) in ces.items():
        got = [str(d.get("doc_id")) for d in docs]
        if len(set(got)) != len(got):
            raise InputError(f"{k}: duplicate doc_id")
        if set(got) != want:
            raise InputError(
                f"{k}: doc set differs from the registered pool ({len(got)} "
                f"vs {len(want)} docs; missing {sorted(want - set(got))[:5]}, "
                f"extra {sorted(set(got) - want)[:5]}) — refusing")
    for k in ("mix_ce", "ctl_ce"):
        sc = ces[k][1].get("scale")
        if sc is None or float(sc) != 1.0:
            raise InputError(f"{k}: parent scored at scale {sc!r}; P3's "
                             "estimand is the parents at scale 1.0")
    return {"pool": pool, "pairs_sha256": sha,
            "manifest": {kk: manifest[kk] for kk in
                         ("path", "sha256", "stem", "cell", "n_docs")},
            "object_scale": ces["object_ce"][1].get("scale"),
            "object_sha256": {k: ces[k][1].get("object_sha256") for k in ces},
            "cells": {k: ces[k][1].get("cell") for k in ces}}


def _pct_ci(samples: np.ndarray, level: float) -> list[float]:
    """Percentile bootstrap interval at ``level`` (two-sided)."""
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(samples, dtype=float), [a, 1.0 - a])
    return [float(lo), float(hi)]


def _fmt(x, nd=4) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


# ---------------------------------------------------------------------------
# Pairing (probe items; ce docs)
# ---------------------------------------------------------------------------


def _by_id(rows: list[dict], key: str, what: str) -> dict:
    out: dict = {}
    for r in rows:
        if key not in r:
            raise InputError(f"{what}: a row lacks {key!r}")
        i = r[key]
        if i in out:
            raise InputError(f"{what}: duplicate {key} {i!r}")
        out[i] = r
    return out


def _refuse_id_mismatch(sets: dict[str, set], key: str) -> None:
    names = list(sets)
    base = sets[names[0]]
    for n in names[1:]:
        if sets[n] != base:
            only_a = sorted(base - sets[n], key=str)[:5]
            only_b = sorted(sets[n] - base, key=str)[:5]
            raise InputError(
                f"{key} sets differ between {names[0]} and {n}: "
                f"{len(base)} vs {len(sets[n])} rows; only in {names[0]}: "
                f"{only_a}; only in {n}: {only_b} — the three files must "
                "score the same frozen set; refusing")


def paired_items(obj_rows: list[dict], mix_rows: list[dict],
                 ctl_rows: list[dict]) -> list[dict]:
    """Pair the three probe score files by ``id`` and keep the in-coverage
    items (377 of them). Refuses if the id sets differ, an id repeats, or
    an item's ``source``/``coverage`` disagree across files (those are
    properties of the frozen probe set, so a disagreement means the files
    were not scored on the same set).

    Each returned item carries ``m_obj``/``m_mix``/``m_ctl`` (``margin_mean``,
    nats/tok, the PRIMARY statistic), the per-item subtraction
    ``sub = m_mix − m_ctl`` (the lane's estimand) and ``d1 = m_obj − sub``."""
    o, m, c = (_by_id(obj_rows, "id", "object probe"),
               _by_id(mix_rows, "id", "mix probe"),
               _by_id(ctl_rows, "id", "ctl probe"))
    _refuse_id_mismatch({"object": set(o), "mix": set(m), "ctl": set(c)},
                        "item id")
    items = []
    for i in sorted(o, key=str):
        ro, rm, rc = o[i], m[i], c[i]
        for fld in ("source", "coverage"):
            vals = {ro.get(fld), rm.get(fld), rc.get(fld)}
            if len(vals) != 1:
                raise InputError(
                    f"item {i!r}: {fld} disagrees across files "
                    f"({ro.get(fld)!r} / {rm.get(fld)!r} / {rc.get(fld)!r}) "
                    "— not the same frozen probe set; refusing")
        if ro.get("coverage") != "in":
            continue
        try:
            mo, mm, mc = (float(ro["margin_mean"]), float(rm["margin_mean"]),
                          float(rc["margin_mean"]))
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"item {i!r}: margin_mean missing or not "
                             f"numeric ({e})") from e
        if not all(math.isfinite(x) for x in (mo, mm, mc)):
            raise InputError(f"item {i!r}: non-finite margin_mean")
        sub = mm - mc
        items.append({"id": i, "source": ro.get("source"),
                      "section_id": ro.get("section_id"),
                      "m_obj": mo, "m_mix": mm, "m_ctl": mc,
                      "sub": sub, "d1": mo - sub})
    if len(items) < 2:
        raise InputError(f"only {len(items)} in-coverage paired items — "
                         "nothing to bootstrap")
    return items


def paired_docs(obj_docs: list[dict], mix_docs: list[dict],
                ctl_docs: list[dict], *,
                noop_rtol: float = NOOP_AGREEMENT_RTOL,
                noop_atol: float = NOOP_AGREEMENT_ATOL) -> list[dict]:
    """Pair the three ``ce_score`` per-doc files by ``doc_id``. Refuses on
    id-set mismatch, duplicate ids, differing ``n_target_tokens``, or a
    per-doc ``nll_noop_sum`` that disagrees beyond tolerance — the three
    files score the same bare model on the same docs, so the no-op leg is
    the free consistency check."""
    o, m, c = (_by_id(obj_docs, "doc_id", "object ce"),
               _by_id(mix_docs, "doc_id", "mix ce"),
               _by_id(ctl_docs, "doc_id", "ctl ce"))
    _refuse_id_mismatch({"object": set(o), "mix": set(m), "ctl": set(c)},
                        "doc_id")
    docs = []
    for i in sorted(o, key=str):
        ro, rm, rc = o[i], m[i], c[i]
        try:
            ntok = {int(ro["n_target_tokens"]), int(rm["n_target_tokens"]),
                    int(rc["n_target_tokens"])}
            noop = [float(ro["nll_noop_sum"]), float(rm["nll_noop_sum"]),
                    float(rc["nll_noop_sum"])]
            act = [float(ro["nll_active_sum"]), float(rm["nll_active_sum"]),
                   float(rc["nll_active_sum"])]
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"doc {i!r}: ce record field missing or not "
                             f"numeric ({e})") from e
        if len(ntok) != 1:
            raise InputError(f"doc {i!r}: n_target_tokens disagrees across "
                             f"files ({sorted(ntok)}) — not the same docs; "
                             "refusing")
        if not all(math.isfinite(x) for x in noop + act):
            raise InputError(f"doc {i!r}: non-finite nll sum")
        ref = noop[0]
        for name, val in (("mix", noop[1]), ("ctl", noop[2])):
            if abs(val - ref) > noop_atol + noop_rtol * abs(ref):
                raise InputError(
                    f"doc {i!r}: nll_noop_sum disagrees between the object "
                    f"file ({ref:.6g}) and the {name} file ({val:.6g}); "
                    f"tolerance rtol={noop_rtol:g} atol={noop_atol:g} — the "
                    "bare legs must agree (same model, same docs); refusing")
        docs.append({"doc_id": i, "family": ro.get("family"),
                     "source_tag": ro.get("source_tag"),
                     "n_target_tokens": ntok.pop(),
                     "noop_obj": noop[0], "noop_mix": noop[1],
                     "noop_ctl": noop[2],
                     "active_obj": act[0], "active_mix": act[1],
                     "active_ctl": act[2]})
    if len(docs) < 2:
        raise InputError(f"only {len(docs)} paired docs — nothing to "
                         "bootstrap")
    return docs


# ---------------------------------------------------------------------------
# Bootstrap machinery (one rng, one index matrix per call)
# ---------------------------------------------------------------------------


def _item_arrays(items: list[dict]):
    return (np.array([it["m_obj"] for it in items], dtype=float),
            np.array([it["m_mix"] for it in items], dtype=float),
            np.array([it["m_ctl"] for it in items], dtype=float))


def _item_boot_means(obj: np.ndarray, mix: np.ndarray, ctl: np.ndarray,
                     n_boot: int, seed: int):
    """Resample items with replacement; return the per-draw means of
    (m_obj, m_mix, m_ctl) so every derived quantity — D1, the estimand, the
    remnant — is computed on the SAME resample (paired/joint bootstrap).
    Takes the :func:`_item_arrays` the caller already holds."""
    if n_boot < 1:
        raise InputError("n_boot must be ≥ 1")
    n = len(obj)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    return obj[idx].mean(axis=1), mix[idx].mean(axis=1), ctl[idx].mean(axis=1)


# ---------------------------------------------------------------------------
# P1 — keeps the values-specific margin
# ---------------------------------------------------------------------------


def p1_tost(items: list[dict], *, n_boot: int = N_BOOT_ITEMS,
            seed: int = BOOT_SEED, margin: float = P1_TOST_MARGIN_NATS,
            ci_level: float = P1_TOST_CI_LEVEL,
            own_ci_level: float = P1_OWN_MARGIN_CI_LEVEL) -> dict:
    """Prediction P1. D1 = m_obj − (m_mix − m_ctl) per item. Pass iff the
    ``ci_level`` paired-bootstrap CI of mean D1 lies within (−margin,
    +margin) AND the ``own_ci_level`` CI of mean m_obj excludes 0. Fail iff
    the D1 CI lies entirely outside the equivalence region; otherwise
    inconclusive. The lane's estimand mean(m_mix − m_ctl) is reported with
    its own CI — the test is against THAT, not a pooled number."""
    obj, mix, ctl = _item_arrays(items)
    ob, mb, cb = _item_boot_means(obj, mix, ctl, n_boot, seed)
    d1_b = ob - (mb - cb)
    est_b = mb - cb
    d1_ci = _pct_ci(d1_b, ci_level)
    obj_ci = _pct_ci(ob, own_ci_level)
    mean_d1 = float((obj - (mix - ctl)).mean())
    mean_obj = float(obj.mean())
    within = (d1_ci[0] > -margin) and (d1_ci[1] < margin)
    outside = (d1_ci[0] >= margin) or (d1_ci[1] <= -margin)
    own_excl0 = (obj_ci[0] > 0.0) or (obj_ci[1] < 0.0)
    if within and own_excl0:
        status, reason = "pass", (
            f"{int(ci_level * 100)}% CI of mean D1 within ±{margin:g} and "
            f"m_obj's {int(own_ci_level * 100)}% CI excludes 0")
    elif outside:
        status, reason = "fail", (
            f"{int(ci_level * 100)}% CI of mean D1 lies entirely outside "
            f"±{margin:g}: the object does not carry the lane's estimand")
    elif within:
        status, reason = "inconclusive", (
            f"D1 equivalent within ±{margin:g} but m_obj's "
            f"{int(own_ci_level * 100)}% CI includes 0 — the object carries "
            "no resolved margin of its own")
    else:
        status, reason = "inconclusive", (
            f"{int(ci_level * 100)}% CI of mean D1 straddles ±{margin:g}")
    return {
        "prediction": "P1", "name": P1_NAME,
        "n_items": len(items), "n_boot": int(n_boot), "seed": int(seed),
        "margin_nats": margin, "tost_ci_level": ci_level,
        "own_margin_ci_level": own_ci_level,
        "mean_d1": mean_d1, "d1_ci": d1_ci,
        "mean_m_obj": mean_obj, "m_obj_ci": obj_ci,
        "estimand_mean": float((mix - ctl).mean()),
        "estimand_ci": _pct_ci(est_b, own_ci_level),
        "estimand_ci_level": own_ci_level,
        "mean_m_mix": float(mix.mean()), "mean_m_ctl": float(ctl.mean()),
        "tost_within_margin": bool(within),
        "own_margin_excludes_zero": bool(own_excl0),
        "status": status, "reason": reason,
    }


# ---------------------------------------------------------------------------
# P2 — loses the register margin
# ---------------------------------------------------------------------------


def p2_remnant(items: list[dict], *, n_boot: int = N_BOOT_ITEMS,
               seed: int = BOOT_SEED, remnant_max: float = P2_REMNANT_MAX,
               ci_level: float = P2_CI_LEVEL) -> dict:
    """Prediction P2. r = mean / mean(m_ctl), bootstrapped jointly
    (both means recomputed per resample). Pass iff the ``ci_level`` CI
    upper bound ≤ ``remnant_max``; fail iff the lower bound exceeds it;
    undefined when mean(m_ctl) ≤ 0 or its CI includes 0."""
    obj, mix, ctl = _item_arrays(items)
    ob, mb, cb = _item_boot_means(obj, mix, ctl, n_boot, seed)
    d1_b = ob - (mb - cb)
    mean_d1 = float((obj - (mix - ctl)).mean())
    mean_ctl = float(ctl.mean())
    ctl_ci = _pct_ci(cb, ci_level)
    out = {
        "prediction": "P2", "name": P2_NAME,
        "n_items": len(items), "n_boot": int(n_boot), "seed": int(seed),
        "remnant_max": remnant_max, "ci_level": ci_level,
        "mean_d1": mean_d1, "mean_m_ctl": mean_ctl, "m_ctl_ci": ctl_ci,
        "remnant": None, "remnant_ci": None,
    }
    if mean_ctl <= 0.0 or ctl_ci[0] <= 0.0:
        out["status"] = "undefined"
        out["reason"] = ("undefined (control margin not resolved): "
                         f"mean m_ctl = {mean_ctl:+.4f}, "
                         f"{int(ci_level * 100)}% CI "
                         f"[{ctl_ci[0]:+.4f}, {ctl_ci[1]:+.4f}] does not "
                         "exclude 0, so a share of it is not a number")
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        r_b = d1_b / cb
    r_ci = _pct_ci(r_b, ci_level)
    out["remnant"] = mean_d1 / mean_ctl
    out["remnant_ci"] = r_ci
    if r_ci[1] <= remnant_max:
        out["status"] = "pass"
        out["reason"] = (f"{int(ci_level * 100)}% CI upper bound "
                         f"{r_ci[1]:.3f} ≤ {remnant_max:g}")
    elif r_ci[0] > remnant_max:
        out["status"] = "fail"
        out["reason"] = (f"{int(ci_level * 100)}% CI lower bound "
                         f"{r_ci[0]:.3f} > {remnant_max:g}: the object "
                         "retains the control's register margin")
    else:
        out["status"] = "inconclusive"
        out["reason"] = (f"{int(ci_level * 100)}% CI "
                         f"[{r_ci[0]:.3f}, {r_ci[1]:.3f}] straddles "
                         f"{remnant_max:g}")
    return out


# ---------------------------------------------------------------------------
# P3 — loses the control-corpus CE gain
# ---------------------------------------------------------------------------


def _gain(active: np.ndarray, noop: np.ndarray) -> float:
    return float(1.0 - active.sum() / noop.sum())


def p3_ce_gain(obj_docs: list[dict], mix_docs: list[dict],
               ctl_docs: list[dict], *, n_boot: int = N_BOOT_DOCS,
               seed: int = BOOT_SEED, gain_margin: float = P3_GAIN_MARGIN,
               parent_share_max: float = P3_PARENT_SHARE_MAX,
               ci_level: float = P3_CI_LEVEL,
               power_caveat_min_docs: int = P3_POWER_CAVEAT_MIN_DOCS,
               noop_rtol: float = NOOP_AGREEMENT_RTOL,
               noop_atol: float = NOOP_AGREEMENT_ATOL) -> dict:
    """Prediction P3 on the control held-out pool. gain_x = 1 − Σ
    nll_active_x / Σ nll_noop over docs (sum/sum, each file's own no-op
    leg, which :func:`paired_docs` has verified agree). Doc-level bootstrap
    (docs resampled with replacement, all three gains recomputed on the
    same resample). Pass iff the ``ci_level`` CI of gain_obj lies within
    (−gain_margin, +gain_margin) AND its upper bound ≤ parent_share_max ×
    min(gain_mix, gain_ctl). Fail iff the CI lies entirely outside that
    region. Undefined when the parents show no gain to lose (min parent
    gain ≤ 0)."""
    docs = paired_docs(obj_docs, mix_docs, ctl_docs, noop_rtol=noop_rtol,
                       noop_atol=noop_atol)
    if n_boot < 1:
        raise InputError("n_boot must be ≥ 1")
    n = len(docs)
    arr = {k: np.array([d[k] for d in docs], dtype=float)
           for k in ("noop_obj", "noop_mix", "noop_ctl",
                     "active_obj", "active_mix", "active_ctl")}
    gains = {x: _gain(arr[f"active_{x}"], arr[f"noop_{x}"])
             for x in ("obj", "mix", "ctl")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    gain_ci = {}
    for x in ("obj", "mix", "ctl"):
        a = arr[f"active_{x}"][idx].sum(axis=1)
        z = arr[f"noop_{x}"][idx].sum(axis=1)
        gain_ci[x] = _pct_ci(1.0 - a / z, ci_level)
    min_parent = min(gains["mix"], gains["ctl"])
    share_cap = parent_share_max * min_parent
    lo, hi = gain_ci["obj"]
    within = (lo > -gain_margin) and (hi < gain_margin)
    under_cap = hi <= share_cap
    out = {
        "prediction": "P3", "name": P3_NAME,
        "n_docs": n,
        "n_target_tokens": int(sum(d["n_target_tokens"] for d in docs)),
        "n_boot": int(n_boot), "seed": int(seed),
        "gain_margin": gain_margin, "parent_share_max": parent_share_max,
        "ci_level": ci_level,
        "gain": gains, "gain_ci": gain_ci,
        "min_parent_gain": min_parent, "share_cap": share_cap,
        "within_margin": bool(within), "under_share_cap": bool(under_cap),
        "power_caveat": (
            f"n_docs = {n} < {power_caveat_min_docs}: the doc-level "
            f"bootstrap is under-powered; equivalence at ±{gain_margin:g} "
            "may be unresolvable on this pool"
            if n < power_caveat_min_docs else None),
        "noop_agreement": {"rtol": noop_rtol, "atol": noop_atol},
    }
    pct = int(ci_level * 100)
    if min_parent <= 0.0:
        out["status"] = "undefined"
        out["reason"] = ("undefined (parents show no control-corpus gain): "
                         f"min(gain_mix, gain_ctl) = {min_parent:+.4f} ≤ 0, "
                         "so there is no gain to lose")
    elif within and under_cap:
        out["status"] = "pass"
        out["reason"] = (f"{pct}% CI of gain_obj within ±{gain_margin:g} "
                         f"and upper bound {hi:+.4f} ≤ {share_cap:+.4f} "
                         f"(= {parent_share_max:g} × min parent gain "
                         f"{min_parent:+.4f})")
    elif lo >= gain_margin or hi <= -gain_margin or lo > share_cap:
        out["status"] = "fail"
        out["reason"] = (f"{pct}% CI of gain_obj [{lo:+.4f}, {hi:+.4f}] lies "
                         f"entirely outside the equivalence region "
                         f"(±{gain_margin:g} and ≤ {share_cap:+.4f}): the "
                         "object retains the control-corpus gain")
    else:
        out["status"] = "inconclusive"
        out["reason"] = (f"{pct}% CI of gain_obj [{lo:+.4f}, {hi:+.4f}] "
                         f"straddles the equivalence boundary "
                         f"(±{gain_margin:g} / cap {share_cap:+.4f})")
    return out


# ---------------------------------------------------------------------------
# Unregistered diagnostics (reported, labeled, never a gate)
# ---------------------------------------------------------------------------


def diagnostics(items: list[dict], calibration: dict | None = None) -> dict:
    obj, mix, ctl = _item_arrays(items)
    sub = mix - ctl
    r = None
    if len(items) > 1:
        so, ss = float(obj.std()), float(sub.std())
        if so > 0 and ss > 0:
            r = float(np.corrcoef(obj, sub)[0, 1])
    out = {
        "unregistered": True,
        "note": ("plan §2.6: item-level readouts reported as diagnostics "
                 "and labeled so; none is a gate"),
        "pearson_r_m_obj_vs_subtraction": r,
        "mean_abs_m_obj_minus_m_mix": float(np.abs(obj - mix).mean()),
        "mean_m_obj_minus_m_mix": float((obj - mix).mean()),
        "arithmetic_residual_inconsistency": None,
    }
    if calibration:
        blk = calibration.get("arithmetic_residual_inconsistency")
        if blk is None and isinstance(calibration.get("diagnostics"), dict):
            blk = calibration["diagnostics"].get(
                "arithmetic_residual_inconsistency")
        out["arithmetic_residual_inconsistency"] = blk
    return out


# ---------------------------------------------------------------------------
# Row verdict
# ---------------------------------------------------------------------------


def _recovery_block(calibration: dict | None) -> dict | None:
    if calibration is None:
        return None
    rec = calibration.get("recovery")
    if not isinstance(rec, dict) or "verdict" not in rec:
        raise InputError("calibration.json lacks a 'recovery' block with a "
                         "'verdict' — not a residual trainer calibration")
    rows_are = rec.get("p1_p3_rows_are")
    if rows_are not in ("confirmatory", "descriptive"):
        raise InputError(f"calibration recovery.p1_p3_rows_are = {rows_are!r};"
                         " expected 'confirmatory' or 'descriptive'")
    per = rec.get("per_stratum") or {}
    return {
        "verdict": rec["verdict"],
        "p1_p3_rows_are": rows_are,
        "degenerate_parents_identical": bool(
            rec.get("degenerate_parents_identical", False)),
        "per_stratum_sentences": {
            s: (v.get("sentence") if isinstance(v, dict) else None)
            for s, v in per.items()},
        "per_stratum": per,
    }


def row_verdict(p1: dict | None, p2: dict | None, p3: dict | None, *,
                probe_descriptive: bool = False,
                calibration: dict | None = None) -> dict:
    """Fold per-prediction statuses into ONE row verdict ∈ :data:`VERDICTS`.

    * A P_resid calibration whose gate did not pass →
      ``not recovered under this trainer + budget``, all predictions
      descriptive, the gate's per-stratum sentences carried verbatim; a gate
      that was ``unresolvable`` everywhere (degenerate branch) →
      ``descriptive-only`` (nothing was tested, so nothing was "not
      recovered").
    * ``probe_descriptive`` (gemma): P1/P2 descriptive; the row is
      judged from P3 alone.
    * Otherwise: confirmatory predictions with status ≠ undefined are the
      row's basis; ``confirmed`` iff every one passes, ``falsified`` iff any
      fails, else ``descriptive-only`` (an inconclusive leg, or no
      applicable leg at all) with the reason spelled out."""
    preds = {"P1": p1, "P2": p2, "P3": p3}
    roles = {}
    for k, p in preds.items():
        if p is None:
            roles[k] = "absent"
        elif probe_descriptive and k in ("P1", "P2"):
            roles[k] = "descriptive"
        else:
            roles[k] = "confirmatory"
    recovery = _recovery_block(calibration)
    out = {"verdict": None, "reason": None, "basis": [], "roles": roles,
           "statuses": {k: (p["status"] if p else None)
                        for k, p in preds.items()},
           "probe_descriptive": bool(probe_descriptive),
           "recovery_gate": recovery}
    if recovery is not None and recovery["p1_p3_rows_are"] == "descriptive":
        out["roles"] = {k: ("absent" if p is None else "descriptive")
                        for k, p in preds.items()}
        sentences = [s for s in recovery["per_stratum_sentences"].values()
                     if s]
        if recovery["verdict"] == "unresolvable":
            out["verdict"] = VERDICT_DESCRIPTIVE
            out["reason"] = (
                "recovery gate unresolvable in every stratum (the parents "
                "are functionally identical on the capture distribution — "
                "the decomposition collapses); P1–P3 rows are descriptive. "
                + " | ".join(sentences))
        else:
            out["verdict"] = VERDICT_NOT_RECOVERED
            out["reason"] = (
                f"recovery gate: {recovery['verdict']}; P1–P3 rows are "
                "descriptive. " + " | ".join(sentences))
        return out
    applicable = [k for k, p in preds.items()
                  if p is not None and out["roles"][k] == "confirmatory"
                  and p["status"] != "undefined"]
    out["basis"] = applicable
    undefined = [k for k, p in preds.items()
                 if p is not None and out["roles"][k] == "confirmatory"
                 and p["status"] == "undefined"]
    notes = []
    if probe_descriptive:
        notes.append("P1/P2 probe legs descriptive (gemma D21 caveat); "
                     "row judged from P3 alone")
    for k in undefined:
        notes.append(f"{k} {preds[k]['reason']}")
    if not applicable:
        out["verdict"] = VERDICT_DESCRIPTIVE
        notes.append("no confirmatory prediction applicable")
    elif any(preds[k]["status"] == "fail" for k in applicable):
        out["verdict"] = VERDICT_FALSIFIED
        notes.append("failed: " + ", ".join(
            k for k in applicable if preds[k]["status"] == "fail"))
    elif all(preds[k]["status"] == "pass" for k in applicable):
        out["verdict"] = VERDICT_CONFIRMED
        notes.append("all applicable predictions pass: "
                     + ", ".join(applicable))
    else:
        out["verdict"] = VERDICT_DESCRIPTIVE
        notes.append("inconclusive: " + "; ".join(
            f"{k} {preds[k]['reason']}" for k in applicable
            if preds[k]["status"] == "inconclusive"))
    out["reason"] = " | ".join(notes)
    assert out["verdict"] in VERDICTS
    return out


# ---------------------------------------------------------------------------
# Whole-lane evaluation
# ---------------------------------------------------------------------------


def evaluate(obj_rows: list[dict], mix_rows: list[dict], ctl_rows: list[dict],
             *, obj_docs: list[dict] | None = None,
             mix_docs: list[dict] | None = None,
             ctl_docs: list[dict] | None = None,
             calibration: dict | None = None,
             probe_descriptive: bool = False, label: str | None = None,
             n_boot_items: int = N_BOOT_ITEMS, n_boot_docs: int = N_BOOT_DOCS,
             seed: int = BOOT_SEED,
             noop_rtol: float = NOOP_AGREEMENT_RTOL,
             inputs: dict | None = None, pool: dict | None = None) -> dict:
    """Run P1/P2 (and P3 when the three ce files are given) on one lane and
    fold them into the row verdict; ``pool`` is the validated P3 pool block
    (:func:`validate_p3_inputs`), recorded as ``p3_pool``. Refuses the
    output if it carries a banned word anywhere (label included)."""
    ce_given = [x is not None for x in (obj_docs, mix_docs, ctl_docs)]
    if any(ce_given) and not all(ce_given):
        raise InputError("P3 needs all three ce files (object, mix, ctl) or "
                         "none")
    items = paired_items(obj_rows, mix_rows, ctl_rows)
    p1 = p1_tost(items, n_boot=n_boot_items, seed=seed)
    p2 = p2_remnant(items, n_boot=n_boot_items, seed=seed)
    p3 = (p3_ce_gain(obj_docs, mix_docs, ctl_docs, n_boot=n_boot_docs,
                     seed=seed, noop_rtol=noop_rtol)
          if all(ce_given) else None)
    row = row_verdict(p1, p2, p3, probe_descriptive=probe_descriptive,
                      calibration=calibration)
    result = {
        "schema": SCHEMA,
        "label": label,
        "constants": {
            "P1_TOST_MARGIN_NATS": P1_TOST_MARGIN_NATS,
            "P1_TOST_CI_LEVEL": P1_TOST_CI_LEVEL,
            "P1_OWN_MARGIN_CI_LEVEL": P1_OWN_MARGIN_CI_LEVEL,
            "P2_REMNANT_MAX": P2_REMNANT_MAX, "P2_CI_LEVEL": P2_CI_LEVEL,
            "P3_GAIN_MARGIN": P3_GAIN_MARGIN,
            "P3_PARENT_SHARE_MAX": P3_PARENT_SHARE_MAX,
            "P3_CI_LEVEL": P3_CI_LEVEL,
            "N_BOOT_ITEMS": N_BOOT_ITEMS, "N_BOOT_DOCS": N_BOOT_DOCS,
            "BOOT_SEED": BOOT_SEED,
            "plan": "~/specs/sdf-base-steering-r2/plan.md §2.6, §4.4",
        },
        "run": {"n_boot_items": int(n_boot_items),
                "n_boot_docs": int(n_boot_docs), "seed": int(seed),
                "bootstrap": "percentile, numpy default_rng(seed); every "
                             "interval in this row (including m_obj's own "
                             "95% CI) is a bootstrap interval"},
        "inputs": inputs or {},
        "p3_pool": pool,
        "n_items_in_coverage": len(items),
        "p1": p1, "p2": p2, "p3": p3,
        "row": row,
        "diagnostics": diagnostics(items, calibration),
    }
    assert_no_banned_words(json.dumps(result), where="derived verdict")
    return result


def summary_lines(result: dict) -> list[str]:
    """One line per prediction plus the row verdict."""
    p1, p2, p3, row = (result["p1"], result["p2"], result["p3"],
                       result["row"])
    roles = row["roles"]
    lines = [
        (f"P1 {P1_NAME} [{roles['P1']}]: mean D1 = {_fmt(p1['mean_d1'])} "
         f"nats/tok, {int(p1['tost_ci_level'] * 100)}% CI "
         f"[{_fmt(p1['d1_ci'][0])}, {_fmt(p1['d1_ci'][1])}] vs "
         f"±{p1['margin_nats']:g}; m_obj = {_fmt(p1['mean_m_obj'])} "
         f"[{_fmt(p1['m_obj_ci'][0])}, {_fmt(p1['m_obj_ci'][1])}]; "
         f"estimand = {_fmt(p1['estimand_mean'])} "
         f"[{_fmt(p1['estimand_ci'][0])}, {_fmt(p1['estimand_ci'][1])}] "
         f"(n = {p1['n_items']}) -> {p1['status']}"),
        (f"P2 {P2_NAME} [{roles['P2']}]: remnant r = "
         + ("n/a" if p2["remnant"] is None else f"{p2['remnant']:.3f}")
         + (", CI n/a" if p2["remnant_ci"] is None else
            f", {int(p2['ci_level'] * 100)}% CI [{p2['remnant_ci'][0]:.3f}, "
            f"{p2['remnant_ci'][1]:.3f}]")
         + f" vs cap {p2['remnant_max']:g}; m_ctl = {_fmt(p2['mean_m_ctl'])} "
         f"-> {p2['status']}"),
    ]
    if p3 is None:
        lines.append(f"P3 {P3_NAME} [absent]: no ce files given")
    else:
        lines.append(
            f"P3 {P3_NAME} [{roles['P3']}]: gain_obj = "
            f"{_fmt(p3['gain']['obj'])}, {int(p3['ci_level'] * 100)}% CI "
            f"[{_fmt(p3['gain_ci']['obj'][0])}, "
            f"{_fmt(p3['gain_ci']['obj'][1])}] vs ±{p3['gain_margin']:g} and "
            f"cap {_fmt(p3['share_cap'])} ({p3['parent_share_max']:g} × min "
            f"parent {_fmt(p3['min_parent_gain'])}; gain_mix = "
            f"{_fmt(p3['gain']['mix'])}, gain_ctl = "
            f"{_fmt(p3['gain']['ctl'])}); n_docs = {p3['n_docs']}"
            + (" (power caveat)" if p3["power_caveat"] else "")
            + f" -> {p3['status']}")
    lab = f" {result['label']}" if result.get("label") else ""
    basis = ", ".join(row["basis"]) if row["basis"] else "none"
    lines.append(f"ROW{lab}: {row['verdict']} (basis: {basis}) — "
                 f"{row['reason']}")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=("Derived-object verdict against the pre-registered "
                     "equivalence margins of plan §2.6 (one lane)."))
    ap.add_argument("--object-probe", required=True,
                    help="per-item probe score jsonl of the derived object")
    ap.add_argument("--mix-probe", required=True,
                    help="per-item probe score jsonl of the MIX parent")
    ap.add_argument("--ctl-probe", required=True,
                    help="per-item probe score jsonl of the CTL parent")
    ap.add_argument("--object-ce", help="the derived app's combined ce_score "
                    "JSON of the object on the control held-out pool "
                    "(ce_score/<model>/<id>__control-heldout__scale<s>.json)")
    ap.add_argument("--mix-ce", help="combined ce_score JSON, mix parent, "
                    "same pool, scale 1")
    ap.add_argument("--ctl-ce", help="combined ce_score JSON, ctl parent, "
                    "same pool, scale 1")
    ap.add_argument("--pools-dir", default=str(DEFAULT_POOLS_DIR),
                    help="staged pools dir holding the REGISTERED pool "
                         f"manifest pairs_{P3_POOL_MANIFEST_STEM}.manifest.json "
                         "(or pairs_<--p3-pool-stem>.manifest.json) the three "
                         "ce files are validated against")
    ap.add_argument("--p3-pool", default=P3_POOL,
                    help="the control-register held-out pool KEY the three "
                         f"ce files must have scored (default {P3_POOL!r}, the "
                         "100k pair's; full - full: constv2ctl2-heldout)")
    ap.add_argument("--p3-pool-stem", default=P3_POOL_MANIFEST_STEM,
                    help="file stem of that pool's REGISTERED manifest under "
                         f"--pools-dir (default {P3_POOL_MANIFEST_STEM!r}; "
                         "full - full: constv2ctl2-heldout)")
    ap.add_argument("--calibration", help="P_resid calibration.json "
                    "(recovery gate, plan §2.5)")
    ap.add_argument("--probe-descriptive", action="store_true",
                    help="gemma D21: P1/P2 are descriptive; judge from P3")
    ap.add_argument("--label", default=None, help="row label (free text)")
    ap.add_argument("--n-boot-items", type=int, default=N_BOOT_ITEMS)
    ap.add_argument("--n-boot-docs", type=int, default=N_BOOT_DOCS)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--noop-rtol", type=float, default=NOOP_AGREEMENT_RTOL,
                    help="per-doc relative tolerance for nll_noop_sum "
                    "agreement across the three ce files")
    ap.add_argument("--out", required=True, help="verdict.json path")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ce_paths = (args.object_ce, args.mix_ce, args.ctl_ce)
    try:
        if any(ce_paths) and not all(ce_paths):
            raise InputError("--object-ce/--mix-ce/--ctl-ce must be given "
                             "together (or not at all)")
        inputs = {}

        def load(k, p):
            rows = read_jsonl(p)
            inputs[k] = {"path": str(p), "sha256": _sha256(p),
                         "n_rows": len(rows)}
            return rows

        probes = {k: load(k, p) for k, p in (("object_probe", args.object_probe),
                                             ("mix_probe", args.mix_probe),
                                             ("ctl_probe", args.ctl_probe))}
        ces: dict = {}
        pool = None
        if all(ce_paths):
            loaded = {}
            for k, p in (("object_ce", args.object_ce),
                         ("mix_ce", args.mix_ce), ("ctl_ce", args.ctl_ce)):
                docs, meta = load_ce_json(p)
                inputs[k] = {"path": str(p), "sha256": _sha256(p),
                             "n_rows": len(docs), **meta}
                loaded[k] = (docs, meta)
            # placed on the REGISTERED pool before any statistic
            pool = validate_p3_inputs(
                loaded, load_pool_manifest(args.pools_dir, args.p3_pool_stem),
                pool=args.p3_pool)
            inputs["pool_manifest"] = pool["manifest"]
            ces = {k: v[0] for k, v in loaded.items()}
        calibration = None
        if args.calibration:
            try:
                calibration = json.loads(
                    Path(args.calibration).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise InputError(f"cannot read calibration "
                                 f"{args.calibration}: {e}") from e
            inputs["calibration"] = {"path": str(args.calibration),
                                     "sha256": _sha256(args.calibration)}
        result = evaluate(
            probes["object_probe"], probes["mix_probe"], probes["ctl_probe"],
            obj_docs=ces.get("object_ce"), mix_docs=ces.get("mix_ce"),
            ctl_docs=ces.get("ctl_ce"), calibration=calibration,
            probe_descriptive=args.probe_descriptive, label=args.label,
            n_boot_items=args.n_boot_items, n_boot_docs=args.n_boot_docs,
            seed=args.seed, noop_rtol=args.noop_rtol, inputs=inputs,
            pool=pool)
    except InputError as e:
        print(f"derived_verdict: input problem: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n",
                   encoding="utf-8")
    for line in summary_lines(result):
        print(line)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
