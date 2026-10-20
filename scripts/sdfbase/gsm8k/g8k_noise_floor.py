#!/usr/bin/env python
"""GSM8K paired-200: the NOISE FLOORS of a model whose two views are identical by construction.

On the ``llama8b`` target both views (``main`` = the base-lane objects transferred onto the instruct checkpoint, ``it`` =
the instruct-direct objects) serve the SAME instruct model; a cell with no object — ``none`` (the bare model) and
``prompted__si_constitution`` (the same SI prefix, no object) — is therefore the same configuration run twice, once per
view. Both legs are kept, and each pair is reported as a MEASURED NOISE FLOOR — the between-view
discrepancy per item and in aggregate on the same judged outcome, never collapsed — with a caution that states what the
two legs SHARE by construction versus what merely happens to match, and CHECKS that they did not differ in serving
geometry, batch composition or judge-cache state; a pair that did differ is labelled as measuring that difference instead.

This post-processor reads a PRODUCTION report of ``scripts/tinycap_gsm8k_paired200.py`` (its gates passed; the per-item
judged outcomes under both readings sit in ``cells[<key>].readings``) and the stores the report's legs point at (the
manifests, the request-evidence and usage-ledger records, the rows). It emits ``noise_floor.json`` and one figure. Nothing
here re-scores an item or re-derives a reading: the estimand is the analysis's own ``run_contrast`` (the paired bootstrap
over items, the McNemar sensitivity, the transitions), so the ``none`` pair reproduces the report's ``noise_floor`` block
leaf for leaf (checked, recorded) and the constitution pair is computed by the same code.

Usage::

    PYTHONPATH=src python scripts/sdfbase/gsm8k/g8k_noise_floor.py --report <final report.json> \\
        --out <dir>/noise_floor.json --figs-dir <dir>/figs [--model-key llama8b] [--cells none,prompted__si_constitution]
        [--bootstrap B --seed S --ci L]   # default: the report's own bootstrap block, so the none pair matches it exactly
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_SCRIPTS = _HERE.parents[1]


def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load_module("tinycap_gsm8k_paired200", _SCRIPTS / "tinycap_gsm8k_paired200.py")
V, T = A.V, A.T
AnalysisError = A.AnalysisError

SCHEMA = "g8k_noise_floor/1"
#: the two configurations the ruling names: the bare model and the prompted short-constitution SI (no object on either)
DEFAULT_CELLS: tuple[str, ...] = (A.CELL_NONE, A.CELL_PROMPTED_CONST)
LABEL_FLOOR, LABEL_MEASURES = "noise_floor", "measures_difference"
#: a pair whose stores were NOT read (``--no-stores``): no attestation exists, so no equivalence is claimed — a third, distinct
#: state (a missing attestation used to default the figure to "floor" and the title to "identical by construction"; so did an
#: explicit `measures_difference` verdict — the figure now derives both from the attestation)
LABEL_UNATTESTED = "unattested"
LABELS: tuple[str, ...] = (LABEL_FLOOR, LABEL_MEASURES, LABEL_UNATTESTED)
#: the row label's second line per state, and the title's claim per state (the affirmative wording ONLY when every computable
#: pair's attestation passed every attestable check)
LABEL_WORD = {LABEL_FLOOR: "floor", LABEL_MEASURES: "measures a difference", LABEL_UNATTESTED: "unattested (stores not read)"}
#: the per-item classes of a pair on one reading (the same judged outcome, main vs it)
CLASS_BOTH_CORRECT, CLASS_BOTH_WRONG, CLASS_MAIN_ONLY, CLASS_IT_ONLY, CLASS_UNSCORED = ("both_correct", "both_wrong", "main_only_correct",
                                                                                        "it_only_correct", "unscored_either")
CLASSES: tuple[str, ...] = (CLASS_BOTH_CORRECT, CLASS_BOTH_WRONG, CLASS_MAIN_ONLY, CLASS_IT_ONLY, CLASS_UNSCORED)
#: GSM8K's scorer is a deterministic string match on the extracted numeric answer (inspect's ``think_aware_scorer`` here);
#: no judge model is consulted, so there is no judge cache whose state could differ between the two legs
GSM8K_SCORER_KEYS: frozenset[str] = frozenset({"think_aware_scorer"})
SCORING_SENTENCE = ("GSM8K scoring is deterministic string matching on the extracted numeric answer (the rows carry exactly the "
                    "`think_aware_scorer` value in {0, 1}; no judge model, no judge backend) — there is no judge cache whose state "
                    "could differ between the two legs")
STEM = "final_g8k_noise_floor"
FIGURE_TITLE_TAIL = "same model, SI prefix, geometry, seeds and request bytes"


def pair_label(pair: Mapping[str, Any]) -> str:
    """The pair's state: the attestation's label when the stores were read, else :data:`LABEL_UNATTESTED` (never a default floor)."""
    att = pair.get("attestation")
    if att is None:
        return LABEL_UNATTESTED
    label = att.get("label")
    if label not in (LABEL_FLOOR, LABEL_MEASURES):
        raise AnalysisError(f"pair {pair.get('name')!r}: attestation label {label!r} is not one of {LABELS[:2]}")
    return label


def figure_title(doc: Mapping[str, Any]) -> str:
    """Two lines (<= 90 characters each), derived from the computable pairs' attestations: the affirmative "identical by
    construction" claim only when EVERY pair attests a floor; a pair that measures a difference names the differing checks;
    an unattested pair (``--no-stores``) says so and claims no equivalence."""
    pairs = [p for p in doc["pairs"] if p["computable"]]
    labels = [pair_label(p) for p in pairs]
    b = doc["bootstrap"]
    key, na = doc["model_key"], A.NA_LABEL
    if pairs and all(lab == LABEL_FLOOR for lab in labels):
        return (f"{key} noise floors: it − main on cells identical by construction (paired, B = {b['B']:,})\n"
                f"{FIGURE_TITLE_TAIL}; {na}")
    if any(lab == LABEL_UNATTESTED for lab in labels):
        n_un = sum(lab == LABEL_UNATTESTED for lab in labels)
        return (f"{key} it − main pairs, UNATTESTED (stores not read on {n_un} of {len(pairs)}; paired, B = {b['B']:,})\n"
                f"equivalence by construction NOT verified — no floor claimed; {na}")
    failing = sorted({c for p, lab in zip(pairs, labels) if lab == LABEL_MEASURES for c in (p["attestation"] or {}).get("failing") or []})
    n_diff = sum(lab == LABEL_MEASURES for lab in labels)
    second = f"differing: {', '.join(failing)}; {na}"
    if len(second) > A.TITLE_MAX_CHARS:
        second = second[:A.TITLE_MAX_CHARS - 1].rstrip(", ") + "…"
    return (f"{key} it − main pairs: {n_diff} of {len(pairs)} measure a DIFFERENCE, not a floor (paired, B = {b['B']:,})\n" + second)


# --------------------------------------------------------------------------- #
# per-item discordance and the aggregate spread                                #
# --------------------------------------------------------------------------- #


def item_class(main: Mapping[str, Any], it: Mapping[str, Any]) -> str:
    if not (main.get("scored") and it.get("scored")):
        return CLASS_UNSCORED
    a, b = float(main["value"]) >= 0.5, float(it["value"]) >= 0.5
    if a and b:
        return CLASS_BOTH_CORRECT
    if a:
        return CLASS_MAIN_ONLY
    if b:
        return CLASS_IT_ONLY
    return CLASS_BOTH_WRONG


def per_item_discordance(items_main: Mapping[str, Mapping[str, Any]], items_it: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The pair's per-item reading: every common item classed on the SAME judged outcome; the discordant ids listed per
    direction (never collapsed into the aggregate alone)."""
    common = sorted(set(items_main) & set(items_it))
    if set(items_main) != set(items_it):
        raise AnalysisError(f"the two legs' item sets differ ({len(items_main)} vs {len(items_it)}; {len(common)} common) — no pairing")
    classes = {i: item_class(items_main[i], items_it[i]) for i in common}
    counts = Counter(classes.values())
    return {"n_items": len(common), "classes": {c: int(counts.get(c, 0)) for c in CLASSES},
            "n_discordant": int(counts.get(CLASS_MAIN_ONLY, 0) + counts.get(CLASS_IT_ONLY, 0)),
            "discordant_ids": {CLASS_MAIN_ONLY: [i for i in common if classes[i] == CLASS_MAIN_ONLY],
                               CLASS_IT_ONLY: [i for i in common if classes[i] == CLASS_IT_ONLY]},
            "n_at_cap_by_leg": {"main": sum(1 for i in common if items_main[i].get("at_cap")), "it": sum(1 for i in common if items_it[i].get("at_cap"))},
            "n_at_cap_both": sum(1 for i in common if items_main[i].get("at_cap") and items_it[i].get("at_cap"))}


def _compact(stats: Mapping[str, Any], acc_by_leg: Mapping[str, Any]) -> dict[str, Any]:
    return {"point": stats["point"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "excludes_zero": stats["excludes_zero"],
            "acc_main": acc_by_leg["main"]["point"], "acc_it": acc_by_leg["it"]["point"],
            "n_scored_by_leg": {k: acc_by_leg[k]["n_scored"] for k in ("main", "it")}}


def spread(items_main: Mapping[str, Mapping[str, Any]], items_it: Mapping[str, Mapping[str, Any]], *, n_resamples: int, seed: int,
           ci: float) -> dict[str, Any]:
    """``it - main`` through the analysis's own :func:`run_contrast` (the report's noise-floor estimand: percentile paired
    bootstrap over items, the stock / W / X axes under both think branches, McNemar, transitions), plus the compact
    stock-primary reading the figure draws."""
    legs = {"it": items_it, "main": items_main}
    full = A.run_contrast(legs, {"it": +1, "main": -1}, A.CONTRAST_NOISE, plus="it", minus="main", n_resamples=n_resamples, seed=seed, ci=ci,
                          file_word=True)
    est = full["estimands"][A.AXIS_STOCK][A.BRANCH_PRIMARY]
    return {**full, "compact": _compact(est["stats"][A.CONTRAST_NOISE], est["acc_by_leg"])}


# --------------------------------------------------------------------------- #
# the stores behind the two legs: the shared-configuration attestation         #
# --------------------------------------------------------------------------- #


def _store_paths(leg: Mapping[str, Any], gen_root: Path | None) -> tuple[Path, Path]:
    """``(cell_dir, out_root)`` of a report leg — the report's absolute paths, re-rooted under ``gen_root`` when given and
    the recorded paths are not on disk (a report read on another machine)."""
    cell_dir, out_root = Path(str(leg["cell_dir"])), Path(str(leg["out_root"]))
    if gen_root is not None and not cell_dir.is_dir():
        marker = "controls"
        for p in (cell_dir, out_root):
            if marker not in p.parts:
                raise AnalysisError(f"cannot re-root {p} under {gen_root}: no `{marker}` component")
        cell_dir = gen_root.joinpath(*cell_dir.parts[cell_dir.parts.index(marker):])
        out_root = gen_root.joinpath(*out_root.parts[out_root.parts.index(marker):])
    for p in (cell_dir, out_root):
        if not p.is_dir():
            raise AnalysisError(f"store directory missing: {p}")
    return cell_dir, out_root


def store_facts(leg: Mapping[str, Any], *, gen_root: Path | None = None) -> dict[str, Any]:
    """What one leg's store says about its configuration and its wire traffic: the manifest's serving / geometry / sampling
    / cap / cell identity fields, the task request bodies by sample (sha256 + bytes), the ledger's task rows by sample, the
    rows' per-item seeds, completion digests and scorer keys."""
    cell_dir, out_root = _store_paths(leg, gen_root)
    m = T._mapping(T._read_json(cell_dir / "manifest.json"))
    prov, serving = T._mapping(m.get("provenance")), T._mapping(T._mapping(m.get("provenance")).get("serving"))
    geom, samp, route, rc = T._mapping(m.get("geometry")), T._mapping(m.get("sampling")), T._mapping(m.get("route")), T._mapping(m.get("row_constants"))
    tiny, binding = T._mapping(m.get("tinycap")), T._mapping(m.get("binding"))
    fw = T._mapping(m.get("frameworks")) or T._mapping(tiny.get("frameworks"))
    fit_prov = T._mapping(m.get("fit_provenance"))
    requests: dict[str, dict[str, Any]] = {}
    for rec in T._iter_jsonl(out_root / A.C.REQUEST_EVIDENCE_FILENAME):
        if rec.get("kind") == "task":
            requests[str(rec["sample"])] = {"body_sha256": rec.get("body_sha256"), "body_bytes": rec.get("body_bytes"), "max_tokens": rec.get("max_tokens"),
                                            "attempt_id": rec.get("attempt_id")}
    ledger: dict[str, dict[str, Any]] = {}
    for row in T._iter_jsonl(out_root / A.C.LEDGER_FILENAME):
        if not str(row.get("task", "")).startswith("probe"):
            ledger[str(row["sample"])] = {k: row.get(k) for k in ("prompt_tokens_pre", "max_tokens_requested", "status", "finish_reason", "thinking",
                                                                   "ts", "latency_s", "body_sha256", "usage_completion_tokens")}
    rows: dict[str, dict[str, Any]] = {}
    for row in T._iter_jsonl(cell_dir / "rows.jsonl"):
        ej = row.get("extra_json")
        ej = json.loads(ej) if isinstance(ej, str) else T._mapping(ej)
        if row.get("item_id") == A.C.AGG_ITEM_ID or row.get("metric") != A.C.METRIC_ACC:
            continue
        rows[str(row["item_id"])] = {"seed": ej.get("seed"), "raw_content_sha256": ej.get("raw_content_sha256"),
                                     "scorer_keys": sorted(T._mapping(ej.get("scores")).keys()), "value": row.get("value"),
                                     "thinking": ej.get(A.C.ROW_KEY_THINKING)}
    ts = sorted(str(r["ts"]) for r in ledger.values() if r.get("ts"))
    lat = [float(r["latency_s"]) for r in ledger.values() if isinstance(r.get("latency_s"), (int, float))]
    summ = T._mapping(leg.get("summary"))
    return {
        "leg": leg.get("leg"), "cell_dir": str(cell_dir), "out_root": str(out_root),
        "model": {"model_id": serving.get("resolved_model_id") or m.get("model_id"), "revision": serving.get("resolved_revision") or m.get("model_revision"),
                  "requested_revision": serving.get("requested_revision")},
        "template_hash": serving.get("chat_template_hash") or fit_prov.get("served_template_hash"),
        "serving_identity": serving.get("serving_identity") or m.get("serving_identity"), "virtual_model": serving.get("virtual_model") or m.get("virtual_model"),
        "geometry": {k: geom.get(k) for k in ("eval", "max_model_len", "max_num_seqs", "budget", "batched_tokens", "budget_source", "sized_by")},
        "concurrency": m.get("concurrency"),
        "sampling": {"mode": samp.get("mode"), "temperature": samp.get("temperature"), "top_p": samp.get("top_p"), "top_k": samp.get("top_k"),
                     "sampling_ref": summ.get("sampling_ref"), "seed_rule": T._mapping(samp.get("seed_rule")).get("inspect")},
        "cap": {"on_cap_by_task": m.get("on_cap_by_task"), "max_tokens_by_task": tiny.get("max_tokens_by_task"), "summary_max_tokens": summ.get("max_tokens")},
        "thinking_mode": T._mapping(m.get("thinking")).get("mode"),
        "cell": {"cell_id": m.get("cell_id"), "harness_cell_id": m.get("harness_cell_id"), "object_artifact_sha256": route.get("object_artifact_sha256")},
        # derived from the hashed configs (the producer's); equal in production for a no-object cell — recorded and compared, but not
        # label-deciding: a differing hash says the hashed config differs at least in a label, not that the serving did
        "identity_hashes": {"intervention_hash": m.get("intervention_hash"), "harness_config_hash": m.get("harness_config_hash")},
        "producer_tree_sha": T._mapping(m.get("producer")).get("tree_sha"),
        "frameworks": {k: fw.get(k) for k in ("inspect_ai", "inspect_evals", "client_lock_sha256")},
        "judge": {"judge_backend": prov.get("judge_backend"), "scorer_keys": sorted({k for r in rows.values() for k in r["scorer_keys"]}),
                  "values": sorted({float(r["value"]) for r in rows.values() if isinstance(r.get("value"), (int, float))})},
        "designed": {"bank_view": route.get("bank_view")},
        "incidental": {"slurm_job_id": rc.get("slurm_job_id"), "ledger_url": binding.get("ledger_url"), "proxy_url": binding.get("proxy_url"),
                       "ts_first": ts[0] if ts else None, "ts_last": ts[-1] if ts else None,
                       "latency_mean_s": (sum(lat) / len(lat)) if lat else None},
        "requests": requests, "ledger": ledger, "rows": rows,
    }


def _eq(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def _present(x: Any) -> bool:
    """Whether a recorded value carries any content (a mapping with every leaf null is absent)."""
    if isinstance(x, Mapping):
        return any(_present(v) for v in x.values())
    return x is not None


def _shared(name: str, a: Any, b: Any) -> dict[str, Any]:
    pa, pb = _present(a), _present(b)
    return {"name": name, "main": a, "it": b, "equal": _eq(a, b), "present": pa and pb, "present_any": pa or pb}


def _check(name: str, passed: bool, detail: str, *, attested: bool = True) -> dict[str, Any]:
    """A check that is NOT attestable (the field is recorded on neither leg) is filed with ``passed: None`` and never
    counts as a difference — the label text names it instead; a field present on ONE leg only is a difference."""
    if not attested:
        return {"name": name, "passed": None, "attested": False, "detail": f"not recorded on either leg — cannot be attested: {detail}"}
    return {"name": name, "passed": bool(passed), "attested": True, "detail": detail}


def _field_check(shared: Mapping[str, Any], detail: str) -> dict[str, Any]:
    return _check(shared["name"], shared["equal"], detail, attested=bool(shared["present_any"]))


def attest_pair(fm: Mapping[str, Any], fi: Mapping[str, Any]) -> dict[str, Any]:
    """The shared-configuration attestation of one pair: ``shared`` = what the two legs share BY CONSTRUCTION, each
    field read from both stores and compared; ``designed_difference`` = the one thing the design varies (the bank view,
    which on a no-object cell of this target routes to the same weights); ``incidental`` = what merely happens to match or
    differ (job ids, ports, wall clock); ``checks`` = the ruling's three (serving geometry, batch composition, judge-cache
    state) plus the identity of everything the request carried; ``label`` = ``noise_floor`` iff every check passed, else
    ``measures_difference`` naming the differing checks."""
    shared = [_shared("model_revision", fm["model"], fi["model"]), _shared("served_template_hash", fm["template_hash"], fi["template_hash"]),
              _shared("serving_identity", fm["serving_identity"], fi["serving_identity"]), _shared("virtual_model", fm["virtual_model"], fi["virtual_model"]),
              _shared("serving_geometry", fm["geometry"], fi["geometry"]), _shared("sampling_row", fm["sampling"], fi["sampling"]),
              _shared("cap", fm["cap"], fi["cap"]), _shared("thinking_mode", fm["thinking_mode"], fi["thinking_mode"]),
              _shared("cell_identity", fm["cell"], fi["cell"]), _shared("producer_tree_sha", fm["producer_tree_sha"], fi["producer_tree_sha"]),
              _shared("frameworks", fm["frameworks"], fi["frameworks"]), _shared("identity_hashes", fm["identity_hashes"], fi["identity_hashes"])]
    by_name = {s["name"]: s for s in shared}
    # the wire: every task request of the pair, by sample, byte for byte
    rq_m, rq_i = fm["requests"], fi["requests"]
    common_rq = sorted(set(rq_m) & set(rq_i))
    n_body_equal = sum(1 for s in common_rq if rq_m[s]["body_sha256"] == rq_i[s]["body_sha256"])
    rows_m, rows_i = fm["rows"], fi["rows"]
    common_items = sorted(set(rows_m) & set(rows_i))
    n_seed_equal = sum(1 for i in common_items if rows_m[i]["seed"] is not None and rows_m[i]["seed"] == rows_i[i]["seed"])
    n_completion_equal = sum(1 for i in common_items if rows_m[i]["raw_content_sha256"] is not None
                             and rows_m[i]["raw_content_sha256"] == rows_i[i]["raw_content_sha256"])
    ld_m, ld_i = fm["ledger"], fi["ledger"]
    common_ld = sorted(set(ld_m) & set(ld_i))
    n_prompt_equal = sum(1 for s in common_ld if ld_m[s]["prompt_tokens_pre"] == ld_i[s]["prompt_tokens_pre"])
    n_cap_equal = sum(1 for s in common_ld if ld_m[s]["max_tokens_requested"] == ld_i[s]["max_tokens_requested"])
    thinking_null = all(r["thinking"] is None for r in list(ld_m.values()) + list(ld_i.values()))
    n_expected = len(common_items)
    n_seed_present = sum(1 for i in common_items if rows_m[i]["seed"] is not None and rows_i[i]["seed"] is not None)
    scorer_keys = sorted({k for f in (fm, fi) for k in f["judge"]["scorer_keys"]})
    scoring_ok = all(set(f["judge"]["scorer_keys"]) <= GSM8K_SCORER_KEYS and f["judge"]["judge_backend"] is None
                     and set(f["judge"]["values"]) <= {0.0, 1.0} for f in (fm, fi))
    geo_present = all(_present(f["geometry"].get("max_model_len")) and _present(f["geometry"].get("max_num_seqs")) for f in (fm, fi))
    checks = [
        _check("serving_geometry", by_name["serving_geometry"]["equal"],
               f"the served geometry entry, max_model_len, S and the budget read from both manifests: {fm['geometry']} vs {fi['geometry']}",
               attested=geo_present),
        _check("batch_composition", fm["concurrency"] is not None and fm["concurrency"] == fi["concurrency"] and len(ld_m) == len(ld_i) == n_expected
               and n_cap_equal == len(common_ld) == n_expected,
               f"S (the standup's max_num_seqs = the client's in-flight concurrency) {fm['concurrency']} vs {fi['concurrency']}; ledger task rows "
               f"{len(ld_m)} vs {len(ld_i)}; max_tokens equal on {n_cap_equal} of {len(common_ld)} samples. The per-step co-scheduled set is not "
               "recorded by either leg and is exactly what a fixed-seed floor measures",
               attested=fm["concurrency"] is not None or fi["concurrency"] is not None),
        _check("judge_cache_state", scoring_ok,
               (SCORING_SENTENCE + f" (scorer keys observed: {scorer_keys}; judge backend {fm['judge']['judge_backend']!r} / {fi['judge']['judge_backend']!r})")
               if scoring_ok else
               f"a judged outcome or a non-binary score is present (scorer keys {fm['judge']['scorer_keys']} / {fi['judge']['scorer_keys']}, judge "
               f"backend {fm['judge']['judge_backend']!r} / {fi['judge']['judge_backend']!r}) — the judge-cache state is NOT attested"),
        _field_check(by_name["model_revision"], f"resolved model and revision {fm['model']} vs {fi['model']}"),
        _field_check(by_name["served_template_hash"], f"chat_template_hash {fm['template_hash']!r} vs {fi['template_hash']!r}"),
        _field_check(by_name["sampling_row"], f"{fm['sampling']} vs {fi['sampling']}"),
        _field_check(by_name["cap"], f"{fm['cap']} vs {fi['cap']}"),
        _field_check(by_name["thinking_mode"], f"{fm['thinking_mode']!r} vs {fi['thinking_mode']!r}"),
        _field_check(by_name["cell_identity"], f"cell id, harness cell, object sha {fm['cell']} vs {fi['cell']}"),
        _field_check(by_name["producer_tree_sha"], f"{fm['producer_tree_sha']!r} vs {fi['producer_tree_sha']!r}"),
        _check("request_bytes", n_expected > 0 and len(common_rq) == n_expected and n_body_equal == n_expected,
               f"task request bodies byte-identical (sha256) on {n_body_equal} of {len(common_rq)} common samples ({len(rq_m)} / {len(rq_i)} task "
               f"records; {n_expected} items); prompt tokens equal on {n_prompt_equal} of {len(common_ld)} ledger rows — the SI prefix, the question, "
               "the sampler keys, max_tokens and the seed all ride the body",
               attested=bool(rq_m) or bool(rq_i)),
        _check("seeds", n_expected > 0 and n_seed_equal == n_expected, f"per-item seeds equal on {n_seed_equal} of {n_expected} items",
               attested=n_seed_present > 0),
        _check("ledger_thinking_absent", thinking_null, "every ledger row's `thinking` is null on both legs (no switch rode the wire)",
               attested=bool(ld_m) or bool(ld_i)),
    ]
    failing = [c["name"] for c in checks if c["attested"] and not c["passed"]]
    not_attestable = [c["name"] for c in checks if not c["attested"]]
    hashes_note = ("; the producer's intervention and harness-config hashes are equal across the two legs" if by_name["identity_hashes"]["equal"]
                   else "; the producer's intervention / harness-config hashes differ across the two legs (derived from the hashed configs, "
                        "which may carry the view label; recorded, not label-deciding)")
    return {
        "shared": shared,
        "designed_difference": {"bank_view": {"main": fm["designed"]["bank_view"], "it": fi["designed"]["bank_view"]},
                                "note": "the ONE thing the design varies; on a cell with no object the route serves the same weights either way "
                                        "(object_artifact_sha256 null on both legs)"},
        "incidental": {"main": fm["incidental"], "it": fi["incidental"],
                       "note": "job ids, endpoints and wall clock — separate standups minutes apart; not part of the construction, expected to differ"},
        "wire": {"n_items": n_expected, "n_request_bodies_identical": n_body_equal, "n_common_requests": len(common_rq),
                 "n_seeds_identical": n_seed_equal, "n_completions_byte_identical": n_completion_equal,
                 "n_prompt_tokens_identical": n_prompt_equal, "n_ledger_rows_common": len(common_ld)},
        "checks": checks, "failing": failing, "not_attestable": not_attestable,
        "label": LABEL_FLOOR if not failing else LABEL_MEASURES,
        "label_text": (("a measured noise floor: the two legs share model revision, SI prefix, served template, serving geometry, sampling row, cap, seeds "
                        "and request bytes by construction; what differs is the standup and its per-step batch composition"
                        + hashes_note + (f" (not attestable from these stores: {', '.join(not_attestable)})" if not_attestable else "")) if not failing else
                       f"NOT a noise floor — the pair measures the difference in: {', '.join(failing)}"),
    }


# --------------------------------------------------------------------------- #
# the pairs                                                                    #
# --------------------------------------------------------------------------- #


def _comparable_in_report(report: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    """Why a set of the REPORT's cells has no joint estimand (the reader's :func:`_comparable`, read off the serialised
    report: the legs live under ``report["legs"]``): a missing / refused cell, differing attempted ids, mixed producers."""
    cells, legs = report["cells"], report["legs"]
    for k in keys:
        c = cells.get(k)
        if c is None:
            return f"{k}: not in the design"
        base = legs.get(c["base_leg"]) or {}
        if not A.usable(base) or c.get("readings") is None:
            return f"{k}: {base.get('missing') or base.get('refused') or 'no readings'}"
        if c.get("rung_leg") is not None and not A.usable(legs.get(c["rung_leg"]) or {}):
            r = legs.get(c["rung_leg"]) or {}
            return f"{k} rung: {r.get('missing') or r.get('refused')}"
    ids = {frozenset(legs[cells[k]["base_leg"]]["attempted_ids"]) for k in keys}
    if len(ids) != 1:
        return "the cells' attempted id sets differ"
    shas = {k: legs[cells[k]["base_leg"]]["producer"]["tree_sha"] for k in keys}
    for k in keys:
        if cells[k].get("rung_leg") is not None:
            shas[k + "@rung"] = legs[cells[k]["rung_leg"]]["producer"]["tree_sha"]
    if len(set(shas.values())) > 1:
        return f"the legs' producer trees differ — a mixed-producer comparison: {shas}"
    return None


def _models_in_both_views(report: Mapping[str, Any]) -> list[str]:
    views = [str(v) for v in T._mapping(report.get("design")).get("views") or []]
    keys = {v.split("/")[0] for v in views}
    return sorted(k for k in keys if f"{k}/{V.VIEW_MAIN}" in views and f"{k}/{V.VIEW_IT}" in views)


def noise_floor_pairs(report: Mapping[str, Any], *, model_key: str, cells: Sequence[str], n_resamples: int, seed: int, ci: float,
                      gen_root: Path | None = None, read_stores: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rcells = report["cells"]
    for cell in cells:
        mk, ik = V.cell_key(model_key, V.VIEW_MAIN, cell), V.cell_key(model_key, V.VIEW_IT, cell)
        rec: dict[str, Any] = {"name": f"{model_key}: {cell}(it) - {cell}(main)", "kind": "noise_floor", "model_key": model_key, "cell": cell,
                               "main_key": mk, "it_key": ik, "computable": False, "reason": None, "readings": {}, "per_item": {}, "attestation": None,
                               "consistency_with_report": None}
        problem = _comparable_in_report(report, [mk, ik])
        if problem:
            rec["reason"] = problem
            out.append(rec)
            continue
        rec["computable"] = True
        for reading in A.READINGS:
            im, ii = A._leg_reading(rcells, mk, reading), A._leg_reading(rcells, ik, reading)
            rec["per_item"][reading] = per_item_discordance(im, ii)
            rec["readings"][reading] = spread(im, ii, n_resamples=n_resamples, seed=seed, ci=ci)
        # the none pair IS the report's own noise_floor block: the same estimand by the same code must agree leaf for leaf
        ref = next((n for n in report.get("noise_floor") or [] if n.get("model_key") == model_key and n.get("computable")), None)
        if cell == A.CELL_NONE and ref is not None:
            same = all(_eq(ref["readings"][r]["estimands"], rec["readings"][r]["estimands"]) for r in A.READINGS)
            rec["consistency_with_report"] = {"checked": True, "equal": same,
                                              "detail": "estimands (every axis, both branches) of the report's noise_floor entry vs this pair's"}
            if not same:
                raise AnalysisError(f"{rec['name']}: this pair's estimands differ from the report's own noise_floor block — bootstrap B / seed / level "
                                    f"not the report's? (report {report.get('bootstrap')}, here B={n_resamples} seed={seed} level={ci})")
        if read_stores:
            legs = report["legs"]
            fm = store_facts(legs[rcells[mk]["base_leg"]], gen_root=gen_root)
            fi = store_facts(legs[rcells[ik]["base_leg"]], gen_root=gen_root)
            att = attest_pair(fm, fi)
            # the stores' raw records stay out of the document (the counts above summarise them)
            rec["attestation"] = att
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# the figure                                                                   #
# --------------------------------------------------------------------------- #


def _num(numbers: list[dict[str, Any]], doc: Mapping[str, Any], name: str, path: str, value: Any) -> None:
    got = T.report_path(doc, path)
    if not T._same_leaf(value, got):
        raise AnalysisError(f"figure number {name!r}: the plotted value {value!r} is not the leaf at {path!r} ({got!r})")
    numbers.append({"name": name, "value": value, "source": path})


def fig_noise_floor(doc: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Left: the aggregate spread it − main per pair (merged filled, 8,192-only hollow; 95 % percentile CI, stock scorer,
    primary branch). Right: the per-item reading per pair — byte-identical completions, discordant items per direction, on
    the merged reading. Hue = the cell family (fixed, never cycled); the label of a pair that is NOT a floor is written
    in place."""
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    pairs = [(j, p) for j, p in enumerate(doc["pairs"]) if p["computable"]]
    n = max(len(pairs), 1)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.0, 0.9 * n + 1.9), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.42})
    A._layout(fig, left=0.19, right=0.985)
    ax.axvline(0.0, color=A.BASELINE, lw=0.8, zorder=2)
    labels: list[str] = []
    ys = list(range(len(pairs)))[::-1]
    for y, (j, p) in zip(ys, pairs):
        color = A.FAMILY_COLOR[A.cell_family(p["cell"])]
        labels.append(f"{A._short(p['cell'])}\n{LABEL_WORD[pair_label(p)]}")
        cm, c8 = p["readings"][A.READING_MERGED]["compact"], p["readings"][A.READING_R8192]["compact"]
        ax.plot([A._pp(cm["ci_low"]), A._pp(cm["ci_high"])], [y + 0.14, y + 0.14], color=A.INK, lw=1.1, zorder=5)
        ax.plot([A._pp(cm["point"])], [y + 0.14], marker="o", ms=6, color=color, mec=A.INK, mew=0.5, ls="", zorder=6)
        ax.plot([A._pp(c8["ci_low"]), A._pp(c8["ci_high"])], [y - 0.14, y - 0.14], color=A.INK2, lw=0.8, zorder=4)
        ax.plot([A._pp(c8["point"])], [y - 0.14], marker="o", ms=5.5, mfc="none", mec=A.INK2, mew=1.0, ls="", zorder=6)
        ax.text(1.01, y, f"{A._pp(cm['point']):+.1f} pp [{A._pp(cm['ci_low']):+.1f}, {A._pp(cm['ci_high']):+.1f}]", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.4, color=A.INK, clip_on=False)
        base = f"pairs.{j}.readings"
        for reading, c in ((A.READING_MERGED, cm), (A.READING_R8192, c8)):
            for f in ("point", "ci_low", "ci_high"):
                _num(numbers, doc, f"{p['name']}.{reading}.{f}", f"{base}.{reading}.compact.{f}", c[f])
        # right: the per-item reading (merged)
        pi = p["per_item"][A.READING_MERGED]
        wire = (p["attestation"] or {}).get("wire") or {}
        bars = [("byte-identical completions", wire.get("n_completions_byte_identical"), f"pairs.{j}.attestation.wire.n_completions_byte_identical", A.MUTED),
                ("discordant: main correct only", pi["classes"][CLASS_MAIN_ONLY], f"pairs.{j}.per_item.{A.READING_MERGED}.classes.{CLASS_MAIN_ONLY}", A.CAT["orange"]),
                ("discordant: it correct only", pi["classes"][CLASS_IT_ONLY], f"pairs.{j}.per_item.{A.READING_MERGED}.classes.{CLASS_IT_ONLY}", A.CAT["aqua"])]
        offsets = (0.26, 0.0, -0.26)
        for (label, value, path, col), dy in zip(bars, offsets):
            if value is None:
                continue
            bx.barh(y + dy, value, height=0.22, color=col, edgecolor="none", zorder=3)
            bx.text(value + 2, y + dy, f"{value}", va="center", ha="left", fontsize=6.4, color=A.INK)
            _num(numbers, doc, f"{p['name']}.{label}", path, value)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("it − main accuracy (percentage points)")
    T._recessive(ax)
    ax.xaxis.grid(True, zorder=0)
    ax.yaxis.grid(False)
    bx.set_yticks(ys)
    bx.set_yticklabels([""] * len(ys))
    bx.set_ylim(-0.7, n - 0.3)
    bx.set_xlim(0, doc["n_items"] * 1.12)
    bx.set_xlabel(f"items (of {doc['n_items']}; merged reading)")
    T._recessive(bx)
    bx.xaxis.grid(True, zorder=0)
    bx.yaxis.grid(False)
    handles = [plt.Line2D([], [], marker="o", color=A.INK, mec=A.INK, ls="", label="merged reading, 95 % CI"),
               plt.Line2D([], [], marker="o", mfc="none", mec=A.INK2, ls="", label="8,192-only reading, 95 % CI"),
               Patch(facecolor=A.MUTED, label="byte-identical completions"), Patch(facecolor=A.CAT["orange"], label="discordant: main correct only"),
               Patch(facecolor=A.CAT["aqua"], label="discordant: it correct only")]
    # below the axes (the tight bounding box at save time includes it, as it does the footers of the analysis's figures)
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=6.0, bbox_to_anchor=(0.5, -0.005))
    A._title(fig, figure_title(doc))
    return fig, numbers


# --------------------------------------------------------------------------- #
# the document and the CLI                                                     #
# --------------------------------------------------------------------------- #


def build(report: Mapping[str, Any], *, model_key: str | None, cells: Sequence[str], n_resamples: int | None, seed: int | None, ci: float | None,
          gen_root: Path | None, read_stores: bool = True) -> dict[str, Any]:
    refusals = A.production_refusals(report)
    if refusals:
        raise AnalysisError("the report is not a production result — " + "; ".join(refusals))
    b = T._mapping(report.get("bootstrap"))
    n_resamples = int(b["B"]) if n_resamples is None else n_resamples
    seed = int(b["seed"]) if seed is None else seed
    ci = float(b["level"]) if ci is None else ci
    both = _models_in_both_views(report)
    if model_key is None:
        if len(both) != 1:
            raise AnalysisError(f"--model-key required: the design runs {both or 'no model'} in both views")
        model_key = both[0]
    elif model_key not in both:
        raise AnalysisError(f"--model-key {model_key!r}: not run in both views by this design ({both})")
    pairs = noise_floor_pairs(report, model_key=model_key, cells=cells, n_resamples=n_resamples, seed=seed, ci=ci, gen_root=gen_root, read_stores=read_stores)
    arm = A._arm_of_model(report, model_key)
    return {"schema": SCHEMA, "experiment": report.get("experiment"), "task": report.get("task"), "sha": report.get("sha"),
            "analysis_code_sha": report.get("analysis_code_sha"), "report": str(report.get("_path") or ""),
            "model_key": model_key, "thinking_mode": arm, "thinking_label": V.thinking_label(arm), "n_items": int(A.N_ITEMS),
            "bootstrap": {"B": n_resamples, "seed": seed, "level": ci}, "cells": list(cells), "pairs": pairs,
            "summary": [{"name": p["name"], "computable": p["computable"], "reason": p["reason"],
                         "label": (pair_label(p) if p["computable"] else None), "failing_checks": (p["attestation"] or {}).get("failing"),
                         **({r: p["readings"][r]["compact"] for r in A.READINGS} if p["computable"] else {}),
                         **({f"n_discordant_{r}": p["per_item"][r]["n_discordant"] for r in A.READINGS} if p["computable"] else {}),
                         "n_completions_byte_identical": ((p["attestation"] or {}).get("wire") or {}).get("n_completions_byte_identical")}
                        for p in pairs],
            "ruling": ("Orchestrator 2026-09-12 (D34): two cells identical by construction are a measured noise floor — the between-view discrepancy "
                       "per item and in aggregate on the same judged outcome, never collapsed; the paper reader takes one view's row for the figure "
                       "and the pair's spread is a caption number; the caution states what is SHARED versus what merely matches and checks serving "
                       "geometry, batch composition and judge-cache state — else the pair measures that difference")}


def write_outputs(doc: dict[str, Any], *, out: Path, figs_dir: Path | None) -> dict[str, Any]:
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Any] = {"written": {}, "numbers": []}
    if figs_dir is not None:
        figs_dir = figs_dir.expanduser().resolve()
        fig, numbers = fig_noise_floor(doc)
        frac = A.axes_height_fraction(fig)
        if frac < A.AXES_MIN_FRACTION:
            raise AnalysisError(f"{STEM}: the axes span {frac:.3f} of the figure height (< {A.AXES_MIN_FRACTION})")
        stem = f"{STEM}_{doc['model_key']}"
        figs["written"][stem] = [str(p) for p in T._save(fig, figs_dir, stem)]
        figs["numbers"] = numbers
    doc["figure"] = figs
    out.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return figs


def log_wandb_resume(url: str, doc: Mapping[str, Any], figs: Mapping[str, Any]) -> str | None:
    """Add the noise-floor figure and its headline numbers to the analysis's OWN W&B run (resumed by id from its URL)."""
    try:
        import wandb
    except ImportError as e:    # pragma: no cover - venv-dependent
        raise AnalysisError("wandb is not importable — omit --wandb-resume-url or use a venv that carries it") from e
    run_id = url.rstrip("/").split("/")[-1]
    from sisteer.wandb_entity import wandb_project
    run = wandb.init(project=wandb_project(A.WANDB_PURPOSE), id=run_id, resume="must")   # the analysis's own project: a resume must not move it
    for stem, paths in figs.get("written", {}).items():
        png = next((p for p in paths if p.endswith(".png")), None)
        if png:
            wandb.log({f"g8k/{stem}": wandb.Image(png)})
    summary: dict[str, Any] = {}
    for s in doc["summary"]:
        if not s["computable"]:
            continue
        for r in A.READINGS:
            for f in ("point", "ci_low", "ci_high"):
                summary[f"noise_floor/{s['name']}/{r}/{f}"] = s[r][f]
            summary[f"noise_floor/{s['name']}/{r}/n_discordant"] = s[f"n_discordant_{r}"]
        summary[f"noise_floor/{s['name']}/label"] = s["label"]
    run.summary.update(summary)
    url_out = getattr(run, "url", None)
    run.finish()
    return url_out


def print_summary(doc: Mapping[str, Any]) -> None:
    print(f"[g8k noise floor] {doc['model_key']} ({doc['thinking_label']}), store pin {doc['sha']}, B = {doc['bootstrap']['B']:,}")
    for s in doc["summary"]:
        if not s["computable"]:
            print(f"  {s['name']}: not computable — {s['reason']}")
            continue
        m, r8 = s[A.READING_MERGED], s[A.READING_R8192]
        print(f"  {s['name']}: merged {A._pp(m['point']):+.1f} pp [{A._pp(m['ci_low']):+.1f}, {A._pp(m['ci_high']):+.1f}] "
              f"(acc main {m['acc_main']:.3f} / it {m['acc_it']:.3f}; {s['n_discordant_' + A.READING_MERGED]} discordant items); "
              f"8,192-only {A._pp(r8['point']):+.1f} pp [{A._pp(r8['ci_low']):+.1f}, {A._pp(r8['ci_high']):+.1f}]; "
              f"byte-identical completions {s['n_completions_byte_identical']}; label {s['label']}"
              + (f" (differs in: {', '.join(s['failing_checks'])})" if s["failing_checks"] else ""))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True, help="a PRODUCTION report of scripts/tinycap_gsm8k_paired200.py (its gates passed)")
    ap.add_argument("--out", type=Path, required=True, help="noise_floor.json")
    ap.add_argument("--figs-dir", type=Path, default=None, help="where the figure (PNG + PDF) is written; absent -> no figure")
    ap.add_argument("--model-key", default=None, help="the model run in both views (default: the design's one such model)")
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS), help="the no-object cells to pair across the views (comma-separated)")
    ap.add_argument("--gen-root", type=Path, default=None, help="re-root the report's store paths here when they are not on disk")
    ap.add_argument("--no-stores", action="store_true", help="skip the store reads (no attestation; the pairs carry readings and discordance only)")
    ap.add_argument("--bootstrap", type=int, default=None, metavar="B", help="default: the report's")
    ap.add_argument("--seed", type=int, default=None, help="default: the report's")
    ap.add_argument("--ci", type=float, default=None, help="default: the report's")
    ap.add_argument("--wandb-resume-url", default=None, help="the analysis run's W&B URL: the figure and the headline numbers join that run")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    report = T._read_json(a.report.expanduser().resolve())
    if not isinstance(report, Mapping):
        raise AnalysisError(f"--report {a.report}: not a JSON object")
    report = {**dict(report), "_path": str(a.report.expanduser().resolve())}
    cells = [c for c in a.cells.split(",") if c]
    doc = build(report, model_key=a.model_key, cells=cells, n_resamples=a.bootstrap, seed=a.seed, ci=a.ci, gen_root=a.gen_root, read_stores=not a.no_stores)
    figs = write_outputs(doc, out=a.out, figs_dir=a.figs_dir)
    print_summary(doc)
    if a.wandb_resume_url:
        url = log_wandb_resume(a.wandb_resume_url, doc, figs)
        doc["wandb_url"] = url
        a.out.expanduser().resolve().write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
        print(f"[g8k noise floor] W&B: {url}")
    print(f"[g8k noise floor] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
