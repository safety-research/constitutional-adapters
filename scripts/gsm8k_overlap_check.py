"""The GSM8K overlap check: the retired instrument's reading
(``s2parity/gsm8k_0shot``, thinking OFF, greedy, cap 512 — READ from the
graded controls store, never generated) against the stock instrument's
thinking-OFF greedy 512 leg (``inspect/gsm8k_paired200``), paired item by
item, per model.

    PYTHONPATH=src python scripts/gsm8k_overlap_check.py --retired-root <graded controls store> \\
        --stock-root <local pull of the wave root>/overlap --ids <canonical ids json> --sha <wave pin> \\
        [--pairing plan|<platinum rows json>] [--models k1,k2,...] [--eval-logs] --out overlap.json

``--sha`` is the STORE PIN — the wave sha every stock leg's producer tree must
carry, the same pin run E takes (``tinycap_gsm8k_paired200.py --sha``); a stock
store whose manifest names another tree is refused by both hashes, never read.
(Until 2026-09-11 this check handed ``producer_problem`` the manifest's OWN
tree sha as the pin, so a store from another tree passed it; a merged pull put
three such stores under the pin's root and the check read their records
against the pin's logs as per-item disagreements.) With ``--eval-logs`` the one
qualifying log is also bound to the manifest's accepted attempt window
(``tinycap_gsm8k_validators.eval_cross_check``).

THE PAIRING is on ``item_id`` (never on question text the retired rows do not
carry): the retired instrument's id is ``parity._content_id("g", i, question)``
= ``g<i:04d>-<sha256(question)[:8]>`` over its OWN selection
(``madrylab/gsm8k-platinum`` main / test at the pinned revision,
``shuffle(seed=0).select(range(200))``), and the stock task's id is
``contracts.gsm8k_sample_id(question)`` = ``gsm8k_<md5(question)[:8]>`` — so
both ids are functions of the SAME byte-exact question text. The check
re-derives that selection (``--pairing plan``: ``parity.gsm8k_parity_rows``
under the pinned revision, in this venv; or a rows file of the same shape),
recomputes BOTH hashes per row, and matches each retired ``acc`` row's id on
its content-hash SUFFIX (a byte-identical question) and, separately, its
POSITION prefix: suffix + prefix match = the same question at the same
position; suffix match + prefix mismatch = the same questions in another
order (noted, fine); suffix mismatch = the texts differ by bytes — a FINDING,
filed, not a failure. With ``--eval-logs`` the stock leg's retained ``.eval``
supplies each sample's input text and both hashes are recomputed over it (the
byte check on the stock side). Whatever the selection yields is filed plainly:
200 / 200 when the stock task draws from the same pinned selection, fewer when
it does not — the incomparability the check exists to surface.

Statistics per model: acc_retired, acc_stock, Δ = stock − retired with
a paired percentile bootstrap 95 % CI (B = 10,000, seed 0; the MATH-500
analysis' ``paired_bootstrap_conditional``, imported), the exact two-sided
McNemar, the disagreement count b + c, the extraction-failed / unparseable
counts. Verdict: ``continuation`` iff the CI lies inside (−5, +5) pp, else
``new_instrument`` — filed just as visibly, with the power note (C8). The two
readings are two instruments and are never pooled; the ``it`` view's retired
store of the same model is read for its AGREEMENT with the ``main`` store and
never pooled either (C9). Nothing here enters a store or a controls slot.

Other stores under the same grade tree carry the retired task too
(``controls_glm`` / ``controls_halfdose`` / ``controls_nemotron_complete`` /
``controls_prompted``): this check reads the ``none`` legs of the views it is
POINTED AT (``--retired-root`` + ``--models``), never a hardcoded tree; a
widening beyond the ``none`` legs is a new invocation, not a code change.

llama-paper-rows (P12): the check presupposes a thinking-OFF arm — the stock instrument's OFF greedy 512 leg against
the retired OFF probe. A model whose family has NO thinking switch (``contracts.NO_THINKING_CHANNEL_FAMILIES``; ``--models
llama8b``) has no such arm, and the retired probe carried no rows of it, so its block is an EXPLICIT ``not_applicable``
record (:func:`not_applicable_block`: the schema, the status, ONE sentence saying why, the pin, the model key, the
family) in the same ``models`` map the switch families' real blocks live in — never a missing block, never a fabricated
leg. No store is read for it (a stock store under ``overlap/<key>/`` is noted, not compared); a switch family never
gets one. The analysis' gate A10 admits it for exactly those families.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sisteer.capeval.manifest import sha256_hex16
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load_module("tinycap_gsm8k_validators", _HERE / "tinycap_gsm8k_validators.py")
M = V.M
T = V.T
AnalysisError = V.AnalysisError

SCHEMA = "gsm8k_overlap/1"
TOLERANCE_PP = 5.0
TOLERANCE = TOLERANCE_PP / 100.0
VERDICT_CONTINUATION, VERDICT_NEW_INSTRUMENT = "continuation", "new_instrument"
CONTRAST = "stock_minus_retired"
RETIRED_GEN_EXPECTED: Mapping[str, Any] = {"do_sample": False, "max_new_tokens": 512}
MATCH_EXACT, MATCH_REORDERED, MATCH_NONE = "matched_exact", "matched_content_reordered", "content_mismatch"
MODELS_DEFAULT: tuple[str, ...] = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash")
EXIT_REFUSED = 3
#: llama-paper-rows (P12): the NO-SWITCH family's block status (the analysis' `gate_overlap` reads this literal — pinned equal by test)
STATUS_NOT_APPLICABLE = "not_applicable"
NOT_APPLICABLE_WHY = ("no thinking-OFF arm exists for a family with no thinking switch (thinking: not available), so there is nothing to "
                      "overlap-check against the retired OFF probe, which carried no rows of this model")


def is_no_switch_model(key: str) -> bool:
    """Whether ``key``'s family has NO thinking switch (``contracts.NO_THINKING_CHANNEL_FAMILIES``)."""
    return C.FAMILY_BY_MODEL_KEY.get(str(key)) in C.NO_THINKING_CHANNEL_FAMILIES


def not_applicable_block(key: str, *, store_pin: str, stock_root: Path | None = None) -> dict[str, Any]:
    """The NO-SWITCH family's overlap block (P12): ``{schema, status: not_applicable, why, pin, model_key, family}`` — the
    analysis' A10 reads exactly these; ``stock_store_present`` notes (never compares) a stock store under the overlap root."""
    if not is_no_switch_model(key):
        raise AnalysisError(f"{key!r} (family {C.FAMILY_BY_MODEL_KEY.get(key)!r}) has a thinking switch — the overlap check applies; "
                            "a not_applicable block is for a no-switch family only (P12)")
    present = bool(stock_root is not None and (Path(stock_root) / key).is_dir())
    return {"schema": SCHEMA, "status": STATUS_NOT_APPLICABLE, "why": NOT_APPLICABLE_WHY, "pin": str(store_pin), "model_key": key,
            "family": C.FAMILY_BY_MODEL_KEY.get(key), "stock_store_present": present, "usable": None, "comparison": None, "verdict": None,
            "reason": NOT_APPLICABLE_WHY}


def retired_id(i: int, question: str) -> str:
    """``parity._content_id("g", i, question)`` — recomputed here from the text."""
    return f"g{i:04d}-{sha256_hex16(question.encode('utf-8'))[:8]}"


def derive_pairing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """From the retired instrument's selection rows (``{item_id, question, answer}``
    in selection order) the two ids per row, both from the text: the row's own
    id must equal :func:`retired_id` (else the rows are not the retired selection)."""
    entries = []
    for i, r in enumerate(rows):
        q = str(r["question"])
        expect = retired_id(i, q)
        if str(r["item_id"]) != expect:
            raise AnalysisError(f"pairing rows: row {i} id {r['item_id']!r} != _content_id over its own question {expect!r} — "
                                "not the retired instrument's selection")
        entries.append({"position": i, "retired_id": expect, "inspect_id": C.gsm8k_sample_id(q), "suffix": expect.split("-", 1)[1],
                        "question_sha256": hashlib.sha256(q.encode("utf-8")).hexdigest()})
    by_inspect = {e["inspect_id"]: e for e in entries}
    if len(by_inspect) != len(entries):
        raise AnalysisError("pairing rows: two questions share an inspect id")
    return {"n_rows": len(entries), "entries": entries, "by_inspect_id": by_inspect,
            "by_suffix": {e["suffix"]: e for e in entries}}


def platinum_rows_from_plan(repo_root: Path) -> list[dict[str, Any]]:
    """The retired selection under the pinned revision, in this venv (needs ``datasets``)."""
    from sisteer.capeval.corpora import load_pins
    from sisteer.capeval.tiers.parity import PARITY_N, gsm8k_parity_rows

    return [dict(r) for r in gsm8k_parity_rows(load_pins(Path(repo_root) / "configs" / "capeval" / "pins.yaml"), PARITY_N)]


def classify_retired_ids(ids: Sequence[str], pairing: Mapping[str, Any]) -> dict[str, Any]:
    """Each retired ``acc`` row's id against the derivation: exact (suffix and
    position), reordered (suffix only), or a content mismatch (a FINDING)."""
    by_suffix = pairing["by_suffix"]
    by_retired = {e["retired_id"]: e for e in pairing["entries"]}
    classes: dict[str, str] = {}
    for rid in ids:
        if rid in by_retired:
            classes[rid] = MATCH_EXACT
            continue
        suffix = rid.split("-", 1)[1] if "-" in rid else ""
        classes[rid] = MATCH_REORDERED if suffix in by_suffix else MATCH_NONE
    counts = Counter(classes.values())
    return {"classes": classes, "n_matched_exact": counts.get(MATCH_EXACT, 0), "n_matched_content_reordered": counts.get(MATCH_REORDERED, 0),
            "n_content_mismatch": counts.get(MATCH_NONE, 0),
            "content_mismatch_ids": sorted(i for i, c in classes.items() if c == MATCH_NONE)[:20]}


def read_retired_store(retired_root: Path, view: str) -> dict[str, Any]:
    """The retired ``none`` store of one view: its identity (``resolved_config.gen``
    must be the OFF greedy 512 the spec assumes — read BEFORE any comparison),
    the 200 ``acc`` rows by id, the ``extraction_failed`` rows."""
    cells = sorted(p for p in (Path(retired_root) / view / "cells").glob("none__controls__*") if (p / "rows.jsonl").is_file()) \
        if (Path(retired_root) / view / "cells").is_dir() else []
    if len(cells) != 1:
        return {"available": False, "view": view, "reason": f"{len(cells)} retired none store(s) under {Path(retired_root) / view / 'cells'} (need exactly 1)"}
    d = cells[0]
    manifest = completion.read_store_manifest(d) or {}
    gen = T._mapping(T._mapping(manifest.get("resolved_config")).get("gen"))
    problems = [f"resolved_config.gen.{k} {gen.get(k)!r} != {v!r}" for k, v in RETIRED_GEN_EXPECTED.items() if gen.get(k) != v]
    acc: dict[str, float] = {}
    failed: dict[str, float] = {}
    n_rows = 0
    for r in T._iter_jsonl(d / "rows.jsonl"):
        if r.get("task") != V.RETIRED_TASK or r.get("item_id") == C.AGG_ITEM_ID:
            continue
        n_rows += 1
        if r.get("domain") != V.RETIRED_DOMAIN:
            problems.append(f"row {r.get('item_id')!r} domain {r.get('domain')!r} != {V.RETIRED_DOMAIN!r}")
        if r.get("metric") == C.METRIC_ACC:
            if r["item_id"] in acc:
                problems.append(f"acc row {r['item_id']!r} twice")
            if r.get("value") not in (0, 1, 0.0, 1.0):
                problems.append(f"acc row {r['item_id']!r} value {r.get('value')!r} not in {{0, 1}}")
            acc[str(r["item_id"])] = float(r["value"])
        elif r.get("metric") == "extraction_failed":
            failed[str(r["item_id"])] = float(r.get("value") or 0.0)
    if len(acc) != V.N_ITEMS:
        problems.append(f"{len(acc)} acc rows of {V.RETIRED_TASK}, not {V.N_ITEMS}")
    return {"available": not problems, "view": view, "cell_dir": str(d), "reason": "; ".join(problems) or None, "gen": dict(gen),
            "model_id": manifest.get("model_id"), "model_revision": manifest.get("model_revision"), "n_task_rows": n_rows,
            "acc": acc, "extraction_failed": failed, "n_extraction_failed": sum(1 for v in failed.values() if v == 1.0)}


def read_stock_store(stock_root: Path, key: str, *, pairing: Mapping[str, Any], canonical_ids: Sequence[str], repo_root: Path,
                     store_pin: str, eval_logs: bool) -> dict[str, Any]:
    """The stock OFF greedy 512 leg of one model: the store's identity through
    the parameterized validator (task, cap 512, arm OFF, greedy row, the model
    pin), its producer held to the STORE PIN (the wave sha, never the
    manifest's own), the outcomes (every draw scored), the rows' ``paired_id``
    held equal to the derivation, and — ``eval_logs`` — the sample inputs' two
    hashes off the one qualifying log, bound to the accepted attempt's window."""
    root = Path(stock_root) / key / "none" / "greedy_off512"
    cd, manifest, why = V.read_store(root)
    if cd is None or manifest is None:
        return {"available": False, "reason": why, "out_root": str(root)}
    summary = completion.read_task_summary(cd, V.TASK)
    if summary is None:
        return {"available": False, "reason": f"no summary for {V.TASK} under {cd}", "out_root": str(root)}
    row = {"model_key": key, "view": V.VIEW_MAIN, "tag": key, "cell": V.CELL_NONE, "cell_slug": V.CELL_NONE, "fitted": False,
           "declared": False, "family": None, "sampling_mode": V.OVERLAP_SAMPLING, "object_sha256": None, "tuple_name": None,
           "served_positions_expected": None}
    fact = {"off_ids": list(T._mapping(manifest.get(C.MANIFEST_KEY_TINYCAP)).get(C.IDENTITY_KEY_THINKING_HEADER_IDS, {}).get("off", [])),
            "on_ids": list(T._mapping(manifest.get(C.MANIFEST_KEY_TINYCAP)).get(C.IDENTITY_KEY_THINKING_HEADER_IDS, {}).get("on", []))}
    greedy = {"temperature": C.GREEDY_SAMPLING.temperature}
    problem = V.identity_problem(manifest, summary, row=row, cap=V.OVERLAP_CAP, model_pin=M.model_pin(repo_root, key), sampling_row=greedy,
                                 fact=fact, thinking_on=False, sampling_mode=V.OVERLAP_SAMPLING)
    if problem:
        return {"available": False, "reason": f"identity: {problem}", "out_root": str(root)}
    if summary.get("thinking") != C.THINKING_OFF or summary.get("max_tokens") != V.OVERLAP_CAP or summary.get("task") != V.TASK:
        return {"available": False, "reason": f"summary thinking / max_tokens / task {summary.get('thinking')!r} / {summary.get('max_tokens')!r} / "
                                              f"{summary.get('task')!r} != off / {V.OVERLAP_CAP} / {V.TASK!r}", "out_root": str(root)}
    tree = T._mapping(manifest.get(C.MANIFEST_KEY_PRODUCER)).get("tree_sha")
    problem = V.producer_problem(manifest, pin=store_pin)
    if problem:
        return {"available": False, "reason": f"producer: {problem}", "out_root": str(root), "producer_tree_sha": tree, "store_pin": store_pin}
    try:
        outcomes = completion.read_task_outcomes(cd, V.TASK)
        completion.verify_outcomes_against_summary(outcomes, summary)
    except Exception as e:      # noqa: BLE001 — the reader's refusal is the reason
        return {"available": False, "reason": f"outcomes: {e}", "out_root": str(root), "producer_tree_sha": tree, "store_pin": store_pin}
    if outcomes is None:
        return {"available": False, "reason": f"no {C.TASK_OUTCOMES_DIR}/{C.task_outcomes_filename(V.TASK)} under {cd} — the all-attempted record is "
                                              "required", "out_root": str(root), "producer_tree_sha": tree, "store_pin": store_pin}
    # the all-attempted record's OWN producer, held to the manifest's (itself held to the pin above) BEFORE a single value
    # is read off it: the summary compare above reads shape and counts, the pairing check below
    # reads ids — neither says which tree wrote the values; an off-pin outcomes file beside a pinned manifest is refused
    # here whether or not the .eval log is read
    header = {k: v for k, v in outcomes.items() if k != "items"}
    problem = V.outcomes_producer_problem(header, manifest_sha=str(tree))
    if problem:
        return {"available": False, "reason": f"outcomes producer: {problem}", "out_root": str(root), "producer_tree_sha": tree,
                "outcomes_producer_tree_sha": header.get("producer_tree_sha"), "store_pin": store_pin}
    items = {str(it["item_id"]): it for it in outcomes["items"]}
    canon = {str(i) for i in canonical_ids}
    if set(items) != canon:
        return {"available": False, "reason": f"the stock leg's attempted ids are not the canonical {len(canon)} (have {len(items)})", "out_root": str(root)}
    unscored = sorted(i for i, it in items.items() if not it["scored"])
    if unscored:
        return {"available": False, "reason": f"{len(unscored)} thinking-OFF draw(s) unscored (first {unscored[:3]}) — an OFF greedy draw always scores",
                "out_root": str(root)}
    rows = T._rows_for_task(cd, V.TASK)
    if len(rows) != len(items):
        return {"available": False, "reason": f"rows.jsonl holds {len(rows)} rows != {len(items)} attempted", "out_root": str(root)}
    by_inspect = pairing["by_inspect_id"]
    bad_pair = [(r["item_id"], r["_extra"].get("paired_id"), (by_inspect.get(str(r["item_id"])) or {}).get("retired_id")) for r in rows
                if (by_inspect.get(str(r["item_id"])) or {}).get("retired_id") != r["_extra"].get("paired_id")]
    if bad_pair:
        return {"available": False, "reason": f"{len(bad_pair)} stock row(s) whose producer paired_id disagrees with the derivation "
                                              f"(first {bad_pair[:2]})", "out_root": str(root)}
    text_check: dict[str, Any] = {"run": False}
    if eval_logs:
        # FAIL CLOSED: --eval-logs asks for the byte check on the stock side, so the check must RUN — a missing, unreadable or
        # non-qualifying log, a disagreeing re-derivation, or fewer than the 200 attempted inputs re-derived makes the leg unusable
        ec = V.eval_cross_check(cd, {k: v for k, v in outcomes.items() if k != "items"}, outcomes["items"], max_tokens=V.OVERLAP_CAP,
                                accepted_attempt=V.accepted_attempt(manifest))
        inputs = ec.get("inputs") or {}
        text_check = {"run": True, "available": ec["available"], "reason": ec.get("reason"), "agrees": ec.get("agrees"),
                      "log_present": ec.get("log_present"), "selected_logs": ec.get("selected_logs"), "attempt_binding": ec.get("attempt_binding"),
                      "n_inputs": len(inputs), "n_required": V.N_ITEMS}
        refused = {"available": False, "out_root": str(root), "text_check": text_check}
        if not ec["available"]:
            return {**refused, "reason": (f"--eval-logs: the stock leg's .eval re-derivation did not run ({ec.get('reason')}) — the sample "
                                          "inputs cannot be hashed, so the byte check on the stock side is not made; the leg is unusable")}
        if not ec["agrees"]:
            return {**refused, "reason": f"the .eval re-derivation disagrees with the record: {ec.get('detail')}"}
        mism_id = [i for i, q in inputs.items() if C.gsm8k_sample_id(q) != i]
        mism_suffix = [i for i, q in inputs.items() if (by_inspect.get(i) or {}).get("suffix") != sha256_hex16(q.encode("utf-8"))[:8]]
        text_check.update({"n_inspect_id_mismatch": len(mism_id), "n_retired_suffix_mismatch": len(mism_suffix),
                           "first_mismatches": (mism_id + mism_suffix)[:5]})
        if set(inputs) != set(items) or len(inputs) != V.N_ITEMS:
            return {**refused, "reason": (f"--eval-logs: {len(inputs)} sample input(s) re-derived from the .eval log, not the {V.N_ITEMS} "
                                          f"attempted ({len(set(items) - set(inputs))} attempted id(s) without an input, "
                                          f"{len(set(inputs) - set(items))} input(s) for ids never attempted) — the byte check is incomplete")}
        if mism_id or mism_suffix:
            return {**refused, "reason": (f"the stock leg's sample inputs do not hash to their ids ({len(mism_id)} inspect-id / "
                                          f"{len(mism_suffix)} retired-suffix mismatches over {len(inputs)} inputs)")}
        text_check["validated"] = True
    values = {i: float(it["value"]) for i, it in items.items()}
    return {"available": True, "reason": None, "out_root": str(root), "cell_dir": str(cd), "values": values,
            "n_unparseable": summary.get("n_unparseable"), "n_at_cap": summary.get("n_at_cap"), "acc_summary": summary.get("acc"),
            "producer_tree_sha": tree, "outcomes_producer_tree_sha": header.get("producer_tree_sha"), "store_pin": store_pin,
            "text_check": text_check}


def compare(stock: Mapping[str, float], retired: Mapping[str, float], pairing: Mapping[str, Any], *, n_resamples: int, seed: int,
            ci: float) -> dict[str, Any]:
    """The paired statistics over the items both legs carry (stock inspect id ->
    the derivation's retired id -> the retired ``acc`` row)."""
    by_inspect = pairing["by_inspect_id"]
    pairs: dict[str, tuple[float, float]] = {}
    unpaired: list[str] = []
    for i, v in stock.items():
        rid = (by_inspect.get(i) or {}).get("retired_id")
        if rid is not None and rid in retired:
            pairs[i] = (v, retired[rid])
        else:
            unpaired.append(i)
    n = len(pairs)
    out: dict[str, Any] = {"n_pairs": n, "n_required": V.N_ITEMS, "complete": n == V.N_ITEMS, "n_stock_unpaired": len(unpaired),
                           "n_retired_unpaired": len(set(retired) - {(by_inspect.get(i) or {}).get("retired_id") for i in pairs}),
                           "first_unpaired": sorted(unpaired)[:5]}
    if n == 0:
        out.update({"acc_stock": None, "acc_retired": None, "delta": None, "bootstrap": None, "mcnemar": None, "verdict": None,
                    "reason": "no paired item"})
        return out
    from sisteer.eval.realign_stats import mcnemar

    legs = {"stock": {i: (1, s) for i, (s, _r) in pairs.items()}, "retired": {i: (1, r) for i, (_s, r) in pairs.items()}}
    boot = M.paired_bootstrap_conditional(legs, {CONTRAST: {"stock": +1, "retired": -1}}, n_resamples=n_resamples, seed=seed, ci=ci)
    st = boot["stats"][CONTRAST]
    mc = mcnemar({i: int(s) for i, (s, _r) in pairs.items()}, {i: int(r) for i, (_s, r) in pairs.items()})
    b = sum(1 for s, r in pairs.values() if s == 1.0 and r == 0.0)
    c = sum(1 for s, r in pairs.values() if s == 0.0 and r == 1.0)
    inside = st["ci_low"] > -TOLERANCE and st["ci_high"] < TOLERANCE
    half_width = (st["ci_high"] - st["ci_low"]) / 2.0
    out.update({
        "acc_stock": boot["acc_by_leg"]["stock"]["point"], "acc_retired": boot["acc_by_leg"]["retired"]["point"],
        "delta": st["point"], "ci_low": st["ci_low"], "ci_high": st["ci_high"], "ci_half_width": half_width,
        "bootstrap": {"B": boot["B"], "seed": boot["seed"], "level": boot["level"], "n_items": boot["n_items"],
                      "replicate_signs": st["replicate_signs"]},
        "mcnemar": {"p_value": mc["p_value"], "n_items": mc["n_items"], "stock_only_correct": mc["a_only"], "retired_only_correct": mc["b_only"],
                    "n_discordant": mc["n_discordant"], "label": "exact two-sided McNemar on the discordant pairs"},
        "disagreement_b_plus_c": b + c, "n_stock_correct_retired_wrong": b, "n_stock_wrong_retired_correct": c,
        "n_both_correct": sum(1 for s, r in pairs.values() if s == 1.0 and r == 1.0),
        "n_both_wrong": sum(1 for s, r in pairs.values() if s == 0.0 and r == 0.0),
        "tolerance_pp": TOLERANCE_PP, "ci_inside_tolerance": inside,
        "verdict": (VERDICT_CONTINUATION if (inside and n == V.N_ITEMS) else VERDICT_NEW_INSTRUMENT),
        "reason": (None if n == V.N_ITEMS else f"only {n} of {V.N_ITEMS} items paired — the two instruments do not share the selection"),
        "power_note": (f"at n = {n} the paired 95 % CI half-width is {100 * half_width:.1f} pp against a ±{TOLERANCE_PP:.0f} pp tolerance: "
                       "a 'continuation' verdict here is a CI-inside-tolerance statement about THIS sample, not an equivalence proof; "
                       "a 'new instrument' verdict may be power-limited as much as a real gap (plan §13 C8)"),
    })
    return out


def agreement(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    """Per-item agreement of two retired stores of one model (main vs it) — recorded, never pooled."""
    ids = sorted(set(a) & set(b))
    agree = sum(1 for i in ids if a[i] == b[i])
    return {"n_common": len(ids), "n_agree": agree, "n_disagree": len(ids) - agree, "agreement": (agree / len(ids)) if ids else None,
            "acc_a": (sum(a[i] for i in ids) / len(ids)) if ids else None, "acc_b": (sum(b[i] for i in ids) / len(ids)) if ids else None,
            "note": "the same model's retired reading under the it view — a re-run, not a second instrument; never pooled (C9)"}


def run(*, retired_root: Path, stock_root: Path, models: Sequence[str], canonical_ids: Sequence[str], ids_path: Path | None,
        pairing_rows: Sequence[Mapping[str, Any]], pairing_source: str, repo_root: Path, store_pin: str, n_resamples: int, seed: int,
        ci: float, eval_logs: bool) -> dict[str, Any]:
    problem = M.pin_problem(store_pin)
    if problem:
        raise AnalysisError(f"--sha: {problem}")
    pairing = derive_pairing(pairing_rows)
    canon = {str(i) for i in canonical_ids}
    in_selection = canon & set(pairing["by_inspect_id"])
    blocks: dict[str, Any] = {}
    for key in models:
        if is_no_switch_model(key):
            # P12: no thinking-OFF arm exists — the EXPLICIT not_applicable record, no store read, never a real block
            blocks[key] = not_applicable_block(key, store_pin=store_pin, stock_root=stock_root)
            continue
        retired = read_retired_store(retired_root, key)
        stock = read_stock_store(stock_root, key, pairing=pairing, canonical_ids=canonical_ids, repo_root=repo_root, store_pin=store_pin,
                                 eval_logs=eval_logs)
        block: dict[str, Any] = {"model_key": key, "retired": {k: v for k, v in retired.items() if k not in ("acc", "extraction_failed")},
                                 "stock": {k: v for k, v in stock.items() if k != "values"}, "usable": retired["available"] and stock["available"]}
        if retired["available"]:
            block["retired_id_match"] = classify_retired_ids(sorted(retired["acc"]), pairing)
        it_view = f"{key}_{V.VIEW_IT}"
        if (Path(retired_root) / it_view).is_dir():
            it_store = read_retired_store(retired_root, it_view)
            block["it_view_retired"] = {k: v for k, v in it_store.items() if k not in ("acc", "extraction_failed")}
            if it_store["available"] and retired["available"]:
                block["it_view_retired"]["agreement_with_main"] = agreement(retired["acc"], it_store["acc"])
        if block["usable"]:
            block["comparison"] = compare(stock["values"], retired["acc"], pairing, n_resamples=n_resamples, seed=seed, ci=ci)
            block["verdict"] = block["comparison"]["verdict"]
            block["reason"] = block["comparison"]["reason"]
        else:
            block["comparison"] = None
            block["verdict"] = None
            block["reason"] = "; ".join(x for x in (retired.get("reason"), stock.get("reason")) if x)
        blocks[key] = block
    return {"schema": SCHEMA, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task_stock": V.TASK, "task_retired": V.RETIRED_TASK, "decode_settings": {"thinking": C.THINKING_OFF, "sampling": V.OVERLAP_SAMPLING,
                                                                                        "max_tokens": V.OVERLAP_CAP},
            "store_pin": store_pin, "ids_path": (str(ids_path) if ids_path else None),
            "ids_sha256": hashlib.sha256(json.dumps(sorted(canon)).encode("utf-8")).hexdigest(), "n_canonical": len(canon),
            "pairing": {"source": pairing_source, "rule": "item_id: retired g<i:04d>-<sha256(question)[:8]> vs stock gsm8k_<md5(question)[:8]>, "
                                                          "both recomputed from the retired selection's question text; suffix = content, prefix = position",
                        "n_rows": pairing["n_rows"], "n_canonical_in_selection": len(in_selection),
                        "n_canonical_outside_selection": len(canon - in_selection)},
            "tolerance_pp": TOLERANCE_PP, "bootstrap": {"B": n_resamples, "seed": seed, "level": ci},
            "retired_root": str(retired_root), "stock_root": str(stock_root), "models": blocks,
            "note": "two instruments, never pooled: the retired s2parity reading (OFF, greedy, 512) and the stock inspect task's OFF greedy 512 leg"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retired-root", type=Path, required=True, help="the graded controls store root: <root>/<view>/cells/none__controls__*/")
    ap.add_argument("--stock-root", type=Path, required=True, help="the overlap legs' root: <root>/<model>/none/greedy_off512/")
    ap.add_argument("--ids", type=Path, required=True, help="the canonical 200 ids (a JSON list) the stock task ran")
    ap.add_argument("--sha", required=True, help="the store pin: the wave sha every stock leg's producer tree carries (run E's --sha)")
    ap.add_argument("--pairing", default="plan", help="'plan' (derive the retired selection under the pinned revision in this venv) or a JSON "
                                                      "list of {item_id, question, answer} rows in selection order")
    ap.add_argument("--models", default=",".join(MODELS_DEFAULT))
    ap.add_argument("--repo-root", type=Path, default=_HERE.parent)
    ap.add_argument("--bootstrap", type=int, default=10_000, metavar="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--eval-logs", action="store_true", help="read the stock legs' .eval logs and hash every sample input (inspect venv)")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    ids = T._read_json(a.ids)
    if not isinstance(ids, list) or len({str(i) for i in ids}) != V.N_ITEMS:
        raise AnalysisError(f"--ids {a.ids}: expected a JSON list of {V.N_ITEMS} unique ids")
    if a.pairing == "plan":
        rows, source = platinum_rows_from_plan(a.repo_root), "parity.gsm8k_parity_rows under configs/capeval/pins.yaml (this venv)"
    else:
        rows = T._read_json(Path(a.pairing))
        source = f"rows file {a.pairing}"
        if not isinstance(rows, list):
            raise AnalysisError(f"--pairing {a.pairing}: expected a JSON list of rows")
    report = run(retired_root=a.retired_root, stock_root=a.stock_root, models=[m for m in a.models.split(",") if m], canonical_ids=[str(i) for i in ids],
                 ids_path=a.ids, pairing_rows=rows, pairing_source=source, repo_root=a.repo_root, store_pin=a.sha, n_resamples=a.bootstrap,
                 seed=a.seed, ci=a.ci, eval_logs=a.eval_logs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    bad = 0
    for key, b in report["models"].items():
        cmp_ = b.get("comparison") or {}
        if b.get("status") == STATUS_NOT_APPLICABLE:
            print(f"[overlap] {key}: not applicable — {b['why']}")
            continue
        if b["usable"] and cmp_.get("complete"):
            print(f"[overlap] {key}: {cmp_['n_pairs']} pairs; acc retired {cmp_['acc_retired']:.3f} stock {cmp_['acc_stock']:.3f}; "
                  f"delta {100 * cmp_['delta']:+.1f} pp CI [{100 * cmp_['ci_low']:+.1f}, {100 * cmp_['ci_high']:+.1f}]; "
                  f"McNemar p {cmp_['mcnemar']['p_value']}; b+c {cmp_['disagreement_b_plus_c']} -> {b['verdict']}")
        else:
            bad += 1
            print(f"[overlap] {key}: NOT usable / incomplete — {b.get('reason')}", file=sys.stderr)
    print(f"[overlap] record -> {a.out}")
    return EXIT_REFUSED if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
