#!/usr/bin/env python
"""lens_stacked_ab — the served A/B of stacked vs per-row reductions inside
the lens hook, on ONE GPU and one tree, per family (branch
lens-grouped-scope, 2026-09-04). Scoping evidence, never a certification.

THE QUESTION. ``lens_adapters._row_reduction`` issues every reduction inside
a vectorized decode group as ONE stacked call by default (the user's decision
of 2026-09-04; ``SISTEER_LENS_STACKED_REDUCTIONS=0`` restores the per-row
call, bit-identical to the per-request one). Job array 382416 measured the
stacked call (75efb47) faster and NOT bit-identical to the per-row one: one
bf16 ulp here and there in the residual stream. Does that
change served tokens MORE than the batch-composition variance production
already accepts at VLLM_BATCH_INVARIANT=0 (exact replay dropped, 2026-08-27)?
Every cross-server token comparison at B > 1 is composition-confounded
(lens_vectorize_gate's module docstring), so this script measures the
stacked effect NEXT TO its own noise floor, on the same node, with the same
prompts, in the same session:

  THREE STANDUPS PER CASE, all from this tree, all at production BI=0,
  prefix caching off, through scripts/lens_gate_standup.py — the mode switch
  is SET explicitly on every side (an unset switch now means stacked, so
  "flag unset" is no longer the per-row side):
    per_row         SISTEER_LENS_STACKED_REDUCTIONS=0 apply_path "vectorized+per_row"
    stacked         SISTEER_LENS_STACKED_REDUCTIONS=1 apply_path "vectorized+stacked"
    stacked_shadow  =1 + SISTEER_LENS_SHADOW_COMPARE=1: every group
                    runs BOTH arithmetics in the worker on identical rows,
                    the ShadowCompareRecord counts the unequal groups per
                    (kind, op, shape, positions, ..., rows) with max |diff|,
                    max ulps at the output's own magnitude, the gate's
                    per-class inputs (the edit-excess unit for linear ops;
                    the gate-scalar ratios for gated ops), the elements
                    compared, and the reference magnitude, and the PER-ROW
                    result is served — so this standup is ALSO a second
                    per-row server. The report also files the gate's own
                    contract verdict on that record (lens_vectorize_gate.
                    reduction_tolerance under the stacked mode).
  WAVES: at every B in --batches, --repeats identical greedy waves of B
  steered requests (mixed prompt lengths, ignore_eos so every request
  decodes the same number of steps) on each standup; one B=1 singleton per
  standup (the same code path at one row: stacked must equal per_row
  exactly — a hard sanity check).
  COMPARISONS, per B, as the fraction of requests whose 64-token output is
  identical (token ids AND every logprob; token ids alone in parentheses)
  and the first divergence position:
    per_row self       run_i vs run_j on the per_row server  (same-server floor)
    per_row x server   per_row run_i vs stacked_shadow run_j (cross-server
                       floor: two processes, both serving per-row arithmetic)
    stacked self       run_i vs run_j on the stacked server
    stacked vs per_row stacked run_i vs per_row run_j (the effect + the
                       cross-server floor)
  If "stacked vs per_row" agrees no less often than "per_row x server", the
  stacked deviation is inside the composition noise; if it agrees markedly
  less, it is not. The in-worker shadow record is the confound-free
  ELEMENTWISE reading of the same standup.
  TIMING: ms/step at --timing-batches on the per_row and stacked standups
  (the gate's closed-loop sampler), the production speedup of the flag.

ADMISSIONS. Composition is not controlled (only held as similar as the
scheduler allows: same prompts, same B, same node, one client); the floor
is measured, not removed. A clean token-level measurement would need the
two arithmetics on the SAME batch under the same scheduling — an in-process
``LLM.generate`` twin run, or a worker-side twin decode — and is out of
scope here. Results: --results-root/<model>/{lens_stacked_ab.json,report.md}.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lens_vectorize_gate as g  # noqa: E402
from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.capeval.serving import lens_worker as lw  # noqa: E402

STACKED_ENV = la.STACKED_REDUCTIONS_ENV
PER_ROW, STACKED, STACKED_SHADOW = "per_row", "stacked", "stacked_shadow"
#: side -> (env_extra, the apply path the worker must attest). The mode switch
#: is set on EVERY side (stacked is the default when unset, so per-row must say
#: so); the paths are the worker's own (lens_worker.apply_path_string).
SIDES: Dict[str, Tuple[Optional[Dict[str, str]], str]] = {
    PER_ROW: ({STACKED_ENV: "0"}, lw.APPLY_PATH_VECTORIZED_PER_ROW),
    STACKED: ({STACKED_ENV: "1"}, lw.APPLY_PATH_VECTORIZED_STACKED),
    STACKED_SHADOW: ({STACKED_ENV: "1", g.SHADOW_ENV: "1"}, lw.APPLY_PATH_SHADOW),
}
#: the reduction mode each side is SET to (what its endpoint must attest)
SIDE_MODE: Dict[str, str] = {PER_ROW: g.REDUCTION_MODE_PER_ROW, STACKED: g.REDUCTION_MODE_STACKED,
                             STACKED_SHADOW: g.REDUCTION_MODE_STACKED}
DEFAULT_CASES = ("psr:relu_gated_vector:gemv:response:skipbos0:both",
                 "lora_all:lora_oproj:rankK:response:skipbos0:both")
DEFAULT_RESULTS_ROOT = "$SHARED/exp/lens-stacked-scope/ab"   # under the cluster's shared root; expanded through sisteer.site
COMPARISONS = ("per_row self", "per_row x server", "stacked self", "stacked vs per_row",
               "stacked vs per_row(shadow server)")


# --------------------------------------------------------------------------- #
# pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def agreement(run_a: Sequence[Mapping[str, Any]], run_b: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Request-wise agreement of two runs of the SAME wave (gate signatures):
    how many requests are identical in token ids and every logprob
    (``same_all``), in token ids alone (``same_tokens``), and where each
    differing request first diverged (token position; ``("lp", pos)`` when
    only a logprob differed)."""
    if len(run_a) != len(run_b):
        raise g.GateError(f"agreement over runs of different size: {len(run_a)} vs {len(run_b)}")
    same_all = same_tok = 0
    first_div: List[Any] = []
    for x, y in zip(run_a, run_b):
        tok_eq = x.get("token_ids") == y.get("token_ids")
        all_eq = tok_eq and x.get("logprobs") == y.get("logprobs") and x.get("top") == y.get("top")
        same_tok += int(tok_eq)
        same_all += int(all_eq)
        if not tok_eq:
            xs, ys = x.get("token_ids") or [], y.get("token_ids") or []
            first_div.append(next((j for j, (u, v) in enumerate(zip(xs, ys)) if u != v), min(len(xs), len(ys))))
        elif not all_eq:
            xs, ys = x.get("logprobs") or [], y.get("logprobs") or []
            first_div.append(("lp", next((j for j, (u, v) in enumerate(zip(xs, ys)) if u != v), None)))
    return {"n": len(run_a), "same_all": same_all, "same_tokens": same_tok, "first_div": first_div}


def pairwise(runs: Sequence[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Every unordered pair of runs on one server."""
    out = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            if runs[i] and runs[j]:
                out.append(agreement(runs[i], runs[j]))
    return out


def cross(runs_a: Sequence[Sequence[Mapping[str, Any]]],
          runs_b: Sequence[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Every (run on A, run on B) pair."""
    return [agreement(a, b) for a in runs_a for b in runs_b if a and b]


def pool(agreements: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pooled over pairs: requests compared, fraction identical (all /
    tokens), median first token divergence among token-differing requests."""
    n = sum(int(a["n"]) for a in agreements)
    sa = sum(int(a["same_all"]) for a in agreements)
    stk = sum(int(a["same_tokens"]) for a in agreements)
    tok_divs = [d for a in agreements for d in a["first_div"] if not isinstance(d, tuple)]
    return {"pairs": len(agreements), "requests": n,
            "same_all": sa, "same_tokens": stk,
            "frac_same_all": (sa / n) if n else None, "frac_same_tokens": (stk / n) if n else None,
            "n_token_divergent": len(tok_divs),
            "median_first_token_div": (statistics.median(tok_divs) if tok_divs else None),
            "min_first_token_div": (min(tok_divs) if tok_divs else None)}


def shadow_rows_table(snapshot: Any) -> List[Dict[str, Any]]:
    """The ShadowCompareRecord's by_key entries as rows sorted by group
    size: key, rows, compared, unequal, max |diff|, max ulps, max |ref|,
    differing elements — the elementwise deviation per kernel shape."""
    if not isinstance(snapshot, Mapping):
        return []
    rows = []
    for k, e in (snapshot.get("by_key") or {}).items():
        if not isinstance(e, Mapping):
            continue
        m = g._ROWS_TAIL.search(str(k))
        rows.append({"key": k, "rows": int(m.group(1)) if m else None,
                     "n_compared": int(e.get("n_compared") or 0), "n_unequal": int(e.get("n_unequal") or 0),
                     "n_unmodified": int(e.get("n_unmodified") or 0),
                     "max_abs_diff": e.get("max_abs_diff"), "max_ulps": e.get("max_ulps"),
                     "max_abs_ref": e.get("max_abs_ref"), "n_unequal_elems": e.get("n_unequal_elems"),
                     # the gate's tolerance fields (lens_vectorize_gate.reduction_tolerance)
                     "n_elems_compared": e.get("n_elems_compared"), "max_ulps_edit": e.get("max_ulps_edit"),
                     "n_flag_disagreements": e.get("n_flag_disagreements")})
    return sorted(rows, key=lambda r: (r["rows"] if r["rows"] is not None else -1, r["key"]))


def _f(x: Any, fmt: str = ".3f") -> str:
    if x is None:
        return "n/a"
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return str(x)


def render_report(doc: Mapping[str, Any]) -> str:
    L = [f"# lens_stacked_ab — {doc['model']['key']}  ({doc['model']['hf']} @ {doc['model']['rev'][:12]})", "",
         f"tree: {doc['tree']} @ {doc['sha'][:12]}   node {doc.get('node')}   job {doc.get('job_id')}   gpu {doc.get('gpu')}",
         f"geometry: max_model_len {doc['geometry']['max_model_len']} x max_num_seqs {doc['geometry']['max_num_seqs']}; "
         f"BI=0, prefix caching off; batches {doc['batches']} x {doc['repeats']} waves; max_tokens {doc['max_tokens']}, "
         f"ignore_eos {doc['ignore_eos']}", "",
         "Every B > 1 comparison is composition-confounded (module docstring): read `stacked vs per_row` AGAINST "
         "`per_row x server` (the cross-server per-row floor), never alone. Cells: fraction of requests whose full "
         "greedy output is identical in tokens AND logprobs (tokens alone in parentheses); `div` = median first "
         "token divergence among token-differing requests.", ""]
    for c in doc["cases"]:
        L.append(f"## {c['case_id']}")
        if c.get("error"):
            L += [f"ERROR: {c['error']}", ""]
            continue
        L.append(f"object: {c.get('objects')} ({c.get('cell')}); apply paths: "
                 + ", ".join(f"{s}={c['apply_path'].get(s)!r}" for s in SIDES) + "")
        b1 = c.get("b1_sanity") or {}
        L.append(f"B=1 sanity (one row = one code path): stacked == per_row: {b1.get('equal')}"
                 + (f" ({b1.get('detail')})" if b1.get("detail") else ""))
        L += ["", "| B | comparison | pairs | requests | identical (all) | identical (tokens) | div |", "|---:|---|---:|---:|---:|---:|---:|"]
        for B, cmp in (c.get("comparisons") or {}).items():
            for name in COMPARISONS:
                p = cmp.get(name)
                if not p:
                    continue
                L.append(f"| {B} | {name} | {p['pairs']} | {p['requests']} | {_f(p['frac_same_all'])} | "
                         f"({_f(p['frac_same_tokens'])}) | {_f(p['median_first_token_div'], '.0f')} |")
        sh = c.get("shadow") or {}
        tol = g.reduction_tolerance(sh, reduction_mode=g.REDUCTION_MODE_STACKED)
        L += ["", f"in-worker shadow (stacked vs per_row on identical rows): {sh.get('n_groups_compared')} groups compared, "
                  f"{sh.get('n_unequal')} unequal, max |diff| {_f(sh.get('max_abs_diff'), '.3g')}, max ulps "
                  f"{_f(sh.get('max_ulps'), '.2f')}, max excess edit ulps (beyond one output ulp, in ulps of the per-row edit) {_f(sh.get('max_ulps_edit'), '.2f')}, "
                  f"max |ref| {_f(sh.get('max_abs_ref'), '.3g')}, "
                  f"{sh.get('n_unequal_elems')} of {sh.get('n_elems_compared')} elements differed",
              f"gate contract (stacked mode; lens_vectorize_gate.reduction_tolerance): "
              f"{'ok' if tol.get('ok') else ('n/a' if tol.get('ok') is None else 'FAIL')} — {tol.get('detail')}", "",
              "| key | rows | compared | unequal | max abs diff | max ulps | max ulps edit | max abs ref | elems differed | elems compared |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in c.get("shadow_rows") or []:
            L.append(f"| {r['key']} | {r['rows']} | {r['n_compared']} | {r['n_unequal']} | {_f(r['max_abs_diff'], '.3g')} | "
                     f"{_f(r['max_ulps'], '.2f')} | {_f(r.get('max_ulps_edit'), '.2f')} | {_f(r['max_abs_ref'], '.3g')} | "
                     f"{r['n_unequal_elems']} | {r.get('n_elems_compared')} |")
        tim = c.get("timing") or {}
        if tim:
            L += ["", "| side | " + " | ".join(f"B={B} ms/step" for B in doc["timing_batches"]) + " |",
                  "|---|" + "---:|" * len(doc["timing_batches"])]
            for side in (PER_ROW, STACKED):
                t = tim.get(side) or {}
                L.append(f"| {side} | " + " | ".join(_f((t.get(str(B)) or {}).get("step_ms"), ".1f") for B in doc["timing_batches"]) + " |")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# the drive                                                                    #
# --------------------------------------------------------------------------- #


def _steered_wave(ctx: g.GateContext, su: g.Standup, prompts, vm, B: int):
    return g.run_batch(su.proxy_url, g._batch_requests(prompts, ["steered"] * B, vm),
                       max_tokens=ctx.max_tokens, ignore_eos=ctx.ignore_eos, top_logprobs=ctx.top_logprobs)


def drive_side(ctx: g.GateContext, su: g.Standup, *, side: str, cell: str, prompts, timing: bool,
               say=g._say) -> Dict[str, Any]:
    vms = g.vmodels_of(su)
    hcell = g.harness_map_of(su.endpoint, [cell])[cell]
    vm = {"passthrough": vms.get("none"), "scale0": vms.get("none__lens0"), "steered": vms.get(hcell)}
    path = g.apply_path_of(su)
    want = SIDES[side][1]
    if path != want:
        raise g.GateError(f"{side} standup attests apply path {path!r}, expected {want!r} — refusing to compare")
    # the endpoint's own attestation of the mode this side was set to (the gate's standup-time rule)
    rm_ok, rm_detail = g.reduction_mode_endpoint_ok(su.endpoint, SIDE_MODE[side])
    if not rm_ok:
        raise g.GateError(f"{side} standup: {rm_detail}")
    rec: Dict[str, Any] = {"apply_path": path, "vmodels": vm, "harness_cell": hcell, "commit": su.commit,
                           "endpoint_lens_reduction_mode": su.endpoint.get("lens_reduction_mode"),
                           "endpoint_stacked_reductions_env": su.endpoint.get("stacked_reductions_env"),
                           "bi_attested": su.endpoint.get("bi_attested"), "runs": {}, "signature_errors": []}
    if side == STACKED_SHADOW:
        g.shadow_snapshot(su, reset=True)
    try:
        g.rpc(su, "sisteer_selection_record", [True])
    except g.RpcError as e:
        rec["evidence_error"] = str(e)
    # B=1 singleton (prompt 0): the one-row code path
    try:
        rec["b1"] = g.run_batch(su.proxy_url, [(vm["steered"], prompts[0][0], {})], max_tokens=ctx.max_tokens,
                                ignore_eos=ctx.ignore_eos, top_logprobs=ctx.top_logprobs)[0]
    except g.SignatureError as e:
        rec["signature_errors"].append(f"B1: {e}")
    for B in ctx.batches:
        runs = []
        for r in range(ctx.repeats):
            try:
                runs.append(_steered_wave(ctx, su, prompts, vm, B))
            except g.SignatureError as e:
                rec["signature_errors"].append(f"B{B} run {r + 1}: {e}")
                runs.append([])
        rec["runs"][str(B)] = runs
        say(f"    {side} B={B}: {len([x for x in runs if x])}/{ctx.repeats} waves; self-agreement "
            f"{pool(pairwise(runs)).get('frac_same_all')}")
    try:
        rec["evidence"] = g.selection_evidence(su)
    except g.RpcError as e:
        rec["evidence_error"] = str(e)
    if side == STACKED_SHADOW:
        rec["shadow"] = g.shadow_snapshot(su, reset=False)
        sn = rec["shadow"] if isinstance(rec["shadow"], Mapping) else {}
        say(f"    shadow: {sn.get('n_groups_compared')} compared, {sn.get('n_unequal')} unequal, "
            f"max |diff| {sn.get('max_abs_diff')}, max ulps {sn.get('max_ulps')}, rows {g.rows_compared(sn)}")
    if timing:
        rec["timing"] = g.measure_timing(ctx, su, vm["steered"], prompts[min(4, len(prompts) - 1)][0])
    return rec


def compare_sides(recs: Mapping[str, Mapping[str, Any]], batches: Sequence[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pr, st, sh = recs[PER_ROW]["runs"], recs[STACKED]["runs"], recs[STACKED_SHADOW]["runs"]
    for B in batches:
        k = str(B)
        out[k] = {"per_row self": pool(pairwise(pr.get(k, []))),
                  "per_row x server": pool(cross(pr.get(k, []), sh.get(k, []))),
                  "stacked self": pool(pairwise(st.get(k, []))),
                  "stacked vs per_row": pool(cross(st.get(k, []), pr.get(k, []))),
                  "stacked vs per_row(shadow server)": pool(cross(st.get(k, []), sh.get(k, [])))}
    return out


def b1_sanity(recs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    a, b = recs[PER_ROW].get("b1"), recs[STACKED].get("b1")
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return {"equal": None, "detail": "a B=1 singleton is missing"}
    ok, why = g.signatures_equal([a], [b])
    return {"equal": bool(ok), "detail": why,
            "note": "one row is one call on both paths; a difference here is scheduler/composition "
                    "noise between two singleton requests, not the flag"}


def run_ab(ctx: g.GateContext, *, cases: Sequence[str], inventory: Mapping[str, Mapping[str, Any]],
           bank_dir: Optional[Path], extra_dirs: Sequence[Path], candsi_dir: Optional[Path],
           extra_origins: Optional[Mapping[Path, str]], timing: bool) -> Dict[str, Any]:
    ms = ctx.ms
    sha = g.tree_facts(ctx.candidate)["sha"]
    doc: Dict[str, Any] = {
        "model": dataclasses.asdict(ms), "tree": str(ctx.candidate), "sha": sha,
        "geometry": {"max_model_len": ctx.max_model_len, "max_num_seqs": ctx.max_num_seqs},
        "batches": list(ctx.batches), "timing_batches": list(ctx.timing_batches), "repeats": ctx.repeats,
        "max_tokens": ctx.max_tokens, "ignore_eos": ctx.ignore_eos, "node": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": (g._nvidia_smi("--query-gpu=name,memory.total", "--format=csv,noheader") or "").strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "sides": {s: SIDES[s][1] for s in SIDES},
        "stacked_env": STACKED_ENV, "cases": []}
    path_json = ctx.results_dir / "lens_stacked_ab.json"
    g._write_json(path_json, doc)
    stage_root = ctx.results_dir / "objects"
    stage_root.mkdir(parents=True, exist_ok=True)
    header_len = g.header_len_for(ms.hf, ms.rev)
    dims_cache: Dict[str, Any] = {}

    def dims_fn():
        if not dims_cache:
            dims_cache.update(g.model_dims(ms.hf, ms.rev))
        return dims_cache
    prompts = None
    B_max = max(ctx.batches)
    for case_id in cases:
        kr: Dict[str, Any] = {"case_id": case_id}
        try:
            if case_id not in inventory:
                raise g.GateError(f"{case_id} is not in this family's served-op inventory: {sorted(inventory)}")
            obj = g.resolve_case_object(ms, case_id, list(inventory[case_id]["arms"]), bank_dir=bank_dir,
                                        extra_dirs=extra_dirs, candsi_dir=candsi_dir, stage_root=stage_root,
                                        header_len=header_len, dims_fn=dims_fn, allow_synthetic=True,
                                        extra_origins=extra_origins)
            kr.update(obj)
            g._say(f"case {case_id}: arm {obj['arm']} cell {obj['cell']} objects={obj['objects']}")
            recs: Dict[str, Any] = {}
            for side in (PER_ROW, STACKED, STACKED_SHADOW):
                su = None
                try:
                    su = g.start_standup(ctx, side=side, tree=ctx.candidate, expect_sha=sha, cell=obj["cell"],
                                         defenses_dir=Path(obj["defenses_dir"]),
                                         tag=f"{case_id.replace(':', '_')}__{side}", env_extra=SIDES[side][0])
                    if prompts is None:
                        vms = g.vmodels_of(su)
                        prompts = g.build_prompts(su.proxy_url, vms["none"], g.prompt_targets(B_max))
                        doc["prompt_tokens"] = [c for _m, c in prompts]
                    recs[side] = drive_side(ctx, su, side=side, cell=obj["cell"], prompts=prompts,
                                            timing=timing and side != STACKED_SHADOW)
                finally:
                    g.stop_standup(su)
            kr["apply_path"] = {s: recs[s]["apply_path"] for s in recs}
            kr["endpoint_lens_reduction_mode"] = {s: recs[s].get("endpoint_lens_reduction_mode") for s in recs}
            kr["endpoint_stacked_reductions_env"] = {s: recs[s].get("endpoint_stacked_reductions_env") for s in recs}
            kr["b1_sanity"] = b1_sanity(recs)
            kr["comparisons"] = compare_sides(recs, ctx.batches)
            kr["shadow"] = recs[STACKED_SHADOW].get("shadow")
            kr["shadow_rows"] = shadow_rows_table(kr["shadow"])
            kr["timing"] = {s: recs[s].get("timing") for s in (PER_ROW, STACKED) if recs[s].get("timing")}
            kr["steering_evidence"] = {s: recs[s].get("evidence") for s in recs}
            kr["signature_errors"] = [f"{s}: {e}" for s in recs for e in recs[s]["signature_errors"]]
            kr["raw"] = {s: {"b1": recs[s].get("b1"), "runs": recs[s]["runs"]} for s in recs}
            g._say(f"case {case_id}: " + "; ".join(
                f"B={B}: stacked-vs-per_row {g_['stacked vs per_row'].get('frac_same_all')} vs floor "
                f"{g_['per_row x server'].get('frac_same_all')}" for B, g_ in kr["comparisons"].items()))
        except Exception as e:  # noqa: BLE001 — recorded per case
            kr["error"] = f"{type(e).__name__}: {e}"
            kr["traceback"] = traceback.format_exc()[-3000:]
            g._say(f"case {case_id} FAILED: {type(e).__name__}: {str(e)[:300]}")
        doc["cases"].append(kr)
        g._write_json(path_json, doc)
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    g._write_json(path_json, doc)
    (ctx.results_dir / "report.md").write_text(render_report(doc))
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(g.MODELS))
    ap.add_argument("--repo-dir", default=str(REPO))
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--objects-root", default=g.DEFAULT_OBJECTS_ROOT)
    ap.add_argument("--candsi-root", default=g.DEFAULT_CANDSI_ROOT)
    ap.add_argument("--extra-objects-dir", action="append", default=None, metavar="PATH[=FAMILY]")
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--batches", default="8,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timing-batches", default="8,32,64")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--measure-s", type=float, default=30.0)
    ap.add_argument("--gpu-mem-util", default="0.90")
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--manifest", default="configs/capeval/manifests/petri_twosi.yaml")
    args = ap.parse_args(argv)
    if os.environ.get(STACKED_ENV) or os.environ.get(g.SHADOW_ENV) or os.environ.get(g.PER_ROW_ENV):
        raise SystemExit(f"refusing: {STACKED_ENV} / {g.SHADOW_ENV} / {g.PER_ROW_ENV} must be UNSET in the "
                         "driver's env — each standup sets its own (standup_env scrubs them, but an inherited "
                         "value here means the submission is not what it says)")
    ms = g.MODELS[args.model]
    repo = Path(args.repo_dir).resolve()
    user = os.environ.get("USER", "user")
    from sisteer import site as _site   # a root written with $SHARED expands here; an unset one refuses by name

    def _root(value: str, key: str) -> Path:
        return _site.expand_path(value.format(user=user), key=key)

    bank_dir = _root(args.objects_root, "--objects-root") / ms.bank
    candsi_dir = _root(args.candsi_root, "--candsi-root") / ms.key
    extra, extra_origins = g.parse_extra_objects(args.extra_objects_dir, user)
    header_len = g.header_len_for(ms.hf, ms.rev)
    inventory = g.plan_inventory(ms, bank_dir=bank_dir, extra_dirs=extra, candsi_dir=candsi_dir,
                                 header_len=header_len, extra_origins=extra_origins)
    inventory = {k: v for k, v in inventory.items() if k not in g.INVENTORY_NOTES}
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    results_dir = _root(args.results_root, "--results-root") / ms.key
    results_dir.mkdir(parents=True, exist_ok=True)
    ctx = g.GateContext(
        ms=ms, candidate=repo, reference=repo, reference_sha=g.tree_facts(repo)["sha"], results_dir=results_dir,
        max_model_len=int(args.max_model_len), max_num_seqs=int(args.max_num_seqs),
        gpu_mem_util=str(args.gpu_mem_util), ready_timeout_s=int(args.ready_timeout_s),
        batches=g.parse_int_list(args.batches, "--batches"),
        timing_batches=g.parse_int_list(args.timing_batches, "--timing-batches"),
        max_tokens=int(args.max_tokens), repeats=max(2, int(args.repeats)), top_logprobs=int(args.top_logprobs),
        ignore_eos=not args.no_ignore_eos, warmup_s=float(args.warmup_s), measure_s=float(args.measure_s),
        manifest=args.manifest, legacy_ab=False, batch_invariant=0)
    g._say(f"lens_stacked_ab {ms.key}: cases {cases}; batches {ctx.batches} x {ctx.repeats}; -> {results_dir}")
    doc = run_ab(ctx, cases=cases, inventory=inventory, bank_dir=bank_dir, extra_dirs=extra, candsi_dir=candsi_dir,
                 extra_origins=extra_origins, timing=not args.no_timing)
    rp = results_dir / "report.md"
    print(rp.read_text() if rp.is_file() else json.dumps(doc, indent=1, default=str)[:4000])
    return 0 if all(not c.get("error") for c in doc["cases"]) else 2


if __name__ == "__main__":
    sys.exit(main())
