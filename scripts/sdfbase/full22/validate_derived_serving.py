#!/usr/bin/env python
"""Prove the two DERIVED cells are ACTUALLY SERVED, against a live standup.

The failure this exists to catch is the known one: a served object that
silently no-ops while every hash and readiness check passes. A text
difference alone does not prove a steer — greedy text is a hypersensitive
readout, and two nominally identical legs can diverge on numeric noise
alone. So this measures a NOISE FLOOR first and only then reads the
signal, on four legs served by the SAME engine in the same session:

  bare       ``petri__base``       — passthrough, no hook
  noop       ``petri__none__lens0`` — the path-matched control:
                                      base weights through the LENS path,
                                      hook ARMED, forced scale 0
  cell       the derived cell's vmodel — the same hook at scale 1
  (repeat)   every leg queried TWICE

Read in order:
  1. DETERMINISM — each leg's two identical requests must return identical
     text. If not, nothing below is interpretable and the run says so.
  2. NOISE FLOOR — ``bare`` vs ``noop`` must be identical. Same weights,
     same tokens, hook armed but contributing nothing; a difference here
     means the harness itself moves the output and no steered-vs-noop gap
     could be attributed to the steer.
  3. THE SIGNAL — the derived cell vs ``noop``. Same weights, same path,
     same armed hook; the ONLY difference is the intervention applied at
     scale 1 instead of 0. A difference here is the steer, and the first
     divergent character index is reported per probe.
  4. IDENTITY — ``/v1/models`` lists each cell's vmodel and attests the
     wave's reduction mode; each cell's ``petri_intervention_hash`` (as
     ECHOED BY THE RESPONSE, not just as configured) is distinct from the
     base's and from the other cell's.

Usage, inside the standup's allocation once endpoint.json exists:
  python validate_derived_serving.py --endpoint <serve_dir>/endpoint.json \\
      --cells <cell_id>,<cell_id> --out <report.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: short, neutral probes — enough tokens for a greedy trajectory to diverge
PROMPTS = [
    "Write one sentence about the sea.",
    "What is 17 plus 26? Answer with the number only.",
    "List three colours.",
    "Explain in one sentence why the sky looks blue.",
]
MAX_TOKENS = 48
NOOP_CELL = "none__lens0"
BASE_CELL = "none"
#: the ENGINE's zero twin: the lens table's ``none__lens0``, the native
#: table's ``none__lora0`` — the noise floor and the positive control both compare against the engine's own zero
NOOP_CELL_BY_ENGINE = {"lens": "none__lens0", "lora_native": "none__lora0"}

#: the engine vocabulary the wave asserts from the AUTHORITATIVE stamps (the
#: shell exports TARGET_ENGINE and passes it as --require-engine): a lens proxy at a reduction mode, or the native
#: LoRA engine. The helper is ``proxy_engine.served_engine_problem``, imported at call time (this module binds no
#: ``sisteer`` name at import: its module-level names are stdlib-only, and the ``sys.path`` bootstrap above is what
#: makes that import resolve); the tuple is ``proxy_engine.ENGINE_EXPECTATIONS`` spelled as a literal for the same
#: reason, pinned equal by test (test_sdfbase_full22_wave).
ENGINE_EXPECTATIONS = ("lens:stacked", "lens:per_row", "lora_native")


def _served_engine_problem(ident, *, expect: str):
    from sisteer.capeval.serving.proxy_engine import served_engine_problem
    return served_engine_problem(ident, expect=expect)


def require_engine_of(require_engine, require_lens_mode: str, *, tag: str) -> str:
    """The engine this run asserts: ``--require-engine`` when given (a ``lens:<mode>`` that disagrees with an
    explicit ``--require-lens-mode`` refuses), else ``lens:<--require-lens-mode>`` — so every existing invocation
    (which names only the lens mode) is byte-identical."""
    if require_engine is None:
        return f"lens:{require_lens_mode}"
    if require_engine not in ENGINE_EXPECTATIONS:
        raise SystemExit(f"{tag} --require-engine {require_engine!r} is not one of {ENGINE_EXPECTATIONS}")
    if require_engine.startswith("lens:") and require_engine != f"lens:{require_lens_mode}":
        raise SystemExit(f"{tag} --require-engine {require_engine} disagrees with --require-lens-mode "
                         f"{require_lens_mode} — name one engine")
    return require_engine


def _post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def complete(base: str, vmodel: str, prompt: str) -> dict:
    """One GREEDY completion plus the intervention hash the RESPONSE
    itself records (the served identity, not the configured one)."""
    out = _post(f"{base}/chat/completions", {
        "model": vmodel, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": 0})
    return {"text": out["choices"][0]["message"].get("content") or "",
            "finish_reason": out["choices"][0].get("finish_reason"),
            "served_hash": out.get("petri_intervention_hash")}


def first_divergence(a: str, b: str) -> int | None:
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def route_index(table: dict, repo: Path | None) -> dict:
    """``{cell id -> route}`` under BOTH vocabularies.

    The routing table keys routes by the HARNESS id
    (``si_constv2_mix_full__<arm>``) because that is what the proxy serves;
    a caller naming cells the way the config does uses the TrainTest id
    (``constv2-mix-full__<arm>``). Indexing only one spelling silently
    reports "no route" for a cell that is served perfectly well — which is
    exactly what this script did on its first run. Both are indexed, via
    the authoritative translation when sisteer is importable and a plain
    structural fallback when it is not."""
    routes = {r["cell_id"]: r for r in table["routes"]}
    to_tt = None
    if repo is not None:
        sys.path.insert(0, str(Path(repo) / "src"))
        try:
            from sisteer.capeval.petri.registry_adapter import traintest_cell_id
            to_tt = traintest_cell_id
        except Exception:  # noqa: BLE001 — the fallback below is structural
            to_tt = None
    for harness, r in list(routes.items()):
        tt = None
        if to_tt is not None:
            try:
                tt = to_tt(harness)
            except Exception:  # noqa: BLE001 — control legs refuse; skip them
                tt = None
        if tt is None and harness.startswith("si_"):
            # structural fallback: si_<si with underscores>__<arm>. Only the
            # SI segment differs (hyphens vs underscores), and the arm never
            # contains "__", so splitting on the first "__" is unambiguous.
            si, _, arm = harness[len("si_"):].partition("__")
            if arm:
                tt = f"{si.replace('_', '-')}__{arm}"
        if tt and tt not in routes:
            routes[tt] = r
    return routes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, type=Path)
    ap.add_argument("--cells", required=True,
                    help="comma-separated TrainTest cell ids to validate")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect-mode", default="stacked")
    ap.add_argument("--require-engine", choices=list(ENGINE_EXPECTATIONS), default=None,
                    help="the ENGINE the proxy and endpoint.json must attest (native-serving-default): "
                         "lens:<mode> or lora_native; default lens:<--expect-mode> (byte-identical)")
    ap.add_argument("--repo", type=Path, default=None,
                    help="repo root, so the harness<->TrainTest cell-id "
                         "translation can come from registry_adapter rather "
                         "than the structural fallback")
    args = ap.parse_args(argv)
    require_engine = require_engine_of(args.require_engine, args.expect_mode, tag="[validate]")
    noop_cell = NOOP_CELL_BY_ENGINE["lora_native" if require_engine == "lora_native" else "lens"]

    ep = json.loads(args.endpoint.read_text())
    base_url = ep["proxy_url"]
    table = json.loads(Path(ep["routing_table"]).read_text())
    routes = route_index(table, args.repo)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    report: dict = {"endpoint": str(args.endpoint), "proxy_url": base_url,
                    "cells_validated": cells, "checks": {}, "legs": {},
                    "cells": {}}
    problems: list[str] = []

    # --- 4a. served identity + the attested reduction mode ----------------
    models = _get(f"{base_url}/models")
    listed = [m["id"] for m in models["data"]]
    ident = models.get("petri_proxy") or {}
    report["checks"]["v1_models"] = {
        "n_models": len(listed), "models": sorted(listed),
        "petri_proxy": ident,
        "endpoint_lens_reduction_mode": ep.get("lens_reduction_mode"),
        "endpoint_lens_reduction_source": ep.get("lens_reduction_source"),
        "endpoint_engine": ep.get("engine"), "require_engine": require_engine, "noop_cell": noop_cell,
        "require_attested_mode": ep.get("require_attested_mode"),
        "served_model": ep.get("model"), "revision": ep.get("model_revision"),
        "max_model_len": ep.get("max_model_len"),
        "max_num_seqs": ep.get("max_num_seqs"),
        "node": ep.get("node"), "slurm_job_id": ep.get("slurm_job_id")}
    engine_problem = _served_engine_problem(ident, expect=require_engine)
    if engine_problem is not None:
        problems.append(f"/v1/models petri_proxy does not attest {require_engine}: {engine_problem}")
    if require_engine == "lora_native":
        if ep.get("engine") != "lora_native":
            problems.append(f"endpoint.json engine={ep.get('engine')!r} != 'lora_native'")
    else:
        got = ep.get("lens_reduction_mode")
        if got != args.expect_mode:
            problems.append(f"endpoint.json lens_reduction_mode={got!r} != "
                            f"{args.expect_mode!r}")

    for needed in (BASE_CELL, noop_cell):
        if needed not in routes:
            problems.append(f"no {needed!r} route — the noise floor and the "
                            "positive control both need it")
    if problems and noop_cell not in routes:
        report["problems"] = problems
        report["verdict"] = "FAIL"
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report["checks"], indent=2))
        return 2

    # --- run every leg twice, on every probe ------------------------------
    legs = {BASE_CELL: routes[BASE_CELL]["virtual_model"],
            noop_cell: routes[noop_cell]["virtual_model"]}
    for cell in cells:
        if cell not in routes:
            problems.append(f"{cell}: no route in the routing table")
            continue
        legs[cell] = routes[cell]["virtual_model"]
        if routes[cell]["virtual_model"] not in listed:
            problems.append(f"{cell}: {routes[cell]['virtual_model']} not in "
                            "/v1/models")

    runs: dict[str, list[dict]] = {}
    for leg, vmodel in legs.items():
        runs[leg] = []
        for prompt in PROMPTS:
            a = complete(base_url, vmodel, prompt)
            b = complete(base_url, vmodel, prompt)
            runs[leg].append({"prompt": prompt, "text": a["text"],
                              "repeat_text": b["text"],
                              "deterministic": a["text"] == b["text"],
                              "finish_reason": a["finish_reason"],
                              "served_hash": a["served_hash"]})
        report["legs"][leg] = {"virtual_model": vmodel,
                               "configured_hash": routes[leg]["intervention_hash"],
                               "served_hash": runs[leg][0]["served_hash"],
                               "probes": runs[leg]}

    # --- 1. determinism ---------------------------------------------------
    nondet = [(leg, r["prompt"]) for leg, rs in runs.items() for r in rs
              if not r["deterministic"]]
    report["checks"]["determinism"] = {
        "all_legs_deterministic": not nondet,
        "nondeterministic": [{"leg": l, "prompt": p} for l, p in nondet]}
    if nondet:
        problems.append(
            f"{len(nondet)} leg/probe pair(s) are NOT deterministic across "
            "two identical requests — a steered-vs-control text difference "
            "cannot be attributed to the steer while this holds")

    # --- 2. the noise floor: bare vs the armed scale-0 leg ----------------
    floor = []
    for i, prompt in enumerate(PROMPTS):
        same = runs[BASE_CELL][i]["text"] == runs[noop_cell][i]["text"]
        floor.append({"prompt": prompt, "identical": same,
                      "first_divergence_char": first_divergence(
                          runs[BASE_CELL][i]["text"], runs[noop_cell][i]["text"])})
    report["checks"]["noise_floor_bare_vs_noop"] = {
        "all_identical": all(f["identical"] for f in floor), "per_probe": floor}
    if not all(f["identical"] for f in floor):
        problems.append(
            "the armed scale-0 control does not reproduce the bare leg "
            "token-for-token — the harness moves the output on its own, so "
            "no steered-vs-control difference is attributable to the steer")

    # --- 3. the signal: each cell vs the armed scale-0 control ------------
    for cell in cells:
        if cell not in runs:
            continue
        probes = []
        for i, prompt in enumerate(PROMPTS):
            steered = runs[cell][i]["text"]
            noop = runs[noop_cell][i]["text"]
            idx = first_divergence(steered, noop)
            probes.append({"prompt": prompt, "steered": steered,
                           "noop_control": noop,
                           "bare": runs[BASE_CELL][i]["text"],
                           "differs_from_noop_control": idx is not None,
                           "first_divergence_char": idx})
        n_diff = sum(p["differs_from_noop_control"] for p in probes)
        rec = {"virtual_model": legs[cell],
               "configured_intervention_hash": routes[cell]["intervention_hash"],
               "served_intervention_hash": runs[cell][0]["served_hash"],
               "n_probes": len(PROMPTS), "n_differ_from_noop_control": n_diff,
               "steer_is_live": n_diff > 0, "probes": probes}
        report["cells"][cell] = rec
        if n_diff == 0:
            problems.append(
                f"{cell}: greedy completions are IDENTICAL to the armed "
                f"scale-0 control on all {len(PROMPTS)} probes — the served "
                "object is a silent no-op")
        if rec["served_intervention_hash"] != rec["configured_intervention_hash"]:
            problems.append(
                f"{cell}: the response echoes intervention hash "
                f"{rec['served_intervention_hash']} but the routing table "
                f"configures {rec['configured_intervention_hash']}")

    # --- 4b. distinct served identities -----------------------------------
    served = {c: report["cells"][c]["served_intervention_hash"]
              for c in cells if c in report["cells"]}
    served[BASE_CELL] = runs[BASE_CELL][0]["served_hash"]
    report["checks"]["served_intervention_hashes"] = served
    if len({v for v in served.values() if v}) != len(served):
        problems.append(f"legs share a served intervention hash: {served}")

    # the two derived cells must also differ from each OTHER in output —
    # two distinct objects serving identical text would mean one is wrong
    if len(cells) == 2 and all(c in runs for c in cells):
        pairwise = [runs[cells[0]][i]["text"] != runs[cells[1]][i]["text"]
                    for i in range(len(PROMPTS))]
        report["checks"]["derived_cells_differ_from_each_other"] = {
            "n_differing_probes": sum(pairwise), "per_probe": pairwise}

    report["problems"] = problems
    report["verdict"] = "PASS" if not problems else "FAIL"
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"verdict": report["verdict"], "problems": problems,
                      "checks": {k: v for k, v in report["checks"].items()},
                      "cells": {c: {k: v for k, v in r.items()
                                    if k != "probes"}
                                for c, r in report["cells"].items()}},
                     indent=2)[:6000])
    return 0 if not problems else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (urllib.error.URLError, OSError) as e:
        print(f"[validate] transport failure: {e}", file=sys.stderr)
        sys.exit(3)
