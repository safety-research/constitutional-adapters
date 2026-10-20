#!/usr/bin/env python
"""The Qwen3.5 per-model serving gate: the hook BITES, at the RECORDED mask.

Gates the two Qwen banks' fitted cells (psr_kl_all / lora_r4 x 3 SIs)
before any wave serving. Reuses petri_steered_smoke's committed machinery
(its worker registry builder, its worker-steered engine, its response
phase) rather than reimplementing an engine path; this driver adds the
assertions the seven invalid scale-1 psr conditions proved were missing,
and hardens the outcome discipline.

WHY THIS GATE EXISTS (2026-08-29/30): a naive bite test passes an arm
that bites at the WRONG positions — `_adapt_psr` served response-fit
objects all-position for weeks while every existing check stayed green.
So the FIRST assertion here is served-ops-match-recorded-spec, for EVERY
kind, via the real committed adapter (`la.get_adapter`), with a negative
control proving the assertion can fail. Only then does hardware evidence
count.

PASS requires ALL of, per cell:
  1. CPU: the adapter's ops carry positions='response', phase='both',
     and EXACTLY the tokenizer-derived assistant header ids, whose count
     equals the blob's own response_header_len (Qwen3.5: 7 — the empty
     <think></think> block; a thinking-ON render MOVES this boundary).
  2. GPU (worker-steered --phase response, the wave geometry, caching
     OFF, eager, no-chunk identity, KV-coverage asserted): steering
     BITES (steered != scale-0 twin on >=1 probe prompt) and the scale-0
     twin is a NO-OP (== unsteered baseline on EVERY probe prompt).
  3. Install attestation: block-site hooked layers == the site-shifted
     set the adapters computed (worker-asserted), AND o_proj-site layers
     == the lora arms' layer set (driver-asserted here: the llama smoke
     never had an o_proj arm, so the worker only asserts the block half).

OUTCOME DISCIPLINE — STRICT: there is no "unavailable" exit-0 here. A
gate that cannot run is a FAILURE (exit nonzero) — three runs in this
campaign reported success while validating nothing, and each lacked
exactly this property.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location(
    "_petri_steered_smoke", REPO_ROOT / "scripts" / "petri_steered_smoke.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules["_petri_steered_smoke"] = smoke
_spec.loader.exec_module(smoke)


class QwenGateError(RuntimeError):
    """Any reason this gate cannot certify — including 'could not run'."""


def bank_cells(bank: Path, cells: Sequence[str]) -> List[Dict[str, Any]]:
    """Load the requested cells from the bank with their provenance.
    Refuses a missing object, a missing/duplicative provenance model pin,
    or a cell whose arm has no registered object kind."""
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS

    out: List[Dict[str, Any]] = []
    for cid in cells:
        d = bank / cid
        obj, prov = d / "object.pt", d / "provenance.json"
        if not obj.is_file() or not prov.is_file():
            raise QwenGateError(
                f"cell {cid!r}: bank dir {d} lacks object.pt/provenance.json"
                " — a gate over a missing object certifies nothing")
        meta = json.loads(prov.read_text())
        arm = cid.split("__", 1)[1] if "__" in cid else cid
        kind = ARM_OBJECT_KINDS.get(arm)
        if kind is None:
            raise QwenGateError(
                f"cell {cid!r}: arm {arm!r} has no registered object kind "
                f"(ARM_OBJECT_KINDS) — record-only arms (lora_r16) are not "
                "matrix-wired and must not be gated as if they were")
        out.append({"cell_id": cid, "kind": kind, "bytes": obj.read_bytes(),
                    "required": True, "path": str(obj),
                    "spec_extra": {"positions": "response"},
                    "provenance": meta})
    models = {(c["provenance"].get("model"),
               c["provenance"].get("model_revision")) for c in out}
    if len(models) != 1 or None in next(iter(models)):
        raise QwenGateError(
            f"bank cells disagree on (model, revision): {sorted(models)} — "
            "one gate certifies one served model")
    return out


def preflight_ops(arms: Sequence[Mapping[str, Any]],
                  header_ids: Sequence[int]) -> Dict[str, Any]:
    """THE SERVED-OPS-MATCH-RECORDED-SPEC ASSERTION, every kind — the
    check whose absence produced seven invalid conditions. Re-adapts each
    arm through the committed adapter and requires positions/phase/header
    to match what the registry records; then the NEGATIVE CONTROL: a
    default-constructed op must FAIL the same predicate, or this check
    could not fail and proves nothing."""
    from sisteer.capeval.serving import lens_adapters as la

    hdr = tuple(int(t) for t in header_ids)
    report: Dict[str, Any] = {"per_cell": {}, "negative_control": None}
    problems: List[str] = []
    block_layers: set = set()
    oproj_layers: set = set()
    for arm in arms:
        spec: Dict[str, Any] = {"kind": arm["kind"]}
        spec.update(arm["spec_extra"])
        ops = la.get_adapter(arm["kind"])(spec, arm["bytes"], b"",
                                          header_ids=hdr)
        facts = smoke.op_facts(ops)
        h = int(arm["provenance"].get("response_header_len", -1))
        cell_report = dict(facts, blob_response_header_len=h)
        report["per_cell"][arm["cell_id"]] = cell_report
        if facts["positions"] != ["response"]:
            problems.append(f"{arm['cell_id']}: served positions "
                            f"{facts['positions']} != ['response'] — the "
                            "recorded mask is not in force (the _adapt_psr "
                            "failure class)")
        if facts["phases"] != ["both"]:
            problems.append(f"{arm['cell_id']}: phases {facts['phases']} != "
                            "['both'] — a prefill-only op leaves generation "
                            "unsteered")
        if tuple(facts["header_ids"] or ()) != hdr:
            problems.append(f"{arm['cell_id']}: op header ids != the "
                            "tokenizer-derived assistant header")
        if h != len(hdr):
            problems.append(
                f"{arm['cell_id']}: blob response_header_len={h} but the "
                f"served tokenizer derives {len(hdr)} header tokens — fit "
                "and serve disagree about where the response begins "
                "(thinking-mode drift moves this boundary)")
        for op in ops:
            (oproj_layers if la.op_site(op) == la.HOOK_SITE_OPROJ
             else block_layers).add(int(op.layer))
    # the negative control: the buggy shape must FAIL this predicate
    bad = la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                    vector=__import__("torch").zeros(8))
    neg = smoke.op_facts((bad,))
    report["negative_control"] = neg
    if neg["positions"] == ["response"]:
        problems.append("negative control BROKEN: a default-constructed op "
                        "reads positions='response' — this assertion can no "
                        "longer fail and certifies nothing")
    if problems:
        raise QwenGateError("served-ops-vs-recorded-spec FAILED:\n- "
                            + "\n- ".join(problems))
    report["block_layers"] = sorted(block_layers)
    report["oproj_layers"] = sorted(oproj_layers)
    return report


def verdict_from_worker(result: Mapping[str, Any],
                        arms: Sequence[Mapping[str, Any]],
                        expected_oproj: Sequence[int]) -> Dict[str, Any]:
    """The hardware half, from the worker's own JSON. Bites + scale-0
    no-op per cell, plus the o_proj install half the worker does not
    assert (its block-half assertion already gated engine exit)."""
    problems: List[str] = []
    if result.get("stage") != "completed":
        problems.append(f"worker ended at stage {result.get('stage')!r} "
                        f"(error: {result.get('error')!r})")
    inst = result.get("install") or {}
    got_oproj = sorted(int(l) for l in inst.get("oproj_layers") or ())
    if got_oproj != sorted(expected_oproj):
        problems.append(f"install oproj_layers {got_oproj} != expected "
                        f"{sorted(expected_oproj)} — the o_proj-site hooks "
                        "(lora kinds) did not land where the adapters "
                        "computed")
    bites = result.get("response_bites") or {}
    errors = result.get("response_errors") or {}
    served = set(result.get("response_served_arm_ids") or ())
    for arm in arms:
        cid = arm["cell_id"]
        if cid in errors:
            problems.append(f"{cid}: RAISED on hardware — "
                            f"{errors[cid].get('error')}")
            continue
        if cid not in served or cid not in bites:
            problems.append(f"{cid}: no hardware evidence gathered — a "
                            "missing probe is a failure, not a pass")
            continue
        b = bites[cid]
        if int(b.get("n_differing", 0)) < 1:
            problems.append(
                f"{cid}: steering does not BITE (steered == scale-0 twin on "
                f"all {b.get('n')} probe prompts) — an inert serve wearing "
                "a defended label")
        if int(b.get("scale0_identical_to_unsteered", -1)) != int(b.get("n")):
            problems.append(
                f"{cid}: scale-0 twin is NOT a no-op "
                f"({b.get('scale0_identical_to_unsteered')}/{b.get('n')} "
                "identical to baseline) — the hook leaks at scale 0")
    kv = (result.get("geometry") or {}).get("kv_cache") or {}
    if kv.get("covers_full_budget") is not True:
        problems.append("KV cache does not back the full no-chunk budget "
                        f"({kv}) — max_num_seqs is a ceiling, not a grant")
    return {"problems": problems, "bites": bites,
            "kv": {k: kv.get(k) for k in ("kv_cache_tokens",
                                          "full_length_concurrency",
                                          "covers_full_budget")}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True,
                    help="the model's scale-12 bank dir (cells inside)")
    ap.add_argument("--cells", required=True,
                    help="comma-separated cell ids to gate")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--skip-worker", action="store_true",
                    help="CPU pre-flight only (CI); the record says so")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {"started": time.time(), "argv": list(sys.argv)}

    def fail(msg: str) -> int:
        record.update(status="failed", error=msg, finished=time.time())
        (out_dir / "qwen_gate_failed.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, default=str))
        print(f"QWEN GATE FAILED: {msg}", file=sys.stderr)
        return 1

    try:
        bank = Path(args.bank)
        arms = bank_cells(bank, [c.strip() for c in args.cells.split(",")
                                 if c.strip()])
        model = arms[0]["provenance"]["model"]
        revision = arms[0]["provenance"]["model_revision"]
        record.update(model=model, revision=revision,
                      cells=[a["cell_id"] for a in arms])

        from transformers import AutoTokenizer

        from sisteer.chat_markers import generation_header_ids
        from sisteer.models import pin_reasoning_template_off
        tok = AutoTokenizer.from_pretrained(model, revision=revision)
        pinned = pin_reasoning_template_off(tok)
        header_ids = tuple(int(t) for t in generation_header_ids(tok))
        record.update(thinking_pin_applied=pinned,
                      assistant_header_ids=list(header_ids))

        pre = preflight_ops(arms, header_ids)
        record["preflight"] = pre

        reg_dir = out_dir / "registry"
        expects = smoke.build_registry(reg_dir, arms, header_ids)
        from sisteer.capeval.serving import lens_steer as ls
        from sisteer.capeval.serving import lens_worker as lw
        reg = ls.WorkerObjectRegistry.load(reg_dir)
        ledger = sorted(lw.cells_needing_ledger(reg))
        arms_json = {
            "arm_ids": [a["cell_id"] for a in arms],
            "required_arm_ids": [a["cell_id"] for a in arms],
            "mixed_arm_ids": [],   # every qwen arm is response-mode
            "response_arm_ids": [a["cell_id"] for a in arms],
            "response_arm_donors": {},
            "assistant_header_ids": list(header_ids),
            "selectors": {}, "expected_ledger_cells": ledger,
            # the worker asserts the BLOCK half only (llama smoke had no
            # o_proj arm); the o_proj half is asserted back here.
            "expected_hook_layers": pre["block_layers"],
        }
        for a in arms:
            e = expects[a["cell_id"]]
            arms_json["selectors"][a["cell_id"]] = {
                "petri_object": a["cell_id"],
                "expect_hash": e["expect_hash"], "scale": 1.0}
            arms_json["selectors"][e["scale0_cell"]] = {
                "petri_object": e["scale0_cell"],
                "expect_hash": e["scale0_hash"], "scale": 0.0}
        aj = out_dir / "gate_arms.json"
        aj.write_text(json.dumps(arms_json, indent=2, sort_keys=True))

        if args.skip_worker:
            record.update(status="preflight_only", finished=time.time())
            (out_dir / "qwen_gate_record.json").write_text(
                json.dumps(record, indent=2, sort_keys=True, default=str))
            print("QWEN GATE preflight-only: CPU assertions passed; NO "
                  "hardware evidence — this record does NOT clear serving")
            return 0

        worker_out = out_dir / "worker_response.json"
        argv_w = [sys.executable, str(REPO_ROOT / "scripts" /
                                      "petri_steered_smoke.py"),
                  "worker-steered", "--model", model,
                  "--revision", revision, "--registry", str(reg_dir),
                  "--arms-json", str(aj), "--out", str(worker_out),
                  "--phase", "response",
                  "--max-model-len", str(args.max_model_len),
                  "--max-num-seqs", str(args.max_num_seqs),
                  "--gpu-memory-utilization",
                  str(args.gpu_memory_utilization)]
        record["worker_argv"] = argv_w
        # THE SMOKE'S OWN CHILD ENV, not a bare inherit (jobs
        # 255287/255288): VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps the
        # engine core in-process so the callable KV-facts RPC needs no
        # serialization (out-of-process it dies "function is not
        # serializable", and the fallback -
        # VLLM_ALLOW_INSECURE_SERIALIZATION=1 - would weaken the exact
        # assertion this gate exists to keep). It also pops
        # VLLM_PLUGINS (entry-point-name trap), pins the V1 runner and
        # explicit BI=0, and prepends src to PYTHONPATH.
        env_w = smoke._child_env(disable_lens=False, batch_invariant="0")
        record["worker_env_pins"] = {k: env_w.get(k) for k in (
            "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_BATCH_INVARIANT")}
        rc = subprocess.run(argv_w, env=env_w).returncode
        if not worker_out.is_file():
            return fail(f"worker exited {rc} and wrote NO result JSON — "
                        "could-not-run is a failure here, never a skip")
        result = json.loads(worker_out.read_text())
        record["worker_exit"] = rc
        record["worker_stage"] = result.get("stage")
        v = verdict_from_worker(result, arms, pre["oproj_layers"])
        record["verdict"] = v
        if rc != 0:
            v["problems"].append(f"worker exit code {rc}")
        if v["problems"]:
            return fail("hardware verdict FAILED:\n- "
                        + "\n- ".join(v["problems"]))
    except QwenGateError as e:
        return fail(str(e))
    except Exception as e:  # STRICT: no unavailable-exit-0 path exists
        import traceback
        record["traceback"] = traceback.format_exc()
        return fail(f"{type(e).__name__}: {e}")

    record.update(status="passed", finished=time.time())
    (out_dir / "qwen_gate_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str))
    print(f"QWEN GATE PASSED for {model} — record at "
          f"{out_dir / 'qwen_gate_record.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
