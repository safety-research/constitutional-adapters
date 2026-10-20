"""The serving probe's HF+PEFT parity check (assertion 4) refuses when the REFERENCE is not the served adapter.

Probe 805161 excluded every (rank, set) pair as `reference_mismatch` on every adapter: the HF side answered the SAME top-1
for every adapter — the base model. The reference had loaded the text-only causal-LM class (tree `model.layers`) through the
harness's causal-LM-first auto-class while the export spells its PEFT names in the checkpoint's own multimodal class tree
(`model.language_model.layers`); PEFT 0.20.0 bound none of the keys, warned, and
kept B = 0. A parity harness that attaches none of the object it checks must refuse, never file a mismatch. This file pins:
* the reference loads the class the CHECKPOINT names (`config.architectures[0]`), never a guessed loader;
* every exported tensor must sit on the wrapped reference under its own name and shape (`attachment_check`);
* PEFT's own layout disagreeing with the export, or HF outputs identical across distinct adapters, is `reference_missing`;
* a launch with ANY unjudged adapter carries `reference_check {status: reference_missing, basis: None}` and the verdict is
  UNCERTIFIED (no menu), never `reference_mismatch`;
* `run --resume --reference-only` re-judges every adapter of a finished record on one GPU, no engine, archives the prior
  reference_check.json + the derived artifacts, and — on a complete record — needs the front door's reopen receipt;
* the carrier maps positional 11 `reference` to that resume (executed, not string-matched).
CPU only; fake engines; synthetic ids; no cluster path, no host name."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import geometry as g
from sisteer.capeval.serving import standup_manifest as sm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import serving_probe as sp  # noqa: E402
from test_serving_probe_parallel import _fake_deps  # noqa: E402
from test_serving_probe_revocation import _mint_receipt  # noqa: E402
from test_serving_probe_smoke import REV, TOY, FakeExporter, _passed  # noqa: E402

MODEL = "qwen35_4b_demo"       # the front door's worked example: the probe subject (a key of lora_modules.ARCH_DIMS_BY_MODEL_KEY; a hybrid)
LM = "base_model.model.model.language_model.layers.3.self_attn.o_proj"      # the export's spelling (the checkpoint's tree)
TXT = "base_model.model.model.layers.3.self_attn.o_proj"                     # the text-only class's tree


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE", sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)


# ------------------------------------------------------------------------------------------------- the pure helpers

def test_reference_model_class_is_the_checkpoints_first_architecture_and_refuses_none():
    assert sp.reference_model_class(SimpleNamespace(architectures=["Gemma4ForConditionalGeneration"])) == "Gemma4ForConditionalGeneration"
    for cfg in (SimpleNamespace(architectures=[]), SimpleNamespace(architectures=None), SimpleNamespace(), SimpleNamespace(architectures=[""])):
        with pytest.raises(RuntimeError, match="names no `architectures`"):
            sp.reference_model_class(cfg)


def test_attachment_check_refuses_every_key_when_the_tree_lacks_the_exports_prefix_and_passes_when_it_carries_it():
    header = {f"{LM}.lora_A.weight": {"shape": [4, 4096], "dtype": "F32"}, f"{LM}.lora_B.weight": {"shape": [2048, 4], "dtype": "F32"}}
    # the text-only class: PEFT's parameters live under `model.layers` — none of ours attach (what probe 805161 did, silently)
    text_tree = {f"{TXT}.lora_A.default.weight": (4, 4096), f"{TXT}.lora_B.default.weight": (2048, 4)}
    att = sp.attachment_check(header, text_tree)
    assert att["pass"] is False and att["n_attached"] == 0 and att["n_missing"] == 2 and sorted(att["missing"]) == sorted(header)
    # the checkpoint's class: every exported tensor sits under its own name and shape
    lm_tree = {f"{LM}.lora_A.default.weight": (4, 4096), f"{LM}.lora_B.default.weight": (2048, 4)}
    ok = sp.attachment_check(header, lm_tree)
    assert ok["pass"] is True and ok["n_attached"] == 2 and ok["missing"] == [] and ok["shape_mismatch"] == []
    # a name that attached with another shape is not the served adapter either
    bad = sp.attachment_check(header, {**lm_tree, f"{LM}.lora_B.default.weight": (2048, 8)})
    assert bad["pass"] is False and bad["shape_mismatch"] == [f"{LM}.lora_B.weight"] and bad["n_attached"] == 1
    assert sp.attachment_check({}, lm_tree)["pass"] is False           # nothing exported = nothing checked


def test_export_decoder_prefix_reads_the_exporters_provenance(tmp_path):
    d = tmp_path / "a"; d.mkdir()
    (d / sp.sa.ADAPTER_CONFIG_FILE).write_text(json.dumps({"r": 4, "petri_decoder_prefix": "model.language_model.layers"}))
    assert sp.export_decoder_prefix(d) == "model.language_model.layers"
    (d / sp.sa.ADAPTER_CONFIG_FILE).write_text(json.dumps({"r": 4}))
    assert sp.export_decoder_prefix(d) is None                          # no provenance and no readable file: None, never a guess


def _hf(top1, lp, *rest):
    top5 = [[top1, lp]] + [list(r) for r in rest]
    while len(top5) < 5:
        top5.append([100 + len(top5), lp - len(top5)])
    return {"top1_id": top1, "top1_logprob": lp, "top5": top5}


def test_identical_outputs_across_distinct_adapters_are_flagged_and_a_lone_adapter_is_not():
    same = _hf(29108, -0.1224699392914772, [760, -2.2474699])
    per = {"/r/a/constitution__lora": {"status": "fail", "hf": same}, "/r/a/control__lora": {"status": "fail", "hf": dict(same)},
           "/r/b/difference__lora": {"status": "pass", "hf": _hf(75297, -4.19654)},
           "/r/c/x": {"status": "reference_missing"}}
    groups = sp.identical_reference_outputs(per)
    assert groups == [["/r/a/constitution__lora", "/r/a/control__lora"]]
    assert sp.identical_reference_outputs({"/r/a": {"status": "pass", "hf": same}}) == []


def test_finish_reference_record_turns_a_layout_disagreement_and_identical_outputs_into_reference_missing():
    base = _hf(29108, -0.12247)
    same = _hf(29108, -0.12247)
    per = {"/r/wo/constitution__lora": {"status": "fail", "hf": same, "vllm": {"id": 23042, "logprob": -4.64}},
           "/r/wo/control__lora": {"status": "fail", "hf": dict(same), "vllm": {"id": 31618, "logprob": -4.43}},
           "/r/mlp/constitution__lora": {"status": "pass", "hf": _hf(58048, -4.69), "vllm": {"id": 58048, "logprob": -4.695}},
           "/r/qkvo/constitution__lora": {"status": "pass", "hf": _hf(86244, -2.85), "vllm": {"id": 86244, "logprob": -2.887}}}
    layouts = {"wo": {"status": "written", "compare": {"pass": True, "n_ours": 20, "n_peft": 20, "only_ours": [], "only_peft": []}},
               "mlp": {"status": "written", "compare": {"pass": False, "n_ours": 400, "n_peft": 400,
                                                        "only_ours": ["base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"],
                                                        "only_peft": ["base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"], "shape_mismatch": []}},
               "qkvo": {"status": "written", "compare": {"pass": True, "n_ours": 80, "n_peft": 80, "only_ours": [], "only_peft": []}}}
    layout_of = {"/r/wo/constitution__lora": "wo", "/r/wo/control__lora": "wo", "/r/mlp/constitution__lora": "mlp", "/r/qkvo/constitution__lora": "qkvo"}
    rec = sp.finish_reference_record(per, layouts, layout_of, base=base, peft_version="0.20.0", reference_class="Gemma4ForConditionalGeneration")
    got = rec["per_adapter"]
    assert got["/r/mlp/constitution__lora"]["status"] == sp.REF_STATUS_MISSING and "disagrees with the export" in got["/r/mlp/constitution__lora"]["reason"]
    assert got["/r/wo/constitution__lora"]["status"] == sp.REF_STATUS_MISSING and "IDENTICAL across 2 distinct adapters" in got["/r/wo/constitution__lora"]["reason"]
    assert got["/r/wo/control__lora"]["status"] == sp.REF_STATUS_MISSING
    assert got["/r/qkvo/constitution__lora"]["status"] == "pass"           # judged, layout agrees, distinct output: untouched
    assert rec["counts"] == {"pass": 1, "fail": 0, sp.REF_STATUS_UNDECIDABLE: 0, sp.REF_STATUS_MISSING: 3, "error": 0}   # the scaled pass rule adds the undecidable count
    assert rec["reference_class"] == "Gemma4ForConditionalGeneration" and rec["status"] == "done"
    assert per["/r/mlp/constitution__lora"]["status"] == "pass"           # the input is not mutated


def test_first_token_parity_is_the_one_rule():
    hf = _hf(23042, -4.6909)
    assert sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6909 + 0.01})["status"] == "pass"
    r = sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6429})
    assert r["status"] == "fail" and r["top1_equal"] is True and r["logprob_abs_delta"] == pytest.approx(0.048) and r["tolerance"] == sp.REFERENCE_LOGPROB_TOL
    assert sp.first_token_parity(hf, {"id": 1, "logprob": -4.6909})["top1_equal"] is False
    assert sp.first_token_parity(hf, {})["status"] == "fail"


def test_launch_reference_check_is_reference_missing_with_no_basis_when_any_adapter_was_not_judged():
    items = [{"name": "control__lora", "dir": "/r/c"}, {"name": "constitution__lora", "dir": "/r/k"}]
    both_pass = {"/r/c": {"status": "pass"}, "/r/k": {"status": "pass"}}
    assert sp.launch_reference_check(items, both_pass)["status"] == "pass"
    assert sp.launch_reference_check(items, both_pass)["basis"] == sp.BASIS_HF_PEFT
    one_fail = {"/r/c": {"status": "pass"}, "/r/k": {"status": "fail", "logprob_abs_delta": 0.24}}
    assert sp.launch_reference_check(items, one_fail) == {**sp.launch_reference_check(items, one_fail), "status": "fail", "basis": sp.BASIS_HF_PEFT}
    for unjudged in ({"status": sp.REF_STATUS_MISSING, "reason": "PEFT attached 0/20"}, {"status": "error", "error": "RuntimeError: boom"}, None):
        per = {"/r/c": {"status": "pass"}}
        if unjudged is not None:
            per["/r/k"] = unjudged
        rc = sp.launch_reference_check(items, per, reference_class="X")
        assert rc["status"] == sp.REF_STATUS_MISSING and rc["basis"] is None and rc["reference_class"] == "X"
        assert "constitution__lora:" in rc["reason"] and (not unjudged or (unjudged.get("reason") or unjudged.get("error")) in rc["reason"])


# ------------------------------------------------------------------------------------- the record: phase + verdict

def _plan(sets=("wo",)):
    plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=sets, ranks=(4,), windows=(4096,), ladder_caps=())
    return plan


def _record(out: Path, plan, *, results_reference=None, base_rows=True):
    """A record whose launches are all passed, with the driver's `reference_pending` items (the HF+PEFT phase's inputs) and
    the engine's base probe on each result; `results_reference` = the reference_check to pre-stamp (None = none yet)."""
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    sp.build_adapter_dirs(plan, out, FakeExporter())
    results = {}
    for rec in plan["launches"]:
        m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
        res = _passed(rec, m, reference=results_reference)
        res["reference_pending"] = [
            {"dir": str(sp.slot_dir(out, m.name, a.name)), "name": a.name, "kind": a.kind, "role": a.role, "rank": a.rank,
             "lora_modules": list(a.lora_modules),
             "vllm_first_token": {"id": 1000 + len(a.name), "logprob": -4.5, "token": f"token_id:{1000 + len(a.name)}"}}
            for a in m.adapters if a.kind == sm.KIND_GENUINE]
        if base_rows:
            res["probes"] = {"base": {"token_ids": [29108], "rows": [{"logprob": -0.1404, "token": "token_id:29108", "top": []}]}, "adapters": {}}
        results[rec["id"]] = res
        sp.write_json_atomic(sp.launch_result_path(out, rec["id"]), res)
    return results


def _ref_deps(plan, judge):
    """The smoke suite's fake engine with `reference` replaced by `judge(pending) -> per_adapter`, recording its calls, and
    `stand_up` armed to fail the test if an engine is ever started."""
    calls = {"reference": [], "stand_up": 0}
    base, _seen = _fake_deps(plan)          # (deps, the per-launch device record)

    def reference(**kw):
        calls["reference"].append([it["dir"] for it in kw["pending"]])
        calls["base_seen"] = [it.get("vllm_base_first_token") for it in kw["pending"]]
        return {"status": "done", "peft_version": "0.20.0", "layouts": {}, "reference_class": "ToyForConditionalGeneration",
                "per_adapter": judge(kw["pending"])}

    def stand_up(**kw):
        calls["stand_up"] += 1
        raise AssertionError("an engine was started")
    return dataclasses.replace(base, reference=reference, stand_up=stand_up), calls


def test_a_reference_missing_adapter_makes_the_launch_reference_missing_and_the_verdict_uncertified_not_a_mismatch(tmp_path):
    out = tmp_path / "rec"; out.mkdir()
    plan = _plan()
    _record(out, plan)

    def judge(pending):
        per = {}
        for it in pending:
            if it["role"] == "constitution":
                per[it["dir"]] = {"status": sp.REF_STATUS_MISSING, "reason": "PEFT attached 0/20 of the exported tensors under their own names"}
            else:
                per[it["dir"]] = {**sp.first_token_parity(_hf(it["vllm_first_token"]["id"], -4.5), it["vllm_first_token"]), "hf": _hf(it["vllm_first_token"]["id"], -4.5)}
        return per
    deps, calls = _ref_deps(plan, judge)
    rec = sp.reference_phase(plan, out, deps)
    assert rec["status"] == "done" and calls["reference"] and calls["base_seen"][0] == {"id": 29108, "logprob": -0.1404, "token": "token_id:29108"}
    results = sp.load_results(out)
    s1cc, s1d, union = (next(r["id"] for r in plan["launches"] if r["topology"] == t) for t in ("S1cc", "S1d", "single"))
    assert set(results) == {s1cc, s1d, union}                               # the pair's union (the served launch class) ran beside them
    res = results[s1cc]                                                      # the constitution adapter is served by S1cc
    assert res["reference_check"]["status"] == sp.REF_STATUS_MISSING and res["reference_check"]["basis"] is None
    assert "constitution__lora: reference_missing" in res["reference_check"]["reason"]
    assert res["reference_check"]["reference_class"] == "ToyForConditionalGeneration"
    assert results[s1d]["reference_check"]["status"] == "pass" and results[s1d]["reference_check"]["basis"] == sp.BASIS_HF_PEFT   # the difference adapter was judged
    verdict, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=out, verdict_uri=str(out / sp.VERDICT_FILE))
    assert verdict["status"] == sp.VERDICT_UNCERTIFIED and verdict["unreferenced_pairs"] == [[4, "wo"]] and allowed["pairs"] == []
    (ex,) = allowed["excluded"]
    assert ex["reason"].startswith("reference_missing:") and s1cc in ex["reason"] and "reference_mismatch" not in ex["reason"]
    # the CLI: exit 6, no menu on disk, and the log names the harness's own refusal
    rc = sp.cmd_verdict(SimpleNamespace(model_key=MODEL, out=out, probe_job_id="t", verdict_uri=None, geometry_dir=None, probe_cluster=None))
    assert rc == sp.EXIT_UNCERTIFIED and not (out / sp.ALLOWED_PAIRS_FILE).exists()
    assert (out / sp.RESULT_FILE).read_text().startswith("state=UNCERTIFIED certifying=false reason=reference_missing")


def test_a_judged_mismatch_is_still_a_mismatch(tmp_path):
    out = tmp_path / "rec"; out.mkdir()
    plan = _plan()
    _record(out, plan)
    deps, _ = _ref_deps(plan, lambda pending: {it["dir"]: {**sp.first_token_parity(_hf(1, -1.0), it["vllm_first_token"]), "hf": _hf(1, -1.0)} for it in pending})
    sp.reference_phase(plan, out, deps)
    results = sp.load_results(out)
    assert len(results) == 3                                                 # wo: S1cc (control, constitution) + S1d (difference) + the union
    for res in results.values():
        assert res["reference_check"]["status"] == "fail" and res["reference_check"]["basis"] == sp.BASIS_HF_PEFT
    verdict, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=out, verdict_uri=str(out / sp.VERDICT_FILE))
    assert verdict["status"] == sp.VERDICT_COMPLETE and allowed["excluded"][0]["reason"].startswith("reference_mismatch:")


def _complete_record(out: Path, plan, deps):
    """A finished, certified record: results with a (failing) reference stamped by a first pass, verdict + menu, complete checkpoint."""
    _record(out, plan)
    sp.reference_phase(plan, out, deps)                                   # first pass: the old (wrong) judgement
    results = sp.load_results(out)
    sp.write_verdict(plan, results, probe_job_id="800", out=out, verdict_uri=str(out / sp.VERDICT_FILE))
    sp.write_progress(plan, results, out)
    assert (out / sp.REFERENCE_CHECK_FILE).is_file() and (out / sp.VERDICT_FILE).is_file()
    assert sp.load_progress(out)["complete"] is True
    return results


def _run_reference_only(out, *, token=None, sets="wo"):
    argv = ["run", "--model-key", MODEL, "--out", str(out), "--resume", "--reference-only", "--plan-args", f"--sets {sets} --ranks 4 --tp 1"]
    if token:
        argv += ["--reopen-token", token]
    return sp.main(argv)


def test_reference_only_rerun_needs_the_receipt_on_a_complete_record_and_rejudges_every_adapter_without_an_engine(tmp_path, monkeypatch):
    out = tmp_path / "rec"; out.mkdir()
    plan = _plan()
    wrong, _ = _ref_deps(plan, lambda pending: {it["dir"]: {**sp.first_token_parity(_hf(29108, -0.12247), it["vllm_first_token"]), "hf": _hf(29108, -0.12247)} for it in pending})
    _complete_record(out, plan, wrong)
    assert json.loads((out / sp.VERDICT_FILE).read_text())["excluded"][0]["reason"].startswith("reference_mismatch:")
    right, calls = _ref_deps(plan, lambda pending: {it["dir"]: {**sp.first_token_parity(_hf(it["vllm_first_token"]["id"], -4.5), it["vllm_first_token"]),
                                                                "hf": _hf(it["vllm_first_token"]["id"], -4.5)} for it in pending})
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: right))
    # no receipt: refused by name, nothing touched
    with pytest.raises(SystemExit, match="revocation receipt"):
        _run_reference_only(out)
    assert calls["reference"] == [] and (out / sp.VERDICT_FILE).is_file() and (out / sp.REFERENCE_CHECK_FILE).is_file()
    # --replan / --no-reference / without --resume: refused
    with pytest.raises(SystemExit, match="neither --replan nor --no-reference"):
        sp.main(["run", "--model-key", MODEL, "--out", str(out), "--resume", "--reference-only", "--replan"])
    with pytest.raises(SystemExit, match="pass it with --resume"):
        sp.main(["run", "--model-key", MODEL, "--out", str(out), "--reference-only"])
    # with the front door's receipt for the CURRENT generation: every adapter re-judged, no engine, artifacts archived
    gen = sp.result_generation(out)["generation"]
    nonce = _mint_receipt(out, gen)
    rc = _run_reference_only(out, token=nonce)
    assert rc == 0 and calls["stand_up"] == 0
    (dirs,) = calls["reference"]                                             # ONE judge call over every launch's pending adapters
    (pair,) = plan["pairs"]
    genuine = [a.name for n in pair["manifests"] for a in sm.Manifest.from_dict(plan["manifests"][n]).adapters if a.kind == sm.KIND_GENUINE]
    assert sorted(pair["manifests"]) == ["wo_r4_S1cc", "wo_r4_S1d", "wo_r4_single"]
    assert sorted(Path(d).name for d in dirs) == sorted(genuine) and len(genuine) == 6      # ALL adapters (every manifest's slot dirs), not "the new ones"
    archived = sorted(p.name for p in (out / sp.LAUNCHES_DIR / sp.SUPERSEDED_DIR).rglob("*") if p.is_file())
    assert sp.REFERENCE_CHECK_FILE in archived and sp.VERDICT_FILE in archived and sp.ALLOWED_PAIRS_FILE not in archived or sp.VERDICT_FILE in archived
    assert not (out / sp.VERDICT_FILE).exists()                              # the old certification is withdrawn before the re-judgement
    assert list((out / sp.RECEIPTS_DIR).glob(f"{gen}.{nonce}.used.*.json"))  # the receipt was consumed
    results = sp.load_results(out)
    assert len(results) == 3                                                 # S1cc, S1d and the pair's union: every launch re-judged
    for res in results.values():
        assert res["reference_check"]["status"] == "pass" and res["reference_check"]["basis"] == sp.BASIS_HF_PEFT
    prog = sp.load_progress(out)
    assert prog["complete"] is True and prog[sp.RESULT_GENERATION_KEY] == sp.result_generation(out)["generation"] != gen
    # the carrier's next step, the verdict, now certifies the pair on the new evidence
    verdict, allowed = sp.write_verdict(plan, results, probe_job_id="801", out=out, verdict_uri=str(out / sp.VERDICT_FILE))
    assert verdict["status"] == sp.VERDICT_COMPLETE and [[p["rank"], p["set"]] for p in allowed["pairs"]] == [[4, "wo"]]
    # a second reference-only run needs a NEW receipt (the old one is consumed, the generation moved)
    with pytest.raises(SystemExit, match="revocation receipt"):
        _run_reference_only(out, token=nonce)


def test_reference_only_refuses_while_a_launch_is_still_to_run(tmp_path, monkeypatch):
    out = tmp_path / "rec"; out.mkdir()
    plan = _plan(sets=("wo", "mlp"))                                       # 5 launches; only one of wo's is run
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    sp.build_adapter_dirs(plan, out, FakeExporter())
    rec = next(r for r in plan["launches"] if r["set"] == "wo")
    m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
    res = _passed(rec, m, reference=None)
    res["reference_pending"] = [{"dir": str(sp.slot_dir(out, m.name, a.name)), "name": a.name, "kind": a.kind, "role": a.role, "rank": a.rank,
                                 "lora_modules": ["o_proj"], "vllm_first_token": {"id": 5, "logprob": -4.5}}
                                for a in m.adapters if a.kind == sm.KIND_GENUINE]
    sp.write_json_atomic(sp.launch_result_path(out, rec["id"]), res)
    sp.write_progress(plan, sp.load_results(out), out)
    deps, calls = _ref_deps(plan, lambda pending: {})
    monkeypatch.setattr(sp.Deps, "production", classmethod(lambda cls: deps))
    with pytest.raises(SystemExit, match="still to run"):
        _run_reference_only(out, sets="wo,mlp")
    assert calls["reference"] == []


# ------------------------------------------------------------------------------------------------------ the carrier

def _sbatch_block(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i) + len(end)
    return text[i:j]


def test_the_carrier_maps_phase_reference_to_the_reference_only_resume_and_refuses_an_unknown_phase(tmp_path):
    text = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text()
    assert 'PHASE="${11:-' in text                                                   # positional 11 (the env prefix may back it)
    resume = _sbatch_block(text, "RESUME=()", "\nfi\n")
    check = _sbatch_block(text, 'case "$PHASE" in', "\nesac\n")
    assert text.index('case "$PHASE" in') < text.index('NGPU="$(echo')       # validated before the width block the parallel test runs
    prog = tmp_path / "probe_progress.json"; prog.write_text(json.dumps({"complete": False}))

    def run(phase, attempt, out_dir):
        script = ("set -u\nlog(){ echo \"$@\"; }\nFINAL_RC=0\n"
                  f"PHASE={phase}\nATTEMPT={attempt}\nOUT={out_dir}\nVENV=/usr\n"
                  + check + "\n" + resume + '\nprintf "%s\\n" "${RESUME[*]}"\n')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    def last(r):
        lines = r.stdout.splitlines()
        return lines[-1] if lines else ""
    r = run("reference", 1, tmp_path)
    assert r.returncode == 0 and last(r) == "--resume --reference-only", r
    r = run("full", 2, tmp_path)
    assert r.returncode == 0 and last(r) == "--resume --replan", r
    r = run("full", 1, tmp_path / "nowhere")
    assert r.returncode == 0 and last(r) == "", r                                    # a fresh record: no resume flags
    r = run("bogus", 1, tmp_path)
    assert r.returncode == 2 and "positional 11 phase must be" in (r.stdout + r.stderr), r
    # the self-resubmission never fires for a reference-only carrier
    esc = _sbatch_block(text, "escalate(){", "\n}\n")
    script = "log(){ echo \"$@\"; }\nstop_requested(){ return 1; }\nPHASE=reference\nATTEMPT=1\nMAX_ATTEMPTS=4\nSCRIPT_PATH=/dev/null\n" + esc + "\nescalate why\necho SUCCESSOR=${SUCCESSOR:-none}\n"
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "single-shot" in r.stdout and "SUCCESSOR=none" in r.stdout, r


# ------------------------------------------------------------------------------------------------- the pass rule's terms


def test_first_token_parity_d161_scales_the_tolerance_by_the_measured_floor_and_the_adapters_own_sensitivity():
    """The measured facts of probe 805161, in ONE formula (the parity constants): pass = top-1 agreement
    AND |d| <= max(0.02, 3 x the base-vs-base floor measured on this record, 2 x the adapter's own one-ulp shift). Facts: a base floor of
    0.0179 measured on that record (20x the 4B models'); the attention adapters agree on top-1 with |d| 0.004-0.24 (the scale-100 wo control's 0.24 is
    admitted by its own one-ulp shift 0.139 -> bound 0.278); no term measured = the unscaled rule."""
    hf = _hf(23042, -4.6909)
    floor = 0.0179
    assert sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6909 - 0.004})["status"] == "pass"
    r = sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6909 - 0.03}, floor=floor)                       # 3 x 0.0179 = 0.0537 > 0.02
    assert r["status"] == "pass" and r["tolerance"] == pytest.approx(0.0537) and r["tolerance_terms"]["floor_term"] == pytest.approx(0.0537)
    assert r["tolerance_terms"]["tol"] == sp.REFERENCE_LOGPROB_TOL and r["tolerance_terms"]["sensitivity_term"] is None
    r = sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6909 - 0.24}, floor=floor)
    assert r["status"] == "fail" and r["top1_equal"] is True and r["logprob_abs_delta"] == pytest.approx(0.24)
    r = sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6909 - 0.24}, floor=floor, sensitivity=0.139)    # 2 x 0.139 = 0.278 admits 0.24
    assert r["status"] == "pass" and r["tolerance"] == pytest.approx(0.278) and r["tolerance_terms"]["sensitivity_term"] == pytest.approx(0.278)
    r = sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6909 - 0.03}, floor=floor, sensitivity=1e-4)     # a tiny term never lowers the bound
    assert r["status"] == "pass" and r["tolerance"] == pytest.approx(0.0537)
    assert sp.first_token_parity(hf, {"id": 23042, "logprob": -4.6909 - 0.019}, sensitivity=1e-4)["tolerance"] == sp.REFERENCE_LOGPROB_TOL
    assert sp.first_token_parity(hf, {"id": 1, "logprob": -4.6909}, floor=floor, sensitivity=10.0)["status"] == "fail"   # top-1 disagreement
    # undecidability is the GATE's, not the rule's: a one-ulp flip of the adapter's own greedy token is `parity_chaotic`
    gate = sp.parity_gate({"first_token_changed": True, "max_abs_delta": None}, {"draws": 3, "flips": 3, "max_shift": 0.609})
    assert gate is not None and gate[0] == "parity_chaotic"
    assert (sp.PARITY_FLOOR_K, sp.PARITY_SENSITIVITY_M, sp.PARITY_SENSITIVITY_DRAWS, sp.PARITY_ULP_REL) == (3.0, 2.0, 3, 2 ** -8)
    assert sp.PARITY_REASON_PREFIXES == ("reference_mismatch:", "reference_undecidable:") and sp.REF_STATUS_UNDECIDABLE == "reference_undecidable"


def test_one_ulp_sensitivity_perturbs_lora_b_only_restores_it_and_reports_a_flip():
    import torch

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Module()
            self.base_layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
            self.base_layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
    pm = Wrapped()
    with torch.no_grad():
        pm.base_layer.lora_A["default"].weight.fill_(1.0); pm.base_layer.lora_B["default"].weight.fill_(3.0)
    a0, b0 = pm.base_layer.lora_A["default"].weight.detach().clone(), pm.base_layer.lora_B["default"].weight.detach().clone()
    seen = []

    def forward(m):
        seen.append((m.base_layer.lora_A["default"].weight.detach().clone(), m.base_layer.lora_B["default"].weight.detach().clone()))
        s_ = float(m.base_layer.lora_B["default"].weight.detach().sum())
        return {"top1_id": 7, "top1_logprob": -s_, "top5": [[7, -s_], [8, -s_ - 1.0]]}
    ref = {"top1_id": 7, "top1_logprob": -float(b0.sum())}
    r = sp.one_ulp_sensitivity(pm, forward, ref)
    assert r["draws"] == 3 and r["n_lora_B"] == 1 and len(r["shifts"]) == 3 and r["flips"] == 0 and r["rel"] == sp.PARITY_ULP_REL
    assert all(torch.equal(a, a0) for a, _ in seen) and all(not torch.equal(b, b0) for _, b in seen)          # lora_B moved, lora_A never
    assert all(float((b / b0 - 1).abs().max()) < 8 * sp.PARITY_ULP_REL for _, b in seen)                   # about one ulp
    assert torch.equal(pm.base_layer.lora_B["default"].weight, b0) and torch.equal(pm.base_layer.lora_A["default"].weight, a0)   # restored
    assert r["max_shift"] == max(r["shifts"]) and 0 < r["max_shift"] < 0.5
    # a forward whose greedy token is not the reference's: the adapter flips under its own perturbation (the chaos gate's input)
    r2 = sp.one_ulp_sensitivity(pm, lambda m: {"top1_id": 8, "top1_logprob": -1.0, "top5": [[8, -1.0], [7, -2.0]]}, ref)
    assert r2["flips"] == 3 and torch.equal(pm.base_layer.lora_B["default"].weight, b0) and sp.parity_gate(None, r2)[0] == "parity_chaotic"
    # no lora_B at all: nothing perturbed, the forwards still run, no flip
    r3 = sp.one_ulp_sensitivity(torch.nn.Linear(2, 2), lambda m: {"top1_id": 7, "top1_logprob": ref["top1_logprob"]}, ref)
    assert r3["n_lora_B"] == 0 and r3["flips"] == 0 and r3["max_shift"] == 0.0


def test_launch_reference_check_files_an_undecidable_adapter_as_undecidable_with_the_hf_peft_basis():
    per = {"/r/a/control__lora": {"status": "pass"},
           "/r/a/constitution__lora": {"status": sp.REF_STATUS_UNDECIDABLE, "gate": "parity_chaotic", "reason": "one bf16 ulp flips the greedy token in 3/3 draws"},
           "/r/a/difference__lora": {"status": "pass"}}
    items = [{"dir": d, "name": d.rsplit("/", 1)[1]} for d in per]
    rc = sp.launch_reference_check(items, per)
    assert rc["status"] == sp.REF_STATUS_UNDECIDABLE and rc["basis"] == sp.BASIS_HF_PEFT
    assert rc["undecidable"] == {"constitution__lora": "parity_chaotic: one bf16 ulp flips the greedy token in 3/3 draws"}
    per["/r/a/difference__lora"] = {"status": sp.REF_STATUS_MISSING, "reason": "x"}                      # an unjudged adapter still outranks it
    assert sp.launch_reference_check(items, per)["status"] == sp.REF_STATUS_MISSING and sp.launch_reference_check(items, per)["basis"] is None
    per["/r/a/difference__lora"] = {"status": "fail"}
    rc = sp.launch_reference_check(items, per)
    assert rc["status"] == sp.REF_STATUS_FAIL and rc["basis"] == sp.BASIS_HF_PEFT                          # A judged FAIL outranks undecidable
    assert rc["undecidable"] == {"constitution__lora": "parity_chaotic: one bf16 ulp flips the greedy token in 3/3 draws"}   # … and the gated one is still listed


def test_an_undecidable_adapter_is_a_decided_exclusion_and_the_menu_is_written_for_the_decidable_pairs(tmp_path):
    """The verdict under the scaled bound: the wo pair passes; every mlp adapter flips its own greedy token under one ulp (the
    chaos gate) -> the mlp pair is EXCLUDED `reference_undecidable:…` (decided, evidence), the verdict is COMPLETE and certifying, the menu
    is WRITTEN with the decidable pair; the geometry of both sets is still folded (their serving is certified) with the undecidable pair
    reading `unverified`; the memo tables every adapter's terms — the bound, the floor term, the one-ulp shift and flips, the gate."""
    out = tmp_path / "rec"; out.mkdir()
    plan = _plan(sets=("wo", "mlp"))
    _record(out, plan)

    def judge(pending):
        per = {}
        for it in pending:
            hf = _hf(it["vllm_first_token"]["id"], -4.5)
            if Path(it["dir"]).parent.name.startswith("mlp_"):
                sens = {"draws": 3, "rel": sp.PARITY_ULP_REL, "shifts": [0.5, 0.61, 0.4], "flips": 3, "max_shift": 0.61, "n_lora_B": 40}
                gate = sp.parity_gate(None, sens)
                per[it["dir"]] = {"status": sp.REF_STATUS_UNDECIDABLE, "gate": gate[0], "reason": gate[1], "hf": hf, "sensitivity": sens}
            else:
                sens = {"draws": 3, "rel": sp.PARITY_ULP_REL, "shifts": [0.001, 0.002, 0.0015], "flips": 0, "max_shift": 0.002, "n_lora_B": 10}
                per[it["dir"]] = {**sp.first_token_parity(hf, it["vllm_first_token"], floor=0.0179, sensitivity=0.002), "hf": hf, "sensitivity": sens}
        return per
    deps, _ = _ref_deps(plan, judge)
    assert sp.reference_phase(plan, out, deps)["status"] == "done"
    results = sp.load_results(out)
    # the mlp S1cc / S1d launches carry the genuine adapters and read undecidable; the S2 launch carries none to judge (an empty
    # reference_pending reads `pass` with no adapter) — the PAIR is undecidable through its role-certifying launches
    s1 = {lid: r for lid, r in results.items() if lid.startswith("mlp_r4_S1")}
    assert sorted(s1) == ["mlp_r4_S1cc_cap8_L4096", "mlp_r4_S1d_cap8_L4096"] and all(r["reference_check"]["status"] == sp.REF_STATUS_UNDECIDABLE and r["reference_check"]["basis"] == sp.BASIS_HF_PEFT
                      and r["reference_check"]["undecidable"] for r in s1.values())
    assert all(r["reference_check"]["status"] == "pass" for lid, r in results.items() if lid.startswith("wo_"))
    verdict, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=out, verdict_uri=str(out / sp.VERDICT_FILE), geometry_dir=tmp_path / "site")
    assert verdict["status"] == sp.VERDICT_COMPLETE and verdict["certifying"] is True and verdict["unreferenced_pairs"] == []
    assert [(p["rank"], p["set"]) for p in allowed["pairs"]] == [(4, "wo")]
    (ex,) = allowed["excluded"]
    assert ex["set"] == "mlp" and ex["decided"] is True and ex["reason"].startswith("reference_undecidable:") \
        and "reference_missing" not in ex["reason"] and "reference_mismatch" not in ex["reason"]
    assert sorted(map(tuple, verdict["serving_certified_pairs"])) == [(4, "mlp"), (4, "wo")] and verdict["geometry_fold"]["status"] == "written"
    ap = {(e["rank"], e["set"]): e["status"] for e in verdict[sp.ADAPTER_PARITY_KEY]["per_pair"]}
    assert ap == {(4, "wo"): "verified", (4, "mlp"): "unverified"} and verdict[sp.ADAPTER_PARITY_KEY]["status"] == "unverified"
    rows = {r["adapter"]: r for r in verdict["memo_inputs"]["reference_parity"] if r["status"] == sp.REF_STATUS_UNDECIDABLE}
    assert rows and all(r["gate"] == "parity_chaotic" and r["one_ulp_flips"] == 3 and r["one_ulp_max_shift"] == 0.61 for r in rows.values())
    passed = {r["adapter"]: r for r in verdict["memo_inputs"]["reference_parity"] if r["status"] == "pass"}
    assert passed and all(r["floor_term"] == pytest.approx(3 * 0.0179) and r["tolerance"] == pytest.approx(3 * 0.0179) for r in passed.values())
    # the CLI: exit 0, the menu on disk with the one decidable pair, RESULT names the undecidable exclusion beside parity + geometry
    rc = sp.cmd_verdict(SimpleNamespace(model_key=MODEL, out=out, probe_job_id="t", verdict_uri=None, geometry_dir=tmp_path / "site", probe_cluster=None))
    line = (out / sp.RESULT_FILE).read_text()
    assert rc == 0 and json.loads((out / sp.ALLOWED_PAIRS_FILE).read_text())["pairs"][0]["set"] == "wo"
    assert line.startswith("state=VERDICT pairs_allowed=1 pairs_excluded=1") and "reference_undecidable:" in line and "parity=unverified" in line and "geometry=written:" in line, line
    memo = sp.render_memo(json.loads((out / sp.VERDICT_FILE).read_text()))
    assert "Reference parity terms (D161/D163, rule (a))" in memo and "0.0179" in memo and "| `reference_undecidable` |" in memo and "parity_chaotic" in memo
    assert "| `pass` | True |" in memo





def test_a_missing_reference_on_one_certifying_launch_dominates_a_pass_on_another_and_is_never_a_mismatch(tmp_path):
    """The mlp pair certifies on S1cc + S1d + S2; S2 carries NO genuine adapter (its reference_check reads `pass` under the hf_peft
    basis with nothing judged) and the S1cc / S1d adapters are `reference_missing` (PEFT attached 0/N). Before: `ref_status` became `pass`
    on S2 and the pair entered the menu `verified`. Now: missing dominates a partial set of passes — the pair is `reference_missing`
    (UNCERTIFIED, no menu), its parity `unverified` naming the S1cc / S1d launches, and it is never relabelled a mismatch. Mixed: S1cc and
    S1d pass but S2 is handed an adapter to judge that reads missing — the same. A judged FAIL beside a missing still excludes the pair as
    a mismatch (decided)."""
    out = tmp_path / "rec"; out.mkdir()
    plan = _plan(sets=("wo", "mlp"))
    _record(out, plan)

    def judge(pending):
        per = {}
        for it in pending:
            hf = _hf(it["vllm_first_token"]["id"], -4.5)
            if Path(it["dir"]).parent.name.startswith("mlp_"):
                per[it["dir"]] = {"status": sp.REF_STATUS_MISSING, "reason": "PEFT attached 0/400 of the exported tensors under their own names"}
            else:
                per[it["dir"]] = {**sp.first_token_parity(hf, it["vllm_first_token"], floor=0.0179), "hf": hf}
        return per
    deps, _ = _ref_deps(plan, judge)
    assert sp.reference_phase(plan, out, deps)["status"] == "done"
    results = sp.load_results(out)
    for lid in ("mlp_r4_S1cc_cap8_L4096", "mlp_r4_S1d_cap8_L4096"):
        assert results[lid]["reference_check"]["status"] == sp.REF_STATUS_MISSING and results[lid]["reference_check"]["basis"] is None
    assert results["mlp_r4_S2_cap8_L4096"]["reference_check"] == {**results["mlp_r4_S2_cap8_L4096"]["reference_check"], "status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {}}   # nothing to judge: the masking `pass`
    verdict, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=out, verdict_uri=str(out / sp.VERDICT_FILE))
    assert verdict["status"] == sp.VERDICT_UNCERTIFIED and verdict["unreferenced_pairs"] == [[4, "mlp"]]
    assert [(p["rank"], p["set"]) for p in allowed["pairs"]] == [(4, "wo")]                     # wo stands in memory; the FILE is withheld
    (ex,) = [e for e in allowed["excluded"] if e["set"] == "mlp"]
    assert ex["reason"].startswith("reference_missing:") and "mlp_r4_S1cc_cap8_L4096" in ex["reason"] and "mlp_r4_S1d_cap8_L4096" in ex["reason"] \
        and "reference_mismatch" not in ex["reason"]
    assert not (out / sp.ALLOWED_PAIRS_FILE).exists()
    ap = {(e["rank"], e["set"]): e for e in verdict[sp.ADAPTER_PARITY_KEY]["per_pair"]}
    assert ap[(4, "mlp")]["status"] == "unverified" and ap[(4, "mlp")]["mismatched_adapters"] == [] and ap[(4, "wo")]["status"] == "verified"
    # mixed: S1cc and S1d pass (each judged its own adapters), S2 is handed an adapter to judge that reads missing -> the pair is still reference_missing
    for lid, res in results.items():
        if lid.startswith("mlp_r4_S1cc"):
            res["reference_check"] = {"status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"control__lora": {"status": "pass"}, "constitution__lora": {"status": "pass"}}}
        if lid.startswith("mlp_r4_S1d"):
            res["reference_check"] = {"status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"difference__lora": {"status": "pass"}}}
        if lid.startswith("mlp_r4_S2"):
            res["reference_check"] = {"status": sp.REF_STATUS_MISSING, "basis": None, "reason": "relu_control__carrier: reference_missing — PEFT attached 0/8",
                                      "per_adapter": {"relu_control__carrier": {"status": sp.REF_STATUS_MISSING}}}
    v2, a2 = sp.write_verdict(plan, results, probe_job_id="t")
    assert v2["status"] == sp.VERDICT_UNCERTIFIED and v2["unreferenced_pairs"] == [[4, "mlp"]] and [(p["rank"], p["set"]) for p in a2["pairs"]] == [(4, "wo")]
    (ex2,) = [e for e in a2["excluded"] if e["set"] == "mlp"]
    assert ex2["reason"].startswith("reference_missing:") and "mlp_r4_S2_cap8_L4096" in ex2["reason"] and "reference_mismatch" not in ex2["reason"]
    # a judged FAIL beside a missing: the pair is excluded as a MISMATCH (decided; a real mismatch is never relabelled), the verdict complete
    for lid, res in results.items():
        if lid.startswith("mlp_r4_S1cc"):
            res["reference_check"] = {"status": "fail", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"control__lora": {"status": "fail"}, "constitution__lora": {"status": "pass"}}}
    v3, a3 = sp.write_verdict(plan, results, probe_job_id="t")
    (ex3,) = [e for e in a3["excluded"] if e["set"] == "mlp"]
    assert ex3["reason"] == "reference_mismatch:control__lora" and ex3["decided"] is True and v3["unreferenced_pairs"] == [] and v3["status"] == sp.VERDICT_COMPLETE
    assert {(e["rank"], e["set"]): e["status"] for e in v3[sp.ADAPTER_PARITY_KEY]["per_pair"]}[(4, "mlp")] == "mismatch"


def test_a_layout_that_errored_was_unavailable_or_was_never_compared_makes_the_adapter_reference_missing():
    """`finish_reference_record` skipped a MISSING compare (`if cmp_ and …`), so an adapter whose independent PEFT layout check
    errored, found peft unavailable, was never attempted, or wrote no compare stayed `pass` — the unsupported-layout case the check exists
    for. Now every adapter the phase mapped needs a WRITTEN layout with a PASSING compare; each failure keeps its diagnostic."""
    hf_ok = lambda i: _hf(i, -4.5)
    per = {"/r/a/x": {"status": "pass", "hf": hf_ok(1)}, "/r/b/x": {"status": "pass", "hf": hf_ok(2)}, "/r/c/x": {"status": "fail", "hf": hf_ok(3)},
           "/r/d/x": {"status": "pass", "hf": hf_ok(4)}, "/r/e/x": {"status": "pass", "hf": hf_ok(5)}, "/r/f/x": {"status": sp.REF_STATUS_UNDECIDABLE, "hf": hf_ok(6), "gate": "parity_chaotic", "reason": "flips"}}
    layouts = {"a": {"status": "error", "error": "RuntimeError: peft save_pretrained failed"},
               "b": {"status": "peft_unavailable", "reason": "import peft failed: No module named peft"},
               "c": {"status": "written", "dir": "/r/layouts/c"},                                         # written, no compare
               "e": {"status": "written", "dir": "/r/layouts/e", "compare": {"pass": True, "n_ours": 20, "n_peft": 20, "only_ours": [], "only_peft": []}},
               "f": {"status": "written", "dir": "/r/layouts/f", "compare": {"pass": True, "n_ours": 20, "n_peft": 20, "only_ours": [], "only_peft": []}}}
    layout_of = {"/r/a/x": "a", "/r/b/x": "b", "/r/c/x": "c", "/r/d/x": "d", "/r/e/x": "e", "/r/f/x": "f"}      # d: never attempted
    rec = sp.finish_reference_record(per, layouts, layout_of)
    got = rec["per_adapter"]
    assert got["/r/a/x"]["status"] == sp.REF_STATUS_MISSING and "status 'error' — RuntimeError: peft save_pretrained failed" in got["/r/a/x"]["reason"]
    assert got["/r/b/x"]["status"] == sp.REF_STATUS_MISSING and "status 'peft_unavailable' — import peft failed" in got["/r/b/x"]["reason"]
    assert got["/r/c/x"]["status"] == sp.REF_STATUS_MISSING and "carries no compare" in got["/r/c/x"]["reason"]
    assert got["/r/d/x"]["status"] == sp.REF_STATUS_MISSING and "no PEFT reference layout was produced" in got["/r/d/x"]["reason"] and "'d'" in got["/r/d/x"]["reason"]
    assert got["/r/e/x"]["status"] == "pass" and got["/r/f/x"]["status"] == sp.REF_STATUS_UNDECIDABLE          # written + passing compare: untouched
    assert rec["counts"][sp.REF_STATUS_MISSING] == 4 and rec["counts"]["pass"] == 1 and rec["counts"][sp.REF_STATUS_UNDECIDABLE] == 1
    # an adapter the phase did not map (a pure caller) is left alone
    assert sp.finish_reference_record({"/r/z/x": {"status": "pass", "hf": hf_ok(9)}}, {}, {})["per_adapter"]["/r/z/x"]["status"] == "pass"
    # the launch check then files the pair's launch as reference_missing with the diagnostic — never a mismatch
    rc = sp.launch_reference_check([{"dir": "/r/a/x", "name": "control__lora"}], got)
    assert rc["status"] == sp.REF_STATUS_MISSING and rc["basis"] is None and "peft save_pretrained failed" in rc["reason"]
