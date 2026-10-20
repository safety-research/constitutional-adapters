"""qwen-seedrep-native-battery — fixer fc3: the S2 PASS derivation's request-log audit refused every controls row with
`no 'request_evidence.jsonl' at the attempt root nor under any child store root` -- the controls driver
(`served_capability_run.py generate`) wires NO request-evidence client, so a REAL controls attempt root (the S2 controls-arditi
root, verified on the volume) carries neither `request_evidence.jsonl` nor `response_evidence.jsonl` anywhere; what it carries is
the driver's own per-draw event stream (`controls_events.jsonl`: cap, sampling_seed as sent, finish_reason, text_sha256, the
engine's completion_id) beside each store, and an engine log whose `Received request` lines state seed, max_tokens, the sampler
and the LoRA name. The audit now reconciles a controls root on THAT shape under two named, QUALIFIED contracts:
`/3-controls-events` (a sampled store: every filed draw bound through its event's completion_id to exactly one engine line whose
seed / cap / sampler / LoRA are re-read from the engine and held to the draw, the seed rule and the frozen row) and
`/3b-mmlu-count` (the MMLU letter readout records no id per item and fires no event: count-level identity only). The verdicts
bind through `arm_binding` as before, the certificate's `_smoke_check` accepts them with the qualification rendered, and a
controls root that DOES carry request evidence -- or any other leg's root -- keeps the original audit path unchanged.

The root here is written through the REAL writers (the judged spool + GenMeta, `ControlsEventWriter`), laid out at the app's
`exp/sampled/controls_cap512__s<s>/cells/<cell>` namespace (three seed namespaces of the one Arditi cell), with an engine log in vLLM 0.25.1's
line shape; then the REAL audit runs over it.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import battery_tables as bt  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from sisteer.eval import controls_events as cev  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
CELL = ac.ARDITI_CELL                      # `arditi_allpos`: the served cell id == the harness id == the LoRA name the engine states
PLAN_ARM = "ar_c0p3__arditi_allpos"        # the plan's ARM LABEL for the Arditi row (F-AO: bound through the verdict's arm_binding, never by name)
CAP = 512
SEEDS = (1, 2, 3)
CONTROL_OF_TASK = {v: k for k, v in ac.CONTROLS_TASK_OF.items()}
# vLLM 0.25.1 request_logger.py:63, the shape of a real controls-arditi engine log (params only, the LoRA the engine served under)
_LINE = ("(APIServer pid=197) INFO 09-20 00:30:23 [request_logger.py:63] Received request {cid}: params: SamplingParams(n=1, presence_penalty={pp}, "
         "frequency_penalty=0.0, repetition_penalty={rp}, temperature={t}, top_p={tp}, top_k={tk}, min_p={mp}, seed={seed}, stop=[], stop_token_ids=[], bad_words=[], "
         "thinking_token_budget=None, include_stop_str_in_output=False, ignore_eos=False, max_tokens={cap}, min_tokens=0, logprobs={lp}, prompt_logprobs=None, "
         "skip_special_tokens=True, spaces_between_special_tokens=True, structured_outputs=None, extra_args=None), lora_request: {lora}.")
_LORA = "LoRARequest(lora_name='{name}', lora_int_id=1, lora_path='/vol/x/{name}', base_model_name=None, tensorizer_config_dict=None, load_inplace=False, is_3d_lora_weight=False)"


def _stub_modal():
    import types

    class _Any:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return self
        def __getattr__(self, name): return _Any()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self): return iter(())
    m = types.ModuleType("modal")
    for name in ("App", "Image", "Volume", "Dict", "Secret", "Function", "FunctionCall", "Cls", "Sandbox", "Proxy", "Queue", "Mount", "Retries", "Tunnel"):
        setattr(m, name, _Any())
    m.__getattr__ = lambda name: _Any()   # type: ignore[attr-defined]
    m.forward = _Any()
    m.enable_output = _Any()
    m.exception = types.SimpleNamespace(NotFoundError=type("NotFoundError", (Exception,), {}), FunctionTimeoutError=type("FunctionTimeoutError", (Exception,), {}))
    return m


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_fc3_controls", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal")
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


def engine_line(cid: str, *, cap: int, seed, sampler: Dict[str, float], lora: str | None, logprobs=None) -> str:
    return _LINE.format(cid=cid, cap=cap, seed=seed, t=sampler["temperature"], tp=sampler["top_p"], tk=sampler["top_k"], mp=sampler["min_p"],
                        pp=sampler["presence_penalty"], rp=sampler["repetition_penalty"], lp=logprobs, lora=(_LORA.format(name=lora) if lora else "None"))


VENDOR = {k: fx.VENDOR_BODY[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")}
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}


def w23_controls_arditi_root(tmp_path: Path, *, n_items: int = 3, view: str = "arditi", arm: str = CELL, engine_lora: "str | None" = CELL,
                             seeds=SEEDS, extra_arms=()) -> dict:
    """The REAL controls attempt root shape (S2 controls-arditi): three seed namespaces of the one Arditi cell under the app's controls
    child dir, each store written by the writer's own classes with `controls_events.jsonl` beside `gen_stats.json`; NO request_evidence.jsonl,
    NO response_evidence.jsonl, NO usage ledger anywhere; MMLU rows without ids; an engine log stating seed / cap / sampler / LoRA per line."""
    root = fx.make_root(tmp_path / "stores", leg="controls", view=view, arms=(arm, *extra_arms))
    draws: Dict[str, dict] = {}                         # request_id -> what was sent (task, item, s)
    cells: Dict[int, Path] = {}
    for s in seeds:
        before = set(r["request_id"] for r in root.records)
        cell = fx.write_controls_store(root, arm=arm, s=s, items_by_control={"xstest": [f"xst{k}" for k in range(n_items)], "or_bench_hard_1k": [f"orb{k}" for k in range(n_items)]},
                                       mmlu_items=[f"mm{k}" for k in range(n_items)], cap=CAP)
        for r in root.records:
            if r["request_id"] not in before:
                draws[r["request_id"]] = {"task": r["task"], "item_id": r["item_id"], "sample_idx": r["sample_idx"], "s": s}
        # the app's namespace for the child store (`seedrep_launch_plan.CONTROLS_OUT_REL`), one per seed
        dest = root.path / "exp" / "sampled" / f"controls_cap{CAP}__s{s}" / "cells" / f"{arm}__controls__t0__a0__92b66dce__0b6527b7"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cell), str(dest))
        shutil.rmtree(root.path / "exp" / "tinycap", ignore_errors=True)
        cells[s] = dest
        # the REAL rows.jsonl of the MMLU readout carries no request / completion id
        rows = [json.loads(l) for l in (dest / "rows.jsonl").read_text().splitlines() if l.strip()]
        (dest / "rows.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("request_id", "completion_id")}) + "\n" for r in rows))
    root.flush_pipeline()
    for name in (ac.EVIDENCE_FILE, ac.RESPONSE_EVIDENCE_FILE, ac.LEDGER_FILE):
        for p, _owner in ac.owned_evidence_files(root.path, name):
            p.unlink()
    resp = {r["request_id"]: r for r in root.responses}
    lines: List[str] = []
    events_by_cell: Dict[int, List[dict]] = {s: [] for s in seeds}
    for rid, d in draws.items():
        r = resp[rid]
        cid = r["completion_id"]
        if d["task"] == ac.MMLU_TASK:
            lines.append(engine_line(cid, cap=1, seed=0, sampler=GREEDY, lora=engine_lora, logprobs=64))
            continue
        control = CONTROL_OF_TASK[d["task"]]
        seed = ac.controls_sampling_seed(d["s"], 0, control, d["item_id"])
        lines.append(engine_line(cid, cap=CAP, seed=seed, sampler=VENDOR, lora=engine_lora))
        events_by_cell[d["s"]].append({"control": control, "cap": CAP, "escalated_from_cap": None, "item_id": d["item_id"], "sampling_seed": seed,
                                       "finish_reason": r["finish_reason"], "at_cap": r["finish_reason"] == "length", "completion_tokens": 17, "prompt_tokens": 40,
                                       "text_sha256": r[ac.RESPONSE_TEXT_SHA_FIELD], "request_id": None, "completion_id": cid, "error": None})
    # the standup's greedy probes, as on the real root
    lines += [engine_line(f"chatcmpl-probe{i:012d}", cap=48, seed=0, sampler=GREEDY, lora=None) for i in range(4)]
    (root.serve / "vllm_server.log").write_text("\n".join(lines) + "\n")
    for s, cell in cells.items():
        w = cev.ControlsEventWriter(cell / cev.CONTROLS_EVENTS_FILENAME, cell_id=arm)
        for e in events_by_cell[s]:
            w.append(e)
        gs = json.loads((cell / "gen_stats.json").read_text())
        for control in ("xstest", "or_bench_hard_1k"):
            gs[control][cev.GEN_STATS_EVENTS_KEY] = str(cell / cev.CONTROLS_EVENTS_FILENAME)
        (cell / "gen_stats.json").write_text(json.dumps(gs))
    return {"root": root, "cells": cells, "draws": draws, "resp": resp, "lines": lines}


def _row():
    return {"standup_id": "controls__arditi__base__shard00", "leg": "controls", "kind": "base", "engine": "arditi", "view": "arditi", "stage": "base", "shard": 0,
            "children": [{"eval_leg": "controls", "arm": PLAN_ARM, "cell": CELL, "s": s, "cap": CAP, "controls": ["xstest", "or_bench_hard_1k", "mmlu"],
                          "task": {"xstest": "judged/xstest_v2", "or_bench_hard_1k": "judged/or_bench_hard_1k", "mmlu": "s2parity/mmlu_letter"}} for s in SEEDS]}


def run_audit(root: Path, out: Path) -> int:
    return rea.main(["--roots", str(root), "--out", str(out), "--legs", "xstest,or_bench_hard_1k,mmlu"])


def test_the_w23_shape_has_no_request_evidence_and_the_old_reader_refused_it(tmp_path):
    w = w23_controls_arditi_root(tmp_path)
    root = w["root"]
    assert not any(root.path.rglob(ac.EVIDENCE_FILE)) and not any(root.path.rglob(ac.RESPONSE_EVIDENCE_FILE)) and not any(root.path.rglob(ac.LEDGER_FILE))
    assert all((c / cev.CONTROLS_EVENTS_FILENAME).is_file() for c in w["cells"].values())
    # F-AR reproduced on the strict reader: the request-evidence records are REQUIRED on every leg but controls
    with pytest.raises(ac.AuditRefusal, match="no 'request_evidence.jsonl' at the attempt root nor under any child store root"):
        ac.read_request_evidence(root.path)
    assert ac.read_request_evidence(root.path, required=False) == []


def test_the_real_audit_agrees_on_the_w23_shaped_controls_arditi_root_under_the_d133_contracts(tmp_path):
    w = w23_controls_arditi_root(tmp_path)
    root = w["root"]
    out = root.path / "audits"
    assert run_audit(root.path, out) == ac.EXIT_OK
    idx = json.loads((out / rea.INDEX_NAME).read_text())["verdicts"]
    assert len(idx) == 9 and {e["leg"] for e in idx} == {"xstest", "or_bench_hard_1k", "mmlu"} and {int(e["s"]) for e in idx} == set(SEEDS)
    for e in idx:
        v = json.loads((out / e["rel"]).read_text())
        assert v["verdict"] == rea.VERDICT_AGREES, (e["rel"], v["problem"])
        assert v["arm"] == CELL and v["arm_binding"]["harness_id"] == CELL and v["arm_binding"]["vmodel"] == f"petri__{CELL}", "bound through the standup's records (F-AN / fp3)"
        assert v["ledger"] == ac.LEDGER_NOT_ON_PATH and "D133" in v["ledger_source"]
        assert all(k in v for k in lp.AUDIT_VERDICT_KEYS) and lp.audit_qualification_problems(v, e["rel"]) == []
        if v["leg"] == "mmlu":
            assert v["reconciliation_contract"] == ac.RECONCILIATION_CONTRACT_MMLU_COUNT and v["attempt_source"] == ac.ATTEMPT_SOURCE_ENGINE_COUNT
            ec = v["checks"]["engine_count"]
            assert ec["n_filed_items"] == 3 and ec["n_sibling_stores_same_weights"] == 3 and ec["n_required"] == 9 and ec["n_engine_mmlu_shaped_under_arm"] == 9
            assert ec["weights"] == f"LoRA {CELL!r}" and v["n_draws_bound"] == 0 and "no per-draw binding" in v["checks"]["row_binding"]["note"]
        else:
            assert v["reconciliation_contract"] == ac.RECONCILIATION_CONTRACT_CONTROLS_EVENTS and v["attempt_source"] == ac.ATTEMPT_SOURCE_CONTROLS_EVENTS
            assert v["n_draws"] == 3 == v["n_draws_bound"] and v["checks"]["controls_events"]["engine_lora_names"] == {CELL: 3}
            assert v["checks"]["controls_events"]["named_by_gen_stats"] is True and v["checks"]["row_binding"]["n_problems"] == 0
        assert "D133" in v["independence_note"] and v["body_digest_check"].startswith("not_available")
        assert "D133" in lp.audit_qualification_line(v)
    summary = (out / rea.SUMMARY_NAME).read_text()
    assert summary.count(ac.RECONCILIATION_CONTRACT_CONTROLS_EVENTS) == 6 and summary.count(ac.RECONCILIATION_CONTRACT_MMLU_COUNT) == 3


def test_the_s2_pass_derivations_audit_step_agrees_through_the_apps_bindings(srv, tmp_path):
    """As the app runs it (`producer_evidence` -> the REAL audit tool with the app's `--legs`, the verdict FILES bound per (child, table leg) through
    the audit's index), then the certificate's `_smoke_check` -- the S2 pass derivation's audit step -- on every bound file: AGREES, qualified."""
    w = w23_controls_arditi_root(tmp_path)
    root = w["root"]
    row = _row()

    def run(argv, **kw):
        # the recorder and the expected-samplers helper are other defects' tools (the fixture's engine log carries no launch attestation)
        if any(str(a).endswith(("record_resolved_sampler.py", "seedrep_ladder_helpers.py")) for a in argv):
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.run(argv, **kw)
    ac.write_json(root.path / lp.RESOLVED_SAMPLER_REL, {"verdict": "AGREES", "source": "fc3 fixture stand-in"})
    ev = srv.producer_evidence(root.path, tree=REPO, W=fx.W_FIXTURE, row=row, serve_dir=root.path / "exp" / "serve", driver_python=sys.executable,
                               env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}, run=run)
    log = (root.path / "children" / "request_evidence_audit.log").read_text()
    assert "9 AGREES, 0 REFUSED" in log and "REFUSED:" not in log, log[-800:]
    aid = ac.read_attempt_root(root.path).attempt_id
    sources = lp.smoke_audit_sources(row, str(root.path), ev)
    assert len(sources) == 9
    facts = []
    for path, child, table_leg in sources:
        rec = ev[lp.audit_verdict_rel(table_leg, child["arm"], child["s"], stage=row["stage"], shard=row["shard"])]
        assert rec.get("absent") is not True and rec["bound_rel"].endswith(f"/{CELL}__s{child['s']}.json"), rec
        f = lp._smoke_check("request_evidence_audit", Path(path).read_text(), path=path, plan={"W": fx.W_FIXTURE}, accepted_attempts=[aid], row=row, aid=aid,
                            root=str(root.path), producer=ev, child=child, table_leg=table_leg)
        assert f["verdict"] == "AGREES" and f["plan_cell"] == CELL and f["plan_arm"] == PLAN_ARM and f["arm_binding"]["harness_id"] == CELL
        assert f["independence_lost"] is True and "D133" in f["qualification"] and lp.audit_qualification_problems(f, path) == []
        facts.append(f)
    tally = lp.audit_reconciliation_tally(facts)
    assert tally["n_controls_events"] == 6 and tally["n_mmlu_count"] == 3 and tally["n_ledger_backed"] == 0 and "D133" in tally["statement"]
    # the recorded facts re-validate at certify (the same rule, without the volume)
    for f in facts:
        assert lp._smoke_source_facts_problems("request_evidence_audit", f, plan={"W": fx.W_FIXTURE}, accepted_attempts=[aid]) == []


@pytest.mark.parametrize("mutate, needle", [
    ("drop_engine_line", "matches 0 engine-log `Received request` line"),
    ("text_sha", "is not the spooled text's"),
    ("no_events", "no controls_events.jsonl beside gen_stats.json"),
    ("wrong_lora", "under LoRA 'other_cell', not the arm's attested adapter 'arditi_allpos'"),
    ("engine_seed", "the ENGINE received seed"),
    ("engine_sampler", "the ENGINE resolved"),
    ("unnamed_stream", "does not name the stream beside it"),
    ("mmlu_short", "the readout did not reach the engine for every item"),
    ("failed_last_event", "records a FAILED attempt"),
])
def test_known_bads_refuse_by_name(tmp_path, mutate, needle):
    w = w23_controls_arditi_root(tmp_path)
    root, cells = w["root"], w["cells"]
    cell = cells[1]
    ev_p = cell / cev.CONTROLS_EVENTS_FILENAME
    events = [json.loads(l) for l in ev_p.read_text().splitlines() if l.strip()]
    log_p = root.serve / "vllm_server.log"
    lines = log_p.read_text().splitlines()
    target = events[0]
    if mutate == "drop_engine_line":
        log_p.write_text("\n".join(l for l in lines if target["completion_id"] not in l) + "\n")
    elif mutate == "text_sha":
        events[0]["text_sha256"] = "0" * 64
        ev_p.write_text("".join(json.dumps(e) + "\n" for e in events))
    elif mutate == "no_events":
        ev_p.unlink()
    elif mutate == "wrong_lora":
        log_p.write_text("\n".join(l.replace(f"lora_name='{CELL}'", "lora_name='other_cell'") if target["completion_id"] in l else l for l in lines) + "\n")
    elif mutate == "engine_seed":
        log_p.write_text("\n".join(l.replace(f"seed={target['sampling_seed']},", "seed=7,") if target["completion_id"] in l else l for l in lines) + "\n")
    elif mutate == "engine_sampler":
        log_p.write_text("\n".join(l.replace("temperature=0.7,", "temperature=1.0,") if target["completion_id"] in l else l for l in lines) + "\n")
    elif mutate == "unnamed_stream":
        gs = json.loads((cell / "gen_stats.json").read_text())
        gs["xstest"].pop(cev.GEN_STATS_EVENTS_KEY)
        (cell / "gen_stats.json").write_text(json.dumps(gs))
    elif mutate == "mmlu_short":
        mm = [l for l in lines if "max_tokens=1," in l]
        log_p.write_text("\n".join(l for l in lines if l != mm[0]) + "\n")
    elif mutate == "failed_last_event":
        w2 = cev.ControlsEventWriter(ev_p, cell_id=CELL)
        w2.append({**{k: target[k] for k in cev.CONTROLS_EVENT_FIELDS if k not in ("v", "kind", "seq", "ts_utc", "cell_id")}, "error": "HTTP 503", "completion_id": None,
                   "finish_reason": None, "at_cap": None, "completion_tokens": None, "prompt_tokens": None, "text_sha256": None})
    out = root.path / "audits"
    assert run_audit(root.path, out) == ac.EXIT_FAILED
    idx = json.loads((out / rea.INDEX_NAME).read_text())["verdicts"]
    refused = [json.loads((out / e["rel"]).read_text()) for e in idx]
    refused = [v for v in refused if v["verdict"] != rea.VERDICT_AGREES]
    assert refused and any(needle in str(v["problem"]) for v in refused), [v["problem"] for v in refused]
    if mutate == "mmlu_short":
        assert all(v["leg"] == "mmlu" for v in refused), "a short MMLU count refuses the MMLU stores only"
    elif mutate != "no_events":
        assert {v["leg"] for v in refused} <= {"xstest"} and all(int(v["s"]) == 1 for v in refused), "one store's defect refuses that store alone"


def test_a_static_root_and_a_controls_root_with_request_evidence_keep_the_d106_path(tmp_path):
    # a static root without request evidence is refused exactly as before (the leniency is controls-only)
    root = fx.make_root(tmp_path / "s", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=2)
    for p, _o in ac.owned_evidence_files(root.path, ac.EVIDENCE_FILE):
        p.unlink()
    with pytest.raises(ac.AuditRefusal, match="no 'request_evidence.jsonl'"):
        rea.load_evidence(ac.read_attempt_root(root.path))
    # a controls root that DOES carry request evidence (the standard fixture shape, its usage ledger beside it): the ledger-backed contract as before --
    # the sampled-controls contracts apply ONLY to a controls root with NO request evidence; with the ledger removed, the ledger-less path, as before
    croot = fx.make_root(tmp_path / "c", leg="controls", view="s1", arms=("none",))
    fx.populate_root(croot, s=1, arms=("none",), n_items=3)
    out = croot.path / "audits"
    assert run_audit(croot.path, out) == ac.EXIT_OK
    verdicts = [json.loads((out / e["rel"]).read_text()) for e in json.loads((out / rea.INDEX_NAME).read_text())["verdicts"]]
    assert len(verdicts) == 3 and {v["reconciliation_contract"] for v in verdicts} == {ac.RECONCILIATION_CONTRACT_LEDGER}
    assert {v["attempt_source"] for v in verdicts} == {"ledger"} and all(v["verdict"] == rea.VERDICT_AGREES for v in verdicts)
    for p, _o in ac.owned_evidence_files(croot.path, ac.LEDGER_FILE):
        p.unlink()
    shutil.rmtree(out)
    assert run_audit(croot.path, out) == ac.EXIT_OK
    verdicts = [json.loads((out / e["rel"]).read_text()) for e in json.loads((out / rea.INDEX_NAME).read_text())["verdicts"]]
    assert {v["reconciliation_contract"] for v in verdicts} == {ac.RECONCILIATION_CONTRACT_LEDGERLESS} and {v["attempt_source"] for v in verdicts} == {"response_evidence"}


def test_the_contract_vocabulary_is_pinned_across_the_producer_the_certificate_and_the_tables():
    assert lp.AUDIT_CONTRACT_CONTROLS_EVENTS == ac.RECONCILIATION_CONTRACT_CONTROLS_EVENTS == bt.CONTRACT_CONTROLS_EVENTS
    assert lp.AUDIT_CONTRACT_MMLU_COUNT == ac.RECONCILIATION_CONTRACT_MMLU_COUNT == bt.CONTRACT_MMLU_COUNT
    assert tuple(lp.AUDIT_CONTRACTS) == tuple(ac.RECONCILIATION_CONTRACTS) == tuple(bt.CONTRACT_ATTEMPT_SOURCE)
    assert lp.AUDIT_ATTEMPT_SOURCE_OF == {v: k for k, v in ac.CONTRACT_OF_ATTEMPT_SOURCE.items()} == bt.CONTRACT_ATTEMPT_SOURCE
    assert ac.CONTROLS_EVENTS_FILENAME == cev.CONTROLS_EVENTS_FILENAME == "controls_events.jsonl"
    # the MMLU engine shape is the driver's letter-read payload
    import served_capability_run as scr
    body = scr._mmlu_scoring_payload("petri__x", "q", system=None, top_logprobs=64, response_prefill=False)
    assert {k: body[k] for k in rea.MMLU_ENGINE_SHAPE} == rea.MMLU_ENGINE_SHAPE
    # the Arditi MMLU store's seed is read off the app's controls namespace (the real path), the lane namespace, and neither -> refuse
    assert ac._s_from_namespace(Path("/r/exp/sampled/controls_cap512__s3/cells/arditi_allpos__controls__t0")) == 3
    assert ac._s_from_namespace(Path("/r/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_s2_sampled_t0/x")) == 2
    assert ac._s_from_namespace(Path("/r/exp/tinycap/cells/x")) is None
    with pytest.raises(ac.AuditRefusal, match="spells two seeds"):
        ac._s_from_namespace(Path("/r/arditi_allpos_seedrep_s2_sampled_t0/controls_cap512__s3/x"))
    # the certificate's qualification reader: the two contracts are ledger-less, body digest not available, a note required
    for cv in (lp.AUDIT_CONTRACT_CONTROLS_EVENTS, lp.AUDIT_CONTRACT_MMLU_COUNT):
        q = {"reconciliation_contract": cv, "ledger": "not_on_path", "ledger_source": "x", "attempt_source": lp.AUDIT_ATTEMPT_SOURCE_OF[cv],
             "evidence_basis": ["controls_events.jsonl"], "body_digest_check": "not_available: none", "independence_note": "D133: ..."}
        assert lp.audit_qualification_problems(q, "t") == [] and "D133" in lp.audit_qualification_line(q)
        assert any("requires ledger in ['not_on_path']" in p for p in lp.audit_qualification_problems({**q, "ledger": "declared"}, "t"))
        assert any("no independence_note" in p for p in lp.audit_qualification_problems({**q, "independence_note": None}, "t"))
        assert any("has no body digest counterpart" in p for p in lp.audit_qualification_problems({**q, "body_digest_check": "checked"}, "t"))
    # battery_tables marks the qualified AGREES per contract, never bare
    r = {"contract_verdict": bt.VERDICT_AGREES, "contract_status": "ok", "independence_lost": True, "contract_version": bt.CONTRACT_CONTROLS_EVENTS}
    assert bt._audit_cell(r) == "AGREES [D133 controls-events]"
    assert bt._audit_cell({**r, "contract_version": f"{bt.CONTRACT_MMLU_COUNT} | {bt.CONTRACT_CONTROLS_EVENTS}"}) == "AGREES [D133 controls-events] [D133 mmlu-count]"
    assert bt._audit_cell({**r, "contract_version": bt.CONTRACT_LEDGERLESS}) == "AGREES [D106 ledger-less]"
    assert bt._audit_cell({**r, "independence_lost": False}) == "AGREES"
