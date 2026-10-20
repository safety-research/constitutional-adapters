"""qwen-seedrep-native-battery: `record_resolved_sampler.py --require-wired` with no `--expect` reads
UNCOMPARED on every row whose clients send TWO samplers by design -- a controls row (xstest / or_bench under the vendor row at the controls cap +
MMLU greedy at max_tokens 1) and a BoN row -- so the S2 / S3 PASS derivations refused `resolved_sampler ... verdict 'UNCOMPARED' is not AGREES`
on every controls and bon row. Fix: PER-TASK expected samplers. The driver-venv helper (`seedrep_ladder_helpers.py expected-samplers`)
resolves each child task's reference from the drivers' own authorities (the config's controls row, the registry's rows by exact ref) into
`<root>/expected_samplers.json`; the app hands it to the recorder as `--expect-tasks`; the recorder partitions the client class by the caps the
tasks declare, compares each group against ITS task's reference and records a verdict per task (`client_task_verdicts`); the pass derivation
accepts a per-task AGREES record and refuses a task that is EMPTY / UNCOMPARED / DIVERGED by name. A task with no expectation is UNCOMPARED by
name, never silently AGREES; two tasks at one cap with different samplers are refused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import record_resolved_sampler as rrs  # noqa: E402
import seedrep_ladder_helpers as helpers  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from test_seedrep_review_fixes import _row, _stub_modal, draft, inputs  # noqa: E402,F401  (the module fixtures)
from test_seedrep_review_fixes import TestSolR3_2_launch_certificate as _CertComponents  # noqa: E402  (aliased: not collected here)

VENDOR = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
XSTEST, ORBENCH, MMLU, BON = "judged/xstest_v2", "judged/or_bench_hard_1k", "s2parity/mmlu_letter", "strongreject_bon"
# the 0.25.1 request_logger.py:63 INFO line, the shape a real controls-arditi engine log carries (params only, no prompt)
_SP = ("(APIServer pid=197) INFO 09-20 00:30:23 [request_logger.py:63] Received request chatcmpl-{i:016x}: params: SamplingParams(n=1, presence_penalty={pp}, "
       "frequency_penalty=0.0, repetition_penalty={rp}, temperature={t}, top_p={tp}, top_k={tk}, min_p=0.0, seed={seed}, stop=[], stop_token_ids=[], "
       "max_tokens={cap}, min_tokens=0, logprobs=None), lora_request: None.")


def _line(i, cap, *, t=0.7, tp=0.8, tk=20, pp=1.5, rp=1.0, seed=None):
    return _SP.format(i=i, cap=cap, t=t, tp=tp, tk=tk, pp=pp, rp=rp, seed=(1000 + i if seed is None else seed))


def controls_log(n_sampled=6, n_mmlu=3, n_probe=2, extra=()):
    lines = [_line(i, 512) for i in range(n_sampled)]
    lines += [_line(100 + i, 1, t=0.0, tp=1.0, tk=0, pp=0.0, seed=0) for i in range(n_mmlu)]
    lines += [_line(200 + i, 48, t=0.0, tp=1.0, tk=0, pp=0.0, seed=0) for i in range(n_probe)]
    return lines + list(extra)


def bon_log(n=8, n_probe=2):
    return [_line(i, 4096) for i in range(n)] + [_line(300 + i, 48, t=0.0, tp=1.0, tk=0, pp=0.0, seed=0) for i in range(n_probe)]


def controls_tasks(**over):
    tasks = {XSTEST: {"caps": [512], "sampler": dict(VENDOR), "source": "cfg"}, ORBENCH: {"caps": [512], "sampler": dict(VENDOR), "source": "cfg"},
             MMLU: {"caps": [1], "sampler": {"temperature": 0.0}, "source": "driver"}}
    for t, spec in over.items():
        tasks[t] = spec
    return {"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": tasks}


def run_recorder(tmp_path, lines, tasks=None, extra=(), name="rec"):
    log = tmp_path / f"{name}.vllm_server.log"
    log.write_text("\n".join(lines) + "\n")
    out = tmp_path / f"{name}.json"
    argv = ["--vllm-log", str(log), "--out", str(out), "--require-wired"]
    if tasks is not None:
        tp = tmp_path / f"{name}.tasks.json"
        tp.write_text(json.dumps(tasks))
        argv += ["--expect-tasks", str(tp)]
    rc = rrs.main(argv + list(extra))
    return rc, (json.loads(out.read_text()) if out.exists() else None)


# --------------------------------------------------------------------------- #
# the recorder                                                                  #
# --------------------------------------------------------------------------- #

def test_two_sampler_controls_row_uncompared_before_agrees_per_task_after(tmp_path):
    rc, before = run_recorder(tmp_path, controls_log(), name="before")
    assert rc == 0 and before["verdict"] == "UNCOMPARED" and "no single reference" in before["why"], "F-AR reproduced: two samplers, no --expect"
    assert "client_task_verdicts" not in before, "without --expect-tasks the record keeps its pre-D133 shape"
    rc, rec = run_recorder(tmp_path, controls_log(), controls_tasks(), name="after")
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    assert rec["client_task_verdicts"] == {XSTEST: "AGREES", ORBENCH: "AGREES", MMLU: "AGREES"}
    c = rec["classes"]["client"]
    assert c["verdict"] == "AGREES" and c["n"] == 9 and c["reference"] is None and c["reference_kind"] == "tasks" and "one reference per task" in c["reference_source"]
    assert c["tasks"][XSTEST]["shares_cap_with"] == [ORBENCH] and c["tasks"][MMLU]["shares_cap_with"] == [] and c["tasks"][MMLU]["n"] == 3
    assert c["by_cap"]["512"]["tasks"] == [ORBENCH, XSTEST] and c["by_cap"]["1"]["verdict"] == "AGREES"
    assert rec["classes"]["probe"]["verdict"] == "AGREES" and rec["classes"]["probe"]["n"] == 2, "the probe class is untouched"
    assert "per task" in rec["why"] and XSTEST in rec["why"] and MMLU in rec["why"]
    assert all(k in rec for k in lp.RESOLVED_SAMPLER_KEYS) and rec["reference"] is None and rec["divergences"] == []
    assert rec["client_task_expectations"]["tasks"][MMLU]["sampler"] == {"temperature": 0.0}, "the expectation document rides in the record"
    # the client caps come from the tasks (the class rule says so); the mmlu cap 1 is a CLIENT cap here, not a probe cap
    assert rec["client_caps"] == ["1", "512"] and "--expect-tasks" in rec["client_cap_source"] and rec["probe_caps"] == ["48"]


def test_a_bon_row_agrees_per_task(tmp_path):
    rc, before = run_recorder(tmp_path, bon_log(), name="bon-before")
    assert before["verdict"] == "UNCOMPARED"
    tasks = {"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": {BON: {"caps": [4096], "sampler": dict(VENDOR), "source": "registry qwen35_4b_it|bon|off|vendor_preset@v1"}}}
    rc, rec = run_recorder(tmp_path, bon_log(), tasks, name="bon-after")
    assert rc == 0 and rec["verdict"] == "AGREES" and rec["client_task_verdicts"] == {BON: "AGREES"} and rec["n_client_requests"] == 8


def test_a_task_with_a_disagreeing_sampler_is_diverged_by_task_name(tmp_path):
    tasks = controls_tasks(**{XSTEST: {"caps": [512], "sampler": {**VENDOR, "temperature": 1.0}, "source": "cfg"},
                              ORBENCH: {"caps": [512], "sampler": {**VENDOR, "temperature": 1.0}, "source": "cfg"}})
    rc, rec = run_recorder(tmp_path, controls_log(), tasks, name="div")
    assert rc == 0, "a finding about the sampler stays exit 0 under --require-wired"
    assert rec["verdict"] == "DIVERGED" and f"task {XSTEST}" in rec["why"] and f"task {ORBENCH}" in rec["why"] and "resolved to a value other than its reference" in rec["why"]
    assert rec["client_task_verdicts"] == {XSTEST: "DIVERGED", ORBENCH: "DIVERGED", MMLU: "AGREES"}
    d = rec["classes"]["client"]["tasks"][XSTEST]["divergences"]
    assert d and d[0]["key"] == "temperature" and d[0]["sent"] == 1.0 and d[0]["resolved"] == 0.7 and d[0]["tasks"] == [ORBENCH, XSTEST] and d[0]["max_tokens"] == "512"
    assert rec["divergences"] == d, "the top-level divergences are the per-task ones, stamped with task and cap"
    # a greedy request at the SAMPLED task's cap is a within-task divergence (the engine resolved greedy for a sampled task)
    rc, rec = run_recorder(tmp_path, controls_log(extra=[_line(900, 512, t=0.0, tp=1.0, tk=0, pp=0.0, seed=0)]), controls_tasks(), name="div2")
    assert rec["verdict"] == "DIVERGED" and rec["client_task_verdicts"][XSTEST] == "DIVERGED" and rec["client_task_verdicts"][MMLU] == "AGREES"


def test_a_task_without_an_expectation_is_uncompared_by_name_never_folded(tmp_path):
    tasks = controls_tasks(**{MMLU: {"caps": [1], "sampler": None, "source": "", "why": "no resolver for this readout"}})
    rc, rec = run_recorder(tmp_path, controls_log(), tasks, name="null")
    assert rc == 0 and rec["verdict"] == "UNCOMPARED"
    assert MMLU in rec["why"] and "no resolver for this readout" in rec["why"] and "never folded" in rec["why"]
    assert rec["client_task_verdicts"] == {XSTEST: "AGREES", ORBENCH: "AGREES", MMLU: "UNCOMPARED"}
    assert rec["classes"]["client"]["tasks"][MMLU]["reference"] is None and rec["classes"]["client"]["tasks"][MMLU]["why"] == "no resolver for this readout"
    # a shared cap where ONE task has no reference leaves BOTH uncompared (their requests cannot be told apart)
    tasks = controls_tasks(**{ORBENCH: {"caps": [512], "sampler": None, "source": "", "why": "x"}})
    rc, rec = run_recorder(tmp_path, controls_log(), tasks, name="null2")
    assert rec["verdict"] == "UNCOMPARED" and rec["client_task_verdicts"][XSTEST] == "UNCOMPARED" == rec["client_task_verdicts"][ORBENCH]
    # a declared task with NO request at its cap is EMPTY (reported; the certificate refuses it by name -- see the pass-reader test)
    rc, rec = run_recorder(tmp_path, controls_log(n_mmlu=0), controls_tasks(), name="empty")
    assert rec["verdict"] == "AGREES" and rec["client_task_verdicts"][MMLU] == "EMPTY" and "EMPTY: no request" in rec["why"]
    # a client request at a cap NO task declares is unclassified, never folded into a task
    rc, rec = run_recorder(tmp_path, controls_log(extra=[_line(950, 2048)]), controls_tasks(), name="stray")
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["n_unclassified_requests"] == 1 and rec["client_task_verdicts"][XSTEST] == "AGREES"
    assert rec["classes"]["unclassified"]["distinct_resolved"][0]["matches_client_reference"] is True, "the stray tuple is still cross-checked against the task references"


def test_refusals_two_tasks_one_cap_different_samplers_and_expect_beside_expect_tasks(tmp_path, capsys):
    tasks = controls_tasks(**{ORBENCH: {"caps": [512], "sampler": {**VENDOR, "temperature": 0.9}, "source": "cfg"}})
    rc, rec = run_recorder(tmp_path, controls_log(), tasks, name="collide")
    assert rc == 2 and rec is None and "DIFFERENT samplers" in capsys.readouterr().err
    rc, rec = run_recorder(tmp_path, controls_log(), controls_tasks(), extra=["--expect", "temperature=0.7"], name="both")
    assert rc == 2 and rec is None and "both name the CLIENT class's reference" in capsys.readouterr().err
    for bad, needle in (({"schema": "x", "tasks": {}}, "not a resolved_sampler_task_expectations/1"), ({"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": {}}, "non-empty object"),
                        ({"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": {XSTEST: {"caps": [], "sampler": VENDOR, "source": ""}}}, "non-empty list of positive ints"),
                        ({"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": {XSTEST: {"caps": [512], "sampler": {"n": 1}, "source": ""}}}, "are not sampler keys"),
                        ({"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": {XSTEST: {"caps": [512], "sampler": {}, "source": ""}}}, "non-empty object of sampler keys, or null"),
                        ({"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": {XSTEST: {"caps": [48], "sampler": VENDOR, "source": ""}}}, "both a client cap")):
        rc, rec = run_recorder(tmp_path, controls_log(), bad, name="bad")
        assert rc == 2 and needle in capsys.readouterr().err, (bad, needle)
    # the per-task verdict vocabulary is the class vocabulary the contracts pin
    from sisteer.capeval.tinycap import contracts as C
    assert {"EMPTY", "AGREES", "DIVERGED", "UNCOMPARED"} <= set(C.RESOLVED_SAMPLER_CLASS_VERDICTS)


# --------------------------------------------------------------------------- #
# the helper: the expectations from the drivers' own authorities                #
# --------------------------------------------------------------------------- #

def test_the_helper_resolves_the_real_controls_bon_and_static_rows_and_names_what_it_cannot(draft, tmp_path):
    import yaml
    from sisteer.eval.cell_controls import controls_gen_config
    from sisteer.eval.attacks import bon_served
    for engine in ("s1", "arditi"):
        row = _row(draft, "controls", engine=engine)
        out = tmp_path / f"controls_{engine}.json"
        assert helpers.expected_samplers(row, REPO, out) == 0
        doc = json.loads(out.read_text())
        assert doc["schema"] == rrs.EXPECT_TASKS_SCHEMA == helpers.EXPECT_TASKS_SCHEMA and doc["standup_id"] == row["standup_id"]
        cfg = yaml.safe_load((REPO / row["children"][0]["config"]).read_text())
        want = {k: controls_gen_config(cfg)[k] for k in rrs.SAMPLER_KEYS}
        assert doc["tasks"][XSTEST] == {"caps": [512], "sampler": want, "source": doc["tasks"][XSTEST]["source"]} and "controls_gen_config(" in doc["tasks"][XSTEST]["source"]
        assert doc["tasks"][ORBENCH]["sampler"] == want and doc["tasks"][ORBENCH]["caps"] == [512]
        assert doc["tasks"][MMLU] == {"caps": [1], "sampler": {"temperature": 0.0}, "source": doc["tasks"][MMLU]["source"]} and "_mmlu_scoring_payload" in doc["tasks"][MMLU]["source"]
        assert set(doc["tasks"]) == {XSTEST, ORBENCH, MMLU}
        # the recorder accepts the document as written (round trip through its reader)
        rrs.read_task_expectations(out)
    row = _row(draft, "bon")
    out = tmp_path / "bon.json"
    assert helpers.expected_samplers(row, REPO, out) == 0
    doc = json.loads(out.read_text())
    from sisteer.registry.decode_policy import PolicyFile
    pf = PolicyFile.load(str(REPO / lp.DECODE_REGISTRY_REL))
    params = pf.by_ref(lp.BATTERY_DECODE_REFS["bon"]).request_params()
    want_bon = {k: params[k] for k in rrs.SAMPLER_KEYS if k in params}                       # the registry's vendor row for BoN, by the plan's exact ref
    assert doc["tasks"] == {BON: {"caps": [4096], "sampler": want_bon, "source": doc["tasks"][BON]["source"]}} and lp.BATTERY_DECODE_REFS["bon"] in doc["tasks"][BON]["source"]
    assert want_bon["temperature"] == 0.7 and set(want_bon) == set(rrs.SAMPLER_KEYS)
    # the same six numbers the driver draws under (bon_served.bon_policy resolves the live default row, which must be that vendor_preset row)
    assert dict(bon_served.bon_sampler(lp.MODEL_KEY, bon_served.BON_VENDOR_PRESET)) == want_bon
    row = _row(draft, "pasttense")
    out = tmp_path / "static.json"
    assert helpers.expected_samplers(row, REPO, out) == 0
    doc = json.loads(out.read_text())
    params = pf.by_ref(lp.BATTERY_DECODE_REFS["static"]).request_params()
    assert doc["tasks"]["static/pasttense"]["sampler"] == {k: params[k] for k in rrs.SAMPLER_KEYS if k in params} and doc["tasks"]["static/pasttense"]["caps"] == [4096]
    assert lp.BATTERY_DECODE_REFS["static"] in doc["tasks"]["static/pasttense"]["source"]
    # a leg with no resolver, a controls child with no config, and children that disagree: named, sampler null, exit 0 (the recorder says UNCOMPARED)
    odd = {"standup_id": "x", "leg": "controls", "children": [
        {"eval_leg": "weird_leg", "task": "w/t", "cap": 77},
        {"eval_leg": "controls", "task": {"xstest": XSTEST}, "cap": 512},
        {"eval_leg": "bon", "task": BON, "cap": 4096},
        {"eval_leg": "bon", "task": BON, "cap": 2048}]}
    out = tmp_path / "odd.json"
    assert helpers.expected_samplers(odd, REPO, out, registry_sampler=lambda ref: dict(VENDOR)) == 0
    doc = json.loads(out.read_text())
    assert doc["tasks"]["w/t"]["sampler"] is None and "no expected-sampler resolver" in doc["tasks"]["w/t"]["why"]
    assert doc["tasks"][XSTEST]["sampler"] is None and "no config" in doc["tasks"][XSTEST]["why"]
    assert doc["tasks"][BON] == {"caps": [2048, 4096], "sampler": dict(VENDOR), "source": doc["tasks"][BON]["source"]}, "one task at two caps: both caps, one reference"
    assert helpers.expected_samplers({"standup_id": "x", "children": []}, REPO, tmp_path / "none.json") == 3
    # the CLI subcommand
    rp = tmp_path / "row.json"
    rp.write_text(json.dumps(_row(draft, "bon"), default=str))
    r = subprocess.run([sys.executable, str(REPO / "scripts/modal/seedrep_ladder_helpers.py"), "expected-samplers", "--row", str(rp), "--tree", str(REPO), "--out", str(tmp_path / "cli.json")],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"})
    assert r.returncode == 0 and json.loads((tmp_path / "cli.json").read_text())["tasks"][BON]["caps"] == [4096], r.stderr[-500:]


def test_the_mmlu_expectation_is_pinned_to_the_drivers_letter_read_payload():
    import served_capability_run as scr
    body = scr._mmlu_scoring_payload("petri__x", "q", system=None, top_logprobs=64, response_prefill=False)
    assert body["max_tokens"] == helpers.MMLU_SCORING_CAP == 1 and body["temperature"] == helpers.MMLU_EXPECTED_SAMPLER["temperature"] == 0.0
    assert not (set(rrs.SAMPLER_KEYS) & set(body)) - {"temperature"}, "the letter read sends no other sampler key: the reference pins temperature alone"
    assert helpers.EXPECT_TASKS_SCHEMA == rrs.EXPECT_TASKS_SCHEMA


# --------------------------------------------------------------------------- #
# the app: producer_evidence hands the recorder the row's expectations           #
# --------------------------------------------------------------------------- #

def _load_served():
    import importlib.util
    spec = importlib.util.spec_from_file_location("seedrep_served_fc3_tasks", REPO / "scripts/modal/seedrep_served.py")
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


def test_producer_evidence_runs_the_helper_and_hands_the_recorder_expect_tasks(draft, tmp_path):
    srv = _load_served()
    row = _row(draft, "controls", engine="arditi")
    root = tmp_path / "root"
    (root / "exp" / "serve").mkdir(parents=True)
    (root / "exp" / "serve" / "endpoint.json").write_text(json.dumps({"vmodels": {"arditi_allpos": "petri__arditi_allpos"}}))
    calls = []

    def fake_run(argv, **kw):
        calls.append(list(argv))
        if argv[1].endswith("seedrep_ladder_helpers.py"):
            assert argv[2] == "expected-samplers" and argv[argv.index("--tree") + 1] == str(REPO)
            row_d = json.loads(Path(argv[argv.index("--row") + 1]).read_text())
            assert row_d["standup_id"] == row["standup_id"]
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps(controls_tasks()))
            return subprocess.CompletedProcess(argv, 0)
        if argv[1].endswith("record_resolved_sampler.py"):
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"verdict": "AGREES", "client_task_verdicts": {}}))
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.CompletedProcess(argv, 2)
    ev = srv.producer_evidence(root, tree=REPO, W=draft["W"], row=row, serve_dir=root / "exp" / "serve", driver_python=sys.executable,
                               env={"PATH": "/usr/bin:/bin"}, run=fake_run)
    rec_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert rec_argv[rec_argv.index("--expect-tasks") + 1] == str(root / lp.EXPECTED_SAMPLERS_REL) and "--expect" not in rec_argv and "--require-wired" in rec_argv
    assert calls.index([a for a in calls if a[1].endswith("seedrep_ladder_helpers.py")][0]) < calls.index(rec_argv), "the helper runs before the recorder"
    assert ev[lp.EXPECTED_SAMPLERS_REL]["writer"] == "scripts/modal/seedrep_ladder_helpers.py" and ev[lp.EXPECTED_SAMPLERS_REL]["rc"] == 0 and "sha256" in ev[lp.EXPECTED_SAMPLERS_REL]
    assert ev[lp.RESOLVED_SAMPLER_REL]["writer"] == lp.PRODUCER_WRITERS["resolved_sampler"]
    # the helper failing: the recorder runs WITHOUT expectations (it will read UNCOMPARED, loud) and the log says why; nothing is invented
    calls.clear()
    root2 = tmp_path / "root2"
    (root2 / "exp" / "serve").mkdir(parents=True)

    def failing_run(argv, **kw):
        calls.append(list(argv))
        if argv[1].endswith("seedrep_ladder_helpers.py"):
            return subprocess.CompletedProcess(argv, 3)
        if argv[1].endswith("record_resolved_sampler.py"):
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"verdict": "UNCOMPARED"}))
        return subprocess.CompletedProcess(argv, 0)
    ev = srv.producer_evidence(root2, tree=REPO, W=draft["W"], row=row, serve_dir=root2 / "exp" / "serve", driver_python=sys.executable, env={"PATH": "/usr/bin:/bin"}, run=failing_run)
    rec_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert "--expect-tasks" not in rec_argv and ev[lp.EXPECTED_SAMPLERS_REL]["absent"] is True and ev[lp.EXPECTED_SAMPLERS_REL]["rc"] == 3
    assert "no per-task sampler expectations" in (root2 / "children" / "resolved_sampler.log").read_text()


# --------------------------------------------------------------------------- #
# the pass derivation (the real reader) over the real recorder's records         #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def components(inputs, draft, tmp_path_factory):
    return _CertComponents()._components(inputs, draft, tmp_path_factory.mktemp("fc3-tasks"))


def _swap_recorder_record(c, smoke_id: str, text: str):
    """The smoke row's `resolved_sampler.json` on the fake volume REPLACED by `text`, its producer binding re-hashed (the app binds by sha)."""
    rid = c["rows_by_smoke"][smoke_id][0]
    root = c["smokes"][smoke_id]["rows"][rid]["attempt_root"]
    texts = dict(c["texts"])
    texts[f"{root}/{lp.RESOLVED_SAMPLER_REL}"] = text
    ev = json.loads(texts[f"{root}/modal_evidence.json"])
    ev[lp.PRODUCER_EVIDENCE_KEY][lp.RESOLVED_SAMPLER_REL]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
    return rid, lp.volume_only(lambda path: texts.get(str(path)))


def test_the_pass_derivation_refuses_the_uncompared_record_and_accepts_the_per_task_agrees(components, tmp_path):
    c = components
    draft_ = c["draft"]
    for smoke_id, lines, tasks in (("S2", controls_log(), controls_tasks()), ("S3", bon_log(), {"schema": rrs.EXPECT_TASKS_SCHEMA, "tasks": {BON: {"caps": [4096], "sampler": dict(VENDOR), "source": "registry"}}})):
        getter = c["getters"][smoke_id]
        # BEFORE: the recorder as the app ran it previously (no expectations) -> UNCOMPARED -> the derivation refuses by name
        _rc, before = run_recorder(tmp_path, lines, name=f"{smoke_id}-before")
        rid, rt = _swap_recorder_record(c, smoke_id, json.dumps(before))
        with pytest.raises(ValueError, match="resolved_sampler: .* verdict 'UNCOMPARED' is not AGREES"):
            lp.derive_smoke_pass(rs=c["rs"], plan=draft_, smoke_id=smoke_id, standup_ids=[rid], accepted_get=getter, read_text=rt)
        # AFTER: per-task AGREES -> derived; the facts carry the per-task verdicts; the record re-validates
        _rc, after = run_recorder(tmp_path, lines, tasks, name=f"{smoke_id}-after")
        assert after["verdict"] == "AGREES"
        rid, rt = _swap_recorder_record(c, smoke_id, json.dumps(after))
        rec = lp.derive_smoke_pass(rs=c["rs"], plan=draft_, smoke_id=smoke_id, standup_ids=[rid], accepted_get=getter, read_text=rt, derived_utc="2026-09-20T00:00:00Z")
        src = rec["checks"]["resolved_sampler"]["sources"][0]
        assert src["verdict"] == "AGREES" and src["client_task_verdicts"] == after["client_task_verdicts"] and all(v == "AGREES" for v in src["client_task_verdicts"].values())
        assert lp.smoke_pass_problems(rec, smoke_id=smoke_id, plan=draft_, execution_wave_uuid=c["waves"][smoke_id], draft_launch_plan_sha=draft_["launch_plan_sha"], read_text=rt,
                                      accepted_get=getter) == []
        # a recorded fact flipped after derivation (a task's verdict edited to AGREES from what the record says) -> certify refuses by name
        bad = copy.deepcopy(rec)
        bad["checks"]["resolved_sampler"]["sources"][0]["client_task_verdicts"][next(iter(after["client_task_verdicts"]))] = "EMPTY"
        probs = lp.smoke_pass_problems(bad, smoke_id=smoke_id, plan=draft_, execution_wave_uuid=c["waves"][smoke_id], draft_launch_plan_sha=draft_["launch_plan_sha"])
        assert any("records per-task verdict(s)" in p and "not AGREES for every task (D133)" in p for p in probs), probs
    # a DIVERGED task -> refused by task name; an EMPTY task under a top-level AGREES -> refused by task name (never silently certified)
    getter = c["getters"]["S2"]
    _rc, div = run_recorder(tmp_path, controls_log(), controls_tasks(**{XSTEST: {"caps": [512], "sampler": {**VENDOR, "temperature": 1.0}, "source": "cfg"},
                                                                       ORBENCH: {"caps": [512], "sampler": {**VENDOR, "temperature": 1.0}, "source": "cfg"}}), name="S2-div")
    rid, rt = _swap_recorder_record(c, "S2", json.dumps(div))
    with pytest.raises(ValueError, match="verdict 'DIVERGED' is not AGREES .*task judged/xstest_v2"):
        lp.derive_smoke_pass(rs=c["rs"], plan=draft_, smoke_id="S2", standup_ids=[rid], accepted_get=getter, read_text=rt)
    _rc, empty = run_recorder(tmp_path, controls_log(n_mmlu=0), controls_tasks(), name="S2-empty")
    assert empty["verdict"] == "AGREES" and empty["client_task_verdicts"][MMLU] == "EMPTY"
    rid, rt = _swap_recorder_record(c, "S2", json.dumps(empty))
    with pytest.raises(ValueError, match="per-task verdict\\(s\\) are not AGREES: \\{'s2parity/mmlu_letter': 'EMPTY'\\}"):
        lp.derive_smoke_pass(rs=c["rs"], plan=draft_, smoke_id="S2", standup_ids=[rid], accepted_get=getter, read_text=rt)
    # a record without the per-task block (a single-sampler row under --expect) keeps the earlier rule
    _rc, single = run_recorder(tmp_path, bon_log(), None, extra=["--expect", "temperature=0.7", "--expect", "top_p=0.8"], name="S3-single")
    assert single["verdict"] == "AGREES" and "client_task_verdicts" not in single
    rid, rt = _swap_recorder_record(c, "S3", json.dumps(single))
    rec = lp.derive_smoke_pass(rs=c["rs"], plan=draft_, smoke_id="S3", standup_ids=[rid], accepted_get=c["getters"]["S3"], read_text=rt)
    assert "client_task_verdicts" not in rec["checks"]["resolved_sampler"]["sources"][0]
