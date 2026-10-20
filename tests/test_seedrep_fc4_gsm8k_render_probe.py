"""qwen-seedrep-native-battery: a gsm8k row's RENDER PROBE -- the driver's ONE
1-item request per thinking arm at max_tokens 512 decoding under the thinking card (`served_capability_run._render_probe`; S3 gsm8k
evidence: `client UNCOMPARED n=1 temperature=0.6 top_p=0.95 top_k=20 ... max_tokens=512`) -- is a PROBE-class request (512 is a pinned probe cap)
that the default greedy probe reference would file as UNCLASSIFIED / DIVERGED. The app now hands the recorder, DERIVED from the plan row by the
driver-venv helper, `--probe-expect kinds --probe-tags <each gsm8k child's <out-root>/request_evidence.jsonl> --expect <the row's thinking-card
sampler>` alongside `--expect-tasks`: the render probe is classified by the driver's own `probe/render` tag and compared against the pinned
card row, the bites by their `probe/bites` tag against greedy, the task requests at the row's cap against the same card row per task. The
controls / static / bon paths are byte-identical (no probe block, no kinds flags).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import record_resolved_sampler as rrs  # noqa: E402
import seedrep_ladder_helpers as helpers  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from sisteer.eval.attacks.served_target import PROBE_MAX_TOKENS  # noqa: E402
from test_seedrep_review_fixes import _row, _stub_modal, draft, inputs  # noqa: E402,F401  (the module fixtures)

GSM8K_TASK = "inspect/gsm8k_paired200"
CARD = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}      # qwen35_4b_it|gsm8k|on|thinking_card@v1, request_params
GREEDY = dict(t=0.0, tp=1.0, tk=0, pp=0.0)
_SP = ("(APIServer pid=197) INFO 09-20 00:30:23 [request_logger.py:63] Received request chatcmpl-{i:016x}: params: SamplingParams(n=1, presence_penalty={pp}, "
       "frequency_penalty=0.0, repetition_penalty=1.0, temperature={t}, top_p={tp}, top_k={tk}, min_p=0.0, seed={seed}, stop=[], max_tokens={cap}, min_tokens=0), "
       "lora_request: None.")
# the S3 gsm8k shape + the task requests a COMPLETED row adds: greedy bites at 48, ONE render probe at 512 under the card, task draws at 8192 under the card
_ATTEST = ("(APIServer pid=1) INFO 09-12 02:05:04 [request_logger.py:28] `--enable-log-requests` is set but the minimum log level is higher than DEBUG. "
           "Only limited information will be logged to minimize overhead. To view more details, set `VLLM_LOGGING_LEVEL=DEBUG`.")


def _line(i, cap, *, t, tp, tk, pp, seed):
    return _SP.format(i=i, cap=cap, t=t, tp=tp, tk=tk, pp=pp, seed=seed)


def w20_gsm8k_log(n_bites=10, n_tasks=3, cap=8192):
    card = dict(t=0.6, tp=0.95, tk=20, pp=0.0)
    lines = [_ATTEST]
    lines += [_line(i, PROBE_MAX_TOKENS, **GREEDY, seed=0) for i in range(n_bites)]
    lines += [_line(100, 512, **card, seed=7)]
    lines += [_line(200 + i, cap, **card, seed=5000 + i) for i in range(n_tasks)]
    return lines


def tag_row(kind, cap, n=0, vmodel="petri__base"):
    return {"request_evidence_version": 2, "ts": "2026-09-20T00:00:00Z", "kind": "probe", "task": f"probe/{kind}", "sample": f"{vmodel}/{n}", "vmodel": vmodel,
            "petri_thinking_on_present": False, "petri_thinking_on_value": None, "enable_thinking_present": True, "enable_thinking_value": True,
            "body_sha256": "0" * 64, "request_id": f"{kind}{n}".ljust(32, "0"), "body_bytes": 181, "max_tokens": cap, "attempt_id": "a1"}


def _load_served():
    spec = importlib.util.spec_from_file_location("seedrep_served_fc4", REPO / "scripts/modal/seedrep_served.py")
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


def test_the_helper_derives_the_gsm8k_probe_block_from_the_plan_row_and_nothing_else_gains_one(draft, tmp_path):
    row = _row(draft, "gsm8k")
    out = tmp_path / "gsm8k.json"
    assert helpers.expected_samplers(row, REPO, out) == 0
    doc = json.loads(out.read_text())
    assert doc["tasks"][GSM8K_TASK]["sampler"] == CARD and doc["tasks"][GSM8K_TASK]["caps"] == [8192]
    probe = doc["probe"]
    assert probe["probe_expect"] == "kinds" and probe["expect"] == CARD and lp.BATTERY_DECODE_REFS["gsm8k"] in probe["source"]
    want_tags = sorted({f"{lp.GSM8K_OUT_REL.format(cap=int(k['cap']), s=int(k['s']), cell=k['cell'])}/request_evidence.jsonl" for k in row["children"]})
    assert probe["tags_rel"] == want_tags and len(want_tags) == len(row["children"]) and all(t.startswith("exp/sampled/gsm8k_thinking_on_cap8192__s1/") for t in want_tags)
    # the recorder accepts the document (the probe block rides along as a top-level key)
    exp = rrs.read_task_expectations(out)
    assert exp["probe"] == probe
    # rows without a gsm8k child gain NO probe block: controls / static / bon paths are byte-identical to fc3's
    for leg in ("controls", "pasttense", "bon"):
        o = tmp_path / f"{leg}.json"
        assert helpers.expected_samplers(_row(draft, leg), REPO, o) == 0
        assert "probe" not in json.loads(o.read_text())
    # an unresolvable card: the block says why with `expect: null` (the recorder then refuses kinds mode loudly, exit 2 -- never a guessed reference)
    blk = helpers.gsm8k_probe_block(row["children"], helpers._plan_module(), registry_sampler=lambda ref: (_ for _ in ()).throw(ValueError("no such row")))
    assert blk["expect"] is None and "no such row" in blk["why"]


def test_producer_evidence_runs_the_real_recorder_in_kinds_mode_on_a_w20_shaped_gsm8k_log_and_every_task_and_kind_agrees(draft, tmp_path):
    srv = _load_served()
    row = _row(draft, "gsm8k")
    root = tmp_path / "root"
    serve = root / "exp" / "serve"
    serve.mkdir(parents=True)
    (serve / "vllm_server.log").write_text("\n".join(w20_gsm8k_log()) + "\n")
    (serve / "endpoint.json").write_text(json.dumps({"vmodels": {"none": "petri__base"}}))
    # the `none` child's request evidence at ITS declared store root (the other children never ran here: their tag files are recorded missing)
    none_child = [k for k in row["children"] if k["cell"] == "none"][0]
    tags_p = root / lp.GSM8K_OUT_REL.format(cap=none_child["cap"], s=none_child["s"], cell="none") / "request_evidence.jsonl"
    tags_p.parent.mkdir(parents=True)
    tags = [tag_row("bites", PROBE_MAX_TOKENS, n=i) for i in range(10)] + [tag_row("render", 512)] + \
        [{**tag_row("x", 8192, n=i), "kind": "task", "task": GSM8K_TASK, "sample": f"g{i}/0"} for i in range(3)]
    tags_p.write_text("".join(json.dumps(t) + "\n" for t in tags))
    calls = []

    def run(argv, **kw):
        calls.append([str(a) for a in argv])
        if any(str(a).endswith("request_evidence_audit.py") for a in argv):
            return subprocess.CompletedProcess(argv, 0)                                 # the audit tool is another check's; not under test here
        return subprocess.run(argv, **kw)                                               # the REAL helper and the REAL recorder, this interpreter
    ev = srv.producer_evidence(root, tree=REPO, W=draft["W"], row=row, serve_dir=serve, driver_python=sys.executable,
                               env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}, run=run)
    rec_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert rec_argv[rec_argv.index("--probe-expect") + 1] == "kinds" and rec_argv[rec_argv.index("--expect-tasks") + 1] == str(root / lp.EXPECTED_SAMPLERS_REL)
    tag_args = [rec_argv[i + 1] for i, a in enumerate(rec_argv) if a == "--probe-tags"]
    assert str(tags_p) in tag_args and len(tag_args) == len(row["children"]) and all(a.startswith(str(root / "exp/sampled/gsm8k_thinking_on_cap8192__s1/")) for a in tag_args)
    expects = sorted(rec_argv[i + 1] for i, a in enumerate(rec_argv) if a == "--expect")
    assert expects == sorted(f"{k}={v}" for k, v in CARD.items())
    assert ev[lp.RESOLVED_SAMPLER_REL]["rc"] == 0 and "sha256" in ev[lp.RESOLVED_SAMPLER_REL], (root / "children" / "resolved_sampler.log").read_text()[-1200:]
    rec = json.loads((root / lp.RESOLVED_SAMPLER_REL).read_text())
    assert rec["verdict"] == "AGREES", rec["why"]
    assert rec["client_task_verdicts"] == {GSM8K_TASK: "AGREES"} and rec["n_client_requests"] == 3
    assert rec["probe_verdict_by_kind"]["render"] == "AGREES" and rec["probe_verdict_by_kind"]["bites"] == "AGREES" and rec["n_probe_requests_by_kind"] == {
        "render": 1, "bites": 10, "lens_on_bite": 0, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 0}
    assert rec["n_unclassified_requests"] == 0 and rec["n_probe_unclassified_requests"] == 0 and rec["classes"]["probe"]["kinds"]["render"]["reference"] == CARD
    assert rec["probe_tags"]["missing"] and str(tags_p) not in rec["probe_tags"]["missing"], "the children that never ran are RECORDED missing, not invented"
    # the certificate's step accepts it (every task AGREES); the SAME log under fc3's argv (no kinds) is not AGREES -- the reason fc4 exists
    facts = lp._smoke_check("resolved_sampler", json.dumps(rec), path=f"/vol/x/{lp.RESOLVED_SAMPLER_REL}", plan={"W": draft["W"]}, accepted_attempts=["a"], row=row, aid="a",
                            root="/vol/x", producer={lp.RESOLVED_SAMPLER_REL: {"sha256": lp._sha_text(json.dumps(rec)), "writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": draft["W"], "rc": 0}})
    assert facts["verdict"] == "AGREES" and facts["client_task_verdicts"] == {GSM8K_TASK: "AGREES"}
    before_out = tmp_path / "before.json"
    rc = rrs.main(["--vllm-log", str(serve / "vllm_server.log"), "--out", str(before_out), "--require-wired", "--expect-tasks", str(root / lp.EXPECTED_SAMPLERS_REL)])
    assert rc == 0 and json.loads(before_out.read_text())["verdict"] != "AGREES"
    # a controls row through the same app path composes NO kinds flags (byte-identical to fc3)
    calls.clear()
    croot = tmp_path / "croot"
    (croot / "exp" / "serve").mkdir(parents=True)
    (croot / "exp" / "serve" / "vllm_server.log").write_text("\n".join([_ATTEST] + [_line(i, 48, **GREEDY, seed=0) for i in range(3)]) + "\n")
    srv.producer_evidence(croot, tree=REPO, W=draft["W"], row=_row(draft, "controls"), serve_dir=croot / "exp" / "serve", driver_python=sys.executable,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}, run=run)
    c_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert "--probe-expect" not in c_argv and "--probe-tags" not in c_argv and "--expect" not in c_argv and "--expect-tasks" in c_argv
