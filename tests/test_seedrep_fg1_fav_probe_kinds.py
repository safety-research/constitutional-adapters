"""qwen-seedrep-native-battery fg1 / F-AV: the request-evidence probe classifier on a gsm8k standup.

THE FINDING, re-validated as the first assertions: the gsm8k `S3` smoke's `resolved_sampler.json` read UNCLASSIFIED_REQUESTS -- "649 x bites: tagged
{'48': 40} vs logged {'48': 649}; 5 x render: tagged {'512': 10} vs logged {'512': 5}". Two mechanisms, neither a defect of
the drivers' tags:

* 609 of the 649 greedy requests at cap 48 are the NATIVE STANDUP's own positive controls (`native_standup.run_positive_controls` via
  `LoraStandup.greedy_ids`: noop_certificate, steering_bites, selection_honored -- temperature 0, max_tokens = its `--probe-max-tokens`, 48),
  fired before READY and stamped with NO `X-Tinycap-Task`; the kind table had no kind for them, and under `--probe-expect kinds` the ONE
  tag claim at cap 48 (the drivers' 40 `bites`) had to account for every greedy request there. A controls FULL root shows the same
  arithmetic in the recorder's default mode: 659 greedy at 48 = 609 standup + 50 tagged child bites.
* the deliberate OFF-arm render probe on a thinking-ON standup is refused by the proxy (400 thinking_regime_mismatch via the ledger's pre-tokenize)
  and never reaches the engine -- but `request_evidence.jsonl` is written at the client's HTTP boundary BEFORE the send, so all 10 render attempts
  (5 stores x 2 arms) were tagged while only the 5 ON arms were logged; the sibling `usage_ledger.jsonl` rows the OFF attempts `rejected: true`.

THE FIX: (1) a `standup_controls` kind (contracts + the recorder's table; greedy-referenced; cap PER STANDUP), DECLARED to the recorder's kinds mode
from the root's OWN records -- `endpoint.json.probe_max_tokens` == `positive_controls.json.probe_set.max_tokens`, a MAXIMUM count summed from the
record's stated control counts (`seedrep_served.probe_declaration_from_root` / `standup_controls_request_bound`; e.g. 22 + 50 + 539 + 77 = 688) --
by both recorder argv builders (`producer_evidence` at run time, `rederive_sampler_verdict` on the prior-root re-audit); the greedy pool at 48 is
split by COUNT: the tag's exact 40 to `bites`, at most the bound to `standup_controls`, anything beyond UNCLASSIFIED by name; (2) the recorder's
tag reader joins a `request_evidence.jsonl` to its sibling `usage_ledger.jsonl` by request id and skips (counts) every tagged attempt the ledger
rejected, exactly as it already skipped a ledger-spelled rejected row. No catch-all: a sampled request at 48, a greedy request above the bound,
an untagged cap, a tag without its ledger row all still refuse by name.

Synthetic throughout: invented request ids, placeholder vmodels, no recorded data."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import record_resolved_sampler as rrs  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from test_record_resolved_sampler import _CELL, _GREEDY, _GREEDY_REF, _TWIN, _probe_lines, _run, _sp, _tag_row  # noqa: E402
from test_seedrep_fc4_gsm8k_render_probe import CARD, GSM8K_TASK, _ATTEST, _line, _load_served, tag_row  # noqa: E402
from test_seedrep_review_fixes import _row, _stub_modal, draft, inputs  # noqa: E402,F401  (module fixtures)

QWEN_CARD = dict(t=0.6, tp=0.95, tk=20, pp=0.0)                  # the thinking card the gsm8k row decodes under (qwen35_4b_it|gsm8k|on|thinking_card@v1)
KINDS = ["--client-cap", "8192", "--probe-expect", "kinds", *[x for k, v in sorted(CARD.items()) for x in ("--expect", f"{k}={v}")]]
N_STANDUP, N_BITES, N_RENDER_TAGGED, N_RENDER_LOGGED, N_LENS = 609, 40, 10, 5, 8      # the S3 smoke's counts (five children)
BOUND = 688                                                                           # the record's declared bound (22 + 50 + 539 + 77)


def gsm8k_log(n_standup=N_STANDUP, n_bites=N_BITES, n_render=N_RENDER_LOGGED, n_lens=N_LENS, n_tasks=6, extra=()):
    """The S3 smoke's engine log, in shape: the standup's untagged greedy controls at 48, the drivers' greedy bites at 48, the ON-arm render probes
    at 512 under the card, the lens-on-bite probes at 256, the task draws at 8192 under the card."""
    lines = [_ATTEST]
    lines += [_line(1000 + i, 48, **_GREEDY, seed=0) for i in range(n_standup)]
    lines += [_line(3000 + i, 48, **_GREEDY, seed=0) for i in range(n_bites)]
    lines += [_line(4000 + i, 512, **QWEN_CARD, seed=i) for i in range(n_render)]
    lines += [_line(5000 + i, 256, **_GREEDY, seed=0) for i in range(n_lens)]
    lines += [_line(6000 + i, 8192, **QWEN_CARD, seed=5000 + i) for i in range(n_tasks)]
    return lines + list(extra)


def gsm8k_tags(n_bites=N_BITES, n_render=N_RENDER_TAGGED, n_lens=N_LENS):
    """The five children's request_evidence.jsonl tag rows, folded into one file: bites on the cell and its twin, TWO render arms per store, lens pairs."""
    rows = [_tag_row("bites", 48, vmodel=(_CELL if i % 2 == 0 else _TWIN), n=i) for i in range(n_bites)]
    rows += [{**_tag_row("render", 512, n=i), "request_id": f"render{i}".ljust(32, "0")} for i in range(n_render)]
    rows += [_tag_row("lens_on_bite", 256, n=i) for i in range(n_lens)]
    return rows


def _ledger_rows(rejected_ids, *, task="probe/render", cap=512):
    """The sibling usage_ledger.jsonl's rows for the OFF-arm render attempts the proxy refused (never reached the engine)."""
    return [{"ts": "2026-09-20T00:00:00Z", "task": task, "sample": f"petri__base/{i}", "vmodel": "petri__base", "thinking": False, "prompt_tokens_pre": 12,
             "max_tokens_requested": cap, "has_tools": False, "n_tools": 0, "status": 400, "error_code": "thinking_regime_mismatch", "rejected": True,
             "usage_prompt_tokens": None, "usage_completion_tokens": None, "finish_reason": None, "latency_s": 0.01, "tokenize_mismatch": None,
             "error_class": "route_refused", "request_id": rid, "body_sha256": "0" * 64} for i, rid in enumerate(rejected_ids)]


def _run_gsm8k(tmp_path, lines, tags, *, ledger_rejected=None, declare=(), tags_name="request_evidence.jsonl"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / tags_name).write_text("".join(json.dumps(r) + "\n" for r in tags))
    if ledger_rejected is not None:
        (tmp_path / "usage_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _ledger_rows(ledger_rejected)))
    extra = [*KINDS, "--probe-tags", str(tmp_path / tags_name)]
    for spec in declare:
        extra += ["--declare-kind", spec]
    return _run(tmp_path, lines, extra=extra)


OFF_RENDER_IDS = [f"render{i}".ljust(32, "0") for i in range(N_RENDER_TAGGED) if i % 2 == 1]      # every store's second (OFF) render attempt


# ------------------------------------------------------------------------------------------------ the recorder: the S3 smoke's exact shape
def test_the_s3_shape_read_unclassified_before_and_agrees_by_kind_with_the_declaration_and_the_ledger_join(tmp_path):
    # RE-VALIDATION: without the standup's declaration and without the ledger join, both readings of the launch report
    rc, rec = _run_gsm8k(tmp_path / "before", gsm8k_log(), gsm8k_tags())
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS", rec["why"]
    k = rec["classes"]["probe"]["kinds"]
    assert k["bites"]["tag_check"]["problems"] == ["tagged {'48': 40} vs logged {'48': 649} — tag and shape disagree in count"]
    assert k["render"]["tag_check"]["problems"] == ["tagged {'512': 10} vs logged {'512': 5} — tag and shape disagree in count"]
    assert rec["n_probe_unclassified_requests"] == 654 and k["lens_on_bite"]["verdict"] == "AGREES"
    # THE FIX: the standup's kind declared from its record (a maximum), the OFF-arm render attempts skipped by the ledger join
    rc, rec = _run_gsm8k(tmp_path / "after", gsm8k_log(), gsm8k_tags(), ledger_rejected=OFF_RENDER_IDS, declare=[f"standup_controls:cap=48:count={BOUND}"])
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    assert rec["n_unclassified_requests"] == 0 and rec["n_probe_unclassified_requests"] == 0 and rec["n_client_requests"] == 6
    assert rec.get("client_task_verdicts") is None or all(v == "AGREES" for v in rec["client_task_verdicts"].values())
    assert rec["n_probe_requests_by_kind"] == {"render": 5, "bites": 40, "lens_on_bite": 8, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 609}
    assert rec["probe_verdict_by_kind"] == {"render": "AGREES", "bites": "AGREES", "lens_on_bite": "AGREES", "logprob_bites": "EMPTY", "load_check": "EMPTY",
                                            "positive_control": "EMPTY", "standup_controls": "AGREES"}
    k = rec["classes"]["probe"]["kinds"]
    b, s = k["bites"], k["standup_controls"]
    assert b["n"] == 40 and b["n_tagged"] == 40 and b["identity"]["source"] == "tag" and b["reference"] == _GREEDY_REF
    assert s["n"] == 609 and s["n_tagged"] == 0 and s["n_declared"] == 1 and s["identity"]["source"] == "declaration" and s["basis"] == "declared_count"
    assert s["reference"] == _GREEDY_REF and s["reference_kind"] == "greedy" and "native_standup.run_positive_controls" in s["reference_source"]
    assert s["join_ambiguous"] is True and s["competing_kinds"] == ["bites"] and b["join_ambiguous"] is True and b["competing_kinds"] == ["standup_controls"], (
        "the multiset COUNTS at a shared cap, it does not bind -- said on both kinds")
    assert s["caps_pinned"] == "per-standup" and any("by COUNT" in x for x in s["identity"]["selection"])
    r = k["render"]
    assert r["n"] == 5 and r["n_tagged"] == 5 and r["verdict"] == "AGREES" and r["reference"] == CARD and r["tag_check"]["agrees"]
    t = rec["probe_tags"]
    assert t["n_probe_rows"] == 58 and t["n_rejected_skipped"] == 5 and t["by_kind"]["render"] == [{"max_tokens": "512", "n": 5}]
    lj = t["ledger_join"]
    assert lj["n_rejected_by_ledger"] == 5 and lj["rejected"] == {"probe/render@512": 5} and lj["joined"] == [str(tmp_path / "after" / "usage_ledger.jsonl")] and lj["absent"] == []
    assert "HTTP boundary BEFORE" in lj["rule"] and "thinking_regime_mismatch" in lj["rule"]
    assert rec["declarations"] and rec["declarations"][0]["kind"] == "standup_controls" and rec["declarations"][0]["count"] == BOUND and rec["declarations"][0]["exact"] is False


def test_the_declaration_is_a_maximum_and_nothing_is_folded(tmp_path):
    # more greedy requests at 48 than the tag + the bound account for: the remainder is UNCLASSIFIED by name, the two kinds still take their shares
    rc, rec = _run_gsm8k(tmp_path / "over", gsm8k_log(n_standup=BOUND + 12), gsm8k_tags(), ledger_rejected=OFF_RENDER_IDS, declare=[f"standup_controls:cap=48:count={BOUND}"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["n_probe_unclassified_requests"] == 12, rec["why"]
    assert rec["n_probe_requests_by_kind"]["standup_controls"] == BOUND and rec["n_probe_requests_by_kind"]["bites"] == 40
    reasons = [x["reason"] for x in rec["classes"]["probe"]["unclassified"]["reasons"]]
    assert any("unassigned at max_tokens 48" in r and "greedy-shaped" in r for r in reasons), reasons
    # a SAMPLED request at 48 is no standup control (the declaration names a greedy kind): unclassified by its shape, never folded
    sampled = [_line(9000, 48, **QWEN_CARD, seed=3)]
    rc, rec = _run_gsm8k(tmp_path / "shape", gsm8k_log(extra=sampled), gsm8k_tags(), ledger_rejected=OFF_RENDER_IDS, declare=[f"standup_controls:cap=48:count={BOUND}"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["n_probe_unclassified_requests"] == 1
    assert any("unassigned at max_tokens 48" in x["reason"] and "sampled-shaped" in x["reason"] for x in rec["classes"]["probe"]["unclassified"]["reasons"])
    # an exact count that is not the log's refuses by name too
    rc, rec = _run_gsm8k(tmp_path / "exact", gsm8k_log(), gsm8k_tags(), ledger_rejected=OFF_RENDER_IDS, declare=["standup_controls:cap=48:count=600!"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["n_probe_unclassified_requests"] > 0


def test_the_ledger_join_is_by_request_id_and_its_absence_leaves_the_tag_standing(tmp_path):
    # no sibling ledger: the ten render tags stand and refuse against the five logged, by name (today's reading), and the record says the sibling is absent
    rc, rec = _run_gsm8k(tmp_path / "noledger", gsm8k_log(), gsm8k_tags(), declare=[f"standup_controls:cap=48:count={BOUND}"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["classes"]["probe"]["kinds"]["render"]["verdict"] == "UNCLASSIFIED"
    assert rec["probe_tags"]["ledger_join"]["absent"] == [str(tmp_path / "noledger" / "usage_ledger.jsonl")] and rec["probe_tags"]["n_rejected_skipped"] == 0
    assert rec["n_probe_requests_by_kind"]["standup_controls"] == 609, "the standup kind is unaffected by the render tags' problem"
    # a ledger whose rejected rows name OTHER request ids joins nothing: the tags stand, the record counts nothing skipped
    rc, rec = _run_gsm8k(tmp_path / "otherids", gsm8k_log(), gsm8k_tags(), ledger_rejected=[f"nobody{i}".ljust(32, "0") for i in range(5)],
                         declare=[f"standup_controls:cap=48:count={BOUND}"])
    assert rec["verdict"] == "UNCLASSIFIED_REQUESTS" and rec["probe_tags"]["n_rejected_skipped"] == 0 and rec["probe_tags"]["ledger_join"]["n_rejected_by_ledger"] == 0
    # the ledger-spelled tag source (a usage_ledger.jsonl passed as --probe-tags) keeps its own `rejected` reading and joins nothing
    ledger_tags = [{"task": r["task"], "sample": r["sample"], "vmodel": r["vmodel"], "status": 200, "rejected": r["request_id"] in OFF_RENDER_IDS,
                    "max_tokens_requested": r["max_tokens"], "request_id": r["request_id"], "body_sha256": r["body_sha256"]} for r in gsm8k_tags()]
    rc, rec = _run_gsm8k(tmp_path / "ledgertags", gsm8k_log(), ledger_tags, declare=[f"standup_controls:cap=48:count={BOUND}"], tags_name="usage_ledger.jsonl")
    assert rc == 0 and rec["verdict"] == "AGREES", rec["why"]
    assert rec["probe_tags"]["n_rejected_skipped"] == 5 and rec["probe_tags"]["ledger_join"]["joined"] == [] and rec["probe_tags"]["ledger_join"]["n_rejected_by_ledger"] == 0


def test_the_kind_is_pinned_to_the_standup_code_and_declared_only_never_tagged_by_the_driver():
    assert tc.PROBE_KIND_STANDUP_CONTROLS == rrs.PROBE_KIND_STANDUP_CONTROLS == "standup_controls" and "standup_controls" in tc.PROBE_KINDS_STANDUP
    assert rrs.PROBE_KIND_TABLE["standup_controls"]["caps"] == rrs.CAP_PER_STANDUP and rrs._pinned_caps("standup_controls") is None
    assert rrs.parse_declaration("standup_controls:cap=48:count=688") == {"kind": "standup_controls", "cap": "48", "count": 688, "exact": False, "raw": "standup_controls:cap=48:count=688"}
    assert rrs.kind_reference("standup_controls", CARD) == rrs.probe_reference()
    src = (REPO / "src/sisteer/capeval/serving/native_standup.py").read_text()
    assert '"max_tokens": int(self.probe_max_tokens), "temperature": 0,' in src and "X-Tinycap-Task" not in src.split("def greedy_ids_with_request_id")[1].split("def ")[0], (
        "the standup's greedy_ids is untagged: the kind is DECLARED from its record, never read off a tag")


# ------------------------------------------------------------------------------------------------ the app: the declaration from the root's records
def w24_shaped_records(serve: Path, *, probe_max_tokens=48, probe_set_cap=48, n_cells=5, n_probes=5, n_prompts=11, n_requests=539, controls=True):
    """`exp/serve/endpoint.json` + `positive_controls.json` in a controls FULL root's SHAPE (its counts; synthetic names)."""
    serve.mkdir(parents=True, exist_ok=True)
    (serve / "endpoint.json").write_text(json.dumps({"vmodels": {"none": "petri__base"}, "probe_max_tokens": probe_max_tokens, "thinking_mode": "on"}))
    if not controls:
        return
    cells = {f"si_synthetic_{i}__arm{i}": {"statistic": "text", "twin": "none__lora0", "twin_vmodel": "petri__none__lora0", "arm": f"arm{i}", "n_probes": n_probes,
                                           "n_differing": n_probes, "first_divergences": []} for i in range(n_cells)}
    pc = {"engine": "lora_native", "standup_attempt_id": "11111111-2222-3333-4444-555555555555", "started_at": "2026-09-20T00:00:00Z",
          "probe_set": {"n": n_prompts, "hash": "0" * 16, "max_tokens": probe_set_cap, "sampling": "greedy"},
          "base_vmodel": "petri__base", "zero_vmodel": "petri__none__lora0", "genuine_vmodels": [f"petri__si_synthetic_{i}" for i in range(n_cells)],
          "zero_adapter_only": False, "output_diagnostics": False, "output_diagnostics_inherited": True, "output_diagnostics_policy": False,
          "order": ["noop_certificate", "salt_isolation", "steering_bites", "selection_honored", "adapter_digests", "boundary_refusal"],
          "controls": {"noop_certificate": {"passed": True, "n_prompts": n_prompts, "all_exact": True, "n_tokens": [48] * n_prompts, "rule": "bit-exact", "wall_s": 7.9},
                       "salt_isolation": {"passed": True, "n_routes": 2 * n_cells - 1, "per_cell": {}, "stable_collisions": []},
                       "steering_bites": {"passed": True, "cells": cells, "wall_s": 20.7},
                       "selection_honored": {"passed": True, "n_requests": n_requests, "verdict": "routing_attested", "wall_s": 9.0},
                       "adapter_digests": {"passed": True, "adapters": {}, "id_name_map": {}},
                       "boundary_refusal": {"passed": None, "reason": "no restricted-mode adapter is served", "restricted_cells": [], "verdict": "not_applicable"}},
          "passed": True, "failed_control": None, "error": None, "finished_at": "2026-09-20T00:01:00Z"}
    (serve / "positive_controls.json").write_text(json.dumps(pc))


def test_probe_declaration_from_root_declares_the_standup_controls_kind_with_a_record_derived_maximum(tmp_path):
    srv = _load_served()
    w24_shaped_records(tmp_path / "exp" / "serve")
    decl = srv.probe_declaration_from_root(tmp_path)
    assert decl["probe_max_tokens"] == 48
    d = decl["declared"]["standup_controls"]
    assert (d["cap"], d["expected_count"], d["count_rule"], d["sampling"], d["conditional"]) == (48, BOUND, "max", "greedy", False)
    assert d["basis"] == {"probe_set_n": 11, "noop_certificate": 22, "steering_bites": 50, "selection_honored_mixed": 539, "selection_honored_solo_max": 77, "boundary_refusal": 0,
                          "rule": d["basis"]["rule"]} and "upper bound" in d["basis"]["rule"]
    assert "positive_controls.json" in d["source"] and "probe_max_tokens" in d["cap_rule"]
    assert BOUND >= N_STANDUP, "the bound covers the S3 smoke's (and the W24 root's) 609 logged standup requests"
    # fc11g's salt-isolation declaration is unchanged beside it, and only the standup kind is ever declared to the recorder
    assert set(decl["declared"]) == {"standup_controls", "salt_isolation"} and decl["declared"]["salt_isolation"] == {
        **decl["declared"]["salt_isolation"], "cap": 1, "expected_count": 9 * 11, "conditional": True}
    assert srv.declare_kind_argv(decl, client_caps={8192}) == ["--declare-kind", f"standup_controls:cap=48:count={BOUND}"]
    assert srv.declare_kind_argv(decl, client_caps={48, 8192}) == [], "never at a client cap: the recorder would fold client requests into the probe class"
    assert srv.DECLARABLE_KINDS == ("standup_controls",) and srv.STANDUP_CONTROLS_KIND == tc.PROBE_KIND_STANDUP_CONTROLS
    # the record must STATE the cap and the greedy sampling: a cap disagreement or a missing record declares nothing (the requests then refuse by name)
    w24_shaped_records(tmp_path / "mismatch" / "exp" / "serve", probe_set_cap=64)
    assert "standup_controls" not in srv.probe_declaration_from_root(tmp_path / "mismatch")["declared"]
    w24_shaped_records(tmp_path / "nopc" / "exp" / "serve", controls=False)
    assert srv.probe_declaration_from_root(tmp_path / "nopc")["declared"] == {}
    bound, why = srv.standup_controls_request_bound({"probe_set": {"n": 11}, "controls": {}})
    assert bound is None and "no positive-control request counts" in why


def test_producer_evidence_declares_the_standup_kind_to_the_real_recorder_and_the_s3_shape_agrees(draft, tmp_path):
    """The integration test, with the smoke's REAL probe traffic added: the standup's 609 untagged greedy controls at 48, the OFF-arm render
    attempt the ledger rejected. The app derives the declaration from the root's records, hands it to the REAL recorder, and every kind AGREES."""
    srv = _load_served()
    row = _row(draft, "gsm8k")
    root = tmp_path / "root"
    serve = root / "exp" / "serve"
    w24_shaped_records(serve)
    (serve / "vllm_server.log").write_text("\n".join(gsm8k_log(n_bites=10, n_render=1, n_lens=0, n_tasks=3)) + "\n")
    none_child = [k for k in row["children"] if k["cell"] == "none"][0]
    store = root / lp.GSM8K_OUT_REL.format(cap=none_child["cap"], s=none_child["s"], cell="none")
    store.mkdir(parents=True)
    render_ids = ["render-on".ljust(32, "0"), "render-off".ljust(32, "0")]
    tags = [tag_row("bites", 48, n=i) for i in range(10)] + [{**tag_row("render", 512, n=i), "request_id": rid} for i, rid in enumerate(render_ids)] + \
        [{**tag_row("x", 8192, n=i), "kind": "task", "task": GSM8K_TASK, "sample": f"g{i}/0"} for i in range(3)]
    (store / "request_evidence.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tags))
    (store / "usage_ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _ledger_rows([render_ids[1]])))
    calls = []

    def run(argv, **kw):
        calls.append([str(a) for a in argv])
        if any(str(a).endswith("request_evidence_audit.py") for a in argv):
            return subprocess.CompletedProcess(argv, 0)
        return subprocess.run(argv, **kw)
    ev = srv.producer_evidence(root, tree=REPO, W=draft["W"], row=row, serve_dir=serve, driver_python=sys.executable,
                               env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}, run=run)
    rec_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert rec_argv[rec_argv.index("--probe-expect") + 1] == "kinds"
    assert [rec_argv[i + 1] for i, a in enumerate(rec_argv) if a == "--declare-kind"] == [f"standup_controls:cap=48:count={BOUND}"]
    assert "--probe-cap" not in rec_argv
    assert ev[lp.RESOLVED_SAMPLER_REL]["rc"] == 0, (root / "children" / "resolved_sampler.log").read_text()[-1500:]
    rec = json.loads((root / lp.RESOLVED_SAMPLER_REL).read_text())
    assert rec["verdict"] == "AGREES", rec["why"]
    assert rec["n_probe_requests_by_kind"] == {"render": 1, "bites": 10, "lens_on_bite": 0, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 609}
    assert rec["probe_tags"]["n_rejected_skipped"] == 1 and rec["probe_tags"]["ledger_join"]["n_rejected_by_ledger"] == 1 and rec["n_unclassified_requests"] == 0
    # the certificate's step accepts the record
    facts = lp._smoke_check("resolved_sampler", json.dumps(rec), path=f"/vol/x/{lp.RESOLVED_SAMPLER_REL}", plan={"W": draft["W"]}, accepted_attempts=["a"], row=row, aid="a",
                            root="/vol/x", producer={lp.RESOLVED_SAMPLER_REL: {"sha256": lp._sha_text(json.dumps(rec)), "writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": draft["W"], "rc": 0}})
    assert facts["verdict"] == "AGREES"
    # RE-VALIDATION: the SAME log and tags under fc4's argv (no declaration; the ledger not joined because the recorder's reader now joins it -- so drop the ledger) refuse
    (store / "usage_ledger.jsonl").unlink()
    before = tmp_path / "before.json"
    rc = rrs.main(["--vllm-log", str(serve / "vllm_server.log"), "--out", str(before), "--require-wired", "--expect-tasks", str(root / lp.EXPECTED_SAMPLERS_REL),
                   "--probe-expect", "kinds", "--probe-tags", str(store / "request_evidence.jsonl"), *[x for k, v in sorted(CARD.items()) for x in ("--expect", f"{k}={v}")]])
    b = json.loads(before.read_text())
    assert b["verdict"] == "UNCLASSIFIED_REQUESTS" and b["classes"]["probe"]["kinds"]["bites"]["tag_check"]["problems"] == ["tagged {'48': 10} vs logged {'48': 619} — tag and shape disagree in count"]
    assert b["classes"]["probe"]["kinds"]["render"]["tag_check"]["problems"] == ["tagged {'512': 2} vs logged {'512': 1} — tag and shape disagree in count"]
    # a controls row composes NO kinds flags and NO declaration (byte-identical to fc3 / fc4)
    calls.clear()
    croot = tmp_path / "croot"
    (croot / "exp" / "serve").mkdir(parents=True)
    (croot / "exp" / "serve" / "vllm_server.log").write_text("\n".join([_ATTEST] + [_line(i, 48, **_GREEDY, seed=0) for i in range(3)]) + "\n")
    srv.producer_evidence(croot, tree=REPO, W=draft["W"], row=_row(draft, "controls"), serve_dir=croot / "exp" / "serve", driver_python=sys.executable,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"}, run=run)
    c_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert "--declare-kind" not in c_argv and "--probe-expect" not in c_argv


def test_rederive_sampler_verdict_declares_under_kinds_mode_and_keeps_probe_cap_for_the_default_mode(draft, tmp_path):
    srv = _load_served()
    row = _row(draft, "gsm8k")
    pulled = tmp_path / "pulled"
    w24_shaped_records(pulled / "exp" / "serve")
    (pulled / "exp" / "serve" / "vllm_server.log").write_text("\n".join(gsm8k_log(n_bites=10, n_render=1, n_lens=0, n_tasks=3)) + "\n")
    (pulled / lp.RESOLVED_SAMPLER_REL).write_text(json.dumps({"verdict": "UNCOMPARED"}))
    calls = []

    def run_tool(argv, *, cwd, log, probe_block=True):
        calls.append(list(argv))
        if "expected-samplers" in argv:
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            doc = {"tasks": {GSM8K_TASK: {"caps": [8192], "sampler": CARD}}}
            if probe_block:
                doc["probe"] = {"probe_expect": "kinds", "expect": CARD, "tags_rel": ["exp/sampled/gsm8k_thinking_on_cap8192__s1/none/request_evidence.jsonl"]}
            out.write_text(json.dumps(doc))
        return 0
    res = srv.rederive_sampler_verdict(pulled=pulled, evidence=tmp_path / "ev", work=tmp_path / "work", tree=REPO, this_sha="5" * 40, row=row, python=sys.executable, run_tool=run_tool)
    rec_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert "--probe-cap" not in rec_argv and rec_argv[rec_argv.index("--probe-expect") + 1] == "kinds"
    assert [rec_argv[i + 1] for i, a in enumerate(rec_argv) if a == "--declare-kind"] == [f"standup_controls:cap=48:count={BOUND}"]
    pr = res["record"]["probe"]
    assert pr["declared_argv"] == ["--declare-kind", f"standup_controls:cap=48:count={BOUND}"] and pr["declared_argv_kinds"] == ["standup_controls"] and pr["probe_caps_argv"] == [1]
    assert pr["declared"]["standup_controls"]["expected_count"] == BOUND and pr["problems"] == []
    # a DEFAULT-mode row (no probe block: a static / controls row) keeps fc11g's `--probe-cap` and declares nothing
    calls.clear()
    import functools
    res2 = srv.rederive_sampler_verdict(pulled=pulled, evidence=tmp_path / "ev2", work=tmp_path / "work2", tree=REPO, this_sha="5" * 40, row=row, python=sys.executable,
                                        run_tool=functools.partial(run_tool, probe_block=False))
    rec_argv = [a for a in calls if a[1].endswith("record_resolved_sampler.py")][0]
    assert "--declare-kind" not in rec_argv and [rec_argv[i + 1] for i, a in enumerate(rec_argv) if a == "--probe-cap"] == ["1", "48"]
    assert res2["record"]["probe"]["declared_argv"] == [] and res2["record"]["probe"]["declared_argv_kinds"] == []


def test_the_certificate_holds_a_maximum_per_kind_off_the_fresh_record(monkeypatch):
    """`smoke_pass_reaudit_problems` (fc11g's consumer): a `count_rule: max` declaration the argv carried is checked per KIND against the recorder's fresh
    per-kind count (at most the bound); one the argv never carried (a default-mode row) is no claim; fc11g's salt_isolation per-cap check is unchanged."""
    monkeypatch.setattr(lp, "_carry_over_module", lambda: type("CO", (), {"reaudit_problems": staticmethod(lambda ra, this_sha: [])})())
    W = "5" * 40

    def ra(**probe):
        base_probe = {"declared": {"standup_controls": {"cap": 48, "expected_count": BOUND, "count_rule": "max", "source": "positive_controls.json"}},
                      "probe_caps_argv": [], "probe_max_tokens": 48, "client_caps": [8192], "n_probe_by_cap": {"48": 649, "512": 5, "256": 8},
                      "n_unclassified": 0, "declared_argv_kinds": ["standup_controls"], "n_by_kind": {"bites": 40, "standup_controls": 609}}
        return {"reaudited_by": W, "evidence_root": "/vol/x/ev", "store_root": "/vol/x/root", "rc": 0, "argv": [], "stores_present": [], "stores_problems": {},
                "producer_evidence": {lp.RESOLVED_SAMPLER_REL: {"sha256": "0" * 64, "writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": W, "rc": 0}},
                "sampler": {"rederived_by": W, "rederived_verdict": "AGREES", "historical_verdict": "UNCOMPARED", "probe": {**base_probe, **probe}}}
    assert lp.smoke_pass_reaudit_problems(ra(), this_W=W, attempt_root="/vol/x/root") == []
    probs = lp.smoke_pass_reaudit_problems(ra(n_by_kind={"bites": 40, "standup_controls": BOUND + 1}), this_W=W, attempt_root="/vol/x/root")
    assert probs == [f"reaudit: probe kind 'standup_controls' (cap 48, positive_controls.json) allows at most {BOUND} request(s); the recorder classified {BOUND + 1} of that kind"]
    assert lp.smoke_pass_reaudit_problems(ra(n_by_kind=None), this_W=W, attempt_root="/vol/x/root")[0].endswith("the recorder classified None of that kind")
    assert lp.smoke_pass_reaudit_problems(ra(declared_argv_kinds=[], n_by_kind=None), this_W=W, attempt_root="/vol/x/root") == [], "a declaration the argv never carried is no claim"
    salt = {"declared": {"salt_isolation": {"cap": 1, "expected_count": 33, "source": "positive_controls.json", "conditional": True}}, "n_probe_by_cap": {"1": 32, "48": 179}}
    assert lp.smoke_pass_reaudit_problems(ra(**salt), this_W=W, attempt_root="/vol/x/root") == [
        "reaudit: probe kind 'salt_isolation' (cap 1, positive_controls.json) expects 33 request(s); the engine log holds 32 at that cap"]
