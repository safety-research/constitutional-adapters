"""fj2: completed attempts are CARRIED across W trees; nothing that completed once is re-run.

Known-bad at 174410edc: the accepted map is keyed `<wave>|wave_uuid|...` and nothing admits a prior wave's attempt --
`select_launch_rows` had no notion of a carried row, the certificate had no `carry_over` component, and a later wave forced by one
defect re-ran all 405 cells. `scripts/modal/seedrep_carry_over.py` is the admission; every clause here runs on SYNTHETIC two-tree
plans, synthetic attempt roots (a dict standing for the Volume API), a synthetic git (blob maps per sha) and a stub readiness
contract -- no Modal, no real RS, no credentials.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402

W_PREV, W_THIS = "a" * 40, "b" * 40
WAVE = "22222222-3333-4444-8444-555555555555"                    # the production wave (RS/wave.json is minted once, shared across W trees)
S5_WAVE = "11111111-2222-4333-8444-555555555555"
AID = "01TESTTASK-deadbeef0001"
SID = "pasttense__s1__base__shard00"
TREE = co.CONTAINER_TREE_PREFIX
STANDUP = "scripts/sdfbase/mt_serving_up.py"
CHILD = "scripts/jailbreak_run.py"
CFG = "configs/sdfbase_full22_qwen35_4b_s1.yaml"
MANIFEST = "configs/capeval/manifests/petri_full22_qwen35_4b_s1.yaml"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- synthetic plans (two trees, same wave)

def _row(W: str, W7: str, *, sid: str = SID, leg: str = "pasttense", stage: str = "base", ladder=None, cap: int = 4096, thinking: bool = False,
         kind: str = "base", timeout_s: int = 57600) -> dict:
    engine, shard = "s1", 0
    return {"standup_id": sid, "leg": leg, "engine": engine, "view": engine, "stage": stage, "shard": shard, "kind": kind, "pool": None, "rung_mode": None,
            "control_key": lp.control_key(W7, WAVE, leg, engine, stage, shard), "accepted_key": lp.accepted_key(W7, WAVE, leg, engine, stage, shard),
            "argv": ["/venv/bin/python", f"{TREE}{W}/{STANDUP}", "--repo-dir", f"{TREE}{W}", "--config", CFG, "--manifest", MANIFEST, "--eval", "families_served"],
            "children": [{"eval_leg": leg, "arm": "none", "cell": "none", "s": 1, "task": f"static/{leg}",
                          "argv": ["{driver_python}", f"{TREE}{W}/{CHILD}", "generate", "--config", CFG, "--attack", leg, "--defense", "none", "--cap", str(cap)]}],
            "geometry": {"eval": "families_served", "L": 11520, "S": 45, "thinking": thinking, "cap": cap, "expected_window_source": "override"},
            "readiness": {"engine": "lora_native", "attested_max_model_len": 11520, "attested_max_num_seqs": 45, "thinking_mode": "on" if thinking else "off"},
            "objects": {}, "adapters": {"none__lora0": {"expected_reused": True, "object_sha256": None, "staged_id": "none__lora0", "intervention_hash": "h0"}},
            "route_set": ["none"], "route_set_harness": ["none"], "cells_served_expected": {"canonical": ["none"], "controls": ["none__lora0"]},
            "modal": {"function": "run_view_leg", "gpu": "H200", "timeout_s": timeout_s}, "ladder": ladder, "rung_consumer": None, "rung_assembly": None,
            "producers": [], "nshards": 1, "eval_legs": [leg]}


def _plan(W: str, *, materialisation: str, rows=None, rungs=()) -> dict:
    W7 = W[:7]
    rows = rows if rows is not None else [_row(W, W7)]
    plan = {"schema": "seedrep_launch_plan/1", "materialisation": materialisation, "W": W, "W7": W7, "wave_uuid": WAVE,
            "results_root": f"/vol/seedrep/{W7}/exp/seedrep_battery_modal", "dicts": {p: lp.dict_name(W7, WAVE, p) for p in lp.DICT_PURPOSES},
            "standups": rows, "rungs": list(rungs), "s5_probe_rows": [], "adapter_hash_set": [], "generated_at": "2026-09-19T00:00:00Z"}
    plan["launch_plan_sha"] = lp.plan_sha(plan)
    return plan


def _prior_dir(tmp: Path, plan: dict, *, s5_wave: str | None = None, name: str = "prior") -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "launch_plan_draft.json").write_text(json.dumps(plan))
    if s5_wave:
        p = d / "smokes" / "S5" / "wave.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"smoke_id": "S5", "wave_uuid": s5_wave}))
    return d


# --------------------------------------------------------------------------- a synthetic COMPLETE attempt root on the "volume"

def _root_files(files: dict, *, plan: dict, row: dict, identity: dict, aid: str = AID, smoke_block=None, status: str = "DONE", audit_verdict: str = "AGREES",
                endpoint_L: int = 11520, n_children: int | None = None) -> str:
    W = plan["W"]
    root = lp.attempt_root_for(identity["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row["kind"])
    n_children = len(row["children"]) if n_children is None else n_children

    def bind(name: str, text: str, writer: str) -> dict:
        return {"sha256": _sha(text), "writer": writer, "tree_sha": W, "rc": 0}
    sampler = json.dumps({**{k: None for k in lp.RESOLVED_SAMPLER_KEYS}, "verdict": "AGREES", "n_client_requests": 45, "class_rule": "x", "reference_source": "vllm_log"})
    k = row["children"][0]
    audit_rel = lp.audit_verdict_rel(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"])
    # F-AO / F-AN: the writer's shape -- the verdict's `arm` is the STORE's spelling (the served cell) and it carries the audit's `arm_binding`
    # (harness id / TrainTest id / vmodel / spelling), which the certificate binds to the plan child's cell through
    cell = str(k.get("cell") or k["arm"])
    harness = lp.harness_id(cell) if cell in lp.HARNESS_ID_OF else cell
    arm_binding = {"arm": cell, "harness_id": harness, "traintest_id": cell, "vmodel": f"petri__{harness}",
                   "spelling": "harness==traintest" if harness == cell else "traintest",
                   "records": {"endpoint.json": "present", "cells_served.json": "present", "routing_table.json": "present", "traintest_spellings": "endpoint.json cells<->harness_cells"}}
    audit = json.dumps({**{kk: None for kk in lp.AUDIT_VERDICT_KEYS}, "leg": k["eval_leg"], "arm": cell, "s": k["s"], "verdict": audit_verdict, "problems": [], "checks": {},
                        "arm_binding": arm_binding, "verdict_file": audit_rel[len(lp.AUDIT_OUT_REL) + 1:],
                        "n_draws": 10, "n_draws_bound": 10, "store_dir": f"{root}/exp/transcripts_results/x", "store_complete": True, "attempt_id": aid, "W": W,
                        "view": row["engine"], "stage": row["stage"], "shard": "shard00", "engine_log": "vllm.log", "resolved_sampler_verdict": "AGREES",
                        "reconciliation_contract": lp.AUDIT_CONTRACT_LEDGERLESS, "ledger": lp.AUDIT_LEDGER_NOT_ON_PATH, "ledger_source": "plan: none declared",
                        "attempt_source": "response_evidence", "evidence_basis": ["request_evidence.jsonl", "response_evidence.jsonl", "vllm.log"],
                        "body_digest_check": "not_available: no ledger on the path", "independence_note": "the ledger's second attempt record is lost"})
    det_leg, det_rel = lp.determinism_rels(row)[0]
    det = json.dumps({"leg": det_leg, "arms": ["none"], "n_items": 3, "n_requests": 3, "n": 3, "identical": 3, "reproduces": True, "misses": [], "standup_id": row["standup_id"]})
    cells = json.dumps({"cells": [], "serving": {"attempt_id": aid, "standup_id": row["standup_id"]}})
    endpoint = json.dumps({"engine": "lora_native", "attested_max_model_len": endpoint_L, "attested_max_num_seqs": 45, "thinking_mode": row["readiness"]["thinking_mode"]})
    cells_served = json.dumps({"canonical": [{"cell_id": "none"}], "controls": [{"cell_id": "none__lora0", "cell_class": "control"}]})
    evidence = {"schema": "seedrep_served/modal_evidence/1", "attempt_id": aid, "W": W, "wave_uuid": identity["wave_uuid"], "standup_id": row["standup_id"],
                "readiness": {"problems": [], "attested_max_model_len": endpoint_L, "attested_max_num_seqs": 45}, "attested_max_num_seqs": 45,
                "object_verification": {"ok": True, "n": 1, "problems": []}, "children": [{"arm": "none", "s": 1, "rc": 0, "eval_leg": row["leg"]} for _ in range(n_children)],
                "standup_argv": row["argv"], "smoke": smoke_block,
                lp.PRODUCER_EVIDENCE_KEY: {lp.RESOLVED_SAMPLER_REL: bind("resolved_sampler", sampler, lp.PRODUCER_WRITERS["resolved_sampler"]),
                                           audit_rel: bind("request_evidence_audit", audit, lp.PRODUCER_WRITERS["request_evidence_audit"]),
                                           lp.CELLS_JSON_REL: bind("adapter_hashes_banked", cells, lp.PRODUCER_WRITERS["adapter_hashes_banked"])},
                lp.DETERMINISM_EVIDENCE_KEY: {det_leg: bind("determinism", det, lp.PRODUCER_WRITERS["determinism"])}}
    files.update({f"{root}/COMPLETE.json": json.dumps({"attempt_id": aid, "control_key": row["control_key"]}),
                  f"{root}/RESULT": f"2026-09-19T00:00:00Z status={status} rc={'0' if status == 'DONE' else '1'} attempt={aid} standup={row['standup_id']} children={n_children}\n",
                  f"{root}/modal_evidence.json": json.dumps(evidence), f"{root}/{lp.RESOLVED_SAMPLER_REL}": sampler, f"{root}/{audit_rel}": audit, f"{root}/{det_rel}": det,
                  f"{root}/{lp.CELLS_JSON_REL}": cells, f"{root}/{lp.SERVE_DIR_REL}/endpoint.json": endpoint, f"{root}/{lp.SERVE_DIR_REL}/cells_served.json": cells_served})
    if (row.get("ladder") or {}).get("cap_next") is not None:
        files[f"{root}/ladder_enqueued.json"] = json.dumps({"schema": "seedrep_ladder_enqueued/1", "attempt_id": aid, "complete": True, "n": 1,
                                                            "entries": [{"item_key": "k", "path": "p", "sha256": "s"}], "streams": [{"rows_read": 5}]})
    return root


# --------------------------------------------------------------------------- synthetic git + readiness contract

class FakeGit:
    """Blob maps per sha; `ls-tree -r -z`, `diff --quiet`, `show sha:path`."""

    def __init__(self, blobs: dict, sources: dict | None = None):
        self.blobs, self.sources, self.calls = blobs, sources or {}, []

    def __call__(self, args):
        self.calls.append(list(args))
        if args[0] == "ls-tree":
            sha, paths = args[3], args[5:]
            return 0, "".join(f"100644 blob {self.blobs[sha][p]}\t{p}\0" for p in paths if p in self.blobs.get(sha, {}))
        if args[0] == "diff":
            a, b, paths = args[2], args[3], args[5:]
            return (1 if any(self.blobs.get(a, {}).get(p) != self.blobs.get(b, {}).get(p) for p in paths) else 0), ""
        if args[0] == "show":
            sha, _, path = args[1].partition(":")
            return (0, self.sources.get((sha, path), "")) if path in self.blobs.get(sha, {}) else (128, "")
        raise AssertionError(args)


def _agd(dicts):
    """The tests' prior-acceptance reader (fj28): the world's Dicts by name."""
    return lambda n: dicts.get(n, {}).get


def _anygit():
    """A FakeGit whose blobs are identical at every synthetic tree these tests use (the certificate's clause (b) recheck, fj25, re-derives 'nothing differs')."""
    return FakeGit({W: _blobs(W) for W in (W_THIS, W_PREV)})


def _blobs(W: str, **over) -> dict:
    files = [STANDUP, CHILD, CFG, MANIFEST, co.PROXY_REL, *lp.PRODUCER_WRITERS.values(), *co.CONTAINER_MODULES]
    b = {f: _sha(f) for f in files}
    b.update(over)
    return b


def fake_readiness(endpoint, cells_served, row, attestation=None):
    """A stub of `seedrep_served.readiness_problems`: the geometry and thinking class against the row's expectations."""
    want = row["readiness"]
    out = []
    if endpoint.get("attested_max_model_len") != want["attested_max_model_len"]:
        out.append(f"attested_max_model_len {endpoint.get('attested_max_model_len')} != {want['attested_max_model_len']}")
    if str(endpoint.get("thinking_mode") or "off") != want["thinking_mode"]:
        out.append(f"thinking_mode {endpoint.get('thinking_mode')!r} != {want['thinking_mode']}")
    return out


def _evaluate(this_plan, priors, files, git, dicts):
    return co.evaluate_carry_over(this_plan=this_plan, this_sha=this_plan["W"], priors=priors, read_text=files.get,
                                  accepted_get_for_dict=lambda name: dicts.get(name, {}).get, git=git, readiness_problems=fake_readiness)


@pytest.fixture
def world(tmp_path):
    prev = _plan(W_PREV, materialisation="final")
    this = _plan(W_THIS, materialisation="final")
    files, dicts = {}, {}
    prow = prev["standups"][0]
    ident = {"W": W_PREV, "W7": prev["W7"], "wave_uuid": WAVE, "results_root": prev["results_root"]}
    root = _root_files(files, plan=prev, row=prow, identity=ident)
    dicts[prev["dicts"]["accepted"]] = {prow["accepted_key"]: AID}
    git = FakeGit({W_PREV: _blobs(W_PREV), W_THIS: _blobs(W_THIS)})
    prior = co.load_prior_source(_prior_dir(tmp_path, prev))
    return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, git=git, prior=prior, root=root, tmp=tmp_path)


# --------------------------------------------------------------------------- (1) identical row + identical producers + complete root -> CARRIED

class TestCarried:
    def test_identical_row_and_producers_and_complete_root_is_carried_with_provenance(self, world):
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 1 and rec["n_refused"] == 0, rec["refused"]
        e = rec["carried"][SID]
        assert e["attempt_id"] == AID and e["attempt_root"] == world.root
        assert e["accepted_key"] == world.this["standups"][0]["accepted_key"]                        # THIS wave's key
        assert e["carried_from"] == {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "attempt_id": AID, "tree_sha": W_PREV, "rs": world.prior["path"],
                                     "accepted_key": world.prev["standups"][0]["accepted_key"], "dict": world.prev["dicts"]["accepted"], "standup_id": SID, "source": "production"}
        assert set(e["clauses"]) == set(co.CLAUSES) and all(e["clauses"][c]["ok"] is True for c in co.CLAUSES)
        ps = e["clauses"]["producer_set"]
        # the W_prev producer-set sha list, per file, at BOTH shas -- what the certificate shows
        assert {f["path"] for f in ps["files"]} >= {STANDUP, CHILD, CFG, MANIFEST, co.PROXY_REL}
        # the evidence writers / checks are NOT producers: excluded from gating by the explicit list, recorded beside
        assert {f["path"] for f in ps["non_eval_modules"]} == {lp.PRODUCER_WRITERS["request_evidence_audit"], lp.PRODUCER_WRITERS["determinism"], lp.PRODUCER_WRITERS["resolved_sampler"]}
        assert all(f["identical"] and f["prev_blob"] == f["this_blob"] == _sha(f["path"]) for f in ps["files"]) and ps["admitted_differences"] == []
        assert ps["git_diff_quiet"]["rc"] == 0 and ps["git_diff_quiet"]["cmd"][:5] == ["git", "diff", "--quiet", W_PREV, W_THIS]
        # the container app / plan modules are recorded, not gating
        assert {c["path"] for c in ps["container_modules"]} == set(co.CONTAINER_MODULES) and not ({f["path"] for f in ps["files"]} & set(co.CONTAINER_MODULES))
        assert e["clauses"]["complete"]["checks"].keys() == {"resolved_sampler", "request_evidence_audit", "determinism"}   # no pool -> no pool_endpoint source
        assert e["clauses"]["engine_identity"]["readiness_problems"] == [] and e["clauses"]["ladder_queue"] == {"ok": True, "producer_row": False}
        # the certificate re-validation accepts it; the record passes on the live Dict once seeded
        assert co.carry_over_problems(rec, plan=world.this, read_text=world.files.get, git=world.git, accepted_get_for_dict=_agd(world.dicts)) == []
        store = {}

        def put(k, v, skip_if_exists=False):                 # modal.Dict.put(skip_if_exists=True) -> False when the key is held
            if skip_if_exists and k in store:
                return False
            store[k] = v
            return True
        seeded = co.seed_accepted(rec, put, store.get)
        assert seeded == {"seeded": [SID], "already_present": []} and store[e["accepted_key"]] == AID
        assert store[co.carried_provenance_key(e["accepted_key"])] == e["carried_from"]
        assert co.carry_over_problems(rec, plan=world.this, read_text=world.files.get, accepted_get=store.get, git=world.git, accepted_get_for_dict=_agd(world.dicts)) == []
        assert co.carried_standup_ids(rec) == {SID: e["carried_from"]}
        # idempotent re-seed; a key held by ANOTHER attempt refuses (a fresh acceptance is never overwritten)
        assert co.seed_accepted(rec, put, store.get) == {"seeded": [], "already_present": [SID]}
        store[e["accepted_key"]] = "someone-else"
        with pytest.raises(ValueError, match="never overwrites an acceptance"):
            co.seed_accepted(rec, put, store.get)

    def test_container_app_change_alone_does_not_block_but_is_recorded(self, world):
        world.git.blobs[W_THIS]["scripts/modal/seedrep_served.py"] = "changed"
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 1
        cm = {c["path"]: c for c in rec["carried"][SID]["clauses"]["producer_set"]["container_modules"]}
        assert cm["scripts/modal/seedrep_served.py"]["identical"] is False and cm["scripts/modal/seedrep_launch_plan.py"]["identical"] is True


# --------------------------------------------------------------------------- (2) one producer file differs -> re-run, clause (b)

class TestProducerSet:
    def test_one_producer_file_differing_names_clause_b(self, world):
        world.git.blobs[W_THIS][CHILD] = "changed-blob"
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 0 and rec["refused"][SID]["clause"] == "producer_set"
        assert CHILD in rec["refused"][SID]["why"] and "git diff --quiet rc 1" in rec["refused"][SID]["why"]
        ps = rec["refused"][SID]["attempts"][0]["clauses"]["producer_set"]
        assert {f["path"]: f["identical"] for f in ps["files"]}[CHILD] is False

    def test_the_standup_manifest_is_a_producer_file(self, world):
        world.git.blobs[W_THIS][MANIFEST] = "one-line-removed"
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "producer_set" and MANIFEST in rec["refused"][SID]["why"]

    def test_a_producer_file_absent_at_one_sha_refuses(self, world):
        del world.git.blobs[W_THIS][co.PROXY_REL]
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "producer_set" and "absent at one of the shas" in rec["refused"][SID]["why"]

    def test_direct_imports_of_a_producer_script_join_the_set(self, world):
        world.git.sources[(W_PREV, CHILD)] = "import sisteer.eval.attacks.bon_served as bs\nimport unified_sweep_run\n"
        for sha in (W_PREV, W_THIS):
            world.git.blobs[sha]["src/sisteer/eval/attacks/bon_served.py"] = "bon-v1"
            world.git.blobs[sha]["scripts/unified_sweep_run.py"] = "usr-v1"
        world.git.blobs[W_THIS]["src/sisteer/eval/attacks/bon_served.py"] = "bon-v2"
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "producer_set" and "bon_served.py" in rec["refused"][SID]["why"]
        derived = rec["refused"][SID]["attempts"][0]["clauses"]["producer_set"]["derived"]
        assert derived["imports"][CHILD] == ["scripts/unified_sweep_run.py", "src/sisteer/eval/attacks/bon_served.py"] and derived["import_depth"] == 1


# --------------------------------------------------------------------------- (3) argv differs -> clause (a)

class TestRowIdentity:
    def test_a_differing_child_argv_names_clause_a(self, world):
        world.this["standups"][0]["children"][0]["argv"][-1] = "8192"                     # a different cap
        world.this["launch_plan_sha"] = lp.plan_sha(world.this)
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "row_identity" and "/children[0]/argv[" in rec["refused"][SID]["why"]

    def test_tree_identity_is_tokenised_not_compared(self, world):
        a = world.prev["standups"][0]
        b = world.this["standups"][0]
        v = co.row_identity_verdict(a, b, prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE}, this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE})
        assert v["ok"] and v["clause"] == "production_row" and v["set_aside"] == ["smoke"]
        n = co.normalise_row(a, W=W_PREV, W7=W_PREV[:7], wave_uuid=WAVE)
        assert n["argv"][1] == f"{TREE}{{W}}/{STANDUP}" and n["accepted_key"].startswith("{W7}|{WAVE}|") and W_PREV not in json.dumps(n)

    def test_a_row_of_this_plan_with_a_smoke_block_is_never_a_target(self, world):
        b = {**world.this["standups"][0], "smoke": {"smoke_id": "S5"}}
        v = co.row_identity_verdict(world.prev["standups"][0], b, prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE}, this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE})
        assert not v["ok"] and "only production rows are carry targets" in v["why"]


# --------------------------------------------------------------------------- (4) root incomplete / audit REFUSED -> clause (c)

class TestComplete:
    def test_result_not_done_names_clause_c(self, world):
        world.files[f"{world.root}/RESULT"] = f"2026-09-19T00:00:00Z status=LADDER_TAILER_ERROR rc=1 attempt={AID}\n"
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "complete" and "RESULT is not `status=DONE rc=0" in rec["refused"][SID]["why"]

    def test_audit_refused_names_clause_c(self, world):
        prow = world.prev["standups"][0]
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": world.prev["results_root"]}
        _root_files(world.files, plan=world.prev, row=prow, identity=ident, audit_verdict="DISAGREES")
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "complete" and "request_evidence_audit" in rec["refused"][SID]["why"] and "not AGREES" in rec["refused"][SID]["why"]

    def test_no_accepted_attempt_in_the_prior_dict_names_clause_c(self, world):
        world.dicts.clear()
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "complete" and "no ACCEPTED attempt" in rec["refused"][SID]["why"]

    def test_a_determinism_record_that_does_not_reproduce_refuses(self, world):
        _leg, rel = lp.determinism_rels(world.prev["standups"][0])[0]
        doc = json.loads(world.files[f"{world.root}/{rel}"])
        doc.update(identical=2, reproduces=False)
        text = json.dumps(doc)
        world.files[f"{world.root}/{rel}"] = text
        ev = json.loads(world.files[f"{world.root}/modal_evidence.json"])
        ev[lp.DETERMINISM_EVIDENCE_KEY][_leg]["sha256"] = _sha(text)
        world.files[f"{world.root}/modal_evidence.json"] = json.dumps(ev)
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "complete" and "criterion 3" in rec["refused"][SID]["why"]


# --------------------------------------------------------------------------- (5) engine identity differs -> clause (d)

class TestEngineIdentity:
    def test_a_different_geometry_in_this_plan_names_clause_d(self, world):
        # this plan's row expects L 11520 like the prior's (clause (a) holds); the ATTEMPT actually served L 8192
        prow = world.prev["standups"][0]
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": world.prev["results_root"]}
        _root_files(world.files, plan=world.prev, row=prow, identity=ident, endpoint_L=8192)
        ev = json.loads(world.files[f"{world.root}/modal_evidence.json"])
        ev["readiness"]["problems"] = []                                          # the prior tree's contract passed it (a weaker reader); THIS tree's must judge
        world.files[f"{world.root}/modal_evidence.json"] = json.dumps(ev)
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "engine_identity" and "attested_max_model_len 8192 != 11520" in rec["refused"][SID]["why"]

    def test_a_missing_ready_record_refuses(self, world):
        del world.files[f"{world.root}/{lp.SERVE_DIR_REL}/cells_served.json"]
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "engine_identity" and "cells_served.json" in rec["refused"][SID]["why"]

    def test_a_reader_that_cannot_judge_is_a_refusal(self, world):
        def boom(*a, **k):
            raise KeyError("expected_window_source")
        rec = co.evaluate_carry_over(this_plan=world.this, this_sha=W_THIS, priors=[world.prior], read_text=world.files.get,
                                     accepted_get_for_dict=lambda name: world.dicts.get(name, {}).get, git=world.git, readiness_problems=boom)
        assert rec["refused"][SID]["clause"] == "engine_identity" and "could not judge" in rec["refused"][SID]["why"]


# --------------------------------------------------------------------------- (6) S1-S4 budgeted smoke refused; an S5 FULL row carries under the named clause

class TestSmokeSources:
    @pytest.mark.parametrize("sid", ["S1", "S1b", "S2", "S4"])
    def test_a_budgeted_smoke_row_is_refused_by_name(self, world, sid):
        ident = lp.smoke_identity(world.prev, sid, S5_WAVE)
        prow = world.prev["standups"][0]
        srow = lp.smoke_row(prow, ident, max_children=5, max_timeout_s=4 * 3600)
        v = co.row_identity_verdict(srow, world.this["standups"][0], prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE},
                                    this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE}, prev_smoke_wave=S5_WAVE)
        assert not v["ok"] and f"ran as smoke {sid!r}" in v["why"] and "BUDGETED" in v["why"]

    def test_an_s3_row_is_a_source_iff_its_shape_is_the_production_rows(self, world):
        """fj12: an S3 smoke row carries under `S3_FULL_CLAUSE` when every production child ran and no child carries a D107 budget flag (this
        one-child pasttense row bounded at five children IS the full shape); bounded below the production child count it stays refused."""
        ident = lp.smoke_identity(world.prev, "S3", S5_WAVE)
        prow = world.prev["standups"][0]
        srow = lp.smoke_row(prow, ident, max_children=5, max_timeout_s=4 * 3600)
        v = co.row_identity_verdict(srow, world.this["standups"][0], prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE},
                                    this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE}, prev_smoke_wave=S5_WAVE)
        assert v["ok"] and v["clause"] == co.S3_FULL_CLAUSE and v["full_smoke_row"] and not v["s5_full_row"]
        wide = copy.deepcopy(world.this["standups"][0])
        wide["children"] = wide["children"] + [dict(wide["children"][0], arm="b", cell="b")]
        v2 = co.row_identity_verdict(srow, wide, prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE}, this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE}, prev_smoke_wave=S5_WAVE)
        assert not v2["ok"] and "ran 1 child(ren), this plan's row has 2" in v2["why"]

    def test_an_s5_full_row_with_production_equal_argv_carries_under_the_named_clause(self, tmp_path):
        prev = _plan(W_PREV, materialisation="draft")                    # a PROVING tree: its smokes ran against the DRAFT
        this = _plan(W_THIS, materialisation="final")
        prow = prev["standups"][0]
        sident = lp.smoke_identity(prev, "S5", S5_WAVE)
        srow = lp.smoke_row(prow, sident, max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
        assert srow["smoke"]["budget"]["kind"] == lp.SMOKE_BUDGET_KIND_FULL and srow["standup_id"] == f"smoke-S5__{SID}"
        files, dicts = {}, {}
        root = _root_files(files, plan=prev, row=srow, identity=sident, smoke_block=srow["smoke"])
        dicts[sident["dicts"]["accepted"]] = {srow["accepted_key"]: AID}
        prior = co.load_prior_source(_prior_dir(tmp_path, prev, s5_wave=S5_WAVE))
        assert prior["smoke_waves"] == {"S5": S5_WAVE} and prior["materialisation"] == "draft"
        git = FakeGit({W_PREV: _blobs(W_PREV), W_THIS: _blobs(W_THIS)})
        rec = _evaluate(this, [prior], files, git, dicts)
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][SID]
        assert e["s5_full_row"] is True and e["clauses"]["row_identity"]["clause"] == co.S5_FULL_CLAUSE
        assert e["clauses"]["row_identity"]["set_aside"] == ["smoke", "standup_id smoke- prefix", "adapters[*].expected_reused", "modal.timeout_s"]
        assert e["carried_from"]["source"] == "s5_smoke" and e["carried_from"]["wave_uuid"] == S5_WAVE and e["carried_from"]["standup_id"] == f"smoke-S5__{SID}"
        assert e["attempt_root"] == root and root.startswith(f"{prev['results_root']}_smoke/S5/")
        assert e["accepted_key"] == this["standups"][0]["accepted_key"]           # admitted under THIS wave's production key
        # the production candidate of the same prior was tried first and refused by name (no accepted attempt under the production identity)
        assert [a["source"] for a in e["refused_before"]] == ["production"] and "no ACCEPTED attempt" in e["refused_before"][0]["why"]
        assert co.carry_over_problems(rec, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) == []

    def test_an_s5_root_whose_evidence_says_it_ran_bounded_is_refused(self, tmp_path):
        prev = _plan(W_PREV, materialisation="draft")
        this = _plan(W_THIS, materialisation="final")
        prow = prev["standups"][0]
        sident = lp.smoke_identity(prev, "S5", S5_WAVE)
        srow = lp.smoke_row(prow, sident, max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
        files, dicts = {}, {}
        _root_files(files, plan=prev, row=srow, identity=sident, smoke_block={**srow["smoke"], "budget": lp.smoke_budget("S3")})
        dicts[sident["dicts"]["accepted"]] = {srow["accepted_key"]: AID}
        prior = co.load_prior_source(_prior_dir(tmp_path, prev, s5_wave=S5_WAVE))
        rec = _evaluate(this, [prior], files, FakeGit({W_PREV: _blobs(W_PREV), W_THIS: _blobs(W_THIS)}), dicts)
        assert rec["n_carried"] == 0 and "not the S5 FULL shape" in rec["refused"][SID]["why"]

    def test_an_s5_probe_row_is_never_a_source_or_target(self, world):
        probe = {**world.prev["standups"][0], "kind": lp.S5_PROBE_KIND}
        v = co.row_identity_verdict(probe, world.this["standups"][0], prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE}, this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE})
        assert not v["ok"] and lp.S5_PROBE_KIND in v["why"]


# --------------------------------------------------------------------------- (e) the ladder queue

class TestLadderQueue:
    def test_a_producer_row_whose_rung_is_in_this_plan_is_refused_by_name(self, tmp_path):
        ladder = {"kind": "static", "cap_next": 8192, "shard": "0", "shards": ["0"], "feeder_shards": ["0"], "producer_of": "8192"}
        rung_this = _row(W_THIS, W_THIS[:7], sid="pasttense__s1__8192__shard00", stage="8192", kind="rung")
        prev = _plan(W_PREV, materialisation="final", rows=[_row(W_PREV, W_PREV[:7], ladder=ladder)])
        this = _plan(W_THIS, materialisation="final", rows=[_row(W_THIS, W_THIS[:7], ladder=ladder)], rungs=[rung_this])
        files, dicts = {}, {}
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": prev["results_root"]}
        _root_files(files, plan=prev, row=prev["standups"][0], identity=ident)
        dicts[prev["dicts"]["accepted"]] = {prev["standups"][0]["accepted_key"]: AID}
        prior = co.load_prior_source(_prior_dir(tmp_path, prev))
        rec = _evaluate(this, [prior], files, FakeGit({W_PREV: _blobs(W_PREV), W_THIS: _blobs(W_THIS)}), dicts)
        assert rec["refused"][SID]["clause"] == "ladder_queue" and "pasttense__s1__8192__shard00" in rec["refused"][SID]["why"]
        assert rec["refused"][SID]["attempts"][0]["clauses"]["complete"]["ladder_inventory"]["n"] == 1        # (c) read the inventory before (e) refused
        # the rung row itself is not a row of the prior plan -> no candidate, never carried
        assert rec["refused"]["pasttense__s1__8192__shard00"]["why"] == "no prior source holds this row"


# --------------------------------------------------------------------------- (7) the certificate re-validation refuses a carried entry with missing provenance

class TestCertificate:
    def test_missing_provenance_or_clause_record_or_stale_root_refuses(self, world):
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert co.carry_over_problems(rec, plan=world.this, read_text=world.files.get, git=world.git, accepted_get_for_dict=_agd(world.dicts)) == []
        bad = copy.deepcopy(rec)
        del bad["carried"][SID]["carried_from"]
        assert any("provenance missing" in p for p in co.carry_over_problems(bad, plan=world.this))
        bad = copy.deepcopy(rec)
        del bad["carried"][SID]["clauses"]["engine_identity"]
        assert any("clause 'engine_identity' missing" in p for p in co.carry_over_problems(bad, plan=world.this))
        bad = copy.deepcopy(rec)
        bad["carried"][SID]["clauses"]["producer_set"]["files"][0]["identical"] = False
        assert any("not a byte-identical set" in p for p in co.carry_over_problems(bad, plan=world.this))
        # the root no longer reads COMPLETE
        world.files[f"{world.root}/RESULT"] = f"status=FAILED rc=1 attempt={AID}"
        assert any("no longer reads COMPLETE" in p for p in co.carry_over_problems(rec, plan=world.this, read_text=world.files.get, git=world.git, accepted_get_for_dict=_agd(world.dicts)))
        # the live Dict names another attempt (or none)
        assert any("accepted Dict names None" in p for p in co.carry_over_problems(rec, plan=world.this, accepted_get={}.get))
        # another plan's record
        other = dict(world.this, W="c" * 40, W7="ccccccc")
        assert any("is not this plan's" in p for p in co.carry_over_problems(rec, plan=other))

    def test_the_certificate_carries_and_revalidates_the_block(self, tmp_path):
        assert "carry_over" in lp.CERTIFICATE_KEYS and lp.CERTIFICATE_KEYS.index("carry_over") == lp.CERTIFICATE_KEYS.index("decode_rows") + 1
        src = inspect.getsource(lp.certificate_problems)
        assert "carry_over_problems(cert[\"carry_over\"]" in src and "PRODUCTION_ACCEPTED_ID" in src
        assert "read_carry_over(Path(rs))" in inspect.getsource(lp.build_certificate)
        assert co.read_carry_over(tmp_path) is None and co.carry_over_problems(None, plan={}) == []
        rec = {"schema": co.CARRY_OVER_SCHEMA, "carried": {}}
        co.write_carry_over(tmp_path, rec)
        assert co.read_carry_over(tmp_path) == rec


# --------------------------------------------------------------------------- (8) at the base tip nothing is carried (pin)

class TestBaseTip:
    def test_no_priors_and_the_same_tree_carry_nothing(self, world):
        rec = _evaluate(world.this, [], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 0 and rec["refused"][SID]["why"] == "no prior source holds this row"
        same = co.load_prior_source(_prior_dir(world.tmp, world.this, name="same"))
        rec = _evaluate(world.this, [same], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 0 and "IS this tree" in rec["refused"][SID]["why"]
        assert co.carried_standup_ids(None) == {} and co.carried_standup_ids(rec) == {}

    def test_a_draft_target_or_an_edited_prior_plan_refuses(self, world):
        with pytest.raises(ValueError, match="FINAL plan only"):
            _evaluate(dict(world.this, materialisation="draft"), [world.prior], world.files, world.git, world.dicts)
        d = _prior_dir(world.tmp, dict(world.prev, launch_plan_sha="0" * 16), name="edited")
        with pytest.raises(ValueError, match="edited after generation"):
            co.load_prior_source(d)
        with pytest.raises(ValueError, match="none of"):
            co.load_prior_source(world.tmp)


# --------------------------------------------------------------------------- the served app: select skips a carried row; launch_production takes the block from the certificate

class _Chain:
    def __getattr__(self, _n):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


class _FakeDict:
    store: dict = {}

    def __init__(self, name):
        self.name = name
        _FakeDict.store.setdefault(name, {})

    @classmethod
    def from_name(cls, name, create_if_missing=False):
        return cls(name)

    def put(self, key, value, skip_if_exists=False):
        d = _FakeDict.store[self.name]
        if skip_if_exists and key in d:
            return False
        d[key] = value
        return True

    def get(self, key, default=None):
        return _FakeDict.store[self.name].get(key, default)


@pytest.fixture(scope="module")
def srv():
    m = types.ModuleType("modal")
    m.App, m.Dict = _App, _FakeDict
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Chain())
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    m.forward = None
    m.FunctionCall = None
    spec = importlib.util.spec_from_file_location("seedrep_served_fj2", REPO / "scripts" / "modal" / "seedrep_served.py")
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal")
    sys.modules["modal"] = m
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


class TestServedApp:
    def test_select_launch_rows_skips_a_carried_row_by_name_and_refuses_an_all_carried_selection(self, srv, capsys):
        this = _plan(W_THIS, materialisation="final", rows=[_row(W_THIS, W_THIS[:7]), _row(W_THIS, W_THIS[:7], sid="codeattack__s1__base__shard00", leg="codeattack")])
        carried = {SID: {"W7": W_PREV[:7], "attempt_id": AID, "tree_sha": W_PREV, "source": "production"}}
        picked, ident = srv.select_launch_rows(this, legs="pasttense,codeattack", carried=carried)
        assert [r["standup_id"] for r in picked] == ["codeattack__s1__base__shard00"] and ident is None
        assert f"CARRIED {SID} from {W_PREV[:7]}/{AID}" in capsys.readouterr().out
        with pytest.raises(srv.LaunchRefused, match="every selected row is CARRIED"):
            srv.select_launch_rows(this, go=SID, carried=carried)
        assert len(srv.select_launch_rows(this, legs="pasttense")[0]) == 1                         # no carry -> unchanged

    def test_launch_production_takes_the_carried_set_from_the_certificate_and_never_spawns_it(self, srv, world, tmp_path, monkeypatch):
        this = _plan(W_THIS, materialisation="final", rows=[_row(W_THIS, W_THIS[:7]), _row(W_THIS, W_THIS[:7], sid="codeattack__s1__base__shard00", leg="codeattack")])
        rec = _evaluate(this, [world.prior], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 1
        rs = tmp_path / "RS"
        rs.mkdir()
        co.write_carry_over(rs, rec)
        cert = {"certificate_sha256": "c" * 64, "plan_sha256": "p" * 64, "launch_plan_sha": this["launch_plan_sha"], "carry_over": rec}
        monkeypatch.setattr(srv, "require_certificate", lambda *a, **k: dict(cert))
        seen = {}

        def fake_steward(*, rs, plan, plan_path, certificate, app_name, sha, tree, app_script, pool_urls, deadline_h, start_token):
            log = Path(rs) / "logs" / "unit.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"STEWARD_STARTED {start_token}\n")
            seen["intent"] = lp.read_production_intent(Path(rs) / lp.PRODUCTION_LAUNCH_FILE)
            return {"launched": True, "unit": "u", "log": str(log)}
        common = dict(plan_path=tmp_path / "plan.json", rs=rs, app_name="seedrep-served-bbbbbbb", sha=W_THIS, pools=None, pool_urls_path=None, cert_p=tmp_path / "cert.json",
                      steward_launch=fake_steward, wait_started=lambda log, token, timeout_s: {"pid": 1, "required_keys": 2}, new_token=lambda: "tok1")
        out = srv.launch_production(this, legs="pasttense,codeattack", carry_over=[world.prior["path"]], **common)
        assert out["mode"] == "new" and out["added"] == ["codeattack__s1__base__shard00"]
        assert sorted(seen["intent"]["rows"]) == ["codeattack__s1__base__shard00"]                 # the carried row is NOT in the fan-out intent
        assert SID in seen["intent"]["required_keys"].values()                                       # ... but its key IS required of the wave (the carried acceptance satisfies it)
        legs_log = (rs / "legs.log").read_text()
        assert f"CARRIED {SID} attempt={AID} from {W_PREV[:7]}/production" in legs_log

    def test_launch_production_refuses_a_carry_over_flag_the_certificate_does_not_carry_or_that_names_other_priors(self, srv, world, tmp_path, monkeypatch):
        this = _plan(W_THIS, materialisation="final")
        rs = tmp_path / "RS"
        rs.mkdir()
        common = dict(plan_path=tmp_path / "plan.json", rs=rs, app_name="x", sha=W_THIS, pools=None, pool_urls_path=None, cert_p=tmp_path / "cert.json")
        monkeypatch.setattr(srv, "require_certificate", lambda *a, **k: {"certificate_sha256": "c" * 64, "plan_sha256": "p" * 64, "launch_plan_sha": this["launch_plan_sha"], "carry_over": None})
        with pytest.raises(srv.LaunchRefused, match="carries no carry-over block"):
            srv.launch_production(this, legs="pasttense", carry_over=[world.prior["path"]], **common)
        rec = _evaluate(this, [world.prior], world.files, world.git, world.dicts)
        co.write_carry_over(rs, rec)
        monkeypatch.setattr(srv, "require_certificate", lambda *a, **k: {"certificate_sha256": "c" * 64, "plan_sha256": "p" * 64, "launch_plan_sha": this["launch_plan_sha"], "carry_over": rec})
        with pytest.raises(srv.LaunchRefused, match="was evaluated over"):
            srv.launch_production(this, legs="pasttense", carry_over=[str(tmp_path / "other")], **common)
        co.write_carry_over(rs, {**rec, "evaluated_utc": "later"})
        with pytest.raises(srv.LaunchRefused, match="is not the certified carry-over record"):
            srv.launch_production(this, legs="pasttense", carry_over=[world.prior["path"]], **common)

    def test_carry_over_evaluate_writes_the_record_seeds_this_waves_dict_and_logs(self, srv, world, tmp_path):
        this = _plan(W_THIS, materialisation="final")
        rs = tmp_path / "RS"
        rs.mkdir()
        store = {}
        out = srv.carry_over_evaluate(this, rs=rs, sha=W_THIS, sources=[world.prior["path"]], accepted_get_for_dict=lambda name: world.dicts.get(name, {}).get,
                                      read_text=world.files.get, git=world.git, accepted_put=lambda k, v, skip_if_exists=False: store.setdefault(k, v) is v, accepted_get=store.get,
                                      readiness=fake_readiness, reaudit=False)
        assert out["n_carried"] == 1 and out["carried"] == [SID] and out["seeded"] == {"seeded": [SID], "already_present": []}
        assert store[this["standups"][0]["accepted_key"]] == AID
        rec = co.read_carry_over(rs)
        assert rec["carried"][SID]["attempt_id"] == AID and co.carry_over_problems(rec, plan=this, read_text=world.files.get, accepted_get=store.get, git=world.git, accepted_get_for_dict=_agd(world.dicts)) == []
        assert f"CARRIED {SID} attempt={AID}" in (rs / "legs.log").read_text()
        assert "seedrep_carry_over.py" in inspect.getsource(srv).count("seedrep_carry_over.py") * "seedrep_carry_over.py"   # shipped into the images

    def test_the_live_getter_resolves_the_production_id_to_this_waves_dict(self, srv):
        this = _plan(W_THIS, materialisation="final")
        _FakeDict.store[this["dicts"]["accepted"]] = {this["standups"][0]["accepted_key"]: AID}
        g = srv.live_smoke_accepted_getters(Path("/nonexistent"), this)(co.PRODUCTION_ACCEPTED_ID)
        assert g(this["standups"][0]["accepted_key"]) == AID and g.record()["dict"] == this["dicts"]["accepted"]


# --------------------------------------------------------------------------- user ruling 18:55Z: non-eval modules excluded by name; the Petri fallback chain is an admitted difference

class TestUserRuling1855:
    def test_a_differing_evidence_writer_does_not_block_and_is_recorded(self, world):
        world.git.blobs[W_THIS][lp.PRODUCER_WRITERS["request_evidence_audit"]] = "audit-v2"
        world.git.blobs[W_THIS]["src/sisteer/capeval/tinycap/contracts.py"] = "certification-v2"
        world.git.blobs[W_PREV]["src/sisteer/capeval/tinycap/contracts.py"] = "certification-v1"
        world.git.sources[(W_PREV, CHILD)] = "from sisteer.capeval.tinycap import contracts as tc\n"
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 1, rec["refused"]
        ps = rec["carried"][SID]["clauses"]["producer_set"]
        ne = {f["path"]: f["identical"] for f in ps["non_eval_modules"]}
        assert ne[lp.PRODUCER_WRITERS["request_evidence_audit"]] is False and ne["src/sisteer/capeval/tinycap/contracts.py"] is False
        assert not ({f["path"] for f in ps["files"]} & set(co.NON_EVAL_MODULES))

    def test_the_driver_itself_stays_gating_even_when_it_also_holds_a_probe(self, world):
        world.this["standups"][0]["children"][0]["argv"][1] = f"{TREE}{W_THIS}/scripts/served_capability_run.py"
        world.prev["standups"][0]["children"][0]["argv"][1] = f"{TREE}{W_PREV}/scripts/served_capability_run.py"
        for plan in (world.this, world.prev):
            plan["launch_plan_sha"] = lp.plan_sha(plan)
        world.prior = co.load_prior_source(_prior_dir(world.tmp, world.prev, name="prior2"))
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": world.prev["results_root"]}
        _root_files(world.files, plan=world.prev, row=world.prev["standups"][0], identity=ident)
        world.git.blobs[W_PREV]["scripts/served_capability_run.py"] = "scr-v1"
        world.git.blobs[W_THIS]["scripts/served_capability_run.py"] = "scr-v2"
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "producer_set" and "served_capability_run.py" in rec["refused"][SID]["why"]

    def test_a_fallback_chain_only_manifest_difference_is_admitted_and_recorded(self, world):
        prev_text = "petri:\n  auditor_model: anthropic/claude-sonnet-5\n  auditor_fallback_models:\n    - openrouter/x-ai/grok-4.6\n    - openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated\n  judge_model: anthropic/claude-opus-5\n"
        this_text = "petri:\n  auditor_model: anthropic/claude-sonnet-5\n  auditor_fallback_models:\n    - openrouter/x-ai/grok-4.6\n  judge_model: anthropic/claude-opus-5\n"
        world.git.blobs[W_THIS][MANIFEST] = "manifest-v2"
        world.git.sources[(W_PREV, MANIFEST)], world.git.sources[(W_THIS, MANIFEST)] = prev_text, this_text
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["n_carried"] == 1, rec["refused"]
        ps = rec["carried"][SID]["clauses"]["producer_set"]
        assert ps["git_diff_quiet"]["rc"] == 1 and ps["admitted_differences"][0]["path"] == MANIFEST
        assert ps["admitted_differences"][0]["removed"] == ["- openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated"] and ps["admitted_differences"][0]["added"] == []
        assert {f["path"]: f.get("admitted_difference") for f in ps["files"]}[MANIFEST] == co.ADMITTED_REASON_CHAIN
        assert co.carry_over_problems(rec, plan=world.this, read_text=world.files.get, git=world.git, accepted_get_for_dict=_agd(world.dicts)) == []
        # a manifest change OUTSIDE the chain lists is not admitted
        world.git.sources[(W_THIS, MANIFEST)] = this_text.replace("claude-opus-5", "claude-opus-4-8")
        rec = _evaluate(world.this, [world.prior], world.files, world.git, world.dicts)
        assert rec["refused"][SID]["clause"] == "producer_set" and "not an admitted difference" in rec["refused"][SID]["why"]

    def test_a_petri_row_with_an_admitted_chain_difference_is_refused_until_the_sidecar_proof_exists(self, tmp_path):
        prev = _plan(W_PREV, materialisation="final", rows=[_row(W_PREV, W_PREV[:7], sid="petri__s1__base__shard00", leg="petri")])
        this = _plan(W_THIS, materialisation="final", rows=[_row(W_THIS, W_THIS[:7], sid="petri__s1__base__shard00", leg="petri")])
        files, dicts = {}, {}
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": prev["results_root"]}
        _root_files(files, plan=prev, row=prev["standups"][0], identity=ident)
        dicts[prev["dicts"]["accepted"]] = {prev["standups"][0]["accepted_key"]: AID}
        blobs = {W_PREV: _blobs(W_PREV), W_THIS: _blobs(W_THIS)}
        for sha in blobs:
            for f in co.LEG_HELPERS["petri"]:
                blobs[sha][f] = "same"
        blobs[W_THIS][MANIFEST] = "manifest-v2"
        git = FakeGit(blobs, {(W_PREV, MANIFEST): "petri:\n  auditor_fallback_models:\n    - a\n    - b\n", (W_THIS, MANIFEST): "petri:\n  auditor_fallback_models:\n    - a\n"})
        rec = _evaluate(this, [co.load_prior_source(_prior_dir(tmp_path, prev))], files, git, dicts)
        assert rec["refused"]["petri__s1__base__shard00"]["clause"] == "producer_set" and "sidecar proof" in rec["refused"]["petri__s1__base__shard00"]["why"]
