"""fj5: the three carry-over P1s, each reproduced through the REAL reader it names.

Known-bad at the assembly tip 2174e5139:
  1. `audit_common.bind_roots_to_plan` rejected a CARRIED attempt's historical W ("ran from W ... but the launch plan is W ..."), then wave,
     then S5 smoke identity, and `coverage_table.build` called it unconditionally -- an accepted-and-skipped row the coverage pipeline
     could not consume. Now a VALIDATED carry mapping (`RS/carry_over.json`) binds the historical root to its planned standup with both
     identities preserved.
  2. `seedrep_carry_over.petri_sidecar_proof` read the aggregate `auditor_chain` tally and the auditor-refusal ledger; the real summary
     producer EXCLUDES judge-refused conversations from that tally and keeps their auditor provenance under `judge_refused_items` (+ the
     judge ledger) -- a Qwen-carried, judge-refused conversation proved as zero uses. Now every outcome's provenance is read and the
     conservation law (`n_items == audited + auditor_refused + judge_refused`) is verified.
  3. The default Petri proof read the local stores outside the evaluator's RecordingReader, so certification never re-read them. Now the
     proof reads only through the reader it is handed; its inputs (absent sidecars as absent) are bound on the entry
     (`evidence_inventory_local`) and re-read by `carry_over_problems`.
"""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import coverage_table as ct  # noqa: E402
import engine_identity_table as eit  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_fj2_carry_over as F  # noqa: E402  -- the synthetic two-tree carry world
from sisteer.capeval.petri import auditor_fallback as af  # noqa: E402
from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402
from test_petri_auditor_fallback import _chain as chain_record  # noqa: E402
from test_petri_judge_refusal import refused_sample  # noqa: E402
from test_petri_store_shape import BEHAVIORS, SIDS, _run_cell, full_dims, make_log, sample, world  # noqa: E402, F401  (world is a fixture)

W_PRIOR = "a" * 40
WAVE_PRIOR = "33333333-4444-4555-8666-777777777777"


@pytest.fixture(scope="module")
def srv():
    """`seedrep_served` under fj2's modal stub (`_App` / `_FakeDict` / `_Chain`), loaded once under its own module name."""
    import importlib.util
    m = types.ModuleType("modal")
    m.App, m.Dict = F._App, F._FakeDict
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: F._Chain())
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: F._Chain(), debian_slim=lambda *a, **k: F._Chain())
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    m.forward = None
    m.FunctionCall = None
    spec = importlib.util.spec_from_file_location("seedrep_served_fj5", REPO / "scripts" / "modal" / "seedrep_served.py")
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
S5_WAVE = "11111111-2222-4333-8444-555555555555"
PIN = "d" * 64
ARMS = ("none", "constv2-mix-full__psr_ce_all")


# =========================================================================== P1 #1: the coverage binder admits VALIDATED carries

def _chain(tmp_path: Path):
    """The coverage test's real chain (audit -> identity -> expected -> plan) over a pasttense wave of s1..s3 + arditi."""
    stores = tmp_path / "stores"
    roots = fx.build_wave(stores, legs=("pasttense",), views=("s1", "s2", "s3"), arms=ARMS, n_items=2, arditi=True, arditi_pin=PIN)
    return stores, roots


def _rekey_root_to_history(root, *, W: str, wave: str, smoke: str | None = None) -> None:
    """Make a fixture root a CARRIED one: its ATTEMPT.json spells the HISTORICAL tree / wave (and, for an S5 FULL source, the smoke row it
    ran as); its standup record's sha follows. Nothing else in the root changes -- the store IS the prior wave's."""
    ap = root.path / ac.ATTEMPT_FILE
    att = json.loads(ap.read_text())
    sid = fx.standup_id(root.leg, root.view, root.stage, root.shard_index)
    att["W"], att["wave_uuid"] = W, wave
    att["control_key"] = fx.control_key(W[:7], wave, root.leg, root.view, root.stage, root.shard_index)
    att["launch_plan_row"] = f"smoke-{smoke}__{sid}" if smoke else sid
    ap.write_text(json.dumps(att))
    sj = root.path / ac.SERVE_DIR_REL / "standup_jobs.jsonl"
    if sj.is_file():
        fx.rewrite_jsonl(sj, lambda r: {**r, "sha": W} if r.get("sha") else r)


def _finish_chain(tmp_path: Path, stores: Path, roots):
    audits = tmp_path / "audits"
    assert ac.run_main(rea.main, ["--stores", str(stores), "--out", str(audits), "--legs", "pasttense"]) == 0
    pins = fx.pins_file(tmp_path / "pins.json", pin=PIN)
    ident = tmp_path / "tables" / "engine_identity.csv"
    assert ac.run_main(eit.main, ["--stores", str(stores), "--out", str(ident), "--arditi-pins", str(pins)]) == 0
    expected = fx.expected_file(tmp_path / "expected.json", arms=ARMS, views=("s1", "s2", "s3"), arditi_seeds=(1, 2, 3), legs=("pasttense",))
    plan = fx.launch_plan_file(tmp_path / "launch_plan.json", list(roots.values()))
    return dict(expected=expected, launch_plan=plan, audit_summary=audits / rea.SUMMARY_NAME, identity_legs=tmp_path / "tables" / "engine_identity_legs.json",
                petri_behaviours=list(fx.PETRI_BEHAVIOURS))


def _carry_record(entries) -> dict:
    """A `seedrep_carry_over/1` record for THIS plan (fixture W / wave) carrying ``entries`` = [(root, hist_W, hist_wave, source)]."""
    carried = {}
    for root, W, wave, source in entries:
        sid = fx.standup_id(root.leg, root.view, root.stage, root.shard_index)
        hist_sid = f"smoke-S5__{sid}" if source == "s5_smoke" else sid
        key = lp.accepted_key(fx.W_FIXTURE[:7], fx.WAVE_UUID, root.leg, root.view, root.stage, root.shard_index)
        carried[sid] = {"attempt_id": root.attempt_id, "accepted_key": key, "attempt_root": f"/vol/seedrep/{W[:7]}/exp/seedrep_battery_modal/x/{root.attempt_id}",
                        "carried_from": {"W": W, "W7": W[:7], "wave_uuid": wave, "attempt_id": root.attempt_id, "tree_sha": W, "rs": "RS/prior/x",
                                         "accepted_key": lp.accepted_key(W[:7], wave, root.leg, root.view, root.stage, root.shard_index), "dict": "d",
                                         "standup_id": hist_sid, "source": source},
                        "clauses": {c: {"ok": True} for c in co.CLAUSES}}
    return {"schema": co.CARRY_OVER_SCHEMA, "W": fx.W_FIXTURE, "W7": fx.W_FIXTURE[:7], "wave_uuid": fx.WAVE_UUID, "carried": carried, "refused": {},
            "n_carried": len(carried), "n_refused": 0}


def _agd(dicts):
    """The tests' prior-acceptance reader (fj28): the world's Dicts by name."""
    return lambda n: dicts.get(n, {}).get


class TestCoverageBinderCarries:
    def test_schema_and_clauses_are_the_carry_modules(self):
        assert ac.CARRY_OVER_SCHEMA == co.CARRY_OVER_SCHEMA and ac.CARRY_CLAUSES == co.CLAUSES and set(ac.CARRY_PROVENANCE_KEYS) <= set(co.PROVENANCE_KEYS) | {"standup_id"}

    def test_a_carried_production_root_and_a_carried_s5_full_root_bind_through_coverage_build(self, tmp_path):
        stores, roots = _chain(tmp_path)
        # the WHOLE leg is carried (the prior-tree shape: every completed view of a leg): three production carries and, for s2, the proving
        # tree's S5 FULL row. The identity table FAILs a leg whose base engines span two trees ("W differs across base engines"),
        # so a leg mixing carried and fresh roots is a separate, named downstream gap -- not the binder's
        prod, s5 = roots[("pasttense", "s1")], roots[("pasttense", "s2")]
        for key in (("pasttense", "s1"), ("pasttense", "s3"), ("pasttense", ac.ARDITI_VIEW)):
            _rekey_root_to_history(roots[key], W=W_PRIOR, wave=WAVE_PRIOR)
        _rekey_root_to_history(s5, W=W_PRIOR, wave=S5_WAVE, smoke="S5")           # a proving tree's S5 FULL carry
        kw = _finish_chain(tmp_path, stores, roots)
        # the pulled roots spell their HISTORICAL identity (never rewritten)
        r_prod = ac.read_attempt_root(prod.path)
        r_s5 = ac.read_attempt_root(s5.path)
        assert (r_prod.W, r_prod.wave_uuid, r_prod.smoke_id) == (W_PRIOR, WAVE_PRIOR, None)
        assert (r_s5.W, r_s5.wave_uuid, r_s5.smoke_id, r_s5.standup_id) == (W_PRIOR, S5_WAVE, "S5", "smoke-S5__pasttense__s2__base__shard00")
        # KNOWN-BAD (the assembly tip): without the carry record the coverage pipeline refuses the carried root by its historical W
        with pytest.raises(ac.AuditRefusal) as ei:
            ct.build(stores, **kw)
        assert "ran from W" in str(ei.value) and W_PRIOR in str(ei.value)
        # with the VALIDATED carry mapping every arm-run is complete, and the carried roots are bound to THEIR planned standups
        rec_path = tmp_path / "carry_over.json"
        rec_path.write_text(json.dumps(_carry_record([(prod, W_PRIOR, WAVE_PRIOR, "production"), (s5, W_PRIOR, S5_WAVE, "s5_smoke"),
                                                      (roots[("pasttense", "s3")], W_PRIOR, WAVE_PRIOR, "production"),
                                                      (roots[("pasttense", ac.ARDITI_VIEW)], W_PRIOR, WAVE_PRIOR, "production")])))
        rows = ct.build(stores, **kw, carry_over=rec_path)
        assert rows and all(r["status"] == ct.COMPLETE for r in rows), [r["status"] for r in rows if r["status"] != ct.COMPLETE]
        plan = ac.read_launch_plan(kw["launch_plan"])
        carried = ac.read_carry_mapping(rec_path, plan)
        bound = ac.bind_roots_to_plan(ac.discover_attempt_roots(stores), plan, carried=carried)
        assert bound["pasttense__s1__base__shard00"].attempt_id == prod.attempt_id and bound["pasttense__s1__base__shard00"].W == W_PRIOR
        assert bound["pasttense__s2__base__shard00"].attempt_id == s5.attempt_id and bound["pasttense__s2__base__shard00"].standup_id == "smoke-S5__pasttense__s2__base__shard00"
        assert bound["pasttense__s1__base__shard00"].carried_from["bound_to"] == "pasttense__s1__base__shard00"
        assert bound["pasttense__s2__base__shard00"].carried_from["source"] == "s5_smoke" and bound["pasttense__s2__base__shard00"].carried_from["wave_uuid"] == S5_WAVE
        assert bound["pasttense__s3__base__shard00"].carried_from["source"] == "production"
        # a root of THIS wave binds without any mapping, exactly as before (the fresh-root path is untouched)
        fresh = fx.make_root(tmp_path / "fresh", leg="pasttense", view="s3", arms=ARMS)
        assert ac.bind_roots_to_plan([ac.read_attempt_root(fresh.path)], plan)["pasttense__s3__base__shard00"].W == fx.W_FIXTURE

    def test_the_mapping_refuses_a_record_that_does_not_describe_the_pulled_root(self, tmp_path):
        stores, roots = _chain(tmp_path)
        prod = roots[("pasttense", "s1")]
        for r in roots.values():
            _rekey_root_to_history(r, W=W_PRIOR, wave=WAVE_PRIOR)
        kw = _finish_chain(tmp_path, stores, roots)
        plan = ac.read_launch_plan(kw["launch_plan"])
        good = _carry_record([(r, W_PRIOR, WAVE_PRIOR, "production") for r in roots.values()])
        assert set(ac.carry_mapping(good, plan)) == {r.attempt_id for r in roots.values()}
        sid = "pasttense__s1__base__shard00"
        # provenance names another tree than the root recorded -> the binder refuses (the pulled root is not the carried one)
        other = copy.deepcopy(good)
        other["carried"][sid]["carried_from"]["W"] = other["carried"][sid]["carried_from"]["tree_sha"] = "b" * 40
        with pytest.raises(ac.AuditRefusal, match="ran from W"):
            ac.bind_roots_to_plan(ac.discover_attempt_roots(stores), plan, carried=ac.carry_mapping(other, plan))
        # the record itself: THIS tree as the provenance, a missing clause, an unknown standup, an S5 source without the smoke spelling
        for mutate, needle in ((lambda r: r["carried"][sid]["carried_from"].update(W=fx.W_FIXTURE, tree_sha=fx.W_FIXTURE), "prior tree"),
                               (lambda r: r["carried"][sid]["clauses"].pop("ladder_queue"), "clause"),
                               (lambda r: r["carried"].update({"pasttense__s9__base__shard00": r["carried"][sid]}), "does not declare|not a standup"),
                               (lambda r: r["carried"][sid]["carried_from"].update(source="s5_smoke"), "smoke-S5__"),
                               (lambda r: r["carried"][sid].pop("carried_from"), "provenance"),
                               (lambda r: r.update(W="c" * 40), "evaluated for W")):
            bad = copy.deepcopy(good)
            mutate(bad)
            with pytest.raises(ac.AuditRefusal, match=needle):
                ac.carry_mapping(bad, plan)
        # a smoke root of THIS wave with NO carry entry is still refused by name -- the S5 admission exists only through the mapping
        _rekey_root_to_history(roots[("pasttense", "s3")], W=fx.W_FIXTURE, wave=fx.WAVE_UUID, smoke="S5")
        without_s3 = copy.deepcopy(good)
        without_s3["carried"].pop("pasttense__s3__base__shard00")
        with pytest.raises(ac.AuditRefusal, match="SMOKE standup"):
            ac.bind_roots_to_plan(ac.discover_attempt_roots(stores), plan, carried=ac.carry_mapping(without_s3, plan))


# =========================================================================== P1 #2: judge-refused conversations enter the zero-use proof

def _read(p: str):
    return Path(p).read_text() if Path(p).is_file() else None


def _drive_with_judge_refusal(world, cell, *, chain_labels, refused=(BEHAVIORS[1], SIDS[1])):
    """The judge-refusal test's driver, with the refused sample carrying the auditor-chain provenance the real log carries."""
    def drive(tasks, *, spec, cell, engine):
        logs = []
        for t in tasks:
            samples = []
            for sid in (t.scenario_ids if t.scenario_ids is not None else SIDS):
                if (t.behavior_id, sid) == refused:
                    s = refused_sample(sid)
                    s["store"] = {af.AUDITOR_CHAIN_STORE_KEY: chain_record(chain_labels)}
                else:
                    s = sample(sid, full_dims(t.behavior_id))
                samples.append(s)
            log = make_log(world, cell, t.dir_key, samples, spec=spec)
            log["location"] = f"/synthetic/{t.behavior_id}.eval"
            logs.append(log)
        return logs
    return drive


class TestJudgeRefusedProvenance:
    def test_a_qwen_carried_judge_refused_conversation_refuses_the_proof_through_the_real_producer(self, world, tmp_path):
        cell = world.cells["si_safety__psr_kl_all"]
        store, result = _run_cell(world, cell, tmp_path, _drive_with_judge_refusal(world, cell, chain_labels=["sonnet-5", "grok-4.6", "qwen-abl"]))
        # the REAL producer's shape: the tally EXCLUDES the judge-refused item, whose chain sits under judge_refused_items + the ledger
        rb = BEHAVIORS[1]
        results = json.loads((store.cell_dir / pr.RESULTS_NAME if hasattr(pr, "RESULTS_NAME") else store.cell_dir / "results.json").read_text())
        assert results["judge_refused"] == {"total": 1} and results["auditor_chain"]["continued_by_qwen"] == 0
        item = results["behaviors"][rb]["judge_refused_items"][0]
        assert item["auditor_chain"]["used"][-1]["label"] == "qwen-abl"
        ledger = json.loads(pr.judge_refused_path(store).read_text())
        assert ledger[pr.JUDGE_REFUSED_OUTCOME][0]["auditor_chain"]["used"][-1]["label"] == "qwen-abl"
        # the proof: refused by name, ONE use, found in the judge-refused conversation; outcome coverage holds
        pf = co.petri_sidecar_proof([str(store.cell_dir)], _read)
        assert pf["ok"] is False and pf["removed_leg"]["uses"] == 1 and "judge-refused" in pf["why"]
        assert pf["per_cell"][0]["judge_refused_removed_leg_uses"] == 1 and pf["per_cell"][0]["judge_refused"] == 1
        assert pf["coverage"]["ok"] and pf["judge_refused_records"] == 1 and pf["used_legs"]["qwen-abl"] == 1
        # the evidence inventory of the proof names every input it read, the auditor ledger recorded ABSENT (the cell has no auditor refusal)
        rr = co.RecordingReader(_read)
        co.petri_sidecar_proof([str(store.cell_dir)], rr)
        assert rr.seen[f"{store.cell_dir}/{co.PETRI_REFUSED_LEDGER}"] is None and rr.seen[f"{store.cell_dir}/{co.PETRI_JUDGE_REFUSED_LEDGER}"] is not None

    def test_a_sonnet_only_judge_refused_conversation_proves_and_the_conservation_law_gates(self, world, tmp_path):
        cell = world.cells["si_safety__psr_kl_all"]
        store, result = _run_cell(world, cell, tmp_path, _drive_with_judge_refusal(world, cell, chain_labels=["sonnet-5"]))
        pf = co.petri_sidecar_proof([str(store.cell_dir)], _read)
        assert pf["ok"] and pf["removed_leg"]["uses"] == 0 and pf["coverage"]["ok"] and pf["per_cell"][0]["judge_refused"] == 1
        # an item landing in no tally: results.json's n_items raised by one -> the proof refuses on coverage, never on faith
        p = store.cell_dir / "results.json"
        doc = json.loads(p.read_text())
        doc["behaviors"][BEHAVIORS[0]]["n_items"] += 1
        p.write_text(json.dumps(doc))
        pf = co.petri_sidecar_proof([str(store.cell_dir)], _read)
        assert pf["ok"] is False and "coverage" in pf["why"] and any("lands in no tally" in x for x in pf["coverage"]["problems"])
        # the judge ledger deleted while the summary counts one judge refusal -> refused (a ledger must hold exactly the counted records)
        p.write_text(json.dumps(json.loads(p.read_text()) | {"behaviors": {k: ({**v, "n_items": v["n_items"] - (1 if k == BEHAVIORS[0] else 0)}) for k, v in doc["behaviors"].items()}}))
        pr.judge_refused_path(store).unlink()
        pf = co.petri_sidecar_proof([str(store.cell_dir)], _read)
        assert pf["ok"] is False and any("judge-refused item(s) but" in x for x in pf["coverage"]["problems"])


# =========================================================================== P1 #3: the proof's local inputs are bound and re-validated

def _petri_carry_world(tmp_path, *, with_bundle=False):
    prev = F._plan(F.W_PREV, materialisation="final", rows=[F._row(F.W_PREV, F.W_PREV[:7], sid="petri__s1__base__shard00", leg="petri")])
    this = F._plan(F.W_THIS, materialisation="final", rows=[F._row(F.W_THIS, F.W_THIS[:7], sid="petri__s1__base__shard00", leg="petri")])
    files, dicts = {}, {}
    ident = {"W": F.W_PREV, "W7": F.W_PREV[:7], "wave_uuid": F.WAVE, "results_root": prev["results_root"]}
    root = F._root_files(files, plan=prev, row=prev["standups"][0], identity=ident)
    if with_bundle:
        ev = json.loads(files[f"{root}/modal_evidence.json"])
        ev["petri"] = {"state": "complete", "kind": "completion", "evidence": {"attempt_id": F.AID, "W7": F.W_PREV[:7], "wave_uuid": F.WAVE, "view": "s1",
                                                                            "verdict": {"detail": [["leg_0_s1", "petri__none__s1", "complete"]], "release": True,
                                                                                        "counts": {"complete": 1, "incomplete": 0, "failed": 0}}}}
        files[f"{root}/modal_evidence.json"] = json.dumps(ev)
    dicts[prev["dicts"]["accepted"]] = {prev["standups"][0]["accepted_key"]: F.AID}
    blobs = {F.W_PREV: F._blobs(F.W_PREV), F.W_THIS: F._blobs(F.W_THIS)}
    for sha in blobs:
        for f in co.LEG_HELPERS["petri"]:
            blobs[sha][f] = "same"
    blobs[F.W_THIS][F.MANIFEST] = "manifest-v2"
    git = F.FakeGit(blobs, {(F.W_PREV, F.MANIFEST): "petri:\n  auditor_fallback_models:\n    - a\n    - b\n", (F.W_THIS, F.MANIFEST): "petri:\n  auditor_fallback_models:\n    - a\n"})
    prior = co.load_prior_source(F._prior_dir(tmp_path, prev))
    return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, git=git, prior=prior, root=root)


def _local_store(cell: str) -> dict:
    """A grading-host-local Petri cell as a dict-backed filesystem: results.json only (no refusal ledgers)."""
    tally = {"sonnet_only": 4, "continued_by_grok": 0, "continued_by_qwen": 0, "auditor_refused_all_legs": 0, "auditor_unavailable": 0}
    beh = {b: {"n_items": 2, "audited": 2, "auditor_refused": {"total": 0}, "auditor_chain": {**tally, "sonnet_only": 2}, "auditor_refused_items": []} for b in ("b1", "b2")}
    return {f"{cell}/results.json": json.dumps({"audited": {"items": 4, "expected": 4}, "auditor_chain": tally, "behaviors": beh})}


class TestProofInputsBound:
    def test_the_evaluator_binds_every_local_proof_input_and_the_certificate_rechecks_them(self, tmp_path):
        w = _petri_carry_world(tmp_path)
        cell = "/home/x/.local/state/petri_native/petri_seedrep_aaaaaaa_22222222_01TESTTASK-deadbeef0001_qwen35_4b_s1/cells/petri__none__s1"
        local = _local_store(cell)
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=F.W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                     git=w.git, readiness_problems=F.fake_readiness, local_read_text=local.get,
                                     petri_proof=lambda **kw: co.petri_sidecar_proof([cell], kw["read_text"]))
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"]["petri__s1__base__shard00"]
        inv = e["evidence_inventory_local"]
        assert set(inv) == {f"{cell}/results.json", f"{cell}/{co.PETRI_REFUSED_LEDGER}", f"{cell}/{co.PETRI_JUDGE_REFUSED_LEDGER}"}
        assert inv[f"{cell}/results.json"] == co._sha_text(local[f"{cell}/results.json"]) and inv[f"{cell}/{co.PETRI_REFUSED_LEDGER}"] is None
        assert f"{cell}/results.json" not in e["evidence_inventory"]          # the volume inventory and the local one are distinct
        ok = lambda l: co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, local_read_text=l.get, git=w.git, accepted_get_for_dict=_agd(w.dicts),  # noqa: E731
                                              petri_proof=lambda **kw: co.petri_sidecar_proof([cell], kw["read_text"]))                # fj28: the certificate re-derives the proof
        assert ok(local) == []
        # KNOWN-BAD at the tip: removing / changing the local results.json or a refusal ledger appearing left the certificate silent
        gone = dict(local); gone.pop(f"{cell}/results.json")
        assert any("no longer present" in p for p in ok(gone))
        changed = dict(local); changed[f"{cell}/results.json"] = changed[f"{cell}/results.json"].replace('"sonnet_only": 4', '"sonnet_only": 3')
        assert any("changed since the evaluation" in p for p in ok(changed))
        appeared = dict(local); appeared[f"{cell}/{co.PETRI_JUDGE_REFUSED_LEDGER}"] = json.dumps({"judge_refused": []})
        assert any("ABSENT at the evaluation and exists now" in p for p in ok(appeared))
        stripped = copy.deepcopy(rec); stripped["carried"]["petri__s1__base__shard00"].pop("evidence_inventory_local")
        assert any("not bound" in p for p in co.carry_over_problems(stripped, plan=w.this, read_text=w.files.get, local_read_text=local.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        # a proof that reads around the reader binds nothing and is refused
        rec2 = co.evaluate_carry_over(this_plan=w.this, this_sha=F.W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                      git=w.git, readiness_problems=F.fake_readiness, local_read_text=local.get,
                                      petri_proof=lambda **kw: co.petri_sidecar_proof([cell], local.get))
        assert rec2["n_carried"] == 0 and "cannot be bound" in rec2["refused"]["petri__s1__base__shard00"]["why"]

    def test_the_served_default_proof_reads_only_through_the_handed_reader(self, tmp_path):
        import importlib.util
        m = types.ModuleType("modal")
        m.App, m.Dict = F._App, F._FakeDict
        m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: F._Chain())
        m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: F._Chain(), debian_slim=lambda *a, **k: F._Chain())
        m.is_local = lambda: False
        m.current_function_call_id = lambda: "fc-test"
        m.forward = None
        m.FunctionCall = None
        spec = importlib.util.spec_from_file_location("seedrep_served_fj5", REPO / "scripts" / "modal" / "seedrep_served.py")
        srv = importlib.util.module_from_spec(spec)
        prior_mod = sys.modules.get("modal")
        sys.modules["modal"] = m
        try:
            spec.loader.exec_module(srv)
        finally:
            if prior_mod is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = prior_mod
        import seedrep_petri_drive as drive
        w = _petri_carry_world(tmp_path, with_bundle=True)
        home = tmp_path / "home"
        cell = f"{drive.out_root_for(home, F.W_PREV[:7], F.WAVE.replace('-', '')[:8], F.AID, drive.run_tag('s1', 1))}/cells/petri__none__s1"
        local = _local_store(cell)
        reads = []
        acc = {}
        out = srv.carry_over_evaluate(w.this, rs=tmp_path / "rs", sha=F.W_THIS, sources=[w.prior["path"]], accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                      read_text=w.files.get, git=w.git, accepted_put=lambda k, v, skip_if_exists=False: acc.setdefault(k, v) == v, accepted_get=acc.get,
                                      readiness=F.fake_readiness, petri_read_text=lambda p: (reads.append(p), local.get(p))[1], petri_home=home,
                                      reaudit=False)                          # this test is about the Petri proof's reader; the re-audit has its own tests
        assert out["n_carried"] == 1, out["refused"]
        rec = co.read_carry_over(tmp_path / "rs")
        inv = rec["carried"]["petri__s1__base__shard00"]["evidence_inventory_local"]
        assert set(inv) == set(reads) == {f"{cell}/results.json", f"{cell}/{co.PETRI_REFUSED_LEDGER}", f"{cell}/{co.PETRI_JUDGE_REFUSED_LEDGER}"}
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, local_read_text=local.get, git=w.git, accepted_get_for_dict=_agd(w.dicts), petri_home=home) == []
        assert any("no longer present" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, local_read_text=lambda p: None, git=w.git, accepted_get_for_dict=_agd(w.dicts), petri_home=home))


# =========================================================================== the identity table admits a VALIDATED carry

class TestIdentityTableCarries:
    def _mixed_leg(self, tmp_path):
        """A pasttense leg whose s1 base engine is CARRIED from W_PRIOR beside three fresh engines of this wave."""
        stores = tmp_path / "stores"
        roots = fx.build_wave(stores, legs=("pasttense",), views=("s1", "s2", "s3"), arms=("none",), n_items=1, arditi=True, arditi_pin=PIN)
        _rekey_root_to_history(roots[("pasttense", "s1")], W=W_PRIOR, wave=WAVE_PRIOR)
        pins = json.loads(fx.pins_file(tmp_path / "pins.json", pin=PIN).read_text())
        return stores, roots, pins

    def test_a_leg_mixing_one_carried_and_three_fresh_engines_passes_with_the_mapping_and_fails_without(self, tmp_path):
        stores, roots, pins = self._mixed_leg(tmp_path)
        carried_aid = roots[("pasttense", "s1")].attempt_id
        discovered = ac.discover_attempt_roots(stores)
        # KNOWN-BAD (the assembly tip): without the mapping the leg FAILs on tree identity
        _rows, legs = eit.build(discovered, pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None)
        assert legs["pasttense"]["verdict"] == "FAIL" and any("W differs across base engines" in p for p in legs["pasttense"]["problems"])
        # with the VALIDATED mapping the carried engine is held to ITS source W, the cross-engine requirement is engine version / config
        rows, legs = eit.build(discovered, pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=fx.W_FIXTURE, carried={carried_aid: W_PRIOR})
        assert legs["pasttense"]["verdict"] == "PASS", legs["pasttense"]["problems"]
        row = next(r for r in rows if r["attempt_id"] == carried_aid)
        assert row["W"] == W_PRIOR and row["carried_from_W"] == W_PRIOR and row["problems"] == []
        assert all(r["carried_from_W"] == eit.NOT_CARRIED and r["W"] == fx.W_FIXTURE for r in rows if r["attempt_id"] != carried_aid)
        assert legs["pasttense"]["identity"]["W"] == fx.W_FIXTURE and legs["pasttense"]["identity"]["carried_from_W"] == [W_PRIOR] and legs["pasttense"]["n_carried"] == 1
        assert legs["pasttense"]["identity"]["model_revision"] == rows[0]["model_revision"]
        assert "carried_from_W" in eit.COLUMNS and set(rows[0]) >= set(eit.COLUMNS)
        # a carried root whose recorded W is NOT the mapping's source W fails by name (the pulled root is not the carried one)
        _rows, legs = eit.build(discovered, pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None, carried={carried_aid: "b" * 40})
        assert legs["pasttense"]["verdict"] == "FAIL" and any("carried standup sha" in p and "carry source W" in p for p in legs["pasttense"]["problems"])
        # a root the mapping does NOT name still fails on a differing W: re-key s2 too, map only s1
        _rekey_root_to_history(roots[("pasttense", "s2")], W="c" * 40, wave=WAVE_PRIOR)
        _rows, legs = eit.build(ac.discover_attempt_roots(stores), pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None, carried={carried_aid: W_PRIOR})
        assert legs["pasttense"]["verdict"] == "FAIL" and any("W differs across base engines not named by the carry mapping" in p for p in legs["pasttense"]["problems"])

    def test_the_cli_takes_the_carry_record_and_the_plan_free_validator_refuses_a_malformed_one(self, tmp_path):
        stores, roots, _pins = self._mixed_leg(tmp_path)
        prod = roots[("pasttense", "s1")]
        rec = _carry_record([(prod, W_PRIOR, WAVE_PRIOR, "production")])
        rec_path = tmp_path / "carry_over.json"
        rec_path.write_text(json.dumps(rec))
        assert ac.carry_source_ws(rec) == {prod.attempt_id: W_PRIOR}
        pins = fx.pins_file(tmp_path / "pins.json", pin=PIN)
        out = tmp_path / "tables" / "engine_identity.csv"
        assert ac.run_main(eit.main, ["--stores", str(stores), "--out", str(out), "--arditi-pins", str(pins), "--W", fx.W_FIXTURE]) == ac.EXIT_FAILED
        assert ac.run_main(eit.main, ["--stores", str(stores), "--out", str(out), "--arditi-pins", str(pins), "--W", fx.W_FIXTURE, "--carry-over", str(rec_path)]) == 0
        legs = json.loads((tmp_path / "tables" / "engine_identity_legs.json").read_text())
        assert legs["pasttense"]["verdict"] == "PASS" and legs["pasttense"]["identity"]["carried_from_W"] == [W_PRIOR]
        sid = "pasttense__s1__base__shard00"
        for mutate, needle in ((lambda r: r["carried"][sid]["clauses"].pop("complete"), "clause"),
                               (lambda r: r["carried"][sid]["carried_from"].update(W=fx.W_FIXTURE), "prior tree"),
                               (lambda r: r["carried"][sid].pop("carried_from"), "provenance"),
                               (lambda r: r.update(schema="x"), "schema")):
            bad = copy.deepcopy(rec)
            mutate(bad)
            with pytest.raises(ac.AuditRefusal, match=needle):
                ac.carry_source_ws(bad)


    def test_all_bases_carried_fresh_rungs_are_still_held_to_one_tree(self, tmp_path):
        """fj6-3: with EVERY base carried there is no fresh base to anchor W, so the rungs of this wave must anchor each other --
        same-W fresh rungs PASS, differing-W fresh rungs FAIL (without --W), a carried rung is held to its own source W."""
        stores = tmp_path / "stores"
        roots = fx.build_wave(stores, legs=("pasttense",), views=("s1", "s2", "s3"), arms=("none",), n_items=1, arditi=True, arditi_pin=PIN)
        for r in roots.values():
            _rekey_root_to_history(r, W=W_PRIOR, wave=WAVE_PRIOR)
        carried = {r.attempt_id: W_PRIOR for r in roots.values()}
        pins = json.loads(fx.pins_file(tmp_path / "pins.json", pin=PIN).read_text())
        rung1 = fx.make_root(stores, leg="pasttense", view="s1", arms=("none",), stage="8192")
        rung1.flush_pipeline()
        rung2 = fx.make_root(stores, leg="pasttense", view="s2", arms=("none",), stage="8192")
        rung2.flush_pipeline()
        # same-W fresh rungs (this wave's fixture W) beside four carried bases -> PASS; the leg's W is the fresh engines'
        rows, legs = eit.build(ac.discover_attempt_roots(stores), pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None, carried=carried)
        assert legs["pasttense"]["verdict"] == "PASS", legs["pasttense"]["problems"]
        assert legs["pasttense"]["n_carried"] == 4 and legs["pasttense"]["identity"]["carried_from_W"] == [W_PRIOR]
        # KNOWN-BAD (fj6-2): a fresh rung from another tree passed because no fresh base anchored w_ref -> now FAILS by name, without --W
        _rekey_root_to_history(rung2, W="c" * 40, wave=fx.WAVE_UUID)
        rows, legs = eit.build(ac.discover_attempt_roots(stores), pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None, carried=carried)
        assert legs["pasttense"]["verdict"] == "FAIL"
        assert any("rung s2/8192/shard00: W" in p and "the uncarried engines'" in p for p in legs["pasttense"]["problems"]), legs["pasttense"]["problems"]
        # a CARRIED rung is held to its own source W, not to the fresh rungs'
        rows, legs = eit.build(ac.discover_attempt_roots(stores), pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None, carried={**carried, rung2.attempt_id: "c" * 40})
        assert legs["pasttense"]["verdict"] == "PASS", legs["pasttense"]["problems"]
        rows, legs = eit.build(ac.discover_attempt_roots(stores), pins=pins, pin_key=eit.DEFAULT_PIN_KEY, W_expected=None, carried={**carried, rung2.attempt_id: "d" * 40})
        assert legs["pasttense"]["verdict"] == "FAIL" and any("carried standup sha" in p for p in legs["pasttense"]["problems"])


# =========================================================================== fj7: re-audit under THIS tree's reader (A) + the carry_refused_hold belt (B)

FITTED = "constv2-mix-full__psr_ce_all"


def _w22_shaped_world(tmp_path, *, strip_arm_binding=True):
    """A completed pasttense row with TWO children -- `none` (a verdict written BEFORE F-AN: no `arm_binding`) and a fitted cell
    (NO verdict at all: the old audit refused it) -- as the genuine pre-F-AN schema left it."""
    def two_children(W, W7):
        r = F._row(W, W7)
        k0 = r["children"][0]
        r["children"] = [k0, {**copy.deepcopy(k0), "arm": FITTED, "cell": FITTED}]
        r["children"][0]["cell"] = "none"
        return r                       # the route set stays the fixture's (`none`): clause (d)'s banked-hash check is not what this world exercises
    prev = F._plan(F.W_PREV, materialisation="final", rows=[two_children(F.W_PREV, F.W_PREV[:7])])
    this = F._plan(F.W_THIS, materialisation="final", rows=[two_children(F.W_THIS, F.W_THIS[:7])])
    files, dicts = {}, {}
    ident = {"W": F.W_PREV, "W7": F.W_PREV[:7], "wave_uuid": F.WAVE, "results_root": prev["results_root"]}
    root = F._root_files(files, plan=prev, row=prev["standups"][0], identity=ident, n_children=2)
    prow = prev["standups"][0]
    k0 = prow["children"][0]
    rel0 = lp.audit_verdict_rel(k0["eval_leg"], k0["arm"], k0["s"], stage=prow["stage"], shard=prow["shard"])
    if strip_arm_binding:
        doc = json.loads(files[f"{root}/{rel0}"])
        doc.pop("arm_binding", None)
        text = json.dumps(doc)
        files[f"{root}/{rel0}"] = text
        ev = json.loads(files[f"{root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][rel0]["sha256"] = F._sha(text)        # the OLD producer bound the pre-F-AN bytes
        files[f"{root}/modal_evidence.json"] = json.dumps(ev)
    dicts[prev["dicts"]["accepted"]] = {prow["accepted_key"]: F.AID}
    git = F.FakeGit({F.W_PREV: F._blobs(F.W_PREV), F.W_THIS: F._blobs(F.W_THIS)})
    prior = co.load_prior_source(F._prior_dir(tmp_path, prev))
    return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, git=git, prior=prior, root=root, rel0=rel0)


def _stub_reaudit(w):
    """A re-audit double in the shape `seedrep_served.reaudit_carried_root` returns: for every (child, table leg) a FRESH verdict with THIS tree's
    W, the store's tree under store_W and an arm_binding, filed under the carried entry's own evidence root; bindings keyed by the plan's rel."""
    calls = []

    def reaudit(*, prev_row, this_row, prior, aid, root, this_plan):
        calls.append({"aid": aid, "root": root})
        ev_root = f"/vol/seedrep/{this_plan['W7']}/{this_plan['wave_uuid']}/carried/{this_row['standup_id']}__{aid}/evidence"
        base = json.loads(w.files[f"{w.root}/{w.rel0}"])
        bindings = {}
        for k in this_row["children"]:
            for leg, plan_rel in lp.audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=this_row["stage"], shard=this_row["shard"]):
                cell = str(k.get("cell") or k["arm"])
                harness = lp.harness_id(cell)
                doc = {**base, "leg": leg, "arm": cell, "s": k["s"], "W": this_plan["W"], "store_W": prior["W"], "audited_at": this_plan["W"], "attempt_id": aid,
                       "store_dir": f"{root}/exp/transcripts_results/{cell}",
                       "arm_binding": {"arm": cell, "harness_id": harness, "traintest_id": cell, "vmodel": f"petri__{harness}", "spelling": "traintest"},
                       "verdict_file": f"request_log/{leg}/{cell}__s{k['s']}.json"}
                text = json.dumps(doc)
                bound_rel = f"{lp.AUDIT_OUT_REL}/request_log/{leg}/{cell}__s{k['s']}.json"
                w.files[f"{ev_root}/{bound_rel}"] = text
                bindings[plan_rel] = {"sha256": co._sha_text(text), "writer": lp.PRODUCER_WRITERS["request_evidence_audit"], "tree_sha": this_plan["W"], "rc": 0,
                                      "bound_rel": bound_rel, "plan_arm": k["arm"], "plan_cell": cell}
        return {"reaudited_by": this_plan["W"], "evidence_root": ev_root, "store_root": root, "producer_evidence": bindings, "rc": 0,
                "argv": ["python", "request_evidence_audit.py", "--audited-at", this_plan["W"]], "tool": lp.PRODUCER_WRITERS["request_evidence_audit"], "n_files": len(bindings)}
    reaudit.calls = calls
    return reaudit


def _eval(w, **kw):
    return co.evaluate_carry_over(this_plan=w.this, this_sha=F.W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                  git=w.git, readiness_problems=F.fake_readiness, **kw)


class TestReauditUnderThisTree:
    def test_before_the_pre_fan_schema_refuses_clause_c_and_the_row_would_be_spawned(self, tmp_path, srv):
        w = _w22_shaped_world(tmp_path)
        rec = _eval(w)
        r = rec["refused"][F.SID]
        assert r["clause"] == "complete" and "arm_binding" in r["why"], r["why"]      # the OLD verdict, read as it is, fails fp3's strict check
        assert r["attempts"][-1]["attempt_id"] == F.AID                                # ... although a COMPLETED, accepted prior attempt exists
        # KNOWN-BAD: with nothing carried the row is a spawn candidate -- canonical completed work regenerated
        picked, _ = srv.select_launch_rows(w.this, go=F.SID, carried={})
        assert [x["standup_id"] for x in picked] == [F.SID]
        # the fitted-cell root has NO verdict at all: with the none verdict intact the refusal names the fitted child's absent verdict
        w2 = _w22_shaped_world(tmp_path / "b", strip_arm_binding=False)
        r2 = _eval(w2)["refused"][F.SID]
        assert r2["clause"] == "complete" and ("bound no" in r2["why"] or "absent" in r2["why"]), r2["why"]

    def test_after_a_reaudit_under_this_tree_the_row_carries_with_fresh_bound_verdicts(self, tmp_path):
        w = _w22_shaped_world(tmp_path)
        reaudit = _stub_reaudit(w)
        rec = _eval(w, reaudit=reaudit)
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][F.SID]
        assert reaudit.calls == [{"aid": F.AID, "root": w.root}]
        assert e["reaudited_by"] == F.W_THIS and e["reaudit_evidence_dir"].startswith(f"/vol/seedrep/{F.W_THIS[:7]}/{F.WAVE}/carried/")
        assert e["reaudit"]["original"] == {"attempt_id": F.AID, "attempt_root": w.root, "W": F.W_PREV, "wave_uuid": F.WAVE} and e["reaudit"]["store_root"] == w.root
        chk = e["clauses"]["complete"]["checks"]["request_evidence_audit"]
        assert chk["reaudit"]["reaudited_by"] == F.W_THIS and chk["reaudit"]["held_to_plan"] == "this"
        assert sorted(s["plan_cell"] for s in chk["sources"]) == sorted(["none", FITTED])                # BOTH children bound, the fitted one too
        assert all(s["arm_binding"]["traintest_id"] == s["plan_cell"] and s["producer"]["tree_sha"] == F.W_THIS for s in chk["sources"])
        assert all(p.startswith(e["reaudit_evidence_dir"]) for p in (s["path"] for s in chk["sources"]))   # read from the entry's OWN evidence dir
        assert not any(p.startswith(w.root + "/audits") for p in e["evidence_inventory"] if p.endswith(".json") and "request_log" in p)
        assert all(f"{e['reaudit_evidence_dir']}/{b['bound_rel']}" in e["evidence_inventory"] for b in e["reaudit"]["producer_evidence"].values())
        # the historical root was never written: its old verdict is untouched
        assert "arm_binding" not in json.loads(w.files[f"{w.root}/{w.rel0}"])
        # the certificate accepts, and refuses a record whose re-audit is not this tree's
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        stale = copy.deepcopy(rec)
        stale["carried"][F.SID]["reaudited_by"] = stale["carried"][F.SID]["reaudit"]["reaudited_by"] = F.W_PREV
        assert any("reaudited_by" in p for p in co.carry_over_problems(stale, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        gone = dict(w.files)
        for p in list(gone):
            if p.startswith(e["reaudit_evidence_dir"]):
                gone.pop(p)
        assert any("no longer on the volume" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=gone.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        # a re-audit that exits non-zero, or raises, refuses the row by name -- never a fall-back to the stale verdicts
        bad = _stub_reaudit(w)
        rec_bad = _eval(w, reaudit=lambda **kw: {**bad(**kw), "rc": 1})
        assert rec_bad["n_carried"] == 0 and "rc 1" in rec_bad["refused"][F.SID]["why"]
        rec_exc = _eval(w, reaudit=lambda **kw: (_ for _ in ()).throw(RuntimeError("modal volume get failed")))
        assert rec_exc["n_carried"] == 0 and "re-audit under this tree failed" in rec_exc["refused"][F.SID]["why"]

    def test_with_the_reaudit_disabled_the_refused_completed_row_is_held_never_spawned(self, tmp_path, srv, capsys):
        w = _w22_shaped_world(tmp_path)
        rec = _eval(w)                                                    # (A) disabled: refused on the schema
        holds = co.carry_refused_holds(rec)
        assert set(holds) == {F.SID} and holds[F.SID]["reason"] == co.CARRY_REFUSED_HOLD and holds[F.SID]["attempt_id"] == F.AID and holds[F.SID]["clause"] == "complete"
        with pytest.raises(srv.LaunchRefused, match="every selected row is HELD"):
            srv.select_launch_rows(w.this, go=F.SID, carry_holds=holds)
        assert f"HOLD {co.CARRY_REFUSED_HOLD} {F.SID}" in capsys.readouterr().out
        other = F._row(F.W_THIS, F.W_THIS[:7], sid="codeattack__s1__base__shard00", leg="codeattack")
        plan2 = F._plan(F.W_THIS, materialisation="final", rows=[w.this["standups"][0], other])
        picked, _ = srv.select_launch_rows(plan2, legs="pasttense,codeattack", carry_holds=holds)
        assert [x["standup_id"] for x in picked] == ["codeattack__s1__base__shard00"]      # the held row is not a spawn candidate; others spawn
        # a refusal with NO accepted attempt anywhere is not a hold (nothing completed exists to protect)
        w3 = _w22_shaped_world(tmp_path / "c")
        w3.dicts.clear()
        assert co.carry_refused_holds(_eval(w3)) == {}
        # the steward's view: the held row is `held` with the carry's reason
        import seedrep_steward as st
        row = w.this["standups"][0]
        tick = st.steward_tick(plan=w.this, calls=[], snapshot={}, holds={F.SID: {"held": True, "reason": co.CARRY_REFUSED_HOLD, **holds[F.SID]}}, state=st.StewardState(),
                               required={row["accepted_key"]: F.SID})
        assert tick["stop"] == "hold" and tick["held"] == [F.SID] and tick["invoke"] == []


class TestReauditCarriedRootRealTool:
    def test_the_w23_audit_runs_over_a_pulled_historical_root_into_the_entrys_own_evidence_dir(self, tmp_path, srv):
        import shutil
        arms = ("none", FITTED)
        root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=arms)
        fx.populate_root(root, s=1, arms=arms, n_items=2)
        this_sha = "b" * 40
        this_row = {"standup_id": "pasttense__s1__base__shard00", "leg": "pasttense", "engine": "s1", "stage": "base", "shard": 0,
                    "children": [{"eval_leg": "pasttense", "arm": a, "cell": a, "s": 1} for a in arms]}
        hist_root = f"/vol/seedrep/{fx.W_FIXTURE[:7]}/exp/seedrep_battery_modal/pasttense/s1/shard00/attempts/{root.attempt_id}"
        uploaded = {}

        def run_get(src, out):
            assert src == hist_root
            shutil.copytree(root.path, Path(out) / root.attempt_id)
        ra = srv.reaudit_carried_root(prev_row=this_row, this_row=this_row, prior={"W": fx.W_FIXTURE}, aid=root.attempt_id, root=hist_root,
                                      this_plan={"W": this_sha, "W7": this_sha[:7], "wave_uuid": F.WAVE}, rs=tmp_path / "rs", this_sha=this_sha,
                                      tree=REPO, run_get=run_get, put_file=lambda rel, data: uploaded.__setitem__(rel, data))
        assert "--plan" not in ra["argv"] and "--audited-at" in ra["argv"]          # the app's own audit argv + the re-audit's tree stamp
        assert ra["rc"] == 0, Path(ra["log"]).read_text()[-2000:]
        assert ra["reaudited_by"] == this_sha and ra["evidence_root"] == f"/vol/seedrep/{this_sha[:7]}/{F.WAVE}/carried/pasttense__s1__base__shard00__{root.attempt_id}/evidence"
        all_bindings = ra["producer_evidence"]
        # fc11c: the re-audit also RE-DERIVES the sampler verdict under this tree (`rederive_sampler_verdict`): two more bindings beside the audit's verdicts
        sampler_b = {k: all_bindings[k] for k in (lp.RESOLVED_SAMPLER_REL, lp.EXPECTED_SAMPLERS_REL)}
        assert all(b["tree_sha"] == this_sha for b in sampler_b.values()) and sampler_b[lp.RESOLVED_SAMPLER_REL]["writer"] == srv.PRODUCER_RECORDER_REL \
            and sampler_b[lp.EXPECTED_SAMPLERS_REL]["writer"] == srv.EXPECTED_SAMPLERS_HELPER_REL, sampler_b
        assert ra["sampler"]["rederived_by"] == this_sha and set(ra["sampler"]) >= {"historical_verdict", "rederived_verdict", "recorder_rc", "expected_samplers_rc"}, ra["sampler"]
        bindings = {k: v for k, v in all_bindings.items() if k not in sampler_b}
        assert len(bindings) == 2 and all(not b.get("absent") and b["tree_sha"] == this_sha and b["rc"] == 0 for b in bindings.values()), bindings
        for plan_rel, b in bindings.items():
            doc = json.loads(uploaded[f"{ra['evidence_root'][len('/vol/'):]}/{b['bound_rel']}"])
            assert doc["W"] == this_sha and doc["store_W"] == fx.W_FIXTURE and doc["audited_at"] == this_sha and doc["attempt_id"] == root.attempt_id
            assert doc["arm_binding"]["traintest_id"] == b["plan_cell"] and doc["verdict"] == "AGREES"
            assert str(doc["store_dir"]).startswith(hist_root[len("/vol/"):])                    # the stores are the historical root's (relative spelling)
        rel_ev = ra["evidence_root"][len("/vol/"):]
        assert f"{rel_ev}/{lp.AUDIT_OUT_REL}/{srv.AUDIT_INDEX_NAME}" in uploaded and f"{rel_ev}/{srv.SERVE_DIR}/endpoint.json" in uploaded
        # the historical root is untouched: no audits/ tree was written under it
        assert not (root.path / lp.AUDIT_OUT_REL / "request_log").exists()


# =========================================================================== fj8: the hold stands whichever clause refused

class TestPriorAcceptanceHolds:
    def _world(self, tmp_path):
        return _w22_shaped_world(tmp_path, strip_arm_binding=False)

    def test_a_timeout_only_row_identity_refusal_over_a_completed_attempt_is_held_never_spawned(self, tmp_path, srv, capsys):
        w = self._world(tmp_path)
        w.this["standups"][0]["modal"]["timeout_s"] = 57660                    # the repro: the ONLY difference; same objects / evals / children
        w.this["launch_plan_sha"] = lp.plan_sha(w.this)
        rec = _eval(w, reaudit=_stub_reaudit(w))
        r = rec["refused"][F.SID]
        assert r["clause"] == "row_identity" and "timeout_s" in r["why"]        # the row-identity assertion is untouched
        paa = r["attempts"][-1]["prior_accepted_attempt"]
        assert paa["attempt_id"] == F.AID and paa["attempt_root"] == w.root and paa["prior_W"] == F.W_PREV and paa["source"] == "production"
        holds = co.carry_refused_holds(rec)
        assert holds[F.SID]["attempt_id"] == F.AID and holds[F.SID]["attempt_root"] == w.root and holds[F.SID]["clause"] == "row_identity"
        with pytest.raises(srv.LaunchRefused, match="every selected row is HELD"):
            srv.select_launch_rows(w.this, go=F.SID, carry_holds=holds)
        assert f"HOLD {co.CARRY_REFUSED_HOLD} {F.SID}" in capsys.readouterr().out
        import seedrep_steward as st
        row = w.this["standups"][0]
        tick = st.steward_tick(plan=w.this, calls=[], snapshot={}, holds={F.SID: {"held": True, **holds[F.SID]}}, state=st.StewardState(), required={row["accepted_key"]: F.SID})
        assert tick["stop"] == "hold" and tick["held"] == [F.SID]
        # KNOWN-BAD at fj7: the hold keyed on `attempt_id`, which a row_identity refusal never set -> the completed row was a spawn candidate
        legacy = copy.deepcopy(rec)
        for a in legacy["refused"][F.SID]["attempts"]:
            a.pop("prior_accepted_attempt", None)
            a.pop("attempt_id", None)
        assert co.carry_refused_holds(legacy) == {}
        picked, _ = srv.select_launch_rows(w.this, go=F.SID, carry_holds=co.carry_refused_holds(legacy))
        assert [x["standup_id"] for x in picked] == [F.SID]

    def test_a_refusal_at_each_of_the_five_clauses_over_an_accepted_prior_attempt_is_held(self, tmp_path):
        # (a) row identity
        w = self._world(tmp_path / "a")
        w.this["standups"][0]["modal"]["timeout_s"] = 57660
        w.this["launch_plan_sha"] = lp.plan_sha(w.this)
        rec = _eval(w, reaudit=_stub_reaudit(w))
        assert rec["refused"][F.SID]["clause"] == "row_identity" and F.SID in co.carry_refused_holds(rec)
        # (b) producer set: the driver differs between the trees
        w = self._world(tmp_path / "b")
        w.git.blobs[F.W_THIS][F.CHILD] = "driver-v2"
        rec = _eval(w, reaudit=_stub_reaudit(w))
        assert rec["refused"][F.SID]["clause"] == "producer_set" and F.SID in co.carry_refused_holds(rec)
        # (c) complete: the root's RESULT is not DONE (the Dict still names the attempt: the belt protects it, the carry refuses it)
        w = self._world(tmp_path / "c")
        w.files[f"{w.root}/RESULT"] = w.files[f"{w.root}/RESULT"].replace("status=DONE rc=0", "status=DRIVERS_FAILED rc=1")
        rec = _eval(w, reaudit=_stub_reaudit(w))
        assert rec["refused"][F.SID]["clause"] == "complete" and F.SID in co.carry_refused_holds(rec)
        # (d) engine identity: this plan's row expects another geometry
        w = self._world(tmp_path / "d")
        w.this["standups"][0]["readiness"]["attested_max_model_len"] = 12288
        w.prev["standups"][0]["readiness"]["attested_max_model_len"] = 12288
        for plan in (w.this, w.prev):
            plan["launch_plan_sha"] = lp.plan_sha(plan)
        w.prior = co.load_prior_source(F._prior_dir(tmp_path / "d", w.prev, name="prior_d"))
        rec = _eval(w, reaudit=_stub_reaudit(w))
        assert rec["refused"][F.SID]["clause"] == "engine_identity" and F.SID in co.carry_refused_holds(rec)
        # (e) ladder queue: a producer row whose rung is in this plan, evaluated without a replay
        ladder = {"kind": "static", "cap_next": 8192, "shard": "0", "shards": ["0"], "feeder_shards": ["0"], "producer_of": "8192"}
        rung = F._row(F.W_THIS, F.W_THIS[:7], sid="pasttense__s1__8192__shard00", stage="8192", kind="rung")
        prev = F._plan(F.W_PREV, materialisation="final", rows=[F._row(F.W_PREV, F.W_PREV[:7], ladder=ladder)])
        this = F._plan(F.W_THIS, materialisation="final", rows=[F._row(F.W_THIS, F.W_THIS[:7], ladder=ladder)], rungs=[rung])
        files, dicts = {}, {}
        ident = {"W": F.W_PREV, "W7": F.W_PREV[:7], "wave_uuid": F.WAVE, "results_root": prev["results_root"]}
        F._root_files(files, plan=prev, row=prev["standups"][0], identity=ident)
        dicts[prev["dicts"]["accepted"]] = {prev["standups"][0]["accepted_key"]: F.AID}
        rec = co.evaluate_carry_over(this_plan=this, this_sha=F.W_THIS, priors=[co.load_prior_source(F._prior_dir(tmp_path / "e", prev))], read_text=files.get,
                                     accepted_get_for_dict=lambda n: dicts.get(n, {}).get, git=F.FakeGit({F.W_PREV: F._blobs(F.W_PREV), F.W_THIS: F._blobs(F.W_THIS)}),
                                     readiness_problems=F.fake_readiness)
        assert rec["refused"][F.SID]["clause"] == "ladder_queue" and F.SID in co.carry_refused_holds(rec)
        assert co.carry_refused_holds(rec)[F.SID]["attempt_id"] == F.AID

    def test_a_refusal_with_no_prior_accepted_attempt_is_not_held_and_spawns(self, tmp_path, srv):
        w = self._world(tmp_path)
        w.this["standups"][0]["modal"]["timeout_s"] = 57660
        w.this["launch_plan_sha"] = lp.plan_sha(w.this)
        w.dicts.clear()                                                        # nothing accepted anywhere: nothing completed to protect
        rec = _eval(w, reaudit=_stub_reaudit(w))
        assert rec["refused"][F.SID]["clause"] == "row_identity" and "prior_accepted_attempt" not in rec["refused"][F.SID]["attempts"][-1]
        assert co.carry_refused_holds(rec) == {}
        picked, _ = srv.select_launch_rows(w.this, go=F.SID, carry_holds=co.carry_refused_holds(rec))
        assert [x["standup_id"] for x in picked] == [F.SID]
