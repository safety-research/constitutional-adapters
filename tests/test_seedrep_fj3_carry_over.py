"""The two gaps that made the carry-over admit almost nothing from the prior wave, plus two further defects in the admission.

Known-bad at 8024980d2 (fj2 merged): (1) clause (e) refused every base row whose rung consumer is in the plan -- the carried
row's at-cap queue lives in the prior wave's ladder root, which this wave's `rung_redraw.py consume` never reads -- and no replay existed; (2) a
Petri row was refused for want of a sidecar proof, and the D111 code removals in the drive / auditor module were not an admitted difference;
(3) a carried acceptance never reached the steward's completion accounting (coverage counted CALLS only) nor `pull_leg` (the root was derived
under the CURRENT results root); (4) the certificate re-validated only RESULT / COMPLETE.json / the attempt id -- a determinism source or READY
record removed after evaluation still certified. Every case here runs on SYNTHETIC two-tree plans and roots, the real `seedrep_ladder` protocol
over its in-memory volume / Dict doubles, a synthetic git, a stub readiness contract -- no Modal, no real RS, no credentials.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _seedrep_testlib as T  # noqa: E402
import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_fj2_carry_over as F  # noqa: E402  -- fj2's synthetic world: plans, roots, FakeGit, the readiness stub

L = T.load_ladder()
RR = T.load_rung_redraw()

W_PREV, W_THIS, WAVE, AID = F.W_PREV, F.W_THIS, F.WAVE, F.AID
WAVE_PREV = "33333333-4444-4555-8666-777777777777"                 # the PRIOR wave (a prior tree's production wave is its own uuid)
LADDER = {"kind": "static", "cap_next": 8192, "shard": "0", "shards": ["0"], "feeder_shards": ["0"], "producer_of": "8192"}
RUNG_SID = "pasttense__s1__8192__shard00"


# --------------------------------------------------------------------------- the synthetic prior wave: a real Producer over the fake volume

def _prior_wave(shared, accepted_prev, *, n=3, aid=AID, W7=W_PREV[:7], wave=WAVE_PREV):
    """The prior wave's base producer run to its CLOSED marker on a SharedVolume: N at-cap static rows enqueued through the real protocol
    (attempt-unique files, inventory, acceptance, marker) plus COMPLETE.json at the attempt root. Returns (spec, root, records)."""
    spec = L.QueueSpec.build(W7, wave, "pasttense", "s1", "static", 8192, ["0"])
    ref = spec.producers[0]
    root = L.producer_attempt_root(ref, aid)
    vol = shared.snapshot(aid)
    prod = L.Producer(vol, spec, ref, aid, root)
    recs = []
    for i in range(n):
        rec = L.static_queue_record(spec, arm="none", s=1, family="pasttense", row=T.static_row(f"g{i}", 1, seed=7 + i),
                                    base_store=f"{root}/exp/private/run/shard00/responses.jsonl", producer=ref, producer_attempt_id=aid,
                                    vmodel="petri__base", sampling=T.PRESET)
        prod.enqueue(rec)
        recs.append(rec)
    vol.write_bytes(L.complete_rel(root), json.dumps({"attempt_id": aid, "n_rows": 7}).encode())
    vol.commit()
    assert L.finish_producer(prod, accepted_prev)["state"] == "accepted"
    assert L.reconcile_queue(vol, accepted_prev, spec)["queue_closed"]
    return spec, root, recs


def _plans():
    prev_row = F._row(W_PREV, W_PREV[:7], ladder=LADDER)
    this_row = F._row(W_THIS, W_THIS[:7], ladder=LADDER)
    rung_this = F._row(W_THIS, W_THIS[:7], sid=RUNG_SID, stage="8192", kind="rung")
    prev = F._plan(W_PREV, materialisation="final", rows=[prev_row])
    prev["wave_uuid"] = WAVE_PREV
    for r in prev["standups"]:
        r["control_key"] = lp.control_key(prev["W7"], WAVE_PREV, r["leg"], r["engine"], r["stage"], r["shard"])
        r["accepted_key"] = lp.accepted_key(prev["W7"], WAVE_PREV, r["leg"], r["engine"], r["stage"], r["shard"])
    prev["dicts"] = {p: lp.dict_name(prev["W7"], WAVE_PREV, p) for p in lp.DICT_PURPOSES}
    prev["launch_plan_sha"] = lp.plan_sha(prev)
    this = F._plan(W_THIS, materialisation="final", rows=[this_row], rungs=[rung_this])
    return prev, this


@pytest.fixture
def ladder_world(tmp_path):
    prev, this = _plans()
    shared = L.SharedVolume()
    accepted_prev = L.MemoryKV(prev["dicts"]["accepted"])
    accepted_this = L.MemoryKV(this["dicts"]["accepted"])
    spec_prev, root_rel, recs = _prior_wave(shared, accepted_prev)
    files, dicts = {}, {}
    prow = prev["standups"][0]
    ident = {"W": W_PREV, "W7": prev["W7"], "wave_uuid": WAVE_PREV, "results_root": prev["results_root"]}
    root_abs = F._root_files(files, plan=prev, row=prow, identity=ident)
    assert root_abs == f"{lp.VOL_PATH}/{root_rel}", (root_abs, root_rel)         # the plan's root spelling IS the protocol's
    files[f"{root_abs}/ladder_enqueued.json"] = json.dumps({"schema": "seedrep_ladder_enqueued/1", "attempt_id": AID, "complete": True, "n": len(recs),
                                                            "entries": [{"item_key": r["item_key"], "path": "p", "sha256": "s"} for r in recs], "streams": [{"rows_read": 5}]})
    dicts[prev["dicts"]["accepted"]] = {prow["accepted_key"]: AID}
    git = F.FakeGit({W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS)})
    prior = co.load_prior_source(F._prior_dir(tmp_path, prev))
    view = shared.read_only_view()
    return types.SimpleNamespace(prev=prev, this=this, shared=shared, view=view, accepted_prev=accepted_prev, accepted_this=accepted_this, spec_prev=spec_prev,
                                 root_rel=root_rel, recs=recs, files=files, dicts=dicts, git=git, prior=prior, tmp=tmp_path)


def _replay_for(w):
    def replay(**kw):
        return co.replay_ladder_queue(L=L, vol=w.view, accepted_kv=w.accepted_this, **kw)
    return replay


def _evaluate(w, **kw):
    return co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get,
                                  accepted_get_for_dict=lambda name: w.dicts.get(name, {}).get, git=w.git, readiness_problems=F.fake_readiness, **kw)


# --------------------------------------------------------------------------- (1) the ladder-queue replay

class TestLadderQueueReplay:
    def test_pin_without_a_replay_the_producer_row_is_refused_by_clause_e(self, ladder_world):
        rec = _evaluate(ladder_world)
        assert rec["refused"][F.SID]["clause"] == "ladder_queue" and "REPLAY" in rec["refused"][F.SID]["why"]

    def test_replay_rekeys_n_records_closes_the_queue_and_the_real_consumer_read_path_sees_them(self, ladder_world):
        w = ladder_world
        rec = _evaluate(w, replay=_replay_for(w))
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][F.SID]
        rp = e["clauses"]["ladder_queue"]["replay"]
        assert rp["ok"] and rp["closed"] and rp["queue_closed"] and rp["n_records"] == 3 and rp["n_written"] == 3 and rp["n_already_present"] == 0
        assert rp["accepted"] == "seeded" and rp["marker"]["n_enqueued"] == 3 and rp["marker"]["synthesised_by"] == co.REPLAY_SYNTHESISED_BY
        assert rp["source_keys"]["queue"].startswith(f"{W_PREV[:7]}|{WAVE_PREV}|") and rp["target_keys"]["queue"].startswith(f"{W_THIS[:7]}|{WAVE}|")
        # THIS wave's queue, read exactly as the rung consumer reads it: CLOSED.json + the accepted projection over this wave's Dict
        this_spec = L.QueueSpec.build(W_THIS[:7], WAVE, "pasttense", "s1", "static", 8192, ["0"])
        assert L.read_closed(w.view, this_spec)["n_items"] == 3
        acc = L.accepted_producer_attempts(w.accepted_this, this_spec)
        assert acc == {"0": AID}
        entries = L.accepted_queue_entries(w.view, None, this_spec, acc)
        assert len(entries) == 3
        for sl, r in entries.items():
            assert (r["W7"], r["wave_uuid"]) == (W_THIS[:7], WAVE)
            src = next(x for x in w.recs if x["item_key"] == r["item_key"])
            assert {k: v for k, v in r.items() if k not in ("W7", "wave_uuid")} == {k: v for k, v in src.items() if k not in ("W7", "wave_uuid")}
            L.check_redraw_sampling(r, where=sl)
        # the prior wave's queue is untouched
        assert len(L.accepted_queue_entries(w.view, w.accepted_prev, w.spec_prev)) == 3
        # the REAL consumer loop over THIS wave's spec, in its own container snapshot: it reads CLOSED.json + the accepted projection, draws every
        # replayed record through the injected draw, elects each, and reports CLOSED and covered -- the carried base's rung completes
        rv = w.shared.snapshot("rung-1")
        drawn = []

        def draw(rec):
            drawn.append(rec["item_key"])
            return RR.Draw(record={"text": "short", "finish_reason": "stop", "completion_tokens": 3, "prompt_tokens": 5, "max_tokens": 8192,
                                   "sampling_seed": rec["sampling_seed"]}, text="short", response=None, request_id=f"rq-{rec['item_key']}",
                           completion_id=f"cmpl-{rec['item_key']}")
        tally = RR.consume(rv, accepted_kv=w.accepted_this, claims_kv=L.MemoryKV("claims"), results_kv=L.MemoryKV("results"), spec=this_spec,
                           attempt_id="rung-1", call_id="fc", draw=draw, max_rounds=2, poll_s=0, sleep=lambda s: None, log=lambda *a: None)
        assert tally["closed"] and tally["covered"] and tally["drawn"] == 3 and tally["won"] == 3 and sorted(drawn) == sorted(r["item_key"] for r in w.recs)
        # the certificate re-validation accepts the record and holds the replay artefacts to the volume
        read = lambda p: (w.files.get(p) if p in w.files else (w.view.read_bytes(co._vol_rel(p)).decode() if w.view.exists(co._vol_rel(p)) else None))  # noqa: E731
        assert co.carry_over_problems(rec, plan=w.this, read_text=read, accepted_get=w.accepted_this.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        # idempotent: a second evaluation replays nothing new and still closes
        rec2 = _evaluate(w, replay=_replay_for(w))
        rp2 = rec2["carried"][F.SID]["clauses"]["ladder_queue"]["replay"]
        assert rp2["n_already_present"] == 3 and rp2["n_written"] == 0 and rp2["accepted"] == "present" and rp2["queue_closed"]

    def test_a_torn_prior_record_refuses_the_row_by_name_and_writes_nothing(self, ladder_world):
        w = ladder_world
        victim = w.recs[1]
        rel = L.queue_entry_rel(w.spec_prev.ladder_root, victim["arm_s"], victim["item_key"], AID)
        w.shared.files[rel] = w.shared.files[rel][:-8] + b'"tor\n'          # torn: bytes differ from the inventory's sha
        rec = _evaluate(w, replay=_replay_for(w))
        assert rec["refused"][F.SID]["clause"] == "ladder_queue" and "sha differs" in rec["refused"][F.SID]["why"]
        this_spec = L.QueueSpec.build(W_THIS[:7], WAVE, "pasttense", "s1", "static", 8192, ["0"])
        assert w.view.walk(f"{this_spec.ladder_root}") == [] and w.accepted_this.get(L.key_str(this_spec.producers[0].key)) is None

    def test_a_record_with_a_broken_sampling_block_refuses_even_when_its_sha_matches(self, ladder_world):
        w = ladder_world
        victim = w.recs[0]
        rel = L.queue_entry_rel(w.spec_prev.ladder_root, victim["arm_s"], victim["item_key"], AID)
        bad = json.loads(w.shared.files[rel].decode())
        del bad["redraw"]["sampling"]["top_k"]
        data = (L.canonical_json(bad) + "\n").encode()
        w.shared.files[rel] = data
        inv_rel = L.inventory_rel(w.root_rel)
        inv = json.loads(w.shared.files[inv_rel].decode())
        for e in inv["entries"]:
            if e["path"] == rel:
                e["sha256"] = L.sha256_hex(data)
        w.shared.files[inv_rel] = (L.canonical_json(inv) + "\n").encode()
        rec = _evaluate(w, replay=_replay_for(w))
        assert rec["refused"][F.SID]["clause"] == "ladder_queue" and "sampler keys" in rec["refused"][F.SID]["why"]

    def test_a_foreign_file_already_under_this_wave_refuses(self, ladder_world):
        w = ladder_world
        this_spec = L.QueueSpec.build(W_THIS[:7], WAVE, "pasttense", "s1", "static", 8192, ["0"])
        victim = w.recs[2]
        w.shared.files[L.queue_entry_rel(this_spec.ladder_root, victim["arm_s"], victim["item_key"], AID)] = b'{"other": true}\n'
        rec = _evaluate(w, replay=_replay_for(w))
        assert rec["refused"][F.SID]["clause"] == "ladder_queue" and "DIFFERENT bytes" in rec["refused"][F.SID]["why"]

    def test_the_certificate_refuses_a_carried_producer_row_whose_replay_record_or_marker_is_gone(self, ladder_world):
        w = ladder_world
        rec = _evaluate(w, replay=_replay_for(w))
        read = lambda p: (w.files.get(p) if p in w.files else (w.view.read_bytes(co._vol_rel(p)).decode() if w.view.exists(co._vol_rel(p)) else None))  # noqa: E731
        stale = copy.deepcopy(rec)
        del stale["carried"][F.SID]["clauses"]["ladder_queue"]["replay"]
        assert any("no complete queue replay record" in p for p in co.carry_over_problems(stale, plan=w.this, read_text=read, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        marker = rec["carried"][F.SID]["clauses"]["ladder_queue"]["replay"]["target_keys"]["marker"]
        del w.shared.files[marker]
        w.view.reload()
        assert any("CLOSED marker" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=read, git=w.git, accepted_get_for_dict=_agd(w.dicts)))


# --------------------------------------------------------------------------- (2) Petri: the D111 admitted hunk set + the sidecar proof

def _sha_exists(sha: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


class TestPetriAdmittedHunks:
    @pytest.mark.skipif(not (_sha_exists("56aff576e") and _sha_exists("8024980d2")), reason="the W22 / W23 shas are not in this clone")
    def test_the_real_w22_to_w23_diff_of_both_files_is_exactly_the_pinned_hunk_set(self):
        git = co.default_git(REPO)
        for path in co.D111_ADMITTED_FILES:
            adm = co.admitted_difference(git, "56aff576e", "8024980d2", path)
            assert adm is not None and adm["kind"] == co.ADMITTED_KIND_CHAIN and adm["reason"] == co.ADMITTED_REASON_D111, path
            # the table is a WHITELIST: every diff hunk pinned (a second blank-line alignment of one hunk sits in it beside the first)
            assert {h["sha256"] for h in adm["hunks"]} <= set(co.D111_ADMITTED_HUNKS[path]) and all(h["label"] for h in adm["hunks"])

    def test_an_unrelated_change_to_an_admitted_file_is_an_extra_hunk_and_is_refused(self):
        a = "x = 1\nDEFAULT_FALLBACK_MODELS = (A, B)\ny = 2\n"
        b_pinned = "x = 1\nDEFAULT_FALLBACK_MODELS = (A,)\ny = 2\n"
        path = "src/sisteer/capeval/petri/auditor_fallback.py"
        hunk = co.diff_hunks(a, b_pinned)[0]["sha256"]
        saved = co.D111_ADMITTED_HUNKS[path]
        try:
            co.D111_ADMITTED_HUNKS[path] = {hunk: "test pin"}
            assert co.d111_admitted_difference(a, b_pinned, path)["n_hunks"] == 1
            assert co.d111_admitted_difference(a, b_pinned + "z = 3\n", path) is None            # an extra hunk outside the pin
            assert co.d111_admitted_difference(a, "x = 1\nDEFAULT_FALLBACK_MODELS = (A, C)\ny = 2\n", path) is None
            # the pin hashes the hunk BODY: a line-number shift elsewhere that adds no hunk of its own is not possible; a shift WITH a change is an extra hunk
            git = F.FakeGit({W_PREV: {path: "v1"}, W_THIS: {path: "v2"}}, {(W_PREV, path): a, (W_THIS, path): b_pinned})
            assert co.admitted_difference(git, W_PREV, W_THIS, path)["kind"] == co.ADMITTED_KIND_CHAIN
        finally:
            co.D111_ADMITTED_HUNKS[path] = saved

    def test_the_removed_leg_buckets_are_the_auditor_modules_spelling(self):
        from sisteer.capeval.petri import auditor_fallback as af
        assert set(co.PETRI_REMOVED_LEG_BUCKETS) <= set(af.CHAIN_KINDS) and af.CHAIN_CONTINUED_QWEN in co.PETRI_REMOVED_LEG_BUCKETS
        assert af.auditor_label(af.DEFAULT_QWEN_MODEL) in co.PETRI_REMOVED_LEG_LABELS and af.DEFAULT_QWEN_MODEL in co.PETRI_REMOVED_LEG_MODELS


def _petri_store(tmp: Path, cells: dict) -> list:
    """`{cell_name: (chain tally, [used-label lists of refused records])}` -> cell dirs with results.json + the refused ledger."""
    out = []
    for name, (tally, refused) in cells.items():
        d = tmp / "cells" / name
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps({"audited": {"items": sum(tally.values()), "expected": 4}, "auditor_chain": tally, "behaviors": {}}))
        (d / "petri_auditor_refused.json").write_text(json.dumps({"version": 1, "auditor_refused": [
            {"behavior": "b", "scenario_id": f"s{i}", "epoch": 1, "shed": {"kind": "content_correlated", "auditor_chain": {"used": [{"label": lab, "model": "m"} for lab in used]}}}
            for i, used in enumerate(refused)]}))
        out.append(str(d))
    return out


class TestPetriSidecarProof:
    def test_sonnet_and_grok_uses_only_is_a_proof(self, tmp_path):
        cells = _petri_store(tmp_path, {"c0": ({"sonnet_only": 3, "continued_by_grok": 1, "continued_by_qwen": 0, "auditor_refused_all_legs": 0, "auditor_unavailable": 0}, []),
                                       "c1": ({"sonnet_only": 4, "continued_by_grok": 0, "continued_by_qwen": 0, "auditor_refused_all_legs": 0, "auditor_unavailable": 1},
                                              [["sonnet-5", "grok-4.6"]])})
        pf = co.petri_sidecar_proof(cells, co.__dict__.get("local_read_text") or (lambda p: Path(p).read_text() if Path(p).is_file() else None))
        assert pf["ok"] and pf["removed_leg"]["uses"] == 0 and pf["cells"] == 2 and pf["audited_items"] == 9 and pf["refused_records"] == 1
        assert pf["used_legs"] == {"sonnet-5": 1, "grok-4.6": 1} and pf["judge_fallback"] == "n/a (not in prior tree)"

    def test_one_use_of_the_removed_leg_refuses_by_name(self, tmp_path):
        read = lambda p: Path(p).read_text() if Path(p).is_file() else None  # noqa: E731
        cells = _petri_store(tmp_path / "a", {"c0": ({"sonnet_only": 3, "continued_by_grok": 0, "continued_by_qwen": 1, "auditor_refused_all_legs": 0, "auditor_unavailable": 0}, [])})
        pf = co.petri_sidecar_proof(cells, read)
        assert not pf["ok"] and "qwen-abl" in pf["why"] and pf["removed_leg"]["uses"] == 1
        cells = _petri_store(tmp_path / "b", {"c0": ({"sonnet_only": 3, "continued_by_grok": 0, "continued_by_qwen": 0, "auditor_refused_all_legs": 0, "auditor_unavailable": 1},
                                                    [["sonnet-5", "grok-4.6", "qwen-abl"]])})
        pf = co.petri_sidecar_proof(cells, read)
        assert not pf["ok"] and pf["removed_leg"]["uses"] == 1 and pf["used_legs"]["qwen-abl"] == 1
        # a cell without results.json is never proven
        assert not co.petri_sidecar_proof([str(tmp_path / "missing")], read)["ok"]

    def _petri_world(self, tmp_path, *, proof_ok: bool):
        prev = F._plan(W_PREV, materialisation="final", rows=[F._row(W_PREV, W_PREV[:7], sid="petri__s1__base__shard00", leg="petri")])
        this = F._plan(W_THIS, materialisation="final", rows=[F._row(W_THIS, W_THIS[:7], sid="petri__s1__base__shard00", leg="petri")])
        files, dicts = {}, {}
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": prev["results_root"]}
        F._root_files(files, plan=prev, row=prev["standups"][0], identity=ident)
        dicts[prev["dicts"]["accepted"]] = {prev["standups"][0]["accepted_key"]: AID}
        blobs = {W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS)}
        for sha in blobs:
            for f in co.LEG_HELPERS["petri"]:
                blobs[sha][f] = "same"
        blobs[W_THIS][F.MANIFEST] = "manifest-v2"
        git = F.FakeGit(blobs, {(W_PREV, F.MANIFEST): "petri:\n  auditor_fallback_models:\n    - a\n    - b\n", (W_THIS, F.MANIFEST): "petri:\n  auditor_fallback_models:\n    - a\n"})
        prior = co.load_prior_source(F._prior_dir(tmp_path, prev))
        calls = []

        def proof(**kw):
            calls.append({**kw, "_proof": proof})
            kw["read_text"]("/synthetic/petri/cells/c0/results.json")      # fj5: a proof reads its inputs through the handed reader (recorded, absent here)
            return {"ok": proof_ok, "cells": 8, "removed_leg": {"uses": 0 if proof_ok else 2}, "why": None if proof_ok else "the removed auditor leg ['qwen-abl'] was reached 2 time(s)"}
        rec = co.evaluate_carry_over(this_plan=this, this_sha=W_THIS, priors=[prior], read_text=files.get, accepted_get_for_dict=lambda n: dicts.get(n, {}).get, git=git,
                                     readiness_problems=F.fake_readiness, petri_proof=proof)
        return rec, calls, this, files, git, dicts

    def test_a_petri_row_with_an_admitted_chain_difference_carries_with_the_proof_and_is_refused_without_it(self, tmp_path):
        rec, calls, this, files, git, dicts = self._petri_world(tmp_path / "ok", proof_ok=True)
        assert rec["n_carried"] == 1 and calls and calls[0]["aid"] == AID and calls[0]["prev_row"]["leg"] == "petri"
        ps = rec["carried"]["petri__s1__base__shard00"]["clauses"]["producer_set"]
        assert ps["petri_sidecar_proof"]["ok"] and ps["admitted_differences"][0]["path"] == F.MANIFEST
        assert co.carry_over_problems(rec, plan=this, read_text=files.get, git=git, accepted_get_for_dict=_agd(dicts), petri_proof=calls[0]["_proof"]) == []
        rec, _calls, _this, _files, _git, _dicts = self._petri_world(tmp_path / "bad", proof_ok=False)
        r = rec["refused"]["petri__s1__base__shard00"]
        assert r["clause"] == "producer_set" and "qwen-abl" in r["why"] and "reached 2" in r["why"]

    def test_the_cell_dirs_come_from_the_completion_bundle_through_the_drives_spelling(self, tmp_path):
        import seedrep_petri_drive as drive
        aid = "01TESTTASK-deadbeef0002"
        ev = {"petri": {"state": "complete", "kind": "completion", "evidence": {"attempt_id": aid, "W7": "abcdef0", "wave_uuid": WAVE, "view": "s1",
                                                                              "verdict": {"detail": [["leg_0_s1", "petri__none__s1", "complete"], ["leg_3_s1", "petri__x__s1", "complete"]]}}}}
        cells = co.petri_cell_dirs(ev, home=tmp_path)
        root = drive.out_root_for(tmp_path, "abcdef0", WAVE.replace("-", "")[:8], aid, drive.run_tag("s1", 1))
        assert cells == [f"{root}/cells/petri__none__s1", f"{root}/cells/petri__x__s1"]
        with pytest.raises(ValueError):
            co.petri_cell_dirs({"petri": {"evidence": {"verdict": {"detail": []}}}}, home=tmp_path)


# --------------------------------------------------------------------------- (3) the steward's completion accounting + pull_leg's root

class TestCarriedCompletionAndRetrieval:
    def test_steward_tick_covers_a_carried_key_by_provenance_and_reaches_done(self):
        import seedrep_steward as st
        plan = F._plan(W_THIS, materialisation="final")
        row = plan["standups"][0]
        required = {row["accepted_key"]: row["standup_id"]}
        state = st.StewardState()
        before = st.steward_tick(plan=plan, calls=[], snapshot={}, holds={}, state=state, required=required)
        assert before["stop"] is None and before["covered"] == 0 and before["required"] == 1                  # the known-bad: never completes
        after = st.steward_tick(plan=plan, calls=[], snapshot={}, holds={}, state=state, required=required, carried={row["accepted_key"]: AID})
        assert after["stop"] == "done" and after["covered"] == 1 and after["carried"] == 1 and after["invoke"] == []
        # a carried key that is NOT required does not count, and never triggers a re-attempt
        other = st.steward_tick(plan=plan, calls=[], snapshot={}, holds={}, state=state, required=required, carried={"W7|x|other": AID})
        assert other["stop"] is None and other["covered"] == 0 and other["carried"] == 0

    def test_pull_leg_resolves_a_carried_root_through_the_carry_record(self, ladder_world, monkeypatch, tmp_path):
        w = ladder_world
        rec = _evaluate(w, replay=_replay_for(w))
        rs = tmp_path / "rs"
        co.write_carry_over(rs, rec)
        # the served pull's default spells `attempt_root_for(results_root, row, att)`; the plan module's is keyword-only -- adapt it the way served does
        default = lambda rr, row, aid: lp.attempt_root_for(rr, leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row["kind"])  # noqa: E731
        resolve = co.attempt_root_resolver(co.read_carry_over(rs), default)
        row = w.this["standups"][0]
        carried_root = rec["carried"][F.SID]["attempt_root"]
        # the served pull spells `attempt_root_for(results_root, row, att)`; the resolver returns the PRIOR tree's root for the carried attempt id
        got = resolve(w.this["results_root"], row, AID)
        assert got == carried_root and W_PREV[:7] in got and W_THIS[:7] not in got
        # a fresh attempt id resolves through the default, under THIS results root
        fresh = resolve(w.this["results_root"], row, "01FRESH-0000000000000001")
        assert W_THIS[:7] in fresh and "01FRESH" in fresh
        assert co.carried_attempt_roots(None) == {} and co.carried_attempt_roots(rec) == {AID: carried_root}


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
    spec = importlib.util.spec_from_file_location("seedrep_served_fj3", REPO / "scripts" / "modal" / "seedrep_served.py")
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


class TestPullLegThroughServed:
    def test_pull_leg_pulls_a_carried_root_from_the_prior_tree(self, srv, tmp_path, ladder_world):
        w = ladder_world
        rec = _evaluate(w, replay=_replay_for(w))
        rs = tmp_path / "rs"
        co.write_carry_over(rs, rec)
        row = w.this["standups"][0]
        acc = F._FakeDict(w.this["dicts"]["accepted"] + "-fj3pull")
        acc.put(row["accepted_key"], AID)
        pulled = []

        def fake_get(src: str, out: Path) -> None:
            pulled.append(src)
        try:
            srv.pull_leg(w.this, "pasttense", tmp_path / "stores", rs, stages="base", accepted=acc, run_get=fake_get)
        except Exception:  # noqa: BLE001 -- the local verification tail (COMPLETE.json + inventories of the pulled tree) is fj2's; the ROOT is what this pins
            pass
        assert pulled, "pull_leg never asked for the carried root"
        assert pulled[0] == rec["carried"][F.SID]["attempt_root"] and W_PREV[:7] in pulled[0] and W_THIS[:7] not in pulled[0]
        # the known-bad spelling (the CURRENT results root) is never asked for
        assert all(w.this["results_root"] not in src for src in pulled)


# --------------------------------------------------------------------------- (4) the certificate re-validates the evidence, not the flags

class TestCertificateRevalidatesEvidence:
    def test_every_evidence_source_is_bound_by_digest_and_its_removal_or_change_refuses(self, ladder_world):
        w = ladder_world
        rec = _evaluate(w, replay=_replay_for(w))
        e = rec["carried"][F.SID]
        inv = e["evidence_inventory"]
        root = e["attempt_root"]
        det_path = f"{root}/{lp.determinism_rels(w.this['standups'][0])[0][1]}"
        assert det_path in inv and f"{root}/{lp.SERVE_DIR_REL}/endpoint.json" in inv and f"{root}/{lp.CELLS_JSON_REL}" in inv and f"{root}/RESULT" in inv
        read = lambda p: (w.files.get(p) if p in w.files else (w.view.read_bytes(co._vol_rel(p)).decode() if w.view.exists(co._vol_rel(p)) else None))  # noqa: E731
        assert co.carry_over_problems(rec, plan=w.this, read_text=read, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        for victim, needle in ((det_path, "no longer on the volume"), (f"{root}/{lp.SERVE_DIR_REL}/endpoint.json", "no longer on the volume")):
            saved = w.files.pop(victim)
            probs = co.carry_over_problems(rec, plan=w.this, read_text=read, git=w.git, accepted_get_for_dict=_agd(w.dicts))
            assert any(victim in p and needle in p for p in probs), (victim, probs)
            w.files[victim] = saved
        w.files[det_path] = w.files[det_path].replace('"reproduces": true', '"reproduces": false')
        probs = co.carry_over_problems(rec, plan=w.this, read_text=read, git=w.git, accepted_get_for_dict=_agd(w.dicts))
        assert any(det_path in p and "changed since the evaluation" in p for p in probs)
        stale = copy.deepcopy(rec)
        del stale["carried"][F.SID]["evidence_inventory"]
        assert any("no evidence inventory" in p for p in co.carry_over_problems(stale, plan=w.this, read_text=read, git=w.git, accepted_get_for_dict=_agd(w.dicts)))


# --------------------------------------------------------------------------- (5) the scoped determinism waiver

W19 = "c65b47ba4" + "0" * 31
WAIVER = {"by": "user", "at": "2026-09-19T20:15Z", "scope": {"W": "c65b47ba4", "leg": "petri", "view": "s1", "legs": list(range(8))},
          "reason": "F-AA refused the in-container check; engine reproducibility proven on the same build by the W20/W22 rows", "decision": "D118"}


def _w19_petri_world(tmp_path, *, view="s1", n_kids=8):
    prev_row = F._row(W19, W19[:7], sid=f"petri__{view}__base__shard00", leg="petri")
    this_row = F._row(W_THIS, W_THIS[:7], sid=f"petri__{view}__base__shard00", leg="petri")
    for r in (prev_row, this_row):
        r["engine"] = r["view"] = view
        k0 = r["children"][0]
        r["children"] = [{**copy.deepcopy(k0), "arm": f"a{i}", "grading_host_drive": {"cell_index": i, "target_decode": "vendor", "target_seed_base": 1}} for i in range(n_kids)]
    prev = F._plan(W19, materialisation="final", rows=[prev_row])
    this = F._plan(W_THIS, materialisation="final", rows=[this_row])
    files, dicts = {}, {}
    ident = {"W": W19, "W7": W19[:7], "wave_uuid": WAVE, "results_root": prev["results_root"]}
    root = F._root_files(files, plan=prev, row=prev["standups"][0], identity=ident, n_children=n_kids)
    # fj2's root writes ONE audit verdict (children[0]); the eight arm-runs each need theirs, bound in modal_evidence.json like the first
    prow = prev["standups"][0]
    k0 = prow["children"][0]
    rel0 = lp.audit_verdict_rel(k0["eval_leg"], k0["arm"], k0["s"], stage=prow["stage"], shard=prow["shard"])
    ev = json.loads(files[f"{root}/modal_evidence.json"])
    for k in prow["children"][1:]:
        rel = lp.audit_verdict_rel(k["eval_leg"], k["arm"], k["s"], stage=prow["stage"], shard=prow["shard"])
        doc = json.loads(files[f"{root}/{rel0}"])
        # F-AO: the verdict's `arm` is the STORE's spelling (the served cell) and its `arm_binding` binds that cell; the plan's label `a{i}` names
        # the child (and the plan-derived filename this double keeps), never the verdict's content -- the certificate binds through `arm_binding`
        cell = str(k.get("cell") or k["arm"])
        doc["arm"] = cell
        doc["arm_binding"] = {**(doc.get("arm_binding") or {}), "arm": cell, "harness_id": cell, "traintest_id": cell, "vmodel": f"petri__{cell}", "spelling": "harness==traintest"}
        text = json.dumps(doc)
        files[f"{root}/{rel}"] = text
        ev[lp.PRODUCER_EVIDENCE_KEY][rel] = {**ev[lp.PRODUCER_EVIDENCE_KEY][rel0], "sha256": F._sha(text)}
    files[f"{root}/modal_evidence.json"] = json.dumps(ev)
    det_rel = lp.determinism_rels(prev["standups"][0])[0][1]
    del files[f"{root}/{det_rel}"]                                                       # the W19 shape: F-AA refused the in-container spot-check
    dicts[prev["dicts"]["accepted"]] = {prev["standups"][0]["accepted_key"]: AID}
    blobs = {W19: F._blobs(W19), W_THIS: F._blobs(W_THIS)}
    for sha in blobs:
        for f in co.LEG_HELPERS["petri"]:
            blobs[sha][f] = "same"
    git = F.FakeGit(blobs)
    prior = co.load_prior_source(F._prior_dir(tmp_path, prev))
    return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, git=git, prior=prior, root=root, det_path=f"{root}/{det_rel}")


def _eval_w(w, waivers=()):
    return co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                  git=w.git, readiness_problems=F.fake_readiness, criterion3_waivers=list(waivers))


class TestCriterion3Waiver:
    def test_the_w19_shaped_petri_root_is_refused_without_a_waiver(self, tmp_path):
        w = _w19_petri_world(tmp_path)
        rec = _eval_w(w)
        r = rec["refused"]["petri__s1__base__shard00"]
        assert r["clause"] == "complete" and "determinism" in r["why"] and "waiver" in r["why"]

    def test_admitted_under_a_covering_waiver_copied_verbatim_and_certified(self, tmp_path):
        w = _w19_petri_world(tmp_path)
        rec = _eval_w(w, [WAIVER])
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"]["petri__s1__base__shard00"]
        assert e["clauses"]["complete"]["waived"] == "criterion3 (D118)" and e["clauses"]["complete"]["checks"]["determinism"]["verdict"] == co.DETERMINISM_WAIVED
        assert e["criterion3_waiver"] == WAIVER and e["evidence_inventory"][w.det_path] is None
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        # the other checks still ran and still gate: an audit that DISAGREES is refused even under the waiver
        w2 = _w19_petri_world(tmp_path / "b")
        for p in list(w2.files):
            if p.endswith(".json") and '"verdict": "AGREES"' in w2.files[p] and "audit" in p:
                w2.files[p] = w2.files[p].replace('"verdict": "AGREES"', '"verdict": "DISAGREES"')
        assert _eval_w(w2, [WAIVER])["n_carried"] == 0

    @pytest.mark.parametrize("scope", [{"view": "s2"}, {"leg": "misalign"}, {"W": "deadbee"}, {"legs": list(range(7))}])
    def test_refused_under_a_waiver_out_of_scope(self, tmp_path, scope):
        w = _w19_petri_world(tmp_path)
        waiver = copy.deepcopy(WAIVER)
        waiver["scope"].update(scope)
        rec = _eval_w(w, [waiver])
        assert rec["n_carried"] == 0 and rec["refused"]["petri__s1__base__shard00"]["clause"] == "complete"

    def test_a_malformed_waiver_is_refused_by_name(self, tmp_path):
        assert co.criterion3_waiver_problems({"by": "user"}) and co.criterion3_waiver_problems({**WAIVER, "scope": {**WAIVER["scope"], "legs": []}})
        assert co.criterion3_waiver_problems({**WAIVER, "decision": ""})
        assert not co.criterion3_waiver_problems(WAIVER)

    def test_the_certificate_refuses_a_carried_row_whose_waiver_was_removed_or_no_longer_covers(self, tmp_path):
        w = _w19_petri_world(tmp_path)
        rec = _eval_w(w, [WAIVER])
        stale = copy.deepcopy(rec)
        del stale["carried"]["petri__s1__base__shard00"]["criterion3_waiver"]
        probs = co.carry_over_problems(stale, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts))
        assert any("criterion3_waiver" in p for p in probs), probs
        narrowed = copy.deepcopy(rec)
        narrowed["carried"]["petri__s1__base__shard00"]["criterion3_waiver"]["scope"]["legs"] = [0, 1]
        assert any("does not cover this row" in p for p in co.carry_over_problems(narrowed, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        # a record whose complete clause forgot the waiver while the determinism source is recorded absent refuses too
        forgot = copy.deepcopy(rec)
        forgot["carried"]["petri__s1__base__shard00"]["clauses"]["complete"].pop("waived")
        forgot["carried"]["petri__s1__base__shard00"]["clauses"]["complete"]["checks"]["determinism"].pop("waived")
        assert any("does not record the waiver" in p for p in co.carry_over_problems(forgot, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))

    def test_the_cli_loader_takes_a_document_or_a_file_and_refuses_a_malformed_one(self, tmp_path, srv):
        assert srv.load_criterion3_waivers("") == []
        assert srv.load_criterion3_waivers(json.dumps(WAIVER)) == [WAIVER] and srv.load_criterion3_waivers(json.dumps([WAIVER])) == [WAIVER]
        f = tmp_path / "w.json"
        f.write_text(json.dumps(WAIVER))
        assert srv.load_criterion3_waivers(str(f)) == [WAIVER]
        with pytest.raises(srv.LaunchRefused):
            srv.load_criterion3_waivers(json.dumps({"by": "user"}))


# --------------------------------------------------------------------------- fj4: W19 as the Petri carry source; mixed queues; the judge reading

class TestW19Source:
    @pytest.mark.skipif(not all(_sha_exists(s) for s in ("c65b47ba4", "56aff576e", "edec20133")), reason="the W19 / W22 / W23 shas are not in this clone")
    def test_w19_and_w22_carry_the_same_pinned_hunk_set_into_w23(self):
        """W19 (c65b47ba4) is the Petri carry source under D118: its three gating Petri files are byte-identical to W22's, so the D111 pin admits
        both -- pinned here on the real blobs, and `D111_PIN_SOURCES` names both."""
        git = co.default_git(REPO)
        assert set(co.D111_PIN_SOURCES) == {"56aff576e", "c65b47ba4"}
        for path in (*co.D111_ADMITTED_FILES, "src/sisteer/capeval/manifest.py"):
            a = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"c65b47ba4:{path}"], capture_output=True, text=True).stdout.strip()
            b = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"56aff576e:{path}"], capture_output=True, text=True).stdout.strip()
            assert a == b, path
        for src in co.D111_PIN_SOURCES:
            for path in co.D111_ADMITTED_FILES:
                adm = co.admitted_difference(git, src, "edec20133", path)
                assert adm is not None and {h["sha256"] for h in adm["hunks"]} <= set(co.D111_ADMITTED_HUNKS[path]) and adm["tables"] == ["D111"], (src, path)
            # manifest.py at edec20133 (before the judge chain) differs by its ONE D111 hunk -- pinned, so the pinned kind admits it under D111 alone
            adm = co.admitted_difference(git, src, "edec20133", "src/sisteer/capeval/manifest.py")
            assert adm is not None and adm["kind"] == co.ADMITTED_KIND_CHAIN and adm["tables"] == ["D111"] and adm["n_hunks"] == 1

    def test_a_w19_pasttense_full_root_without_its_determinism_record_is_refused_by_c_even_with_the_d118_waiver(self, tmp_path):
        """The D118 waiver's scope is petri / s1 / legs 0-7; a W19 pasttense root lacking the spot-check is out of scope -> (c) refuses. W22's
        pasttense FULL (which has the record) is the pasttense source."""
        prev = F._plan(W19, materialisation="final", rows=[F._row(W19, W19[:7])])
        this = F._plan(W_THIS, materialisation="final", rows=[F._row(W_THIS, W_THIS[:7])])
        files, dicts = {}, {}
        ident = {"W": W19, "W7": W19[:7], "wave_uuid": WAVE, "results_root": prev["results_root"]}
        root = F._root_files(files, plan=prev, row=prev["standups"][0], identity=ident)
        del files[f"{root}/{lp.determinism_rels(prev['standups'][0])[0][1]}"]
        dicts[prev["dicts"]["accepted"]] = {prev["standups"][0]["accepted_key"]: AID}
        prior = co.load_prior_source(F._prior_dir(tmp_path, prev))
        rec = co.evaluate_carry_over(this_plan=this, this_sha=W_THIS, priors=[prior], read_text=files.get, accepted_get_for_dict=lambda n: dicts.get(n, {}).get,
                                     git=F.FakeGit({W19: F._blobs(W19), W_THIS: F._blobs(W_THIS)}), readiness_problems=F.fake_readiness, criterion3_waivers=[WAIVER])
        r = rec["refused"][F.SID]
        assert r["clause"] == "complete" and "determinism" in r["why"] and "absent" in r["why"]
        assert co.covering_waiver([WAIVER], W=W19, row=prev["standups"][0]) is None


class TestMixedQueueSteward:
    def test_reconcile_plan_queues_closes_a_queue_with_one_carried_and_one_fresh_producer(self, tmp_path):
        """fj4 (fj3 gap 1): the steward's per-queue root resolver finds a CARRIED producer's stub under the ladder root and the fresh one under
        the app's spelling; the queue closes once both are accepted -- through the actual `reconcile_plan_queues`."""
        import seedrep_steward as st
        prev, this = _plans()
        for plan in (prev, this):
            for r in plan["standups"]:
                r["ladder"] = dict(LADDER, shards=["0", "1"], feeder_shards=["0", "1"])
        shared = L.SharedVolume()
        accepted_prev, accepted_this = L.MemoryKV("prev"), L.MemoryKV("this")
        # the prior wave: shard 0 produced 3 records (the carried one)
        spec_prev = L.QueueSpec.build(W_PREV[:7], WAVE_PREV, "pasttense", "s1", "static", 8192, ["0", "1"])
        c0 = T.Container(L, shared, spec_prev, "0", AID)
        for i in range(3):
            c0.enqueue_row(T.static_row(f"g{i}", 1, seed=7 + i))
        c0.complete()
        assert c0.finish(accepted_prev)["state"] == "accepted"
        # replay shard 0 into THIS wave (the carry), with the app's results-root spelling as the default resolver
        default_root_of = lp.producer_attempt_root_of(this["results_root"])
        rp = co.replay_ladder_queue(L=L, vol=shared.read_only_view(), accepted_kv=accepted_this, prev_row=prev["standups"][0], prior={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE_PREV},
                                    this_plan=this, aid=AID, prev_root=c0.root, default_root_of=default_root_of)
        assert rp["ok"] and rp["closed"] and not rp["queue_closed"] and rp["producers"] == {"0": "CLOSED", "1": "PRODUCER_MISSING"}
        # the steward's pass: still open (shard 1 has no attempt yet)
        plan = {"W7": W_THIS[:7], "wave_uuid": WAVE, "results_root": this["results_root"],
                "queues": [{"leg": "pasttense", "engine": "s1", "cap_next": "8192", "producers": [{"shard": 0}, {"shard": 1}]}]}
        state = st.StewardState()
        out = st.reconcile_plan_queues(L, plan, vol=shared.read_only_view(), accepted_kv=accepted_this, state=state, log=lambda *a: None)
        spec_this = L.QueueSpec.build(W_THIS[:7], WAVE, "pasttense", "s1", "static", 8192, ["0", "1"])
        assert out["closed"] == [] and out["open"][spec_this.ladder_root] == {"0": "CLOSED", "1": "PRODUCER_MISSING"}
        # a FRESH shard-1 attempt of THIS wave, at the app's root spelling, accepted -> the steward's next pass closes the queue
        c1 = T.Container(L, shared, spec_this, "1", "01FRESH-0000000000000001")
        c1.root = default_root_of(c1.ref, "01FRESH-0000000000000001")
        c1.producer = L.Producer(c1.vol, spec_this, c1.ref, "01FRESH-0000000000000001", c1.root)
        c1.enqueue_row(T.static_row("h0", 1, seed=99))
        c1.complete()
        assert c1.finish(accepted_this)["state"] == "accepted"
        out = st.reconcile_plan_queues(L, plan, vol=shared.read_only_view(), accepted_kv=accepted_this, state=state, log=lambda *a: None)
        assert out["closed"] == [spec_this.ladder_root] and L.read_closed(shared.read_only_view(), spec_this)["n_items"] == 4


class TestJudgeFallbackReading:
    def test_a_w23_store_reports_judge_uses_or_names_the_missing_tally(self, tmp_path):
        read = lambda p: Path(p).read_text() if Path(p).is_file() else None  # noqa: E731
        base = {"sonnet_only": 2, "continued_by_grok": 0, "continued_by_qwen": 0, "auditor_refused_all_legs": 0, "auditor_unavailable": 0}
        cells = _petri_store(tmp_path / "a", {"c0": (base, [])})
        res = json.loads(Path(cells[0], "results.json").read_text())
        res["judge_chain"] = {"opus_only": 2, "continued_by_grok": 1}
        res["judge_refused"] = {"total": 1}
        # fj5: the proof verifies outcome coverage, so the synthetic summary's behaviors must account for the judge-refused item
        res["behaviors"] = {"b": {"n_items": 3, "audited": 2, "auditor_refused": {"total": 0}, "auditor_refused_items": [], "judge_refused": {"total": 1},
                                  "judge_refused_items": [{"scenario_id": "s1", "epoch": 1, "auditor_chain": {"used": [{"label": "sonnet-5", "model": "m"}]}}],
                                  "auditor_chain": {**base, "sonnet_only": 2}}}
        Path(cells[0], "results.json").write_text(json.dumps(res))
        Path(cells[0], co.PETRI_JUDGE_REFUSED_LEDGER).write_text(json.dumps({"judge_refused": [{"scenario_id": "s1", "auditor_chain": {"used": [{"label": "sonnet-5"}]}}]}))
        pf = co.petri_sidecar_proof(cells, read, judge_chain_in_prior=True)
        assert pf["ok"] and pf["judge_fallback"]["uses_per_leg"] == {"opus_only": 2, "continued_by_grok": 1} and pf["judge_fallback"]["judge_refused_total"] == 1
        cells = _petri_store(tmp_path / "b", {"c0": (base, [])})
        pf = co.petri_sidecar_proof(cells, read, judge_chain_in_prior=True)
        assert pf["judge_fallback"]["uses_per_leg"] is None and "no judge_chain tally" in pf["judge_fallback"]["source"]
        assert co.petri_sidecar_proof(cells, read, judge_chain_in_prior=False)["judge_fallback"] == "n/a (not in prior tree)"


class TestD113AdmittedHunks:
    FK2_TIP = "b6754f7ba"                   # the assembly tip with fk2b merged (byte-identical to fk2's branch tip 1862aea8c on these files)
    FILES = ("src/sisteer/capeval/petri/auditor_fallback.py", "src/sisteer/capeval/manifest.py", "src/sisteer/capeval/tiers/petri_runner.py")

    def test_the_three_judge_chain_files_are_pinned_and_labelled(self):
        for f in self.FILES:
            assert co.D113_ADMITTED_HUNKS[f] and all(label for label in co.D113_ADMITTED_HUNKS[f].values()), f
        assert not set(co.D113_ADMITTED_HUNKS["src/sisteer/capeval/manifest.py"]) & set(co.D111_ADMITTED_HUNKS["src/sisteer/capeval/manifest.py"])

    @pytest.mark.skipif(not all(_sha_exists(s) for s in ("c65b47ba4", "56aff576e", "b6754f7ba")), reason="the W19 / W22 / W23 shas are not in this clone")
    def test_the_real_w22_and_w19_to_w23_diffs_are_exactly_the_pinned_d111_plus_d113_sets(self):
        git = co.default_git(REPO)
        for src in co.D111_PIN_SOURCES:
            for f in self.FILES:
                adm = co.admitted_difference(git, src, self.FK2_TIP, f)
                assert adm is not None and adm["kind"] == co.ADMITTED_KIND_CHAIN, (src, f)
                pinned = set(co.D111_ADMITTED_HUNKS.get(f, {})) | set(co.D113_ADMITTED_HUNKS[f])
                assert {h["sha256"] for h in adm["hunks"]} <= pinned and "D113" in adm["tables"], (src, f)
                assert all(h["table"] in ("D111", "D113") and h["label"] for h in adm["hunks"])
            # the drive carries D111 only at the assembly tip
            adm = co.admitted_difference(git, src, self.FK2_TIP, "scripts/modal/seedrep_petri_drive.py")
            assert adm["tables"] == ["D111"]
        # W22 -> W23 is admitted for BOTH Petri sources: the whole gating set of a Petri row differs only by admitted hunks + chain-list manifests
        rc, _ = git(["diff", "--quiet", "56aff576e", self.FK2_TIP, "--", *self.FILES])
        assert rc == 1

    @pytest.mark.skipif(not _sha_exists("b6754f7ba"), reason="the W23 assembly tip is not in this clone")
    def test_a_hunk_outside_the_judge_chain_code_refuses_the_file(self):
        git = co.default_git(REPO)
        f = "src/sisteer/capeval/tiers/petri_runner.py"
        _rc, a = git(["show", f"56aff576e:{f}"])
        _rc, b = git(["show", f"{self.FK2_TIP}:{f}"])
        assert co.d111_admitted_difference(a, b, f) is not None
        assert co.d111_admitted_difference(a, b + "\nUNRELATED = 1\n", f) is None
        assert co.d111_admitted_difference(a, b.replace("JUDGE_FALLBACK_METADATA_KEY", "JUDGE_FALLBACK_METADATA_KEY_X", 1), f) is None



class TestD120AdmittedHunks:
    """opus-4-8 before grok-4.6 in both chains (tip 8681c01bd = 1f2fd34bb + the generator rules 7e322a766)."""
    TIP = "8681c01bd"
    CODE = ("src/sisteer/capeval/petri/auditor_fallback.py", "src/sisteer/capeval/manifest.py", "src/sisteer/capeval/tiers/petri_runner.py", "scripts/modal/seedrep_petri_drive.py")
    MANIFESTS = ("configs/capeval/manifests/petri_full22_qwen35_4b_s1.yaml", "configs/capeval/manifests/petri_full22_qwen35_4b_s2.yaml",
                 "configs/capeval/manifests/petri_full22_qwen35_4b_s3.yaml", "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_manifest.yaml")

    @pytest.mark.skipif(not all(_sha_exists(s) for s in ("c65b47ba4", "56aff576e", "8681c01bd")), reason="the W19 / W22 / fk2c shas are not in this clone")
    def test_both_sources_are_admitted_against_the_d120_tip_with_every_decision_named_and_an_unrelated_hunk_refused(self):
        git = co.default_git(REPO)
        assert [t[0] for t in co.ADMITTED_HUNK_TABLES][:3] == ["D111", "D113", "D120"]
        for src in co.D111_PIN_SOURCES:
            adm = co.admitted_difference(git, src, self.TIP, "src/sisteer/capeval/petri/auditor_fallback.py")
            assert adm is not None and adm["tables"] == ["D111", "D113", "D120"], (src, adm and adm.get("tables"))
            assert any(h["table"] == "D120" and "opus-4-8" in h["label"] for h in adm["hunks"])
            for f in self.CODE[1:]:
                adm = co.admitted_difference(git, src, self.TIP, f)
                assert adm is not None and adm["kind"] == co.ADMITTED_KIND_CHAIN and "D120" not in adm["tables"], (src, f)
            for m in self.MANIFESTS:
                adm = co.admitted_difference(git, src, self.TIP, m)
                assert adm is not None and adm["kind"] == co.ADMITTED_KIND_CHAIN, (src, m)
                assert "- anthropic/claude-opus-4-8" in adm["added"] and "- openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated" in adm["removed"]
                assert set(adm["keys_changed"]) <= {"auditor_fallback_models", "judge_fallback_models"} and "judge_fallback_models" in adm["keys_changed"]
            # the four plan manifests at the tip are the regenerated ones the ruling names
            for m, sha16 in zip(self.MANIFESTS, ("412d0b79a40039c6", "1ec11f221765bf88", "96d789b6f2704157", "0f85baba20caa796")):
                rc, text = git(["show", f"{self.TIP}:{m}"])
                assert rc == 0 and hashlib.sha256(text.encode()).hexdigest()[:16] == sha16, m
        # an unrelated hunk in the D120 file, or a non-chain line in a manifest, still refuses
        _rc, a = git(["show", f"56aff576e:src/sisteer/capeval/petri/auditor_fallback.py"])
        _rc, b = git(["show", f"{self.TIP}:src/sisteer/capeval/petri/auditor_fallback.py"])
        assert co.d111_admitted_difference(a, b + "\nUNRELATED = 1\n", "src/sisteer/capeval/petri/auditor_fallback.py") is None
        _rc, ma = git(["show", f"56aff576e:{self.MANIFESTS[0]}"])
        _rc, mb = git(["show", f"{self.TIP}:{self.MANIFESTS[0]}"])
        import re as _re
        mutated = _re.sub(r"^(\s*judge_model:\s*).*$", r"\1anthropic/claude-opus-4-8", mb, count=1, flags=_re.M)
        assert mutated != mb and "judge_model: anthropic/claude-opus-4-8" in mutated
        fake = F.FakeGit({W_PREV: {self.MANIFESTS[0]: "v1"}, W_THIS: {self.MANIFESTS[0]: "v2"}}, {(W_PREV, self.MANIFESTS[0]): ma, (W_THIS, self.MANIFESTS[0]): mutated})
        assert co.admitted_difference(fake, W_PREV, W_THIS, self.MANIFESTS[0]) is None       # the judge MODEL changed: not a chain line


class TestD136AdmittedHunks:
    """fj9 (D136, fr5 W24): the registry-emission hunks admitted; the recorder module NON_EVAL."""
    W24_BASE, W24_TIP, W23_ASM = "ff96b7210", "5e0b3c9b0", "2174e5139"
    STORE_SIDE = ("scripts/served_capability_run.py", "scripts/cell_controls_run.py", "src/sisteer/capeval/store.py")
    PETRI = ("src/sisteer/capeval/petri/auditor_fallback.py", "src/sisteer/capeval/manifest.py", "src/sisteer/capeval/tiers/petri_runner.py", "scripts/modal/seedrep_petri_drive.py")

    def test_the_recorder_module_is_non_eval_and_the_tables_name_four_decisions(self):
        assert "src/sisteer/registry/hooks.py" not in co.NON_EVAL_MODULES and len(co.D136_ADMITTED_HUNKS["src/sisteer/registry/hooks.py"]) == 13   # fj11: gating, pinned
        assert [t[0] for t in co.ADMITTED_HUNK_TABLES][:4] == ["D111", "D113", "D120", "D136"]
        assert set(co.D136_ADMITTED_HUNKS) == set(self.STORE_SIDE) | {"src/sisteer/registry/hooks.py"} and all(len(co.D136_ADMITTED_HUNKS[f]) == 1 for f in self.STORE_SIDE)

    @pytest.mark.skipif(not all(_sha_exists(s) for s in ("ff96b7210", "5e0b3c9b0", "2174e5139", "56aff576e", "c65b47ba4")), reason="the W23 / W24 shas are not in this clone")
    def test_fr5s_store_side_hunks_are_admitted_from_the_w24_base_and_an_unrelated_hunk_refuses(self):
        git = co.default_git(REPO)
        for f in self.STORE_SIDE:
            # the W23 assembly's blobs equal the W24 base's on these files: the W23 wave's rows meet exactly fr5's hunk on W24
            _rc, a = git(["rev-parse", f"{self.W23_ASM}:{f}"])
            _rc, b = git(["rev-parse", f"{self.W24_BASE}:{f}"])
            assert a.strip() == b.strip(), f
            adm = co.admitted_difference(git, self.W24_BASE, self.W24_TIP, f)
            assert adm is not None and adm["tables"] == ["D136"] and adm["kind"] == co.ADMITTED_KIND_REGISTRY and adm["n_hunks"] == 1, (f, adm)
            assert "note_generated" in adm["hunks"][0]["label"]
            _rc, x = git(["show", f"{self.W24_BASE}:{f}"])
            _rc, y = git(["show", f"{self.W24_TIP}:{f}"])
            assert co.d111_admitted_difference(x, y + "\nUNRELATED = 1\n", f) is None                          # an extra hunk refuses the file
            assert co.d111_admitted_difference(x, y.replace("note_generated(spec.task", "note_generated(spec.task2", 1) if "spec.task" in y else y + "\n#x\n", f) is None
        # the whole controls / gsm8k producer set from the W23 assembly to fr5's tip: only the three pinned files + the NON_EVAL recorder differ
        v = co.producer_set_verdict(["scripts/sdfbase/mt_serving_up.py", *self.STORE_SIDE, "src/sisteer/registry/hooks.py", co.PROXY_REL, co.LEDGER_REL],
                                    prev_sha=self.W23_ASM, this_sha=self.W24_TIP, git=git)
        assert v["ok"], v.get("why")
        assert {a["path"] for a in v["admitted_differences"]} == set(self.STORE_SIDE) | {"src/sisteer/registry/hooks.py"} and not v["non_eval_modules"] or all(n["identical"] for n in v["non_eval_modules"])
        # both Petri sources against the W24 tip: D111 + D113 + D120 named on the auditor module / manifest.py / petri_runner.py, no D136 (fr5
        # touched no Petri producer). The DRIVE is REFUSED from both sources at this tip: fs2's F-AR (44aa5b1d1, the local relay port) is
        # grading-host plumbing, not row-shaping code, but it is not a NAMED admission -- pinned only under a named decision
        for src in co.D111_PIN_SOURCES:
            adm = co.admitted_difference(git, src, self.W24_TIP, "src/sisteer/capeval/petri/auditor_fallback.py")
            assert adm is not None and adm["tables"] == ["D111", "D113", "D120"], (src, adm and adm.get("tables"))
            for f in self.PETRI[1:3]:
                adm = co.admitted_difference(git, src, self.W24_TIP, f)
                assert adm is not None and "D136" not in adm["tables"], (src, f)
            assert co.admitted_difference(git, src, self.W24_TIP, "scripts/modal/seedrep_petri_drive.py")["tables"] == ["D111", "D137"]   # F-AR admitted (fj10)
            assert co.admitted_difference(git, src, "8681c01bd", "scripts/modal/seedrep_petri_drive.py")["tables"] == ["D111"]           # ... D111 alone at the D120 tip
        # the earlier wave's controls / gsm8k rows stay refused on served_capability_run.py: the probe + driver change is not an admitted
        # difference (file-level gating) -- the admission table rescues the later source, not the earlier one
        assert co.admitted_difference(git, "56aff576e", self.W24_TIP, "scripts/served_capability_run.py") is None


class TestD137AdmittedHunks:
    """fj9 (D137): fs2's F-AR relay-port plumbing in the Petri drive admitted for both Petri sources at the W24 tip."""
    W24_TIP = "5e0b3c9b0"
    DRIVE = "scripts/modal/seedrep_petri_drive.py"

    @pytest.mark.skipif(not all(_sha_exists(s) for s in ("5e0b3c9b0", "56aff576e", "c65b47ba4", "8681c01bd")), reason="the W19 / W22 / W23 / W24 shas are not in this clone")
    def test_both_petri_sources_admit_the_drive_at_the_w24_tip_and_an_unrelated_drive_hunk_refuses(self):
        git = co.default_git(REPO)
        assert [t[0] for t in co.ADMITTED_HUNK_TABLES][:5] == ["D111", "D113", "D120", "D136", "D137"] and len(co.D137_ADMITTED_HUNKS[self.DRIVE]) == 5
        union = set()
        for src in co.D111_PIN_SOURCES:
            adm = co.admitted_difference(git, src, self.W24_TIP, self.DRIVE)
            assert adm is not None and adm["tables"] == ["D111", "D137"] and adm["kind"] == f"{co.ADMITTED_KIND_PLUMBING}+{co.ADMITTED_KIND_CHAIN}", (src, adm and adm.get("tables"))
            assert {h["sha256"] for h in adm["hunks"] if h["table"] == "D137"} == set(co.D137_ADMITTED_HUNKS[self.DRIVE])
            union |= set(adm["tables"])
            for f in ("src/sisteer/capeval/petri/auditor_fallback.py", "src/sisteer/capeval/manifest.py", "src/sisteer/capeval/tiers/petri_runner.py"):
                adm = co.admitted_difference(git, src, self.W24_TIP, f)
                assert adm is not None, (src, f)
                union |= set(adm["tables"])
        assert union == {"D111", "D113", "D120", "D137"}                  # the whole Petri producer set: every decision named, nothing else
        _rc, a = git(["show", f"56aff576e:{self.DRIVE}"])
        _rc, b = git(["show", f"{self.W24_TIP}:{self.DRIVE}"])
        assert co.d111_admitted_difference(a, b + "\nUNRELATED = 1\n", self.DRIVE) is None
        assert co.d111_admitted_difference(a, b.replace("LOCAL_PORT_MIN = 1024", "LOCAL_PORT_MIN = 2048", 1), self.DRIVE) is None


    def test_a_leg_restricted_table_is_seen_only_by_carries_of_its_legs(self, monkeypatch):
        """fj10: the D138 machinery -- a table restricted to LEGS admits a hunk for a carry of one of those legs and for no other; the D138 table
        itself is EMPTY until fd3's served_capability_run.py hunks are ruled on (fj10's verification: they execute for controls)."""
        path = "scripts/served_capability_run.py"
        assert co.D138_ADMITTED_HUNKS[path] == {} and [t[0] for t in co.ADMITTED_HUNK_TABLES][5] == "D138" and co.ADMITTED_HUNK_TABLES[5][4] == frozenset({"controls"})
        a, b = "x = 1\ny = 2\n", "x = 1\nprobe()\ny = 2\n"
        h = co.diff_hunks(a, b)[0]["sha256"]
        monkeypatch.setitem(co.D138_ADMITTED_HUNKS[path], h, "synthetic: the probe call")
        assert co.d111_admitted_difference(a, b, path, "controls")["tables"] == ["D138"] and co.d111_admitted_difference(a, b, path, "controls")["kind"] == co.ADMITTED_KIND_OTHER_LEG
        assert co.d111_admitted_difference(a, b, path, "gsm8k") is None and co.d111_admitted_difference(a, b, path, None) is None
        git = F.FakeGit({W_PREV: {path: "v1"}, W_THIS: {path: "v2"}}, {(W_PREV, path): a, (W_THIS, path): b})
        assert co.producer_set_verdict([path], prev_sha=W_PREV, this_sha=W_THIS, git=git, leg="controls")["ok"]
        assert not co.producer_set_verdict([path], prev_sha=W_PREV, this_sha=W_THIS, git=git, leg="gsm8k")["ok"]

    @pytest.mark.skipif(not all(_sha_exists(s) for s in ("56aff576e", "5e0b3c9b0")), reason="the W22 / W24 shas are not in this clone")
    def test_fd3s_served_capability_run_hunks_stay_refused_for_both_legs_pending_the_ruling(self):
        git = co.default_git(REPO)
        for leg in ("controls", "gsm8k", None):
            assert co.admitted_difference(git, "56aff576e", "5e0b3c9b0", "scripts/served_capability_run.py", leg) is None, leg


# --------------------------------------------------------------------------- fj12: the S3 FULL-shape source (W19's Petri legs live under the S3 wave's Dict)

W19_REAL = "c65b47ba4e3def0fda7159f497e60900b8ec4631"
W19_PROD_WAVE = "bad224cf-9d2d-4882-905b-c715bc83c8cf"
W19_S3_WAVE = "c3b864ed-2ba3-48fb-90f4-1987f26e6cdf"
W19_PETRI_AID = "01M2WNTNMW8AQ7CJHTC1D3Y5SR-79e88ba275a1"
W19_S3_DICT = "seedrep-c65b47b-c3b864ed-accepted"
W19_S3_KEY = "c65b47b|c3b864ed-2ba3-48fb-90f4-1987f26e6cdf|petri|s1|base|0"


def _petri_row(W, W7, *, wave):
    r = F._row(W, W7, sid="petri__s1__base__shard00", leg="petri")
    k0 = r["children"][0]
    r["children"] = [{**copy.deepcopy(k0), "arm": f"a{i}", "cell": f"a{i}", "argv": None, "grading_host_drive": {"cell_index": i, "target_decode": "vendor", "target_seed_base": 1}} for i in range(8)]
    r["control_key"] = lp.control_key(W7, wave, "petri", "s1", "base", 0)
    r["accepted_key"] = lp.accepted_key(W7, wave, "petri", "s1", "base", 0)
    return r


def _plan_at(W, *, wave, rows, materialisation):
    plan = F._plan(W, materialisation=materialisation, rows=rows)
    plan["wave_uuid"] = wave
    plan["dicts"] = {p: lp.dict_name(W[:7], wave, p) for p in lp.DICT_PURPOSES}
    plan["launch_plan_sha"] = lp.plan_sha(plan)
    return plan


class TestS3FullShapeSource:
    def _w19_world(self, tmp_path):
        """The REAL W19 acceptance shape: the eight Petri legs of view s1 completed as the S3 smoke and were accepted under the S3 WAVE's Dict
        (`seedrep-c65b47b-c3b864ed-accepted`, key `c65b47b|<S3 wave>|petri|s1|base|0`); the production Dict of that tree names nothing; the root
        lacks the determinism record (F-AA) -- carried under the D118 waiver."""
        prev = _plan_at(W19_REAL, wave=W19_PROD_WAVE, rows=[_petri_row(W19_REAL, W19_REAL[:7], wave=W19_PROD_WAVE)], materialisation="draft")
        this = _plan_at(W_THIS, wave=WAVE, rows=[_petri_row(W_THIS, W_THIS[:7], wave=WAVE)], materialisation="final")
        prow = prev["standups"][0]
        sident = lp.smoke_identity(prev, "S3", W19_S3_WAVE)
        assert sident["dicts"]["accepted"] == W19_S3_DICT
        srow = lp.smoke_row(prow, sident, max_children=8, max_timeout_s=int(prow["modal"]["timeout_s"]))
        assert srow["accepted_key"] == W19_S3_KEY and srow["standup_id"] == "smoke-S3__petri__s1__base__shard00" and srow["smoke"]["budget"]["kind"] == lp.SMOKE_BUDGET_KIND_ITEM
        assert not any(f in (k.get("argv") or []) for k in srow["children"] for f in lp.SMOKE_BUDGET_FLAGS)      # Petri children carry no D107 flag
        files, dicts = {}, {}
        root = F._root_files(files, plan=prev, row=srow, identity=sident, aid=W19_PETRI_AID, smoke_block=srow["smoke"], n_children=8)
        k0 = srow["children"][0]
        rel0 = lp.audit_verdict_rel(k0["eval_leg"], k0["arm"], k0["s"], stage=srow["stage"], shard=srow["shard"])
        ev = json.loads(files[f"{root}/modal_evidence.json"])
        for k in srow["children"][1:]:
            rel = lp.audit_verdict_rel(k["eval_leg"], k["arm"], k["s"], stage=srow["stage"], shard=srow["shard"])
            doc = json.loads(files[f"{root}/{rel0}"])
            doc["arm"] = k["arm"]
            doc["arm_binding"] = {**doc["arm_binding"], "arm": k["arm"], "harness_id": k["arm"], "traintest_id": k["arm"], "vmodel": f"petri__{k['arm']}"}
            text = json.dumps(doc)
            files[f"{root}/{rel}"] = text
            ev[lp.PRODUCER_EVIDENCE_KEY][rel] = {**ev[lp.PRODUCER_EVIDENCE_KEY][rel0], "sha256": F._sha(text)}
        files[f"{root}/modal_evidence.json"] = json.dumps(ev)
        del files[f"{root}/{lp.determinism_rels(srow)[0][1]}"]                       # the W19 shape (F-AA)
        dicts[W19_S3_DICT] = {W19_S3_KEY: W19_PETRI_AID}                               # the production Dict is ABSENT: `dicts.get(name, {})` -> None
        d = F._prior_dir(tmp_path, prev)
        (d / "smokes" / "S3").mkdir(parents=True, exist_ok=True)
        (d / "smokes" / "S3" / "wave.json").write_text(json.dumps({"smoke_id": "S3", "wave_uuid": W19_S3_WAVE, "minted_at": "2026-09-19T05:10:33Z"}))
        prior = co.load_prior_source(d)
        assert prior["smoke_waves"] == {"S3": W19_S3_WAVE}
        blobs = {W19_REAL: F._blobs(W19_REAL), W_THIS: F._blobs(W_THIS)}
        for sha in blobs:
            for f in co.LEG_HELPERS["petri"]:
                blobs[sha][f] = "same"
        waiver = {"by": "user", "at": "2026-09-19T20:15Z", "scope": {"W": "c65b47ba4", "leg": "petri", "view": "s1", "legs": list(range(8))},
                  "reason": "F-AA refused the in-container check; engine reproducibility proven on the same build by the W20/W22 rows", "decision": "D118"}
        return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, git=F.FakeGit(blobs), prior=prior, root=root, waiver=waiver, sident=sident)

    def test_w19s_petri_legs_carry_from_the_s3_wave_dict_under_the_named_clause(self, tmp_path):
        w = self._w19_world(tmp_path)
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                     git=w.git, readiness_problems=F.fake_readiness, criterion3_waivers=[w.waiver])
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"]["petri__s1__base__shard00"]
        assert e["attempt_id"] == W19_PETRI_AID and e["clauses"]["row_identity"]["clause"] == co.S3_FULL_CLAUSE and e["full_smoke_row"] and not e["s5_full_row"]
        cf = e["carried_from"]
        assert (cf["source"], cf["dict"], cf["accepted_key"], cf["wave_uuid"], cf["standup_id"]) == ("s3_smoke", W19_S3_DICT, W19_S3_KEY, W19_S3_WAVE, "smoke-S3__petri__s1__base__shard00")
        assert cf["W"] == W19_REAL and e["attempt_root"].startswith(f"{w.prev['results_root']}_smoke/S3/") and e["criterion3_waiver"]["decision"] == "D118"
        # KNOWN-BAD pinned in the record: the production identity was tried first and holds nothing (the W19 Dict is the S3 wave's)
        assert [a["source"] for a in e["refused_before"]] == ["production"] and "no ACCEPTED attempt" in e["refused_before"][0]["why"]
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        # the coverage binder's mapping accepts the s3_smoke source with its smoke-S3__ spelling
        ac = _ac()                                                                              # fj27: one-file: this module puts scripts/sdfbase/seedrep on sys.path itself
        row = w.this["standups"][0]
        plan = ac.LaunchPlan(path=Path("plan"), W=W_THIS, wave_uuid=WAVE, rows={row["standup_id"]: {**row, "view": "s1", "stage": "base", "shard": "shard00"}})
        m = ac.carry_mapping(rec, plan)
        assert m[W19_PETRI_AID].source == "s3_smoke" and m[W19_PETRI_AID].historical_standup_id == "smoke-S3__petri__s1__base__shard00" and m[W19_PETRI_AID].wave_uuid == W19_S3_WAVE
        # a record whose row identity says S3 but whose provenance says production refuses at certify
        bad = copy.deepcopy(rec)
        bad["carried"]["petri__s1__base__shard00"]["carried_from"]["source"] = "production"
        assert any("not the smoke" in p for p in co.carry_over_problems(bad, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))

    def test_a_budgeted_bon_s3_row_is_refused_by_name_and_s1_s2_s4_stay_refused(self, tmp_path):
        prev = _plan_at(W19_REAL, wave=W19_PROD_WAVE, rows=[F._row(W19_REAL, W19_REAL[:7], sid="bon__s1__base__shard00", leg="bon")], materialisation="draft")
        this = _plan_at(W_THIS, wave=WAVE, rows=[F._row(W_THIS, W_THIS[:7], sid="bon__s1__base__shard00", leg="bon")], materialisation="final")
        prow = prev["standups"][0]
        sident = lp.smoke_identity(prev, "S3", W19_S3_WAVE)
        srow = lp.smoke_row(prow, sident, max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
        assert any(a.startswith(lp.SMOKE_BON_LIMIT_FLAG) or a == lp.SMOKE_BON_LIMIT_FLAG for k in srow["children"] for a in k["argv"])   # bon at 10 goals
        v = co.row_identity_verdict(srow, this["standups"][0], prev={"W": W19_REAL, "W7": W19_REAL[:7], "wave_uuid": W19_PROD_WAVE},
                                    this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE}, prev_smoke_wave=W19_S3_WAVE)
        assert not v["ok"] and "BUDGETED" in v["why"] and lp.SMOKE_BON_LIMIT_FLAG in v["why"]
        for sid in ("S1", "S2", "S4"):
            ident = lp.smoke_identity(prev, sid, W19_S3_WAVE)
            srow2 = lp.smoke_row(_petri_row(W19_REAL, W19_REAL[:7], wave=W19_PROD_WAVE), ident, max_children=8, max_timeout_s=57600)
            v2 = co.row_identity_verdict(srow2, _petri_row(W_THIS, W_THIS[:7], wave=WAVE), prev={"W": W19_REAL, "W7": W19_REAL[:7], "wave_uuid": W19_PROD_WAVE},
                                         this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE}, prev_smoke_wave=W19_S3_WAVE)
            assert not v2["ok"] and f"ran as smoke {sid!r}" in v2["why"]
        # through the evaluator: the bon S3 candidate is tried and refused by name, nothing carries
        files, dicts = {}, {}
        F._root_files(files, plan=prev, row=srow, identity=sident, smoke_block=srow["smoke"])
        dicts[sident["dicts"]["accepted"]] = {srow["accepted_key"]: AID}
        d = F._prior_dir(tmp_path, prev)
        (d / "smokes" / "S3").mkdir(parents=True, exist_ok=True)
        (d / "smokes" / "S3" / "wave.json").write_text(json.dumps({"smoke_id": "S3", "wave_uuid": W19_S3_WAVE}))
        rec = co.evaluate_carry_over(this_plan=this, this_sha=W_THIS, priors=[co.load_prior_source(d)], read_text=files.get, accepted_get_for_dict=lambda n: dicts.get(n, {}).get,
                                     git=F.FakeGit({W19_REAL: F._blobs(W19_REAL), W_THIS: F._blobs(W_THIS)}), readiness_problems=F.fake_readiness)
        r = rec["refused"]["bon__s1__base__shard00"]
        assert rec["n_carried"] == 0 and [a["source"] for a in r["attempts"]] == ["production", "s3_smoke"] and "BUDGETED" in r["attempts"][1]["why"]

class TestD136bHooksGating:
    """fj11: the registry recorder shapes persisted manifest bytes (the driver probes `run_payload` for the row constants it
    lifts), so hooks.py is GATING; its reviewed emission-only changes admit through exact pins; anything else refuses."""
    H = "src/sisteer/registry/hooks.py"

    @pytest.mark.skipif(not all(_sha_exists(s) for s in ("56aff576e", "c65b47ba4", "ff96b7210", "5e0b3c9b0")), reason="the W19 / W22 / W23 / W24 shas are not in this clone")
    def test_fr2_and_fr5_hooks_hunks_admit_by_pin_from_every_source_and_an_unrelated_change_refuses(self):
        git = co.default_git(REPO)
        for src, n in (("56aff576e", 13), ("c65b47ba4", 13), ("ff96b7210", 10)):
            adm = co.admitted_difference(git, src, "5e0b3c9b0", self.H)
            assert adm is not None and adm["tables"] == ["D136"] and adm["kind"] == co.ADMITTED_KIND_REGISTRY and adm["n_hunks"] == n, (src, adm and adm.get("n_hunks"))
            assert all(("fr2" in h["label"] or "fr5" in h["label"]) for h in adm["hunks"])
        _rc, a = git(["show", f"56aff576e:{self.H}"])
        _rc, b = git(["show", f"5e0b3c9b0:{self.H}"])
        assert co.d111_admitted_difference(a, b + "\nUNRELATED = 1\n", self.H) is None
        # a change to the store-consumed surface the driver probes (`run_payload` / `_seed_base`) is never admitted
        assert co.d111_admitted_difference(a, b.replace('"seed_base": self._seed_base(),', '"seed_base": None,', 1), self.H) is None
        # the whole controls / gsm8k producer set from the W23 assembly to fr5's tip still admits, hooks.py now GATING and pinned (not exempt)
        v = co.producer_set_verdict(["scripts/sdfbase/mt_serving_up.py", "scripts/served_capability_run.py", "scripts/cell_controls_run.py", "src/sisteer/capeval/store.py", self.H, co.PROXY_REL, co.LEDGER_REL],
                                    prev_sha="2174e5139", this_sha="5e0b3c9b0", git=git)
        assert v["ok"] and self.H in {x["path"] for x in v["admitted_differences"]}

    @pytest.mark.skipif(not (_sha_exists("c98306e60") and _sha_exists("5e0b3c9b0")), reason="the historical recorder commit is not in this clone")
    def test_the_historical_recorder_change_sol_reproduced_refuses_again(self):
        """Pre-c98306e60 hooks (no `run.seed_base` fallback to the row constants -> a top-level `seed` in the persisted manifest) against the W24
        tip: the seed_base / row-constant hunks are not pinned, so the recorder difference REFUSES -- as the old (gating) reader did before fj9's
        exemption admitted it without a pin."""
        git = co.default_git(REPO)
        assert co.admitted_difference(git, "c98306e60^", "5e0b3c9b0", self.H) is None
        assert co.admitted_difference(git, "c98306e60^", "c98306e60", self.H) is None
        v = co.producer_set_verdict([self.H], prev_sha="c98306e60^", this_sha="5e0b3c9b0", git=git, leg="controls")
        assert not v["ok"] and self.H in (v.get("why") or "")


class TestBudgetedSmokeAcceptanceNeverHolds:
    """fj13: an accepted BUDGETED smoke is completed smoke work, not the production row's -- its carry refusal is recorded but it
    never establishes a `carry_refused_hold`; a FULL-shape acceptance (production, S5, W19's eight-leg Petri S3) still does."""

    def _bon_s3_world(self, tmp_path):
        prev = _plan_at(W19_REAL, wave=W19_PROD_WAVE, rows=[F._row(W19_REAL, W19_REAL[:7], sid="bon__s1__base__shard00", leg="bon")], materialisation="draft")
        this = _plan_at(W_THIS, wave=WAVE, rows=[F._row(W_THIS, W_THIS[:7], sid="bon__s1__base__shard00", leg="bon")], materialisation="final")
        prow = prev["standups"][0]
        sident = lp.smoke_identity(prev, "S3", W19_S3_WAVE)
        srow = lp.smoke_row(prow, sident, max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
        files, dicts = {}, {}
        root = F._root_files(files, plan=prev, row=srow, identity=sident, smoke_block=srow["smoke"])
        for k in srow["children"]:                                                       # fj15: the executed argv on record agrees with the budget (10 goals)
            files[f"{root}/children/{k['eval_leg']}__{k['arm']}__s{k['s']}.argv.json"] = json.dumps({"argv": list(k["argv"])})
        dicts[sident["dicts"]["accepted"]] = {srow["accepted_key"]: AID}            # the 10-goal S3 BoN smoke IS accepted; the production Dict is empty
        d = F._prior_dir(tmp_path, prev)
        (d / "smokes" / "S3").mkdir(parents=True, exist_ok=True)
        (d / "smokes" / "S3" / "wave.json").write_text(json.dumps({"smoke_id": "S3", "wave_uuid": W19_S3_WAVE}))
        return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, prior=co.load_prior_source(d),
                                     git=F.FakeGit({W19_REAL: F._blobs(W19_REAL), W_THIS: F._blobs(W_THIS)}))

    def test_an_accepted_budgeted_bon_s3_smoke_is_refused_not_held_and_the_production_row_spawns(self, tmp_path, srv, capsys):
        w = self._bon_s3_world(tmp_path)
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                     git=w.git, readiness_problems=F.fake_readiness)
        r = rec["refused"]["bon__s1__base__shard00"]
        s3 = next(a for a in r["attempts"] if a["source"] == "s3_smoke")
        assert "BUDGETED" in s3["why"] and s3["clause"] == "row_identity"                 # the refusal is explicit and recorded ...
        paa = s3["prior_accepted_attempt"]
        assert paa["attempt_id"] == AID and paa["full_shape"] is False and paa["holds"] is False and any("D107" in p for p in paa["candidate_shape_problems"])
        assert paa["executed_shape"]["shape"] == co.EXECUTED_SHAPE_LIMITED and paa["executed_shape"]["executed"]["value"] == "10"   # fj14 / fj15: the EXECUTED argv, the budget agreeing
        assert co.carry_refused_holds(rec) == {}                                        # ... but it is no hold: (i)
        picked, _ = srv.select_launch_rows(w.this, go="bon__s1__base__shard00", carry_holds=co.carry_refused_holds(rec))
        assert [x["standup_id"] for x in picked] == ["bon__s1__base__shard00"]          # the production row runs
        assert "HOLD" not in capsys.readouterr().out

    def test_a_full_shape_s3_acceptance_carries_and_still_holds_when_only_its_identity_fails(self, tmp_path, srv):
        w = TestS3FullShapeSource()._w19_world(tmp_path)
        kw = dict(this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get, git=w.git,
                  readiness_problems=F.fake_readiness, criterion3_waivers=[w.waiver])
        rec = co.evaluate_carry_over(this_plan=w.this, **kw)
        assert rec["n_carried"] == 1                                                         # (ii) the eight-leg Petri S3 carries
        # (iii) the same completed full-shape attempt whose row identity fails is HELD, never spawned. (A smoke source's `modal.timeout_s` is
        # SET ASIDE by clause (a) -- the smoke's container bound -- so the difference here is scientific: this plan's row caps at 2048.)
        w.this["standups"][0]["geometry"]["cap"] = 2048
        w.this["launch_plan_sha"] = lp.plan_sha(w.this)
        rec2 = co.evaluate_carry_over(this_plan=w.this, **kw)
        r = rec2["refused"]["petri__s1__base__shard00"]
        s3 = next(a for a in r["attempts"] if a["source"] == "s3_smoke")
        assert s3["clause"] == "row_identity" and "/geometry/cap" in s3["why"] and s3["prior_accepted_attempt"]["full_shape"] is True and s3["prior_accepted_attempt"]["holds"] is True
        holds = co.carry_refused_holds(rec2)
        assert holds["petri__s1__base__shard00"]["attempt_id"] == W19_PETRI_AID
        with pytest.raises(srv.LaunchRefused, match="every selected row is HELD"):
            srv.select_launch_rows(w.this, go="petri__s1__base__shard00", carry_holds=holds)
        # a production acceptance (no smoke block) is full-shape by definition: the fj8 protection is untouched
        assert co.smoke_shape_problems(w.this["standups"][0], w.this["standups"][0]) == []


class TestExecutedShapeBoundToTheAttempt:
    """fj14: the hold's `full_shape` comes from evidence BOUND to the accepted attempt (its recorded children / budget / per-child
    argv against the PRIOR tree's production row), never from the candidate today's `smoke_row` reconstructs."""

    def _world(self, tmp_path, *, budget_block, child_argv=None, drop_evidence=False):
        w = TestBudgetedSmokeAcceptanceNeverHolds()._bon_s3_world(tmp_path)
        root = next(p for p in w.files if p.endswith("/modal_evidence.json"))[: -len("/modal_evidence.json")]
        ev = json.loads(w.files[f"{root}/modal_evidence.json"])
        ev["smoke"] = {k: v for k, v in ev["smoke"].items() if k != "budget"}         # the W19-era container wrote NO budget block
        if budget_block is not None:
            ev["smoke"]["budget"] = budget_block
        w.files[f"{root}/modal_evidence.json"] = json.dumps(ev)
        for p in [q for q in w.files if q.startswith(f"{root}/children/") and q.endswith(".argv.json")]:
            del w.files[p]                                                                   # the fj13 world's flagged argv: this world spells its own
        if child_argv is not None:
            for k in ev["children"]:
                w.files[f"{root}/children/{k['eval_leg']}__{k['arm']}__s{k['s']}.argv.json"] = json.dumps({"argv": child_argv})
        if drop_evidence:
            del w.files[f"{root}/modal_evidence.json"]
        w.root = root
        return w

    def _eval(self, w):
        return co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                      git=w.git, readiness_problems=F.fake_readiness)

    def test_a_pre_d107_full_budget_acceptance_is_held_although_todays_reconstruction_carries_the_flag(self, tmp_path, srv):
        w = self._world(tmp_path, budget_block=None, child_argv=["/venv/bin/python", "scripts/sdfbase/full22/bon_served_run.py", "--cap", "4096"])
        rec = self._eval(w)
        r = rec["refused"]["bon__s1__base__shard00"]
        s3 = next(a for a in r["attempts"] if a["source"] == "s3_smoke")
        assert "BUDGETED" in s3["why"]                                                        # today's reconstruction carries --limit-goals ...
        paa = s3["prior_accepted_attempt"]
        assert paa["candidate_shape_problems"] and paa["executed_shape"]["shape"] == co.EXECUTED_SHAPE_FULL and paa["full_shape"] is True and paa["holds"] is True
        assert paa["executed_shape"]["source"] == "children/*.argv.json"                       # ... but the attempt EXECUTED the full budget
        holds = co.carry_refused_holds(rec)
        assert holds["bon__s1__base__shard00"]["attempt_id"] == AID and holds["bon__s1__base__shard00"]["executed_shape"]["shape"] == "full"
        with pytest.raises(srv.LaunchRefused, match="every selected row is HELD"):            # canonical work: held, never re-run
            srv.select_launch_rows(w.this, go="bon__s1__base__shard00", carry_holds=holds)

    def test_a_genuine_ten_goal_acceptance_is_refused_and_not_held(self, tmp_path, srv):
        w = self._world(tmp_path, budget_block={"kind": lp.SMOKE_BUDGET_KIND_ITEM, "bon_limit_goals": 10, "misalign_limit_items": None},
                        child_argv=["scripts/sdfbase/full22/bon_served_run.py", lp.SMOKE_BON_LIMIT_FLAG, "10"])
        rec = self._eval(w)
        paa = next(a for a in rec["refused"]["bon__s1__base__shard00"]["attempts"] if a["source"] == "s3_smoke")["prior_accepted_attempt"]
        assert paa["executed_shape"]["shape"] == co.EXECUTED_SHAPE_LIMITED and "the recorded smoke.budget agrees" in paa["executed_shape"]["why"] and paa["full_shape"] is False
        assert co.carry_refused_holds(rec) == {}
        assert [x["standup_id"] for x in srv.select_launch_rows(w.this, go="bon__s1__base__shard00", carry_holds={})[0]] == ["bon__s1__base__shard00"]
        # the flag recorded on the executed argv (a pre-D107 container that nonetheless ran a bounded child) is proof too
        w2 = self._world(tmp_path / "b", budget_block=None, child_argv=["scripts/sdfbase/full22/bon_served_run.py", lp.SMOKE_BON_LIMIT_FLAG, "10"])
        paa2 = next(a for a in self._eval(w2)["refused"]["bon__s1__base__shard00"]["attempts"] if a["source"] == "s3_smoke")["prior_accepted_attempt"]
        assert paa2["executed_shape"]["shape"] == co.EXECUTED_SHAPE_LIMITED and lp.SMOKE_BON_LIMIT_FLAG in paa2["executed_shape"]["why"]

    def test_absent_or_unreadable_evidence_retains_the_hold_with_the_reason_named(self, tmp_path):
        w = self._world(tmp_path, budget_block=None)                                           # no budget block, no argv records (metadata alone never decides)
        paa = next(a for a in self._eval(w)["refused"]["bon__s1__base__shard00"]["attempts"] if a["source"] == "s3_smoke")["prior_accepted_attempt"]
        assert paa["executed_shape"]["shape"] == co.EXECUTED_SHAPE_UNKNOWN and "not readable" in paa["executed_shape"]["why"] and paa["holds"] is True
        assert "RETAINED" in paa["hold_note"]
        w2 = self._world(tmp_path / "b", budget_block=None, drop_evidence=True)
        rec2 = self._eval(w2)
        paa2 = next(a for a in rec2["refused"]["bon__s1__base__shard00"]["attempts"] if a["source"] == "s3_smoke")["prior_accepted_attempt"]
        assert paa2["executed_shape"]["shape"] == co.EXECUTED_SHAPE_UNKNOWN and "absent" in paa2["executed_shape"]["why"]
        assert "bon__s1__base__shard00" in co.carry_refused_holds(rec2)

    def test_the_eight_leg_petri_s3_still_carries_with_a_full_executed_shape(self, tmp_path):
        w = TestS3FullShapeSource()._w19_world(tmp_path)
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                     git=w.git, readiness_problems=F.fake_readiness, criterion3_waivers=[w.waiver])
        assert rec["n_carried"] == 1
        # the carried entry carries the acceptance it was admitted under, classified FULL from its own evidence (eight children, no D107 budget for petri)
        paa = rec["carried"]["petri__s1__base__shard00"]["prior_accepted_attempt"]
        assert paa["executed_shape"]["shape"] == co.EXECUTED_SHAPE_FULL and paa["attempt_id"] == W19_PETRI_AID and f"{w.root}/modal_evidence.json" in paa["evidence_inventory"]
        # the same reading, direct, with the SMOKE row (the evidence's identity) and the identity axes -- and refused when handed the production row's identity
        prow = w.prev["standups"][0]
        srow = lp.smoke_row(prow, w.sident, max_children=8, max_timeout_s=int(prow["modal"]["timeout_s"]))
        ev = json.loads(w.files[f"{w.root}/modal_evidence.json"])
        assert co.executed_shape(evidence=ev, prod_row=prow, prev_row=srow, root=w.root, read_text=w.files.get, aid=W19_PETRI_AID, W=W19_REAL, wave_uuid=W19_S3_WAVE)["shape"] == co.EXECUTED_SHAPE_FULL
        assert co.executed_shape(evidence=ev, prod_row=prow, prev_row=prow, root=w.root, read_text=w.files.get)["shape"] == co.EXECUTED_SHAPE_UNKNOWN



class TestShapeEvidenceIdentityAndDigests:
    """fj15: the shape evidence must be THIS attempt's; the recorded budget never overrides the executed argv; every shape input is
    digest-bound and a released hold is re-validated on it at certify."""

    def _w(self, tmp_path, **kw):
        return TestExecutedShapeBoundToTheAttempt()._world(tmp_path, **kw)

    def _eval(self, w):
        return TestExecutedShapeBoundToTheAttempt()._eval(w)

    def _paa(self, rec):
        return next(a for a in rec["refused"]["bon__s1__base__shard00"]["attempts"] if a["source"] == "s3_smoke")["prior_accepted_attempt"]

    def test_evidence_naming_another_attempt_with_no_children_retains_the_hold(self, tmp_path):
        w = self._w(tmp_path, budget_block=None, child_argv=["bon_served_run.py"])
        ev = json.loads(w.files[f"{w.root}/modal_evidence.json"])
        ev["attempt_id"], ev["children"] = "01OTHER-000000000000000000", []
        w.files[f"{w.root}/modal_evidence.json"] = json.dumps(ev)
        paa = self._paa(self._eval(w))
        assert paa["executed_shape"]["shape"] == co.EXECUTED_SHAPE_UNKNOWN and "not this attempt's evidence" in paa["executed_shape"]["why"] and paa["holds"] is True
        for k, bad in (("W", "b" * 40), ("wave_uuid", "99999999-8888-4777-8666-555555555555"), ("standup_id", "smoke-S3__bon__s2__base__shard00")):
            w2 = self._w(tmp_path / k, budget_block=None, child_argv=["bon_served_run.py"])
            ev = json.loads(w2.files[f"{w2.root}/modal_evidence.json"])
            ev[k] = bad
            w2.files[f"{w2.root}/modal_evidence.json"] = json.dumps(ev)
            assert self._paa(self._eval(w2))["executed_shape"]["shape"] == co.EXECUTED_SHAPE_UNKNOWN, k

    def test_metadata_never_wins_over_the_executed_argv(self, tmp_path):
        # recorded item_budget 10, executed argv WITHOUT the flag -> contradiction -> unknown, held (fj14 said limited)
        w = self._w(tmp_path / "a", budget_block={"kind": lp.SMOKE_BUDGET_KIND_ITEM, "bon_limit_goals": 10, "misalign_limit_items": None},
                    child_argv=["scripts/sdfbase/full22/bon_served_run.py", "--cap", "4096"])
        paa = self._paa(self._eval(w))
        assert paa["executed_shape"]["shape"] == co.EXECUTED_SHAPE_UNKNOWN and "contradicts" in paa["executed_shape"]["why"] and paa["holds"] is True
        # recorded full_shape, `--limit-goals 10` added to the argv file -> contradiction -> unknown, held (fj14 read the budget alone)
        w2 = self._w(tmp_path / "b", budget_block={"kind": lp.SMOKE_BUDGET_KIND_FULL, "bon_limit_goals": None, "misalign_limit_items": None},
                     child_argv=["scripts/sdfbase/full22/bon_served_run.py", lp.SMOKE_BON_LIMIT_FLAG, "10"])
        assert self._paa(self._eval(w2))["executed_shape"]["shape"] == co.EXECUTED_SHAPE_UNKNOWN
        # recorded 10 but the argv executed 5 -> contradiction; recorded 10 and every argv 10 -> limited (the consistent genuine case releases)
        w3 = self._w(tmp_path / "c", budget_block={"kind": lp.SMOKE_BUDGET_KIND_ITEM, "bon_limit_goals": 10, "misalign_limit_items": None},
                     child_argv=["bon_served_run.py", f"{lp.SMOKE_BON_LIMIT_FLAG}=5"])
        assert self._paa(self._eval(w3))["executed_shape"]["shape"] == co.EXECUTED_SHAPE_UNKNOWN
        w4 = self._w(tmp_path / "d", budget_block={"kind": lp.SMOKE_BUDGET_KIND_ITEM, "bon_limit_goals": 10, "misalign_limit_items": None},
                     child_argv=["bon_served_run.py", f"{lp.SMOKE_BON_LIMIT_FLAG}=10"])
        rec4 = self._eval(w4)
        assert self._paa(rec4)["executed_shape"]["shape"] == co.EXECUTED_SHAPE_LIMITED and co.carry_refused_holds(rec4) == {}

    def test_a_released_hold_is_void_when_its_evidence_changes_after_the_evaluation(self, tmp_path):
        w = self._w(tmp_path, budget_block={"kind": lp.SMOKE_BUDGET_KIND_ITEM, "bon_limit_goals": 10, "misalign_limit_items": None},
                    child_argv=["bon_served_run.py", lp.SMOKE_BON_LIMIT_FLAG, "10"])
        rec = self._eval(w)
        paa = self._paa(rec)
        assert paa["holds"] is False and set(paa["evidence_inventory"]) >= {f"{w.root}/modal_evidence.json"} and any(p.endswith(".argv.json") for p in paa["evidence_inventory"])
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        # the probe: the FULL-budget evidence restored after the evaluation -> the release is void at certify
        restored = dict(w.files)
        ev = json.loads(restored[f"{w.root}/modal_evidence.json"])
        ev["smoke"].pop("budget", None)
        restored[f"{w.root}/modal_evidence.json"] = json.dumps(ev)
        for p in list(restored):
            if p.endswith(".argv.json"):
                restored[p] = json.dumps({"argv": ["bon_served_run.py"]})
        probs = co.carry_over_problems(rec, plan=w.this, read_text=restored.get, git=w.git, accepted_get_for_dict=_agd(w.dicts))
        assert any("released hold" in p and "changed since the evaluation" in p for p in probs), probs
        gone = {p: t for p, t in w.files.items() if not p.endswith(".argv.json")}
        assert any("no longer on the volume" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=gone.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        stripped = copy.deepcopy(rec)
        for a in stripped["refused"]["bon__s1__base__shard00"]["attempts"]:
            a.get("prior_accepted_attempt", {}).pop("evidence_inventory", None)
        assert any("binds no shape evidence" in p for p in co.carry_over_problems(stripped, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        # a RETAINED hold needs no re-validation (it protects, never releases)
        w2 = self._w(tmp_path / "held", budget_block=None)
        rec2 = self._eval(w2)
        assert "bon__s1__base__shard00" in co.carry_refused_holds(rec2) and co.carry_over_problems(rec2, plan=w2.this, read_text=(lambda p: None), git=w2.git, accepted_get_for_dict=_agd(w2.dicts)) == []


class TestCertificateReauditCheckPerCarriedEntry:
    """fj16: the re-audit validation runs inside the carried loop for EVERY entry, independent of read_text; the released-hold
    inventory validation stays separate; empty maps raise nothing."""

    def _two_row_record(self):
        rows = [F._row(W_THIS, W_THIS[:7]), F._row(W_THIS, W_THIS[:7], sid="codeattack__s1__base__shard00", leg="codeattack")]
        plan = F._plan(W_THIS, materialisation="final", rows=rows)
        good_ra = {"reaudited_by": W_THIS, "evidence_root": f"/vol/seedrep/{W_THIS[:7]}/{WAVE}/carried/x/evidence", "producer_evidence": {"r": {"sha256": "s"}},
                   "rc": 0, "argv": ["a"], "store_root": "/vol/seedrep/x"}
        def entry(sid, ra):
            row = next(r for r in rows if r["standup_id"] == sid)
            return {"accepted_key": row["accepted_key"], "attempt_id": AID, "attempt_root": f"/vol/seedrep/{W_PREV[:7]}/exp/x/{AID}",
                    "carried_from": {k: (W_PREV if k in ("W", "tree_sha") else (AID if k == "attempt_id" else "x")) for k in co.PROVENANCE_KEYS},
                    "clauses": {**{c: {"ok": True} for c in co.CLAUSES}, "producer_set": {"ok": True, "this_sha": W_THIS, "files": [{"path": "x", "identical": True}]}},
                    "reaudited_by": ra["reaudited_by"], "reaudit": copy.deepcopy(ra)}          # each entry its OWN record (a shared one aliased the probes)
        rec = {"schema": co.CARRY_OVER_SCHEMA, "W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE, "launch_plan_sha": plan["launch_plan_sha"],
               "carried": {F.SID: entry(F.SID, good_ra), "codeattack__s1__base__shard00": entry("codeattack__s1__base__shard00", good_ra)}, "refused": {}}
        return plan, rec

    def test_mixed_carried_and_refused_the_stale_carried_reaudit_refuses(self):
        plan, rec = self._two_row_record()
        rec["carried"][F.SID]["reaudited_by"] = rec["carried"][F.SID]["reaudit"]["reaudited_by"] = W_PREV        # the probe: a prior-tree re-audit
        rec["refused"]["renellm__s1__base__shard00"] = {"accepted_key": "k", "attempts": [{"clause": "row_identity", "why": "x", "prior_accepted_attempt": {"attempt_id": AID, "holds": True}}]}
        probs = co.carry_over_problems(rec, plan=plan)                                    # no read_text: the metadata check still runs
        assert any(F.SID in p and "reaudited_by" in p for p in probs), probs
        assert not any("codeattack" in p for p in probs)

    def test_multiple_carried_entries_the_first_bad_one_refuses_and_rc_1_refuses(self):
        plan, rec = self._two_row_record()
        rec["carried"][F.SID]["reaudit"]["rc"] = 1                                          # the FIRST entry is the bad one
        probs = co.carry_over_problems(rec, plan=plan)
        assert any(F.SID in p and "rc 1" in p for p in probs) and not any("codeattack" in p for p in probs)
        plan2, rec2 = self._two_row_record()
        assert co.carry_over_problems(rec2, plan=plan2) == []                               # two good carried entries: clean

    def test_empty_maps_raise_nothing_and_report_nothing(self):
        plan = F._plan(W_THIS, materialisation="final")
        rec = {"schema": co.CARRY_OVER_SCHEMA, "W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE, "launch_plan_sha": plan["launch_plan_sha"], "carried": {}, "refused": {}}
        assert co.carry_over_problems(rec, plan=plan) == [] and co.carry_over_problems(rec, plan=plan, read_text=lambda p: None, git=_anygit(), accepted_get_for_dict=_agd({})) == []


# --------------------------------------------------------------------------- fj17: this tree's S5 FULL row is the production row's canonical attempt

CONTROLS_LADDER = {"kind": "controls", "cap_next": 2048, "shard": "0", "shards": ["0"], "feeder_shards": ["0"], "producer_of": "2048"}
CONTROLS_SID, CONTROLS_RUNG_SID = "controls__s1__base__shard00", "controls__s1__2048__shard00"


def _agd(dicts):
    """The tests' prior-acceptance reader (fj28): the world's Dicts by name."""
    return lambda n: dicts.get(n, {}).get


def _anygit():
    """A FakeGit whose blobs are identical at every synthetic tree the carry tests use -- the certificate's clause (b) recheck (fj25) then re-derives
    'nothing differs', which is what those records say."""
    return F.FakeGit({W: F._blobs(W) for W in (W_THIS, W_PREV, "c" * 40, "d" * 40, "e" * 40, "f" * 40)})


def _ac():
    """`audit_common` (other carry tests put `scripts/sdfbase/seedrep` on sys.path at import; this module alone must too)."""
    p = REPO / "scripts" / "sdfbase" / "seedrep"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import audit_common as ac
    return ac


def _controls_evidence(files: dict, root: str, row: dict, W: str) -> None:
    """fj2's synthetic root writes ONE audit verdict / determinism record per row; a CONTROLS child is bound by one verdict per table leg (xstest,
    or_bench_hard_1k, mmlu on the base row) and one determinism record per table leg -- written here from the same documents, bound in the
    evidence, the caller-composed `request_log/controls/...` binding nobody reads removed."""
    def bind(text: str, writer: str) -> dict:
        return {"sha256": F._sha(text), "writer": writer, "tree_sha": W, "rc": 0}
    ev = json.loads(files[f"{root}/modal_evidence.json"])
    k = row["children"][0]
    written = lp.audit_verdict_rel(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"])
    audit = json.loads(files.pop(f"{root}/{written}"))
    ev[lp.PRODUCER_EVIDENCE_KEY].pop(written, None)
    for leg, rel in lp.audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"]):
        text = json.dumps({**audit, "leg": leg, "verdict_file": rel[len(lp.AUDIT_OUT_REL) + 1:]})
        files[f"{root}/{rel}"] = text
        ev[lp.PRODUCER_EVIDENCE_KEY][rel] = bind(text, lp.PRODUCER_WRITERS["request_evidence_audit"])
    det0_leg, det0_rel = lp.determinism_rels(row)[0]
    det = json.loads(files[f"{root}/{det0_rel}"])
    for leg, rel in lp.determinism_rels(row):
        text = json.dumps({**det, "leg": leg})
        files[f"{root}/{rel}"] = text
        ev[lp.DETERMINISM_EVIDENCE_KEY][leg] = bind(text, lp.PRODUCER_WRITERS["determinism"])
    files[f"{root}/modal_evidence.json"] = json.dumps(ev)


class TestCurrentTreeS5Full:
    """fj17: a completed, accepted S5 FULL-shape smoke row on the CURRENT tree is the production row's canonical
    attempt -- carried as `current_s5_full`, never spawned, never re-audited (its verdicts are already this tree's readers' work)."""

    def _w24_world(self, tmp_path):
        """THIS plan's controls base row (cap 512, producer of the 2048 rung) + that rung; the `S5` FULL smoke of the base row
        completed and was accepted under the `S5` wave's Dict on THIS tree; `--carry-over RS` names this very RS."""
        base = F._row(W_THIS, W_THIS[:7], sid=CONTROLS_SID, leg="controls", ladder=CONTROLS_LADDER, cap=512)
        rung = {**F._row(W_THIS, W_THIS[:7], sid=CONTROLS_RUNG_SID, leg="controls", stage="2048", kind="rung", cap=2048), "rung_mode": lp.RUNG_MODE_CONSUME}
        this = F._plan(W_THIS, materialisation="final", rows=[base], rungs=[rung])
        sident = lp.smoke_identity(this, "S5", F.S5_WAVE)
        srow = lp.smoke_row(base, sident, max_children=len(base["children"]), max_timeout_s=int(base["modal"]["timeout_s"]))
        assert srow["smoke"]["budget"]["kind"] == lp.SMOKE_BUDGET_KIND_FULL and srow["ladder"] == CONTROLS_LADDER
        files, dicts = {}, {}
        root = F._root_files(files, plan=this, row=srow, identity=sident, smoke_block=srow["smoke"])
        _controls_evidence(files, root, srow, W_THIS)
        dicts[sident["dicts"]["accepted"]] = {srow["accepted_key"]: AID}
        d = F._prior_dir(tmp_path, this, s5_wave=F.S5_WAVE, name="rs")
        (d / "launch_plan.json").write_text(json.dumps(this))          # the CURRENT RS holds the FINAL plan
        return types.SimpleNamespace(this=this, files=files, dicts=dicts, prior=co.load_prior_source(d), root=root, base=base, srow=srow,
                                     git=F.FakeGit({W_THIS: F._blobs(W_THIS)}))

    @staticmethod
    def _replay_into(files: dict):
        """A replay stub that leaves what the certificate re-reads on the 'volume': the carried producer's CLOSED marker and the queue's CLOSED.json."""
        def replay(**kw):
            marker, closed = f"seedrep/{kw['this_plan']['W7']}/{kw['this_plan']['wave_uuid']}/ladder/carried_marker", f"seedrep/{kw['this_plan']['W7']}/{kw['this_plan']['wave_uuid']}/ladder/CLOSED.json"
            files[f"{lp.VOL_PATH}/{marker}"] = files[f"{lp.VOL_PATH}/{closed}"] = "{}"
            return {"schema": co.REPLAY_SCHEMA, "ok": True, "closed": True, "n_records": 1, "queue_closed": True, "closed_json": closed, "source_keys": {},
                    "target_keys": {"marker": marker}}
        return replay

    def test_the_s5_controls_full_acceptance_on_this_tree_is_the_base_rows_attempt_and_is_not_spawned(self, tmp_path, srv, capsys):
        w = self._w24_world(tmp_path)
        assert w.prior["W"] == w.this["W"] and w.prior["smoke_waves"] == {"S5": F.S5_WAVE}
        replays = []

        real = self._replay_into(w.files)

        def replay(**kw):
            replays.append((kw["prior"]["W"], kw.get("prev_wave_uuid"), kw["this_plan"]["wave_uuid"]))
            return real(**kw)

        def no_reaudit(**kw):
            raise AssertionError("no re-audit on a same-tree source")
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                     git=w.git, readiness_problems=F.fake_readiness, replay=replay, reaudit=no_reaudit)
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][CONTROLS_SID]
        cf = e["carried_from"]
        assert cf["source"] == co.CURRENT_S5_SOURCE and cf["W"] == cf["tree_sha"] == W_THIS and cf["wave_uuid"] == F.S5_WAVE and cf["standup_id"] == f"smoke-S5__{CONTROLS_SID}"
        assert e["clauses"]["row_identity"]["clause"] == co.S5_FULL_CLAUSE and e["s5_full_row"] and e["smoke_source"] == "S5"
        assert e["reaudited_by"] == co.REAUDIT_NOT_REQUIRED_SAME_TREE and e["reaudit"] is None and "reaudit_evidence_dir" not in e
        assert e["attempt_id"] == AID and e["attempt_root"] == w.root
        # the ladder queue is replayed from the S5 WAVE's ladder root (the wave the attempt was accepted under), into this wave's
        assert replays == [(W_THIS, F.S5_WAVE, WAVE)] and e["clauses"]["ladder_queue"]["consumers"] == [CONTROLS_RUNG_SID]
        assert e["clauses"]["producer_set"]["prev_sha"] == W_THIS == e["clauses"]["producer_set"]["this_sha"]  # the same tree: byte-identical by construction
        assert e["prior_accepted_attempt"]["full_shape"] is True and e["prior_accepted_attempt"]["source"] == co.CURRENT_S5_SOURCE
        # the rung is a production row with no source of its own (never an S5 candidate): refused by name, holding nothing
        assert set(rec["refused"]) == {CONTROLS_RUNG_SID} and co.carry_refused_holds(rec) == {}
        assert [(a["source"], a["why"]) for a in rec["refused"][CONTROLS_RUNG_SID]["attempts"]] == [(co.CURRENT_S5_SOURCE, "the prior tree IS this tree; a rung row has no S5 smoke of its own -- no source for it here")]
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        # launch selection: `--legs controls` finds its only base row CARRIED (never spawned); the rung still spawns (`--go`, the steward's spelling)
        carried = co.carried_standup_ids(rec)
        with pytest.raises(srv.LaunchRefused, match="every selected row is CARRIED"):
            srv.select_launch_rows(w.this, legs="controls", carried=carried, carry_holds=co.carry_refused_holds(rec))
        assert f"CARRIED {CONTROLS_SID}" in capsys.readouterr().out
        picked, _ = srv.select_launch_rows(w.this, go=CONTROLS_RUNG_SID, carried=carried, carry_holds=co.carry_refused_holds(rec))
        assert [x["standup_id"] for x in picked] == [CONTROLS_RUNG_SID]
        # the coverage binder's mapping / the identity table's source-W view admit the same-tree source with its smoke-S5__ spelling
        ac = _ac()
        plan = ac.LaunchPlan(path=Path("plan"), W=W_THIS, wave_uuid=WAVE, rows={CONTROLS_SID: {**w.base, "view": "s1", "stage": "base", "shard": "shard00"}})
        m = ac.carry_mapping(rec, plan)
        assert m[AID].source == co.CURRENT_S5_SOURCE and m[AID].W == W_THIS and m[AID].wave_uuid == F.S5_WAVE and m[AID].historical_standup_id == f"smoke-S5__{CONTROLS_SID}"
        assert ac.carry_source_ws(rec) == {AID: W_THIS}

    def test_the_certificate_and_the_binder_refuse_a_misnamed_same_tree_source(self, tmp_path):
        w = self._w24_world(tmp_path)
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                     git=w.git, readiness_problems=F.fake_readiness, replay=self._replay_into(w.files))
        assert rec["n_carried"] == 1
        ac = _ac()
        plan = ac.LaunchPlan(path=Path("plan"), W=W_THIS, wave_uuid=WAVE, rows={CONTROLS_SID: {**w.base, "view": "s1", "stage": "base", "shard": "shard00"}})
        # (1) a same-tree provenance under a PRIOR-tree source name: the same-tree re-audit marker is not admitted, the binder refuses
        bad = copy.deepcopy(rec)
        bad["carried"][CONTROLS_SID]["carried_from"]["source"] = "s5_smoke"
        probs = co.carry_over_problems(bad, plan=w.this)
        assert any("admitted only for this tree's own S5 FULL source" in p for p in probs) and any("provenance names THIS tree" in p for p in probs)
        with pytest.raises(ac.AuditRefusal, match="must not name this tree"):
            ac.carry_mapping(bad, plan)
        with pytest.raises(ac.AuditRefusal, match="not a prior tree"):
            ac.carry_source_ws(bad)
        # (2) a PRIOR-tree provenance under the current source name
        bad2 = copy.deepcopy(rec)
        bad2["carried"][CONTROLS_SID]["carried_from"]["W"] = bad2["carried"][CONTROLS_SID]["carried_from"]["tree_sha"] = W_PREV
        assert any("must be THIS tree's" in p for p in co.carry_over_problems(bad2, plan=w.this))
        with pytest.raises(ac.AuditRefusal, match="S5 FULL source must"):
            ac.carry_mapping(bad2, plan)
        # (3) a same-tree entry that claims a re-audit record anyway
        bad3 = copy.deepcopy(rec)
        bad3["carried"][CONTROLS_SID]["reaudit"] = {"reaudited_by": W_THIS, "rc": 0}
        assert any("with no re-audit record" in p for p in co.carry_over_problems(bad3, plan=w.this))
        # (4) a prior-tree entry may never wear the same-tree marker
        bad4 = copy.deepcopy(rec)
        bad4["carried"][CONTROLS_SID]["carried_from"].update(W=W_PREV, tree_sha=W_PREV, source="s5_smoke")
        assert any("admitted only for this tree's own S5 FULL source" in p for p in co.carry_over_problems(bad4, plan=w.this))

    def test_a_same_tree_source_with_only_a_budgeted_s3_smoke_carries_nothing_holds_nothing_and_the_row_spawns(self, tmp_path, srv):
        base = F._row(W_THIS, W_THIS[:7], sid="bon__s1__base__shard00", leg="bon")
        this = F._plan(W_THIS, materialisation="final", rows=[base])
        sident = lp.smoke_identity(this, "S3", F.S5_WAVE)
        srow = lp.smoke_row(base, sident, max_children=len(base["children"]), max_timeout_s=int(base["modal"]["timeout_s"]))
        assert srow["smoke"]["budget"]["kind"] == lp.SMOKE_BUDGET_KIND_ITEM                    # bon's S3 is BUDGETED by construction
        files, dicts = {}, {}
        root = F._root_files(files, plan=this, row=srow, identity=sident, smoke_block=srow["smoke"])
        for k in srow["children"]:
            files[f"{root}/children/{k['eval_leg']}__{k['arm']}__s{k['s']}.argv.json"] = json.dumps({"argv": list(k["argv"])})
        dicts[sident["dicts"]["accepted"]] = {srow["accepted_key"]: AID}
        d = F._prior_dir(tmp_path, this, name="rs")
        (d / "smokes" / "S3").mkdir(parents=True, exist_ok=True)
        (d / "smokes" / "S3" / "wave.json").write_text(json.dumps({"smoke_id": "S3", "wave_uuid": F.S5_WAVE}))
        prior = co.load_prior_source(d)
        assert prior["smoke_waves"] == {"S3": F.S5_WAVE}
        rec = co.evaluate_carry_over(this_plan=this, this_sha=W_THIS, priors=[prior], read_text=files.get, accepted_get_for_dict=lambda n: dicts.get(n, {}).get,
                                     git=F.FakeGit({W_THIS: F._blobs(W_THIS)}), readiness_problems=F.fake_readiness)
        # a same-tree source offers ONLY its S5 FULL row: with no S5 wave nothing is a candidate -- the budgeted S3 acceptance is never even consulted
        r = rec["refused"]["bon__s1__base__shard00"]
        assert rec["n_carried"] == 0 and [a["source"] for a in r["attempts"]] == [co.CURRENT_S5_SOURCE] and "minted no S5 smoke wave" in r["attempts"][0]["why"]
        assert r["attempts"][0].get("attempt_id") is None and "prior_accepted_attempt" not in r["attempts"][0]       # the S3 acceptance was never read
        assert co.carry_refused_holds(rec) == {} and co.carry_over_problems(rec, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) == []
        picked, _ = srv.select_launch_rows(this, go="bon__s1__base__shard00", carried=co.carried_standup_ids(rec), carry_holds=co.carry_refused_holds(rec))
        assert [x["standup_id"] for x in picked] == ["bon__s1__base__shard00"]

    def test_source_order_prior_production_then_this_trees_s5_full_then_the_prior_smokes(self, tmp_path):
        w = self._w24_world(tmp_path / "cur")
        prev_wave = "44444444-5555-4666-8777-888888888888"
        prev = F._plan(W_PREV, materialisation="final", rows=[F._row(W_PREV, W_PREV[:7], sid=CONTROLS_SID, leg="controls", ladder=CONTROLS_LADDER, cap=512)])
        prior_prev = co.load_prior_source(F._prior_dir(tmp_path / "prev", prev, s5_wave=prev_wave, name="prior"))
        w.git.blobs[W_PREV] = F._blobs(W_PREV)
        # the PRIOR tree's S5 FULL row is ALSO accepted -- the order decides which one carries
        psident = lp.smoke_identity(prev, "S5", prev_wave)
        psrow = lp.smoke_row(prev["standups"][0], psident, max_children=1, max_timeout_s=int(prev["standups"][0]["modal"]["timeout_s"]))
        proot = F._root_files(w.files, plan=prev, row=psrow, identity=psident, aid="01TESTTASK-deadbeef0002", smoke_block=psrow["smoke"])
        _controls_evidence(w.files, proot, psrow, W_PREV)
        w.dicts[psident["dicts"]["accepted"]] = {psrow["accepted_key"]: "01TESTTASK-deadbeef0002"}
        for priors in ([w.prior, prior_prev], [prior_prev, w.prior]):                      # operator order never changes the source order
            rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=priors, read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                         git=w.git, readiness_problems=F.fake_readiness, replay=self._replay_into(w.files), reaudit=lambda **kw: (_ for _ in ()).throw(AssertionError("same tree")))
            e = rec["carried"][CONTROLS_SID]
            # the prior PRODUCTION acceptance was tried first (none accepted), then THIS tree's S5 FULL won; the prior S5 FULL was never reached
            assert [a["source"] for a in e["refused_before"]] == ["production"] and e["refused_before"][0]["prior_W"] == W_PREV
            assert e["carried_from"]["source"] == co.CURRENT_S5_SOURCE and e["attempt_id"] == AID
        # without this tree's S5 wave the prior tree's S5 FULL carries as before (fj12's order: production, S5, S3; the historical verdicts read
        # as they are here -- the re-audit of a prior-tree source has its own tests)
        w.prior["smoke_waves"].pop("S5")
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior, prior_prev], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                     git=w.git, readiness_problems=F.fake_readiness, replay=self._replay_into(w.files))
        e = rec["carried"][CONTROLS_SID]
        assert e["carried_from"]["source"] == "s5_smoke" and e["carried_from"]["W"] == W_PREV and e["attempt_id"] == "01TESTTASK-deadbeef0002"
        assert e.get("reaudited_by") is None and e.get("reaudit") is None                              # no re-audit given: no same-tree marker either
        # the prior production (none accepted), then this tree named (no S5 wave) -- in source order -- before the prior S5 FULL won
        assert [(a["source"], a["clause"]) for a in e["refused_before"]] == [("production", "complete"), (co.CURRENT_S5_SOURCE, "row_identity")]
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []


# --------------------------------------------------------------------------- fj18: a never-created PRODUCTION accepted Dict is an empty acceptance map

class NotFoundError(Exception):
    """The shape of `modal.exception.NotFoundError` (matched by name: the evaluator imports no modal)."""


class TestMissingProductionDictIsEmpty:
    """fj18: the first real carry-over run aborted at `modal.Dict.from_name(<prior production accepted Dict>)` -- no production row ever ran on any
    prior tree, so that Dict never existed. The evaluator reads a never-created PRODUCTION Dict as an empty acceptance map (recorded
    `dict_missing`); a smoke Dict's absence is never swallowed; a Dict that exists but names nothing is 'no candidate' as before."""

    def _s5_world(self, tmp_path):
        prev = F._plan(W_PREV, materialisation="draft")
        this = F._plan(W_THIS, materialisation="final")
        prow = prev["standups"][0]
        sident = lp.smoke_identity(prev, "S5", F.S5_WAVE)
        srow = lp.smoke_row(prow, sident, max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
        files, dicts = {}, {}
        root = F._root_files(files, plan=prev, row=srow, identity=sident, smoke_block=srow["smoke"])
        dicts[sident["dicts"]["accepted"]] = {srow["accepted_key"]: AID}
        d = F._prior_dir(tmp_path, prev, s5_wave=F.S5_WAVE)
        _launched(d, W_PREV, {"S5": F.S5_WAVE}, production=None)                      # fj20: the record -- S5 launched, no production launch
        prior = co.load_prior_source(d)
        return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, prior=prior, root=root, prod_dict=prev["dicts"]["accepted"], s5_dict=sident["dicts"]["accepted"],
                                     git=F.FakeGit({W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS)}))

    @staticmethod
    def _getter(dicts, missing):
        """The live getter's shape: `modal.Dict.from_name` is lazy, so a never-created Dict raises at its first `.get`."""
        def for_dict(name):
            def get(key):
                if name in missing:
                    raise NotFoundError(f"Dict '{name}' not found in environment 'main'")
                return dicts.get(name, {}).get(key)
            return get
        return for_dict

    def _eval(self, w, missing=(), reaudit=False):
        return co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=self._getter(w.dicts, set(missing)),
                                      git=w.git, readiness_problems=F.fake_readiness)

    def test_a_missing_production_dict_is_reported_and_the_s5_full_row_carries(self, tmp_path):
        w = self._s5_world(tmp_path)
        assert w.prod_dict not in w.dicts and w.prod_dict != w.s5_dict
        rec = self._eval(w, missing={w.prod_dict})
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][F.SID]
        assert e["carried_from"]["source"] == "s5_smoke" and e["attempt_id"] == AID and e["attempt_root"] == w.root
        prod, = e["refused_before"]
        assert prod["source"] == "production" and prod["dict_missing"] == w.prod_dict == prod["dict"] and prod["attempt_id"] is None
        assert prod["clause"] == "complete" and "does not exist" in prod["why"] and "production source offers nothing" in prod["why"] and "records no production launch" in prod["archive"]
        assert "prior_accepted_attempt" not in prod                                                # nothing was accepted: nothing to hold on
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == [] and co.carry_refused_holds(rec) == {}
        assert "dict_missing" not in e and rec["carried"][F.SID]["carried_from"]["dict"] == w.s5_dict     # the S5 Dict that exists kept its live read

    def test_a_production_dict_that_exists_but_names_nothing_is_no_candidate_as_before(self, tmp_path):
        w = self._s5_world(tmp_path)
        w.dicts[w.prod_dict] = {}
        rec = self._eval(w)
        prod, = rec["carried"][F.SID]["refused_before"]
        assert prod["source"] == "production" and "no ACCEPTED attempt" in prod["why"] and "dict_missing" not in prod and prod["attempt_id"] is None
        # a row with NO S5 acceptance under a missing production Dict is refused by name and holds nothing (a never-run row spawns)
        w2 = self._s5_world(tmp_path / "two")
        w2.dicts.pop(w2.s5_dict)
        rec2 = self._eval(w2, missing={w2.prod_dict})
        r = rec2["refused"][F.SID]
        assert rec2["n_carried"] == 0 and [a["source"] for a in r["attempts"]] == ["production", "s5_smoke"]
        assert r["attempts"][0]["dict_missing"] == w2.prod_dict and "dict_missing" not in r["attempts"][1] and "no ACCEPTED attempt" in r["attempts"][1]["why"]
        assert co.carry_refused_holds(rec2) == {}

    def test_a_missing_smoke_dict_the_archive_records_as_launched_is_never_swallowed(self, tmp_path):
        w = self._s5_world(tmp_path)
        w.dicts[w.prod_dict] = {}
        with pytest.raises(ValueError, match=f"{w.s5_dict}.*s5_smoke.*vanished.*records wave {F.S5_WAVE}") as ei:
            self._eval(w, missing={w.s5_dict})
        assert isinstance(ei.value.__cause__, NotFoundError)
        # and any OTHER exception from a getter propagates unchanged, production source included
        def boom(name):
            def get(key):
                raise RuntimeError("modal client timeout")
            return get
        with pytest.raises(RuntimeError, match="modal client timeout"):
            co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=boom, git=w.git, readiness_problems=F.fake_readiness)
        assert co.is_dict_missing(NotFoundError("x")) and not co.is_dict_missing(KeyError("x")) and not co.is_dict_missing(RuntimeError("NotFoundError"))


def _launched(prior_dir: Path, W: str, launched: dict, production=None) -> Path:
    """The fj19 per-prior LAUNCH record the archive may carry: which smoke waves (and whether the production wave) this tree launched."""
    p = prior_dir / co.SMOKES_LAUNCHED_REL
    p.write_text(json.dumps({"schema": co.SMOKES_LAUNCHED_SCHEMA, "W": W, "launched": launched, "production": production}))
    return p


def _mint_all_smokes(prior_dir: Path, waves: dict) -> None:
    """The REAL archive layout: every prior carries ALL smoke wave mints (minted once in RS, shared by every W) -- a wave file is not a launch."""
    for sid, wave in waves.items():
        p = prior_dir / "smokes" / sid / "wave.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"smoke_id": sid, "wave_uuid": wave}))


# --------------------------------------------------------------------------- fj19: never-minted vs vanished, decided from the archive's LAUNCH record

#: distinct first-8 chars per wave (a Dict is named `seedrep-<W7>-<wave8>-accepted`): none may share a prefix with the production WAVE or S5_WAVE
SHARED_MINTS = {"S1": "a1a1a1a1-a1a1-4a1a-8a1a-a1a1a1a1a1a1", "S1b": "b1b1b1b1-b1b1-4b1b-8b1b-b1b1b1b1b1b1", "S2": "c2c2c2c2-c2c2-4c2c-8c2c-c2c2c2c2c2c2",
                "S3": "d3d3d3d3-d3d3-4d3d-8d3d-d3d3d3d3d3d3", "S4": "e4e4e4e4-e4e4-4e4e-8e4e-e4e4e4e4e4e4", "S5": F.S5_WAVE}
assert len({w[:8] for w in (*SHARED_MINTS.values(), WAVE)}) == len(SHARED_MINTS) + 1


class TestMissingDictNeverMintedVsVanished:
    """fj19: the real carry-over aborted next on a prior's missing Dict -- that prior never launched the smoke, but its archive (like every
    prior's) carries the SHARED smoke mints, so the candidate existed and fj18's non-production re-raise fired. The distinction is now the
    archive's LAUNCH record (`smokes_launched.json`): absent or not naming the smoke -> the source was never launched, an empty acceptance
    (`dict_missing` + `archive`); naming this very smoke -> vanished, refused by name. The same rule for every source, production included."""

    def _prior(self, tmp_path, W, *, dicts, files, accepted_smoke=None, name="prior", launched=None, production=None):
        """A prior tree whose archive carries ALL shared smoke mints; Dicts exist only for the smokes it ran (``dicts`` keys); ``accepted_smoke``
        (a smoke id) writes that smoke's FULL-shape row of SID as accepted."""
        prev = F._plan(W, materialisation="draft")
        d = F._prior_dir(tmp_path, prev, name=name)
        _mint_all_smokes(d, SHARED_MINTS)
        if launched is not None or production is not None:
            _launched(d, W, launched or {}, production)
        prow = prev["standups"][0]
        idents = {sid: lp.smoke_identity(prev, sid, wave) for sid, wave in SHARED_MINTS.items()}
        if accepted_smoke:
            sident = idents[accepted_smoke]
            srow = lp.smoke_row(prow, sident, max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
            F._root_files(files, plan=prev, row=srow, identity=sident, smoke_block=srow["smoke"], aid=f"01TESTTASK-{W[:8]}")
            dicts.setdefault(sident["dicts"]["accepted"], {})[srow["accepted_key"]] = f"01TESTTASK-{W[:8]}"
        prior = co.load_prior_source(d)
        assert set(prior["smoke_waves"]) == set(SHARED_MINTS)                     # the archive 'records' every wave -- for every prior
        return types.SimpleNamespace(prev=prev, prior=prior, dir=d, idents=idents, prod_dict=prev["dicts"]["accepted"],
                                     dict_of={sid: i["dicts"]["accepted"] for sid, i in idents.items()})

    @staticmethod
    def _getter(dicts):
        """The live shape: a Dict that was never created raises NotFoundError at its first read; an existing one answers (None when empty)."""
        def for_dict(name):
            def get(key):
                if name not in dicts:
                    raise NotFoundError(f"Dict '{name}' not found in environment 'main'")
                return dicts[name].get(key)
            return get
        return for_dict

    def _eval(self, this, priors, files, dicts, Ws):
        return co.evaluate_carry_over(this_plan=this, this_sha=W_THIS, priors=priors, read_text=files.get, accepted_get_for_dict=self._getter(dicts),
                                      git=F.FakeGit({W: F._blobs(W) for W in (W_THIS, *Ws)}), readiness_problems=F.fake_readiness)

    def test_a_record_omitting_the_smoke_makes_its_missing_dict_empty_and_the_s3_source_carries(self, tmp_path):
        files, dicts = {}, {}
        this = F._plan(W_THIS, materialisation="final")
        w = self._prior(tmp_path, W_PREV, dicts=dicts, files=files, accepted_smoke="S3", launched={"S3": SHARED_MINTS["S3"]})     # ran S3 (FULL for a static leg), never S5
        assert w.dict_of["S5"] not in dicts and w.dict_of["S3"] in dicts
        rec = self._eval(this, [w.prior], files, dicts, [W_PREV])
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][F.SID]
        assert e["carried_from"]["source"] == "s3_smoke" and e["clauses"]["row_identity"]["clause"] == co.S3_FULL_CLAUSE
        by_source = {a["source"]: a for a in e["refused_before"]}
        assert list(by_source) == ["production", "s5_smoke"] and not any(a.get("launch_history_unknown") for a in e["refused_before"])
        assert by_source["production"]["dict_missing"] == w.prod_dict and "records no production launch" in by_source["production"]["archive"]
        assert by_source["s5_smoke"]["dict_missing"] == w.dict_of["S5"] and "records no S5 launch" in by_source["s5_smoke"]["archive"] and co.SMOKES_LAUNCHED_REL in by_source["s5_smoke"]["archive"]
        assert "does not exist" in by_source["s5_smoke"]["why"] and "s5_smoke source offers nothing" in by_source["s5_smoke"]["why"]
        assert co.carry_refused_holds(rec) == {} and co.carry_over_problems(rec, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) == []

    def test_sol_a_retained_complete_root_whose_dict_is_missing_with_no_record_is_held_not_spawned(self, tmp_path, srv):
        """fj19: the S5 FULL root is COMPLETE on the volume and its acceptance existed; with the Dict unreadable and no launch record the
        row got neither carry nor hold and `select_launch_rows` spawned it. Now: launch history UNKNOWN -> refused by name, HELD."""
        files, dicts = {}, {}
        this = F._plan(W_THIS, materialisation="final")
        w = self._prior(tmp_path, W_PREV, dicts=dicts, files=files, accepted_smoke="S5")              # no launch record
        assert w.prior["smokes_launched"] is None
        rec_ok = self._eval(this, [w.prior], files, dicts, [W_PREV])
        assert rec_ok["n_carried"] == 1 and rec_ok["carried"][F.SID]["carried_from"]["source"] == "s5_smoke"         # the Dict reads: carried, no record needed
        dicts.pop(w.dict_of["S5"])                                                                     # the same root; the Dict is gone; no record
        rec = self._eval(this, [w.prior], files, dicts, [W_PREV])
        assert rec["n_carried"] == 0
        attempts = rec["refused"][F.SID]["attempts"]
        assert [(a["source"], a.get("launch_history_unknown"), a.get("dict_missing")) for a in attempts] == \
               [("production", True, w.prod_dict), ("s5_smoke", True, w.dict_of["S5"]), ("s3_smoke", True, w.dict_of["S3"])]
        s5 = attempts[1]
        assert s5["clause"] == "complete" and "launch history unknown" in s5["why"] and co.SMOKES_LAUNCHED_SCHEMA in s5["archive"] and co.CARRY_REFUSED_HOLD in s5["why"]
        assert "does not exist" not in s5["why"] and "prior_accepted_attempt" not in s5
        holds = co.carry_refused_holds(rec)
        h = holds[F.SID]
        assert h["reason"] == co.CARRY_REFUSED_HOLD and h["hold_kind"] == co.DICT_HISTORY_UNKNOWN and h["attempt_id"] is None and h["dict_missing"] == w.dict_of["S3"]
        assert h["source"] == "s3_smoke" and co.SMOKES_LAUNCHED_REL in h["note"]
        with pytest.raises(srv.LaunchRefused, match="every selected row is HELD"):
            srv.select_launch_rows(this, go=F.SID, carried=co.carried_standup_ids(rec), carry_holds=holds)
        # fj21: the hold is ALSO a certificate problem -- a retry pass / --reattempt spawns without reading holds, so no certificate issues
        probs = co.carry_over_problems(rec, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts))
        assert [p for p in probs if not p.startswith(f"carry hold {co.DICT_HISTORY_UNKNOWN} on {F.SID}")] == [] and len(probs) == 1
        # the record names the wave -> the Dict VANISHED: refused by name (fj19); the record omits S5 -> never launched, empty, the row spawns (nothing completed)
        _launched(w.dir, W_PREV, {"S5": F.S5_WAVE})
        with pytest.raises(ValueError, match="vanished"):
            self._eval(this, [co.load_prior_source(w.dir)], files, dicts, [W_PREV])
        _launched(w.dir, W_PREV, {"S3": SHARED_MINTS["S3"]})
        dicts[w.dict_of["S3"]] = {}                                                                # the record says S3 launched: its Dict exists (nothing accepted)
        rec3 = self._eval(this, [co.load_prior_source(w.dir)], files, dicts, [W_PREV])
        assert rec3["n_carried"] == 0 and co.carry_refused_holds(rec3) == {} and all(not a.get("launch_history_unknown") and a.get("dict_missing") for a in rec3["refused"][F.SID]["attempts"][:2])
        assert "no ACCEPTED attempt" in rec3["refused"][F.SID]["attempts"][2]["why"]
        assert [x["standup_id"] for x in srv.select_launch_rows(this, go=F.SID, carried={}, carry_holds={})[0]] == [F.SID]

    def test_a_launched_wave_whose_dict_is_missing_vanished_and_refuses_by_name_for_smoke_and_production(self, tmp_path):
        files, dicts = {}, {}
        this = F._plan(W_THIS, materialisation="final")
        w = self._prior(tmp_path, W_PREV, dicts=dicts, files=files, accepted_smoke="S3", launched={"S5": F.S5_WAVE, "S3": SHARED_MINTS["S3"]})
        with pytest.raises(ValueError, match=f"{w.dict_of['S5']} of the s5_smoke source .* vanished: the archive records wave {F.S5_WAVE} launched for S5") as ei:
            self._eval(this, [w.prior], files, dicts, [W_PREV])
        assert isinstance(ei.value.__cause__, NotFoundError)
        # production likewise: the archive records the production wave launched -> its missing Dict is a vanished Dict, never an empty one
        w2 = self._prior(tmp_path / "p2", W_PREV, dicts=dicts, files=files, name="prior2", production=WAVE)
        assert w2.prev["wave_uuid"] == WAVE and w2.prod_dict not in dicts
        with pytest.raises(ValueError, match=f"{w2.prod_dict} of the production source .* vanished: the archive records wave {WAVE} launched for production"):
            self._eval(this, [w2.prior], files, dicts, [W_PREV])
        # ... and a record naming a DIFFERENT wave for the source is an inconsistent archive: launch history unknown -> refused, HELD
        files3, dicts3 = {}, {}                                                                   # a tree of its own: no Dict of any kind exists
        w3 = self._prior(tmp_path / "p3", W_PREV, dicts=dicts3, files=files3, name="prior3", launched={"S5": "99999999-9999-4999-8999-999999999999"}, production="99999999-9999-4999-8999-999999999998")
        rec = self._eval(this, [w3.prior], files3, dicts3, [W_PREV])
        by_source = {a["source"]: a for a in rec["refused"][F.SID]["attempts"]}
        assert by_source["production"]["launch_history_unknown"] and "inconsistent archive" in by_source["production"]["archive"] and f"candidate wave is {WAVE}" in by_source["production"]["archive"]
        assert by_source["s5_smoke"]["launch_history_unknown"] and f"candidate wave is {F.S5_WAVE}" in by_source["s5_smoke"]["archive"]
        assert by_source["s3_smoke"]["dict_missing"] and not by_source["s3_smoke"].get("launch_history_unknown")          # S3 absent from the record: never launched
        assert co.carry_refused_holds(rec)[F.SID]["hold_kind"] == co.DICT_HISTORY_UNKNOWN

    def test_the_launch_record_refuses_by_name_when_malformed_or_for_another_tree(self, tmp_path):
        prev = F._plan(W_PREV, materialisation="draft")
        d = F._prior_dir(tmp_path, prev)
        _launched(d, W_THIS, {"S5": F.S5_WAVE})
        with pytest.raises(ValueError, match="records tree"):
            co.load_prior_source(d)
        (d / co.SMOKES_LAUNCHED_REL).write_text(json.dumps({"schema": co.SMOKES_LAUNCHED_SCHEMA, "W": W_PREV, "launched": {"S5": 5}}))
        with pytest.raises(ValueError, match="smoke id -> wave uuid"):
            co.load_prior_source(d)
        (d / co.SMOKES_LAUNCHED_REL).unlink()
        assert co.load_prior_source(d)["smokes_launched"] is None
        v, note = co.missing_dict_verdict({"smokes_launched": None, "path": "/rs/prior/x"}, source="production", wave_uuid=WAVE)
        assert v == co.DICT_HISTORY_UNKNOWN and note.startswith(f"/rs/prior/x carries no {co.SMOKES_LAUNCHED_REL}: launch history unknown -- write the record ({co.SMOKES_LAUNCHED_SCHEMA}")
        rec = {"smokes_launched": {"launched": {"S5": F.S5_WAVE}, "production": None, "path": "/rs/prior/x/smokes_launched.json"}}
        assert co.missing_dict_verdict(rec, source="s5_smoke", wave_uuid=F.S5_WAVE)[0] == co.DICT_VANISHED
        assert co.missing_dict_verdict(rec, source="current_s5_full", wave_uuid=F.S5_WAVE)[0] == co.DICT_VANISHED
        assert co.missing_dict_verdict(rec, source="s3_smoke", wave_uuid=SHARED_MINTS["S3"])[0] == co.DICT_NEVER_LAUNCHED
        assert co.missing_dict_verdict(rec, source="production", wave_uuid=WAVE)[0] == co.DICT_NEVER_LAUNCHED
        assert co.missing_dict_verdict(rec, source="s5_smoke", wave_uuid=SHARED_MINTS["S3"])[0] == co.DICT_HISTORY_UNKNOWN

    def test_the_w25_shaped_four_prior_world_evaluates_only_with_all_four_launch_records(self, tmp_path):
        """README-shaped facts: W22 launched S1/S1b/S2/S3/S4/S5 (S5 FULL accepted for the row), W19 S3+S5 (S5 Dict exists, nothing accepted for the
        row), W23 S2 only, W24 S3+S5 (nothing accepted for the row); production null everywhere; no production Dict exists; every archive
        carries all six shared mints. With all four records the evaluation carries W22's S5 with every other missing Dict `never_launched`;
        with one record missing the row's candidates from that prior are launch-history-unknown (held, had nothing carried)."""
        files, dicts = {}, {}
        this = F._plan(W_THIS, materialisation="final")
        W22, W19, W23, W24 = "c" * 40, "d" * 40, "e" * 40, "f" * 40
        facts = {W22: ("S1", "S1b", "S2", "S3", "S4", "S5"), W19: ("S3", "S5"), W23: ("S2",), W24: ("S3", "S5")}
        ws = {}
        for W, name, accepted in ((W22, "w22", "S5"), (W19, "w19", None), (W23, "w23", None), (W24, "w24", None)):
            ws[W] = self._prior(tmp_path / name, W, dicts=dicts, files=files, accepted_smoke=accepted, name=name,
                                launched={sid: SHARED_MINTS[sid] for sid in facts[W]}, production=None)
            for sid in facts[W]:
                dicts.setdefault(ws[W].dict_of[sid], {})                                     # the launched smokes' Dicts exist (empty unless accepted)
        order = [ws[W19].prior, ws[W23].prior, ws[W22].prior, ws[W24].prior]                  # operator order: W22 third
        rec = self._eval(this, order, files, dicts, list(facts))
        assert rec["n_carried"] == 1 and rec["carried"][F.SID]["carried_from"]["W"] == W22 and rec["carried"][F.SID]["carried_from"]["source"] == "s5_smoke"
        tried = [(a["prior_W"][:1], a["source"], "dict_missing" in a, bool(a.get("launch_history_unknown"))) for a in rec["carried"][F.SID]["refused_before"]]
        # four production Dicts never launched (production null), then the S5 sources in operator order: W19's S5 Dict EXISTS and names nothing for
        # the row, W23 never launched S5 (record omits it), W22's carries; nothing is launch-history-unknown
        assert tried == [("d", "production", True, False), ("e", "production", True, False), ("c", "production", True, False), ("f", "production", True, False),
                         ("d", "s5_smoke", False, False), ("e", "s5_smoke", True, False)]
        assert "no ACCEPTED attempt" in rec["carried"][F.SID]["refused_before"][4]["why"]
        assert co.carry_over_problems(rec, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) == [] and co.carry_refused_holds(rec) == {}
        # one record missing (W23's): its candidates are launch-history-unknown -- refused by name; the row still carries from W22, so no hold is needed
        (ws[W23].dir / co.SMOKES_LAUNCHED_REL).unlink()
        order[1] = co.load_prior_source(ws[W23].dir)
        rec2 = self._eval(this, order, files, dicts, list(facts))
        unknown = [(a["prior_W"][:1], a["source"]) for a in rec2["carried"][F.SID]["refused_before"] if a.get("launch_history_unknown")]
        assert rec2["n_carried"] == 1 and unknown == [("e", "production"), ("e", "s5_smoke")]
        # ... and had W22's S5 acceptance been unreadable too, the row would be HELD on the unknown history rather than spawned
        dicts.pop(ws[W22].dict_of["S5"])
        with pytest.raises(ValueError, match="vanished"):                                        # W22's record names S5: its missing Dict is a vanished one
            self._eval(this, order, files, dicts, list(facts))
        (ws[W22].dir / co.SMOKES_LAUNCHED_REL).unlink()
        order[2] = co.load_prior_source(ws[W22].dir)
        rec3 = self._eval(this, order, files, dicts, list(facts))
        assert rec3["n_carried"] == 0 and co.carry_refused_holds(rec3)[F.SID]["hold_kind"] == co.DICT_HISTORY_UNKNOWN


# --------------------------------------------------------------------------- fj21: certification refuses while a launch-history-unknown hold stands

class TestCertificateRefusesUnknownHistoryHolds:
    """fj21: the fj20 hold blocks `select_launch_rows`, but a retry pass / --reattempt spawns without reading holds -- so `carry_over_problems` (the
    certificate's carry-block validator, `lp.certificate_problems` -> `co.carry_over_problems`) names every launch-history-unknown hold: no
    certificate issues, --go's re-validation refuses, until the prior's launch record is written and the carry re-evaluated."""

    def test_one_unknown_history_hold_is_a_named_problem_and_a_record_clears_it(self, tmp_path):
        T2 = TestMissingDictNeverMintedVsVanished()
        files, dicts = {}, {}
        this = F._plan(W_THIS, materialisation="final")
        w = T2._prior(tmp_path, W_PREV, dicts=dicts, files=files, accepted_smoke="S5")            # COMPLETE S5 root, no launch record
        dicts.pop(w.dict_of["S5"])                                                                 # the Dict is gone
        rec = T2._eval(this, [w.prior], files, dicts, [W_PREV])
        assert co.carry_refused_holds(rec)[F.SID]["hold_kind"] == co.DICT_HISTORY_UNKNOWN
        for probs in (co.carry_over_problems(rec, plan=this), co.carry_over_problems(rec, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts))):
            named = [p for p in probs if p.startswith(f"carry hold {co.DICT_HISTORY_UNKNOWN} on {F.SID}: write ")]
            assert len(named) == 1 and f"{w.prior['path']}/{co.SMOKES_LAUNCHED_REL} ({co.SMOKES_LAUNCHED_SCHEMA}) and re-evaluate before certify" in named[0], probs
        # the certificate validator IS this function: a problem here refuses issuance (pinned the way fj2 pins the carry block's consumption)
        import inspect
        assert 'co.carry_over_problems(cert["carry_over"]' in inspect.getsource(lp.certificate_problems)
        # the hold resolved by a record naming S3 only (S5 never launched here) -> no hold, nothing named; a record naming S5 -> vanished (refused at evaluation)
        _launched(w.dir, W_PREV, {"S3": SHARED_MINTS["S3"]}); dicts[w.dict_of["S3"]] = {}
        rec2 = T2._eval(this, [co.load_prior_source(w.dir)], files, dicts, [W_PREV])
        assert co.carry_refused_holds(rec2) == {} and co.carry_over_problems(rec2, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) == []
        # an accepted-attempt hold (fj7 (B): a completed attempt the carry refused) is NOT a certificate problem -- the launcher's belt as before
        w3 = T2._prior(tmp_path / "p3", W_PREV, dicts=dicts, files=files, accepted_smoke="S5", name="prior3", launched={"S5": F.S5_WAVE})
        root3 = lp.attempt_root_for(w3.idents["S5"]["results_root"], leg="pasttense", view="s1", stage="base", shard=0, attempt_id=f"01TESTTASK-{W_PREV[:8]}", kind="base")
        files[f"{root3}/RESULT"] = "status=FAILED rc=1"
        rec3 = T2._eval(this, [w3.prior], files, dicts, [W_PREV])
        assert rec3["n_carried"] == 0 and co.carry_refused_holds(rec3)[F.SID]["hold_kind"] == "accepted_attempt"
        assert not [p for p in co.carry_over_problems(rec3, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) if "carry hold" in p]

    def test_the_four_prior_w25_world_with_its_records_has_no_such_hold_and_is_clean(self, tmp_path):
        T2 = TestMissingDictNeverMintedVsVanished()
        files, dicts = {}, {}
        this = F._plan(W_THIS, materialisation="final")
        W22, W19, W23, W24 = "c" * 40, "d" * 40, "e" * 40, "f" * 40
        facts = {W22: ("S1", "S1b", "S2", "S3", "S4", "S5"), W19: ("S1", "S1b", "S2", "S3", "S5"), W23: ("S2",), W24: ("S3", "S5")}   # the real records' shape
        ws = {}
        for W, name, accepted in ((W22, "w22", "S5"), (W19, "w19", None), (W23, "w23", None), (W24, "w24", None)):
            ws[W] = T2._prior(tmp_path / name, W, dicts=dicts, files=files, accepted_smoke=accepted, name=name, launched={sid: SHARED_MINTS[sid] for sid in facts[W]}, production=None)
            for sid in facts[W]:
                dicts.setdefault(ws[W].dict_of[sid], {})
        rec = T2._eval(this, [ws[W19].prior, ws[W23].prior, ws[W22].prior, ws[W24].prior], files, dicts, list(facts))
        assert rec["n_carried"] == 1 and co.carry_refused_holds(rec) == {}
        assert co.carry_over_problems(rec, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) == []
        # take W23's record away: the row still carries from W22 (no hold), so the certificate stays clean; take W22's acceptance away too and it refuses by name
        (ws[W23].dir / co.SMOKES_LAUNCHED_REL).unlink()
        order = [ws[W19].prior, co.load_prior_source(ws[W23].dir), ws[W22].prior, ws[W24].prior]
        assert co.carry_over_problems(T2._eval(this, order, files, dicts, list(facts)), plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts)) == []
        dicts.pop(ws[W22].dict_of["S5"]); (ws[W22].dir / co.SMOKES_LAUNCHED_REL).unlink(); order[2] = co.load_prior_source(ws[W22].dir)
        rec3 = T2._eval(this, order, files, dicts, list(facts))
        probs = co.carry_over_problems(rec3, plan=this, read_text=files.get, git=_anygit(), accepted_get_for_dict=_agd(dicts))
        assert len(probs) == 1 and probs[0].startswith(f"carry hold {co.DICT_HISTORY_UNKNOWN} on {F.SID}: write ") and ws[W23].prior["path"] in probs[0] and ws[W22].prior["path"] in probs[0]


# --------------------------------------------------------------------------- byte-identical reproduction is not applicable to a sampled decode; a budgeted acceptance holds nothing

class TestD180Criterion3SampledDecode:
    """fj22: the first real carry-over resolved the right FULL attempts and refused all three at clause (c) on `determinism` -- the pasttense
    record PRESENT with 3 of 40 reproduced (a sampled decode never replays byte-identically), misalign / controls ABSENT. A present, diverged record on
    a SAMPLED row (the resolved_sampler's client temperatures > 0) is a QUALIFICATION on the carried entry; a GREEDY row keeps the refusal; an
    absent record keeps the waiver path. And a budgeted acceptance refused as "never a carry source" holds nothing."""

    MISSES = [{"arm": "none", "item_id": "jbb_harmful_34", "sample_idx": 0, "first_divergence_char": 495, "first_divergence_token": 71}]

    def _world(self, tmp_path, *, temperature, diverged=True, absent=False):
        prev = F._plan(W_PREV, materialisation="final")
        this = F._plan(W_THIS, materialisation="final")
        files, dicts = {}, {}
        prow = prev["standups"][0]
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": prev["results_root"], "dicts": {"accepted": prev["dicts"]["accepted"]}}
        root = F._root_files(files, plan=prev, row=prow, identity=ident)
        dicts[prev["dicts"]["accepted"]] = {prow["accepted_key"]: AID}
        det_leg, det_rel = lp.determinism_rels(prow)[0]
        ev = json.loads(files[f"{root}/modal_evidence.json"])
        if absent:
            del files[f"{root}/{det_rel}"]
        elif diverged:
            det = json.dumps({"leg": det_leg, "arms": ["none"], "n_items": 20, "n_requests": 40, "n": 40, "identical": 3, "reproduces": False, "misses": self.MISSES, "standup_id": prow["standup_id"]})
            files[f"{root}/{det_rel}"] = det
            ev[lp.DETERMINISM_EVIDENCE_KEY][det_leg] = {"sha256": F._sha(det), "writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": W_PREV, "rc": 0}
        sampler = json.dumps({**json.loads(files[f"{root}/{lp.RESOLVED_SAMPLER_REL}"]),
                              "distinct_resolved": [{"sampler": {"temperature": temperature, "top_p": 0.8, "top_k": 20}, "n": 417}]})
        files[f"{root}/{lp.RESOLVED_SAMPLER_REL}"] = sampler
        ev[lp.PRODUCER_EVIDENCE_KEY][lp.RESOLVED_SAMPLER_REL] = {"sha256": F._sha(sampler), "writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": W_PREV, "rc": 0}
        files[f"{root}/modal_evidence.json"] = json.dumps(ev)
        prior = co.load_prior_source(F._prior_dir(tmp_path, prev))
        _launched(Path(prior["path"]), W_PREV, {}, production=WAVE); prior = co.load_prior_source(Path(prior["path"]))
        return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, prior=prior, root=root, det_path=f"{root}/{det_rel}",
                                     git=F.FakeGit({W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS)}))

    def _eval(self, w, **kw):
        return co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=lambda n: w.dicts.get(n, {}).get,
                                      git=w.git, readiness_problems=F.fake_readiness, **kw)

    @staticmethod
    def _set_sampler(w, doc):
        """Rewrite the attempt's resolved_sampler record AND its producer binding (the evidence names the bytes)."""
        text = json.dumps(doc)
        w.files[f"{w.root}/{lp.RESOLVED_SAMPLER_REL}"] = text
        ev = json.loads(w.files[f"{w.root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][lp.RESOLVED_SAMPLER_REL]["sha256"] = F._sha(text)
        w.files[f"{w.root}/modal_evidence.json"] = json.dumps(ev)

    def test_a_sampled_row_with_a_present_diverged_record_carries_with_the_qualification(self, tmp_path):
        w = self._world(tmp_path, temperature=0.7)
        rec = self._eval(w)
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][F.SID]
        c3 = e["criterion3"]
        assert c3["verdict"] == co.CRITERION3_NOT_APPLICABLE_SAMPLED == "not_applicable_sampled_decode" and c3["ruling"] == "D180"
        assert (c3["reproduced"], c3["resent"], c3["misses"], c3["records"]) == (3, 40, self.MISSES, [w.det_path])
        assert c3["decode"]["temperatures"] == [0.7] and c3["decode"]["source"] == [{"path": f"{w.root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": F._sha(w.files[f"{w.root}/{lp.RESOLVED_SAMPLER_REL}"]),
                                                                              "authority": co.DECODE_AUTHORITY_HISTORICAL}]                    # fj23: the verified source, named
        det = e["clauses"]["complete"]["checks"]["determinism"]
        assert det["ok"] and det["criterion3"] == c3 and "waived" not in det
        src, = det["sources"]
        assert src["reproduces"] is False and src["identical"] == 3 and src["n_requests"] == 40 and src["criterion3"] == co.CRITERION3_NOT_APPLICABLE_SAMPLED
        assert w.det_path in e["evidence_inventory"] and f"{w.root}/{lp.RESOLVED_SAMPLER_REL}" in e["evidence_inventory"]    # both digest-bound
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        # the certificate refuses a qualification that is not D180's, a stale copy on the entry, or one resting on a greedy temperature
        for mutate, needle in ((lambda r: r["carried"][F.SID]["criterion3"].update(ruling="D999"), "is not the complete clause's"),
                               (lambda r: r["carried"][F.SID]["clauses"]["complete"]["checks"]["determinism"]["criterion3"].update(verdict="reproduced"), "is not D180's"),
                               (lambda r: r["carried"][F.SID]["clauses"]["complete"]["checks"]["determinism"]["criterion3"]["decode"].update(temperatures=[0.0]), "is not D180's"),
                               (lambda r: r["carried"][F.SID]["clauses"]["complete"]["checks"]["determinism"].update(waived="x", verdict=co.DETERMINISM_WAIVED), "both waived")):
            bad = copy.deepcopy(rec); mutate(bad)
            assert any(needle in p for p in co.carry_over_problems(bad, plan=w.this)), (needle, co.carry_over_problems(bad, plan=w.this))
        # a row whose record reproduced in full carries with NO qualification (unchanged)
        w2 = self._world(tmp_path / "ok", temperature=0.7, diverged=False)
        e2 = self._eval(w2)["carried"][F.SID]
        assert "criterion3" not in e2 and "criterion3" not in e2["clauses"]["complete"]["checks"]["determinism"]

    def test_a_greedy_row_with_a_diverged_record_still_refuses_and_an_unstated_decode_fails_closed(self, tmp_path):
        w = self._world(tmp_path, temperature=0.0)
        rec = self._eval(w)
        r = rec["refused"][F.SID]
        assert rec["n_carried"] == 0 and r["clause"] == "complete" and "3 of 40 re-sent request(s) reproduced byte-identically" in r["why"] and "criterion 3 requires every one" in r["why"]
        assert co.carry_refused_holds(rec)[F.SID]["hold_kind"] == "accepted_attempt"                      # a FULL acceptance refused on (c): held, as before
        # a mixed decode (one greedy class among sampled ones) or a record stating no temperature: determinism applies in full
        w3 = self._world(tmp_path / "mixed", temperature=0.7)
        doc = json.loads(w3.files[f"{w3.root}/{lp.RESOLVED_SAMPLER_REL}"]); doc["distinct_resolved"].append({"sampler": {"temperature": 0.0}, "n": 3})
        self._set_sampler(w3, doc)
        assert "criterion 3 requires every one" in self._eval(w3)["refused"][F.SID]["why"]
        doc["distinct_resolved"] = []; self._set_sampler(w3, doc)                                     # the record states no decode: fail closed, by name (fj23)
        why = self._eval(w3)["refused"][F.SID]["why"]
        assert "the D180 exemption cannot be judged" in why and "states no client decode temperature" in why
        # a diverged record whose bytes are not the producer's refuses on the binding, never on the relaxed text
        w4 = self._world(tmp_path / "tamper", temperature=0.7)
        w4.files[w4.det_path] = w4.files[w4.det_path] + " "
        assert "not the bytes the producer bound" in self._eval(w4)["refused"][F.SID]["why"]
        assert co.resolved_decode_temperatures({"distinct_resolved": []}) is None and co.resolved_decode_temperatures({"classes": {"client": {"distinct_resolved": [{"sampler": {"temperature": 0.7}}]}}}) == [0.7]
        assert co.resolved_decode_temperatures({"distinct_resolved": [{"sampler": {"temperature": "0.7"}}]}) is None

    # ---- fj23: the decode comes from the record the resolved-sampler check VERIFIED, the historical one cross-checked

    @staticmethod
    def _rederiving_reaudit(w, *, fresh_temperature=None, after=None):
        """A re-audit double in fc11d's shape (`seedrep_served.reaudit_carried_root`): every audit verdict re-issued under THIS tree into the carried
        evidence root, and the sampler verdict RE-DERIVED there -- the historical record re-issued (``fresh_temperature`` overrides its client
        temperature) and bound under `resolved_sampler.json` with tree_sha = this W. ``after(kw)`` runs once the fresh record exists, so a
        caller can tamper the HISTORICAL bytes after the fresh derivation."""
        def reaudit(*, prev_row, this_row, prior, aid, root, this_plan):
            ev = json.loads(w.files[f"{root}/modal_evidence.json"])
            hist = lp.producer_map_for("request_evidence_audit", ev)
            ev_root = f"/vol/seedrep/{this_plan['W7']}/{this_plan['wave_uuid']}/carried/{this_row['standup_id']}__{aid}/evidence"
            bindings = {}
            hist_rs = json.loads(w.files[f"{root}/{lp.RESOLVED_SAMPLER_REL}"])
            fresh_doc = {**hist_rs, "why": f"re-derived under {this_plan['W'][:7]}"}
            if fresh_temperature is not None:
                fresh_doc["distinct_resolved"] = [{"sampler": {**r["sampler"], "temperature": fresh_temperature}, "n": r["n"]} for r in hist_rs["distinct_resolved"]]
            fresh_rs = json.dumps(fresh_doc)
            w.files[f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}"] = fresh_rs
            for path, child, table_leg in lp.smoke_audit_sources(prev_row, root, hist):
                doc = json.loads(w.files[path])
                doc.update(W=this_plan["W"], store_W=prior["W"], audited_at=this_plan["W"], resolved_sampler_verdict="AGREES",
                           resolved_sampler_source={"path": f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": F._sha(fresh_rs), "rederived_by": this_plan["W"]})
                text = json.dumps(doc)
                bound_rel = path[len(root) + 1:]
                w.files[f"{ev_root}/{bound_rel}"] = text
                plan_rel = lp.audit_verdict_rel(table_leg, child["arm"], child["s"], stage=this_row["stage"], shard=this_row["shard"])
                bindings[plan_rel] = {"sha256": F._sha(text), "writer": lp.PRODUCER_WRITERS["request_evidence_audit"], "tree_sha": this_plan["W"], "rc": 0, "bound_rel": bound_rel,
                                      "plan_arm": child["arm"], "plan_cell": str(child.get("cell") or child["arm"]), "store_arm": doc.get("arm")}
            bindings[lp.RESOLVED_SAMPLER_REL] = {"sha256": F._sha(fresh_rs), "writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": this_plan["W"], "rc": 0}
            rec = {"reaudited_by": this_plan["W"], "evidence_root": ev_root, "store_root": root, "producer_evidence": bindings, "rc": 0,
                   "argv": ["python", lp.PRODUCER_WRITERS["request_evidence_audit"], "--audited-at", this_plan["W"]], "tool": lp.PRODUCER_WRITERS["request_evidence_audit"],
                   "n_files": len(bindings), "sampler": {"historical_verdict": hist_rs.get("verdict"), "rederived_verdict": "AGREES", "rederived_by": this_plan["W"], "recorder_rc": 0}}
            if after is not None:
                after(root=root, ev_root=ev_root)
            return rec
        return reaudit

    def test_sol_a_verified_greedy_fresh_sampler_with_a_tampered_historical_one_at_0_7_never_earns_the_exemption(self, tmp_path):
        """Producer-bound historical and fresh samplers both greedy, replay 3/40 -> refused; then ONLY the historical bytes changed to 0.7
        (its producer digest unchanged, the fresh still 0) -> fj22 carried the row qualified; now refused by name."""
        w = self._world(tmp_path, temperature=0.0)
        rec = self._eval(w, reaudit=self._rederiving_reaudit(w))
        assert rec["n_carried"] == 0 and "criterion 3 requires every one" in rec["refused"][F.SID]["why"]        # both greedy: refused (correct before and after)

        w2 = self._world(tmp_path / "tampered", temperature=0.0)

        def tamper(root, ev_root):                                                                    # runs AFTER the fresh record was derived from the untampered bytes
            doc = json.loads(w2.files[f"{root}/{lp.RESOLVED_SAMPLER_REL}"])
            doc["distinct_resolved"] = [{"sampler": {**r["sampler"], "temperature": 0.7}, "n": r["n"]} for r in doc["distinct_resolved"]]
            w2.files[f"{root}/{lp.RESOLVED_SAMPLER_REL}"] = json.dumps(doc)                       # bytes changed, the producer's digest NOT
        rec2 = self._eval(w2, reaudit=self._rederiving_reaudit(w2, after=tamper))
        assert rec2["n_carried"] == 0
        why = rec2["refused"][F.SID]["why"]
        assert "the D180 exemption cannot be judged" in why and "is not the bytes its producer bound" in why and "tampered evidence never grounds" in why, why
        assert co.carry_refused_holds(rec2)[F.SID]["hold_kind"] == "accepted_attempt"                 # a FULL acceptance refused on (c): held, not spawned

    def test_fresh_sampled_and_historical_sampled_qualify_with_the_fresh_source_recorded_and_cross_checked(self, tmp_path):
        w = self._world(tmp_path, temperature=0.7)
        rec = self._eval(w, reaudit=self._rederiving_reaudit(w))
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][F.SID]
        c3 = e["criterion3"]
        src, = c3["decode"]["source"]
        assert c3["decode"]["authority"] == co.DECODE_AUTHORITY_REAUDIT == src["authority"] and "/carried/" in src["path"] and src["path"].endswith(f"/evidence/{lp.RESOLVED_SAMPLER_REL}")
        assert src["sha256"] == F._sha(w.files[src["path"]]) and c3["decode"]["temperatures"] == [0.7]
        hist, = c3["decode"]["cross_check"]["historical"]
        assert c3["decode"]["cross_check"]["agrees"] is True and hist["path"] == f"{w.root}/{lp.RESOLVED_SAMPLER_REL}" and hist["sha256"] == F._sha(w.files[hist["path"]])
        assert src["path"] in e["evidence_inventory"] and hist["path"] in e["evidence_inventory"]      # both digest-bound, re-read at certify
        smp = e["clauses"]["complete"]["checks"]["resolved_sampler"]
        assert smp["sources"][0]["path"] == src["path"] and smp["sources"][0]["sha256"] == src["sha256"] and smp["reaudit"]["evidence_root"] in src["path"]
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []
        # a qualification whose recorded source is rewritten, or lacks the source / authority, refuses at certify
        bad = copy.deepcopy(rec); bad["carried"][F.SID]["criterion3"]["decode"]["source"] = []; bad["carried"][F.SID]["clauses"]["complete"]["checks"]["determinism"]["criterion3"]["decode"]["source"] = []
        assert any("verified resolved-sampler source" in p for p in co.carry_over_problems(bad, plan=w.this))
        w.files[src["path"]] = w.files[src["path"]] + " "
        assert any("no longer reads the verified bytes" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)))
        # the fresh record and a VERIFIED historical one that disagree: refused by name (never the exemption on either alone)
        w3 = self._world(tmp_path / "disagree", temperature=0.0)
        rec3 = self._eval(w3, reaudit=self._rederiving_reaudit(w3, fresh_temperature=0.7))
        assert rec3["n_carried"] == 0 and "disagree" in rec3["refused"][F.SID]["why"] and "[0.7]" in rec3["refused"][F.SID]["why"] and "[0.0]" in rec3["refused"][F.SID]["why"]

    def test_without_a_fresh_sampler_the_verified_historical_binding_is_the_authority_and_a_broken_one_refuses(self, tmp_path):
        w = self._world(tmp_path, temperature=0.7)
        rec = self._eval(w)                                                                           # no re-audit: the producer-bound historical record
        c3 = rec["carried"][F.SID]["criterion3"]
        assert c3["decode"]["authority"] == co.DECODE_AUTHORITY_HISTORICAL and c3["decode"]["cross_check"] is None
        assert c3["decode"]["source"] == [{"path": f"{w.root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": F._sha(w.files[f"{w.root}/{lp.RESOLVED_SAMPLER_REL}"]), "authority": co.DECODE_AUTHORITY_HISTORICAL}]
        # the historical binding broken (bytes changed, digest unchanged) and no fresh record: the resolved_sampler check refuses first -- never the exemption
        w2 = self._world(tmp_path / "broken", temperature=0.7)
        w2.files[f"{w2.root}/{lp.RESOLVED_SAMPLER_REL}"] = w2.files[f"{w2.root}/{lp.RESOLVED_SAMPLER_REL}"] + " "
        rec2 = self._eval(w2)
        assert rec2["n_carried"] == 0 and rec2["refused"][F.SID]["why"].startswith("check 'resolved_sampler'") and "not the bytes the producer bound" in rec2["refused"][F.SID]["why"]
        # the helper itself: no verified source -> refuses
        assert co.verified_resolved_decode(None, root=w.root, prev_row=w.prev["standups"][0], evidence={}, read_text=w.files.get)["ok"] is False
        assert co.verified_resolved_decode({"sources": []}, root=w.root, prev_row=w.prev["standups"][0], evidence={}, read_text=w.files.get)["ok"] is False

    def test_an_absent_record_keeps_the_d118_waiver_path(self, tmp_path):
        w = self._world(tmp_path, temperature=0.7, absent=True)
        rec = self._eval(w)
        assert rec["n_carried"] == 0 and "absent on the volume" in rec["refused"][F.SID]["why"] and "criterion 3: a waiver" in rec["refused"][F.SID]["why"]
        waiver = {"by": "operator", "at": "2026-09-20T00:00:00Z", "scope": {"W": W_PREV, "leg": "pasttense", "view": "s1", "legs": [0]},       # the waiver format wants a non-empty legs list even off-Petri
                  "reason": "D180: sampled decode", "decision": "criterion 3 waived by D180"}
        rec2 = self._eval(w, criterion3_waivers=[waiver])
        assert rec2["n_carried"] == 1, rec2["refused"]
        e = rec2["carried"][F.SID]
        assert e["criterion3_waiver"] == waiver and e["clauses"]["complete"]["checks"]["determinism"]["verdict"] == co.DETERMINISM_WAIVED and "criterion3" not in e
        assert co.carry_over_problems(rec2, plan=w.this, read_text=w.files.get, git=w.git, accepted_get_for_dict=_agd(w.dicts)) == []


class TestBudgetedAcceptanceHoldsNothing:
    """fj22 (2): an acceptance refused ONLY because the smoke was BUDGETED (never a carry source; `budgeted: true` on the attempt)
    holds nothing when the EVIDENCE proves the budget (executed shape `limited` -- fj13); an `unknown` executed shape RETAINS the hold even on a
    budgeted candidate (the fj14 rule, pinned by TestExecutedShapeBoundToTheAttempt -- lifting it is out of scope for this item); a
    proven-FULL acceptance refused for any reason stays held."""

    def _rec(self, attempts):
        return {"schema": co.CARRY_OVER_SCHEMA, "carried": {}, "refused": {"bon__s1__base__shard00": {"accepted_key": "k", "attempts": attempts}}}

    @staticmethod
    def _paa(shape, full_shape=None):
        return {"attempt_id": AID, "attempt_root": "/vol/x", "executed_shape": {"shape": shape, "why": "t"}, "full_shape": (shape != co.EXECUTED_SHAPE_LIMITED) if full_shape is None else full_shape,
                "holds": shape != co.EXECUTED_SHAPE_LIMITED}

    def test_budgeted_refusals_hold_nothing_when_the_evidence_proves_the_budget(self):
        budgeted_unknown = {"prior": "p", "prior_W": W_PREV, "source": "s3_smoke", "clause": "row_identity", "budgeted": True, "attempt_id": AID,
                            "why": "the prior S3 row's child none s1 carries the smoke-only budget flag(s) ['--limit-goals'] -- BUDGETED (D107), never a carry source",
                            "prior_accepted_attempt": self._paa(co.EXECUTED_SHAPE_UNKNOWN)}
        assert co.carry_refused_holds(self._rec([budgeted_unknown]))["bon__s1__base__shard00"]["attempt_id"] == AID   # unknown RETAINS -- budgeted or not
        budgeted_limited = {**budgeted_unknown, "prior_accepted_attempt": self._paa(co.EXECUTED_SHAPE_LIMITED)}
        assert co.carry_refused_holds(self._rec([budgeted_limited])) == {}                              # the evidence proves the budget: no hold (fj13), the row spawns
        budgeted_full = {**budgeted_unknown, "prior_accepted_attempt": self._paa(co.EXECUTED_SHAPE_FULL)}
        h = co.carry_refused_holds(self._rec([budgeted_full]))["bon__s1__base__shard00"]                # the evidence PROVES full, the reconstruction says budgeted: held
        assert h["hold_kind"] == "accepted_attempt" and h["attempt_id"] == AID
        identity = {**budgeted_unknown, "budgeted": False, "why": "the row spells differently in the prior plan: ['children[0].argv[3]']", "prior_accepted_attempt": self._paa(co.EXECUTED_SHAPE_FULL)}
        assert co.carry_refused_holds(self._rec([identity]))["bon__s1__base__shard00"]["attempt_id"] == AID    # a FULL acceptance refused on identity: held
        identity_unknown = {**identity, "prior_accepted_attempt": self._paa(co.EXECUTED_SHAPE_UNKNOWN)}
        assert co.carry_refused_holds(self._rec([identity_unknown]))["bon__s1__base__shard00"]["attempt_id"] == AID   # unknown RETAINS the hold when not budgeted
        # the launcher: the budgeted-and-proven-limited row spawns; the identity-refused row does not
        assert co.carry_refused_holds(self._rec([budgeted_limited, identity])) != {} and co.carry_refused_holds(self._rec([budgeted_limited])) == {}

    def test_the_evaluator_marks_both_budgeted_refusal_sites(self, tmp_path):
        # (a) the budgeted flag on today's reconstruction (bon S3) and the S1/S1b/S2/S4 budgeted-by-id branch
        prev_row = F._row(W_PREV, W_PREV[:7], sid="bon__s1__base__shard00", leg="bon")
        this_row = F._row(W_THIS, W_THIS[:7], sid="bon__s1__base__shard00", leg="bon")
        prev = F._plan(W_PREV, materialisation="draft", rows=[prev_row])
        sident = lp.smoke_identity(prev, "S3", F.S5_WAVE)
        srow = lp.smoke_row(prev_row, sident, max_children=1, max_timeout_s=60)
        v = co.row_identity_verdict(srow, this_row, prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE}, this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE}, prev_smoke_wave=F.S5_WAVE)
        assert not v["ok"] and v["budgeted"] is True and "BUDGETED (D107)" in v["why"]
        s1row = lp.smoke_row(prev_row, lp.smoke_identity(prev, "S1", F.S5_WAVE), max_children=1, max_timeout_s=60)
        v1 = co.row_identity_verdict(s1row, this_row, prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE}, this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE}, prev_smoke_wave=F.S5_WAVE)
        assert not v1["ok"] and v1["budgeted"] is True
        ok = co.row_identity_verdict(F._row(W_PREV, W_PREV[:7]), F._row(W_THIS, W_THIS[:7]), prev={"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE}, this={"W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE})
        assert ok["ok"] and "budgeted" not in ok
        # the evaluator copies the marker onto the attempt record (the hold reads it there)
        src = inspect.getsource(co.evaluate_carry_over)
        assert 'rec["budgeted"] = True' in src and "budgeted=True)" in src


# --------------------------------------------------------------------------- fj24: the launch tree's producer-file differences, admitted by HUNK

D183_W22, D183_W24, D183_LAUNCH = "56aff576e2975804602aec7d12ba1c98c805adfd", "6fdd5e52bd6aec62b2389c185bd1d094d902703d", "0f00f7fae98e8c719617f6d01bbdc4eb2c763c88"
D183_TRIPLES = [  # (prior, path, leg, table, n_hunks) -- the real (prior W -> launch tree) diffs
    (D183_W22, "src/sisteer/capeval/tiers/petri_runner.py", "pasttense", "D183-petri_runner", 10),
    (D183_W22, "src/sisteer/eval/attacks/attacker_pool.py", "pasttense", "D183-attacker_pool", 19),
    (D183_W22, "scripts/modal/seedrep_ladder_helpers.py", "misalign", "D183-ladder_helpers", 6),
    (D183_W22, "src/sisteer/capeval/tiers/petri_runner.py", "misalign", "D183-petri_runner", 10),
    (D183_W22, "src/sisteer/registry/hooks.py", "misalign", "D183-registry_hooks", 17),
    (D183_W24, "scripts/served_capability_run.py", "controls", "D183-served_fs4", 4),
    (D183_W24, "src/sisteer/capeval/tiers/petri_runner.py", "controls", "D183-petri_runner", 2),
    (D183_W24, "src/sisteer/registry/hooks.py", "controls", "D183-registry_hooks", 4),
]


def _real_git_or_skip():
    for sha in (D183_W22, D183_W24, D183_LAUNCH):
        if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode != 0:
            pytest.skip(f"the real tree {sha[:9]} is not in this checkout's object store")
    return co.default_git(REPO)


class TestD183AdmittedHunks:
    """fj24 (D183): the three FULL carries refused at clause (b) on producer files whose W22 / W24 -> launch changes cannot alter a static /
    controls / misalign row's generations or scoring. Each (prior, file) diff is admitted BY HUNK against the real git objects; an unlisted
    differing file, an extra hunk, or another leg's carry still refuse; the admission is validated at certify against the pinned tables."""

    def test_the_eight_real_prior_to_launch_diffs_are_admitted_by_hunk_and_labelled(self):
        git = _real_git_or_skip()
        for prior, path, leg, table, n in D183_TRIPLES:
            adm = co.admitted_difference(git, prior, D183_LAUNCH, path, leg)
            assert adm is not None, (prior[:7], path, leg)
            assert adm["tables"] == [table] and adm["n_hunks"] == n and all(h["table"] == table and "->launch" in h["label"] for h in adm["hunks"]), (path, adm["tables"], adm["n_hunks"])
            assert "D183" in adm["reason"] and adm["kind"] in (co.ADMITTED_KIND_OFF_PATH, co.ADMITTED_KIND_REGISTRY)
        # the labels attribute the hunks to the ruling's commits (blame at the launch tree)
        assert all("dbf326ead" in lab or "f29562721" in lab for lab in co.D183_PETRI_RUNNER_HUNKS["src/sisteer/capeval/tiers/petri_runner.py"].values())
        assert all("ac0401136" in lab for lab in co.D183_SERVED_FS4_HUNKS["scripts/served_capability_run.py"].values())
        assert "eecba9039" not in json.dumps(co.D183_SERVED_FS4_HUNKS)                                            # fd3: not admitted

    def test_the_three_real_producer_sets_pass_clause_b_and_an_unlisted_file_or_another_leg_refuses(self):
        git = _real_git_or_skip()
        sets = {("pasttense", D183_W22): ["src/sisteer/capeval/tiers/petri_runner.py", "src/sisteer/eval/attacks/attacker_pool.py"],
                ("misalign", D183_W22): ["scripts/modal/seedrep_ladder_helpers.py", "src/sisteer/capeval/tiers/petri_runner.py", "src/sisteer/registry/hooks.py"],
                ("controls", D183_W24): ["scripts/served_capability_run.py", "src/sisteer/capeval/tiers/petri_runner.py", "src/sisteer/registry/hooks.py"]}
        for (leg, prior), files in sets.items():
            v = co.producer_set_verdict(files, prev_sha=prior, this_sha=D183_LAUNCH, git=git, leg=leg)
            assert v["ok"], (leg, v.get("why"))
            assert sorted(a["path"] for a in v["admitted_differences"]) == sorted(files) and v["git_diff_quiet"]["rc"] == 1
            assert all(f["admitted_difference"] for f in v["files"])
        # an unlisted differing file still refuses by name
        v = co.producer_set_verdict(["src/sisteer/capeval/tiers/petri_runner.py", "scripts/modal/seedrep_served.py"], prev_sha=D183_W22, this_sha=D183_LAUNCH, git=git, leg="pasttense")
        assert not v["ok"] and "scripts/modal/seedrep_served.py" in v["why"] and "not an admitted difference" in v["why"]
        assert [a["path"] for a in v["admitted_differences"]] == ["src/sisteer/capeval/tiers/petri_runner.py"]        # the pinned file still admits; the set refuses on the other
        # a PETRI row's carry never sees the petri_runner table (that module IS its producer); a multi-turn row never sees attacker_pool's
        assert co.admitted_difference(git, D183_W22, D183_LAUNCH, "src/sisteer/capeval/tiers/petri_runner.py", "petri") is None
        assert co.admitted_difference(git, D183_W22, D183_LAUNCH, "src/sisteer/eval/attacks/attacker_pool.py", "pair") is None
        assert co.admitted_difference(git, D183_W22, D183_LAUNCH, "scripts/modal/seedrep_ladder_helpers.py", "controls") is None       # the controls-sampling path IS a controls row's
        assert co.admitted_difference(git, D183_W24, D183_LAUNCH, "scripts/served_capability_run.py", "pasttense") is None

    def test_a_listed_file_with_an_extra_unadmitted_hunk_refuses(self):
        git = _real_git_or_skip()
        path = "src/sisteer/capeval/tiers/petri_runner.py"
        prev_text, this_text = git(["show", f"{D183_W22}:{path}"])[1], git(["show", f"{D183_LAUNCH}:{path}"])[1]
        fake = F.FakeGit({D183_W22: {path: "blob-a"}, "e" * 40: {path: "blob-b"}}, sources={(D183_W22, path): prev_text, ("e" * 40, path): this_text + "\n# one more line the ruling never saw\n"})
        assert co.admitted_difference(fake, D183_W22, "e" * 40, path, "pasttense") is None
        v = co.producer_set_verdict([path], prev_sha=D183_W22, this_sha="e" * 40, git=fake, leg="pasttense")
        assert not v["ok"] and path in v["why"]
        fake_ok = F.FakeGit({D183_W22: {path: "blob-a"}, "e" * 40: {path: "blob-b"}}, sources={(D183_W22, path): prev_text, ("e" * 40, path): this_text})
        assert co.producer_set_verdict([path], prev_sha=D183_W22, this_sha="e" * 40, git=fake_ok, leg="pasttense")["ok"]

    def test_the_admission_surfaces_in_the_certificates_carry_block_and_a_forged_one_refuses(self):
        git = _real_git_or_skip()
        path = "src/sisteer/capeval/tiers/petri_runner.py"
        adm = co.admitted_difference(git, D183_W22, D183_LAUNCH, path, "pasttense")
        row = F._row(W_THIS, W_THIS[:7])
        plan = F._plan(W_THIS, materialisation="final", rows=[row])
        entry = {"accepted_key": row["accepted_key"], "attempt_id": AID, "attempt_root": f"/vol/seedrep/{W_PREV[:7]}/exp/x/{AID}",
                 "carried_from": {k: (W_PREV if k in ("W", "tree_sha") else (AID if k == "attempt_id" else "x")) for k in co.PROVENANCE_KEYS},
                 "clauses": {**{c: {"ok": True} for c in co.CLAUSES},
                             "producer_set": {"ok": True, "this_sha": W_THIS, "prev_sha": W_PREV, "files": [{"path": path, "identical": False, "admitted_difference": adm["reason"]}],
                                              "admitted_differences": [adm]}}}
        rec = {"schema": co.CARRY_OVER_SCHEMA, "W": W_THIS, "W7": W_THIS[:7], "wave_uuid": WAVE, "launch_plan_sha": plan["launch_plan_sha"], "carried": {F.SID: entry}, "refused": {}}
        assert co.carry_over_problems(rec, plan=plan) == []
        assert rec["carried"][F.SID]["clauses"]["producer_set"]["admitted_differences"][0]["tables"] == ["D183-petri_runner"]     # what the certificate's carry block shows
        forged = copy.deepcopy(rec); forged["carried"][F.SID]["clauses"]["producer_set"]["admitted_differences"][0]["hunks"][0]["sha256"] = "0" * 64
        assert any("not pinned for leg 'pasttense'" in p for p in co.carry_over_problems(forged, plan=plan))
        other_table = copy.deepcopy(rec); other_table["carried"][F.SID]["clauses"]["producer_set"]["admitted_differences"][0]["tables"] = ["D999"]
        assert any("unknown table" in p for p in co.carry_over_problems(other_table, plan=plan))
        # the same admission on a PETRI row's entry is another leg's table: refused
        prow = F._row(W_THIS, W_THIS[:7], sid="petri__s1__base__shard00", leg="petri")
        pplan = F._plan(W_THIS, materialisation="final", rows=[prow])
        prec = copy.deepcopy(rec); prec["launch_plan_sha"] = pplan["launch_plan_sha"]
        prec["carried"] = {"petri__s1__base__shard00": {**copy.deepcopy(entry), "accepted_key": prow["accepted_key"]}}
        assert any("not pinned for leg 'petri'" in p for p in co.carry_over_problems(prec, plan=pplan))


# --------------------------------------------------------------------------- fj25: clause (b) RE-DERIVED at certification

class TestProducerSetRecheckAtCertify:
    """fj25: `carry_over_problems` trusted the record's producer-set block -- recorded hash/table pairs were checked against the constants, nothing was
    recomputed, and an admission without `hunks` bypassed validation. Now, in evidence mode, clause (b) is re-derived from the trusted git objects
    and the archived plan row: the differing set must equal the admitted set, every admission's hunks / lines / blobs must equal the re-derived
    ones, a pinned-hunk-kind admission without hunks refuses by name, and missing objects refuse by name."""

    def _worlds(self, tmp_path):
        """One identical world (carried) and one whose CHILD driver -- a scoring producer -- differs between the trees (refused at (b))."""
        same = F.FakeGit({W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS)}, sources={(W_PREV, F.CHILD): "print('v1')\n", (W_THIS, F.CHILD): "print('v1')\n"})
        diff = F.FakeGit({W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS, **{F.CHILD: "changed-blob"})},
                         sources={(W_PREV, F.CHILD): "print('v1')\n", (W_THIS, F.CHILD): "print('v2')  # scoring changed\n"})
        prev = F._plan(W_PREV, materialisation="final"); this = F._plan(W_THIS, materialisation="final")
        prow = prev["standups"][0]
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": prev["results_root"], "dicts": {"accepted": prev["dicts"]["accepted"]}}
        files, dicts = {}, {}
        root = F._root_files(files, plan=prev, row=prow, identity=ident)
        dicts[prev["dicts"]["accepted"]] = {prow["accepted_key"]: AID}
        d = F._prior_dir(tmp_path, prev); _launched(d, W_PREV, {}, production=WAVE)
        prior = co.load_prior_source(d)
        ev = lambda git: co.evaluate_carry_over(this_plan=this, this_sha=W_THIS, priors=[prior], read_text=files.get, accepted_get_for_dict=lambda n: dicts.get(n, {}).get,
                                                git=git, readiness_problems=F.fake_readiness)
        return types.SimpleNamespace(this=this, files=files, dicts=dicts, root=root, same=same, diff=diff, eval=ev)

    @staticmethod
    def _petri_adm():
        git = _real_git_or_skip()
        return co.admitted_difference(git, D183_W22, D183_LAUNCH, "src/sisteer/capeval/tiers/petri_runner.py", "pasttense")

    def test_sol_1_a_changed_scoring_producer_refuses_at_evaluation(self, tmp_path):
        w = self._worlds(tmp_path)
        rec = w.eval(w.diff)
        assert rec["n_carried"] == 0 and rec["refused"][F.SID]["clause"] == "producer_set" and F.CHILD in rec["refused"][F.SID]["why"]
        assert w.eval(w.same)["n_carried"] == 1

    def test_sol_2_a_changed_file_forged_under_an_unrelated_valid_d183_admission_refuses_at_certify(self, tmp_path):
        w = self._worlds(tmp_path)
        rec = w.eval(w.same)
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.same, accepted_get_for_dict=_agd(w.dicts)) == []
        # the forgery: CHILD differs on the real trees (w.diff), the record claims it was admitted under a VALID D183 petri_runner admission
        adm = {**self._petri_adm(), "path": F.CHILD}
        forged = copy.deepcopy(rec)
        ps = forged["carried"][F.SID]["clauses"]["producer_set"]
        f = next(x for x in ps["files"] if x["path"] == F.CHILD); f.update(identical=False, admitted_difference=adm["reason"], this_blob="changed-blob")
        ps["admitted_differences"] = [adm]
        probs = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=w.diff, accepted_get_for_dict=_agd(w.dicts))
        assert probs, "the forged admission certified"
        assert any(f"{F.CHILD}: re-derived over the trees, the difference is NOT an admitted one" in p for p in probs), probs
        assert any("the admission's blobs" in p and "(a stale admission)" in p for p in probs), probs
        # and the same forgery against the record's own (identical) trees: the admitted file does not differ -> refused by name
        probs2 = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=w.same, accepted_get_for_dict=_agd(w.dicts))
        assert any("admission RECORDS" in p and "are not the files that actually differ" in p and f"record for a non-differing file: ['{F.CHILD}']" in p for p in probs2), probs2
        assert any("files MARKED admitted" in p and f"marked but not differing: ['{F.CHILD}']" in p for p in probs2), probs2
        # a differing file the record marks IDENTICAL (no admission at all) refuses by name too
        plain = copy.deepcopy(rec)
        plain_probs = co.carry_over_problems(plain, plan=w.this, read_text=w.files.get, git=w.diff, accepted_get_for_dict=_agd(w.dicts))
        assert any(f"are not the files that actually differ between {W_PREV[:12]} and {W_THIS[:12]}" in p and f"missing record: ['{F.CHILD}']" in p for p in plain_probs), plain_probs
        assert any(f"differing but unmarked: ['{F.CHILD}']" in p for p in plain_probs) and any("recorded blobs" in p and "are not the trees'" in p for p in plain_probs)

    def test_sol_3_a_d183_kind_admission_without_hunks_refuses_by_name(self, tmp_path):
        w = self._worlds(tmp_path)
        rec = w.eval(w.same)
        forged = copy.deepcopy(rec)
        ps = forged["carried"][F.SID]["clauses"]["producer_set"]
        ps["admitted_differences"] = [{"path": "scripts/unpinned/anything.py", "kind": co.ADMITTED_KIND_OFF_PATH, "reason": co.ADMITTED_REASON_D183_PETRI, "tables": ["D183-petri_runner"]}]
        probs = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=w.same, accepted_get_for_dict=_agd(w.dicts))
        assert any("an admission of kind 'off_path_code' carries no hunks" in p and "refusing" in p for p in probs), probs
        assert any("admission RECORDS" in p and "record for a non-differing file: ['scripts/unpinned/anything.py']" in p for p in probs), probs
        # metadata mode (no read_text) keeps fj24's pin check, which also refuses it
        assert any("hunk" in p.lower() or "table" in p.lower() for p in co.carry_over_problems(forged, plan=w.this))

    def test_missing_git_objects_and_a_stale_blob_refuse_by_name_and_the_recheck_needs_evidence(self, tmp_path):
        w = self._worlds(tmp_path)
        rec = w.eval(w.same)
        nogit = F.FakeGit({})                                                                        # the trees are not in this repository
        probs = co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=nogit, accepted_get_for_dict=_agd(w.dicts))
        assert any("are not available" in p and "cannot be re-derived; refusing" in p for p in probs), probs
        # the real repository does not hold the synthetic trees either: the default git refuses by name (the certificate's mode)
        assert any("are not available" in p and "cannot be re-derived; refusing" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, accepted_get_for_dict=_agd(w.dicts)))
        # a recorded blob that is not the tree's (a stale record) refuses by name
        stale = copy.deepcopy(rec)
        next(x for x in stale["carried"][F.SID]["clauses"]["producer_set"]["files"] if x["path"] == F.CHILD)["this_blob"] = "not-the-trees-blob"
        assert any("recorded blobs" in p and "are not the trees'" in p for p in co.carry_over_problems(stale, plan=w.this, read_text=w.files.get, git=w.same, accepted_get_for_dict=_agd(w.dicts)))
        # the evidence gone: the producer set cannot be re-derived -> refused by name (never skipped)
        gone = {k: v for k, v in w.files.items() if not k.endswith("/modal_evidence.json")}
        assert any("modal_evidence.json is absent" in p and "cannot be re-derived" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=gone.get, git=w.same, accepted_get_for_dict=_agd(w.dicts)))

    def test_the_three_real_carries_pass_the_recheck_and_a_metadata_only_call_is_unchanged(self, tmp_path):
        """The real (prior, file) admissions re-derive equal to themselves at certify (real git objects); metadata mode runs fj24's pin check only."""
        git = _real_git_or_skip()
        for prior, path, leg, table, n in D183_TRIPLES:
            adm = co.admitted_difference(git, prior, D183_LAUNCH, path, leg)
            again = co.admitted_difference(git, prior, D183_LAUNCH, path, leg)
            assert sorted(h["sha256"] for h in adm["hunks"]) == sorted(h["sha256"] for h in again["hunks"]) and adm["tables"] == again["tables"] == [table]
        v = co.producer_set_verdict(["src/sisteer/capeval/tiers/petri_runner.py"], prev_sha=D183_W22, this_sha=D183_LAUNCH, git=git, leg="pasttense")
        adm = v["admitted_differences"][0]
        assert adm["prev_blob"] == v["files"][0]["prev_blob"] and adm["this_blob"] == v["files"][0]["this_blob"] and adm["prev_blob"] != adm["this_blob"]


# --------------------------------------------------------------------------- fj26 (two P1s): three-set equality, and the provenance bound to the evidence

class TestRecheckBindsProvenanceAndRequiresRecords:
    """fj26: P1-A -- coverage passed on files merely MARKED admitted, without an admission record, and only present records were re-derived; now the
    differing set (git), the marked set and the record set must be pairwise equal and every ACTUAL differing file's admission is re-derived.
    P1-B -- the prior tree and archive came from carried_from alone; now the attempt's own evidence binds W / wave / attempt / standup and the
    attempt root's components before any archive is opened, and the archived row's identity is re-run against the consuming row."""

    MANIFEST_V1 = "petri:\n  auditor_fallback_models:\n    - a\n    - b\n"
    MANIFEST_V2 = "petri:\n  auditor_fallback_models:\n    - a\n"

    def _world(self, tmp_path):
        prev = F._plan(W_PREV, materialisation="final"); this = F._plan(W_THIS, materialisation="final")
        prow = prev["standups"][0]
        ident = {"W": W_PREV, "W7": W_PREV[:7], "wave_uuid": WAVE, "results_root": prev["results_root"], "dicts": {"accepted": prev["dicts"]["accepted"]}}
        files, dicts = {}, {}
        root = F._root_files(files, plan=prev, row=prow, identity=ident)
        dicts[prev["dicts"]["accepted"]] = {prow["accepted_key"]: AID}
        d = F._prior_dir(tmp_path, prev); _launched(d, W_PREV, {}, production=WAVE)
        prior = co.load_prior_source(d)
        # git A: only the MANIFEST differs, by an admitted chain-list change (the legitimately admitted module); git B: the CHILD scoring driver differs too
        src = {(W_PREV, F.MANIFEST): self.MANIFEST_V1, (W_THIS, F.MANIFEST): self.MANIFEST_V2, (W_PREV, F.CHILD): "print('v1')\n", (W_THIS, F.CHILD): "print('v2')  # scoring changed\n"}
        git_a = F.FakeGit({W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS, **{F.MANIFEST: "manifest-v2"})}, sources=src)
        git_b = F.FakeGit({W_PREV: F._blobs(W_PREV), W_THIS: F._blobs(W_THIS, **{F.MANIFEST: "manifest-v2", F.CHILD: "child-v2"})}, sources=src)
        ev = lambda git: co.evaluate_carry_over(this_plan=this, this_sha=W_THIS, priors=[prior], read_text=files.get, accepted_get_for_dict=lambda n: dicts.get(n, {}).get,
                                                git=git, readiness_problems=F.fake_readiness)
        return types.SimpleNamespace(prev=prev, this=this, files=files, dicts=dicts, prior=prior, root=root, git_a=git_a, git_b=git_b, eval=ev, tmp=tmp_path)

    def test_sol_p1a_a_differing_file_marked_admitted_without_an_admission_record_refuses(self, tmp_path):
        w = self._world(tmp_path)
        rec = w.eval(w.git_a)                                                                        # the manifest's chain change admitted, carried
        assert rec["n_carried"] == 1 and [a["path"] for a in rec["carried"][F.SID]["clauses"]["producer_set"]["admitted_differences"]] == [F.MANIFEST]
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(w.dicts)) == []
        assert w.eval(w.git_b)["refused"][F.SID]["clause"] == "producer_set"                        # the driver's change: refused at evaluation
        # the forgery: keep the valid manifest admission, record the driver's blobs CORRECTLY, mark it admitted, supply NO driver record
        forged = copy.deepcopy(rec)
        ps = forged["carried"][F.SID]["clauses"]["producer_set"]
        f = next(x for x in ps["files"] if x["path"] == F.CHILD); f.update(identical=False, admitted_difference="forged mark", this_blob="child-v2")
        probs = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=w.git_b, accepted_get_for_dict=_agd(w.dicts))
        assert probs, "the mark-without-record certified"
        assert any(f"admission RECORDS ['{F.MANIFEST}'] are not the files that actually differ" in p and f"missing record: ['{F.CHILD}']" in p for p in probs), probs
        assert any("files MARKED admitted" in p and "are not the admission RECORDS' paths" in p for p in probs), probs
        assert not any("stale admission" in p for p in probs)                                           # nothing else was wrong with the manifest admission

    def test_sol_p1b_a_provenance_redirected_to_a_matching_third_archive_refuses_on_the_evidence_binding(self, tmp_path):
        w = self._world(tmp_path)
        rec = w.eval(w.git_a)
        # a THIRD tree whose producer contents equal this tree's, archived with the same row; the record's provenance and producer block regenerated against it
        W3 = "c" * 40
        third = F._plan(W3, materialisation="final")
        d3 = F._prior_dir(w.tmp / "third", third, name="third"); _launched(d3, W3, {}, production=WAVE)
        git3 = F.FakeGit({W3: F._blobs(W_THIS, **{F.MANIFEST: "manifest-v2", F.CHILD: "child-v2"}), W_THIS: F._blobs(W_THIS, **{F.MANIFEST: "manifest-v2", F.CHILD: "child-v2"}),
                          W_PREV: F._blobs(W_PREV)})
        forged = copy.deepcopy(rec)
        e = forged["carried"][F.SID]
        e["carried_from"].update(W=W3, tree_sha=W3, W7=W3[:7], rs=str(d3))
        ps = e["clauses"]["producer_set"]
        ps.update(prev_sha=W3, this_sha=W_THIS, admitted_differences=[])
        for x in ps["files"]:
            x.update(identical=True, prev_blob=git3.blobs[W3][x["path"]], this_blob=git3.blobs[W_THIS][x["path"]]); x.pop("admitted_difference", None)
        probs = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=git3, accepted_get_for_dict=_agd(w.dicts))
        assert probs, "the redirected provenance certified"
        assert any("the prior accepted Dict" in p and "names no attempt under" in p and f"(W {W3[:12]}" in p for p in probs), probs        # fj28: the producer never accepted under W3
        dicts3 = dict(w.dicts); dicts3[lp.dict_name(W3[:7], WAVE, "accepted")] = {lp.accepted_key(W3[:7], WAVE, "pasttense", "s1", "base", 0): AID}   # were it planted: the root is not W3's canonical one
        probs = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=git3, accepted_get_for_dict=_agd(dicts3))
        assert any("is not the canonical root the accepted record and the consuming row spell" in p and f"W7 {W3[:7]}" in p for p in probs), probs
        # ... and with the root moved to W3's canonical root and the files copied UNREWRITTEN, the evidence's own W refuses the redirected provenance
        alt_root = w.root.replace(f"/{W_PREV[:7]}/", f"/{W3[:7]}/")
        for k in list(w.files):
            if k.startswith(w.root + "/"):
                w.files[k.replace(w.root, alt_root)] = w.files[k]
        moved_prov = copy.deepcopy(forged); moved_prov["carried"][F.SID]["attempt_root"] = alt_root
        probs = co.carry_over_problems(moved_prov, plan=w.this, read_text=w.files.get, git=git3, accepted_get_for_dict=_agd(dicts3))
        assert any("does not bind the provenance" in p and f"W: evidence '{W_PREV[:20]}' != provenance '{W3[:20]}'" in p and "a redirected provenance refuses by name" in p for p in probs), probs
        # the bare W mismatch alone (archive untouched): refused on the evidence binding too
        bare = copy.deepcopy(rec); bare["carried"][F.SID]["carried_from"].update(W=W3, tree_sha=W3)
        assert any("is not the canonical root" in p for p in co.carry_over_problems(bare, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(dicts3)))   # the root is under aaaaaaa
        assert any("names no attempt under" in p for p in co.carry_over_problems(bare, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(w.dicts)))     # no acceptance under W3
        # a wave / attempt / standup disagreement refuses the same way; the attempt root's components too
        for key, val, needle in (("wave_uuid", "99999999-9999-4999-8999-999999999999", "names no attempt under"), ("attempt_id", "01TESTTASK-other", "the provenance '01TESTTASK-other'"),
                                 ("standup_id", "smoke-S5__" + F.SID, "standup_id: evidence")):
            b = copy.deepcopy(rec); b["carried"][F.SID]["carried_from"][key] = val
            assert any(needle in p for p in co.carry_over_problems(b, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(w.dicts))), key
        moved = copy.deepcopy(rec)
        moved["carried"][F.SID]["attempt_root"] = alt_root
        assert any("is not the canonical root the accepted record and the consuming row spell" in p for p in co.carry_over_problems(moved, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(w.dicts)))

    def test_the_archived_rows_identity_is_re_run_against_the_consuming_row(self, tmp_path):
        w = self._world(tmp_path)
        rec = w.eval(w.git_a)
        # the archived plan rewritten so the row spells differently (another cap): the recorded row_identity clause is not trusted
        plan_path = Path(w.prior["plan_path"]); doc = json.loads(plan_path.read_text())
        doc["standups"][0]["geometry"]["cap"] = 2048
        doc["launch_plan_sha"] = lp.plan_sha(doc); plan_path.write_text(json.dumps(doc))
        probs = co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(w.dicts))
        assert any("identity against the consuming row does NOT re-derive ok" in p for p in probs), probs


# --------------------------------------------------------------------------- fj28 (fourth round): the prior accepted record first; the evidence bound at the canonical root; the proof requirement derived

class TestRecheckAuthorityChain:
    """fj28: P1 -- the provenance binding read modal_evidence.json from the ENTRY's root and validated whichever paths the recorded inventory listed;
    a combined root + evidence substitution with the ORIGINAL inventory retained certified. Now the PRIOR accepted record (the producer's own write)
    is resolved first from the consuming row and the (wave) the entry names, the canonical root is derived from it, and the evidence and
    completion sources AT that root must carry the inventory's digests. P2 -- the Petri sidecar-proof requirement is derived from the archived
    row's leg / mode and the recomputed differences, and the proof re-derived from its inputs."""

    def test_sol_p1_root_and_evidence_substitution_with_the_original_inventory_refuses(self, tmp_path):
        T = TestRecheckBindsProvenanceAndRequiresRecords(); w = T._world(tmp_path)
        rec = w.eval(w.git_a)
        assert co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(w.dicts)) == []
        # the composition: provenance + producer blobs redirected to a content-matching third tree; attempt_root moved to the third tree's canonical
        # root with copied attempt files whose evidence W is rewritten; the ORIGINAL inventory (pointing at the genuine old root) left unchanged
        W3 = "c" * 40
        third = F._plan(W3, materialisation="final"); d3 = F._prior_dir(w.tmp / "third", third, name="third"); _launched(d3, W3, {}, production=WAVE)
        git3 = F.FakeGit({W3: F._blobs(W_THIS, **{F.MANIFEST: "manifest-v2", F.CHILD: "child-v2"}), W_THIS: F._blobs(W_THIS, **{F.MANIFEST: "manifest-v2", F.CHILD: "child-v2"}), W_PREV: F._blobs(W_PREV)})
        alt_root = w.root.replace(f"/{W_PREV[:7]}/", f"/{W3[:7]}/")
        for k in list(w.files):
            if k.startswith(w.root + "/"):
                w.files[k.replace(w.root, alt_root)] = w.files[k]
        ev3 = json.loads(w.files[f"{alt_root}/modal_evidence.json"]); ev3["W"] = W3; w.files[f"{alt_root}/modal_evidence.json"] = json.dumps(ev3)
        forged = copy.deepcopy(rec); e = forged["carried"][F.SID]
        e["carried_from"].update(W=W3, tree_sha=W3, W7=W3[:7], rs=str(d3)); e["attempt_root"] = alt_root
        ps = e["clauses"]["producer_set"]; ps.update(prev_sha=W3, this_sha=W_THIS, admitted_differences=[])
        for x in ps["files"]:
            x.update(identical=True, prev_blob=git3.blobs[W3][x["path"]], this_blob=git3.blobs[W_THIS][x["path"]]); x.pop("admitted_difference", None)
        # (i) the third tree has NO accepted record: the prior Dict lookup refuses first
        probs = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=git3, accepted_get_for_dict=_agd(w.dicts))
        assert any("the prior accepted Dict" in p and "names no attempt under" in p and f"W {W3[:12]}" in p for p in probs), probs
        # (ii) even were the attacker able to plant that acceptance, the moved root's evidence is not what the ORIGINAL inventory binds
        dicts3 = dict(w.dicts); dicts3[lp.dict_name(W3[:7], WAVE, "accepted")] = {lp.accepted_key(W3[:7], WAVE, "pasttense", "s1", "base", 0): AID}
        probs2 = co.carry_over_problems(forged, plan=w.this, read_text=w.files.get, git=git3, accepted_get_for_dict=_agd(dicts3))
        assert any(f"{alt_root}/modal_evidence.json is not bound in the evidence inventory (no digest)" in p for p in probs2), probs2
        assert any(f"{alt_root}/RESULT is not bound in the evidence inventory" in p for p in probs2) and any(f"{alt_root}/COMPLETE.json is not bound" in p for p in probs2)
        # (iii) with the inventory ALSO rewritten to the moved paths, the bytes differ from the genuine digests only where forged -- the evidence W was rewritten
        forged3 = copy.deepcopy(forged); e3 = forged3["carried"][F.SID]
        e3["evidence_inventory"] = {k.replace(w.root, alt_root): v for k, v in e3["evidence_inventory"].items()}
        probs3 = co.carry_over_problems(forged3, plan=w.this, read_text=w.files.get, git=git3, accepted_get_for_dict=_agd(dicts3))
        assert any(f"{alt_root}/modal_evidence.json: the bytes at the canonical root" in p and "are not the inventory's" in p for p in probs3), probs3

    def test_the_entrys_attempt_id_and_root_must_be_the_accepted_records_and_the_reader_is_required(self, tmp_path):
        T = TestRecheckBindsProvenanceAndRequiresRecords(); w = T._world(tmp_path)
        rec = w.eval(w.git_a)
        other = copy.deepcopy(rec); other["carried"][F.SID]["attempt_id"] = other["carried"][F.SID]["carried_from"]["attempt_id"] = "01TESTTASK-other"
        probs = co.carry_over_problems(other, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(w.dicts))
        assert any(f"the prior accepted record names attempt '{AID}'" in p and "the entry names '01TESTTASK-other'" in p for p in probs), probs
        gone = {k: v for k, v in w.dicts.items()}; gone[w.prev["dicts"]["accepted"]] = {}
        assert any("names no attempt under" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=_agd(gone)))
        def raising(name):
            def get(key): raise NotFoundError(f"Dict '{name}' not found")
            return get
        assert any("names no acceptance under" in p and "NotFoundError" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=raising))
        # no reader injected: the default is the LIVE Modal Dicts -- not importable here (or a stub without Dict) -> refused by name, never skipped
        assert any("names no acceptance under" in p and "no live accepted-Dict reader" in p for p in co.carry_over_problems(rec, plan=w.this, read_text=w.files.get, git=w.git_a, accepted_get_for_dict=None))

    def test_sol_p2_the_proof_requirement_is_derived_and_a_failing_or_missing_proof_refuses(self, tmp_path):
        T = TestPetriSidecarProof()
        rec, calls, this, files, git, dicts = T._petri_world(tmp_path / "ok", proof_ok=True)
        assert rec["n_carried"] == 1
        proof_stub = calls[0]["_proof"]
        ok = co.carry_over_problems(rec, plan=this, read_text=files.get, git=git, accepted_get_for_dict=_agd(dicts), petri_proof=proof_stub)
        assert ok == [], ok
        # probe 1: the recorded proof FAILED (removed_leg.uses = 2) -> refused by name (the record contains a proof; that decides nothing)
        p1 = copy.deepcopy(rec); p1["carried"]["petri__s1__base__shard00"]["clauses"]["producer_set"]["petri_sidecar_proof"] = {"ok": False, "cells": 8, "removed_leg": {"uses": 2}, "why": "the removed auditor leg ['qwen-abl'] was reached 2 time(s)"}
        probs = co.carry_over_problems(p1, plan=this, read_text=files.get, git=git, accepted_get_for_dict=_agd(dicts), petri_proof=proof_stub)
        assert any("the recorded Petri sidecar proof FAILED" in p for p in probs) and any("needs the covering sidecar proof; the record carries a FAILING one" in p for p in probs), probs
        # probe 2: proof AND evidence_inventory_local removed -> the requirement stands (derived from the row and the recomputed chain difference): refused
        p2 = copy.deepcopy(rec); e2 = p2["carried"]["petri__s1__base__shard00"]
        e2["clauses"]["producer_set"].pop("petri_sidecar_proof"); e2.pop("evidence_inventory_local")
        probs = co.carry_over_problems(p2, plan=this, read_text=files.get, git=git, accepted_get_for_dict=_agd(dicts), petri_proof=proof_stub)
        assert any("needs the covering sidecar proof; the record carries none -- refusing" in p for p in probs), probs
        # the proof re-derived at certify must HOLD: a proof that re-derives failing refuses, and its inputs must be the recorded inventory
        def failing(**kw):
            kw["read_text"]("/synthetic/petri/cells/c0/results.json")
            return {"ok": False, "cells": 8, "removed_leg": {"uses": 1}, "why": "reached once"}
        probs = co.carry_over_problems(rec, plan=this, read_text=files.get, git=git, accepted_get_for_dict=_agd(dicts), petri_proof=failing)
        assert any("re-derived at certify does NOT hold" in p and "removed-leg uses 1" in p for p in probs), probs
        def other_inputs(**kw):
            kw["read_text"]("/synthetic/petri/cells/c1/results.json")
            return {"ok": True, "cells": 8, "removed_leg": {"uses": 0}, "why": None}
        probs = co.carry_over_problems(rec, plan=this, read_text=files.get, git=git, accepted_get_for_dict=_agd(dicts), petri_proof=other_inputs)
        assert any("re-derived inputs" in p and "are not the recorded evidence_inventory_local" in p for p in probs), probs
        # without an injected proof the default re-derives from the evidence's completion bundle -- the synthetic evidence has none: refused by name
        probs = co.carry_over_problems(rec, plan=this, read_text=files.get, git=git, accepted_get_for_dict=_agd(dicts))
        assert any("cannot be re-derived from the attempt's evidence" in p and "completion bundle" in p for p in probs), probs

    def test_the_requirement_covers_a_petri_mode_misalign_row_with_a_petri_runner_difference(self):
        # the plan generator's leg table is the authority on MODE (a plan row carries none: the archived misalign row reads `mode: None`)
        assert co.PETRI_MODE_LEGS == tuple(sorted(k for k, v in lp.leg_table(lp.S4_DEFAULTS).items() if v.get("mode") == "petri")) == ("misalign", "petri") or \
               set(co.PETRI_MODE_LEGS) == {k for k, v in lp.leg_table(lp.S4_DEFAULTS).items() if v.get("mode") == "petri"}
        row = {"leg": "misalign"}                                                                 # as archived: no mode field
        assert co.is_petri_mode_row(row) and co.is_petri_mode_row({"leg": "misalign", "mode": "petri"}) and not co.is_petri_mode_row({"leg": "pasttense", "mode": "attacks"})
        assert co.petri_proof_required(row, [{"path": "src/sisteer/capeval/tiers/petri_runner.py", "kind": co.ADMITTED_KIND_OFF_PATH}])
        assert co.petri_proof_required(row, [{"path": F.MANIFEST, "kind": co.ADMITTED_KIND_CHAIN}])
        assert not co.petri_proof_required(row, [{"path": "src/sisteer/registry/hooks.py", "kind": co.ADMITTED_KIND_REGISTRY}])
        assert co.petri_proof_required({"leg": "petri"}, [{"path": "src/sisteer/registry/hooks.py", "kind": co.ADMITTED_KIND_REGISTRY}])      # the petri LEG: any admission
        assert not co.petri_proof_required({"leg": "controls", "mode": "attacks"}, [{"path": "src/sisteer/capeval/tiers/petri_runner.py", "kind": co.ADMITTED_KIND_OFF_PATH}])
