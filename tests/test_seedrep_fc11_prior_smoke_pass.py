"""qwen-seedrep-native-battery -- fixer fc11: the S1 / S1b / S2 / S3 PASS records DERIVED FROM PRIOR TREES.

The launch blocker: `certificate_problems` requires `smokes[sid]` for S1, S1b, S2, S3 (S4 through the waiver), and the only writer
(`derive_smoke_pass`) resolves each row's accepted attempt through THIS plan's smoke Dict (`seedrep-<W>-<wave>-accepted`), empty on a new
tree -- the smokes completed on earlier trees (S1 / S1b / S2 / S3 bon+misalign, S2 controls, S3 petri). Mirroring the fc9 S5
prior-W admission and the carry-over's fj7 re-audit, `derive_smoke_pass_from_priors` resolves the rows through the PRIOR archives' smoke Dicts
(`RS/prior/<W>`, the carry-over's `load_prior_source` layout), binds each prior row to this plan's row by identity (`smoke_row_identity`: leg,
view, stage, shard, kind, seed bases, L/S/thinking, served objects' shas, the children), re-audits the request evidence with THIS tree
(`reaudit`, the carry-over's callable), reads every other producer file at the historical root held to the PRIOR tree, and stamps
`measured_under_W` / `measured_under_wave` / `attempt_root` per row + `reaudited_by: <this W>`. `smoke_pass_problems` admits such a record
ONLY with the stamps, a clean re-audit and every row bound; refuses by name otherwise. The world is the certificate fixture's fake volume
(`TestSolR3_2_launch_certificate._components`) under a PRIOR tree, and a plan shaped like this W over the same rows.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import seedrep_s5_record as S5R  # noqa: E402
from test_seedrep_review_fixes import _plan, _stub_modal, draft, inputs  # noqa: E402,F401  (the module fixtures)
from test_seedrep_review_fixes import TestSolR3_2_launch_certificate as _CertComponents  # noqa: E402  (aliased: not collected here)
from test_seedrep_review_fixes import TestDoseStandups as _Dose  # noqa: E402
from _env import host_site_path  # noqa: E402  (tests/ helper: a host-only data path through the site config)

W25 = "5" * 40                        # this tree; the fixture's prior is `2` * 40 (test_seedrep_review_fixes.W)
PRIOR_IDS = ("S1", "S1b", "S2", "S3")


def _stub_reaudit(texts, this_W, *, sampler_verdict="AGREES", arms=None, stores_present=None):
    """A re-audit double in `seedrep_served.reaudit_carried_root`'s shape (the fj5 tests' `_stub_reaudit`, on the certificate fixture's world): for every
    (child, table leg) of the row the historical verdict re-issued under THIS tree's W (`store_W` the prior's), filed under this wave's carried
    evidence root and bound by the plan's rel with `bound_rel`, sha, writer, tree_sha = this W, rc 0. fc11c: the sampler verdict RE-DERIVED by this
    tree's recorder (`rederive_sampler_verdict`'s shape) -- the historical record re-issued with ``sampler_verdict`` into the evidence root, bound
    under `resolved_sampler.json` with tree_sha = this W, and the `sampler` record (historical vs re-derived)."""
    calls = []

    def reaudit(*, prev_row, this_row, prior, aid, root, this_plan):
        calls.append({"sid": this_row["standup_id"], "aid": aid, "root": root, "prior_W": prior["W"]})
        ev = json.loads(texts[f"{root}/modal_evidence.json"])
        hist = lp.producer_map_for("request_evidence_audit", ev)
        ev_root = f"/vol/seedrep/{this_plan['W7']}/{this_plan['wave_uuid']}/carried/{this_row['standup_id']}__{aid}/evidence"
        bindings = {}
        # fc11d: the sampler verdict is re-derived FIRST and handed to the audit (`--resolved-sampler`): every verdict carries it with its provenance
        hist_rs = json.loads(texts[f"{root}/{lp.RESOLVED_SAMPLER_REL}"])
        fresh_rs = json.dumps({**hist_rs, "verdict": sampler_verdict, "why": f"re-derived under {this_W[:7]}", "reference_source": "--expect-tasks (this tree's expected-samplers)"})
        texts[f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}"] = fresh_rs
        rs_source = {"path": f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": lp._sha_text(fresh_rs), "rederived_by": this_W}
        present = {}
        for path, child, table_leg in lp.smoke_audit_sources(prev_row, root, hist):
            cell = str(child.get("cell") or child["arm"])
            if arms is not None and cell not in arms:
                continue                                                     # fc11f: the (fake) pulled root holds only these arms' stores
            doc = json.loads(texts[path])
            rel_store = f"exp/sampled/{table_leg}__s{child['s']}/{doc.get('arm')}"
            present[(table_leg, str(child["s"]), str(doc.get("arm")))] = {"leg": table_leg, "s": child["s"], "arm": str(doc.get("arm")), "store_dir": rel_store,
                                                                          "identity": f"{table_leg}|s{child['s']}|{doc.get('arm')}|{rel_store}"}
            doc.update(W=this_W, store_W=prior["W"], audited_at=this_W, resolved_sampler_verdict=sampler_verdict, resolved_sampler_source=rs_source,
                       historical_resolved_sampler_verdict=hist_rs.get("verdict"), historical_resolved_sampler_path=f"{root}/{lp.RESOLVED_SAMPLER_REL}")
            text = json.dumps(doc)
            bound_rel = path[len(root) + 1:]
            texts[f"{ev_root}/{bound_rel}"] = text
            plan_rel = lp.audit_verdict_rel(table_leg, child["arm"], child["s"], stage=this_row["stage"], shard=this_row["shard"])
            bindings[plan_rel] = {"sha256": lp._sha_text(text), "writer": lp.PRODUCER_WRITERS["request_evidence_audit"], "tree_sha": this_W, "rc": 0, "bound_rel": bound_rel,
                                  "plan_arm": child["arm"], "plan_cell": str(child.get("cell") or child["arm"]), "store_arm": doc.get("arm")}
        bindings[lp.RESOLVED_SAMPLER_REL] = {"sha256": lp._sha_text(fresh_rs), "writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": this_W, "rc": 0}
        bindings[lp.EXPECTED_SAMPLERS_REL] = {"absent": True, "why": "stub", "writer": "scripts/modal/seedrep_ladder_helpers.py", "tree_sha": this_W, "rc": 0}
        sampler = {"historical_verdict": hist_rs.get("verdict"), "rederived_verdict": sampler_verdict, "rederived_by": this_W, "recorder_rc": 0, "expected_samplers_rc": 0,
                   "vllm_log": f"{root}/exp/serve/vllm_server.log", "expect_tasks": True, "why": None, "basis": lp.SMOKE_PASS_SAMPLER_BASIS,
                   "probe": {"declared": {}, "probe_caps_argv": [], "probe_max_tokens": 48, "client_caps": [4096], "n_probe_by_cap": {"48": 8}, "n_unclassified": 0}}
        return {"reaudited_by": this_W, "evidence_root": ev_root, "store_root": root, "producer_evidence": bindings, "rc": 0,
                "argv": ["python", lp.PRODUCER_WRITERS["request_evidence_audit"], "--audited-at", this_W], "tool": lp.PRODUCER_WRITERS["request_evidence_audit"], "n_files": len(bindings),
                "sampler": sampler, "stores_present": (stores_present if stores_present is not None else [present[k_] for k_ in sorted(present)]),
                "stores_problems": {}, "table_legs": sorted({k_[0] for k_ in present}), "arms_planned": sorted({str(k.get("cell") or k["arm"]) for k in this_row["children"]})}
    reaudit.calls = calls
    return reaudit


@pytest.fixture(scope="module")
def world(inputs, draft, tmp_path_factory):
    """The certificate fixture's fake smoke world under the PRIOR tree (`2`*40), archived the way RS/prior/<W7> is (launch_plan_draft.json +
    smokes/<Sid>/wave.json), and a current-W draft + final over the same rows with an RS carrying the same minted smoke waves."""
    tmp = tmp_path_factory.mktemp("fc11")
    c = _CertComponents()._components(inputs, draft, tmp)
    prior_plan = c["draft"]
    archive = tmp / "prior" / prior_plan["W7"]
    archive.mkdir(parents=True)
    (archive / "launch_plan_draft.json").write_text(json.dumps(prior_plan, default=str))
    for sid in lp.SMOKE_PASS_IDS + ("S5",):
        wp = lp.smoke_wave_path(archive, sid)
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(lp.smoke_wave_path(c["rs"], sid).read_text())
    prior = co.load_prior_source(archive)
    assert prior["W"] == prior_plan["W"] and prior["smoke_waves"]["S1"] == c["waves"]["S1"]
    ro, pins = _Dose._peer(inputs)
    draft25 = _plan(inputs, W=W25, peer_roster=ro, peer_pins=pins)
    final25 = _plan(inputs, W=W25, materialisation="final", s4=lp.S4_DEFAULTS, peer_roster=ro, peer_pins=pins)
    assert lp.compare_draft_final(draft25, final25) == [] and draft25["W"] == W25 and draft25["wave_uuid"] == prior_plan["wave_uuid"]
    rs25 = tmp / "RS25"
    shutil.copytree(c["rs"] / lp.SMOKES_DIR, rs25 / lp.SMOKES_DIR)
    (rs25 / "launch_plan.json").write_text(json.dumps(final25, default=str))
    (rs25 / "launch_plan_draft.json").write_text(json.dumps(draft25, default=str))
    texts = dict(c["texts"])
    for sid in PRIOR_IDS:
        _smoke_spell_cells(texts, sid, c["smokes"][sid]["rows"])
    by_dict = {lp.smoke_identity(prior_plan, sid, c["waves"][sid])["dicts"]["accepted"]: c["accepted"][sid] for sid in lp.SMOKE_PASS_IDS}
    return dict(c=c, prior=prior, archive=archive, draft25=draft25, final25=final25, rs25=rs25, texts=texts, by_dict=by_dict,
                read_text=lp.volume_only(lambda p: texts.get(str(p))), for_dict=lambda name: by_dict.get(name, {}).get)


def _smoke_spell_cells(texts, sid, rows):
    """fc11j: the real container serves the SMOKE row (`smoke_row`'s spelling) and its `exp/serve/cells.json` says so; the certificate fixture wrote the plan id."""
    for rid, r in rows.items():
        root = r["attempt_root"]
        cp = f"{root}/{lp.CELLS_JSON_REL}"
        if cp not in texts:
            continue
        doc = json.loads(texts[cp])
        doc["serving"]["standup_id"] = lp.smoke_standup_id(sid, rid)
        texts[cp] = json.dumps(doc)
        ev = json.loads(texts[f"{root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][lp.CELLS_JSON_REL]["sha256"] = lp._sha_text(texts[cp])
        texts[f"{root}/modal_evidence.json"] = json.dumps(ev)


def _derive(w, sid, *, plan=None, priors=None, reaudit=None, rows=None, read_text=None):
    plan = plan or w["draft25"]
    reaudit = reaudit or _stub_reaudit(w["texts"], plan["W"])
    return lp.derive_smoke_pass_from_priors(rs=w["rs25"], plan=plan, smoke_id=sid, standup_ids=rows or w["c"]["rows_by_smoke"][sid], priors=priors or [w["prior"]],
                                            accepted_get_for_dict=w["for_dict"], read_text=read_text or w["read_text"], reaudit=reaudit, derived_utc="2026-09-20T00:00:00Z")


def _problems(w, rec, sid, *, plan=None, read_text="default", for_dict="default"):
    plan = plan or w["final25"]
    return lp.smoke_pass_problems(rec, smoke_id=sid, plan=plan, execution_wave_uuid=lp.smoke_execution_wave(w["rs25"], sid), draft_launch_plan_sha=w["draft25"]["launch_plan_sha"],
                                  read_text=(w["read_text"] if read_text == "default" else read_text), accepted_get_for_dict=(w["for_dict"] if for_dict == "default" else for_dict))


class TestDerivationFromPriors:
    def test_s1_s1b_s2_s3_derive_from_the_prior_archive_with_the_stamps_and_this_trees_reaudit(self, world):
        w = world
        for sid in PRIOR_IDS:
            reaudit = _stub_reaudit(w["texts"], W25)
            rec = _derive(w, sid, reaudit=reaudit)
            assert rec["W"] == W25 and rec["reaudited_by"] == W25 and rec["schema"] == lp.SMOKE_PASS_SCHEMA and rec["verdict"] == "PASS" and rec["rc"] == 0
            assert tuple(rec) == lp.SMOKE_PASS_KEYS + lp.SMOKE_PASS_PRIOR_KEYS
            assert rec["execution_wave_uuid"] == w["c"]["waves"][sid] and rec["production_wave_uuid"] == w["final25"]["wave_uuid"]
            assert set(rec["measured_under_W"].values()) == {w["prior"]["W"]} and set(rec["measured_under_wave"].values()) == {w["c"]["waves"][sid]}
            assert rec["priors"] == [{"path": str(w["archive"]), "plan_path": str(w["archive"] / "launch_plan_draft.json"), "W": w["prior"]["W"], "W7": w["prior"]["W7"],
                                      "launch_plan_sha": w["prior"]["launch_plan_sha"], "smoke_wave": w["c"]["waves"][sid]}]
            for rid, r in rec["rows"].items():
                assert r["measured_under_W"] == w["prior"]["W"] and r["attempt_root"].startswith(f"/vol/seedrep/{w['prior']['W7']}/exp/") and f"_smoke/{sid}/" in r["attempt_root"]
                assert r["accepted_dict"] == f"seedrep-{w['prior']['W7']}-{w['c']['waves'][sid].replace('-', '')[:8]}-accepted"
                assert r["identity"] == lp.smoke_row_identity(lp.plan_row_by_id(w["final25"], rid)) == lp.smoke_row_identity(lp.plan_row_by_id(w["prior"]["plan"], rid))
                assert r["reaudit"]["reaudited_by"] == W25 and r["reaudit"]["store_root"] == r["attempt_root"] and r["reaudit"]["rc"] == 0
                assert r["accepted_attempt"] == w["c"]["smokes"][sid]["rows"][rid]["accepted_attempt"]       # the prior tree's own derivation named the same attempt
            assert reaudit.calls and all(x["prior_W"] == w["prior"]["W"] for x in reaudit.calls)
            assert set(rec["checks"]) == set(lp.SMOKE_CHECKS[sid])
            aud = rec["checks"]["request_evidence_audit"]
            assert aud["reaudit"] == {"reaudited_by": W25, "held_to_plan": "this", "basis": lp.SMOKE_PASS_PRIOR_BASIS}
            assert all(s["path"].startswith(f"/vol/seedrep/{W25[:7]}/{w['final25']['wave_uuid']}/carried/") and s["producer"]["tree_sha"] == W25 for s in aud["sources"])
            for name, c_ in rec["checks"].items():
                if name not in lp.SMOKE_PASS_REDERIVED_CHECKS:
                    assert all(s["producer"]["tree_sha"] == w["prior"]["W"] for s in c_["sources"]), name       # the historical producer, held to the PRIOR tree
            # fc11c: the sampler verdict is THIS tree's re-derivation at the evidence root, never the historical record
            smp = rec["checks"]["resolved_sampler"]
            assert smp["reaudit"] == {"reaudited_by": W25, "held_to_plan": "this", "basis": lp.SMOKE_PASS_PRIOR_BASIS}
            assert all(s["path"].endswith(f"/evidence/{lp.RESOLVED_SAMPLER_REL}") and "/carried/" in s["path"] and s["producer"]["tree_sha"] == W25 for s in smp["sources"])
            for rid, r in rec["rows"].items():
                assert r["sampler"] == {"historical_verdict": "AGREES", "historical_path": f"{r['attempt_root']}/{lp.RESOLVED_SAMPLER_REL}", "rederived_verdict": "AGREES",
                                        "rederived_by": W25, "rederived_path": f"{r['reaudit']['evidence_root']}/{lp.RESOLVED_SAMPLER_REL}", "why": lp.SMOKE_PASS_SAMPLER_BASIS,
                                        "probe": r["reaudit"]["sampler"]["probe"]}
                assert "log-byte verification" in r["sampler"]["why"] and "not hashed" in r["sampler"]["why"]        # no claim the engine log's bytes are verified
                assert r["smoke_scope"] == lp.SMOKE_SCOPE_FULL and r["arms_planned"] == sorted({str(k.get("cell") or k["arm"]) for k in lp.plan_row_by_id(w["final25"], rid)["children"]})
                assert r["arms_audited"] == sorted(e["identity"] for e in r["reaudit"]["stores_present"]) and r["reaudit"]["stores_present"]       # fc11i: an identity list
                assert all("|s" in x and x.count("|") == 3 for x in r["arms_audited"])
                assert r["served_standup_id"] == {"plan": rid, "smoke": lp.smoke_standup_id(sid, rid)} and r["qualifications"] == []                # fc11j
            if sid == "S1b":
                # fc11j: S1b's fitted arms exercise the bank check (S1's `none`-arm smoke served no fitted object): cells.json names the served SMOKE row
                assert rec["checks"]["adapter_hashes_banked"]["sources"] and all(s_["n_planned"] >= 1 for s_ in rec["checks"]["adapter_hashes_banked"]["sources"])
            assert all(s["resolved_sampler_source"]["sha256"] == rec["rows"][rid]["reaudit"]["producer_evidence"][lp.RESOLVED_SAMPLER_REL]["sha256"]
                       for rid in rec["rows"] for s in aud["sources"] if s["path"].startswith(rec["rows"][rid]["reaudit"]["evidence_root"] + "/"))
            assert _problems(w, rec, sid) == []
            assert _problems(w, rec, sid, read_text=None, for_dict=None) == []                             # validates on its own content too
            assert lp.is_prior_smoke_pass(rec) and not lp.is_prior_smoke_pass(w["c"]["smokes"][sid])

    def test_a_smoke_may_draw_rows_from_several_priors_in_order(self, world, tmp_path):
        """S2 draws pasttense/arditi from one prior tree and controls/arditi from another; here the SAME archive twice under two paths stands for two
        priors -- the first with no accepted attempt for the row is skipped, the second wins, and both are named."""
        w = world
        empty = tmp_path / "prior-empty"
        shutil.copytree(w["archive"], empty)
        prior_empty = co.load_prior_source(empty)
        prior_empty = {**prior_empty, "W": "7" * 40, "W7": "7777777"}                          # another tree whose Dict is empty
        rec = _derive(w, "S1", priors=[prior_empty, w["prior"]])
        assert set(rec["measured_under_W"].values()) == {w["prior"]["W"]} and [p["W"] for p in rec["priors"]] == [w["prior"]["W"]]
        assert _problems(w, rec, "S1") == []

    def test_the_prior_that_is_this_tree_is_refused(self, world):
        w = world
        with pytest.raises(ValueError, match="IS this tree"):
            _derive(w, "S1", plan=w["c"]["draft"], priors=[w["prior"]])
        with pytest.raises(ValueError, match="no prior archive holds an accepted attempt"):
            _derive(w, "S1", priors=[{**w["prior"], "W": "7" * 40, "W7": "7777777"}])

    def test_identity_mismatch_refuses_by_field_at_derivation(self, world):
        w = world
        plan = copy.deepcopy(w["draft25"])
        rid = w["c"]["rows_by_smoke"]["S1"][0]
        row = lp.plan_row_by_id(plan, rid)
        row["seed_bases"] = [3]
        with pytest.raises(ValueError, match="identity field 'seed_bases' differs"):
            _derive(w, "S1", plan=plan)
        plan = copy.deepcopy(w["draft25"])
        row = lp.plan_row_by_id(plan, rid)
        cell = next(c for c in row["objects"])
        row["objects"][cell]["object_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="identity field 'objects' differs"):
            _derive(w, "S1", plan=plan)

    def test_s4_never_derives_from_a_prior(self, world):
        with pytest.raises(ValueError, match="S4 is the D133 waiver"):
            _derive(world, "S4")

    def test_a_reaudit_that_is_not_this_trees_or_fails_refuses(self, world):
        w = world

        def stale(**kw):
            ra = _stub_reaudit(w["texts"], W25)(**kw)
            return {**ra, "reaudited_by": w["prior"]["W"]}
        with pytest.raises(ValueError, match="re-audit record is not this tree's clean audit"):
            _derive(w, "S1", reaudit=stale)

        def broken(**kw):
            raise RuntimeError("audit tool missing")
        with pytest.raises(ValueError, match="re-audit under this tree failed: RuntimeError"):
            _derive(w, "S1", reaudit=broken)


class TestValidation:
    def test_a_prior_record_without_the_reaudit_or_stamps_refuses_by_name(self, world):
        w = world
        rec = _derive(w, "S2")
        rid = rec["standup_ids"][0]
        bad = copy.deepcopy(rec)
        del bad["rows"][rid]["reaudit"]
        assert any("lacks the prior stamps ['reaudit']" in p for p in _problems(w, bad, "S2"))
        bad = copy.deepcopy(rec)
        bad["reaudited_by"] = w["prior"]["W"]
        assert any("reaudited_by" in p and "not this tree" in p for p in _problems(w, bad, "S2"))
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["reaudit"]["rc"] = 1
        assert any("exited rc 1" in p for p in _problems(w, bad, "S2"))
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["reaudit"]["store_root"] = bad["rows"][rid]["attempt_root"] + "x"
        assert any("store_root" in p and "historical attempt root" in p for p in _problems(w, bad, "S2"))
        bad = copy.deepcopy(rec)
        for k in lp.SMOKE_PASS_PRIOR_KEYS:
            bad.pop(k)
        # without the stamps it is read as THIS tree's own derivation, and fails there: the roots are the prior tree's, its Dict never named them
        assert any("is not the root of accepted attempt" in p or "accepted Dict names attempt None NOW" in p for p in _problems(w, bad, "S2"))

    def test_identity_mismatch_refuses_by_field_at_validation(self, world):
        w = world
        rec = _derive(w, "S3")
        rid = rec["standup_ids"][0]
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["identity"]["seed_bases"] = [2]
        bad["rows"][rid]["identity_sha256"] = lp.smoke_row_identity_sha(bad["rows"][rid]["identity"])
        probs = _problems(w, bad, "S3")
        assert any("identity field 'seed_bases' differs" in p for p in probs)
        bad = copy.deepcopy(rec)
        cell = next(c for c in bad["rows"][rid]["identity"]["objects"])
        bad["rows"][rid]["identity"]["objects"][cell] = "e" * 64
        probs = _problems(w, bad, "S3")
        assert any("identity field 'objects' differs" in p for p in probs) and any("identity_sha256 is not the digest" in p for p in probs)
        # this plan's row edited after the record -- the recorded (prior) identity no longer binds
        plan = copy.deepcopy(w["final25"])
        lp.plan_row_by_id(plan, rid)["geometry"]["S"] = 7
        assert any("identity field 'geometry' differs" in p for p in _problems(w, rec, "S3", plan=plan))

    def test_a_prior_stamp_that_is_this_plans_w_and_a_missing_row_refuse(self, world):
        w = world
        rec = _derive(w, "S1")
        rid = rec["standup_ids"][0]
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["measured_under_W"] = W25
        bad["measured_under_W"][rid] = W25
        assert any("measured_under_W IS this plan's W" in p for p in _problems(w, bad, "S1"))
        bad = copy.deepcopy(rec)
        del bad["rows"][rid]
        assert any("standup_ids / accepted disagree with rows" in p for p in _problems(w, bad, "S1"))
        bad = copy.deepcopy(rec)
        bad["W"] = w["prior"]["W"]
        assert any("is not this plan's" in p and "measured_under_W" in p for p in _problems(w, bad, "S1"))

    def test_the_prior_dict_and_the_volume_are_re_read(self, world):
        w = world
        rec = _derive(w, "S1b")
        rid = rec["standup_ids"][0]
        superseded = {n: {k: (v + "-new") for k, v in d.items()} for n, d in w["by_dict"].items()}
        probs = _problems(w, rec, "S1b", for_dict=lambda name: superseded.get(name, {}).get)
        assert any("names attempt" in p and "NOW" in p for p in probs)
        gone = dict(w["texts"])
        gone.pop(f"{rec['rows'][rid]['attempt_root']}/RESULT")
        probs = _problems(w, rec, "S1b", read_text=lp.volume_only(lambda p: gone.get(str(p))))
        assert any("RESULT is no longer readable" in p for p in probs)
        edited = dict(w["texts"])
        path = rec["checks"]["request_evidence_audit"]["sources"][0]["path"]
        edited[path] = json.dumps({**json.loads(edited[path]), "verdict": "DISAGREES"})
        probs = _problems(w, rec, "S1b", read_text=lp.volume_only(lambda p: edited.get(str(p))))
        assert any(path in p and "changed since the record was derived" in p for p in probs)

    def test_this_trees_own_records_still_take_the_live_path(self, world):
        w, c = world, world["c"]
        for sid in PRIOR_IDS:
            rec = c["smokes"][sid]
            assert not lp.is_prior_smoke_pass(rec)
            assert lp.smoke_pass_problems(rec, smoke_id=sid, plan=c["draft"], execution_wave_uuid=c["waves"][sid], read_text=c["read_text"], accepted_get=c["accepted"][sid].get) == []
            # the same record under this plan is refused as before (its W is the prior's) -- a prior tree's record without the stamps is never admitted
            assert any(f"W {c['draft']['W']!r} is not this plan's" in p for p in _problems(w, rec, sid))


def smoke_probs(probs):
    """The S1-S4 lines of a problem list (the other components -- S0, S5, the W gate, the decode rows -- are absent from the hand-assembled certificate)."""
    return [p for p in probs if p.split(":")[0] in lp.SMOKE_PASS_IDS or any(p.startswith(f"no {s_} PASS record") for s_ in lp.SMOKE_PASS_IDS)]


def _prior_cert(w, rs):
    """A certificate over this plan whose S1-S3 are the prior-tree records and S4 the waiver (the other components None: this exercises the
    smoke loop), written under ``rs`` with its plan files; returns (rs, cert, the volume reader with the waiver roots)."""
    from test_seedrep_fc3_s4_waiver import volume_with_roots, waiver_doc
    shutil.copytree(w["rs25"], rs)
    (rs / lp.S4_WAIVER_FILE).write_text(json.dumps(waiver_doc(W25)))
    rt = volume_with_roots(w["read_text"])
    smokes = {sid: _derive(w, sid) for sid in PRIOR_IDS}
    smokes = S5R.admit_s4_waiver(rs=rs, plan=w["final25"], smokes_doc=smokes, read_text=rt)
    assert lp.is_smoke_waiver_entry(smokes["S4"])
    reads = {}
    for sid in PRIOR_IDS:
        entries = {r["accepted_key"]: r["accepted_attempt"] for r in smokes[sid]["rows"].values()}
        reads[sid] = {"dict": lp.prior_smoke_pass_dict_name(smokes[sid]), "read_utc": "2026-09-20T00:00:00Z", "entries": entries, "digest": lp.accepted_reads_digest(entries)}
    waves = {sid: lp.smoke_execution_wave(rs, sid) for sid in lp.SMOKE_PASS_IDS + ("S5",)}
    cert = {"schema": lp.CERTIFICATE_SCHEMA, "W": W25, "W7": W25[:7], "production_wave_uuid": w["final25"]["wave_uuid"], "launch_plan_sha": w["final25"]["launch_plan_sha"],
            "draft_launch_plan_sha": w["draft25"]["launch_plan_sha"], "draft_final_problems": [], "plan_sha256": lp.sha256_file(rs / "launch_plan.json"), "smoke_waves": waves,
            "s0": None, "smokes": smokes, "accepted_dicts": reads, "s5": None, "utilisation": None, "utilisation_waiver": None, "w_gate": None, "decode_rows": None, "carry_over": None,
            "launch_set": None, "s5_coverage": None, "issued_utc": "2026-09-20T00:00:00Z", "identity_rule": lp.WAVE_IDENTITY_RULE}
    cert["certificate_sha256"] = lp.certificate_sha256(cert)
    (rs / lp.CERTIFICATE_FILE).write_text(json.dumps(cert))
    return rs, cert, rt


class TestCliAndCertify:
    def test_pass_prior_writes_the_record_once_and_rederive_smokes_reproduces_it(self, world, tmp_path):
        w = world
        rs = tmp_path / "RS"
        shutil.copytree(w["rs25"], rs)
        reaudit = _stub_reaudit(w["texts"], W25)
        for sid in PRIOR_IDS:
            argv = ["pass", "--rs", str(rs), "--plan", str(rs / "launch_plan_draft.json"), "--smoke-id", sid, "--rows", ",".join(w["c"]["rows_by_smoke"][sid]),
                    "--prior", str(w["archive"])]
            assert S5R.main(argv, read_text=w["read_text"], accepted_get_for_dict=w["for_dict"], reaudit=reaudit) == 0
            assert S5R.main(argv, read_text=w["read_text"], accepted_get_for_dict=w["for_dict"], reaudit=reaudit) == 2          # write-once
        doc = json.loads((rs / "smokes.json").read_text())
        assert set(doc) == set(PRIOR_IDS) and all(lp.is_prior_smoke_pass(doc[sid]) and doc[sid]["reaudited_by"] == W25 for sid in PRIOR_IDS)
        fresh = S5R.rederive_smokes(rs=rs, plan=w["final25"], draft=w["draft25"], smokes_doc=doc, accepted_get_for=lambda sid: (lambda k: None), read_text=w["read_text"],
                                    accepted_get_for_dict=w["for_dict"])
        for sid in PRIOR_IDS:
            assert {k: v for k, v in fresh[sid].items() if k != "derived_utc"} == {k: v for k, v in doc[sid].items() if k != "derived_utc"}
        with pytest.raises(ValueError, match="certify needs the prior Dict reader"):
            S5R.rederive_smokes(rs=rs, plan=w["final25"], draft=w["draft25"], smokes_doc=doc, accepted_get_for=lambda sid: (lambda k: None), read_text=w["read_text"])
        edited = copy.deepcopy(doc)
        edited["S1"]["rows"][edited["S1"]["standup_ids"][0]]["reaudit"]["store_root"] += "x"
        with pytest.raises(ValueError, match="recorded re-audit is of attempt"):
            S5R.rederive_smokes(rs=rs, plan=w["final25"], draft=w["draft25"], smokes_doc=edited, accepted_get_for=lambda sid: (lambda k: None), read_text=w["read_text"],
                                accepted_get_for_dict=w["for_dict"])
        moved = copy.deepcopy(doc)
        moved["S1"]["priors"][0]["path"] = str(tmp_path / "not-here")
        with pytest.raises(ValueError, match="not at hand"):
            S5R.rederive_smokes(rs=rs, plan=w["final25"], draft=w["draft25"], smokes_doc=moved, accepted_get_for=lambda sid: (lambda k: None), read_text=w["read_text"],
                                accepted_get_for_dict=w["for_dict"])

    def test_pass_prior_refuses_a_prior_that_is_this_tree_and_s4(self, world, tmp_path, capsys):
        w = world
        rs = tmp_path / "RS"
        shutil.copytree(w["c"]["rs"], rs)                                              # the PRIOR tree's own RS: the plan there IS the prior
        argv = ["pass", "--rs", str(rs), "--plan", str(rs / "launch_plan_draft.json"), "--smoke-id", "S1", "--rows", ",".join(w["c"]["rows_by_smoke"]["S1"]), "--prior", str(w["archive"])]
        assert S5R.main(argv, read_text=w["read_text"], accepted_get_for_dict=w["for_dict"], reaudit=_stub_reaudit(w["texts"], W25)) == 2
        assert "IS this tree" in capsys.readouterr().err
        argv = ["pass", "--rs", str(w["rs25"]), "--plan", str(w["rs25"] / "launch_plan_draft.json"), "--smoke-id", "S4", "--rows", ",".join(w["c"]["rows_by_smoke"]["S4"]), "--prior", str(w["archive"])]
        assert S5R.main(argv, read_text=w["read_text"], accepted_get_for_dict=w["for_dict"], reaudit=_stub_reaudit(w["texts"], W25)) == 2
        assert "S4 is the D133 waiver" in capsys.readouterr().err

    def test_certificate_problems_admits_the_prior_records_beside_the_s4_waiver(self, world, tmp_path):
        """The certificate's smoke loop over a current-W plan: S1-S3 from the prior archive, S4 the waiver -- no smoke problem remains; the accepted-Dict
        read names the PRIOR Dicts (`prior_smoke_pass_dict_name`); a record stripped of its re-audit is refused inside the certificate too."""
        w = world
        rs, cert, rt = _prior_cert(w, tmp_path / "RS")
        waves = cert["smoke_waves"]
        probs = lp.certificate_problems(cert, plan=w["final25"], plan_path=rs / "launch_plan.json", read_text=rt, accepted_get_for=lambda sid: (lambda k: None),
                                        accepted_get_for_dict=w["for_dict"])
        assert smoke_probs(probs) == [], probs
        assert probs, "the other components (S0, S5, W gate, decode rows) are absent here and refused as before"
        # the issuer must have read the PRIOR Dicts, not this plan's (empty) smoke Dict
        bad = copy.deepcopy(cert)
        bad["accepted_dicts"]["S1"]["dict"] = lp.smoke_identity(w["final25"], "S1", waves["S1"])["dicts"]["accepted"]
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("S1: accepted_dicts names Dict" in p for p in lp.certificate_problems(bad, plan=w["final25"], read_text=rt, accepted_get_for_dict=w["for_dict"]))
        # a prior-tree record without its re-audit is refused inside the certificate
        bad = copy.deepcopy(cert)
        rid = bad["smokes"]["S3"]["standup_ids"][0]
        del bad["smokes"]["S3"]["rows"][rid]["reaudit"]
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any("S3: row" in p and "lacks the prior stamps ['reaudit']" in p for p in lp.certificate_problems(bad, plan=w["final25"], read_text=rt, accepted_get_for_dict=w["for_dict"]))
        # a missing smoke id is still refused by name
        bad = copy.deepcopy(cert)
        del bad["smokes"]["S2"]
        bad["certificate_sha256"] = lp.certificate_sha256(bad)
        assert any(p == "no S2 PASS record in the certificate" for p in lp.certificate_problems(bad, plan=w["final25"], read_text=rt, accepted_get_for_dict=w["for_dict"]))


class TestFc11bLaunchBoundary:
    """After certify, a prior smoke's accepted Dict entry removed or superseded (the old attempt's evidence still readable) must refuse
    at --go / fan-out / re-attempt -- so the live prior-Dict reader is threaded through `require_certificate` -> `go_preconditions` ->
    `certificate_problems`, and a prior record with NO reader at consumption is refused by name (fail closed)."""

    def test_go_preconditions_holds_every_prior_record_to_the_live_prior_dicts(self, world, tmp_path):
        w = world
        rs, cert, rt = _prior_cert(w, tmp_path / "RS")
        go = lambda reader: lp.go_preconditions(w["final25"], certificate=cert, plan_path=rs / "launch_plan.json", read_text=rt,       # noqa: E731
                                                accepted_get_for=lambda sid: (lambda k: None), accepted_get_for_dict=reader)
        assert smoke_probs(go(w["for_dict"])) == []                                                                 # unchanged -> admitted
        removed = go(lambda name: (lambda k: None))                                                                 # acceptance removed -> refused, every prior smoke
        for sid in PRIOR_IDS:
            assert any(p.startswith(f"{sid}: row ") and "names attempt None NOW" in p for p in removed), (sid, removed)
        superseded = go(lambda name: (lambda k, _n=name: (w["by_dict"].get(_n, {}).get(k) or "x") + "-superseded"))   # another attempt accepted -> refused
        for sid in PRIOR_IDS:
            assert any(p.startswith(f"{sid}: row ") and "-superseded' NOW" in p for p in superseded), (sid, superseded)
        absent = go(None)                                                                                            # reader absent -> refused by name, fail closed
        for sid in PRIOR_IDS:
            assert any(p.startswith(f"{sid}: a prior-tree PASS record") and "cannot be consumed without the live prior-Dict reader" in p and "fail closed" in p for p in absent), (sid, absent)
        assert not any(p.startswith("S4:") and "prior-Dict reader" in p for p in absent)                            # the waiver is not a prior record
        # a certificate WITHOUT prior records needs no such reader (this tree's own records take the live path as before)
        assert lp.prior_smoke_reader_problems({**cert, "smokes": {"S4": cert["smokes"]["S4"]}}, accepted_get_for_dict=None) == []
        assert lp.prior_smoke_reader_problems(None, accepted_get_for_dict=None) == []

    def test_require_certificate_reads_the_prior_dicts_live_by_name(self, world, tmp_path):
        """The app's boundary under the stub modal: the prior smoke Dicts populated as they were at certify -> no S1-S3 problem in the refusal (the other
        components are absent here); emptied -> every prior row refused as removed; another attempt accepted for the key -> refused as superseded.
        The default reader is `live_accepted_dict_getters()` -- the same shape `certify` records -- so no call site can leave it out."""
        import importlib.util
        from test_seedrep_review_fixes import FakeDict, FakeVOL
        w = world
        rs, cert, rt = _prior_cert(w, tmp_path / "RS")
        spec = importlib.util.spec_from_file_location("seedrep_served_fc11b", REPO / "scripts/modal/seedrep_served.py")
        srv = importlib.util.module_from_spec(spec)
        prior_mod = sys.modules.get("modal")
        sys.modules["modal"] = _stub_modal()
        try:
            spec.loader.exec_module(srv)
        finally:
            if prior_mod is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = prior_mod
        vol_before, dict_before = dict(FakeVOL.files), {k: dict(v) for k, v in FakeDict.store.items()}
        try:
            for path, text in w["texts"].items():
                FakeVOL.files[path[len("/vol/"):]] = text
            for name, entries in w["by_dict"].items():
                FakeDict.store[name] = dict(entries)

            def refusal():
                with pytest.raises(srv.LaunchRefused) as ei:
                    srv.require_certificate(w["final25"], plan_path=rs / "launch_plan.json", rs=rs, cert_p=rs / lp.CERTIFICATE_FILE)
                return [p.strip() for p in str(ei.value).split("): ", 1)[1].split("; ")]
            probs = refusal()
            assert [p for p in smoke_probs(probs) if not p.startswith("S4")] == [], probs                          # unchanged: S1-S3 admitted at the boundary
            for name in w["by_dict"]:
                FakeDict.store[name] = {}
            probs = refusal()
            for sid in PRIOR_IDS:
                assert any(p.startswith(f"{sid}: row ") and "names attempt None NOW" in p for p in probs), (sid, probs)      # removed
            for name, entries in w["by_dict"].items():
                FakeDict.store[name] = {k: v + "-superseded" for k, v in entries.items()}
            probs = refusal()
            for sid in PRIOR_IDS:
                assert any(p.startswith(f"{sid}: row ") and "-superseded' NOW" in p for p in probs), (sid, probs)           # superseded
        finally:
            FakeVOL.files.clear()
            FakeVOL.files.update(vol_before)
            FakeDict.store.clear()
            FakeDict.store.update(dict_before)
        # the wiring, pinned: the boundary forwards the live by-name reader; both seam-bearing consumers thread it; the re-attempt takes the default
        src = (REPO / "scripts/modal/seedrep_served.py").read_text()
        body = src.split("def require_certificate(", 1)[1].split("\ndef ", 1)[0]
        assert "accepted_get_for_dict=(accepted_get_for_dict or live_accepted_dict_getters())" in body
        assert src.count("require_certificate(plan_d, plan_path=plan_path, rs=rs, cert_p=cert_p, accepted_get_for=accepted_get_for, accepted_get_for_dict=accepted_get_for_dict)") == 2
        assert "cert = require_certificate(plan_d, plan_path=Path(plan), rs=rs_p, cert_p=cert_p)" in src.split("def main(", 1)[1]
        lp_src = (REPO / "scripts/modal/seedrep_launch_plan.py").read_text()
        go = lp_src.split("def go_preconditions(", 1)[1].split("\ndef ", 1)[0]
        assert "accepted_get_for_dict=accepted_get_for_dict" in go and "prior_smoke_reader_problems(certificate, accepted_get_for_dict=accepted_get_for_dict)" in go


class TestFc11cSamplerRederivation:
    """fc11c cause (1): the prior recorder stored UNCOMPARED on every two-sampler row although the engine log resolves every client request to the
    preset; a prior row's PASS is held to THIS tree's re-derivation (`rederive_sampler_verdict` in the re-audit), the stored verdict recorded beside it."""

    def _uncompared(self, w, sid):
        """The world with every row's HISTORICAL recorder record reading UNCOMPARED (the F-AR-era shape), re-bound by sha in the producer's evidence the way
        the prior container bound it."""
        texts = dict(w["texts"])
        for r in w["c"]["smokes"][sid]["rows"].values():
            path = f"{r['attempt_root']}/{lp.RESOLVED_SAMPLER_REL}"
            texts[path] = json.dumps({**json.loads(texts[path]), "verdict": "UNCOMPARED", "reference": None, "reference_source": None,
                                      "why": "client requests were read, but there is no single reference to compare them against (pass --expect, ...)"})
            ev_path = f"{r['attempt_root']}/modal_evidence.json"
            ev = json.loads(texts[ev_path])
            ev[lp.PRODUCER_EVIDENCE_KEY][lp.RESOLVED_SAMPLER_REL]["sha256"] = lp._sha_text(texts[path])
            # the prior in-container audit read that record and embedded UNCOMPARED in EVERY store verdict: the historical audit docs say so too
            prow = lp.plan_row_by_id(w["c"]["draft"], next(k for k, v in w["c"]["smokes"][sid]["rows"].items() if v is r))
            for apath, _child, _leg in lp.smoke_audit_sources(prow, r["attempt_root"], lp.producer_map_for("request_evidence_audit", ev)):
                texts[apath] = json.dumps({**json.loads(texts[apath]), "resolved_sampler_verdict": "UNCOMPARED"})
                rel = apath[len(r["attempt_root"]) + 1:]
                for b in ev[lp.PRODUCER_EVIDENCE_KEY].values():
                    if isinstance(b, dict) and b.get("bound_rel") == rel:
                        b["sha256"] = lp._sha_text(texts[apath])
            texts[ev_path] = json.dumps(ev)
        return texts

    def test_a_stored_uncompared_with_a_log_resolving_to_the_preset_passes_on_the_rederived_agrees(self, world):
        w = world
        for sid in ("S1", "S1b", "S2"):
            texts = self._uncompared(w, sid)
            rt = lp.volume_only(lambda p, _t=texts: _t.get(str(p)))
            # this tree's own derivation over such a root refuses (the stored verdict) -- what the real `pass --prior` run hit
            with pytest.raises(ValueError, match="verdict 'UNCOMPARED' is not AGREES"):
                lp.derive_smoke_pass(rs=w["c"]["rs"], plan=w["c"]["draft"], smoke_id=sid, standup_ids=w["c"]["rows_by_smoke"][sid], accepted_get=w["c"]["accepted"][sid].get, read_text=rt)
            rec = _derive(w, sid, reaudit=_stub_reaudit(texts, W25, sampler_verdict="AGREES"), read_text=rt)
            for r in rec["rows"].values():
                assert r["sampler"]["historical_verdict"] == "UNCOMPARED" and r["sampler"]["rederived_verdict"] == "AGREES" and r["sampler"]["rederived_by"] == W25
                assert r["reaudit"]["sampler"]["historical_verdict"] == "UNCOMPARED" and r["reaudit"]["producer_evidence"][lp.RESOLVED_SAMPLER_REL]["tree_sha"] == W25
            assert all(s["verdict"] == "AGREES" and s["path"].startswith(f"/vol/seedrep/{W25[:7]}/") for s in rec["checks"]["resolved_sampler"]["sources"])
            assert _problems(w, rec, sid, read_text=rt) == []
            # the historical root's verdict edited after the record -> refused (the stamp is re-read)
            edited = dict(texts)
            r0 = next(iter(rec["rows"].values()))
            edited[r0["sampler"]["historical_path"]] = json.dumps({**json.loads(texts[r0["sampler"]["historical_path"]]), "verdict": "DIVERGED"})
            probs = _problems(w, rec, sid, read_text=lp.volume_only(lambda p, _t=edited: _t.get(str(p))))
            assert any("reads verdict 'DIVERGED' today, the record says 'UNCOMPARED'" in p for p in probs), probs

    def test_a_log_resolving_to_another_sampler_refuses_by_name(self, world):
        w = world
        texts = self._uncompared(w, "S1")
        rt = lp.volume_only(lambda p, _t=texts: _t.get(str(p)))
        with pytest.raises(ValueError, match="re-derived the sampler verdict as 'DIVERGED' over the historical engine log, not AGREES") as ei:
            _derive(w, "S1", reaudit=_stub_reaudit(texts, W25, sampler_verdict="DIVERGED"), read_text=rt)
        assert "the stored 'UNCOMPARED' does not stand in" in str(ei.value)
        # fc11d: a re-audit whose AUDIT docs kept the historical embedded verdict (the fixture's old AGREES field, or the prior tree's UNCOMPARED) while
        # the fresh record says AGREES -- the audit never read the re-derived record -- refuses at the audit check, by name
        def stale_audit(**kw):
            ra = _stub_reaudit(texts, W25)(**kw)
            for rel, b in ra["producer_evidence"].items():
                if b.get("bound_rel"):
                    apath = f"{ra['evidence_root']}/{b['bound_rel']}"
                    doc = json.loads(texts[apath])
                    doc.pop("resolved_sampler_source"); doc["resolved_sampler_verdict"] = "UNCOMPARED"
                    texts[apath] = json.dumps(doc); b["sha256"] = lp._sha_text(texts[apath])
            return ra
        with pytest.raises(ValueError, match="reads the root's resolved sampler as 'UNCOMPARED', not AGREES"):
            _derive(w, "S1", reaudit=stale_audit, read_text=rt)
        # an audit whose recorded source sha is not the bound fresh record's bytes -> refused by name
        def wrong_source(**kw):
            ra = _stub_reaudit(texts, W25)(**kw)
            for rel, b in ra["producer_evidence"].items():
                if b.get("bound_rel"):
                    apath = f"{ra['evidence_root']}/{b['bound_rel']}"
                    doc = json.loads(texts[apath]); doc["resolved_sampler_source"]["sha256"] = "0" * 64
                    texts[apath] = json.dumps(doc); b["sha256"] = lp._sha_text(texts[apath])
            return ra
        with pytest.raises(ValueError, match="not the bound resolved_sampler.json .* the audit's sampler verdict is not the re-derived record's"):
            _derive(w, "S1", reaudit=wrong_source, read_text=rt)
        # a re-audit that never re-derived the sampler (an fc11-era record) refuses too
        def no_sampler(**kw):
            ra = _stub_reaudit(texts, W25)(**kw)
            ra.pop("sampler")
            ra["producer_evidence"].pop(lp.RESOLVED_SAMPLER_REL)
            return ra
        with pytest.raises(ValueError, match="sampler verdict was not re-derived by this tree"):
            _derive(w, "S1", reaudit=no_sampler, read_text=rt)

    def test_the_rederived_stamps_are_validated_not_trusted(self, world):
        w = world
        rec = _derive(w, "S1b")
        rid = rec["standup_ids"][0]
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["sampler"]["rederived_verdict"] = "UNCOMPARED"
        probs = _problems(w, bad, "S1b")
        assert any("must be THIS tree's re-derivation reading AGREES" in p for p in probs), probs
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["sampler"]["rederived_by"] = w["prior"]["W"]
        assert any("must be THIS tree's re-derivation reading AGREES" in p for p in _problems(w, bad, "S1b"))
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["reaudit"]["sampler"]["rederived_verdict"] = "DIVERGED"
        assert any("re-derived the sampler verdict as 'DIVERGED'" in p for p in _problems(w, bad, "S1b"))
        bad = copy.deepcopy(rec)
        del bad["rows"][rid]["sampler"]
        assert any("lacks the prior stamps ['sampler']" in p for p in _problems(w, bad, "S1b"))
        # the sampler source re-pointed at the HISTORICAL record (AGREES in this world) is refused: not under the row's re-audit evidence root
        bad = copy.deepcopy(rec)
        src = bad["checks"]["resolved_sampler"]["sources"][0]
        hist_path = bad["rows"][rid]["sampler"]["historical_path"]
        src["path"] = hist_path
        src["sha256"] = lp._sha_text(w["texts"][hist_path])
        probs = _problems(w, bad, "S1b")
        assert any("is not under any row's re-audit evidence root" in p for p in probs), probs


class TestFc11cMisalignLayout:
    """fc11c cause (2): the prior S3 misalign root keeps its stores as `exp/misalign/agentic__s1__{murder,leaking}/<arm>/` (the plan's MISALIGN_OUT_REL, the
    runner's `<out>/<cell>/` shape); `discover_stores` recognised `agentic/<arm>` only, so the re-audit read `no misalign_leaking store` (rc 2)."""

    @staticmethod
    def _cell(cell_dir: Path, *, arm: str, behaviours, n=2):
        cell_dir.mkdir(parents=True, exist_ok=True)
        items = [f"{b}_item{i}" for b in behaviours for i in range(n)]
        (cell_dir / "meta.json").write_text(json.dumps({"cell_id": arm, "seed_policy": "per_response", "seed_base": 1, "sampler": "vendor", "n_samples": 1,
                                                        "item_ids": items, "behaviors": list(behaviours)}))
        rows = [{"family": "agentic", "cell_id": arm, "virtual_model": f"petri__{arm}", "item_id": it, "sample_idx": 0, "behavior": it.split("_item")[0],
                 "response": f"r-{it}", "finish_reason": "stop", "request_id": f"rid-{it}", "completion_id": f"cid-{it}"} for it in items]
        (cell_dir / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    @staticmethod
    def _root(path: Path):
        import audit_common as ac
        path.mkdir(parents=True, exist_ok=True)
        return ac.AttemptRoot(path=path, attempt_id="ta-1", W="5" * 40, leg="misalign", view="s1", shard="shard00", stage="base", started_utc=None, attempt={})

    def test_the_w22_per_behaviour_layout_is_found_for_both_behaviours_and_recorded(self, tmp_path):
        import audit_common as ac
        root = self._root(tmp_path / "root")
        for beh in ("murder", "leaking"):
            for arm in ("none", "si_constv2_mix_full__psr_ce_all"):
                self._cell(root.path / "exp/misalign" / f"agentic__s1__{beh}" / arm, arm=arm, behaviours=[beh])
        (root.path / "exp/misalign/agentic__s1__leaking/request_evidence.jsonl").write_text("")            # the prior root keeps the child's evidence beside the cells
        for beh in ("murder", "leaking"):
            stores = ac.discover_stores(root, f"misalign_{beh}")
            assert sorted(s_.arm for s_ in stores) == ["none", "si_constv2_mix_full__psr_ce_all"] and all(s_.leg == f"misalign_{beh}" and s_.complete for s_ in stores)
            assert all(s_.store_dir.parent.name == f"agentic__s1__{beh}" for s_ in stores)
            assert all(any(n == f"store_layout={ac.MISALIGN_LAYOUT_PER_BEHAVIOUR} (agentic__s1__{beh}/{s_.arm})" for n in s_.notes) for s_ in stores)
            assert all({d.item_id.split("_item")[0] for d in s_.draws} == {beh} for s_ in stores)

    def test_the_flat_layout_is_still_found_and_both_together_refuse_as_ambiguous(self, tmp_path):
        import audit_common as ac
        root = self._root(tmp_path / "root")
        self._cell(root.path / "exp/misalign/agentic/none", arm="none", behaviours=["murder", "leaking"])
        for beh in ("murder", "leaking"):
            (store,) = ac.discover_stores(root, f"misalign_{beh}")
            assert store.arm == "none" and store.n_rows == 2 and any(n == f"store_layout={ac.MISALIGN_LAYOUT_FLAT} (agentic/none)" for n in store.notes)
        self._cell(root.path / "exp/misalign/agentic__s1__murder/none", arm="none", behaviours=["murder"])
        with pytest.raises(ac.AuditRefusal, match="misalign_murder stores under BOTH layouts .* ambiguous"):
            ac.discover_stores(root, "misalign_murder")
        (store,) = ac.discover_stores(root, "misalign_leaking")                                              # leaking has one layout: unaffected
        assert store.store_dir.parent.name == "agentic"
        # an out dir of ANOTHER behaviour never serves this leg, and a root with no misalign store still refuses by name
        empty = self._root(tmp_path / "empty")
        self._cell(empty.path / "exp/misalign/agentic__s1__leaking/none", arm="none", behaviours=["leaking"])
        with pytest.raises(ac.AuditRefusal, match="no misalign_murder store under"):
            ac.discover_stores(empty, "misalign_murder")

    def test_inventory_stores_lists_identities_and_names_an_unclassifiable_sibling(self, tmp_path):
        """fc11i: `inventory_stores` -- every store by identity (leg, seed, arm, path); a misalign sibling without its meta.json (a discovery `continue` before) is
        a NAMED problem under `unclassified`, never a silent skip; so is an unknown layout."""
        import audit_common as ac
        root = self._root(tmp_path / "root")
        for beh in ("murder", "leaking"):
            self._cell(root.path / "exp/misalign" / f"agentic__s1__{beh}" / "none", arm="none", behaviours=[beh])
        inv = ac.inventory_stores(root.path, ["misalign_murder", "misalign_leaking"], root=root)
        assert inv["problems"] == {} and inv["unclassified"] == []
        assert sorted(e["identity"] for e in inv["stores"]) == ["misalign_leaking|s1|none|exp/misalign/agentic__s1__leaking/none", "misalign_murder|s1|none|exp/misalign/agentic__s1__murder/none"]
        stray = root.path / "exp/misalign/agentic__s1__murder/stray"
        stray.mkdir()
        (stray / "responses.jsonl").write_text(json.dumps({"item_id": "x", "sample_idx": 0, "behavior": "murder"}) + "\n")     # no meta.json
        inv = ac.inventory_stores(root.path, ["misalign_murder", "misalign_leaking"], root=root)
        assert inv["unclassified"] == ["exp/misalign/agentic__s1__murder/stray/responses.jsonl"] and len(inv["stores"]) == 2
        odd = root.path / "exp/misalign/weird_layout/none"
        self._cell(odd, arm="none", behaviours=["murder"])
        inv = ac.inventory_stores(root.path, ["misalign_murder", "misalign_leaking"], root=root)
        assert "exp/misalign/weird_layout/none/responses.jsonl" in inv["unclassified"]
        # fc11k: a stray NESTED beneath a recognized cell dir is just as unclassified -- no reader consumed it (an ancestor test hid it)
        nested = root.path / "exp/misalign/agentic__s1__leaking/none/unknown-layout/stray"
        nested.mkdir(parents=True)
        (nested / "rows.jsonl").write_text(json.dumps({"task": "x"}) + "\n")
        (nested / "responses.jsonl").write_text(json.dumps({"item_id": "y", "sample_idx": 0, "behavior": "leaking"}) + "\n")
        inv = ac.inventory_stores(root.path, ["misalign_murder", "misalign_leaking"], root=root)
        assert {"exp/misalign/agentic__s1__leaking/none/unknown-layout/stray/rows.jsonl", "exp/misalign/agentic__s1__leaking/none/unknown-layout/stray/responses.jsonl"} <= set(inv["unclassified"])
        assert all(e["consumed_files"] == [f"{e['store_dir']}/responses.jsonl"] for e in inv["stores"])

    def test_a_static_roots_own_shard_files_are_consumed_not_flagged(self, tmp_path):
        """fc11k: the recognized stores' own candidate files -- a static store's per-shard responses.jsonl among them -- are consumed by their reader and never
        flagged; a stray rows.jsonl beneath one of those shard dirs is."""
        import audit_common as ac
        import seedrep_audit_fixtures as fx
        root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
        fx.populate_root(root, s=1, arms=("none",), n_items=2)
        inv = ac.inventory_stores(root.path, ["pasttense"])
        assert inv["problems"] == {} and inv["unclassified"] == [] and inv["stores"], inv
        assert all(e["consumed_files"] and all(f.endswith("/responses.jsonl") for f in e["consumed_files"]) for e in inv["stores"])
        first_store = root.path / inv["stores"][0]["store_dir"]
        stray = first_store / "unknown-layout" / "stray"
        stray.mkdir(parents=True)
        (stray / "rows.jsonl").write_text(json.dumps({"task": "x"}) + "\n")
        inv2 = ac.inventory_stores(root.path, ["pasttense"])
        assert inv2["unclassified"] == [str((stray / "rows.jsonl").relative_to(root.path).as_posix())] and len(inv2["stores"]) == len(inv["stores"])

    def test_the_audit_verdict_records_the_store_notes_with_the_layout(self):
        src = (REPO / "scripts/sdfbase/seedrep/request_evidence_audit.py").read_text()
        assert src.count('"store_notes": list(store.notes)') == 3, "every store verdict literal carries the store's notes (the matched misalign layout among them)"

    def test_the_real_pulled_w22_misalign_root_reads_when_present(self):
        """Read-only over the pulled prior S3 misalign root (RS/carry_over/reaudit/misalign__s1__base__shard00/<aid>/pulled/...), when it is on
        this machine: both behaviours discover their stores under the per-behaviour layout (the same reader the carry-over's clause (c) re-audit runs)."""
        import audit_common as ac
        rs = host_site_path("$STATE/romp/candidate-si-refit/run-state/full22/seedrep_battery")
        pulled = sorted((rs / "carry_over/reaudit/misalign__s1__base__shard00").glob("*/pulled")) if rs.is_dir() else []
        roots = [p for d in pulled for p in d.rglob("ATTEMPT.json")]
        if not roots:
            pytest.skip("no pulled W22 misalign root under RS/carry_over/reaudit on this machine")
        ar = ac.read_attempt_root(roots[0].parent)
        assert ar.leg == "misalign"
        for beh in ("murder", "leaking"):
            stores = ac.discover_stores(ar, f"misalign_{beh}")
            assert stores and all(s_.store_dir.parent.name == f"agentic__{ar.view}__{beh}" for s_ in stores), [str(s_.store_dir) for s_ in stores]
            assert all(any(n.startswith(f"store_layout={ac.MISALIGN_LAYOUT_PER_BEHAVIOUR}") for n in s_.notes) for s_ in stores)


class TestFc11dRealBoundary:
    """fc11d: through the REAL producer / auditor boundary -- `reaudit_carried_root` with this tree's recorder and auditor. Positive case: the
    REAL pulled prior S1 root (RS/carry_over/reaudit/pasttense__s1__base__shard00/<aid>/pulled, read-only, copied into the work dir) with the real prior plan
    row: the stored verdict is UNCOMPARED, the engine log resolves every client request to the preset -> the sampler is re-derived FIRST, the auditor
    reads it as an explicit input (`--resolved-sampler`), every verdict carries AGREES with `resolved_sampler_source {path, sha256, rederived_by}` and
    the historical UNCOMPARED alongside; the PASS validator's audit check admits; a source sha that is not the bound record's refuses; another tree's
    re-derivation refuses. Negative case (the audit fixtures' synthetic root, the same real plan row): a log resolving to another sampler -> refused
    by name, the historical verdict never standing in."""

    RS = host_site_path("$STATE/romp/candidate-si-refit/run-state/full22/seedrep_battery")
    W22_ROW = "pasttense__s1__base__shard00"
    #: fc11e: the REAL pulled prior roots of this standup id, BY ATTEMPT ID -- the carry-over pulls other attempts of the same standup id under the same
    #: `carry_over/reaudit/<standup id>/<aid>/pulled` dir (the carried S5 FULL bases), so "the" root is never the first dir found
    W22_S1_SMOKE_AID = "01M2XC3PXY9ZJY9ANBMYG459CR-78e42e05e14a"          # the S1 smoke attempt: 472 client requests (+ 609 probes) in the engine log
    W22_S5_FULL_AID = "01M2XD1H6WDGXZ65BS0NRNMZRR-ccf6feacddc8"           # the carried S5 FULL base of the same row (48 children): 4,184 client requests
    #: (counted in its engine log: 4,184 `max_tokens=4096` + 909 `max_tokens=48` probes = 5,093 `Received request` lines; the prior record reads the same)
    W22_ROOTS = ((W22_S1_SMOKE_AID, 472), (W22_S5_FULL_AID, 4184))

    @staticmethod
    def _served():
        import importlib.util
        spec = importlib.util.spec_from_file_location("seedrep_served_fc11d", REPO / "scripts/modal/seedrep_served.py")
        srv = importlib.util.module_from_spec(spec)
        prior_mod = sys.modules.get("modal")
        sys.modules["modal"] = _stub_modal()
        try:
            spec.loader.exec_module(srv)
        finally:
            if prior_mod is None:
                sys.modules.pop("modal", None)
            else:
                sys.modules["modal"] = prior_mod
        return srv

    def _w22(self, aid: Optional[str] = None, row_id: Optional[str] = None):
        """The real prior archive's plan row and, with ``aid``, THAT attempt's pulled root (`carry_over/reaudit/<row>/<aid>/pulled/<volume-relative root>`) and
        its volume spelling -- selected by attempt id, never the first dir under the standup id; skips by name when the archive / that aid is absent."""
        plan_p = self.RS / "prior/56aff57/launch_plan_draft.json"
        if not plan_p.is_file():
            pytest.skip("the W22 archive (RS/prior/56aff57) is not on this machine")
        prior = co.load_prior_source(plan_p.parent)
        row_id = row_id or self.W22_ROW
        row = lp.plan_row_by_id(prior["plan"], row_id)
        if aid is None:
            return prior, row, None, None
        pulled = self.RS / f"carry_over/reaudit/{row_id}/{aid}/pulled"
        att = sorted(pulled.rglob("ATTEMPT.json")) if pulled.is_dir() else []
        att = [p for p in att if p.parent.name == aid]
        if len(att) != 1:
            pytest.skip(f"the pulled W22 root of attempt {aid} ({pulled}) is not on this machine")
        src_root = att[0].parent
        assert json.loads(att[0].read_text())["attempt_id"] == aid
        return prior, row, src_root, f"/vol/{src_root.relative_to(pulled).as_posix()}"

    def _reaudit(self, tmp_path, *, src_root: Path, aid: str, prior_W: str, row, hist_root: Optional[str] = None, this_sha="5" * 40):
        import shutil
        srv = self._served()
        hist_root = hist_root or f"/vol/seedrep/{prior_W[:7]}/exp/seedrep_battery_modal_smoke/S1/pasttense/s1/shard00/attempts/{aid}"
        uploaded = {}

        def run_get(src, out):
            shutil.copytree(src_root, Path(out) / aid)
        ra = srv.reaudit_carried_root(prev_row=row, this_row=row, prior={"W": prior_W}, aid=aid, root=hist_root,
                                      this_plan={"W": this_sha, "W7": this_sha[:7], "wave_uuid": "22222222-3333-4444-8444-555555555555"}, rs=tmp_path / "rs", this_sha=this_sha,
                                      tree=REPO, run_get=run_get, put_file=lambda rel, data: uploaded.__setitem__(rel, data))
        return srv, ra, hist_root, this_sha, uploaded

    @pytest.mark.parametrize("aid,n_client", W22_ROOTS, ids=["s1_smoke", "s5_full"])
    def test_the_real_w22_s1_root_rederives_agrees_the_auditor_reads_it_explicitly_and_the_validator_admits(self, tmp_path, aid, n_client):
        prior, row, src_root, hist_root = self._w22(aid)
        assert json.loads((src_root / lp.RESOLVED_SAMPLER_REL).read_text())["verdict"] == "UNCOMPARED"          # what the prior recorder stored
        srv, ra, hist_root, this_sha, uploaded = self._reaudit(tmp_path, src_root=src_root, aid=aid, prior_W=prior["W"], row=row, hist_root=hist_root)
        assert "--resolved-sampler" in ra["argv"] and ra["argv"].index("--resolved-sampler") > ra["argv"].index("--audited-at")
        assert ra["rc"] == 0, Path(ra["log"]).read_text()[-2000:]
        smp = ra["sampler"]
        assert smp["historical_verdict"] == "UNCOMPARED" and smp["rederived_verdict"] == "AGREES" and smp["recorder_rc"] == 0 and smp["expected_samplers_rc"] == 0 \
            and smp["expect_tasks"] is True and smp["basis"] == lp.SMOKE_PASS_SAMPLER_BASIS, smp
        # fc11g: the s-view standups' salt-isolation control ran at the probe cap (no one-token requests): declared, conditional, nothing unclassified
        assert smp["probe"]["declared"]["salt_isolation"]["conditional"] is True and "1" not in smp["probe"]["n_probe_by_cap"] and smp["probe"]["n_unclassified"] == 0, smp["probe"]
        ev_dir = Path(ra["local_evidence_root"])
        fresh = json.loads((ev_dir / lp.RESOLVED_SAMPLER_REL).read_text())
        assert fresh["verdict"] == "AGREES" and fresh["client_task_verdicts"] == {"static/pasttense": "AGREES"} and fresh["n_client_requests"] == n_client
        fresh_sha = lp.sha256_file(ev_dir / lp.RESOLVED_SAMPLER_REL)
        assert ra["producer_evidence"][lp.RESOLVED_SAMPLER_REL]["sha256"] == fresh_sha
        assert json.loads((src_root / lp.RESOLVED_SAMPLER_REL).read_text())["verdict"] == "UNCOMPARED"          # the historical root is untouched
        audits = {rel: b for rel, b in ra["producer_evidence"].items() if b.get("bound_rel")}
        assert audits, ra["producer_evidence"]
        child = row["children"][0]
        for rel, b in audits.items():
            text = (ev_dir / b["bound_rel"]).read_text()
            doc = json.loads(text)
            assert doc["verdict"] == "AGREES" and doc["W"] == this_sha and doc["store_W"] == prior["W"]
            assert doc["resolved_sampler_verdict"] == "AGREES" and doc["historical_resolved_sampler_verdict"] == "UNCOMPARED"
            assert doc["resolved_sampler_source"] == {"path": str(ev_dir / lp.RESOLVED_SAMPLER_REL), "sha256": fresh_sha, "rederived_by": this_sha}
            assert doc["historical_resolved_sampler_path"].endswith(f"/{aid}/{lp.RESOLVED_SAMPLER_REL}")
            k = next(k_ for k_ in row["children"] if k_["arm"] == b["plan_arm"])
            common = dict(accepted_attempts=[aid], row=row, aid=aid, root=str(ev_dir), child=k, table_leg="pasttense", store_root=hist_root)
            # the PASS validator's audit check (what derive_smoke_pass_from_priors and the carry's clause (c) run): admitted, the provenance recorded
            facts = lp._smoke_check("request_evidence_audit", text, path=f"{ev_dir}/{b['bound_rel']}", plan={"W": this_sha}, producer=ra["producer_evidence"], **common)
            assert facts["verdict"] == "AGREES" and facts["resolved_sampler_source"]["sha256"] == fresh_sha and facts["historical_resolved_sampler_verdict"] == "UNCOMPARED"
            # the fresh record's bound bytes differ from what the audit read -> refused by name
            producer2 = json.loads(json.dumps(ra["producer_evidence"]))
            producer2[lp.RESOLVED_SAMPLER_REL]["sha256"] = "1" * 64
            with pytest.raises(ValueError, match="not the bound resolved_sampler.json .* the audit's sampler verdict is not the re-derived record's"):
                lp._smoke_check("request_evidence_audit", text, path=f"{ev_dir}/{b['bound_rel']}", plan={"W": this_sha}, producer=producer2, **common)
            # the audit's sampler source re-derived by ANOTHER tree (the audit doc re-issued, re-bound) -> refused by name
            other = json.dumps({**doc, "resolved_sampler_source": {**doc["resolved_sampler_source"], "rederived_by": "6" * 40}})
            producer3 = json.loads(json.dumps(ra["producer_evidence"]))
            producer3[rel]["sha256"] = lp._sha_text(other)
            with pytest.raises(ValueError, match="took its sampler verdict from a record re-derived by '666666666666', not this tree"):
                lp._smoke_check("request_evidence_audit", other, path=f"{ev_dir}/{b['bound_rel']}", plan={"W": this_sha}, producer=producer3, **common)
            assert child  # the loop body ran for a bound verdict
        assert lp.smoke_pass_reaudit_problems(ra, this_W=this_sha, attempt_root=hist_root) == []
        rel_ev = ra["evidence_root"][len("/vol/"):]
        assert f"{rel_ev}/{lp.RESOLVED_SAMPLER_REL}" in uploaded and f"{rel_ev}/{lp.EXPECTED_SAMPLERS_REL}" in uploaded

    def test_a_log_resolving_to_another_sampler_is_refused_by_name_through_the_real_tools(self, tmp_path):
        import seedrep_audit_fixtures as fx
        prior, row, _, _ = self._w22()
        root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
        fx.populate_root(root, s=1, arms=("none",), n_items=2)
        log_p = root.path / "exp/serve/vllm_server.log"
        log_p.write_text(log_p.read_text().replace("SamplingParams(n=1, ", "SamplingParams(n=1, temperature=0.95, top_p=0.5, top_k=7, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0, "))
        (root.path / lp.RESOLVED_SAMPLER_REL).write_text(json.dumps({"verdict": "UNCOMPARED"}))
        srv, ra, hist_root, this_sha, _ = self._reaudit(tmp_path, src_root=root.path, aid=root.attempt_id, prior_W=fx.W_FIXTURE, row=row)
        assert ra["sampler"]["historical_verdict"] == "UNCOMPARED" and ra["sampler"]["rederived_verdict"] != "AGREES", ra["sampler"]
        probs = lp.smoke_pass_reaudit_problems(ra, this_W=this_sha, attempt_root=hist_root)
        assert any("re-derived the sampler verdict as" in p and "not AGREES" in p and "the stored 'UNCOMPARED' does not stand in" in p for p in probs), probs
        if ra["sampler"]["rederived_verdict"] is not None:                                   # the recorder filed a finding: the audit read THAT, never the stored one
            assert ra["sampler"]["rederived_verdict"] == "DIVERGED"
            for rel, b in ra["producer_evidence"].items():
                if b.get("bound_rel") and not b.get("absent"):
                    doc = json.loads((Path(ra["local_evidence_root"]) / b["bound_rel"]).read_text())
                    assert doc["resolved_sampler_verdict"] == "DIVERGED" and doc["historical_resolved_sampler_verdict"] == "UNCOMPARED"

    def test_the_auditor_refuses_a_resolved_sampler_without_the_rederiving_tree(self, tmp_path):
        import audit_common as ac
        import request_evidence_audit as rea
        p = tmp_path / "rs.json"
        p.write_text(json.dumps({"verdict": "AGREES"}))
        with pytest.raises(ac.AuditRefusal, match="--resolved-sampler needs --audited-at"):
            rea.main(["--roots", str(tmp_path), "--out", str(tmp_path / "out"), "--legs", "pasttense", "--resolved-sampler", str(p)])


class TestFc11dCarryClauseC:
    """fc11d (3): the carry-over's clause (c) over a prior-shaped carried base whose stored sampler verdict is UNCOMPARED -- with a re-audit that re-derived the
    sampler (the binding under the evidence root, the audit docs carrying it with provenance) the row CARRIES and the check reads the fresh record; a
    re-audit that did not re-derive it (an fc11-era one) still reads the historical record and refuses at clause (c)."""

    def _world(self, tmp_path):
        import test_seedrep_fj5_carry_fixes as fj5
        w = fj5._w22_shaped_world(tmp_path, strip_arm_binding=False)
        rs_path = f"{w.root}/{lp.RESOLVED_SAMPLER_REL}"
        hist = json.loads(w.files[rs_path])
        w.files[rs_path] = json.dumps({**hist, "verdict": "UNCOMPARED", "reference": None, "reference_source": None})
        ev = json.loads(w.files[f"{w.root}/modal_evidence.json"])
        ev[lp.PRODUCER_EVIDENCE_KEY][lp.RESOLVED_SAMPLER_REL]["sha256"] = fj5.F._sha(w.files[rs_path])
        w.files[f"{w.root}/modal_evidence.json"] = json.dumps(ev)
        return fj5, w

    def test_clause_c_reads_the_rederived_sampler_record_when_the_reaudit_binds_one(self, tmp_path):
        fj5, w = self._world(tmp_path)
        rec = fj5._eval(w, reaudit=fj5._stub_reaudit(w))                                  # an fc11-era re-audit: the historical record is read -> refused
        assert w.this["standups"][0]["standup_id"] in rec["refused"]
        why = rec["refused"][w.this["standups"][0]["standup_id"]]["why"]
        assert "resolved_sampler" in why and "'UNCOMPARED' is not AGREES" in why, why
        base = fj5._stub_reaudit(w)

        def rederiving(**kw):
            ra = base(**kw)
            ev_root = ra["evidence_root"]
            fresh = json.dumps({**json.loads(w.files[f"{w.root}/{lp.RESOLVED_SAMPLER_REL}"]), "verdict": "AGREES", "reference": {"temperature": 0.7}, "reference_source": "--expect-tasks"})
            w.files[f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}"] = fresh
            ra["producer_evidence"][lp.RESOLVED_SAMPLER_REL] = {"sha256": fj5.F._sha(fresh), "writer": lp.PRODUCER_WRITERS["resolved_sampler"], "tree_sha": ra["reaudited_by"], "rc": 0}
            ra["sampler"] = {"historical_verdict": "UNCOMPARED", "rederived_verdict": "AGREES", "rederived_by": ra["reaudited_by"], "recorder_rc": 0, "expected_samplers_rc": 0,
                             "vllm_log": f"{w.root}/exp/serve/vllm_server.log", "expect_tasks": True, "why": None, "basis": lp.SMOKE_PASS_SAMPLER_BASIS, "probe": {}}
            src = {"path": f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": fj5.F._sha(fresh), "rederived_by": ra["reaudited_by"]}
            for rel, b in ra["producer_evidence"].items():
                if b.get("bound_rel"):
                    apath = f"{ev_root}/{b['bound_rel']}"
                    w.files[apath] = json.dumps({**json.loads(w.files[apath]), "resolved_sampler_verdict": "AGREES", "resolved_sampler_source": src, "historical_resolved_sampler_verdict": "UNCOMPARED"})
                    b["sha256"] = fj5.F._sha(w.files[apath])
            return ra
        rec = fj5._eval(w, reaudit=rederiving)
        sid = w.this["standups"][0]["standup_id"]
        assert sid in rec["carried"], rec["refused"].get(sid)
        chk = rec["carried"][sid]["clauses"]["complete"]["checks"]["resolved_sampler"]
        assert chk["reaudit"]["held_to_plan"] == "this" and chk["reaudit"]["sampler"]["historical_verdict"] == "UNCOMPARED" and chk["reaudit"]["sampler"]["rederived_verdict"] == "AGREES"
        assert all(s["path"].startswith(rec["carried"][sid]["reaudit_evidence_dir"] + "/") and s["producer"]["tree_sha"] == fj5.F.W_THIS for s in chk["sources"])
        aud = rec["carried"][sid]["clauses"]["complete"]["checks"]["request_evidence_audit"]
        assert all(s["resolved_sampler_source"]["rederived_by"] == fj5.F.W_THIS and s["historical_resolved_sampler_verdict"] == "UNCOMPARED" for s in aud["sources"])


class TestFc11fSmokeScope:
    """fc11f: a prior SMOKE row's per-arm audit set is the set of arms whose stores the pulled prior root HOLDS (a root from a run with
    `--smoke-max-children 5` holds five children of the `none` arm: one arm's stores, one verdict), never the plan row's full arm list; every
    store present needs a bound verdict; an empty / unenumerated set refuses; a carried PRODUCTION row keeps the full-arm requirement."""

    @staticmethod
    def _multi_arm_smoke(w):
        """A multi-arm S1 world: the world's smoke rows are single-cell (Arditi-view rows; the bon base), so the certificate fixture's `_smoke_world` builds the
        accepted attempt + producer files of a multi-cell `pasttense` s-view row under the SAME prior plan / S1 Dict, merged into a copy of the world.
        Returns (w2, rid, row, cells, table legs)."""
        c = w["c"]
        prior_plan = w["prior"]["plan"]
        row = next(r for r in prior_plan["standups"] if r["leg"] == "pasttense" and len({str(k.get("cell") or k["arm"]) for k in r["children"]}) >= 2)
        rid = row["standup_id"]
        accepted, texts = _CertComponents._smoke_world(prior_plan, prior_plan, c["rs"], c["waves"], {"S1": [rid]})
        ident_s1 = lp.smoke_identity(prior_plan, "S1", c["waves"]["S1"])
        aid = accepted["S1"][lp.accepted_key(ident_s1["W7"], c["waves"]["S1"], row["leg"], row["engine"], row["stage"], row["shard"])]
        _smoke_spell_cells(texts, "S1", {rid: {"attempt_root": lp.attempt_root_for(ident_s1["results_root"], leg=row["leg"], view=row["engine"], stage=row["stage"], shard=row["shard"], attempt_id=aid, kind=row.get("kind"))}})
        name = ident_s1["dicts"]["accepted"]
        by_dict = {k: dict(v) for k, v in w["by_dict"].items()}
        by_dict[name] = {**by_dict.get(name, {}), **accepted["S1"]}
        texts2 = {**w["texts"], **texts}
        w2 = {**w, "texts": texts2, "by_dict": by_dict, "read_text": lp.volume_only(lambda p_: texts2.get(str(p_))), "for_dict": lambda n: by_dict.get(n, {}).get}
        cells = sorted({str(k.get("cell") or k["arm"]) for k in row["children"]})
        legs = sorted({leg for k in row["children"] for leg, _ in lp.audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"])})
        return w2, rid, row, cells, legs

    def test_a_pulled_root_with_one_arms_stores_passes_as_a_subset(self, world):
        w, rid, row, cells, legs = self._multi_arm_smoke(world)
        one = cells[0]
        rec = _derive(w, "S1", rows=[rid], reaudit=_stub_reaudit(w["texts"], W25, arms={one}))
        r = rec["rows"][rid]
        assert r["arms_planned"] == cells and len(cells) > 1 and r["smoke_scope"] == lp.SMOKE_SCOPE_SUBSET
        assert {e["arm"] for e in r["reaudit"]["stores_present"]} == {one} and {e["leg"] for e in r["reaudit"]["stores_present"]} == set(legs)
        assert r["arms_audited"] == sorted(e["identity"] for e in r["reaudit"]["stores_present"]) and all(x.split("|")[2] == one for x in r["arms_audited"])
        aud = rec["checks"]["request_evidence_audit"]
        # the goal-shard children of one arm-run share ONE verdict file per table leg (the inventory multiset rule), so distinct FILES == table legs
        assert {s_["plan_cell"] for s_ in aud["sources"]} == {one} and all(s_["verdict"] == "AGREES" for s_ in aud["sources"]) and len({s_["path"] for s_ in aud["sources"]}) == len(legs)
        assert _problems(w, rec, "S1") == [] and _problems(w, rec, "S1", read_text=None, for_dict=None) == []
        # the stamps are re-derived at validation, never trusted
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["smoke_scope"] = lp.SMOKE_SCOPE_FULL
        assert any("smoke_scope 'full' is not what the re-audit's stores + bindings derive ('subset')" in p for p in _problems(w, bad, "S1"))
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["arms_audited"] = bad["rows"][rid]["arms_audited"][:-1]
        assert any("arms_audited" in p and "is not what the re-audit's stores + bindings derive" in p for p in _problems(w, bad, "S1"))
        # the same root with every arm's stores audited is the FULL scope (the fc11e-era expectation), with more sources
        full = _derive(w, "S1", rows=[rid])
        assert full["rows"][rid]["smoke_scope"] == lp.SMOKE_SCOPE_FULL and sorted({x.split("|")[2] for x in full["rows"][rid]["arms_audited"]}) == cells \
            and len({s_["path"] for s_ in full["checks"]["request_evidence_audit"]["sources"]}) == len(legs) * len(cells)

    def test_two_arms_stores_but_one_audit_refuses_by_name_and_so_does_an_empty_or_unenumerated_set(self, world):
        w, rid, row, cells, legs = self._multi_arm_smoke(world)
        one, second = cells[0], cells[1]
        leg0 = legs[0]
        s0 = str(row["children"][0]["s"])
        ident = lambda leg, s_, arm: {"leg": leg, "s": int(s_), "arm": arm, "store_dir": f"exp/sampled/{leg}__s{s_}/{arm}", "identity": f"{leg}|s{s_}|{arm}|exp/sampled/{leg}__s{s_}/{arm}"}  # noqa: E731
        two = [ident(leg, s0, a) for leg in legs for a in (one, second)]
        with pytest.raises(ValueError, match=rf"{len(legs)} store\(s\) of the pulled root have no bound audit verdict -- a store without an audit is refused: \['{leg0}\|s{s0}\|{second}\|"):
            _derive(w, "S1", rows=[rid], reaudit=_stub_reaudit(w["texts"], W25, arms={one}, stores_present=two))
        with pytest.raises(ValueError, match="holds NO store of the row's table legs"):
            _derive(w, "S1", rows=[rid], reaudit=_stub_reaudit(w["texts"], W25, arms={one}, stores_present=[]))
        def unenumerated(**kw):
            ra = _stub_reaudit(w["texts"], W25, arms={one})(**kw)
            ra.pop("stores_present")
            return ra
        with pytest.raises(ValueError, match=r"reaudit: lacks \['stores_present'\]"):
            _derive(w, "S1", rows=[rid], reaudit=unenumerated)
        def refused_leg(**kw):
            ra = _stub_reaudit(w["texts"], W25, arms={one})(**kw)
            ra["stores_problems"] = {leg0: "AuditRefusal: malformed store"}
            return ra
        with pytest.raises(ValueError, match=rf"stores refused to read for leg {leg0}: AuditRefusal: malformed store"):
            _derive(w, "S1", rows=[rid], reaudit=refused_leg)
        # a verdict bound for a store identity the root does not hold
        with pytest.raises(ValueError, match=rf"verdict\(s\) bound for store identit\(ies\) the pulled root does not hold: \['{leg0}\|s{s0}\|{one} "):
            _derive(w, "S1", rows=[rid], reaudit=_stub_reaudit(w["texts"], W25, arms={one}, stores_present=[ident(leg, s0, second) for leg in legs]))
        # fc11i: nine physical stores (three legs x seeds 1-3, one arm) with only the seed-1 verdicts bound -> refused naming the six unbound identities
        nine = [ident(leg, s_, one) for leg in legs for s_ in (1, 2, 3)]
        with pytest.raises(ValueError) as ei:
            _derive(w, "S1", rows=[rid], reaudit=_stub_reaudit(w["texts"], W25, arms={one}, stores_present=nine))
        msg = str(ei.value)
        n_missing = len(nine) - len([e for e in nine if str(e["s"]) == s0])
        assert f"{n_missing} store(s) of the pulled root have no bound audit verdict" in msg and all(e["identity"] in msg for e in nine if str(e["s"]) != s0), msg
        # fc11i: a valid store beside a candidate store file the reader could not classify -> refused naming the sibling
        def sibling(**kw):
            ra = _stub_reaudit(w["texts"], W25, arms={one})(**kw)
            ra["stores_problems"] = {"unclassified": ["exp/misalign/agentic__s1__murder/stray/responses.jsonl"]}
            return ra
        with pytest.raises(ValueError, match=r"candidate store file\(s\) no table leg of the row could classify .*exp/misalign/agentic__s1__murder/stray/responses.jsonl"):
            _derive(w, "S1", rows=[rid], reaudit=sibling)
        # an arm-name set (the fc11f shape) is no longer an inventory
        with pytest.raises(ValueError, match="stores_present is not an identity inventory"):
            _derive(w, "S1", rows=[rid], reaudit=_stub_reaudit(w["texts"], W25, arms={one}, stores_present={leg0: [one]}))

    def test_a_production_carry_with_a_missing_arm_is_still_refused(self, tmp_path):
        """The carry-over's clause (c) (`complete_verdict`) keeps the FULL-arm requirement: a re-audit that bound one of the row's two children refuses."""
        import test_seedrep_fj5_carry_fixes as fj5
        w = fj5._w22_shaped_world(tmp_path, strip_arm_binding=False)
        base = fj5._stub_reaudit(w)

        def one_arm(**kw):
            ra = base(**kw)
            keep = next(iter(ra["producer_evidence"]))
            ra["producer_evidence"] = {keep: ra["producer_evidence"][keep]}
            ra["stores_present"] = [{"leg": "pasttense", "s": 1, "arm": ra["producer_evidence"][keep]["plan_cell"], "store_dir": "exp/x", "identity": f"pasttense|s1|{ra['producer_evidence'][keep]['plan_cell']}|exp/x"}]
            return ra
        rec = fj5._eval(w, reaudit=one_arm)
        sid = w.this["standups"][0]["standup_id"]
        assert sid in rec["refused"], rec["carried"].keys()
        why = rec["refused"][sid]["why"]
        assert "request_evidence_audit" in why and ("absent on the volume" in why or "the producer bound no" in why), why     # the missing arm, by name
        assert sid in fj5._eval(w, reaudit=base)["carried"]                       # the full-arm re-audit still carries


class TestFc11gProbeKinds:
    """fc11g (A): the prior Arditi S2 root's engine log holds 33 one-token greedy requests -- its `positive_controls.json` salt_isolation control (3 routes x
    the 11-prompt probe set) -- which the recorder's default probe class (PROBE_MAX_TOKENS alone) reads as UNCLASSIFIED_REQUESTS; the re-derivation
    declares that cap from the root's OWN record, names the kind, holds the count to it, and refuses anything left unclassified."""

    W22_ARDITI_ROW, W22_ARDITI_AID = "pasttense__arditi__base__shard00", "01M2XD12ZJGCGEHDFMN15EXKMR-c5c447f38ec7"

    def test_the_real_w22_arditi_root_declares_its_salt_isolation_probes_and_rederives_agrees(self, tmp_path):
        rb = TestFc11dRealBoundary()
        prior, row, src_root, hist_root = rb._w22(self.W22_ARDITI_AID, row_id=self.W22_ARDITI_ROW)
        decl = rb._served().probe_declaration_from_root(src_root)
        assert decl["probe_max_tokens"] == 48 and decl["declared"]["salt_isolation"]["cap"] == 1 and decl["declared"]["salt_isolation"]["expected_count"] == 33, decl
        srv, ra, hist_root, this_sha, uploaded = rb._reaudit(tmp_path, src_root=src_root, aid=self.W22_ARDITI_AID, prior_W=prior["W"], row=row, hist_root=hist_root)
        smp = ra["sampler"]
        assert smp["historical_verdict"] == "UNCOMPARED" and smp["rederived_verdict"] == "AGREES", smp
        assert smp["probe"]["probe_caps_argv"] == [1] and smp["probe"]["n_probe_by_cap"] == {"1": 33, "48": 179} and smp["probe"]["n_unclassified"] == 0, smp["probe"]
        assert smp["probe"]["declared"]["salt_isolation"]["conditional"] is True and "one-token greedy" in smp["probe"]["declared"]["salt_isolation"]["cap_rule"]
        fresh = json.loads((Path(ra["local_evidence_root"]) / lp.RESOLVED_SAMPLER_REL).read_text())
        assert fresh["n_client_requests"] == 178 and fresh["client_task_verdicts"] == {"static/pasttense": "AGREES"} and fresh["n_unclassified_requests"] == 0
        assert lp.smoke_pass_reaudit_problems(ra, this_W=this_sha, attempt_root=hist_root) == []
        assert ra["rc"] == 0 and all(json.loads((Path(ra["local_evidence_root"]) / b["bound_rel"]).read_text())["resolved_sampler_verdict"] == "AGREES"
                                     for b in ra["producer_evidence"].values() if b.get("bound_rel"))

    def test_a_declared_probe_kind_whose_count_differs_or_a_leftover_unclassified_request_refuses_by_name(self, world):
        w = world
        good = _stub_reaudit(w["texts"], W25)
        def miscounted(**kw):
            ra = good(**kw)
            ra["sampler"]["probe"] = {"declared": {"salt_isolation": {"cap": 1, "expected_count": 33, "source": "positive_controls.json", "conditional": True}}, "probe_caps_argv": [1],
                                      "probe_max_tokens": 48, "client_caps": [4096], "n_probe_by_cap": {"1": 32, "48": 179}, "n_unclassified": 0}
            return ra
        with pytest.raises(ValueError, match=r"probe kind 'salt_isolation' \(cap 1, positive_controls.json\) expects 33 request\(s\); the engine log holds 32 at that cap"):
            _derive(w, "S1", reaudit=miscounted)
        def leftover(**kw):
            ra = good(**kw)
            ra["sampler"]["probe"] = {"declared": {}, "probe_caps_argv": [], "probe_max_tokens": 48, "client_caps": [4096], "n_probe_by_cap": {"48": 8}, "n_unclassified": 5}
            return ra
        with pytest.raises(ValueError, match=r"5 request\(s\) of the historical engine log match no client cap and no declared probe kind"):
            _derive(w, "S1", reaudit=leftover)
        rec = _derive(w, "S1")
        rid = rec["standup_ids"][0]
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["sampler"]["probe"]["n_unclassified"] = 1
        assert any("probe classification disagrees with its re-audit record" in p for p in _problems(w, bad, "S1"))

    def test_a_petri_row_is_refused_by_name_with_the_sidecar_route(self, world):
        """fc11g (B): a Petri smoke root is a proxy root (no store, no engine log, no audits/); this derivation names the sidecar route it does not take yet."""
        w = world
        plan = copy.deepcopy(w["draft25"])
        petri = next((r for r in plan["standups"] if r["leg"] == "petri"), None)
        if petri is None:
            pytest.skip("the plan has no petri row")
        prior = copy.deepcopy(w["prior"])
        with pytest.raises(ValueError, match="a Petri row's prior root is a proxy root .*evidence_kind petri_sidecar"):
            lp.derive_smoke_pass_from_priors(rs=w["rs25"], plan=plan, smoke_id="S3", standup_ids=[petri["standup_id"]], priors=[prior],
                                             accepted_get_for_dict=lambda n: (lambda k: "ta-petri"), read_text=w["read_text"], reaudit=_stub_reaudit(w["texts"], W25))


class TestFc11hCarryHoldsAtEverySpawnBoundary:
    """fc11h: a row held by the certificate's carry-over block was refused by `select_launch_rows` only; `reattempt_pass` and the
    stale-intent recovery spawned it. `certified_launch_guard` -- the ONE guard at the launcher's selection, the fan-out (intent, fresh and recovered
    spawns) and the re-attempt's `guarded_spawn` (which the recovery path spawns through) -- now refuses every held id by name and hold kind, reading
    the holds off the SEALED certificate."""

    @staticmethod
    def _cert(held, unheld_ok=True):
        record = {"schema": co.CARRY_OVER_SCHEMA, "refused": {
            held: {"accepted_key": f"5555555|w|pasttense|s1|base|0", "clause": "engine_identity", "why": "the attempt's engine is not the one this plan's row describes",
                   "attempts": [{"prior": "RS/prior/56aff57", "prior_W": "5" * 6 + "6" * 34, "source": "production", "clause": "engine_identity",
                                 "why": "the attempt's engine is not the one this plan's row describes",
                                 "prior_accepted_attempt": {"attempt_id": "01M2X-held-attempt", "attempt_root": "/vol/seedrep/5555556/exp/x", "dict": "seedrep-5555556-w-accepted"}}]}}}
        return {"launch_set": None, "carry_over": record}

    def test_the_guard_refuses_a_held_row_by_name_and_hold_kind_and_spawns_the_unheld(self):
        srv = TestFc11dRealBoundary._served()
        held, other = "pasttense__s1__base__shard00", "bon__s1__base__shard00"
        cert = self._cert(held)
        assert co.carry_refused_holds(cert["carry_over"]) and list(co.carry_refused_holds(cert["carry_over"])) == [held]
        for where in ("production launch", "fan-out", "fan-out spawn", "fan-out re-spawn", "re-attempt spawn"):
            with pytest.raises(srv.LaunchRefused) as ei:
                srv.certified_launch_guard(cert, [other, held], where=where)
            msg = str(ei.value)
            assert msg.startswith(f"[seedrep] {where} REFUSED") and f"row {held} is HELD by the certificate's carry-over block (hold {co.CARRY_REFUSED_HOLD}; clause engine_identity; prior attempt 01M2X-held-attempt" in msg
            assert other not in msg.split("REFUSED", 1)[1]
            srv.certified_launch_guard(cert, [other], where=where)                                    # the unheld row spawns
        srv.certified_launch_guard({"launch_set": None, "carry_over": None}, [held, other], where="fan-out")   # no carry block: nothing held
        # a launch-set violation and a hold report together, both by name
        cert2 = {**cert, "launch_set": [held]}
        with pytest.raises(srv.LaunchRefused) as ei:
            srv.certified_launch_guard(cert2, [held, other], where="fan-out")
        assert f"row {other} is not in the certified launch set" in str(ei.value) and f"row {held} is HELD" in str(ei.value)

    def test_every_spawn_boundary_runs_the_guard_including_the_stale_intent_recovery(self):
        src = (REPO / "scripts/modal/seedrep_served.py").read_text()
        guard = src.split("def certified_launch_guard(", 1)[1].split("\ndef ", 1)[0]
        assert "co.carry_refused_holds(cert.get(\"carry_over\")" in guard and "is HELD by the certificate's carry-over block" in guard
        fan = src.split("def run_fanout(", 1)[1].split("\ndef ", 1)[0]
        assert fan.count("certified_launch_guard(cert, ") == 3
        lp_body = src.split("def launch_production(", 1)[1].split("\ndef ", 1)[0]
        assert 'certified_launch_guard(cert, [r["standup_id"] for r in picked], where="production launch")' in lp_body
        rp = src.split("def reattempt_pass(", 1)[1].split("\ndef ", 1)[0]
        assert 'certified_launch_guard(certificate, [row["standup_id"]], where="re-attempt spawn")' in rp and "spawn=guarded_spawn" in rp
        # the stale-intent recovery spawns ONLY through the policy's `spawn` argument -- the re-attempt's guarded_spawn -- never a bare spawn_row
        pol = src.split("def apply_reattempt_policy(", 1)[1].split("\ndef ", 1)[0]
        rec = pol.split("for stale in stale_unspawned_intents(", 1)[1].split("for c in calls", 1)[0]
        assert "new_cid = spawn(rows_by_sid[sid], it)" in rec and "spawn_row(" not in rec
        assert "reattempt_pass(plan_d, rs_p, run_name, pools, certificate=cert)" in src.split("def main(", 1)[1]


class TestFc11iIdentityInventoryOnTheRealW23Root:
    """fc11i: the real pulled prior controls / Arditi root (read-only, copied): nine store identities (xstest / or_bench_hard_1k / mmlu x seeds 1-3, arm
    arditi_allpos); the full real re-audit binds all nine -> scope `full`, arms_audited the nine identities; the same re-audit with the six seed-2/3
    bindings dropped -> refused naming the six; a store removed from the recorded inventory -> caught by re-enumerating the pulled root."""

    W23_ROW, W23_AID = "controls__arditi__base__shard00", "01M2Y2QKSF0ZYXCDNM1EKTJVFR-067903fedf7f"

    def test_nine_identities_all_bound_pass_full_and_six_dropped_refuse_by_name(self, tmp_path):
        import audit_common as ac
        rs = TestFc11dRealBoundary.RS
        plan_p = rs / "prior/6d69006/launch_plan_draft.json"
        pulled = rs / f"carry_over/reaudit/{self.W23_ROW}/{self.W23_AID}/pulled"
        att = [p for p in (sorted(pulled.rglob("ATTEMPT.json")) if pulled.is_dir() else []) if p.parent.name == self.W23_AID]
        if not plan_p.is_file() or len(att) != 1:
            pytest.skip("the W23 archive / pulled controls-arditi root is not on this machine")
        prior = co.load_prior_source(plan_p.parent)
        row = lp.plan_row_by_id(prior["plan"], self.W23_ROW)
        src_root = att[0].parent
        inv = ac.inventory_stores(src_root, ["xstest", "or_bench_hard_1k", "mmlu"])
        assert inv["problems"] == {} and inv["unclassified"] == [] and len(inv["stores"]) == 9
        assert {(e["leg"], e["s"], e["arm"]) for e in inv["stores"]} == {(leg, s_, "arditi_allpos") for leg in ("xstest", "or_bench_hard_1k", "mmlu") for s_ in (1, 2, 3)}
        rb = TestFc11dRealBoundary()
        hist_root = f"/vol/{src_root.relative_to(pulled).as_posix()}"
        srv, ra, hist_root, this_sha, uploaded = rb._reaudit(tmp_path, src_root=src_root, aid=self.W23_AID, prior_W=prior["W"], row=row, hist_root=hist_root)
        assert ra["rc"] == 0 and len(ra["stores_present"]) == 9 and ra["stores_problems"] == {}, (ra["stores_problems"], Path(ra["log"]).read_text()[-1500:])
        bound = {rel: b for rel, b in ra["producer_evidence"].items() if b.get("bound_rel")}
        assert len(bound) == 9, sorted(ra["producer_evidence"])
        scope = lp.prior_smoke_audit_scope(row, ra, reenumerate=True)
        assert scope["problems"] == [] and scope["smoke_scope"] == lp.SMOKE_SCOPE_FULL and len(scope["arms_audited"]) == 9 and len(scope["children"]) == 3, scope["problems"]
        assert ra["sampler"]["rederived_verdict"] == "AGREES" and ra["sampler"]["probe"]["n_unclassified"] == 0
        ra2 = json.loads(json.dumps(ra))
        for rel in list(ra2["producer_evidence"]):
            if rel.startswith("audits/") and (rel.endswith("__s2.json") or rel.endswith("__s3.json")):
                ra2["producer_evidence"][rel] = {"absent": True, "why": "dropped", "writer": "x", "tree_sha": this_sha, "rc": 0}
        scope2 = lp.prior_smoke_audit_scope(row, ra2, reenumerate=True)
        assert any(p.startswith("6 store(s) of the pulled root have no bound audit verdict") for p in scope2["problems"]), scope2["problems"]
        named = [e["identity"] for e in ra["stores_present"] if e["s"] in (2, 3)]
        assert len(named) == 6 and all(any(n in p for p in scope2["problems"]) for n in named)
        ra3 = json.loads(json.dumps(ra))
        ra3["stores_present"] = ra3["stores_present"][:-1]
        assert any("re-enumerating the pulled root" in p and "the record 8" in p for p in lp.prior_smoke_audit_scope(row, ra3, reenumerate=True)["problems"])
        # fc11k: an unknown candidate NESTED beneath a recognized controls cell of the pulled root (a copy; the source is read-only) --
        # no manifest, no reader consumed it -- is a named unclassified problem at re-enumeration and refuses the scope
        import shutil
        pulled_copy = tmp_path / "pulled_copy"
        shutil.copytree(Path(ra["pulled_root"]), pulled_copy)
        cell = pulled_copy / ra["stores_present"][0]["store_dir"]
        stray = cell / "unknown-layout" / "stray"
        stray.mkdir(parents=True)
        (stray / "rows.jsonl").write_text(json.dumps({"task": "x"}) + "\n")
        inv2 = ac.inventory_stores(pulled_copy, ["xstest", "or_bench_hard_1k", "mmlu"])
        stray_rel = (stray / "rows.jsonl").relative_to(pulled_copy).as_posix()
        assert inv2["unclassified"] == [stray_rel] and len(inv2["stores"]) == 9
        ra4 = json.loads(json.dumps(ra))
        ra4["pulled_root"] = str(pulled_copy)
        probs4 = lp.prior_smoke_audit_scope(row, ra4, reenumerate=True)["problems"]
        assert any("re-enumerating the pulled root" in p and stray_rel in p for p in probs4), probs4


class TestFc11jServedSmokeRowAndD178:
    """(1) A prior smoke root's cells.json names the SERVED smoke row (`lp.smoke_standup_id`, smoke_row's spelling), which the bank check is held
    to -- both spellings stamped; another standup / attempt still refuses. (2) A producer replay artefact absent at the prior root (the prior bon
    smoke's determinism audit, rc 2) is a named qualification, never a pass; present-and-disagreeing still refuses; any other absent check refuses."""

    def test_s1b_fitted_arms_pass_under_the_smoke_spelling_and_other_spellings_refuse(self, world):
        w = world
        rec = _derive(w, "S1b")
        rid = rec["standup_ids"][0]
        assert rec["rows"][rid]["served_standup_id"] == {"plan": rid, "smoke": f"smoke-S1b__{rid}"} == {"plan": rid, "smoke": lp.smoke_standup_id("S1b", rid)}
        assert rec["checks"]["adapter_hashes_banked"]["sources"] and _problems(w, rec, "S1b") == []
        root = rec["rows"][rid]["attempt_root"]
        cp = f"{root}/{lp.CELLS_JSON_REL}"
        for label, standup, attempt in (("the plan row's spelling", rid, None), ("another smoke id", lp.smoke_standup_id("S2", rid), None), ("another attempt", None, "ta-other")):
            texts = dict(w["texts"])
            doc = json.loads(texts[cp])
            if standup is not None:
                doc["serving"]["standup_id"] = standup
            if attempt is not None:
                doc["serving"]["attempt_id"] = attempt
            texts[cp] = json.dumps(doc)
            ev = json.loads(texts[f"{root}/modal_evidence.json"])
            ev[lp.PRODUCER_EVIDENCE_KEY][lp.CELLS_JSON_REL]["sha256"] = lp._sha_text(texts[cp])
            texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
            rt = lp.volume_only(lambda p_, _t=texts: _t.get(str(p_)))
            with pytest.raises(ValueError, match="adapter_hashes_banked: .* was served for attempt .* not "):
                _derive(w, "S1b", reaudit=_stub_reaudit(texts, W25), read_text=rt)
        bad = copy.deepcopy(rec)
        bad["rows"][rid]["served_standup_id"]["smoke"] = rid
        assert any("served_standup_id must record the plan row" in p for p in _problems(w, bad, "S1b"))
        # the spelling helper IS smoke_row's
        row = lp.plan_row_by_id(w["final25"], rid)
        assert lp.smoke_row(row, {"smoke_id": "S1b", "wave_uuid": w["c"]["waves"]["S1b"], "W": W25, "W7": W25[:7], "results_root": "/vol/x", "ladder_root": "/vol/y", "dicts": w["final25"]["dicts"]},
                            max_children=5, max_timeout_s=60)["standup_id"] == lp.smoke_standup_id("S1b", rid)

    def _bon_without_determinism(self, w):
        texts = dict(w["texts"])
        rec0 = w["c"]["smokes"]["S3"]
        gone = {}
        for rid, r in rec0["rows"].items():
            root = r["attempt_root"]
            row = lp.plan_row_by_id(w["final25"], rid)
            for leg, rel in lp.determinism_rels(row):
                gone[rid] = gone.get(rid, []) + [f"{root}/{rel}"]
                texts.pop(f"{root}/{rel}", None)
            ev = json.loads(texts[f"{root}/modal_evidence.json"])
            for leg in list(ev.get(lp.DETERMINISM_EVIDENCE_KEY) or {}):
                ev[lp.DETERMINISM_EVIDENCE_KEY][leg] = {"absent": True, "why": f"{leg}.json was not written (tool rc 2)", "rc": 2, "writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": w["prior"]["W"]}
            texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
        return texts, gone

    def test_a_bon_root_without_its_determinism_audit_passes_with_the_d178_qualification(self, world):
        w = world
        texts, gone = self._bon_without_determinism(w)
        rt = lp.volume_only(lambda p_, _t=texts: _t.get(str(p_)))
        rec = _derive(w, "S3", reaudit=_stub_reaudit(texts, W25), read_text=rt)
        for rid, paths in gone.items():
            quals = rec["rows"][rid]["qualifications"]
            assert [q["artefact"] for q in quals] == paths and all(q["check"] == "determinism" and q["status"] == lp.PRIOR_NOT_PRODUCED_STATUS and q["ruling"] == "D178"
                                                                     and q["measured_under_W"] == w["prior"]["W"] and "D135" in q["why"] and q["producer_binding"]["absent"] is True for q in quals)
        det = rec["checks"]["determinism"]
        assert det["ok"] is True and det["sources"] == [] and [(x["standup_id"], x["artefact"]) for x in det["not_produced"]] == [(rid, p_) for rid, ps in gone.items() for p_ in ps]
        assert _problems(w, rec, "S3", read_text=rt) == [] and _problems(w, rec, "S3", read_text=None, for_dict=None) == []
        # the artefact appearing on the volume later -> the record no longer stands (re-derive)
        assert any("EXISTS on the volume today" in p for p in _problems(w, rec, "S3"))          # the world's texts still hold the determinism files
        # qualification edited: another check / a made-up artefact / the check's not_produced not restating the rows
        rid = rec["standup_ids"][0]
        bad = copy.deepcopy(rec); bad["rows"][rid]["qualifications"][0]["check"] = "request_evidence_audit"
        assert any("is not a D178 not_produced_at_prior_W of an admitted check" in p for p in _problems(w, bad, "S3", read_text=rt))
        bad = copy.deepcopy(rec); bad["rows"][rid]["qualifications"][0]["artefact"] = bad["rows"][rid]["attempt_root"] + "/audits/determinism/other.json"
        assert any("not a producer artefact of check determinism" in p for p in _problems(w, bad, "S3", read_text=rt))
        bad = copy.deepcopy(rec); bad["checks"]["determinism"]["not_produced"] = []
        assert any("does not restate the rows' D178 qualifications" in p for p in _problems(w, bad, "S3", read_text=rt))

    def test_a_present_diverged_determinism_audit_still_refuses_and_so_does_any_other_absent_check(self, world):
        w = world
        rec0 = w["c"]["smokes"]["S3"]
        rid, r = next(iter(rec0["rows"].items()))
        root = r["attempt_root"]
        row = lp.plan_row_by_id(w["final25"], rid)
        leg, rel = lp.determinism_rels(row)[0]
        texts = dict(w["texts"])
        doc = json.loads(texts[f"{root}/{rel}"]); doc["reproduces"] = False; doc["identical"] = 0; doc["misses"] = ["i0"]
        texts[f"{root}/{rel}"] = json.dumps(doc)
        ev = json.loads(texts[f"{root}/modal_evidence.json"]); ev[lp.DETERMINISM_EVIDENCE_KEY][leg]["sha256"] = lp._sha_text(texts[f"{root}/{rel}"]); texts[f"{root}/modal_evidence.json"] = json.dumps(ev)
        # fg1 / F-AW on this path: the diverged record STILL refuses -- the world's resolved-sampler record states no client decode (`distinct_resolved`
        # is a count there), so the exemption cannot be judged and the byte-identical requirement applies in full (the fail-closed reading)
        with pytest.raises(ValueError, match=r"determinism: .*0 of 4 re-sent request\(s\) reproduced byte-identically -- criterion 3 requires every one, and the D180 exemption cannot be judged"):
            _derive(w, "S3", reaudit=_stub_reaudit(texts, W25), read_text=lp.volume_only(lambda p_, _t=texts: _t.get(str(p_))))
        # a different absent artefact -- the row's resolved_sampler at the historical root is not the qualification's business, but an absent AUDIT verdict is: refused
        def audit_gone(**kw):
            ra = _stub_reaudit(w["texts"], W25)(**kw)
            for rel_, b in ra["producer_evidence"].items():
                if b.get("bound_rel"):
                    w_texts_path = f"{ra['evidence_root']}/{b['bound_rel']}"
                    ra["_gone"] = w_texts_path
                    break
            return ra
        texts2 = dict(w["texts"])
        ra_box = {}
        def audit_gone2(**kw):
            ra = _stub_reaudit(texts2, W25)(**kw)
            rel_ = next(k for k, b in ra["producer_evidence"].items() if b.get("bound_rel"))
            texts2.pop(f"{ra['evidence_root']}/{ra['producer_evidence'][rel_]['bound_rel']}", None)
            return ra
        with pytest.raises(ValueError, match="check 'request_evidence_audit': the producer's .* is absent on the volume"):
            _derive(w, "S3", reaudit=audit_gone2, read_text=lp.volume_only(lambda p_, _t=texts2: _t.get(str(p_))))
        assert set(lp.PRIOR_NOT_PRODUCED_ADMITTED) == {"determinism"}

    def test_the_real_w22_roots_read_only(self):
        """The S1b s2 root's cells.json names the served smoke row exactly as `lp.smoke_standup_id` spells it and the bank check admits it under that spelling
        (refusing the plan spelling); the bon root has no audits/determinism/bon.json and its evidence records the replay as absent, rc 2."""
        rs = TestFc11dRealBoundary.RS
        s1b = rs / "carry_over/reaudit/pasttense__s2__base__shard00/01M2XD12V98YMPDGS09PGWS3WR-0559db11fbe7/pulled"
        bon = rs / "carry_over/reaudit/bon__s1__base__shard00/01M2XD134CBQ8B83G527DTNZFR-2c9dada85033/pulled"
        plan_p = rs / "prior/56aff57/launch_plan_draft.json"
        a1 = [p for p in (sorted(s1b.rglob("ATTEMPT.json")) if s1b.is_dir() else []) if p.parent.name == "01M2XD12V98YMPDGS09PGWS3WR-0559db11fbe7"]
        a3 = [p for p in (sorted(bon.rglob("ATTEMPT.json")) if bon.is_dir() else []) if p.parent.name == "01M2XD134CBQ8B83G527DTNZFR-2c9dada85033"]
        if not plan_p.is_file() or len(a1) != 1 or len(a3) != 1:
            pytest.skip("the W22 archive / pulled S1b s2 and bon roots are not on this machine")
        prior = co.load_prior_source(plan_p.parent)
        root1 = a1[0].parent
        cells = json.loads((root1 / lp.CELLS_JSON_REL).read_text())
        assert cells["serving"]["standup_id"] == lp.smoke_standup_id("S1b", "pasttense__s2__base__shard00") == "smoke-S1b__pasttense__s2__base__shard00"
        row = lp.plan_row_by_id(prior["plan"], "pasttense__s2__base__shard00")
        ev = json.loads((root1 / "modal_evidence.json").read_text())
        aid = "01M2XD12V98YMPDGS09PGWS3WR-0559db11fbe7"
        hist_root = f"/vol/{root1.relative_to(s1b).as_posix()}"
        text = (root1 / lp.CELLS_JSON_REL).read_text()
        facts = lp._smoke_check("adapter_hashes_banked", text, path=f"{hist_root}/{lp.CELLS_JSON_REL}", plan=prior["plan"], accepted_attempts=[aid],
                                row={**row, "standup_id": lp.smoke_standup_id("S1b", row["standup_id"])}, aid=aid, root=hist_root, producer=lp.producer_map_for("adapter_hashes_banked", ev))
        assert facts["n_planned"] >= 1 and facts["n_banked"] >= facts["n_planned"]
        with pytest.raises(ValueError, match="was served for attempt .* / row 'smoke-S1b__pasttense__s2__base__shard00', not .* / 'pasttense__s2__base__shard00'"):
            lp._smoke_check("adapter_hashes_banked", text, path=f"{hist_root}/{lp.CELLS_JSON_REL}", plan=prior["plan"], accepted_attempts=[aid], row=row, aid=aid, root=hist_root,
                            producer=lp.producer_map_for("adapter_hashes_banked", ev))
        root3 = a3[0].parent
        assert not (root3 / "audits" / "determinism" / "bon.json").exists()
        ev3 = json.loads((root3 / "modal_evidence.json").read_text())
        assert ev3[lp.DETERMINISM_EVIDENCE_KEY]["bon"]["absent"] is True and ev3[lp.DETERMINISM_EVIDENCE_KEY]["bon"]["rc"] == 2
        brow = lp.plan_row_by_id(prior["plan"], "bon__s1__base__shard00")
        assert [rel for _l, rel in lp.determinism_rels(brow)] == ["audits/determinism/bon.json"]


class TestFc11mReaderLoadedFromThisTree:
    """fc11m: the validator's store reader must be THIS tree's `scripts/sdfbase/seedrep/audit_common.py`, never whichever `audit_common`
    an ambient import would answer: (a) a foreign importable copy first on sys.path -> the lane's file is loaded; (b) a CACHED foreign module in
    sys.modules -> refused by name, the foreign path recorded; (c) the flat image layout -> refused by name, recorded as a re-enumeration problem."""

    WANT = (REPO / lp.AUDIT_COMMON_REL).resolve()

    @staticmethod
    def _foreign(tmp_path):
        d = tmp_path / "foreign"
        d.mkdir()
        (d / "audit_common.py").write_text("FOREIGN = True\n")          # different content: no inventory_stores
        return d

    def test_a_foreign_copy_first_on_sys_path_never_wins(self, tmp_path):
        d = self._foreign(tmp_path)
        saved_mod, saved_path = sys.modules.get("audit_common"), list(sys.path)
        try:
            sys.modules.pop("audit_common", None)
            sys.path.insert(0, str(d))
            mod = lp._audit_common_module()
            assert Path(mod.__file__).resolve() == self.WANT and hasattr(mod, "inventory_stores") and not hasattr(mod, "FOREIGN"), mod.__file__
            assert sys.modules["audit_common"] is mod
            assert lp.load_audit_common(REPO) is mod                                   # the cached module of THIS tree is reused
        finally:
            sys.path[:] = saved_path
            if saved_mod is not None:
                sys.modules["audit_common"] = saved_mod
            else:
                sys.modules.pop("audit_common", None)

    def test_a_cached_foreign_module_is_refused_by_name(self, tmp_path):
        import types
        d = self._foreign(tmp_path)
        saved_mod = sys.modules.get("audit_common")
        try:
            fake = types.ModuleType("audit_common")
            fake.__file__ = str(d / "audit_common.py")
            fake.inventory_stores = lambda *a, **k: {"stores": [], "problems": {}, "unclassified": []}     # implements the interface -- must still not run
            sys.modules["audit_common"] = fake
            with pytest.raises(ImportError, match=r"already imported from '.*foreign/audit_common.py', not this tree's .*scripts/sdfbase/seedrep/audit_common.py -- refused"):
                lp._audit_common_module()
            # the validator records it as a re-enumeration problem, never running the foreign reader
            row = {"standup_id": "x", "leg": "pasttense", "engine": "s1", "stage": "base", "shard": 0, "children": []}
            ra = {"stores_present": [{"leg": "pasttense", "s": 1, "arm": "none", "store_dir": "exp/x", "identity": "pasttense|s1|none|exp/x"}], "stores_problems": {},
                  "producer_evidence": {}, "pulled_root": str(tmp_path), "table_legs": ["pasttense"]}
            probs = lp.prior_smoke_audit_scope(row, ra, reenumerate=True)["problems"]
            assert any("could not be re-enumerated: ImportError" in p and "foreign/audit_common.py" in p for p in probs), probs
        finally:
            if saved_mod is not None:
                sys.modules["audit_common"] = saved_mod
            else:
                sys.modules.pop("audit_common", None)

    def test_the_flat_image_layout_still_refuses_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lp, "__file__", "/root/seedrep_launch_plan.py")
        with pytest.raises(ImportError, match=r"/root/seedrep_launch_plan.py is not at <repo>/scripts/modal \(the image's flat install\)"):
            lp._audit_common_module()
        row = {"standup_id": "x", "leg": "pasttense", "engine": "s1", "stage": "base", "shard": 0, "children": []}
        ra = {"stores_present": [{"leg": "pasttense", "s": 1, "arm": "none", "store_dir": "exp/x", "identity": "pasttense|s1|none|exp/x"}], "stores_problems": {},
              "producer_evidence": {}, "pulled_root": str(tmp_path), "table_legs": ["pasttense"]}
        probs = lp.prior_smoke_audit_scope(row, ra, reenumerate=True)["problems"]
        assert any("could not be re-enumerated: ImportError" in p and "flat install" in p for p in probs), probs
        # a tree without the reader file refuses by name too
        with pytest.raises(ImportError, match="is not on this tree"):
            lp.load_audit_common(tmp_path)
        import ast
        src = (REPO / "scripts/modal/seedrep_launch_plan.py").read_text()
        hits = sorted({n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == "parents"})
        assert len(hits) == 2, hits                                                    # the runtime-identity pin: exactly the two guarded repo walks
        srv_src = (REPO / "scripts/modal/seedrep_served.py").read_text()
        assert "ac = lp.load_audit_common(Path(tree))" in srv_src and 'importlib.import_module("audit_common")' not in srv_src
