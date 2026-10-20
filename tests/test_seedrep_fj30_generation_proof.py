"""fj30 (D191): a GENERATION-ONLY sidecar proof for a petri-MODE row whose attempt has no Petri completion bundle (the misalign leg) -- the operator's
document at RS/carry_over/proofs/<standup_id>__<attempt_id>.json is RE-DERIVED from the volume bytes by the evaluator and by the certificate, and
held to the document field by field; the scope guard keeps a Petri-LEG row on the D115 store proof."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import misalign_generation_proof as mgp  # noqa: E402
import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_fj2_carry_over as F  # noqa: E402

W_PREV, W_THIS, WAVE, AID = F.W_PREV, F.W_THIS, F.WAVE, F.AID
SID = "misalign__s1__base__shard00"
#: the two trees' manifests, mirroring the real launch change: the huihui auditor leg REMOVED, opus-4-8 ADDED to the auditor chain, a judge chain ADDED
MANIFEST_V1 = ("petri:\n  auditor_model: anthropic/claude-sonnet-5\n  auditor_fallback_models:\n    - openrouter/x-ai/grok-4.6\n"
               "    - openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated\n  judge_model: anthropic/claude-opus-5\n")
MANIFEST_V2 = ("petri:\n  auditor_model: anthropic/claude-sonnet-5\n  auditor_fallback_models:   # D120\n    - anthropic/claude-opus-4-8\n    - openrouter/x-ai/grok-4.6\n"
               "  judge_fallback_models:\n    - anthropic/claude-opus-4-8\n    - openrouter/x-ai/grok-4.6\n  judge_model: anthropic/claude-opus-5\n")
REMOVED, ADDED = "openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated", "anthropic/claude-opus-4-8"
CELLS = ("none", "prompted__safety")
BEHAVIORS = ("leaking", "murder")
from sisteer.capeval.petri.defense_cells import virtual_model_name as _vm  # noqa: E402 -- the tree's spelling: none -> petri__base
from _env import host_site_path  # noqa: E402  (tests/ helper: a host-only data path through the site config)


def _agd(dicts):
    return lambda n: dicts.get(n, {}).get


def _launched(prior_dir: Path, W: str, launched: dict, production=None) -> None:
    (prior_dir / co.SMOKES_LAUNCHED_REL).write_text(json.dumps({"schema": co.SMOKES_LAUNCHED_SCHEMA, "W": W, "launched": launched, "production": production}))


def _misalign_row(W: str, W7: str, *, sid: str = SID, leg: str = "misalign") -> dict:
    """A misalign row in the archived plan's shape: children run the misalign runner with `--out {attempt_root}/exp/misalign/agentic__s1__<beh>`
    and `--only-cells <cell>` (the store the proof reads is derived from these), no `mode` field (the leg table carries the mode)."""
    row = F._row(W, W7, sid=sid, leg=leg)
    row["children"] = [{"eval_leg": f"misalign_{beh}", "arm": cell, "cell": cell, "s": 1, "task": f"agentic/{beh}",
                        "argv": ["{driver_python}", "-m", "sisteer.capeval.misalign.runner", "--proxy-url", "{proxy_url}", "--cells", "{attempt_root}/exp/serve/cells.json",
                                 "--family", "agentic", "--out", f"{{attempt_root}}/exp/misalign/agentic__s1__{beh}", "--n", "10", "--behaviors", beh, "--max-tokens", "3000",
                                 "--sampler", "vendor", "--only-cells", cell]}
                    for beh in BEHAVIORS for cell in CELLS]
    row["eval_legs"] = [f"misalign_{b}" for b in BEHAVIORS]
    row["cells_served_expected"] = {"canonical": list(CELLS), "controls": []}                  # the target set derives from HERE, never from a document
    return row


def _every_child_audited(files: dict, root: str, row: dict, W: str) -> None:
    """fj2's synthetic root writes ONE audit verdict (the first child's); a misalign row has four children -- re-issue the verdict per child (and its
    producer binding), the way fj24's controls helper does, so clause (c) reads every one."""
    ev = json.loads(files[f"{root}/modal_evidence.json"])
    k0 = row["children"][0]
    written = lp.audit_verdict_rel(k0["eval_leg"], k0["arm"], k0["s"], stage=row["stage"], shard=row["shard"])
    audit = json.loads(files[f"{root}/{written}"])
    for k in row["children"]:
        for leg, rel in lp.audit_verdict_rels(k["eval_leg"], k["arm"], k["s"], stage=row["stage"], shard=row["shard"]):
            cell = str(k.get("cell") or k["arm"])
            text = json.dumps({**audit, "leg": leg, "arm": cell, "s": k["s"], "verdict_file": rel[len(lp.AUDIT_OUT_REL) + 1:],
                               "arm_binding": {**audit["arm_binding"], "arm": cell, "harness_id": cell, "traintest_id": cell, "vmodel": f"petri__{cell}"}})
            files[f"{root}/{rel}"] = text
            ev[lp.PRODUCER_EVIDENCE_KEY][rel] = {"sha256": F._sha(text), "writer": lp.PRODUCER_WRITERS["request_evidence_audit"], "tree_sha": W, "rc": 0}
    ev["children"] = [{"arm": k["arm"], "s": k["s"], "rc": 0, "eval_leg": k["eval_leg"]} for k in row["children"]]
    det0_leg, det0_rel = lp.determinism_rels(row)[0]
    det = json.loads(files[f"{root}/{det0_rel}"])
    for leg, rel in lp.determinism_rels(row):                                        # one determinism record per TABLE leg (two behaviours here)
        text = json.dumps({**det, "leg": leg})
        files[f"{root}/{rel}"] = text
        ev[lp.DETERMINISM_EVIDENCE_KEY][leg] = {"sha256": F._sha(text), "writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": W, "rc": 0}
    files[f"{root}/modal_evidence.json"] = json.dumps(ev)


def _store(files: dict, root: str, *, taint: str | None = None) -> None:
    """The misalign attempt's generation-only store on the fake volume: per behaviour a proxy response_evidence.jsonl, per cell meta.json + responses.jsonl."""
    for beh in BEHAVIORS:
        out = f"{root}/exp/misalign/agentic__s1__{beh}"
        files[f"{out}/response_evidence.jsonl"] = "".join(json.dumps({"request_id": f"r{i}", "vmodel": _vm("none"), "status": 200}) + "\n" for i in range(3))
        for cell in CELLS:
            vm = _vm(cell)
            files[f"{out}/{cell}/meta.json"] = json.dumps({"cell_id": cell, "virtual_model": vm, "served_models": [_vm(c) for c in CELLS], "params": {"sampler": "vendor"}})
            rows = [{"item_id": f"it{i}", "sample_idx": 0, "model_reported": vm, "virtual_model": vm, "response": "ok"} for i in range(4)]
            if taint and beh == "murder" and cell == "none":
                rows[2]["response"] = f"judged by {taint}"
            files[f"{out}/{cell}/responses.jsonl"] = "".join(json.dumps(r) + "\n" for r in rows)


class World:
    """A misalign carry shaped like the earlier wave's: the FULL smoke of the misalign row accepted under the prior wave; the MANIFEST differs
    by an admitted chain-list change (the chain-kind admission triggers the sidecar-proof requirement on a petri-MODE row); the evidence has
    `petri: None`."""

    def __init__(self, tmp_path: Path, *, taint: str | None = None, leg_row=None):
        prev_row = leg_row(W_PREV, W_PREV[:7]) if leg_row else _misalign_row(W_PREV, W_PREV[:7])
        this_row = leg_row(W_THIS, W_THIS[:7]) if leg_row else _misalign_row(W_THIS, W_THIS[:7])
        self.prev = F._plan(W_PREV, materialisation="draft", rows=[prev_row])
        self.this = F._plan(W_THIS, materialisation="final", rows=[this_row])
        self.sident = lp.smoke_identity(self.prev, "S5", F.S5_WAVE)
        self.srow = lp.smoke_row(prev_row, self.sident, max_children=len(prev_row["children"]), max_timeout_s=int(prev_row["modal"]["timeout_s"]))
        self.files, self.dicts = {}, {}
        self.root = F._root_files(self.files, plan=self.prev, row=self.srow, identity=self.sident, smoke_block=self.srow["smoke"], n_children=len(self.srow["children"]))
        _every_child_audited(self.files, self.root, self.srow, W_PREV)
        ev = json.loads(self.files[f"{self.root}/modal_evidence.json"]); ev["petri"] = None
        self.files[f"{self.root}/modal_evidence.json"] = json.dumps(ev)
        _store(self.files, self.root, taint=taint)
        self.dicts[self.sident["dicts"]["accepted"]] = {self.srow["accepted_key"]: AID}
        d = F._prior_dir(tmp_path, self.prev, s5_wave=F.S5_WAVE); _launched(d, W_PREV, {"S5": F.S5_WAVE}, production=None)
        self.prior = co.load_prior_source(d)
        src = {(W_PREV, F.MANIFEST): MANIFEST_V1, (W_THIS, F.MANIFEST): MANIFEST_V2}
        helpers = {f: "same" for f in (*co.LEG_HELPERS.get("petri", ()), *co.LEG_HELPERS.get("misalign", ()), "src/sisteer/capeval/misalign/runner.py")}
        self.git = F.FakeGit({W_PREV: F._blobs(W_PREV, **helpers), W_THIS: F._blobs(W_THIS, **{**helpers, F.MANIFEST: "manifest-v2"})}, sources=src)
        self.expected = mgp.expected_inputs(self.srow["children"], self.root)
        self.policy = co.generation_proof_policy(self.srow, git=self.git, prev_W=W_PREV, this_W=W_THIS)

    def document(self, *, policy=None, **over) -> dict:
        """The operator's document, derived from the (fake) volume bytes exactly as the CLI would over a pull, under the tree-derived policy."""
        pol = policy or self.policy
        doc = mgp.derive_over(self.files.get, root=self.root, files=self.expected, removed=pol["removed"], added=pol["added"], target=pol["target"], policy=pol)
        return {"kind": co.GENERATION_PROOF_KIND, "ruling": co.GENERATION_PROOF_RULING, **doc, **over}

    def evaluate(self, doc, **kw):
        docs = {(SID, AID): doc} if doc is not None else {}
        return co.evaluate_carry_over(this_plan=self.this, this_sha=W_THIS, priors=[self.prior], read_text=self.files.get, accepted_get_for_dict=_agd(self.dicts),
                                      git=self.git, readiness_problems=F.fake_readiness, generation_proof_doc=lambda sid, aid: docs.get((sid, aid)), **kw)

    def certify(self, rec, read_text=None):
        return co.carry_over_problems(rec, plan=self.this, read_text=read_text or self.files.get, git=self.git, accepted_get_for_dict=_agd(self.dicts))


class TestGenerationProofAdmits:
    def test_the_document_is_re_derived_from_the_volume_bytes_and_carried_with_the_d191_record(self, tmp_path):
        w = World(tmp_path)
        assert co.petri_proof_required(w.srow, [{"path": F.MANIFEST, "kind": co.ADMITTED_KIND_CHAIN}]) and co.is_petri_mode_row(w.srow)
        assert len(w.expected) == 1 + 2 + 2 * 2 * 2                                              # modal_evidence + 2 response_evidence + (meta + rows) x 4 cells
        rec = w.evaluate(w.document())
        assert rec["n_carried"] == 1, rec["refused"]
        e = rec["carried"][SID]
        p = e["clauses"]["producer_set"]["petri_sidecar_proof"]
        assert p["kind"] == "generation_only" and p["ruling"] == "D191" and p["ok"] is True and p["cells"] == 4 and p["n_rows"] == 16
        assert p["removed_leg"]["uses"] == 0 and p["added_legs"]["uses"] == 0 and p["served_models"] == ["petri__base", "petri__prompted__safety"] and p["foreign_models"] == []
        assert p["policy"]["removed"] == [REMOVED, "qwen-abl"] and p["policy"]["added"] == [ADDED, "grok-4.6", "openrouter/x-ai/grok-4.6", "opus-4-8"] and p["target_models"] == ["petri__base", "petri__prompted__safety"]
        assert p["policy"]["prev"][F.MANIFEST]["auditor_fallback_models"] == ["openrouter/x-ai/grok-4.6", REMOVED] and p["policy"]["this"][F.MANIFEST]["judge_fallback_models"] == [ADDED, "openrouter/x-ai/grok-4.6"]
        assert set(p["inputs"]) == set(w.expected) and all(v for v in p["inputs"].values()) and p["bytes_read"] > 0
        assert p["judge"] == {"in_attempt": False, "site": "scripts/sdfbase/full22/misalign/misalign_judge.py (the grading host, D8)"}
        assert p["basis"]["chain"]["this"][F.MANIFEST]["judge_fallback_models"] == [ADDED, "openrouter/x-ai/grok-4.6"] and p["basis"]["chain"]["manifests"] == [F.MANIFEST]
        assert len(p["proof_sha256"]) == 64 and "evidence_inventory_local" not in e
        assert all(path in e["evidence_inventory"] for path in w.expected)                            # every proof input digest-bound by the recording reader
        assert w.certify(rec) == []

    def test_without_a_document_the_misalign_row_refuses_by_name(self, tmp_path):
        w = World(tmp_path)
        rec = w.evaluate(None)
        assert rec["n_carried"] == 0
        why = rec["refused"][SID]["why"]
        assert "needs the sidecar proof" in why and f"no D191 proof document at RS/{co.GENERATION_PROOF_REL}/{SID}__{AID}.json" in why
        assert co.carry_refused_holds(rec)[SID]["hold_kind"] == "accepted_attempt"                   # a FULL acceptance: held, never re-spawned


class TestGenerationProofRefusals:
    def test_a_document_saying_ok_while_the_bytes_show_one_removed_leg_use_refuses_the_re_derivation_wins(self, tmp_path):
        w = World(tmp_path, taint=REMOVED)
        doc = w.document()
        assert doc["ok"] is False and doc["removed_leg"]["uses"] == 1 and "REMOVED leg(s) were reached 1 time(s)" in doc["why"]     # the honest document already fails ...
        forged = {**doc, "ok": True, "why": None, "removed_leg": {**doc["removed_leg"], "uses": 0, "per_leg": {k: 0 for k in doc["removed_leg"]["per_leg"]}}}
        rec = w.evaluate(forged)                                                                      # ... a forged one is re-derived
        assert rec["n_carried"] == 0
        why = rec["refused"][SID]["why"]
        assert "re-derived from the volume bytes, 'ok' is not the document's (the re-derivation wins" in why and "'removed_leg' is not the document's" in why

    def test_sol_p1_a_document_whose_removed_list_is_not_the_tree_derived_policy_refuses(self, tmp_path):
        """The real removed model in a response's model_reported; a document regenerated over the same bytes with removed list
        ['unused-synthetic-leg'] used to carry (the document decided what the derivation searched for). Now the lists must equal the trees' policy."""
        w = World(tmp_path, taint=REMOVED)
        bogus = {**w.policy, "removed": ["unused-synthetic-leg"], "removed_models": ["unused-synthetic-leg"]}
        doc = w.document(policy=bogus)
        assert doc["ok"] is True and doc["removed_leg"]["uses"] == 0                                  # over its OWN list the document is 'clean' ...
        rec = w.evaluate(doc)
        assert rec["n_carried"] == 0
        why = rec["refused"][SID]["why"]
        assert f"the document's removed legs ['unused-synthetic-leg'] are not the tree-derived policy ['{REMOVED}', 'qwen-abl']" in why and "refusing" in why, why
        assert "the document's policy block is not the one derived from the trees" in why
        # the same for the added list and the target list
        rec2 = w.evaluate(w.document(policy={**w.policy, "added": ["nothing"]}))
        assert "the document's added legs ['nothing'] are not the tree-derived policy" in rec2["refused"][SID]["why"]
        rec3 = w.evaluate(w.document(policy={**w.policy, "target": ["petri__base"]}))
        assert "the document's target models ['petri__base'] are not the tree-derived policy" in rec3["refused"][SID]["why"]

    def test_sol_p2_an_added_chain_model_observed_in_the_store_fails_the_proof(self, tmp_path):
        w = World(tmp_path, taint=ADDED)
        doc = w.document()
        # one observed `anthropic/claude-opus-4-8` matches TWO added identities (the model id and its tree-derived alias `opus-4-8`): 2 uses, both named
        assert doc["ok"] is False and doc["added_legs"]["uses"] == 2 and doc["added_legs"]["per_leg"][ADDED] == 1 and doc["added_legs"]["per_leg"]["opus-4-8"] == 1
        assert doc["removed_leg"]["uses"] == 0 and "ADDED leg(s) were reached 2 time(s)" in doc["why"]
        rec = w.evaluate(doc)
        assert rec["n_carried"] == 0 and "the re-derived proof does NOT hold" in rec["refused"][SID]["why"] and "ADDED leg(s)" in rec["refused"][SID]["why"]
        forged = {**doc, "ok": True, "why": None, "added_legs": {**doc["added_legs"], "uses": 0, "per_leg": {k: 0 for k in doc["added_legs"]["per_leg"]}}}
        assert "'added_legs' is not the document's" in w.evaluate(forged)["refused"][SID]["why"]

    def test_a_foreign_model_name_in_the_store_refuses_by_name(self, tmp_path):
        w = World(tmp_path)
        k = next(p for p in w.expected if p.endswith("/prompted__safety/meta.json"))
        w.files[k] = json.dumps({**json.loads(w.files[k]), "served_models": [_vm("none"), _vm("prompted__safety"), "petri__stranger"]})
        doc = w.document()
        assert doc["ok"] is False and doc["foreign_models"] == ["petri__stranger"] and "not in the archived row's target set: ['petri__stranger']" in doc["why"]
        rec = w.evaluate(doc)
        assert rec["n_carried"] == 0 and "observed model name(s) are not in the archived row's target set" in rec["refused"][SID]["why"]
        forged = {**doc, "ok": True, "why": None, "foreign_models": []}
        assert "'foreign_models' is not the document's" in w.evaluate(forged)["refused"][SID]["why"]

    def test_a_document_without_bytes_read_or_with_an_unreadable_manifest_refuses(self, tmp_path):
        w = World(tmp_path)
        doc = w.document(); doc.pop("bytes_read")
        assert "records no bytes_read" in w.evaluate(doc)["refused"][SID]["why"]
        # the manifest not readable at the prior tree: no policy, no constant -- refused by name
        nogit = F.FakeGit({W_PREV: {k: v for k, v in F._blobs(W_PREV).items() if k != F.MANIFEST}, W_THIS: F._blobs(W_THIS, **{F.MANIFEST: "manifest-v2"})}, sources={(W_THIS, F.MANIFEST): MANIFEST_V2})
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=_agd(w.dicts), git=nogit,
                                     readiness_problems=F.fake_readiness, generation_proof_doc=lambda s, a: w.document())
        assert rec["n_carried"] == 0                                                                  # (the producer set refuses the absent manifest first; the policy never has a constant to fall back on)
        try:
            co.generation_proof_policy(w.srow, git=nogit, prev_W=W_PREV, this_W=W_THIS)
        except ValueError as e:
            assert "is not readable at the prior tree" in str(e) and "cannot be derived" in str(e)
        else:
            raise AssertionError("a manifest unreadable at one tree must refuse")

    def test_an_input_outside_the_canonical_root_a_missing_input_and_an_extra_input_refuse(self, tmp_path):
        w = World(tmp_path)
        doc = w.document()
        outside = {**doc, "inputs": {**doc["inputs"], "/vol/seedrep/ccccccc/exp/other/modal_evidence.json": "0" * 64}}
        why = w.evaluate(outside)["refused"][SID]["why"]
        assert "lie outside the canonical root" in why and "are not files the archived row's children name" in why
        k = next(p for p in doc["inputs"] if p.endswith("/none/responses.jsonl"))
        missing = {**doc, "inputs": {p: v for p, v in doc["inputs"].items() if p != k}}
        assert "store file(s) the archived row's children name are not among the document's inputs" in w.evaluate(missing)["refused"][SID]["why"]
        extra = {**doc, "inputs": {**doc["inputs"], f"{w.root}/exp/misalign/agentic__s1__murder/stray/meta.json": "0" * 64}}
        assert "are not files the archived row's children name" in w.evaluate(extra)["refused"][SID]["why"]
        # a store file the children name but the volume lacks: re-derivation refuses (never a partial proof)
        gone = dict(w.files); gone.pop(k)
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=gone.get, accepted_get_for_dict=_agd(w.dicts), git=w.git,
                                     readiness_problems=F.fake_readiness, generation_proof_doc=lambda s, a: doc)
        assert "cannot be re-derived from the volume bytes" in rec["refused"][SID]["why"] and "FileNotFoundError" in rec["refused"][SID]["why"]

    def test_a_generation_only_proof_on_a_petri_leg_row_refuses_by_name(self, tmp_path):
        def petri_row(W, W7):
            r = _misalign_row(W, W7, sid="petri__s1__base__shard00", leg="petri")           # a petri-LEG row with misalign-shaped children
            for k in r["children"]:
                k["eval_leg"] = "petri"
            return r
        w = World(tmp_path, leg_row=petri_row)
        doc = w.document()
        docs = {("petri__s1__base__shard00", AID): doc}
        rec = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=_agd(w.dicts), git=w.git,
                                     readiness_problems=F.fake_readiness, generation_proof_doc=lambda sid, aid: docs.get((sid, aid)))
        # the petri LEG never looks for a generation document (the D115 store proof applies): with no store-proof reader it refuses on that
        r = rec["refused"]["petri__s1__base__shard00"]
        assert rec["n_carried"] == 0 and "needs the sidecar proof" in r["why"] and "no D191 proof document" not in r["why"]
        # a Petri store-proof reader that hands back a generation_only record is refused by kind
        rec2 = co.evaluate_carry_over(this_plan=w.this, this_sha=W_THIS, priors=[w.prior], read_text=w.files.get, accepted_get_for_dict=_agd(w.dicts), git=w.git,
                                      readiness_problems=F.fake_readiness, petri_proof=lambda **kw: (kw["read_text"]("/x/results.json"), {**doc, "ok": True})[1])
        assert "only a D191 document over a misalign attempt's own bytes may carry that kind" in rec2["refused"]["petri__s1__base__shard00"]["why"]
        # and at certify: a carried misalign record re-labelled as a petri-leg row's is out of scope
        assert "out of scope" in (co.generation_proof_scope_problem({"leg": "petri"}, source="s5_smoke", evidence={"petri": None}) or "") or \
               "keeps the D115 store proof" in co.generation_proof_scope_problem({"leg": "petri"}, source="s5_smoke", evidence={"petri": None})
        assert "carries a Petri completion bundle" in co.generation_proof_scope_problem({"leg": "misalign"}, source="s5_smoke", evidence={"petri": {"evidence": {}}})
        assert "stands only for a" in co.generation_proof_scope_problem({"leg": "misalign"}, source="production", evidence={"petri": None})

    def test_a_wrong_ruling_label_or_kind_refuses(self, tmp_path):
        w = World(tmp_path)
        assert "ruling label is 'D999', not 'D191'" in w.evaluate(w.document(ruling="D999"))["refused"][SID]["why"]
        assert "kind is 'store', not 'generation_only'" in w.evaluate(w.document(kind="store"))["refused"][SID]["why"]
        assert "schema is 'x'" in w.evaluate(w.document(schema="x"))["refused"][SID]["why"]

    def test_the_certificate_refuses_when_the_volume_bytes_differ_from_the_recorded_digests_or_the_record_is_edited(self, tmp_path):
        w = World(tmp_path)
        rec = w.evaluate(w.document())
        assert rec["n_carried"] == 1 and w.certify(rec) == []
        # the volume bytes change under the canonical root: the re-derivation's inputs / bytes differ from the record -> refused by name (both layers)
        k = next(p for p in w.expected if p.endswith("/prompted__safety/meta.json"))
        changed = dict(w.files); changed[k] = json.dumps({**json.loads(w.files[k]), "served_models": [_vm("prompted__safety"), REMOVED]})
        probs = w.certify(rec, read_text=changed.get)
        assert any("generation_only proof (D191)" in p and "'inputs' is not the document's" in p for p in probs), probs
        assert any("do not reproduce the recorded proof; refusing" in p for p in probs), probs
        assert any("changed since the evaluation" in p for p in probs)                              # the recording reader's inventory catches it too
        # the record edited: a removed-leg use hidden, the ruling relabelled, an input dropped
        edited = copy.deepcopy(rec); pr = edited["carried"][SID]["clauses"]["producer_set"]["petri_sidecar_proof"]
        pr["removed_leg"] = {**pr["removed_leg"], "uses": 3}
        assert any("'removed_leg' is not the document's" in p for p in w.certify(edited)), w.certify(edited)
        relabelled = copy.deepcopy(rec); relabelled["carried"][SID]["clauses"]["producer_set"]["petri_sidecar_proof"]["ruling"] = "D183"
        assert any("ruling label is 'D183', not 'D191'; refusing" in p for p in w.certify(relabelled))
        dropped = copy.deepcopy(rec); pr = dropped["carried"][SID]["clauses"]["producer_set"]["petri_sidecar_proof"]; pr["inputs"] = {p: v for p, v in pr["inputs"].items() if not p.endswith("/none/meta.json")}
        assert any("are not among the document's inputs" in p for p in w.certify(dropped))
        stale_bytes = copy.deepcopy(rec); stale_bytes["carried"][SID]["clauses"]["producer_set"]["petri_sidecar_proof"]["bytes_read"] = 1
        assert any("'bytes_read' is not the document's" in p for p in w.certify(stale_bytes))
        # a petri-LEG entry carrying a generation_only proof at certify: out of scope by name
        petri = copy.deepcopy(rec)
        petri["carried"][SID]["clauses"]["producer_set"]["petri_sidecar_proof"]["kind"] = co.GENERATION_PROOF_KIND
        assert w.certify(petri) == []                                                                 # (the same record; the scope check reads the ARCHIVED row's leg, which is misalign)


PULLED = Path("/tmp/launch_premint/fj29_misalign_w22/pulled")
REAL_ROOT = "/vol/seedrep/56aff57/exp/seedrep_battery_modal_smoke/S5/misalign/s1/shard00/attempts/01M2XD1H6W9CT4XFNSTZ7VED8R-b4cbfce42621"
RS = host_site_path("$STATE/romp/candidate-si-refit/run-state/full22/seedrep_battery")


class TestTheRealW22MisalignDocument:
    """The real misalign proof document (tests/fixtures/seedrep_battery/misalign_w22_generation_proof.json, digests pinned from the fj29 pull)
    admits under a reader serving those bytes; local to the grading host (skipped where the pull or the archive is absent)."""

    def test_the_pinned_document_re_derives_over_the_pulled_bytes_and_admits_for_the_archived_w22_row(self):
        if not (PULLED / REAL_ROOT[5:] / "modal_evidence.json").is_file() or not (RS / "prior" / "56aff57" / "launch_plan_draft.json").is_file():
            pytest.skip("the fj29 pull / the W22 archive are not on this machine")
        doc = json.loads((REPO / "tests" / "fixtures" / "seedrep_battery" / "misalign_w22_generation_proof.json").read_text())
        assert doc["kind"] == "generation_only" and doc["ruling"] == "D191" and len(doc["inputs"]) == 35 and doc["ok"] is True and doc["bytes_read"] == 87303616
        assert doc["policy"]["removed"] == ["openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated", "qwen-abl"] and doc["policy"]["added"] == ["anthropic/claude-opus-4-8", "grok-4.6", "openrouter/x-ai/grok-4.6", "opus-4-8"]
        assert len(doc["target_models"]) == 9 and doc["foreign_models"] == []
        plan = json.loads((RS / "prior" / "56aff57" / "launch_plan_draft.json").read_text())
        prow = next(r for r in plan["standups"] if r["standup_id"] == SID)
        srow = lp.smoke_row(prow, lp.smoke_identity(plan, "S5", "42c2793a-8160-462c-86e6-e39687cd6b13"), max_children=len(prow["children"]), max_timeout_s=int(prow["modal"]["timeout_s"]))
        reader = mgp.local_reader(PULLED / REAL_ROOT[5:], volume_root=REAL_ROOT)
        ev = json.loads(reader(f"{REAL_ROOT}/modal_evidence.json"))
        assert ev["attempt_id"] == "01M2XD1H6W9CT4XFHNSTZ7VED8R-b4cbfce42621".replace("XFHN", "XFN") and ev.get("petri") is None
        git = co.default_git(REPO)
        record, probs = co.generation_proof_record(doc, prev_row=srow, source="s5_smoke", root=REAL_ROOT, evidence=ev, read_text=reader, git=git,
                                                   prev_W="56aff576e2975804602aec7d12ba1c98c805adfd", this_W="0f00f7fae98e8c719617f6d01bbdc4eb2c763c88")
        assert probs == [] and record is not None, probs
        assert record["ok"] is True and record["cells"] == 16 and record["n_rows"] == 8000 and record["removed_leg"]["uses"] == 0 and record["bytes_read"] == 87303616
        assert set(record["inputs"]) == set(doc["inputs"]) and all(record["inputs"][k] == v for k, v in doc["inputs"].items())
        # the document forged to ok while a single pulled byte-stream shows a use: the re-derivation wins
        tainted_reader = lambda p: (reader(p) or "").replace("petri__base", "openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated", 1) if p.endswith("/none/meta.json") else reader(p)  # noqa: E731
        record2, probs2 = co.generation_proof_record(doc, prev_row=srow, source="s5_smoke", root=REAL_ROOT, evidence=ev, read_text=tainted_reader, git=git,
                                                     prev_W="56aff576e2975804602aec7d12ba1c98c805adfd", this_W="0f00f7fae98e8c719617f6d01bbdc4eb2c763c88")
        assert record2 is None and any("'inputs' is not the document's" in p for p in probs2) and any("'removed_leg' is not the document's" in p for p in probs2)
