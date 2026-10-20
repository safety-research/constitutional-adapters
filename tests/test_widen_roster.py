"""bcb-bfcl-200-widen S3: the lane roster passes its OWN rules against the checked-in
pins and ``contracts.MODELS``; every rule is pinned against a synthetic bad roster; the
readers return the lane's cells / legs / lanes / generations; ``glm46vflash`` is refused
anywhere but ``excluded_families``; the launch-time snapshot recomputes and refuses a
relabelled sidecar."""
from __future__ import annotations

import copy
import json

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import widen_roster as W

ROSTER = W.load_roster()
PINS = W.load_pins()
DERIVED = "constv2-mix-full__lora_r4_ce_minus_ctl"


def _bad(mutate):
    r = copy.deepcopy(ROSTER)
    mutate(r)
    return W.validate(r, PINS, C.MODELS)


def test_the_qwen_it_subtraction_is_held_out_of_pooled_means_by_declaration():
    """The qwen35_4b it LoRA subtraction — 0/200 on BigCodeBench, every reply a
    refusal — is REPORTED, held OUT of pooled category means; the declaration is the entry's `pooling` block and the readers
    return it by cell. The note says REFUSAL, quotes an opening verbatim and names the confound."""
    held = W.held_out_cells(ROSTER, "qwen35_4b_it")
    assert list(held) == [DERIVED] and held[DERIVED]["held_out"] is True and "the coordinator" in held[DERIVED]["ruled_by"]
    reason = held[DERIVED]["reason"]
    for needle in ("refuses every item", "200/200", "I can't help with creating code designed to bypass security controls", "CONFOUND", "OR-Bench", "GSM8K",
                   "2.20", "3.89"):
        assert needle in reason, needle
    assert "collapse" not in reason.lower()
    assert W.pooling_of(ROSTER, "qwen35_4b_it", "diff_lora") == held[DERIVED] and W.pooling_of(ROSTER, "qwen35_4b_it", "none") is None
    assert W.held_out_cells(ROSTER, "qwen35_4b_main") == {}                     # the balanced base-lane pair pools
    with pytest.raises(W.RosterError, match="no entry"):
        W.pooling_of(ROSTER, "qwen35_4b_it", "nope")


def test_the_checked_in_roster_passes_its_own_validation():
    assert W.validate(ROSTER, PINS, C.MODELS) == []
    W.validate_or_raise(ROSTER, PINS, C.MODELS)
    assert W.waves(ROSTER) == ["gemma4e4b_it", "gemma4e4b_main", "qwen35_4b_main", "qwen35_4b_it",
                               "nemotron3nano4b_main", "llama8b_it", "llama8b_main",
                               # the nine batch-64 three-seed Qwen waves
                               "qwen35_4b_s1", "qwen35_4b_s1_arditi", "qwen35_4b_s1p",
                               "qwen35_4b_s2", "qwen35_4b_s2_arditi", "qwen35_4b_s2p",
                               "qwen35_4b_s3", "qwen35_4b_s3_arditi", "qwen35_4b_s3p",
                               # b64-capability-agentic-queue (2026-09-23): the six all-models one-seed waves (gemma / llama / nemotron x s1 / s1p)
                               "gemma4e4b_s1", "gemma4e4b_s1p", "llama8b_s1", "llama8b_s1p",
                               "nemotron3nano4b_s1", "nemotron3nano4b_s1p"]
    assert ROSTER["excluded_families"] == ["glm46vflash"] and W.glm_problems(ROSTER) == []
    # the header declares the lane's serving path and the reader holds it to its ONE constant
    assert W.serving_path(ROSTER) == W.SERVING_PATH == C.SERVING_ENGINE_LORA_NATIVE == "lora_native" and W.SERVING_PATH_KEY == "serving_path"
    assert (W.ROSTER_PATH).is_file() and W.roster_sha256() == W.roster_sha256(W.ROSTER_PATH)


def test_readers_return_the_lanes_cells_legs_and_lanes():
    # PSR is OUT by default — both PSR cells are BLOCKED, their object blocks
    # (the pins) kept so the entries stay identifiable; the constitution SI is IN as a runnable prompted cell of the it wave.
    # The batch-64 control LoRA and LoRA subtraction are RUNNABLE beside the
    # batch-8 constitution LoRA (a MIXED SET — the constitution arm is not re-fit for this lane; every row stamps its generation)
    assert W.runnable_cells(ROSTER, "gemma4e4b_it") == ["none", "prompted__safety_refusal", "prompted__si_constitution",
                                                        "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", DERIVED]
    assert W.object_generation(ROSTER, "gemma4e4b_it", "ctl_lora") == ("batch64", None) == W.object_generation(ROSTER, "gemma4e4b_it", "diff_lora")
    assert "ruling (b)" in W.entries(ROSTER, "gemma4e4b_it")["const_lora"]["rerun_trigger"]      # the batch-8 cell stays, by ruling
    assert W.runnable_cells(ROSTER, "gemma4e4b_main") == ["none", DERIVED]
    for wname, eid, cell in (("gemma4e4b_it", "const_psr", "constv2-mix-full__psr_ce_all"),
                             ("gemma4e4b_main", "diff_psr", "constv2-mix-full__psr_ce_all_minus_ctl_distilled")):
        e = W.entries(ROSTER, wname)[eid]
        assert e["status"] == "blocked" and e["cell_id"] == cell and e["reason"].startswith("ruling of 2026-09-11") and "PSR is out of the paper by default" in e["reason"]
        assert e["unblock"] == "a ruling reinstates PSR" and W.pin_of(ROSTER, wname, eid) == PINS[W.wave(ROSTER, wname)["family"]][e["object"]["staged_id"]]["object_sha256"]
        assert W.queued_or_blocked(ROSTER, wname)[eid]["status"] == "blocked"
    si = W.entries(ROSTER, "gemma4e4b_it")["const_si"]
    assert (si["cell_id"], si["status"], si["paper_arm"], si["kind"]) == ("prompted__si_constitution", "runnable", "si_constitution", "prompted")
    assert "object" not in si and C.parse_launcher_cell(si["cell_id"]).kind == C.CELL_KIND_PROMPTED and W.pin_of(ROSTER, "gemma4e4b_it", "const_si") is None
    assert W.runnable_cells(ROSTER, "qwen35_4b_main") == ["none", DERIVED]
    assert W.runnable_cells(ROSTER, "nemotron3nano4b_main") == ["none", DERIVED]
    assert W.runnable_cells(ROSTER, "qwen35_4b_it") == ["none", DERIVED]      # the batch-64 LoRA subtraction, now servable
    # the two llama8b waves are RUNNABLE (none + the LoRA subtraction); the it wave is the paper's
    # instruct-direct llama lane, the main wave its base->instruct transfer (the qwen shape).
    # Their BFCL leg is BLOCKED — the pinned Instruct template rendered no tool definitions (both cells
    # 56/200 = exactly the irrelevance items), an instrument failure; the analysis WITHDRAWS the measured BFCL leg
    # (the eval is SKIPPED for Llama — the withdrawal is final, there is no unblock)
    assert W.runnable_cells(ROSTER, "llama8b_it") == ["none", DERIVED] == W.runnable_cells(ROSTER, "llama8b_main")
    # none FIRST even when the roster lists it elsewhere
    r = copy.deepcopy(ROSTER)
    ents = r["waves"]["qwen35_4b_main"]["entries"]
    r["waves"]["qwen35_4b_main"]["entries"] = {"diff_lora": ents["diff_lora"], "none": ents["none"]}
    assert W.runnable_cells(r, "qwen35_4b_main") == ["none", DERIVED]
    assert W.runnable_legs(ROSTER, "gemma4e4b_it") == ["bigcodebench_200", "bfcl_200"]
    assert W.runnable_legs(ROSTER, "qwen35_4b_main") == ["bigcodebench_200", "bfcl_200"]      # the live probe PASSED (80df464)
    # nemotron_h's live probe PASSED at 8365992 (2026-09-23): the nemotron BFCL legs run too
    assert W.runnable_legs(ROSTER, "nemotron3nano4b_main") == ["bigcodebench_200", "bfcl_200"]
    assert W.runnable_legs(ROSTER, "llama8b_it") == ["bigcodebench_200"] == W.runnable_legs(ROSTER, "llama8b_main")
    for wname in ("llama8b_it", "llama8b_main"):
        st, detail = W.legs(ROSTER, wname)["bfcl_200"]
        assert st == "blocked" and detail["reason"].startswith("instrument failure:") and "b48c47f644389271" in detail["reason"]
        # the rule 2026-09-13: the function-calling eval is SKIPPED for Llama — no unblock path, the rows are final
        assert detail["unblock"].startswith("NONE — SKIPPED FOR LLAMA by the ruling of 2026-09-13") and "no function calling" in detail["unblock"]
    assert W.paper_lane(ROSTER, "gemma4e4b_it") == {"family": "gemma4e4b", "lane_kind": "instruct-direct", "view": "gemma4e4b_it"}
    assert W.paper_lane(ROSTER, "nemotron3nano4b_main")["lane_kind"] == "chat-ckpt"
    assert W.paper_lane(ROSTER, "llama8b_it") == {"family": "llama8b", "lane_kind": "instruct-direct", "view": "llama8b_it"}
    assert W.paper_lane(ROSTER, "llama8b_main") == {"family": "llama8b", "lane_kind": "base->instruct", "view": "llama8b"}
    for wname, slug, sid, sha in (("llama8b_it", "llama8b-f22it", "derived_itdoc__" + DERIVED,
                                   "e85220c9e5d2b9aa4377fea7a9a590048c693118975be9f36ae579b0eeff6f13"),
                                  ("llama8b_main", "llama8b-f22", "derived__" + DERIVED,
                                   "4375b67a0fb0621e9ee296041d0842cb24294112e18424a803e8e54e9538a740")):
        w = W.wave(ROSTER, wname)
        assert w["model_row"] == slug and w["objects_dir"] == C.model_row(slug).objects_dir
        assert w["entries"]["diff_lora"]["object"]["staged_id"] == sid and W.pin_of(ROSTER, wname, "diff_lora") == sha
        assert W.object_generation(ROSTER, wname, "diff_lora") == ("final", None)
        assert "16 epochs" in w["entries"]["diff_lora"]["note"]              # the Llama lane's caveat rides with the entry
        # the derived-provenance preflight reads the parents at /vol/out: staged only once the orchestrator records it
        # state-independent: the shipped roster's `prerequisites_met` moves as the orchestrator records the parents'
        # staging — the DECLARED prerequisite is what is pinned; the unmet face on a copy
        (declared,) = w["prerequisites"]
        assert declared.startswith("S4: the derived object's two parents staged at /vol/out/")
        assert w["prerequisites_met"] in ([], [declared])
        unmet = copy.deepcopy(ROSTER); unmet["waves"][wname]["prerequisites_met"] = []
        assert W.prerequisites(unmet, wname) == [declared]
        assert ("llama8b_it_doc" if wname == "llama8b_it" else "llama8b_base") in declared
    assert W.object_generation(ROSTER, "gemma4e4b_it", "const_lora") == ("batch8-provisional", "gemma4e4b_it_doc_eb64")
    assert W.object_generation(ROSTER, "gemma4e4b_main", "diff_lora") == ("final", None)
    assert W.object_generation(ROSTER, "gemma4e4b_it", "none") == (None, None) == W.object_generation(ROSTER, "gemma4e4b_it", "si_safety")
    qb = W.queued_or_blocked(ROSTER, "gemma4e4b_it")
    assert set(qb) == {"const_psr", "ctl_psr", "diff_psr", "safety_lora", "safety_psr", "alphasteer_as_fit"}
    # EVERY PSR entry is blocked (none is queued behind a fit batch); the LoRA / matrix ones stay queued
    for eid in ("const_psr", "ctl_psr", "diff_psr", "safety_psr"):
        assert qb[eid]["status"] == "blocked" and qb[eid]["reason"].startswith("ruling of 2026-09-11") and "PSR is out" in qb[eid]["reason"], eid
        assert qb[eid]["unblock"].startswith("a ruling reinstates PSR")
    assert all(qb[eid]["status"] == "queued" for eid in ("safety_lora", "alphasteer_as_fit"))
    assert qb["alphasteer_as_fit"]["cell_id"] is None and qb["alphasteer_as_fit"]["status"] == "queued"
    assert all(q["reason"] and q["unblock"] for q in qb.values())
    assert ROSTER["waves"]["qwen35_4b_main"]["prerequisites"] == ["S6e: qwen35_4b geometry budget MEASURED on this branch (P2-9)"]
    assert W.prerequisites(ROSTER, "qwen35_4b_main") in ([], ROSTER["waves"]["qwen35_4b_main"]["prerequisites"])
    assert W.prerequisites(ROSTER, "gemma4e4b_it") == []
    assert W.pins_by_cell(ROSTER, "gemma4e4b_it") == {          # runnable OBJECT entries only: the blocked PSR cell's pin is not launched
        "constv2-mix-full__lora_r4_ce": "4c1a5fc3b382ddb86f9dce20ef7d31dac735b516014017d19fad0c74c0092353",
        # the batch-64 pair beside the batch-8 constitution LoRA — a mixed set
        "constv2ctl2-full__lora_r4_ce": "04a90de8edd863345bf182adad717801b391096b350253c72bcb077b97885ed8",
        DERIVED: "8b7d10301a1fd04fbe4c71f21ae8912bae60ffc56df34acd10fa07a960b2080e"}
    assert W.pin_of(ROSTER, "gemma4e4b_it", "const_psr") == "fdab3ade9b0d8d627ce2519b0fa72552fd3fa44537d8ea12e6f9727f11808b5b"
    assert W.pin_of(ROSTER, "nemotron3nano4b_main", "diff_lora") == "27bab56e9fcea25e397b8d5b918c91a03a42fc279561894a01189713370bec41"
    assert W.pin_of(ROSTER, "nemotron3nano4b_main", "none") is None
    assert W.entry_of_cell(ROSTER, "gemma4e4b_main", DERIVED) == "diff_lora"
    with pytest.raises(W.RosterError):
        W.entry_of_cell(ROSTER, "gemma4e4b_main", "nope")
    # every runnable object sha equals the pins entry (the rule the validator enforces)
    for wname in W.waves(ROSTER):
        fam = W.wave(ROSTER, wname)["family"]
        for _, _, obj in W.runnable_entries(ROSTER, wname):
            if obj:
                assert PINS[fam][obj["staged_id"]]["object_sha256"] == obj["sha256"]


@pytest.mark.parametrize("needle, mutate", [
    # a header naming the lens (or no serving path at all) refuses by name — the roster is what a reader of the wave sees
    ("serving_path 'lens' != 'lora_native'", lambda r: r.__setitem__("serving_path", "lens")),
    ("serving_path None != 'lora_native'", lambda r: r.pop("serving_path")),
    ("names the excluded family", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["none"].__setitem__("note", "glm46vflash too")),
    ("names the excluded family", lambda r: r["waves"].__setitem__("glm46vflash_main", r["waves"]["qwen35_4b_it"])),
    ("!= the pins entry", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["object"].__setitem__("sha256", "0" * 64)),
    ("not a sha256 hex", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["object"].__setitem__("sha256", "abc")),
    ("no pins entry", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["object"].__setitem__("staged_id", "itdoc__nope")),
    ("needs a non-empty unblock", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["safety_lora"].pop("unblock")),
    ("needs a non-empty reason", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["safety_lora"].__setitem__("reason", "  ")),
    ("needs a runnable `none`", lambda r: r["waves"]["qwen35_4b_main"]["entries"]["none"].update(status="queued", reason="x", unblock="y")),
    ("document-corpus pseudo-instruction", lambda r: r["waves"]["gemma4e4b_it"]["entries"].__setitem__(
        "bad", {"cell_id": "prompted__constv2-mix-full", "status": "runnable"})),
    ("needs a non-null cell_id", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["alphasteer_as_fit"].__setitem__("status", "runnable")),
    ("names its superseded_by_key", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["const_lora"]["object"].pop("superseded_by_key")),
    ("names its rerun_trigger", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["const_lora"].pop("rerun_trigger")),
    ("not in ['final', 'batch8-provisional', 'batch64']", lambda r: r["waves"]["gemma4e4b_main"]["entries"]["diff_lora"]["object"].__setitem__("object_generation", "batch128")),   # batch64 is a real lineage
    # every lane row carries a manifest since nemotron_h's PASS (2026-09-23) — point a BFCL-running wave at a legacy manifest-less
    # row (qwen35_9b, LEGACY_MANIFESTLESS_ROWS): the served / objects_dir disagreements are named too, and so is the manifest
    ("has no petri_manifest", lambda r: r["waves"]["nemotron3nano4b_main"].__setitem__("model_row", "qwen35_9b")),
    ("!= row 'qwen35_4b-f22'", lambda r: r["waves"]["qwen35_4b_main"].__setitem__("objects_dir", "/vol/objects/scale-12/qwen35_4b")),
    ("!= row 'gemma4e4b-f22it'", lambda r: r["waves"]["gemma4e4b_it"]["served"].__setitem__("rev", "0" * 40)),
    ("is not a registry slug", lambda r: r["waves"]["gemma4e4b_it"].__setitem__("model_row", "gemma4e4b-nope")),
    ("names a model_row", lambda r: r["waves"]["gemma4e4b_it"].__setitem__("model_row", None)),
    ("status 'pending' not in", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["safety_lora"].__setitem__("status", "pending")),
    ("a queued leg needs a non-empty unblock", lambda r: r["waves"]["qwen35_4b_main"]["legs"].__setitem__("bfcl_200", {"status": "queued", "reason": "x"})),   # every qwen leg is runnable
    ("a blocked leg needs a non-empty unblock", lambda r: r["waves"]["llama8b_it"]["legs"]["bfcl_200"].pop("unblock")),
    ("prerequisites_met names", lambda r: r["waves"]["qwen35_4b_main"].__setitem__("prerequisites_met", ["something else"])),
    ("must not carry an `object` block", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["none"].__setitem__("object", {"sha256": "0" * 64})),
    ("is also entry", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["ctl_psr"].__setitem__("cell_id", "constv2ctl2-full__lora_r4_ce")),
    ("is not 16 hex", lambda r: r["waves"]["gemma4e4b_main"]["entries"]["diff_lora"].__setitem__("intervention_hash_expected", "zz")),
    ("sampling_clamp_override", lambda r: r.__setitem__("sampling_clamp_override", {"from": 0.01, "to": 0.01, "why": "x"})),
    ("contracts.SAMPLING_CLAMP_OVERRIDE", lambda r: r.__setitem__("sampling_clamp_override", {"from": 0.02, "to": 0.001, "why": "x"})),
    # the duplicate-cell rule keys on the SAME spelling it stores (code review: raw value vs str)
    ("is also entry", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["ctl_psr"].__setitem__("cell_id", "constv2ctl2-full__lora_r4_ce")
     or r["waves"]["gemma4e4b_it"]["entries"]["ctl_lora"].__setitem__("cell_id", "constv2ctl2-full__lora_r4_ce")),
    ("!= 'protocol'", lambda r: r.__setitem__("sampling", "card")),
    ("excluded_families", lambda r: r.__setitem__("excluded_families", [])),
    # the pooling declaration: exact keys, a bool, a reason + ruling when held out, runnable, never `none`
    ("pooling must be {held_out: bool, ruled_by, reason}", lambda r: r["waves"]["qwen35_4b_it"]["entries"]["diff_lora"]["pooling"].pop("reason")),
    ("pooling must be {held_out: bool, ruled_by, reason}", lambda r: r["waves"]["qwen35_4b_it"]["entries"]["diff_lora"]["pooling"].__setitem__("held_out", "yes")),
    ("needs a non-empty pooling.reason", lambda r: r["waves"]["qwen35_4b_it"]["entries"]["diff_lora"]["pooling"].__setitem__("reason", " ")),
    ("needs a non-empty pooling.ruled_by", lambda r: r["waves"]["qwen35_4b_it"]["entries"]["diff_lora"]["pooling"].__setitem__("ruled_by", None)),
    ("pooling on a queued entry", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["safety_lora"].__setitem__(       # still queued (ctl_lora runs since #107)
        "pooling", {"held_out": True, "ruled_by": "x", "reason": "y"})),
    ("pooling on the `none` anchor", lambda r: r["waves"]["gemma4e4b_it"]["entries"]["none"].__setitem__(
        "pooling", {"held_out": True, "ruled_by": "x", "reason": "y"})),
])
def test_every_rule_refuses_a_synthetic_bad_roster(needle, mutate):
    problems = _bad(mutate)
    assert problems and any(needle in p for p in problems), (needle, problems)


def test_a_runnable_qwen_bfcl_leg_refuses_when_the_live_record_is_withdrawn(monkeypatch):
    """The second gate, still exercised now that qwen3_5's live PASS is filed (2026-09-23, staged tree
    80df464): with the record withdrawn, the shipped roster's runnable qwen bfcl_200 legs refuse by name; with it, the roster passes."""
    assert W.validate(ROSTER, PINS, C.MODELS) == []
    monkeypatch.setitem(C.TOOL_PARSER_LIVE_PASS, "qwen3_5", None)
    problems = _bad(lambda r: None)
    assert problems and any("bfcl_200 is runnable but family 'qwen3_5': the S5 tool-parser probe's LIVE half has not PASSED" in p for p in problems), problems


def test_an_unmet_prerequisite_is_what_the_launcher_refuses_on_and_recording_it_clears_it():
    # state-independent: the shipped roster's `prerequisites_met` moves as the orchestrator records
    # prerequisites (the Nemotron parents are recorded), so the test clears the record first
    r = copy.deepcopy(ROSTER)
    wave = r["waves"]["nemotron3nano4b_main"]
    assert wave["prerequisites"], "the wave under test must declare a prerequisite"
    wave["prerequisites_met"] = []
    assert W.prerequisites(r, "nemotron3nano4b_main") == list(wave["prerequisites"])
    wave["prerequisites_met"] = list(wave["prerequisites"])
    assert W.prerequisites(r, "nemotron3nano4b_main") == [] and W.validate(r, PINS, C.MODELS) == []
    # a recorded string that is not a declared prerequisite is a roster defect, not a met prerequisite
    wave["prerequisites_met"] = ["something else"]
    assert any("not a prerequisite" in p for p in W.validate(r, PINS, C.MODELS))


def test_the_launch_time_snapshot_recomputes_and_a_relabelled_sidecar_refuses(tmp_path):
    sha = W.roster_sha256()
    fitted = W.snapshot(ROSTER, "gemma4e4b_it", "const_lora", roster_sha256=sha, launched_at="2026-09-11T00:00:00Z")
    assert tuple(fitted) == W.SNAPSHOT_FIELDS and W.snapshot_problems(fitted) == []
    assert fitted["object_generation"] == "batch8-provisional" and fitted["superseded_by_key"] == "gemma4e4b_it_doc_eb64"
    assert fitted["object_sha256"] == "4c1a5fc3b382ddb86f9dce20ef7d31dac735b516014017d19fad0c74c0092353"
    base = W.snapshot(ROSTER, "gemma4e4b_it", "none", roster_sha256=sha, launched_at="2026-09-11T00:00:00Z")
    assert base["object_generation"] is None and base["object_sha256"] is None and W.snapshot_problems(base) == []
    relabelled = dict(fitted, object_generation="final", superseded_by_key=None)
    assert any("does not recompute" in p for p in W.snapshot_problems(relabelled))
    stale_sha = dict(fitted, object_sha256="0" * 64)
    assert any("does not recompute" in p for p in W.snapshot_problems(stale_sha))
    half = dict(base, object_sha256="0" * 64)
    half["snapshot_sha256"] = W.snapshot_digest(half)
    assert any("both null" in p for p in W.snapshot_problems(half))
    with pytest.raises(W.RosterError, match="not runnable"):
        W.snapshot(ROSTER, "gemma4e4b_it", "safety_lora", roster_sha256=sha, launched_at="t")
    # the launcher's argument: inline JSON or a path, keyed by cell id, every block validated
    doc = {fitted["cell_id"]: fitted, base["cell_id"]: base}
    assert W.parse_snapshot_arg(json.dumps(doc)) == doc
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(doc))
    assert W.parse_snapshot_arg(str(p)) == doc and W.parse_snapshot_arg("") == {}
    with pytest.raises(W.RosterError, match="does not recompute"):
        W.parse_snapshot_arg(json.dumps({fitted["cell_id"]: relabelled}))
    with pytest.raises(W.RosterError, match="names cell_id"):
        W.parse_snapshot_arg(json.dumps({"other": fitted}))
    assert W.SNAPSHOT_FILENAME == "lane_roster.json"


def test_the_filed_snapshot_is_immutable_and_only_the_launch_time_may_move(tmp_path):
    """Code review: a store root's `lane_roster.json` is written once. A relaunch
    is COMPATIBLE iff every content field but `launched_at` equals the filed block (the retry
    goes to `lane_roster.retries.jsonl`, never over the original); any other difference — a
    relabelled generation, another object sha, another roster sha, another entry — is a
    conflict, named per field, and both blocks must be valid snapshots to compare at all."""
    sha = W.roster_sha256()
    filed = W.snapshot(ROSTER, "gemma4e4b_it", "const_lora", roster_sha256=sha, launched_at="2026-09-11T00:00:00Z")
    retry = W.snapshot(ROSTER, "gemma4e4b_it", "const_lora", roster_sha256=sha, launched_at="2026-09-12T09:00:00Z")
    assert retry != filed and W.snapshot_conflicts(filed, retry) == []
    assert W.SNAPSHOT_IMMUTABLE_FIELDS == ("roster_sha256", "wave", "entry_id", "cell_id", "object_generation",
                                           "superseded_by_key", "object_sha256")
    assert W.SNAPSHOT_RETRY_FILENAME == "lane_roster.retries.jsonl"
    relabelled = dict(retry, object_generation="final", superseded_by_key=None)
    relabelled["snapshot_sha256"] = W.snapshot_digest(relabelled)             # recomputed: a "valid" block, another label
    (why,) = [p for p in W.snapshot_conflicts(filed, relabelled) if p.startswith("object_generation")]
    assert "filed 'batch8-provisional'" in why and "'final'" in why
    assert any(p.startswith("superseded_by_key") for p in W.snapshot_conflicts(filed, relabelled))
    other_roster = dict(retry, roster_sha256="0" * 64)
    other_roster["snapshot_sha256"] = W.snapshot_digest(other_roster)
    assert [p.split(":")[0] for p in W.snapshot_conflicts(filed, other_roster)] == ["roster_sha256"]
    other_object = dict(retry, object_sha256="1" * 64)
    other_object["snapshot_sha256"] = W.snapshot_digest(other_object)
    assert [p.split(":")[0] for p in W.snapshot_conflicts(filed, other_object)] == ["object_sha256"]
    # an invalid block on either side is named as such before any field compare
    assert any(p.startswith("incoming snapshot:") for p in W.snapshot_conflicts(filed, dict(retry, snapshot_sha256="0" * 64)))
    assert any(p.startswith("existing record:") for p in W.snapshot_conflicts({"x": 1}, retry))


def test_the_roster_file_itself_names_glm_only_in_excluded_families():
    text = W.ROSTER_PATH.read_text()
    lines = [ln for ln in text.splitlines() if "glm" in ln.lower()]
    assert lines == [ln for ln in lines if ln.lstrip().startswith("excluded_families:")], lines
    doc = yaml.safe_load(text)
    assert doc == ROSTER


SEED_ARMS = ["none", "prompted__safety_refusal", "prompted__si_constitution", "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all",
             "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", DERIVED]
PAIRED = "constv2-mix-full__psr_ce_all_minus_ctl"


def test_the_nine_b64_seed_waves_read_as_planned():
    """Per seed s: the eight-cell seed wave on row
    qwen35_4b-f22s<s> (the ReLU arms are IN; every object pinned to the batch-64 lineage, generation batch64), the
    Arditi wave on the SAME row (QUEUED whole: the preflight cannot derive the served-only Class B arm's hash;
    its entries carry NO object block, so the inventory test asks no hash of it), and the paired-write wave on its own
    row (the composed psr_paired_write cell; a runnable `none` the roster requires, a FREE replicate never the anchor).
    BFCL is RUNNABLE on every qwen wave (2026-09-23: the LIVE half of the tool-parser probe PASSED on the staged tree 80df464
    and its verdict is checked in); the bank + parents + Arditi-extra staging prerequisites are DECLARED and RECORDED met."""
    for s_ in ("1", "2", "3"):
        seed, arditi, paired = f"qwen35_4b_s{s_}", f"qwen35_4b_s{s_}_arditi", f"qwen35_4b_s{s_}p"
        assert W.runnable_cells(ROSTER, seed) == SEED_ARMS
        assert W.runnable_legs(ROSTER, seed) == ["bigcodebench_200", "bfcl_200"] == W.runnable_legs(ROSTER, paired)
        w = W.wave(ROSTER, seed)
        assert (w["model_row"], w["objects_dir"], w["view"], w["lane_kind"]) == (f"qwen35_4b-f22s{s_}", f"/vol/objects/sdfbase/full22/qwen35_4b/s{s_}", f"qwen35_4b_s{s_}", "base->instruct")
        assert w["objects_dir"] == C.model_row(w["model_row"]).objects_dir == C.full22_objects_dir("qwen35_4b", f"s{s_}")
        pins = W.pins_by_cell(ROSTER, seed)
        assert set(pins) == set(SEED_ARMS) - {"none", "prompted__safety_refusal", "prompted__si_constitution"}
        for cell, sha in pins.items():
            eid = W.entry_of_cell(ROSTER, seed, cell)
            obj = W.entries(ROSTER, seed)[eid]["object"]
            prefix = "derived_base64s" if cell == DERIVED else "base64s"
            assert obj["staged_id"] == f"{prefix}{s_}__{cell}" and PINS["qwen35_4b"][obj["staged_id"]]["object_sha256"] == sha
            assert W.object_generation(ROSTER, seed, eid) == ("batch64", None)
            assert obj["fit"]["model_id"] == "Qwen/Qwen3.5-4B-Base" and obj["fit"]["revision"].startswith("1001bb4d")
            assert obj["kind"] == {"psr_ce_all": "psr", "lora_r4_ce": "lora", "lora_r4_ce_minus_ctl": "lora_subtraction"}[cell.split("__")[1]]
        # every hash the seed row's fitted cells need is pinned (the inventory test cross-checks the set)
        assert set(pins) <= set(C.EXPECTED_INTERVENTION_HASHES[w["model_row"]])
        # the two prompted arms are runnable prompted entries, object-free
        for eid, cell in (("si_safety", "prompted__safety_refusal"), ("const_si", "prompted__si_constitution")):
            e = W.entries(ROSTER, seed)[eid]
            assert (e["cell_id"], e["status"], e["kind"]) == (cell, "runnable", "prompted") and "object" not in e
        # BFCL: RUNNABLE on every qwen wave of the family (and on the two earlier qwen waves) — the LIVE half of the
        # tool-parser probe PASSED on the staged tree 80df464 and its verdict is checked in (contracts.TOOL_PARSER_LIVE_PASS["qwen3_5"])
        for wname in (seed, arditi, paired, "qwen35_4b_main", "qwen35_4b_it"):
            st, _detail = W.legs(ROSTER, wname)["bfcl_200"]
            assert st == "runnable", wname
            assert C.model_row(W.wave(ROSTER, wname)["model_row"]).petri_manifest == C.PETRI_MANIFEST_FULL22_QWEN35_4B
        assert C.TOOL_PARSER_LIVE_PASS["qwen3_5"] and C.tool_parser_live_gate_problem("qwen3_5") is None
        # prerequisites: the geometry line, the bank, the derived parents' staging and the Arditi extra the seed
        # config enumerates are DECLARED — and RECORDED met (2026-09-23: staged and sha-verified by the coordinator)
        decl = w["prerequisites"]
        assert len(decl) == 4 and w["prerequisites_met"] == decl and W.prerequisites(ROSTER, seed) == []
        assert decl[1].startswith("S4 (bank staging): the seed bank") and f"/full22/qwen35_4b/s{s_} " in decl[1]
        assert decl[2].startswith("S4: the derived object's parents staged at /vol/out/qwen35_4b_base_eb64_s" + s_)
        assert decl[3].startswith("S4 (bank staging): the Arditi extra staged INSIDE") and "ba2212f0" in decl[3]
        # the Arditi wave: the same row and bank; RUNNABLE — the refusal-steer cell pinned to the bank extra
        # (ar_c0p3__arditi_allpos, kind vector_all, sidecar fit_provenance.json, generation `final`), `none` a free replicate,
        # the extra's staging DECLARED as an unmet prerequisite
        a = W.wave(ROSTER, arditi)
        assert (a["model_row"], a["objects_dir"], a["view"]) == (w["model_row"], w["objects_dir"], w["view"])
        assert W.runnable_cells(ROSTER, arditi) == ["none", "arditi_allpos"] and W.queued_or_blocked(ROSTER, arditi) == {}
        e = W.entries(ROSTER, arditi)["arditi_allpos"]
        assert e["kind"] == "lane_extra" and e["object"]["staged_id"] == "ar_c0p3__arditi_allpos" and e["object"]["kind"] == "vector_all"
        assert e["object"]["sidecar"] == "fit_provenance.json" and e["object"]["object_generation"] == "final" and e["object"]["fit"]["served_layer"] == 12
        assert W.pins_by_cell(ROSTER, arditi) == {"arditi_allpos": "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"}
        assert PINS["qwen35_4b"]["ar_c0p3__arditi_allpos"]["object_sha256"] == e["object"]["sha256"] and PINS["qwen35_4b"]["ar_c0p3__arditi_allpos"]["kind"] == "vector_all"
        assert "FREE" in W.entries(ROSTER, arditi)["none"]["note"]
        decl_a = a["prerequisites"]                        # the extra AND the seed bank (the same row's standup registers both), both met
        assert len(decl_a) == 2 and decl_a[0].startswith("S4 (bank staging): the Arditi extra staged INSIDE") and "ba2212f0" in decl_a[0]
        assert decl_a[1].startswith("S4 (bank staging): the seed bank") and f"/full22/qwen35_4b/s{s_} " in decl_a[1]
        assert a["prerequisites_met"] == decl_a and W.prerequisites(ROSTER, arditi) == []
        # the row's inventory covers the seed cells AND the extra (the derivation through the served-only Class B branch)
        assert set(C.EXPECTED_INTERVENTION_HASHES[w["model_row"]]) == set(pins) | {"arditi_allpos"}
        # the paired wave: its own row / bank, the composed cell pinned to the paired_eb64s<s> lineage, `none` a free replicate
        pw = W.wave(ROSTER, paired)
        assert (pw["model_row"], pw["objects_dir"], pw["view"]) == (f"qwen35_4b-f22s{s_}p", f"/vol/objects/sdfbase/full22/qwen35_4b/s{s_}p", f"qwen35_4b_s{s_}p")
        assert W.runnable_cells(ROSTER, paired) == ["none", PAIRED]
        obj = W.entries(ROSTER, paired)["diff_psr"]["object"]
        assert obj["staged_id"] == f"paired_eb64s{s_}__{PAIRED}" and obj["kind"] == "psr_paired_write" and obj["object_generation"] == "batch64"
        assert W.pins_by_cell(ROSTER, paired) == {PAIRED: PINS["qwen35_4b"][obj["staged_id"]]["object_sha256"]}
        assert set(C.EXPECTED_INTERVENTION_HASHES[pw["model_row"]]) == {PAIRED}
        assert "FREE" in W.entries(ROSTER, paired)["none"]["note"] and W.pooling_of(ROSTER, paired, "none") is None
        assert len(pw["prerequisites"]) == 3 and "psr_ce_all" in pw["prerequisites"][2] and pw["prerequisites_met"] == pw["prerequisites"] and W.prerequisites(ROSTER, paired) == []
    # the roster still passes every rule with these waves in it
    assert W.validate(ROSTER, PINS, C.MODELS) == []


def test_bank_staging_prerequisites_must_cover_every_fitted_cell_the_rows_config_enumerates():
    """The seed configs list `arditi_allpos` (a lane extra) beside the five seed objects, and every
    standup of the row builds its registry over the WHOLE bank — so a seed wave whose prerequisites named the five
    objects but not the extra could be truthfully marked met with the extra missing, the wrapper would admit even the
    `none` canary, and the registry would refuse inside a metered container. `validate` now holds every wave that
    declares bank-staging prerequisites to name every fitted cell of its row's config; the shipped roster
    does (the seed waves declare the extra, the Arditi waves declare the seed bank); a wave declaring no staging line
    is not held (its bank predates the discipline)."""
    for s_ in ("1", "2", "3"):
        seed, arditi, paired = f"qwen35_4b_s{s_}", f"qwen35_4b_s{s_}_arditi", f"qwen35_4b_s{s_}p"
        cfg_cells = W.config_fitted_cells(C.model_row(f"qwen35_4b-f22s{s_}"))
        assert "arditi_allpos" in cfg_cells and len(cfg_cells) == 6
        for wname in (seed, arditi):
            w = W.wave(ROSTER, wname)
            lines = W.bank_staging_prerequisites(w)
            assert len(lines) == 2 and any("Arditi extra" in ln for ln in lines) and any("the seed bank" in ln for ln in lines), wname
            assert W.staging_coverage_problems(w, C.model_row(f"qwen35_4b-f22s{s_}"), where=wname) == []
        assert len(W.bank_staging_prerequisites(W.wave(ROSTER, paired))) == 1
        assert W.staging_coverage_problems(W.wave(ROSTER, paired), C.model_row(f"qwen35_4b-f22s{s_}p"), where=paired) == []
    # drop the extra's line from a seed wave -> the validator names the uncovered cell on THAT wave
    r = copy.deepcopy(ROSTER)
    w = r["waves"]["qwen35_4b_s1"]
    w["prerequisites"] = [p for p in w["prerequisites"] if "Arditi extra" not in p]
    w["prerequisites_met"] = [p for p in w["prerequisites_met"] if "Arditi extra" not in p]      # the record follows the declaration
    problems = W.validate(r, PINS, C.MODELS)
    assert any(p.startswith("waves.qwen35_4b_s1:") and "['arditi_allpos']" in p and "declare the staging on THIS wave" in p for p in problems), problems
    # drop the seed bank line from the Arditi wave -> the five seed objects are uncovered there
    r = copy.deepcopy(ROSTER)
    w = r["waves"]["qwen35_4b_s2_arditi"]
    w["prerequisites"] = [p for p in w["prerequisites"] if "the seed bank" not in p]
    w["prerequisites_met"] = [p for p in w["prerequisites_met"] if "the seed bank" not in p]
    (why,) = [p for p in W.validate(r, PINS, C.MODELS) if p.startswith("waves.qwen35_4b_s2_arditi:")]
    for c in ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"):
        assert c in why
    # whole-token match: a bank line naming only the DERIVED `…lora_r4_ce_minus_ctl` id does not cover its parent `…lora_r4_ce`
    r = copy.deepcopy(ROSTER)
    w = r["waves"]["qwen35_4b_s3"]
    w["prerequisites"] = [p for p in w["prerequisites"] if "the seed bank" not in p] + [
        W.BANK_STAGING_PREFIX + " the seed bank staged for base64s3__constv2-mix-full__psr_ce_all, base64s3__constv2ctl2-full__psr_ce_all, "
        "base64s3__constv2ctl2-full__lora_r4_ce, derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl"]
    w["prerequisites_met"] = [p for p in w["prerequisites_met"] if "the seed bank" not in p]
    (why,) = [p for p in W.validate(r, PINS, C.MODELS) if p.startswith("waves.qwen35_4b_s3:")]
    assert "['constv2-mix-full__lora_r4_ce']" in why
    # a wave with no staging line is not held (the earlier waves' banks predate the discipline)
    assert W.bank_staging_prerequisites(W.wave(ROSTER, "qwen35_4b_main")) == [] and W.staging_coverage_problems(W.wave(ROSTER, "qwen35_4b_main"), C.model_row("qwen35_4b-f22"), where="x") == []
    # a row whose config is not on this tree is not held either (nothing to read)
    assert W.config_fitted_cells(C.model_row("qwen35_4b-f22")._replace(cfg="configs/tinycap/nope.yaml")) is None
    assert W.BANK_STAGING_PREFIX == "S4 (bank staging):"
    # the row configs are read from the tree handed in (the staged tree in a container; the checkout by default)
    assert W.validate(ROSTER, PINS, C.MODELS, tree=W.REPO) == [] and W.config_fitted_cells(C.model_row("qwen35_4b-f22s1"), W.REPO) == cfg_cells



# --------------------------------------------------------------------------- #
# capeval-full-tasks: the FULL lane's sibling roster                             #
# --------------------------------------------------------------------------- #


def test_the_full_lane_roster_reads_through_the_same_reader_with_the_full_legs():
    """The sibling document names the FULL legs; the reader resolves a lane's legs through
    ``lane_legs`` (never ``LEGS`` directly), so every 200-lane reading is byte-identical and the
    full roster's runnable legs are the full atoms in BCB-then-BFCL order."""
    assert W.LANE_FULL == "bcb-bfcl-full" and W.LEGS_FULL == ("bigcodebench_full", "bfcl_full")
    assert W.LEGS_OF_LANE == {W.LANE: W.LEGS, W.LANE_FULL: W.LEGS_FULL}
    assert W.BFCL_LEG_OF_LANE == {W.LANE: "bfcl_200", W.LANE_FULL: "bfcl_full"}
    assert W.lane_of(ROSTER) == W.LANE and W.lane_legs(ROSTER) == W.LEGS
    with pytest.raises(W.RosterError, match="not one of"):
        W.legs_of_lane("bcb-bfcl-whatever")
    full = W.load_roster(W.ROSTER_FULL_PATH)
    assert W.lane_of(full) == W.LANE_FULL and W.lane_legs(full) == W.LEGS_FULL
    assert full["legs"] == ["inspect/bigcodebench_full", "inspect/bfcl_full"]
    # the six eight-arm waves, then the COMPLETION waves: the three Qwen Arditi waves (view = the seed view; entries
    # none + arditi_allpos) and the six paired-write views (<family>_s<s>p: entries none + the composed difference ReLU)
    # ... then the INSTRUCT-DIRECT waves: the 200 roster's qwen35_4b_it / llama8b_it with the five ordered arms as entries
    assert W.waves(full) == ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3", "gemma4e4b_s1", "llama8b_s1", "nemotron3nano4b_s1",
                             "qwen35_4b_s1_arditi", "qwen35_4b_s2_arditi", "qwen35_4b_s3_arditi",
                             "qwen35_4b_s1p", "qwen35_4b_s2p", "qwen35_4b_s3p", "gemma4e4b_s1p", "llama8b_s1p", "nemotron3nano4b_s1p",
                             "qwen35_4b_it", "llama8b_it", "gemma4e4b_it"]
    IT = ("qwen35_4b_it", "llama8b_it", "gemma4e4b_it")
    for w in IT:
        assert W.wave(full, w)["lane_kind"] == "instruct-direct" and W.wave(full, w)["model_row"] == w.replace("_it", "-f22it")
        assert W.runnable_cells(full, w)[0] == "none" and "never launched" in W.entries(full, w)["none"]["note"].lower()
    q, ll = W.entries(full, "qwen35_4b_it"), W.entries(full, "llama8b_it")
    assert W.runnable_cells(full, "qwen35_4b_it") == ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", DERIVED]
    assert W.runnable_cells(full, "llama8b_it") == ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", DERIVED]   # The difference LoRA joined the order
    g = W.entries(full, "gemma4e4b_it")
    assert W.runnable_cells(full, "gemma4e4b_it") == ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", DERIVED]
    assert (g["const_lora"]["object"]["staged_id"], g["const_lora"]["object"]["sha256"][:8], g["const_lora"]["object"]["object_generation"]) == ("itdoc__constv2-mix-full__lora_r4_ce", "4c1a5fc3", "batch8-provisional")
    assert (g["ctl_lora"]["object"]["sha256"][:8], g["diff_lora"]["object"]["sha256"][:8]) == ("04a90de8", "8b7d1030") and g["si_safety"]["status"] == "blocked" and g["const_si"]["status"] == "blocked"
    assert (q["const_lora"]["object"]["staged_id"], q["const_lora"]["object"]["sha256"][:8], q["const_lora"]["object"]["object_generation"]) == \
        ("itdoc__constv2-mix-full__lora_r4_ce", "0018b32b", "batch8-provisional") and q["const_lora"]["object"]["superseded_by_key"] == "qwen35_4b_it_doc_eb64"
    assert (q["ctl_lora"]["object"]["staged_id"], q["ctl_lora"]["object"]["sha256"][:8], q["ctl_lora"]["object"]["object_generation"]) == ("itdoc64__constv2ctl2-full__lora_r4_ce", "bace45e2", "batch64")
    assert (q["diff_lora"]["object"]["staged_id"], q["diff_lora"]["object"]["sha256"][:8]) == ("derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl", "5e9b0009") and "both legs" in q["diff_lora"]["note"].lower()
    assert (ll["const_lora"]["object"]["staged_id"], ll["const_lora"]["object"]["sha256"][:8]) == ("itdoc__constv2-mix-full__lora_r4_ce", "529788ca") and "d1it__" in ll["const_lora"]["note"]
    assert (ll["ctl_lora"]["object"]["staged_id"], ll["ctl_lora"]["object"]["sha256"][:8]) == ("itdoc__constv2ctl2-full__lora_r4_ce", "7bf26f8c")
    assert ll["diff_lora"]["status"] == "runnable" and ll["diff_lora"]["object"]["sha256"][:8] == "e85220c9" and "added by the corrected order" in ll["diff_lora"]["note"]
    assert "both legs" in q["diff_lora"]["note"].lower()
    # every instruct-direct object is the harness roster's staged id for the cell on the `it` view (the reader's identity source)
    harness = yaml.safe_load((W.REPO / "configs" / "sdfbase_full22_roster.yaml").read_text())
    for w, mk in (("qwen35_4b_it", "qwen35_4b"), ("llama8b_it", "llama8b")):
        cells = harness["targets"][mk]["views"]["it"]["cells"]
        for e in W.entries(full, w).values():
            if e.get("object") and e.get("status") == "runnable":
                assert cells[e["cell_id"]] == e["object"]["staged_id"], (w, e["cell_id"])
    for s_ in (1, 2, 3):
        assert W.runnable_cells(full, f"qwen35_4b_s{s_}_arditi") == ["none", "arditi_allpos"] and W.wave(full, f"qwen35_4b_s{s_}_arditi")["view"] == f"qwen35_4b_s{s_}"
        assert W.entries(full, f"qwen35_4b_s{s_}_arditi")["arditi_allpos"]["object"]["staged_id"] == "ar_c0p3__arditi_allpos"
        assert W.runnable_cells(full, f"qwen35_4b_s{s_}p") == ["none", "constv2-mix-full__psr_ce_all_minus_ctl"]
        assert W.entries(full, f"qwen35_4b_s{s_}p")["diff_psr"]["object"]["staged_id"] == f"paired_eb64s{s_}__constv2-mix-full__psr_ce_all_minus_ctl"
    for mk, batch in (("gemma4e4b", "eb8"), ("llama8b", "eb64"), ("nemotron3nano4b", "eb8")):
        assert W.wave(full, f"{mk}_s1p")["model_row"] == f"{mk}-f22s1p"
        assert W.entries(full, f"{mk}_s1p")["diff_psr"]["object"]["staged_id"] == f"paired_{batch}__constv2-mix-full__psr_ce_all_minus_ctl"
    # the same header terms as the 200 roster (protocol decoding, the clamp override, thinking off, the pins, the serving path)
    for key in ("sampling", "sampling_clamp_override", "thinking", "pins", "serving_path", "excluded_families", "object_generations"):
        assert full[key] == ROSTER[key], key
    for wname in W.waves(full):
        legs = W.runnable_legs(full, wname)
        assert legs[0] == "bigcodebench_full"
        assert legs == (["bigcodebench_full"] if wname.startswith("llama8b") else ["bigcodebench_full", "bfcl_full"]), wname
        if wname in IT:
            continue          # the instruct-direct waves ADD entries beyond the 200 roster's (the ordered arms) — pinned above
        # entries copied from the 200 roster's wave of the same name (eight arms on the seed waves; two on the Arditi / paired waves)
        assert set(W.entries(full, wname)) == set(W.entries(ROSTER, wname)), wname
        assert W.runnable_cells(full, wname) == W.runnable_cells(ROSTER, wname), wname
        assert W.wave(full, wname)["model_row"] == W.wave(ROSTER, wname)["model_row"]
    # llama's BFCL leg carries the D104 block verbatim (plus the inheritance note)
    st, det = W.legs(full, "llama8b_s1")["bfcl_full"]
    assert st == "blocked" and "no function calling" in det["unblock"] and "bfcl_full inherits" in det["reason"]


def test_the_full_lane_roster_validates_and_the_legs_are_held_to_the_lane():
    full = W.load_roster(W.ROSTER_FULL_PATH)
    assert W.validate(full, PINS, C.MODELS) == []
    # a full roster naming a 200 leg refuses by name; a 200 roster naming a full leg refuses by name
    bad = copy.deepcopy(full)
    bad["waves"]["gemma4e4b_s1"]["legs"]["bigcodebench_200"] = "runnable"
    assert any("legs.bigcodebench_200: not one of ['bigcodebench_full', 'bfcl_full']" in p for p in W.validate(bad, PINS, C.MODELS))
    bad = copy.deepcopy(full)
    bad["legs"] = ["inspect/bigcodebench_200", "inspect/bfcl_200"]
    assert any("legs" in p and "lane's two pinned row tasks" in p for p in W.validate(bad, PINS, C.MODELS))
    bad = copy.deepcopy(ROSTER)
    bad["waves"]["gemma4e4b_it"]["legs"]["bfcl_full"] = "runnable"
    assert any("legs.bfcl_full: not one of ['bigcodebench_200', 'bfcl_200']" in p for p in W.validate(bad, PINS, C.MODELS))
    bad = copy.deepcopy(full)
    bad["lane"] = "bcb-bfcl-whatever"
    assert any("lane 'bcb-bfcl-whatever' !=" in p for p in W.validate(bad, PINS, C.MODELS))
    # the live-parser gate guards the LANE's BFCL leg by name (a withdrawn live record refuses `bfcl_full`, never `bfcl_200`)
    saved = C.tool_parser_live_gate_problem
    C.tool_parser_live_gate_problem = lambda family, tree=None, **kw: "the live parser record is withdrawn (synthetic)"
    try:
        probs = W.validate(full, PINS, C.MODELS)
    finally:
        C.tool_parser_live_gate_problem = saved
    assert probs and all("bfcl_full is runnable but the live parser record is withdrawn" in p for p in probs), probs
    assert not any("bfcl_200" in p for p in probs)
