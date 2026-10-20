"""qwen-seedrep-native-battery: an S4 PASS record was never produced on any tree (one wave stopped, another drained at its
deadline with the crescendo `none` child COMPLETE), and `certificate_problems` refuses `no S4 PASS record` -- so no wave certificate could
issue. The fix: the reader admits a DOCUMENTED S4 WAIVER (`RS/s4_waiver.json`, closed schema) in place of the S4 PASS record -- for S4 only, only
when every field is present and typed, only when the named attempt roots exist on the volume (their RESULT re-read and held to the recorded
line) -- and records `smokes.S4 = {status: WAIVED, waiver: {path, sha256, document_sha256, document}}` so every reader renders it as a
qualification. `go_preconditions` (what `--go` re-validates) accepts the waived certificate; the pass records of S1-S3 are validated as before.

Known-bads, each refused by name: a waiver missing a field / carrying an extra one / free text / empty evidence / a root not on the volume /
a re-worded RESULT / another tree / any smoke id but S4; a WAIVED entry under S3; an edited document inside a re-sealed certificate; the CLI
refusing the waiver file beside a derived S4 PASS record.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_launch_plan as lp  # noqa: E402
from test_seedrep_review_fixes import FakeVOL, _stub_modal, draft, inputs  # noqa: E402,F401  (the module fixtures)
from test_seedrep_review_fixes import TestSolR3_2_launch_certificate as _CertComponents  # noqa: E402  (aliased: not collected here)

# The AUTHORIZED source attempts: the two drained smoke attempts the waiver table names, by their ids and standups; the reader admits no
# other attempt, and holds each to its own ATTEMPT.json / RESULT on the volume
CRES_ID = next(a for a, e in lp.S4_WAIVER_AUTHORIZED.items() if e["leg"] == "crescendo")
ACT_ID = next(a for a, e in lp.S4_WAIVER_AUTHORIZED.items() if e["leg"] == "actorattack")
CRES_ROOT, ACT_ROOT = lp.s4_waiver_authorized_root(CRES_ID), lp.s4_waiver_authorized_root(ACT_ID)
CRES_RESULT = f"2026-09-19T20:17:29Z status=DRIVERS_FAILED rc=1 attempt={CRES_ID} standup=smoke-S4__crescendo__s1__base__shard00 children=5 failed=4 S=10 elapsed_s=8416.9"
ACT_RESULT = f"2026-09-19T20:19:12Z status=DRIVERS_FAILED rc=1 attempt={ACT_ID} standup=smoke-S4__actorattack__s1__base__shard00 children=5 failed=5 S=10 elapsed_s=8519.5"
W22 = lp.S4_WAIVER_PRIOR_W7 + "6e2975804602aec7d12ba1c98c805adfd"


def _attempt_json(aid: str, standup: str, *, W: str = W22, row: str = None) -> str:
    return json.dumps({"schema": "seedrep_attempt/1", "attempt_id": aid, "W": W, "wave_uuid": "a79287e0-0db6-411e-95b2-83b90f58c4d0",
                       "launch_plan_row": row or standup, "control_key": f"{W[:7]}/a79287e0-0db6-411e-95b2-83b90f58c4d0/{standup.split('__')[1]}/s1/base/0"})


def waiver_doc(W: str) -> dict:
    return {"schema": lp.S4_WAIVER_SCHEMA, "smoke_id": "S4", "W": W, "waived_by": "user ruling D133 via orchestrator",
            "evidence": [{"attempt_id": CRES_ID, "standup_id": "smoke-S4__crescendo__s1__base__shard00", "attempt_root": CRES_ROOT, "RESULT": CRES_RESULT,
                          "facts": {"none_child": {"rc": 0, "goals_run": "65/65", "n_target_forwards_total": 506, "attack_failures": 0},
                                    "fitted_children": "4 x rc -15 (SIGTERM at the D114 drain 20:17:02Z), not failed", "pools": "accepting the bearer"}},
                         {"attempt_id": ACT_ID, "standup_id": "smoke-S4__actorattack__s1__base__shard00", "attempt_root": ACT_ROOT, "RESULT": ACT_RESULT,
                          "facts": {"children": "5 x drained at the D114 deadline; no child completed"}}],
            "pools_geometry": "H200:1 (D116) -- not exercised live under a completed S4",
            "what_the_waiver_does_not_prove": list(lp.S4_WAIVER_LIMITATIONS),
            "utc": "2026-09-20T03:10:00Z"}


def volume_with_roots(read_text, extra=None):
    texts = {f"{CRES_ROOT}/RESULT": CRES_RESULT + "\n", f"{ACT_ROOT}/RESULT": ACT_RESULT + "\n",
             f"{CRES_ROOT}/ATTEMPT.json": _attempt_json(CRES_ID, "smoke-S4__crescendo__s1__base__shard00"),
             f"{ACT_ROOT}/ATTEMPT.json": _attempt_json(ACT_ID, "smoke-S4__actorattack__s1__base__shard00"), **(extra or {})}

    def read(path):
        return texts[path] if path in texts else read_text(path)
    return read


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def components(inputs, draft, tmp_path_factory):
    return _CertComponents()._components(inputs, draft, tmp_path_factory.mktemp("fc3-s4"))


def _waived_smokes(c):
    doc = waiver_doc(c["final"]["W"])
    text = json.dumps(doc, indent=1) + "\n"
    smokes = {sid: rec for sid, rec in c["smokes"].items() if sid != "S4"}
    smokes["S4"] = lp.smoke_waiver_entry(doc, path=f"RS/{lp.S4_WAIVER_FILE}", file_sha256=_sha(text))
    return smokes, doc


def test_a_documented_s4_waiver_certifies_and_go_accepts_it(components):
    c = components
    final, draft_, proof = c["final"], c["draft"], c["proof"]
    smokes, doc = _waived_smokes(c)
    rt = volume_with_roots(c["read_text"])
    reads = {sid: r for sid, r in c["reads"].items() if sid != "S4"}
    # BEFORE (the known-bad this fixes): no S4 PASS record and no waiver -> refused by name
    without = {sid: rec for sid, rec in smokes.items() if sid != "S4"}
    with pytest.raises(ValueError, match="no S4 PASS record in the certificate \\(nor an admitted s4_waiver.json waiver, D133\\)"):
        lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft_, s0_record=c["s0"], smokes_doc=without, w_gate=c["w_gate"], decode_proof=proof,
                             rs=c["rs"], registry_sha256=proof["registry_sha256"], read_text=rt, accepted_reads=reads)
    # AFTER: the waiver stands in for S4 -- issued, sealed, and what --go re-validates accepts it
    cert = lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=draft_, s0_record=c["s0"], smokes_doc=smokes, w_gate=c["w_gate"], decode_proof=proof,
                                rs=c["rs"], registry_sha256=proof["registry_sha256"], read_text=rt, accepted_reads=reads)
    s4 = cert["smokes"]["S4"]
    assert s4["status"] == lp.SMOKE_WAIVED == "WAIVED" and s4["smoke_id"] == "S4" and tuple(s4) == lp.SMOKE_WAIVER_ENTRY_KEYS
    assert tuple(s4["waiver"]) == lp.SMOKE_WAIVER_REF_KEYS and s4["waiver"]["document"] == doc and s4["waiver"]["path"].endswith("s4_waiver.json")
    assert "S4" not in cert["accepted_dicts"], "a waived smoke has no accepted-Dict read to record"
    assert all(cert["smokes"][sid]["schema"] == lp.SMOKE_PASS_SCHEMA for sid in ("S1", "S1b", "S2", "S3")), "the other smokes stay derived PASS records"
    assert lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"], read_text=rt) == []
    assert lp.go_preconditions(final, certificate=json.loads(json.dumps(cert)), registry_sha256=proof["registry_sha256"], read_text=rt) == [], "JSON round trip keeps the seal"
    # without a volume reader (no read_text) the roots cannot be re-read -- the document still validates on its own content
    assert lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"]) == []
    # the roots GONE from the volume -> --go refuses by name (the waiver named evidence that no longer exists)
    probs = lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"], read_text=c["read_text"])
    assert any("is not on the volume" in p and CRES_ID in p for p in probs) and any(ACT_ID in p for p in probs), probs
    # a RESULT re-worded on the volume -> refused
    reworded = volume_with_roots(c["read_text"], {f"{CRES_ROOT}/RESULT": CRES_RESULT.replace("failed=4", "failed=0") + "\n"})
    probs = lp.go_preconditions(final, certificate=cert, registry_sha256=proof["registry_sha256"], read_text=reworded)
    assert any("is not the recorded line" in p for p in probs), probs


S1_ATTEMPT = "01TESTS1PASTTENSE0000000000-cccccccccccc"


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.pop("pools_geometry"), "lacks ['pools_geometry']"),
    (lambda d: d.update(reason="because"), "outside the closed schema ['reason']"),
    (lambda d: d.update(smoke_id="S3"), "a smoke waiver is admitted for S4 only"),
    (lambda d: d.update(W="f" * 40), "is not this plan's"),
    (lambda d: d.update(waived_by="the orchestrator decided"), "does not name the ruling D133"),
    (lambda d: d.update(waived_by="user ruling D118 via orchestrator"), "does not name the ruling D133"),
    (lambda d: d.update(waived_by=""), "waived_by missing or empty"),
    (lambda d: d.update(evidence=[]), "an empty waiver waives nothing"),
    (lambda d: d.update(evidence=["the W22 crescendo run looked fine"]), "free text is not evidence"),
    # fc7: an invented waiver over an unrelated S1 attempt -- refused as unauthorized AND as missing the authorized attempts
    (lambda d: d["evidence"].__setitem__(0, {**d["evidence"][0], "attempt_id": S1_ATTEMPT, "standup_id": "smoke-S1__pasttense__s1__base__shard00",
                                             "attempt_root": f"/vol/seedrep/1111111/exp/seedrep_battery_modal_smoke/S1/pasttense/s1/shard00/attempts/{S1_ATTEMPT}",
                                             "RESULT": f"2026-09-19T00:00:00Z status=DONE rc=0 attempt={S1_ATTEMPT} standup=smoke-S1__pasttense__s1__base__shard00 children=8 failed=0 S=45 elapsed_s=900"}),
     "is not one of D133's authorized source attempts"),
    (lambda d: d["evidence"].pop(1), "does not name D133's authorized source attempt(s)"),
    (lambda d: d["evidence"][0].update(standup_id="smoke-S4__actorattack__s1__base__shard00"), "is not the authorized attempt's standup"),
    (lambda d: d["evidence"][0].update(attempt_root=CRES_ROOT.replace("/56aff57/", "/1111111/")), "is not the authorized attempt's root"),
    (lambda d: d["evidence"][0].update(RESULT=CRES_RESULT.replace("failed=4", "failed=0")), "does not state the authorized attempt's outcome"),
    (lambda d: d["evidence"][0].update(RESULT="drained, all good"), "does not state the authorized attempt's outcome"),
    (lambda d: d["evidence"][0].pop("facts"), "evidence[0]: lacks ['facts']"),
    (lambda d: d["evidence"][0].update(facts={}), "facts must be a non-empty object"),
    (lambda d: d["evidence"][0]["facts"].pop("pools"), "facts.pools missing or empty"),
    (lambda d: d["evidence"][0]["facts"]["none_child"].update(goals_run="64/65"), "facts.none_child.goals_run = '64/65', not the recorded '65/65'"),
    (lambda d: d["evidence"][1].update(facts={"note": "fine"}), "facts.children missing or empty"),
    (lambda d: d["evidence"].append(dict(d["evidence"][0])), "is named twice"),
    (lambda d: d.update(what_the_waiver_does_not_prove=[]), "non-empty list of non-empty statements"),
    (lambda d: d.update(what_the_waiver_does_not_prove=["a completed fitted-cell multi-turn store on this tree"]), "lacks the required limitation statement(s)"),
    (lambda d: d.update(pools_geometry="H200:8 exercised under load"), "does not state ['H200:1', 'D116']"),
    (lambda d: d.update(utc="yesterday"), "is not an ISO-8601 timestamp"),
    (lambda d: d.update(schema="seedrep_smoke_waiver/0"), "schema 'seedrep_smoke_waiver/0' !="),
])
def test_the_waiver_reader_refuses_a_malformed_or_free_text_waiver_by_name(mutate, needle):
    plan = {"W": "a" * 40}
    good = waiver_doc(plan["W"])
    assert lp.s4_waiver_problems(good, plan=plan, read_text=volume_with_roots(lambda p: None)) == []
    bad = copy.deepcopy(good)
    mutate(bad)
    probs = lp.s4_waiver_problems(bad, plan=plan, read_text=volume_with_roots(lambda p: None))
    assert any(needle in p for p in probs), (needle, probs)
    assert lp.s4_waiver_problems("waived", plan=plan) == ["s4_waiver: not an object"]


def test_the_attempts_own_identity_record_on_the_volume_binds_the_standup(components):
    c = components
    plan = {"W": c["final"]["W"]}
    doc = waiver_doc(plan["W"])
    assert lp.s4_waiver_problems(doc, plan=plan, read_text=volume_with_roots(lambda p: None)) == []
    # ATTEMPT.json naming another launch_plan_row / attempt / tree, or absent -> refused by name (the RESULT alone does not bind the standup)
    for extra, needle in (({f"{CRES_ROOT}/ATTEMPT.json": _attempt_json(CRES_ID, "smoke-S4__crescendo__s1__base__shard00", row="smoke-S1__pasttense__s1__base__shard00")}, "records launch_plan_row"),
                          ({f"{CRES_ROOT}/ATTEMPT.json": _attempt_json("01OTHER", "smoke-S4__crescendo__s1__base__shard00")}, "ATTEMPT.json names attempt"),
                          ({f"{CRES_ROOT}/ATTEMPT.json": _attempt_json(CRES_ID, "smoke-S4__crescendo__s1__base__shard00", W="c" * 40)}, "not the W22 tree 56aff57"),
                          ({f"{CRES_ROOT}/ATTEMPT.json": "not json"}, "is not on the volume (or not JSON)")):
        probs = lp.s4_waiver_problems(doc, plan=plan, read_text=volume_with_roots(lambda p: None, extra))
        assert any(needle in p for p in probs), (needle, probs)
    rt = volume_with_roots(lambda p: None)
    gone = lambda p: None if p.endswith("/ATTEMPT.json") else rt(p)          # noqa: E731
    assert any("ATTEMPT.json is not on the volume" in p for p in lp.s4_waiver_problems(doc, plan=plan, read_text=gone))
    # the authorized roots are COMPOSED the way the app spells every smoke attempt root, never typed
    assert CRES_ROOT == f"/vol/seedrep/56aff57/exp/seedrep_battery_modal_smoke/S4/crescendo/s1/shard00/attempts/{CRES_ID}"
    assert ACT_ROOT == f"/vol/seedrep/56aff57/exp/seedrep_battery_modal_smoke/S4/actorattack/s1/shard00/attempts/{ACT_ID}"


def test_a_waiver_for_any_id_but_s4_and_an_edited_document_are_refused(components):
    c = components
    final, proof = c["final"], c["proof"]
    smokes, doc = _waived_smokes(c)
    rt = volume_with_roots(c["read_text"])
    reads = {sid: r for sid, r in c["reads"].items() if sid != "S4"}
    cert = lp.build_certificate(plan=final, plan_path=c["plan_p"], draft=c["draft"], s0_record=c["s0"], smokes_doc=smokes, w_gate=c["w_gate"], decode_proof=proof,
                                rs=c["rs"], registry_sha256=proof["registry_sha256"], read_text=rt, accepted_reads=reads)
    # the same WAIVED entry under S3 -> refused: only S4 may be waived
    bad = {**cert, "smokes": {**cert["smokes"], "S3": copy.deepcopy(cert["smokes"]["S4"])}}
    bad["certificate_sha256"] = lp.certificate_sha256(bad)
    probs = lp.go_preconditions(final, certificate=bad, registry_sha256=proof["registry_sha256"], read_text=rt)
    assert any("S3: a smoke WAIVER is admitted only for S4 (D133) -- S3 needs a derived PASS record" in p for p in probs), probs
    # the document edited INSIDE the certificate: the seal breaks; re-sealed, the document digest breaks
    edited = copy.deepcopy(cert)
    edited["smokes"]["S4"]["waiver"]["document"]["pools_geometry"] = "H200:8 exercised under load"
    assert any("seal does not match" in p for p in lp.go_preconditions(final, certificate=edited, registry_sha256=proof["registry_sha256"], read_text=rt))
    edited["certificate_sha256"] = lp.certificate_sha256(edited)
    probs = lp.go_preconditions(final, certificate=edited, registry_sha256=proof["registry_sha256"], read_text=rt)
    assert any("document_sha256 is not the digest of the embedded document" in p for p in probs), probs
    # a WAIVED entry with the S4 document but the wrong smoke_id / status / keys
    for over, needle in (({"smoke_id": "S3"}, "is not a WAIVED S4 entry"), ({"status": "PASS"}, "is not a WAIVED S4 entry"), ({"rows": {}}, "are not exactly")):
        e = {**copy.deepcopy(cert["smokes"]["S4"]), **over}
        if "rows" not in over:
            probs = lp.smoke_waiver_entry_problems(e, smoke_id="S4", plan=final)
        else:
            probs = lp.smoke_waiver_entry_problems(e, smoke_id="S4", plan=final)
        assert any(needle in p for p in probs), (over, probs)
    ref_bad = copy.deepcopy(cert["smokes"]["S4"])
    ref_bad["waiver"]["path"] = "RS/waiver.txt"
    assert any("does not name s4_waiver.json" in p for p in lp.smoke_waiver_entry_problems(ref_bad, smoke_id="S4", plan=final))
    # the entry is recognised by its status alone; a PASS record is never mistaken for one
    assert lp.is_smoke_waiver_entry(cert["smokes"]["S4"]) and not lp.is_smoke_waiver_entry(cert["smokes"]["S3"]) and not lp.is_smoke_waiver_entry(None)


def test_certify_cli_admits_rs_s4_waiver_json_and_refuses_it_beside_a_derived_s4_pass(components, tmp_path, monkeypatch):
    """The CLI path the orchestrator runs: `pass` for S1-S3 (S4 never ran to completion, so no `pass`), `RS/s4_waiver.json` written, `w-gate`,
    `decode-proof`, `certify` -> the sealed certificate carries `smokes.S4 = WAIVED`; a second RS with a derived S4 PASS beside the waiver file
    refuses (nothing is picked silently)."""
    c = components
    final, rs0 = c["final"], c["rs"]
    spec = importlib.util.spec_from_file_location("seedrep_s5_record_fc3", REPO / "scripts/modal/seedrep_s5_record.py")
    S5R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S5R)
    stub = _stub_modal()
    monkeypatch.setitem(sys.modules, "modal", stub)
    for sid in lp.SMOKE_PASS_IDS + ("S5",):
        d = stub.Dict.from_name(lp.smoke_identity(c["draft"], sid, c["waves"][sid])["dicts"]["accepted"], create_if_missing=False)
        for k, v in c["accepted"][sid].items():
            d.put(k, v)
    FakeVOL.files.clear()
    for path, text in c["texts"].items():
        FakeVOL.files[path[len("/vol/"):]] = text
    FakeVOL.files[f"{CRES_ROOT}/RESULT"[len("/vol/"):]] = CRES_RESULT + "\n"
    FakeVOL.files[f"{ACT_ROOT}/RESULT"[len("/vol/"):]] = ACT_RESULT + "\n"
    FakeVOL.files[f"{CRES_ROOT}/ATTEMPT.json"[len("/vol/"):]] = _attempt_json(CRES_ID, "smoke-S4__crescendo__s1__base__shard00")
    FakeVOL.files[f"{ACT_ROOT}/ATTEMPT.json"[len("/vol/"):]] = _attempt_json(ACT_ID, "smoke-S4__actorattack__s1__base__shard00")
    # a fresh RS beside the components' (their smokes.json may already hold records from other tests of the module)
    rs = tmp_path / "RS"
    rs.mkdir()
    for sid in lp.SMOKE_PASS_IDS + ("S5",):
        wp = lp.smoke_wave_path(rs, sid)
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps({"smoke_id": sid, "wave_uuid": c["waves"][sid]}))
    (rs / "s0_gates.json").write_text(json.dumps(c["s0"]))
    plan_p, draft_p = rs / "launch_plan.json", rs / "launch_plan_draft.json"
    plan_p.write_bytes(c["plan_p"].read_bytes())
    draft_p.write_bytes(c["draft_p"].read_bytes())
    for sid in ("S1", "S1b", "S2", "S3"):
        assert S5R.main(["pass", "--rs", str(rs), "--plan", str(plan_p), "--draft", str(draft_p), "--smoke-id", sid, "--rows", ",".join(c["rows_by_smoke"][sid])]) == 0, sid
    # the S5 block as the certificate test records it
    rec_p = tmp_path / "rec.json"
    for ref in c["smokes"]["S5"]["references"]:
        rec_p.write_text(json.dumps(ref))
        assert S5R.main(["reference", "--rs", str(rs), "--plan", str(draft_p), "--record", str(rec_p)]) == 0
    assert S5R.main(["window", "--rs", str(rs), "--plan", str(draft_p), "--rows", ",".join(c["rows_by_smoke"]["S5"])]) == 0
    rec_p.write_text(json.dumps(c["smokes"]["S5"]["bars"]))
    assert S5R.main(["bars", "--rs", str(rs), "--plan", str(draft_p), "--record", str(rec_p)]) == 0
    assert S5R.main(["w-gate", "--rs", str(rs), "--plan", str(plan_p), "--baseline-sha", "b" * 40]) == 0
    tree = tmp_path / "tree"
    (tree / "configs/registry").mkdir(parents=True)
    (tree / "configs/registry/decode_policy.json").write_text(c["reg"].read_text())
    (tree / "src").symlink_to(REPO / "src")
    assert S5R.main(["decode-proof", "--rs", str(rs), "--tree", str(tree)]) == 0
    certify = ["certify", "--rs", str(rs), "--plan", str(plan_p), "--draft", str(draft_p), "--tree", str(tree)]
    # no S4 PASS and no waiver file -> refused as before
    assert S5R.main(certify) == 2 and not (rs / lp.CERTIFICATE_FILE).exists()
    # a malformed waiver file -> refused by name, never issued
    (rs / lp.S4_WAIVER_FILE).write_text(json.dumps({**waiver_doc(final["W"]), "evidence": []}))
    assert S5R.main(certify) == 2 and not (rs / lp.CERTIFICATE_FILE).exists()
    # the documented waiver -> issued, sealed, WAIVED recorded, --go's predicate accepts it against the volume
    doc = waiver_doc(final["W"])
    (rs / lp.S4_WAIVER_FILE).write_text(json.dumps(doc, indent=1) + "\n")
    assert S5R.main(certify) == 0
    cert = json.loads((rs / lp.CERTIFICATE_FILE).read_text())
    assert cert["smokes"]["S4"]["status"] == "WAIVED" and cert["smokes"]["S4"]["waiver"]["document"] == doc
    assert cert["smokes"]["S4"]["waiver"]["sha256"] == lp.sha256_file(rs / lp.S4_WAIVER_FILE) and cert["smokes"]["S4"]["waiver"]["path"] == "RS/s4_waiver.json"
    assert cert["certificate_sha256"] == lp.certificate_sha256(cert)
    assert lp.go_preconditions(final, certificate=cert, registry_sha256=cert["decode_rows"]["registry_sha256"],
                               read_text=lambda p: FakeVOL.files.get(str(p)[len("/vol/"):]) if str(p).startswith("/vol/") else None) == []
    # both a derived S4 PASS (the world has the S4 rows) and the waiver file -> refused, nothing picked
    rs2 = tmp_path / "RS2"
    rs2.mkdir()
    for sid in lp.SMOKE_PASS_IDS + ("S5",):
        wp = lp.smoke_wave_path(rs2, sid)
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_text(json.dumps({"smoke_id": sid, "wave_uuid": c["waves"][sid]}))
    (rs2 / "smokes.json").write_text((rs / "smokes.json").read_text())
    (rs2 / "w_gate.json").write_text((rs / "w_gate.json").read_text())
    (rs2 / "decode_ref_proof.json").write_text((rs / "decode_ref_proof.json").read_text())
    (rs2 / "s0_gates.json").write_text(json.dumps(c["s0"]))
    (rs2 / lp.S4_WAIVER_FILE).write_text(json.dumps(doc, indent=1) + "\n")
    assert S5R.main(["pass", "--rs", str(rs2), "--plan", str(plan_p), "--draft", str(draft_p), "--smoke-id", "S4", "--rows", ",".join(c["rows_by_smoke"]["S4"])]) == 0
    with pytest.raises(ValueError, match="both a derived S4 PASS record .* exist"):
        S5R.certify(rs=rs2, plan_path=plan_p, draft_path=draft_p, tree=tree, read_text=lambda p: FakeVOL.files.get(str(p)[len("/vol/"):]) if str(p).startswith("/vol/") else None,
                    accepted_get_for=lambda sid: c["getters"][sid])
    assert S5R.main(["certify", "--rs", str(rs2), "--plan", str(plan_p), "--draft", str(draft_p), "--tree", str(tree)]) == 2
    assert not (rs2 / lp.CERTIFICATE_FILE).exists()


def test_battery_tables_read_the_waived_smoke_as_a_qualification(tmp_path):
    import battery_tables as bt
    rs = tmp_path / "RS"
    rs.mkdir()
    assert bt.load_smoke_waivers(rs) is None                                     # no certificate: nothing waived, nothing invented
    doc = waiver_doc("a" * 40)
    (rs / "launch_certificate.json").write_text(json.dumps({"schema": lp.CERTIFICATE_SCHEMA, "smokes": {"S3": {"schema": lp.SMOKE_PASS_SCHEMA, "verdict": "PASS"},
                                                                                                        "S4": lp.smoke_waiver_entry(doc, path="RS/s4_waiver.json", file_sha256="0" * 64)}}))
    got = bt.load_smoke_waivers(rs)
    assert got == {"S4": "WAIVED (user ruling D133 via orchestrator; H200:1 (D116) -- not exercised live under a completed S4)"}
    (rs / "launch_certificate.json").write_text(json.dumps({"schema": lp.CERTIFICATE_SCHEMA, "smokes": {"S3": {"schema": lp.SMOKE_PASS_SCHEMA, "verdict": "PASS"}}}))
    assert bt.load_smoke_waivers(rs) is None                                     # a certificate with every smoke PASSED renders no waiver line
    # the markdown states it when present and keeps its shape when absent (additive)
    base = {"partial": False, "inputs": {"rs": "RS", "grades": "G"}, "legs": {}, "table_status": "x", "warnings": [],
            "evidence": {"n_readable": 0, "n_arm_runs": 0, "n_admissible": 0, "n_coverage_complete": 0, "n_audit_agrees": 0, "n_identity_pass_legs": 0, "n_legs": 0,
                         "rejected_legs": [], "n_audit_ledger_backed": 0, "n_audit_ledgerless": 0, "n_audit_unqualified": 0}}
    plain = bt.render_md(base)
    assert "Smoke waivers" not in plain
    with_w = bt.render_md({**base, "smoke_waivers": got or {"S4": "WAIVED (user ruling D133 via orchestrator; H200:1)"}})
    assert "Smoke waivers (D133): S4: WAIVED (" in with_w and "a qualification, not a pass" in with_w
