"""qwen-seedrep-native-battery fg1 / F-AW: the determinism spot-check on a SAMPLED gsm8k row is a recorded qualification,
on the smoke-PASS paths too.

THE FINDING, re-validated as the first assertions: `SMOKE_CHECKS["S3"]` gates the PASS on determinism and `_smoke_check("determinism")` demanded a
byte-identical replay (`reproduces: true`, `identical == n_requests`) whatever the row's decode; the gsm8k `S3` smoke (thinking ON, the Qwen thinking
card at temperature 0.6) replayed 2 of 20 byte-identically and would have refused both `derive_smoke_pass` and `derive_smoke_pass_from_priors` --
although the carry-over path (`seedrep_carry_over.complete_verdict`) already held that a diverged replay under a VERIFIED
sampled decode is NOT APPLICABLE (a qualification), never a refusal and never a waiver.

THE FIX: the qualification is ONE function, `seedrep_carry_over.determinism_d180_relaxation` (extracted from `complete_verdict`, which still uses
it), and `seedrep_launch_plan._determinism_source` routes every determinism source of both derivations and both certify-time re-checks through
it: the decode is read from the resolved-sampler record the row's `resolved_sampler` check VERIFIED (under a prior-tree re-audit the fresh
re-derived record, cross-checked against the historical one), every client temperature > 0 qualifies, the source states the REAL counts
(`reproduces: false`, `identical` 2 of 20) plus `criterion3: not_applicable_sampled_decode`, `criterion3_ruling: D180`, `criterion3_decode`, and
the check carries the aggregated `criterion3` block the carry-over's certificate already shows. A GREEDY row keeps the byte-identical check (its
divergence refuses by name); an exemption that cannot be judged refuses by name; a tampered qualification fails at certify by name. No waiver.

Synthetic throughout: the certificate suite's fake smoke world, placeholder shas, no recorded data."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_carry_over as co  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_review_fixes as RF  # noqa: E402
from test_seedrep_review_fixes import draft, inputs, srv  # noqa: E402,F401  (module fixtures register by import)
from test_seedrep_fc11_prior_smoke_pass import W25, _derive, _problems, _stub_reaudit, world  # noqa: E402,F401  (the prior-tree world; module fixture)

MISSES = [{"arm": "none", "item_id": "gsm8k_11111111", "sample_idx": 0, "first_divergence_char": 412, "first_divergence_token": 63},
          {"arm": "none", "item_id": "gsm8k_22222222", "sample_idx": 0, "first_divergence_char": 9, "first_divergence_token": 2}]
THINKING_CARD = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
NA = lp.DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED


# ------------------------------------------------------------------------------------------------------- the same-tree world (fq2's shape)
def _world(draft, tmp_path, leg="gsm8k"):
    T = RF.TestSolR3_2_launch_certificate()
    rs = tmp_path / "RS"
    wave = RF._uuid4(7)
    wp = lp.smoke_wave_path(rs, "S3")
    wp.parent.mkdir(parents=True, exist_ok=True)
    wp.write_text(json.dumps({"smoke_id": "S3", "wave_uuid": wave}))
    rid = RF._row(draft, leg)["standup_id"]
    accepted, texts = T._smoke_world(draft, draft, rs, {"S3": wave}, {"S3": [rid]})
    getter = lp.RecordingGetter(lp.smoke_identity(draft, "S3", wave)["dicts"]["accepted"], accepted["S3"].get)
    det_path = [p for p in texts if p.endswith(f"/audits/determinism/{leg}.json")][0]
    root = det_path[: -len(f"/audits/determinism/{leg}.json")]
    return dict(rs=rs, rid=rid, texts=texts, getter=getter, wave=wave, det_path=det_path, root=root, leg=leg)


def _rebind(texts, root, key, rel, text):
    """Rewrite one producer file AND its binding under modal_evidence.json (the app names the bytes)."""
    texts[f"{root}/{rel}"] = text
    ev = json.loads(texts[f"{root}/modal_evidence.json"])
    if key == lp.DETERMINISM_EVIDENCE_KEY:
        ev[key][rel.split("/")[-1][: -len(".json")]]["sha256"] = lp._sha_text(text)
    else:
        ev[key][rel]["sha256"] = lp._sha_text(text)
    texts[f"{root}/modal_evidence.json"] = json.dumps(ev)


def _diverge(w, *, identical=2, n_requests=20):
    doc = {**json.loads(w["texts"][w["det_path"]]), "n_items": 20, "n_requests": n_requests, "n": n_requests, "identical": identical, "reproduces": identical == n_requests,
           "misses": MISSES if identical < n_requests else []}
    _rebind(w["texts"], w["root"], lp.DETERMINISM_EVIDENCE_KEY, f"audits/determinism/{w['leg']}.json", json.dumps(doc))
    return doc


def _sampler(w, temperatures):
    doc = {**json.loads(w["texts"][f"{w['root']}/{lp.RESOLVED_SAMPLER_REL}"]),
           "distinct_resolved": [{"sampler": {**THINKING_CARD, "temperature": t}, "n": 1020} for t in temperatures]}
    _rebind(w["texts"], w["root"], lp.PRODUCER_EVIDENCE_KEY, lp.RESOLVED_SAMPLER_REL, json.dumps(doc))


def _derive_same(w, draft):
    return lp.derive_smoke_pass(rs=w["rs"], plan=draft, smoke_id="S3", standup_ids=[w["rid"]], accepted_get=w["getter"],
                                read_text=lp.volume_only(lambda p_: w["texts"].get(str(p_))), derived_utc="2026-09-20T00:00:00Z")


def _problems_same(w, draft, rec, *, read_text=True):
    return lp.smoke_pass_problems(rec, smoke_id="S3", plan=draft, execution_wave_uuid=w["wave"], accepted_get=w["getter"],
                                  read_text=(lp.volume_only(lambda p_: w["texts"].get(str(p_))) if read_text else None))


def test_a_sampled_gsm8k_rows_diverged_replay_derives_s3_with_the_d180_qualification_and_certifies(draft, tmp_path):
    w = _world(draft, tmp_path)
    _diverge(w)
    _sampler(w, [0.6])
    rec = _derive_same(w, draft)
    det = rec["checks"]["determinism"]
    assert det["ok"] is True and len(det["sources"]) == 1
    s = det["sources"][0]
    assert s["path"] == w["det_path"] and s["sha256"] == lp._sha_text(w["texts"][w["det_path"]]), "the REAL bytes' digest, not the relaxed text's"
    assert (s["reproduces"], s["identical"], s["n_requests"], s["leg"]) == (False, 2, 20, "gsm8k")
    assert s["criterion3"] == NA == co.CRITERION3_NOT_APPLICABLE_SAMPLED == "not_applicable_sampled_decode" and s["criterion3_ruling"] == "D180"
    rs_path = f"{w['root']}/{lp.RESOLVED_SAMPLER_REL}"
    assert s["criterion3_decode"] == {"temperatures": [0.6], "authority": co.DECODE_AUTHORITY_HISTORICAL,
                                      "source": [{"path": rs_path, "sha256": lp._sha_text(w["texts"][rs_path]), "authority": co.DECODE_AUTHORITY_HISTORICAL}]}
    assert s["producer"] == {"writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": draft["W"], "rc": 0}
    c3 = det["criterion3"]
    assert (c3["verdict"], c3["ruling"], c3["reproduced"], c3["resent"], c3["misses"], c3["records"]) == (NA, "D180", 2, 20, MISSES, [w["det_path"]])
    assert c3["decode"]["temperatures"] == [0.6] and c3["decode"]["cross_check"] is None and "sampled decode" in c3["decode"]["why"]
    # the certificate re-derives the qualification from the volume and accepts; offline it holds the recorded shape consistent
    assert _problems_same(w, draft, rec) == []
    assert _problems_same(w, draft, rec, read_text=False) == []
    # the S3 PASS record keeps its closed top-level schema (the block lives on the check)
    assert set(rec) == set(lp.SMOKE_PASS_KEYS)


def test_a_greedy_row_keeps_the_byte_identical_check_and_an_unjudgeable_exemption_refuses_by_name(draft, tmp_path):
    w = _world(draft, tmp_path)
    _diverge(w)
    _sampler(w, [0.0])
    with pytest.raises(ValueError, match=r"2 of 20 re-sent request\(s\) reproduced byte-identically .*criterion 3 requires every one"):
        _derive_same(w, draft)
    # a mixed decode (one greedy task beside a sampled one) is greedy for the exemption's purposes: every temperature must be > 0
    _sampler(w, [0.6, 0.0])
    with pytest.raises(ValueError, match="criterion 3 requires every one"):
        _derive_same(w, draft)
    # the world's own resolved-sampler record states no client decode (`distinct_resolved` is a count there): the exemption cannot be judged -- refused by name
    w2 = _world(draft, tmp_path / "w2")
    _diverge(w2)
    with pytest.raises(ValueError, match="the D180 exemption cannot be judged.*states no client decode temperature"):
        _derive_same(w2, draft)
    # a record that REPRODUCES derives exactly as before: no qualification key on the source, no block on the check
    w3 = _world(draft, tmp_path / "w3")
    _sampler(w3, [0.6])
    rec = _derive_same(w3, draft)
    s = rec["checks"]["determinism"]["sources"][0]
    assert s["reproduces"] is True and "criterion3" not in s and "criterion3" not in rec["checks"]["determinism"]
    assert _problems_same(w3, draft, rec) == [] and _problems_same(w3, draft, rec, read_text=False) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda r, s, c: s.update(identical=20), "recomputed facts"),
    (lambda r, s, c: s.pop("criterion3"), "criterion 3 not stated"),
    (lambda r, s, c: s.update(criterion3="reproduced"), "not D180's qualification"),
    (lambda r, s, c: s["criterion3_decode"].update(temperatures=[0.0]), "not D180's qualification"),
    (lambda r, s, c: c.pop("criterion3"), "carry the D180 qualification but the check records no criterion3 block"),
    (lambda r, s, c: c["criterion3"]["decode"].update(temperatures=[0.0]), "rests on decode temperatures"),
    (lambda r, s, c: c["criterion3"].update(reproduced=20), "are not the qualified sources' sums"),
    (lambda r, s, c: c["criterion3"].update(ruling="D999"), "is not D180's"),
])
def test_a_qualification_edited_after_derivation_fails_at_certify_by_name(draft, tmp_path, mutate, needle):
    w = _world(draft, tmp_path)
    _diverge(w)
    _sampler(w, [0.6])
    rec = _derive_same(w, draft)
    bad = copy.deepcopy(rec)
    c = bad["checks"]["determinism"]
    mutate(bad, c["sources"][0], c)
    probs_online = _problems_same(w, draft, bad)
    probs_offline = _problems_same(w, draft, bad, read_text=False)
    assert any(needle in p for p in probs_online + probs_offline), (needle, probs_online, probs_offline)


def test_the_resolved_sampler_record_rewritten_after_derivation_no_longer_grounds_the_exemption(draft, tmp_path):
    """fj23's rule on this path: the decode is read from the record the resolved_sampler check VERIFIED (its recorded sha) -- a record rewritten to a
    greedy temperature (bytes changed, whatever the binding says) fails the re-hash by name, never re-grounds the qualification."""
    w = _world(draft, tmp_path)
    _diverge(w)
    _sampler(w, [0.6])
    rec = _derive_same(w, draft)
    _sampler(w, [0.0])
    probs = _problems_same(w, draft, rec)
    assert any("changed since the record was derived" in p for p in probs), probs
    assert any("cannot be judged" in p or "no longer passes on its content" in p or "recomputed facts" in p for p in probs), probs


def test_complete_verdict_still_uses_the_one_relaxation_and_its_messages_are_unchanged():
    """The carry-over's clause (c) calls the extracted function; the relaxation has one home."""
    src = (REPO / "scripts/modal/seedrep_carry_over.py").read_text()
    assert src.count("def determinism_d180_relaxation(") == 1 and "relax = determinism_d180_relaxation(text, path=path, root=root, producer=producer" in src
    assert "class D180Undeterminable(ValueError)" in src and 'CRITERION3_NOT_APPLICABLE_SAMPLED = lp.DETERMINISM_CRITERION3_NOT_APPLICABLE_SAMPLED' in src
    lsrc = (REPO / "scripts/modal/seedrep_launch_plan.py").read_text()
    for fn in ("derive_smoke_pass(", "derive_smoke_pass_from_priors(", "smoke_pass_problems(", "smoke_pass_prior_problems("):
        body = lsrc.split(f"def {fn}", 1)[1].split("\ndef ", 1)[0]
        assert "_determinism_source(" in body, fn
    assert co.CRITERION3_RULING == lp.DETERMINISM_CRITERION3_RULING == "D180"
    # the relaxation itself: a reproducing record and a non-record are None; a diverged one under a sampled verified decode relaxes; greedy is None
    root, path = "/vol/x/root", "/vol/x/root/audits/determinism/gsm8k.json"
    det = json.dumps({"leg": "gsm8k", "arms": ["none"], "n_items": 20, "n_requests": 20, "n": 20, "identical": 2, "reproduces": False, "misses": MISSES})
    rs_text = json.dumps({"verdict": "AGREES", "distinct_resolved": [{"sampler": {**THINKING_CARD}, "n": 1020}]})
    producer = {"audits/determinism/gsm8k.json": {"sha256": lp._sha_text(det), "writer": lp.PRODUCER_WRITERS["determinism"], "tree_sha": "5" * 40, "rc": 0}}
    sampler_check = {"ok": True, "sources": [{"path": f"{root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": lp._sha_text(rs_text)}]}
    read = {f"{root}/{lp.RESOLVED_SAMPLER_REL}": rs_text}.get
    relax = co.determinism_d180_relaxation(det, path=path, root=root, producer=producer, sampler_check=sampler_check, prev_row={}, evidence={}, read_text=read)
    assert relax["real_sha"] == lp._sha_text(det) and json.loads(relax["text"])["identical"] == 20 and json.loads(relax["text"])["reproduces"] is True
    assert relax["producer"]["audits/determinism/gsm8k.json"]["sha256"] == lp._sha_text(relax["text"]) and relax["diverged"]["verdict"] == NA
    assert co.determinism_d180_relaxation(json.dumps({**json.loads(det), "identical": 20, "reproduces": True}), path=path, root=root, producer=producer,
                                          sampler_check=sampler_check, prev_row={}, evidence={}, read_text=read) is None
    greedy_rs = json.dumps({"verdict": "AGREES", "distinct_resolved": [{"sampler": {**THINKING_CARD, "temperature": 0.0}, "n": 1020}]})
    greedy_check = {"ok": True, "sources": [{"path": f"{root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": lp._sha_text(greedy_rs)}]}
    assert co.determinism_d180_relaxation(det, path=path, root=root, producer=producer, sampler_check=greedy_check, prev_row={}, evidence={},
                                          read_text={f"{root}/{lp.RESOLVED_SAMPLER_REL}": greedy_rs}.get) is None
    with pytest.raises(co.D180Undeterminable, match="cannot be judged"):
        co.determinism_d180_relaxation(det, path=path, root=root, producer=producer, sampler_check=None, prev_row={}, evidence={}, read_text=read)
    with pytest.raises(co.D180Undeterminable, match="is not the bytes the producer bound"):
        co.determinism_d180_relaxation(det, path=path, root=root, producer={"audits/determinism/gsm8k.json": {**producer["audits/determinism/gsm8k.json"], "sha256": "0" * 64}},
                                       sampler_check=sampler_check, prev_row={}, evidence={}, read_text=read)


# ------------------------------------------------------------------------------------------------------- the prior-tree path (fc11's world)
@pytest.fixture
def pw(world):
    """The fc11 world with its texts snapshotted and restored: this module's tests rewrite a prior root's records in place."""
    snap = dict(world["texts"])
    yield world
    world["texts"].clear()
    world["texts"].update(snap)


def _prior_row_root(w, sid):
    return w["c"]["smokes"][sid]["rows"]


def test_a_prior_tree_sampled_row_derives_the_s3_pass_with_the_qualification_from_the_rederived_decode(pw):
    w = pw
    rows = w["c"]["rows_by_smoke"]["S3"]
    bon = [rid for rid in rows if lp.plan_row_by_id(w["draft25"], rid)["leg"] == "bon"][0]
    root = w["c"]["smokes"]["S3"]["rows"][bon]["attempt_root"]
    det_rel = "audits/determinism/bon.json"
    texts = w["texts"]
    det = {**json.loads(texts[f"{root}/{det_rel}"]), "n_items": 20, "n_requests": 40, "n": 40, "identical": 3, "reproduces": False, "misses": MISSES[:1]}
    _rebind(texts, root, lp.DETERMINISM_EVIDENCE_KEY, det_rel, json.dumps(det))
    hist = {**json.loads(texts[f"{root}/{lp.RESOLVED_SAMPLER_REL}"]), "distinct_resolved": [{"sampler": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}, "n": 400}]}
    _rebind(texts, root, lp.PRODUCER_EVIDENCE_KEY, lp.RESOLVED_SAMPLER_REL, json.dumps(hist))
    # RE-VALIDATION of the finding on this path is the failing-first: before the fix this derivation refused the sampled decode by name
    rec = _derive(w, "S3")
    det_c = rec["checks"]["determinism"]
    s = [x for x in det_c["sources"] if x["path"] == f"{root}/{det_rel}"][0]
    assert (s["reproduces"], s["identical"], s["n_requests"], s["criterion3"], s["criterion3_ruling"]) == (False, 3, 40, NA, "D180")
    assert s["criterion3_decode"]["temperatures"] == [0.7] and s["criterion3_decode"]["authority"] == co.DECODE_AUTHORITY_REAUDIT, "the decode is THIS tree's re-derived record's"
    ev_root = rec["rows"][bon]["reaudit"]["evidence_root"]
    assert s["criterion3_decode"]["source"] == [{"path": f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": lp._sha_text(texts[f"{ev_root}/{lp.RESOLVED_SAMPLER_REL}"]),
                                                 "authority": co.DECODE_AUTHORITY_REAUDIT}]
    c3 = det_c["criterion3"]
    assert (c3["reproduced"], c3["resent"], c3["records"]) == (3, 40, [f"{root}/{det_rel}"]) and c3["decode"]["cross_check"] == {"historical": [
        {"path": f"{root}/{lp.RESOLVED_SAMPLER_REL}", "sha256": lp._sha_text(texts[f"{root}/{lp.RESOLVED_SAMPLER_REL}"])}], "agrees": True}
    others = [x for x in det_c["sources"] if x["path"] != f"{root}/{det_rel}"]
    assert all("criterion3" not in x and x["reproduces"] is True for x in others), "any other row's reproducing record is unchanged"
    assert det_c["criterion3"]["records"] == [f"{root}/{det_rel}"] and len(det_c["sources"]) == len(rows)
    assert _problems(w, rec, "S3") == []
    assert _problems(w, rec, "S3", read_text=None) == []
    # a tampered historical record (greedy bytes under the old binding) cannot re-ground the exemption at certify: refused by name (fj23)
    texts[f"{root}/{lp.RESOLVED_SAMPLER_REL}"] = json.dumps({**hist, "distinct_resolved": [{"sampler": {"temperature": 0.0}, "n": 400}]})
    probs = _problems(w, rec, "S3")
    assert probs and any("tampered evidence never grounds" in p or "changed since the record was derived" in p or "no longer passes" in p for p in probs), probs


def test_a_prior_tree_greedy_row_keeps_the_byte_identical_refusal(pw):
    w = pw
    rows = w["c"]["rows_by_smoke"]["S3"]
    bon = [rid for rid in rows if lp.plan_row_by_id(w["draft25"], rid)["leg"] == "bon"][0]
    root = w["c"]["smokes"]["S3"]["rows"][bon]["attempt_root"]
    det_rel = "audits/determinism/bon.json"
    det = {**json.loads(w["texts"][f"{root}/{det_rel}"]), "n_items": 20, "n_requests": 40, "n": 40, "identical": 3, "reproduces": False, "misses": MISSES[:1]}
    _rebind(w["texts"], root, lp.DETERMINISM_EVIDENCE_KEY, det_rel, json.dumps(det))
    hist = {**json.loads(w["texts"][f"{root}/{lp.RESOLVED_SAMPLER_REL}"]), "distinct_resolved": [{"sampler": {"temperature": 0.0}, "n": 400}]}
    _rebind(w["texts"], root, lp.PRODUCER_EVIDENCE_KEY, lp.RESOLVED_SAMPLER_REL, json.dumps(hist))
    with pytest.raises(ValueError, match=r"3 of 40 re-sent request\(s\) reproduced byte-identically .*criterion 3 requires every one"):
        _derive(w, "S3")
