"""A review finding on commit 493831ea — the same-call re-execution that ACCEPTS over results it never drew: the accepted
attempt must carry the request pipeline of every elected result it relies on, verified and manifested, so the request-evidence
audit binds the store rows from the pulled root alone. Everything synthetic; the request client and the audit are loaded by
path (``_seedrep_testlib.peer_file``; the tests SKIP by name when neither location has them)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402
import test_seedrep_sol_r2_fixes as R2  # noqa: E402  (the loopback engine proxy + the ledger / engine-log synthesis)

L = T.load_ladder()
RR = T.load_rung_redraw()

W7, WAVE = T.W7, T.WAVE
CALL = "fc-C"


def _rung_attempt_rel(view: str, shard: str, attempt_id: str) -> str:
    """A rung attempt's root in the ONE spelling (`L.attempt_root_for` == `lp.attempt_root_for`): `…/<view>/rungs/<cap>/shard<NN>/attempts/<id>`."""
    return L.attempt_root_for(W7, leg="pasttense", view=view, stage="8192", shard=shard, attempt_id=attempt_id)


def _pipeline_into(root: Path, proxy, C) -> None:
    """What the container's PROXY and ENGINE left in an attempt root for the requests the proxy saw (its TreeSyncer shipped them
    before the container was lost): the usage ledger at the root, the engine log under `serve/`."""
    rows, lines = R2._ledger_and_engine_log(proxy, C)
    with (root / L.LEDGER_FILE).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    (root / "serve").mkdir(parents=True, exist_ok=True)
    with (root / "serve" / "vllm_server.log").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _world(tmp_path, *, view="s1", arm="none", seeds=(1,)):
    """An audit base-root fixture (g0 at cap per seed slot) + the CLOSED 8192 static queue of its at-cap rows on a LocalVolume."""
    T.load_request_evidence_client()
    ac, rea, fx = T.load_ws3g_audit()
    from sisteer.capeval.tinycap import contracts as C

    stores = tmp_path / "stores"
    pin = ("a" * 64) if view == "arditi" else None
    base = fx.make_root(stores, leg="pasttense", view=view, arms=(arm,), arditi_pin=pin)
    shards = {s: fx.write_static_store(base, family="pasttense", arm=arm, s=s, goals=["g0", "g1"], at_cap_goals=("g0",)) for s in seeds}
    base.flush_pipeline()
    spec = L.QueueSpec.build(W7, WAVE, "pasttense", view, "static", 8192, ["0"])
    vol_root = tmp_path / "vol"
    vol, kv = L.LocalVolume(vol_root), T.kvs(L)
    records = [L.static_queue_record(spec, arm=arm, s=s, family="pasttense", row=row, base_store=str(sh / "responses.jsonl"),
                                     producer=spec.producers[0], producer_attempt_id="a1", vmodel=f"petri__{arm}", sampling=T.PRESET)
               for s, sh in shards.items() for row in ac.read_jsonl(sh / "responses.jsonl") if row["finish_reason"] == "length"]
    T.close_queue(L, vol, spec, records, kv["accepted"])
    return dict(base=base, shards=shards, spec=spec, vol=vol, vol_root=vol_root, kv=kv, records=records, stores=stores, pin=pin,
                view=view, arm=arm, seeds=seeds, peers=(ac, rea, fx, C))


def _execution(w, *, attempt_id, call_id, shard, attempt_root, evidence_dir, adopt_into, draw, **kw):
    """One consumer execution with its request evidence at ``evidence_dir`` (the client, built directly)."""
    ev = None if evidence_dir is None else RR.RungEvidence(evidence_dir, leg="pasttense", attempt_id=attempt_id)
    tally = RR.consume(w["vol"], accepted_kv=w["kv"]["accepted"], claims_kv=w["kv"]["claims"], results_kv=w["kv"]["results"], spec=w["spec"],
                       attempt_id=attempt_id, call_id=call_id, draw=draw(ev), shard=shard, attempt_root=attempt_root, adopt_into=adopt_into,
                       evidence_dir=evidence_dir, sleep=kw.pop("sleep", lambda s: None), log=kw.pop("log", lambda m: None), **kw)
    return tally, ev


def _predecessor(w, *, attempt_id="att-A", call_id=CALL, shard="0", pipeline=True):
    """Execution A: draws everything through the loopback engine proxy with its evidence under ITS root on the volume, elects, and
    is LOST before acceptance (nothing is finished). Its proxy's ledger and engine log are shipped into its root when ``pipeline``."""
    rel = _rung_attempt_rel(w["view"], shard or "0", attempt_id)
    root = w["vol_root"] / rel
    root.mkdir(parents=True, exist_ok=True)
    proxy = R2._engine_proxy()
    try:
        tally, ev = _execution(w, attempt_id=attempt_id, call_id=call_id, shard=shard, attempt_root=rel, evidence_dir=root, adopt_into=root,
                               draw=lambda ev: RR.http_draw_fn(proxy.url, evidence=ev))
    finally:
        proxy.close()
    assert tally["drawn"] == tally["won"] == len(w["records"]) and tally["adopted"] == [] and tally["covered"]
    if pipeline:
        _pipeline_into(root, proxy, w["peers"][3])
    return rel, root, tally


def _successor_root(w, *, attempt_id=None):
    """Execution B's attempt root = a rung-root fixture (ATTEMPT.json, serve dir) with an engine log of its own but no
    requests — B drew nothing."""
    ac, rea, fx, C = w["peers"]
    rung = fx.make_root(w["stores"], leg="pasttense", view=w["view"], arms=(w["arm"],), stage="8192", arditi_pin=w["pin"])
    (rung.serve / "vllm_server.log").write_text("", encoding="utf-8")
    return rung


def _write_rung_store(w, rung) -> None:
    """The rung store per seed slot in escalate_static's emission from the ELECTED materialised records (as the r2 test does)."""
    ac, rea, fx, C = w["peers"]
    esc = ac.escalate_static_module()
    for s, shard in w["shards"].items():
        elected = L.load_elected(w["vol"], w["kv"]["results"], w["spec"], L.arm_s(w["arm"], s))
        base_rows, base_prog = ac.read_jsonl(shard / "responses.jsonl"), ac.read_jsonl(shard / "progress.jsonl")
        out_rows, regen_per_goal, n_after = [], {}, 0
        for b in base_rows:
            row = dict(b)
            row["escalated_from_cap"] = 4096
            src_cap = int(esc.row_source_cap(b, store_cap=4096, where=str(shard)))
            el = elected.get(L.item_key("static", "pasttense", b["goal_id"], int(b["sample_idx"])))
            if el is None:
                row["regenerated"], row["source_cap"] = False, src_cap
                n_after += int(bool(b.get("at_cap")))
            else:
                r = el.doc["record"]
                row.update(response=r["text"], decode={**b["decode"], "cap": 8192}, finish_reason=r["finish_reason"], completion_tokens=r["completion_tokens"],
                           prompt_tokens=r["prompt_tokens"], max_tokens=8192, at_cap=r["finish_reason"] == "length", regenerated=True, source_cap=8192,
                           source_completion_tokens=b.get("completion_tokens"), source_finish_reason=b.get("finish_reason"), source_draw_cap=src_cap,
                           boundary_landing=int(r["completion_tokens"]) < src_cap, prefix_extension=True, identical=False, first_divergence_char=None,
                           first_divergence_token=None, request_id=r[RR.ROW_REQUEST_ID_FIELD], completion_id=r[RR.ROW_COMPLETION_ID_FIELD])
                regen_per_goal[b["goal_id"]] = regen_per_goal.get(b["goal_id"], 0) + 1
                n_after += int(row["at_cap"])
            esc.row_source_cap(row, store_cap=8192, where=str(shard))
            out_rows.append(row)
        rdir = rung.path / "exp" / fx._private_run(w["view"], 8192).format(s=s) / "pasttense" / w["arm"] / "shard00"
        progress = [{**p, "n_target_forwards": regen_per_goal.get(p["goal_id"], 0), "n_regenerated": regen_per_goal.get(p["goal_id"], 0),
                     "n_copied": int(p.get("n_variants", 0)) - regen_per_goal.get(p["goal_id"], 0)} for p in base_prog]
        block_to = {**base_rows[0]["decode"], "cap": 8192}
        fx._jsonl(rdir / "responses.jsonl", out_rows)
        fx._jsonl(rdir / "progress.jsonl", progress)
        fx._json(rdir / "served_target.json", {"target_engine": "served", "virtual_model": f"petri__{w['arm']}", "cell_id": w["arm"],
                                                "manifest_stage": "final", "decode": block_to})
        n_regen = sum(regen_per_goal.values())
        fx._json(rdir / "escalation.json", {"attack": "pasttense", "defense": w["arm"], "cap_from": 4096, "cap_to": 8192, "n_rows": len(out_rows),
                                             "n_at_cap_before": sum(1 for r in base_rows if r.get("at_cap")), "n_regenerated": n_regen,
                                             "n_at_cap_after": n_after, "n_copied": len(out_rows) - n_regen,
                                             "n_rows_by_source_cap": {str(c): sum(1 for r in out_rows if r["source_cap"] == c)
                                                                      for c in sorted({r["source_cap"] for r in out_rows})},
                                             "seed_base": s, "trial": int(block_to["trial"]), "arm": block_to["arm"], "decode": block_to,
                                             "re_draw_not_continuation": True})
    fx._json(rung.path / ac.COMPLETE_FILE, {"written_utc": "2026-01-01T02:00:00Z", "attempt_id": rung.attempt_id})


def _lines(p: Path) -> list[str]:
    return [ln for ln in p.read_text(encoding="utf-8").split("\n") if ln.strip()] if p.is_file() else []


# =========================================================================== #
# the finding, end to end: A elects and is lost; B (same call, new attempt) accepts — with A's evidence, verified and manifested
# =========================================================================== #


def test_sol_r3_a_same_call_re_execution_adopts_its_predecessors_evidence_and_ws3gs_real_audit_binds_every_row(tmp_path):
    w = _world(tmp_path)
    ac, rea, fx, C = w["peers"]
    a_rel, a_root, ta = _predecessor(w)                                   # A: 3 requests? no — 2 at-cap items drawn, elected, then LOST
    rung = _successor_root(w)
    b_rel = _rung_attempt_rel("s1", "0", rung.attempt_id)
    tb, evb = _execution(w, attempt_id=rung.attempt_id, call_id=CALL, shard="0", attempt_root=b_rel, evidence_dir=rung.path, adopt_into=rung.path,
                         draw=lambda ev: (lambda r: pytest.fail("B re-drew an elected item")))
    # B drew nothing, relied on A's two elected results, and ADOPTED their request pipeline
    assert (tb["drawn"], tb["already_elected"], tb["closed"], tb["covered"], tb["foreign_results"], tb["adoption_skipped"]) == (0, 2, True, True, 0, 0)
    (ad,) = tb["adopted"]
    assert (ad["attempt_id"], ad["n_items"], ad["manifest"]) == ("att-A", 2, "adopted_from/att-A/adoption.json") and len(ad["sha256"]) == 64
    elected_values = {v["request_id"]: v for k, v in w["kv"]["results"].items()}
    assert all(v["attempt_id"] == "att-A" and v["call_id"] == CALL and v["shard"] == "0" and v["attempt_root"] == a_rel for v in elected_values.values())
    # B's root now holds EXACTLY A's covering lines in its own four files (the ones the audit locates by name, exactly one each)
    recs, resp = _lines(rung.path / L.EVIDENCE_REQUEST_FILE), _lines(rung.path / L.EVIDENCE_RESPONSE_FILE)
    ledger, eng = _lines(rung.path / L.LEDGER_FILE), _lines(rung.serve / "vllm_server.log")
    assert {json.loads(r)["request_id"] for r in recs} == set(elected_values) and len(recs) == len(resp) == len(ledger) == len(eng) == 2
    assert {json.loads(r)["extra"]["attempt_id"] for r in recs} == {"att-A"}                     # the records stay A's, verbatim
    assert all(any(C.is_engine_id_of(L._RECEIVED_RE.search(ln).group(1), v["completion_id"]) for ln in eng) for v in elected_values.values())
    man = json.loads((rung.path / "adopted_from" / "att-A" / "adoption.json").read_text())
    assert man["schema"] == "seedrep_adopted_evidence/1" and man["source"] == {"attempt_id": "att-A", "attempt_root": a_rel}
    assert man["adopted_by"] == {"attempt_id": rung.attempt_id, "call_id": CALL, "shard": "0"}
    assert {it["request_id"] for it in man["items"]} == set(elected_values) and man["appended_now"] == {"request_evidence": 2, "response_evidence": 2,
                                                                                                        "usage_ledger": 2, "engine_lines": 2}
    for kind, name in (("request", L.EVIDENCE_REQUEST_FILE), ("response", L.EVIDENCE_RESPONSE_FILE), ("ledger", L.LEDGER_FILE)):
        assert man["source_files"][kind]["sha256"] == L.sha256_hex((a_root / name).read_bytes())          # the source, hashed as read
    assert man["source_files"]["engine_log"]["sha256"] == L.sha256_hex((a_root / "serve" / "vllm_server.log").read_bytes())
    idx = json.loads((rung.path / "adopted_from" / "ADOPTED.json").read_text())
    assert [e["attempt_id"] for e in idx["adoptions"]] == ["att-A"] and idx["adoptions"][0]["sha256"] == ad["sha256"]
    # the declared pull inputs ONLY: B's root (accepted) + the base root — A's root is never pulled — and the REAL audit binds every row
    _write_rung_store(w, rung)
    (v,) = R2._real_audit(rung, w["base"], w["peers"], tmp_path / "audits")
    assert v["verdict"] == "AGREES", v["problem"]
    assert (v["n_draws"], v["n_draws_bound"], v["n_records_task"], v["checks"]["row_binding"]["n_copied_forward"]) == (4, 4, 2, 2)
    # idempotent: a re-entered B (same attempt id) adopts again and appends NOTHING twice — the audit would refuse a duplicated request id
    before = {p: p.read_bytes() for p in (rung.path / L.EVIDENCE_REQUEST_FILE, rung.path / L.EVIDENCE_RESPONSE_FILE, rung.path / L.LEDGER_FILE,
                                            rung.serve / "vllm_server.log")}
    tb2, _ = _execution(w, attempt_id=rung.attempt_id, call_id=CALL, shard="0", attempt_root=b_rel, evidence_dir=rung.path, adopt_into=rung.path,
                        draw=lambda ev: (lambda r: pytest.fail("drew")))
    assert tb2["adopted"][0]["n_items"] == 2 and {p: p.read_bytes() for p in before} == before
    man2 = json.loads((rung.path / "adopted_from" / "att-A" / "adoption.json").read_text())
    assert man2["appended_now"] == {"request_evidence": 0, "response_evidence": 0, "usage_ledger": 0, "engine_lines": 0} and man2["copied_sha256"] == man["copied_sha256"]
    (v2,) = R2._real_audit(rung, w["base"], w["peers"], tmp_path / "audits2")
    assert v2["verdict"] == "AGREES", v2["problem"]


def test_sol_r3_known_bad_a_predecessor_whose_evidence_is_missing_or_torn_cannot_be_adopted_and_the_successor_refuses(tmp_path):
    w = _world(tmp_path)
    ac, rea, fx, C = w["peers"]
    a_rel, a_root, _ = _predecessor(w)
    rung = _successor_root(w)
    b_rel = _rung_attempt_rel("s1", "0", rung.attempt_id)
    run_b = lambda **kw: _execution(w, attempt_id=rung.attempt_id, call_id=CALL, shard="0", attempt_root=b_rel, evidence_dir=rung.path,  # noqa: E731
                                    adopt_into=rung.path, draw=lambda ev: (lambda r: pytest.fail("drew")), adopt_retries=0, **kw)
    # KNOWN-BAD 1: A's request evidence never reached the volume -> B refuses; nothing is written into B's root, nothing accepted
    saved = (a_root / L.EVIDENCE_REQUEST_FILE).read_bytes()
    (a_root / L.EVIDENCE_REQUEST_FILE).unlink()
    with pytest.raises(L.LadderRefusal, match="cannot be adopted.*no task request record"):
        run_b()
    assert not (rung.path / L.EVIDENCE_REQUEST_FILE).exists() and not (rung.path / "adopted_from").exists() and not (rung.path / L.LEDGER_FILE).exists()
    (a_root / L.EVIDENCE_REQUEST_FILE).write_bytes(saved)
    # KNOWN-BAD 2: the engine log lacks the completion's `Received request` line
    log_p = a_root / "serve" / "vllm_server.log"
    saved_log = log_p.read_text()
    log_p.write_text(saved_log.split("\n", 1)[1])
    with pytest.raises(L.LadderRefusal, match="matches 0 engine-log"):
        run_b()
    log_p.write_text(saved_log)
    # KNOWN-BAD 3: a ledger row whose body digest is not the record's (two records of one wire)
    rows = [json.loads(ln) for ln in _lines(a_root / L.LEDGER_FILE)]
    (a_root / L.LEDGER_FILE).write_text("".join(json.dumps({**r, "body_sha256": "0" * 64}) + "\n" for r in rows))
    with pytest.raises(L.LadderRefusal, match="body_sha256 differs"):
        run_b()
    (a_root / L.LEDGER_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    # KNOWN-BAD 4: the source names another completion for the request than the elected result does
    (a_root / L.LEDGER_FILE).write_text("".join(json.dumps({**r, "completion_id": "chatcmpl-other"}) + "\n" for r in rows))
    with pytest.raises(L.LadderRefusal, match="authoritative completion"):
        run_b()
    (a_root / L.LEDGER_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    # KNOWN-BAD 5: B's own root has no engine log to merge into -> refused by name
    (rung.serve / "vllm_server.log").unlink()
    with pytest.raises(L.LadderRefusal, match="no vllm_server.log under the attempt root"):
        run_b()
    (rung.serve / "vllm_server.log").write_text("")
    # a predecessor still SYNCING: its ledger lands after B's first read -> adopted on the retry (bounded)
    led = (a_root / L.LEDGER_FILE).read_bytes()
    (a_root / L.LEDGER_FILE).unlink()
    waits = []

    def arrives(s):
        waits.append(s)
        (a_root / L.LEDGER_FILE).write_bytes(led)
    tb, _ = _execution(w, attempt_id=rung.attempt_id, call_id=CALL, shard="0", attempt_root=b_rel, evidence_dir=rung.path, adopt_into=rung.path,
                       draw=lambda ev: (lambda r: pytest.fail("drew")), adopt_retries=2, poll_s=7.0, sleep=arrives)
    assert waits == [7.0] and tb["adopted"][0]["n_items"] == 2 and len(_lines(rung.path / L.LEDGER_FILE)) == 2


# =========================================================================== #
# whose results are mine to adopt: same call, or same shard; another shard's are its own; unknown provenance refuses
# =========================================================================== #


def test_sol_r3_adoption_rules_same_shard_other_call_adopts_another_shard_is_foreign_unknown_slot_refuses_and_a_smoke_skips(tmp_path):
    w = _world(tmp_path)
    # a fresh spawn of the SAME shard after the predecessor's call went terminal: adopted (the slot binds them)
    a_rel, a_root, _ = _predecessor(w, attempt_id="att-old", call_id="fc-OLD", shard="0")
    rung = _successor_root(w)
    tb, _ = _execution(w, attempt_id=rung.attempt_id, call_id="fc-NEW", shard="0", attempt_root=_rung_attempt_rel("s1", "0", rung.attempt_id),
                       evidence_dir=rung.path, adopt_into=rung.path, draw=lambda ev: (lambda r: pytest.fail("drew")))
    assert tb["adopted"][0]["attempt_id"] == "att-old" and tb["foreign_results"] == 0 and len(_lines(rung.path / L.EVIDENCE_REQUEST_FILE)) == 2
    # a smoke (no evidence side): the adoption is skipped and counted, never attempted
    ts, _ = _execution(w, attempt_id="smoke-B", call_id="fc-S", shard="0", attempt_root=None, evidence_dir=None, adopt_into=None,
                       draw=lambda ev: (lambda r: pytest.fail("drew")))
    assert ts["adopted"] == [] and ts["adoption_skipped"] == 1 and ts["covered"]
    # evidence ON but no attempt root to merge into: refused by name (the caller passes --attempt-root-local, or ATTEMPT.json marks the root)
    with pytest.raises(L.LadderRefusal, match="pass --attempt-root-local"):
        _execution(w, attempt_id="B-noroot", call_id="fc-N", shard="0", attempt_root=None, evidence_dir=tmp_path / "ev", adopt_into=None,
                   draw=lambda ev: (lambda r: pytest.fail("drew")))

    # ANOTHER shard's live / dead attempt elected the results: not this slot's to adopt (its own accepted root carries them)
    w2 = _world(tmp_path / "two")
    _predecessor(w2, attempt_id="att-r1", call_id="fc-R1", shard="1", pipeline=False)
    rung2 = _successor_root(w2)
    tf, _ = _execution(w2, attempt_id=rung2.attempt_id, call_id="fc-R0", shard="0", attempt_root=_rung_attempt_rel("s1", "0", rung2.attempt_id),
                       evidence_dir=rung2.path, adopt_into=rung2.path, draw=lambda ev: (lambda r: pytest.fail("drew")))
    assert (tf["adopted"], tf["foreign_results"], tf["covered"]) == ([], 2, True) and not (rung2.path / L.EVIDENCE_REQUEST_FILE).exists()

    # KNOWN-BAD: a claimant whose slot cannot be told (no shard) and whose call is not ours -> refused, never silently adopted or ignored
    w3 = _world(tmp_path / "three")
    _predecessor(w3, attempt_id="att-noshard", call_id="fc-X", shard=None, pipeline=False)
    rung3 = _successor_root(w3)
    with pytest.raises(L.LadderRefusal, match="slot is unknown"):
        _execution(w3, attempt_id=rung3.attempt_id, call_id="fc-Y", shard="0", attempt_root=_rung_attempt_rel("s1", "0", rung3.attempt_id),
                   evidence_dir=rung3.path, adopt_into=rung3.path, draw=lambda ev: (lambda r: pytest.fail("drew")))


# =========================================================================== #
# the pieces: the claimant's provenance, the inventory listing, the root marker
# =========================================================================== #


def test_sol_r3_claimant_provenance_rides_the_document_and_the_elected_value_and_the_inventory_lists_adoptions(tmp_path):
    shared, kv = L.SharedVolume(), T.kvs(L)
    q8 = T.static_spec(L, shards=("0",))
    c = T.Container(L, shared, q8, "0", "a1")
    rec = c.enqueue_row(T.static_row("g0", 1, seed=7))
    rv = shared.snapshot("rung-1")
    record = {"text": "t", "finish_reason": "stop", "completion_tokens": 3, "prompt_tokens": 1, "max_tokens": 8192, "sampling_seed": 7}
    path, doc = L.write_materialised(rv, q8, queue_record=rec, attempt_id="rung-1", call_id="fc-1", record=record, request_id="rq", completion_id="cmpl",
                                     shard="0", attempt_root="seedrep/x/attempts/rung-1")
    assert doc["claimant"] == {"attempt_id": "rung-1", "call_id": "fc-1", "shard": "0", "attempt_root": "seedrep/x/attempts/rung-1"}
    rv.commit()
    assert L.elect_result(kv["results"], q8, doc, path)
    assert kv["results"].get(doc["key"]) | {"path": None} == {"attempt_id": "rung-1", "path": None, "text_sha256": doc["text_sha256"], "request_id": "rq",
                                                             "completion_id": "cmpl", "call_id": "fc-1", "shard": "0", "attempt_root": "seedrep/x/attempts/rung-1"}
    # a smoke that names neither leaves the claimant as before (additive)
    _, doc2 = L.write_materialised(rv, q8, queue_record=rec, attempt_id="rung-2", call_id="fc-2", record=record, request_id="rq2", completion_id="c2")
    assert doc2["claimant"] == {"attempt_id": "rung-2", "call_id": "fc-2"}
    # the next queue's inventory lists the adoptions the consumer made
    q16 = q8.next_spec(["0"])
    prod = L.Producer(rv, q16, q16.producers[0], "rung-1", L.producer_attempt_root(q16.producers[0], "rung-1"))
    prod.adopted = [{"attempt_id": "att-A", "n_items": 2, "manifest": "adopted_from/att-A/adoption.json", "sha256": "f" * 64}]
    inv = prod.write_inventory()
    assert inv["adopted"] == prod.adopted and json.loads(rv.read_bytes(prod.inventory_path))["adopted"][0]["attempt_id"] == "att-A"
    # the attempt-root marker: the nearest ancestor holding ATTEMPT.json (the evidence dir sits two levels below it)
    root = tmp_path / "attempt"
    (root / "exp" / "request_evidence").mkdir(parents=True)
    assert L.attempt_root_marker(root / "exp" / "request_evidence") is None
    (root / "ATTEMPT.json").write_text("{}")
    assert L.attempt_root_marker(root / "exp" / "request_evidence") == root.resolve() and L.attempt_root_marker(root) == root.resolve()
    # the CLI parses the three new flags and the marker default reaches consume through main's wiring
    a = RR.build_parser().parse_args(["consume", "--vol-root", "/vol", "--W7", W7, "--wave-uuid", WAVE, "--leg", "pasttense", "--view", "s1", "--kind", "static",
                                      "--cap-next", "8192", "--producers", "0", "--attempt-id", "x", "--call-id", "y", "--proxy-url", "http://h/v1", "--width", "2",
                                      "--shard", "0", "--attempt-root-local", "/workspace/attempt", "--adopt-retries", "5"])
    assert (a.shard, a.attempt_root_local, a.adopt_retries) == ("0", "/workspace/attempt", 5)
