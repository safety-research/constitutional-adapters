"""The four drivers' ``--from-materialised`` assembly seams: with the
base store COMPLETE and the queue CLOSED, the EXISTING driver computes its eligible set exactly as today and takes
each item's elected, VERIFIED materialised draw — a missing item is drawn by the assembly itself, an extra one is
ignored with a note, a mismatching elected row is REFUSED and re-drawn — and the output is byte-identical to the
non-streaming path on the same base store. Without the flags every driver is byte-identical to before (their own
suites pin that). Known-bad inputs (an open queue, a torn materialised file, a row bound to another base row or a
superseded producer) fail closed.

Static reuses the sampled static lane's builders (``tests/test_sampled_static.py``), controls the offline world of
``tests/test_sampled_controls.py`` (imported fixtures), misalign the runner's ``FakeHttp``, GSM8K the Inspect rung
adapter's synthetic store. Everything synthetic."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import _seedrep_testlib as T  # noqa: E402
import test_sampled_static as tss  # noqa: E402  (the sampled static lane's builders)
from test_sampled_controls import (  # noqa: E402,F401  (fixtures + helpers of the offline controls world)
    AT_CAP_64,
    SEED_BASE as CONTROLS_SEED_BASE,
    _rung_argv,
    ITEM_OF_PROMPT,
    META,
    SPOOL,
    _canned_draw,
    _cell_dir,
    _meta_rows,
    _spool_lines,
    ladder,
    sampled,
    world,
)
from test_misalign_runner import CELLS, ITEMS, REV, FakeHttp  # noqa: E402
import test_inspect_rung as tir  # noqa: E402

import served_capability_run as srun  # noqa: E402
from sisteer.capeval.misalign import runner as R  # noqa: E402
from sisteer.capeval.tiers import inspect_rung as IR  # noqa: E402
from sisteer.eval import sampled_controls as scp  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402

L = T.load_ladder()
RR = T.load_rung_redraw()


def _kv(purpose):
    return L.MemoryKV(L.dict_name(T.W7, T.WAVE, purpose))


def _close_queue(vol, spec, shard, aid, records_by_key, kv_accepted):
    return T.close_queue(L, vol, spec, list(records_by_key.values()), kv_accepted, shard=shard, aid=aid)


def _materialise(vol, spec, rec, *, record, text, response=None, attempt="rung-1", results=None):
    assert text == L.record_text(record)
    return T.materialise_elect(L, vol, spec, rec, record, results, attempt=attempt, response=response)


def _bytes(shard_dir: Path, names):
    return {n: (shard_dir / n).read_bytes() for n in names}


# --------------------------------------------------------------------------- #
# seam 1 — static: byte-identical to the non-streaming rung                     #
# --------------------------------------------------------------------------- #


def _static_world(tmp_path, monkeypatch, chat):
    es = tss._load_script("escalate_static")
    tss._base_store(tmp_path)
    target, _ = tss._wire_escalate(monkeypatch, es, chat)
    cfg_path = tss._esc_config(tmp_path)
    return es, target, cfg_path


def _esc_json(shard):
    esc = json.loads((shard / "escalation.json").read_text())
    # the two argv paths differ by private root; written_at is a clock; n_target_forwards counts ENGINE draws only
    return {k: v for k, v in esc.items() if k not in ("written_at", "n_target_forwards", "config", "routing_table")}


def _static_ladder(tmp_path, base_rows, at_cap_keys, *, records=None, extra=None):
    """A CLOSED static 8192 queue on a LocalVolume under tmp: the base's at-cap rows enqueued by one accepted
    producer, the elected materialised draws = ``records`` (item key -> turn record), optional ``extra`` items."""
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = T.static_spec(L, shards=("0",))
    acc, res = _kv("accepted"), _kv("results")
    ref = spec.producers[0]
    by_key = {}
    for r in base_rows:
        if (r["goal_id"], r["sample_idx"]) in at_cap_keys or (extra and (r["goal_id"], r["sample_idx"]) in extra):
            rr = dict(r)
            if (r["goal_id"], r["sample_idx"]) not in at_cap_keys:
                rr["finish_reason"] = "length"; rr["at_cap"] = True          # an "extra": queued though not eligible here
            rec = L.static_queue_record(spec, arm="none", s=tss.SEED_BASE, family="pasttense", row=rr, base_store="exp/base/shard00/responses.jsonl",
                                        producer=ref, producer_attempt_id="a1", vmodel="petri__base", sampling=tss.QWEN)
            by_key[rec["item_key"]] = rec
    _close_queue(vol, spec, "0", "a1", by_key, acc)
    if records:
        for ik, rec_ in records.items():
            _materialise(vol, spec, by_key[ik], record=rec_, text=rec_["text"], results=res)
    snap = tmp_path / "results.json"
    L.snapshot_kv(res, snap)
    return vol_root / spec.ladder_root, snap, spec, by_key, res, vol


def _base_rows(tmp_path):
    return (tss._rows(tmp_path / tss.BASE_RUN / "pasttense" / "none" / "shard00" / "responses.jsonl")
            + tss._rows(tmp_path / tss.BASE_RUN / "pasttense" / "none" / "shard01" / "responses.jsonl"))


AT_CAP = {("g0", 1), ("g1", 1), ("g3", 0)}


def _reference(tmp_path, monkeypatch, capsys):
    """The non-streaming rung on the same base store: the reference bytes AND the exact records the engine returned
    (drawn from the same deterministic stub, so the materialised draws are what the engine would have produced)."""
    chat = tss._StubChat(decide=tss._redraw)
    es, target, cfg_path = _static_world(tmp_path, monkeypatch, chat)
    assert es.main(tss._esc_argv(tmp_path, cfg_path)) == 0
    capsys.readouterr()
    shard = tmp_path / tss.ESC_RUN / "pasttense" / "none" / "shard00"
    ref = _bytes(shard, ["responses.jsonl", "progress.jsonl"]), _esc_json(shard)
    rows = _base_rows(tmp_path)
    records = {}
    fresh = st.ServedChatTarget(tss._StubChat(decide=tss._redraw), virtual_model="petri__base", cell_id="none",
                                route_info={"virtual_model": "petri__base"})
    for r in rows:
        if (r["goal_id"], r["sample_idx"]) in AT_CAP:
            (rec,) = fresh.generate_detailed([r["attack_prompt"]], max_new_tokens=8192, decode=tss.QWEN, seeds=[r["sampling_seed"]])
            records[L.item_key("static", "pasttense", r["goal_id"], r["sample_idx"])] = rec
    return es, ref, rows, records


class _Refusing(tss._StubChat):
    """An engine that must NOT be reached for the given prompts."""

    def __init__(self, forbidden, decide):
        super().__init__(decide=decide)
        self.forbidden = set(forbidden)

    def __call__(self, payload):
        prompt = payload["messages"][-1]["content"]
        assert prompt not in self.forbidden, f"the engine was asked for {prompt!r}, which had an elected materialised draw"
        return super().__call__(payload)


def test_static_assembly_is_byte_identical_to_the_non_streaming_rung_and_never_reaches_the_engine(tmp_path, monkeypatch, capsys):
    es, (ref_bytes, ref_esc), rows, records = _reference(tmp_path, monkeypatch, capsys)
    ladder_root, snap, spec, by_key, res, vol = _static_ladder(tmp_path, rows, AT_CAP, records=records)
    # a second private root with the same base store; an engine that refuses every eligible prompt
    root2 = tmp_path / "second"
    tss._base_store(root2)
    chat = _Refusing({"g0/1", "g1/1", "g3/0"}, decide=tss._redraw)
    target, _ = tss._wire_escalate(monkeypatch, es, chat)
    cfg2 = tss._esc_config(root2)
    argv = tss._esc_argv(root2, cfg2, **{"--from-materialised": str(ladder_root), "--results": str(snap)})
    assert es.main(argv) == 0
    out = capsys.readouterr()
    shard2 = root2 / tss.ESC_RUN / "pasttense" / "none" / "shard00"
    assert _bytes(shard2, ["responses.jsonl", "progress.jsonl"]) == ref_bytes          # byte-identical rows
    assert _esc_json(shard2) == ref_esc
    esc2 = json.loads((shard2 / "escalation.json").read_text())
    assert esc2["n_target_forwards"] == 0 and chat.payloads == []                       # nothing drawn here
    manifest = json.loads((shard2 / "served_target.json").read_text())
    asm = manifest["evidence"]["seedrep_assembly"]
    assert asm["n_eligible"] == 3 and asm["n_elected"] == 3 and asm["n_verified"] == 3 and asm["n_materialised"] == 3
    assert asm["refused"] == [] and asm["extra"] == [] and asm["n_drawn_by_assembly"] == 0 and asm["arm_s"] == f"none__s{tss.SEED_BASE}"
    assert asm["seed_base"] == tss.SEED_BASE                                              # the SOURCE store's seed base
    assert "3 verified, 0 refused" in out.out


def test_static_assembly_draws_missing_items_refuses_mismatches_and_ignores_extras(tmp_path, monkeypatch, capsys):
    es, (ref_bytes, ref_esc), rows, records = _reference(tmp_path, monkeypatch, capsys)
    # elected: g0/1 (good), g1/1 (bound to ANOTHER base row: refused), g2/0 (not at cap: extra); g3/0 missing
    partial = {k: v for k, v in records.items() if not k.endswith("__g3__0")}
    ladder_root, snap, spec, by_key, res, vol = _static_ladder(tmp_path, rows, AT_CAP, extra={("g2", 0)})
    good = by_key["static__pasttense__g0__1"]
    bad = copy.deepcopy(by_key["static__pasttense__g1__1"]); bad["base_row_sha256"] = "f" * 64
    xtra = by_key["static__pasttense__g2__0"]
    _materialise(vol, spec, good, record=partial["static__pasttense__g0__1"], text=partial["static__pasttense__g0__1"]["text"], results=res)
    _materialise(vol, spec, bad, record=partial["static__pasttense__g1__1"], text=partial["static__pasttense__g1__1"]["text"], results=res)
    xrec = {"text": "never used", "finish_reason": "stop", "completion_tokens": 2, "prompt_tokens": 1, "max_tokens": 8192,
            "sampling_seed": xtra["sampling_seed"]}
    _materialise(vol, spec, xtra, record=xrec, text="never used", results=res)
    L.snapshot_kv(res, snap)
    root2 = tmp_path / "second"
    tss._base_store(root2)
    chat = _Refusing({"g0/1"}, decide=tss._redraw)
    tss._wire_escalate(monkeypatch, es, chat)
    argv = tss._esc_argv(root2, tss._esc_config(root2), **{"--from-materialised": str(ladder_root), "--results": str(snap)})
    assert es.main(argv) == 0
    capsys.readouterr()
    shard2 = root2 / tss.ESC_RUN / "pasttense" / "none" / "shard00"
    assert _bytes(shard2, ["responses.jsonl", "progress.jsonl"]) == ref_bytes          # still byte-identical
    assert sorted(p["messages"][-1]["content"] for p in chat.payloads) == ["g1/1", "g3/0"]  # refused + missing drawn here
    asm = json.loads((shard2 / "served_target.json").read_text())["evidence"]["seedrep_assembly"]
    assert asm["n_verified"] == 1 and asm["n_materialised"] == 1 and asm["n_drawn_by_assembly"] == 2
    assert len(asm["refused"]) == 1 and "base row" in asm["refused"][0] and asm["extra"] == ["static__pasttense__g2__0"]
    assert json.loads((shard2 / "escalation.json").read_text())["n_target_forwards"] == 2


def test_static_assembly_refuses_an_open_queue_a_torn_file_and_a_missing_snapshot(tmp_path, monkeypatch, capsys):
    es, _, rows, records = _reference(tmp_path, monkeypatch, capsys)
    ladder_root, snap, spec, by_key, res, vol = _static_ladder(tmp_path, rows, AT_CAP, records=records)
    root2 = tmp_path / "second"
    tss._base_store(root2)
    tss._wire_escalate(monkeypatch, es, tss._StubChat(decide=tss._redraw))
    cfg2 = tss._esc_config(root2)
    # --from-materialised without --results is a refusal (exit 2)
    with pytest.raises(SystemExit) as ei:
        es.main(tss._esc_argv(root2, cfg2, **{"--from-materialised": str(ladder_root)}))
    assert ei.value.code == 2 and "--results" in capsys.readouterr().err
    # a torn materialised file: the Dict's sha is not the text's -> fail closed, nothing written
    path = Path(str(vol.root / L.materialised_rel(spec.ladder_root, f"none__s{tss.SEED_BASE}", "static__pasttense__g0__1", "rung-1")))
    doc = json.loads(path.read_text()); doc["record"]["text"] += "!"; path.write_text(json.dumps(doc))
    with pytest.raises(L.LadderRefusal, match="text_sha256"):
        es.main(tss._esc_argv(root2, cfg2, **{"--from-materialised": str(ladder_root), "--results": str(snap)}))
    assert not (root2 / tss.ESC_RUN / "pasttense" / "none" / "shard00").exists()
    # an OPEN queue (no CLOSED.json) refuses
    (ladder_root / "CLOSED.json").unlink()
    with pytest.raises(L.LadderRefusal, match="CLOSED.json"):
        es.main(tss._esc_argv(root2, cfg2, **{"--from-materialised": str(ladder_root), "--results": str(snap)}))


# --------------------------------------------------------------------------- #
# seam 2 — controls: the post seam, byte-identical spool + sidecar              #
# --------------------------------------------------------------------------- #


def _store_files(cell_dir: Path):
    return {n: (cell_dir / "private" / n).read_bytes() for n in list(SPOOL.values()) + list(META.values())}


def _test_world_ladder(monkeypatch):
    """The offline controls world climbs 64 -> 128 -> 256 (scp.CONTROLS_CAP_LADDER is patched alike by the ``ladder``
    fixture); the protocol's ladder AND its stage vocabulary follow it for the test."""
    monkeypatch.setitem(L.LADDERS, "controls", (64, 128, 256))
    monkeypatch.setattr(L, "STAGES", (*L.STAGES, "128", "256"))


def test_controls_assembly_answers_at_cap_redraws_from_elected_rows_and_the_store_is_byte_identical(ladder, monkeypatch, tmp_path):
    _test_world_ladder(monkeypatch)
    # the reference rung at 128 with a floor sample of 1
    srun.main(_rung_argv(ladder, ladder.rung128, 128, ladder.base_root, "--floor-sample", "1"))
    ref_dir = _cell_dir(ladder.rung128)
    ref_files = _store_files(ref_dir)
    ref_gen = json.loads((ref_dir / "manifest.json").read_text())["resolved_config"]["gen"]
    posted_ref = [dict(p) for p in ladder.posted]
    del ladder.posted[:]
    # the CLOSED 128 queue: the base's two at-cap items, elected draws = the canned responses at 128
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = L.QueueSpec.build(T.W7, T.WAVE, "controls", "s1", "controls", 128, ["0"])
    acc, res = _kv("accepted"), _kv("results")
    base_cell = ladder.base_cell
    cell_id = json.loads((base_cell / "manifest.json").read_text())["cell_id"]
    by_key = {}
    for control, meta_name in META.items():
        for m in _meta_rows(base_cell, meta_name):
            if not m["at_cap"]:
                continue
            prompt = next(p for p, iid in ITEM_OF_PROMPT.items() if iid == m["item_id"])
            ev = {"item_id": m["item_id"], "at_cap": True, "finish_reason": m["finish_reason"], "sampling_seed": m["sampling_seed"],
                  "text_sha256": m["text_sha256"]}
            rec = L.controls_queue_record(spec, arm=cell_id, s=CONTROLS_SEED_BASE, control=control, event=ev, prompt=prompt, system=None,
                                          base_store="exp/base/private/gens.jsonl", producer=spec.producers[0], producer_attempt_id="a1",
                                          vmodel="petri__x", sampling=T.PRESET)
            by_key[rec["item_key"]] = rec
    assert len(by_key) == 2 and {r["identity"]["item_id"] for r in by_key.values()} == AT_CAP_64
    _close_queue(vol, spec, "0", "a1", by_key, acc)
    for rec in by_key.values():
        text, fr, ct = _canned_draw(rec["identity"]["item_id"], 128)
        response = {"choices": [{"message": {"content": text}, "finish_reason": fr}],
                    "usage": {"prompt_tokens": 17, "completion_tokens": ct, "total_tokens": 17 + ct}}
        record = {"text": text, "finish_reason": fr, "completion_tokens": ct, "prompt_tokens": 17, "max_tokens": 128,
                  "sampling_seed": rec["sampling_seed"]}
        _materialise(vol, spec, rec, record=record, text=text, response=response, results=res)
    snap = tmp_path / "results.json"
    L.snapshot_kv(res, snap)
    # the assembled rung into a fresh root: the two at-cap prompts must never reach the poster
    at_cap_prompts = {p for p, iid in ITEM_OF_PROMPT.items() if iid in AT_CAP_64}
    real_poster = srun._proxy_poster

    def guarded_poster(url):
        cm = real_poster(url)

        class _CM:
            async def __aenter__(self):
                post = await cm.__aenter__()

                async def guarded(req):
                    assert req["messages"][-1]["content"] not in at_cap_prompts, "an elected item reached the engine"
                    return await post(req)
                return guarded

            async def __aexit__(self, *a):
                return await cm.__aexit__(*a)
        return _CM()

    monkeypatch.setattr(srun, "_proxy_poster", guarded_poster)
    out2 = ladder.tmp_path / "controls_cap128_assembled"
    srun.main(_rung_argv(ladder, out2, 128, ladder.base_root, "--floor-sample", "1",
                         "--from-materialised", str(vol_root / spec.ladder_root), "--results", str(snap)))
    cell2 = _cell_dir(out2)
    assert cell2.name == ref_dir.name
    assert _store_files(cell2) == ref_files                                             # spool + meta byte-identical
    assert json.loads((cell2 / "manifest.json").read_text())["resolved_config"]["gen"] == ref_gen
    # exactly the floor-sample draw reached the engine; the two at-cap draws came from the elected rows
    assert len(ladder.posted) == len(posted_ref) - 2
    assert not {p["messages"][-1]["content"] for p in ladder.posted} & at_cap_prompts
    s = RR.wrap_controls_post.last_summary
    assert s["n_elected"] == 2 and s["n_verified"] == 2 and s["n_materialised"] == 2 and s["refused"] == [] and s["extra"] == []
    assert s["n_drawn_by_assembly"] == len(ladder.posted)


def test_controls_assembly_refuses_outside_an_escalation_run_and_re_draws_a_mismatch(ladder, monkeypatch, tmp_path):
    _test_world_ladder(monkeypatch)
    with pytest.raises(SystemExit, match="rides an escalation run"):
        srun.main(ladder.argv(tmp_path / "base2", "--from-materialised", str(tmp_path), "--results", str(tmp_path / "x.json")))
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = L.QueueSpec.build(T.W7, T.WAVE, "controls", "s1", "controls", 128, ["0"])
    acc, res = _kv("accepted"), _kv("results")
    base_cell = ladder.base_cell
    cell_id = json.loads((base_cell / "manifest.json").read_text())["cell_id"]
    m = next(m for m in _meta_rows(base_cell, META["xstest"]) if m["at_cap"])
    prompt = next(p for p, iid in ITEM_OF_PROMPT.items() if iid == m["item_id"])
    ev = {"item_id": m["item_id"], "at_cap": True, "finish_reason": m["finish_reason"], "sampling_seed": m["sampling_seed"],
          "text_sha256": "0" * 64}                                                        # bound to ANOTHER base text
    rec = L.controls_queue_record(spec, arm=cell_id, s=CONTROLS_SEED_BASE, control="xstest", event=ev, prompt=prompt, system=None, base_store="b",
                                  producer=spec.producers[0], producer_attempt_id="a1", vmodel="petri__x", sampling=T.PRESET)
    _close_queue(vol, spec, "0", "a1", {rec["item_key"]: rec}, acc)
    text, fr, ct = _canned_draw(m["item_id"], 128)
    _materialise(vol, spec, rec, record={"text": "WRONG", "finish_reason": fr, "completion_tokens": ct, "prompt_tokens": 17,
                                         "max_tokens": 128, "sampling_seed": rec["sampling_seed"]}, text="WRONG", results=res)
    snap = tmp_path / "results.json"
    L.snapshot_kv(res, snap)
    out2 = ladder.tmp_path / "controls_cap128_mismatch"
    srun.main(_rung_argv(ladder, out2, 128, ladder.base_root, "--floor-sample", "0",
                         "--from-materialised", str(vol_root / spec.ladder_root), "--results", str(snap)))
    s = RR.wrap_controls_post.last_summary
    assert s["n_elected"] == 1 and s["n_verified"] == 0 and len(s["refused"]) == 1 and "base row" in s["refused"][0]
    # the mismatching item was re-drawn by the engine and the store carries the ENGINE's text, never "WRONG"
    assert prompt in {p["messages"][-1]["content"] for p in ladder.posted}
    spool = _spool_lines(_cell_dir(out2), SPOOL["xstest"])
    assert next(r for r in spool[1:] if r["item_id"] == m["item_id"])["text"] == text


# --------------------------------------------------------------------------- #
# seam 3 — misalign: elected rows seeded into the NEW root, the runner draws the rest
# --------------------------------------------------------------------------- #

PROSE = " ".join(f"w{i * 7919 % 1013}" for i in range(300))
LOOP = "ab" * 300


def _misalign_row(cell, item, s, resp, fr, *, beh):
    return {"family": "agentic", "cell_id": cell, "virtual_model": "petri__base", "item_id": item, "sample_idx": s, "behavior": beh,
            "prompt_sha": "f" * 64, "response": resp, "finish_reason": fr, "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "model_reported": "petri__base"}


def _misalign_source(tmp_path):
    src = tmp_path / "src" / "agentic"
    d = src / "none"
    d.mkdir(parents=True)
    rows = [_misalign_row("none", "leaking_explicit-none", 0, "done", "stop", beh="leaking"),
            _misalign_row("none", "leaking_explicit-none", 1, PROSE, "length", beh="leaking"),        # selected
            _misalign_row("none", "murder_explicit-none", 0, PROSE + " x", "length", beh="murder"),   # selected
            _misalign_row("none", "murder_explicit-none", 1, LOOP, "length", beh="murder")]           # a loop: degenerate
    (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "meta.json").write_text(json.dumps({"cell_id": "none", "seed_base": 1, "params": {"max_tokens": 3000}}))
    return src, rows


def _misalign_ladder(tmp_path, rows, *, tamper=None):
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = L.QueueSpec.build(T.W7, T.WAVE, "misalign", "s1", "misalign", 16384, ["0"])
    acc, res = _kv("accepted"), _kv("results")
    by_key = {}
    for r in rows:
        if r["finish_reason"] == "length" and R.zlib_ratio(r["response"]) >= R.RERUN_MIN_ZLIB_RATIO:
            rec = L.misalign_queue_record(spec, arm="none", s=1, row=r, messages=[{"role": "user", "content": "m"}],
                                          base_store="exp/agentic/none/responses.jsonl", producer=spec.producers[0],
                                          producer_attempt_id="a1", sampling=T.PRESET)
            by_key[rec["item_key"]] = rec
    _close_queue(vol, spec, "0", "a1", by_key, acc)
    # the elected 16,384 draw for (leaking, 1): a full runner row; (murder, 0) is left for the runner
    rec = by_key["misalign__leaking_explicit-none__1"]
    if tamper:
        rec = tamper(copy.deepcopy(rec))
    row = _misalign_row("none", "leaking_explicit-none", 1, PROSE + " and then it acted", "stop", beh="leaking")
    _materialise(vol, spec, rec, record=row, text=row["response"], results=res)
    snap = tmp_path / "results.json"
    L.snapshot_kv(res, snap)
    return vol_root / spec.ladder_root, snap


def _misalign_argv(tmp_path, cells_path, src, out, ladder_root=None, snap=None):
    argv = ["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells_path), "--family", "agentic", "--n", "2", "--out", str(out),
            "--max-tokens", "16384", "--rerun-truncated-from", str(src)]
    if ladder_root is not None:
        argv += ["--from-materialised", str(ladder_root)]
    if snap is not None:
        argv += ["--results", str(snap)]
    return argv


def test_misalign_assembly_seeds_elected_rows_and_the_runner_draws_only_the_rest(tmp_path, monkeypatch):
    src, rows = _misalign_source(tmp_path)
    ladder_root, snap = _misalign_ladder(tmp_path, rows)
    fake = FakeHttp(["petri__base"])
    monkeypatch.setattr(R, "HttpxJson", lambda api_key, timeout_s: fake)
    monkeypatch.setattr(R, "load_items", lambda family, behaviors: ([dict(i) for i in ITEMS], REV))
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps([CELLS[0]]))
    out = tmp_path / "out16k" / "agentic"
    assert R.main(_misalign_argv(tmp_path, cells_path, src, out, ladder_root, snap)) == 0
    got = [json.loads(l) for l in (out / "none" / "responses.jsonl").read_text().splitlines() if l.strip()]
    assert [(r["item_id"], r["sample_idx"]) for r in got] == [("leaking_explicit-none", 1), ("murder_explicit-none", 0)]
    assert got[0]["response"] == PROSE + " and then it acted" and got[0]["finish_reason"] == "stop"   # the seeded elected row
    assert got[1]["response"].startswith("reply:")                                                    # drawn by the runner
    assert len(fake.posts) == 1 and fake.posts[0][1]["max_tokens"] == 16384
    prov = [json.loads(l) for l in (out / "none" / "materialised_seeded.jsonl").read_text().splitlines()]
    assert prov == [{"item_id": "leaking_explicit-none", "sample_idx": 1, "attempt_id": "rung-1",
                     "request_id": "rq-misalign__leaking_explicit-none__1", "completion_id": "cmpl-misalign__leaking_explicit-none__1",
                     "path": prov[0]["path"], "text_sha256": T.sha(PROSE + " and then it acted")}]
    meta = json.loads((out / "none" / "meta.json").read_text())
    assert meta["present"] == 2 and meta["generated_this_run"] == 1
    # a re-run is idempotent: nothing seeded twice, nothing drawn
    assert R.main(_misalign_argv(tmp_path, cells_path, src, out, ladder_root, snap)) == 0
    assert len([l for l in (out / "none" / "responses.jsonl").read_text().splitlines() if l.strip()]) == 2 and len(fake.posts) == 1


def test_misalign_assembly_refuses_a_mismatch_and_the_runner_draws_it_itself(tmp_path, monkeypatch):
    src, rows = _misalign_source(tmp_path)

    def tamper(rec):
        rec["base_row_sha256"] = "0" * 64                                   # bound to another base row
        return rec
    ladder_root, snap = _misalign_ladder(tmp_path, rows, tamper=tamper)
    fake = FakeHttp(["petri__base"])
    monkeypatch.setattr(R, "HttpxJson", lambda api_key, timeout_s: fake)
    monkeypatch.setattr(R, "load_items", lambda family, behaviors: ([dict(i) for i in ITEMS], REV))
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps([CELLS[0]]))
    out = tmp_path / "out16k" / "agentic"
    assert R.main(_misalign_argv(tmp_path, cells_path, src, out, ladder_root, snap)) == 0
    got = [json.loads(l) for l in (out / "none" / "responses.jsonl").read_text().splitlines() if l.strip()]
    assert len(got) == 2 and all(r["response"].startswith("reply:") for r in got) and len(fake.posts) == 2
    assert not (out / "none" / "materialised_seeded.jsonl").exists()
    # --from-materialised without --results, or without the ladder, refuses
    with pytest.raises(R.MisalignRunnerError, match="--results"):
        R.main(_misalign_argv(tmp_path, cells_path, src, tmp_path / "o2", ladder_root, None))
    with pytest.raises(R.MisalignRunnerError, match="rides --rerun-truncated-from"):
        R.main(["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells_path), "--family", "agentic", "--n", "1",
                "--out", str(tmp_path / "o3"), "--from-materialised", str(ladder_root), "--results", str(snap)])


# --------------------------------------------------------------------------- #
# seam 4 — GSM8K: the adapter's subset, cross-checked against the CLOSED queue   #
# --------------------------------------------------------------------------- #


def _gsm8k_queue(tmp_path, items, *, view="s1"):
    vol_root = tmp_path / "vol"
    vol = L.LocalVolume(vol_root)
    spec = L.QueueSpec.build(T.W7, T.WAVE, "gsm8k", view, "gsm8k", 16384, ["0"])
    acc, res = _kv("accepted"), _kv("results")
    by_key = {}
    for it in items:
        rec = L.gsm8k_queue_record(spec, arm="none", s=1, item=it, base_store="exp/cells/x/outcomes/inspect__gsm8k_paired200.json",
                                   producer=spec.producers[0], producer_attempt_id="a1")
        by_key[rec["item_key"]] = rec
    _close_queue(vol, spec, "0", "a1", by_key, acc)
    snap = tmp_path / "results.json"
    L.snapshot_kv(res, snap)
    return vol_root / spec.ladder_root, snap


def test_gsm8k_assembly_returns_the_adapters_subset_and_cross_checks_the_queue(tmp_path, monkeypatch):
    tir._forbid_peer_exit_rule(monkeypatch)
    ids = tir._ids()
    at_cap = {i: tir.CAP for i in ids[:8]}                                  # 4 % at cap: 2 loops + 6 truncated
    src = tir._store(tmp_path / "src", cap=tir.CAP, at_cap=at_cap)
    texts = tir._texts(ids, loops=ids[:2])
    outcomes = [{"item_id": i, "sample_idx": 0, "stop_reason": "max_tokens", "at_cap": True, "seed": n} for n, i in enumerate(ids[:8])]
    ladder_root, snap = _gsm8k_queue(tmp_path, outcomes)                   # the tailer queues EVERY at-cap item
    record, subset, verdict = IR.subset_from_materialised_queue(ladder_root, snap, source_out_root=src, arm="none", seed_base=1, texts=texts)
    assert record["exit"] == "escalate" and subset is not None
    assert subset["ids"] == sorted(ids[2:8]) and subset["cap"] == tir.RUNG and subset["source_cap"] == tir.CAP
    assert verdict == {"exit": "escalate", "arm_s": "none__s1", "n_eligible": 6, "n_queued": 8, "missing_from_queue": [],
                       "extra_in_queue": sorted(ids[:2]), "subset_ids": sorted(ids[2:8]), "cap": 16384}
    # a queue that lacks an eligible item is NOTED (Inspect draws the whole subset); a queued item the source never
    # filed at cap REFUSES
    ladder2, snap2 = _gsm8k_queue(tmp_path / "two", outcomes[:5])
    _, _, v2 = IR.subset_from_materialised_queue(ladder2, snap2, source_out_root=src, arm="none", seed_base=1, texts=texts)
    assert v2["missing_from_queue"] == sorted(ids[5:8]) and v2["n_queued"] == 5
    foreign = outcomes + [{"item_id": ids[50], "sample_idx": 0, "stop_reason": "max_tokens", "at_cap": True, "seed": 50}]
    ladder3, snap3 = _gsm8k_queue(tmp_path / "three", foreign)
    with pytest.raises(L.LadderRefusal, match="never filed at cap"):
        IR.subset_from_materialised_queue(ladder3, snap3, source_out_root=src, arm="none", seed_base=1, texts=texts)
    # the record builder refuses an item that is not at cap
    with pytest.raises(L.LadderRefusal, match="not at cap"):
        L.gsm8k_queue_record(L.QueueSpec.build(T.W7, T.WAVE, "gsm8k", "s1", "gsm8k", 16384, ["0"]), arm="none", s=1,
                             item={"item_id": "x", "sample_idx": 0, "stop_reason": "stop", "at_cap": False}, base_store="b",
                             producer=L.QueueSpec.build(T.W7, T.WAVE, "gsm8k", "s1", "gsm8k", 16384, ["0"]).producers[0],
                             producer_attempt_id="a1")
