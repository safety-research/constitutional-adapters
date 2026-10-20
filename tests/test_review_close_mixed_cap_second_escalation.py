"""Closing review, P1 — the 16,384 static rung accepts the ORDINARY mixed-cap 8,192 store.

Before: `escalate_static.load_base_store` required EVERY base row's whole `decode` block to equal this rung's
`block_from` — the block at the base store's cap. That is true of a 4,096 BASE store (one producer, one cap) and false
of every 8,192 RUNG store the 8,192 producer writes: it copies rows that terminated at 4,096 forward UNCHANGED (their
decode blocks keep `cap: 4096`, `source_cap: 4096`, `regenerated: false` — the two-decode-blocks design of DESIGN.md)
and only the re-drawn rows carry `cap: 8192`. So the 16,384 phase (`static_esc16_t<T>`) refused the one
input it exists for — a pair with rows terminated at 4,096 AND > 1 % coherent truncation at 8,192 — on its first
carried row (`decode.cap=4096` vs the expected 8,192), after paying for its 16,384 standup and before re-drawing
anything. The round-2 reachability fixtures gave every row one cap and missed the producer–consumer mismatch.

Now: every row is validated against its PROVENANCE-VERIFIED source cap (`escalate_static.row_source_cap`): a base row
(no provenance keys) at the store's cap; a rung row per its own marks — `regenerated` true = drawn at the store's cap,
false = copied from a rung below, and in both cases `source_cap` must agree with the row's own `decode.cap` — with the
sampler / arm / seed_base / trial and the seed rule checked as strictly as before at THAT cap. The source cap is
PRESERVED through every further copy (a row that terminated at 4,096 still says 4,096 inside the 16,384 store; the
old code would have relabelled it 8,192), `boundary_landing` compares against the cap the base text was actually drawn
at, `escalation.json` counts `n_rows_by_source_cap`, and the round-4 resume path (`existing_complete`) applies the same
per-row rule to the rung root it is about to resume.

The chain here is REAL end to end, nothing hand-shaped after the 4,096 base: the real 4,096 → 8,192 producer
(`escalate_static.main`, stub chat) fed by the real gate's disposition file (`ladder_gate.py static` over the 4,096 store,
exactly as the sbatch's `f22s_static_gate` runs it) makes the mixed-cap 8,192 store; the real gate over THAT store issues
the ESCALATE 16384 items; the real 16,384 producer consumes them; the real ceiling gate reads the 16,384 store; the
resume path is exercised on the real root. The negative cases tamper ONE carried row of the real 8,192 store.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import test_sampled_static as tss  # the sampled static lane's builders (after the tests/ path insert)

SAMPLED = tss.SAMPLED_DIR
PY = sys.executable
BASE_RUN, ESC_RUN = tss.BASE_RUN, tss.ESC_RUN
ESC16_RUN = "run_sampled_t0_cap16384"
N_GOALS = 100                                    # x 2 samples = 200 rows, so 3 truncated rows are 1.5 % (> the 1 % trigger)
LOOP = "a b c d " * 6                            # 4-gram rep_ratio >= 0.5: the gate's LOOP class, never escalated
COHERENT = "one two three four five six seven eight nine ten eleven twelve"
#: the 4,096 base's at-cap items: one loop (excluded by the 4,096 gate, copied forward AT CAP by the 8,192 producer) and
#: six coherent truncations (3 % > 1 %: the 4,096 gate escalates them)
LOOP_AT_4096 = {("g0", 0)}
COHERENT_AT_4096 = {("g1", 0), ("g1", 1), ("g2", 0), ("g2", 1), ("g3", 0), ("g3", 1)}
#: of the six re-drawn at 8,192: three terminate (copied forward into the 16,384 store WITH source_cap 8,192), three stay at
#: cap with coherent text (1.5 % > 1 %: the 8,192 gate escalates exactly these to 16,384)
TERMINATE_AT_8192 = {("g1", 0), ("g1", 1), ("g2", 0)}
STILL_AT_CAP_8192 = {("g2", 1), ("g3", 0), ("g3", 1)}


def _at_cap_text(gid, idx):
    return f"{COHERENT} {gid} {idx}"


def _base_store(root):
    """The 4,096 base over two shards (goals g0..g49 / g50..g99): 200 rows, 7 at cap."""
    shards = [[], []]
    for g in range(N_GOALS):
        gid = f"g{g}"
        for idx in (0, 1):
            key = (gid, idx)
            if key in LOOP_AT_4096:
                row = tss._srow(gid, idx, "length", response=LOOP)
            elif key in COHERENT_AT_4096:
                row = tss._srow(gid, idx, "length", response=_at_cap_text(gid, idx))
            else:
                row = tss._srow(gid, idx, "stop")
            shards[g // 50].append(row)
    tss._write_store(root, BASE_RUN, "pasttense", "none", shards)


def _rung_config(tmp_path, *, run, cap, base_run, base_cap, name):
    cfg = tss._cfg(private_run=run, attacks={"static_max_new_tokens": cap, "target_max_new_tokens": 512},
                   sampled_decode=tss._sampled_block(static_cap=cap, escalated_from={
                       "kind": "static", "cap": base_cap, "private_run": base_run}))
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg))
    return p


def _gate(tmp_path, run, cap):
    """`ladder_gate.py static` exactly as the sbatch's f22s_static_gate runs it (no sibling here) -> (line, items path)."""
    ep = tmp_path / "endpoint.json"
    ep.write_text(json.dumps({"proxy_url": "http://127.0.0.1:1/v1", "upstream_timeout_s": 7200}))
    gdir = tmp_path / "gates" / f"static_cap{cap}"
    gdir.mkdir(parents=True, exist_ok=True)
    items = gdir / "pasttense__none.items.jsonl"
    argv = ["static", "--private-root", str(tmp_path), "--private-run", run, "--attack", "pasttense", "--defense", "none",
            "--cap", str(cap), "--endpoint", str(ep), "--out", str(gdir / "pasttense__none.gate.json"),
            "--decision-out", str(gdir / "pasttense__none.decision.json"), "--items-out", str(items)]
    r = subprocess.run([PY, str(SAMPLED / "ladder_gate.py"), *argv], capture_output=True, text=True, cwd=REPO, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip(), items


def _redraw_8192(payload):
    """The 8,192 re-draw oracle: g1/0 prefix-extends and terminates above 4,096; g1/1 identical, terminated below 4,096
    (a boundary landing); g2/0 prefix-extends; g2/1 and g3/0 prefix-extend but stay at cap; g3/1 FORKS and stays at cap."""
    prompt = payload["messages"][-1]["content"]
    gid, idx = prompt.split("/")
    base = _at_cap_text(gid, int(idx))
    if prompt == "g1/0":
        return base + " and then it finished", "stop", 5000
    if prompt == "g1/1":
        return base, "stop", 900
    if prompt == "g2/0":
        return base + " concluded", "stop", 6000
    if prompt in ("g2/1", "g3/0"):
        return base + " continuing onward", "length", 8192
    if prompt == "g3/1":
        return f"{COHERENT} {gid} FORK {idx} continuing onward", "length", 8192
    raise AssertionError(f"unexpected 8,192 re-draw of {prompt!r} (a terminated or excluded row was re-drawn)")


def _text_at_8192(gid, idx):
    text, _, _ = _redraw_8192({"messages": [{"content": f"{gid}/{idx}"}]})
    return text


def _redraw_16384(payload):
    """The 16,384 re-draw oracle over the three items the 8,192 gate escalates: g2/1 identical, terminated below its 8,192
    source cap (a boundary landing); g3/0 prefix-extends and terminates; g3/1 stays at the ceiling."""
    prompt = payload["messages"][-1]["content"]
    gid, idx = prompt.split("/")
    base = _text_at_8192(gid, int(idx))
    if prompt == "g2/1":
        return base, "stop", 3000
    if prompt == "g3/0":
        return base + " and done", "stop", 12000
    if prompt == "g3/1":
        return base + " still going", "length", 16384
    raise AssertionError(f"unexpected 16,384 re-draw of {prompt!r} (a carried row was re-drawn)")


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _by_key(rows):
    return {(r["goal_id"], r["sample_idx"]): r for r in rows}


def _block(cap):
    return {"do_sample": True, **tss.QWEN, "cap": cap, "arm": tss.ARM, "seed_base": tss.SEED_BASE, "trial": 0}


def _mixed_cap_8192_store(monkeypatch, tmp_path, capsys):
    """The real 4,096 -> 8,192 chain: base store, the real gate's disposition, the real producer. Returns the escalate_static
    module, the 8,192 shard dir and its rows keyed by item."""
    es = tss._load_script("escalate_static")
    _base_store(tmp_path)
    line, items4096 = _gate(tmp_path, BASE_RUN, 4096)
    assert line == f"ESCALATE 8192 {items4096}", line
    chat8 = tss._StubChat(decide=_redraw_8192)
    tss._wire_escalate(monkeypatch, es, chat8)
    cfg8 = _rung_config(tmp_path, run=ESC_RUN, cap=8192, base_run=BASE_RUN, base_cap=4096, name="esc8192.yaml")
    assert es.main(tss._esc_argv(tmp_path, cfg8, **{"--items": str(items4096)})) == 0
    capsys.readouterr()
    assert sorted(p["messages"][-1]["content"] for p in chat8.payloads) == sorted(f"{g}/{i}" for g, i in COHERENT_AT_4096)
    shard8 = tmp_path / ESC_RUN / "pasttense" / "none" / "shard00"
    rows8 = _rows(shard8 / "responses.jsonl")
    by8 = _by_key(rows8)
    # THE PRODUCER'S OWN SHAPE, the fact the round-2 fixtures missed: a rung store carries TWO decode blocks
    assert {r["decode"]["cap"] for r in rows8} == {4096, 8192}
    assert sum(1 for r in rows8 if r["decode"]["cap"] == 4096) == 194 and sum(1 for r in rows8 if r["decode"]["cap"] == 8192) == 6
    for key, r in by8.items():
        if key in COHERENT_AT_4096:
            assert r["regenerated"] is True and r["source_cap"] == 8192 and r["decode"] == _block(8192)
        else:
            assert r["regenerated"] is False and r["source_cap"] == 4096 and r["decode"] == _block(4096)
    assert by8[("g0", 0)][es.EXCLUDED_KEY] == "loop" and by8[("g0", 0)]["at_cap"] is True
    assert {k for k, r in by8.items() if r["at_cap"]} == STILL_AT_CAP_8192 | LOOP_AT_4096
    return es, shard8, by8


def _snapshot(shard):
    return {p.name: p.read_bytes() for p in shard.iterdir()}


def _refused(es, argv, capsys):
    with pytest.raises(SystemExit) as ei:
        es.main(argv)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "[escalate-static] REFUSED" in err, err
    return err


# --------------------------------------------------------------------------- the positive chain: 4,096 -> 8,192 -> 16,384 -> resume

def test_the_16384_rung_regenerates_exactly_the_gates_items_of_a_real_mixed_cap_8192_store_and_resumes_idempotently(
        monkeypatch, tmp_path, capsys):
    es, shard8, by8 = _mixed_cap_8192_store(monkeypatch, tmp_path, capsys)
    # the real gate over the mixed-cap 8,192 store: 3 coherent truncations of 200 = 1.5 % -> ESCALATE 16384, the loop excluded
    line, items8192 = _gate(tmp_path, ESC_RUN, 8192)
    assert line == f"ESCALATE 16384 {items8192}", line
    disp = _rows(items8192)
    assert {(r["goal_id"], r["sample_idx"]) for r in disp[1:] if r["escalate"]} == STILL_AT_CAP_8192
    assert {(r["goal_id"], r["sample_idx"]): r["excluded_as"] for r in disp[1:] if not r["escalate"]} == {("g0", 0): "loop"}

    # ---- (2) the 16,384 generation path over the mixed-cap store: must NOT refuse; re-draws exactly the gate's items
    chat16 = tss._StubChat(decide=_redraw_16384)
    _, built = tss._wire_escalate(monkeypatch, es, chat16)
    cfg16 = _rung_config(tmp_path, run=ESC16_RUN, cap=16384, base_run=ESC_RUN, base_cap=8192, name="esc16384.yaml")
    argv16 = tss._esc_argv(tmp_path, cfg16, **{"--items": str(items8192), "--max-model-len": "18432"})
    assert es.main(argv16) == 0
    out = capsys.readouterr()
    summary = json.loads(out.out.strip().splitlines()[-1])
    assert built["max_model_len"] == 18432 and built["cell_id"] == "none"
    sent = {p["messages"][-1]["content"]: p for p in chat16.payloads}
    assert set(sent) == {f"{g}/{i}" for g, i in STILL_AT_CAP_8192}                      # exactly the eligible items
    for prompt, p in sent.items():
        gid, idx = prompt.split("/")
        assert p["max_tokens"] == 16384 and p["seed"] == by8[(gid, int(idx))]["sampling_seed"]   # the base seeds, unchanged
    shard16 = tmp_path / ESC16_RUN / "pasttense" / "none" / "shard00"
    assert sorted(p.name for p in shard16.iterdir()) == ["escalation.json", "progress.jsonl", "responses.jsonl", "served_target.json"]
    rows16 = _rows(shard16 / "responses.jsonl")
    assert [(r["goal_id"], r["sample_idx"]) for r in rows16] == [(r["goal_id"], r["sample_idx"]) for r in _rows(shard8 / "responses.jsonl")]
    by16 = _by_key(rows16)
    # three source caps live in the ceiling store, each row at the block of the rung that produced ITS text
    assert {r["decode"]["cap"] for r in rows16} == {4096, 8192, 16384}
    for key, r in by16.items():
        b = by8[key]
        assert r["escalated_from_cap"] == 8192
        if key in STILL_AT_CAP_8192:
            assert r["regenerated"] is True and r["source_cap"] == 16384 and r["decode"] == _block(16384)
            assert r["max_tokens"] == 16384 and r["sampling_seed"] == b["sampling_seed"]
            assert r["source_finish_reason"] == "length" and r["source_completion_tokens"] == 8192
            assert r["source_draw_cap"] == 8192                    # the cap the base text was drawn at, recorded
        else:
            # copied forward: byte-identical to the 8,192 row but for the rung's own provenance keys — and the source cap
            # is PRESERVED (a row that terminated at 4,096 still says 4,096; one that terminated at 8,192 says 8,192)
            assert r["regenerated"] is False
            assert r["source_cap"] == b["source_cap"] == r["decode"]["cap"] == (8192 if key in TERMINATE_AT_8192 else 4096)
            assert {k: v for k, v in r.items() if k not in ("regenerated", "source_cap", "escalated_from_cap")} == \
                {k: v for k, v in b.items() if k not in ("regenerated", "source_cap", "escalated_from_cap")}
    assert by16[("g0", 0)][es.EXCLUDED_KEY] == "loop" and by16[("g0", 0)]["at_cap"] is True and by16[("g0", 0)]["source_cap"] == 4096
    ident, ext, ceiling = by16[("g2", 1)], by16[("g3", 0)], by16[("g3", 1)]
    assert ident["identical"] and ident["prefix_extension"] and ident["boundary_landing"] is True    # 3,000 < its 8,192 source cap
    assert ext["prefix_extension"] and not ext["identical"] and ext["boundary_landing"] is False and ext["at_cap"] is False
    assert ceiling["at_cap"] is True and ceiling["finish_reason"] == "length" and ceiling["boundary_landing"] is False
    esc = json.loads((shard16 / "escalation.json").read_text())
    for k, v in {"cap_from": 8192, "cap_to": 16384, "n_rows": 200, "n_at_cap_before": 4, "n_regenerated": 3, "n_copied": 197,
                 "n_at_cap_after": 2, "n_at_cap_after_regenerated": 1, "n_at_cap_after_excluded": 1, "n_escalation_excluded": 1,
                 "n_escalation_excluded_by_class": {"loop": 1, "repetitive": 0, "boundary": 0},
                 "n_rows_by_source_cap": {"4096": 194, "8192": 3, "16384": 3},
                 "n_prefix_extension": 3, "n_fork": 0, "n_identical": 1, "n_boundary_landing": 1,
                 "base_private_run": ESC_RUN, "private_run": ESC16_RUN, "item_filter": True,
                 "items_file_sha256": hashlib.sha256(items8192.read_bytes()).hexdigest()}.items():
        assert esc[k] == v, (k, esc[k])
    assert summary["n_regenerated"] == 3 and summary["n_rows_by_source_cap"] == {"4096": 194, "8192": 3, "16384": 3}
    doc = json.loads((shard16 / "served_target.json").read_text())
    assert doc["manifest_stage"] == "final" and doc["decode"] == _block(16384) and doc["n_sampled_requests"] == 3
    prog = _rows(shard16 / "progress.jsonl")
    assert sum(p["n_regenerated"] for p in prog) == 3 and sum(p["n_copied"] for p in prog) == 197
    # the 8,192 store was never touched
    assert _by_key(_rows(shard8 / "responses.jsonl")) == by8
    # the real ceiling gate reads the three-cap store: 1 coherent truncation of 200 (0.5 %) + the loop -> exit_le_1pct
    line, _ = _gate(tmp_path, ESC16_RUN, 16384)
    assert line == "EXIT exit_le_1pct", line
    g = json.loads((tmp_path / "gates" / "static_cap16384" / "pasttense__none.gate.json").read_text())
    assert g["counts"]["n_at_cap"] == 2 and g["counts"]["n_loop"] == 1 and g["counts"]["n_truncated"] == 1

    # ---- (3) the resume path over the real three-cap root: idempotent, sidecar-validated, nothing written, no server contact
    n_requests, before = len(chat16.payloads), _snapshot(shard16)
    assert es.main(argv16) == 0
    again = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert again == esc and len(chat16.payloads) == n_requests and _snapshot(shard16) == before
    # the resume applies the same per-row rule to the root it is about to resume: a carried row whose provenance disagrees
    # with its decode block is not this rung's store (refused, root left alone), and the restored root resumes again
    resp16 = shard16 / "responses.jsonl"
    tampered = [dict(r, source_cap=16384) if (r["goal_id"], r["sample_idx"]) == ("g5", 0) else r for r in rows16]
    resp16.write_text("".join(json.dumps(r) + "\n" for r in tampered))
    err = _refused(es, argv16, capsys)
    assert "g5" in err and "source_cap=16384" in err and "decode block" in err, err
    assert len(chat16.payloads) == n_requests and (shard16 / "served_target.json").read_bytes() == before["served_target.json"]
    assert not (shard16.parent / es.STAGING_NAME).exists()
    resp16.write_bytes(before["responses.jsonl"])
    assert es.main(argv16) == 0 and _snapshot(shard16) == before
    # the sidecar identity check of round 4 still governs the resume: a sidecar recording another target decode refuses
    doc_p = shard16 / "served_target.json"
    doc_p.write_text(json.dumps({**doc, "decode": {**doc["decode"], "cap": 8192}}))
    assert "different target decode" in _refused(es, argv16, capsys)
    doc_p.write_bytes(before["served_target.json"])
    assert es.main(argv16) == 0 and _snapshot(shard16) == before and len(chat16.payloads) == n_requests


# --------------------------------------------------------------------------- the negative cases: a carried row whose source cap disagrees with its provenance

def test_a_carried_row_whose_source_cap_disagrees_with_its_provenance_refuses_the_16384_rung_before_any_request(
        monkeypatch, tmp_path, capsys):
    es, shard8, by8 = _mixed_cap_8192_store(monkeypatch, tmp_path, capsys)
    line, items8192 = _gate(tmp_path, ESC_RUN, 8192)
    assert line.startswith("ESCALATE 16384 ")
    chat16 = tss._StubChat(decide=_redraw_16384)
    tss._wire_escalate(monkeypatch, es, chat16)
    cfg16 = _rung_config(tmp_path, run=ESC16_RUN, cap=16384, base_run=ESC_RUN, base_cap=8192, name="esc16384.yaml")
    argv16 = tss._esc_argv(tmp_path, cfg16, **{"--items": str(items8192), "--max-model-len": "18432"})
    resp8 = shard8 / "responses.jsonl"
    pristine = resp8.read_bytes()
    rows8 = _rows(resp8)

    def tamper(key, **over):
        out = []
        for r in rows8:
            if (r["goal_id"], r["sample_idx"]) == key:
                r = {k: v for k, v in r.items() if k not in over.get("_drop", ())}
                r.update({k: v for k, v in over.items() if k != "_drop"})
            out.append(r)
        resp8.write_text("".join(json.dumps(r) + "\n" for r in out))
        err = _refused(es, argv16, capsys)
        assert key[0] in err, err
        assert chat16.payloads == [] and not (tmp_path / ESC16_RUN).exists()      # no server contact, nothing written
        resp8.write_bytes(pristine)
        return err

    copied, regen = ("g7", 1), ("g1", 0)                        # a row terminated at 4,096; a row re-drawn (and terminated) at 8,192
    # a copied row claiming the 8,192 rung produced it while its text's decode block says 4,096
    assert "source_cap=8192" in tamper(copied, source_cap=8192) and "decode block" in tamper(copied, source_cap=8192)
    # ... and the mirror image: the decode block relabelled 8,192 over a source_cap of 4,096
    assert "cap=8192" in tamper(copied, decode={**by8[copied]["decode"], "cap": 8192})
    # a copied row marked regenerated (a re-draw would have carried the 8,192 block)
    assert "regenerated" in tamper(copied, regenerated=True)
    # a copied row whose marks are consistent with each other but claim the store's own cap while marked copied
    # refused before any request either by this tool's provenance check ("not a rung below") or, since main's rung-shaped
    # source check (642080a5) runs first, by "at or below the rung it was copied by" — both name the contradiction
    err = tamper(copied, source_cap=8192, decode={**by8[copied]["decode"], "cap": 8192})
    assert ("not a rung below" in err) or ("at or below the rung it was copied by" in err), err
    # a re-drawn row whose marks say it was drawn at 4,096 (the mirror: consistent marks, wrong rung)
    assert "regenerated" in tamper(regen, source_cap=4096, decode={**by8[regen]["decode"], "cap": 4096})
    # torn provenance: a rung row with `regenerated` but no `source_cap`
    assert "torn" in tamper(copied, _drop=("source_cap",))
    # the sampler / trial checks still hold at the source cap: another temperature on a carried 4,096 row
    assert "not this rung's base block" in tamper(copied, decode={**by8[copied]["decode"], "temperature": 0.9})
    assert "not this rung's base block" in tamper(copied, decode={**by8[copied]["decode"], "trial": 1})
    # ... and the seed rule
    assert "seeds do not reproduce" in tamper(copied, sampling_seed=by8[copied]["sampling_seed"] + 1)
    # the restored store runs
    assert es.main(argv16) == 0
    capsys.readouterr()
    assert len(chat16.payloads) == 3


def test_row_source_cap_is_the_one_rule_both_paths_read():
    """The base-store read and the round-4 resume validate rows through the same helper, and the copied-forward source
    cap is what the producer writes — never this rung's base cap."""
    es = tss._load_script("escalate_static")
    src = (SAMPLED / "escalate_static.py").read_text()
    assert src.count("assert_row_block(") >= 3                  # the definition, load_base_store, existing_complete
    assert 'row["source_cap"] = rung["cap_from"]' not in src    # the relabelling the finding names
    # a base row: the store's cap; a rung row: its own marks; disagreement refuses
    assert es.row_source_cap({"decode": {"cap": 4096}}, store_cap=4096, where="w") == 4096
    assert es.row_source_cap({"decode": {"cap": 4096}, "regenerated": False, "source_cap": 4096, "escalated_from_cap": 4096},
                             store_cap=8192, where="w") == 4096
    assert es.row_source_cap({"decode": {"cap": 8192}, "regenerated": True, "source_cap": 8192, "escalated_from_cap": 4096},
                             store_cap=8192, where="w") == 8192
    for bad in ({"decode": {"cap": 4096}, "regenerated": False, "source_cap": 8192, "escalated_from_cap": 4096},
                {"decode": {"cap": 8192}, "regenerated": False, "source_cap": 8192, "escalated_from_cap": 4096},
                {"decode": {"cap": 4096}, "regenerated": True, "source_cap": 4096, "escalated_from_cap": 4096},
                {"decode": {"cap": 4096}, "regenerated": False, "source_cap": 4096},
                {"decode": {"cap": 4096}, "regenerated": "no", "source_cap": 4096, "escalated_from_cap": 4096},
                {"decode": {"cap": 4096}, "regenerated": False, "source_cap": True, "escalated_from_cap": 4096},
                {"decode": {"cap": 4096}, "regenerated": False, "source_cap": 4096, "escalated_from_cap": 8192}):
        with pytest.raises(SystemExit) as ei:
            es.row_source_cap(bad, store_cap=8192, where="w")
        assert ei.value.code == 2, bad
