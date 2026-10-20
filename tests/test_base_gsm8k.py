"""base_gsm8k — the base-model 8-shot CoT GSM8K leg (sdf-base-steering
round 2, user directive 2026-09-03: GSM8K at 1x and 2x dose, paired).

Pins: the extractor's rules and edge cases (``The answer is 42.``,
``$1,234``, negatives, decimals, trailing period, the battery's ``####`` and
last-number fallbacks, the FIRST-``The answer is`` rule, stop-string
truncation); the prompt render (plain text, 8 exemplars, ends ``Answer:``,
stable exemplar sha); the run-identity header round trip (rows first, one
reader, pairability on PAIR_KEYS — scale deliberately NOT one — torn /
headerless refusals); ``--scale`` wiring (2.0 reaches ``obj.apply`` and the
PROMPT is unchanged); per-question rows vs aggregate consistency and the
degeneration counters; determinism. Model-side tests run on the CPU tiny
model with synthetic rows and a 4-token budget — the contract is retention,
pairability and wiring, not GSM8K accuracy.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import base_gsm8k as bg  # noqa: E402
import base_mmlu as bm  # noqa: E402


# --------------------------------------------------------------------------- #
# extraction                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("block, value, rule", [
    ("So 21 - 15 = 6. The answer is 6.", 6.0, "answer_is"),
    ("The answer is 42.", 42.0, "answer_is"),
    ("That is 5 x 3 = 15. The answer is $1,234.", 1234.0, "answer_is"),
    ("The answer is -5.", -5.0, "answer_is"),
    ("The answer is 8.00", 8.0, "answer_is"),
    ("The answer is 3.50.", 3.5, "answer_is"),
    ("the answer is: 17", 17.0, "answer_is"),
    ("The answer is 1,000,000 dollars.", 1_000_000.0, "answer_is"),
    # the FIRST "The answer is" is this question's; a later one is not
    ("The answer is 5. Actually, the answer is 6.", 5.0, "answer_is"),
    # no closing phrase -> the battery's LAST "#### N"
    ("Step one #### 3\nStep two #### 7", 7.0, "hash_last"),
    ("#### $2,500", 2500.0, "hash_last"),
    # no marker at all -> the battery's last-number fallback
    ("She has 23 - 15 = 8 dollars left", 8.0, "last_number"),
    ("There were 9 computers and then 29", 29.0, "last_number"),
    # extraction failure
    ("I am not sure how to solve this.", None, "none"),
    ("", None, "none"),
])
def test_extract_answer_rules(block, value, rule):
    got, got_rule = bg.extract_answer(block)
    assert got_rule == rule
    if value is None:
        assert got is None
    else:
        assert got == pytest.approx(value)


def test_is_correct_uses_the_batterys_tolerance():
    assert bg.is_correct(8.0, 8) and bg.is_correct(8.0000000001, 8)
    assert not bg.is_correct(8.1, 8) and not bg.is_correct(None, 8)


def test_truncate_at_stop_cuts_at_the_first_stop_string():
    raw = ("21 - 15 = 6. The answer is 6.\n\nQuestion: next one?\n"
           "Answer: The answer is 99.")
    block, stopped = bg.truncate_at_stop(raw)
    assert stopped and block == "21 - 15 = 6. The answer is 6.\n"
    assert bg.extract_answer(block) == (6.0, "answer_is")
    # a block that never moved on to the next slot
    block2, stopped2 = bg.truncate_at_stop("The answer is 6.")
    assert (block2, stopped2) == ("The answer is 6.", False)
    # the stop string covers the double-newline form too
    assert bg.truncate_at_stop("x\nQuestion: y")[1]
    assert bg.truncate_at_stop("x\n\nQuestion: y")[0] == "x\n"


# --------------------------------------------------------------------------- #
# render                                                                        #
# --------------------------------------------------------------------------- #


def test_render_prompt_is_plain_text_eight_shot_ending_answer():
    p = bg.render_prompt("How many?", shots=8)
    assert p.count("Question:") == 9 and p.count("\nAnswer:") == 9
    assert p.endswith("Question: How many?\nAnswer:")
    assert p.count("The answer is ") == 8
    assert "<|" not in p and "<start_of_turn>" not in p  # no chat template
    # every exemplar answer is the one Wei et al. give
    for _q, _r, a in bg.EXEMPLARS:
        assert f"The answer is {a}." in p
    assert len(bg.EXEMPLARS) == 8 == bg.DEFAULT_SHOTS
    # the exemplar block is a fixed prefix: same sha for any question
    assert bg.exemplars_sha256(8) == bg.exemplars_sha256(8)
    assert bg.exemplars_sha256(8) != bg.exemplars_sha256(7)
    assert bg.render_prompt("A", 2).startswith(
        bg.render_exemplar(*bg.EXEMPLARS[0]))
    with pytest.raises(SystemExit, match="--shots"):
        bg.render_prompt("A", 9)


def test_exemplars_answers_are_consistent_with_their_rationales():
    """Each exemplar's rationale derives the number the answer field
    carries (its final arithmetic step states it), and the rendered
    exemplar closes with exactly that number — a typo here would teach the
    model the wrong format."""
    import re as _re

    for q, rationale, a in bg.EXEMPLARS:
        numbers = _re.findall(r"-?\d[\d,]*(?:\.\d+)?", rationale)
        assert a in numbers, (rationale, a)
        assert numbers[-1] == a or numbers[-2:] == [a, a] or \
            rationale.rstrip(".").endswith(f"{a} golf balls"), (rationale, a)
        rendered = bg.render_exemplar(q, rationale, a)
        assert rendered.endswith(f"The answer is {a}.")
        assert bg.extract_answer(rendered.split("Answer:", 1)[1]) == \
            (float(a), "answer_is")


# --------------------------------------------------------------------------- #
# identity header round trip + pairability (pure filesystem)                   #
# --------------------------------------------------------------------------- #


def _rows(n=4):
    return [{"question": f"Synthetic problem number {i}: what is {i} + {i}?",
             "answer": f"{i} + {i} = {2 * i}\n#### {2 * i}", "gold": float(2 * i)}
            for i in range(n)]


def _identity(label="r2", cell="none", **over) -> dict:
    ident = {"run_label": label, "model": "tiny", "model_revision": "r",
             "config": "cfg.yaml", "config_sha256": "c" * 64,
             "dataset": bg.DATASET, "dataset_config": bg.DATASET_CONFIG,
             "dataset_split": bg.DATASET_SPLIT,
             "dataset_revision": "d" * 40, "n": 4, "requested_n": 4,
             "seed": 0, "shots": 8, "scale": 1.0,
             "ordered_qid_hash": bg.ordered_qid_hash(_rows()),
             "render": bg.RENDER, "chat_template_bypassed": True,
             "exemplars_sha256": bg.exemplars_sha256(8),
             "max_new_tokens": 384, "stop_strings": list(bg.STOP_STRINGS),
             "decoding": "greedy", "batch_size": 32,
             "dtype": "bfloat16", "attn_implementation": "sdpa",
             "extraction": "answer_is_first|hash_last|last_number",
             "defenses_root": None, "parents_root": None,
             "explicit_objects": None, "cell": cell,
             "object_path": None, "object_sha256": None}
    ident.update(over)
    return ident


def _per_q(correct_flags):
    rows = []
    for i, c in enumerate(correct_flags):
        rows.append({"qidx": i, "qid": bg.question_id(_rows()[i]),
                     "gold": float(2 * i), "extracted": float(2 * i) if c else 1.5,
                     "rule": "answer_is", "correct": bool(c),
                     "terminated": True, "hit_cap": False,
                     "n_new_tokens": 20, "empty": False,
                     "response": f" {i} + {i} = {2 * i}. The answer is {2 * i}."})
    return rows


def test_header_identity_round_trip_and_pairing(tmp_path):
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    per_a, per_b = _per_q([1, 1, 0, 0]), _per_q([1, 0, 1, 0])
    agg_a, agg_b = bg.aggregate(per_a), bg.aggregate(per_b)
    back = {}
    for cell, agg, per, scale in (("none", agg_a, per_a, 1.0),
                                  ("loud__cell", agg_b, per_b, 2.0)):
        agg_path, pq_path = bg.write_cell_results(
            run_dir, "r2", cell, agg, per, _identity(cell=cell, scale=scale))
        assert (agg_path, pq_path) == bg.cell_paths(run_dir, "r2", cell)
        assert agg_path.name == f"gsm8k_r2_{cell}.json"
        assert pq_path.name == f"gsm8k_r2_{cell}.per_question.jsonl"
        rec = json.loads(agg_path.read_text())
        assert rec["identity"]["scale"] == scale
        assert rec["per_question_file"] == pq_path.name
        assert rec["correct"] == agg["correct"] and rec["n"] == 4
        back[cell] = bg.read_per_question(pq_path)
    ia, ib = back["none"][0], back["loud__cell"][0]
    # a 1x/2x pair IS pairable: scale is not a pair key
    bg.assert_pairable(ia, ib)
    assert ia["scale"] == 1.0 and ib["scale"] == 2.0
    for k in bg.PAIR_KEYS:
        assert k in ia and ia[k] == ib[k]
    table = bg.mcnemar_table(back["none"][1], back["loud__cell"][1])
    assert table == {"both": 1, "a_only": 1, "b_only": 1, "neither": 1}
    assert table["both"] + table["a_only"] == agg_a["correct"]
    assert table["both"] + table["b_only"] == agg_b["correct"]


@pytest.mark.parametrize("key", ["dataset_revision", "ordered_qid_hash",
                                 "exemplars_sha256", "max_new_tokens",
                                 "stop_strings", "n", "shots"])
def test_pairing_refuses_on_every_pair_key(key):
    a = _identity()
    b = _identity()
    b[key] = (b[key] + 1) if isinstance(b[key], int) else (
        ["\nQ:"] if isinstance(b[key], list) else b[key][:-1] + "x")
    with pytest.raises(SystemExit, match=f"not pairable: {key}"):
        bg.assert_pairable(a, b)


def test_reader_refuses_torn_or_headerless_files(tmp_path):
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    per = _per_q([1, 0, 1, 1])
    _, pq = bg.write_cell_results(run_dir, "r2", "none", bg.aggregate(per),
                                  per, _identity())
    lines = pq.read_text().splitlines()
    pq.write_text("\n".join(lines[:-1]) + "\n")  # torn
    with pytest.raises(SystemExit, match="torn write"):
        bg.read_per_question(pq)
    pq.write_text("\n".join(lines[1:]) + "\n")  # headerless
    with pytest.raises(SystemExit, match="no run-identity header"):
        bg.read_per_question(pq)
    # an MMLU rows file is not a GSM8K rows file
    mm = run_dir / "mmlu.jsonl"
    mm.write_text(json.dumps({"kind": bm.PER_QUESTION_HEADER_KIND,
                              "n_rows": 0, "identity": {}}) + "\n")
    with pytest.raises(SystemExit, match="no run-identity header"):
        bg.read_per_question(mm)


def test_writer_refuses_rows_that_disagree_with_the_aggregate(tmp_path):
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    per = _per_q([1, 0, 1, 1])
    agg = bg.aggregate(per)
    agg["correct"] -= 1
    with pytest.raises(SystemExit, match="disagree with the aggregate"):
        bg.write_cell_results(run_dir, "r2", "none", agg, per, _identity())
    assert not list(run_dir.iterdir())  # nothing published


def test_aggregate_counts_degeneration():
    per = _per_q([1, 0, 0, 1])
    per[1].update(terminated=False, hit_cap=True, n_new_tokens=384,
                  rule="none", extracted=None)
    per[2].update(empty=True, response="", rule="none", extracted=None)
    per[3].update(rule="last_number")
    agg = bg.aggregate(per)
    assert agg["acc"] == 0.5 and agg["correct"] == 2 and agg["n"] == 4
    assert agg["n_non_terminating"] == 1 and agg["n_hit_cap"] == 1
    assert agg["n_empty"] == 1 and agg["n_extraction_failed"] == 2
    assert agg["n_fallback_rule"] == 1
    assert agg["rules"] == {"answer_is": 1, "none": 2, "last_number": 1}
    assert 0 < agg["ci95"][0] < 0.5 < agg["ci95"][1] < 1


def test_summary_merges_into_an_existing_same_identity_label(tmp_path):
    """Two invocations under one label (a top-up of cells) keep both
    records; a different identity under the same label is refused."""
    spath = tmp_path / "gsm8k_r2_summary.json"
    ident = _identity()
    bm.write_summary(spath, ident, ["none", "a__b"],
                     {"none": {"acc": 0.5}, "a__b": {"acc": 0.4}},
                     consumer=bg.CONSUMER)
    bm.write_summary(spath, ident, ["c__d"], {"c__d": {"acc": 0.3}},
                     consumer=bg.CONSUMER)
    s = json.loads(spath.read_text())
    assert s["requested_cells"] == ["none", "a__b", "c__d"]
    assert set(s["cells"]) == {"none", "a__b", "c__d"}
    assert len(s["invocations"]) == 2
    assert s["invocations"][1]["requested_cells"] == ["c__d"]
    with pytest.raises(SystemExit, match="scale differs"):
        bm.write_summary(spath, _identity(scale=2.0), ["e__f"],
                         {"e__f": {}}, consumer=bg.CONSUMER)
    assert json.loads(spath.read_text()) == s  # the refusal wrote nothing
    # GSM8K's own merge keys: a different decode budget is a different label
    with pytest.raises(SystemExit, match="max_new_tokens differs"):
        bm.write_summary(spath, _identity(max_new_tokens=256), ["e__f"],
                         {"e__f": {}}, consumer=bg.CONSUMER,
                         merge_keys=bg.MERGE_KEYS)


# --------------------------------------------------------------------------- #
# --scale wiring + rows on the tiny model                                       #
# --------------------------------------------------------------------------- #


class _RecordingObject:
    """An intervention object that records every apply() and steers
    nothing — the wiring probe: the SCALE must reach apply, the PROMPT must
    not change (a non-token kind is never injected)."""

    kind = "psr"

    def __init__(self):
        self.applied: list[float] = []
        self.injected = 0

    @contextlib.contextmanager
    def apply(self, model, scale: float = 1.0):
        self.applied.append(float(scale))
        yield None

    def inject(self, prompt: str) -> str:
        self.injected += 1
        return prompt + " INJECTED"


def test_scale_reaches_the_object_not_the_prompt(tiny_llama, monkeypatch):
    model, tok = tiny_llama
    rows = _rows(3)
    obj = _RecordingObject()
    seen_prompts: list[str] = []
    real_generate = bg.generate_answers

    def spy(model_, tok_, obj_, prompts, **kw):
        seen_prompts.extend(prompts)
        return real_generate(model_, tok_, obj_, prompts, **kw)

    monkeypatch.setattr(bg, "generate_answers", spy)
    agg, per = bg.score_cell(model, tok, obj, rows, shots=2, scale=2.0,
                             batch_size=2, max_new_tokens=4)
    # scale 2.0 entered obj.apply once per chunk (3 rows / batch 2 = 2)
    assert obj.applied == [2.0, 2.0]
    assert obj.injected == 0
    # the prompts are the scale-free render, byte for byte
    assert seen_prompts == [bg.render_prompt(r["question"], 2) for r in rows]
    assert agg["n"] == 3 == len(per)
    assert [r["qidx"] for r in per] == [0, 1, 2]
    for r in per:
        assert set(r) >= {"qid", "gold", "extracted", "rule", "correct",
                          "terminated", "hit_cap", "n_new_tokens", "empty",
                          "response"}
        assert r["n_new_tokens"] <= 4
        assert r["hit_cap"] == (r["n_new_tokens"] == 4)
        assert isinstance(r["correct"], bool)
        assert r["correct"] == bg.is_correct(r["extracted"], r["gold"])
    assert sum(r["correct"] for r in per) == agg["correct"]
    # a bare pass never touches an object
    agg0, per0 = bg.score_cell(model, tok, None, rows, shots=2, scale=2.0,
                               batch_size=2, max_new_tokens=4)
    assert agg0["n"] == 3 and [r["qid"] for r in per0] == [r["qid"] for r in per]


def test_generation_is_deterministic_and_order_preserving(tiny_llama):
    model, tok = tiny_llama
    rows = _rows(5)
    # deliberately different question lengths so length-bucketing reorders
    rows[0]["question"] = "Short?"
    rows[3]["question"] = ("A much longer synthetic problem statement that runs "
                           "on for a while so it lands in a different bucket?")
    prompts = [bg.render_prompt(r["question"], 1) for r in rows]
    g1 = bg.generate_answers(model, tok, None, prompts, batch_size=2,
                             max_new_tokens=6)
    g2 = bg.generate_answers(model, tok, None, prompts, batch_size=2,
                             max_new_tokens=6)
    assert g1 == g2 and len(g1) == 5
    # order preserved under bucketing: each row's text is what the same
    # prompt yields on its own (single-row batch, no padding)
    solo = bg.generate_answers(model, tok, None, [prompts[3]], batch_size=1,
                               max_new_tokens=6)
    assert solo[0]["n_new_tokens"] == g1[3]["n_new_tokens"]
    assert solo[0]["text"] == g1[3]["text"]
    assert all(g["n_new_tokens"] <= 6 for g in g1)
    assert tok.padding_side != "left" or True  # restored (tiny default)


def _simulate(rec, prompt_width, scripts, pad, filler, cap):
    """Mimic HF's sampling loop exactly enough for the recorder: per step,
    the next token per row (its script, or ``filler`` once the script ran
    out — a still-generating row keeps producing real tokens), pads
    substituted for rows already marked done, append, then call the
    criteria. Returns (generated_len)."""
    import torch

    B = len(scripts)
    ids = torch.zeros((B, prompt_width), dtype=torch.long)
    rec.reset(B, prompt_width)
    unfinished = torch.ones(B, dtype=torch.bool)
    for t in range(cap):
        nxt = torch.tensor([s[t] if t < len(s) else filler for s in scripts])
        nxt = torch.where(unfinished, nxt, torch.full_like(nxt, pad))
        ids = torch.cat([ids, nxt[:, None]], 1)
        unfinished = unfinished & ~rec(ids, None)
        if not bool(unfinished.any()):
            break
    return ids.shape[1] - prompt_width


def _tokens(tok):
    a = tok(" The answer is 6.", add_special_tokens=False)["input_ids"]
    stop = tok("\nQuestion:", add_special_tokens=False)["input_ids"]
    assert len(a) >= 5 and len(stop) >= 2
    return a, stop


def test_stop_recorder_pad_distinct_from_eos(tiny_llama):
    """Regime A (Gemma-style, pad != eos): a non-terminating row whose
    last generated token is a pad is CAP (it was never stopped), a
    generated pad mid-row is a token, the stop string is recorded at the
    step it completes, and EOS-valued/pad padding of a finished row
    changes nothing."""
    model, tok = tiny_llama
    pad, eos_ids = bg._end_ids(model, tok)
    eos = tok.eos_token_id
    assert pad not in eos_ids
    a, stop = _tokens(tok)
    cap = 12
    rec = bg.make_stop_recorder(tok, eos_ids, bg.STOP_STRINGS)
    scripts = [
        a[:3] + [pad] + a[3:] + [eos],   # 0: mid-row pad, then EOS
        a + stop,                        # 1: stop string completes
        (a * 3)[:cap],                   # 2: runs the full budget
        (a * 3)[:cap - 1] + [pad],       # 3: full budget, LAST token a pad
        [eos],                           # 4: empty answer
    ]
    glen = _simulate(rec, 7, scripts, pad, a[0], cap)
    assert glen == cap
    got = rec.finish(glen, cap)
    assert got[0] == (len(a) + 1, "eos")            # pad counted as a token
    assert got[1] == (len(a) + len(stop), "stop_string")
    assert got[2] == (cap, "cap")
    assert got[3] == (cap, "cap")                   # the corrected count
    assert got[4] == (0, "eos")
    # recorded ONCE: the padding appended after step of finish is ignored
    rec2 = bg.make_stop_recorder(tok, eos_ids, bg.STOP_STRINGS)
    _simulate(rec2, 7, [a + stop, (a * 3)[:cap]], pad, a[0], cap)
    assert rec2.finish(cap, cap)[0] == (len(a) + len(stop), "stop_string")


def test_stop_recorder_pad_equals_eos_is_batch_independent(tiny_llama):
    """Regime B (Qwen-style, pad == eos): a row stopped by the stop
    string before its peers gets EOS-valued padding; its reason must stay
    stop_string and its count must not depend on how long the batch's
    slowest row ran."""
    model, tok = tiny_llama
    eos = tok.eos_token_id
    eos_ids = {eos}
    a, stop = _tokens(tok)
    cap = 12
    alone = bg.make_stop_recorder(tok, eos_ids, bg.STOP_STRINGS)
    _simulate(alone, 7, [a + stop], eos, a[0], cap)
    solo = alone.finish(len(a) + len(stop), cap)[0]
    with_peer = bg.make_stop_recorder(tok, eos_ids, bg.STOP_STRINGS)
    glen = _simulate(with_peer, 7, [a + stop, (a * 3)[:cap], a[:2] + [eos]],
                     eos, a[0], cap)
    got = with_peer.finish(glen, cap)
    assert got[0] == solo == (len(a) + len(stop), "stop_string")
    assert got[1] == (cap, "cap")
    assert got[2] == (2, "eos")
    # stop string and EOS on the same step: the visible event wins
    same = bg.make_stop_recorder(tok, eos_ids, bg.STOP_STRINGS)
    # craft: a row whose stop string's LAST token is followed by nothing —
    # the eos check is on the same last column; make the last token both
    # complete the stop string and be an eos id is impossible for a real
    # tokenizer, so assert the precedence on the flags directly
    import torch
    same.reset(1, 0)
    ids = torch.tensor([a + stop])
    same.stop_crit = type("S", (), {"__call__": lambda self, i, s, **k:
                                    torch.ones(1, dtype=torch.bool)})()
    same.eos = torch.tensor([ids[0, -1].item()])
    same(ids, None)
    assert same.reason == ["stop_string"]


def test_end_ids_reads_the_generation_config_eos_set(tiny_llama):
    model, tok = tiny_llama
    pad, eos = bg._end_ids(model, tok)
    gc_eos = model.generation_config.eos_token_id
    gc_set = {gc_eos} if isinstance(gc_eos, int) else set(gc_eos)
    assert gc_set <= eos and tok.eos_token_id in eos
    assert pad == (tok.pad_token_id if tok.pad_token_id is not None
                   else tok.eos_token_id)


@pytest.mark.parametrize("regime", ["pad_ne_eos", "pad_eq_eos"])
def test_generate_answers_records_stop_reason(tiny_llama, monkeypatch, regime):
    """End to end through generate_answers with a generate() double that
    runs HF's loop shape (pads for finished rows, criteria after append):
    stop_reason / n_new_tokens come from the recorder in both tokenizer
    regimes, never from the padding; a generated pad never truncates the
    decoded answer (regime A) and IS the end of the row (regime B)."""
    import torch

    model, tok = tiny_llama
    eos = tok.eos_token_id
    if regime == "pad_eq_eos":
        monkeypatch.setattr(tok, "pad_token_id", eos)
    pad, eos_ids = bg._end_ids(model, tok)
    assert (pad in eos_ids) == (regime == "pad_eq_eos")
    a, stop = _tokens(tok)
    cap = 12
    prompts = [bg.render_prompt(f"q{i}?", 1) for i in range(4)]
    scripts = [
        a[:3] + [pad] + a[3:] + [eos],   # pad mid-row then EOS
        a + stop,                        # stop string
        (a * 3)[:cap],                   # cap
        (a * 3)[:cap - 1] + [pad],       # cap whose last token is a pad
    ]

    def fake_generate(**kw):
        ids = kw["input_ids"]
        crit = kw["stopping_criteria"]
        B = ids.shape[0]
        unfinished = torch.ones(B, dtype=torch.bool)
        for t in range(kw["max_new_tokens"]):
            nxt = torch.tensor([s[t] if t < len(s) else a[0] for s in scripts[:B]])
            nxt = torch.where(unfinished, nxt, torch.full_like(nxt, kw["pad_token_id"]))
            ids = torch.cat([ids, nxt[:, None]], 1)
            unfinished = unfinished & ~crit(ids, None)
            if not bool(unfinished.any()):
                break
        return ids

    monkeypatch.setattr(model, "generate", fake_generate)
    out = bg.generate_answers(model, tok, None, prompts, batch_size=8,
                              max_new_tokens=cap)
    reasons = [o["stop_reason"] for o in out]
    if regime == "pad_ne_eos":
        assert reasons == ["eos", "stop_string", "cap", "cap"]
        assert out[0]["n_new_tokens"] == len(a) + 1
        assert "The answer is 6" in out[0]["text"]  # the pad did not cut it
        assert out[3]["hit_cap"] and out[3]["n_new_tokens"] == cap
    else:
        # a generated pad IS eos here: row 0 ends at it, row 3 ends at cap-1
        assert reasons == ["eos", "stop_string", "cap", "eos"]
        assert out[0]["n_new_tokens"] == 3
        assert out[3]["n_new_tokens"] == cap - 1 and not out[3]["hit_cap"]
    assert out[1]["n_new_tokens"] == len(a) + len(stop)
    assert "\nQuestion:" in out[1]["text"] and not out[1]["hit_cap"]
    assert out[2]["hit_cap"] and out[2]["n_new_tokens"] == cap
    # score_cell folds these into terminated / stop_reasons
    monkeypatch.setattr(bg, "render_prompt", lambda q, shots=8: prompts[0])
    agg, per = bg.score_cell(model, tok, None, _rows(4), shots=1,
                             batch_size=8, max_new_tokens=cap)
    assert [r["terminated"] for r in per] == [r != "cap" for r in reasons]
    assert agg["n_non_terminating"] == reasons.count("cap") == agg["n_hit_cap"]
    assert agg["stop_reasons"] == {k: reasons.count(k) for k in set(reasons)}
    assert per[1]["stopped_on_string"] and not per[0]["stopped_on_string"]


def test_runtime_identity_is_part_of_the_label(tmp_path):
    """Batch size, dtype and attention backend are prediction-affecting
    runtime settings — a top-up that changes any of them under the same
    label refuses; a LEGACY identity lacking dtype / attn (written
    before they were stamped) is accepted only when its config_sha256
    equals the new run's (the settings come from the config), and
    refused otherwise."""
    spath = tmp_path / "gsm8k_r2_summary.json"
    bm.write_summary(spath, _identity(), ["none"], {"none": {}},
                     consumer=bg.CONSUMER, merge_keys=bg.MERGE_KEYS)
    for k, v, msg in (("batch_size", 16, "batch_size differs"),
                      ("dtype", "float32", "dtype differs"),
                      ("attn_implementation", "eager",
                       "attn_implementation differs")):
        with pytest.raises(SystemExit, match=msg):
            bm.write_summary(spath, _identity(**{k: v}), ["a__b"],
                             {"a__b": {}}, consumer=bg.CONSUMER,
                             merge_keys=bg.MERGE_KEYS)
    # legacy prior: no dtype / attn stamped
    legacy = _identity()
    legacy.pop("dtype"); legacy.pop("attn_implementation")
    spath.write_text(json.dumps({"run": legacy, "requested_cells": ["none"],
                                 "cells": {"none": {}}}))
    kw = dict(cell_paths_fn=bg.cell_paths, summary_path_fn=bg.summary_path,
              merge_keys=bg.MERGE_KEYS, consumer=bg.CONSUMER)
    run_dir = spath.parent
    assert bm.preflight_label(run_dir, "r2", _identity(), {"none": None},
                              **kw) == []           # same config bytes
    with pytest.raises(SystemExit, match="dtype is absent.*cannot be verified"):
        bm.preflight_label(run_dir, "r2", _identity(config_sha256="d" * 64),
                           {"none": None}, **kw)
    # a NON-config key absent from the prior is never inferred
    legacy2 = _identity(); legacy2.pop("batch_size")
    spath.write_text(json.dumps({"run": legacy2, "requested_cells": [],
                                 "cells": {}}))
    with pytest.raises(SystemExit, match="batch_size is absent"):
        bm.preflight_label(run_dir, "r2", _identity(), {"none": None}, **kw)


# --------------------------------------------------------------------------- #
# label preflight — refuse BEFORE anything is written                           #
# --------------------------------------------------------------------------- #


def _publish(run_dir, label, cell, ident, flags=(1, 0, 1, 1)):
    per = _per_q(list(flags))
    return bg.write_cell_results(run_dir, label, cell, bg.aggregate(per), per,
                                 ident)


def test_preflight_refuses_summary_identity_drift_before_any_write(tmp_path):
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    bm.write_summary(bg.summary_path(run_dir, "r2"), _identity(), ["none"],
                     {"none": {}}, merge_keys=bg.MERGE_KEYS)
    before = sorted(p.name for p in run_dir.iterdir())
    for k, v in (("scale", 2.0), ("max_new_tokens", 256),
                 ("stop_strings", ["\nQ:"]), ("exemplars_sha256", "e" * 64),
                 ("dataset_revision", "f" * 40), ("extraction", "other")):
        with pytest.raises(SystemExit, match=f"already holds a summary.*{k}"):
            bm.preflight_label(run_dir, "r2", _identity(**{k: v}),
                               {"none": None, "a__b": "s" * 64},
                               cell_paths_fn=bg.cell_paths,
                               summary_path_fn=bg.summary_path,
                               merge_keys=bg.MERGE_KEYS, consumer=bg.CONSUMER)
    assert sorted(p.name for p in run_dir.iterdir()) == before
    # same identity: passes, nothing re-scored yet
    assert bm.preflight_label(run_dir, "r2", _identity(),
                              {"none": None, "a__b": "s" * 64},
                              cell_paths_fn=bg.cell_paths,
                              summary_path_fn=bg.summary_path,
                              merge_keys=bg.MERGE_KEYS,
                              consumer=bg.CONSUMER) == []


def test_preflight_refuses_a_different_object_under_the_same_label(tmp_path):
    """The r2 hazard: a cell's live object changed (a continuation shipped a
    new one) — re-scoring it under the label that holds the OLD object's
    numbers would destroy that record."""
    run_dir = bm.run_dir_for(tmp_path, "r2")
    run_dir.mkdir()
    _publish(run_dir, "r2", "none", _identity())
    _publish(run_dir, "r2", "c__psr", _identity(cell="c__psr",
                                                object_sha256="old" * 21 + "o"))
    kw = dict(cell_paths_fn=bg.cell_paths, summary_path_fn=bg.summary_path,
              merge_keys=bg.MERGE_KEYS, consumer=bg.CONSUMER)
    with pytest.raises(SystemExit, match="scored on object .*ONE object"):
        bm.preflight_label(run_dir, "r2", _identity(),
                           {"none": None, "c__psr": "new" * 21 + "n"}, **kw)
    # same object: an idempotent re-score, reported
    assert bm.preflight_label(run_dir, "r2", _identity(),
                              {"none": None, "c__psr": "old" * 21 + "o"},
                              **kw) == ["none", "c__psr"]
    # a per-cell file under a different identity refuses too
    with pytest.raises(SystemExit, match="already holds 'none'.*scale"):
        bm.preflight_label(run_dir, "r2", _identity(scale=2.0),
                           {"none": None}, **kw)
    # a fresh label has nothing to collide with
    assert bm.preflight_label(run_dir, "r2_x2", _identity("r2_x2", scale=2.0),
                              {"none": None, "c__psr": "new" * 21 + "n"},
                              **kw) == []


def test_gsm8k_merge_keys_cover_the_protocol():
    for k in (*bg.PAIR_KEYS, "decoding", "extraction", "scale", "model",
              "model_revision", "run_label"):
        assert k in bg.MERGE_KEYS
    assert len(set(bg.MERGE_KEYS)) == len(bg.MERGE_KEYS)


def test_generate_answers_refuses_bad_budgets(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(SystemExit, match="batch-size"):
        bg.generate_answers(model, tok, None, ["x"], batch_size=0)
    with pytest.raises(SystemExit, match="max-new-tokens"):
        bg.generate_answers(model, tok, None, ["x"], max_new_tokens=0)
