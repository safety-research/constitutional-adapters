"""The PER-MODEL batched-token budget in ``sisteer.capeval.serving.geometry``
(the rule, 2026-09-03: make the per-model budget, then measure each
model's throughput on the cluster) — CPU-only.

Covered: every canonical model has an entry in the analyzer's paste block (the
2026-09-04 sweep, folded in verbatim — the eight literals are pinned; llama8b's row
is the sdfbase-llama8b serving gate's 2026-09-10 lens-stack measurement, 475,136 =
29 x 16384, which replaced both the swept 458,752 and the 2026-09-06 ceiling), TWO
of which are shadowed by conservative ceilings from the 2026-09-06 lens-stack
engine-init failures (:data:`geometry._PROVISIONAL_BUDGETS`), while the certified
llama8b Petri footprint (393,216, citing vllm_server.py's negative result)
stays the no-model default; ``budget_for`` returns the six unshadowed rows as
``measured`` with NO warning and the shadowed two as
``provisional-conservative`` WITH one, a future unmeasured key as
the Llama-certified default WITH a warning naming the model, ``None`` as the
silent default, and refuses an unknown key; the hybrid families cap S at 256
through ``seqs_at`` and ``resolve_geometry`` sizes through it;
``identity_seqs(L, model_key=...)`` and ``resolve_geometry(eval,
model_key=...)`` size S from the model's budget (a fake measured entry proves
the plumbing) while every existing call stays Llama-sized; ``ModelBudget``
refuses a budget that is not its probe's product; ``describe_budgets`` lists
every canonical key; the paste-block markers exist in the module source.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import vllm_server as vs

REPO = Path(__file__).resolve().parents[1]


def _fake_budget(key: str, budget: int, L: int = 8192) -> sg.ModelBudget:
    return sg.ModelBudget(
        model_key=key, batched_token_budget=budget,
        basis=sg.BudgetBasis(
            method="test", max_model_len=L, max_num_seqs=budget // L,
            gpu_memory_utilization=0.9, kv_tokens_reported=budget + 1000,
            probe_job_id="1", measured_on="2026-09-04", source="test"))


def _swept(key: str) -> sg.ModelBudget:
    """The analyzer's paste-block entry for ``key`` (the 2026-09-04 sweep, or llama8b's
    2026-09-10 lens-stack row), whether or not a provisional ceiling shadows it in
    ``BUDGET_BY_MODEL``."""
    return next(b for b in sg._MEASURED_BUDGETS if b.model_key == key)


#: the two rows still SHADOWED by a conservative ceiling (2026-09-06, the
#: tiny-capability standup smokes): the ceiling, its S at the certified 16384 window,
#: and the product that was OBSERVED to fail (None = not smoked, halved as the
#: same-stack precaution). llama8b's ceiling was replaced by its lens-stack measurement
#: on 2026-09-10 (the sdfbase-llama8b serving gate); qwen35_4b's halving was lifted on
#: 2026-09-11 (six lens-stack standups at 148 x 4864 = the swept 720,896) and its row was
#: RE-MEASURED 2026-09-13 at the MATH-500 window on the lens stack (741,888 = 42 x 17,664).
PROVISIONAL = {
    "qwen35_9b": (196_608, 12, 589_824),
    "llama3b":   (327_680, 20, None),
}


def test_model_keys_are_the_canonical_eight():
    assert sg.MODEL_KEYS == ("llama8b", "llama3b", "gemma4e4b", "gemma4e2b",
                             "qwen35_9b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
    assert len(set(sg.MODEL_KEYS)) == len(sg.MODEL_KEYS)


#: the eight MEASURED budgets: budget, probe S at the basis window, attested KV
#: tokens, the first refused product, the probe job — the analyzer's paste block.
#: Six are the 2026-09-04 sweep at L=8192; llama8b is the 2026-09-10 serving gate's
#: lens-stack row at L=16384 and qwen35_4b the 2026-09-13 lens-stack budget walk AT the
#: MATH-500 window L=17664 (MEASURED_BASIS names the exceptions)
MEASURED = {
    "llama8b":         (475_136, 29, 505_504, 507_904, "476440"),
    "llama3b":         (655_360, 80, 733_712, 720_896, "375170"),
    "gemma4e4b":       (589_824, 72, 625_473, 655_360, "65674"),
    "gemma4e2b":       (786_432, 96, 960_543, 917_504, "65677"),
    "qwen35_9b":       (589_824, 72, 720_033, 655_360, "65727"),
    "qwen35_4b":       (741_888, 42, 850_259, 777_216, "100196"),
    "glm46vflash":     (786_432, 96, 867_584, 917_504, "375192"),
    "nemotron3nano4b": (786_432, 96, 993_416, 917_504, "65668"),
}
#: model key -> (basis window, measured_on) where a row is NOT the 2026-09-04 sweep at 8192
MEASURED_BASIS = {"llama8b": (16384, "2026-09-10"), "qwen35_4b": (17664, "2026-09-13")}


def test_the_certified_default_entry_is_the_llama_petri_footprint():
    b = sg._LLAMA8B_CERTIFIED
    assert b.certified is True and b.model_key == "llama8b"
    assert b.batched_token_budget == sg.BATCHED_TOKEN_BUDGET == 393_216
    assert (b.basis.max_model_len, b.basis.max_num_seqs) == (16384, vs.DEFAULT_MAX_NUM_SEQS)
    assert b.basis.first_refused == 524_288            # 32 x 16384, the wall
    assert b.basis.gpu_memory_utilization == 0.90
    assert "vllm_server.py" in b.basis.source           # cites the negative result
    assert "524,288" in b.basis.method and "does NOT fit" in b.basis.method
    assert b.seqs_at(16384) == 24 and b.seqs_at(8192) == 48
    # it is the NO-MODEL default, not the llama8b table entry any more
    assert sg.budget_for(None).entry is b
    assert sg.BUDGET_BY_MODEL["llama8b"] is not b


def test_all_eight_models_are_measured_to_the_pinned_literals():
    """The analyzer's paste block holds the six 2026-09-04 measurements verbatim
    plus llama8b's 2026-09-10 and qwen35_4b's 2026-09-13 lens-stack rows — a provisional ceiling SHADOWS a row
    in ``BUDGET_BY_MODEL``, it never edits the generated block, so a measurement
    stays on the record until a newer one replaces it."""
    assert tuple(sg.BUDGET_BY_MODEL) == sg.MODEL_KEYS == tuple(MEASURED)
    for key, (budget, S, kv, wall, job) in MEASURED.items():
        b = _swept(key)
        L, day = MEASURED_BASIS.get(key, (8192, "2026-09-04"))
        assert b.source == sg.MEASURED_BUDGET_SOURCE and not b.provisional, key
        assert b.certified is False, key
        assert b.batched_token_budget == budget == S * L, key
        assert (b.basis.max_model_len, b.basis.max_num_seqs) == (L, S), key
        assert b.basis.kv_tokens_reported == kv and b.basis.first_refused == wall, key
        assert b.basis.probe_job_id == job and b.basis.measured_on == day, key
        assert b.basis.gpu_memory_utilization == 0.9 and "kv_coverage_problem" in b.basis.method
        assert b.basis.first_refused > budget and "wall reached" in b.basis.note, key
        assert len(sg._MEASURED_BUDGETS) == 8
    # the block in the source is the analyzer's, verbatim (markers + header line)
    src = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    blk = src[src.index("# MEASURED BY scripts/served_decode_sweep.py --phase budget  (BEGIN)"):
              src.index("# MEASURED BY scripts/served_decode_sweep.py --phase budget  (END)")]
    assert "# (pasted by scripts/geom_sweep_analyze.py" in blk        # + the <SHARED> note
    assert blk.count("ModelBudget(") == 8 and "<no measured entries yet>" not in blk


def test_two_rows_are_provisional_ceilings_from_the_2026_09_06_failures():
    """llama8b OOM'd at 458,752 and qwen35_9b raised CUBLAS_STATUS_EXECUTION_FAILED
    at 589,824 in engine init on the lens stack at L=16384; those two rows were
    replaced by ceilings, and their unsmoked siblings halved. Gemma is untouched —
    gemma4e4b PASSED at the same 589,824. llama8b's ceiling was itself replaced on
    2026-09-10 by the serving gate's lens-stack measurement (29 x 16384), and qwen35_4b's
    halving was lifted 2026-09-11 (six lens-stack standups at 148 x 4864 = the swept
    720,896) before its row was re-measured at 17,664 on 2026-09-13, so two ceilings remain."""
    assert tuple(b.model_key for b in sg._PROVISIONAL_BUDGETS) == tuple(PROVISIONAL)
    assert "llama8b" not in PROVISIONAL and "qwen35_4b" not in PROVISIONAL and len(sg._PROVISIONAL_BUDGETS) == 2
    for key, (ceiling, S16k, refused) in PROVISIONAL.items():
        b = sg.BUDGET_BY_MODEL[key]
        assert b.source == sg.PROVISIONAL_BUDGET_SOURCE == "provisional-conservative", key
        assert b.provisional and b.certified is False, key
        assert b.batched_token_budget == ceiling, key
        # the ceiling is expressed AT the certified window, and it LOWERS the sweep's
        assert b.basis.max_model_len == sg.PETRI_MAX_MODEL_LEN, key
        assert b.basis.max_num_seqs == S16k == b.seqs_at(sg.PETRI_MAX_MODEL_LEN), key
        assert ceiling < _swept(key).batched_token_budget, key
        # nothing was measured: no probe, no attested KV, measured_on says so
        assert b.basis.measured_on == sg.UNMEASURED and b.basis.probe_job_id is None, key
        assert b.basis.kv_tokens_reported is None, key
        assert b.basis.first_refused == refused, key
        assert "CONSERVATIVE CEILING, not a measurement" in b.basis.method, key
        assert "2026-09-06" in b.basis.source, key
    # the one with evidence names its failure; the remaining precaution says it has none,
    # and the lifted one records what lifted it
    assert "CUBLAS_STATUS_EXECUTION_FAILED" in sg.BUDGET_BY_MODEL["qwen35_9b"].basis.note
    assert "NOT observed to fail" in sg.BUDGET_BY_MODEL["llama3b"].basis.note
    assert "PRECAUTION LIFTED 2026-09-11" in sg.BUDGET_BY_MODEL["qwen35_4b"].basis.note
    assert "993,378" in sg.BUDGET_BY_MODEL["qwen35_4b"].basis.note and "job 90773" in sg.BUDGET_BY_MODEL["qwen35_4b"].basis.note
    # ... and the 2026-09-13 row names what it supersedes (the sweep row AND the lift), in the same note
    assert "Supersedes the 2026-09-04 bare-engine sweep row 720,896" in sg.BUDGET_BY_MODEL["qwen35_4b"].basis.note
    # llama8b is MEASURED again — on the lens stack this time (the 2026-09-10 serving gate): the
    # row records the 2026-09-06 failure it supersedes, the wall it reached and the margin it keeps
    m = sg.BUDGET_BY_MODEL["llama8b"]
    assert m is _swept("llama8b") and not m.provisional and m.basis.measured_on == "2026-09-10"
    assert "28 x 16384" in m.basis.note and "did NOT reproduce" in m.basis.note
    assert "30 x 16384 = 491,520 ALSO passed" in m.basis.note and m.batched_token_budget == 475_136
    assert "mt_serving_up.py" in m.basis.method and "kv_coverage_problem" in m.basis.method
    # NO Gemma row moved (gemma4e4b passed at 589,824, the budget qwen35_9b died on)
    for key in ("llama8b", "gemma4e4b", "gemma4e2b", "glm46vflash", "nemotron3nano4b"):
        assert sg.BUDGET_BY_MODEL[key] is _swept(key), key
        assert not sg.BUDGET_BY_MODEL[key].provisional, key
    assert sg.BUDGET_BY_MODEL["gemma4e4b"].batched_token_budget == 589_824


def test_llama8b_row_is_the_serving_gates_measured_475_136():
    """sdfbase-llama8b (2026-09-10): the serving gate walked readiness ladders on
    the LENS stack itself (jobs 476439 smoke / 476440 ladder, tree fa9eb1c4, one H200 of
    143,771 MiB) and found the wall at KV coverage near 491.7k batched tokens, not at
    the profile forward — at 16,384 (cap 1,024) S 14 / 23 / 27 / 29 / 30 pass and 31 is
    refused (477,856 < 507,904); at 8,704 S 26 / 43 / 52 / 56 pass, 57 refused; at
    45,056 S 5 / 8 / 10 pass, 11 refused. The 2026-09-06 torch.OutOfMemoryError at 28 x
    16,384 (92.47 GiB held after the profile forward) did NOT reproduce. The row is
    29 x 16,384 = 475,136 — one rung under the literal largest pass, because 30 passes by
    only 160 KV tokens on a 143,771-MiB card and the cluster also has 139.80-GiB H200s —
    and it IS llama8b's ``_MEASURED_BUDGETS`` paste-block row: the swept
    458,752 and the 229,376 ceiling it supersedes are named in its ``basis.note``, and the
    llama8b row of ``_PROVISIONAL_BUDGETS`` is gone. It is the STARTING S only: Modal's KV
    capacity decides at readiness via kv_shrink."""
    budget, S16k, kv, wall, job = MEASURED["llama8b"]
    assert MEASURED_BASIS["llama8b"] == (sg.PETRI_MAX_MODEL_LEN, "2026-09-10")
    b = sg.BUDGET_BY_MODEL["llama8b"]
    assert b is _swept("llama8b") and b is not sg._LLAMA8B_CERTIFIED
    assert "llama8b" not in {p.model_key for p in sg._PROVISIONAL_BUDGETS}
    assert b.source == sg.MEASURED_BUDGET_SOURCE == "measured" and not b.provisional and b.certified is False
    assert b.batched_token_budget == budget == 475_136 == S16k * sg.PETRI_MAX_MODEL_LEN
    assert (b.basis.max_model_len, b.basis.max_num_seqs) == (sg.PETRI_MAX_MODEL_LEN, 29)
    assert b.basis.gpu_memory_utilization == 0.9 and b.basis.measured_on == "2026-09-10"
    assert b.basis.kv_tokens_reported == kv == 505_504 and b.basis.first_refused == wall == 507_904 == 31 * 16384
    assert b.basis.probe_job_id == job == "476440" and budget <= kv < wall
    text = " ".join((b.basis.method, b.basis.source, b.basis.note))
    for s_ in ("did NOT reproduce", "92.47 GiB", "torch.OutOfMemoryError", "476440", "476439", "fa9eb1c4",
               "143,771", "139.80", "160 KV tokens", "14/23/27/29/30 pass", "31 kv_under_coverage", "477,856",
               "26/43/52/56 pass", "5/8/10 pass", "491.7k", "STARTING S", "kv_shrink", "kv_coverage_problem",
               "STACKED-LENS", "458,752", "229,376", "D103"):
        assert s_ in text, s_
    assert b.basis.source.startswith("<SHARED>/")
    assert not re.search(r"/home/|/workspace", text)
    # it is the paste block's own row (one row per key; the block still holds all eight, verbatim)
    assert len(sg._MEASURED_BUDGETS) == 8 and sum(m.model_key == "llama8b" for m in sg._MEASURED_BUDGETS) == 1
    src = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    blk = src[src.index("# MEASURED BY scripts/served_decode_sweep.py --phase budget  (BEGIN)"):
              src.index("# MEASURED BY scripts/served_decode_sweep.py --phase budget  (END)")]
    assert blk.count("ModelBudget(") == 8 and "batched_token_budget=475_136" in blk
    assert "_SERVING_GATE_BUDGETS" not in src
    # budget_for: measured, no warning; it supersedes BOTH the certified 393,216 and the sweep's 458,752
    r = sg.budget_for("llama8b")
    assert r.measured and r.per_model and not r.provisional and r.source == "measured"
    assert r.batched_token_budget == 475_136 and r.entry is b
    # the S it buys: the certified window, the capability window, the tinycap and IFEval windows
    assert sg.identity_seqs(16384, model_key="llama8b") == 29
    assert b.seqs_at(16384) == 29 and b.seqs_at(8192) == 58 and b.seqs_at(1536) == 309
    assert b.seqs_at(2816) == 168 and b.seqs_at(7424) == 64 and 64 * 7424 == 475_136
    assert b.seqs_at(2528) == 187 and b.seqs_at(4576) == 103 and b.seqs_at(8672) == 54
    assert b.seqs_at(11520) == 41 and not b.hybrid and b.seqs_cap is None
    r = sg.resolve_geometry("petri", model_key="llama8b")
    assert (r.max_num_seqs, r.budget_tokens, r.budget_source) == (29, 475_136, sg.MEASURED_BUDGET_SOURCE)
    assert r.is_identity and "model llama8b: measured]" in r.describe() and "PROVISIONAL" not in r.describe()
    md = sg.describe_budgets()
    assert "| llama8b | 475,136 | measured | no | - | 29 | 58 | 309 | 16384 x 29 | 0.90 | 505504 | 507904 | 476440 | 2026-09-10 |" in md
    assert "476439 / 476440" in md and "_SERVING_GATE_BUDGETS" not in md
    # the builder's refusal still guards this row: a ceiling that does not LOWER 475,136 is refused at import
    with pytest.raises(ValueError, match="does not LOWER"):
        _build_with(sg.ModelBudget(
            model_key="llama8b", batched_token_budget=475_136, source=sg.PROVISIONAL_BUDGET_SOURCE,
            basis=sg.BudgetBasis("m", 16384, 29, 0.9, None, None, sg.UNMEASURED, "s")))
    assert sg.BUDGET_BY_MODEL["llama8b"].batched_token_budget == 475_136   # not mutated

def _build_with(extra: sg.ModelBudget) -> dict:
    """``geometry._build_budget_table`` with one more provisional row spliced in."""
    saved = sg._PROVISIONAL_BUDGETS
    try:
        sg._PROVISIONAL_BUDGETS = saved + (extra,)
        return sg._build_budget_table()
    finally:
        sg._PROVISIONAL_BUDGETS = saved


def test_a_provisional_ceiling_must_lower_the_row_it_shadows():
    """The shadow only ever shrinks a number. If a re-run sweep pastes a budget
    at or below a stale ceiling, the table build fails instead of letting the
    ceiling silently win — and a ceiling can never be certified."""
    with pytest.raises(ValueError, match="does not LOWER"):
        _build_with(sg.ModelBudget(                       # 1,048,576 > gemma4e4b's 589,824
            model_key="gemma4e4b", batched_token_budget=1_048_576,
            source=sg.PROVISIONAL_BUDGET_SOURCE,
            basis=sg.BudgetBasis("m", 16384, 64, 0.9, None, None, sg.UNMEASURED, "s")))
    # ... and one that really does lower it builds fine
    lower = _build_with(sg.ModelBudget(
        model_key="gemma4e4b", batched_token_budget=16384 * 30,
        source=sg.PROVISIONAL_BUDGET_SOURCE,
        basis=sg.BudgetBasis("m", 16384, 30, 0.9, None, None, sg.UNMEASURED, "s")))
    assert lower["gemma4e4b"].batched_token_budget == 491_520
    assert sg.BUDGET_BY_MODEL["gemma4e4b"].batched_token_budget == 589_824   # not mutated
    with pytest.raises(ValueError, match="cannot be certified"):
        sg.ModelBudget(model_key="llama8b", batched_token_budget=16384 * 4, certified=True,
                       source=sg.PROVISIONAL_BUDGET_SOURCE,
                       basis=sg.BudgetBasis("m", 16384, 4, 0.9, None, None, sg.UNMEASURED, "s"))
    with pytest.raises(ValueError, match="source must be one of"):
        sg.ModelBudget(model_key="llama8b", batched_token_budget=16384 * 4,
                       source=sg.DEFAULT_BUDGET_SOURCE,
                       basis=sg.BudgetBasis("m", 16384, 4, 0.9, None, None, sg.UNMEASURED, "s"))


def test_hybrid_families_cap_S_through_seqs_at():
    assert sg.HYBRID_MODEL_KEYS == ("qwen35_9b", "qwen35_4b", "nemotron3nano4b")
    assert sg.HYBRID_SEQS_CAP == 256
    # the canary that set the cap was measured against the SWEPT 720,896 (469 at 1536); the row is
    # the 2026-09-13 lens-stack measurement at 17,664 (741,888) now, and the cap still binds at 1536
    q = _swept("qwen35_4b")
    assert q.hybrid and q.seqs_cap == 256
    assert sg.identity_seqs(1536, budget=q.batched_token_budget) == 483      # the raw quotient ...
    assert q.seqs_at(1536) == 256                                          # ... capped (canary 373464)
    assert q.seqs_at(16384) == 45 and q.seqs_at(8192) == 90                # under the cap: the quotient
    # the row actually SERVED is that measured row (its halved shadow was lifted 2026-09-11 on
    # six lens-stack standups at 148 x 4864, then re-measured at 17,664): the cap binds at 1536
    qp = sg.BUDGET_BY_MODEL["qwen35_4b"]
    assert qp is q and qp.hybrid and not qp.provisional and qp.seqs_cap == 256
    assert qp.seqs_at(1536) == 256 and qp.seqs_at(4864) == 152 and qp.seqs_at(8192) == 90 and qp.seqs_at(16384) == 45
    assert qp.seqs_at(17664) == 42 == qp.basis.max_num_seqs and qp.basis.max_model_len == 17664
    assert "PRECAUTION LIFTED 2026-09-11" in qp.basis.note and "148 x 4864" in qp.basis.note
    n = sg.BUDGET_BY_MODEL["nemotron3nano4b"]
    assert n.hybrid and n.seqs_at(1536) == 256 and n.seqs_at(7424) == 105
    g = sg.BUDGET_BY_MODEL["glm46vflash"]                                  # dense: no cap
    assert not g.hybrid and g.seqs_cap is None and g.seqs_at(1536) == 512
    assert sg.BUDGET_BY_MODEL["llama3b"].seqs_at(1536) == 213              # provisional 327,680


def test_a_hybrid_window_ceiling_lowers_S_at_its_window_only():
    """gsm8k-paired200-paper-objects overlap leg (job 483254_2): nemotron's budget quotient at the
    `capability:gen_long` window 4608 is 170 (783,360 tokens), but the hybrid allocator reported 167.55x = 772,049
    tokens and readiness refused. A hybrid cache's token capacity is not S x L (6 Mamba state blocks + ceil(L / 976)
    attention blocks per request from a ~1,830-block pool: 4608 -> 11 blocks, 8192 -> 15, 10752 -> 18, 18944 -> 26),
    so the measured verdict is recorded as a per-WINDOW ceiling applied inside `seqs_at` — never a lower budget
    (which would move 73 -> 71 at 10752 under rows already served) and never a runtime clamp."""
    n = sg.BUDGET_BY_MODEL["nemotron3nano4b"]
    c = sg.HYBRID_WINDOW_SEQS_CEILINGS[("nemotron3nano4b", 4608)]
    assert isinstance(c, sg.WindowSeqsCeiling) and n.window_ceiling(4608) is c
    assert sg.identity_seqs(4608, budget=n.batched_token_budget) == 170          # the quotient that was refused
    assert n.seqs_at(4608) == c.max_num_seqs == 167 == int(c.kv_max_concurrency)
    assert 167 * 4608 == 769_536 <= c.kv_tokens_reported == 772_049 < 170 * 4608
    assert c.probe_job_id == "483254_2" and c.gpu_memory_utilization == 0.9
    # every other window is exactly the quotient (under the hybrid cap): the ON geometry 73 @ 10752 rows 27 / 28
    # were served at, the rung's 41 @ 18944, the budget probe's 96 @ 8192, and the 256 cap at 1536
    assert n.window_ceiling(10752) is None and n.window_ceiling(18944) is None
    assert n.seqs_at(10752) == 73 and n.seqs_at(18944) == 41 and n.seqs_at(8192) == 96 and n.seqs_at(1536) == 256
    # and the engine's reports these windows were held to: 73 <= 101.72x (measured), 41 <= ~70x (26 blocks / request)
    blocks = 1_831                                                                # 101.72 x 18 at 10752
    assert 73 <= blocks / 18 and 41 <= blocks / 26 and 170 > 1_843 / 11 >= 167
    # a dense model has no ceilings, whatever the window
    assert sg.BUDGET_BY_MODEL["glm46vflash"].window_ceiling(4608) is None
    assert sg.BUDGET_BY_MODEL["glm46vflash"].seqs_at(4608) == 786_432 // 4608 == 170
    # the resolved geometry the standup composes is the identity at 167 and names the ceiling
    r = sg.resolve_geometry("capability:gen_long", model_key="nemotron3nano4b")
    assert (r.max_model_len, r.max_num_seqs, r.seqs_source) == (4608, 167, "model_budget")
    assert r.is_identity and r.identity_S == 167 and r.batched_tokens == 769_536
    head = r.describe().splitlines()[0]
    assert "max_num_seqs=167 (model_budget); identity;" in head
    assert "hybrid window ceiling: S 167 not 170 at L=4608" in head and "167.55x" in head and "483254_2" in head
    # the ON window is untouched by the ceiling and says nothing about it
    r_on = sg.resolve_geometry("capability:gen_long_thinking", model_key="nemotron3nano4b")
    assert (r_on.max_model_len, r_on.max_num_seqs) == (10752, 73) and "window ceiling" not in r_on.describe()
    # describe_budgets names it
    md = sg.describe_budgets()
    assert "- nemotron3nano4b @ 4608: S 167 (the allocator's 167.55x = 772,049 KV tokens" in md
    assert "| nemotron3nano4b | 786,432 | measured | no | 256 | 48 | 96 | 256 |" in md   # the row itself is unchanged


def test_a_window_ceiling_must_be_measured_hybrid_and_below_the_quotient():
    """The row refuses inconsistent records at construction, and the import-time check refuses a ceiling that does
    not LOWER the budget's S at its window (a stale ceiling after a re-measured budget must be dropped, not kept)."""
    ok = {"model_key": "nemotron3nano4b", "max_model_len": 4608, "max_num_seqs": 167, "kv_max_concurrency": 167.55,
          "kv_tokens_reported": 772_049, "gpu_memory_utilization": 0.9, "probe_job_id": "j", "measured_on": "d", "note": "n",
          "batched_token_budget": 786_432}                                             # the profile the refused launch was composed from

    sg.WindowSeqsCeiling(**ok)
    with pytest.raises(ValueError, match="not a hybrid model key"):
        sg.WindowSeqsCeiling(**{**ok, "model_key": "glm46vflash"})
    with pytest.raises(ValueError, match="block size"):
        sg.WindowSeqsCeiling(**{**ok, "max_model_len": 4600})
    with pytest.raises(ValueError, match="floor"):
        sg.WindowSeqsCeiling(**{**ok, "max_num_seqs": 166})                   # not floor(167.55)
    with pytest.raises(ValueError, match="floor"):
        sg.WindowSeqsCeiling(**{**ok, "max_num_seqs": 168})
    with pytest.raises(ValueError, match="exceeds the KV"):
        sg.WindowSeqsCeiling(**{**ok, "kv_tokens_reported": 769_000})
    table = {**sg.BUDGET_BY_MODEL, **sg.ONBOARDED_BUDGET_BY_MODEL}                # BOTH budget tables (an onboarded key's row lives in the second; X7)
    sg._check_window_ceilings(table)                                           # the shipped table passes
    stale = sg.WindowSeqsCeiling(**{**ok, "max_num_seqs": 170, "kv_max_concurrency": 170.4,
                                    "kv_tokens_reported": 790_000})
    saved = dict(sg.HYBRID_WINDOW_SEQS_CEILINGS)
    try:
        sg.HYBRID_WINDOW_SEQS_CEILINGS[("nemotron3nano4b", 4608)] = stale
        with pytest.raises(ValueError, match="does not LOWER the budget's 170"):
            sg._check_window_ceilings(table)
        sg.HYBRID_WINDOW_SEQS_CEILINGS.clear()
        sg.HYBRID_WINDOW_SEQS_CEILINGS[("qwen35_4b", 1536)] = sg.WindowSeqsCeiling(
            **{**ok, "model_key": "qwen35_4b", "max_model_len": 1536, "max_num_seqs": 100, "kv_max_concurrency": 100.2,
               "kv_tokens_reported": 160_000, "batched_token_budget": sg.BUDGET_BY_MODEL["qwen35_4b"].batched_token_budget})   # that row's profile
        assert sg.BUDGET_BY_MODEL["qwen35_4b"].seqs_at(1536) == 100             # under the capped 256: applied
        sg._check_window_ceilings(table)
    finally:
        sg.HYBRID_WINDOW_SEQS_CEILINGS.clear()
        sg.HYBRID_WINDOW_SEQS_CEILINGS.update(saved)
    for name in ("HYBRID_WINDOW_SEQS_CEILINGS", "WindowSeqsCeiling"):
        assert name in sg.__all__


def test_budget_for_measured_default_none_and_unknown(caplog, monkeypatch):
    # every UNSHADOWED canonical key: measured, no warning
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        for key, (budget, *_rest) in MEASURED.items():
            if key in PROVISIONAL:
                continue
            r = sg.budget_for(key)
            assert r.measured and r.source == sg.MEASURED_BUDGET_SOURCE == "measured", key
            assert r.per_model and not r.provisional, key
            assert r.batched_token_budget == budget and r.entry.model_key == key, key
            assert r.as_dict()["source"] == "measured" and r.as_dict()["certified"] is False
            assert r.as_dict()["provisional"] is False, key
    assert not [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    # a SHADOWED key: its own (smaller) row, source provisional-conservative, and a
    # warning naming the ceiling and the failure it came from — never silent
    for key, (ceiling, *_rest) in PROVISIONAL.items():
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
            r = sg.budget_for(key)
        assert r.source == sg.PROVISIONAL_BUDGET_SOURCE and r.provisional, key
        assert r.per_model and not r.measured, key      # per_model, NOT measured, sizes it
        assert r.batched_token_budget == ceiling and r.entry.model_key == key, key
        assert r.as_dict()["provisional"] is True and r.as_dict()["source"] == \
            "provisional-conservative", key
        warn = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
        assert warn and "PROVISIONAL" in warn[0] and key in warn[0], key
        assert f"{ceiling:,}" in warn[0] and "NOT a measurement" in warn[0], key
        assert "--phase budget" in warn[0] and "_PROVISIONAL_BUDGETS" in warn[0], key
    # the lens-stack measurement supersedes the certified 393,216, the bare-engine 458,752
    # and the 229,376 ceiling
    assert sg.budget_for("llama8b").batched_token_budget == 475_136
    # a future key added to MODEL_KEYS before its sweep: the Llama default + a WARNING
    caplog.clear()
    monkeypatch.delitem(sg.BUDGET_BY_MODEL, "qwen35_4b")
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        r = sg.budget_for("qwen35_4b")
    assert r.source == sg.DEFAULT_BUDGET_SOURCE == "default_llama_certified"
    assert not r.measured and r.batched_token_budget == 393_216
    assert r.model_key == "qwen35_4b" and r.entry.model_key == "llama8b"
    warn = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warn and "qwen35_4b" in warn[0].getMessage()
    assert "NO measured" in warn[0].getMessage()
    assert "served_decode_sweep.py --phase budget" in warn[0].getMessage()
    assert r.as_dict()["source"] == "default_llama_certified"
    # None = no model named: the default, SILENTLY (every pre-existing caller)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        r = sg.budget_for(None)
    assert r.batched_token_budget == 393_216 and r.model_key is None
    assert not [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    # a typo is refused, never a silent default
    with pytest.raises(sg.UnknownModelKey):
        sg.budget_for("glm46v_flash")
    with pytest.raises(sg.UnknownModelKey):
        sg.budget_for("")


def test_identity_seqs_uses_the_model_budget(monkeypatch, caplog):
    # existing calls: Llama-sized, unchanged
    assert sg.identity_seqs(16384) == 24 and sg.identity_seqs(7424) == 52
    # the measured budgets (raw quotient — identity_seqs is arithmetic; the hybrid
    # cap lives in ModelBudget.seqs_at / resolve_geometry)
    assert sg.identity_seqs(16384, model_key="llama8b") == 29        # 475,136 // 16384 (lens-stack row)
    assert sg.identity_seqs(16384, model_key="gemma4e4b") == 36      # 589,824 // 16384
    assert sg.identity_seqs(1536, model_key="qwen35_4b") == 483       # 741,888 // 1536, uncapped
    # an unmeasured key -> the default S, with the warning
    monkeypatch.delitem(sg.BUDGET_BY_MODEL, "gemma4e4b")
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        assert sg.identity_seqs(16384, model_key="gemma4e4b") == 24
    assert any("gemma4e4b" in rec.getMessage() for rec in caplog.records)
    # a fake measured entry sizes S from ITS budget (2,097,152 = the qwen35_4b KV the
    # live 16384 x 24 engine reported, 128 full 16k sequences)
    monkeypatch.setitem(sg.BUDGET_BY_MODEL, "qwen35_4b", _fake_budget("qwen35_4b", 2_097_152))
    assert sg.identity_seqs(16384, model_key="qwen35_4b") == 128
    assert sg.identity_seqs(7424, model_key="qwen35_4b") == 282
    # an explicit budget still wins over everything
    assert sg.identity_seqs(16384, 393_216, model_key="qwen35_4b") == 24
    # the table entries themselves stay Llama-sized
    assert sg.TABLE["petri"].max_num_seqs == 24
    assert sg.TABLE["capability"].max_num_seqs == 52


def test_resolve_geometry_derives_S_from_the_model_budget(monkeypatch):
    monkeypatch.setitem(sg.BUDGET_BY_MODEL, "qwen35_4b", _fake_budget("qwen35_4b", 2_097_152))
    # no model: exactly the old resolution
    r0 = sg.resolve_geometry("capability")
    assert (r0.max_model_len, r0.max_num_seqs, r0.seqs_source) == (7424, 52, "table")
    assert r0.model_key is None and r0.budget is None
    assert r0.budget_tokens == 393_216 and r0.as_dict()["budget"] == 393_216
    assert r0.as_dict()["budget_source"] == "default_llama_certified"
    # the model's budget sizes S at the table window — through seqs_at, so the
    # hybrid per-slot cap applies (282 -> 256) and the pair still counts as identity
    r = sg.resolve_geometry("capability", model_key="qwen35_4b")
    assert (r.max_model_len, r.max_num_seqs) == (7424, 256)
    assert r.seqs_source == "model_budget" and r.window_source == "table"
    assert r.is_identity and r.identity_S == 256 and not r.over_budget and not r.overridden
    assert r.budget_tokens == 2_097_152 and r.budget_source == "measured"
    assert r.as_dict()["model_key"] == "qwen35_4b" and r.as_dict()["budget"] == 2_097_152
    assert "model qwen35_4b: measured; hybrid per-slot cap: S 256 not 282" in r.describe()
    # ... and at an overridden window (under the cap: the raw quotient)
    r = sg.resolve_geometry("capability", max_model_len=16384, model_key="qwen35_4b")
    assert r.max_num_seqs == 128 and r.is_identity and "per-slot cap" not in r.describe()
    # the certified Petri entry too: 16384 x 128 for a model whose KV holds it
    r = sg.resolve_geometry("petri", model_key="qwen35_4b")
    assert (r.max_model_len, r.max_num_seqs) == (16384, 128)
    # the budget GUARD is held to the model's budget: 24 x 16384 on an explicit
    # 128-way override is fine for qwen (under 2,097,152) ...
    r = sg.resolve_geometry("petri", max_num_seqs=100, model_key="qwen35_4b",
                            allow_over_budget=True)
    assert not r.over_budget
    # ... while 200 x 16384 = 3,276,800 exceeds it and refuses, naming the model
    with pytest.raises(sg.GeometryOverBudget) as ei:
        sg.resolve_geometry("petri", max_num_seqs=200, model_key="qwen35_4b")
    assert "measured qwen35_4b" in str(ei.value) and "2,097,152" in str(ei.value)
    # the REAL measured entries: the canary's window on qwen35_4b (469 -> 256) and a
    # dense model at the same window (uncapped, 512)
    monkeypatch.delitem(sg.BUDGET_BY_MODEL, "qwen35_4b")
    monkeypatch.setitem(sg.BUDGET_BY_MODEL, "qwen35_4b", sg._MEASURED_BUDGETS[5])
    r = sg.resolve_geometry("capability:short", model_key="qwen35_4b")
    assert (r.max_num_seqs, r.identity_S, r.is_identity) == (256, 256, True)
    assert r.budget_tokens == 741_888
    r = sg.resolve_geometry("capability:short", model_key="glm46vflash")
    assert r.max_num_seqs == 512 and r.is_identity
    r = sg.resolve_geometry("petri", model_key="llama8b")
    assert r.max_num_seqs == 29 and r.budget_tokens == 475_136        # the 2026-09-10 lens-stack row
    assert r.budget_source == sg.MEASURED_BUDGET_SOURCE and r.is_identity
    assert "model llama8b: measured]" in r.describe()
    # a model still on a ceiling resolves from it and says so
    r = sg.resolve_geometry("petri", model_key="qwen35_9b")
    assert r.max_num_seqs == 12 and r.budget_tokens == 196_608
    assert r.budget_source == sg.PROVISIONAL_BUDGET_SOURCE and r.is_identity
    assert "model qwen35_9b: provisional-conservative — PROVISIONAL" in r.describe()
    # an UNMEASURED key resolves to the Llama default S (with the warning)
    monkeypatch.delitem(sg.BUDGET_BY_MODEL, "gemma4e2b")
    r = sg.resolve_geometry("capability", model_key="gemma4e2b")
    assert r.max_num_seqs == 52 and r.seqs_source == "model_budget"
    assert r.budget_source == "default_llama_certified"
    assert "UNMEASURED" in r.describe()
    # the fallback path also takes the model budget (the REAL qwen35_4b entry is in
    # place here: 741,888 // 8192 = 90, under the cap)
    r = sg.resolve_geometry(None, max_model_len=8192, fallback=(10240, 64),
                            model_key="qwen35_4b", allow_over_budget=True)
    assert r.max_num_seqs == 90 and r.seqs_source == "model_budget"
    # an explicit S still wins
    r = sg.resolve_geometry("capability", max_num_seqs=40, model_key="qwen35_4b",
                            allow_over_budget=True)
    assert r.max_num_seqs == 40 and r.seqs_source == "override"
    # an unknown model key is refused
    with pytest.raises(sg.UnknownModelKey):
        sg.resolve_geometry("capability", model_key="nope")


def test_model_budget_refuses_inconsistent_records():
    with pytest.raises(ValueError):          # budget != probe product
        sg.ModelBudget("qwen35_4b", 2_000_000, sg.BudgetBasis(
            "m", 8192, 128, 0.9, None, None, "2026-09-04", "s"))
    with pytest.raises(ValueError):          # unknown key
        _fake_budget("qwen4b", 1_048_576)
    with pytest.raises(ValueError):          # window not block-aligned
        sg.ModelBudget("qwen35_4b", 8190 * 2, sg.BudgetBasis(
            "m", 8190, 2, 0.9, None, None, "2026-09-04", "s"))
    ok = _fake_budget("qwen35_4b", 1_048_576)
    assert ok.as_dict()["basis"]["max_num_seqs"] == 128


def test_describe_budgets_lists_every_key_and_the_paste_block_is_marked():
    md = sg.describe_budgets()
    rows = [l for l in md.splitlines() if l.startswith("| ") and not l.startswith("| model_key")]
    assert len(rows) == 8
    for key in sg.MODEL_KEYS:
        src = "provisional-conservative" if key in PROVISIONAL else "measured"
        assert re.search(rf"^\| {key} \| [\d,]+ \| {src} \| no \|", md, re.M), key
    assert "default_llama_certified" not in "\n".join(rows)
    # columns: model_key | budget | source | certified | hybrid_cap | S@16384 | S@8192 | S@1536 | ...
    assert "| llama8b | 475,136 | measured | no | - | 29 | 58 | 309 |" in md
    assert "| qwen35_9b | 196,608 | provisional-conservative | no | 256 | 12 | 24 | 128 |" in md
    assert "| qwen35_4b | 741,888 | measured | no | 256 | 45 | 90 | 256 |" in md
    # a ceiling declares itself unmeasured in the row, not only in the source column
    for key in PROVISIONAL:
        row = next(l for l in rows if l.startswith(f"| {key} |"))
        assert row.rstrip().endswith("| unmeasured |"), key
    assert "not a measurement" in md and "_PROVISIONAL_BUDGETS" in md
    assert "| glm46vflash | 786,432 | measured | no | - | 48 | 96 | 512 |" in md
    assert "| nemotron3nano4b | 786,432 | measured | no | 256 | 48 | 96 | 256 |" in md
    assert "hybrid_cap = the per-slot-state cap 256" in md
    assert "default = the Llama-8B-certified 393,216" in md
    src = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    assert "# MEASURED BY scripts/served_decode_sweep.py --phase budget  (BEGIN)" in src
    assert "# MEASURED BY scripts/served_decode_sweep.py --phase budget  (END)" in src
    assert "_MEASURED_BUDGETS: Tuple[ModelBudget, ...] = (" in src
    for name in ("MODEL_KEYS", "ModelBudget", "BudgetBasis", "budget_for", "describe_budgets",
                 "BUDGET_BY_MODEL", "UnknownModelKey", "HYBRID_MODEL_KEYS", "HYBRID_SEQS_CAP"):
        assert name in sg.__all__


def test_qwen35_4b_row_is_the_math500_window_measurement_and_agrees_with_the_entry_and_cap():
    """math500-thinking-on-full22 (2026-09-13): the Qwen3.5-4B serving
    budget is WIDENED back to a MEASURED value — the battery's own instrument
    (``scripts/served_decode_sweep.py --phase budget`` through the production standup)
    run AT the MATH-500 window on the lens stack — honouring the 16,384 cap; never a
    hand revert of the halved provisional 360,448 (S 20 at 17,664, the width the wave
    ran at) and never the bare-engine 8192 sweep row (720,896, S 40 at 17,664) standing
    in for a measurement at this window. Pins: the row's basis window IS the
    served-geometry entry's window; the entry's cap is unchanged; the window holds every
    MATH-500 prompt at the cap; the S every sizing site resolves at this window is the S
    the probe passed at; and the provenance (job, KV, wall, stack, record path) is on the
    row beside the number."""
    from sisteer.capeval.tinycap import contracts as C
    q = sg.BUDGET_BY_MODEL["qwen35_4b"]
    g = sg.geometry_for("capability:gen_long_math")
    # the row is MEASURED, at the entry's window, and the product is exact
    assert q.source == sg.MEASURED_BUDGET_SOURCE and not q.provisional and not q.certified and q.hybrid
    assert q.basis.max_model_len == g.max_model_len == 17_664
    assert q.basis.max_num_seqs == 42 and q.batched_token_budget == 42 * 17_664 == 741_888
    # the cap is unchanged, and the window holds prompt max + cap (the entry's own sizing branch)
    assert g.basis.requested_max_tokens == C.MATH500_MAX_TOKENS_DEFAULT == 16_384
    assert g.basis.max_prompt + C.MATH500_MAX_TOKENS_DEFAULT <= g.max_model_len == q.basis.max_model_len
    # the S every sizing site resolves at this window is the S the probe passed at; the hybrid cap does not bind
    assert q.seqs_at(17_664) == q.basis.max_num_seqs == 42 < sg.HYBRID_SEQS_CAP
    r = sg.resolve_geometry(g.key, strict_identity=True, model_key="qwen35_4b")
    assert (r.max_model_len, r.max_num_seqs, r.seqs_source) == (17_664, 42, "model_budget")
    assert r.is_identity and not r.over_budget and r.budget_tokens == 741_888
    # history, spelled: the halved provisional ceiling (S 20 at 17,664 — the MATH-500 wave's width; 33 at the 10,752
    # window it used to resolve to) and the swept row's 40 at this window; the measurement sits above both
    assert sg.identity_seqs(17_664, budget=360_448) == 20 and sg.identity_seqs(10_752, budget=360_448) == 33
    assert sg.identity_seqs(17_664, budget=720_896) == 40 < q.seqs_at(17_664)
    # provenance beside the number: the wall, the KV that held the product, the job, the day, the record, the stack
    b = q.basis
    assert b.first_refused == 44 * 17_664 == 777_216 and b.kv_tokens_reported == 850_259 >= q.batched_token_budget
    assert b.probe_job_id == "100196" and b.measured_on == "2026-09-13" and b.gpu_memory_utilization == 0.9
    assert "exp/geom-sweep-L17664/qwen35_4b/budget.json" in b.source
    assert "geom-budget-qwen35_4b-L17664_100196.out" in b.source
    for s in ("--budget-len 17664", "si_constitution__psr_kl_all", "vllm 0.25.1+cu129", "16,384 cap",
              "kv_coverage_problem", "geom_budget_window.sbatch"):
        assert s in b.method, s
    for s in ("48.14x at 17664", "probes: 5", "44 x 17664", "777,216 kv_under_coverage", "41.05x", "bisection 42",
              "850,259", "wall reached", "Supersedes the 2026-09-04 bare-engine sweep row 720,896", "360,448",
              "17,664 -> 42"):
        assert s in b.note, s

