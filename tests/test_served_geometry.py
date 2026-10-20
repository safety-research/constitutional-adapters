"""The per-eval served geometry table (``sisteer.capeval.serving.geometry``)
and its wiring into every served launcher — CPU-only, no vllm, no Modal.

Served vLLM geometry stops being one number inherited everywhere from
Petri's 16384 x 24; each eval gets its own default, MEASURED to the p99
of prompt+completion on its real records (docs/served_geometry_p99.md,
the 2026-09-03 measurement pass), and under the no-chunking identity a
smaller window at the SAME batched-token budget is a free bigger batch.
Covered:

(a) the table — every entry's L and S pinned to the measured literals with
    its basis fields; the SIZING RULE validator passes on every entry and
    refuses an entry whose clamp fires on most calls (the negative test);
    S == identity_seqs(L) everywhere (Petri's certified 24 IS the identity);
    the ``petri`` entry equals the composer defaults AND the smoke gate's
    PETRI_* pins; sub-entries resolve by ``eval:task`` and by ``task=``;
    the informational ``petri:<model>`` entries are refused by resolution;
(b) a SOURCE-GREP (tests/test_fast_sdpa.py style): no launcher under
    scripts/modal/, the capctrl standup, or the families drivers passes a
    literal 16384 window to the composer/standup — only the Petri render is
    allowlisted;
(c) resolution — table default < explicit override, the over-budget refusal
    and its opt-out, the legacy fallback; the grokmt / capctrl / standup /
    families / capability-driver / BoN-client surfaces all resolve from the
    table; the in-repo measurement doc exists and is path-free.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import vllm_server as vs

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _basis(**kw):
    base = dict(n_calls=100, p99_total=2000, max_total=2500, p99_prompt=1500,
                requested_max_tokens=512, clamp_rate_at_L=0.01, frac_over_L=0.01,
                families_measured=("m",), measured_on="2026-09-03", source="x")
    base.update(kw)
    return sg.GeometryBasis(**base)


# --------------------------------------------------------------------------- #
# (a) the table                                                                #
# --------------------------------------------------------------------------- #


def test_budget_is_the_certified_petri_footprint():
    # derived from the composer (DEFAULT_MAX_NUM_SEQS x its default window),
    # and it must be THE number: 24 x 16384 = 393,216
    assert sg.PETRI_MAX_MODEL_LEN == 16384
    assert sg.BATCHED_TOKEN_BUDGET == vs.DEFAULT_MAX_NUM_SEQS * 16384 == 393_216


def test_identity_seqs_and_next_block():
    assert sg.identity_seqs(16384) == 24
    assert sg.identity_seqs(8704) == 45
    assert sg.identity_seqs(7424) == 52
    assert sg.identity_seqs(11520) == 34
    assert sg.identity_seqs(3456) == 113
    assert sg.identity_seqs(768) == 512
    assert sg.identity_seqs(2 ** 40) == 1                 # never 0
    with pytest.raises(ValueError):
        sg.identity_seqs(0)
    assert sg.next_block(8487) == 8496 and sg.next_block(8496) == 8496
    assert sg.next_block(1) == 16 and sg.next_block(3391) == 3392


#: every entry's MEASURED literals (L, S, and the basis fields the rule
#: requires) — the 2026-09-03 measurement numbers, docs/served_geometry_p99.md
PINNED = {
    "petri": dict(L=16384, S=24, n=55781, p99_total=18385, max_total=23209,
                  p99_prompt=17714, max_tokens=1024, clamp=0.0051, over=None,
                  fams=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_4b", "qwen35_9b"),
                  sized_by="p99_total"),
    "multiturn_attacks": dict(L=8704, S=45, n=4598, p99_total=8487, max_total=12657,
                              p99_prompt=7496, max_tokens=2048, clamp=0.0137, over=0.0091,
                              fams=("gemma4e4b",), sized_by="p99_total"),
    "multiturn_attacks:pair": dict(L=3456, S=113, n=1217, p99_total=1864, max_total=3229,
                                   p99_prompt=1343, max_tokens=2048, clamp=0.0082, over=0.0,
                                   fams=("gemma4e4b",), sized_by="p99_prompt_plus_max_tokens"),
    "multiturn_attacks:crescendo": dict(L=9984, S=39, n=1343, p99_total=9788, max_total=12657,
                                        p99_prompt=9443, max_tokens=2048, clamp=0.0298,
                                        over=0.0074, fams=("gemma4e4b",), sized_by="p99_total"),
    "multiturn_attacks:actorattack": dict(L=6656, S=59, n=2038, p99_total=6579, max_total=9519,
                                          p99_prompt=5558, max_tokens=2048, clamp=0.0334,
                                          over=0.0098, fams=("gemma4e4b",), sized_by="p99_total"),
    "capability": dict(L=7424, S=52, n=74480, p99_total=7332, max_total=7701, p99_prompt=6946,
                       max_tokens=512, clamp=0.015, over=0.009,
                       fams=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                       sized_by="p99_total"),
    "capability:short": dict(L=1536, S=256, n=1500, p99_total=1076, max_total=1076, p99_prompt=575,
                             max_tokens=512, clamp=0.0, over=0.0, fams=("llama3b",),
                             sized_by="p99_total"),
    "families_served": dict(L=11520, S=34, n=129500, p99_total=8317, max_total=11466,
                            p99_prompt=7352, max_tokens=4096, clamp=0.0, over=0.0,
                            fams=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                            sized_by="p99_prompt_plus_max_tokens"),
    "families_served:short": dict(L=4864, S=80, n=90650, p99_total=2695, max_total=4792,
                                  p99_prompt=762, max_tokens=4096, clamp=0.0, over=0.0,
                                  fams=("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                                  sized_by="p99_prompt_plus_max_tokens"),
    "families_prompted": dict(L=7424, S=52, n=265216, p99_total=7284, max_total=7305,
                              p99_prompt=6777, max_tokens=512, clamp=0.0, over=0.0,
                              fams=("llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                              sized_by="p99_total"),
    "bon": dict(L=7424, S=52, n=994560, p99_total=7261, max_total=7305, p99_prompt=6777,
                max_tokens=512, clamp=0.0007, over=0.0004,
                fams=("llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                sized_by="p99_total"),
    "bon:short": dict(L=768, S=512, n=729344, p99_total=683, max_total=734, p99_prompt=118,
                      max_tokens=512, clamp=0.0, over=0.0,
                      fams=("llama8b", "llama3b", "gemma4e4b", "gemma4e2b", "qwen35_9b", "qwen35_4b"),
                      sized_by="p99_total"),
    "petri:llama8b": dict(L=11776, S=33, n=13814, p99_total=11575, max_total=15566, p99_prompt=11261,
                          max_tokens=1024, clamp=0.0142, over=0.0085, fams=("llama8b",),
                          sized_by="p99_total"),
    "petri:gemma4e4b": dict(L=15872, S=24, n=18832, p99_total=15758, max_total=20647,
                            p99_prompt=15024, max_tokens=1024, clamp=0.0129, over=0.0085,
                            fams=("gemma4e4b",), sized_by="p99_total"),
    "petri:gemma4e2b": dict(L=15616, S=25, n=7703, p99_total=15443, max_total=19896,
                            p99_prompt=14871, max_tokens=1024, clamp=0.0122, over=0.0086,
                            fams=("gemma4e2b",), sized_by="p99_total"),
    "petri:qwen35_4b": dict(L=17408, S=22, n=8242, p99_total=17395, max_total=21695,
                            p99_prompt=16562, max_tokens=1024, clamp=0.0110, over=0.0101,
                            fams=("qwen35_4b",), sized_by="p99_total"),
    "petri:qwen35_9b": dict(L=18432, S=21, n=7190, p99_total=18385, max_total=23209,
                            p99_prompt=17714, max_tokens=1024, clamp=0.0104, over=0.0097,
                            fams=("qwen35_9b",), sized_by="p99_total"),
    # tinycap: MEASURED 2026-09-04
    # from the six pass-1 usage ledgers by scripts/tinycap_geometry_measure.py
    # (proposal_v2 — Evals' verdict: cap-less calls OUTSIDE the clamp basis,
    # the grader route its own informational key; PW11). Each literal below was
    # copied DELIBERATELY from proposal_v2/geometry_pinned.py.txt, not
    # regenerated: a regenerated pin cannot catch a measurer regression.
    "capability:tiny": dict(L=1792, S=219, n=5242, p99_total=1665, max_total=3039,
                            p99_prompt=1664, max_tokens=1, clamp=0.0038, over=0.0038,
                            fams=("gemma4e4b",), sized_by="p99_total", measured_on="2026-09-04",
                            n_rejected=0, n_capless=0),
    # HELD as measured: sized from the ~90-token safety SI only, and
    # safety is the SHORT SI — an ANALYTIC provisional window instead:
    # p99_prompt(tiny) 1,664 + si_irrelevant 6,560 + template delta 5 = 8,229
    # (+ max_tokens 1 -> floor 8,240) -> 8,448 x 46
    "capability:tiny_prompted": dict(L=8448, S=46, n=0, p99_total=None, max_total=None,
                                     p99_prompt=8229, max_tokens=1, clamp=None, over=None,
                                     fams=("gemma4e4b",), sized_by="analytic_p99_prompt_plus_max_tokens",
                                     measured_on="unmeasured", provisional=True, analytic=True),
    # n 5,880 = the capped calls only; the 6,348 cap-less grader calls of the
    # v1 proposal (which read 2,816 x 139 on the total-p99 branch) are
    # capability:grader — over the capped calls the clamp at 2,816 is 8.67%
    "capability:gen_long": dict(L=4608, S=85, n=5880, p99_total=2789, max_total=2945,
                                p99_prompt=2349, max_tokens=2048, clamp=0.0, over=0.0,
                                fams=("gemma4e4b",), sized_by="p99_prompt_plus_max_tokens",
                                measured_on="2026-09-04", n_rejected=0, n_capless=0),
    "capability:gen_long_thinking": dict(L=10752, S=36, n=6422, p99_total=7342, max_total=8715,
                                         p99_prompt=2348, max_tokens=8192, clamp=0.0, over=0.0,
                                         fams=("gemma4e4b",), sized_by="p99_prompt_plus_max_tokens",
                                         measured_on="2026-09-04", n_rejected=0, n_capless=0),
    "capability:agent": dict(L=3328, S=118, n=3981, p99_total=2297, max_total=6050,
                             p99_prompt=2173, max_tokens=1024, clamp=0.0093, over=0.0008,
                             fams=("gemma4e4b",), sized_by="p99_prompt_plus_max_tokens",
                             measured_on="2026-09-04", n_rejected=0, n_capless=0),
    "capability:agent_thinking": dict(L=6400, S=61, n=3981, p99_total=2832, max_total=6703,
                                      p99_prompt=2175, max_tokens=4096, clamp=0.0093, over=0.0003,
                                      fams=("gemma4e4b",), sized_by="p99_prompt_plus_max_tokens",
                                      measured_on="2026-09-04", n_rejected=0, n_capless=0),
    # MATH-500's OWN entry: MEASURED from the MATH-500 prompt
    # budget (4,000 renders: 500 problems x 4 full22 models x {bare, prompted-safety}
    # under the thinking-ON render), NOT a pooled ledger; sized to hold EVERY prompt
    # at the 16,384 pass — 1,041 + 16,384 = 17,425 -> floor 17,440 -> 17,664 x 22 —
    # because assert_pass_fits_window refuses a clamped cap-bound sample. Each
    # literal below was copied from the eight budget artifacts, not regenerated.
    "capability:gen_long_math": dict(L=17664, S=22, n=4000, p99_total=None, max_total=None,
                                     p99_prompt=540, max_prompt=1041, max_tokens=16384, clamp=0.0,
                                     over=0.0, fams=("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"),
                                     sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-09",
                                     n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # MATH-500's LONG-instruction prompt classes: the si_constitution /
    # si_irrelevant system turns — 8,000 renders (500 problems x 4 models x 2 classes x
    # 2 arms; the longest, 7,622, is an OFF render) — 7,622 + 16,384 = 24,006 -> floor
    # 24,016 -> 24,064 x 16. Each literal copied from the sixteen measurements.
    "capability:gen_long_math_longsi": dict(L=24064, S=16, n=8000, p99_total=None, max_total=None,
                                            p99_prompt=7024, max_prompt=7622, max_tokens=16384, clamp=0.0,
                                            over=0.0, fams=("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"),
                                            sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-10",
                                            n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # the GSM8K PAIRED-200 ladder's 16,384 RUNG (gsm8k-paired200-paper-objects lane B, 2026-09-10): MEASURED from the
    # paired-200 prompt budget (1,600 renders: 200 items x 4 full22 models x {bare, prompted-safety} under the thinking-ON
    # render), sized to hold EVERY prompt at the rung — 2,462 + 16,384 = 18,846 -> floor 18,848 -> 18,944 x 20. Each
    # literal below was copied from the eight 16,384 artifacts. The experiment's 8,192 pass REUSES gen_long_thinking
    # (2,462 + 8,192 = 10,654 <= 10,752, margin 98); not a tinycap sub-entry (the eight keys are untouched).
    "capability:gen_long_gsm8k_rung": dict(L=18944, S=20, n=1600, p99_total=None, max_total=None,
                                           p99_prompt=2400, max_prompt=2462, max_tokens=16384, clamp=0.0,
                                           over=0.0, fams=("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"),
                                           sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-10",
                                           n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # the CONSTITUTION-prompted GSM8K cells (llama-paper-rows, the six-arm design of 2026-09-12): MEASURED from the paired-200
    # prompt budget under the si_constitution system turn on the Llama tokenizer (200 renders; prompt max 8,007, p99 7,975) and
    # sized to hold EVERY prompt at the cap — 8,007 + 8,192 = 16,199 -> floor 16,208 -> 16,384 x 24 at the 8,192 pass (the
    # certified standup geometry itself); 8,007 + 16,384 = 24,391 -> floor 24,400 -> 24,576 x 16 at the rung. Not tinycap sub-entries.
    "capability:gen_long_gsm8k_longsi": dict(L=16384, S=24, n=200, p99_total=None, max_total=None,
                                             p99_prompt=7975, max_prompt=8007, max_tokens=8192, clamp=0.0,
                                             over=0.0, fams=("llama8b",),
                                             sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-12",
                                             n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    "capability:gen_long_gsm8k_rung_longsi": dict(L=24576, S=16, n=200, p99_total=None, max_total=None,
                                                  p99_prompt=7975, max_prompt=8007, max_tokens=16384, clamp=0.0,
                                                  over=0.0, fams=("llama8b",),
                                                  sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-12",
                                                  n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # the CONSTITUTION-prompted GSM8K cells on the full22 panel's three SWITCH families (math-eight-arms-extension E2, plan P3:
    # the long-SI class is scoped by FAMILY — the Llama entries above hold the Llama render, these the full22 families'):
    # MEASURED 2026-09-13 from the paired-200 prompt budget under the si_constitution system turn on gemma4e4b / qwen35_4b /
    # nemotron3nano4b (600 renders; prompt max 8,338 / 8,332 / 8,288, pooled p99 8,305) and sized to hold EVERY prompt at the
    # cap — 8,338 + 8,192 = 16,530 -> floor 16,544 -> 16,640 x 23 at the 8,192 pass; 8,338 + 16,384 = 24,722 -> floor 24,736 ->
    # 24,832 x 15 at the rung. Not tinycap sub-entries.
    "capability:gen_long_gsm8k_longsi_f22": dict(L=16640, S=23, n=600, p99_total=None, max_total=None,
                                                 p99_prompt=8305, max_prompt=8338, max_tokens=8192, clamp=0.0,
                                                 over=0.0, fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b"),
                                                 sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-13",
                                                 n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    "capability:gen_long_gsm8k_rung_longsi_f22": dict(L=24832, S=15, n=600, p99_total=None, max_total=None,
                                                      p99_prompt=8305, max_prompt=8338, max_tokens=16384, clamp=0.0,
                                                      over=0.0, fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b"),
                                                      sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-13",
                                                      n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # the Qwen seed-replicate battery (one S x L per leg): the wave's COMMON GSM8K class — COPIES of the two
    # `_f22` long-SI entries' MEASURED values (the same 600 renders, prompt max 8,338) under a name every prompt class of the matrix resolves
    # to, short prompts a fortiori; selected by the matrix's `geometry_class`, never by prompt class; no new measurement
    "capability:gen_long_gsm8k_common_f22": dict(L=16640, S=23, n=600, p99_total=None, max_total=None,
                                                 p99_prompt=8305, max_prompt=8338, max_tokens=8192, clamp=0.0,
                                                 over=0.0, fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b"),
                                                 sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-13",
                                                 n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    "capability:gen_long_gsm8k_rung_common_f22": dict(L=24832, S=15, n=600, p99_total=None, max_total=None,
                                                      p99_prompt=8305, max_prompt=8338, max_tokens=16384, clamp=0.0,
                                                      over=0.0, fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b"),
                                                      sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-13",
                                                      n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # the IFEval battery's three cap windows: MEASURED 2026-09-11
    # from the IFEval PROMPT BUDGET (4,328 renders: 541 items x 4 battery models x {bare, prompted-safety} under the
    # thinking-OFF render; configs/sdfbase/ifeval/geometry_measurement.json at tree 9046696c), sized to hold EVERY
    # prompt at each cap of contracts.IFEVAL_CAP_LADDER — 468 + cap -> the 16-block floor IS the window (headroom 0):
    # 2,528 x 155 / 4,576 x 85 / 8,672 x 45. One thinking-OFF arm, so one key per cap and no _thinking twin; not
    # tinycap sub-entries (the eight keys are untouched).
    "capability:ifeval_2048": dict(L=2528, S=155, n=4328, p99_total=None, max_total=None,
                                   p99_prompt=201, max_prompt=468, max_tokens=2048, clamp=0.0, over=0.0,
                                   fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"),
                                   sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-11",
                                   n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    "capability:ifeval_4096": dict(L=4576, S=85, n=4328, p99_total=None, max_total=None,
                                   p99_prompt=201, max_prompt=468, max_tokens=4096, clamp=0.0, over=0.0,
                                   fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"),
                                   sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-11",
                                   n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    "capability:ifeval_8192": dict(L=8672, S=45, n=4328, p99_total=None, max_total=None,
                                   p99_prompt=201, max_prompt=468, max_tokens=8192, clamp=0.0, over=0.0,
                                   fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"),
                                   sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-11",
                                   n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # the constitution-prompted cells' three windows (the follow-up): MEASURED from the IFEval
    # prompt budget's CONSTITUTION class (2,164 renders: 541 items x 4 battery models with the prompted__si_constitution
    # system turn under the thinking-OFF render; geometry_measurement.json entries_longsi at tree 5b01824e), sized to
    # hold EVERY prompt at each cap — 6,385 + cap -> the 16-block ceiling (headroom 15): 8,448 x 46 / 10,496 x 37 /
    # 14,592 x 26. One key per cap, no _thinking twin; not tinycap sub-entries.
    "capability:ifeval_longsi_2048": dict(L=8448, S=46, n=2164, p99_total=None, max_total=None,
                                          p99_prompt=6132, max_prompt=6385, max_tokens=2048, clamp=0.0, over=0.0,
                                          fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"),
                                          sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-12",
                                          n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    "capability:ifeval_longsi_4096": dict(L=10496, S=37, n=2164, p99_total=None, max_total=None,
                                          p99_prompt=6132, max_prompt=6385, max_tokens=4096, clamp=0.0, over=0.0,
                                          fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"),
                                          sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-12",
                                          n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    "capability:ifeval_longsi_8192": dict(L=14592, S=26, n=2164, p99_total=None, max_total=None,
                                          p99_prompt=6132, max_prompt=6385, max_tokens=8192, clamp=0.0, over=0.0,
                                          fams=("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"),
                                          sized_by="max_prompt_plus_max_tokens", measured_on="2026-09-12",
                                          n_rejected=0, n_capless=0, sized_to_max_prompt=True),
    # the model-grader traffic on the base route: informational, all cap-less
    "capability:grader": dict(L=2304, S=170, n=6348, p99_total=2150, max_total=2436,
                              p99_prompt=1722, max_tokens=0, clamp=0.0, over=0.0019,
                              fams=("gemma4e4b",), sized_by="p99_total", measured_on="2026-09-04",
                              n_rejected=0, n_capless=6348, informational=True),
}

#: the dates entries were measured on — a SET, not one pin (P16): entries
#: measured on different days each carry their own (the tinycap entries:
#: 2026-09-04); "unmeasured" is the provisional sentinel
TINYCAP_MEASURED_ON = "2026-09-04"
#: MATH-500's own entry was measured (its prompt budget) on the day of the 16,384 ruling
MATH500_MEASURED_ON = "2026-09-09"
#: the long-instruction classes' entry, measured the next day
MATH500_LONGSI_MEASURED_ON = "2026-09-10"
#: the GSM8K constitution-prompted entries (llama-paper-rows, the six-arm design), measured on the Llama tokenizer
GSM8K_LONGSI_MEASURED_ON = "2026-09-12"
#: the constitution-prompted GSM8K cells on the full22 panel's three switch families (math-eight-arms-extension E2, P3)
GSM8K_LONGSI_F22_MEASURED_ON = "2026-09-13"
#: the IFEval battery's three cap windows, measured from the IFEval prompt budget (G-LENGTHS)
IFEVAL_MEASURED_ON = "2026-09-11"
#: the constitution-prompted cells' three windows (the follow-up), measured the next day
IFEVAL_LONGSI_MEASURED_ON = "2026-09-12"
MEASURED_ON_DATES = {sg.MEASURED_ON, TINYCAP_MEASURED_ON, MATH500_MEASURED_ON, MATH500_LONGSI_MEASURED_ON,
                     IFEVAL_MEASURED_ON, IFEVAL_LONGSI_MEASURED_ON, GSM8K_LONGSI_MEASURED_ON, GSM8K_LONGSI_F22_MEASURED_ON, sg.UNMEASURED}


def test_every_entry_is_pinned_to_the_measurement():
    assert set(sg.TABLE) == set(PINNED), "an entry was added or removed without a pin"
    for key, p in PINNED.items():
        g = sg.TABLE[key]
        assert g.key == key
        assert (g.max_model_len, g.max_num_seqs) == (p["L"], p["S"]), key
        b = g.basis
        assert b.n_calls == p["n"], key
        assert b.p99_total == p["p99_total"], key
        assert b.max_total == p["max_total"], key
        assert b.p99_prompt == p["p99_prompt"], key
        assert b.requested_max_tokens == p["max_tokens"], key
        assert b.clamp_rate_at_L == p["clamp"], key
        assert b.frac_over_L == p["over"], key
        assert b.families_measured == p["fams"], key
        # per-entry date: the original pass's entries all say 2026-09-03; a
        # provisional entry says so; a later-measured entry carries its own day
        assert b.measured_on == p.get("measured_on", sg.MEASURED_ON), key
        assert b.measured_on in MEASURED_ON_DATES, key
        assert b.n_rejected == p.get("n_rejected"), key
        assert b.n_capless == p.get("n_capless"), key
        assert b.max_prompt == p.get("max_prompt"), key
        assert b.sized_to_max_prompt == p.get("sized_to_max_prompt", False), key
        assert g.provisional == p.get("provisional", False), key
        assert g.informational == p.get("informational", False), key
        assert (b.analytic is not None) == p.get("analytic", False), key
        assert sg.MEASUREMENT_DOC in b.source and b.source, key
        assert g.sized_by == p["sized_by"], key
        # S = identity_seqs(L) ALWAYS; no override anywhere (Petri's 24 is the identity)
        assert g.seqs_override is None and g.is_identity, key
        assert not g.over_budget, key
        # the sizing rule holds (construction already asserted it; make it visible)
        assert sg.sizing_problem(g) is None, key
    assert sg.KNOWN_EVALS == ("petri", "multiturn_attacks", "capability", "families_served",
                              "families_prompted", "bon")
    assert set(sg.KNOWN_KEYS) == set(PINNED)
    assert sg.MEASURED_ON == "2026-09-03"      # the ORIGINAL pass; not a table-wide pin
    assert sg.provisional_keys() == tuple(k for k, p in PINNED.items() if p.get("provisional"))
    assert sg.provisional_keys() == ("capability:tiny_prompted",)
    assert set(sg.informational_keys()) == {k for k, p in PINNED.items() if p.get("informational")} | {
        k for k in PINNED if k.startswith("petri:")}
    # the eight served tinycap keys are the contracts' eight, exactly; the grader is not one
    from sisteer.capeval.tinycap import contracts as C
    assert sg.TINYCAP_KEYS == C.GEOMETRY_SUBENTRY_KEYS and len(sg.TINYCAP_KEYS) == 8
    assert C.GEOMETRY_GRADER_KEY == "capability:grader" in sg.TABLE
    assert C.GEOMETRY_GRADER_KEY not in sg.TINYCAP_KEYS


# --------------------------------------------------------------------------- #
# (a') the tinycap sub-entries: five MEASURED, tiny_prompted ANALYTIC           #
#      provisional, the grader INFORMATIONAL (Lane D -> PW11)                   #
# --------------------------------------------------------------------------- #

#: the five tinycap keys Evals accepted as measured (2026-09-04) and the window
#: each resolves to for pass 2 — at the certified budget and at gemma4e4b's
#: 589,824 (the served model; PW9 sizes S through resolve_geometry(model_key=))
MEASURED_TINYCAP = {
    "capability:tiny": (1792, 219, 329),
    "capability:gen_long": (4608, 85, 128),
    "capability:gen_long_thinking": (10752, 36, 54),
    "capability:agent": (3328, 118, 177),
    "capability:agent_thinking": (6400, 61, 92),
}


def test_measured_tinycap_subentries_resolve_for_pass_2_at_their_windows():
    """The five accepted rows: MEASURED (not provisional, not informational),
    every basis field from the ledger, S = identity_seqs(L) at the certified
    budget and the model's budget // L through ``resolve_geometry(model_key=)``
    — what the launcher's pass 2 serves."""
    from sisteer.capeval.tinycap import contracts as C
    for key, (L, S_cert, S_gemma) in MEASURED_TINYCAP.items():
        g = sg.geometry_for(key)
        assert not g.provisional and not g.informational and not g.certified
        assert not g.requires_recertification and g.eval == "capability"
        assert (g.max_model_len, g.max_num_seqs) == (L, S_cert) and g.is_identity
        assert g.basis.measured and g.basis.measured_on == TINYCAP_MEASURED_ON
        assert g.basis.n_rejected == 0 and g.basis.n_capless == 0 and g.basis.analytic is None
        assert g.basis.n_capped == g.basis.n_calls > 0 and g.basis.rejection_rate == 0.0
        assert "MEASURED from the pass-1 usage ledgers" in g.basis.note
        assert "section 6" in g.basis.source and "618f9dd" in g.basis.source
        assert sg.sizing_problem(g) is None
        r = sg.resolve_geometry(key)
        assert (r.max_model_len, r.max_num_seqs) == (L, S_cert) and r.window_source == "table"
        assert not r.provisional and not r.measurement_pass and not r.overridden
        rm = sg.resolve_geometry(key, model_key=C.MODEL_SLUG, strict_identity=True)
        assert (rm.max_model_len, rm.max_num_seqs) == (L, S_gemma)
        assert rm.seqs_source == "model_budget" and rm.is_identity and not rm.over_budget
        assert rm.max_num_seqs == sg.BUDGET_BY_MODEL[C.MODEL_SLUG].seqs_at(L) == 589_824 // L
        assert "MEASUREMENT PASS" not in rm.describe() and "PROVISIONAL" not in rm.describe()
    # gen_long: the branch Evals asked for — the SECOND, over the capped calls only
    g = sg.geometry_for("capability:gen_long")
    assert g.sized_by == sg.SIZED_BY_PROMPT_PLUS_MAX
    assert g.basis.clamp_rate_at_total_sizing == 0.0867 > sg.CLAMP_RATE_CEILING
    assert g.basis.floor == sg.next_block(2349 + 2048) == 4400 and 4400 <= 4608 <= 4400 + sg.SIZING_HEADROOM
    assert "EXCLUDED" in g.basis.note and "6,348" in g.basis.note and "capability:grader" in g.basis.note
    # agent / agent_thinking read the caps the ledger recorded (1,024 / 4,096), not the plan's
    assert sg.geometry_for("capability:agent").basis.requested_max_tokens == 1024
    assert sg.geometry_for("capability:agent_thinking").basis.requested_max_tokens == 4096
    # the top-level capability entry and its short sub-entry are untouched
    assert sg.resolve_geometry("capability").max_model_len == 7424
    assert sg.resolve_geometry("capability:short").max_num_seqs == 256


def test_tiny_prompted_is_the_one_analytic_provisional_entry():
    """Evals' hold: the pass-1 measurement of tiny_prompted was taken under
    the ~90-token safety SI — the SHORT SI — so the entry keeps ``provisional``
    with an ANALYTIC window: p99_prompt(tiny) 1,664 + si_irrelevant 6,560
    (the longest SI, gemma-4 tokens) + the system-turn template delta 5 =
    8,229; + max_tokens 1 -> block floor 8,240 -> L 8,448 (256-rounded), S 46
    / 69. Refused by resolve_geometry; served for measurement by
    pass1_geometry at the certified footprint, never at 8,448."""
    g = sg.geometry_for("capability:tiny_prompted")
    assert g.provisional and not g.informational and sg.provisional_keys() == (g.key,)
    assert (g.max_model_len, g.max_num_seqs) == (8448, 46) and g.is_identity
    assert sg.identity_seqs(8448) == 46 and 589_824 // 8448 == 69
    b = g.basis
    assert b.analytic and b.p99_prompt == 8229 and b.requested_max_tokens == 1
    assert b.p99_total is None and b.max_total is None and b.n_calls == 0
    assert b.clamp_rate_at_L is None and b.frac_over_L is None and b.n_rejected is None
    assert b.n_capless is None and b.n_capped is None and not b.measured
    assert b.measured_on == sg.UNMEASURED and g.sized_by == sg.SIZED_BY_ANALYTIC
    assert b.floor == sg.next_block(8229 + 1) == 8240 and 8240 <= 8448 <= 8240 + sg.SIZING_HEADROOM
    for s in ("1,664", "6,560", "+ 5", "8,229", "8,240", "8,448"):
        assert s in b.analytic, s
    for s in ("HELD", "SHORT", "2,210", "92", "9,603", "0.38%", "--extra-rows"):
        assert s in b.note, s
    assert "ANALYTIC" in b.source and "section 6" in b.source
    # refused for a real standup, with the analytic reason
    with pytest.raises(sg.GeometryProvisional, match="ANALYTIC"):
        sg.resolve_geometry(g.key)
    with pytest.raises(sg.GeometryProvisional):
        sg.resolve_geometry(g.key, model_key="gemma4e4b")
    # the measurement pass serves the footprint, labelled provisional, never 8,448
    r = sg.pass1_geometry(g.key)
    assert (r.max_model_len, r.max_num_seqs) == (sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS)
    assert r.provisional and r.measurement_pass and r.window_source == r.seqs_source == sg.PROVISIONAL_SOURCE
    assert r.entry is g and r.eval == g.key
    d = r.as_dict()
    assert d["provisional"] is True and d["measurement_pass"] is True
    assert d["sized_by"] == sg.SIZED_BY_ANALYTIC and d["basis"]["analytic"]
    assert "PROVISIONAL" in r.describe() and "ANALYTIC" in r.describe() and "8448" in r.describe()
    assert sg.pass1_geometry("capability", "tiny_prompted").entry is g


#: MATH-500's window and the S each full22 model's budget buys at it, vs at the
#: 10,752 MATH used to resolve to: x(10,752 / 17,664) = 0.609 concurrency
MATH500_WINDOW = 17664
MATH500_SEQS = {"gemma4e4b": (33, 54), "qwen35_4b": (42, 69), "glm46vflash": (44, 73), "nemotron3nano4b": (44, 73)}   # qwen35_4b: 741,888 = 42 x 17,664, MEASURED at this window 2026-09-13 (69 = 741,888 // 10,752)


def test_gen_long_math_is_math500s_own_measured_entry_sized_to_the_prompt_max():
    """MATH-500's one pass moved to 16,384 on 2026-09-09 and no registered
    window holds prompt + 16,384 (gen_long_thinking is 10,752), so MATH gets its
    OWN entry — MEASURED (resolve_geometry refuses a provisional one, and the
    footprint 16,384 cannot hold the cap either) from the MATH-500 prompt
    budget: 4,000 exact renders (500 problems x 4 models x bare / prompted),
    pooled p99 540, max 1,041. Sized by the MAX-PROMPT branch, not the p99:
    ``assert_pass_fits_window`` refuses a MATH task when any cap-bound sample
    has prompt + cap > L, so the clamp is a refusal and every prompt must fit —
    1,041 + 16,384 = 17,425 -> floor 17,440 -> L 17,664 (headroom 224, margin
    239; the p99 branch's floor 16,928 would have refused it at 736). Clamp 0
    by construction. Both arms resolve here, S from each model's budget."""
    from sisteer.capeval.tinycap import contracts as C
    g = sg.geometry_for("capability:gen_long_math")
    assert g.key == "capability:gen_long_math" in sg.TINYCAP_KEYS
    assert not g.provisional and not g.informational and not g.certified and not g.requires_recertification
    assert (g.max_model_len, g.max_num_seqs) == (MATH500_WINDOW, 22) and g.is_identity
    b = g.basis
    assert b.measured and b.measured_on == MATH500_MEASURED_ON and b.analytic is None
    assert b.sized_to_max_prompt and b.max_prompt == 1041 and b.p99_prompt == 540
    assert b.requested_max_tokens == C.MATH500_MAX_TOKENS_DEFAULT == 16384
    assert b.p99_total is None and b.max_total is None and b.clamp_rate_at_total_sizing is None
    assert b.n_calls == 4000 and b.n_rejected == 0 and b.n_capless == 0 and b.n_capped == 4000
    assert b.clamp_rate_at_L == 0.0 and b.frac_over_L == 0.0
    assert b.families_measured == ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
    # the branch and its arithmetic
    assert g.sized_by == b.sized_by == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX == "max_prompt_plus_max_tokens"
    assert b.floor == sg.next_block(1041 + 16384) == 17440
    assert 17440 <= MATH500_WINDOW <= 17440 + sg.SIZING_HEADROOM and MATH500_WINDOW % 256 == 0
    assert MATH500_WINDOW - (1041 + 16384) == 239                      # over the longest prompt
    assert sg.next_block(540 + 16384) == 16928 and MATH500_WINDOW - 16928 == 736 > sg.SIZING_HEADROOM
    assert sg.sizing_problem(g) is None
    for s in ("PROMPT BUDGET", "4,000", "1,041", "17,425", "17,440", "17,664", "540", "assert_pass_fits_window",
              "NOT measured", "16384", "no _thinking twin"):
        assert s in b.note, s
    assert "section 6" in b.source and "tinycap_oos_prompt_budget.py --task math500" in b.source
    # pass 2 resolves it — at the certified budget and at every full22 model's
    r = sg.resolve_geometry(g.key, strict_identity=True)
    assert (r.max_model_len, r.max_num_seqs, r.window_source) == (MATH500_WINDOW, 22, "table")
    for model, (seqs, seqs_at_10752) in MATH500_SEQS.items():
        rm = sg.resolve_geometry(g.key, strict_identity=True, model_key=model)
        assert (rm.max_model_len, rm.max_num_seqs, rm.seqs_source) == (MATH500_WINDOW, seqs, "model_budget"), model
        assert rm.max_num_seqs == sg.BUDGET_BY_MODEL[model].seqs_at(MATH500_WINDOW) and rm.is_identity
        assert not rm.over_budget
        assert sg.BUDGET_BY_MODEL[model].seqs_at(10752) == seqs_at_10752, model
    # pass 1 serves it at the footprint as a measurement re-run, like every measured tinycap key
    p1 = sg.pass1_geometry(g.key, model_key="gemma4e4b")
    assert (p1.max_model_len, p1.window_source, p1.eval) == (sg.PETRI_MAX_MODEL_LEN, sg.MEASUREMENT_SOURCE, g.key)
    # the entries MATH resolved to before are byte-identical to their pins (the golden holds them too)
    assert (sg.TABLE["capability:gen_long_thinking"].max_model_len, sg.TABLE["capability:gen_long"].max_model_len) == (10752, 4608)
    # the describe() table renders the branch and the max-prompt column
    md = sg.describe()
    assert "| max_prompt |" in md.splitlines()[0] and "max_prompt_plus_max_tokens" in md
    row = [l for l in md.splitlines() if l.startswith("| capability:gen_long_math |")][0]
    assert "| 17664 | 22 |" in row and "| 1041 |" in row and "| 16384 |" in row and "2026-09-09" in row


#: The long-instruction classes' window and the S the four models buy at it
#: (vs gen_long_math's 33 / 20 / 44 / 44 at 17,664: x(17,664 / 24,064) = 0.734)
MATH500_LONGSI_WINDOW = 24064
MATH500_LONGSI_SEQS = {"gemma4e4b": 24, "qwen35_4b": 30, "glm46vflash": 32, "nemotron3nano4b": 32}   # qwen35_4b: 741,888 // 24064 (the 2026-09-13 row)
#: per model x class: (ON max, OFF max) — copied from the sixteen measurements
MATH500_LONGSI_MAXIMA = {
    "gemma4e4b": {"si_constitution": (6981, 6979), "si_irrelevant": (7514, 7512)},
    "qwen35_4b": {"si_constitution": (6900, 6902), "si_irrelevant": (7620, 7622)},
    "glm46vflash": {"si_constitution": (6729, 6733), "si_irrelevant": (7265, 7269)},
    "nemotron3nano4b": {"si_constitution": (6887, 6887), "si_irrelevant": (7584, 7584)},
}


def test_gen_long_math_longsi_is_the_long_instruction_classes_own_measured_entry():
    """As of 2026-09-10, `capability:gen_long_math`
    was sized from the bare and safety-SI renders (max 1,041) but assigned to
    EVERY prompted MATH configuration, while the si_constitution / si_irrelevant
    system turns alone are ~5.9-6.7k tokens — so those cells get their OWN
    measured entry: 8,000 renders (500 problems x 4 models x 2 classes x BOTH
    arms; the longest, qwen x si_irrelevant x OFF, is 7,622), pooled p99 7,024,
    sized by the same MAX-PROMPT branch — 7,622 + 16,384 = 24,006 -> floor 24,016
    -> L 24,064 (headroom 48, margin 58; the p99 branch's floor 23,408 would
    have refused it at 656). Clamp 0 by construction; both arms resolve here; S
    from each model's budget. `gen_long_math` itself is untouched."""
    from sisteer.capeval.tinycap import contracts as C
    g = sg.geometry_for("capability:gen_long_math_longsi")
    assert g.key == "capability:gen_long_math_longsi" in sg.TINYCAP_KEYS
    assert not g.provisional and not g.informational and not g.certified and not g.requires_recertification
    assert (g.max_model_len, g.max_num_seqs) == (MATH500_LONGSI_WINDOW, 16) and g.is_identity
    b = g.basis
    assert b.measured and b.measured_on == MATH500_LONGSI_MEASURED_ON and b.analytic is None
    assert b.sized_to_max_prompt and b.max_prompt == 7622 and b.p99_prompt == 7024
    assert b.requested_max_tokens == C.MATH500_MAX_TOKENS_DEFAULT == 16384
    assert b.p99_total is None and b.max_total is None and b.clamp_rate_at_total_sizing is None
    assert b.n_calls == 8000 and b.n_rejected == 0 and b.n_capless == 0 and b.n_capped == 8000
    assert b.clamp_rate_at_L == 0.0 and b.frac_over_L == 0.0
    assert b.families_measured == ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
    # the measured maxima: the entry holds EVERY one of them at the cap, and its
    # max_prompt IS the largest (an OFF render)
    maxima = [n for cls in MATH500_LONGSI_MAXIMA.values() for pair in cls.values() for n in pair]
    assert max(maxima) == b.max_prompt == 7622 and all(n + 16384 <= MATH500_LONGSI_WINDOW for n in maxima)
    assert min(maxima) == 6729 > sg.TABLE["capability:gen_long_math"].basis.max_prompt == 1041
    # the branch and its arithmetic
    assert g.sized_by == b.sized_by == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX
    assert b.floor == sg.next_block(7622 + 16384) == 24016
    assert 24016 <= MATH500_LONGSI_WINDOW <= 24016 + sg.SIZING_HEADROOM and MATH500_LONGSI_WINDOW % 256 == 0
    assert MATH500_LONGSI_WINDOW - (7622 + 16384) == 58
    assert sg.next_block(7024 + 16384) == 23408 and MATH500_LONGSI_WINDOW - 23408 == 656 > sg.SIZING_HEADROOM
    assert sg.sizing_problem(g) is None
    for s_ in ("PROMPT BUDGET", "LONG-instruction", "8,000", "7,622", "7,024", "24,006", "24,016", "24,064",
               "si_constitution", "si_irrelevant", "assert_pass_fits_window", "NOT measured", "16384",
               "MATH500_PROMPT_CLASS_OF_SI", "no _thinking twin", "D80"):
        assert s_ in b.note, s_
    assert "section 6" in b.source and "--system-prompt-cell" in b.source and "8,000 renders" in b.source
    # pass 2 resolves it — at the certified budget and at every full22 model's
    r = sg.resolve_geometry(g.key, strict_identity=True)
    assert (r.max_model_len, r.max_num_seqs, r.window_source) == (MATH500_LONGSI_WINDOW, 16, "table")
    for model, seqs in MATH500_LONGSI_SEQS.items():
        rm = sg.resolve_geometry(g.key, strict_identity=True, model_key=model)
        assert (rm.max_model_len, rm.max_num_seqs, rm.seqs_source) == (MATH500_LONGSI_WINDOW, seqs, "model_budget"), model
        assert rm.max_num_seqs == sg.BUDGET_BY_MODEL[model].seqs_at(MATH500_LONGSI_WINDOW) and rm.is_identity
        assert not rm.over_budget
        assert sg.BUDGET_BY_MODEL[model].seqs_at(MATH500_WINDOW) == MATH500_SEQS[model][0], model
    # the sibling and both gen_long entries are untouched
    assert (sg.TABLE["capability:gen_long_math"].max_model_len, sg.TABLE["capability:gen_long_math"].basis.max_prompt) == (17664, 1041)
    assert (sg.TABLE["capability:gen_long_thinking"].max_model_len, sg.TABLE["capability:gen_long"].max_model_len) == (10752, 4608)
    # the contracts map both classes to the two entries, and only those
    assert C.MATH500_GEOMETRY_KEY_OF_CLASS == {"short": "capability:gen_long_math", "long_si": g.key}
    assert set(C.MATH500_GEOMETRY_KEY_OF_CLASS.values()) <= set(sg.TINYCAP_KEYS)
    md = sg.describe()
    row = [l for l in md.splitlines() if l.startswith("| capability:gen_long_math_longsi |")][0]
    assert "| 24064 | 16 |" in row and "| 7622 |" in row and "| 16384 |" in row and "2026-09-10" in row


#: the GSM8K paired-200 ladder's rung window and the S the four models buy at it
GSM8K_RUNG_WINDOW = 18944
GSM8K_RUNG_MEASURED_ON = "2026-09-10"


def test_gen_long_gsm8k_rung_is_the_paired200_ladders_own_measured_entry():
    """The stock 10-shot GSM8K prompt is ~2.1-2.5k
    tokens, so the experiment's 8,192 pass FITS the existing ``capability:gen_long_thinking`` (10,752; the entry the
    contracts' class map resolves for a thinking-ON GSM8K control) with margin 98 and REUSES it, while no registered
    window holds a 10-shot prompt + the 16,384 rung — the rung gets its OWN entry, MEASURED from the paired-200 prompt
    budget (1,600 renders) and sized to the prompt MAX like gen_long_math (a clamped redraw is not a 16,384 draw):
    2,462 + 16,384 = 18,846 -> floor 18,848 -> 18,944 (headroom 96). Served through resolve_geometry with S from each
    model's budget; not a tinycap sub-entry (the eight keys stay the contracts' eight)."""
    from sisteer.capeval.tinycap import contracts as C
    g = sg.geometry_for("capability:gen_long_gsm8k_rung")
    assert g.key not in sg.TINYCAP_KEYS and g.key not in sg.informational_keys()
    assert not g.provisional and not g.informational and not g.certified and not g.requires_recertification
    assert (g.max_model_len, g.max_num_seqs) == (GSM8K_RUNG_WINDOW, 20) and g.is_identity and g.eval == "capability"
    b = g.basis
    assert b.measured and b.measured_on == GSM8K_RUNG_MEASURED_ON and b.analytic is None
    assert b.sized_to_max_prompt and b.max_prompt == 2462 and b.p99_prompt == 2400 and b.requested_max_tokens == 16384
    assert b.p99_total is None and b.max_total is None and b.clamp_rate_at_total_sizing is None
    assert b.n_calls == 1600 and b.n_rejected == 0 and b.n_capless == 0 and b.n_capped == 1600
    assert b.clamp_rate_at_L == 0.0 and b.frac_over_L == 0.0 and g.sized_by == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX
    assert b.floor == sg.next_block(2462 + 16384) == 18848 and GSM8K_RUNG_WINDOW - b.floor == 96 <= sg.SIZING_HEADROOM
    assert sg.sizing_problem(g) is None and sg.MEASUREMENT_DOC in b.source and "gsm8k_paired200" in b.source
    # the 8,192 pass fits the existing thinking entry with margin (the plan's reuse rule): no new 8,192 entry
    assert sg.TABLE["capability:gen_long_thinking"].max_model_len == 10752 and 10752 - (2462 + 8192) == 98
    assert C.geometry_subentry(C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_GSM8K], prompted=False, thinking=True) == "capability:gen_long_thinking"
    # the thinking-OFF overlap leg (OFF max 2,375 + the table's 512) fits the OFF class's entry
    assert sg.TABLE["capability:gen_long"].max_model_len == 4608 and 2375 + 512 <= 4608
    assert C.geometry_subentry(C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_GSM8K], prompted=False, thinking=False) == "capability:gen_long"
    # served through resolve_geometry like capability:short, S from each model's budget; pass1 refuses (not a tinycap key)
    expected_s = {"gemma4e4b": 31, "qwen35_4b": 39, "glm46vflash": 41, "nemotron3nano4b": 41}   # qwen35_4b: 741,888 // 18944
    for key, s_ in expected_s.items():
        r = sg.resolve_geometry(g.key, model_key=key)
        assert (r.max_model_len, r.max_num_seqs) == (GSM8K_RUNG_WINDOW, s_), key
        assert sg.BUDGET_BY_MODEL[key].seqs_at(GSM8K_RUNG_WINDOW) == s_ and 2462 + 16384 <= r.max_model_len
    with pytest.raises(ValueError, match="not a tinycap sub-entry"):
        sg.pass1_geometry(g.key)
    assert len(sg.TINYCAP_KEYS) == 8 and sg.TINYCAP_KEYS == C.GEOMETRY_SUBENTRY_KEYS
    md = sg.describe()
    row = [l for l in md.splitlines() if l.startswith("| capability:gen_long_gsm8k_rung |")][0]
    assert "| 18944 | 20 |" in row and "| 2462 |" in row and "| 16384 |" in row and GSM8K_RUNG_MEASURED_ON in row


#: the IFEval ladder's three windows: cap -> (L, S at the certified budget); every L is 468 + cap rounded up to
#: the 16-block, so the floor IS the window
IFEVAL_WINDOWS = {2048: (2528, 155), 4096: (4576, 85), 8192: (8672, 45)}


def test_ifeval_entries_are_the_battery_ladders_own_measured_entries():
    """IFEval's prompts are short (541 items; pooled p99 201, max
    468 under the four battery tokenizers, the safety system turn a constant +88..93), and the battery runs ONE
    thinking-OFF arm at the caps of ``contracts.IFEVAL_CAP_LADDER`` (2,048 pass, 4,096 / 8,192 re-draw rungs), so
    each cap gets its OWN entry, MEASURED from the IFEval prompt budget (4,328 renders) and sized to the prompt MAX
    like gen_long_math — the driver refuses a cap-bound draw the window clamped, so every prompt must fit at the
    cap: 468 + cap -> the 16-block floor, which IS the window (headroom 0; the rule accepts a window equal to its
    floor). Served through resolve_geometry with S from each model's budget (the STARTING S; Modal's kv_shrink may
    lower it at readiness); not tinycap sub-entries (the eight keys stay the contracts' eight);
    ``contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP`` maps every cap to a registered key."""
    from sisteer.capeval.tinycap import contracts as C
    assert tuple(C.IFEVAL_GEOMETRY_ENTRY_BY_CAP) == C.IFEVAL_CAP_LADDER == tuple(IFEVAL_WINDOWS)
    for key in C.IFEVAL_GEOMETRY_ENTRY_BY_CAP.values():
        assert key in sg.TABLE and key in sg.KNOWN_KEYS, key                 # every value is a registered key
    expected_s = {"gemma4e4b": {2048: 233, 4096: 128, 8192: 68},             # 589,824 (measured)
                  "qwen35_4b": {2048: 256, 4096: 162, 8192: 85},             # 741,888 (measured 2026-09-13 at 17,664): 293 -> the 256 cap at 2528
                  "nemotron3nano4b": {2048: 256, 4096: 171, 8192: 90},       # 786,432; 311 -> the 256 cap at 2528
                  "llama8b": {2048: 187, 4096: 103, 8192: 54}}               # 475,136 (the serving gate)
    for cap, key in C.IFEVAL_GEOMETRY_ENTRY_BY_CAP.items():
        L, S = IFEVAL_WINDOWS[cap]
        assert key == f"capability:ifeval_{cap}"
        g = sg.geometry_for(key)
        assert g is sg.geometry_for("capability", f"ifeval_{cap}") and g.task == f"ifeval_{cap}"
        assert g.key not in sg.TINYCAP_KEYS and g.key not in sg.informational_keys()
        assert not g.provisional and not g.informational and not g.certified and not g.requires_recertification
        assert (g.max_model_len, g.max_num_seqs) == (L, S) and g.is_identity and g.eval == "capability"
        assert g.max_num_seqs == sg.identity_seqs(L) and not g.over_budget
        b = g.basis
        assert b.measured and b.measured_on == IFEVAL_MEASURED_ON and b.analytic is None
        assert b.sized_to_max_prompt and b.max_prompt == 468 and b.p99_prompt == 201 and b.requested_max_tokens == cap
        assert b.p99_total is None and b.max_total is None and b.clamp_rate_at_total_sizing is None
        assert b.n_calls == 4328 and b.n_rejected == 0 and b.n_capless == 0 and b.n_capped == 4328
        assert b.families_measured == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
        assert b.clamp_rate_at_L == 0.0 and b.frac_over_L == 0.0 and g.sized_by == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX
        # the floor IS the window: the rule accepts headroom 0, and the block floor is 12 tokens over the longest
        # prompt at the cap; the p99 branch (201 + cap) lands 272 under and would have accepted the same L, so the
        # max branch is the rule (no clamped draw), not a rescue
        assert b.floor == sg.next_block(468 + cap) == L and L - b.floor == 0 and L - (468 + cap) == 12
        assert L - sg.next_block(201 + cap) == 272 <= sg.SIZING_HEADROOM
        assert sg.sizing_problem(g) is None and sg.MEASUREMENT_DOC in b.source
        for s_ in ("geometry_measurement.json", "9046696c", "measure_ifeval_geometry.py", "966cd895", "4,328",
                   "thinking-OFF"):
            assert s_ in b.source, (key, s_)
        for s_ in ("MEASURED from the IFEval PROMPT BUDGET", "assert_pass_fits_window", "EVERY prompt at the cap",
                   f"468 + {cap:,}", "thinking-OFF arm", "no _thinking twin", "kv_shrink", "hybrid per-slot cap",
                   "475,136", "NOT a tinycap sub-entry", f"IFEVAL_GEOMETRY_ENTRY_BY_CAP[{cap}]",
                   "353 / 445", "375 / 468", "375 / 463", "370 / 463", "92 / 93 / 88 / 93", "p99 201"):
            assert s_ in b.note, (key, s_)
        # served through resolve_geometry like capability:short, S from each model's budget through seqs_at (the
        # hybrid per-slot cap, then any window ceiling); the pair stays the identity at the model's budget
        for mk, by_cap in expected_s.items():
            r = sg.resolve_geometry(key, model_key=mk)
            assert (r.max_model_len, r.max_num_seqs, r.seqs_source) == (L, by_cap[cap], "model_budget"), (key, mk)
            assert sg.BUDGET_BY_MODEL[mk].seqs_at(L) == by_cap[cap] and 468 + cap <= r.max_model_len
            assert r.is_identity and not r.over_budget and r.window_source == "table"
            assert f"{by_cap[cap]} / " in b.note or f"/ {by_cap[cap]} " in b.note or f"/ {by_cap[cap]} at" in b.note
        # pass1 refuses (not a tinycap key), with or without a model
        with pytest.raises(ValueError, match="not a tinycap sub-entry"):
            sg.pass1_geometry(key)
        with pytest.raises(ValueError, match="not a tinycap sub-entry"):
            sg.pass1_geometry(key, model_key="gemma4e4b")
        row = [l for l in sg.describe().splitlines() if l.startswith(f"| {key} |")][0]
        assert f"| {L} | {S} |" in row and "| 468 |" in row and f"| {cap} |" in row and IFEVAL_MEASURED_ON in row
        assert "max_prompt_plus_max_tokens" in row
    # nemotron at 2,528: the quotient 311 is held to the hybrid per-slot cap (HYBRID_SEQS_CAP 256) and the
    # resolution says so; the two longer windows sit under the cap; no window ceiling is recorded at any of them
    n = sg.BUDGET_BY_MODEL["nemotron3nano4b"]
    assert sg.identity_seqs(2528, budget=n.batched_token_budget) == 311 > sg.HYBRID_SEQS_CAP == n.seqs_at(2528) == 256
    assert n.seqs_at(4576) == 171 < 256 and n.seqs_at(8672) == 90
    assert all(n.window_ceiling(L) is None for L, _ in IFEVAL_WINDOWS.values())
    rn = sg.resolve_geometry("capability:ifeval_2048", model_key="nemotron3nano4b")
    assert "hybrid per-slot cap: S 256 not 311" in rn.describe() and rn.identity_S == 256
    assert "per-slot cap" not in sg.resolve_geometry("capability:ifeval_4096", model_key="nemotron3nano4b").describe()
    # llama8b sizes from the serving gate's MEASURED 475,136, not the retired 229,376 ceiling
    r = sg.resolve_geometry("capability:ifeval_2048", model_key="llama8b")
    assert r.budget_tokens == 475_136 and r.budget_source == sg.MEASURED_BUDGET_SOURCE and r.max_num_seqs == 187
    assert "model llama8b: measured" in r.describe() and "PROVISIONAL" not in r.describe()
    # one thinking-OFF arm: no _thinking twin, exactly the ladder's three keys, and the eight tinycap keys untouched
    assert not any(k.startswith("capability:ifeval_") and k.endswith("_thinking") for k in sg.TABLE)
    assert {k for k in sg.TABLE if k.startswith("capability:ifeval_")} == set(C.IFEVAL_GEOMETRY_ENTRIES) == (
        set(C.IFEVAL_GEOMETRY_ENTRY_BY_CAP.values()) | set(C.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI.values()))
    assert C.THINKING_BY_TASK[C.IFEVAL_TASK] == C.THINKING_OFF
    assert len(sg.TINYCAP_KEYS) == 8 and sg.TINYCAP_KEYS == C.GEOMETRY_SUBENTRY_KEYS
    # the three windows label nothing else in the decode sweep (no other entry shares them)
    sw = _load("_sweep_window_labels_ifeval", "scripts/served_decode_sweep.py")
    for cap, (L, _) in IFEVAL_WINDOWS.items():
        assert sw.window_labels(L) == (f"capability:ifeval_{cap}",)


IFEVAL_LONGSI_WINDOWS = {2048: (8448, 46), 4096: (10496, 37), 8192: (14592, 26)}


def test_ifeval_longsi_entries_are_the_constitution_prompted_cells_own_measured_entries():
    """The follow-up: the SHORT constitution SI (``prompted__si_constitution``, 30,952 chars) adds a CONSTANT
    ~5.9-6.0k tokens per item, so a constitution-prompted IFEval cell renders ~6.4k-token prompts — ~14x the short
    classes' 468 max — and cannot share their window (468 + 2,048 = 2,528 would clamp every draw). One entry per cap,
    MEASURED from the prompt budget's constitution class (2,164 renders) and sized to the class max like the base
    entries: 6,385 + cap -> the 16-block ceiling; S about 0.3x the short cells' at the same cap; selected by PROMPT
    CLASS through ``contracts.ifeval_geometry_entry`` (never by cap alone); never a tinycap sub-entry."""
    from sisteer.capeval.tinycap import contracts as C
    assert tuple(C.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI) == C.IFEVAL_CAP_LADDER == tuple(IFEVAL_LONGSI_WINDOWS)
    expected_s = {"gemma4e4b": {2048: 69, 4096: 56, 8192: 40}, "qwen35_4b": {2048: 87, 4096: 70, 8192: 50},
                  "nemotron3nano4b": {2048: 93, 4096: 74, 8192: 53}, "llama8b": {2048: 56, 4096: 45, 8192: 32}}
    for cap, key in C.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI.items():
        L, S = IFEVAL_LONGSI_WINDOWS[cap]
        assert key == f"capability:ifeval_longsi_{cap}" and key in sg.TABLE and key in sg.KNOWN_KEYS
        g = sg.geometry_for(key)
        assert g.task == f"ifeval_longsi_{cap}" and g.key not in sg.TINYCAP_KEYS and g.key not in sg.informational_keys()
        assert not g.provisional and not g.informational and not g.certified
        assert (g.max_model_len, g.max_num_seqs) == (L, S) and g.is_identity and g.max_num_seqs == sg.identity_seqs(L)
        b = g.basis
        assert b.measured and b.measured_on == IFEVAL_LONGSI_MEASURED_ON and b.sized_to_max_prompt
        assert b.max_prompt == 6385 and b.p99_prompt == 6132 and b.requested_max_tokens == cap and b.n_calls == 2164
        assert b.families_measured == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
        assert b.floor == sg.next_block(6385 + cap) == L and 0 <= L - (6385 + cap) < sg.VLLM_BLOCK_SIZE
        # the p99 branch would NOT hold the longest prompt at the cap — the max branch is the rule here, not a rescue
        assert sg.next_block(6132 + cap) < 6385 + cap
        assert sg.sizing_problem(g) is None and sg.MEASUREMENT_DOC in b.source
        for s_ in ("geometry_measurement.json", "entries_longsi", "5b01824e", "measure_ifeval_geometry.py", "966cd895",
                   "2,164", "prompted__si_constitution", "thinking-OFF"):
            assert s_ in b.source, (key, s_)
        for s_ in ("CONSTITUTION class", "6,385", "6,132", "6,032 / 5,963 / 5,952 / 5,877", "~14x", "ifeval_geometry_entry",
                   f"IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI[{cap}]", "kv_shrink", "NOT a tinycap sub-entry", "0.3x"):
            assert s_ in b.note, (key, s_)
        for mk, by_cap in expected_s.items():
            r = sg.resolve_geometry(key, model_key=mk)
            assert (r.max_model_len, r.max_num_seqs, r.seqs_source) == (L, by_cap[cap], "model_budget"), (key, mk)
            assert sg.BUDGET_BY_MODEL[mk].seqs_at(L) == by_cap[cap] and 6385 + cap <= r.max_model_len
            # about a third of the short cells' S at the same cap (the window is ~3.3x wider)
            short = sg.resolve_geometry(C.IFEVAL_GEOMETRY_ENTRY_BY_CAP[cap], model_key=mk).max_num_seqs
            assert 0.25 <= by_cap[cap] / short <= 0.62, (key, mk, by_cap[cap], short)
        with pytest.raises(ValueError, match="not a tinycap sub-entry"):
            sg.pass1_geometry(key, model_key="gemma4e4b")
        # the routing: the constitution cell alone selects this entry; every other cell the short one
        assert C.ifeval_geometry_entry(C.CELL_PROMPTED_CONSTITUTION, cap) == key
        for cell in ("none", "prompted__safety_refusal", "constv2-mix-full__lora_r4_ce", "alphasteer"):
            assert C.ifeval_geometry_entry(cell, cap) == C.IFEVAL_GEOMETRY_ENTRY_BY_CAP[cap]
        with pytest.raises(C.IfevalPromptClassUnmeasured, match="no measured IFEval prompt class"):
            C.ifeval_geometry_entry("prompted__si_irrelevant", cap)
    # the windows are labelled by the sweep (tiny_prompted's provisional 8,448 is filtered, so 8,448 is this entry's)
    sw = _load("_sweep_window_labels_ifeval_longsi", "scripts/served_decode_sweep.py")
    for cap, (L, _) in IFEVAL_LONGSI_WINDOWS.items():
        assert sw.window_labels(L) == (f"capability:ifeval_longsi_{cap}",)
    assert C.IFEVAL_GEOMETRY_ENTRIES == (*C.IFEVAL_GEOMETRY_ENTRY_BY_CAP.values(), *C.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI.values())
    assert C.IFEVAL_GEOMETRY_ENTRY_BY_CLASS == {"short": C.IFEVAL_GEOMETRY_ENTRY_BY_CAP, "long_si": C.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI}


def test_max_prompt_sizing_branch_rules():
    """The MAX-PROMPT branch's own rules: floor = next block >= max_prompt +
    max_tokens within the headroom; the clamp at L must be EXACTLY 0 (a
    clamped call contradicts "every prompt fits"); max_prompt cannot be below
    p99_prompt; the flag needs the number; an analytic basis cannot take it;
    a provisional entry may carry neither (measured quantities). And the
    motivating fact: the p99 branch refuses the window MATH needs."""
    kw = dict(p99_total=None, p99_prompt=540, max_prompt=1041, requested_max_tokens=16384,
              clamp_rate_at_L=0.0, frac_over_L=0.0, sized_to_max_prompt=True)
    ok = sg.ServedGeometry("e", 17664, sg.identity_seqs(17664), _basis(**kw))
    assert ok.sized_by == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX and ok.basis.floor == 17440
    assert ok.basis.measured and sg.sizing_problem(ok) is None
    # the p99 branch on the SAME measurement refuses 17,664 (736 over its 16,928 floor)
    with pytest.raises(ValueError, match="736 tokens above the sizing floor 16928"):
        sg.ServedGeometry("e", 17664, sg.identity_seqs(17664),
                          _basis(**{**kw, "sized_to_max_prompt": False}))
    # below the max floor, or more than the headroom above it, refused
    with pytest.raises(ValueError, match="below the sizing floor 17440"):
        sg.ServedGeometry("e", 17408, sg.identity_seqs(17408), _basis(**kw))
    with pytest.raises(ValueError, match="headroom"):
        sg.ServedGeometry("e", 17440 + 1024, sg.identity_seqs(17440 + 1024), _basis(**kw))
    # a non-zero clamp on a max-sized window: max_prompt is not the longest prompt
    with pytest.raises(ValueError, match="clamp rate 0.0010 .* max_prompt 1041"):
        sg.ServedGeometry("e", 17664, sg.identity_seqs(17664), _basis(**{**kw, "clamp_rate_at_L": 0.001}))
    with pytest.raises(ValueError, match="only an informational entry may omit it"):
        sg.ServedGeometry("e", 17664, sg.identity_seqs(17664), _basis(**{**kw, "clamp_rate_at_L": None}))
    # basis-level refusals
    with pytest.raises(ValueError, match="below p99_prompt"):
        _basis(**{**kw, "max_prompt": 500})
    with pytest.raises(ValueError, match="needs max_prompt"):
        _basis(**{**kw, "max_prompt": None})
    with pytest.raises(ValueError, match="positive int"):
        _basis(**{**kw, "max_prompt": 0})
    with pytest.raises(ValueError, match="MEASURED branch"):
        _basis(**{**kw, "clamp_rate_at_L": None, "frac_over_L": None, "analytic": "derived"})
    # max_prompt alone (no flag) is just another measured number: the p99 branches size as before
    plain = _basis(p99_total=2000, p99_prompt=1500, max_prompt=1900, requested_max_tokens=256, clamp_rate_at_L=0.0)
    assert plain.sized_by == sg.SIZED_BY_TOTAL and plain.floor == sg.next_block(2000) and plain.measured
    # a provisional entry may not carry the max-prompt fields
    for extra in ({"max_prompt": 1041}, {"max_prompt": 1041, "sized_to_max_prompt": True}):
        prov = sg.GeometryBasis(n_calls=0, p99_total=None, max_total=None, p99_prompt=None,
                                requested_max_tokens=16384, clamp_rate_at_L=None, frac_over_L=None,
                                families_measured=("gemma4e4b",), measured_on=sg.UNMEASURED, source="s", **extra)
        with pytest.raises(ValueError, match="provisional entry carries measured basis fields"):
            sg.ServedGeometry("e", sg.PETRI_MAX_MODEL_LEN, 24, prov, provisional=True)
    # every pre-existing entry has the fields OFF: nothing about them moved
    for key, g in sg.TABLE.items():
        if key not in ("capability:gen_long_math", "capability:gen_long_math_longsi", "capability:gen_long_gsm8k_rung",
                       "capability:gen_long_gsm8k_longsi", "capability:gen_long_gsm8k_rung_longsi",
                       "capability:gen_long_gsm8k_longsi_f22", "capability:gen_long_gsm8k_rung_longsi_f22",
                       "capability:gen_long_gsm8k_common_f22", "capability:gen_long_gsm8k_rung_common_f22",
                       "capability:ifeval_2048", "capability:ifeval_4096", "capability:ifeval_8192",
                       "capability:ifeval_longsi_2048", "capability:ifeval_longsi_4096", "capability:ifeval_longsi_8192"):
            assert g.basis.max_prompt is None and not g.basis.sized_to_max_prompt, key
    assert "SIZED_BY_MAX_PROMPT_PLUS_MAX" in sg.__all__


def test_grader_entry_is_informational_and_served_by_nothing():
    """The model-grader traffic (Inspect's MATH equivalence + OR-Bench refusal
    graders on the unsteered base route): its own key with its own p99, all
    6,348 calls CAP-LESS (clamp rate 0 by construction), refused by
    resolve_geometry AND by pass1_geometry — the grader rides the target's
    server; nobody stands one up for it."""
    g = sg.geometry_for("capability:grader")
    assert g.informational and not g.provisional and not g.requires_recertification
    assert (g.max_model_len, g.max_num_seqs) == (2304, 170) and g.is_identity
    b = g.basis
    assert b.n_calls == b.n_capless == 6348 and b.n_capped == 0 and b.n_rejected == 0
    assert b.requested_max_tokens == 0 and b.clamp_rate_at_L == 0.0 and b.frac_over_L == 0.0019
    assert g.sized_by == sg.SIZED_BY_TOTAL and b.floor == sg.next_block(2150) == 2160
    assert "INFORMATIONAL" in b.note and "CAP-LESS" in b.note and "gen_long" in b.note
    assert "inspect/math=3710" in b.note and "inspect/or_bench_hard_1k=2638" in b.note
    with pytest.raises(sg.GeometryInformational, match="grader"):
        sg.resolve_geometry(g.key)
    with pytest.raises(sg.GeometryInformational):
        sg.resolve_geometry(g.key, model_key="gemma4e4b", max_num_seqs=8)
    with pytest.raises(ValueError, match="informational"):
        sg.pass1_geometry(g.key)
    assert g.key in sg.informational_keys() and g.key not in sg.TINYCAP_KEYS


def test_pass1_geometry_serves_the_footprint_for_every_tinycap_key_and_nothing_else():
    """pass1_geometry after the landing: a MEASURED tinycap entry is still
    served at the certified footprint for a measurement re-run (labelled
    MEASUREMENT_SOURCE — a pass 1 extends the ledger, e.g. the fitted cells'
    generation tasks the accepted pools did not contain), the provisional one
    as before (PROVISIONAL_SOURCE); a pre-tinycap eval and the grader refuse."""
    for key in sg.TINYCAP_KEYS:
        g = sg.geometry_for(key)
        r = sg.pass1_geometry(key)
        assert (r.max_model_len, r.max_num_seqs) == (sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS)
        assert r.eval == key and r.entry is g and r.measurement_pass and not r.overridden and not r.legacy
        assert r.is_identity and not r.over_budget
        src = sg.PROVISIONAL_SOURCE if g.provisional else sg.MEASUREMENT_SOURCE
        assert r.window_source == r.seqs_source == src and r.provisional == g.provisional
        d = r.as_dict()
        assert d["measurement_pass"] is True and d["provisional"] is g.provisional
        desc = r.describe()
        assert ("PROVISIONAL" in desc) == g.provisional
        assert ("MEASUREMENT PASS" in desc) == (not g.provisional)
        if not g.provisional:
            assert f"measured window {g.max_model_len}" in desc and src in desc
    # a measured NON-tinycap entry is not served through pass1_geometry
    with pytest.raises(ValueError, match="not a tinycap sub-entry"):
        sg.pass1_geometry("capability:short")
    with pytest.raises(ValueError, match="not a tinycap sub-entry"):
        sg.pass1_geometry("petri")
    with pytest.raises(ValueError, match="informational"):
        sg.pass1_geometry("capability:grader")
    assert sg.MEASUREMENT_SOURCE != sg.PROVISIONAL_SOURCE


def test_pass1_geometry_sizes_s_from_the_model_budget():
    """E7 -> the per-model budgets: the measurement pass keeps the certified
    WINDOW but sizes S from the served MODEL's measured budget, the derivation
    ``resolve_geometry(model_key=)`` performs for a measured entry — gemma4e4b's
    589,824 buys 36 at the Petri window where the Llama-certified footprint
    serves 24. Without a key nothing changes."""
    rb = sg.budget_for("gemma4e4b")
    for key in sg.TINYCAP_KEYS:                    # PW11: measured AND provisional alike
        prov = sg.geometry_for(key).provisional
        r = sg.pass1_geometry(key, model_key="gemma4e4b")
        assert r.max_model_len == sg.PETRI_MAX_MODEL_LEN
        assert r.window_source == (sg.PROVISIONAL_SOURCE if prov else sg.MEASUREMENT_SOURCE)
        assert r.max_num_seqs == rb.entry.seqs_at(sg.PETRI_MAX_MODEL_LEN) == 589_824 // sg.PETRI_MAX_MODEL_LEN == 36
        assert r.seqs_source == "model_budget" and r.model_key == "gemma4e4b" and r.budget == rb
        assert r.budget_tokens == r.batched_tokens == 589_824 and r.budget_source == sg.MEASURED_BUDGET_SOURCE
        assert r.is_identity and not r.over_budget and r.measurement_pass and not r.overridden and not r.legacy
        assert r.provisional is prov and r.eval == key and r.entry is sg.geometry_for(key)
        d = r.as_dict()
        assert (d["model_key"], d["budget"], d["seqs_source"]) == ("gemma4e4b", 589_824, "model_budget")
        assert d["provisional"] is prov and d["measurement_pass"] is True
        assert d["identity"] is True and d["over_budget"] is False
        desc = r.describe()
        assert ("PROVISIONAL" if prov else "MEASUREMENT PASS") in desc
        assert "max_num_seqs=36 (model_budget)" in desc
        assert "model gemma4e4b: measured" in desc and "OVER BUDGET" not in desc
    # the same S resolve_geometry derives for the served window with that key
    ref = sg.resolve_geometry("capability", max_model_len=sg.PETRI_MAX_MODEL_LEN, model_key="gemma4e4b")
    assert (ref.max_num_seqs, ref.seqs_source) == (36, "model_budget")
    # llama8b: its OWN row (the 2026-09-10 lens-stack MEASUREMENT 475,136 = 29 x 16384, the sdfbase-llama8b
    # serving gate; it replaced the 2026-09-06 provisional 229,376 ceiling), not the certified 393,216 — 29,
    # not 24 — and the resolution says so
    r = sg.pass1_geometry("capability:tiny", model_key="llama8b")
    assert r.max_num_seqs == 475_136 // sg.PETRI_MAX_MODEL_LEN == 29 and r.budget_tokens == 475_136
    assert r.budget_source == sg.MEASURED_BUDGET_SOURCE and r.budget.per_model
    assert r.budget.measured and not r.budget.provisional
    assert "model llama8b: measured" in r.describe() and "PROVISIONAL" not in r.describe()
    # a model whose own row is still a PROVISIONAL ceiling is served from THAT row (smaller than the certified
    # default) and the resolution says so: qwen35_9b's 196,608 -> 12, not 24
    r = sg.pass1_geometry("capability:tiny", model_key="qwen35_9b")
    assert r.max_num_seqs == 196_608 // sg.PETRI_MAX_MODEL_LEN == 12 and r.budget_tokens == 196_608
    assert r.budget_source == sg.PROVISIONAL_BUDGET_SOURCE and r.budget.per_model
    assert not r.budget.measured and r.budget.provisional
    assert "model qwen35_9b: provisional-conservative — PROVISIONAL" in r.describe()
    # a hybrid family: seqs_at's per-slot cap applies (inactive at this window: 45 < 256; 741,888 // 16,384, the 2026-09-13 row)
    q = sg.pass1_geometry("capability:tiny", model_key="qwen35_4b")
    assert q.max_num_seqs == min(741_888 // sg.PETRI_MAX_MODEL_LEN, sg.HYBRID_SEQS_CAP) == 45
    assert q.max_num_seqs == sg.BUDGET_BY_MODEL["qwen35_4b"].seqs_at(sg.PETRI_MAX_MODEL_LEN)
    # an unknown key refuses (never a silent default)
    with pytest.raises(sg.UnknownModelKey):
        sg.pass1_geometry("capability:tiny", model_key="gemma4-e4b")
    # no key: the footprint's own S and sources, exactly as before (PW11: tiny is
    # MEASURED, so its pass-1 source is the measurement one; the provisional
    # tiny_prompted still reads provisional_footprint)
    r0 = sg.pass1_geometry("capability:tiny")
    assert (r0.max_num_seqs, r0.seqs_source, r0.model_key, r0.budget) == (
        vs.DEFAULT_MAX_NUM_SEQS, sg.MEASUREMENT_SOURCE, None, None)
    p0 = sg.pass1_geometry("capability:tiny_prompted")
    assert (p0.max_num_seqs, p0.seqs_source, p0.model_key, p0.budget) == (
        vs.DEFAULT_MAX_NUM_SEQS, sg.PROVISIONAL_SOURCE, None, None)
    assert r0.budget_tokens == sg.BATCHED_TOKEN_BUDGET and r0.is_identity
    # a measured NON-tinycap entry is still not served through pass1_geometry, key or no key
    with pytest.raises(ValueError, match="not a tinycap sub-entry"):
        sg.pass1_geometry("capability:short", model_key="gemma4e4b")


def test_unserved_entries_do_not_label_windows_in_the_sweep():
    """The decode sweep labels each window with the table keys SERVED at it
    (row provenance, figure legend, the NOT MEASURED line). A PROVISIONAL
    entry (tiny_prompted: analytic 8,448, served for measurement at the
    footprint, never at 8,448), an ``informational`` one (the grader at 2,304)
    and the ``requires_recertification`` petri:<model> rows are never served
    (``resolve_geometry`` refuses all three), so none may label a window; the
    five MEASURED tinycap entries do label theirs."""
    sw = _load("_sweep_window_labels", "scripts/served_decode_sweep.py")
    # the certified footprint is ALSO the constitution-prompted GSM8K cells' 8,192-pass window (capability:gen_long_gsm8k_longsi,
    # served there; llama-paper-rows 2026-09-12): both label it, and the top-level eval still names the standup geometry
    assert sw.window_labels(sg.PETRI_MAX_MODEL_LEN) == ("petri", "capability:gen_long_gsm8k_longsi")
    assert sw.eval_key_for_window(sg.PETRI_MAX_MODEL_LEN) == "petri"
    assert not any(k.startswith("petri:") for k in sw.window_labels(sg.PETRI_MAX_MODEL_LEN))
    assert "capability" in sw.window_labels(sg.geometry_for("capability").max_model_len)
    for key in sg.provisional_keys() + sg.informational_keys():
        assert key not in sw.window_labels(sg.TABLE[key].max_model_len), key
    # tiny_prompted's provisional 8,448 labels nothing of its own; since the follow-up the SAME window is SERVED by the
    # constitution-prompted cells' pass-1 entry (6,385 + 2,048 -> 8,448, a coincidence of arithmetic), so the label is that
    # entry's and never tiny_prompted's
    assert sw.window_labels(8448) == ("capability:ifeval_longsi_2048",) and sw.eval_key_for_window(8448) == "capability:ifeval_longsi_2048"
    assert "capability:tiny_prompted" not in sw.window_labels(8448)
    assert sw.window_labels(2304) == () and sw.eval_key_for_window(2304) is None     # grader
    for key, (L, _, _) in MEASURED_TINYCAP.items():
        assert sw.window_labels(L) == (key,) and sw.eval_key_for_window(L) == key
    # the analyzer's NOT MEASURED line is the sweep's label, not its own table walk
    an_src = (REPO / "scripts/geom_sweep_analyze.py").read_text()
    assert "sw.window_labels(int(L))" in an_src and "not sg.TABLE[k].requires_recertification" not in an_src


def test_provisional_entries_must_be_empty_and_at_the_footprint():
    empty = dict(n_calls=0, p99_total=None, max_total=None, p99_prompt=None,
                 requested_max_tokens=1, clamp_rate_at_L=None, frac_over_L=None,
                 families_measured=("m",), measured_on=sg.UNMEASURED, source="x")
    ok = sg.ServedGeometry("e", sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS,
                           sg.GeometryBasis(**empty), task="t", provisional=True)
    assert ok.provisional and sg.sizing_problem(ok) is None and ok.sized_by == "unmeasured"
    # numbers in a provisional entry: it is a measured entry wearing the wrong flag
    with pytest.raises(ValueError, match="measured basis fields"):
        sg.ServedGeometry("e", sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS,
                          sg.GeometryBasis(**dict(empty, n_calls=10)), provisional=True)
    with pytest.raises(ValueError, match="measured basis fields"):
        sg.ServedGeometry("e", sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS,
                          sg.GeometryBasis(**dict(empty, p99_prompt=100)), provisional=True)
    with pytest.raises(ValueError, match="measured basis fields"):
        sg.ServedGeometry("e", sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS,
                          sg.GeometryBasis(**dict(empty, clamp_rate_at_L=0.0)), provisional=True)
    with pytest.raises(ValueError, match="measured_on must be"):
        sg.ServedGeometry("e", sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS,
                          sg.GeometryBasis(**dict(empty, measured_on="2026-09-03")),
                          provisional=True)
    # and it may not name a window of its own (that would be a guess)
    with pytest.raises(ValueError, match="certified footprint"):
        sg.ServedGeometry("e", 2048, sg.identity_seqs(2048), sg.GeometryBasis(**empty),
                          provisional=True)
    # a NON-provisional empty basis is refused as before (cannot size)
    with pytest.raises(ValueError, match="unmeasured"):
        sg.ServedGeometry("e", 2048, sg.identity_seqs(2048), sg.GeometryBasis(**empty))


def test_analytic_provisional_entries_carry_a_derived_window_on_the_prompt_plus_cap_floor():
    """PW11 (Evals' hold on tiny_prompted): a provisional entry may carry an
    ANALYTIC window — the derivation stated, the derived prompt bound as
    p99_prompt, nothing measured from calls — held to the prompt+cap floor
    and the headroom like any window; every other shape is refused."""
    empty = dict(n_calls=0, p99_total=None, max_total=None, p99_prompt=None,
                 requested_max_tokens=1, clamp_rate_at_L=None, frac_over_L=None,
                 families_measured=("m",), measured_on=sg.UNMEASURED, source="x")
    an = dict(empty, p99_prompt=8229, analytic="1664 + 6560 + 5")
    ok = sg.ServedGeometry("e", 8448, sg.identity_seqs(8448), sg.GeometryBasis(**an), provisional=True)
    assert ok.provisional and sg.sizing_problem(ok) is None
    assert ok.sized_by == sg.SIZED_BY_ANALYTIC and ok.basis.floor == 8240 and not ok.basis.measured
    assert sg.ServedGeometry("e", 8240, sg.identity_seqs(8240), sg.GeometryBasis(**an), provisional=True)
    # below the floor / past the headroom: refused like any window
    with pytest.raises(ValueError, match="below the analytic floor"):
        sg.ServedGeometry("e", 8192, sg.identity_seqs(8192), sg.GeometryBasis(**an), provisional=True)
    with pytest.raises(ValueError, match="headroom"):
        sg.ServedGeometry("e", 8240 + 1024, sg.identity_seqs(9264), sg.GeometryBasis(**an), provisional=True)
    # an analytic basis needs its prompt bound, and derives no completion
    with pytest.raises(ValueError, match="derived prompt bound"):
        sg.ServedGeometry("e", 8448, sg.identity_seqs(8448),
                          sg.GeometryBasis(**dict(an, p99_prompt=None)), provisional=True)
    with pytest.raises(ValueError, match="PROMPT bound only"):
        sg.ServedGeometry("e", 8448, sg.identity_seqs(8448),
                          sg.GeometryBasis(**dict(an, p99_total=8300)), provisional=True)
    # nothing measured from calls may ride along
    for k, v in (("n_calls", 10), ("clamp_rate_at_L", 0.0), ("frac_over_L", 0.0),
                 ("n_capless", 0), ("clamp_rate_at_total_sizing", 0.0)):
        with pytest.raises(ValueError, match="measured basis fields"):
            sg.ServedGeometry("e", 8448, sg.identity_seqs(8448),
                              sg.GeometryBasis(**dict(an, **{k: v}, **({"n_calls": 10} if k != "n_calls" else {}))),
                              provisional=True)
    with pytest.raises(ValueError, match="measured_on must be"):
        sg.ServedGeometry("e", 8448, sg.identity_seqs(8448),
                          sg.GeometryBasis(**dict(an, measured_on="2026-09-04")), provisional=True)
    # an analytic basis on a NON-provisional entry is a guess wearing a measurement's flag
    with pytest.raises(ValueError, match="provisional by definition"):
        sg.ServedGeometry("e", 8448, sg.identity_seqs(8448), sg.GeometryBasis(**an))
    # informational + provisional is a contradiction
    with pytest.raises(ValueError, match="cannot be provisional"):
        sg.ServedGeometry("e", 8448, sg.identity_seqs(8448), sg.GeometryBasis(**an),
                          provisional=True, informational=True)
    # the derivation must be stated, not an empty string
    with pytest.raises(ValueError, match="analytic must be"):
        sg.GeometryBasis(**dict(an, analytic="  "))
    # the plain unmeasured provisional shape is unchanged
    assert sg.ServedGeometry("e", sg.PETRI_MAX_MODEL_LEN, vs.DEFAULT_MAX_NUM_SEQS,
                             sg.GeometryBasis(**empty), provisional=True).sized_by == sg.SIZED_BY_UNMEASURED


def test_geometry_basis_n_capless_field():
    """``n_capless`` (Evals' verdict): the cap-less calls of a basis — in the
    p99 pools, outside the clamp basis; ``n_capped`` is the clamp basis."""
    b = _basis(n_calls=100, n_capless=40)
    assert b.n_capless == 40 and b.n_capped == 60
    assert _basis().n_capless is None and _basis().n_capped is None
    for bad in (-1, True, 2.5):
        with pytest.raises(ValueError, match="n_capless"):
            _basis(n_capless=bad)
    with pytest.raises(ValueError, match="SUBSET"):
        _basis(n_calls=10, n_capless=11)
    g = sg.ServedGeometry("e", 2048, sg.identity_seqs(2048), b)
    assert g.as_dict()["basis"]["n_capless"] == 40
    # an informational entry resolves for inspection and refuses to launch
    info = sg.ServedGeometry("e", 2048, sg.identity_seqs(2048), _basis(n_capless=100, requested_max_tokens=0),
                             informational=True)
    assert info.informational and sg.sizing_problem(info) is None and info.basis.n_capped == 0


def test_geometry_basis_n_rejected_field():
    b = _basis(n_rejected=3)
    assert b.n_rejected == 3 and b.rejection_rate == 0.03
    assert _basis().n_rejected is None and _basis().rejection_rate is None
    g = sg.ServedGeometry("e", 2048, sg.identity_seqs(2048), b)
    assert g.as_dict()["basis"]["n_rejected"] == 3
    d = sg.resolve_geometry("multiturn_attacks").describe()
    assert "n_rejected" not in d                        # absent when not measured
    for bad in (-1, True, 2.5):
        with pytest.raises(ValueError, match="n_rejected"):
            _basis(n_rejected=bad)
    with pytest.raises(ValueError, match="SUBSET"):
        _basis(n_calls=10, n_rejected=11)
    # the describe() table carries the column and the provisional / analytic /
    # informational flags; the measured tiny row reads its ledger numbers
    md = sg.describe()
    assert "| n_rejected |" in md.splitlines()[0]
    row = [l for l in md.splitlines() if l.startswith("| capability:tiny_prompted |")][0]
    assert "| unmeasured |" in row and "provisional,analytic" in row and "| 8448 | 46 |" in row
    row = [l for l in md.splitlines() if l.startswith("| capability:grader |")][0]
    assert "informational" in row and "| 2304 | 170 |" in row and "| 6348 |" in row
    row = [l for l in md.splitlines() if l.startswith("| capability:tiny |")][0]
    assert "| 1792 | 219 |" in row and "| 2026-09-04 |" in row and "| 0 |" in row and "provisional" not in row
    assert "OVER" not in md.split("budget =")[0]


# --------------------------------------------------------------------------- #
# (a'') the measurement script sizes from a ledger, both branches, rejected    #
#       calls counted (Lane D)                                                 #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def measure():
    return _load("_tinycap_geometry_measure", "scripts/tinycap_geometry_measure.py")


def _ledger_row(task, prompt, *, completion=None, cap=1, thinking=False, rejected=False,
                sample="0/0", error_class=None):
    """A synthetic ledger row in the contract's flat schema. A rejected row
    is the ledger's ``context_length`` class (E12 item 5) unless
    ``error_class`` says otherwise; a served row is ``none``."""
    from sisteer.capeval.tinycap import contracts as C
    if error_class is None:
        error_class = (C.LEDGER_ERROR_CLASS_CONTEXT_LENGTH if rejected else C.LEDGER_ERROR_CLASS_NONE)
    row = dict(ts="2026-09-03T00:00:00.000+00:00", task=task, sample=sample, vmodel="petri__x",
               thinking=thinking, prompt_tokens_pre=prompt, max_tokens_requested=cap,
               has_tools=False, n_tools=0, status=400 if rejected else 200,
               error_code="upstream_error" if rejected else None, rejected=rejected,
               usage_prompt_tokens=None if rejected else prompt,
               usage_completion_tokens=None if rejected else completion,
               finish_reason=None if rejected else "stop", latency_s=0.1,
               tokenize_mismatch=None if rejected else False, error_class=error_class,
               # the per-attempt identity columns (a synthetic row
               # carries them null; the measurer never reads them)
               request_id=None, body_sha256=None)
    assert tuple(row) == C.LEDGER_ROW_FIELDS
    return row


def _write_ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_measure_sizes_both_branches_and_counts_rejected_calls(measure, tmp_path):
    import json as _json
    # branch 1 (agent, fitted): prompts 201..300, one-token completions, cap 1
    tiny = [_ledger_row("inspect/bfcl_200", 200 + i, completion=1, cap=1, sample=f"{i}/0")
            for i in range(1, 101)]
    # branch 2 (gen_long, fitted, thinking off): 100 served calls, prompts
    # 1005..1500 (step 5), completions 100, cap 2048 — plus ONE rejected call
    # whose prompt (15000) + cap did not fit the footprint. (The full GSM8K
    # task: MATH-500 rows pool into their own class in BOTH
    # arms — capability:gen_long_math — so a gen_long fixture no longer uses them.)
    gen = [_ledger_row("inspect/gsm8k", 1000 + 5 * i, completion=100, cap=2048, sample=f"m{i}/0")
           for i in range(1, 101)]
    gen.append(_ledger_row("inspect/gsm8k", 15000, cap=2048, rejected=True, sample="m_rej/0"))
    # a row tagged with a task OUTSIDE the tinycap families (a legacy control) is
    # skipped and counted; an UNTAGGED row is skipped and counted only under
    # --allow-untagged (it refuses otherwise — the negative test below)
    legacy = [_ledger_row("gsm8k", 500, completion=50, cap=512), dict(_ledger_row(None, 10), task=None)]
    led = tmp_path / "fitted" / "usage_ledger.jsonl"
    _write_ledger(led, tiny + gen + legacy)
    # the cell kind is inferred from the store manifest beside the ledger
    (led.parent / "cells" / "k").mkdir(parents=True)
    (led.parent / "cells" / "k" / "manifest.json").write_text(
        _json.dumps({"cell_id": "safety_refusal__psr_kl_all"}))
    assert measure.infer_cell_kind(led) == "fitted"

    res = measure.measure([(led, "fitted")], measured_on="2026-09-04", allow_untagged=True)
    assert res["skipped_rows"] == {"gsm8k": 1, measure.UNTAGGED_KEY: 1}
    assert res["refused_keys"] == [] and res["rule"]["allow_untagged"] is True
    by_key = {p["key"]: p for p in res["proposals"]}
    assert set(by_key) == {"capability:agent", "capability:gen_long"}
    assert set(res["absent_keys"]) == {"capability:tiny", "capability:tiny_prompted",
                                       "capability:gen_long_thinking", "capability:agent_thinking",
                                       "capability:gen_long_math", "capability:gen_long_math_longsi"}

    # --- branch 1, hand-sized: p99_total = 300 + 1 = 301 (nearest rank 99 of 100 ->
    # prompt 299 + 1 = 300? no: sorted totals 202..301, ceil(0.99*100)=99 -> 300);
    t = by_key["capability:agent"]
    assert t["status"] == "proposed" and t["n_calls"] == 100 and t["n_rejected"] == 0
    assert t["n_capped"] == 100 and t["n_capless"] == 0 and t["informational"] is False
    assert t["p99_total"] == 300 and t["max_total"] == 301 and t["p99_prompt"] == 299
    assert t["L_at_total_sizing"] == 512 and t["clamp_rate_at_total_sizing"] == 0.0
    assert (t["L"], t["S"], t["sized_by"]) == (512, 768, sg.SIZED_BY_TOTAL)
    assert t["clamp_rate_at_L"] == 0.0 and t["frac_over_L"] == 0.0
    e = t["_entry"]
    assert isinstance(e, sg.ServedGeometry) and e.key == "capability:agent" and not e.provisional
    assert e.basis.n_rejected == 0 and e.basis.measured_on == "2026-09-04"
    assert e.basis.n_capless == 0 and not e.informational
    assert sg.sizing_problem(e) is None and e.is_identity and not e.over_budget
    assert e.basis.floor == sg.next_block(300) == 304 and 304 <= 512 <= 304 + sg.SIZING_HEADROOM

    # --- branch 2, hand-sized: 101 measured calls; totals: served 1105..1600,
    # rejected 15000 + 2048 = 17048; p99 (rank ceil(0.99*101)=100) = 1600 ->
    # L_total = 1792; at 1792 EVERY call requests prompt + 2048 > 1792 -> clamp 100%
    # -> second branch: p99_prompt (rank 100 of 101) = 1500; L = round256(1500 + 2048
    # = 3548) = 3584; S = 393216 // 3584 = 109; clamp at 3584 = prompt > 1536 -> the
    # rejected call only = 1/101; frac_over 3584 = the rejected total only = 1/101.
    g = by_key["capability:gen_long"]
    assert g["status"] == "proposed" and g["n_calls"] == 101 and g["n_rejected"] == 1
    assert g["n_unmeasured"] == 0 and g["n_tokenize_mismatch"] == 0 and g["n_unexpected_rejected"] == 0
    assert g["p99_total"] == 1600 and g["max_total"] == 17048 and g["p99_prompt"] == 1500
    assert g["L_at_total_sizing"] == 1792 and g["clamp_rate_at_total_sizing"] == 1.0
    assert (g["L"], g["S"], g["sized_by"]) == (3584, 109, sg.SIZED_BY_PROMPT_PLUS_MAX)
    assert g["clamp_rate_at_L"] == pytest.approx(1 / 101) == g["frac_over_L"]
    assert g["requested_max_tokens"] == 2048 and g["caps"] == [2048]
    e = g["_entry"]
    assert e.basis.n_rejected == 1 and e.basis.rejection_rate == pytest.approx(1 / 101)
    assert e.basis.clamp_rate_at_total_sizing == 1.0 and e.sized_by == sg.SIZED_BY_PROMPT_PLUS_MAX
    assert e.basis.floor == sg.next_block(1500 + 2048) == 3552 and 3552 <= 3584 <= 3552 + 512
    assert sg.sizing_problem(e) is None and sg.identity_seqs(3584) == 109
    assert "1 rejected" in e.basis.note and "second branch" in e.basis.note
    assert "excluded from the p99" not in e.basis.note        # R6: the note no longer claims an exclusion
    assert "refused, never sized" in e.basis.note
    # the rendered literals name the key, the field and the measured date
    py = measure.render_entry_py(e)
    assert "task='gen_long'" in py and "n_rejected=1" in py and "identity_seqs(3584)" in py
    pin = measure.render_pinned_row(e)
    assert pin.startswith("    'capability:gen_long': dict(L=3584, S=109, n=101")
    assert "n_rejected=1" in pin and "measured_on='2026-09-04'" in pin
    assert "n_capless=0" in pin and "informational" not in pin and "n_capless=0" in py
    md = measure.render_doc_rows(res["proposals"])
    assert "| capability:gen_long | 101 | 1 |" in md and "3,584 / 109" in md
    assert "| cap-less (outside the clamp basis) | sized by |" in md.splitlines()[0]
    assert "| 0 | p99_prompt_plus_max_tokens |" in md


def test_measure_refuses_thin_pools_mixed_kinds_and_writes_the_proposal_files(measure, tmp_path):
    # a smoke-sized pool proposes NOTHING
    led = tmp_path / "smoke" / "usage_ledger.jsonl"
    _write_ledger(led, [_ledger_row("inspect/bfcl_200", 100 + i, completion=1) for i in range(5)])
    res = measure.measure([(led, "prompted")], measured_on="2026-09-04")
    (p,) = res["proposals"]
    assert p["key"] == "capability:agent" and p["status"] == "insufficient"
    assert "_entry" not in p and "min_calls" in p["reason"]
    # thinking-ON rows pool into the _thinking twin; prompted-ness comes from the cell kind
    assert measure.classify(_ledger_row("inspect/bfcl_200", 1, thinking=True), prompted=True) == \
        "capability:agent_thinking"
    assert measure.classify(_ledger_row("evalplus/mbpp_plus:open", 1, thinking=True),
                            prompted=False) == "capability:gen_long_thinking"
    # the removed loglikelihood family's rows pool into no class (an old ledger's rows are skipped)
    assert measure.classify(_ledger_row("served_ll/tinyTruthfulQA", 1, thinking=True),
                            prompted=False) is None
    assert measure.classify(_ledger_row("mmlu", 1), prompted=False) is None
    # a ledger with no manifest beside it and no explicit kind refuses
    with pytest.raises(measure.MeasureError, match="cannot infer"):
        measure.infer_cell_kind(led)
    with pytest.raises(measure.MeasureError, match="cell kind"):
        measure.pool_rows([(_ledger_row("inspect/bfcl_200", 1), "none")])
    # a torn trailing line is dropped and counted
    led.write_text(led.read_text() + '{"ts": "x", "task": "inspect/bfc')
    rows, torn = measure.read_ledger(led)
    assert len(rows) == 5 and torn == 1
    # the CLI writes the four proposal files — once it knows the grader route:
    # a ledger with no manifests beside it (this synthetic one) needs
    # --grader-vmodel, else the grader's calls could pool into the target's
    # classes unnoticed (PW11)
    out = tmp_path / "proposal"
    with pytest.raises(SystemExit, match="grader route"):
        measure.main(["--ledger", f"{led}:prompted", "--out", str(out),
                      "--measured-on", "2026-09-04", "--min-calls", "3"])
    rc = measure.main(["--ledger", f"{led}:prompted", "--out", str(out), "--grader-vmodel", "petri__base",
                       "--measured-on", "2026-09-04", "--min-calls", "3"])
    assert rc == 0
    prop = json.loads((out / "geometry_proposal.json").read_text())
    assert prop["measured_on"] == "2026-09-04" and prop["torn_lines"] == 1
    assert prop["inputs"][0]["grader_vmodel"] == "petri__base" and prop["inputs"][0]["n_grader_rows"] == 0
    assert "clamp_basis" in prop["rule"] and "grader_route" in prop["rule"]
    assert prop["proposals"][0]["status"] == "proposed"
    assert prop["proposals"][0]["entry"]["key"] == "capability:agent"
    assert "_entry" not in prop["proposals"][0]
    assert (out / "geometry_proposal.md").read_text().startswith("## 6. tinycap suite")
    ent = (out / "geometry_entries.py.txt").read_text()
    assert "eval='capability', task='agent'" in ent and "n_rejected=0" in ent
    assert "'capability:agent': dict(" in (out / "geometry_pinned.py.txt").read_text()
    # nearest-rank percentiles, stated
    assert measure.percentile([1, 2, 3, 4], 0.5) == 2 and measure.percentile([1, 2, 3, 4], 0.99) == 4
    assert measure.round_up(3548) == 3584 and measure.round_up(512) == 512


def _clean_pool_rows(n=30):
    """A pool big enough to propose: n served gen_long calls, thinking off."""
    return [_ledger_row("inspect/gsm8k", 1000 + i, completion=100, cap=2048, sample=f"m{i}/0")
            for i in range(n)]


_REJECTED = dict(rejected=True, usage_prompt_tokens=None, usage_completion_tokens=None,
                 finish_reason=None, tokenize_mismatch=None)


@pytest.mark.parametrize("case", ["unmeasured_prompt", "tokenize_mismatch", "transport_error",
                                  "status_502", "status_429", "no_completion_usage",
                                  "schema_400", "unsupported_field_400", "auth_401",
                                  "route_refused_404", "context_length_class_with_wrong_status",
                                  "legacy_row_without_error_class", "null_error_class"])
def test_measure_refuses_a_compromised_ledger_and_proposes_no_entry(measure, tmp_path, case):
    """R6 + E12 item 5 (negative): ONE row whose prompt was never measured,
    whose server prompt count disputes the pre-tokenize, whose completion
    usage is missing, or whose rejection the LEDGER did not classify
    ``context_length`` — a transport death, a 5xx, a 429, a tool-schema or
    unsupported-field 400 (``bad_request_other``), an auth or route
    refusal, a ``context_length`` row whose status is not the 400 the
    class implies, a legacy row WITHOUT the column — makes the pool
    REFUSED: no ``ServedGeometry``, no literal in the entries file, a
    non-zero exit — while a clean pool beside it still proposes. Before
    E12 item 5 every non-ledger HTTP 400 was scored as ``prompt + cap``;
    before R6 every failure was."""
    bad = _ledger_row("inspect/gsm8k", 1200, completion=100, cap=2048, sample="bad/0")
    if case == "unmeasured_prompt":
        bad.update(prompt_tokens_pre=None, status=None, error_code="ledger_tokenize_failed",
                   error_class="ledger_blocked", **_REJECTED)
        needle = "unmeasured"
    elif case == "tokenize_mismatch":
        bad.update(tokenize_mismatch=True, usage_prompt_tokens=1201)
        needle = "tokenize_mismatch"
    elif case == "transport_error":
        bad.update(status=None, error_code="ledger_transport_error", error_class="transport", **_REJECTED)
        needle = "error_class transport"
    elif case == "status_502":
        bad.update(status=502, error_code="http_502", error_class="server_error", **_REJECTED)
        needle = "error_class server_error"
    elif case == "status_429":
        bad.update(status=429, error_code="http_429", error_class="rate_limited", **_REJECTED)
        needle = "error_class rate_limited"
    elif case == "no_completion_usage":
        bad.update(usage_completion_tokens=None)
        needle = "usage_completion_tokens"
    elif case == "schema_400":
        # a tool-schema 400: the same status as a context-length refusal, nothing to do with the window
        bad.update(status=400, error_code="BadRequestError", error_class="bad_request_other", **_REJECTED)
        needle = "error_class bad_request_other"
    elif case == "unsupported_field_400":
        bad.update(status=400, error_code="invalid_request_error", error_class="bad_request_other", **_REJECTED)
        needle = "error_class bad_request_other"
    elif case == "auth_401":
        bad.update(status=401, error_code="bad_token", error_class="auth", **_REJECTED)
        needle = "error_class auth"
    elif case == "route_refused_404":
        bad.update(status=404, error_code="unknown_model", error_class="route_refused", **_REJECTED)
        needle = "error_class route_refused"
    elif case == "context_length_class_with_wrong_status":
        bad.update(status=502, error_code="http_502", error_class="context_length", **_REJECTED)
        needle = "error_class context_length with status 502"
    elif case == "legacy_row_without_error_class":
        bad.update(status=400, error_code="upstream_error", **_REJECTED)
        del bad["error_class"]
        needle = "no error_class"
    else:
        bad.update(status=400, error_code="upstream_error", error_class=None, **_REJECTED)
        needle = "no error_class"
    tiny = [_ledger_row("inspect/bfcl_200", 200 + i, completion=1, cap=1, sample=f"{i}/0")
            for i in range(30)]
    led = tmp_path / "usage_ledger.jsonl"
    _write_ledger(led, tiny + _clean_pool_rows() + [bad])
    res = measure.measure([(led, "fitted")], measured_on="2026-09-04")
    by_key = {p["key"]: p for p in res["proposals"]}
    g = by_key["capability:gen_long"]
    assert g["status"] == "refused" and "_entry" not in g and "entry" not in g
    assert needle in g["reason"] and "compromised" in g["reason"]
    assert res["refused_keys"] == ["capability:gen_long"]
    # the compromising row is counted in its own class, never as prompt + cap
    if case in ("unmeasured_prompt", "no_completion_usage"):
        assert g["n_unmeasured"] == 1 and g["n_calls"] == 30
    elif case == "tokenize_mismatch":
        assert g["n_tokenize_mismatch"] == 1
    else:
        assert g["n_unexpected_rejected"] == 1 and g["n_rejected"] == 0 and g["n_calls"] == 30
        [label] = g["unexpected_rejections"]
        assert label.startswith(needle)
    # the clean pool beside it still proposes
    assert by_key["capability:agent"]["status"] == "proposed"
    # the CLI: no literal for the refused key, a non-zero exit
    out = tmp_path / "proposal"
    rc = measure.main(["--ledger", f"{led}:fitted", "--out", str(out), "--measured-on", "2026-09-04",
                       "--grader-vmodel", "petri__base"])
    assert rc == 1
    entries = (out / "geometry_entries.py.txt").read_text()
    assert "task='gen_long'" not in entries and "task='agent'" in entries
    prop = json.loads((out / "geometry_proposal.json").read_text())
    assert prop["refused_keys"] == ["capability:gen_long"]
    assert "REFUSED" in (out / "geometry_proposal.md").read_text()


def test_measure_refuses_untagged_rows_unless_told_to_skip_them(measure, tmp_path):
    """The 2026-09-03 smoke's ledger carried task=null on every row (the tiny
    posters did not yet send X-Tinycap-Task): the pool is keyed by task
    family, so the measurement REFUSES rather than guess a class;
    ``--allow-untagged`` skips and counts them, recorded in the rule."""
    from sisteer.capeval.tinycap import contracts as C
    rows = _clean_pool_rows() + [dict(_ledger_row(None, 500, completion=1, cap=1), task=None)]
    led = tmp_path / "usage_ledger.jsonl"
    _write_ledger(led, rows)
    with pytest.raises(measure.MeasureError, match="no task attribution"):
        measure.measure([(led, "fitted")], measured_on="2026-09-04")
    with pytest.raises(measure.MeasureError, match="X-Tinycap-Task"):
        measure.pool_rows([(r, "fitted") for r in rows])
    res = measure.measure([(led, "fitted")], measured_on="2026-09-04", allow_untagged=True)
    assert res["skipped_rows"] == {measure.UNTAGGED_KEY: 1} and res["rule"]["allow_untagged"] is True
    assert [p["status"] for p in res["proposals"]] == ["proposed"]
    out = tmp_path / "proposal"
    with pytest.raises(SystemExit, match="no task attribution"):
        measure.main(["--ledger", f"{led}:fitted", "--out", str(out), "--grader-vmodel", "petri__base"])
    assert measure.main(["--ledger", f"{led}:fitted", "--out", str(out), "--allow-untagged",
                         "--grader-vmodel", "petri__base"]) == 0
    # the ledger's context_length class is the ONE rejection scored as prompt + cap (E12 item 5):
    # a generic 400 — whatever its error_code — is never that class by status alone
    ok = _ledger_row("inspect/gsm8k", 15000, cap=2048, rejected=True)
    assert ok["error_class"] == "context_length" and measure.rejection_class(ok) is None
    assert measure.rejection_class(dict(ok, error_code="BadRequestError", error_class="bad_request_other")) == \
        "error_class bad_request_other"
    assert measure.rejection_class(dict(ok, status=401, error_code="bad_token", error_class="auth")) == "error_class auth"
    assert measure.rejection_class(dict(ok, status=503, error_code="http_503", error_class="server_error")) == \
        "error_class server_error"
    assert measure.rejection_class(dict(ok, status=None, error_code="ledger_bad_json", error_class="ledger_blocked")) == \
        "error_class ledger_blocked"
    assert measure.rejection_class(dict(ok, status=None, error_code="ledger_transport_error", error_class="transport")) == \
        "error_class transport"
    assert measure.rejection_class(dict(ok, status=502, error_class="context_length")) == \
        "error_class context_length with status 502"
    assert "no error_class" in measure.rejection_class({k: v for k, v in ok.items() if k != "error_class"})
    assert "no error_class" in measure.rejection_class(dict(ok, error_class=None))
    assert measure.CONTEXT_LENGTH_REJECTION_CLASS == C.LEDGER_ERROR_CLASS_CONTEXT_LENGTH == "context_length"
    assert not hasattr(measure, "LEDGER_OWN_ERROR_PREFIX")        # the class replaced the prefix heuristic


def test_measure_keeps_capless_rows_out_of_the_clamp_basis_and_files_grader_traffic_separately(
        measure, tmp_path):
    """Evals' geometry verdict (2026-09-04), the gen_long finding in miniature.
    100 CAPPED gen_long calls (cap 256): 92 with prompt 300, 8 with prompt
    1,100 — at the total-p99 window 1,280 the 8 long ones clamp (8% > the 5%
    ceiling -> second branch, L = round-up(1,100 + 256) = 1,536); 200 CAP-LESS
    target calls (prompt 300) plus one cap-less call with a 5,000-token prompt;
    and 200 cap-less calls on the GRADER ROUTE. Pooled the way v1 did, the
    cap-less rows would have diluted the clamp to 8 / 501 = 1.6% and kept the
    first branch, and the 5,000-token cap-less prompt would have COUNTED as a
    clamp (prompt + 0 > L). Now: (1) a cap-less row never enters a clamp count
    — numerator or denominator — while it stays in the prompt / total p99
    pools; (2) rows on the grader route (the manifests' grader_virtual_model)
    pool into capability:grader, informational, with their own p99; (3) a
    grader-route row on a task with no model grader refuses; (4) a ledger whose
    target shares the grader's route (a none cell) refuses; (5) a ledger whose
    manifests record no route refuses without --grader-vmodel."""
    import json as _json

    from sisteer.capeval.tinycap import contracts as C
    G = "petri__base"
    capped = ([_ledger_row("inspect/gsm8k", 300, completion=50, cap=256, sample=f"s{i}/0") for i in range(92)]
              + [_ledger_row("inspect/gsm8k", 1100, completion=50, cap=256, sample=f"l{i}/0") for i in range(8)])
    # cap-less calls on the TARGET route, same key as the capped ones (gsm8k
    # here: MATH-500 rows pool into their own key)
    capless = [_ledger_row("inspect/gsm8k", 300, completion=2, cap=None, sample=f"c{i}/0") for i in range(200)]
    capless.append(_ledger_row("inspect/gsm8k", 5000, completion=2, cap=None, sample="huge/0"))
    # grader-route rows carry a MODEL-GRADED task (xstest): MATH-500 is judge-free
    grader = [dict(_ledger_row("inspect/xstest", 340, completion=2, cap=None, sample=f"g{i}/0"), vmodel=G)
              for i in range(200)]
    led = tmp_path / "prompted" / "usage_ledger.jsonl"
    _write_ledger(led, capped + capless + grader)
    (led.parent / "cells" / "k").mkdir(parents=True)
    manifest = led.parent / "cells" / "k" / "manifest.json"
    manifest.write_text(_json.dumps({"cell_id": "prompted__safety_refusal",
                                     "route": {"virtual_model": "petri__prompted__safety",
                                               "grader_virtual_model": G, "tool_calls": True}}))
    # the route is read from the manifests beside the ledger, kind included
    route = measure.read_manifest_route(led)
    assert (route.kind, route.target_vmodel, route.grader_vmodel) == ("prompted", "petri__prompted__safety", G)
    assert measure.ledger_spec(str(led)) == (led, "prompted", G, "safety_refusal")   # + the cell's SI
    assert measure.ledger_spec(f"{led}:fitted") == (led, "fitted", G, None)      # the suffix overrides the kind only (no SI)

    res = measure.measure([(led, "prompted", G)], measured_on="2026-09-04")
    by_key = {p["key"]: p for p in res["proposals"]}
    assert set(by_key) == {"capability:gen_long", "capability:grader"} and res["refused_keys"] == []
    assert res["inputs"][0]["grader_vmodel"] == G and res["inputs"][0]["n_grader_rows"] == 200
    assert "capability:grader" not in res["absent_keys"]          # never a served sub-entry
    # (1) gen_long: 301 calls, 201 cap-less; the clamp basis is the 100 capped calls
    g = by_key["capability:gen_long"]
    assert (g["n_calls"], g["n_capped"], g["n_capless"]) == (301, 100, 201)
    assert g["p99_total"] == 1150 and g["L_at_total_sizing"] == 1280 and g["max_total"] == 5002
    assert g["clamp_rate_at_total_sizing"] == pytest.approx(8 / 100)          # NOT 8 / 301, NOT 9 / 301
    assert (g["L"], g["S"], g["sized_by"]) == (1536, 256, sg.SIZED_BY_PROMPT_PLUS_MAX)
    assert g["p99_prompt"] == 1100 and g["caps"] == [256] and g["requested_max_tokens"] == 256
    assert g["clamp_rate_at_L"] == 0.0 and g["frac_over_L"] == pytest.approx(1 / 301)   # the 5,002 total
    e = g["_entry"]
    assert e.basis.n_capless == 201 and e.basis.n_capped == 100 and not e.informational
    assert "201 of them CAP-LESS" in e.basis.note and "outside the clamp basis of 100 capped calls" in e.basis.note
    assert "8.0% of the capped calls" in e.basis.note
    # the negative, on the pool itself: a cap-less row contributes to NO clamp count at ANY window
    pools, _ = measure.pool_rows([(r, "prompted", G) for r in capped + capless + grader])
    pool = pools["capability:gen_long"]
    for L in (256, 1280, 1536, 4096, 16384):
        capped_only = sum(1 for p, c in zip(pool.prompt, pool.cap) if c is not None and p + c > L)
        assert pool.clamp_rate_at(L) == capped_only / 100, L
    assert pool.clamp_rate_at(1280) == 0.08 and pool.clamp_rate_at(4096) == 0.0
    assert pool.n_capless == 201 and pool.n_capped == 100 and pool.caps == [256]
    grader_pool = pools["capability:grader"]
    assert grader_pool.n_capless == grader_pool.n_calls == 200 and grader_pool.n_capped == 0
    assert grader_pool.clamp_rate_at(1) == 0.0 and grader_pool.caps == [] and grader_pool.requested_max_tokens == 0
    # (2) the grader key: informational, its own p99, clamp 0 by construction
    gr = by_key["capability:grader"]
    assert gr["informational"] is True and gr["n_calls"] == gr["n_capless"] == 200 and gr["n_capped"] == 0
    assert gr["p99_total"] == 342 and gr["p99_prompt"] == 340 and gr["requested_max_tokens"] == 0 and gr["caps"] == []
    assert (gr["L"], gr["S"], gr["sized_by"]) == (512, 768, sg.SIZED_BY_TOTAL)
    assert gr["clamp_rate_at_L"] == 0.0 and gr["clamp_rate_at_total_sizing"] == 0.0
    ge = gr["_entry"]
    assert ge.informational and not ge.provisional and ge.key == C.GEOMETRY_GRADER_KEY
    assert ge.basis.n_capless == 200 and "INFORMATIONAL" in ge.basis.note and "by construction" in ge.basis.note
    with pytest.raises(sg.GeometryInformational):
        sg.resolve_geometry(ge.key)                       # the table's own grader entry refuses too
    py = measure.render_entry_py(ge)
    assert "task='grader'" in py and "n_capless=200" in py and py.rstrip().endswith("informational=True),")
    pin = measure.render_pinned_row(ge)
    assert pin.startswith("    'capability:grader': dict(L=512, S=768, n=200") and "informational=True" in pin
    assert "n_capless=200" in pin and "max_tokens=0" in pin
    md = measure.render_doc_rows(res["proposals"])
    assert "| capability:grader | 200 | 0 |" in md and "| 200 | p99_total (informational) |" in md
    assert "| capability:gen_long | 301 | 0 |" in md and "| 201 | p99_prompt_plus_max_tokens |" in md
    # (3) grader-route traffic on a task with no model grader is unexplained: refused
    bad = dict(_ledger_row("inspect/gsm8k", 300, completion=2, cap=None), vmodel=G)
    with pytest.raises(measure.MeasureError, match="no model grader"):
        measure.classify(bad, prompted=True, grader_vmodel=G)
    with pytest.raises(measure.MeasureError, match="unexplained traffic"):
        measure.pool_rows([(bad, "prompted", G)])
    assert measure.classify(bad, prompted=True) == "capability:gen_long"       # no route known: by task
    assert measure.classify(grader[0], prompted=True, grader_vmodel=G) == C.GEOMETRY_GRADER_KEY
    # a MATH row not on the grader route pools by task — into MATH-500's own key
    # (capability:gen_long_math in both arms), never gen_long — the prompted
    # cell's SI names the prompt class (short: safety_refusal); ON the grader route it is
    # unexplained traffic (MATH has no model grader)
    math_row = _ledger_row("inspect/math", 340, completion=2, cap=None, sample="m/0")
    assert measure.classify(math_row, prompted=True, grader_vmodel="petri__other",
                            si="safety_refusal") == "capability:gen_long_math"
    with pytest.raises(measure.MeasureError, match="no model grader"):
        measure.classify(dict(math_row, vmodel=G), prompted=True, grader_vmodel=G, si="safety_refusal")
    # (4) a none cell: target and grader share the route -> not separable
    manifest.write_text(_json.dumps({"cell_id": "none", "route": {"virtual_model": G, "grader_virtual_model": G}}))
    with pytest.raises(measure.MeasureError, match="share the route"):
        measure.ledger_spec(str(led))
    # (5) manifests without a route block: refused without --grader-vmodel, taken with it
    manifest.write_text(_json.dumps({"cell_id": "prompted__safety_refusal"}))
    assert measure.read_manifest_route(led) == measure.LedgerRoute("prompted", None, None, "safety_refusal")
    with pytest.raises(measure.MeasureError, match="no route block"):
        measure.ledger_spec(str(led))
    assert measure.ledger_spec(str(led), grader_override=G) == (led, "prompted", G, "safety_refusal")
    # manifests disagreeing on the route refuse
    (led.parent / "cells" / "k2").mkdir()
    (led.parent / "cells" / "k2" / "manifest.json").write_text(_json.dumps(
        {"cell_id": "prompted__safety_refusal", "route": {"virtual_model": "x", "grader_virtual_model": "y"}}))
    with pytest.raises(measure.MeasureError, match="disagree on the route"):
        measure.read_manifest_route(led)
    (led.parent / "cells" / "k2" / "manifest.json").unlink()
    # the CLI end to end, route from the manifests: grader literal emitted, informational
    manifest.write_text(_json.dumps({"cell_id": "prompted__safety_refusal",
                                     "route": {"virtual_model": "petri__prompted__safety",
                                               "grader_virtual_model": G}}))
    out = tmp_path / "proposal"
    assert measure.main(["--ledger", str(led), "--out", str(out), "--measured-on", "2026-09-04"]) == 0
    ent = (out / "geometry_entries.py.txt").read_text()
    assert "task='grader'" in ent and "informational=True" in ent and "task='gen_long'" in ent
    prop = json.loads((out / "geometry_proposal.json").read_text())
    assert prop["inputs"][0]["n_grader_rows"] == 200 and "cap-less" in prop["rule"]["clamp_basis"]


def test_grader_key_and_model_graded_tasks_are_contract_names(measure):
    from sisteer.capeval.tinycap import contracts as C
    assert measure.GRADER_KEY == C.GEOMETRY_GRADER_KEY == "capability:grader"
    # MATH-500 is scored by SymPy alone — no grader leg exists, so
    # `inspect/math` is not model-graded
    assert measure.GRADER_TASKS == C.MODEL_GRADED_INSPECT_TASKS == (
        "inspect/xstest", "inspect/or_bench_hard_1k")
    assert C.inspect_task("math") not in C.MODEL_GRADED_INSPECT_TASKS
    assert set(C.MODEL_GRADED_INSPECT_TASKS) <= set(C.INSPECT_TASKS)
    assert C.GEOMETRY_GRADER_KEY not in C.GEOMETRY_SUBENTRY_KEYS and C.GEOMETRY_GRADER_KEY in sg.TABLE
    assert "capability:grader" in measure.SOURCE and "cap-less" in measure.SOURCE


def test_math500_rows_pool_into_their_own_key_in_both_arms(measure):
    """The measurer reads the SAME class map the
    launcher resolves geometry from, so a MATH-500 ledger row pools into
    ``capability:gen_long_math`` whatever its arm or cell kind — the entry it is
    served under — and never into either gen_long entry, whose bases it would
    otherwise distort with 16,384-token requests. Every other gen_long task is
    unchanged."""
    from sisteer.capeval.tinycap import contracts as C
    math = C.inspect_task("math")
    for thinking in (False, True):
        row = _ledger_row(math, 560, completion=100, cap=C.MATH500_MAX_TOKENS_DEFAULT, thinking=thinking)
        # a fitted ledger (no system turn) and the safety-SI ledger: the short class's entry
        assert measure.classify(row, prompted=False) == "capability:gen_long_math"
        assert measure.classify(row, prompted=True, si="safety_refusal") == "capability:gen_long_math"
        # a long-SI ledger's MATH rows pool into the LONG class's entry — never the 1,041 one
        for si in ("si_constitution", "si_irrelevant"):
            assert measure.classify(row, prompted=True, si=si) == "capability:gen_long_math_longsi"
        # a prompted ledger whose SI is unknown to the measurer refuses rather than guessing a class
        with pytest.raises(measure.MeasureError, match="pools into no measured sub-entry"):
            measure.classify(row, prompted=True)
        with pytest.raises(measure.MeasureError, match="NO measured prompt class"):
            measure.classify(row, prompted=True, si="si_hhh")
        # the SI is a prompted ledger's fact: a fitted row ignores it
        assert measure.classify(row, prompted=False, si="si_irrelevant") == "capability:gen_long_math"
    # pool_rows carries the SI as the tuple's fourth element (the CLI's ledger_spec builds it)
    pools, _ = measure.pool_rows([(_ledger_row(math, 7000, completion=100, cap=16384), "prompted", "g", "si_irrelevant"),
                                  (_ledger_row(math, 500, completion=100, cap=16384), "prompted", "g", "safety_refusal"),
                                  (_ledger_row(math, 500, completion=100, cap=16384), "fitted", "g")])
    assert {k: len(p.prompt) for k, p in pools.items()} == {"capability:gen_long_math_longsi": 1, "capability:gen_long_math": 2}
    assert measure.classify(_ledger_row(C.GSM8K_STANDARD_TASK, 1, thinking=True), prompted=False) == "capability:gen_long_thinking"
    assert measure.classify(_ledger_row(C.GSM8K_STANDARD_TASK, 1), prompted=False) == "capability:gen_long"
    assert measure.classify(_ledger_row("evalplus/humaneval_plus", 1), prompted=True) == "capability:gen_long"
    assert measure.classify(_ledger_row("inspect/or_bench_hard_1k", 1), prompted=False) == "capability:gen_long"
    assert measure.classify(_ledger_row("inspect/xstest", 1), prompted=False) == "capability:gen_long"
    # a MATH row on the GRADER route is unexplained traffic: MATH-500 is judge-free
    # (a pre-change MATH ledger with grader calls refuses
    # by name rather than pooling); a model-graded task's grader row still keys
    # to the informational grader entry
    grader = dict(_ledger_row(math, 340, completion=2, cap=None), vmodel="petri__base")
    with pytest.raises(measure.MeasureError, match="no model grader"):
        measure.classify(grader, prompted=False, grader_vmodel="petri__base")
    xstest = dict(_ledger_row(C.inspect_task("xstest"), 340, completion=2, cap=None), vmodel="petri__base")
    assert measure.classify(xstest, prompted=False, grader_vmodel="petri__base") == C.GEOMETRY_GRADER_KEY


def test_both_gsm8k_row_tasks_pool_into_gen_long(measure):
    """PW3 / U4(a): the standard paired GSM8K (``inspect/gsm8k_paired200``,
    control ``gsm8k``) and the in-depth full set (``inspect/gsm8k``, ``gsm8k_full``)
    are both gen_long traffic — the thinking twin under thinking-ON."""
    from sisteer.capeval.tinycap import contracts as C
    assert measure.classify(_ledger_row(C.GSM8K_STANDARD_TASK, 1), prompted=False) == "capability:gen_long"
    assert measure.classify(_ledger_row(C.GSM8K_FULL_TASK, 1), prompted=False) == "capability:gen_long"
    assert measure.classify(_ledger_row(C.GSM8K_STANDARD_TASK, 1, thinking=True), prompted=True) == \
        "capability:gen_long_thinking"
    assert measure.classify(_ledger_row("gsm8k", 1), prompted=False) is None   # a control name is not a row task


def test_sizing_rule_branches_and_headroom():
    # total-p99 branch: L within [floor, floor + headroom]
    for key in ("multiturn_attacks", "capability", "bon", "families_prompted",
                "multiturn_attacks:crescendo", "multiturn_attacks:actorattack"):
        g = sg.TABLE[key]
        floor = sg.next_block(g.basis.p99_total)
        assert floor <= g.max_model_len <= floor + sg.SIZING_HEADROOM, key
        assert g.basis.floor == floor
    # prompt+max_tokens branch: forced by a >5% clamp at total sizing
    for key, rate in (("multiturn_attacks:pair", 1.0), ("families_served", 0.20),
                      ("families_served:short", 1.0)):
        g = sg.TABLE[key]
        assert g.basis.clamp_rate_at_total_sizing == rate > sg.CLAMP_RATE_CEILING
        floor = sg.next_block(g.basis.p99_prompt + g.basis.requested_max_tokens)
        assert floor <= g.max_model_len <= floor + sg.SIZING_HEADROOM, key
        assert g.basis.floor == floor
    assert sg.next_block(1343 + 2048) == 3392 and sg.TABLE["multiturn_attacks:pair"].max_model_len == 3456
    assert sg.next_block(7352 + 4096) == 11456 and sg.TABLE["families_served"].max_model_len == 11520
    # the certified Petri window is BELOW its p99 by design and says so
    petri = sg.TABLE["petri"]
    assert petri.certified and petri.max_model_len < petri.basis.p99_total
    assert sg.sizing_problem(petri) is None
    assert sg.CLAMP_RATE_CEILING == 0.05 and sg.SIZING_HEADROOM == 512


def test_sizing_rule_refuses_a_clamp_that_fires_on_most_calls():
    """The negative test: total-p99 sizing of an arm that runs to its cap —
    clamp rate 1.0 at that L — is refused; the rule sends it to the
    prompt + max_tokens branch, which is accepted."""
    with pytest.raises(ValueError, match="clamp rate 1.0000 .* exceeds the 5% ceiling"):
        sg.ServedGeometry("e", sg.next_block(1864), sg.identity_seqs(sg.next_block(1864)),
                          _basis(p99_total=1864, p99_prompt=1343, requested_max_tokens=2048,
                                 clamp_rate_at_L=1.0))
    # the same arm sized by the rule's second branch passes
    ok = sg.ServedGeometry("e", 3456, sg.identity_seqs(3456),
                           _basis(p99_total=1864, p99_prompt=1343, requested_max_tokens=2048,
                                  clamp_rate_at_L=0.0082, clamp_rate_at_total_sizing=1.0))
    assert ok.sized_by == sg.SIZED_BY_PROMPT_PLUS_MAX
    # 6% is over the ceiling too; 5% exactly is not
    with pytest.raises(ValueError, match="exceeds the 5% ceiling"):
        sg.ServedGeometry("e", 2000, sg.identity_seqs(2000), _basis(clamp_rate_at_L=0.06))
    assert sg.ServedGeometry("e", 2000, sg.identity_seqs(2000), _basis(clamp_rate_at_L=0.05))


def test_entry_construction_refuses_other_violations():
    with pytest.raises(ValueError, match="block size"):
        sg.ServedGeometry("e", 2004, 196, _basis())
    with pytest.raises(ValueError, match="identity_seqs"):
        sg.ServedGeometry("e", 2000, 24, _basis())                 # no override reason
    with pytest.raises(ValueError, match="below the sizing floor"):
        sg.ServedGeometry("e", 1984, sg.identity_seqs(1984), _basis(p99_total=2000))
    with pytest.raises(ValueError, match="headroom"):
        sg.ServedGeometry("e", 2000 + 1024, sg.identity_seqs(3024), _basis(p99_total=2000))
    with pytest.raises(ValueError, match="unmeasured"):
        sg.ServedGeometry("e", 2000, sg.identity_seqs(2000), _basis(clamp_rate_at_L=None))
    # an informational entry may leave the clamp rate unmeasured
    assert sg.ServedGeometry("e", 2000, sg.identity_seqs(2000), _basis(clamp_rate_at_L=None),
                             requires_recertification=True)
    # a certified window may sit below its p99 — but not above the clamp ceiling
    assert sg.ServedGeometry("e", 1024, sg.identity_seqs(1024), _basis(p99_total=2000),
                             certified=True)
    with pytest.raises(ValueError, match="exceeds the 5% ceiling"):
        sg.ServedGeometry("e", 1024, sg.identity_seqs(1024),
                          _basis(p99_total=2000, clamp_rate_at_L=0.3), certified=True)
    # the conservative branch needs p99_prompt
    with pytest.raises(ValueError, match="cannot size"):
        sg.ServedGeometry("e", 2000, sg.identity_seqs(2000),
                          _basis(p99_total=None, p99_prompt=None))


def test_petri_entry_equals_composer_defaults_and_smoke_pins():
    g = sg.geometry_for("petri")
    params = inspect.signature(vs.build_server_launch).parameters
    assert g.max_model_len == params["max_model_len"].default == 16384
    assert g.max_num_seqs == vs.DEFAULT_MAX_NUM_SEQS == 24
    assert g.is_identity and not g.over_budget and g.certified
    assert g.batched_tokens == sg.BATCHED_TOKEN_BUDGET
    smoke = _load("_petri_steered_smoke_for_geometry", "scripts/petri_steered_smoke.py")
    assert g.max_model_len == smoke.PETRI_MAX_MODEL_LEN
    assert g.max_num_seqs == smoke.PETRI_MAX_NUM_SEQS
    assert sg.BATCHED_TOKEN_BUDGET == smoke.PETRI_MAX_NUM_BATCHED_TOKENS
    # the measured basis replaced the floor-rationale placeholder: every
    # per-model p99, the 284 rejections, max_tokens 1024
    n = g.basis.note
    for s in ("11,575", "15,443", "15,758", "17,395", "18,385", "284", "max_tokens=1024",
              "gemma4e4b 113", "qwen35_9b 71", "qwen35_4b 62", "gemma4e2b 37", "llama8b 1"):
        assert s in n, s
    assert g.basis.requested_max_tokens == 1024
    # the per-model sub-entries are informational and say why
    for model in ("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_4b", "qwen35_9b"):
        sub = sg.geometry_for("petri", task=model)
        assert sub.requires_recertification and sub.is_identity
        assert "petri-gate16k" in sub.basis.note and "test_petri_geometry_cert" in sub.basis.note
        assert "UP, not down" in sub.basis.note
    # and the composer prose no longer claims the target turns carry no max_tokens
    src = (REPO / "src/sisteer/capeval/serving/vllm_server.py").read_text()
    assert "carry no max_tokens" not in src
    assert "max_tokens=1024" in src


def test_sub_entries_resolve_by_key_and_by_task_and_informational_ones_refuse():
    assert sg.parse_key("capability", "short") == "capability:short"
    assert sg.parse_key("capability:short") == "capability:short"
    assert sg.parse_key("capability") == "capability"
    with pytest.raises(sg.UnknownEvalGeometry):
        sg.parse_key("capability:short", task="long")
    assert sg.geometry_for("multiturn_attacks:crescendo") is sg.geometry_for("multiturn_attacks", task="crescendo")
    r = sg.resolve_geometry("multiturn_attacks:crescendo")
    assert (r.max_model_len, r.max_num_seqs) == (9984, 39) and r.eval == "multiturn_attacks:crescendo"
    r = sg.resolve_geometry("capability:short")
    assert (r.max_model_len, r.max_num_seqs) == (1536, 256)
    for model in ("llama8b", "gemma4e4b", "gemma4e2b", "qwen35_4b", "qwen35_9b"):
        assert sg.geometry_for(f"petri:{model}").requires_recertification
        with pytest.raises(sg.GeometryRequiresRecertification, match="petri-gate16k"):
            sg.resolve_geometry(f"petri:{model}")
    assert sg.geometry_for("capability:grader").informational
    with pytest.raises(sg.GeometryInformational, match="rides the target's server"):
        sg.resolve_geometry("capability:grader")
    with pytest.raises(sg.UnknownEvalGeometry) as ei:
        sg.geometry_for("mystery")
    assert isinstance(ei.value, KeyError)
    for name in sg.KNOWN_EVALS:
        assert name in str(ei.value)
    with pytest.raises(sg.UnknownEvalGeometry):
        sg.geometry_for("capability:nope")


def test_describe_renders_every_entry_as_markdown():
    md = sg.describe()
    lines = md.splitlines()
    assert lines[0].startswith("| key | L | S | SxL | identity | sized_by |")
    assert set(lines[1]) <= set("|-")
    for key in sg.KNOWN_KEYS:
        assert any(l.startswith(f"| {key} |") for l in lines), key
    assert "393,216" in md and "| 8704 | 45 |" in md and "| 11520 | 34 |" in md
    assert "requires_recertification" in md and "certified" in md
    assert "informational" in md and "provisional,analytic" in md
    assert "OVER" not in md.split("budget =")[0]          # nothing is over budget any more
    assert sg.MEASUREMENT_DOC in md


def test_measurement_doc_is_in_repo_and_path_free():
    doc = (REPO / sg.MEASUREMENT_DOC).read_text()
    # (patterns composed so this file itself never spells out a path prefix)
    for bad in ("/" + "home/", "/" + "workspace", "/" + "vol/"):
        assert bad not in doc, bad
    # the condensed rows (eval x arm x family), each section present
    for head in ("## 1. Multi-turn", "## 2. Capability", "## 3. Single-shot", "## 4. Best-of-N",
                 "## 5. Petri"):
        assert head in doc, head
    n_rows = sum(1 for l in doc.splitlines()
                 if l.startswith("| ") and not l.startswith(("| task", "| arm", "| model", "|---")))
    assert n_rows >= 110
    # the corrected premises live here too
    assert "5,872 / 6,032 / 5,963" in doc and "6,380 / 6,565 / 6,683" in doc
    assert "max_tokens` = 1024" in doc and "284" in doc


# --------------------------------------------------------------------------- #
# (c) resolution                                                               #
# --------------------------------------------------------------------------- #


def test_resolve_table_default_then_override_then_budget_guard():
    r = sg.resolve_geometry("multiturn_attacks")
    assert (r.max_model_len, r.max_num_seqs) == (8704, 45)
    assert r.window_source == r.seqs_source == "table" and r.is_identity
    assert not r.overridden and not r.over_budget
    r = sg.resolve_geometry("multiturn_attacks", max_num_seqs=24)
    assert (r.max_model_len, r.max_num_seqs) == (8704, 24)
    assert r.seqs_source == "override" and r.overridden and not r.is_identity
    r = sg.resolve_geometry("multiturn_attacks", max_model_len=16384, max_num_seqs=24)
    assert r.batched_tokens == sg.BATCHED_TOKEN_BUDGET and not r.over_budget
    with pytest.raises(sg.GeometryOverBudget, match="allow_over_budget"):
        sg.resolve_geometry("multiturn_attacks", max_model_len=16384)   # 45 x 16384
    r = sg.resolve_geometry("multiturn_attacks", max_model_len=16384, allow_over_budget=True)
    assert r.over_budget and "OVER BUDGET" in r.describe()


def test_resolve_strict_identity_guard():
    # default: only the budget is guarded (grokmt / capctrl keep their argv)
    r = sg.resolve_geometry("multiturn_attacks", max_num_seqs=24)
    assert not r.is_identity and not r.over_budget
    # strict: a non-identity override refuses unless allowed
    with pytest.raises(sg.GeometryIdentityBroken, match="identity_seqs"):
        sg.resolve_geometry("multiturn_attacks", max_num_seqs=24, strict_identity=True)
    r = sg.resolve_geometry("multiturn_attacks", max_num_seqs=24, strict_identity=True,
                            allow_over_budget=True)
    assert not r.is_identity
    # an identity pair passes strict; the table default passes strict
    assert sg.resolve_geometry("multiturn_attacks", max_model_len=16384, max_num_seqs=24,
                               strict_identity=True).is_identity
    assert sg.resolve_geometry("capability", strict_identity=True).is_identity
    # over budget is refused as over budget first
    with pytest.raises(sg.GeometryOverBudget):
        sg.resolve_geometry("capability", max_model_len=16384, strict_identity=True)
    # the legacy fallback is labelled as such
    r = sg.resolve_geometry(None, fallback=(10240, 64))
    assert r.legacy and "LEGACY OVER BUDGET" in r.describe()
    r = sg.resolve_geometry("families_served")
    assert not r.legacy


def test_resolve_fallback_and_refusals():
    r = sg.resolve_geometry(None, fallback=(10240, 64))
    assert (r.max_model_len, r.max_num_seqs) == (10240, 64)
    assert r.window_source == r.seqs_source == "fallback" and r.entry is None
    assert r.over_budget                       # the legacy default, flagged not refused
    r = sg.resolve_geometry(None, max_model_len=2048, max_num_seqs=8)
    assert r.window_source == r.seqs_source == "override"
    with pytest.raises(ValueError, match="nothing is guessed"):
        sg.resolve_geometry(None, max_model_len=2048)
    with pytest.raises(sg.UnknownEvalGeometry):
        sg.resolve_geometry("nope")
    with pytest.raises(ValueError):
        sg.resolve_geometry("petri", max_num_seqs=0)


def test_describe_of_a_resolution_names_geometry_and_basis():
    d = sg.resolve_geometry("multiturn_attacks").describe()
    assert "max_model_len=8704 (table)" in d and "max_num_seqs=45 (table)" in d
    assert "identity" in d and "391,680 of the 393,216" in d
    assert "p99_total=8487" in d and "n_calls=4598" in d and "clamp_rate_at_L=0.0137" in d
    d = sg.resolve_geometry("families_served").describe()
    assert "(p99_prompt_plus_max_tokens)" in d and "requested_max_tokens=4096" in d
    d = sg.resolve_geometry("petri").describe()
    assert "certified window" in d


# --------------------------------------------------------------------------- #
# (b) source grep: no launcher hands the composer a literal Petri window       #
# --------------------------------------------------------------------------- #

_LITERAL_WINDOW = re.compile(
    r'''--max-model-len["']\s*,\s*["']16384["']'''      # argv literal to the standup
    r"|max_model_len\s*=\s*16384\b"                     # composer kwarg literal
    r"|--max-model-len[^\n]*default\s*=\s*16384\b")     # argparse default literal

#: the Petri surfaces that legitimately serve the Petri window — nothing else
_ALLOWLIST = {"scripts/modal/petri_agentloop_control.py"}


def test_no_launcher_passes_a_literal_16384_window():
    files = (sorted((REPO / "scripts" / "modal").glob("*.py"))
             + [REPO / "scripts" / "capctrl_serving_up.py"]
             + sorted((REPO / "scripts").glob("petri_families_*.py")))
    offenders = []
    for p in files:
        rel = str(p.relative_to(REPO))
        if rel in _ALLOWLIST:
            continue
        if _LITERAL_WINDOW.search(p.read_text()):
            offenders.append(rel)
    assert not offenders, (
        "launchers passing a literal 16384 window to the composer/standup — "
        "resolve it from sisteer.capeval.serving.geometry instead: "
        f"{offenders}")
    # and the allowlisted Petri render still does (so the regex is live)
    assert _LITERAL_WINDOW.search(
        (REPO / "scripts/modal/petri_agentloop_control.py").read_text())


# --------------------------------------------------------------------------- #
# (c) the launchers resolve from the table                                     #
# --------------------------------------------------------------------------- #


def _modal_stub():
    """Install THIS suite's modal stub for the duration of a script import
    and hand back whatever was there. `setdefault` used to keep a FOREIGN
    stub left by an earlier module in the same suite run (the sdfbase fits
    tests install theirs at import), whose ``Secret`` lacks ``from_dict`` —
    grokmt_served.py builds its OpenRouter secret at module level, so the
    import ERRORed at fixture setup on the joined tree (sdfbase-mt, 2026-09-04).
    Always our own stub, restored afterwards (the pattern test_twosi_fits_app
    adopted for the same reason)."""
    from test_grokmt_served_resume import _stub_modal
    prev = sys.modules.get("modal")
    sys.modules["modal"] = _stub_modal()
    return prev


def _restore_modal(prev):
    if prev is None:
        sys.modules.pop("modal", None)
    else:
        sys.modules["modal"] = prev


@pytest.fixture(scope="module")
def grokmt():
    prev = _modal_stub()
    try:
        yield _load("_grokmt_served_geometry", "scripts/modal/grokmt_served.py")
    finally:
        _restore_modal(prev)


@pytest.fixture(scope="module")
def capctrl():
    prev = _modal_stub()
    try:
        yield _load("_capctrl_served_geometry", "scripts/modal/capctrl_served.py")
    finally:
        _restore_modal(prev)


def test_grokmt_target_defaults_come_from_the_table(grokmt, tmp_path):
    assert grokmt.TARGET_EVAL == "multiturn_attacks" and grokmt.TARGET_KEY == "gemma4e4b"
    geom = grokmt.resolve_target_geometry(None)
    # the table window; S from the TARGET MODEL's measured budget (589,824 // 8704)
    assert (geom.max_model_len, geom.max_num_seqs) == (8704, 67)
    assert geom.seqs_source == "model_budget" and geom.budget_tokens == 589_824
    argv = grokmt._target_standup_cmd("/tree", tmp_path, tmp_path / "STOP", geom)
    # derived factors travel as --eval / --model-key, never as explicit dims (review S3)
    assert "--max-model-len" not in argv and "--max-num-seqs" not in argv
    assert argv[argv.index("--eval") + 1] == "multiturn_attacks"
    assert argv[argv.index("--model-key") + 1] == "gemma4e4b"
    over = grokmt.resolve_target_geometry(None, target_window=16384, target_seqs=24)
    argv = grokmt._target_standup_cmd("/tree", tmp_path, tmp_path / "STOP", over)
    assert argv[argv.index("--max-model-len") + 1] == "16384"
    assert argv[argv.index("--max-num-seqs") + 1] == "24"
    # the driver learns the launched window
    cmd = grokmt._driver_cmd("/tree", attack="pair", cell="none", arm="qwen", sub=0,
                             subshards=3, ids_path="ids", private_root="pr",
                             proxy_url="http://127.0.0.1:1/v1", rt="rt.json",
                             attacker_port=1234, target_window=geom.max_model_len)
    assert cmd[cmd.index("--target-max-model-len") + 1] == "8704"
    # the run_smoke / main signatures: 0 = unset (Modal has no None)
    for fn in (grokmt.run_smoke, grokmt.main):
        params = inspect.signature(fn).parameters
        assert params["target_seqs"].default == 0
        assert params["target_window"].default == 0
        assert params["allow_over_budget"].default is False


def test_grokmt_overrides_win_and_over_budget_refuses(grokmt):
    g = grokmt.resolve_target_geometry(None, target_seqs=24)
    assert (g.max_model_len, g.max_num_seqs) == (8704, 24) and g.seqs_source == "override"
    g = grokmt.resolve_target_geometry(None, target_window=16384, target_seqs=24)
    assert (g.max_model_len, g.max_num_seqs) == (16384, 24) and not g.over_budget
    # a window override alone re-sizes S from the model's budget (589,824 // 16384)
    g = grokmt.resolve_target_geometry(None, target_window=16384)
    assert g.max_num_seqs == 36 and not g.over_budget and g.seqs_source == "model_budget"
    # the guard is held to the MODEL's budget: 60 x 16384 = 983,040 > 589,824
    with pytest.raises(sg.GeometryOverBudget) as ei:
        grokmt.resolve_target_geometry(None, target_window=16384, target_seqs=60)
    assert "measured gemma4e4b" in str(ei.value)
    g = grokmt.resolve_target_geometry(None, target_window=16384, target_seqs=60,
                                       allow_over_budget=True)
    assert g.over_budget and g.max_num_seqs == 60


def test_capctrl_cells_default_from_the_table_and_rows_may_override(capctrl, tmp_path):
    assert capctrl.CAPABILITY_EVAL == "capability"
    for model, row in capctrl.MODELS.items():
        assert "seqs" not in row, (model, "the lane default lives in the table now")
        g = capctrl.cell_geometry(row)
        assert (g.max_model_len, g.max_num_seqs) == (7424, 52), model    # the measured entry
        assert g.window_source == g.seqs_source == "table"
    g = capctrl.cell_geometry({"seqs": 16})
    assert g.max_num_seqs == 16 and g.seqs_source == "override"
    with pytest.raises(sg.GeometryOverBudget):
        capctrl.cell_geometry({"seqs": 64})                             # 64 x 7424 = 475,136
    g = capctrl.cell_geometry({"seqs": 64, "allow_over_budget": True})
    assert g.over_budget
    # a row that wants Petri's window back must also narrow S: 52 x 16384 is
    # 851,968 tokens, refused; 24 x 16384 is the certified footprint
    with pytest.raises(sg.GeometryOverBudget):
        capctrl.cell_geometry({"window": 16384})
    g = capctrl.cell_geometry({"window": 16384, "seqs": 24})
    assert (g.max_model_len, g.max_num_seqs) == (16384, 24) and not g.over_budget
    # standup argv carries the resolved geometry; the driver fan-out follows S
    geom = capctrl.cell_geometry(capctrl.MODELS["llama8b"])
    argv = capctrl._standup_cmd("/tree", "cfg.yaml", "cell", "ddir", tmp_path,
                                tmp_path / "STOP", geom, 64)
    # a table-derived geometry travels as --eval only (review S3); an override as dims
    assert "--max-model-len" not in argv and "--max-num-seqs" not in argv
    assert argv[argv.index("--eval") + 1] == "capability"
    geom = capctrl.cell_geometry({"window": 16384, "seqs": 24})
    argv = capctrl._standup_cmd("/tree", "cfg.yaml", "cell", "ddir", tmp_path,
                                tmp_path / "STOP", geom, 64)
    assert argv[argv.index("--max-model-len") + 1] == "16384"
    assert argv[argv.index("--max-num-seqs") + 1] == "24"
    src = (REPO / "scripts/modal/capctrl_served.py").read_text()
    assert '"--concurrency", str(geom.max_num_seqs)' in src


def test_standup_eval_sets_defaults_flags_win_legacy_otherwise():
    up = _load("_capctrl_serving_up_geometry", "scripts/capctrl_serving_up.py")
    r = up.resolve_standup_geometry(None, None, None)
    assert (r.max_model_len, r.max_num_seqs) == (up.LEGACY_MAX_MODEL_LEN, up.LEGACY_MAX_NUM_SEQS) == (10240, 64)
    assert r.window_source == "fallback"
    r = up.resolve_standup_geometry("multiturn_attacks", None, None)
    assert (r.max_model_len, r.max_num_seqs) == (8704, 45)
    # the eval[:task] syntax reaches the standup unchanged
    r = up.resolve_standup_geometry("multiturn_attacks:crescendo", None, None)
    assert (r.max_model_len, r.max_num_seqs) == (9984, 39)
    r = up.resolve_standup_geometry("capability:short", None, None)
    assert (r.max_model_len, r.max_num_seqs) == (1536, 256)
    # explicit flags win — as an identity pair (a lone --max-model-len 12288
    # would leave S=52 = 638,976 tokens, which F4 refuses below)
    r = up.resolve_standup_geometry("capability", 12288, 32)
    assert (r.max_model_len, r.max_num_seqs) == (12288, 32) and r.is_identity
    assert r.window_source == r.seqs_source == "override"
    with pytest.raises(sg.GeometryOverBudget):
        up.resolve_standup_geometry("capability", 12288, None)
    # explicit flags with no eval: exactly the pre-table behaviour
    r = up.resolve_standup_geometry(None, 16384, 24)
    assert (r.max_model_len, r.max_num_seqs) == (16384, 24)
    # F4: ONLY the legacy no---eval path accepts an over-budget product, and
    # says LEGACY OVER BUDGET; with --eval an override that breaks the budget
    # or the identity REFUSES unless --allow-over-budget
    r = up.resolve_standup_geometry(None, None, None)
    assert r.over_budget and r.legacy and "LEGACY OVER BUDGET" in r.describe()
    with pytest.raises(sg.GeometryOverBudget):
        up.resolve_standup_geometry("capability", 16384, None)      # 52 x 16384 = 851,968
    with pytest.raises(sg.GeometryOverBudget):
        up.resolve_standup_geometry("capability", None, 64)
    with pytest.raises(sg.GeometryIdentityBroken):
        up.resolve_standup_geometry("capability", None, 24)         # within budget, not identity
    r = up.resolve_standup_geometry("capability", 16384, None, allow_over_budget=True)
    assert r.over_budget and "OVER BUDGET" in r.describe() and not r.legacy
    r = up.resolve_standup_geometry("capability", None, 24, allow_over_budget=True)
    assert not r.is_identity and not r.over_budget
    r = up.resolve_standup_geometry("capability", 16384, 24)         # an identity pair: fine
    assert r.is_identity and not r.over_budget
    assert "--allow-over-budget" in (REPO / "scripts/capctrl_serving_up.py").read_text()
    # informational entries are refused even here
    with pytest.raises(sg.GeometryRequiresRecertification):
        up.resolve_standup_geometry("petri:qwen35_9b", None, None)
    # and the CLI exposes --eval with None defaults for the two flags
    src = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert 'ap.add_argument("--eval"' in src and "eval[:task]" in src
    assert 'ap.add_argument("--max-model-len", type=int, default=None' in src
    assert 'ap.add_argument("--max-num-seqs", type=int, default=None' in src


def test_families_drivers_default_from_the_table_env_wins(monkeypatch):
    for k in ("WAVE_MAX_MODEL_LEN", "WAVE_MAX_NUM_SEQS", "WAVE_MAX_CONCURRENCY"):
        monkeypatch.delenv(k, raising=False)
    for k in ("WAVE_MODEL", "WAVE_MODEL_KEY"):
        monkeypatch.delenv(k, raising=False)
    fam = _load("_fam_served_geometry", "scripts/petri_families_served_integration.py")
    g = sg.geometry_for("families_served")
    # the table window; S from the wave's MODEL (llama8b: 475,136 // 11520 — the
    # 2026-09-10 lens-stack row; was 19 under the 229,376 ceiling), not the table's 34
    assert fam.MODEL_KEY == "llama8b"
    assert (fam.MAX_MODEL_LEN, fam.MAX_NUM_SEQS) == (g.max_model_len, 41) and g.max_num_seqs == 34
    assert fam.MAX_CONCURRENCY == fam.MAX_NUM_SEQS
    r = fam.resolved_geometry()
    assert r.window_source == "table" and r.seqs_source == "model_budget" and not r.over_budget
    pr = _load("_fam_prompted_geometry", "scripts/petri_families_prompted.py")
    g = sg.geometry_for("families_prompted")
    assert pr.MODEL_KEY == "llama8b"
    assert (pr.MAX_MODEL_LEN, pr.MAX_NUM_SEQS) == (g.max_model_len, 64) and g.max_num_seqs == 52
    assert pr.SERVED_MAX_CONCURRENCY == pr.MAX_NUM_SEQS
    assert pr.resolved_geometry().seqs_source == "model_budget"
    # the env overrides still win — but F4: an override that breaks the budget
    # or the identity REFUSES unless WAVE_ALLOW_OVER_BUDGET=1 is exported too
    # (the old qwen_wave.sbatch literal 16384 x 32 = 524,288 went through here)
    monkeypatch.setenv("WAVE_MAX_MODEL_LEN", "16384")
    monkeypatch.setenv("WAVE_MAX_NUM_SEQS", "32")
    monkeypatch.delenv("WAVE_ALLOW_OVER_BUDGET", raising=False)
    fam2 = _load("_fam_served_geometry_env", "scripts/petri_families_served_integration.py")
    assert (fam2.MAX_MODEL_LEN, fam2.MAX_NUM_SEQS, fam2.MAX_CONCURRENCY) == (16384, 32, 32)
    with pytest.raises(sg.GeometryOverBudget):
        fam2.resolved_geometry()
    pr2 = _load("_fam_prompted_geometry_env", "scripts/petri_families_prompted.py")
    assert (pr2.MAX_MODEL_LEN, pr2.MAX_NUM_SEQS, pr2.SERVED_MAX_CONCURRENCY) == (16384, 32, 32)
    with pytest.raises(sg.GeometryOverBudget):
        pr2.resolved_geometry()
    monkeypatch.setenv("WAVE_ALLOW_OVER_BUDGET", "1")
    r = fam2.resolved_geometry()
    assert r.window_source == r.seqs_source == "override" and r.over_budget
    assert pr2.resolved_geometry().over_budget
    # an identity-breaking (but within-budget) override refuses the same way
    # (8 x 16384 = 131,072, inside llama8b's 475,136 measured budget and not its
    # identity S of 29; 16 was within-budget only under the pre-2026-09-06 458,752)
    monkeypatch.setenv("WAVE_ALLOW_OVER_BUDGET", "0")
    monkeypatch.setenv("WAVE_MAX_NUM_SEQS", "8")
    fam3 = _load("_fam_served_geometry_env2", "scripts/petri_families_served_integration.py")
    with pytest.raises(sg.GeometryIdentityBroken):
        fam3.resolved_geometry()
    # the qwen wave resolves through the table now: no exported literals
    sb = (REPO / "scripts/sbatch/qwen_wave.sbatch").read_text()
    assert "WAVE_MAX_MODEL_LEN=16384" not in sb and "WAVE_MAX_NUM_SEQS=32" not in sb
    assert "WAVE_MAX_CONCURRENCY=64" not in sb and "families_served" in sb
    # the wave threads its launched window into the sweep's served clients
    src = (REPO / "scripts/petri_families_served_integration.py").read_text()
    assert '"--served-max-model-len", str(MAX_MODEL_LEN)' in src


def test_client_widths_derive_from_the_table():
    from sisteer.eval.attacks import bon_served as bs
    assert bs.DEFAULT_MAX_CONCURRENCY == sg.geometry_for("bon").max_num_seqs == 52
    import served_capability_run as srun
    assert srun.CAPABILITY_GEOMETRY is sg.geometry_for("capability")
    cap = {}
    orig = srun.cmd_generate
    srun.cmd_generate = lambda a: cap.update(vars(a))
    try:
        srun.main(["generate", "--proxy-url", "http://x/v1", "--routing-table",
                   "/tmp/rt.json", "--cell", "safety_refusal__psr_kl_all"])
    finally:
        srun.cmd_generate = orig
    assert cap["concurrency"] == sg.geometry_for("capability").max_num_seqs == 52
    # unified_sweep_run: the BoN client default is the table's, and the
    # launched window can be threaded (--served-max-model-len)
    import unified_sweep_run as usr
    src = (REPO / "scripts/unified_sweep_run.py").read_text()
    assert '"--served-max-model-len"' in src
    assert "max_model_len=getattr(args, \"served_max_model_len\", None)" in src
    assert usr is not None


def test_jailbreak_run_window_flag_is_provenance_only_never_a_table_default():
    """The driver does NOT assume a table window when --target-max-model-len is
    omitted: the standup may have been launched at another geometry (the
    un-parameterised standup is 10,240), and the server's stated window
    governs the clamp anyway (served_window). A launcher that knows its
    geometry passes the flag (grokmt does)."""
    import jailbreak_run
    p = argparse.ArgumentParser()
    jailbreak_run._add_target_engine_args(p)
    assert p.parse_args([]).target_max_model_len is None
    assert p.parse_args(["--target-max-model-len", "8704"]).target_max_model_len == 8704
    src = (REPO / "scripts/jailbreak_run.py").read_text()
    assert 'geometry_for("multiturn_attacks")' not in src
    assert "max_model_len=(int(window) if window is not None else None)" in src
    # and the served-target manifest is rewritten AFTER generation -- with the target decode block (2026-09-10) and,
    # since 2026-09-16, the RESOLVED decode (target_decode_resolution: mode, preset / temperature, model key, the pin).
    # This is a pin on the literal call text, wrapped call and all: it breaks on ANY reformatting of the call, semantic
    # or not (it did on 2026-09-16 when the call gained `extra=` and wrapped) -- the known cost of a structural guard,
    # paid rather than loosened. Extending the call again means re-pinning the exact new text here.
    assert ('write_target_manifest(out_dir, defense, stage="final", decode=mt_decode_block,\n'
            '                                  extra={"target_decode_resolution": mt["record"]})') in src


def test_stale_gate_comment_says_24():
    src = (REPO / "scripts/petri_cache_isolation_gate.py").read_text()
    assert "raised 8 -> 32" not in src
    assert "raised 8 -> 24" in src



# --------------------------------------------------------------------------- #
# The 4,096-cap multi-turn windows are EXPLICIT, from measured maxima on complete stores;  #
# the table's multiturn_attacks entries refuse a 4,096 driver                              #
# --------------------------------------------------------------------------- #


def test_multiturn_cap4096_windows_are_the_ruled_values_from_measured_maxima():
    assert sg.MULTITURN_CAP4096_CAP == 4096
    assert sg.MULTITURN_CAP4096_WINDOWS == {"crescendo": 49152, "pair": 16384}
    # each window is a block multiple ABOVE its measured maximum context on a complete store, and the
    # next-smaller power-of-two-ish block would have clamped it (the reason for 49,152 not 32,768)
    assert sg.MULTITURN_CAP4096_MEASURED_MAX_CONTEXT == {"crescendo": 41433, "pair": 12825}
    for atk, L in sg.MULTITURN_CAP4096_WINDOWS.items():
        m = sg.MULTITURN_CAP4096_MEASURED_MAX_CONTEXT[atk]
        assert L % sg.VLLM_BLOCK_SIZE == 0 and L > m
    assert 32768 < sg.MULTITURN_CAP4096_MEASURED_MAX_CONTEXT["crescendo"] < 49152
    assert 8704 < sg.MULTITURN_CAP4096_MEASURED_MAX_CONTEXT["pair"] < 16384
    # ActorAttack is deliberately absent: available through an explicit window, never the default
    assert "actorattack" not in sg.MULTITURN_CAP4096_WINDOWS
    assert "R2-D253" in sg.MULTITURN_CAP4096_SOURCE


def test_multiturn_window_is_the_largest_of_the_jobs_attacks_and_refuses_an_unmeasured_one():
    assert sg.multiturn_window("crescendo") == 49152 and sg.multiturn_window(["pair"]) == 16384
    assert sg.multiturn_window("crescendo,pair") == 49152 and sg.multiturn_window(" pair , crescendo ") == 49152
    with pytest.raises(KeyError, match="actorattack"):
        sg.multiturn_window("crescendo,actorattack")
    with pytest.raises(KeyError, match="measured"):
        sg.multiturn_window(["tap"])
    with pytest.raises(ValueError):
        sg.multiturn_window("")


def test_a_table_window_refuses_a_driver_cap_above_the_cap_it_was_measured_under():
    """THE CAP GUARD: multiturn_attacks (8,704, measured under a 2,048 cap) may not serve a 4,096-cap
    driver from the table -- that is the clamping that was measured (129 distinct max_tokens, 3 killed
    goals). An explicit window passes; a cap at or below the basis passes; no cap stated = as before."""
    e = sg.geometry_for("multiturn_attacks")
    assert e.max_model_len == 8704 and e.basis.requested_max_tokens == 2048
    with pytest.raises(sg.GeometryCapExceedsBasis, match="MEASURED MAXIMUM"):
        sg.resolve_geometry("multiturn_attacks", requested_max_tokens=4096)
    with pytest.raises(sg.GeometryCapExceedsBasis):
        sg.resolve_geometry("multiturn_attacks:pair", requested_max_tokens=4096, model_key="gemma4e4b")
    ok = sg.resolve_geometry("multiturn_attacks", requested_max_tokens=2048)
    assert (ok.max_model_len, ok.window_source) == (8704, "table")
    assert sg.resolve_geometry("multiturn_attacks").max_model_len == 8704            # no cap stated: unchanged
    # the explicit window the instrument passes: resolves, S from the model's budget, identity kept
    r = sg.resolve_geometry("multiturn_attacks", max_model_len=sg.multiturn_window("crescendo"),
                            requested_max_tokens=4096, model_key="gemma4e4b")
    assert (r.max_model_len, r.window_source, r.seqs_source) == (49152, "override", "model_budget")
    assert r.max_num_seqs == sg.BUDGET_BY_MODEL["gemma4e4b"].seqs_at(49152) == 12
    # S is the MODEL's budget at the window (main's table row for qwen35_4b is provisional today; the
    # other trees carry their own measured rows -- 44 at 16,384 on one of them), never a hand-typed number
    r2 = sg.resolve_geometry("multiturn_attacks", max_model_len=16384, requested_max_tokens=4096, model_key="qwen35_4b")
    assert (r2.max_model_len, r2.max_num_seqs) == (16384, sg.BUDGET_BY_MODEL["qwen35_4b"].seqs_at(16384))
    assert r2.max_num_seqs * 16384 <= sg.BUDGET_BY_MODEL["qwen35_4b"].batched_token_budget
    # the standup's resolver threads the cap through
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_capctrl_up_capguard", REPO / "scripts/capctrl_serving_up.py")
    up = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = up            # its dataclasses resolve their module through sys.modules
    spec.loader.exec_module(up)
    with pytest.raises(sg.GeometryCapExceedsBasis):
        up.resolve_standup_geometry("multiturn_attacks", None, None, model_key="gemma4e4b", requested_max_tokens=4096)
    g = up.resolve_standup_geometry("multiturn_attacks", 49152, None, model_key="gemma4e4b", requested_max_tokens=4096)
    assert (g.max_model_len, g.max_num_seqs) == (49152, 12)


def test_gen_long_gsm8k_longsi_entries_hold_the_constitution_prompted_paired200_cells_at_both_rungs():
    """llama-paper-rows, the six-arm design: prompted__si_constitution joins the GSM8K paired-200 matrix.
    The short constitution SI is 5,872 Llama tokens, so the prompted prompt max is 8,007 (measured 2026-09-12 on the pinned Llama
    tokenizer's no-switch render, 200 renders) and NEITHER shared entry holds it (8,007 + 8,192 > 10,752; 8,007 + 16,384 > 18,944,
    and > MATH's longsi 24,064): two entries of its own, the MAX-PROMPT branch — 16,384 at the 8,192 pass (= the certified standup
    geometry) and 24,576 at the rung; S from each model's budget (llama8b 29 / 19); not tinycap sub-entries; pass1 refuses."""
    from sisteer.capeval.tinycap import contracts as C
    for key, L, s_cert, cap, s_llama in (("capability:gen_long_gsm8k_longsi", 16384, 24, 8192, 29),
                                          ("capability:gen_long_gsm8k_rung_longsi", 24576, 16, 16384, 19)):
        g = sg.geometry_for(key)
        assert g.key not in sg.TINYCAP_KEYS and g.key not in sg.informational_keys() and g.eval == "capability"
        assert not g.provisional and not g.informational and not g.certified and not g.requires_recertification
        assert (g.max_model_len, g.max_num_seqs) == (L, s_cert) and g.is_identity
        b = g.basis
        assert b.measured and b.measured_on == GSM8K_LONGSI_MEASURED_ON and b.analytic is None
        assert b.sized_to_max_prompt and b.max_prompt == 8007 and b.p99_prompt == 7975 and b.requested_max_tokens == cap
        assert b.n_calls == 200 and b.n_rejected == 0 and b.n_capless == 0 and b.n_capped == 200 and b.families_measured == ("llama8b",)
        assert b.clamp_rate_at_L == 0.0 and b.frac_over_L == 0.0 and g.sized_by == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX
        assert b.floor == sg.next_block(8007 + cap) and 0 < L - b.floor == 176 <= sg.SIZING_HEADROOM
        assert sg.sizing_problem(g) is None and sg.MEASUREMENT_DOC in b.source and "prompted__si_constitution" in b.source and "21e06f48" in b.source
        r = sg.resolve_geometry(key, model_key="llama8b")
        assert (r.max_model_len, r.max_num_seqs) == (L, s_llama) and sg.BUDGET_BY_MODEL["llama8b"].seqs_at(L) == s_llama and 8007 + cap <= L
        with pytest.raises(ValueError, match="not a tinycap sub-entry"):
            sg.pass1_geometry(key)
        row = [l for l in sg.describe().splitlines() if l.startswith(f"| {key} |")][0]
        assert f"| {L} | {s_cert} |" in row and "| 8007 |" in row and f"| {cap} |" in row and GSM8K_LONGSI_MEASURED_ON in row
    # the shared entries do NOT hold the constitution prompt (why these exist); MATH's longsi does not either
    assert 8007 + 8192 > sg.TABLE["capability:gen_long_thinking"].max_model_len and 8007 + 16384 > sg.TABLE["capability:gen_long_gsm8k_rung"].max_model_len
    assert 8007 + 16384 > sg.TABLE["capability:gen_long_math_longsi"].max_model_len
    assert len(sg.TINYCAP_KEYS) == 8 and sg.TINYCAP_KEYS == C.GEOMETRY_SUBENTRY_KEYS
    # the safety-prompted and bare Llama renders stay in the shared entries (bare max 2,135, safety-prompted 2,223 on the Llama tokenizer)
    assert 2223 + 8192 <= sg.TABLE["capability:gen_long_thinking"].max_model_len and 2223 + 16384 <= sg.TABLE["capability:gen_long_gsm8k_rung"].max_model_len


def test_gen_long_gsm8k_longsi_f22_entries_hold_the_full22_families_constitution_prompted_cells_at_both_rungs():
    """The long-SI class is SCOPED BY FAMILY. Under the four full22 families'
    tokenizers and thinking-ON render the constitution-prompted paired-200 prompt max is 8,338 (gemma4e4b; qwen 8,332, nemotron
    8,288 — measured 2026-09-13 on the grading host, 600 renders; instruction length is a property of the rendered prompt in a
    template), so the Llama-scoped entries (16,384 / 24,576, max 8,007) do NOT hold them: two SEPARATELY SCOPED
    entries, the MAX-PROMPT branch — 16,640 at the 8,192 pass and 24,832 at the rung; S from each model's budget (35 / 43 / 47;
    23 / 29 / 31); not tinycap sub-entries; pass1 refuses; the Llama-scoped entries stay EXACTLY as registered."""
    from sisteer.capeval.tinycap import contracts as C
    per_model = {8192: {"gemma4e4b": 35, "qwen35_4b": 44, "nemotron3nano4b": 47}, 16384: {"gemma4e4b": 23, "qwen35_4b": 29, "nemotron3nano4b": 31}}   # qwen35_4b: 741,888 // 16640 = 44, // 24832 = 29
    for key, L, s_cert, cap, llama_key, llama_L in (("capability:gen_long_gsm8k_longsi_f22", 16640, 23, 8192, "capability:gen_long_gsm8k_longsi", 16384),
                                                    ("capability:gen_long_gsm8k_rung_longsi_f22", 24832, 15, 16384, "capability:gen_long_gsm8k_rung_longsi", 24576)):
        g = sg.geometry_for(key)
        assert g.key not in sg.TINYCAP_KEYS and g.key not in sg.informational_keys() and g.eval == "capability"
        assert not g.provisional and not g.informational and not g.certified and not g.requires_recertification
        assert (g.max_model_len, g.max_num_seqs) == (L, s_cert) and g.is_identity
        b = g.basis
        assert b.measured and b.measured_on == GSM8K_LONGSI_F22_MEASURED_ON and b.analytic is None
        assert b.sized_to_max_prompt and b.max_prompt == 8338 and b.p99_prompt == 8305 and b.requested_max_tokens == cap
        assert b.n_calls == 600 and b.n_rejected == 0 and b.n_capless == 0 and b.n_capped == 600
        assert b.families_measured == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")
        assert b.clamp_rate_at_L == 0.0 and b.frac_over_L == 0.0 and g.sized_by == sg.SIZED_BY_MAX_PROMPT_PLUS_MAX
        assert b.floor == sg.next_block(8338 + cap) and 0 < L - b.floor == 96 <= sg.SIZING_HEADROOM
        assert sg.sizing_problem(g) is None and sg.MEASUREMENT_DOC in b.source and "prompted__si_constitution" in b.source and "21e06f48" in b.source
        for mk, s_ in per_model[cap].items():
            r = sg.resolve_geometry(key, model_key=mk)
            assert (r.max_model_len, r.max_num_seqs) == (L, s_) and sg.BUDGET_BY_MODEL[mk].seqs_at(L) == s_ and 8338 + cap <= L, mk
        # the Llama-scoped entry does NOT hold these renders (why the scoped entry exists) and stays EXACTLY as PR-A registered it
        assert 8338 + cap > sg.TABLE[llama_key].max_model_len == llama_L and sg.TABLE[llama_key].basis.families_measured == ("llama8b",)
        with pytest.raises(ValueError, match="not a tinycap sub-entry"):
            sg.pass1_geometry(key)
        row = [l for l in sg.describe().splitlines() if l.startswith(f"| {key} |")][0]
        assert f"| {L} | {s_cert} |" in row and "| 8338 |" in row and f"| {cap} |" in row and GSM8K_LONGSI_F22_MEASURED_ON in row
    assert len(sg.TINYCAP_KEYS) == 8 and sg.TINYCAP_KEYS == C.GEOMETRY_SUBENTRY_KEYS
