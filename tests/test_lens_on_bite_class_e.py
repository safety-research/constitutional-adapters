"""Predicate (C) takes the CONCLUSIVE logprob-liveness verdict as its liveness evidence (the class E refusal).

The MATH-500 nemotron Arditi row (alpha 0.05 of the residual norm) passed identity (object == pin) and the
class-C LOGPROB liveness control (median first-token divergence 0.27 nats against a measured floor of 0, 5/5 probes
conclusive; text bites 0/5), then predicate (C) — the lens-ON "bite", ONE greedy render probe compared as TEXT against the
base route — refused it: the two replies were byte-identical. A gate demanding a token change certifies the DRIVER's
ability to see an effect, not the object's presence; at that dose an unchanged greedy render is the EXPECTED reading for a
refusal direction. The rule, with two conditions:

  (1) predicate (C) may accept the logprob verdict ONLY when it is CONCLUSIVE by the control's OWN criterion
      (``contracts.logprob_liveness_conclusive_problems`` == []), and
  (2) ONLY as a substitute for the TOKEN bite — the reasoning-block rule, the ON regime and the identity check (B′) are
      untouched, hard gates.

Every cell admitted this way carries the fact ON THE ROW, in the ruling's words: "live by logprob at this dose; the
render probe's reply is byte-identical to the base route, so no token bite was observable". An inconclusive verdict
refuses as before, text unchanged (the refusal is the result). A cell whose token bite IS observed writes the record it
always wrote, byte for byte.

Covered here, one layer at a time and end to end: the contracts verdict, the driver's ``_lens_on_bite`` (a scripted proxy
client), the MATH-500 reader (``certification_check``) and the GSM8K reader (``tinycap_gsm8k_validators``), the launcher's
row-identity stamp, and the analysis's row-level fact + paper consequence (``liveness_evidence_notes``, the footnote line).
Every block is built by running the producer's OWN helpers over scripted inputs, never a hand-typed copy of a shape."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gsm8k_synthetic_store as G  # noqa: E402
import math500_synthetic_store as S  # noqa: E402
import served_capability_run as srun  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from test_lens_liveness_logprob import Chat, _evidence  # noqa: E402
from test_tinycap_math500_thinking import GEMMA, NEMO, OBJECT, M, _logprob_gated_bites, _producer_certification_inputs  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # a @dataclass in the module resolves its own module through sys.modules
    spec.loader.exec_module(mod)
    return mod


CELLS = _load("m5t_cells_class_e", REPO / "scripts" / "sdfbase" / "math500" / "cells.py")
V = _load("tinycap_gsm8k_paired200_class_e", REPO / "scripts" / "tinycap_gsm8k_paired200.py").V

#: the ruling's sentence and the row-38 refusal text, VERBATIM — the contracts carry both as constants
SENTENCE = "live by logprob at this dose; the render probe's reply is byte-identical to the base route, so no token bite was observable"
NO_BITE = ("the lens-route reply is byte-identical to the base-route reply under the same render "
           "— the object did not bite under thinking-ON")
STAMP = {C.STAMP_BLOCK_KEY: {C.STAMP_KEY_RENDER_REGIME: C.RENDER_REGIME_THINKING_ON}}


def _conclusive_block(*, arm: str = "arditi", delta: float = 0.27) -> dict:
    """The class-C control's OWN record on a route live by logprob and inert by text (row 38's shape: static_r, a
    measured floor of 0, every probe conclusive) — the REAL ``_served_evidence`` over the scripted client."""
    ev, err = _evidence(Chat(steered_text="same words", twin_text="same words", delta=delta), cell_id=arm, arm=arm)
    assert err is None
    return ev[C.OOS_EVIDENCE_BITES_BLOCK][C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]


def _dead_block(*, arm: str = "arditi") -> dict:
    """The control's record on a route the statistic could NOT certify (steered == twin, bit-identical)."""
    ev, err = _evidence(Chat(steered_text="same words", twin_text="same words", delta=0.0), cell_id=arm, arm=arm)
    assert err is not None
    return ev[C.OOS_EVIDENCE_BITES_BLOCK][C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]


def _unevaluable_block(*, arm: str = "arditi") -> dict:
    """Too few conclusive probes (3 of 5 on disjoint top-K support): UNEVALUABLE by the control's own rule."""
    ev, err = _evidence(Chat(steered_text="same words", twin_text="same words", delta=0.27, disjoint_probes=(0, 1, 2)),
                        cell_id=arm, arm=arm)
    assert err is not None and "UNEVALUABLE" in err
    return ev[C.OOS_EVIDENCE_BITES_BLOCK][C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS]


def _lens_reply(fact, answer="Rayleigh scattering."):
    return {**S.think_shaped_completion(fact, answer), "render_regime": C.RENDER_REGIME_THINKING_ON}


def _fact(model_key=NEMO):
    return S.RENDER_FACTS[S.FAMILY_BY_KEY[model_key]]


def _admitted_record(fact, lp, *, cell=OBJECT) -> dict:
    """Predicate (C)'s record THROUGH the contracts helper on IDENTICAL replies with a conclusive verdict offered, plus the
    two vmodels the driver appends — what the driver files on a class-E admission."""
    lens = _lens_reply(fact)
    rec = C.lens_on_bite_verdict(lens_reply=lens, base_reply=dict(lens), family=fact["family"], logprob_liveness=lp)
    rec["lens_vmodel"], rec["base_vmodel"] = S.vmodel_of(cell), S.vmodel_of(S.CELL_BARE)
    return rec


def _logprob_bites(lp: dict | None = None, *, cell=OBJECT, arm="arditi") -> dict:
    """The driver's logprob-gated ``steering_bites`` block (kind static_r through the arditi arm) with the twin / render
    arm the MATH-500 reader expects; ``lp`` replaces the ruling block when given (a dead or edited verdict)."""
    chat = Chat(steered_text="w", twin_text="w", delta=0.27)
    ev, _err = _evidence(chat, cell_id=arm, arm=arm)
    b = dict(ev[C.OOS_EVIDENCE_BITES_BLOCK])
    b[C.OOS_EVIDENCE_BITES_TWIN] = C.TWIN_CELL_ID
    b[C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM] = C.THINKING_OFF
    b["arm"] = cell.split("__")[-1]
    if lp is not None:
        b[C.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS] = lp
    return b


# --------------------------------------------------------------------------------------------------------------------- #
# the contracts verdict                                                                                                   #
# --------------------------------------------------------------------------------------------------------------------- #


def test_the_constants_are_the_rulings_words_verbatim():
    assert C.LENS_ON_BITE_LOGPROB_SENTENCE == SENTENCE
    assert C.LENS_ON_BITE_NO_BITE_PROBLEM == NO_BITE
    assert C.LENS_ON_BITE_EVIDENCE_LOGPROB == "logprob_conclusive_no_token_bite" and C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY == "liveness_evidence"
    assert C.LENS_ON_BITE_EVIDENCE_TOKEN_BITE == "token_bite"
    # the launcher's stdlib-only stamp names the same manifest key the driver writes
    assert CELLS.MANIFEST_KEY_LENS_ON_BITE_LIVENESS == C.MANIFEST_KEY_LENS_ON_BITE_LIVENESS == "lens_on_bite_liveness_evidence"
    assert CELLS.ROW_IDENTITY_LIVENESS_KEY == "liveness_evidence"


def test_a_conclusive_logprob_verdict_stands_in_for_an_unobserved_token_bite_and_the_fact_is_on_the_record():
    """(a): identical lens / base replies + the same run's CONCLUSIVE verdict -> admitted, verified, with the evidence
    label, the sentence VERBATIM, the unobserved-bite observation and the deciding numbers copied from the block."""
    fact, lp = _fact(NEMO), _conclusive_block()
    assert C.logprob_liveness_conclusive_problems(lp) == []
    lens = _lens_reply(fact)
    rec = C.lens_on_bite_verdict(lens_reply=lens, base_reply=dict(lens), family=fact["family"], logprob_liveness=lp)
    assert rec["verified"] is True and rec["differs_from_base"] is False and rec["reasoning_present"] is True and rec["problems"] == []
    assert rec[C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY] == C.LENS_ON_BITE_EVIDENCE_LOGPROB
    assert rec[C.LENS_ON_BITE_SENTENCE_KEY] == SENTENCE
    assert rec[C.LENS_ON_BITE_TOKEN_BITE_KEY] == {"observed": False, "observation": NO_BITE}
    ref = rec[C.LENS_ON_BITE_LOGPROB_REF_KEY]
    assert ref == C.lens_on_bite_logprob_ref(lp)
    assert ref["kind"] == "static_r" and ref["verified"] is True and ref["median_divergence"] == pytest.approx(0.27)
    assert ref["measured_noise_floor"] == 0.0 and ref["bar"] == pytest.approx(1e-3) and ref["binding"] == "abs_floor"
    assert ref["n_conclusive"] == ref["n_probes"] == 5 and ref["min_conclusive_probes"] == 3
    assert "steering_bites.logprob_liveness" in ref["source"]
    # the ordinary keys are all still there, so every reader of the ordinary record still reads this one
    assert all(k in rec for k in ("applicable", "verified", "reasoning_present", "differs_from_base", "lens_render_regime",
                                  "lens_finish_reason", "base_finish_reason", "problems"))
    # the manifest block: the durable, row-level form
    blk = C.lens_on_bite_liveness_manifest_block(rec)
    assert blk[C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY] == C.LENS_ON_BITE_EVIDENCE_LOGPROB and blk[C.LENS_ON_BITE_SENTENCE_KEY] == SENTENCE
    assert blk[C.LENS_ON_BITE_LOGPROB_REF_KEY] == ref and "D95" in blk["rule"] and "identity" in blk["rule"]


def test_an_inconclusive_verdict_refuses_with_the_text_unchanged():
    """(b): a dead verdict (bit-identical logprobs), an UNEVALUABLE one (too few conclusive probes), an edited one (the
    count under the minimum, the sentence gone, the median under the bar) and no verdict at all all refuse — with EXACTLY
    the message the predicate raised before the ruling."""
    fact = _fact(NEMO)
    lens = _lens_reply(fact)
    with pytest.raises(C.LensOnBiteError) as before:
        C.lens_on_bite_verdict(lens_reply=lens, base_reply=dict(lens), family=fact["family"])
    assert str(before.value) == f"{C.REC_THINKING_UNSERVABLE}: predicate (C) lens-ON bite failed — {NO_BITE}"
    good = _conclusive_block()
    cases = {
        "dead": _dead_block(), "unevaluable": _unevaluable_block(),
        "below_min": {**good, "n_conclusive": 2},
        "no_sentence": {**good, "not_a_dose_axis": "a dose axis"},
        "under_bar": {**good, "divergence": {**good["divergence"], "steered_vs_scale0_median": 0.0, "clears_bar": False}},
        "no_floor": {**good, "noise_floor": {**good["noise_floor"], "measured": None}},
        "empty": {}, "not_a_block": "verified",
    }
    for name, lp in cases.items():
        problems = C.logprob_liveness_conclusive_problems(lp)
        assert problems, name
        with pytest.raises(C.LensOnBiteError) as e:
            C.lens_on_bite_verdict(lens_reply=lens, base_reply=dict(lens), family=fact["family"], logprob_liveness=lp)
        assert str(e.value) == str(before.value), name
    assert any("not verified" in p for p in C.logprob_liveness_conclusive_problems(cases["dead"]))
    assert any("do not reach the control's own minimum" in p for p in C.logprob_liveness_conclusive_problems(cases["below_min"]))
    assert any("not-a-dose-axis" in p for p in C.logprob_liveness_conclusive_problems(cases["no_sentence"]))
    assert any("does not clear the bar" in p for p in C.logprob_liveness_conclusive_problems(cases["under_bar"]))
    assert any("not a measured number" in p for p in C.logprob_liveness_conclusive_problems(cases["no_floor"]))


def test_only_the_token_bite_may_be_substituted_the_other_rules_and_identity_stay_hard():
    """(c): a conclusive verdict does NOT excuse a missing reasoning block or an OFF regime on the lens route (rule (1)
    and the regime are not the token bite), and the identity check (B′) is a different predicate the substitution never
    reaches — the driver runs (C) only when (B′) held, and the reader reads (B′) beside (C)."""
    fact, lp = _fact(NEMO), _conclusive_block()
    lens = _lens_reply(fact)
    with pytest.raises(C.LensOnBiteError, match="NO reasoning block"):
        C.lens_on_bite_verdict(lens_reply={**lens, "content": "ANSWER: 7", "reasoning": None},
                               base_reply={**lens, "content": "ANSWER: 7", "reasoning": None}, family=fact["family"], logprob_liveness=lp)
    with pytest.raises(C.LensOnBiteError, match="render_regime"):
        C.lens_on_bite_verdict(lens_reply={**lens, "render_regime": C.RENDER_REGIME_THINKING_OFF},
                               base_reply={**lens, "render_regime": C.RENDER_REGIME_THINKING_OFF}, family=fact["family"], logprob_liveness=lp)
    # identity: (B′) refuses a wrong pin regardless of any bite evidence
    endpoint = S.standup_endpoint(NEMO, OBJECT, pin=S.object_pin(NEMO))
    with pytest.raises(C.OutOfSupportAttestationError):
        C.lens_position_attested(endpoint, cell=OBJECT, expected_sha256="ff" * 32, model_key=NEMO)
    # and the driver reaches predicate (C) ONLY inside the branch (B′) passed — pinned as text (a live proxy is needed to run it)
    src = (REPO / "scripts" / "served_capability_run.py").read_text(encoding="utf-8")
    call = src.index("logprob_liveness=(bites.get(tc.OOS_EVIDENCE_BITES_LOGPROB_LIVENESS)")
    guard = src.rindex("elif position_error is None:", 0, call)
    assert 0 < call - guard < 900 and "_lens_on_bite(" in src[guard:call]
    # the block is offered ONLY when the logprob gate ruled the cell
    assert "if bites.get(tc.OOS_EVIDENCE_BITES_LIVENESS_GATE) == tc.LENS_LIVENESS_LOGPROB else None" in src[call:call + 300]
    # and the manifest key is written ONLY on an admitted record
    assert "manifest[tc.MANIFEST_KEY_LENS_ON_BITE_LIVENESS] = tc.lens_on_bite_liveness_manifest_block(" in src[call:call + 900]


def test_an_observed_token_bite_writes_the_record_it_always_wrote():
    """(d): with a bite observed the offered verdict changes NOTHING — the record is byte-identical to the one without the
    argument, carries no evidence key, and the conclusive-or-not question is never asked."""
    fact, lp = _fact(NEMO), _conclusive_block()
    lens, base = _lens_reply(fact, "Rayleigh scattering."), _lens_reply(fact, "Because of scattering.")
    plain = C.lens_on_bite_verdict(lens_reply=lens, base_reply=base, family=fact["family"])
    offered = C.lens_on_bite_verdict(lens_reply=lens, base_reply=base, family=fact["family"], logprob_liveness=lp)
    assert offered == plain and C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY not in offered and plain["differs_from_base"] is True
    assert json.dumps(offered, sort_keys=True) == json.dumps(plain, sort_keys=True)
    # a dead verdict offered beside an observed bite is equally irrelevant
    assert C.lens_on_bite_verdict(lens_reply=lens, base_reply=base, family=fact["family"], logprob_liveness=_dead_block()) == plain


# --------------------------------------------------------------------------------------------------------------------- #
# the driver                                                                                                              #
# --------------------------------------------------------------------------------------------------------------------- #


class _BiteChat:
    """A scripted proxy for predicate (C)'s two probes: the same think-shaped reply on both routes (``same``) or a
    different one on the lens route; every response carries the ON render stamp."""

    def __init__(self, *, same: bool):
        self.same, self.calls = same, []

    def __call__(self, payload):
        self.calls.append(dict(payload))
        lens = payload["model"] != "petri__none"
        text = "deep thoughts</think>\nANSWER: 7" if (self.same or not lens) else "other thoughts</think>\nANSWER: 8"
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], **STAMP}


def test_the_driver_admits_by_logprob_logs_the_fact_and_refuses_unchanged_on_an_inconclusive_verdict(capsys):
    lp = _conclusive_block()
    rec, err = srun._lens_on_bite(_BiteChat(same=True), lens_vmodel="petri__arditi", base_vmodel="petri__none",
                                  family="nemotron_h", logprob_liveness=lp)
    assert err is None and rec["verified"] is True and rec["differs_from_base"] is False
    assert rec[C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY] == C.LENS_ON_BITE_EVIDENCE_LOGPROB and rec[C.LENS_ON_BITE_SENTENCE_KEY] == SENTENCE
    assert rec["lens_vmodel"] == "petri__arditi" and rec["base_vmodel"] == "petri__none"
    out = capsys.readouterr().out
    assert "predicate (C) admitted by LOGPROB liveness (class E, D95)" in out and SENTENCE in out and "conclusive 5/5" in out
    # (b) at the driver: the refusal text is the one row 38 saw; the record says why the offered verdict could not stand in
    rec0, err0 = srun._lens_on_bite(_BiteChat(same=True), lens_vmodel="petri__arditi", base_vmodel="petri__none", family="nemotron_h")
    rec1, err1 = srun._lens_on_bite(_BiteChat(same=True), lens_vmodel="petri__arditi", base_vmodel="petri__none",
                                    family="nemotron_h", logprob_liveness=_dead_block())
    assert err0 == err1 == f"{C.REC_THINKING_UNSERVABLE}: predicate (C) lens-ON bite failed — {NO_BITE}"
    assert rec0["verified"] is False and "logprob_liveness_substitution" not in rec0
    sub = rec1["logprob_liveness_substitution"]
    assert sub["offered"] is True and sub["admitted"] is False and any("not verified" in p for p in sub["problems"])
    assert {k: v for k, v in rec1.items() if k != "logprob_liveness_substitution"} == rec0
    # no gate ruled (nothing offered): byte for byte the old path
    rec2, err2 = srun._lens_on_bite(_BiteChat(same=True), lens_vmodel="petri__arditi", base_vmodel="petri__none",
                                    family="nemotron_h", logprob_liveness=None)
    assert (rec2, err2) == (rec0, err0)
    assert "admitted by LOGPROB" not in capsys.readouterr().out


def test_the_driver_with_a_bite_observed_is_unchanged_whatever_is_offered(capsys):
    plain = srun._lens_on_bite(_BiteChat(same=False), lens_vmodel="petri__arditi", base_vmodel="petri__none", family="nemotron_h")
    offered = srun._lens_on_bite(_BiteChat(same=False), lens_vmodel="petri__arditi", base_vmodel="petri__none",
                                 family="nemotron_h", logprob_liveness=_conclusive_block())
    assert plain == offered and plain[1] is None and plain[0]["differs_from_base"] is True
    assert C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY not in plain[0] and "admitted by LOGPROB" not in capsys.readouterr().out


# --------------------------------------------------------------------------------------------------------------------- #
# the MATH-500 reader                                                                                                     #
# --------------------------------------------------------------------------------------------------------------------- #


def _cert(evidence, manifest, row, fact, *, model_key=NEMO, object_pin=None):
    return M.certification_check(evidence, manifest, row=row, fact=fact, family=fact["family"],
                                 object_pin=object_pin or S.object_pin(model_key))


def test_the_math500_reader_admits_the_record_on_the_evidences_own_terms_and_puts_the_fact_on_the_row():
    """(a) at the reader: the admitted record passes ONLY beside the same leg's logprob-gated, conclusive steering_bites;
    ``certification.liveness_evidence`` names the evidence and the sentence rides beside it."""
    evidence, manifest, row, fact = _producer_certification_inputs(NEMO)
    lp = _conclusive_block()
    ok = {**evidence, S.EVIDENCE_LENS_ON_BITE: _admitted_record(fact, lp), C.OOS_EVIDENCE_BITES_BLOCK: _logprob_bites(lp)}
    rec = _cert(ok, manifest, row, fact)
    assert rec["failures"] == [] and rec["passed"] is True
    assert rec["liveness_evidence"] == C.LENS_ON_BITE_EVIDENCE_LOGPROB and rec["liveness_sentence"] == SENTENCE
    assert rec[S.EVIDENCE_LENS_ON_BITE][C.LENS_ON_BITE_TOKEN_BITE_KEY] == {"observed": False, "observation": NO_BITE}
    # (d) at the reader: the ordinary record reads as before and is labelled by the reader as the token bite
    plain = _cert(evidence, manifest, row, fact)
    assert plain["failures"] == [] and plain["liveness_evidence"] == C.LENS_ON_BITE_EVIDENCE_TOKEN_BITE and plain["liveness_sentence"] is None


def test_the_math500_reader_refuses_every_way_the_admission_can_be_unsupported():
    evidence, manifest, row, fact = _producer_certification_inputs(NEMO)
    lp = _conclusive_block()
    admitted = _admitted_record(fact, lp)

    def fails(bite, bites, needle):
        d = " ".join(_cert({**evidence, S.EVIDENCE_LENS_ON_BITE: bite, C.OOS_EVIDENCE_BITES_BLOCK: bites}, manifest, row, fact)["failures"])
        assert needle in d, (needle, d)
        return d

    # the record's say-so alone: beside a TEXT-gated leg (a token bite was demanded there) it is refused by name
    fails(admitted, S.steering_bites_record(OBJECT), "liveness_gate is 'text'")
    # beside a logprob gate whose verdict is NOT conclusive (the dead block; the count under the minimum)
    fails(admitted, _logprob_bites(_dead_block()), "NOT conclusive")
    fails(admitted, _logprob_bites({**lp, "n_conclusive": 2}), "NOT conclusive")
    # the copied numbers must equal the block's
    fails({**admitted, C.LENS_ON_BITE_LOGPROB_REF_KEY: {**admitted[C.LENS_ON_BITE_LOGPROB_REF_KEY], "median_divergence": 9.0}},
          _logprob_bites(lp), "disagrees with steering_bites.logprob_liveness on ['median_divergence']")
    # the sentence verbatim; the observation; an unknown evidence label; a bite that WAS observed yet claims the substitution
    fails({**admitted, C.LENS_ON_BITE_SENTENCE_KEY: "live enough"}, _logprob_bites(lp), "not the ruling's sentence verbatim")
    fails({**admitted, C.LENS_ON_BITE_TOKEN_BITE_KEY: {"observed": True}}, _logprob_bites(lp), "not the unobserved-bite observation")
    fails({**admitted, C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY: "vibes"}, _logprob_bites(lp), "is not 'logprob_conclusive_no_token_bite'")
    fails({**admitted, "differs_from_base": True}, _logprob_bites(lp), "UNOBSERVED token bite only")
    # the OLD rule is untouched: a verified-looking record with no bite and NO evidence key is refused as before
    fails({**S.lens_on_bite_record(OBJECT, fact), "differs_from_base": False}, _logprob_bites(lp), "differs_from_base is False")
    # (c) at the reader: identity is a separate hard gate — a wrong pin refuses the leg however (C) was admitted
    bad = _cert({**evidence, S.EVIDENCE_LENS_ON_BITE: admitted, C.OOS_EVIDENCE_BITES_BLOCK: _logprob_bites(lp)}, manifest, row, fact,
                object_pin="ff" * 32)
    assert bad["passed"] is False and any(S.EVIDENCE_LENS_POSITION in f and "object_digest" in f for f in bad["failures"])


def test_the_reader_helpers_label_the_evidence_and_the_no_switch_record_is_untouched():
    fact, lp = _fact(NEMO), _conclusive_block()
    assert M.liveness_evidence_of(_admitted_record(fact, lp), fitted=True) == C.LENS_ON_BITE_EVIDENCE_LOGPROB
    assert M.liveness_evidence_of(S.lens_on_bite_record(OBJECT, fact), fitted=True) == C.LENS_ON_BITE_EVIDENCE_TOKEN_BITE
    assert M.liveness_evidence_of(S.lens_on_bite_record(OBJECT, fact), fitted=False) is None
    assert M.liveness_evidence_of({"applicable": True, "verified": False, "error": "x"}, fitted=True) is None
    assert M.liveness_evidence_of(C.no_switch_lens_on_bite_record(), fitted=True) is None
    assert M._lens_on_bite_problems(C.no_switch_lens_on_bite_record(), route_vmodel="petri__x", cert_class=C.CERT_CLASS_NO_SWITCH,
                                    bites=_logprob_bites(lp)) == []


# --------------------------------------------------------------------------------------------------------------------- #
# the GSM8K reader                                                                                                        #
# --------------------------------------------------------------------------------------------------------------------- #

G8K_ROWS = G.matrix_rows()
G8K_CELL = "constv2-mix-full__psr_ce_all"      # an UNDECLARED fitted cell on the panel view


def _g8k_inputs(lp, *, bite=None, bites=None):
    row = G.row_for(G8K_ROWS, "qwen35_4b", "main", G8K_CELL)
    assert row["fitted"] and not row["declared"]
    # the launch matrix's derived field the validator reads (the loader adds it; the synthetic row builder does not)
    row = {**row, "expected_stamps": list(G.stamps_for(True, False))}
    fact = S.RENDER_FACTS[row["family"]]
    spell = C.cell_spellings(row["cell"], declared=False)
    rec = bite if bite is not None else _admitted_record(fact, lp)
    rec = {**rec, "lens_vmodel": spell.vmodel, "base_vmodel": C.cell_spellings(G.CELL_NONE, declared=False).vmodel}
    evidence = {M.EVIDENCE_RENDER_CERTIFIED: S.render_certified_record(fact["family"], fact),
                M.EVIDENCE_RENDER_PROBE: S.render_probe_record(fact),
                M.EVIDENCE_LENS_POSITION: G.lens_position_record(row),
                M.EVIDENCE_LENS_ON_BITE: rec,
                C.OOS_EVIDENCE_BITES_BLOCK: bites if bites is not None else _logprob_bites(lp, cell=G8K_CELL)}
    manifest = {C.MANIFEST_KEY_ROUTE: {"virtual_model": spell.vmodel, C.ROUTE_KEY_SERVE_DECLARATIONS: {}, C.LENS_SERVED_POSITIONS_KEY: ["all"]},
                C.LENS_SERVED_POSITIONS_KEY: ["all"]}
    return evidence, manifest, row, fact


def test_the_gsm8k_reader_admits_the_same_record_and_learns_the_logprob_gate_it_did_not_know():
    """The GSM8K nemotron Arditi row runs on the same driver; its reader delegates predicate (C) to the MATH-500 module
    and, until now, read the steering-bites control with the TEXT rule only (``n_differing >= 1``) — it would have
    refused any logprob-gated leg with inert text. It now reads the gate as the MATH-500 reader does."""
    lp = _conclusive_block()
    evidence, manifest, row, fact = _g8k_inputs(lp)
    cert = V.certification_check(evidence, manifest, row=row, fact=fact)
    assert cert["passed"] is True, cert["failures"]
    assert cert[M.EVIDENCE_LENS_ON_BITE][C.LENS_ON_BITE_LIVENESS_EVIDENCE_KEY] == C.LENS_ON_BITE_EVIDENCE_LOGPROB
    # the class-C shape alone (an ordinary bite record beside a logprob-gated block with 0 text bites) passes too
    ordinary = G.lens_on_bite_record(row, fact)
    evidence2, *_ = _g8k_inputs(lp, bite=ordinary)
    assert V.certification_check(evidence2, manifest, row=row, fact=fact)["passed"] is True
    # and every way the admission is unsupported refuses, as in the MATH-500 reader
    for bites, needle in ((S.steering_bites_record(G8K_CELL), "liveness_gate is 'text'"),
                          (_logprob_bites(_dead_block(), cell=G8K_CELL), "NOT conclusive")):
        ev, *_ = _g8k_inputs(lp, bites=bites)
        d = " ".join(V.certification_check(ev, manifest, row=row, fact=fact)["failures"])
        assert needle in d, (needle, d)
    # the bites rule by gate: logprob-gated + inert text passes; text-gated + inert text refuses as before; a dead
    # logprob verdict refuses; an unknown gate refuses by name
    assert V.steering_bites_problems(_logprob_bites(lp)) == []
    assert any("n_differing 0 < 1" in p for p in V.steering_bites_problems(S.steering_bites_record(G8K_CELL, n_differing=0)))
    assert any("not verified" in p for p in V.steering_bites_problems(_logprob_bites(_dead_block())))
    assert any("liveness_gate 'rho'" in p for p in V.steering_bites_problems({**_logprob_bites(lp), C.OOS_EVIDENCE_BITES_LIVENESS_GATE: "rho"}))
    assert V.steering_bites_problems(S.steering_bites_record(G8K_CELL)) == []


# --------------------------------------------------------------------------------------------------------------------- #
# the launcher's row-identity stamp                                                                                       #
# --------------------------------------------------------------------------------------------------------------------- #


def test_the_row_identity_stamp_copies_the_manifest_fact_onto_the_row_only_when_the_manifest_carries_it(tmp_path):
    import test_sdfbase_math500_sbatch as TS
    row = TS.M.row_for("nemotron3nano4b", TS.M.CELL_LORA_CTL2, TS.M.SAMPLING_THINKING_CARD)
    sha = "cd" * 20
    block = TS._producer_sampling_block(row)
    ref = C.sampling_ref(block)
    fact_block = C.lens_on_bite_liveness_manifest_block(_admitted_record(_fact(NEMO), _conclusive_block()))
    TS._producer_store(tmp_path / "admitted", row, sha, block=block, gs_ref=ref, item_refs=(ref, ref, ref),
                       manifest_over={C.MANIFEST_KEY_LENS_ON_BITE_LIVENESS: fact_block})
    rec = TS.M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / "admitted"))
    assert rec["complete"] is True and rec["schema"] == "m5t_row_identity/3"
    assert rec[TS.M.ROW_IDENTITY_LIVENESS_KEY] == fact_block
    assert rec[TS.M.ROW_IDENTITY_LIVENESS_KEY][C.LENS_ON_BITE_SENTENCE_KEY] == SENTENCE
    assert tuple(rec["identity"]) == TS.M.ROW_IDENTITY_KEYS            # the canonical identity block did not move
    # a cell with a token bite writes no manifest key -> the stamp is byte-identical to before (no key at all)
    TS._producer_store(tmp_path / "plain", row, sha, block=block, gs_ref=ref, item_refs=(ref, ref, ref))
    plain = TS.M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / "plain"))
    assert plain["complete"] is True and TS.M.ROW_IDENTITY_LIVENESS_KEY not in plain
    assert TS.M.ROW_IDENTITY_LIVENESS_KEY not in TS.M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / "none"))


# --------------------------------------------------------------------------------------------------------------------- #
# the analysis's row-level fact and the paper consequence                                                                 #
# --------------------------------------------------------------------------------------------------------------------- #


def _leg(model_key, cell, kind, *, evidence, sentence=None):
    return {"model_key": model_key, "cell": cell, "mode": "thinking_card", "fitted": True, "kind": kind, "view": "main",
            "certification": {"liveness_evidence": evidence, "liveness_sentence": sentence,
                              M.EVIDENCE_LENS_POSITION: {"served_positions": ["all"]}}}


def test_the_analysis_carries_the_fact_per_row_and_states_the_paper_consequence(monkeypatch):
    admitted = _leg(NEMO, "arditi", "arditi", evidence=C.LENS_ON_BITE_EVIDENCE_LOGPROB, sentence=SENTENCE)
    ordinary = _leg(GEMMA, "arditi", "arditi", evidence=C.LENS_ON_BITE_EVIDENCE_TOKEN_BITE)
    rows = {f"{NEMO}/arditi/thinking_card": M.provenance_row(admitted), f"{GEMMA}/arditi/thinking_card": M.provenance_row(ordinary)}
    assert rows[f"{NEMO}/arditi/thinking_card"]["liveness_evidence"] == C.LENS_ON_BITE_EVIDENCE_LOGPROB
    assert rows[f"{NEMO}/arditi/thinking_card"]["liveness_sentence"] == SENTENCE
    assert rows[f"{GEMMA}/arditi/thinking_card"]["liveness_evidence"] == "token_bite" and rows[f"{GEMMA}/arditi/thinking_card"]["liveness_sentence"] is None
    report = {"provenance": {"cells": rows}, "gates": {}, "comparisons": []}
    notes = M.liveness_evidence_notes(report)
    assert set(notes["by_leg"]) == {f"{NEMO}/arditi/thinking_card"} and "D95" in notes["rule"]
    entry = notes["by_leg"][f"{NEMO}/arditi/thinking_card"]
    assert entry["sentence"] == SENTENCE and entry["model_key"] == NEMO and entry["arm"] == "arditi"
    assert entry["consequence"] == (f"{NEMO} arditi at its operating point does not move a single token of the render probe — "
                                    "bears on how much defence the arm buys at that dose")
    assert M.liveness_evidence_notes({"provenance": {"cells": {k: v for k, v in rows.items() if GEMMA in k}}}) == {"rule": None, "by_leg": {}}
    # the disclosure notes and the figure footnote carry it (the heavy note builders stubbed: they need a full report)
    monkeypatch.setattr(M, "served_geometry_notes", lambda r: {})
    monkeypatch.setattr(M, "conditional_vs_score_as_is_notes", lambda r: {})
    d48 = M.d48_notes(report)
    assert d48["liveness_evidence"] == notes
    parts = M._d48_footnote_parts(report)
    assert any(p.startswith(f"live by logprob, no token bite (D95): {NEMO} arditi") and "byte-identical to the base route" in p for p in parts)
    # an excluded family's leg leaves the footnote (the notes keep it)
    hidden = {**report, "figures": {M.FIGURES_EXCLUDED_KEY: [NEMO]}}
    assert not any("D95" in p for p in M._d48_footnote_parts(hidden)) and M.d48_notes(hidden)["liveness_evidence"] == notes
