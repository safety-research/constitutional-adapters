"""S2-parity probe tests.

Base lane, CPU: the conftest tiny_llama drives the real Engine protocol
path (per-call ctx untouched — arm=None here; hook hygiene is
test_capeval_hf_engine's job); a fake engine drives the extraction and
row-emission logic with controlled generations.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval import store as cs
from sisteer.capeval.corpora import CorpusError
from sisteer.capeval.serving import GenOut, PositionScore
from sisteer.capeval.tiers import parity

REPO = Path(__file__).parents[1]
REV = "0123456789abcdef0123456789abcdef01234567"

TEMPLATE = {
    "sweep": "parity_test", "run_id": "r0", "model_id": "test-org/tiny",
    "model_revision": REV, "arm_name": "none", "kind": "none", "coeff": None,
    "intervention_hash": "a1b2c3d4e5f60718", "engine": "hf", "trial_id": 1,
    "seed": 1, "attempt": 0, "harness_config_hash": "0f1e2d3c4b5a6978",
    "env_digest": "99aabbccddeeff00", "slurm_job_id": None,
}
DOMAIN_OF = {parity.PARITY_TASK_GSM8K: "math_s2parity",
             parity.PARITY_TASK_MMLU: "knowledge_s2parity"}

GSM8K_ROWS = [
    {"item_id": "g0000-aaaaaaaa", "question": "Two plus two?", "answer": "x\n#### 4"},
    {"item_id": "g0001-bbbbbbbb", "question": "Three times three?", "answer": 9.0},
    {"item_id": "g0002-cccccccc", "question": "Five minus one?", "answer": "#### 4"},
]
MMLU_ROWS = [
    {"item_id": "m0000-dddddddd", "question": "Pick B.",
     "choices": ["one", "two", "three", "four"], "answer": 1},
    {"item_id": "m0001-eeeeeeee", "question": "Pick D.",
     "choices": ["ay", "bee", "sea", "dee"], "answer": 3},
]


def _spec(order_seed=1, gsm8k=GSM8K_ROWS, mmlu=MMLU_ROWS):
    return parity.ParitySpec(template=dict(TEMPLATE), domain_of=dict(DOMAIN_OF),
                             gsm8k_rows=list(gsm8k), mmlu_rows=list(mmlu),
                             order_seed=order_seed)


def _store(tmp_path, key="none__tier01__t1__a0__a1b2c3d4__0f1e2d3c"):  # synthetic cell key — gitleaks:allow
    return cs.CellStore(tmp_path, key, row_constants=TEMPLATE)


class FakeEngine:
    """Canned generations keyed by question text; letter NLLs canned per
    question so the mmlu pick is controlled. Records every gen_cfg it
    receives (the batch_size pass-through contract)."""

    name = "hf"

    def __init__(self, tok, gen_by_question, nll_by_question):
        self.tokenizer = tok
        self._gen = gen_by_question
        self._nll = nll_by_question
        self.gen_cfgs = []
        self.prompts_seen = []
        self.gen_arms = []
        self.score_arms = []

    def generate(self, prompts, arm, gen_cfg):
        self.gen_cfgs.append(dict(gen_cfg))
        self.gen_arms.append(arm)
        self.prompts_seen.extend(prompts)
        outs = []
        for p in prompts:
            match = [t for q, t in self._gen.items() if q in p]
            assert len(match) == 1, p
            outs.append(GenOut(text=match[0], gen_token_logprobs=[], ended_with_eos=True))
        return outs

    def score_corpus(self, corpus, arm=None, atom_ids=None):
        self.score_arms.append(arm)
        for item in corpus.items:
            qid = item.item_id.split("::")[0]
            letter = item.item_id.split("::")[1]
            yield PositionScore(self._nll[qid][letter], None)


@pytest.fixture()
def tok(tiny_llama):
    return tiny_llama[1]


@pytest.fixture()
def legacy_gsm8k(monkeypatch):
    """PW3: the gsm8k probe is the RETIRED bespoke instrument; its generation
    leg runs only under the legacy switch. Tests that exercise the leg's
    mechanics (extraction, ordering, batch size, arm ctx) opt in here."""
    from sisteer.capeval.tinycap import contracts as tc

    monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")


def test_gsm8k_leg_refuses_without_the_legacy_switch(tmp_path, tok, monkeypatch):
    """PW3 (U2-impl 4): parity.run with gsm8k rows and no SISTEER_LEGACY_GSM8K
    refuses BEFORE any generation, naming the standard (inspect/gsm8k on the
    served path) and the switch; the row loader itself stays callable."""
    from sisteer.capeval.tinycap import contracts as tc

    monkeypatch.delenv(tc.GSM8K_LEGACY_ENV, raising=False)
    gens = {r["question"]: "#### 4" for r in GSM8K_ROWS}
    nlls = {r["item_id"]: {"A": 0.1, "B": 1.0, "C": 2.0, "D": 3.0} for r in MMLU_ROWS}
    eng = FakeEngine(tok, gens, nlls)
    store = _store(tmp_path)
    with pytest.raises(parity.ParityError) as ei:
        parity.run(eng, None, _spec(), store)
    msg = str(ei.value)
    assert "RETIRED" in msg and "inspect/gsm8k" in msg and tc.GSM8K_LEGACY_ENV in msg
    assert "s2parity/gsm8k_0shot" in msg
    assert eng.prompts_seen == [] and eng.gen_cfgs == []      # nothing generated
    assert store._parsed_rows() == []                           # nothing filed
    # the switch must be exactly "1"
    monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "yes")
    with pytest.raises(parity.ParityError, match="RETIRED"):
        parity.run(eng, None, _spec(), store)


def test_mmlu_runs_alone_when_the_gsm8k_rows_are_empty(tmp_path, tok, monkeypatch):
    """PW3: the HF controls driver and the capeval runtime hand parity.run an
    EMPTY gsm8k row set; the mmlu probe runs, the summary has no gsm8k key,
    no gsm8k row is filed, and no legacy switch is needed. Both empty refuses."""
    from sisteer.capeval.tinycap import contracts as tc

    monkeypatch.delenv(tc.GSM8K_LEGACY_ENV, raising=False)
    nlls = {"m0000-dddddddd": {"A": 3.0, "B": 0.1, "C": 2.0, "D": 4.0},
            "m0001-eeeeeeee": {"A": 0.5, "B": 1.0, "C": 2.0, "D": 3.0}}
    eng = FakeEngine(tok, {}, nlls)
    store = _store(tmp_path)
    summary = parity.run(eng, None, _spec(gsm8k=[]), store)
    assert set(summary) == {"mmlu"} and summary["mmlu"]["n"] == 2
    assert eng.gen_cfgs == []   # the generation leg never ran
    tasks = {r["task"] for r in store._parsed_rows()}
    assert tasks == {parity.PARITY_TASK_MMLU}
    with pytest.raises(parity.ParityError, match="empty row sets"):
        parity.run(eng, None, _spec(gsm8k=[], mmlu=[]), _store(tmp_path / "none"))


def test_loaders_refuse_without_pins():
    with pytest.raises(CorpusError, match="gsm8k_platinum"):
        parity.gsm8k_parity_rows({"datasets": {}})
    with pytest.raises(CorpusError, match="mmlu"):
        parity.mmlu_parity_rows({"datasets": {}})


def test_fake_engine_rows_extraction_and_aggregates(tmp_path, tok, legacy_gsm8k):
    gens = {"Two plus two?": "think... #### 4",          # strict hit -> acc 1
            "Three times three?": "the answer is 8",     # fallback, wrong -> 0
            "Five minus one?": "no digits here at all"}  # extraction fail -> 0
    nlls = {"m0000-dddddddd": {"A": 3.0, "B": 0.1, "C": 2.0, "D": 4.0},   # -> B == gold
            "m0001-eeeeeeee": {"A": 0.5, "B": 1.0, "C": 2.0, "D": 3.0}}   # -> A != gold (D)
    store = _store(tmp_path)
    summary = parity.run(FakeEngine(tok, gens, nlls), None, _spec(), store)
    assert summary["gsm8k"]["n"] == 3 and summary["mmlu"]["n"] == 2
    assert summary["gsm8k"]["acc"] == pytest.approx(1 / 3)
    assert summary["mmlu"]["acc"] == pytest.approx(1 / 2)

    rows = store._parsed_rows()
    by = {(r["task"], r["item_id"], r["metric"]): r for r in rows}
    assert by[(parity.PARITY_TASK_GSM8K, "g0000-aaaaaaaa", "acc")]["value"] == 1.0
    assert by[(parity.PARITY_TASK_GSM8K, "g0002-cccccccc", "extraction_failed")]["value"] == 1.0
    assert by[(parity.PARITY_TASK_GSM8K, "g0001-bbbbbbbb", "extraction_failed")]["value"] == 0.0
    assert by[(parity.PARITY_TASK_MMLU, "m0000-dddddddd", "acc")]["value"] == 1.0
    assert by[(parity.PARITY_TASK_MMLU, "m0001-eeeeeeee", "acc")]["value"] == 0.0
    # domains resolve through the registry mapping; aggregates exist —
    # under their OWN metric name, never the per-item "acc" (2026-08-17
    # incident: an "acc"-named aggregate row rode into pairing as a 201st
    # item)
    assert by[(parity.PARITY_TASK_GSM8K, "g0000-aaaaaaaa", "acc")]["domain"] == "math_s2parity"
    assert by[(parity.PARITY_TASK_MMLU, "__aggregate__",
               parity.PARITY_AGG_METRIC)]["domain"] == "knowledge_s2parity"
    assert (parity.PARITY_TASK_MMLU, "__aggregate__", "acc") not in by
    # per-item extras carry the audit trail
    extra = json.loads(by[(parity.PARITY_TASK_MMLU, "m0000-dddddddd", "acc")]["extra_json"])
    assert extra == {"pick": "B", "gold": "B"}


def test_item_ids_stable_across_trials_orders_differ(tmp_path, tok, legacy_gsm8k):
    gsm8k = [{"item_id": f"g{i:04d}-{'ab' * 4}", "question": f"What is {i} plus {i}?",
              "answer": float(2 * i)} for i in range(6)]
    gens = {r["question"]: "#### 4" for r in gsm8k}
    nlls = {r["item_id"]: {"A": 0.1, "B": 1.0, "C": 2.0, "D": 3.0} for r in MMLU_ROWS}
    eng = FakeEngine(tok, gens, nlls)
    s1 = _store(tmp_path / "t1")
    parity.run(eng, None, _spec(order_seed=1, gsm8k=gsm8k), s1)
    s2 = _store(tmp_path / "t2", key="none__tier01__t2__a0__a1b2c3d4__0f1e2d3c")  # synthetic cell key — gitleaks:allow
    parity.run(eng, None, _spec(order_seed=2, gsm8k=gsm8k), s2)

    def acc_rows(store, task):
        return {r["item_id"]: r["position_order"] for r in store._parsed_rows()
                if r["task"] == task and r["metric"] == "acc"
                and r["item_id"] != "__aggregate__"}

    a1, a2 = acc_rows(s1, parity.PARITY_TASK_GSM8K), acc_rows(s2, parity.PARITY_TASK_GSM8K)
    assert set(a1) == set(a2)                      # pairing key: stable item_ids
    assert a1 != a2                                # §7.1: own order per trial
    # resume idempotence: re-running the same trial adds NO rows
    n_before = len(s1._parsed_rows())
    parity.run(eng, None, _spec(order_seed=1, gsm8k=gsm8k), s1)
    assert len(s1._parsed_rows()) == n_before


def test_gsm8k_configured_batch_size_reaches_the_engine(tmp_path, tok, legacy_gsm8k):
    """The configured batch size rides in gen_cfg all the way to
    engine.generate — the ENGINE chunks (fresh ctx per chunk), so
    dropping the key here meant one 200-prompt batch and an OOM."""
    gens = {r["question"]: "#### 4" for r in GSM8K_ROWS}
    nlls = {r["item_id"]: {"A": 0.1, "B": 1.0, "C": 2.0, "D": 3.0}
            for r in MMLU_ROWS}
    eng = FakeEngine(tok, gens, nlls)
    parity.run(eng, None, _spec(), _store(tmp_path))
    [cfg] = eng.gen_cfgs  # one gsm8k generate call
    assert cfg["batch_size"] == 8  # the ParitySpec default (§9)
    assert cfg["max_new_tokens"] == 512 and cfg["do_sample"] is False


class _SelectorArm:
    """An arm handle stub resolving a SELECTOR position policy ("last").
    ("response" is no longer a refusal here — it takes the deployment-exact
    response-mask scoring branch, tested in
    test_response_mask_capabilities.py.)"""

    name, kind, coeff, intervention_hash = "sel", "steering", 1.0, "22" * 8
    resolved_spec = {"positions": "last"}

    def ctx(self):
        import contextlib

        return contextlib.nullcontext()

    def transform_prompt(self, s):
        return s


class _GeneratedArm:
    """An arm handle stub resolving the engine-phase generated policy."""

    name, kind, coeff, intervention_hash = "gen", "steering", 1.0, "33" * 8
    resolved_spec = {"positions": "all", "steer_positions": "generated"}

    def ctx(self):
        raise AssertionError(
            "the mmlu probe must never enter a generated-policy arm's ctx")

    def transform_prompt(self, s):
        return s


def test_mmlu_scores_arm_free_under_generated_policy(tmp_path, tok, legacy_gsm8k):
    """Generated-only serving (2026-08-20): the letter probe's single
    scored position is the FIRST GENERATED one, whose predictor is the
    prompt's last token — a prompt position, served by the plain base
    model under the policy. The probe must therefore score ARM-FREE
    (the exact instrument value for that serving; base-by-construction),
    while gsm8k — generation-based — still runs under the real arm."""
    gens = {r["question"]: "#### 4" for r in GSM8K_ROWS}
    nlls = {r["item_id"]: {"A": 0.1, "B": 1.0, "C": 2.0, "D": 3.0}
            for r in MMLU_ROWS}
    eng = FakeEngine(tok, gens, nlls)
    arm = _GeneratedArm()
    summary = parity.run(eng, arm, _spec(), _store(tmp_path))
    assert eng.score_arms == [None]   # mmlu scored arm-free (never the ctx)
    assert eng.gen_arms == [arm]      # gsm8k still served under the arm
    assert summary["mmlu"]["n"] == 2 and summary["gsm8k"]["n"] == 3


def test_mmlu_refuses_selector_position_arm(tmp_path, tok, legacy_gsm8k):
    """The letter-logprob probe appends the candidate letter, so a
    selector policy would steer the letter itself — refused until the
    v1 selector-aware scorer exists (engine-generic: this fires even
    on engines without the hf_engine-side check)."""
    gens = {r["question"]: "#### 4" for r in GSM8K_ROWS}
    nlls = {r["item_id"]: {"A": 0.1, "B": 1.0, "C": 2.0, "D": 3.0}
            for r in MMLU_ROWS}
    eng = FakeEngine(tok, gens, nlls)
    with pytest.raises(parity.ParityError, match="v1"):
        parity.run(eng, _SelectorArm(), _spec(), _store(tmp_path))


def test_mmlu_pick_equals_direct_letter_logit_argmax(tmp_path, tiny_llama, legacy_gsm8k):
    """The protocol-path decision (argmin letter NLL via score_corpus)
    must equal the S2 panel's rule (argmax over letter logits of one
    forward) on the real tiny model."""
    import torch

    from sisteer.capeval.serving.hf_engine import HFEngine
    from sisteer.eval.capability_ext import _letter_token_ids, _letters
    from sisteer.generate import encode_kwargs

    model, tok = tiny_llama
    engine = HFEngine("test-org/tiny", revision=None, preloaded=(model, tok))
    rows = MMLU_ROWS
    store = _store(tmp_path)
    spec = _spec(gsm8k=[{"item_id": "g0", "question": "q", "answer": 1.0}],
                 mmlu=rows)
    gens = {"q": "#### 1"}

    class GenOnlyEngine(HFEngine):
        def generate(self, prompts, arm, gen_cfg):  # tiny model rambles; pin gsm8k
            return [GenOut(gens["q"], [], True) for _ in prompts]

    eng = GenOnlyEngine("test-org/tiny", revision=None, preloaded=(model, tok))
    parity.run(eng, None, spec, store)
    picks = {r["item_id"]: json.loads(r["extra_json"])["pick"]
             for r in store._parsed_rows()
             if r["task"] == parity.PARITY_TASK_MMLU and r["metric"] == "acc"
             and r["item_id"] != "__aggregate__"}

    letters = _letters(4)
    letter_ids = _letter_token_ids(tok, letters)
    for row in rows:
        prompt = parity._mmlu_prompt(tok, row)
        enc = tok(prompt, return_tensors="pt", **encode_kwargs(tok, prompt))
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]
        direct = letters[int(logits[letter_ids].argmax())]
        assert picks[row["item_id"]] == direct


def test_runtime_dispatch_gating(tmp_path, monkeypatch):
    """Parity runs for (calibration|sweep) x hf ONLY — wired at the end
    of runtime.run_tier1 (no driver/event-order changes)."""
    from sisteer.capeval import runtime as rt
    from sisteer.capeval.tinycap import contracts as tc

    calls = []
    specs = []
    monkeypatch.setattr(rt, "_chat_template", lambda m: "T")
    monkeypatch.setattr(parity, "run", lambda e, a, s, w: (calls.append(s.template["kind"]),
                                                            specs.append(s)) and {"stub": True})
    # PW3: without the legacy switch the runtime must not even LOAD the
    # retired probe's rows; the mmlu probe runs alone and the skip is recorded
    monkeypatch.delenv(tc.GSM8K_LEGACY_ENV, raising=False)
    monkeypatch.setattr(parity, "gsm8k_parity_rows",
                        lambda pins: pytest.fail("retired gsm8k rows loaded without the switch"))
    monkeypatch.setattr(parity, "mmlu_parity_rows", lambda pins: [{"item_id": "m", "question": "q", "choices": ["a", "b", "c", "d"], "answer": 0}])
    from sisteer.capeval.tiers import evalplus_runner
    monkeypatch.setattr(evalplus_runner, "generate_samples",
                        lambda **kw: kw["store"].write_generation_complete() or [])

    class Eng:
        name = "hf"
        tokenizer = object()

    def mk_manifest(mode, engine):
        return cm.Manifest(
            sweep=f"pt_{mode}_{engine}", mode=mode, base="test-org/tiny",
            base_revision=REV, engine=engine, tiers=[0, 1], confirmatory=False,
            target=None, margins={},
            ablations=[cm.AblationSpec(name="none", kind="none")])

    cell = dataclasses.make_dataclass(
        "C", ["index", "arm", "kind", "coeff", "trial", "tiers", "time_class"])(
        0, "none", "none", None, 1, (0, 1), "full")

    for mode, engine_name, expect in (
            ("calibration", "hf", True),
            ("control", "hf", False),
            ("calibration", "vllm", False)):
        calls.clear()
        specs.clear()
        m = mk_manifest(mode, engine_name)
        store, _ = rt.resolve_cell(m, cell, 0, tmp_path / f"{mode}_{engine_name}", 1)
        summary = rt.run_tier1(Eng(), m, cell, store, seed=1)
        assert bool(calls) is expect, (mode, engine_name)
        if expect:
            assert specs[0].gsm8k_rows == [] and len(specs[0].mmlu_rows) == 1
            assert "retired" in summary["s2parity_gsm8k"]
            assert "inspect/gsm8k" in summary["s2parity_gsm8k"]
    # under the switch the legacy probe's rows are loaded and handed over
    monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")
    monkeypatch.setattr(parity, "gsm8k_parity_rows",
                        lambda pins: [{"item_id": "g", "question": "q", "answer": 1.0}])
    specs.clear()
    m = mk_manifest("calibration", "hf")
    store, _ = rt.resolve_cell(m, cell, 0, tmp_path / "legacy", 1)
    summary = rt.run_tier1(Eng(), m, cell, store, seed=1)
    assert len(specs[0].gsm8k_rows) == 1 and "s2parity_gsm8k" not in summary


def test_parity_spec_system_default_none_preserves_prompts(tmp_path, tok, legacy_gsm8k):
    """capeval-controls §3.2: the additive ParitySpec.system kwarg.
    Default None keeps both probes' prompts byte-identical to the
    instruments' own renders; a set system flows into gsm8k prompts and
    the mmlu render (same user text, SI as system message)."""
    row = MMLU_ROWS[0]
    # None path == the instrument's own render, byte-for-byte
    assert parity._mmlu_prompt(tok, row) == parity._capability_mmlu_prompt(
        tok, row["question"], row["choices"])
    si = "SYNTHETIC SI TEXT for the prompted cell"
    with_sys = parity._mmlu_prompt(tok, row, system=si)
    assert si in with_sys
    assert row["question"] in with_sys
    assert with_sys.endswith("Answer:")
    # the same A-D user text rides along (guards drift from capability's)
    for letter, choice in zip("ABCD", row["choices"]):
        assert f"{letter}. {choice}" in with_sys

    gens = {r["question"]: "#### 4" for r in GSM8K_ROWS}
    nlls = {r["item_id"]: {"A": 0.1, "B": 1.0, "C": 2.0, "D": 3.0}
            for r in MMLU_ROWS}
    eng = FakeEngine(tok, gens, nlls)
    spec = _spec()
    spec.system = si
    parity.run(eng, None, spec, _store(tmp_path))
    assert eng.prompts_seen  # the gsm8k generate leg ran
    assert all(si in p for p in eng.prompts_seen)
