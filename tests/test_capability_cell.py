"""System threading on the capability entries + the cell-taking wrapper.

CPU only (tiny-llama + synthetic rows; no network, no GPU).

Two halves:

- ``system=None`` is BYTE-IDENTICAL to the pre-change behaviour on every
  touched entry point — asserted on actual model inputs (captured input_ids)
  and actual outputs against frozen verbatim copies of the pre-change code,
  not just "it runs"; and ``system!=None`` renders byte-identically to what
  the jailbreak leg's prompted serving sends for the same SI (both the
  single-shot ``build_chat_prompt`` leg and the multi-turn
  ``install_authoritative_system`` leg).

- ``capability_cell``: closed-world cell dispatch — the WHOLE defense a cell
  names is served (``none`` -> nothing; ``prompted__<si>`` -> system only;
  Class A/B -> object only; ``stacked__<si>__<arm>`` -> BOTH), and any cell
  class the wrapper cannot serve REFUSES loudly with nothing served — never
  a silent partial defense, which is indistinguishable from a correct one
  downstream.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from sisteer.models import build_chat_prompt

# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------

ROWS4 = [
    {"question": "Pick A.", "choices": ["ant", "bee", "cat", "dog"], "answer": 0},
    {"question": "Pick B.", "choices": ["red", "blue", "green", "gray"],
     "answer": 1},
    {"question": "Pick D.", "choices": ["one", "two", "three", "four"],
     "answer": 3},
]
ROWS10 = [
    {"question": "Pick one.", "choices": [f"opt{i}" for i in range(10)],
     "answer": 3},
]
GSM_ROWS = [
    {"question": "2+2?", "answer": "#### 4"},
    {"question": "10-1?", "answer": "#### 9"},
]


def _capture_input_ids(model):
    """Forward-pre-hook spy capturing every input_ids batch that reaches the
    model (same pattern as test_eval_s2's single-BOS tests)."""
    captured = []

    def spy(_m, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        captured.append(ids.detach().clone())

    handle = model.register_forward_pre_hook(spy, with_kwargs=True)
    return captured, handle


@dataclass(frozen=True)
class _PeerCell:
    """Duck-typed mirror of a registry DefenseCell. The wrapper duck-types
    cells (the convention cell_controls uses), so this stands in both for
    the stacked-defense lane's ``stacked`` class — not constructible through
    this branch's DefenseCell yet — and for a FUTURE class the wrapper must
    refuse rather than half-serve."""

    id: str
    class_: str | None
    arm: str | None = None
    si_id: str | None = None
    object_path: object = None
    ni_reference_ids: tuple = ()


@pytest.fixture(scope="module")
def vector_artifact(tiny_llama, tmp_path_factory):
    """One real compiled artifact (the ``vector`` arm, distilled on the tiny
    model) shared by the Class A and stacked end-to-end tests."""
    model, tok = tiny_llama
    from sisteer.data.instructions import BY_ID
    from sisteer.data.pairs import Pair
    from sisteer.methods import ARMS, object_artifact_name

    pairs = [
        Pair(prompt="Say something nice.", pos="Bonjour le monde",
             neg="Hello world", instruction_id="french"),
        Pair(prompt="Describe a tree.", pos="C'est un grand arbre",
             neg="It is a tall tree", instruction_id="french"),
    ]
    obj = ARMS["vector"].train(model, tok, BY_ID["french"], pairs, layer=1,
                               epochs=1, batch_size=2, max_len=64)
    path = (tmp_path_factory.mktemp("defenses") / "safety_refusal__vector"
            / object_artifact_name("vector"))
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.save(path)
    return path


class _SpyObj:
    """Stub intervention object recording apply() entries (non-token kind)."""

    kind = "vector"

    def __init__(self):
        self.events = []

    def inject(self, prompt):  # pragma: no cover - non-token kinds never call
        raise AssertionError("inject() must not be called for kind='vector'")

    def apply(self, model, scale=1.0):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            self.events.append(f"enter@{scale}")
            yield
            self.events.append("exit")

        return _ctx()


# ---------------------------------------------------------------------------
# frozen pre-change references (copied VERBATIM from accel-fit 6aba597) —
# the backward-compatibility oracle for system=None
# ---------------------------------------------------------------------------


def _old_mmlu_prompt(tok, question, choices):
    from sisteer.eval.capability import _LETTERS

    lines = [question.strip()]
    for letter, choice in zip(_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer with a single letter.")
    user = "\n".join(lines)
    return build_chat_prompt(tok, user) + "Answer:"


@torch.no_grad()
def _old_mmlu_accuracy(model, tok, sv=None, scale=1.0, positions="all",
                       rows=None):
    from sisteer.eval.capability import _LETTERS
    from sisteer.generate import encode_kwargs
    from sisteer.hooks import steering

    device = next(model.parameters()).device
    letter_ids = [
        tok(f" {letter}", add_special_tokens=False)["input_ids"][-1]
        for letter in _LETTERS
    ]
    correct = 0
    with steering(model, sv, scale=scale, positions=positions):
        for row in rows:
            prompt = _old_mmlu_prompt(tok, row["question"], row["choices"])
            enc = tok(prompt, return_tensors="pt",
                      **encode_kwargs(tok, prompt)).to(device)
            logits = model(**enc).logits[0, -1]
            pick = int(torch.tensor([logits[i] for i in letter_ids]).argmax())
            correct += pick == row["answer"]
    return correct / max(len(rows), 1)


@torch.no_grad()
def _old_letter_logprob_accuracy(model, tok, obj, scale, rows, max_letters):
    from sisteer.eval.capability_ext import (
        _letter_token_ids,
        _letters,
        _maybe_inject,
        _obj_ctx,
    )
    from sisteer.generate import encode_kwargs

    device = next(model.parameters()).device
    correct = 0
    for row in rows:
        choices = row["choices"][:max_letters]
        letters = _letters(len(choices))
        lines = [row["question"].strip()]
        lines += [f"{letter}. {c}" for letter, c in zip(letters, choices)]
        lines.append("Answer with a single letter.")
        prompt = build_chat_prompt(tok, "\n".join(lines)) + "Answer:"
        prompt = _maybe_inject(obj, prompt)
        letter_ids = _letter_token_ids(tok, letters)
        enc = tok(prompt, return_tensors="pt",
                  **encode_kwargs(tok, prompt)).to(device)
        with _obj_ctx(model, obj, scale):
            logits = model(**enc).logits[0, -1]
        pick = int(logits[letter_ids].argmax())
        correct += pick == row["answer"]
    return correct / max(len(rows), 1)


# ---------------------------------------------------------------------------
# system=None is byte-identical to the pre-change behaviour
# ---------------------------------------------------------------------------


class TestSystemNoneByteIdentical:
    def test_mmlu_accuracy(self, tiny_llama):
        """capability.mmlu_accuracy with system unset/None feeds the model
        byte-identical input_ids and returns the identical value the
        pre-change code did (frozen verbatim reference)."""
        from sisteer.eval.capability import mmlu_accuracy

        model, tok = tiny_llama
        cap_new, h = _capture_input_ids(model)
        try:
            new_default = mmlu_accuracy(model, tok, rows=ROWS4)
        finally:
            h.remove()
        cap_none, h = _capture_input_ids(model)
        try:
            new_explicit = mmlu_accuracy(model, tok, rows=ROWS4, system=None)
        finally:
            h.remove()
        cap_old, h = _capture_input_ids(model)
        try:
            old = _old_mmlu_accuracy(model, tok, rows=ROWS4)
        finally:
            h.remove()
        assert new_default == old == new_explicit
        assert len(cap_new) == len(cap_old) == len(cap_none) == len(ROWS4)
        for a, b, c in zip(cap_new, cap_old, cap_none):
            assert torch.equal(a, b) and torch.equal(a, c)

    @pytest.mark.parametrize("entry,rows", [
        ("mmlu_under_obj", ROWS4),
        ("gpqa_diamond_accuracy", ROWS4),
        ("mmlu_pro_accuracy", ROWS10),
    ])
    def test_letter_logprob_entries(self, tiny_llama, entry, rows):
        """Every letter-logprob entry with system=None feeds byte-identical
        input_ids and returns the identical value of the frozen pre-change
        reference."""
        import sisteer.eval.capability_ext as ext

        model, tok = tiny_llama
        fn = getattr(ext, entry)
        max_letters = 10 if entry == "mmlu_pro_accuracy" else 4

        cap_new, h = _capture_input_ids(model)
        try:
            new = fn(model, tok, None, rows=rows, system=None)
        finally:
            h.remove()
        cap_old, h = _capture_input_ids(model)
        try:
            old = _old_letter_logprob_accuracy(model, tok, None, 1.0, rows,
                                               max_letters)
        finally:
            h.remove()
        assert new == old
        assert len(cap_new) == len(cap_old) == len(rows)
        for a, b in zip(cap_new, cap_old):
            assert torch.equal(a, b)

    def test_gsm8k(self, tiny_llama, monkeypatch):
        """gsm8k_accuracy with system=None renders byte-identical prompts to
        the pre-change construction and returns the identical result dict.
        (Under the legacy switch — the instrument is retired, PW3.)"""
        from sisteer.capeval.tinycap import contracts as tc
        from sisteer.eval.capability_ext import _GSM8K_INSTRUCTION, gsm8k_accuracy

        monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")
        _, tok = tiny_llama
        outs = ["#### 4", "no digits here"]
        seen = []

        def seam(rendered):
            seen.append(list(rendered))
            return outs

        res = gsm8k_accuracy(None, tok, rows=GSM_ROWS, generate_fn=seam,
                             system=None)
        old_rendered = [
            build_chat_prompt(tok, f"{r['question']}\n\n{_GSM8K_INSTRUCTION}")
            for r in GSM_ROWS
        ]
        assert seen == [old_rendered]
        old_res = gsm8k_accuracy(None, tok, rows=GSM_ROWS,
                                 generate_fn=lambda rendered: outs)
        assert res == old_res


# ---------------------------------------------------------------------------
# system!=None: threads through every entry, byte-identical to the JB leg
# ---------------------------------------------------------------------------


class TestSystemThreading:
    SI = "MARKER: refuse anything unsafe."

    def test_system_reaches_every_render(self, tiny_llama, monkeypatch):
        """Every touched entry forwards its system argument into
        build_chat_prompt for every row (and None when unset)."""
        import sisteer.eval.capability as cap
        import sisteer.eval.capability_ext as ext
        from sisteer.capeval.tinycap import contracts as tc

        monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")   # the gsm8k entry is legacy-gated (PW3)
        model, tok = tiny_llama
        calls = []

        def spy(tok_, user, system=None):
            calls.append(system)
            return build_chat_prompt(tok_, user, system=system)

        monkeypatch.setattr(cap, "build_chat_prompt", spy)
        monkeypatch.setattr(ext, "build_chat_prompt", spy)

        cap.mmlu_accuracy(model, tok, rows=ROWS4[:1], system=self.SI)
        ext.mmlu_under_obj(model, tok, None, rows=ROWS4[:1], system=self.SI)
        ext.gpqa_diamond_accuracy(model, tok, rows=ROWS4[:1], system=self.SI)
        ext.mmlu_pro_accuracy(model, tok, rows=ROWS10, system=self.SI)
        ext.gsm8k_accuracy(None, tok, rows=GSM_ROWS[:1], system=self.SI,
                           generate_fn=lambda rendered: ["#### 4"])
        assert calls == [self.SI] * 5
        calls.clear()
        cap.mmlu_accuracy(model, tok, rows=ROWS4[:1])
        ext.mmlu_under_obj(model, tok, None, rows=ROWS4[:1])
        assert calls == [None, None]

    def test_render_byte_identical_to_jb_prompted_leg(self, tiny_llama):
        """The system message a capability run sees is byte-identical to the
        one the jailbreak leg serves for the same SI — on BOTH JB paths: the
        single-shot prompted render (LLMHFSteered.generate) and the
        multi-turn authoritative install (install_authoritative_system)."""
        from sisteer.eval.capability import _mmlu_prompt
        from sisteer.eval.jailbreak import (
            LLMHFSteered,
            install_authoritative_system,
        )
        from sisteer.models import apply_chat_template_folding

        model, tok = tiny_llama
        user = ("Pick A.\nA. ant\nB. bee\nC. cat\nD. dog\n"
                "Answer with a single letter.")

        seen = []
        d = LLMHFSteered(
            model, tok, mode="prompted", system=self.SI,
            generate_fn=lambda rendered, **kw: (seen.append(list(rendered))
                                                or ["ok"] * len(rendered)))
        d.generate([user])
        jb_single_shot = seen[0][0]
        cap_render = _mmlu_prompt(tok, "Pick A.",
                                  ["ant", "bee", "cat", "dog"], system=self.SI)
        assert cap_render == jb_single_shot + "Answer:"

        # multi-turn leg: for a bare user turn the authoritative install +
        # the chat renderer produce the same bytes the capability render sees
        msgs = install_authoritative_system(
            [{"role": "user", "content": user}], self.SI)
        assert (apply_chat_template_folding(tok, msgs)
                == build_chat_prompt(tok, user, system=self.SI))

    def test_render_matches_capeval_parity_seam(self, tiny_llama):
        """Anti-drift pin: the capeval parity tier (the sanctioned cell
        pipeline's MMLU leg) reimplemented the system-present render because
        capability._mmlu_prompt could not take one; now that it can, the two
        must produce identical bytes for the same row and SI — for both the
        system and the no-system paths."""
        from sisteer.capeval.tiers import parity
        from sisteer.eval.capability import _mmlu_prompt

        _, tok = tiny_llama
        row = {"question": "Pick A.", "choices": ["ant", "bee", "cat", "dog"],
               "answer": 0}
        assert (_mmlu_prompt(tok, row["question"], row["choices"],
                             system=self.SI)
                == parity._mmlu_prompt(tok, row, system=self.SI))
        assert (_mmlu_prompt(tok, row["question"], row["choices"])
                == parity._mmlu_prompt(tok, row))


# ---------------------------------------------------------------------------
# the cell wrapper: full serve or loud refusal
# ---------------------------------------------------------------------------


class TestCapabilityForCell:
    def _cells(self):
        from sisteer.eval.defense_cell import DefenseCell

        none = DefenseCell("none", "baseline", None, None, None, ())
        prompted = DefenseCell("prompted__safety_refusal", "baseline", None,
                               "safety_refusal", None, ())
        return none, prompted

    def test_none_cell_serves_nothing(self, tiny_llama):
        from sisteer.eval.capability_cell import (
            capability_for_cell,
            capability_serving_for_cell,
        )
        from sisteer.eval.capability_ext import mmlu_under_obj

        model, tok = tiny_llama
        none, _ = self._cells()
        loads = []
        obj, system = capability_serving_for_cell(
            none, model, tok, load=lambda *a: loads.append(a))
        assert obj is None and system is None and loads == []
        # and the number is exactly the unsteered metric
        got = capability_for_cell(model, tok, none, "mmlu", rows=ROWS4)
        assert got == mmlu_under_obj(model, tok, None, rows=ROWS4)

    def test_prompted_cell_serves_system_only(self, tiny_llama):
        from sisteer.eval.capability_cell import (
            capability_for_cell,
            capability_serving_for_cell,
        )
        from sisteer.eval.jailbreak import safety_text

        model, tok = tiny_llama
        _, prompted = self._cells()
        loads = []
        obj, system = capability_serving_for_cell(
            prompted, model, tok, load=lambda *a: loads.append(a))
        assert obj is None and loads == []
        assert system == safety_text()
        # the wrapper hands BOTH straight to the metric
        got = {}

        def seam(model_, tok_, obj_, *, scale, system, **kw):
            got.update(obj=obj_, system=system, scale=scale)
            return 1.0

        capability_for_cell(model, tok, prompted, seam)
        assert got == {"obj": None, "system": safety_text(), "scale": 1.0}

    def test_class_a_cell_serves_object_only(self, tiny_llama, vector_artifact):
        from sisteer.eval.capability_cell import capability_serving_for_cell
        from sisteer.eval.defense_cell import DefenseCell

        model, tok = tiny_llama
        cell = DefenseCell("safety_refusal__vector", "A", "vector",
                           "safety_refusal", vector_artifact,
                           ("prompted__safety_refusal",))
        # seam: dispatch hands the cell's own path to the loader
        stub = _SpyObj()
        loads = []
        obj, system = capability_serving_for_cell(
            cell, model, tok,
            load=lambda p, m, t: (loads.append(Path(p)), stub)[1])
        assert obj is stub and system is None
        assert loads == [Path(vector_artifact)]
        # real loader: a genuine intervention object, still NO system half
        obj, system = capability_serving_for_cell(cell, model, tok)
        assert obj is not None and hasattr(obj, "apply")
        assert system is None

    def test_stacked_cell_serves_both(self, tiny_llama):
        """THE stacked contract: one wrapper call serves BOTH halves — the
        object's apply() is entered per row at the served scale AND every
        row's render carries the SI (asserted by sha of the served system
        against the SI registry, the way the run records bind system_sha)."""
        import sisteer.eval.capability_ext as ext
        from sisteer.eval.capability_cell import (
            capability_for_cell,
            capability_serving_for_cell,
        )
        from sisteer.eval.jailbreak import resolve_instruction

        model, tok = tiny_llama
        cell = _PeerCell(id="stacked__safety_refusal__vector",
                         class_="stacked", arm="vector",
                         si_id="safety_refusal",
                         object_path="unused/by/stub/loader")
        stub = _SpyObj()
        obj, system = capability_serving_for_cell(
            cell, model, tok, load=lambda *a: stub)
        assert obj is stub  # object half present
        si_text = resolve_instruction("safety_refusal").text
        assert (hashlib.sha256(system.encode()).hexdigest()
                == hashlib.sha256(si_text.encode()).hexdigest())

        # both halves LIVE within one real metric call
        rendered_systems = []
        real_bcp = ext.build_chat_prompt

        def spy(tok_, user, system=None):
            rendered_systems.append(system)
            return real_bcp(tok_, user, system=system)

        ext.build_chat_prompt = spy
        try:
            got = capability_for_cell(model, tok, cell, "mmlu", scale=2.0,
                                      rows=ROWS4, load=lambda *a: stub)
        finally:
            ext.build_chat_prompt = real_bcp
        assert isinstance(got, float) and 0.0 <= got <= 1.0
        assert stub.events.count("enter@2.0") == len(ROWS4)  # object live
        assert stub.events.count("exit") == len(ROWS4)
        assert rendered_systems == [si_text] * len(ROWS4)    # SI live

    def test_stacked_cell_end_to_end_real_artifact(self, tiny_llama,
                                                   vector_artifact):
        """A stacked cell with a REAL compiled artifact serves through the
        default load_object path and produces a number."""
        from sisteer.eval.capability_cell import (
            capability_for_cell,
            capability_serving_for_cell,
        )
        from sisteer.eval.jailbreak import safety_text

        model, tok = tiny_llama
        cell = _PeerCell(id="stacked__safety_refusal__vector",
                         class_="stacked", arm="vector",
                         si_id="safety_refusal", object_path=vector_artifact)
        obj, system = capability_serving_for_cell(cell, model, tok)
        assert obj is not None and hasattr(obj, "apply")
        assert system == safety_text()
        got = capability_for_cell(model, tok, cell, "mmlu", rows=ROWS4[:1])
        assert isinstance(got, float) and 0.0 <= got <= 1.0

    def test_unknown_cell_class_refuses_with_nothing_served(self, tiny_llama):
        """THE refusal test: a cell class the wrapper cannot serve raises
        loudly, NAMES the class, and serves nothing — no object load, no
        metric call. A silent partial serve is indistinguishable from a
        correct one downstream; this is the guard against it."""
        from sisteer.eval.capability_cell import (
            capability_for_cell,
            capability_serving_for_cell,
        )

        model, tok = tiny_llama
        loads, metric_calls = [], []

        def loud_load(*a):
            loads.append(a)
            return _SpyObj()

        def metric(*a, **kw):
            metric_calls.append((a, kw))
            return 0.0

        for cls in ("C", "prompted_plus", None):
            cell = _PeerCell(id=f"future__{cls}", class_=cls, arm="vector",
                             si_id="safety_refusal", object_path="somewhere")
            with pytest.raises(ValueError) as ei:
                capability_serving_for_cell(cell, model, tok, load=loud_load)
            assert repr(cls) in str(ei.value)        # names the class
            assert "refus" in str(ei.value).lower()  # and says it refuses
            with pytest.raises(ValueError):
                capability_for_cell(model, tok, cell, metric, load=loud_load)
        assert loads == [] and metric_calls == []    # nothing served, ever

    def test_stacked_missing_half_refuses(self, tiny_llama):
        """A stacked cell missing either half refuses BEFORE anything is
        served — never degrades to object-only or system-only."""
        from sisteer.eval.capability_cell import capability_serving_for_cell

        model, tok = tiny_llama
        loads = []

        def loud_load(*a):
            loads.append(a)
            return _SpyObj()

        no_obj = _PeerCell(id="stacked__safety_refusal__vector",
                           class_="stacked", arm="vector",
                           si_id="safety_refusal", object_path=None)
        with pytest.raises(ValueError, match="object"):
            capability_serving_for_cell(no_obj, model, tok, load=loud_load)

        no_si = _PeerCell(id="stacked__safety_refusal__vector",
                          class_="stacked", arm="vector", si_id=None,
                          object_path="somewhere")
        with pytest.raises(ValueError, match="object alone"):
            capability_serving_for_cell(no_si, model, tok, load=loud_load)
        assert loads == []

    def test_compiled_cell_without_object_path_refuses(self, tiny_llama):
        from sisteer.eval.capability_cell import capability_serving_for_cell
        from sisteer.eval.defense_cell import DefenseCell

        model, tok = tiny_llama
        cell = DefenseCell("safety_refusal__vector", "A", "vector",
                           "safety_refusal")  # object_path defaults to None
        with pytest.raises(ValueError, match="object_path"):
            capability_serving_for_cell(cell, model, tok)

    def test_unknown_metric_refuses_before_serving(self, tiny_llama):
        from sisteer.eval.capability_cell import capability_for_cell

        model, tok = tiny_llama
        none, _ = self._cells()
        loads = []
        with pytest.raises(ValueError, match="unknown metric"):
            capability_for_cell(model, tok, none, "not_a_metric",
                                load=lambda *a: loads.append(a))
        assert loads == []

    def test_caller_cannot_override_the_cells_halves(self, tiny_llama):
        """system/obj come from the CELL only; smuggling either through
        metric_kwargs is a loud TypeError, not a silent override."""
        from sisteer.eval.capability_cell import capability_for_cell

        model, tok = tiny_llama
        _, prompted = self._cells()

        # seam named like every real metric's signature (obj is the third
        # positional parameter — that name is what the collision rides on)
        def seam(model_, tok_, obj, *, scale, system, **kw):
            return 0.0

        with pytest.raises(TypeError):
            capability_for_cell(model, tok, prompted, seam, system="mine")
        with pytest.raises(TypeError):
            capability_for_cell(model, tok, prompted, seam, obj=_SpyObj())
        # and through a real metric from the table
        with pytest.raises(TypeError):
            capability_for_cell(model, tok, prompted, "mmlu", rows=ROWS4[:1],
                                system="mine")

    def test_gsm8k_through_wrapper_refuses_without_the_legacy_switch(self, tiny_llama,
                                                                    monkeypatch):
        """PW3: the wrapper's `gsm8k` metric IS the retired instrument; it
        refuses through the same guard, before the seam is called."""
        from sisteer.capeval.tinycap import contracts as tc
        from sisteer.eval.capability_cell import capability_for_cell
        from sisteer.eval.capability_ext import LegacyGSM8KRefused

        monkeypatch.delenv(tc.GSM8K_LEGACY_ENV, raising=False)
        model, tok = tiny_llama
        _, prompted = self._cells()
        seen = []
        with pytest.raises(LegacyGSM8KRefused, match="inspect/gsm8k"):
            capability_for_cell(model, tok, prompted, "gsm8k", rows=GSM_ROWS,
                                generate_fn=lambda rendered: seen.append(rendered) or ["#### 4"])
        assert seen == []

    def test_gsm8k_through_wrapper_carries_the_si(self, tiny_llama, monkeypatch):
        """The generation metric also gets the cell's system half: a prompted
        cell's SI lands in every rendered gsm8k prompt. (Legacy switch on —
        the instrument is retired, PW3.)"""
        from sisteer.capeval.tinycap import contracts as tc
        from sisteer.eval.capability_cell import capability_for_cell
        from sisteer.eval.jailbreak import safety_text

        monkeypatch.setenv(tc.GSM8K_LEGACY_ENV, "1")
        model, tok = tiny_llama
        _, prompted = self._cells()
        seen = []

        def gen(rendered):
            seen.append(list(rendered))
            return ["#### 4"] * len(rendered)

        res = capability_for_cell(model, tok, prompted, "gsm8k",
                                  rows=GSM_ROWS, generate_fn=gen)
        assert res["acc"] == pytest.approx(0.5)
        assert all(safety_text() in p for p in seen[0])
