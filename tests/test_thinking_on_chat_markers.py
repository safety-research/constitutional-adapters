"""The THINKING-ON header derivation and its per-template CERTIFICATION
(``chat_markers.thinking_on_certification`` / ``generation_header_ids_under``
/ ``thinking_header_facts[_or_error]`` — docs/served_thinking_on.md), on the
three wave templates.

Two layers of evidence, both offline:

* the REAL vendored chat templates (``tests/fixtures/chat_templates/*.jinja``
  — the pinned revisions' public template files, byte for byte) rendered
  through transformers' own jinja environment by a marker-aware regex
  tokenizer (:class:`TemplateTok`): control tokens are one id each, other
  text splits on ``\\S+|\\s+`` so ``\\n`` and ``\\n\\n`` are distinct tokens —
  the merge that made the OFF probe refuse Qwen ON is reproduced
  structurally. This pins the header STRUCTURE (decoded text), the moves
  verdict, the OFF-absent-in-ON fact and gemma's unchanged header, whatever
  the real vocabulary's ids;
* the REAL pinned tokenizers, cache-gated: run when the pinned revision is
  in the local HF cache, or under ``SISTEER_TEST_TOKENIZER_ROOT`` (a dir of
  ``<org>--<model>/<revision>/`` tokenizer files), else skipped — never
  fetched. These pin the ids and lengths the docs table records.

Plus the OFF derivation's byte-identity: ``generation_header_ids_under(tok,
enable_thinking=False)`` IS ``generation_header_ids`` under the OFF pin, and
every derivation leaves the tokenizer's own pin exactly as it found it.
"""

from __future__ import annotations

import os
import re
import types
from pathlib import Path

import pytest

from sisteer import chat_markers as cmk
from sisteer.models import pin_reasoning_template_off

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "chat_templates"

#: model key -> (fixture, markers, control markers, bos token, HF id, revision)
TEMPLATES = {
    "qwen35_4b": ("qwen35_4b@851bf6e8.jinja",
                  ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
                  {"<|im_start|>", "<|im_end|>"}, None,
                  "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    "gemma4e4b": ("gemma4e4b@ee0ef602.jinja",
                  ["<bos>", "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>"],
                  {"<bos>", "<|turn>", "<turn|>"}, "<bos>",
                  "google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
    "nemotron3nano4b": ("nemotron3nano4b@dfaf35de.jinja",
                        ["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
                        {"<|im_start|>", "<|im_end|>"}, None,
                        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                        "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"),
}

#: the header TEXT each template renders under each regime (from the real
#: pinned tokenizers, 2026-09-09; the docs table) — what the template fakes
#: must reproduce
OFF_TEXT = {
    "qwen35_4b": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "gemma4e4b": "<|turn>model\n",
    "nemotron3nano4b": "<|im_start|>assistant\n<think></think>",
}
ON_TEXT = {
    "qwen35_4b": "<|im_start|>assistant\n<think>\n",
    "gemma4e4b": "<|turn>model\n",
    "nemotron3nano4b": "<|im_start|>assistant\n<think>\n",
}
VERDICT = {"qwen35_4b": "certified", "gemma4e4b": "trivially_in_support",
           "nemotron3nano4b": "certified"}
#: the real pinned tokenizers' ids (docs/served_thinking_on.md)
REAL_OFF = {"qwen35_4b": (248045, 74455, 198, 248068, 271, 248069, 271),
            "gemma4e4b": (105, 4368, 107),
            "nemotron3nano4b": (10, 1503, 19464, 1010, 12, 13)}
REAL_ON = {"qwen35_4b": (248045, 74455, 198, 248068, 198),
           "gemma4e4b": (105, 4368, 107),
           "nemotron3nano4b": (10, 1503, 19464, 1010, 12, 1010)}


class TemplateTok:
    """A REAL chat template rendered by transformers' jinja environment,
    tokenized by a marker-aware regex (module docstring)."""

    def __init__(self, template: str, markers, control, bos_token=None):
        self.chat_template = template
        self.bos_token = bos_token
        self.eos_token = None
        self._tok2id: dict = {}
        self._id2tok: dict = {}
        self.added_tokens_decoder = {}
        self.all_special_ids: list = []
        for t in markers:
            tid = self._intern(t)
            self.added_tokens_decoder[tid] = types.SimpleNamespace(special=(t in control))
        self._re = re.compile("|".join(re.escape(t) for t in
                                       sorted(markers, key=len, reverse=True)))

    def _intern(self, piece):
        if piece not in self._tok2id:
            tid = len(self._tok2id)
            self._tok2id[piece] = tid
            self._id2tok[tid] = piece
        return self._tok2id[piece]

    def _encode(self, text):
        ids, pos = [], 0
        for m in self._re.finditer(text):
            for piece in re.findall(r"\S+|\s+", text[pos:m.start()]):
                ids.append(self._intern(piece))
            ids.append(self._intern(m.group()))
            pos = m.end()
        for piece in re.findall(r"\S+|\s+", text[pos:]):
            ids.append(self._intern(piece))
        return ids

    def decode(self, ids, **kw):
        return "".join(self._id2tok[i] for i in ids)

    def __call__(self, text, add_special_tokens=False, **kw):
        return {"input_ids": self._encode(text)}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kw):
        from transformers.utils.chat_template_utils import render_jinja_template
        out = render_jinja_template(conversations=[list(messages)],
                                    chat_template=self.chat_template,
                                    add_generation_prompt=add_generation_prompt,
                                    bos_token=self.bos_token or "",
                                    eos_token=self.eos_token or "", **kw)
        text = out
        while isinstance(text, (list, tuple)):
            text = text[0]
        return self._encode(text) if tokenize else text


def _template_tok(key) -> TemplateTok:
    pytest.importorskip("transformers")
    fixture, markers, control, bos, _model, _rev = TEMPLATES[key]
    tok = TemplateTok((FIXTURES / fixture).read_text(encoding="utf-8"), markers, control,
                      bos_token=bos)
    assert pin_reasoning_template_off(tok)   # every wave template carries the switch
    return tok


def _real_tok(key):
    """The REAL pinned tokenizer, from the local HF cache or
    ``SISTEER_TEST_TOKENIZER_ROOT`` — never fetched; skipped when absent."""
    transformers = pytest.importorskip("transformers")
    _fixture, _m, _c, _b, model, rev = TEMPLATES[key]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = os.environ.get("SISTEER_TEST_TOKENIZER_ROOT")
    candidates = []
    if root:
        candidates.append(Path(root) / model.replace("/", "--") / rev)
    for path in candidates:
        if path.is_dir():
            return transformers.AutoTokenizer.from_pretrained(str(path))
    try:
        return transformers.AutoTokenizer.from_pretrained(model, revision=rev,
                                                          local_files_only=True)
    except Exception as e:  # noqa: BLE001 — not cached here: skip, never fetch
        pytest.skip(f"pinned {model} tokenizer @ {rev[:8]} not in the local HF cache "
                    f"and SISTEER_TEST_TOKENIZER_ROOT unset ({type(e).__name__})")


# --------------------------------------------------------------------------- #
# the vendored templates: structure, verdicts, byte-identity of the OFF path    #
# --------------------------------------------------------------------------- #


def test_fixtures_are_the_pinned_templates():
    """The vendored files are the pinned revisions' public chat templates,
    pinned by sha256 prefix so a silent edit is a test failure."""
    import hashlib
    want = {"qwen35_4b@851bf6e8.jinja": "a4aee8afcf2e0711",
            "gemma4e4b@ee0ef602.jinja": "0a2c8073c878ab1d",
            "nemotron3nano4b@dfaf35de.jinja": "ab7813c3abdd9cb6"}
    for name, sha in want.items():
        got = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()[:16]
        assert got == sha, (name, got)
        assert "enable_thinking" in (FIXTURES / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_certification_on_the_real_template_structure(key):
    tok = _template_tok(key)
    before = dict(tok._sisteer_template_kwargs)
    cert = cmk.thinking_on_certification(tok)
    # the pin is restored exactly
    assert tok._sisteer_template_kwargs == before == {"enable_thinking": False}
    assert cert["switch"] is True
    assert cert["verdict"] == VERDICT[key], cert
    assert tok.decode(cert["off_ids"]) == OFF_TEXT[key]
    assert tok.decode(cert["on_ids"]) == ON_TEXT[key]
    assert cert["moves"] is (OFF_TEXT[key] != ON_TEXT[key])
    checks = cert["checks"]
    assert checks["clean_suffix"] is True and checks["tail"] is True
    # THE REQUIRED RULE: the ON prompt opens a completed THINKING turn
    assert checks["thinking_turn_prefix"] is True
    # the landing's plain-turn rule: passes everywhere but Nemotron-3, whose
    # empty-reasoning completed turn collapses to <think></think>
    assert checks["plain_turn_prefix"] is (key != "nemotron3nano4b")
    if key == "nemotron3nano4b":
        assert "<think></think>" in checks["plain_turn_reason"]
    if cert["moves"]:
        # the OFF header never occurs in the ON render (so the two regimes'
        # tails cannot be confused)
        assert checks["off_header_absent_in_on_render"] is True
    else:
        assert checks["off_header_absent_in_on_render"] is None
    # the OFF derivation is generation_header_ids under the OFF pin, exactly
    assert tuple(cert["off_ids"]) == tuple(cmk.generation_header_ids(tok))
    assert cmk.generation_header_ids_under(tok, enable_thinking=False) == tuple(cert["off_ids"])
    # TWO RULES, stated: the per-cell derivation (`generation_header_ids_under`, the
    # registry's header every worker cross-checks at preflight — the PLAIN-turn rule of the
    # fit-support-transfer landing) and the standup-wide certification (the THINKING-turn
    # rule, the plain-turn verdict recorded beside it). They agree wherever the plain-turn
    # rule holds; on Nemotron-3 the plain-turn derivation REFUSES (its empty-reasoning turn
    # collapses to <think></think>) while the certification passes — so a `--thinking-on`
    # standup on Nemotron serves an all-position wave under the certified fact, and a
    # header-dependent cell there refuses at the registry derivation, never by guesswork
    if checks["plain_turn_prefix"]:
        assert cmk.generation_header_ids_under(tok, enable_thinking=True) == tuple(cert["on_ids"])
    else:
        with pytest.raises(cmk.ChatMarkerError):
            cmk.generation_header_ids_under(tok, enable_thinking=True)


def test_qwen_structure_widths_match_the_real_tokenizer():
    """On Qwen3.5 the fake's token split reproduces the real widths: 7 OFF
    (an empty closed think block) vs 5 ON (an open <think>), sharing the
    first 4 ids; the merge (``\\n`` vs ``\\n\\n`` after <think>) is exactly
    what the OFF probe refused on."""
    tok = _template_tok("qwen35_4b")
    cert = cmk.thinking_on_certification(tok)
    assert len(cert["off_ids"]) == 7 and len(cert["on_ids"]) == 5
    assert cert["off_ids"][:4] == cert["on_ids"][:4]
    assert tok.decode([cert["off_ids"][4]]) == "\n\n" and tok.decode([cert["on_ids"][4]]) == "\n"


def test_gemma_on_render_adds_a_system_block_and_keeps_the_header():
    """gemma-4's switch inserts <|think|> into a LEADING SYSTEM block (a
    prompted cell's system text lands inside that same block); the
    generation header is unchanged -> trivially in support."""
    tok = _template_tok("gemma4e4b")
    on = {"enable_thinking": True}
    plain = tok.apply_chat_template([{"role": "user", "content": "Hi"}],
                                    add_generation_prompt=True, **on)
    assert plain.startswith("<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi")
    assert plain.endswith("<|turn>model\n")
    with_sys = tok.apply_chat_template(
        [{"role": "system", "content": "SYNTHETIC SYSTEM PREFIX"},
         {"role": "user", "content": "Hi"}], add_generation_prompt=True, **on)
    assert with_sys.startswith("<bos><|turn>system\n<|think|>\nSYNTHETIC SYSTEM PREFIX<turn|>\n")
    off = tok.apply_chat_template([{"role": "user", "content": "Hi"}],
                                  add_generation_prompt=True, enable_thinking=False)
    assert off == "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
    facts = cmk.thinking_header_facts(tok)
    assert facts["moves"] is False and facts["on_ids"] == facts["off_ids"]


def test_nemotron_plain_rule_flip_reproduces_the_landing_refusal(monkeypatch):
    """With the landing's rule REQUIRED, Nemotron-3 is refused (the
    fit-support-transfer verdict); the other two are unchanged."""
    monkeypatch.setattr(cmk, "THINKING_ON_REQUIRES_PLAIN_TURN_PREFIX", True)
    cert = cmk.thinking_on_certification(_template_tok("nemotron3nano4b"))
    assert cert["verdict"] == "refused" and cert["on_ids"] is None
    assert "plain_turn_prefix" in cert["reason"]
    assert cert["requires_plain_turn_prefix"] is True
    # the per-cell derivation IS the plain-turn rule on this tree (flag or no flag): it
    # refuses Nemotron with the probe's own words, and the non-fatal facts reader records
    # the PARTIAL fact with that reason (recorded, never replaced)
    with pytest.raises(cmk.ChatMarkerError, match="refusing to guess assistant markers"):
        cmk.generation_header_ids_under(_template_tok("nemotron3nano4b"), enable_thinking=True)
    facts, err = cmk.thinking_header_facts_or_error(_template_tok("nemotron3nano4b"))
    assert facts == {"switch": True, "moves": None, "on_ids": None,
                     "off_ids": list(cmk.generation_header_ids(_template_tok("nemotron3nano4b")))}
    assert err and err.startswith("ChatMarkerError")
    for key in ("qwen35_4b", "gemma4e4b"):
        assert cmk.thinking_on_certification(_template_tok(key))["verdict"] == VERDICT[key]


def test_facts_and_facts_or_error_agree_on_the_wave_templates():
    """The two readers agree on every template; on Nemotron-3 the hard reader
    REFUSES (the plain-turn rule) and the non-fatal one records the partial
    fact — while the standup-wide certification certifies it under the
    thinking-turn rule (test_certification_on_the_real_template_structure):
    the two rules' one divergence, stated here rather than hidden."""
    for key in TEMPLATES:
        tok = _template_tok(key)
        cert = cmk.thinking_on_certification(tok)
        if cert["checks"]["plain_turn_prefix"]:
            facts = cmk.thinking_header_facts(tok)
            assert cmk.thinking_header_facts_or_error(tok) == (facts, None)
            assert set(facts) == {"switch", "moves", "off_ids", "on_ids"}
            assert facts["moves"] is (facts["on_ids"] != facts["off_ids"])
            assert facts["on_ids"] == cert["on_ids"]
        else:
            assert key == "nemotron3nano4b"
            with pytest.raises(cmk.ChatMarkerError):
                cmk.thinking_header_facts(tok)
            partial, err = cmk.thinking_header_facts_or_error(tok)
            assert partial["on_ids"] is None and partial["moves"] is None and err
            assert cert["verdict"] == "certified"          # the thinking-turn rule's verdict


# --------------------------------------------------------------------------- #
# refusals and the pin discipline on synthetic tokenizers                      #
# --------------------------------------------------------------------------- #


class _NoSwitchTok:
    """A Llama-shaped tokenizer: no thinking switch at all."""

    chat_template = "{% for m in messages %}<h>{{ m.role }}</h>{{ m.content }}<e>{% endfor %}" \
                    "{% if add_generation_prompt %}<h>assistant</h>{% endif %}"

    def __init__(self):
        self._tok2id, self._id2tok = {}, {}
        self.added_tokens_decoder, self.all_special_ids = {}, []
        for t in ("<h>", "</h>", "<e>"):
            tid = self._intern(t)
            self.added_tokens_decoder[tid] = types.SimpleNamespace(special=True)
        self._re = re.compile(r"<h>|</h>|<e>")

    def _intern(self, p):
        if p not in self._tok2id:
            self._tok2id[p] = len(self._tok2id)
            self._id2tok[self._tok2id[p]] = p
        return self._tok2id[p]

    def _encode(self, text):
        ids, pos = [], 0
        for m in self._re.finditer(text):
            ids += [self._intern(x) for x in re.findall(r"\S+|\s+", text[pos:m.start()])]
            ids.append(self._intern(m.group()))
            pos = m.end()
        ids += [self._intern(x) for x in re.findall(r"\S+|\s+", text[pos:])]
        return ids

    def decode(self, ids, **kw):
        return "".join(self._id2tok[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        s = ""
        for m in messages:
            s += f"<h>{m['role']}</h>{m['content']}<e>"
        if add_generation_prompt:
            s += "<h>assistant</h>"
        return self._encode(s) if tokenize else s


def test_no_switch_template_has_no_on_render():
    tok = _NoSwitchTok()
    assert not pin_reasoning_template_off(tok)
    cert = cmk.thinking_on_certification(tok)
    assert cert["switch"] is False and cert["verdict"] == "no_switch"
    assert cert["on_ids"] is None and cert["moves"] is None
    assert tuple(cert["off_ids"]) == tuple(cmk.generation_header_ids(tok))
    with pytest.raises(cmk.ChatMarkerError, match="no thinking switch"):
        cmk.generation_header_ids_under(tok, enable_thinking=True)
    assert cmk.generation_header_ids_under(tok, enable_thinking=False) == tuple(cert["off_ids"])
    assert not hasattr(tok, "_sisteer_template_kwargs")   # the (absent) pin is restored
    assert cmk.thinking_header_facts(tok) == {"switch": False, "moves": None,
                                             "off_ids": list(cert["off_ids"]), "on_ids": None}
    with pytest.raises(cmk.ChatMarkerError, match="must be a bool"):
        cmk.generation_header_ids_under(tok, enable_thinking=1)  # type: ignore[arg-type]


def test_pin_is_restored_whatever_it_was():
    tok = _template_tok("qwen35_4b")
    for pin in ({"enable_thinking": False}, {"enable_thinking": True}, None):
        if pin is None:
            delattr(tok, "_sisteer_template_kwargs")
        else:
            tok._sisteer_template_kwargs = dict(pin)
        cmk.generation_header_ids_under(tok, enable_thinking=True)
        cmk.generation_header_ids_under(tok, enable_thinking=False)
        cmk.thinking_on_certification(tok)
        if pin is None:
            assert not hasattr(tok, "_sisteer_template_kwargs")
        else:
            assert tok._sisteer_template_kwargs == pin


# --------------------------------------------------------------------------- #
# the REAL pinned tokenizers (cache-gated)                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_real_tokenizer_certification(key):
    tok = _real_tok(key)
    pin_reasoning_template_off(tok)
    cert = cmk.thinking_on_certification(tok)
    assert cert["verdict"] == VERDICT[key], cert
    assert tuple(cert["off_ids"]) == REAL_OFF[key]
    assert tuple(cert["on_ids"]) == REAL_ON[key]
    assert cert["moves"] is (REAL_ON[key] != REAL_OFF[key])
    assert cert["checks"]["thinking_turn_prefix"] is True
    assert cert["checks"]["plain_turn_prefix"] is (key != "nemotron3nano4b")
    assert tok.decode(list(REAL_OFF[key])) == OFF_TEXT[key]
    assert tok.decode(list(REAL_ON[key])) == ON_TEXT[key]
    # the OFF path is untouched and the pin restored
    assert tuple(cmk.generation_header_ids(tok)) == REAL_OFF[key]
    assert tok._sisteer_template_kwargs == {"enable_thinking": False}
    # the worker's startup cross-check is main's PER-CELL derivation (the plain-turn
    # rule): it accepts the certified header where that rule derives it (Qwen3.5),
    # ADMITS an ON header equal to the OFF one (gemma-4: a header that does not
    # move is no reason not to serve), and REFUSES
    # Nemotron-3, whose ON header the plain-turn rule cannot derive — so a `--thinking-on`
    # standup there declares no header-dependent cell (the certification carries the
    # all-position wave; a declaring cell refuses before any GPU minute)
    from sisteer.capeval.serving import lens_steer as ls
    if not cert["checks"]["plain_turn_prefix"]:
        assert key == "nemotron3nano4b"
        with pytest.raises(ls.LensSteerError):
            ls.verify_assistant_header_thinking_on(tok, REAL_ON[key])
    else:
        ls.verify_assistant_header_thinking_on(tok, REAL_ON[key])
        if cert["moves"]:
            with pytest.raises(ls.LensSteerError, match="mismatch"):
                ls.verify_assistant_header_thinking_on(tok, REAL_OFF[key][:2] + (1,))
    ls.verify_assistant_header(tok, REAL_OFF[key])
