"""The chat TEMPLATE is guarded by hash, and the serving identity is a stamp.

Ported from the psr-hard-mse-arms lane (commit "Guard the chat TEMPLATE by
hash, not just the model id") for the NousResearch re-run: the meta-llama
Llama-3.1-8B-Instruct template silently injects a ~25-token default system
block ("Cutting Knowledge Date / Today Date") even when no system message is
passed, so a no-system-prompt arm served under it is NOT unprompted, and any
comparison spanning the two mirrors' templates carries a silent prompt
confound. That already cost a headline result in this lane. The HASH is the
real invariant, not the model id — a differently-named mirror carrying the
injecting template walks straight past a name check.
"""

import warnings

import pytest

from sisteer.models import (
    _TEMPLATE_HASH_METALLAMA,
    _TEMPLATE_HASH_NOUSRESEARCH,
    build_chat_prompt,
    chat_template_hash,
    check_chat_template,
    serving_identity,
)

NSR_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
NSR_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
META_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
META_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"


class _Tok:
    def __init__(self, tmpl, name_or_path="stub/model"):
        self.chat_template = tmpl
        self.name_or_path = name_or_path


def test_chat_template_hash_guard_catches_the_injecting_template():
    """The template HASH is the real invariant, not the model id.

    A name check passes any mirror under a different name, including one
    carrying the template that injects a ~25-token default system block.
    The petri lane found the same defect from the opposite direction by
    freezing chat_template_hash, so the two hashes below are independently
    confirmed.
    """
    assert _TEMPLATE_HASH_NOUSRESEARCH != _TEMPLATE_HASH_METALLAMA

    assert chat_template_hash(_Tok(None)) is None, "no template -> no hash"
    # a tokenizer with an unknown template is not flagged (only the known-bad
    # one is) — this can never become a nuisance warning on other models
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_chat_template(_Tok("{{ messages }}"))
    assert not w, "an unrelated template must not warn"


def _pinned_tok(model, revision):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=True)
    except Exception as e:  # cache miss, offline resolution failure, ...
        pytest.skip(f"pinned {model} tokenizer @ {revision[:8]} not in the "
                    f"local HF cache ({type(e).__name__})")


def test_the_two_mirrors_templates_actually_diverge():
    """The measured divergence the re-run exists to fix, pinned end to end:
    the two mirrors' template hashes are the recorded constants, and the SAME
    no-system prompt renders 25 tokens heavier through meta-llama's template
    (43 vs 18 through build_chat_prompt) — the injected default system block.
    """
    nsr = _pinned_tok(NSR_MODEL, NSR_REVISION)
    meta = _pinned_tok(META_MODEL, META_REVISION)
    assert chat_template_hash(nsr) == _TEMPLATE_HASH_NOUSRESEARCH
    assert chat_template_hash(meta) == _TEMPLATE_HASH_METALLAMA

    user = "What is the capital of France?"
    n_nsr = len(nsr(build_chat_prompt(nsr, user),
                    add_special_tokens=True)["input_ids"])
    n_meta = len(meta(build_chat_prompt(meta, user),
                      add_special_tokens=True)["input_ids"])
    assert (n_meta, n_nsr) == (43, 18), (
        "the ~25-token injected system block moved — re-derive every "
        "template-overhead number in configs/two_si.yaml")

    # the guard FIRES on the real injecting template ...
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = check_chat_template(meta)
    assert h == _TEMPLATE_HASH_METALLAMA
    assert any("INJECTS" in str(x.message) for x in w)
    with pytest.raises(ValueError, match="INJECTS"):
        check_chat_template(meta, strict=True)
    # ... and stays silent on the clean mirror
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = check_chat_template(nsr)
    assert h == _TEMPLATE_HASH_NOUSRESEARCH
    assert not w


def test_serving_identity_stamps_requested_and_resolved_and_hash():
    """serving_identity measures the LIVE tokenizer/model — requested pin
    from the config, resolved id/revision from what transformers actually
    loaded, template hash from the tokenizer — the stamp every cell manifest
    and controls_summary carries."""

    class _Cfg:
        pass

    class _MCfg:
        _name_or_path = "resolved/model-id"
        _commit_hash = "feedface" * 5

    class _Model:
        config = _MCfg()

    cfg = {"model": "requested/model-id", "model_revision": "aa" * 20}
    ident = serving_identity(_Tok("{{ messages }}", name_or_path="tok/path"),
                             cfg, model=_Model())
    assert ident == {
        "requested_model": "requested/model-id",
        "requested_revision": "aa" * 20,
        "resolved_model_id": "resolved/model-id",
        "resolved_revision": "feedface" * 5,
        "chat_template_hash": chat_template_hash(_Tok("{{ messages }}")),
    }
    # tokenizer-only fallback (no model handed over): id from the tokenizer
    ident = serving_identity(_Tok(None, name_or_path="tok/path"), cfg)
    assert ident["resolved_model_id"] == "tok/path"
    assert ident["resolved_revision"] is None
    assert ident["chat_template_hash"] is None
