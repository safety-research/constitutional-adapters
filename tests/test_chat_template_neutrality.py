"""A chat template must not smuggle its own system directive into the pool.

Qwen3.8's template prepends "Reasoning effort is set to xhigh. Please think
carefully through the task..." AHEAD of the caller's system content and opens
a <think> block. Rendered that way:

  * the `none` condition is not undefended, it is thinking-instructed;
  * every SI sits on top of a second, stronger directive it did not ask for,
    confounding the SI contrast the pool exists to measure;
  * responses carry reasoning traces, so the grader scores a different object
    than it does for the other models in the same campaign.

Detected by PROBE rather than by model-id match, so a model that stops needing
the workaround stops getting it.
"""
import pytest

from sisteer.sampling.engine import (build_chat, template_extra_kwargs,
                                     _injects_extra_system, _NEUTRALITY_PROBE)


class _FakeTok:
    """Minimal stand-in: injects a preamble unless enable_thinking is False."""
    def __init__(self, injects=True, honours_flag=True):
        self.injects, self.honours_flag = injects, honours_flag

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True, **kw):
        sys_txt = next((m["content"] for m in msgs
                        if m["role"] == "system"), "")
        pre = ""
        if self.injects and not (self.honours_flag
                                 and kw.get("enable_thinking") is False):
            pre = "Reasoning effort is set to xhigh. Think carefully.\n\n"
        return f"<|im_start|>system\n{pre}{sys_txt}<|im_end|>\n"


def test_injecting_template_is_detected_and_neutralised():
    assert template_extra_kwargs(_FakeTok()) == {"enable_thinking": False}


def test_clean_template_gets_no_kwargs():
    """Llama and Gemma must render byte-identically to before."""
    assert template_extra_kwargs(_FakeTok(injects=False)) == {}


def test_template_that_ignores_the_flag_gets_no_false_promise():
    """If the escape hatch does not work, return {} rather than a kwarg that
    silently does nothing -- a workaround that does not work is worse than
    none, because it reads as handled."""
    assert template_extra_kwargs(_FakeTok(honours_flag=False)) == {}


def test_build_chat_applies_the_extra():
    tok = _FakeTok()
    dirty = build_chat(tok, "SI", "u")
    clean = build_chat(tok, "SI", "u", template_extra_kwargs(tok))
    assert "Reasoning effort" in dirty
    assert "Reasoning effort" not in clean


def test_probe_ignores_role_scaffolding_only():
    assert not _injects_extra_system(
        f"<|im_start|>system\n{_NEUTRALITY_PROBE}")
    assert _injects_extra_system(
        f"<|im_start|>system\nExtra prose.\n{_NEUTRALITY_PROBE}")
