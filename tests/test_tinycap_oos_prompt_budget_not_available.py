"""The Llama paper rows — ``scripts/tinycap_oos_prompt_budget.py --arm not_available``:
the NO-SWITCH arm renders with NO ``chat_template_kwargs`` (the fake tokenizer records the call), the artifact's
``arm`` is ``not_available`` (so ``g8k_budget_artifact``'s arm equality holds) and its ``render`` block carries an
EMPTY kwargs map, the cap is the explicit ``--cap`` and NOTHING ELSE (no table applies: refused by name without
one), and the G-FIT sidecar arithmetic at the two GSM8K entries — 10,752 for the 8,192 pass, 18,944 for the rung —
is the same fit rule the switched families use. Synthetic messages; a fake tokenizer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_oos_prompt_budget_ifeval import _head_sha
from test_tinycap_oos_prompt_budget import SCRIPT, _FakeTok, _g8k_msgs, _load

from sisteer.capeval.tinycap import contracts as C

NA = C.THINKING_NOT_AVAILABLE


class _NoSwitchTok(_FakeTok):
    """A template with NO thinking switch: the header is the same width whatever kwargs come (none should), and
    every call's kwargs are recorded so the test can assert the render sent NONE."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        assert not kw, f"a no-switch render must pass NO chat_template_kwargs, got {kw!r}"
        self.calls.append({"messages": [dict(m) for m in messages], "add_generation_prompt": add_generation_prompt,
                           "tokenize": tokenize, "kwargs": dict(kw)})
        body = sum(len(m["content"]) for m in messages) // 4
        return list(range(body + 4))            # the 4-id assistant header


def _arm(tmp_path, monkeypatch):
    B = _load("tinycap_oos_prompt_budget_na_under_test", SCRIPT)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: NousResearch/Meta-Llama-3.1-8B-Instruct\nmodel_revision: d10aef7999a2b5ba950ab3974312feeedbfe0b77\n"
                   "explicit_cells: [none]\n")
    tok = _NoSwitchTok()
    monkeypatch.setattr(B, "_load_config_tokenizer",
                        lambda path, key: ("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77", key, tok))
    monkeypatch.setattr(B, "gsm8k_paired200_class_messages", lambda root, *, system_prompt=None: _g8k_msgs())
    monkeypatch.setattr(B, "system_prompt_for_cell", lambda cell: None)
    return B, tok, ["--task", "gsm8k_paired200", "--config", str(cfg), "--model-key", "llama8b"]


def test_render_kwargs_are_empty_and_the_cap_needs_the_explicit_flag(tmp_path, monkeypatch):
    B, _tok, _base = _arm(tmp_path, monkeypatch)
    assert B.render_kwargs(NA) == {} and B.render_kwargs(C.THINKING_OFF) == {C.ENABLE_THINKING_KEY: False}
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="pass --cap explicitly"):
        B.cap_for(NA)
    # the switch arms' rule is untouched
    assert B.cap_for(C.THINKING_ON) == (8192, C.GSM8K_STANDARD_TASK) and B.cap_for(C.THINKING_OFF)[0] == 512


def test_the_not_available_artifact_and_its_fit_sidecars_at_the_two_gsm8k_entries(tmp_path, monkeypatch):
    B, tok, base = _arm(tmp_path, monkeypatch)
    out = tmp_path / "budgets" / "llama8b" / "none" / "8192" / C.OOS_PROMPT_BUDGET_FILENAME
    head = _head_sha()                                                   # --tree-sha is the producer's own HEAD, never a fake
    rc = B.main([*base, "--arm", NA, "--cap", "8192", "--geometry-eval", "capability:gen_long_thinking",
                 "--out", str(out), "--tree-sha", head])
    assert rc == 0
    doc = C.read_oos_prompt_budget(out)
    assert doc["arm"] == NA and doc["cap"] == 8192 and doc["cap_task"] == C.GSM8K_STANDARD_TASK and doc["n"] == 200
    assert doc["render"] == {"add_generation_prompt": True, C.CHAT_TEMPLATE_KWARGS_KEY: {}}       # NO switch in the render
    assert doc["tree_sha"] == head and doc["model_slug"] == "llama8b"
    assert all(call["kwargs"] == {} and call["add_generation_prompt"] for call in tok.calls)
    side = json.loads(Path(str(out) + ".fit.json").read_text())
    assert side["arm"] == NA and side["fits"] is True and side["window"] == 10752 and side["cap"] == 8192
    assert side["geometry_eval"] == "capability:gen_long_thinking" and side["tree_sha"] == head
    assert side["margin"] == 10752 - side["prompt_max"] - 8192 and side["task"] == "gsm8k_paired200"
    # the rung against its entry
    out_r = tmp_path / "budgets" / "llama8b" / "none" / "16384" / C.OOS_PROMPT_BUDGET_FILENAME
    assert B.main([*base, "--arm", NA, "--cap", "16384", "--geometry-eval", "capability:gen_long_gsm8k_rung",
                   "--out", str(out_r), "--tree-sha", head]) == 0
    side_r = json.loads(Path(str(out_r) + ".fit.json").read_text())
    assert side_r["window"] == 18944 and side_r["margin"] == 18944 - side_r["prompt_max"] - 16384 and side_r["arm"] == NA
    # the run-side equality the driver applies to the artifact: the arm and the cap ARE the run's
    C.prompt_budget_matches_run(doc, arm=NA, cap=8192)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="serves the 'on' arm"):
        C.prompt_budget_matches_run(doc, arm=C.THINKING_ON, cap=8192)
    # without --cap the no-switch arm refuses by name (no table applies)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="pass --cap explicitly"):
        B.main([*base, "--arm", NA, "--out", str(tmp_path / "x.json")])
