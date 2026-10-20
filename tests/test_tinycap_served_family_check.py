"""The served driver's two family-keyed gates on the paper's models, pinned in EXECUTION form (the driver's own functions):

* `served_capability_run._check_served_family` compares the SERVED config's ``model_type`` (``vllm_server.model_family``) to the
  sampling model key's family BY NAME -- a model_type that is not the family refuses; ``None`` (nothing family-keyed can be read
  off the config) admits; no alias fold, no prefix guess: a served architecture joins a family's render fact only through its own
  measured header, never through a name;
* `_check_served_reasoning_parser` on a recorded endpoint (``reasoning_parser: null``, as every Qwen gsm8k standup records) admits a
  thinking-ON gsm8k run through the family's CERTIFIED fact (the `implicit_open` split IS the parser) and, with no fact (a key with
  no family: ``contracts.family_of_model_key`` -> None), refuses by name.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.serving import vllm_server  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402

FACT = tc.FAMILY_RENDER_FACTS["qwen3_5"]


def _endpoint(key: str = "qwen35_4b", hf: str = "Qwen/Qwen3.5-4B", fact=FACT) -> dict:
    """The keys the served driver's family / reasoning-parser gates read, in endpoint.recorded.json's shape (synthetic values)."""
    hdr = {"moves": True, "off_ids": list(fact.off_ids), "on_ids": list(fact.on_ids), "switch": bool(fact.thinking_switch)}
    return {"model": hf, "model_key": key, "model_revision": "0" * 40, "chat_template_hash": "0" * 16, "engine": "lora_native",
            "reasoning_parser": None, "thinking_mode": "off", "default_chat_template_kwargs": '{"enable_thinking": false}',
            "thinking_header": hdr, "thinking_header_error": None,
            "thinking_certification": {**hdr, "model_key": key, "verdict": "certified", "reason": None,
                                       "checks": {"clean_suffix": True, "plain_turn_prefix": True, "tail": True, "thinking_turn_prefix": True,
                                                  "off_header_absent_in_on_render": True, "plain_turn_reason": None, "thinking_turn_reason": None}}}


def test_check_served_family_compares_the_served_model_type_to_the_keys_family_by_name(monkeypatch):
    monkeypatch.setattr(vllm_server, "model_family", lambda model_id, revision=None: "qwen3_5")
    srun._check_served_family("Qwen/Qwen3.5-4B", "rev", family="qwen3_5", model_key="qwen35_4b")
    # another family's server refuses by name
    monkeypatch.setattr(vllm_server, "model_family", lambda model_id, revision=None: "gemma4")
    with pytest.raises(SystemExit, match="is family 'gemma4'") as e:
        srun._check_served_family("Qwen/Qwen3.5-4B", "rev", family="qwen3_5", model_key="qwen35_4b")
    assert "'qwen35_4b'" in str(e.value) and "'qwen3_5'" in str(e.value)
    # a variant architecture is NOT folded into its namesake family: refused by name, never a prefix guess
    monkeypatch.setattr(vllm_server, "model_family", lambda model_id, revision=None: "qwen3_5_next")
    with pytest.raises(SystemExit, match="is family 'qwen3_5_next'"):
        srun._check_served_family("org/variant", "rev", family="qwen3_5", model_key="qwen35_4b")
    # None = nothing family-keyed can be checked off the config
    monkeypatch.setattr(vllm_server, "model_family", lambda model_id, revision=None: None)
    srun._check_served_family("x/unknown", "rev", family="qwen3_5", model_key="qwen35_4b")
    # every paper family is its own served model_type (the serving library's vocabulary IS the registry's `family`)
    for key, fam in tc.FAMILY_BY_MODEL_KEY.items():
        monkeypatch.setattr(vllm_server, "model_family", lambda model_id, revision=None, fam=fam: fam)
        srun._check_served_family(sg.MODEL_HF_IDS[key], "rev", family=fam, model_key=key)


def _parser_check(endpoint, render_fact):
    buf = io.StringIO()
    with redirect_stdout(buf):
        srun._check_served_reasoning_parser([tc.CONTROL_GSM8K], tc.THINKING_ON, endpoint, by_task=None,
                                            allow_out_of_support=False, render_fact=render_fact, require_certified=True)
    return buf.getvalue()


def test_a_recorded_endpoint_that_composes_no_parser_is_admitted_by_the_certified_fact_and_refused_without_a_family():
    ep = _endpoint()
    assert ep["reasoning_parser"] is None and ep["thinking_mode"] == "off"      # as every Qwen gsm8k standup records
    fact = tc.render_fact_for(tc.family_of_model_key("qwen35_4b"))
    assert fact is not None and fact.evidence_status == tc.RENDER_EVIDENCE_CERTIFIED
    out = _parser_check(ep, fact)
    assert "IS the parser" in out and "'qwen3_5'" in out and fact.split_mode in out
    # the live rule the driver applies afterwards -- render_certified's tuple equality -- holds on the recorded header
    hdr = ep["thinking_header"]
    assert (tuple(hdr["off_ids"]), tuple(hdr["on_ids"])) == (fact.off_ids, fact.on_ids)
    # a key with no family (every key outside contracts.FAMILY_BY_MODEL_KEY: the front door's demo key among them) has no fact,
    # and the same endpoint refuses by name rather than guessing a parser
    for key in sg.ONBOARDED_MODEL_KEYS:
        assert tc.family_of_model_key(key) is None and tc.render_fact_for(tc.family_of_model_key(key)) is None, key
    with pytest.raises(SystemExit, match="standup composed NO reasoning parser"):
        _parser_check(_endpoint(), None)
