"""The canonical IFEval battery — the ``vendor_preset`` sampling mode.

DECODING = the vendor preset from each model's
``generation_config.json`` — gemma 1.0 / 0.95 / 64, Nemotron 1.0 / 0.95 (the
sampled lanes' user-supplied top_p 1.0 RECORDED beside it), Llama 0.6 / 0.9,
Qwen its vendor INSTRUCT general row 0.7 / 0.8 / 20 + min_p 0 + presence_penalty
1.5 (which MUST ride per request: the server default never covers it).
The rows are built BY REFERENCE to the constants that already carry the numbers.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tiers import inspect_served as isv

from test_inspect_served import _api, fake  # noqa: F401 — the fixture + the provider helper

VENDOR = C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY
IFEVAL = C.inspect_task(C.IFEVAL_TASK_NAME)


def test_the_mode_exists_and_resolves_exactly_the_four_paper_models():
    assert C.SAMPLING_VENDOR_PRESET == "vendor_preset" and C.SAMPLING_VENDOR_PRESET in C.SAMPLING_MODES
    assert C.SAMPLING_MODES[-1] == C.SAMPLING_VENDOR_PRESET
    assert C.check_sampling_arg("vendor_preset") == "vendor_preset"
    assert C.SAMPLING_TABLE_BY_MODE[C.SAMPLING_VENDOR_PRESET] is VENDOR
    assert C.SAMPLING_VENDOR_PRESET in C.SAMPLING_TABLE_MODES
    assert set(VENDOR) == {"gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"}
    for key in VENDOR:
        assert C.sampling_for(key, C.SAMPLING_VENDOR_PRESET) is VENDOR[key]
    # never a default temperature: every other key refuses by name
    for other in ("glm46vflash", "gemma4e2b", "qwen35_9b", "llama3b", "no-such-model"):
        with pytest.raises(C.SamplingUnverified) as ei:
            C.sampling_for(other, C.SAMPLING_VENDOR_PRESET)
        msg = str(ei.value)
        assert other in msg and "VENDOR_PRESET_SAMPLING_BY_MODEL_KEY" in msg and "vendor_preset" in msg


def test_the_rows_numbers_equal_the_fixed_decisions_and_their_referenced_constants():
    # the pinned values, number by number
    assert VENDOR["gemma4e4b"][:3] == (1.0, 0.95, 64)
    assert VENDOR["nemotron3nano4b"][:3] == (1.0, 0.95, None)
    assert VENDOR["llama8b"][:3] == (0.6, 0.9, None)
    q = VENDOR["qwen35_4b"]
    assert q[:3] == (0.7, 0.8, 20) and (q.min_p, q.presence_penalty, q.repetition_penalty) == (0.0, 1.5, 1.0)
    for key in ("gemma4e4b", "nemotron3nano4b", "llama8b"):
        r = VENDOR[key]
        assert (r.min_p, r.presence_penalty, r.repetition_penalty) == (None, None, None), key
    # BY REFERENCE (P3): the same numbers as the constants that already carry them
    assert VENDOR["gemma4e4b"][:3] == C.THINKING_SAMPLING_BY_MODEL_KEY["gemma4e4b"][:3]
    assert VENDOR["nemotron3nano4b"][:3] == C.THINKING_SAMPLING_BY_MODEL_KEY["nemotron3nano4b"][:3]
    ref = C.QWEN35_VENDOR_ROWS["instruct_general"]
    assert q[:3] == ref[:3] and (q.min_p, q.presence_penalty, q.repetition_penalty) == (
        ref.min_p, ref.presence_penalty, ref.repetition_penalty)
    assert q[:3] == C.BATTERY_SAMPLING_BY_MODEL_KEY["qwen35_4b"][:3]              # the battery's Qwen row too
    assert VENDOR["llama8b"][:3] == (C._LLAMA_CARD["temperature"], C._LLAMA_CARD["top_p"], C._LLAMA_CARD["top_k"])
    # the identity hashes numbers, not the mode: gemma's vendor_preset identity == its thinking_card identity
    k = {IFEVAL: 2}
    assert C.sampling_identity(VENDOR["gemma4e4b"], k) == C.sampling_identity(
        C.THINKING_SAMPLING_BY_MODEL_KEY["gemma4e4b"], k)


def test_the_source_prose_states_the_provenance_the_reverification_and_nemotrons_recorded_alternative():
    """The rows' ``source`` prose is the matrix's spelling (its ifeval_contracts fallback; the frozen
    matrix records it per row) so the merge-agreement test holds; the re-verification record
    (P4: re-read at the pin 2026-09-12) rides beside the rows in VENDOR_PRESET_REVERIFICATION."""
    for key in ("gemma4e4b", "nemotron3nano4b", "llama8b"):
        assert VENDOR[key].source.startswith("generation_config.json of"), key   # P43's label: the shipped file
    assert "ee0ef602" in VENDOR["gemma4e4b"].source and "top_k 64" in VENDOR["gemma4e4b"].source
    assert "P4)" in VENDOR["gemma4e4b"].source and "P4)" in VENDOR["llama8b"].source     # re-verified at the pin
    assert "dfaf35de" in VENDOR["nemotron3nano4b"].source
    # P43: the sampled lanes' top_p 1.0 is a GRANDFATHERED exception, recorded beside the shipped file
    assert "user-supplied top_p 1.0" in VENDOR["nemotron3nano4b"].source
    assert "GRANDFATHERED" in VENDOR["nemotron3nano4b"].source and "SETTLED" in VENDOR["nemotron3nano4b"].source
    assert "BATTERY_SAMPLING_BY_MODEL_KEY['nemotron3nano4b']" in VENDOR["nemotron3nano4b"].source
    assert C.BATTERY_SAMPLING_BY_MODEL_KEY["nemotron3nano4b"].top_p == 1.0          # the alternative's number
    # Llama: the SERVED snapshot d10aef79 ships the file (0.6 / 0.9)
    assert "d10aef79" in VENDOR["llama8b"].source and "_LLAMA_CARD" in VENDOR["llama8b"].source
    assert "INHERITED" not in VENDOR["llama8b"].source                             # it ships one; nothing inherited
    # Qwen: no file ships; the vendor INSTRUCT general row; presence_penalty rides per request
    assert "ships no generation_config.json" in VENDOR["qwen35_4b"].source
    assert "instruct_general" in VENDOR["qwen35_4b"].source
    assert "SENT PER REQUEST" in VENDOR["qwen35_4b"].source and "not server-defaultable" in VENDOR["qwen35_4b"].source
    # the re-verification record: every row, re-read at the pin, the Llama file identical to meta-llama's
    rv = C.VENDOR_PRESET_REVERIFICATION
    assert set(rv) == set(VENDOR)
    for key in ("gemma4e4b", "nemotron3nano4b", "llama8b"):
        assert "re-read 2026-09-12" in rv[key], key
    assert "0e9e39f2" in rv["llama8b"] and "re-checked 2026-09-12" in rv["qwen35_4b"]
    assert "user-supplied top_p 1.0 (pre-ruling), shipped file 0.95" in rv["nemotron3nano4b"]
    assert "presence_penalty is NOT server-defaultable and is SENT per request" in rv["qwen35_4b"]


def test_the_qwen_server_default_never_carries_presence_penalty_so_the_request_must():
    q = VENDOR["qwen35_4b"]
    composed = json.loads(C.server_default_generation_config(q))
    assert composed == {"min_p": 0.0, "repetition_penalty": 1.0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
    assert "presence_penalty" not in composed and "presence_penalty" not in C.SERVER_DEFAULTABLE_SAMPLING_KEYS
    # the composed server default for this model IS the same vendor row's defaultable keys
    assert C.SERVER_DEFAULT_SAMPLING_BY_MODEL["Qwen/Qwen3.5-4B"] is C.QWEN35_VENDOR_ROWS["instruct_general"]
    assert C.server_default_generation_config(C.SERVER_DEFAULT_SAMPLING_BY_MODEL["Qwen/Qwen3.5-4B"]) == \
        C.server_default_generation_config(q)
    # the identity + manifest block carry the three optional terms for Qwen, none for the others
    ident = C.sampling_identity(q, {IFEVAL: 2})
    assert (ident[C.IDENTITY_KEY_MIN_P], ident[C.IDENTITY_KEY_PRESENCE_PENALTY],
            ident[C.IDENTITY_KEY_REPETITION_PENALTY]) == (0.0, 1.5, 1.0)
    assert ident[C.IDENTITY_KEY_SEED_RULE][IFEVAL] == C.SEED_RULE_IFEVAL
    assert C.sampling_block_fields(q) == (*C.SAMPLING_BLOCK_FIELDS, "min_p", "presence_penalty", "repetition_penalty")
    assert C.sampling_block_fields(VENDOR["gemma4e4b"]) == C.SAMPLING_BLOCK_FIELDS


def test_the_provider_puts_the_qwen_row_on_the_wire_with_presence_penalty_top_level(fake, monkeypatch):
    """How the row reaches the wire: temperature / top_p /
    presence_penalty as GenerateConfig fields (inspect_ai's OpenAI params emit
    them top-level), top_k / min_p / repetition_penalty in extra_body (vLLM reads
    them off the body), seed top-level by the IFEval rule for an IFEval request."""
    from inspect_ai.model import GenerateConfig
    from inspect_ai.model._openai import openai_completion_params

    api = _api(fake, si=None, sampling=VENDOR["qwen35_4b"])
    extra: Dict[str, Any] = {}
    out = api.sampling_params(GenerateConfig(), extra, task=IFEVAL, item="1000", idx=1)
    assert out == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
    assert extra == {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    # the FULL fragment the provider builds for an IFEval request
    monkeypatch.setattr(isv, "current_request_identity", lambda: (IFEVAL, "1000", 1))
    cfg = api.tinycap_config(GenerateConfig())
    assert cfg.presence_penalty == 1.5 and cfg.temperature == 0.7 and cfg.top_p == 0.8
    assert cfg.seed == isv.ifeval_sample_seed("1000", 1) == 3178680996
    assert cfg.extra_body["top_k"] == 20 and cfg.extra_body["min_p"] == 0.0 and cfg.extra_body["repetition_penalty"] == 1.0
    assert cfg.extra_headers[C.HEADER_TASK] == IFEVAL and cfg.extra_headers[C.HEADER_SAMPLE] == "1000/1"
    # inspect_ai 0.3.260's OpenAI params emit presence_penalty and seed at TOP level
    params = openai_completion_params(model="m", config=cfg, tools=False)
    assert params["presence_penalty"] == 1.5 and params["seed"] == 3178680996
    assert params["temperature"] == 0.7 and params["top_p"] == 0.8
    assert "presence_penalty" not in params.get("extra_body", {})           # never buried in the body extras
    # a different explicit presence_penalty from a caller refuses (the row is frozen per model)
    with pytest.raises(isv.InspectServedError, match="presence_penalty=1.0 against this model's frozen row"):
        api.sampling_params(GenerateConfig(presence_penalty=1.0), {}, task=IFEVAL, item="1000", idx=0)


def test_the_gemma_row_sends_no_penalty_and_no_min_p(fake, monkeypatch):
    from inspect_ai.model import GenerateConfig
    from inspect_ai.model._openai import openai_completion_params

    api = _api(fake, si=None, sampling=VENDOR["gemma4e4b"])
    extra: Dict[str, Any] = {}
    out = api.sampling_params(GenerateConfig(), extra, task=IFEVAL, item="1000", idx=0)
    assert out == {"temperature": 1.0, "top_p": 0.95, "top_k": 64} and extra == {"top_k": 64}
    monkeypatch.setattr(isv, "current_request_identity", lambda: (IFEVAL, "1000", 0))
    cfg = api.tinycap_config(GenerateConfig())
    params = openai_completion_params(model="m", config=cfg, tools=False)
    assert "presence_penalty" not in params and params["seed"] == isv.ifeval_sample_seed("1000", 0)
    assert set(cfg.extra_body) & {"min_p", "repetition_penalty"} == set()


def test_the_driver_flag_accepts_the_mode():
    """The driver's ``--sampling`` choices ARE ``contracts.SAMPLING_MODES`` (one
    list, read by reference), so the sixth mode is accepted without a driver edit."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "served_capability_run.py").read_text()
    m = re.search(r"g\.add_argument\(tc\.FLAG_SAMPLING, dest=\"sampling\", choices=list\(tc\.SAMPLING_MODES\)", src)
    assert m, "the driver's --sampling must take its choices from contracts.SAMPLING_MODES"
    assert "vendor_preset" in C.SAMPLING_MODES
