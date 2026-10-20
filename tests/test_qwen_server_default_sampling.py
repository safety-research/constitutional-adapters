"""Server-side sampling default for targets that ship NO generation_config.json.

vLLM fills every sampling key a request omits from the model's shipped
``generation_config.json`` (``to_sampling_params``, read at the 0.25.1 tag),
so gemma / GLM / Nemotron requests that send only ``temperature`` — BoN, the
4,096-cap multi-turn re-runs, the Petri target, agentic misalignment — decode
at their vendor's top_p / top_k, while a Qwen3.5 request identical in shape
decoded at vLLM's neutral defaults (top_p 1.0, top_k off): the pinned Qwen
snapshots ship no such file. These tests pin the fix: the composer hands the
vendor's INSTRUCT / non-thinking general row for the Qwen3.5 rows to
``--override-generation-config`` for exactly those models, as a
wrapper-owned, recorded, premise-checked launch setting.
"""
from __future__ import annotations

import json

import pytest

import huggingface_hub
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C

from test_petri_serving_gate import (  # noqa: E402
    THINK_TEMPLATE_HASH, _spec, _think_template, _tpl_spec, _chat_template)

QWEN4B = "Qwen/Qwen3.5-4B"
QWEN9B = "Qwen/Qwen3.5-9B"
QWEN4B_REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_JSON = '{"min_p":0.0,"repetition_penalty":1.0,"temperature":0.7,"top_k":20,"top_p":0.8}'


@pytest.fixture(autouse=True)
def _no_cached_generation_config(monkeypatch):
    """The grading host / CPU-test default: the hub was never asked (None). The
    premise-check tests override this per case."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: None)


def _qwen_launch(tmp_path, model=QWEN4B, revision=QWEN4B_REV, **kw):
    return vs.build_server_launch(
        _spec(target_model=model, target_revision=revision,
              chat_template_hash=THINK_TEMPLATE_HASH,
              serving_launch_flags=["--enforce-eager"]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=_think_template(tmp_path), **kw)


# ---------------------------------------------------------------------------
# contracts: the rows, the pinned default, the liftable keys, the composer
# ---------------------------------------------------------------------------

def test_the_four_vendor_rows_are_the_users_numbers():
    r = C.QWEN35_VENDOR_ROWS
    assert set(r) == {"thinking_general", "thinking_precise_coding",
                      "instruct_general", "instruct_reasoning"}
    assert r["thinking_general"][:3] == (1.0, 0.95, 20)
    assert r["thinking_general"].presence_penalty == 1.5
    assert r["thinking_precise_coding"][:3] == (0.6, 0.95, 20)
    assert r["thinking_precise_coding"].presence_penalty == 0.0
    assert r["instruct_general"][:3] == (0.7, 0.8, 20)
    assert r["instruct_general"].presence_penalty == 1.5
    assert r["instruct_reasoning"][:3] == (1.0, 0.95, 20)
    assert r["instruct_reasoning"].presence_penalty == 1.5
    for row in r.values():
        assert row.min_p == 0.0 and row.repetition_penalty == 1.0


def test_the_pinned_server_default_is_the_instruct_general_row_for_both_qwens():
    """The served render is pinned thinking-OFF and the omit-params workloads
    are general chat, so the non-thinking general row is the default."""
    assert set(C.SERVER_DEFAULT_SAMPLING_BY_MODEL) == {QWEN4B, QWEN9B}
    for model in (QWEN4B, QWEN9B):
        assert C.SERVER_DEFAULT_SAMPLING_BY_MODEL[model] is C.QWEN35_VENDOR_ROWS["instruct_general"]


def test_the_pinned_default_matches_the_sampled_reruns_qwen_preset():
    """The sampled static / over-refusal re-run already sent this exact row
    per request (BATTERY_SAMPLING_BY_MODEL_KEY); the server default makes the
    omit-params paths agree with it instead of with vLLM's neutral defaults."""
    a = C.SERVER_DEFAULT_SAMPLING_BY_MODEL[QWEN4B]
    b = C.BATTERY_SAMPLING_BY_MODEL_KEY["qwen35_4b"]
    for f in C.SAMPLING_ROW_FIELDS:
        if f == "source":
            continue
        assert getattr(a, f) == getattr(b, f), f


def test_defaultable_keys_are_exactly_what_vllm_lifts_and_exclude_presence_penalty():
    """ModelConfig.get_diff_sampling_param (0.25.1) lifts repetition_penalty,
    temperature, top_k, top_p, min_p (+ max_new_tokens, a cap); the protocol's
    presence_penalty field defaults to 0.0, never None, so a generation-config
    presence penalty is never applied — it must not be smuggled in."""
    assert set(C.SERVER_DEFAULTABLE_SAMPLING_KEYS) == {
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty"}
    assert "presence_penalty" not in C.SERVER_DEFAULTABLE_SAMPLING_KEYS
    assert "max_new_tokens" not in C.SERVER_DEFAULTABLE_SAMPLING_KEYS


def test_composer_emits_only_liftable_set_keys_deterministically():
    row = C.QWEN35_VENDOR_ROWS["instruct_general"]
    js = C.server_default_generation_config(row)
    assert js == EXPECTED_JSON
    parsed = json.loads(js)
    assert "presence_penalty" not in parsed
    assert isinstance(parsed["top_k"], int)
    # a row with no optional terms emits only its three numbers
    sparse = C.SamplingRow(temperature=0.5, top_p=None, top_k=7, source="t")
    assert json.loads(C.server_default_generation_config(sparse)) == {
        "temperature": 0.5, "top_k": 7}


# ---------------------------------------------------------------------------
# the launcher: composed for the pinned models only, recorded, wrapper-owned
# ---------------------------------------------------------------------------

def test_qwen_launch_composes_the_override_and_records_it(tmp_path):
    launch = _qwen_launch(tmp_path)
    cmd = list(launch.cmd)
    i = cmd.index("--override-generation-config")
    assert cmd[i + 1] == EXPECTED_JSON
    assert launch.override_generation_config == EXPECTED_JSON
    # composed exactly once, next to the thinking pin, and the pin still holds
    assert cmd.count("--override-generation-config") == 1
    assert "--default-chat-template-kwargs" in cmd
    assert "--generation-config" not in cmd  # vLLM's default "auto" merges the override


def test_qwen_9b_is_pinned_too(tmp_path):
    launch = _qwen_launch(tmp_path, model=QWEN9B,
                          revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    assert launch.override_generation_config == EXPECTED_JSON


@pytest.mark.parametrize("model", [
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-synthetic",
    "google/gemma-4-E4B-it-synthetic",
    "meta-llama/Llama-3.1-8B-Instruct",
])
def test_models_with_a_shipped_file_compose_nothing(tmp_path, model):
    """Every launch to date keeps its identity: no flag, None on the record."""
    if "Llama" in model:
        launch = vs.build_server_launch(
            _tpl_spec(target_model=model, serving_launch_flags=["--enforce-eager"]),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=_chat_template(tmp_path))
    else:
        launch = _qwen_launch(tmp_path, model=model, revision="0" * 40)
    assert "--override-generation-config" not in launch.cmd
    assert launch.override_generation_config is None


@pytest.mark.parametrize("flags", [
    ["--override-generation-config", '{"temperature": 0.1}'],
    ['--override-generation-config={"temperature": 0.1}'],
    ["--generation-config", "vllm"],
    ["--override_generation_config", "{}"],
])
def test_manifest_may_not_name_the_default_sampling_flags(tmp_path, flags):
    """A manifest naming either spelling would change every omit-params
    decode behind the recorded launch identity — wrapper-owned, refused."""
    with pytest.raises(vs.VllmServerError, match="wrapper-owned"):
        vs.build_server_launch(
            _spec(target_model=QWEN4B, target_revision=QWEN4B_REV,
                  chat_template_hash=THINK_TEMPLATE_HASH,
                  serving_launch_flags=["--enforce-eager", *flags]),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=_think_template(tmp_path))


# ---------------------------------------------------------------------------
# the premise is checked against the cache the launch serves from
# ---------------------------------------------------------------------------

def test_a_cached_generation_config_refuses_rather_than_masking_it(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: str(tmp_path / "snap" / "generation_config.json"))
    with pytest.raises(vs.VllmServerError, match="generation_config.json"):
        _qwen_launch(tmp_path)


def test_the_no_exist_sentinel_and_an_unasked_cache_both_compose(tmp_path, monkeypatch):
    sentinel = huggingface_hub.file_download._CACHED_NO_EXIST
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: sentinel)
    assert _qwen_launch(tmp_path).override_generation_config == EXPECTED_JSON
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda *a, **k: None)
    assert _qwen_launch(tmp_path).override_generation_config == EXPECTED_JSON


def test_a_hub_probe_failure_is_not_a_launch_failure(tmp_path, monkeypatch):
    """No hub / no cache dir means nothing to mask; the pin still composes."""
    def boom(*a, **k):
        raise OSError("no cache")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", boom)
    assert _qwen_launch(tmp_path).override_generation_config == EXPECTED_JSON
