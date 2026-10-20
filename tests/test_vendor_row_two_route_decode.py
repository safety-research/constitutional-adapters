"""One vendor sampling row, two routes to the engine — does every key arrive?

A served Qwen3.5 target's vendor row reaches vLLM by TWO routes:

* the SERVER-DEFAULT route. An omit-params request (``target_decode.
  GENERATION_CONFIG`` — BoN, the 4,096-cap multi-turn re-runs, the Petri
  target, agentic misalignment; ``served_target.py``) carries NO sampling key,
  and vLLM fills what it can from the standup's ``--override-generation-config``,
  which ``vllm_server._server_default_generation_config`` composes as
  ``contracts.server_default_generation_config(SERVER_DEFAULT_SAMPLING_BY_MODEL
  [model])``. vLLM lifts ONLY ``contracts.SERVER_DEFAULTABLE_SAMPLING_KEYS``
  from a generation config (``ModelConfig.get_diff_sampling_param``, 0.25.1);
  every other key stays at the OpenAI protocol's field default. For
  ``presence_penalty`` that default is ``0.0``, never None, so no generation
  config can set it — the constant's own note says so.
* the SAMPLED route. The static families and the judged over-refusal controls
  (``jailbreak_run._sampled_decode``; ``cell_controls._sampled_controls_gen`` ->
  ``served_capability_run``) put the row's keys on EVERY request body: exactly
  ``sisteer.eval.sampled_decode.SAMPLER_KEYS`` ∩ the preset
  (``default_sampled_block``; ``bon_served.sampler_preset`` on the wire).

Both routes are "the vendor row" — its two statements are pinned equal — and
they are NOT the same decode: the row's ``presence_penalty`` 1.5 rides the
sampled route and falls to vLLM's neutral 0.0 on the server-default route.
That gap is by design and documented on the constant. What nothing pinned is
the STRUCTURE that keeps it the only one — that every key the row states
reaches the engine by at least one route — so a key added to the yaml row
later could decode at vLLM's default on both routes with no test noticing.

Per model stated on both routes (the yaml's ``presets.<key>.model`` ∩
``SERVER_DEFAULT_SAMPLING_BY_MODEL`` — Qwen3.5-4B alone today; a future server
default for another model inherits the check):

1. the keys the server default LIFTS (the composed JSON) equal
   ``SERVER_DEFAULTABLE_SAMPLING_KEYS`` ∩ the row's keys — the composer neither
   smuggles a key vLLM would drop nor lifts one the sampled lane never sends.
2. every REMAINING row key is in the driver's per-request send set, read from
   ``sampled_decode.SAMPLER_KEYS`` itself. A literal here would silently
   accept a new key the driver does not send; the module's tuple will not.

Neither the agreement check's claim (the two statements agree on their
numbers) nor ``test_qwen_server_default_sampling``'s (the defaultable set's
members, the composed JSON's bytes): this is the route property over the
row's KEYS.

WHAT THIS MODULE DOES NOT GUARD. Moving a key into or out of
``SERVER_DEFAULTABLE_SAMPLING_KEYS`` passes here, by construction:
``server_default_generation_config`` derives its lift from that same
constant, so both sides of check 1 move together, and a key that stops being
lifted still rides the sampled route, so check 2 holds too (verified by
perturbation: ``min_p`` removed from the constant and ``presence_penalty``
added to it are both green here). The constant's MEMBERSHIP — what vLLM
0.25.1 actually lifts, and that ``presence_penalty`` is not among it — is
pinned by ``test_qwen_server_default_sampling.
test_defaultable_keys_are_exactly_what_vllm_lifts_and_exclude_presence_penalty``.
Two tests, one seam between them: that one holds the set; this one holds
that every row key crosses it by some route. Do not read the set as pinned
here.

The yaml is read RAW (``yaml.safe_load``), not through
``sampled_decode.load_decode_presets``: the loader refuses a non-sampler key
before this module could name what it means for the two routes, and the
named failure IS the deliverable.

Provenance of the ``0.0`` the messages name: vLLM is not installed in the
shared test venv, so :data:`VLLM_PROTOCOL_DEFAULT` restates the note on
``contracts.SERVER_DEFAULTABLE_SAMPLING_KEYS`` (pinned there to 0.25.1's
``ChatCompletionRequest`` field defaults), not a fresh read of vLLM's source.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import sampled_decode as SD

REPO = Path(__file__).resolve().parents[1]
#: the sampled lane's ONE data file (sampled_decode.DECODE_PRESETS_PATH)
SAMPLED_DECODE_YAML = REPO / "configs" / "sdfbase_sampled_decode.yaml"

#: What vLLM's OpenAI protocol decodes a key at when NO route carries it: the
#: ``ChatCompletionRequest`` field defaults (0.25.1) — the numbers the note on
#: ``contracts.SERVER_DEFAULTABLE_SAMPLING_KEYS`` and the mt-cap4096 records
#: ("vllm_defaults_govern_top_p_top_k") name. MESSAGE CONTENT ONLY: no
#: assertion below reads this table.
VLLM_PROTOCOL_DEFAULT: Mapping[str, Any] = {
    "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    "min_p": 0.0, "top_p": 1.0, "top_k": "off", "temperature": 1.0,
}


def _engine_default(key: str) -> Any:
    return VLLM_PROTOCOL_DEFAULT.get(key, "vLLM's protocol default")


def _yaml_rows() -> Dict[str, Dict[str, Any]]:
    """``{<hf model id>: <preset dict>}`` — the row the sampled lane sends,
    per ``presets.<key>``; two keys naming one model make the join ambiguous
    and refuse — unless their rows are IDENTICAL (model, revision, preset), the
    worked example qwen35_4b_demo carrying the paper's qwen35_4b row verbatim
    (sampled_decode.preset_key_for_model collapses those to the first key)."""
    doc = yaml.safe_load(SAMPLED_DECODE_YAML.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, tuple] = {}
    for key, entry in doc["presets"].items():
        model = entry["model"]
        row = (entry.get("revision"), dict(entry["preset"]))
        if model in out:
            assert seen[model] == row, f"presets.{key} restates {model!r} with a DIFFERENT row; the join is ambiguous"
            continue
        seen[model] = row
        out[model] = dict(entry["preset"])
    return out


def _both_routes() -> List[str]:
    """The models with a row on BOTH routes: a yaml preset (sampled) and a
    server default (omit-params)."""
    if not SAMPLED_DECODE_YAML.is_file():
        return []
    return sorted(set(_yaml_rows()) & set(C.SERVER_DEFAULT_SAMPLING_BY_MODEL))


BOTH_ROUTES = _both_routes()


def test_at_least_one_model_is_stated_on_both_routes():
    """An empty join would make the parametrized check below a silent skip.
    Qwen3.5-4B is on both routes today; if the join key (the HF model id) has
    drifted, that is itself the divergence this module exists to catch."""
    assert SAMPLED_DECODE_YAML.is_file(), f"{SAMPLED_DECODE_YAML.relative_to(REPO)} is not in this tree"
    assert BOTH_ROUTES, (
        f"no model has both a sampled-lane row ({SAMPLED_DECODE_YAML.name} presets: "
        f"{sorted(_yaml_rows())}) and a server default (SERVER_DEFAULT_SAMPLING_BY_MODEL: "
        f"{sorted(C.SERVER_DEFAULT_SAMPLING_BY_MODEL)}); the route check would guard nothing")


@pytest.mark.parametrize("model", BOTH_ROUTES)
def test_every_key_of_the_vendor_row_reaches_the_engine_by_some_route(model: str):
    row = C.SERVER_DEFAULT_SAMPLING_BY_MODEL[model]          # what the server default lifts FROM
    preset = _yaml_rows()[model]                             # what the sampled lane SENDS
    row_keys = set(preset)
    liftable = set(C.SERVER_DEFAULTABLE_SAMPLING_KEYS)
    sent = set(SD.SAMPLER_KEYS)                              # the driver's per-request send set

    # (1) the server-default route lifts exactly the row's liftable keys
    lifted = set(json.loads(C.server_default_generation_config(row)))
    expected_lift = liftable & row_keys
    assert lifted == expected_lift, _lift_mismatch(model, row, preset, lifted, expected_lift)

    # (2) every key the server default cannot lift rides the sampled route per request
    unlifted = row_keys - lifted
    stranded = unlifted - sent
    assert not stranded, _stranded(model, preset, unlifted, stranded)


def _lift_mismatch(model, row, preset, lifted, expected_lift) -> str:
    lines = [
        f"{model}: the server-default route and the sampled route decode DIFFERENT keys of the vendor row.",
        f"  lifted by --override-generation-config (contracts.server_default_generation_config): {sorted(lifted)}",
        f"  liftable keys of the row the sampled lane sends (SERVER_DEFAULTABLE_SAMPLING_KEYS ∩ "
        f"{SAMPLED_DECODE_YAML.name} preset): {sorted(expected_lift)}",
    ]
    for k in sorted(lifted - expected_lift):
        lines.append(
            f"  {k}: omit-params requests decode at {getattr(row, k)!r} (lifted from the contracts row) "
            f"while sampled requests never send it and decode at vLLM's default {_engine_default(k)!r}")
    for k in sorted(expected_lift - lifted):
        lines.append(
            f"  {k}: sampled requests send {preset[k]!r} while omit-params requests decode at vLLM's "
            f"default {_engine_default(k)!r} — the composer did not lift a liftable key")
    return "\n".join(lines)


def _stranded(model, preset, unlifted, stranded) -> str:
    lines = [
        f"{model}: the vendor row states {sorted(stranded)}, and NO route carries "
        f"{'it' if len(stranded) == 1 else 'them'} to the engine.",
        f"  server-default route: vLLM lifts only SERVER_DEFAULTABLE_SAMPLING_KEYS="
        f"{C.SERVER_DEFAULTABLE_SAMPLING_KEYS} from --override-generation-config;",
        f"  sampled route: the drivers send only sampled_decode.SAMPLER_KEYS={SD.SAMPLER_KEYS} per request.",
    ]
    for k in sorted(stranded):
        lines.append(f"  {k}: the row says {preset[k]!r}; EVERY request decodes at vLLM's protocol default "
                     f"{_engine_default(k)!r} instead.")
    riding = sorted(unlifted - stranded)
    if riding:
        lines.append(
            "  For comparison, the row's unlifted keys that DO ride the sampled route today — "
            + "; ".join(f"{k} {preset[k]!r} on sampled requests vs vLLM's {_engine_default(k)!r} on "
                        f"omit-params requests" for k in riding)
            + " — already make the two routes decode differently; a new unlifted key must at least be "
              "sent per request the same way (add it to the driver's send set), or be made liftable.")
    return "\n".join(lines)
