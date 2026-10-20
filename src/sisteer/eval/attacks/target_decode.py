"""The multi-turn TARGET's decode mode, in one place (the 4,096-cap canonical instrument).

Two layers, deliberately kept apart:

**The CONFIG-LEVEL mode** (``attacks.target_decode`` in a config; :data:`CONFIG_MODES`) names how a
multi-turn run chooses its target sampler when no explicit temperature is given:

* :data:`VENDOR_PRESET` -- THE CAMPAIGN DEFAULT since 2026-09-16: vendor presets for multi-turn, not
  the shipped generation config. The served model's VENDOR sampler preset from
  ``configs/sdfbase_sampled_decode.yaml`` (``jailbreak_run.vendor_preset_for``, keyed by ``--model-key``
  or resolved from the config's ``model`` / ``model_revision`` pins) rides EXPLICITLY on every target
  request: exactly the preset's keys, the same mechanism ``--multiturn-target-decode vendor`` has used
  since 2026-09-10. On the wire this is an EXPLICIT temperature whose ``temperature_source`` is
  ``vendor_preset``. Why it differs from the generation config for ONE model: gemma-4-E4B-it and
  Llama-3.1-8B-Instruct ship a ``generation_config.json`` equal to their vendor row, so the switch only
  changes whether the keys ride explicitly; Qwen3.5-4B ships NO generation_config.json and its vendor row
  carries ``presence_penalty 1.5``, which vLLM can never take as a server default
  (``contracts.SERVER_DEFAULTABLE_SAMPLING_KEYS`` excludes it; the standup's
  ``--override-generation-config`` omits it), so an omit-params Qwen request decodes at presence_penalty
  0.0 and only the vendor preset sends the 1.5. Nemotron's YAML row is a DECLARED departure from its
  shipped file (top_p 1.0 vs 0.95, ``deliberate_departure`` in the YAML); THE DEFAULT PATH PINS IT TO THE
  SHIPPED FILE (since 2026-09-16: six nemotron 4,096 multi-turn roots record top_p 0.95, so the default
  sends 0.95 and stays pairable with every nemotron anchor drawn;
  ``jailbreak_run._pin_vendor_preset_to_shipped_file`` takes every key the pinned shipped file states, so
  only nemotron moves), while an explicit ``--multiturn-target-decode vendor`` still sends the YAML row
  as it stands (1.0) -- which of the two ran is recorded (``preset_basis``, ``departure_from_yaml``),
  never inferred.
  The preset is served request content: on an HF (in-process) target this mode REFUSES and asks for an
  explicit decode, and a served model with no YAML row refuses too (a temperature nobody vouched for is
  never a default -- the same rule as ``sisteer.eval.sampled_decode`` and ``contracts.sampling_for``).
* :data:`GENERATION_CONFIG` -- the default from 2026-09-11 to 2026-09-16 ("each model's own shipped
  decode" as a mechanism): the request SAMPLES and carries NO sampling key at all, so the server fills
  every key (temperature, top_p, top_k, min_p, repetition_penalty) from the served model's generation
  config -- the shipped ``generation_config.json`` (gemma-4-E4B-it, Nemotron-3-Nano, GLM-4.6V-Flash) or,
  for a model that ships none, the standup's ``--override-generation-config`` (Qwen3.5:
  ``contracts.SERVER_DEFAULT_SAMPLING_BY_MODEL``). Still one config key away
  (``attacks.target_decode: generation_config``) and byte-identical to what it sent as the default.
* :data:`GREEDY` -- ``temperature: 0.0`` on the request (the pre-2026-09-11 default; the filed 512-cap
  multi-turn stores; ``--target-greedy`` / ``attacks.target_decode: greedy``). Request bytes unchanged
  from every earlier wave.

**The WIRE-LEVEL mode** is what ``temperature`` (``float | None``) on a target call names, and what
:func:`decode_mode` / :func:`temperature_source` classify: ``None`` -> :data:`GENERATION_CONFIG` (no
sampling key), ``0.0`` -> :data:`GREEDY`, ``t > 0`` -> :data:`EXPLICIT` (``temperature: t``, optionally
with the extra keys the caller names, ``served_target.DECODE_EXTRA_KEYS``, the server's generation
config filling the rest). A vendor preset is EXPLICIT on the wire (its own temperature + keys); the
records tell it apart by ``temperature_source == "vendor_preset"`` and ``mode == "vendor_preset"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sisteer.registry import decode_policy as _decode_policy  # THE decode-policy authority (stdlib-only reader)

GENERATION_CONFIG = "generation_config"
GREEDY = "greedy"
EXPLICIT = "explicit"
VENDOR_PRESET = "vendor_preset"
#: every mode a target decode RECORD may name (wire-level + the config-level vendor preset)
MODES = (GENERATION_CONFIG, GREEDY, EXPLICIT, VENDOR_PRESET)
#: the values ``attacks.target_decode`` takes in a config (an explicit temperature is
#: ``attacks.target_temperature`` / ``--target-temperature`` instead)
CONFIG_MODES = (VENDOR_PRESET, GENERATION_CONFIG, GREEDY)
#: THE CAMPAIGN DEFAULT for an unspecified ``attacks.target_decode`` (2026-09-16)
DEFAULT_CONFIG_MODE = VENDOR_PRESET
#: the greedy boundary: the registry's `<model>|multi_turn|off|greedy` rows' temperature (0.0; the filed 512-cap
#: stores' row, superseded by the generation_config instrument, kept as the classification boundary)
GREEDY_TEMPERATURE: float = float(_decode_policy.load().model_independent(
    eval_category="multi_turn", thinking_mode="off", mode="greedy").numbers()["temperature"])

#: what ``last_decode["temperature_source"]`` says about where the effective temperature came from
SOURCE_REQUEST = "request"
SOURCE_SERVER_GENERATION_CONFIG = "server_generation_config"
SOURCE_VENDOR_PRESET = "vendor_preset"


def do_sample_for(temperature: Optional[float]) -> bool:
    """Whether a target call with this ``temperature`` SAMPLES: None (the server's generation
    config) and any t > 0 sample; 0.0 is greedy."""
    return temperature is None or float(temperature) > GREEDY_TEMPERATURE


def decode_mode(temperature: Optional[float]) -> str:
    """The WIRE-LEVEL mode of a call carrying ``temperature`` (a vendor preset is EXPLICIT here)."""
    if temperature is None:
        return GENERATION_CONFIG
    return GREEDY if float(temperature) <= GREEDY_TEMPERATURE else EXPLICIT


def campaign_default_temperature(model_id: str) -> Optional[float]:
    """What the REGISTRY's default multi-turn decode policy sends as the target's temperature for ``model_id`` (a
    model id, legacy key or HF repo): None when that row sends nothing (:data:`GENERATION_CONFIG`, the 4,096-cap
    canonical instrument every filed multi-turn anchor was drawn under), else the temperature it sends. Refuses
    (``NoPolicy``) for a model with no row.

    NOT the harness's config-level default: since 2026-09-16 :data:`DEFAULT_CONFIG_MODE` is :data:`VENDOR_PRESET`
    (the served model's vendor row from ``configs/sdfbase_sampled_decode.yaml``, sent explicitly), a mode the
    registry's ``multi_turn`` policy rows seed only for Qwen3.5-4B (``qwen35_4b_it|multi_turn|off|vendor_preset``,
    2026-09-17) — THE Qwen default since the registry integration, so for ``qwen35_4b_it`` this returns 0.7 (sent)
    while the filed 4,096-cap anchors' row (``qwen35_4b_it|multi_turn|off|server_default``, superseded, kept by
    name) sent nothing; every other served model's default row still sends nothing. A NEW multi-turn run records
    what it sent natively (``decode_sent_json``); this helper describes the registry's default row, not any filed
    store's."""
    policy = _decode_policy.load().resolve(model_id, eval_category="multi_turn", thinking_mode="off")
    t = policy.request_params().get("temperature")
    return None if t is None else float(t)


def temperature_source(temperature: Optional[float], do_sample: bool) -> str:
    return SOURCE_SERVER_GENERATION_CONFIG if (do_sample and temperature is None) else SOURCE_REQUEST


# --------------------------------------------------------------------------- #
# The seed-replicate battery's REGISTRY vendor rows                            #
# --------------------------------------------------------------------------- #

#: the registry's sampling mode every battery TARGET row carries (the fourth segment of its policy_ref)
VENDOR_PRESET_MODE = VENDOR_PRESET
#: the version every battery row is pinned at -- part of the EXACT ref (``...@v1``), never "the default version"
VENDOR_PRESET_POLICY_VERSION = 1
#: the six sampling keys a vendor row must SEND on every request (``decode_policy.PARAM_KEYS`` by reference;
#: presence_penalty among them -- it is not server-defaultable, ``contracts.SERVER_DEFAULTABLE_SAMPLING_KEYS``
#: excludes it, so a row that leaves it to the server is NOT the vendor row)
VENDOR_ROW_SENT_KEYS = tuple(_decode_policy.PARAM_KEYS)


class VendorPolicyRefused(RuntimeError):
    """The registry row a battery target asked for cannot be sent: absent, superseded, not the vendor mode, not
    sending all six sampling keys, or not the ref the caller expected. The caller refuses to launch; nothing is
    invented in its place."""


def vendor_preset_policy_ref(model_id: str, segment: str) -> str:
    """The EXACT policy ref of a battery target row -- ``<model_id>|<segment>|off|vendor_preset@v1``: the only
    spelling :func:`registry_vendor_row` resolves, through ``PolicyFile.by_ref``."""
    return f"{model_id}|{segment}|off|{VENDOR_PRESET_MODE}@v{VENDOR_PRESET_POLICY_VERSION}"


_CAP_RULE_RE = re.compile(r"\b(\d{3,6})\s+(?:per|tokens?)\b")


def cap_in_max_new_tokens_rule(rule: Optional[str]) -> Optional[int]:
    """The per-request cap a row's ``max_new_tokens_rule`` NAMES when it is phrased ``<N> per <unit>`` / ``<N> tokens``
    (``4096 per reply`` -> 4096, ``1024 per turn`` -> 1024); None for every other phrasing (``512 store + 4096
    climb``, ``per instrument, never resolved here: ...``) -- the cap then belongs to the instrument and no
    cross-check applies (a plan cap must match a row that states one)."""
    if not rule:
        return None
    found = sorted({int(m) for m in _CAP_RULE_RE.findall(str(rule))})
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class VendorRow:
    """One resolved battery row: the exact ref, the registry ``Policy``, and the six sampling keys AS SENT (in
    ``decode_policy.PARAM_KEYS`` order) plus the export it was read from."""

    ref: str
    policy: "_decode_policy.Policy"
    decode: Dict[str, Any]
    policy_file: str
    policy_set_sha256: Optional[str]

    def record(self) -> Dict[str, Any]:
        """The provenance block a manifest / run record carries for the row (its identity, never the numbers alone --
        those ride the request and the row's ``decode`` block)."""
        p = self.policy
        return {"policy_id": p.policy_id, "policy_ref": p.policy_ref, "policy_version": int(p.policy_version),
                "model_id": p.model_id, "mode": p.mode, "decode_source": p.decode_source, "seed_rule": p.seed_rule,
                "max_new_tokens_rule": p.max_new_tokens_rule, "superseded_by_policy_id": p.superseded_by_policy_id,
                "policy_file": self.policy_file, "policy_set_sha256": self.policy_set_sha256}


def registry_vendor_row(model_key: str, segment: str, *, path: Optional[str] = None,
                        expect_ref: Optional[str] = None) -> VendorRow:
    """THE battery's decode authority for one target leg:
    the row ``<model_id>|<segment>|off|vendor_preset@v1`` of the served model (``model_key`` normalised through the
    export's ``model_spellings``: ``qwen35_4b`` -> ``qwen35_4b_it``), resolved by ``PolicyFile.by_ref`` ALONE and
    refused (:class:`VendorPolicyRefused`) when it is ABSENT (``NoPolicy`` -- numbers are never invented),
    SUPERSEDED (``by_ref`` does not exclude a superseded row; ``Policy.superseded_by_policy_id`` carries the flag),
    not in the vendor mode, or not SENDING every one of the six sampling keys. ``expect_ref`` (a ref typed on a CLI)
    must equal the derived ref exactly -- a row of another model or segment refuses.

    NEVER ``PolicyFile.resolve(..., mode="vendor_preset")``: with ``mode`` it ignores ``eval_id`` and returns the
    model-wide ``sampling_mode_policy`` row (the IFEval row for qwen35_4b_it); never ``find()`` or a default lookup
    either (tests/test_bon_vendor_preset.py pins the wrong lookup as known-wrong). ``path`` names the
    export to read (tests hand a fixture); None = ``$SISTEER_DECODE_POLICY`` / the committed export."""
    pf = _decode_policy.load(path)
    model_id = pf.model_id(model_key)
    ref = vendor_preset_policy_ref(model_id, segment)
    if expect_ref is not None and str(expect_ref) != ref:
        raise VendorPolicyRefused(f"policy ref {expect_ref!r} is not the served model's {segment} vendor row {ref!r} "
                                  f"(model key {model_key!r} -> {model_id}); refusing")
    try:
        pol = pf.by_ref(ref)
    except _decode_policy.NoPolicy as e:
        raise VendorPolicyRefused(f"the decode-policy export {pf.path} has no row {ref!r} ({e}); the harness never "
                                  "invents sampling numbers -- the row lands with the registry (D9), not here") from e
    if pol.superseded_by_policy_id:
        raise VendorPolicyRefused(f"decode-policy row {ref!r} is SUPERSEDED by {pol.superseded_by_policy_id!r}; a "
                                  "superseded row is never sent -- name the successor by its own exact ref")
    if pol.mode != VENDOR_PRESET_MODE:
        raise VendorPolicyRefused(f"decode-policy row {ref!r} is mode {pol.mode!r}, not {VENDOR_PRESET_MODE!r}")
    params = pol.request_params()
    missing = [k for k in VENDOR_ROW_SENT_KEYS if k not in params]
    if missing:
        raise VendorPolicyRefused(f"decode-policy row {ref!r} does not SEND {missing} (params_sent "
                                  f"{list(pol.params_sent or ())}); the battery's contract sends all six sampling keys "
                                  "on every target request (presence_penalty cannot be a server default)")
    for k in VENDOR_ROW_SENT_KEYS:
        v = params[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise VendorPolicyRefused(f"decode-policy row {ref!r}: {k}={v!r} is not a number")
    decode = {k: params[k] for k in VENDOR_ROW_SENT_KEYS}
    return VendorRow(ref=ref, policy=pol, decode=decode, policy_file=str(pf.path),
                     policy_set_sha256=(pf.header or {}).get("policy_set_sha256"))
