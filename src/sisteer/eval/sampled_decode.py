"""The vendor-preset decode DEFAULT of the static and over-refusal drivers.

The user's ruling (2026-09-11, relayed): the SAMPLED re-run is the canonical
static and over-refusal instrument — the vendor sampling preset per item seed,
the static attack families at cap 4,096 with the item-filtered rung to 8,192,
the over-refusal controls at 512 -> 2,048 -> 8,192 — and main selects it BY
DEFAULT. Greedy T=0 is an explicit opt-in (``--greedy``), recorded in
provenance.

HOW THE DEFAULT IS INJECTED. The sampled re-run keyed everything on a
config's top-level ``sampled_decode`` block (``jailbreak_run._sampled_decode``,
``cell_controls.controls_gen_config``), rendered per model by
``scripts/sdfbase/full22/sampled/make_sampled_configs.py`` from the ONE data
file ``configs/sdfbase_sampled_decode.yaml``. This module moves that render
INTO the drivers: when a served run's config carries NO block and ``--greedy``
was not given, :func:`default_sampled_block` builds the generator-shaped block
for the CONFIG's served model (matched to a preset by HF id and, when the
config pins one, revision) and the driver injects it into its in-memory
config — so every validator, store key, sidecar, ladder rung and provenance
field downstream behaves exactly as the sampled re-run's did. A served model
with NO row in the data file REFUSES (:class:`DecodePresetUnavailable`) naming
``--greedy`` as the way out: a temperature nobody vouched for is never a
default (the same rule as ``tinycap.contracts.sampling_for``). A config that
carries its own block is untouched (the sampled re-run's generated configs).

WHAT ``--greedy`` RECORDS. The opt-in leaves the config as it is (the greedy
store's identity, ``controls_gen_config(cfg) == {max_new_tokens, do_sample:
False}``, is byte-identical to every existing greedy store's), and the driver
writes :func:`greedy_opt_in_record` into provenance (``served_target.json`` /
``gen_stats.json``) so a reader can tell "greedy because someone asked" from
"greedy because the code predates the default".

RESOLUTION IS BY SERVED MODEL. The data file's rows name ``model`` +
``revision``; a config names ``model`` (+ ``model_revision``). A row matches
on the HF id; when both name a revision they must agree (a preset read from
one snapshot never rides a config that serves another — the generator's
``check_pins`` rule, applied at the driver). The roster is NOT consulted: a
row for a target the roster does not (yet) name — Llama-3.1-8B-Instruct's,
whose wave lands separately — is a legal row.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
#: the ONE data file (make_sampled_configs.PRESETS)
DECODE_PRESETS_PATH = REPO / "configs" / "sdfbase_sampled_decode.yaml"
#: the sampler keys a preset may carry, in request order (make_sampled_configs.ALLOWED_SAMPLER_KEYS ==
#: bon_served.SAMPLER_KEYS == cell_controls.SAMPLED_DECODE_SAMPLER_KEYS; pinned equal by test)
SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
SOURCE_KINDS = ("generation_config", "model_card", "vendor_recommendation")
#: the provenance record's ``decode_default`` values
DEFAULT_VENDOR_PRESET = "vendor_preset_default"
DEFAULT_GREEDY_OPT_IN = "greedy_opt_in"
GREEDY_FLAG = "--greedy"


class DecodePresetUnavailable(RuntimeError):
    """No vendor row for the served model (or a revision mismatch): the
    driver refuses rather than sampling at a number nobody vouched for."""


class DecodePresetsMalformed(RuntimeError):
    """The data file does not satisfy its own contract (a non-sampler key, a
    bad source_kind, a row without a source, a vendor_recommendation without
    the shipped file's values beside it)."""


def load_decode_presets(path: Optional[Path] = None) -> Dict[str, Any]:
    """The data file, validated by the generator's rules (make_sampled_configs.load_presets'
    per-row checks, torch-free and importable from the drivers): every preset carries only
    sampler keys, a known ``source_kind``, a non-empty ``source``, ``model`` + ``revision``;
    a ``vendor_recommendation`` records ``shipped_generation_config`` that DIFFERS from the
    preset. The ``readouts`` block is passed through unread (the generator validates it)."""
    path = Path(path) if path is not None else DECODE_PRESETS_PATH
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict) or not isinstance(d.get("presets"), dict) or not d["presets"]:
        raise DecodePresetsMalformed(f"{path}: no presets mapping")
    for k in ("arm", "n_trials", "seed_base", "applies_to"):
        if k not in d:
            raise DecodePresetsMalformed(f"{path}: top-level {k!r} missing")
    if isinstance(d["seed_base"], bool) or not isinstance(d["seed_base"], int):
        raise DecodePresetsMalformed(f"{path}: seed_base must be an int, got {d['seed_base']!r}")
    for key, p in d["presets"].items():
        if not isinstance(p, dict) or not isinstance(p.get("preset"), dict) or not p["preset"]:
            raise DecodePresetsMalformed(f"{path}: preset {key} has no preset mapping")
        bad = [k for k in p["preset"] if k not in SAMPLER_KEYS]
        if bad:
            raise DecodePresetsMalformed(f"{path}: preset {key} carries non-sampler keys {bad} "
                                         f"(allowed: {SAMPLER_KEYS})")
        for k, v in p["preset"].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise DecodePresetsMalformed(f"{path}: preset {key}.{k}={v!r} is not a number")
        if "temperature" not in p["preset"]:
            raise DecodePresetsMalformed(f"{path}: preset {key} states no temperature")
        if p.get("source_kind") not in SOURCE_KINDS:
            raise DecodePresetsMalformed(f"{path}: preset {key} source_kind {p.get('source_kind')!r} "
                                         f"must be one of {SOURCE_KINDS}")
        shipped = p.get("shipped_generation_config")
        if p["source_kind"] == "vendor_recommendation" and not shipped:
            raise DecodePresetsMalformed(f"{path}: preset {key} is a vendor_recommendation but records no "
                                         "shipped_generation_config (the file's values must sit beside the "
                                         "preset that runs)")
        if shipped and shipped == p["preset"]:
            raise DecodePresetsMalformed(f"{path}: preset {key} records shipped_generation_config identical "
                                         "to the preset — the field exists to show a DIFFERENCE")
        if not p.get("source"):
            raise DecodePresetsMalformed(f"{path}: preset {key} has no source")
        for k in ("model", "revision"):
            if not isinstance(p.get(k), str) or not p[k]:
                raise DecodePresetsMalformed(f"{path}: preset {key} has no {k}")
    g = ((d.get("readouts") or {}).get("gsm8k") or {}) if isinstance(d.get("readouts"), dict) else {}
    for key, mode in (g.get("thinking_by_key") or {}).items():
        # a model with NO thinking mode runs the GSM8K readout thinking-OFF (the sdfbase-llama8b wave's
        # block): keys must be preset keys, values the strings "on" / "off", each with its stated reason
        if key not in d["presets"]:
            raise DecodePresetsMalformed(f"{path}: readouts.gsm8k.thinking_by_key names {key}, which has no preset")
        if mode not in ("on", "off"):
            raise DecodePresetsMalformed(f"{path}: readouts.gsm8k.thinking_by_key.{key} must be \"on\" or \"off\", got {mode!r}")
        if not (g.get("thinking_by_key_source") or {}).get(key):
            raise DecodePresetsMalformed(f"{path}: readouts.gsm8k.thinking_by_key.{key} needs thinking_by_key_source.{key}")
    return d


def presets_sha256(path: Optional[Path] = None) -> str:
    """The data file's sha256 — recorded in provenance so a store says WHICH table it drew under."""
    path = Path(path) if path is not None else DECODE_PRESETS_PATH
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preset_key_for_model(presets: Mapping[str, Any], model: Any, revision: Any = None) -> str:
    """The preset KEY whose row serves ``model`` (an HF id; the revision, when both
    name one, must agree). Refuses (:class:`DecodePresetUnavailable`) an unknown model
    or a revision mismatch, naming :data:`GREEDY_FLAG` as the way out."""
    if not isinstance(model, str) or not model:
        raise DecodePresetUnavailable(
            f"the config names no served model (model={model!r}) — the vendor decode preset "
            f"resolves per served model; pass {GREEDY_FLAG} for argmax decoding")
    rows = {k: p for k, p in presets["presets"].items() if p["model"] == model}
    if not rows:
        known = sorted(p["model"] for p in presets["presets"].values())
        raise DecodePresetUnavailable(
            f"no vendor sampling preset for served model {model!r} in "
            f"the decode presets file (rows: {known}) — a temperature nobody vouched for is "
            f"never a default; add a verified row, or pass {GREEDY_FLAG} for argmax decoding "
            "(recorded as the opt-in)")
    if revision:
        matched = {k: p for k, p in rows.items() if p["revision"] == revision}
        if not matched:
            raise DecodePresetUnavailable(
                f"the vendor sampling preset for {model!r} was read from revision "
                f"{[p['revision'] for p in rows.values()]} but this config serves revision "
                f"{revision!r} — a preset read from one snapshot never rides a config that serves "
                f"another; re-verify the row for this revision, or pass {GREEDY_FLAG}")
        rows = matched
    if len(rows) > 1:
        # two KEYS may serve ONE checkpoint with the SAME preset (the worked example qwen35_4b_demo serves the paper's Qwen3.5-4B and
        # carries the qwen35_4b row verbatim, 2026-10-08): identical rows are one preset, not an ambiguity — the first key in file
        # order (the paper's) names it; rows that DIFFER for one (model, revision) — in preset, source_kind, revision or the
        # shipped generation config they record — are still refused below
        first = next(iter(rows))
        same = {k: p for k, p in rows.items()
                if p.get("preset") == rows[first].get("preset") and p.get("source_kind") == rows[first].get("source_kind")
                and p.get("revision") == rows[first].get("revision")
                and p.get("shipped_generation_config") == rows[first].get("shipped_generation_config")}
        if len(same) == len(rows):
            rows = {first: rows[first]}
    if len(rows) != 1:
        raise DecodePresetUnavailable(
            f"{len(rows)} presets name served model {model!r}: {sorted(rows)} — ambiguous; "
            "the data file must name one row per (model, revision)")
    return next(iter(rows))


def default_sampled_block(presets: Mapping[str, Any], key: str, *, trial: int,
                          static_cap: Optional[int], controls_cap: Optional[int]) -> Dict[str, Any]:
    """The generator-shaped ``sampled_decode`` block for preset ``key``
    (make_sampled_configs.expected_block's data, minus the per-config
    ``escalated_from`` a rendered rung carries; ``readouts`` included so both
    drivers' validators admit it): ``{arm, trial, n_trials, seed_base, <preset
    keys>, source, source_kind, applies_to, static_cap, controls_cap, readouts}``.
    ``static_cap`` / ``controls_cap`` are the CONFIG's own caps
    (``attacks.static_max_new_tokens`` / the controls cap, ``cell_controls.controls_cap``:
    ``attacks.controls_max_new_tokens``, else ``attacks.target_max_new_tokens``) —
    the validators hold the block to them."""
    p = presets["presets"][key]
    block: Dict[str, Any] = {"arm": presets["arm"], "trial": int(trial),
                             "n_trials": int(presets["n_trials"]),
                             "seed_base": int(presets["seed_base"])}
    block.update({k: p["preset"][k] for k in SAMPLER_KEYS if k in p["preset"]})
    block.update({"source": p["source"], "source_kind": p["source_kind"],
                  "applies_to": list(presets["applies_to"])})
    if static_cap is not None:
        block["static_cap"] = int(static_cap)
    if controls_cap is not None:
        block["controls_cap"] = int(controls_cap)
    if isinstance(presets.get("readouts"), dict):
        block["readouts"] = _readouts_for(presets, key)
    return block


def _readouts_for(presets: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """make_sampled_configs.expected_readouts' data (the generator's rendering,
    kept equal by test): the per-readout sub-block the drivers pass through."""
    r = presets["readouts"]
    g = r.get("gsm8k") or {}
    p = presets["presets"][key]
    think = (g.get("presets") or {}).get(key)
    mode = (g.get("thinking_by_key") or {}).get(key, g.get("thinking"))
    if think is not None:
        preset, source = dict(think["preset"]), think["source"]
    else:
        preset = {k: p["preset"][k] for k in SAMPLER_KEYS if k in p["preset"]}
        reason = ((g.get("thinking_by_key_source") or {}).get(key)
                  if mode != g.get("thinking") else g.get("reuse_source"))
        source = (f"{p['model']} @ {p['revision'][:8]}: {reason} -- the preset that runs "
                  f"({', '.join(f'{k} {v}' for k, v in preset.items())}; source_kind {p['source_kind']})")
    out: Dict[str, Any] = {}
    if g:
        out["gsm8k"] = {"cap": int(g["cap"]), "thinking": mode, "preset": preset, "source": source}
    if isinstance(r.get("mmlu"), dict):
        out["mmlu"] = {"instrument": r["mmlu"]["instrument"], "note": r["mmlu"]["note"]}
    return out


def decode_resolution_record(*, mode: str, key: Optional[str], presets: Optional[Mapping[str, Any]],
                             model: Any, revision: Any, reason: str,
                             path: Optional[Path] = None) -> Dict[str, Any]:
    """The provenance record every served run writes beside its store
    (``served_target.json`` / ``gen_stats.json``): HOW the decode was chosen —
    ``vendor_preset_default`` (the data file's row for the served model, named
    by key, model, revision, source) or ``greedy_opt_in`` (``--greedy``) — never
    a store whose decode arm has to be inferred from its rows."""
    path = Path(path) if path is not None else DECODE_PRESETS_PATH
    rec: Dict[str, Any] = {"decode_default": mode, "reason": reason,
                           "served_model": model, "served_revision": revision,
                           "presets_file": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
                           "presets_sha256": presets_sha256(path) if Path(path).is_file() else None,
                           "preset_key": key}
    if presets is not None and key is not None:
        p = presets["presets"][key]
        rec.update({"preset_model": p["model"], "preset_revision": p["revision"],
                    "source_kind": p["source_kind"], "source": p["source"],
                    "preset": {k: p["preset"][k] for k in SAMPLER_KEYS if k in p["preset"]}})
    return rec


def greedy_opt_in_record(*, model: Any, revision: Any, why: str = GREEDY_FLAG) -> Dict[str, Any]:
    """:func:`decode_resolution_record` for the ``--greedy`` opt-in."""
    return decode_resolution_record(mode=DEFAULT_GREEDY_OPT_IN, key=None, presets=None,
                                    model=model, revision=revision,
                                    reason=f"argmax decoding by explicit opt-in ({why}); the vendor "
                                           "sampling preset is the default")


def resolve_default_decode(cfg: Mapping[str, Any], *, greedy: bool, trial: int,
                           static_cap: Optional[int] = None, controls_cap: Optional[int] = None,
                           model: Any = None, revision: Any = None,
                           path: Optional[Path] = None) -> "tuple[dict, dict]":
    """``(cfg_out, record)``: the config to drive and the provenance record.

    - the config already carries ``sampled_decode`` -> returned UNTOUCHED (a
      copy), record ``decode_default: "config_block"`` (the sampled re-run's
      generated configs; their block is the contract);
    - ``greedy`` -> untouched, :func:`greedy_opt_in_record`;
    - otherwise the served model's row (``model`` / ``revision`` default to the
      config's ``model`` / ``model_revision``) is resolved and the block
      injected; no row -> :class:`DecodePresetUnavailable` (the caller turns it
      into a refusal naming ``--greedy``)."""
    path = Path(path) if path is not None else DECODE_PRESETS_PATH   # resolved at CALL time (tests / callers re-point it)
    cfg_out = dict(cfg)
    model = cfg.get("model") if model is None else model
    revision = cfg.get("model_revision") if revision is None else revision
    if cfg.get("sampled_decode") is not None:
        return cfg_out, {"decode_default": "config_block", "served_model": model, "served_revision": revision,
                         "reason": "the config carries its own sampled_decode block (a sampled re-run "
                                   "config); nothing injected"}
    if greedy:
        return cfg_out, greedy_opt_in_record(model=model, revision=revision)
    presets = load_decode_presets(path)
    key = preset_key_for_model(presets, model, revision)
    cfg_out["sampled_decode"] = default_sampled_block(presets, key, trial=trial,
                                                      static_cap=static_cap, controls_cap=controls_cap)
    rec = decode_resolution_record(mode=DEFAULT_VENDOR_PRESET, key=key, presets=presets, model=model,
                                   revision=revision, path=path,
                                   reason="the vendor sampling preset is the canonical instrument's "
                                          "default (the user's ruling 2026-09-11); the config carried no "
                                          "sampled_decode block, so the served model's row was injected")
    return cfg_out, rec
