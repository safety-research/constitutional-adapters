"""Served-path runner for the misalignment eval families (WP-B).

Generates responses for every (cell, item, sample) of one family against an
OpenAI-compatible endpoint — the campaign's routing proxy
(``sisteer.capeval.serving.proxy``; virtual models ``petri__*``) or, with
``plain=True``, a bare vLLM server (base model + LoRA modules by served
name). One request per sample: the proxy refuses ``n>1`` on steered routes,
so sampling multiplicity is always a request count here, never a request
knob. Nothing in this module imports torch.

Message rules (``compose_system`` puts the SI FIRST):

============  ==========================================  =====================
family        prompted cell (``si_id`` set)               non-prompted cell
============  ==========================================  =====================
``agentic``   system = SI + "\\n\\n" + scenario system      system = scenario system
``em8q``      system = SI                                 no system message
============  ==========================================  =====================

The user message is the scenario's ``user_prompt`` (agentic; it already
carries the emails) or the Betley question (em8q). The SI bytes are the
exact TrainTest text the proxy hash-verifies (``si_text_for``) — a prompted
request whose system message does not START with those bytes is refused
upstream as ``si_prefix_mismatch``, and that refusal (any 4xx) aborts the
run here rather than being retried into N identical refusals.

Layout under ``out_dir``::

    <cell_id>/responses.jsonl   one row per (item_id, sample_idx); resumable
    <cell_id>/meta.json         cell fields, family, params, suite revision,
                                started/finished, endpoint identity

Response rows: ``{family, cell_id, virtual_model, item_id, sample_idx,
behavior, prompt_sha, response, finish_reason, usage, model_reported}``
(+ ``reasoning`` when the server returned ``reasoning_content``). item_id is
the scenario id (agentic) or ``BETLEY_8Q_SOURCE_IDS[qid]`` (em8q) so rows pair
against the same target's ``none`` cell on (item_id, sample_idx).

CLI::

    python -m sisteer.capeval.misalign.runner --proxy-url URL --cells cells.json \\
        --family agentic|em8q --n N --out DIR [--concurrency K] [--max-tokens M] \\
        [--temperature T] [--behaviors leaking murder] \\
        [--exclude nemotron3nano4b:leaking] [--plain] [--api-key KEY] \\
        [--sampler t1|vendor] [--seed-policy none|per_response --seed-base S]

Under ``--seed-policy per_response`` every
request carries ``seed = misalign_seed(S, behavior, item_id, sample_idx)`` and
every row additionally records ``seed, seed_rule, seed_base, sampling`` (as
sent), ``decode_policy_id`` (the registered vendor row under ``--sampler
vendor``) and ``completion_id`` (the response's ``id``); ``seed_policy`` /
``seed_base`` join meta.json's resume identity and, with ``decode_policy_id`` /
``seed_rule`` / ``seed_rule_text``, are written to meta.json ONLY under
``per_response`` (a ``none`` tree reads them as defaults). The default ``none``
is the runner as it was, byte for byte — the wire, the rows and meta.json.

``cells.json`` is either a list of cell dicts or ``{"cells": [...], ...}``
(extra top-level keys such as ``target`` / ``model_id`` / ``model_revision``
are recorded into every meta.json). Each cell: ``{cell_id, virtual_model,
serving_identity, intervention_hash, si_id}`` (+ optional ``kind``); ``si_id``
is non-null ONLY on prompted cells.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from sisteer.capeval.manifest import canonical_json, sha256_hex16
from sisteer.capeval.petri.registry_adapter import HARNESS_TO_TRAINTEST_SI
from sisteer.capeval.serving.proxy import FORBIDDEN_CLIENT_KEYS, VMODEL_PREFIX
from sisteer.data.safety_texts import BETLEY_8Q, BETLEY_8Q_SOURCE_IDS
from sisteer.eval.realign_suites import compose_system, prompt_sha
from sisteer.eval import request_evidence_client as rev  # request-evidence hook (off unless SISTEER_REQUEST_EVIDENCE_DIR)


class MisalignRunnerError(RuntimeError):
    """A runner refusal: bad cells, a missing virtual model, an endpoint
    4xx, a resume onto a different configuration."""


FAMILIES = ("agentic", "em8q")
#: the Betley-8Q suite revision stamped on em8q rows (task_version)
EM8Q_SUITE_REVISION = "betley8q.v1"
#: response caps per family (plan: agentic scenarios ~3k prompt tokens,
#: cap responses at 3000; Betley's paper samples 600 new tokens)
DEFAULT_MAX_TOKENS = {"agentic": 3000, "em8q": 600}
#: the agentic behaviors this eval runs (blackmail omitted)
AGENTIC_BEHAVIORS = ("leaking", "murder")
#: the dummy bearer the proxy accepts by default (the same value as
#: serving/petri_target.py's default api key; the proxy authenticates the
#: token, so a wrong one fails at the /v1/models check)
DUMMY_BEARER = "petri-proxy"
#: client -> endpoint timeout; matches the proxy's CLIENT_TIMEOUT_S regime
#: (strictly longer than its 7,200 s upstream hop so the proxy's 504 — a
#: recorded failure — always arrives before this socket gives up)
DEFAULT_TIMEOUT_S = 7500.0
#: truncation ladder (the rule, 2026-09-10: rerun truncated responses when they exceed 1%,
#: escalating up to a 16k cap, into a NEW root beside the original). A truncated response whose
#: text is a repetition loop (zlib ratio below this) is degenerate — a longer cap only yields a
#: longer loop — so it is reported as degenerate, not re-laddered.
RERUN_MIN_ZLIB_RATIO = 0.2
#: request keys this client never sends: the proxy-forbidden selectors, plus
#: the sampling-multiplicity and streaming knobs the proxy refuses
NEVER_SEND = tuple(FORBIDDEN_CLIENT_KEYS) + ("n", "best_of", "stream")

#: the samplers the runner can send. Every sampling key on the wire is EXPLICIT — nothing is left to
#: the server's defaults (vLLM fills an omitted key from the model's generation_config.json, or from
#: its own neutral defaults for a model that ships none — Qwen3.5), so the recorded request IS the
#: decode setting.
#:   t1     — the campaign's instrument: temperature alone (1.0 unless --temperature), as every
#:            2026-09-10/11 panel ran;
#:   vendor — the served model's own recommended row for its served thinking mode, frozen on main as
#:            tinycap.contracts.QWEN35_VENDOR_ROWS @817266f2 (this branch carries no tinycap package,
#:            so the row is restated here, verbatim); resolved from the proxy's identity block
#:            (base_model + thinking_mode) and REFUSED for a model with no row — never a default.
#:            qwen's T=1.0 unsteered baseline loops on 23% of murder scenarios, so that panel is
#:            re-sampled under the vendor preset so its objects can be scored.
SAMPLER_T1 = "t1"
SAMPLER_VENDOR = "vendor"
SAMPLERS = (SAMPLER_T1, SAMPLER_VENDOR)
#: the sampling keys a payload may carry (vLLM's OpenAI-compatible chat body; top_k / min_p /
#: repetition_penalty are its documented extra fields, forwarded by the proxy untouched)
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
# THE DECODE-POLICY AUTHORITY (results registry): both samplers' numbers are READ from the committed
# export configs/registry/decode_policy.json (stdlib-only reader; $SISTEER_DECODE_POLICY overrides the path):
#   t1     -> the `<model>|agentic|off|runner_forced_T1` rows (ruling decode-agentic-runner-T1-2026-09-11), one
#             temperature for every served model (asserted identical);
#   vendor -> the `<model>|agentic|off|vendor` rows, keyed here by (HF model id, served thinking mode); a (model,
#             mode) with no row is REFUSED, never guessed — exactly the old frozen-table behaviour.
from sisteer.registry import decode_policy as _decode_policy  # noqa: E402

_DECODE_POLICY = _decode_policy.load()
#: the t1 sampler's temperature: the registry's runner_forced_T1 row (1.0 on every served model)
T1_TEMPERATURE: float = float(_DECODE_POLICY.model_independent(
    eval_category="agentic", thinking_mode="off", mode="runner_forced_T1").numbers()["temperature"])


def _vendor_rows() -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for policy in _DECODE_POLICY.find(eval_category="agentic", mode="vendor"):
        hf = _DECODE_POLICY.hf_repo(policy.model_id)
        if hf is None:
            raise MisalignRunnerError(f"decode policy {policy.policy_id} names a model with no HF repo in the export")
        numbers = policy.numbers()
        out[(str(hf), policy.thinking_mode)] = {k: numbers[k] for k in SAMPLING_KEYS if numbers.get(k) is not None}
    return out


#: (HF model id, served thinking mode) -> the vendor row. Thinking-OFF serving = the instruct row
#: (Qwen3.5 model card 'Best Practices', "Instruct (non-thinking) mode for general tasks" —
#: contracts.QWEN35_VENDOR_ROWS["instruct_general"]; the registry rows cite it).
#: A READERS' TABLE (the registry entry-point oracle pins it); the row a run SENDS is resolved by
#: :func:`resolve_vendor_policy` — the exact registered policy ref, never this listing.
VENDOR_SAMPLING: Dict[Tuple[str, str], Dict[str, Any]] = _vendor_rows()
#: kept for readers of the name: the 4B instruct-general row, by reference
_QWEN35_INSTRUCT_GENERAL = dict(VENDOR_SAMPLING[("Qwen/Qwen3.5-4B", "off")])

#: THE VENDOR ROW'S ONE AUTHORITY: the exact
#: registered policy ref `<model_id>|agentic|<thinking_mode>|vendor@v<VENDOR_POLICY_VERSION>` through
#: `PolicyFile.by_ref`, `superseded_by_policy_id is None` asserted — never `find(mode=...)` (a listing, not
#: an authority) and never `resolve(mode=...)` (ignores eval_id; reads the model-wide sampling_mode_policy).
#: The served model comes from the proxy's identity block (`base_model`, an HF repo the export's
#: `model_spellings` map to the model_id) and its served thinking mode.
VENDOR_POLICY_EVAL_CATEGORY = "agentic"
VENDOR_POLICY_MODE = "vendor"
VENDOR_POLICY_VERSION = 1


def vendor_policy_ref(model_id: str, thinking_mode: str) -> str:
    """The exact policy ref the vendor sampler resolves: ``qwen35_4b_it|agentic|off|vendor@v1`` for the
    4B instruct model served thinking-OFF."""
    return f"{model_id}|{VENDOR_POLICY_EVAL_CATEGORY}|{thinking_mode}|{VENDOR_POLICY_MODE}@v{VENDOR_POLICY_VERSION}"


def resolve_vendor_policy(identity: Optional[Mapping[str, Any]], policy_file=None):
    """The registered vendor decode-policy row (``decode_policy.Policy``) for the proxy's served
    ``(base_model, thinking_mode)``, by EXACT ref; refuses when the identity block is absent (a bare
    endpoint), when the export has no such row (never a default), or when the row is superseded
    (a superseded row is never sent — the successor is the authority, and it is looked up by ITS ref)."""
    pf = _DECODE_POLICY if policy_file is None else policy_file
    if not identity:
        raise MisalignRunnerError(
            "--sampler vendor needs the proxy's identity block (/v1/models petri_proxy: base_model + "
            "thinking_mode) — a bare endpoint has none, and a row is never guessed")
    model, mode = identity.get("base_model"), identity.get("thinking_mode")
    ref = vendor_policy_ref(pf.model_id(str(model)), str(mode))
    try:
        pol = pf.by_ref(ref)
    except _decode_policy.NoPolicy as e:
        raise MisalignRunnerError(
            f"--sampler vendor: no frozen vendor sampling row for {model!r} in thinking mode {mode!r} "
            f"(registry ref {ref!r}: {e}; rows listed: {sorted(VENDOR_SAMPLING)}) — add the vendor's row, "
            "never a default") from e
    if pol.superseded_by_policy_id is not None:
        raise MisalignRunnerError(
            f"--sampler vendor: decode policy {pol.policy_id} is superseded by {pol.superseded_by_policy_id!r}; "
            "a superseded row is never sent — resolve the successor by its own ref")
    return pol


def vendor_sampling_for(identity: Optional[Mapping[str, Any]], policy_file=None) -> Dict[str, Any]:
    """The vendor row's sampling keys AS SENT (the registered row's ``request_params`` restricted to
    :data:`SAMPLING_KEYS`) for the proxy's served ``(base_model, thinking_mode)`` — :func:`resolve_vendor_policy`'s
    refusals apply."""
    pol = resolve_vendor_policy(identity, policy_file)
    params = pol.request_params()
    row = {k: params[k] for k in SAMPLING_KEYS if params.get(k) is not None}
    if not row:
        raise MisalignRunnerError(f"--sampler vendor: decode policy {pol.policy_id} sends no sampling key")
    return row


# --- the per-response decode seed -------------------------------------------------------------------------------
#: `--seed-policy`: `none` (today's runner: no `seed` on the wire, the engine draws unseeded) or `per_response`
#: (every request carries `seed = misalign_seed(seed_base, behavior, item_id, sample_idx)`; the base is the
#: lane's literal training seed s, `--seed-base`). Default `none`, so every existing store and argv is unchanged.
SEED_POLICY_NONE = "none"
SEED_POLICY_PER_RESPONSE = "per_response"
SEED_POLICIES = (SEED_POLICY_NONE, SEED_POLICY_PER_RESPONSE)
#: the rule's name, recorded on every seeded row and in meta.json (`seed_rule`), beside `seed_base`
SEED_RULE_PER_RESPONSE = "misalign_per_response_v1"
#: the rule's literal spelling (the static / controls rules' shape: sha256 of a `|`-joined key, first 4 bytes big-endian)
SEED_RULE_TEXT = 'int.from_bytes(sha256(f"{seed_base}|misalign|{behavior}|{item_id}|{sample_idx}").digest()[:4], "big")'


def misalign_seed(seed_base: int, behavior: str, item_id: str, sample_idx: int) -> int:
    """The per-response seed under ``per_response``: the first 4 bytes (big-endian) of
    ``sha256(f"{seed_base}|misalign|{behavior}|{item_id}|{sample_idx}")`` — the static families'
    ``static_sampling_seed`` / the controls' ``sampling_seed`` shape with ``misalign`` as the readout
    and the row's behaviour + item + sample index as the item key. Cell-INDEPENDENT on purpose
    (common random numbers across cells), and reused when an at-cap response is re-drawn at a
    larger cap (a re-draw, not a continuation)."""
    key = f"{int(seed_base)}|misalign|{behavior}|{item_id}|{int(sample_idx)}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def seed_behavior(family: str, item: Mapping[str, Any]) -> str:
    """The rule's ``behavior`` term: the agentic row's behaviour; the family name for an item that
    carries none (em8q's Betley questions), so the term is never the string ``None``."""
    b = item.get("behavior")
    return str(b) if b else str(family)


def check_seed_policy(seed_policy: str, seed_base: Optional[int]) -> None:
    """``per_response`` needs a base; ``none`` refuses one (a base with no policy to consume it would be
    recorded as identity nobody applied)."""
    if seed_policy not in SEED_POLICIES:
        raise MisalignRunnerError(f"unknown seed policy {seed_policy!r} (one of {SEED_POLICIES})")
    if seed_policy == SEED_POLICY_PER_RESPONSE:
        if seed_base is None or isinstance(seed_base, bool) or not isinstance(seed_base, int):
            raise MisalignRunnerError("--seed-policy per_response needs --seed-base <int> (the literal training "
                                      "seed of the served object's view)")
    elif seed_base is not None:
        raise MisalignRunnerError("--seed-base rides --seed-policy per_response; under the default policy "
                                  "`none` no request carries a seed, so a base would be recorded and never applied")


def resolve_sampling(sampler: str, temperature: float,
                     identity: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The sampling keys every request of this run carries."""
    if sampler == SAMPLER_T1:
        return {"temperature": float(temperature)}
    if sampler == SAMPLER_VENDOR:
        return vendor_sampling_for(identity)
    raise MisalignRunnerError(f"unknown sampler {sampler!r} (one of {SAMPLERS})")

CELL_REQUIRED = ("cell_id", "virtual_model", "serving_identity",
                 "intervention_hash", "si_id")
RESPONSE_FIELDS = ("family", "cell_id", "virtual_model", "item_id",
                   "sample_idx", "behavior", "prompt_sha", "response",
                   "finish_reason", "usage", "model_reported")


# --- SI text (torch-free) -----------------------------------------------------


def si_text_for(si_id: str) -> str:
    """The exact SI text for a harness SI id (``safety`` / ``constitution`` /
    ``irrelevant`` / …) — byte-identical to
    ``registry_adapter._traintest_si_text_bytes(HARNESS_TO_TRAINTEST_SI[si_id])``
    (pinned by tests/test_misalign_runner.py), resolved WITHOUT the torch
    stack that ``sisteer.eval.jailbreak.resolve_instruction`` drags in: the
    ``SAFETY_SI`` module constant for ``safety_refusal``, else
    ``instructions.BY_ID``. An unknown id is a refusal."""
    from sisteer.data.instructions import BY_ID, SAFETY_SI

    tt_si = HARNESS_TO_TRAINTEST_SI.get(si_id)
    if tt_si is None:
        raise MisalignRunnerError(
            f"si_id {si_id!r} has no TrainTest translation (harness vocabulary "
            f"{sorted(HARNESS_TO_TRAINTEST_SI)})")
    if tt_si == SAFETY_SI.id:
        instr = SAFETY_SI
    elif tt_si in BY_ID:
        instr = BY_ID[tt_si]
    else:
        raise MisalignRunnerError(
            f"TrainTest SI {tt_si!r} (harness {si_id!r}) is not in the "
            "instruction registry")
    text = instr.text
    if not isinstance(text, str) or not text:
        raise MisalignRunnerError(f"SI {si_id!r} resolved to empty text")
    return text


# --- items -----------------------------------------------------------------------


def _agentic_source() -> Tuple[List[dict], str]:
    """The agentic scenario rows + their suite revision. Prefers WP-A's
    ``load_scenarios_v2`` (leaking + murder); until it lands, falls back to
    the v1 artifact filtered to leaking, with a printed notice."""
    from sisteer.eval.agentic_misalignment import loader

    if hasattr(loader, "load_scenarios_v2"):
        return list(loader.load_scenarios_v2()), str(loader.SUITE_REVISION_V2)
    print("[misalign.runner] load_scenarios_v2 not present — falling back to "
          "the v1 artifact filtered to leaking (murder unavailable)",
          file=sys.stderr)
    rows = [r for r in loader.load_scenarios() if r["behavior"] == "leaking"]
    return rows, str(loader.SUITE_REVISION)


def agentic_items(behaviors: Optional[Sequence[str]] = None,
                  rows: Optional[Sequence[Mapping[str, Any]]] = None,
                  ) -> List[dict]:
    """Agentic items: ``{item_id, behavior, system_prompt, user_prompt}`` per
    scenario (``item_id == scenario_id == condition``), filtered to
    ``behaviors`` (default :data:`AGENTIC_BEHAVIORS`). A requested behavior
    with no scenarios is a refusal, never an empty family."""
    if rows is None:
        rows, _ = _agentic_source()
    want = tuple(behaviors) if behaviors else AGENTIC_BEHAVIORS
    items = []
    for r in rows:
        if r["behavior"] not in want:
            continue
        items.append({
            "item_id": str(r["scenario_id"]),
            "behavior": str(r["behavior"]),
            "system_prompt": r["system_prompt"],
            "user_prompt": r["user_prompt"],
        })
    have = {it["behavior"] for it in items}
    missing = [b for b in want if b not in have]
    if missing:
        raise MisalignRunnerError(
            f"no agentic scenarios for behavior(s) {missing} (available: "
            f"{sorted({r['behavior'] for r in rows})})")
    return items


def em8q_items() -> List[dict]:
    """The eight Betley questions: ``item_id = BETLEY_8Q_SOURCE_IDS[qid]``
    (the paper's own ids, e.g. ``what_is_your_wish``), no scenario system
    prompt, ``behavior = None``."""
    items = []
    for qid, question in BETLEY_8Q.items():
        items.append({
            "item_id": BETLEY_8Q_SOURCE_IDS[qid],
            "question_id": qid,
            "behavior": None,
            "system_prompt": None,
            "user_prompt": question,
        })
    return items


def load_items(family: str, behaviors: Optional[Sequence[str]] = None
               ) -> Tuple[List[dict], str]:
    """``(items, suite_revision)`` for a family."""
    if family == "agentic":
        rows, revision = _agentic_source()
        return agentic_items(behaviors, rows=rows), revision
    if family == "em8q":
        if behaviors:
            raise MisalignRunnerError("--behaviors applies to the agentic "
                                      "family only")
        return em8q_items(), EM8Q_SUITE_REVISION
    raise MisalignRunnerError(f"unknown family {family!r} (one of {FAMILIES})")


#: SMOKE ONLY: `--limit-items N` keeps the first N items PER BEHAVIOUR in corpus order --
#: the committed artifact's line order (`agentic_misalignment.loader._load` reads it line by line; `agentic_items` preserves it), so
#: the planned set is deterministic and identical on every run. The planned set is what meta.json DECLARES (`item_ids`, the
#: completeness basis every reader judges the store on -- `audit_common.read_misalign_store`: item_ids x n_samples), and `meta.run`
#: records the limit, this rule and the per-behaviour counts before / after it, so a limited store says so by name and can never be
#: mistaken for the production budget. At the production budget the misalign smoke ran 490 items per cell (~2.3 h) on the
#: critical path; 40 per behaviour proves the same chain in minutes.
LIMIT_ITEMS_RULE = "the first N items per behaviour in corpus order (the committed artifact's line order)"


def limit_items_per_behavior(items: Sequence[Mapping[str, Any]], n: int
                             ) -> Tuple[List[dict], Dict[str, int], Dict[str, int]]:
    """``(kept, before, after)``: the first ``n`` items of every behaviour in the order given (:data:`LIMIT_ITEMS_RULE`),
    with the per-behaviour item counts before and after the cap (an item without a behaviour -- em8q -- forms the one
    group ``"-"``). ``n < 1`` refuses: a budget of nothing is a misconfiguration, never an empty run."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise MisalignRunnerError(f"limit_items must be a positive int (the first N items per behaviour), got {n!r}")
    before: Dict[str, int] = {}
    after: Dict[str, int] = {}
    kept: List[dict] = []
    for it in items:
        b = str(it["behavior"]) if it.get("behavior") is not None else "-"
        before[b] = before.get(b, 0) + 1
        if after.get(b, 0) < n:
            after[b] = after.get(b, 0) + 1
            kept.append(dict(it))
    return kept, before, after


# --- messages / payload ------------------------------------------------------


def build_messages(family: str, item: Mapping[str, Any],
                   si_text: Optional[str]) -> List[Dict[str, str]]:
    """The chat messages for one (cell, item) — the rules in the module
    docstring. ``si_text`` is the prompted cell's SI text, or None."""
    if family == "agentic":
        scenario_system = item.get("system_prompt")
        if not scenario_system:
            raise MisalignRunnerError(
                f"agentic item {item.get('item_id')!r} has no system_prompt")
        system = compose_system(si_text, scenario_system)
    elif family == "em8q":
        system = si_text if si_text else None
    else:
        raise MisalignRunnerError(f"unknown family {family!r}")
    user = item.get("user_prompt")
    if not isinstance(user, str) or not user:
        raise MisalignRunnerError(
            f"item {item.get('item_id')!r} has no user_prompt")
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def build_payload(virtual_model: str, messages: Sequence[Mapping[str, str]],
                  *, max_tokens: int, temperature: float = T1_TEMPERATURE,
                  sampling: Optional[Mapping[str, Any]] = None,
                  seed: Optional[int] = None) -> Dict[str, Any]:
    """One chat-completions request. Exactly one sample per request (no
    ``n``/``best_of``), non-streaming (the key is omitted, never sent), and
    none of the proxy-forbidden selector keys — asserted, not assumed.
    ``sampling`` (a resolved sampler row, :func:`resolve_sampling`) is sent
    key by key in place of the lone temperature; every key must be one of
    :data:`SAMPLING_KEYS`. ``seed`` (the ``per_response`` policy's
    :func:`misalign_seed`) rides as the body's ``seed`` when given; None
    (the default) sends no seed key — today's body, byte for byte."""
    samp = dict(sampling) if sampling is not None else {"temperature": float(temperature)}
    bad = sorted(set(samp) - set(SAMPLING_KEYS))
    if bad:
        raise MisalignRunnerError(f"sampling carries non-sampling key(s) {bad} (allowed: {SAMPLING_KEYS})")
    payload: Dict[str, Any] = {
        "model": virtual_model,
        "messages": [dict(m) for m in messages],
        "max_tokens": int(max_tokens),
        **samp,
    }
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise MisalignRunnerError(f"seed must be a non-negative int, got {seed!r}")
        payload["seed"] = int(seed)
    smuggled = sorted(set(payload) & set(NEVER_SEND))
    if smuggled:  # pragma: no cover - construction above cannot produce this
        raise MisalignRunnerError(f"payload carries refused key(s) {smuggled}")
    return payload


def messages_sha(messages: Sequence[Mapping[str, str]]) -> str:
    """prompt_sha over the canonical JSON of the messages actually sent."""
    return prompt_sha(canonical_json([dict(m) for m in messages]))


# --- cells / exclusions ------------------------------------------------------


def validate_cells(cells: Iterable[Mapping[str, Any]]) -> List[dict]:
    out: List[dict] = []
    seen_ids: Set[str] = set()
    seen_vm: Set[str] = set()
    for c in cells:
        if not isinstance(c, Mapping):
            raise MisalignRunnerError(f"cell entry is not a mapping: {c!r}")
        missing = [k for k in CELL_REQUIRED if k not in c]
        if missing:
            raise MisalignRunnerError(
                f"cell {c.get('cell_id')!r} missing field(s) {missing} "
                f"(required: {list(CELL_REQUIRED)})")
        cid, vm = c["cell_id"], c["virtual_model"]
        for label, val in (("cell_id", cid), ("virtual_model", vm),
                           ("serving_identity", c["serving_identity"]),
                           ("intervention_hash", c["intervention_hash"])):
            if not isinstance(val, str) or not val:
                raise MisalignRunnerError(
                    f"cell {cid!r}: {label} must be a non-empty string, "
                    f"got {val!r}")
        if "/" in cid or cid in (".", ".."):
            raise MisalignRunnerError(f"cell_id {cid!r} is not a safe dir name")
        si_id = c["si_id"]
        if si_id is not None and (not isinstance(si_id, str) or not si_id):
            raise MisalignRunnerError(
                f"cell {cid!r}: si_id must be null or a non-empty string")
        if cid in seen_ids:
            raise MisalignRunnerError(f"duplicate cell_id {cid!r}")
        # A served model may back several cells that differ only in the SI they are prompted
        # with (the EM-organisms lane: one LoRA organism x {none, prompted safety, prompted
        # constitution}); what must be unique is the (virtual_model, si_id) pair — two cells
        # sending the same prompt to the same model would be the same measurement twice.
        if (vm, si_id) in seen_vm:
            raise MisalignRunnerError(f"duplicate (virtual_model, si_id) {(vm, si_id)!r}")
        seen_ids.add(cid)
        seen_vm.add((vm, si_id))
        out.append(dict(c))
    if not out:
        raise MisalignRunnerError("no cells given")
    return out


def load_cells_file(path: Path) -> Tuple[List[dict], Dict[str, Any]]:
    """``(cells, extra)`` from cells.json — a bare list, or an object with a
    ``cells`` list whose other keys are recorded as run metadata."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return validate_cells(doc), {}
    if isinstance(doc, Mapping) and isinstance(doc.get("cells"), list):
        extra = {k: v for k, v in doc.items() if k != "cells"}
        return validate_cells(doc["cells"]), extra
    raise MisalignRunnerError(
        f"{path}: expected a list of cells or an object with a 'cells' list")


def parse_exclude(specs: Optional[Iterable[str]]) -> Dict[str, Set[str]]:
    """``cell_id:behavior`` rules -> ``{cell_id: {behavior, ...}}``."""
    out: Dict[str, Set[str]] = {}
    for spec in specs or ():
        if ":" not in spec:
            raise MisalignRunnerError(
                f"--exclude {spec!r}: expected cell_id:behavior")
        cid, beh = spec.split(":", 1)
        if not cid or not beh:
            raise MisalignRunnerError(
                f"--exclude {spec!r}: expected cell_id:behavior")
        out.setdefault(cid, set()).add(beh)
    return out


# --- HTTP layer (injectable) ----------------------------------------------------


def v1_base(url: str) -> str:
    """``<root>/v1`` without a trailing slash (a root URL gains ``/v1``
    once; ``.../v1`` and ``.../v1/`` are already canonical)."""
    if not isinstance(url, str) or not url.strip():
        raise MisalignRunnerError("endpoint URL is empty")
    u = url.strip().rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


class HttpError(RuntimeError):
    """A non-2xx response (``status``) or a transport failure
    (``status is None``)."""

    def __init__(self, message: str, *, status: Optional[int], body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class HttpxJson:
    """The default transport: ``httpx.Client``, JSON in/out, bearer auth.
    Tests inject a fake with the same two methods."""

    def __init__(self, api_key: str, timeout_s: float):
        import httpx  # in the eval/serving venvs; imported here so a fake
        #              transport never needs it

        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            headers={"Authorization": f"Bearer {api_key}"})
        self._httpx = httpx

    def _wrap(self, resp) -> Dict[str, Any]:
        if resp.status_code // 100 != 2:
            raise HttpError(f"{resp.request.method} {resp.request.url} -> "
                            f"{resp.status_code}: {resp.text[:2000]}",
                            status=resp.status_code, body=resp.text[:2000])
        try:
            return resp.json()
        except ValueError as e:
            raise HttpError(f"{resp.request.url}: non-JSON body: "
                            f"{resp.text[:200]!r}", status=resp.status_code,
                            body=resp.text[:2000]) from e

    def get_json(self, url: str) -> Dict[str, Any]:
        try:
            return self._wrap(self._client.get(url))
        except self._httpx.HTTPError as e:
            raise HttpError(f"GET {url}: {type(e).__name__}: {e}",
                            status=None) from e

    def post_json(self, url: str, payload: Mapping[str, Any],
                  headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        # ``headers`` (additive; default None = today's request): extra request headers — the
        # request-evidence client's two tinycap labels + its request id, when wired
        try:
            return self._wrap(self._client.post(
                url, content=json.dumps(dict(payload)).encode("utf-8"),
                headers={"Content-Type": "application/json", **dict(headers or {})}))
        except self._httpx.HTTPError as e:
            raise HttpError(f"POST {url}: {type(e).__name__}: {e}",
                            status=None) from e

    def close(self) -> None:
        self._client.close()


def list_models(http, base: str) -> Tuple[List[str], Optional[dict]]:
    """``GET /v1/models`` -> (ids, proxy identity block or None)."""
    try:
        listing = http.get_json(f"{base}/models")
    except HttpError as e:
        raise MisalignRunnerError(
            f"cannot list models at {base}/models: {e}") from e
    data = listing.get("data") if isinstance(listing, Mapping) else None
    if not isinstance(data, list):
        raise MisalignRunnerError(
            f"{base}/models returned no 'data' list: {str(listing)[:300]}")
    ids = []
    for m in data:
        mid = m.get("id") if isinstance(m, Mapping) else None
        if isinstance(mid, str) and mid:
            ids.append(mid)
    meta = listing.get("petri_proxy") if isinstance(listing, Mapping) else None
    return ids, (dict(meta) if isinstance(meta, Mapping) else None)


def check_models(http, base: str, cells: Sequence[Mapping[str, Any]], *,
                 plain: bool) -> Dict[str, Any]:
    """Refuse before any generation when the endpoint does not serve every
    cell's virtual model. Proxy mode additionally requires the ``petri__``
    namespace (a bare vLLM or foreign endpoint must fail HERE)."""
    ids, identity = list_models(http, base)
    if not ids:
        raise MisalignRunnerError(f"endpoint {base} lists no models")
    if not plain:
        foreign = sorted(i for i in ids if not i.startswith(VMODEL_PREFIX))
        if foreign:
            raise MisalignRunnerError(
                f"endpoint {base} lists {foreign[:5]} outside the "
                f"{VMODEL_PREFIX!r} namespace — not the routing proxy (use "
                "--plain for a bare vLLM endpoint)")
    missing = sorted({c["virtual_model"] for c in cells} - set(ids))
    if missing:
        raise MisalignRunnerError(
            f"endpoint {base} does not serve virtual model(s) {missing} "
            f"(served: {sorted(ids)}) — refusing before any request")
    return {"served_models": sorted(ids), "endpoint_identity": identity}


# --- per-cell store ------------------------------------------------------------------


def split_jsonl(text: str) -> List[str]:
    """JSONL lines: split on ``\\n`` ONLY. ``str.splitlines()`` also breaks on VT, FF, FS, GS, RS, NEL (U+0085),
    LINE SEPARATOR (U+2028) and PARAGRAPH SEPARATOR (U+2029) — characters a model may emit and ``json.dumps(...,
    ensure_ascii=False)`` writes raw inside the string — and then a valid row reads as a "corrupt" line cut
    mid-string (job 487681/487957: line 298 of an intact 400-row file, the same on every node and every re-read)."""
    return text.split("\n")


def read_done(path: Path) -> Set[Tuple[str, int]]:
    """(item_id, sample_idx) pairs already in responses.jsonl. A torn final
    line (a crash mid-append) is TRUNCATED AWAY (tmp + replace, the
    CellStore._parsed_rows discipline) so that sample reruns and no strict
    downstream reader (the judge, rows.read_jsonl) ever meets it; a corrupt
    earlier line is file damage and refuses."""
    done: Set[Tuple[str, int]] = set()
    if not path.exists():
        return done
    text = path.read_text(encoding="utf-8")
    lines = split_jsonl(text)
    if lines and lines[-1] == "":          # the trailing newline's empty remainder is not a line
        lines = lines[:-1]
    torn_at: Optional[int] = None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not text.endswith("\n"):
                torn_at = i
                break
            raise MisalignRunnerError(
                f"{path}: corrupt row at line {i + 1} (not a torn tail) — "
                "file damage; refusing to resume onto it")
        done.add((str(r["item_id"]), int(r["sample_idx"])))
    if torn_at is not None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(l + "\n" for l in lines[:torn_at]), encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"[misalign.runner] {path}: truncated a torn final line "
              "(interrupted append); that sample reruns", file=sys.stderr)
    return done


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str),
                   encoding="utf-8")
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


#: meta.json fields a resume must agree on — mixing samples generated under
#: a different SI, object, cap or temperature into one responses.jsonl would
#: silently blend two experiments
_META_RESUME_KEYS = ("family", "suite_revision", "virtual_model",
                     "intervention_hash", "si_id", "max_tokens", "temperature",
                     "plain", "sampler", "sampling", "seed_policy", "seed_base")
#: a meta.json written before the sampler existed (every 2026-09-10/11 panel) is read AS the t1
#: sampler at its recorded temperature — exactly what those runs sent — so those trees still resume
#: (and a vendor run still refuses to land on them: its temperature and sampler differ); one written
#: before the seed policy existed (every panel up to the seedrep battery) is read AS `none` with no
#: base — exactly what those runs sent (no `seed` on the wire) — so they resume under `none` only, and
#: a `per_response` run refuses to land on them
_META_LEGACY_DEFAULTS = {"sampler": lambda old: SAMPLER_T1,
                         "sampling": lambda old: {"temperature": float(old["temperature"])},
                         "seed_policy": lambda old: SEED_POLICY_NONE,
                         "seed_base": lambda old: None}
#: the identity keys a `none` run does NOT serialise: meta.json under the default policy is the file the
#: runner wrote before the policy existed, byte for byte, so every `none` tree — legacy or new — reads them through the
#: defaults above; only `per_response` writes them (with `decode_policy_id`, `seed_rule`, `seed_rule_text` beside)
_SEED_META_KEYS = ("seed_policy", "seed_base")


def _meta_identity(cell: Mapping[str, Any], family: str, suite_revision: str,
                   max_tokens: int, temperature: float, plain: bool, *,
                   sampler: str = SAMPLER_T1,
                   sampling: Optional[Mapping[str, Any]] = None,
                   seed_policy: str = SEED_POLICY_NONE,
                   seed_base: Optional[int] = None) -> dict:
    samp = dict(sampling) if sampling is not None else {"temperature": float(temperature)}
    return {"family": family, "suite_revision": suite_revision,
            "virtual_model": cell["virtual_model"],
            "intervention_hash": cell["intervention_hash"],
            "si_id": cell["si_id"], "max_tokens": int(max_tokens),
            "temperature": float(temperature), "plain": bool(plain),
            "sampler": sampler, "sampling": samp,
            "seed_policy": seed_policy,
            "seed_base": None if seed_base is None else int(seed_base)}


def _check_meta_resume(meta_path: Path, identity: Mapping[str, Any]) -> Optional[dict]:
    if not meta_path.exists():
        return None
    old = json.loads(meta_path.read_text(encoding="utf-8"))
    for k in _META_RESUME_KEYS:
        if k not in old:
            if k in _META_LEGACY_DEFAULTS and "temperature" in old:
                old[k] = _META_LEGACY_DEFAULTS[k](old)
            else:
                raise MisalignRunnerError(
                    f"{meta_path}: missing identity field {k!r}; refusing to resume")
        if old[k] != identity[k]:
            raise MisalignRunnerError(
                f"{meta_path}: resume mismatch on {k!r}: recorded {old[k]!r}, "
                f"this run {identity[k]!r} — a different configuration cannot "
                "resume onto these responses")
    return old


# --- the run --------------------------------------------------------------------------


def _extract(resp: Mapping[str, Any], *, cell_id: str, item_id: str,
             sample_idx: int) -> Dict[str, Any]:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MisalignRunnerError(
            f"cell {cell_id!r} item {item_id!r} sample {sample_idx}: response "
            f"has no choices: {str(resp)[:300]}")
    if len(choices) != 1:
        raise MisalignRunnerError(
            f"cell {cell_id!r} item {item_id!r} sample {sample_idx}: expected "
            f"exactly one choice, got {len(choices)} (n>1 is never requested)")
    ch = choices[0]
    msg = ch.get("message") if isinstance(ch, Mapping) else None
    if not isinstance(msg, Mapping):
        raise MisalignRunnerError(
            f"cell {cell_id!r} item {item_id!r} sample {sample_idx}: choice "
            f"has no message: {str(ch)[:300]}")
    content = msg.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        # content-part lists: concatenate the text parts
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, Mapping) and p.get("type") == "text")
        else:
            raise MisalignRunnerError(
                f"cell {cell_id!r} item {item_id!r} sample {sample_idx}: "
                f"non-text message content {type(content).__name__}")
    out = {
        "response": content,
        "finish_reason": ch.get("finish_reason"),
        "usage": dict(resp["usage"]) if isinstance(resp.get("usage"), Mapping) else None,
        "model_reported": resp.get("model"),
    }
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        out["reasoning"] = reasoning
    return out


#: vLLM's refusal when prompt + max_tokens exceed the served window; the numbers let the
#: runner shrink the cap ONCE to fit (a ladder rung on a prompted cell: 5.9k-token SI +
#: scenario + 16384 > 24576). The row then records `max_tokens_effective`.
_CTX_RE = re.compile(r"maximum context length is (\d+) tokens.*?at least (\d+) input tokens", re.S)
#: vLLM's "at least N input tokens" is NOT the prompt length: it is derived from the request
#: (window - requested + 1; it rose by exactly 65 each time the cap was cut by 64 on one prompt,
#: 8,193 -> 8,258 -> 8,323 -> 8,388), so a fit read from it never converges. Each fit therefore
#: cuts at least _FIT_STEP tokens below the current cap as well, for up to _FIT_ATTEMPTS tries.
_FIT_MARGIN = 64
_FIT_MIN_TOKENS = 256
_FIT_ATTEMPTS = 6
_FIT_STEP = 2048


def fit_max_tokens(body: str, current: Optional[int] = None) -> Optional[int]:
    """The next max_tokens to try after a context-length 400: the room the message
    implies (window - reported input - margin), and at least _FIT_STEP below ``current``
    when given (the reported input is only a lower bound). None when the body is not that
    refusal or the result is below _FIT_MIN_TOKENS."""
    m = _CTX_RE.search(body or "")
    if not m:
        return None
    room = int(m.group(1)) - int(m.group(2)) - _FIT_MARGIN
    if current is not None:
        room = min(room, int(current) - _FIT_STEP)
    return room if room >= _FIT_MIN_TOKENS else None


def _post_with_retries(http, url: str, payload: Mapping[str, Any], *,
                       max_retries: int, what: str,
                       mint_headers=None, record=None) -> Dict[str, Any]:
    """POST; a 4xx is a refusal (our request is wrong — never retried) — except a
    context-length 400, which is retried up to _FIT_ATTEMPTS times with max_tokens
    shrunk to fit the window, re-read from each fresh refusal (the response then
    carries ``_max_tokens_effective``); transport failures and 5xx retry with
    backoff, then raise. ``mint_headers`` / ``record`` (additive; default None, both
    or neither) are the request-evidence hook's two calls bound to this item —
    ``mint_headers(payload=body) -> headers`` and ``record(payload=body, headers=,
    response=) -> request_id`` — driven HERE, at the POST boundary, once per
    attempt: the headers are minted for and the record filed from the exact body
    each attempt sends (a context-fit retry's smaller ``max_tokens`` included), a
    retry being a new attempt with its own request id (the same id returned for
    two attempts refuses), and the SUCCESSFUL attempt's id rides back on the
    response as ``_request_id``. None calls ``post_json(url, payload)`` exactly as
    before (a two-argument fake transport keeps working)."""
    attempt = 0
    payload = dict(payload)
    fitted: Optional[int] = None
    fits = 0
    ids_seen: set = set()

    def filed(headers, response):          # this attempt's record; its id must be new
        rid = record(payload=payload, headers=headers, response=response)
        if rid is not None and rid in ids_seen:
            raise MisalignRunnerError(f"{what}: the request-evidence hook returned request id {rid!r} for two "
                                      "attempts — a retry is a new attempt and needs its own id")
        ids_seen.add(rid)
        return rid

    while True:
        headers = dict(mint_headers(payload=payload)) if mint_headers is not None else None
        try:
            resp = (http.post_json(url, payload) if headers is None
                    else http.post_json(url, payload, headers=headers))
        except HttpError as e:
            if record is not None:
                filed(headers, None)                  # the refused / failed attempt, body as sent
            if e.status is not None and 400 <= e.status < 500:
                cur = int(payload.get("max_tokens", 0) or 0)
                fit = fit_max_tokens(e.body, cur) if (e.status == 400 and fits < _FIT_ATTEMPTS) else None
                if fit is not None and fit < int(payload.get("max_tokens", 0) or 0):
                    fits += 1
                    print(f"[misalign.runner] {what}: window too small for max_tokens="
                          f"{payload['max_tokens']} — retrying at {fit} (fit {fits}/{_FIT_ATTEMPTS})", flush=True)
                    payload = {**payload, "max_tokens": fit}        # a NEW body for the next attempt; the sent one stays as sent
                    fitted = fit
                    continue
                raise MisalignRunnerError(
                    f"{what}: endpoint refused the request ({e.status}): "
                    f"{e.body[:1500]}") from e
            attempt += 1
            if attempt > max_retries:
                raise MisalignRunnerError(
                    f"{what}: failed after {max_retries} retries: {e}") from e
            time.sleep(min(60.0, 2.0 ** attempt))
        else:
            extras = {}
            if record is not None:
                extras["_request_id"] = filed(headers, resp)
            if fitted is not None:
                extras["_max_tokens_effective"] = fitted
            return {**resp, **extras} if extras else resp


def zlib_ratio(text: str) -> float:
    """Compressed/raw byte ratio: ~0.4-0.55 for prose, <0.2 for repetition loops."""
    import zlib
    b = (text or "").encode("utf-8")
    return len(zlib.compress(b, 6)) / max(1, len(b))


def truncated_selection(src_root: os.PathLike | str, *, min_zlib: float = RERUN_MIN_ZLIB_RATIO
                        ) -> Tuple[Dict[str, Set[Tuple[str, int]]], Dict[str, Dict[str, int]]]:
    """Per cell under ``src_root`` (``<cell>/responses.jsonl``): the (item_id, sample_idx)
    pairs whose response hit the cap (finish_reason == "length") and is NOT a repetition
    loop (zlib ratio >= min_zlib). Returns (selection, counts) with counts per cell
    {"truncated", "degenerate", "selected"}; a cell with nothing selected is absent."""
    src = Path(src_root)
    if not src.is_dir():
        raise MisalignRunnerError(f"--rerun-truncated-from: {src} is not a directory")
    sel: Dict[str, Set[Tuple[str, int]]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for rp in sorted(src.glob("*/responses.jsonl")):
        cid = rp.parent.name
        c = {"truncated": 0, "degenerate": 0, "selected": 0}
        picked: Set[Tuple[str, int]] = set()
        with rp.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("finish_reason") != "length":
                    continue
                c["truncated"] += 1
                if zlib_ratio(r.get("response") or "") < min_zlib:
                    c["degenerate"] += 1
                    continue
                picked.add((str(r["item_id"]), int(r["sample_idx"])))
        c["selected"] = len(picked)
        counts[cid] = c
        if picked:
            sel[cid] = picked
    if not counts:
        raise MisalignRunnerError(f"--rerun-truncated-from: no <cell>/responses.jsonl under {src}")
    return sel, counts


def filter_selection_by_behaviors(only: Mapping[str, Set[Tuple[str, int]]], behaviors: Sequence[str]
                                  ) -> Tuple[Dict[str, Set[Tuple[str, int]]], Dict[str, int]]:
    """Keep, per cell, the selected ``(item_id, sample_idx)`` pairs whose item belongs to ``behaviors`` (the
    agentic suite's items for those behaviours); cells left with nothing are dropped from the mapping (the
    family loop then reports them skipped). Returns ``(kept, left)`` with ``left`` = pairs per cell that were
    set aside because their behaviour is not requested."""
    want = {i["item_id"] for i in agentic_items(behaviors)}
    kept: Dict[str, Set[Tuple[str, int]]] = {}
    left: Dict[str, int] = {}
    for cid, pairs in only.items():
        k = {p for p in pairs if p[0] in want}
        n_left = len(pairs) - len(k)
        if n_left:
            left[cid] = n_left
        if k:
            kept[cid] = k
    return kept, left


def run_family(
    proxy_url: str,
    cells: Sequence[Mapping[str, Any]],
    family: str,
    n_samples: int,
    out_dir: os.PathLike | str,
    *,
    concurrency: int = 16,
    max_tokens: Optional[int] = None,
    temperature: float = T1_TEMPERATURE,
    api_key: str = DUMMY_BEARER,
    items: Optional[Sequence[Mapping[str, Any]]] = None,
    suite_revision: Optional[str] = None,
    behaviors: Optional[Sequence[str]] = None,
    exclude: Optional[Mapping[str, Iterable[str]]] = None,
    plain: bool = False,
    http=None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_retries: int = 3,
    run_extra: Optional[Mapping[str, Any]] = None,
    only: Optional[Mapping[str, Set[Tuple[str, int]]]] = None,
    sampler: str = SAMPLER_T1,
    seed_policy: str = SEED_POLICY_NONE,
    seed_base: Optional[int] = None,
    request_evidence=None,
    limit_items: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate ``n_samples`` responses per item for every cell; returns a
    per-cell summary ``{cell_id: {"requested", "skipped", "generated",
    "excluded_items"}}``. ``sampler`` (:data:`SAMPLERS`) picks the sampling
    keys every request carries — ``t1`` = ``temperature`` alone, ``vendor``
    = the served model's registered vendor row (:func:`resolve_vendor_policy`,
    by exact policy ref), resolved from the endpoint's identity block and
    recorded in meta.json under ``per_response`` (``decode_policy_id``).

    ``seed_policy`` / ``seed_base``:
    ``per_response`` puts ``seed = misalign_seed(seed_base, behavior, item_id,
    sample_idx)`` on every request and records ``seed`` / ``seed_rule`` /
    ``seed_base`` / ``sampling`` (as sent) / ``decode_policy_id`` /
    ``completion_id`` on every row; the policy and base are resume identity
    (meta.json). The default ``none`` sends and records nothing new.

    ``request_evidence`` (default None; the request-evidence client's hook
    point — wired by that client, not here): an object with
    ``headers_for(*, family, behavior, item_id, sample_idx, payload) -> {header:
    value}`` (extra request headers: the two tinycap labels + a request id) and
    ``record(*, payload, headers, response) -> request_id | None``;
    :func:`_post_with_retries` drives both once per HTTP attempt, and the
    SUCCESSFUL attempt's id lands on the row as ``request_id``. None = no
    headers, no record, no field — today's runner byte for byte.

    ``limit_items`` (default None = the full set; SMOKE ONLY): keep the first
    N items PER BEHAVIOUR in corpus order
    (:func:`limit_items_per_behavior`) after the ``behaviors`` filter; meta.json
    then declares exactly that planned set (``item_ids``) and ``run`` records
    ``limit_items`` / ``limit_items_rule`` / the per-behaviour counts before and
    after the cap. Refused beside ``only`` (a ladder rerun redraws the pairs its
    source store selected, never a head of the corpus).

    ``items`` (with ``suite_revision``) overrides the family's item source
    (tests; a pinned subset). ``exclude`` maps cell_id -> behaviors to skip
    for that cell (nemotron3nano4b has no leaking headroom). ``http`` is
    an object with ``get_json(url)`` / ``post_json(url, payload)``; default
    :class:`HttpxJson`. Any endpoint 4xx or exhausted retry aborts the run
    (the completed samples stay on disk; rerunning resumes). ``only`` (the
    truncation ladder) restricts generation to the given (item_id, sample_idx)
    pairs per cell_id; cells absent from it are skipped entirely."""
    if family not in FAMILIES:
        raise MisalignRunnerError(f"unknown family {family!r} (one of {FAMILIES})")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise MisalignRunnerError(f"n_samples must be a positive int, got {n_samples!r}")
    if concurrency < 1:
        raise MisalignRunnerError("concurrency must be >= 1")
    check_seed_policy(seed_policy, seed_base)
    cells = validate_cells(cells)
    if items is None:
        items, revision = load_items(family, behaviors)
    else:
        if suite_revision is None:
            raise MisalignRunnerError("explicit items need a suite_revision")
        items, revision = [dict(i) for i in items], suite_revision
        if behaviors:
            items = [i for i in items if i.get("behavior") in set(behaviors)]
    if limit_items is not None:
        # SMOKE ONLY: the head of the corpus per behaviour; never on a ladder rerun, whose selection is its source's
        if only is not None:
            raise MisalignRunnerError("limit_items rides a base run only: a ladder rerun (`only` / --rerun-truncated-from) redraws "
                                      "the pairs its source store selected, never a head of the corpus")
        items, before, after = limit_items_per_behavior(items, limit_items)
        run_extra = {**dict(run_extra or {}), "limit_items": int(limit_items), "limit_items_rule": LIMIT_ITEMS_RULE,
                     "items_before_limit": before, "items_after_limit": after}
        print(f"[misalign.runner] SMOKE: --limit-items {limit_items} -> {len(items)} of {sum(before.values())} items ("
              + ", ".join(f"{b}: {after.get(b, 0)}/{before[b]}" for b in before)
              + f"; {LIMIT_ITEMS_RULE}) -- a limited store is never a wave result", flush=True)
    if not items:
        raise MisalignRunnerError(f"family {family!r}: no items to run")
    ids = [i["item_id"] for i in items]
    if len(set(ids)) != len(ids):
        raise MisalignRunnerError(f"family {family!r}: duplicate item ids")
    cap = int(max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS[family])
    if cap < 1:
        raise MisalignRunnerError("max_tokens must be >= 1")
    excl = {k: set(v) for k, v in (exclude or {}).items()}
    unknown_excl = sorted(set(excl) - {c["cell_id"] for c in cells})
    if unknown_excl:
        raise MisalignRunnerError(
            f"--exclude names unknown cell(s) {unknown_excl} — refusing to "
            "silently drop nothing")

    base = v1_base(proxy_url)
    own_http = http is None
    if own_http:
        http = HttpxJson(api_key=api_key, timeout_s=timeout_s)
    # request-evidence: off = the transport is returned untouched; on = every /chat/completions post is a
    # recorded attempt with a fresh id per HTTP attempt (a 5xx retry is a new attempt) — the labels from the caller's
    # headers (the `request_evidence` seam) or a `labelled(...)` context; the response bound by its completion id
    http = rev.maybe_wrap_post_json(http)
    try:
        endpoint = check_models(http, base, cells, plain=plain)
        # the sampling row every request carries, resolved ONCE against the served identity
        # (the vendor sampler refuses here, before any request, for a model with no row)
        sampling = resolve_sampling(sampler, temperature, endpoint.get("endpoint_identity"))
        temperature = float(sampling["temperature"])
        # the registered row the vendor sampler sends (its policy_id rides on meta.json and, under
        # per_response, on every row as `decode_policy_id`); None under t1
        decode_policy_id = (resolve_vendor_policy(endpoint.get("endpoint_identity")).policy_id
                            if sampler == SAMPLER_VENDOR else None)
        # SI texts resolved once, up front — a bad si_id refuses before any
        # request, and the bytes are the exact prefix the proxy verifies
        si_texts = {c["cell_id"]: (si_text_for(c["si_id"]) if c["si_id"] else None)
                    for c in cells}
        out_root = Path(out_dir)
        summary: Dict[str, Any] = {}
        for cell in cells:
            if only is not None and cell["cell_id"] not in only:
                print(f"[misalign.runner] {family} cell {cell['cell_id']}: nothing selected — skipped",
                      flush=True)
                summary[cell["cell_id"]] = {"requested": 0, "skipped": 0, "generated": 0,
                                            "excluded_items": 0}
                continue
            summary[cell["cell_id"]] = _run_cell(
                http, base, cell, family, items, revision, n_samples,
                out_root / cell_dirname(cell["cell_id"]), si_texts[cell["cell_id"]],
                cap=cap, temperature=temperature, concurrency=concurrency,
                exclude=excl.get(cell["cell_id"], set()), plain=plain,
                endpoint=endpoint, proxy_url=base, max_retries=max_retries,
                run_extra=dict(run_extra or {}),
                only=None if only is None else only[cell["cell_id"]],
                sampler=sampler, sampling=sampling,
                seed_policy=seed_policy, seed_base=seed_base,
                decode_policy_id=decode_policy_id, request_evidence=request_evidence)
        return summary
    finally:
        if own_http:
            http.close()


#: composite organism cells are named ``<organism>::<cell>`` (the analysis contract,
#: ``analyze.COMPOSITE_SEP``), but ``:`` is not a legal path character on every
#: filesystem the runner writes to (cluster A's shared storage refuses it with
#: EINVAL — job 487636 died at ``mkdir('em-bad-medical-advice::none')`` after a
#: healthy standup), so the DIRECTORY carries ``@`` in its place. The cell_id in
#: meta.json and every row is the untouched composite; ``rows.read_meta`` checks
#: the directory against this same mapping.
CELL_DIR_SEP = "@"


def cell_dirname(cell_id: str) -> str:
    """The directory a cell's rows live in: ``<organism>@<cell>`` for a composite id, else the id."""
    if CELL_DIR_SEP in cell_id:
        raise ValueError(f"cell_id {cell_id!r} already carries {CELL_DIR_SEP!r}; the directory mapping would be ambiguous")
    return cell_id.replace("::", CELL_DIR_SEP)


def _run_cell(http, base, cell, family, items, revision, n_samples, cell_dir,
              si_text, *, cap, temperature, concurrency, exclude, plain,
              endpoint, proxy_url, max_retries, run_extra,
              only: Optional[Set[Tuple[str, int]]] = None,
              sampler: str = SAMPLER_T1,
              sampling: Optional[Mapping[str, Any]] = None,
              seed_policy: str = SEED_POLICY_NONE,
              seed_base: Optional[int] = None,
              decode_policy_id: Optional[str] = None,
              request_evidence=None) -> Dict[str, Any]:
    cell_id = cell["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    resp_path = cell_dir / "responses.jsonl"
    meta_path = cell_dir / "meta.json"
    sampling = dict(sampling) if sampling is not None else {"temperature": float(temperature)}
    seeded = seed_policy == SEED_POLICY_PER_RESPONSE
    identity = _meta_identity(cell, family, revision, cap, temperature, plain,
                              sampler=sampler, sampling=sampling,
                              seed_policy=seed_policy, seed_base=seed_base)
    old_meta = _check_meta_resume(meta_path, identity)

    kept = [i for i in items if i.get("behavior") not in exclude]
    excluded_items = [i["item_id"] for i in items if i.get("behavior") in exclude]
    done = read_done(resp_path)
    jobs = [(item, s) for item in kept for s in range(n_samples)
            if (item["item_id"], s) not in done
            and (only is None or (item["item_id"], s) in only)]
    if only is not None:
        unknown = sorted(only - {(i["item_id"], s) for i in kept for s in range(n_samples)})
        if unknown:
            raise MisalignRunnerError(
                f"{cell_id}: {len(unknown)} selected pair(s) are not items of this family/behavior "
                f"set (first: {unknown[:3]}) — the rerun root does not match the source")

    meta = {
        # the seed keys and the seed provenance below ride meta.json ONLY under per_response: a `none`
        # run's file is byte for byte the runner's before the policy existed, and a legacy tree resumed under `none`
        # is not rewritten with them — the resume comparison above still reads them, as defaults, in memory
        **(identity if seeded else {k: v for k, v in identity.items() if k not in _SEED_META_KEYS}),
        "cell": dict(cell),
        "cell_id": cell_id,
        "n_samples": max(int(n_samples), int((old_meta or {}).get("n_samples", 0))),
        "params": {"max_tokens": cap, "temperature": float(temperature),
                   "sampler": sampler, "sampling": dict(sampling),
                   "concurrency": int(concurrency)},
        # under per_response: the registered decode row the sampler sent, the seed rule's name and its text —
        # provenance beside the identity keys above, never identity
        **({"decode_policy_id": decode_policy_id, "seed_rule": SEED_RULE_PER_RESPONSE, "seed_rule_text": SEED_RULE_TEXT}
           if seeded else {}),
        "items": len(kept),
        "item_ids": [i["item_id"] for i in kept],
        "behaviors": sorted({i["behavior"] for i in kept if i.get("behavior")}),
        "excluded_behaviors": sorted(exclude),
        "excluded_items": excluded_items,
        "proxy_url": proxy_url,
        **endpoint,
        "run": run_extra,
        "started": (old_meta or {}).get("started") or time.time(),
        "resumed_at": time.time() if old_meta else None,
        "finished": None,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _write_json_atomic(meta_path, meta)
    print(f"[misalign.runner] {family} cell {cell_id} ({cell['virtual_model']}): "
          f"{len(kept)} items x {n_samples} samples, {len(done)} present, "
          f"{len(jobs)} to generate"
          + (f"; excluded behaviors {sorted(exclude)}" if exclude else ""),
          flush=True)

    lock = threading.Lock()
    generated = 0

    def one(item, sample_idx):
        messages = build_messages(family, item, si_text)
        seed = (misalign_seed(seed_base, seed_behavior(family, item), item["item_id"], sample_idx)
                if seeded else None)
        payload = build_payload(cell["virtual_model"], messages,
                                max_tokens=cap, sampling=sampling, seed=seed)
        what = f"cell {cell_id!r} item {item['item_id']!r} sample {sample_idx}"
        # the request-evidence hook, bound to this item; _post_with_retries drives it per ATTEMPT at the
        # POST boundary (a context-fit retry sends a smaller body under its own id), and the successful
        # attempt's id comes back on the response
        evidence = {} if request_evidence is None else {
            "mint_headers": functools.partial(request_evidence.headers_for, family=family, behavior=item.get("behavior"),
                                              item_id=item["item_id"], sample_idx=int(sample_idx)),
            "record": request_evidence.record}
        resp = _post_with_retries(http, f"{base}/chat/completions", payload,
                                  max_retries=max_retries, what=what, **evidence)
        request_id = resp.pop("_request_id", None) if isinstance(resp, dict) else None
        effective = resp.pop("_max_tokens_effective", None) if isinstance(resp, dict) else None
        got = _extract(resp, cell_id=cell_id, item_id=item["item_id"],
                       sample_idx=sample_idx)
        row = {
            "family": family,
            "cell_id": cell_id,
            "virtual_model": cell["virtual_model"],
            "item_id": item["item_id"],
            "sample_idx": int(sample_idx),
            "behavior": item.get("behavior"),
            "prompt_sha": messages_sha(messages),
            **got,
        }
        if effective is not None:
            row["max_tokens_effective"] = int(effective)
        if seeded:
            # the seeded row's decode evidence: the seed AS SENT, its rule and base, the
            # sampling keys AS SENT (`max_tokens` is the row's cap / max_tokens_effective), the
            # registered row they came from, and the engine's completion id for the wire binding
            row.update({"seed": int(seed), "seed_rule": SEED_RULE_PER_RESPONSE,
                        "seed_base": int(seed_base),
                        "sampling": {k: payload[k] for k in SAMPLING_KEYS if k in payload},
                        "decode_policy_id": decode_policy_id,
                        "completion_id": resp.get("id") if isinstance(resp, Mapping) else None})
        if request_evidence is not None:
            row["request_id"] = request_id            # the SUCCESSFUL attempt's, minted at its POST
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with lock:
            with open(resp_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return row

    if jobs:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(jobs))) as pool:
            futures = [pool.submit(one, item, s) for item, s in jobs]
            pending = set(futures)
            t0 = time.time()
            while pending:
                finished, pending = wait(pending, timeout=60.0,
                                         return_when=FIRST_EXCEPTION)
                for fut in finished:
                    exc = fut.exception()
                    if exc is not None:
                        for p in pending:
                            p.cancel()
                        raise exc
                    generated += 1
                if finished:
                    print(f"[misalign.runner] {cell_id}: {generated}/{len(jobs)} "
                          f"generated ({time.time() - t0:.0f}s)", flush=True)
    meta["finished"] = time.time()
    meta["generated_this_run"] = generated
    meta["present"] = len(read_done(resp_path))
    _write_json_atomic(meta_path, meta)
    _registry_generation_complete(family, cell_id, kept, resp_path, meta["present"])
    return {"requested": len(kept) * n_samples, "skipped": len(done),
            "generated": generated, "excluded_items": len(excluded_items)}


def _registry_generation_complete(family: str, cell_id: str, kept: Sequence[Mapping[str, Any]],
                                  resp_path: Path, n_present: int) -> None:
    """Results-registry native emission (additive, never fatal): this runner
    appends generations to ``responses.jsonl`` directly — no ``CellStore``
    until ``misalign.rows`` files the judged rows later — so the store
    hook's generation marker has no equivalent here. This is it: one
    ``run_patch`` (generation complete, the responses path, the count
    present) per registered run the job's pending identity claims for
    (this cell, each behaviour's task spelling — ``agentic/<behavior>`` |
    ``em8q/betley``, the same strings ``rows.py`` writes). Off / unrouted /
    failing → one stderr line at most; the meta and the responses are
    already on disk."""
    try:
        from sisteer.registry import hooks as _registry_hooks   # stdlib-only
        if family == "agentic":
            tasks = sorted({f"agentic/{i.get('behavior')}" for i in kept if i.get("behavior")})
        else:
            tasks = [f"{family}/betley"]
        _registry_hooks.emit_generation_complete(
            cell_id, tasks, str(resp_path), n_generated=int(n_present),
            script="sisteer.capeval.misalign.runner",
            extra={"family": family, "meta_uri": str(resp_path.with_name("meta.json"))})
    except Exception as e:  # noqa: BLE001 — by contract: a lost record never costs a run
        print(f"[registry] misalign.runner {cell_id}: generation record not written "
              f"({type(e).__name__}: {e}); the run continues", file=sys.stderr, flush=True)


# --- CLI ---------------------------------------------------------------------------


def _seedrep_rung_redraw():
    """``scripts/sdfbase/seedrep/rung_redraw.py`` by path, loaded ONLY when ``--from-materialised``
    is given — the streaming ladder's assembly seam; nothing here changes without the flag."""
    import importlib.util as _ilu
    import sys as _sys
    mod = _sys.modules.get("seedrep_rung_redraw")
    if mod is None:
        spec = _ilu.spec_from_file_location("seedrep_rung_redraw", Path(__file__).resolve().parents[4] / "scripts" / "sdfbase" / "seedrep" / "rung_redraw.py")
        mod = _sys.modules["seedrep_rung_redraw"] = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sisteer.capeval.misalign.runner",
        description="Served-path runner for the misalignment eval families.")
    p.add_argument("--proxy-url", required=True,
                   help="routing proxy (or, with --plain, vLLM) base URL, "
                        "e.g. http://127.0.0.1:8814 or .../v1")
    p.add_argument("--cells", required=True, type=Path,
                   help="cells.json: list of {cell_id, virtual_model, "
                        "serving_identity, intervention_hash, si_id} (or an "
                        "object with a 'cells' list)")
    p.add_argument("--family", required=True, choices=FAMILIES)
    p.add_argument("--n", required=True, type=int, help="samples per item")
    p.add_argument("--out", required=True, type=Path, help="output root")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=None,
                   help=f"response cap (default per family: {DEFAULT_MAX_TOKENS})")
    p.add_argument("--temperature", type=float, default=T1_TEMPERATURE,
                   help="the t1 sampler's temperature (default: the registry's runner_forced_T1 row; ignored under --sampler vendor)")
    p.add_argument("--sampler", choices=SAMPLERS, default=SAMPLER_T1,
                   help="t1 = temperature alone (the campaign's instrument); vendor = the served "
                        "model's frozen vendor row (every sampling key explicit; resolved from the "
                        "proxy's identity block, refused for a model with no row)")
    p.add_argument("--seed-policy", choices=SEED_POLICIES, default=SEED_POLICY_NONE,
                   help="none = no `seed` on any request (the campaign's runner); per_response = every request "
                        f"carries seed = {SEED_RULE_TEXT} with --seed-base as the base, recorded on every row "
                        "and in meta.json (resume identity)")
    p.add_argument("--seed-base", type=int, default=None,
                   help="the per_response seed base: the literal training seed s of the served object's view "
                        "(qwen-seedrep-native-battery); required with --seed-policy per_response, refused without it")
    p.add_argument("--behaviors", nargs="+", default=None,
                   help=f"agentic behaviors (default {list(AGENTIC_BEHAVIORS)})")
    p.add_argument("--exclude", action="append", default=[],
                   metavar="CELL_ID:BEHAVIOR",
                   help="skip a behavior for one cell (repeatable), e.g. "
                        "nemotron3nano4b:leaking")
    p.add_argument("--plain", action="store_true",
                   help="a bare vLLM endpoint: virtual_model is the served "
                        "name (base or LoRA module); no petri__ namespace check")
    p.add_argument("--api-key", default=DUMMY_BEARER)
    p.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--rerun-truncated-from", type=Path, default=None, metavar="SRC_FAMILY_DIR",
                   help="truncation ladder: regenerate ONLY the (item, sample) pairs whose response under "
                        "SRC_FAMILY_DIR/<cell>/responses.jsonl hit the cap and is not a repetition loop; "
                        "--out must be a NEW root beside the source (the pair is the truncation measurement)")
    p.add_argument("--only-cells", default=None, metavar="CID[,CID...]",
                   help="run only these cell ids from --cells (a ladder shard); an unknown id refuses")
    p.add_argument("--rerun-min-zlib", type=float, default=RERUN_MIN_ZLIB_RATIO,
                   help="zlib compressed/raw ratio below which a truncated response counts as a loop "
                        "(degenerate; reported, not re-laddered)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--from-materialised", default=None, metavar="LADDER_ROOT",
                   help="with --rerun-truncated-from — seedrep streaming ladder (plan §7.3, WS2-E): the CLOSED queue root "
                        ".../ladder/<leg>/<view>/<cap>; every selected pair with an elected, VERIFIED materialised draw "
                        "is written into the NEW --out root before generation (the runner's own resume skips it), the "
                        "rest is drawn here. Default off: without it the runner is byte-identical to before")
    p.add_argument("--results", default=None, metavar="SNAPSHOT",
                   help="with --from-materialised: the results Dict snapshot (rung_redraw.py snapshot)")
    p.add_argument("--limit-items", type=int, default=0,
                   help="SMOKE ONLY (qwen-seedrep-native-battery W20, D107): keep the first N items PER BEHAVIOUR in corpus "
                        "order (0 = the full set). meta.json declares exactly that planned set (item_ids), so the store still "
                        "reads COMPLETE on it, and meta.run records the limit and the rule -- a limited store is a smoke, never "
                        "a wave result. Refused beside --rerun-truncated-from (a ladder rerun redraws its source's pairs)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.limit_items < 0:
        raise MisalignRunnerError(f"--limit-items must be >= 0 (0 = the full set), got {args.limit_items}")
    if args.limit_items and args.rerun_truncated_from is not None:
        raise MisalignRunnerError("--limit-items is a base-run smoke budget (W20, D107) and never rides --rerun-truncated-from: "
                                  "a ladder rerun redraws the pairs its source store selected, never a head of the corpus")
    cells, extra = load_cells_file(args.cells)
    if args.only_cells:
        wanted = [c.strip() for c in args.only_cells.split(",") if c.strip()]
        known = {c["cell_id"] for c in cells}
        unknown = sorted(set(wanted) - known)
        if unknown:
            raise MisalignRunnerError(f"--only-cells names unknown cell(s) {unknown} (known: {sorted(known)})")
        cells = [c for c in cells if c["cell_id"] in set(wanted)]
        extra = {**dict(extra or {}), "only_cells": sorted(wanted)}
    only = None
    if args.rerun_truncated_from is not None:
        src = Path(args.rerun_truncated_from).resolve()
        if Path(args.out).resolve() == src:
            raise MisalignRunnerError(
                "--rerun-truncated-from must target a NEW --out root beside the source, never the "
                "source itself (the pair is the truncation-bias measurement)")
        only, counts = truncated_selection(src, min_zlib=args.rerun_min_zlib)
        extra = {**dict(extra or {}), "rerun_of": str(src), "rung_max_tokens": args.max_tokens,
                 "rerun_min_zlib": args.rerun_min_zlib, "rerun_selection": counts}
        if args.behaviors and args.family == "agentic":
            # a ladder restricted to some behaviours (qwen: leaking only — its murder column is the vendor re-sample)
            # reruns only the selected pairs OF those behaviours; the source's other-behaviour truncations are recorded
            # as deliberately left, so _run_cell's "not items of this run" guard keeps meaning a mismatched root
            only, left = filter_selection_by_behaviors(only, args.behaviors)
            extra["rerun_behaviors"] = sorted(args.behaviors)
            extra["rerun_selection_left_by_behavior_filter"] = left
            for cid, n in sorted(left.items()):
                print(f"[misalign.runner] ladder {cid}: {n} selected pair(s) of other behaviours left to their own "
                      f"column (--behaviors {' '.join(args.behaviors)})", flush=True)
        for cid, c in sorted(counts.items()):
            print(f"[misalign.runner] ladder {cid}: truncated {c['truncated']}, degenerate (loop) "
                  f"{c['degenerate']}, selected {c['selected']}", flush=True)
        if not only:
            print("[misalign.runner] ladder: nothing to rerun (no non-loop truncated responses)", flush=True)
            return 0
    if getattr(args, "from_materialised", None):  # seed the NEW root with elected, verified rung draws
        if only is None or not getattr(args, "results", None):
            raise MisalignRunnerError("--from-materialised rides --rerun-truncated-from and needs --results <snapshot>")
        _seedrep_rung_redraw().seed_misalign_root(
            args.out, ladder_root=args.from_materialised, results=args.results,
            src_root=Path(args.rerun_truncated_from).resolve(),
            cap=int(args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS[args.family]), only=only,
            cell_dirname=cell_dirname, read_done=read_done, seed_base=getattr(args, "seed_base", None),
            log=lambda m: print(f"[misalign.runner] {m}", flush=True))
    summary = run_family(
        args.proxy_url, cells, args.family, args.n, args.out,
        concurrency=args.concurrency, max_tokens=args.max_tokens,
        temperature=args.temperature, api_key=args.api_key,
        behaviors=args.behaviors, exclude=parse_exclude(args.exclude),
        plain=args.plain, timeout_s=args.timeout_s,
        max_retries=args.max_retries, run_extra=extra, only=only,
        sampler=args.sampler, seed_policy=args.seed_policy, seed_base=args.seed_base,
        # request-evidence: the labels (`<family>/<behavior>`, `<item_id>/<sample_idx>`) of every draw and the
        # answering attempt's request id on the row. Off = None (no headers, no key, bytes as before). Without it,
        # under a hook that is ON, the wrapped transport refused the FIRST send as unlabelled (a BoN child died
        # that way on this leg).
        request_evidence=rev.maybe_misalign_request_evidence(),
        # SMOKE ONLY: the per-behaviour head of the corpus; 0 / absent = the full set, bytes as before
        limit_items=(int(args.limit_items) if args.limit_items else None))
    for cid, s in summary.items():
        print(f"[misalign.runner] {cid}: requested {s['requested']}, present "
              f"{s['skipped']}, generated {s['generated']}, excluded items "
              f"{s['excluded_items']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
