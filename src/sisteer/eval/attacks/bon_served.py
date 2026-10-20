"""Best-of-N generation issued CONCURRENTLY against the steered vLLM server.

Why this module exists (measured four times over: on the long-SI prompted /
stacked / cross-stacked conditions BoN alone ran ~5 h per condition and blew
the 6 h array walltime). BoN issues 7,168 generations per condition
(112 goals x 64 samples); the legacy path (:func:`sisteer.eval.attacks.bon.
bon_generate`) drove them through an in-process HF ``model.generate`` — one
prompt at a time, each re-prefilling the whole ~5.9k-token constitution SI
prefix and blocking a decode chunk on its LONGEST member (97.7% of responses
are short refusals). The serving stack that fixes this already exists — a
prefix-caching vLLM server behind :class:`~sisteer.capeval.serving.proxy.
RoutingProxy` — but BoN was never pointed at it. This module does exactly
that, and NOTHING ELSE moves: serving is opt-in and the in-process
``bon_generate`` remains the default engine. (``bon_generate`` itself has since
moved its DEFAULT width from 1 to ``DEFAULT_BON_BATCH_SIZE`` = 64, so the
"one prompt at a time" above describes the legacy width-1 default this module
was measured against — reproducible today only via an explicit
``batch_size=1``.)

THREE independent wins, and this module is honest about which apply where:

- **Continuous batching (the decode-side win, applies to EVERY cell).** The
  server retires a finished sequence and backfills the next the instant a
  slot frees, instead of waiting for a chunk's longest member. On a corpus
  that is overwhelmingly short refusals this is the dominant speedup, and it
  needs the client to issue many requests CONCURRENTLY (bounded fan-out, not
  fixed-width synchronous chunks) so the scheduler always has work to backfill.
- **Concurrent fan-out vs. a serial loop (applies to every cell).** Even with
  no caching, N in-flight requests beat one-at-a-time HTTP round-trips.
- **Prefix caching of the SI prefix (applies to UNHOOKED cells ONLY).** The
  ~5.9k SI prefix is prefilled once and reused across all 7,168 requests —
  BUT only for cells whose salt is CELL-STABLE (the ``none`` / ``prompted``
  baselines and LoRA arms). Selector-carrying (lens) cells — the vector / PSR
  arms and every ``stacked`` cell — get a REQUEST-UNIQUE salt by design
  (:meth:`RoutingProxy.cache_salt_for`), so their prefills never resume from
  cache and prefix caching contributes NOTHING to them. This is not a bug to
  route around: a steered prefill served from cache would skip the forward and
  the steering hooks would silently never fire, and a prefill-broadcast method
  (alphasteer derives its edit from the LAST prompt token) is unsound under a
  reused prefix with a different augmented suffix anyway. So the long-SI
  ``stacked`` / ``cross-stacked`` conditions get their speedup from continuous
  batching + fan-out, NOT from caching; only the ``prompted`` long-SI cell
  also gets the caching win. Recovering the SI-prefix win for lens cells needs
  a separate SI-exclusion design (proxy module docstring) and is out of scope.

THE CORRECTNESS BOUNDARY IS THE SALT, and BoN is the case most likely to get
it wrong: 64 samples per goal share an identical SI prefix, so a mis-scoped
salt either serves one cell's steered KV under another cell's identity (a
mis-defended model labelled defended — passes every content test silently) or
throws away the entire caching win. This module NEVER computes a salt itself:
``cache_salt`` is a FORBIDDEN client key, and the proxy owns the per-cell
policy. We go THROUGH the proxy (never a bare ``base_url``) so the single
tested mechanism assigns the salt, and we ASSERT — before generating — that no
two distinct cells the run will serve can ever share a cached prefix
(:func:`assert_cell_salt_isolation`). We do not ship the fan-out without that
assertion.

THE OTHER SILENT FAILURE MODE preserved: vLLM has exactly one worker-extension
slot, and a launch that does not claim it serves the stock extension and
steers NOTHING. The launch-time gate is strong but has no generation-time
proof, so for every STEERED cell this module runs a steering-bites probe
through the SAME serving path right before the run and refuses on an inert arm
(:func:`assert_steering_bites`, over
:func:`sisteer.capeval.serving.fidelity._check_steering_bites`).

RESUME IS NON-NEGOTIABLE and is NOT weakened for concurrency. Each response
row is appended, flushed and ``fsync``'d under a write lock BEFORE its
(goal, sample, trial) unit counts as covered — the response row IS the durable
terminal record (BoN has no separate progress file; its resume granularity is
per-variant, strictly finer than the per-goal static-family contract). The
write section holds no ``await``, so asyncio cancellation can never tear a row;
only a real SIGKILL can, and it can leave at most ONE torn tail line, which
:func:`~sisteer.eval.attacks.bon._load_covered` already skips — that one unit
regenerates, nothing duplicates. Rows land in completion order, not sample
order; that is fine because :func:`~sisteer.eval.attacks.bon.load_bon_responses`
reassembles by ``sample_idx``.

ENGINE PROVENANCE: every served row carries ``"engine": "vllm_served"``, and a
served run REFUSES to append to a responses file that holds any row without it
(and vice versa). Mixing HF and vLLM rows in one responses file — hence one
figure — is invalid (the engines are not bit-identical; the contrast is
internal to a run), and the refusal makes it impossible to do by accident.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Tuple

from ...capeval import manifest as cm
from ...capeval.serving import fidelity
from ...capeval.serving import served_window as sw
from ...capeval.serving.geometry import geometry_for
from ...capeval.serving.proxy import (FORBIDDEN_CLIENT_KEYS, STEERED_ROUTE_PATHS,
                                      RouteEntry, RoutingProxy)
from ...capeval.tinycap import contracts as tc
from ..jailbreak_artifacts import iter_jsonl_rows
from .. import request_evidence_client as rev  # request-evidence hook (off unless SISTEER_REQUEST_EVIDENCE_DIR)
from .bon import (BON_SEED_RULE, DECODE_SEED_POLICY_BON, DECODE_SEED_POLICY_REPLICATE, DEFAULT_N_MAX, DEFAULT_SIGMA,
                  LEGACY_SEEDLESS_RULE, REPLICATE_ROW_KEYS, _load_covered, _resume_key, augment, bon_decode_seed_v1,
                  bon_seed, check_decode_seed_policy, decode_seed_for, seed_rule_for_policy)

#: The provenance tag stamped on every served row (see module docstring).
SERVED_ENGINE = "vllm_served"

#: THE SERVING ENGINE beside it: ``serving_engine`` in
#: {"lens", "lora_native"} — additive, unhashed, the row-facing field that keeps a lens-era
#: row and a native-era row apart where the served identity cannot (a passthrough leg reads
#: ``proxy_vllm+passthrough+…`` on both engines). Read from the proxy's /v1/models identity
#: block (``engine.kind``), never from the environment; a row WITHOUT it predates the native
#: engine and was lens by construction. Vocabulary by reference to the torch-free
#: ``sisteer.capeval.tinycap.contracts`` — one spelling across every writer.
SERVING_ENGINE_FIELD = tc.SERVED_ENGINE_FIELD
SERVING_ENGINE_LENS = tc.SERVING_ENGINE_LENS
SERVING_ENGINE_LORA_NATIVE = tc.SERVING_ENGINE_LORA_NATIVE
SERVING_ENGINES = tuple(tc.SERVING_ENGINES)


def resolve_serving_engine(serving_engine: Optional[str], proxy_identity: Any, *, what: str) -> str:
    """The engine a served BoN / static defense stamps: an explicitly STATED
    ``serving_engine`` (validated against the vocabulary) wins; else it is
    derived from the proxy's ``/v1/models`` ``petri_proxy`` identity block
    (``served_target.serving_engine_from_identity`` — the ONE rule: a native
    ``engine.kind`` -> ``lora_native``, a torn identity refuses, anything
    else -> ``lens``); with NEITHER — a bare transport handed in by a caller
    that never loaded the identity — the defense is LENS by the era rule
    (every such caller today drives a lens standup: the sweep's static seam,
    the petri_bon_* measurement scripts). A native caller MUST hand over the
    identity (``over_client`` / ``over_proxy`` do) or state the engine; this
    default is never read from the environment."""
    if serving_engine is not None:
        if serving_engine not in SERVING_ENGINES:
            raise BonServedError(
                f"{what}: serving_engine must be one of {list(SERVING_ENGINES)}, got {serving_engine!r}")
        return str(serving_engine)
    if proxy_identity is None:
        return SERVING_ENGINE_LENS
    # lazy: served_target imports this module at import time (a cycle at module level)
    from .served_target import ServedTargetError, serving_engine_from_identity
    try:
        return serving_engine_from_identity(proxy_identity)
    except ServedTargetError as e:
        raise BonServedError(f"{what}: {e}") from e

#: The ONLY request-body sampler keys a served client may send when a run
#: decodes under a vendor preset (the full22 SAMPLED re-run, DESIGN.md): the
#: preset carries exactly the subset the vendor states (a model without top_k
#: has no top_k key), plus the per-item ``seed``. vLLM's chat API accepts each
#: at top level and the proxy forwards bodies wholesale except
#: FORBIDDEN_CLIENT_KEYS. Anything outside this tuple is REFUSED, never
#: forwarded on faith — ``n`` in particular (every draw is its own request).
SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "repetition_penalty")

# THE DECODE-POLICY AUTHORITY (results registry): best-of-N decodes at EACH SERVED MODEL'S VENDOR
# PRESET — the registry's `<model>|bon|off|vendor_preset` row, the same numbers the sampled static lane sends
# (`same_values_as` the model's `|static|off|vendor_preset` row, asserted at seed load), presence_penalty included
# (Qwen's 1.5 rides the request; it can never be a server default). The ruling
# decode-bon-vendor-preset-2026-09-17 REPLACED the grandfathered model-independent temperature 1.0
# (`<model>|bon|off|grandfathered_T1`, ruling decode-bon-T1-grandfathered-2026-09-11): those rows are SUPERSEDED
# in the registry and stay readable by explicit reference, because every existing store was drawn under them —
# a resume of such a store, and the climb of one, asks for them BY NAME (:data:`BON_HARNESS_DEFAULT`). There is
# no model-independent default any more: a caller that names no model and no preset is refused, never served a
# temperature nobody vouched for. Every number is read from the committed export — never a literal here.
from sisteer.registry import decode_policy as _decode_policy  # noqa: E402  (stdlib-only reader)

#: the registry eval the BoN rows key on (evals.yaml `strongreject_bon`; every store spells it `bon`)
BON_EVAL_ID = "strongreject_bon"
#: ``bon_decode_preset`` labels. THE DEFAULT: the served model's vendor preset, resolved from the registry.
BON_VENDOR_PRESET = "vendor_preset"
#: the NON-DEFAULT grandfathered T=1 row, by EXPLICIT opt-in only — the sampler every store drawn before
#: 2026-09-17 was drawn under (and the canonical instrument's sampler since 2026-10-14: live, not superseded;
#: never the default); a resume of one of those stores (and its climb) must name it, so the
#: store is never relabelled and never mixed (`_validate_row`)
BON_HARNESS_DEFAULT = "harness_default"
#: an EXPLICIT sampler block handed in by the caller (the sdfbase-llama8b wave's retired `vendor` face, kept
#: because existing rows carry the label); the caller vouches for the numbers, the registry does not
BON_EXPLICIT_PRESET = "vendor"
BON_REGISTRY_LABELS = (BON_VENDOR_PRESET, BON_HARNESS_DEFAULT)
#: the registry `mode` of the grandfathered rows (their policy_ref tail)
_GRANDFATHERED_MODE = "grandfathered_T1"


@dataclass(frozen=True)
class BonSampler:
    """The sampler ONE best-of-N run draws under: the request-body keys (``decode``, SAMPLER_KEYS only, the
    temperature always among them), the ``bon_decode_preset`` label every row records, and the registry policy
    the numbers came from (``None`` for an explicit block the caller vouched for)."""
    decode: Mapping[str, Any]
    label: str
    policy_id: Optional[str] = None

    def request_params(self) -> dict:
        return {k: self.decode[k] for k in SAMPLER_KEYS if k in self.decode}


def _ordered(params: Mapping[str, Any]) -> dict:
    return {k: params[k] for k in SAMPLER_KEYS if k in params}


def bon_policy(model: Any, label: str = BON_VENDOR_PRESET) -> "_decode_policy.Policy":
    """The registry row best-of-N draws under for ``model`` (a model_id, a legacy key such as ``gemma4e4b``, or
    the served HF repo): :data:`BON_VENDOR_PRESET` -> the LIVE default row for eval ``strongreject_bon``, which
    must be a ``vendor_preset`` row (the registry's default; anything else refuses by name);
    :data:`BON_HARNESS_DEFAULT` -> the NON-DEFAULT ``<model>|bon|off|grandfathered_T1`` row by explicit reference
    (readable, never the default; superseded from 2026-09-17 to 2026-10-14, live since the rule that the
    canonical best-of-N stays T = 1). A model with no row refuses (``NoPolicy``) — the harness never invents a
    number."""
    if label not in BON_REGISTRY_LABELS:
        raise BonServedError(f"bon_policy: {label!r} is not a registry label {list(BON_REGISTRY_LABELS)}")
    pf = _decode_policy.load()
    mid = pf.model_id(str(model))
    if mid not in pf.models:
        raise BonServedError(
            f"bon_policy: {model!r} names no model the decode-policy export knows (model_id / legacy key / HF repo); "
            "best-of-N resolves its sampler PER SERVED MODEL — name the served model")
    if label == BON_HARNESS_DEFAULT:
        return pf.by_ref(f"{mid}|bon|off|{_GRANDFATHERED_MODE}")
    policy = pf.resolve(mid, eval_id=BON_EVAL_ID, thinking_mode="off")
    if policy.mode != BON_VENDOR_PRESET:
        raise BonServedError(
            f"bon_policy: the registry's default best-of-N row for {mid} is {policy.policy_id} (mode {policy.mode!r}), "
            f"not a {BON_VENDOR_PRESET} row — the export disagrees with the 2026-09-17 ruling; fix the seed")
    return policy


def bon_sampler(model: Any, label: str = BON_VENDOR_PRESET) -> dict:
    """Exactly the request-body sampler keys best-of-N sends for ``model`` under ``label`` (SAMPLER_KEYS order)."""
    return _ordered(bon_policy(model, label).request_params())


def resolve_bon_sampler(decode: Optional[Mapping[str, Any]], decode_label: Optional[str],
                        model: Any = None) -> BonSampler:
    """The ONE rule that turns a caller's ``(decode, decode_label, model)`` into the sampler a run draws under:

    * ``decode`` None (the default path): ``decode_label`` is a registry label (default
      :data:`BON_VENDOR_PRESET`); the vendor preset needs the served ``model`` (it differs per model); the
      grandfathered row may be resolved model-independently (its numbers are identical on every model by
      construction and the export asserts it) so a legacy store can be read without naming its model;
    * ``decode`` given: an EXPLICIT block the caller vouches for, label :data:`BON_EXPLICIT_PRESET` unless the
      caller names another non-registry label; naming a registry label beside an explicit block refuses — the
      registry resolves those, and a block that happened to match would still be the caller's number.
    """
    if decode is not None:
        problem = sampler_preset_problem(decode)
        if problem is not None:
            raise BonServedError(f"bon decode preset: {problem}")
        if decode_label in BON_REGISTRY_LABELS:
            raise BonServedError(
                f"bon sampler: label {decode_label!r} is the registry's to resolve (pass model= and no decode); an "
                "explicit decode block is labelled by its caller")
        return BonSampler(_ordered(decode), decode_label or BON_EXPLICIT_PRESET, None)
    label = decode_label or BON_VENDOR_PRESET
    if label not in BON_REGISTRY_LABELS:
        raise BonServedError(
            f"bon sampler: no decode block and {label!r} is not a registry label {list(BON_REGISTRY_LABELS)} — "
            f"a run names the served model (model=) and draws at its {BON_VENDOR_PRESET}, or names "
            f"{BON_HARNESS_DEFAULT!r} to finish a store drawn under the grandfathered row")
    if model is None:
        if label == BON_HARNESS_DEFAULT:
            policy = _decode_policy.load().model_independent(eval_id=BON_EVAL_ID, thinking_mode="off",
                                                             mode=_GRANDFATHERED_MODE)
            return BonSampler(_ordered(policy.request_params()), label, policy.policy_id)
        raise BonServedError(
            f"bon sampler: the {BON_VENDOR_PRESET} differs per served model and no model was named — pass model= "
            "(the config's `model`, a legacy key such as gemma4e4b, or the routing table's base_model); best-of-N "
            "has had no model-independent default since the 2026-09-17 ruling")
    policy = bon_policy(model, label)
    return BonSampler(_ordered(policy.request_params()), label, policy.policy_id)


#: a pinned HF revision as the registry states one: 40 lowercase hex; anything else (None, "unverified") is UNKNOWN and compares to nothing
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _known_revision(rev: Any) -> Optional[str]:
    s = str(rev).strip().lower() if rev is not None else ""
    return s if _HEX40.match(s) else None


def same_checkpoint(pf: "_decode_policy.PolicyFile", a: str, b: str, *, declared_revisions: Sequence[Any] = ()) -> bool:
    """Do two model_ids the export knows pin THE SAME CHECKPOINT: the same ``hf_repo`` (both stated), and no pinned
    ``hf_revision`` that disagrees -- between the two rows, and between each row and every ``declared_revisions`` entry a
    caller read off its own declarations (a config's ``model_revision``, a routing table's ``base_revision``) -- wherever
    both sides state one (an ``unverified`` or absent revision is unknown and refutes nothing; a repo nobody states never
    counts as shared). The same model_id is trivially the same checkpoint."""
    if a == b:
        return True
    ra, rb = pf.models.get(a) or {}, pf.models.get(b) or {}
    repo_a, repo_b = ra.get("hf_repo"), rb.get("hf_repo")
    if not (isinstance(repo_a, str) and repo_a and repo_a == repo_b):
        return False
    pinned = [r for r in (_known_revision(ra.get("hf_revision")), _known_revision(rb.get("hf_revision"))) if r]
    declared = [r for r in (_known_revision(x) for x in declared_revisions) if r]
    return len(set(pinned) | set(declared)) <= 1


def served_model_identity(model: Any, base_model: Any = None, model_key: Any = None, *,
                          model_revision: Any = None, base_revision: Any = None,
                          pf: Optional["_decode_policy.PolicyFile"] = None) -> str:
    """THE LOGICAL MODEL a best-of-N run resolves its sampler for, from three declarations: the config's served ``model``
    (the run's own; refused when absent or unknown to the export), the routing table's ``base_model`` (what the standup
    serves) and the launcher's ``model_key`` (``--model-key``, optional). The rule:

    * no key, or a key naming the config's model_id: ``model`` itself, unchanged -- every existing invocation is byte-identical;
    * a key naming ANOTHER model_id of THE SAME CHECKPOINT (:func:`same_checkpoint`: the export's ``hf_repo`` equal, its pinned
      ``hf_revision`` equal to the config's ``model_revision`` and the table's ``base_revision`` wherever both sides state one):
      the KEY's model_id. The worked example ``qwen35_4b_demo`` is the paper's ``Qwen/Qwen3.5-4B @ 851bf6e8`` re-pinned under its
      own key, whose decode rows are its own (ruling decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09); the HF repo spelling
      resolves to the paper's row (vocab.model_spelling_owner), so a run registered under the key must resolve and file under
      the key's rows -- never another model's -- and its config can only spell the repo;
    * a key naming a different checkpoint, or a table serving one: refused (:class:`BonServedError`) -- a sampler read for one
      model never rides a serve of another.
    """
    pf = pf or _decode_policy.load()
    if not isinstance(model, str) or not model:
        raise BonServedError("the config names no served `model`; the best-of-N sampler resolves per served model")
    mid = pf.model_id(model)
    if mid not in pf.models:
        raise BonServedError(f"the config's model {model!r} is not a model the decode-policy export knows -- no best-of-N row "
                             "to draw under; add the model to the registry seeds first")
    identity, identity_id = model, mid
    if model_key is not None:
        kid = pf.model_id(str(model_key))
        if kid != mid:
            if kid in pf.models and same_checkpoint(pf, kid, mid, declared_revisions=(model_revision, base_revision)):
                identity, identity_id = kid, kid
            else:
                why = (f"{kid} pins {(pf.models.get(kid) or {}).get('hf_repo')!r} @ {(pf.models.get(kid) or {}).get('hf_revision')!r}; "
                       f"{mid} pins {(pf.models.get(mid) or {}).get('hf_repo')!r} @ {(pf.models.get(mid) or {}).get('hf_revision')!r}; "
                       f"the config declares revision {model_revision!r}, the routing table {base_revision!r}"
                       if kid in pf.models else f"{kid} is not a model the decode-policy export knows")
                raise BonServedError(f"--model-key {model_key!r} ({kid}) disagrees with the config's model {model!r} ({mid}) -- "
                                     f"not the same checkpoint ({why}); name one served model")
    base = base_model
    if isinstance(base, str) and base and pf.model_id(base) in pf.models:
        bid = pf.model_id(base)
        if not same_checkpoint(pf, bid, identity_id, declared_revisions=(base_revision,)):
            raise BonServedError(f"the routing table serves base_model {base!r} ({bid}) but the config names {model!r} ({mid}) "
                                 f"-- refusing to resolve a sampler for one and draw on the other")
    return identity


def label_for_decode(model: Any, decode: Mapping[str, Any]) -> str:
    """The registry label (:data:`BON_REGISTRY_LABELS` in order: the vendor preset first, the grandfathered row second)
    whose sampler for ``model`` SENDS every sampling key ``decode`` STATES at the stated value. ``decode`` is an instrument's
    six decode numbers keyed by ``decode_policy.PARAM_KEYS`` (``None`` = not stated / not sent -- an instrument that leaves a key
    to the server is reproduced by any label that agrees on the keys it does state: the rule applied to
    the pipeline-test models, whose vendor rows send temperature 1.0 and the T=1.0 instrument states nothing else). Refuses
    (:class:`BonServedError`) when neither label reproduces it, naming the stated numbers and each label's sampler -- a run
    never draws a sampler the instrument it registers under does not state."""
    stated = {k: decode.get(k) for k in _decode_policy.PARAM_KEYS if decode.get(k) is not None}
    if not stated:
        raise BonServedError("label_for_decode: the instrument states no decode number at all; nothing to reproduce")
    tried = {}
    for label in BON_REGISTRY_LABELS:
        try:
            sampler = bon_sampler(model, label)
        except (BonServedError, _decode_policy.NoPolicy) as exc:
            tried[label] = f"no sampler ({exc})"
            continue
        off = {k: (v, sampler.get(k)) for k, v in stated.items()
               if k not in sampler or float(sampler[k]) != float(v)}
        if not off:
            return label
        tried[label] = f"sends {sampler}; disagrees on {sorted(off)}"
    raise BonServedError(f"no registry best-of-N label reproduces the instrument's stated decode {stated} for {model!r}: "
                         + "; ".join(f"{k}: {v}" for k, v in tried.items()))


def canonical_bon_label(model: Any, serving_path: str, *, seeds_dir: str) -> Tuple[str, Dict[str, Any]]:
    """``(label, facts)``: the registry label whose sampler reproduces THE CANONICAL best-of-N instrument the registry's planner
    binds for ``model`` via ``serving_path`` (``jobplan.canonical_instrument`` over ``seeds_dir``, the same call a job's start-of-job
    registration makes) -- so a run that registers under that instrument draws exactly the sampler it states, and a run whose
    instrument no label reproduces refuses by name before a byte lands. ``facts`` carries the instrument's id, seed id,
    ``decode_source`` and the six stated numbers for the run record. The seeds are the caller's checkout's (``configs/registry``);
    PyYAML and the seeds loader are imported here, never at module load."""
    from sisteer.registry import jobplan as JP   # lazy: PyYAML + the seeds loader live behind it
    pf = _decode_policy.load()
    mid = pf.model_id(str(model))
    if mid not in pf.models:
        raise BonServedError(f"canonical best-of-N: {model!r} names no model the decode-policy export knows")
    try:
        idx = JP.load_seed_index(seeds_dir)
        payload, extra = JP.canonical_instrument(idx, BON_EVAL_ID, mid, serving_path)
    except LookupError as exc:
        raise BonServedError(f"canonical best-of-N: {exc}") from exc
    decode = {k: payload.get(f"decode_{k}") for k in _decode_policy.PARAM_KEYS}
    label = label_for_decode(mid, decode)
    return label, {"instrument_id": str(payload["instrument_id"]), "seed_instrument_id": extra.get("seed_instrument_id"),
                   "decode_source": payload.get("decode_source"), "decode": decode, "serving_path": serving_path,
                   "seeds_dir": str(seeds_dir), "seeds_revision": idx.seeds_revision}


def sampler_preset_problem(decode: Mapping[str, Any]) -> Optional[str]:
    """Why ``decode`` is NOT a sendable sampler preset (``None`` = it is): a key
    outside :data:`SAMPLER_KEYS`, a non-numeric value (bools included — a
    ``True`` where a float belongs is the classic silent-coercion bug), or a
    preset that states no ``temperature`` (the request ALWAYS carries a
    temperature field, so a preset without one would silently sample at
    whatever the caller's greedy default put there). Shared by every served
    client so the refusal is one rule, not one per class."""
    if not isinstance(decode, Mapping):
        return f"decode preset must be a mapping, got {type(decode).__name__}"
    bad = sorted(set(decode) - set(SAMPLER_KEYS))
    if bad:
        return (f"decode preset carries key(s) outside the sampler set {bad}; "
                f"allowed: {list(SAMPLER_KEYS)}")
    for k in SAMPLER_KEYS:
        if k not in decode:
            continue
        v = decode[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"decode preset {k}={v!r} is not a number"
    if "temperature" not in decode:
        return ("decode preset states no temperature — the request always "
                "carries one, so a preset without it would sample at the "
                "caller's default rather than the vendor's")
    return None


def sampler_preset(decode: Mapping[str, Any]) -> dict:
    """The preset's sendable keys in :data:`SAMPLER_KEYS` order — EXACTLY the
    keys present (no defaults filled in), after :func:`sampler_preset_problem`
    has ruled it clean. Raises :class:`BonServedError` otherwise."""
    problem = sampler_preset_problem(decode)
    if problem is not None:
        raise BonServedError(problem)
    return {k: decode[k] for k in SAMPLER_KEYS if k in decode}


def _geometry_stamp() -> dict:
    """The run's geometry stamp (``{"geometry_id": ...}`` when the launcher
    exported one through ``run_identity.stamp_env``) — merged into every served
    row so a store mixing two serving geometries is detectable row by row
    (review R1: bf16 greedy output is not batch-composition invariant)."""
    from sisteer.capeval.serving.run_identity import geometry_stamp
    return geometry_stamp()

#: Default ceiling on in-flight requests. Continuous batching wants the
#: scheduler kept busy; this bounds how many the client fans out at once so a
#: pathological run cannot open thousands of sockets. It is the ``bon``
#: entry's ``max_num_seqs`` in :mod:`sisteer.capeval.serving.geometry` (the
#: per-eval geometry ruling, 2026-09-03), so the client fan-out cannot exceed
#: the server width the table launches by construction — the old literal 64
#: sat above the 24-way servers some legs talked to. Right-size to the
#: launched ``max_num_seqs`` at the call site when it differs.
DEFAULT_MAX_CONCURRENCY = geometry_for("bon").max_num_seqs


class BonServedError(RuntimeError):
    """A wiring / provenance error on the served BoN path (loud, never a
    silent degrade to a worse path — authoritative-source rule)."""


# --------------------------------------------------------------------------- #
# Per-cell KV-cache isolation assertion (the correctness boundary).           #
# --------------------------------------------------------------------------- #


def _is_lens(route: RouteEntry) -> bool:
    """True iff this route carries a steering selector — the leg the proxy
    salts REQUEST-UNIQUE. Read from the serving identity, the same source
    :meth:`RoutingProxy._rewrite` keys on. Genuinely LENS-ONLY: the salt
    regime below keys on it (a native-LoRA leg is cell-stable, like base)."""
    return cm.parse_serving_identity(route.serving_identity).path == "lens"


def is_steered(route: RouteEntry) -> bool:
    """True iff this route serves a STEERING INTERVENTION — a lens hook OR a
    native LoRA adapter. The predicate
    the drivers gate the steering-bites proof on: a lora cell steers the
    model exactly as a lens cell does, so it needs the same inert-serve proof
    against its path-matched scale-0 twin (``proxy_engine.scale0_twin``);
    only the salt regime stays lens-only (:func:`_is_lens`). One literal,
    ``proxy.STEERED_ROUTE_PATHS``, shared with ``proxy_engine.is_steered_route``."""
    return cm.parse_serving_identity(route.serving_identity).path in STEERED_ROUTE_PATHS


def cell_salt_isolation_report(routes: Sequence[RouteEntry]) -> dict:
    """Diagnose the per-cell salt policy over the routes a run will serve,
    WITHOUT a server — pure string derivation, CPU-only.

    Returns a dict with, per cell: whether it is a lens leg, whether its salt
    is stable across draws (the caching precondition for unhooked/LoRA legs)
    or request-unique across draws (the isolation precondition for lens legs),
    and the set of stable-salt collisions across DISTINCT cells (which must be
    empty)."""
    per_cell: dict = {}
    stable_owners: dict = {}
    collisions: list = []
    for r in routes:
        lens = _is_lens(r)
        # a NATIVE route at a restricted position mode (the residual slot's
        # `response` / `prompt_last` / `prefill_all`) is minted REQUEST-UNIQUE like a lens leg —
        # the salt is derived exactly as the proxy derives it, mode included
        mode = getattr(r, "petri_position_mode", None)
        restricted = bool(mode is not None and str(mode) != "all")
        # the pre-stage-2 call shape on every route without a mode (a LoRA / base / lens route):
        # the mode is passed only when the route carries one
        kw = {"position_mode": mode} if mode is not None else {}
        s1 = RoutingProxy.cache_salt_for(r, lens=lens, **kw)
        s2 = RoutingProxy.cache_salt_for(r, lens=lens, **kw)
        entry = {
            "cell_id": r.cell_id,
            "lens": lens,
            "position_mode": mode,
            "restricted_native": restricted,
            "stable": (s1 == s2),
            "salt_example": s1,
        }
        per_cell[r.virtual_model] = entry
        if not lens:
            # a cell-stable salt: two DISTINCT cells resolving to the same
            # salt could share a cached prefix — the exact leak the salt
            # exists to prevent.
            prev = stable_owners.get(s1)
            if prev is not None and prev != r.cell_id:
                collisions.append((prev, r.cell_id, s1))
            stable_owners[s1] = r.cell_id
    return {"per_cell": per_cell, "stable_collisions": collisions}


def assert_cell_salt_isolation(routes: Sequence[RouteEntry]) -> None:
    """FAIL LOUDLY if two distinct cells the run will serve could ever share a
    cached prefix, or if a cell's salt policy is wrong for its leg. This is the
    assertion the task requires before the fan-out ships.

    A KV block is shareable only when its ``cache_salt`` matches, so proving
    that distinct cells never produce the same salt proves they can never share
    a cached (possibly steered) prefix. Concretely:

    - **No cross-cell collision.** Two DISTINCT cell ids must never yield the
      same cell-stable salt (lens legs are request-unique, so they cannot
      collide by construction).
    - **Unhooked / LoRA legs are cell-STABLE.** Their salt must be identical
      across draws, or the SI prefix would never cache and the whole speedup on
      those legs evaporates (a regression guard, not merely a nicety).
    - **Lens legs are request-UNIQUE.** Their salt must differ across draws, or
      a steered prefill could resume from cache with the hooks never firing —
      the silent mis-defense.

    Raises :class:`BonServedError` on any violation (never warns-and-continues:
    an undetected salt collision is the worst outcome in this project)."""
    if not routes:
        raise BonServedError(
            "assert_cell_salt_isolation: no routes to isolate — an empty "
            "route set proves nothing; refusing rather than passing vacuously")
    report = cell_salt_isolation_report(routes)
    if report["stable_collisions"]:
        pairs = ", ".join(f"{a!r}<->{b!r}" for a, b, _ in
                          report["stable_collisions"])
        raise BonServedError(
            "assert_cell_salt_isolation: distinct cells resolve to the SAME "
            f"cache_salt ({pairs}) — with prefix caching on they could serve "
            "one cell's cached (steered) KV under another cell's identity; "
            "per-cell isolation is NOT holding")
    for vm, e in report["per_cell"].items():
        if e["lens"] and e["stable"]:
            raise BonServedError(
                f"assert_cell_salt_isolation: lens cell {e['cell_id']!r} "
                f"(vmodel {vm!r}) has a STABLE salt — a steered prefill must "
                "never resume from cache (the hooks cannot fire on cached "
                "rows); it must be request-unique")
        if e.get("restricted_native"):
            # a restricted-mode NATIVE route must be request-unique: a stable salt here
            # would let a steered prefill resume from cache under another request's boundary
            if e["stable"]:
                raise BonServedError(
                    f"assert_cell_salt_isolation: restricted-mode native cell {e['cell_id']!r} "
                    f"(vmodel {vm!r}, position mode {e['position_mode']!r}) has a STABLE salt — "
                    "the residual slot's restricted modes are minted request-unique (D12)")
            continue
        if (not e["lens"]) and (not e["stable"]):
            raise BonServedError(
                f"assert_cell_salt_isolation: unhooked/LoRA cell "
                f"{e['cell_id']!r} (vmodel {vm!r}) has an UNSTABLE salt — its "
                "SI prefix would never cache, defeating the speedup this path "
                "exists for")


# --------------------------------------------------------------------------- #
# Generation-time inert-serve proof (steering must bite).                      #
# --------------------------------------------------------------------------- #

#: A steering-bites probe: given an arm id, returns the token-id lists for the
#: arm at its serving scale and for the SAME requests with the SAME object
#: selected at scale 0 — the evidence :func:`fidelity._check_steering_bites`
#: rules on. Kept as an injectable seam so the proof runs against the SAME
#: serving path the audit uses (and so tests exercise it with no server).
SteeringProbe = Callable[[str], "tuple[list, list]"]  # arm -> (steered, scale0)


def assert_steering_bites(arms: Sequence[str], probe: SteeringProbe) -> dict:
    """Prove every STEERED arm the run will serve actually changes the output,
    by probing it through the serving path and ruling on the evidence with
    :func:`fidelity._check_steering_bites`. Raises
    :class:`~sisteer.capeval.serving.fidelity.FidelityError` on an inert arm
    (the stock-extension / silently-unhooked failure mode) — never accepts a
    served model on faith. Returns the per-arm divergence diagnostic."""
    if not arms:
        raise BonServedError(
            "assert_steering_bites: no arms to probe — refusing to certify a "
            "steered run with zero evidence")
    evidence = {"arms": {}}
    for arm in arms:
        steered_ids, scale0_ids = probe(arm)
        evidence["arms"][arm] = {"steered_ids": steered_ids,
                                 "scale0_ids": scale0_ids}
    diag: dict = {}
    fidelity._check_steering_bites(evidence, diag)  # raises on an inert arm
    return diag


# --------------------------------------------------------------------------- #
# The served defense adapter (one cell, concurrent fan-out through the proxy). #
# --------------------------------------------------------------------------- #

#: An async chat transport: request dict -> OpenAI chat-completion dict. This
#: is the ONLY thing the served defense talks to, so a RoutingProxy (in
#: process) and a tunnelled ProxyClientEngine (over the wire) are the same
#: seam — and a test drives it with a stub, no server.
AsyncChat = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]


#: The BoN typed-error ledger beside ``bon_responses.jsonl``: one row per
#: (goal, sample, trial) whose prompt could not fit the serving window. A
#: SEPARATE file on purpose — ``_load_covered`` reads responses only, so the
#: slot stays UNCOVERED (a resume retries it; ``bon_curves`` refuses to grade
#: the goal as incomplete) instead of a phantom "" sample graded as a refusal.
#: APPEND-ONLY HISTORY: a row is never retired here; a later successful
#: response row for the same (goal, sample, trial) SUPERSEDES it, and the
#: live grader counts such rows as ``n_errors_superseded`` (provenance), not
#: as uncovered slots (grade_bon_live.assert_bon_raw_complete).
BON_ERRORS_FILE = "bon_errors.jsonl"
#: The served-BoN manifest (the BoN twin of served_target.json), written in a
#: ``finally`` after generation so its counters are authoritative.
BON_SERVED_MANIFEST = "bon_served_manifest.json"


@dataclass(frozen=True)
class ServedTurn:
    """One served generation with its per-call facts: the text, the server's
    ``finish_reason``, the usage counts, the ``max_tokens`` actually sent,
    ``max_tokens_clamped_from`` when the once-only window clamp fired, and
    ``window_mismatch`` (threaded vs server) when the launcher's window
    disagreed with the server's. A prompt that cannot fit is NOT a turn — it
    raises :class:`~sisteer.capeval.serving.served_window.PromptOverWindow`."""

    text: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    max_tokens: int
    max_tokens_clamped_from: Optional[int] = None
    window_mismatch: Optional[dict] = None
    prompt_tokens_source: Optional[str] = None
    #: the engine's completion id (``resp["id"]``; request evidence) — read by the BoN row
    #: writer ONLY when a request-headers hook is in play; ``turn_record`` never emits it (rows byte-identical)
    completion_id: Optional[str] = None
    #: the ``X-Tinycap-Request-Id`` of the attempt that PRODUCED this turn (see ``generate_one_detailed``'s
    #: ``attempt_headers_fn``); None without a hook. Read by the BoN row writer beside ``completion_id``; never in
    #: ``turn_record``.
    request_id: Optional[str] = None


def turn_record(turn: ServedTurn, *, sampling_seed: Optional[int]) -> dict:
    """The DETAILED per-generation dict the ``generate_detailed`` surfaces
    return (``ServedStaticDefense`` here, ``ServedChatTarget`` in
    served_target): the text plus the per-call facts a sampled row records
    (``finish_reason`` is what the at-cap ladder keys on), the ``max_tokens``
    actually sent, the seed the draw was issued with, and the clamp fields
    ONLY when the once-only window clamp fired — so a row's key set says
    whether a clamp happened without a null-valued placeholder."""
    rec = {"text": turn.text,
           "finish_reason": turn.finish_reason,
           "completion_tokens": turn.completion_tokens,
           "prompt_tokens": turn.prompt_tokens,
           "max_tokens": int(turn.max_tokens),
           "sampling_seed": (int(sampling_seed) if sampling_seed is not None
                             else None)}
    if turn.max_tokens_clamped_from is not None:
        rec["max_tokens_clamped_from"] = int(turn.max_tokens_clamped_from)
        rec["prompt_tokens_source"] = turn.prompt_tokens_source
    if turn.window_mismatch is not None:
        rec["window_mismatch"] = dict(turn.window_mismatch)
    return rec


class ServedDefense:
    """A per-cell, server-backed BoN defense: it renders one chat request per
    variant and issues it through ``achat`` (the proxy), never touching
    ``cache_salt`` (the proxy injects the correct per-cell salt). It presents
    just enough of the legacy ``LLMHFSteered`` surface for BoN — an
    ``n_target_forwards`` counter (JB7) and a coroutine that returns one
    response — while the CONCURRENCY lives in :func:`bon_generate_served`.

    Construct it from a proxy or a proxy client, NEVER from a bare
    ``base_url``: routing/salt/prefix-verification are the proxy's, and a raw
    URL would bypass every one of them (the exact thing the task forbids)."""

    def __init__(
        self,
        achat: AsyncChat,
        *,
        virtual_model: str,
        cell_id: str,
        system: Optional[str] = None,
        max_model_len: Optional[int] = None,
        atokenize: Optional[AsyncChat] = None,
        serving_engine: Optional[str] = None,
        proxy_identity: Optional[Mapping[str, Any]] = None,
        achat_with_headers: Optional[Callable[[Mapping[str, Any], Mapping[str, str]], Awaitable[Any]]] = None,
    ) -> None:
        if isinstance(achat, (str, bytes)) or not callable(achat):
            raise BonServedError(
                "ServedDefense needs an async chat callable (a RoutingProxy or "
                "ProxyClientEngine transport), never a bare base_url — routing, "
                "the per-cell cache_salt and SI-prefix verification are the "
                "proxy's to own; a raw URL bypasses all three")
        if not virtual_model:
            raise BonServedError("ServedDefense needs a virtual_model name")
        #: THE SERVING ENGINE this defense's rows are stamped with:
        #: stated, or derived once from the proxy identity handed in (see
        #: :func:`resolve_serving_engine` for the era default)
        self.serving_engine = resolve_serving_engine(serving_engine, proxy_identity,
                                                     what="ServedDefense")
        if max_model_len is not None and (
                isinstance(max_model_len, bool) or int(max_model_len) < 1):
            raise BonServedError(
                f"max_model_len must be a positive int, got {max_model_len!r}")
        self._achat = achat
        #: the request-evidence seam: the SAME transport taking per-request headers
        #: (``ProxyClientEngine.chat_completions(payload, headers=...)`` — the tinycap task / sample labels and the
        #: per-attempt request id ride beside the auth header, the usage ledger consumes them). None (every caller
        #: today) = requests go through ``achat`` exactly as before; a caller that hands headers to a defense
        #: without this transport is refused, never silently un-headered.
        self._achat_with_headers = achat_with_headers if callable(achat_with_headers) else None
        self.virtual_model = virtual_model
        self.cell_id = cell_id
        self.system = system
        self.n_target_forwards = 0
        #: the serving window the launcher resolved (geometry.py), arming the
        #: once-only max_tokens clamp (served_window); None still clamps
        #: against the window the server states in its rejection.
        self.max_model_len = int(max_model_len) if max_model_len is not None else None
        #: clamp bookkeeping: calls served with a clamped max_tokens, calls
        #: that raised PromptOverWindow, and the distinct window disagreements.
        self.n_clamped = 0
        self.n_prompt_over_window = 0
        self.n_window_mismatch = 0
        self.window_mismatches: list = []
        #: the proxy's read-only /tokenize (async): resolves a FLOOR-ONLY
        #: rejection's exact prompt length before the clamp; None -> the typed
        #: error with reason floor_only_unresolvable (never a guess).
        self._atokenize = atokenize if callable(atokenize) else None
        self.n_tokenize_resolved = 0
        self.n_floor_only_unresolvable = 0

    async def _plan(self, exc: BaseException, req: Mapping[str, Any]
                    ) -> Optional[sw.ClampPlan]:
        """The clamp plan for a failed request (None when not a context-length
        400); a floor-only rejection is resolved through /tokenize on the
        IDENTICAL messages, or filed floor_only_unresolvable."""
        plan = sw.plan_for_exception(exc, requested_max_tokens=req["max_tokens"],
                                     window=self.max_model_len)
        if plan is None or not plan.needs_exact_prompt_tokens:
            return plan
        if self._atokenize is None:
            self.n_floor_only_unresolvable += 1
            return plan.unresolvable("no /tokenize transport on this defense")
        try:
            count = sw.exact_prompt_tokens(await self._atokenize(sw.tokenize_payload(req)))
        except Exception as exc2:  # noqa: BLE001 — 404 / connection / shape: never guess
            self.n_floor_only_unresolvable += 1
            return plan.unresolvable(f"/tokenize failed: {type(exc2).__name__}: "
                                     f"{str(exc2)[:200]}")
        self.n_tokenize_resolved += 1
        return plan.with_exact_prompt_tokens(count)

    def _note_mismatch(self, plan: sw.ClampPlan) -> None:
        """Log LOUDLY (once per distinct pair) and record a disagreement
        between the threaded window and the server's stated one."""
        mm = plan.mismatch_record()
        if mm is None:
            return
        self.n_window_mismatch += 1
        if mm not in self.window_mismatches:
            self.window_mismatches.append(mm)
            print(f"[bon-served] WINDOW MISMATCH for {self.virtual_model}: the "
                  f"launcher threaded max_model_len={mm['threaded']} but the server "
                  f"states {mm['server']} — the server's value governs the clamp; "
                  "fix the launcher's geometry provenance", flush=True)

    def clamp_report(self) -> dict:
        return {"max_model_len": self.max_model_len,
                "min_clamped_completion": sw.MIN_CLAMPED_COMPLETION,
                "n_clamped": self.n_clamped,
                "n_prompt_over_window": self.n_prompt_over_window,
                "n_window_mismatch": self.n_window_mismatch,
                "window_mismatches": list(self.window_mismatches),
                "n_tokenize_resolved": self.n_tokenize_resolved,
                "n_floor_only_unresolvable": self.n_floor_only_unresolvable,
                "n_target_forwards": self.n_target_forwards}

    @classmethod
    def over_proxy(cls, proxy: RoutingProxy, **kw) -> "ServedDefense":
        """Wrap an in-process :class:`RoutingProxy` (its
        ``handle_chat_completions`` coroutine is the transport; its
        ``handle_tokenize`` resolves floor-only rejections)."""
        kw.setdefault("atokenize", getattr(proxy, "handle_tokenize", None))
        if "proxy_identity" not in kw and "serving_engine" not in kw:
            # the in-process proxy's own /v1/models identity block (its engine
            # attestation rides it when the standup is native) — pure, no I/O
            payload_fn = getattr(proxy, "models_payload", None)
            if callable(payload_fn):
                kw["proxy_identity"] = (payload_fn() or {}).get("petri_proxy")
        return cls(proxy.handle_chat_completions, **kw)

    @classmethod
    def over_client(cls, client, **kw) -> "ServedDefense":
        """Wrap a synchronous ``ProxyClientEngine`` by running its blocking
        ``chat_completions`` in a worker thread, so many requests still fan out
        concurrently from one event loop (and its ``tokenize`` the same way)."""
        # request-evidence: off = the client is returned untouched; on = every chat_completions through it is
        # a recorded attempt — the labels + id from the caller's headers (the `request_headers_fn` seam) or from a
        # `labelled(...)` context, a fresh id per HTTP attempt otherwise; the response bound by its completion id
        client = rev.maybe_attach_proxy_client(client)

        async def achat(payload):
            return await asyncio.to_thread(client.chat_completions, dict(payload))

        async def achat_with_headers(payload, headers):
            # the client's own header-carrying call (ProxyClientEngine.chat_completions(payload, headers=...)):
            # taken ONLY when a request carries headers; the plain call above stays byte-identical
            return await asyncio.to_thread(client.chat_completions, dict(payload), headers=dict(headers))
        kw.setdefault("achat_with_headers", achat_with_headers)
        tokenize = getattr(client, "tokenize", None)
        if tokenize is not None and "atokenize" not in kw:
            async def atokenize(payload):
                return await asyncio.to_thread(tokenize, dict(payload))
            kw["atokenize"] = atokenize
        if "proxy_identity" not in kw and "serving_engine" not in kw:
            # the identity block ProxyClientEngine.load captured (None before load,
            # or on a proxy predating the block: lens by era)
            kw["proxy_identity"] = getattr(client, "proxy_identity", None)
        return cls(achat, **kw)

    def _request(self, variant: str, *, sampling_seed: int,
                 max_new_tokens: int, decode: Mapping[str, Any]) -> dict:
        messages = []
        if self.system is not None:
            # prompted / stacked cells: the SI rides the system turn and the
            # proxy verifies its hash (H8). Non-prompted cells send no system
            # turn — the proxy refuses an unexpected registered prefix.
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": variant})
        req = {
            "model": self.virtual_model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "seed": sampling_seed,
        }
        # THE SAMPLER: exactly the preset's keys ride the body (a temperature
        # always among them — sampler_preset refuses a block without one);
        # anything outside SAMPLER_KEYS is refused here. The served model's
        # vendor preset by default (resolve_bon_sampler), never a positional
        # constant.
        if sampling_seed is None:
            # the same rule served_target._payload holds for the multi-turn target: a preset draw with no
            # per-draw seed cannot be reproduced -> refused, never `seed: null` on the wire. Every caller
            # today passes an int (the fan-out derives one per unit; ServedStaticDefense refuses seeds=None
            # under a preset), so no existing request changes.
            raise BonServedError(
                f"cell {self.cell_id!r}: a decode preset needs a per-draw sampling_seed — an unseeded "
                "sampled draw cannot be reproduced; refusing")
        req.update(sampler_preset(decode))
        # defensive: this path must NEVER smuggle a proxy-owned key.
        bad = sorted(set(req) & set(FORBIDDEN_CLIENT_KEYS))
        if bad:
            raise BonServedError(
                f"ServedDefense would send reserved proxy key(s) {bad} — "
                "these are the proxy's to inject, never a client's")
        return req

    async def _send(self, req: Mapping[str, Any], request_headers: Optional[Mapping[str, str]]) -> Any:
        """ONE request over the transport: ``achat`` (every request today), or the header-carrying transport when the
        caller hands per-request headers (request evidence) — a defense without one REFUSES rather
        than sending the labels' request without them."""
        if request_headers is None:
            # request-evidence: the served target's guard, restated for this transport — with the hook ON a task
            # send that no caller labelled (no attempt-headers hook, no `labelled(...)` context) is REFUSED by name
            # here, before the wire. The client behind `over_client` may carry `unlabelled_kind=probe/bites` for the
            # steering-bites probe (bon_served_run attaches it once, as build_served_target does), and without this
            # guard an unlabelled DRAW would be recorded as a probe — a mislabel the audit cannot see. Off = a no-op.
            rev.require_label(f"bon served defense {self.cell_id!r}")
            return await self._achat(req)
        if self._achat_with_headers is None:
            raise BonServedError(
                f"cell {self.cell_id!r}: request headers were built for this draw but the defense has no "
                "header-carrying transport (ServedDefense(achat_with_headers=...); over_client provides one) — "
                "refusing to send the request un-labelled")
        return await self._achat_with_headers(req, request_headers)

    async def generate_one_detailed(self, variant: str, *, sampling_seed: int,
                                    max_new_tokens: int,
                                    decode: Optional[Mapping[str, Any]] = None,
                                    temperature: Optional[float] = None,
                                    attempt_headers_fn: Optional[Callable[[], Mapping[str, str]]] = None
                                    ) -> ServedTurn:
        """Issue ONE sampled generation for one augmented variant and return
        the :class:`ServedTurn` (text + per-call facts). Counts one real target
        forward (JB7). THE SAMPLER IS THE CALLER'S TO NAME: ``decode`` (a
        sampler block, SAMPLER_KEYS only — the served model's vendor preset,
        :func:`bon_sampler`, on the default path), or an explicit
        ``temperature`` alone (the degenerate one-key block: the old wire shape
        of the grandfathered runs). Neither -> refused: since the 2026-09-17
        ruling there is no model-independent default temperature.

        THE WINDOW CLAMP (served_window, the per-eval geometry ruling): a
        context-length 400 is retried ONCE with ``max_tokens`` clamped to the
        room the SERVER-STATED window leaves (the threaded window is
        provenance + a cross-check); a prompt that cannot fit (or a second
        rejection) raises the TYPED ``PromptOverWindow`` — never an empty
        string in place of a response. Any other failure propagates.

        ``attempt_headers_fn`` (request evidence): called ONCE PER ACTUAL SEND — the
        first attempt AND the clamped retry each — to mint THAT attempt's headers (the two tinycap labels and a FRESH
        ``X-Tinycap-Request-Id``), so the ledger holds one id per attempted call and the evidence records one id per
        attempt; the returned turn carries the SUCCESSFUL attempt's ``request_id`` beside its ``completion_id``. A
        hook that mints no id, or re-mints an id this call already sent, is refused before the send (one id on two
        attempts is exactly what ``request_evidence_reconciliation`` cannot reconcile). None (every caller today) =
        no headers, the request exactly as before."""
        if decode is None:
            if temperature is None:
                raise BonServedError(
                    "generate_one_detailed: no sampler — pass decode= (the served model's BoN policy, "
                    "bon_sampler(model), or an explicit block) or temperature=; best-of-N has no "
                    "model-independent default temperature since the 2026-09-17 ruling")
            decode = {"temperature": temperature}
        elif temperature is not None and not (isinstance(decode.get("temperature"), (int, float))
                                              and float(decode["temperature"]) == float(temperature)):
            raise BonServedError(
                f"generate_one_detailed: temperature={temperature!r} disagrees with decode['temperature']="
                f"{decode.get('temperature')!r} — name the sampler once")
        self.n_target_forwards += 1
        req = self._request(variant, sampling_seed=sampling_seed,
                            max_new_tokens=max_new_tokens, decode=decode)
        attempt_ids: list = []

        async def send(body: Mapping[str, Any]) -> Any:
            # ONE attempt: its own headers, its own request id — one request id per actual send
            headers = None
            if attempt_headers_fn is not None:
                headers = dict(attempt_headers_fn())
                rid = tc._header_value(headers, tc.HEADER_REQUEST_ID)      # the gate's own header reader (contracts)
                if not rid:
                    raise BonServedError(
                        f"cell {self.cell_id!r}: the attempt-headers hook returned no {tc.HEADER_REQUEST_ID} — every "
                        "attempt is minted an id at the HTTP boundary; refusing to send an unidentified request")
                if rid in attempt_ids:
                    raise BonServedError(
                        f"cell {self.cell_id!r}: the attempt-headers hook re-minted request id {rid!r} for the retry of one "
                        "draw — one id on two attempts cannot be reconciled against the ledger (plan §9.1); mint a fresh "
                        "id per send")
                attempt_ids.append(rid)
            return await self._send(body, headers)

        clamped_from: Optional[int] = None
        mismatch: Optional[dict] = None
        prompt_source: Optional[str] = None
        resp: Any = None
        try:
            resp = await send(req)
        except Exception as exc:  # noqa: BLE001 — only a context-length 400 is handled
            plan = await self._plan(exc, req)
            if plan is None:
                raise
            self._note_mismatch(plan)
            mismatch = plan.mismatch_record()
            prompt_source = plan.prompt_tokens_source
            if plan.action == sw.RETRY:
                clamped_from = int(req["max_tokens"])
                req["max_tokens"] = int(plan.max_tokens)
                try:
                    resp = await send(req)
                except Exception as exc2:  # noqa: BLE001
                    plan2 = await self._plan(exc2, req)
                    if plan2 is None:
                        raise
                    plan = plan2.failed(
                        f"rejected again after the single clamped retry "
                        f"({clamped_from} -> {req['max_tokens']}): {plan2.reason}")
                    resp = None
            if resp is None:
                self.n_prompt_over_window += 1
                raise sw.PromptOverWindow.from_plan(
                    plan, requested_max_tokens=int(max_new_tokens))
        try:
            choice = resp["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BonServedError(
                f"served response for cell {self.cell_id!r} lacks "
                f"choices[0].message.content: {resp!r}") from exc
        if clamped_from is not None:
            self.n_clamped += 1
        usage = (resp.get("usage") or {}) if isinstance(resp, Mapping) else {}
        cid = resp.get("id") if isinstance(resp, Mapping) else None
        return ServedTurn(text=text,
                          finish_reason=(choice.get("finish_reason")
                                         if isinstance(choice, Mapping) else None),
                          prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"),
                          max_tokens=int(req["max_tokens"]),
                          max_tokens_clamped_from=clamped_from,
                          window_mismatch=mismatch,
                          prompt_tokens_source=prompt_source,
                          completion_id=(str(cid) if isinstance(cid, str) and cid else None),
                          # the attempt that produced this response: the LAST id minted (the retry's when it fired)
                          request_id=(attempt_ids[-1] if attempt_ids else None))

    async def generate_one(self, variant: str, *, sampling_seed: int,
                           max_new_tokens: int,
                           decode: Optional[Mapping[str, Any]] = None,
                           temperature: Optional[float] = None) -> str:
        """Issue ONE sampled generation for one augmented variant and return
        the assistant text (``generate_one_detailed().text``; a prompt that
        cannot fit raises ``PromptOverWindow``). Counts one real target
        forward (JB7). The sampler is the caller's to name, as there."""
        turn = await self.generate_one_detailed(
            variant, sampling_seed=sampling_seed, max_new_tokens=max_new_tokens,
            decode=decode, temperature=temperature)
        return turn.text


# --------------------------------------------------------------------------- #
# The served STATIC-family adapter (whole family fanned out through the proxy).#
# --------------------------------------------------------------------------- #


class ServedStaticDefense:
    """``run_static``'s synchronous, batched ``.generate(prompts)`` surface,
    backed by concurrent fan-out through the routing proxy — the ONE served
    adapter every non-BoN static/held-out family reaches (they all funnel
    through :func:`sisteer.eval.attacks.static_common.run_static`, whose only
    contact with the target is ``defense.generate(chunk)`` +
    ``defense.batch_size`` + ``defense.n_target_forwards``).

    ``run_static`` slices its goal-major work list into consecutive chunks of
    ``batch_size`` and issues one ``.generate(chunk)`` per slice; we set
    ``batch_size`` ENORMOUS so a whole family is ONE chunk and every request is
    in flight at once (bounded by ``max_concurrency`` == the server's
    ``max_num_seqs``) — true continuous batching, not synchronous waves. This is
    the same adapter the measured reference driver
    (``scripts/petri_families_prompted.py``) used to reach the 20.5-37.6x
    per-family and 27.3x composite speedups, so the sweep runs the PROVEN
    fan-out rather than a reinvented one.

    Wraps a :class:`ServedDefense` (which owns the per-cell request build, the
    forbidden-key guard, and the ``n_target_forwards`` counter, JB7), so it is
    ALWAYS through the proxy — never a bare ``base_url`` — and greedy
    (temperature 0) to match the HF arm's ``do_sample=False``.

    ``engine_tag`` is ``SERVED_ENGINE``: ``run_static`` stamps it on every row
    it writes (so a served responses file is self-describing and never silently
    mixes with HF rows — the guard is
    :func:`static_common._assert_static_engine_consistent`), and its presence
    also switches ``run_static`` onto the LOUD error path (a failed chunk leaves
    its goals UNCOVERED to regenerate on resume, never finalized-as-error).

    ERROR POLICY (``raise_on_error``). Default ``True`` — the sweep default:
    any request that fails after ``max_retries`` makes ``.generate`` raise a
    :class:`BonServedError`, so ``run_static`` leaves the whole (single) chunk's
    goals uncovered and the run fails loudly rather than writing error
    placeholders into a store that then looks complete (the swallow-batch-
    exception landmine). ``False`` records failures as ``__SERVED_ERROR__`` text
    and counts them in ``n_errors`` WITHOUT raising — the mode the timing-only
    reference driver uses so one overflow cannot nuke a whole family's single
    chunk; a driver in that mode MUST surface ``n_errors > 0`` itself.

    A ``PromptOverWindow`` (served_window: the prompt does not fit the
    serving window even with ``max_tokens`` clamped) is filed under the SAME
    error policy — a family row must be a real generation, never an empty
    stand-in; the clampable majority is rescued by the single clamped retry
    inside :meth:`ServedDefense.generate_one_detailed`, and the unclampable
    rest is counted in ``n_prompt_over_window`` and never retried
    (deterministic: no retry can make the prompt fit)."""

    engine_tag = SERVED_ENGINE

    def __init__(self, served: "ServedDefense", *, loop: asyncio.AbstractEventLoop,
                 max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
                 raise_on_error: bool = True, max_retries: int = 2) -> None:
        if not isinstance(served, ServedDefense):
            raise BonServedError(
                "ServedStaticDefense needs a ServedDefense (built THROUGH the "
                "routing proxy), never a bare base_url or raw client — routing, "
                "the per-cell cache_salt and SI-prefix verification are the "
                "proxy's to own")
        self._sd = served
        self._loop = loop
        self.max_concurrency = max(1, int(max_concurrency))
        self.raise_on_error = bool(raise_on_error)
        self.max_retries = max(0, int(max_retries))
        # ENORMOUS so run_static makes ONE chunk per family (whole family in
        # flight at once, bounded by max_concurrency).
        self.batch_size = 10 ** 9
        self.n_errors = 0
        self.n_prompt_over_window = 0
        self.max_inflight = 0

    @property
    def cell_id(self) -> str:
        return self._sd.cell_id

    @property
    def system(self):
        return self._sd.system

    @property
    def n_target_forwards(self) -> int:
        return self._sd.n_target_forwards

    @property
    def serving_engine(self) -> str:
        """The wrapped defense's serving engine — what
        ``run_static`` stamps on every row (``served_target.serving_engine_of``
        reaches it through this wrapper)."""
        return self._sd.serving_engine

    def _fanout(self, prompts: list, *, max_new_tokens: int,
                decode: Optional[Mapping[str, Any]], seeds: list,
                labels: Optional[list] = None) -> list:
        """The ONE concurrent fan-out both surfaces share: one request per
        prompt (bounded by ``max_concurrency``), retries per the error policy,
        results in PROMPT order. Each slot is a :class:`ServedTurn`, or — only
        under ``raise_on_error=False`` — an ``Exception`` standing for the
        exhausted failure (the caller renders the placeholder). Greedy when
        ``decode`` is None (temperature 0.0), the preset otherwise; ``seeds``
        is aligned with ``prompts``. ``labels`` (request evidence):
        one ``(task, item_id, sample_idx)`` per prompt or None — each request
        is sent under its label's context (``request_evidence_client.labelled``,
        a ContextVar the ``to_thread`` transport worker inherits), so the
        recorder labels the exact bytes; with the hook off the context is inert."""
        inflight = {"n": 0, "max": 0}
        errors: list = []
        temperature = 0.0 if decode is None else float(decode["temperature"])
        label_list: list = [None] * len(prompts) if labels is None else list(labels)

        async def _run():
            sem = asyncio.Semaphore(self.max_concurrency)
            out = [None] * len(prompts)

            async def one(i, p, seed, lab):
                async with sem:
                    inflight["n"] += 1
                    inflight["max"] = max(inflight["max"], inflight["n"])
                    try:
                        last = None
                        for attempt in range(self.max_retries + 1):
                            try:
                                with rev.labelled_or_none(lab):
                                    turn = await self._sd.generate_one_detailed(
                                        p, sampling_seed=seed,
                                        max_new_tokens=max_new_tokens,
                                        temperature=temperature, decode=decode)
                            except sw.PromptOverWindow as exc:
                                # deterministic: the prompt does not fit; no
                                # retry can change that — file it, once
                                self.n_prompt_over_window += 1
                                last = exc
                                break
                            except Exception as exc:  # noqa: BLE001
                                last = exc
                                if attempt < self.max_retries:
                                    await asyncio.sleep(0.5 * (attempt + 1))
                                continue
                            out[i] = turn
                            return
                        # retries exhausted: record; raise/keep decided after gather
                        self.n_errors += 1
                        errors.append((i, last))
                        out[i] = last
                    finally:
                        inflight["n"] -= 1

            await asyncio.gather(*(asyncio.create_task(one(i, p, s, lab))
                                   for i, (p, s, lab) in enumerate(zip(prompts, seeds, label_list))))
            return out

        res = self._loop.run_until_complete(_run())
        self.max_inflight = max(self.max_inflight, inflight["max"])
        if errors and self.raise_on_error:
            _, first = errors[0]
            raise BonServedError(
                f"served static generation for cell {self.cell_id!r}: "
                f"{len(errors)}/{len(prompts)} request(s) failed after "
                f"{self.max_retries} retries (first: {type(first).__name__}: "
                f"{first}) — refusing to return a family with error "
                "placeholders; the failed goals are left UNCOVERED so a resume "
                "regenerates them (never a store that looks complete but isn't)")
        return res

    @staticmethod
    def _placeholder(exc: BaseException) -> str:
        return f"__SERVED_ERROR__ {type(exc).__name__}: {exc}"

    def generate(self, prompts, *, max_new_tokens: int = 512) -> list:
        """Issue one greedy generation per prompt CONCURRENTLY and return the
        responses in prompt order. Raises (default) or records-and-counts a
        per-request failure per ``raise_on_error``."""
        prompts = list(prompts)
        if not prompts:
            return []
        res = self._fanout(prompts, max_new_tokens=max_new_tokens, decode=None,
                           seeds=[0] * len(prompts))
        return [r.text if isinstance(r, ServedTurn) else self._placeholder(r)
                for r in res]

    def generate_detailed(self, prompts, *, max_new_tokens: int = 512,
                          decode: Optional[Mapping[str, Any]] = None,
                          seeds: Optional[Sequence[int]] = None,
                          labels: Optional[Sequence[Tuple[str, str, int]]] = None) -> list:
        """The DETAILED twin of :meth:`generate` (the surface ``run_static``
        drives under a sampler preset): one request per prompt CONCURRENTLY,
        returning :func:`turn_record` dicts in prompt order (text,
        finish_reason, usage, max_tokens, sampling_seed, clamp fields when a
        clamp fired). ``decode`` is the vendor preset (SAMPLER_KEYS only, sent
        with ``do_sample`` semantics — a non-zero temperature); when given,
        ``seeds`` (one per prompt) is REQUIRED — an unseeded sampled draw
        cannot be reproduced and is refused. Without ``decode`` the draws are
        greedy exactly like :meth:`generate` (seed 0 unless ``seeds`` is
        given). A failed request raises per ``raise_on_error`` (default), or
        yields a placeholder record with ``error`` set.

        ``labels`` (request evidence — the same contract as
        ``ServedChatTarget.generate_detailed``): one ``(task, item_id,
        sample_idx)`` per prompt, the X-Tinycap labels each request carries;
        ``run_static`` passes the store's own ``(static/<family>, goal_id,
        sample_idx)``. Aligned with ``prompts`` or refused. With the hook ON a
        record then carries ``{request_id, completion_id}`` (the process
        client's binding for the completion); off, nothing is added."""
        prompts = list(prompts)
        if not prompts:
            return []
        label_list: list = [None] * len(prompts) if labels is None else [tuple(x) for x in labels]
        if len(label_list) != len(prompts):
            raise BonServedError(
                f"generate_detailed: {len(label_list)} labels for {len(prompts)} "
                "prompts — one label per prompt, aligned")
        if decode is not None:
            sampler_preset(decode)  # refuses a bad preset before any request
            if seeds is None:
                raise BonServedError(
                    "generate_detailed(decode=...) needs one sampling seed per "
                    "prompt — a sampled draw without a recorded seed cannot be "
                    "reproduced; refusing")
        seed_list = ([0] * len(prompts) if seeds is None
                     else [int(s) for s in seeds])
        if len(seed_list) != len(prompts):
            raise BonServedError(
                f"generate_detailed: {len(seed_list)} seeds for "
                f"{len(prompts)} prompts — one seed per prompt, aligned")
        res = self._fanout(prompts, max_new_tokens=max_new_tokens,
                           decode=decode, seeds=seed_list, labels=label_list)
        out = []
        for r, seed in zip(res, seed_list):
            if isinstance(r, ServedTurn):
                rec = turn_record(r, sampling_seed=seed)
                # request-evidence binding (hook ON only): the recorder bound this
                # completion id to the attempt it recorded; ``{}`` when off / unknown
                if r.completion_id:
                    rec.update(rev.row_fields({"id": r.completion_id}))
                out.append(rec)
            else:
                out.append({"text": self._placeholder(r), "finish_reason": None,
                            "completion_tokens": None, "prompt_tokens": None,
                            "max_tokens": int(max_new_tokens),
                            "sampling_seed": seed,
                            "error": f"{type(r).__name__}: {r}"})
        return out


# --------------------------------------------------------------------------- #
# The concurrent BoN generation loop (resume/fsync preserved).                #
# --------------------------------------------------------------------------- #


def _assert_engine_consistent(jsonl: Path, *, serving_engine: Optional[str] = None) -> None:
    """Refuse to resume/append a served run onto a responses file that holds
    any non-served row (or the reverse) — never mix engines in one file, hence
    one figure. A torn tail line is tolerated (skipped), matching resume.

    ``serving_engine`` (None = not compared, the pre-landing call) ALSO holds
    every existing row's ``serving_engine`` to this run's: a row WITHOUT the
    field predates the native engine and was lens by construction, so a lens
    run resumes it and a native run refuses; a row naming the other engine
    refuses either way. Called before the try/finally in
    :func:`bon_generate_served`, so a refusal leaves the store AND its
    manifest byte-identical."""
    if serving_engine is not None and serving_engine not in SERVING_ENGINES:
        raise BonServedError(f"_assert_engine_consistent: serving_engine must be one of "
                             f"{list(SERVING_ENGINES)}, got {serving_engine!r}")
    if not jsonl.exists():
        return
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail from a SIGKILL mid-append
        if rec.get("engine") != SERVED_ENGINE:
            raise BonServedError(
                f"{jsonl} holds a row with engine="
                f"{rec.get('engine')!r} (not {SERVED_ENGINE!r}) — refusing to "
                "mix HF and vLLM-served responses in one file; they are not "
                "bit-identical and belong to different figures")
        if serving_engine is not None:
            got = rec.get(SERVING_ENGINE_FIELD)
            have = SERVING_ENGINE_LENS if got is None else got
            if have != serving_engine:
                raise BonServedError(
                    f"{jsonl} holds a row drawn on the {have!r} serving engine (unit "
                    f"{(rec.get('goal_id'), rec.get('sample_idx'), rec.get('trial'))}"
                    + ("; the row records no serving_engine, so it predates the native engine and "
                       "was lens by construction" if SERVING_ENGINE_FIELD not in rec else "")
                    + f") while this run serves on {serving_engine!r} — refusing to append: a store "
                    "stays single-engine (a lens row and a native row are two measurements, and the "
                    "passthrough identity cannot tell them apart); nothing was written. Run this "
                    "cell into a fresh dir")
        # one serving GEOMETRY per store as well (run_identity, review R1)
        from sisteer.capeval.serving.run_identity import geometry_stamp_problem
        problem = geometry_stamp_problem(rec)
        if problem is not None:
            raise BonServedError(
                f"{jsonl}: {problem} — refusing to append rows of this run's "
                "serving geometry onto a store of another; rerun into a fresh dir")


#: the FOUR fields that identify a current-format served row (the row writer records all of them since
#: 2026-09-10): the sampler sent, its preset label, the unit's seed and the rule that assigned it. A row is
#: LEGACY only when it carries NONE of them; a row carrying some but not all is torn (or another producer's) and
#: UNVERIFIABLE — never classified by the two it happens to lack (a rule-only row passed as historical under a
#: harness-default resume and was counted as covered)
ROW_IDENTITY_KEYS = ("decode", "bon_decode_preset", "sampling_seed", "seed_rule")


def _sent_sampler(decode: Optional[Mapping[str, Any]], decode_label: Optional[str] = None,
                  model: Any = None) -> dict:
    """The sampler a run sends — :func:`resolve_bon_sampler`'s ``decode`` (kept as the small entry point the
    drivers and tests read; there is no model-independent default behind it any more)."""
    return dict(resolve_bon_sampler(decode, decode_label, model).decode)


def _preset_label(decode: Optional[Mapping[str, Any]], decode_label: Optional[str], model: Any = None) -> str:
    """The label a run records as ``bon_decode_preset`` — :func:`resolve_bon_sampler`'s ``label``:
    :data:`BON_VENDOR_PRESET` on the default path, :data:`BON_HARNESS_DEFAULT` by explicit opt-in,
    :data:`BON_EXPLICIT_PRESET` (or the caller's own) for an explicit block."""
    return resolve_bon_sampler(decode, decode_label, model).label


def _validate_row(rec: Mapping[str, Any], *, jsonl: Path, want: Mapping[str, Any], want_label: str,
                  decode_seed_policy: Optional[str] = None, decode_seed_base: Optional[int] = None) -> str:
    """ONE response row against this run: returns ``"current"`` (a row of the current format whose
    sampler AND seed check out) or ``"legacy"`` (a pre-record row accepted under
    :data:`LEGACY_SEEDLESS_RULE`); raises :class:`BonServedError` otherwise. A row is covered only
    when its ``sampling_seed`` is the one :func:`bon_seed` assigns its (goal, sample, trial) under
    the ``seed_rule`` it names — a foreign producer's or a corrupted row with the same key,
    geometry and preset but another seed is a different sample identity and is never accepted; a
    current-format row without a seed or a rule is UNVERIFIABLE and refused, never assumed
    compatible. Legacy means NONE of :data:`ROW_IDENTITY_KEYS` on the row: a row that kept
    ``seed_rule`` or ``bon_decode_preset`` while losing ``decode`` and ``sampling_seed`` is torn,
    not historical.

    Under the opt-in ``replicate_v1`` decode-seed policy the row's
    ``seed_rule`` is ``bon_seed_v1+decode:replicate_v1`` and the row ALSO carries :data:`REPLICATE_ROW_KEYS`;
    its seeds are recomputed under the row's OWN recorded policy and base — ``augment_seed`` must be
    :func:`bon_seed` (the frozen variant identity) and ``sampling_seed`` :func:`bon_decode_seed_v1` at the
    row's base — and a run under ANOTHER base or policy refuses the store (never a relabel). A
    ``bon_seed_v1`` run (the default, ``decode_seed_policy`` None) validates exactly as before."""
    policy, run_base = check_decode_seed_policy(decode_seed_policy, decode_seed_base)
    want_rule = seed_rule_for_policy(policy)
    present = [k for k in ROW_IDENTITY_KEYS if k in rec]
    if not present:
        # the explicit legacy rule: drawn at the harness default (the grandfathered T=1 row) before the record
        # existed — NONE of the four identity fields is on the row (a row with any of them is a current-format
        # row, whole or torn). Only a run that NAMES that row (BON_HARNESS_DEFAULT) may resume such a store: the
        # vendor-preset default (2026-09-17) is a different sampler and would mix two in one store.
        if want_label != BON_HARNESS_DEFAULT:
            raise BonServedError(
                f"{jsonl} holds LEGACY rows (no sampler, no preset label, no seed and no seed rule recorded; drawn at "
                f"the harness default before 2026-09-10) and this run asks for the preset {want} ({want_label}) — "
                f"refusing to relabel them or to mix samplers in one store; regenerate into a fresh dir, or name "
                f"{BON_HARNESS_DEFAULT!r} to finish the store under the sampler it was drawn under")
        if policy != DECODE_SEED_POLICY_BON:
            raise BonServedError(
                f"{jsonl} holds LEGACY rows (drawn under the frozen bon_seed before any seed record existed) and this run "
                f"asks for the decode-seed policy {policy} (base {run_base}) — a different sample identity; refusing to "
                "relabel them. Regenerate into a fresh dir")
        return "legacy"
    if len(present) != len(ROW_IDENTITY_KEYS):
        raise BonServedError(
            f"{jsonl}: row {(rec.get('goal_id'), rec.get('sample_idx'), rec.get('trial'))} carries {present} but not "
            "all of decode / bon_decode_preset / sampling_seed / seed_rule — an UNVERIFIABLE row (torn, or another "
            "producer's), never legacy; its coverage cannot be accepted. Regenerate into a fresh dir")
    if dict(rec["decode"]) != dict(want) or rec.get("bon_decode_preset") != want_label:
        raise BonServedError(
            f"{jsonl}: the store's rows were drawn under a different sampler "
            f"({rec.get('bon_decode_preset')}: {rec['decode']}) than this run's ({want_label}: {want}) — "
            "refusing to resume or relabel; a mixture is never a store. Use a fresh dir")
    if rec["seed_rule"] != want_rule:
        raise BonServedError(
            f"{jsonl}: row {(rec.get('goal_id'), rec.get('sample_idx'), rec.get('trial'))} names the seed rule "
            f"{rec['seed_rule']!r}, this harness implements {want_rule!r} (decode-seed policy {policy}) — another "
            "producer's sample identity, or a store drawn under another decode-seed policy; refusing its coverage")
    try:
        gid, idx, trial = rec["goal_id"], int(rec["sample_idx"]), int(rec["trial"])
        seed = int(rec["sampling_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BonServedError(f"{jsonl}: row with an unreadable (goal_id, sample_idx, trial, sampling_seed): {exc}")
    if policy == DECODE_SEED_POLICY_BON:
        expect = bon_seed(gid, idx, trial)
        if seed != expect:
            raise BonServedError(
                f"{jsonl}: row ({gid}, {idx}, {trial}) records sampling_seed {seed} but {BON_SEED_RULE} assigns "
                f"{expect} — a different sample identity (a foreign producer's row, or a corrupted one) under the same "
                "key, geometry and preset; refusing its coverage. Regenerate into a fresh dir")
        return "current"
    # replicate_v1: the row's OWN recorded policy fields, all of them, then both seeds recomputed under them
    missing = [k for k in REPLICATE_ROW_KEYS if k not in rec]
    if missing:
        raise BonServedError(
            f"{jsonl}: row ({gid}, {idx}, {trial}) names {want_rule!r} but carries no {missing} — an UNVERIFIABLE "
            "replicate row (torn, or another producer's); its coverage cannot be accepted. Regenerate into a fresh dir")
    try:
        row_policy, row_base, augment_seed = str(rec["decode_seed_policy"]), int(rec["decode_seed_base"]), int(rec["augment_seed"])
    except (TypeError, ValueError) as exc:
        raise BonServedError(f"{jsonl}: row ({gid}, {idx}, {trial}) with unreadable replicate fields: {exc}")
    if row_policy != policy:
        raise BonServedError(
            f"{jsonl}: row ({gid}, {idx}, {trial}) records decode_seed_policy {row_policy!r} under seed_rule {want_rule!r} — "
            "torn; refusing its coverage")
    if row_base != run_base:
        raise BonServedError(
            f"{jsonl}: the store's rows were drawn under decode_seed_base {row_base} and this run asks for {run_base} — "
            "a different replicate's sample identity; refusing to resume or relabel (use the view's own base, or a fresh dir)")
    if augment_seed != bon_seed(gid, idx, trial):
        raise BonServedError(
            f"{jsonl}: row ({gid}, {idx}, {trial}) records augment_seed {augment_seed} but {BON_SEED_RULE} assigns "
            f"{bon_seed(gid, idx, trial)} — a different VARIANT identity; refusing its coverage")
    expect = bon_decode_seed_v1(gid, idx, trial, row_base)
    if seed != expect:
        raise BonServedError(
            f"{jsonl}: row ({gid}, {idx}, {trial}) records sampling_seed {seed} but {want_rule} at decode_seed_base "
            f"{row_base} assigns {expect} — a different sample identity (a foreign producer's row, or a corrupted one) "
            "under the same key, geometry and preset; refusing its coverage. Regenerate into a fresh dir")
    return "current"


def _validated_rows(jsonl: Path, decode: Optional[Mapping[str, Any]] = None, decode_label: Optional[str] = None,
                    model: Any = None, *, sampler: Optional[BonSampler] = None,
                    decode_seed_policy: Optional[str] = None, decode_seed_base: Optional[int] = None):
    """Every parseable row of the store validated (:func:`_validate_row`) against THIS run's sampler — the
    resolved :class:`BonSampler` when handed in, else :func:`resolve_bon_sampler` of the caller's
    ``(decode, decode_label, model)`` — and its decode-seed policy (qwen-base-seed-replicates; ``None`` =
    ``bon_seed_v1``), yielding ``(kind, rec)``; a torn tail is skipped exactly as resume does
    (:func:`~sisteer.eval.jailbreak_artifacts.iter_jsonl_rows`)."""
    if sampler is None:
        sampler = resolve_bon_sampler(decode, decode_label, model)
    for rec in iter_jsonl_rows(jsonl):
        yield _validate_row(rec, jsonl=jsonl, want=sampler.decode, want_label=sampler.label,
                            decode_seed_policy=decode_seed_policy, decode_seed_base=decode_seed_base), rec


def _assert_sampler_consistent(jsonl: Path, decode: Optional[Mapping[str, Any]] = None,
                               decode_label: Optional[str] = None, model: Any = None, *,
                               sampler: Optional[BonSampler] = None,
                               decode_seed_policy: Optional[str] = None, decode_seed_base: Optional[int] = None) -> int:
    """A store's rows were drawn under ONE sampler and ONE seed rule, and a run may only resume it
    under the same ones. Rows carry ``decode`` + ``bon_decode_preset`` + ``sampling_seed`` +
    ``seed_rule`` since 2026-09-10 and every row's seed must reproduce from
    ``bon_seed(goal_id, sample_idx, trial)``; a row without any record is LEGACY
    (:data:`LEGACY_SEEDLESS_RULE`) and is never relabelled: a preset run over a legacy store
    refuses, the harness default may resume it and the manifest counts the rows under that explicit
    rule. Returns the number of legacy rows. Called BEFORE any manifest write, so a refusal leaves
    the store exactly as it was."""
    return sum(1 for kind, _ in _validated_rows(jsonl, decode, decode_label, model, sampler=sampler,
                                                 decode_seed_policy=decode_seed_policy, decode_seed_base=decode_seed_base)
               if kind == "legacy")


def _load_covered_validated(jsonl: Path, decode: Optional[Mapping[str, Any]] = None,
                            decode_label: Optional[str] = None, model: Any = None, *,
                            sampler: Optional[BonSampler] = None,
                            decode_seed_policy: Optional[str] = None, decode_seed_base: Optional[int] = None) -> set:
    """The covered (goal_id, sample_idx, trial) units — ONLY rows whose sampler and seed were
    validated for this run (or legacy rows under the explicit rule) count; nothing is accepted
    before its ``bon_seed`` is recomputed."""
    return {_resume_key(rec) for _, rec in _validated_rows(jsonl, decode, decode_label, model, sampler=sampler,
                                                            decode_seed_policy=decode_seed_policy, decode_seed_base=decode_seed_base)}


def _repair_torn_tail(jsonl: Path) -> None:
    """If the file ends without a trailing newline — a SIGKILL caught a
    previous append mid-write — add one BEFORE resuming. The torn partial line
    stays unparseable and is skipped by :func:`_load_covered` (so its unit is
    uncovered and regenerates), but it can no longer swallow the first resumed
    row by gluing onto it. A strict strengthening of the legacy per-append
    contract: it changes no covered unit, only prevents a second corruption,
    so resume recovers in ONE pass instead of several. No-op on a clean file."""
    if not jsonl.exists() or jsonl.stat().st_size == 0:
        return
    with jsonl.open("rb") as fb:
        fb.seek(-1, os.SEEK_END)
        if fb.read(1) != b"\n":
            with jsonl.open("ab") as fa:
                fa.write(b"\n")
                fa.flush()
                os.fsync(fa.fileno())


async def _bon_generate_served_async(
    defense: ServedDefense,
    goals,
    *,
    jsonl: Path,
    n_max: int,
    sigma: float,
    trial: int,
    max_new_tokens: int,
    max_concurrency: int,
    errors_jsonl: Optional[Path] = None,
    sampler: BonSampler,
    decode_seed_policy: Optional[str] = None,
    decode_seed_base: Optional[int] = None,
    request_headers_fn: Optional[Callable[[str, int, int], Mapping[str, str]]] = None,
) -> dict:
    """The concurrent loop. Returns ``{"n_work", "n_written",
    "n_prompt_over_window"}``. ``sampler`` (:func:`resolve_bon_sampler`) is THE
    sampler every draw is issued under — the served model's vendor preset on
    the default path (2026-09-17), the grandfathered T=1 row by explicit opt-in,
    or an explicit block — and is recorded on every response row as ``decode``
    (the request-body keys ACTUALLY sent, SAMPLER_KEYS only), ``bon_decode_preset``
    (its label) and ``decode_policy_id`` (the registry row, None for an explicit
    block), so a store says how it was drawn instead of leaving the sampler
    implicit. A unit whose prompt cannot fit
    (``PromptOverWindow``) writes NO responses row — its slot stays uncovered
    for a resume and ``bon_curves`` reports the goal incomplete — and files a
    typed error row in ``errors_jsonl`` (``BON_ERRORS_FILE`` beside the
    responses by default).

    ``decode_seed_policy`` / ``decode_seed_base``: ``None`` =
    ``bon_seed_v1``, the unit's :func:`bon_seed` drives both the augmentation and the decode and every
    row is byte-identical to before; ``replicate_v1`` keeps the augmentation on :func:`bon_seed` (the
    frozen variant bytes) and draws the DECODE at :func:`bon_decode_seed_v1` over the base, recording
    ``seed_rule`` ``bon_seed_v1+decode:replicate_v1`` and :data:`REPLICATE_ROW_KEYS` on every row.

    ``request_headers_fn`` (the request-evidence seam): ``(goal_id,
    sample_idx, trial) -> headers`` of ONE ATTEMPT of that draw — the two tinycap labels and a FRESH
    ``X-Tinycap-Request-Id``; see ``generate_one_detailed(attempt_headers_fn=)`` for the per-send contract. The row
    then carries ``request_id`` (the served attempt's) and ``completion_id`` (``resp["id"]``). None
    (every caller today) = no headers, no extra keys: rows and bytes exactly as before."""
    policy, seed_base = check_decode_seed_policy(decode_seed_policy, decode_seed_base)
    seed_rule = seed_rule_for_policy(policy)
    _repair_torn_tail(jsonl)
    errors_jsonl = errors_jsonl if errors_jsonl is not None \
        else jsonl.parent / BON_ERRORS_FILE
    sent_sampler = dict(sampler.decode)
    # coverage is accepted per row only after its sampler AND its seed check out
    covered = _load_covered_validated(jsonl, sampler=sampler, decode_seed_policy=policy, decode_seed_base=seed_base)
    # deterministic goal-major, sample-order work list of the units NOT already
    # covered — the SAME variant identity as every other engine (augment seeded
    # by bon_seed(goal, sample, trial), under EVERY decode-seed policy), so
    # sample_idx still maps to the right augmentation and the ASR@N prefix
    # property holds; the decode seed is the policy's (== the augment seed under
    # bon_seed_v1, the replicate's derivation under replicate_v1).
    work: list = []  # (goal_id, sample_idx, variant, seed, augment_seed)
    for g in goals:
        for sample_idx in range(n_max):
            if (g.goal_id, sample_idx, trial) in covered:
                continue
            augment_seed = bon_seed(g.goal_id, sample_idx, trial)
            seed = decode_seed_for(g.goal_id, sample_idx, trial, policy=policy, decode_seed_base=seed_base)
            work.append((g.goal_id, sample_idx,
                         augment(g.text, sigma, random.Random(augment_seed)),
                         seed, augment_seed))
    # the WORKLOAD this run was asked to cover and how much of it a durable row already covers: `units.n_planned` /
    # `units.n_done` are what `audit_common.read_bon_store` keys completeness on (a surviving-row count equals
    # len(rows) by construction, so a store with a `PromptOverWindow` slot read COMPLETE).
    # n_done counts planned (goal, sample, trial) slots with a validated row: the ones covered before this run plus the
    # ones written now; an over-window slot writes no row and stays UNCOVERED (n_uncovered > 0 -> the reader refuses).
    n_planned = len(goals) * int(n_max)
    n_covered_before = n_planned - len(work)
    counts = {"n_planned": n_planned, "n_work": len(work), "n_written": 0, "n_prompt_over_window": 0,
              "n_done": n_covered_before, "n_uncovered": len(work)}

    def _finish() -> dict:
        counts["n_done"] = n_covered_before + int(counts["n_written"])
        counts["n_uncovered"] = n_planned - int(counts["n_done"])                       # slots without a response row (legacy meaning)
        # a prompt_over_window slot is a DECIDED no-fit (typed row in bon_errors.jsonl; deterministic, never retried into a fit):
        # the store is COMPLETE when every planned slot is decided — covered or no-fit — and UNDECIDED slots are the ones a
        # resume must still draw (bon_climb.store_completion, the one predicate the writer and the climb readers share)
        counts["n_nofit"] = int(counts.get("n_nofit_proven", 0))                    # PROVEN no-fits only
        counts["n_undecided"] = n_planned - int(counts["n_done"]) - int(counts["n_nofit"])
        counts["decided"] = counts["n_undecided"] == 0
        return counts

    if not work:
        return _finish()

    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    write_lock = asyncio.Lock()
    with jsonl.open("a") as f, errors_jsonl.open("a") as ferr:
        async def run_one(gid, sample_idx, variant, seed, augment_seed):
            async with sem:
                # THIS unit's attempt-headers hook (None today; see generate_one_detailed)
                attempt_headers_fn = ((lambda: request_headers_fn(gid, sample_idx, trial))
                                      if request_headers_fn is not None else None)
                try:
                    turn = await defense.generate_one_detailed(
                        variant, sampling_seed=seed, max_new_tokens=max_new_tokens,
                        decode=sent_sampler,
                        **({"attempt_headers_fn": attempt_headers_fn} if attempt_headers_fn is not None else {}))
                except sw.PromptOverWindow as err:
                    # a TYPED error, never a response: no responses row (the
                    # slot stays UNCOVERED — a resume retries it, bon_curves
                    # refuses the goal as incomplete), one ledger row.
                    counts["n_prompt_over_window"] += 1
                    from .bon_climb import classify_over_window, NOFIT_PROVEN   # the one classifier the climb readers use
                    _rec = err.record(); _kind = classify_over_window(_rec)
                    counts["n_nofit_proven"] = int(counts.get("n_nofit_proven", 0)) + (1 if _kind == NOFIT_PROVEN else 0)
                    counts["n_over_window_undecided"] = int(counts.get("n_over_window_undecided", 0)) + (0 if _kind == NOFIT_PROVEN else 1)
                    erow = json.dumps({
                        "goal_id": gid, "sample_idx": sample_idx, "trial": trial,
                        "engine": SERVED_ENGINE, SERVING_ENGINE_FIELD: defense.serving_engine,
                        "cell_id": defense.cell_id,
                        **_geometry_stamp(),
                        **_rec, "nofit_kind": _kind,
                    })
                    async with write_lock:
                        ferr.write(erow + "\n")
                        ferr.flush()
                        os.fsync(ferr.fileno())
                    return
            rec = {
                "goal_id": gid, "sample_idx": sample_idx, "trial": trial,
                "variant": variant, "response": turn.text,
                # the engine tag and, beside it, THE SERVING ENGINE: lens /
                # lora_native, so a passthrough row of either era is self-describing
                "engine": SERVED_ENGINE, SERVING_ENGINE_FIELD: defense.serving_engine,
                "cell_id": defense.cell_id,
                "bon_max_concurrency": int(max_concurrency),
                "finish_reason": turn.finish_reason,
                # the sampler ACTUALLY sent (the seed is the unit's bon_seed): a
                # store no longer leaves its temperature implicit; the registry
                # row it came from rides beside it (None for an explicit block)
                "decode": dict(sent_sampler),
                "bon_decode_preset": sampler.label,
                "decode_policy_id": sampler.policy_id,
                # the unit's seed and the rule that assigned it: a resume recomputes bon_seed(goal,
                # sample, trial) and refuses a row that does not reproduce it
                "sampling_seed": int(seed),
                "seed_rule": seed_rule,
                # replicate_v1 ONLY (qwen-base-seed-replicates): the policy, its base and the frozen
                # augmentation seed beside the decode seed — a bon_seed_v1 row carries none of these
                **({"decode_seed_policy": policy, "decode_seed_base": int(seed_base), "augment_seed": int(augment_seed)}
                   if policy == DECODE_SEED_POLICY_REPLICATE else {}),
                # the client-minted request id of the attempt that PRODUCED the response (the
                # retry's when the clamp fired) + the engine's completion id, ONLY when a request-headers hook is in play —
                # a hook-less run writes neither key (rows byte-identical)
                **({"request_id": turn.request_id, "completion_id": turn.completion_id} if request_headers_fn is not None else {}),
                **_geometry_stamp(),
            }
            # the window clamp's per-call facts (served_window): the original
            # max_tokens when the once-only clamp fired; the window
            # disagreement when the launcher's provenance was off
            if turn.max_tokens_clamped_from is not None:
                rec["max_tokens_clamped_from"] = turn.max_tokens_clamped_from
                rec["prompt_tokens_source"] = turn.prompt_tokens_source
            if turn.window_mismatch is not None:
                rec["window_mismatch"] = turn.window_mismatch
            row = json.dumps(rec)
            # DURABILITY BEFORE COVERAGE: one writer at a time (no interleaved
            # lines), each row flushed+fsync'd before this unit is durable.
            # No await inside this block, so cancellation cannot tear a row.
            async with write_lock:
                f.write(row + "\n")
                f.flush()
                os.fsync(f.fileno())
                counts["n_written"] += 1

        tasks = [asyncio.create_task(run_one(*item)) for item in work]
        await asyncio.gather(*tasks)
    return _finish()


def bon_generate_served(
    defense: ServedDefense,
    goals,
    *,
    out_dir: Path,
    routes: Sequence[RouteEntry],
    n_max: int = DEFAULT_N_MAX,
    sigma: float = DEFAULT_SIGMA,
    trial: int = 0,
    max_new_tokens: int = 512,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    steering_arms: Sequence[str] = (),
    steering_probe: Optional[SteeringProbe] = None,
    decode: Optional[Mapping[str, Any]] = None,
    decode_label: Optional[str] = None,
    model: Any = None,
    decode_seed_policy: Optional[str] = None,
    decode_seed_base: Optional[int] = None,
    decode_policy: Optional[Mapping[str, Any]] = None,
    request_headers_fn: Optional[Callable[[str, int, int], Mapping[str, str]]] = None,
) -> Path:
    """Draw ``n_max`` augmented variants per goal and generate one response
    each CONCURRENTLY against the steered vLLM server (through ``defense``'s
    proxy transport), persisting every response to
    ``out_dir/bon_responses.jsonl`` and resuming per (goal, sample, trial).
    Returns the JSONL path.

    THE SAMPLER (:func:`resolve_bon_sampler`): by
    default the SERVED MODEL'S VENDOR PRESET — ``model`` names it (the config's
    ``model``, a legacy key, or the routing table's ``base_model``) and the
    registry's ``<model>|bon|off|vendor_preset`` row supplies the request-body
    keys, Qwen's ``presence_penalty`` 1.5 included. ``decode_label``
    :data:`BON_HARNESS_DEFAULT` names the superseded grandfathered T=1 row
    instead (to finish a store drawn under it — never mixed, never relabelled);
    an explicit ``decode`` block is the caller's own sampler
    (:data:`BON_EXPLICIT_PRESET`). No model and no block refuses. The per-draw
    seed rule and the augmentation are untouched by the sampler: the attack's
    diversity is the augmentation, not the temperature.

    Before any generation, two preconditions are ENFORCED (never skipped):

    1. :func:`assert_cell_salt_isolation` over ``routes`` — the run's cells can
       never share a cached prefix (the correctness boundary). ``routes`` must
       be the FULL set the run serves, not just this cell's, so the proof is
       mutual.
    2. If this cell is steered (``steering_arms`` non-empty), a
       :func:`assert_steering_bites` probe through the SAME serving path proves
       the arms bite; a steered cell with no probe REFUSES rather than trust an
       unproven server. Unhooked cells (none/prompted) carry no arm — the
       proxy's per-request SI-prefix verification is their analogous guard.

    ``decode_seed_policy`` / ``decode_seed_base``: the opt-in decode-seed policy
    (:func:`_bon_generate_served_async`); the (policy, base) pair is validated
    BEFORE any gate runs, so a misconfigured run touches nothing.

    ``decode_policy``: the REGISTRY row's identity the ``vendor`` preset was
    resolved from (``target_decode.VendorRow.record()``), written into the manifest as ``decode_policy``; None (every
    other preset) adds no key. ``request_headers_fn``: the request-evidence seam
    (:func:`_bon_generate_served_async`); None = bytes exactly as before.

    Grading is NOT done here (JB6: no local pre-filter) — every response is
    Batch-graded later by :func:`sisteer.eval.attacks.bon.bon_curves`."""
    try:
        decode_seed_policy, decode_seed_base = check_decode_seed_policy(decode_seed_policy, decode_seed_base)
    except ValueError as e:
        raise BonServedError(f"bon decode-seed policy: {e}") from None
    assert_cell_salt_isolation(routes)
    if steering_arms:
        if steering_probe is None:
            raise BonServedError(
                f"cell {defense.cell_id!r} is steered (arms "
                f"{list(steering_arms)}) but no steering_probe was given — "
                "refusing to serve a steered run with no generation-time proof "
                "that steering bites (the silently-unhooked failure mode)")
        assert_steering_bites(steering_arms, steering_probe)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "bon_responses.jsonl"
    # the engine-tag AND serving-engine gate: before the try/finally, so a
    # cross-engine attempt leaves the store and its manifest byte-identical
    _assert_engine_consistent(jsonl, serving_engine=getattr(defense, "serving_engine", None))
    # the sampler is resolved ONCE, before any gate: a refusal (no model, an unknown label, a block outside
    # SAMPLER_KEYS) leaves the store and its manifest byte-identical
    sampler = resolve_bon_sampler(decode, decode_label, model)
    # the sampler identity gate runs BEFORE the try/finally: a refusal must leave the manifest alone
    n_legacy = _assert_sampler_consistent(jsonl, sampler=sampler,
                                          decode_seed_policy=decode_seed_policy, decode_seed_base=decode_seed_base)

    counts: dict = {}
    try:
        counts = asyncio.run(_bon_generate_served_async(
            defense, goals, jsonl=jsonl, n_max=n_max, sigma=sigma, trial=trial,
            max_new_tokens=max_new_tokens, max_concurrency=max_concurrency,
            sampler=sampler,
            decode_seed_policy=decode_seed_policy, decode_seed_base=decode_seed_base,
            request_headers_fn=request_headers_fn))
        # `units` then carries n_planned / n_done / n_uncovered (see _bon_generate_served_async): the keys
        # `audit_common.read_bon_store` keys completeness on (an earlier writer carried only n_work / n_written /
        # n_prompt_over_window and every real store read as "the run did not finish"). When the
        # run RAISED, counts stay {} and the manifest carries none of them: the run did not finish, and it says so.
    finally:
        # the BoN twin of served_target.json, written AFTER generation so the
        # clamp counters are the real ones (never a pre-generation zero)
        write_bon_served_manifest(out_dir, defense, counts=counts,
                                  max_concurrency=max_concurrency,
                                  sampler=sampler,
                                  n_legacy_unrecorded_rows=n_legacy,
                                  decode_seed_policy=decode_seed_policy, decode_seed_base=decode_seed_base,
                                  decode_policy=decode_policy)
    n_over = int(counts.get("n_prompt_over_window", 0))
    if n_over:
        print(f"[bon-served] {defense.cell_id}: {n_over} sample(s) NOT generated — "
              f"prompt_over_window (typed error rows in {out_dir / BON_ERRORS_FILE}); "
              "their slots are left UNCOVERED, so bon_curves will refuse those goals "
              "as incomplete and a resume retries them (regenerate at a wider window)",
              flush=True)
    return jsonl


def write_bon_served_manifest(out_dir: Path, defense: ServedDefense, *,
                              counts: Mapping[str, Any], max_concurrency: int,
                              sampler: Optional[BonSampler] = None,
                              decode: Optional[Mapping[str, Any]] = None,
                              decode_label: Optional[str] = None,
                              model: Any = None,
                              n_legacy_unrecorded_rows: int = 0,
                              decode_seed_policy: Optional[str] = None,
                              decode_seed_base: Optional[int] = None,
                              decode_policy: Optional[Mapping[str, Any]] = None) -> Path:
    """``bon_served_manifest.json``: the served-BoN run's clamp / error
    counters (``ServedDefense.clamp_report``) + the loop's unit counts, so a
    reader knows how many slots were left uncovered and why. ``sampler`` is the
    run's (:func:`resolve_bon_sampler`; resolved here from ``decode`` /
    ``decode_label`` / ``model`` when not handed in). The decode-seed
    POLICY the run drew under is recorded on every manifest (``decode_seed_policy``
    ``bon_seed_v1`` with a null base for every run before the policy existed)."""
    policy, base = check_decode_seed_policy(decode_seed_policy, decode_seed_base)
    out_dir = Path(out_dir)
    if sampler is None:
        sampler = resolve_bon_sampler(decode, decode_label, model)
    p = out_dir / BON_SERVED_MANIFEST
    engine = getattr(defense, "serving_engine", None)
    p.write_text(json.dumps({
        "engine": SERVED_ENGINE,
        # THE SERVING ENGINE the store was drawn on: a real ServedDefense always
        # states it; a duck-typed stand-in without it writes no key (bytes unchanged)
        **({SERVING_ENGINE_FIELD: engine} if engine is not None else {}),
        "cell_id": defense.cell_id,
        **_geometry_stamp(),
        "virtual_model": defense.virtual_model,
        "bon_max_concurrency": int(max_concurrency),
        "window_clamp": defense.clamp_report(),
        # the sampler the run drew under (every row carries the same block) and the registry row it came from
        "decode": dict(sampler.decode),
        "bon_decode_preset": sampler.label,
        "decode_policy_id": sampler.policy_id,
        # rows drawn before the sampler was recorded (harness default): counted, never relabelled
        "n_legacy_unrecorded_rows": int(n_legacy_unrecorded_rows),
        # the seed policy every current row is validated against, and the explicit rule the legacy
        # (seedless, samplerless) rows were accepted under — null when the store holds none
        "seed_rule": seed_rule_for_policy(policy),
        # qwen-base-seed-replicates: the decode-seed policy (bon_seed_v1 = the frozen rule alone, base null;
        # replicate_v1 = the frozen augmentation + the replicate's decode base)
        "decode_seed_policy": policy,
        "decode_seed_base": base,
        # qwen-seedrep-native-battery: the registry row the `vendor` preset was resolved from (its identity, the
        # numbers ride `decode` above); absent under every other preset, so those manifests are byte-identical
        **({"decode_policy": dict(decode_policy)} if decode_policy is not None else {}),
        "n_legacy_seedless_rows": int(n_legacy_unrecorded_rows),
        "legacy_rows_rule": (LEGACY_SEEDLESS_RULE if n_legacy_unrecorded_rows else None),
        "units": dict(counts),
        "errors_file": BON_ERRORS_FILE,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, default=str))
    return p
