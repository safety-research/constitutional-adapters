"""Harness-facing native emission — STDLIB ONLY, importable under ``python3 -S``.

The harness writers (``capeval.store.CellStore``, the native serving standup, the judging drivers,
the fit sidecar writers, the Modal launchers) call the small hooks in this module; every hook is
ADDITIVE and behaviour-identical when emission is off: no store byte, path or exit code changes,
and a hook that fails prints ONE ``[registry]`` line on stderr and returns — a lost record must
never cost a run (the harvester's pull and the adapters are the backstop).

THE SWITCH (one rule, applied by every hook through :func:`emission_enabled`):

1. ``SISTEER_REGISTRY_DISABLED=1`` → OFF, whatever else is set (the byte-identity tests use it)
2. an explicit ``registry_emit=`` keyword on the hook → that value
3. ``REGISTRY_EMIT=1`` → ON (spool from :func:`sisteer.registry.emit.spool_root`, else
   ``~/registry/records``); ``REGISTRY_EMIT=0`` → OFF
4. otherwise ON iff a spool resolves: ``$SISTEER_REGISTRY_SPOOL`` is set, or ``$SHARED/registry``
   exists on a cluster, or ``/vol/registry`` on a Modal container, or ``~/registry`` exists on the
   grading host — and OFF otherwise (an unprovisioned host emits nothing and says so once).

RUN IDENTITY. A store hook cannot invent the registry-grain identity (``eval_id``, ``item_set_id``,
``instrument_id`` and the other required ``run`` columns — an eval is never
named without its version). It is supplied by the driver (``registry_identity=`` on
``runtime.resolve_cell``), or read from the planned-run file the submit wrapper stored beside the
job (``$SISTEER_REGISTRY_RUN_IDENTITY`` → ``$SHARED/registry/pending/<jobid>.json``, exported by
``s2_init`` / ``f22_init``). Without one the hook stays inert and prints what it would need. When the
identity carries a ``run_id`` (a registered job), the store's records join THAT run; otherwise the id
is minted from the store's own key through :func:`sisteer.registry.identity.run_id`, exactly as the
adapter that later reads the same store will mint it.

THE DEFAULT (2026-09-17, every producer on main emits without opting in): a
``CellStore`` built with no recorder resolves one itself through :func:`default_store_recorder` — the
identity from the job's pending file (which the job now writes ITSELF at start: ``emit_cli register
--from-config``, one run per declared eval x object, each with its ROUTING facts), the switch above
unchanged. A job-start pending file carries a ``runs`` list; :func:`identity_from_env` turns it into the
``{route: identity}`` map — keyed ``"<object>|<task spelling>"`` for every producer spelling of the
run's eval, plus the bare spelling when only one registered run carries it — and the recorder routes each
row by ``(spec.arm_name, row.task)``, then by the task alone (with or without its ``<family>/`` prefix).
A row no registered run claims is counted and named once, never filed onto a run it does not belong to.
"""
from __future__ import annotations

import atexit
import dataclasses
import functools
import hashlib
import json
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from . import derive as D
from . import decode_evidence as DE
from . import emit, identity
from .identity import run_dir_name
from . import records as R
from .records import epoch_ms_to_utc, utc_now

EMIT_ENV = "REGISTRY_EMIT"
RUN_IDENTITY_ENV = "SISTEER_REGISTRY_RUN_IDENTITY"
TAG = "[registry]"

#: default declared degeneration measures: every store hook writes a ``not_measured`` row per
#: (measure, scope) it declares but did not compute, so absence is a state, never a missing row.
#: the two at-cap estimands are two measures (the canonical re-tokenized reading and the producer's
#: finish-reason declaration), never one column; the union classifier and the first-512 loop detector beside them.
#: the producer's at-cap declaration measure (a finish_reason in the at-cap set): the ONE spelling the judging drivers' measured rows
#: use, equal to adapters._items_common.AT_CAP_FINISH_REASON (tests/test_padding_registration.py asserts the two agree)
AT_CAP_FINISH_REASON = "at_token_cap_finish_reason"
DEFAULT_DECLARED_DEGENERATION: Tuple[Tuple[str, str], ...] = (
    ("union5", "all"), ("at_token_cap_retokenized", "all"), (AT_CAP_FINISH_REASON, "all"),
    ("loop_first512", "first_512_tokens"))
_AGGREGATION_OF_MEASURE = {"union5": "union", "defended_degenerate": "union"}
DEGENERATION_DEFINITION_VERSION = "1"
#: fr5 (2a): the metric id of the null-valued ``item_result`` a judged route files per GENERATED item before any judge
#: rules -- the item exists, its value does not yet (``judgment_id ''``); the judge's rows carry the real metrics later
GENERATED_METRIC_ID = "generated"
#: fr5 (2b): a producer's whole-run aggregate row (``item_id == "__aggregate__"``, or a metric spelled ``*_aggregate``:
#: parity's ``acc_aggregate``, the judged ``xst_refused_bounds_aggregate`` ...) is a ``run_metric``, never an ``item_result`` --
#: filed as an item it made the mmlu shard union 201 items against the plan's 200 and every mmlu run REFUSED admission. The
#: same rule and the same registry-grain spelling as the cellstore ADAPTER's (``metric_id`` = the spelling minus ``_aggregate``,
#: ``producer_metric_spelling`` = the producer's own).
AGGREGATE_ITEM_ID = "__aggregate__"
AGGREGATE_SUFFIX = "_aggregate"

#: capeval ``engine`` spellings → registry ``serving_path`` (SERVING_PATHS)
ENGINE_TO_SERVING_PATH = {"hf": "hf", "lora_native": "lora_native", "lens": "lens", "vllm": "passthrough",
                          "passthrough": "passthrough"}
#: sidecar ``positions_at_fit`` spellings → registry ``fit_positions`` (FIT_POSITIONS)
POSITIONS_AT_FIT_TO_FIT_POSITIONS = {"all": "all", "response": "response_incl_assistant_header",
                                     "last": "last_token_of_appended_assistant_header",
                                     "completion_mean": "completion_mean"}

_UNVERIFIED = "unverified"
_SHA1_LEN = 40
_SHA256_LEN = 64

#: The fold's non-statement marker (``reducer.UNSTATED_COLUMNS_KEY``, spelled here because this module imports no reducer; pinned equal by
#: the tests): ``extra.unstated_columns = [<column>, ...]`` on a ``run`` record names columns the record CARRIES but does not STATE -- the
#: fold applies them where no row exists yet (or the standing value is null, or itself came from a marked statement) and skips them over
#: a value an unmarked statement supplied, that value's provenance entry standing with it.
UNSTATED_COLUMNS_KEY = "unstated_columns"
#: The ``run`` columns a STORE'S registration carries as the PLAN'S copies when the run was registered before the store opened (the
#: identity it reads carries the registered run id): the positions mode the selected instrument declares and the multi-turn scope that
#: follows from it. A store observes neither -- the standup's endpoint record does, and it may have stamped the served mode and its scope
#: before the store's ``CellStore`` opened -- so :meth:`StoreRecorder.run_payload` marks both unstated on that record: the plan's values are
#: carried for the record's shape (NOT NULL columns) and stand only where nothing stated the column before; an endpoint patch that stated
#: the served mode earlier keeps its value and its confidence entry whatever the store's record restates. A store that IS the run's first
#: registration (no registered run id in its identity) states them as it always did.
STORE_PLANNED_COLUMNS: Tuple[str, ...] = ("applied_positions_mode", "multi_turn_steering_scope")


# --------------------------------------------------------------------------------------------
# the switch
# --------------------------------------------------------------------------------------------


def notice(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def _truthy(v: Optional[str]) -> Optional[bool]:
    if v is None or v.strip() == "":
        return None
    return v.strip().lower() in ("1", "true", "yes", "on")


def emission_enabled(explicit: Optional[bool] = None) -> bool:
    """The one switch (module docstring): DISABLED env wins, then the explicit keyword, then
    ``REGISTRY_EMIT``, then "a spool resolves"."""
    if _truthy(os.environ.get(emit.DISABLED_ENV)):
        return False
    if explicit is not None:
        return bool(explicit)
    forced = _truthy(os.environ.get(EMIT_ENV))
    if forced is not None:
        return forced
    return emit.spool_root() is not None


def resolve_spool(spool: Optional[str] = None) -> Optional[str]:
    """The spool a hook writes to: an explicit path, else :func:`emit.spool_root`, else (when
    ``REGISTRY_EMIT=1`` forced emission on with no spool provisioned) ``~/registry/records``."""
    if spool:
        return spool
    root = emit.spool_root()
    if root:
        return root
    if _truthy(os.environ.get(EMIT_ENV)):
        return os.path.join(os.path.expanduser(emit.HOME_REGISTRY), "records")
    return None


def guarded(what: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: a hook never raises into the workload; a failure is one stderr line."""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)          # hook-fix-5: keep the real signature visible (__wrapped__) — the fit
        def wrapper(*a, **kw):        # writers feature-detect optional kwargs via inspect.signature(hook) before passing them
            try:
                return fn(*a, **kw)
            except Exception as exc:  # noqa: BLE001 — by contract: a lost record never costs a run
                notice(f"{what}: hook failed ({type(exc).__name__}: {exc}); the run continues, records for this step are lost")
                return None
        return wrapper
    return deco


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _confidence_entry(reason: str) -> Dict[str, str]:
    return {"confidence": _UNVERIFIED, "reason": reason}


#: how this hook names itself as the SOURCE of a field_confidence_json entry it writes on its own account (not a sidecar key, not the
#: writer's statement): the module path of the object statement
OBJECT_HOOK_SOURCE = "sisteer.registry.hooks.object_from_sidecar"
#: the entry the hook writes under field_confidence_json['fit_code_sha'] whenever the object payload it returns states fit_code_sha
#: 'unverified': the reason is the one the fit_sidecars adapter files for the same fact, the source
#: this hook
FIT_CODE_SHA_UNVERIFIED_ENTRY: Mapping[str, str] = {"confidence": _UNVERIFIED, "reason": "no sidecar records the fit code sha",
                                                     "source": OBJECT_HOOK_SOURCE}


#: the seed file the multi-turn scope derivation reads the eval's category from, relative to the PRODUCING checkout (the one this
#: module runs from: the store registers itself from the checkout that serves it, so the seeds beside it are the job's own)
EVALS_SEED_REL = os.path.join("configs", "registry", "evals.yaml")


def _producing_checkout_root() -> str:
    """The checkout this module runs from: ``<root>/src/sisteer/registry/hooks.py`` -> ``<root>``."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def seeded_eval_category(eval_id: object, root: Optional[str] = None) -> Tuple[Optional[str], str]:
    """``(eval_category, source)`` of ``eval_id`` from the producing checkout's ``configs/registry/evals.yaml`` -- the same seed the
    job-start planner reads through the SeedIndex -- or ``(None, why)`` when the seeds do not say: no eval id, no seed file beside this
    checkout, an unreadable file, an eval the seeds do not name, a row without a category. Never a guess: the caller files the scope
    ``unverified`` with ``why`` as the reason."""
    if not isinstance(eval_id, str) or not eval_id:
        return None, "the registry identity names no eval_id"
    path = os.path.join(root or _producing_checkout_root(), EVALS_SEED_REL)
    if not os.path.isfile(path):
        return None, f"no {EVALS_SEED_REL} beside the producing checkout's registry package"
    try:
        import yaml  # the harness venv has it; a plain interpreter without it reads as the seeds being unavailable, never a guess
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001 -- by contract: an unreadable seed file never costs a run
        return None, f"{EVALS_SEED_REL} unreadable ({type(exc).__name__}: {exc})"
    rows = doc.get("evals") if isinstance(doc, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and str(row.get("eval_id")) == eval_id:
            cat = row.get("eval_category")
            if isinstance(cat, str) and cat:
                return cat, f"{EVALS_SEED_REL} of the producing checkout"
            return None, f"{EVALS_SEED_REL} names no eval_category for eval {eval_id!r}"
    return None, f"eval {eval_id!r} is not in {EVALS_SEED_REL} of the producing checkout"


def served_positions_mode(value: object) -> Tuple[Optional[str], Optional[str]]:
    """The ONE applied-positions mode a standup's ``served_positions[<served key>]`` names, as ``(mode, None)``; or ``(None, None)``
    when the entry carries nothing to read (absent, an empty string, a dict without a mode key); or ``(None, why)`` when it carries
    something that is NOT one mode -- the caller then stamps ``unverified`` with ``why``. Shapes: a bare string; a ``{mode | position_mode}``
    dict; the proxy's validated LIST of modes (one element is the normal case; several equal strings read as that one string; an empty
    list, a non-string element or two DIFFERENT modes are not one mode, and the first element is never taken for the whole)."""
    if isinstance(value, dict):
        value = value.get("mode") or value.get("position_mode")
    if isinstance(value, list):
        if not value:
            return None, "served_positions lists no mode ([])"
        if not all(isinstance(m, str) and m for m in value):
            return None, f"served_positions list carries a non-string or empty element: {value!r}"
        distinct = sorted(set(value))
        if len(distinct) > 1:
            return None, f"served_positions lists {len(distinct)} different modes {distinct}; one mode cannot be read from them"
        return distinct[0], None
    if isinstance(value, str) and value:
        return value, None
    return None, None


def served_mode_entry(served_key: str) -> Dict[str, str]:
    """The confidence entry of a mode the standup's endpoint record attests for ``served_key``: the served fact, ``recorded``. On the
    patch it rides beside the mode, and the fold reads a patch's confidence JSON per column, so the row's job-start entry for the column
    (a seeded instrument that declared no mode) gives way to it while the row's other entries stand."""
    return {"confidence": "recorded", "source": f"served_positions[{served_key!r}] of the standup's endpoint record"}


#: The served modes whose registry reading depends on the OBJECT'S KIND. The proxy's ``served_positions`` spells the MASK it applies
#: within a forward, not the registry's whole-generation mode: for a prefill-only op the mask covers prompt rows alone, so the proxy's
#: ``all`` (every prompt row) is the registry's ``prefill_last_broadcast`` on an AlphaSteer object and ``prefill_all`` on a static
#: direction served through the lens, and its ``last`` (the last prompt row) is ``prompt_last`` there -- while on a LoRA, a ReLU-steer or
#: a vector object the same ``all`` is every token of every prefill and decode. Every other served mode (``response``,
#: ``response_all_turns``, ``non_template``, ``prefill_all``, ``prompt_last``) names the same positions in both vocabularies for every
#: kind whose adapter admits it, so it reads as itself whether or not the kind is known.
KIND_DEPENDENT_SERVED_MODES: Tuple[str, ...] = ("all", "last")
#: The lens path's spelling of a prefill-only static direction's mask -> the registry mode: the inverse of
#: ``capeval.petri.defense_cells.SERVED_MODE_BY_STEER_POSITIONS`` for ``static_r`` (``prefill_all`` -> ``all``, ``prompt_last`` -> ``last``;
#: the stamp the thinking-ON gate classifies, pinned equal by test), read back here for the registry kinds the lens adapters serve at
#: prefill alone (``lens_adapters._adapt_static_r`` / ``_adapt_caa_static_r``: decode steps are never edited). AlphaSteer needs no row:
#: :func:`derive.applied_positions_mode` reads its kind before any standup mode. On the native path the same objects write at every
#: position of every forward (the residual slot), so the spelling is read on the lens path alone.
LENS_SERVED_SPELLING: Dict[str, Dict[str, str]] = {
    "static_r_arditi": {"all": "prefill_all", "last": "prompt_last"},
    "caa_static_r": {"all": "prefill_all"},
}


def object_kind_from_spelling(spelling: object) -> Optional[str]:
    """A registry object kind from a kind spelling a serving record carries: the registry's own (``vocab.OBJECT_KINDS``) or a producer's
    legacy spelling (:data:`LEGACY_KIND_TO_OBJECT_KIND`: ``psr`` -> ``relu_steer``, ``lora_r4`` -> ``lora``, ...); None for anything else,
    which the caller treats as no kind at all rather than a guess."""
    from . import vocab as V
    if not isinstance(spelling, str) or not spelling:
        return None
    if spelling in V.OBJECT_KINDS:
        return spelling
    return object_kind_for_legacy_kind(spelling)


def served_adapter_kind(endpoint: Mapping[str, Any], served_key: str) -> Tuple[Optional[str], Optional[str]]:
    """``(object kind, source)`` of the adapter the endpoint record ATTESTS for ``served_key``, or ``(None, why)``: the engine identity's
    adapter block (``serving_attestation.engine_identity``, the validated attestation the standup hands the hook beside endpoint.json and
    the fanout folds in from ``serving_attestation.json``) carries the exported adapter's ``petri_source_kind`` -- the object kind the
    export wrote into the adapter's config (``lora_export``), None on the zero adapter and on a record older than that field. The
    endpoint's own ``adapters`` block is read for the same key when the identity carries none."""
    eng_id = endpoint.get("engine_identity")
    for adapters, where in ((eng_id.get("adapters") if isinstance(eng_id, Mapping) else None, "engine_identity.adapters"),
                            (endpoint.get("adapters"), "adapters")):
        block = adapters.get(served_key) if isinstance(adapters, Mapping) else None
        if not isinstance(block, Mapping):
            continue
        spelling = block.get("petri_source_kind")
        if spelling is None:
            continue
        kind = object_kind_from_spelling(spelling)
        if kind is None:
            return None, f"the endpoint's {where}[{served_key!r}].petri_source_kind {spelling!r} is not a known object kind spelling"
        return kind, f"the endpoint's {where}[{served_key!r}].petri_source_kind"
    return None, f"neither the served runs file entry nor the endpoint's attested adapter for {served_key!r} names the object's kind"


def served_run_object_kind(run: Mapping[str, object], endpoint: Optional[Mapping[str, Any]] = None,
                           served_key: Optional[str] = None) -> Optional[str]:
    """The object kind of a served run, or None when nothing says it: the served runs file entry's ``object_kind`` when it carries one,
    else the kind its ``object_id`` states by its spelling (:func:`identity.object_kind_from_id`: ``none`` for the anchor,
    ``system_instruction`` for an ``si:<sha>`` id -- a prompted object steers no positions whatever the endpoint serves on the route), else
    the kind the endpoint attests for the served adapter (:func:`served_adapter_kind`). A staged artifact's sha names no kind and an entry
    that carries none leaves it unknown: the mode reading then refuses the kind-dependent served modes rather than guessing, and the scope
    derivation reads a steered object (the rule reads the kind for the anchor and the prompt alone)."""
    kind = run.get("object_kind")
    if isinstance(kind, str) and kind:
        return kind
    kind = identity.object_kind_from_id(run.get("object_id"))
    if kind is not None:
        return kind
    if endpoint is not None and served_key is not None:
        return served_adapter_kind(endpoint, served_key)[0]
    return None


def served_mode_for_registry(served_mode: str, kind: Optional[str], serving_path: Optional[str], served_key: str,
                             kind_source: Optional[str] = None) -> Tuple[str, Dict[str, object]]:
    """The registry's ``applied_positions_mode`` for the ONE served mode the endpoint record spells for ``served_key``
    (:func:`served_positions_mode`), read against the object's kind and the serving path, with its confidence entry:

    * the kind known: :func:`derive.applied_positions_mode` -- the registry's one rule table -- over the served mode as the standup mode
      (an AlphaSteer object reads ``prefill_last_broadcast`` and an antidote ``last`` whatever mask the proxy spells, the anchor and a
      prompted object ``none``, every other kind the served mode itself, ``recorded``), after the lens path's spelling of a prefill-only
      static direction's mask is read back (:data:`LENS_SERVED_SPELLING`, ``derived``); a static direction's ``all`` / ``last`` on a
      serving path the record does not name is ``unverified`` (the path decides whether the mask is the prefill's or every forward's);
    * the kind unknown: a served mode in :data:`KIND_DEPENDENT_SERVED_MODES` is ``unverified`` with the reason -- never recorded as
      every-token steering, which is what ``all`` would claim for a prefill-only object -- and any other known mode is itself,
      ``recorded``; a spelling outside the vocabulary is ``unverified`` naming it."""
    from . import vocab as V
    source = f"served_positions[{served_key!r}] = {served_mode!r} of the standup's endpoint record"
    if kind is None:
        if served_mode in KIND_DEPENDENT_SERVED_MODES:
            return _UNVERIFIED, _confidence_entry(
                f"{source} spells the mask the proxy applies within a forward, whose registry mode depends on the object's kind (an "
                f"AlphaSteer object or a static direction on the lens path steers the prefill alone; a LoRA every token); "
                f"{kind_source or 'nothing names the kind'}")
        if served_mode in V.APPLIED_POSITIONS_MODES:
            return served_mode, served_mode_entry(served_key)
        return _UNVERIFIED, _confidence_entry(f"{source} is not a known applied-positions mode")
    spelled = served_mode
    lens_spelling = LENS_SERVED_SPELLING.get(kind, {}).get(served_mode)
    if lens_spelling is not None:
        if serving_path == "lens":
            spelled = lens_spelling
        elif serving_path is None:
            return _UNVERIFIED, _confidence_entry(
                f"{source} on a {kind} object: on the lens path the mask is the prefill's ({lens_spelling}), on the native path every "
                f"forward's ({served_mode}), and the record names no serving path")
    mode, confidence_, rule = D.applied_positions_mode(serving_path or "", kind, None, standup_mode=spelled)
    if lens_spelling is not None and spelled == lens_spelling and mode == lens_spelling:
        return str(mode), D.confidence(mode, "derived", f"{source} on the lens path: a {kind} object is served at prefill alone, so the "
                                                        f"proxy's {served_mode!r} mask is the registry's {lens_spelling}")
    if confidence_ == "recorded":
        return str(mode), served_mode_entry(served_key)
    return str(mode), D.confidence(mode, confidence_, f"{rule} ({source}, read against the object's kind)")


def served_run_eval_category(run: Mapping[str, object]) -> Tuple[Optional[str], str]:
    """``(eval_category, source)`` for one entry of the served runs file (:func:`load_runs_file`): the entry's own ``eval_category`` when
    it carries one; else the producing checkout's seeds (:func:`seeded_eval_category`) read for the entry's ``eval_id``, or, absent that
    too, for the eval the registry run id itself names (its second segment -- every registry run id carries it, so the hook needs no
    more than the id the launching shell wrote); ``(None, why)`` when none of them says, and the caller files the scope ``unverified``."""
    category = run.get("eval_category")
    if isinstance(category, str) and category:
        return category, "the served runs file"
    eval_id = run.get("eval_id")
    if not (isinstance(eval_id, str) and eval_id):
        try:
            eval_id = identity.parse_run_id(str(run.get("run_id")))["eval_id"]
        except ValueError:
            return None, f"run_id {str(run.get('run_id'))!r} is not a registry run id and the served runs file names no eval_id"
    return seeded_eval_category(eval_id)


def served_scope(mode: str, run: Mapping[str, object], kind: Optional[str] = None) -> Tuple[str, Dict[str, object]]:
    """The multi-turn steering scope that follows a served mode, with its confidence entry, for the endpoint patch: derived
    (:func:`derive.multi_turn_steering_scope`) from the mode, the object's kind (``kind``, else :func:`served_run_object_kind` over the
    entry; a kind nothing names reads as a steered object, the rule reading the kind for the anchor and the prompt alone) and the eval's
    category (:func:`served_run_eval_category`); ``unverified`` with the reason when the category cannot be read, and the derivation's own
    ``unverified`` when the mode is (an ambiguous served list on a multi-turn eval). Emitted beside every mode the endpoint stamps, so the
    scope the registration derived never stands beside a served mode it no longer follows from."""
    kind = kind or served_run_object_kind(run) or "steered"
    if kind == "none":
        scope, confidence_, source = D.multi_turn_steering_scope(mode, "", kind)
        return str(scope), D.confidence(scope, confidence_, source)
    category, source = served_run_eval_category(run)
    if category is None:
        return _UNVERIFIED, _confidence_entry(f"multi_turn_steering_scope not derived: {source}")
    scope, confidence_, scope_source = D.multi_turn_steering_scope(mode, category, kind)
    return str(scope), D.confidence(scope, confidence_, f"{scope_source} (eval_category from {source})")


def _sha_or_unverified(value: Optional[str], length: int, fc: MutableMapping[str, object], field: str, why: str) -> str:
    if isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value.lower()):
        return value.lower()
    fc[field] = _confidence_entry(f"{why}: {value!r} is not a {length}-hex digest")
    return _UNVERIFIED


# --------------------------------------------------------------------------------------------
# run identity (what the driver / the planned-run file supplies)
# --------------------------------------------------------------------------------------------

#: ``run`` columns the store hook fills ITSELF (from the resolved spec, the producer block, the
#: store); everything else required must come from the supplied identity.
_HOOK_FILLED_RUN_FIELDS = frozenset({
    "run_id", "campaign", "producer_run_key_legacy", "producer_object_key_legacy", "lifecycle_state", "code_sha",
    "substrate", "substrate_source", "started_utc", "queued_utc", "model_revision", "intervention_hash",
    "harness_config_hash", "env_digest", "strength_coefficient", "cluster_job_id", "store_uri", "trial_index",
    "seed_base", "field_confidence_json", "extra_json", "instrument_match", "serving_path", "applied_positions_mode",
    "multi_turn_steering_scope", "harness", "prompt_template_sha256", "transfer_kind", "model_id",
})
#: identity fields that MUST be supplied (never defaulted): the eval and its version, the object, the
#: instrument facts a store cannot see.
REQUIRED_IDENTITY_FIELDS: Tuple[str, ...] = tuple(
    f for f in R.REQUIRED_FIELDS["run"] if f not in _HOOK_FILLED_RUN_FIELDS)


def missing_identity_fields(ident: Mapping[str, object]) -> List[str]:
    """The required identity fields ``ident`` lacks (empty = usable)."""
    return [f for f in REQUIRED_IDENTITY_FIELDS if ident.get(f) in (None, "")]


def identity_from_env() -> Optional[Dict[str, object]]:
    """The planned-run file ``$SISTEER_REGISTRY_RUN_IDENTITY`` points at (the submit wrapper's
    pending file, exported by the job shell), parsed; None when unset / unreadable (one notice)."""
    p = os.environ.get(RUN_IDENTITY_ENV)
    if not p:
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        notice(f"{RUN_IDENTITY_ENV}={p!r} unreadable ({exc}); store emission stays inert")
        return None
    if isinstance(data, dict) and isinstance(data.get("runs"), list) and data["runs"]:
        routed = routed_identities(data["runs"])
        if routed:
            return routed
    if isinstance(data, dict) and isinstance(data.get("plan"), dict):
        plan = dict(data["plan"])
        if data.get("run_id") and not plan.get("run_id"):
            plan["run_id"] = data["run_id"]      # the pending file binds the run id at its top level
        data = plan
    return data if isinstance(data, dict) else None


ROUTE_SEP = "|"
#: the harness's DERIVED no-op control leg: `none__lora0` on a native standup, `none__lens0` on a lens one — enumerated
#: by defense_cells.enumerate_cells from the manifest and NEVER an object (every petri / misalign manifest header says so;
#: petri_registry_handoff.py close-leg files nothing for it: the harness's derived no-op control `none__lora0` is
#: never a run). A store whose cell is one binds to NO run: not the `none` object's (the kind=="none" fallback routed it there on
#: 2026-10-01 and wrote a foreign item row onto the `none` object's run), not its own.
DERIVED_CONTROL_OBJECT_RE = re.compile(r"^none__(?:lora|lens)\d*$")


def derived_control_object(spec: Mapping[str, Any]) -> Optional[str]:
    """Named with the registry noun -- the serving-path control OBJECT, `none__lora0` / `none__lens0`. The derived control leg's cell id when ``spec`` (a resolved store spec or its row constants) is one, else None."""
    if str(spec.get("kind", "")) != "none":
        return None
    for k in ("cell_id", "arm_name"):
        v = spec.get(k)
        if isinstance(v, str) and DERIVED_CONTROL_OBJECT_RE.match(v):
            return v
    return None


def routed_identities(runs: Sequence[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    """The ``{route: identity}`` map of a job-start pending file's ``runs`` list. Each entry is
    ``{"run_id", "plan", "route": {"object": <the config's object entry>, "tasks": [<producer spellings>]}}``;
    the identity is the plan with its run id. Keys: ``"<object>|<spelling>"`` for every spelling, and the bare
    spelling when exactly one registered run carries it (a store whose spec names no object still routes)."""
    out: Dict[str, Dict[str, object]] = {}
    bare: Dict[str, List[Dict[str, object]]] = {}
    for r in runs:
        if not isinstance(r, Mapping) or not r.get("run_id") or not isinstance(r.get("plan"), Mapping):
            continue
        ident = dict(r["plan"])
        ident["run_id"] = str(r["run_id"])
        route = r.get("route") if isinstance(r.get("route"), Mapping) else {}
        obj = str(route.get("object") or ident.get("producer_object_key_legacy") or "")
        tasks = [str(t) for t in (route.get("tasks") or [])] or [str(ident.get("eval_id") or "")]
        for t in tasks:
            if not t:
                continue
            if obj:
                out[f"{obj}{ROUTE_SEP}{t}"] = ident
            bare.setdefault(t, []).append(ident)
    for t, idents in bare.items():
        if len(idents) == 1 and t not in out:
            out[t] = idents[0]
    return out


def normalise_identities(ident: Optional[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    """``{route: identity}`` from either ONE identity (route ``'*'``: every row of the store) or a
    mapping ``{task_name: identity}`` (a store holding several evals routes each row by its task)."""
    if not ident:
        return {}
    if "eval_id" in ident or "run_id" in ident:
        return {"*": dict(ident)}
    out: Dict[str, Dict[str, object]] = {}
    for k, v in ident.items():
        if isinstance(v, Mapping):
            out[str(k)] = dict(v)
    return out


# --------------------------------------------------------------------------------------------
# the store recorder (CellStore hook)
# --------------------------------------------------------------------------------------------


class StoreRecorder:
    """The recorder a ``CellStore`` calls: ``on_manifest`` → ``run`` + ``run_state running``;
    ``on_row`` → buffered ``item_result`` (+ ``item_flags``) rows, whole-row, one per (item, metric),
    flushed every 500 rows, on completion and at interpreter exit; ``on_generation_complete`` →
    a ``run_patch``; ``on_results`` → ``run_state generated``, the completion ``run_patch``,
    ``artifact`` pointers, ``run_degeneration not_measured`` rows for the declared measures, and a
    final push. One ``RecordWriter`` per run the store holds (one per routed eval)."""

    FLUSH_EVERY = 500

    def __init__(self, store_key: str, resolved_spec: Mapping[str, Any], identities: Mapping[str, Mapping[str, object]],
                 *, spool: Optional[str] = None, push_script: Optional[str] = None, script: str = "capeval.store.CellStore",
                 code_sha: Optional[str] = None, session: Optional[str] = None,
                 declared_degeneration: Sequence[Tuple[str, str]] = DEFAULT_DECLARED_DEGENERATION,
                 disabled: bool = False, row_constants: Optional[Mapping[str, Any]] = None) -> None:
        self.store_key = store_key
        self.spec = dict(resolved_spec)
        # the store's row constants, kept beside the spec: ``on_manifest`` replaces ``spec`` with the producer's
        # manifest, and a producer whose seed lives only in its row constants (never lifted to the manifest) would
        # otherwise lose it — ``run.seed_base`` reads them as its fallback
        self.row_constants: Optional[Dict[str, Any]] = dict(row_constants) if isinstance(row_constants, Mapping) else None
        self.identities = {k: dict(v) for k, v in identities.items()}
        self.spool = spool
        self.push_script = push_script or os.environ.get(emit.PUSH_SCRIPT_ENV)
        self.producer = emit.producer_block(script, code_sha, session)
        self.declared_degeneration = tuple(declared_degeneration)
        self.disabled = disabled
        self.writers: Dict[str, emit.RecordWriter] = {}     # route -> writer
        self.run_ids: Dict[str, str] = {}                    # route -> run_id
        self._items: Dict[str, List[Dict[str, object]]] = {}
        self._metrics: Dict[str, List[Dict[str, object]]] = {}     # route -> buffered run_metric rows (the producer's aggregates)
        self._flags: Dict[str, List[Dict[str, object]]] = {}
        self._flagged_items: Dict[str, set] = {}
        self._item_ids: Dict[str, set] = {}
        self._ts: Dict[str, float] = {}
        self._unrouted: Dict[str, int] = {}
        self._finished = False
        self.manifest_path: Optional[str] = None
        self.store_dir: Optional[str] = None
        self.run_payloads: Dict[str, Dict[str, object]] = {}
        #: decode-recording: per route, whether the manifest-time read filed the decode columns (else `on_results` tries the
        #: standup's request recorder, `<store root>/resolved_sampler.json`, which lands after the manifest)
        self._decode_filed: Dict[str, bool] = {}
        #: decode-recording (Major 1): per route, whether the store-level recorder document MAY fill the completion patch
        #: -- only a route whose eval is a tinycap ATOM the manifest states no request for (never mmlu / xstest / or_bench_hard_1k, whose
        #: request the DRIVER builds and the manifest states; never a route whose eval is unknown) -- and the route's producer spelling,
        #: the task the recorder document must cover
        self._decode_recorder_eligible: Dict[str, bool] = {}
        self._decode_spelling: Dict[str, Optional[str]] = {}
        #: set once the bound store is the harness's derived control leg — every hook files nothing after that
        self._derived_control_object: Optional[str] = derived_control_object(self.spec)
        #: the items the store ALREADY held when this recorder bound it (a resumed store), per route — counted
        #: into n_items_generated, never re-emitted
        self._seeded_items: Dict[str, int] = {}
        atexit.register(self._atexit)

    # -- identity ---------------------------------------------------------------------------------
    @staticmethod
    def object_key_of(spec: Mapping[str, Any]) -> str:
        """The store's object discriminator: ``<arm>@<intervention8>`` for a steered store, ``none``
        for an unsteered one (the same rule the store adapter's ``cell_key`` grammar applies)."""
        if str(spec.get("kind", "")) == "none":
            return "none"
        return f"{spec.get('arm_name')}@{str(spec.get('intervention_hash', ''))[:8]}"

    def _campaign(self, ident: Mapping[str, object]) -> str:
        return str(ident.get("campaign") or self.spec.get("sweep") or "uncategorised")

    def _run_id(self, ident: Mapping[str, object]) -> str:
        if ident.get("run_id"):
            return str(ident["run_id"])
        return identity.run_id(self._campaign(ident), self.store_key, self.object_key_of(self.spec),
                               str(ident["eval_id"]), str(ident["instrument_id"]))

    def _writer(self, route: str) -> emit.RecordWriter:
        w = self.writers.get(route)
        if w is None:
            ident = self.identities[route]
            rid = self._run_id(ident)
            self.run_ids[route] = rid
            w = emit.RecordWriter(self.spool, self._campaign(ident), rid, self.producer, push_script=self.push_script,
                                  disabled=self.disabled)
            self.writers[route] = w
        return w

    def _object_candidates(self) -> List[str]:
        """The object entries this store may stand for, in the job's spelling: the spec's ``cell_id`` / ``arm_name``
        (the Petri and misalignment stores name the object there; a controls store names its CONTROL token, which
        matches no object), then every registered object whose ``object_id`` starts with the sha the store recorded
        for its served object (the served controls' ``fit_provenance.object_sha`` — 16 hex — or an ``object_sha256``)."""
        out: List[str] = []
        for k in ("cell_id", "arm_name"):
            v = self.spec.get(k)
            if isinstance(v, str) and v:
                # the controls stores' token is the object id with `__` → `--` (cell_controls.store_key_token, injective):
                # try the spelling as written and its inverse, so `constv2-mix-full--lora_r4_ce` reaches the job's entry
                for cand in (v, v.replace("--", "__")):
                    if cand not in out:
                        out.append(cand)
        # the served controls lay each object's stores under `<root>/<object entry>/cells/<key>`: the directory two above
        # the store names the object when the spec does not (a Petri / capeval root there matches no registered object)
        if self.store_dir:
            above = os.path.basename(os.path.dirname(os.path.dirname(self.store_dir.rstrip(os.sep))))
            if above and above not in out:
                out.append(above)
        shas: List[str] = []
        fp = self.spec.get("fit_provenance")
        if isinstance(fp, Mapping):
            for k in ("object_sha256", "object_sha"):
                if isinstance(fp.get(k), str) and fp[k]:
                    shas.append(str(fp[k]).lower())
        for k in ("object_sha256", "object_sha", "served_object_sha256"):
            if isinstance(self.spec.get(k), str) and self.spec[k]:
                shas.append(str(self.spec[k]).lower())
        if shas:
            for route, ident in self.identities.items():
                if ROUTE_SEP not in route:
                    continue
                oid = str(ident.get("object_id") or "").lower()
                obj = route.split(ROUTE_SEP, 1)[0]
                if oid and any(oid.startswith(h) for h in shas) and obj not in out:
                    out.append(obj)
        # the unsteered object's cell routes as `none`; the harness's DERIVED control leg (`none__lora0`) is kind `none` too
        # but never an object — it reaches no candidate, so it binds to no run
        if str(self.spec.get("kind", "")) == "none" and "none" not in out and not derived_control_object(self.spec):
            out.append("none")
        return out

    def _object(self) -> Optional[str]:
        """The object entry this store routes by: the first candidate a registered route names, else the first."""
        cands = self._object_candidates()
        for c in cands:
            if any(r.split(ROUTE_SEP, 1)[0] == c for r in self.identities if ROUTE_SEP in r):
                return c
        return cands[0] if cands else None

    def _route(self, task: Optional[str]) -> Optional[str]:
        """``"<object>|<task>"`` first (a job-start map; every object candidate tried), then the task alone, each also
        with the task's ``<family>/`` prefix stripped (``petri/self_preservation`` → ``self_preservation``), then the
        one-identity route ``*``."""
        if task is not None:
            spellings = [task]
            if "/" in task:
                spellings.append(task.rsplit("/", 1)[1])
            for obj in self._object_candidates():
                for t in spellings:
                    if f"{obj}{ROUTE_SEP}{t}" in self.identities:
                        return f"{obj}{ROUTE_SEP}{t}"
            for t in spellings:
                if t in self.identities:
                    return t
        if "*" in self.identities:
            return "*"
        return None

    def _own_routes(self) -> List[str]:
        """The routes THIS store's records may open a run for: with a routed map, only those whose object component
        is one of this store's object candidates (a job registers every object x eval; one store holds one object);
        a map with no object components, or the ``*`` route, is the store's own."""
        routes = list(self.identities)
        if "*" in self.identities:
            return ["*"]
        keyed = [r for r in routes if ROUTE_SEP in r]
        if not keyed:
            return routes
        cands = self._object_candidates()
        return [r for r in keyed if r.split(ROUTE_SEP, 1)[0] in cands]

    # -- payload builders -----------------------------------------------------------------------------
    def _seed_base(self) -> Optional[object]:
        """``run.seed_base``: the resolved spec's ``seed`` first; else the row constants' ``seed``, then their
        ``seed_base`` (the recorder's own constants, then the ``row_constants`` copy the manifest archives); else
        None. A producer whose seed rides only its row constants (``seed_base = s`` on every row) is recorded
        without lifting the seed to its manifest. Never raises: a malformed carrier reads as absent."""
        try:
            if self.spec.get("seed") is not None:
                return self.spec.get("seed")
            for carrier in (self.row_constants, self.spec.get("row_constants")):
                if not isinstance(carrier, Mapping):
                    continue
                for key in ("seed", "seed_base"):
                    if carrier.get(key) is not None:
                        return carrier.get(key)
        except Exception:  # noqa: BLE001 — by contract: a lost fact never costs a run
            return None
        return None

    def run_payload(self, route: str) -> Dict[str, object]:
        """The ``run`` record for one routed identity: the identity's declared facts + the resolved
        spec's hashes + the producer's facts, ``unverified`` (with a reason) where a fact is absent.

        When the identity carries a registered run id (the submit wrapper's or the job-start registration's ``run`` record is in the log
        before this one), the record's ``applied_positions_mode`` and ``multi_turn_steering_scope`` are the PLAN'S copies, marked unstated
        (:data:`STORE_PLANNED_COLUMNS`, ``extra.unstated_columns``): the fold keeps the value and the confidence entry an earlier unmarked
        statement supplied -- the registration's, or the standup's endpoint patch that stamped the served mode and its scope before the
        store opened -- and applies the plan's values only where nothing stated the column yet. A full record's confidence JSON would
        otherwise be the row's whole, and the store's copy of the plan would erase the endpoint's observation and its provenance."""
        ident = dict(self.identities[route])
        spec = self.spec
        fc: Dict[str, object] = dict(ident.get("field_confidence_json") or {}) if isinstance(ident.get("field_confidence_json"), dict) else {}
        kind = str(spec.get("kind", "none"))
        started = utc_now()
        mode = ident.get("applied_positions_mode", "none" if kind == "none" else _UNVERIFIED)
        # the multi-turn steering scope: the identity's own statement when it carries one (the job-start planner derives it from the
        # selected instrument's mode, the seeded eval's category and the object's kind); else DERIVED here from the mode, the category
        # the producing checkout's seeds give the eval and the kind the identity's object id states (`si:<sha>` is a prompted object,
        # which steers no positions; the store's legacy kind spelling stands in for every other id) -- never the single-turn default a
        # store used to stamp on every steered run
        if "multi_turn_steering_scope" in ident:
            scope = ident["multi_turn_steering_scope"]
        elif kind == "none":
            scope = "not_applicable"
        else:
            category, source = seeded_eval_category(ident.get("eval_id"))
            if category is None:
                scope = _UNVERIFIED
                fc["multi_turn_steering_scope"] = _confidence_entry(f"multi_turn_steering_scope not derived: {source}")
            else:
                object_kind = identity.object_kind_from_id(ident.get("object_id")) or kind
                scope, scope_conf, scope_src = D.multi_turn_steering_scope(str(mode), category, object_kind)
                fc["multi_turn_steering_scope"] = D.confidence(scope, scope_conf, f"{scope_src} (eval_category from {source})")
        p: Dict[str, object] = {
            "run_id": self._run_id(ident),
            "campaign": self._campaign(ident),
            "producer_run_key_legacy": str(ident.get("producer_run_key_legacy") or self.store_key),
            "producer_object_key_legacy": str(ident.get("producer_object_key_legacy") or self.object_key_of(spec)),
            # a store registering itself at manifest time proves `running`; the schema requires every *_utc of a
            # REACHED state, so the queue instant it never observed is stamped equal to the start (a submit
            # wrapper's earlier `queued` event, when there was one, owns the column in the fold)
            "lifecycle_state": "running",
            "queued_utc": started,
            "started_utc": started,
            "model_id": str(ident.get("model_id") or spec.get("model_id")),
            "model_revision": _sha_or_unverified(spec.get("model_revision"), _SHA1_LEN, fc, "model_revision", "store manifest model_revision"),
            "object_id": ident.get("object_id", "none" if kind == "none" else None),
            "transfer_kind": ident.get("transfer_kind", "none" if kind == "none" else None),
            "instrument_match": ident.get("instrument_match", _UNVERIFIED),
            "harness": ident.get("harness", "bespoke_sisteer"),
            "prompt_template_sha256": ident.get("prompt_template_sha256", _UNVERIFIED),
            "serving_path": ident.get("serving_path") or ENGINE_TO_SERVING_PATH.get(str(spec.get("engine", "")), _UNVERIFIED),
            "applied_positions_mode": mode,
            "multi_turn_steering_scope": scope,
            "code_sha": self.producer["code_sha"],
            "substrate": self.producer["substrate"],
            "substrate_source": "derived",
            "intervention_hash": spec.get("intervention_hash"),
            "harness_config_hash": spec.get("harness_config_hash"),
            "env_digest": spec.get("env_digest"),
            "strength_coefficient": spec.get("coeff"),
            "cluster_job_id": spec.get("slurm_job_id") or os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_hostname": self.producer["host"],
            "store_uri": self.store_dir,
            "trial_index": int(spec.get("trial_id", 0) or 0),
            "seed_base": self._seed_base(),
            "extra": {"store_key": self.store_key, "attempt": spec.get("attempt"), "arm_name_legacy": spec.get("arm_name"),
                      "kind_legacy": kind, "tiers": list(spec.get("tiers") or []), "engine_legacy": spec.get("engine"),
                      "route": route, "registered_at": "store_manifest" if not ident.get("run_id") else "submit_wrapper"},
        }
        # every declared identity fact travels verbatim (item_set_id, instrument_id, eval_id, token cap, decode …)
        for k, v in ident.items():
            if k in ("field_confidence_json", "extra", "plan", "submit_evidence_uri", "scheduler_state") or k in p and k != "model_id":
                continue
            if k in R.NATURAL_KEYS["run"] or k in ("run_id",):
                continue
            p[k] = v
        for k in ("instrument_match", "prompt_template_sha256", "serving_path", "applied_positions_mode"):
            if p.get(k) == _UNVERIFIED and k not in fc:
                fc[k] = _confidence_entry(f"{k} not supplied by the driver's registry identity and not observable from the store")
        # decode-recording (the registry owner, 2026-10-13: decode_policy_id 'unrecorded' is a PRODUCER gap): the request's OBSERVED
        # sampler from the manifest, selected by the EVAL this route names (never by what the shared manifest happens to carry) --
        # decode_sent_json + the typed columns + a `recorded` entry naming the artifact, or an `unverified` entry naming why not
        self._apply_decode_evidence(route, p, fc)
        # the store runs on the GPU node — its own environment states the hardware (env ladder, nvidia-smi,
        # sacct AllocTRES as the fallback); an absent fact stays NULL, never a guess. The identity's own values win.
        hw = emit.hardware_from_env(job_id=p.get("cluster_job_id"))
        for field, src_key in (("gpu_count", "gpu_count_source"), ("gpu_type", "gpu_type_source")):
            if p.get(field) is None and hw.get(field) is not None:
                p[field] = hw[field]
                fc[field] = {"confidence": "recorded", "reason": f"{hw.get(src_key)} read at the store manifest"}
        if p.get("code_sha") == _UNVERIFIED:
            fc["code_sha"] = _confidence_entry("git rev-parse HEAD unavailable in the producing checkout")
        if fc:
            p["field_confidence_json"] = fc
        if isinstance(ident.get("extra"), dict):
            p["extra"].update(ident["extra"])   # type: ignore[union-attr]
        if ident.get("run_id"):
            # the run was registered before this store opened: its mode and scope here are the plan's copies, carried for the record's
            # shape and not stated (STORE_PLANNED_COLUMNS) -- a served mode an endpoint patch stamped in between stands, with its entry
            extra: Dict[str, object] = p["extra"]    # type: ignore[assignment]
            marked = extra.get(UNSTATED_COLUMNS_KEY)
            marked = [str(c) for c in marked] if isinstance(marked, list) else []
            extra[UNSTATED_COLUMNS_KEY] = marked + [c for c in STORE_PLANNED_COLUMNS if c in p and c not in marked]
        return p

    def _route_spelling(self, route: str) -> Tuple[Optional[str], Optional[str]]:
        """The PRODUCER spelling of the route's eval, or ``(None, why)``: the task component of a task-routed key (``<object>|xstest``,
        ``<object>|or_bench_hard_1k``, ``<object>|mmlu`` -- the pending file's ``route.tasks`` are the producer's spellings), the bare
        spelling route itself, or -- for the one-identity route ``*`` -- the manifest's ``controls`` list when it names exactly ONE
        control. Several controls under ``*`` name no request: unverified by name, nothing filed."""
        if ROUTE_SEP in route:
            return route.split(ROUTE_SEP, 1)[1], None
        if route != "*":
            return route, None
        controls = self.spec.get("controls")
        if isinstance(controls, list) and len(controls) == 1 and controls[0]:
            return str(controls[0]), None
        if isinstance(controls, list) and len(controls) > 1:
            return None, (f"one identity, {len(controls)} controls ({', '.join(str(c) for c in controls)}): which request this run sent "
                          f"is not stated")
        return None, "the manifest names no controls list and the route names no task: which request this run sent is not stated"

    def _apply_decode_evidence(self, route: str, p: Dict[str, object], fc: Dict[str, object]) -> None:
        """decode-recording: ``decode_evidence.from_store_manifest`` over the resolved spec (the producer's manifest at ``on_manifest``) for
        the route's own eval -- ``request_samplers[<spelling>]`` first, ``resolved_config.gen`` only for a SAMPLED over-refusal control
        with ``do_sample`` true, else nothing. Sets the decode columns, the per-column ``recorded`` entries and
        ``extra.decode_evidence``; an absence, a contradiction or any exception files the ``unverified`` entry naming it and NO decode
        column (the recorder never raises into the store: a lost fact never costs a run). ``decode_resolved_json`` is never set here --
        the recorder document does not exist at manifest time (``on_results`` reads it)."""
        self._decode_filed[route] = False
        self._decode_recorder_eligible[route] = False
        self._decode_spelling[route] = None
        try:
            spelling, why = self._route_spelling(route)
            self._decode_spelling[route] = spelling
            rs = self.spec.get(DE.MANIFEST_KEY_REQUEST_SAMPLERS)
            # the recorder document (the standup's, store-level) describes the ATOMS' requests: a legacy control's request is the driver's
            # and the manifest's to state (an entry under request_samplers, or nothing on a legacy manifest -- unverified by name, never
            # the recorder's answer); a route with no spelling names no request at all
            self._decode_recorder_eligible[route] = bool(spelling) and spelling not in DE.LEGACY_CONTROL_SPELLINGS and not (
                isinstance(rs, Mapping) and spelling in rs)
            ev = DE.from_store_manifest(self.spec, spelling, store_dir=self.store_dir, why_no_spelling=why)
        except DE.DecodeEvidenceProblem as exc:
            fc["decode_sent_json"] = _confidence_entry(f"decode evidence refused ({type(exc).__name__}): {exc}")
            return
        except Exception as exc:  # noqa: BLE001 -- by contract: a lost fact never costs a run
            fc["decode_sent_json"] = _confidence_entry(f"decode evidence not read ({type(exc).__name__}: {exc})")
            return
        if not ev.recorded():
            fc["decode_sent_json"] = _confidence_entry(str(ev.why))
            return
        p.update(ev.columns())
        fc.update(ev.confidence_entries())
        extra = p.get("extra")
        if isinstance(extra, dict):
            extra[DE.EXTRA_KEY] = ev.extra_block()
        self._decode_filed[route] = True

    # -- hooks -------------------------------------------------------------------------------------------
    def on_manifest(self, resolved_spec: Mapping[str, Any], manifest_path: str) -> None:
        self.spec = dict(resolved_spec)
        self.manifest_path = str(manifest_path)
        self.store_dir = os.path.dirname(str(manifest_path))
        self._derived_control_object = derived_control_object(self.spec)
        if self._derived_control_object:
            # the close-leg's reading, said once — no run opened, no row filed, no refusal of the job
            notice(f"store {self.store_key}: cell {self._derived_control_object!r} is the harness's derived no-op control leg — never a "
                   "registered object, never a run (petri_registry_handoff.py close-leg); nothing opened, nothing recorded")
            return
        self._seed_items_from_store()
        own = self._dedupe_by_run(self._own_routes())
        if not own:
            notice(f"store {self.store_key}: none of the {len(self.identities)} registered runs names this store's object "
                   f"({self._object()!r}); nothing opened at the manifest (rows route by task if one matches)")
        elif len(own) == 1:
            # ONE run for this store's object: it is this store's run, opened at the manifest as before
            self._open(own[0])
        # several registered evals for this object (a Petri store's behaviours, a job's controls): a store holds the rows
        # of the evals it actually runs — each run opens at its FIRST routed row (on_row), never on the manifest alone

    def _seed_items_from_store(self) -> None:
        """A recorder bound to a NON-EMPTY store (a CellStore resumed over an existing cell dir — the overflow re-run
        drive, a restarted leg) counts the items the store already holds into ``_item_ids`` per route, so ``on_results``
        reports the run's STORE TOTAL (the field every coverage predicate reads from one row) and not this pass's
        increment. Count only: nothing is re-emitted, no run is opened, no route becomes touched. A torn final line (a
        write cut mid-way) is skipped like ``CellStore._parsed_rows`` skips it; any other undecodable line stops the
        seeding with a notice."""
        if not self.store_dir:
            return
        rows_path = os.path.join(self.store_dir, "rows.jsonl")
        if not os.path.isfile(rows_path):
            return
        try:
            with open(rows_path, encoding="utf-8") as fh:
                lines = fh.read().split("\n")
        except OSError as exc:
            notice(f"store {self.store_key}: existing rows.jsonl unreadable ({type(exc).__name__}: {exc}); n_items_generated counts this pass only")
            return
        if lines and lines[-1] == "":
            lines.pop()
        seeded: Dict[str, set] = {}
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                if i == len(lines) - 1:
                    break            # a torn final line: the interrupted item re-runs and files itself
                notice(f"store {self.store_key}: rows.jsonl line {i + 1} undecodable; n_items_generated seeded from the rows before it")
                break
            if not isinstance(row, Mapping):
                continue
            metric = str(row.get("metric", ""))
            item = row.get("item_id")
            if row.get("domain") == "_meta" or metric.startswith("ts_") or item in (None, "", AGGREGATE_ITEM_ID) or metric.endswith(AGGREGATE_SUFFIX):
                continue
            route = self._route(row.get("task"))
            if route is None:
                continue
            seeded.setdefault(route, set()).add(str(item))
        if not seeded:
            return
        for route, items in seeded.items():
            self._item_ids.setdefault(route, set()).update(items)
            self._seeded_items[route] = len(items)
        notice(f"store {self.store_key}: resumed over a store holding {sum(len(v) for v in seeded.values())} item(s) across "
               f"{len(seeded)} run(s); n_items_generated will report each run's store total, not this pass's increment (D784)")

    def _open(self, route: str) -> None:
        """``run`` (running) + ``run_state running`` for one route's run, ONCE PER SPOOL: once in this recorder, and never again
        when the run's record directory under the spool already holds a ``run`` record -- a RESUMED store (a second CellStore over
        the same cell dir: the judge's grading-time re-open, a restarted driver) re-validates its manifest and files a
        ``run_state running`` for the run that exists, never a second ``run`` payload (``on_generated`` opened
        the judged runs in the driver's recorder, the grading re-open minted them again -- 5 ``run`` payloads for 3 runs; the
        pin counts one per logical run). A run whose spool holds no ``run`` yet is minted here (the ONE opening point for
        ``on_generated``, ``on_row`` and the aggregate alike).

        Deduplication assumes serialized openings for each (spool, campaign, run_id). In the production driver/judge flow,
        the judge opens the run only after the driver has published its terminal RESULT and finished registry writes;
        overlapping drivers or judges for that identity are unsupported. The directory scan and emission are not atomic
        and do not guarantee exactly-once emission under concurrent openers."""
        rid = self._run_id(self.identities[route])
        if any(self._run_id(self.identities[r]) == rid for r in self.run_payloads):
            return
        w = self._writer(route)
        payload = self.run_payload(route)
        self.run_payloads[route] = payload
        if not self._run_recorded(w):
            w.emit_run(payload)
        w.emit_state("running", "completion_artifact", self.manifest_path or "", event_utc=str(payload["started_utc"]),
                     scheduler_state=os.environ.get("SLURM_JOB_STATE"))
        w.push()

    @staticmethod
    def _run_recorded(w: emit.RecordWriter) -> bool:
        """Did an earlier STORE recorder of this spool already open the run -- a ``run`` record in the writer's run directory whose
        payload carries ``extra.store_key``, the mark ONLY :meth:`run_payload` writes? Neither the submit WRAPPER's ``queued`` record
        (``write_planned_run``; the cluster path shares ``$SHARED/registry`` between wrapper and job) nor the job-start REGISTRATION's
        ``running`` record (``emit_cli register --from-config``, the pending-file path the Petri / misalignment drivers run under -- fr8:
        a lifecycle-state heuristic mistook it for a store opening and the stores minted nothing) counts: the store's own ``run``
        payload carries the facts neither can know (the hashes, the model revision, the store uri, the route) and is minted over them.
        A disabled writer (no spool) answers False; an unreadable record is not a prior opening."""
        d = getattr(w, "dir", None)
        if not d or not os.path.isdir(d):
            return False
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return False
        for fn in names:
            m = R.RECORD_FILE_RE.match(fn)
            if not m or m.group("type") != "run":
                continue
            try:
                for _i, rec, _sha in R.read_record_file(os.path.join(d, fn), compute_sha=False):
                    payload = rec.get("payload") if isinstance(rec, Mapping) else None
                    extra = payload.get("extra") if isinstance(payload, Mapping) else None
                    if isinstance(extra, Mapping) and extra.get("store_key"):
                        return True
            except Exception:  # noqa: BLE001 -- an unreadable record is not evidence of an opening
                continue
        return False

    def _dedupe_by_run(self, routes: Iterable[str]) -> List[str]:
        """Several route keys (one per producer spelling) name ONE run; open / close it once."""
        seen: set = set()
        out: List[str] = []
        for r in routes:
            rid = self._run_id(self.identities[r])
            if rid in seen:
                continue
            seen.add(rid)
            out.append(r)
        return out

    def _touched_routes(self) -> List[str]:
        """The routes whose runs this store opened or filed rows for — what ``on_results`` closes."""
        routes = list(self.run_payloads) + [r for r in self.writers if r not in self.run_payloads]
        return self._dedupe_by_run(routes)

    def on_row(self, row: Mapping[str, Any]) -> None:
        if self._derived_control_object:
            return                      # the derived control leg files nothing
        if dataclasses.is_dataclass(row):
            row = dataclasses.asdict(row)   # type: ignore[arg-type]
        metric = str(row.get("metric", ""))
        domain = row.get("domain")
        if domain == "_meta" or metric.startswith("ts_"):
            try:
                self._ts[metric] = float(row.get("value"))
            except (TypeError, ValueError):
                pass
            return
        route = self._route(row.get("task"))
        if route is None:
            t = str(row.get("task"))
            self._unrouted[t] = self._unrouted.get(t, 0) + 1
            if self._unrouted[t] == 1:
                notice(f"store {self.store_key}: rows of task {t!r} (object {self._object()!r}) match no registered run; not recorded "
                       "(register the eval for this job, or pass registry_identity=)")
            return
        self._open(route)
        rid = self._writer(route).run_id
        created = row.get("created_at")
        try:
            created_utc = epoch_ms_to_utc(int(float(created) * 1000))
        except (TypeError, ValueError):
            created_utc = utc_now()
        item = str(row.get("item_id"))
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            value = None
        if item == AGGREGATE_ITEM_ID or metric.endswith(AGGREGATE_SUFFIX):
            # the producer's whole-run aggregate is a run_metric (the adapter's rule and spelling), not an item
            self._metrics.setdefault(route, []).append(aggregate_metric_row(rid, metric, value, row.get("extra_json"), created_utc,
                                                                            task=row.get("task")))
            if len(self._metrics[route]) >= self.FLUSH_EVERY:
                self.flush(route)
            return
        extra = {"task": row.get("task"), "task_version": row.get("task_version"), "domain": domain, "tier": row.get("tier"),
                 "attempt": row.get("attempt"), "seed": row.get("seed"), "position_order": row.get("position_order"),
                 "wall_clock_s": row.get("wall_clock_s"), "row_extra_json": row.get("extra_json")}
        self._items.setdefault(route, []).append({
            "run_id": rid, "item_id": item, "metric_id": metric, "trial_index": int(row.get("trial_id", 0) or 0),
            "sample_index": 0, "turn_index": -1, "judgment_id": "", "value": value, "label": None, "api_pool": None,
            "created_utc": created_utc, "judged_utc": None, "source_revision_id": "", "withdrawn_by_revision_id": None,
            "extra": extra})
        self._item_ids.setdefault(route, set()).add(item)
        flags = flags_from_row_extra(row.get("extra_json"))
        if flags and item not in self._flagged_items.setdefault(route, set()):
            self._flagged_items[route].add(item)
            self._flags.setdefault(route, []).append(item_flags_row(rid, item, int(row.get("trial_id", 0) or 0), flags))
        if len(self._items[route]) >= self.FLUSH_EVERY:
            self.flush(route)

    def on_generated(self, task: str, item_ids: Sequence[str], uri: str) -> None:
        """A judged task's GENERATED items (``CellStore.note_generated``): ONE null-valued ``item_result``
        (``metric_id`` :data:`GENERATED_METRIC_ID`, ``judgment_id ''``) per item the task generated and has no row for
        yet, on the run the task routes to -- opened here if this is its first evidence -- and FLUSHED at once (a judged
        store reaches ``on_results`` only on the grading host, never in the container). At completion the shard union
        then holds the plan's item count for a run whose rows the judge files later; before this, xstest / or_bench runs
        held serving + completion patches alone and every one REFUSED admission. An item that already has a row is left
        alone; a task no registered run claims is counted and named once, as ``on_row`` names it."""
        if self._derived_control_object:
            return
        route = self._route(task)
        if route is None:
            t = str(task)
            self._unrouted[t] = self._unrouted.get(t, 0) + 1
            if self._unrouted[t] == 1:
                notice(f"store {self.store_key}: generated items of task {t!r} (object {self._object()!r}) match no registered run; not recorded")
            return
        self._open(route)
        rid = self._writer(route).run_id
        now = utc_now()
        have = self._item_ids.setdefault(route, set())
        n_new = 0
        for item in item_ids:
            item = str(item)
            if not item or item in have:
                continue
            have.add(item)
            n_new += 1
            self._items.setdefault(route, []).append({
                "run_id": rid, "item_id": item, "metric_id": GENERATED_METRIC_ID, "trial_index": 0, "sample_index": 0, "turn_index": -1,
                "judgment_id": "", "value": None, "label": None, "api_pool": None, "created_utc": now, "judged_utc": None,
                "source_revision_id": "", "withdrawn_by_revision_id": None,
                "extra": {"task": str(task), "stage": "generated", "generation_uri": str(uri)}})
        if n_new:
            self.flush(route)

    def flush(self, route: Optional[str] = None) -> None:
        routes = [route] if route else list(dict.fromkeys(list(self._items) + list(self._flags) + list(self._metrics)))
        for r in routes:
            rows = self._items.pop(r, [])
            if rows:
                self._writer(r).emit_items(rows)
            flags = self._flags.pop(r, [])
            if flags:
                self._writer(r).emit_item_flags(flags)
            metrics = self._metrics.pop(r, [])
            if metrics:
                self._writer(r).emit_metrics(metrics)
            if rows or flags or metrics:
                self._writer(r).push()

    def on_generation_complete(self, marker_path: str) -> None:
        if self._derived_control_object:
            return
        self.flush()
        for route in self._touched_routes():
            self._writer(route).emit_patch(fields={"extra_json": {"generation_complete_utc": utc_now(), "generation_complete_uri": str(marker_path)}})

    def walltime(self) -> Optional[float]:
        """``store_span``: the span of the store's own ``ts_*`` timestamp rows (≥ 2 stamps), else None."""
        if len(self._ts) < 2:
            return None
        return round(max(self._ts.values()) - min(self._ts.values()), 3)

    def on_results(self, summary: Mapping[str, Any], results_path: str) -> None:
        if self._derived_control_object:
            self._finished = True
            return                      # never a run, so never `generated`
        self.flush()
        results_path = str(results_path)
        sha = _sha256_file(results_path) if os.path.isfile(results_path) else None
        now = utc_now()
        wall = self.walltime()
        rows_path = os.path.join(os.path.dirname(results_path), "rows.jsonl")
        for route in self._touched_routes():
            w = self._writer(route)
            w.emit_state("generated", "completion_artifact", results_path, event_utc=now)
            fields: Dict[str, object] = {
                "n_items_generated": len(self._item_ids.get(route, ())),
                "store_completion_artifact_uri": results_path,
                "store_uri": os.path.dirname(results_path),
            }
            if sha:
                fields["store_completion_sha256"] = sha
            if wall is not None:
                fields["walltime_s"] = wall
                fields["walltime_source"] = "store_span"
            if not self._decode_filed.get(route) and self._decode_recorder_eligible.get(route):
                # decode-recording (a'): the manifest-time read filed nothing for an ATOM route (a thinking_card / gsm8k cell: the sampler is
                # applied server-side) and the standup's request recorder may have landed since (`<store root>/resolved_sampler.json`,
                # written at standup end): the completion patch then carries `sent` (27b) + `resolved` + their entries; otherwise it is
                # byte-identical to before. A legacy-control route (mmlu / xstest / or_bench_hard_1k) never reads it: its request is the
                # driver's, the manifest states it or states nothing, and the recorder describes the atoms'. The document's DECLARED
                # reference (sent_rows 0) derives the sent set against the pinned request builder at THIS producer's code sha (`derived`,
                # never `recorded`); the resolved values stay recorded. A producer block whose code sha is `unverified` pins nothing, so
                # the reader files no sent column and the notice says so -- the run's own code_sha column reads `unverified` beside it.
                # The server-resolved sampler the document OBSERVED is filed either way: a document that resolved one sampler under AGREES
                # but proves no sent set (no pin for its declared reference; two client-row samplers; a reference naming no key) still
                # puts the resolved column (`recorded`), the recorder's reason on the sent entry (the fold replaces the manifest-time
                # entry per column) and the evidence block -- the declared keys among its witnesses -- on the completion patch; the
                # sent columns stay empty, so the route stays unfiled. The cellstore adapter files the same document the same way. A
                # document that resolved nothing of this run (a non-agreeing verdict, a task it does not cover) leaves the patch as it
                # was: the manifest-time reason stands and the notice says why.
                try:
                    dev = DE.from_recorder_document(os.path.dirname(results_path), task=self._decode_spelling.get(route),
                                                    code_sha=self.producer["code_sha"])
                    if dev.recorded():
                        fields.update(dev.patch_fields())
                        self._decode_filed[route] = True
                    elif dev.resolved_read():
                        fields.update(dev.patch_fields())
                        notice(f"store {self.store_key}: the recorder document fills no sent column for route {route!r}: {dev.why} -- the "
                               f"completion patch carries the server-resolved sampler (recorded) and the reason")
                    elif dev.evidence_path:
                        notice(f"store {self.store_key}: the recorder document fills no decode column for route {route!r}: {dev.why}")
                except Exception as exc:  # noqa: BLE001 -- a lost fact never costs a run
                    notice(f"store {self.store_key}: recorder document not read ({type(exc).__name__}: {exc}); the completion patch carries no decode columns")
            w.emit_patch(fields=fields)
            arts = []
            if sha:
                arts.append({"artifact_id": sha, "kind": "completion", "uri_primary": results_path, "bytes": os.path.getsize(results_path),
                             "created_utc": now, "locations_json": [results_path], "extra": {"run_id": w.run_id}})
            if os.path.isfile(rows_path):
                rsha = _sha256_file(rows_path)
                arts.append({"artifact_id": rsha, "kind": "rows", "uri_primary": rows_path, "bytes": os.path.getsize(rows_path),
                             "created_utc": now, "locations_json": [rows_path], "extra": {"run_id": w.run_id}})
            for a in arts:
                w.emit_artifact(a)
            deg = not_measured_rows(w.run_id, self.declared_degeneration, computed=(), computed_utc=now)
            if deg:
                w.emit_degeneration(deg)
            w.push(final=True)
        self._finished = True

    def _atexit(self) -> None:
        try:
            if any(self._items.values()) or any(self._flags.values()) or any(self._metrics.values()):
                self.flush()
            if not self._finished:
                for w in self.writers.values():
                    w.push(final=True)
        except Exception as exc:  # noqa: BLE001
            notice(f"store {self.store_key}: exit flush failed ({type(exc).__name__}: {exc})")


def aggregate_metric_row(run_id: str, metric: str, value: Optional[float], extra_json: Optional[str], computed_utc: str,
                         *, task: Optional[object] = None) -> Dict[str, object]:
    """The ``run_metric`` row of a producer's aggregate store row -- the cellstore ADAPTER's exact shape, so a natively
    recorded store and a re-read legacy store mint the same registry-grain metric: ``metric_id`` = the producer's spelling minus
    ``_aggregate``, ``producer_metric_spelling`` = that spelling, ``n`` from the row's ``extra_json`` when it is an int, the remaining
    extra facts under ``extra``, ``computed_by producer``, ``judgment_id ''`` (a generation-time aggregate; the judge's own carry theirs)."""
    ex: Dict[str, object] = {}
    if isinstance(extra_json, str) and extra_json:
        try:
            parsed = json.loads(extra_json)
            ex = dict(parsed) if isinstance(parsed, dict) else {}
        except ValueError:
            ex = {}
    n = ex.get("n")
    return {"run_id": run_id, "judgment_id": "",
            "metric_id": metric[:-len(AGGREGATE_SUFFIX)] if metric.endswith(AGGREGATE_SUFFIX) else metric,
            "threshold": -1.0, "computation_id": "", "producer_metric_spelling": metric, "value": value,
            "n": int(n) if isinstance(n, int) and not isinstance(n, bool) else None, "computed_by": "producer", "computed_utc": computed_utc,
            "source_uri": None, "extra": dict({k: v for k, v in ex.items() if k != "n"}, **({"task": task} if task is not None else {}))}


def flags_from_row_extra(extra_json: Optional[str]) -> Dict[str, object]:
    """Best-effort ``item_flags`` facts from a row's ``extra_json`` (at cap, finish reason, tokens,
    chars); empty when the row carries none of the recognised spellings."""
    if not extra_json or not isinstance(extra_json, str):
        return {}
    try:
        d = json.loads(extra_json)
    except ValueError:
        return {}
    if not isinstance(d, dict):
        return {}
    out: Dict[str, object] = {}
    for k in ("at_cap", "at_token_cap", "hit_cap", "truncated"):
        if k in d and d[k] is not None:
            out["at_token_cap"] = int(bool(d[k]))
            break
    for k in ("finish_reason", "stop_reason"):
        if isinstance(d.get(k), str):
            out["finish_reason"] = d[k]
            break
    for k in ("response_tokens", "completion_tokens", "n_tokens", "gen_tokens", "tokens"):
        if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool):
            out["response_tokens"] = int(d[k])
            break
    for k in ("response_chars", "n_chars", "chars"):
        if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool):
            out["response_chars"] = int(d[k])
            break
    return out


def item_flags_row(run_id: str, item_id: str, trial_index: int, flags: Mapping[str, object],
                   sample_index: int = 0, turn_index: int = -1, judgment_id: str = "") -> Dict[str, object]:
    """A WHOLE ``item_flags`` row: every column present, unknown ones null."""
    row: Dict[str, object] = {c: None for c in R.ITEM_WHOLE_ROW_COLUMNS["item_flags"]}
    row.update({"run_id": run_id, "item_id": item_id, "trial_index": trial_index, "sample_index": sample_index,
                "turn_index": turn_index, "judgment_id": judgment_id, "source_revision_id": ""})
    for k, v in flags.items():
        if k in row:
            row[k] = v
    return row


def not_measured_rows(run_id: str, declared: Iterable[Tuple[str, str]], computed: Iterable[Tuple[str, str]],
                      computed_utc: Optional[str] = None, judgment_id: str = "",
                      computed_by: str = "producer") -> List[Dict[str, object]]:
    """One WHOLE ``run_degeneration`` row with ``measurement_state = not_measured`` for every
    declared (measure, scope) the producer did not compute — enumerate the declared set and subtract."""
    done = set(tuple(c) for c in computed)
    out = []
    for measure, scope in declared:
        if (measure, scope) in done:
            continue
        row: Dict[str, object] = {c: None for c in R.ITEM_WHOLE_ROW_COLUMNS["run_degeneration"]}
        row.update({"run_id": run_id, "judgment_id": judgment_id, "measure": measure, "scope": scope, "source_revision_id": "",
                    "measurement_state": "not_measured", "aggregation": _AGGREGATION_OF_MEASURE.get(measure, "single_detector"),
                    "definition_version": DEGENERATION_DEFINITION_VERSION, "computed_by": computed_by,
                    "computed_utc": computed_utc or utc_now(), "pooling_json": {}, "threshold_params_json": {},
                    "notes": "declared by the producer, not computed at this step"})
        out.append(row)
    return out


def measured_row(run_id: str, measure: str, scope: str, numerator: int, denominator: int, denominator_unit: str,
                 judgment_id: str = "", computed_utc: Optional[str] = None, computed_by: str = "producer",
                 threshold_params: Optional[Mapping[str, object]] = None, source_uri: Optional[str] = None) -> Dict[str, object]:
    """A WHOLE measured ``run_degeneration`` row (rate = numerator / denominator)."""
    row: Dict[str, object] = {c: None for c in R.ITEM_WHOLE_ROW_COLUMNS["run_degeneration"]}
    row.update({"run_id": run_id, "judgment_id": judgment_id, "measure": measure, "scope": scope, "source_revision_id": "",
                "measurement_state": "measured",
                "numerator": int(numerator), "denominator": int(denominator), "denominator_unit": denominator_unit,
                "rate": (float(numerator) / float(denominator)) if denominator else None,
                "aggregation": _AGGREGATION_OF_MEASURE.get(measure, "single_detector"),
                "definition_version": DEGENERATION_DEFINITION_VERSION, "computed_by": computed_by,
                "computed_utc": computed_utc or utc_now(), "pooling_json": {}, "threshold_params_json": dict(threshold_params or {}),
                "source_uri": source_uri})
    if denominator == 0:
        row["measurement_state"] = "not_applicable"
        row["numerator"] = row["denominator"] = row["denominator_unit"] = row["rate"] = None
    return row


def store_recorder(store_key: str, resolved_spec: Mapping[str, Any], registry_identity: Optional[Mapping[str, object]] = None,
                   *, registry_emit: Optional[bool] = None, spool: Optional[str] = None, push_script: Optional[str] = None,
                   script: str = "capeval.runtime.resolve_cell", session: Optional[str] = None,
                   declared_degeneration: Sequence[Tuple[str, str]] = DEFAULT_DECLARED_DEGENERATION,
                   row_constants: Optional[Mapping[str, Any]] = None) -> Optional[StoreRecorder]:
    """The recorder ``runtime.resolve_cell`` hands its ``CellStore`` — None when emission is off, when
    no spool resolves, or when no usable identity is supplied (one notice naming what is missing).
    ``row_constants``: the store's cell-constant row columns, the fallback carrier of ``run.seed_base``."""
    if not emission_enabled(registry_emit):
        return None
    spool_path = resolve_spool(spool)
    if not spool_path:
        notice(f"store {store_key}: emission on but no spool resolves (set SISTEER_REGISTRY_SPOOL); not recorded")
        return None
    idents = normalise_identities(registry_identity if registry_identity is not None else identity_from_env())
    if not idents:
        notice(f"store {store_key}: no registry identity (eval_id, item_set_id, instrument_id, …) — pass registry_identity= "
               f"or export {RUN_IDENTITY_ENV}=<planned-run.json>; not recorded")
        return None
    bad = {r: missing_identity_fields(i) for r, i in idents.items() if missing_identity_fields(i) and not i.get("run_id")}
    if bad:
        notice(f"store {store_key}: registry identity incomplete, not recorded: " +
               "; ".join(f"route {r!r} lacks {m}" for r, m in bad.items()))
        return None
    try:
        return StoreRecorder(store_key, resolved_spec, idents, spool=spool_path, push_script=push_script, script=script,
                             session=session, declared_degeneration=declared_degeneration, row_constants=row_constants)
    except Exception as exc:  # noqa: BLE001
        notice(f"store {store_key}: recorder not built ({type(exc).__name__}: {exc}); not recorded")
        return None


_DEFAULT_NOTICED: set = set()


def default_store_recorder(store_key: str, row_constants: Optional[Mapping[str, Any]], *, script: str = "capeval.store.CellStore",
                           session: Optional[str] = None) -> Optional[StoreRecorder]:
    """What ``CellStore.__init__`` resolves when it is handed no recorder (the default since 2026-09-17): the one
    switch, then the identity from the job's pending file (``$SISTEER_REGISTRY_RUN_IDENTITY``). None — exactly the
    old behaviour — when emission is off, no spool resolves, or the job registered nothing; the reason is printed
    ONCE per process per store key so a wave of stores does not repeat it. ``row_constants`` stands in for the
    resolved spec until ``on_manifest`` replaces it (they carry ``arm_name`` / ``kind`` / ``intervention_hash`` —
    the object discriminator — in every harness store)."""
    if not emission_enabled():
        return None
    if not os.environ.get(RUN_IDENTITY_ENV):
        if store_key not in _DEFAULT_NOTICED:
            _DEFAULT_NOTICED.add(store_key)
            notice(f"store {store_key}: emission on but {RUN_IDENTITY_ENV} is unset (the job did not register itself); not recorded")
        return None
    return store_recorder(store_key, dict(row_constants or {}), None, script=script, session=session, row_constants=row_constants)


# --------------------------------------------------------------------------------------------
# a producer that generates without a store (the misalignment runner's responses.jsonl)
# --------------------------------------------------------------------------------------------


@guarded("generation complete")
def emit_generation_complete(object_entry: str, tasks: Sequence[str], marker_path: str, *, n_generated: Optional[int] = None,
                             script: str, registry_emit: Optional[bool] = None, spool: Optional[str] = None,
                             session: Optional[str] = None, extra: Optional[Mapping[str, object]] = None,
                             code_sha: Optional[str] = None) -> List[str]:
    """One ``run_patch`` (``extra_json.generation_complete_utc`` / ``_uri``, ``n_items_generated`` when known) per
    registered run that claims (``object_entry``, one of ``tasks``) in the job's pending identity — the generation
    step of a producer whose rows are filed later by another writer (the misalignment runner writes
    ``responses.jsonl``; ``misalign.rows`` files the store). Returns the paths written; empty when off / unrouted."""
    if not emission_enabled(registry_emit):
        return []
    spool_path = resolve_spool(spool)
    if not spool_path:
        return []
    idents = normalise_identities(identity_from_env())
    if not idents:
        return []
    producer = emit.producer_block(script, code_sha, session)   # fr5 (5): the caller's tree sha when it knows it (the Modal app does), else git
    out: List[str] = []
    seen: set = set()
    for t in tasks:
        cands = [f"{object_entry}{ROUTE_SEP}{t}", t]
        if "/" in t:
            cands += [f"{object_entry}{ROUTE_SEP}{t.rsplit('/', 1)[1]}", t.rsplit("/", 1)[1]]
        ident = next((idents[c] for c in cands if c in idents), idents.get("*"))
        if not ident or not ident.get("run_id") or ident["run_id"] in seen:
            continue
        seen.add(ident["run_id"])
        w = emit.RecordWriter(spool_path, str(ident.get("campaign") or identity.parse_run_id(str(ident["run_id"]))["campaign"]),
                              str(ident["run_id"]), producer)
        fields: Dict[str, object] = {"extra_json": dict({"generation_complete_utc": utc_now(), "generation_complete_uri": str(marker_path)},
                                                        **(dict(extra) if extra else {}))}
        if n_generated is not None:
            fields["n_items_generated"] = int(n_generated)
        p = w.emit_patch(fields=fields)
        if p:
            out.append(p)
        w.push()
    if not seen:
        notice(f"{script}: no registered run claims object {object_entry!r} for tasks {list(tasks)}; generation not recorded")
    return out


# --------------------------------------------------------------------------------------------
# the served-run identity patch (native standup ``write_endpoint``; S7 verify-and-refuse)
# --------------------------------------------------------------------------------------------

RUNS_FILE = "registry_runs.json"


def serving_patch_payload(endpoint: Mapping[str, Any], served_key: str, run: Mapping[str, object]) -> Dict[str, object]:
    """The ``run_patch`` for ONE served run from the standup's ``endpoint.json`` record: the OBSERVED
    serving facts under ``fields`` and the IMMUTABLE re-observations under ``verify`` (the harvester
    compares them to the run row and REFUSES the patch on any mismatch — a standup that served a run
    under a different path than it was minted with is a producer defect, not an update).

    ``run`` = ``{"run_id", "expected": {serving_path?, intervention_hash?, harness_config_hash?,
    model_revision?, env_digest?}}``: what the run was minted with; a fact the endpoint observes itself
    (``engine`` → serving_path, the adapter block's intervention hash, ``model_revision``) is taken from
    the endpoint, the rest from ``expected`` (the caller's re-observation of its own immutable facts).

    When the endpoint's ``served_positions`` names the served key, the patch stamps ``applied_positions_mode``
    (:func:`served_positions_mode` for the one served mode, :func:`served_mode_for_registry` for the registry's
    reading of it against the object's kind -- the entry's ``object_kind``, an ``object_id`` that spells one, else the
    adapter the endpoint attests for the key -- and the serving path) AND the ``multi_turn_steering_scope`` that
    follows it (:func:`served_scope`), each with its confidence entry under ``fields.field_confidence_json`` -- the
    mode's ``recorded``, ``derived`` where the kind or the lens spelling decided it, or ``unverified`` with the reason
    when the served list is not one mode or the served mask cannot be read without a kind nothing names; the scope's
    derivation, or ``unverified`` with the reason when the eval's category cannot be read. The entry may carry
    ``eval_id``, ``eval_category``, ``object_id`` or ``object_kind`` to say what the derivations would otherwise read
    from the run id, the endpoint and the producing checkout's seeds.
    """
    fields: Dict[str, object] = {}
    verify: Dict[str, object] = {}
    att_seqs = endpoint.get("attested_max_num_seqs")
    if att_seqs is not None:
        fields["served_max_num_seqs"] = int(att_seqs)
        fields["served_max_num_seqs_source"] = "attested"
    elif endpoint.get("max_num_seqs") is not None:
        fields["served_max_num_seqs"] = int(endpoint["max_num_seqs"])
        fields["served_max_num_seqs_source"] = "runner_concurrency"
    window = endpoint.get("attested_max_model_len", endpoint.get("max_model_len"))
    if window is not None:
        fields["context_window_tokens"] = int(window)
    versions = endpoint.get("serving_versions")
    eng_id = endpoint.get("engine_identity")
    if not isinstance(versions, dict) and isinstance(eng_id, dict):
        versions = eng_id.get("versions") if isinstance(eng_id.get("versions"), dict) else eng_id
    if isinstance(versions, dict) and versions.get("vllm"):
        fields["serving_engine_version"] = f"vllm {versions['vllm']}"
    for src_key, dst in (("serving_identity", "serving_identity"), ("run_identity", "serving_identity"),
                         ("serving_config_sha256", "serving_config_sha256")):
        v = endpoint.get(src_key)
        if isinstance(v, str) and v and dst not in fields:
            fields[dst] = v
    adapters = endpoint.get("adapters") if isinstance(endpoint.get("adapters"), dict) else {}
    block = adapters.get(served_key) if isinstance(adapters, dict) else None
    if isinstance(block, dict):
        fields["adapter_fingerprint"] = hashlib.sha256(identity.canonical_json(block).encode("utf-8")).hexdigest()
        layers = block.get("layers") or block.get("target_layers")
        if isinstance(layers, list):
            fields["applied_layers_json"] = layers
        ih = block.get("petri_intervention_hash") or block.get("intervention_hash")
        if isinstance(ih, str) and ih:
            verify["intervention_hash"] = ih
    engine = endpoint.get("engine")
    if isinstance(engine, str) and engine:
        verify["serving_path"] = ENGINE_TO_SERVING_PATH.get(engine, engine)
    expected = run.get("expected") if isinstance(run.get("expected"), Mapping) else {}
    positions = endpoint.get("served_positions")
    confidence: Dict[str, object] = {}
    if isinstance(positions, dict) and served_key in positions:
        # the proxy writes and validates `served_positions` as {<served key>: [mode, ...]} (a non-empty LIST per key), the older
        # standups a bare string or a {mode} dict; one distinct mode, whatever the shape, is the served fact and wins over the job-start
        # derivation -- read against the object's kind and the serving path, because the proxy spells the MASK it applies within a
        # forward (its `all` is every prompt row on a prefill-only AlphaSteer or lens static direction, every token on a LoRA); a list
        # that is not one mode, or a kind-dependent mask with no kind to read it against, stamps `unverified` with the reason in the
        # patch's own confidence entry (the validator reads it there; the fold merges a patch's entries into the row's per column),
        # never its first element and never every-token steering by default
        served, positions_reason = served_positions_mode(positions.get(served_key))
        kind = served_run_object_kind(run, endpoint, served_key)
        if served is not None:
            serving_path = verify.get("serving_path") or (expected.get("serving_path") if isinstance(expected.get("serving_path"), str) else None)
            kind_source = served_adapter_kind(endpoint, served_key)[1] if kind is None else None
            fields["applied_positions_mode"], confidence["applied_positions_mode"] = served_mode_for_registry(
                served, kind, serving_path or None, served_key, kind_source)
        elif positions_reason is not None:
            fields["applied_positions_mode"] = _UNVERIFIED
            confidence["applied_positions_mode"] = _confidence_entry(positions_reason)
        if "applied_positions_mode" in fields:
            # the multi-turn steering scope follows the mode: re-derived here from the served mode, the eval's category and the object's
            # kind, so a restricted-position standup over a registration on the every-token instrument moves the scope with the mode
            fields["multi_turn_steering_scope"], confidence["multi_turn_steering_scope"] = served_scope(str(fields["applied_positions_mode"]), run, kind)
    pc = endpoint.get("prefix_caching", endpoint.get("prefix_cache_policy"))
    if isinstance(pc, str) and pc in ("on", "off", "skip_reading"):
        fields["prefix_cache_policy"] = pc
    elif isinstance(pc, bool):
        fields["prefix_cache_policy"] = "on" if pc else "off"
    rev = endpoint.get("model_revision") or endpoint.get("revision")
    if isinstance(rev, str) and len(rev) == _SHA1_LEN:
        verify["model_revision"] = rev
    for k in R.RUN_VERIFIED_FIELDS:
        if k not in verify and expected.get(k) not in (None, ""):
            verify[k] = expected[k]
    fields["extra_json"] = {"served_key": served_key, "endpoint_attempt": endpoint.get("attempt_id") or endpoint.get("standup_attempt"),
                            "proxy_url": endpoint.get("proxy_url"), "node": endpoint.get("node"),
                            "slurm_job_id": endpoint.get("slurm_job_id"), "standup_front": endpoint.get("standup_front")}
    if confidence:
        fields["field_confidence_json"] = confidence
    return {"run_id": str(run["run_id"]), "fields": fields, "verify": verify}


def load_runs_file(out_dir: str) -> Optional[Dict[str, Dict[str, object]]]:
    """``<out>/registry_runs.json``: ``{served key: {run_id, campaign, expected: {...}}}`` written by
    the launching shell from the planned runs; None when absent. ADDITIVE (2026-09-19, the seedrep battery's
    combined containers): a served key may carry a LIST of such run entries when one served route is
    measured by several registered runs (the controls store's xstest + or_bench + mmlu, Petri's six behaviour
    runs); :func:`emit_serving_patches` writes one ``run_patch`` per entry, keyed by the same served key."""
    p = os.path.join(str(out_dir), RUNS_FILE)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


@guarded("serving standup")
def emit_serving_patches(endpoint: Mapping[str, Any], runs: Optional[Mapping[str, Mapping[str, object]]] = None,
                         *, out_dir: Optional[str] = None, spool: Optional[str] = None, push_script: Optional[str] = None,
                         registry_emit: Optional[bool] = None, script: str = "capeval.serving.native_standup.write_endpoint",
                         session: Optional[str] = None) -> List[str]:
    """The ``write_endpoint`` hook: one ``run_patch`` per served run named in ``runs`` (else in
    ``<out_dir>/registry_runs.json``). Returns the record paths written (empty when off / no runs)."""
    if not emission_enabled(registry_emit):
        return []
    spool_path = resolve_spool(spool)
    if not spool_path:
        notice("serving standup: emission on but no spool resolves; serving facts not recorded")
        return []
    if runs is None and out_dir:
        runs = load_runs_file(out_dir)
    if not runs:
        notice(f"serving standup: no run identities for the served keys (pass registry_runs or write {RUNS_FILE} beside endpoint.json); "
               "serving facts not recorded")
        return []
    producer = emit.producer_block(script, session=session)
    out: List[str] = []
    for served_key, entry in runs.items():
        # one run per served key (the original shape) or a LIST of runs sharing one served route (additive, see load_runs_file)
        entries = list(entry) if isinstance(entry, (list, tuple)) else [entry]
        for run in entries:
            if not isinstance(run, Mapping) or not run.get("run_id"):
                notice(f"serving standup: run entry for {served_key!r} has no run_id; skipped")
                continue
            p = _serving_patch_for(endpoint, served_key, run, spool_path, producer, push_script)
            if p:
                out.append(p)
    return out


def _serving_patch_for(endpoint: Mapping[str, Any], served_key: str, run: Mapping[str, object], spool_path: str,
                       producer: Mapping[str, object], push_script: Optional[str]) -> Optional[str]:
    """ONE served run's ``run_patch`` (the body :func:`emit_serving_patches` runs per entry); None when the endpoint
    observes nothing for it."""
    payload = serving_patch_payload(endpoint, served_key, run)
    if not payload["fields"] and not payload["verify"]:
        return None
    w = emit.RecordWriter(spool_path, str(run.get("campaign") or identity.parse_run_id(str(run["run_id"]))["campaign"]),
                          str(run["run_id"]), producer, push_script=push_script)
    p = w.emit_patch(fields=payload["fields"], verify=payload["verify"])
    w.push()
    return p


# --------------------------------------------------------------------------------------------
# judging (the grading drivers' 3–5 lines)
# --------------------------------------------------------------------------------------------


class JudgingScope:
    """``judgment`` claim → judge item rows / metrics / degeneration → ``judgment complete|partial``
    + ``run_state judged`` + ``run_patch`` (n_items_judged, judge_id) + ``not_measured`` rows + final push.

    Use as a context manager: an exception inside the block files ``status = partial`` (never
    swallowed — the driver's own error handling is untouched)."""

    def __init__(self, run_id: str, campaign: str, judge_id: str, *, api_pool: Optional[str] = None, session: Optional[str] = None,
                 spool: Optional[str] = None, push_script: Optional[str] = None, script: str = "judging",
                 declared_degeneration: Sequence[Tuple[str, str]] = DEFAULT_DECLARED_DEGENERATION,
                 rubric_sha256: Optional[str] = None, store_uri: Optional[str] = None, disabled: bool = False,
                 claim: Optional[Mapping[str, object]] = None) -> None:
        self.run_id = run_id
        self.campaign = campaign
        self.judge_id = judge_id
        self.api_pool = api_pool
        self.session = session if session is not None else (emit.session_from_env() or f"{os.environ.get('USER', 'user')}@{os.uname().nodename}")
        # a claim registered through the registry (``pending.claim``: the same judgment row, already ingested
        # under the harvester lock) binds this scope to ITS judgment_id / claimed_utc, so the completion record
        # folds onto the claimed row rather than opening a second judgment
        self.claim_registered = bool(claim)
        self.claimed_utc = str(claim["claimed_utc"]) if claim and claim.get("claimed_utc") else utc_now()
        self.judgment_id = (str(claim["judgment_id"]) if claim and claim.get("judgment_id")
                            else identity.judgment_id(run_id, judge_id, self.session, self.claimed_utc))
        self.writer = emit.RecordWriter(spool, campaign, run_id, emit.producer_block(script, session=session),
                                        push_script=push_script, disabled=disabled)
        self.declared_degeneration = tuple(declared_degeneration)
        self.computed_degeneration: List[Tuple[str, str]] = []
        self.n_items = 0
        self.n_refusals = 0
        self.n_errors = 0
        self.rubric_sha256 = rubric_sha256
        self.store_uri = store_uri
        self.closed = False
        #: set ONLY after the completion record is written: `closed` flips before the write, so a caller that must know
        #: whether the judgment is durable -- the registry-worklist analyze's claim cleanup -- keys on this, never on `closed`
        self.completion_emitted = False

    def _judgment(self, status: str, **more) -> Dict[str, object]:
        p: Dict[str, object] = {"judgment_id": self.judgment_id, "run_id": self.run_id, "judge_id": self.judge_id, "status": status,
                                "claimed_by_session": self.session, "claimed_utc": self.claimed_utc, "api_pool": self.api_pool,
                                "grading_code_sha": self.writer.producer["code_sha"], "producer_session": self.session,
                                "store_uri": self.store_uri}
        if self.rubric_sha256:
            p["rubric_sha256"] = self.rubric_sha256
        p.update(more)
        return {k: v for k, v in p.items() if v is not None}

    def claim(self) -> "JudgingScope":
        if not self.claim_registered:   # a registry claim already holds this judgment row in_progress
            self.writer.emit_judgment(self._judgment("in_progress", started_utc=self.claimed_utc))
            self.writer.push()
        return self

    __enter__ = claim

    def items(self, rows: Sequence[Mapping[str, object]]) -> Optional[str]:
        """Judge ``item_result`` rows — WHOLE rows; ``judgment_id`` / ``api_pool`` / ``judged_utc`` filled in."""
        now = utc_now()
        full = []
        for r in rows:
            row: Dict[str, object] = {c: None for c in R.ITEM_WHOLE_ROW_COLUMNS["item_result"]}
            row.update({"run_id": self.run_id, "trial_index": 0, "sample_index": 0, "turn_index": -1, "source_revision_id": "",
                        "judgment_id": self.judgment_id, "api_pool": self.api_pool, "judged_utc": now, "created_utc": now})
            row.update(dict(r))
            full.append(row)
        self.n_items += len({(str(r["item_id"]), r.get("trial_index", 0), r.get("sample_index", 0), r.get("turn_index", -1)) for r in full})
        return self.writer.emit_items(full)

    def metrics(self, rows: Sequence[Mapping[str, object]]) -> Optional[str]:
        """``run_metric`` rows with the scope's defaults filled in; the writer's best-of-N guard (:func:`vocab.bon_metric_refusal`,
        `emit.RecordWriter.emit_metrics`) then refuses a legacy `asr_at_n_max` row, an `asr_at_n<N>` row without `extra.bon_n`, or a
        best-of-N threshold outside the live conventions -- :class:`emit.MetricRefused`, nothing written."""
        now = utc_now()
        full = [dict({"run_id": self.run_id, "judgment_id": self.judgment_id, "computed_by": "producer", "computed_utc": now,
                      "threshold": -1.0, "computation_id": ""}, **dict(r)) for r in rows]
        return self.writer.emit_metrics(full)

    def degeneration(self, rows: Sequence[Mapping[str, object]]) -> Optional[str]:
        full = []
        for r in rows:
            row = dict(r)
            row.setdefault("judgment_id", self.judgment_id)
            row.setdefault("source_revision_id", "")      # natively emitted: the revision tag is the '' sentinel
            full.append(row)
            self.computed_degeneration.append((str(row["measure"]), str(row["scope"])))
        return self.writer.emit_degeneration(full)

    def complete(self, status: str = "complete", n_items_judged: Optional[int] = None, n_judge_refusals: Optional[int] = None,
                 n_errors: Optional[int] = None, **more) -> None:
        if self.closed:
            return
        self.closed = True
        now = utc_now()
        n = self.n_items if n_items_judged is None else int(n_items_judged)
        self.writer.emit_judgment(self._judgment(status, started_utc=self.claimed_utc, finished_utc=now, n_items_judged=n,
                                                 n_judge_refusals=self.n_refusals if n_judge_refusals is None else n_judge_refusals,
                                                 n_errors=self.n_errors if n_errors is None else n_errors, **more))
        self.completion_emitted = True       # the record is on the spool: the durable boundary (a raise above leaves it False)
        deg = not_measured_rows(self.run_id, self.declared_degeneration, self.computed_degeneration, computed_utc=now,
                                judgment_id=self.judgment_id)
        if deg:
            self.writer.emit_degeneration(deg)
        if status in ("complete", "partial"):
            self.writer.emit_state("judged", "completion_artifact", self.store_uri or "", event_utc=now)
            self.writer.emit_patch(fields={"n_items_judged": n, "judge_id": self.judge_id, "judged_utc": now})
        self.writer.push(final=True)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.n_errors += 1
            self.complete("partial")
        else:
            self.complete("complete")
        return False


def judgment_scope(driver: str, run: Mapping[str, object], judge: Mapping[str, object], *, api_pool: Optional[str] = None,
                   spool: Optional[str] = None, push_script: Optional[str] = None, session: Optional[str] = None,
                   registry_emit: Optional[bool] = None, store_uri: Optional[str] = None,
                   declared_degeneration: Sequence[Tuple[str, str]] = DEFAULT_DECLARED_DEGENERATION,
                   claim: Optional[Mapping[str, object]] = None) -> Optional[JudgingScope]:
    """The 3–5 lines a grading driver adds: ``scope = judgment_scope("scripts/x.py", run, judge, api_pool=args.org)``;
    ``with scope: scope.items(...); scope.metrics(...)``. ``run`` = ``{run_id, campaign}``; ``judge`` = either
    ``{judge_id}`` or the four identity fields (``judge_model``, ``rubric_sha256``, ``protocol_sha256``,
    ``threshold_rule``) from which the id is minted. ``claim`` = ``{judgment_id, claimed_utc}`` of a claim the
    driver registered through the registry (``pending.claim`` / the ``--worklist registry`` path): the scope
    then completes THAT judgment row instead of opening its own. None (a no-op) when emission is off."""
    if not emission_enabled(registry_emit):
        return None
    spool_path = resolve_spool(spool)
    if not spool_path:
        notice(f"{driver}: emission on but no spool resolves; judgment not recorded")
        return None
    if not run.get("run_id"):
        notice(f"{driver}: no run_id for the judged store; judgment not recorded")
        return None
    jid = str(judge.get("judge_id") or identity.judge_id(judge))
    campaign = str(run.get("campaign") or identity.parse_run_id(str(run["run_id"]))["campaign"])
    return JudgingScope(str(run["run_id"]), campaign, jid, api_pool=api_pool, session=session, spool=spool_path,
                        push_script=push_script, script=driver, declared_degeneration=declared_degeneration,
                        rubric_sha256=judge.get("rubric_sha256") if isinstance(judge.get("rubric_sha256"), str) else None,
                        store_uri=store_uri, claim=claim)


# --------------------------------------------------------------------------------------------
# objects from fit sidecars (the fit writers' hook; shared with the sidecar adapter)
# --------------------------------------------------------------------------------------------

#: ``fit_selection.json``: the fit's SELECTION -- ``{fit_search_id, fit_lr_selected_by,
#: lora_rank, lora_modules, per_expert_rank, architecture_selection_ruling_id, init_scheme, ...}`` -- serialised by the trainer into
#: its temp cell dir before publication (so the rename carries it into the published dir and the COMPLETE manifest hashes it) and read
#: here AUTHORITATIVELY when the caller passes no selection kwargs (the derived-object builders write their own from the parents').
FIT_SELECTION_SIDECAR = "fit_selection.json"
SIDECAR_NAMES: Tuple[str, ...] = ("train_budget_meta.json", "fit_meta.json", "fit_provenance.json", "derived_meta.json", "meta.json",
                                  "summary.json", "COMPLETE", FIT_SELECTION_SIDECAR)
#: the selection sidecar's keys → the `object` COLUMN each fills (the architecture lands in the REAL columns the schema has);
#: everything else in the sidecar rides ``extra`` (per_expert_rank, lora_set_id, trainable_params, init_scheme, the ruling id ...)
_FIT_SELECTION_COLUMNS: Dict[str, str] = {"fit_search_id": "fit_search_id", "fit_lr_selected_by": "fit_lr_selected_by",
                                          "lora_rank": "fit_rank", "fit_rank": "fit_rank", "fit_alpha": "fit_alpha",
                                          "lora_modules": "fit_target_modules_json", "fit_target_modules_json": "fit_target_modules_json",
                                          "fit_layers_json": "fit_layers_json"}
_OBJECT_CANDIDATES: Tuple[str, ...] = ("object.pt", "object.safetensors", "adapter_model.safetensors", "adapter_model.bin")
_KNOWN_TBM_KEYS = frozenset({"arm", "si_id", "positions_at_fit", "response_header_len", "batch_size", "resolved_batch_size", "max_len",
                             "epochs", "seed", "lr", "lr_schedule", "warmup_steps", "max_grad_norm", "length_sorted",
                             "teacher_prefix_cache", "trainer", "completion_tokens", "object_sha256", "model_id", "model_revision",
                             "revision_source", "template_hash", "site", "target_layers", "cell_id", "fit_record", "training_budget_kind",
                             "sequence_max_len", "sample_manifest_sha"})


def find_object_artifact(obj_dir: str) -> Optional[str]:
    for name in _OBJECT_CANDIDATES:
        p = os.path.join(obj_dir, name)
        if os.path.isfile(p):
            return p
    cands = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith((".pt", ".safetensors"))]
    cands = [c for c in cands if os.path.isfile(c)]
    return max(cands, key=os.path.getsize) if cands else None


def read_sidecars(obj_dir: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for name in SIDECAR_NAMES:
        p = os.path.join(obj_dir, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                out[name] = data if isinstance(data, dict) else {"_value": data}
            except ValueError:
                out[name] = {"_unparseable": True}
    return out


#: ``derivation_params_json`` of a derived object, built from its ``derived_meta.json`` the way the adapter-filed rows were
#: (adapters/fit_sidecars.py: every in-paper `composed` / `lora_subtraction` row carries exactly this shape). The native
#: derived emitters (psr_residual_distill compose-v / distill, lora_subtract) filed NO params before 2026-10-02: the hook
#: read ``derived_meta.json:params``, a key no writer records.
_DERIVED_PARENT_KEYS = ("cell_id", "object_sha256", "fit_fingerprint", "sample_manifest_sha", "data_scale")


def derivation_params_from_derived_meta(dm: Mapping[str, object]) -> Optional[Dict[str, object]]:
    """The derivation record of a derived object from its sidecar: derivation text, scale match, comparison, builder,
    built_at, each parent's identity (``cell_id``, ``object_sha256``, ``fit_fingerprint``, ``sample_manifest_sha``,
    ``data_scale``), plus the distillation facts (teacher, lr_mult + how it was selected, recovery verdict) when the
    sidecar records them. None for an empty / non-mapping sidecar."""
    if not isinstance(dm, Mapping) or not dm:
        return None
    params: Dict[str, object] = {"derivation_text": dm.get("derivation"), "scale_matched": dm.get("scale_matched"),
                                 "comparison": dm.get("comparison"), "builder": dm.get("builder"), "built_at": dm.get("built_at"),
                                 "parents": {r: {k: (dm.get(r) or {}).get(k) for k in _DERIVED_PARENT_KEYS}   # type: ignore[union-attr]
                                             for r in ("parent_pos", "parent_neg") if isinstance(dm.get(r), Mapping)}}
    if isinstance(dm.get("teacher"), Mapping):
        params["teacher"] = dict(dm["teacher"])  # type: ignore[arg-type]
    if isinstance(dm.get("trainer"), Mapping):
        tr = dm["trainer"]  # type: ignore[assignment]
        params["lr_mult"] = tr.get("lr_mult")  # type: ignore[union-attr]
        params["lr_mult_selected_by"] = tr.get("lr_mult_selected_by")  # type: ignore[union-attr]
    if "recovery_verdict" in dm:
        params["recovery_verdict"] = dm.get("recovery_verdict")
    return params


#: The fit-time PROVENANCE columns the hook used to derive from the EMITTING context (the producer block's code sha / host / substrate /
#: session, the current SLURM_JOB_ID). Right at the first emission — the fit job is the emitter — and WRONG on every later re-emission or
#: kept-object replay, which stated the replaying job's context as the fit's (the class had already overwritten the parents' rows once,
#: with a re-emission's srun). Precedence now: the cell's `fit_job_provenance.json` sidecar (an authoritative statement written from the
#: fit's own records: the RESULT line's job + node, the funnel attempt record's pinned sha) > an explicit `fit_provenance` the caller
#: states (a replay passing the object's PRIOR record's statement) > the emitting context.
#: NOT `fit_provenance.json`: the class-B fitter already writes a sidecar of that name (resolved model / writer / positions — the
#: `prov` block object_from_sidecar reads), so the job-provenance statement has its own file
FIT_PROVENANCE_SIDECAR = "fit_job_provenance.json"
FIT_PROVENANCE_KEYS: Tuple[str, ...] = ("fit_code_sha", "fit_job_id", "fit_node_hostname", "fit_substrate", "fit_producer_session",
                                        "fit_started_utc", "fit_finished_utc", "fit_walltime_s", "fit_gpu_type", "fit_gpu_count",
                                        "fit_wandb_run_id", "fit_wandb_url")


def read_fit_provenance_sidecar(obj_dir: str) -> Optional[Dict[str, object]]:
    """``<obj_dir>/fit_job_provenance.json`` when present: ``{<FIT_PROVENANCE_KEYS>…, artifact_created_utc?, sources: {...}, note?,
    written_by, written_utc}``; None when absent. Malformed → ValueError (never silently ignored)."""
    p = os.path.join(obj_dir, FIT_PROVENANCE_SIDECAR)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not any(k in doc for k in FIT_PROVENANCE_KEYS):
        raise ValueError(f"{p}: not a fit-provenance sidecar (expected some of {FIT_PROVENANCE_KEYS})")
    return doc


def apply_fit_provenance(payload: Dict[str, object], artifact: Dict[str, object], prov: Optional[Mapping[str, object]]) -> None:
    """State ``prov``'s fit-time columns on the object payload (a None value UNSTATES the column: omission keeps the row's value) and its
    ``artifact_created_utc`` on the artifact; record its sources / note under ``extra.fit_provenance`` so the row explains itself."""
    if not prov:
        return
    for k in FIT_PROVENANCE_KEYS:
        if k in prov:
            if prov[k] is None:
                payload.pop(k, None)
            else:
                payload[k] = prov[k]
    if prov.get("artifact_created_utc"):
        artifact["created_utc"] = str(prov["artifact_created_utc"])
    meta = {k: v for k, v in (("sources", prov.get("sources")), ("note", prov.get("note")), ("written_by", prov.get("written_by")),
                              ("written_utc", prov.get("written_utc")), ("restated_from", prov.get("_source"))) if v}
    if meta:
        payload["extra"] = {**dict(payload.get("extra") or {}), "fit_provenance": meta}


#: LORADOSE (the native-record audit of the six dose-copy emissions): a DOSE COPY's statement comes from the sidecar the
#: producer wrote and verified (scale_lora_dose.py / scale_psr_dose.py: ``dose_scaled_from`` in derived_meta.json or train_budget_meta.json,
#: ``staged_from.json`` beside it), never from the copied PARENT identity — before this the hook filed a dose copy as approved at base_scale
#: NULL (= nominal 1.0 to the served-strength payload and the notebook's dose arms), with the parent builder's fit_writer and the parent's cell id.
DOSE_SCALE_DERIVATION = "dose_scale"
DOSE_COPY_PRODUCERS: Tuple[str, ...] = ("scripts/sdfbase/scale_lora_dose.py", "scripts/sdfbase/scale_psr_dose.py")
STAGED_FROM_SIDECAR = "staged_from.json"


#: the ``field_confidence_json`` source a dose copy's ``fit_effective_batch`` states when it is its PARENT's: the parent's sha prefix
#: and the sidecar key the value is read from
DOSE_COPY_EFFECTIVE_BATCH_SOURCE = ("the parent's effective batch ({parent}…), read from {sidecar}:effective_batch (the PARENT's fit record copied "
                                    "beside the copy); a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13)")


def _apply_dose_copy_statement(payload: Dict[str, object], obj_dir: str, *, dm: Mapping[str, object], tbm: Mapping[str, object],
                               derivation_params_json: Optional[Mapping[str, object]], fc: Dict[str, object],
                               tbm_present: bool = True) -> None:
    """(a) ``base_scale`` = the verified ``dose_scaled_from.scale`` (a finite non-zero float; a caller-stated base_scale / scale must agree);
    (b) ``fit_writer`` = the stamp's ``producer`` (one of DOSE_COPY_PRODUCERS); (c) ``campaign_cell_id_legacy`` = the copy's OWN staged id
    from staged_from.json (the parent's cell id is what the copied identity sidecar carries). Refuses by name when the stamp is missing.

    (d) ``fit_effective_batch`` — this path OWNS the column: a dose-scaled copy was not fit, so its effective batch is its PARENT's,
    stated as ``derived``, else NULL by name. The copy's ``train_budget_meta.json`` is the parent's fit record re-bound to the new bytes
    (scale_lora_dose.py / scale_psr_dose.py), so its ``effective_batch`` IS the parent's: an int there is the column, with
    ``field_confidence_json.fit_effective_batch`` = {confidence derived, source :data:`DOSE_COPY_EFFECTIVE_BATCH_SOURCE` naming the
    parent's sha prefix}. Anything else — a recorded value that is not an int, a copied record that names none, a copy carrying no
    train_budget_meta.json at all (a derived parent's copy carries derived_meta.json alone) — pops the column: NEVER the micro-batch
    (``resolved_batch_size`` / ``batch_size``, the generic line's stand-in for a FITTED object's sidecar that records no effective
    batch), with the unverified entry under ``field_confidence_json`` and the same reason under ``extra.fit_effective_batch_unresolved``.
    ``tbm_present`` says whether the copy carries the sidecar at all (``tbm`` is ``{}`` either way). Every other
    ``field_confidence_json`` entry is kept — the object's field_confidence_json is replaced WHOLE by the last folded record."""
    dsf = dm.get("dose_scaled_from") if isinstance(dm.get("dose_scaled_from"), Mapping) else tbm.get("dose_scaled_from")
    if not isinstance(dsf, Mapping):
        raise ValueError(f"{obj_dir}: derivation {DOSE_SCALE_DERIVATION!r} but no sidecar carries a `dose_scaled_from` stamp (the producer's verified "
                         "scale, parent sha and verification record) — a dose copy is described from its own stamp, never from the copied parent "
                         "identity; not written")
    try:
        scale = float(dsf.get("scale"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        scale = float("nan")
    if not (scale == scale) or scale in (0.0,) or scale in (float("inf"), float("-inf")):
        raise ValueError(f"{obj_dir}: dose_scaled_from.scale={dsf.get('scale')!r} is not a finite non-zero number; not written")
    stated = dict(derivation_params_json or {})
    for k in ("base_scale", "scale"):
        if stated.get(k) is not None and float(stated[k]) != scale:   # type: ignore[arg-type]
            raise ValueError(f"{obj_dir}: the caller states derivation_params_json.{k}={stated[k]!r} but the sidecar's verified scale is {scale:g}; "
                             "the stamp is the statement — not written")
    payload["base_scale"] = scale
    producer = dsf.get("producer")
    if isinstance(producer, str) and producer:
        payload["fit_writer"] = producer
        if producer not in DOSE_COPY_PRODUCERS:
            fc["fit_writer"] = _confidence_entry(f"dose_scaled_from.producer {producer!r} is not one of the known dose-copy producers {list(DOSE_COPY_PRODUCERS)}")
    else:
        fc["fit_writer"] = _confidence_entry("dose_scaled_from names no producer")
    sf_path = os.path.join(obj_dir, STAGED_FROM_SIDECAR)
    staged_id = None
    if os.path.isfile(sf_path):
        try:
            with open(sf_path, "r", encoding="utf-8") as fh:
                sf = json.load(fh)
            staged_id = sf.get("staged_id") if isinstance(sf, Mapping) else None
        except ValueError:
            staged_id = None
    if isinstance(staged_id, str) and staged_id:
        payload["campaign_cell_id_legacy"] = staged_id
    else:
        fc["campaign_cell_id_legacy"] = _confidence_entry(f"no {STAGED_FROM_SIDECAR} with a staged_id beside the copy: the value is the copied parent identity's cell id")
    payload["extra"] = {**dict(payload.get("extra") or {}),      # type: ignore[dict-item]
                        "dose_copy": {"scale": scale, "producer": producer, "producer_commit": dsf.get("producer_commit"),
                                      "parent_object_sha256": dsf.get("parent_object_sha256"), "staged_id": staged_id,
                                      "verification": {c.get("check"): c.get("status") for c in ((dsf.get("verification") or {}).get("checks") or [])
                                                       if isinstance(c, Mapping)} if isinstance(dsf.get("verification"), Mapping) else None}}
    # (d) the effective batch: the PARENT's, stated as derived, else NULL by name — never the micro-batch on this path
    parent_sha = dsf.get("parent_object_sha256")
    parent = str(parent_sha)[:12] if isinstance(parent_sha, str) and parent_sha else "parent sha unrecorded"
    eb = tbm.get("effective_batch") if tbm_present else None
    if _is_int(eb):
        payload["fit_effective_batch"] = eb
        fc["fit_effective_batch"] = {"confidence": "derived", "source": DOSE_COPY_EFFECTIVE_BATCH_SOURCE.format(parent=parent, sidecar="train_budget_meta.json")}
    else:
        if not tbm_present:
            why = ("the copy carries no train_budget_meta.json (its identity sidecar is derived_meta.json: a derived parent's record names no effective "
                   "batch), so the parent's effective batch is unrecorded beside the copy")
        elif "effective_batch" not in tbm:
            why = "the copied record train_budget_meta.json names no effective_batch (the PARENT's fit record records none beside the copy)"
        else:
            why = f"train_budget_meta.json:effective_batch is {eb!r} ({type(eb).__name__}), not an int"
        why = f"{why}; a dose-scaled copy was not fit, so its effective batch is its PARENT's or NULL — never the copy's micro-batch (the user 2026-10-13)"
        payload.pop("fit_effective_batch", None)
        fc["fit_effective_batch"] = _confidence_entry(why)
        payload["extra"] = {**dict(payload.get("extra") or {}), "fit_effective_batch_unresolved": why}   # type: ignore[dict-item]


#: (e) the roots an artifact path must not live under to be recorded as an object's primary location: a record naming a path that vanishes
#: at reboot (lane 06's dry run named /tmp/.../mint/<sid>/object.pt) would fold a location nothing can read later. The spool's own temp
#: root is exempt (a test / scratch world whose spool is as ephemeral as its artifacts), and so is a spool whose temp root lies INSIDE the
#: artifact's (a per-task $TMPDIR under /tmp: refuse_non_durable_artifact).
NON_DURABLE_ROOTS: Tuple[str, ...] = ("/tmp", "/var/tmp", "/dev/shm")


def _non_durable_root(path: str) -> Optional[str]:
    rp = os.path.realpath(path)
    roots = list(NON_DURABLE_ROOTS) + ([os.path.realpath(os.environ["TMPDIR"])] if os.environ.get("TMPDIR") else [])
    hits = [r.rstrip("/") for r in roots if rp == r.rstrip("/") or rp.startswith(r.rstrip("/") + "/")]
    return max(hits, key=len) if hits else None          # the most specific root ($TMPDIR under /tmp names $TMPDIR)


def refuse_non_durable_artifact(art_path: str, spool_path: Optional[str], where: str) -> None:
    """(e): a non-durable artifact location is refused unless the spool itself is as ephemeral — its own temp root is the artifact's, or lies
    INSIDE it (a per-task $TMPDIR below the /tmp the artifact sits in: the cluster gate's layout, the checkout copied to /tmp/<task>/sisteer-<sha>
    and the suite's TMPDIR at /tmp/<task>/tmp, 2026-10-13), so a record in that spool cannot outlive the location it names. The converse — the
    artifact minted under $TMPDIR, the spool under the wider /tmp — still refuses: that spool may outlive the mint. A durable spool refuses every
    non-durable artifact."""
    root = _non_durable_root(art_path)
    if root is None or spool_path is None:
        return
    spool_root = _non_durable_root(spool_path)
    if spool_root is not None and (spool_root == root or spool_root.startswith(root + "/")):
        return
    raise ValueError(f"{where}: the artifact {art_path} lives under the non-durable root {root} — an object record names the PERSISTENT bank "
                     "cell's bytes (the location a reader can open later), never a scratch mint; move the cell into the bank first; not written "
                     f"(the spool {spool_path} is {'under ' + spool_root if spool_root else 'durable'}, so a record there could outlive the artifact)")


def registry_root_for_spool(spool_path: Optional[str], registry_root: Optional[str] = None) -> Optional[str]:
    """The registry root the idempotency check reads: an explicit one, else $SISTEER_REGISTRY_ROOT, else the spool's parent when the spool is
    its `records/` dir, else the grading host default (emit.HOME_REGISTRY)."""
    if registry_root:
        return os.path.expanduser(registry_root)
    env = os.environ.get("SISTEER_REGISTRY_ROOT")
    if env:
        return os.path.expanduser(env)
    if spool_path and os.path.basename(os.path.normpath(spool_path)) == "records":
        return os.path.dirname(os.path.normpath(spool_path))
    return os.path.expanduser(emit.HOME_REGISTRY)


def existing_object_statements(object_id: str, *, spool_path: Optional[str], campaign: str, registry_root: Optional[str]) -> Dict[str, object]:
    """(d) what already states this object: the native record dir's object records (``<spool>/<campaign>/object__<sha16>/*-object.jsonl[.gz]``)
    and the registry's object + artifact rows (read-only, mode=ro; an absent database is 'no registry'). Returns
    ``{"records": [paths], "row": {status, record_uri} | None, "artifact": {created_utc, uri_primary} | None, "registry": path | None}``."""
    import glob
    import gzip
    out: Dict[str, object] = {"records": [], "row": None, "artifact": None, "registry": None}
    if spool_path and campaign:          # a trainer's dry-run preflight may carry campaign=None (the job env names it later): nothing to scan, nothing to refuse
        d = os.path.join(spool_path, campaign, run_dir_name(f"object__{object_id[:16]}"))
        for p in sorted(glob.glob(os.path.join(d, "*-object.jsonl")) + glob.glob(os.path.join(d, "*-object.jsonl.gz"))):
            opener = gzip.open if p.endswith(".gz") else open
            try:
                with opener(p, "rt", encoding="utf-8") as fh:   # type: ignore[call-arg]
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        pay = rec.get("payload") if isinstance(rec, Mapping) and isinstance(rec.get("payload"), Mapping) else rec
                        if str((pay or {}).get("object_id")) == object_id:
                            out["records"].append(p)   # type: ignore[union-attr]
                            break
            except (OSError, ValueError):
                out["records"].append(p + " (unreadable)")   # type: ignore[union-attr]
    root = registry_root_for_spool(spool_path, registry_root)
    if root:
        from . import db as DB
        path = DB.db_path(root)
        if os.path.isfile(path):
            out["registry"] = path
            conn = DB.connect(path, readonly=True)
            try:
                r = conn.execute("SELECT status, record_uri FROM object WHERE object_id = ?", (object_id,)).fetchone()
                if r is not None:
                    out["row"] = {"status": r[0], "record_uri": r[1]}
                a = conn.execute("SELECT created_utc, uri_primary FROM artifact WHERE artifact_id = ?", (object_id,)).fetchone()
                if a is not None:
                    out["artifact"] = {"created_utc": a[0], "uri_primary": a[1]}
            finally:
                conn.close()
    return out


def _provenance_coefficient(prov: Mapping[str, object]) -> Optional[Tuple[str, object]]:
    """``(source, c)`` when ``fit_provenance.json`` records the object's own coefficient — top-level ``c``, else the first block
    (``prov[<name>]`` a mapping) carrying one; None when it records none."""
    if not isinstance(prov, Mapping):
        return None
    if prov.get("c") is not None:
        return ("fit_provenance.json:c", prov["c"])
    for name, blk in prov.items():
        if isinstance(blk, Mapping) and blk.get("c") is not None:
            return (f"fit_provenance.json:{name}.c", blk["c"])
    return None


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _provenance_served_layers(prov: Mapping[str, object]) -> List[Tuple[str, int]]:
    """EVERY integer ``served_layer`` ``fit_provenance.json`` records, as ``(source, served_layer)`` in file order: the TOP-LEVEL value
    (``fit_provenance.json:served_layer``) first, then each nested block's (``fit_provenance.json:<block>.served_layer``). Empty when it
    records none. The served block is the layer a vector_all blob is served at, a zero-based decoder block."""
    if not isinstance(prov, Mapping):
        return []
    out: List[Tuple[str, int]] = []
    if _is_int(prov.get("served_layer")):
        out.append(("fit_provenance.json:served_layer", int(prov["served_layer"])))   # type: ignore[arg-type]
    for name, blk in prov.items():
        if isinstance(blk, Mapping) and _is_int(blk.get("served_layer")):
            out.append((f"fit_provenance.json:{name}.served_layer", int(blk["served_layer"])))   # type: ignore[arg-type]
    return out


def _provenance_served_layer(prov: Mapping[str, object]) -> Optional[Tuple[str, int]]:
    """The ONE unambiguous served block ``fit_provenance.json`` records, ``(source, served_layer)`` — the first of
    :func:`_provenance_served_layers` when every recorded value (the top-level one and every block's) agrees; None when it records none.
    Two values that DISAGREE are a provenance that names no one served block: ``ValueError`` naming each source and its value (the
    former reader took the first block it met and never saw the top-level value or a second block, so an edited pin disagreeing with a
    later block compared against the wrong layer, or against none)."""
    found = _provenance_served_layers(prov)
    if not found:
        return None
    values = sorted({v for _s, v in found})
    if len(values) > 1:
        raise ValueError("fit_provenance.json records conflicting served layers — " + ", ".join(f"{s} = {v}" for s, v in found)
                         + " — so it names no one served block to compare an explicit fit_layers_json against")
    return found[0]


def _explicit_layers_source(layers: Sequence[int], *, dm: Mapping[str, object], prov: Mapping[str, object]) -> str:
    """Where the sidecars RECORD an explicitly passed layer list: ``derived_meta.json:layers`` when the derived sidecar carries that list,
    ``fit_provenance.json:served_layer`` / ``fit_provenance.json:<block>.served_layer`` when a recorded served layer (the top-level value or
    a block's) is the list's one element; else the writer's own statement (the hook reports where the list is written down, never what
    convention it has)."""
    want = [int(x) for x in layers]
    if isinstance(dm, Mapping) and isinstance(dm.get("layers"), list) and [int(x) for x in dm["layers"]] == want:
        return "derived_meta.json:layers"
    if len(want) == 1:
        for source, served in _provenance_served_layers(prov):
            if served == want[0]:
                return source
    return "stated by the writer (no sidecar records this layer list)"


def object_from_sidecar(obj_dir: str, *, campaign: str, object_kind: str, paper_label: str, object_role: str,
                        fit_target_kind: str, fit_objective: str, fit_thinking_render: str = "unverified",
                        fit_model_id: Optional[str] = None, status: str = "reference", derivation: str = "none",
                        fit_teacher_si_id: Optional[str] = None, fit_corpus_id: Optional[str] = None,
                        producer: Optional[Mapping[str, object]] = None, extra: Optional[Mapping[str, object]] = None,
                        script: Optional[str] = None,
                        fit_search_id: Optional[str] = None, fit_lr_selected_by: Optional[str] = None,
                        fit_rank: Optional[int] = None, fit_alpha: Optional[float] = None,
                        fit_target_modules_json: Optional[Sequence[str]] = None, fit_layers_json: Optional[Sequence[int]] = None,
                        fit_corpus_sha256: Optional[str] = None, fit_replicate_label: Optional[str] = None,
                        fit_selection: Optional[Mapping[str, object]] = None,
                        fit_contrast_sets_json: Optional[Mapping[str, object]] = None, fit_capture_site: Optional[str] = None,
                        parent_object_ids_json: Optional[Sequence[str]] = None, derivation_params_json: Optional[Mapping[str, object]] = None,
                        fit_writer: Optional[str] = None, fit_provenance: Optional[Mapping[str, object]] = None,
                        base_scale: Optional[float] = None, fit_layer_index_convention: Optional[str] = None
                        ) -> Tuple[Dict[str, object], Dict[str, object]]:
    """``(object payload, artifact payload)`` for the fitted object under ``obj_dir``, read from its
    sidecars (``train_budget_meta.json`` / ``fit_meta.json`` / ``fit_provenance.json`` /
    ``derived_meta.json`` / ``fit_selection.json``) and the artifact's own bytes. The WRITER states what it fitted (kind, label,
    role, target kind, objective) — a sidecar cannot; the hook never guesses those. Every fact the
    sidecar lacks is ``unverified`` with a reason in ``field_confidence_json``.

    The fit SELECTION and the LoRA ARCHITECTURE land in the REAL columns
    ``fit_search_id``, ``fit_lr_selected_by``, ``fit_rank``, ``fit_alpha``, ``fit_target_modules_json``, ``fit_layers_json``: from the
    keyword arguments when the caller states them, else from the published ``fit_selection.json`` (:data:`FIT_SELECTION_SIDECAR`,
    read authoritatively); the sidecar's other keys (``per_expert_rank``, ``lora_set_id``, ``trainable_params``, ``init_scheme``,
    ``architecture_selection_ruling_id``, ``architecture_selected_by``) ride ``extra``. ``script`` is accepted for callers that pass
    the writer's name through (``emit_object_from_sidecar`` consumes it before this call).

    Two statements a WRITER may make that no sidecar can (demo-import, the committed-object importer; Q14): ``base_scale`` on an
    object whose ``derivation`` is not ``dose_scale`` (a dose copy's scale is its stamp's, `_apply_dose_copy_statement`, and a
    caller value there is refused) — refused when ``fit_provenance.json`` records its own coefficient (``c``, top-level or in a
    block) that disagrees, recorded in ``field_confidence_json.base_scale`` with that source; and ``fit_layer_index_convention``
    (one of ``vocab.LAYER_INDEX_CONVENTIONS``) beside an EXPLICIT ``fit_layers_json``: when the sidecars state no layer list of
    their own, the convention of an explicit list is the writer's to state and is REQUIRED (the hook never guesses one); when the
    sidecars DO state a list (``train_budget_meta.json:target_layers`` / ``fit_meta.json:layers``, the trainers' recorded
    zero-based decoder blocks) the hook's own ``zero_based_decoder_block`` stands and a stated convention must equal it. A stated
    convention is recorded in ``field_confidence_json.fit_layer_index_convention`` naming where the sidecars record the list the
    writer handed over (``derived_meta.json:layers``, ``fit_provenance.json:served_layer``, ``fit_provenance.json:<block>.served_layer``),
    else the writer's statement. An explicit ``zero_based_decoder_block`` list is compared against the ONE served block
    ``fit_provenance.json`` records — its top-level ``served_layer`` and every block's are read, two that disagree refuse by name, and a
    list that is not ``[served_layer]`` refuses by name.

    ``fit_code_sha``: whenever the payload states the schema's ``unverified`` (a caller's ``fit_provenance`` says so, or the producer block's
    ``code_sha`` is emit.git_sha's fallback), the hook writes ``field_confidence_json.fit_code_sha`` =
    :data:`FIT_CODE_SHA_UNVERIFIED_ENTRY` itself — the writer that states the value meets validate.py's rule for it."""
    art = find_object_artifact(obj_dir)
    if art is None:
        raise FileNotFoundError(f"no object artifact under {obj_dir} ({', '.join(_OBJECT_CANDIDATES)})")
    sha = _sha256_file(art)
    side = read_sidecars(obj_dir)
    tbm = side.get("train_budget_meta.json") or {}
    fm = side.get("fit_meta.json") or {}
    prov = side.get("fit_provenance.json") or {}
    dm = side.get("derived_meta.json") or {}
    sel: Dict[str, object] = dict(fit_selection) if fit_selection is not None else dict(side.get(FIT_SELECTION_SIDECAR) or {})
    sel.pop("_unparseable", None)
    fc: Dict[str, object] = {}
    positions_at_fit = tbm.get("positions_at_fit") or prov.get("positions_at_fit") or fm.get("positions_at_fit")
    if fit_target_kind == "derived":
        fit_positions = "inherited_from_parents"
    elif positions_at_fit is None:
        fit_positions = _UNVERIFIED
        fc["fit_positions"] = _confidence_entry("no positions_at_fit in any sidecar")
    else:
        fit_positions = POSITIONS_AT_FIT_TO_FIT_POSITIONS.get(str(positions_at_fit), _UNVERIFIED)
        if fit_positions == _UNVERIFIED:
            fc["fit_positions"] = _confidence_entry(f"positions_at_fit={positions_at_fit!r} has no registry spelling")
    seed = tbm.get("seed", fm.get("seed"))
    seed_conf = "recorded" if isinstance(seed, int) and not isinstance(seed, bool) else ("not_applicable" if fit_target_kind in ("derived", "none", "declared_prompt") else _UNVERIFIED)
    if seed_conf == _UNVERIFIED:
        fc["fit_seed_confidence"] = _confidence_entry("no seed in any sidecar")
    if fit_thinking_render == _UNVERIFIED:
        fc["fit_thinking_render"] = _confidence_entry("the writer did not state the thinking render of the fit")
    layers = tbm.get("target_layers") if isinstance(tbm.get("target_layers"), list) else (fm.get("layers") if isinstance(fm.get("layers"), list) else None)
    hyper = {k: v for k, v in tbm.items() if k not in _KNOWN_TBM_KEYS} if tbm else {}
    for k in ("lr", "lr_schedule", "epochs", "warmup_steps", "max_grad_norm", "batch_size", "resolved_batch_size", "max_len", "completion_tokens"):
        if k in tbm:
            hyper[k] = tbm[k]
    tbm_model_rev = tbm.get("model_revision") or prov.get("resolved_revision")
    payload: Dict[str, object] = {
        "object_id": sha, "object_kind": object_kind, "paper_label": paper_label, "campaign": campaign, "object_role": object_role,
        "fit_target_kind": fit_target_kind, "fit_thinking_render": fit_thinking_render, "fit_objective": fit_objective,
        "fit_positions": fit_positions, "fit_seed_confidence": seed_conf, "derivation": derivation, "artifact_sha256": sha,
        "artifact_bytes": os.path.getsize(art), "status": status,
        "campaign_cell_id_legacy": tbm.get("cell_id") or fm.get("cell_id"),
        "fit_teacher_si_id": fit_teacher_si_id if fit_teacher_si_id is not None else (tbm.get("si_id") if fit_target_kind == "system_instruction_distillation" else None),
        "fit_corpus_id": fit_corpus_id if fit_corpus_id is not None else (tbm.get("si_id") if fit_target_kind == "corpus_document_fit" else None),
        "fit_model_id": fit_model_id or tbm.get("model_id") or prov.get("resolved_model_id"),
        "fit_response_header_len": tbm.get("response_header_len"),
        "fit_layers_json": layers, "fit_layer_count": len(layers) if isinstance(layers, list) else None,
        "fit_layer_index_convention": "zero_based_decoder_block" if isinstance(layers, list) else None,
        "fit_capture_site": tbm.get("site") if tbm.get("site") in ("resid_pre", "resid_post") else None,
        "fit_completion_tokens": tbm.get("completion_tokens"),
        "fit_hyperparameters_json": hyper or None,
        "fit_lr": tbm.get("lr"), "fit_lr_schedule": tbm.get("lr_schedule"),
        # the EFFECTIVE batch the trainers record (`effective_batch` = micro-batch x accum_steps, the key the objects_bank adapter files the
        # column from); `resolved_batch_size` / `batch_size` are the MICRO-batch and stand in only for a sidecar that records no effective batch
        # (the batch-64 fits had been stated at their micro-batch 2). This generic line is for objects that were FIT; a dose copy's column is
        # owned by _apply_dose_copy_statement below (the parent's, stated as derived, else NULL by name — never the micro-batch)
        "fit_effective_batch": (tbm["effective_batch"] if isinstance(tbm.get("effective_batch"), int) and not isinstance(tbm.get("effective_batch"), bool)
                                else tbm.get("resolved_batch_size", tbm.get("batch_size"))),
        "fit_epochs_cap": tbm.get("epochs"), "fit_seed": seed if seed_conf == "recorded" else None,
        "fit_writer": fit_writer or tbm.get("trainer") or prov.get("writer"),
        "fit_code_sha": (producer or {}).get("code_sha"), "fit_substrate": (producer or {}).get("substrate"),
        "fit_job_id": os.environ.get("SLURM_JOB_ID"), "fit_node_hostname": (producer or {}).get("host"),
        "fit_producer_session": (producer or {}).get("session"),
        "parent_object_ids_json": dm.get("parent_object_ids") if isinstance(dm.get("parent_object_ids"), list) else None,
        # a derived object's derivation record: the sidecar's own `params` block when a writer records one, else built from the
        # sidecar the way the adapter-filed rows were (derivation_params_from_derived_meta); the caller's explicit kwarg wins below
        "derivation_params_json": (dm.get("params") if isinstance(dm.get("params"), dict)
                                   else (derivation_params_from_derived_meta(dm) if fit_target_kind == "derived" else None)),
        "artifact_locations_json": [art],
        "object_sidecars_json": sorted(side),
        "extra": {"sidecar_object_sha256": tbm.get("object_sha256") or fm.get("object_sha256"), "positions_at_fit": positions_at_fit,
                  "template_hash": tbm.get("template_hash") or prov.get("chat_template_hash"), "fit_model_revision": tbm_model_rev,
                  **(dict(extra) if extra else {})},
    }
    recorded = payload["extra"]["sidecar_object_sha256"]  # type: ignore[index]
    if isinstance(recorded, str) and recorded and recorded != sha:
        raise ValueError(f"{obj_dir}: the sidecar records object_sha256 {recorded[:12]}… but the artifact hashes to {sha[:12]}…; refusing to describe it")
    if derivation == DOSE_SCALE_DERIVATION:
        _apply_dose_copy_statement(payload, obj_dir, dm=dm, tbm=tbm, derivation_params_json=derivation_params_json, fc=fc,
                                   tbm_present="train_budget_meta.json" in side)
    # -- the selection and the architecture: the caller's kwargs win, the published fit_selection.json fills what they leave unstated
    explicit = {"fit_search_id": fit_search_id, "fit_lr_selected_by": fit_lr_selected_by, "fit_rank": fit_rank, "fit_alpha": fit_alpha,
                "fit_target_modules_json": (sorted(str(m) for m in fit_target_modules_json) if fit_target_modules_json is not None else None),
                "fit_layers_json": (list(fit_layers_json) if fit_layers_json is not None else None),
                "fit_corpus_sha256": fit_corpus_sha256, "fit_replicate_label": fit_replicate_label,
                # the refusal direction (contrastive_direction: the harmful / harmless capture sets) and a derived object's parents,
                # when the writer states them (a sidecar's derived_meta.json fills the latter otherwise)
                "fit_contrast_sets_json": dict(fit_contrast_sets_json) if fit_contrast_sets_json is not None else None,
                "fit_capture_site": fit_capture_site if fit_capture_site in ("resid_pre", "resid_post") else None,
                "parent_object_ids_json": list(parent_object_ids_json) if parent_object_ids_json is not None else None,
                "derivation_params_json": dict(derivation_params_json) if derivation_params_json is not None else None}
    for key, col in _FIT_SELECTION_COLUMNS.items():
        if explicit.get(col) is None and sel.get(key) is not None:
            v = sel[key]
            if col == "fit_target_modules_json" and isinstance(v, (list, tuple)):
                v = sorted(str(m) for m in v)
            explicit[col] = v
    if fit_target_kind == "corpus_document_fit" and explicit.get("fit_corpus_sha256") is None:
        # the pool the corpus fit read: the selection sidecar's digest, else the trainer's recorded one (a sample manifest sha / a pool
        # sha in train_budget_meta.json), else unverified with the reason
        if sel.get("fit_corpus_sha256") is not None:
            explicit["fit_corpus_sha256"] = sel["fit_corpus_sha256"]
        else:
            recorded_pool = next((tbm.get(k) for k in ("fit_corpus_sha256", "pool_sha256", "corpus_sha256", "sample_manifest_sha")
                                  if isinstance(tbm.get(k), str) and len(str(tbm.get(k))) == _SHA256_LEN), None)
            if recorded_pool is None:
                explicit["fit_corpus_sha256"] = _UNVERIFIED
                fc["fit_corpus_sha256"] = _confidence_entry("no pool / sample-manifest sha256 in train_budget_meta.json or fit_selection.json")
            else:
                explicit["fit_corpus_sha256"] = str(recorded_pool).lower()
    payload.update({col: v for col, v in explicit.items() if v is not None})
    if payload.get("fit_rank") is not None and payload.get("fit_alpha") is None:
        payload["fit_alpha"] = float(payload["fit_rank"])          # LoRA scaling 1: alpha = r
    from . import vocab as V
    if fit_layer_index_convention is not None and fit_layer_index_convention not in V.LAYER_INDEX_CONVENTIONS:
        raise ValueError(f"{obj_dir}: fit_layer_index_convention {fit_layer_index_convention!r} is not one of {list(V.LAYER_INDEX_CONVENTIONS)}; not written")
    if payload.get("fit_layers_json") is not None and isinstance(payload["fit_layers_json"], list):
        payload["fit_layer_count"] = len(payload["fit_layers_json"])
        if isinstance(layers, list):
            # the sidecar-derived list: the trainers record zero-based decoder blocks (above); a writer's statement must agree
            if fit_layer_index_convention is not None and fit_layer_index_convention != payload["fit_layer_index_convention"]:
                raise ValueError(f"{obj_dir}: the caller states fit_layer_index_convention {fit_layer_index_convention!r} but the sidecar's layer list "
                                 f"({'train_budget_meta.json:target_layers' if isinstance(tbm.get('target_layers'), list) else 'fit_meta.json:layers'}) is "
                                 f"recorded as {payload['fit_layer_index_convention']!r}; not written")
        elif fit_layer_index_convention is None:
            # an EXPLICIT list over sidecars that state none: the convention is the writer's to state, never the hook's to guess (the
            # former `setdefault` over the None written above never set it, so the five committed explicit-layer rows stated a list and
            # NO convention — demo-import P1-1)
            raise ValueError(f"{obj_dir}: explicit fit_layers_json for {obj_dir} needs a stated fit_layer_index_convention (one of "
                             f"{list(V.LAYER_INDEX_CONVENTIONS)}); the sidecars record no layer list and the hook never guesses one; not written")
        else:
            if fit_layer_index_convention == "zero_based_decoder_block":
                # an explicit ZERO-BASED list against a provenance block that records the SERVED block: the two describe the same layer, so a
                # list that is not [served_layer] is a writer's error (an edited pin over the same bytes), refused by name — never filed as
                # "stated by the writer" (the registry owner's review of this diff). The served block is the ONE unambiguous value the
                # provenance records — top-level or in any block; two that disagree refuse by name before any comparison
                try:
                    served = _provenance_served_layer(prov)
                except ValueError as exc:
                    raise ValueError(f"{obj_dir}: {exc}; not written") from None
                if served is not None and [int(x) for x in payload["fit_layers_json"]] != [served[1]]:
                    raise ValueError(f"{obj_dir}: the caller states fit_layers_json={list(payload['fit_layers_json'])!r} (zero_based_decoder_block) but "
                                     f"{served[0]} records the served block {served[1]}; the object's own provenance is the statement; not written")
            payload["fit_layer_index_convention"] = fit_layer_index_convention
            src = _explicit_layers_source(payload["fit_layers_json"], dm=dm, prov=prov)
            fc["fit_layer_index_convention"] = {"confidence": "recorded", "source": src}
            fc["fit_layers_json"] = {"confidence": "recorded", "source": src}
    elif fit_layer_index_convention is not None:
        raise ValueError(f"{obj_dir}: fit_layer_index_convention {fit_layer_index_convention!r} stated with no fit_layers_json to describe; not written")
    if base_scale is not None:
        # a writer's base_scale (the refusal direction's operating coefficient): never on a dose copy (its stamp is the statement, above),
        # and never against a coefficient the object's own provenance records
        if derivation == DOSE_SCALE_DERIVATION:
            raise ValueError(f"{obj_dir}: the caller states base_scale={base_scale!r} on a {DOSE_SCALE_DERIVATION!r} copy — the dose stamp is the "
                             "statement (dose_scaled_from.scale); not written")
        try:
            bs = float(base_scale)
        except (TypeError, ValueError):
            raise ValueError(f"{obj_dir}: base_scale={base_scale!r} is not a number; not written")
        if not (bs == bs) or bs in (float("inf"), float("-inf")):
            raise ValueError(f"{obj_dir}: base_scale={base_scale!r} is not a finite number; not written")
        own = _provenance_coefficient(prov)
        if own is not None and float(own[1]) != bs:
            raise ValueError(f"{obj_dir}: the caller states base_scale={bs:g} but {own[0]} records {own[1]!r}; the object's own provenance is the "
                             "statement; not written")
        payload["base_scale"] = bs
        fc["base_scale"] = {"confidence": "recorded", "source": own[0] if own is not None else "stated by the writer (no sidecar records a coefficient)"}
    sel_extra = {k: v for k, v in sel.items() if k not in _FIT_SELECTION_COLUMNS}
    if sel_extra:
        payload["extra"] = {**sel_extra, **payload["extra"]}   # type: ignore[dict-item]  -- the caller's extra wins over the sidecar's
    if payload.get("fit_lr_selected_by") == "manual":
        payload["extra"].setdefault("architecture_selected_by", "manual")   # type: ignore[union-attr]
    if fc:
        payload["field_confidence_json"] = fc
    payload = {k: v for k, v in payload.items() if v is not None}
    artifact = {"artifact_id": sha, "kind": "object_weights", "uri_primary": art, "bytes": os.path.getsize(art), "created_utc": utc_now(),
                "locations_json": [art], "extra": {"object_id": sha, "sidecars": sorted(side)}}
    # fit-time provenance: the sidecar's statement first, else the caller's (a replay restating the prior record), else the context above
    apply_fit_provenance(payload, artifact, read_fit_provenance_sidecar(str(obj_dir)) or fit_provenance)
    if payload.get("fit_code_sha") == _UNVERIFIED:
        # the schema's spelling for a fit whose code sha no record carries needs its field_confidence_json entry (validate.py refuses the bare
        # literal at harvest), and the WRITER that states the value is the one that owes it — whichever way the hook came to state it: a
        # caller's fit_provenance, or a producer block whose checkout had no HEAD (emit.git_sha's fallback). The entry names this hook as its
        # source, the way the fit_sidecars adapter's entry names that adapter's reason
        fc_all: Dict[str, object] = dict(payload.get("field_confidence_json") or {})   # type: ignore[arg-type]
        fc_all["fit_code_sha"] = dict(FIT_CODE_SHA_UNVERIFIED_ENTRY)
        payload["field_confidence_json"] = fc_all
    return payload, artifact


#: the fit_target_kinds whose `object` payload this hook REFUSES to write when the registry's pairing CHECK would refuse it
#: (validate.fit_target_pairing_problem, the DB-exact mirror). `derived` tonight (2026-10-02: the native derived emitters' objective
#: `ce` died at the CHECK after the job had recorded "emitted"). The ingest-side validator covers EVERY kind; the writer-side gate is
#: not yet widened to the rest because two sidecar writers build payloads the CHECK would refuse by construction — a two-SI
#: distillation record from train_budget_meta.json lacks fit_teacher_si_text_sha256 / fit_prompt_pool_id / fit_prompt_pool_sha256,
#: the class-B alphasteer record lacks fit_contrast_sets_json (neither has ever reached the DB: all such rows are adapter-filed) —
#: and turning those writers into loud failures is their lanes' change, not a side effect of this one.
WRITER_REFUSES_UNPAIRABLE_KINDS = ("derived",)


def refuse_unpairable_object(payload: Mapping[str, object], where: str) -> None:
    """The WRITER'S refusal of an `object` payload the registry's fit_target_kind pairing CHECK would refuse at ingest, for the
    kinds in :data:`WRITER_REFUSES_UNPAIRABLE_KINDS`: raised BEFORE any record is written, so the producer records `failed`
    evidence naming the column instead of an "emitted" record that dies, unread, at the grading host."""
    if payload.get("fit_target_kind") not in WRITER_REFUSES_UNPAIRABLE_KINDS:
        return
    from . import validate as _VAL
    problem = _VAL.fit_target_pairing_problem(payload)
    if problem is not None:
        raise ValueError(f"{where}: the object record would be REFUSED by the registry — {problem[0]}: {problem[1]}; not written")


@guarded("fit sidecar writer")
def emit_object_from_sidecar(obj_dir: str, *, campaign: str, object_kind: str, paper_label: str, object_role: str, fit_target_kind: str,
                             fit_objective: str, script: str, registry_emit: Optional[bool] = None, spool: Optional[str] = None,
                             push_script: Optional[str] = None, session: Optional[str] = None,
                             weights_identity: Optional[Mapping[str, str]] = None, dry_run: bool = False, force: Optional[str] = None,
                             registry_root: Optional[str] = None, **more) -> List[str]:
    """The fit writers' hook: ``object`` + ``artifact`` records for the object under ``obj_dir``, into
    ``<spool>/<campaign>/object__<sha16>/``. Returns the paths written (empty when off). ``dry_run=True`` builds every record the
    hook WOULD write (the object, its artifact, the epoch rows and alias artifacts) and returns them as ``[(record_type, payload)]``
    without touching the spool, the evidence or the switch — the re-emission preflight's view of the exact payloads.
    ``fit_provenance`` (in ``more``) reaches the object and every epoch payload (object_from_sidecar)."""
    if not dry_run:
        if not emission_enabled(registry_emit):
            return []
        spool_path = resolve_spool(spool)
        if not spool_path:
            notice(f"{script}: emission on but no spool resolves; object not recorded")
            return []
    else:
        spool_path = None
    producer = emit.producer_block(script, session=session)
    payload, artifact = object_from_sidecar(str(obj_dir), campaign=campaign, object_kind=object_kind, paper_label=paper_label,
                                            object_role=object_role, fit_target_kind=fit_target_kind, fit_objective=fit_objective,
                                            producer=producer, **more)
    refuse_unpairable_object(payload, str(obj_dir))          # the writer refuses what the registry would, BEFORE the spool
    main_sha = str(artifact["artifact_id"])
    # LORADOSE (e): the artifact's location must outlive the mint; (d): ONE native statement per object_id — a second emission is
    # refused unless the caller forces it with a reason, and then the prior statement rides the record (no silent re-approval of a withdrawn
    # or superseded object; the artifact's original created_utc is kept, never overwritten by the re-emission's clock). The dry run runs
    # both checks too, so a preview refuses exactly what the write would.
    probe_spool = spool_path if not dry_run else resolve_spool(spool)
    if str(payload.get("derivation")) == DOSE_SCALE_DERIVATION:      # (e) is the DOSE COPIES' rule: a trainer's synthetic cell under /tmp (tests/test_reemit_preflight.py) still publishes
        refuse_non_durable_artifact(str(artifact["uri_primary"]), probe_spool, str(obj_dir))
    prior = existing_object_statements(main_sha, spool_path=probe_spool, campaign=campaign, registry_root=registry_root)
    if prior["records"] or prior["row"] is not None:
        what = []
        if prior["records"]:
            what.append(f"{len(prior['records'])} native record file(s) ({prior['records'][0]}, …)")   # type: ignore[index]
        if prior["row"] is not None:
            what.append(f"registry row status={prior['row']['status']!r} ({prior['row']['record_uri']})")   # type: ignore[index]
        # the one sanctioned re-emission without force: a REPLAY that restates the object's PRIOR record (`fit_provenance` carrying
        # `_source`, the prior record's uri) — it re-states provenance deliberately, with its source on the record
        restating = (more.get("fit_provenance") or {}).get("_source") if isinstance(more.get("fit_provenance"), Mapping) else None
        prior_status = (prior["row"] or {}).get("status")   # type: ignore[union-attr]
        reason = force.strip() if isinstance(force, str) and force.strip() else None
        if reason is None and restating:
            if prior_status in ("withdrawn", "superseded") and str(payload.get("status")) in ("approved", "reference"):
                raise ValueError(f"{obj_dir}: object {main_sha[:12]}… is {prior_status!r} in the registry ({prior['row']['record_uri']}); a replay restating "   # type: ignore[index]
                                 f"{restating} would re-approve it silently — pass force=<reason> if that is the intent; nothing written")
            reason = f"replay restating the prior record {restating}"
        if reason is None and dry_run:
            # the preview answers "what WOULD be written" for an already-stated object too (the trainers' preflight reads it after a publish);
            # the finding rides the previewed payload so an auditor sees both the shape and the refusal the write would make
            payload["extra"] = {**dict(payload.get("extra") or {}),   # type: ignore[dict-item]
                                "would_refuse": {"already_stated": what, "prior_status": prior_status, "override": "force=<reason>"}}
            notice(f"{script}: DRY RUN — object {main_sha[:12]}… is already stated natively ({'; '.join(what)}); a write would be REFUSED without force=<reason>")
        elif reason is None:
            raise ValueError(f"{obj_dir}: object {main_sha[:12]}… is already stated natively — {'; '.join(what)} — a second emission would restate "
                             "its status and provenance; pass force=<reason> to re-emit deliberately (the prior statement is then carried on the "
                             "record); nothing written")
        if reason is not None and prior_status in ("withdrawn", "superseded") and str(payload.get("status")) in ("approved", "reference"):
            notice(f"{script}: FORCED re-emission of {main_sha[:12]}… re-states status {payload.get('status')!r} over the registry's {prior_status!r} — "
                   f"reason: {reason}")
        if reason is not None:
            payload["extra"] = {**dict(payload.get("extra") or {}),   # type: ignore[dict-item]
                                "force": {"reason": reason, "prior_records": list(prior["records"]), "prior_status": prior_status,
                                          "prior_record_uri": (prior["row"] or {}).get("record_uri"),   # type: ignore[union-attr]
                                          "prior_artifact_uri": (prior["artifact"] or {}).get("uri_primary")}}   # type: ignore[union-attr]
        if prior["artifact"] is not None and (prior["artifact"] or {}).get("created_utc"):   # type: ignore[union-attr]
            artifact["created_utc"] = str(prior["artifact"]["created_utc"])   # type: ignore[index]
    # The fit's durable EPOCH-BOUNDARY checkpoints (epoch_1, epoch_2 and final are the named roles; every kept epoch is
    # listed in <obj_dir>/epoch_checkpoints.json) so the steered battery can serve any of them. Absent index -> nothing
    # extra (a cell without one renders exactly as before).
    # The CE trainer restores the BEST epoch's weights after the callbacks, so the published object's bytes commonly
    # ARE an epoch's (object_id == that epoch's sha). The registry folds full records by natural key with later values
    # winning, so a second `object` record under the same object_id (epoch extra, a self-parent, epoch-only locations)
    # would silently overwrite the published object's provenance and primary location.
    # Rule: ONE object row per distinct sha. An epoch whose sha equals the published object's (or an earlier epoch's)
    # is recorded as a ROLE MEMBERSHIP on that canonical object (extra.epoch_checkpoints) and as an additional location
    # of its artifact — never a second record. Distinct-sha epochs get their own object + artifact rows, parented to
    # the published object (never to themselves).
    # When the CELL attests the fit's job provenance, every epoch object row and artifact must state it too — from the
    # epoch dir's OWN sidecar (its own original created_utc, the fit's job/node/sha/substrate; written from that epoch's
    # fit-job record) — never the emitting context; a cell sidecar without the epoch sidecars is refused BEFORE any
    # write (fail closed)
    object_prov = read_fit_provenance_sidecar(str(obj_dir))
    epoch_arts = epoch_checkpoint_artifacts(str(obj_dir), object_id=main_sha)
    # hook-fix-3: the membership map is keyed by the epoch's UNIQUE key (`epoch_NN`), one entry per
    # epoch with its own identity (epoch_id, step, losses, sha, path, object_id) — never by the role name, which collapsed every
    # unnamed epoch onto one `intermediate` slot; the named roles (epoch_1, epoch_2, final) are a separate alias map onto those keys
    epochs: Dict[str, Dict[str, object]] = {}
    named_map: Dict[str, str] = {}
    canonical_of: Dict[str, str] = {main_sha: main_sha}      # sha -> the object_id that carries it
    # hook-fix-5: WEIGHT-content identity (canonical hash of the tensors) beside byte identity — the trainer's
    # best-epoch restore gives the published object an epoch's exact weights but a different file sha, which byte
    # identity cannot see; `weights_identity` (path -> weights hash, from the trainer) makes that equivalence a fact
    widentity = {str(k): str(v) for k, v in (weights_identity or {}).items()}
    published_w = widentity.get(str(artifact["uri_primary"]))
    w_owner: Dict[str, str] = {published_w: main_sha} if published_w else {}     # weights hash -> the object_id that first carried it
    distinct: List[Dict[str, object]] = []
    alias_paths: Dict[str, List[str]] = {}                   # canonical sha -> extra locations of the same bytes
    for ea in sorted(epoch_arts, key=lambda a: int(a["extra"].get("epoch") or 0)):
        sha = str(ea["artifact_id"]); ex = ea["extra"]
        ekey = f"epoch_{int(ex.get('epoch') or 0):02d}"
        entry = {"epoch": ex.get("epoch"), "epoch_id": ex.get("epoch_id"), "named_roles": list(ex.get("named_roles") or []), "step": ex.get("step"),
                 "heldout_loss": ex.get("heldout_loss"), "train_loss_last": ex.get("train_loss_last"), "artifact_sha256": sha,
                 "path": str(ea["uri_primary"])}
        if sha in canonical_of:
            entry["object_id"] = canonical_of[sha]
            entry["same_bytes_as"] = "published_object" if canonical_of[sha] == main_sha else f"epoch_object:{canonical_of[sha][:16]}"
            alias_paths.setdefault(canonical_of[sha], []).append(str(ea["uri_primary"]))
        else:
            canonical_of[sha] = sha
            entry["object_id"] = sha
            distinct.append(ea)
        w = widentity.get(str(ea["uri_primary"]))
        if w:
            entry["weights_sha256"] = w
            if w in w_owner and w_owner[w] != entry["object_id"]:
                entry["same_weights_as"] = "published_object" if w_owner[w] == main_sha else f"epoch_object:{w_owner[w][:16]}"
            else:
                w_owner.setdefault(w, str(entry["object_id"]))
        epochs[ekey] = entry
        for r in (ex.get("named_roles") or []):
            named_map[str(r)] = ekey
    if epoch_arts:
        payload["extra"] = {**dict(payload.get("extra") or {}),
                            "epoch_checkpoints": {"index": EPOCH_CHECKPOINT_INDEX, "epochs": epochs, "named": named_map,
                                                  "same_bytes_as_published": sorted(e["epoch_id"] for e in epochs.values()
                                                                                    if e.get("same_bytes_as") == "published_object" and e.get("epoch_id")),
                                                  "published_weights_sha256": published_w,
                                                  "same_weights_as_published": sorted(e["epoch_id"] for e in epochs.values()
                                                                                      if e.get("same_weights_as") == "published_object" and e.get("epoch_id"))}}
        if published_w:
            payload["extra"]["weights_sha256"] = published_w
        if alias_paths.get(main_sha):
            artifact["locations_json"] = list(dict.fromkeys(list(artifact.get("locations_json") or []) + alias_paths[main_sha]))
            artifact["extra"] = {**dict(artifact.get("extra") or {}), "epoch_checkpoint_aliases": alias_paths[main_sha]}
    if object_prov is not None:
        # fix-4: the refusal must precede ANY write — the parent's two records are written next, the epoch rows
        # after them — so every epoch dir that gets its own object row is checked for its sidecar HERE, not inside the epoch loop
        missing = [os.path.dirname(str(a["uri_primary"])) for a in distinct
                   if os.path.exists(os.path.join(os.path.dirname(str(a["uri_primary"])), "train_budget_meta.json"))
                   and read_fit_provenance_sidecar(os.path.dirname(str(a["uri_primary"]))) is None]
        if missing:
            raise ValueError(f"{obj_dir}: the cell attests the fit's job provenance ({FIT_PROVENANCE_SIDECAR}) but {len(missing)} epoch "
                             f"dir(s) carry none (first: {missing[0]}) — their object rows and artifacts would state the EMITTING context; "
                             "write each epoch's sidecar from its own fit-job record (fix-4) — nothing written")
    w = _DryRunWriter() if dry_run else emit.RecordWriter(spool_path, campaign, f"object__{main_sha[:16]}", producer, push_script=push_script)
    out = [p for p in (w.emit_object(payload), w.emit_artifact(artifact)) if p]
    for epoch_art in distinct:
        epoch_dir = os.path.dirname(str(epoch_art["uri_primary"]))
        ep_sha = str(epoch_art["artifact_id"])
        if alias_paths.get(ep_sha):
            epoch_art["locations_json"] = list(dict.fromkeys(list(epoch_art.get("locations_json") or []) + alias_paths[ep_sha]))
            epoch_art["extra"] = {**dict(epoch_art["extra"]), "epoch_checkpoint_aliases": alias_paths[ep_sha]}
        if os.path.exists(os.path.join(epoch_dir, "train_budget_meta.json")):
            # an epoch checkpoint with its own identity sidecar (written by the trainer's finalize_epoch_checkpoints:
            # the cell's meta re-keyed to the epoch artefact's sha) is an OBJECT in its own right — its own `object`
            # row (object_id = the epoch artefact's sha, servable by pinned sha like any staged object) carrying the
            # epoch identity in extra, plus its artifact row.
            if object_prov is not None and read_fit_provenance_sidecar(epoch_dir) is None:
                raise ValueError(f"{epoch_dir}: the cell attests the fit's job provenance ({FIT_PROVENANCE_SIDECAR}) but this epoch dir "
                                 "carries none — its object row and artifact would state the EMITTING context; write the epoch's sidecar "
                                 "from its own fit-job record (fix-4) — nothing written")
            try:
                ep_more = {k: v for k, v in more.items() if k != "extra"}
                ep_more["extra"] = {**dict((more.get("extra") or {})), **dict(epoch_art["extra"]), "parent_object_id": main_sha}
                ep_payload, ep_artifact = object_from_sidecar(epoch_dir, campaign=campaign, object_kind=object_kind, paper_label=paper_label,
                                                              object_role=object_role, fit_target_kind=fit_target_kind, fit_objective=fit_objective,
                                                              producer=producer, **ep_more)
                refuse_unpairable_object(ep_payload, epoch_dir)          # the epoch rows share the kind: the same writer-side refusal
                if str(ep_payload["object_id"]) == main_sha or str(ep_payload["object_id"]) != ep_sha:
                    raise RuntimeError(f"epoch object sha {str(ep_payload['object_id'])[:12]} is not the epoch artefact's {ep_sha[:12]} "
                                       f"(published object {main_sha[:12]}) — a re-emission under an existing object_id is never written")
                # the epoch artifact carries its PROVENANCE: role + the published object it belongs to (never itself) — the
                # validator admits an artifact under epoch_ckpts/ only with such a marker (the pre-fix duplicate had none)
                ep_artifact["extra"] = {**dict(ep_artifact.get("extra") or {}), "role": "epoch_checkpoint", "parent_object_id": main_sha,
                                        "named_role": epoch_art["extra"].get("named_role"), "epoch": epoch_art["extra"].get("epoch"),
                                        "epoch_id": epoch_art["extra"].get("epoch_id")}
                if alias_paths.get(ep_sha):
                    # a later epoch with THESE bytes (a plateau) is a location of this epoch's artifact, not a record of its own
                    ep_artifact["locations_json"] = list(dict.fromkeys(list(ep_artifact.get("locations_json") or []) + alias_paths[ep_sha]))
                    ep_artifact["extra"]["epoch_checkpoint_aliases"] = alias_paths[ep_sha]
                    ep_payload["extra"] = {**dict(ep_payload.get("extra") or {}), "epoch_checkpoint_aliases": alias_paths[ep_sha]}
            except Exception as e:  # noqa: BLE001
                print(f"[registry] {epoch_dir}: per-epoch object row NOT emitted ({type(e).__name__}: {e}); its artifact row still is", file=sys.stderr)
            else:
                for rec in (w.emit_object(ep_payload), w.emit_artifact(ep_artifact)):
                    if rec:
                        out.append(rec)
                continue
        rec = w.emit_artifact(epoch_art)
        if rec:
            out.append(rec)
    w.push(final=True)
    if dry_run:
        return w.records  # type: ignore[return-value]
    return out


class _DryRunWriter:
    """The RecordWriter's three calls, collecting ``(record_type, payload)`` instead of writing (emit_object_from_sidecar dry_run)."""

    def __init__(self) -> None:
        self.records: List[Tuple[str, Dict[str, object]]] = []

    def emit_object(self, payload: Mapping[str, object]) -> str:
        self.records.append(("object", json.loads(json.dumps(payload, default=str))))
        return f"dry-run:object:{str(payload.get('object_id', ''))[:16]}"

    def emit_artifact(self, payload: Mapping[str, object]) -> str:
        self.records.append(("artifact", json.loads(json.dumps(payload, default=str))))
        return f"dry-run:artifact:{str(payload.get('artifact_id', ''))[:16]}"

    def push(self, final: bool = False) -> int:
        return 0


#: the harness's legacy object-kind spellings → ``object_kind`` (OBJECT_KINDS); a spelling not listed is NOT recorded (never guessed).
#: The corpus-fit pipeline's own spellings joined 2026-09-29 (paper-companion-repo, review C1): ``lora_r4_ce`` (the rank-4 W_O LoRA fit by
#: CE on a document corpus), ``psr_ce_all`` (the ReLU steer fit by CE at every position), the LoRA subtraction and the paired-write
#: composition of their control twins -- the arms `train_two_si_objects.py` publishes for this pipeline, which the map did not name.
LEGACY_KIND_TO_OBJECT_KIND: Dict[str, str] = {
    "vector": "vector", "vector_all": "vector_all", "lora_r1": "lora", "lora_r1_all": "lora", "lora_r4": "lora",
    "psr": "relu_steer", "antidote": "antidote", "diffmean": "diffmean", "caa_static_r": "caa_static_r",
    "static_r_arditi": "static_r_arditi", "alphasteer": "alphasteer",
    "lora_r4_ce": "lora", "psr_ce_all": "relu_steer",
    "lora_r4_ce_minus_ctl": "lora_subtraction", "psr_ce_all_minus_ctl": "composed",
}
#: the two-SI campaign's system-instruction ids → ``object_role`` (OBJECT_ROLES)
SI_TO_OBJECT_ROLE: Dict[str, str] = {"safety_refusal": "safety", "si_constitution": "constitution",
                                     "si_constitution_full": "constitution", "si_irrelevant": "irrelevant"}
#: the corpus-fit pipeline's document pools → ``object_role`` (review C1): the constitution corpus and its budget-matched control, by
#: the pool's PREFIX (`constv2-mix-full`, `constv2-mix-heldout`, ...; `constv2ctl2-full`, ...) -- every other pool spelling is unmapped
CORPUS_PREFIX_TO_OBJECT_ROLE: Tuple[Tuple[str, str], ...] = (("constv2-mix-", "constitution"), ("constv2ctl2-", "control"))
#: (object family, role) → ``paper_label`` (PAPER_LABELS); anything else is ``other`` (a label, never a guess)
_LABEL_FAMILY = {"lora": "lora", "relu_steer": "relu", "relu_steer_distilled_difference": "relu", "lora_subtraction": "lora"}
_LABEL_ROLE = {"constitution": "constitution", "control": "control", "irrelevant": "control", "safety": "safety",
               "difference": "difference"}


EPOCH_CHECKPOINT_INDEX = "epoch_checkpoints.json"


def epoch_checkpoint_artifacts(obj_dir: str, *, object_id: str) -> List[Dict[str, object]]:
    """The `artifact` payloads for a published cell's epoch-boundary checkpoints: read from
    ``<obj_dir>/epoch_checkpoints.json`` (the trainer's index), one per listed checkpoint whose artefact exists and
    hashes to the recorded sha; kind ``object_weights``, ``extra.role`` ``epoch_checkpoint``, ``extra.named_role`` one
    of epoch_1 / epoch_2 / final / intermediate, with the epoch, step, losses and run identity the index recorded.
    No index -> []. A listed artefact that is missing or whose bytes do not match its recorded sha is SKIPPED with a
    notice, never described (the record must say what is on disk)."""
    idx_path = os.path.join(obj_dir, EPOCH_CHECKPOINT_INDEX)
    if not os.path.exists(idx_path):
        return []
    try:
        idx = json.loads(open(idx_path).read())
    except Exception as e:  # noqa: BLE001
        print(f"[registry] {idx_path}: unreadable epoch-checkpoint index ({type(e).__name__}) — no epoch artifacts recorded", file=sys.stderr)
        return []
    named = {v: k for k, v in (idx.get("named") or {}).items()}   # "epoch_02" -> "epoch_2", "epoch_06" -> "final"
    out: List[Dict[str, object]] = []
    for c in idx.get("checkpoints") or []:
        rel = c.get("artifact"); path = os.path.join(obj_dir, str(rel)) if rel else None
        if not path or not os.path.exists(path):
            print(f"[registry] {obj_dir}: epoch {c.get('epoch')} checkpoint {rel!r} missing — not recorded", file=sys.stderr)
            continue
        sha = _sha256_file(path)
        if c.get("artifact_sha256") and str(c["artifact_sha256"]) != sha:
            print(f"[registry] {obj_dir}: epoch {c.get('epoch')} checkpoint {rel!r} hashes to {sha[:12]}…, the index recorded "
                  f"{str(c['artifact_sha256'])[:12]}… — not recorded", file=sys.stderr)
            continue
        epoch_dir = os.path.basename(os.path.dirname(path))
        roles = sorted(k for k, v in (idx.get("named") or {}).items() if v == epoch_dir)
        named_role = "final" if "final" in roles else (roles[0] if roles else "intermediate")
        out.append({"artifact_id": sha, "kind": "object_weights", "uri_primary": path, "bytes": os.path.getsize(path),
                    "created_utc": c.get("written_utc") or utc_now(), "locations_json": [path],
                    "extra": {"object_id": object_id, "role": "epoch_checkpoint", "named_role": named_role, "named_roles": roles,
                              "epoch_id": c.get("epoch_id"), "checkpoint_key": c.get("checkpoint_key"),
                              "parent_object_sha256": c.get("parent_object_sha256"),
                              "epoch": c.get("epoch"), "step": c.get("step"), "heldout_loss": c.get("heldout_loss"),
                              "train_loss_last": c.get("train_loss_last"), "stopped": c.get("stopped"), "epochs_cap": c.get("epochs_cap"),
                              "cell_id": c.get("cell_id"), "slurm_job_id": c.get("slurm_job_id"), "hostname": c.get("hostname"),
                              "code_sha": c.get("code_sha"), "index": EPOCH_CHECKPOINT_INDEX}})
    return out


def object_kind_for_legacy_kind(spelling: str) -> Optional[str]:
    return LEGACY_KIND_TO_OBJECT_KIND.get(str(spelling))


def object_role_for_si(si_id: Optional[str]) -> Optional[str]:
    """The role of the fit TARGET the trainer names in the `<si>__<arm>` cell id: a system instruction id (the two-SI campaign), or --
    since 2026-09-29 (paper-companion-repo, review C1) -- a document pool of the corpus-fit pipeline (`constv2-mix-*` → constitution,
    `constv2ctl2-*` → control), so `train_two_si_objects._registry_publish` records the corpus fits it publishes. None otherwise."""
    if not si_id:
        return None
    return SI_TO_OBJECT_ROLE.get(str(si_id)) or object_role_for_corpus(si_id)


def object_role_for_corpus(corpus_id: Optional[str]) -> Optional[str]:
    """``constitution`` / ``control`` for a document pool of the corpus-fit pipeline (by prefix, :data:`CORPUS_PREFIX_TO_OBJECT_ROLE`);
    None for any other spelling (never guessed)."""
    if not corpus_id:
        return None
    for prefix, role in CORPUS_PREFIX_TO_OBJECT_ROLE:
        if str(corpus_id).startswith(prefix):
            return role
    return None


def object_role_for_corpus_or_si(target: Optional[str]) -> Optional[str]:
    """The role of a fit TARGET spelling -- a system instruction id (the two-SI campaign) or a document pool (the corpus-fit pipeline).
    :func:`object_role_for_si` already tries both, in that order; this name says so at the call site."""
    return object_role_for_si(target)


def paper_label_for(object_kind: str, object_role: str) -> str:
    fam = _LABEL_FAMILY.get(object_kind)
    role = _LABEL_ROLE.get(object_role)
    if object_kind == "static_r_arditi":
        return "arditi"
    if object_kind == "vector_all" and object_role == "baseline_published":
        return "arditi_allpos"                     # the refusal direction at every position: the role the live arditi_allpos rows carry (review C P1)
    if object_kind == "alphasteer":
        return "alphasteer_as_fit"
    if object_kind == "relu_steer_distilled_difference" and role == "difference":
        return "difference_relu_distilled"          # difference_relu is the undistilled paired write
    if object_kind == "composed" and role == "difference":
        return "difference_relu"
    if object_kind == "lora_subtraction" and role == "difference":
        return "difference_lora"
    if fam and role:
        return f"{role}_{fam}"
    return "other"


# --------------------------------------------------------------------------------------------
# Modal launchers (the grading host side): a spawned function call is a queued run
# --------------------------------------------------------------------------------------------


@guarded("modal launcher")
def queued_modal(run_planned: Mapping[str, object], *, modal_app_id: Optional[str], modal_call_id: Optional[str],
                 script: str, spool: Optional[str] = None, registry_emit: Optional[bool] = None,
                 session: Optional[str] = None) -> List[str]:
    """Right after ``fn.spawn(...)`` in a launcher's LOCAL ``main()``: the planned ``run`` (queued) +
    ``run_state queued`` + a ``run_patch`` carrying the Modal app / call ids, into the launcher host's
    spool (the grading host → ingested directly). Strict: an unresolved instrument refuses (nothing written)."""
    if not emission_enabled(registry_emit):
        return []
    spool_path = resolve_spool(spool)
    if not spool_path:
        notice(f"{script}: emission on but no spool resolves; queued run not recorded")
        return []
    plan = dict(run_planned)
    plan.setdefault("substrate", "modal")
    plan.setdefault("substrate_source", "recorded")
    plan.setdefault("scheduler_state", "SPAWNED")
    paths = emit.write_planned_run(spool_path, plan, script, session=session, strict=True)
    ids = identity.run_id(str(plan["campaign"]), str(plan["producer_run_key_legacy"]), str(plan["producer_object_key_legacy"]),
                          str(plan["eval_id"]), str(plan["instrument_id"]))
    w = emit.RecordWriter(spool_path, str(plan["campaign"]), ids, emit.producer_block(script, session=session))
    fields = {k: v for k, v in (("modal_app_id", modal_app_id), ("modal_call_id", modal_call_id)) if v}
    if fields:
        p = w.emit_patch(fields=fields)
        if p:
            paths.append(p)
    return paths
