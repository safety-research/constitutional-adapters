"""The JOB-SIDE plan builder: a job registers ITSELF at start from what it already knows (every
producer on main emits by default — nothing here needs a submit wrapper).

Module import is STDLIB ONLY (``python3 -S -c "import sisteer.registry.jobplan"`` works); the builder itself
imports PyYAML and the seeds loader (``adapters.dimensions``) lazily, so it runs under the compute node's
plain ``python3`` (the interpreter ``f22_select_auto`` already relies on) before any venv exists.

What the job knows at ``f22_resolve_config`` / ``s2_init`` time, and what each fact buys:

* the resolved config (``configs/sdfbase_full22_<tag>.yaml``, ``configs/misalign_full22_<target>.yaml``): its
  ``explicit_cells`` list is the job's OBJECT set — ``none`` (the unsteered anchor), ``prompted__<si>`` (a declared
  prompt: ``object_id = si:<sha256 of the text>`` from the seeds' ``si_texts``), or a staged object whose id
  the roster maps to a staged directory and ``sdfbase_full22_pins.json`` to its ``object_sha256``;
* the model key (+ the campaign's view suffix): ``legacy_spellings.yaml model_keys`` → ``model_id`` and
  ``transfer_kind``;
* the evals the job runs (declared by the job kind — the shell's table in ``full22/common.sh``, or
  ``F22_REGISTRY_EVALS``): for each, the ONE canonical instrument the seeds declare for (eval, served model,
  serving path) — ``adapters.dimensions.build_records`` mints the same ``instrument_id`` the registry's
  dimension load mints, so the job's runs join the seeded instrument row without a lookup table — and the
  eval's ``producer_spellings`` (the strings a store writes in its rows' ``task`` column), which is how the
  store recorder routes each row onto its run;
* the scheduler facts: job id, job name, the checkout sha, the substrate.

One run per (eval, object) the job DECLARES, at the registry grain (identity.run_id over the JOB's execution
key ``<job_name>@<job_id>`` and the object's id). A pair whose identity cannot be stated honestly — an object
the roster / pins do not name, an eval with no canonical instrument for this model + path, a plan the
validator refuses — is SKIPPED with a reason (``PlanBundle.skipped``), never guessed.
The store's own facts (intervention hash, harness hash, model revision, the completion artifact) arrive when
the store files its manifest under the same run id (``hooks.StoreRecorder``); the harvester's adapters remain
the backstop lineage for anything a job never registered.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from . import emit, identity
from .records import file_sha256, utc_now

UNVERIFIED = "unverified"
#: the object ids of a `prompted__<si_id>` entry: a declared prompt, no artifact
PROMPTED_PREFIX = "prompted__"
#: the kind the scope derivation is handed for a staged object, whose sha states none: the rule reads the kind for `none` and
#: `system_instruction` alone (identity.object_kind_from_id names both from the id), so any other spelling is the steered case
STEERED_OBJECT_KIND = "steered"
#: registered-at spelling every plan this module builds carries (``run.extra_json.registered_at``)
REGISTERED_AT = "job_start"
#: The padded static wave's per-level registration: the job
#: exports a JSON block naming the level and the filler CONTRACT -- ``{"tokens": <pad>, "filler_sha256": <64-hex>, "mode", "placement",
#: "separator", "tolerance", "cap": <completion cap>, "window": <the standup's max_model_len>}`` (:func:`padding_block_of` renders it
#: from the level's generated config + the filler file) -- and :func:`build_plans` then registers every declared eval on a DERIVED
#: instrument: the canonical seeded static instrument for (eval, model, path) with the level's ``padding_tokens`` /
#: ``padding_identity_sha256``, the single-pass cap, the window and one trial (identity.INSTRUMENT_ID_OPTIONAL_FIELDS, migration 0012;
#: the same fields the ``static_padding`` reader mints from a filed store), emitting the ``instrument`` record beside the runs. Five
#: levels are five instruments; the p0 control files ``padding_tokens 0`` WITH the siblings' contract, never NULL.
PADDING_ENV = "SISTEER_REGISTRY_PADDING_JSON"
#: ``sisteer.eval.attacks.static_padding.SEPARATOR`` / ``DEFAULT_TOLERANCE``, restated so this module stays importable under
#: ``python3 -S`` (tests/test_padding_registration.py asserts the two agree with the harness module)
PADDING_SEPARATOR = "\n\n"
PADDING_DEFAULT_TOLERANCE = 0.02
PADDING_CONTRACT_KEYS = identity.PADDING_IDENTITY_KEYS      # exactly the keys identity.padding_identity_sha256 hashes (and refuses on)


def padding_stamp(block: Mapping[str, Any]) -> Dict[str, Any]:
    """The level block's stamp under ``extra.padding`` of a plan and of its derived instrument: the level and the contract keys (the cap
    and the window are instrument columns, not part of the stamp)."""
    return {k: block[k] for k in ("tokens",) + PADDING_CONTRACT_KEYS}
#: default seed / roster / pins locations relative to a checkout root
SEEDS_REL = os.path.join("configs", "registry")
ROSTER_REL = os.path.join("configs", "sdfbase_full22_roster.yaml")
PINS_REL = os.path.join("configs", "sdfbase_full22_pins.json")


class JobPlanRefused(ValueError):
    """The whole registration is refused (no config, no seeds, an unknown model key): nothing is written."""


@dataclass
class PlannedRun:
    plan: Dict[str, Any]
    route: Dict[str, Any]            # {"object": <object key>, "tasks": [<producer spellings>], "eval_id": ...}
    run_id: str


@dataclass
class PlanBundle:
    runs: List[PlannedRun] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)      # (what, why)
    notes: List[str] = field(default_factory=list)
    #: dimension records the job registers BESIDE its runs (the padded wave's per-level `instrument`): (record_type, payload) pairs the
    #: run plans reference by id; `emit_cli.register_from_config` writes them first so the harvester folds the parent before the runs
    dimensions: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)

    def as_pending(self, job_id: str, job_name: Optional[str], registered_utc: Optional[str] = None) -> Dict[str, Any]:
        """The pending-file document ``emit_cli`` writes: the FIRST run at the top level (the shape every existing
        reader — ``emit_cli state`` / ``push``, ``hooks.identity_from_env`` — already understands) plus ``runs``,
        one entry per registered (eval, object) with its routing facts."""
        first = self.runs[0]
        return {"run_id": first.run_id, "job_id": str(job_id), "job_name": job_name, "registered_utc": registered_utc or utc_now(),
                "registered_at": REGISTERED_AT,
                "plan": dict(first.plan, run_id=first.run_id),
                "runs": [{"run_id": r.run_id, "route": dict(r.route), "plan": dict(r.plan, run_id=r.run_id)} for r in self.runs],
                "skipped": [{"what": w, "why": y} for w, y in self.skipped],
                "dimensions": [{"record_type": rt, "id": dimension_id(rt, payload)} for rt, payload in self.dimensions]}


def dimension_id(record_type: str, payload: Mapping[str, Any]) -> Optional[str]:
    """The id column of a dimension payload (the pending file's ``dimensions[].id``; the record directory ``emit_cli`` names it by)."""
    for k in ("instrument_id", "item_set_id", "eval_id", "model_id", "judge_id"):
        if payload.get(k):
            return str(payload[k])
    return None


# --------------------------------------------------------------------------------------------
# readers (lazy PyYAML)
# --------------------------------------------------------------------------------------------


def _yaml(path: str) -> Dict[str, Any]:
    import yaml  # the node's python3 carries PyYAML (f22_select_auto's precondition); never under python3 -S
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise JobPlanRefused(f"{path}: not a YAML mapping")
    return data


def _json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise JobPlanRefused(f"{path}: not a JSON object")
    return data


def declared_objects(config: Mapping[str, Any]) -> List[str]:
    """The config's ``explicit_cells`` (the harness's object enumeration; ``none`` first by convention)."""
    ids = config.get("explicit_cells")
    if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids):
        raise JobPlanRefused("config declares no explicit_cells list; the job's object set cannot be stated")
    return list(ids)


def _split_legacy(spelling: str) -> List[str]:
    return [s.strip() for s in str(spelling).split("|") if s.strip()]


def model_for_key(legacy_rows: Sequence[Mapping[str, Any]], model_key: str) -> Tuple[str, str]:
    """``(model_id, transfer_kind)`` for a harness model key through ``legacy_spellings.yaml model_keys``
    (exact match on one of the row's ``|``-separated spellings; a row without ``transfer_kind`` is refused)."""
    for row in legacy_rows:
        if model_key in _split_legacy(str(row.get("legacy", ""))):
            mid = row.get("model_id")
            tk = row.get("transfer_kind")
            if not isinstance(mid, str) or not mid:
                raise JobPlanRefused(f"model key {model_key!r}: legacy_spellings names no model_id")
            if not isinstance(tk, str) or not tk:
                raise JobPlanRefused(f"model key {model_key!r}: legacy_spellings names no transfer_kind")
            return mid, tk
    raise JobPlanRefused(f"model key {model_key!r} is not a legacy model spelling in legacy_spellings.yaml model_keys")


# --------------------------------------------------------------------------------------------
# seeds: the canonical instrument per (eval, model, serving path) and the evals' producer spellings
# --------------------------------------------------------------------------------------------


@dataclass
class SeedIndex:
    instruments: List[Tuple[Dict[str, Any], Dict[str, Any]]]     # (payload, extra) of every seeded instrument record
    evals: Dict[str, Dict[str, Any]]                             # eval_id -> payload
    item_sets: Dict[str, Dict[str, Any]]                         # item_set_id -> payload
    si_sha: Dict[str, str]                                       # si_id -> si_text_sha256
    legacy_model_keys: List[Dict[str, Any]]
    seeds_revision: str


def load_seed_index(seeds_dir: str) -> SeedIndex:
    from .adapters import dimensions as DIM   # PyYAML + the validator; never imported at module load
    seeds = DIM.load_seeds(seeds_dir)
    bundle = DIM.build_records(seeds, emit.producer_block("sisteer.registry.jobplan"), verify_si=False)
    instruments = [(r["payload"], r["payload"].get("extra") or {}) for r in bundle.records.get("instrument", [])]
    evals = {str(r["payload"]["eval_id"]): r["payload"] for r in bundle.records.get("eval", [])}
    item_sets = {str(r["payload"]["item_set_id"]): r["payload"] for r in bundle.records.get("item_set", [])}
    si_sha: Dict[str, str] = {}
    for row in seeds.si_texts:
        sid, sha = row.get("si_id"), row.get("si_text_sha256")
        if isinstance(sid, str) and isinstance(sha, str) and len(sha) == 64:
            si_sha[sid] = sha
    return SeedIndex(instruments, evals, item_sets, si_sha, list(seeds.legacy.get("model_keys") or []), bundle.revision_id)


def _paper_status(payload: Mapping[str, Any], extra: Mapping[str, Any]) -> str:
    return str(payload.get("paper_status") or extra.get("paper_status") or "")


def _instrument_models(extra: Mapping[str, Any]) -> List[str]:
    ids = extra.get("model_ids")
    if isinstance(ids, list):
        return [str(x) for x in ids]
    if extra.get("model_id"):
        return [str(extra["model_id"])]
    return []


def canonical_instrument(idx: SeedIndex, eval_id: str, model_id: str, serving_path: str,
                         positions_mode: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The ONE canonical seeded instrument for (eval, served model, serving path). A model-specific decode row binds the
    instrument to its model; a model-independent row (no ``model_ids``) matches every model; an instrument whose seed
    leaves ``serving_path`` unverified matches every path. None → refused naming the eval; several → refused naming them.

    With ``positions_mode`` (the mode the job will SERVE, ``emit_cli register --positions-mode``): the candidates are the seeded
    instruments for (eval, model, path) whose ``applied_positions_mode`` IS that mode, whatever their ``paper_status`` -- the
    seeds carry the non-canonical variants (``paper_status: sensitivity``) beside the canonical every-token row, and ``instrument_id``
    is run identity, so the served mode has to pick the row at registration; a run bound to the every-token instrument reads
    ``instrument_match`` mismatch on the mode for good. Several candidates: the ``canonical`` one wins when exactly one is; none, or
    several with no single canonical among them, is refused by name. A mode outside the vocabulary is refused by name before any
    row is read. Without a mode the rule above is unchanged, so every caller that passes none registers exactly as before."""
    if positions_mode is not None:
        from . import vocab as V   # a tuple of strings; kept out of the module's import line so the stdlib-only import stays visible
        if positions_mode not in V.APPLIED_POSITIONS_MODES:
            raise LookupError(f"positions mode {positions_mode!r} is not an applied-positions mode (vocab.APPLIED_POSITIONS_MODES); "
                              f"no instrument selected for eval {eval_id!r}")
    hits = []
    for payload, extra in idx.instruments:
        if str(payload.get("eval_id")) != eval_id:
            continue
        if positions_mode is None:
            if _paper_status(payload, extra) != "canonical":
                continue
        elif str(payload.get("applied_positions_mode") or "") != positions_mode:
            continue
        models = _instrument_models(extra)
        if models and model_id not in models:
            continue
        sp = payload.get("serving_path")
        if isinstance(sp, str) and sp not in (UNVERIFIED, "") and sp != serving_path:
            continue
        hits.append((payload, extra))
    if positions_mode is not None:
        if not hits:
            raise LookupError(f"no seeded instrument with applied_positions_mode {positions_mode!r} for eval {eval_id!r} on {model_id} "
                              f"via {serving_path}")
        if len(hits) > 1:
            canonical = [h for h in hits if _paper_status(*h) == "canonical"]
            if len(canonical) == 1:
                return canonical[0]
            names = sorted({str(e.get("seed_instrument_id")) for _, e in hits})
            raise LookupError(f"{len(hits)} seeded instruments with applied_positions_mode {positions_mode!r} for eval {eval_id!r} on "
                              f"{model_id} via {serving_path} and not exactly one canonical among them: {names}")
        return hits[0]
    if not hits:
        raise LookupError(f"no canonical seeded instrument for eval {eval_id!r} on {model_id} via {serving_path}")
    if len(hits) > 1:
        names = sorted({str(e.get("seed_instrument_id")) for _, e in hits})
        raise LookupError(f"{len(hits)} canonical seeded instruments for eval {eval_id!r} on {model_id} via {serving_path}: {names}")
    return hits[0]


def producer_spellings(idx: SeedIndex, eval_id: str) -> List[str]:
    ev = idx.evals.get(eval_id)
    if ev is None:
        raise LookupError(f"eval {eval_id!r} is not a seeded eval (configs/registry/evals.yaml)")
    sp = ev.get("producer_spellings_json")
    out = [str(x) for x in sp] if isinstance(sp, list) else []
    if eval_id not in out:
        out.insert(0, eval_id)
    return out


# --------------------------------------------------------------------------------------------
# objects: the config's ids → registry object ids
# --------------------------------------------------------------------------------------------


def object_id_for(entry: str, *, model_key: str, roster_selection: str, roster: Optional[Mapping[str, Any]], pins: Optional[Mapping[str, Any]],
                  si_sha: Mapping[str, str]) -> str:
    """``none`` → ``none``; ``prompted__<si>`` → ``si:<sha>``; anything else through the roster's
    ``targets.<model_key>.views.<view>.cells`` → staged id → ``pins[<model_key>][<staged>].object_sha256``."""
    if entry == "none":
        return "none"
    if entry.startswith(PROMPTED_PREFIX):
        si = entry[len(PROMPTED_PREFIX):]
        sha = si_sha.get(si)
        if not sha:
            raise LookupError(f"prompted entry {entry!r}: si_texts.yaml names no sha256 for {si!r}")
        return f"si:{sha}"
    if not roster or not pins:
        raise LookupError(f"{entry!r}: a staged object needs the roster and the pins to name its sha (none given)")
    try:
        staged = roster["targets"][model_key]["views"][roster_selection]["cells"][entry]
    except (KeyError, TypeError):
        raise LookupError(f"{entry!r}: the roster maps no staged id for target {model_key!r} selection {roster_selection!r}") from None
    if not isinstance(staged, str) or not staged:
        raise LookupError(f"{entry!r}: the roster maps it to no staged directory (a prompted or unstaged entry)")
    pin = (pins.get(model_key) or {}).get(staged) if isinstance(pins.get(model_key), dict) else None
    sha = pin.get("object_sha256") if isinstance(pin, dict) else None
    if not isinstance(sha, str) or len(sha) != 64:
        raise LookupError(f"{entry!r}: pins name no object_sha256 for staged id {staged!r} under {model_key!r}")
    return sha


# --------------------------------------------------------------------------------------------
# the plans
# --------------------------------------------------------------------------------------


def _confidence(reason: str) -> Dict[str, str]:
    return {"confidence": UNVERIFIED, "reason": reason}


def selected_pairs(pairs: Optional[Sequence[str]], *, evals: Sequence[str], declared: Sequence[str]) -> Optional[Set[Tuple[str, str]]]:
    """static-reg-fix-2: the (eval, object) pairs a phase's per-pair gate selected — the ONLY runs it registers. ``pairs`` is a
    list of ``eval:object`` tokens (a token may itself hold a comma-separated list, the shell's ``F22_REGISTRY_PAIRS``); None = every
    declared eval x declared object (the job draws them all). Refuses by name: no pair at all, a token without ``eval:object``, an eval the
    job does not declare, an object the config does not enumerate — a pair the gate named that the job cannot draw is a contradiction, and
    a run that will never be drawn must not exist in the registry."""
    if pairs is None:
        return None
    out: Set[Tuple[str, str]] = set()
    for tok in pairs:
        for t in str(tok).split(","):
            t = t.strip()
            if not t:
                continue
            if ":" not in t:
                raise JobPlanRefused(f"--pairs token {t!r}: a pair is `eval:object`")
            ev, obj = t.split(":", 1)
            if ev not in list(evals):
                raise JobPlanRefused(f"--pairs names eval {ev!r}, which this job does not declare (evals: {list(evals)})")
            if obj not in list(declared):
                raise JobPlanRefused(f"--pairs names object {obj!r}, which the config does not enumerate (explicit_cells: {list(declared)})")
            out.add((ev, obj))
    if not out:
        raise JobPlanRefused("--pairs names no (eval, object) pair — a phase whose gate selected nothing registers nothing and stands nothing up")
    return out


def build_plans(*, config_path: str, model_key: str, evals: Sequence[str], serving_path: str, campaign: str, job_id: str,
                job_name: Optional[str], roster_selection: str = "main", seeds_dir: Optional[str] = None, roster_path: Optional[str] = None,
                pins_path: Optional[str] = None, code_sha: Optional[str] = None, substrate: Optional[str] = None,
                objects: Optional[Sequence[str]] = None, lifecycle_state: str = "running",
                extra: Optional[Mapping[str, Any]] = None, now: Optional[str] = None,
                padding: Optional[Mapping[str, Any]] = None, producer_key: Optional[str] = None, token_cap: Optional[int] = None,
                token_cap_rung_index: Optional[int] = None, ladder: Optional[Sequence[int]] = None, pass_label: Optional[str] = None,
                trial_index: Optional[int] = None, pairs: Optional[Sequence[str]] = None,
                positions_mode: Optional[str] = None) -> PlanBundle:
    """Every run a job can register for itself: one plan per (declared eval, declared object). See the module
    docstring for where each fact comes from. Raises :class:`JobPlanRefused` when nothing can be registered at
    all (no config / seeds / model); per-pair problems land in ``PlanBundle.skipped``.

    ``padding`` (else ``$SISTEER_REGISTRY_PADDING_JSON``, :data:`PADDING_ENV`): the padded static wave's level block -- every
    declared eval is then registered on the level's DERIVED instrument (:func:`padded_instrument`) and the instrument record
    rides ``PlanBundle.dimensions``. Every plan stamps ``run.view_legacy`` = the job's model-key spelling (``<key>`` or
    ``<key>_<selection>``), the producer-side mapping the paper views read (review C11).

    ``positions_mode``: the applied-positions mode the job will serve; the instrument per eval is then the seeded one carrying that
    mode (:func:`canonical_instrument`), the plan's ``applied_positions_mode`` is the SELECTED instrument's as always, and the request
    is recorded under ``extra.positions_mode_requested``. Unset: the canonical instrument, as every registration before it."""
    if not os.path.isfile(config_path):
        raise JobPlanRefused(f"config {config_path!r} does not exist")
    root = _checkout_root(config_path)
    seeds_dir = seeds_dir or os.path.join(root, SEEDS_REL)
    if not os.path.isdir(seeds_dir):
        raise JobPlanRefused(f"no registry seeds at {seeds_dir} (configs/registry of the checkout)")
    config = _yaml(config_path)
    declared = list(objects) if objects else declared_objects(config)
    idx = load_seed_index(seeds_dir)
    model_id, transfer_kind = model_for_key(idx.legacy_model_keys, model_key)
    roster = _yaml(roster_path) if roster_path and os.path.isfile(roster_path) else (
        _yaml(os.path.join(root, ROSTER_REL)) if os.path.isfile(os.path.join(root, ROSTER_REL)) else None)
    pins = _json(pins_path) if pins_path and os.path.isfile(pins_path) else (
        _json(os.path.join(root, PINS_REL)) if os.path.isfile(os.path.join(root, PINS_REL)) else None)
    # the roster is keyed by the campaign's target key (the model key WITHOUT the view suffix)
    target_key = model_key[:-(len(roster_selection) + 1)] if roster_selection != "main" and model_key.endswith("_" + roster_selection) else model_key
    producer = emit.producer_block("sisteer.registry.jobplan", code_sha=code_sha, substrate=substrate)
    now = now or utc_now()
    bundle = PlanBundle(notes=[f"seeds revision {idx.seeds_revision}", f"model {model_key} -> {model_id} ({transfer_kind})"])
    if not evals:
        raise JobPlanRefused("no evals declared for this job (F22_REGISTRY_EVALS / the job-kind table); nothing to register")
    execution_key = f"{job_name or 'job'}@{job_id}"
    if producer_key is not None:
        # the sampled lane keys a static run by the rung's PRIVATE RUN NAME (`sdfbase_full22_<tag>_sampled_t<T>[_cap<N>]`,
        # the registry-wide convention the other models' adapters wrote), so two rungs of one trial are two runs; the run id hashes this key
        pk = str(producer_key).strip()
        if not pk or not all(ch.isalnum() or ch in "_.-" for ch in pk):
            raise JobPlanRefused(f"--producer-key {producer_key!r}: a producer key is one token of [A-Za-z0-9_.-]")
        execution_key = pk
    cap_facts = token_cap_facts(token_cap=token_cap, token_cap_rung_index=token_cap_rung_index, ladder=ladder, pass_label=pass_label, trial_index=trial_index)
    padding_block = _padding_block(padding)
    if padding_block is not None:
        bundle.notes.append(f"padded static registration: level {padding_block['tokens']} tokens, contract {padding_block['padding_identity_sha256'][:12]}…")
    resolved_objects: List[Tuple[str, str]] = []
    for entry in declared:
        try:
            resolved_objects.append((entry, object_id_for(entry, model_key=target_key, roster_selection=roster_selection, roster=roster, pins=pins, si_sha=idx.si_sha)))
        except LookupError as exc:
            bundle.skipped.append((f"object {entry}", str(exc)))
    selected = selected_pairs(pairs, evals=evals, declared=declared)
    if selected is not None:
        bundle.notes.append(f"gate-selected pairs: {len(selected)} of {len(evals)}x{len(declared)} declared "
                            f"({', '.join(sorted(f'{e}:{o}' for e, o in selected))})")
    if positions_mode is not None:
        bundle.notes.append(f"positions mode requested: {positions_mode} (the instrument per eval is the seeded one carrying it)")
    for eval_id in evals:
        try:
            inst, inst_extra = canonical_instrument(idx, eval_id, model_id, serving_path, positions_mode=positions_mode)
            spellings = producer_spellings(idx, eval_id)
        except LookupError as exc:
            bundle.skipped.append((f"eval {eval_id}", str(exc)))
            continue
        plan_extra: Dict[str, Any] = {}
        if positions_mode is not None:
            plan_extra["positions_mode_requested"] = positions_mode
        # the eval's category from the seeds (the SeedIndex carries every seeded eval payload): what the multi-turn scope is derived from
        eval_category = idx.evals.get(eval_id, {}).get("eval_category")
        if padding_block is not None:
            try:
                inst, inst_extra = padded_instrument(inst, inst_extra, padding_block, serving_path=serving_path, model_id=model_id, now=now)
            except (ValueError, LookupError) as exc:
                bundle.skipped.append((f"eval {eval_id}", f"padded instrument refused: {exc}"))
                continue
            bundle.dimensions.append(("instrument", dict(inst)))
            plan_extra["padding"] = padding_stamp(padding_block)
            plan_extra["padding_identity_sha256"] = padding_block["padding_identity_sha256"]
        inst, item_set = per_behaviour_item_set(idx, eval_id, inst)
        for entry, obj_id in resolved_objects:
            if selected is not None and (eval_id, entry) not in selected:
                continue   # static-reg-fix-2: a pair the gate did not select is never drawn — no run exists for it
            if cap_facts is not None:
                inst_ladder = _instrument_ladder(inst)
                if not inst_ladder:
                    # static-reg-fix-2: a rung registration needs the instrument's OWN seeded ladder — an instrument that declares
                    # none has nothing for the rung index to index (the paper reads the ladder's final rung from the instrument)
                    bundle.skipped.append((f"eval {eval_id}", f"the instrument {inst['instrument_id'][:12]} seeds no cap ladder (token_cap_ladder_json), so the "
                                                               f"registered rung cap {cap_facts['token_cap_tokens']} has no ladder to index"))
                    break
                if int(cap_facts["token_cap_tokens"]) not in inst_ladder:
                    bundle.skipped.append((f"eval {eval_id}", f"the instrument {inst['instrument_id'][:12]} ladders {inst_ladder}, which does not hold the "
                                                               f"registered rung cap {cap_facts['token_cap_tokens']}"))
                    break
            plan = _plan(inst, inst_extra, item_set, campaign=campaign, execution_key=execution_key, object_entry=entry, object_id=obj_id,
                         model_id=model_id, transfer_kind=transfer_kind, serving_path=serving_path, producer=producer, now=now,
                         lifecycle_state=lifecycle_state, job_id=job_id, job_name=job_name, view_legacy=model_key, rung=cap_facts,
                         eval_category=eval_category,
                         extra=dict(extra or {}, config=_config_stamp(config_path, root),
                                    model_key=model_key, view_legacy=roster_selection, objects_declared=list(declared), evals_declared=list(evals),
                                    seeds_revision=idx.seeds_revision, **plan_extra,
                                    **({"registered_pairs": sorted(f"{e}:{o}" for e, o in selected)} if selected is not None else {})))
            try:
                rid = emit.check_planned_run(plan, producer)
            except emit.PlanRefused as exc:
                bundle.skipped.append((f"eval {eval_id} object {entry}", str(exc)))
                continue
            bundle.runs.append(PlannedRun(plan, {"object": entry, "object_id": obj_id, "eval_id": eval_id, "tasks": spellings}, rid))
    return bundle


# --------------------------------------------------------------------------------------------
# the padded static wave: one DERIVED instrument per level (module constant PADDING_ENV)
# --------------------------------------------------------------------------------------------


def _padding_block(padding: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The level block from the argument, else ``$SISTEER_REGISTRY_PADDING_JSON``; None when neither is set. Validated: ``tokens``
    a non-negative int, every contract key present and non-null, ``cap`` a positive int, ``window`` a positive int or absent;
    ``padding_identity_sha256`` is computed here (identity.padding_identity_sha256 over the contract)."""
    raw: Any = padding
    if raw is None:
        env = os.environ.get(PADDING_ENV, "").strip()
        if not env:
            return None
        try:
            raw = json.loads(env)
        except ValueError as exc:
            raise JobPlanRefused(f"{PADDING_ENV} is not JSON: {exc}")
    if not isinstance(raw, Mapping):
        raise JobPlanRefused(f"{PADDING_ENV}: expected a JSON object, got {type(raw).__name__}")
    block = dict(raw)
    try:
        tokens = int(block["tokens"])
    except (KeyError, TypeError, ValueError):
        raise JobPlanRefused(f"padding block lacks an integer 'tokens' (the level): {block}")
    if tokens < 0:
        raise JobPlanRefused(f"padding block: tokens {tokens} < 0")
    block.setdefault("separator", PADDING_SEPARATOR)
    block.setdefault("tolerance", PADDING_DEFAULT_TOLERANCE)
    missing = [k for k in PADDING_CONTRACT_KEYS if block.get(k) is None]
    if missing:
        raise JobPlanRefused(f"padding block lacks the contract key(s) {missing} (every level, the p0 control included, names its siblings' contract; D560)")
    try:
        cap = int(block["cap"])
    except (KeyError, TypeError, ValueError):
        raise JobPlanRefused(f"padding block lacks an integer 'cap' (the single-pass completion cap): {block}")
    if cap <= 0:
        raise JobPlanRefused(f"padding block: cap {cap} <= 0")
    window = block.get("window")
    if window is not None:
        try:
            window = int(window)
        except (TypeError, ValueError):
            raise JobPlanRefused(f"padding block: window {window!r} is not an integer")
        if window <= 0:
            raise JobPlanRefused(f"padding block: window {window} <= 0")
    out = {"tokens": tokens, "cap": cap, "window": window}
    out.update({k: block[k] for k in PADDING_CONTRACT_KEYS})
    out["tolerance"] = float(out["tolerance"])
    out["padding_identity_sha256"] = identity.padding_identity_sha256(out)
    return out


def padded_instrument(base: Mapping[str, Any], base_extra: Mapping[str, Any], block: Mapping[str, Any], *, serving_path: str, model_id: str,
                      now: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(payload, extra)`` of the level's instrument: the canonical seeded static instrument ``base`` with the store's OWN facts --
    the single-pass cap as a one-level ladder read ``fixed``, the standup window, one trial, the job's serving path -- PLUS
    ``padding_tokens`` and ``padding_identity_sha256`` (identity.INSTRUMENT_ID_OPTIONAL_FIELDS, migration 0012), minted through
    ``identity.instrument_id`` exactly as the ``static_padding`` reader mints it from a filed store (its ``padded_instrument_fields``,
    the same harness_version spelling), so a natively registered level and a later re-read of its store meet on one id. Refuses a
    derived id equal to the base's (a padded instrument is never the unpadded one)."""
    from .adapters import static_padding as _SP    # lazy: pure python, never at module load (python3 -S)
    from . import decode_policy as _DP
    decode = {k: base.get(f"decode_{k}") for k in _DP.PARAM_KEYS}
    thinking = str(base.get("thinking_mode") or "off")
    harness_version = ("scripts/jailbreak_run.py + sisteer.eval.attacks.served_target (ServedStaticDefense) + sisteer.eval.attacks.static_padding "
                       f"({_SP.PADDING_SCHEMA}: PG-19 filler, placement {block.get('placement')}, mode {block.get('mode')}) @ the generating tree "
                       "(PROVENANCE.md beside the mirrored stores; not a producer field)")
    fields, changed = _SP.padded_instrument_fields(base, pad=int(block["tokens"]), block=block, cap=int(block["cap"]), window=block.get("window"),
                                                   decode=decode, decode_source=str(base.get("decode_source") or UNVERIFIED),
                                                   harness_version=harness_version, judge_id=base.get("judge_id"), serving_path=serving_path,
                                                   thinking_mode=thinking)
    if fields["instrument_id"] == base.get("instrument_id"):
        raise ValueError(f"the derived padded instrument hashes to the base instrument's id {str(base.get('instrument_id'))[:12]}…")
    payload: Dict[str, Any] = dict(base)
    payload.update(fields)                      # the derived instrument_id included
    payload["notes"] = (f"{'Zero-point' if int(block['tokens']) == 0 else 'Padded'} static instrument registered natively at job start: the canonical "
                        f"seeded static instrument {str(base.get('instrument_id'))[:12]}… for {base.get('eval_id')} on {model_id} with the level "
                        f"({int(block['tokens']):,} tokens), the filler contract, the single-pass cap {int(block['cap']):,}, the window and one trial "
                        "(identity.INSTRUMENT_ID_OPTIONAL_FIELDS, migration 0012; D560 zero point).")
    extra: Dict[str, Any] = {k: v for k, v in base_extra.items() if k != "seed_instrument_id"}
    extra.update({"derived_from_instrument_id": base.get("instrument_id"), "derived_from_seed_instrument": base_extra.get("seed_instrument_id"),
                  "padding": padding_stamp(block), "padding_identity_sha256": block["padding_identity_sha256"],
                  "changed_fields": changed, "model_id": model_id, "registered_at": REGISTERED_AT, "registered_utc": now or utc_now()})
    payload["extra"] = extra
    return payload, extra


def padding_block_of(config_path: str, filler_path: str, *, window: Optional[int] = None, cap: Optional[int] = None) -> Dict[str, Any]:
    """The level block a padded sbatch exports as ``$SISTEER_REGISTRY_PADDING_JSON``, from the level's generated config
    (``attacks.static_padding`` — ``tokens`` / ``placement`` / ``mode`` [/ ``tolerance``]; ABSENT on the p0 control, which then
    takes ``tokens`` 0 and its siblings' contract from ``<same dir>/*_p<N>.yaml``, N > 0) and the filler FILE (its sha256 is
    the contract's ``filler_sha256``). ``cap`` defaults to the config's ``attacks.static_max_new_tokens``."""
    cfg = _yaml(config_path)
    attacks = cfg.get("attacks") if isinstance(cfg.get("attacks"), Mapping) else {}
    block = attacks.get("static_padding") if isinstance(attacks.get("static_padding"), Mapping) else None
    if block is None:
        # the zero point: the contract of a padded sibling in the same generated family
        stem = os.path.basename(config_path)
        head = stem.rsplit("_p", 1)[0]
        sibs = sorted(glob.glob(os.path.join(os.path.dirname(config_path), head + "_p*.yaml")))
        for sib in sibs:
            if os.path.abspath(sib) == os.path.abspath(config_path):
                continue
            sc = _yaml(sib)
            sb = (sc.get("attacks") or {}).get("static_padding") if isinstance(sc.get("attacks"), Mapping) else None
            if isinstance(sb, Mapping) and int(sb.get("tokens") or 0) > 0:
                block = dict(sb, tokens=0)
                break
        if block is None:
            raise JobPlanRefused(f"{config_path}: no attacks.static_padding block and no padded sibling `{head}_p<N>.yaml` to take the contract from (D560)")
    out = {"tokens": int(block.get("tokens") or 0), "placement": block.get("placement"), "mode": block.get("mode"),
           "separator": block.get("separator", PADDING_SEPARATOR), "tolerance": float(block.get("tolerance", PADDING_DEFAULT_TOLERANCE)),
           "filler_sha256": file_sha256(filler_path), "cap": int(cap if cap is not None else attacks.get("static_max_new_tokens")), "window": window}
    if out["placement"] is None or out["mode"] is None:
        raise JobPlanRefused(f"{config_path}: attacks.static_padding lacks placement / mode")
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """``python3 -m sisteer.registry.jobplan padding-block --config CFG --filler FILE [--window W] [--cap C]`` prints the JSON block a
    padded sbatch exports as ``$SISTEER_REGISTRY_PADDING_JSON`` before its registration step."""
    import argparse
    ap = argparse.ArgumentParser(prog="python3 -m sisteer.registry.jobplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("padding-block")
    pb.add_argument("--config", required=True)
    pb.add_argument("--filler", required=True)
    pb.add_argument("--window", type=int, default=None)
    pb.add_argument("--cap", type=int, default=None)
    pi = sub.add_parser("padded-instrument-id", help="the 64-hex instrument_id a padded level registers on (for the front door's gate inventory)")
    pi.add_argument("--seeds", required=True, help="the registry seeds directory (configs/registry)")
    pi.add_argument("--model-id", required=True)
    pi.add_argument("--eval", required=True, dest="eval_id")
    pi.add_argument("--serving-path", default="lora_native")
    pi.add_argument("--padding-json", required=True, help="the block padding-block prints")
    args = ap.parse_args(argv)
    if args.cmd == "padding-block":
        print(json.dumps(padding_block_of(args.config, args.filler, window=args.window, cap=args.cap), sort_keys=True))
        return 0
    if args.cmd == "padded-instrument-id":
        idx = load_seed_index(args.seeds)
        base, base_extra = canonical_instrument(idx, args.eval_id, args.model_id, args.serving_path)
        block = _padding_block(json.loads(args.padding_json))
        payload, _extra = padded_instrument(base, base_extra, block, serving_path=args.serving_path, model_id=args.model_id)
        print(json.dumps({"instrument_id": payload["instrument_id"], "base_instrument_id": base["instrument_id"], "eval_id": args.eval_id,
                          "padding_tokens": block["tokens"], "padding_identity_sha256": block["padding_identity_sha256"],
                          "token_cap_tokens": block["cap"], "context_window_tokens": block["window"]}, sort_keys=True))
        return 0
    return 2


def _config_stamp(config_path: str, root: str) -> str:
    """extra.config: the config's path RELATIVE to its checkout when a checkout holds it (`configs/registry` under `root`), else ABSOLUTE —
    a narrowed copy under $SHARED/registry/pending/ (cells-fix) has no checkout, and `_checkout_root`'s grandparent fallback once made it
    `pending/<job>.config.yaml`, which the closer resolved against the job's checkout (837557, 2026-10-02)."""
    cp = os.path.abspath(config_path)
    if os.path.isdir(os.path.join(root, "configs", "registry")) and cp.startswith(os.path.abspath(root) + os.sep):
        return os.path.relpath(cp, root)
    return cp


def _checkout_root(config_path: str) -> str:
    """The checkout a config lives in: the nearest ancestor holding ``configs/registry``, else the config's grandparent."""
    p = os.path.abspath(config_path)
    d = os.path.dirname(p)
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "configs", "registry")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(os.path.dirname(p))


_INSTRUMENT_TO_RUN = ("harness", "harness_version", "scoring_rule", "prompt_template_sha256", "decode_seed_rule", "decode_source",
                      "token_cap_tokens", "thinking_mode")


#: a per-behaviour Petri eval `petri_<behaviour>` registers against the item set DECLARED for it (selected by eval
#: ownership, never by the spelling `petri_scenarios_<behaviour>` — a seed revision may rename or version the ten-scenario set)
PETRI_BEHAVIOUR_EVAL_PREFIX = "petri_"
PETRI_POOLED_EVAL = "petri_pooled"


def per_behaviour_item_set(idx: SeedIndex, eval_id: str, inst: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(instrument as the plan reads it, item set payload)`` for one eval. A per-behaviour Petri eval (``petri_<behaviour>``, not the
    pooled one) plans against the item set whose seed declaration NAMES this eval (``eval_id: petri_<behaviour>``; six such sets exist,
    the ten-scenario partition of petri_scenarios_6x10) — the plan's ``item_set_id`` and ``n_items_planned`` follow that set
    while the instrument row and its id stay the seeded ones (the instrument's identity is the eval's protocol, not the run's planned
    count). Selection is by OWNERSHIP, never by spelling: EVERY item set whose seeded ``eval_id`` names the
    behaviour is an owner — the instrument's own set included (excluding it let a versioned replacement B pass as
    the sole claimant beside the instrument's own matching set A, planning instrument A with set B). Two or more owners refuse by name
    (one behaviour, one ten-scenario set); the instrument's own set as the sole owner is kept; a different sole owner is selected; an
    owner stating no usable ``n_items`` refuses too (its plan would misreport). The instrument's own (pooled) set stands when no set
    declares the eval at all — the earlier seeds folded — and for every non-Petri or pooled eval."""
    own = idx.item_sets.get(str(inst.get("item_set_id")), {})
    if not eval_id.startswith(PETRI_BEHAVIOUR_EVAL_PREFIX) or eval_id == PETRI_POOLED_EVAL:
        return dict(inst), own
    own_id = str(inst.get("item_set_id") or "")
    owners = sorted((str(sid), p) for sid, p in idx.item_sets.items()
                    if isinstance(p, Mapping) and str(p.get("eval_id") or "") == eval_id)
    if not owners:
        return dict(inst), own
    if len(owners) > 1:
        raise JobPlanRefused(f"eval {eval_id}: {len(owners)} item sets declare it ({', '.join(sid for sid, _ in owners)}"
                             f"{'; the instrument binds ' + own_id if own_id in {sid for sid, _ in owners} else ''}) — one behaviour, one "
                             "ten-scenario set; the seeds must name exactly one, never a guess here")
    sid, per = owners[0]
    n = per.get("n_items")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise JobPlanRefused(f"eval {eval_id}: its declared item set {sid} states no usable n_items ({n!r}) — a per-behaviour plan "
                             "against it would misreport the planned count; fix the seeds")
    if sid == own_id:
        return dict(inst), own                     # the instrument's own set is the sole owner: kept, nothing moves
    return {**dict(inst), "item_set_id": sid}, per


PASS_LABELS = ("base_cap", "ladder_rung")


def _instrument_ladder(inst: Mapping[str, Any]) -> List[int]:
    """The instrument's own cap ladder (``token_cap_ladder_json``: a list, or its JSON text), [] when it declares none."""
    v = inst.get("token_cap_ladder_json")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return []
    return [int(x) for x in v] if isinstance(v, (list, tuple)) else []


def token_cap_facts(*, token_cap: Optional[int], token_cap_rung_index: Optional[int], ladder: Optional[Sequence[int]], pass_label: Optional[str],
               trial_index: Optional[int]) -> Optional[Dict[str, Any]]:
    """static-reg-fix: the attained rung a static phase registers its runs at — ``token_cap_tokens`` = the DRAWN cap, the rung's index
    in the ladder and the ladder itself (``[4096, 8192, 16384]``) — all three together or none; ``cap == ladder[token_cap_rung_index]``; an optional
    ``pass_label`` (base_cap for rung 0, ladder_rung above) and ``trial_index``. Refuses by name: the cap is a RUN_IDENTITY field no patch can
    correct later, and the rung / ladder are not patchable at all."""
    given = [x is not None for x in (token_cap, token_cap_rung_index, ladder)]
    if not any(given) and pass_label is None and trial_index is None:
        return None
    if not all(given):
        raise JobPlanRefused("--token-cap, --rung-index and --ladder go together (the drawn cap, its index, the ladder it indexes)")
    try:
        cap = int(token_cap); idx = int(token_cap_rung_index); lad = [int(x) for x in ladder]
    except (TypeError, ValueError):
        raise JobPlanRefused(f"--token-cap {token_cap!r} / --rung-index {token_cap_rung_index!r} / --ladder {ladder!r}: integers, please")
    if len(lad) < 2 or lad != sorted(lad) or len(set(lad)) != len(lad) or any(x <= 0 for x in lad):
        raise JobPlanRefused(f"--ladder {lad}: an ascending list of at least two positive caps")
    if idx < 0 or idx >= len(lad) or lad[idx] != cap:
        raise JobPlanRefused(f"--token-cap {cap} is not ladder[{idx}] of {lad} — the run's cap must be the rung it names")
    if pass_label is not None:
        if pass_label not in PASS_LABELS:
            raise JobPlanRefused(f"--pass-label {pass_label!r}: one of {PASS_LABELS}")
        if (pass_label == "base_cap") != (idx == 0):
            raise JobPlanRefused(f"--pass-label {pass_label} does not fit rung index {idx} (base_cap is rung 0, ladder_rung above it)")
    if trial_index is not None and (isinstance(trial_index, bool) or int(trial_index) < 0):
        raise JobPlanRefused(f"--trial-index {trial_index!r}: a non-negative integer")
    return {"token_cap_tokens": cap, "token_cap_rung_index": idx, "token_cap_ladder_json": lad, "pass_label": pass_label,
            "trial_index": (int(trial_index) if trial_index is not None else None)}


def _plan(inst: Mapping[str, Any], inst_extra: Mapping[str, Any], item_set: Mapping[str, Any], *, campaign: str, execution_key: str,
          object_entry: str, object_id: str, model_id: str, transfer_kind: str, serving_path: str, producer: Mapping[str, Any], now: str,
          lifecycle_state: str, job_id: str, job_name: Optional[str], extra: Mapping[str, Any], view_legacy: str,
          rung: Optional[Mapping[str, Any]] = None, eval_category: Optional[str] = None) -> Dict[str, Any]:
    from . import derive as D   # stdlib-only (the rule table and the vocabulary); kept lazy so the module's import line stays as it is
    unsteered = object_id == "none"
    fc: Dict[str, Any] = {}
    mode = "none" if unsteered else str(inst.get("applied_positions_mode") or UNVERIFIED)
    # the multi-turn steering scope is DERIVED here from the selected instrument's mode, the seeded eval's category and the kind the
    # object's id states (`si:<sha>` is a prompted object, which steers no positions whatever mode the instrument declares; a staged
    # artifact's sha states no kind, and the rule reads the kind for the anchor and the prompt alone) -- never the single-turn default a
    # job-start registration used to stamp on every multi-turn, Petri and agentic run; a category the seeds do not name leaves it
    # unverified with the reason, never a guess
    if unsteered:
        scope, scope_entry = "not_applicable", None
    elif eval_category is None:
        scope, scope_conf, scope_src = UNVERIFIED, UNVERIFIED, f"the seeds name no eval_category for eval {inst.get('eval_id')!r}"
        scope_entry = D.confidence(scope, scope_conf, scope_src)
    else:
        kind = identity.object_kind_from_id(object_id) or STEERED_OBJECT_KIND
        scope, scope_conf, scope_src = D.multi_turn_steering_scope(mode, str(eval_category), kind)
        scope_entry = D.confidence(scope, scope_conf, scope_src)
    p: Dict[str, Any] = {
        "campaign": campaign,
        "producer_run_key_legacy": execution_key,
        "producer_object_key_legacy": object_entry,
        "lifecycle_state": lifecycle_state,
        "queued_utc": now,
        # the producer's transfer-setting spelling on the COLUMN the paper views read (review C11): every job-start registration
        # before 2026-09-29 parked it in extra.view_legacy only, so all 320 such runs read NULL here
        "view_legacy": view_legacy,
        "model_id": model_id,
        "object_id": object_id,
        "transfer_kind": "none" if unsteered else transfer_kind,
        "eval_id": str(inst["eval_id"]),
        "item_set_id": str(inst["item_set_id"]),
        "instrument_id": str(inst["instrument_id"]),
        "instrument_match": UNVERIFIED,
        "serving_path": serving_path,
        "applied_positions_mode": mode,
        "multi_turn_steering_scope": scope,
        "code_sha": producer.get("code_sha", UNVERIFIED),
        "substrate": producer.get("substrate", "grading_host"),
        "substrate_source": "derived",
        "cluster_job_id": str(job_id),
        "submit_evidence_uri": f"slurm://job/{job_id}" if producer.get("substrate") in ("cluster_a", "cluster_b") else f"job://{job_id}",
    }
    if lifecycle_state == "running":
        p["started_utc"] = now
    if job_name:
        p["job_name"] = job_name
    for k in _INSTRUMENT_TO_RUN:
        v = inst.get(k)
        p[k] = v if v not in (None, "") else UNVERIFIED
    if rung is not None:
        # static-reg-fix: the run's cap is the rung it DREW at, not the instrument's read-at cap; the rung index and the ladder ride the
        # run (neither is patchable afterwards); the instrument stays the seeded one — the rung is told apart by the producer key, never the id
        p["token_cap_tokens"] = int(rung["token_cap_tokens"])
        p["token_cap_rung_index"] = int(rung["token_cap_rung_index"])
        p["token_cap_ladder_json"] = list(rung["token_cap_ladder_json"])
        if rung.get("pass_label"):
            p["pass_label"] = rung["pass_label"]
        if rung.get("trial_index") is not None:
            p["trial_index"] = int(rung["trial_index"])
    n_items = item_set.get("n_items")
    if isinstance(n_items, int) and not isinstance(n_items, bool):
        p["n_items_planned"] = n_items
    fc["instrument_match"] = _confidence("declared from the seeds at job start; the harvester compares the served decode to the instrument")
    if p["prompt_template_sha256"] == UNVERIFIED:
        fc["prompt_template_sha256"] = _confidence("the seeded instrument records no prompt-template hash")
    if p["applied_positions_mode"] == UNVERIFIED:
        fc["applied_positions_mode"] = _confidence("the seeded instrument declares no positions mode")
    if scope_entry is not None:
        fc["multi_turn_steering_scope"] = scope_entry
    if p["code_sha"] == UNVERIFIED:
        fc["code_sha"] = _confidence("git rev-parse HEAD unavailable in the registering checkout")
    p["field_confidence_json"] = fc
    p["extra"] = dict(extra, registered_at=REGISTERED_AT, grain="declared_eval_x_object", seed_instrument_id=inst_extra.get("seed_instrument_id"),
                      **({"rung_registration": dict(rung)} if rung is not None else {}))
    return p


if __name__ == "__main__":
    raise SystemExit(_main())
