"""The HF POST-LOAD intervention hash of a fitted object — the identity a
re-run must reproduce, derived without weights and without a GPU.

WHY THIS EXISTS, AND WHY IT IS NOT THE SERVED HASH. The campaign's published
capability numbers were produced on the HF engine, so every old store records
the hash ``HFEngine.arm`` computed after LOADING the object:
``resolve_and_hash(spec, hydrated_object, coeff, …)`` over the artifact bytes
plus the RESOLVED spec (the object's own layers, positions, header length…).
The lens-served route stamps something else: ``PetriCell.intervention_hash`` —
``manifest.petri_intervention_hash`` over the artifact bytes, the sidecar
bytes and the REGISTRY's resolved spec (the blob's own stored non-tensor
fields plus ``kind`` / ``arm`` / ``si_id``, as ``registry_adapter.
TrainTestCellSource`` resolves it), computed by ``defense_cells.
enumerate_cells`` and copied verbatim onto ``RouteEntry.intervention_hash``
by ``proxy.routes_from_cells``. THREE numbers exist for one object file, and
on gemma4e4b ``safety_refusal__psr_kl_all`` they are: HF post-load
``d50f46c19733de40`` (the old HF store; the published expectation), the lens
ROUTE hash ``d6e1b51a96588745`` (every tinycap served store, same artifact
sha — §PW21-stage), and ``runtime.cell_intervention_hash`` ``8d7131521d4e8f88``
— the capability-manifest engines' STATIC spec hash, which this module used
to report as "the lens static hash" although NOTHING on the tinycap standup's
proxy route stamps it (E40 M4: the route hash is what a served store records;
the field named ``lens_static_intervention_hash`` was the third number).

So a preflight that compared the SERVED hash against the inventory's expected
hashes would mismatch on every cell and prove nothing. This module reproduces
the HF derivation instead, which is what those expectations are, so
"the object we are about to serve IS the object the published number came
from" becomes a CPU check that runs before any GPU minute — and it reports,
beside it, the hash the route WILL stamp (``lens_route_intervention_hash``,
derived through the registry adapter's own resolution), so a served store's
identity can be predicted from the preflight and pinned by a test against
``RouteEntry.intervention_hash``.

HOW IT AVOIDS WEIGHTS. ``PSRObject.load_blob`` and its siblings read only
``hidden_size(model)`` and ``get_decoder_layers(model)`` off the model, so a
weightless shell built from the pinned HF config on ``torch.device("meta")`` is
enough. Nothing is moved to a device and nothing is downloaded: the config and
tokenizer come from the local/volume HF cache (``HF_HUB_OFFLINE``).

Provenance: this is §PW21-stage's ``derive_ihash.py``, which verified both
staged objects against their old stores (llama3b ``93b6466dd791cc7e``, nors
``35bc0822bfdc7dba``), carried into the repo as a module so the launcher's
preflight runs the SAME derivation rather than a second one. The one deliberate
difference from ``HFEngine._steering_arm`` is documented at the call site: no
``obj.to_(model.device)`` — a meta shell has no device, and the move cannot
enter the hash, which covers artifact bytes plus resolved-spec fields only.

CLI (what the launcher's preflight runs, in a subprocess against the STAGED
tree so the derivation is the code the wave runs)::

    python -m sisteer.capeval.tinycap.object_identity --request /path/req.json

``req.json``::

    {"model": "google/gemma-4-E4B-it", "revision": "<40-hex>",
     "config": "/vol/tinycap/repos/sisteer-<sha>/configs/two_si_gemma4e4b_reduced.yaml",
     "config_overrides": {"layer": 15},          # optional (llama3b's nulls)
     "objects_dir": "/vol/objects/scale-12/gemma4e4b",
     "allow_out_of_support_thinking": false,      # optional (PW-oos, U10 item 5)
     "targets": [{"cell": "safety_refusal__psr_kl_all",
                  "si_id": "safety_refusal", "arm": "psr_kl_all",
                  "expected": "d50f46c19733de40"}]}

and one JSON object is printed on stdout::

    {"model": …, "revision": …, "ok": true|false, "derivation": …,
     "chat_template_hash": …, "generation_header_len": 3,
     "results": [{"cell": …, "intervention_hash": …, "expected": …,
                  "match": true, "artifact_sha256": …, "resolved_spec": {…},
                  "response_header_len": 3, "header_guard_will_pass": true,
                  "header_guard": "checked_against_served_generation_header",
                  "harness_cell_id": "si_safety__psr_kl_all",
                  "lens_route_intervention_hash": …} …]}

A target that raises is reported with ``error`` and turns ``ok`` false — the
sweep is never cut short, so one bad object does not hide the state of the
other nine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# torch-free: the named transfer experiments and their (fit, served) pairs
from sisteer.capeval.serving import serve_declaration as sd

#: The exact composition this module performs, recorded in every result so a
#: reader never has to guess which of the two hashes a number is.
DERIVATION = (
    "sisteer.methods._hydrate_object(artifact, meta-shell, tok) + "
    "capeval.serving.resolved_spec.resolve_and_hash(spec, obj, coeff, EngineError, "
    "arm_response_mask=HFEngine._arm_response_mask) == HFEngine._steering_arm "
    "minus obj.to_(model.device)")
#: What the LENS route stamps instead, for the same object — recorded beside the
#: HF hash so the two are never confused (and so a served store's hash can be
#: predicted from here). This is the derivation ``defense_cells.enumerate_cells``
#: performs for a non-stacked object cell, over the registry entry
#: ``registry_adapter.TrainTestCellSource.get(<harness id>)`` supplies; the
#: proxy copies the result onto ``RouteEntry.intervention_hash`` unchanged.
ROUTE_DERIVATION = (
    "capeval.manifest.petri_intervention_hash(TrainTestCellSource.get(harness_cell_id)"
    ".resolved_spec, artifact_bytes, sidecar_bytes) == defense_cells.enumerate_cells -> "
    "PetriCell.intervention_hash == proxy.routes_from_cells -> RouteEntry.intervention_hash")
#: NOT reported any more (E40 M4): ``runtime.cell_intervention_hash(spec, coeff)``
#: is the capability-manifest engines' STATIC spec hash; nothing on the tinycap
#: standup's proxy route stamps it, and reporting it as "the lens static hash"
#: had a reader take it for the served store's identity.
RUNTIME_STATIC_HASH_NOTE = ("runtime.cell_intervention_hash is the capability-manifest engines' "
                            "static spec hash — not what the lens ROUTE stamps")
#: The serving pins every published capability number was produced at, and that
#: every old store records; see :func:`load_config`.
SERVED_POSITIONS = "response"
SERVED_SCALE = 1.0

#: Why the fit/serve header guard's PREDICTED verdict came out the way it did,
#: recorded on every result beside ``header_guard_will_pass`` so a pass says
#: what it means instead of being silent. See :func:`header_guard_prediction`.
HEADER_GUARD_NOT_APPLICABLE = "not_applicable_no_recorded_header"
HEADER_GUARD_CHECKED = "checked_against_served_generation_header"
HEADER_GUARD_MISMATCH = "refuse_recorded_header_differs_from_served"
HEADER_GUARD_NOT_EVALUATED = "not_evaluated_derivation_raised"
#: PW-oos (U10 item 5): the recorded header differs from the header the run
#: will SERVE, and ``--allow-out-of-support-thinking`` was named — so the
#: mismatch is REPORTED (with both widths and both id lists) instead of
#: blocking the plan. Never reachable without the override.
HEADER_GUARD_OVERRIDDEN = "predicted_refuse_overridden"
#: A NAMED transfer experiment (``serve_declaration.SERVE_EXPERIMENTS``) that
#: serves the response-fit object at ``all`` positions
#: (``fit_response_serve_all_positions``, spelled ``<cell>@all_positions``): the
#: served mask is not header-located, so the lens leg has NO width to compare
#: the record against and never fires — a recorded header, stale or not, is
#: not refused. Said by name, so a pass there is never read as "checked".
HEADER_GUARD_NOT_COMPARED_ALL_POSITIONS = "not_compared_named_all_positions_serve_has_no_header_mask"

#: The served position modes whose mask is LOCATED by the assistant header —
#: the only modes under which the lens leg compares the recorded width
#: (``lens_adapters.RESPONSE_MASK_MODES``, pinned equal by the executing test;
#: a literal here because that module imports torch and this one runs on CPU
#: before any GPU minute). :data:`SERVED_POSITIONS` is one of them.
HEADER_MASKED_SERVES: Tuple[str, ...] = ("response", "response_all_turns")


def header_guard_prediction(recorded: object, served_header_len: int, *,
                            allow_out_of_support: bool = False,
                            serve_experiment: Optional[str] = None
                            ) -> Tuple[bool, str]:
    """The fit/serve header guard's verdict for one object, PREDICTED —
    ``(will_pass, reason)``.

    ``allow_out_of_support`` (PW-oos) turns a MISMATCH into
    :data:`HEADER_GUARD_OVERRIDDEN` — ``will_pass`` true, because under the
    override the launch is not blocked, and the mismatch itself is the finding
    the run is there to record. It never turns a PASS into anything else, and
    with the flag absent this function is unchanged.

    THE RULE THE REAL GUARDS RUN IS KEYED ON WHETHER THE OBJECT RECORDS A
    HEADER, NEVER ON ITS KIND. All three of them:

    * ``eval.jailbreak._check_recorded_header`` — the fit-mask leg, reached
      from ``HFEngine._arm_response_mask`` inside :func:`derive_one` itself:
      it walks ``fit_response_header_len`` / ``response_header_len`` and
      "``None`` (all-position fits, fresh objects) is not a record and
      passes"; a recorded int that differs from this tokenizer's width
      raises.
    * ``serving.lens_adapters._fit_leg_positions`` — the lens serving leg, the
      guard a tinycap standup actually runs: ``fit_h =
      blob.get("response_header_len")`` and the width comparison happens only
      ``if fit_h is not None``.
    * ``serving.lora_export._export_rank1_lora`` / ``_export_multilayer_lora``
      — the PEFT export leg: the record is validated when present and its
      absence is not an error; it is never COMPARED against a served width,
      because a native adapter carries no mask to compare one to.

    So an object that records NO header has nothing for the guard to compare
    and the guard does not refuse it. Every LoRA blob is that object — a
    native adapter, not a header-masked lens object — and predicting a
    refusal on it failed every ``*__lora_r4`` cell of the 10-cell Gemma waves
    at preflight (O22, 2026-09-06): a prediction bug, not a serving fact.

    The guards' OTHER conjunct — that the serve is response-masked
    (``lens_adapters._served_positions`` -> ``mode in RESPONSE_MASK_MODES``:
    ``response`` or the all-turns carve-out ``response_all_turns``, which runs
    the same width comparison over every assistant-turn span) — is satisfied
    by construction on this path and is deliberately not
    re-derived here: :func:`load_config` FORCES ``steer_positions =``
    :data:`SERVED_POSITIONS`, so every target resolves to a response-masked
    spec. Reading it back off the object instead (``positions == "response"``)
    is what made a set_fit-only object, which never grows the deployment
    attributes, look like an all-position serve.

    ``serve_experiment`` is the ONE way off that pin: a cell whose resolved
    spec NAMES a transfer experiment (``serve_declaration.SERVE_EXPERIMENTS``,
    hashed into its identity and spelled ``<cell>@all_turns`` /
    ``<cell>@all_positions``) is served at the positions its name describes
    (``serve_declaration.EXPERIMENT_FIT_SERVE_PAIRS``), and the lens leg's
    width comparison runs only where that mask is header-located
    (:data:`HEADER_MASKED_SERVES`). So under
    ``fit_response_serve_all_turns`` the verdicts are exactly the pinned
    path's — the same comparison over more assistant-turn spans — while under
    ``fit_response_serve_all_positions`` there is no header mask to compare
    against, the leg never fires, and a recorded header PASSES with
    :data:`HEADER_GUARD_NOT_COMPARED_ALL_POSITIONS` as its stated reason
    (a stale record is not a refusal there; the fit/serve REGIME guard,
    ``lens_adapters._check_fit_serve_positions``, is what admits the
    deployment, by name, and refuses a name the deployment does not match —
    that conjunct is not predicted here). A name outside the closed
    vocabulary is a ``ValueError``: the guard refuses it as a malformed
    declaration, so there is no verdict to predict. ``None`` (every tinycap
    target today) is the pinned path, unchanged.

    All of this is checked by EXECUTING the three guards on synthetic bytes
    (``tests/test_object_identity.py::
    test_header_guard_prediction_mirrors_the_real_guards``), never by grepping
    their source for a condition: the all-turns landing widened the lens leg's
    condition without changing a single verdict, and a source pin cannot tell
    that rewrite from a regression.
    """
    if serve_experiment is not None:
        if serve_experiment not in sd.SERVE_EXPERIMENTS:
            raise ValueError(
                f"serve_experiment={serve_experiment!r} is not a named transfer "
                f"experiment ({list(sd.SERVE_EXPERIMENTS)}); the fit/serve guard "
                "refuses it as a malformed declaration, so there is no header "
                "verdict to predict")
        served_positions = sd.EXPERIMENT_FIT_SERVE_PAIRS[serve_experiment][1]
    else:
        served_positions = SERVED_POSITIONS
    if recorded is None:
        return True, HEADER_GUARD_NOT_APPLICABLE
    if served_positions not in HEADER_MASKED_SERVES:
        # the named all-positions serve: no id-located mask, so the lens leg
        # has nothing to compare the record against and does not fire
        return True, HEADER_GUARD_NOT_COMPARED_ALL_POSITIONS
    if recorded == served_header_len:
        return True, HEADER_GUARD_CHECKED
    if allow_out_of_support:
        return True, HEADER_GUARD_OVERRIDDEN
    return False, HEADER_GUARD_MISMATCH


def build_meta_shell(model_id: str, revision: str) -> Tuple[Any, str, str]:
    """A weightless model shell at the pinned revision, on the meta device.

    Returns ``(shell, class name, how)``. The object loaders read only
    ``hidden_size`` and the decoder ModuleList off it, so no parameter is ever
    materialised; ``accelerate.init_empty_weights`` is the fallback for a class
    whose ``from_config`` does not tolerate a meta context."""
    import torch
    from transformers import AutoConfig

    from sisteer.models import auto_model_class

    hcfg = AutoConfig.from_pretrained(model_id, revision=revision)
    cls = auto_model_class(model_id, revision=revision, config=hcfg)
    try:
        with torch.device("meta"):
            shell = cls.from_config(hcfg)
        how = "cls.from_config(cfg) under torch.device('meta') — no weights allocated"
    except Exception as e:   # the fallback IS the point; the reason is recorded in `how`
        from accelerate import init_empty_weights
        with init_empty_weights():
            shell = cls.from_config(hcfg)
        how = f"accelerate.init_empty_weights (meta ctx failed: {type(e).__name__}: {e})"
    shell.eval()
    return shell, cls.__name__, how


def load_config(path: str | Path, overrides: Optional[Mapping[str, object]] = None
                ) -> Dict[str, object]:
    """The two-SI config with the launcher's overlay applied and the serving
    pins the HF engine assumes (``steer_positions``/``scale``) made explicit —
    the same shape ``arm_spec_for_cell`` is called with at serve time."""
    import yaml

    cfg = dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    cfg.update(dict(overrides or {}))
    # The published numbers were produced at these serving pins and the old
    # stores record them, so the derivation FORCES them rather than defaulting:
    # a config that later grew a different value would silently derive the hash
    # of a different intervention than the one being reproduced.
    cfg["steer_positions"] = SERVED_POSITIONS
    cfg["scale"] = SERVED_SCALE
    return cfg


def artifact_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bank_artifact_path(objects_dir: str | Path, cell_id: str) -> Path:
    """The artifact a cell's object lives at in a bank: the standup's OWN
    resolver (``defense_cell.object_path_for`` — directory keyed on the cell
    id, filename on the arm through ``methods.object_artifact_name``:
    ``object.safetensors`` for the SteeringVector-container arms such as
    antidote, ``object.pt`` for torch blobs; the legacy ``lora_r1_all`` dir
    honoured), never a typed ``object.pt`` (E40 H2: an antidote preflight
    reported its valid object missing). A baseline id has no artifact."""
    from sisteer.eval.defense_cell import object_path_for

    if not str(objects_dir):
        raise ValueError(f"cell {cell_id!r}: no objects_dir to resolve the artifact in")
    path = object_path_for(objects_dir, cell_id)
    if path is None:
        raise ValueError(f"cell {cell_id!r} is a baseline: it has no object artifact")
    return Path(path)


def derive_one(*, artifact: str | Path, cell_id: str, si_id: Optional[str], arm: str,
               cfg: Mapping[str, object], shell, tok, coeff: float = 1.0,
               served_only_class_b: bool = False) -> Dict[str, object]:
    """The HF post-load intervention hash of ONE object, plus everything a
    preflight wants to say about it: the resolved spec, the object's recorded
    ``response_header_len`` (the fit/serve guard's left-hand side), and the
    artifact sha256. The hash the lens ROUTE stamps is a separate derivation
    (:func:`route_intervention_hash`). ``si_id`` is None on a bare Class B
    cell (SI-agnostic; class ``B``). ``served_only_class_b`` (the launcher's ``identity_request``
    marks a ``defense_cells.SERVED_ONLY_CLASS_B_IDS`` target) FORCES the served-only branch;
    a bare id outside ``methods.ARMS`` takes it too (the grading host's CLI requests carry no marker)."""
    from pathlib import Path as _Path

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving.hf_engine import EngineError, HFEngine
    from sisteer.capeval.serving.resolved_spec import resolve_and_hash
    from sisteer.eval.cell_controls import arm_spec_for_cell
    from sisteer.methods import _hydrate_object

    cell = SimpleNamespace(id=cell_id, arm=arm, si_id=si_id,
                           class_=("B" if si_id is None else "A"),
                           object_path=str(artifact), stacked=False)
    obj = None
    if served_only_class_b or (si_id is None and arm not in methods_arms()):
        # a SERVED-ONLY Class B cell — an object another lane
        # stages under a bare id with NO trainer on this tree (defense_cells.SERVED_ONLY_CLASS_B_IDS: `arditi_allpos`),
        # which the explicit-cells registry admits by name but `arm_spec_for_cell` refuses ("not registered in
        # methods.ARMS"), so the --pf identity derivation refused every such cell. The spec is built from the OBJECT and
        # its identity sidecar instead (:func:`served_only_class_b_spec`) — guarded by the blob's stored kind and the
        # sidecar's presence + verdict, never by the cell's name
        obj = _hydrate_object(_Path(artifact), shell, tok)
        kwargs = served_only_class_b_spec(cell_id=cell_id, artifact=artifact, obj=obj, cfg=cfg)
    else:
        kwargs = arm_spec_for_cell(cell, dict(cfg))
    if kwargs is None:
        raise ValueError(f"cell {cell_id!r} resolves to no ablation spec (a baseline?) — "
                         "there is no object identity to derive")
    spec = cm.AblationSpec(**kwargs)
    # HFEngine._steering_arm, line for line, minus load_object's trailing
    # obj.to_(model.device): a meta shell has no device to move to, and the
    # move cannot enter the hash (artifact bytes + resolved-spec fields only).
    eng = HFEngine(str(cfg["model"]), revision=str(cfg["model_revision"]),
                   preloaded=(shell, tok))
    if obj is None:
        obj = _hydrate_object(_Path(spec.artifact), shell, tok)
    resolved, ihash = resolve_and_hash(spec, obj, coeff, EngineError,
                                       arm_response_mask=eng._arm_response_mask)
    return {
        "cell": cell_id,
        "intervention_hash": ihash,
        "derivation": DERIVATION,
        "resolved_spec": dict(resolved),
        "response_header_len": getattr(obj, "response_header_len", None),
        "positions": getattr(obj, "positions", None),
        "object_kind": getattr(obj, "kind", None),
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256(artifact),
        "ablation_spec_kwargs": dict(kwargs),
    }


def methods_arms() -> Mapping[str, object]:
    """``methods.ARMS`` — the trainer registry `arm_spec_for_cell` dispatches on (imported lazily: torch)."""
    from sisteer.methods import ARMS
    return ARMS


#: The object KINDS a served-only Class B cell may carry: the residual-slot
#: constant kinds the native engine serves (``native_residual.CONSTANT_KINDS``) that another lane stages under a
#: bare id with no trainer on this tree — today the Arditi all-positions direction (``vector_all``). The guard is the
#: blob's STORED kind (the same field `methods.load_object` dispatches on), never the cell's name.
SERVED_ONLY_CLASS_B_KINDS: Tuple[str, ...] = ("vector_all",)


def served_only_class_b_spec(*, cell_id: str, artifact: str | Path, obj: object,
                             cfg: Mapping[str, object]) -> Dict[str, object]:
    """The ``AblationSpec`` kwargs of a SERVED-ONLY Class B object — `arm_spec_for_cell`'s shape
    (``kind sisteer_object``, the artifact, the config's ``scale`` as the coeff), built from the OBJECT rather than a
    trainer registration: the hydrated blob's stored kind must be one of :data:`SERVED_ONLY_CLASS_B_KINDS`; the dir
    must declare the ``fit_provenance.json`` identity contract (``train_budget.object_identity_kind``) and pass its
    verdict (``fit_provenance_problems``: the fit-time record, the sidecar's sha == the bytes); the ``layer`` is the
    object's OWN single layer (a multi-layer object DECLARES its whole set as ``layers``, which F9 holds equal — the
    object applies at its own layers either way, so nothing is picked). Refuses by name otherwise."""
    from sisteer.capeval.serving.resolved_spec import object_layers
    from sisteer.eval.train_budget import FIT_PROVENANCE_NAME, fit_provenance_problems, object_identity_kind

    path = Path(artifact)
    kind = getattr(obj, "kind", None)
    if kind not in SERVED_ONLY_CLASS_B_KINDS:
        raise ValueError(
            f"cell {cell_id!r}: arm {cell_id!r} is not registered in methods.ARMS and the artifact {path} hydrates to "
            f"kind {kind!r}, not a served-only Class B kind {list(SERVED_ONLY_CLASS_B_KINDS)} — refusing to guess a spec")
    identity = object_identity_kind(path)
    if identity != "fit_provenance":
        raise ValueError(
            f"cell {cell_id!r}: a served-only Class B object is identified by its {FIT_PROVENANCE_NAME} sidecar beside the "
            f"artifact; {path.parent} declares {identity!r} — refusing (the bank staged the object without its identity record)")
    problems = fit_provenance_problems(path, cell_id)
    if problems:
        raise ValueError(f"cell {cell_id!r}: {FIT_PROVENANCE_NAME} does not certify {path}: " + "; ".join(problems))
    layers = object_layers(obj)
    if not layers:
        raise ValueError(f"cell {cell_id!r}: the {kind!r} object at {path} reports no layer — nothing to hold F9 to")
    spec: Dict[str, object] = {
        "name": cell_id,
        "kind": "sisteer_object",
        "artifact": str(path),
        "layer": int(layers[0]),
        "coeffs": [float(cfg.get("scale", 1.0))],
    }
    if len(layers) > 1:
        spec["layers"] = [int(x) for x in layers]
    return spec


def route_cell_source(cfg: Mapping[str, object], objects_dir: str | Path):
    """The ``TrainTestCellSource`` the standup enumerates ``objects_dir`` with
    — the registry-building tail of ``registry_adapter.traintest_cell_source``
    (minus the PetriSpec-side checks that need a manifest), with the SAME
    config dispatch the standup makes:

    * a config carrying ``defense_cell.EXPLICIT_CELLS_KEY`` (the lane's
      generated ``configs/tinycap/bcbw_*.yaml``, the cluster's
      ``sdfbase_full22_*``) builds the EXPLICIT registry — exactly the
      enumerated ids, every artifact validated at build time with its one
      identity sidecar (``registry_from_explicit_cells``) — over ``objects_dir``
      handed in as the ABSOLUTE ``defenses_dir`` (``resolve_defenses_dir`` rule
      1), so the two resolvers agree on the bank by construction;
    * any other config builds the two-SI CARTESIAN registry
      (``defense_cell.build_registry`` from ``si_ids`` / class arm lists,
      module defaults when absent) — byte-for-byte the historical path.

    Before this a ``constv2-*`` cell could never be resolved: the Cartesian
    registry knows only ``si_ids x class_a_arms``, so the preflight's route
    derivation refused every explicit-cells object cell as ABSENT while the
    standup (explicit) would have served it. The real-bytes test pins this
    against ``traintest_cell_source`` itself."""
    from sisteer.capeval.petri.registry_adapter import TrainTestCellSource
    from sisteer.eval import defense_cell as tt

    ddir = Path(objects_dir)
    if tt.EXPLICIT_CELLS_KEY in cfg:
        cfg_abs = {k: v for k, v in dict(cfg).items() if k != "defenses_dir"}
        cfg_abs["defenses_dir"] = str(ddir)
        registry = tt.registry_from_explicit_cells(cfg_abs, repo_root=None)
        return TrainTestCellSource(cells=tt.by_id(registry), defenses_dir=ddir)
    registry = tt.build_registry(
        ddir, si_ids=tuple(cfg.get("si_ids", tt.DEFAULT_SI_IDS)),
        class_a_arms=tuple(cfg.get("class_a_arms", tt.CLASS_A_ARMS)),
        class_b_arms=tuple(cfg.get("class_b_arms", tt.CLASS_B_ARMS)))
    return TrainTestCellSource(cells=tt.by_id(registry), defenses_dir=ddir)


def pin_match(artifact_sha256: object, pin: object) -> Optional[bool]:
    """Whether an artifact's sha256 equals the ROSTER PIN of the staged object
    it is served from (``configs/sdfbase_full22_pins.json`` ``object_sha256``)
    — the check that carries meaning for an object nobody has served on the HF
    engine yet (its derived HF hash is a RECORD until the bootstrap pins it).
    ``None`` when no pin was given (a two-SI cell), never a guessed match; a
    malformed pin (not 64 hex) is a mismatch."""
    if pin is None:
        return None
    p = str(pin).strip().lower()
    if len(p) != 64 or any(c not in "0123456789abcdef" for c in p):
        return False
    return str(artifact_sha256).strip().lower() == p


def route_intervention_hash(*, cfg: Mapping[str, object], objects_dir: str | Path,
                            cell_id: str, artifact: str | Path) -> Dict[str, object]:
    """The intervention hash the lens ROUTE will stamp for ``cell_id`` served
    from ``objects_dir`` (:data:`ROUTE_DERIVATION`): the harness id the standup
    translates the cell to, the registry adapter's own resolution of the
    artifact, and ``petri_intervention_hash`` over it — exactly what
    ``enumerate_cells`` files on ``PetriCell.intervention_hash`` and the proxy
    copies onto ``RouteEntry.intervention_hash``. Refuses when the registry has
    no such cell (the standup would enumerate it ABSENT), when its artifact is
    not the one the HF derivation hashed (two resolvers disagreeing about the
    bank), or when the adapter could not resolve the artifact."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri.registry_adapter import harness_cell_id

    harness_id = harness_cell_id(cell_id)
    entry = route_cell_source(cfg, objects_dir).get(harness_id)
    if entry is None:
        raise ValueError(
            f"cell {cell_id!r} ({harness_id!r}) is not in the TrainTest registry built over "
            f"{objects_dir} — the standup would enumerate it ABSENT")
    if not entry.object_path or Path(entry.object_path).resolve() != Path(artifact).resolve():
        raise ValueError(
            f"cell {cell_id!r}: the registry serves {entry.object_path!r} but the HF derivation "
            f"hashed {str(artifact)!r} — two resolvers disagree about the bank; refusing")
    if entry.resolved_spec is None:
        raise ValueError(
            f"cell {cell_id!r}: the registry adapter could not resolve {entry.object_path} "
            "(artifact or declared sidecar missing / unreadable) — ABSENT on the standup")
    artifact_bytes = Path(entry.object_path).read_bytes()
    sidecar_bytes = Path(entry.sidecar_path).read_bytes() if entry.sidecar_path else b""
    return {
        "harness_cell_id": harness_id,
        "lens_route_intervention_hash": cm.petri_intervention_hash(
            dict(entry.resolved_spec), artifact_bytes=artifact_bytes, sidecar_bytes=sidecar_bytes),
        "lens_route_derivation": ROUTE_DERIVATION,
    }


#: The one-user-turn probe both header renders are taken over (the trailing
#: assistant header does not depend on its content) — the same probe the
#: driver's ``measure_thinking_header_by_arm`` uses, so the two measurements
#: are of one thing.
HEADER_PROBE_MESSAGES = ({"role": "user", "content": "x"},)


def header_ids_under_arm(tok, arm: str) -> Tuple[int, ...]:
    """The generation-prompt assistant header ids this tokenizer appends under
    ONE thinking arm: ``add_generation_prompt`` True minus False, with that
    arm's ``chat_template_kwargs`` passed into the render.

    The SAME difference ``chat_markers.generation_header_ids`` takes, but with
    the arm NAMED — that helper takes no template kwargs and reads the
    tokenizer's pin, which is exactly why it cannot answer the ON question. The
    arm-to-kwargs mapping is ``contracts.chat_template_kwargs``'s, not a second
    copy, and the driver's ``_header_ids_under_arm`` is this function's twin on
    the serving side. Raises when the False render is not a token PREFIX of the
    True one (the header cannot then be read off the pair at all)."""
    from sisteer.capeval.tinycap import contracts as C

    kw = dict(C.chat_template_kwargs(arm)[C.CHAT_TEMPLATE_KWARGS_KEY])
    probe = [dict(m) for m in HEADER_PROBE_MESSAGES]

    def ids(agp: bool) -> Tuple[int, ...]:
        out = tok.apply_chat_template(probe, add_generation_prompt=agp,
                                      tokenize=True, **kw)
        # transformers 5.x answers tokenize=True with a BatchEncoding
        raw = out["input_ids"] if hasattr(out, "keys") else out
        return tuple(int(t) for t in raw)

    full, head = ids(True), ids(False)
    if full[:len(head)] != head:
        raise ValueError(
            f"the add_generation_prompt=False render is not a token prefix of the True "
            f"render under the {arm!r} arm — the appended assistant header cannot be read "
            "off the pair, so the served header width is unmeasurable here")
    return full[len(head):]


def derive_request(request: Mapping[str, object]) -> Dict[str, object]:
    """Every target of ONE model, sharing one tokenizer and one meta shell.

    The tokenizer is pinned thinking-OFF exactly as the standup pins it
    (``models.pin_reasoning_template_off``) — a reasoning template's UNPINNED
    header is a different header (Qwen: 5 tokens instead of 7), so an unpinned
    prediction would describe a world the server never serves.

    ``pins`` in the request (bcb-bfcl-200-widen plan S1-e; ac69a7a2) —
    ``{cell_id: object_sha256}`` from the lane's roster pins — or a per-target
    ``pin_sha256`` (the launcher's ``identity_request`` carries the roster
    entry's sha there): each result then records ``pin_sha256`` / ``pin_match``
    (artifact sha == pin) and a mismatch turns ``ok`` false; with no pin
    ``pin_match`` is null and nothing else changes.

    ``allow_out_of_support_thinking`` in the request (PW-oos, U10 item 5) says
    the launch NAMED ``--allow-out-of-support-thinking`` and will serve some
    tasks thinking-ON. Two things change and nothing else: BOTH arms' headers
    are measured and recorded (``generation_header_len_by_arm`` /
    ``generation_header_ids_by_arm``), and each object's guard is predicted
    against the arm that MOVES — the served ON header — so a mismatch is
    reported as :data:`HEADER_GUARD_OVERRIDDEN` instead of failing the
    preflight. With the key absent nothing is measured twice and every field is
    what it was."""
    from sisteer.chat_markers import generation_header_ids
    from sisteer.models import (
        load_tokenizer,
        pin_reasoning_template_off,
        serving_identity,
    )

    model_id = str(request["model"])
    revision = str(request["revision"])
    cfg = load_config(str(request["config"]), request.get("config_overrides"))
    cfg["model"], cfg["model_revision"] = model_id, revision
    objects_dir = str(request.get("objects_dir") or "")
    coeff = float(request.get("coeff", 1.0))
    pins = dict(request.get("pins") or {})

    tok = load_tokenizer(model_id, revision=revision)
    pin_reasoning_template_off(tok)
    header_len = len(generation_header_ids(tok))
    sid = serving_identity(tok, {"model": model_id, "model_revision": revision}, model=None)
    shell, clsname, how = build_meta_shell(model_id, revision)
    allow_oos = bool(request.get("allow_out_of_support_thinking"))

    out: Dict[str, object] = {
        "model": model_id, "revision": revision, "config": str(request["config"]),
        "config_overrides": dict(request.get("config_overrides") or {}),
        "chat_template_hash": sid["chat_template_hash"],
        "generation_header_len": header_len,
        "model_shell": {"class": clsname, "how": how},
        "derivation": DERIVATION,
        "ok": True,
        "results": [],
    }
    # PW-oos: under the override the run serves SOME tasks thinking-ON, so the
    # header the guard is predicted against is the ON one, and BOTH are
    # recorded — the widths and the ids are the finding the experiment exists
    # to state, so they belong in the preflight report whether or not the
    # measurement can be taken.
    guard_len = header_len
    if allow_oos:
        from sisteer.capeval.tinycap import contracts as C

        out["out_of_support_thinking"] = True
        out["out_of_support_thinking_note"] = C.OUT_OF_SUPPORT_THINKING_NOTE
        by_arm: Dict[str, object] = {}
        ids_by_arm: Dict[str, object] = {}
        for arm in (C.THINKING_OFF, C.THINKING_ON):
            try:
                arm_ids = header_ids_under_arm(tok, arm)
            except Exception as e:                     # noqa: BLE001 — recorded, never guessed
                by_arm[arm] = None
                ids_by_arm[arm] = f"{type(e).__name__}: {e}"
                continue
            by_arm[arm] = len(arm_ids)
            ids_by_arm[arm] = [int(i) for i in arm_ids]
        out["generation_header_len_by_arm"] = by_arm
        out["generation_header_ids_by_arm"] = ids_by_arm
        out["generation_header_moves"] = (
            None if None in by_arm.values()
            else ids_by_arm[C.THINKING_OFF] != ids_by_arm[C.THINKING_ON])
        if isinstance(by_arm.get(C.THINKING_ON), int):
            guard_len = int(by_arm[C.THINKING_ON])
    results: List[Dict[str, object]] = []
    for target in request.get("targets") or ():
        cell_id = str(target["cell"])
        rec: Dict[str, object] = {"cell": cell_id}
        try:
            artifact = target.get("artifact") or bank_artifact_path(objects_dir, cell_id)
            rec["artifact"] = str(artifact)
            si_id = target.get("si_id")           # None on a bare Class B cell (SI-agnostic)
            rec.update(derive_one(artifact=artifact, cell_id=cell_id,
                                  si_id=(str(si_id) if si_id is not None else None),
                                  arm=str(target["arm"]),
                                  cfg=cfg, shell=shell, tok=tok, coeff=coeff,
                                  # the launcher's mark rides only when the target carries it (every other
                                  # request — the grading host's CLI, the tests' doubles — is byte-identical to before)
                                  **({"served_only_class_b": True} if target.get("served_only_class_b") else {})))
            # the hash the lens ROUTE will stamp (E40 M4) — a separate derivation
            # through the registry adapter, reported so a served store's identity
            # is predictable from here and never mistaken for the HF one
            rec.update(route_intervention_hash(cfg=cfg, objects_dir=objects_dir,
                                               cell_id=cell_id, artifact=artifact))
            # the fit/serve guard the standup will run, PREDICTED here: the
            # object's recorded header length against the header the served
            # tokenizer derives under the same pinned render — and only when
            # the object records one, which is the real guards' own rule
            # (header_guard_prediction; O22)
            rec["generation_header_len"] = header_len
            if allow_oos:
                # PW-oos: the guard is predicted against the arm this run will
                # SERVE the ON tasks under, not the pinned-OFF derivation —
                # naming both so the comparison is readable, never implied
                rec["header_guard_compared_against"] = {
                    "arm": "on", "generation_header_len": guard_len,
                    "pinned_off_generation_header_len": header_len}
            (rec["header_guard_will_pass"],
             rec["header_guard"]) = header_guard_prediction(
                 rec.get("response_header_len"), guard_len,
                 allow_out_of_support=allow_oos)
            # the roster pin (bcb-bfcl-200-widen S1-e): the check that carries
            # meaning on a staged object — recorded beside the HF hash, never in
            # its place; a per-target pin wins over the request map
            pin = target.get("pin_sha256", pins.get(cell_id))
            rec["pin_sha256"] = pin
            rec["pin_match"] = pin_match(rec.get("artifact_sha256"), pin)
            expected = target.get("expected")
            rec["expected"] = expected
            if expected is None:
                # E40 H1: an object with NO published expectation cannot be
                # verified, so it REFUSES — the derived hash is still reported so
                # it can be pinned (--expect-hashes) and the wave re-run against
                # a stated identity. It used to pass as "unverified".
                rec["match"] = False
                rec["no_expectation"] = (
                    "no expected HF post-load intervention hash for this cell — neither the "
                    "contracts' inventory table nor --expect-hashes names one; the derived "
                    f"hash is {rec['intervention_hash']!r}")
            else:
                rec["match"] = rec["intervention_hash"] == expected
        except BaseException as e:   # one bad object must not hide the other nine
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["match"] = False
            rec["header_guard_will_pass"] = False
            rec["header_guard"] = HEADER_GUARD_NOT_EVALUATED
        if not rec.get("match") or not rec.get("header_guard_will_pass"):
            out["ok"] = False
        if rec.get("pin_match") is False:
            out["ok"] = False
        results.append(rec)
    out["results"] = results
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--request", required=True,
                    help="path to the request JSON, or '-' for stdin")
    ap.add_argument("--out", default="", help="write the report here as well as to stdout")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    text = sys.stdin.read() if a.request == "-" else Path(a.request).read_text(encoding="utf-8")
    report = derive_request(json.loads(text))
    blob = json.dumps(report, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(blob, encoding="utf-8")
    print(blob)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
