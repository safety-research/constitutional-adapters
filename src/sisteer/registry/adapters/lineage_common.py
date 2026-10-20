"""Shared helpers for the lineage adapters (r2 export, grades roots, Petri stats, BoN stores,
degeneracy unions, GLM descope).

What lives here and why:

* :class:`Seeds` — the dimension ids every run row needs (`model_id`, `eval_id`, `item_set_id`, the
  `instrument_id` / `judge_id` shas) read from the seed YAMLs (`configs/registry/*.yaml`).
  The seed files carry READABLE instrument / judge ids; the spec's primary keys are shas over the
  canonical fields, so this module derives them with :func:`identity.instrument_id` /
  :func:`identity.judge_id` over the NORMALISED row (the same row the dimensions adapter inserts).
  COUPLING POINT: `adapters/dimensions.py` must derive the same shas or these runs stay
  dependency-blocked; every run row also records `extra.instrument_seed_id` / `extra.judge_seed_id`
  so the binding can be re-pinned mechanically. Tests build a :class:`Seeds` from small synthetic dicts.
* :func:`producer_key_for_r2_row` — the ONE spelling of the producer key every B1 adapter mints a
  run id from for a row the r2 export keys by (view, cell, dose, pass); the degeneracy and BoN
  adapters attach to the r2 runs only because they use this function.
* :class:`ObjectResolver` — `(view, cell_id)` → the object's sha256 from the wave's VIEW_MANIFESTs
  (`$STAGE/sdfbase/full22_objects/<target>/cells_view/<view>/VIEW_MANIFEST.json`, the Petri arm
  view manifests) and `si:<sha>` for prompted cells from `si_texts.yaml`; an unresolvable object is a
  ROW refusal, never a placeholder (the spec forbids defaulting a provenance field and `object_id` is a
  foreign key).
* :func:`run_payload_base` — the run row's NOT NULL provenance fields with `field_confidence_json`
  entries for every derived / unverified value.
* :class:`RowRefusals` — the per-row refusal ledger (the Petri two-source guard, an unresolvable object,
  an undeclared instrument); rendered as `source_state` rows (json path `$.row_refusals`) so refusals
  are durable and queryable, and printed by the live smoke.
* Input helpers: :func:`checked_json` reads a JSON input through :func:`base.check_input`.

Nothing here reads a transcript, a prompt or a generation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import derive as DV
from .. import identity as I
from .. import site_forms as SF
from .. import vocab as V
from . import dose_grammar as DG
from .base import ROW_REFUSALS_PATH, AdapterRefusal, Input, SourceStateRow, check_input, inputs_fingerprint, row_refusals_uri, store_open, site_root

HOME = os.path.expanduser("~")
CAMPAIGN = "sdfbase_full22"
ROUND_ID = "r2"
RS_DEFAULT = site_root("$STATE/romp/candidate-si-refit/run-state/full22")   # the full22 run-state tree, as declared_common.RUN_STATE
R2_DATA_DEFAULT = os.path.join(HOME, "reviews", "sdf-base-steering-r2", "data")
CONSOLIDATED_DEFAULT = site_root("$STAGE/sdfbase/fullscale/consolidated")
OBJECTS_DEFAULT = site_root("$STAGE/sdfbase/full22_objects")
PETRI_MANIFESTS_DEFAULT = os.path.join(HOME, ".local", "state", "petri_arm_views")
SEEDS_DIR_ENV = "REGISTRY_SEEDS_DIR"

#: Deterministic ruling ids the B1 adapters reference (the rulings seed adopts them or re-points the rows).
DESCOPE_GLM_RULING_ID = "ruling:r2:descope:model:glm46vflash"


def supersede_instrument_ruling_id(instrument_seed_id: str) -> str:
    return f"ruling:r2:supersede:instrument:{instrument_seed_id}"


def expand(path: str) -> str:
    return os.path.expanduser(path) if path.startswith("~") else path


def json_safe(v):
    """A parked producer value as JSON: a non-finite float becomes the string `Infinity` / `-Infinity` / `NaN` (JSON has no
    spelling for them and the registry's `extra_json` must be valid JSON); everything else as is. `json_unsafe` reverses it."""
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return "NaN" if v != v else ("Infinity" if v > 0 else "-Infinity")
    return v


def json_unsafe(v):
    if isinstance(v, str) and v in ("Infinity", "-Infinity", "NaN"):
        return float(v.replace("Infinity", "inf").lower() if v != "NaN" else "nan")
    return v


def under(root: str, path: str) -> str:
    """A path a lane file names: absolute (or `~`) as given, else relative to the run-state root `root`."""
    p = expand(str(path))
    return p if os.path.isabs(p) else os.path.join(root, p)


def rel_home(path: str) -> str:
    p = os.path.abspath(expand(path))
    return "~" + p[len(HOME):] if p.startswith(HOME + os.sep) or p == HOME else p


def checked_json(path: str, role: str = "report", allow_symlink: bool = False,
                 expected_sha256: Optional[str] = None) -> Tuple[Input, Any]:
    """Hash + check the input, then parse it."""
    inp = check_input(expand(path), role=role, allow_symlink=allow_symlink, expected_sha256=expected_sha256)
    with store_open(inp.path, "r", encoding="utf-8") as fh:
        try:
            return inp, json.load(fh)
        except json.JSONDecodeError as exc:
            raise AdapterRefusal(path, f"not JSON: {exc.msg}") from exc


def confidence_entry(conf: str, source: str) -> Dict[str, str]:
    return DV.confidence(None, conf, source)


def unverified(reason: str) -> Dict[str, str]:
    return {"confidence": "unverified", "reason": reason}


# --------------------------------------------------------------------------------------------
# cell grammar
# --------------------------------------------------------------------------------------------

#: `<view>` → (base target key, dose, view key inside the object bank)
_VIEW_RE = re.compile(r"^(?P<base>[a-z0-9]+(?:_[0-9]+b)?)(?P<it>_it)?(?P<half>_halfdose)?$")

PROMPTED_SI = {"prompted__safety_refusal": "safety_refusal", "prompted__safety": "safety_refusal",
               "prompted__si_constitution": "si_constitution", "prompted__constitution": "si_constitution",
               "prompted__si_constitution_full": "si_constitution_full"}


def split_view_legacy(view: str) -> Tuple[str, float, str]:
    """`gemma4e4b_it_halfdose` → (`gemma4e4b`, 0.5, `it_halfdose`); `llama8b_shortconst` → (`llama8b`, 1.0, `shortconst`);
    `qwen35_4b_ar_c0p15` → (`qwen35_4b`, the NOMINAL c / c_served, `ar_c0p15`) — the all-positions rung, whose bank view key is the
    rung spelling and whose dose is the raw coefficient the tag spells (adapters/dose_grammar.py; a `_t<T>` replicate suffix is dropped)."""
    spec = DG.parse_arditi_allpos_tag(view)
    if spec is not None:
        return (spec.model_key or view), spec.nominal, spec.rung
    if view.endswith("_shortconst") or view.endswith("_fullconst"):
        base, suffix = view.rsplit("_", 1)
        return base, 1.0, suffix
    half = view.endswith("_halfdose")
    v = view[:-len("_halfdose")] if half else view
    it = v.endswith("_it")
    base = v[:-3] if it else v
    key = ("it" if it else "main") if not half else ("it_halfdose" if it else "halfdose")
    return base, (0.5 if half else 1.0), key


def cell_grammar_legacy(cell_id: str) -> Dict[str, Optional[str]]:
    """Object kind / paper label / role / si_id from the campaign's cell grammar (legacy_spellings.yaml object_kinds)."""
    if cell_id == "none" or cell_id.startswith("none__"):
        return {"object_kind": "none", "paper_label": "none", "object_role": "baseline_none", "si_id": None, "derivation": "none"}
    if cell_id in PROMPTED_SI:
        si = PROMPTED_SI[cell_id]
        label = {"safety_refusal": "safety_si", "si_constitution": "constitution_si", "si_constitution_full": "constitution_si_full"}[si]
        return {"object_kind": "system_instruction", "paper_label": label, "object_role": "prompt", "si_id": si, "derivation": "none"}
    if DG.is_arditi_allpos_key(cell_id):
        # the ALL-POSITIONS refusal steer (adapters/arditi_allpos.py; declared_common.ARMS["arditi_allpos"]): a `vector_all` object under the
        # paper label `arditi_allpos`, on the raw_coefficient dose axis — never the prefill `arditi` arm below, which it was folded into
        # until 2026-09-20 (the r2 export's multi-turn rows of this cell resolved to the prefill alpha = 0.05 object under the label `arditi`)
        return {"object_kind": DG.OBJECT_KIND, "paper_label": DG.PAPER_LABEL, "object_role": DG.OBJECT_ROLE, "si_id": None, "derivation": "none"}
    if cell_id == "arditi":
        return {"object_kind": "static_r_arditi", "paper_label": "arditi", "object_role": "safety", "si_id": None, "derivation": "none"}
    if cell_id == "alphasteer":
        return {"object_kind": "alphasteer", "paper_label": "alphasteer_rebake", "object_role": "safety", "si_id": None, "derivation": "none"}
    if "__" not in cell_id:
        return {"object_kind": None, "paper_label": None, "object_role": None, "si_id": None, "derivation": None}
    # a STAGED spelling (`base__<si>__<arm>`, `d0p5_eb64s1__<si>__<arm>`, `derived_itdoc64__<si>__<arm>`) names its fit variant before the
    # stem; the grammar reads the stem (derive.campaign_stem_legacy) — the prefix is fit provenance, never the arm
    stem, prefix = DV.campaign_stem_legacy(cell_id)
    if prefix == "stacked":
        # `stacked__<si>__<arm>` (defense_cell.py): a stack of the inner object, its own kind — never the inner cell's label
        return {"object_kind": "stacked", "paper_label": "other", "object_role": None, "si_id": None, "derivation": "stack"}
    si, arm = str(stem).split("__", 1)
    role = {"constv2-mix-full": "constitution", "constv2ctl2-full": "control", "safety_refusal": "safety", "si_irrelevant": "irrelevant"}.get(si)
    if arm.endswith("_minus_ctl_distilled"):
        # the residual-distilled student: its own label since 2026-09-19 (difference_relu is the undistilled paired write)
        return {"object_kind": "relu_steer_distilled_difference", "paper_label": "difference_relu_distilled", "object_role": "difference", "si_id": si,
                "derivation": "residual_distillation"}
    if arm.endswith("_minus_ctl") and arm.startswith("psr"):
        # the undistilled ReLU-steer paired write (kind `composed`, derivation `composition`); the difference_relu arm
        return {"object_kind": "composed", "paper_label": "difference_relu", "object_role": "difference", "si_id": si, "derivation": "composition"}
    if arm.endswith("_minus_ctl"):
        return {"object_kind": "lora_subtraction", "paper_label": "difference_lora", "object_role": "difference", "si_id": si,
                "derivation": "lora_subtraction"}
    if arm.startswith("psr"):
        label = {"constitution": "constitution_relu", "control": "control_relu", "safety": "safety_relu"}.get(role or "", "other")
        return {"object_kind": "relu_steer", "paper_label": label, "object_role": role, "si_id": si, "derivation": "none"}
    if arm.startswith("lora"):
        label = {"constitution": "constitution_lora", "control": "control_lora", "safety": "safety_lora"}.get(role or "", "other")
        return {"object_kind": "lora", "paper_label": label, "object_role": role, "si_id": si, "derivation": "none"}
    return {"object_kind": None, "paper_label": None, "object_role": role, "si_id": si, "derivation": None}


# --------------------------------------------------------------------------------------------
# producer keys
# --------------------------------------------------------------------------------------------


def producer_key_for_r2_row(view: str, cell_id: str, dose: float, pass_label: Optional[str] = None) -> str:
    """The r2 export's own row key as a producer EXECUTION key: `(view, cell_id, dose, pass)`."""
    d = f"{float(dose):g}" if dose is not None else ""
    return f"r2:{view}|{cell_id}|{d}|{pass_label or ''}"


_GRADES_ROOT_RE = re.compile(r"^grades_[A-Za-z0-9]+_(ch|bu|md|dv|rp)_[A-Za-z0-9_]+$")


def producer_key_for_grades_root(grades_root: str) -> str:
    """A grades root is its own execution key; the object discriminator is the aggregate's cell entry."""
    return grades_root.rstrip("/")


def petri_object_key(view: str, cell_id: str) -> str:
    """The Petri object key repeats across tags, so the tag joins it (identity.split_producer_key petri_run_key)."""
    return I.split_producer_key("petri_run_key", "x", object_key=cell_id, scope={"tag": view}).object_key


# --------------------------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------------------------

_INT_OR_NONE = ("n_shot", "token_cap_tokens", "reply_cap_tokens", "context_window_tokens", "n_trials", "bon_n", "max_turns", "decode_top_k")
_REAL_OR_NONE = ("decode_temperature", "decode_top_p", "decode_min_p", "decode_repetition_penalty", "decode_presence_penalty")
#: seed spellings → the vocabulary the row is inserted with (README "vocabulary the seeds need")
DECODE_SOURCE_SPELLINGS = {"vendor_preset_per_model": "vendor_recommendation", "model_card_thinking_card_per_model": "model_card",
                           "vendor_preset": "vendor_recommendation", "server_default": "vllm_default"}
READ_AT_SPELLINGS = {"final_rung": "ladder_final"}
EVAL_CATEGORY_SPELLINGS = {"static": "static_attack", "bon": "best_of_n", "em_organism": "misalignment", "multi-turn": "multi_turn",
                           "over-refusal": "over_refusal"}


def _none_if_unverified(v):
    return None if v in ("unverified", None, "") else v


def _int_or_none(v):
    v = _none_if_unverified(v)
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _real_or_none(v):
    v = _none_if_unverified(v)
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _split_legacy(spelling: str) -> List[str]:
    """`"a | b | c (note)"` → [a, b, c]; a parenthesised note is dropped from each alternative."""
    out = []
    for part in str(spelling).split("|"):
        p = part.strip()
        p = re.sub(r"\s*\(.*\)\s*$", "", p).strip()
        if p:
            out.append(p)
    return out


@dataclass
class Seeds:
    """The dimension seeds an adapter consults (`configs/registry/*.yaml`)."""
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    item_sets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    judges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    instruments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    si_texts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    legacy: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    source_dir: Optional[str] = None
    #: The `adapters/dimensions.py` loader, when this checkout has it, is the authority for the dimension rows the
    #: registry holds; instrument / judge ids then come from its rows (see `b0_index`). False → this module's derivation.
    use_b0_loader: bool = True
    _b0: Any = field(default=None, repr=False, compare=False)

    # ---- loading ------------------------------------------------------------------------------
    @classmethod
    def from_dir(cls, seeds_dir: str) -> "Seeds":
        d = expand(seeds_dir)
        need = ("models.yaml", "evals.yaml", "item_sets.yaml", "judges.yaml", "instruments.yaml", "si_texts.yaml", "legacy_spellings.yaml")
        for n in need:
            if not os.path.isfile(os.path.join(d, n)):
                raise AdapterRefusal(os.path.join(d, n), "seed file missing (PR-2 lands configs/registry/*.yaml; set REGISTRY_SEEDS_DIR)")
        s = cls(source_dir=d)
        s.models = {m["model_id"]: m for m in _load_yaml(os.path.join(d, "models.yaml"))["models"]}
        s.evals = {e["eval_id"]: e for e in _load_yaml(os.path.join(d, "evals.yaml"))["evals"]}
        s.item_sets = {i["item_set_id"]: i for i in _load_yaml(os.path.join(d, "item_sets.yaml"))["item_sets"]}
        s.judges = {j["judge_id"]: j for j in _load_yaml(os.path.join(d, "judges.yaml"))["judges"]}
        s.instruments = {i["instrument_id"]: i for i in _load_yaml(os.path.join(d, "instruments.yaml"))["instruments"]}
        s.si_texts = {t["si_id"]: t for t in _load_yaml(os.path.join(d, "si_texts.yaml"))["si_texts"]}
        legacy_spellings = _load_yaml(os.path.join(d, "legacy_spellings.yaml"))
        s.legacy = {k: v for k, v in legacy_spellings.items() if isinstance(v, list)}
        return s

    @classmethod
    def locate(cls, repo_root: Optional[str] = None) -> "Seeds":
        """`$REGISTRY_SEEDS_DIR`, else `<repo>/configs/registry/` when the seeds are present there."""
        env = os.environ.get(SEEDS_DIR_ENV)
        if env:
            return cls.from_dir(env)
        here = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        cand = os.path.join(here, "configs", "registry")
        if os.path.isfile(os.path.join(cand, "instruments.yaml")):
            return cls.from_dir(cand)
        raise AdapterRefusal(cand, f"no dimension seeds found (set {SEEDS_DIR_ENV} or land PR-2's configs/registry/*.yaml)")

    # ---- models -------------------------------------------------------------------------------
    def model_for_view_legacy(self, view: str) -> Dict[str, Any]:
        """The served `model_id`, the fit model and the transfer kind for a legacy view key."""
        base, dose, view_key = split_view_legacy(view)
        lookup = base if view_key in ("main", "halfdose") else (f"{base}_it" if view_key in ("it", "it_halfdose") else view)
        if view_key.startswith("ar_c"):
            lookup = f"{base}_{view_key}"        # the standup's spelling without a `_t<T>` replicate suffix (dose_grammar)
        for row in self.legacy.get("model_keys", []):
            for alt in _split_legacy(row.get("legacy", "")):
                if alt == lookup:
                    fit = row.get("fit_model_id")
                    tk = row.get("transfer_kind")
                    if isinstance(fit, str) and fit.startswith("context:"):
                        # the llama lane: the `main` views serve base-fit objects, the `it` views instruct-fit ones (legacy_spellings note)
                        fit = f"{base}_base" if view_key in ("main", "halfdose") else f"{base}_it"
                        tk = "base_to_instruct" if fit.endswith("_base") else "instruct_direct"
                    if fit is None and (self.models.get(row["model_id"]) or {}).get("checkpoint_kind") == "chat":
                        fit, tk = row["model_id"], "chat_checkpoint"   # one checkpoint: objects fit where they are served
                    if tk is None and fit is None:
                        tk = "prompted_only" if view_key in ("shortconst", "fullconst") else None
                    return {"model_id": row["model_id"], "fit_model_id": fit, "transfer_kind": tk, "base": base, "dose": dose,
                            "view_key": view_key, "confidence": "recorded", "source": "legacy_spellings.yaml model_keys"}
        # fall back to the models' own producer spellings (a placeholder spelling also by its expansion on this host, after every
        # spelling as written)
        for expanded in (False, True):
            for mid, m in self.models.items():
                sp = m.get("producer_spellings") or []
                if lookup in (SF.with_expansions(sp) if expanded else [str(x) for x in sp]):
                    return {"model_id": mid, "fit_model_id": None, "transfer_kind": None, "base": base, "dose": dose,
                            "view_key": view_key, "confidence": "recorded", "source": "models.yaml producer_spellings"}
        return {"model_id": None, "fit_model_id": None, "transfer_kind": None, "base": base, "dose": dose, "view_key": view_key,
                "confidence": "unverified", "source": f"legacy model spelling {view!r} not in legacy_spellings.yaml model_keys"}

    def model_thinking_channel(self, model_id: str) -> Optional[str]:
        m = self.models.get(model_id) or {}
        return m.get("thinking_channel") if m.get("thinking_channel") in V.THINKING_CHANNELS else None

    def model_for_hf_id(self, hf_id: str) -> Optional[str]:
        """The model an HF id (or a producer spelling) names; a spelling two rows carry resolves through the shared ownership rule
        (:func:`vocab.model_spelling_owner`: the paper's checkpoint over the worked example's re-pin of it), never by
        the file order; two rows of one rank name nothing. A producer spelling in site-placeholder form also names its expansion on
        this host, when no row spells that literal itself."""
        claimants = {mid: m.get("paper_status") for mid, m in self.models.items()
                     if m.get("hf_repo") == hf_id or hf_id in [str(x) for x in (m.get("producer_spellings") or [])]}
        if not claimants:
            claimants = {mid: m.get("paper_status") for mid, m in self.models.items()
                         if hf_id in SF.with_expansions(m.get("producer_spellings") or [])}
        return V.model_spelling_owner(claimants)

    # ---- evals / item sets -------------------------------------------------------------------
    def eval_for_spelling(self, spelling: str) -> Optional[str]:
        if spelling in self.evals:
            return spelling
        for eid, e in self.evals.items():
            if spelling in [str(x) for x in (e.get("producer_spellings") or [])]:
                return eid
        return None

    def item_set_for(self, eval_id: str, instrument_seed_id: Optional[str] = None) -> Optional[str]:
        if instrument_seed_id and instrument_seed_id in self.instruments:
            inst = self.instruments[instrument_seed_id]
            evs = list(inst.get("eval_ids") or [])
            sets = list(inst.get("item_set_ids") or [])
            if eval_id in evs and len(sets) == len(evs):
                return sets[evs.index(eval_id)]
            if len(sets) == 1:
                return sets[0]
        e = self.evals.get(eval_id) or {}
        return e.get("default_item_set_id")

    def eval_category(self, eval_id: str) -> Optional[str]:
        cat = (self.evals.get(eval_id) or {}).get("eval_category")
        return EVAL_CATEGORY_SPELLINGS.get(cat, cat)

    # ---- dimension-loader coupling -------------------------------------------------------------
    def b0_index(self) -> Optional[Dict[str, Any]]:
        """The dimension rows the `adapters/dimensions.py` loader builds from the same seed directory, indexed
        for the adapters: instruments by (seed instrument id, eval_id) — a vendor-preset instrument is EXPANDED PER MODEL
        there (decode fields filled from the decode policy, `extra.model_id`), so a lookup may need the model — and
        judges by seed id. None when that module is not on this checkout (the adapters then derive the ids over the
        normalised seed row with the same identity functions; the two derivations differ where that loader fills decode
        fields / expands per model — the documented coupling), or when `use_b0_loader` is off / the seeds are synthetic.
        A loader refusal is raised, not hidden: the registry would refuse the very same seeds."""
        if self._b0 is not None:
            return self._b0 or None
        if not self.use_b0_loader or not self.source_dir:
            self._b0 = False
            return None
        try:
            from . import dimensions as DIM
        except ImportError:
            self._b0 = False
            return None
        from .. import harvest as H
        try:
            bundle = DIM.build_records(DIM.load_seeds(self.source_dir), dict(H.harvester_producer(""), script="adapters/lineage_common.py Seeds.b0_index"),
                                       None, verify_si=False)
        except Exception as exc:   # a SeedRefusal names file / row / field; nothing else is expected
            raise AdapterRefusal(self.source_dir, f"WS-B0 dimension loader refused the seeds: {exc}") from exc
        inst: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        judges: Dict[str, Dict[str, Any]] = {}
        for r in bundle.all_records():
            p = r["payload"]
            ex = p.get("extra") or {}
            if r["record_type"] == "instrument":
                inst.setdefault((str(ex.get("seed_instrument_id")), str(p.get("eval_id"))), []).append(p)
            elif r["record_type"] == "judge":
                judges[str(ex.get("seed_judge_id"))] = p
        self._b0 = {"instruments": inst, "judges": judges, "revision_id": bundle.revision_id}
        return self._b0

    def model_id_for_view_legacy(self, view: Optional[str]) -> Optional[str]:
        """The served model's id for a producer's model/view spelling (None when it maps to no model seed)."""
        if not view:
            return None
        try:
            return self.model_for_view_legacy(view).get("model_id")
        except Exception:
            return None

    # ---- judges -------------------------------------------------------------------------------
    def judge_row(self, seed_id: str) -> Dict[str, Any]:
        b0 = self.b0_index()
        if b0 is not None:
            row = b0["judges"].get(seed_id)
            if row is None:
                raise KeyError(f"judge seed {seed_id!r} has no WS-B0 row")
            return dict(row)
        j = self.judges.get(seed_id)
        if j is None:
            raise KeyError(f"judge seed {seed_id!r} not in judges.yaml")
        backend = j.get("judge_backend")
        if isinstance(backend, list):
            backend = backend[0] if backend else "unverified"
        if backend not in V.JUDGE_BACKENDS:
            backend = "unverified"
        row = {
            "judge_model": j.get("judge_model"), "judge_role": j.get("judge_role", "judge"), "rubric_name": j.get("rubric_name"),
            "rubric_version": str(j.get("rubric_version")) if j.get("rubric_version") is not None else None,
            "rubric_sha256": j.get("rubric_sha256") or "unverified", "protocol_sha256": j.get("protocol_sha256") or "unverified",
            "score_scale": str(j.get("score_scale")) if j.get("score_scale") is not None else None,
            "threshold_rule": str(j.get("threshold_rule")) if j.get("threshold_rule") not in (None, "unverified") else None,
            "label_vocabulary_json": j.get("label_vocabulary") if j.get("label_vocabulary") is not None else [],
            "judge_backend": backend,
        }
        row["judge_id"] = I.judge_id(row)
        return row

    def judge_id(self, seed_id: str) -> str:
        return str(self.judge_row(seed_id)["judge_id"])

    def judge_for_string(self, text: Optional[str]) -> Optional[str]:
        """A producer's judge string → the seed judge id (legacy_spellings.yaml judge_strings; then judge_model + rubric)."""
        if not text:
            return None
        t = str(text)
        low = t.lower()
        best: Tuple[int, Optional[str]] = (0, None)
        for row in self.legacy.get("judge_strings", []):
            jid = row.get("judge_id")
            if not isinstance(jid, str) or jid not in self.judges:
                continue
            for raw in str(row.get("legacy", "")).split("|"):
                for alt in (raw.strip(), re.sub(r"\s*\(.*\)\s*$", "", raw.strip()).strip()):
                    if alt and alt.lower() in low and len(alt) > best[0]:
                        best = (len(alt), jid)
        if best[1]:
            return best[1]
        for jid, j in self.judges.items():
            model = str(j.get("judge_model", "")).lower()
            rub = str(j.get("rubric_name", "")).lower()
            ver = str(j.get("rubric_version", "")).lower()
            if model and model in low and rub and (rub in low or (rub + "_" + ver) in low):
                return jid
        return None

    # ---- instruments --------------------------------------------------------------------------
    def instrument_row(self, seed_id: str, eval_id: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """The instrument ROW for (seed entry, eval[, served model]); its sha is the `instrument_id`. With
        the `adapters/dimensions.py` loader on the checkout the row is the loader's (per model where the
        seed is expanded per model); otherwise the normalised seed row derived here (model-independent)."""
        b0 = self.b0_index()
        if b0 is not None:
            cands = b0["instruments"].get((seed_id, eval_id)) or []
            if not cands:
                raise KeyError(f"instrument seed {seed_id!r} has no WS-B0 row for eval {eval_id!r}")
            if len(cands) == 1:
                return dict(cands[0])
            # The dimensions loader writes one row per model whose decode policy differs (`extra.model_id`) and
            # ONE shared row for the models whose policies agree (`extra.model_id` null, `extra.model_ids` lists them)
            ex = lambda c: (c.get("extra") or {})
            by_model = [c for c in cands if ex(c).get("model_id") == model_id]
            if len(by_model) == 1:
                return dict(by_model[0])
            shared = [c for c in cands if ex(c).get("model_id") is None and model_id in (ex(c).get("model_ids") or [])]
            if len(shared) == 1:
                return dict(shared[0])
            raise KeyError(f"instrument seed {seed_id!r} / eval {eval_id!r} is expanded per model by WS-B0 ({len(cands)} rows); "
                           f"model {model_id!r} selects {len(by_model) + len(shared)}")
        inst = self.instruments.get(seed_id)
        if inst is None:
            raise KeyError(f"instrument seed {seed_id!r} not in instruments.yaml")
        if eval_id not in (inst.get("eval_ids") or []):
            raise KeyError(f"instrument seed {seed_id!r} does not declare eval {eval_id!r}")
        item_set = self.item_set_for(eval_id, seed_id)
        judge_seed = inst.get("judge_id")
        judge_sha = self.judge_id(judge_seed) if judge_seed and judge_seed in self.judges else None
        ladder = inst.get("token_cap_ladder_json")
        if ladder in (None, "unverified"):
            ladder = []
        row: Dict[str, Any] = {
            "eval_id": eval_id, "item_set_id": item_set,
            "harness": inst.get("harness"), "harness_version": str(inst.get("harness_version") or "unverified"),
            "harness_config_hash": str(inst.get("harness_config_hash") or "unverified"),
            "n_shot": _int_or_none(inst.get("n_shot")),
            "prompt_template_sha256": inst.get("prompt_template_sha256") or "unverified",
            "cot_prompting": _none_if_unverified(inst.get("cot_prompting")),
            "scoring_rule": inst.get("scoring_rule"),
            "answer_extraction": _none_if_unverified(inst.get("answer_extraction")),
            "token_cap_tokens": _int_or_none(inst.get("token_cap_tokens")),
            "token_cap_ladder_json": I.canonical_json(list(ladder)),
            "ladder_promotion_rule": inst.get("ladder_promotion_rule"),
            "read_at": READ_AT_SPELLINGS.get(inst.get("read_at"), _none_if_unverified(inst.get("read_at"))),
            "reply_cap_tokens": _int_or_none(inst.get("reply_cap_tokens")),
            "context_window_tokens": _int_or_none(inst.get("context_window_tokens")),
            "decode_source": DECODE_SOURCE_SPELLINGS.get(inst.get("decode_source"), inst.get("decode_source")),
            "decode_temperature": _real_or_none(inst.get("decode_temperature")),
            "decode_top_p": _real_or_none(inst.get("decode_top_p")),
            "decode_top_k": _int_or_none(inst.get("decode_top_k")),
            "decode_min_p": _real_or_none(inst.get("decode_min_p")),
            "decode_repetition_penalty": _real_or_none(inst.get("decode_repetition_penalty")),
            "decode_presence_penalty": _real_or_none(inst.get("decode_presence_penalty")),
            "decode_seed_rule": _none_if_unverified(inst.get("decode_seed_rule")),
            "thinking_mode": _none_if_unverified(inst.get("thinking_mode")),
            "n_trials": _int_or_none(inst.get("n_trials")),
            "bon_n": _int_or_none(inst.get("bon_n")),
            "max_turns": _int_or_none(inst.get("max_turns")),
            "serving_path": inst.get("serving_path") or "unverified",
            "applied_positions_mode": inst.get("applied_positions_mode") or "unverified",
            "attacker_model_id": inst.get("attacker_model_id"),
            "attacker_decode_json": I.canonical_json(inst["attacker_decode_json"]) if inst.get("attacker_decode_json") is not None else None,
            "judge_id": judge_sha,
            "prefill_scheme": inst.get("prefill_scheme"),
            "notes": inst.get("notes"),
        }
        row["instrument_id"] = I.instrument_id(row)
        return row

    def instrument_id(self, seed_id: str, eval_id: str, model_id: Optional[str] = None) -> str:
        return str(self.instrument_row(seed_id, eval_id, model_id)["instrument_id"])

    def instrument_seed_ids(self) -> List[str]:
        return list(self.instruments)

    # ---- system instructions ------------------------------------------------------------------
    def si_object_id(self, si_id: str) -> Optional[str]:
        t = self.si_texts.get(si_id)
        if not t or not t.get("si_text_sha256") or t["si_text_sha256"] == "unverified":
            return None
        return I.object_id("system_instruction", si_text_sha256=str(t["si_text_sha256"]))


# --------------------------------------------------------------------------------------------
# objects
# --------------------------------------------------------------------------------------------


@dataclass
class ResolvedObject:
    object_id: Optional[str]
    object_kind: Optional[str]
    paper_label: Optional[str]
    object_role: Optional[str]
    confidence: str
    source: str
    manifest_uri: Optional[str] = None
    staged_id: Optional[str] = None
    derivation: Optional[str] = None
    si_id: Optional[str] = None


class ObjectResolver:
    """(view, cell_id) → the object's identity. Reads only named VIEW_MANIFEST files (bounded)."""

    def __init__(self, seeds: Seeds, objects_root: str = OBJECTS_DEFAULT, petri_manifests_root: str = PETRI_MANIFESTS_DEFAULT,
                 extra_manifests: Optional[Sequence[str]] = None):
        self.seeds = seeds
        self.objects_root = expand(objects_root)
        self.petri_manifests_root = expand(petri_manifests_root)
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._extra = list(extra_manifests or [])
        self.inputs: Dict[str, Input] = {}

    def _manifest(self, path: str) -> Optional[Dict[str, Any]]:
        if path in self._cache:
            return self._cache[path]
        if not os.path.isfile(path):
            self._cache[path] = None
            return None
        inp, data = checked_json(path, role="report")
        self.inputs[path] = inp
        self._cache[path] = data
        return data

    def view_manifest_path_legacy(self, view: str) -> str:
        base, dose, key = split_view_legacy(view)
        return os.path.join(self.objects_root, base, "cells_view", key, "VIEW_MANIFEST.json")

    def arm_manifest_path_legacy(self, view: str, arm: str) -> str:
        base, _, _ = split_view_legacy(view)
        return os.path.join(self.petri_manifests_root, base, arm, "VIEW_MANIFEST.json")

    def resolve(self, view: str, cell_id: str, served_sha256: Optional[str] = None) -> ResolvedObject:
        g = cell_grammar_legacy(cell_id)
        kind = g["object_kind"]
        if kind == "none":
            return ResolvedObject("none", "none", "none", "baseline_none", "derived", "cell grammar: unsteered", derivation="none")
        if kind == "system_instruction":
            oid = self.seeds.si_object_id(g["si_id"] or "")
            if oid is None:
                return ResolvedObject(None, kind, g["paper_label"], "prompt", "unverified",
                                      f"si_texts.yaml carries no sha256 for {g['si_id']!r}", si_id=g["si_id"])
            return ResolvedObject(oid, kind, g["paper_label"], "prompt", "recorded", "si_texts.yaml si_text_sha256", si_id=g["si_id"],
                                  derivation="none")
        if served_sha256 and re.fullmatch(r"[0-9a-f]{64}", str(served_sha256)):
            return ResolvedObject(str(served_sha256), kind, g["paper_label"], g["object_role"], "recorded",
                                  "the row's served object sha256 (petri_served_object_sha256)", derivation=g["derivation"], si_id=g["si_id"])
        if kind in ("static_r_arditi", "alphasteer", DG.OBJECT_KIND):
            manifest_key_legacy = {"static_r_arditi": "arditi", "alphasteer": "alphasteer", DG.OBJECT_KIND: DG.OBJECT_KEY_LEGACY}[kind]
            p = self.arm_manifest_path_legacy(view, manifest_key_legacy)
            vm = self._manifest(p)
            if vm and vm.get("entries"):
                e = vm["entries"][0]
                sha = e.get("object_sha256")
                if sha:
                    return ResolvedObject(str(sha), kind, g["paper_label"], g["object_role"], "recorded",
                                          "Petri arm view VIEW_MANIFEST.json entries[0].object_sha256", manifest_uri=rel_home(p),
                                          staged_id=e.get("staged_id"), derivation=g["derivation"])
            if kind != DG.OBJECT_KIND:
                return ResolvedObject(None, kind, g["paper_label"], g["object_role"], "unverified",
                                      f"no arm view manifest names the served object for ({view}, {cell_id}); the multi-turn arm rows' "
                                      f"operating-point object sha is not on the row", derivation=g["derivation"])
            # the all-positions arm: no Petri arm view on this machine yet, so the bank VIEW manifest of the rung is tried below
            # (`cells_view/ar_c<tok>/VIEW_MANIFEST.json`, whose one entry is the staged `ar_c<tok>__arditi_allpos`)
        if kind is None:
            return ResolvedObject(None, None, None, None, "unverified", f"cell id {cell_id!r} matches no known grammar")
        p = self.view_manifest_path_legacy(view)
        vm = self._manifest(p)
        # the bank view of the row's view first, then every EXTRA manifest the adapter declared (a pinned path each: petri_stats'
        # all-positions ladder manifest, registry follow-ups-4 item 12); the entry's provenance names the manifest it came from
        for src, m in ([(p, vm)] if vm else []) + [(x, self._manifest(x)) for x in self._extra]:
            for e in (m or {}).get("entries") or []:
                if e.get("cell_id") == cell_id and e.get("object_sha256"):
                    return ResolvedObject(str(e["object_sha256"]), kind, g["paper_label"], g["object_role"], "recorded",
                                          "VIEW_MANIFEST.json entries[].object_sha256", manifest_uri=rel_home(src), staged_id=e.get("staged_id"),
                                          derivation=g["derivation"], si_id=g["si_id"])
        extra = (" nor in the declared extra manifest(s) " + ", ".join(rel_home(x) for x in self._extra)) if self._extra else ""
        return ResolvedObject(None, kind, g["paper_label"], g["object_role"], "unverified",
                              f"no VIEW_MANIFEST entry for ({view}, {cell_id}) under {rel_home(p)}{extra}", derivation=g["derivation"], si_id=g["si_id"])


# --------------------------------------------------------------------------------------------
# row refusals (per-row, durable through source_state)
# --------------------------------------------------------------------------------------------


#: THE LEDGER RULE (adapter re-anchoring plan commit 7). A reader's `$.row_refusals` ledger belongs to ONE
#: read: the group it was filed for, under the pass stamp (`AdapterContext.now`) that read ran at. Its `source_state` row is keyed on the GROUP
#: (`base.row_refusals_uri(source_id)`, `input_sha256` the inputs' fingerprint), never on one of the group's files, so the key never moves when
#: a file leaves or discovery re-orders; it is written on every read of the group, the empty list included, so a cleared refusal is seen to
#: clear. A re-activation of a committed revision replays THAT revision's own state rows (`runner.committed_state_records`) and never asks the
#: adapter; a ledger asked for its group under another pass stamp refuses by name rather than serving a later read's list as the state of an
#: earlier revision.
ROW_REFUSALS_LEDGER_RULE = ("the $.row_refusals ledger is per read: keyed on the group (registry://row-refusals/<source_id>), written on every read "
                            "of the group, the empty list included; a re-activation replays the committed revision's own state rows and never "
                            "asks the adapter; asked for its group under another pass stamp, the ledger refuses by name")


@dataclass
class RowRefusals:
    adapter: str
    items: List[Dict[str, str]] = field(default_factory=list)
    #: the read this ledger belongs to -- ``(group, pass stamp)``, bound by the adapter's ``read`` -- or None for a ledger no read owns
    read_of: Optional[Tuple[str, str]] = None

    def add(self, key: str, why: str, input_uri: str = "") -> None:
        self.items.append({"key": key, "why": why, "input": input_uri})

    def __len__(self) -> int:
        return len(self.items)

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for it in self.items:
            reason = it["why"].split(":")[0][:80]
            out[reason] = out.get(reason, 0) + 1
        return out

    def members(self) -> List[str]:
        return sorted(f"{it['key']} -- {it['why']}"[:400] for it in self.items)

    def source_state_rows(self, source_id: str, input_uri: str, input_sha256: Optional[str]) -> List[SourceStateRow]:
        return [SourceStateRow(source_id, input_uri, ROW_REFUSALS_PATH, 1, 1, self.members(), input_sha256)]

    def ledger_rows(self, ctx, group: str, inputs: Sequence[Input], source_id: str) -> List[SourceStateRow]:
        """The ``$.row_refusals`` state row of ``group`` under :data:`ROW_REFUSALS_LEDGER_RULE`: the row -- keyed on the group, the inputs'
        fingerprint as its sha, the list as it stands (empty included) -- when this ledger is the read of ``group`` under ``ctx``'s pass stamp;
        NO row for a group this ledger was not filed for (the caller's content rows stand on their own: a ledger says nothing about a group it
        did not read); and a REFUSAL by name for its own group asked under another pass stamp -- the only caller that could ask is one
        collecting state rows for a revision this read did not produce (a re-activation), and the committed revision's own rows are what
        that revision's state is."""
        if self.read_of is None or self.read_of[0] != group:
            return []
        if self.read_of[1] != getattr(ctx, "now", None):
            raise AdapterRefusal(source_id, f"{self.adapter}: the row_refusals ledger of {group!r} was asked for under pass stamp {getattr(ctx, 'now', None)!r} "
                                            f"but belongs to the read at {self.read_of[1]!r}; a ledger is per read -- a re-activation replays the committed "
                                            f"revision's own state rows and never asks the adapter ({ROW_REFUSALS_LEDGER_RULE})")
        return [SourceStateRow(source_id, row_refusals_uri(source_id), ROW_REFUSALS_PATH, 1, 1, self.members(), inputs_fingerprint(inputs))]


# --------------------------------------------------------------------------------------------
# run payload
# --------------------------------------------------------------------------------------------


def substrate_from_grades_root(root: Optional[str], fallback_path: str) -> Tuple[str, str, Dict[str, str]]:
    """`_bu_` / `_ch_` in a grades root name the cluster; otherwise the report path's substrate (derived)."""
    if root:
        if "_bu_" in root:
            return "cluster_a", "derived", confidence_entry("derived", "grades root name carries the cluster tag _bu_")
        if "_ch_" in root:
            return "cluster_b", "derived", confidence_entry("derived", "grades root name carries the cluster tag _ch_")
    sub, conf, why = DV.substrate_from_path(expand(fallback_path))
    if sub == "unverified":
        return "grading_host", "derived", unverified(why)
    return str(sub), str(conf), confidence_entry(str(conf), f"{why}; the report is a grading host mirror, the generating substrate is not on the row")


def run_payload_base(seeds: Seeds, *, run_id: str, producer_key: str, object_key: str, view_legacy: str, cell_id_legacy: str,
                     obj: ResolvedObject, eval_id: str, instrument_row: Mapping[str, Any], instrument_seed_id: str,
                     lifecycle_state: str, when_utc: str, when_source: str, token_cap_tokens: Optional[int],
                     judge_id: Optional[str], judge_seed_id: Optional[str], substrate: Tuple[str, str, Dict[str, str]],
                     lane_source_legacy: str, store_uri: Optional[str], pass_label: Optional[str] = None,
                     instrument_match: str = "unverified", extra: Optional[Dict[str, Any]] = None,
                     notes: Optional[str] = None) -> Dict[str, Any]:
    """The NOT NULL provenance of a backfilled run row, every derived value named in field_confidence_json."""
    mv = seeds.model_for_view_legacy(view_legacy)
    fc: Dict[str, Any] = {}
    model_id = mv["model_id"]
    if model_id is None:
        raise KeyError(mv["source"])
    fc["model_id"] = confidence_entry(mv["confidence"], mv["source"])
    if obj.object_id is None:
        raise KeyError(obj.source)
    fc["object_id"] = confidence_entry(obj.confidence, obj.source)
    okind = obj.object_kind or "none"
    if okind == "none":
        tk = ("none", "derived", "object_kind=none")
    elif okind == "system_instruction":
        tk = ("prompted_only", "derived", "object_kind=system_instruction")
    elif mv.get("transfer_kind") in V.TRANSFER_KINDS:
        tk = (mv["transfer_kind"], "recorded", "legacy_spellings.yaml model_keys transfer_kind")
    else:
        tk = DV.transfer_kind(mv.get("fit_model_id"), model_id, okind, run_checkpoint_kind=(seeds.models.get(model_id) or {}).get("checkpoint_kind"))
        if tk[0] not in V.TRANSFER_KINDS:
            # `unverified` is a confidence, never a transfer kind: a row carrying it in the NOT NULL vocabulary column is refused by the
            # harvester and holds its whole revision (the cellstore alphasteer_f22 stores, 2026-09-18 to 09-30); refused HERE by name instead,
            # the way an unresolved model or object is (the callers turn a KeyError into a row refusal naming the row)
            raise KeyError(f"transfer_kind: {tk[2]} (model {model_id}, object {obj.object_id}); the model key names no transfer_kind in "
                           "legacy_spellings.yaml and the derivation resolves none")
    fc["transfer_kind"] = confidence_entry(tk[1], tk[2])
    serving_path = str(instrument_row.get("serving_path") or "unverified")
    if okind in ("none", "system_instruction"):
        serving_path = "passthrough"
        fc["serving_path"] = confidence_entry("derived", "unsteered / prompted cell: no object on the served path")
    elif serving_path == "unverified":
        fc["serving_path"] = unverified("the instrument declares no serving path and the row records none")
    else:
        fc["serving_path"] = confidence_entry("derived", f"declared by instrument {instrument_seed_id} (instrument_match {instrument_match})")
    apm = DV.applied_positions_mode(serving_path, okind)
    fc["applied_positions_mode"] = confidence_entry(apm[1], apm[2])
    cat = seeds.eval_category(eval_id) or ""
    mts = DV.multi_turn_steering_scope(str(apm[0]), cat, okind)
    fc["multi_turn_steering_scope"] = confidence_entry(mts[1], mts[2])
    thinking = instrument_row.get("thinking_mode") or "off"
    if seeds.model_thinking_channel(model_id) == "not_available":
        thinking = "not_available"
        fc["thinking_mode"] = confidence_entry("derived", "model.thinking_channel = not_available")
    else:
        fc["thinking_mode"] = confidence_entry("derived", f"declared by instrument {instrument_seed_id}")
    cap = token_cap_tokens if token_cap_tokens is not None else instrument_row.get("token_cap_tokens")
    if cap is None:
        raise KeyError(f"no token cap on the row and none declared by instrument {instrument_seed_id}")
    fc["token_cap_tokens"] = confidence_entry("recorded" if token_cap_tokens is not None else "derived",
                                              "the row's decode_cap_tokens" if token_cap_tokens is not None else f"instrument {instrument_seed_id}")
    fc["code_sha"] = unverified("the report carries a provenance string, not the producing checkout's sha (plan §2.5)")
    fc["prompt_template_sha256"] = unverified("not recorded on the report row")
    fc["decode_sent_json"] = unverified("the export's `decode` string is declared prose, never parsed (extra.decode_declared_legacy)")
    fc["node_hostname"] = unverified("not on the report row")
    fc["gpu_type"] = unverified("not on the report row")
    fc["queued_utc"] = confidence_entry("derived", when_source)
    fc["started_utc"] = confidence_entry("derived", when_source)
    fc["generated_utc"] = confidence_entry("derived", when_source)
    fc["instrument_id"] = confidence_entry("derived", f"seed instrument {instrument_seed_id} (sha over the normalised seed row; extra.instrument_seed_id)")
    fc["substrate"] = substrate[2]
    if instrument_match == "unverified":
        fc["instrument_match"] = unverified("the report row carries no sent decode / serving record to compare against the declared instrument")
    if (instrument_row.get("harness_version") or "unverified") == "unverified":
        fc["harness_version"] = unverified(f"instrument seed {instrument_seed_id} records no harness version")
    if (instrument_row.get("decode_source") or "unverified") == "unverified":
        fc["decode_source"] = unverified(f"instrument seed {instrument_seed_id} records no decode source")
    if judge_id is None:
        fc["judge_id"] = unverified("no judge string on the row maps to a seed judge")
    item_set = instrument_row.get("item_set_id")
    if item_set is None:
        raise KeyError(f"instrument {instrument_seed_id} names no item set for eval {eval_id}")
    payload: Dict[str, Any] = {
        "run_id": run_id, "campaign": CAMPAIGN, "producer_run_key_legacy": producer_key, "producer_object_key_legacy": object_key,
        "lifecycle_state": lifecycle_state, "queued_utc": when_utc,
        "started_utc": when_utc if lifecycle_state in ("running", "generated") else None,
        "generated_utc": when_utc if lifecycle_state == "generated" else None,
        "model_id": model_id, "object_id": obj.object_id, "transfer_kind": tk[0],
        "eval_id": eval_id, "item_set_id": item_set, "instrument_id": instrument_row["instrument_id"], "instrument_match": instrument_match,
        "harness": instrument_row.get("harness"), "harness_version": instrument_row.get("harness_version") or "unverified",
        "n_shot": instrument_row.get("n_shot"), "cot_prompting": instrument_row.get("cot_prompting"),
        "scoring_rule": instrument_row.get("scoring_rule"), "prompt_template_sha256": "unverified",
        "judge_id": judge_id, "pass_label": pass_label, "trial_index": 0,
        "decode_seed_rule": instrument_row.get("decode_seed_rule") or "none", "decode_policy_match": "no_policy",
        "decode_sent_json": {}, "decode_resolved_json": {},
        "decode_source": instrument_row.get("decode_source") or "unverified", "token_cap_tokens": int(cap),
        "token_cap_ladder_json": instrument_row.get("token_cap_ladder_json") or "[]",
        "reply_cap_tokens": instrument_row.get("reply_cap_tokens"), "context_window_tokens": instrument_row.get("context_window_tokens"),
        "thinking_mode": thinking, "serving_path": serving_path, "applied_positions_mode": apm[0], "multi_turn_steering_scope": mts[0],
        "history_resent_each_turn": 1 if cat in V.MULTI_TURN_EVAL_CATEGORIES else None,
        "code_sha": "unverified", "substrate": substrate[0], "substrate_source": substrate[1],
        "attacker_model_id": instrument_row.get("attacker_model_id") if (seeds.evals.get(eval_id) or {}).get("attacker_required") else None,
        "store_uri": store_uri, "view_legacy": view_legacy, "lane_source_legacy": lane_source_legacy, "campaign_cell_id_legacy": cell_id_legacy,
        "field_confidence_json": fc, "notes": notes,
        "extra": dict(extra or {}, instrument_seed_id=instrument_seed_id, judge_seed_id=judge_seed_id, object_resolution=obj.source,
                      object_manifest=obj.manifest_uri, object_staged_id=obj.staged_id),
    }
    if okind not in ("none", "system_instruction"):
        spec = DG.parse_arditi_allpos_tag(view_legacy)
        if spec is not None and DG.is_arditi_allpos_key(cell_id_legacy):
            # an all-positions rung: the dose is the RAW coefficient the tag spells, in raw_coefficient, with the nominal c / c_served
            # DERIVED beside it (dose_grammar) — never the filed 1.0 / 0.5 of the fitted-object views; the ONE strength stamp of the row
            payload.update(spec.strength_columns())
            fc.update(spec.strength_confidence("the standup spelling (view_legacy)"))
            payload["extra"].update(spec.extra())
        else:
            payload.update(strength_nominal=float(mv["dose"]), strength_nominal_source="filed")
            fc["strength_nominal"] = confidence_entry("recorded", "the export's dose column (1.0 fitted strength; 0.5 half-dose)")
            is_op = DV.fitted_is_operating_point(okind, mv["dose"])
            if is_op is not None:
                # a fitted LoRA / ReLU object, its difference or the paired write: the filed nominal is a campaign nominal with no unit declared
                # by the record (`nominal_only`), and 1.0 IS the operating point (the roster's strength_rule_from_dose)
                payload.update(is_operating_point=is_op, strength_unit="nominal_only")
                fc["is_operating_point"] = confidence_entry("derived", DV.FITTED_OPERATING_POINT_RULE)
                fc["strength_unit"] = confidence_entry("unverified", "the record declares no dose unit; the dose is the campaign nominal")
            else:
                # a published baseline (static_r_arditi / alphasteer, or vector_all on a view that spells no `ar_c` rung): its dose axis is alpha,
                # lambda or the raw coefficient, and its own reader (arditi_strength / alphasteer_rebake / arditi_allpos) rules the operating
                # point; the export's dose column is not it
                fc["is_operating_point"] = unverified(f"object kind {okind!r} is on its own dose axis; its own reader rules the operating point, the export's "
                                                      "dose column (1.0 / 0.5) is the served-key nominal, not that axis")
    return payload


def judgment_payload(run_id: str, judge_id: str, n_items: Optional[int], when_utc: str, store_uri: Optional[str],
                     api_pool: Optional[str] = None, extra: Optional[Dict[str, Any]] = None,
                     claimed_instrument_id: Optional[str] = None, claimed_required_judge_id: Optional[str] = None) -> Dict[str, Any]:
    """A complete judgment payload. ``claimed_instrument_id`` / ``claimed_required_judge_id``: the run's instrument and required
    judge as the adapter knows them from its own run payload, stated only when passed -- an adapter that does not know the instrument
    omits them (NULL reads as current on the paper side), and no existing caller's bytes move."""
    out: Dict[str, Any] = {"judgment_id": I.judgment_id(run_id, judge_id), "run_id": run_id, "judge_id": judge_id, "finished_utc": when_utc,
                           "n_items_judged": n_items, "status": "complete", "api_pool": api_pool, "store_uri": store_uri, "extra": dict(extra or {})}
    if claimed_instrument_id is not None:
        out["claimed_instrument_id"] = claimed_instrument_id
    if claimed_required_judge_id is not None:
        out["claimed_required_judge_id"] = claimed_required_judge_id
    return out


#: The columns of a ``run`` row that ARE the run's identity: the inputs ``identity.run_id`` is minted from (campaign, the producer's run
#: and object keys, eval, instrument), the dimension ids they resolve to (model, object, judge) and the producer's own spellings beside
#: them. Two readers of one store spell every one of these alike (they mint the same id), so a stats reader STATES them; every other
#: column of the row is the owner's.
RUN_IDENTITY_COLUMNS: Tuple[str, ...] = ("run_id", "campaign", "producer_run_key_legacy", "producer_object_key_legacy", "model_id", "object_id",
                                         "eval_id", "instrument_id", "judge_id", "view_legacy", "campaign_cell_id_legacy")
#: the marker's key (``reducer.UNSTATED_COLUMNS_KEY``) and the flag's carrier (``reducer.CONFIDENCE_COLUMN``), spelled here so the
#: adapters import no reducer
UNSTATED_COLUMNS_KEY = "unstated_columns"
CONFIDENCE_COLUMN = "field_confidence_json"


def stats_reader_run_record(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """A stats reader's ``run`` record (2026-10-07). The reader mints the export's run id and files per-run statistics
    (items, metrics, judgments, contrasts); the run ROW's provenance belongs to the row's owner (the registration owns a run's
    identity / provenance columns; the export owns the paper rows). Two full records restating one row per-column-latest
    flipped the live row's ``lane_source_legacy``, ``generated_utc`` / ``queued_utc`` / ``started_utc``, ``store_uri``, ``serving_path``,
    ``notes``, the counts and ``field_confidence_json`` on every re-read of either adapter -- a column moving without new information.

    The record STATES the identity columns (:data:`RUN_IDENTITY_COLUMNS`); every other column it carries with a value is marked
    ``extra.unstated_columns`` -- the fold applies it when no row exists yet, where the standing row has no value (the sidecar's
    ``serving_identity`` / ``n_items_planned`` fill a hole the export leaves), or where the standing value came from a marked statement
    (the reader's own earlier value, refreshed -- ``reducer.UNSTATED_REFRESH_TABLES``); never over an owner's value. A column it carries as
    NULL is carried and marked too -- a WITHDRAWAL: the fold clears a value a marked statement supplied (the reader's own earlier
    ``serving_identity`` / ``intervention_hash`` once the corrected sidecar no longer attests them) and moves nothing over an owner's value
    or a standing null. Before round 2 of the review (2026-10-07) the record OMITTED a null, so the reader could never retract what it had
    filed: a value the sidecar had withdrawn stood as current, with no refusal.
    ``field_confidence_json`` is the carrier of the fold's per-column ``unstated`` flag, never a marked column itself: the record keeps the
    entries of its marked columns only (its evidence for what it may apply), so its re-read moves no provenance of a column it does not
    state -- the owner's identity entries stand. ``extra`` merges key-wise in the fold as before. Returns a new dict; ``payload`` is not
    changed."""
    out: Dict[str, Any] = {}
    unstated: List[str] = []
    for k, v in payload.items():
        if k == "extra" or k in RUN_IDENTITY_COLUMNS:
            out[k] = v
            continue
        if k == CONFIDENCE_COLUMN:
            continue                      # the carrier is built below
        out[k] = v                        # a null included: marked, it withdraws the reader's own earlier value and nothing else (the fold)
        unstated.append(k)
    conf = payload.get(CONFIDENCE_COLUMN) or {}
    if isinstance(conf, str):
        try:
            conf = json.loads(conf)
        except ValueError:
            conf = {}
    out[CONFIDENCE_COLUMN] = {c: conf[c] for c in unstated if isinstance(conf, dict) and c in conf}
    extra = dict(out.get("extra") or {})
    extra[UNSTATED_COLUMNS_KEY] = sorted(unstated)
    out["extra"] = extra
    return out


def iso_ms(text: Optional[str], fallback: str) -> str:
    """A producer timestamp (ISO with offset / seconds / fractional) → the registry's UTC-with-ms shape; else fallback."""
    if not text:
        return fallback
    from datetime import datetime, timezone
    t = str(text).strip()
    try:
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    except ValueError:
        return fallback


def num_or_none(v):
    """NaN / pandas NA → None; numpy scalars → Python."""
    if v is None:
        return None
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except TypeError:
        pass
    if hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    return v


def int_or_none(v):
    v = num_or_none(v)
    if v is None or isinstance(v, bool):
        return None if v is None else int(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------------
# stand-in dimension records (tests and the live smoke ONLY — adapters/dimensions.py is the authority)
# --------------------------------------------------------------------------------------------

_STANDIN_NOTE = "STAND-IN row written by the WS-B1 smoke / tests from the seed YAMLs; WS-B0 adapters/dimensions.py owns the real row"


def standin_dimension_records(seeds: Seeds, producer: Mapping[str, Any], now: str, only_instruments: Optional[Iterable[str]] = None,
                              round_id: str = ROUND_ID) -> List[Dict[str, Any]]:
    """Records for paper_round, model, eval, item_set, judge, instrument, si_text and the `none` object, derived from the
    seeds with the SAME id derivations the adapters use. Rows whose seed values fall outside this checkout's vocabulary
    are still emitted (the harvester refuses them by name — a visible gap, never a silent default)."""
    from .. import records as R

    def rec(rt, payload):
        p = dict(payload)
        fc = dict(p.get("field_confidence_json") or {})
        for k, v in p.items():   # every 'unverified' value files its reason (the validator's rule for any table)
            if v == "unverified" and k not in fc:
                fc[k] = unverified("stand-in dimension row (tests / smoke): the seed carries no value")
        if fc:
            p["field_confidence_json"] = fc
        return R.make_record(rt, p, producer, I.adapter_record_id(p, rt), now)

    out: List[Dict[str, Any]] = [rec("paper_round", {"round_id": round_id, "opened_utc": now, "description": _STANDIN_NOTE})]
    for mid, m in seeds.models.items():
        gcp = m.get("generation_config_present")
        out.append(rec("model", {
            "model_id": mid, "family": m.get("family"), "size_label": str(m.get("size_label")), "checkpoint_kind": m.get("checkpoint_kind"),
            "hf_repo": m.get("hf_repo") or "unverified", "hf_revision": m.get("hf_revision") or "unverified",
            "mirror_of": m.get("mirror_of"), "chat_template_sha256": m.get("chat_template_sha256") or "unverified",
            "chat_template_note": m.get("chat_template_note"), "tokenizer_sha256": m.get("tokenizer_sha256") or "unverified",
            "generation_config_present": (int(gcp) if isinstance(gcp, (int, bool)) and not isinstance(gcp, str) else None),
            "generation_config_json": m.get("generation_config_json"), "thinking_channel": m.get("thinking_channel"),
            "thinking_header_moves": _int_or_none(m.get("thinking_header_moves")), "architecture_note": m.get("architecture_note"),
            "paper_status": m.get("paper_status"), "notes": _STANDIN_NOTE}))
    for eid, e in seeds.evals.items():
        out.append(rec("eval", {"eval_id": eid, "eval_category": EVAL_CATEGORY_SPELLINGS.get(e.get("eval_category"), e.get("eval_category")),
                                "headline_metric_id": e.get("headline_metric_id"), "metric_direction": e.get("metric_direction"),
                                "default_item_set_id": e.get("default_item_set_id"), "attacker_required": int(bool(e.get("attacker_required"))),
                                "producer_spellings_json": [str(x) for x in (e.get("producer_spellings") or [])],
                                "paper_status": e.get("paper_status"), "notes": _STANDIN_NOTE}))
    set_eval: Dict[str, str] = {}
    for inst in seeds.instruments.values():
        for ev, st in zip(inst.get("eval_ids") or [], inst.get("item_set_ids") or []):
            set_eval.setdefault(st, ev)
    for sid, s in seeds.item_sets.items():
        out.append(rec("item_set", {"item_set_id": sid, "eval_id": s.get("eval_id") or set_eval.get(sid), "dataset_version": str(s.get("dataset_version")),
                                    "n_items": _int_or_none(s.get("n_items")), "items_sha256": s.get("items_sha256") or "unverified",
                                    "source_uri": str(s.get("source_uri")), "item_id_scheme": str(s.get("item_id_scheme")),
                                    "subsample_rule": s.get("subsample_rule"), "pairing_note": s.get("pairing_note"),
                                    "held_out_of_fit": _int_or_none(s.get("held_out_of_fit"))}))
    for jid in seeds.judges:
        row = seeds.judge_row(jid)
        row["notes"] = _STANDIN_NOTE
        row["extra"] = {"seed_id": jid}
        out.append(rec("judge", row))
    for tid, t in seeds.si_texts.items():
        if t.get("si_text_sha256") and t["si_text_sha256"] != "unverified":
            out.append(rec("si_text", {"si_id": tid, "si_text_sha256": t["si_text_sha256"], "si_kind": t.get("si_kind"),
                                       "source_uri": str(t.get("source_uri")), "notes": _STANDIN_NOTE}))
    want = set(only_instruments) if only_instruments is not None else None
    for sid, inst in seeds.instruments.items():
        if want is not None and sid not in want:
            continue
        for ev in inst.get("eval_ids") or []:
            try:
                row = seeds.instrument_row(sid, ev)
            except KeyError:
                continue
            row["notes"] = _STANDIN_NOTE
            row["extra"] = {"seed_id": sid}
            out.append(rec("instrument", row))
    out.append(rec("object", {"object_id": "none", "object_kind": "none", "paper_label": "none", "campaign": CAMPAIGN, "object_role": "baseline_none",
                              "fit_target_kind": "none", "fit_thinking_render": "not_applicable", "fit_objective": "not_applicable",
                              "fit_positions": "not_applicable", "fit_seed_confidence": "not_applicable", "derivation": "none",
                              "artifact_sha256": "none", "status": "approved", "notes": _STANDIN_NOTE}))
    return out


def standin_object_record(obj: ResolvedObject, seeds: Seeds, producer: Mapping[str, Any], now: str, fit_model_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """A minimal `object` row for a resolved sha (smoke / tests only): kind, label, role and a paired-nullness-valid fit_target_kind."""
    from .. import records as R
    if obj.object_id in (None, "none") or obj.object_kind in (None, "none"):
        return None
    kind = obj.object_kind
    p: Dict[str, Any] = {"object_id": obj.object_id, "object_kind": kind, "paper_label": obj.paper_label or "other", "campaign": CAMPAIGN,
                         "object_role": obj.object_role or "other", "fit_thinking_render": "unverified", "fit_positions": "unverified",
                         "fit_seed_confidence": "unverified", "derivation": obj.derivation or "none", "status": "approved", "notes": _STANDIN_NOTE,
                         "field_confidence_json": {"fit_positions": unverified(_STANDIN_NOTE), "fit_thinking_render": unverified(_STANDIN_NOTE)},
                         "extra": {"staged_id": obj.staged_id, "manifest": obj.manifest_uri}}
    if kind == "system_instruction":
        sha = obj.object_id.split(":", 1)[1]
        p.update(fit_target_kind="declared_prompt", fit_objective="not_applicable", si_text_sha256=sha, artifact_sha256=sha)
    else:
        p["artifact_sha256"] = obj.object_id
    if kind == "system_instruction":
        pass
    elif kind in ("lora_subtraction", "relu_steer_distilled_difference"):
        p.update(fit_target_kind="derived", fit_objective="not_applicable", parent_object_ids_json=[], derivation_params_json={"note": _STANDIN_NOTE},
                 derivation="lora_subtraction" if kind == "lora_subtraction" else "residual_distillation")
    elif kind in ("static_r_arditi", "alphasteer", "diffmean", "caa_static_r", "antidote", "vector", "vector_all"):
        # `vector_all` is the all-positions Arditi direction (a contrastive direction found by the paper's search): without it here the
        # stand-in fell to the instruction-distillation branch below and minted a row the paired-nullness CHECK refuses
        p.update(fit_target_kind="contrastive_direction", fit_objective={"static_r_arditi": "arditi_search", "alphasteer": "alphasteer",
                                                                          "antidote": "antidote", "vector": "arditi_search",
                                                                          "vector_all": "arditi_search"}.get(kind, "diffmean"),
                 fit_contrast_sets_json={"note": _STANDIN_NOTE}, fit_model_id=fit_model_id, fit_layers_json=[], derivation="none")
    else:
        si = obj.si_id or ""
        if si in ("constv2-mix-full", "constv2ctl2-full"):
            p.update(fit_target_kind="corpus_document_fit", fit_objective="ce", fit_corpus_id=si, fit_corpus_sha256="unverified", fit_model_id=fit_model_id,
                     fit_layers_json=[], derivation="none")
        else:
            teacher = seeds.si_texts.get(si) or {}
            if not teacher.get("si_text_sha256"):
                return None
            p.update(fit_target_kind="system_instruction_distillation", fit_objective="kl" if "kl" in (obj.staged_id or "") else "ce",
                     fit_teacher_si_id=si, fit_teacher_si_text_sha256=teacher["si_text_sha256"], fit_prompt_pool_id="unverified",
                     fit_prompt_pool_sha256="unverified", fit_model_id=fit_model_id, fit_layers_json=[], derivation="none")
    if p.get("fit_model_id") is None and p["fit_target_kind"] not in ("derived", "declared_prompt"):
        return None
    for k, v in list(p.items()):
        if v == "unverified":
            p["field_confidence_json"][k] = unverified(_STANDIN_NOTE)
    return R.make_record("object", p, producer, I.adapter_record_id(p, "object"), now)
