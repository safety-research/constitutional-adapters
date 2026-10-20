"""Derivation rules for provenance fields, each cited in the glossary.

Every function returns ``(value, confidence, source_key)`` where ``confidence`` is one of
``vocab.CONFIDENCE`` and ``source_key`` names the evidence used (or the evidence that was missing
for ``unverified``) — the caller records it in ``field_confidence_json``. No function ever
defaults a provenance field silently: when the rule cannot decide it returns ``unverified`` with
the missing evidence named.
"""
from __future__ import annotations

import json
import re
from typing import Container, Dict, Mapping, Optional, Tuple, Union

from . import vocab as V

Derived = Tuple[object, str, str]


def confidence(value, confidence_: str, source_key: str) -> Dict[str, str]:
    """One ``field_confidence_json`` entry."""
    entry = {"confidence": confidence_, "source": source_key}
    if confidence_ == "unverified":
        entry["reason"] = source_key
    return entry


# --------------------------------------------------------------------------------------------
# transfer_kind
# --------------------------------------------------------------------------------------------

def transfer_kind(fit_model_id: Optional[str], run_model_id: str, object_kind: str,
                  run_checkpoint_kind: Optional[str] = None, fit_checkpoint_kind: Optional[str] = None) -> Derived:
    """``fit_model_id`` vs ``run.model_id``: same id → instruct_direct; fit on a base checkpoint served
    on an instruct one → base_to_instruct; a chat checkpoint (one model) → chat_checkpoint;
    system_instruction → prompted_only; none → none."""
    if object_kind == "none":
        return ("none", "derived", "object_kind=none")
    if object_kind == "system_instruction":
        return ("prompted_only", "derived", "object_kind=system_instruction")
    if not fit_model_id:
        return ("unverified", "unverified", "object.fit_model_id missing")
    if run_checkpoint_kind == "chat":
        return ("chat_checkpoint", "derived", "model.checkpoint_kind=chat")
    if fit_model_id == run_model_id:
        return ("instruct_direct", "derived", "fit_model_id == run.model_id")
    if (fit_checkpoint_kind == "base" or fit_model_id.endswith("_base")) and (
            run_checkpoint_kind == "instruct" or run_model_id.endswith("_it")):
        return ("base_to_instruct", "derived", "fit on base checkpoint, served on instruct checkpoint")
    return ("unverified", "unverified",
            f"fit_model_id={fit_model_id} vs run.model_id={run_model_id} matches no transfer rule")


# --------------------------------------------------------------------------------------------
# fit_positions from sidecar / blob keys
# --------------------------------------------------------------------------------------------

_POSITIONS_KEY_PRECEDENCE = (
    ("train_budget_meta.json", "positions"),
    ("fit_provenance.json", "positions_at_fit"),
    ("fit_meta.json", "positions"),
    ("blob", "positions"),
    ("blob", "steer_positions"),
    ("blob", "fit_positions"),
)

_POSITION_SPELLINGS = {
    "all": "all",
    "response": "response_incl_assistant_header",
    "response_incl_assistant_header": "response_incl_assistant_header",
    "last": "last_token_of_appended_assistant_header",
    "last_token": "last_token_of_appended_assistant_header",
    "completion_mean": "completion_mean",
}


def fit_positions(object_kind: str, sidecars: Mapping[str, Mapping[str, object]]) -> Derived:
    """``sidecars`` maps a sidecar name (``train_budget_meta.json`` … ``blob``) to its parsed keys."""
    if object_kind in ("none", "system_instruction"):
        return ("not_applicable", "derived", f"object_kind={object_kind}")
    if object_kind in ("lora_subtraction", "relu_steer_distilled_difference", "composed", "stacked"):
        return ("inherited_from_parents", "derived", f"object_kind={object_kind} (derived object)")
    if object_kind == "static_r_arditi":
        return ("searched_post_instruction_span", "derived", "object_kind=static_r_arditi")
    if object_kind in ("diffmean", "caa_static_r", "alphasteer"):
        return ("last_token_of_appended_assistant_header", "derived", f"object_kind={object_kind} capture rule")
    if object_kind == "antidote":
        return ("last_token_of_appended_assistant_header", "derived", "object_kind=antidote capture rule")
    for sidecar, key in _POSITIONS_KEY_PRECEDENCE:
        sc = sidecars.get(sidecar)
        if sc and key in sc and sc[key] is not None:
            raw = str(sc[key])
            mapped = _POSITION_SPELLINGS.get(raw)
            if mapped:
                return (mapped, "recorded", f"{sidecar}:{key}")
            return ("unverified", "unverified", f"{sidecar}:{key}={raw!r} is not a known positions spelling")
    return ("unverified", "unverified", "no sidecar or blob key names the fit positions "
            "(train_budget_meta.json:positions, fit_provenance.json:positions_at_fit, fit_meta.json:positions, blob positions)")


# --------------------------------------------------------------------------------------------
# applied_positions_mode (rule table, glossary §"applied positions")
# --------------------------------------------------------------------------------------------

_NATIVE_RESIDUAL_KINDS = ("relu_steer", "vector", "vector_all", "diffmean", "static_r_arditi", "caa_static_r",
                          "relu_steer_distilled_difference")


def applied_positions_mode(serving_path: str, object_kind: str, fit_positions_value: Optional[str] = None,
                           standup_mode: Optional[str] = None) -> Derived:
    if object_kind == "none" or serving_path == "passthrough":
        return ("none", "derived", "unsteered / passthrough")
    if object_kind == "system_instruction":
        return ("none", "derived", "prompted object steers no positions")
    if object_kind == "alphasteer":
        return ("prefill_last_broadcast", "derived", "object_kind=alphasteer")
    if object_kind == "antidote":
        return ("last", "derived", "object_kind=antidote")
    if standup_mode:
        # a standup record that names the served mode is the serving fact for EVERY object kind, the natively served LoRA kinds
        # included (a LoRA served on the non-template positions is not steering every token); a mode outside the vocabulary is
        # refused by name for every kind rather than falling through to a kind rule that would read `all`
        if standup_mode in V.APPLIED_POSITIONS_MODES:
            return (standup_mode, "recorded", "standup record served_positions / POSITION_MODES")
        return ("unverified", "unverified", f"standup mode {standup_mode!r} is not a known applied-positions mode")
    if serving_path == "lora_native" and object_kind in ("lora", "lora_subtraction"):
        return ("all", "derived", "natively served LoRA steers every token")
    if serving_path in ("lora_native", "hf", "lens") and object_kind in _NATIVE_RESIDUAL_KINDS + ("lora", "lora_subtraction"):
        if fit_positions_value == "all":
            return ("all", "derived", "fit_positions=all")
        if fit_positions_value == "response_incl_assistant_header":
            return ("response", "derived", "fit_positions=response_incl_assistant_header")
        return ("unverified", "unverified",
                f"no standup record names the served positions and fit_positions={fit_positions_value!r} maps to no mode")
    return ("unverified", "unverified", f"serving_path={serving_path} object_kind={object_kind}: no rule")


def multi_turn_steering_scope(applied_mode: str, eval_category: str, object_kind: str) -> Derived:
    if object_kind == "none":
        return ("not_applicable", "derived", "object_kind=none")
    if eval_category not in V.MULTI_TURN_EVAL_CATEGORIES:
        return ("single_turn", "derived", f"eval_category={eval_category} is single-turn")
    if object_kind == "system_instruction":
        # a prompted object steers no positions, so no conversation position is steered whatever the mode column carries (a job-start
        # registration copies the selected instrument's protocol mode onto the run); read after the category so a prompted run on a
        # single-turn eval keeps `single_turn`, as the mode table's `none` row gives it when the mode is derived from the kind
        return ("not_applicable", "derived", "object_kind=system_instruction: a prompted object steers no positions")
    table = {
        "all": "every_turn_prefill_and_decode",
        # the scaffold exclusion is a position-class fact the mode column carries; every turn and both phases are still steered
        "non_template": "every_turn_prefill_and_decode",
        "response": "final_header_and_current_decode_only",
        "response_all_turns": "every_assistant_span",
        "prompt_last": "prefill_only_each_turn",
        "prefill_all": "prefill_only_each_turn",
        "prefill_last_broadcast": "prefill_only_each_turn",
        "last": "prefill_only_each_turn",
        "none": "not_applicable",
    }
    if applied_mode in table:
        return (table[applied_mode], "derived", f"applied_positions_mode={applied_mode} on a multi-turn eval")
    return ("unverified", "unverified", f"applied_positions_mode={applied_mode!r} unresolved")


# --------------------------------------------------------------------------------------------
# substrate: a producer field that names it, a reader's rule, a path on a shared root, or a declared default
# --------------------------------------------------------------------------------------------

#: The shared-root PREFIX rule (anchored: the two clusters' roots share a leading path segment, so cluster A's root must be
#: tested before cluster B's per-user root; a substring test would read the first as the second). A path here is a per-run
#: MARKER (a routing table, a serving dir, a store, a usage ledger) and what it yields is ``derived`` -- a prefix is a
#: derivation, never a producer field naming the substrate.
_SUBSTRATE_ROOTS: Tuple[Tuple[str, str, str], ...] = (
    ("/workspace-vast/", "cluster_a", "/workspace-vast/ shared root"),
    ("runpod:", "runpod", "pod network volume path (launch.sh run dir)"),
    ("/workspace/runs/", "runpod", "pod network volume path (launch.sh run dir)"),
    ("twosi-fits:", "modal", "Modal volume path"),
    ("/vol/", "modal", "Modal volume path"),
    ("modal://", "modal", "Modal volume path"),
)
_CLUSTER_B_ROOT = re.compile(r"^/workspace/[^/]+/")
#: what a reader may pass as the substrate it knows: ``(field_name, value)`` for a producer field, ``(value, why)`` for a rule /
#: a default, or a bare value (a declared default, never a producer field)
SubstrateClaim = Union[Tuple[str, object], str, None]

#: the basis a bare default carries when its reader named none -- the label is honest (``assumed_default``), the reason says why it is thin
DEFAULT_SUBSTRATE_BASIS = "the reader's declared default for its source; no producer field, per-run marker or documented basis was named"


def substrate_root_of_path(path: Optional[str]) -> Optional[Tuple[str, str]]:
    """``(substrate, why)`` for a path on a known shared root (``derived`` evidence), ``("grading_host", ...)`` for a grading-host-local path,
    None when the path names no known root. The grading host case is separate because a grading host path is usually the MIRROR of a store."""
    p = str(path or "")
    if not p:
        return None
    for prefix, sub, why in _SUBSTRATE_ROOTS:
        if p.startswith(prefix):
            return sub, why
    if _CLUSTER_B_ROOT.match(p):
        return "cluster_b", "/workspace/<user> shared root"
    if p.startswith("/home/") or p.startswith("~"):
        return "grading_host", "grading-host-local path"
    return None


def _claim(x) -> Optional[Tuple[str, str]]:
    """Normalise a two-tuple claim to ``(a, b)`` of strings; anything else is no claim (a bare string is handled by the caller)."""
    if isinstance(x, (tuple, list)) and len(x) == 2 and x[0] is not None and x[1] is not None:
        return str(x[0]), str(x[1])
    return None


def substrate_from_path(path: Optional[str], recorded: SubstrateClaim = None, *, derived: SubstrateClaim = None,
                        default: SubstrateClaim = None) -> Derived:
    """The producing substrate of a store, by evidence strength -- ``(value, confidence, why)``:

    1. ``recorded=(field_name, value)``: a producer FIELD that literally names the substrate (``cluster=cluster_a`` in a drive log, a
       store's ``substrate`` field, a manifest's ``served_cluster``) -> ``recorded``. ONLY a named field earns ``recorded``: a bare
       string here is a reader's declared default and is filed as one (below), never as recorded -- the constant used to come back
       stamped ``recorded`` whether or not any record said so (497 Modal rows over cluster-generated stores, 2026-09-21).
    2. ``derived=(value, why)``: the reader's own documented rule over producer fields (a cluster tag in a store name, a node + config
       pair, a ladder symlink's realpath) -> ``derived``.
    3. ``path`` on a cluster's shared root, a Modal volume or a pod volume (:func:`substrate_root_of_path`) -> ``derived``. A default
       the path contradicts does not stand: the path wins and the ``why`` names the contradiction.
    4. ``default=(value, basis)`` (or a bare value): what the reader knows about its source as a whole -- its submit scripts, its
       store manifests checked once -- not what this record carries -> ``assumed_default`` with the basis. A grading host ``path`` is the
       store's MIRROR and does not outrank a default; a path on no known root does not either.
    5. a grading host path alone -> ``("grading_host", "derived", ...)``; nothing -> ``("unverified", "unverified", <what was missing>)``.

    ``run.substrate`` admits no ``unverified`` value: the emitter spells an unknown substrate ``grading_host`` with ``substrate_source =
    unverified`` and this function's ``why`` as the reason in ``field_confidence_json``."""
    notes = []
    rec = _claim(recorded)
    if isinstance(recorded, str) and recorded:
        # a bare value in the recorded slot is the reader's own knowledge of its source, not a producer field: file it as the default it is
        if default is None:
            default = (recorded, DEFAULT_SUBSTRATE_BASIS + " (passed as a bare value in the recorded slot)")
        notes.append(f"a bare {recorded!r} was passed as recorded and is filed as a declared default")
    if rec is not None:
        field, value = rec
        if value in V.SUBSTRATES:
            return (value, "recorded", f"producer field {field} names the substrate")
        notes.append(f"producer field {field} names {value!r}, not a substrate")
    der = _claim(derived)
    if der is not None:
        value, why = der
        if value in V.SUBSTRATES:
            return (value, "derived", why)
        notes.append(f"the reader's rule derived {value!r}, not a substrate ({why})")
    dflt = _claim(default)
    if dflt is None and isinstance(default, str) and default:
        dflt = (default, DEFAULT_SUBSTRATE_BASIS)
    if dflt is not None and dflt[0] not in V.SUBSTRATES:
        notes.append(f"the reader's declared default {dflt[0]!r} is not a substrate")
        dflt = None
    p = str(path or "")
    root = substrate_root_of_path(p)
    if root is not None and root[0] != "grading_host":
        value, why = root
        if dflt is not None:
            why += (" (agrees with the reader's declared default)" if dflt[0] == value
                    else f"; the reader's declared default {dflt[0]!r} is contradicted by the path and does not stand")
        return (value, "derived", why)
    if dflt is not None:
        value, basis = dflt
        if root is not None:
            basis += f"; the path {p[:80]!r} is the grading host mirror of the store, not where it was generated"
        elif p:
            basis += f"; the path {p[:80]!r} names no known substrate root"
        else:
            basis += "; no per-run path was named"
        return (value, "assumed_default", basis)
    if root is not None:
        return (root[0], "derived", root[1])
    missing = f"path {p!r} names no known substrate root" if p else "no producer field, reader rule, path or declared default names the substrate"
    if notes:
        missing += " (" + "; ".join(notes) + ")"
    return ("unverified", "unverified", missing)


#: a small JSON walk: the FIRST string under ``record`` (preferring the dotted key paths in ``prefer``) that sits on a known
#: non-grading-host shared root, returned with the key path that carried it -- the per-run witness a reader names in its ``why``
def first_shared_root_path(record: Mapping[str, object], prefer: Tuple[str, ...] = (), limit: int = 20000) -> Optional[Tuple[str, str]]:
    def get(o, dotted: str):
        cur = o
        for part in dotted.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                return None
            cur = cur[part]
        return cur

    def is_marker(v) -> bool:
        if not isinstance(v, str):
            return False
        r = substrate_root_of_path(v.strip('"'))
        return r is not None and r[0] != "grading_host"

    for key in prefer:
        v = get(record, key)
        if is_marker(v):
            return key, v.strip('"')
    seen = 0
    stack = [("", record)]
    while stack and seen < limit:
        prefix, node = stack.pop(0)
        seen += 1
        if isinstance(node, Mapping):
            for k, v in node.items():
                kp = f"{prefix}.{k}" if prefix else str(k)
                if is_marker(v):
                    return kp, v.strip('"')
                if isinstance(v, (Mapping, list, tuple)):
                    stack.append((kp, v))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                kp = f"{prefix}[{i}]"
                if is_marker(v):
                    return kp, v.strip('"')
                if isinstance(v, (Mapping, list, tuple)):
                    stack.append((kp, v))
    return None


def substrate_derived_from_record(record: Mapping[str, object], label: str, prefer: Tuple[str, ...] = ()) -> Optional[Tuple[str, str]]:
    """``(substrate, why)`` -- the ``derived=`` claim for :func:`substrate_from_path` -- from the first shared-root path a record
    carries (``label`` names the record for the ``why``; ``prefer`` the key paths to try first), else None."""
    hit = first_shared_root_path(record, prefer)
    if hit is None:
        return None
    key, path = hit
    root = substrate_root_of_path(path)
    if root is None:
        return None
    return root[0], f"{label}:{key} ({path[:72]}{'…' if len(path) > 72 else ''}) is on the {root[1]}"


# --------------------------------------------------------------------------------------------
# decode: resolved from sent + generation_config
# --------------------------------------------------------------------------------------------

DECODE_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
VLLM_DEFAULTS = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}


def decode_resolved(sent: Mapping[str, object], generation_config: Optional[Mapping[str, object]],
                    fallback: Optional[Mapping[str, object]] = None, fallback_name: str = "vllm_default") -> Dict[str, object]:
    """Fill un-sent keys from the served snapshot's ``generation_config.json``, marking each as
    ``not_sent_resolved_to``; a model with no file resolves to ``fallback`` (vLLM defaults) marked
    ``assumed_default``."""
    out: Dict[str, object] = {}
    for k in DECODE_KEYS:
        if k in sent and sent[k] is not None:
            out[k] = {"value": sent[k], "source": "sent"}
        elif generation_config and k in generation_config:
            out[k] = {"value": generation_config[k], "source": "not_sent_resolved_to", "from": "generation_config.json"}
        else:
            fb = fallback if fallback is not None else VLLM_DEFAULTS
            out[k] = {"value": fb.get(k), "source": "not_sent_resolved_to", "from": fallback_name,
                      "confidence": "assumed_default"}
    return out


# --------------------------------------------------------------------------------------------
# strength
# --------------------------------------------------------------------------------------------

def strength_nominal(coefficient: Optional[float], operating_point_coefficient: Optional[float],
                     filed_nominal: Optional[float] = None) -> Derived:
    """The producer-FILED nominal wins (``strength_nominal_source = filed``); else coefficient /
    ruled operating-point coefficient (``derived``); else unverified."""
    if filed_nominal is not None:
        return (float(filed_nominal), "recorded", "producer-filed nominal (dose column)")
    if coefficient is not None and operating_point_coefficient:
        return (float(coefficient) / float(operating_point_coefficient), "derived",
                "coefficient / ruled operating-point coefficient")
    return ("unverified", "unverified", "no filed nominal and no operating_point ruling to derive it from")


def is_operating_point(filed_id: Optional[str], ruled_id: Optional[str], object_sha: Optional[str] = None,
                       ruled_object_sha: Optional[str] = None) -> Derived:
    """Bind the OBJECT HASH when a ruling names one, else compare the filed id to the ruled id."""
    if ruled_object_sha and object_sha:
        return (1 if object_sha == ruled_object_sha else 0, "derived", "operating_point ruling target object hash")
    if ruled_id is None:
        return (None, "unverified", "no operating_point ruling for this key")
    return (1 if filed_id == ruled_id else 0, "derived", "filed operating-point id vs ruled id")


#: The object kinds whose dose is a NOMINAL multiplier on the fitted write — the fitted LoRA / ReLU objects, their differences and
#: the undistilled paired write (`composed`). The campaign's shipped object is nominal 1.0, and 1.0 IS the operating point
#: (the rule, 2026-09-20; the roster's `strength_rule_from_dose`: `1.0 -> operating_point; any other filed nominal x ->
#: fixed:<x>`; the strength ladder and relu_paired readers file `is_operating_point = (dose == 1.0)` the same way). The published
#: baselines (`static_r_arditi`, `alphasteer`, `vector_all`) are on their own dose axes: alpha, lambda and the raw coefficient c, whose
#: operating points their own readers rule (arditi_strength, alphasteer_rebake, arditi_allpos) — never a 1.0 here.
FITTED_DOSE_OBJECT_KINDS: Tuple[str, ...] = ("lora", "lora_subtraction", "relu_steer", "relu_steer_distilled_difference", "composed")
FITTED_OPERATING_NOMINAL = 1.0
FITTED_OPERATING_POINT_RULE = ("dose 1.0 is the operating point of the fitted LoRA / ReLU objects and the paired write (the user's ruling, "
                               "2026-09-20; the roster's strength_rule_from_dose reads 1.0 as operating_point)")
#: The dose a SERVED KEY spells (the full22 generator's standup / target-directory spelling): a bare key is the campaign nominal 1.0,
#: `_halfdose` is 0.5, `_d<tok>` the token (`_d0p25` 0.25, `_dm1` -1.0, `_d1p5` 1.5). The one home of that grammar: `paper._served_key_dose`
#: reads the frame's `dose` from it where a run files no strength, and the legacy-store readers file `run.strength_nominal` from it.
SERVED_KEY_DOSE_SUFFIX = re.compile(r"_(?P<word>halfdose|d(?P<neg>m?)(?P<whole>\d+)(?:p(?P<frac>\d+))?)$")
#: The SEEDED dose-family standups (the fit-seed replicates of the difference objects, 2026-09-22; the control objects at -1,
#: 2026-09-25) spell the dose in the MIDDLE: `<model>_d<tok>_<family>s<N>` -- the dose token, then the family segment `<family>s<N>` naming
#: the seed-free object family and the fit seed N of the batch-64 bank it serves (`strength_mtdiff.STANDUP_RE` / `FAMILY_SEGMENT_RE`, the one
#: rule; `sampled_static.tag_model_key` strips the same spelling for the model key). :data:`SEEDED_DOSE_FAMILIES` is the ONE home of the
#: family alternation every reader of the spelling builds its pattern from: `loradiff64` the difference LoRA, `distdiff64` the distilled
#: difference relu_steer, `ctllora64` the control LoRA and `ctlrelu64` the control relu_steer (the batch-64 seed CONTROL parents dose-scaled,
#: the control-negative battery); `strength_mtdiff.FAMILIES` declares each family's unit and object key and is pinned equal to this tuple
#: at import. Read BEFORE the trailing grammar: to it these tags are bare (1.0), and the sidecar cross-check then refused all eleven seeded
#: statics by name (2026-09-24, `_items_common.fitted_strength_payload`). A tag spelling neither grammar is still the bare key.
SEEDED_DOSE_FAMILIES: Tuple[str, ...] = ("distdiff64", "loradiff64", "ctllora64", "ctlrelu64")
_SEEDED_DOSE_FAMILY_ALTERNATION = "|".join(re.escape(f) for f in SEEDED_DOSE_FAMILIES)
SERVED_KEY_SEEDED_DOSE = re.compile(r"_d(?P<neg>m?)(?P<whole>\d+)(?:p(?P<frac>\d+))?_(?:" + _SEEDED_DOSE_FAMILY_ALTERNATION + r")s[1-9]\d*$")
#: a served-controls target directory names its decode replicate after the tag (`qwen35_4b_it_t1`, `qwen35_4b_it_d0p5_t0`,
#: `qwen35_4b_d1p25_loradiff64s1_t0`): the suffix is dropped before the dose token is read, so a replicate of a dose view keeps that view's dose
_SERVED_KEY_REPLICATE_SUFFIX = re.compile(r"_t\d+$")


def served_key_dose(view: Optional[str]) -> Optional[float]:
    """The dose a served key (a standup tag / target-directory name) spells: 1.0 for a bare key, 0.5 for `_halfdose`, `_d0p25` -> 0.25,
    `_dm1` -> -1.0, and the seeded dose-family spelling `_d<tok>_<family>s<N>` (:data:`SEEDED_DOSE_FAMILIES`) its token
    (`qwen35_4b_d1p25_loradiff64s1` -> 1.25, `qwen35_4b_dm0p5_distdiff64s1` -> -0.5, `qwen35_4b_dm1_ctllora64s2` -> -1.0); a trailing `_t<T>`
    decode-replicate suffix is dropped first; None for no key."""
    if not isinstance(view, str) or not view:
        return None
    key = _SERVED_KEY_REPLICATE_SUFFIX.sub("", view)
    m = SERVED_KEY_SEEDED_DOSE.search(key) or SERVED_KEY_DOSE_SUFFIX.search(key)
    if m is None:
        return 1.0
    if m.re is SERVED_KEY_DOSE_SUFFIX and m.group("word") == "halfdose":
        return 0.5
    whole, frac = m.group("whole"), m.group("frac")
    v = float(f"{whole}.{frac}" if frac is not None else whole)
    return -v if m.group("neg") else v


def fitted_is_operating_point(object_kind: Optional[str], nominal: Optional[float]) -> Optional[int]:
    """1 iff a FITTED object (:data:`FITTED_DOSE_OBJECT_KINDS`) is served at the campaign nominal 1.0, 0 at any other nominal; None for
    every other object kind (unsteered, prompted, a published baseline on its own dose axis) and for no nominal."""
    if object_kind not in FITTED_DOSE_OBJECT_KINDS or nominal is None:
        return None
    return 1 if abs(float(nominal) - FITTED_OPERATING_NOMINAL) < 1e-9 else 0


# --------------------------------------------------------------------------------------------
# legacy spellings (mapping supplied by configs/registry/legacy_spellings.yaml)
# --------------------------------------------------------------------------------------------

def model_id_from_legacy(view_legacy: str, mapping: Mapping[str, str]) -> Derived:
    if view_legacy in mapping:
        return (mapping[view_legacy], "recorded", "legacy_spellings.yaml")
    return ("unverified", "unverified", f"legacy model spelling {view_legacy!r} not in legacy_spellings.yaml")


def eval_id_from_spelling(spelling: str, mapping: Mapping[str, str]) -> Derived:
    if spelling in mapping:
        return (mapping[spelling], "recorded", "legacy_spellings.yaml / eval.producer_spellings_json")
    return ("unverified", "unverified", f"eval spelling {spelling!r} not mapped")


# --------------------------------------------------------------------------------------------
# the campaign stem of a staged spelling (the paper label is keyed on the STEM, never on the fit-variant prefix)
# --------------------------------------------------------------------------------------------

#: A staged object directory name is ``<fit-variant prefix>__<stem>`` where the STEM is the campaign's ``<si>__<arm>`` pair
#: (``constv2-mix-full__lora_r4_ce``, ``safety_refusal__psr_kl_all``, ``constv2-mix-full__psr_ce_all_minus_ctl_distilled``) and the
#: prefix is fit provenance / dose, never part of the paper's arm: ``base`` / ``base64s<s>`` (the base-checkpoint fits and their
#: batch-64 seed replicates), ``itdoc`` / ``itdoc64`` (the instruct-checkpoint document fits at effective batch 8 / 64), ``halfdose``
#: / ``halfdoseit``, ``d<dose>`` / ``dm<dose>`` with an ``it`` / ``it64`` / ``s<s>`` / ``_eb64s<s>`` tail (dose-scaled copies),
#: ``derived`` / ``derived_<variant>`` (subtractions, distillations and paired writes), ``si12`` (the scale-12 two-SI fits) and
#: ``candsi`` (the candidate-instruction distillations, whose stem ``A10__lora_r4`` names the candidate). A ``_withdrawn/`` directory
#: appends ``.withdrawn-<tag>`` to the whole name. The prefix rides ``object.extra_json.fit_variant_prefix_legacy`` (its meaning is
#: in ``fit_model_id`` / ``base_scale`` / ``fit_replicate_label`` / ``fit_effective_batch``); the paper label, the roster and the
#: rulings read the stem.
_WITHDRAWN_SUFFIX_RE = re.compile(r"\.withdrawn[-A-Za-z0-9_.]*$")


def campaign_stem_legacy(spelling: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(stem, fit-variant prefix)`` of a campaign / staged spelling: the stem is the LAST two ``__``-joined segments (the
    ``<si>__<arm>`` pair), the prefix everything before them (None when the spelling is already a stem or a bare arm).
    ``base__constv2-mix-full__lora_r4_ce`` → ``("constv2-mix-full__lora_r4_ce", "base")``; ``d0p5_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl``
    → ``("constv2-mix-full__psr_ce_all_minus_ctl", "d0p5_eb64s1")``; ``candsi__A10__lora_r4`` → ``("A10__lora_r4", "candsi")``;
    ``constv2ctl2-full__lora_r4_ce`` → ``(itself, None)``; ``none`` → ``("none", None)``."""
    if spelling is None:
        return None, None
    text = _WITHDRAWN_SUFFIX_RE.sub("", str(spelling))
    parts = text.split("__")
    if len(parts) <= 2:
        return text, None
    return "__".join(parts[-2:]), "__".join(parts[:-2])


# --------------------------------------------------------------------------------------------
# fit_target_kind from sidecar shapes
# --------------------------------------------------------------------------------------------

class UnclassifiableSidecar(ValueError):
    """The sidecar satisfies none of the fit_target_kind rules; lists the keys it saw."""


#: The legacy teacher roster, consulted ONLY when the caller passes no ``teacher_ids`` (adapters pass the `si_text`
#: seed rows, so a candidate instruction the seeds know is a teacher and one they do not is unclassifiable, never guessed).
_SI_TEACHERS = ("safety_refusal", "si_constitution", "si_constitution_full", "si_irrelevant")


#: the sidecars the dose-scale staging writes its record into (all three carry the same three keys)
DOSE_SCALE_DERIVATION_SIDECARS: Tuple[str, ...] = ("staged_from.json", "fit_provenance.json", "object.pt.provenance.json")


def dose_scale_derivation(sidecars: Mapping[str, Mapping[str, object]]) -> Optional[Tuple[str, Mapping[str, object]]]:
    """The Arditi lane's dose-scale spelling, when a sidecar carries it: ``derivation: dose_scale`` beside a non-empty
    ``parent_object_ids`` list (``params.base_scale`` names the factor). Returns ``(sidecar name, the mapping)`` of the first
    sidecar in :data:`DOSE_SCALE_DERIVATION_SIDECARS` that spells it, else None. The lane writes the record into
    ``staged_from.json``, ``fit_provenance.json`` (the PARENT's fit record, copied) and ``object.pt.provenance.json`` alike
    (scripts/arditi/make_arditi_allpos_objects.py, the c = 0.15 rung of 2026-09-19)."""
    for name in DOSE_SCALE_DERIVATION_SIDECARS:
        d = sidecars.get(name)
        if not isinstance(d, Mapping) or str(d.get("derivation") or "") != "dose_scale":
            continue
        parents = d.get("parent_object_ids")
        if isinstance(parents, list) and parents and all(isinstance(x, str) for x in parents):
            return name, d
    return None


def _dose_stamp(sidecars: Mapping[str, Mapping[str, object]]) -> Optional[str]:
    """The sidecar key that marks a dose-scaled copy — ``object.pt.provenance.json:dose_scale``, a
    ``train_budget_meta.json:dose_scaled_from`` block, or the staged ``derivation: dose_scale`` + ``parent_object_ids``
    record (:func:`dose_scale_derivation`) — or None. Mirrors the adapter's own reader: a bare non-mapping ``dose_scaled_from``
    is not a stamp, and a ``derivation`` word without a parent list is not one either."""
    keys = []
    if (sidecars.get("object.pt.provenance.json") or {}).get("dose_scale") is not None:
        keys.append("object.pt.provenance.json:dose_scale")
    if isinstance((sidecars.get("train_budget_meta.json") or {}).get("dose_scaled_from"), Mapping):
        keys.append("train_budget_meta.json:dose_scaled_from")
    spelled = dose_scale_derivation(sidecars)
    if spelled is not None:
        keys.append(f"{spelled[0]}:derivation=dose_scale+parent_object_ids")
    return " / ".join(keys) if keys else None


def fit_target_kind(object_kind: str, sidecars: Mapping[str, Mapping[str, object]],
                    corpus_ids: Mapping[str, str], teacher_ids: Optional[Container[str]] = None) -> Derived:
    """Decide the kind from the sidecar's own fields, in the documented precedence; refuse ambiguity.

    ``teacher_ids`` is the set of instruction ids the registry knows as teachers (the `si_text` seed rows); when None
    the legacy roster :data:`_SI_TEACHERS` stands in. A dose stamp is read BEFORE ``train_budget_meta.json``: a
    dose-scaled copy carries its PARENT's fit record, which would otherwise classify it as the parent's kind."""
    if "derived_meta.json" in sidecars:
        return ("derived", "recorded", "derived_meta.json present")
    dose = _dose_stamp(sidecars)
    if dose is not None:
        return ("derived", "recorded", dose + " (a dose-scaled copy; train_budget_meta.json is the parent's fit record)")
    if object_kind == "none":
        return ("none", "derived", "object_kind=none")
    if object_kind == "system_instruction":
        return ("declared_prompt", "derived", "object_kind=system_instruction")
    if "fit_meta.json" in sidecars or object_kind in ("diffmean", "caa_static_r", "static_r_arditi", "alphasteer", "antidote"):
        return ("contrastive_direction", "recorded", "fit_meta.json / contrastive object kind")
    tbm = sidecars.get("train_budget_meta.json")
    if tbm:
        si_id = tbm.get("si_id")
        data_mode = str(tbm.get("data_mode", ""))
        teacher_forward = tbm.get("teacher_forward")
        teachers: Container[str] = _SI_TEACHERS if teacher_ids is None else teacher_ids
        is_corpus = si_id in corpus_ids or "corpus" in data_mode or "document" in data_mode
        is_teacher = si_id in teachers and bool(teacher_forward)
        if is_corpus and is_teacher:
            raise UnclassifiableSidecar(
                f"train_budget_meta.json is ambiguous: si_id={si_id!r} reads as both a corpus and a teacher "
                f"(data_mode={data_mode!r}, teacher_forward={teacher_forward!r}); add a legacy_spellings override")
        if is_corpus:
            return ("corpus_document_fit", "recorded", "train_budget_meta.json:si_id names a document corpus")
        if is_teacher:
            return ("system_instruction_distillation", "recorded",
                    "train_budget_meta.json:si_id + teacher_forward" + ("" if teacher_ids is None else " (si_id has an si_text row)"))
    raise UnclassifiableSidecar(
        "no fit_target_kind rule matched; sidecars seen: "
        + json.dumps({k: sorted(v.keys()) for k, v in sidecars.items()}, sort_keys=True))
