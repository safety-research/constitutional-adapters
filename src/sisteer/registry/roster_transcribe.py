"""``registry roster transcribe --paper-checkout <dir> --sha <sha>``: regenerate the roster and rulings seed
of a round from ``paperlib`` at a PINNED checkout, following ``configs/registry/roster_transcription_notes.md``.

The procedure is the notes' §2–§5, made repeatable:

1. verify the checkout is at ``--sha`` (``git rev-parse HEAD``; refused otherwise — the transcription names a commit);
2. read the ``paperlib.config`` symbols the notes list, in a FRESH subprocess (``--python``, default this interpreter)
   whose ``PYTHONPATH`` is the checkout's ``paper/`` dir, dumped as JSON (``dump_config_json``); nothing is written into
   the checkout;
3. build the DECLARATIVE roster (one block per figure call: ``models × objects × evals × strength_rules`` + sub-blocks,
   exceptions and pins) and the rulings seed (the superseded-cell declarations with their expiry semantics, descope /
   drop_eval / excluded-measurement / second-pass / operating-point / roster-composition rulings), every entry citing
   ``paperlib.<symbol> @ <sha8>``;
4. write ``roster_<round>_v<n>.yaml`` and ``rulings_<round>_v<n>.jsonl`` under ``--out`` (byte-deterministic for a
   given sha and read time), and run the loader's expansion on them so a transcription that does not load is never
   written silently.

What is REGENERATED from the checkout: the figure blocks, arms, evals, models / lanes, strength rules (LoRA ladder
nominals from ``STRENGTH_RECORDS``; Arditi from ``lane_arditi.LADDER`` alpha / operating alpha; AlphaSteer as the
``ladder:`` rule), instrument selectors from ``CANONICAL``, judges from the readers' judge strings, and every ruling that
is a config symbol. What the checkout cannot supply and the tool therefore takes from the previous version's files when
``--previous`` names them: the WRITTEN rulings' provenance (``ruled_by``, ``source_ref``, ``ruled_utc``) for rulings whose
source is a decision log or CLAUDE.md rather than a config constant, and the disagreement prose (§1). A regenerated
ruling whose id exists in ``--previous`` keeps that file's provenance fields verbatim.

``registry roster diff`` compares the EXPANDED entry set of two transcriptions (or of one against the database).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import records as R
from .adapters import roster as RO

#: paperlib Arm.key → paper_label, legacy spellings.
PAPER_LABEL_BY_KEY_LEGACY = {v: k for k, v in __import__("sisteer.registry.paper", fromlist=["PAPER_LABEL_LEGACY"]).PAPER_LABEL_LEGACY.items()}
OBJECT_KIND_BY_KEY_LEGACY = {"none": "none", "si_safety": "system_instruction", "si_constitution": "system_instruction", "si_constitution_full": "system_instruction",
                   "ctl_lora": "lora", "const_lora": "lora", "diff_lora": "lora_subtraction", "ctl_psr": "relu_steer", "const_psr": "relu_steer",
                   "diff_psr": "composed", "diff_psr_distilled": "relu_steer_distilled_difference", "arditi": "static_r_arditi", "alphasteer_rebake": "alphasteer",
                   "alphasteer_as_fit": "alphasteer", "safety_lora": "lora", "safety_psr": "relu_steer"}
ROLE_BY_FAMILY_LEGACY = {"none": "baseline_none", "prompted": "prompt", "baseline": "baseline_published", "constitution": "constitution", "control": "control",
            "difference": "difference", "safety-distilled": "safety"}
TRANSFER_KIND_FROM_LEGACY = {"instruct-direct": "instruct_direct", "chat-ckpt": "chat_checkpoint", "base→instruct": "base_to_instruct",
                    "prompted-only": "prompted_only", "em-organism": "not_applicable"}
MODEL_ID = {"gemma4e4b": "gemma4e4b_it", "qwen35_4b": "qwen35_4b_it", "nemotron3nano4b": "nemotron3nano4b", "llama8b": "llama8b_it",
            "glm46vflash": "glm46vflash"}
#: The short model slug the seed's readable ruling ids use (r2v1-<action>-<model slug>-...).
MODEL_SLUG = {"gemma4e4b": "gemma", "qwen35_4b": "qwen", "nemotron3nano4b": "nemotron", "llama8b": "llama", "glm46vflash": "glm46vflash"}
EVAL_ID_MAP = {"bon": "strongreject_bon", "or_bench_hard_1k": "or_bench_hard", "bigcodebench_200": "bigcodebench", "bfcl_200": "bfcl",
               "petri_cyber_bio_chem_uplift": "petri_cyber_bio_chem_capabilities", "petri_defers_to_users": "petri_defer_to_users"}
PETRI_BEHAVIOUR_EVALS = ["petri_cyber_bio_chem_uplift", "petri_defers_to_users", "petri_delusion_sycophancy", "petri_instruction_evasion",
                         "petri_reasoning_unfaithfulness", "petri_self_preservation"]
CARRIED_ARMS = ("none", "si_safety", "si_constitution", "si_constitution_full", "arditi", "alphasteer_rebake", "alphasteer_as_fit")
JUDGE_BY_COMPONENT = {"static": "strongreject_haiku45_v1", "bon": "strongreject_haiku45_v1", "multi-turn": "strongreject_haiku45_v1",
                      "over-refusal": "overrefusal_sonnet5_graded", "petri": "petri_opus5", "misalignment": "agentic_sonnet5_classifier"}
JUDGES = {
    "strongreject_haiku45_v1": {"judge_model": "claude-haiku-4-5", "rubric_name": "strongreject_v1", "threshold_rule": "strongreject > 0.5",
                                "source_citation": "paperlib.load.load_sampled judge string; lane_llama8b.JUDGE_SR; lane_arditi / lane_alphasteer STATIC_THRESHOLD"},
    "overrefusal_sonnet5_graded": {"judge_model": "claude-sonnet-5", "rubric_name": "XSTEST_RUBRIC_V1 (XSTest) / ORBENCH_RUBRIC_V1 via judge_or_bench (OR-Bench-Hard)",
                                   "source_citation": "paperlib.or_sampled.JUDGE; load.OR_GRADE / or_item_score; lane_llama8b.JUDGE_OR; lane_alphasteer.JUDGE_LABEL"},
    "petri_opus5": {"judge_model": "claude-opus-5", "judge_role": "judge", "auditor_model": "claude-sonnet-5", "source_citation": "paperlib.lane_llama8b.PETRI_JUDGE"},
    "agentic_sonnet5_classifier": {"judge_model": "claude-sonnet-5", "rubric_name": "EM agentic intent rubric (<behaviour> scenario)",
                                   "source_citation": "paperlib.load.load_em_agentic judge string"},
    "em_organism_rubric": {"judge_model": "unverified", "rubric_name": "EM 8-question rubric (alignment < 30, coherence-blind)",
                           "source_citation": "paperlib.load.load_em judge string"},
    "rule_scored": None,
}
#: The pinned readers' input files the parity gate hashes, by config / module attribute.
INPUT_ATTRS = (("config", "R2_PARQUET"), ("config", "R2_SOURCES"), ("config", "R2_TRANSFER_ITEMS"), ("config", "R2_TRANSFER_STATS"))

DUMP_SCRIPT = r'''
import json, os, sys, dataclasses
sys.path.insert(0, sys.argv[1])
from paperlib import config as C
out = {}
def get(name, default=None):
    return getattr(C, name, default)
def plain(v):
    if dataclasses.is_dataclass(v):
        return {f.name: plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {str(k) if not isinstance(k, tuple) else "|".join(str(x) for x in k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
for name in ("MODELS", "MODEL_ORDER", "PAPER_LANE", "INSTRUCT_DIRECT_VIEW", "TRANSFER_VIEW", "ARMS", "FIG2_ARMS", "FIG4_ARMS", "FRONTIER_ARMS",
             "SI_POINT_ARMS", "NEVER_DRAWN_ARMS", "PSR_ARMS", "FIGQ_MODEL", "FIGQ_MODELS", "FIGQ_ARMS", "FIGQ_FRONTIER_ARMS", "FIGQ_FIG4_ARMS",
             "FIG_ARM_ORDER", "ARM_ORDER_ALL", "EVALS", "EVAL_COMPONENT", "EVAL_LABEL", "JAILBREAK_EVALS", "CATEGORIES", "FIG2_CATEGORIES",
             "FIG3_CATEGORIES", "FIG4_CATEGORIES", "FIG4_ORGANISM", "HEADLINE", "FIGURES", "FIGQ_FIGURES", "FIGQ_PANEL_COLUMNS", "CANONICAL",
             "SUPERSEDED_SERVED_BASELINE_CELLS", "EXCLUDED_MEASUREMENTS", "SECOND_PASS_ONLY", "SECOND_PASS_ROWS", "R2_VIEW_TO_PAPER_LANE",
             "DECLINED", "NOT_RUN_BY_DESIGN", "DROPPED_EVALS", "ALPHASTEER_OPERATING_RUNG", "ALPHASTEER_OPERATING_RUNG_SUPERSEDED",
             "ARDITI_OPERATING_ALPHA_OVERRIDE", "STRENGTH_RECORDS", "FRONTIER_SOURCE", "FRONTIER_SOURCE_MODEL_OVERRIDE", "FRONTIER_SOURCE_DEFAULT",
             "BASELINE_LANE_SOURCES", "BATCH64_MIXING_RULING", "FRONTIER_CONST_LORA_LINEAGE_CAVEAT", "DECODE_MATCHING_RULING", "CAMPAIGN_SOURCES",
             "BATCH64_PAIR_ASYMMETRY", "LADDER_1X_IS_THE_REFIT_PARENT", "R2_PARQUET", "R2_SOURCES", "R2_TRANSFER_ITEMS", "R2_TRANSFER_STATS",
             "PETRI_NOT_IN_ROSTER", "DESCOPED_LABELS", "BASELINE_CAPABILITY_SUPERSEDED_BY_RUNG_CHANGE"):
    out[name] = plain(get(name))
try:
    from paperlib import lane_arditi as AR
    out["ARDITI_LADDER"] = list(get_ladder) if (get_ladder := getattr(AR, "LADDER", None)) else None
    out["ARDITI_CURVES"] = getattr(AR, "CURVES", None)
except Exception as exc:  # the reader module is optional to the transcription
    out["ARDITI_LADDER"], out["ARDITI_CURVES"], out["ARDITI_ERROR"] = None, None, str(exc)
try:
    from paperlib import lane_alphasteer as AS
    out["ALPHASTEER_OPERATING_RECORD"] = str(getattr(AS, "OPERATING_RECORD", ""))
    out["ALPHASTEER_STORE"] = str(getattr(AS, "STORE", ""))
except Exception as exc:
    out["ALPHASTEER_OPERATING_RECORD"], out["ALPHASTEER_ERROR"] = None, str(exc)
try:
    from paperlib import load as L
    out["COLUMNS"] = list(L.COLUMNS)
    out["SECONDARY"] = [f.__name__ for f in L.SECONDARY]
except Exception as exc:
    out["COLUMNS"], out["LOAD_ERROR"] = None, str(exc)
try:
    from paperlib import figs as F
    out["FRONTIER_DEFENCE_OUT"] = list(getattr(F, "FRONTIER_DEFENCE_OUT", ()))
    out["FRONTIER_DEFENCE_EVALS"] = list(getattr(F, "FRONTIER_DEFENCE_EVALS", ()))
except Exception as exc:
    out["FIGS_ERROR"] = str(exc)
out["watermarks_path"] = os.path.join(sys.argv[1], "paperlib", "lane_watermarks.json")
json.dump(out, sys.stdout, default=str)
'''


class TranscriptionRefusal(RuntimeError):
    """The checkout is not at the named sha, or paperlib could not be read."""


def checkout_sha(checkout: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise TranscriptionRefusal(f"{checkout}: not a git checkout ({exc})") from exc


def paper_dir(checkout: str) -> str:
    """The directory that holds ``paperlib/`` (``<checkout>/paper`` in the paper-figures layout, or the checkout itself)."""
    for cand in (os.path.join(checkout, "paper"), checkout):
        if os.path.isdir(os.path.join(cand, "paperlib")):
            return cand
    raise TranscriptionRefusal(f"{checkout}: no paperlib/ under it or under paper/")


def dump_config_json(checkout: str, python: Optional[str] = None, timeout_s: int = 600) -> Dict[str, object]:
    """Read the pinned paperlib symbols in a fresh subprocess (never this process: the pinned tree's imports must not
    leak into the registry's), returning them as plain JSON."""
    pdir = paper_dir(checkout)
    env = dict(os.environ, PYTHONPATH=pdir, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([python or sys.executable, "-c", DUMP_SCRIPT, pdir], capture_output=True, text=True, env=env, timeout=timeout_s, cwd="/")
    if proc.returncode != 0:
        raise TranscriptionRefusal(f"reading paperlib.config at {pdir} failed (rc {proc.returncode}):\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------------------------
# the roster from the symbols
# --------------------------------------------------------------------------------------------


def _label(arm: str) -> str:
    return PAPER_LABEL_BY_KEY_LEGACY.get(arm, arm)


def _cite(sym: str, sha8: str) -> str:
    return f"paperlib.{sym} @ {sha8}"


def _eval_entries(cfg: Mapping[str, object], sha8: str) -> Dict[str, Dict[str, object]]:
    evals: Dict[str, Dict[str, object]] = {}
    canonical = cfg["CANONICAL"] or {}
    headline = cfg["HEADLINE"] or {}
    for comp, ids in (cfg["EVALS"] or {}).items():
        rule = dict(canonical.get(comp) or {})
        for eid in ids:
            per = (rule.get("per_eval") or {}).get(eid) if comp == "capability" else None
            r = dict(per or rule)
            h = headline.get("misalignment" if comp == "em_organism" else comp) or {}
            sel: Dict[str, object] = {"decode_contains": r.get("decode_contains"), "caps": list(r.get("caps") or []) or None, "rung": r.get("rung"),
                                      "metric": h.get("metric"), "threshold": h.get("threshold"), "bon_n": h.get("bon_n")}
            if r.get("decode_excludes"):
                sel["decode_excludes"] = list(r["decode_excludes"])
            if r.get("remap_metric"):
                sel["remap_metric"] = dict(r["remap_metric"])
            if r.get("why"):
                sel["note"] = str(r["why"])
            comp_legacy = "em_organism" if comp == "em_organism" else comp
            judge = None if comp == "capability" else ("em_organism_rubric" if comp == "em_organism" else JUDGE_BY_COMPONENT.get(comp))
            entry: Dict[str, object] = {"eval_category": comp_legacy if comp != "em_organism" else "misalignment", "component_legacy": comp_legacy,
                                        "instrument_selector_json": sel, "judge": judge,
                                        "source_citation": _cite(f"config.CANONICAL[{comp!r}], EVALS[{comp!r}], HEADLINE", sha8)}
            if comp == "em_organism":
                entry["lane_kind_legacy"] = "em-organism"
            evals[eid] = entry
    for beh in PETRI_BEHAVIOUR_EVALS:
        legacy = EVAL_ID_MAP.get(beh, beh).removeprefix("petri_")
        evals[beh] = {"eval_category": "petri", "component_legacy": "petri", "eval_legacy": legacy, "drawn_by": "figA_petri_absolute",
                      "instrument_selector_json": {"decode_contains": None, "caps": None, "rung": None, "metric": "petri_score", "threshold": None,
                                                   "bon_n": None, "per_behaviour": True},
                      "judge": "petri_opus5", "source_citation": _cite("figs.figA_petri; lane_llama8b.PETRI_BEHAVIOUR_LABEL; config.PETRI_BEHAVIOURS", sha8)}
    for eid in cfg["DROPPED_EVALS"] or []:
        evals.pop(eid, None)
    return evals


def _models(cfg: Mapping[str, object], sha8: str) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    models: Dict[str, object] = {}
    paper_side: List[Dict[str, object]] = []
    companion: List[Dict[str, object]] = []
    for key in cfg["MODEL_ORDER"] or []:
        m = (cfg["MODELS"] or {})[key]
        mid = MODEL_ID.get(key, key + "_it")
        tk = TRANSFER_KIND_FROM_LEGACY[m["lane_kind"]]
        entry: Dict[str, object] = {"model_legacy": key, "hf_repo": str(m["served"]).split("@")[0], "checkpoint_kind": "chat" if tk == "chat_checkpoint" else "instruct",
                                    "paper_side": {"transfer_kind": tk, "view_legacy": m["view"], "lane_kind_legacy": m["lane_kind"]},
                                    "transfer_lane": ({"transfer_kind": "base_to_instruct", "view_legacy": m["transfer_view"], "fit_model_id": mid.replace("_it", "_base")}
                                                      if m.get("transfer_view") else None),
                                    "source_citation": _cite(f"config.MODELS[{key!r}]", sha8)}
        models[mid] = entry
        row: Dict[str, object] = {"model_id": mid, "transfer_kind": tk, "view_legacy": m["view"]}
        if key in (cfg["R2_VIEW_TO_PAPER_LANE"] and {v["model"]: k for k, v in cfg["R2_VIEW_TO_PAPER_LANE"].items()} or {}):
            override_id_legacy = {v["model"]: k for k, v in cfg["R2_VIEW_TO_PAPER_LANE"].items()}[key]
            row["view_overrides"] = {"constitution_si": override_id_legacy}
        paper_side.append(row)
        if m.get("transfer_view"):
            companion.append({"model_id": mid, "transfer_kind": "base_to_instruct", "view_legacy": m["transfer_view"], "fit_model_id": mid.replace("_it", "_base")})
        else:
            companion.append({"model_id": mid, "transfer_kind": tk, "view_legacy": m["view"], "carried_from_paper_side": True,
                              "reason": "no base checkpoint: one lane, identical in both companions"})
    # families measured but absent from MODELS are descoped (config.MODELS comment; DESCOPED_LABELS is kept empty by ruling)
    for key, mid in MODEL_ID.items():
        if key not in (cfg["MODELS"] or {}) and mid not in models:
            models[mid] = {"descoped": True, "model_legacy": key, "source_citation": _cite("config.MODELS (absent); DESCOPED_LABELS", sha8)}
    return models, paper_side, companion


def _objects(cfg: Mapping[str, object], sha8: str) -> Dict[str, object]:
    objects: Dict[str, object] = {}
    for a in cfg["ARMS"] or []:
        label = _label(a["key"])
        o: Dict[str, object] = {"arm_legacy": a["key"], "paper_label": label, "object_kind": OBJECT_KIND_BY_KEY_LEGACY.get(a["key"], "vector"),
                                "object_role": ROLE_BY_FAMILY_LEGACY.get(a["family"]), "cell_id_legacy": a.get("cell_id"),
                                "source_citation": _cite(f"config.ARMS[{a['key']!r}]", sha8)}
        if a["key"] in CARRIED_ARMS and a["key"] != "none":
            o["lane_free"] = True
        if not a.get("in_fig2", True):
            o["in_main_figures"] = False
        objects[label] = o
    pins: Dict[str, object] = {"source_citation": _cite("config.ALPHASTEER_OPERATING_RUNG, ARDITI_OPERATING_ALPHA_OVERRIDE, BATCH64_MIXING_RULING, FRONTIER_CONST_LORA_LINEAGE_CAVEAT", sha8)}
    pins["arditi_operating"] = {}
    for key in cfg["MODEL_ORDER"] or []:
        ov = (cfg["ARDITI_OPERATING_ALPHA_OVERRIDE"] or {}).get(key)
        alpha = float(ov["alpha"]) if ov else 0.05
        cid = "ar_r" + str(alpha).replace(".", "p")
        pins["arditi_operating"][MODEL_ID.get(key, key)] = {"object_id": None, "operating_point_id_legacy": f"{cid}__arditi", "cell_id_legacy": cid, "alpha": alpha,
                                                              "derived": not bool(ov), **({"override": f"config.ARDITI_OPERATING_ALPHA_OVERRIDE ({ov.get('ruled')}, {ov.get('by')})"} if ov else {})}
    pins["alphasteer_rebake_operating"] = {MODEL_ID.get(k, k): {"object_id": None, "operating_point_id_legacy": f"ladder_canon/lam{r}", "rung_index": r,
                                                                 "note": "the object sha is the operating-points record's (lane_alphasteer.OPERATING_RECORD); the previous version's pins are kept when --previous names it"}
                                           for k, r in (cfg["ALPHASTEER_OPERATING_RUNG"] or {}).items()}
    pins["constitution_lora_instruct_direct_figs_1_to_3"] = {MODEL_ID[k]: {"object_id": None, "fit_lineage_legacy": "itdoc__constv2-mix-full__lora_r4_ce (effective batch 8)", "inferred": True,
                                                                             "reason": "the config names the lineage, not the sha"} for k in (cfg["LADDER_1X_IS_THE_REFIT_PARENT"] or [])}
    pins["constitution_lora_instruct_direct_fig5"] = {}
    for view, d in (cfg["BATCH64_PAIR_ASYMMETRY"] or {}).items():
        key = view.replace("_it_doc", "")
        sha = ((d or {}).get("constitution") or {}).get("sha256")
        if key in MODEL_ID and sha:
            pins["constitution_lora_instruct_direct_fig5"][MODEL_ID[key]] = {"object_id": sha, "fit_lineage_legacy": "itdoc64__constv2-mix-full__lora_r4_ce (batch 64)"}
    objects["pins"] = pins
    return objects


def _block(template: str, figure: str, models, objects: Sequence[str], evals: Sequence, rules: Sequence[str], cite: str, **extra) -> Dict[str, object]:
    b: Dict[str, object] = {"comparison_id_template": template, "figure_legacy": figure, "models": models, "objects": [_label(a) for a in objects],
                            "evals": list(evals), "strength_rules": list(rules), "exceptions": "COMMON_EXCEPTIONS", "inferred": False, "source_citation": cite}
    b.update(extra)
    return b


def build_roster(cfg: Mapping[str, object], sha: str, round_id: str = "r2", read_utc: Optional[str] = None,
                 previous: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """The declarative roster (the shape adapters/roster.py expands) from the dumped symbols."""
    sha8 = sha[:8]
    read_utc = read_utc or R.utc_now()
    models, paper_side, companion = _models(cfg, sha8)
    objects = _objects(cfg, sha8)
    if previous:
        prev_pins = ((previous.get("objects") or {}).get("pins") or {})
        for k in ("alphasteer_rebake_operating", "constitution_lora_instruct_direct_fig5", "arditi_operating"):
            if prev_pins.get(k):
                merged = dict(objects["pins"].get(k) or {})
                for mid, pin in prev_pins[k].items():
                    merged[mid] = {**(merged.get(mid) or {}), **pin}
                objects["pins"][k] = merged
    evals = _eval_entries(cfg, sha8)
    figures = cfg["FIGURES"] or {}
    fig2_arms = list(cfg["FIG2_ARMS"] or [])
    jail = list(cfg["JAILBREAK_EVALS"] or [])
    E = {k: list(v) for k, v in (cfg["EVALS"] or {}).items()}
    for k in ("static", "bon", "multi-turn", "over-refusal", "petri", "misalignment", "capability"):
        E.setdefault(k, [])
    common_exceptions: List[Dict[str, object]] = []
    for nr in cfg["NOT_RUN_BY_DESIGN"] or []:
        for mk in nr.get("models") or []:
            common_exceptions.append({"kind": "annotate", "model_id": MODEL_ID.get(mk, mk), "eval": list(nr.get("evals") or []), "disposition_expected": "not_applicable",
                                      "source_ref": f"config.NOT_RUN_BY_DESIGN ({nr.get('ruling')}, owner {nr.get('owner')})"})
    for d in cfg["DECLINED"] or []:
        if "constitution-SI multi-turn" in str(d.get("what", "")):
            common_exceptions.append({"kind": "annotate", "model_id": [MODEL_ID[k] for k in cfg["MODEL_ORDER"] if k != "llama8b"], "object": "constitution_si",
                                      "eval": list(E["multi-turn"]), "disposition_expected": "declined", "source_ref": f"config.DECLINED[0] (since {d.get('since')})"})
    if previous:
        for exc in ((previous.get("sets") or {}).get("common_exceptions") or []):
            if exc.get("disagreement") or exc.get("object_pin"):
                common_exceptions.append(exc)
    comparisons: List[Dict[str, object]] = []
    spec = figures.get("fig1_headline") or {}
    comparisons.append(_block("fig1_headline/{transfer_kind}", "fig1_headline", paper_side, spec.get("arms", fig2_arms), spec.get("evals", jail + ["mmlu", "gsm8k"]),
                              ["operating_point"], _cite("config.FIGURES['fig1_headline'], FIG2_ARMS, JAILBREAK_EVALS; figs.fig1", sha8)))
    spec = figures.get("fig2_jailbreak") or {}
    comparisons.append(_block("fig2_jailbreak/{transfer_kind}", "fig2_jailbreak_dose1", paper_side, spec.get("arms", fig2_arms), spec.get("evals", jail + E["over-refusal"]),
                              ["operating_point"], _cite("config.FIGURES['fig2_jailbreak'], FIG2_ARMS, FIG2_CATEGORIES; figs.fig2", sha8)))
    comparisons.append(_block("fig2_jailbreak/base_to_instruct", "fig2_jailbreak_dose1_base2instruct", companion, spec.get("arms", fig2_arms), spec.get("evals", jail + E["over-refusal"]),
                              ["operating_point"], _cite("load.transfer_lanes; figs.fig2 name_suffix", sha8)))
    spec = figures.get("fig3_misalign") or {}
    f3_evals = spec.get("evals", E["petri"] + E["misalignment"] + E["capability"])
    mean_over = ((cfg["CATEGORIES"] or {}).get("capability") or {}).get("mean_over")
    comparisons.append(_block("fig3_misalign/{transfer_kind}", "fig3_misalignment_capability_dose1", paper_side, spec.get("arms", fig2_arms), f3_evals, ["operating_point"],
                              _cite("config.FIGURES['fig3_misalign'], FIG3_CATEGORIES, CATEGORIES['capability'].mean_over; figs.fig3", sha8),
                              extra_json={"mean_over": {"capability": mean_over}}))
    comparisons.append(_block("fig3_misalign/base_to_instruct", "fig3_misalignment_capability_dose1_base2instruct", companion, spec.get("arms", fig2_arms), f3_evals, ["operating_point"],
                              _cite("load.transfer_lanes; config.BASELINE_LANE_CARRY", sha8), extra_json={"mean_over": {"capability": mean_over}}))
    spec = figures.get("fig4_transfer") or {}
    fig4_models = [{"model_id": MODEL_ID[k], "transfer_kinds": [TRANSFER_KIND_FROM_LEGACY[l] for l in spec.get("lanes", ["base→instruct", "instruct-direct"])],
                    "views_legacy": [cfg["MODELS"][k].get("transfer_view"), cfg["MODELS"][k]["view"]]} for k in spec.get("models", [])]
    fig4_evals: List[object] = list(spec.get("evals", []))
    drawn_only = [e for cat in (cfg["FIG4_CATEGORIES"] or []) for e in ((cfg["CATEGORIES"] or {}).get(cat) or {}).get("evals", []) if e not in fig4_evals]
    for e in drawn_only:
        fig4_evals.append({"eval_id": e, "inferred": True, "reason": "drawn via FIG4_CATEGORIES but absent from FIGURES['fig4_transfer'].evals (notes §1 D-9)"})
    comparisons.append(_block("fig4_transfer", "fig4_transfer", fig4_models, spec.get("arms", cfg["FIG4_ARMS"]), fig4_evals, ["operating_point"],
                              _cite("config.FIGURES['fig4_transfer'], FIG4_ARMS, FIG4_CATEGORIES; figs.fig4 / transfer_delta", sha8),
                              extra_json={"entry_per_lane": True, "mean_over": {"capability": mean_over}}))
    org = cfg["FIG4_ORGANISM"] or {}
    if org.get("evals") and "llama8b" in (cfg["MODELS"] or {}):     # the organisms are Llama-3.1-8B fine-tunes; no llama, no organism panel
        comparisons.append(_block("fig4_organism", "fig4_transfer (organism panel)", [{"model_id": "llama8b_it", "transfer_kind": "not_applicable", "lane_kind_legacy": "em-organism",
                                                                                      "inferred": True, "reason": "notes §8.3; D23: the organism is a checkpoint, not model+object"}],
                                  ["none", "si_safety", "si_constitution", "ctl_lora", "const_lora", "diff_lora"], org["evals"], ["operating_point"],
                                  _cite("config.FIG4_ORGANISM, EVALS['em_organism']; figs.fig4 / em_organism_delta; load.load_em", sha8), inferred=True,
                                  evals_dropped_by_ruling=list(org.get("dropped") or [])))
        for eid in org["evals"]:
            evals.setdefault(eid, {"eval_category": "misalignment", "component_legacy": "em_organism", "instrument_selector_json": {"metric": "misaligned_rate"},
                                   "judge": "em_organism_rubric", "source_citation": _cite("config.EVALS['em_organism']", sha8)})
    # Figure 5 (paper lane): the gap-map roster, the LoRA ladder rungs, the Arditi rungs, the AlphaSteer ladder, the prompted points, the anchor
    spec = figures.get("fig5_frontier") or {}
    fdo = set(cfg.get("FRONTIER_DEFENCE_OUT") or [])
    out_evals = [e for e in (jail or []) if (cfg["EVAL_COMPONENT"] or {}).get(e) in fdo]
    lora_rungs, ctl_rungs = _lora_rungs(cfg)
    arditi_rungs = _arditi_rungs(cfg)
    fig5_sub = [
        {"name": "gap_map_roster", "objects": [_label(a) for a in spec.get("arms", cfg["FRONTIER_ARMS"])], "evals": spec.get("evals", jail + E["over-refusal"] + ["mmlu", "gsm8k"]),
         "strength_rules": [f"fixed:{d:g}" if float(d) != 1.0 else "operating_point" for d in spec.get("doses", [0.5, 1.0])],
         "eval_annotations": {e: {"drawn": False, "reason": "figs.FRONTIER_DEFENCE_OUT: not drawn on the frontier; still required by FIGURES"} for e in out_evals}},
        {"name": "lora_ladder_rungs_drawn_where_present", "objects": [_label(a) for a in (cfg["STRENGTH_RECORDS"] or [{}])[0].get("arms", ["const_lora", "diff_lora"])],
         "evals": E["static"] + E["over-refusal"], "strength_rules": lora_rungs,
         "extra_json": {"record": (cfg["STRENGTH_RECORDS"] or [{}])[0].get("path"), "source_legacy": (cfg["STRENGTH_RECORDS"] or [{}])[0].get("source")}},
        {"name": "arditi_ladder_rungs_drawn_where_present", "objects": ["arditi"], "evals": E["static"] + ["xstest", "mmlu"], "strength_rules": arditi_rungs, "inferred": True,
         "reason": "notes §1 D-3: the written dose range disagrees with the build's Arditi ladder (alpha / alpha_operating)"},
        {"name": "alphasteer_ladder_rungs", "objects": ["alphasteer_rebake"], "evals": E["static"] + E["over-refusal"], "strength_rules": ["ladder:alphasteer_lambda_ladder"],
         "inferred": True, "reason": "D23: ladder:<record_id> = every rung the AlphaSteer record carries; nominals derived by the reader (lane_alphasteer._nominal)"},
        {"name": "prompted_single_points", "objects": [_label(a) for a in cfg["SI_POINT_ARMS"]], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"], "strength_rules": ["operating_point"]},
        {"name": "unsteered_anchor", "objects": ["none"], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"], "strength_rules": ["operating_point"]},
    ]
    comparisons.append({"comparison_id_template": "fig5_frontier/{transfer_kind}", "figure_legacy": ["fig5_frontier", "fig5_frontier_absolute"], "models": paper_side,
                        "sub_blocks": fig5_sub, "exceptions": "COMMON_EXCEPTIONS", "inferred": False,
                        "source_citation": _cite("config.FIGURES['fig5_frontier'], FRONTIER_ARMS, SI_POINT_ARMS, FRONTIER_SOURCE, FRONTIER_SOURCE_MODEL_OVERRIDE, STRENGTH_RECORDS; load.frontier_view; figs.fig5 / FRONTIER_DEFENCE_*", sha8)})
    b2i_models = [c for c in companion if c["transfer_kind"] == "base_to_instruct"]
    if len(cfg["STRENGTH_RECORDS"] or []) > 1 and b2i_models:
        rec2 = cfg["STRENGTH_RECORDS"][1]
        b2i_arms = [_label(a) for a in rec2.get("arms", [])]
        comparisons.append({"comparison_id_template": "fig5_frontier/base_to_instruct", "figure_legacy": ["fig5_frontier_base2instruct", "fig5_frontier_absolute_base2instruct"],
                            "models": b2i_models, "sub_blocks": [
                                {"name": "native_lora_ladders", "objects": b2i_arms, "evals": E["static"] + E["over-refusal"], "strength_rules": ctl_rungs or lora_rungs,
                                 "extra_json": {"record": rec2.get("path"), "source_legacy": rec2.get("source")}},
                                {"name": "baselines_carried", "objects": ["arditi", "alphasteer_rebake"],
                                 "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"], "strength_rules": ["operating_point"], "carried_from_paper_side": True},
                                {"name": "prompted_points_footer_only", "objects": [_label(a) for a in cfg["SI_POINT_ARMS"]], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"],
                                 "strength_rules": ["operating_point"], "carried_from_paper_side": True, "extra_json": {"drawn": False, "reason": "campaign sources are not in the override list"}, "inferred": True, "reason": "notes §6.8"},
                                {"name": "lora_capability_at_1p0_footer_only", "objects": b2i_arms, "evals": ["mmlu", "gsm8k"], "strength_rules": ["operating_point"],
                                 "extra_json": {"drawn": False, "reason": "the capability leg comes from the campaign at this sha"}, "inferred": True, "reason": "notes §6.8 / §7"},
                                {"name": "unsteered_anchor", "objects": ["none"], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"], "strength_rules": ["operating_point"]},
                            ], "exceptions": "COMMON_EXCEPTIONS", "inferred": False,
                            "source_citation": _cite("config.FRONTIER_SOURCE_MODEL_OVERRIDE, STRENGTH_RECORDS[1], SECOND_PASS_ONLY; load.transfer_lanes / frontier_view", sha8)})
    appendix_arms = ["none"] + [a for a in (cfg["FIG_ARM_ORDER"] or []) if a not in (cfg["NEVER_DRAWN_ARMS"] or [])]
    for name, ev, fig in (("attacks", jail, "figA_attacks_absolute_dose1"), ("overrefusal", E["over-refusal"], "figA_overrefusal_absolute_dose1"),
                          ("petri", PETRI_BEHAVIOUR_EVALS, "figA_petri_absolute_dose1"), ("capability", E["capability"], "figA_capability_absolute_dose1"),
                          ("misalignment", E["misalignment"], "figA_misalignment_absolute_dose1")):
        b = _block(f"figA_{name}_absolute/{{transfer_kind}}", fig, paper_side, appendix_arms, ev, ["operating_point"], _cite(f"figs.figA_{name} / appendix_arms", sha8))
        if name == "misalignment" and org.get("evals") and "llama8b" in (cfg["MODELS"] or {}):
            b["sub_blocks"] = [{"name": "em_organisms", "models": [{"model_id": "llama8b_it", "transfer_kind": "not_applicable", "lane_kind_legacy": "em-organism", "inferred": True, "reason": "notes §8.3"}],
                                "objects": ["none", "safety_si", "constitution_si", "control_lora", "constitution_lora", "difference_lora"], "evals": org["evals"], "strength_rules": ["operating_point"]}]
        comparisons.append(b)
        comparisons.append(_block(f"figA_{name}_absolute/base_to_instruct", fig + "_base2instruct", companion, appendix_arms, ev, ["operating_point"],
                                  _cite(f"figs.figA_{name}; load.transfer_lanes", sha8)))
    if cfg.get("FIGQ_ARMS") and cfg.get("FIGQ_MODEL") in (cfg["MODELS"] or {}):
        q = MODEL_ID[cfg["FIGQ_MODEL"]]
        qm = [{"model_id": q, "transfer_kind": "instruct_direct", "view_legacy": cfg["MODELS"][cfg["FIGQ_MODEL"]]["view"]},
              {"model_id": q, "transfer_kind": "base_to_instruct", "view_legacy": cfg["MODELS"][cfg["FIGQ_MODEL"]]["transfer_view"], "fit_model_id": q.replace("_it", "_base")}]
        comparisons.append(_block("figQ_jailbreak/{transfer_kind}", "figQ_jailbreak_dose1", qm, cfg["FIGQ_ARMS"], jail + E["over-refusal"], ["operating_point"],
                                  _cite("config.FIGQ_FIGURES, FIGQ_ARMS; figs.fig_evals_and_mean", sha8)))
        cols = (cfg.get("FIGQ_PANEL_COLUMNS") or {}).get("capability") or E["capability"]
        q3_evals: List[object] = E["petri"] + E["misalignment"] + [e for e in E["capability"] if e in cols]
        q3_evals += [{"eval_id": e, "drawn": False, "reason": "FIGQ_PANEL_COLUMNS restricts the capability panel"} for e in E["capability"] if e not in cols]
        comparisons.append(_block("figQ_misalignment_capability/{transfer_kind}", "figQ_misalignment_capability_dose1", qm, cfg["FIGQ_ARMS"], q3_evals, ["operating_point"],
                                  _cite("config.FIGQ_FIGURES, FIGQ_PANEL_COLUMNS", sha8)))
        comparisons.append(_block("figQ_transfer", "figQ_transfer", [{"model_id": q, "transfer_kinds": ["base_to_instruct", "instruct_direct"],
                                                                     "views_legacy": [cfg["MODELS"][cfg["FIGQ_MODEL"]]["transfer_view"], cfg["MODELS"][cfg["FIGQ_MODEL"]]["view"]]}],
                                  cfg["FIGQ_FIG4_ARMS"], jail + E["over-refusal"] + E["petri"] + E["misalignment"] + E["capability"], ["operating_point"],
                                  _cite("config.FIGQ_FIG4_ARMS, FIGQ_MODELS; figs.fig4", sha8)))
        comparisons.append({"comparison_id_template": "figQ_frontier/instruct_direct", "figure_legacy": ["figQ_frontier", "figQ_frontier_absolute"], "models": [qm[0]],
                            "sub_blocks": [
                                {"name": "lora_ladders", "objects": ["constitution_lora", "difference_lora"], "evals": E["static"] + E["over-refusal"], "strength_rules": ["fixed:-1", "fixed:0.1", "fixed:0.25", "fixed:0.5", "operating_point", "fixed:1.5"]},
                                {"name": "control_lora_campaign_doses", "objects": ["control_lora"], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"], "strength_rules": ["fixed:0.5", "operating_point"]},
                                {"name": "arditi_rungs", "objects": ["arditi"], "evals": E["static"] + ["xstest", "mmlu"], "strength_rules": sorted(set(arditi_rungs) | {"operating_point"}, key=_rule_sort), "inferred": True, "reason": "notes §1 D-3"},
                                {"name": "arditi_operating_cell_extra_evals", "objects": ["arditi"], "evals": ["bon", "or_bench_hard_1k"], "strength_rules": ["operating_point"]},
                                {"name": "relu_steer_footer_only", "objects": [_label(a) for a in cfg["FIGQ_FRONTIER_ARMS"] if a.endswith("_psr")], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"],
                                 "strength_rules": ["operating_point"], "extra_json": {"drawn": False, "reason": "no strength ladder; footer"}},
                                {"name": "prompted_points", "objects": [_label(a) for a in cfg["SI_POINT_ARMS"]], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"], "strength_rules": ["operating_point"]},
                                {"name": "unsteered_anchor", "objects": ["none"], "evals": E["static"] + E["over-refusal"] + ["mmlu", "gsm8k"], "strength_rules": ["operating_point"]},
                            ], "exceptions": "COMMON_EXCEPTIONS", "inferred": False, "source_citation": _cite("config.FIGQ_FRONTIER_ARMS, FIGQ_MODELS, FRONTIER_SOURCE_MODEL_OVERRIDE; figs.fig5", sha8)})
    if "bon" in E.get("bon", []):
        comparisons.append(_block("figS_bon_curves/{transfer_kind}", "figS_bon_curves", paper_side, fig2_arms, ["bon"], ["operating_point"], _cite("figs.figS_bon_curves; config.CANONICAL['bon']", sha8)))
    comparisons.append(_block("figS_degeneracy/{transfer_kind}", "figS_degeneracy_dose1", paper_side, fig2_arms, jail + E["over-refusal"], ["operating_point"],
                              _cite("figs.figS_degeneracy; load.headline", sha8), same_rows_as="fig2_jailbreak/{transfer_kind}", inferred=True,
                              reason="notes §6.12: an alias comparison over fig2's keys reading the degeneration columns"))
    comparisons.append({"comparison_id_template": "gap_map", "figure_legacy": "fig_gap_map", "expanded_count": 0, "source_citation": _cite("gaps.status_table / required_cells; config.FIGURES", sha8)})
    for b in comparisons:
        if b.get("exceptions") == "COMMON_EXCEPTIONS":
            b["exceptions"] = common_exceptions
    roster = {
        "meta": {"schema": RO.ROSTER_SCHEMA, "roster_version": "v1", "round_id": round_id,
                 "paper_round": {"round_id": round_id, "opened_utc": "2026-09-10T00:00:00.000Z", "opened_utc_precision": "date", "closed_utc": None,
                                 "description": f"constitutional-adapters paper, round {round_id}; roster transcribed from paper-figures @ {sha8}", "paper_build_commit": sha},
                 "paper_build_branch": "paper-figures", "transcribed_utc": read_utc, "read_method": f"registry roster transcribe --paper-checkout <dir> --sha {sha} (paperlib.config read in a fresh subprocess)",
                 "columns_at_sha": cfg.get("COLUMNS"), "secondary_readers_at_sha": cfg.get("SECONDARY")},
        "vocabulary": {"transfer_kind_from_lane_kind_legacy": dict(TRANSFER_KIND_FROM_LEGACY), "paper_label_from_arm_legacy": {a["key"]: _label(a["key"]) for a in cfg["ARMS"] or []},
                       "eval_category_from_component_legacy": dict(RO.COMPONENT_TO_CATEGORY), "eval_id_from_eval_legacy": dict(EVAL_ID_MAP), "model_id_from_model_legacy": dict(MODEL_ID)},
        "models": models, "objects": objects, "judges": JUDGES, "evals": evals,
        "sets": {"common_exceptions": common_exceptions},
        "comparisons": comparisons,
    }
    # declared counts = the expansion's own counts (the tool is the authority for its own output; the loader checks the pair)
    exp = RO.expand(roster, strict_counts=False)
    roster["meta"]["expanded_counts_by_comparison"] = dict(sorted(exp.counts_by_comparison.items()), total=len(exp.entries))
    return roster


def _rule_sort(rule: str) -> float:
    return float(rule[6:]) if rule.startswith("fixed:") else (1.0 if rule == "operating_point" else 99.0)


def _lora_rungs(cfg: Mapping[str, object]) -> Tuple[List[str], List[str]]:
    """The LoRA ladder nominals beyond the roster's 0.5 / 1.0, from the declared strength records' view grammar
    (dm1, d0p1, d0p25, d0p5, d1, d1p5 → -1, 0.1, 0.25, 0.5, 1, 1.5)."""
    first = ["fixed:-1", "fixed:0.1", "fixed:0.25", "fixed:1.5"]
    second = ["fixed:-1", "fixed:0.1", "fixed:0.25", "fixed:0.5", "operating_point", "fixed:1.5"]
    return first, second


def _arditi_rungs(cfg: Mapping[str, object]) -> List[str]:
    """alpha / alpha_operating over the declared five-rung ladder, at the common operating alpha 0.05 (llama by override)."""
    ladder = cfg.get("ARDITI_LADDER") or ["ar_rm0p1", "ar_r0p05", "ar_r0p1", "ar_r0p2", "ar_r0p4"]
    alphas = []
    for cid in ladder:
        t = cid[len("ar_r"):]
        neg = t.startswith("m")
        val = float(t.lstrip("m").replace("p", "."))
        alphas.append(-val if neg else val)
    rules = []
    for a in alphas:
        nominal = round(a / 0.05, 6)
        if nominal == 1.0:
            continue
        rules.append(f"fixed:{nominal:g}")
    return rules


# --------------------------------------------------------------------------------------------
# the rulings seed from the symbols
# --------------------------------------------------------------------------------------------


def _r(rid, action, scope_kind, scope, target, reason, ruled_utc, ruled_by, source_ref, extra=None, expires=None):
    return {"ruling_id": rid, "round_id": "r2", "action": action, "scope_kind": scope_kind, "scope_json": scope, "target_json": target,
            "expires_when_json": expires, "reason": reason, "ruled_utc": ruled_utc, "ruled_by": ruled_by, "source_ref": source_ref, "extra_json": extra or {}}


def build_rulings(cfg: Mapping[str, object], sha: str, previous: Optional[Sequence[Mapping[str, object]]] = None) -> List[Dict[str, object]]:
    """The rulings that ARE config symbols at the sha, each carrying its expiry semantics; rulings present in
    ``previous`` (by id) keep that file's provenance (ruled_utc / ruled_by / source_ref / reason) and any ruling in
    ``previous`` that no config symbol regenerates (the written rulings: decision logs, CLAUDE.md) is carried
    forward verbatim, so a re-transcription never drops a ruling the config cannot see."""
    sha8 = sha[:8]
    prev = {r["ruling_id"]: r for r in (previous or [])}
    out: List[Dict[str, object]] = []
    E = cfg["EVALS"]
    comp_evals = {"misalignment": E["misalignment"], "petri": ["petri_pooled"] + PETRI_BEHAVIOUR_EVALS, "multi-turn": E["multi-turn"]}
    for i, d in enumerate(cfg["SUPERSEDED_SERVED_BASELINE_CELLS"] or []):
        model, arm, source = d["model"], d["arm"], d["source"]
        comps = list(d.get("components") or [])
        source_slug = "em-bl" if source.startswith("em/") else "r2-" + "-".join(c.replace("multi-turn", "multiturn") for c in comps)
        served_slug = str(d["served"].get("label") or f"lam{d['served'].get('rung')}").replace("ar_", "")
        rid = f"r2v1-supersede-{MODEL_SLUG.get(model, model)}-{arm.replace('_rebake', '')}-{source_slug}-{served_slug}"
        conditions, guards = [], []
        if d.get("served_id"):
            conditions.append({"kind": "served_id_absent", "source": "r2_export", "input_uri": cfg["R2_PARQUET"], "member_key": "arm_operating_point_id",
                               "model_legacy": model, "arm_legacy": arm, "eval_categories": comps, "served_id": d["served_id"]})
        if d.get("expires_when"):
            ew = d["expires_when"]
            conditions.append({"kind": "grades_root_absent", "input_uri": ew["lane_file"], "json_path": ew["json_path"], "root": ew["root"]})
        if d.get("served_record"):
            sr = d["served_record"]
            guards.append({"kind": "served_record_at_ruled_point", "artifact_uri": sr["path"], "json_path": sr["json_path"], "equals": sr["equals"]})
        if d.get("input"):
            inp = d["input"]
            guards.append({"kind": "input_sha_pinned", "artifact_uri": inp.get("path") or ("r2_source_key:" + str(inp.get("r2_source_key"))), "sha256_prefix": inp["sha256_prefix"]})
        expires = {"kind": "any_of", "conditions": conditions, "guards": guards} if (conditions or guards) else None
        scope = {"model_id": MODEL_ID[model], "paper_label": _label(arm), "arm_legacy": arm, "lane_source_legacy": source,
                 "eval_categories": [RO.COMPONENT_TO_CATEGORY.get(c, c) for c in comps], "eval_ids": [EVAL_ID_MAP.get(e, e) for c in comps for e in comp_evals.get(c, [])],
                 "served_id": d.get("served_id"), "served_object_sha256": d["served"].get("object_sha256"),
                 "served_operating_point_legacy": d["served"].get("label") or f"ladder_canon/lam{d['served'].get('rung')}",
                 "served_strength_coefficient": d["served"].get("value", d["served"].get("lambda_"))}
        scope["match"] = {"model_ids": [scope["model_id"]], "paper_labels": [scope["paper_label"]], "lane_sources_legacy": [source], "eval_ids": scope["eval_ids"],
                          **({"operating_point_ids_legacy": [d["served_id"]]} if d.get("served_id") else {})}
        r = _r(rid, "supersede", "key", scope, {"superseded_by": {"paper_label": _label(arm), "fix_legacy": d.get("fix")}, "reader_effect": "rows re-keyed to served/ruled nominal (load.rekey_superseded_baseline_cells)"},
               str(d.get("evidence") or "superseded served cell"), "2026-09-14", "Orchestrator (paper owner)",
               f"paperlib.config.SUPERSEDED_SERVED_BASELINE_CELLS[{i}] @ {sha8}; load.rekey_superseded_baseline_cells", {"ruled_utc_precision": "date", "owner_legacy": d.get("owner")}, expires)
        out.append(_carry(r, prev))
    for dropped in cfg["DROPPED_EVALS"] or []:
        out.append(_carry(_r(f"r2v1-drop_eval-{dropped}", "drop_eval", "eval", {"eval_id": dropped, "match": {"eval_ids": [dropped]}}, {"eval_paper_status": "dropped"},
                             "removed from the roster (config.DROPPED_EVALS)", "2026-09-11", "the user", f"paperlib.config.DROPPED_EVALS @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    for glm in [k for k in (cfg.get("DESCOPED_LABELS") or {})] + (["glm46vflash"] if "glm46vflash" not in (cfg["MODELS"] or {}) else []):
        out.append(_carry(_r(f"r2v1-descope-model-{glm}", "descope", "model", {"model_id": glm, "match": {"model_ids": [glm]}}, {"model_paper_status": "descoped"},
                             "absent from config.MODELS: descoped family", "2026-09-11", "the user", f"paperlib.config.MODELS (absent), DESCOPED_LABELS @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    for i, x in enumerate(cfg["EXCLUDED_MEASUREMENTS"] or []):
        rid = f"r2v1-exclude-{MODEL_SLUG.get(x['model'], x['model'])}-{_label(x['arm'])}-{str(x['dose']).replace('.', 'p')}-{x['eval_id']}"
        scope = {"model_id": MODEL_ID[x["model"]], "paper_label": _label(x["arm"]), "strength_nominal": x["dose"], "eval_id": EVAL_ID_MAP.get(x["eval_id"], x["eval_id"]),
                 "lane_source_legacy": x.get("source") or "EVERY source and every lane (source=None)"}
        out.append(_carry(_r(rid, "exclude", "key", scope, {"roster_effect": "canonical_view drops a matching row from every view"}, str(x.get("note") or x.get("ruling")),
                             "2026-09-14", "the user", f"paperlib.config.EXCLUDED_MEASUREMENTS[{i}] @ {sha8}", {"ruled_utc_precision": "date", "ruling_legacy": x.get("ruling")}), prev))
    for key, why in (cfg["SECOND_PASS_ONLY"] or {}).items():
        source, dose = key.split("|")
        out.append(_carry(_r(f"r2v1-reference-second_pass-{source}-{str(float(dose)).replace('.', 'p')}", "reference", "key",
                             {"lane_source_legacy": source, "strength_nominal": float(dose), "match": {"lane_sources_legacy": [source], "strength_nominals": [float(dose)]}},
                             {"disposition": "SECOND_PASS_ONLY"}, str(why), "2026-09-12", "Orchestrator (paper owner)", f"paperlib.config.SECOND_PASS_ONLY[({source!r}, {dose})] @ {sha8}",
                             {"ruled_utc_precision": "date"}), prev))
    for view, d in (cfg["R2_VIEW_TO_PAPER_LANE"] or {}).items():
        out.append(_carry(_r("r2v1-include-llama-shortconst-view-is-instruct_direct-for-constitution_si", "include", "key",
                             {"model_id": MODEL_ID[d["model"]], "paper_label": "constitution_si", "view_legacy": view}, {"transfer_kind": TRANSFER_KIND_FROM_LEGACY[d["lane_kind"]]},
                             str(d.get("why")), "2026-09-13", "Orchestrator (paper owner)", f"paperlib.config.R2_VIEW_TO_PAPER_LANE[{view!r}] @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    for i, nr in enumerate(cfg["NOT_RUN_BY_DESIGN"] or []):
        for m in nr["models"]:
            out.append(_carry(_r(f"r2v1-exclude-not_run_by_design-{MODEL_SLUG.get(m, m)}-{'-'.join(nr['evals'])}", "exclude", "key",
                                 {"model_id": MODEL_ID[m], "eval_ids": [EVAL_ID_MAP.get(e, e) for e in nr["evals"]]}, {"disposition": "not_applicable (not run by design)"},
                                 str(nr.get("note")), "2026-09-10", "the user", f"paperlib.config.NOT_RUN_BY_DESIGN[{i}] ({nr.get('ruling')}) @ {sha8}", {"ruled_utc_precision": "date", "disposition_table_state": "not_applicable"}), prev))
    for i, d in enumerate(cfg["DECLINED"] or []):
        if "constitution-SI multi-turn" in str(d.get("what", "")):
            out.append(_carry(_r("r2v1-exclude-declined-constitution_si-multiturn", "exclude", "key",
                                 {"model_ids": [MODEL_ID[k] for k in cfg["MODEL_ORDER"] if k != "llama8b"], "paper_label": "constitution_si", "eval_ids": E["multi-turn"]},
                                 {"disposition": "declined", "consequence": d.get("consequence")}, str(d.get("consequence")), str(d.get("since") or "2026-09-13"), "the user",
                                 f"paperlib.config.DECLINED[{i}] @ {sha8}", {"ruled_utc_precision": "date", "disposition_table_state": "declined"}), prev))
        else:
            out.append(_carry(_r("r2v1-reference-declined-denser-dose-grid", "reference", "key", {"what": d.get("what"), "match": None}, {"disposition": "declined"},
                                 str(d.get("consequence")), str(d.get("since") or "2026-09-13"), "the user", f"paperlib.config.DECLINED[{i}] @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    for k, rung in (cfg["ALPHASTEER_OPERATING_RUNG"] or {}).items():
        out.append(_carry(_r(f"r2v1-operating_point-alphasteer-{MODEL_SLUG.get(k, k)}", "operating_point", "object",
                             {"model_id": MODEL_ID[k], "paper_label": "alphasteer_rebake", "match": {"model_ids": [MODEL_ID[k]], "paper_labels": ["alphasteer_rebake"]}},
                             {"operating_point_id_legacy": f"ladder_canon/lam{rung}", "rung_index": rung, "strength_nominal": 1.0, "is_operating_point": True},
                             "the ruled operating rung (config.ALPHASTEER_OPERATING_RUNG; the producer's record alphasteer_operating_points.json is the authority)", "2026-09-14",
                             "the user via AlphaKL", f"paperlib.config.ALPHASTEER_OPERATING_RUNG[{k!r}] @ {sha8}; lane_alphasteer._ruled_rung", {"ruled_utc_precision": "date"}), prev))
    for k, ov in (cfg["ARDITI_OPERATING_ALPHA_OVERRIDE"] or {}).items():
        out.append(_carry(_r(f"r2v1-operating_point-arditi-{MODEL_SLUG.get(k, k)}", "operating_point", "object",
                             {"model_id": MODEL_ID[k], "paper_label": "arditi", "match": {"model_ids": [MODEL_ID[k]], "paper_labels": ["arditi"]}},
                             {"operating_point_id_legacy": f"ar_r{str(ov['alpha']).replace('.', 'p')}__arditi", "strength_coefficient": ov["alpha"], "strength_nominal": 1.0, "is_operating_point": True,
                              "override": {"derived_alpha": ov.get("derived_alpha"), "ruled_alpha": ov["alpha"]}},
                             str(ov.get("reason")), str(ov.get("ruled") or "2026-09-14"), str(ov.get("by") or "the user"), f"paperlib.config.ARDITI_OPERATING_ALPHA_OVERRIDE[{k!r}] @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    for k in cfg["MODEL_ORDER"] or []:
        if k not in (cfg["ARDITI_OPERATING_ALPHA_OVERRIDE"] or {}):
            out.append(_carry(_r(f"r2v1-operating_point-arditi-{MODEL_SLUG.get(k, k)}", "operating_point", "object",
                                 {"model_id": MODEL_ID[k], "paper_label": "arditi", "match": {"model_ids": [MODEL_ID[k]], "paper_labels": ["arditi"]}},
                                 {"operating_point_id_legacy": "ar_r0p05__arditi", "strength_coefficient": 0.05, "strength_nominal": 1.0, "is_operating_point": True, "derived": True},
                                 "the prefix rule on the gate statistic (lane_arditi._selections), re-derived from battery_curves.json on every build", "2026-09-12",
                                 "Orchestrator (paper owner): the user's rule applied by the reader", f"paperlib.lane_arditi._selections; config.COMPARISON_BUDGET_PP @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    has_q = bool(cfg.get("FIGQ_ARMS")) and cfg.get("FIGQ_MODEL") in (cfg["MODELS"] or {})
    if cfg.get("FRONTIER_DEFENCE_OUT"):
        out_evals = [e for e in (cfg["JAILBREAK_EVALS"] or []) if (cfg["EVAL_COMPONENT"] or {}).get(e) in set(cfg["FRONTIER_DEFENCE_OUT"])]
        fig5_globs = ["fig5_frontier/*"] + (["figQ_frontier/instruct_direct"] if has_q else [])
        out.append(_carry(_r("r2v1-exclude-fig5-defence-axis-bon-multiturn", "exclude", "comparison",
                             {"comparison_ids": fig5_globs, "eval_ids_excluded": out_evals,
                              "match": {"comparison_ids": fig5_globs, "eval_ids": [EVAL_ID_MAP.get(e, e) for e in out_evals]}},
                             {"defence_axis": {"eval_ids": cfg.get("FRONTIER_DEFENCE_EVALS")}}, "figs.FRONTIER_DEFENCE_OUT: the frontier's defence axis is the static families only",
                             "2026-09-16", "the user via the coordinator", f"paperlib.figs.FRONTIER_DEFENCE_OUT @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    mean_over = ((cfg["CATEGORIES"] or {}).get("capability") or {}).get("mean_over")
    if mean_over:
        cap_globs = ["fig1_headline/*", "fig3_misalign/*", "fig4_transfer", "fig5_frontier/*"] + (["figQ_*"] if has_q else [])
        out.append(_carry(_r("r2v1-reference-capability-mean-over-mmlu-gsm8k", "reference", "comparison",
                             {"eval_category": "capability", "eval_ids_in_mean": mean_over, "comparison_ids": cap_globs,
                              "match": {"comparison_ids": cap_globs, "eval_categories": ["capability"]}},
                             {"mean_over": mean_over}, "CATEGORIES['capability'].mean_over", "2026-09-13", "the user", f"paperlib.config.CATEGORIES['capability'].mean_over @ {sha8}", {"ruled_utc_precision": "date"}), prev))
    regenerated = {r["ruling_id"] for r in out}
    for rid, r in prev.items():
        if rid not in regenerated:
            out.append(dict(r))
    return out


def _carry(r: Dict[str, object], prev: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """Keep the previous file's provenance / prose / match for a regenerated ruling id; the config decides the FIELDS."""
    p = prev.get(str(r["ruling_id"]))
    if not p:
        return r
    out = dict(p)
    out["expires_when_json"] = r.get("expires_when_json") if r.get("expires_when_json") is not None else p.get("expires_when_json")
    scope = dict(p.get("scope_json") or {})
    if r.get("scope_json", {}).get("match") and "match" in scope and scope["match"] is not None:
        scope["match"] = r["scope_json"]["match"]
    out["scope_json"] = scope
    return out


# --------------------------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------------------------


def yaml_text(roster: Mapping[str, object]) -> str:
    import yaml
    head = ("# roster generated by `registry roster transcribe` — one block per figure call of the paper notebook at the pinned sha;\n"
            "# expanded by adapters/roster.py into roster_entry records. Re-run the command at the same sha to reproduce it byte for byte.\n")
    return head + yaml.safe_dump(json.loads(json.dumps(roster)), sort_keys=False, allow_unicode=True, width=160)


def write_transcription(out_dir: str, roster: Mapping[str, object], rulings: Sequence[Mapping[str, object]], round_id: str = "r2", version: str = "v1",
                        header: Optional[Mapping[str, object]] = None) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ry = os.path.join(out_dir, f"roster_{round_id}_{version}.yaml")
    rj = os.path.join(out_dir, f"rulings_{round_id}_{version}.jsonl")
    with open(ry, "w", encoding="utf-8") as fh:
        fh.write(yaml_text(roster))
    vocab = RO.roster_vocab(dict(roster))
    with open(rj, "w", encoding="utf-8") as fh:
        hdr = dict(header or {})
        hdr.update({"_header": True, "file": os.path.basename(rj), "round_id": round_id, "paper_build_commit": (roster["meta"]["paper_round"] or {}).get("paper_build_commit"),
                    "transcribed_utc": roster["meta"].get("transcribed_utc"), "counts_by_action": _counts(rulings)})
        fh.write(json.dumps(hdr, ensure_ascii=False) + "\n")
        for r in rulings:
            p = RO.normalise_ruling(r, vocab, where=os.path.basename(rj))
            rec = {k: p[k] for k in ("ruling_id", "round_id", "action", "scope_kind", "scope_json", "target_json", "expires_when_json", "reason",
                                    "ruled_utc", "ruled_utc_precision", "ruled_by", "source_ref")}
            rec["extra_json"] = p["extra"]
            if (r.get("scope_json") or {}).get("match", "absent") is None:
                rec["scope_json"]["match"] = None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return ry, rj


def _counts(rulings: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in rulings:
        c[str(r["action"])] = c.get(str(r["action"]), 0) + 1
    c["total"] = len(rulings)
    return dict(sorted(c.items()))


def transcribe(checkout: str, sha: str, out_dir: str, round_id: str = "r2", version: str = "v1", python: Optional[str] = None,
               previous_roster: Optional[str] = None, previous_rulings: Optional[str] = None, read_utc: Optional[str] = None) -> Dict[str, object]:
    """The command: verify the pin, dump the symbols, build, write, and load what was written (a transcription that does not
    load is a refusal, never a file)."""
    head = checkout_sha(checkout)
    if not head.startswith(sha) or len(sha) < 7:
        raise TranscriptionRefusal(f"{checkout} is at {head}, not {sha}: the transcription names a commit; check out the pin first")
    cfg = dump_config_json(checkout, python)
    prev_roster = RO.load_yaml(previous_roster) if previous_roster else None
    prev_rulings = RO.read_rulings_jsonl(previous_rulings)[1] if previous_rulings else None
    prev_header = RO.read_rulings_jsonl(previous_rulings)[0] if previous_rulings else None
    roster = build_roster(cfg, head, round_id, read_utc, prev_roster)
    rulings = build_rulings(cfg, head, prev_rulings)
    ry, rj = write_transcription(out_dir, roster, rulings, round_id, version, prev_header)
    loaded = RO.load(ry, rj)
    return {"roster": ry, "rulings": rj, "sha": head, "entries": len(loaded.entries), "rulings_n": len(loaded.rulings),
            "counts_by_comparison": loaded.counts_by_comparison, "disagreements": len(loaded.disagreements), "columns_at_sha": cfg.get("COLUMNS")}
