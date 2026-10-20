"""ReLU's natively served steering lane -> registry.

Roots under ``run-state/full22/relu_native/``: the grades roots ``grades_relu_native_bu_<model>/<model>/two_si_<attack>_aggregate.json``
(per cell ``attack_metrics``: ``asr`` by threshold, ``n_jailbroken`` / ``n_goals``, ``per_goal`` best scores, ``completion.gated``;
grader sha in ``grading_provenance.json``) and ``two_si_bon_aggregate.json``; the done marker is the aggregate file plus the
merged pull gate (``completion.gated`` per cell). The PAIRED records (analysis truth, pin P8i)
``<run-state>/relu_native/paired/<model>__<eval>__<cell>.json`` (schema ``paired_vs_anchor/1``)
carry the contrast vs the anchor with the four pairing columns the export stamps — ``pairing_allowance``, ``anchor_engine``,
``anchor_view``, ``anchor_root`` — read here through ``sisteer.eval.anchor_pairing.PASSTHROUGH_ANCHOR_ALLOWANCE`` when the producing
lane's module is importable (never retyped), else the record's own string is carried and marked.

GSM8K ReLU-steer cards: the three declared-anchor (lens ``none``) rows are REFUSED with cause "cross-engine contrast not
admissible (vLLM O106)" — the record's own ``status = refused`` — and the three native-anchor rows
(``paired/llama8b__gsm8k_native_none__<cell>.json``, anchor lora_native) are the admissible readings. Every run emitted here is
natively served (``serving_path = lora_native``), object identity = the record's ``object_sha``; the anchor runs come from the
paired record's anchor block (the campaign's lens ``none`` grades root or the native ``none`` card).

THE WORKTREE DEPENDENCY IS RETIRED (2026-09-28). ``PAIRED_ROOT`` used to point into the worktree ``relu-native-P``
(branch ``relu-steering-native-evals``, detached at f224a7f8), where the 71 ``paired/<model>__<eval>__<cell>.json``
records were WORKING-TREE files -- only ``paired/.gitkeep`` was committed -- pinned by the P8i sha. That branch sits
1,622 commits behind main and was never going to land, so the records were instead copied byte-identical (sha256 per
file, all 71, the 6 GSM8K cards included) into the run-state tree, where every OTHER root this adapter reads already
lives: ``PAIRED_ROOT`` is now ``<RS>/paired`` and no path in this module points at a worktree.

What is deliberately KEPT from that arrangement: an absent root is still a ``missing`` status, never a crash
(:func:`paired_root_state`; ``status_extra(...)["missing"]`` names the path; every multi-turn row carries
``paired_record_status = missing``; the GSM8K cards, which exist in the paired records alone, are named as not read). The
guarantee was never about the worktree -- it is about reading a contrast that is not there -- so it outlives the move.
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import re
import sys
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .. import identity
from .base import AdapterContext, AdapterRefusal, Input, RecordBatch, SourceStateRow, check_input, store_glob, store_listdir, store_listdir_if_scoped
from . import declared_common as DC
from .declared_common import ARMS, EVAL_LABEL, HOME, MODELS, MULTITURN_ATTACKS, STATIC_FAMILIES, blank_row, cohen_h, finite, fnum, read_json, utc_ms, wilson

RS = os.path.join(DC.RUN_STATE, "relu_native")
#: the paired records now live in the run-state tree beside every other root this adapter reads
#: (2026-09-28). They were working-tree files in the UNLANDED worktree relu-native-P at pin
#: P8i bacc632c until then; that dependency is RETIRED and the worktree is gone. The 71 records
#: were copied byte-identical (sha256 per file) before the repoint, and the historical pin is
#: kept only so a reader of an older frame can place them.
PAIRED_ROOT = os.path.join(RS, "paired")
#: The paired records' PREVIOUS root -- the one the 2026-09-21 accepted heads recorded in their inputs_json: the worktree
#: `relu-native-P` (branch relu-steering-native-evals, pin P8i), which 3d2a7af97 (2026-09-28) moved to :data:`PAIRED_ROOT`; the
#: worktree is gone and every one of the 69 files under the new root is byte-identical to the head's (sha256 per file, re-verified 2026-10-06).
#: A rename is a RETIREMENT and is never inferred (runner.INPUTS_DROPPED_RULE): without this declaration the runner's inputs-dropped guard
#: held the four groups since 2026-09-29 (`inputs_dropped_unexplained` 64-88 % against a 10 % threshold), so the paper's relu_native rows
#: stayed frozen at the 09-21 head. Declared through :meth:`ReluNativeAdapter.inputs_retired`;
#: nothing is re-pointed, no symlink, no seed moves.
RETIRED_PAIRED_ROOT = os.path.join(HOME, "si-steering", "worktrees", "relu-native-P", "results", "relu_steering_native", "paired")
PAIRED_PIN = "bacc632c"        # P8i — the pin these records were TAKEN at; provenance, not a location
PAIRED_ROOT_DEPENDENCY = (f"the paired records live in the run-state lane at {PAIRED_ROOT}, beside the grades roots this "
                          "adapter already reads; no worktree, no unlanded branch, no working-tree files. Retired the "
                          f"relu-native-P dependency on 2026-09-28 (records copied byte-identical from pin P8i "
                          f"{PAIRED_PIN}); an absent root is still a `missing` status, never a crash")
SOURCE = "relu_native"
CELLS = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl_distilled", "constv2ctl2-full__psr_ce_all")
MODELS_RELU = ("llama8b", "gemma4e4b", "qwen35_4b", "nemotron3nano4b")
GSM8K_REFUSAL_CAUSE = "cross-engine contrast not admissible (vLLM O106)"
#: The GSM8K native cards' substrate when their record names no cluster marker (`assumed_default`, the basis filed): the ReLU native
#: GSM8K ladder was submitted to cluster A (the run-state README's `submit_relu.sh raw cluster_a g8k-ladder-plan-psr …`); the paired
#: record's `steered.root` is the grading host mirror under `gsm8k_native/` and carries no tag or shared-root path.
GSM8K_SUBSTRATE_DEFAULT = ("cluster_a", "the ReLU native GSM8K ladder was submitted to cluster A (run-state README: submit_relu.sh raw cluster_a "
                           "g8k-ladder-plan-psr); the paired record's steered.root is the grading host mirror and carries no cluster marker")


def substrate_of_root(root: Optional[str], what: str) -> Optional[Tuple[str, str]]:
    """The ``derived`` claim off a grades root's cluster TAG -- ``grade_relu.sh`` names every root ``grades_relu_native_<clu>_<model>``
    after the cluster the stores were pulled from (``bu`` / ``ch``; the pulled serving attestations of every model name the same
    root) -- else the root's own path on a shared root; None when the root carries neither (the anchor roots ``grades_anchor4096_*``)."""
    r = str(root or "")
    if not r:
        return None
    base = os.path.basename(r.rstrip("/")) or r
    if "_bu_" in base or base.endswith("_bu"):
        return "cluster_a", f"{what} {base} carries the cluster tag _bu_ (grade_relu.sh names the root after the cluster the stores were pulled from)"
    if "_ch_" in base or base.endswith("_ch"):
        return "cluster_b", f"{what} {base} carries the cluster tag _ch_ (grade_relu.sh names the root after the cluster the stores were pulled from)"
    hit = DC.D.substrate_root_of_path(r)
    if hit is not None and hit[0] != "grading_host":
        return hit[0], f"{what} {r[:72]} is on the {hit[1]}"
    return None
def passthrough_anchor_allowance() -> Tuple[Optional[str], str]:
    """(PASSTHROUGH_ANCHOR_ALLOWANCE, where it came from): the lane's own constant, else None.

    Until 2026-09-28 this fell back to scraping the constant out of the relu-native-P worktree's source when the module was not
    importable. That fallback was dead — ``sisteer.eval.anchor_pairing`` is in this tree and the first branch always won — and it
    was a second worktree dependency hiding behind the same blanket test exemption as PAIRED_ROOT, so it went with the worktree.
    """
    try:
        mod = importlib.import_module("sisteer.eval.anchor_pairing")
        return str(getattr(mod, "PASSTHROUGH_ANCHOR_ALLOWANCE")), "sisteer.eval.anchor_pairing (this tree)"
    except Exception:
        return None, "not importable"


def _model_of_relu(model_key: str) -> str:
    return MODELS[model_key]["model_id"]


def paired_root_state(paired_root: str = PAIRED_ROOT) -> Dict[str, object]:
    """Whether the paired records are on this host: ``present`` with the record count, or ``missing`` with the consequence spelled
    out. A missing tree is a STATUS, not a crash: the grades roots still read, without contrasts or GSM8K cards. On an unscoped host, which reads no store, the listing is skipped."""
    base = {"path": paired_root, "pin": PAIRED_PIN,
            "dependency": PAIRED_ROOT_DEPENDENCY}
    if os.path.isdir(paired_root):
        n = sum(1 for fn in store_listdir_if_scoped(paired_root) if fn.endswith(".json") and "__" in fn)
        return dict(base, status="present", n_records=n)
    return dict(base, status="missing",
                consequence=("no paired record on this host: the multi-turn and best-of-N rows file WITHOUT a contrast, no pairing stamp "
                             "and no object sha from the paired record; the GSM8K ReLU-steer cards (read from the paired records alone) "
                             "are not filed"))


class ReluNativeAdapter:
    name = "relu_native"
    #: the shared emitter's park change rides in (`paper_dose`) — see `declared_common.EMITTER_VERSION`
    EMISSION_VERSION = DC.emission_version(3)   # 2: the distilled cell's arm / label (difference_relu_distilled);
    #                                             3: the substrate off the grades root's cluster tag (derived); the anchors' unknown; the GSM8K
    #                                             cards' an assumed default with its basis -- never a `recorded` constant (2026-09-21)
    covers_sources: Tuple[str, ...] = ("relu_native",)
    rng_scope = "none"
    allow_symlinks = False
    campaign = "relu_native"

    def __init__(self, rs: str = RS, paired_root: str = PAIRED_ROOT):
        self.rs, self.paired_root = rs, paired_root
        self.roots = (rs, paired_root)
        self.last_report: Optional[DC.EmitReport] = None
        self.last_facts: Dict[str, object] = {}
        self.allowance, self.allowance_source = passthrough_anchor_allowance()
        self.paired_state = paired_root_state(paired_root)

    def discover(self) -> List[Input]:
        out = []
        for model in MODELS_RELU:
            g = os.path.join(self.rs, f"grades_relu_native_bu_{model}", model)
            if not os.path.isdir(g):
                continue
            for fn in sorted(store_listdir(g)):
                if fn.endswith("_aggregate.json") or fn in ("grading_provenance.json", "completion.json"):
                    out.append(check_input(os.path.join(g, fn), role="report", allow_symlink=self.allow_symlinks))
        if os.path.isdir(self.paired_root):
            for fn in sorted(store_listdir(self.paired_root)):
                if fn.endswith(".json") and "__" in fn:
                    out.append(check_input(os.path.join(self.paired_root, fn), role="report", allow_symlink=self.allow_symlinks))
        if not out:
            raise AdapterRefusal(self.rs, "no relu_native grades root or paired record found")
        return out

    def comparison_group(self, inp: Input) -> str:
        base = os.path.basename(inp.uri)
        if inp.uri.startswith(self.paired_root):
            return base.split("__")[0]
        m = re.search(r"grades_relu_native_bu_([a-z0-9_]+)/", inp.uri)
        return m.group(1) if m else "relu"

    def inputs_retired(self, group: str) -> List[str]:
        """The inputs a read of ``group`` (a model key) drops ON PURPOSE (the arditi_vendor_mt precedent): the group's paired records
        under :data:`RETIRED_PAIRED_ROOT`, the worktree root the 2026-09-21 heads recorded, since moved. The same files stand
        byte-identical under :data:`PAIRED_ROOT`, so the read drops nothing a reader could miss -- it renames; the runner's inputs-dropped
        guard reads this declaration (canonicalised through the pass's identity map) and counts them `inputs_retired`, never `inputs_dropped`."""
        return [os.path.join(RETIRED_PAIRED_ROOT, f"{group}__*.json")]

    # -- rows -------------------------------------------------------------------------------------
    def _paired(self, model: str, eval_id: str, cell: str) -> Optional[dict]:
        p = os.path.join(self.paired_root, f"{model}__{eval_id}__{cell}.json")
        return read_json(p) if os.path.exists(p) else None

    def long_rows(self, model: str) -> Tuple[List[dict], Dict[str, object]]:
        rows: List[dict] = []
        facts: Dict[str, object] = dict(model=model, refused=[], admitted=[], cross_engine=[], allowance_source=self.allowance_source,
                                        paired_root=dict(self.paired_state))
        if self.paired_state["status"] == "missing":
            facts["missing"] = [self.paired_root]
            facts["gsm8k_cards"] = f"not read: the paired record root is missing on this host ({PAIRED_ROOT_DEPENDENCY})"
        g = os.path.join(self.rs, f"grades_relu_native_bu_{model}", model)
        prov = read_json(os.path.join(g, "grading_provenance.json")) if os.path.exists(os.path.join(g, "grading_provenance.json")) else {}
        spec = MODELS[model]
        lane_kind, view = ("base→instruct" if spec.get("transfer_view") else "chat-ckpt"), (spec.get("transfer_view") or spec["view"])
        # multi-turn + bon aggregates
        for fn in sorted(store_glob(os.path.join(g, "two_si_*_aggregate.json"))):
            agg = read_json(fn)
            attack = str(agg.get("attack"))
            when = utc_ms(agg["written_at"]) if agg.get("written_at") else DC.mtime_utc(fn)
            engine = str(agg.get("serving_engine") or "")
            for cell, c in (agg.get("cells") or {}).items():
                if cell not in CELLS:
                    continue
                a = ARMS[cell]
                am = c.get("attack_metrics") or {}
                comp = (c.get("completion") or {}).get("gated") or {}
                paired = self._paired(model, attack, cell)
                sha = (paired or {}).get("steered", {}).get("object_sha") if paired else None
                sha = sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) else None
                if attack == "bon":
                    n_goals = int(am.get("n_goals") or (paired or {}).get("steered", {}).get("n") or 0)
                    for n_str, asr in (am.get("asr_at_n") or {}).items():
                        N = int(n_str)
                        k = round(float(asr) * n_goals)
                        lo, hi = wilson(k, n_goals) if n_goals else (float("nan"), float("nan"))
                        row = blank_row(source=SOURCE, model=model, lane_kind=lane_kind, view=view, dose=1.0, cell_id=cell, arm=a.key, family=a.family, object_type=a.object_type,
                                        component="bon", eval_id="bon", eval=EVAL_LABEL["bon"], metric="asr", threshold=0.5, bon_n=N, value=float(asr), k=float(k), n=n_goals,
                                        ci_lo=lo, ci_hi=hi, decode_cap_tokens=int(c.get("decode_cap_tokens") or 4096), decode="sampled T=1.0 (best-of-N), natively served", status="graded",
                                        judge="claude-haiku-4-5 / strongreject_v1", object_sha256=sha if sha else float("nan"), provenance=f"{fn} ›cells.{cell}.attack_metrics.asr_at_n[{N}]",
                                        thinking_mode=("not_available" if model == "llama8b" else "off"))
                        row.update(self._reg(model, cell, sha, when, engine, c, prov, fn, key=f"{SOURCE}|{model}|bon|{cell}|{N}", bon_n=N, harness="bespoke_sisteer",
                                             decode={"temperature": 1.0}, decode_source="grandfathered_T1", seed_rule="per_draw", cap=int(c.get("decode_cap_tokens") or 4096), n_planned=n_goals))
                        rows.append(row)
                    row = blank_row(source=SOURCE, model=model, lane_kind=lane_kind, view=view, dose=1.0, cell_id=cell, arm=a.key, family=a.family, object_type=a.object_type,
                                    component="bon", eval_id="bon", eval=EVAL_LABEL["bon"], metric="bon_nauc", threshold=0.5, bon_n=75, value=fnum(am.get("mean_normalized_auc")), n=n_goals,
                                    decode="sampled T=1.0 (best-of-N), natively served", status="graded", judge="claude-haiku-4-5 / strongreject_v1", object_sha256=sha if sha else float("nan"),
                                    provenance=f"{fn} ›cells.{cell}.attack_metrics.mean_normalized_auc", thinking_mode=("not_available" if model == "llama8b" else "off"))
                    row.update(self._reg(model, cell, sha, when, engine, c, prov, fn, key=f"{SOURCE}|{model}|bon|{cell}|nauc", bon_n=75, harness="bespoke_sisteer",
                                         decode={"temperature": 1.0}, decode_source="grandfathered_T1", seed_rule="per_draw", cap=int(c.get("decode_cap_tokens") or 4096), n_planned=n_goals))
                    if finite(row["value"]):
                        rows.append(row)
                    continue
                asr = fnum((am.get("asr") or {}).get("0.5"))
                k, n = fnum(am.get("n_jailbroken")), fnum(am.get("n_goals"))
                lo, hi = wilson(k, n) if finite(k) and finite(n) and n else (float("nan"), float("nan"))
                per_goal = am.get("per_goal") or {}
                capped = int(am.get("n_capped") or 0)
                row = blank_row(source=SOURCE, model=model, lane_kind=lane_kind, view=view, dose=1.0, cell_id=cell, arm=a.key, family=a.family, object_type=a.object_type,
                                component=DC.EVAL_COMPONENT.get(attack, "multi-turn"), eval_id=attack, eval=EVAL_LABEL.get(attack, attack), metric="asr", threshold=0.5, bon_n=float("nan"),
                                value=asr, k=k, n=n, ci_lo=lo, ci_hi=hi, at_cap_share=(capped / n if finite(n) and n else float("nan")), decode_cap_tokens=int(c.get("decode_cap_tokens") or 4096),
                                decode=f"sampled at the target's shipped generation config, natively served ({engine})", status="graded", judge="claude-haiku-4-5 / strongreject_v1",
                                object_sha256=sha if sha else float("nan"), provenance=f"{fn} ›cells.{cell}.attack_metrics", thinking_mode=("not_available" if model == "llama8b" else "off"),
                                note=f"{attack}: ASR {asr:.3f} at StrongREJECT > 0.5 over {int(n) if finite(n) else '?'} goals; natively served ({engine})")
                row.update(self._reg(model, cell, sha, when, engine, c, prov, fn, key=f"{SOURCE}|{model}|{attack}|{cell}", harness=("actorattack" if attack == "actorattack" else "pyrit"),
                                     decode=(paired or {}).get("steered", {}).get("store_decode", {}).get("sampler_normalised") if paired else None, decode_source="generation_config",
                                     seed_rule="none", cap=int(c.get("decode_cap_tokens") or 4096), n_planned=int(n) if finite(n) else None, max_turns=10,
                                     attacker=str(((comp.get("attacker_engine") or {}).get("attacker_model")) or "")))
                row["_degeneration"] = {"unit": "items", "counts": {"at_token_cap": (capped, int(n))} if finite(n) else {}}
                row["_items"] = [{"item_id": gid, "metric_id": "strongreject_best", "value": float(pg.get("best_score")), "flags": {"at_token_cap": int(bool(pg.get("capped")))}}
                                 for gid, pg in sorted(per_goal.items()) if pg.get("best_score") is not None]
                self._attach_contrast(row, paired, model, attack, cell, facts, rows)
                rows.append(row)
        # GSM8K cards: the paired records ARE the readings (the native card stores are the per-item adapter's)
        for cell in CELLS:
            for ev in ("gsm8k", "gsm8k_native_none"):
                p = self._paired(model, ev, cell)
                if p is None:
                    continue
                self._gsm8k_rows(model, ev, cell, p, lane_kind, view, rows, facts)
        return rows, facts

    def _reg(self, model: str, cell: str, sha: Optional[str], when: str, engine: str, c: Mapping[str, object], prov: Mapping[str, object], fn: str, *, key: str,
             harness: str, decode, decode_source: str, seed_rule: str, cap: int, n_planned: Optional[int], bon_n: Optional[int] = None, max_turns: Optional[int] = None,
             attacker: Optional[str] = None) -> dict:
        sp, _c, _s = DC.serving_path_from_identity(str(c.get("serving_identity") or engine))
        grader_sha_legacy = str(((prov.get("grader") or {}).get("code_sha")) or "")
        return dict(_when=when, _code_sha="unverified", _code_sha_reason="the aggregate names the grader's provenance, not the producer's code sha", _object_id=sha,
                    _substrate_derived=substrate_of_root(os.path.dirname(os.path.dirname(fn)), "the grades root"), _substrate_path="",
                    _substrate_reason="the grades root carries no cluster tag", _producer_key=f"grades_relu_native_bu_{model}", _object_key=cell, _store_uri=fn, _key=key,
                    _instrument={"harness": harness, "harness_version": "unverified", "token_cap_tokens": cap, "decode": {k: v for k, v in (decode or {}).items() if k in DC.D.DECODE_KEYS and v is not None},
                                 "decode_source": decode_source, "decode_seed_rule": seed_rule, "thinking_mode": ("not_available" if model == "llama8b" else "off"), "serving_path": sp,
                                 "serving_identity": c.get("serving_identity"), "bon_n": bon_n, "max_turns": max_turns,
                                 "attacker_model_id": ("qwen36_27b_abliterated_attacker" if attacker and "Qwen3.6-27B-abliterated" in attacker else None),
                                 "instrument_config_hash": f"relu_native_{harness}_{cap}", "fit_positions": "all", "standup_mode": None},
                    _judge=DC.judge_fields(**DC.STRONGREJECT_JUDGE), _n_items_planned=n_planned, _strength_coefficient=fnum(c.get("served_alpha")) if finite(fnum(c.get("served_alpha"))) else 1.0,
                    _is_operating_point=1, _degeneration={"unit": "items"},
                    _extra={"serving_engine": engine, "serving_identity": c.get("serving_identity"), "si_id": c.get("si_id"), "training_pool": c.get("training_pool"), "cell_class": c.get("class")})

    def _attach_contrast(self, row: dict, paired: Optional[dict], model: str, eval_id: str, cell: str, facts: dict, rows: List[dict]) -> None:
        if not paired:
            if self.paired_state["status"] == "missing":
                # the dependency is named on the row, so a contrast-less reading is never mistaken for a producer that filed none
                row["_extra"] = dict(row.get("_extra") or {}, paired_record_status="missing", paired_record_dependency=PAIRED_ROOT_DEPENDENCY)
            return
        stamp = self._stamp(paired)
        row["_extra"] = dict(row.get("_extra") or {}, **stamp, paired_record_pin=PAIRED_PIN)
        if paired.get("status") != "ok" or not paired.get("stats"):
            facts["refused"].append(f"{model}/{eval_id}/{cell}: {'; '.join(paired.get('refusals') or [])[:200]}")
            return
        st = paired["stats"]
        anchor = paired.get("anchor") or {}
        anchor_row = self._anchor_row(model, eval_id, anchor, paired, row)
        if anchor_row is not None:
            rows.append(anchor_row)
            row.update(delta=float(st["delta"]), delta_lo=float(st["ci95"][0]), delta_hi=float(st["ci95"][1]), delta_p=fnum(st.get("p_value")),
                       delta_method=f"{(st.get('bootstrap') or {}).get('kind')}; {st.get('p_method')}", _anchor_key=anchor_row["_key"],
                       _anchor_kind=("same_standup_none" if stamp.get("anchor_engine") == stamp.get("steered_engine") else "borrowed"), _n_pairs=int(st.get("n_pairs") or 0))
            facts["admitted"].append(f"{model}/{eval_id}/{cell}")
            if stamp.get("pairing_allowance"):
                facts["cross_engine"].append(f"{model}/{eval_id}/{cell}")

    def _stamp(self, paired: dict) -> dict:
        rec_allow = paired.get("pairing_allowance")
        allow = self.allowance if (rec_allow and self.allowance and rec_allow == self.allowance) else rec_allow
        return {"pairing_allowance": allow, "pairing_allowance_source": (self.allowance_source if allow == self.allowance and self.allowance else "the paired record's own string (lane constant not importable)"),
                "anchor_engine": paired.get("anchor_engine"), "anchor_view": paired.get("anchor_view"), "anchor_root": paired.get("anchor_root"),
                "steered_engine": paired.get("steered_engine"), "pairing_admitted": paired.get("pairing_admitted")}

    def _anchor_row(self, model: str, eval_id: str, anchor: dict, paired: dict, steered: dict) -> Optional[dict]:
        val = anchor.get("value")
        if val is None:
            return None
        n = int(anchor.get("n") or 0)
        v = float(val) / 100.0 if eval_id not in ("gsm8k", "gsm8k_native_none") else float(val) / 100.0
        key = f"{SOURCE}|anchor|{model}|{eval_id}|{paired.get('anchor_root')}"
        row = blank_row(source=SOURCE, model=model, lane_kind=steered["lane_kind"], view=steered["view"], dose=1.0, cell_id="none", arm="none", family="none", object_type="none",
                        component=steered["component"], eval_id=steered["eval_id"], eval=steered["eval"], metric=steered["metric"], threshold=steered.get("threshold"), bon_n=float("nan"),
                        value=v, n=n, decode_cap_tokens=fnum(anchor.get("cap") or anchor.get("decode_cap_tokens")), status="graded", judge=steered.get("judge"),
                        decode=f"anchor served by {paired.get('anchor_engine')} ({anchor.get('serving_identity')})", thinking_mode=steered.get("thinking_mode"),
                        provenance=f"{anchor.get('aggregate') or anchor.get('root')}; anchor block of the paired record", note=f"the anchor of the {steered['cell_id']} contrast (anchor_view {paired.get('anchor_view')})")
        if steered["metric"] == "asr":
            k = round(v * n)
            lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
            row.update(k=float(k), ci_lo=lo, ci_hi=hi)
        else:
            k = ((paired.get("stats") or {}).get("n_correct_by_leg") or {}).get("none")
            row.update(k=fnum(k))
        sp, _c, _s = DC.serving_path_from_identity(str(anchor.get("serving_identity") or paired.get("anchor_engine") or ""))
        inst = dict(steered["_instrument"], serving_path=sp, serving_identity=anchor.get("serving_identity"), attacker_model_id=steered["_instrument"].get("attacker_model_id"))
        agg, root = anchor.get("aggregate"), anchor.get("root")
        # the anchor's own record dates the anchor run (several paired records borrow one anchor; each paired record's mtime is not its time):
        # the aggregate file where the anchor block names one, else the anchor root directory itself
        if isinstance(agg, str) and os.path.isfile(agg):
            when = DC.mtime_utc(agg)
        elif isinstance(root, str) and os.path.isdir(root):
            when = DC.mtime_utc(root)
        else:
            when = steered["_when"]
        # the anchor's substrate: its grades root's cluster tag where it carries one; the campaign's `grades_anchor4096_*` roots carry
        # none and no shared-root path, so their substrate is UNKNOWN (never the steered root's constant)
        row.update(_when=when, _code_sha="unverified", _code_sha_reason="the paired record names the anchor's grades root, not its code sha", _object_id=None,
                   _substrate_derived=substrate_of_root(anchor.get("root") or paired.get("anchor_root"), "the anchor's grades root"), _substrate_path="",
                   _substrate_reason=f"the paired record's anchor root {str(anchor.get('root') or paired.get('anchor_root') or '')[:80]!r} carries no cluster tag and no "
                                     "shared-root path; the anchor's generating substrate is not on the record",
                   _producer_key=str(anchor.get("root") or paired.get("anchor_root") or "anchor"), _object_key="none",
                   _store_uri=anchor.get("aggregate") or anchor.get("root"), _key=key, _instrument=inst, _judge=steered.get("_judge"), _n_items_planned=n or None,
                   _degeneration={"unit": "items"}, _extra={"anchor_of": steered["cell_id"], "anchor_engine": paired.get("anchor_engine"), "anchor_view": paired.get("anchor_view")})
        return row

    def _gsm8k_rows(self, model: str, ev: str, cell: str, p: dict, lane_kind: str, view: str, rows: List[dict], facts: dict) -> None:
        a = ARMS[cell]
        st = p.get("steered") or {}
        sha = st.get("object_sha") if isinstance(st.get("object_sha"), str) and re.fullmatch(r"[0-9a-f]{64}", str(st.get("object_sha"))) else None
        when = utc_ms(p["written_at"]) if p.get("written_at") else DC.mtime_utc(os.path.join(self.paired_root, f"{model}__{ev}__{cell}.json"))
        stamp = self._stamp(p)
        status_ok = p.get("status") == "ok" and bool(p.get("stats"))
        stats = p.get("stats") or {}
        val = st.get("value")
        row = blank_row(source=SOURCE, model=model, lane_kind=lane_kind, view=view, dose=1.0, cell_id=cell, arm=a.key, family=a.family, object_type=a.object_type,
                        component="capability", eval_id="gsm8k", eval=EVAL_LABEL["gsm8k"], metric="acc", threshold=float("nan"), bon_n=float("nan"),
                        value=(float(val) / 100.0 if val is not None else float("nan")), n=int(st.get("n") or 200) if st.get("n") else float("nan"),
                        k=fnum(((stats.get("n_correct_by_leg") or {}).get("cell"))), decode_cap_tokens=8192, decode="thinking not available (llama), k=1, cap 8192 (+16384 rung); natively served",
                        judge="inspect_evals gsm8k scorer (10-shot, 'ANSWER: N' match)", thinking_mode="not_available", object_sha256=sha if sha else float("nan"),
                        provenance=f"{self.paired_root}/{model}__{ev}__{cell}.json (pin {PAIRED_PIN}); steered.leg {((st.get('leg') or {}).get('leg'))}",
                        status=("graded" if status_ok else f"refused: {GSM8K_REFUSAL_CAUSE} — {'; '.join(p.get('refusals') or [])[:300]}"),
                        note=(f"GSM8K ReLU-steer card vs the {p.get('anchor_engine')} anchor ({ev})" if status_ok else f"{GSM8K_REFUSAL_CAUSE}: the declared lens `none` anchor cannot be paired with a natively served steered card (D30)"))
        if not status_ok:
            row.update(value=float("nan"))
            facts["refused"].append(f"{model}/{ev}/{cell}: {GSM8K_REFUSAL_CAUSE}")
        row.update(_when=when, _code_sha=(str(((st.get("leg") or {}).get("producer") or {}).get("tree_sha")) if len(str(((st.get("leg") or {}).get("producer") or {}).get("tree_sha") or "")) == 40 else "unverified"),
                   _object_id=sha, _substrate_derived=substrate_of_root(st.get("root"), "the card's root"), _substrate_default=GSM8K_SUBSTRATE_DEFAULT, _substrate_path="",
                   _producer_key=str(((st.get("leg") or {}).get("leg")) or f"{st.get('root')}|{cell}"), _object_key=cell,
                   _store_uri=st.get("root"), _key=f"{SOURCE}|{model}|{ev}|{cell}", _model_revision=None,
                   _instrument={"harness": "inspect_ai", "harness_version": "unverified", "token_cap_tokens": 8192, "ladder": (8192, 16384), "read_at": "ladder_final", "decode": {},
                                "decode_source": "generation_config", "decode_seed_rule": "per_item", "thinking_mode": "not_available", "serving_path": "lora_native", "serving_identity": st.get("serving_identity"),
                                "n_shot": 10, "cot_prompting": "few_shot_cot", "scoring_rule": "numeric_match", "instrument_config_hash": "gsm8k_paired200_native_llama", "fit_positions": "all"},
                   _confidence={"decode_source": ("derived", "the llama card row: the shipped generation config (paired record: resolved_sampler null on the native card)")},
                   _judge=None, _n_items_planned=200, _degeneration={"unit": "items"}, _strength_coefficient=1.0, _is_operating_point=1,
                   _extra=dict(stamp, paired_record_pin=PAIRED_PIN, refusals=p.get("refusals")))
        if status_ok:
            anchor_row = self._anchor_row(model, ev, p.get("anchor") or {}, p, row)
            if anchor_row is not None:
                anchor_row.update(_instrument=dict(row["_instrument"], serving_path=DC.serving_path_from_identity(str((p.get("anchor") or {}).get("serving_identity") or ""))[0]))
                rows.append(anchor_row)
                row.update(delta=float(stats["delta"]), delta_lo=float(stats["ci95"][0]), delta_hi=float(stats["ci95"][1]), delta_p=fnum(stats.get("p_value")),
                           delta_method=str((stats.get("bootstrap") or {}).get("kind") or "paired bootstrap"), _anchor_key=anchor_row["_key"], _anchor_kind="same_standup_none",
                           _n_pairs=int(stats.get("n_pairs") or 200))
            facts["admitted"].append(f"{model}/{ev}/{cell}")
        rows.append(row)

    def read(self, ctx: AdapterContext, group: str, inputs: Sequence[Input]) -> Iterator[RecordBatch]:
        if group not in MODELS_RELU:
            return
        rows, facts = self.long_rows(group)
        self.last_facts[group] = facts
        batch = RecordBatch(f"{self.name}:{group}", group)
        rep = DC.emit_rows(batch, ctx, rows, adapter=self.name, campaign=self.campaign, reader="relu_native.paired_vs_anchor",
                           seed_provenance="producer statistics read from the paired records (B=10000 seed 1), none recomputed", code_sha="unverified",
                           code_sha_reason="the grades aggregates name the grader provenance only")
        self.last_report = rep
        yield batch

    def status_extra(self, group: str) -> Dict[str, object]:
        """The facts of the last read of ``group`` plus the paired-record dependency: ``paired_root`` (present / missing, the
        worktree, branch and pin) and, when the tree is absent, ``missing`` naming its path -- the paper's gate reads `missing` by name."""
        out: Dict[str, object] = dict(self.last_facts.get(group) or {})
        out.setdefault("paired_root", dict(self.paired_state))
        if self.paired_state["status"] == "missing":
            out.setdefault("missing", [self.paired_root])
        if self.last_report is not None:
            out["row_refusals"] = list(self.last_report.refusals)
        return out

    def source_states(self, ctx: AdapterContext, inputs: Sequence[Input]) -> List[SourceStateRow]:
        out = []
        for i in inputs:
            if i.uri.endswith("_aggregate.json"):
                d = read_json(i.path)
                out.append(SourceStateRow(f"{self.name}:{self.comparison_group(i)}", i.uri, "$.cells", 1, 1, sorted((d.get("cells") or {}).keys()), i.sha256))
        return out


def register() -> None:
    from .base import register as _register
    _register(ReluNativeAdapter.name, lambda: ReluNativeAdapter())
