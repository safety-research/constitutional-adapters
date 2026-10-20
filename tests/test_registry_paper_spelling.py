"""The paper view's IDENTITY SPELLINGS reproduce the readers' (the parity gate: every legacy-adapted source broke on key spelling with
equal values). One fixture row per fixed class, rendered through `paper.render_long_row` — no database — against the
reader's own spelling (the reader literal, parsed from the pinned paper checkout where it is present on this machine, else the string
the reader files as recorded in the parity report). A guarded summary test re-renders the live registry read-only and reports the
per-source key matches against the FINAL run's pin keys (counts, never a total assertion).
"""
from __future__ import annotations

import ast
import json
import math
import os
import sqlite3

import pytest

from sisteer.registry import paper as P
from sisteer.registry.adapters import declared_common as DC

pytestmark = pytest.mark.live_db   # this module opens the live registry: land.sh's local step runs it (`pytest -m live_db`)

KEY = ("source", "model", "lane_kind", "arm", "eval_id", "metric", "threshold", "decode_cap_tokens", "judge", "dose")
PIN_PAPER = os.path.join(DC.MAIN_CHECKOUT, "worktrees", "paper-figures-pin-c53ae173", "paper", "paperlib")   # the pinned paper checkout
PIN_L1 = os.path.expanduser(os.environ.get("REGISTRY_PIN_L1", ""))   # the assembled level-1 pin of the parity work (on Slurm: <replica root>/pins/); unset skips


def _row(**kw):
    base = dict(lane_source_legacy="sampled_lane", view_legacy="gemma4e4b", model_id="gemma4e4b_it", transfer_kind="base_to_instruct",
                paper_label="other", object_role="constitution", object_kind="lora", campaign_cell_id_legacy="constv2-mix-full__lora_r4_ce",
                eval_id="pasttense", eval_category="static_attack", metric_id="asr", producer_metric_spelling="attack_metrics.asr[0.5]",
                threshold=0.5, token_cap_tokens=8192, judge_model="claude-haiku-4-5", judge_rubric_name="strongreject", strength_nominal=None,
                value=0.3, k=78, n=259, run_extra_json="{}", metric_extra_json="{}")
    base.update(kw)
    return base


def _render(**kw):
    return P.render_long_row(_row(**kw), list(KEY) + ["family", "component"])


def _reader_literal(module: str, name: str):
    """A module-level string constant of a pinned reader, read by AST (no paperlib import: the paper venv is not this one)."""
    path = os.path.join(PIN_PAPER, module)
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


# ---- arm: `other` objects are spelled from the campaign cell (config.ARM_BY_CELL in every reader) ---------------------------------
def test_arm_of_an_other_labelled_object_is_the_cell_s_paper_key():
    out = _render()
    assert out["arm"] == "const_lora" and out["family"] == "constitution"
    assert _render(campaign_cell_id_legacy="constv2ctl2-full__lora_r4_ce", object_role="control")["arm"] == "ctl_lora"
    # across all five families: the difference_relu arm (diff_psr) is the UNDISTILLED paired write; the residual-distilled
    # student renders its own key, diff_psr_distilled, and is no paper arm
    out_dist = _render(campaign_cell_id_legacy="constv2-mix-full__psr_ce_all_minus_ctl_distilled", object_kind="relu_steer_distilled_difference")
    assert out_dist["arm"] == "diff_psr_distilled" and out_dist["family"] == "difference"
    assert _render(paper_label="difference_relu_distilled", campaign_cell_id_legacy="constv2-mix-full__psr_ce_all_minus_ctl_distilled",
                   object_kind="relu_steer_distilled_difference")["arm"] == "diff_psr_distilled"
    out_pw = _render(campaign_cell_id_legacy="constv2-mix-full__psr_ce_all_minus_ctl", object_kind="composed", object_role="difference")
    assert out_pw["arm"] == "diff_psr" and out_pw["family"] == "difference"
    # an object that carries its own label keeps it
    assert _render(paper_label="constitution_si", campaign_cell_id_legacy="prompted__si_constitution")["arm"] == "si_constitution"
    # THE OBJECT'S LABEL WINS OVER A PRODUCER COLUMN (the paper side, 2026-09-20: four bcb_bfcl_200_widen rows carried the producer's
    # `diff_psr` on the retired distilled student's cell). A parked reader word that disagrees rides the row as `producer_arm_legacy`.
    out = _render(run_extra_json=json.dumps({"paper_arm": "arditi_allpos"}))
    assert out["arm"] == "const_lora" and out["producer_arm_legacy"] == "arditi_allpos"
    # the parked word is read only where the object carries NO paper label (no own label, no stem any table spells)
    out = _render(paper_label="other", object_kind="vector_all", object_role="baseline_published", campaign_cell_id_legacy="ar_c0p3__arditi_allpos",
                  run_extra_json=json.dumps({"paper_arm": "arditi_allpos", "paper_family": "baseline"}))
    assert out["arm"] == "arditi_allpos" and out["family"] == "baseline" and out["producer_arm_legacy"] is None
    # the two cases the paper side asked for: a producer `diff_psr` on the DISTILLED student spells the student's own key, and a paired-write
    # `composed` object under a producer `other` spells diff_psr — the object decides, the producer's word is the fact
    out = _render(paper_label="difference_relu_distilled", object_kind="relu_steer_distilled_difference", object_role="difference",
                  campaign_cell_id_legacy="constv2-mix-full__psr_ce_all_minus_ctl_distilled", lane_source_legacy="bcb_bfcl_200_widen",
                  run_extra_json=json.dumps({"paper_arm": "diff_psr", "paper_family": "difference"}))
    assert out["arm"] == "diff_psr_distilled" and out["family"] == "difference" and out["producer_arm_legacy"] == "diff_psr"
    out = _render(paper_label="difference_relu", object_kind="composed", object_role="difference",
                  campaign_cell_id_legacy="d1_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl", run_extra_json=json.dumps({"paper_arm": "other"}))
    assert out["arm"] == "diff_psr" and out["producer_arm_legacy"] == "other"
    # the same two objects labelled `other` by a stale read still spell from the stem, over the producer's word
    out = _render(paper_label="other", object_kind="relu_steer_distilled_difference", object_role="difference",
                  campaign_cell_id_legacy="derived__constv2-mix-full__psr_ce_all_minus_ctl_distilled", run_extra_json=json.dumps({"paper_arm": "diff_psr"}))
    assert out["arm"] == "diff_psr_distilled" and out["producer_arm_legacy"] == "diff_psr"
    # agreement parks no fact
    assert _render(paper_label="constitution_lora", run_extra_json=json.dumps({"paper_arm": "const_lora"}))["producer_arm_legacy"] is None
    # a run served under a STAGED spelling (the fit variant before the stem) spells the stem's arm: the prefix is fit provenance, never the arm
    assert _render(campaign_cell_id_legacy="itdoc64__constv2ctl2-full__lora_r4_ce", object_role="control")["arm"] == "ctl_lora"
    assert _render(campaign_cell_id_legacy="d0p5_eb64s1__constv2-mix-full__lora_r4_ce_minus_ctl", object_kind="lora_subtraction",
                   object_role="difference")["arm"] == "diff_lora"
    assert _render(campaign_cell_id_legacy="derived__constv2-mix-full__psr_ce_all_minus_ctl", object_kind="composed", object_role="difference")["arm"] == "diff_psr"
    assert _render(campaign_cell_id_legacy="si12__safety_refusal__psr_kl_all", object_kind="relu_steer", object_role="safety")["arm"] == "safety_psr"
    # the objects adapter now labels these objects itself; a labelled object renders through PAPER_LABEL_LEGACY whatever its cell spelling
    assert _render(paper_label="control_relu", campaign_cell_id_legacy="base__constv2ctl2-full__psr_ce_all", object_kind="relu_steer",
                   object_role="control")["arm"] == "ctl_psr"


# ---- lane_kind: the CELL's lane on unsteered / prompted rows, from the view --------------------------------------------------------
def test_unsteered_and_prompted_rows_take_the_view_s_lane():
    assert _render(transfer_kind="none", object_kind="none", paper_label="none", campaign_cell_id_legacy="none")["lane_kind"] == "base→instruct"
    assert _render(transfer_kind="prompted_only", object_kind="system_instruction", view_legacy="gemma4e4b_it")["lane_kind"] == "instruct-direct"
    assert _render(transfer_kind="none", object_kind="none", view_legacy="nemotron3nano4b_d0p25")["lane_kind"] == "chat-ckpt"
    assert _render(transfer_kind="none", object_kind="none", view_legacy="llama8b_dm1")["lane_kind"] == "base→instruct"
    # the export's own lane rides its rows (fullconst stays prompted-only); its shortconst view is re-stamped to llama's paper lane
    r2 = dict(lane_source_legacy="r2_export", transfer_kind="prompted_only", object_kind="system_instruction")
    assert _render(view_legacy="llama8b_fullconst", run_extra_json=json.dumps({"lane_kind": "prompted-only"}), **r2)["lane_kind"] == "prompted-only"
    assert _render(view_legacy="llama8b_shortconst", run_extra_json=json.dumps({"lane_kind": "prompted-only"}), **r2)["lane_kind"] == "instruct-direct"
    # a steered row keeps the object's transfer kind; the declared readers' parked lane wins
    assert _render(transfer_kind="instruct_direct")["lane_kind"] == "instruct-direct"
    assert _render(run_extra_json=json.dumps({"paper_lane_kind": "chat-ckpt"}))["lane_kind"] == "chat-ckpt"


# ---- judge: the reader's string, not the judge model alone --------------------------------------------------------------------
def test_judge_is_the_reader_s_spelling():
    assert _render()["judge"] == "claude-haiku-4-5 / strongreject_v1"
    parked = "auditor claude-sonnet-5, judge claude-opus-5 (Petri)"
    assert _render(run_extra_json=json.dumps({"judge_legacy": parked}), judge_model="claude-opus-5")["judge"] == parked
    ctl = _render(lane_source_legacy="sampled_lane_controls", eval_category="over_refusal", eval_id="xstest", metric_id="over_refusal_graded",
                  judge_model="claude-sonnet-5", judge_rubric_name="xstest_cpi")["judge"]
    expected = _reader_literal("or_sampled.py", "JUDGE") or P.SAMPLED_OVER_REFUSAL_JUDGE_LEGACY
    assert ctl == expected
    # a source with no spelling rule renders the judge model alone (unchanged behaviour)
    assert _render(lane_source_legacy="multiturn_grades")["judge"] == "claude-haiku-4-5"


# ---- metric: the registry id, never the producer's spelling; best-of-N curve ids fold into `asr` -------------------------------
def test_metric_is_the_registry_id_spelled_for_the_paper():
    assert _render()["metric"] == "asr"
    assert _render(metric_id="asr_at_n16", producer_metric_spelling="asr")["metric"] == "asr"
    assert _render(metric_id="asr_held_in", producer_metric_spelling="by_category_group.held_in")["metric"] == "asr_held_in"
    assert _render(metric_id="asr_bon_cap4096", producer_metric_spelling="asr_4096_cf_t0.5")["metric"] == "asr_bon_cap4096"


# ---- decode_cap_tokens: None where the reader files none ----------------------------------------------------------------------
def test_decode_cap_is_none_where_the_reader_files_none():
    assert _render(eval_id="mmlu", eval_category="capability", metric_id="acc", token_cap_tokens=1)["decode_cap_tokens"] is None
    assert _render()["decode_cap_tokens"] == 8192
    assert _render(lane_source_legacy="r2_export", eval_category="petri", metric_id="petri_score", token_cap_tokens=1024)["decode_cap_tokens"] is None
    assert _render(lane_source_legacy="r2_export", eval_category="petri", metric_id="at_cap_share", token_cap_tokens=1024)["decode_cap_tokens"] == 1024
    assert _render(lane_source_legacy="em/agentic/qwen35_4b", eval_category="agentic", metric_id="misaligned_rate", token_cap_tokens=16384)["decode_cap_tokens"] is None


# ---- dose: the view's dose where the run records none -------------------------------------------------------------------------
def test_dose_is_the_view_s_dose_where_the_run_records_none():
    assert _render()["dose"] == 1.0                                                     # the sampled lane: every row at 1.0
    assert _render(view_legacy="gemma4e4b_it_d0p25")["dose"] == 0.25
    assert _render(view_legacy="llama8b_dm1")["dose"] == -1.0
    assert _render(lane_source_legacy="r2_export", view_legacy="qwen35_4b_halfdose", object_kind="none", paper_label="none",
                   campaign_cell_id_legacy="none", transfer_kind="none")["dose"] == 0.5
    assert _render(strength_nominal=0.75)["dose"] == 0.75                                # a recorded strength is never overwritten
    assert _render(lane_source_legacy="strength_lane", run_extra_json=json.dumps({"paper_dose": 1.5}))["dose"] == 1.5
    assert _render(lane_source_legacy="strength_lane")["dose"] is None                   # a steered run with no strength: unknown, not 1.0
    # the all-positions refusal steer parks its RAW coefficient c as the paper word: the reader's own word renders unchanged,
    # ahead of the nominal the registry files, so the paper's own re-key (load.rekey_arditi_allpos_nominal) receives c
    assert _render(lane_source_legacy="arditi_allpos", strength_nominal=1.0, object_kind="vector_all",
                   run_extra_json=json.dumps({"paper_dose": 0.3}))["dose"] == 0.3
    assert _render(lane_source_legacy="arditi_allpos", strength_nominal=None, object_kind="none", paper_label="none",
                   campaign_cell_id_legacy="none", transfer_kind="none", run_extra_json=json.dumps({"paper_dose": 0.3}))["dose"] == 0.3


def test_served_key_dose_suffixes():
    assert P._served_key_dose("gemma4e4b") == 1.0 and P._served_key_dose("nemotron3nano4b_d1p5") == 1.5
    assert P._served_key_dose("qwen35_4b_d1p25_loradiff64s1") == 1.25 and P._served_key_dose("qwen35_4b_dm0p5_distdiff64s1_t0") == -0.5
    # an unsteered row beside a seeded dose standup carries the dose it was served beside, the way a `_halfdose` one does
    assert _render(transfer_kind="none", object_kind="none", view_legacy="qwen35_4b_d1p25_loradiff64s1")["dose"] == 1.25
    assert P._served_key_base("nemotron3nano4b_d1p5") == "nemotron3nano4b" and P._served_key_base("gemma4e4b_halfdose") == "gemma4e4b"
    assert P._served_key_dose(None) is None


# ---- the live registry against the FINAL run's pin keys (read-only; skipped where either is absent) ------------------------------
@pytest.mark.skipif(not os.path.exists(PIN_L1), reason="the c53ae173 parity dump is not on this machine")
def test_live_rendering_matches_the_pin_keys_per_source(live_paper_long):
    def norm(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "NA"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    def kstr(r):
        return "|".join(norm(r.get(k)) for k in KEY)
    pin = [json.loads(l) for l in open(PIN_L1, encoding="utf-8") if l.strip()]
    pin_keys = {}
    for r in pin:
        if r.get("value") is not None:
            pin_keys.setdefault(r["source"], set()).add(kstr(r))
    cols = [c for c in KEY if c in P.LONG_COLUMN_SOURCES]
    reg_keys = {}
    excluded = {}
    # `WHERE value IS NOT NULL` of the former query, applied in Python over the one shared scan (the session fixture of
    # tests/registry_live_fixtures.py, registered by conftest)
    for row in live_paper_long.rows:
        if row["value"] is None:
            continue
        row = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
        why = P.paper_frame_exclusion(row)
        if why is not None:                       # a registry-only row (no reader source / a per-trial pass): not a paper row
            excluded[why.split(":")[0]] = excluded.get(why.split(":")[0], 0) + 1
            continue
        out = P.render_long_row(row, cols)
        reg_keys.setdefault(str(out.get("source")), set()).add(kstr(out))
    print("\nregistry-only rows left out of the paper frame:", excluded)
    matched = {s: len(ks & reg_keys.get(s, set())) for s, ks in pin_keys.items()}
    print("\nper-source pin keys matched by the live rendering:", {s: f"{matched[s]}/{len(ks)}" for s, ks in sorted(pin_keys.items())})
    # the fixed classes: the declared readers' words and the cell / view / judge spellings reach the frame
    for s in ("llama8b_lane", "strength_lane", "bcb_bfcl_200", "evalcheck/gsm8k_paired200", "sampled_lane_controls", "r2_export"):
        assert matched.get(s, 0) > 0, (s, matched.get(s))


# ---- decode_cap_tokens: the export's over-refusal RATE rows file no cap (their sampled-ladder `_full` rows keep theirs) ------------
def test_export_over_refusal_rate_rows_file_no_cap():
    r2 = dict(lane_source_legacy="r2_export", eval_category="over_refusal", eval_id="xstest", token_cap_tokens=512, judge_model="claude-sonnet-5")
    for mid in ("over_refusal_rate", "over_refusal_rate_terminated", "over_refusal_delta_allitems", "over_refusal_rate_unpaired"):
        assert _render(metric_id=mid, **r2)["decode_cap_tokens"] is None, mid
    assert _render(metric_id="over_refusal_rate_full", **dict(r2, token_cap_tokens=2048))["decode_cap_tokens"] == 2048
    assert _render(metric_id="over_refusal_rate_full_plus_partial", **dict(r2, token_cap_tokens=8192))["decode_cap_tokens"] == 8192
    # the rule is the export's: another source's over-refusal rate keeps the cap it records
    assert _render(metric_id="over_refusal_rate", **dict(r2, lane_source_legacy="strength_lane"))["decode_cap_tokens"] == 512


# ---- metric: the export's greedy `recovered_<x>__t0` reading is the pin's `recovered_<x>`; the sampled one stays registry-only ------
def test_recovered_t0_metric_is_the_pin_s_recovered_metric():
    for x in ("asr", "at_cap_share", "degenerate_share", "loop_share", "mean_best_score"):
        assert _render(lane_source_legacy="r2_export", metric_id=f"recovered_{x}__t0")["metric"] == f"recovered_{x}"
        assert _render(lane_source_legacy="r2_export", metric_id=f"recovered_{x}__sampled")["metric"] == f"recovered_{x}__sampled"
    assert _render(lane_source_legacy="r2_export", metric_id="recovered_asr_folded_counterfactual")["metric"] == "recovered_asr_folded_counterfactual"
    assert _render(lane_source_legacy="r2_export", metric_id="recovered_vs_others_asr_gap")["metric"] == "recovered_vs_others_asr_gap"


# ---- overrefusal_items: the per-item reader's judge literal and the target directory's dose ---------------------------------------
def test_overrefusal_items_rows_take_the_reader_s_judge_and_the_target_s_dose():
    base = dict(lane_source_legacy="overrefusal_items", eval_category="over_refusal", eval_id="xstest", metric_id="over_refusal_graded",
                judge_model="claude-sonnet-5", judge_rubric_name="xstest_cpi", token_cap_tokens=512, view_legacy="gemma4e4b")
    out = _render(**base)
    expected = _reader_literal("lane_llama8b.py", "JUDGE_OR") or P.OVERREFUSAL_ITEMS_JUDGE_LEGACY    # the pin's literal, where present
    assert out["judge"] == expected == P.OVERREFUSAL_ITEMS_JUDGE_LEGACY
    assert _render(**dict(base, eval_id="or_bench_hard", judge_rubric_name="or_bench_cpi"))["judge"] == P.OVERREFUSAL_ITEMS_JUDGE_LEGACY
    # a steered object with no recorded strength: the dose the target directory spells (`_or_view_meta`: 0.5 on `_halfdose`, else 1.0)
    assert out["dose"] == 1.0 and out["decode_cap_tokens"] == 512 and out["lane_kind"] == "base→instruct"
    assert _render(**dict(base, view_legacy="gemma4e4b_halfdose"))["dose"] == 0.5
    assert _render(**dict(base, view_legacy="nemotron3nano4b_halfdose", transfer_kind="chat_checkpoint"))["lane_kind"] == "chat-ckpt"
    # the source-less run the adapter used to file renders the judge model alone — the source IS the spelling
    assert _render(**dict(base, lane_source_legacy=None))["judge"] == "claude-sonnet-5"


# ---- which rows are paper rows: no reader source / a per-trial pass of a pooled source ----------------------------------------------
def test_paper_frame_exclusion_names_the_registry_only_rows():
    assert P.paper_frame_exclusion(_row(lane_source_legacy="r2_export")) is None
    assert P.paper_frame_exclusion(_row(lane_source_legacy="sampled_lane", pass_label="pooled_t0_t1")) is None
    assert P.paper_frame_exclusion(_row(lane_source_legacy="r2_export", pass_label="base_cap")) is None       # no pass rule for the export
    for label in ("base_cap", "ladder_rung", None):      # a sampled row must carry the filed pass; an unlabelled one is not the pooled reading
        why = P.paper_frame_exclusion(_row(lane_source_legacy="sampled_lane", pass_label=label))
        assert why and str(label) in why and "pooled_t0_t1" in why
    for label in ("base_cap", "ladder_rung"):
        why = P.paper_frame_exclusion(_row(lane_source_legacy="sampled_lane", pass_label=label))
        assert why and label in why and "pooled_t0_t1" in why
    for src in (None, "", float("nan")):
        assert P.paper_frame_exclusion(_row(lane_source_legacy=src, campaign="drill_20260917T180118Z_c64")) == P.NO_SOURCE_EXCLUSION_REASON


def test_paper_long_frame_leaves_the_registry_only_rows_out(tmp_path):
    """The paper-schema rendering path consults the exclusion: a source-less drill row and a sampled per-trial row are not rendered,
    the pooled row and a sourced row are. A stand-in `v_paper_long` table carries just the columns the derivations read (the run's
    headline resolution reads `eval_id` and the `eval` table's accepted ids too, 2026-09-21)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE v_paper_long (run_id TEXT, round_id TEXT, in_current_paper INTEGER, lane_source_legacy TEXT, pass_label TEXT, "
                 "campaign TEXT, model_id TEXT, metric_id TEXT, value REAL, run_extra_json TEXT, metric_extra_json TEXT, "
                 "eval_id TEXT DEFAULT 'pasttense', headline_metric_id TEXT DEFAULT 'asr')")
    conn.execute("CREATE TABLE eval (eval_id TEXT PRIMARY KEY, headline_metric_id TEXT, extra_json TEXT)")
    conn.execute("INSERT INTO eval VALUES ('pasttense', 'asr', '{}')")
    conn.execute("CREATE TABLE v_run_degeneration_current (run_id TEXT, judgment_id TEXT, measure TEXT, rate REAL, numerator REAL, denominator REAL, "
                 "denominator_unit TEXT, pooling_json TEXT, aggregation TEXT, definition_version TEXT, scope TEXT, measurement_state TEXT)")
    rows = [("r-drill", "r1", 1, None, "base_cap", "drill_20260917T180118Z_c64", "gemma4e4b_it", "asr", 0.1),
            ("r-trial", "r1", 1, "sampled_lane", "ladder_rung", "full22", "gemma4e4b_it", "asr", 0.2),
            ("r-pooled", "r1", 1, "sampled_lane", "pooled_t0_t1", "full22", "gemma4e4b_it", "asr", 0.3),
            ("r-export", "r1", 1, "r2_export", None, "full22", "gemma4e4b_it", "asr", 0.4)]
    conn.executemany("INSERT INTO v_paper_long (run_id, round_id, in_current_paper, lane_source_legacy, pass_label, campaign, model_id, metric_id, "
                     "value, run_extra_json, metric_extra_json) VALUES (?,?,?,?,?,?,?,?,?,'{}','{}')", rows)
    frame = P.paper_long_frame(conn, columns=["source", "model", "metric", "value"])
    assert sorted(frame["run_id"]) == ["r-export", "r-pooled"]
    assert sorted(frame["source"]) == ["r2_export", "sampled_lane"] and list(frame["model"].unique()) == ["gemma4e4b"]


# ---- effect_kind / delta_method: the readers' prose, not the registry's tokens ----------------------------------------------------
def test_effect_kind_and_delta_method_are_the_reader_s_sentences():
    cols = ["effect_kind", "delta_method"]
    # the export's own words ride the contrast (the r2 adapter's `*_legacy` park): its sentence, and ABSENT where it stated no method
    cx = json.dumps({"effect_kind_legacy": "cohen_h defence (1\u2212ASR) vs own unsteered", "delta_method_legacy": None})
    out = P.render_long_row(_row(lane_source_legacy="r2_export", effect_kind="cohens_h", delta_method="producer-filed (method not stated on the row)",
                                 contrast_extra_json=cx), cols)
    assert out["effect_kind"] == "cohen_h defence (1\u2212ASR) vs own unsteered" and out["delta_method"] is None
    cx2 = json.dumps({"effect_kind_legacy": "cohen_h accuracy vs own unsteered", "delta_method_legacy": "paired bootstrap over goals (10,000 draws, seed 0) + exact McNemar; join"})
    out = P.render_long_row(_row(lane_source_legacy="r2_export", eval_category="capability", effect_kind="cohens_h", delta_method="x", contrast_extra_json=cx2), cols)
    assert out["delta_method"] == "paired bootstrap over goals (10,000 draws, seed 0) + exact McNemar; join"
    # the sampled lane: the reader's sentences, the delta method over the contrast's n_pairs (load.py:3102 / :3104 at the pin)
    out = P.render_long_row(_row(effect_kind="cohens_h", delta_method="paired_bootstrap", n_pairs=259), cols)
    assert out["effect_kind"] == "Cohen's h of defence success (1 \u2212 ASR), positive = better"
    assert out["delta_method"] == "paired on 259 goals (per-goal success share over trials): percentile bootstrap 5000; Wilcoxon signed-rank p"
    ctl = P.render_long_row(_row(lane_source_legacy="sampled_lane_controls", eval_category="over_refusal", effect_kind=None, delta_method="paired_bootstrap", n_pairs=1318), cols)
    assert ctl["effect_kind"] is None and ctl["delta_method"].startswith("paired on 1318 benign items (graded score, per-item mean over trials; each cell at its own final rung)")
    # the strength / all-positions readers type an ASCII hyphen; a source with no rule keeps the registry's token
    assert P.render_long_row(_row(lane_source_legacy="strength_lane", effect_kind="cohens_h"), cols)["effect_kind"] == "Cohen's h of defence success (1 - ASR), positive = better"
    assert P.render_long_row(_row(lane_source_legacy="arditi_allpos", eval_category="capability", effect_kind="cohens_h"), cols)["effect_kind"] == \
        "Cohen's h of accuracy vs the same-standup unsteered arm, positive = better"
    assert P.render_long_row(_row(lane_source_legacy="multiturn_grades", effect_kind="cohens_h", delta_method="paired_bootstrap", n_pairs=10), cols) == \
        {**P.render_long_row(_row(lane_source_legacy="multiturn_grades"), cols), "effect_kind": "cohens_h", "delta_method": "paired_bootstrap"}
    # a token with no n_pairs on the row cannot be spelled: the token stands rather than a sentence with a hole
    assert P.render_long_row(_row(delta_method="paired_bootstrap", n_pairs=None), cols)["delta_method"] == "paired_bootstrap"
