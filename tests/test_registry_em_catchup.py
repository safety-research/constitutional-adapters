"""EM catch-up: the ``qwen35_4b_allpos`` panel, the post-pin paper payload, the THREE published column generations,
and ``relu_native``'s worktree dependency.

What each half pins:

* **the stem.** The Qwen Arditi all-positions arm on the misalignment instrument is published as a single
  ``qwen35_4b_allpos_vendor`` tree — a VENDOR-ONLY STEM, with no T=1.0 tree to be superseded — and the enumeration must read it as
  the stem's LIVE panel under the paper's stem source ``em/agentic/qwen35_4b_allpos``. Before the stem existed the directory parsed
  as ``kind == unknown`` and its whole group refused, so the panel was absent from the registry with nothing naming it.
* **the post-pin payload** the paper's f33dca5d reader computes and the adapter did not carry: ``loop_share`` / ``at_cap_share`` as
  columns with their counts as degeneration measures, ``judge_blocked`` / ``lower_bound`` (a judge-censored cell is a FLOOR, not a
  point), the degeneracy decomposition, the staged operating point, and the engine fields.
* **the column set is ADDITIVE and never pinned.** Three generations coexist in the published panels and
  all three must read: no engine column; summary ``engine`` + paired ``engine`` / ``anchor_engine`` / ``cross_engine``; and those
  plus ``standup_engine`` / ``anchor_standup_engine``. A column the adapter REQUIRES and the file lacks refuses by name.
* **the corrected cross-engine semantics.** The ENGINE is the standup's, the ROUTE is the leg's: a passthrough or lens route on a
  ``lora_native`` standup is that native engine's own leg and crosses nothing. The defect this replaced read ``cross_engine`` True
  on every pair of the all-positions panel; the KNOWN-BAD here is the same file WITHOUT the standup columns, which falls back to the
  pre-fix route rule and SAYS SO in the authority.

Everything is synthetic (invented rates, placeholder shas, hostname ``TESTHOST``) except the two live-root grammar tests, which read
directory NAMES only.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import validate as VAL
from sisteer.registry.adapters import em_misalign as EM
from sisteer.registry.adapters import relu_native as RN
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

NOW = "2026-09-17T12:00:00.000Z"
PRODUCER = {"script": "tests.em_catchup", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
PASS, LENS, LORA = "proxy_vllm+passthrough+interleaved", "proxy_vllm+lens+interleaved", "proxy_vllm+lora+interleaved"
ALLPOS_STEM, ALLPOS_DIR = "qwen35_4b_allpos", "qwen35_4b_allpos_vendor"
ALLPOS_CELL = "arditi_allpos"
LORA_CELL, LORA_ID = "si_constv2_mix_full__lora_r4_ce", "constv2-mix-full__lora_r4_ce"
#: the metric rows the live panels leave without a `k` (they are not shares of n)
CONTINUOUS_METRICS = ("response_chars", "zlib_ratio")


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER))


def _adapter(root, grade, **kw):
    return EM.EmMisalignAdapter(root=root, full22_pins=kw.pop("pins", None), baseline_pins=kw.pop("bpins", None),
                                behaviour_source=kw.pop("behaviour_source", {}), grade_root=grade, **kw)


def _read(ad, group):
    recs = []
    ctx = AF.primed_ctx(ad, now=NOW, producer=PRODUCER, groups={group: []})     # the frame of the pass AFTER the fold: every own anchor current
    for batch in ad.read(ctx, group, []):
        recs.extend(batch.records)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:4]
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out, ad.last_report


def _panel(root: str, stem: str, arms, behaviours=("leaking",), metrics=None, engines=None, standup=None, cross=None,
           anchor_standup=None, drop_columns=()):
    """A panel in the published EM shape, written here so the test controls the BEHAVIOUR set and the column set.

    ``arms``: [(raw_arm, harmful, n)] — the harmful rate per behaviour (``n`` is per arm, so a judge-censored arm carries fewer).
    ``metrics``: {arm: {metric: estimate}} extra metric rows (loop / at_cap / short / response_chars / zlib_ratio / harmful_fuzzy).
    ``engines`` / ``standup`` add the summary columns of generations 2 and 3; ``cross`` / ``anchor_standup`` the paired ones.
    ``drop_columns`` removes a column the adapter reads (the known-bad)."""
    import pandas as pd
    d = os.path.join(root, stem)
    os.makedirs(d, exist_ok=True)
    srows, prows = [], []
    none_rate = next(a[1] for a in arms if a[0] == "none")
    for beh in behaviours:
        for arm, harm, n in arms:
            base = {"harmful": harm, "harmful_reasoned": harm * 0.9, "harmful_bare": 0.0, "degenerate": 0.01}
            base.update((metrics or {}).get(arm, {}))
            for metric, est in base.items():
                # the live panels carry `k` for every SHARE metric (harmful, degenerate, short, loop, at_cap, the harmful_* variants)
                # and NaN for the two continuous ones (response_chars, zlib_ratio) — the counts the degeneration measures read
                share = metric not in CONTINUOUS_METRICS
                srows.append(dict(target=stem, arm=arm, family="agentic", behavior=beh, metric=metric, n=n,
                                  k=(round(est * n) if share else None), estimate=est,
                                  ci_low=max(0.0, est - 0.05) if metric == "harmful" else None,
                                  ci_high=est + 0.05 if metric == "harmful" else None, ci_method="wilson95",
                                  boot_low=None, boot_high=None, note=None))
            if arm != "none":
                prows.append(dict(target=stem, arm=arm, ref="none", family="agentic", behavior=beh, metric="harmful", n_pairs=n,
                                  mean_arm=harm, mean_none=none_rate, diff=harm - none_rate, boot_low=harm - none_rate - 0.05,
                                  boot_high=harm - none_rate + 0.05, arm_only=5.0, none_only=15.0, mcnemar_p=0.04, note=None))
    # the em8q coherence rows every panel carries: ONE per arm, outside the behaviour loop (the live panels' shape — a second row per
    # arm would make the reader's `set_index("arm").estimate` lookup a Series instead of a scalar)
    for arm, _h, n in arms:
        srows.append(dict(target=stem, arm=arm, family="em8q", behavior="betley", metric="coherence", n=n, k=None,
                          estimate=90.0 - (5 if arm != "none" else 0), ci_low=88.0, ci_high=92.0, ci_method="bootstrap95",
                          boot_low=88.0, boot_high=92.0, note=None))
    s = pd.DataFrame(srows)
    if engines is not None:
        s["engine"] = s.arm.map(engines)
    if standup is not None:
        s["standup_engine"] = s.arm.map(standup)
    for c in drop_columns:
        if c in s.columns:
            s = s.drop(columns=[c])
    s.to_csv(os.path.join(d, "summary.csv"), index=False)
    pr = pd.DataFrame(prows)
    if engines is not None:
        pr["engine"] = pr.arm.map(engines)
        pr["anchor_engine"] = engines.get("none")
        pr["cross_engine"] = pr.arm.map(lambda a: bool((cross or {}).get(a, False)))
    if anchor_standup is not None:
        pr["standup_engine"] = pr.arm.map(standup or {})
        pr["anchor_standup_engine"] = anchor_standup
    pr.to_csv(os.path.join(d, "paired.csv"), index=False)
    return dict(dir=d, stem=stem)


def _allpos_world(tmp_path, *, generation: int = 3, staged: bool = True, blocked: int = 0):
    """The all-positions panel as EM published it: one native standup, cells [none, arditi_allpos] plus the ``none__lora0`` twin
    (unrostered), leaking + murder, the vendor decode. ``generation`` picks which CSV column set the files carry."""
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    n_arm = 510 - blocked
    arms = (("none", 0.375, 510), (ALLPOS_CELL, 0.08, n_arm), ("none__lora0", 0.386, 510))
    engines = {"none": PASS, ALLPOS_CELL: LORA, "none__lora0": LORA} if generation >= 2 else None
    standup = {a[0]: "lora_native" for a in arms} if generation >= 3 else None
    _panel(root, ALLPOS_DIR, arms, behaviours=("leaking", "murder"),
           metrics={ALLPOS_CELL: {"loop": 0.031, "at_cap": 0.031, "short": 0.0, "response_chars": 11740.0, "zlib_ratio": 0.409,
                                  "harmful_fuzzy": 0.08},
                    "none": {"loop": 0.0, "at_cap": 0.0, "short": 0.0, "response_chars": 8907.0, "zlib_ratio": 0.425,
                             "harmful_fuzzy": 0.375}},
           engines=engines, standup=standup, cross=({ALLPOS_CELL: False, "none__lora0": False} if generation >= 2 else None),
           anchor_standup=("lora_native" if generation >= 3 else None))
    legs = [("none", PASS, "710fee4c6673f7e6"), (ALLPOS_CELL, LORA, "79e26caa1912a88c"), ("none__lora0", LORA, "9902269837734103")]
    store = F.build_em_store(grade, ALLPOS_STEM, legs=legs, bank_objects={ALLPOS_CELL: F.SHA_A}, subdir="misalign_vendor")
    if staged:
        full = json.load(open(store, encoding="utf-8"))
        full["substrate"] = "modal"
        full["endpoint"]["bank_objects"][ALLPOS_CELL]["staged_from"] = {
            "c": 0.3, "dose_unit": "raw_coefficient", "staged_id": "ar_c0p3__arditi_allpos", "lane": "arditi_allpos",
            "phase": "both", "object_sha256": F.SHA_A}
        with open(store, "w", encoding="utf-8") as fh:
            json.dump(full, fh)
    return root, grade


# ============================================================================================
# 1. the stem: a vendor-only stem IS the stem's live panel
# ============================================================================================

def test_the_allpos_stem_is_in_the_panel_vocabulary_and_its_vendor_tree_is_the_live_panel(tmp_path):
    root, grade = _allpos_world(tmp_path)
    assert ALLPOS_STEM in EM.EM_STEMS and EM.EM_STEMS[ALLPOS_STEM] == ("qwen35_4b", "instruct-direct", "qwen35_4b_it", "it")
    p = EM.parse_panel_name(ALLPOS_DIR, EM.EM_STEMS)
    assert p.kind == "agentic" and p.base == ALLPOS_STEM and p.variant == "vendor" and p.group == ALLPOS_STEM
    ad = _adapter(root, grade)
    assert {ad.comparison_group(i) for i in ad.discover()} == {ALLPOS_STEM}
    rows, facts = ad.long_rows(ALLPOS_STEM)
    # the source is the paper's STEM spelling, not the directory's
    assert {r["source"] for r in rows} == {f"em/agentic/{ALLPOS_STEM}"}
    assert f"em/agentic/{ALLPOS_STEM}" in ad.covers_sources and ad.status_sources(ALLPOS_STEM) == (f"em/agentic/{ALLPOS_STEM}",)
    assert facts["stems"] == [dict(stem=ALLPOS_STEM, dir=ALLPOS_DIR, laddered=False, live=True, variant="vendor",
                                   pinned_instrument=False, vendor_only_stem=True)]
    # both behaviours, both paper arms; the standup's `none__lora0` twin is the lane's own control, not a paper arm
    assert {(r["cell_id"], r["eval_id"]) for r in rows} == {(c, e) for c in ("none", ALLPOS_CELL)
                                                            for e in ("agentic_leaking", "agentic_murder")}
    assert facts["unmapped_arms"] == {ALLPOS_DIR: ["none__lora0"]}
    assert all(not r["_superseded"] for r in rows)                 # nothing supersedes it: there is no T=1.0 tree
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert arm["_extra"]["panel_dir"] == ALLPOS_DIR and arm["_extra"]["panel_variant"] == "vendor"
    assert arm["_extra"]["panel_pinned"] is False and arm["_extra"]["panel_summary_sha256"]
    assert arm["_instrument"]["token_cap_tokens"] == 16384 and arm["_instrument"]["decode_source"] == "vendor_recommendation"
    # the store's own `substrate` field reaches the emitter as the producer field it is -- (field, value), filed `recorded` (2026-09-21)
    assert arm["_instrument"]["serving_path"] == "lora_native" and arm["_substrate"] == ("cells_full.json:substrate", "modal")
    assert arm["status"] == "measured" and arm["value"] == pytest.approx(8.0)


def test_a_vendor_tree_beside_a_plain_tree_is_still_a_panel_of_its_own(tmp_path):
    """The vendor-only rule is narrow: it fires only where the stem has NO plain and NO ladder tree. A ``_vendor_leak`` or unpinned
    ``_vendor`` tree beside a plain tree keeps reading as a panel in its own right, under its own directory source."""
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    _panel(root, "qwen35_4b_it", (("none", 0.3, 100), (LORA_CELL, 0.2, 100)), behaviours=("murder",))
    _panel(root, "qwen35_4b_it_vendor", (("none", 0.1, 100), (LORA_CELL, 0.05, 100)), behaviours=("murder",))
    for sub in ("misalign", "misalign_vendor"):
        F.build_em_store(grade, "qwen35_4b_it", legs=[("none", PASS, "a" * 16), (LORA_CELL, LENS, "b" * 16)],
                         bank_objects={LORA_ID: F.SHA_A}, subdir=sub)
    rows, facts = _adapter(root, grade).long_rows("qwen35_4b_it")
    assert {r["source"] for r in rows} == {"em/agentic/qwen35_4b_it", "em/agentic/qwen35_4b_it_vendor"}
    assert not any(s.get("vendor_only_stem") for s in facts["stems"])
    assert {s["dir"] for s in facts["stems"]} == {"qwen35_4b_it", "qwen35_4b_it_vendor"}


def test_live_root_grammar_reads_the_allpos_directory(tmp_path):
    """The names EM publishes today all parse to a readable kind or a documented skip — the all-positions tree included. Directory
    NAMES only; no panel is read."""
    names = ["em_organisms", "gemma4e4b", "gemma4e4b_bl", "gemma4e4b_bl__lam3-superseded", "gemma4e4b_it", "gemma4e4b_it_ladder",
             "glm46vflash", "glm46vflash_ladder", "llama8b_bl", "llama8b_bl__r0p1-superseded", "llama8b_it", "llama8b_it_ladder",
             "llama8b_main_ladder", "llama8b_org", "llama8b_org__minimal-template", "nemotron3nano4b", "nemotron3nano4b_16k",
             "nemotron3nano4b_bl", "nemotron3nano4b_ladder", "paper_style", "qwen35_4b", ALLPOS_DIR, "qwen35_4b_bl",
             "qwen35_4b_bl_vendor", "qwen35_4b_bl_vendor_leak", "qwen35_4b_it", "qwen35_4b_it_ladder", "qwen35_4b_it_vendor",
             "qwen35_4b_it_vendor_leak", "qwen35_4b_ladder", "qwen35_4b_vendor", "qwen35_4b_vendor_leak"]
    parsed = {n: EM.parse_panel_name(n, EM.EM_STEMS) for n in names}
    assert not [n for n, p in parsed.items() if p.kind == "unknown"], [n for n, p in parsed.items() if p.kind == "unknown"]
    assert parsed[ALLPOS_DIR].group == ALLPOS_STEM


# ============================================================================================
# 2. the post-pin paper payload
# ============================================================================================

def test_the_degeneracy_decomposition_lands_in_its_own_columns_and_as_counted_measures(tmp_path):
    """The paper reads ``loop_share`` and ``at_cap_share`` as columns of their own beside the producer's ``degenerate`` union; the
    registry files the same numbers AND their counts as ``run_degeneration`` measures, so a share is never a bare float."""
    root, grade = _allpos_world(tmp_path)
    rows, _ = _adapter(root, grade).long_rows(ALLPOS_STEM)
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert arm["loop_share"] == pytest.approx(0.031) and arm["at_cap_share"] == pytest.approx(0.031)
    dec = arm["_extra"]["degeneracy_decomposition"]
    assert dec["loop"] == pytest.approx(0.031) and dec["at_cap"] == pytest.approx(0.031) and dec["short"] == 0.0
    assert dec["response_chars"] == pytest.approx(11740.0) and dec["anchor_response_chars"] == pytest.approx(8907.0)
    assert dec["zlib_ratio"] == pytest.approx(0.409) and dec["harmful_fuzzy"] == pytest.approx(0.08)
    assert arm["_degeneration"]["counts"]["at_token_cap"] == (16, 510) and arm["_degeneration"]["counts"]["classifier"] == (16, 510)
    # EVERY run's records carry its own shares as COUNTED measures, each from the producer's own k (the steered runs file beside the
    # anchors since declared_common.ARMS transcribed the arm, 2026-09-17): the anchor loops and truncates on nothing, the steered arm
    # on 16 of 510, and each cell's union5 is the producer's own `degenerate` share
    bt, rep = _read(_adapter(root, grade), ALLPOS_STEM)
    anchor_ids = {r["run_id"] for r in bt["run"] if r["object_id"] == "none"}
    steered_ids = {r["run_id"] for r in bt["run"] if r["object_id"] != "none"}
    assert len(anchor_ids) == len(steered_ids) == 2
    by_measure = {}
    for d in bt["run_degeneration"]:
        by_measure.setdefault(d["measure"], []).append(d)
    assert set(by_measure) == {"at_token_cap", "classifier", "union5"}
    assert all(d["measurement_state"] == "measured" and d["denominator"] == 510 for ds in by_measure.values() for d in ds)
    for measure in ("at_token_cap", "classifier"):
        assert {d["numerator"] for d in by_measure[measure] if d["run_id"] in anchor_ids} == {0}
        assert {d["numerator"] for d in by_measure[measure] if d["run_id"] in steered_ids} == {16}
    assert {d["numerator"] for d in by_measure["union5"]} == {5}          # the fixture's `degenerate` 0.01 x 510 on both cells
    for x in bt["run_metric"]:
        share = 0.031 if x["run_id"] in steered_ids else 0.0
        assert x["frac_at_token_cap"] == pytest.approx(share) and x["frac_degenerate_classifier"] == pytest.approx(share)
    assert all(x["frac_degenerate_union5"] == pytest.approx(0.01) for x in bt["run_metric"])


def test_a_judge_censored_arm_is_filed_as_a_lower_bound_with_the_blocked_count(tmp_path):
    """The file states no blocked count, so it is the unsteered arm's ``n`` minus this arm's, and on a harmful-rate metric the
    blocked items censor non-randomly TOWARD harm — the rate is a FLOOR. Filed as data
    (``judge_blocked`` / ``lower_bound`` + ``n_abstentions``), never as a silent point."""
    root, grade = _allpos_world(tmp_path, blocked=28)
    rows, facts = _adapter(root, grade).long_rows(ALLPOS_STEM)
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    none = [r for r in rows if r["cell_id"] == "none" and r["eval_id"] == "agentic_leaking"][0]
    assert arm["_extra"]["judge_blocked"] == 28 and arm["_extra"]["lower_bound"] is True and arm["_extra"]["n_anchor_items"] == 510
    assert arm["n"] == 482 and bool(arm["suggestive"]) is True
    assert "LOWER BOUND, NOT A POINT" in arm["note"] and "28 of 510 items judge-blocked (n = 482)" in arm["note"]
    assert none["_extra"]["judge_blocked"] == 0 and none["_extra"]["lower_bound"] is False and bool(none["suggestive"]) is False
    assert facts["judge_blocked"] == {ALLPOS_DIR: {ALLPOS_CELL: 28}}
    # the blocked count rides the READING (n_abstentions) as well as the row, so a censored cell is visible in the metric table
    arm["_n_abstentions"] = arm["_n_abstentions"]         # (filed by the adapter; asserted below on the anchor's own record)
    assert arm["_n_abstentions"] == 28 and arm["_n_items_planned"] == 510
    # KNOWN-BAD: every arm scored on all 510 -> no bound anywhere, and no note claiming one
    root2, grade2 = _allpos_world(tmp_path / "clean", blocked=0)
    rows2, facts2 = _adapter(root2, grade2).long_rows(ALLPOS_STEM)
    assert not any(r["_extra"]["lower_bound"] for r in rows2) and "judge_blocked" not in facts2
    assert not any("LOWER BOUND" in r["note"] for r in rows2)


def test_the_staged_operating_point_becomes_the_coefficient_and_the_operating_point_id(tmp_path):
    """The staged objects carry their own served point in ``bank_objects[cell].staged_from`` (the dose ``c``, its unit and the
    ``staged_id`` the paper keys operating points on). The declared unit ``raw_coefficient`` is in ``vocab.STRENGTH_UNITS``
    (2026-09-17), so the unit is filed as declared with no confidence flag, and the declared spelling also rides ``extra_json`` for
    the paper's legacy readers; a unit the vocabulary did not admit would file ``unverified`` WITH the reason instead — a row never
    claims a unit it cannot spell."""
    from sisteer.registry import vocab as V
    root, grade = _allpos_world(tmp_path, staged=True)
    rows, _ = _adapter(root, grade).long_rows(ALLPOS_STEM)
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert arm["_strength_coefficient"] == 0.3 and arm["arm_operating_point_id"] == "ar_c0p3__arditi_allpos"
    assert arm["_extra"]["staged_from"]["staged_id"] == "ar_c0p3__arditi_allpos" and arm["_extra"]["staged_from"]["c"] == 0.3
    assert "raw_coefficient" in V.STRENGTH_UNITS and arm["_strength_unit"] == "raw_coefficient"
    assert "strength_unit" not in arm["_confidence"]                      # the unit is the vocabulary's: nothing to flag
    assert arm["_extra"]["strength_unit_declared_legacy"] == "raw_coefficient"
    # a store that stages nothing leaves the coefficient at the campaign nominal and files no operating-point id
    root2, grade2 = _allpos_world(tmp_path / "bare", staged=False)
    rows2, _ = _adapter(root2, grade2).long_rows(ALLPOS_STEM)
    a2 = [r for r in rows2 if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert a2["_strength_coefficient"] == 1.0 and "_strength_unit" not in a2 and a2["_extra"]["staged_from"] is None


# ============================================================================================
# 3. the three column generations, and the corrected cross-engine semantics
# ============================================================================================

def test_generation_3_standup_columns_decide_the_cross_engine_flag(tmp_path):
    """THE ENGINE IS THE STANDUP'S, THE ROUTE IS THE LEG'S: all three legs of the all-positions panel ran on one
    ``lora_native`` standup, so the passthrough-routed anchor and the lora-routed arm cross NOTHING. The producer's attested flag is
    the authority and the standup rule is the cross-check that agrees."""
    root, grade = _allpos_world(tmp_path, generation=3)
    rows, facts = _adapter(root, grade).long_rows(ALLPOS_STEM)
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert arm["_extra"]["cross_engine_contrast"] is False and arm["_extra"]["cross_engine_attested"] is True
    assert arm["_extra"]["cross_engine_class_rule"] is False and arm["_extra"]["cross_engine_disagreement"] is False
    assert "producer paired.csv cross_engine (attested" in arm["_extra"]["cross_engine_authority"]
    assert "standup engine class rule" in arm["_extra"]["cross_engine_authority"]
    assert arm["_extra"]["serving_engine"] == LORA and arm["_extra"]["standup_engine"] == "lora_native"
    assert arm["_extra"]["anchor_serving_engine"] == PASS and arm["_extra"]["anchor_standup_engine"] == "lora_native"
    assert arm["_contrast_extra"]["standup_engine"] == "lora_native" and arm["_contrast_extra"]["anchor_standup_engine"] == "lora_native"
    assert facts.get("cross_engine_disagreements") is None
    # nothing to allow: the contrast does not cross engines, so no pairing allowance is declared on it
    assert "pairing_allowance" not in arm["_contrast_extra"]


def test_generation_2_falls_back_to_the_route_rule_and_says_so(tmp_path):
    """The same panel WITHOUT the standup columns: the pre-fix route rule stands in (the arm's route against the panel's single
    steered route) and the authority NAMES it as pre-fix, so a row derived under the old rule is never mistaken for one derived
    under the new. This is the shape every ``*_vendor_leak`` tree carries today."""
    root, grade = _allpos_world(tmp_path, generation=2)
    rows, facts = _adapter(root, grade).long_rows(ALLPOS_STEM)
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert arm["_extra"]["standup_engine"] is None and arm["_extra"]["anchor_standup_engine"] is None
    assert arm["_extra"]["cross_engine_contrast"] is False and arm["_extra"]["cross_engine_attested"] is True
    auth = arm["_extra"]["cross_engine_authority"]
    assert "PRE-FIX" in auth and "route" in auth and "standup engine class rule" not in auth
    assert arm["_extra"]["panel_steered_engine_classes"] == ["lens", "lora_native"] or arm["_extra"]["panel_steered_engine_classes"]


def test_generation_1_no_engine_column_reads_and_is_neither_refused_nor_marked(tmp_path):
    """A tree that says nothing about engines (every pre-2026-09-16 panel) can neither be refused nor marked: the flag is None with
    the reason, and the rows still file."""
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    _panel(root, "gemma4e4b", (("none", 0.2, 100), (LORA_CELL, 0.1, 100)))
    pins = F.build_em_pins(str(tmp_path / "pins.json"))
    rows, facts = _adapter(root, grade, pins=pins).long_rows("gemma4e4b")
    arm = [r for r in rows if r["cell_id"] == LORA_ID][0]
    assert arm["_extra"]["serving_engine"] is None and arm["_extra"]["standup_engine"] is None
    assert arm["_extra"]["cross_engine_contrast"] is None and arm["_extra"]["cross_engine_attested"] is False
    assert "no engine stated for the arm" in arm["_extra"]["cross_engine_authority"]
    assert arm["value"] == pytest.approx(10.0) and arm["status"] == "measured"      # the reading is unaffected
    assert facts.get("cross_engine_disagreements") is None


def test_known_bad_a_producer_flag_disagreeing_with_the_standup_rule_is_recorded_never_smoothed(tmp_path):
    """The producer attests cross_engine True while both legs ran on ONE standup: both cannot be right. The attested flag stays the
    authority (it is the producer's own measurement) and the disagreement is recorded on the row and in the status facts."""
    root, grade = _allpos_world(tmp_path, generation=3)
    import pandas as pd
    p = pd.read_csv(os.path.join(root, ALLPOS_DIR, "paired.csv"))
    p.loc[p.arm == ALLPOS_CELL, "cross_engine"] = True
    p.to_csv(os.path.join(root, ALLPOS_DIR, "paired.csv"), index=False)
    rows, facts = _adapter(root, grade).long_rows(ALLPOS_STEM)
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert arm["_extra"]["cross_engine_contrast"] is True and arm["_extra"]["cross_engine_class_rule"] is False
    assert arm["_extra"]["cross_engine_disagreement"] is True
    assert len(facts["cross_engine_disagreements"]) == 2                      # one per behaviour
    assert all("producer cross_engine=True vs standup engine class rule" in x for x in facts["cross_engine_disagreements"])
    assert all("arm standup 'lora_native'" in x and "anchor standup 'lora_native'" in x for x in facts["cross_engine_disagreements"])


@pytest.mark.parametrize("col", ["metric", "estimate", "arm", "n"])
def test_known_bad_a_summary_column_the_adapter_reads_refuses_by_name(tmp_path, col):
    """The column set is ADDITIVE and unpinned — a writer may add columns without notice — but a column the reader NEEDS must exist:
    its absence is a named refusal, not a KeyError mid-panel and not a silently thinner panel."""
    root, grade = str(tmp_path / f"figs_{col}"), str(tmp_path / "grade")
    _panel(root, "gemma4e4b", (("none", 0.2, 100), (LORA_CELL, 0.1, 100)), drop_columns=(col,))
    with pytest.raises(AdapterRefusal, match=f"missing the column.*{col}"):
        _adapter(root, grade).long_rows("gemma4e4b")


def test_additive_columns_are_read_without_a_code_edit(tmp_path):
    """The other direction: a column the adapter has never heard of changes nothing. A future writer may add anything."""
    import pandas as pd
    root, grade = _allpos_world(tmp_path, generation=3)
    for fn in ("summary.csv", "paired.csv"):
        path = os.path.join(root, ALLPOS_DIR, fn)
        df = pd.read_csv(path)
        df["a_column_from_the_future"] = "whatever"
        df["another_one"] = 42
        df.to_csv(path, index=False)
    rows, facts = _adapter(root, grade).long_rows(ALLPOS_STEM)
    assert len(rows) == 4 and facts["refused"] == []
    arm2 = [r for r in rows if r["cell_id"] == ALLPOS_CELL and r["eval_id"] == "agentic_leaking"][0]
    assert arm2["value"] == pytest.approx(8.0) and arm2["_extra"]["cross_engine_contrast"] is False


def test_required_column_guard_is_exact_about_what_it_needs():
    """The guard's own contract, in both directions."""
    import pandas as pd
    df = pd.DataFrame([{c: 1 for c in EM.REQUIRED_SUMMARY_COLUMNS}])
    EM.require_columns(df, "<synthetic>", EM.REQUIRED_SUMMARY_COLUMNS, "summary.csv")      # exactly the required set: fine
    EM.require_columns(df.assign(extra=1), "<synthetic>", EM.REQUIRED_SUMMARY_COLUMNS, "summary.csv")   # additive: fine
    with pytest.raises(AdapterRefusal, match="missing the column"):
        EM.require_columns(df.drop(columns=["metric"]), "<synthetic>", EM.REQUIRED_SUMMARY_COLUMNS, "summary.csv")
    assert "behavior" in EM.REQUIRED_SUMMARY_COLUMNS and "behavior" not in EM.REQUIRED_ORG_SUMMARY_COLUMNS


def test_the_allpos_arm_is_in_the_registry_arm_vocabulary_so_its_runs_file(tmp_path):
    """The gap ``adapters/arditi_allpos.py`` shared, closed 2026-09-17: ``declared_common.ARMS`` transcribes the arm (the paper's
    config.ARMS at f60facb5) and ``EM_SPELLING_TO_ARMS_KEY`` carries its identity spelling, so the emitter resolves the cell by name
    and the steered runs FILE beside the anchors — nothing under ``row_refusals``, no ``arms_not_in_registry_vocabulary`` fact, and
    this module's landing pads for an untranscribed arm (``EM_LOCAL_ARMS`` / ``EM_EXTRA_SPELLINGS``) are empty. (Before the
    transcription this test pinned the open gap: the anchors filed, both steered runs were refused by name, visibly.)"""
    from sisteer.registry.adapters import declared_common as DC
    from sisteer.registry import vocab as V
    assert ALLPOS_CELL in DC.ARMS and DC.EM_SPELLING_TO_ARMS_KEY[ALLPOS_CELL] == ALLPOS_CELL
    a = DC.ARMS[ALLPOS_CELL]
    assert a.family == "baseline" and a.object_kind == "vector_all" and a.paper_label == "arditi_allpos" and a.paper_label in V.PAPER_LABELS
    assert EM.EM_LOCAL_ARMS == {} and EM.EM_EXTRA_SPELLINGS == {}
    root, grade = _allpos_world(tmp_path)
    ad = _adapter(root, grade)
    rows, facts = ad.long_rows(ALLPOS_STEM)
    assert "arms_not_in_registry_vocabulary" not in facts
    bt, rep = _read(ad, ALLPOS_STEM)
    anchors = [r for r in bt["run"] if r["object_id"] == "none"]
    steered = [r for r in bt["run"] if r["object_id"] != "none"]
    assert len(anchors) == 2 and len(steered) == 2 and {r["object_id"] for r in steered} == {F.SHA_A}, "both cells file"
    assert all(r["serving_path"] == "lora_native" and r["applied_positions_mode"] == "all" and r["strength_unit"] == "raw_coefficient"
               for r in steered)
    assert rep.refusals == [] and ad.status_extra(ALLPOS_STEM)["row_refusals"] == []   # visible on the status page: nothing refused
    # the ROWS are labelled by the shared table
    arm = [r for r in rows if r["cell_id"] == ALLPOS_CELL][0]
    assert arm["arm"] == ALLPOS_CELL and arm["family"] == "baseline" and arm["object_sha256"] == F.SHA_A
    assert arm["_object_id"] == F.SHA_A and arm["_instrument"]["fit_positions"] == "all"


# ============================================================================================
# 4. relu_native: the worktree dependency, made explicit
# ============================================================================================

def test_the_paired_records_live_in_run_state_and_no_path_points_at_a_worktree():
    """The worktree dependency is RETIRED (2026-09-28): the paired records sit in the run-state lane beside every other root this
    adapter reads, so no path here depends on a checkout being present, on a branch being landed, or on working-tree files. This
    test replaces the one that pinned the old arrangement — it must fail if anyone repoints PAIRED_ROOT back into a worktree."""
    assert RN.PAIRED_ROOT == os.path.join(RN.RS, "paired"), RN.PAIRED_ROOT
    assert "worktrees" not in RN.PAIRED_ROOT and "worktrees" not in RN.PAIRED_ROOT_DEPENDENCY
    assert RN.PAIRED_PIN in RN.PAIRED_ROOT_DEPENDENCY          # the pin the records were TAKEN at, kept as provenance
    assert not hasattr(RN, "PAIRED_ROOT_WORKTREE") and not hasattr(RN, "PAIRED_ROOT_BRANCH")


def test_an_absent_paired_root_is_a_missing_status_not_a_crash(tmp_path):
    """The known-bad the dependency exists for: with the worktree gone the grades roots still read, the rows file WITHOUT contrasts,
    and the status names the missing path — never an exception, never a silently contrast-less read."""
    absent = str(tmp_path / "gone")
    state = RN.paired_root_state(absent)
    assert state["status"] == "missing" and state["path"] == absent and "GSM8K" in state["consequence"]
    fx = F.build_relu_native(str(tmp_path / "relu"), RN.passthrough_anchor_allowance()[0])
    ad = RN.ReluNativeAdapter(rs=fx["rs"], paired_root=absent)
    assert ad.paired_state["status"] == "missing"
    rows, facts = ad.long_rows("llama8b")
    assert rows, "the grades aggregates still read without the paired records"
    assert facts["missing"] == [absent] and facts["paired_root"]["status"] == "missing"
    assert "not read" in facts["gsm8k_cards"]
    mt = [r for r in rows if r["eval_id"] == "actorattack" and r["cell_id"] == fx["cell"]][0]
    assert mt["_extra"]["paired_record_status"] == "missing" and RN.PAIRED_ROOT in mt["_extra"]["paired_record_dependency"]
    import math
    assert math.isnan(mt["delta"]) and "_anchor_key" not in mt          # no contrast is invented from a missing record
    assert not [r for r in rows if r["eval_id"] == "gsm8k"]             # the GSM8K cards live in the paired records alone
    x = ad.status_extra("llama8b")
    assert x["missing"] == [absent] and x["paired_root"]["pin"] == RN.PAIRED_PIN


def test_a_present_paired_root_reports_present_with_its_record_count(tmp_path):
    fx = F.build_relu_native(str(tmp_path / "relu"), RN.passthrough_anchor_allowance()[0])
    ad = RN.ReluNativeAdapter(rs=fx["rs"], paired_root=fx["paired"])
    assert ad.paired_state["status"] == "present" and ad.paired_state["n_records"] == 3
    rows, facts = ad.long_rows("llama8b")
    assert facts["paired_root"]["status"] == "present" and "missing" not in facts
    mt = [r for r in rows if r["eval_id"] == "actorattack" and r["cell_id"] == fx["cell"]][0]
    assert "paired_record_status" not in mt["_extra"] and mt["_anchor_kind"] == "borrowed"
    assert ad.status_extra("llama8b")["paired_root"]["n_records"] == 3
