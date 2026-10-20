"""The front-door WORKED EXAMPLE's model key, `qwen35_4b_demo`.

The public tree's worked example runs on Qwen3.5-4B, but the real `qwen35_4b` phase spellings are PAPER objects
(`qwen35_4b_s1` / `_s2` / `_s3` are the batch-64 fit-seed replicates, `qwen35_4b_d0p75` a paper dose run), so the recipe's
phase names get a namespace of their own: model key `qwen35_4b_demo`, fit checkpoint `qwen35_4b_demo_base`, both re-pinning
the SAME checkpoints as `qwen35_4b_it` / `qwen35_4b_base` (copied pins), `paper_status reference`, `run_in_this_work: false` on
the reproduce side. The example is ONE steered run: `battery` (the row's declared wave) carries every intervention including
its unsteered controls, `d0p75` / `d0p75_relu` are its ladder at 0.75 -- no `u0`, no `s2`. This module pins:

* the two models.yaml rows (the copied pins, the fields, the producer spellings -- no HF id among them) and the four
  legacy_spellings.yaml model_keys rows (the fields the gemma4_31b rows carry, the fit seed, the dose), resolvable by
  `jobplan.model_for_key` exactly as tests/test_legacy_spellings_cover_phase_views.py will demand once reproduce/models.yaml
  names the key (a separate branch; this branch lands first, so the predicate is run here on the registry side alone).
* OUT OF THE PAPER FRAME: `reference` (never a served model, so no `*` decode row expands onto it: the greedy controls, the MMLU letter
  read and the protocol rows refuse by name), no roster entry names it (`v_run_current` reads `not_in_roster` for any run registered
  under it: the roster join is exact on model_id). Its explicit decode rows are the paper's `qwen35_4b_it` rows under its own key
  (the ruling of 2026-10-09, `decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09`: the same checkpoint under the
  same pins runs the same instruments under the same presets) -- tests/test_registry_seeds_demo_instruments.py pins the mirror.
* the HF repo spellings still name the PAPER's rows everywhere a reader resolves them -- `adapters/dimensions.legacy_model_map`,
  the export's `model_spellings` (`PolicyFile.model_id`), `fit_sidecars.load_seeds` (yaml and connection), `lineage_common.Seeds.
  model_for_hf_id`, `_items_common.Dims.model` (the DB fallback: two rows carry the repo), `_items_common.model_map_by_hf_repo`
  and `strength_mtdiff.StrengthMtdiffAdapter._controls_dims` (the controls stores' resolver; a last-wins comprehension there made
  every XSTest / OR-Bench instrument lookup refuse once the demo rows were appended) -- through
  `vocab.model_spelling_owner` (in_paper outranks reference; two claimants of one rank still drop the spelling, as before).
* the committed decode-policy export is the export of a fresh registry loaded from these seeds (both copies, sha pinned).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pytest

from sisteer.registry import db as DB
from sisteer.registry import decode_policy as DP
from sisteer.registry import harvest as H
from sisteer.registry import jobplan as JP
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import dimensions as D
from sisteer.registry.adapters import fit_sidecars as FS
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import strength_mtdiff as SM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
EXPORT = os.path.join(ROOT, "configs", "registry", "decode_policy.json")
PACKAGE_EXPORT = os.path.join(ROOT, "src", "sisteer", "registry", "decode_policy.json")
PROD = synth.producer("tests.registry_seeds_demo_model")

DEMO, DEMO_BASE = "qwen35_4b_demo", "qwen35_4b_demo_base"
PAPER, PAPER_BASE = "qwen35_4b_it", "qwen35_4b_base"
HF_SERVED, HF_FIT = "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base"
#: the three phase names of the worked example (one steered run + its ladder), registered as `<key>_<phase>` (common.sh f22_registry_register
#: key_tag=$KEY_$VIEW); the recipe's u0 / s2 (gemma4_31b's unsteered-first and residual groups) have NO demo spelling by the rule
PHASES = ("battery", "d0p75", "d0p75_relu")      # the one wave is the row's DECLARED `wave: battery` (demo-import M3; Q6: rewritten, no s1 alias)
NO_PHASES = ("u0", "s2", "s1")                   # s1: the wave's former spelling — rewritten to battery with NO alias (Q6)
#: the gemma4_31b rows the demo rows are modeled on -- the same fields per phase
TEMPLATE = "gemma4_31b"


@pytest.fixture(scope="module")
def seeds():
    return D.load_seeds(SEEDS)


@pytest.fixture(scope="module")
def bundle(seeds):
    return D.build_records(seeds, PROD, now="2026-10-08T00:00:00.000Z")


@pytest.fixture(scope="module")
def rows(seeds):
    return list(seeds.legacy["model_keys"])


@pytest.fixture(scope="module")
def pf():
    return DP.PolicyFile.load(EXPORT)


def _row_for(rows, spelling):
    hits = [r for r in rows if spelling in [x.strip() for x in str(r.get("legacy", "")).split("|")]]
    assert len(hits) == 1, (spelling, len(hits))
    return hits[0]


# --------------------------------------------------------------------------------------------
# models.yaml: the two rows re-pin the paper's checkpoints under the example's own key
# --------------------------------------------------------------------------------------------


def test_the_two_rows_re_pin_the_papers_checkpoints_under_the_examples_key(seeds):
    by_id = {m["model_id"]: m for m in seeds.models}
    demo, base, paper, paper_base = by_id[DEMO], by_id[DEMO_BASE], by_id[PAPER], by_id[PAPER_BASE]
    # the SAME pins as the paper's rows -- copied, never invented
    assert (demo["hf_repo"], demo["hf_revision"]) == (paper["hf_repo"], paper["hf_revision"]) == (HF_SERVED, "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    assert (base["hf_repo"], base["hf_revision"]) == (paper_base["hf_repo"], paper_base["hf_revision"]) == (HF_FIT, "1001bb4d826a52d1f399e183466143f4da7b741b")
    assert demo["family"] == base["family"] == "qwen35" and demo["size_label"] == base["size_label"] == "4B"
    assert demo["checkpoint_kind"] == "instruct" and base["checkpoint_kind"] == "base"
    # the same snapshot carries the same measured facts (the served row) and the same unread ones (the fit row)
    assert demo["chat_template_hash16"] == paper["chat_template_hash16"] == "a4aee8afcf2e0711"
    assert demo["generation_config_present"] == paper["generation_config_present"] == 0
    assert (demo["thinking_channel"], demo["thinking_header_moves"]) == (paper["thinking_channel"], paper["thinking_header_moves"]) == ("available", 1)
    assert base["thinking_channel"] == "not_available" and base["generation_config_present"] == "unverified"
    # out of the paper frame by status, and said so
    assert demo["paper_status"] == base["paper_status"] == "reference"
    for m in (demo, base):
        assert "run_in_this_work: false" in m["notes"] and "never a result" in m["notes"], m["model_id"]
    assert "view_legacy_prefix qwen35_4b_demo" in demo["notes"]
    # the producer spellings are the key and its five phases on the SERVED row (run.view_legacy), the fit id alone on the fit row,
    # and NO HF id on either: the repo spellings stay the paper rows' (test_the_hf_repo_spellings_still_name_the_papers_rows_everywhere)
    assert list(demo["producer_spellings"]) == [DEMO, *(f"{DEMO}_{p}" for p in PHASES)]
    assert list(base["producer_spellings"]) == [DEMO_BASE]
    for m in (demo, base):
        assert not any("/" in str(x) for x in m["producer_spellings"]), m["producer_spellings"]
    # the rows follow the paper's in the file: fit_sidecars.load_seeds (setdefault) and lineage_common.model_for_hf_id (first match)
    # resolve an HF repo to the FIRST row that names it
    order = [m["model_id"] for m in seeds.models]
    assert order.index(PAPER) < order.index(DEMO) and order.index(PAPER_BASE) < order.index(DEMO_BASE)
    assert order[-2:] == [DEMO_BASE, DEMO]


# --------------------------------------------------------------------------------------------
# legacy_spellings.yaml model_keys: the key and its five phases, modeled on the gemma4_31b rows
# --------------------------------------------------------------------------------------------


def test_every_phase_spelling_resolves_as_the_cover_test_will_demand(rows):
    """tests/test_legacy_spellings_cover_phase_views.py: every phase of a front-door model in reproduce/models.yaml resolves
    through jobplan.model_for_key to the served id with a transfer kind. The demo key is not in reproduce/models.yaml on this
    branch (the pipeline session's branch adds it after this one lands), so the predicate is run here for the five phases."""
    sys.path.insert(0, os.path.join(ROOT, "reproduce", "lib"))
    import models as M  # noqa: E402  (reproduce/lib/models.py: the phase grammar the cover test filters views with)
    served, tk = JP.model_for_key(rows, DEMO)
    assert (served, tk) == (DEMO, "base_to_instruct")
    for phase in PHASES:
        # a phase view by the reader's classification: the grammar's dose views, and the row's DECLARED wave `battery` (outside the grammar)
        assert M.is_phase_view(DEMO, phase) and (M.PHASE_VIEW_RE.match(phase) or phase == M.declared_wave(DEMO)), phase
        assert JP.model_for_key(rows, f"{DEMO}_{phase}") == (DEMO, "base_to_instruct"), phase
    # a phase nobody registered stays a refusal by name (exact match, one cited row per spelling) -- u0 and s2 included: the example is one run
    for spelling in (*(f"{DEMO}_{p}" for p in NO_PHASES), f"{DEMO}_s3", f"{DEMO}_d0p5", f"{DEMO}_s2r", f"{DEMO}_it"):
        with pytest.raises(JP.JobPlanRefused, match="is not a legacy model spelling"):
            JP.model_for_key(rows, spelling)


def test_the_six_rows_carry_the_fields_of_the_rows_they_are_modeled_on(rows):
    # the demo's declared wave `battery` is modeled on the template's steered wave `s1` (the same fields: a fit seed, no strength)
    template_phase = {"battery": "s1"}
    for spelling, tmpl in ((DEMO, TEMPLATE), *((f"{DEMO}_{p}", f"{TEMPLATE}_{template_phase.get(p, p)}") for p in PHASES)):
        row, template = _row_for(rows, spelling), _row_for(rows, tmpl)
        assert set(row) == set(template), (spelling, sorted(set(row) ^ set(template)))
        assert row["model_id"] == DEMO and row["fit_model_id"] == DEMO_BASE and row["transfer_kind"] == "base_to_instruct", spelling
        assert row.get("strength_nominal") == template.get("strength_nominal"), spelling
        # the SAME field, the demo's OWN label: the template's phases carry the paper's `s1`; the demo's carry `s3`, the fit-your-own default
        # training seed (the rule of 2026-10-12, matching the one-seed objects/DEMO.json; the registry owner's s1 -> s3 ruling of
        # 2026-10-13 05:56 — the label only describes the lane, no reader stamps it onto a demo record)
        assert ("fit_replicate_label" in row) == ("fit_replicate_label" in template), spelling
        if "fit_replicate_label" in template:
            assert template["fit_replicate_label"] == "s1" and row["fit_replicate_label"] == "s3", (spelling, row.get("fit_replicate_label"))
        assert "worked example" in row["source"] and "out of the paper frame" in row["source"], spelling
    # s1 is the example's SINGLE run: every intervention, its unsteered controls included
    s1 = _row_for(rows, f"{DEMO}_battery")["source"]
    assert "SINGLE run" in s1 and "unsteered controls" in s1 and "battery" in s1
    # the one fit seed of the recipe on every phase (each serves a fitted object) is seed 3, labelled s3 and cited to the rule in every
    # row's source, where "fit seed 1" stood (RED before the s1 -> s3 edit of configs/registry/legacy_spellings.yaml); the ladder at 0.75 on the dose rows
    assert [_row_for(rows, f"{DEMO}_{p}").get("fit_replicate_label") for p in PHASES] == ["s3", "s3", "s3"]
    for p in PHASES:
        src = _row_for(rows, f"{DEMO}_{p}")["source"]
        assert "fit seed 3 (the authors' rule: the fit-your-own default training seed is 3, matching the one-seed objects/DEMO.json)" in src, (p, src)
        assert "fit seed 1" not in src, (p, src)
    assert [_row_for(rows, f"{DEMO}_{p}").get("strength_nominal") for p in PHASES] == [None, 0.75, 0.75]
    # no row for u0 or s2 exists anywhere in the table
    for p in NO_PHASES:
        assert not [r for r in rows if f"{DEMO}_{p}" in [x.strip() for x in str(r.get("legacy", "")).split("|")]], p
    # the paper's spellings are untouched and never the demo's: the fit-seed replicates (exact rows) still name the paper model, and
    # the paper's dose spellings are no model_keys rows at all (the readers resolve them by grammar through the key `qwen35_4b`)
    for paper_spelling in ("qwen35_4b", "qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"):
        assert JP.model_for_key(rows, paper_spelling)[0] == PAPER, paper_spelling
    with pytest.raises(JP.JobPlanRefused, match="is not a legacy model spelling"):
        JP.model_for_key(rows, "qwen35_4b_d0p75")
    demo_spellings = {DEMO, DEMO_BASE, *(f"{DEMO}_{p}" for p in (*PHASES, *NO_PHASES))}
    for r in rows:                                   # some rows carry no model_id (context-dependent entries); none of them is the demo's
        if r.get("model_id") != DEMO:
            assert not (demo_spellings & {x.strip() for x in str(r.get("legacy", "")).split("|")}), r.get("legacy")


# --------------------------------------------------------------------------------------------
# out of the paper frame: reference, no wildcard decode row, no roster entry
# --------------------------------------------------------------------------------------------


def test_out_of_the_paper_frame(seeds, bundle, pf):
    # never a served model: no `*` decode row expands onto it. Its explicit rows are the paper's qwen35_4b_it rows under its own key (the
    # front-door session's ruling of 2026-10-09; tests/test_registry_seeds_demo_instruments.py pins the mirror), and the fit checkpoint has none
    assert not ({DEMO, DEMO_BASE} & set(D.served_model_ids(seeds.models)))
    demo_rows = {pid: e for pid, (payload, e) in bundle.policies.rows.items() if payload["model_id"] == DEMO}
    assert demo_rows and not [pid for pid in demo_rows if demo_rows[pid].get("wildcard_expanded_from")]
    assert not [pid for pid, (payload, _) in bundle.policies.rows.items() if payload["model_id"] == DEMO_BASE]
    assert not [w for w in bundle.warnings if "demo" in w], bundle.warnings
    assert pf.models[DEMO]["paper_status"] == pf.models[DEMO_BASE]["paper_status"] == "reference"
    assert not [p for p in pf.policies if p["model_id"] == DEMO_BASE]
    for kw in (dict(eval_category="static_attack", thinking_mode="off"), dict(eval_category="multi_turn", thinking_mode="off"),
               dict(eval_category="agentic", thinking_mode="off"), dict(eval_category="petri", thinking_mode="off"),
               dict(eval_id="gsm8k", thinking_mode="on"), dict(eval_id="strongreject_bon"), dict(mode="thinking_card"), dict(mode="vendor_preset")):
        mine, theirs = pf.resolve(DEMO, **kw), pf.resolve(PAPER, **kw)
        assert mine.model_id == DEMO and mine.policy_id == DEMO + theirs.policy_id[len(PAPER):] and mine.params == theirs.params, kw
    # the wildcard-bound reads have no twin except MMLU, which the recipe runs: the demo's explicit greedy row, bound by the full 1,531-item
    # instrument. The protocol rows (BigCodeBench / BFCL, not in the recipe's families) have none: the planner mints no canonical instrument
    # for them on the demo, and a direct resolve() falls through C18's eval-over-category rule to the `capability|off|card` twin, by name --
    # never the protocol numbers (tests/test_registry_seeds_demo_instruments.py pins both as they stand)
    assert pf.resolve(DEMO, eval_id="mmlu", thinking_mode="off").policy_id == f"{DEMO}|mmlu|off|greedy@v1"
    assert pf.resolve(DEMO, eval_id="bigcodebench", thinking_mode="off").policy_id == f"{DEMO}|capability|off|card@v1"
    with pytest.raises(LookupError, match="no canonical seeded instrument"):
        JP.canonical_instrument(JP.load_seed_index(SEEDS), "bigcodebench", DEMO, "lora_native")
    # no roster of any version names the key, and the roster join is exact on model_id, so a run registered under it reads not_in_roster
    rosters = sorted(glob.glob(os.path.join(SEEDS, "roster_r2_v*.yaml")))
    assert rosters
    named = set()
    for path in rosters:
        text = open(path, encoding="utf-8").read()
        assert DEMO not in text, path
        if PAPER in text:
            named.add(PAPER)
    assert PAPER in named, "the scan is meaningful only if the paper's model is in a roster"
    assert "WHEN pm.run_id IS NULL THEN 'not_in_roster'" in S.SELECTION_STATE_SQL
    assert "re.model_id = r.model_id" in S.VIEW_SQL["v_roster_match"]
    # and BEFORE any include ruling can select it (migration 0028; the live full-suite BigCodeBench / BFCL include
    # names no model, and both CASEs tested `include` before the roster branch): the worked example's KEYS read not_in_roster / 0 in both
    # CASEs of v_run_current ahead of `include` (schema_def.WORKED_EXAMPLE_MODEL_IDS -- by key, not by paper_status: `reference` models such
    # as the 35B are in the frame by roster or ruling, and the 26 appendix runs its include selects stay selected) -- the synthetic scenario
    # is tests/test_registry_views_0028.py
    sel, pap = S.SELECTION_STATE_SQL, S.IN_CURRENT_PAPER_SQL
    assert set(S.WORKED_EXAMPLE_MODEL_IDS) == {DEMO, DEMO_BASE} and S.WORKED_EXAMPLE_MODEL_SQL == f"r.model_id IN ('{DEMO}', '{DEMO_BASE}')"
    assert sel.index(S.WORKED_EXAMPLE_MODEL_SQL) < sel.index("WHEN d.action = 'include' THEN 'selected'") < sel.index("WHEN pm.run_id IS NULL THEN 'not_in_roster'")
    assert pap.index(S.WORKED_EXAMPLE_MODEL_SQL) < pap.index("WHEN d.action = 'include' THEN 1")
    assert "paper_status" not in S.SELECTION_STATE_SQL and "paper_status" not in S.IN_CURRENT_PAPER_SQL        # never a frame change for a `reference` model
    by_status = {m["model_id"]: m["paper_status"] for m in seeds.models}
    assert all(by_status[k] == "reference" for k in S.WORKED_EXAMPLE_MODEL_IDS) and all(k in by_status for k in S.WORKED_EXAMPLE_MODEL_IDS)
    assert {"gemma4_31b", "qwen35_35b_a3b", "gemma4e2b_it"} <= {k for k, v in by_status.items() if v == "reference"} and not ({"gemma4_31b", "qwen35_35b_a3b", "gemma4e2b_it"} & set(S.WORKED_EXAMPLE_MODEL_IDS))


# --------------------------------------------------------------------------------------------
# the HF repo spellings stay the paper's: vocab.model_spelling_owner in every map a reader resolves through
# --------------------------------------------------------------------------------------------


def test_model_spelling_owner_prefers_the_papers_status_and_drops_a_same_rank_clash():
    assert V.model_spelling_owner({}) is None
    assert V.model_spelling_owner({"only": "reference"}) == "only"
    assert V.model_spelling_owner({PAPER: "in_paper", DEMO: "reference"}) == PAPER
    assert V.model_spelling_owner({DEMO: "reference", PAPER: "in_paper"}) == PAPER
    assert V.model_spelling_owner({"a": "descoped", "b": "reference"}) == "a"
    assert V.model_spelling_owner({"a": "in_paper", "b": "descoped"}) == "a"
    assert V.model_spelling_owner({"a": "in_paper", "b": "in_paper"}) is None            # the clash the map always dropped
    assert V.model_spelling_owner({"a": "reference", "b": "reference"}) is None
    assert V.model_spelling_owner({"a": "reference", "b": None}) == "a"                   # an unknown status ranks last


def _seeds_with(models, legacy_rows=()):
    return D.Seeds(seeds_dir="<memory>", models=list(models), evals=[], item_sets=[], judges=[], instruments=[], si_texts=[],
                   decode={"policies": [], "sampling_mode_policy": [], "rulings": {}, "citations": {}},
                   legacy={"model_keys": list(legacy_rows)}, base_commit="unverified")


def test_legacy_model_map_applies_the_owner_rule_and_keeps_the_same_rank_clash_dropped():
    paper = {"model_id": "p_it", "hf_repo": "org/repo", "paper_status": "in_paper", "producer_spellings": ["p"]}
    demo = {"model_id": "p_demo", "hf_repo": "org/repo", "paper_status": "reference", "producer_spellings": ["p_demo_u0"]}
    other = {"model_id": "q_ref", "hf_repo": "org/other", "paper_status": "reference", "producer_spellings": ["shared"]}
    third = {"model_id": "r_ref", "hf_repo": "org/third", "paper_status": "reference", "producer_spellings": ["shared"]}
    m = D.legacy_model_map(_seeds_with([paper, demo, other, third], [{"legacy": "org/repo", "model_id": "p_it"}]))
    assert m["org/repo"] == "p_it" and m["p"] == "p_it" and m["p_demo_u0"] == "p_demo" and m["p_demo"] == "p_demo"
    assert "shared" not in m and m["org/other"] == "q_ref" and m["org/third"] == "r_ref"
    # a legacy row naming an unknown model contributes nothing (as before)
    m2 = D.legacy_model_map(_seeds_with([paper], [{"legacy": "ghost", "model_id": "nobody"}]))
    assert "ghost" not in m2 and m2["org/repo"] == "p_it"


def test_export_policy_applies_the_owner_rule_to_the_exports_spellings(tmp_path):
    conn = DB.open_fresh(str(tmp_path / "r.sqlite"))
    cols = ("model_id", "family", "size_label", "checkpoint_kind", "hf_repo", "hf_revision", "chat_template_sha256", "thinking_channel",
            "paper_status", "record_uri", "ingest_seq", "extra_json")
    rev = "0" * 40
    rows = [("p_it", "qwen35", "4B", "instruct", "org/repo", rev, "unverified", "available", "in_paper", "x", 1, json.dumps({"producer_spellings": ["p"]})),
            ("p_demo", "qwen35", "4B", "instruct", "org/repo", rev, "unverified", "available", "reference", "x", 2, json.dumps({"producer_spellings": ["p_demo_u0"]})),
            ("q_ref", "qwen35", "4B", "instruct", "org/other", rev, "unverified", "available", "reference", "x", 3, json.dumps({"producer_spellings": ["shared"]})),
            ("r_ref", "qwen35", "4B", "instruct", "org/third", rev, "unverified", "available", "reference", "x", 4, json.dumps({"producer_spellings": ["shared"]}))]
    conn.executemany(f"INSERT INTO model ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows)
    data = DP.export_policy(conn)
    sp = data["model_spellings"]
    assert sp["org/repo"] == "p_it" and sp["p"] == "p_it" and sp["p_demo_u0"] == "p_demo"
    assert "shared" not in sp and sp["org/other"] == "q_ref" and sp["org/third"] == "r_ref"
    assert set(data["models"]) == {"p_it", "p_demo", "q_ref", "r_ref"} and data["models"]["p_demo"]["paper_status"] == "reference"
    conn.close()


def test_the_hf_repo_spellings_still_name_the_papers_rows_everywhere(seeds, pf):
    mmap = D.legacy_model_map(seeds)
    assert mmap[HF_SERVED] == PAPER and mmap[HF_FIT] == PAPER_BASE
    assert mmap[DEMO] == DEMO and mmap[DEMO_BASE] == DEMO_BASE and all(mmap[f"{DEMO}_{p}"] == DEMO for p in PHASES)
    assert D.adapter_legacy_spellings(seeds=seeds)["model_ids"][HF_SERVED] == PAPER
    # the export the harness entry points read (target_decode / manifest / bon_served / the misalignment runner: PolicyFile.model_id)
    assert pf.model_id(HF_SERVED) == PAPER and pf.model_id(HF_FIT) == PAPER_BASE
    assert pf.model_id(DEMO) == DEMO and all(pf.model_id(f"{DEMO}_{p}") == DEMO for p in PHASES) and pf.model_id(DEMO_BASE) == DEMO_BASE
    assert pf.models[DEMO]["hf_repo"] == pf.models[PAPER]["hf_repo"] == HF_SERVED
    # the trainer's sidecar reader and the lineage helper
    fs = FS.load_seeds(SEEDS)
    assert fs.model_id_by_hf_repo[HF_SERVED] == PAPER and fs.model_id_by_hf_repo[HF_FIT] == PAPER_BASE
    assert fs.model_revision_by_id[DEMO] == fs.model_revision_by_id[PAPER] and fs.model_revision_by_id[DEMO_BASE] == fs.model_revision_by_id[PAPER_BASE]
    lc = LC.Seeds.from_dir(SEEDS)
    assert lc.model_for_hf_id(HF_SERVED) == PAPER and lc.model_for_hf_id(HF_FIT) == PAPER_BASE
    assert lc.model_for_hf_id(f"{DEMO}_battery") == DEMO
    for phase in PHASES:
        mv = lc.model_for_view_legacy(f"{DEMO}_{phase}")
        assert (mv["model_id"], mv["fit_model_id"], mv["transfer_kind"], mv["confidence"]) == (DEMO, DEMO_BASE, "base_to_instruct", "recorded"), phase
    assert lc.model_for_view_legacy(f"{DEMO}_s3")["model_id"] is None


def test_model_map_by_hf_repo_applies_the_owner_rule():
    rows = [{"model_id": "p_it", "hf_repo": "org/repo", "paper_status": "in_paper"},
            {"model_id": "p_demo", "hf_repo": "org/repo", "paper_status": "reference"},
            {"model_id": "q_ref", "hf_repo": "org/other", "paper_status": "reference"},
            {"model_id": "r_ref", "hf_repo": "org/other", "paper_status": "reference"},
            {"model_id": "no_repo", "paper_status": "in_paper"}]
    assert IC.model_map_by_hf_repo(rows) == {"org/repo": "p_it"}                              # a sequence of rows
    assert IC.model_map_by_hf_repo({r["model_id"]: r for r in rows}) == {"org/repo": "p_it"}  # a {model_id: row} mapping
    assert IC.model_map_by_hf_repo(reversed(rows)) == {"org/repo": "p_it"}                    # never the order
    # the committed seeds: both Qwen repos name the paper's rows, every other repo its one row
    lc = LC.Seeds.from_dir(SEEDS)
    m = IC.model_map_by_hf_repo(lc.models)
    assert m[HF_SERVED] == PAPER and m[HF_FIT] == PAPER_BASE and m["google/gemma-4-31B-it"] == "gemma4_31b"
    assert len(m) == len({x["hf_repo"] for x in lc.models.values()})


def test_the_controls_resolver_of_the_strength_reader_names_the_papers_rows_and_finds_both_control_instruments(tmp_path):
    """`StrengthMtdiffAdapter._controls_dims()` built its own hf_repo -> model_id map with a last-wins
    comprehension over models.yaml, so with the demo rows appended `Qwen/Qwen3.5-4B` resolved to qwen35_4b_demo while the instrument
    map was built for qwen35_4b_it (TRANSFER_MODEL_SPELLING): the shared controls reader resolved every XSTest / OR-Bench store's
    manifest repo to the demo, the instrument lookup raised Unresolved, and `_emit_controls()` skipped the stores. The resolver now
    goes through the shared ownership rule; the inherited control-negative reader shares the method."""
    seeds = LC.Seeds.from_dir(SEEDS)
    for cls in (SM.StrengthMtdiffAdapter, SM.CtlNeg1MultiturnAdapter):
        adapter = cls(root=str(tmp_path / cls.__name__), seeds=seeds, bank_root=str(tmp_path / "bank"))
        dims, rows, ladders, missing = adapter._controls_dims("0" * 64)
        assert missing == {}, (cls.__name__, missing)
        assert dims.models[HF_SERVED] == PAPER and dims.models[HF_FIT] == PAPER_BASE, cls.__name__
        assert dims.model(HF_SERVED) == PAPER                                   # the shared reader's call, no connection behind it
        assert sorted(rows) == sorted(SM.OR_CONTROLS) and set(rows) == {"xstest", "or_bench_hard_1k"}
        for ctl in SM.OR_CONTROLS:
            key = (SM.OR_INSTRUMENTS[ctl], SM.OR_EVALS[ctl], PAPER)
            assert key in dims.instruments, (cls.__name__, ctl, sorted(dims.instruments))
            inst_id, item_set_id, extra = dims.instruments[key]
            assert inst_id and item_set_id and extra["model_id"] == PAPER
        assert not any(mid in (DEMO, DEMO_BASE) for mid in dims.models.values()) or dims.models[HF_SERVED] == PAPER
        assert dims.model_kinds[PAPER] == "instruct" and dims.model_kinds[DEMO] == "instruct"


def test_dims_db_fallback_and_the_sidecar_reader_resolve_a_twice_pinned_repo_to_the_papers_row(tmp_path):
    """`_items_common.Dims.model` falls back to `SELECT ... FROM model WHERE hf_repo = ?` and refused when two rows carried the repo;
    `fit_sidecars.load_seeds(conn=...)` took the first row in rowid order. Both go through the ownership rule now, on a registry
    loaded from the committed seeds; a same-rank pair stays unresolved."""
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    D.write_seed_records(root, SEEDS, producer=PROD, now="2026-10-08T00:00:00.000Z")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    with DB.reading(DB.db_path(root)) as conn:
        dims = IC.Dims(conn=conn)
        assert dims.model(HF_SERVED) == PAPER and dims.model(HF_FIT) == PAPER_BASE
        assert dims.model_kinds == {PAPER: "instruct", PAPER_BASE: "base"}
        assert dims.model("google/gemma-4-31B-it") == "gemma4_31b"
        with pytest.raises(IC.Unresolved, match="no model row carries this hf_repo"):
            dims.model("nobody/none")
        fs = FS.load_seeds(None, conn=conn)
        assert fs.model_id_by_hf_repo[HF_SERVED] == PAPER and fs.model_id_by_hf_repo[HF_FIT] == PAPER_BASE
        assert fs.model_revision_by_id[DEMO] == fs.model_revision_by_id[PAPER]
    # a same-rank pair is a clash the fallback still refuses by name
    conn = DB.open_fresh(str(tmp_path / "pair.sqlite"))
    cols = ("model_id", "family", "size_label", "checkpoint_kind", "hf_repo", "hf_revision", "chat_template_sha256", "thinking_channel",
            "paper_status", "record_uri", "ingest_seq")
    conn.executemany(f"INSERT INTO model ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [("a_ref", "qwen35", "4B", "instruct", "org/shared", "0" * 40, "unverified", "available", "reference", "x", 1),
                      ("b_ref", "qwen35", "4B", "instruct", "org/shared", "0" * 40, "unverified", "available", "reference", "x", 2),
                      ("c_it", "qwen35", "4B", "instruct", "org/owned", "0" * 40, "unverified", "available", "in_paper", "x", 3),
                      ("c_demo", "qwen35", "4B", "instruct", "org/owned", "0" * 40, "unverified", "available", "reference", "x", 4)])
    with pytest.raises(IC.Unresolved, match="2 model rows of the same paper_status"):
        IC.Dims(conn=conn).model("org/shared")
    assert IC.Dims(conn=conn).model("org/owned") == "c_it"
    fs = FS.load_seeds(None, conn=conn)
    assert fs.model_id_by_hf_repo == {"org/owned": "c_it"} and set(fs.model_revision_by_id) == {"a_ref", "b_ref", "c_it", "c_demo"}
    conn.close()
    # the lineage helper on a same-rank pair names nothing; on the owned repo the paper's row
    lc = LC.Seeds(source_dir="<memory>")
    lc.models = {"a_ref": {"hf_repo": "org/shared", "paper_status": "reference"}, "b_ref": {"hf_repo": "org/shared", "paper_status": "reference"},
                 "c_it": {"hf_repo": "org/owned", "paper_status": "in_paper"}, "c_demo": {"hf_repo": "org/owned", "paper_status": "reference", "producer_spellings": ["c_demo_s1"]}}
    assert lc.model_for_hf_id("org/shared") is None and lc.model_for_hf_id("org/owned") == "c_it" and lc.model_for_hf_id("c_demo_s1") == "c_demo"


def test_decode_policy_module_imports_standalone_without_a_package(tmp_path):
    """A module-level `from . import vocab` broke the harness trees that load decode_policy.py by path
    (tests/test_tinycap_launcher_port.py) or as a package of __init__ + decode_policy.py + decode_policy.json (tests/test_evalplus_served.py).
    The vocabulary import lives inside export_policy(), the one function that needs it."""
    import importlib.util
    import shutil
    import subprocess
    import sys
    src = DP.__file__                 # the module under test (as test_tinycap_launcher_port copies it), never a path built from this file's
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    assert "\nfrom . import vocab" not in text and "from . import vocab as V" in text   # inside export_policy only
    probe = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('dp', sys.argv[1]); m = importlib.util.module_from_spec(spec); "
             "sys.modules['dp'] = m; spec.loader.exec_module(m); print(m.EXPORT_VERSION)")
    copy = tmp_path / "decode_policy.py"
    shutil.copy(src, copy)
    out = subprocess.run([sys.executable, "-c", probe, str(copy)], text=True, capture_output=True, env={"PATH": os.environ.get("PATH", "")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(DP.EXPORT_VERSION)


# --------------------------------------------------------------------------------------------
# the whole seed set through the real loader, and the committed export is that registry's export
# --------------------------------------------------------------------------------------------


def test_seeds_load_into_a_fresh_registry_and_the_committed_export_is_its_export(tmp_path):
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    bundle, paths = D.write_seed_records(root, SEEDS, producer=PROD, now="2026-10-08T00:00:00.000Z")
    assert paths
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert bundle.counts()["model"] == 24
    with DB.reading(DB.db_path(root)) as conn:
        assert DP.check_all(conn, [EXPORT, PACKAGE_EXPORT]) == []
        live = DP.export_policy(conn)
    assert {DEMO, DEMO_BASE} <= set(live["models"]) and live["model_spellings"][HF_SERVED] == PAPER
    with open(EXPORT, encoding="utf-8") as fh:
        committed = json.load(fh)
    assert committed["header"]["policy_set_sha256"] == DP.policy_set_sha256(committed) == DP.policy_set_sha256(live)
