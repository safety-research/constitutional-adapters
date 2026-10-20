"""The Qwen3.5-4B EM MODEL ORGANISMS panels through the EM adapter, on SYNTHETIC fixtures:

* two panels under the figures root -- ``qwen35_4b_org`` (the T = 1.0 RECORD rung) and ``qwen35_4b_org_vendor`` (the vendor preset, the
  default from 2026-09-17) -- are two comparison groups with their own PROVISIONAL sources ``em/organisms/qwen35_4b_org`` and
  ``em/organisms/qwen35_4b_org_vendor`` (the rule: a ``_vendor`` tree beside a plain tree is its own panel).
* every row's eval is ``em8q`` and its MODEL is the organism (``qwen35_4b_org_financial`` / ``_medical`` / ``_sports``; the bare legs are the
  clean ``qwen35_4b_it``), read off the composite ``<organism>::<defence>`` arm column the EM analyze.py writes.
* the decode per tree is what the judged store's ``em8q/<cell>/meta.json`` SENT (sampler ``t1`` -> temperature 1.0 alone, ``vendor`` -> the six
  vendor keys), cross-checked against the tree's declaration -- a disagreement refuses the cell by name, a missing record falls back to the
  declaration and says so; ``status_extra`` carries ``sampler`` and the cap-hit share per arm (the clean model at the 600 cap on
  57.8 % of answers on the record rung) as facts.
* the Llama organism stem keeps its behaviour (``em_organisms`` group, ``em/em_organisms`` source; ``_vendor`` refused).

Everything here is invented (tests/registry_declared_fixtures.py + local CSV / JSON writers); nothing reads a real store.
"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from sisteer.registry import adapters as ADP
from sisteer.registry import sources as SRC
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import em_misalign as EM
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal
from sisteer.registry import synth

import registry_declared_fixtures as F  # noqa: E402
import _anchor_frame_fixtures as AF  # noqa: E402  (emitter 4: in-memory reads need a primed anchor frame)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = "2026-09-18T03:00:00.000Z"
PRODUCER = {"script": "tests.em_organisms_qwen", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}
PASS, LENS = "proxy_vllm+passthrough+interleaved", "proxy_vllm+lens+interleaved"
LORA_CELL, PSR_CELL, SAFETY = "si_constv2_mix_full__lora_r4_ce", "si_constv2_mix_full__psr_ce_all", "prompted__safety"
LORA_ID, PSR_ID = "constv2-mix-full__lora_r4_ce", "constv2-mix-full__psr_ce_all"
ORGS = ("em-financial_r32", "em-medical_r32", "em-sports_r32")
ORG_MODEL = {"em-financial_r32": "qwen35_4b_org_financial", "em-medical_r32": "qwen35_4b_org_medical", "em-sports_r32": "qwen35_4b_org_sports"}
VENDOR = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
T1 = {"temperature": 1.0}
#: an arm the LIVE EM panel carries that the shared paper vocabulary (declared_common.ARMS / EM_SPELLING_TO_ARMS_KEY, the paper's config.ARMS
#: transcription) does NOT map: the non-distilled paired ReLU difference. It must be COUNTED as unmapped and left unfiled -- an arm identity is
#: never guessed, and widening a shared table every adapter keys on is not this adapter's call (it is reported instead).
UNMAPPED_LIVE = "si_constv2_mix_full__psr_ce_all_minus_ctl"
#: (organism or None) x defence -> (misaligned rate, coherence, at-cap share); the bare legs are the clean model
CELLS = {None: {"none": (0.0, 92.6, 0.5775), "none__lens0": (0.0, 92.5, 0.5675), UNMAPPED_LIVE: (0.0, 65.0, 0.4), SAFETY: (0.0, 93.0, 0.1875), LORA_CELL: (0.0, 60.0, 0.935), PSR_CELL: (0.005, 70.0, 0.70)},
         "em-financial_r32": {"none": (0.31, 80.0, 0.0), SAFETY: (0.12, 82.0, 0.0), LORA_CELL: (0.28, 71.0, 0.02), PSR_CELL: (0.20, 75.0, 0.01)},
         "em-medical_r32": {"none": (0.22, 81.0, 0.0), SAFETY: (0.10, 83.0, 0.0), LORA_CELL: (0.25, 70.0, 0.0), PSR_CELL: (0.15, 76.0, 0.0)},
         "em-sports_r32": {"none": (0.06, 85.0, 0.0), SAFETY: (0.02, 86.0, 0.0), LORA_CELL: (0.05, 72.0, 0.0), PSR_CELL: (0.04, 78.0, 0.0)}}


def _ctx() -> AdapterContext:
    return AdapterContext(root="", now=NOW, producer=dict(PRODUCER))


def _arm(org, dfc):
    return dfc if org is None else f"{org}::{dfc}"


def _panel(root: str, name: str, cells=CELLS, scale: float = 1.0) -> str:
    """A synthetic organism panel in the EM analyze.py shape (columns + engine / standup_engine; metrics misaligned, em, coherence, at_cap)."""
    import pandas as pd
    summ, pair = [], []
    for org, by_dfc in cells.items():
        for dfc, (rate, cohv, cap) in by_dfc.items():
            rate = rate * scale
            arm = _arm(org, dfc)
            eng = LENS if dfc not in ("none", SAFETY) else PASS
            base = dict(target=name.split("_vendor")[0], arm=arm, family="em8q", behavior="betley", n=400, ci_method="wilson95", note=None, engine=eng, standup_engine=None)
            summ.append(dict(base, metric="misaligned", k=int(round(rate * 400)), estimate=rate, ci_low=max(0.0, rate - 0.04), ci_high=rate + 0.04, boot_low=rate - 0.03, boot_high=rate + 0.03))
            summ.append(dict(base, metric="em", k=int(round(rate * 0.8 * 400)), estimate=rate * 0.8, ci_low=max(0.0, rate * 0.8 - 0.04), ci_high=rate * 0.8 + 0.04, boot_low=rate * 0.8 - 0.03, boot_high=rate * 0.8 + 0.03))
            summ.append(dict(base, metric="coherence", k=float("nan"), estimate=cohv, ci_low=cohv - 2, ci_high=cohv + 2, ci_method="bootstrap95", boot_low=cohv - 2, boot_high=cohv + 2))
            summ.append(dict(base, metric="at_cap", k=int(round(cap * 400)), estimate=cap, ci_low=max(0.0, cap - 0.04), ci_high=min(1.0, cap + 0.04), boot_low=cap - 0.03, boot_high=cap + 0.03))
            if dfc != "none":
                b = by_dfc["none"][0] * scale
                pair.append(dict(target=base["target"], arm=arm, ref=_arm(org, "none"), family="em8q", behavior="betley", metric="misaligned", n_pairs=400, mean_arm=rate,
                                 mean_none=b, diff=rate - b, boot_low=rate - b - 0.05, boot_high=rate - b + 0.05, arm_only=10, none_only=30, mcnemar_p=0.01, note=None,
                                 engine=eng, anchor_engine=PASS, standup_engine=None, anchor_standup_engine=None, cross_engine=False))
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(summ).to_csv(os.path.join(d, "summary.csv"), index=False)
    pd.DataFrame(pair).to_csv(os.path.join(d, "paired.csv"), index=False)
    return d


def _store(grade: str, subdir: str, sampler: str, meta_for=None, drop_meta=()) -> str:
    """The judged store: cells_full.json legs for every cell (organism legs keyed `<organism>::<defence>`) + em8q/<cell@dir>/meta.json with the
    sampler the runner recorded (``meta_for`` overrides the sampler per cell; ``drop_meta`` cells get no meta.json)."""
    legs = []
    i = 0
    for org, by_dfc in CELLS.items():
        for dfc in by_dfc:
            legs.append((_arm(org, dfc), LENS if dfc not in ("none", SAFETY, "none__lens0") else PASS, f"{i:016x}"))
            i += 1
    d = F.build_em_store(grade, "qwen35_4b_org", legs, {LORA_ID: F.SHA_A, PSR_ID: F.SHA_B}, subdir=subdir, wave_sha="13f9d0eb" + "0" * 32)
    store = os.path.dirname(d)
    for cid, _ident, _key in legs:
        if cid in drop_meta:
            continue
        s = (meta_for or {}).get(cid, sampler)
        md = os.path.join(store, "em8q", cid.replace("::", "@"))
        os.makedirs(md, exist_ok=True)
        with open(os.path.join(md, "meta.json"), "w") as fh:
            json.dump({"cell_id": cid, "family": "em8q", "sampler": s, "sampling": dict(VENDOR if s == "vendor" else T1), "temperature": 1.0,
                       "max_tokens": 600, "n_samples": 50, "slurm_job_id": "547787" if s == "t1" else "549505"}, fh)
    return store


def _adapter(root, grade):
    return EM.EmMisalignAdapter(root=root, full22_pins=None, baseline_pins=None, behaviour_source={}, grade_root=grade)


def _world(tmp_path, vendor: bool = True):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    _panel(root, "qwen35_4b_org")
    _store(grade, "misalign_org", "t1")
    if vendor:
        _panel(root, "qwen35_4b_org_vendor", scale=0.5)
        _store(grade, "misalign_org_vendor", "vendor")
    return root, grade


def _read_all(adapter):
    """Every group read, with the per-group reports SUMMED (``last_report`` is the last group's, one report per read)."""
    inputs = adapter.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(adapter.comparison_group(i), []).append(i)
    recs, reports = [], []
    ctx = AF.primed_ctx(adapter, now=NOW, producer=PRODUCER, groups=groups)     # the frame of the pass AFTER the fold: every own anchor current
    for g, ins in sorted(groups.items()):
        for batch in adapter.read(ctx, g, ins):
            recs.extend(batch.records)
        reports.append(adapter.last_report)
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    total = {k: sum(int(getattr(r, k)) for r in reports) for k in ("runs", "contrasts", "metrics", "dispositions", "superseded")}
    total["refusals"] = [x for r in reports for x in r.refusals]
    return recs, total


def _by_type(recs):
    out = {}
    for r in recs:
        out.setdefault(r["record_type"], []).append(r["payload"])
    return out


# --------------------------------------------------------------------------------------------
# 1. the grammar and the groups
# --------------------------------------------------------------------------------------------


def test_the_two_qwen_organism_trees_are_two_groups_with_their_own_provisional_sources():
    p = EM.parse_panel_name("qwen35_4b_org", EM.EM_STEMS)
    v = EM.parse_panel_name("qwen35_4b_org_vendor", EM.EM_STEMS)
    assert p.kind == "organism" and p.base == "qwen35_4b_org" and p.variant is None and p.group == "qwen35_4b_org"
    assert v.kind == "organism" and v.base == "qwen35_4b_org" and v.variant == "vendor" and v.group == "qwen35_4b_org_vendor"
    # the Llama organism stem is unchanged: its group is the paper's, and it admits no _vendor tree
    assert EM.parse_panel_name("llama8b_org", EM.EM_STEMS).group == "em_organisms"
    assert EM.parse_panel_name("llama8b_org_vendor", EM.EM_STEMS).kind == "unknown"
    # other suffixes on the Qwen stem are refused by name (a record copy `__<tag>` is admitted, as for Llama); a figure-only directory
    # (`qwen35_4b_org_eb8_record`, no summary.csv) is never a panel
    assert EM.parse_panel_name("qwen35_4b_org_ladder", EM.EM_STEMS).kind == "unknown"
    assert EM.parse_panel_name("qwen35_4b_org__minimal-template", EM.EM_STEMS).tag == "minimal-template"
    assert EM.parse_panel_name("qwen35_4b_org_eb8_record", EM.EM_STEMS).kind == "unknown"
    ad = EM.EmMisalignAdapter()
    assert EM.QWEN_ORG_SOURCE == "em/organisms/qwen35_4b_org" and EM.QWEN_ORG_VENDOR_SOURCE == "em/organisms/qwen35_4b_org_vendor"
    assert {EM.QWEN_ORG_SOURCE, EM.QWEN_ORG_VENDOR_SOURCE, "em/em_organisms"} <= set(ad.covers_sources)
    assert ad.status_sources("qwen35_4b_org") == (EM.QWEN_ORG_SOURCE,) and ad.status_sources("qwen35_4b_org_vendor") == (EM.QWEN_ORG_VENDOR_SOURCE,)
    assert ad.status_sources("em_organisms") == ("em/em_organisms",)
    # the organism -> model map names the three seeded organism rows
    assert EM.QWEN_ORG_MODELS == ORG_MODEL


def test_the_sources_are_declared_provisionally_and_covered_by_the_em_adapter_alone():
    with open(os.path.join(REPO, "configs", "registry", "sources_r2_v1.yaml"), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = {r["source"]: r for r in data["sources"]}
    adds = {a["source"]: a for a in data["meta"]["additions"]}
    for src in (EM.QWEN_ORG_SOURCE, EM.QWEN_ORG_VENDOR_SOURCE):
        assert names[src]["owner"] == "the authors" and src in adds
        a = adds[src]
        # NOT a paper-declared source yet: the registry declares it provisionally, so the addition carries no paper commit and says who owns the spelling
        assert a["paper_build_commit"] is None and a["declared_by"].startswith("registry") and a["provisional"] is True and "PROVISIONAL" in a["why"]
        assert a["producer_message_utc"] == "2026-09-18T02:24Z"
    covered = SRC.covered_sources(ADP.load_all())
    assert covered[EM.QWEN_ORG_SOURCE] == [("em_misalign", 0)] and covered[EM.QWEN_ORG_VENDOR_SOURCE] == [("em_misalign", 0)]


# --------------------------------------------------------------------------------------------
# 2. the rows: organism = model, em8q = eval, decode per tree from the store's own record
# --------------------------------------------------------------------------------------------


def test_record_rung_rows_carry_the_organism_model_ids_em8q_and_the_t1_decode_from_meta(tmp_path):
    root, grade = _world(tmp_path, vendor=False)
    ad = _adapter(root, grade)
    assert {ad.comparison_group(i) for i in ad.discover()} == {"qwen35_4b_org"}
    rows, facts = ad.long_rows("qwen35_4b_org")
    assert rows and {r["source"] for r in rows} == {EM.QWEN_ORG_SOURCE} and {r["eval_id"] for r in rows} == {"em8q"}
    assert {r["_item_set_id"] for r in rows} == {"em8q_questions"} and {r["model"] for r in rows} == {"qwen35_4b"}
    assert all(not r["_superseded"] for r in rows)
    by = {(r["_extra"]["organism"], r["_extra"]["defence"]): r for r in rows}
    # the organism legs are the organism MODELS; the bare legs are the clean qwen35_4b_it
    for org in ORGS:
        for dfc, cell in (("none", "none"), (SAFETY, "prompted__safety_refusal"), (LORA_CELL, LORA_ID), (PSR_CELL, PSR_ID)):
            r = by[(org, dfc)]
            assert r["_model_id"] == ORG_MODEL[org] and r["cell_id"] == cell and r["view"] == org and r["lane_kind"] == "em-organism"
    assert by[(EM.QWEN_ORG_BARE, "none")]["_model_id"] == "qwen35_4b_it" and by[(EM.QWEN_ORG_BARE, LORA_CELL)]["_model_id"] == "qwen35_4b_it"
    # the store's own record decides the decode: sampler t1 -> temperature 1.0 alone, runner_forced_T1, no seed, the record rung's instrument
    r = by[("em-financial_r32", SAFETY)]
    assert r["_instrument"]["decode"] == {"temperature": 1.0} and r["_instrument"]["decode_source"] == "runner_forced_T1"
    assert r["_instrument"]["decode_seed_rule"] == "none" and r["_instrument"]["token_cap_tokens"] == 600 and r["_instrument"]["thinking_mode"] == "off"
    assert r["_instrument"]["instrument_config_hash"] == "em_organism_em8q_t1" and r["_instrument"]["harness"] == "em_organism"
    assert r["_extra"]["sampler"] == "t1" and r["_extra"]["sampler_source"].endswith("meta.json") and r["_extra"]["decode_declared_by_tree"] == "t1"
    assert r["_extra"]["panel_dir"] == "qwen35_4b_org" and r["_extra"]["panel_variant"] is None and r["_extra"]["organism_model_id"] == ORG_MODEL["em-financial_r32"]
    assert r["value"] == pytest.approx(12.0) and r["k"] == 48 and r["n"] == 400 and r["coherence"] == pytest.approx(82.0)
    assert r["delta"] == pytest.approx(-19.0) and r["_anchor_key"] == f"{EM.QWEN_ORG_SOURCE}|qwen35_4b_org|em-financial_r32|none"
    assert r["at_cap_share"] == pytest.approx(0.0) and r["_degeneration"]["counts"]["at_token_cap"] == (0, 400) and r["_degeneration"]["unit"] == "items"
    assert r["_producer_key"].startswith("misalign_org/qwen35_4b_org/") and r["_producer_key"].endswith("|em-financial_r32")
    assert r["_instrument"]["serving_path"] == "passthrough" and r["_extra"]["serving_engine"] == PASS
    lora = by[("em-medical_r32", LORA_CELL)]
    assert lora["object_sha256"] == F.SHA_A and lora["_instrument"]["serving_path"] == "lens" and lora["_strength_coefficient"] == 1.0 and lora["_is_operating_point"] == 1
    # the clean model's cap-hit share and every arm's, as status facts (on the record rung)
    assert facts["sampler"] == "t1" and facts["decode_declared_by_tree"] == "t1" and facts["samplers_recorded"] == ["t1"]
    assert facts["cap_hit_share_clean_model"] == pytest.approx(0.5775)
    assert facts["cap_hit_share"]["none"] == pytest.approx(0.5775) and facts["cap_hit_share"][f"em-financial_r32::{LORA_CELL}"] == pytest.approx(0.02)
    assert facts["cap_hit_share"][LORA_CELL] == pytest.approx(0.935)
    # the two defences the shared paper vocabulary does not map are COUNTED by name and left unfiled (never guessed): the lane's own lens
    # control and the non-distilled paired ReLU difference the live panel carries
    assert facts["organisms"] == sorted(ORGS) and facts["unmapped_arms"] == {"qwen35_4b_org": sorted(["none__lens0", UNMAPPED_LIVE])}
    assert UNMAPPED_LIVE not in DC.EM_SPELLING_TO_ARMS_KEY and not [r for r in rows if r["_extra"]["defence"] == UNMAPPED_LIVE]
    assert facts["cap_hit_share"][UNMAPPED_LIVE] == pytest.approx(0.4)      # its statistics are still read as facts
    assert facts["n_rows"] == len(rows) == 3 * 4 + 4       # three organisms x four defences + the four mapped bare legs (none__lens0 unmapped)
    # status_extra carries the same facts
    ad.last_facts["qwen35_4b_org"] = facts
    assert ad.status_extra("qwen35_4b_org")["sampler"] == "t1" and ad.status_extra("qwen35_4b_org")["cap_hit_share_clean_model"] == pytest.approx(0.5775)


def test_vendor_rung_is_its_own_panel_with_the_vendor_decode_from_meta_and_is_not_superseded(tmp_path):
    root, grade = _world(tmp_path)
    ad = _adapter(root, grade)
    assert {ad.comparison_group(i) for i in ad.discover()} == {"qwen35_4b_org", "qwen35_4b_org_vendor"}
    rows, facts = ad.long_rows("qwen35_4b_org_vendor")
    assert {r["source"] for r in rows} == {EM.QWEN_ORG_VENDOR_SOURCE} and all(not r["_superseded"] for r in rows)
    r = next(x for x in rows if x["_extra"]["organism"] == "em-sports_r32" and x["_extra"]["defence"] == SAFETY)
    assert r["_instrument"]["decode"] == VENDOR and r["_instrument"]["decode_source"] == "model_card" and r["_instrument"]["decode_seed_rule"] == "none"
    assert r["_instrument"]["instrument_config_hash"] == "em_organism_em8q_vendor" and r["_extra"]["sampler"] == "vendor"
    assert r["_extra"]["panel_dir"] == "qwen35_4b_org_vendor" and r["_extra"]["panel_variant"] == "vendor" and r["_model_id"] == ORG_MODEL["em-sports_r32"]
    assert r["_producer_key"].startswith("misalign_org_vendor/qwen35_4b_org/") and r["_key"].startswith(f"{EM.QWEN_ORG_VENDOR_SOURCE}|qwen35_4b_org_vendor|")
    assert r["value"] == pytest.approx(1.0) and r["decode"] == EM.EM_ORG_VENDOR_DECODE
    assert facts["sampler"] == "vendor" and facts["cap_hit_share_clean_model"] == pytest.approx(0.5775)
    # the record rung beside it reads unchanged, under its own source, and neither supersedes the other
    rows_t1, facts_t1 = ad.long_rows("qwen35_4b_org")
    assert {x["source"] for x in rows_t1} == {EM.QWEN_ORG_SOURCE} and facts_t1["sampler"] == "t1" and all(not x["_superseded"] for x in rows_t1)
    assert {x["_instrument"]["instrument_config_hash"] for x in rows_t1} == {"em_organism_em8q_t1"}
    # the store directories the two groups read are the two roots
    assert [os.path.relpath(d, grade) for d in ad._group_store_dirs("qwen35_4b_org")] == ["misalign_org/qwen35_4b_org"]
    assert [os.path.relpath(d, grade) for d in ad._group_store_dirs("qwen35_4b_org_vendor")] == ["misalign_org_vendor/qwen35_4b_org"]


def test_a_meta_sampler_disagreeing_with_the_tree_refuses_the_cell_by_name_and_a_missing_meta_falls_back_with_a_reason(tmp_path):
    root, grade = str(tmp_path / "figs"), str(tmp_path / "grade")
    _panel(root, "qwen35_4b_org_vendor", scale=0.5)
    _store(grade, "misalign_org_vendor", "vendor", meta_for={f"em-medical_r32::{SAFETY}": "t1"}, drop_meta=(f"em-sports_r32::{LORA_CELL}",))
    _panel(root, "qwen35_4b_org")
    _store(grade, "misalign_org", "t1")
    ad = _adapter(root, grade)
    rows, facts = ad.long_rows("qwen35_4b_org_vendor")
    bad = next(x for x in rows if x["_extra"]["organism"] == "em-medical_r32" and x["_extra"]["defence"] == SAFETY)
    assert bad["status"].startswith("refused:") and "sampler 't1'" in bad["status"] and "vendor" in bad["status"]
    assert any("em-medical_r32" in s and "meta.json" in s for s in facts["refused"])
    assert sorted(facts["samplers_recorded"]) == ["t1", "vendor"]
    missing = next(x for x in rows if x["_extra"]["organism"] == "em-sports_r32" and x["_extra"]["defence"] == LORA_CELL)
    assert missing["status"] == "measured" and missing["_extra"]["sampler"] == "vendor" and missing["_extra"]["sampler_source"] == "the tree's declaration (no meta.json beside the judged cell)"
    assert missing["_confidence"]["decode_source"][0] == "assumed_default" and "meta.json" in missing["_confidence"]["decode_source"][1]
    assert missing["_instrument"]["decode"] == VENDOR
    # a refused cell files a DISPOSITION carrying the refusal, never a run with a number (state `held` -- the reader put the number out of
    # reach; DISPOSITIONS has no `refused` value, and this is the same convention the engine-disagreement refusal follows)
    recs, rep = _read_all(ad)
    assert rep["refusals"] == []    # not an emitter refusal: the row's own status carries it
    bt = _by_type(recs)
    held = [d for d in bt["disposition"] if d["state"] == "held" and "meta.json" in d["reason"]]
    assert len(held) == 1 and "sampler 't1'" in held[0]["reason"] and held[0]["eval_id"] == "em8q"
    assert rep["dispositions"] == 1 and bad["_key"].endswith(f"|em-medical_r32|{SAFETY}")
    # and the refused cell is NOT among the VENDOR rung's runs (the record rung's own medical safety cell agrees with its tree and files)
    vendor_runs = [r for r in bt["run"] if r["lane_source_legacy"] == EM.QWEN_ORG_VENDOR_SOURCE]
    assert not [r for r in vendor_runs if r["model_id"] == ORG_MODEL["em-medical_r32"] and r["campaign_cell_id_legacy"] == "prompted__safety_refusal"]
    assert [r for r in bt["run"] if r["lane_source_legacy"] == EM.QWEN_ORG_SOURCE and r["model_id"] == ORG_MODEL["em-medical_r32"]
            and r["campaign_cell_id_legacy"] == "prompted__safety_refusal"]


def test_records_file_with_the_organism_models_em8q_and_contrasts_against_each_organisms_own_none(tmp_path):
    root, grade = _world(tmp_path)
    ad = _adapter(root, grade)
    recs, rep = _read_all(ad)
    assert rep["refusals"] == [] and rep["superseded"] == 0
    bt = _by_type(recs)
    runs = {r["run_id"]: r for r in bt["run"]}
    assert rep["runs"] == len(runs) == 2 * (3 * 4 + 4)   # two rungs x (three organisms x four defences + four mapped bare legs)
    assert {r["eval_id"] for r in runs.values()} == {"em8q"} and {r["item_set_id"] for r in runs.values()} == {"em8q_questions"}
    assert {r["model_id"] for r in runs.values()} == set(ORG_MODEL.values()) | {"qwen35_4b_it"}
    assert {r["lane_source_legacy"] for r in runs.values()} == {EM.QWEN_ORG_SOURCE, EM.QWEN_ORG_VENDOR_SOURCE}
    assert {r["thinking_mode"] for r in runs.values()} == {"off"} and {r["token_cap_tokens"] for r in runs.values()} == {600}
    assert {r["harness"] for r in runs.values()} == {"em_organism"} and {r["multi_turn_steering_scope"] for r in runs.values()} <= {"single_turn", "not_applicable"}
    # two instruments (the two rungs) and the decode each run SENT
    t1_runs = [r for r in runs.values() if r["lane_source_legacy"] == EM.QWEN_ORG_SOURCE]
    vd_runs = [r for r in runs.values() if r["lane_source_legacy"] == EM.QWEN_ORG_VENDOR_SOURCE]
    assert {r["decode_sent_json"] if isinstance(r["decode_sent_json"], str) else json.dumps(r["decode_sent_json"], sort_keys=True) for r in t1_runs} == {json.dumps(T1, sort_keys=True)}
    assert all((r["decode_sent_json"] if isinstance(r["decode_sent_json"], dict) else json.loads(r["decode_sent_json"])) == VENDOR for r in vd_runs)
    assert len({r["instrument_id"] for r in t1_runs} | {r["instrument_id"] for r in vd_runs}) >= 2
    assert {r["instrument_id"] for r in t1_runs}.isdisjoint({r["instrument_id"] for r in vd_runs})
    # every organism defence contrasts against ITS organism's none; the bare defences against the clean none
    anchors = {c["run_id"]: c["anchor_run_id"] for c in bt["run_contrast"]}
    assert rep["contrasts"] == 2 * (3 * 3 + 3)
    for rid, aid in anchors.items():
        assert runs[aid]["model_id"] == runs[rid]["model_id"] and runs[aid]["object_id"] == "none" and runs[aid]["lane_source_legacy"] == runs[rid]["lane_source_legacy"]
    # the cap-hit share is a counted degeneration measure on every run
    at_cap = {d["run_id"]: d for d in bt["run_degeneration"] if d["measure"] == "at_token_cap"}
    clean_t1 = next(r for r in t1_runs if r["model_id"] == "qwen35_4b_it" and r["object_id"] == "none")
    assert at_cap[clean_t1["run_id"]]["measurement_state"] == "measured" and at_cap[clean_t1["run_id"]]["numerator"] == 231 and at_cap[clean_t1["run_id"]]["rate"] == pytest.approx(0.5775)
    # the judgment names the em8q judge fields the seeds declare
    # ONE judge identity across both rungs (a judge record is minted per batch, so two groups write it twice under the same id)
    assert all(j["judge_id"] for j in bt["judgment"]) and len({j["judge_id"] for j in bt["judge"]}) == 1
    assert {j["rubric_name"] for j in bt["judge"]} == {"em_8_question"}


def test_a_missing_vendor_tree_is_no_group_and_a_missing_record_tree_refuses_the_group_by_name(tmp_path):
    root, grade = _world(tmp_path, vendor=False)
    ad = _adapter(root, grade)
    assert {ad.comparison_group(i) for i in ad.discover()} == {"qwen35_4b_org"}
    with pytest.raises(AdapterRefusal, match="qwen35_4b_org_vendor"):
        ad.long_rows("qwen35_4b_org_vendor")
    # the read report and the status sources are per group
    assert ad.status_sources("qwen35_4b_org_vendor") == (EM.QWEN_ORG_VENDOR_SOURCE,)


def test_the_live_root_names_parse_including_the_organism_trees():
    names = ["qwen35_4b_org", "qwen35_4b_org_vendor", "llama8b_org", "llama8b_org__minimal-template", "em_organisms", "qwen35_4b_it_vendor", "paper_style"]
    parsed = {n: EM.parse_panel_name(n, EM.EM_STEMS) for n in names}
    assert not [n for n, p in parsed.items() if p.kind == "unknown"]
    assert {p.group for p in parsed.values()} == {"qwen35_4b_org", "qwen35_4b_org_vendor", "em_organisms", "qwen35_4b_it", "skipped/paper_style"}
