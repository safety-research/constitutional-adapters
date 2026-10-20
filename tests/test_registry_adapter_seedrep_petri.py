"""The `seedrep_petri` reader on a SYNTHETIC store shaped like the Petri seed-1 store (invented run keys, a synthetic tree sha,
three audited objects, invented scenario names and scores; no transcript text), folding onto synthetic PREMINT-QUEUED rows in a scratch
registry built from the REAL seeds.

What it pins:

* IDENTITY: every run id equals `identity.run_id(seedrep_battery, <the premint producer key>, <the roster's staged id / none / prompted__<si>>,
  <eval>, <the premint's instrument for the eval>)` and, where a queued row exists, the queued row's own id; the instrument and judge come from
  the queued rows, the object from the roster's s1 selection and the pins; an object the premint never queued files as a NEW run under the same
  key, flagged `extra.admission.premint_row = absent`;
* COUNTS: `n_items_planned` 10, `n_items_generated` 10 (every scenario ran), `n_items_judged` = the scored count; the metric's `n` is the scored
  count and `n_abstentions` the refusals; one item per scored scenario on `petri_score` and the two shared dimensions, one item per refused
  scenario with `value NULL` and `label judge_refused`;
* CONTRASTS paired on scenario id against the store's own `none` per behaviour: `n_pairs` = the shared scored scenarios, the unshared counts in
  `extra.pairing`, the delta the hand-computed mean over the shared scenarios, the exact sign test, `anchor_kind same_standup_none`;
* the two PROVENANCE facts on every run: the waiver record read and named, and the re-audit absent;
* the FOLD: the queued rows advance queued -> generated -> judged -> ingested (the harvester's completion), the new runs land ingested too, a
  queued row of an object the store does not hold stays queued;
* the MINTED binding (v2, the seed-2 / seed-3 stores, no premint): the instrument minted from the seeded Petri row + the store's declared
  changes (the vendor decode sent, per-scenario-turn seeds, the attested window, the tree the completing drive ran, the config's auditor chain), the
  runs under the audit's name in the cluster campaign, the completing drive joined byte-for-byte through `leg_*.rc` (tree, cluster, org; the seed's
  tree is its `none` anchor's, a cell completed on another tree is refused by name), a mixed
  auditor / judge org filed as api_pool NULL with the drives per org, an overflow-withheld audit filed as one pending disposition per behaviour eval,
  an enumerated-only audit filing nothing, a non-overflow withholding refused by name;
* REFUSALS: a second root declaring the same seed, a root whose drive record is off the pin, and a missing waiver refuse the read as a whole; a
  results.json that its rows do not reproduce, an object listed `absent`, or missing store files refuse THAT object and nothing else; no
  premint-queued rows in the connected registry, or premint rows whose instruments an exclude ruling retired, refuse the group by name;
* the POOLED run (v3): one `petri_pooled` run per (seed, object) DERIVED from the six behaviour runs' files — the value the mean of the six
  behaviour means to the digit, `n` the scored items, `n_items_planned` 60, `n_abstentions` the refusals (+ withheld), every (behaviour, scenario)
  item filed again under it, the paired contrast against the seed's `none` pooled run, the judge markers rolled up onto its judgment (a judge
  mix recorded as a mix); its instrument minted per seed (the premint binding: from the premint's six rows with `extra.premint_family`, or
  bound to a queued `petri_pooled` row when the premint key set carries one; the minted binding: the seeded pooled row + the store's changes);
  a withheld audit's seventh pending disposition; an object whose six did not all file has no pooled run;
* the producer's per-row judge provenance (`judge_model_used`, the re-judge / kept-score fields) carried onto every item when present, the row's
  own `judge_model_used` winning over the mirror's derivation; a results.json whose sha no rc line names still refuses the object (v2's join).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil

import pytest

import _b1_fixtures as F
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import seedrep_petri as SP
from sisteer.registry.adapters.base import AdapterContext, AdapterRefusal
from sisteer.registry import synth

B = SP.BEHAVIOURS
TREE = synth.code_sha("c")
DRIVE_ID = "20260102T000000Z.000001"
ATTEMPT = "01SYNTHATTEMPT0000000000-00c0ffee0000"
AUDIT = "petri_full22_qwen35_4b_s1"
BUNDLE = "8812ba083bb5473c"
PK = "seedrep_battery:synth00:petri:s1"
NOW = "2026-09-21T09:00:00.000Z"
QUEUED_UTC = "2026-01-03T00:00:00.000Z"
LORA = "constv2-mix-full__lora_r4_ce"
LORA_STORE = "si_constv2_mix_full__lora_r4_ce"
STAGED = f"base64s1__{LORA}"
PROMPTED = "prompted__safety_refusal"
PROMPTED_STORE = "prompted__safety"
EXTRA = "constv2ctl2-full__lora_r4_ce"
EXTRA_STAGED = f"base64s1__{EXTRA}"
OBJ = hashlib.sha256(b"synthetic seedrep petri lora s1").hexdigest()
OBJ_EXTRA = hashlib.sha256(b"synthetic seedrep petri control lora s1").hexdigest()
MODEL_REV = "8" * 40
POLICY = "qwen35_4b_it|petri|off|vendor_preset@v1"
KEYS = {"none": "aaaa000000000001", LORA_STORE: "bbbb000000000002", PROMPTED_STORE: "cccc000000000003"}
KIND = {"none": "none", LORA_STORE: "lora_all", PROMPTED_STORE: "prompted"}
VARIANT = {"none": None, LORA_STORE: None, PROMPTED_STORE: "PROMPTED_SAFETY"}
BASE = {"none": 6.0, LORA_STORE: 4.0, PROMPTED_STORE: 5.0}
STEP = {"none": 0.1, LORA_STORE: 0.2, PROMPTED_STORE: 0.15}
#: the terminal judge refusals: (store spelling, behaviour) -> the refused scenario indices
REFUSED = {("none", "reasoning_unfaithfulness"): (9,), (PROMPTED_STORE, "cyber_bio_chem_capabilities"): (0,)}
T0 = 1789800000.0


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _write(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return path


def scenario(b: str, i: int) -> str:
    return f"{b[:8]}_scene_{i}"


def score(store_key: str, i: int) -> float:
    return BASE[store_key] + i * STEP[store_key]


def scored_indices(store_key: str, b: str):
    return [i for i in range(SP.N_SCENARIOS) if i not in REFUSED.get((store_key, b), ())]


def _entry(store_key: str) -> dict:
    return {"cell_class": {"none": "base", "prompted": "prompted"}.get(KIND[store_key], "class_a"), "cell_id": store_key, "control_of": None,
            "harness_config_hash": "7c50830278513300", "intervention_hash": hashlib.sha256(store_key.encode()).hexdigest()[:16], "kind": KIND[store_key],
            "object_path": None, "petri_run_key": KEYS[store_key],
            "serving_identity": "proxy_vllm+lora+interleaved" if KIND[store_key] == "lora_all" else "proxy_vllm+passthrough+interleaved",
            "si_id": None, "si_prefix_hash": None, "virtual_model_name": f"petri__{store_key}"}


def _object_files(root: str, store_key: str, idx: int, seed: int = 1, audit: str = AUDIT, state: str = "complete", withheld=()) -> None:
    """One object's store files. `state` complete: manifest, rows, results, empty withheld ledger (+ refusals); `incomplete`: manifest, rows and a
    withheld ledger naming `withheld` = [(behaviour, scenario, reason)] with NO results.json; `enumerated`: the manifest alone."""
    e = _entry(store_key)
    key = KEYS[store_key]
    d = os.path.join(root, "cells", key)
    variant = VARIANT[store_key]
    td = {"max_tokens": 1024, "mode": "vendor", "policy_id": POLICY, "request_params": dict(F.VENDOR_DECODE), "seed_rule": "per_scenario_turn"}
    lines, refused, behaviors = [], [], {}
    for bi, b in enumerate(B):
        idxs = scored_indices(store_key, b)
        for i in idxs:
            bkey = f"{b}/{variant}" if variant else b
            ex = {"auditor_chain": {"auditor_turns": 12, "used": [{"label": "sonnet-5", "model": "anthropic/claude-sonnet-5", "switch_turn": 1}], "version": 1},
                  "behavior_dir_key": bkey, "dimension": b, "epoch": 1, "sampling": dict(F.VENDOR_DECODE), "scenario_id": scenario(b, i), "seed_base": seed,
                  "seed_rule": "per_scenario_turn", "serving_engine": "lora_native", "target_decode": td}
            for metric, value in ((b, score(store_key, i)), ("eval_awareness", 1.0 + i * 0.05), ("scenario_realism", 8.0)):
                exm = dict(ex, dimension=(("cyber_bio_chem_operational_uplift" if b == "cyber_bio_chem_capabilities" else b) if metric == b else metric))
                lines.append(json.dumps({"arm_name": store_key, "attempt": 0, "coeff": None, "created_at": T0 + 100 * idx + bi + i / 100.0, "domain": "petri_behavior",
                                         "engine": e["serving_identity"], "env_digest": "ba9205dffa5ce700", "extra_json": json.dumps(exm, separators=(",", ":")),
                                         "harness_config_hash": e["harness_config_hash"], "intervention_hash": e["intervention_hash"], "item_id": scenario(b, i),
                                         "kind": KIND[store_key], "metric": metric, "model_id": "Qwen/Qwen3.5-4B", "model_revision": MODEL_REV, "position_order": None,
                                         "run_id": f"{audit}.{store_key}.{key[:8]}", "seed": 0, "slurm_job_id": None, "sweep": audit, "task": f"petri/{b}",
                                         "task_version": BUNDLE, "tier": -1, "trial_id": 0, "value": value, "wall_clock_s": None}, separators=(",", ":")))
        for i in REFUSED.get((store_key, b), ()):
            refused.append({"auditor_chain": {"auditor_turns": 11, "used": [{"label": "sonnet-5", "model": "anthropic/claude-sonnet-5", "switch_turn": 1}], "version": 1},
                            "behavior": b, "dimensions": [b, "eval_awareness", "scenario_realism"], "epoch": 1, "eval_log": f"/synthetic/{key}.eval",
                            "judge_model": "anthropic/claude-opus-5", "outcome": "judge_refused", "reason": "judge refused to score (anthropic/claude-opus-5): synthetic content filter",
                            "sample_id": scenario(b, i), "sample_uuid": f"uuid{i}", "scenario_id": scenario(b, i)})
        n = len(idxs)
        behaviors[b] = {"audited": n, "auditor_chain": {"sonnet_only": n}, "auditor_refused": {"total": 0}, "auditor_refused_items": [], "incomplete": False,
                        "mean": {b: sum(score(store_key, i) for i in idxs) / n, "eval_awareness": sum(1.0 + i * 0.05 for i in idxs) / n, "scenario_realism": 8.0},
                        "n_items": SP.N_SCENARIOS, "n_scenarios": SP.N_SCENARIOS, "rows_filed": 3 * n, "unresolved": 0,
                        "judge_refused_items": [x for x in refused if x["behavior"] == b]}
    n_scored = sum(len(scored_indices(store_key, b)) for b in B)
    _write(os.path.join(d, "manifest.json"), {"arm_name": store_key, "attempt": 0, "cell_id": store_key, "coeff": None, "engine": e["serving_identity"], "env_digest": "ba9205dffa5ce700",
                                              "harness_config_hash": e["harness_config_hash"], "intervention_hash": e["intervention_hash"], "kind": KIND[store_key],
                                              "model_id": "Qwen/Qwen3.5-4B", "model_revision": MODEL_REV, "petri_run_key": key, "run_id": f"{audit}.{store_key}.{key[:8]}",
                                              "scenario_bundle_hash": BUNDLE, "seed": 0, "serving_engine": "lora_native", "slurm_job_id": None, "sweep": audit,
                                              "target_decode": td, "target_seed_base": seed, "trial_id": 0, "virtual_model_name": f"petri__{store_key}"})
    if state == "enumerated":
        return
    _write(os.path.join(d, "rows.jsonl"), "\n".join(lines) + "\n")
    if state == "incomplete":
        unresolved = [{"behavior": b, "scenario_id": sc, "sample_id": sc, "epoch": 1, "dimension": b, "reason": reason} for b, sc, reason in withheld]
        _write(os.path.join(d, "petri_incomplete.json"), {"cell_id": store_key, "petri_run_key": key, "unresolved": unresolved, "version": 1})
        return
    results = {"audited": {"expected": len(B) * SP.N_SCENARIOS, "items": n_scored}, "auditor_chain": {"sonnet_only": n_scored}, "auditor_refused": {"total": 0},
               "behaviors": behaviors, "cell_id": store_key, "completed_at": T0 + 100 * idx + 50.0, "petri_run_key": key, "rows_emitted": 3 * n_scored, "unscored": 0}
    if refused:
        results["judge_refused"] = {"total": len(refused)}
        by_b = {}
        for x in refused:
            by_b.setdefault(x["behavior"], {"total": 0})["total"] += 1
        _write(os.path.join(d, "petri_judge_refused.json"), {"by_behavior": by_b, "cell_id": store_key, "judge_refused": refused, "petri_run_key": key, "version": 1})
    _write(os.path.join(d, "results.json"), results)
    _write(os.path.join(d, "petri_incomplete.json"), {"cell_id": store_key, "petri_run_key": key, "unresolved": [], "version": 1})


def _store(tmp_path, name: str = "store", objects=("none", LORA_STORE, PROMPTED_STORE)) -> dict:
    """A synthetic store root + waiver + roster / pins naming its objects; every path under tmp_path."""
    root = str(tmp_path / name)
    os.makedirs(os.path.join(root, f"behaviors-{BUNDLE}"), exist_ok=True)
    canonical = [_entry(k) for k in objects]
    _write(os.path.join(root, "cells.json"), {"absent": [], "audit": AUDIT, "canonical": canonical, "cells_skipped": [],
                                               "controls": [dict(_entry("none"), cell_id="none__lora0", cell_class="control", control_of="none", petri_run_key="dddd000000000004")],
                                               "expected_cells": list(objects), "flags_dropped": [], "serving_path": "lora_native",
                                               "target_decode": {"mode": "vendor", "policy": POLICY}, "written_at": T0})
    sched = [{"view": "s1", "cell_index": i, "cell_id": k, "vmodel": f"petri__{k}", "run_key": KEYS[k], "seed_base": 1, "width": 3, "tag": "qwen35_4b_s1", "petri_pass": ""}
             for i, k in enumerate(objects)]
    _write(os.path.join(root, "modal_drive.json"), {"schema": "seedrep_petri_drive/1", "drive_id": DRIVE_ID, "substrate": "modal", "view": "s1", "stage": "base", "shard": "0",
                                                    "W": TREE, "W7": TREE[:7], "attempt_id": ATTEMPT, "proxy_sha": TREE, "auditor": {"pool": "highprio", "key_sha256_12": "000000000000"},
                                                    "schedule": sched, "started_at": "2026-01-02T00:00:00Z", "finished_at": "2026-01-02T01:10:00Z",
                                                    "launches": [{"record_name": f"leg_{i}_s1", "tag": "qwen35_4b_s1", "t": T0 + 100 * i} for i in range(len(objects))],
                                                    "verdict": {"counts": {"complete": len(objects), "incomplete": 0, "failed": 0}, "nruns": len(objects), "release": True,
                                                                "detail": [[f"leg_{i}_s1", KEYS[k], "complete"] for i, k in enumerate(objects)]}})
    for i, k in enumerate(objects):
        _object_files(root, k, i)
    waiver = _write(str(tmp_path / f"{name}_waiver.json"),
                    [{"by": "synthetic launch agent", "at": "2026-01-03T00:00:00Z", "scope": {"W": TREE, "leg": "petri", "view": "s1", "legs": list(range(len(objects)))},
                      "reason": "synthetic: the sampled auditor chain never byte-reproduces; completed work is not re-run", "decision": "waive criterion 3 for the carried legs"}])
    import yaml
    roster = {"targets": {"qwen35_4b": {"views": {"s1": {"cells": {LORA: STAGED, EXTRA: EXTRA_STAGED, PROMPTED: None, "prompted__si_constitution": None}}}}}}
    roster_path = _write(str(tmp_path / f"{name}_roster.yaml"), yaml.safe_dump(roster))
    pins_path = _write(str(tmp_path / f"{name}_pins.json"), {"qwen35_4b": {STAGED: {"object_sha256": OBJ, "source": "synthetic"}, EXTRA_STAGED: {"object_sha256": OBJ_EXTRA, "source": "synthetic"}}})
    return {"root": root, "waiver": waiver, "roster": roster_path, "pins": pins_path}


def _adapter(fx: dict, seeds, roots=None, **kw) -> SP.SeedrepPetriAdapter:
    pin = dict(SP.STORE_PIN, W=TREE, drive_id=DRIVE_ID, attempt_id=ATTEMPT)
    return SP.SeedrepPetriAdapter(roots=roots or (fx["root"],), seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], waiver_path=fx["waiver"],
                                  premint_producer_key=kw.pop("premint_producer_key", PK), store_pin=kw.pop("store_pin", pin), **kw)


def _resolved(oid: str, staged: str, label: str, role: str) -> LC.ResolvedObject:
    return LC.ResolvedObject(object_id=oid, object_kind="lora", paper_label=label, object_role=role, confidence="recorded", source="synthetic",
                             staged_id=staged, si_id="constv2-mix-full" if role == "constitution" else "constv2ctl2-full")


def _instrument_records(seeds, now: str, with_pooled: bool = False):
    """Synthetic PREMINT instruments: the seeded `petri_full22_as_run` row per behaviour eval with a synthetic harness_version, minted through
    identity.instrument_id the way the premint mints its derived instruments (with_pooled: a `petri_pooled` one beside them, the shape a premint
    that queues the pooled eval would carry). Returns (records, {eval: instrument payload})."""
    from sisteer.registry.adapters import dimensions as DIM
    dim_recs = DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    out, by_eval = [], {}
    for b in B + ((SP.POOLED_EVAL,) if with_pooled else ()):
        ev = seeds.eval_for_spelling(b)
        seeded_id = seeds.instrument_row("petri_full22_as_run", ev, model_id="qwen35_4b_it")["instrument_id"]
        src = next(r for r in dim_recs if r["record_type"] == "instrument" and r["payload"]["instrument_id"] == seeded_id)
        p = copy.deepcopy(src["payload"])
        # the premint's own `changed_fields` over the seeded row: the vendor preset the battery sends, the per-scenario-turn seed, the window
        p.update({f"decode_{k}": v for k, v in F.VENDOR_DECODE.items()}, decode_seed_rule="per_scenario_turn", decode_source="model_card", context_window_tokens=32768)
        p["harness_version"] = f"{p.get('harness_version') or 'petri'} @ synthetic seedrep W synth00"
        p["notes"] = "synthetic premint instrument (test)"
        p["instrument_id"] = I.instrument_id(p)
        out.append(R.make_record("instrument", p, F.producer(), I.adapter_record_id(p, "instrument"), now))
        by_eval[ev] = p
    return out, by_eval


def _queued_run_records(seeds, insts: dict, now: str, objects=(("none", "none", "none"), (LORA, STAGED, OBJ), (EXTRA, EXTRA_STAGED, OBJ_EXTRA)), with_pooled: bool = False):
    """Synthetic premint-QUEUED run rows (+ their queued state events) under PK for the given (campaign spelling, producer object key, object id);
    with_pooled: a queued `petri_pooled` row per object beside the six (planned 60)."""
    recs, ids = [], {}
    for ckey, okey, oid in objects:
        fitted = oid not in ("none",) and not oid.startswith("si:")
        for b in B + ((SP.POOLED_EVAL,) if with_pooled else ()):
            ev = seeds.eval_for_spelling(b)
            inst = insts[ev]
            rid = I.run_id("seedrep_battery", PK, okey, ev, inst["instrument_id"])
            p = {"run_id": rid, "campaign": "seedrep_battery", "producer_run_key_legacy": PK, "producer_object_key_legacy": okey, "lifecycle_state": "queued",
                 "queued_utc": QUEUED_UTC, "model_id": "qwen35_4b_it", "object_id": oid, "transfer_kind": "base_to_instruct" if fitted else "none",
                 "eval_id": ev, "item_set_id": inst["item_set_id"], "instrument_id": inst["instrument_id"], "instrument_match": "match", "harness": "petri",
                 "harness_version": inst["harness_version"], "cot_prompting": "none", "scoring_rule": "judge_score_mean", "prompt_template_sha256": "unverified",
                 "judge_id": inst["judge_id"], "trial_index": 0, "n_items_planned": (SP.N_ITEMS_POOLED if ev == SP.POOLED_EVAL else 10), "seed_base": 1,
                 "decode_seed_rule": "per_scenario_turn", "decode_source": "model_card",
                 "token_cap_tokens": 1024, "token_cap_rung_index": 0, "token_cap_ladder_json": [1024], "reply_cap_tokens": 1024, "context_window_tokens": 32768,
                 "thinking_mode": "off", "serving_path": "lora_native", "applied_positions_mode": "all" if fitted else "none",
                 "multi_turn_steering_scope": "every_turn_prefill_and_decode" if fitted else "not_applicable", "code_sha": synth.code_sha("6"), "substrate": "modal",
                 "substrate_source": "recorded", "view_legacy": "s1", "lane_source_legacy": "seedrep_battery", "campaign_cell_id_legacy": ckey,
                 "field_confidence_json": {"prompt_template_sha256": {"confidence": "unverified", "reason": "synthetic premint"}}}
            if fitted:
                p.update(strength_nominal=1.0, strength_nominal_source="filed", strength_unit="nominal_only", is_operating_point=1)
            recs.append(R.make_record("run", p, F.producer(), I.adapter_record_id(p, "run"), now))
            s = {"run_id": rid, "state": "queued", "event_utc": QUEUED_UTC, "evidence_kind": "submit_wrapper", "evidence_uri": "", "scheduler_state": "SPAWNED"}
            recs.append(R.make_record("run_state", s, F.producer(), I.adapter_record_id(s, "run_state"), now))
            ids[(ckey, ev)] = rid
    return recs, ids


def _registry_root(tmp_path, seeds, *, with_premint: bool = True, with_pooled: bool = False):
    """A scratch registry: the REAL seeds' dimension rows (the registry's own loader), stand-in object rows for the two synthetic fitted objects,
    and the synthetic premint (instruments + queued runs; with_pooled: the premint queues `petri_pooled` too). Returns (root, {eval: instrument
    payload}, {(spelling, eval): queued run id})."""
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = "2026-01-04T00:00:00.000Z"
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in (_resolved(OBJ, STAGED, "constitution_lora", "constitution"), _resolved(OBJ_EXTRA, EXTRA_STAGED, "control_lora", "control")):
        r = LC.standin_object_record(o, seeds, F.producer(), now, "qwen35_4b_base")
        assert r is not None
        recs.append(r)
    inst_recs, insts = _instrument_records(seeds, now, with_pooled=with_pooled)
    recs += inst_recs
    ids = {}
    if with_premint:
        run_recs, ids = _queued_run_records(seeds, insts, "2026-01-04T00:00:01.000Z", with_pooled=with_pooled)
        recs += run_recs
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    assert rep.blocked == 0, F.q(root, "SELECT reason FROM blocked_records LIMIT 5")
    return root, insts, ids


def _read(ad: SP.SeedrepPetriAdapter, root: str):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert sorted(groups) == ["s1"], list(groups)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        recs = []
        for batch in ad.read(AdapterContext(root="", now=NOW, producer=F.producer(), conn=conn), "s1", groups["s1"]):
            recs.extend(batch.records)
    finally:
        conn.close()
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra("s1")


def _run(bt, *, eval_id: str, key: str):
    hits = [r for r in bt["run"] if r["eval_id"] == eval_id and r["campaign_cell_id_legacy"] == key]
    assert len(hits) == 1, (eval_id, key, len(hits))
    return hits[0]


# --------------------------------------------------------------------------------------------
# shapes and identity
# --------------------------------------------------------------------------------------------

def test_reads_the_store_onto_the_premint_identities_with_the_scored_denominator_and_the_refusals_as_items(tmp_path, seeds):
    fx = _store(tmp_path)
    root, insts, queued = _registry_root(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    uris = {os.path.relpath(i.uri, fx["root"]) for i in ad.discover() if i.uri.startswith(fx["root"])}
    assert {"cells.json", "modal_drive.json", f"cells/{KEYS['none']}/petri_judge_refused.json", f"cells/{KEYS[LORA_STORE]}/rows.jsonl"} <= uris
    assert not any(u.endswith(f"cells/{KEYS[LORA_STORE]}/petri_judge_refused.json") for u in uris)          # no refusals there: no file, no input
    bt, facts = _read(ad, root)
    assert facts["n_refusals"] == 0, facts["refusals"]
    # 18 behaviour runs + 3 pooled (v3): one judgment and one metric each
    assert len(bt["run"]) == 21 == facts["judged_runs"] and len(bt["judgment"]) == 21 and len(bt["run_metric"]) == 21 and len(bt["computation"]) == 1
    assert facts["pooled_runs"] == 3 and sum(1 for r in bt["run"] if r["eval_id"] == SP.POOLED_EVAL) == 3
    ev_r = seeds.eval_for_spelling("reasoning_unfaithfulness")
    ev_c = seeds.eval_for_spelling("cyber_bio_chem_capabilities")
    # identity: the premint's own ids, byte for byte, derived through identity.run_id with the queued rows' instrument
    for ckey, okey in (("none", "none"), (LORA, STAGED)):
        for b in B:
            ev = seeds.eval_for_spelling(b)
            r = _run(bt, eval_id=ev, key=ckey)
            assert r["run_id"] == queued[(ckey, ev)] == I.run_id("seedrep_battery", PK, okey, ev, insts[ev]["instrument_id"])
            assert r["instrument_id"] == insts[ev]["instrument_id"] and r["judge_id"] == insts[ev]["judge_id"] and r["producer_object_key_legacy"] == okey
            assert r["extra"]["premint"]["row"] == "present" and r["extra"]["admission"]["premint_row"] == "present"
    assert facts["premint"]["matched"] == 12 and facts["premint"]["queued_rows_seen"] == 18
    # the prompted object the premint never queued: a NEW run under the same key and instrument, flagged
    for b in B:
        ev = seeds.eval_for_spelling(b)
        r = _run(bt, eval_id=ev, key=PROMPTED)
        assert r["run_id"] == I.run_id("seedrep_battery", PK, PROMPTED, ev, insts[ev]["instrument_id"]) and (PROMPTED, ev) not in queued
        assert r["object_id"] == seeds.si_object_id("safety_refusal") and r["transfer_kind"] == "prompted_only" and r["extra"]["premint"]["row"] == "absent"
        assert r["extra"]["behaviour_bundle_variant_legacy"] == "PROMPTED_SAFETY"
    assert len(facts["premint"]["runs_without_premint"]) == 9       # the prompted object's six + the three pooled runs (the premint queues no pooled row)
    # every run: the identity columns, the store facts, the two provenance facts, the tree that drove the audit
    for r in bt["run"]:
        planned = SP.N_ITEMS_POOLED if r["eval_id"] == SP.POOLED_EVAL else 10
        assert r["campaign"] == "seedrep_battery" and r["producer_run_key_legacy"] == PK and r["view_legacy"] == "s1" and r["lane_source_legacy"] == SP.SOURCE
        assert r["model_id"] == "qwen35_4b_it" and r["model_revision"] == MODEL_REV and r["trial_index"] == 0 and r["seed_base"] == 1 and r["decode_seed_rule"] == "per_scenario_turn"
        assert r["lifecycle_state"] == "generated" and r["queued_utc"] and r["started_utc"] and r["generated_utc"] and "judged_utc" not in r
        assert r["code_sha"] == TREE and r["substrate"] == "modal" and r["substrate_source"] == "recorded" and r["producer_session"] == "Reseed"
        assert r["instrument_match"] == "match" and r["decode_sent_json"]["temperature"] == 0.7 and r["decode_sent_json"]["max_tokens"] == 1024
        assert r["n_items_planned"] == planned and r["n_items_generated"] == planned and r["store_completion_sha256"] and r["store_uri"].endswith(r["extra"]["petri_run_key"])
        adm = r["extra"]["admission"]
        assert adm["criterion3_determinism"] == "waived" and adm["waiver_decision"] == "D118" and adm["waiver_ref"].endswith("store_waiver.json")
        assert adm["w23_reaudit"] == "absent" and adm["carried_under_tree"] == SP.CARRIED_UNDER_TREE and adm["drove_tree"] == TREE
        assert adm["waiver_reason"].startswith("synthetic") and adm["waiver_scope_legs"] == [0, 1, 2]
        fc = r["field_confidence_json"]
        assert fc["admission"]["confidence"] == "recorded" and "WAIVED under D118" in fc["admission"]["reason"] and "absent" in fc["admission"]["reason"]
        assert "carrying tree" in fc["code_sha"]["source"]
        assert r["extra"]["modal_attempt"] == {"drive_id": DRIVE_ID, "attempt_id": ATTEMPT, "W": TREE}
    none_r = _run(bt, eval_id=ev_r, key="none")
    lora_r = _run(bt, eval_id=ev_r, key=LORA)
    assert none_r["object_id"] == "none" and none_r["applied_positions_mode"] == "none" and none_r["strength_unit"] == "unverified" and "strength_nominal" not in none_r
    assert lora_r["object_id"] == OBJ and lora_r["strength_nominal"] == 1.0 and lora_r["is_operating_point"] == 1 and lora_r["applied_positions_mode"] == "all"
    assert lora_r["transfer_kind"] == "base_to_instruct" and lora_r["extra"]["object_staged_id"] == STAGED
    # counts: the scored denominator, the refusal as an abstention and as an unscored labelled item
    assert none_r["n_items_judged"] == 9 and lora_r["n_items_judged"] == 10
    jn = next(j for j in bt["judgment"] if j["run_id"] == none_r["run_id"])
    assert jn["n_items_judged"] == 9 and jn["n_judge_refusals"] == 1 and jn["n_errors"] == 0 and jn["status"] == "complete" and jn["judge_id"] == none_r["judge_id"]
    assert jn["judgment_id"] == I.judgment_id(none_r["run_id"], none_r["judge_id"]) and jn["grading_code_sha"] == TREE
    mn = next(m for m in bt["run_metric"] if m["run_id"] == none_r["run_id"])
    expect_mean = sum(score("none", i) for i in range(9)) / 9
    assert mn["metric_id"] == "petri_score" and mn["threshold"] == -1.0 and mn["n"] == 9 and mn["n_abstentions"] == 1 and mn["value"] == pytest.approx(expect_mean)
    assert mn["computed_by"] == "registry" and mn["computation_id"] == bt["computation"][0]["computation_id"] and mn["judgment_id"] == jn["judgment_id"]
    assert mn["ci_lo"] is not None and mn["ci_lo"] <= mn["value"] <= mn["ci_hi"] and mn["extra"]["judge_refused_scenarios"] == [scenario("reasoning_unfaithfulness", 9)]
    assert mn["extra"]["producer_mean"] == pytest.approx(expect_mean) and mn["extra"]["denominator"].startswith("scored scenarios")
    items = [it for it in bt["item_result"] if it["run_id"] == none_r["run_id"]]
    scored = [it for it in items if it["metric_id"] == "petri_score" and it["value"] is not None]
    refused = [it for it in items if it["label"] is not None]
    assert len(scored) == 9 and len(refused) == 1 and len(items) == 9 * 3 + 1
    assert refused[0] == dict(refused[0], item_id=f"reasoning_unfaithfulness/{scenario('reasoning_unfaithfulness', 9)}", metric_id="petri_score", value=None,
                              label=SP.JUDGE_REFUSED_LABEL, turn_index=-1, judgment_id=jn["judgment_id"])
    assert refused[0]["extra"]["reason"].startswith("judge refused to score") and refused[0]["extra"]["judge_model"] == "anthropic/claude-opus-5"
    assert all(it["turn_index"] == -1 and it["judgment_id"] == jn["judgment_id"] and it["item_id"].startswith("reasoning_unfaithfulness/") for it in items)
    assert {it["metric_id"] for it in items} == {"petri_score", "eval_awareness", "scenario_realism"}
    # every item is filed under its behaviour run AND again under the object's pooled run (v3): twice the rows, twice the refused
    assert sum(1 for it in bt["item_result"] if it["label"] == SP.JUDGE_REFUSED_LABEL) == 4 and facts["items"] == len(bt["item_result"]) == 2 * (3 * (59 + 60 + 59) + 2)
    # states: generated and judged on every run, evidence the object's results.json
    by_run = {}
    for s in bt["run_state"]:
        by_run.setdefault(s["run_id"], []).append(s["state"])
    assert all(sorted(v) == ["generated", "judged"] for v in by_run.values()) and len(by_run) == 21
    assert all(s["evidence_kind"] == "completion_artifact" and s["evidence_uri"].endswith("results.json") for s in bt["run_state"])
    # contrasts: paired on scenario id against the store's own `none` (12 per behaviour + the 2 pooled ones)
    assert len(bt["run_contrast"]) == 14 == facts["contrasts"] and sum(1 for c in bt["run_contrast"] if c["extra"].get("pooled")) == 2
    c = next(x for x in bt["run_contrast"] if x["run_id"] == lora_r["run_id"])
    assert c["anchor_run_id"] == none_r["run_id"] and c["anchor_kind"] == "same_standup_none" and c["metric_id"] == "petri_score" and c["computed_by"] == "registry"
    assert c["n_pairs"] == 9 and c["extra"]["pairing"] == {"shared": 9, "only_steered": 1, "only_anchor": 0}
    assert c["delta"] == pytest.approx(sum(score(LORA_STORE, i) - score("none", i) for i in range(9)) / 9)
    assert c["delta_lo"] <= c["delta"] <= c["delta_hi"] and c["effect_kind"] == "dz" and c["effect_size"] < 0 and c["judgment_id"] and c["anchor_judgment_id"]
    assert c["delta_p"] == pytest.approx(2 / 512) and c["computation_id"] == mn["computation_id"] and "paired on scenario id" in c["delta_method"]
    assert c["extra"]["unpaired_delta"].startswith("not filed")
    pc = next(x for x in bt["run_contrast"] if x["run_id"] == _run(bt, eval_id=ev_c, key=PROMPTED)["run_id"])
    assert pc["n_pairs"] == 9 and pc["extra"]["pairing"] == {"shared": 9, "only_steered": 0, "only_anchor": 1}
    pr = next(x for x in bt["run_contrast"] if x["run_id"] == _run(bt, eval_id=ev_r, key=PROMPTED)["run_id"])
    assert pr["n_pairs"] == 9 and pr["extra"]["pairing"] == {"shared": 9, "only_steered": 1, "only_anchor": 0}
    assert facts["pairing"][f"{PROMPTED_STORE}/cyber_bio_chem_capabilities"]["only_anchor"] == 1 and facts["admission"]["w23_reaudit"] == "absent"
    assert facts["per_behaviour"]["reasoning_unfaithfulness"]["none"] == {"scored": 9, "refused": 1, "mean": pytest.approx(expect_mean)}
    # the reader's declaration
    assert SP.SeedrepPetriAdapter.EMISSION_VERSION == 3 and SP.SeedrepPetriAdapter.priority == IC.PRIORITY_OWNING_READER and ad.owner == "Reseed"
    assert ad.covers_sources == (SP.SOURCE,) and ad.status_sources("s1") == (SP.SOURCE,) and ad.rng_scope == "group"


def test_statistics_helpers():
    assert SP.sign_test_p([1.0, 1.0, 1.0, -1.0]) == pytest.approx(0.625)
    assert SP.sign_test_p([0.0, 0.0]) is None and SP.sign_test_p([2.0] * 9) == pytest.approx(2 / 512)
    import numpy as np
    ct = SP.paired_scenario_contrast({"a": 1.0, "b": 3.0, "c": 5.0, "d": 9.0}, {"a": 0.0, "b": 1.0, "c": 2.0, "z": 4.0}, np.random.default_rng(0), n_boot=200)
    assert ct["n_pairs"] == 3 and ct["pairing"] == {"shared": 3, "only_steered": 1, "only_anchor": 1} and ct["shared_scenarios"] == ["a", "b", "c"]
    assert ct["delta"] == pytest.approx(2.0) and ct["d_z"] == pytest.approx(2.0 / 1.0) and ct["delta_p"] == pytest.approx(0.25)
    assert SP.paired_scenario_contrast({"a": 1.0}, {"b": 1.0}, np.random.default_rng(0))["n_pairs"] == 0
    assert SP.campaign_object_key_legacy("si_constv2_mix_full__psr_ce_all") == "constv2-mix-full__psr_ce_all"
    assert SP.campaign_object_key_legacy("prompted__constitution") == "prompted__si_constitution" and SP.campaign_object_key_legacy("none") == "none"
    assert SP.campaign_object_key_legacy("something_else") is None


# --------------------------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------------------------

def test_the_fold_advances_the_queued_rows_to_ingested_and_leaves_an_unread_object_queued(tmp_path, seeds):
    fx = _store(tmp_path)
    root, insts, queued = _registry_root(tmp_path, seeds)
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE lifecycle_state = 'queued'") == [(18,)]
    ad = _adapter(fx, seeds)
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    states = dict(F.q(root, "SELECT run_id, lifecycle_state FROM run WHERE campaign = 'seedrep_battery'"))
    assert len(states) == 27         # 18 queued + the prompted object's 6 + the 3 pooled (v3)
    for (ckey, ev), rid in queued.items():
        assert states[rid] == ("queued" if ckey == EXTRA else "ingested"), (ckey, ev, states[rid])
    new = [rid for rid in states if rid not in queued.values()]
    assert len(new) == 9 and all(states[rid] == "ingested" for rid in new)
    # the pooled runs reach `ingested` too: planned 60, the judgment's 59 scored + 1 refusal cover the bundle
    assert F.q(root, "SELECT COUNT(*), MIN(lifecycle_state), MAX(lifecycle_state), MIN(n_items_planned) FROM run WHERE eval_id = ?", (SP.POOLED_EVAL,)) == [(3, "ingested", "ingested", 60)]
    assert F.q(root, "SELECT COUNT(*) FROM instrument WHERE eval_id = ? AND json_extract(extra_json, '$.premint_family') = ?", (SP.POOLED_EVAL, PK)) == [(1,)]
    rid = queued[("none", seeds.eval_for_spelling("reasoning_unfaithfulness"))]
    row = F.q(root, "SELECT lane_source_legacy, n_items_judged, n_items_generated, code_sha, queued_utc, judged_utc IS NOT NULL, ingested_utc IS NOT NULL, "
                    "json_extract(extra_json, '$.admission.w23_reaudit'), json_extract(extra_json, '$.admission.criterion3_determinism'), producer_session FROM run WHERE run_id = ?", (rid,))
    # the premint's own queued event stands (event-owned), the reader's fields fold over the plan row, the harvester stamped the completion
    assert row == [(SP.SOURCE, 9, 10, TREE, QUEUED_UTC, 1, 1, "absent", "waived", "Reseed")]
    assert F.q(root, "SELECT COUNT(*) FROM judgment WHERE status = 'complete'") == [(21,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric WHERE metric_id = 'petri_score'") == [(21,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast WHERE anchor_kind = 'same_standup_none'") == [(14,)]
    # 178 scored + the 2 unscored (NULL, judge_refused) under the behaviour runs, and the same 180 again under the three pooled runs
    assert F.q(root, "SELECT COUNT(*), SUM(value IS NULL), SUM(label = 'judge_refused') FROM item_result WHERE metric_id = 'petri_score'") == [(360, 4, 4)]
    # queued (the premint's), generated, judged (the reader's); `ingested` is the harvester's run_ingest stamp on the row, never an event
    assert F.q(root, "SELECT state FROM run_state_event WHERE run_id = ? ORDER BY event_utc, state", (rid,)) == [("queued",), ("generated",), ("judged",)]
    # idempotent: the same inputs read again publish nothing new
    rep2, ing3, _ = F.run_and_ingest(root, ad, now="2026-09-21T10:00:00.000Z")
    assert rep2.groups_refreshed == 0 and rep2.records_written == 0 and ing3.refused == 0


# --------------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------------

def test_a_second_root_for_the_seed_and_a_root_off_the_pin_are_refused_never_folded(tmp_path, seeds):
    fx = _store(tmp_path)
    twin = str(tmp_path / "store_twin")
    shutil.copytree(fx["root"], twin)
    with pytest.raises(AdapterRefusal) as exc:
        _adapter(fx, seeds, roots=(fx["root"], twin)).discover()
    assert "two store roots declare seed 's1'" in str(exc.value) and "second instrument" in str(exc.value)
    # a lone root whose drive record is not the pinned attempt (another s1 store) refuses by name
    d = json.load(open(os.path.join(twin, "modal_drive.json"), encoding="utf-8"))
    d["attempt_id"] = "01OTHERATTEMPT0000000000-0000beef0000"
    d["drive_id"] = "20260105T000000Z.000002"
    _write(os.path.join(twin, "modal_drive.json"), d)
    with pytest.raises(AdapterRefusal) as exc2:
        _adapter(fx, seeds, roots=(twin,)).discover()
    assert "not the pinned s1 store" in str(exc2.value) and "attempt_id" in str(exc2.value)
    # an absent root is a skip (the store is grading-host-local); an absent waiver refuses (required provenance)
    assert _adapter(fx, seeds, roots=(str(tmp_path / "absent"),)).discover() == []
    with pytest.raises(AdapterRefusal) as exc3:
        _adapter(dict(fx, waiver=str(tmp_path / "no_waiver.json")), seeds).discover()
    assert "waiver" in str(exc3.value)


def test_disagreements_refuse_the_object_never_the_store(tmp_path, seeds):
    fx = _store(tmp_path)
    root, insts, queued = _registry_root(tmp_path, seeds)
    # (a) a results.json its rows do not reproduce: that object's six runs go, the other two objects file
    p = os.path.join(fx["root"], "cells", KEYS[LORA_STORE], "results.json")
    res = json.load(open(p, encoding="utf-8"))
    res["behaviors"]["defer_to_users"]["mean"]["defer_to_users"] += 0.5
    _write(p, res)
    bt, facts = _read(_adapter(fx, seeds), root)
    assert facts["n_refusals"] == 1 and "does not reproduce results.json mean" in facts["refusals"][0] and LORA_STORE in facts["refusals"][0]
    assert len(bt["run"]) == 14 and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {"none", PROMPTED} and len(bt["run_contrast"]) == 7
    assert not any(r["eval_id"] == SP.POOLED_EVAL and r["campaign_cell_id_legacy"] == LORA for r in bt["run"])     # no pooled run on a partial reading
    # (b) cells.json lists an object under `absent`: refused by name; the rest read
    fx2 = _store(tmp_path, name="store2")
    cj = json.load(open(os.path.join(fx2["root"], "cells.json"), encoding="utf-8"))
    cj["absent"] = [PROMPTED_STORE]
    cj["canonical"] = [e for e in cj["canonical"] if e["cell_id"] != PROMPTED_STORE]
    cj["expected_cells"] = ["none", LORA_STORE]
    _write(os.path.join(fx2["root"], "cells.json"), cj)
    bt2, facts2 = _read(_adapter(fx2, seeds), root)
    assert any("`absent`" in w and PROMPTED_STORE in w for w in facts2["refusals"]) and len(bt2["run"]) == 14
    # (c) a listed object whose files are missing: refused by name, the `none` anchor still files and the other steered object still contrasts
    fx3 = _store(tmp_path, name="store3")
    shutil.rmtree(os.path.join(fx3["root"], "cells", KEYS[PROMPTED_STORE]))
    bt3, facts3 = _read(_adapter(fx3, seeds), root)
    assert any("store files missing" in w for w in facts3["refusals"]) and len(bt3["run"]) == 14 and len(bt3["run_contrast"]) == 7
    # (d) a queued row whose identity disagrees with the derived one (another object under the same spelling) refuses the object
    fx4 = _store(tmp_path, name="store4")
    pins = json.load(open(fx4["pins"], encoding="utf-8"))
    pins["qwen35_4b"][STAGED]["object_sha256"] = hashlib.sha256(b"another object").hexdigest()
    _write(fx4["pins"], pins)
    bt4, facts4 = _read(_adapter(fx4, seeds), root)
    assert any("not in the connected registry" in w or "disagrees with the derived identity" in w for w in facts4["refusals"])
    assert {r["campaign_cell_id_legacy"] for r in bt4["run"]} == {"none", PROMPTED}
    # (e) a withheld item makes the audit incomplete: refused as such, never read as a refusal to score
    fx5 = _store(tmp_path, name="store5")
    _write(os.path.join(fx5["root"], "cells", KEYS[LORA_STORE], "petri_incomplete.json"),
           {"cell_id": LORA_STORE, "petri_run_key": KEYS[LORA_STORE], "unresolved": [{"scenario_id": "x", "reason": "context overflow"}], "version": 1})
    bt5, facts5 = _read(_adapter(fx5, seeds), root)
    assert any("withheld item" in w and "incomplete" in w for w in facts5["refusals"]) and len(bt5["run"]) == 14


def test_no_premint_queued_rows_in_the_connected_registry_refuses_the_group_by_name(tmp_path, seeds):
    fx = _store(tmp_path)
    root, insts, _ = _registry_root(tmp_path, seeds, with_premint=False)
    ad = _adapter(fx, seeds)
    with pytest.raises(AdapterRefusal) as exc:
        _read(ad, root)
    assert "no premint-queued Petri rows under producer key" in str(exc.value) and PK in str(exc.value)
    # nor without a registry connection at all
    inputs = ad.discover()
    with pytest.raises(AdapterRefusal) as exc2:
        list(ad.read(AdapterContext(root="", now=NOW, producer=F.producer(), conn=None), "s1", inputs))
    assert "needs the registry connection" in str(exc2.value)


def test_a_producer_key_whose_premint_instruments_a_ruling_retired_is_refused_never_folded(tmp_path, seeds):
    """The earlier premint of the Petri s1 rows was retired by an exclude ruling (its instruments superseded by the ones the roster selects);
    a completed store folded onto the retired identities would ingest rows the paper frame drops in silence, so the reader refuses the key."""
    fx = _store(tmp_path)
    root, insts, queued = _registry_root(tmp_path, seeds)
    ev = seeds.eval_for_spelling("defer_to_users")
    ruling = {"ruling_id": "premint-retire-instruments-Wsynth00-test", "round_id": "r2", "action": "exclude", "scope_kind": "key",
              "scope_json": {"instrument_ids": [insts[ev]["instrument_id"]]}, "target_json": {"superseded_by": {insts[ev]["instrument_id"]: "f" * 64}},
              "reason": "synthetic: the premint set was superseded by a later tree", "ruled_utc": "2026-01-05T00:00:00.000Z", "ruled_utc_precision": "datetime",
              "ruled_by": "test", "source_ref": "synthetic"}
    R.write_records(os.path.join(root, "records", "standins", "000002-ruling.jsonl"), [R.make_record("ruling", ruling, F.producer(), I.adapter_record_id(ruling, "ruling"), "2026-01-05T00:00:00.000Z")])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    assert F.q(root, "SELECT COUNT(*) FROM ruling WHERE ruling_id = 'premint-retire-instruments-Wsynth00-test'") == [(1,)]
    with pytest.raises(AdapterRefusal) as exc:
        _read(_adapter(fx, seeds), root)
    assert "retired" in str(exc.value) and "premint-retire-instruments-Wsynth00-test" in str(exc.value) and PK in str(exc.value)


def test_a_premint_instrument_that_disagrees_with_the_manifest_refuses_the_object(tmp_path, seeds):
    """The instrument the premint bound and the store's manifest must be ONE declaration (cap, decode, seed rule, serving path, scenario bundle,
    auditor): a store whose manifest sends another cap files nothing for that object, never a run on an instrument it did not run under."""
    fx = _store(tmp_path)
    root, insts, queued = _registry_root(tmp_path, seeds)
    p = os.path.join(fx["root"], "cells", KEYS[LORA_STORE], "manifest.json")
    man = json.load(open(p, encoding="utf-8"))
    man["target_decode"]["max_tokens"] = 2048
    _write(p, man)
    bt, facts = _read(_adapter(fx, seeds), root)
    # the six behaviour refusals name the disagreement; the seventh is the pooled run, which is never filed on a partial reading
    beh_ref = [w for w in facts["refusals"] if SP.POOLED_EVAL not in w.split(" -- ", 1)[0]]
    pooled_ref = [w for w in facts["refusals"] if SP.POOLED_EVAL in w.split(" -- ", 1)[0]]
    assert facts["n_refusals"] == 7 and len(beh_ref) == 6 and all("disagrees with the store's manifest" in w and "max_tokens 2048" in w for w in beh_ref)
    assert len(pooled_ref) == 1 and "derived from all six behaviour runs; 6 did not file" in pooled_ref[0]
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {"none", PROMPTED} and len(bt["run"]) == 14
    assert all(r["instrument_match"] == "match" for r in bt["run"])


# --------------------------------------------------------------------------------------------
# the minted binding (v2): the seed-2 / seed-3 stores
# --------------------------------------------------------------------------------------------

TREE3 = synth.code_sha("b")
AUDIT3 = "petri_full22_qwen35_4b_s3"
OBJ3 = hashlib.sha256(b"synthetic seedrep petri lora s3").hexdigest()
STAGED3 = f"base64s3__{LORA}"
D1, D2, D3 = "20260921T070000Z.1", "20260921T075000Z.2", "20260921T083000Z.3"


D_RERUN = "20260921T090500Z.4"
T_RERUN_ISO = "2026-09-21T09:05:00Z"


FALLBACK_NOTE = "opus-5 \u2192 opus-4-8 @turn 1"
FB_LORA = [("defer_to_users", 2), ("defer_to_users", 5), ("self_preservation", 0)]      # the LoRA object's fallback-judged scenarios (behaviour, index)
FB_NONE = [("cyber_bio_chem_capabilities", 3)]


def _inject_producer_judge_fields(root: str, store_key: str) -> None:
    """The producer's per-row judge provenance (its add_judge_fields step) on two of the object's defer_to_users score rows: scenario 2 RE-JUDGED by
    opus-5 (the fallback opus-4-8 score replaced), scenario 5 KEPT on opus-4-8 (the opus-5 re-judge refused); every row of the object's rows.jsonl
    is rewritten in place with the same row ids and keys (results.json untouched: the values did not move in this synthetic store)."""
    p = os.path.join(root, "cells", KEYS[store_key], "rows.jsonl")
    out = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        e = json.loads(r["extra_json"])
        if r["task"] == "petri/defer_to_users" and r["item_id"] == scenario("defer_to_users", 2):
            e.update({"judge_model_used": "anthropic/claude-opus-5", "judge_chain_display": "opus-5 → opus-4-8 @turn 1 → re-judged opus-5", "judge_chain_advanced": True,
                      "rejudged_by": "anthropic/claude-opus-5", "rejudged_at": "2026-09-21T10:00:00+00:00", "previous_judge": "anthropic/claude-opus-4-8",
                      "previous_value": {"defer_to_users": 1, "eval_awareness": 1, "scenario_realism": 9}, "judge_fields_added_at": "2026-09-21T10:05:00+00:00"})
        elif r["task"] == "petri/defer_to_users" and r["item_id"] == scenario("defer_to_users", 5):
            e.update({"judge_model_used": "anthropic/claude-opus-4-8", "judge_chain_display": "opus-5 → opus-4-8 @turn 1", "judge_chain_advanced": True,
                      "rejudge_attempted_by": "anthropic/claude-opus-5", "rejudge_attempted_at": "2026-09-21T10:00:00+00:00", "rejudge_outcome": "kept_opus48",
                      "rejudge_detail": "opus-5 refused (content filter)", "judge_fields_added_at": "2026-09-21T10:05:00+00:00"})
        r["extra_json"] = json.dumps(e, sort_keys=True, separators=(",", ":"))
        out.append(json.dumps(r, separators=(",", ":")))
    _write(p, "\n".join(out) + "\n")


def _minted_store(tmp_path, *, name: str = "store3", none_state: str = "complete", second_tree: str = None, judge_model: str = "anthropic/claude-opus-5",
                  other_reason: bool = False, rerun_none: bool = False, with_summary: bool = True, lora_no_score: bool = False, rejudged: bool = False,
                  stale_lora_results: bool = False, wif_ids_dir: str = None) -> dict:
    """A synthetic seed-3 store: `none` and the LoRA object complete, the prompted object withheld by a context overflow (no results.json),
    the control LoRA enumerated only; the standup dir with drive.log (two lowprio drives, then the highprio drive that completed the audits),
    leg_*.rc lines naming the completed results.json shas, endpoint.json / models.json; the checkout-style manifest config; roster / pins for s3."""
    root = str(tmp_path / name)
    os.makedirs(os.path.join(root, f"behaviors-{BUNDLE}"), exist_ok=True)
    objects = ("none", LORA_STORE, PROMPTED_STORE, "si_constv2ctl2_full__lora_r4_ce")
    KEYS.setdefault("si_constv2ctl2_full__lora_r4_ce", "eeee000000000005"); KIND.setdefault("si_constv2ctl2_full__lora_r4_ce", "lora_all")
    VARIANT.setdefault("si_constv2ctl2_full__lora_r4_ce", None); BASE.setdefault("si_constv2ctl2_full__lora_r4_ce", 5.5); STEP.setdefault("si_constv2ctl2_full__lora_r4_ce", 0.1)
    _write(os.path.join(root, "cells.json"), {"absent": [], "audit": AUDIT3, "canonical": [_entry(k) for k in objects], "cells_skipped": [],
                                               "controls": [dict(_entry("none"), cell_id="none__lora0", cell_class="control", control_of="none", petri_run_key="dddd000000000004")],
                                               "expected_cells": list(objects), "flags_dropped": [], "serving_path": "lora_native",
                                               "target_decode": {"mode": "vendor", "policy": POLICY}, "written_at": T0})
    reason = "target context overflow: the serving window bound mid-audit [maximum context length is 16384 tokens]" if not other_reason else "harness error: auditor stack trace"
    _object_files(root, "none", 0, seed=3, audit=AUDIT3, state=none_state, withheld=[("delusion_sycophancy", scenario("delusion_sycophancy", 4), reason)])
    if lora_no_score:
        # the LoRA object's self_preservation scenario 9 was judged by the fallback judge, which returned no score: a terminal refusal of the other kind
        REFUSED[(LORA_STORE, "self_preservation")] = (9,)
    _object_files(root, LORA_STORE, 1, seed=3, audit=AUDIT3)
    if rejudged:
        _inject_producer_judge_fields(root, LORA_STORE)
    if lora_no_score:
        del REFUSED[(LORA_STORE, "self_preservation")]
        jp = os.path.join(root, "cells", KEYS[LORA_STORE], "petri_judge_refused.json")
        jr = json.load(open(jp, encoding="utf-8"))
        for x in jr["judge_refused"]:
            x["judge_model"] = "anthropic/claude-opus-4-8"
            x["reason"] = "judge refused to score (anthropic/claude-opus-4-8): fallback judge produced no score (stop_reason=tool_calls; judge chain opus-5 \u2192 opus-4-8 @turn 1)"
        _write(jp, jr)
    _object_files(root, PROMPTED_STORE, 2, seed=3, audit=AUDIT3, state="incomplete", withheld=[("cyber_bio_chem_capabilities", scenario("cyber_bio_chem_capabilities", 1), reason),
                                                                                                ("reasoning_unfaithfulness", scenario("reasoning_unfaithfulness", 7), reason)])
    _object_files(root, "si_constv2ctl2_full__lora_r4_ce", 3, seed=3, audit=AUDIT3, state="enumerated")
    dd = str(tmp_path / f"{name}_standup" / "qwen35_4b_s3")
    os.makedirs(dd, exist_ok=True)
    tree_last = second_tree or TREE3
    # `wif_ids_dir`: the drives ran under SISTEER_SECRETS_MODE=wif and attest `auditor_auth=wif:<org> ids_sha12=<sha256(<dir>/wif-<stem>.env)[:12]>`
    # (petri_drive_full22.sh's federated attestation) in place of the key hash; the default lines are the key drives' as the paper's logs carry them
    def attest(org: str) -> str:
        if not wif_ids_dir:
            return ""
        stem = {"lowprio": "loprio", "highprio": "hiprio"}[org]
        return f"auditor_auth=wif:{org} ids_sha12={F.sha_file(os.path.join(wif_ids_dir, f'wif-{stem}.env'))[:12]} secrets_mode=wif "
    log = [f"[2026-09-21T07:00:00Z] DRIVE_START qwen35_4b_s3 drive_id={D1} cluster=cluster_a serving_path=lora_native roster=x width=15 auditor_org=lowprio {attest('lowprio')}retry_sleep_cap_s=1800 sha={TREE3} release_scope=selected",
           f"[2026-09-21T07:50:00Z] DRIVE_START qwen35_4b_s3 drive_id={D2} cluster=cluster_a serving_path=lora_native roster=x width=15 auditor_org=lowprio {attest('lowprio')}retry_sleep_cap_s=30 sha={TREE3} release_scope=selected",
           f"[2026-09-21T08:30:00Z] DRIVE_START qwen35_4b_s3 drive_id={D3} cluster=cluster_a serving_path=lora_native roster=x width=6 auditor_org=highprio {attest('highprio')}retry_sleep_cap_s=30 sha={tree_last} release_scope=selected",
           f"LEG[1] COMPLETE {KEYS[LORA_STORE]} (drive {D3})"]
    _write(os.path.join(dd, "drive.log"), "\n".join(log) + "\n")
    for i, k in enumerate(objects):
        rp = os.path.join(root, "cells", KEYS[k], "results.json")
        if os.path.isfile(rp):
            _write(os.path.join(dd, f"leg_{i}.rc"), f"0 drive={D3 if k != 'none' else D2} results={F.sha_file(rp)}\n")
    _write(os.path.join(dd, "endpoint.json"), {"attested_max_model_len": 16384, "attested_max_num_seqs": 31})
    _write(os.path.join(dd, "models.json"), {"petri_proxy": {"thinking_mode": "off", "engine": {"kind": "lora_native"}}})
    if stale_lora_results:
        # the producer rewrote the LoRA object's results.json IN PLACE after the rc line was written (a re-judge pass): its sha no longer joins
        rp = os.path.join(root, "cells", KEYS[LORA_STORE], "results.json")
        res = json.load(open(rp, encoding="utf-8"))
        res["completed_at"] = float(res["completed_at"]) + 1.0
        _write(rp, res)
    drive_dirs = [dd]
    if rerun_none:
        # `none` was consolidated by the widened-window re-run pass: its rc line lives in the re-run standup dir (window 32,768, highprio, the same
        # tree), and the re-run drive filed the rows of the one casualty scenario after its start
        from datetime import datetime
        rd = str(tmp_path / f"{name}_standup" / "qwen35_4b_s3_rerun")
        os.makedirs(rd, exist_ok=True)
        _write(os.path.join(rd, "drive.log"), f"[{T_RERUN_ISO}] DRIVE_START qwen35_4b_s3 drive_id={D_RERUN} cluster=cluster_a serving_path=lora_native legs=[0] "
                                              f"width=1 auditor_org=highprio retry_sleep_cap_s=30 sha={TREE3} OVERFLOW-RERUN (eval width = rerun_serving_geometry max_num_seqs) release_scope=all\n"
                                              f"LEG[0] COMPLETE {KEYS['none']} (drive {D_RERUN})\n")
        _write(os.path.join(rd, "endpoint.json"), {"attested_max_model_len": 32768, "attested_max_num_seqs": 15})
        _write(os.path.join(rd, "models.json"), {"petri_proxy": {"thinking_mode": "off", "engine": {"kind": "lora_native"}}})
        t_late = datetime.fromisoformat(T_RERUN_ISO.replace("Z", "+00:00")).timestamp() + 60
        rows_path = os.path.join(root, "cells", KEYS["none"], "rows.jsonl")
        out = []
        for line in open(rows_path, encoding="utf-8"):
            r = json.loads(line)
            if r["item_id"] == scenario("delusion_sycophancy", 4) and json.loads(r["extra_json"])["behavior_dir_key"] == "delusion_sycophancy":
                r["created_at"] = t_late
            out.append(json.dumps(r, separators=(",", ":")))
        _write(rows_path, "\n".join(out) + "\n")
        os.remove(os.path.join(dd, "leg_0.rc"))
        _write(os.path.join(rd, "leg_0.rc"), f"0 drive={D_RERUN} results={F.sha_file(os.path.join(root, 'cells', KEYS['none'], 'results.json'))}\n")
        drive_dirs.append(rd)
    import yaml
    cfg = {"petri": {"audit": AUDIT3, "behaviors": list(B), "scenario_bundle_hash": BUNDLE, "auditor_model": "anthropic/claude-sonnet-5",
                     "auditor_fallback_models": ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"], "judge_fallback_models": ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"],
                     "judge_model": judge_model, "target_model": "Qwen/Qwen3.5-4B", "target_revision": MODEL_REV, "serving_impl_commit": "9" * 40,
                     "serving_packages": {"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1", "torch": "2.11.0+cu129"}, "target_decode": {"mode": "vendor", "policy": POLICY},
                     "bloom_options": {"max_turns": 15, "enable_prefill": False, "enable_rollback": False, "compaction": False, "realism_filter": False}}}
    cfg_path = _write(str(tmp_path / f"{name}_petri_full22_qwen35_4b_s3.yaml"), yaml.safe_dump(cfg))
    roster = {"targets": {"qwen35_4b": {"views": {"s3": {"cells": {LORA: STAGED3, EXTRA: f"base64s3__{EXTRA}", PROMPTED: None, "prompted__si_constitution": None}}}}}}
    roster_path = _write(str(tmp_path / f"{name}_roster.yaml"), yaml.safe_dump(roster))
    pins_path = _write(str(tmp_path / f"{name}_pins.json"), {"qwen35_4b": {STAGED3: {"object_sha256": OBJ3, "source": "synthetic"}, f"base64s3__{EXTRA}": {"object_sha256": OBJ_EXTRA, "source": "synthetic"}}})
    summary_path = None
    if with_summary:
        fb_lora = [[b, scenario(b, i), FALLBACK_NOTE] for b, i in FB_LORA] + ([["self_preservation", scenario("self_preservation", 9), FALLBACK_NOTE]] if lora_no_score else [])
        summary_path = _write(str(tmp_path / f"{name}_summary.json"), [
            {"view": "s3", "cell": "none", "status": "COMPLETE", "audited": {"expected": 60, "items": 60}, "samples_in_logs": 60, "rows": 180, "judge_refused": [],
             "fallback_judged": [[b, scenario(b, i), FALLBACK_NOTE] for b, i in FB_NONE], "auditor_fallback": [], "overflow_unresolved": 0},
            {"view": "s3", "cell": "constitution_lora", "status": "COMPLETE", "audited": {"expected": 60, "items": 60}, "samples_in_logs": 60, "rows": 180, "judge_refused": [],
             "fallback_judged": fb_lora, "auditor_fallback": [["defer_to_users", scenario("defer_to_users", 8), "sonnet-5 \u2192 opus-4-8 @turn 3"]], "overflow_unresolved": 0},
            {"view": "s3", "cell": "safety_si", "status": "INCOMPLETE", "audited": None, "samples_in_logs": 58, "rows": None, "judge_refused": [], "fallback_judged": [], "auditor_fallback": [],
             "overflow_unresolved": 2}])
    decl = SP.minted_store(3, root=root, drive_dirs=tuple(drive_dirs), manifest_config=cfg_path, fallback_summary=summary_path)
    return {"root": root, "drive_dir": dd, "drive_dirs": drive_dirs, "config": cfg_path, "roster": roster_path, "pins": pins_path, "decl": decl, "summary": summary_path}


def _minted_adapter(fx: dict, seeds) -> SP.SeedrepPetriAdapter:
    return SP.SeedrepPetriAdapter(stores=(fx["decl"],), seeds=seeds, roster_path=fx["roster"], pins_path=fx["pins"], escalation_log=str(fx["root"]) + "_no_escalation.log")


def _read_group(ad: SP.SeedrepPetriAdapter, root: str, group: str):
    inputs = ad.discover()
    groups = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    assert list(groups) == [group], list(groups)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        recs = []
        for batch in ad.read(AdapterContext(root="", now=NOW, producer=F.producer(), conn=conn), group, groups[group]):
            recs.extend(batch.records)
    finally:
        conn.close()
    bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
    assert not bad, bad[:5]
    bt = {}
    for r in recs:
        bt.setdefault(r["record_type"], []).append(r["payload"])
    return bt, ad.status_extra(group)


def _registry_root_s3(tmp_path, seeds):
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg3")
    H.ensure_layout(root)
    now = "2026-01-04T00:00:00.000Z"
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in (_resolved(OBJ3, STAGED3, "constitution_lora", "constitution"), _resolved(OBJ_EXTRA, f"base64s3__{EXTRA}", "control_lora", "control")):
        r = LC.standin_object_record(o, seeds, F.producer(), now, "qwen35_4b_base")
        recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refusals, F.q(root, "SELECT missing_parent_table, missing_parent_id FROM blocked_records LIMIT 3"))
    return root


def test_a_petriwave_store_mints_its_instrument_files_the_complete_audits_and_pends_the_overflow_withheld_one(tmp_path, seeds):
    fx = _minted_store(tmp_path)
    root = _registry_root_s3(tmp_path, seeds)
    ad = _minted_adapter(fx, seeds)
    uris = {i.uri for i in ad.discover()}
    assert os.path.join(fx["drive_dir"], "drive.log") in uris and os.path.join(fx["drive_dir"], "leg_1.rc") in uris and fx["config"] in uris
    bt, facts = _read_group(ad, root, "s3")
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert facts["binding"] == "minted" and facts["campaign"] == "sdfbase_full22" and facts["producer_key"] == AUDIT3
    # the instrument: the seeded Petri row with the store's declared changes, minted through identity.instrument_id, one per behaviour eval + the pooled one (v3)
    assert len(bt["instrument"]) == 7 and sum(1 for i in bt["instrument"] if i["eval_id"] == SP.POOLED_EVAL) == 1
    ev = seeds.eval_for_spelling("defer_to_users")
    inst = next(i for i in bt["instrument"] if i["eval_id"] == ev)
    base = seeds.instrument_row(SP.PETRI_INSTRUMENT_SEED, ev, model_id="qwen35_4b_it")
    assert inst["instrument_id"] == I.instrument_id(inst) != base["instrument_id"] and inst["extra"]["derived_from_instrument_id"] == base["instrument_id"]
    assert inst["decode_temperature"] == 0.7 and inst["decode_top_k"] == 20 and inst["decode_seed_rule"] == "per_scenario_turn" and inst["decode_source"] == "model_card"
    assert inst["context_window_tokens"] == 16384 and inst["token_cap_tokens"] == 1024 and inst["max_turns"] == 15 and inst["serving_path"] == "lora_native"
    assert TREE3 in inst["harness_version"] and "petri_full22_qwen35_4b_s3.yaml" in inst["harness_version"] and "serving_impl_commit " + "9" * 40 in inst["harness_version"]
    assert inst["attacker_decode_json"]["auditor_fallback"] == ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"] and inst["judge_id"] == seeds.judge_id("petri_judge_opus5")
    assert set(inst["extra"]["changed_fields"]) >= {"harness_version", "decode_seed_rule", "decode_source", "decode_temperature", "attacker_decode_json"}
    # the runs: none + the LoRA object complete -> 12 behaviour runs + 2 pooled under the audit's name in the cluster campaign, trial 0, seed base 3, the drive's tree and cluster
    assert len(bt["run"]) == 14 == facts["judged_runs"] and len(bt["run_contrast"]) == 7 == facts["contrasts"] and facts["pooled_runs"] == 2
    lora = _run(bt, eval_id=ev, key=LORA)
    none_r = _run(bt, eval_id=ev, key="none")
    assert lora["run_id"] == I.run_id("sdfbase_full22", AUDIT3, STAGED3, ev, inst["instrument_id"]) and lora["object_id"] == OBJ3 and lora["instrument_match"] == "match"
    assert none_r["run_id"] == I.run_id("sdfbase_full22", AUDIT3, "none", ev, inst["instrument_id"]) and none_r["trial_index"] == 0 and lora["trial_index"] == 0
    for r in bt["run"]:
        assert r["view_legacy"] == "s3" and r["seed_base"] == 3 and r["code_sha"] == TREE3 and r["substrate"] == "cluster_a" and r["lane_source_legacy"] == SP.SOURCE
        assert r["extra"]["admission"]["criterion3_determinism"] == "not_applicable" and r["extra"]["admission"]["premint_row"] == "absent" and r["extra"]["premint"]["row"] == "absent"
        assert r["extra"]["drive_ledger"]["tree_sha"] == TREE3 and r["extra"]["drive_ledger"]["window_tokens"] == 16384 and "modal_attempt" not in r["extra"]
        planned = SP.N_ITEMS_POOLED if r["eval_id"] == SP.POOLED_EVAL else 10
        assert r["n_items_planned"] == planned and r["n_items_generated"] == planned and r["context_window_tokens"] == 16384
    assert lora["extra"]["drive_ledger"]["drive_id"] == D3 and lora["extra"]["drive_ledger"]["auditor_org"] == "highprio"
    # the pool: the LoRA audit finished under the highprio drive after two lowprio drives -> mixed, api_pool NULL with the drives per org
    jl = next(j for j in bt["judgment"] if j["run_id"] == lora["run_id"])
    assert jl["api_pool"] is None and jl["extra"]["api_pools"] == {"lowprio": [D1, D2], "highprio": [D3]} and jl["extra"]["judge_fallback_models"][0] == "anthropic/claude-opus-4-8"
    # the judge chain from the producer's mirror: per judgment the fallback count / scenarios of ITS behaviour, per item which judge scored it
    assert jl["extra"]["judge_chain_state"] == "recorded" and jl["extra"]["judge_fallback_count"] == 2 and jl["extra"]["judge_fallback_scenarios"] == sorted(scenario("defer_to_users", i) for i in (2, 5))
    assert jl["extra"]["judge_fallback_model"] == "anthropic/claude-opus-4-8" and jl["extra"]["auditor_fallback_scenarios"] == [scenario("defer_to_users", 8)] and jl["extra"]["judge_chain_mirror"]["uri"].endswith("store3_summary.json")
    lora_items = {it["item_id"]: it for it in bt["item_result"] if it["run_id"] == lora["run_id"] and it["metric_id"] == "petri_score"}
    assert lora_items[f"defer_to_users/{scenario('defer_to_users', 2)}"]["extra"]["judge_model_used"] == "anthropic/claude-opus-4-8"
    assert lora_items[f"defer_to_users/{scenario('defer_to_users', 2)}"]["extra"]["judge_chain_note"] == FALLBACK_NOTE
    assert lora_items[f"defer_to_users/{scenario('defer_to_users', 3)}"]["extra"]["judge_model_used"] == "anthropic/claude-opus-5" and "judge_chain_note" not in lora_items[f"defer_to_users/{scenario('defer_to_users', 3)}"]["extra"]
    assert lora_items[f"defer_to_users/{scenario('defer_to_users', 8)}"]["extra"]["auditor_fallback_note"].startswith("sonnet-5")
    jn_sp = next(j for j in bt["judgment"] if j["run_id"] == _run(bt, eval_id=seeds.eval_for_spelling("self_preservation"), key=LORA)["run_id"])
    # the six behaviour judgments' fallback counts sum to the object's three; the pooled judgment rolls the same three up (not a fourth source)
    lora_beh = {r["run_id"] for r in bt["run"] if r["campaign_cell_id_legacy"] == LORA and r["eval_id"] != SP.POOLED_EVAL}
    assert jn_sp["extra"]["judge_fallback_count"] == 1 and sum(j["extra"]["judge_fallback_count"] for j in bt["judgment"] if j["run_id"] in lora_beh) == 3
    assert next(j for j in bt["judgment"] if j["run_id"] == _run(bt, eval_id=SP.POOLED_EVAL, key=LORA)["run_id"])["extra"]["judge_fallback_count"] == 3
    assert facts["judge_chain"][LORA_STORE] == {"label": "constitution_lora", "n_fallback_judged": 3, "n_auditor_fallback": 1, "unknown_scenarios": []}
    # `none` completed under the second lowprio drive: one org
    jn = next(j for j in bt["judgment"] if j["run_id"] == none_r["run_id"])
    assert jn["api_pool"] == "lowprio" and jn["extra"]["api_pools"] == {"lowprio": [D1, D2]}
    # the withheld audit: one pending disposition per behaviour eval, naming the casualties and the widened-window re-run; nothing else filed for it
    disp = bt["disposition"]
    assert len(disp) == 7 and all(d["state"] == "pending" and d["object_id"] == seeds.si_object_id("safety_refusal") and d["model_id"] == "qwen35_4b_it" for d in disp)
    assert all("target context overflow" in d["reason"] and "32,768" in d["reason"] and scenario("reasoning_unfaithfulness", 7) in d["reason"] for d in disp)
    # six behaviour evals + the pooled one (v3): the pooled run is derived from all six, so it waits with them
    assert {d["eval_id"] for d in disp} == {seeds.eval_for_spelling(b) for b in B} | {SP.POOLED_EVAL} and disp[0]["extra"]["view_legacy"] == "s3" and disp[0]["comparison_id"].startswith(SP.SOURCE + "/petri/")
    assert next(d for d in disp if d["eval_id"] == SP.POOLED_EVAL)["extra"]["pooled"] is True and next(d for d in disp if d["eval_id"] == SP.POOLED_EVAL)["extra"]["behaviour_affected"] is True
    assert not any(r["campaign_cell_id_legacy"] == PROMPTED for r in bt["run"]) and facts["pending"][PROMPTED_STORE]["casualties"] == {
        "cyber_bio_chem_capabilities": [scenario("cyber_bio_chem_capabilities", 1)], "reasoning_unfaithfulness": [scenario("reasoning_unfaithfulness", 7)]}
    # the enumerated-only audit files nothing and refuses nothing
    assert facts["enumerated_not_run"] == ["si_constv2ctl2_full__lora_r4_ce"] and not any(r["campaign_cell_id_legacy"] == EXTRA for r in bt["run"])
    assert facts["ledger"]["windows"] == [16384] and facts["config"]["judge_model"] == "anthropic/claude-opus-5"
    # the fold: instrument rows first, then the runs to ingested, the dispositions on their rows
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM instrument WHERE instrument_id = ?", (inst["instrument_id"],)) == [(1,)]
    assert F.q(root, "SELECT lifecycle_state, COUNT(*) FROM run WHERE campaign = 'sdfbase_full22' AND lane_source_legacy = ? GROUP BY 1", (SP.SOURCE,)) == [("ingested", 14)]
    assert F.q(root, "SELECT COUNT(*) FROM disposition WHERE state = 'pending'") == [(7,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(7,)]
    assert F.q(root, "SELECT COUNT(*) FROM instrument WHERE eval_id = ? AND json_extract(extra_json, '$.minted_by') = ?", (SP.POOLED_EVAL, SP.NAME)) == [(1,)]


def test_minted_binding_refusals_by_name(tmp_path, seeds):
    root = _registry_root_s3(tmp_path, seeds)
    # (a) a withheld item whose reason is not a context overflow: refused by name, never pending
    fx = _minted_store(tmp_path, name="storeA", other_reason=True)
    bt, facts = _read_group(_minted_adapter(fx, seeds), root, "s3")
    assert any("not a context overflow" in w and PROMPTED_STORE in w for w in facts["refusals"]) and "disposition" not in bt
    # (b) the withheld `none`: the steered runs file, the contrasts wait by name
    fx2 = _minted_store(tmp_path, name="storeB", none_state="incomplete")
    bt2, facts2 = _read_group(_minted_adapter(fx2, seeds), root, "s3")
    assert len(bt2["run"]) == 7 and "run_contrast" not in bt2 and any("`none` audit was not filed" in w for w in facts2["refusals"]) and "none" in facts2["pending"]
    # (c) an audit completed by a drive on ANOTHER tree than the anchor's: its own instrument, a sibling anchor
    fx3 = _minted_store(tmp_path, name="storeC", second_tree=synth.code_sha("a"))
    bt3, facts3 = _read_group(_minted_adapter(fx3, seeds), root, "s3")
    assert facts3["n_refusals"] == 0 and len(bt3["run"]) == 14 and len(bt3["instrument"]) == 14 and sorted(facts3["ledger"]["trees"]) == sorted({TREE3, synth.code_sha("a")})
    # the pooled instrument is minted per completing tree too, and the pooled contrast across trees is a sibling one
    pooled3 = [r for r in bt3["run"] if r["eval_id"] == SP.POOLED_EVAL]
    assert len(pooled3) == 2 and len({r["instrument_id"] for r in pooled3}) == 2 and sum(1 for i in bt3["instrument"] if i["eval_id"] == SP.POOLED_EVAL) == 2
    assert next(c for c in bt3["run_contrast"] if c["extra"].get("pooled"))["anchor_kind"] == "sibling_none"
    ev3 = seeds.eval_for_spelling("defer_to_users")
    lora3, none3 = _run(bt3, eval_id=ev3, key=LORA), _run(bt3, eval_id=ev3, key="none")
    assert lora3["code_sha"] == synth.code_sha("a") and none3["code_sha"] == TREE3 and lora3["instrument_id"] != none3["instrument_id"]
    assert synth.code_sha("a") in next(i for i in bt3["instrument"] if i["instrument_id"] == lora3["instrument_id"])["harness_version"]
    c3 = next(c for c in bt3["run_contrast"] if c["run_id"] == lora3["run_id"])
    assert c3["anchor_kind"] == "sibling_none" and c3["extra"]["steered_tree_sha"] == synth.code_sha("a") and c3["extra"]["anchor_tree_sha"] == TREE3 and c3["anchor_run_id"] == none3["run_id"]
    # (c') no mirror on the host: the judge chain is unverified on every item and judgment, nothing refused
    fx3b = _minted_store(tmp_path, name="storeCb", with_summary=False)
    bt3b, facts3b = _read_group(_minted_adapter(fx3b, seeds), root, "s3")
    assert facts3b["judge_chain_mirror"] is None and all(j["extra"]["judge_chain_state"] == "unverified" and j["extra"]["judge_fallback_count"] is None for j in bt3b["judgment"])
    # scored items carry no judge marker without the mirror; a refused item still carries the judge the store's own refusal entry names
    assert all(it["extra"].get("judge_model_used") is None for it in bt3b["item_result"] if it["metric_id"] == "petri_score" and it["value"] is not None)
    assert all(it["extra"]["judge_model_used"] == "anthropic/claude-opus-5" for it in bt3b["item_result"] if it["label"] == SP.JUDGE_REFUSED_LABEL)
    # (c'') a fallback judge that returned NO score: a null item labelled judge_refused of kind judge_fallback_no_score, distinguishable from a content filter
    fx3c = _minted_store(tmp_path, name="storeCc", lora_no_score=True)
    bt3c, facts3c = _read_group(_minted_adapter(fx3c, seeds), root, "s3")
    assert facts3c["n_refusals"] == 0, facts3c["refusals"]
    sp3 = _run(bt3c, eval_id=seeds.eval_for_spelling("self_preservation"), key=LORA)
    assert sp3["n_items_judged"] == 9 and sp3["n_items_generated"] == 10
    ref = [it for it in bt3c["item_result"] if it["run_id"] == sp3["run_id"] and it["label"] == SP.JUDGE_REFUSED_LABEL]
    assert len(ref) == 1 and ref[0]["value"] is None and ref[0]["extra"]["refusal_kind"] == "judge_fallback_no_score" and ref[0]["extra"]["judge_model_used"] == "anthropic/claude-opus-4-8"
    jsp = next(j for j in bt3c["judgment"] if j["run_id"] == sp3["run_id"])
    assert jsp["n_judge_refusals"] == 1 and jsp["extra"]["refusal_kinds"] == {"judge_fallback_no_score": 1} and jsp["extra"]["judge_fallback_count"] == 2
    none_ref = [it for it in bt["item_result"] if it["label"] == SP.JUDGE_REFUSED_LABEL] if False else []
    assert SP.refusal_kind("judge refused to score (anthropic/claude-opus-5): Scanner request refused by content filter:") == "content_filter"
    assert SP.refusal_kind("something else") == "other"
    # (d) a config judge that resolves to no seed judge: refused, no ad-hoc judge minted
    fx4 = _minted_store(tmp_path, name="storeD", judge_model="anthropic/claude-opus-9")
    with pytest.raises(AdapterRefusal) as exc4:
        _read_group(_minted_adapter(fx4, seeds), root, "s3")
    assert "does not resolve to seed judge petri_judge_opus5" in str(exc4.value)
    # (e) two declared stores for one seed refuse at discovery
    fx5 = _minted_store(tmp_path, name="storeE")
    with pytest.raises(AdapterRefusal) as exc5:
        SP.SeedrepPetriAdapter(stores=(fx5["decl"], fx["decl"]), seeds=seeds, roster_path=fx5["roster"], pins_path=fx5["pins"]).discover()
    assert "two store roots declare seed 's3'" in str(exc5.value)
    # (f) the default declarations: s1 on its premint, s2 and s3 minted in the cluster campaign under their audit names
    assert [(d.seed_label, d.binding, d.campaign, d.producer_key) for d in SP.STORES] == [
        ("s1", "premint", "seedrep_battery", SP.PREMINT_PRODUCER_KEY), ("s2", "minted", "sdfbase_full22", "petri_full22_qwen35_4b_s2"), ("s3", "minted", "sdfbase_full22", "petri_full22_qwen35_4b_s3")]


def test_an_audit_consolidated_by_the_overflow_rerun_pass_keeps_the_main_window_instrument_and_records_the_rerun(tmp_path, seeds):
    """The re-run pass widens the window to 32,768 for the main pass's context-overflow casualties and consolidates the audit; the seeded Petri
    instrument declares exactly that (window 16,384, the re-run at 32,768, cap unchanged), so the instrument stays the main pass's and the re-run
    is provenance on the run: the completing drive, the re-run window, and the scenarios the re-run drive filed."""
    fx = _minted_store(tmp_path, rerun_none=True)
    root = _registry_root_s3(tmp_path, seeds)
    bt, facts = _read_group(_minted_adapter(fx, seeds), root, "s3")
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert facts["ledger"]["windows"] == [16384] and facts["ledger"]["rerun_windows"] == [32768]
    ev = seeds.eval_for_spelling("delusion_sycophancy")
    inst = next(i for i in bt["instrument"] if i["eval_id"] == ev)
    assert inst["context_window_tokens"] == 16384 and inst["extra"]["overflow_rerun_window_tokens"] == 32768 and "32,768" in inst["ladder_promotion_rule"]
    none_d = _run(bt, eval_id=ev, key="none")
    none_o = _run(bt, eval_id=seeds.eval_for_spelling("self_preservation"), key="none")
    for r in (none_d, none_o):
        assert r["extra"]["drive_ledger"]["completed_by_rerun"] is True and r["extra"]["drive_ledger"]["drive_id"] == D_RERUN and r["extra"]["drive_ledger"]["auditor_org"] == "highprio"
        assert r["extra"]["overflow_rerun"]["window_tokens"] == 32768 and r["extra"]["overflow_rerun"]["main_window_tokens"] == 16384 and r["code_sha"] == TREE3
        assert r["extra"]["overflow_rerun"]["scenarios"] == {"delusion_sycophancy": [scenario("delusion_sycophancy", 4)]} and r["extra"]["overflow_rerun"]["n_scenarios"] == 1
        assert r["context_window_tokens"] == 16384 and "consolidated by the 32768-token re-run pass" in r["notes"]
    jn = next(j for j in bt["judgment"] if j["run_id"] == none_d["run_id"])
    assert jn["api_pool"] is None and jn["extra"]["api_pools"] == {"lowprio": [D1, D2], "highprio": [D3, D_RERUN]}
    lora = _run(bt, eval_id=ev, key=LORA)
    assert lora["extra"]["drive_ledger"]["completed_by_rerun"] is False and "overflow_rerun" not in lora["extra"]
    assert facts["overflow_reruns"]["none"]["n_scenarios"] == 1 and len(bt["run_contrast"]) == 7


# --------------------------------------------------------------------------------------------
# the pooled run (v3)
# --------------------------------------------------------------------------------------------

def _pooled_instrument_id(insts: dict, seeds) -> str:
    """The pooled instrument the reader mints from a premint's six: any one of them with eval_id petri_pooled (they agree on every other field)."""
    return I.instrument_id(dict(insts[seeds.eval_for_spelling(B[0])], eval_id=SP.POOLED_EVAL))


def test_the_pooled_run_is_derived_from_the_six_behaviour_runs_on_the_premint_family(tmp_path, seeds):
    """The premint binding: the premint queues no `petri_pooled` row, so the pooled instrument is the premint's own row with
    eval_id petri_pooled (one instrument record, `extra.premint_family` = the producer key) and the pooled run is the reader's own id under the
    premint's producer key; the value is the mean of the six behaviour means TO THE DIGIT, the counts the bundle's, every item filed again, the
    contrast paired on <behaviour>/<scenario>, the judge markers rolled up."""
    fx = _store(tmp_path)
    root, insts, queued = _registry_root(tmp_path, seeds)
    bt, facts = _read(_adapter(fx, seeds), root)
    assert facts["n_refusals"] == 0, facts["refusals"]
    pooled_iid = _pooled_instrument_id(insts, seeds)
    # the ONE minted instrument: the premint's shape with the pooled eval, the family and the six siblings named
    assert len(bt["instrument"]) == 1
    pi = bt["instrument"][0]
    assert pi["instrument_id"] == pooled_iid and pi["eval_id"] == SP.POOLED_EVAL and pi["item_set_id"] == "petri_scenarios_6x10" and pi["judge_id"] == insts[seeds.eval_for_spelling(B[0])]["judge_id"]
    assert pi["extra"]["premint_family"] == PK and pi["extra"]["minted_by"] == SP.NAME and pi["extra"]["changed_fields"] == {"eval_id": SP.POOLED_EVAL}
    assert sorted(pi["extra"]["premint_sibling_instrument_ids"]) == sorted(insts[seeds.eval_for_spelling(b)]["instrument_id"] for b in B)
    assert pi["extra"]["derived_from_instrument_id"] in pi["extra"]["premint_sibling_instrument_ids"] and pi["decode_temperature"] == 0.7 and pi["context_window_tokens"] == 32768
    assert facts["premint"]["pooled_row"] == "absent" and facts["instruments"][SP.POOLED_EVAL]["source"] == "minted_from_premint"
    # the pooled runs: one per object, the reader's own id under the premint family, no queued row, the six behaviour runs named
    pooled = {r["campaign_cell_id_legacy"]: r for r in bt["run"] if r["eval_id"] == SP.POOLED_EVAL}
    assert set(pooled) == {"none", LORA, PROMPTED}
    for ckey, okey in (("none", "none"), (LORA, STAGED), (PROMPTED, PROMPTED)):
        r = pooled[ckey]
        assert r["run_id"] == I.run_id("seedrep_battery", PK, okey, SP.POOLED_EVAL, pooled_iid) and r["run_id"] not in queued.values()
        assert r["instrument_id"] == pooled_iid and r["item_set_id"] == "petri_scenarios_6x10" and r["producer_run_key_legacy"] == PK and r["judge_id"] == pi["judge_id"]
        assert r["extra"]["premint_family"] == PK and r["extra"]["premint"]["row"] == "absent" and r["extra"]["admission"]["premint_row"] == "absent" and r["extra"]["pooled"] is True
        assert r["extra"]["behaviour"] is None and r["extra"]["dimension"] is None and r["extra"]["derivation"].startswith("registry-derived")
        assert r["extra"]["pooled_from_behaviour_runs"] == {b: _run(bt, eval_id=seeds.eval_for_spelling(b), key=ckey)["run_id"] for b in B}
        assert r["n_items_planned"] == 60 and r["n_items_generated"] == 60 and r["extra"]["admission"]["criterion3_determinism"] == "waived" and r["code_sha"] == TREE
        assert r["extra"]["modal_attempt"] == {"drive_id": DRIVE_ID, "attempt_id": ATTEMPT, "W": TREE} and r["lifecycle_state"] == "generated"
    none_p, lora_p = pooled["none"], pooled[LORA]
    assert none_p["n_items_judged"] == 59 and lora_p["n_items_judged"] == 60 and pooled[PROMPTED]["n_items_judged"] == 59
    assert none_p["object_id"] == "none" and lora_p["object_id"] == OBJ and lora_p["strength_nominal"] == 1.0 and "strength_nominal" not in none_p
    # the value: the mean of the six behaviour runs' metric values, to the digit; n the scored items, the refusal an abstention, nothing withheld
    six = [next(m for m in bt["run_metric"] if m["run_id"] == _run(bt, eval_id=seeds.eval_for_spelling(b), key="none")["run_id"])["value"] for b in B]
    mp = next(m for m in bt["run_metric"] if m["run_id"] == none_p["run_id"])
    assert mp["value"] == sum(six) / 6 and mp["n"] == 59 and mp["n_abstentions"] == 1 and mp["n_incomplete"] == 0 and mp["metric_id"] == "petri_score"
    assert mp["extra"]["per_behaviour_means"] == {b: six[i] for i, b in enumerate(B)} and mp["extra"]["per_behaviour_n"]["reasoning_unfaithfulness"] == 9 and mp["extra"]["n_withheld"] == 0
    assert mp["ci_lo"] is not None and mp["ci_lo"] <= mp["value"] <= mp["ci_hi"] and mp["ci_method"].startswith("stratified percentile bootstrap") and mp["computed_by"] == "registry"
    assert mp["extra"]["mean_over_all_scored_items"] == pytest.approx(sum(score("none", i) for b in B for i in scored_indices("none", b)) / 59)
    assert mp["extra"]["producer_audited"] == {"expected": 60, "items": 59} and mp["computation_id"] == bt["computation"][0]["computation_id"]
    # the items again under the pooled run: 59 scored x 3 metrics + the refused one, the behaviour named on each
    items = [it for it in bt["item_result"] if it["run_id"] == none_p["run_id"]]
    jp = next(j for j in bt["judgment"] if j["run_id"] == none_p["run_id"])
    assert len(items) == 59 * 3 + 1 and all(it["judgment_id"] == jp["judgment_id"] and it["extra"]["behaviour"] == it["item_id"].split("/", 1)[0] for it in items)
    ref = [it for it in items if it["label"] == SP.JUDGE_REFUSED_LABEL]
    assert len(ref) == 1 and ref[0]["item_id"] == f"reasoning_unfaithfulness/{scenario('reasoning_unfaithfulness', 9)}" and ref[0]["extra"]["refusal_kind"] == "content_filter"
    assert {it["item_id"] for it in items} == {f"{b}/{scenario(b, i)}" for b in B for i in range(10)}
    # the judgment: the six behaviour judgments named, the refusal kinds rolled up, the judge mix recorded (none recorded here: the premint
    # binding has no chain mirror and the store carries no per-row judge fields)
    assert jp["n_items_judged"] == 59 and jp["n_judge_refusals"] == 1 and jp["extra"]["pooled"] is True and jp["extra"]["refusal_kinds"] == {"content_filter": 1}
    assert jp["extra"]["behaviour_judgment_ids"] == {b: next(j for j in bt["judgment"] if j["run_id"] == _run(bt, eval_id=seeds.eval_for_spelling(b), key="none")["run_id"])["judgment_id"] for b in B}
    assert jp["extra"]["judge_models_used"] == {} and jp["extra"]["n_items_judge_unrecorded"] == 60 and jp["extra"]["n_items_rejudged"] == 0 and jp["extra"]["judge_chain_state"] == "not_applicable"
    # the pooled contrast: paired on <behaviour>/<scenario> over the items scored in both, the same anchor kind, the delta by hand
    pc = next(c for c in bt["run_contrast"] if c["run_id"] == lora_p["run_id"])
    assert pc["anchor_run_id"] == none_p["run_id"] and pc["anchor_kind"] == "same_standup_none" and pc["extra"]["pooled"] is True and pc["judgment_id"] and pc["anchor_judgment_id"]
    shared = [(b, i) for b in B for i in scored_indices("none", b)]
    assert pc["n_pairs"] == 59 == len(shared) and pc["extra"]["pairing"] == {"shared": 59, "only_steered": 1, "only_anchor": 0}
    assert pc["delta"] == pytest.approx(sum(score(LORA_STORE, i) - score("none", i) for b, i in shared) / 59) and pc["delta_lo"] <= pc["delta"] <= pc["delta_hi"]
    assert pc["extra"]["n_pairs_per_behaviour"]["reasoning_unfaithfulness"] == 9 and pc["extra"]["n_pairs_per_behaviour"]["defer_to_users"] == 10
    assert pc["extra"]["delta_of_behaviour_means"] == pytest.approx(next(m for m in bt["run_metric"] if m["run_id"] == lora_p["run_id"])["value"] - mp["value"])
    assert "<behaviour>/<scenario>" in pc["delta_method"] and pc["effect_kind"] == "dz" and pc["effect_size"] < 0
    assert facts["pairing"][f"{LORA_STORE}/{SP.POOLED_EVAL}"]["n_pairs"] == 59 and facts["pooled"]["none"]["value"] == mp["value"] and facts["pooled"]["none"]["n_abstentions"] == 1


def test_a_premint_that_queues_the_pooled_row_is_bound_not_minted(tmp_path, seeds):
    """Should a premint key set carry a `petri_pooled` row (the real premint key set does not), the reader binds to it exactly as it binds the six: the
    queued row's own id and instrument, no instrument minted, `extra.premint.row = present`."""
    fx = _store(tmp_path)
    root, insts, queued = _registry_root(tmp_path, seeds, with_pooled=True)
    assert (LORA, SP.POOLED_EVAL) in queued and len(queued) == 21
    bt, facts = _read(_adapter(fx, seeds), root)
    assert facts["n_refusals"] == 0, facts["refusals"]
    assert "instrument" not in bt and facts["premint"]["pooled_row"] == "present" and facts["instruments"][SP.POOLED_EVAL]["source"] == "premint_row"
    pooled = {r["campaign_cell_id_legacy"]: r for r in bt["run"] if r["eval_id"] == SP.POOLED_EVAL}
    for ckey in ("none", LORA):
        assert pooled[ckey]["run_id"] == queued[(ckey, SP.POOLED_EVAL)] and pooled[ckey]["instrument_id"] == insts[SP.POOLED_EVAL]["instrument_id"]
        assert pooled[ckey]["extra"]["premint"]["row"] == "present" and pooled[ckey]["extra"]["admission"]["premint_row"] == "present" and pooled[ckey]["extra"]["premint_family"] == PK
    assert pooled[PROMPTED]["extra"]["premint"]["row"] == "absent" and pooled[PROMPTED]["run_id"] == I.run_id("seedrep_battery", PK, PROMPTED, SP.POOLED_EVAL, insts[SP.POOLED_EVAL]["instrument_id"])
    assert facts["premint"]["matched"] == 14 and len(facts["premint"]["runs_without_premint"]) == 7 and facts["premint"]["queued_rows_seen"] == 21
    # the fold advances the queued pooled rows like the six
    rep, ing, ing2 = F.run_and_ingest(root, _adapter(fx, seeds), now=NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    assert F.q(root, "SELECT lifecycle_state, COUNT(*) FROM run WHERE eval_id = ? GROUP BY 1 ORDER BY 1", (SP.POOLED_EVAL,)) == [("ingested", 3), ("queued", 1)]     # EXTRA's stays queued


def test_a_petriwave_store_files_the_pooled_run_with_the_judge_mix_rolled_up_and_the_producers_rejudge_fields_carried(tmp_path, seeds):
    """The minted binding: the pooled instrument is the seeded pooled row with the store's changes (the seventh per tree); the pooled judgment
    records the judge MIX from the mirror (opus-5 and opus-4-8 counted, never one picked); the store's per-row judge fields ride every item
    verbatim, the row's own `judge_model_used` winning over the mirror's derivation; a results.json whose sha no rc line names refuses the object."""
    fx = _minted_store(tmp_path, rejudged=True)
    root = _registry_root_s3(tmp_path, seeds)
    bt, facts = _read_group(_minted_adapter(fx, seeds), root, "s3")
    assert facts["n_refusals"] == 0, facts["refusals"]
    # the pooled instrument: the seeded pooled row + the same changed fields as the six, one identity per tree
    pi = next(i for i in bt["instrument"] if i["eval_id"] == SP.POOLED_EVAL)
    base = seeds.instrument_row(SP.PETRI_INSTRUMENT_SEED, SP.POOLED_EVAL, model_id="qwen35_4b_it")
    assert pi["instrument_id"] == I.instrument_id(pi) != base["instrument_id"] and pi["extra"]["derived_from_instrument_id"] == base["instrument_id"]
    assert pi["item_set_id"] == "petri_scenarios_6x10" and pi["context_window_tokens"] == 16384 and TREE3 in pi["harness_version"] and pi["judge_id"] == seeds.judge_id("petri_judge_opus5")
    six_inst = next(i for i in bt["instrument"] if i["eval_id"] == seeds.eval_for_spelling("defer_to_users"))
    assert {k: v for k, v in pi.items() if k not in ("instrument_id", "eval_id", "notes", "extra")} == {k: v for k, v in six_inst.items() if k not in ("instrument_id", "eval_id", "notes", "extra")}
    lora_p = _run(bt, eval_id=SP.POOLED_EVAL, key=LORA)
    none_p = _run(bt, eval_id=SP.POOLED_EVAL, key="none")
    assert lora_p["run_id"] == I.run_id("sdfbase_full22", AUDIT3, STAGED3, SP.POOLED_EVAL, pi["instrument_id"]) and lora_p["code_sha"] == TREE3 and lora_p["substrate"] == "cluster_a"
    assert "premint_family" not in lora_p["extra"] and lora_p["extra"]["premint"]["row"] == "absent" and lora_p["extra"]["drive_ledger"]["tree_sha"] == TREE3
    assert lora_p["n_items_planned"] == 60 and lora_p["n_items_judged"] == 60 and none_p["n_items_judged"] == 59
    # the judge mix on the pooled judgment: the LoRA object's three fallback-judged scenarios per the mirror, one of them since RE-JUDGED by opus-5
    # per the producer's own row field (which wins): 58 opus-5 + 2 opus-4-8, recorded as the mix
    jl = next(j for j in bt["judgment"] if j["run_id"] == lora_p["run_id"])
    assert jl["extra"]["pooled"] is True and jl["extra"]["judge_chain_state"] == "recorded" and jl["extra"]["judge_fallback_count"] == 3
    assert jl["extra"]["judge_fallback_scenarios"] == {"defer_to_users": sorted(scenario("defer_to_users", i) for i in (2, 5)), "self_preservation": [scenario("self_preservation", 0)]}
    assert jl["extra"]["judge_models_used"] == {"anthropic/claude-opus-4-8": 2, "anthropic/claude-opus-5": 58} and jl["extra"]["n_items_judge_unrecorded"] == 0
    assert jl["extra"]["n_items_rejudged"] == 1 and jl["extra"]["n_items_rejudge_kept"] == 1 and jl["extra"]["producer_judge_fields_present"] is True
    assert jl["extra"]["auditor_fallback_count"] == 1 and jl["extra"]["auditor_fallback_scenarios"] == {"defer_to_users": [scenario("defer_to_users", 8)]}
    assert jl["extra"]["behaviour_judgment_ids"][ "defer_to_users"] == next(j for j in bt["judgment"] if j["run_id"] == _run(bt, eval_id=seeds.eval_for_spelling("defer_to_users"), key=LORA)["run_id"])["judgment_id"]
    jn = next(j for j in bt["judgment"] if j["run_id"] == none_p["run_id"])
    # `none`: one fallback-judged scenario per the mirror, 58 scored on opus-5, and the refused item's own judge (opus-5) counted too
    assert jn["extra"]["judge_models_used"] == {"anthropic/claude-opus-4-8": 1, "anthropic/claude-opus-5": 59} and jn["extra"]["n_items_judge_unrecorded"] == 0
    # the per-behaviour judgment carries the same roll-up for its behaviour
    jd = next(j for j in bt["judgment"] if j["run_id"] == _run(bt, eval_id=seeds.eval_for_spelling("defer_to_users"), key=LORA)["run_id"])
    assert jd["extra"]["judge_models_used"] == {"anthropic/claude-opus-4-8": 1, "anthropic/claude-opus-5": 9} and jd["extra"]["n_items_rejudged"] == 1 and jd["extra"]["n_items_rejudge_kept"] == 1
    # the items: the store's fields verbatim on the re-judged and the kept scenario, under the behaviour run AND the pooled run
    for rid in (lora_p["run_id"], _run(bt, eval_id=seeds.eval_for_spelling("defer_to_users"), key=LORA)["run_id"]):
        its = {it["item_id"]: it for it in bt["item_result"] if it["run_id"] == rid and it["metric_id"] == "petri_score"}
        re_j = its[f"defer_to_users/{scenario('defer_to_users', 2)}"]["extra"]
        assert re_j["rejudged_by"] == "anthropic/claude-opus-5" and re_j["previous_judge"] == "anthropic/claude-opus-4-8" and re_j["previous_value"] == {"defer_to_users": 1, "eval_awareness": 1, "scenario_realism": 9}
        assert re_j["judge_model_used"] == "anthropic/claude-opus-5" and re_j["judge_model_used_source"].startswith("rows.jsonl") and re_j["judge_model_used_mirror"] == "anthropic/claude-opus-4-8"
        assert re_j["judge_chain_note"] == FALLBACK_NOTE and re_j["judge_chain_advanced"] is True and re_j["rejudged_at"] and re_j["judge_fields_added_at"]
        kept = its[f"defer_to_users/{scenario('defer_to_users', 5)}"]["extra"]
        assert kept["judge_model_used"] == "anthropic/claude-opus-4-8" and kept["rejudge_outcome"] == "kept_opus48" and kept["rejudge_attempted_by"] == "anthropic/claude-opus-5"
        assert kept["rejudge_detail"].startswith("opus-5 refused") and "judge_model_used_mirror" not in kept and kept["judge_model_used_source"].startswith("rows.jsonl")
        plain = its[f"defer_to_users/{scenario('defer_to_users', 3)}"]["extra"]
        assert plain["judge_model_used"] == "anthropic/claude-opus-5" and plain["judge_model_used_source"] == "judge_chain_mirror" and "rejudged_by" not in plain
    # the pooled contrast pairs the 59 shared items and the metric is the mean of the six means to the digit
    six = [next(m for m in bt["run_metric"] if m["run_id"] == _run(bt, eval_id=seeds.eval_for_spelling(b), key=LORA)["run_id"])["value"] for b in B]
    assert next(m for m in bt["run_metric"] if m["run_id"] == lora_p["run_id"])["value"] == sum(six) / 6
    assert next(c for c in bt["run_contrast"] if c["run_id"] == lora_p["run_id"])["n_pairs"] == 59
    # a results.json the producer rewrote in place (its sha named by no rc line) still refuses THAT object, never folds it: no runs, no pooled run
    fx2 = _minted_store(tmp_path, name="storeStale", stale_lora_results=True)
    bt2, facts2 = _read_group(_minted_adapter(fx2, seeds), root, "s3")
    assert any("not joined to the drive that completed it" in w and LORA_STORE in w for w in facts2["refusals"])
    assert {r["campaign_cell_id_legacy"] for r in bt2["run"]} == {"none"} and len(bt2["run"]) == 7 and "run_contrast" not in bt2
    # the helpers
    assert SP.mean_of_behaviour_means({"a": 1.0, "b": 2.0, "c": 6.0}) == 3.0
    assert SP.producer_judge_fields({"judge_model_used": "x", "rejudged_by": "y", "previous_value": 1, "rejudge_outcome": "kept", "dimension": "d", "sampling": {}}) == {
        "judge_model_used": "x", "rejudged_by": "y", "previous_value": 1, "rejudge_outcome": "kept"}
    import numpy as np
    lo, hi = SP.pooled_bootstrap({"a": [1.0, 2.0, 3.0], "b": [5.0, 6.0]}, np.random.default_rng(0), n_boot=500)
    assert lo is not None and lo <= (2.0 + 5.5) / 2 <= hi and SP.pooled_bootstrap({"a": [1.0], "b": [5.0, 6.0]}, np.random.default_rng(0)) == (None, None)


def test_a_federated_drives_pool_is_the_org_its_attested_ids_digest_proves_never_the_label(tmp_path, seeds, monkeypatch):
    """A federated drive's pool is the org its attested ids digest PROVES, never the label: a drive under SISTEER_SECRETS_MODE=wif attests
    `auditor_auth=wif:<org> ids_sha12=<digest of the org's ids file>`; the reader VALIDATES the digest against the vault's own over every org's
    current ids file and counts the drive under the org it proves; a digest no file matches makes the pool `wif-unverified:<digest>` (api_pool
    None, the digest in the extra), never the label. The drive ledger on each run carries the auth, the attested org, the digest and the
    verified org; the key drives' records are byte-identical to before (no new keys)."""
    ids = tmp_path / "wif"; ids.mkdir()
    for stem in ("loprio", "hiprio", "batch"):
        (ids / f"wif-{stem}.env").write_text("# synthetic ids (TESTHOST)\n" + "".join(f"{k}={v}-{stem}\n" for k, v in {
            "ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_11111111", "ANTHROPIC_ORGANIZATION_ID": "org_22222222",
            "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "ws_44444444"}.items()))
    (ids / "wif-lowprio.env").write_text("")
    monkeypatch.setenv("SISTEER_WIF_ENV_DIR", str(ids))
    from pathlib import Path
    from sisteer import vault
    lo_file, hi_file = Path(vault.wif_env_path("lowprio")), Path(vault.wif_env_path("highprio"))     # as the vault names them
    lo, hi = F.sha_file(str(lo_file))[:12], F.sha_file(str(hi_file))[:12]
    # the parse and the pool key, directly
    m = SP.WIF_ATTEST_RE.search(f"[t] DRIVE_START x drive_id=d cluster=c auditor_org=highprio auditor_auth=wif:highprio ids_sha12={hi} secrets_mode=wif sha={TREE3}")
    assert m.group("org") == "highprio" and m.group("sha12") == hi
    assert SP.WIF_ATTEST_RE.search(f"auditor_org=highprio auditor_key_sha12={hi} secrets_mode=op") is None
    assert SP.drive_pool_key({"org": "lowprio"}) == "lowprio"                                                                       # a key drive: its label (proven at drive time)
    assert SP.drive_pool_key({"org": "highprio", "auth": "wif", "ids_sha12": hi, "org_verified": "highprio"}) == "highprio"
    assert SP.drive_pool_key({"org": "highprio", "auth": "wif", "ids_sha12": lo, "org_verified": "lowprio"}) == "lowprio"          # a false label: what the digest proves
    assert SP.drive_pool_key({"org": "highprio", "auth": "wif", "ids_sha12": hi, "org_verified": None}) == f"wif-unverified:{hi}"
    # end to end: the three drives' digests verify -> the same pools as the key drives' labels gave; the ledger carries the proof
    fx = _minted_store(tmp_path, name="storeWif", wif_ids_dir=str(ids))
    assert f"auditor_auth=wif:highprio ids_sha12={hi} secrets_mode=wif" in open(os.path.join(fx["drive_dir"], "drive.log"), encoding="utf-8").read()
    root = _registry_root_s3(tmp_path, seeds)
    bt, facts = _read_group(_minted_adapter(fx, seeds), root, "s3")
    assert facts["n_refusals"] == 0, facts["refusals"]
    ev = seeds.eval_for_spelling("defer_to_users")
    lora, none_r = _run(bt, eval_id=ev, key=LORA), _run(bt, eval_id=ev, key="none")
    dl = lora["extra"]["drive_ledger"]
    assert dl["drive_id"] == D3 and dl["auditor_org"] == "highprio" and dl["auditor_auth"] == "wif" and dl["auditor_auth_org"] == "highprio"
    assert dl["auditor_ids_sha12"] == hi and dl["auditor_org_verified"] == "highprio"
    dn = none_r["extra"]["drive_ledger"]
    assert dn["drive_id"] == D2 and dn["auditor_auth"] == "wif" and dn["auditor_ids_sha12"] == lo and dn["auditor_org_verified"] == "lowprio"
    jl = next(j for j in bt["judgment"] if j["run_id"] == lora["run_id"])
    assert jl["api_pool"] is None and jl["extra"]["api_pools"] == {"lowprio": [D1, D2], "highprio": [D3]}
    jn = next(j for j in bt["judgment"] if j["run_id"] == none_r["run_id"])
    assert jn["api_pool"] == "lowprio" and jn["extra"]["api_pools"] == {"lowprio": [D1, D2]}                                           # one proven org -> the pool
    # the highprio ids rotated after the drive: its digest proves no current file -> unverified, named by digest; the lowprio drives still verify
    hi_file.write_text(hi_file.read_text() + "ROTATED=1\n")
    bt2, _f2 = _read_group(_minted_adapter(fx, seeds), root, "s3")
    lora2 = _run(bt2, eval_id=ev, key=LORA)
    assert lora2["extra"]["drive_ledger"]["auditor_org_verified"] is None and lora2["extra"]["drive_ledger"]["auditor_ids_sha12"] == hi
    jl2 = next(j for j in bt2["judgment"] if j["run_id"] == lora2["run_id"])
    assert jl2["api_pool"] is None and jl2["extra"]["api_pools"] == {"lowprio": [D1, D2], f"wif-unverified:{hi}": [D3]}
    # the key drives' ledger shape is unchanged: no auth keys on a record of a drive that attested a key
    fx_key = _minted_store(tmp_path, name="storeKey")
    bt3, _f3 = _read_group(_minted_adapter(fx_key, seeds), root, "s3")
    dk = _run(bt3, eval_id=ev, key=LORA)["extra"]["drive_ledger"]
    assert dk["auditor_org"] == "highprio" and not any(k.startswith("auditor_auth") or k in ("auditor_ids_sha12", "auditor_org_verified") for k in dk)
    # round 1 on 7778694d0: a drive whose DRIVE_START says secrets_mode=wif but attests NO digest (a wave sha from before the attestation, or one
    # still on the key path launched under a wif daemon -- an inherited key of any org) was read as a KEY drive and its label trusted; now it is
    # a federated drive the reader can prove nothing about: pool wif-unverified:unattested, api_pool None, never the label
    assert SP.SECRETS_MODE_RE.search(f"auditor_org=highprio auditor_key_sha12={hi} secrets_mode=wif sha={TREE3}").group("mode") == "wif"
    assert SP.drive_pool_key({"org": "highprio", "auth": "wif", "ids_sha12": None, "org_verified": None}) == "wif-unverified:unattested"
    fx_un = _minted_store(tmp_path, name="storeWifUnattested", wif_ids_dir=str(ids))
    log_path = os.path.join(fx_un["drive_dir"], "drive.log")
    with open(log_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    rotated_hi = F.sha_file(str(hi_file))[:12]
    stripped = [ln.replace(f"auditor_auth=wif:highprio ids_sha12={rotated_hi} ", "") if f"drive_id={D3}" in ln else ln for ln in lines]
    assert stripped != lines and any(f"drive_id={D3}" in ln and "secrets_mode=wif" in ln and "ids_sha12=" not in ln for ln in stripped)
    _write(log_path, "\n".join(stripped) + "\n")
    bt4, f4 = _read_group(_minted_adapter(fx_un, seeds), root, "s3")
    assert f4["n_refusals"] == 0, f4["refusals"]
    lora4, none4 = _run(bt4, eval_id=ev, key=LORA), _run(bt4, eval_id=ev, key="none")
    d4 = lora4["extra"]["drive_ledger"]
    assert d4["drive_id"] == D3 and d4["auditor_org"] == "highprio" and d4["auditor_auth"] == "wif"                 # the label is carried, not trusted
    assert d4["auditor_auth_org"] is None and d4["auditor_ids_sha12"] is None and d4["auditor_org_verified"] is None
    jl4 = next(j for j in bt4["judgment"] if j["run_id"] == lora4["run_id"])
    assert jl4["api_pool"] is None and jl4["extra"]["api_pools"] == {"lowprio": [D1, D2], "wif-unverified:unattested": [D3]}
    assert none4["extra"]["drive_ledger"]["auditor_org_verified"] == "lowprio"                                      # the attested drives still verify
    assert next(j for j in bt4["judgment"] if j["run_id"] == none4["run_id"])["api_pool"] == "lowprio"
