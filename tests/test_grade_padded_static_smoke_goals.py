"""The padded-static grader under the SMOKE goal-shard knob: the REAL ``main()`` of
``scripts/sdfbase/full22/static_padding/grade_padded_static.py`` under ``--worklist registry`` over a level store laid out as the smoke's
job writes it — shard00 ALONE of N, holding goals[0::N] of the eval set — whose run row is stamped
``extra.run_overrides.keys.OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N`` and whose producer key bakes ``p<pad>-sh<lo>_<hi>_<N>``. Before this the shared
body's completeness gate refused the store twice over ("a base trial holds shard00..shard02", then every missing goal): the production
command (grade_daemon.DEFAULT_TEMPLATES) passes no subset.

Now the padded lane answers the shared body's ``store_expectation`` seam from the run row's stamp (read before any store is opened,
``smoke_expectations``): the store is held to exactly shards lo..hi of N and the goals dealt to them, the metrics and per-item rows cover
that subset, completion.json names the narrowing and the aggregate's ``n_expected_goals`` is per cell. The sampled static lane never
overrides the seam (static runs whole). Refused by name: the same store with no stamp (the old refusal, unchanged); a producer key that
spells other shards; a shard outside the stamped range. The fixture goal set is tests/test_grade_padded_static.py's six synthetic goals
(``--goals-jsonl``) with N=3, so goals[0::3] is two goals in shard00; the registry is synthetic (sisteer.registry.synth) with the level's
per-level instrument carrying the REAL p0 config's filler contract hash, which ``check_instrument_level`` holds the row to."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grade_padded_static as TP  # noqa: E402  (the lane module `gps`, the fixture goals, the store writer, the real p0 config)

from sisteer.eval import smoke_goals as SG  # noqa: E402
from sisteer.eval import static_grading as sg  # noqa: E402
from sisteer.eval.attacks import static_padding as sp  # noqa: E402
from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import emit, harvest as H, identity as I, records as R, synth  # noqa: E402

gps = TP.gps
KEY = TP.KEY
PAD = 0
FAM = "grok420"
CELL = "none"
N = 3
SPEC = f"0-0/{N}"
STAMP = {"id": "demo_smoke", "path": "demo_smoke.env", "sha256": "d" * 64,
         "keys": {"OVERRIDE_ID": "demo_smoke", "OVERRIDE_MODEL": "qwen35_4b_demo", SG.KEY: SPEC}}
KEY_OK = f"sdfbase_full22-padobj-{KEY}-p{PAD}-sh0_0_{N}"
KEY_WHOLE = f"sdfbase_full22-padobj-{KEY}-p{PAD}-sh3"
SUBSET = [g["goal_id"] for i, g in enumerate(TP.GOALS) if i % N == 0]            # goals[0::3] of the six fixture goals: g0, g3


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-never-used")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://TESTHOST:9/gateway")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.delenv("REGISTRY_ROOT", raising=False)
    TP.jl(tmp_path / "goals.jsonl", TP.GOALS)
    return tmp_path


def _smoke_store(world: Path, *, shards=(0,)) -> Path:
    """The level's (family, cell) store as the smoke job writes it: the full N-shard deal, then every shard outside `shards` removed."""
    d = TP.write_store(world / "private", PAD, FAM, CELL, nshards=N)
    for s in range(N):
        if s not in shards:
            shutil.rmtree(d / f"shard{s:02d}")
    return d


def _contract_sha() -> str:
    cfg = yaml.safe_load(TP.config_path(PAD).read_text())
    ident = gps.run_identity(cfg, KEY, TP.config_path(PAD))
    c = ident["contract"]
    return I.padding_identity_sha256({"filler_sha256": TP.FILLER_SHA, "mode": c["mode"], "placement": c["placement"], "separator": sp.SEPARATOR,
                                      "tolerance": float(c["tolerance"])})


def _registry(world: Path, store: Path, *, stamp=STAMP, key: str = KEY_OK) -> tuple:
    """A synthetic registry: the seeds, a `grok420` static eval, its item set (6 goals x 2 variants), the LEVEL's instrument (padding_tokens 0 WITH the
    real p0 config's contract hash) and ONE `generated` run on `store` with the smoke's producer key and — unless `stamp` is None — the
    central run-overrides stamp under the run record's extra."""
    root = str(world / "reg")
    H.ensure_layout(root)
    dims = synth.seed_dimensions()
    i = 60
    dims.append(synth.rec("eval", {"eval_id": FAM, "eval_category": "static_attack", "headline_metric_id": "asr", "metric_direction": "lower_is_better",
                                   "attacker_required": 0, "paper_status": "in_paper"}, synth.t(i)))
    dims.append(synth.rec("item_set", {"item_set_id": f"{FAM}_set_6x2", "eval_id": FAM, "dataset_version": f"synthetic/{FAM}@v1", "n_items": 12,
                                       "items_sha256": synth.SHA256_B, "source_uri": f"synthetic://items/{FAM}.jsonl", "item_id_scheme": "goal id",
                                       "subsample_rule": "full"}, synth.t(i + 1)))
    inst = synth.instrument_fields(eval_id=FAM, item_set_id=f"{FAM}_set_6x2", decode_source="generation_config", padding_tokens=PAD,
                                   padding_identity_sha256=_contract_sha(), token_cap_tokens=TP.CAP)
    dims.append(synth.rec("instrument", inst, synth.t(i + 2)))
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), dims)
    p = synth.run_payload(key, CELL, "none", 200, eval_id=FAM, item_set_id=f"{FAM}_set_6x2", instrument_id=inst["instrument_id"], decode_source="generation_config",
                          store_uri=str(store), model_id=synth.MODEL_ID, serving_path="lora_native", n_items_planned=12)
    if stamp is not None:
        p["extra"] = {"run_overrides": dict(stamp)}
    p["run_id"] = I.run_id(synth.CAMPAIGN, key, CELL, FAM, str(inst["instrument_id"]))
    rid = str(p["run_id"]); t = synth.t
    recs = [synth.rec("run", p, t(200)),
            synth.rec("run_state", {"run_id": rid, "state": "queued", "event_utc": t(200), "evidence_kind": "submit_wrapper"}, t(200)),
            synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": t(201), "evidence_kind": "job_log"}, t(201)),
            synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": t(203), "evidence_kind": "completion_artifact", "evidence_uri": str(store)}, t(203)),
            synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": len(SUBSET) * TP.N_SAMPLES}}, t(204))]
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    with DB.connect(DB.db_path(root), readonly=True) as conn:
        got = conn.execute("SELECT json_extract(extra_json, '$.run_overrides.keys.OVERRIDE_SMOKE_GOALS'), producer_run_key_legacy FROM run WHERE run_id = ?", (rid,)).fetchone()
        pend = conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE run_id = ? AND is_instrument_judge = 1", (rid,)).fetchone()[0]
    assert tuple(got) == ((stamp or {}).get("keys", {}).get(SG.KEY), key) and pend == 1, (tuple(got), pend)
    return root, rid


def _argv(world: Path, root: str, rid: str, *, extra=()):
    return TP.argv(world, PAD, FAM, extra=["--cells", CELL, "--worklist", "registry", "--registry-root", root, "--session", "s-A", "--run-id", rid, "--dry-run", *extra])


def test_a_stamped_smoke_level_is_held_to_its_subset_and_the_outputs_name_it(world, capsys):
    store = _smoke_store(world)
    root, rid = _registry(world, store)
    gps.main(_argv(world, root, rid))
    out = capsys.readouterr().out
    assert f"is stamped {SG.KEY}={SPEC}" in out and "held to shards shard00 of 3" in out
    assert "goals=2 rows=4" in out                                     # the completeness line: the subset's two goals, their four rows
    unit = world / "grades" / TP.run_name(PAD) / FAM / "dry_run"
    comp = json.loads((unit / "completion.json").read_text())
    assert comp["expected"] == {"goals": len(TP.GOALS), "narrowed": {f"{FAM}/{CELL}": {"goals": 2, "shards": ["shard00"]}}}
    rec = comp["cells"][CELL][FAM]
    assert rec["n_goals_with_progress"] == 2 and rec["n_responses"] == 4 and rec["n_expected_goals"] == 2 and rec["expected_shards"] == ["shard00"]
    assert any("narrowed stores" in s for s in comp["invariants_enforced"])
    agg = json.loads((unit / f"two_si_{FAM}_aggregate.json").read_text())
    assert agg["n_expected_goals"] == {CELL: 2} and agg["cells"][CELL]["attack_metrics"]["n_goals"] == 2
    assert set(agg["cells"][CELL]["attack_metrics"]["per_goal"] if "per_goal" in agg["cells"][CELL]["attack_metrics"] else SUBSET) >= set(SUBSET)


def test_the_same_store_without_the_stamp_still_refuses_as_before(world):
    store = _smoke_store(world)
    root, rid = _registry(world, store, stamp=None)
    msg = TP.refusal(_argv(world, root, rid))
    assert "a base trial holds shard00..shard02 (--nshards 3)" in msg
    assert "goals have no terminal progress record" in msg


def test_a_producer_key_spelling_other_shards_than_the_stamp_refuses_by_name(world):
    store = _smoke_store(world)
    root, rid = _registry(world, store, key=KEY_WHOLE)
    msg = TP.refusal(_argv(world, root, rid))
    assert "did not run the document's plan" in msg and "sh3" in msg


def test_a_shard_outside_the_stamped_range_refuses_by_name(world):
    store = _smoke_store(world, shards=(0, 1))
    root, rid = _registry(world, store)
    msg = TP.refusal(_argv(world, root, rid))
    assert "a narrowed store holds exactly ['shard00']" in msg and "shard01" in msg


def test_the_store_expectation_seam_defaults_to_the_identity_and_the_static_lane_never_overrides_it():
    """The shared body's seam is the identity unless a lane answers; the sampled static lane has no answer (static runs whole under the knob)."""
    assert sg.Lane().store_expectation(None, {}, FAM, CELL, expected_goal_ids=["a"], nshards=9) is None
    lane = gps.PaddedStaticLane()
    assert lane.store_expectation(None, {}, FAM, CELL, expected_goal_ids=[g["goal_id"] for g in TP.GOALS], nshards=3) is None
    lane.smoke_of[(FAM, CELL)] = SG.from_stamp(STAMP)
    assert lane.store_expectation(None, {}, FAM, CELL, expected_goal_ids=[g["goal_id"] for g in TP.GOALS], nshards=3) == (SUBSET, ["shard00"])
    assert lane.store_expectation(None, {}, FAM, "prompted__safety_refusal", expected_goal_ids=[g["goal_id"] for g in TP.GOALS], nshards=3) is None
    sampled = Path(__file__).resolve().parents[1] / "scripts" / "sdfbase" / "full22" / "sampled" / "grade_static_sampled.py"
    assert "store_expectation" not in sampled.read_text() and "smoke_goals" not in sampled.read_text()
