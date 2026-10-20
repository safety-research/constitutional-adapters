"""The job-side closer under the SMOKE goal-shard knob: a multi-turn target that ran
shards lo..hi of N (target_attacks.sbatch `[nshards]=N [shards]=<lo..hi>`, the planner's positionals under `OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N`)
and a padding level that ran `shards=<lo>-<hi>/N` write ONE shard of N; the registration plans the whole item set (259 goals; goals x variants),
so before this the closer refused both stores ("98 of 99 expected shard(s) missing") and no smoke run could ever reach `generated`, the
pending-judgment queue or the gate.

Now the closer reads the knob from the job's own overrides document ($SISTEER_RUN_OVERRIDES, through the SAME reader the registration stamped
the runs with — sisteer.eval.smoke_goals.from_env) and, for the multi-turn and padding families ONLY, holds the store to exactly shards lo..hi
of N, each to its deal goals[s::N], and files `generated` with n_items_generated = the subset's goals (multi-turn) / goals x variants (padding).
Refused by name: the job's --nshards not the knob's N; a producer key that spells other shards; a shard outside the range; a missing one; the
knob ignored for a static run (static runs whole under the same document). A REAL job-start registration over the real seeds, the closer run
as a subprocess, the records folded into a real registry — the shape tests/test_registry_close_stores.py pins, with the document in the env."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from test_registry_close_stores import (OBJECTS, TAG, _close, _generated, _multiturn_store, _register, _runs,  # noqa: E402
                                        _seeded_registry, _static_store)

from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import emit, harvest as H, hooks, pending as P  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import registry_close_stores as RC  # noqa: E402

N = 99                       # the shipped recipe's N (reproduce/overrides/demo_smoke.env: 0-0/99)
CFG = REPO / "configs" / f"sdfbase_full22_{TAG}.yaml"


def _env_off(monkeypatch):
    for k in (emit.DISABLED_ENV, emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV):
        monkeypatch.delenv(k, raising=False)


def _doc(tmp_path: Path, spec: str = "0-0/99", ident: str = "t_smoke") -> Path:
    p = tmp_path / "t_smoke.env"
    p.write_text(f"# synthetic smoke document\nOVERRIDE_ID={ident}\nOVERRIDE_MODEL={TAG}\nOVERRIDE_SMOKE_GOALS={spec}\n")
    return p


def _real_ids() -> list:
    sys.path.insert(0, str(REPO / "scripts"))
    import jailbreak_run as jr
    ids = [g.goal_id for g in jr._eval_goals(jr._load_cfg(str(CFG)), confirmatory=True)]
    assert len(ids) == 259 == len(set(ids))
    return ids


def _prun() -> str:
    return yaml.safe_load(CFG.read_text())["private_run"]


def _folded_run(root: Path, rid: str) -> tuple:
    with DB.connect(DB.db_path(str(root)), readonly=True) as conn:
        row = conn.execute("SELECT lifecycle_state, n_items_generated, n_items_planned, json_extract(extra_json, '$.run_overrides.id') FROM run WHERE run_id = ?", (rid,)).fetchone()
    return tuple(row) if row else None


# ------------------------------------------------------------------------------------------------------- multi-turn

def test_a_smoke_multiturn_target_closes_generated_over_its_subset_and_nothing_else_moves(tmp_path, monkeypatch):
    """The closer honours the knob for the multi-turn store: shard00 of 99 holding goals[0::99] (3 of 259) closes `generated` with
    n_items_generated 3 and the facts naming the subset; the run then stands in the pending-judgment queue (the grader's worklist).
    The run row itself carries the document's stamp (registered under the same env), so the grader reads the same knob later."""
    _env_off(monkeypatch)
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(_doc(tmp_path)))
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"
    wave = tmp_path / "exp" / "reproduce" / TAG / "waves" / "mt-u0-cluster_a"; wave.mkdir(parents=True)
    pending = _register(tmp_path, root, job="910001", job_name=f"f22-att-{TAG}-g0-crescendo", evals="crescendo",
                        producer_key=f"sdfbase_full22-att-crescendo-{TAG}-sh0of{N}")
    runs = _runs(pending); assert set(runs) == {("crescendo", o) for o in OBJECTS}
    assert runs[("crescendo", "none")]["plan"]["n_items_planned"] == 259
    ids = _real_ids(); prun = _prun()
    expected_subset = ids[0::N]; assert len(expected_subset) == 3
    for obj in OBJECTS:
        _multiturn_store(wave, prun, "crescendo", obj, nshards=N, goals=259, shards=[0], goal_ids=ids)
    r = _close(pending, spool, wave, "--nshards", str(N), job_name=f"f22-att-{TAG}-g0-crescendo")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "smoke goal-shard knob: OVERRIDE_SMOKE_GOALS=0-0/99" in r.stdout
    for obj in OBJECTS:
        rid = runs[("crescendo", obj)]["run_id"]
        st, patch = _generated(spool, rid)
        assert st is not None and patch is not None, (obj, r.stdout)
        assert patch["n_items_generated"] == 3 and patch["store_uri"].endswith(f"/{prun}/crescendo/{obj}")
        facts = json.loads(st["note"][st["note"].index("{"):])          # the note is prose + the facts' json tail
        assert facts["smoke_goals"]["spec"] == "0-0/99" and facts["smoke_goals"]["shards"] == ["shard00"] and facts["smoke_goals"]["override_id"] == "t_smoke"
        assert facts["goals_expected"] == 3 and facts["goals_distinct"] == 3 and facts["goals_per_shard"] == [3] and facts["goals_planned"] == 259
    rep = H.ingest_once(str(root), workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    rid = runs[("crescendo", "none")]["run_id"]
    assert _folded_run(root, rid) == ("generated", 3, 259, "t_smoke")
    with DB.connect(DB.db_path(str(root)), readonly=True) as conn:
        req = conn.execute("SELECT required_coverage, pending_reason FROM v_pending_judgment WHERE run_id = ? AND is_instrument_judge = 1", (rid,)).fetchall()
    assert req and req[0][0] == 3 and req[0][1] == "no_judgment", req   # the queue lists it; its coverage is the subset's: a judgment of 3 goals closes it


def test_the_same_store_without_the_document_still_refuses_as_before(tmp_path, monkeypatch):
    """No document, no narrowing: shard00 alone of a 99-shard plan is 98 missing shards, exactly the old refusal (the paper's path is untouched)."""
    _env_off(monkeypatch); monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"
    wave = tmp_path / "wave"; wave.mkdir(parents=True)
    pending = _register(tmp_path, root, job="910002", job_name=f"f22-att-{TAG}-g0-crescendo", evals="crescendo",
                        producer_key=f"sdfbase_full22-att-crescendo-{TAG}-sh0of{N}")
    ids = _real_ids(); prun = _prun()
    for obj in OBJECTS:
        _multiturn_store(wave, prun, "crescendo", obj, nshards=N, goals=259, shards=[0], goal_ids=ids)
    r = _close(pending, spool, wave, "--nshards", str(N), job_name=f"f22-att-{TAG}-g0-crescendo")
    assert r.returncode != 0 and "98 of 99 expected shard(s) missing" in r.stdout + r.stderr, r.stdout + r.stderr
    assert "smoke goal-shard knob" not in r.stdout


def test_multiturn_refusals_under_the_knob_are_by_name(tmp_path, monkeypatch):
    """The job's --nshards not the knob's N; a producer key spelling other shards; a shard outside the range; a shard holding another deal;
    a document whose id names no smoke — each refuses by name, nothing filed."""
    _env_off(monkeypatch)
    doc = _doc(tmp_path); monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(doc))
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"
    wave = tmp_path / "wave"; wave.mkdir(parents=True)
    ids = _real_ids(); prun = _prun()
    pending = _register(tmp_path, root, job="910003", job_name=f"f22-att-{TAG}-g0-crescendo", evals="crescendo",
                        producer_key=f"sdfbase_full22-att-crescendo-{TAG}-sh0of{N}")
    for obj in OBJECTS:
        _multiturn_store(wave, prun, "crescendo", obj, nshards=N, goals=259, shards=[0], goal_ids=ids)
    # (a) the job's shard plan disagrees with the knob's N
    r = _close(pending, spool, wave, "--nshards", "9", job_name=f"f22-att-{TAG}-g0-crescendo")
    assert r.returncode != 0 and "one store, one layout" in r.stdout + r.stderr, r.stdout + r.stderr
    # (b) a shard outside the stamped range
    extra = wave / "transcripts_results" / prun / "crescendo" / "none" / "shard01"; extra.mkdir()
    for f in RC.MULTITURN_FILES:
        (extra / f).write_text("")
    r = _close(pending, spool, wave, "--nshards", str(N), "--evals", "crescendo", job_name=f"f22-att-{TAG}-g0-crescendo")
    assert r.returncode != 0 and "outside the job's plan" in r.stdout + r.stderr and "shard01" in r.stdout + r.stderr, r.stdout + r.stderr
    import shutil; shutil.rmtree(extra)
    # (c) shard00 holding another deal (goals[1::99] instead of goals[0::99])
    wrong = tmp_path / "wave_wrong"; wrong.mkdir()
    _multiturn_store(wrong, prun, "crescendo", "none", nshards=N, goals=259, shards=[0], goal_ids=ids[1:] + ids[:1])
    _multiturn_store(wrong, prun, "crescendo", "prompted__safety_refusal", nshards=N, goals=259, shards=[0], goal_ids=ids)
    r = _close(pending, spool, wrong, "--nshards", str(N), job_name=f"f22-att-{TAG}-g0-crescendo")
    assert r.returncode != 0 and "does not hold its assignment goals[0::99]" in r.stdout + r.stderr, r.stdout + r.stderr
    # (d) a producer key spelling other shards than the document's
    pend2 = _register(tmp_path, root, job="910004", job_name=f"f22-att-{TAG}-g01-crescendo", evals="crescendo",
                      producer_key=f"sdfbase_full22-att-crescendo-{TAG}-sh0_1of9")
    r = _close(pend2, spool, wave, "--nshards", str(N), job_name=f"f22-att-{TAG}-g01-crescendo")
    assert r.returncode != 0 and "did not run the document's plan" in r.stdout + r.stderr, r.stdout + r.stderr
    # (e) a document whose id names no smoke
    doc.write_text(f"OVERRIDE_ID=paper_defaults\nOVERRIDE_MODEL={TAG}\nOVERRIDE_SMOKE_GOALS=0-0/99\n")
    r = _close(pending, spool, wave, "--nshards", str(N), job_name=f"f22-att-{TAG}-g0-crescendo")
    assert r.returncode != 0 and "names no smoke" in r.stdout + r.stderr, r.stdout + r.stderr
    # the refused run (`none`, the one every case above broke) never filed; its valid sibling object closed on its own, per run, as the closer does
    assert _generated(spool, _runs(pending)[("crescendo", "none")]["run_id"]) == (None, None)
    assert _generated(spool, _runs(pend2)[("crescendo", "none")]["run_id"]) == (None, None)


# ------------------------------------------------------------------------------------------------------- static runs whole

def test_a_static_run_under_the_smoke_document_still_needs_every_shard(tmp_path, monkeypatch):
    """The knob narrows the multi-turn targets and the padding levels only: a static run registered under the same document (every smoke row
    is stamped) is held to its nine shards, as before — shard00 alone refuses, nine shards close."""
    _env_off(monkeypatch); monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(_doc(tmp_path)))
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _register(tmp_path, root, job="910005", job_name=f"f22s-{TAG}-static_base_t0", evals="pasttense")
    runs = _runs(pending)
    plan = runs[("pasttense", "none")]["plan"]; goals, variants = (int(x) for x in plan["item_set_id"].rsplit("_", 1)[1].split("x"))
    _static_store(wave, 0, "pasttense", "none", nshards=9, goals=goals, variants=variants, shards=[0])
    _static_store(wave, 0, "pasttense", "prompted__safety_refusal", nshards=9, goals=goals, variants=variants)
    r = _close(pending, spool, wave, "--nshards", "9", job_name=f"f22s-{TAG}-static_base_t0")
    out = r.stdout + r.stderr
    assert "8 of 9 expected shard(s) missing" in out, out
    assert _generated(spool, runs[("pasttense", "none")]["run_id"]) == (None, None)
    assert _generated(spool, runs[("pasttense", "prompted__safety_refusal")]["run_id"])[0] is not None


# ------------------------------------------------------------------------------------------------------- padding

PADDED_CFG_REL = f"configs/sdfbase/static_padding/sdfbase_full22_{TAG}.padstatic_obj_p0.yaml"      # the real tree's obj-family level p0


def _padded_pending(tmp_path: Path, root: Path, *, job: str, spec_key: str) -> Path:
    """A padobj registration (the steered family's level p0) on the real seeds, with the level block the sbatch exports as
    $SISTEER_REGISTRY_PADDING_JSON (jobplan._padding_block: tokens + the contract keys + cap). The config is written at the padded lane's
    RELATIVE path inside the test checkout, so the stamp (`extra.config`, relative to the checkout) resolves against the repo to the real
    level config, whose `private_run` is `pad_static_<key>_obj_p0` (registry_close_stores.pad_private_run)."""
    from sisteer.registry import emit_cli, synth
    co = tmp_path / f"checkout_{job}"; (co / "configs").mkdir(parents=True)
    os.symlink(REPO / "configs" / "registry", co / "configs" / "registry")
    cfg = co / PADDED_CFG_REL; cfg.parent.mkdir(parents=True)
    cfg.write_text(yaml.safe_dump({"explicit_cells": list(OBJECTS), "private_run": f"pad_static_{TAG}_obj_p0"}))
    pend = tmp_path / "pending"
    block = {"tokens": 0, "filler_sha256": "0" * 64, "mode": "fixed", "placement": "prefix", "separator": "\n\n", "tolerance": 0.02, "cap": 16384, "window": 98304}
    old = os.environ.get("SISTEER_REGISTRY_PADDING_JSON")
    os.environ["SISTEER_REGISTRY_PADDING_JSON"] = json.dumps(block)
    try:
        rc = emit_cli.main(["register", "--from-config", str(cfg), "--model-key", TAG, "--evals", "grok420", "--serving-path", "lora_native",
                            "--campaign", "sdfbase_full22", "--job-id", job, "--job-name", f"padobj-{TAG}-p0", "--pending-dir", str(pend),
                            "--spool", str(root / "records"), "--code-sha", synth.code_sha("a"), "--substrate", "cluster_a", "--no-push",
                            "--producer-key", f"sdfbase_full22-padobj-{TAG}-p0-sh{spec_key}"])
    finally:
        if old is None:
            os.environ.pop("SISTEER_REGISTRY_PADDING_JSON", None)
        else:
            os.environ["SISTEER_REGISTRY_PADDING_JSON"] = old
    assert rc == 0
    return pend / f"{job}.json"


def test_a_smoke_padding_level_closes_over_its_subsets_goals_and_variants(tmp_path, monkeypatch):
    """The padding level ran `shards=0-0/99`: its store is shard00 holding goals[0::99] of the item set, every variant of each. The closer
    holds it to the subset's goals x variants and files `generated` with that count; the same store under `--nshards 3` (another layout)
    or without the document refuses."""
    _env_off(monkeypatch); monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(_doc(tmp_path)))
    root = tmp_path / "reg"; _seeded_registry(root); spool = root / "records"; wave = tmp_path / "wave"
    pending = _padded_pending(tmp_path, root, job="910006", spec_key=f"0_0_{N}")
    runs = _runs(pending)
    key = next(k for k in runs if k[0] == "grok420")
    plan = runs[key]["plan"]
    goals, variants = (int(x) for x in plan["item_set_id"].rsplit("_", 1)[1].split("x")); assert goals * variants == plan["n_items_planned"]
    assert RC.pad_level_of(runs[key]) == 0, runs[key]["plan"].get("extra")
    why_p, prun = RC.pad_private_run(runs[key], repo_root=str(REPO)); assert why_p is None, why_p
    sub_goals = len(range(0, goals, N)); sub_planned = sub_goals * variants
    pad_root = wave / "static_padding" / "transcripts_results"
    for (_e, obj) in runs:
        d = pad_root / prun / "grok420" / obj / "shard00"; d.mkdir(parents=True)
        prog = [{"goal_id": f"g{i}", "outcome": "done"} for i in range(0, goals, N)]
        resp = [{"goal_id": f"g{i}", "sample_idx": v, "response": "r"} for i in range(0, goals, N) for v in range(variants)]
        (d / "progress.jsonl").write_text("".join(json.dumps(x) + "\n" for x in prog))
        (d / "responses.jsonl").write_text("".join(json.dumps(x) + "\n" for x in resp))
        (d / "served_target.json").write_text("{}")
    # another layout for the same store refuses FIRST (a closed run is skipped afterwards, so the refusal is checked on the fresh registration)
    r = _close(pending, spool, wave, "--nshards", "3", job_name=f"padobj-{TAG}-p0")
    assert r.returncode != 0 and "one store, one layout" in r.stdout + r.stderr, r.stdout + r.stderr
    for run in runs.values():
        assert _generated(spool, run["run_id"]) == (None, None)
    r = _close(pending, spool, wave, "--nshards", str(N), job_name=f"padobj-{TAG}-p0")
    assert r.returncode == 0, r.stdout + r.stderr
    for k, run in runs.items():
        st, patch = _generated(spool, run["run_id"])
        assert st is not None and patch["n_items_generated"] == sub_planned, (k, r.stdout)
        facts = json.loads(st["note"][st["note"].index("{"):])
        assert facts["smoke_goals"]["goals_expected"] == sub_goals and facts["smoke_goals"]["responses_expected"] == sub_planned
        assert facts["shards_expected"] == ["shard00"] and facts["layout"] == "padding"
    # the same store without the document refuses as before — on a fresh registry (the producer key is job-independent, so a second
    # registration under the first registry is the same run id, already generated and skipped by the closer, as it should be)
    monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    root2 = tmp_path / "reg2"; _seeded_registry(root2)
    pend2 = _padded_pending(tmp_path, root2, job="910007", spec_key=f"0_0_{N}")
    r = _close(pend2, root2 / "records", wave, "--nshards", str(N), job_name=f"padobj-{TAG}-p0")
    assert r.returncode != 0 and "98 of 99 expected shard(s) missing" in r.stdout + r.stderr, r.stdout + r.stderr
    for run in _runs(pend2).values():
        assert _generated(root2 / "records", run["run_id"]) == (None, None)
