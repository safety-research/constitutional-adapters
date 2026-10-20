"""The native-emission regression test through the REAL store factory.

``scripts/served_capability_run.py::_store_for`` is the served-controls driver's ONE ``CellStore`` construction (not
edited here). The earlier emission test drove ``hooks.store_recorder`` directly, so it stayed green while the production
factory emitted nothing. This file exercises the factory itself, under exactly the environment a wave child carries
(``SISTEER_REGISTRY_RUN_IDENTITY`` = the controls store map, ``SISTEER_REGISTRY_SPOOL`` = the attempt spool).

What main provides: ``CellStore(...)`` built with NO recorder argument resolves
``hooks.default_store_recorder`` from the job's pending identity, so the factory records without a line of change on the
merged tree. Those tests run only where that mechanism is present (``cs.NO_RECORDER`` / ``hooks.default_store_recorder``);
on a tree cut before the merge they are SKIPPED with the reason — not xfailed, because their failing there is the base's
age, not a defect.

What main does NOT provide, and the branch does (234e95d9; xfail(strict=True) here so it turns into a loud XPASS
the moment it merges): this base's factory hardcodes ``"seed": 0`` while the recorder files ``run.seed_base`` from that
template BEFORE it looks at the identity — so a plan declaring ``seed_base`` 1 / 2 / 3 lands as ``seed_base 0``. The contract
the branch implemented: ``_store_for`` ITSELF resolves the child's exported plan (``_registry_run_identity()`` off
``$SISTEER_REGISTRY_RUN_IDENTITY``, independent of ``SISTEER_REGISTRY_DISABLED``) and derives the template seed with
``_production_seed(identity, _gen_cfg(cfg))`` — the identity's ``seed_base`` (the plan's literal s) held equal to
``controls_gen_config(cfg)["seed_base"]`` for sampled controls (and ``--inspect-seed-base`` for GSM8K) — so a caller threading
nothing (this file's exact call shape) files ``seed_base`` = s; identity ABSENT keeps ``seed`` 0 byte for byte (additive-only,
deliberate). The two former ``WS1C_SEED`` strict xfails were REMOVED once both branches landed in one tree
(22289de6 and ce01745a: they XPASS); the premise test (identity absent -> 0) stays true by design and is kept. The identities here are the producer's own (``registry_plans.build`` over the three controls legs), never
hand-written.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import registry_plans as RP  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.tiers.judged import TASK_ORBENCH, TASK_XSTEST  # noqa: E402
from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU  # noqa: E402
from sisteer.registry import emit, hooks, identity, synth  # noqa: E402
from sisteer.registry import records as R  # noqa: E402

MAIN_DEFAULT_RECORDER = hasattr(cs, "NO_RECORDER") and hasattr(hooks, "default_store_recorder")
needs_main = pytest.mark.skipif(not MAIN_DEFAULT_RECORDER, reason=(
    "main's default store recorder (PR #207, 525607e3: CellStore resolves hooks.default_store_recorder when built with no recorder) "
    "is not on this tree; the test runs on the reseed ∪ main merge"))

W = synth.code_sha("c")
W7 = W[:7]
CAMPAIGN = "synthbattery"
SEED = 2
VIEW = f"s{SEED}"
REV = "0123456789abcdef0123456789abcdef01234567"
IHASH, CHASH = "a1b2c3d4e5f60718", "0f1e2d3c4b5a6978"
CELL_ID = "constv2ctl2-full__lora_r4_ce"
TASKS = {TASK_XSTEST: ("xstest", "xstest"), TASK_ORBENCH: ("or_bench", "or_bench_hard"), PARITY_TASK_MMLU: ("mmlu", "mmlu")}


@functools.lru_cache(maxsize=None)
def _real_controls_map() -> dict:
    """The producer's OWN identities (never hand-written): ``registry_plans.build`` over the three controls legs, the
    ``_stores/controls/<arm>__s<s>.json`` map the wave's controls child exports — keyed by the writers' task names, one run per eval,
    ``seed_base`` = the view's literal training seed. Built once per session (~10 s); the committed export resolves the two controls rows."""
    inp = RP.load_inputs(campaign=CAMPAIGN, w_sha=W)
    m = RP.build(inp, ["xstest", "or_bench", "mmlu"]).files[RP.store_relpath("controls", CELL_ID, SEED)]
    assert set(m) == set(TASKS), (sorted(m), sorted(TASKS))                       # the writers' task names route the rows
    for task, (leg, eval_id) in TASKS.items():
        ident = m[task]
        assert (ident["eval_id"], ident["seed_base"], ident["extra"]["container"], ident["extra"]["arm"], ident["extra"]["view"], ident["extra"]["leg"]) == \
               (eval_id, SEED, "controls", CELL_ID, VIEW, leg), ident
    return m


def _controls_map(root: Path) -> tuple:
    m = _real_controls_map()
    p = root / "plans" / RP.store_relpath("controls", CELL_ID, SEED)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    return p, m


def _sampled_cfg(seed_base: int) -> dict:
    """A controls config the way the lane's sampled configs carry it: the vendor preset + the view's seed base."""
    return {"scale": 1.0, "attacks": {"controls_max_new_tokens": 512},
            "sampled_decode": {"arm": "vendor_recommended", "trial": 0, "n_trials": 1, "seed_base": seed_base, "controls_cap": 512,
                               "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "applies_to": ["controls"]}}


def _drive(store: cs.CellStore, template: dict, n: int = 2) -> None:
    """Exactly what the driver does to a store: manifest, rows per task, results LAST."""
    store.write_cell_manifest({**template, "tiers": [1], "cell_index": 0})
    for task, (leg, _eval) in TASKS.items():
        for i in range(1, n + 1):
            store.append_row(cs.ResultRow(sweep=template["sweep"], run_id=template["run_id"], model_id=template["model_id"],
                                          model_revision=template["model_revision"], arm_name=template["arm_name"], kind=template["kind"],
                                          coeff=template["coeff"], intervention_hash=template["intervention_hash"], engine=template["engine"],
                                          tier=1, task=task, task_version="v1", domain="overrefusal_judged", item_id=f"{leg}-q{i:03d}",
                                          trial_id=0, seed=template["seed"], attempt=0, position_order=i, metric="over_refusal", value=0.0,
                                          extra_json=None, harness_config_hash=template["harness_config_hash"], env_digest=template["env_digest"],
                                          slurm_job_id=None, created_at=1755000000.0 + i, wall_clock_s=None))
    store.write_results({})


def _records(spool: Path):
    return [rec for p in R.iter_record_files(str(spool)) for _i, rec, _s in R.read_record_file(p)]


@pytest.fixture
def child_env(tmp_path, monkeypatch):
    """The environment a wave child carries: the identity door and the attempt-unique spool; emission forced on."""
    for k in (emit.DISABLED_ENV, emit.PUSH_SCRIPT_ENV, "SLURM_JOB_ID", "SLURM_JOB_STATE"):
        monkeypatch.delenv(k, raising=False)
    plan, m = _controls_map(tmp_path)
    spool = tmp_path / "attempt" / "registry" / "records"
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(plan))
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))
    monkeypatch.setenv(hooks.EMIT_ENV, "1")
    monkeypatch.setattr(emit.socket, "gethostname", lambda: synth.TESTHOST)
    if hasattr(hooks, "_DEFAULT_NOTICED"):
        hooks._DEFAULT_NOTICED.clear()
    return {"plan": plan, "map": m, "spool": spool, "out": tmp_path / "exp"}


def _factory(cfg: dict, out_root: Path):
    """The REAL driver's store factory, called with nothing but the base keywords — the exact shape the contract guarantees for
    ANY caller under the child env: ``_store_for`` (234e95d9) itself resolves the exported plan (``_registry_run_identity()``) and derives
    the template seed (``_production_seed(identity, _gen_cfg(cfg))``); the production caller additionally threads the same identity /
    seed / resolved config explicitly, which changes nothing. On this base (no such contract) the call is today's: ``seed`` 0 whatever
    the env says — the two strict xfails below hold here and XPASS the moment the contract merges."""
    import served_capability_run as srun   # the real driver; imported the way tests/test_sampled_controls.py does
    return srun._store_for(SimpleNamespace(id=CELL_ID), cfg, out_root, ihash=IHASH, chash=CHASH, model_id="qwen35_4b_it", model_revision=REV)


@needs_main
def test_the_real_factory_records_every_controls_run_under_the_child_env(child_env):
    """Main's mechanism through the production factory: three runs, rows routed by the writers' task names, nothing when disabled."""
    store, template = _factory(_sampled_cfg(SEED), child_env["out"])
    assert store.recorder is not None, "CellStore built by _store_for resolved no recorder under SISTEER_REGISTRY_RUN_IDENTITY"
    _drive(store, template)
    recs = _records(child_env["spool"])
    want = {t: i["run_id"] for t, i in child_env["map"].items()}
    runs = {r["payload"]["run_id"]: r["payload"] for r in recs if r["record_type"] == "run"}
    assert set(runs) == set(want.values())                                    # one `run` per registered (object, eval)
    for t, rid in want.items():
        ident = child_env["map"][t]
        assert (runs[rid]["object_id"], runs[rid]["eval_id"], runs[rid]["producer_object_key_legacy"]) == \
            (ident["object_id"], ident["eval_id"], ident["producer_object_key_legacy"])
        assert runs[rid]["extra"]["registered_at"] == "submit_wrapper" and runs[rid]["extra"]["store_key"] == store.key
        assert runs[rid]["intervention_hash"] == IHASH and runs[rid]["harness_config_hash"] == CHASH and runs[rid]["model_revision"] == REV
    items = [r["payload"] for r in recs if r["record_type"] == "item_result"]
    assert len(items) == 3 * 2 and all(it["run_id"] == want[it["extra"]["task"]] for it in items)   # routed by the store's task name
    gen = [r["payload"] for r in recs if r["record_type"] == "run_state" and r["payload"]["state"] == "generated"]
    assert sorted(g["run_id"] for g in gen) == sorted(want.values())          # the single-container store completes itself
    # KNOWN-BAD / the byte-identity half: DISABLED, the same factory builds a store with no recorder and writes no record
    import os
    os.environ[emit.DISABLED_ENV] = "1"
    try:
        off, tpl_off = _factory(_sampled_cfg(SEED), child_env["out"].with_name("exp_off"))
        assert off.recorder is None and tpl_off == template | {"env_digest": tpl_off["env_digest"]}
        _drive(off, tpl_off)
    finally:
        del os.environ[emit.DISABLED_ENV]
    assert len(_records(child_env["spool"])) == len(recs)


@needs_main
def test_the_real_factory_files_the_production_seed_base(child_env):
    """End to end: the emitted `run` records carry the view's seed base (the plan's 2), not the template's hardcoded 0."""
    cfg = _sampled_cfg(SEED)
    store, template = _factory(cfg, child_env["out"])
    _drive(store, template)
    runs = [r["payload"] for r in _records(child_env["spool"]) if r["record_type"] == "run"]
    assert len(runs) == 3 and {r["seed_base"] for r in runs} == {SEED}


def test_the_factory_template_seed_is_the_production_seed_base(child_env, monkeypatch):
    """The contract checked on EVERY tree under the wave child's env — the identity exported, the ONLY state in
    which the production caller resolves a seed at all: the row template's `seed` equals the seed base the driver's own config
    reader (`_gen_cfg` → controls_gen_config) resolves from the sampled config — the plan's literal training seed."""
    monkeypatch.setenv(emit.DISABLED_ENV, "1")        # the template alone; no emission needed
    import served_capability_run as srun
    cfg = _sampled_cfg(SEED)
    gen = srun._gen_cfg(cfg)
    assert gen["seed_base"] == SEED and gen["do_sample"] is True          # the driver's own reader sees the production seed base
    assert {i["seed_base"] for i in child_env["map"].values()} == {SEED} and hooks.identity_from_env() is not None
    _store, template = _factory(cfg, child_env["out"])
    assert template["seed"] == gen["seed_base"] == SEED


def test_the_contract_inputs_hold_on_this_tree(tmp_path, monkeypatch):
    """Pinned facts the two xfails rest on, so an XPASS cannot come from a changed premise: the driver's reader resolves the sampled
    seed base and a greedy config resolves none; the producer's identity declares seed_base == the sampled config's (the equality
    _production_seed holds them to); and with NO identity exported the factory stamps seed 0 and attaches no recorder — the
    additive-only default every existing store was written under, TRUE BY DESIGN on this base and on the merged tree alike
    (deliberately so), so this test is kept as written, not inverted. The defect itself — identity PRESENT, seed still 0 —
    is what the two strict xfails measure."""
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    monkeypatch.delenv(hooks.RUN_IDENTITY_ENV, raising=False)
    import served_capability_run as srun
    assert srun._gen_cfg(_sampled_cfg(SEED))["seed_base"] == SEED
    greedy = srun._gen_cfg({"attacks": {"controls_max_new_tokens": 512}})
    assert greedy["do_sample"] is False and "seed_base" not in greedy
    assert {i["seed_base"] for i in _real_controls_map().values()} == {SEED} == {srun._gen_cfg(_sampled_cfg(SEED))["seed_base"]}
    assert hooks.identity_from_env() is None
    store, template = _factory(_sampled_cfg(SEED), tmp_path / "exp")
    assert template["seed"] == 0 and template["engine"] == "vllm" and template["intervention_hash"] == IHASH   # identity absent: byte for byte
    assert isinstance(store, cs.CellStore) and store.row_constants == template
    assert store.recorder is None                                            # no identity (and DISABLED): no recorder on any tree
