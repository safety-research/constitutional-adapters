"""The Qwen seed-replicate battery: the served-controls / GSM8K store factory
(``served_capability_run._store_for``) attaches the registry recorder and stamps the PRODUCTION seed —
additively, only when ``$SISTEER_REGISTRY_RUN_IDENTITY`` is exported.

The identities are the REAL planner's (``registry_plans.identity_for``: ``seed_base = s``, no ``seed``),
frozen under tests/fixtures/seedrep_ws1c_identities/ with their PROVENANCE.json and re-derived BYTE FOR BYTE through the
real producer on every run — the tree's file when the W merge has brought it, else the recorded commit's file out of the
shared object store (``git show``) — never hand-written (a hand-written ``seed`` field is what hid a production
seed of 0); the record's producer commit is held to its recorded file sha (a mis-named commit is caught).

  * ``_production_seed`` reads ``seed_base`` and holds it to EVERY base the run draws under — the sampled config's
    ``seed_base``, GSM8K's ``--inspect-seed-base`` — REFUSING a disagreement (a plan saying 2 over a config saying 1
    is an error, never a silent 1); GSM8K with no sampled block -> the identity's ``seed_base``, never the template's
    0; an identity spelling the seed as ``seed`` with no ``seed_base`` REFUSES.
  * identity UNSET: no recorder, ``seed`` 0, no record, the manifest byte for byte.
  * base, resume and rung runs of the real driver under the controls MAP identity: template / manifest / every
    ``run`` record / every stored row carry s; the known-bad plan-2-over-config-1 refuses before any store exists.
  * feature-detect: the manifest lift is the complement of what the tree's ``StoreRecorder.run_payload`` reads
    through ``row_constants`` itself — PROBED on the recorder: main's REAL registry package before (9211f0de) and
    after (72faab6c, the ``_seed_base`` fallback) in a subprocess through the driver's own probe
    construction, and the tree's recorder patched into each state (no fallback: the whole template lifted; a
    ``seed_base``-only fallback: everything but ``seed``; a complete fallback: nothing the recorder reads) — and
    ``run.seed_base`` is s in every state.

Offline; the sampled controls world of tests/test_sampled_controls.py."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional, Tuple

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.eval import sampled_controls as scp  # noqa: E402
from sisteer.registry import hooks, emit  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import synth  # noqa: E402
from _seedrep_testlib import _load  # noqa: E402
from seedrep_history_fixtures import pinned_commit_or_skip  # noqa: E402
from test_sampled_controls import META, SEED_BASE, _cell_dir, _meta_rows, sampled, world  # noqa: E402,F401

FIXTURES = REPO / "tests" / "fixtures" / "seedrep_ws1c_identities"
PROVENANCE = json.loads((FIXTURES / "PROVENANCE.json").read_text())
PRODUCER_PATH = REPO / PROVENANCE["producer_path"]              # the producer's file, on the tree once merged
#: the commit PROVENANCE first mis-named (its file hashes differently) — the KNOWN-BAD provenance the check must catch
MISRECORDED_PRODUCER_COMMIT = "eda9fba2ea94fcec621dd64374d04a3debe8bb67"
CONTROLS_ROUTES = {"judged/xstest_v2", "judged/or_bench_hard_1k", "s2parity/mmlu_letter"}
GREEDY = {"do_sample": False, "max_new_tokens": 8192}        # a GSM8K / tinycap child's gen: no sampled block
XSTEST_META = META["xstest"]
ROW_KEYS = ("sweep", "run_id", "model_id", "model_revision", "arm_name", "kind", "coeff", "intervention_hash", "engine",
            "trial_id", "seed", "attempt", "harness_config_hash", "env_digest", "slurm_job_id")
#: row constants the manifest carries natively (the lift's setdefault never touches them)
NATIVE_MANIFEST_KEYS = {"model_id", "model_revision", "intervention_hash", "harness_config_hash", "engine"}
#: what this tree's ``StoreRecorder.run_payload`` reads off the spec into the run record (hooks.py) — the
#: expectation the patched-fallback states below are checked against
RECORDER_READS = {"seed", "coeff", "trial_id", "env_digest", "slurm_job_id", "attempt", "arm_name", "kind", "engine"}


def frozen_identity(kind: str, s: int) -> dict:
    """The real producer's identity for the ``none`` arm at seed ``s``: ``gsm8k`` = one identity (route ``*``),
    ``controls`` = the per-task map ``{store task: identity}`` the controls child exports."""
    return json.loads((FIXTURES / f"{kind}__none__s{s}.json").read_text())


def _sampled_gen(seed_base: int) -> dict:
    return {"do_sample": True, "seed_base": seed_base, "max_new_tokens": 64}


def _records(spool: Path):
    out = []
    if spool.exists():
        for p in R.iter_record_files(str(spool)):
            for _, rec, _sha in R.read_record_file(p):
                out.append(rec)
    return out


def _runs(spool: Path):
    return [r["payload"] for r in _records(spool) if r["record_type"] == "run"]


def _manifest(cell_dir: Path) -> dict:
    return json.loads((cell_dir / "manifest.json").read_text())


def _cfg_with_seed_base(sampled, seed_base):
    cfg = yaml.safe_load(Path(sampled.cfg_path).read_text())
    cfg["sampled_decode"]["seed_base"] = seed_base
    p = sampled.tmp_path / f"sampled_seed{seed_base}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _argv(sampled, out_root, cfg_path=None, *extra):
    argv = sampled.argv(out_root, *extra)
    if cfg_path is not None:
        argv[argv.index("--config") + 1] = str(cfg_path)
    return argv


def _export(monkeypatch, tmp_path, identity, spool: Path) -> None:
    """The container's environment: the plan file (``registry_plans.child_env`` exports its path, content as written)
    and the attempt spool; nothing disables emission. The cleanup runs FIRST, before any store exists (it once sat
    after a ``return`` in ``_file_first_rows`` and never ran): an inherited ``SISTEER_REGISTRY_DISABLED`` /
    ``REGISTRY_EMIT=0`` would stop the recorder these tests assert on, and an inherited ``SISTEER_REGISTRY_PUSH`` would
    hand the synthetic rows' run records to the LIVE push script -- emission stays in the temporary spool, always."""
    for var in (emit.DISABLED_ENV, hooks.EMIT_ENV, emit.PUSH_SCRIPT_ENV):
        monkeypatch.delenv(var, raising=False)
    p = tmp_path / "planned.json"
    p.write_text(json.dumps(identity))
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(p))
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))


def _file_first_rows(out_root: Path) -> Path:
    """main 062f2281 (#207, native emission by default): a store registered for SEVERAL evals (the controls child's
    ``{task: identity}`` map) opens each run at its FIRST ROUTED ROW, never at the manifest alone — and the generate-only
    controls child files no controls row itself (the judged tier files them downstream), so right after ``srun.main`` the
    spool holds no ``run`` yet. This routes ONE row per registered task through the SAME store ``srun.main`` wrote (re-opened
    on its key; ``CellStore(recorder=None)`` resolves main's default recorder off ``$SISTEER_REGISTRY_RUN_IDENTITY`` and the
    manifest's ``row_constants`` — the template ``_store_for`` filed, seed included), exactly what the row-filer does at
    grading time — and what the contract guarantees is the SEED on that template and on every run record that opens,
    whenever it opens (``_store_for``'s own docstring: on a self-registering tree the store resolves main's default and the
    run records carry ``seed_base`` from the template's ``seed``). Returns the cell dir."""
    cell_dir = _cell_dir(out_root)
    rc = _manifest(cell_dir)["row_constants"]
    store = cs.CellStore(out_root, cell_dir.name, row_constants=rc)
    store.write_cell_manifest(_manifest(cell_dir))                        # a resume re-validates the manifest (on_manifest)
    for i, task in enumerate(sorted(CONTROLS_ROUTES)):
        store.append_row(cs.ResultRow(sweep=rc["sweep"], run_id=rc["run_id"], model_id=rc["model_id"], model_revision=rc["model_revision"],
                                      arm_name=rc["arm_name"], kind=rc["kind"], coeff=rc["coeff"], intervention_hash=rc["intervention_hash"],
                                      engine=rc["engine"], tier=1, task=task, task_version="v1", domain="overrefusal_judged",
                                      item_id=f"probe-{i:03d}", trial_id=rc["trial_id"], seed=rc["seed"], attempt=rc["attempt"], position_order=i,
                                      metric="over_refusal", value=0.0, extra_json=None, harness_config_hash=rc["harness_config_hash"],
                                      env_digest=rc["env_digest"], slurm_job_id=rc.get("slurm_job_id"), created_at=1755000000.0 + i, wall_clock_s=None))
    return cell_dir


def _template(tmp_path) -> dict:
    """The factory's row template (identity absent), the shape the probe is keyed on."""
    return srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / "shape", ihash="a" * 16, chash="b" * 16,
                           model_id="m", model_revision="r" * 40)[1]


@pytest.fixture(autouse=True)
def _fresh_probe():
    """The recorder probe is memoised per process; every test reads the tree's (or its patched) recorder afresh."""
    srun._recorder_row_constant_reads.cache_clear()
    yield
    srun._recorder_row_constant_reads.cache_clear()


# --------------------------------------------------------------------------- the fixtures are the producer's

def _git_out(args: list, branch: str, what: str) -> bytes:
    """``git <args>`` against this repo's object store (every worktree and fetched branch shares it) for ``what``, whose commit
    is ``args[1]``'s: present, or the exact-absence rule of ``seedrep_history_fixtures.pinned_commit_or_skip`` (a skip only for a
    source-only tree or a shallow clone; a Git access failure or a full checkout lacking it FAILS naming ``branch``)."""
    commit = str(args[1]).split(":", 1)[0]
    pinned_commit_or_skip(commit, branch, what, repo=REPO, anchor=Path(__file__))
    proc = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)
    if proc.returncode != 0:
        pytest.fail(f"git {' '.join(args)} failed — {what} (fetch origin {branch} first): {proc.stderr.decode()[-300:]}")
    return proc.stdout


def _producer_bytes_at(commit: str, path: str) -> bytes:
    """The producer's file at ``commit`` (the producer's branch, fetched into the same repository)."""
    return _git_out(["show", f"{commit}:{path}"], PROVENANCE["producer_branch"], f"the producer's {path} at {commit[:8]}")


def check_provenance(prov: Mapping) -> None:
    """PROVENANCE names a commit AND the sha256 of the producer's file there; the two must agree, or the record
    is not saying which producer wrote the fixtures."""
    got = hashlib.sha256(_producer_bytes_at(prov["producer_commit"], prov["producer_path"])).hexdigest()
    assert got == prov["producer_file_sha256"], (
        f"PROVENANCE.json names producer commit {prov['producer_commit'][:8]} but its {prov['producer_path']} hashes to "
        f"{got[:16]}…, not the recorded {prov['producer_file_sha256'][:16]}… — the record mis-names the producer")


def real_producer(tmp_path: Path):
    """The producer's ``registry_plans`` module — the tree's file when present, else PROVENANCE's commit's file via ``git show``,
    loaded by path (the seedrep suite's ``_load``; the copy sits under a repo-shaped directory since the module's
    ``REPO`` is ``parents[3]``); this tree's ``sisteer`` either way."""
    src = PRODUCER_PATH
    if not src.is_file():
        src = tmp_path / PROVENANCE["producer_path"]
        src.parent.mkdir(parents=True)
        src.write_bytes(_producer_bytes_at(PROVENANCE["producer_commit"], PROVENANCE["producer_path"]))
    return _load(f"registry_plans_{PROVENANCE['producer_commit'][:8]}", str(src))


def test_the_frozen_identities_are_the_real_producers_seed_base_and_no_seed():
    assert PROVENANCE["contract_read_by_the_factory"] == {"seed field": "seed_base", "absent": "seed"}
    for s in (1, 2, 3):                                                                 # history-independent: outside the skipped portion
        g = frozen_identity("gsm8k", s)
        assert g["seed_base"] == s == g["extra"]["s"] and "seed" not in g and g["eval_id"] == "gsm8k"
        assert g["extra"]["store_task"] == "inspect/gsm8k_paired200" and g["producer_object_key_legacy"] == "none"
        assert hooks.normalise_identities(g) == {"*": g} and not hooks.missing_identity_fields(g)
        c = frozen_identity("controls", s)
        assert set(c) == CONTROLS_ROUTES and hooks.normalise_identities(c) == c
        for route, ident in c.items():
            assert ident["seed_base"] == s and "seed" not in ident and ident["extra"]["store_task"] == route
            assert not hooks.missing_identity_fields(ident) and ident["run_id"]
        assert {i["eval_id"] for i in c.values()} == {"xstest", "or_bench_hard", "mmlu"}
    check_provenance(PROVENANCE)                                                        # the named commit wrote these bytes (object store)


def test_known_bad_a_provenance_naming_a_commit_whose_file_hashes_differently_is_refused():
    # the record as it was first written: an earlier commit's id over a later commit's file sha
    with pytest.raises(AssertionError, match=r"mis-names the producer"):
        check_provenance({**PROVENANCE, "producer_commit": MISRECORDED_PRODUCER_COMMIT})


def test_the_frozen_identities_are_re_derived_byte_for_byte_through_the_real_producer(tmp_path):
    RP = real_producer(tmp_path)
    inputs = PROVENANCE["inputs"]
    inp = RP.load_inputs(campaign=inputs["campaign"], w_sha=inputs["w_sha"], policy=REPO / inputs["policy"],
                         roster=REPO / inputs["roster"], pins=REPO / inputs["pins"], arms=REPO / inputs["arms"],
                         arditi_pins=REPO / inputs["arditi_pins"], seeds_dir=REPO / inputs["seeds_dir"])
    for s in (1, 2, 3):
        none = next(a for a in RP.arms_for_seed(inp, s) if a.kind == "none")
        leg = RP.LEGS["gsm8k"]
        fields, prov = RP.instrument_fields(inp, leg, "gsm8k")
        live = RP.identity_for(inp, leg, "gsm8k", none, s, fields, prov["instrument_id"], inp.task_names["gsm8k"]("gsm8k"))
        # the file IS the producer's output, through the producer's own serialiser
        assert RP._canonical_bytes(live) == (FIXTURES / f"gsm8k__none__s{s}.json").read_bytes() and RP.check_identity(live)
        routes = {}
        for leg_name in RP.STORE_LEGS["controls"]:
            leg = RP.LEGS[leg_name]
            eid = leg.eval_ids[0]
            fields, prov = RP.instrument_fields(inp, leg, eid)
            task = inp.task_names[leg_name](eid)
            routes[task] = RP.identity_for(inp, leg, eid, none, s, fields, prov["instrument_id"], task)
        assert RP._canonical_bytes(routes) == (FIXTURES / f"controls__none__s{s}.json").read_bytes()
        assert all(RP.check_identity(i) for i in routes.values())


# --------------------------------------------------------------------------- the pure seed rule
def test_production_seed_reads_seed_base_holds_it_to_every_effective_base_and_refuses_disagreements():
    g2 = frozen_identity("gsm8k", 2)
    c1, c2 = frozen_identity("controls", 1), frozen_identity("controls", 2)
    # GSM8K: no sampled block; the Inspect replicate base in force -> the identity's seed_base, never the template's 0
    assert srun._production_seed(g2, GREEDY, inspect_seed_base=2) == 2
    assert srun._production_seed(g2, GREEDY) == 2
    with pytest.raises(SystemExit, match=r"disagrees across the registry identity's seed_base 2, --inspect-seed-base 1"):
        srun._production_seed(g2, GREEDY, inspect_seed_base=1)
    # the sampled controls (a per-task map): the config's base must be the plan's
    assert srun._production_seed(c1, _sampled_gen(1)) == 1
    with pytest.raises(SystemExit, match=r"registry identity's seed_base 2, the config's sampled_decode\.seed_base 1"):
        srun._production_seed(c2, _sampled_gen(1))                                  # plan 2 over config 1: an error, never 1
    with pytest.raises(SystemExit, match=r"several seed_base values \[1, 2\]"):
        srun._production_seed({**c1, "judged/or_bench_hard_1k": c2["judged/or_bench_hard_1k"]}, _sampled_gen(1))
    # no identity: the effective base; nothing in force: None
    assert srun._production_seed(None, _sampled_gen(SEED_BASE)) == SEED_BASE and srun._production_seed(None, GREEDY) is None
    assert srun._production_seed(None, None, inspect_seed_base=3) == 3 and srun._production_seed(None, None) is None
    # an identity that declares no seed base takes the effective base; two effective bases must agree
    stripped = {k: v for k, v in g2.items() if k != "seed_base"}
    assert srun._production_seed(stripped, _sampled_gen(1)) == 1
    assert srun._production_seed(stripped, GREEDY, inspect_seed_base=3) == 3 and srun._production_seed(stripped, GREEDY) is None
    with pytest.raises(SystemExit, match=r"disagrees across the config's sampled_decode\.seed_base 1, --inspect-seed-base 2"):
        srun._production_seed(stripped, _sampled_gen(1), inspect_seed_base=2)
    # KNOWN-BAD — the spelling that hid P1-1: `seed` with no `seed_base` (the old hand-written fixtures) REFUSES,
    # instead of being read as the seed or skipped past to 0 — alone and as a route of a map
    legacy = {**stripped, "seed": 2}
    with pytest.raises(SystemExit, match=r"declares seed=2 but no seed_base"):
        srun._production_seed(legacy, GREEDY)
    with pytest.raises(SystemExit, match=r"route 'judged/xstest_v2'\) declares seed=2 but no seed_base"):
        srun._production_seed({"judged/xstest_v2": legacy}, _sampled_gen(2))
    assert srun._production_seed({**g2, "seed": 9}, GREEDY) == 2                    # `seed` beside `seed_base` is not consulted


# --------------------------------------------------------------------------- the real driver
def test_identity_unset_no_recorder_seed_zero_no_record_and_the_manifest_shape_as_it_was(sampled, monkeypatch, tmp_path):
    monkeypatch.delenv(hooks.RUN_IDENTITY_ENV, raising=False)
    spool = tmp_path / "spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(spool))                                    # a spool alone changes nothing
    assert srun._registry_run_identity() is None
    srun.main(_argv(sampled, sampled.base_root))
    m = _manifest(_cell_dir(sampled.base_root))
    rc = m["row_constants"]
    assert rc["seed"] == 0 and rc["kind"] == "cell" and set(rc) == set(ROW_KEYS)
    assert not (set(rc) - NATIVE_MANIFEST_KEYS) & set(m) and m["engine"] == "vllm"   # nothing lifted without the identity
    assert _records(spool) == []
    store, template = srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / "f", ihash="a" * 16, chash="b" * 16,
                                      model_id="m", model_revision="r" * 40)
    assert store.recorder is None and template["seed"] == 0 and store.row_constants["seed"] == 0


def test_base_run_under_the_real_controls_identity_stamps_s_on_template_manifest_every_run_record_and_the_rows(
        sampled, monkeypatch, tmp_path):
    spool = tmp_path / "spool"
    ident = frozen_identity("controls", 1)
    _export(monkeypatch, tmp_path, ident, spool)
    got = srun._registry_run_identity()
    assert got == ident and set(hooks.normalise_identities(got)) == CONTROLS_ROUTES
    # the call site threads the Inspect replicate base into the cross-check (None here: no Inspect task is filed —
    # the offline controls world cannot run under `--inspect-seed-policy replicate`, so the wiring is checked here)
    seen = {}
    real = srun._production_seed

    def spy(identity, gen, **kw):
        seen.update(kw)
        return real(identity, gen, **kw)
    monkeypatch.setattr(srun, "_production_seed", spy)
    srun.main(_argv(sampled, sampled.base_root, _cfg_with_seed_base(sampled, 1)))     # the s1 view's config
    assert seen == {"inspect_seed_base": None}
    cell_dir = _cell_dir(sampled.base_root)
    m = _manifest(cell_dir)
    rc = m["row_constants"]
    assert rc["seed"] == 1                                                              # the template
    lift = srun._manifest_row_constant_lift(rc)                                          # this tree's recorder decides the lift
    assert all(m[k] == v for k, v in lift.items()) and set(lift) | RECORDER_READS >= set(rc) - NATIVE_MANIFEST_KEYS
    assert m["intervention_hash"] == rc["intervention_hash"]                           # an existing key is never overridden
    # nothing opens at the MANIFEST of a multi-eval store; a judged route's run opens when its GENERATED items are
    # filed (the store tells the recorder the spooled generations -- one null-valued item_result each, so the completion union holds the
    # plan's count), so right after srun.main the two judged runs exist, seed_base 1; mmlu spools nothing and opens at its first routed row
    gen_runs = _runs(spool)
    assert {r["extra"]["route"] for r in gen_runs} == {"judged/xstest_v2", "judged/or_bench_hard_1k"} and len(gen_runs) == 2
    assert all(r["seed_base"] == 1 for r in gen_runs)
    _file_first_rows(sampled.base_root)                                                  # ...each run opens at its FIRST routed row
    all_runs = _runs(spool)
    runs = {r["run_id"]: r for r in all_runs}
    assert set(runs) == {i["run_id"] for i in ident.values()} and len(runs) == 3
    assert len(all_runs) == 3                                                            # ONE `run` payload per logical run per spool -- the grading-time
    #                                                                                      re-open (a second CellStore over the cell dir) files `running` again, never a second `run`
    for route, i in ident.items():
        run = runs[i["run_id"]]
        assert run["seed_base"] == 1 == i["seed_base"] and run["eval_id"] == i["eval_id"]
        assert run["producer_object_key_legacy"] == "none" and run["extra"]["s"] == 1 and run["extra"]["route"] == route
        assert run["extra"]["store_key"] == cell_dir.name and run["decode_seed_rule"] == i["decode_seed_rule"]
        assert run["strength_coefficient"] == rc["coeff"] and run["env_digest"] == rc["env_digest"]
        assert run["extra"]["arm_name_legacy"] == rc["arm_name"] and run["extra"]["kind_legacy"] == "cell"
        assert run["intervention_hash"] == m["intervention_hash"] and run["harness_config_hash"] == m["harness_config_hash"]
    recs = _records(spool)
    assert "running" in {r["payload"]["state"] for r in recs if r["record_type"] == "run_state"}
    # spools at generate time; the JUDGED rows are the judge's (the probe rows above sit buffered, unflushed). The generate-time
    # item_results are exactly the null-valued GENERATED placeholders of the two judged routes -- one per spooled generation, value None,
    # judgment_id '', metric `generated` -- never a judged value
    gen_items = [r["payload"] for r in recs if r["record_type"] == "item_result"]
    assert gen_items and all(it["metric_id"] == hooks.GENERATED_METRIC_ID and it["value"] is None and it["judgment_id"] == "" for it in gen_items)
    assert {it["run_id"] for it in gen_items} == {ident["judged/xstest_v2"]["run_id"], ident["judged/or_bench_hard_1k"]["run_id"]}
    meta = _meta_rows(cell_dir, XSTEST_META)
    assert meta and all(r["sampling_seed"] == srun.sampling_seed(1, 0, "xstest", r["item_id"]) for r in meta)
    # the ROW path on the factory's store: a row appended under the template lands as item_result.extra.seed == 1
    store, template = srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / "f", ihash="a" * 16, chash="b" * 16,
                                      model_id=synth.MODEL_ID, model_revision="c" * 40, seed=1, registry_identity=got,
                                      resolved={"driver": "served_capability_run_v2"})
    assert isinstance(store.recorder, hooks.StoreRecorder) and template["seed"] == 1
    store.write_cell_manifest({**template, "row_constants": dict(template)})
    store.append_row(cs.ResultRow(**{k: template[k] for k in ROW_KEYS}, tier=1, task="judged/xstest_v2", task_version="v2",
                                  domain="safety", item_id="xst0000-aaaaaaaa", position_order=None, metric="compliance",
                                  value=1.0, extra_json=None, created_at=1755000000.0, wall_clock_s=None))
    store.recorder.flush()
    items = [r["payload"] for r in _records(spool) if r["record_type"] == "item_result" and r["payload"]["metric_id"] != hooks.GENERATED_METRIC_ID]
    assert len(items) == 1 and items[0]["extra"]["seed"] == 1 and items[0]["item_id"] == "xst0000-aaaaaaaa"
    assert items[0]["run_id"] == ident["judged/xstest_v2"]["run_id"]


def test_on_a_self_registering_store_module_the_factory_attaches_nothing(tmp_path, monkeypatch):
    """main @ 525607e3 (#207): ``CellStore`` resolves its own default recorder and exports ``NO_RECORDER``; the
    factory then attaches nothing — one recorder, never two — while the production seed still rides the template."""
    import unittest.mock as um
    ident = frozen_identity("controls", 2)
    with um.patch.object(cs, "NO_RECORDER", object(), create=True):
        seen = {}
        real_init = cs.CellStore.__init__

        def spy(self, *a, **kw):
            seen["recorder"] = kw.get("recorder")
            seen["seed"] = kw["row_constants"]["seed"]
            return real_init(self, *a, **kw)
        with um.patch.object(cs.CellStore, "__init__", spy):
            srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / "g", ihash="a" * 16, chash="b" * 16,
                            model_id=synth.MODEL_ID, model_revision="c" * 40, seed=2, registry_identity=ident,
                            resolved={"driver": "served_capability_run_v2"})
    assert seen == {"recorder": None, "seed": 2}


def test_known_bad_plan_seed_base_2_over_config_seed_base_1_refuses_before_any_store(sampled, monkeypatch, tmp_path):
    spool = tmp_path / "spool"
    _export(monkeypatch, tmp_path, frozen_identity("controls", 2), spool)
    with pytest.raises(SystemExit, match=r"registry identity's seed_base 2, the config's sampled_decode\.seed_base 1"):
        srun.main(_argv(sampled, sampled.base_root, _cfg_with_seed_base(sampled, 1)))
    assert not sampled.base_root.exists() and sampled.posted == [] and not spool.exists()
    # the fixture's own campaign base under the same plan: the same refusal, nothing written
    with pytest.raises(SystemExit, match=r"seed_base 2, the config's sampled_decode\.seed_base 20260909"):
        srun.main(_argv(sampled, sampled.base_root))
    assert not sampled.base_root.exists() and sampled.posted == [] and not spool.exists()


def test_resume_and_a_rung_under_the_identity_carry_s(sampled, monkeypatch, tmp_path):
    monkeypatch.setattr(scp, "CONTROLS_CAP_LADDER", (64, 128, 256))
    spool = tmp_path / "spool"
    ident = frozen_identity("controls", 1)
    _export(monkeypatch, tmp_path, ident, spool)
    cfg1 = _cfg_with_seed_base(sampled, 1)
    srun.main(_argv(sampled, sampled.base_root, cfg1))
    base_cell = _file_first_rows(sampled.base_root)                                      # main #207: the runs open at their first routed rows
    n_posted = len(sampled.posted)
    assert n_posted and len(_runs(spool)) == 3
    # RESUME: the same argv under the same identity — no re-draw; the manifest re-validates with seed 1; a fresh recorder finds the three
    # runs in the spool and says `running` for each again -- never a second `run` payload (one per logical run per spool)
    srun.main(_argv(sampled, sampled.base_root, cfg1))
    assert len(sampled.posted) == n_posted and _manifest(base_cell)["row_constants"]["seed"] == 1
    _file_first_rows(sampled.base_root)
    runs = _runs(spool)
    assert len(runs) == 3 and {r["seed_base"] for r in runs} == {1}
    assert {r["run_id"] for r in runs} == {i["run_id"] for i in ident.values()}
    base_manifest = str(base_cell / "manifest.json")
    running = [r["payload"] for r in _records(spool) if r["record_type"] == "run_state" and r["payload"]["state"] == "running"]
    assert len([st for st in running if st["evidence_uri"] == base_manifest]) >= 6                     # the base opening + the resume's re-open, per run
    # RUNG: 64 -> 128 from the completed base, under the identity; the rung's gen keeps the plan's base, so the
    # cross-check passes and the rung store's template / run records carry 1
    rung = sampled.tmp_path / "controls_cap128"
    srun.main(_argv(sampled, rung, cfg1, "--controls-cap", "128", "--escalate-from-root", str(sampled.base_root)))
    assert len(sampled.posted) > n_posted and all(req["max_tokens"] == 128 for req in sampled.posted[n_posted:])
    rung_cell = _file_first_rows(rung)                                                    # the rung store's runs, at their first routed rows
    assert _manifest(rung_cell)["row_constants"]["seed"] == 1 and rung_cell.name != base_cell.name
    # the rung store opens the SAME three logical runs this spool already holds -- `running` with the RUNG's manifest as evidence,
    # never a fourth `run` payload (a production rung is its own attempt, its own spool, and mints there); the rung's rows carry seed 1
    assert len(_runs(spool)) == 3 and {r["run_id"] for r in _runs(spool)} == {i["run_id"] for i in ident.values()}
    rung_manifest = str(rung_cell / "manifest.json")
    rung_open = [r["payload"] for r in _records(spool) if r["record_type"] == "run_state" and r["payload"]["evidence_uri"] == rung_manifest]
    # 2 (the rung's own generation opened the two judged runs, 2a) + 3 (the rung's first-rows re-open, every route) = 5 `running` events, 3 runs
    assert len(rung_open) == 5 and {st["run_id"] for st in rung_open} == {i["run_id"] for i in ident.values()} and all(st["state"] == "running" for st in rung_open)
    rung_meta = _meta_rows(rung_cell, XSTEST_META)
    assert rung_meta and all(r["sampling_seed"] == srun.sampling_seed(1, 0, "xstest", r["item_id"]) for r in rung_meta)
    # a rung planned at seed 2 over the same s1 base refuses before a store exists
    _export(monkeypatch, tmp_path, frozen_identity("controls", 2), spool)
    rung2 = sampled.tmp_path / "controls_cap128_plan2"
    with pytest.raises(SystemExit, match=r"registry identity's seed_base 2, the config's sampled_decode\.seed_base 1"):
        srun.main(_argv(sampled, rung2, cfg1, "--controls-cap", "128", "--escalate-from-root", str(sampled.base_root)))
    assert not rung2.exists()


# ---------------------------------------------------------------------------- the cleanup runs; emission stays local
def _sentinel_push_script(tmp_path: Path) -> Tuple[Path, Path]:
    """A stand-in for the LIVE ``push_records.sh`` a wave shell may leave in ``$SISTEER_REGISTRY_PUSH``: its only act is to
    append its argv to a marker file, so a marker present after a run means synthetic records reached a push."""
    marker = tmp_path / "push_invoked.txt"
    script = tmp_path / "push_records_sentinel.sh"
    script.write_text(f'#!/bin/sh\necho "$@" >> "{marker}"\nexit 0\n')
    script.chmod(0o755)
    return script, marker


def test_export_clears_the_inherited_registry_switches_before_the_plan_file_is_exported(monkeypatch, tmp_path):
    """The three-variable cleanup is the FIRST act of ``_export`` -- before the plan file,
    before any store -- and ``_file_first_rows`` carries no unreachable tail."""
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    monkeypatch.setenv(hooks.EMIT_ENV, "0")
    monkeypatch.setenv(emit.PUSH_SCRIPT_ENV, str(tmp_path / "nonexistent_push.sh"))
    assert not hooks.emission_enabled()                                                  # the inherited environment WOULD stop the recorder
    _export(monkeypatch, tmp_path, frozen_identity("controls", 1), tmp_path / "spool")
    assert not {emit.DISABLED_ENV, hooks.EMIT_ENV, emit.PUSH_SCRIPT_ENV} & set(os.environ)
    assert hooks.emission_enabled() and hooks.resolve_spool() == str(tmp_path / "spool")
    assert srun._registry_run_identity() == frozen_identity("controls", 1)
    src = inspect.getsource(_export)
    assert src.index("monkeypatch.delenv") < src.index('"planned.json"')                 # the order is pinned: cleanup, then the door
    assert "delenv" not in inspect.getsource(_file_first_rows)


@pytest.mark.parametrize("inherited", [{emit.DISABLED_ENV: "1"}, {hooks.EMIT_ENV: "0"}, {emit.DISABLED_ENV: "1", hooks.EMIT_ENV: "0"}],
                         ids=["registry_disabled", "emit_off", "both"])
def test_under_inherited_disable_switches_and_a_live_push_callback_emission_stays_in_the_temporary_spool_and_the_callback_never_fires(
        sampled, monkeypatch, tmp_path, inherited):
    """Through the real driver: with ``SISTEER_REGISTRY_DISABLED=1`` / ``REGISTRY_EMIT=0`` and a
    push callback inherited from the shell, the run records still land in the temporary spool (the recorder starts) and the
    callback is never invoked (nothing reaches a live registry). KNOWN-BAD first, through production's own writer: under the
    inherited callback alone, ``RecordWriter.push`` DOES call it -- the sentinel is a real detector, not a tautology."""
    script, marker = _sentinel_push_script(tmp_path)
    monkeypatch.setenv(emit.PUSH_SCRIPT_ENV, str(script))
    for var in (emit.DISABLED_ENV, hooks.EMIT_ENV):
        monkeypatch.delenv(var, raising=False)
    w = emit.RecordWriter(str(tmp_path / "probe_spool"), "probe-campaign", "run-probe", {"script": "probe"})
    assert w.push_script == str(script) and not w.disabled
    w.emit_run({"run_id": "run-probe", "campaign": "probe-campaign"})
    assert w.push(final=True) == 0 and marker.exists() and marker.read_text().strip().endswith("--final")
    marker.unlink()
    for k, v in inherited.items():
        monkeypatch.setenv(k, v)
    assert not hooks.emission_enabled()                                                  # the shell's environment, as inherited
    spool = tmp_path / "spool"
    _export(monkeypatch, tmp_path, frozen_identity("controls", 1), spool)
    srun.main(_argv(sampled, sampled.base_root, _cfg_with_seed_base(sampled, 1)))
    _file_first_rows(sampled.base_root)
    runs = _runs(spool)
    assert len(runs) == 3 and {r["seed_base"] for r in runs} == {1}                    # emission: local, in the temporary spool
    assert not marker.exists()                                                           # the inherited callback never fired
    assert not (tmp_path / "probe_spool").exists() or _runs(tmp_path / "probe_spool") == [{"run_id": "run-probe", "campaign": "probe-campaign"}]


# ---------------------------------------------------------------------------- the row_constants fallback, probed
def _with_fallback(monkeypatch, keys):
    """The recorder as a tree WITH the ``row_constants`` fallback for exactly ``keys`` reads it: each
    covered row constant is promoted from the manifest's nested ``row_constants`` when the top level lacks it, and the
    nested block is then hidden from the tree's own ``run_payload`` — so the state under test is exactly the one
    declared, ``keys=()`` (no fallback) included, whichever registry state this tree carries."""
    orig = hooks.StoreRecorder.run_payload

    def patched(self, route):
        spec = dict(self.spec)
        rc = spec.pop("row_constants", None) or {}
        spec.update({k: rc[k] for k in keys if k in rc and spec.get(k) is None})
        saved, self.spec = self.spec, spec
        try:
            return orig(self, route)
        finally:
            self.spec = saved
    monkeypatch.setattr(hooks.StoreRecorder, "run_payload", patched)


def test_the_probe_is_memoised_and_the_lift_is_the_templates_complement(monkeypatch, tmp_path):
    template = _template(tmp_path)
    shape = srun._template_shape(template)
    _with_fallback(monkeypatch, ())                                                     # a recorder with NO fallback
    assert srun._recorder_row_constant_reads(shape) == frozenset()
    assert srun._recorder_row_constant_reads(shape) is srun._recorder_row_constant_reads(shape)     # memoised
    assert srun._manifest_row_constant_lift(template) == template                        # the whole template lifted
    srun._recorder_row_constant_reads.cache_clear()
    _with_fallback(monkeypatch, ("seed",))
    assert srun._recorder_row_constant_reads(shape) == {"seed"}
    assert srun._manifest_row_constant_lift(template) == {k: v for k, v in template.items() if k != "seed"}
    srun._recorder_row_constant_reads.cache_clear()
    _with_fallback(monkeypatch, ROW_KEYS)
    covered = srun._recorder_row_constant_reads(shape)
    assert RECORDER_READS <= covered and not {"sweep", "run_id"} & covered              # never read by the recorder at all
    assert set(srun._manifest_row_constant_lift(template)) == set(template) - covered


#: main before and after the ``StoreRecorder._seed_base`` change (``run.seed_base`` falls back to the
#: store's row constants) — the two REAL registry states the probe must tell apart, on main's own package
MAIN_PRE_212 = "9211f0de1699c66921c4066eacd46fd0aa12e8b3"
MAIN_POST_212 = "72faab6ca00befc1ad1a6cf03fea1543cfce01ec"
_PROBE_DRIVER = """
import json, sys
from sisteer.registry import hooks
a = json.load(sys.stdin)
r = hooks.StoreRecorder("probe", a["spec"], a["identities"], **a["kwargs"])
print("PAYLOAD " + json.dumps(r.run_payload("*"), default=str))
"""


def _registry_package_at(sha: str, tmp_path: Path) -> Path:
    """main's ``sisteer.registry`` package at ``sha``, extracted with ``git archive`` — the real code, never a copy
    typed here; a clone that has not fetched main fails loudly."""
    dest = tmp_path / f"registry_{sha[:8]}"
    dest.mkdir()
    archive = _git_out(["archive", sha, "src/sisteer/__init__.py", "src/sisteer/registry"], "main", f"main's registry package at {sha[:8]}")
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    return dest / "src"


def _under(src: Path, driver: str, stdin: dict, tag: str, env: Optional[dict] = None) -> dict:
    """Run ``driver`` with the registry package under ``src`` on the path — a subprocess, so THAT package and not
    this tree's is the one imported — feeding it ``stdin`` as JSON and reading back the JSON it prints after
    ``tag``."""
    env = {**os.environ, **(env or {}), "PYTHONPATH": str(src)}
    proc = subprocess.run([sys.executable, "-c", driver], text=True, capture_output=True, env=env, timeout=300,
                          input=json.dumps(stdin))
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith(tag + " "))
    return json.loads(line[len(tag) + 1:])


def _probe_payload_under(src: Path, sentinels: dict) -> dict:
    """The probe recorder's ``run`` payload as the registry package under ``src`` builds it — the driver's exact
    construction (``_PROBE_IDENTITY`` / ``_PROBE_RECORDER_KWARGS``)."""
    return _under(src, _PROBE_DRIVER, {"spec": {"row_constants": sentinels}, "identities": {"*": srun._PROBE_IDENTITY},
                                       "kwargs": srun._PROBE_RECORDER_KWARGS}, "PAYLOAD")


def test_the_probe_tells_mains_real_row_constants_fallback_212_from_main_before_it(tmp_path):
    template = _template(tmp_path)
    shape = srun._template_shape(template)
    sentinels = srun._probe_sentinels(shape)
    before = srun._covered_reads(sentinels, _probe_payload_under(_registry_package_at(MAIN_PRE_212, tmp_path), sentinels))
    after_payload = _probe_payload_under(_registry_package_at(MAIN_POST_212, tmp_path), sentinels)
    after = srun._covered_reads(sentinels, after_payload)
    assert "seed" not in before and "seed" in after and after - before == {"seed"}    # #212 adds exactly the seed read
    assert after_payload["seed_base"] == sentinels["seed"]                              # run.seed_base through row_constants


# ---------------------------------------------------------------------------- the template seed is what main files
_MAIN_STORE_DRIVER = """
import json, sys
from sisteer.registry import hooks
a = json.load(sys.stdin)
rec = hooks.default_store_recorder(a["key"], a["template"], script="served_capability_run._store_for")
assert rec is not None, "main's default recorder did not resolve off the exported plan file"
rec.on_manifest(a["manifest"], a["manifest_path"])
print("PAYLOADS " + json.dumps({route: rec.run_payload(route) for route in rec.identities}, default=str))
"""


def _factory_template(tmp_path: Path, identity, seed) -> dict:
    """The row template THIS tree's factory builds under ``identity`` with an explicit ``seed`` (0 rebuilds today's
    hardcoded template, which the fixed factory no longer produces under an identity)."""
    return srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / f"template_{seed}", ihash="a" * 16, chash="b" * 16,
                           model_id=synth.MODEL_ID, model_revision="c" * 40, seed=seed, registry_identity=identity,
                           resolved={"driver": "served_capability_run_v2"})[1]


def _main_run_payloads(src: Path, tmp_path: Path, template: Mapping, identity: Mapping, *, lift: bool) -> dict:
    """The ``run`` payloads main's REAL recorder (the package under ``src``) writes for this factory's ``template``
    under ``identity``, driven as main's ``CellStore`` drives it: ``default_store_recorder(key, row_constants=
    template)`` off the exported plan file, ``on_manifest`` with the manifest as this driver writes it (the row
    constants nested; ``lift`` adds the top-level copies), then per route the payload ``_open`` emits."""
    work = Path(tempfile.mkdtemp(dir=tmp_path))
    (work / "plan.json").write_text(json.dumps(identity))
    manifest = {**{k: template[k] for k in NATIVE_MANIFEST_KEYS}, "row_constants": dict(template)}
    if lift:
        manifest.update({k: v for k, v in template.items() if k not in NATIVE_MANIFEST_KEYS})
    env = {hooks.RUN_IDENTITY_ENV: str(work / "plan.json"), emit.SPOOL_ENV: str(work / "spool"),
           **{var: "" for var in (emit.DISABLED_ENV, hooks.EMIT_ENV, emit.PUSH_SCRIPT_ENV)}}
    return _under(src, _MAIN_STORE_DRIVER, {"key": "controls-s2", "template": template, "manifest": manifest,
                                            "manifest_path": str(work / "manifest.json")}, "PAYLOADS", env)


def test_the_factory_resolves_the_child_env_and_the_config_itself_when_the_caller_threads_nothing(monkeypatch, tmp_path):
    """In the producer's real-factory test shape: ``_store_for(cell, cfg, out, ihash=, chash=, model_id=,
    model_revision=)`` and nothing else, under the environment a wave child carries — the template's ``seed`` is the
    production seed (the exported plan's ``seed_base``, held to the config's ``sampled_decode.seed_base``) and the store
    records under that identity; KNOWN-BAD: a config at 1 under the s2 plan refuses before any store exists; identity
    unset: ``seed 0`` — today's template, the additive-only default (the producer's premise pin holds there by design)."""
    ident = frozen_identity("controls", 2)
    _export(monkeypatch, tmp_path, ident, tmp_path / "spool")

    def cfg(seed_base):          # a controls config the way the lane's sampled configs carry it
        return {"scale": 1.0, "attacks": {"controls_max_new_tokens": 512},
                "sampled_decode": {"arm": "vendor_recommended", "trial": 0, "n_trials": 1, "seed_base": seed_base, "controls_cap": 512,
                                   "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "applies_to": ["controls"]}}

    def factory(c, out):
        return srun._store_for(SimpleNamespace(id="none"), c, out, ihash="a" * 16, chash="b" * 16, model_id=synth.MODEL_ID,
                               model_revision="c" * 40)
    assert srun._gen_cfg(cfg(2))["seed_base"] == 2 and srun._gen_cfg(cfg(2))["do_sample"] is True
    store, template = factory(cfg(2), tmp_path / "s2")
    assert template["seed"] == 2 == store.row_constants["seed"]
    assert isinstance(store.recorder, hooks.StoreRecorder) and set(store.recorder.identities) == CONTROLS_ROUTES
    with pytest.raises(SystemExit, match=r"registry identity's seed_base 2, the config's sampled_decode\.seed_base 1"):
        factory(cfg(1), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    monkeypatch.delenv(hooks.RUN_IDENTITY_ENV)
    store0, template0 = factory(cfg(2), tmp_path / "none")
    assert template0["seed"] == 0 and store0.recorder is None                            # no identity: today's template


@pytest.mark.parametrize("lift", [False, True], ids=["row_constants_nested_only", "lifted_to_the_top_level"])
def test_an_s2_controls_child_files_seed_base_2_through_mains_real_recorder_from_the_template_not_the_identity(tmp_path, lift):
    """Main's recorder resolves ``run.seed_base`` from the ROW CONSTANT before the planned
    identity (``_seed_base``: the spec's ``seed``, else the row constants'; an identity key the payload already has is
    never copied over it), so the factory's TEMPLATE must carry the production seed. Through main's REAL package at
    72faab6c (its hooks.py is byte-identical on origin/main at 015489fe): the s2 template files 2, lift or no lift;
    KNOWN-BAD: today's ``seed 0`` template under the SAME s2 identity files 0 — the identity's ``seed_base`` does not
    rescue it."""
    ident = frozen_identity("controls", 2)
    production = _factory_template(tmp_path, ident, srun._production_seed(ident, _sampled_gen(2)))
    hardcoded = _factory_template(tmp_path, ident, 0)
    assert production["seed"] == 2 and hardcoded["seed"] == 0 and {k: v for k, v in production.items() if k != "seed"} == \
        {k: v for k, v in hardcoded.items() if k != "seed"}
    src = _registry_package_at(MAIN_POST_212, tmp_path)
    for template, filed in ((production, 2), (hardcoded, 0)):
        payloads = _main_run_payloads(src, tmp_path, template, ident, lift=lift)
        assert set(payloads) == CONTROLS_ROUTES
        assert {p["seed_base"] for p in payloads.values()} == {filed}
        for route, p in payloads.items():
            assert p["run_id"] == ident[route]["run_id"] and p["eval_id"] == ident[route]["eval_id"] and p["extra"]["s"] == 2
            assert p["strength_coefficient"] == (template["coeff"] if lift else None)   # the OTHER row constants still need the lift


@pytest.mark.parametrize("fallback", [(), ("seed",), ROW_KEYS], ids=["none", "seed_base_only", "complete"])
def test_a_row_constants_fallback_skips_the_lift_for_exactly_what_it_covers_and_run_seed_base_is_s(
        sampled, monkeypatch, tmp_path, fallback):
    _with_fallback(monkeypatch, fallback)
    spool = tmp_path / "spool"
    _export(monkeypatch, tmp_path, frozen_identity("controls", 1), spool)
    srun.main(_argv(sampled, sampled.base_root, _cfg_with_seed_base(sampled, 1)))
    m = _manifest(_cell_dir(sampled.base_root))
    rc = m["row_constants"]
    assert rc["seed"] == 1 and m["engine"] == "vllm"                                    # the template always carries s
    lifted = {k for k in set(rc) - NATIVE_MANIFEST_KEYS if k in m}
    # the lift is the complement of what the fallback lets the recorder read back: everything, everything but seed,
    # nothing the recorder reads
    assert lifted == set(rc) - NATIVE_MANIFEST_KEYS - (set(fallback) & RECORDER_READS)
    assert all(m[k] == rc[k] for k in lifted)
    _file_first_rows(sampled.base_root)                                                  # main #207: the runs open at their first routed rows
    runs = _runs(spool)
    assert len(runs) == 3 and {r["seed_base"] for r in runs} == {1}                    # s in EVERY state, lifted or read back
    for run in runs:
        assert run["strength_coefficient"] == rc["coeff"] and run["env_digest"] == rc["env_digest"]
        assert run["extra"]["arm_name_legacy"] == rc["arm_name"] and run["extra"]["kind_legacy"] == "cell"
        assert run["extra"]["engine_legacy"] == "vllm" and run["trial_index"] == 0 and run["extra"]["attempt"] == 0


# ---------------------------------------------------------------------------- a present but unusable identity door refuses
def _door(monkeypatch, tmp_path, content):
    """``$SISTEER_REGISTRY_RUN_IDENTITY`` exported at a plan file with ``content`` (None: the file is not there)."""
    p = tmp_path / "door" / "planned.json"
    p.parent.mkdir(exist_ok=True)
    if content is not None:
        p.write_text(content)
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, str(p))
    return p


@pytest.mark.parametrize("content, why", [
    (None, "cannot be read"),
    ("{not json", "is not JSON"),
    ("{}", "names no planned run"),
    ('{"plan_schema": 1, "campaign": "x"}', "names no planned run"),
    ("[1, 2]", "names no planned run"),
], ids=["missing_file", "malformed_json", "empty_object", "no_identity_fields", "not_an_object"])
def test_a_present_but_unusable_identity_door_refuses_before_any_store_or_request(sampled, monkeypatch, tmp_path, content, why):
    """The variable PRESENT but unloadable used to read as absence — an UNREGISTERED seed-0 store under
    requests still drawn at s. Present now means: nonempty path, JSON, a nonempty normalised route set; anything
    else is SystemExit before the store exists or the first request goes out — from the pure switch, from the
    factory with nothing threaded, and from the real driver."""
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "spool"))
    _door(monkeypatch, tmp_path, content)
    with pytest.raises(SystemExit, match=why):
        srun._registry_run_identity()
    with pytest.raises(SystemExit, match=why):
        srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / "f", ihash="a" * 16, chash="b" * 16, model_id="m",
                        model_revision="r" * 40)
    assert not (tmp_path / "f").exists()
    with pytest.raises(SystemExit, match=why):
        srun.main(_argv(sampled, sampled.base_root, _cfg_with_seed_base(sampled, 2)))
    assert not sampled.base_root.exists() and sampled.posted == [] and not (tmp_path / "spool").exists()


def test_an_empty_identity_door_refuses_and_a_truly_unset_one_is_todays_store(monkeypatch, tmp_path):
    monkeypatch.setenv(hooks.RUN_IDENTITY_ENV, "")
    with pytest.raises(SystemExit, match="is empty"):
        srun._registry_run_identity()
    monkeypatch.delenv(hooks.RUN_IDENTITY_ENV)
    assert srun._registry_run_identity() is None                                       # genuinely absent: today's store
    store, template = srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / "f", ihash="a" * 16, chash="b" * 16,
                                      model_id="m", model_revision="r" * 40)
    assert template["seed"] == 0 and store.recorder is None


# ---------------------------------------------------------------------------- a door whose route the recorder would refuse
@pytest.mark.parametrize("plan, route, missing_first", [
    ({"eval_id": "xstest"}, "'*'", "object_id"),                                                        # a partial SINGLE identity
    (lambda s2: {"judged/xstest_v2": {}, "judged/or_bench_hard_1k": s2["judged/or_bench_hard_1k"]},
     "'judged/xstest_v2'", "object_id"),                                                              # a MIXED map: one empty route
], ids=["partial_single_identity", "mixed_map_with_an_empty_route"])
def test_a_door_whose_route_the_recorder_would_refuse_as_incomplete_refuses_before_any_store_or_request(
        sampled, monkeypatch, tmp_path, plan, route, missing_first):
    """``normalise_identities`` admits a route that is present and UNUSABLE — ``hooks.store_recorder`` then drops the
    WHOLE map as incomplete (a route with no ``run_id`` and ``missing_identity_fields``) while the driver runs on, unregistered.
    The door now applies the recorder's own rule to every route; a hit is SystemExit naming the route and its missing fields —
    before the store exists or the first request goes out, on all three surfaces."""
    plan = plan(frozen_identity("controls", 2)) if callable(plan) else plan
    spool = tmp_path / "spool"
    _export(monkeypatch, tmp_path, plan, spool)
    why = rf"registry recorder would refuse as incomplete — route {re.escape(route)} lacks \['{missing_first}'"
    with pytest.raises(SystemExit, match=why):
        srun._registry_run_identity()
    with pytest.raises(SystemExit, match=why):
        srun._store_for(SimpleNamespace(id="none"), {}, tmp_path / "f", ihash="a" * 16, chash="b" * 16, model_id="m",
                        model_revision="r" * 40)
    assert not (tmp_path / "f").exists()
    with pytest.raises(SystemExit, match=why):
        srun.main(_argv(sampled, sampled.base_root, _cfg_with_seed_base(sampled, 2)))
    assert not sampled.base_root.exists() and sampled.posted == [] and not spool.exists()
    # the recorder's rule is the one applied: the same plan is what hooks.store_recorder itself declines
    assert hooks.store_recorder("k", {}, plan, registry_emit=True, spool=str(spool)) is None
    # the rule is per route: a route that carries a run_id is registered already and passes without the fields
    assert hooks.normalise_identities(frozen_identity("controls", 2))["judged/or_bench_hard_1k"]["run_id"]
