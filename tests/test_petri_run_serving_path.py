"""``scripts/petri_run.py --serving-path`` and the runner's served-engine seams.

What is pinned here and why:

- ``--serving-path lora_native`` enumerates the EFFECTIVE native configuration:
  the manifest's launch flags pass through ``native_standup.effective_launch_
  flags`` BEFORE enumeration (so ``harness_config_hash`` is what the standup
  hashed), skip-class cells (a PSR object) are SKIPPED BY NAME and recorded,
  ``none__lens0`` is never derived and ``none__lora0`` always is.
- ``--serving-path lens`` (the default) is EXACTLY today's enumeration and
  ``cells.json`` payload — the pre-change ``dc.enumerate_cells`` call with no
  new keyword and the five-key record — so every frozen lens record still
  compares equal.
- ``--verify-standup`` + ``--verify-routing-table`` hold the enumeration to the
  standup's ``cells_served.json`` and ``routing_table.json`` cell by cell
  (vmodel, serving identity, intervention hash, harness hash, run key; equal
  sets; every route) and refuse BEFORE any leg on any disagreement; both are
  REQUIRED under native.
- ``serving_engine`` rides the cell manifest (top level, beside ``engine`` =
  the serving IDENTITY) and every row's ``extra_json``; ``row_constants`` stays
  the frozen ResultRow template; a resume into a cell dir whose manifest names
  or (by era: absent = lens) implies the other engine refuses with nothing
  written, from both the driver and the runner body; the live proxy is held to
  the native engine before the first sample.

Synthetic world throughout (the store-shape world's builders + a registry
with one PSR, one LoRA, the base and a prompted cell); no vLLM, no model
loads, no real records. CPU-only."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import petri_run as prun  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402
from test_petri_store_shape import world  # noqa: E402,F401

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402
from sisteer.capeval.serving import proxy as sproxy  # noqa: E402
from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.capeval.serving import native_residual as _nr  # noqa: E402


@pytest.fixture(autouse=True)
def _stage1_world(monkeypatch):
    """THE STAGE-1 WORLD (native-psr-serving step 7 flipped both flags): this module's
    enumeration mechanics — flag stripping, the skip RECORDED by name, the derived zero twin,
    the record bytes, the standup verification — are exercised with the RESPONSE-fit psr cell
    as the skip-class cell, which it is only while the position class is unarmed. The flags
    are pinned False here (the worker refuses to load a restricted payload while its own flag
    is False, so the two arm together); ``test_stage2_serves_the_response_psr_when_its_record_
    carries_the_header_and_refuses_when_it_does_not`` re-arms them and covers the stage-2
    world."""
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", False)
    monkeypatch.setattr(_nr, "STAGE2_POSITION_CLASSES", False)

LENS, NATIVE = tc.SERVING_ENGINE_LENS, tc.SERVING_ENGINE_LORA_NATIVE
CELLS = ["none", "prompted__safety", "si_safety__psr_kl_all", "si_safety__lora_r4"]
PSR, LORA = "si_safety__psr_kl_all", "si_safety__lora_r4"
NATIVE_IDENTITY = {"engine": {"kind": NATIVE, "enforce_eager": False, "cudagraph_mode": "FULL",
                              "adapters": {}},
                   "base_model": "meta-llama/Llama-3.1-8B-Instruct"}
LENS_IDENTITY = {"lens_reduction_mode": "stacked", "base_model": "meta-llama/Llama-3.1-8B-Instruct"}


def _bf(spec):
    return {k: f"# behavior {k}\nsynthetic eval bytes\n".encode()
            for k in cm.petri_expected_behavior_keys(spec)}


def _source(root: Path) -> dc.MappingCellSource:
    """The base, a prompted cell, a PSR object (lens-only: SKIPPED by name under
    native) and a natively servable LoRA object (the ``+lora+`` leg)."""
    root.mkdir(parents=True, exist_ok=True)
    psr, lora = root / "psr.pt", root / "lora.pt"
    psr.write_bytes(b"synthetic-psr-object-bytes")
    lora.write_bytes(b"synthetic-lora-all-object-bytes")
    return dc.MappingCellSource({
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "prompted__safety": dc.RegistryEntry(cell_id="prompted__safety", kind="prompted",
                                             si_id="safety", si_prefix=shape.SAFETY_PREFIX),
        PSR: dc.RegistryEntry(cell_id=PSR, kind="psr", si_id="safety", object_path=str(psr),
                              serving_path="lens",
                              resolved_spec={"kind": "psr", "si_id": "safety", "layers": [0, 2],
                                             "positions": "response"}),
        LORA: dc.RegistryEntry(cell_id=LORA, kind="lora_all", si_id="safety", object_path=str(lora),
                               serving_path="lora",
                               resolved_spec={"kind": "lora_all", "si_id": "safety", "layers": [0, 2],
                                              "rank": 4, "positions": "all",
                                              "response_header_len": None}),
    })


def _spec(**over) -> cm.PetriSpec:
    return shape._spec("ab" * 8, audit="petri-serving-path-test", cells=list(CELLS), **over)


def _enum_lens(spec, source):
    return dc.enumerate_cells(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)


def _path_of(cell) -> str:
    return cm.parse_serving_identity(cell.serving_identity).path


def _block(cell) -> dict:
    zero = cell.cell_id.endswith(sproxy.LORA_ZERO_CELL_SUFFIX)
    return {"name": cell.cell_id, "dir": f"/synthetic/bank/{cell.cell_id}",
            "config_sha256": hashlib.sha256(f"cfg:{cell.cell_id}".encode()).hexdigest(),
            "weights_sha256": hashlib.sha256(f"w:{cell.cell_id}".encode()).hexdigest(),
            "r": 4, "petri_intervention_hash": cell.intervention_hash,
            "petri_zero_adapter": zero, "petri_source_kind": None if zero else "lora_all"}


def _standup_evidence(tmp: Path, cell_set: dc.PetriCellSet, source, spec, *, native: bool):
    """A standup's ``cells_served.json`` + ``routing_table.json`` exported from
    the SAME enumeration (the shape ``native_standup.LoraStandup._cells_served``
    writes: ``engine``, ``canonical`` / ``controls`` as PetriCell dicts, the
    model pins) — version 2 with an adapter block per lora leg under native,
    today's version-1 document under lens."""
    tmp.mkdir(parents=True, exist_ok=True)
    record = {"model": spec.target_model, "model_revision": spec.target_revision,
              "canonical": [dataclasses.asdict(c) for c in cell_set.canonical],
              "controls": [dataclasses.asdict(c) for c in cell_set.controls]}
    if native:
        record["engine"] = NATIVE
    (tmp / "cells_served.json").write_text(json.dumps(record, indent=1))
    blocks = ({c.cell_id: _block(c) for c in cell_set.all_served() if _path_of(c) == "lora"}
              if native else None)
    sproxy.export_routing_table(cell_set, source, tmp / "routing_table.json",
                                base_model=spec.target_model, serving_mode=spec.serving_mode,
                                exact_replay_strategy="batch_invariant",
                                base_revision=spec.target_revision, adapter_blocks=blocks)
    return record, tmp / "cells_served.json", tmp / "routing_table.json"


# ---------------------------------------------------------------------------
# 1. the two enumerations
# ---------------------------------------------------------------------------


def test_stage2_serves_the_response_psr_when_its_record_carries_the_header_and_refuses_when_it_does_not(
        tmp_path, monkeypatch):
    """STAGE 2 (native-psr-serving §1.8, step 7): with the position class ARMED a RESPONSE-fit psr
    record that carries its header width is SERVED natively — a ``+lora+`` leg with no skip — and
    the same record WITHOUT the width REFUSES the enumeration by name (fail closed: the width is
    what the worker locates the boundary with; stage 1 skipped such a cell by name instead)."""
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", True)
    monkeypatch.setattr(_nr, "STAGE2_POSITION_CLASSES", True)
    spec, source = _spec(), _source(tmp_path / "bank")
    # known-bad first: the module's record (no response_header_len) cannot be judged -> refused, named
    with pytest.raises(prun.DriverError, match=f"{PSR}.*cannot be served on the native.*response_header_len|"
                                                f"{PSR}.*cannot be judged"):
        prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    # the complete record, registered on the lora path (how the registry adapter registers a natively
    # served residual cell — the ``+lora+`` leg, like LORA above): served, no skip, the mode readable
    entry = source.get(PSR)
    complete = dataclasses.replace(entry, serving_path="lora",
                                   resolved_spec={**entry.resolved_spec, "response_header_len": 3})
    source2 = dc.MappingCellSource({**{c: source.get(c) for c in CELLS}, PSR: complete})
    native = prun.enumerate_petri_native(spec, source2, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    ids = [c.cell_id for c in native.cell_set.all_served()]
    assert PSR in ids and native.skipped == () and native.cell_set.absent == ()
    by_id = {c.cell_id: c for c in native.cell_set.all_served()}
    assert _path_of(by_id[PSR]) == "lora"
    assert dc.native_position_mode("psr", complete.resolved_spec) == ("response", 3)


def test_native_enumeration_strips_flags_skips_by_name_and_derives_the_lora_twin(tmp_path):
    spec, source = _spec(), _source(tmp_path / "bank")
    assert spec.serving_launch_flags == ["--enable-lora", "--enforce-eager"]
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec),
                                         prompt_files=shape.PROMPT_FILES)
    ids = [c.cell_id for c in native.cell_set.all_served()]
    assert ids == ["none", "prompted__safety", LORA, dc.NOOP_LORA_CELL_ID]
    assert dc.NOOP_LENS_CELL_ID not in ids and PSR not in ids
    assert native.cell_set.absent == ()
    # the skip is RECORDED by name, kind and the predicate's reason — never a silent drop
    assert [(s.cell_id, s.kind) for s in native.skipped] == [(PSR, "psr")]
    assert isinstance(native.skipped[0], ra.SkippedCell) and "psr" in native.skipped[0].reason
    # the flags the standup strips are exactly the ones dropped here (ONE derivation),
    # recorded in MANIFEST order (the record says what the manifest carried)
    assert set(native.flags_dropped) == set(ns.STRIPPED_MANIFEST_FLAGS)
    assert list(native.flags_dropped) == ["--enable-lora", "--enforce-eager"]
    assert native.effective_spec.serving_launch_flags == []
    assert native.effective_spec.cells == ["none", "prompted__safety", LORA]
    # ... so every leg is HASHED under the effective spec, not the manifest's
    by_id = {c.cell_id: c for c in native.cell_set.all_served()}
    lens_by_id = {c.cell_id: c for c in _enum_lens(spec, source).all_served()}
    for cid in ("none", "prompted__safety", LORA):
        assert by_id[cid].harness_config_hash != lens_by_id[cid].harness_config_hash
        assert by_id[cid].petri_run_key != lens_by_id[cid].petri_run_key
        assert by_id[cid].harness_config_hash == cm.petri_harness_config_hash(
            native.effective_spec, behavior_files=_bf(native.effective_spec),
            prompt_files=shape.PROMPT_FILES, serving_identity=by_id[cid].serving_identity)
    assert _path_of(by_id[LORA]) == "lora" and _path_of(by_id["none"]) == "passthrough"
    assert by_id[dc.NOOP_LORA_CELL_ID].control_of == "none"
    # the public seam spells the same set
    assert prun.enumerate_petri(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES,
                                serving_path=prun.SERVING_PATH_LORA_NATIVE) == native.cell_set


def test_lens_enumeration_and_record_are_the_pre_change_bytes(tmp_path):
    spec, source = _spec(), _source(tmp_path / "bank")
    # the lens path is EXACTLY the pre-change call — no new keyword reaches enumerate_cells
    cell_set = prun.enumerate_petri(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    assert cell_set == _enum_lens(spec, source)
    ids = [c.cell_id for c in cell_set.all_served()]
    assert ids == CELLS + [dc.NOOP_LENS_CELL_ID, dc.NOOP_LORA_CELL_ID]   # the legacy hybrid table
    # ... and the cells.json payload keeps the five pre-change keys, no more
    payload = prun._cells_payload(cell_set, spec)
    assert list(payload) == ["audit", "expected_cells", "canonical", "controls", "absent"]
    assert payload["expected_cells"] == CELLS
    # the native payload adds exactly the three additive keys and keeps the manifest's cells
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec),
                                         prompt_files=shape.PROMPT_FILES)
    npay = prun._cells_payload(native.cell_set, spec, native=native)
    assert list(npay) == ["audit", "expected_cells", "canonical", "controls", "absent",
                          "serving_path", "cells_skipped", "flags_dropped"]
    assert npay["expected_cells"] == CELLS
    assert npay["serving_path"] == prun.SERVING_PATH_LORA_NATIVE
    assert npay["cells_skipped"] == [dict(native.skipped[0]._asdict())]
    assert npay["flags_dropped"] == ["--enable-lora", "--enforce-eager"]   # manifest order
    # a record written before the flag reads as lens; a native record names its path
    prun._check_record_serving_path(payload, prun.SERVING_PATH_LENS, tmp_path / "cells.json")
    prun._check_record_serving_path(npay, prun.SERVING_PATH_LORA_NATIVE, tmp_path / "cells.json")
    with pytest.raises(prun.DriverError, match="written under --serving-path lens"):
        prun._check_record_serving_path(payload, prun.SERVING_PATH_LORA_NATIVE, tmp_path / "cells.json")
    with pytest.raises(prun.DriverError, match="written under --serving-path lora_native"):
        prun._check_record_serving_path(npay, prun.SERVING_PATH_LENS, tmp_path / "cells.json")


def test_a_passthrough_leg_shares_its_run_key_across_engines_when_nothing_is_stripped(tmp_path):
    """WHY the explicit cross-engine check exists: a manifest whose
    launch flags carry nothing the native standup strips gives the ``none`` leg
    the SAME harness hash and run key on both engines — the store's own
    identity fields cannot tell a lens-era ``none`` dir from a native one."""
    spec, source = _spec(serving_launch_flags=["--max-num-seqs", "8"]), _source(tmp_path / "bank")
    lens = {c.cell_id: c for c in _enum_lens(spec, source).all_served()}
    native = {c.cell_id: c for c in prun.enumerate_petri_native(
        spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES).cell_set.all_served()}
    assert native["none"].petri_run_key == lens["none"].petri_run_key
    assert native["none"].serving_identity == lens["none"].serving_identity
    assert native["prompted__safety"].petri_run_key == lens["prompted__safety"].petri_run_key


def test_row_constants_stay_the_frozen_column_template():
    cell = dc.PetriCell(cell_id="none", cell_class="base", kind="none", si_id=None, object_path=None,
                        virtual_model_name="petri__base",
                        serving_identity="proxy_vllm+passthrough+exact_replay",
                        intervention_hash="11" * 8, harness_config_hash="22" * 8,
                        petri_run_key="33" * 8, control_of=None, si_prefix_hash=None)
    keys = tuple(prun._row_constants(_spec(), cell, 0))
    assert keys == ("sweep", "run_id", "model_id", "model_revision", "arm_name", "kind", "coeff",
                    "intervention_hash", "engine", "trial_id", "seed", "attempt",
                    "harness_config_hash", "env_digest", "slurm_job_id")
    assert tc.SERVED_ENGINE_FIELD not in keys


# ---------------------------------------------------------------------------
# 2. the standup's evidence
# ---------------------------------------------------------------------------


def test_standup_verification_passes_on_evidence_from_the_same_enumeration(tmp_path):
    spec, source = _spec(), _source(tmp_path / "bank")
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec),
                                         prompt_files=shape.PROMPT_FILES)
    record, rec_path, table_path = _standup_evidence(tmp_path / "standup", native.cell_set, source,
                                                     spec, native=True)
    table = sproxy.load_routing_table(table_path)
    assert table["version"] == 2 and set(table["adapters"]) == {LORA, dc.NOOP_LORA_CELL_ID}
    assert prun.standup_verification_problem(native.cell_set, prun.load_standup_record(rec_path), table,
                                             serving_path=prun.SERVING_PATH_LORA_NATIVE, spec=spec) is None
    # the lens enumeration against a lens standup's version-1 evidence passes too
    lens_set = _enum_lens(spec, source)
    _, lrec, ltable = _standup_evidence(tmp_path / "lens-standup", lens_set, source, spec, native=False)
    assert sproxy.load_routing_table(ltable)["version"] == 1
    assert prun.standup_verification_problem(lens_set, prun.load_standup_record(lrec),
                                             sproxy.load_routing_table(ltable),
                                             serving_path=prun.SERVING_PATH_LENS, spec=spec) is None


def test_standup_verification_names_every_disagreement(tmp_path):
    spec, source = _spec(), _source(tmp_path / "bank")
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec),
                                         prompt_files=shape.PROMPT_FILES)
    record, _rec_path, table_path = _standup_evidence(tmp_path / "standup", native.cell_set, source,
                                                      spec, native=True)
    table = sproxy.load_routing_table(table_path)
    NP = prun.SERVING_PATH_LORA_NATIVE

    def problem(rec=record, tab=table, path=NP, cells=native.cell_set):
        return prun.standup_verification_problem(cells, rec, tab, serving_path=path, spec=spec)

    # per-field disagreements, each named with both values
    for field in prun.STANDUP_CELL_FIELDS:
        rec = copy.deepcopy(record)
        rec["canonical"][2][field] = "ff" * 8
        p = problem(rec)
        assert p is not None and f"cell {LORA!r}: {field}" in p and "ff" * 8 in p
    # the served sets: a leg the standup did not serve / a leg it serves that is not enumerated
    rec = copy.deepcopy(record)
    dropped = rec["canonical"].pop(2)
    assert f"NOT served by the standup: ['{LORA}']" in problem(rec)
    rec = copy.deepcopy(record)
    rec["controls"].append({**dropped, "cell_id": "si_safety__extra"})
    assert "NOT enumerated here: ['si_safety__extra']" in problem(rec)
    # a missing route; a route whose identity disagrees
    routes = list(table["routes"])
    assert f"NOT routed by the routing table: ['{LORA}']" in problem(
        tab={**table, "routes": tuple(r for r in routes if r.cell_id != LORA)})
    bent = tuple(dataclasses.replace(r, serving_identity="proxy_vllm+lens+exact_replay")
                 if r.cell_id == LORA else r for r in routes)
    p = problem(tab={**table, "routes": bent})
    assert f"cell {LORA!r}: route serving_identity='proxy_vllm+lens+exact_replay'" in p
    # the record's engine must match the path — both ways
    assert "records engine='lora_native' but this run is --serving-path lens" in problem(path=prun.SERVING_PATH_LENS)
    rec = copy.deepcopy(record)
    del rec["engine"]
    assert "records engine=None but this run is --serving-path lora_native" in problem(rec)
    # a native run needs a version-2 table with the adapter blocks
    assert "without adapter blocks" in problem(tab={**table, "version": 1, sproxy.ADAPTERS_KEY: None})
    # the standup is THIS model's
    rec = copy.deepcopy(record)
    rec["model_revision"] = "0" * 40
    assert "model_revision='0000" in problem(rec)
    assert "base_model='other/model'" in problem(tab={**table, "base_model": "other/model"})
    # every disagreement is listed, not the first only
    rec = copy.deepcopy(record)
    rec["canonical"][0]["petri_run_key"] = "ee" * 8
    rec["canonical"][1]["harness_config_hash"] = "dd" * 8
    p = problem(rec)
    assert "disagrees with the standup's evidence in 2 way(s)" in p


def test_load_standup_record_refuses_the_pre_launch_record_and_junk(tmp_path):
    p = tmp_path / "cells_served.json"
    with pytest.raises(prun.DriverError, match="no such file"):
        prun.load_standup_record(p)
    p.write_text(json.dumps({"engine": NATIVE, "stage": "pre_launch", "canonical": [], "controls": []}))
    with pytest.raises(prun.DriverError, match="PRE-LAUNCH"):
        prun.load_standup_record(p)
    p.write_text(json.dumps({"engine": NATIVE, "canonical": []}))
    with pytest.raises(prun.DriverError, match="no 'controls' list"):
        prun.load_standup_record(p)
    p.write_text("[]")
    with pytest.raises(prun.DriverError, match="not a mapping"):
        prun.load_standup_record(p)


def test_verify_flags_are_required_under_native_and_paired_under_lens(tmp_path):
    spec, source = _spec(), _source(tmp_path / "bank")
    cell_set = _enum_lens(spec, source)
    NP, LP = prun.SERVING_PATH_LORA_NATIVE, prun.SERVING_PATH_LENS
    args = SimpleNamespace(verify_standup=None, verify_routing_table=None)
    with pytest.raises(prun.DriverError, match="requires --verify-standup and --verify-routing-table"):
        prun.verify_standup_or_refuse(args, spec, cell_set, NP)
    with pytest.raises(prun.DriverError, match="requires --verify-routing-table"):
        prun.verify_standup_or_refuse(SimpleNamespace(verify_standup="x", verify_routing_table=None),
                                      spec, cell_set, NP)
    # lens: optional, but the two go together
    prun.verify_standup_or_refuse(args, spec, cell_set, LP)
    with pytest.raises(prun.DriverError, match="go together"):
        prun.verify_standup_or_refuse(SimpleNamespace(verify_standup="x", verify_routing_table=None),
                                      spec, cell_set, LP)
    # a caller-built namespace that predates the flag reads the lens default
    assert prun.serving_path_of(SimpleNamespace()) == LP
    with pytest.raises(prun.DriverError, match="not one of"):
        prun.serving_path_of(SimpleNamespace(serving_path="hf"))


def test_parser_spells_the_top_level_flags():
    args = prun.build_parser().parse_args(
        ["--manifest", "m.yaml", "--serving-path", "lora_native", "--verify-standup", "a.json",
         "--verify-routing-table", "b.json", "print-cells"])
    assert args.serving_path == "lora_native" and args.cmd == "print-cells"
    assert args.verify_standup == Path("a.json") and args.verify_routing_table == Path("b.json")
    args = prun.build_parser().parse_args(["run", "--cell-index", "0"])
    assert args.serving_path == "lens" and args.verify_standup is None
    with pytest.raises(SystemExit):
        prun.build_parser().parse_args(["--serving-path", "hf", "print-cells"])


# ---------------------------------------------------------------------------
# 3. the driver end to end: print-cells, run, the cross-engine refusal
# ---------------------------------------------------------------------------


class _Ran(Exception):
    """The runner was reached (raised by the stub so the summary path is not walked)."""


def _wire(monkeypatch, source, *, identity, calls):
    """The real driver over the synthetic registry: the registry factory and
    the two file loaders injected, the runtime seam stubbed to an engine
    exposing ``identity`` and a ``run_petri`` that records its call and raises."""
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    monkeypatch.setattr(prun, "CELL_SOURCE_FACTORY", lambda _spec: source)
    monkeypatch.setattr(prun, "load_behavior_files", lambda s: _bf(s))
    monkeypatch.setattr(prun, "load_prompt_files", lambda _s: dict(shape.PROMPT_FILES))

    def run_petri(engine, spec, cell, store, seed):
        calls.append({"cell": cell.cell_id, "store": store,
                      "manifest": json.loads(store.manifest_path.read_text())})
        raise _Ran()

    import sisteer.capeval as _cap
    engine = SimpleNamespace(proxy_identity=identity) if identity is not None else object()
    rt_stub = SimpleNamespace(make_engine=lambda _ns: engine, run_petri=run_petri)
    monkeypatch.setitem(sys.modules, "sisteer.capeval.runtime", rt_stub)
    monkeypatch.setattr(_cap, "runtime", rt_stub, raising=False)


def _args(out_root, *, serving_path, standup=None, table=None, cell_index=None, endpoint=None):
    return SimpleNamespace(out_root=str(out_root), serving_path=serving_path,
                           verify_standup=None if standup is None else str(standup),
                           verify_routing_table=None if table is None else str(table),
                           served_endpoint=None if endpoint is None else str(endpoint),
                           cell_index=cell_index, cell_id=None, seed=None, overflow_rerun=False)


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_print_cells_and_run_under_native_record_and_stamp_the_engine(tmp_path, monkeypatch, capsys):
    spec, source, calls = _spec(), _source(tmp_path / "bank"), []
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec),
                                         prompt_files=shape.PROMPT_FILES)
    _, rec, table = _standup_evidence(tmp_path / "standup", native.cell_set, source, spec, native=True)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=calls)
    out_root = tmp_path / "out"
    NP = prun.SERVING_PATH_LORA_NATIVE

    # print-cells: refuses without the evidence, freezes the native record with it
    with pytest.raises(prun.DriverError, match="requires --verify-standup"):
        prun.cmd_print_cells(_args(out_root, serving_path=NP), spec)
    assert not (out_root / prun.CELLS_JSON).exists()
    assert prun.cmd_print_cells(_args(out_root, serving_path=NP, standup=rec, table=table), spec) == 0
    record = json.loads((out_root / prun.CELLS_JSON).read_text())
    assert record["serving_path"] == NP and record["expected_cells"] == CELLS
    assert [c["cell_id"] for c in record["canonical"]] == ["none", "prompted__safety", LORA]
    assert [c["cell_id"] for c in record["controls"]] == [dc.NOOP_LORA_CELL_ID]
    assert record["cells_skipped"][0]["cell_id"] == PSR and record["flags_dropped"] == ["--enable-lora", "--enforce-eager"]
    out = capsys.readouterr().out
    assert f"skipped by name ['{PSR}']" in out and "standup evidence verified: 4 served leg(s)" in out
    # the frozen record is immutable: the same enumeration is a no-op, a lens run refuses by name
    assert prun.cmd_print_cells(_args(out_root, serving_path=NP, standup=rec, table=table), spec) == 0
    with pytest.raises(prun.DriverError, match="written under --serving-path lora_native"):
        prun.cmd_run(_args(out_root, serving_path=prun.SERVING_PATH_LENS, cell_index=0), spec)

    # run: the leg's manifest names the engine top-level; row_constants stays the template
    with pytest.raises(_Ran):
        prun.cmd_run(_args(out_root, serving_path=NP, standup=rec, table=table, cell_index=2), spec)
    [call] = calls
    assert call["cell"] == LORA
    manifest = call["manifest"]
    assert manifest[tc.SERVED_ENGINE_FIELD] == NATIVE
    assert manifest["engine"] == "proxy_vllm+lora+exact_replay"
    assert tc.SERVED_ENGINE_FIELD not in manifest["row_constants"]
    assert pr.recorded_serving_engine(manifest) == NATIVE
    # and the live proxy is held to the engine BEFORE the runner is reached
    calls.clear()
    _wire(monkeypatch, source, identity=LENS_IDENTITY, calls=calls)
    with pytest.raises(prun.DriverError, match="not the native multi-LoRA engine"):
        prun.cmd_run(_args(out_root, serving_path=NP, standup=rec, table=table, cell_index=0), spec)
    assert calls == []


def test_run_native_into_a_lens_era_cell_dir_refuses_with_nothing_written(tmp_path, monkeypatch):
    spec, source, calls = _spec(), _source(tmp_path / "bank"), []
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec),
                                         prompt_files=shape.PROMPT_FILES)
    _, rec, table = _standup_evidence(tmp_path / "standup", native.cell_set, source, spec, native=True)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=calls)
    out_root = tmp_path / "out"
    NP = prun.SERVING_PATH_LORA_NATIVE
    assert prun.cmd_print_cells(_args(out_root, serving_path=NP, standup=rec, table=table), spec) == 0
    # a LENS-era cell dir at the native `none` leg's key: a manifest with the store's identity
    # fields and no serving_engine (written before the key existed) plus a row file
    none = native.cell_set.canonical[0]
    cell_dir = out_root / "cells" / none.petri_run_key
    cell_dir.mkdir(parents=True)
    (cell_dir / "manifest.json").write_text(json.dumps({
        "intervention_hash": none.intervention_hash, "harness_config_hash": none.harness_config_hash,
        "model_revision": spec.target_revision, "engine": none.serving_identity,
        "cell_id": none.cell_id, "petri_run_key": none.petri_run_key,
        "row_constants": prun._row_constants(spec, none, 0)}, indent=1))
    (cell_dir / "rows.jsonl").write_text('{"task": "petri", "item_id": "scenario_alpha"}\n')
    before = _snapshot(out_root)
    with pytest.raises(pr.PetriRunnerError, match="lens by era") as ei:
        prun.cmd_run(_args(out_root, serving_path=NP, standup=rec, table=table, cell_index=0), spec)
    assert "this run serves on the 'lora_native' engine" in str(ei.value)
    assert _snapshot(out_root) == before and calls == []
    # a dir whose manifest NAMES the other engine refuses the same way
    m = json.loads((cell_dir / "manifest.json").read_text())
    m[tc.SERVED_ENGINE_FIELD] = LENS
    (cell_dir / "manifest.json").write_text(json.dumps(m, indent=1))
    before = _snapshot(out_root)
    with pytest.raises(pr.PetriRunnerError, match="records serving_engine='lens'"):
        prun.cmd_run(_args(out_root, serving_path=NP, standup=rec, table=table, cell_index=0), spec)
    assert _snapshot(out_root) == before and calls == []


def test_run_lens_resumes_a_lens_era_dir_and_stamps_lens(tmp_path, monkeypatch):
    """The in-flight lens campaign: a lens run onto a cell dir written
    before the key existed validates as before and the rewritten manifest
    names ``lens``; a lens record carries no serving_path and needs no
    standup evidence."""
    spec, source, calls = _spec(), _source(tmp_path / "bank"), []
    _wire(monkeypatch, source, identity=None, calls=calls)
    out_root = tmp_path / "out"
    LP = prun.SERVING_PATH_LENS
    assert prun.cmd_print_cells(_args(out_root, serving_path=LP), spec) == 0
    record = json.loads((out_root / prun.CELLS_JSON).read_text())
    assert set(record) == {"audit", "expected_cells", "canonical", "controls", "absent", "written_at"}
    none = dc.PetriCell(**record["canonical"][0])
    cell_dir = out_root / "cells" / none.petri_run_key
    cell_dir.mkdir(parents=True)
    (cell_dir / "manifest.json").write_text(json.dumps({
        "intervention_hash": none.intervention_hash, "harness_config_hash": none.harness_config_hash,
        "model_revision": spec.target_revision, "engine": none.serving_identity,
        "cell_id": none.cell_id, "petri_run_key": none.petri_run_key,
        "row_constants": prun._row_constants(spec, none, 0)}, indent=1))
    with pytest.raises(_Ran):
        prun.cmd_run(_args(out_root, serving_path=LP, cell_index=0), spec)
    [call] = calls
    assert call["cell"] == "none" and call["manifest"][tc.SERVED_ENGINE_FIELD] == LENS
    # ... and a native run into that dir now refuses by the named engine
    calls.clear()


# ---------------------------------------------------------------------------
# 4. the runner's seams
# ---------------------------------------------------------------------------


def test_refuse_cross_engine_resume_reads_absent_as_lens(tmp_path):
    cell_dir = tmp_path / "cells" / ("ab" * 8)
    pr.refuse_cross_engine_resume(cell_dir, NATIVE)            # a fresh cell: nothing to refuse
    cell_dir.mkdir(parents=True)
    (cell_dir / "manifest.json").write_text(json.dumps({"engine": "proxy_vllm+passthrough+exact_replay"}))
    pr.refuse_cross_engine_resume(cell_dir, LENS)
    pr.refuse_cross_engine_resume(cell_dir / "manifest.json", LENS)
    with pytest.raises(pr.PetriRunnerError, match="lens by era"):
        pr.refuse_cross_engine_resume(cell_dir, NATIVE)
    (cell_dir / "manifest.json").write_text(json.dumps({tc.SERVED_ENGINE_FIELD: NATIVE}))
    pr.refuse_cross_engine_resume(cell_dir, NATIVE)
    with pytest.raises(pr.PetriRunnerError, match="records serving_engine='lora_native'"):
        pr.refuse_cross_engine_resume(cell_dir, LENS)
    with pytest.raises(pr.PetriRunnerError, match="not one of"):
        pr.refuse_cross_engine_resume({tc.SERVED_ENGINE_FIELD: "hf"}, LENS)
    with pytest.raises(pr.PetriRunnerError, match="not one of"):
        pr.refuse_cross_engine_resume({}, "hf")
    assert pr.recorded_serving_engine({}) == LENS and pr.recorded_serving_engine({tc.SERVED_ENGINE_FIELD: NATIVE}) == NATIVE


def test_resolve_serving_engine_reads_the_live_identity_and_holds_the_declaration():
    lora = dc.PetriCell(cell_id=LORA, cell_class="class_a", kind="lora_all", si_id="safety",
                        object_path=None, virtual_model_name=f"petri__{LORA}",
                        serving_identity="proxy_vllm+lora+exact_replay", intervention_hash="11" * 8,
                        harness_config_hash="22" * 8, petri_run_key="33" * 8, control_of=None,
                        si_prefix_hash=None)
    lens = dataclasses.replace(lora, cell_id=PSR, kind="psr", serving_identity="proxy_vllm+lens+exact_replay")
    base = dataclasses.replace(lora, cell_id="none", cell_class="base", kind="none", si_id=None,
                               virtual_model_name="petri__base",
                               serving_identity="proxy_vllm+passthrough+exact_replay")
    native_engine = SimpleNamespace(proxy_identity=NATIVE_IDENTITY)
    lens_engine = SimpleNamespace(proxy_identity=LENS_IDENTITY)
    assert pr.resolve_serving_engine(native_engine, lora) == NATIVE
    assert pr.resolve_serving_engine(native_engine, base) == NATIVE       # a passthrough leg's identity cannot tell
    assert pr.resolve_serving_engine(lens_engine, base) == LENS
    assert pr.resolve_serving_engine(object(), base) == LENS               # no identity block, no declaration
    assert pr.resolve_serving_engine(object(), base, NATIVE) == NATIVE     # the caller's word, nothing live to hold it to
    with pytest.raises(pr.PetriRunnerError, match="declares serving_engine='lens' but the live proxy"):
        pr.resolve_serving_engine(native_engine, base, LENS)
    with pytest.raises(pr.PetriRunnerError, match="is lens-served"):
        pr.resolve_serving_engine(native_engine, lens)
    with pytest.raises(pr.PetriRunnerError, match="torn"):
        pr.resolve_serving_engine(SimpleNamespace(proxy_identity={"engine": {"kind": NATIVE},
                                                                  tc.SERVED_ENGINE_FIELD: LENS}), base)
    with pytest.raises(pr.PetriRunnerError, match="not one of"):
        pr.resolve_serving_engine(object(), base, "hf")


def test_convert_eval_log_stamps_the_engine_into_every_row_and_none_writes_no_key(world):  # noqa: F811
    cell = world.cells["si_safety__psr_kl_all"]
    b = "self_preservation"
    samples = [shape.sample(sid, shape.full_dims(b)) for sid in shape.SIDS]
    rows, ledger = shape.convert(world, cell, b, samples, serving_engine=LENS)
    assert ledger == [] and rows
    assert all(json.loads(r.extra_json)[tc.SERVED_ENGINE_FIELD] == LENS for r in rows)
    plain, _ = shape.convert(world, cell, b, samples)
    assert all(tc.SERVED_ENGINE_FIELD not in json.loads(r.extra_json) for r in plain)
    # the rows are otherwise identical (the field is additive, inside extra_json only)
    strip = lambda r: {k: (json.dumps({kk: vv for kk, vv in json.loads(v).items()  # noqa: E731
                                       if kk != tc.SERVED_ENGINE_FIELD}, sort_keys=True)
                           if k == "extra_json" else v)
                       for k, v in dataclasses.asdict(r).items() if k not in ("created_at",)}
    assert [strip(r) for r in rows] == [strip(r) for r in plain]
    assert cs.ROW_COLUMNS == tuple(f.name for f in dataclasses.fields(cs.ResultRow))
    assert tc.SERVED_ENGINE_FIELD not in cs.ROW_COLUMNS


def test_runner_body_refuses_a_cross_engine_dir_before_any_write(world, tmp_path):  # noqa: F811
    cell = world.cells["none"]
    store = cs.CellStore(tmp_path / "sweep", cell.petri_run_key)
    store.cell_dir.mkdir(parents=True)
    (store.cell_dir / "manifest.json").write_text(json.dumps({
        "intervention_hash": cell.intervention_hash, "harness_config_hash": cell.harness_config_hash,
        "model_revision": world.spec.target_revision, "engine": cell.serving_identity,
        tc.SERVED_ENGINE_FIELD: NATIVE}))
    before = _snapshot(tmp_path)
    with pytest.raises(pr.PetriRunnerError, match="records serving_engine='lora_native'"):
        pr._run(object(), world.spec, cell, store, seed=world.spec.bloom_options.seed,
                driver=shape._driver(world, cell), config_root=world.config_root,
                bundle_dir=world.bundle.bundle_dir, work_root=tmp_path / "work",
                log_root=tmp_path / "logs", prompt_files=shape.PROMPT_FILES)
    assert _snapshot(tmp_path) == before
    # the same body, told the OTHER engine by the caller against a native identity, refuses too
    with pytest.raises(pr.PetriRunnerError, match="declares serving_engine='lens'"):
        pr._run(SimpleNamespace(proxy_identity=NATIVE_IDENTITY), world.spec, cell, store,
                seed=world.spec.bloom_options.seed, driver=shape._driver(world, cell),
                config_root=world.config_root, bundle_dir=world.bundle.bundle_dir,
                work_root=tmp_path / "work", log_root=tmp_path / "logs",
                prompt_files=shape.PROMPT_FILES, serving_engine=LENS)
    assert _snapshot(tmp_path) == before


# ---------------------------------------------------------------------------
# 5. the driver hands the RUNNER the spec the legs were hashed under
# ---------------------------------------------------------------------------


def _native_runner_world(tmp_path: Path):
    """A REAL runner world (test_petri_store_shape's builders: behavior definitions on disk under a config root, a
    frozen scenario bundle) whose manifest pins the two flags the native launch strips and lists the PSR cell the
    native standup skips, over the four-cell synthetic registry above. ``bf_of(spec)`` is the on-disk behavior map
    pruned to the spec's expected keys — what the runner hashes."""
    from sisteer.capeval.petri import scenarios as psc

    root = tmp_path / "world"
    config_root = shape._make_config_root(root / "configs")
    src = shape._make_scenario_source(root)
    bundle = psc.freeze_scenarios(src, root / "bundles", behaviors=shape.BEHAVIORS)
    spec = shape._spec(bundle.content_hash, audit="petri-serving-path-runner-test", cells=list(CELLS))
    assert spec.serving_launch_flags == ["--enable-lora", "--enforce-eager"]
    bf_all = psc.behavior_files(config_root, shape.BEHAVIORS)

    def bf_of(s):
        return {k: bf_all[k] for k in cm.petri_expected_behavior_keys(s)}

    return spec, _source(tmp_path / "bank"), config_root, bundle, bf_of


def test_run_hands_the_real_runner_the_effective_native_spec(tmp_path, monkeypatch, capsys):
    """Review r1 (d): under ``lora_native`` ``cmd_run`` must hand ``rt.run_petri`` the EFFECTIVE spec the legs were
    enumerated and hashed under (the manifest's cells minus the skipped PSR cell, ``serving_launch_flags`` minus the
    stripped ``--enable-lora`` / ``--enforce-eager``), never the manifest spec. The runtime seam here is the REAL runner
    body (``petri_runner._run`` with only the Inspect driver, the scenario bundle dir, the work/log roots and the prompt
    bytes injected — its harness-hash validation, engine resolution, resume gate and row filing all run), so the
    runner's own H10 check judges the spec the driver hands over. Before the fix the manifest spec reached it and every
    native leg refused at "harness config hash recomputed" before its first sample; now the leg COMPLETES. The second
    half pins WHY: the same runner over the manifest spec is exactly that refusal."""
    spec, source, config_root, bundle, bf_of = _native_runner_world(tmp_path)
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    monkeypatch.setattr(prun, "CELL_SOURCE_FACTORY", lambda _spec: source)
    monkeypatch.setattr(prun, "load_behavior_files", bf_of)
    monkeypatch.setattr(prun, "load_prompt_files", lambda _s: dict(shape.PROMPT_FILES))
    native = prun.enumerate_petri_native(spec)
    eff = native.effective_spec
    assert eff.serving_launch_flags == [] and list(eff.cells) == ["none", "prompted__safety", LORA]
    _, rec, table = _standup_evidence(tmp_path / "standup", native.cell_set, source, spec, native=True)

    handed = []

    def run_petri(engine, spec_in, cell, store, seed):
        handed.append(spec_in)
        # the fake Inspect driver stamps the metadata the runner itself derives from the spec it was handed
        world = SimpleNamespace(spec=spec_in, bf=bf_of(spec_in))
        return pr._run(engine, spec_in, cell, store, seed, driver=shape._driver(world, cell),
                       config_root=config_root, bundle_dir=bundle.bundle_dir, work_root=tmp_path / "work",
                       log_root=tmp_path / "logs", prompt_files=shape.PROMPT_FILES, serving_engine=NATIVE)

    import sisteer.capeval as _cap
    engine = SimpleNamespace(proxy_identity=NATIVE_IDENTITY)
    rt_stub = SimpleNamespace(make_engine=lambda _ns: engine, run_petri=run_petri)
    monkeypatch.setitem(sys.modules, "sisteer.capeval.runtime", rt_stub)
    monkeypatch.setattr(_cap, "runtime", rt_stub, raising=False)

    out_root, NP = tmp_path / "out", prun.SERVING_PATH_LORA_NATIVE
    assert prun.cmd_print_cells(_args(out_root, serving_path=NP, standup=rec, table=table), spec) == 0
    record = json.loads((out_root / prun.CELLS_JSON).read_text())
    idx = [c["cell_id"] for c in record["canonical"]].index(LORA)
    assert prun.cmd_run(_args(out_root, serving_path=NP, standup=rec, table=table, cell_index=idx), spec) == 0
    [spec_in] = handed
    assert spec_in == eff and spec_in.serving_launch_flags == [] and spec_in != spec
    cell = native.cell_set.canonical[idx]
    assert cell.cell_id == LORA and _path_of(cell) == "lora"
    store = cs.CellStore(out_root, cell.petri_run_key)
    assert store.is_done(), "the leg did not complete through the real runner"
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest[tc.SERVED_ENGINE_FIELD] == NATIVE and manifest["harness_config_hash"] == cell.harness_config_hash
    assert f"cell {LORA} COMPLETE" in capsys.readouterr().out
    # the manifest spec is what the runner REFUSES: its hash covers the stripped flags (the pre-fix hand-over)
    with pytest.raises(pr.PetriRunnerError, match="harness config hash recomputed"):
        pr._run(engine, spec, cell, cs.CellStore(tmp_path / "sweep2", cell.petri_run_key), spec.bloom_options.seed,
                driver=shape._driver(SimpleNamespace(spec=spec, bf=bf_of(spec)), cell), config_root=config_root,
                bundle_dir=bundle.bundle_dir, work_root=tmp_path / "work2", log_root=tmp_path / "logs2",
                prompt_files=shape.PROMPT_FILES, serving_engine=NATIVE)


def test_run_under_lens_hands_the_runner_the_manifest_spec_unchanged(tmp_path, monkeypatch):
    """The lens hand-over is byte-identical: ``rt.run_petri`` receives the manifest spec object itself."""
    spec, source, calls = _spec(), _source(tmp_path / "bank"), []
    handed = []

    def run_petri(engine, spec_in, cell, store, seed):
        handed.append(spec_in)
        raise _Ran()

    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    monkeypatch.setattr(prun, "CELL_SOURCE_FACTORY", lambda _spec: source)
    monkeypatch.setattr(prun, "load_behavior_files", lambda s: _bf(s))
    monkeypatch.setattr(prun, "load_prompt_files", lambda _s: dict(shape.PROMPT_FILES))
    import sisteer.capeval as _cap
    rt_stub = SimpleNamespace(make_engine=lambda _ns: SimpleNamespace(proxy_identity=LENS_IDENTITY), run_petri=run_petri)
    monkeypatch.setitem(sys.modules, "sisteer.capeval.runtime", rt_stub)
    monkeypatch.setattr(_cap, "runtime", rt_stub, raising=False)
    out_root = tmp_path / "out"
    assert prun.cmd_print_cells(_args(out_root, serving_path=prun.SERVING_PATH_LENS), spec) == 0
    with pytest.raises(_Ran):
        prun.cmd_run(_args(out_root, serving_path=prun.SERVING_PATH_LENS, cell_index=0), spec)
    assert handed == [spec] and handed[0] is spec


# ---------------------------------------------------------------------------
# the legs come from the proxy's declared cells, written by the REAL standup writer
# ---------------------------------------------------------------------------

import shutil  # noqa: E402

_ATTESTATION = {"engine": {"enforce_eager": False, "cudagraph_mode": "FULL", "cudagraph_full_capture_max": 64},
                "lora_config": {"max_loras": 4, "max_lora_rank": 16}, "id_name_map": {},
                "scheduler_geometry": {"worker_reports": [{"max_num_seqs": 48, "max_model_len": 8192}]}}
_SPEC_S2R = "petri_full22_synth_s2r"
A32, B32, C32 = "a" * 32, "b" * 32, "c" * 32


def _tt(harness_id: str) -> str:
    """The TrainTest / roster spelling ``--cell`` names for a harness id (what the front hands requested_cells_explicit)."""
    return ra.traintest_cell_id(harness_id)


def _real_standup_records(out: Path, native, source, spec, *, requested, attempt: str, served=None, write_endpoint: bool = True,
                          monkeypatch=None):
    """THE REAL WRITER: ``native_standup.LoraStandup`` built over THIS enumeration under stubs (the
    server, proxy, launch and attestation faked; everything the records derive from the enumeration real), its
    ``cells_requested_explicit`` / ``_harness`` from :func:`native_standup.requested_cells_explicit` over the roster spellings
    ``requested`` names -- exactly what ``lora_serving_up.bring_up_lora`` hands the core -- then ``write_cells_served()`` writes the
    FINAL record and ``write_endpoint()`` the readiness record beside it. ``served`` overrides the harness ids the standup reports
    as served (default: every enumerated canonical leg). Returns ``(cells_served.json, routing_table.json, endpoint.json)``. No key is
    hand-inserted anywhere; the records are the writer's bytes."""
    out.mkdir(parents=True, exist_ok=True)
    _, _, table_path = _standup_evidence(out, native.cell_set, source, spec, native=True)   # the exported version-2 table; the hand record is overwritten by the writer below
    table = json.loads(table_path.read_text())
    canonical = [c.cell_id for c in native.cell_set.canonical]
    harness_cells = tuple(served if served is not None else canonical)
    explicit, explicit_h = ns.requested_cells_explicit([_tt(h) for h in requested])
    lora = [c for c in native.cell_set.all_served() if _path_of(c) == "lora" and not c.cell_id.endswith(sproxy.LORA_ZERO_CELL_SUFFIX)]
    geometry = SimpleNamespace(max_model_len=8192, max_num_seqs=48, seqs_source="config",
                               as_dict=lambda: {"max_model_len": 8192, "max_num_seqs": 48})
    proxy = SimpleNamespace(virtual_models=lambda: [c.virtual_model_name for c in native.cell_set.all_served()])
    launch = SimpleNamespace(cudagraph_capture_sizes=(1, 2, 4), default_chat_template_kwargs=None, override_generation_config=None,
                             server_default_policy="contract_override")
    if monkeypatch is not None:
        monkeypatch.setattr(ns.socket, "gethostname", lambda: "TESTHOST")
    fields = {f.name for f in dataclasses.fields(ns.LoraStandup)}
    kw = dict(out=out, model=spec.target_model, revision=spec.target_revision, model_key="synthmodel_8b", eval="petri", geometry=geometry,
              launch=launch, server=None, proxy=proxy, proxy_loop=None, proxy_url="http://127.0.0.1:1/v1", proxy_port=1, vllm_port=2,
              table_path=table_path, table=table, tokenizer=None, manifest_path=None, config_path=None, config_used=out / "config.yaml",
              defenses_dir=out / "defenses", adapter_bank=out / "bank", cells=tuple(_tt(h) for h in harness_cells), harness_cells=harness_cells,
              lora_cells=tuple(c.cell_id for c in lora), prompted_cells_admitted=(),
              vmodels={c.cell_id: c.virtual_model_name for c in native.cell_set.all_served()},
              adapters={c.cell_id: _block(c) for c in lora}, zero_adapter={"name": dc.NOOP_LORA_CELL_ID}, adapters_reused={},
              object_artifact_sha256_by_cell={c.cell_id: hashlib.sha256(c.cell_id.encode()).hexdigest() for c in lora},
              attestation_path=out / "serving_attestation.json", attestation=_ATTESTATION, engine_identity={"versions": {"vllm": "0.0"}},
              chat_template_hash=getattr(spec, "chat_template_hash", "0" * 16), chat_template_sha256="0" * 64, manifest_flags_dropped=(),
              geometry_override_reason=None, tool_calls=True, petri_agent_loop=True, upstream_timeout_s=30.0, probe_max_tokens=16,
              max_logprobs=None, gpu_mem_util="0.9", cell_set=native.cell_set, thinking_header=None, thinking_header_error=None,
              attempt_id=attempt, mode="petri", positive_controls={"verdict": "pass"},
              cells_requested_explicit=explicit, cells_requested_explicit_harness=explicit_h)
    su = ns.LoraStandup(**{k: v for k, v in kw.items() if k in fields})
    rec = su.write_cells_served()
    ep = out / prun.ENDPOINT_JSON
    if write_endpoint:
        ep = su.write_endpoint()
    return rec, table_path, ep


def test_the_real_writer_declares_what_was_named_and_the_reader_freezes_exactly_those_legs(tmp_path, monkeypatch, capsys):
    """Round trip: the REAL native_standup writer records cells_requested_explicit (the roster
    spelling named) and cells_requested_explicit_harness (its harness id) -- `none` only when named, [] / [] when nothing was -- in the
    FINAL cells_served.json AND endpoint.json, and petri_run freezes from THAT file: exactly the declared canonical legs, no controls
    twin without `none`, the record carrying the declaration while expected_cells stays the manifest; `run` re-derives it (no drift)
    and drives leg 0 = the declared cell. Nothing is hand-inserted into any record."""
    source, calls = _source(tmp_path / "bank"), []
    spec = shape._spec("ab" * 8, audit=_SPEC_S2R, cells=list(CELLS))     # a steered-wave audit: a partial declaration IS its contract
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    rec, table, ep = _real_standup_records(tmp_path / "standup", native, source, spec, requested=[LORA], attempt=A32, monkeypatch=monkeypatch)
    written, endpoint = json.loads(rec.read_text()), json.loads(ep.read_text())
    assert written[ns.REQUESTED_EXPLICIT_HARNESS_KEY] == [LORA] == endpoint[ns.REQUESTED_EXPLICIT_HARNESS_KEY]
    assert written[ns.REQUESTED_EXPLICIT_KEY] == [_tt(LORA)] == endpoint[ns.REQUESTED_EXPLICIT_KEY] and _tt(LORA) != LORA
    assert written["cells_requested"][0] == "none" and written["harness_cells"][0] == "none"          # served implicitly ...
    assert "none" not in written[ns.REQUESTED_EXPLICIT_KEY]                                           # ... not named, so not declared
    assert written.get("stage") != "pre_launch" and written[ns.ENDPOINT_ATTEMPT_KEY] == endpoint[ns.ENDPOINT_ATTEMPT_KEY] == A32
    assert written["engine"] == NATIVE and written["model"] == spec.target_model
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=calls)
    out_root, NP = tmp_path / "out", prun.SERVING_PATH_LORA_NATIVE
    assert prun.cmd_print_cells(_args(out_root, serving_path=NP, standup=rec, table=table), spec) == 0
    record = json.loads((out_root / prun.CELLS_JSON).read_text())
    assert [c["cell_id"] for c in record["canonical"]] == [LORA] and record["controls"] == [] and record[prun.REQUESTED_HARNESS_KEY] == [LORA]
    assert record["expected_cells"] == CELLS                                                   # the manifest's universe, unchanged
    out = capsys.readouterr().out
    assert "standup evidence verified: 1 served leg(s)" in out and f"declared cells {[LORA]}" in out and "no `none`, no controls twin" in out
    assert prun.cmd_print_cells(_args(out_root, serving_path=NP, standup=rec, table=table), spec) == 0     # immutable: a no-op
    with pytest.raises(_Ran):
        prun.cmd_run(_args(out_root, serving_path=NP, standup=rec, table=table, cell_index=0), spec)
    [call] = calls
    assert call["cell"] == LORA
    with pytest.raises(prun.DriverError, match="out of range 0..0"):
        prun.cmd_run(_args(out_root, serving_path=NP, standup=rec, table=table, cell_index=1), spec)
    # `none` named too: the controls twin comes along
    rec2, table2, _ = _real_standup_records(tmp_path / "standup2", native, source, spec, requested=["none", LORA], attempt=B32)
    assert json.loads(rec2.read_text())[ns.REQUESTED_EXPLICIT_KEY] == ["none", _tt(LORA)]
    out2 = tmp_path / "out2"
    assert prun.cmd_print_cells(_args(out2, serving_path=NP, standup=rec2, table=table2), spec) == 0
    record2 = json.loads((out2 / prun.CELLS_JSON).read_text())
    assert [c["cell_id"] for c in record2["canonical"]] == ["none", LORA] and [c["cell_id"] for c in record2["controls"]] == [dc.NOOP_LORA_CELL_ID]
    # nothing named (a u0 proxy): the writer records [] / [] and the reader treats the record as undeclared -- the whole enumeration, as before
    spec0 = _spec()
    native0 = prun.enumerate_petri_native(spec0, source, behavior_files=_bf(spec0), prompt_files=shape.PROMPT_FILES)
    rec0, table0, _ = _real_standup_records(tmp_path / "standup0", native0, source, spec0, requested=[], attempt=C32)
    written0 = json.loads(rec0.read_text())
    assert written0[ns.REQUESTED_EXPLICIT_KEY] == [] == written0[ns.REQUESTED_EXPLICIT_HARNESS_KEY] and written0["cells_requested"][0] == "none"
    out0 = tmp_path / "out0"
    assert prun.cmd_print_cells(_args(out0, serving_path=NP, standup=rec0, table=table0), spec0) == 0
    record0 = json.loads((out0 / prun.CELLS_JSON).read_text())
    assert len(record0["canonical"]) == 3 and len(record0["controls"]) == 1 and prun.REQUESTED_HARNESS_KEY not in record0
    # the reader's key spellings ARE the writer's
    assert prun.REQUESTED_HARNESS_KEY == ns.REQUESTED_EXPLICIT_HARNESS_KEY and prun.REQUESTED_ROSTER_KEY == ns.REQUESTED_EXPLICIT_KEY
    assert prun.STANDUP_ATTEMPT_KEY == ns.ENDPOINT_ATTEMPT_KEY and prun.ENDPOINT_JSON == ns.ENDPOINT_FILE


def test_a_declaration_is_bound_to_the_live_attempt_before_anything_is_frozen_or_run(tmp_path, monkeypatch):
    """Two attempts of the same model over IDENTICAL routes and leg hashes with different declared
    subsets. Attempt A's served record (declares [LORA]) beside attempt B's endpoint (the same declaration, another attempt id) refuses
    by name as a stale pair; a third attempt's endpoint with ANOTHER declaration is refused for the declaration first --
    anything is frozen -- and, after a bound freeze, before a leg runs; a served record or an endpoint stating no attempt id refuses;
    no endpoint record at all refuses naming --served-endpoint and the sibling; --served-endpoint at A's own endpoint binds. Both
    attempts' records are the real writer's bytes; the stale pairing is the drive's pulled layout (one state directory)."""
    source = _source(tmp_path / "bank")
    spec = shape._spec("ab" * 8, audit=_SPEC_S2R, cells=list(CELLS))     # a steered-wave audit (partial declarations are its contract)
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=[])
    NP = prun.SERVING_PATH_LORA_NATIVE
    rec_a, table_a, ep_a = _real_standup_records(tmp_path / "A", native, source, spec, requested=[LORA], attempt=A32, monkeypatch=monkeypatch)
    shutil.copy(ep_a, tmp_path / "ep_a_pulled.json")                                                        # the drive's pulled copy of A's endpoint
    _, _, ep_b = _real_standup_records(tmp_path / "B", native, source, spec, requested=[LORA], attempt=B32)      # the same declaration, another attempt
    assert json.loads(ep_a.read_text())["routing_table"] != json.loads(ep_b.read_text())["routing_table"]      # two directories ...
    assert json.loads(table_a.read_text())["routes"] == json.loads((tmp_path / "B" / "routing_table.json").read_text())["routes"]  # ... identical routes
    # the bound pair freezes; then B's endpoint lands where A's served record lives (a restart between the drive's two pulls)
    out = tmp_path / "out"
    assert prun.cmd_print_cells(_args(out, serving_path=NP, standup=rec_a, table=table_a), spec) == 0
    shutil.copy(ep_b, tmp_path / "A" / prun.ENDPOINT_JSON)
    with pytest.raises(prun.DriverError, match=f"is attempt {A32} but the live endpoint .* is attempt {B32}"):
        prun.cmd_print_cells(_args(tmp_path / "out_stale", serving_path=NP, standup=rec_a, table=table_a), spec)
    assert not (tmp_path / "out_stale" / prun.CELLS_JSON).exists()
    # another attempt AND another declaration: the declaration disagreement is named first, nothing frozen
    _, _, ep_c = _real_standup_records(tmp_path / "C", native, source, spec, requested=["none", LORA], attempt=C32)
    shutil.copy(ep_c, tmp_path / "A" / prun.ENDPOINT_JSON)
    with pytest.raises(prun.DriverError, match=r"A/cells_served.json declares \['si_safety__lora_r4'\] but .*A/endpoint.json declares \['none', 'si_safety__lora_r4'\] -- torn"):
        prun.cmd_print_cells(_args(tmp_path / "out_stale2", serving_path=NP, standup=rec_a, table=table_a), spec)
    assert not (tmp_path / "out_stale2" / prun.CELLS_JSON).exists()
    shutil.copy(ep_b, tmp_path / "A" / prun.ENDPOINT_JSON)                                                  # back to the stale same-declaration pair
    with pytest.raises(prun.DriverError, match="not bound to the live attempt"):                              # run re-binds per leg
        prun.cmd_run(_args(out, serving_path=NP, standup=rec_a, table=table_a, cell_index=0), spec)
    # --served-endpoint at A's own endpoint (the pulled record) binds again
    assert prun.cmd_print_cells(_args(out, serving_path=NP, standup=rec_a, table=table_a, endpoint=tmp_path / "ep_a_pulled.json"), spec) == 0
    # no attempt id: on the served record (the writer wrote what it was given) and on the endpoint alone
    rec_n, table_n, _ = _real_standup_records(tmp_path / "N", native, source, spec, requested=[LORA], attempt="")
    with pytest.raises(prun.DriverError, match=f"N/cells_served.json states no {prun.STANDUP_ATTEMPT_KEY}"):
        prun.cmd_print_cells(_args(tmp_path / "out_n", serving_path=NP, standup=rec_n, table=table_n), spec)
    rec_e, table_e, ep_e = _real_standup_records(tmp_path / "E", native, source, spec, requested=[LORA], attempt=C32)
    shutil.copy(tmp_path / "N" / prun.ENDPOINT_JSON, ep_e)
    with pytest.raises(prun.DriverError, match=f"E/endpoint.json states no {prun.STANDUP_ATTEMPT_KEY}"):
        prun.cmd_print_cells(_args(tmp_path / "out_e", serving_path=NP, standup=rec_e, table=table_e), spec)
    # no endpoint record at all
    rec_m, table_m, _ = _real_standup_records(tmp_path / "M", native, source, spec, requested=[LORA], attempt=C32, write_endpoint=False)
    with pytest.raises(prun.DriverError, match="no endpoint record at .*M/endpoint.json -- pass --served-endpoint"):
        prun.cmd_print_cells(_args(tmp_path / "out_m", serving_path=NP, standup=rec_m, table=table_m), spec)
    # an undeclared u0 record needs no binding (the whole enumeration is attempt-independent and verified cell by cell)
    spec0 = _spec()
    native0 = prun.enumerate_petri_native(spec0, source, behavior_files=_bf(spec0), prompt_files=shape.PROMPT_FILES)
    rec_u, table_u, _ = _real_standup_records(tmp_path / "U", native0, source, spec0, requested=[], attempt=C32, write_endpoint=False)
    assert prun.cmd_print_cells(_args(tmp_path / "out_u", serving_path=NP, standup=rec_u, table=table_u), spec0) == 0
    # the pure predicate, both torn-records faces
    same = {prun.STANDUP_ATTEMPT_KEY: A32, prun.REQUESTED_HARNESS_KEY: [LORA]}
    assert prun.attempt_binding_problem(same, same, standup_path="s", endpoint_path="e") is None
    assert "carries no" in prun.attempt_binding_problem(same, {prun.STANDUP_ATTEMPT_KEY: A32}, standup_path="s", endpoint_path="e")   # an endpoint declaring nothing is torn
    assert "declares nothing" in prun.attempt_binding_problem({prun.STANDUP_ATTEMPT_KEY: A32}, same, standup_path="s", endpoint_path="e")
    assert prun.attempt_binding_problem({prun.STANDUP_ATTEMPT_KEY: A32}, {prun.STANDUP_ATTEMPT_KEY: B32}, standup_path="s", endpoint_path="e") is None  # two undeclared records: nothing to bind
    assert "torn" in prun.attempt_binding_problem(same, {prun.STANDUP_ATTEMPT_KEY: A32, prun.REQUESTED_HARNESS_KEY: ["none", LORA]}, standup_path="s", endpoint_path="e")


def test_a_u0_declaration_must_name_the_complete_baseline_exactly(tmp_path, monkeypatch):
    """A u0 / legacy audit's legs are the whole baseline (its frozen expected_cells advertises the
    manifest), so a PARTIAL declaration there refuses by name instead of silently narrowing it; the complete baseline -- every
    enumerated canonical leg, `none` included -- declared exactly freezes the same legs as an undeclared record, plus the declaration."""
    spec, source = _spec(), _source(tmp_path / "bank")
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=[])
    NP = prun.SERVING_PATH_LORA_NATIVE
    assert not prun.steered_wave_audit(spec)
    rec_p, table_p, _ = _real_standup_records(tmp_path / "P", native, source, spec, requested=[LORA], attempt=A32, monkeypatch=monkeypatch)
    with pytest.raises(prun.DriverError, match="u0 / legacy audit whose legs are the whole baseline.*declare the complete baseline exactly"):
        prun.cmd_print_cells(_args(tmp_path / "out_p", serving_path=NP, standup=rec_p, table=table_p), spec)
    complete = [c.cell_id for c in native.cell_set.canonical]
    assert complete == ["none", "prompted__safety", LORA]
    rec_c, table_c, _ = _real_standup_records(tmp_path / "C", native, source, spec, requested=complete, attempt=B32)
    assert prun.cmd_print_cells(_args(tmp_path / "out_c", serving_path=NP, standup=rec_c, table=table_c), spec) == 0
    rec_u, table_u, _ = _real_standup_records(tmp_path / "U", native, source, spec, requested=[], attempt=C32, write_endpoint=False)
    assert prun.cmd_print_cells(_args(tmp_path / "out_u", serving_path=NP, standup=rec_u, table=table_u), spec) == 0
    declared, undeclared = (json.loads((tmp_path / d / prun.CELLS_JSON).read_text()) for d in ("out_c", "out_u"))
    assert declared["canonical"] == undeclared["canonical"] and declared["controls"] == undeclared["controls"]
    assert [c["cell_id"] for c in declared["controls"]] == [dc.NOOP_LORA_CELL_ID] and declared["expected_cells"] == CELLS
    assert declared[prun.REQUESTED_HARNESS_KEY] == complete and prun.REQUESTED_HARNESS_KEY not in undeclared
    assert prun.complete_baseline_problem(native.cell_set, complete, spec, "p") is None
    assert "missing ['none', 'prompted__safety']" in prun.complete_baseline_problem(native.cell_set, [LORA], spec, "p")


def test_a_declaration_outside_the_universe_not_enumerated_or_not_served_refuses_by_name_and_a_steered_wave_audit_needs_one(tmp_path, monkeypatch):
    """Refusals, each over the REAL writer's record of a steered-wave audit (…_s<N>r): a declared id the harness grammar admits but
    the manifest does not list; one the enumeration does not carry (skipped by name under native); one the standup does not serve
    (requested ⊄ harness_cells); a steered-wave record declaring nothing (the writer's [] when nothing was named) -- the whole enumeration
    would audit none and the twin against the rule; and the declared cell alone freezes. The duplicate check is the pure reader's
    (the writer de-duplicates, so no record of its carries one)."""
    source = _source(tmp_path / "bank")
    spec_s = shape._spec("ab" * 8, audit=_SPEC_S2R, cells=list(CELLS))      # the module's _spec fixes `audit`; build the steered one directly
    native = prun.enumerate_petri_native(spec_s, source, behavior_files=_bf(spec_s), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=[])
    NP = prun.SERVING_PATH_LORA_NATIVE
    assert prun.steered_wave_audit(spec_s) and not prun.steered_wave_audit(_spec())
    foreign = "si_safety__vector_all"                                       # a harness id the grammar admits, outside this manifest's cells
    assert foreign not in CELLS and ra.harness_cell_id(_tt(foreign)) == foreign
    rec, table, _ = _real_standup_records(tmp_path / "o1", native, source, spec_s, requested=[LORA, foreign], attempt=A32, monkeypatch=monkeypatch)
    with pytest.raises(prun.DriverError, match="outside the manifest's universe"):
        prun.cmd_print_cells(_args(tmp_path / "out1", serving_path=NP, standup=rec, table=table), spec_s)
    rec, table, _ = _real_standup_records(tmp_path / "o2", native, source, spec_s, requested=[PSR], attempt=A32)   # skipped by name under native
    with pytest.raises(prun.DriverError, match="does not carry"):
        prun.cmd_print_cells(_args(tmp_path / "out2", serving_path=NP, standup=rec, table=table), spec_s)
    rec, table, _ = _real_standup_records(tmp_path / "o3", native, source, spec_s, requested=[LORA], attempt=A32, served=["none", "prompted__safety"])
    with pytest.raises(prun.DriverError, match="does not serve"):
        prun.cmd_print_cells(_args(tmp_path / "out3", serving_path=NP, standup=rec, table=table), spec_s)
    with pytest.raises(prun.DriverError, match="names a cell twice"):
        prun.requested_cells_of({prun.REQUESTED_HARNESS_KEY: [LORA, LORA]}, "p")
    rec, table, _ = _real_standup_records(tmp_path / "o4", native, source, spec_s, requested=[], attempt=A32)
    assert json.loads(rec.read_text())[ns.REQUESTED_EXPLICIT_HARNESS_KEY] == []
    with pytest.raises(prun.DriverError, match="steered-wave view and drives only the cells the proxy declared"):
        prun.cmd_print_cells(_args(tmp_path / "out4", serving_path=NP, standup=rec, table=table), spec_s)
    rec, table, _ = _real_standup_records(tmp_path / "o5", native, source, spec_s, requested=[LORA], attempt=A32)
    assert prun.cmd_print_cells(_args(tmp_path / "out5", serving_path=NP, standup=rec, table=table), spec_s) == 0
    assert [c["cell_id"] for c in json.loads((tmp_path / "out5" / prun.CELLS_JSON).read_text())["canonical"]] == [LORA]


def test_torn_declarations_refuse_by_name_and_freeze_nothing(tmp_path, monkeypatch):
    """The reader never depends on the producer being the only writer. (1) served-declares /
    endpoint-silent: a served record declaring [LORA] paired -- through the sibling or --served-endpoint -- with a same-attempt endpoint
    that declares nothing ([] from the real writer when nothing was named; and the pre-declaration shape with the key absent) refuses
    by name, naming both files, and nothing is frozen. (2) endpoint-declares / served-silent: an undeclared served record (the u0
    path) beside an endpoint that declares [LORA] refuses the disagreement by name instead of freezing the whole enumeration; nothing
    is frozen. Every record is the real writer's; the one legacy shape is the writer's endpoint minus the key it did not exist to write."""
    source = _source(tmp_path / "bank")
    spec_s = shape._spec("ab" * 8, audit=_SPEC_S2R, cells=list(CELLS))
    native_s = prun.enumerate_petri_native(spec_s, source, behavior_files=_bf(spec_s), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=[])
    NP = prun.SERVING_PATH_LORA_NATIVE
    # (1) the served record declares [LORA]; the endpoint beside it is the same attempt's but declares nothing
    rec, table, ep = _real_standup_records(tmp_path / "D", native_s, source, spec_s, requested=[LORA], attempt=A32, monkeypatch=monkeypatch)
    _, _, ep_silent = _real_standup_records(tmp_path / "S", native_s, source, spec_s, requested=[], attempt=A32)
    assert json.loads(ep_silent.read_text())[ns.REQUESTED_EXPLICIT_HARNESS_KEY] == []
    shutil.copy(ep_silent, ep)
    out1 = tmp_path / "out1"
    with pytest.raises(prun.DriverError, match=r"D/cells_served.json declares \['si_safety__lora_r4'\] but .*D/endpoint.json declares \[\] -- torn"):
        prun.cmd_print_cells(_args(out1, serving_path=NP, standup=rec, table=table), spec_s)
    assert not (out1 / prun.CELLS_JSON).exists()
    # the same through --served-endpoint, and the pre-declaration endpoint shape (the key absent)
    legacy = json.loads(ep_silent.read_text()); del legacy[ns.REQUESTED_EXPLICIT_HARNESS_KEY]; del legacy[ns.REQUESTED_EXPLICIT_KEY]
    (tmp_path / "legacy_endpoint.json").write_text(json.dumps(legacy))
    with pytest.raises(prun.DriverError, match=r"D/cells_served.json declares .* but .*legacy_endpoint.json carries no cells_requested_explicit_harness"):
        prun.cmd_print_cells(_args(out1, serving_path=NP, standup=rec, table=table, endpoint=tmp_path / "legacy_endpoint.json"), spec_s)
    assert not (out1 / prun.CELLS_JSON).exists()
    # (2) an undeclared served record (the u0 path) beside an endpoint that declares: the disagreement refuses, the u0 path does not run
    spec0 = _spec()
    native0 = prun.enumerate_petri_native(spec0, source, behavior_files=_bf(spec0), prompt_files=shape.PROMPT_FILES)
    rec_u, table_u, ep_u = _real_standup_records(tmp_path / "U", native0, source, spec0, requested=[], attempt=B32)
    out2 = tmp_path / "out2"
    assert prun.cmd_print_cells(_args(out2, serving_path=NP, standup=rec_u, table=table_u), spec0) == 0     # agreeing (both undeclared): frozen
    frozen = json.loads((out2 / prun.CELLS_JSON).read_text())
    assert len(frozen["canonical"]) == 3 and prun.REQUESTED_HARNESS_KEY not in frozen
    _, _, ep_decl = _real_standup_records(tmp_path / "V", native0, source, spec0, requested=["none", "prompted__safety", LORA], attempt=B32)
    shutil.copy(ep_decl, ep_u)                                                                # the endpoint now declares; the served record does not
    out3 = tmp_path / "out3"
    with pytest.raises(prun.DriverError, match=r"U/endpoint.json declares \['none', 'prompted__safety', 'si_safety__lora_r4'\] but .*U/cells_served.json declares nothing"):
        prun.cmd_print_cells(_args(out3, serving_path=NP, standup=rec_u, table=table_u), spec0)
    assert not (out3 / prun.CELLS_JSON).exists()                                               # nothing frozen under the torn pair
    with pytest.raises(prun.DriverError, match="declares nothing"):                              # and no leg runs off the earlier freeze
        prun.cmd_run(_args(out2, serving_path=NP, standup=rec_u, table=table_u, cell_index=0), spec0)
    assert json.loads((out2 / prun.CELLS_JSON).read_text()) == frozen                          # the earlier record stands, untouched


def test_a_same_set_reordered_endpoint_declaration_binds_and_the_served_records_order_is_frozen(tmp_path, monkeypatch):
    """The declaration contract is SET equality. Two real writers of the SAME attempt declaring the same
    harness cells in different orders -- the served record [none, LORA], the endpoint [LORA, none] (an endpoint re-serialised or
    backfilled in another order) -- bind, and the frozen record keeps the served record's order; a genuinely different set still refuses."""
    source = _source(tmp_path / "bank")
    spec_s = shape._spec("ab" * 8, audit=_SPEC_S2R, cells=list(CELLS))
    native_s = prun.enumerate_petri_native(spec_s, source, behavior_files=_bf(spec_s), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=[])
    NP = prun.SERVING_PATH_LORA_NATIVE
    rec, table, ep = _real_standup_records(tmp_path / "X", native_s, source, spec_s, requested=["none", LORA], attempt=A32, monkeypatch=monkeypatch)
    _, _, ep_reordered = _real_standup_records(tmp_path / "Y", native_s, source, spec_s, requested=[LORA, "none"], attempt=A32)
    assert json.loads(ep_reordered.read_text())[ns.REQUESTED_EXPLICIT_HARNESS_KEY] == [LORA, "none"]
    shutil.copy(ep_reordered, ep)                                                              # the same attempt, the same set, another order
    assert prun.attempt_binding_problem(json.loads(rec.read_text()), json.loads(ep.read_text()), standup_path="s", endpoint_path="e") is None
    out = tmp_path / "out"
    assert prun.cmd_print_cells(_args(out, serving_path=NP, standup=rec, table=table), spec_s) == 0
    record = json.loads((out / prun.CELLS_JSON).read_text())
    assert record[prun.REQUESTED_HARNESS_KEY] == ["none", LORA]                                 # the served record's order, not the endpoint's
    assert [c["cell_id"] for c in record["canonical"]] == ["none", LORA] and [c["cell_id"] for c in record["controls"]] == [dc.NOOP_LORA_CELL_ID]
    assert prun.cmd_print_cells(_args(out, serving_path=NP, standup=rec, table=table), spec_s) == 0   # immutable: a no-op
    # a different SET of the same size still refuses (the negatives are unchanged)
    _, _, ep_other = _real_standup_records(tmp_path / "Z", native_s, source, spec_s, requested=["none", "prompted__safety"], attempt=A32)
    shutil.copy(ep_other, ep)
    with pytest.raises(prun.DriverError, match="torn records"):
        prun.cmd_print_cells(_args(tmp_path / "out2", serving_path=NP, standup=rec, table=table), spec_s)
    assert not (tmp_path / "out2" / prun.CELLS_JSON).exists()
