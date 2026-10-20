"""A SYNTHETIC shared-standup mirror (the tinycap-shared-standup set) DERIVED from the synthetic full-suite mirror of `_bcbw_full_fixtures`, the way
`scripts/build_shared_fixture.py` derives the real fixture from the canonical solo stores: every `-shared` store is a byte copy of a solo store
rewritten into the row-engine shape (the suffix, `manifest.usage_ledger` / `route.bcb_sandbox.owner` on the new path and the row's FunctionCall, the
`row_engine` block, the served-digest map, the row endpoint on `served_geometry.json`, RESULT's `standup=` token, the three `standup/` files), with the
`-shared` records beside it: `launch_sets_shared.json` (every row + `view` the `_shared` spelling, `selection`, `comparison_kind`, `solo_run_id` minted by the
SOLO reader, `solo_run_ids`, `solo_store`, `standup_id`, `launched_as`), the gate records under the `-shared` suffix (+ `standup_id`, `engine_cells`, `row_engine`,
`row_dir`, `verification_pass`, `comparison`, `compared_against.comparability_class`, the recorded `max_loras` / `residual_slot_armed`), the three-number record
`three_numbers_<eval>_full_shared.json`, `relaunches.json` for a `-shared-r2` relaunch, `rows_launch.tsv`, `bcbw_schedule_shared.json`, and the SOLO root's roster
set, pins, item-set index and pin files copied verbatim. The solo root stays beside it: the shared reader reads the comparison's other side from it.

The gemma4e4b family's `s1` row (`none`, the constitution ReLU key, the prompted key) is the row; knobs, one hazard each (`REQUIRED_RECORDS.md` section 6):
`repeat` (the `-shared-rep2` none, hazard 2), `it_none` (the instruct row's OWN none as a store under `gemma4e4b-f22it`, its lane_roster citing a roster of
ANOTHER tree matched by hash -- hazards 4 / 5), `relaunched` (the ReLU key relaunched `-shared-r2` on its own standup with the parked `-shared` first attempt mirrored
COMPLETE -- hazards 6 / 7), `spelling_override` (a row's `view` spelled the solo way, hazard 3), `solo_run_id_override` (a solo id that is not the solo
reader's, hazard 1), `repeat_row_without_segment` (the repeat row's launched_as.suffix without `-rep2`: two rows on one key), `standup_override` (RESULT's token
another id), `row_engine_keys_override`, `no_solo_stores` (the comparison's other side missing: the runs file, the contrasts wait), `gate_class_override`.

The store-attestation decision of 2026-09-29 (the first REAL mirror, landing 1; `_bcbw_common.SHARED_ROW_DIR_RULE` / `SHARED_ENDPOINT_TUPLE_RULE` /
`SHARED_IT_REPEAT_RULE`) adds the real mirror's shapes as knobs: `row_dir_absolute` (every `row_engine.row_dir` and gate `row_dir` spelled volume-absolute,
`/vol/tinycap/gen/rows/<row>/<suffix>`), `row_dir_override` (the none store's spelling verbatim, for the tail refusal), `manifest_map_null` (every manifest's
`route.object_artifact_sha256_by_cell` null and the gate facts' copy null; the endpoint keeps the row's map), `endpoint_last_key` (landing 1's row runner: every
store's `endpoint.cell` / `harness_cell` name the row's LAST served key), `endpoint_edit` (a per-store edit of the endpoint, for the cross-check refusals) and
`it_repeat` (the instruct row's own none served a second time, `-shared-rep2` under `gemma4e4b-f22it`: class repeat, repeat_floor, no solo run, no solo store)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from typing import Callable, Dict, List, Optional

import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bfcl_full_shared as BFS
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import cpu_verify as CV
from sisteer.registry.adapters import bfcl_full_shared_l2 as BFS2
from sisteer.registry.adapters import bfcl_full_shared_l3 as BFS3
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import bigcodebench_full_shared as BCS
from sisteer.registry.adapters import bigcodebench_full_shared_l2 as BCS2
from sisteer.registry.adapters import bigcodebench_full_shared_l3 as BCS3
from sisteer.registry.adapters import lineage_common as LC

MK, MODEL_ID, TAG, SEL, ROW_DIR = Z.MK, Z.MODEL_ID, Z.TAG, Z.SEL, Z.ROW_DIR
NONE, RELU, PROMPTED = Z.NONE, Z.RELU, Z.PROMPTED
IT_TAG, IT_SEL, IT_ROW_DIR = Z.IT_TAG, Z.IT_SEL, Z.IT_ROW_DIR
TREE, TREE2, NOW = Z.TREE, Z.TREE2, Z.NOW
SOLO = {"bigcodebench": BCF, "bfcl": BFF}
SHARED = {"bigcodebench": BCS, "bfcl": BFS}
ADAPTERS = {"bigcodebench": BCS.BigcodebenchFullSharedAdapter, "bfcl": BFS.BfclFullSharedAdapter}
#: landing 2: the same rows under the `_l2` sources -- the `mirror(landing=2)` knob builds the root `<eval>_full_shared_l2` and the launch set spelling the BASE
#: source under evals.<eval>.own_source, the way the bound set does
SHARED_L2 = {"bigcodebench": BCS2, "bfcl": BFS2}
ADAPTERS_L2 = {"bigcodebench": BCS2.BigcodebenchFullSharedL2Adapter, "bfcl": BFS2.BfclFullSharedL2Adapter}
#: landing 3: the gemma row's tuples under the `_l3` sources -- `mirror(landing=3, **LANDING3_SHAPE)` builds the root `<eval>_full_shared_l3`
#: in landing 3's three shapes (the fourteen-key verified block through the REAL emitter, the re-stamp carrying standup= before verified_by=, the legacy route scalar
#: spelling the row's fitted digest on EVERY store); the launch set spells the BASE source under own_source as landing 2's does
SHARED_L3 = {"bigcodebench": BCS3, "bfcl": BFS3}
ADAPTERS_L3 = {"bigcodebench": BCS3.BigcodebenchFullSharedL3Adapter, "bfcl": BFS3.BfclFullSharedL3Adapter}
SHARED_BY_LANDING = {1: SHARED, 2: SHARED_L2, 3: SHARED_L3}
ADAPTERS_BY_LANDING = {1: ADAPTERS, 2: ADAPTERS_L2, 3: ADAPTERS_L3}
#: the real landing-3 stores' shapes, as `mirror(...)` knobs (as published 2026-10-01; the three differences from landing 2's 19 stores)
LANDING3_SHAPE = {"verified": True, "verified_fields": "current", "stamp_standup": True, "route_scalar": True}
#: landing 2's CPU verification pass (the `verified` knob): the verifier's Modal function call, the re-stamp's timestamp
VERIFIER_FC = "fc-01SYNTHETICVERIFIER000000000"
VERIFIED_AT = "2026-09-30T07:36:27Z"
SYNTHETIC_IMAGE = "ghcr.io/synthetic/sandbox@sha256:" + "0" * 64
VERIFIED_FIELD_SETS = ("v1", "current")


def verification_block(fc: str = VERIFIER_FC, state: str = B.VERIFICATION_STATE_VERIFIED, fields: str = "v1") -> dict:
    """The `manifest.row_engine.verification_pass` block of a verified store (the gate record carries the same block). ``fields="v1"``: landing 2's NINE-key set
    (`contracts.VERIFICATION_PASS_VERIFIED_FIELDS_V1`, the real landing-2 mirror's keys, hand-written and pinned to the contract); ``fields="current"``: landing 3's
    FOURTEEN-key set built by the REAL emitter `cpu_verify.verified_block` (`contracts.VERIFICATION_PASS_VERIFIED_FIELDS`: the V1 nine + verifier_tree_sha,
    generation_tree_sha, executed_image, executed_verify_timeout_s, binding_version), so the fixture proves the reader against the emitter's shape and not a copy of it.
    ``state`` other than verified overrides the emitter's (the `pending` hazard)."""
    facts = {"app_name": "inspect_modal_sandbox", "block_network": True, "cpu": 1.0, "image": SYNTHETIC_IMAGE, "memory_mib": 512, "provider": "modal_owned", "timeout_s": 10800,
             "verify_timeout_s": 30}
    if fields == "v1":
        doc = {"state": state, "function_call_id": fc, "mechanism": "M2_replay_vehicle", "n_items": Z.N_FULL, "max_sandboxes": 8, "started_at": 1790752345.8, "finished_at": 1790753786.0,
               "sandbox_facts": dict(facts), "vehicle_log_sha256": hashlib.sha256(b"synthetic vehicle log").hexdigest()}
        assert set(doc) == set(C.VERIFICATION_PASS_VERIFIED_FIELDS_V1), sorted(doc)
        return doc
    assert fields == "current", fields
    # the real landing-3 block's sandbox_facts carry binding_version too; the emitter asserts its own key order against the contract
    doc = CV.verified_block(mechanism="M2_replay_vehicle", function_call_id=fc, started_at=1790818321.1655607, finished_at=1790819079.243149, sandbox_facts=dict(facts, binding_version="1"),
                            max_sandboxes=8, n_items=Z.N_FULL, vehicle_log_sha256=hashlib.sha256(b"synthetic vehicle log").hexdigest(), verifier_tree_sha=TREE, generation_tree_sha=TREE,
                            executed_image=SYNTHETIC_IMAGE, executed_verify_timeout_s=30, binding_version="1")
    assert tuple(doc) == C.VERIFICATION_PASS_VERIFIED_FIELDS and len(doc) == 14, sorted(doc)
    doc["state"] = state
    return doc


def stamp_line(row_dir_out: str, shared_suffix: str, fc: str = VERIFIER_FC, standup: Optional[str] = None) -> str:
    """The CPU verifier's re-stamp, RESULT's LAST line on a verified store: COMPLETE at the tree, `verified_by=` the verifier's call. Landing 2's shape carries NO
    standup= token (``standup`` None); the later shape -- every landing-3 store -- carries ``standup=<manifest id>`` BEFORE ``verified_by=``."""
    tok = f" standup={standup}" if standup else ""
    return f"{B.RESULT_HEAD} name=row-{row_dir_out}-{shared_suffix.split('-' + B.SHARED_REPEAT_MARKER)[0]} rc=0 result=COMPLETE status=OK sha={TREE} at={VERIFIED_AT}{tok} verified_by={fc}"
#: the row's object keys in the harness's order and their harness ids; the standup ids (32 hex) of the row launch, the repeat (the same), the relaunch (its own)
ROW_KEYS = [NONE, RELU, PROMPTED]
STANDUP_A = hashlib.md5(b"synthetic shared standup A").hexdigest()
STANDUP_B = hashlib.md5(b"synthetic shared standup B").hexdigest()
STANDUP_IT = hashlib.md5(b"synthetic shared standup it").hexdigest()
ROW_FC = "fc-01SYNTHETICSHAREDROW000000A"
ROW_FC_R2 = "fc-01SYNTHETICSHAREDROW00000R2"
ROW_FC_IT = "fc-01SYNTHETICSHAREDROW0000IT0"
SPELLING = B.SharedSpec(solo_source="x", solo_root_name="x").spelling_of


def _read(p: str):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _shared_suffix(solo_suffix: str, extra: str = "") -> str:
    return f"{solo_suffix}-{B.SHARED_STORE_MARKER}{extra}"


def _row_record_dir(row_dir_out: str, shared_suffix: str, absolute: bool = False) -> str:
    """The row-record dir a store's manifest and gate spell: ``rows/<row>/<row launch suffix>`` (the suffix without its `-rep<N>` segment), volume-absolute when asked."""
    rel = f"rows/{row_dir_out}/{shared_suffix.split('-' + B.SHARED_REPEAT_MARKER)[0]}"
    return f"{B.VOLUME_HEAD}{rel}" if absolute else rel


def _copy_store(solo_root: str, shared_root: str, row_dir: str, okey: str, solo_suffix: str, shared_suffix: str, *, standup_id: str, fc: str, served: List[str],
                digests: Dict[str, str], max_loras: int, residual_armed: bool, new_row_dir: Optional[str] = None, new_wave: Optional[str] = None, owner_key: Optional[str] = None,
                result_standup: Optional[str] = None, row_engine_edit: Optional[Callable[[dict], None]] = None, endpoint_standup: Optional[str] = None,
                row_dir_absolute: bool = False, row_dir_override: Optional[str] = None, manifest_map_null: bool = False, endpoint_key: Optional[str] = None,
                endpoint_edit: Optional[Callable[[str, dict], None]] = None, verified: Optional[dict] = None, stamp_fc: Optional[str] = None,
                result_no_standup_line: bool = False, stamp_standup: bool = False, route_scalar: Optional[str] = None) -> str:
    """Copy one solo store into the shared root under ``shared_suffix`` (and ``new_row_dir`` for the instruct row's own none), rewriting what the row engine
    rewrites; returns the store key. ``row_dir_absolute`` / ``row_dir_override`` spell ``row_engine.row_dir``; ``manifest_map_null`` leaves the manifest's route
    map null (the endpoint keeps the row's map); ``endpoint_key`` is what the endpoint's `cell` / `harness_cell` name (this store's key by default); ``endpoint_edit``
    is a last edit of the endpoint ``(okey, endpoint) -> None``. ``verified`` (landing 2) is the ``verification_pass`` block: the manifest carries it and RESULT takes
    landing 2's shape -- the row runner's ``result=GENERATED`` line carrying ``standup=`` (``result_standup`` overrides that token), then the verifier's re-stamp
    (:func:`stamp_line`; ``stamp_fc`` overrides its ``verified_by``) as the LAST line without ``standup=``; ``result_no_standup_line`` drops the runner's line so no
    RESULT line carries the token at all. ``stamp_standup`` (landing 3's later shape) gives RESULT the real landing-3 stores' THREE lines -- the harness's
    first ``phase=generate`` line with no token, the row runner's GENERATED line with it, the re-stamp carrying ``standup=<manifest id>`` before ``verified_by=``.
    ``route_scalar`` (landing 3) spells ``manifest.route.object_artifact_sha256`` and the endpoint's scalar on THIS store (the real landing-3 harness wrote the row's one
    fitted digest into every store's scalar, the unsteered and prompted ones included); None leaves both null, the landing-1 / landing-2 shape."""
    src = os.path.join(solo_root, B.STORES_HEAD, row_dir, okey, B.THINKING_DIR, solo_suffix)
    row_dir_out = new_row_dir or row_dir
    dst = os.path.join(shared_root, B.STORES_HEAD, row_dir_out, okey, B.THINKING_DIR, shared_suffix)
    shutil.copytree(src, dst)
    store_key = next(k for k in os.listdir(os.path.join(dst, "cells")))
    kd = os.path.join(dst, "cells", store_key)
    man = _read(os.path.join(kd, "manifest.json"))
    rel = B.rel_store(row_dir_out, okey, shared_suffix)
    man["usage_ledger"] = f"{B.VOLUME_HEAD}{rel}/usage_ledger.jsonl"
    man["route"]["bcb_sandbox"]["owner"] = {"cell": (owner_key or okey), "function_call_id": fc, "run_id": f"{TREE[:7]}-{standup_id[:8]}"}
    man["route"]["bcb_sandbox"]["sandbox_record"] = f"{B.VOLUME_HEAD}{rel}/watch/sandboxes_{fc}.jsonl"
    man["route"]["object_artifact_sha256"] = route_scalar
    man["route"]["object_artifact_sha256_by_cell"] = None if manifest_map_null else dict(digests)
    man["row_engine"] = {"row_engine": True, "standup_id": standup_id, "engine_cells": list(served), "max_loras": max_loras, "residual_slot_armed": residual_armed,
                         "compile_cache": {"hit": False, "key": f"synthetic:{MK}:vllm-pin:L16384:S31:r8:n{max_loras}", "vllm_cache_dir": None},
                         "row_dir": (row_dir_override if row_dir_override is not None else _row_record_dir(row_dir_out, shared_suffix, row_dir_absolute)),
                         "verification_pass": (copy.deepcopy(verified) if verified is not None else None)}
    if row_engine_edit is not None:
        row_engine_edit(man["row_engine"])
    X.write(os.path.join(kd, "manifest.json"), man)
    geo = _read(os.path.join(dst, B.GEOMETRY_FILE))
    ep = geo["endpoint"]
    ep_key = endpoint_key or okey
    ep.update({"standup_attempt_id": (endpoint_standup or standup_id), "cell": ep_key, "harness_cell": X.harness_id(ep_key), "cells": list(served), "harness_cells": [X.harness_id(k) for k in served],
               "zero_adapter_only": False, "object_artifact_sha256": route_scalar, "object_artifact_sha256_by_cell": dict(digests), "residual_slot": {"armed": residual_armed, "cells": {}}})
    ep["lora_config"] = dict(ep.get("lora_config") or {}, max_loras=max_loras)
    if endpoint_edit is not None:
        endpoint_edit(okey, ep)
    geo["standup_argv"] = ["python", "capctrl_serving_up.py"] + [x for k in served for x in ("--cell", k)] + ["--tool-calls"]
    X.write(os.path.join(dst, B.GEOMETRY_FILE), geo)
    with open(os.path.join(dst, "RESULT"), "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if verified is None:
        lines[-1] = f"{lines[-1]} standup={result_standup or standup_id}"
    else:
        # landing 2's RESULT: the row runner's GENERATED line carries standup=; the CPU verifier's re-stamp is the LAST line, verified_by= and no standup= token.
        # landing 3's (stamp_standup): the harness's first `phase=generate` line without a token, the runner's line, the re-stamp WITH standup=<manifest id> before verified_by=
        runner = lines[-1].replace("result=COMPLETE status=OK", "result=GENERATED status=GENERATED", 1) + f" standup={result_standup or standup_id}"
        first = ([f"{B.RESULT_HEAD} name={okey} rc=0 result=GENERATED status=GENERATED sha={TREE} at={B.parse_result_line(lines[-1]).get('at')} phase=generate tasks=inspect/synthetic_full"]
                 if stamp_standup else [])
        lines = first + ([] if result_no_standup_line else [runner]) + [stamp_line(row_dir_out, shared_suffix, stamp_fc or str(verified.get("function_call_id") or VERIFIER_FC),
                                                                                   standup=((result_standup or standup_id) if stamp_standup else None))]
    X.write(os.path.join(dst, "RESULT"), "\n".join(lines) + "\n")
    if new_wave is not None:
        lr = _read(os.path.join(dst, B.LAUNCHED_ENTRY_FILE))
        lr["wave"] = new_wave
        X.write(os.path.join(dst, B.LAUNCHED_ENTRY_FILE), lr)
    X.write(os.path.join(dst, "standup", "endpoint.json"), dict(ep))
    X.write(os.path.join(dst, "standup", "serving_attestation.json"), {"synthetic_fixture": True, "standup_attempt_id": standup_id, "adapters": {}})
    X.write(os.path.join(dst, "standup", "vllm_server.log"), "# SYNTHETIC FIXTURE\n")
    os.makedirs(os.path.join(dst, "watch"), exist_ok=True)
    X.write(os.path.join(dst, "watch", f"sandboxes_{fc}.jsonl"), json.dumps({"owner": {"cell": (owner_key or okey), "function_call_id": fc}}) + "\n")
    return store_key


def _gate(solo_gate: dict, *, shared_suffix: str, standup_id: str, served: List[str], max_loras: int, residual_armed: bool, comparison: dict, compared_against: Optional[dict],
          row_dir_rec: str, digests: Dict[str, str], note: Optional[str] = None, row: Optional[str] = None, wave: Optional[str] = None, store_key: Optional[str] = None,
          facts_map_null: bool = False, verification_pass: Optional[dict] = None) -> dict:
    g = copy.deepcopy(solo_gate)
    old = g["suffix"]
    g["suffix"] = shared_suffix
    if row is not None:
        g["row"] = row
    if wave is not None:
        g["wave"] = wave
    g["attested_from"] = [a.replace(old, shared_suffix, 1) for a in g["attested_from"]]
    g["done_markers"] = [d.replace(old, shared_suffix, 1) for d in g["done_markers"]]
    if store_key is not None:
        g["attested_from"] = [a.replace(a.split("/cells/")[1].split("/")[0], store_key) if "/cells/" in a else a for a in g["attested_from"]]
        g["done_markers"] = [d.replace(d.split("/cells/")[1].split("/")[0], store_key) for d in g["done_markers"]]
    g["facts"][0]["manifest"] = g["facts"][0]["manifest"].replace(old, shared_suffix, 1)
    g["facts"][0]["route.object_artifact_sha256_by_cell"] = None if facts_map_null else dict(digests)
    g["facts"][0]["recorded_flags"].update(max_loras=max_loras, residual_slot_armed=residual_armed, zero_adapter_only=False)
    g.update(standup_id=standup_id, engine_cells=list(served), row_engine=True, row_dir=row_dir_rec, verification_pass=copy.deepcopy(verification_pass), comparison=comparison)
    if compared_against is not None:
        g["compared_against"] = compared_against
    elif note is not None:
        g["comparability_note"] = note
    g.pop("cross_tree", None)
    return g


def _compared(this: str, prior: Optional[str], record: str) -> dict:
    return {"record": record, "prior_verdict": "PASS", "engines_equal": True, "normalised_flags_equal": True,
            "comparability_class": ("same_standup" if prior == this else "cross_standup"), "standup_ids": {"this": this, "prior": prior}, "require_class": ("same_standup" if prior == this else "cross_standup")}


def mirror(tmp_path, seeds: LC.Seeds, *, eval_id: str, repeat: bool = True, it_none: bool = False, relaunched: bool = False, spelling_override: Optional[str] = None,
           solo_run_id_override: Optional[str] = None, repeat_row_without_segment: bool = False, standup_override: Optional[str] = None,
           row_engine_keys_override: Optional[Callable[[dict], None]] = None, no_solo_stores: bool = False, gate_class_override: Optional[str] = None,
           launch_override: Optional[Callable[[dict], None]] = None, three_override: Optional[Callable[[dict], None]] = None, solo_kw: Optional[dict] = None,
           row_dir_absolute: bool = False, row_dir_override: Optional[str] = None, manifest_map_null: bool = False, endpoint_last_key: bool = False,
           endpoint_edit: Optional[Callable[[str, dict], None]] = None, it_repeat: bool = False, landing: int = 1, verified: bool = False,
           verified_state_override: Optional[str] = None, verified_by_override: Optional[str] = None, verified_result_no_standup: bool = False,
           own_source_override: Optional[str] = None, verified_fields: str = "v1", stamp_standup: bool = False, route_scalar: bool = False) -> dict:
    """The solo full mirror (`_bcbw_full_fixtures.mirror`, the gemma s1 / s1p rows) under `<tmp>/solo/` and the SHARED mirror derived from its s1 row under
    `<tmp>/shared/<eval>_full_shared/`; returns the fixture dict (`root` the shared root, `solo_root`, the solo fixture's `roster` / `pins` paths, the store keys,
    the solo run ids, the standup ids). ``landing=2`` / ``landing=3`` builds the `_l2` / `_l3` source's root (`<tmp>/shared/<eval>_full_shared_l<N>/`) with the
    three-number record's `source` that landing's name and the launch set's evals.<eval>.own_source spelling the BASE source (the bound sets do;
    ``own_source_override`` spells another); ``verified`` gives every store the verification shape (:func:`verification_block` with ``verified_fields`` -- "v1" the
    nine-key landing-2 block, "current" the fourteen-key landing-3 block through the real emitter -- RESULT's re-stamp last line; ``verified_state_override`` another
    state, ``verified_by_override`` another verifier call on the none store's re-stamp, ``verified_result_no_standup`` drops the none store's row-runner line);
    ``stamp_standup`` gives the re-stamp the later shape's ``standup=`` token before ``verified_by=`` (landing 3; the three-number rows' result_line follows); ``route_scalar``
    spells the row's fitted digest into EVERY store's legacy route scalar (landing 3). ``LANDING3_SHAPE`` is the real landing-3 stores' knob set."""
    Z.bind_seeds(seeds)
    it_none = it_none or it_repeat        # the instruct row's repeat pairs against the row's own none: the none store comes with it
    mods = SHARED_BY_LANDING[landing]
    vblock = verification_block(state=(verified_state_override or B.VERIFICATION_STATE_VERIFIED), fields=verified_fields) if verified else None
    # the real mirror's shapes (the decision of 2026-09-29; landing 2's verification pass; landing 3's token-present re-stamp and non-null route scalar), applied to EVERY
    # store copied below -- the scalar is the row's ONE fitted digest (the ReLU key's object), on the none and prompted stores too, as the real landing-3 harness wrote it
    shape_kw = dict(row_dir_absolute=row_dir_absolute, manifest_map_null=manifest_map_null, endpoint_edit=endpoint_edit, verified=vblock, stamp_standup=stamp_standup,
                    route_scalar=(Z.OBJ if route_scalar else None))
    solo_dir = tmp_path / "solo"
    solo_dir.mkdir(parents=True, exist_ok=True)
    fx = Z.mirror(solo_dir, seeds, eval_id=eval_id, root=str(solo_dir / f"{eval_id}_full"), **(solo_kw or {}))
    solo_root = fx["root"]
    root = str(tmp_path / "shared" / f"{eval_id}_full_shared{f'_l{landing}' if landing > 1 else ''}")
    recs = os.path.join(root, B.RECORDS_DIR)
    os.makedirs(recs)
    solo_recs = os.path.join(solo_root, B.RECORDS_DIR)
    for name in os.listdir(solo_recs):
        if name.startswith("gate_") or name.startswith("launch_sets") or name.startswith("three_numbers") or name in (B.ROWS_LAUNCH_FILE, B.SCHEDULE_FILE, B.RELAUNCHES_FILE):
            continue
        shutil.copy2(os.path.join(solo_recs, name), os.path.join(recs, name))     # the roster set, the pins, item_sets.json and the pin files: byte copies
    s2 = Z.seeds_for(seeds)
    inst = {ev: str(s2.instrument_row(SOLO[ev].INSTRUMENT_SEED, ev, model_id=MODEL_ID)["instrument_id"]) for ev in Z.TASKS_FULL}
    solo_ls = _read(os.path.join(solo_recs, "launch_sets.json"))
    solo_three = {ev: _read(os.path.join(solo_recs, Z.THREE_NUMBERS_FILES[ev])) for ev in Z.TASKS_FULL}
    solo_suffix = X.suffix_of(TAG, NONE, tree=TREE)
    digests = {X.harness_id(RELU): Z.OBJ}
    served = list(ROW_KEYS)
    max_loras = 2
    launch: Dict[str, List[dict]] = {ev: [] for ev in Z.TASKS_FULL}
    three = {ev: dict({k: v for k, v in solo_three[ev].items() if k != "rows"}, source=mods[ev].SOURCE, rows={}) for ev in Z.TASKS_FULL}
    keys: Dict[str, str] = {}
    solo_ids: Dict[str, Dict[str, str]] = {}
    standups: Dict[str, str] = {}
    gate_files: Dict[str, str] = {}
    none_gate_rec = f"records/gate_{TAG}_{NONE}_{_shared_suffix(solo_suffix)}.json"

    def solo_store_block(okey: str) -> dict:
        key = fx["keys"][okey]
        return {"source": {ev: SOLO[ev].SOURCE for ev in Z.TASKS_FULL}, "row_dir": ROW_DIR, "cell": okey, "suffix": solo_suffix, "sha": TREE,
                "mirror_rel": B.rel_store(ROW_DIR, okey, solo_suffix), "store_key": key, "producer_key": f"{TAG}/{key}", "tinycap_run_id": f"{TREE[:7]}-synthetic"}

    def solo_rows(okey: str) -> Dict[str, dict]:
        return {ev: next(r for r in solo_ls["evals"][ev]["launch"] if r["cell"] == okey and r["view"] == SEL) for ev in Z.TASKS_FULL}

    def add_row(okey: str, *, shared_suffix: str, kind: str, standup_id: str, cls: Optional[str] = None, tuple_suffix: str = "", extra_row: Optional[dict] = None,
                spelling: Optional[str] = None, tag: str = TAG, row_dir: str = ROW_DIR, selection: str = SEL, launch_tag: Optional[str] = None, solo_block: Optional[dict] = None,
                solo_run_ids: Optional[Dict[str, str]] = None, three_key: Optional[str] = None, store_key: Optional[str] = None, stamp_fc: Optional[str] = None) -> None:
        for ev in Z.TASKS_FULL:
            base_row = copy.deepcopy(solo_rows(okey)[ev])
            row = dict(base_row)
            row.update({"view": (spelling or SPELLING(tag)), "selection": selection, "comparison_kind": kind, "standup_id": standup_id, "tag": tag,
                        "bcbw_wave": (launch_tag or B.launch_tag_of(tag, okey)), "tuple_id": f"{selection}/{okey}{tuple_suffix}",
                        "launched_as": {"wave": (launch_tag or B.launch_tag_of(tag, okey)), "row_dir": row_dir, "suffix": shared_suffix, "sha": TREE}, "sha": TREE,
                        "solo_run_id": (solo_run_ids[ev] if solo_run_ids else None), "solo_run_ids": (dict(solo_run_ids) if solo_run_ids else None), "solo_store": solo_block})
            if cls is not None:
                row["class"] = cls
            if extra_row:
                row.update(copy.deepcopy(extra_row))
            launch[ev].append(row)
            tr = copy.deepcopy(solo_three[ev]["rows"][f"{TAG}/{okey}"])
            tr.update({"key": (three_key or f"{tag}/{okey}{tuple_suffix}"), "tag": tag, "view": (spelling or SPELLING(tag)), "store": B.rel_store(row_dir, okey, shared_suffix),
                       # the RESULT last line the store carries: the row runner's with standup= (landing 1), or the verifier's re-stamp (landing 2; with the token on landing 3)
                       "result_line": (stamp_line(row_dir, shared_suffix, stamp_fc or VERIFIER_FC, standup=(standup_id if stamp_standup else None)) if vblock is not None
                                       else f"{tr['result_line']} standup={standup_id}"),
                       "launched_as": dict(row["launched_as"]), "comparison_kind": kind, "standup_id": standup_id,
                       "solo_run_id": (dict(solo_run_ids) if solo_run_ids else None), "solo_store": solo_block, "delta_value_vs_solo": (0.0 if solo_run_ids else None)})
            if store_key is not None:
                tr["summary"] = f"cells/{store_key}/summary/{Z.TASKS_FULL[ev][2]}"
            if tag == IT_TAG:
                tr.update(seed=None, paired=False)
            three[ev]["rows"][tr["key"]] = tr

    # the eight-shaped row launch (three keys here): every s1 store copied under `-shared` on standup A
    for okey in ROW_KEYS:
        is_relaunched = relaunched and okey == RELU
        sfx = _shared_suffix(solo_suffix)
        key = _copy_store(solo_root, root, ROW_DIR, okey, solo_suffix, sfx, standup_id=STANDUP_A, fc=ROW_FC, served=served, digests=digests, max_loras=max_loras, residual_armed=True,
                          result_standup=(standup_override if (standup_override and okey == NONE) else None),
                          row_engine_edit=(row_engine_keys_override if (row_engine_keys_override and okey == NONE) else None),
                          row_dir_override=(row_dir_override if (row_dir_override is not None and okey == NONE) else None),
                          endpoint_key=(served[-1] if endpoint_last_key else None), stamp_fc=(verified_by_override if okey == NONE else None),
                          result_no_standup_line=(verified_result_no_standup and okey == NONE), **shape_kw)
        keys[okey] = key
        standups[okey] = STANDUP_A
        ids = {ev: SOLO[ev].run_id_for(TAG, fx["keys"][okey], okey, inst[ev]) for ev in Z.TASKS_FULL}
        if solo_run_id_override is not None and okey == NONE:
            ids = {ev: solo_run_id_override for ev in Z.TASKS_FULL}
        solo_ids[okey] = ids
        solo_gate = _read(os.path.join(solo_recs, f"gate_{TAG}_{okey}_{solo_suffix}.json"))
        comparison = {"comparison_kind": B.COMPARISON_KIND_SHARED_VS_SOLO, "solo_run_id": dict(ids), "solo_store": solo_store_block(okey)}
        ca = _compared(STANDUP_A, None, f"{solo_recs}/gate_{TAG}_{okey}_{solo_suffix}.json") if okey == NONE else _compared(STANDUP_A, STANDUP_A, none_gate_rec)
        if gate_class_override is not None and okey == RELU:
            ca["comparability_class"] = gate_class_override
        g = _gate(solo_gate, shared_suffix=sfx, standup_id=STANDUP_A, served=served, max_loras=max_loras, residual_armed=True, comparison=comparison, compared_against=ca,
                  row_dir_rec=_row_record_dir(ROW_DIR, sfx, row_dir_absolute), digests=digests, facts_map_null=manifest_map_null, verification_pass=vblock)
        gate_files[okey] = X.write(os.path.join(recs, f"gate_{TAG}_{okey}_{sfx}.json"), g)
        if is_relaunched:
            # the relaunch (`-shared-r2`, its OWN standup B): the parked `-shared` first attempt stays mirrored COMPLETE (the fixture's hazard 7); the row names the r2 store
            sfx2 = _shared_suffix(solo_suffix, "-r2")
            _copy_store(solo_root, root, ROW_DIR, okey, solo_suffix, sfx2, standup_id=STANDUP_B, fc=ROW_FC_R2, served=[okey], digests=digests, max_loras=2, residual_armed=False, **shape_kw)
            standups[okey] = STANDUP_B
            g2 = _gate(solo_gate, shared_suffix=sfx2, standup_id=STANDUP_B, served=[okey], max_loras=2, residual_armed=False, comparison=comparison,
                       compared_against=_compared(STANDUP_B, STANDUP_A, none_gate_rec), row_dir_rec=_row_record_dir(ROW_DIR, sfx2, row_dir_absolute), digests=digests,
                       facts_map_null=manifest_map_null, verification_pass=vblock)
            X.write(os.path.join(recs, f"gate_{TAG}_{okey}_{sfx2}.json"), g2)
            X.write(os.path.join(recs, B.RELAUNCHES_FILE), {f"{TAG}/{okey}": {"tag": "shared-r2", "sha": TREE, "parked_suffix": sfx, "parked_suffixes": [sfx], "cells": [okey],
                                                              "standup_id": STANDUP_B, "relaunched_standup_id": STANDUP_A, "why": "synthetic: a subset relaunch on its own standup"}})
            add_row(okey, shared_suffix=sfx2, kind=B.COMPARISON_KIND_SHARED_VS_SOLO, standup_id=STANDUP_B, solo_block=solo_store_block(okey), solo_run_ids=ids,
                    extra_row={"relaunched_from": {"suffix": sfx, "suffixes": [sfx], "why": "synthetic: the first attempt parked"}, "recovery": {"relaunch": "r2", "class": "cross_standup"}})
            continue
        add_row(okey, shared_suffix=sfx, kind=B.COMPARISON_KIND_SHARED_VS_SOLO, standup_id=STANDUP_A, solo_block=solo_store_block(okey), solo_run_ids=ids,
                spelling=(spelling_override if (spelling_override and okey == NONE) else None), stamp_fc=(verified_by_override if okey == NONE else None))
    if repeat:
        # the row's none driven again on the SAME standup: `-shared-rep2`, class repeat, comparison_kind repeat_floor, keyed s1/none@rep2 (hazard 2)
        sfx_rep = _shared_suffix(solo_suffix, "-rep2")
        _copy_store(solo_root, root, ROW_DIR, NONE, solo_suffix, sfx_rep, standup_id=STANDUP_A, fc=ROW_FC, served=served, digests=digests, max_loras=max_loras, residual_armed=True, owner_key="none@rep2",
                    endpoint_key=(served[-1] if endpoint_last_key else None), **shape_kw)
        solo_gate = _read(os.path.join(solo_recs, f"gate_{TAG}_{NONE}_{solo_suffix}.json"))
        comparison = {"comparison_kind": B.COMPARISON_KIND_REPEAT_FLOOR, "solo_run_id": dict(solo_ids[NONE]), "solo_store": solo_store_block(NONE)}
        g = _gate(solo_gate, shared_suffix=sfx_rep, standup_id=STANDUP_A, served=served, max_loras=max_loras, residual_armed=True, comparison=comparison,
                  compared_against=_compared(STANDUP_A, STANDUP_A, none_gate_rec), row_dir_rec=_row_record_dir(ROW_DIR, sfx_rep, row_dir_absolute), digests=digests,
                  facts_map_null=manifest_map_null, verification_pass=vblock)
        X.write(os.path.join(recs, f"gate_{TAG}_{NONE}_{sfx_rep}.json"), g)
        add_row(NONE, shared_suffix=(_shared_suffix(solo_suffix) if repeat_row_without_segment else sfx_rep), kind=B.COMPARISON_KIND_REPEAT_FLOOR, standup_id=STANDUP_A, cls=B.REPEAT_CLASS,
                tuple_suffix="@rep2", solo_block=solo_store_block(NONE), solo_run_ids=solo_ids[NONE],
                extra_row={"repeat": {"of": NONE, "index": 2, "key": "none@rep2", "store_of": B.rel_store(ROW_DIR, NONE, _shared_suffix(solo_suffix))}})
    it_key = None
    if it_none:
        # the instruct row's OWN none (the instruct checkpoint with no adapter): a copy of the s1 none under `gemma4e4b-f22it`, wave `gemma4e4b_it`, its lane_roster
        # citing the roster of ANOTHER tree (hazard 5: matched by hash within the set), class unsteered, comparison_kind it_none_comparison, no solo run, no anchor
        it_sfx = f"bcbw-{IT_TAG}-{TREE[:7]}-{B.SHARED_STORE_MARKER}"
        it_served, it_digests = [NONE, Z.IT_KEY], {X.harness_id(Z.IT_KEY): Z.IT_OBJ}
        it_key = _copy_store(solo_root, root, ROW_DIR, NONE, solo_suffix, it_sfx, standup_id=STANDUP_IT, fc=ROW_FC_IT, served=it_served, digests=it_digests,
                             max_loras=2, residual_armed=False, new_row_dir=IT_ROW_DIR, new_wave=IT_TAG, endpoint_key=(it_served[-1] if endpoint_last_key else None), **shape_kw)
        # the cross-tree roster citation: a second roster file (another tree, the same waves) in the set, cited by hash
        roster_bytes = open(os.path.join(recs, Z.ROSTER_FILE), "rb").read()
        other = roster_bytes + b"\n# another producer tree's copy of the same roster (synthetic)\n"
        other_sha = hashlib.sha256(other).hexdigest()
        other_name = f"{Z.ROSTER_FILE[:-5]}.{TREE2[:7]}.yaml"
        with open(os.path.join(recs, other_name), "wb") as fh:
            fh.write(other)
        idx = _read(os.path.join(recs, B.ROSTERS_INDEX_FILE))
        entries = idx["rosters"] if isinstance(idx.get("rosters"), list) else None
        if entries is not None:
            entries.append({"roster_sha256": other_sha, "file": other_name, "tree_sha": TREE2, "pins_file": Z.PINS_FILE_T1, "pins_sha256": fx["pins_sha256"], "stores": []})
        else:
            idx["rosters"][other_sha] = {"file": other_name, "tree_sha": TREE2, "stores": [], "pins": Z.PINS_FILE_T1}
        X.write(os.path.join(recs, B.ROSTERS_INDEX_FILE), idx)
        it_dir = os.path.join(root, B.STORES_HEAD, IT_ROW_DIR, NONE, B.THINKING_DIR, it_sfx)
        lr = _read(os.path.join(it_dir, B.LAUNCHED_ENTRY_FILE))
        lr["roster_sha256"] = other_sha
        X.write(os.path.join(it_dir, B.LAUNCHED_ENTRY_FILE), lr)
        solo_gate = _read(os.path.join(solo_recs, f"gate_{TAG}_{NONE}_{solo_suffix}.json"))
        comparison = {"comparison_kind": B.COMPARISON_KIND_IT_NONE, "solo_run_id": None, "solo_store": None, "anchor_role": None}
        g = _gate(solo_gate, shared_suffix=it_sfx, standup_id=STANDUP_IT, served=it_served, max_loras=2, residual_armed=False, comparison=comparison, compared_against=None,
                  row_dir_rec=_row_record_dir(IT_ROW_DIR, it_sfx, row_dir_absolute), digests=it_digests, note="a row-engine row's FIRST record: no prior record is needed", row=IT_ROW_DIR,
                  wave=IT_TAG, store_key=it_key, facts_map_null=manifest_map_null, verification_pass=vblock)
        it_gate_rec = X.write(os.path.join(recs, f"gate_{IT_TAG}_{NONE}_{it_sfx}.json"), g)
        it_decl = {"block": B.INSTRUCT_BLOCK, "transfer_kind": B.INSTRUCT_TRANSFER_KIND, "seed_base": None, "anchor_tuple": None, "model_key": MK, "cell_slug": NONE}
        add_row(NONE, shared_suffix=it_sfx, kind=B.COMPARISON_KIND_IT_NONE, standup_id=STANDUP_IT, cls=B.IT_NONE_CLASS, tag=IT_TAG, row_dir=IT_ROW_DIR, selection=IT_SEL, store_key=it_key,
                extra_row=dict(it_decl))
        standups["it/none"] = STANDUP_IT
        if it_repeat:
            # the instruct row's OWN none served a SECOND time on the same standup (`-shared-rep2` under the `_it` row; the real mirror's `qwen35_4b_it|none@rep2`): class repeat,
            # comparison_kind repeat_floor, NO solo run and NO solo store (the `_it` none has no solo twin), the repeat block; its gate compares against the it none's record
            # (same_standup); its three-number row `<it tag>/none@rep2` with delta_value_vs_solo null (SHARED_IT_REPEAT_RULE)
            it_sfx_rep = f"{it_sfx}-{B.SHARED_REPEAT_MARKER}2"
            _copy_store(solo_root, root, ROW_DIR, NONE, solo_suffix, it_sfx_rep, standup_id=STANDUP_IT, fc=ROW_FC_IT, served=it_served, digests=it_digests, max_loras=2, residual_armed=False,
                        new_row_dir=IT_ROW_DIR, new_wave=IT_TAG, owner_key="none@rep2", endpoint_key=(it_served[-1] if endpoint_last_key else None), **shape_kw)
            comparison = {"comparison_kind": B.COMPARISON_KIND_REPEAT_FLOOR, "solo_run_id": None, "solo_store": None, "repeat_of": {"cell": NONE, "key": "none@rep2",
                          "store": B.rel_store(IT_ROW_DIR, NONE, it_sfx)}, "anchor_role": None}
            g = _gate(solo_gate, shared_suffix=it_sfx_rep, standup_id=STANDUP_IT, served=it_served, max_loras=2, residual_armed=False, comparison=comparison,
                      compared_against=_compared(STANDUP_IT, STANDUP_IT, it_gate_rec), row_dir_rec=_row_record_dir(IT_ROW_DIR, it_sfx_rep, row_dir_absolute), digests=it_digests,
                      row=IT_ROW_DIR, wave=IT_TAG, store_key=it_key, facts_map_null=manifest_map_null, verification_pass=vblock)
            X.write(os.path.join(recs, f"gate_{IT_TAG}_{NONE}_{it_sfx_rep}.json"), g)
            add_row(NONE, shared_suffix=it_sfx_rep, kind=B.COMPARISON_KIND_REPEAT_FLOOR, standup_id=STANDUP_IT, cls=B.REPEAT_CLASS, tag=IT_TAG, row_dir=IT_ROW_DIR, selection=IT_SEL,
                    store_key=it_key, tuple_suffix="@rep2",
                    extra_row=dict(it_decl, repeat={"of": NONE, "index": 2, "key": "none@rep2", "store_of": B.rel_store(IT_ROW_DIR, NONE, it_sfx)}))
            standups["it/none@rep2"] = STANDUP_IT
    for ev in Z.TASKS_FULL:
        three[ev]["n_rows"] = len(launch[ev])
        three[ev]["n_complete"] = len(three[ev]["rows"])
        if three_override and ev == eval_id:
            three_override(three[ev])
        X.write(os.path.join(recs, SHARED[ev].THREE_NUMBERS_FILE), three[ev])
    launch_doc = {"schema": B.LAUNCH_SET_SCHEMA, "derived_at": NOW, "trees": [{"sha": TREE, "role": "primary"}],
                  "comparison_kinds": {k: "synthetic" for k in B.COMPARISON_KINDS},
                  # landing 2's bound set spells the BASE source here (SHARED_L2_RUN_ID_RULE: recorded as provenance, the root binds); landing 1's spells its own
                  "evals": {ev: {"own_source": (own_source_override or SHARED[ev].SOURCE), "launch": launch[ev], "carried": [], "carried_late": [],
                                 "canonical_decode": solo_ls["evals"][ev].get("canonical_decode")} for ev in Z.TASKS_FULL}}
    if launch_override:
        launch_override(launch_doc)
    X.write(os.path.join(recs, B.SHARED_LAUNCH_SET_FILE), launch_doc)
    X.write(os.path.join(recs, B.ROWS_LAUNCH_FILE), "launched_at\tsha\twave\trow\tcells\tstage\n")
    X.write(os.path.join(recs, B.SHARED_SCHEDULE_FILE), {"written": NOW, "ceiling": {"h": 16.0}, "planned": []})
    if no_solo_stores:
        shutil.rmtree(os.path.join(solo_root, B.STORES_HEAD))
    return {"root": root, "solo_root": solo_root, "solo": fx, "roster": fx["roster"], "pins": fx["pins"], "arditi_roster": fx["arditi_roster"], "arditi_pins": fx["arditi_pins"],
            "keys": keys, "solo_keys": fx["keys"], "solo_run_ids": solo_ids, "instruments": inst, "standups": standups, "it_key": it_key, "specs": fx["specs"], "all_specs": fx["all_specs"],
            "gate_files": gate_files, "solo_suffix": solo_suffix, "landing": landing, "verified": (vblock is not None), "verifier_fc": VERIFIER_FC,
            "verified_fields": (verified_fields if vblock is not None else None), "stamp_standup": stamp_standup, "route_scalar": (Z.OBJ if route_scalar else None)}


def adapter(eval_id: str, fx: dict, seeds: LC.Seeds, **seed_kw):
    """The shared reader over the fixture (landing 1's, or the `_l2` / `_l3` reader for a `landing=2` / `landing=3` fixture), its solo side the solo root beside it,
    with the seeds copy that declares the fixture's item set."""
    cls = ADAPTERS_BY_LANDING[int(fx.get("landing") or 1)][eval_id]
    return cls(root=fx["root"], seeds=Z.seeds_for(seeds, **seed_kw), roster_path=fx["roster"], pins_path=fx["pins"], arditi_roster_path=fx["arditi_roster"],
               arditi_pins_path=fx["arditi_pins"], solo_root=fx["solo_root"])


def solo_adapter(eval_id: str, fx: dict, seeds: LC.Seeds):
    """The SOLO reader over the solo root beside the shared fixture."""
    return Z.adapter(SOLO[eval_id].BigcodebenchFullAdapter if eval_id == "bigcodebench" else SOLO[eval_id].BfclFullAdapter, fx["solo"], seeds)


def read_all(ad, conn=None):
    """Every group of the shared mirror read on `conn`, or on a frame primed from the adapter's own emissions with the solo runs it names made current
    (`X.primed_frame`: the steady state after both sources' folds): ``{group: (records by type, status extra)}``; every record validated."""
    from sisteer.registry import validate as VAL
    inputs = ad.discover()
    groups: Dict[str, list] = {}
    for i in inputs:
        groups.setdefault(ad.comparison_group(i), []).append(i)
    conn = X.primed_conn(ad, conn)
    out = {}
    for g in sorted(groups):
        recs = []
        for batch in ad.read(X.ctx(conn), g, groups[g]):
            recs.extend(batch.records)
        bad = [str(v) for v in (VAL.validate_record(r, path="<test>") for r in recs) if isinstance(v, VAL.Malformed)]
        assert not bad, bad[:5]
        bt: Dict[str, list] = {}
        for r in recs:
            bt.setdefault(r["record_type"], []).append(r["payload"])
        out[g] = (bt, ad.status_extra(g))
    return out, inputs


def runs_by_key(bt) -> Dict[str, dict]:
    """The runs of a group keyed ``<producer key's store half>`` -- the object key with the repeat segment (``none``, ``none@rep2``, ...) read off extra.shared_standup.row_key."""
    return {r["extra"]["shared_standup"]["row_key"]: r for r in bt.get("run", [])}
