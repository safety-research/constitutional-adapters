"""The DRIVER side of a
shared standup per model row: ``served_capability_run.py generate --row-engine-record
<path>`` (``contracts.FLAG_ROW_ENGINE_RECORD``) copies the row runner's per-cell record
into ``manifest["row_engine"]`` (``contracts.MANIFEST_KEY_ROW_ENGINE``), top-level and
UNHASHED like ``producer``, with EXACTLY the keys ``contracts.ROW_ENGINE_MANIFEST_FIELDS``
(``compile_cache`` exactly ``contracts.ROW_ENGINE_COMPILE_CACHE_FIELDS``;
``verification_pass`` null in landing 1; the reader returns the block in the contract's
order and the store writer sorts keys on disk). The record is read and held to the contract
BEFORE any store is touched (``contracts.read_row_engine_record``: a wrong key set refuses
naming the missing / unknown keys; the same rule the row runner validates what it writes
through, ``validate_row_engine_record``). Without the flag the manifest is BYTE-IDENTICAL
to the golden captured from the pre-change driver (the landing branch's base,
264ec44cf) on the native world — ``tests/fixtures/tinycap_row_engine/
driver_manifest_native_gsm8k_no_flag.json`` (paths under the tmp dir, the producer tree
sha, the env digest and timestamps normalised, ``tinycap_golden.normalise``); the
hashed served identity, ``harness_config_hash`` and the cell-key directory are IDENTICAL
with and without the flag, so a row-engine store keys exactly as a solo store of the
same cell (it lives under its own ``-shared`` suffix, never under a solo store's path).
``driver_argv(row_engine_record=...)`` appends the flag LAST, after ``--attempt-id``;
None -> byte-identical argv. ``_check_manifest_serving_identity`` is untouched.

The golden is the real thing this module proves, so it is compared FIRST in the no-flag
test; a departure names the leaf (``tinycap_golden.leaf_diffs``). Rewriting it is a
deliberate manifest-shape change, never maintenance.

The world: ``test_served_capability_native.native_world`` (a native LoRA cell on a
native standup record — the engine a row engine actually is) driving the standard
GSM8K control through the faked Inspect leg of ``test_tinycap_identity_version``, so the
manifest carries the tinycap blocks the shared-store contract reads:
``provenance`` and the route block's ``object_artifact_sha256`` +
``object_artifact_sha256_by_cell``, read off the standup record — both asserted
unchanged under the flag, since the driver copies the record's map and never derives
the scalar from the row (``route.object_artifact_sha256`` is whatever the row
engine's record states; null when several digests are served is the STANDUP's fact).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_served_capability as LENS  # noqa: E402  (module: stub_framework_versions, _percell_argv, _cell_dir, srun)
import tinycap_golden as golden  # noqa: E402
from test_served_capability_native import (  # noqa: F401  (`native_world` is a fixture)
    HARNESS_CELL,
    OBJECT_SHA,
    native_world,
)
from test_tinycap_identity_version import _fake_gsm8k_leg  # noqa: E402

from sisteer.capeval.tinycap import contracts as tc

srun = LENS.srun
REPO = LENS.REPO
GOLDEN_DIR = REPO / "tests" / "fixtures" / "tinycap_row_engine"
GOLDEN = GOLDEN_DIR / "driver_manifest_native_gsm8k_no_flag.json"
#: THIS set's volatile leaves — the identity-version set's table, copied (each golden set
#: passes the table its fixtures were captured under; the tables are never merged).
_VOLATILE = {"tree_sha": "<TREE_SHA>", "env_digest": "<ENV_DIGEST>", "written_at": "<TS>", "ts": "<TS>"}
#: the row runner's per-cell record (``srv/cells/<cell>/row_engine.json``),
#: synthetic, in the contract's shape — what module A writes to the path it passes with the flag
RECORD = {
    "row_engine": True,
    "standup_id": "attempt-11111111-2222-3333-4444-555555555555",
    "engine_cells": ["none", HARNESS_CELL, "prompted__safety"],
    "max_loras": 2,
    "residual_slot_armed": False,
    "compile_cache": {"hit": True, "key": "llama8b:vllm-0.11.0:S4:L4096:ml2:r1",
                      "vllm_cache_dir": "/root/.cache/vllm/torch_compile_cache/0123abcd"},
    "row_dir": "/vol/tinycap/gen/rows/llama8b-f22s1/bcbw-llama8b_f22s1-264ec44-shared",
    "verification_pass": None,
}


def _normalise(obj, *, tmp_path: Path):
    return golden.normalise(obj, tmp_path=tmp_path, volatile=_VOLATILE)


def _write_record(tmp_path: Path, record, name: str = "row_engine.json") -> Path:
    p = tmp_path / "srv_cells" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2) if not isinstance(record, str) else record)
    return p


def _run(world, monkeypatch, *, extra=(), out_root=None):
    """One ``generate`` pass of the standard GSM8K control on the native world's LoRA
    cell (the faked Inspect leg files the task); returns ``(cell_dir, manifest)``."""
    LENS.stub_framework_versions(monkeypatch)
    _fake_gsm8k_leg(monkeypatch, [])
    argv = LENS._percell_argv(world, tc.CONTROL_GSM8K, out_root=out_root)
    srun.main(argv + [tc.FLAG_THINKING, tc.THINKING_OFF, *extra])
    cell_dir = LENS._cell_dir(out_root if out_root is not None else world.out_root)
    return cell_dir, json.loads((cell_dir / "manifest.json").read_text())


def _golden_shape(cell_dir: Path, manifest: dict, *, tmp_path: Path) -> dict:
    return {"cell_dir_name": cell_dir.name, "harness_config_hash": manifest["harness_config_hash"],
            "identity": manifest["resolved_config"]["tinycap"],
            "manifest": _normalise(manifest, tmp_path=tmp_path)}


# --------------------------------------------------------------------------- #
# the manifest: no flag -> the pre-change golden, byte for byte                #
# --------------------------------------------------------------------------- #


def test_without_the_flag_the_manifest_is_byte_identical_to_the_pre_change_golden(native_world, monkeypatch):
    """Without the flag the manifest is byte-identical. The golden was
    captured from the driver BEFORE the flag existed (264ec44cf, the landing branch's
    base) on this world; the compare runs FIRST, so a capture on the pre-change tree
    (where the names below do not exist) writes the fixture before anything else runs."""
    w = native_world
    cell_dir, manifest = _run(w, monkeypatch)
    golden.compare_or_write(GOLDEN, _golden_shape(cell_dir, manifest, tmp_path=w.tmp_path))
    assert "row_engine" not in manifest and tc.MANIFEST_KEY_ROW_ENGINE not in manifest
    # the shared-store contract's per-cell digest: `provenance.artifact_sha256`
    # is the fitted object's sha on every run; the route block's digest scalar + by-cell map ride
    # only when the run was TOLD a digest to compare (`--expect-object-sha256`, the row engine's
    # E40 H1 door) — this run was not, so both are null, as the golden pins
    assert manifest[tc.MANIFEST_KEY_PROVENANCE]["artifact_sha256"] == OBJECT_SHA
    route = manifest[tc.MANIFEST_KEY_ROUTE]
    assert route[tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] is None
    assert route[tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL] is None


# --------------------------------------------------------------------------- #
# the manifest under the flag: the block, exactly; nothing else moves          #
# --------------------------------------------------------------------------- #


def test_with_the_flag_the_block_is_exactly_the_record_and_nothing_else_in_the_manifest_moves(
        native_world, monkeypatch):
    """Under ``--row-engine-record <path>`` the manifest gains ONE top-level key,
    ``row_engine``, holding the record's values under exactly
    ``ROW_ENGINE_MANIFEST_FIELDS`` in the contract's order (``compile_cache`` in
    ``ROW_ENGINE_COMPILE_CACHE_FIELDS``' order, whatever order the file spelled them);
    every other leaf of the manifest — ``provenance``, the route block's digest map,
    the hashed identity, ``harness_config_hash``, the cell-key directory — is the
    golden's, byte for byte."""
    w = native_world
    # the file spells the keys in a scrambled order: the manifest's order is the contract's
    scrambled = {k: RECORD[k] for k in reversed(tc.ROW_ENGINE_MANIFEST_FIELDS)}
    scrambled["compile_cache"] = {k: RECORD["compile_cache"][k] for k in reversed(tc.ROW_ENGINE_COMPILE_CACHE_FIELDS)}
    path = _write_record(w.tmp_path, scrambled)
    cell_dir, manifest = _run(w, monkeypatch, extra=[tc.FLAG_ROW_ENGINE_RECORD, str(path)])
    block = manifest[tc.MANIFEST_KEY_ROW_ENGINE]
    assert block == RECORD
    # exactly the contract's keys (the store writer sorts keys on disk — `_write_json_atomic`,
    # sort_keys=True — so the FILE's order is alphabetical; the contract's order is the reader's
    # return, pinned by the contracts test below)
    assert set(block) == set(tc.ROW_ENGINE_MANIFEST_FIELDS) and len(block) == len(tc.ROW_ENGINE_MANIFEST_FIELDS)
    assert set(block["compile_cache"]) == set(tc.ROW_ENGINE_COMPILE_CACHE_FIELDS)
    assert tuple(block) == tuple(sorted(tc.ROW_ENGINE_MANIFEST_FIELDS))
    assert block["verification_pass"] is None                       # landing 1
    # nothing else moved: the manifest minus the block IS the golden's manifest
    want = json.loads(GOLDEN.read_text())
    got = _golden_shape(cell_dir, manifest, tmp_path=w.tmp_path)
    rest = {k: v for k, v in got["manifest"].items() if k != tc.MANIFEST_KEY_ROW_ENGINE}
    assert golden.leaf_diffs(rest, want["manifest"]) == []
    assert set(got["manifest"]) - set(want["manifest"]) == {tc.MANIFEST_KEY_ROW_ENGINE}
    assert got["harness_config_hash"] == want["harness_config_hash"]
    assert got["identity"] == want["identity"]
    assert got["cell_dir_name"] == want["cell_dir_name"]
    # named, because the shared-store contract reads them per cell:
    # the provenance block and the route block's digest scalar + by-cell map are the
    # standup record's facts, untouched by the row-engine block
    assert got["manifest"][tc.MANIFEST_KEY_PROVENANCE] == want["manifest"][tc.MANIFEST_KEY_PROVENANCE]
    assert got["manifest"][tc.MANIFEST_KEY_ROUTE] == want["manifest"][tc.MANIFEST_KEY_ROUTE]
    assert manifest[tc.MANIFEST_KEY_PROVENANCE]["artifact_sha256"] == OBJECT_SHA


def test_the_route_digest_facts_and_the_provenance_digest_are_unaffected_by_the_flag_when_a_digest_is_stated(
        native_world, monkeypatch):
    """The row engine's per-cell driver runs the way a fitted cell's driver always does —
    TOLD the digest the standup was held to (`--expect-object-sha256`, E40 H1) — and the
    route block then carries the standup record's digest scalar and by-cell map
    (`_attestation_route_block`), while `provenance.artifact_sha256` is the fitted object's
    sha. The shared-store reader takes the per-cell digest from
    `provenance.artifact_sha256` plus the by-cell map, and the scalar is whatever the
    STANDUP record states (null when a row engine serves several digests). The driver's
    flag changes none of these: two passes, both told the digest, one under the flag."""
    w = native_world
    path = _write_record(w.tmp_path, RECORD)
    told = [tc.FLAG_EXPECT_OBJECT_SHA256, OBJECT_SHA]
    plain_dir, plain = _run(w, monkeypatch, extra=told)
    flag_dir, flagged = _run(w, monkeypatch, extra=[*told, tc.FLAG_ROW_ENGINE_RECORD, str(path)],
                             out_root=w.tmp_path / "out_flagged")
    assert plain_dir.name == flag_dir.name and plain["harness_config_hash"] == flagged["harness_config_hash"]
    for m in (plain, flagged):
        route = m[tc.MANIFEST_KEY_ROUTE]
        assert route[tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] == OBJECT_SHA
        assert route[tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL] == {HARNESS_CELL: OBJECT_SHA}
        assert m[tc.MANIFEST_KEY_PROVENANCE]["artifact_sha256"] == OBJECT_SHA
    assert plain[tc.MANIFEST_KEY_ROUTE] == flagged[tc.MANIFEST_KEY_ROUTE]
    assert plain[tc.MANIFEST_KEY_PROVENANCE] == flagged[tc.MANIFEST_KEY_PROVENANCE]
    assert flagged[tc.MANIFEST_KEY_ROW_ENGINE] == RECORD and tc.MANIFEST_KEY_ROW_ENGINE not in plain
    assert {k for k in flagged if k not in plain} == {tc.MANIFEST_KEY_ROW_ENGINE}
    assert golden.leaf_diffs({k: v for k, v in flagged.items() if k != tc.MANIFEST_KEY_ROW_ENGINE},
                             plain) == []


def test_the_hashed_identity_is_identical_with_and_without_the_flag(native_world, monkeypatch):
    """The served-identity hash MUST NOT include the row_engine block: two passes on
    one world, one with the flag and one without, produce the same
    ``harness_config_hash``, ``intervention_hash``, the same whole ``resolved_config``
    (the hashed identity: ``tinycap`` + ``serving`` + the rest) and the same cell-key
    directory name. And by construction: ``row_engine`` is in no identity field list,
    and the manifest key is top-level like ``producer`` / ``quality`` — NOT one of
    ``TINYCAP_MANIFEST_KEYS`` (the additive tinycap blocks)."""
    w = native_world
    path = _write_record(w.tmp_path, RECORD)
    plain_dir, plain = _run(w, monkeypatch)
    other_root = w.tmp_path / "out_flagged"
    flag_dir, flagged = _run(w, monkeypatch, extra=[tc.FLAG_ROW_ENGINE_RECORD, str(path)], out_root=other_root)
    assert plain_dir.name == flag_dir.name
    for key in ("harness_config_hash", "intervention_hash", "resolved_config", "row_constants",
                "serving_identity", "virtual_model", "cell_id"):
        assert plain[key] == flagged[key], key
    assert flagged["resolved_config"]["tinycap"]["served"] == plain["resolved_config"]["tinycap"]["served"]
    assert tc.MANIFEST_KEY_ROW_ENGINE in flagged and tc.MANIFEST_KEY_ROW_ENGINE not in plain
    assert flagged[tc.MANIFEST_KEY_ROW_ENGINE] == RECORD
    # by construction, in the contracts
    assert tc.MANIFEST_KEY_ROW_ENGINE == "row_engine"
    assert "row_engine" not in tc.SERVED_IDENTITY_FIELDS
    for protocol in (False, True):
        for native in (False, True):
            assert "row_engine" not in tc.served_identity_fields(protocol_sampling=protocol, native=native)
    assert "row_engine" not in tc.TINYCAP_IDENTITY_FIELDS
    assert tc.MANIFEST_KEY_ROW_ENGINE not in tc.TINYCAP_MANIFEST_KEYS
    assert tc.MANIFEST_KEY_ROW_ENGINE not in tc.ROUTE_BLOCK_FIELDS
    # the resume guard is untouched: it reads no row-engine key
    import inspect
    assert "row_engine" not in inspect.getsource(srun._check_manifest_serving_identity)


# --------------------------------------------------------------------------- #
# refusals: a wrong key set, before any store is touched                       #
# --------------------------------------------------------------------------- #


def _without(record, key):
    return {k: v for k, v in record.items() if k != key}


@pytest.mark.parametrize("mutate,needles", [
    (lambda r: _without(r, "row_dir"), ["missing ['row_dir']"]),
    (lambda r: {**r, "watcher": "x"}, ["unknown ['watcher']"]),
    (lambda r: {**_without(r, "standup_id"), "standup": "a"}, ["missing ['standup_id']", "unknown ['standup']"]),
    (lambda r: {**r, "compile_cache": _without(r["compile_cache"], "vllm_cache_dir")},
     ["compile_cache", "missing ['vllm_cache_dir']"]),
    (lambda r: {**r, "compile_cache": {**r["compile_cache"], "reason": "no_tarball"}},
     ["compile_cache", "unknown ['reason']"]),
    (lambda r: {**r, "row_engine": False}, ["row_engine must be true"]),
    (lambda r: {**r, "engine_cells": []}, ["engine_cells"]),
    (lambda r: {**r, "max_loras": "2"}, ["max_loras"]),
    (lambda r: {**r, "residual_slot_armed": "no"}, ["residual_slot_armed"]),
    (lambda r: {**r, "verification_pass": "pending"}, ["verification_pass"]),
    (lambda r: "{not json", ["unreadable"]),
    (lambda r: [r], ["expected a JSON object"]),
], ids=["missing-key", "extra-key", "missing-and-extra", "cache-missing-key", "cache-extra-key",
        "row_engine-false", "engine_cells-empty", "max_loras-str", "residual-str", "verification-str",
        "not-json", "not-an-object"])
def test_a_wrong_record_refuses_by_name_before_any_store_is_touched(native_world, monkeypatch, mutate, needles):
    """A record whose key set is not EXACTLY ``ROW_ENGINE_MANIFEST_FIELDS`` (or whose
    ``compile_cache`` is not exactly ``ROW_ENGINE_COMPILE_CACHE_FIELDS``), a record that
    says ``row_engine: false``, a wrongly typed field, an unreadable file: each refuses
    naming the flag, the path and the departure — before the store root exists."""
    w = native_world
    path = _write_record(w.tmp_path, mutate(json.loads(json.dumps(RECORD))))
    with pytest.raises(SystemExit) as ei:
        _run(w, monkeypatch, extra=[tc.FLAG_ROW_ENGINE_RECORD, str(path)])
    msg = str(ei.value)
    assert msg.startswith(f"[served-cap] {tc.FLAG_ROW_ENGINE_RECORD}") and str(path) in msg, msg
    for needle in needles:
        assert needle in msg, (needle, msg)
    assert not w.out_root.exists()                                  # nothing was touched


def test_a_missing_record_file_refuses_by_name_before_any_store_is_touched(native_world, monkeypatch):
    w = native_world
    path = w.tmp_path / "nowhere" / "row_engine.json"
    with pytest.raises(SystemExit, match="unreadable") as ei:
        _run(w, monkeypatch, extra=[tc.FLAG_ROW_ENGINE_RECORD, str(path)])
    assert str(path) in str(ei.value) and not w.out_root.exists()


# --------------------------------------------------------------------------- #
# the contracts: the reader's one rule, the argv builder, the parser           #
# --------------------------------------------------------------------------- #


def test_validate_row_engine_record_is_the_one_rule_and_returns_the_block_in_contract_order():
    """``validate_row_engine_record`` (what ``read_row_engine_record`` applies to the file
    and what the row runner validates its own write through) admits the contract's
    shape and returns the block in ``ROW_ENGINE_MANIFEST_FIELDS`` order with
    ``compile_cache`` in ``ROW_ENGINE_COMPILE_CACHE_FIELDS`` order; ``verification_pass``
    admits null (landing 1) or a JSON object (landing 2's ``{"state": ...}``), nothing
    else; ``vllm_cache_dir`` admits a path or null (a verdict with no cache line)."""
    assert tc.ROW_ENGINE_MANIFEST_FIELDS == ("row_engine", "standup_id", "engine_cells", "max_loras",
                                             "residual_slot_armed", "compile_cache", "row_dir", "verification_pass")
    assert tc.ROW_ENGINE_COMPILE_CACHE_FIELDS == ("hit", "key", "vllm_cache_dir")
    assert tc.FLAG_ROW_ENGINE_RECORD == "--row-engine-record"
    scrambled = {k: RECORD[k] for k in reversed(tc.ROW_ENGINE_MANIFEST_FIELDS)}
    scrambled["compile_cache"] = {k: RECORD["compile_cache"][k] for k in reversed(tc.ROW_ENGINE_COMPILE_CACHE_FIELDS)}
    out = tc.validate_row_engine_record(scrambled)
    assert out == RECORD and tuple(out) == tc.ROW_ENGINE_MANIFEST_FIELDS
    assert tuple(out["compile_cache"]) == tc.ROW_ENGINE_COMPILE_CACHE_FIELDS
    assert out is not scrambled and out["compile_cache"] is not scrambled["compile_cache"]
    # landing 2's shape and a cold verdict without a cache line are admitted
    pend = tc.validate_row_engine_record({**RECORD, "verification_pass": {"state": "pending", "tasks": ["x"]}})
    assert pend["verification_pass"] == {"state": "pending", "tasks": ["x"]}
    cold = tc.validate_row_engine_record({**RECORD, "compile_cache": {"hit": False, "key": "k", "vllm_cache_dir": None}})
    assert cold["compile_cache"] == {"hit": False, "key": "k", "vllm_cache_dir": None}
    assert issubclass(tc.RowEngineRecordError, ValueError)
    # each departure, by name (`where` prefixes every message)
    for bad, needle in [
        ({**RECORD, "row_engine": 1}, "row_engine must be true"),
        ({**RECORD, "standup_id": ""}, "standup_id"),
        ({**RECORD, "standup_id": 7}, "standup_id"),
        ({**RECORD, "engine_cells": "none"}, "engine_cells"),
        ({**RECORD, "engine_cells": ["none", "none"]}, "engine_cells"),
        ({**RECORD, "engine_cells": ["none", 3]}, "engine_cells"),
        ({**RECORD, "max_loras": True}, "max_loras"),
        ({**RECORD, "max_loras": -1}, "max_loras"),
        ({**RECORD, "residual_slot_armed": 0}, "residual_slot_armed"),
        ({**RECORD, "compile_cache": None}, "compile_cache"),
        ({**RECORD, "compile_cache": {**RECORD["compile_cache"], "hit": "yes"}}, "compile_cache.hit"),
        ({**RECORD, "compile_cache": {**RECORD["compile_cache"], "key": ""}}, "compile_cache.key"),
        ({**RECORD, "compile_cache": {**RECORD["compile_cache"], "vllm_cache_dir": 3}}, "compile_cache.vllm_cache_dir"),
        ({**RECORD, "row_dir": ""}, "row_dir"),
        ({**RECORD, "verification_pass": []}, "verification_pass"),
        (_without(RECORD, "compile_cache"), "missing ['compile_cache']"),
        ({**RECORD, "extra": 1}, "unknown ['extra']"),
        ("a string", "expected a JSON object"),
    ]:
        with pytest.raises(tc.RowEngineRecordError, match="^my-record: ") as ei:
            tc.validate_row_engine_record(bad, where="my-record")
        assert needle in str(ei.value), (needle, str(ei.value))


def test_read_row_engine_record_reads_the_file_through_the_rule_and_names_the_path(tmp_path):
    p = _write_record(tmp_path, RECORD)
    assert tc.read_row_engine_record(p) == RECORD
    assert tc.read_row_engine_record(str(p)) == RECORD
    bad = _write_record(tmp_path, _without(RECORD, "row_dir"), name="bad.json")
    with pytest.raises(tc.RowEngineRecordError, match="missing \\['row_dir'\\]") as ei:
        tc.read_row_engine_record(bad)
    assert str(ei.value).startswith(str(bad))
    with pytest.raises(tc.RowEngineRecordError, match="unreadable"):
        tc.read_row_engine_record(tmp_path / "absent.json")
    torn = _write_record(tmp_path, "{", name="torn.json")
    with pytest.raises(tc.RowEngineRecordError, match="unreadable"):
        tc.read_row_engine_record(torn)


def test_driver_argv_appends_the_record_path_last_and_is_byte_identical_without_it():
    """``driver_argv(row_engine_record=<path>)`` appends ``--row-engine-record <path>``
    LAST — after the IFEval tail and after ``--attempt-id``; None -> byte-identical
    argv; an empty path refuses by name. NOT in ``TINYCAP_DRIVER_FLAGS`` (the
    always-emitted probe set): like every flag emitted only when given, a row launch
    adds it to the preflight's argparse probe itself."""
    kw = dict(python="p", script="s", proxy_url="u", routing_table="r", cell="none", config="c",
              defenses_dir="d", out_root="o", controls=["bigcodebench_full", "bfcl_full"], thinking="off",
              concurrency=64, judge_model="j")
    plain = tc.driver_argv(**kw)
    assert tc.driver_argv(**kw, row_engine_record=None) == plain
    assert tc.FLAG_ROW_ENGINE_RECORD not in plain
    rec = "/vol/tinycap/gen/rows/r/s/srv/cells/none/row_engine.json"
    assert tc.driver_argv(**kw, row_engine_record=rec) == plain + [tc.FLAG_ROW_ENGINE_RECORD, rec]
    # after everything: the sampling slot, the protocol record, the IFEval tail, the attempt id
    full = tc.driver_argv(**kw, sampling_mode=tc.SAMPLING_PROTOCOL, protocol_check="/p.json", identity_version=2,
                          bank_view="main", attempt_id="a1", row_engine_record=rec)
    assert full[-2:] == [tc.FLAG_ROW_ENGINE_RECORD, rec]
    assert full[-4:-2] == [tc.FLAG_ATTEMPT_ID, "a1"]
    assert full[:-2] == tc.driver_argv(**kw, sampling_mode=tc.SAMPLING_PROTOCOL, protocol_check="/p.json",
                                       identity_version=2, bank_view="main", attempt_id="a1")
    # the ladder pair, the digest and the budget path all come before it
    lad = tc.driver_argv(**dict(kw, controls=["bfcl_200"]),
                         cap_override=tc.format_cap_overrides({tc.LADDER_ROW_TASKS[0]: tc.LADDER_CAP_TO}),
                         item_selection_json=tc.format_item_selection_json({tc.LADDER_ROW_TASKS[0]: ["BigCodeBench/17"]}),
                         expect_object_sha256="0" * 64, oos_prompt_budget="/b.json", row_engine_record=rec)
    assert lad[-2:] == [tc.FLAG_ROW_ENGINE_RECORD, rec] and lad[-4] == tc.FLAG_ITEM_SELECTION_JSON
    with pytest.raises(ValueError, match=f"driver_argv: {tc.FLAG_ROW_ENGINE_RECORD} needs a path"):
        tc.driver_argv(**kw, row_engine_record="")
    assert tc.FLAG_ROW_ENGINE_RECORD not in tc.TINYCAP_DRIVER_FLAGS
    assert tc.FLAG_ROW_ENGINE_RECORD not in tc.IFEVAL_DRIVER_FLAGS
    assert "FLAG_ROW_ENGINE_RECORD" in tc.__all__ and "ROW_ENGINE_MANIFEST_FIELDS" in tc.__all__


def test_the_drivers_parser_accepts_the_flag_and_defaults_it_to_none():
    base = ["generate", "--proxy-url", "http://127.0.0.1:1/v1", "--routing-table", "/r.json", "--cell", "none"]
    ap = srun.build_parser()
    assert ap.parse_args(base).row_engine_record is None
    assert ap.parse_args(base + [tc.FLAG_ROW_ENGINE_RECORD, "/x/row_engine.json"]).row_engine_record == "/x/row_engine.json"
