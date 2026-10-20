"""``served_capability_run`` on a
NATIVE multi-LoRA standup — the LORA-ROUTE WORLD.

The lens world (``test_served_capability.py``) pins the lens path byte for
byte; this module builds the OTHER engine's world exactly as the native core
writes it — a version-2 routing table with ``+lora+`` routes, adapter blocks
and the derived ``none__lora0`` zero adapter (``defense_cells.enumerate_cells
(lens_controls=False, lora_controls=True)`` + ``proxy.export_routing_table
(adapter_blocks=)``), an ``endpoint.json`` shaped as
``native_standup.LoraStandup.endpoint()`` (``engine`` / ``serving_engine`` =
``lora_native``, ``reduction_mode`` = ``not_applicable``, NO
``lens_reduction_mode``, ``cells_skipped``, the ``geometry.adjustments``),
and a ``/v1/models`` identity block carrying ``proxy.engine_payload``'s
``engine`` block — and drives the REAL driver over it with a scripted proxy
that stamps the REAL non-lens triple (cell-stable / not-consulted / render)
on every leg and refuses the lens declaration key the way the proxy does
(``response_prefill_not_lens``). Pinned below:

* the bites proof is owed by the STEERED route (``bs.is_steered``) against
  its PATH-MATCHED twin ``none__lora0`` (``twin_cell_id_for``), and the
  probe's stamps verify under each LEG's engine;
* ``steer_positions == "all"`` (a native adapter applies at every token),
  ``mmlu.response_prefill is False`` (the lens re-location DECLARATION is not
  sent), the seeded MMLU payload equals the lens payload MINUS that key, the
  logprob bites applicable with the key absent from every request;
* the hashed identity carries ``serving_engine`` (present iff native; the lens
  key set unchanged), ``reduction_mode == not_applicable``; the manifest, the
  rows and the route block record the engine the stored-record consumers read;
* ``_standup_endpoint``'s native branch (no lens mode expected, a torn record
  refuses, ``cells_skipped`` honoured, ``--serving-path`` held to the record);
* the ROUTE-AWARE live-proxy compare (a lora cell's adapter by intervention
  hash and digests; ``none`` / a prompted cell as passthrough legs) and the
  §1.8 geometry copy (a gemma-shaped kernel step accepted by the validators; a
  bare smaller S refused).
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import served_capability_run as srun  # noqa: E402
import tinycap_math500_thinking as M  # noqa: E402

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import proxy_engine as pe  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402

# the LENS world's shared pieces (loaders, the scripted tokenizer / tokenize,
# the canned logprob and completion shapes, the ledger record, the LENS live
# identity and endpoint writer) — imported as a MODULE so none of its tests
# are re-collected here
import test_served_capability as LENS  # noqa: E402
import test_petri_proxy as tp  # noqa: E402
import test_tinycap_steered_arm_contract as ARM  # noqa: E402

BASE_MODEL, REV, TEMPLATE_HASH, PROXY_URL = LENS.BASE_MODEL, LENS.REV, LENS.TEMPLATE_HASH, LENS.PROXY_URL
NATIVE = tc.SERVING_ENGINE_LORA_NATIVE
#: the TrainTest id the driver is pointed at and its harness spelling (the
#: fixture source's ONE lora-served method, `lora_r1`, on the safety SI)
TT_CELL = "safety_refusal__lora_r1"
HARNESS_CELL = "si_safety__lora_r1"
ZERO = tc.TWIN_CELL_ID_LORA                     # none__lora0
PROMPTED = "prompted__safety"
LORA_ID = "proxy_vllm+lora+interleaved"
PASS_ID = "proxy_vllm+passthrough+interleaved"
#: the fixture standup's window; S is the IDENTITY S at that window (the §1.8
#: contract re-derives it from the block's own budget and refuses a bare
#: smaller S — the lens world's S=4 override is exactly what a native record
#: may not carry without a named adjustment)
NATIVE_WINDOW = 4096
NATIVE_SEQS = sg.identity_seqs(NATIVE_WINDOW)
OBJECT_SHA = hashlib.sha256(b"synthetic-lora-object-bytes-for-tests").hexdigest()
#: the proxy's refusal code for the lens declaration key on a non-lens route
RESPONSE_PREFILL_NOT_LENS = getattr(px, "RESPONSE_PREFILL_NOT_LENS", "response_prefill_not_lens")
SALT_CELL_STABLE = getattr(tc, "SALT_REGIME_CELL_STABLE", getattr(px, "SALT_REGIME_CELL_STABLE", "cell-stable"))
POLICY_NOT_CONSULTED = getattr(tc, "POLICY_NOT_CONSULTED", getattr(px, "POLICY_NOT_CONSULTED", "not-consulted"))


# --------------------------------------------------------------------------- #
# the native records, shaped as the core writes them                           #
# --------------------------------------------------------------------------- #


def _digests(name: str) -> dict:
    """Two 64-hex digests per adapter name (the shape ``lora_export`` records)."""
    return {"config_sha256": hashlib.sha256(f"cfg:{name}".encode()).hexdigest(),
            "weights_sha256": hashlib.sha256(f"w:{name}".encode()).hexdigest()}


def _write_cfg(tmp_path: Path) -> Path:
    """The lens world's config with the LORA arm in the Class A vocabulary and
    the synthetic compiled object at the registry's path for the lora cell."""
    cfg = {
        "model": BASE_MODEL, "model_revision": REV, "dtype": "bfloat16", "scale": 1.0,
        "si_ids": ["safety_refusal", "si_constitution"],
        "class_a_arms": ["lora_r1"], "class_b_arms": [],
        "defenses_dir": str(tmp_path / "tt_cells"),
        "attacks": {"target_max_new_tokens": 64}, "completion_tokens": 64,
    }
    p = tmp_path / "two_si_native_test.yaml"
    p.write_text(yaml.safe_dump(cfg))
    obj = tmp_path / "tt_cells" / TT_CELL / "object.pt"
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(b"synthetic-lora-object-bytes-for-tests")
    return p


def native_source(tmp_path: Path):
    """The registry a NATIVE standup partitions: the base, one prompted cell and
    ONE natively servable LoRA object — kind ``lora_all`` with ``positions ==
    "all"`` (what ``native_lora_unservable_reason`` admits; the petri_proxy
    fixture's ``obj_lora_r1`` kind is a lens-era spelling the native enumeration
    refuses by name, exactly as it refuses a PSR cell)."""
    artifact = tmp_path / f"{HARNESS_CELL}.pt"
    artifact.write_bytes(b"synthetic-lora-object-bytes:" + HARNESS_CELL.encode())
    entries = {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        PROMPTED: dc.RegistryEntry(cell_id=PROMPTED, kind="prompted", si_id="safety",
                                   si_prefix=tp.SAFETY_PREFIX),
        HARNESS_CELL: dc.RegistryEntry(
            cell_id=HARNESS_CELL, kind="lora_all", si_id="safety", object_path=str(artifact),
            serving_path="lora",
            resolved_spec={"kind": "lora_all", "si_id": "safety", "positions": "all", "rank": 1,
                           "lora_alpha": 1, "scale": 1.0}),
    }
    return dc.MappingCellSource(entries)


def native_cells(tmp_path: Path):
    """ONE native standup's cell set: the unprompted base, one prompted cell and
    the lora cell, enumerated the way the native standup enumerates
    (``lens_controls=False, lora_controls=True``: ``none__lora0`` derived,
    ``none__lens0`` never)."""
    (tmp_path / "objects").mkdir(exist_ok=True)
    src = native_source(tmp_path / "objects")
    spec = tp._spec(serving_mode="interleaved", cells=["none", PROMPTED, HARNESS_CELL])
    cells = dc.enumerate_cells(spec, src, behavior_files=tp._behavior_files(spec),
                               prompt_files=tp.PROMPT_FILES, lens_controls=False, lora_controls=True)
    return cells, src


def adapter_blocks(routes) -> dict:
    """The version-2 table's ``adapters`` map: one validated block per lora
    route (``proxy.ADAPTER_BLOCK_KEYS``), the zero adapter flagged."""
    blocks = {}
    for r in routes:
        if cm.parse_serving_identity(r.serving_identity).path != "lora":
            continue
        zero = r.cell_id.endswith(px.LORA_ZERO_CELL_SUFFIX)
        blocks[r.cell_id] = {"name": r.cell_id, "dir": f"/bank/{r.cell_id}", **_digests(r.cell_id),
                             "r": 1, "petri_intervention_hash": r.intervention_hash,
                             "petri_zero_adapter": zero, "petri_source_kind": None if zero else "lora_r1"}
    return blocks


def native_routing_table(tmp_path: Path) -> tuple[Path, dict]:
    """Export the VERSION-2 table a native standup writes and load it back the
    way the driver does (``load_routing_table``: adapter blocks validated)."""
    cells, src = native_cells(tmp_path)
    routes = px.routes_from_cells(cells)
    table_path = tmp_path / "routing_table.json"
    px.export_routing_table(cells, src, table_path, base_model=BASE_MODEL, serving_mode="interleaved",
                            adapter_blocks=adapter_blocks(routes))
    return table_path, px.load_routing_table(table_path)


def native_geometry(max_model_len: int = NATIVE_WINDOW, max_num_seqs: int = NATIVE_SEQS,
                    adjustments=None) -> dict:
    """``LoraStandup._geometry_block``: the resolver's ``as_dict()`` + the §1.8
    ``adjustments`` list (``[]`` = identity)."""
    block = sg.resolve_geometry(None, max_model_len=max_model_len, max_num_seqs=max_num_seqs).as_dict()
    block[sg.ADJUSTMENTS_KEY] = copy.deepcopy(adjustments) if adjustments is not None else []
    return block


def engine_adapters(table: dict) -> dict:
    """The ATTESTED per-adapter entries (``proxy.ENGINE_ADAPTER_KEYS``) for the
    table's adapter blocks: digests equal to the blocks', the zero adapter's
    ``petri_intervention_hash`` None (the loaded config carries none)."""
    out = {}
    for i, (name, block) in enumerate(sorted(table[px.ADAPTERS_KEY].items()), start=1):
        out[name] = {"config_sha256": block["config_sha256"], "weights_sha256": block["weights_sha256"],
                     "fingerprint_bf16": f"fp-{name}", "lora_int_id": i,
                     "petri_intervention_hash": None if block["petri_zero_adapter"] else block["petri_intervention_hash"],
                     "petri_zero_adapter": bool(block["petri_zero_adapter"]),
                     # native-psr-serving (attestation schema 2): the kind and the residual identity
                     "petri_source_kind": block.get("petri_source_kind"),
                     "residual_fingerprint": None, "position_mode": None}
    return out


def write_native_endpoint(table_path: Path, table: dict, *, geometry: dict | None = None,
                          cells_skipped=(), **extra) -> Path:
    """The native core's ``endpoint.json`` beside the table — the key set of
    ``LoraStandup.endpoint()`` (capctrl's / mt's common keys, the lora-native
    additions, ``_common_facts``): ``engine`` AND ``serving_engine`` =
    ``lora_native``, ``reduction_mode`` = ``not_applicable`` and NO
    ``lens_reduction_mode``, ``cells_skipped``, the §1.8 geometry block, the
    attested engine state, the adapter summary. ``**extra`` overrides / adds
    (pass ``LENS.ABSENT`` to drop a key)."""
    geometry = geometry if geometry is not None else native_geometry()
    routes = list(table["routes"])
    blocks = table[px.ADAPTERS_KEY]
    adapters = engine_adapters(table)
    harness_cells = [r.cell_id for r in routes if r.control_of is None]
    ep = {
        "proxy_url": PROXY_URL, "proxy_port": 1, "vllm_port": 2, "routing_table": str(table_path),
        "mode": "attacks", "eval": None, "model": BASE_MODEL, "model_key": "llama8b",
        "model_revision": REV, "revision": REV,
        "cell": HARNESS_CELL, "harness_cell": HARNESS_CELL,
        "cells": harness_cells, "harness_cells": harness_cells, "cells_requested": harness_cells,
        "served_models": [r.virtual_model for r in routes],
        "pid": 0, "node": "TESTHOST", "slurm_job_id": None,
        "max_model_len": geometry["max_model_len"], "max_num_seqs": geometry["max_num_seqs"],
        "seqs_source": geometry["seqs_source"],
        "attested_max_num_seqs": geometry["max_num_seqs"], "attested_max_model_len": geometry["max_model_len"],
        "geometry": geometry,
        "geometry_override_reason": (geometry[sg.ADJUSTMENTS_KEY][0]["kind"] if geometry.get(sg.ADJUSTMENTS_KEY) else None),
        "geometry_lowered": None, sa.LORA_KERNEL_CEILING_KEY: None,
        "tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True, tc.PROXY_IDENTITY_AGENT_LOOP_ENTRY: True,
        "proxy_token": None, "proxy_upstream_timeout_s": None,
        srun.STANDUP_TEMPLATE_HASH_KEY: TEMPLATE_HASH, "chat_template_sha256": "cd" * 32,
        tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: OBJECT_SHA,
        tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {HARNESS_CELL: OBJECT_SHA},
        "max_logprobs": 20, "written_at": "2026-09-13T00:00:00+00:00", "started": "2026-09-13T00:00:00+00:00",
        # the lora-native additions
        tc.ENDPOINT_KEY_ENGINE: NATIVE, "attempt_id": "attempt-1111",
        "serving_attestation": str(table_path.parent / "serving_attestation.json"),
        "cells_served": str(table_path.parent / "cells_served.json"),
        "positive_controls": str(table_path.parent / "positive_controls.json"),
        "adapter_bank": "/bank",
        "adapters": {k: dict(v) for k, v in blocks.items()}, "zero_adapter": dict(blocks[ZERO]),
        "adapters_reused": {},
        "attested_enforce_eager": False, "attested_cudagraph_mode": "FULL_AND_PIECEWISE",
        "attested_model_runner": None, "lora_config": {"max_loras": 2, "max_lora_rank": 1},
        "id_name_map": {str(v["lora_int_id"]): k for k, v in adapters.items()},
        "vmodels": {r.cell_id: r.virtual_model for r in routes},
        "lora_cells": [HARNESS_CELL], "prompted_cells_admitted": [PROMPTED],
        "manifest_flags_dropped": ["--enforce-eager"], "routing_table_version": 2,
        "probe_max_tokens": 48, "thinking_header": None, "thinking_header_error": "no thinking-header fact recorded",
        # _common_facts
        tc.SERVED_ENGINE_FIELD: NATIVE,
        srun.STANDUP_CELLS_SKIPPED_KEY: [dict(s) for s in cells_skipped],
        "thinking_mode": "off", "thinking_declaration_key": "petri_thinking_on",
        "thinking_certification": None, "thinking_declared_cells": [],
        "default_chat_template_kwargs": None, "override_generation_config": None,
        "reasoning_parser": None, "upstream_timeout_s": 600.0,
        "text_only_mm": None, "text_only_mm_record": None, "enable_log_requests": False,
        "mrope_attestation": None, "mrope_attestation_error": None,
        "mamba_backend_attestation": None, "mamba_backend_attestation_error": None,
        "kv_shrink": None, "expected_digests_source": "cli", "zero_adapter_only": False,
        tc.REDUCTION_MODE_FIELD: tc.REDUCTION_MODE_NOT_APPLICABLE,
        "lens_salt_policy": tc.NOT_APPLICABLE, "require_attested_mode": None, "standup_front": "capctrl",
        **extra}
    ep = {k: v for k, v in ep.items() if v is not LENS.ABSENT}
    assert tc.PROXY_IDENTITY_REDUCTION_MODE not in ep or extra.get(tc.PROXY_IDENTITY_REDUCTION_MODE) is not None
    path = table_path.parent / srun.STANDUP_ENDPOINT_FILENAME
    path.write_text(json.dumps(ep, indent=2))
    return path


def native_live_identity(table: dict) -> dict:
    """The live ``/v1/models`` ``petri_proxy`` block a NATIVE proxy publishes
    (``RoutingProxy.models_payload``): the common facts, NO
    ``lens_reduction_mode`` (nothing attested a lens mode), and the ``engine``
    block ``proxy.engine_payload`` echoes from the attestation."""
    return {"version": 1, "serving_mode": "interleaved", "exact_replay_strategy": None,
            "base_model": BASE_MODEL, tc.PROXY_IDENTITY_AGENT_LOOP_ENTRY: True,
            "agent_loop_render": {"mode": "explicit", "loaded": True, "source": "T"},
            "thinking_mode": "off", "thinking_declaration_key": "petri_thinking_on",
            tc.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {HARNESS_CELL: OBJECT_SHA},
            px.ENGINE_KEY: {"kind": px.ENGINE_KIND_LORA_NATIVE, "enforce_eager": False,
                            "cudagraph_mode": "FULL_AND_PIECEWISE", "adapters": engine_adapters(table)}}


def native_stamps(payload) -> dict:
    """The REAL triple the proxy stamps on every NON-LENS leg (a LoRA adapter or
    the zero adapter; ``RoutingProxy.salt_regime_for(lens=False)``): the
    cell-stable salt regime, the not-consulted policy, and the render regime
    off the validated ``chat_template_kwargs`` (absent -> unspecified)."""
    kwargs = payload.get(tc.CHAT_TEMPLATE_KWARGS_KEY) or {}
    flag = kwargs.get(tc.ENABLE_THINKING_KEY) if isinstance(kwargs, dict) else None
    render = (tc.RENDER_REGIME_THINKING_ON if flag is True
              else tc.RENDER_REGIME_THINKING_OFF if flag is False
              else tc.RENDER_REGIME_UNSPECIFIED)
    return {tc.STAMP_KEY_SALT_REGIME: SALT_CELL_STABLE,
            tc.STAMP_KEY_RESUMED_POLICY: POLICY_NOT_CONSULTED,
            tc.STAMP_KEY_RENDER_REGIME: render}


class NativeFakeChat(LENS.FakeChat):
    """The lens world's scripted proxy client on a NATIVE standup: every vmodel
    it answers for is a non-lens leg, so every response wears the REAL non-lens
    triple, and — the proxy's own refusal matrix — a body carrying the lens
    re-location declaration (``petri_response_prefill``) is a 400
    ``response_prefill_not_lens`` BEFORE any forward. The world asserts that
    refusal is NEVER reached: a lora route sends the seeded body without the key."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.refusals = []

    def regime_stamps(self, payload) -> dict:
        return native_stamps(payload)

    def chat_completions(self, payload, *, headers=None):
        if payload.get(px.RESPONSE_PREFILL_KEY) is not None:
            self.refusals.append(json.loads(json.dumps(payload)))
            raise px.ProxyError("proxy /v1/chat/completions returned 400: " + json.dumps(
                {"error": {"message": f"{px.RESPONSE_PREFILL_KEY} on a non-lens route",
                           "type": "invalid_request_error", "code": RESPONSE_PREFILL_NOT_LENS}}))
        return super().chat_completions(payload, headers=headers)


@pytest.fixture()
def native_world(tmp_path, monkeypatch):
    """The lens world's offline machinery over the NATIVE records: the v2 table,
    the native ``endpoint.json`` + ledger record, the native live identity, a
    scripted poster that refuses the lens declaration key, and the native fake
    chat (steered voice on the lora route, plain voice on ``none__lora0``)."""
    cfg_path = _write_cfg(tmp_path)
    table_path, table = native_routing_table(tmp_path)
    write_native_endpoint(table_path, table)
    LENS.write_ledger_endpoint(table_path)
    routes = {r.cell_id: r for r in table["routes"]}
    route, twin = routes[HARNESS_CELL], routes[ZERO]
    assert set(routes) == {"none", PROMPTED, HARNESS_CELL, ZERO}, sorted(routes)   # no lens leg, no none__lens0

    monkeypatch.setattr(srun, "_xstest_rows", lambda pins: [dict(r) for r in LENS.XST_ROWS])
    monkeypatch.setattr(srun, "_orbench_rows", lambda pins: [dict(r) for r in LENS.ORB_ROWS])
    # the seam takes the width keyword since mmlu-full-1531 (`full=`); this world serves the default head
    monkeypatch.setattr(srun, "_mmlu_rows", lambda pins, full=False: [dict(r) for r in LENS.MMLU_ROWS])
    monkeypatch.setattr(srun, "_load_served_tok", lambda mid, rev: LENS.FakeServedTok())
    # the LENS expectation switch is set the way every launcher sets it — on a
    # native record it is RECORDED not consulted, so its value cannot matter;
    # the world sets it to the value that would REFUSE a keyless lens record
    # (stacked) to prove the native branch never reads it
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, tc.REDUCTION_MODE_STACKED)
    monkeypatch.setitem(sg.MODEL_HF_IDS, "llama8b", BASE_MODEL)

    posted, refused = [], []

    import contextlib

    @contextlib.asynccontextmanager
    async def fake_poster(proxy_url):
        async def post(req):
            posted.append(json.loads(json.dumps(req)))
            if req.get(px.RESPONSE_PREFILL_KEY) is not None:
                refused.append(json.loads(json.dumps(req)))
                raise px.ProxyError("proxy /v1/chat/completions returned 400: " + json.dumps(
                    {"error": {"code": RESPONSE_PREFILL_NOT_LENS}}))
            if req.get("continue_final_message"):
                user = next(m["content"] for m in req["messages"] if m["role"] == "user")
                return LENS._logprob_response(LENS._canned_letter_logprobs(
                    req["model"] == route.virtual_model, user))
            user = req["messages"][-1]["content"]
            return {"choices": [{"message": {"content": LENS._canned_completion(user)},
                                 "finish_reason": "stop"}]}
        yield post

    monkeypatch.setattr(srun, "_proxy_poster", fake_poster)
    chat = NativeFakeChat({route.virtual_model: "steered voice says alpha",
                           twin.virtual_model: "plain voice says beta",
                           routes["none"].virtual_model: "plain voice says beta"},
                          steered_model=route.virtual_model)
    live = native_live_identity(table)
    monkeypatch.setattr(srun, "_make_client",
                        lambda url, rt: SimpleNamespace(chat_completions=chat.chat_completions,
                                                        tokenize=LENS.live_tokenize,
                                                        proxy_identity=copy.deepcopy(live)))
    monkeypatch.setattr(srun, "_tok_evidence", lambda mid, rev: {
        "served_template_hash": TEMPLATE_HASH, "serve_header_ids": [128006, 78191, 128007, 271],
        "serve_header_len": 4})
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw:
                        SimpleNamespace(model_id=BASE_MODEL, revision=REV, trained_template_hash=TEMPLATE_HASH,
                                        object_sha="ab" * 8, status="verified", source="fit_provenance.json",
                                        positions_at_fit="all"))
    out_root = tmp_path / "out"
    argv = ["generate", "--proxy-url", PROXY_URL, "--routing-table", str(table_path),
            "--cell", TT_CELL, "--config", str(cfg_path), "--defenses-dir", str(tmp_path / "tt_cells"),
            "--out-root", str(out_root), "--controls", LENS.LEGACY_GEN_CONTROLS,
            "--concurrency", "4", "--greedy"]
    return SimpleNamespace(argv=argv, out_root=out_root, posted=posted, refused=refused, route=route,
                           twin=twin, routes=routes, chat=chat, table_path=table_path, table=table,
                           live=live, cfg_path=cfg_path, tmp_path=tmp_path)


def _cell_dir(out_root: Path) -> Path:
    cells = sorted((out_root / "cells").iterdir())
    assert len(cells) == 1, cells
    return cells[0]


def _manifest(world) -> dict:
    return json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())


def _evidence(world) -> dict:
    return json.loads((world.out_root / "evidence.json").read_text())


def _endpoint(world) -> dict:
    return json.loads((world.table_path.parent / srun.STANDUP_ENDPOINT_FILENAME).read_text())


# --------------------------------------------------------------------------- #
# the full generate pass on the lora route                                     #
# --------------------------------------------------------------------------- #


def test_native_generate_end_to_end(native_world, capsys):
    w = native_world
    srun.main(w.argv)
    manifest, evidence = _manifest(w), _evidence(w)

    # the lora route is STEERED: the bites proof ran, verified, against the
    # engine's own zero cell — never the lens twin
    bites = evidence["steering_bites"]
    assert bites["applicable"] is True and bites["verified"] is True
    assert bites["twin"] == ZERO == tc.twin_cell_id_for(w.route)
    assert bites["n_differing"] == 5
    assert evidence["salt_isolation"]["ok"] is True
    # the engine stamp: serving path lora, NO lens mechanism (decode_mechanism /
    # mask are lens facts, None on a native adapter)
    assert evidence["engine_stamp"]["serving_path"] == "lora"
    assert evidence["engine_stamp"]["decode_mechanism"] is None
    assert evidence["mask"] is None
    assert evidence["serving_identity"] == LORA_ID
    # identity, new rows only: a native adapter applies at every token
    assert manifest["steer_positions"] == "all"
    assert manifest["serving_path"] == "lora"
    # the ENGINE on the manifest (unhashed), the hashed identity and the row constants
    assert manifest[tc.SERVED_ENGINE_FIELD] == NATIVE
    assert manifest["resolved_config"]["serving"][tc.SERVED_ENGINE_FIELD] == NATIVE
    assert manifest["resolved_config"]["serving"]["serving_identity"] == LORA_ID
    # the ROW schema (store.ResultRow) has no engine-name column: rows are keyed by
    # engine through harness_config_hash (the hashed identity above), the manifest
    # carries the readable stamp; the template is a valid ResultRow prefix
    assert tc.SERVED_ENGINE_FIELD not in manifest["row_constants"]
    assert manifest["row_constants"]["engine"] == "vllm"
    assert manifest["row_constants"]["harness_config_hash"] == manifest["harness_config_hash"]
    # the MMLU declaration bit stays the lens fact: False on a lora route
    assert manifest["resolved_config"]["mmlu"]["response_prefill"] is False
    # the proxy's refusal was NEVER reached: no request carried the lens key
    assert w.chat.refusals == [] and w.refused == []
    for call in w.chat.calls:
        assert px.RESPONSE_PREFILL_KEY not in call and "cache_salt" not in call
    for req in w.posted:
        assert px.RESPONSE_PREFILL_KEY not in req
        assert req["model"] == w.route.virtual_model
    # the report line names the steered route's twin
    out = capsys.readouterr().out
    assert f"twin={ZERO}" in out and "n/a (" not in out.split("steering bites:")[-1].splitlines()[0]
    # the legacy controls path (J3's leg: no tinycap atoms) says which engine it read, and where from
    assert f"{tc.SERVED_ENGINE_FIELD}={NATIVE!r} (read off standup record" in out and "unstated" in out
    # ... and NEVER consulted the lens expectation switch (set to the value that would
    # refuse a keyless lens record) — the run got this far
    assert tc.LENS_STACKED_REDUCTIONS_ENV not in out


def test_native_mmlu_end_to_end(native_world):
    """The MMLU decision: the seeded teacher-forced read is IDENTICAL
    on a lora route MINUS the lens re-location declaration key; the logprob
    bites run against ``none__lora0`` with the key absent from every request."""
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU
    w = native_world
    argv = [a if a != LENS.LEGACY_GEN_CONTROLS else "mmlu" for a in w.argv]
    srun.main(argv)
    cell_dir = _cell_dir(w.out_root)

    scoring = [r for r in w.posted if r.get("continue_final_message")]
    assert len(scoring) == len(LENS.MMLU_ROWS)
    for req in scoring:
        assert px.RESPONSE_PREFILL_KEY not in req
        assert req["model"] == w.route.virtual_model
        assert [m["role"] for m in req["messages"]] == ["user", "assistant"]
        assert req["messages"][-1]["content"] == srun.MMLU_ANSWER_SEED
        assert req["add_generation_prompt"] is False
        assert req["max_tokens"] == 1 and req["logprobs"] is True
        # the lora payload == the lens payload minus the ONE key
        user = req["messages"][0]["content"]
        lens_payload = srun._mmlu_scoring_payload(w.route.virtual_model, user, system=None,
                                                  top_logprobs=srun.MMLU_TOP_LOGPROBS, response_prefill=True)
        assert px.RESPONSE_PREFILL_KEY in lens_payload
        assert req == {k: v for k, v in lens_payload.items() if k != px.RESPONSE_PREFILL_KEY}
    assert w.refused == [] and w.chat.refusals == []

    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert manifest["resolved_config"]["mmlu"] == {"answer_seed": srun.MMLU_ANSWER_SEED,
                                                    "top_logprobs": srun.MMLU_TOP_LOGPROBS,
                                                    "response_prefill": False}
    rows = [json.loads(l) for l in (cell_dir / "rows.jsonl").read_text().splitlines() if l.strip()]
    assert rows and {r["engine"] for r in rows} == {"vllm"}
    # every row is keyed by the engine-bearing identity hash (the manifest's)
    assert {r["harness_config_hash"] for r in rows} == {manifest["harness_config_hash"]}
    assert manifest["resolved_config"]["serving"][tc.SERVED_ENGINE_FIELD] == NATIVE
    acc = {r["item_id"]: r for r in rows if r["task"] == PARITY_TASK_MMLU and r["metric"] == "acc"
           and r["item_id"] != "__aggregate__"}
    assert acc["m0000-dddddddd"]["value"] == 1.0 and acc["m0001-eeeeeeee"]["value"] == 0.0

    evidence = _evidence(w)
    lp = evidence["logprob_steering_bites"]
    assert lp["applicable"] and lp["verified"] is True and lp["twin"] == ZERO
    assert evidence["mmlu_scoring"] == {"route_path": "lora", "declaration_sent": False,
                                        "steering_at_scored_position": "adapter, all positions"}
    probe_calls = [c for c in w.chat.calls if c.get("continue_final_message")]
    assert len(probe_calls) == 3 * len(LENS.MMLU_ROWS)
    assert all(px.RESPONSE_PREFILL_KEY not in c for c in probe_calls)
    assert len([c for c in probe_calls if c["model"] == w.twin.virtual_model]) == 2 * len(LENS.MMLU_ROWS)


def test_native_unsteered_none_cell_records_no_bites_and_names_the_route(native_world, capsys):
    """The ``none`` cell on the SAME native table (a passthrough leg): no
    steering to prove — bites ``applicable False`` naming an UNSTEERED route
    (not "not a lens route"), the engine still stamped."""
    w = native_world
    argv = [a if a != TT_CELL else "none" for a in w.argv]
    srun.main(argv)
    evidence, manifest = _evidence(w), _manifest(w)
    assert evidence["steering_bites"]["applicable"] is False
    assert "unsteered route" in evidence["steering_bites"]["reason"]
    assert "passthrough" in evidence["steering_bites"]["reason"]
    assert evidence["engine_stamp"]["serving_path"] == "passthrough"
    assert manifest[tc.SERVED_ENGINE_FIELD] == NATIVE
    assert manifest["resolved_config"]["serving"][tc.SERVED_ENGINE_FIELD] == NATIVE
    assert "steer_positions" not in manifest["resolved_config"]
    assert "n/a (unsteered route)" in capsys.readouterr().out


def test_a_lens_run_never_gains_the_engine_key(world):
    """The LENS world (imported): the manifest, identity and rows carry NO
    ``serving_engine`` — the byte-identity the stored-record consumers rely on
    (absent = lens by era)."""
    srun.main(world.argv)
    cell_dir = LENS._cell_dir(world.out_root)
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert tc.SERVED_ENGINE_FIELD not in manifest
    assert tc.SERVED_ENGINE_FIELD not in manifest["resolved_config"]["serving"]
    assert tc.SERVED_ENGINE_FIELD not in manifest["row_constants"]
    assert manifest["steer_positions"] == "response"
    evidence = json.loads((world.out_root / "evidence.json").read_text())
    assert evidence["steering_bites"]["twin"] == tc.TWIN_CELL_ID


# the lens world's fixture, re-exported so the test above can take it
world = LENS.world


# --------------------------------------------------------------------------- #
# the hashed identity                                                          #
# --------------------------------------------------------------------------- #


def test_served_identity_carries_the_engine_iff_native(native_world):
    ep = _endpoint(native_world)
    ident = srun._served_identity(ep, concurrency=4)
    assert tuple(ident) == tc.served_identity_fields(native=True)
    assert ident[tc.SERVED_ENGINE_FIELD] == NATIVE
    assert ident[tc.REDUCTION_MODE_FIELD] == tc.REDUCTION_MODE_NOT_APPLICABLE
    assert ident["max_num_seqs"] == NATIVE_SEQS and ident["max_model_len"] == NATIVE_WINDOW
    # the LENS record (the lens world's shape): the key set is today's, by identity
    lens_ep = LENS._ep()
    lens_ident = srun._served_identity(lens_ep, concurrency=4)
    assert tuple(lens_ident) == tc.served_identity_fields() == tc.SERVED_IDENTITY_FIELDS
    assert tc.SERVED_ENGINE_FIELD not in lens_ident
    assert lens_ident[tc.REDUCTION_MODE_FIELD] == tc.REDUCTION_MODE_PER_ROW


def test_resolved_config_hash_is_byte_identical_for_lens_and_gains_the_engine_for_native(native_world):
    from sisteer.capeval import store as cs
    w = native_world
    cell = SimpleNamespace(id=TT_CELL)
    cfg = yaml.safe_load(Path(w.cfg_path).read_text())
    kw = dict(system=None, limit=None, judge_model="judge", route=w.route, model_id=BASE_MODEL,
              model_revision=REV)
    unstated = srun._resolved_config(cell, cfg, *[kw.pop(k) for k in ("system", "limit", "judge_model")], **kw)
    kw2 = dict(route=w.route, model_id=BASE_MODEL, model_revision=REV)
    lens = srun._resolved_config(cell, cfg, None, None, "judge", serving_engine=tc.SERVING_ENGINE_LENS, **kw2)
    native = srun._resolved_config(cell, cfg, None, None, "judge", serving_engine=NATIVE, **kw2)
    assert unstated == lens and cs.harness_config_hash(unstated) == cs.harness_config_hash(lens)
    assert tc.SERVED_ENGINE_FIELD not in lens["serving"]
    assert native["serving"][tc.SERVED_ENGINE_FIELD] == NATIVE
    assert cs.harness_config_hash(native) != cs.harness_config_hash(lens)
    with pytest.raises(ValueError, match="serving_engine 'hf' is not one of"):
        srun._resolved_config(cell, cfg, None, None, "judge", serving_engine="hf", **kw2)


def test_manifest_serving_identity_guard_refuses_a_native_resume_onto_a_lens_store(native_world, tmp_path):
    ep = _endpoint(native_world)
    store = SimpleNamespace(manifest_path=tmp_path / "manifest.json", cell_dir=tmp_path)
    check = srun._check_manifest_serving_identity
    check(store, ep, concurrency=4, pass_no=None)                                  # fresh: fine
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": None},
         "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True,
                   tc.REDUCTION_MODE_FIELD: tc.REDUCTION_MODE_PER_ROW},              # a LENS store
         "geometry": {"max_model_len": NATIVE_WINDOW, "max_num_seqs": NATIVE_SEQS},
         "concurrency": 4}))
    with pytest.raises(srun.ServingIdentityMismatch) as ei:
        check(store, ep, concurrency=4, pass_no=None)
    msg = str(ei.value)
    assert tc.SERVED_ENGINE_FIELD in msg and "serving engine" in msg
    assert f"{tc.REDUCTION_MODE_FIELD}: the cell manifest records 'per_row'" in msg
    # the same store written by a native run resumes
    store.manifest_path.write_text(json.dumps(
        {"thinking": {"reasoning_parser": None},
         "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True,
                   tc.REDUCTION_MODE_FIELD: tc.REDUCTION_MODE_NOT_APPLICABLE, tc.SERVED_ENGINE_FIELD: NATIVE},
         "geometry": {"max_model_len": NATIVE_WINDOW, "max_num_seqs": NATIVE_SEQS},
         "concurrency": 4}))
    check(store, ep, concurrency=4, pass_no=None)


def test_manifest_serving_identity_guard_reads_the_clamp_override_off_the_served_block_and_names_an_absent_key(native_world, tmp_path):
    """A PROTOCOL standup's identity carries `sampling_clamp_override`; the manifest records it in the hashed
    served block (resolved_config.tinycap.served), which is where the resume guard reads it — a manifest WITH it resumes, one
    WITHOUT it is a NAMED mismatch. Before the fix `have[k]` raised KeyError on the first in-container resume under a protocol
    standup (llama8b_it difference LoRA, BCB-full, acb7434) and the generated leg was lost to a traceback."""
    ep = {**_endpoint(native_world), tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE: dict(tc.SAMPLING_CLAMP_OVERRIDE)}
    store = SimpleNamespace(manifest_path=tmp_path / "manifest.json", cell_dir=tmp_path)
    check = srun._check_manifest_serving_identity
    base = {"thinking": {"reasoning_parser": None},
            "route": {"tool_calls": False, tc.AGENT_LOOP_ENTRY_KEY: True,
                      tc.REDUCTION_MODE_FIELD: tc.REDUCTION_MODE_NOT_APPLICABLE, tc.SERVED_ENGINE_FIELD: NATIVE},
            "geometry": {"max_model_len": NATIVE_WINDOW, "max_num_seqs": NATIVE_SEQS},
            "concurrency": 4}
    # the manifest a protocol run writes: the served block carries the override verbatim -> a legitimate resume
    store.manifest_path.write_text(json.dumps({**base, "resolved_config": {"tinycap": {"served": {
        tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE: dict(tc.SAMPLING_CLAMP_OVERRIDE)}}}}))
    check(store, ep, concurrency=4, pass_no=None)
    # a manifest without the key (no served block at all): a named mismatch, never a KeyError
    store.manifest_path.write_text(json.dumps(base))
    with pytest.raises(srun.ServingIdentityMismatch) as ei:
        check(store, ep, concurrency=4, pass_no=None)
    msg = str(ei.value)
    assert f"{tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE}: the cell manifest records <absent from the cell manifest>" in msg
    # a served block recording a DIFFERENT override is named with both values
    store.manifest_path.write_text(json.dumps({**base, "resolved_config": {"tinycap": {"served": {
        tc.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE: {**tc.SAMPLING_CLAMP_OVERRIDE, "to": 0.5}}}}}))
    with pytest.raises(srun.ServingIdentityMismatch, match="sampling_clamp_override: the cell manifest records"):
        check(store, ep, concurrency=4, pass_no=None)
    # and a standup WITHOUT the override (card sampling) never asks the manifest for it
    check(store, _endpoint(native_world), concurrency=4, pass_no=None)


# --------------------------------------------------------------------------- #
# _standup_endpoint — the native branch                                        #
# --------------------------------------------------------------------------- #


def test_standup_endpoint_native_branch_reads_no_lens_mode_and_holds_the_flag(native_world, capsys):
    w = native_world
    ep = srun._standup_endpoint(w.table_path, cell=TT_CELL)
    assert ep[tc.ENDPOINT_KEY_ENGINE] == NATIVE and tc.PROXY_IDENTITY_REDUCTION_MODE not in ep
    out = capsys.readouterr().out
    assert f"{tc.REDUCTION_MODE_FIELD}={tc.REDUCTION_MODE_NOT_APPLICABLE!r}" in out
    assert "0 cell(s) skipped" in out
    # the expectation held to the record: the right engine passes, the wrong one refuses by name
    srun._standup_endpoint(w.table_path, cell=TT_CELL, expect_serving_path=NATIVE)
    with pytest.raises(SystemExit, match=f"{tc.REC_SERVING_ENGINE_MISMATCH}.*expects 'lens'"):
        srun._standup_endpoint(w.table_path, cell=TT_CELL, expect_serving_path=tc.SERVING_ENGINE_LENS)


def test_standup_endpoint_native_record_stating_a_lens_mode_is_torn(native_world):
    w = native_world
    write_native_endpoint(w.table_path, w.table, **{tc.PROXY_IDENTITY_REDUCTION_MODE: tc.REDUCTION_MODE_STACKED})
    with pytest.raises(SystemExit, match="no lens hook to attest a reduction mode"):
        srun._standup_endpoint(w.table_path, cell=TT_CELL)


def test_standup_endpoint_honours_cells_skipped(native_world, capsys):
    w = native_world
    skipped = [{"cell_id": "si_safety__psr_kl_all", "kind": "obj_psr_kl_all",
                "reason": "no PEFT form (psr): skipped by name"}]
    write_native_endpoint(w.table_path, w.table, cells_skipped=skipped)
    # another cell's run proceeds, printing the skipped cell by name
    srun._standup_endpoint(w.table_path, cell=TT_CELL)
    out = capsys.readouterr().out
    assert "SKIPPED cell 'si_safety__psr_kl_all'" in out and "1 cell(s) skipped" in out
    # the skipped cell's OWN run refuses, in either spelling, naming --serving-path lens
    for spelling in ("safety_refusal__psr_kl_all", "si_safety__psr_kl_all"):
        with pytest.raises(SystemExit, match="was SKIPPED by the native standup.*--serving-path lens"):
            srun._standup_endpoint(w.table_path, cell=spelling)
    # a malformed claim is refused, never read around
    path = write_native_endpoint(w.table_path, w.table)
    ep = json.loads(path.read_text())
    ep[srun.STANDUP_CELLS_SKIPPED_KEY] = ["si_safety__psr_kl_all"]          # bare ids, not records
    path.write_text(json.dumps(ep))
    with pytest.raises(SystemExit, match="not a list of .cell_id, kind, reason. records"):
        srun._standup_endpoint(w.table_path, cell=TT_CELL)


def test_a_lens_record_refuses_a_native_expectation_and_stays_byte_identical(tmp_path, monkeypatch):
    table_path, _table = LENS._routing_table(tmp_path)
    LENS.write_standup_endpoint(table_path)
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, tc.REDUCTION_MODE_PER_ROW)
    before = json.loads((table_path.parent / srun.STANDUP_ENDPOINT_FILENAME).read_text())
    ep = srun._standup_endpoint(table_path)
    assert ep == before and tc.ENDPOINT_KEY_ENGINE not in ep
    srun._standup_endpoint(table_path, expect_serving_path=tc.SERVING_ENGINE_LENS)
    with pytest.raises(SystemExit, match=f"{tc.REC_SERVING_ENGINE_MISMATCH}.*attests engine='lens'"):
        srun._standup_endpoint(table_path, expect_serving_path=NATIVE)


def test_run_serving_engine_reads_the_record_beside_the_table_or_lens_by_era(native_world, tmp_path):
    w = native_world
    args = SimpleNamespace(routing_table=str(w.table_path), serving_path=None)
    assert srun._run_serving_engine(args, None) == NATIVE
    assert srun._run_serving_engine(SimpleNamespace(routing_table=str(w.table_path), serving_path=NATIVE), None) == NATIVE
    with pytest.raises(SystemExit, match=tc.REC_SERVING_ENGINE_MISMATCH):
        srun._run_serving_engine(SimpleNamespace(routing_table=str(w.table_path), serving_path="lens"), None)
    # no record at all: lens by era — unless an expectation was stated, which then has nothing to hold to
    bare = tmp_path / "bare" / "routing_table.json"
    bare.parent.mkdir()
    bare.write_text("{}")
    assert srun._run_serving_engine(SimpleNamespace(routing_table=str(bare), serving_path=None), None) == "lens"
    with pytest.raises(SystemExit, match="no standup record at"):
        srun._run_serving_engine(SimpleNamespace(routing_table=str(bare), serving_path=NATIVE), None)


# --------------------------------------------------------------------------- #
# the route-aware live-proxy compare                                           #
# --------------------------------------------------------------------------- #


def _client(ident):
    return SimpleNamespace(proxy_identity=copy.deepcopy(ident))


def _served(ep):
    return srun._served_identity(ep, concurrency=4)


def test_live_proxy_identity_binds_a_lora_route_and_both_passthrough_legs(native_world):
    w, ep = native_world, _endpoint(native_world)
    served = _served(ep)
    for cell in (HARNESS_CELL, "none", PROMPTED):
        got = srun._check_live_proxy_identity(_client(w.live), ep, served, route=w.routes[cell], table=w.table)
        assert got[px.ENGINE_KEY]["kind"] == NATIVE
    # the binding itself, by leg: the lora route + its zero twin; a passthrough leg alone
    held = srun._hold_native_route_binding(w.live, route=w.routes[HARNESS_CELL], table=w.table)
    assert [(l["role"], l["cell_id"], l["path"]) for l in held["legs"]] == [
        ("route", HARNESS_CELL, "lora"), ("twin", ZERO, "lora")]
    held = srun._hold_native_route_binding(w.live, route=w.routes[PROMPTED], table=w.table)
    assert [(l["role"], l["cell_id"], l["path"]) for l in held["legs"]] == [("route", PROMPTED, "passthrough")]
    # without the table a native run's route cannot be bound — refused, never skipped
    with pytest.raises(srun.EndpointBindingMismatch, match="needs the loaded routing table"):
        srun._check_live_proxy_identity(_client(w.live), ep, served, route=w.routes[HARNESS_CELL])


@pytest.mark.parametrize("mutate, needle", [
    (lambda live, w: live[px.ENGINE_KEY]["adapters"][HARNESS_CELL].__setitem__("petri_intervention_hash", "9" * 16),
     "intervention_hash"),
    (lambda live, w: live[px.ENGINE_KEY]["adapters"][HARNESS_CELL].__setitem__("weights_sha256", "f" * 64),
     "weights_sha256: table"),
    (lambda live, w: live[px.ENGINE_KEY]["adapters"].pop(HARNESS_CELL), "no live adapter of that name"),
    (lambda live, w: live[px.ENGINE_KEY]["adapters"].pop(ZERO), f"lora twin {ZERO!r}: no live adapter"),
    (lambda live, w: live.__setitem__("base_model", "someone/else"), "base_model: live 'someone/else'"),
])
def test_live_proxy_identity_refuses_a_lora_route_the_inventory_does_not_bind(native_world, mutate, needle):
    w, ep = native_world, _endpoint(native_world)
    live = copy.deepcopy(w.live)
    mutate(live, w)
    with pytest.raises(srun.EndpointBindingMismatch, match=needle):
        srun._check_live_proxy_identity(_client(live), ep, _served(ep), route=w.routes[HARNESS_CELL], table=w.table)


def test_live_proxy_identity_refuses_an_adapter_under_a_passthrough_cells_name(native_world):
    w, ep = native_world, _endpoint(native_world)
    live = copy.deepcopy(w.live)
    live[px.ENGINE_KEY]["adapters"]["none"] = {**_digests("none"), "fingerprint_bf16": "fp-none", "lora_int_id": 9,
                                              "petri_intervention_hash": "c" * 16, "petri_zero_adapter": False}
    with pytest.raises(srun.EndpointBindingMismatch, match="passthrough route 'none': the live proxy serves an ADAPTER"):
        srun._check_live_proxy_identity(_client(live), ep, _served(ep), route=w.routes["none"], table=w.table)


def test_live_proxy_identity_binds_a_prompted_prefix_to_the_tables_spec(native_world):
    w, ep = native_world, _endpoint(native_world)
    table = dict(w.table)
    table["prompted_prefixes"] = tuple(
        px.PromptedPrefixSpec(virtual_model=s.virtual_model, byte_length=s.byte_length, sha16="f" * 16)
        for s in w.table["prompted_prefixes"])
    with pytest.raises(srun.EndpointBindingMismatch, match="si_prefix_hash .* != the table's prompted_prefixes"):
        srun._check_live_proxy_identity(_client(w.live), ep, _served(ep), route=w.routes[PROMPTED], table=table)
    table["prompted_prefixes"] = ()
    with pytest.raises(srun.EndpointBindingMismatch, match="has no prompted_prefixes spec"):
        srun._check_live_proxy_identity(_client(w.live), ep, _served(ep), route=w.routes[PROMPTED], table=table)


def test_live_proxy_identity_refuses_the_wrong_engine_in_both_directions(native_world, tmp_path, monkeypatch):
    w, ep = native_world, _endpoint(native_world)
    # a LENS proxy (no engine block) answering for a native record
    with pytest.raises(srun.EndpointBindingMismatch, match=f"{tc.REC_SERVING_ENGINE_MISMATCH}.*engine.kind is None"):
        srun._check_live_proxy_identity(_client(LENS.LIVE_IDENTITY), ep, _served(ep), route=w.routes[HARNESS_CELL],
                                        table=w.table)
    # a native proxy that ALSO attests a lens mode is torn
    torn = copy.deepcopy(w.live)
    torn[tc.PROXY_IDENTITY_REDUCTION_MODE] = tc.REDUCTION_MODE_STACKED
    with pytest.raises(srun.EndpointBindingMismatch, match="torn identity"):
        srun._check_live_proxy_identity(_client(torn), ep, _served(ep), route=w.routes[HARNESS_CELL], table=w.table)
    # a NATIVE proxy answering for a lens record
    table_path, table = LENS._routing_table(tmp_path)
    LENS.write_standup_endpoint(table_path)
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, tc.REDUCTION_MODE_PER_ROW)
    lens_ep = srun._standup_endpoint(table_path)
    lens_served = srun._served_identity(lens_ep, concurrency=4)
    with pytest.raises(srun.EndpointBindingMismatch, match=f"{tc.REC_SERVING_ENGINE_MISMATCH}.*attests engine=None"):
        srun._check_live_proxy_identity(_client(w.live), lens_ep, lens_served)
    # ... and the lens compare on a lens record runs exactly as before (route/table given or not)
    lens_route = {r.cell_id: r for r in table["routes"]}["si_safety__psr_kl_all"]
    got = srun._check_live_proxy_identity(_client(LENS.LIVE_IDENTITY), lens_ep, lens_served, route=lens_route, table=table)
    assert got == LENS.LIVE_IDENTITY


# --------------------------------------------------------------------------- #
# the probe's stamps, per LEG's engine                                         #
# --------------------------------------------------------------------------- #


#: the rows in the spelling the probe hands the judge (``usr._served_probe_fn``:
#: ``tc.stamp_regimes(resp)._asdict()`` — StampRegimes' field names), written out
#: as LITERAL values so the test states what a native leg wears rather than
#: deriving it from the function under test
NATIVE_TRIPLE = {"salt_regime": SALT_CELL_STABLE, "resumed_policy": POLICY_NOT_CONSULTED,
                 "render_regime": tc.RENDER_REGIME_THINKING_OFF}
LENS_TRIPLE = {"salt_regime": tc.SALT_REGIME_REQUEST_UNIQUE, "resumed_policy": tc.RESUMED_PREFILL_CONDITIONAL,
               "render_regime": tc.RENDER_REGIME_THINKING_OFF}
STEERED, SCALE0 = "steered", "scale0"


def test_probe_stamps_verify_under_each_legs_engine():
    import unified_sweep_run as usr
    assert (usr.PROBE_LEG_STEERED, usr.PROBE_LEG_SCALE0) == (STEERED, SCALE0)
    # the literal rows ARE the contract's rows (pinned, so a vocabulary move is loud)
    assert NATIVE_TRIPLE == tc.expected_stamp_regimes(lens=False, arm=tc.THINKING_OFF,
                                                      policy=POLICY_NOT_CONSULTED)._asdict()
    assert LENS_TRIPLE == tc.expected_stamp_regimes(lens=True, arm=tc.THINKING_OFF,
                                                    policy=tc.RESUMED_PREFILL_CONDITIONAL)._asdict()
    native_policies = {STEERED: POLICY_NOT_CONSULTED, SCALE0: POLICY_NOT_CONSULTED}
    # a native leg wearing the REAL present triple verifies
    assert srun._probe_stamp_problem({STEERED: [NATIVE_TRIPLE] * 3, SCALE0: [NATIVE_TRIPLE] * 3},
                                     tc.THINKING_OFF, out_of_support=False, policies=native_policies,
                                     lens_by_leg={STEERED: False, SCALE0: False}) is None
    # a lens leg is unchanged: lens_by_leg None (every pre-native caller) reads every leg as lens
    lens_policies = {STEERED: tc.RESUMED_PREFILL_CONDITIONAL, SCALE0: tc.RESUMED_PREFILL_CONDITIONAL}
    assert srun._probe_stamp_problem({STEERED: [LENS_TRIPLE], SCALE0: [LENS_TRIPLE]}, tc.THINKING_OFF,
                                     out_of_support=False, policies=lens_policies) is None
    assert srun._probe_stamp_problem({STEERED: [LENS_TRIPLE], SCALE0: [LENS_TRIPLE]}, tc.THINKING_OFF,
                                     out_of_support=False, policies=lens_policies,
                                     lens_by_leg={STEERED: True, SCALE0: True}) is None
    # a native leg wearing a LENS triple refuses, naming the engine it was held as
    why = srun._probe_stamp_problem({STEERED: [LENS_TRIPLE], SCALE0: [NATIVE_TRIPLE]}, tc.THINKING_OFF,
                                    out_of_support=False, policies=native_policies,
                                    lens_by_leg={STEERED: False, SCALE0: False})
    assert why and "non-lens (native adapter)" in why and STEERED in why
    # a lens leg wearing the native triple refuses as before
    why = srun._probe_stamp_problem({STEERED: [NATIVE_TRIPLE], SCALE0: [LENS_TRIPLE]}, tc.THINKING_OFF,
                                    out_of_support=False, policies=lens_policies,
                                    lens_by_leg={STEERED: True, SCALE0: True})
    assert why and "for a lens leg" in why
    # a leg with no engine reading cannot be held to a row
    why = srun._probe_stamp_problem({STEERED: [NATIVE_TRIPLE], SCALE0: [NATIVE_TRIPLE]}, tc.THINKING_OFF,
                                    out_of_support=False, policies=native_policies, lens_by_leg={STEERED: False})
    assert why and "no engine reading" in why
    # a native leg whose derived policy is not `not-consulted` has no row (the contract's ValueError, named)
    why = srun._probe_stamp_problem({STEERED: [NATIVE_TRIPLE]}, tc.THINKING_OFF, out_of_support=False,
                                    policies={STEERED: tc.RESUMED_PREFILL_CONDITIONAL}, lens_by_leg={STEERED: False})
    assert why and "no expected row can be derived" in why


# --------------------------------------------------------------------------- #
# the §1.8 geometry block                                                      #
# --------------------------------------------------------------------------- #


def _kernel(model_key: str, S_identity: int, L: int) -> dict:
    step = sg.kernel_ceiling_adjustment(model_key, L, S_identity)
    assert step is not None
    return step


def test_geometry_block_copies_a_gemma_shaped_kernel_step_the_validators_accept(native_world, tmp_path):
    w = native_world
    L = ARM.L
    block = ARM._geom("gemma4e4b", served_S=107, adjustments=[_kernel("gemma4e4b", 121, L)])
    block["model_key"] = "gemma4e4b"
    write_native_endpoint(w.table_path, w.table, geometry=block)
    ep = srun._standup_endpoint(w.table_path, cell=TT_CELL)
    copied = srun._geometry_block(ep, tmp_path)
    assert copied[sg.ADJUSTMENTS_KEY] == block[sg.ADJUSTMENTS_KEY]
    assert copied["max_num_seqs"] == 107 and copied["identity"] is False
    assert copied["source"] == srun.STANDUP_ENDPOINT_FILENAME
    # the validators' rows accept the copied block (the ONE §1.8 rule)
    man = ARM._native(model_key="gemma4e4b", served_S=107, geometry={k: v for k, v in copied.items()
                                                                     if k not in (tc.GEOMETRY_PASS_FIELD, "source")})
    assert M.arm_contract_problem(man, row=ARM._row(model_key="gemma4e4b")) is None
    # the identity of the hashed served block follows the ADJUSTED S
    assert srun._served_identity(ep, concurrency=107)["max_num_seqs"] == 107


def test_geometry_block_refuses_a_bare_smaller_s_and_a_kernel_step_that_does_not_lower(native_world, tmp_path):
    w = native_world
    # a bare smaller S with no adjustment
    write_native_endpoint(w.table_path, w.table, geometry=native_geometry(max_num_seqs=NATIVE_SEQS - 1))
    ep = srun._standup_endpoint(w.table_path, cell=TT_CELL)
    with pytest.raises(SystemExit, match="does not hold the §1.8 adjustment contract.*names no adjustment"):
        srun._geometry_block(ep, tmp_path)
    # a kernel step on llama at 4864, where D does not lower
    L = ARM.L
    gem = _kernel("gemma4e4b", 121, L)
    block = ARM._geom("llama8b", served_S=gem["to_S"], adjustments=[{**gem, "from_S": 97}])
    write_native_endpoint(w.table_path, w.table, geometry=block)
    ep = srun._standup_endpoint(w.table_path, cell=TT_CELL)
    with pytest.raises(SystemExit, match="does not hold the §1.8 adjustment contract"):
        srun._geometry_block(ep, tmp_path)
    # the identity block passes, and a llama-shaped native row needs no step
    write_native_endpoint(w.table_path, w.table, geometry=native_geometry())
    ep = srun._standup_endpoint(w.table_path, cell=TT_CELL)
    assert srun._geometry_block(ep, tmp_path)[sg.ADJUSTMENTS_KEY] == []


def test_geometry_block_on_a_lens_record_is_copied_without_the_rule(tmp_path, monkeypatch):
    """The lens world's fixture serves S=4 at 4096 (an override, not the
    identity) — a LENS record is copied exactly as before; no §1.8 rule runs."""
    table_path, _ = LENS._routing_table(tmp_path)
    LENS.write_standup_endpoint(table_path)
    monkeypatch.setenv(tc.LENS_STACKED_REDUCTIONS_ENV, tc.REDUCTION_MODE_PER_ROW)
    ep = srun._standup_endpoint(table_path)
    block = srun._geometry_block(ep, tmp_path)
    assert block["max_num_seqs"] == LENS.STANDUP_SEQS and block["identity"] is False
    assert sg.ADJUSTMENTS_KEY not in block


# --------------------------------------------------------------------------- #
# the twin rule and the steered gate, on routes                                #
# --------------------------------------------------------------------------- #


def test_twin_and_steered_predicates_on_the_native_table(native_world):
    w = native_world
    assert bs.is_steered(w.route) and not bs._is_lens(w.route)
    assert not bs.is_steered(w.routes["none"]) and not bs.is_steered(w.routes[PROMPTED])
    assert pe.scale0_twin(w.route, list(w.routes.values())).cell_id == ZERO
    assert tc.twin_cell_id_for(w.route) == ZERO and tc.twin_vmodel_for(w.route) == w.twin.virtual_model
    assert pe.scale0_twin_suffix(w.route) == px.LORA_ZERO_CELL_SUFFIX
    with pytest.raises(ValueError, match="has no scale-0 twin"):
        tc.twin_cell_id_for(w.routes["none"])


# --------------------------------------------------------------------------- #
# the REAL legacy-controls manifest states its engine to the store readers,    #
# and a lens pair of it refuses                                                #
# --------------------------------------------------------------------------- #


def test_the_legacy_controls_manifest_states_its_engine_to_the_store_readers_and_a_lens_pair_refuses(native_world, tmp_path):
    """The REAL manifest of a native run with NO tinycap atoms (this world's legacy
    XSTest / OR-Bench controls) carries no `route` block, so `read_store_reduction_mode` reads "nothing stated" off it —
    the hole: pairing's cross-engine guard used that reading alone — while `read_store_serving_engine` reads `lora_native`
    off the two spellings the writer stamps (top-level and the hashed `resolved_config.serving`). Its LENS counterpart is
    the same manifest without those two keys (the writer's `iff native` rule; `test_a_lens_run_never_gains_the_engine_key`
    pins that a lens run has neither), and `pairing.compare_sides` over the two refuses as different serving engines."""
    from sisteer.capeval.tiers import pairing as P
    w = native_world
    srun.main(w.argv)
    manifest = _manifest(w)
    where = str(_cell_dir(w.out_root) / "manifest.json")
    assert tc.MANIFEST_KEY_ROUTE not in manifest and manifest[tc.SERVED_ENGINE_FIELD] == NATIVE
    assert manifest["resolved_config"]["serving"][tc.SERVED_ENGINE_FIELD] == NATIVE
    mode = tc.read_store_reduction_mode(manifest, where=where)
    assert (mode.mode, mode.refusal, mode.serving_engine) == (None, None, None) and "not a served-lens store" in mode.why
    eng = tc.read_store_serving_engine(manifest, where=where)
    assert (eng.engine, eng.stated, eng.refusal) == (NATIVE, NATIVE, None)
    assert eng.why == f"{where} {['resolved_config.serving.serving_engine', 'serving_engine']} = {NATIVE!r}"
    # the record with ONLY the two engine keys deleted is NOT a lens counterpart: its identity spellings still say
    # `+lora+`, so it is a native record torn of its stamps — reading it as lens by era is how a torn native store
    # slipped past the cross-engine guard. It refuses by name now
    torn = copy.deepcopy(manifest)
    del torn[tc.SERVED_ENGINE_FIELD]
    del torn["resolved_config"]["serving"][tc.SERVED_ENGINE_FIELD]
    assert "+lora+" in torn["serving_identity"] and torn["serving_path"] == "lora"
    torn_eng = tc.read_store_serving_engine(torn, where="<torn>")
    assert (torn_eng.engine, torn_eng.stated) == (None, None) and torn_eng.refusal and "a torn native record" in torn_eng.refusal
    # the LENS counterpart is the same manifest as the lens writer would write it: no engine key (the `iff native` rule) AND
    # the lens path in every identity spelling — read as lens BY ERA on that positive evidence
    lens = copy.deepcopy(torn)
    lens["serving_identity"] = lens["serving_identity"].replace("+lora+", "+lens+")
    lens["resolved_config"]["serving"]["serving_identity"] = lens["resolved_config"]["serving"]["serving_identity"].replace("+lora+", "+lens+")
    lens["serving_path"] = "lens"
    lens_eng = tc.read_store_serving_engine(lens, where="<lens>")
    assert (lens_eng.engine, lens_eng.stated, lens_eng.refusal) == (tc.SERVING_ENGINE_LENS, None, None) and "by era" in lens_eng.why

    def store(name, man):
        d = tmp_path / name / "cells" / "k"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(man))
        rows = [{"task": "xstest", "metric": "acc", "item_id": f"i{i}", "value": v, "engine": "vllm"}
                for i, v in enumerate((1.0, 0.0, 1.0))]
        (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return P.Side(label=name, rows_path=d / "rows.jsonl", task="xstest", engine="vllm", metric="acc")

    with pytest.raises(P.PairingError) as ei:
        P.compare_sides(store("native", manifest), store("lens", lens))
    assert "different serving engines" in str(ei.value) and "by era" in str(ei.value)
    with pytest.raises(P.PairingError, match="a torn native record"):        # the torn record never pairs, with either side
        P.compare_sides(store("torn", torn), store("lens2", lens))
    rep = P.compare_sides(store("native_a", manifest), store("native_b", manifest))
    assert rep["sides"]["a"]["serving_engine"] == NATIVE and rep["sides"]["a"]["reduction_mode"] is None
