"""Synthetic attempt roots for the Qwen seed-replicate battery's audit tables.

Every store is written in its writer's REAL row schema (static_common.run_static rows, bon_served rows + manifest,
multiturn.run rows + progress, misalign.runner RESPONSE_FIELDS + meta.json, Petri CellStore rows, tinycap CellStore
rows + outcomes + done markers, judged.GenSpool + GenMeta sidecars) with INVENTED content (goal ids `g0…`, item ids
`item0…`, placeholder shas), and every target request goes through the PRODUCTION header builder
(`proxy_engine.ledger_headers`) and `contracts.request_evidence_record` on the exact wire bytes — the usage-ledger row,
the engine-log `Received request` line, the retained body and the response record are written beside it, so the
audit's identity / coverage / sampling / seed / binding gates all run against the real shapes. Nothing here is
recorded data.
"""
from __future__ import annotations

import base64
import hashlib
from collections import Counter
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.serving import vllm_server as _VS  # noqa: E402  (the composer's capture-sizes rule)
from sisteer.capeval.serving.proxy_engine import ledger_headers  # noqa: E402
import audit_common as ac  # noqa: E402

W_FIXTURE = "f" * 40
WAVE_UUID = "11111111-2222-3333-4444-555555555555"
MODEL_REVISION = "851bf6e8"
#: the launch plan's label for the Arditi arm (`seedrep_launch_plan.ARDITI_ARM_ID`); its `cell` is `arditi_allpos`
ARDITI_PLAN_ARM = "ar_c0p3__arditi_allpos"
LAUNCH_PLAN_SCHEMA = "seedrep_launch_plan/1"
T0 = datetime(2026, 9, 18, 0, 0, 0, tzinfo=timezone.utc)
VENDOR_BODY = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
THINKING_CARD_BODY = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
CAP_OF_LEG = {**{f: 4096 for f in ac.STATIC_FAMILIES}, "bon": 4096, **{a: 4096 for a in ac.MULTITURN_ATTACKS},
              **{c: 512 for c in ac.CONTROLS}, "mmlu": 1, "gsm8k": 8192, "petri": 1024, "misalign_leaking": 3000,
              "misalign_murder": 3000}
PETRI_BEHAVIOURS = ("beh_a", "beh_b")


def synthetic_petri_seed(seed_base: int, behaviour: str, scenario_id: str, turn: int) -> int:
    """A stand-in for `petri_target_seed` (the fixture injects it through `audit_common.PETRI_SEED_RULE_OVERRIDE`)."""
    key = f"{int(seed_base)}|petri|{behaviour}|{scenario_id}|{int(turn)}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


#: `petri_target_seed` has not landed on this tree: the fixture installs its synthetic stand-in through the audit's
#: injection point so the Petri stores it writes and the audit that reads them share ONE rule (a test that wants the
#: "rule unavailable → REFUSED" path sets the override back to None)
ac.PETRI_SEED_RULE_OVERRIDE = synthetic_petri_seed


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def standup_id(leg: str, view: str, stage: str, shard: int) -> str:
    """`seedrep_launch_plan.standup_id`: `<leg>__<engine>__<stage>__shard<NN>`."""
    return f"{leg}__{view}__{stage}__shard{int(shard):02d}"


def control_key(W7: str, wave_uuid: str, leg: str, view: str, stage: str, shard: int) -> str:
    """`seedrep_launch_plan.control_key`: the six axes joined by `/` (shard as an int)."""
    return "/".join(str(x) for x in (W7, wave_uuid, leg, view, stage, int(shard)))


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def rewrite_jsonl(path: Path, fn) -> None:
    """Map `fn` over a JSONL file's rows in place (a row `fn` returns None for is DROPPED) — the one read-edit-write idiom
    the tests use to plant a synthetic defect."""
    _jsonl(path, [r for r in (fn(r) for r in ac.read_jsonl(path)) if r is not None])


def pins_file(path: Path, *, pin: str, kind: str = "vector_all") -> Path:
    """`configs/arditi_allpos_pins.json` in its real shape for the lane's ONE Arditi object."""
    _json(path, {ac.MODEL_KEY: {ac.ARDITI_STAGED_ID: {"object_sha256": pin, "kind": kind, "served_layer": 12, "c": 0.3}}})
    return path


class Root:
    """One synthetic accepted attempt root: ATTEMPT.json, the serve dir, the request pipeline files, samplers."""

    def __init__(self, path: Path, *, leg: str, view: str, arms: Sequence[str], S: int, L: int, shard: str = "shard00",
                 stage: str = "base", thinking: str = "off", W: str = W_FIXTURE, arditi_pin: Optional[str] = None,
                 max_logprobs: int = 20, template_kwargs: Optional[dict] = None, server_default_policy: str = "contract_override",
                 from_S: Optional[int] = None, harness_of: Optional[Mapping[str, str]] = None) -> None:
        self.path = path
        self.leg, self.view, self.shard, self.stage, self.thinking, self.W = leg, view, shard, stage, thinking, W
        self.S, self.L = S, L
        self.arms = list(arms)
        self.attempt_id = f"ta-{leg}-{view}-{shard}-{stage}-{uuid.uuid4().hex[:8]}"
        self.shard_index = int(str(shard)[len("shard"):]) if str(shard).startswith("shard") else int(shard)
        self.standup_id = standup_id(leg, view, stage, self.shard_index)
        self.control_key = control_key(W[:7], WAVE_UUID, leg, view, stage, self.shard_index)
        # the arm -> served HARNESS cell id map, the standup's own identity seam (`lp.HARNESS_ID_OF`;
        # `constv2-mix-full__lora_r4_ce` -> `si_constv2_mix_full__lora_r4_ce`). Identity by default (the two spellings coincide, as
        # for `none` / `arditi_allpos`). `self.vmodels` stays keyed by the ARM the store writers hand; the ENDPOINT record keys
        # `vmodels` by the harness id and spells both lists (`cells` <-> `harness_cells`), exactly as `native_standup` writes it.
        self.harness_of: Dict[str, str] = dict(harness_of or {})
        self.vmodels = {a: f"petri__{self.harness_id(a)}" for a in self.arms}
        self.records: List[dict] = []
        self.ledger: List[dict] = []
        self.engine_lines: List[str] = []
        self.responses: List[dict] = []
        self.started = T0
        self._t = T0
        path.mkdir(parents=True, exist_ok=True)
        # `seedrep_served.attempt_record`'s shape: `launch_plan_row` = the plan row's standup_id, `control_key` = the six axes
        _json(path / ac.ATTEMPT_FILE, {"schema": "seedrep_attempt/1", "attempt_id": self.attempt_id, "call_id": "fc-" + uuid.uuid4().hex[:8],
                                       "task_id": "ta-x", "W": W, "wave_uuid": WAVE_UUID, "launch_plan_row": self.standup_id,
                                       "control_key": self.control_key, "prior_attempts_seen": [], "started_utc": _iso(T0),
                                       "resumed_from": None, "superseded_by": None, "samplers_started_utc": _iso(T0)})
        # the app's declared layout (`seedrep_served.ATTEMPT_ROOT_LAYOUT`: `exp/serve/{endpoint,cells_served,serving_attestation}.json`,
        # `standup_jobs.jsonl`) — `audit_common.SERVE_DIR_REL`; the readers look HERE for the standup record (F-AE)
        serve = path / ac.SERVE_DIR_REL
        serve.mkdir(parents=True, exist_ok=True)
        # the REAL attestation shape (verified on a controls-arditi root): `adapters` keyed by the LoRA NAME the engine registered each
        # adapter under == the harness cell id (`arditi_allpos`, `none__lora0`), each with its `lora_int_id`; `id_name_map` = the engine's int id ->
        # that name. `none` and the prompted cells are served on the base weights: no adapter, routed passthrough.
        fitted = [a for a in self.arms if a != "none" and not a.startswith("prompted__")]
        adapters = {self.harness_id(a): {"fingerprint_bf16": _sha(f"fp|{a}|{view}"), "residual_fingerprint": (_sha(f"res|{a}") if "psr" in a else None),
                                         "petri_source_kind": ("vector_all" if a == ac.ARDITI_CELL else ("psr" if "psr" in a else "lora")), "lora_int_id": i + 1}
                    for i, a in enumerate(fitted)}
        adapters["none__lora0"] = {"fingerprint_bf16": _sha("zero"), "residual_fingerprint": None, "petri_source_kind": None, "lora_int_id": len(fitted) + 1}
        self.lora_of = {self.harness_id(a): self.harness_id(a) for a in fitted}          # cell -> the LoRA name the engine states; base cells absent
        fs = from_S if from_S is not None else max(S + 1, (S * 5) // 4)
        pin = arditi_pin or _sha("arditi-object")
        obj_by_cell = {self.harness_id(a): _sha(f"obj|{a}") for a in self.arms if a not in ("none",) and not a.startswith("prompted__")}
        if ac.ARDITI_CELL in obj_by_cell:
            obj_by_cell[ac.ARDITI_CELL] = pin
        self.endpoint = {
            "proxy_url": "http://127.0.0.1:18080/v1", "mode": "petri", "eval": "capability:fixture", "model": "Qwen/Qwen3.5-4B",
            "model_key": ac.MODEL_KEY, "model_revision": MODEL_REVISION, "revision": MODEL_REVISION, "engine": "lora_native",
            "vmodels": {self.harness_id(a): vm for a, vm in self.vmodels.items()}, "cells": list(self.arms),
            "harness_cells": [self.harness_id(a) for a in self.arms], "cells_requested": list(self.arms),
            "attested_max_num_seqs": S, "attested_max_model_len": L,
            "max_num_seqs": S, "max_model_len": L,
            "geometry": {"max_model_len": L, "max_num_seqs": S, "window_source": "table",
                         "adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": fs, "to_S": S, "D": 4096, "ceiling": 524287}]},
            "attested_cudagraph_mode": "FULL", "attested_cudagraph_full_capture_max": S, "attested_enforce_eager": False,
            # the list the composer hands --cudagraph-capture-sizes: the REAL composer function for this S (a function of the
            # attested S and the FULL-capture rule — never an independent identity axis)
            "composed_cudagraph_capture_sizes": list(_VS.cudagraph_capture_sizes_for(S)),
            "lora_config": {"max_loras": 6, "max_lora_rank": 8, "max_cpu_loras": 6, "lora_dtype": "auto", "fully_sharded_loras": False,
                            "target_modules": None},
            "default_chat_template_kwargs": template_kwargs if template_kwargs is not None else {"enable_thinking": thinking == "on"},
            "server_default_policy": server_default_policy, "thinking_mode": thinking, "max_logprobs": max_logprobs,
            "object_artifact_sha256_by_cell": obj_by_cell, "expected_object_sha256_by_cell": dict(obj_by_cell),
            "serving_config_sha256": _sha(f"serving-config|{leg}|{stage}"), "config_used": f"configs/fixture_{view}.yaml",
            "serving_attestation": str(serve / ac.ATTESTATION_FILE), "written_at": _iso(T0)}
        self.attestation = {"kind": "lora_native", "schema_version": 2, "model": "Qwen/Qwen3.5-4B", "revision": MODEL_REVISION,
                            "scheduler_geometry": {"worker_reports": [{"max_num_seqs": S, "max_model_len": L}]},
                            "engine": {"enforce_eager": False, "cudagraph_mode": "FULL", "cudagraph_full_capture_max": S},
                            "lora_config": dict(self.endpoint["lora_config"]), "adapters": adapters,
                            "id_name_map": {str(blk["lora_int_id"]): name for name, blk in adapters.items()}}
        self.serve = serve
        self._write_serve()

    # -- the serve dir ----------------------------------------------------------
    def harness_id(self, arm: str) -> str:
        """The served HARNESS cell id of `arm` (`harness_of`, identity by default)."""
        return self.harness_of.get(arm, arm)

    def vmodel_for(self, arm: str) -> str:
        """The vmodel of `arm` in EITHER spelling — the arm the static / BoN / multi-turn drivers name their store dirs by, or the
        harness id the misalign runner names its cell dir by (`cells.json`'s `cell_id`); the fixtures here write a misalign store
        under the harness spelling exactly as the real runner does."""
        if arm in self.vmodels:
            return self.vmodels[arm]
        by_h = {self.harness_id(a): v for a, v in self.vmodels.items()}
        if arm in by_h:
            return by_h[arm]
        raise KeyError(f"{arm!r} is neither an arm of this root ({sorted(self.vmodels)}) nor a harness spelling of one ({sorted(by_h)})")

    def _write_serve(self) -> None:
        _json(self.serve / ac.ENDPOINT_FILE, self.endpoint)
        _json(self.serve / ac.ATTESTATION_FILE, self.attestation)
        # the standup's OTHER identity records, as `native_standup` writes them and `seedrep_served.ATTEMPT_ROOT_LAYOUT` ships them:
        # `cells_served.json` (both spellings + the canonical / controls rows `cell_id` -> `virtual_model_name`) and `routing_table.json`
        # (`routes[*].cell_id` -> `virtual_model`), derived from the endpoint record's OWN `vmodels` so the three records agree by construction
        vm = dict(self.endpoint.get("vmodels") or {})
        tt_of = {self.harness_id(a): a for a in self.arms}
        def _identity(h):      # the REAL record's `serving_identity`: the base weights for `none` / a prompted cell, a LoRA route otherwise
            return "proxy_vllm+passthrough+interleaved" if (h == "none" or h.startswith("prompted__")) else "proxy_vllm+lora+interleaved"
        rows = [{"cell_class": ("base" if h == "none" else ("prompted" if h.startswith("prompted__") else "lora")), "cell_id": h,
                 "control_of": None, "virtual_model_name": v, "kind": ("none" if h == "none" else ("prompted" if h.startswith("prompted__") else "lora")), "si_id": None,
                 "serving_identity": _identity(h), "traintest_cell": tt_of.get(h)} for h, v in vm.items()]
        _json(self.serve / ac.CELLS_SERVED_FILE, {"mode": self.endpoint.get("mode"), "eval": self.endpoint.get("eval"), "engine": "lora_native",
                                                  "cells_requested": list(self.endpoint.get("cells_requested") or self.arms),
                                                  "harness_cells": list(self.endpoint.get("harness_cells") or [self.harness_id(a) for a in self.arms]),
                                                  "expected_object_sha256_by_cell": dict(self.endpoint.get("expected_object_sha256_by_cell") or {}),
                                                  "canonical": [r for r in rows if r["cell_id"] != "none__lora0"],
                                                  "controls": [{**r, "cell_class": "control", "control_of": "none"} for r in rows if r["cell_id"] == "none__lora0"],
                                                  "written_at": _iso(T0)})
        _json(self.serve / ac.ROUTING_TABLE_FILE, {"version": 2, "base_model": self.endpoint.get("model"), "serving_mode": "interleaved",
                                                   "routes": [{"virtual_model": v, "cell_id": h, "cell_class": ("base" if h == "none" else ("prompted" if h.startswith("prompted__") else "lora")),
                                                               "serving_identity": _identity(h),
                                                               "intervention_hash": _sha(f"iv|{h}")[:16], "control_of": None, "si_prefix_hash": None}
                                                              for h, v in vm.items()]})
        _jsonl(self.serve / ac.STANDUP_JOBS_FILE, [{"job_id": "manual", "node": "TESTHOST", "key": ac.MODEL_KEY, "view": self.view,
                                                    "mode": "petri", "eval": "capability:fixture", "sha": self.W, "engine": "lora_native",
                                                    "serving_path": "lora_native", "max_num_seqs": self.S, "max_model_len": self.L,
                                                    "attested_max_num_seqs": self.S, "attested_max_model_len": self.L,
                                                    "thinking_mode": self.thinking, "written_at": _iso(T0)}])

    def rewrite_endpoint(self, **fields: Any) -> None:
        self.endpoint.update(fields)
        self._write_serve()

    def drop_standup_jobs(self) -> None:
        """The earlier Modal app shape: no `standup_jobs.jsonl` was ever written."""
        (self.serve / ac.STANDUP_JOBS_FILE).unlink()

    def write_standup_jobs_line(self, line: Mapping[str, Any]) -> None:
        _jsonl(self.serve / ac.STANDUP_JOBS_FILE, [dict(line)])

    def write_gsm8k_cell_copies(self, cells: Sequence[str], *, s: int = 1) -> List[Path]:
        """The GSM8K assembly's per-cell serve dirs as `seedrep_served.gsm8k_child_serve_dir` writes them:
        `<root>/exp/serve_gsm8k/<cell>__s<s>/{routing_table.json, endpoint.json}` — the standup's endpoint record copied with
        `routing_table` rewritten to the copy and `ledger_serve_copy_of` naming the standup's (F-AE: CELL records)."""
        out: List[Path] = []
        for cell in cells:
            d = self.path / ac.GSM8K_CELL_SERVE_DIR_REL / f"{cell}__s{s}"
            d.mkdir(parents=True, exist_ok=True)
            _json(d / "routing_table.json", {"version": 1, "routes": {}})
            ep = dict(self.endpoint)
            ep["routing_table"] = str(d / "routing_table.json")
            ep[ac.CELL_ENDPOINT_COPY_OF_FIELD] = str(self.serve / ac.ENDPOINT_FILE)
            _json(d / ac.ENDPOINT_FILE, ep)
            out.append(d / ac.ENDPOINT_FILE)
        return out

    def split_pipeline_by_store_root(self, store_root_of_arm: Mapping[str, Path], *, empty_ledgers: Sequence[Path] = ()) -> Dict[str, Path]:
        """The REAL base-child layout (`seedrep_launch_plan.child_store_root`: a child's evidence dir IS its store root; `gsm8k_ledger_argv`:
        its usage ledger beside it): the root-level `request_evidence.jsonl` / `usage_ledger.jsonl` / `response_evidence.jsonl` this
        fixture wrote are re-laid PER ARM into `store_root_of_arm[arm]` and the root-level files removed. `empty_ledgers` = ledger paths of
        children that started (the writer creates the file at start) but drew nothing. Returns {arm: store root}."""
        arm_of_rid = {str(r["request_id"]): str(r["extra"]["arm"]) for r in self.records}
        for name in (ac.EVIDENCE_FILE, ac.LEDGER_FILE, ac.RESPONSE_EVIDENCE_FILE):   # a previous split's child copies go first
            for prev, owner in ac.owned_evidence_files(self.path, name):
                if owner == ac.OWNER_CHILD_STORE:
                    prev.unlink()
        for arm, d in store_root_of_arm.items():
            d.mkdir(parents=True, exist_ok=True)
            _jsonl(d / ac.EVIDENCE_FILE, [r for r in self.records if r["extra"]["arm"] == arm])
            _jsonl(d / ac.LEDGER_FILE, [r for r in self.ledger if arm_of_rid.get(str(r.get("request_id"))) == arm])
            _jsonl(d / ac.RESPONSE_EVIDENCE_FILE, [r for r in self.responses if arm_of_rid.get(str(r.get("request_id"))) == arm])
        for p in empty_ledgers:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("")
        for name in (ac.EVIDENCE_FILE, ac.LEDGER_FILE, ac.RESPONSE_EVIDENCE_FILE):
            (self.path / name).unlink(missing_ok=True)
        return dict(store_root_of_arm)

    def pipeline_files(self, name: str) -> List[Path]:
        """Every owned copy of `name` on this root (the attempt root's and each child store's) — what a test rewriting the
        pipeline must rewrite, whichever layout the leg has."""
        return [p for p, _ in ac.owned_evidence_files(self.path, name)]

    def teardown_endpoint(self, *, restore: bool) -> None:
        """`mt_serving_up.py`'s teardown unlinks the live `endpoint.json`; with `restore` the Modal app's
        `restore_endpoint_record` (fy) puts the READY bytes back at the SAME path once the standup process has exited."""
        live = self.serve / ac.ENDPOINT_FILE
        ready = live.read_bytes()
        live.unlink()
        if restore:
            live.write_bytes(ready)

    # -- the request pipeline -----------------------------------------------------
    def send(self, task: str, item_id: str, sample_idx: int, vmodel: str, body: Mapping[str, Any], *, ok: bool = True,
             finish_reason: str = "stop", text: Optional[str] = None, completion_tokens: int = 17, prompt_tokens: int = 40,
             body_raw: Optional[bytes] = None, s: Optional[int] = None) -> Tuple[str, Optional[str], str]:
        """One TARGET attempt on the wire: returns (request_id, completion_id | None on a rejected call, text). `s` = the
        seed base stamped on the record's extra block (default: the view's; the Arditi root passes the store's)."""
        rid = uuid.uuid4().hex
        headers = {**ledger_headers(task, item_id, sample_idx), C.HEADER_REQUEST_ID: rid}
        raw = body_raw if body_raw is not None else json.dumps(dict(body), separators=(",", ":")).encode("utf-8")
        # the request record as `RequestEvidenceClient.record_request` writes it: the contracts v3 record over the EXACT wire
        # bytes + `extra` + the attempt id — plus the ADDITIVE `body_b64` the determinism check reads (the recorder writes
        # body_sha256 and body_bytes, a length; the bytes themselves are the field the recorder must add — the consumer
        # refuses without it, see determinism_check.bodies_from_records)
        extra = {"leg": self.leg, "arm": vmodel.replace("petri__", ""), "s": ac.seed_of_view(self.view) if s is None else s,
                 "attempt_id": self.attempt_id}
        rec = C.request_evidence_record(raw, headers=headers, ts=_iso(self._t))
        rec["extra"] = dict(extra)
        rec[C.REQUEST_EVIDENCE_ATTEMPT_FIELD] = self.attempt_id
        rec[ac.REQUEST_BODY_B64_FIELD] = base64.b64encode(raw).decode("ascii")
        self.records.append(rec)
        row = dict.fromkeys(C.LEDGER_ROW_FIELDS)
        cid: Optional[str] = None
        text = text if text is not None else f"response for {item_id}/{sample_idx} under {vmodel}"
        if ok:
            cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            self.engine_lines.append(f"INFO [request_logger.py:64] Received request {cid}-{uuid.uuid4().hex[:8]}: params: "
                                     f"SamplingParams(n=1, seed={body.get('seed')}, max_tokens={body.get('max_tokens')}), lora_request: None.")
        # the response record as `record_response` writes it — one row per attempt that returned or died, RESPONSE_EVIDENCE_FIELDS in
        # order + `extra` — plus the lane's ADDITIVE `text_sha256` on a 2xx (the response side of the determinism check,
        # which the consumer refuses without); a rejected attempt records status 500, no completion and the error text
        # version 2: the version first, `text_sha256` + `text_sha256_of` after `finish_reason` — the content hash on a 2xx
        # (`text_sha256_of = "content"`, the same text the stores persist), both None on a rejected attempt
        resp_row = {"response_evidence_version": ac.RESPONSE_EVIDENCE_VERSION, "ts": _iso(self._t), "request_id": rid, "completion_id": cid,
                    "status": 200 if ok else 500, "vmodel": vmodel if ok else None, "finish_reason": finish_reason if ok else None,
                    ac.RESPONSE_TEXT_SHA_FIELD: _sha(text) if ok else None, ac.RESPONSE_TEXT_SHA_OF_FIELD: "content" if ok else None,
                    "task": task, "sample": C.sample_header_value(item_id, sample_idx), "error": None if ok else "HTTP 500: synthetic failure"}
        assert tuple(resp_row) == ac.RESPONSE_EVIDENCE_FIELDS
        resp_row["extra"] = dict(extra)
        self.responses.append(resp_row)
        row.update(ts=_iso(self._t), task=task, sample=C.sample_header_value(item_id, sample_idx), vmodel=vmodel,
                   thinking=body.get("chat_template_kwargs", {}).get("enable_thinking") if isinstance(body.get("chat_template_kwargs"), dict) else None,
                   prompt_tokens_pre=prompt_tokens, max_tokens_requested=body.get("max_tokens"), has_tools=False, n_tools=0,
                   status=200 if ok else 500, error_code=None if ok else "http_500", rejected=not ok,
                   usage_prompt_tokens=prompt_tokens if ok else None, usage_completion_tokens=completion_tokens if ok else None,
                   finish_reason=finish_reason if ok else None, latency_s=0.5, tokenize_mismatch=False if ok else None,
                   error_class="none" if ok else "server_error", request_id=rid, body_sha256=C.wire_body_sha256(raw))
        row["completion_id"] = cid
        self.ledger.append(row)
        self._t += timedelta(seconds=1)
        return rid, cid, text

    def flush_pipeline(self, *, complete: bool = True, result_rc: Optional[str] = None, n_scrapes: int = 12, running: Optional[int] = None,
                       inflight: Optional[int] = None, sm: int = 92, tok_per_scrape: Optional[int] = None) -> None:
        _jsonl(self.path / ac.EVIDENCE_FILE, self.records)
        _jsonl(self.path / ac.LEDGER_FILE, self.ledger)
        (self.serve / "vllm_server.log").write_text("\n".join(self.engine_lines) + "\n", encoding="utf-8")
        _jsonl(self.path / ac.RESPONSE_EVIDENCE_FILE, self.responses)
        self.write_samplers(n_scrapes=n_scrapes, running=running, inflight=inflight, sm=sm, tok_per_scrape=tok_per_scrape)
        if result_rc is not None:
            (self.path / ac.RESULT_FILE).write_text(f"{_iso(self._t)} job=manual target=fixture rc={result_rc} agentic_rc={result_rc}\n", encoding="utf-8")
        if complete:
            _json(self.path / ac.COMPLETE_FILE, {"written_utc": _iso(T0 + timedelta(hours=2)), "attempt_id": self.attempt_id})
        if self.leg == "gsm8k" and self.records:
            # the REAL layout of a GSM8K row (fz2): one usage ledger + one evidence pair PER CHILD STORE ROOT (`gsm8k_ledger_argv`,
            # `child_store_root`), nothing at the attempt root — the readers refuse an attempt-level ledger on a GSM8K row by name
            arms = sorted({str(r["extra"]["arm"]) for r in self.records})
            s = {int(r["extra"]["s"]) for r in self.records if r["extra"].get("s") is not None}
            seed = s.pop() if len(s) == 1 else ac.seed_of_view(self.view)
            self.split_pipeline_by_store_root({arm: self.path / GSM8K_OUT_REL.format(cap=GSM8K_FIXTURE_CAP, s=seed, cell=arm) for arm in arms})

    def write_samplers(self, *, n_scrapes: int = 12, running: Optional[int] = None, inflight: Optional[int] = None, sm: int = 92,
                       tok_per_scrape: Optional[int] = None, period_s: int = 10, dmon_timestamps: bool = True,
                       stall_scrapes: Sequence[int] = (), reset_scrapes: Sequence[int] = ()) -> None:
        """`stall_scrapes` = scrape indices after which the token counter does NOT advance although the engine reports
        Running > 0 (a decode stall: an eligible interval at 0 tok/s); `reset_scrapes` = scrape indices at which the counter
        reads 0 again (an engine restart: the interval INTO that scrape is non-monotone and ineligible)."""
        run = self.S if running is None else running
        infl = self.S if inflight is None else inflight
        tps = tok_per_scrape if tok_per_scrape is not None else 50 * self.S * period_s
        metrics, infl_rows, dmon = [], [], []
        gen = 0
        dmon.append("#Date       Time        gpu    sm   mem   enc   dec   jpg   ofa    fb  bar1  ccpm" if dmon_timestamps
                    else "# gpu    sm   mem   enc   dec   jpg   ofa    fb  bar1  ccpm")
        dmon.append("#YYYYMMDD   HH:MM:SS    Idx     %     %     %     %     %     %    MB    MB    MB" if dmon_timestamps
                    else "# Idx     %     %     %     %     %     %    MB    MB    MB")
        for k in range(n_scrapes):
            t = T0 + timedelta(seconds=period_s * k)
            r = 0 if k == 0 else run
            if k in reset_scrapes:
                gen = 0
            metrics.append({"ts": _iso(t), "vllm:num_requests_running": r, "vllm:num_requests_waiting": max(0, infl - r),
                            "vllm:generation_tokens_total": gen, "vllm:prompt_tokens_total": 1000 * k, "vllm:gpu_cache_usage_perc": 0.5})
            gen += tps if (r and k not in stall_scrapes) else 0
            infl_rows.append({"ts": _iso(t), "inflight": 0 if k == 0 else infl})
            s = 3 if k == 0 else sm
            prefix = f"{t.strftime('%Y%m%d')}   {t.strftime('%H:%M:%S')}    " if dmon_timestamps else ""
            dmon.append(f"{prefix}0    {s:2d}    40     0     0     0     0  90000     4     0")
        _jsonl(self.path / ac.METRICS_FILE, metrics)
        _jsonl(self.path / ac.INFLIGHT_FILE, infl_rows)
        (self.path / ac.DMON_FILE).write_text("\n".join(dmon) + "\n", encoding="utf-8")

    # -- decode bodies -------------------------------------------------------------
    def vendor_body(self, vmodel: str, item: str, cap: int, seed: int, **extra: Any) -> Dict[str, Any]:
        return {"model": vmodel, "messages": [{"role": "user", "content": f"prompt {item}"}], "max_tokens": cap, **VENDOR_BODY, "seed": seed, **extra}


# --------------------------------------------------------------------------- #
# store writers per leg — each sends the requests THROUGH the root's pipeline   #
# --------------------------------------------------------------------------- #

def _private_run(view: str, cap: Optional[int] = None) -> str:
    base = f"{ac.MODEL_KEY}_{view}_sampled_t0" if view != ac.ARDITI_VIEW else f"arditi_allpos_{ac.MODEL_KEY}_ar_c0p3_seedrep_s{{s}}_sampled_t0"
    return base + (f"_cap{cap}" if cap else "")


def write_static_store(root: Root, *, family: str, arm: str, s: int, goals: Sequence[str], n_variants: int = 2, trial: int = 0,
                       cap: int = 4096, at_cap_goals: Sequence[str] = (), task: Optional[str] = None, omit_body_keys: Sequence[str] = (),
                       stage_cap: Optional[int] = None, bind_rows: bool = True) -> Path:
    task = task or ac.static_task(family)
    vmodel = root.vmodels[arm]
    pr = _private_run(root.view, stage_cap).format(s=s)
    shard = root.path / "exp" / pr / family / arm / "shard00"
    rows, progress = [], []
    decode = {"do_sample": True, **VENDOR_BODY, "cap": cap, "arm": "vendor_recommended", "seed_base": s, "trial": trial}
    for g in goals:
        for i in range(n_variants):
            seed = ac.static_sampling_seed(s, trial, family, g, i)
            body = root.vendor_body(vmodel, f"{g}/{i}", cap, seed)
            for k in omit_body_keys:
                body.pop(k, None)
            fr = "length" if g in at_cap_goals else "stop"
            rid, cid, text = root.send(task, g, i, vmodel, body, finish_reason=fr, completion_tokens=cap if fr == "length" else 17, s=s)
            row = {"goal_id": g, "sample_idx": i, "trial": trial, "variant_key": f"v{i}_t{trial}", "variant_name": f"variant_{i}",
                   "family": family, "variant": f"prompt {g}/{i}", "attack_prompt": f"prompt {g}/{i}", "response": text,
                   "engine": "vllm_served", "serving_engine": "lora_native", "decode": dict(decode), "sampling_seed": seed,
                   "finish_reason": fr, "completion_tokens": cap if fr == "length" else 17, "prompt_tokens": 40, "max_tokens": cap,
                   "at_cap": fr == "length"}
            if bind_rows:
                row.update(request_id=rid, completion_id=cid)
            rows.append(row)
        progress.append({"goal_id": g, "category": "cat", "family": family, "n_variants": n_variants, "n_target_forwards": n_variants,
                         "outcome": "generated", "capped": False})
    _jsonl(shard / "responses.jsonl", rows)
    _jsonl(shard / "progress.jsonl", progress)
    _json(shard / "served_target.json", {"target_engine": "served", "virtual_model": vmodel, "cell_id": arm, "manifest_stage": "final",
                                          "decode": decode})
    return shard


def write_static_rung(root: Root, *, base_shard: Path, family: str, arm: str, s: int, cap_from: int, cap_to: int,
                      regenerate: Optional[Sequence[Tuple[str, int]]] = None, still_at_cap: Sequence[str] = ()) -> Path:
    """A static RUNG store emitted the way `escalate_static.py` emits one (its `main`'s row loop, mirrored key for key —
    the real writer needs a served target, so the fixture reproduces its emission and holds every row to the writer's
    own rule): every source row is copied forward VERBATIM — its decode block (`cap` = the rung that produced its text),
    its request_id / completion_id (an attempt on the SOURCE root, never this one) — plus `escalated_from_cap`,
    `regenerated: false` and the PRESERVED `source_cap`; the source's at-cap rows (or `regenerate`) are RE-DRAWN at
    `cap_to` through THIS root's pipeline with the same sampling_seed: `decode` = the block at cap_to, `regenerated:
    true`, `source_cap = cap_to`, the divergence fields. So the rung root is a genuinely MIXED-CAP store. Every emitted
    row is validated through `escalate_static.row_source_cap` (a fixture row contradicting the writer's provenance rule
    fails here, not in the audit)."""
    esc = ac.escalate_static_module()
    base_rows = ac.read_jsonl(base_shard / "responses.jsonl")
    base_prog = ac.read_jsonl(base_shard / "progress.jsonl")
    vmodel = root.vmodels[arm]
    task = ac.static_task(family)
    shard = root.path / "exp" / _private_run(root.view, cap_to).format(s=s) / family / arm / "shard00"
    regen_keys = (set(map(tuple, regenerate)) if regenerate is not None
                  else {(r["goal_id"], r["sample_idx"]) for r in base_rows if r.get("at_cap")})
    out_rows: List[dict] = []
    regen_per_goal: Dict[str, int] = {}
    n_after = 0
    for base in base_rows:
        row = dict(base)
        row["escalated_from_cap"] = cap_from
        src_cap = int(esc.row_source_cap(base, store_cap=cap_from, where=str(base_shard)))
        key = (base["goal_id"], base["sample_idx"])
        if key not in regen_keys:
            row["regenerated"] = False
            row["source_cap"] = src_cap  # PRESERVED through every copy (escalate_static: never relabelled)
            n_after += int(bool(base.get("at_cap")))
        else:
            fr = "length" if base["goal_id"] in still_at_cap else "stop"
            ct = cap_to if fr == "length" else 23
            body = root.vendor_body(vmodel, f"{base['goal_id']}/{base['sample_idx']}", cap_to, base["sampling_seed"])
            rid, cid, text = root.send(task, base["goal_id"], base["sample_idx"], vmodel, body, finish_reason=fr, completion_tokens=ct,
                                       text=f"{base['response']} (re-drawn at {cap_to})", s=s)
            row.update(response=text, decode={**base["decode"], "cap": cap_to}, finish_reason=fr, completion_tokens=ct, prompt_tokens=40,
                       max_tokens=cap_to, at_cap=fr == "length", regenerated=True, source_cap=cap_to,
                       source_completion_tokens=base.get("completion_tokens"), source_finish_reason=base.get("finish_reason"),
                       source_draw_cap=src_cap, boundary_landing=ct < src_cap, prefix_extension=True, identical=False,
                       first_divergence_char=None, first_divergence_token=None, request_id=rid, completion_id=cid)
            regen_per_goal[base["goal_id"]] = regen_per_goal.get(base["goal_id"], 0) + 1
            n_after += int(row["at_cap"])
        esc.row_source_cap(row, store_cap=cap_to, where=str(shard))  # the writer's rule validates the fixture's row
        out_rows.append(row)
    progress = []
    for p in base_prog:
        n_re = regen_per_goal.get(p["goal_id"], 0)
        progress.append({**p, "n_target_forwards": n_re, "n_regenerated": n_re, "n_copied": int(p.get("n_variants", 0)) - n_re})
    block_to = {**base_rows[0]["decode"], "cap": cap_to}
    _jsonl(shard / "responses.jsonl", out_rows)
    _jsonl(shard / "progress.jsonl", progress)
    _json(shard / "served_target.json", {"target_engine": "served", "virtual_model": vmodel, "cell_id": arm, "manifest_stage": "final",
                                          "decode": block_to})
    n_regen = sum(regen_per_goal.values())
    _json(shard / "escalation.json", {"attack": family, "defense": arm, "cap_from": cap_from, "cap_to": cap_to, "n_rows": len(out_rows),
                                      "n_at_cap_before": sum(1 for r in base_rows if r.get("at_cap")), "n_regenerated": n_regen,
                                      "n_at_cap_after": n_after, "n_copied": len(out_rows) - n_regen,
                                      "n_rows_by_source_cap": {str(c): sum(1 for r in out_rows if r["source_cap"] == c)
                                                               for c in sorted({r["source_cap"] for r in out_rows})},
                                      "seed_base": s, "trial": int(block_to["trial"]), "arm": block_to["arm"], "decode": block_to,
                                      "re_draw_not_continuation": True})
    return shard


def write_bon_store(root: Root, *, arm: str, s: int, goals: Sequence[str], n_draws: int = 2, trial: int = 0, cap: int = 4096,
                    at_cap_goals: Sequence[str] = ()) -> Path:
    vmodel = root.vmodels[arm]
    cell = root.path / "exp" / _private_run(root.view).format(s=s) / arm
    rows = []
    for g in goals:
        for i in range(n_draws):
            seed = ac.bon_decode_seed_v1(g, i, trial, s)
            fr = "length" if g in at_cap_goals else "stop"
            rid, cid, text = root.send(ac.BON_TASK, g, i, vmodel, root.vendor_body(vmodel, f"{g}/{i}", cap, seed), finish_reason=fr, s=s)
            rows.append({"goal_id": g, "sample_idx": i, "trial": trial, "variant": f"aug {g}/{i}", "response": text, "engine": "vllm_served",
                         "serving_engine": "lora_native", "cell_id": arm, "bon_max_concurrency": 8, "finish_reason": fr,
                         "decode": dict(VENDOR_BODY), "bon_decode_preset": "vendor", "sampling_seed": seed,
                         "seed_rule": "bon_decode_seed_v1", "decode_seed_policy": "replicate_v1", "decode_seed_base": s,
                         "augment_seed": 12345, "request_id": rid, "completion_id": cid})
    _jsonl(cell / "bon_responses.jsonl", rows)
    _json(cell / "bon_served_manifest.json", {"engine": "vllm_served", "cell_id": arm, "virtual_model": vmodel, "bon_max_concurrency": 8,
                                              "decode": dict(VENDOR_BODY), "bon_decode_preset": "vendor", "seed_rule": "bon_decode_seed_v1",
                                              "decode_seed_policy": "replicate_v1", "decode_seed_base": s,
                                              "units": {"n_planned": len(rows), "n_done": len(rows), "n_uncovered": 0, "n_work": len(rows),
                                                        "n_written": len(rows), "n_prompt_over_window": 0}, "errors_file": "bon_errors.jsonl"})
    return cell


def write_multiturn_store(root: Root, *, attack: str, arm: str, s: int, goals: Sequence[str], n_turns: int = 2, trial: int = 0,
                          cap: int = 4096, at_cap_goals: Sequence[str] = ()) -> Path:
    vmodel = root.vmodels[arm]
    task = ac.multiturn_task(attack)
    shard = root.path / "exp" / _private_run(root.view).format(s=s) / attack / arm / "shard00"
    rows, progress = [], []
    td = {**VENDOR_BODY, "do_sample": True, "cap": cap, "arm": "vendor_recommended", "seed_base": s, "trial": trial,
          "seed_rule": "multiturn_seed", "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}
    for g in goals:
        for turn in range(n_turns):
            seed = ac._multiturn_seed(s, trial, attack, g, turn)
            fr = "length" if g in at_cap_goals else "stop"
            rid, cid, text = root.send(task, g, turn, vmodel, root.vendor_body(vmodel, f"{g}/{turn}", cap, seed), finish_reason=fr,
                                       completion_tokens=cap if fr == "length" else 17, s=s)
            rows.append({"goal_id": g, "order": turn, "response": text, "serving_engine": "lora_native", "decode": {**VENDOR_BODY, "seed": seed},
                         "finish_reason": fr, "request_id": rid, "completion_id": cid})
        progress.append({"goal_id": g, "category": "cat", "n_target_forwards": n_turns, "outcome": "failure", "actors_used": 0,
                         "capped": False, "served_window": root.L, "target_decode": dict(td)})
    _jsonl(shard / "responses.jsonl", rows)
    _jsonl(shard / "progress.jsonl", progress)
    _json(shard / "served_target.json", {"target_engine": "served", "virtual_model": vmodel, "cell_id": arm, "manifest_stage": "final",
                                          "target_decode": dict(td), "decode": dict(td)})      # `decode` = the driver's identity block (the real shape)
    return shard


def write_misalign_store(root: Root, *, arm: str, s: int, items: Mapping[str, str], n_samples: int = 2, cap: int = 3000,
                         at_cap_items: Sequence[str] = (), loop_items: Sequence[str] = (), stage_cap: Optional[int] = None,
                         only_pairs: Optional[Sequence[Tuple[str, int]]] = None) -> Path:
    """`items` = {item_id: behavior}; `loop_items` are at cap with a degenerate (repeating) text. `arm` may be the HARNESS spelling (the
    real runner's cell dir name, F-AO) — the vmodel resolves through `root.vmodel_for`."""
    vmodel = root.vmodel_for(arm)
    rule, _ = ac.misalign_seed_rule()
    tree = "misalign_16k" if stage_cap else "misalign"
    cell = root.path / "exp" / tree / ac.MODEL_KEY / "agentic" / arm
    rows = []
    for item, beh in items.items():
        for k in range(n_samples):
            if only_pairs is not None and (item, k) not in only_pairs:
                continue
            seed = int(rule(s, beh, item, k))
            fr = "length" if (item in at_cap_items or item in loop_items) else "stop"
            text = ("ab" * 400) if item in loop_items else None
            rid, cid, text = root.send(ac.misalign_task(beh), item, k, vmodel, root.vendor_body(vmodel, f"{item}/{k}", cap, seed),
                                       finish_reason=fr, text=text, s=s)
            rows.append({"family": "agentic", "cell_id": arm, "virtual_model": vmodel, "item_id": item, "sample_idx": k, "behavior": beh,
                         "prompt_sha": _sha(f"prompt {item}"), "response": text, "finish_reason": fr,
                         "usage": {"prompt_tokens": 40, "completion_tokens": 17}, "model_reported": vmodel,
                         "request_id": rid, "completion_id": cid})
    _jsonl(cell / "responses.jsonl", rows)
    _json(cell / "meta.json", {"family": "agentic", "suite_revision": "fixture", "virtual_model": vmodel, "intervention_hash": _sha(arm),
                               "si_id": "constv2", "max_tokens": cap, "temperature": 0.7, "plain": False, "sampler": "vendor",
                               "sampling": dict(VENDOR_BODY), "seed_policy": "per_response", "seed_base": s, "cell_id": arm,
                               "n_samples": n_samples, "params": {"max_tokens": cap, "temperature": 0.7, "sampler": "vendor",
                                                                   "sampling": dict(VENDOR_BODY), "concurrency": 8},
                               "items": len(items), "item_ids": list(items), "behaviors": sorted(set(items.values()))})
    return cell


def _validate_target_turns(turns: Sequence[Mapping[str, Any]], *, task: str, sid: str) -> None:
    """The fixture's list through the REAL validators: `petri_target_turns_of_sample` (the converter's own rule) when
    the lane client is on the tree, and always the audit's fail-closed reader (`audit_common.petri_target_turns`)."""
    try:
        from sisteer.eval import request_evidence_client as rev  # type: ignore[import-not-found]
        rev.petri_target_turns_of_sample({"store": {rev.PETRI_TARGET_TURNS_STORE_KEY: list(turns)}}, task=task, scenario_id=sid)
    except ImportError:
        pass
    ac.petri_target_turns([{"task": task, "item_id": sid, "metric": "probe", "extra_json": json.dumps({ac.PETRI_TARGET_TURNS_KEY: list(turns)})}],
                          where=Path("fixture"))


def write_petri_store(root: Root, *, arm: str, s: int, scenarios: Mapping[str, str], n_turns: int = 2, cap: int = 1024,
                      dimensions: Sequence[str] = ("dim_x", "dim_y"), failed_turns: Mapping[str, Sequence[int]] = (),
                      at_cap_scenarios: Sequence[str] = ()) -> Path:
    """`scenarios` = {scenario_id: behaviour}; rows per (scenario, dimension) in `petri_runner.convert_eval_log`'s shape
    where every dimension row carries the SAME ordered `extra_json.target_turns` — every TARGET attempt of the
    scenario in wire order as `{turn (1-based), request_id, completion_id, status, task}`; a turn in `failed_turns[sid]`
    is attempted once with a 500 (an entry with no completion — never a draw) and then retried at the same turn. No ids
    ride the row itself (nor a finish reason: the target turns of a scenario in `at_cap_scenarios` end `length` on the WIRE —
    the response hook's row and the ledger row — which is the only place the at-cap table can read a Petri truncation from).
    The list is validated through the real rules before it is written."""
    vmodel = root.vmodels[arm]
    cell = root.path / "exp" / "petri" / "cells" / f"{arm.replace('__', '--')}--petri--{_sha(arm)[:16]}"
    rows = []
    failed_turns = dict(failed_turns) if failed_turns else {}
    for sid, beh in scenarios.items():
        task = ac.petri_task(beh)
        turns: List[Dict[str, Any]] = []
        fr = "length" if sid in at_cap_scenarios else "stop"
        for turn in range(1, n_turns + 1):
            seed = synthetic_petri_seed(s, beh, sid, turn)
            body = root.vendor_body(vmodel, f"{sid}/{turn}", cap, seed)
            if turn in failed_turns.get(sid, ()):
                rid, _cid, _ = root.send(task, sid, turn, vmodel, body, ok=False, s=s)
                turns.append({"turn": turn, "request_id": rid, "completion_id": None, "status": 500, "task": task})
            rid, cid, _ = root.send(task, sid, turn, vmodel, body, finish_reason=fr, s=s)
            turns.append({"turn": turn, "request_id": rid, "completion_id": cid, "status": 200, "task": task})
        _validate_target_turns(turns, task=task, sid=sid)
        for dim in dimensions:
            rows.append({"task": task, "item_id": sid, "metric": dim, "value": 3.0, "domain": "petri_behavior", "seed_base": s,
                         "extra_json": json.dumps({"dimension": dim, "behavior_dir_key": beh, "scenario_id": sid, "epoch": 1, "auditor_chain": [],
                                                   ac.PETRI_TARGET_TURNS_KEY: turns}, sort_keys=True)})
    _jsonl(cell / "rows.jsonl", rows)
    _json(cell / "manifest.json", {"cell_id": arm, "petri_run_key": _sha(arm)[:16]})
    _json(cell / "results.json", {"n_rows": len(rows)})
    return cell


def _tinycap_cell(root: Root, arm: str) -> Path:
    return root.path / "exp" / "tinycap" / "cells" / f"{arm.replace('__', '--')}--controls--{_sha(arm)[:12]}"


def write_gsm8k_store(root: Root, *, arm: str, s: int, items: Sequence[str], cap: int = 8192, at_cap_items: Sequence[str] = (),
                      stage_cap: Optional[int] = None) -> Path:
    vmodel = root.vmodels[arm]
    task = ac.GSM8K_TASK
    cell = _tinycap_cell(root, arm) if not stage_cap else root.path / "exp" / f"tinycap_cap{stage_cap}" / "cells" / f"{arm.replace('__', '--')}--g8k"
    rows, outcome_items = [], []
    for it in items:
        seed = C.inspect_replicate_seed(it, 0, s)
        body = {"model": vmodel, "messages": [{"role": "user", "content": f"q {it}"}], "max_tokens": cap, **THINKING_CARD_BODY, "seed": seed,
                **C.chat_template_kwargs(C.THINKING_ON)}
        fr = "length" if it in at_cap_items else "stop"
        rid, cid, _ = root.send(task, it, 0, vmodel, body, finish_reason=fr, s=s)
        rows.append({"task": task, "item_id": it, "metric": "acc", "value": 1.0, "domain": "math"})
        outcome_items.append({"item_id": it, "sample_idx": 0, "attempt": 1, "scored": True, "value": 1.0, "unparseable": False,
                              "stop_reason": "max_tokens" if fr == "length" else "stop", "at_cap": fr == "length",
                              "think_block_present": True, "think_block_unterminated": False, "stray_open_tag": False,
                              "content_empty_with_reasoning": False, "prompt_tokens": 40, "completion_tokens": 17, "seed": seed,
                              "request_id": rid, "completion_id": cid})
    rows.append({"task": task, "item_id": C.AGG_ITEM_ID, "metric": "acc", "value": 1.0, "domain": "math"})
    _jsonl(cell / "rows.jsonl", rows)
    _json(cell / "manifest.json", {"cell_id": arm, "seeds": {"inspect": {"rule": C.inspect_replicate_seed_rule(s),
                                                                         "replicate": {"policy": C.INSPECT_SEED_POLICY_REPLICATE, "seed_base": s}}}})
    from sisteer.capeval.tinycap import completion as _completion
    _json(cell / C.TASK_OUTCOMES_DIR / _completion.marker_filename(task),
          {"task": task, "item_set": "paired200", "thinking": "on", "think_split_mode": "x", "unterminated_policy": "x", "max_tokens": cap,
           "samples_per_item": 1, "n_attempted_items": len(items), "n_attempted_samples": len(items), "n_scored": len(items), "n_unscored": 0,
           "n_scored_samples": len(items), "n_unscored_samples": 0, "n_stop_reason": {"stop": len(items) - len(at_cap_items), "max_tokens": len(at_cap_items)},
           "n_at_cap": len(at_cap_items), "n_think_block_present": len(items), "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0,
           "n_completion_tokens": 17 * len(items), "producer_tree_sha": root.W, "items": outcome_items})
    # fg2: `n_rows` is the REAL writer's count -- `inspect_served.run_one_task` passes `finish_task(n_rows=n_rows_emitted)`,
    # the per-item / per-sample rows it emitted and NOT the two aggregate rows ("Counted the way the old formula counted: the two aggregate
    # rows are not in it"); the fixture counted the aggregate row (`len(items) + 1`), which no writer does
    _json(cell / C.DONE_DIR / _completion.marker_filename(task), {"task": task, "n_items": len(items), "n_rows": len(items),
                                                                    "n_requests": len(items), "n_rejected": 0, "wall_clock_s": 1.0,
                                                                    "finished_at": _iso(T0)})
    return cell


def _ws_tok(texts: Sequence[str], add_special_tokens: bool = False) -> Dict[str, Any]:
    """A whitespace tokenizer in the served tokenizer's calling shape — `_fork_fields` needs one for a forked re-draw."""
    return {"input_ids": [t.split() for t in texts]}


def write_controls_store(root: Root, *, arm: str, s: int, items_by_control: Mapping[str, Sequence[str]], mmlu_items: Sequence[str] = (),
                         trial: int = 0, cap: int = 512, at_cap: Mapping[str, Sequence[str]] = (), stage_cap: Optional[int] = None,
                         texts: Optional[Mapping[str, str]] = None, source: Optional[Path] = None,
                         regenerate: Optional[Mapping[str, Sequence[str]]] = None) -> Path:
    """One controls/MMLU cell dir written through the controls writer's OWN classes and functions: the spool by
    `judged.GenSpool` (header line, fsynced appends), every meta row by `GenMeta.append` (which refuses any key outside
    META_ROW_KEYS [+ META_REGEN_KEYS] — a request id on a meta row is impossible), a regenerated row's prefix-vs-fork
    fields by `sampled_controls._fork_fields`, and the `gen_stats[control]` block by `sampled_controls._meta_stats` over
    the bound meta (so it carries the writer's `cap`, `n_at_cap`, `n_copied`, ... — never a hand-shaped block). The
    Inspect-family binding is `response_evidence.jsonl` (written by `Root.send`), never the row. A RUNG store
    (`stage_cap` + `source` = the base cell dir) mirrors `_generate_control`'s rung loop: the items in `regenerate[control]`
    are RE-DRAWN at `stage_cap` through this root's pipeline with the same seed (regenerated true, source_cap = this cap,
    META_REGEN_KEYS), every other item is COPIED forward with its source row verbatim (META_COPIED_KEYS, regenerated false,
    source_cap kept) — a genuinely mixed-cap store whose copied rows' attempts live on the SOURCE root. `texts` =
    {item_id: response text} forces the served text per item (two items sharing ONE text is the GenMeta item-identity case).
    The MMLU prefill rows go to rows.jsonl (task `s2parity/mmlu_letter`, seed 0 greedy, ids on the row)."""
    from sisteer.capeval.tiers import judged
    from sisteer.eval.sampled_controls import GenMeta, META_COPIED_KEYS, _fork_fields, _meta_stats
    if (stage_cap is None) != (source is None):
        raise ValueError("a rung controls store (stage_cap) is written FROM its source cell dir (source), as the writer does")
    vmodel = root.vmodels[arm]
    cell = _tinycap_cell(root, arm) if not stage_cap else root.path / "exp" / f"controls_cap{stage_cap}" / f"{ac.MODEL_KEY}_{root.view}_t{trial}" / "cells" / f"{arm.replace('__', '--')}--controls"
    cap = stage_cap or cap
    gen_stats: Dict[str, Any] = {}
    at_cap = dict(at_cap) if at_cap else {}
    regenerate = dict(regenerate or {})
    for control, items in items_by_control.items():
        task = ac.CONTROLS_TASK_OF[control]
        gen_cfg = {"max_new_tokens": cap, "do_sample": True, "batch_size": 8, **VENDOR_BODY, "seed_base": s, "trial": trial, "arm": "vendor_recommended"}
        header = {"task": task, "task_version": "judged_v1", "intervention_hash": _sha(arm), "model_revision": MODEL_REVISION, "gen_cfg": gen_cfg,
                  "system_sha": None, "items_sha": _sha("items")}
        spool_path = cell / "private" / f"gens_{control}.jsonl"
        spool = judged.GenSpool(spool_path, header)
        meta = GenMeta(spool_path)
        src_texts: Dict[str, str] = {}
        src_bound: Dict[str, dict] = {}
        if source is not None:
            src_path = source / "private" / f"gens_{control}.jsonl"
            src_spool = judged.GenSpool(src_path, ac.read_jsonl(src_path)[0]["__spool_header__"], create=False)
            src_texts = src_spool.entries()
            src_bound, _ = GenMeta(src_path, repair_torn=False).bound(src_texts)
        regen_ids = set(regenerate.get(control, ())) if source is not None else set()
        for it in items:
            if source is not None and it not in regen_ids:
                row = {k: src_bound[it][k] for k in META_COPIED_KEYS}
                row["regenerated"] = False
                row["floor_sample"] = False
                meta.append(row)                     # the writer's copy-forward: the source row verbatim, source_cap KEPT
                spool.append(it, src_texts[it])
                continue
            seed = ac.controls_sampling_seed(s, trial, control, it)
            fr = "length" if it in at_cap.get(control, ()) else "stop"
            # a re-drawn row is a NEW draw: its text differs from the source's (as the writer's re-draw does), so the text hash can
            # tell the rung's response from the base's
            redrawn = f"{src_texts[it]} (re-drawn at {cap})" if source is not None else None
            rid, cid, text = root.send(task, it, trial, vmodel, root.vendor_body(vmodel, it, cap, seed), finish_reason=fr, s=s,
                                       text=(texts or {}).get(it, redrawn))
            ct = cap if fr == "length" else 17
            row = {"item_id": it, "sampling_seed": seed, "max_tokens": cap, "finish_reason": fr, "completion_tokens": ct, "prompt_tokens": 40,
                   "at_cap": fr == "length", "regenerated": source is not None, "floor_sample": False, "source_cap": cap, "text_sha256": _sha(text)}
            if source is not None:
                row.update(_fork_fields(src_texts[it], text, _ws_tok, src_bound[it], ct))
            meta.append(row)
            spool.append(it, text)
        bound_now, n_superseded = meta.bound(spool.entries())
        stats = {"control": control, "n_new": len(items), "n_total": len(items), "spool": str(spool.path),
                 "finish_reasons": dict(sorted(Counter(r["finish_reason"] for r in bound_now.values()).items()))}
        stats.update(_meta_stats(bound_now, cap=cap, gen=gen_cfg, n_superseded=n_superseded, meta_path=meta.path))
        gen_stats[control] = stats
    rows = []
    for it in mmlu_items:
        body = {"model": vmodel, "messages": [{"role": "user", "content": f"mmlu {it}"}, {"role": "assistant", "content": "Answer:"}],
                "max_tokens": 1, "temperature": 0.0, "seed": 0, "logprobs": True, "top_logprobs": 64, "continue_final_message": True,
                "add_generation_prompt": False}
        rid, cid, _ = root.send(ac.MMLU_TASK, it, 0, vmodel, body, finish_reason="length", completion_tokens=1, s=s)
        rows.append({"task": ac.MMLU_TASK, "item_id": it, "metric": "acc", "value": 1.0, "domain": "knowledge", "request_id": rid, "completion_id": cid})
    if mmlu_items:  # the parity emitter's aggregate row is MMLU's only completion artefact (not a tinycap task: no done marker)
        rows.append({"task": ac.MMLU_TASK, "item_id": C.AGG_ITEM_ID, "metric": "acc", "value": 1.0, "domain": "knowledge"})
    _jsonl(cell / "rows.jsonl", rows)
    _json(cell / "manifest.json", {"cell_id": arm})
    _json(cell / "gen_stats.json", gen_stats)
    return cell


# --------------------------------------------------------------------------- #
# whole-wave builder                                                            #
# --------------------------------------------------------------------------- #

DEFAULT_ARMS = ("none", "constv2-mix-full__psr_ce_all", "prompted__si_constitution")
#: the REAL store root of a GSM8K child (`seedrep_launch_plan.GSM8K_OUT_REL`, held equal by tests/test_seedrep_fz2_ledger_ownership.py):
#: its `--out-root`, i.e. its request-evidence dir (`child_store_root`) and the dir its usage ledger is started in (`gsm8k_ledger_argv`)
GSM8K_OUT_REL = "exp/sampled/gsm8k_thinking_on_cap{cap}__s{s}/{cell}"
GSM8K_FIXTURE_CAP = 8192
GEOMETRY = {**{f: (45, 11520) for f in ac.STATIC_FAMILIES}, "bon": (45, 11520), "controls": (70, 7424), "gsm8k": (31, 16640),
            "petri": (15, 32768), "misalign": (21, 24576), "pair": (21, 24576), "crescendo": (10, 49152), "actorattack": (10, 49152)}


def make_root(stores: Path, *, leg: str, view: str, arms: Sequence[str] = DEFAULT_ARMS, stage: str = "base", shard: str = "shard00",
              arditi_pin: Optional[str] = None, harness_of: Optional[Mapping[str, str]] = None) -> Root:
    S, L = GEOMETRY[leg]
    if stage != "base":
        S, L = {"2048": (51, 10240), "8192": (31, 16384), "16384": (21, 24576)}.get(stage, (S, L))
    thinking = "on" if leg == "gsm8k" else "off"
    return Root(stores / leg / view / shard if stage == "base" else stores / leg / view / f"{shard}_cap{stage}", leg=leg, view=view,
                arms=arms, S=S, L=L, shard=shard, stage=stage, thinking=thinking, arditi_pin=arditi_pin,
                max_logprobs=64 if leg == "controls" else 20, harness_of=harness_of)


def populate_root(root: Root, *, s: int, arms: Sequence[str] = DEFAULT_ARMS, n_items: int = 3, complete: bool = True,
                  at_cap: Mapping[str, Sequence[str]] = ()) -> Dict[str, Path]:
    """Every table leg of the root's container leg, one store per arm, `n_items` items each."""
    items = [f"item{k}" for k in range(n_items)]
    goals = [f"g{k}" for k in range(n_items)]
    out: Dict[str, Path] = {}
    at_cap = dict(at_cap) if at_cap else {}
    for arm in arms:
        leg = root.leg
        if leg in ac.STATIC_FAMILIES:
            out[arm] = write_static_store(root, family=leg, arm=arm, s=s, goals=goals, at_cap_goals=at_cap.get(leg, ()))
        elif leg == "bon":
            out[arm] = write_bon_store(root, arm=arm, s=s, goals=goals, at_cap_goals=at_cap.get(leg, ()))
        elif leg in ac.MULTITURN_ATTACKS:
            out[arm] = write_multiturn_store(root, attack=leg, arm=arm, s=s, goals=goals)
        elif leg == "misalign":
            beh = {f"sc{k}": ("leaking" if k % 2 == 0 else "murder") for k in range(n_items * 2)}
            out[arm] = write_misalign_store(root, arm=arm, s=s, items=beh, at_cap_items=at_cap.get("misalign", ()),
                                            loop_items=at_cap.get("misalign_loop", ()))
        elif leg == "petri":
            scen = {f"sc{k}": PETRI_BEHAVIOURS[k % 2] for k in range(n_items * 2)}
            out[arm] = write_petri_store(root, arm=arm, s=s, scenarios=scen)
        elif leg == "gsm8k":
            out[arm] = write_gsm8k_store(root, arm=arm, s=s, items=items, at_cap_items=at_cap.get("gsm8k", ()))
        elif leg == "controls":
            out[arm] = write_controls_store(root, arm=arm, s=s, items_by_control={"xstest": items, "or_bench_hard_1k": [f"ob{k}" for k in range(n_items)]},
                                            mmlu_items=[f"mm{k}" for k in range(n_items)],
                                            at_cap={c: at_cap.get(c, ()) for c in ac.CONTROLS})
        else:
            raise ValueError(leg)
    root.flush_pipeline(complete=complete, result_rc="0" if root.leg == "misalign" else None)   # a GSM8K root is laid out per child store there
    return out


def expected_file(path: Path, *, arms: Sequence[str] = DEFAULT_ARMS, views: Sequence[str] = ("s1",), arditi_seeds: Sequence[int] = (),
                  legs: Sequence[str] = ac.LEGS) -> Path:
    runs = [{"arm": a, "view": v, "s": ac.seed_of_view(v)} for v in views for a in arms]
    runs += [{"arm": ac.ARDITI_CELL, "view": ac.ARDITI_VIEW, "s": s} for s in arditi_seeds]
    _json(path, {"legs": list(legs), "arm_runs": runs})
    return path


def _plan_row(*, leg: str, view: str, stage: str, shard: int, nshards: int, arms: Sequence[str], W: str, wave_uuid: str) -> Dict[str, Any]:
    """One `standups[]` / `rungs[]` row in the generator's shape (`seedrep_launch_plan._row`): the keys the audit reads
    plus the identity fields — `arm_runs[].cell` is the served cell, `.arm` the plan's label (Arditi: ar_c0p3__arditi_allpos)."""
    seeds = [ac.seed_of_view(view)] if view != ac.ARDITI_VIEW else [1, 2, 3]
    arm_runs = [{"arm": (ARDITI_PLAN_ARM if a == ac.ARDITI_CELL else a), "cell": a, "s": s, "object": None} for a in arms for s in seeds]
    return {"standup_id": standup_id(leg, view, stage, shard), "kind": "base" if stage == "base" else "rung", "leg": leg,
            "eval_legs": ac.table_legs_of(leg), "engine": view, "view": view, "stage": str(stage),
            "shard": int(shard), "nshards": int(nshards), "seed_bases": seeds,
            "control_key": control_key(W[:7], wave_uuid, leg, view, stage, shard),
            "modal": {"function": "run_rung" if stage != "base" else ("run_arditi_leg" if view == ac.ARDITI_VIEW else "run_view_leg"),
                      "gpu": "H200", "retries": 0, "max_containers": 1},
            "arm_runs": arm_runs, "producers": [], "expected_h200_h": [0.5, 1.0]}


def launch_plan_file(path: Path, roots: Sequence[Root], *, W: str = W_FIXTURE, wave_uuid: str = WAVE_UUID,
                     extra: Sequence[Tuple[str, str, str, int, Sequence[str]]] = (),
                     extra_arms: Optional[Mapping[Tuple[str, str, str, int], Sequence[str]]] = None) -> Path:
    """`RS/launch_plan.json` (schema seedrep_launch_plan/1) declaring one standup per fixture root (+ `extra` rows
    `(leg, view, stage, shard, arms)` with NO root — the absent-planned-shard case; `extra_arms` = arms ADDED to the row
    of `(leg, view, stage, shard)` — a standup the plan says serves an arm its attempt never ran); `nshards` counts the
    rows per (leg, view, stage). Hand-shaped to the generator's row schema: the real generator needs the roster, pins,
    arms table and wave.json and is not importable from this tree — the gate runs ITS output through
    `audit_common.read_launch_plan`."""
    specs = [(r.leg, r.view, r.stage, r.shard_index, list(r.arms) + list((extra_arms or {}).get((r.leg, r.view, r.stage, r.shard_index), ())))
             for r in roots] + [tuple(e) for e in extra]
    counts: Dict[Tuple[str, str, str], int] = {}
    for leg, view, stage, _shard, _arms in specs:
        counts[(leg, view, stage)] = counts.get((leg, view, stage), 0) + 1
    rows = [_plan_row(leg=leg, view=view, stage=stage, shard=shard, nshards=counts[(leg, view, stage)], arms=arms, W=W, wave_uuid=wave_uuid)
            for leg, view, stage, shard, arms in specs]
    plan = {"schema": LAUNCH_PLAN_SCHEMA, "materialisation": "final", "W": W, "W7": W[:7], "wave_uuid": wave_uuid,
            "model_key": ac.MODEL_KEY, "standups": [r for r in rows if r["kind"] == "base"], "rungs": [r for r in rows if r["kind"] == "rung"],
            "arm_runs_per_leg": 27}
    _json(path, plan)
    return path


def s5_reference_record(*, L: int, S: int, thinking: bool, tok_s: float, W: str = W_FIXTURE, wave_uuid: str = WAVE_UUID) -> Dict[str, Any]:
    """ONE S5 reference record in the schema `utilisation_table.S5_REFERENCE_KEYS` defines (no producer exists on this
    tree, so the record is validated through the CONSUMER's rules — `s5_reference_problems` must be empty — the only
    rules there are)."""
    import utilisation_table as ut
    rec = {"L": int(L), "S": int(S), "thinking": bool(thinking), "method": ut.S5_REFERENCE_METHOD, "tok_s_aggregate": float(tok_s),
           "duration_s": 1500.0, "store_complete": True, "sampling_coverage": 0.97, "n_eligible_intervals": 148, "W": W, "wave_uuid": wave_uuid,
           "attempt_id": f"ta-s5-{L}x{S}-{'on' if thinking else 'off'}", "source": "s5_smoke"}
    problems = ut.s5_reference_problems(rec, W=W, wave_uuid=wave_uuid)
    assert not problems, problems
    return rec


def s5_references(*, tok_s: float = 2000.0, W: str = W_FIXTURE, wave_uuid: str = WAVE_UUID) -> Dict[str, Any]:
    """`RS/smokes.json`'s shape: `{"S5": {"references": [...]}}` over every fixture window class (the rung windows too)."""
    seen = set()
    refs = []
    for leg, (S, L) in GEOMETRY.items():
        key = (L, S, leg == "gsm8k")
        if key not in seen:
            seen.add(key)
            refs.append(s5_reference_record(L=L, S=S, thinking=key[2], tok_s=tok_s, W=W, wave_uuid=wave_uuid))
    for S, L in ((51, 10240), (31, 16384), (21, 24576)):
        if (L, S, False) not in seen:
            seen.add((L, S, False))
            refs.append(s5_reference_record(L=L, S=S, thinking=False, tok_s=tok_s, W=W, wave_uuid=wave_uuid))
    return {"S5": {"references": refs, "multiturn": []}}


def s5_file(path: Path, *, tok_s: float = 2000.0, W: str = W_FIXTURE, wave_uuid: str = WAVE_UUID) -> Path:
    _json(path, s5_references(tok_s=tok_s, W=W, wave_uuid=wave_uuid))
    return path


def expected_hours_file(path: Path, *, hours: float = 3.0) -> Path:
    _json(path, {leg: hours for leg in ac.CONTAINER_LEGS})
    return path


def build_wave(stores: Path, *, legs: Sequence[str] = ac.CONTAINER_LEGS, views: Sequence[str] = ("s1",), arms: Sequence[str] = DEFAULT_ARMS,
               n_items: int = 3, arditi: bool = False, arditi_pin: Optional[str] = None) -> Dict[Tuple[str, str], Root]:
    """A minimal complete wave: every container leg × view (+ the Arditi engine at s ∈ {1,2,3} when `arditi`)."""
    roots: Dict[Tuple[str, str], Root] = {}
    for leg in legs:
        for view in views:
            r = make_root(stores, leg=leg, view=view, arms=arms)
            populate_root(r, s=ac.seed_of_view(view) or 1, arms=arms, n_items=n_items)
            roots[(leg, view)] = r
        if arditi:
            r = make_root(stores, leg=leg, view=ac.ARDITI_VIEW, arms=(ac.ARDITI_CELL,), arditi_pin=arditi_pin)
            for s in (1, 2, 3):
                _populate_arditi(r, s=s, n_items=n_items)
            r.flush_pipeline(complete=True, result_rc="0" if leg == "misalign" else None)
            roots[(leg, ac.ARDITI_VIEW)] = r
    return roots


def _populate_arditi(root: Root, *, s: int, n_items: int) -> None:
    goals = [f"g{k}" for k in range(n_items)]
    arm = ac.ARDITI_CELL
    leg = root.leg
    if leg in ac.STATIC_FAMILIES:
        write_static_store(root, family=leg, arm=arm, s=s, goals=goals)
    elif leg == "bon":
        write_bon_store(root, arm=arm, s=s, goals=goals)
    elif leg in ac.MULTITURN_ATTACKS:
        write_multiturn_store(root, attack=leg, arm=arm, s=s, goals=goals)
    elif leg == "misalign":
        pass  # the Arditi misalign root runs the runner three times into three namespaces; the fixture covers the view roots
    elif leg == "petri":
        pass
    elif leg == "gsm8k":
        pass
    elif leg == "controls":
        pass
