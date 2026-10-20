"""``scripts/sdfbase/lora_serving_up.py`` — the LORA-NATIVE standup,
CPU-only.

The REAL ``bring_up_lora`` on the pinned full22 campaign configs (gemma's
``constv2ctl2-full__lora_r4_ce`` — the certification's own gemma target,
kind ``lora_all`` — plus the admitted ``prompted__safety_refusal``) over
a view-shaped synthetic bank whose target cell is a REAL ``lora_all`` blob
sized off the model's o_proj modules (``lora_export.oproj_target_modules``
on the cached config): every derivation is real — the lora-native cell
source, ``enumerate_cells(lens_controls=False)``, A's bank export, the
version-2 table + on-disk block re-hash, C's composer and its readiness
attestation against a scripted lora-native upstream, D's proxy with the
engine attestation on a REAL aiohttp thread, and the five positive controls
over HTTP. Only the ``vllm serve`` child is replaced: the composed launch
runs with its argv swapped for the scripted stand-in (``_FAKE_UPSTREAM``,
C's ``_FAKE_SERVER`` extended with ``/v1/chat/completions``), so the REAL
``launch_server`` performs readiness and writes the real attestation.

Skips (never fetches) when the pinned tokenizer/config is not in the local
HF cache. Every negative pins ONE deviation: a zero adapter that is not a
no-op (exit 4), a mixed batch whose ENGINE BOOKKEEPING binds the zero
adapter's requests to the genuine adapter (exit 4, the structural selection
verdict), a recorder that saw nothing (exit 4), a PSR cell (exit 2), a
swapped object (exit 2, before any export), a version-1 table (exit 2), a
dead engine (exit 3). An OUTPUT-only disagreement in a concurrent batch with
honest bookkeeping (the job-489308 shape) PASSES and is recorded."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_petri_serving_gate import _native_residual_report, _native_sched_report  # noqa: E402
from test_sdfbase_full22_wave import DERIVED_ARMS, _stage, _stage_derived  # noqa: E402
from test_serve_declarations_attestation import _cached_tok  # noqa: E402
from lora_snapshot_fixtures import snapshot_like  # noqa: E402

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.capeval.serving import geometry as sg  # noqa: E402,F401  (imported BEFORE any test monkeypatches vs.build_server_launch: geometry reads its signature at import)
from sisteer.capeval.serving import lora_export as lx  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402
from sisteer.eval.defense_cell import arm_of, sha256_file  # noqa: E402

#: the certification's gemma target: fit-lane id -> harness id
GEMMA_CELL = "constv2ctl2-full__lora_r4_ce"
GEMMA_HCELL = "si_constv2ctl2_full__lora_r4_ce"
PROMPTED = "prompted__safety_refusal"
PROMPTED_H = "prompted__safety"
PSR_CELL = "constv2ctl2-full__psr_ce_all"
ZERO = dc.NOOP_LORA_CELL_ID
BASE_IDS = [11, 12, 13, 14, 15]
GENUINE_IDS = [21, 22, 23, 24, 25]
OTHER_IDS = [31, 32, 33]
EVAL = "families_served:short"


def _load_lsu():
    spec = importlib.util.spec_from_file_location(
        "_lora_serving_up_under_test", REPO / "scripts/sdfbase/lora_serving_up.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_lora_serving_up_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lsu():
    return _load_lsu()


# --------------------------------------------------------------------------- #
# the scripted LORA-NATIVE upstream (C's _FAKE_SERVER + chat completions)      #
# --------------------------------------------------------------------------- #

#: argv: port model conf_path. The conf carries the readiness answers exactly as
#: C's fake (models_data with LoRA roots, scheduler_results) plus a CHAT table
#: keyed by the UPSTREAM model name the proxy sends (the base id, the adapter
#: names): {"ids": [...], "text": "..."}. ``prompt_ids_by_key`` maps the
#: canonical JSON of a messages list to the prompt ids echoed back (the gate's
#: render-agreement check reads them). Two scripted deviations: ``zero_differs``
#: (the zero adapter answers OTHER ids — the no-op certificate must fail) and
#: ``contaminate_zero_when_concurrent`` (the zero adapter answers the GENUINE
#: adapter's ids whenever another request is in flight while its BOOKKEEPING stays
#: honest — the job-489308 shape: solo calls agree, the concurrent batch does not;
#: the selection control RECORDS this and passes), and
#: ``misroute_zero_when_concurrent`` (the same, but the engine's bookkeeping ALSO
#: binds the zero request to the genuine adapter — a real misroute the structural
#: ``recorder_broken`` makes the recorder arm but record nothing (everything
#: missing). The fake mints a unique EXTERNAL ``chatcmpl-*`` id per response and,
#: like the engine's ``InputProcessor.assign_request_id``, an INTERNAL id
#: ``<external>-<8 hex>`` under which its per-process recorder records the
#: request — the check joins them (``serving_attestation.is_engine_id_of``). The
#: three ``sisteer_*_lora_routing_*`` RPCs carry the CLIENT-minted EPOCH: arm
#: adopts it and resets (idempotent for the live one, ``already_armed`` for
#: another); observe/disarm refuse another — every refusal an ``ok: False``
#: PAYLOAD in a 200 body, as the worker answers it (the client's reader refuses).
#: A "step" is one request's service moment: the set of INTERNAL ids
#: in flight (``chat_delay_s`` makes the standup's concurrent batches overlap
#: deterministically, so the check finds >= 2 of the control's selections in one
#: step). The payload names the V2 runner class (canary #4 ran V2).
_FAKE_UPSTREAM = r"""
import json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port, model, conf_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
conf = json.loads(open(conf_path).read())
threading.Timer(float(conf.get("lifetime_s", 120)), lambda: os._exit(0)).start()
TOKEN = os.environ.get("VLLM_API_KEY", "")
LOCK = threading.Lock()
INFLIGHT = [0]
INFLIGHT_NAMES = []
INFLIGHT_EIDS = []           # the INTERNAL (engine) ids of the in-flight requests
LORA_IDS = {k: int(v) for k, v in (conf.get("lora_int_ids") or {}).items()}
COUNTER = [0]
ARMED = [False]
EPOCH = [None]
ROUTING = {}                 # engine request id -> {"lora_int_id", "lora_name"}
INCONSISTENT = {}
STEPS_LIST = []              # one sorted engine-id list per "step"
RUNNER_CLASS = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
# THE PER-REQUEST FAILURE LEDGER (native-psr-serving §1.8, fact 15), scripted: a chat for a
# model named in conf["failures_by_model"] (optionally only when its messages carry that
# entry's "marker") records an UNLOCATED-boundary entry under the response's EXTERNAL id;
# `sisteer_take_request_failure` claims-and-clears it, or answers the FAULT payload for every
# id when conf["ledger_fault"] is set (the latch, after readiness). The residual report is
# LIVE after the first chat (ledger_size = unclaimed entries, ledger_fault = the conf flag), so
# readiness reads the conf's own values and the standup's post-controls read sees the truth.
LEDGER = {}
SAW_CHAT = [False]
CHAT_LOG = conf.get("chat_log_path")


def _refusal(kind, detail):
    # an EXPECTED refusal rides as an ok: False payload inside a 200 body (review P2,
    # transport): the client's reader refuses on it, the executor's queues stay aligned
    return {"ok": False, "refusal": kind, "detail": detail, "live_epoch": EPOCH[0],
            "runner_kind": "v2", "runner_class": RUNNER_CLASS}


def _summary():
    return {"ok": True, "armed": ARMED[0], "epoch": EPOCH[0], "runner_class": RUNNER_CLASS,
            "runner_kind": "v2", "n_steps": len(STEPS_LIST), "steps_overflow": False,
            "overflow": False, "n_observed": len(ROUTING), "n_inconsistent": len(INCONSISTENT),
            "snapshot_errors": 0, "errors": [], "cap": 50000, "steps_cap": 50000}


def _recorder_payload():
    with LOCK:
        return {**_summary(), "observations": dict(ROUTING), "inconsistent": dict(INCONSISTENT),
                "steps": [list(x) for x in STEPS_LIST]}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# the worker's engine-id join (native_residual.external_request_ids): the external ids an ENGINE id can
# belong to — itself, minus one `-<8 hex>` suffix, and a `<i>_` child's parent forms
_SUFFIX = re.compile(r"-[0-9a-f]{8}\Z")
_CHILD = re.compile(r"\A\d+_")


def _external_ids(eid):
    out = [eid]
    m = _SUFFIX.search(eid)
    if m is not None and m.start() > 0:
        out.append(eid[:m.start()])
    parent = _CHILD.sub("", eid, count=1)
    if parent and parent != eid:
        out.append(parent)
        m = _SUFFIX.search(parent)
        if m is not None and m.start() > 0:
            out.append(parent[:m.start()])
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        if not conf.get("require_token", True):
            return True
        return self.headers.get("Authorization") == "Bearer " + TOKEN

    def do_GET(self):
        if self.path == "/v1/models":
            if not self._authed():
                self._json(401, {"error": "Unauthorized"})
                return
            self._json(200, {"data": conf.get("models_data") or [{"id": model}]})
        else:
            self._json(404, {"detail": "nope"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        try:
            payload = json.loads(body.decode() or "{}")
        except ValueError:
            payload = {}
        if self.path == "/collective_rpc" and conf.get("mount_rpc", True):
            if not self._authed():
                self._json(401, {"error": "Unauthorized"})
                return
            method = payload.get("method")
            args = payload.get("args") or []
            if method == "sisteer_scheduler_config":
                results = conf.get("scheduler_results", [])
            elif method == "sisteer_arm_lora_routing_recorder":
                ep = args[0] if args else None
                if not isinstance(ep, str) or not ep:
                    results = [_refusal("bad_epoch", "epoch must be a non-empty string")]
                else:
                    with LOCK:
                        if ARMED[0] and ep != EPOCH[0]:
                            results = [_refusal("already_armed", "armed by epoch %s" % EPOCH[0])]
                        else:
                            already = ARMED[0]
                            if not already:                   # ADOPT the client's epoch, fresh records
                                ARMED[0] = True
                                EPOCH[0] = ep
                                ROUTING.clear(); INCONSISTENT.clear(); del STEPS_LIST[:]
                            results = [{"ok": True, "armed": True, "epoch": EPOCH[0],
                                        "already_armed": already, "runner_class": RUNNER_CLASS,
                                        "runner_kind": "v2"}]
            elif method == "sisteer_lora_routing_observations":
                ep = args[0] if args else None
                if not isinstance(ep, str) or not ep:
                    results = [_refusal("bad_epoch", "epoch must be a non-empty string")]
                elif EPOCH[0] is None:
                    results = [_refusal("not_armed", "never armed")]
                elif ep != EPOCH[0]:
                    results = [_refusal("epoch_mismatch", "observations asked for %r, live %r" % (ep, EPOCH[0]))]
                else:
                    results = [_recorder_payload()]
            elif method == "sisteer_mamba_backend_report":
                # the PORTED Nemotron backend RPC (native-serving-default): the scripted worker's
                # per-layer report(s), conf-supplied; absent = an engine with no such method (an
                # empty results list the judge refuses as "no report from any worker")
                results = conf.get("mamba_results", [])
            elif method == "sisteer_residual_slot_report":
                # THE RESIDUAL SLOT's readiness RPC (native-psr-serving §1.4): the scripted worker's
                # report(s), conf-supplied from the launcher's own expectation; LIVE ledger fields
                # once any chat was served (the post-controls read, §1.8)
                results = [dict(r) for r in conf.get("residual_results", [])]
                if SAW_CHAT[0]:
                    with LOCK:
                        for r in results:
                            r["ledger_size"] = len(LEDGER)
                            r["ledger_fault"] = bool(conf.get("ledger_fault"))
            elif method == "sisteer_take_request_failure":
                # the sixth RPC (§1.4): claim-and-clear, or the fault payload for ANY id once latched
                rid = str(args[0]) if args else ""
                with LOCK:
                    if conf.get("ledger_fault"):
                        results = [{"fault": "ledger_overflow", "req_id": rid,
                                    "unclaimed": len(LEDGER), "cap": 1024}]
                    else:
                        # the worker's claim: EVERY entry whose ENGINE id belongs to the claimed completion id
                        hits = [e for e in list(LEDGER) if rid in _external_ids(e)]
                        first = None
                        for e in hits:
                            entry = LEDGER.pop(e)
                            first = first or entry
                        results = [None if first is None else dict(first, claimed_engine_ids=hits)]
            elif method == "sisteer_disarm_lora_routing_recorder":
                ep = args[0] if args else None
                if not isinstance(ep, str) or not ep:
                    results = [_refusal("bad_epoch", "epoch must be a non-empty string")]
                elif EPOCH[0] is None:
                    results = [_refusal("not_armed", "never armed")]
                elif ep != EPOCH[0]:
                    results = [_refusal("epoch_mismatch", "disarm asked for %r, live %r" % (ep, EPOCH[0]))]
                else:
                    with LOCK:
                        was, ARMED[0] = ARMED[0], False
                        results = [{**_summary(), "disarmed": True, "was_armed": was, "restored": was}]
            else:
                results = conf.get("results", [])
            n_workers = int(conf.get("rpc_workers", 1))
            self._json(200, {"results": [json.loads(json.dumps(r)) for r in results] * n_workers
                             if results else results})
            return
        if self.path == "/v1/chat/completions":
            if not self._authed():
                self._json(401, {"error": "Unauthorized"})
                return
            self._chat(payload)
            return
        if self.path == "/tokenize":
            if not self._authed():
                self._json(401, {"error": "Unauthorized"})
                return
            ids = conf.get("prompt_ids_by_key", {}).get(canonical(payload.get("messages")), [1, 2, 3])
            self._json(200, {"count": len(ids), "tokens": ids, "max_model_len": 4096})
            return
        self._json(404, {"detail": "Not Found"})

    def _chat(self, payload):
        name = payload.get("model")
        bound = name if name in LORA_IDS else None         # the engine binds the request here
        with LOCK:
            INFLIGHT[0] += 1
            INFLIGHT_NAMES.append(name)
            mixed = any(n == conf.get("genuine_name") for n in INFLIGHT_NAMES)
            COUNTER[0] += 1
            rid = "chatcmpl-fake-%d" % COUNTER[0]                   # the EXTERNAL id (the response's)
            eid = "%s-%s" % (rid, os.urandom(4).hex())             # the engine's INTERNAL id
            INFLIGHT_EIDS.append(eid)
        try:
            time.sleep(float(conf.get("chat_delay_s", 0.0)))
            with LOCK:
                mixed = mixed or any(n == conf.get("genuine_name") for n in INFLIGHT_NAMES)
                step_ids = sorted(INFLIGHT_EIDS)
            table = conf.get("chat", {})
            entry = table.get(name)
            if entry is None:
                self._json(404, {"error": {"message": "The model `%s` does not exist." % name}})
                return
            # OUTPUT contamination (the job-489308 shape): the zero adapter answers the
            # GENUINE adapter's ids whenever a genuine request is in flight beside it —
            # a homogeneous (same-size, single-selection) batch stays honest, the mixed
            # batch does not; the engine's BOOKKEEPING stays honest (bound to zero)
            zero_hit = name == conf.get("zero_name") and mixed
            if (conf.get("contaminate_zero_when_concurrent")
                    or conf.get("misroute_zero_when_concurrent")) and zero_hit:
                entry = table[conf["genuine_name"]]
            # a REAL misroute: the bookkeeping ALSO binds the request to the genuine adapter
            if conf.get("misroute_zero_when_concurrent") and zero_hit:
                bound = conf["genuine_name"]
            if ARMED[0] and not conf.get("recorder_broken"):
                with LOCK:
                    obs = {"lora_int_id": LORA_IDS.get(bound, 0), "lora_name": bound}
                    prev = ROUTING.get(eid)
                    if prev is None:
                        ROUTING[eid] = obs
                    elif prev != obs:
                        INCONSISTENT.setdefault(eid, [prev]).append(obs)
                    STEPS_LIST.append(step_ids)
            ids, text = list(entry["ids"]), entry["text"]
            if payload.get("max_tokens") is not None:
                ids = ids[:int(payload["max_tokens"])]
            with LOCK:
                SAW_CHAT[0] = True
                if CHAT_LOG:
                    with open(CHAT_LOG, "a") as fh:      # every body the upstream SAW
                        fh.write(json.dumps({"model": name, "id": rid, "payload": payload}) + "\n")
                fail = (conf.get("failures_by_model") or {}).get(name)
                scripted = fail is not None and (fail.get("marker") is None
                                                 or fail["marker"] in canonical(payload.get("messages")))
                # the WORKER's reading of a header-less render (O49's boundary_refusal control: a user turn
                # under add_generation_prompt: false): the response header occurs NOWHERE -> UNLOCATED,
                # ledgered under the ENGINE id (production-shaped, never the completion id);
                # conf["boundary_control_broken"] scripts the broken join / worker: nothing is ledgered
                headerless = (payload.get("add_generation_prompt") is False
                              and not payload.get("continue_final_message")
                              and not conf.get("boundary_control_broken"))
                if scripted or headerless:
                    LEDGER[eid] = {"req_id": eid, "external_req_id": rid, "lora_int_id": LORA_IDS.get(name, 0),
                                   "cell": name,
                                   # conf["boundary_control_error_type"] scripts an UNRELATED worker failure on
                                   # the header-less probe (the pass-for-the-wrong-reason known-bad)
                                   "error_type": (conf.get("boundary_control_error_type") if headerless and not scripted
                                                  else None) or "ResponseBoundaryUnlocated",
                                   "prompt_len": 7, "header_len": 4,
                                   "detail": ((fail or {}).get("detail") if scripted else None)
                                   or "the assistant header ids occur NOWHERE in the prompt ids"}
            choice = {"index": 0, "finish_reason": "stop",
                      "message": {"role": "assistant", "content": text}}
            if payload.get("return_token_ids"):
                choice["token_ids"] = ids
            if payload.get("logprobs"):
                if payload.get("return_tokens_as_token_ids"):
                    # the engine's `token_id:<int>` spelling (proxy_engine.first_position_logprobs_by_id
                    # refuses any other), with a COMMON filler support {1..4} whose logprobs depend on
                    # the entry's ids — two models answering the same ids give bit-identical payloads
                    # (the zero-payload known-bad), different ids a divergence above the noise floor
                    def _top(t):
                        top = (conf.get("top_logprobs_by_model") or {}).get(name)
                        if top is not None:
                            return [{"token": "token_id:%d" % int(k), "logprob": float(v)}
                                    for k, v in top.items()]
                        return ([{"token": "token_id:%d" % t, "logprob": -0.1}]
                                + [{"token": "token_id:%d" % f, "logprob": -1.0 - 0.01 * ids[0] - 0.1 * f}
                                   for f in (1, 2, 3, 4)])
                    choice["logprobs"] = {"content": [
                        {"token": "token_id:%d" % t, "logprob": -0.1, "top_logprobs": _top(t)}
                        for t in ids]}
                else:
                    choice["logprobs"] = {"content": [
                        {"token": "t%d" % t, "logprob": -0.1,
                         "top_logprobs": [{"token": "t%d" % t, "logprob": -0.1},
                                          {"token": "u", "logprob": -3.0}]} for t in ids]}
            out = {"id": rid, "object": "chat.completion", "created": 0, "model": name,
                   "choices": [choice],
                   "usage": {"prompt_tokens": 3, "completion_tokens": len(ids),
                             "total_tokens": 3 + len(ids)}}
            if payload.get("return_token_ids"):
                out["prompt_token_ids"] = conf.get("prompt_ids_by_key", {}).get(
                    canonical(payload.get("messages")), [1, 2, 3])
            self._json(200, out)
        finally:
            with LOCK:
                INFLIGHT[0] -= 1
                INFLIGHT_NAMES.remove(name)
                INFLIGHT_EIDS.remove(eid)


srv = ThreadingHTTPServer(("127.0.0.1", port), H)
srv.daemon_threads = True
srv.serve_forever()
"""


#: the REAL launcher, captured once: every fake below swaps the child's argv and
#: hands the launch to THIS, so two fakes installed in one test never chain
_REAL_LAUNCH_SERVER = vs.launch_server


def fake_upstream_launch_server(tmp_path, *, chat, prompt_ids_by_key=None, zero_name=ZERO,
                                genuine_name=None, zero_differs=False,
                                contaminate_zero_when_concurrent=False, chat_delay_s=0.0,
                                misroute_zero_when_concurrent=False, recorder_broken=False,
                                rpc_workers=1, captured=None, mamba_results=None,
                                shortfall_first=None, residual_over=None,
                                failures_by_model=None, ledger_fault=False, chat_log_path=None,
                                top_logprobs_by_model=None, boundary_control_broken=False,
                                boundary_control_error_type=None):
    """A stand-in for ``vllm_server.launch_server``: keeps the REAL composed
    launch (argv is what the composer emitted; every adapter dir real) but
    swaps its command for the scripted upstream, writes the upstream's conf
    from the launcher's OWN expectations (so readiness agrees by construction
    — the fake's `/v1/models` roots are the composed dirs, its scheduler report
    is ``expected_scheduler_state(launch)`` + the native report per adapter),
    and calls the real ``launch_server`` — real readiness, real attestation."""
    real = _REAL_LAUNCH_SERVER          # never a previously installed fake (they would chain)
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake_upstream.py"
    script.write_text(_FAKE_UPSTREAM)

    def _launch(launch, *, log_path, **kw):
        exp_lora = sa.expected_native_lora_state_for(launch.lora_modules, launch.lora_adapter_dirs)
        # THE RESIDUAL SLOT (native-psr-serving): the fake worker's report agrees with the
        # launcher's own expectation by construction (unarmed on a LoRA-only launch)
        exp_residual = sa.expected_residual_slot_state(launch.lora_modules, launch.lora_adapter_dirs)
        residual_results = [_native_residual_report(exp_residual, **(residual_over or {}))]
        expected = vs.expected_scheduler_state(launch)
        budget = int(launch.max_num_seqs) * int(launch.max_model_len)
        # the fake worker's RAW module names spelled from the adapters' canonical keys (``L<N>.<site>.o_proj``
        # -> ``model.layers.<N>.<site>.o_proj``): gemma / llama / qwen sit under ``self_attn``, Nemotron's
        # attention layers under ``mixer`` — the launcher canonicalises both to the attested keys
        def _raw(key):
            n, site = key[1:].split(".", 1)
            return f"model.layers.{n}.{site}"
        exec_names = sorted({_raw(k) for a in exp_lora["adapters"].values() for k in a["module_keys"]})
        rep = _native_sched_report(exp_lora, max_lora_rank=launch.max_lora_rank, **expected,
                                   kv_num_gpu_blocks=math.ceil(budget / 16) + 8,
                                   kv_block_size=16, kv_num_cache_groups=1,
                                   lora_execution_module_names=exec_names)
        for a in rep.get("lora_adapters") or []:
            a["raw_module_names"] = [_raw(k) for k in a["module_keys"]]
            # the exact-name bindings vLLM's own lookup resolves (serving_attestation's P1 rule): the gate's
            # builder derived them from its Llama `self_attn` spelling BEFORE this override, so a `mixer`
            # (Nemotron) worker must re-bind every execution module to ITS factor of the same exact name
            factors = set(a["raw_module_names"])
            a["execution_bindings"] = {m: (m if m in factors else None) for m in exec_names}
        table = {k: dict(v) for k, v in chat.items()}
        if zero_differs and zero_name in table:
            table[zero_name] = {"ids": list(OTHER_IDS), "text": "an OTHER reply"}
        conf = {"results": [], "scheduler_results": [rep],
                "models_data": ([{"id": launch.model}]
                                + [{"id": n, "root": launch.lora_adapter_dirs[n]}
                                   for n in launch.lora_modules]),
                "mount_rpc": True, "require_token": True, "chat": table,
                "prompt_ids_by_key": dict(prompt_ids_by_key or {}),
                "zero_name": zero_name, "genuine_name": genuine_name,
                "contaminate_zero_when_concurrent": bool(contaminate_zero_when_concurrent),
                "misroute_zero_when_concurrent": bool(misroute_zero_when_concurrent),
                "recorder_broken": bool(recorder_broken), "rpc_workers": int(rpc_workers),
                # the engine's name -> lora_int_id binding, exactly the launcher's expectation
                # (the frontend's static registration order = the attestation's id_name_map)
                "lora_int_ids": {n: int(exp_lora["adapters"][n]["lora_int_id"])
                                 for n in launch.lora_modules},
                "chat_delay_s": float(chat_delay_s),
                "mamba_results": list(mamba_results or []),
                "residual_results": residual_results,
                # native-psr-serving §1.8: the scripted per-request failure ledger (see _FAKE_UPSTREAM)
                "failures_by_model": dict(failures_by_model or {}),
                "ledger_fault": bool(ledger_fault),
                "chat_log_path": (str(chat_log_path) if chat_log_path else None),
                "top_logprobs_by_model": dict(top_logprobs_by_model or {}),
                # O49: the scripted BROKEN join — a header-less render is served and NOT ledgered
                "boundary_control_broken": bool(boundary_control_broken),
                "boundary_control_error_type": boundary_control_error_type}
        conf_path = tmp_path / f"fake_upstream_{launch.port}_{len(launches)}.json"
        conf_path.write_text(json.dumps(conf))
        fake = launch.__class__(**{**launch.__dict__,
                                   "cmd": (sys.executable, str(script), str(launch.port),
                                           launch.model, str(conf_path))})
        launches.append(launch)
        if captured is not None:
            captured["launch"] = launch
            captured["launches"] = list(launches)
        if shortfall_first is not None and len(launches) == 1:
            # the FIRST launch refuses readiness for the ONE reason a KV-derived S may answer:
            # every pin held, the container's KV cache backs fewer than S full-length rows
            Path(log_path).write_text("engine log of the first (KV-short) launch\n")
            conc = float(shortfall_first)
            raise vs.KvCoverageShortfall(
                f"KV cache backs only {conc:.2f}-way concurrency at L={launch.max_model_len}",
                kv_backed_concurrency=conc, kv_capacity_tokens=int(conc * int(launch.max_model_len)),
                basis=vs.KV_BASIS_UNIFORM, max_model_len=int(launch.max_model_len),
                max_num_seqs=int(launch.max_num_seqs))
        return real(fake, log_path=log_path, ready_timeout_s=60, install_timeout_s=30)

    launches: list = []

    return _launch


# --------------------------------------------------------------------------- #
# the synthetic bank view                                                      #
# --------------------------------------------------------------------------- #


def _campaign(key):
    cfg_path = REPO / f"configs/sdfbase_full22_{key}.yaml"
    man_path = REPO / f"configs/capeval/manifests/petri_full22_{key}.yaml"
    return cfg_path, man_path, yaml.safe_load(cfg_path.read_text())


def _blob_bytes(d) -> bytes:
    buf = io.BytesIO()
    torch.save(d, buf)
    return buf.getvalue()


def lora_all_blob(model: str, revision: str, *, rank: int = 4, seed: int = 7,
                  layers: str = "uniform") -> dict:
    """A REAL ``lora_all`` blob for ``model``: one (A [d_in, r], B [d_out, r])
    fp32 pair per o_proj module of the cached config, sized by
    ``lora_export.oproj_target_modules``. ``layers="uniform"`` keeps the layers
    sharing the model's MOST COMMON o_proj shape (gemma-4-E4B's 35 local-
    attention layers; every layer on nemotron / qwen) — the subset A's export
    admits today; ``"all"`` is the campaign object's real layer set (see
    ``test_the_real_gemma_target_object_is_refused_by_the_export_today``)."""
    from collections import Counter

    modules = lx.oproj_target_modules(model, revision)
    shapes = {int(n.split(".layers.")[1].split(".")[0]): s for n, s in modules.items()}
    if layers == "uniform":
        modal = Counter(shapes.values()).most_common(1)[0][0]
        keep = sorted(l for l, s in shapes.items() if s == modal)
    else:
        keep = sorted(shapes)
    g = torch.Generator().manual_seed(seed)
    A, B = {}, {}
    for layer in keep:
        d_in, d_out = shapes[layer]
        A[str(layer)] = torch.randn(d_in, rank, generator=g)
        B[str(layer)] = torch.randn(d_out, rank, generator=g) * 0.1
    return {"kind": "lora_all", "layers": keep, "rank": rank, "A": A, "B": B,
            "positions": "all", "response_header_len": None}


from _env import host_site_path  # noqa: E402
REAL_GEMMA_OBJECT = (host_site_path("$STAGE/sdfbase/full22_objects/gemma4e4b/cells")      # the grading host's bank (absent elsewhere: skips)
                     / "base__constv2ctl2-full__lora_r4_ce/object.pt")


def _gemma_layer_widths(model: str, revision: str) -> dict:
    """gemma-4-E4B's o_proj ``in_features`` per layer off the cached config:
    2048 on the 35 local-attention layers, 4096 on the 7 global ones."""
    modules = lx.oproj_target_modules(model, revision)
    return {int(n.split(".layers.")[1].split(".")[0]): s[0] for n, s in modules.items()}


def test_the_real_gemma_target_object_exports_under_its_own_per_layer_shapes(tmp_path):
    """The certification's REAL gemma object (pins sha 8c0f6d97..., a 42-layer
    ``lora_all`` blob whose A is [2048, 4] on the 35 local-attention layers and
    [4096, 4] on the 7 global ones) exports: each layer's factors are held to
    ITS o_proj module's shape, and C's launcher-side fingerprint accepts
    the result. This was E's strict-xfail tripwire until the export dropped
    its one-shape-across-layers premise."""
    from collections import Counter

    from safetensors.torch import load_file

    from sisteer.capeval.serving import worker_reports as wr

    if not REAL_GEMMA_OBJECT.is_file():
        pytest.skip(f"the staged gemma object is not at {REAL_GEMMA_OBJECT}")
    _, _, cfg = _campaign("gemma4e4b")
    cached_or_skip(cfg)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    assert sha256_file(REAL_GEMMA_OBJECT).startswith("8c0f6d97")
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_all", "positions": "all", "rank": 4, "intervention_hash": "ab" * 8},
        REAL_GEMMA_OBJECT.read_bytes(), tmp_path / "adapter",
        base_model=cfg["model"], base_revision=cfg["model_revision"],
        # a snapshot shaped like gemma's real one (this cache is config-only)
        snapshot_dir=snapshot_like(cfg["model"], tmp_path / "snapshot",
                                   revision=cfg["model_revision"]))
    widths = _gemma_layer_widths(cfg["model"], cfg["model_revision"])
    assert Counter(widths.values()) == {2048: 35, 4096: 7}
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    a_shapes = {int(k.split(".layers.")[1].split(".")[0]): tuple(t.shape)
                for k, t in tensors.items() if k.endswith("lora_A.weight")}
    assert a_shapes == {l: (4, w) for l, w in widths.items()}           # per module, both widths
    acfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert acfg["r"] == 4 and acfg["lora_alpha"] == 4.0 and acfg["target_modules"] == ["o_proj"]
    assert acfg["layers_to_transform"] == list(range(42))
    assert acfg["petri_decoder_prefix"] == "model.language_model.layers"
    fingerprint, keys = wr.expected_adapter_fingerprint_bf16(adapter)   # C accepts it
    assert len(fingerprint) == 64 and len(keys) == 42
    assert lx.adapter_digests(adapter).r == 4


def test_a_full_42_layer_gemma_blob_exports_and_a_layer_that_does_not_fit_its_module_refuses(tmp_path):
    """Cache-only twin of the real-object test (no staged object needed): a
    synthetic ``lora_all`` blob over EVERY gemma o_proj layer, sized off the
    cached config's own per-module shapes, exports through the derived layout;
    the same blob with one global layer's A shrunk to the local width refuses
    naming THAT layer and its module's shape — never a neighbour's."""
    _, _, cfg = _campaign("gemma4e4b")
    cached_or_skip(cfg)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    blob = lora_all_blob(cfg["model"], cfg["model_revision"], layers="all")
    assert len(blob["layers"]) == 42
    snap = snapshot_like(cfg["model"], tmp_path / "snapshot", revision=cfg["model_revision"])
    adapter = lx.export_genuine_adapter(
        {"kind": "lora_all", "positions": "all", "rank": 4}, _blob_bytes(blob),
        tmp_path / "all", base_model=cfg["model"], base_revision=cfg["model_revision"],
        snapshot_dir=snap)
    assert json.loads((adapter / lx.ADAPTER_CONFIG).read_text())["layers_to_transform"] == list(range(42))
    widths = _gemma_layer_widths(cfg["model"], cfg["model_revision"])
    global_layer = min(l for l, w in widths.items() if w == 4096)     # 5
    bad = dict(blob, A=dict(blob["A"], **{str(global_layer): torch.randn(2048, 4)}))
    with pytest.raises(lx.LoraExportError, match=rf"layer {global_layer} has \(d_in, d_out\)=\(2048, 2560\) "
                                                 rf"but the base's own o_proj at that layer is .*\(4096, 2560\)"):
        lx.export_genuine_adapter({"kind": "lora_all", "positions": "all", "rank": 4}, _blob_bytes(bad),
                                  tmp_path / "bad", base_model=cfg["model"],
                                  base_revision=cfg["model_revision"], snapshot_dir=snap)


def psr_blob(model: str, revision: str, *, seed: int = 9, positions: str = "all",
             header: int | None = None) -> dict:
    """A REAL ``psr`` blob for ``model`` (``PSRObject.save``'s shape): a gate vector,
    a gate bias and a direction per decoder block, fp32, sized off the cached
    config's residual width (``lora_export._derived_dims``)."""
    hidden, n_layers = lx._derived_dims(model, revision)
    g = torch.Generator().manual_seed(seed)
    layers = list(range(n_layers))
    return {"kind": "psr", "layers": layers, "train_direction": False, "positions": positions,
            "response_header_len": header,
            "directions": {str(l): torch.randn(hidden, generator=g) * 0.5 for l in layers},
            "gate_w": {str(l): torch.randn(hidden, generator=g) / math.sqrt(hidden) for l in layers},
            "gate_b": {str(l): torch.full((1,), 0.5) for l in layers}}


def stage_real_psr(bank: Path, cell_id: str, model: str, revision: str, *, seed=9) -> str:
    """Stage ``cell_id`` as a REAL all-positions psr object (native-psr-serving:
    a natively SERVED residual cell) with the budget sidecar the explicit registry
    validates. Returns the object's sha256."""
    from sisteer.methods import object_artifact_name

    d = bank / cell_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / object_artifact_name(arm_of(cell_id))
    p.write_bytes(_blob_bytes(psr_blob(model, revision, seed=seed)))
    sha = sha256_file(p)
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"cell_id": cell_id, "completion_tokens": 512, "object_sha256": sha}))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    return sha


def stage_real_lora_all(bank: Path, cell_id: str, model: str, revision: str, *, seed=7) -> str:
    """Stage ``cell_id`` in the bank view as a REAL ``lora_all`` object with the
    budget sidecar (the identity contract the explicit-cells registry validates).
    Returns the object's sha256."""
    from sisteer.methods import object_artifact_name

    d = bank / cell_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / object_artifact_name(arm_of(cell_id))
    p.write_bytes(_blob_bytes(lora_all_blob(model, revision, seed=seed)))
    sha = sha256_file(p)
    (d / "train_budget_meta.json").write_text(json.dumps(
        {"cell_id": cell_id, "completion_tokens": 512, "object_sha256": sha}))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    return sha


def _restage_as_response_fit_psr(bank, c):
    """native-psr-serving: an all-positions psr cell is natively SERVABLE now, so the
    skip-path tests stage their PSR cell as a RESPONSE-fit object (the shape of the
    scale-12 ``psr_kl_all`` cells) — a stage-2 position mode the native standup still
    skips by name. Overwrites ``_stage``'s minimal stub and re-stamps its sidecar."""
    d = bank / c
    (p,) = [q for q in d.iterdir() if q.suffix in (".pt", ".safetensors")]
    g = torch.Generator().manual_seed(7)
    torch.save({"kind": "psr", "layers": [0], "train_direction": False, "positions": "response",
                "response_header_len": 3,
                "directions": {"0": torch.randn(4, generator=g)},
                "gate_w": {"0": torch.randn(4, generator=g)},
                "gate_b": {"0": torch.randn(1, generator=g)}}, p)
    meta_path = d / "train_budget_meta.json"
    meta = json.loads(meta_path.read_text())
    if "object_sha256" in meta:
        meta["object_sha256"] = sha256_file(p)
    meta_path.write_text(json.dumps(meta))


def fake_bank(tmp_path, cfg, *, real_cells=(GEMMA_CELL,)):
    """Every object cell of the config staged the way the view stages it
    (derived arms -> derived_meta.json, the rest a budget sidecar), with the
    ``real_cells`` as REAL lora_all blobs and every psr cell a RESPONSE-fit blob
    (``_restage_as_response_fit_psr``). Returns ``(bank, {cell: sha})``."""
    bank = tmp_path / "cells_view" / "main"
    shas = {}
    for c in cfg["explicit_cells"]:
        if c == "none" or c.startswith("prompted__"):
            continue
        if c in real_cells:
            shas[c] = stage_real_lora_all(bank, c, cfg["model"], cfg["model_revision"])
        elif arm_of(c) in DERIVED_ARMS:
            _stage_derived(bank, c, c)
        else:
            _stage(bank, c, c)
            if ra.ARM_OBJECT_KINDS.get(arm_of(c)) == "psr":
                _restage_as_response_fit_psr(bank, c)
    return bank, shas


_TOK_PRESENT: dict = {}


def snapshot_seam(monkeypatch, tmp_path, cfg) -> Path:
    """The exporter spells every module name off the SERVED snapshot's
    weights manifest, and a developer / CI cache holds the pinned bases'
    config + tokenizer only — so every test that drives the REAL
    ``bring_up_lora`` (the fake standups here, the CLI test, the certification
    driver's end-to-end tests) points the exporter's snapshot seam
    (``lx.checkpoint_snapshot_dir``) at a snapshot shaped like the real
    checkpoint for the rest of the test: ``lora_snapshot_fixtures.snapshot_like``
    — the measured o_proj names + decoys, a real header-only safetensors for
    gemma. In-process is where the export runs for all of them (the driver
    calls ``lsu.bring_up_lora`` directly; only its oracle / gate children are
    subprocesses, and those are faked). Production resolves ``(model,
    revision)`` through the HF cache; nothing here reaches it."""
    snap = snapshot_like(cfg["model"], tmp_path / "snapshot", revision=cfg["model_revision"])
    monkeypatch.setattr(lx, "checkpoint_snapshot_dir", lambda *a, **k: snap)
    return snap


def cached_or_skip(cfg):
    key = (cfg["model"], cfg["model_revision"])
    if key not in _TOK_PRESENT:
        _TOK_PRESENT[key] = _cached_tok(*key)
    return _TOK_PRESENT[key]


def default_chat(model: str, genuine: str = GEMMA_HCELL) -> dict:
    return {model: {"ids": list(BASE_IDS), "text": "the base reply"},
            ZERO: {"ids": list(BASE_IDS), "text": "the base reply"},
            genuine: {"ids": list(GENUINE_IDS), "text": "a steered reply"}}


def synthetic_view(tmp_path, key, explicit_cells):
    """A SYNTHETIC wave (config + manifest) narrowed to ``explicit_cells`` off ``key``'s
    campaign pair — the skip path needs a view that still lists a PSR cell beside a LoRA
    cell. Returns ``(cfg_path, man_path, cfg)``."""
    cfg_path0, man_path0, cfg = _campaign(key)
    cfg = dict(cfg); cfg["explicit_cells"] = list(explicit_cells)
    man = yaml.safe_load(man_path0.read_text())
    block = man["petri"] if "petri" in man else man
    block["cells"] = [ra.harness_cell_id(c) for c in explicit_cells]
    d = tmp_path / "synthetic_wave"; d.mkdir(parents=True, exist_ok=True)
    cfg_path = d / f"sdfbase_full22_{key}_synthetic.yaml"
    man_path = d / f"petri_full22_{key}_synthetic.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    man_path.write_text(yaml.safe_dump(man, sort_keys=False))
    return cfg_path, man_path, cfg


def bring_up_fake_standup(lsu, tmp_path, monkeypatch, *, key="gemma4e4b", cells=(GEMMA_CELL, PROMPTED),
                          real_cells=(GEMMA_CELL,), expect=None, fake_kwargs=None, run_controls=True,
                          bank=None, extra=None, campaign=None):
    """The REAL ``bring_up_lora`` on ``key``'s campaign config over a synthetic
    bank, the vllm child replaced by the scripted upstream. Returns
    ``(standup, captured)``; the caller tears the standup down. ``campaign``
    (``(cfg_path, man_path, cfg)``) replaces the campaign pair (a synthetic view)."""
    cfg_path, man_path, cfg = campaign if campaign is not None else _campaign(key)
    cached_or_skip(cfg)
    view, shas = fake_bank(tmp_path, cfg, real_cells=real_cells) if bank is None else bank
    out = tmp_path / "out"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    snapshot_seam(monkeypatch, tmp_path, cfg)
    monkeypatch.setattr(lsu.socket, "gethostname", lambda: "TESTHOST")
    captured: dict = {"shas": shas}
    genuine_h = ra.harness_cell_id(real_cells[0]) if real_cells else GEMMA_HCELL
    # chat_delay_s > 0 so the standup's concurrent batches OVERLAP in the fake (its
    # "mixed step" count is what the structural selection verdict requires >= 1 of)
    fk = {"chat": default_chat(cfg["model"], genuine_h), "genuine_name": genuine_h,
          "chat_delay_s": 0.05}
    fk.update(fake_kwargs or {})
    monkeypatch.setattr(vs, "launch_server",
                        fake_upstream_launch_server(tmp_path, captured=captured, **fk))
    if expect is None:
        expect = [f"{c}={shas[c]}" for c in real_cells if c in shas]
    kwargs = dict(repo_dir=REPO, config=cfg_path, manifest=man_path, defenses_dir=view,
                  out=out, model_key=key, eval=EVAL, cell=(list(cells) if cells is not None else None),
                  adapter_bank=tmp_path / "adapter_bank", expect_object_sha256=expect,
                  probe_max_tokens=8, mode="static", run_controls=run_controls)
    kwargs.update(extra or {})
    su = lsu.bring_up_lora(**kwargs)
    captured["out"] = out
    return su, captured


# --------------------------------------------------------------------------- #
# the pure helpers                                                             #
# --------------------------------------------------------------------------- #


def test_select_cells_lora_admits_prompted_from_the_si_vocabulary_and_refuses_unlisted_objects(lsu):
    want = ["none", "a__psr_kl_all", GEMMA_CELL]
    vocab = ("constv2ctl2-full", "safety_refusal")
    assert lsu.select_cells_lora(want, None, si_vocabulary=vocab) == (want, [])
    assert lsu.select_cells_lora(want, [], si_vocabulary=vocab) == (want, [])
    # s3r-fix-2: a steered-wave config lists no `none` — the base route is served implicitly, first (the core derives its
    # none__lora0 control twin from it); the measured cells stay the config's
    assert lsu.select_cells_lora(["a__psr_kl_all", GEMMA_CELL], None, si_vocabulary=vocab) == (["none", "a__psr_kl_all", GEMMA_CELL], [])
    assert lsu.select_cells_lora([GEMMA_CELL], [], si_vocabulary=vocab) == (["none", GEMMA_CELL], [])
    assert lsu.select_cells_lora(want, ["none"], si_vocabulary=vocab) == (["none"], [])
    assert lsu.select_cells_lora(want, [GEMMA_CELL, "none", GEMMA_CELL], si_vocabulary=vocab) == \
        (["none", GEMMA_CELL], [])
    # a prompted baseline over a config-named SI is admitted beyond explicit_cells
    assert lsu.select_cells_lora(want, [GEMMA_CELL, PROMPTED], si_vocabulary=vocab) == \
        (["none", GEMMA_CELL, PROMPTED], [PROMPTED])
    # ... but not over an SI the config never names, and never an unlisted OBJECT cell
    with pytest.raises(lsu.StandupRefused, match="not in the config's SI vocabulary"):
        lsu.select_cells_lora(want, ["prompted__si_irrelevant"], si_vocabulary=vocab)
    with pytest.raises(lsu.StandupRefused, match="not one of the config's explicit_cells"):
        lsu.select_cells_lora(want, ["b__lora_r4_ce"], si_vocabulary=vocab)
    with pytest.raises(lsu.StandupRefused, match="not one of the config's explicit_cells"):
        lsu.select_cells_lora(want, [f"{PROMPTED}@thinking"], si_vocabulary=vocab)
    with pytest.raises(lsu.StandupRefused):
        lsu.select_cells_lora(want, [""], si_vocabulary=vocab)


def test_config_si_vocabulary_names_the_gemma_configs_sis_beyond_its_explicit_cells():
    _, _, cfg = _campaign("gemma4e4b")
    vocab = _load_lsu().config_si_vocabulary(cfg)
    assert PROMPTED not in cfg["explicit_cells"]          # the prompted-baseline case, exactly
    assert "safety_refusal" in vocab and "constv2ctl2-full" in vocab
    assert "si_constitution" in vocab                       # max_len_by_si's keys count


def test_manifest_flags_for_lora_native_drops_and_records_both_pins(lsu):
    kept, dropped = lsu.manifest_flags_for_lora_native(
        ["--enforce-eager", "--chat-template-content-format", "string", "--enable-lora"])
    assert kept == ["--chat-template-content-format", "string"]
    assert dropped == ["--enforce-eager", "--enable-lora"]
    assert lsu.manifest_flags_for_lora_native([]) == ([], [])
    assert lsu.STRIPPED_MANIFEST_FLAGS == ("--enforce-eager", "--enable-lora")
    assert lsu.EVAL_BY_MODE == {"attacks": "multiturn_attacks", "petri": "petri",
                                "static": "families_served", "capability": "capability"}
    assert (lsu.EXIT_REFUSED, lsu.EXIT_DEAD_ENGINE, lsu.EXIT_CONTROLS_FAILED) == (2, 3, 4)


def test_probe_set_is_the_five_probe_prompts_plus_the_frozen_gate_set(lsu):
    from petri_serving_gate import GATE_PROMPT_SET
    from sisteer.eval.attacks.served_target import PROBE_PROMPTS
    prompts = lsu.probe_prompt_set()
    assert len(prompts) == len(PROBE_PROMPTS) + len(GATE_PROMPT_SET) == 11
    assert prompts[0] == ({"role": "user", "content": PROBE_PROMPTS[0]},)
    assert prompts[-1] == tuple(dict(m) for m in GATE_PROMPT_SET[-1])
    # the fake upstream keys prompts by the same canonical JSON the harness hashes by
    for p in prompts:
        assert cm.canonical_json(list(p)) == json.dumps(list(p), sort_keys=True,
                                                        separators=(",", ":"))


def test_the_launcher_passes_model_key_and_no_literal_window(lsu):
    src = (REPO / "scripts/sdfbase/lora_serving_up.py").read_text()
    assert 'ap.add_argument("--model-key", required=True' in src
    assert "resolve_standup_geometry(" in src and "model_key=model_key" in src
    import inspect
    params = inspect.signature(lsu.bring_up_lora).parameters
    # the library seam's keyword names ARE the CLI flags (F composes the same dict)
    for flag in ("repo_dir", "config", "manifest", "defenses_dir", "out", "model_key", "eval", "cell",
                 "adapter_bank", "expect_object_sha256", "max_model_len", "max_num_seqs",
                 "allow_over_budget", "geometry_override_reason", "gpu_mem_util", "max_logprobs",
                 "proxy_port", "vllm_port", "ready_timeout_s", "upstream_timeout_s",
                 "no_petri_agent_loop", "probe_max_tokens"):
        assert flag in params, flag
    assert params["cell"].default is None and params["eval"].kind == params["eval"].KEYWORD_ONLY
    assert 'ap.add_argument("--max-model-len", type=int, default=None)' in src
    assert 'ap.add_argument("--max-num-seqs", type=int, default=None)' in src


# --------------------------------------------------------------------------- #
# the real bring-up on the gemma campaign config                               #
# --------------------------------------------------------------------------- #


def test_bring_up_lora_end_to_end_on_gemma_writes_every_record(lsu, tmp_path, monkeypatch):
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch)
    try:
        out, sha = cap["out"], cap["shas"][GEMMA_CELL]
        ep = json.loads((out / "endpoint.json").read_text())
        cs = json.loads((out / "cells_served.json").read_text())
        pc = json.loads((out / "positive_controls.json").read_text())
        table = json.loads((out / "routing_table.json").read_text())
        att = sa.read_serving_attestation(out / vs.SERVING_ATTESTATION_FILE)   # strict reader
        # the served legs: base, the admitted prompted baseline, the cell, the zero leg —
        # and NO lens leg
        routes = {r["cell_id"]: r for r in table["routes"]}
        assert set(routes) == {"none", PROMPTED_H, GEMMA_HCELL, ZERO}
        assert table["version"] == 2 and set(table[px.ADAPTERS_KEY]) == {GEMMA_HCELL, ZERO}
        assert cm.parse_serving_identity(routes[GEMMA_HCELL]["serving_identity"]).path == "lora"
        assert cm.parse_serving_identity(routes[ZERO]["serving_identity"]).path == "lora"
        assert set(ep["served_models"]) == {"petri__base", f"petri__{PROMPTED_H}",
                                            f"petri__{GEMMA_HCELL}", f"petri__{ZERO}"}
        # THE VERIFIED OBJECT MAP: harness id -> sha, equal in both files, neither the zero
        # leg nor the prompted cell in it
        assert ep["object_artifact_sha256_by_cell"] == {GEMMA_HCELL: sha}
        assert cs["object_artifact_sha256_by_cell"] == ep["object_artifact_sha256_by_cell"]
        assert ep["object_artifact_sha256"] == sha
        assert su.proxy.object_artifact_sha256_by_cell == {GEMMA_HCELL: sha}
        # the engine, the attestation facts, the adapters
        assert ep["engine"] == cs["engine"] == "lora_native"
        assert ep["attested_enforce_eager"] is False and ep["attested_cudagraph_mode"] == "FULL_AND_PIECEWISE"
        assert ep["lora_config"] == att["lora_config"] and ep["id_name_map"] == att["id_name_map"]
        assert ep["id_name_map"] == {"1": ZERO, "2": GEMMA_HCELL}       # sorted-name order
        assert ep["serving_attestation"] == str(out / vs.SERVING_ATTESTATION_FILE)
        assert set(ep["adapters"]) == {GEMMA_HCELL} and ep["zero_adapter"]["name"] == ZERO
        block = ep["adapters"][GEMMA_HCELL]
        assert set(block) == set(px.ADAPTER_BLOCK_KEYS)
        assert block["petri_intervention_hash"] == routes[GEMMA_HCELL]["intervention_hash"]
        assert block["petri_zero_adapter"] is False and block["petri_source_kind"] == "lora_all"
        assert block["r"] == 4 and ep["zero_adapter"]["r"] == 1
        assert ep["zero_adapter"]["petri_intervention_hash"] == routes["none"]["intervention_hash"] \
            == routes[ZERO]["intervention_hash"]
        assert block == table[px.ADAPTERS_KEY][GEMMA_HCELL] == cs["adapters"][GEMMA_HCELL]
        assert att["adapters"][GEMMA_HCELL]["config_sha256"] == block["config_sha256"]
        assert att["adapters"][GEMMA_HCELL]["weights_sha256"] == block["weights_sha256"]
        assert Path(block["dir"]).parent == (tmp_path / "adapter_bank").resolve()
        assert Path(block["dir"]).name == routes[GEMMA_HCELL]["intervention_hash"]
        assert ep["adapters_reused"] == {GEMMA_HCELL: False, ZERO: False}
        # the records' other keys
        assert ep["vmodels"] == {"none": "petri__base", PROMPTED_H: f"petri__{PROMPTED_H}",
                                 GEMMA_HCELL: f"petri__{GEMMA_HCELL}", ZERO: f"petri__{ZERO}"}
        assert ep["model"] == "google/gemma-4-E4B-it" and ep["revision"] == ep["model_revision"]
        assert ep["chat_template_sha256"] == hashlib.sha256(
            (out / "chat_template.jinja").read_bytes()).hexdigest() == cs["chat_template_sha256"]
        assert ep["chat_template_hash"] == ep["chat_template_sha256"][:16]
        assert ep["prompted_cells_admitted"] == [PROMPTED] == cs["prompted_cells_admitted"]
        assert ep["manifest_flags_dropped"] == ["--enforce-eager"] == cs["manifest_flags_dropped"]
        assert ep["routing_table_version"] == 2 == cs["routing_table_version"]
        assert ep["cells"] == ["none", GEMMA_CELL, PROMPTED] == cs["cells_requested"]
        assert ep["harness_cells"] == ["none", GEMMA_HCELL, PROMPTED_H]
        assert ep["lora_cells"] == [GEMMA_HCELL] and ep["eval"] == EVAL and ep["mode"] == "static"
        # THE LORA KERNEL INDEX CEILING (job 488932): the table's S=121 at L=4864 exceeds
        # (2^31-1) // 4096 = 524,287 tokens (the zero adapter covers gemma's 4096-wide
        # global-attention o_proj), so the standup lowered S to 107 BEFORE the launch under
        # the named reason and every record says so; the attestation carries the numbers
        assert ep["max_num_seqs"] == 107 == cap["launch"].max_num_seqs == att["expected"]["max_num_seqs"]
        assert ep["geometry_override_reason"] == sa.LORA_KERNEL_CEILING_REASON == "lora_kernel_index_ceiling"
        low = ep["geometry_lowered"]
        assert (low["reason"], low["requested_S"], low["S"], low["L"], low["D"], low["ceiling"]) == \
            ("lora_kernel_index_ceiling", 121, 107, 4864, 4096, 524_287)
        assert low["product_requested"] == 588_544 and low["product"] == 520_448 and low["requested_override_reason"] is None
        assert low["widest_module"].startswith(f"{ZERO}:") and low["widest_module"].endswith(".o_proj")
        assert cs["geometry_lowered"] == low and cs["lora_kernel_index_ceiling"] == ep["lora_kernel_index_ceiling"]
        assert ep["lora_kernel_index_ceiling"] == {"D": 4096, "widest_module": low["widest_module"], "ceiling": 524_287,
                                                   "L": 4864, "S": 107, "product": 520_448, "limit": 2**31 - 1}
        assert att["lora_kernel_index_ceiling"] == 524_287 and att["lora_max_module_dim"] == 4096
        assert cap["launch"].lora_kernel_index_ceiling == 524_287 and cap["launch"].lora_max_module_dim == 4096
        assert ep["geometry"]["seqs_source"] == "override" and ep["tool_calls"] is False
        # ... and ONLY seqs: the window the table resolved keeps its `table` source through the lowering
        assert ep["geometry"]["window_source"] == "table" and ep["geometry"]["max_model_len"] == 4864
        assert ep["proxy_token"] is None and ep["node"] == "TESTHOST"
        assert ep["positive_controls"] == str(out / "positive_controls.json")
        assert ep["max_model_len"] == su.geometry.max_model_len == cap["launch"].max_model_len
        assert ep["attested_max_num_seqs"] == su.geometry.max_num_seqs
        for key in ("proxy_url", "routing_table", "model_key", "geometry", "agent_loop_entry",
                    "proxy_upstream_timeout_s", "cells_served", "adapter_bank", "started"):
            assert key in ep, key
        # the derived config and the pre-launch facts in cells_served
        assert Path(cs["config_used"]).name == lsu.DERIVED_CONFIG_FILE
        derived = yaml.safe_load(Path(cs["config_used"]).read_text())
        assert derived["explicit_cells"][-1] == PROMPTED and derived["model"] == ep["model"]
        assert cs["attestation"] == {"enforce_eager": False, "cudagraph_mode": "FULL_AND_PIECEWISE",
                                     # the FULL surface (2026-09-16): the dispatcher's largest FULL key IS the served S
                                     "cudagraph_full_capture_max": su.geometry.max_num_seqs,
                                     "lora_config": att["lora_config"],
                                     "id_name_map": att["id_name_map"],
                                     "model_runner": att["model_runner"]}
        # the composer put S in the capture list and the workers attest a FULL graph at S — the fact the
        # 2026-09-16 collapse (S=51 on the default list, FULL largest 48) lacked; the endpoint carries all three
        assert ep["attested_cudagraph_full_capture_max"] == su.geometry.max_num_seqs == att["engine"]["cudagraph_full_capture_max"]
        assert su.geometry.max_num_seqs in ep["composed_cudagraph_capture_sizes"] == list(su.launch.cudagraph_capture_sizes)
        assert ep["attested_cudagraph_full_capture_sizes"] == att["engine"]["cudagraph_full_capture_sizes"]
        assert max(ep["attested_cudagraph_full_capture_sizes"]) == su.geometry.max_num_seqs
        assert {c["cell_id"] for c in cs["canonical"]} == {"none", PROMPTED_H, GEMMA_HCELL}
        assert [c["cell_id"] for c in cs["controls"]] == [ZERO]
        # the five positive controls, all passed, in order
        assert pc["passed"] is True and pc["failed_control"] is None
        assert pc["order"] == list(lsu.CONTROL_NAMES)
        assert all(pc["controls"][n]["passed"] is True for n in lsu.CONTROL_NAMES if n != "boundary_refusal")
        assert pc["controls"]["boundary_refusal"]["verdict"] == "not_applicable" and pc["controls"]["boundary_refusal"]["restricted_cells"] == []   # no response-boundary cell served (O49)
        assert pc["controls"]["noop_certificate"]["n_prompts"] == 11
        assert pc["controls"]["steering_bites"]["cells"][GEMMA_HCELL]["twin"] == ZERO
        sel = pc["controls"]["selection_honored"]
        # THE STRUCTURAL VERDICT (job 489308): the engine's own per-request LoRA
        # bookkeeping, read through the worker's routing recorder over the dev RPC route
        assert sel["passed"] is True and sel["verdict"] == "routing_attested"
        assert sel["policy"] == "native" and sel["zero_adapter"] == f"petri__{ZERO}"
        assert sel["genuine"] == [f"petri__{GEMMA_HCELL}"]
        # 3 references + the base repeat + 3 rotated layouts, 33 rows each, every one observed
        # — JOINED through the engine's `-<8 hex>` renaming (the fake mints internal ids
        # distinct from the response ids, as InputProcessor.assign_request_id does)
        assert sel["n_requests"] == 33 * 7 == sel["n_observed"] and sel["n_missing"] == 0
        assert sel["id_join"]["n_suffixed"] == 33 * 7 and sel["id_join"]["n_exact"] == 0
        assert sel["n_ambiguous"] == 0 and sel["n_foreign"] >= 0          # the solo diagnostics' own calls
        assert sel["n_inconsistent"] == 0 and sel["n_misrouted"] == 0 and sel["exemption"] is None
        # the epoch, the runner (canary #4 ran V2) — in the record and the attestation
        assert len(sel["epoch"]) == 32 and sel["recorder"]["arm"]["epoch"] == sel["epoch"]
        assert sel["runner_kind"] == "v2" and sel["runner_class"].endswith("gpu.model_runner.GPUModelRunner")
        assert att["model_runner"] == {"class": "vllm.v1.worker.gpu.model_runner.GPUModelRunner", "kind": "v2"}
        assert ep["attested_model_runner"] == att["model_runner"] == cs["attestation"]["model_runner"]
        assert sel["per_selection"] == {
            "petri__base": {"n": 99, "n_observed": 99, "adapter": None, "lora_int_id": 0},
            f"petri__{ZERO}": {"n": 66, "n_observed": 66, "adapter": ZERO, "lora_int_id": 1},
            f"petri__{GEMMA_HCELL}": {"n": 66, "n_observed": 66, "adapter": GEMMA_HCELL, "lora_int_id": 2}}
        assert sel["selection_adapter"] == {"petri__base": None, f"petri__{ZERO}": ZERO,
                                            f"petri__{GEMMA_HCELL}": GEMMA_HCELL}
        assert sel["id_name_map"] == att["id_name_map"] == {"1": ZERO, "2": GEMMA_HCELL}
        # the fake records one "step" per request served while armed (the 231 control rows
        # plus the solo diagnostics' uncached calls); the real recorder records engine steps.
        # The mixed count is computed over THIS control's joined requests.
        assert sel["steps"]["n_steps"] >= 33 * 7 and sel["steps"]["n_steps_mixed"] >= 1
        assert sel["steps"]["max_distinct_selections_per_step"] >= 2
        assert sel["steps"]["basis"] == "computed over this control's joined requests only"
        assert sel["recorder"]["arm"]["armed"] is True and sel["recorder"]["arm"]["n_workers"] == 1
        assert sel["recorder"]["disarm"]["disarmed"] is True and sel["recorder"]["disarm"]["was_armed"] is True
        assert sel["recorder"]["disarm"]["epoch"] == sel["epoch"]
        assert sel["recorder"]["recorder_health"]["n_workers"] == 1
        # the OUTPUT comparisons are RECORDED under numerics, never the verdict: the
        # equal-size rule, the rotated layouts, positional distinguishability, the base
        # repeat floor and the batch-invariance block
        num = sel["numerics"]
        assert num["gating"] is False and num["rule"] == "equal-size" and "verdict" not in num
        assert num["n_batch"] == 33 and num["n_layouts"] == 3 and num["n_mixed_rows"] == 99
        assert num["n_submitted_rows"] == 33 * 7 and "expected_routing" not in num
        assert num["selection_first_divergences"] == [] and num["n_mismatched_rows"] == 0
        assert num["positional_distinguishability"]["n_undistinguishable"] == 0
        assert num["positional_distinguishability"]["n_pairs_checked"] == 33 * 4
        assert num["zero_equals_base"] is True and num["signatures_distinct"] is True
        assert num["whole_layout"]["zero_vs_base"]["n_divergent_rows"] == 0
        assert num["fixed_size_repeat_floor"]["n_divergent_rows"] == 0
        assert num["fixed_size_repeat_floor"]["selection"] == "petri__base"
        bi = num["batch_invariance"]
        assert "recorded only" in bi["note"]
        assert set(bi["solo_vs_same_size"]) == {"petri__base", f"petri__{ZERO}", f"petri__{GEMMA_HCELL}"}
        assert all(v["n"] == 11 and v["n_divergent"] == 0 for v in bi["solo_vs_same_size"].values())
        assert all(v["n_pairs"] == 22 and v["n_divergent"] == 0 for v in bi["within_batch_repeats"].values())
        # the admission's reader accepts the record's shape
        import multilora_admission as adm
        assert adm.selection_control_problems(sel) == []
        assert adm.controls_summary(pc)["all_passed"] is True
        assert set(pc["controls"]["adapter_digests"]["adapters"]) == {GEMMA_HCELL, ZERO}
        assert cs["positive_controls"] == pc
        # the composed launch: lora-native, the manifest's eager pin stripped, the qwen/gemma
        # unowned flags kept, adapters in sorted order, the splice
        cmd = list(cap["launch"].cmd)
        assert cap["launch"].lora_native and "--no-enforce-eager" in cmd and "--enforce-eager" not in cmd
        assert cmd[cmd.index("--chat-template-content-format") + 1] == "string"
        assert cap["launch"].lora_modules == (ZERO, GEMMA_HCELL)
        assert cmd[-2:] == ["--gpu-memory-utilization", "0.9"] and "--max-logprobs" not in cmd
        assert cmd[cmd.index("--max-lora-rank") + 1] == "8"
        # the LIVE proxy: the engine block in /v1/models and the stamp on a routed response
        payload = su.proxy.models_payload()["petri_proxy"]
        assert payload[px.ENGINE_KEY]["kind"] == "lora_native"
        assert set(payload[px.ENGINE_KEY]["adapters"]) == {GEMMA_HCELL, ZERO}
        resp = su.chat({"model": f"petri__{GEMMA_HCELL}", "max_tokens": 4, "temperature": 0,
                        "return_token_ids": True,
                        "messages": [{"role": "user", "content": "hi"}]})
        assert resp["petri"]["engine"] == "lora_native"
        assert resp["petri"]["engine_state"] == {"enforce_eager": False,
                                                 "cudagraph_mode": "FULL_AND_PIECEWISE"}
        assert resp["choices"][0]["token_ids"] == GENUINE_IDS[:4]
        assert resp["petri"]["serving_identity"] == "proxy_vllm+lora+interleaved"
        # A's bank: the adapter dir is content-addressed and its digests re-verify
        dig = lx.adapter_digests(Path(block["dir"]))
        assert (dig.config_sha256, dig.weights_sha256, dig.r) == (
            block["config_sha256"], block["weights_sha256"], 4)
        px.verify_adapter_block_on_disk(block)
        assert su.server_rc() is None
    finally:
        su.tear_down()
    assert not (cap["out"] / "endpoint.json").exists()          # unlinked at teardown


def test_a_second_bring_up_reuses_the_banked_adapters(lsu, tmp_path, monkeypatch):
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch)
    su.tear_down()
    cfg_path, man_path, cfg = _campaign("gemma4e4b")
    bank_view = tmp_path / "cells_view" / "main"
    su2, cap2 = bring_up_fake_standup(
        lsu, tmp_path / "again", monkeypatch, bank=(bank_view, cap["shas"]))
    try:
        ep2 = json.loads((cap2["out"] / "endpoint.json").read_text())
        assert ep2["adapters_reused"] == {GEMMA_HCELL: False, ZERO: False}   # a fresh bank dir
    finally:
        su2.tear_down()
    # the SAME bank: verified reuse, byte-identical blocks
    su3, cap3 = bring_up_fake_standup(
        lsu, tmp_path / "third", monkeypatch, bank=(bank_view, cap["shas"]),
        extra={"adapter_bank": tmp_path / "adapter_bank"})
    try:
        ep3 = json.loads((cap3["out"] / "endpoint.json").read_text())
        ep1 = su.endpoint()
        assert ep3["adapters_reused"] == {GEMMA_HCELL: True, ZERO: True}
        assert ep3["adapters"] == ep1["adapters"] and ep3["zero_adapter"] == ep1["zero_adapter"]
    finally:
        su3.tear_down()


# --------------------------------------------------------------------------- #
# the rc contract                                                              #
# --------------------------------------------------------------------------- #


def test_a_zero_adapter_that_is_not_a_noop_fails_the_certificate_with_rc_4(lsu, tmp_path,
                                                                         monkeypatch):
    with pytest.raises(lsu.PositiveControlsFailed, match="noop_certificate") as ei:
        bring_up_fake_standup(lsu, tmp_path, monkeypatch, fake_kwargs={"zero_differs": True})
    rec = ei.value.record
    assert ei.value.rc == 4 and rec["passed"] is False and rec["failed_control"] == "noop_certificate"
    assert rec["controls"]["noop_certificate"]["passed"] is False
    assert "scale0_noop" in rec["controls"]["noop_certificate"]["error"]
    assert rec["controls"]["selection_honored"]["passed"] is None        # never ran
    out = tmp_path / "out"
    assert json.loads((out / "positive_controls.json").read_text()) == rec
    assert not (out / "endpoint.json").exists()
    # the server was torn down: nothing answers on the REAL composed vllm port
    # (the earlier assertion connected to port 1 — `len(launch_cmd) and 1` —
    # and so proved nothing; efficiency review 2026-09-11)
    import socket as _socket
    served = json.loads((out / "cells_served.json").read_text())
    cmd = list(served["launch_cmd"])
    vllm_port = int(cmd[cmd.index("--port") + 1])          # the composed vllm child's port
    assert vllm_port > 1024, cmd
    deadline = time.monotonic() + 10
    while True:
        try:
            with _socket.create_connection(("127.0.0.1", vllm_port), timeout=0.2):
                pass
        except OSError:
            break
        assert time.monotonic() < deadline, f"the vllm child still answers on {vllm_port}"
        time.sleep(0.2)


def test_a_mixed_batch_whose_engine_binds_the_zero_requests_to_the_genuine_adapter_fails_selection_rc_4(
        lsu, tmp_path, monkeypatch):
    """A REAL misroute: the engine's own bookkeeping shows the zero
    adapter's requests bound to the genuine adapter whenever a genuine request
    is in flight — the structural verdict refuses naming the selection and the
    adapters, never a request id or an output."""
    with pytest.raises(lsu.PositiveControlsFailed, match="selection_honored") as ei:
        bring_up_fake_standup(lsu, tmp_path, monkeypatch,
                              fake_kwargs={"misroute_zero_when_concurrent": True,
                                           "chat_delay_s": 0.15})
    rec = ei.value.record
    assert rec["failed_control"] == "selection_honored" and ei.value.rc == 4
    # the sequential controls before it PASSED — solo calls were honest
    for name in ("noop_certificate", "salt_isolation", "steering_bites"):
        assert rec["controls"][name]["passed"] is True, name
    err = rec["controls"]["selection_honored"]["error"]
    assert "served under an adapter other than their selection's" in err
    assert f"request(s) for 'petri__{ZERO}' ran under ['{GEMMA_HCELL}'] instead of '{ZERO}'" in err
    assert "chatcmpl-" not in err                          # never a request id
    assert rec["controls"]["adapter_digests"]["passed"] is None
    assert not (tmp_path / "out" / "endpoint.json").exists()


def test_an_output_only_disagreement_in_a_concurrent_batch_is_recorded_and_passes(
        lsu, tmp_path, monkeypatch):
    """The job-489308 shape (gemma canary #3): the zero adapter's OUTPUT in a
    concurrent batch differs from base's while every solo call agrees and the
    engine's bookkeeping is honest. Earlier this refused as contamination;
    now the structural verdict passes and the disagreement is a recorded
    number under numerics."""
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch,
                                    fake_kwargs={"contaminate_zero_when_concurrent": True,
                                                 "chat_delay_s": 0.15})
    try:
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        assert pc["passed"] is True
        sel = pc["controls"]["selection_honored"]
        assert sel["passed"] is True and sel["verdict"] == "routing_attested"
        num = sel["numerics"]
        # the zero rows disagreed where a genuine request was in flight beside them
        assert num["n_mismatched_rows"] > 0 or num["whole_layout"]["zero_vs_base"]["n_divergent_rows"] > 0
        assert num["selection_agreement"][f"petri__{ZERO}"]["n_rows"] == 33
        assert num["gating"] is False
        assert (cap["out"] / "endpoint.json").exists()
    finally:
        su.tear_down()


def test_a_recorder_that_saw_nothing_fails_selection_rc_4(lsu, tmp_path, monkeypatch):
    """A worker whose recorder never armed (or a server without one) answers
    every request as missing: refused by name, never read as 'nothing
    misrouted'."""
    with pytest.raises(lsu.PositiveControlsFailed, match="selection_honored") as ei:
        bring_up_fake_standup(lsu, tmp_path, monkeypatch, fake_kwargs={"recorder_broken": True})
    err = ei.value.record["controls"]["selection_honored"]["error"]
    assert "never observed by the engine's LoRA bookkeeping" in err
    assert "'petri__base': 99/99" in err and "0 engine ids recorded" in err and ei.value.rc == 4


def test_an_explicit_psr_cell_is_refused_before_launch_rc_2_naming_the_escape_hatch(lsu, tmp_path, monkeypatch):
    """native-serving-default: an EXPLICIT --cell the engine cannot serve (here a RESPONSE-fit psr —
    a stage-2 position mode under native-psr-serving) refuses (exit 2) and names --serving-path lens —
    the whole-view standup skips such a cell by name instead (the test below)."""
    # STAGE-1 WORLD (native-psr-serving step 7 flipped both flags): the skip-by-name of a stage-2 mode is
    # pinned here with both flags monkeypatched False — the stage-2 path is tested in
    # tests/test_native_position_class.py / test_native_position_mode.py
    from sisteer.capeval.petri import defense_cells as _dc
    from sisteer.capeval.serving import native_residual as _nr
    monkeypatch.setattr(_dc, "NATIVE_STAGE2_ARMED", False)
    monkeypatch.setattr(_nr, "STAGE2_POSITION_CLASSES", False)

    launched = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: launched.append(1))
    with pytest.raises(lsu.StandupRefused, match="psr_ce_all.*cannot be served on the native engine") as ei:
        bring_up_fake_standup(lsu, tmp_path, monkeypatch, cells=(GEMMA_CELL, PSR_CELL),
                              expect=[f"{GEMMA_CELL}=" + "ab" * 32, f"{PSR_CELL}=" + "cd" * 32])
    assert ei.value.rc == 2 and launched == [] and "--serving-path lens" in str(ei.value)
    # a declared variant of a LoRA cell is a REFUSAL too (no declaration surface on a native adapter), by name
    with pytest.raises(lsu.StandupRefused, match="declaration") as ei2:
        bring_up_fake_standup(lsu, tmp_path / "declared", monkeypatch, cells=(f"{GEMMA_CELL}@thinking",),
                              expect=[f"{GEMMA_CELL}@thinking=" + "ab" * 32])
    assert ei2.value.rc == 2 and launched == [] and "--serving-path lens" in str(ei2.value)


def test_a_whole_view_native_standup_skips_the_psr_cells_by_name(lsu, tmp_path, monkeypatch):
    """A view that still lists a PSR cell has it SKIPPED BY NAME — logged,
    recorded in cells_served.json / endpoint.json `cells_skipped`, absent from the served vmodels — never a
    second lens standup, never a silent drop; the LoRA cell beside it serves. The digests are asked for the
    SERVED fitted cells only."""
    # STAGE-1 WORLD (native-psr-serving step 7 flipped both flags): the skip-by-name of a stage-2 mode is
    # pinned here with both flags monkeypatched False — the stage-2 path is tested in
    # tests/test_native_position_class.py / test_native_position_mode.py
    from sisteer.capeval.petri import defense_cells as _dc
    from sisteer.capeval.serving import native_residual as _nr
    monkeypatch.setattr(_dc, "NATIVE_STAGE2_ARMED", False)
    monkeypatch.setattr(_nr, "STAGE2_POSITION_CLASSES", False)

    campaign = synthetic_view(tmp_path, "gemma4e4b", ["none", GEMMA_CELL, PSR_CELL])
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch, cells=None, campaign=campaign)
    try:
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        cs = json.loads((cap["out"] / "cells_served.json").read_text())
        psr_h = ra.harness_cell_id(PSR_CELL)
        assert ep["cells_skipped"] == cs["cells_skipped"] == [
            {"cell_id": psr_h, "kind": "psr", "reason": ep["cells_skipped"][0]["reason"]}]
        # native-psr-serving: the synthetic psr cell is fit at 'response' — a stage-2 mode, skipped by name
        assert "native position mode 'response'" in ep["cells_skipped"][0]["reason"]
        assert "stage-2" in ep["cells_skipped"][0]["reason"]
        assert su.cells_skipped[0].cell_id == psr_h
        assert ep["cells"] == ["none", GEMMA_CELL] and ep["harness_cells"] == ["none", GEMMA_HCELL]
        assert set(ep["served_models"]) == {"petri__base", f"petri__{GEMMA_HCELL}", f"petri__{ZERO}"}
        assert f"petri__{psr_h}" not in ep["served_models"]
        assert ep["lora_cells"] == [GEMMA_HCELL] and ep["engine"] == ep["serving_engine"] == "lora_native"
        assert ep["object_artifact_sha256_by_cell"] == {GEMMA_HCELL: cap["shas"][GEMMA_CELL]}
        assert ep["expected_digests_source"] == "cli" and ep["thinking_mode"] == "off"
        assert "lens_reduction_mode" not in ep and ep["reduction_mode"] == "not_applicable"
        assert ep["geometry"]["adjustments"][0]["kind"] == "lora_kernel_index_ceiling"
        assert set(ep["geometry"]["adjustments"][0]) == {"kind", "from_S", "to_S", "D", "ceiling"}
        assert (ep["geometry"]["adjustments"][0]["from_S"], ep["geometry"]["adjustments"][0]["to_S"]) == (121, 107)
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        assert pc["passed"] is True and all(pc["controls"][n]["passed"] is True for n in lsu.CONTROL_NAMES if n != "boundary_refusal")
        assert pc["controls"]["boundary_refusal"]["verdict"] == "not_applicable" and pc["controls"]["boundary_refusal"]["restricted_cells"] == []   # no response-boundary cell served (O49)
    finally:
        su.tear_down()


def test_a_swapped_object_refuses_before_any_export_rc_2(lsu, tmp_path, monkeypatch):
    exported = []
    monkeypatch.setattr(lx, "export_or_reuse", lambda *a, **k: exported.append(1))
    launched = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: launched.append(1))
    with pytest.raises(lsu.StandupRefused, match="replaced since the preflight") as ei:
        bring_up_fake_standup(lsu, tmp_path, monkeypatch, expect=[f"{GEMMA_CELL}=" + "ab" * 32])
    assert ei.value.rc == 2 and exported == [] and launched == []
    assert not (tmp_path / "adapter_bank").exists()
    # ... and a fitted cell with NO digest refuses too (E45 H1)
    with pytest.raises(lsu.StandupRefused, match="expect-object-sha256"):
        bring_up_fake_standup(lsu, tmp_path / "nodigest", monkeypatch, expect=[])


def test_a_version_1_table_refuses_to_proceed_rc_2(lsu, tmp_path, monkeypatch):
    real_export = px.export_routing_table

    def stale_exporter(*a, **k):
        k.pop("adapter_blocks", None)      # a tree whose exporter writes version 1 only
        return real_export(*a, **k)
    monkeypatch.setattr(px, "export_routing_table", stale_exporter)
    launched = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: launched.append(1))
    # the version-1 refusal now fires INSIDE the exporter (native-positions-resolve: the plain LoRA cell attests `all`
    # from its kind, and a position mode never reaches a version-1 table) — the same refusal, one step earlier; a
    # stale tree whose cells attest no mode still refuses on the loaded table's version
    with pytest.raises(lsu.StandupRefused, match=r"version[- ]1") as ei:
        bring_up_fake_standup(lsu, tmp_path, monkeypatch)
    assert ei.value.rc == 2 and launched == []


def test_a_prompted_only_request_set_is_a_zero_adapter_only_standup(lsu, tmp_path, monkeypatch):
    """A cell set with NO LoRA cell (here the prompted baseline alone)
    stands up NATIVELY with the zero adapter as the one registered adapter — base and steered
    configurations share one engine — with steering_bites / selection_honored recorded not_applicable
    (passed: None, verdict not_applicable) and the other three controls run; the zero leg derives
    without a live lora route (enumerate_cells(lora_controls=True))."""
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch, cells=(PROMPTED,), expect=[])
    try:
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        cs = json.loads((cap["out"] / "cells_served.json").read_text())
        assert su.zero_adapter_only and ep["lora_cells"] == [] and ep["adapters"] == {}
        assert ep["zero_adapter"]["name"] == ZERO and ep["id_name_map"] == {"1": ZERO}
        assert set(ep["served_models"]) == {"petri__base", f"petri__{PROMPTED_H}", f"petri__{ZERO}"}
        assert ep["object_artifact_sha256_by_cell"] == {} and ep["object_artifact_sha256"] is None
        assert [c["cell_id"] for c in cs["controls"]] == [ZERO] and cs["zero_adapter_only"] is True
        assert pc["passed"] is True and pc["failed_control"] is None and pc["zero_adapter_only"] is True
        for name in ("noop_certificate", "salt_isolation", "adapter_digests"):
            assert pc["controls"][name]["passed"] is True, name
        for name in ("steering_bites", "selection_honored"):
            assert pc["controls"][name]["passed"] is None and pc["controls"][name]["verdict"] == "not_applicable", name
        assert "no lora cell" in pc["controls"]["steering_bites"]["reason"]
        assert "single adapter" in pc["controls"]["selection_honored"]["reason"]
        assert list(cap["launch"].lora_modules) == [ZERO]
        assert su.proxy.models_payload()["petri_proxy"][px.ENGINE_KEY]["kind"] == "lora_native"
    finally:
        su.tear_down()


def test_geometry_override_needs_a_reason_and_records_it(lsu, tmp_path, monkeypatch):
    from sisteer.capeval.serving import geometry as sg
    ident = sg.resolve_geometry(EVAL, model_key="gemma4e4b", strict_identity=True)
    half = ident.max_num_seqs // 2
    # a non-identity S without the reason is refused by the strict guard ...
    with pytest.raises(lsu.StandupRefused, match="geometry"):
        bring_up_fake_standup(lsu, tmp_path / "a", monkeypatch, extra={"max_num_seqs": half})
    # ... a reason for nothing is refused ...
    with pytest.raises(lsu.StandupRefused, match="reason for nothing"):
        bring_up_fake_standup(lsu, tmp_path / "b", monkeypatch,
                              extra={"geometry_override_reason": "mamba_cache_blocks_under_graphs"})
    # ... and the reason + the override serve the lowered S and record why
    su, cap = bring_up_fake_standup(
        lsu, tmp_path / "c", monkeypatch,
        extra={"max_num_seqs": half, "geometry_override_reason": "mamba_cache_blocks_under_graphs"})
    try:
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        assert ep["max_num_seqs"] == half == ep["attested_max_num_seqs"]
        assert ep["geometry_override_reason"] == "mamba_cache_blocks_under_graphs"
        assert ep["geometry"]["seqs_source"] == "override"
        # a requested S already under the LoRA kernel index ceiling is served as requested:
        # the ceiling is recorded, nothing is lowered, the caller's reason stands
        assert ep["geometry_lowered"] is None and ep["lora_kernel_index_ceiling"]["S"] == half
        assert ep["lora_kernel_index_ceiling"]["ceiling"] == 524_287
    finally:
        su.tear_down()
    # the driver's own path: a pre-lowered S=107 WITH the ceiling reason is accepted as is
    su2, cap2 = bring_up_fake_standup(
        lsu, tmp_path / "d", monkeypatch,
        extra={"max_num_seqs": 107, "geometry_override_reason": lsu.sa.LORA_KERNEL_CEILING_REASON})
    try:
        ep2 = json.loads((cap2["out"] / "endpoint.json").read_text())
        assert ep2["max_num_seqs"] == 107 and ep2["geometry_lowered"] is None
        assert ep2["geometry_override_reason"] == "lora_kernel_index_ceiling"
    finally:
        su2.tear_down()


def test_cli_main_rc_0_on_stop_file_and_3_on_a_dead_engine(lsu, tmp_path, monkeypatch):
    cfg_path, man_path, cfg = _campaign("gemma4e4b")
    cached_or_skip(cfg)
    view, shas = fake_bank(tmp_path, cfg)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    snapshot_seam(monkeypatch, tmp_path, cfg)
    monkeypatch.setattr(lsu.socket, "gethostname", lambda: "TESTHOST")
    monkeypatch.setattr(vs, "launch_server", fake_upstream_launch_server(
        tmp_path, chat=default_chat(cfg["model"]), genuine_name=GEMMA_HCELL))

    def argv(out, stop):
        return ["--repo-dir", str(REPO), "--config", str(cfg_path), "--manifest", str(man_path),
                "--defenses-dir", str(view), "--out", str(out), "--stop-file", str(stop),
                "--model-key", "gemma4e4b", "--mode", "static", "--cell", GEMMA_CELL,
                "--cell", PROMPTED, "--expect-object-sha256", f"{GEMMA_CELL}={shas[GEMMA_CELL]}",
                "--adapter-bank", str(tmp_path / "adapter_bank"), "--probe-max-tokens", "8"]

    # rc 0: the stop file is already there -> READY, endpoint.json while ready, torn down
    out0, stop0 = tmp_path / "out0", tmp_path / "STOP0"
    stop0.touch()
    seen = {}

    def snapshot_then_stop(stop_file, server, **kw):
        seen["endpoint"] = json.loads((out0 / "endpoint.json").read_text())
        return lsu.wait_until_stop.__wrapped__(stop_file, server, **kw) \
            if hasattr(lsu.wait_until_stop, "__wrapped__") else 0
    monkeypatch.setattr(lsu, "wait_until_stop", snapshot_then_stop)
    assert lsu.main(argv(out0, stop0)) == 0
    assert seen["endpoint"]["engine"] == "lora_native" and not (out0 / "endpoint.json").exists()
    # rc 3: the engine dies while serving
    out3, stop3 = tmp_path / "out3", tmp_path / "STOP3"
    real_wait = _load_lsu().wait_until_stop

    def kill_then_wait(stop_file, server, **kw):
        server.kill()
        server.wait(timeout=10)
        return real_wait(stop_file, server, poll_s=0.05)
    monkeypatch.setattr(lsu, "wait_until_stop", kill_then_wait)
    assert lsu.main(argv(out3, stop3)) == 3
    # rc 2 / 4 at the CLI: a swapped object; a zero adapter that is not a no-op
    out2, stop2 = tmp_path / "out2", tmp_path / "STOP2"
    bad = argv(out2, stop2)
    bad[bad.index("--expect-object-sha256") + 1] = f"{GEMMA_CELL}=" + "ab" * 32
    assert lsu.main(bad) == 2
    monkeypatch.setattr(vs, "launch_server", fake_upstream_launch_server(
        tmp_path / "z", chat=default_chat(cfg["model"]), genuine_name=GEMMA_HCELL, zero_differs=True))
    assert lsu.main(argv(tmp_path / "out4", tmp_path / "STOP4")) == 4
    # neither --mode nor --eval is an argparse error
    with pytest.raises(SystemExit):
        lsu.main([a for a in argv(tmp_path / "out5", tmp_path / "STOP5") if a not in ("--mode", "static")])


# --------------------------------------------------------------------------- #
# native serving default: the eight options the wave front gained              #
# --------------------------------------------------------------------------- #


def test_the_wave_front_gained_exactly_the_nine_keyword_only_options_each_defaulting_off(lsu):
    """Eight from the native serving front + ``server_default_policy`` (the server-default
    sampling policy, ``contract_override`` = every launch to date byte for byte)."""
    import inspect
    params = inspect.signature(lsu.bring_up_lora).parameters
    assert lsu.NATIVE_FRONT_OPTIONS == ("thinking_on", "reasoning_parser", "text_only_mm", "enable_log_requests",
                                        "sampling_clamp_override", "mamba_backend_contract", "kv_derived_seqs",
                                        "attest_mrope", "server_default_policy")
    want = {"thinking_on": False, "reasoning_parser": None, "text_only_mm": "off", "enable_log_requests": False,
            "sampling_clamp_override": None, "mamba_backend_contract": None, "kv_derived_seqs": False,
            "attest_mrope": False, "server_default_policy": "contract_override"}
    for name, default in want.items():
        assert params[name].kind == params[name].KEYWORD_ONLY and params[name].default == default, name
    # the core's own interface: explicit tool_calls, the adapter bank, the same options
    from sisteer.capeval.serving import native_standup as ns
    core = inspect.signature(ns.bring_up_native).parameters
    assert core["tool_calls"].default is core["tool_calls"].empty and core["adapter_bank"].default is core["adapter_bank"].empty
    assert core["eval"].kind == core["eval"].KEYWORD_ONLY and core["geometry"].kind == core["geometry"].KEYWORD_ONLY
    for name in ("thinking_on", "reasoning_parser", "sampling_clamp_override", "text_only_mm", "mamba_backend_contract",
                 "kv_derived_seqs", "enable_log_requests", "server_default_policy", "probe_max_tokens", "gpu_mem_util", "max_logprobs",
                 "proxy_port", "vllm_port", "ready_timeout_s", "install_timeout_s", "upstream_timeout_s",
                 "petri_agent_loop", "geometry_override_reason", "allow_over_budget", "mode", "run_controls"):
        assert name in core, name
    assert [f.name for f in dataclasses_fields(ns.EffectiveServing)][:14] == [
        "spec", "cell_source", "lens_source", "served_cells", "cells_skipped", "tokenizer", "chat_template_bytes",
        "config_used", "defenses_dir", "model_key", "model", "revision", "expected_digests", "expected_digests_source"]
    assert lsu.effective_launch_flags is ns.effective_launch_flags
    assert lsu.manifest_flags_for_lora_native(["--enforce-eager", "x"]) == ns.effective_launch_flags(["--enforce-eager", "x"])


def dataclasses_fields(cls):
    import dataclasses as _dc
    return _dc.fields(cls)


def test_kv_derived_seqs_relaunches_once_and_records_kernel_then_kv_adjustments(lsu, tmp_path, monkeypatch):
    """On a readiness KvCoverageShortfall under kv_derived_seqs the core re-launches ONCE at
    floor(the reported concurrency) COMPOSED AFTER the kernel lowering — recorded as geometry.adjustments
    [kernel, kv] (the kv step's from_S is the kernel step's to_S), validated by the contract; the
    override reason keeps the FIRST kind; without the opt-in the refusal stands."""
    from sisteer.capeval.serving import geometry as sg
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch, fake_kwargs={"shortfall_first": 100.5},
                                    extra={"kv_derived_seqs": True})
    try:
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        adj = ep["geometry"]["adjustments"]
        assert [a["kind"] for a in adj] == ["lora_kernel_index_ceiling", "kv_shrink"]
        assert (adj[0]["from_S"], adj[0]["to_S"]) == (121, 107) and (adj[1]["from_S"], adj[1]["to_S"]) == (107, 100)
        assert ep["max_num_seqs"] == 100 == cap["launch"].max_num_seqs == ep["attested_max_num_seqs"]
        assert [l.max_num_seqs for l in cap["launches"]] == [107, 100]          # ONE re-launch, after the kernel step
        assert ep["geometry"]["kv_shrink"] == ep["kv_shrink"] == {k: v for k, v in adj[1].items() if k not in ("kind", "from_S", "to_S")}
        assert ep["geometry"]["seqs_source"] == "kv_derived" and ep["geometry"]["identity"] is False
        assert ep["geometry_override_reason"] == "lora_kernel_index_ceiling"   # the FIRST adjustment's kind
        assert (cap["out"] / lsu.ns.KV_SHORTFALL_LOG).read_text().startswith("engine log of the first")
        assert sg.kv_shrink_problem(ep["kv_shrink"], configured_max_num_seqs=107, max_model_len=4864,
                                    served_max_num_seqs=100) is None
        if hasattr(sg, "geometry_adjustment_problem"):
            assert sg.geometry_adjustment_problem(ep["geometry"], model_key="gemma4e4b", L=4864, S_identity=121,
                                                  S_served=100, native=True) is None
            # the negatives the contract names: a bare smaller S, steps reversed
            bare = {**ep["geometry"], "adjustments": []}
            assert sg.geometry_adjustment_problem(bare, model_key="gemma4e4b", L=4864, S_identity=121, S_served=100)
            rev = {**ep["geometry"], "adjustments": list(reversed(adj))}
            assert sg.geometry_adjustment_problem(rev, model_key="gemma4e4b", L=4864, S_identity=121, S_served=100)
    finally:
        su.tear_down()
    # the kernel step ALONE validates too (the e2e's records), and without the opt-in the refusal stands
    su2, cap2 = bring_up_fake_standup(lsu, tmp_path / "kernel_only", monkeypatch)
    try:
        ep2 = json.loads((cap2["out"] / "endpoint.json").read_text())
        assert [a["kind"] for a in ep2["geometry"]["adjustments"]] == ["lora_kernel_index_ceiling"]
        assert ep2["kv_shrink"] is None and "kv_shrink" not in ep2["geometry"]
        if hasattr(sg, "geometry_adjustment_problem"):
            assert sg.geometry_adjustment_problem(ep2["geometry"], model_key="gemma4e4b", L=4864, S_identity=121,
                                                  S_served=107, native=True) is None
    finally:
        su2.tear_down()
    with pytest.raises(vs.KvCoverageShortfall):
        bring_up_fake_standup(lsu, tmp_path / "no_opt_in", monkeypatch, fake_kwargs={"shortfall_first": 100.5})


def test_thinking_on_composes_the_family_parser_on_the_launch_argv_and_declares_nothing(lsu, tmp_path, monkeypatch):
    """--thinking-on + --reasoning-parser reach build_server_launch — the LAUNCH ARGV carries
    --reasoning-parser <family pin> and the ON template pin; gemma-4's header does not move
    (trivially_in_support), the proxy runs thinking_mode on, the records say declared_cells [] (a native
    standup has no header-dependent cell to declare); a parser outside the family's pin refuses at compose."""
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch,
                                    extra={"thinking_on": True, "reasoning_parser": "gemma4"})
    try:
        cmd = list(cap["launch"].cmd)
        assert cmd[cmd.index("--reasoning-parser") + 1] == "gemma4"
        assert '"enable_thinking": true' in cmd[cmd.index("--default-chat-template-kwargs") + 1]
        assert "--no-enforce-eager" in cmd and "--lora-modules" in cmd
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        cs = json.loads((cap["out"] / "cells_served.json").read_text())
        assert ep["thinking_mode"] == cs["thinking_mode"] == "on" and ep["reasoning_parser"] == "gemma4"
        assert ep["thinking_certification"]["verdict"] == "trivially_in_support"
        assert ep["thinking_declared_cells"] == [] and ep["thinking_header"]["switch"] is True
        assert ep["default_chat_template_kwargs"] == cap["launch"].default_chat_template_kwargs
        assert su.proxy.models_payload()["petri_proxy"]["thinking_mode"] == "on"
        # a lora-route request needs no declaration under the ON proxy (the controls already ran)
        resp = su.chat({"model": f"petri__{GEMMA_HCELL}", "max_tokens": 4, "temperature": 0, "return_token_ids": True,
                        "messages": [{"role": "user", "content": "hi"}]})
        assert resp["choices"][0]["token_ids"] == GENUINE_IDS[:4]
    finally:
        su.tear_down()
    with pytest.raises(vs.VllmServerError, match="reasoning_parser"):
        bring_up_fake_standup(lsu, tmp_path / "badparser", monkeypatch,
                              extra={"thinking_on": True, "reasoning_parser": "deepseek_r1"})


def test_the_sampling_clamp_override_wraps_the_launch_argv_and_is_attested(lsu, tmp_path, monkeypatch):
    """--sampling-clamp-override reaches build_server_launch — the composed argv runs
    through the vllm_entry wrapper with --sisteer-max-temp 0.001 — and the clamp is RECORDED in
    endpoint.json exactly as capctrl records it AND as an additive block of serving_attestation.json the
    strict reader still reads; every other option (--enable-log-requests, --text-only-mm) rides the argv too."""
    import shutil

    from sisteer.capeval.tinycap.contracts import BFCL_PROTOCOL_TEMPERATURE
    # the wrapper resolves the SERVING interpreter off the `vllm` console script (absent in this CPU venv): point
    # it at this interpreter's bin/, and refuse the .pth route so nothing is written into the shared venv
    real_which = shutil.which
    monkeypatch.setattr(vs.shutil, "which",
                        lambda name, *a, **k: (str(Path(sys.executable).parent / "vllm") if name == "vllm"
                                               else real_which(name, *a, **k)))
    monkeypatch.setattr(vs, "drop_shim_pth", lambda *a, **k: (_ for _ in ()).throw(
        vs.VllmServerError("test: the .pth route is refused here")))
    # text_only_mm "on": the pin composed for ANY target (vllm_server.text_only_mm_decision — under "auto" gemma-4,
    # multimodal but without a verified family entry, records False with the reason, exactly as capctrl does)
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch,
                                    extra={"sampling_clamp_override": BFCL_PROTOCOL_TEMPERATURE,
                                           "enable_log_requests": True, "text_only_mm": "on"})
    try:
        cmd = list(cap["launch"].cmd)
        assert "sisteer.capeval.serving.vllm_entry" in cmd and "--sisteer-max-temp" in cmd
        assert cmd[cmd.index("--sisteer-max-temp") + 1] == str(BFCL_PROTOCOL_TEMPERATURE)
        assert "--enable-log-requests" in cmd
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        rec = ep["sampling_clamp_override"]
        assert rec["to"] == BFCL_PROTOCOL_TEMPERATURE and rec == dict(cap["launch"].sampling_clamp_override)
        assert ep["sampling_clamp_override_mechanism"] == getattr(cap["launch"], "sampling_clamp_override_mechanism", None)
        assert ep["enable_log_requests"] is True and ep["text_only_mm"] is True   # the pin composed ("on")
        assert ep["text_only_mm_record"]["source"] == "standup" and vs.TEXT_ONLY_MM_FLAG in cmd   # ... and rides the argv
        assert ep["text_only_mm_record"]["source"] == "standup" and "--limit-mm-per-prompt" in cmd
        att = sa.read_serving_attestation(cap["out"] / vs.SERVING_ATTESTATION_FILE)     # the strict reader
        assert att["sampling_clamp_override"]["to"] == BFCL_PROTOCOL_TEMPERATURE
        assert att["sampling_clamp_override"]["mechanism"] == ep["sampling_clamp_override_mechanism"]
    finally:
        su.tear_down()
    # absent: no such key anywhere (the card identity stands)
    su2, cap2 = bring_up_fake_standup(lsu, tmp_path / "stock", monkeypatch)
    try:
        ep2 = json.loads((cap2["out"] / "endpoint.json").read_text())
        assert "sampling_clamp_override" not in ep2 and "--sisteer-max-temp" not in cap2["launch"].cmd
        assert "sampling_clamp_override" not in sa.read_serving_attestation(cap2["out"] / vs.SERVING_ATTESTATION_FILE)
    finally:
        su2.tear_down()


def test_a_nemotron_shaped_worker_answers_the_mamba_rpc_and_a_failing_report_refuses(lsu, tmp_path, monkeypatch):
    """On a nemotron_h target the core asks the PORTED sisteer_mamba_backend_report over the dev RPC
    route and judges it as capctrl does — a conformant report passes (recorded in endpoint.json and as the
    attestation's additive mamba_backend block; the kernel step fires 161 -> 86 at 4864, D = 5120), a
    short report REFUSES (rc 2) after tearing the server down, and no contract refuses before any RPC."""
    import importlib.metadata as md

    from test_capctrl_standup_attestations import _conformant_live_report
    from sisteer.capeval.serving import mamba_backend as mb
    contract = mb.load_contract(REPO / mb.CONTRACT_RELPATH)
    monkeypatch.setattr(md, "version", lambda dist: contract["vllm_version"] if dist == "vllm" else "0")
    cfg_path, man_path, cfg = _campaign("nemotron3nano4b")
    cached_or_skip(cfg)
    good = _conformant_live_report(contract)
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch, key="nemotron3nano4b",
                                    cells=(GEMMA_CELL,), fake_kwargs={"mamba_results": [good]})
    try:
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        att = ep["mamba_backend_attestation"]
        assert att["ok"] is True and att["rpc_method"] == "sisteer_mamba_backend_report" == mb.RPC_METHOD
        assert att["expected_n_mamba_layers"] == 21 and len(att["reports"]) == 1 and ep["mamba_backend_attestation_error"] is None
        assert mb.endpoint_attestation_problems(ep) == []
        rec = sa.read_serving_attestation(cap["out"] / vs.SERVING_ATTESTATION_FILE)
        assert rec["mamba_backend"]["ok"] is True and rec["mamba_backend"]["contract_sha256"] == att["contract_sha256"]
        # the kernel step on the hybrid: D = 5120 (o_proj in_features), ceiling 419,430 -> S 86 at 4864
        adj = ep["geometry"]["adjustments"]
        assert [a["kind"] for a in adj] == ["lora_kernel_index_ceiling"] and adj[0]["D"] == 5120
        assert adj[0]["to_S"] == 86 == ep["max_num_seqs"] and adj[0]["ceiling"] == (2**31 - 1) // 5120
        assert ep["mrope_attestation"] is None and "not asked" in ep["mrope_attestation_error"]
    finally:
        su.tear_down()
    short = _conformant_live_report(contract, n_mamba=20)
    with pytest.raises(lsu.StandupRefused, match="backend attestation FAILED") as ei:
        bring_up_fake_standup(lsu, tmp_path / "short", monkeypatch, key="nemotron3nano4b", cells=(GEMMA_CELL,),
                              fake_kwargs={"mamba_results": [short]})
    assert ei.value.rc == 2 and not (tmp_path / "short" / "out" / "endpoint.json").exists()
    with pytest.raises(lsu.StandupRefused, match="no sisteer_mamba_backend_report report from any worker"):
        bring_up_fake_standup(lsu, tmp_path / "norpc", monkeypatch, key="nemotron3nano4b", cells=(GEMMA_CELL,))
    monkeypatch.setattr(lsu, "default_mamba_contract", lambda repo: None)
    launched = []
    monkeypatch.setattr(vs, "launch_server", lambda *a, **k: launched.append(1))
    with pytest.raises(lsu.StandupRefused, match="no Nemotron backend contract"):
        bring_up_fake_standup(lsu, tmp_path / "nocontract", monkeypatch, key="nemotron3nano4b", cells=(GEMMA_CELL,))
    assert launched == []          # refused before any launch


def test_attest_mrope_is_recorded_not_applicable_and_glm_refuses(lsu, tmp_path, monkeypatch):
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch, extra={"attest_mrope": True})
    try:
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        assert ep["mrope_attestation"] is None and ep["mrope_attestation_error"].startswith("not_applicable")
    finally:
        su.tear_down()
    from sisteer.capeval.serving import native_standup as ns
    assert ns.GLM_MODEL_KEY == "glm46vflash"


def test_resolve_adapter_bank_has_three_sources_and_the_export_race_yields_one_verified_entry(lsu, tmp_path, monkeypatch):
    """The explicit bank wins; else the cluster default; else $SISTEER_ADAPTER_BANK_ROOT/<model_key>;
    else a refusal naming all three. The bank is content-addressed and the publish is ONE rename: two
    exporters racing on one root in-process resolve to ONE entry (one exported, one reused, byte-verified),
    and a tampered winner refuses by digest on the next reuse."""
    from concurrent.futures import ThreadPoolExecutor

    from sisteer.capeval.serving import native_standup as ns
    monkeypatch.setattr(ns, "shared_root", lambda: None)
    monkeypatch.delenv(ns.ADAPTER_BANK_ROOT_ENV, raising=False)
    with pytest.raises(ns.StandupRefused) as ei:
        ns.resolve_adapter_bank(None, model_key="gemma4e4b")
    for source in ("--adapter-bank", "SISTEER_SHARED_ROOT", ns.ADAPTER_BANK_ROOT_ENV):
        assert source in str(ei.value), source
    monkeypatch.setenv(ns.ADAPTER_BANK_ROOT_ENV, str(tmp_path / "vol"))
    assert ns.resolve_adapter_bank(None, model_key="gemma4e4b") == (tmp_path / "vol" / "gemma4e4b").resolve()
    assert ns.resolve_adapter_bank(tmp_path / "explicit", model_key="gemma4e4b") == (tmp_path / "explicit").resolve()
    with pytest.raises(ns.StandupRefused, match="model-key"):
        ns.resolve_adapter_bank(None, model_key=None)
    monkeypatch.setattr(ns, "shared_root", lambda: tmp_path / "shared")
    assert ns.resolve_adapter_bank(None, model_key="gemma4e4b") == (tmp_path / "shared/exp/multilora_adapters/gemma4e4b").resolve()
    # the race: the ZERO adapter's export (the same install-or-reuse path every genuine cell takes)
    cfg_path, man_path, cfg = _campaign("gemma4e4b")
    cached_or_skip(cfg)
    snapshot_seam(monkeypatch, tmp_path, cfg)
    bank = tmp_path / "race_bank"
    # The two exporters race in THREADS here (production's two exporters are two containers, i.e. two processes):
    # the meta-skeleton the exporter builds sets and restores torch's process-wide default dtype around its
    # construction, and two interleaved threads can restore it to bfloat16 — which then makes every later
    # `torch.randn` in this pytest process bf16 (the next test's REAL lora_all blob refused as "not this
    # object's blob"). The bank's publish discipline under test is filesystem-level and unaffected; restore the
    # default dtype so the leak stays inside this test.
    default_dtype = torch.get_default_dtype()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(lx.export_or_reuse_zero, bank, base_model=cfg["model"], base_revision=cfg["model_revision"])
                    for _ in range(2)]
            results = [f.result() for f in futs]
    finally:
        torch.set_default_dtype(default_dtype)
    dirs = {str(Path(d).resolve()) for d, _, _ in results}
    assert len(dirs) == 1 and sorted(r[2] for r in results) in ([False, True], [False, False])
    digests = {(r[1].config_sha256, r[1].weights_sha256) for r in results}
    assert len(digests) == 1
    assert len(list(bank.glob(".tmp-*"))) == 0                       # no private export left behind
    # a tampered winner refuses by digest on reuse
    winner = Path(next(iter(dirs)))
    weights = winner / sa.ADAPTER_WEIGHTS_FILE
    weights.write_bytes(weights.read_bytes()[:-1] + bytes([weights.read_bytes()[-1] ^ 1]))
    with pytest.raises(lx.LoraExportError):
        lx.export_or_reuse_zero(bank, base_model=cfg["model"], base_revision=cfg["model_revision"])


# --------------------------------------------------------------------------- #
# endpoint.json is ONE attempt's record                                        #
# --------------------------------------------------------------------------- #

#: what a standup that died without ``tear_down`` (SIGTERM / SIGKILL / a node
#: failure) leaves behind: a readiness record pointing at a proxy that is gone
_STALE_ENDPOINT = {"proxy_url": "http://127.0.0.1:1/v1", "engine": "lora_native",
                   "standup_attempt_id": "dead" * 8, "node": "TESTHOST", "stale": True}
_HEX32 = r"[0-9a-f]{32}"


def test_a_prior_attempts_endpoint_is_quarantined_before_the_controls_and_this_attempts_is_stamped(
        lsu, tmp_path, monkeypatch):
    """``bring_up_lora`` accepted an existing out dir WITHOUT invalidating
    its ``endpoint.json``; a retry with the same out dir kept a crashed
    standup's marker through the preflight, the launch and all five controls
    (with a reused explicit ``--proxy-port`` it pointed at the NEW proxy
    before the controls had passed). Now the prior record is moved aside at
    the start of every attempt — the controls hook sees NO endpoint — and
    the endpoint this attempt publishes carries its nonce, as do
    ``cells_served.json`` and ``positive_controls.json``."""
    import re
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / "endpoint.json").write_text(json.dumps(_STALE_ENDPOINT))
    seen: dict = {}
    real_controls = lsu.run_positive_controls

    def controls_hook(su, **kw):
        # every controls function receives the core's diagnostics policy
        seen["endpoint_during_controls"] = (out / "endpoint.json").exists()
        seen["attempt_id"] = su.attempt_id
        seen["stale_dirs"] = sorted(p.name for p in out.glob(f"{lsu.STALE_DIR_PREFIX}*"))
        seen["output_diagnostics"] = kw.get("output_diagnostics")
        return real_controls(su, **kw)
    monkeypatch.setattr(lsu, "run_positive_controls", controls_hook)
    su, cap = bring_up_fake_standup(lsu, tmp_path, monkeypatch)
    try:
        assert seen["endpoint_during_controls"] is False            # no readiness before the controls
        assert seen["attempt_id"] == su.attempt_id and re.fullmatch(_HEX32, su.attempt_id)
        assert len(seen["stale_dirs"]) == 1                          # quarantined BEFORE the controls
        stale = sorted(out.glob(f"{lsu.STALE_DIR_PREFIX}*/endpoint.json"))
        assert len(stale) == 1 and json.loads(stale[0].read_text()) == _STALE_ENDPOINT
        assert stale[0].parent.name == seen["stale_dirs"][0]
        ep = json.loads((out / "endpoint.json").read_text())
        cs = json.loads((out / "cells_served.json").read_text())
        pc = json.loads((out / "positive_controls.json").read_text())
        assert ep[lsu.ENDPOINT_ATTEMPT_KEY] == cs[lsu.ENDPOINT_ATTEMPT_KEY] == pc[lsu.ENDPOINT_ATTEMPT_KEY] \
            == su.attempt_id
        assert lsu.ENDPOINT_ATTEMPT_KEY == "standup_attempt_id"
        assert ep["proxy_url"] == su.proxy_url and "stale" not in ep      # THIS attempt's proxy
        assert su.endpoint()[lsu.ENDPOINT_ATTEMPT_KEY] == su.attempt_id
    finally:
        su.tear_down()
    assert not (out / "endpoint.json").exists()                     # unpublished at teardown
    assert stale[0].is_file()                                        # a rename, never a deletion


def test_a_failed_bring_up_never_leaves_an_endpoint_behind(lsu, tmp_path, monkeypatch):
    """F3, the failure faces: a stale endpoint seeded before (a) a control
    failure after the launch and (b) a refusal before it — in both, no
    ``endpoint.json`` remains and the stale one sits aside; (c) a second
    stale record gets its own quarantine dir, nothing is clobbered."""
    import re
    # (a) the zero adapter is not a no-op: rc 4 after the launch
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / "endpoint.json").write_text(json.dumps(_STALE_ENDPOINT))
    with pytest.raises(lsu.PositiveControlsFailed, match="noop_certificate"):
        bring_up_fake_standup(lsu, tmp_path, monkeypatch, fake_kwargs={"zero_differs": True})
    assert not (out / "endpoint.json").exists()
    assert len(list(out.glob(f"{lsu.STALE_DIR_PREFIX}*/endpoint.json"))) == 1
    pc = json.loads((out / "positive_controls.json").read_text())
    assert pc["passed"] is False and re.fullmatch(_HEX32, pc[lsu.ENDPOINT_ATTEMPT_KEY])
    assert pc[lsu.ENDPOINT_ATTEMPT_KEY] != _STALE_ENDPOINT["standup_attempt_id"]
    # (b) a swapped object: refused BEFORE any launch, the stale record quarantined all the same
    out2 = tmp_path / "b" / "out"
    out2.mkdir(parents=True)
    (out2 / "endpoint.json").write_text(json.dumps(_STALE_ENDPOINT))
    launched = []
    monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: launched.append(1))
    with pytest.raises(lsu.StandupRefused, match="replaced since the preflight"):
        bring_up_fake_standup(lsu, tmp_path / "b", monkeypatch, expect=[f"{GEMMA_CELL}=" + "ab" * 32])
    assert launched == [] and not (out2 / "endpoint.json").exists()
    stale2 = list(out2.glob(f"{lsu.STALE_DIR_PREFIX}*/endpoint.json"))
    assert len(stale2) == 1 and json.loads(stale2[0].read_text()) == _STALE_ENDPOINT
    # (c) another crashed attempt's record: a second quarantine dir, the first untouched
    (out2 / "endpoint.json").write_text(json.dumps(dict(_STALE_ENDPOINT, stale=2)))
    with pytest.raises(lsu.StandupRefused, match="replaced since the preflight"):
        bring_up_fake_standup(lsu, tmp_path / "b", monkeypatch, expect=[f"{GEMMA_CELL}=" + "ab" * 32])
    assert not (out2 / "endpoint.json").exists()
    assert len(list(out2.glob(f"{lsu.STALE_DIR_PREFIX}*/endpoint.json"))) == 2
    assert json.loads(stale2[0].read_text()) == _STALE_ENDPOINT
    # the helpers alone: idempotent, and a rename never a deletion
    assert lsu.unpublish_endpoint(out2) is False
    (out2 / "endpoint.json").write_text("{}")
    d = lsu.quarantine_prior_endpoint(out2, attempt_id="ab" * 16)
    assert d is not None and (d / "endpoint.json").is_file() and not (out2 / "endpoint.json").exists()
    assert lsu.quarantine_prior_endpoint(out2, attempt_id="ab" * 16) is None


def test_requested_cells_explicit_records_exactly_what_was_named(lsu):
    """The front's declaration -- stripped, order kept, de-duplicated, ``none`` only when named, () / () when nothing
    was; a blank entry and a spelling the harness grammar cannot translate refuse. The explicit harness ids are among what
    ``select_cells_lora`` serves for the same request."""
    assert lsu.requested_cells_explicit(None) == ((), ()) and lsu.requested_cells_explicit([]) == ((), ())
    names, harness = lsu.requested_cells_explicit([f" {GEMMA_CELL} ", GEMMA_CELL, "none"])
    assert names == (GEMMA_CELL, "none") and harness == (GEMMA_HCELL, "none")
    with pytest.raises(lsu.StandupRefused, match="non-empty"):
        lsu.requested_cells_explicit([" "])
    with pytest.raises(lsu.StandupRefused, match="nonexistent_arm"):
        lsu.requested_cells_explicit(["safety_refusal__nonexistent_arm"])
    tt, _ = lsu.select_cells_lora(["none", GEMMA_CELL, PROMPTED], [GEMMA_CELL], si_vocabulary=())
    assert tt == ["none", GEMMA_CELL] and set(harness) <= {ra.harness_cell_id(c) for c in tt}
    assert lsu.REQUESTED_EXPLICIT_KEY == "cells_requested_explicit" and lsu.REQUESTED_EXPLICIT_HARNESS_KEY == "cells_requested_explicit_harness"
