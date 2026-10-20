"""Shared builders for the seedrep streaming-ladder tests: the protocol module loaded BY PATH (``scripts/modal``
shadows the ``modal`` package on ``sys.path``, so never by bare name), a synthetic wave (placeholder wave id / uuid),
queue specs, containers over the two-snapshot fake volume, and a synthetic sampled static base store. Everything is
SYNTHETIC: invented goal ids, placeholder shas, no recorded data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

W7 = "0c19251"
WAVE = "11111111-2222-4333-8444-555555555555"
WAVE2 = "99999999-8888-4777-8666-555555555555"
PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}


def _load(name: str, rel: str):
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_ladder():
    return _load("seedrep_ladder", "scripts/modal/seedrep_ladder.py")


def load_rung_redraw():
    load_ladder()
    return _load("seedrep_rung_redraw", "scripts/sdfbase/seedrep/rung_redraw.py")


def static_spec(L, *, cap_next=8192, leg="pasttense", view="s1", shards=("0", "1"), wave=WAVE):
    return L.QueueSpec.build(W7, wave, leg, view, "static", cap_next, list(shards))


def static_row(goal_id, sample_idx, *, seed, cap=4096, finish="length", text=None, trial=0, seed_base=1):
    """A sampled static row (the ``run_static`` shape ``escalate_static`` reads); ``seed_base`` = the view's."""
    text = text if text is not None else f"resp {goal_id} {sample_idx}"
    return {"goal_id": goal_id, "sample_idx": int(sample_idx), "trial": trial, "variant_key": f"{sample_idx}:{trial}",
            "family": "pasttense", "attack_prompt": f"{goal_id}/{sample_idx}", "response": text,
            "decode": {"do_sample": True, **PRESET, "cap": cap, "arm": "vendor_recommended", "seed_base": int(seed_base), "trial": trial},
            "sampling_seed": int(seed), "finish_reason": finish,
            "completion_tokens": cap if finish == "length" else 10, "prompt_tokens": 5, "max_tokens": cap,
            "at_cap": finish == "length"}


class Container:
    """ONE attempt of ONE producer: its own snapshot of the shared volume, its attempt root, its Producer."""

    def __init__(self, L, shared, spec, shard: str, attempt_id: str, *, arm="none", s=1, family="pasttense"):
        self.L, self.spec, self.shard, self.attempt_id = L, spec, shard, attempt_id
        self.vol = shared.snapshot(attempt_id)
        self.ref = next(p for p in spec.producers if p.shard == shard)
        self.root = L.producer_attempt_root(self.ref, attempt_id)         # the app's shard<NN> spelling, the reconciler's default
        self.producer = L.Producer(self.vol, spec, self.ref, attempt_id, self.root)
        self.arm, self.s, self.family = arm, s, family
        self.records: list[dict] = []

    def enqueue_row(self, row, *, vmodel="petri__base", commit=False, base_store=None):
        rec = self.L.static_queue_record(self.spec, arm=self.arm, s=self.s, family=self.family, row=row,
                                         base_store=base_store or f"{self.root}/exp/private/run/shard00/responses.jsonl",
                                         producer=self.ref, producer_attempt_id=self.attempt_id, vmodel=vmodel,
                                         sampling=PRESET)
        self.producer.enqueue(rec)
        self.records.append(rec)
        if commit:
            self.vol.commit()
        return rec

    def complete(self):
        self.vol.write_bytes(self.L.complete_rel(self.root), json.dumps({"n_rows": 7, "attempt_id": self.attempt_id}).encode())
        self.vol.commit()

    def finish(self, accepted_kv, **kw):
        return self.L.finish_producer(self.producer, accepted_kv, **kw)


def kvs(L):
    return {p: L.MemoryKV(L.dict_name(W7, WAVE, p)) for p in ("accepted", "claims", "results")}


def rows3():
    """The three-item base store every producer test enqueues (g0/1, g1/1, g3/0 at cap 4096)."""
    return [static_row("g0", 1, seed=7), static_row("g1", 1, seed=8), static_row("g3", 0, seed=9)]


def close_queue(L, vol, spec, records, accepted_kv, *, shard="0", aid="a1"):
    """ONE producer attempt on a LocalVolume run to closure: enqueue every record -> COMPLETE -> inventory -> acceptance ->
    marker -> CLOSED.json. Returns the Producer."""
    ref = next(p for p in spec.producers if p.shard == shard)
    root = L.producer_attempt_root(ref, aid)
    prod = L.Producer(vol, spec, ref, aid, root)
    for rec in records:
        prod.enqueue(rec)
    vol.write_bytes(L.complete_rel(root), b"{}")
    assert L.finish_producer(prod, accepted_kv)["state"] == "accepted"
    assert L.reconcile_queue(vol, accepted_kv, spec)["queue_closed"]
    return prod


def materialise_elect(L, vol, spec, rec, record, results_kv, *, attempt="rung-1", response=None):
    """A rung attempt's draw of ``rec``: the materialised file committed and ELECTED; returns its path."""
    path, doc = L.write_materialised(vol, spec, queue_record=rec, attempt_id=attempt, call_id="fc-rung", record=record,
                                     request_id=f"rq-{rec['item_key']}", completion_id=f"cmpl-{rec['item_key']}", response=response)
    vol.commit()
    assert L.elect_result(results_kv, spec, doc, path)
    return path


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# a loopback chat-completions server + the proxy's REAL prompted-prefix check   #
# --------------------------------------------------------------------------- #


class StubProxy:
    """A loopback ``/v1/chat/completions`` server for the HTTP consumer: ``decide(body) -> (text, finish_reason,
    completion_tokens)``; ``verify(body)`` runs first and may raise a ``ProxyRefusal`` (answered as the proxy's
    OpenAI-style error with ``error.code``, e.g. ``si_prefix_missing``). Every request (headers + body) is recorded."""

    def __init__(self, decide=None, verify=None):
        import http.server
        import threading

        self.bodies: list[dict] = []
        self.decide = decide or (lambda b: (f"reply:{b['messages'][-1]['content']}", "stop", 5))
        self.verify = verify
        #: the completion id of the n-th request (``cmpl-<n>`` by default; the evidence tests hand in an engine-shaped minter)
        self.completion_id = lambda n: f"cmpl-{n}"
        #: handlers run on a ThreadingHTTPServer: the append + the completion-id stamp are one step under this lock, on the
        #: HANDLER-LOCAL dict (2026-09-20, at the landing gate: two handlers appended before either stamped, so the
        #: later one stamped the OTHER request's dict and its own never got a completion_id -> KeyError in the ledger reader)
        self._bodies_lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                return None

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                body = json.loads(raw.decode("utf-8"))
                seen = {"path": self.path, "headers": {k: v for k, v in self.headers.items()}, "body": body, "raw": raw}
                with outer._bodies_lock:
                    outer.bodies.append(seen)
                    seen["completion_id"] = outer.completion_id(len(outer.bodies))
                try:
                    if outer.verify is not None:
                        outer.verify(body)
                except Exception as e:  # noqa: BLE001 -- answered as the proxy answers
                    err = json.dumps({"error": {"message": str(e), "code": getattr(e, "code", "error"), "type": type(e).__name__}}).encode()
                    self.send_response(int(getattr(e, "http_status", 400)))
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                text, fr, ct = outer.decide(body)
                resp = {"id": seen["completion_id"], "object": "chat.completion", "model": body["model"],
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": fr}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": int(ct), "total_tokens": 11 + int(ct)}}
                out = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def prompted_prefix_verifier(vmodel: str, si_text: str):
    """``verify(body)`` running the proxy's REAL ``RoutingProxy._verify_prompted_prefix`` (H8) for requests to ``vmodel``:
    a ``PromptedPrefixSpec`` over the SI bytes and a ``RouteEntry`` carrying its ``si_prefix_hash``, exactly as the
    launcher exports them; any other vmodel passes unverified (a ``none`` route has no prefix)."""
    import types

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving import proxy as px

    data = si_text.encode("utf-8")
    sha16 = cm.sha256_hex16(data)
    route = px.RouteEntry(virtual_model=vmodel, cell_id=vmodel.split("__", 1)[-1], cell_class="prompted",
                          serving_identity="proxy_vllm:interleaved:synthetic", intervention_hash="cc" * 8, si_prefix_hash=sha16)
    fake_self = types.SimpleNamespace(_prefixes={vmodel: px.PromptedPrefixSpec(vmodel, len(data), sha16)},
                                      _system_text=px.RoutingProxy._system_text)

    def verify(body):
        if body.get("model") == vmodel:
            px.RoutingProxy._verify_prompted_prefix(fake_self, route, body)

    return verify


# --------------------------------------------------------------------------- #
# peer modules (the request-evidence client, the audit) by path, read-only    #
# --------------------------------------------------------------------------- #

#: where a peer workstream's file is looked for, in order: an environment override `SEEDREP_PEER_<WS>` naming a checkout
#: root, THIS tree (after the W merge every file is in one tree), then the sibling linked worktree of the named branch
#: (the pre-merge layout, `<repo>/worktrees/<branch>`). Absent everywhere -> the test SKIPS by name.
PEER_BRANCHES = {"WS1DP": "seedrep-ws1dp", "WS3G": "seedrep-ws3g", "WS2D": "seedrep-ws2d"}


def peer_file(ws: str, rel: str):
    """The path of ``rel`` for peer ``ws`` (``WS1DP`` | ``WS3G``), or ``None``."""
    import os
    roots = [REPO, (REPO.parent if REPO.parent.name == "worktrees" else REPO / "worktrees") / PEER_BRANCHES[ws]]
    override = os.environ.get(f"SEEDREP_PEER_{ws}")
    if override:
        roots.insert(0, Path(override))
    for r in roots:
        p = Path(r) / rel
        if p.is_file():
            return p
    return None


def _load_abs(name: str, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_request_evidence_client():
    """``sisteer.eval.request_evidence_client`` — imported by name when this tree has it, else loaded by path from
    the sibling worktree UNDER THAT NAME (so ``rung_redraw``'s ``from sisteer.eval import request_evidence_client`` finds
    it); ``pytest.skip`` when neither exists."""
    import importlib
    import pytest

    name = "sisteer.eval.request_evidence_client"
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    p = peer_file("WS1DP", "src/sisteer/eval/request_evidence_client.py")
    if p is None:
        pytest.skip("WS1-D′'s request_evidence_client is on neither this tree nor a sibling worktree")
    import sisteer.eval as pkg
    mod = _load_abs(name, p)
    setattr(pkg, "request_evidence_client", mod)
    return mod


def load_ws3g_audit():
    """``audit_common`` + ``request_evidence_audit`` + the synthetic-root fixtures (``seedrep_audit_fixtures``),
    by path (this tree first, else the sibling worktree); ``pytest.skip`` when absent. Returns ``(ac, rea, fx)``."""
    import pytest

    files = {"audit_common": "scripts/sdfbase/seedrep/audit_common.py",
             "request_evidence_audit": "scripts/sdfbase/seedrep/request_evidence_audit.py",
             "seedrep_audit_fixtures": "tests/seedrep_audit_fixtures.py"}
    paths = {k: peer_file("WS3G", v) for k, v in files.items()}
    if any(p is None for p in paths.values()):
        pytest.skip(f"WS3-G's audit modules are on neither this tree nor a sibling worktree ({[k for k, p in paths.items() if p is None]})")
    return tuple(sys.modules.get(name) or _load_abs(name, paths[name]) for name in files)


def load_ws2d_launch_plan():
    """``scripts/modal/seedrep_launch_plan.py`` by path (stdlib-only at import), for pinning the attempt-root spelling to ITS
    encoder; ``pytest.skip`` when absent."""
    import pytest

    p = peer_file("WS2D", "scripts/modal/seedrep_launch_plan.py")
    if p is None:
        pytest.skip("WS2-D's seedrep_launch_plan.py is on neither this tree nor a sibling worktree")
    return sys.modules.get("seedrep_launch_plan") or _load_abs("seedrep_launch_plan", p)
