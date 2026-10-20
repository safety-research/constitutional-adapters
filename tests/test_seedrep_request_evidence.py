"""Qwen seed-replicate battery — the request-evidence client.

Gate 2, per leg: the PRODUCTION headers (the store's canonical row task through ``store_task_name``, the
slash-free item id / numeric index through ``label_headers``) go through the REAL ``contracts.
request_evidence_record`` (non-null ``item_id``, numeric ``sample_idx``) and the FULL ``contracts.
request_evidence_reconciliation`` against STORE-DERIVED filed pairs (the module's ``filed_pairs_*`` over
synthetic rows of each store's real row schema) with a matching fake ledger, the leg's registry sampling
row and its seed closure → ADMITTED (``None``) for every one of the nine legs. Known-bad: the lane labels
``gsm8k/none``, ``inspect/xstest``, ``inspect/or_bench_hard_1k`` → "filed item(s) have NO recorded attempt".

Gate 3: a forced first-attempt failure + retry through the REAL ``ProxyClientEngine`` against a local fake
proxy → two records, two ledger rows (one rejected), the filed row's ``request_id`` / ``completion_id`` bind
to EXACTLY ONE engine-log request (``serving_attestation.is_engine_id_of``).

Every transport the drivers use is exercised against the same fake proxy: the proxy client (exact urllib
bytes), an aiohttp session under the TraceConfig (exact chunk bytes), an httpx client under the hooks, the
misalign runner's ``HttpxJson`` under ``wrap_post_json``, the tinycap Inspect provider's response hook (row
binding onto ``ItemResult``), and the usage ledger's ``completion_id`` column.

Default-off invariance: with ``SISTEER_REQUEST_EVIDENCE_DIR`` absent every ``maybe_*`` is the identity,
the served target's ``turn_record`` / BoN's ``ServedTurn`` / the misalign row / the Inspect row / the ledger
row keep their pre-hook key sets, and the misalign transport posts byte-identical bytes.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from sisteer.capeval.misalign import runner as mr
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import usage_ledger as ul
from sisteer.capeval.serving.proxy import ProxyClientEngine, ProxyError
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import request_evidence_client as rev
from sisteer.eval import sampled_controls as sc
from sisteer.eval.attacks import bon
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import served_target as st
from sisteer.eval.attacks.pyrit_adapters import multiturn_seed
from sisteer.eval.attacks.static_common import static_sampling_seed

VENDOR = C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY["qwen35_4b"]
GSM8K_ROW = C.SamplingRow(temperature=0.6, top_p=0.95, top_k=20, source="thinking_card (test)", min_p=0.0,
                          presence_penalty=0.0)
GREEDY = C.GREEDY_SAMPLING
VMODEL = "petri__base"
TWIN = "petri__none__lora0"
S = 2   # the seed base under test (a literal training seed)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _hook_off(monkeypatch):
    """Every test starts with the lane switch OFF and the per-env cache empty."""
    monkeypatch.delenv(rev.ENV_DIR, raising=False)
    monkeypatch.delenv(rev.ENV_EXTRA, raising=False)
    monkeypatch.delenv(rev.ENV_SEED_BASE, raising=False)
    rev.reset_env_cache()
    yield
    rev.reset_env_cache()


@pytest.fixture()
def rec(tmp_path) -> rev.RequestEvidenceClient:
    return rev.RequestEvidenceClient(tmp_path / "ev", extra={"leg": "static_pasttense", "arm": "none", "s": S,
                                                            "attempt_id": "ta-11111111-2222"})


class _FakeProxy:
    """A loopback OpenAI-compatible stand-in: records every raw body + headers, answers with a completion
    whose ``id`` counts up; ``fail_next`` makes the next chat POST a 500 (JSON error body)."""

    def __init__(self) -> None:
        outer = self
        self.raw: List[bytes] = []
        self.headers: List[Dict[str, str]] = []
        self.paths: List[str] = []
        self.n = 0
        self.fail_next = 0

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                return

            def _send(self, code, doc):
                data = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"object": "list", "data": [{"id": VMODEL, "owned_by": "petri"},
                                                            {"id": TWIN, "owned_by": "petri"}]})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                outer.paths.append(self.path)
                outer.raw.append(raw)
                outer.headers.append({k: v for k, v in self.headers.items()})
                body = json.loads(raw or b"{}")
                if self.path.endswith("/tokenize"):
                    self._send(200, {"count": 7, "max_model_len": 16384, "model": body.get("model", "m")})
                    return
                if outer.fail_next > 0:
                    outer.fail_next -= 1
                    self._send(500, {"error": {"message": "engine hiccup", "code": "boom"}})
                    return
                outer.n += 1
                self._send(200, {"id": f"cmpl-{outer.n}", "object": "chat.completion", "created": 0,
                                 "model": body.get("model", "m"),
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": f"reply {outer.n}"}}],
                                 "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}})

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        host, port = self._srv.server_address[:2]
        return f"http://{host}:{port}/v1"

    def chat_posts(self) -> List[Tuple[bytes, Dict[str, str]]]:
        return [(r, h) for p, r, h in zip(self.paths, self.raw, self.headers) if p.endswith("/chat/completions")]

    def close(self):
        self._srv.shutdown()
        self._srv.server_close()


@pytest.fixture()
def proxy():
    p = _FakeProxy()
    try:
        yield p
    finally:
        p.close()


def _lower(h: Mapping[str, str]) -> Dict[str, str]:
    return {k.lower(): v for k, v in h.items()}


def _hdr(h: Mapping[str, str], name: str) -> Optional[str]:
    return _lower(h).get(name.lower())


def _vendor_body(vmodel: str, prompt: str, seed: int, *, row: C.SamplingRow = VENDOR, cap: int = 4096,
                 thinking: bool = False) -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": vmodel, "messages": [{"role": "user", "content": prompt}], "max_tokens": cap,
                            "temperature": row.temperature, "seed": seed,
                            **C.chat_template_kwargs(C.THINKING_ON if thinking else C.THINKING_OFF)}
    for k in ("top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty"):
        v = getattr(row, k)
        if v is not None:
            body[k] = v
    return body


def _mmlu_body(vmodel: str, item: str) -> Dict[str, Any]:
    return {"model": vmodel, "messages": [{"role": "user", "content": item}, {"role": "assistant", "content": "Answer:"}],
            "temperature": 0.0, "max_tokens": 1, "seed": 0, "logprobs": True, "top_logprobs": 64,
            "continue_final_message": True, "add_generation_prompt": False, **C.chat_template_kwargs(C.THINKING_OFF)}


def _ledger_row(raw: bytes, headers: Mapping[str, str], *, status: int = 200) -> Dict[str, Any]:
    """What the usage ledger writes for one attempt: the labels and id it CONSUMED off the wire, the digest of
    the bytes it RECEIVED (the same bytes the client recorded)."""
    row = dict.fromkeys(C.LEDGER_ROW_FIELDS)
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        body = {}
    row.update(ul.body_facts(body) if isinstance(body, Mapping) else {})   # vmodel / thinking / cap, as the ledger reads them
    row.update(task=_hdr(headers, C.HEADER_TASK), sample=_hdr(headers, C.HEADER_SAMPLE),
               request_id=_hdr(headers, C.HEADER_REQUEST_ID), body_sha256=C.wire_body_sha256(raw),
               rejected=not (200 <= status < 300), status=status)
    return row


def _wire(body: Mapping[str, Any]) -> bytes:
    return json.dumps(body).encode("utf-8")


# --------------------------------------------------------------------------- #
# the canonical task names come from the writers                               #
# --------------------------------------------------------------------------- #

def test_store_task_names_are_the_writers_own_strings():
    from sisteer.capeval.tiers import judged, parity
    from sisteer.capeval.tiers.petri_runner import PETRI_TASK_PREFIX

    assert rev.store_task_name(rev.LEG_GSM8K) == C.inspect_task(C.GSM8K_PAIRED_TASK_NAME) == "inspect/gsm8k_paired200"
    assert rev.store_task_name(rev.LEG_XSTEST) == judged.TASK_XSTEST == "judged/xstest_v2"
    assert rev.store_task_name(rev.LEG_OR_BENCH) == judged.TASK_ORBENCH == "judged/or_bench_hard_1k"
    assert rev.store_task_name(rev.LEG_MMLU) == parity.PARITY_TASK_MMLU == "s2parity/mmlu_letter"
    assert rev.store_task_name(rev.LEG_STATIC, "pasttense") == "static/pasttense"
    assert rev.store_task_name(rev.LEG_BON) == "strongreject_bon"
    assert rev.store_task_name(rev.LEG_MULTI_TURN, "crescendo") == "multi_turn/crescendo"
    assert rev.store_task_name(rev.LEG_MISALIGN, "leaking") == "agentic/leaking"
    assert rev.store_task_name(rev.LEG_MISALIGN, "murder", family="em8q") == "em8q/murder"
    assert rev.store_task_name(rev.LEG_PETRI, "self_preservation") == PETRI_TASK_PREFIX + "self_preservation"
    # known-bad: an unknown leg, a missing sub-key, a sub-key with a slash
    with pytest.raises(rev.RequestEvidenceError):
        rev.store_task_name("controls")
    with pytest.raises(rev.RequestEvidenceError):
        rev.store_task_name(rev.LEG_STATIC)
    with pytest.raises(rev.RequestEvidenceError):
        rev.store_task_name(rev.LEG_MULTI_TURN, "pair/2")


def test_label_headers_are_ledger_headers_and_refuse_a_slash_or_a_bad_index():
    h = rev.label_headers("static/pasttense", "jbb_0012", 3)
    assert h == {C.HEADER_TASK: "static/pasttense", C.HEADER_SAMPLE: C.sample_header_value("jbb_0012", 3)}
    assert h[C.HEADER_SAMPLE] == "jbb_0012/3"
    for bad in ("a/b", "", None, 12):
        with pytest.raises(rev.RequestEvidenceError):
            rev.label_headers("static/pasttense", bad, 0)  # type: ignore[arg-type]
    with pytest.raises(rev.RequestEvidenceError):
        rev.label_headers("static/pasttense", "g", -1)
    with pytest.raises(rev.RequestEvidenceError):
        rev.label_headers("static/pasttense", "g", True)  # type: ignore[arg-type]
    with pytest.raises(rev.RequestEvidenceError):
        rev.label_headers("", "g", 0)


def test_filed_pairs_builders_use_the_same_task_strings():
    static_rows = [{"family": "renellm", "goal_id": "g1", "sample_idx": 0}, {"family": "renellm", "goal_id": "g1", "sample_idx": 1},
                   {"family": "renellm", "goal_id": "g2", "sample_idx": 0}]
    assert rev.filed_pairs_static(static_rows) == {("static/renellm", "g1"), ("static/renellm", "g2")}
    assert rev.filed_pairs_bon([{"goal_id": "j1", "sample_idx": 4}, {"goal_id": "j1", "sample_idx": 5}]) == {("strongreject_bon", "j1")}
    assert rev.filed_pairs_multiturn([{"goal_id": "g9", "order": 0}], "pair") == {("multi_turn/pair", "g9")}
    assert rev.filed_pairs_misalign([{"family": "agentic", "behavior": "murder", "item_id": "m_03", "sample_idx": 0}]) == {("agentic/murder", "m_03")}
    keys = [("petri/self_preservation", "scn_01", "misaligned"), ("petri/self_preservation", "__aggregate__", "n")]
    assert rev.filed_pairs_petri(keys) == C.filed_item_pairs(keys) == {("petri/self_preservation", "scn_01")}
    assert rev.filed_pairs_tinycap([("inspect/gsm8k_paired200", "gsm8k_00ab", "acc")]) == {("inspect/gsm8k_paired200", "gsm8k_00ab")}
    assert rev.filed_pairs_judged("judged/xstest_v2", [{"item_id": "xs_1"}]) == {("judged/xstest_v2", "xs_1")}


# --------------------------------------------------------------------------- #
# gate 2: per leg, the production headers through the REAL reconciliation      #
# --------------------------------------------------------------------------- #

def _misalign_seed(s: int, behavior: str, item_id: str, sample_idx: int) -> int:
    """The misalign leg's seed rule (restated here as the leg's closure)."""
    return int(hashlib.sha256(f"{s}|misalign|{behavior}|{item_id}|{sample_idx}".encode()).hexdigest()[:8], 16)


def _petri_seed(s: int, behaviour: str, scenario_id: str, turn: int) -> int:
    """The petri leg's ``petri_target_seed``."""
    return int.from_bytes(hashlib.sha256(f"{s}|petri|{behaviour}|{scenario_id}|{turn}".encode()).digest()[:4], "big")


def _legs() -> Dict[str, dict]:
    """Per leg: the canonical task, the (item, idx) draws a store would file, the filed pairs DERIVED from that
    store's own row schema through the module's builders, the seed closure, the body builder, the row + render."""
    static_task = rev.store_task_name(rev.LEG_STATIC, "pasttense")
    static_rows = [{"family": "pasttense", "goal_id": g, "sample_idx": i, "trial": 0} for g in ("hb_0001", "jbb_0002") for i in (0, 1)]
    bon_task = rev.store_task_name(rev.LEG_BON)
    bon_rows = [{"goal_id": g, "sample_idx": i, "trial": 0} for g in ("jbb_0007",) for i in (0, 1, 2)]
    mt_task = rev.store_task_name(rev.LEG_MULTI_TURN, "pair")
    mt_rows = [{"goal_id": "hb_0042", "order": t} for t in (0, 1, 2)]
    mis_task = rev.store_task_name(rev.LEG_MISALIGN, "leaking")
    mis_rows = [{"family": "agentic", "behavior": "leaking", "item_id": f"leak_{i:02d}", "sample_idx": k} for i in (1, 2) for k in (0, 1)]
    petri_task = rev.store_task_name(rev.LEG_PETRI, "self_preservation")
    petri_keys = [(petri_task, "scn_01", "misaligned"), (petri_task, "scn_02", "misaligned"), (petri_task, C.AGG_ITEM_ID, "n")]
    gsm_task = rev.store_task_name(rev.LEG_GSM8K)
    gsm_keys = [(gsm_task, "gsm8k_00ab", "acc"), (gsm_task, "gsm8k_11cd", "acc"), (gsm_task, C.AGG_ITEM_ID, "acc_aggregate")]
    xs_task, ob_task, mmlu_task = rev.store_task_name(rev.LEG_XSTEST), rev.store_task_name(rev.LEG_OR_BENCH), rev.store_task_name(rev.LEG_MMLU)
    xs_rows, ob_rows = [{"item_id": "xstest_aaaa1111"}, {"item_id": "xstest_bbbb2222"}], [{"item_id": "orb_0001"}]
    mmlu_keys = [(mmlu_task, "mmlu_0001", "acc"), (mmlu_task, "mmlu_0002", "acc"), (mmlu_task, C.AGG_ITEM_ID, "n")]
    vend = lambda item, idx, seed: _vendor_body(VMODEL, f"prompt {item} {idx}", seed)  # noqa: E731
    return {
        "static": dict(task=static_task, draws=[(r["goal_id"], r["sample_idx"]) for r in static_rows],
                       filed=rev.filed_pairs_static(static_rows), row=VENDOR, thinking=False,
                       seed=lambda t, i, k: static_sampling_seed(S, 0, t.split("/", 1)[1], i, k), body=vend),
        "bon": dict(task=bon_task, draws=[(r["goal_id"], r["sample_idx"]) for r in bon_rows], filed=rev.filed_pairs_bon(bon_rows),
                    row=VENDOR, thinking=False, seed=lambda t, i, k: bon.bon_decode_seed_v1(i, k, 0, S), body=vend),
        "multi_turn": dict(task=mt_task, draws=[(r["goal_id"], r["order"]) for r in mt_rows], filed=rev.filed_pairs_multiturn(mt_rows, "pair"),
                           row=VENDOR, thinking=False, seed=lambda t, i, k: multiturn_seed(S, 0, "pair", i, k), body=vend),
        "misalign": dict(task=mis_task, draws=[(r["item_id"], r["sample_idx"]) for r in mis_rows], filed=rev.filed_pairs_misalign(mis_rows),
                         row=VENDOR, thinking=False, seed=lambda t, i, k: _misalign_seed(S, t.split("/", 1)[1], i, k),
                         body=lambda item, idx, seed: _vendor_body(VMODEL, f"prompt {item}", seed, cap=3000)),
        "petri": dict(task=petri_task, draws=[("scn_01", 1), ("scn_01", 2), ("scn_02", 1)], filed=rev.filed_pairs_petri(petri_keys),
                      row=VENDOR, thinking=False, seed=lambda t, i, k: _petri_seed(S, t.split("/", 1)[1], i, k),
                      body=lambda item, idx, seed: _vendor_body(VMODEL, f"turn {idx} of {item}", seed, cap=1024)),
        "gsm8k": dict(task=gsm_task, draws=[("gsm8k_00ab", 0), ("gsm8k_11cd", 0)], filed=rev.filed_pairs_tinycap(gsm_keys),
                      row=GSM8K_ROW, thinking=True, seed=lambda t, i, k: C.inspect_replicate_seed(i, k, S),
                      body=lambda item, idx, seed: _vendor_body(VMODEL, item, seed, row=GSM8K_ROW, cap=8192, thinking=True)),
        "xstest": dict(task=xs_task, draws=[(r["item_id"], 0) for r in xs_rows], filed=rev.filed_pairs_judged(xs_task, xs_rows),
                       row=VENDOR, thinking=False, seed=lambda t, i, k: sc.sampling_seed(S, 0, "xstest", i),
                       body=lambda item, idx, seed: _vendor_body(VMODEL, item, seed, cap=512)),
        "or_bench_hard_1k": dict(task=ob_task, draws=[(r["item_id"], 0) for r in ob_rows], filed=rev.filed_pairs_judged(ob_task, ob_rows),
                                 row=VENDOR, thinking=False, seed=lambda t, i, k: sc.sampling_seed(S, 0, "or_bench_hard_1k", i),
                                 body=lambda item, idx, seed: _vendor_body(VMODEL, item, seed, cap=512)),
        "mmlu": dict(task=mmlu_task, draws=[("mmlu_0001", 0), ("mmlu_0002", 0)], filed=rev.filed_pairs_tinycap(mmlu_keys),
                     row=GREEDY, thinking=False, seed=lambda t, i, k: 0, body=lambda item, idx, seed: _mmlu_body(VMODEL, item)),
    }


def _evidence_for(rec: rev.RequestEvidenceClient, leg: dict, *, task_override: Optional[str] = None):
    """Every draw of the leg as ONE attempt each: the production headers (the label context + the minted id),
    the record from the wire bytes, the ledger's row of the same attempt."""
    records, rows = [], []
    for item, idx in leg["draws"]:
        with rev.labelled(task_override or leg["task"], item, idx):
            body = leg["body"](item, idx, leg["seed"](leg["task"], item, idx))
            hdrs = rec.headers_for(body)
        raw = _wire(body)
        records.append(rec.record_request(raw, hdrs))
        rows.append(_ledger_row(raw, hdrs))
    return records, rows


@pytest.mark.parametrize("name", sorted(_legs()))
def test_gate2_every_leg_is_admitted_by_the_real_reconciliation(rec, name):
    leg = _legs()[name]
    records, rows = _evidence_for(rec, leg)
    # the v3 record parses the trial identity off the production header
    for r, (item, idx) in zip(records, leg["draws"]):
        assert r["kind"] == "task" and r["task"] == leg["task"]
        assert r["item_id"] == item and r["sample_idx"] == idx and isinstance(r["sample_idx"], int)
        assert r["sample"] == C.sample_header_value(item, idx) and r["request_id"]
        assert r["vmodel"] == VMODEL and r["extra"] == rec.extra
    # the on-disk records are the returned ones, one JSON line each
    on_disk = [json.loads(l) for l in rec.request_path.read_text().splitlines()]
    assert on_disk == records
    # THE VERDICT: identity both ways, coverage over the store-derived filed pairs, the sampling + seed gate
    verdict = C.request_evidence_reconciliation(
        records, rows, filed=leg["filed"], expected_enable_thinking=leg["thinking"],
        expected_sampling=leg["row"], expected_seed=leg["seed"], target_vmodel=VMODEL)
    assert verdict is None, verdict


@pytest.mark.parametrize("name,bad_task", [("gsm8k", "gsm8k/none"), ("xstest", "inspect/xstest"),
                                           ("or_bench_hard_1k", "inspect/or_bench_hard_1k")])
def test_gate2_known_bad_lane_labels_are_refused_as_unattempted_filed_items(rec, name, bad_task):
    """The rev-5/6 defect the plan closed: a task header that is not the store's own task (a lane label; the
    INSPECT variants of the sampled controls) reconciles on identity — the ledger copied the same header — but
    every filed item then has no recorded attempt."""
    leg = _legs()[name]
    records, rows = _evidence_for(rec, leg, task_override=bad_task)
    verdict = C.request_evidence_reconciliation(
        records, rows, filed=leg["filed"], expected_enable_thinking=leg["thinking"],
        expected_sampling=leg["row"], expected_seed=leg["seed"], target_vmodel=VMODEL)
    assert verdict is not None and "filed item(s) have NO recorded attempt" in verdict, verdict
    # and the canonical label on the SAME evidence passes (the label is the only difference)
    rec2 = rev.RequestEvidenceClient(rec.out_dir / "ok", seed_base=S)
    good_records, good_rows = _evidence_for(rec2, leg)
    assert C.request_evidence_reconciliation(good_records, good_rows, filed=leg["filed"], expected_enable_thinking=leg["thinking"],
                                             expected_sampling=leg["row"], expected_seed=leg["seed"], target_vmodel=VMODEL) is None


def test_gate2_a_wrong_seed_or_an_omitted_presence_penalty_is_refused(rec):
    leg = _legs()["static"]
    records, rows = _evidence_for(rec, leg)
    wrong = lambda t, i, k: leg["seed"](t, i, k) + 1  # noqa: E731
    v = C.request_evidence_reconciliation(records, rows, filed=leg["filed"], expected_enable_thinking=False,
                                          expected_sampling=VENDOR, expected_seed=wrong, target_vmodel=VMODEL)
    assert v is not None and "seed" in v
    # a body without presence_penalty (a server default would fill the value; the WIRE shows absence)
    item, idx = leg["draws"][0]
    with rev.labelled(leg["task"], item, idx):
        body = leg["body"](item, idx, leg["seed"](leg["task"], item, idx))
        body.pop("presence_penalty")
        hdrs = rec.headers_for(body)
    raw = _wire(body)
    r, row = rec.record_request(raw, hdrs), _ledger_row(raw, hdrs)
    assert r["presence_penalty_present"] is False
    v = C.request_evidence_reconciliation([r], [row], filed={(leg["task"], item)}, expected_enable_thinking=False,
                                          expected_sampling=VENDOR, expected_seed=leg["seed"], target_vmodel=VMODEL)
    assert v is not None and "presence_penalty was NOT on the wire" in v


def test_an_unlabelled_send_is_refused_and_a_probe_call_site_gets_a_probe_label(rec):
    body = _vendor_body(VMODEL, "p", 1)
    with pytest.raises(rev.RequestEvidenceError, match="unlabelled"):
        rec.headers_for(body)
    h = rec.headers_for(body, unlabelled_kind=rev.PROBE_KIND_BITES)
    assert (h[C.HEADER_TASK], h[C.HEADER_SAMPLE]) == (C.probe_task_label(C.PROBE_KIND_BITES), f"{VMODEL}/0")
    h2 = rec.headers_for(body, unlabelled_kind=rev.PROBE_KIND_BITES)
    assert h2[C.HEADER_SAMPLE] == f"{VMODEL}/1" and h2[C.HEADER_REQUEST_ID] != h[C.HEADER_REQUEST_ID]
    r = rec.record_request(_wire(body), h)
    assert r["kind"] == "probe" and r["item_id"] is None and r["sample_idx"] is None
    # a caller's own labels + id are kept verbatim; one label without the other refuses
    mine = {C.HEADER_TASK: "static/grok420", C.HEADER_SAMPLE: "g/0", C.HEADER_REQUEST_ID: "rid-mine"}
    assert rec.headers_for(body, headers=mine) == mine
    with pytest.raises(rev.RequestEvidenceError, match="one tinycap label"):
        rec.headers_for(body, headers={C.HEADER_TASK: "static/grok420"})
    # the label context refuses a slashed item id at set time
    with pytest.raises(rev.RequestEvidenceError):
        with rev.labelled("static/grok420", "a/b", 0):
            pass


# --------------------------------------------------------------------------- #
# the proxy-client transport (ServedChatTarget / BoN): exact bytes, gate 3      #
# --------------------------------------------------------------------------- #

def _client(proxy: _FakeProxy) -> ProxyClientEngine:
    c = ProxyClientEngine(proxy.base_url, identity="proxy_vllm+lora+interleaved", timeout_s=10)
    c.load(require_virtual_models=[VMODEL])
    return c


def test_attach_proxy_client_records_the_exact_urllib_bytes_and_binds_the_response(rec, proxy):
    client = rec.attach_proxy_client(_client(proxy))
    task = rev.store_task_name(rev.LEG_STATIC, "codeattack")
    body = _vendor_body(VMODEL, "hello", static_sampling_seed(S, 0, "codeattack", "hb_0001", 0))
    with rev.labelled(task, "hb_0001", 0):
        resp = client.chat_completions(body)
    [(raw, wire_h)] = proxy.chat_posts()
    [r] = [json.loads(l) for l in rec.request_path.read_text().splitlines()]
    assert r["body_sha256"] == C.wire_body_sha256(raw) and r["body_bytes"] == len(raw)
    assert (_hdr(wire_h, C.HEADER_TASK), _hdr(wire_h, C.HEADER_SAMPLE)) == (task, "hb_0001/0")
    assert _hdr(wire_h, C.HEADER_REQUEST_ID) == r["request_id"] and len(r["request_id"]) == 32
    assert (r["seed"], r["presence_penalty_present"], r["presence_penalty_value"]) == (body["seed"], True, 1.5)
    [resp_row] = [json.loads(l) for l in rec.response_path.read_text().splitlines()]
    assert set(resp_row) == set(rev.RESPONSE_EVIDENCE_FIELDS) | {rev.EXTRA_FIELD}   # written sort_keys=True, like the records
    assert (resp_row["request_id"], resp_row["completion_id"], resp_row["status"], resp_row["finish_reason"]) == (r["request_id"], "cmpl-1", 200, "stop")
    assert resp_row["task"] == task and resp_row["sample"] == "hb_0001/0" and resp_row["extra"] == rec.extra
    assert rec.row_fields(resp) == {"request_id": r["request_id"], "completion_id": "cmpl-1"}
    assert rec.row_fields({"id": "cmpl-unknown"}) == {} and rec.binding_of(None) is None
    # attaching twice is idempotent
    assert rec.attach_proxy_client(client) is client
    # the ledger row of that attempt (built as the ledger would) reconciles with the record
    verdict = C.request_evidence_reconciliation([r], [_ledger_row(raw, wire_h)], filed={(task, "hb_0001")},
                                                expected_enable_thinking=False, expected_sampling=VENDOR,
                                                expected_seed=lambda t, i, k: static_sampling_seed(S, 0, "codeattack", i, k),
                                                target_vmodel=VMODEL)
    assert verdict is None, verdict


def test_gate3_a_failed_first_attempt_and_its_retry_bind_the_filed_row_to_exactly_one_engine_request(rec, proxy):
    client = rec.attach_proxy_client(_client(proxy))
    task = rev.store_task_name(rev.LEG_BON)
    body = _vendor_body(VMODEL, "variant", bon.bon_decode_seed_v1("jbb_0007", 3, 0, S))
    proxy.fail_next = 1
    with rev.labelled(task, "jbb_0007", 3):
        with pytest.raises(ProxyError):
            client.chat_completions(body)          # attempt 1: the engine 500s
        resp = client.chat_completions(body)       # attempt 2: served
    posts = proxy.chat_posts()
    assert len(posts) == 2
    records = [json.loads(l) for l in rec.request_path.read_text().splitlines()]
    responses = [json.loads(l) for l in rec.response_path.read_text().splitlines()]
    assert len(records) == 2 and records[0]["request_id"] != records[1]["request_id"]
    assert [x["status"] for x in responses] == [500, 200]
    assert responses[0]["completion_id"] is None and "engine hiccup" in responses[0]["error"]
    assert responses[1]["completion_id"] == "cmpl-1" and responses[1]["error"] is None
    # the ledger: two rows, the first rejected — both reconcile by identity; coverage is the filed BoN item
    rows = [_ledger_row(raw, h, status=(500 if k == 0 else 200)) for k, (raw, h) in enumerate(posts)]
    verdict = C.request_evidence_reconciliation(records, rows, filed={(task, "jbb_0007")}, expected_enable_thinking=False,
                                                expected_sampling=VENDOR,
                                                expected_seed=lambda t, i, k: bon.bon_decode_seed_v1(i, k, 0, S), target_vmodel=VMODEL)
    assert verdict is None, verdict
    # THE BINDING: the filed row carries the SECOND attempt's request id and the engine's completion id …
    binding = rec.row_fields(resp)
    assert binding == {"request_id": records[1]["request_id"], "completion_id": "cmpl-1"}
    # … and exactly ONE engine-log request is that completion (the external id + one -<8 hex> suffix)
    engine_log = ["Received request cmpl-1-0a1b2c3d", "Received request cmpl-1-0a1b2c3d-deadbeef",
                  "Received request cmpl-10-0a1b2c3d", "Received request cmpl-2-ffffffff"]
    ids = [re.sub(r"^Received request ", "", line) for line in engine_log]
    bound = [e for e in ids if sa.is_engine_id_of(e, binding["completion_id"])]
    assert bound == ["cmpl-1-0a1b2c3d"]
    # the failed attempt has no completion: it binds to no engine request, and a row that named it would be a failure
    assert responses[0]["completion_id"] is None
    assert not any(sa.is_engine_id_of(e, responses[0]["completion_id"]) for e in ids)


def test_served_chat_target_labels_static_draws_and_stamps_the_binding(rec, proxy, monkeypatch):
    monkeypatch.setenv(rev.ENV_DIR, str(rec.out_dir))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(S))
    rev.reset_env_cache()
    client = rev.maybe_attach_proxy_client(_client(proxy), unlabelled_kind=rev.PROBE_KIND_BITES)
    target = st.ServedChatTarget(client.chat_completions, virtual_model=VMODEL, cell_id="none", serving_engine="lora_native")
    preset = {k: getattr(VENDOR, k) for k in bs.SAMPLER_KEYS if getattr(VENDOR, k) is not None}
    task = rev.store_task_name(rev.LEG_STATIC, "pasttense")
    seeds = [static_sampling_seed(S, 0, "pasttense", "hb_0001", 0), static_sampling_seed(S, 0, "pasttense", "hb_0002", 0)]
    # with the hook ON, an unlabelled task request is refused by name — nothing reaches the wire
    with pytest.raises(rev.RequestEvidenceError, match="no label"):
        target.generate_detailed(["p1"], max_new_tokens=64, decode=preset, seeds=seeds[:1])
    assert proxy.chat_posts() == []
    # a misaligned labels list refuses too
    with pytest.raises(st.ServedTargetError, match="labels"):
        target.generate_detailed(["p1", "p2"], max_new_tokens=64, decode=preset, seeds=seeds, labels=[(task, "hb_0001", 0)])
    details = target.generate_detailed(["p1", "p2"], max_new_tokens=64, decode=preset, seeds=seeds,
                                       labels=[(task, "hb_0001", 0), (task, "hb_0002", 0)])
    posts = proxy.chat_posts()
    assert [_hdr(h, C.HEADER_SAMPLE) for _r, h in posts] == ["hb_0001/0", "hb_0002/0"]
    assert all(_hdr(h, C.HEADER_TASK) == task for _r, h in posts)
    from_env = rev.from_env()
    records = [json.loads(l) for l in from_env.request_path.read_text().splitlines()]
    assert [d["request_id"] for d in details] == [r["request_id"] for r in records]
    assert [d["completion_id"] for d in details] == ["cmpl-1", "cmpl-2"]
    assert target.last_request_binding == {"request_id": records[1]["request_id"], "completion_id": "cmpl-2"}
    assert rev.last_binding_fields(target) == target.last_request_binding
    # the positive-control probes (the same client, no label) are probe/bites attempts, never refused
    probe = st.probe_fn(client.chat_completions, VMODEL, TWIN)
    probe("off")
    probe_records = [json.loads(l) for l in from_env.request_path.read_text().splitlines()][2:]
    assert probe_records and all(r["kind"] == "probe" and r["task"] == C.probe_task_label(C.PROBE_KIND_BITES) for r in probe_records)


def test_ws1a_per_send_contract_mints_distinct_ids_and_refuses_a_reminted_one(rec, proxy, monkeypatch):
    """The BoN seam: `request_headers_fn(goal_id, sample_idx, trial)` is called once per ACTUAL
    send — the first attempt AND each window-clamped retry — and must mint a FRESH X-Tinycap-Request-Id each call; a
    hook re-minting one draw's id is refused before the send. Here: two calls for one draw give two ids and two
    records, each bound to its own completion; presenting an already-sent id again is refused at the transport
    (nothing goes out) and at the recorder."""
    monkeypatch.setenv(rev.ENV_DIR, str(rec.out_dir))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(S))
    rev.reset_env_cache()
    client = rev.maybe_attach_proxy_client(_client(proxy))
    fn = rev.maybe_request_headers_fn(rev.LEG_BON)
    body = _vendor_body(VMODEL, "variant", bon.bon_decode_seed_v1("jbb_0007", 3, 0, S))
    h1 = fn("jbb_0007", 3, 0)                 # attempt 1 of the draw
    resp1 = client.chat_completions(body, headers=h1)
    clamped = {**body, "max_tokens": 2048}      # the clamped retry: a different body, a NEW id from the same hook
    h2 = fn("jbb_0007", 3, 0)
    resp2 = client.chat_completions(clamped, headers=h2)
    assert h1[C.HEADER_REQUEST_ID] != h2[C.HEADER_REQUEST_ID]
    assert (h1[C.HEADER_SAMPLE], h2[C.HEADER_SAMPLE]) == ("jbb_0007/3", "jbb_0007/3")
    from_env = rev.from_env()
    recs = [json.loads(l) for l in from_env.request_path.read_text().splitlines()]
    assert [r["request_id"] for r in recs] == [h1[C.HEADER_REQUEST_ID], h2[C.HEADER_REQUEST_ID]]
    assert [r["max_tokens"] for r in recs] == [4096, 2048]                        # each record = that send's exact body
    assert [r["body_sha256"] for r in recs] == [C.wire_body_sha256(raw) for raw, _h in proxy.chat_posts()]
    assert rev.row_fields(resp1) == {"request_id": h1[C.HEADER_REQUEST_ID], "completion_id": "cmpl-1"}
    assert rev.row_fields(resp2) == {"request_id": h2[C.HEADER_REQUEST_ID], "completion_id": "cmpl-2"}   # the served attempt's
    # a re-minted id (a hook handing the first attempt's headers to the retry) is refused BEFORE the send
    with pytest.raises(rev.RequestEvidenceError, match="already sent"):
        client.chat_completions(clamped, headers=h1)
    assert len(proxy.chat_posts()) == 2
    with pytest.raises(rev.RequestEvidenceError, match="already sent"):
        from_env.headers_for(clamped, headers=h2)
    with pytest.raises(rev.RequestEvidenceError, match="second time"):
        from_env.record_request(_wire(clamped), h1)
    assert len(from_env.request_path.read_text().splitlines()) == 2
    # the contract's header reader is the one reader on both sides (no third copy)
    assert rev._header is (getattr(C, "header_value", None) or C._header_value)


def test_served_chat_target_set_request_context_is_ws1a_multiturn_seam(rec, proxy, monkeypatch):
    """The drivers call `defense.set_request_context(kind="multi_turn", attack, goal_id, turn, trial)` before each
    forward and read `defense.last_request_id` / `last_completion_id` onto the row: the context labels exactly the
    NEXT send (multi_turn/<attack>, <goal_id>/<turn>) and is consumed by it; a second send without one is refused."""
    monkeypatch.setenv(rev.ENV_DIR, str(rec.out_dir))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(S))
    rev.reset_env_cache()
    client = rev.maybe_attach_proxy_client(_client(proxy), unlabelled_kind=rev.PROBE_KIND_BITES)
    target = st.ServedChatTarget(client.chat_completions, virtual_model=VMODEL, cell_id="none", serving_engine="lora_native")
    preset = {k: getattr(VENDOR, k) for k in bs.SAMPLER_KEYS if getattr(VENDOR, k) is not None}
    msgs = [{"role": "user", "content": "turn 0"}]
    target.set_request_context(kind="multi_turn", attack="pair", goal_id="hb_0042", turn=0, trial=0)
    reply = target.chat(msgs, max_new_tokens=64, do_sample=True, temperature=0.7, decode=preset,
                        sampling_seed=multiturn_seed(S, 0, "pair", "hb_0042", 0))
    [(raw, h)] = proxy.chat_posts()
    assert reply == "reply 1" and (_hdr(h, C.HEADER_TASK), _hdr(h, C.HEADER_SAMPLE)) == ("multi_turn/pair", "hb_0042/0")
    assert target.last_request_id == _hdr(h, C.HEADER_REQUEST_ID) and target.last_completion_id == "cmpl-1"
    assert target.last_request_binding == {"request_id": target.last_request_id, "completion_id": "cmpl-1"}
    # consumed: the next forward needs its own context (an unlabelled task request is refused, nothing sent)
    with pytest.raises(rev.RequestEvidenceError, match="no label"):
        target.chat(msgs, max_new_tokens=64, do_sample=True, temperature=0.7, decode=preset, sampling_seed=5)
    assert len(proxy.chat_posts()) == 1
    # ActorAttack's spelling and the static kind; an unknown kind / a missing index refuse
    target.set_request_context(kind="multi_turn", attack="actorattack", goal_id="hb_0042", turn=3, trial=0)
    target.chat(msgs, max_new_tokens=64, do_sample=True, temperature=0.7, decode=preset, sampling_seed=6)
    assert _hdr(proxy.chat_posts()[-1][1], C.HEADER_SAMPLE) == "hb_0042/3" and _hdr(proxy.chat_posts()[-1][1], C.HEADER_TASK) == "multi_turn/actorattack"
    target.set_request_context(kind="static", family="grok420", goal_id="g9", sample_idx=2)
    assert target._request_context == rev.Label("static/grok420", "g9", 2)
    with pytest.raises(st.ServedTargetError, match="unknown kind"):
        target.set_request_context(kind="petri", goal_id="x", turn=0)
    with pytest.raises(st.ServedTargetError, match="needs goal_id"):
        target.set_request_context(kind="multi_turn", attack="pair", goal_id="g", turn=None)
    # a `labelled(...)` context wins over a pending request context (the pending one is left for its own send)
    target.set_request_context(kind="multi_turn", attack="pair", goal_id="later", turn=9, trial=0)
    with rev.labelled("multi_turn/crescendo", "now", 1):
        target.chat(msgs, max_new_tokens=64, do_sample=True, temperature=0.7, decode=preset, sampling_seed=7)
    assert _hdr(proxy.chat_posts()[-1][1], C.HEADER_SAMPLE) == "now/1" and target._request_context.item_id == "later"
    # OFF: the seam is a no-op (no label, nothing recorded, nothing refused)
    monkeypatch.delenv(rev.ENV_DIR)
    rev.reset_env_cache()
    plain = st.ServedChatTarget(_client(proxy).chat_completions, virtual_model=VMODEL, cell_id="none", serving_engine="lora_native")
    plain.set_request_context(kind="multi_turn", attack="pair", goal_id="hb_0042", turn=0, trial=0)
    assert plain._request_context is None
    plain.chat(msgs, max_new_tokens=64, do_sample=True, temperature=0.7, decode=preset, sampling_seed=8)
    assert C.HEADER_TASK.lower() not in _lower(proxy.chat_posts()[-1][1]) and plain.last_request_id is None


def test_bon_served_defense_over_client_labels_each_draw_and_the_turn_carries_the_binding(rec, proxy, monkeypatch):
    monkeypatch.setenv(rev.ENV_DIR, str(rec.out_dir))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(S))
    rev.reset_env_cache()
    defense = bs.ServedDefense.over_client(_client(proxy), virtual_model=VMODEL, cell_id="none", serving_engine="lora_native")
    preset = {k: getattr(VENDOR, k) for k in bs.SAMPLER_KEYS if getattr(VENDOR, k) is not None}
    seed = bon.bon_decode_seed_v1("jbb_0007", 0, 0, S)

    async def one():
        with rev.labelled(rev.store_task_name(rev.LEG_BON), "jbb_0007", 0):
            return await defense.generate_one_detailed("variant", sampling_seed=seed, max_new_tokens=64,
                                                       temperature=0.7, decode=preset)
    turn = asyncio.run(one())
    [(raw, h)] = proxy.chat_posts()
    assert (_hdr(h, C.HEADER_TASK), _hdr(h, C.HEADER_SAMPLE)) == ("strongreject_bon", "jbb_0007/0")
    rid = _hdr(h, C.HEADER_REQUEST_ID)
    from_env = rev.from_env()
    [r] = [json.loads(l) for l in from_env.request_path.read_text().splitlines()]
    [resp_row] = [json.loads(l) for l in from_env.response_path.read_text().splitlines()]
    assert r["request_id"] == rid and r["body_sha256"] == C.wire_body_sha256(raw) and r["extra"]["s"] == S
    assert (resp_row["request_id"], resp_row["completion_id"]) == (rid, "cmpl-1") and turn.text == "reply 1"
    assert from_env.bindings() == {"cmpl-1": rid}
    # The seam: bon_served_run's `request_headers_fn(goal_id, sample_idx, trial)` -> the labels + a fresh id the
    # attached transport KEEPS (the writer copies that id onto the row beside resp["id"])
    fn = rev.maybe_request_headers_fn(rev.LEG_BON)
    h1, h2 = fn("jbb_0007", 1, 0), fn("jbb_0007", 2, 0)
    assert (h1[C.HEADER_TASK], h1[C.HEADER_SAMPLE], h2[C.HEADER_SAMPLE]) == ("strongreject_bon", "jbb_0007/1", "jbb_0007/2")
    assert len(h1[C.HEADER_REQUEST_ID]) == 32 and h1[C.HEADER_REQUEST_ID] != h2[C.HEADER_REQUEST_ID]
    c2 = rev.maybe_attach_proxy_client(_client(proxy))
    resp2 = c2.chat_completions(_vendor_body(VMODEL, "v2", 7), headers=h1)
    assert _hdr(proxy.chat_posts()[-1][1], C.HEADER_REQUEST_ID) == h1[C.HEADER_REQUEST_ID]      # the caller's id, kept
    assert rev.row_fields(resp2) == {"request_id": h1[C.HEADER_REQUEST_ID], "completion_id": "cmpl-2"}
    monkeypatch.delenv(rev.ENV_DIR)
    rev.reset_env_cache()
    assert rev.maybe_request_headers_fn(rev.LEG_BON) is None                                          # off: their default


# --------------------------------------------------------------------------- #
# the other transports: aiohttp (controls / MMLU), httpx (Inspect), misalign    #
# --------------------------------------------------------------------------- #

def test_aiohttp_trace_config_records_the_exact_chunk_bytes_and_mints_the_id(rec, proxy):
    import aiohttp

    task = rev.store_task_name(rev.LEG_XSTEST)
    body = _vendor_body(VMODEL, "xstest_aaaa1111", sc.sampling_seed(S, 0, "xstest", "xstest_aaaa1111"), cap=512)

    async def go():
        async with aiohttp.ClientSession(trace_configs=[rec.aiohttp_trace_config()]) as session:
            # the controls poster's shape: json= body, the two labels as headers= (the id is minted by the trace)
            async with session.post(f"{proxy.base_url}/chat/completions", json=body,
                                    headers=rev.label_headers(task, "xstest_aaaa1111", 0)) as r:
                return r.status, json.loads(await r.text())
    status, resp = asyncio.run(go())
    assert status == 200
    [(raw, wire_h)] = proxy.chat_posts()
    [r] = [json.loads(l) for l in rec.request_path.read_text().splitlines()]
    assert r["body_sha256"] == C.wire_body_sha256(raw) and r["task"] == task and r["sample"] == "xstest_aaaa1111/0"
    assert _hdr(wire_h, C.HEADER_REQUEST_ID) == r["request_id"]
    [resp_row] = [json.loads(l) for l in rec.response_path.read_text().splitlines()]
    assert (resp_row["request_id"], resp_row["completion_id"], resp_row["status"]) == (r["request_id"], "cmpl-1", 200)
    assert rec.row_fields(resp) == {"request_id": r["request_id"], "completion_id": "cmpl-1"}
    # the labelled context works for aiohttp too (no headers= on the call), and a non-chat POST is ignored
    async def go2():
        async with aiohttp.ClientSession(trace_configs=[rec.aiohttp_trace_config()]) as session:
            with rev.labelled(task, "xstest_bbbb2222", 0):
                async with session.post(f"{proxy.base_url}/chat/completions", json=body) as r:
                    await r.text()
            async with session.post(f"{proxy.base_url}/tokenize", json={"model": VMODEL, "messages": []}) as r:
                await r.text()
    asyncio.run(go2())
    assert len(rec.request_path.read_text().splitlines()) == 2 and rec.n_responses == 2
    assert rev.maybe_aiohttp_trace_configs() == []   # off by default


def test_httpx_hooks_mint_and_bind_on_a_bare_openai_style_client(rec, proxy):
    import httpx

    task = rev.store_task_name(rev.LEG_PETRI, "self_preservation")
    body = _vendor_body(VMODEL, "turn 1", _petri_seed(S, "self_preservation", "scn_01", 1), cap=1024)

    async def go():
        async with httpx.AsyncClient() as client:
            rec.install_httpx(client)
            rec.install_httpx(client)   # idempotent
            assert sum(len(v) for v in client.event_hooks.values()) == 2
            with rev.labelled(task, "scn_01", 1):
                r = await client.post(f"{proxy.base_url}/chat/completions", json=body)
            return r.json()
    resp = asyncio.run(go())
    [(raw, wire_h)] = proxy.chat_posts()
    [r] = [json.loads(l) for l in rec.request_path.read_text().splitlines()]
    assert r["body_sha256"] == C.wire_body_sha256(raw) and (r["task"], r["sample"]) == (task, "scn_01/1")
    assert _hdr(wire_h, C.HEADER_REQUEST_ID) == r["request_id"] and _hdr(wire_h, C.HEADER_SAMPLE) == "scn_01/1"
    assert rec.row_fields(resp) == {"request_id": r["request_id"], "completion_id": "cmpl-1"}


def test_wrap_post_json_labels_the_misalign_runners_real_transport(rec, proxy):
    http = rec.wrap_post_json(mr.HttpxJson(api_key="k", timeout_s=10))
    task = rev.store_task_name(rev.LEG_MISALIGN, "leaking")
    payload = mr.build_payload(VMODEL, [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}], max_tokens=3000,
                               sampling={k: getattr(VENDOR, k) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")})
    payload["seed"] = _misalign_seed(S, "leaking", "leak_01", 0)
    seam = rev.MisalignRequestEvidence(rec)     # the run_family(request_evidence=) object
    try:
        # (a) the labelled context (a caller with no seam) and (b) the seam: headers_for -> post_json(headers=) -> record
        with rev.labelled(task, "leak_01", 0):
            resp = http.post_json(f"{proxy.base_url}/chat/completions", payload)
        hdrs = seam.headers_for(family="agentic", behavior="leaking", item_id="leak_02", sample_idx=1, payload=payload)
        assert hdrs == {C.HEADER_TASK: task, C.HEADER_SAMPLE: "leak_02/1"}                     # labels ONLY: the id is per attempt
        resp2 = http.post_json(f"{proxy.base_url}/chat/completions", payload, headers=hdrs)
        assert seam.record(payload=payload, headers=hdrs, response=resp2) == _hdr(proxy.chat_posts()[-1][1], C.HEADER_REQUEST_ID)
        assert seam.record(response={"id": "cmpl-never"}) is None
        assert http.get_json(f"{proxy.base_url}/models")["data"]
    finally:
        http.close()
    (raw, wire_h), (raw2, wire_h2) = proxy.chat_posts()
    r, r2 = [json.loads(l) for l in rec.request_path.read_text().splitlines()]
    assert raw == json.dumps(dict(payload)).encode("utf-8") and r["body_sha256"] == C.wire_body_sha256(raw)
    assert (_hdr(wire_h, C.HEADER_TASK), _hdr(wire_h, C.HEADER_SAMPLE), _hdr(wire_h, C.HEADER_REQUEST_ID)) == (task, "leak_01/0", r["request_id"])
    assert (_hdr(wire_h2, C.HEADER_SAMPLE), _hdr(wire_h2, C.HEADER_REQUEST_ID)) == ("leak_02/1", r2["request_id"]) and r2["request_id"] != r["request_id"]
    assert rec.row_fields(resp) == {"request_id": r["request_id"], "completion_id": "cmpl-1"}
    assert rev.maybe_misalign_request_evidence() is None                                            # off by default
    # a refused attempt is recorded with its status and re-raised unchanged
    proxy.fail_next = 1
    http2 = rec.wrap_post_json(mr.HttpxJson(api_key="k", timeout_s=10))
    try:
        with rev.labelled(task, "leak_02", 0), pytest.raises(mr.HttpError):
            http2.post_json(f"{proxy.base_url}/chat/completions", payload)
    finally:
        http2.close()
    rows = [json.loads(l) for l in rec.response_path.read_text().splitlines()]
    assert rows[-1]["status"] == 500 and rows[-1]["completion_id"] is None and "500" in rows[-1]["error"]


# --------------------------------------------------------------------------- #
# the Inspect provider's response hook -> the filed row                          #
# --------------------------------------------------------------------------- #

def test_tinycap_provider_response_hook_binds_the_row_and_default_off_leaves_rows_untouched(rec, monkeypatch):
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    from inspect_ai.model._generate_config import GenerateConfig as GC
    import test_inspect_served as tis

    fake = tis._Fake([tis.VMODEL, tis.BASE_VMODEL])
    try:
        identity = {"idx": 0}
        monkeypatch.setattr(isv, "current_request_identity", lambda: (C.inspect_task("math"), "math_00ab", identity["idx"]))
        api = tis._api(fake, si=None, thinking=C.THINKING_ON, reasoning_parser="gemma4")
        api.request_evidence_sink = rec.record_request
        api.response_evidence_sink = rec.response_sink
        api._install_wire_hook()
        out = tis._gen(api, tis.SHAPE_MATH, config=GC(max_tokens=8192))
        out = out[0] if isinstance(out, tuple) else out          # this SDK's generate returns (ModelOutput, ModelCall)
        [r] = [json.loads(l) for l in rec.request_path.read_text().splitlines()]
        [resp_row] = [json.loads(l) for l in rec.response_path.read_text().splitlines()]
        assert resp_row["request_id"] == r["request_id"] and resp_row["completion_id"] == "cmpl-test" and resp_row["status"] == 200
        assert out.metadata[isv.WIRE_BINDING_METADATA_KEY] == {"request_id": r["request_id"], "completion_id": "cmpl-test"}
        # the log adapter copies it onto the row — final AND rung rows go through the same ItemResult
        log = tis._Log([tis._Sample("math_00ab", {"s": tis._Score("C", "4")}, output=out)])
        [res] = isv.results_from_log(log)
        assert (res.request_id, res.completion_id) == (r["request_id"], "cmpl-test")
        extra, prov = res.extra(), res.sample_provenance()
        assert extra["request_id"] == r["request_id"] and extra["completion_id"] == "cmpl-test"
        assert prov["request_id"] == r["request_id"] and prov["completion_id"] == "cmpl-test"
        # the id the SDK parsed is held to the bound one (a disagreement refuses in on_response)
        isv._WIRE_BINDING.set({"request_id": "x", "completion_id": "cmpl-other"})
        with pytest.raises(isv.InspectServedError, match="another attempt"):
            api.on_response({"id": "cmpl-test", "model": tis.VMODEL, "choices": [], **fake.stamps(tis._route_of(tis.VMODEL))})
        isv._WIRE_BINDING.set(None)
        # DEFAULT OFF: a provider with the request sink only (every existing MATH-500 / IFEval run) installs no
        # response hook, stamps no metadata, and its rows carry neither key — byte-identical to before
        plain = tis._api(fake, si=None, thinking=C.THINKING_ON, reasoning_parser="gemma4")
        plain.request_evidence_sink = lambda raw, *, headers: None
        plain._install_wire_hook()
        assert plain.response_evidence_sink is None
        out2 = tis._gen(plain, tis.SHAPE_MATH, config=GC(max_tokens=8192))
        out2 = out2[0] if isinstance(out2, tuple) else out2
        assert isv.WIRE_BINDING_METADATA_KEY not in (out2.metadata or {})
        [res2] = isv.results_from_log(tis._Log([tis._Sample("math_00ab", {"s": tis._Score("C", "4")}, output=out2)]))
        assert res2.request_id is None and "request_id" not in res2.extra() and "completion_id" not in res2.sample_provenance()
        # a response sink without the request sink is refused: nothing minted the id it would bind
        cls = isv._register_provider()
        with pytest.raises(isv.InspectServedError, match="response_evidence_sink"):
            cls(model_name=C.inspect_model_name(tis.VMODEL), base_url=fake.base_url, api_key="t", config=GC(), si=None,
                thinking=C.THINKING_OFF, route=tis._route_of(tis.VMODEL), response_evidence_sink=rec.response_sink)
    finally:
        fake.close()


def test_run_inspect_tasks_defaults_the_response_sink_from_the_env(monkeypatch, tmp_path):
    """The env switch reaches the target provider through run_inspect_tasks (the Inspect controls entry point):
    the additive kwarg defaults to None; with SISTEER_REQUEST_EVIDENCE_DIR (+ the seed base) set, the process
    client's response_sink is what the target's build_model receives when a request sink is on."""
    import inspect as _inspect

    sig = _inspect.signature(isv.run_inspect_tasks)
    assert "response_evidence_sink" in sig.parameters and sig.parameters["response_evidence_sink"].default is None
    assert "response_evidence_sink" in _inspect.signature(isv.build_model).parameters
    assert rev.from_env() is None
    monkeypatch.setenv(rev.ENV_DIR, str(tmp_path / "ev"))
    monkeypatch.setenv(rev.ENV_SEED_BASE, "3")
    rev.reset_env_cache()
    c = rev.from_env()
    assert c is not None and c.response_sink.__func__ is rev.RequestEvidenceClient.record_response and c.seed_base == 3


# --------------------------------------------------------------------------- #
# the usage ledger's completion_id column                                       #
# --------------------------------------------------------------------------- #

def test_usage_ledger_completion_id_column_is_appended_only_under_the_flag(tmp_path):
    import test_usage_ledger as tul

    base = {k: None for k in C.LEDGER_ROW_FIELDS}
    assert tuple(ul.ledger_row(**base)) == C.LEDGER_ROW_FIELDS
    stamped = ul.ledger_row(**base, **{ul.LEDGER_COMPLETION_ID_FIELD: "cmpl-9"})
    assert tuple(stamped) == (*C.LEDGER_ROW_FIELDS, ul.LEDGER_COMPLETION_ID_FIELD)
    both = ul.ledger_row(**base, **{C.LEDGER_ATTEMPT_ID_FIELD: "a1", ul.LEDGER_COMPLETION_ID_FIELD: "cmpl-9"})
    assert tuple(both) == (*C.LEDGER_ROW_FIELDS, C.LEDGER_ATTEMPT_ID_FIELD, ul.LEDGER_COMPLETION_ID_FIELD)
    assert ul.LEDGER_COMPLETION_ID_FIELD not in C.LEDGER_ROW_FIELDS
    hdrs = {C.HEADER_TASK: "static/pasttense", C.HEADER_SAMPLE: "g/0", C.HEADER_REQUEST_ID: "rid1"}
    body = _wire(_vendor_body(VMODEL, "p", 1))
    # ON: the served row carries the response's id; a rejected row carries null in the same column
    led = ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / "on.jsonl", transport=tul.FakeTransport(), record_completion_id=True)
    asyncio.run(led.handle_chat(body, hdrs))
    led_rej = ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / "rej.jsonl", transport=tul.FakeTransport(tokenize_status=500),
                             record_completion_id=True)
    asyncio.run(led_rej.handle_chat(body, hdrs))
    [row] = ul.read_rows(tmp_path / "on.jsonl")
    [rej] = ul.read_rows(tmp_path / "rej.jsonl")
    assert tuple(row) == (*C.LEDGER_ROW_FIELDS, ul.LEDGER_COMPLETION_ID_FIELD) and row["completion_id"] == "x"
    assert row["request_id"] == "rid1" and row["body_sha256"] == C.wire_body_sha256(body)
    assert tuple(rej) == tuple(row) and rej["completion_id"] is None and rej["rejected"] is True
    # OFF (the default): the old bytes — exactly LEDGER_ROW_FIELDS
    led_off = ul.UsageLedger("http://127.0.0.1:1/v1", tmp_path / "off.jsonl", transport=tul.FakeTransport())
    asyncio.run(led_off.handle_chat(body, hdrs))
    [off] = ul.read_rows(tmp_path / "off.jsonl")
    assert tuple(off) == C.LEDGER_ROW_FIELDS and led_off.record_completion_id is False
    ap = ul.build_parser()
    ns = ap.parse_args(["--upstream", "http://x/v1", "--port", "1", "--ledger-path", "l.jsonl", "--record-completion-id"])
    assert ns.record_completion_id is True
    assert ap.parse_args(["--upstream", "http://x/v1", "--port", "1", "--ledger-path", "l.jsonl"]).record_completion_id is False


# --------------------------------------------------------------------------- #
# the Petri target: per-behaviour labelled copy, the turn definition            #
# --------------------------------------------------------------------------- #

def test_instrument_inspect_target_labels_each_generate_with_ws1b_turn_and_leaves_the_original_alone(rec, monkeypatch):
    import httpx

    seen: List[Optional[rev.Label]] = []

    async def inner_generate(input, tools, tool_choice, config):  # noqa: A002
        seen.append(rev.current_label())
        return "out"

    api = SimpleNamespace(http_client=httpx.AsyncClient(), generate=inner_generate)
    model = SimpleNamespace(api=api, config="cfg")
    monkeypatch.setattr(rev, "_active_sample_identity", lambda: ("scn_07", 1))
    task = rev.store_task_name(rev.LEG_PETRI, "defer_to_users")
    labelled = rec.instrument_inspect_target(model, task=task)
    assert labelled is not model and labelled.api is not api and labelled.api.http_client is api.http_client
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}]
    assert asyncio.run(labelled.api.generate(input=msgs, tools=[], tool_choice="auto", config=None)) == "out"
    assert seen == [rev.Label(task, "scn_07", 2)]            # 1 + one assistant message: the turn index
    assert rev.petri_turn_index(msgs) == 2 and rev.petri_turn_index([]) == 1
    assert asyncio.run(api.generate(input=msgs, tools=[], tool_choice="auto", config=None)) == "out"
    assert seen[-1] is None                                    # the original api is unlabelled
    assert getattr(api.http_client, "_sisteer_request_evidence_hooks") is rec   # the shared transport carries the hooks
    monkeypatch.setattr(rev, "_active_sample_identity", lambda: (None, 1))
    with pytest.raises(rev.RequestEvidenceError, match="outside an Inspect sample"):
        asyncio.run(labelled.api.generate(input=msgs, tools=[], tool_choice="auto", config=None))
    # the headers hook body: the two labels from a TargetTurn, nothing else
    turn = SimpleNamespace(task=task, item_id="scn_07", sample_idx=3, seed=1, seed_base=S)
    assert rev.petri_turn_headers(turn) == {C.HEADER_TASK: task, C.HEADER_SAMPLE: "scn_07/3"}
    asyncio.run(api.http_client.aclose())


def test_maybe_instrument_inspect_target_installs_ws1b_hook_when_the_seam_exists(rec, monkeypatch, tmp_path):
    from sisteer.capeval.tiers import petri_runner as pr

    model = SimpleNamespace(api=SimpleNamespace(http_client=None, generate=None))
    assert rev.maybe_instrument_inspect_target(model, task="petri/x") is model          # off: identity
    monkeypatch.setenv(rev.ENV_DIR, str(tmp_path / "ev"))
    monkeypatch.setenv(rev.ENV_SEED_BASE, "1")
    rev.reset_env_cache()
    monkeypatch.setattr(pr, "TARGET_TURN_HEADERS_HOOK", None, raising=False)
    with pytest.raises(rev.RequestEvidenceError, match="http_client"):
        rev.maybe_instrument_inspect_target(model, task="petri/x")
    assert pr.TARGET_TURN_HEADERS_HOOK is rev.petri_turn_headers


def test_maybe_install_petri_target_hooks_is_the_runners_one_call(rec, monkeypatch, tmp_path):
    """The wiring petri_runner._inspect_driver carries (position-stable, after its `if not tasks` guard): off =
    the target untouched and no seam touched; on = the request + response httpx hooks on the target api's
    http_client (shared by every per-behaviour copy the runner builds) and TARGET_TURN_HEADERS_HOOK
    pointed at petri_turn_headers; a target without an http_client refuses."""
    import httpx
    from sisteer.capeval.tiers import petri_runner as pr

    client = httpx.AsyncClient()
    model = SimpleNamespace(api=SimpleNamespace(http_client=client), config="cfg")
    monkeypatch.setattr(pr, "TARGET_TURN_HEADERS_HOOK", None, raising=False)
    assert rev.maybe_install_petri_target_hooks(model) is model and client.event_hooks == {"request": [], "response": []}
    assert pr.TARGET_TURN_HEADERS_HOOK is None
    monkeypatch.setenv(rev.ENV_DIR, str(tmp_path / "ev"))
    monkeypatch.setenv(rev.ENV_SEED_BASE, "3")
    rev.reset_env_cache()
    assert rev.maybe_install_petri_target_hooks(model) is model
    assert len(client.event_hooks["request"]) == 1 and len(client.event_hooks["response"]) == 1
    assert pr.TARGET_TURN_HEADERS_HOOK is rev.petri_turn_headers
    rev.maybe_install_petri_target_hooks(model)                                         # idempotent per client
    assert len(client.event_hooks["request"]) == 1 and len(client.event_hooks["response"]) == 1
    with pytest.raises(rev.RequestEvidenceError, match="http_client"):
        rev.maybe_install_petri_target_hooks(SimpleNamespace(api=SimpleNamespace()))
    # the runner's own seam is not always on the tree; the source carries the call regardless
    src = Path(pr.__file__).read_text()
    assert "rev.maybe_install_petri_target_hooks(target)" in src and src.index("if not tasks:") < src.index("rev.maybe_install_petri_target_hooks(target)")
    asyncio.run(client.aclose())


# --------------------------------------------------------------------------- #
# default-off invariance                                                        #
# --------------------------------------------------------------------------- #

def test_default_off_every_entry_point_is_the_identity_and_rows_keep_their_keys(monkeypatch):
    assert rev.from_env() is None and rev.enabled() is False
    client = ProxyClientEngine("http://127.0.0.1:1/v1", identity="proxy_vllm+lora+interleaved")
    assert rev.maybe_attach_proxy_client(client) is client and "chat_completions" not in vars(client) and client.on_request is None
    http = object()
    assert rev.maybe_wrap_post_json(http) is http
    assert rev.maybe_aiohttp_trace_configs() == [] and rev.row_fields({"id": "cmpl-1"}) == {}
    assert rev.multiturn_label_fn("pair", {"gid": "g"}) is None and rev.require_label("x") is None
    with rev.labelled_or_none(None) as lab:
        assert lab is None
    # the served target: a scripted chat_fn, no hook -> turn_record has exactly the pre-hook keys
    def chat_fn(req):
        return {"id": "cmpl-z", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    target = st.ServedChatTarget(chat_fn, virtual_model=VMODEL, cell_id="none", serving_engine="lora_native")
    [d] = target.generate_detailed(["p"], max_new_tokens=8)
    assert set(d) == {"text", "finish_reason", "completion_tokens", "prompt_tokens", "max_tokens", "sampling_seed"}
    assert target.last_request_binding is None and rev.last_binding_fields(target) == {}
    turn = bs.ServedTurn(text="t", finish_reason="stop", prompt_tokens=1, completion_tokens=1, max_tokens=8)
    assert rev.turn_fields(turn) == {} and set(bs.turn_record(turn, sampling_seed=None)) == set(d)
    assert rev.maybe_request_headers_fn(rev.LEG_BON) is None and rev.maybe_misalign_request_evidence() is None
    assert target._request_context is None
    # the Inspect row
    res = isv.ItemResult(item_id="i", value=1.0, sample_idx=0, seed=0, answer=None, prompt_tokens=None,
                         completion_tokens=None, paired_id=None, scores={})
    assert "request_id" not in res.extra() and "completion_id" not in res.sample_provenance()
    # the misalign transport posts the SAME bytes and headers as before when nothing is passed
    calls = []
    http = mr.HttpxJson.__new__(mr.HttpxJson)
    http._client = SimpleNamespace(post=lambda url, **kw: (calls.append((url, kw)), SimpleNamespace(status_code=200, json=lambda: {"id": "c"}, text="{}"))[1])
    import httpx
    http._httpx = httpx
    payload = {"model": VMODEL, "messages": [{"role": "user", "content": "u"}], "max_tokens": 1, "temperature": 0.7}
    http.post_json("http://x/v1/chat/completions", payload)
    [(url, kw)] = calls
    assert kw == {"content": json.dumps(dict(payload)).encode("utf-8"), "headers": {"Content-Type": "application/json"}}


def test_from_env_parses_extra_and_refuses_a_non_object(monkeypatch, tmp_path):
    monkeypatch.setenv(rev.ENV_DIR, str(tmp_path / "ev"))
    monkeypatch.setenv(rev.ENV_EXTRA, json.dumps({"leg": "bon", "arm": "prompted__si_constitution", "s": 3, "attempt_id": "ta-1"}))
    rev.reset_env_cache()
    c = rev.from_env()
    assert c is not None and c.extra == {"leg": "bon", "arm": "prompted__si_constitution", "s": 3, "attempt_id": "ta-1"}
    assert rev.from_env() is c                                  # one instance per env: the binding table is shared
    assert c.request_path == tmp_path / "ev" / C.REQUEST_EVIDENCE_FILENAME
    assert c.response_path == tmp_path / "ev" / rev.RESPONSE_EVIDENCE_FILENAME
    monkeypatch.setenv(rev.ENV_EXTRA, "[1, 2]")
    rev.reset_env_cache()
    with pytest.raises(rev.RequestEvidenceError, match="JSON object"):
        rev.from_env()
    # the seed base is REQUIRED (the Arditi audit partitions one task by extra.s): no s anywhere refuses,
    # a disagreeing pair refuses, the env scalar alone suffices and lands in extra.s
    monkeypatch.setenv(rev.ENV_EXTRA, json.dumps({"leg": "bon"}))
    rev.reset_env_cache()
    with pytest.raises(rev.RequestEvidenceError, match="seed base"):
        rev.from_env()
    monkeypatch.setenv(rev.ENV_SEED_BASE, "2")
    rev.reset_env_cache()
    assert rev.from_env().extra == {"leg": "bon", "s": 2}
    monkeypatch.setenv(rev.ENV_EXTRA, json.dumps({"leg": "bon", "s": 3}))
    rev.reset_env_cache()
    with pytest.raises(rev.RequestEvidenceError, match="disagreeing"):
        rev.from_env()
    monkeypatch.setenv(rev.ENV_SEED_BASE, "x")
    rev.reset_env_cache()
    with pytest.raises(rev.RequestEvidenceError, match="non-negative integer"):
        rev.from_env()
    with pytest.raises(rev.RequestEvidenceError, match="seed base"):
        rev.RequestEvidenceClient(tmp_path / "no_s")
    with pytest.raises(rev.RequestEvidenceError, match="seed base"):
        rev.RequestEvidenceClient(tmp_path / "bool_s", seed_base=True)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the audit reads what this client writes: task names agree, extra.s partitions the Arditi engine      #
# --------------------------------------------------------------------------- #

def _ws3g_modules():
    """`audit_common` + `request_evidence_audit` (read-only): from this tree when present, else the
    sibling lane worktree on the grading host; loaded by file so their sys.path insertion never shadows this tree."""
    import importlib.util
    import sys as _sys

    repo = Path(__file__).resolve().parents[1]
    candidates = [repo / "scripts" / "sdfbase" / "seedrep", repo.parent / "seedrep-ws3g" / "scripts" / "sdfbase" / "seedrep"]
    here = next((d for d in candidates if (d / "request_evidence_audit.py").is_file() and (d / "audit_common.py").is_file()), None)
    if here is None:
        pytest.skip("WS3-G's request_evidence_audit.py is not on this tree (nor in the sibling worktree); the merged reseed runs it")
    before = list(_sys.path)
    # the two names are registered ONLY for the duration of the load (`request_evidence_audit` binds `import audit_common` to the copy
    # loaded here), then the PRIOR module objects are put back: replacing `sys.modules["audit_common"]` for good handed every later
    # importer a second `audit_common` — a second `AuditRefusal` class — while tools imported earlier (`determinism_check`) kept the
    # first, so `pytest.raises(ac.AuditRefusal)` in a later test module could not catch the refusal the tool raised
    # (test_proxy_metrics_passthrough failed only when this file ran before it)
    prior = {name: _sys.modules.get(name) for name in ("audit_common", "request_evidence_audit")}
    mods = {}
    try:
        for name in ("audit_common", "request_evidence_audit"):
            spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            _sys.modules[name] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            mods[name] = mod
    finally:
        for name, m in prior.items():
            if m is None:
                _sys.modules.pop(name, None)
            else:
                _sys.modules[name] = m
        _sys.path[:] = before   # drop whatever their modules inserted (their worktree's src must not shadow ours)
    return mods["audit_common"], mods["request_evidence_audit"]


def test_ws3g_audit_agrees_on_every_task_name_and_refuses_an_arditi_record_without_extra_s(rec):
    ac, rea = _ws3g_modules()
    # the header side and the audit's filed-pairs side spell every leg the same (their drift check, over their vocabulary)
    assert ac.check_client_task_names(rev) == []
    assert set(ac.LEGS) == set(rev.TABLE_LEGS)
    # every record this client writes carries extra.s = the run's seed base, as their reader reads it
    leg = _legs()["static"]
    records, rows = _evidence_for(rec, leg)
    assert all(rea.record_extra_s(r) == S for r in records)
    # the constructive half: a record WITHOUT extra.s is exactly what audit_root refuses on an Arditi root — and this
    # client cannot produce one (a client needs s to exist; every record carries extra)
    stripped = [{**r, "extra": {k: v for k, v in r["extra"].items() if k != "s"}} for r in records]
    unstamped = [r for r in stripped if r.get("kind") == C.REQUEST_EVIDENCE_KIND_TASK and rea.record_extra_s(r) is None]
    assert len(unstamped) == len(records) and rea.record_extra_s({"extra": {"s": True}}) is None
    assert rea.record_extra_s({"extra": None}) is None and rea.record_extra_s({}) is None
    # the partition: the Arditi store at seed S admits these records; the SAME task at another seed admits none,
    # and the reconciliation over that partition refuses coverage (a record without its base attributes to no store)
    task = leg["task"]
    known = [rev.store_task_name(l) for l in rev.TABLE_LEGS if l != rev.LEG_PETRI]
    items = sorted({i for _t, i in leg["filed"]})
    mine, mine_rows = rea.store_partition(records, rows, tasks=[task], target_vmodel=VMODEL, known_tasks=known, filed_items=items, extra_s=S)
    assert len(mine) == len(records) and len(mine_rows) == len(rows)
    assert C.request_evidence_reconciliation(mine, mine_rows, filed=leg["filed"], expected_enable_thinking=False,
                                             expected_sampling=VENDOR, expected_seed=leg["seed"], target_vmodel=VMODEL) is None
    other, other_rows = rea.store_partition(records, rows, tasks=[task], target_vmodel=VMODEL, known_tasks=known, filed_items=items, extra_s=S + 1)
    assert other == [] and other_rows == []
    none_stamped, _ = rea.store_partition(stripped, rows, tasks=[task], target_vmodel=VMODEL, known_tasks=known, filed_items=items, extra_s=S)
    assert none_stamped == []
    v = C.request_evidence_reconciliation(other, rows, filed=leg["filed"], expected_enable_thinking=False,
                                          expected_sampling=VENDOR, expected_seed=leg["seed"], target_vmodel=VMODEL)
    assert v is not None
