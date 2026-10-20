"""The misalign runner's per-response decode seed and its vendor row by EXACT registry ref.

  * the wire body under ``--sampler vendor --seed-policy per_response --seed-base s`` is exactly
    ``{model, messages, max_tokens, temperature 0.7, top_p 0.8, top_k 20, min_p 0.0,
    presence_penalty 1.5, repetition_penalty 1.0, seed}`` with the vendor row's VALUES and
    ``seed = misalign_seed(s, behavior, item_id, sample_idx)``; seeds differ across s, items and
    samples and reproduce;
  * the vendor row resolves through ``PolicyFile.by_ref("qwen35_4b_it|agentic|off|vendor@v1")``
    with the not-superseded assertion — a fixture export whose row is superseded refuses, one with
    no such ref refuses, ``resolve_vendor_policy``'s source names neither ``find(`` nor
    ``resolve(``;
  * meta.json identity gains ``seed_policy`` / ``seed_base``: a per_response run resumes onto its
    own tree with no new requests, refuses a ``none`` tree and a LEGACY tree (written before the
    keys existed), and a ``none`` run refuses a per_response tree; the legacy tree resumes under
    ``none``;
  * every seeded row carries ``seed, seed_rule, seed_base, sampling, decode_policy_id,
    completion_id``; the default policy's rows and payloads carry none of them (byte for byte the
    runner as it was);
  * the flag pair's refusals; the CLI reaches ``run_family``; the request-evidence hook point is
    inert by default (a two-argument transport is called with two arguments) and, when an object is
    wired, receives the body AS SENT and puts ``request_id`` on the row.

Offline; a fake transport."""
from __future__ import annotations

import functools
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sisteer.capeval.misalign import runner as R
from sisteer.registry import decode_policy as dp
from _seedrep_testlib import _load  # noqa: E402
from seedrep_history_fixtures import pinned_commit_or_skip  # noqa: E402
from test_misalign_runner import CELLS, ITEMS, QWEN_IDENTITY, QWEN_VENDOR_ROW, REV, FakeHttp, _rows  # noqa: F401

REPO = Path(__file__).resolve().parents[1]
VENDOR_REF = "qwen35_4b_it|agentic|off|vendor@v1"
SEEDED_EXTRA = ("seed", "seed_rule", "seed_base", "sampling", "decode_policy_id", "completion_id")
#: what ONLY a per_response run writes to meta.json; a `none` run's file carries none of them
META_SEED_KEYS = ("seed_policy", "seed_base", "decode_policy_id", "seed_rule", "seed_rule_text")
#: the runner before the seed-policy hunks — the tip as merged into this branch, the last runner in its lineage without them
RUNNER_BEFORE_WS1C = "3a668b0dbccef4bbd2ceb4a1bf7a27239b07cbe5"
FROZEN_CLOCK = 1755000000.0


def _run(tmp_path, http, *, out="out", cells=CELLS, n=2, mod=R, **kw):
    kw.setdefault("items", ITEMS)
    kw.setdefault("suite_revision", REV)
    kw.setdefault("max_tokens", 3000)
    return mod.run_family("http://127.0.0.1:1/v1", cells, "agentic", n, tmp_path / out,
                          concurrency=4, http=http, **kw)



def runner_before_ws1c(tmp_path):
    """The runner at :data:`RUNNER_BEFORE_WS1C` out of the object store, loaded by path (absolute imports only) over
    this tree's ``sisteer`` — the oracle for what the default policy must still write."""
    pinned_commit_or_skip(RUNNER_BEFORE_WS1C, "seedrep-ws2e", "the misalign runner before WS1-C", repo=REPO, anchor=Path(__file__))
    show = subprocess.run(["git", "-C", str(REPO), "show", f"{RUNNER_BEFORE_WS1C}:src/sisteer/capeval/misalign/runner.py"],
                          capture_output=True)
    if show.returncode != 0:
        pytest.fail(f"git show {RUNNER_BEFORE_WS1C[:8]} failed — fetch origin seedrep-ws2e first: {show.stderr.decode()[-300:]}")
    src = tmp_path / "runner_before_ws1c.py"
    src.write_bytes(show.stdout)
    return _load(f"misalign_runner_{RUNNER_BEFORE_WS1C[:8]}", str(src))


def _metas(root: Path) -> dict:
    return {cid: (root / cid / "meta.json").read_bytes() for cid in ("none", "prompted__safety")}


def _seeded(tmp_path, http, s, **kw):
    return _run(tmp_path, http, sampler="vendor", seed_policy="per_response", seed_base=s, **kw)


# --------------------------------------------------------------------------- the rule
def test_misalign_seed_is_the_stated_rule_and_differs_across_base_item_and_sample():
    import hashlib
    for s in (1, 2, 3):
        for it in ITEMS:
            for k in (0, 1, 9):
                key = f"{s}|misalign|{it['behavior']}|{it['item_id']}|{k}"
                want = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
                assert R.misalign_seed(s, it["behavior"], it["item_id"], k) == want
                assert 0 <= want < 2 ** 32
    a = R.misalign_seed(1, "leaking", "leaking_explicit-none", 0)
    assert a == R.misalign_seed(1, "leaking", "leaking_explicit-none", 0)          # reproduces
    assert a != R.misalign_seed(2, "leaking", "leaking_explicit-none", 0)          # another seed view
    assert a != R.misalign_seed(1, "leaking", "leaking_explicit-none", 1)          # another sample
    assert a != R.misalign_seed(1, "murder", "leaking_explicit-none", 0)           # another behaviour
    assert a != R.misalign_seed(1, "leaking", "murder_explicit-none", 0)           # another item
    # the rule's recorded text IS the rule
    assert R.SEED_RULE_TEXT == 'int.from_bytes(sha256(f"{seed_base}|misalign|{behavior}|{item_id}|{sample_idx}").digest()[:4], "big")'
    assert R.SEED_RULE_PER_RESPONSE == "misalign_per_response_v1"
    # em8q items carry no behaviour: the family stands in, never the string None
    assert R.seed_behavior("em8q", R.em8q_items()[0]) == "em8q"
    assert R.seed_behavior("agentic", ITEMS[1]) == "murder"


def test_the_flag_pair_refuses_a_base_without_the_policy_and_the_policy_without_a_base(tmp_path):
    R.check_seed_policy("none", None)
    R.check_seed_policy("per_response", 3)
    with pytest.raises(R.MisalignRunnerError, match="needs --seed-base"):
        R.check_seed_policy("per_response", None)
    with pytest.raises(R.MisalignRunnerError, match="needs --seed-base"):
        R.check_seed_policy("per_response", True)
    with pytest.raises(R.MisalignRunnerError, match="rides --seed-policy per_response"):
        R.check_seed_policy("none", 1)
    with pytest.raises(R.MisalignRunnerError, match="unknown seed policy"):
        R.check_seed_policy("per_turn", 1)
    http = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    with pytest.raises(R.MisalignRunnerError, match="needs --seed-base"):
        _run(tmp_path, http, seed_policy="per_response")
    with pytest.raises(R.MisalignRunnerError, match="rides --seed-policy"):
        _run(tmp_path, http, seed_base=1)
    assert http.posts == []                                          # refused before any request
    with pytest.raises(R.MisalignRunnerError, match="non-negative int"):
        R.build_payload("petri__base", R.build_messages("agentic", ITEMS[0], None), max_tokens=8, seed=-1)


# --------------------------------------------------------------------------- the wire body
def test_exact_wire_body_vendor_values_and_a_seed_per_scenario_from_s(tmp_path):
    for s in (1, 2, 3):
        http = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
        _seeded(tmp_path, http, s, out=f"s{s}")
        assert len(http.posts) == 2 * len(ITEMS) * 2
        for _url, p in http.posts:
            assert list(p) == ["model", "messages", "max_tokens", "temperature", "top_p", "top_k", "min_p",
                               "presence_penalty", "repetition_penalty", "seed"]
            assert {k: p[k] for k in QWEN_VENDOR_ROW} == QWEN_VENDOR_ROW == \
                {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                 "repetition_penalty": 1.0}
            assert p["max_tokens"] == 3000 and not set(p) & set(R.NEVER_SEND)
        # the seed per (behavior, item, sample_idx) is the rule under THIS s, read back off the rows
        for cell in CELLS:
            rows = _rows(tmp_path / f"s{s}" / cell["cell_id"] / "responses.jsonl")
            assert len(rows) == 4
            for r in rows:
                assert r["seed"] == R.misalign_seed(s, r["behavior"], r["item_id"], r["sample_idx"])
                assert r["seed_rule"] == "misalign_per_response_v1" and r["seed_base"] == s
                assert r["sampling"] == QWEN_VENDOR_ROW and r["decode_policy_id"] == VENDOR_REF
                assert r["completion_id"] == "cmpl-1"                 # the fake transport's response id
            sent = {(p["messages"][-1]["content"], p["seed"]) for _u, p in http.posts if p["model"] == cell["virtual_model"]}
            assert {(next(i["user_prompt"] for i in ITEMS if i["item_id"] == r["item_id"]), r["seed"]) for r in rows} == sent
            meta = json.loads((tmp_path / f"s{s}" / cell["cell_id"] / "meta.json").read_text())
            assert meta["seed_policy"] == "per_response" and meta["seed_base"] == s
            assert meta["seed_rule"] == "misalign_per_response_v1" and meta["seed_rule_text"] == R.SEED_RULE_TEXT
            assert meta["decode_policy_id"] == VENDOR_REF and meta["sampler"] == "vendor"
    # the same (item, sample) under s=1 and s=2 differ in the seed alone
    r1 = {(r["item_id"], r["sample_idx"]): r for r in _rows(tmp_path / "s1" / "none" / "responses.jsonl")}
    r2 = {(r["item_id"], r["sample_idx"]): r for r in _rows(tmp_path / "s2" / "none" / "responses.jsonl")}
    for k in r1:
        assert r1[k]["seed"] != r2[k]["seed"] and r1[k]["seed_base"] == 1 and r2[k]["seed_base"] == 2
        assert {kk: v for kk, v in r1[k].items() if kk not in ("seed", "seed_base")} == \
            {kk: v for kk, v in r2[k].items() if kk not in ("seed", "seed_base")}


def test_the_default_policy_is_the_runner_as_it_was_no_seed_no_new_row_fields(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http, sampler="vendor")
    for _u, p in http.posts:
        assert "seed" not in p and list(p) == ["model", "messages", "max_tokens", "temperature", "top_p", "top_k",
                                                "min_p", "presence_penalty", "repetition_penalty"]
    for r in _rows(tmp_path / "out" / "none" / "responses.jsonl"):
        assert not set(SEEDED_EXTRA) & set(r) and "request_id" not in r
    meta = json.loads((tmp_path / "out" / "none" / "meta.json").read_text())
    assert not set(META_SEED_KEYS) & set(meta) and meta["sampler"] == "vendor"   # no seed keys, no seed provenance under `none`
    http2 = FakeHttp(["petri__base", "petri__prompted__safety"])
    _run(tmp_path, http2, out="t1")
    for _u, p in http2.posts:
        assert set(p) == {"model", "messages", "max_tokens", "temperature"}
    meta = json.loads((tmp_path / "t1" / "none" / "meta.json").read_text())
    assert not set(META_SEED_KEYS) & set(meta) and meta["sampler"] == "t1"


# --------------------------------------------------------------------------- resume identity
def test_resume_semantics_own_tree_resumes_none_and_legacy_trees_refuse(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _seeded(tmp_path, http, 2)
    n0 = len(http.posts)
    _seeded(tmp_path, http, 2)                                       # complete: no new request
    assert len(http.posts) == n0
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch on 'seed_base'"):
        _seeded(tmp_path, http, 3)                                   # another seed view never lands here
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch on 'seed_policy'"):
        _run(tmp_path, http, sampler="vendor")                        # nor the unseeded instrument
    # a `none` tree refuses a per_response resume
    http_n = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http_n, out="plain", sampler="vendor")
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch on 'seed_policy'"):
        _seeded(tmp_path, http_n, 2, out="plain")
    assert ("seed_policy", "seed_base") == R._META_RESUME_KEYS[-2:]                    # history-independent: outside the skipped portion
    assert R._META_LEGACY_DEFAULTS["seed_policy"]({}) == "none" and R._META_LEGACY_DEFAULTS["seed_base"]({}) is None
    # a LEGACY tree — meta.json written by the REAL runner before the seed policy existed — reads as `none`, resumes
    # under it only, and is not given the keys it never had (the object store: skip/fail decided in runner_before_ws1c)
    legacy = tmp_path / "legacy"
    http_l = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http_l, out="legacy", mod=runner_before_ws1c(tmp_path), sampler="vendor")
    n1 = len(http_l.posts)
    _run(tmp_path, http_l, out="legacy", sampler="vendor")
    assert len(http_l.posts) == n1
    assert not set(META_SEED_KEYS) & set(json.loads((legacy / "none" / "meta.json").read_text()))
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch on 'seed_policy'"):
        _seeded(tmp_path, http_l, 1, out="legacy")


# --------------------------------------------------------------------------- the vendor row by exact ref
def _fixture_pf(*, superseded=None, drop=False, extra_rows=()):
    pf_real = dp.load()
    row = next(p for p in pf_real.policies if p["policy_id"] == VENDOR_REF)
    rows = [] if drop else [{**row, "superseded_by_policy_id": superseded}]
    rows += list(extra_rows)
    data = {"export_version": 1, "policies": rows, "sampling_mode_policy": [],
            "models": {"qwen35_4b_it": {"hf_repo": "Qwen/Qwen3.5-4B"}},
            "model_spellings": {"Qwen/Qwen3.5-4B": "qwen35_4b_it"}, "evals": {}, "eval_spellings": {}}
    return dp.PolicyFile(data, "<fixture>"), row


def test_vendor_row_is_by_exact_ref_not_superseded_and_never_a_listing(tmp_path):
    pol = R.resolve_vendor_policy(QWEN_IDENTITY)
    assert pol.policy_id == VENDOR_REF == pol.policy_ref + "@v1" and pol.superseded_by_policy_id is None
    assert R.vendor_policy_ref("qwen35_4b_it", "off") == VENDOR_REF
    assert pol.request_params() == QWEN_VENDOR_ROW and R.vendor_sampling_for(QWEN_IDENTITY) == QWEN_VENDOR_ROW
    # THE default for its key since the rule of 2026-09-21 (decode-vendor-default-all-evals-2026-09-21; before: 0); by_ref does not
    # need the flag either way -- the row is resolved by exact ref
    assert pol.is_default == 1
    src = inspect.getsource(R.resolve_vendor_policy)
    assert "by_ref(" in src and ".find(" not in src and ".resolve(" not in src
    # the fixture export: the same row, superseded -> refused; absent -> refused; a v2 beside it -> v1 still exact
    pf, row = _fixture_pf(superseded="qwen35_4b_it|agentic|off|vendor@v2")
    with pytest.raises(R.MisalignRunnerError, match="superseded by 'qwen35_4b_it\\|agentic\\|off\\|vendor@v2'"):
        R.resolve_vendor_policy(QWEN_IDENTITY, policy_file=pf)
    pf, _ = _fixture_pf(drop=True)
    with pytest.raises(R.MisalignRunnerError, match="no frozen vendor sampling row .* names 0 rows"):
        R.vendor_sampling_for(QWEN_IDENTITY, policy_file=pf)
    v2 = {**row, "policy_id": "qwen35_4b_it|agentic|off|vendor@v2", "policy_version": 2, "temperature": 0.9, "is_default": 1}
    pf, _ = _fixture_pf(extra_rows=[v2])
    got = R.resolve_vendor_policy(QWEN_IDENTITY, policy_file=pf)
    assert got.policy_id == VENDOR_REF and got.request_params()["temperature"] == 0.7   # @v1 exactly, not the default v2
    # the wrong-model / wrong-mode / bare-endpoint refusals stand
    with pytest.raises(R.MisalignRunnerError, match="no frozen vendor sampling row"):
        R.vendor_sampling_for({"base_model": "Qwen/Qwen3.5-4B", "thinking_mode": "on"})
    with pytest.raises(R.MisalignRunnerError, match="identity block"):
        R.resolve_vendor_policy(None)
    # a run against a fixture whose row is superseded refuses BEFORE any request
    http = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    pf, _ = _fixture_pf(superseded="x@v9")
    import unittest.mock as um
    with um.patch.object(R, "_DECODE_POLICY", pf):
        with pytest.raises(R.MisalignRunnerError, match="superseded"):
            _seeded(tmp_path, http, 1, out="never")
    assert http.posts == []


# --------------------------------------------------------------------------- the CLI and the evidence hook
def test_cli_flags_reach_run_family(tmp_path, monkeypatch):
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps(CELLS))
    seen = {}

    def fake_run_family(*a, **kw):
        seen.update(kw)
        return {c["cell_id"]: {"requested": 0, "skipped": 0, "generated": 0, "excluded_items": 0} for c in CELLS}
    monkeypatch.setattr(R, "run_family", fake_run_family)
    rc = R.main(["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells), "--family", "agentic", "--n", "10",
                 "--out", str(tmp_path / "o"), "--behaviors", "leaking", "murder", "--max-tokens", "3000",
                 "--sampler", "vendor", "--seed-policy", "per_response", "--seed-base", "2"])
    assert rc == 0 and seen["seed_policy"] == "per_response" and seen["seed_base"] == 2 and seen["sampler"] == "vendor"
    seen.clear()
    R.main(["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells), "--family", "agentic", "--n", "1",
            "--out", str(tmp_path / "o2")])
    assert seen["seed_policy"] == "none" and seen["seed_base"] is None            # the defaults
    with pytest.raises(SystemExit):
        R.main(["--proxy-url", "x", "--cells", str(cells), "--family", "agentic", "--n", "1", "--out", "o",
                "--seed-policy", "per_turn"])


class EvidenceHttp(FakeHttp):
    """A transport that ACCEPTS headers (the wired client's) and records them."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.headers = []

    def post_json(self, url, payload, headers=None):
        self.headers.append(headers)
        return super().post_json(url, payload)


class FakeEvidence:
    def __init__(self):
        self.calls = []

    def headers_for(self, *, family, behavior, item_id, sample_idx, payload):
        return {"X-Tinycap-Task": f"{family}/{behavior}", "X-Tinycap-Sample": f"{item_id}/{sample_idx}",
                "X-Tinycap-Request-Id": f"req-{item_id}-{sample_idx}"}

    def record(self, *, payload, headers, response):
        self.calls.append((json.loads(json.dumps(payload)), dict(headers),
                           response.get("id") if response is not None else None))     # None: a refused attempt
        return headers["X-Tinycap-Request-Id"]


def test_request_evidence_hook_is_inert_by_default_and_binds_headers_and_request_id_when_wired(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _seeded(tmp_path, http, 1)                                       # a two-argument transport: no headers passed
    assert http.posts and all("request_id" not in r for r in _rows(tmp_path / "out" / "none" / "responses.jsonl"))
    http2 = EvidenceHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    ev = FakeEvidence()
    _seeded(tmp_path, http2, 1, out="ev", request_evidence=ev)
    assert len(ev.calls) == len(http2.posts) == 2 * len(ITEMS) * 2
    for (payload, headers, cid), (_u, sent), h in zip(ev.calls, http2.posts, http2.headers):
        assert payload == sent and headers == h and cid == "cmpl-1"  # the body AS SENT, the headers as posted
        assert headers["X-Tinycap-Sample"].split("/")[0] in {i["item_id"] for i in ITEMS}
    rows = _rows(tmp_path / "ev" / "none" / "responses.jsonl")
    assert {r["request_id"] for r in rows} == {f"req-{r['item_id']}-{r['sample_idx']}" for r in rows}
    assert all(r["completion_id"] == "cmpl-1" for r in rows)
    # the default path calls post_json(url, payload) — two arguments — so today's transports keep working
    assert http2.headers and all(h is not None for h in http2.headers)
    http3 = EvidenceHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http3, out="plain")
    assert http3.headers == [None] * len(http3.posts)


# --------------------------------------------------------------------------- evidence at the POST boundary
class MintingEvidence(FakeEvidence):
    """The hook as the contract reads: a FRESH request id per ``headers_for`` call (a retry is a new attempt)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def headers_for(self, **kw):
        h = super().headers_for(**kw)
        self.n += 1
        h["X-Tinycap-Request-Id"] += f"-{self.n:03d}"
        return h


FIT_WINDOW, FIT_INPUT, WIDE_CAP, FITTED_CAP = 24576, 8192, 16384, 14336


class ContextFitHttp(EvidenceHttp):
    """vLLM's context-length 400 on the FIRST attempt of every (cell, item, sample) at the wide cap, then success —
    ``fit_max_tokens`` reads ``min(24576 - 8192 - 64, 16384 - 2048) = 14,336`` off the refusal, so the retry sends
    the same body with ``max_tokens`` 14,336."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.refused = set()

    def post_json(self, url, payload, headers=None):
        resp = super().post_json(url, payload, headers=headers)                       # recorded like every post
        key = (payload["model"], payload["messages"][-1]["content"], payload.get("seed"))
        if payload["max_tokens"] > FITTED_CAP and key not in self.refused:
            self.refused.add(key)
            raise R.HttpError("refused", status=400, body=(
                f"This model's maximum context length is {FIT_WINDOW} tokens. However, you requested "
                f"{FIT_INPUT + payload['max_tokens']} tokens ({FIT_INPUT} in the messages, {payload['max_tokens']} in the "
                f"completion). Please reduce the length of the messages or completion. (at least {FIT_INPUT} input tokens)"))
        return resp


def _fit_run(tmp_path, http, ev, out="fit"):
    return _seeded(tmp_path, http, 1, out=out, n=1, max_tokens=WIDE_CAP, request_evidence=ev)


def test_context_fit_retry_records_each_attempts_exact_body_under_its_own_id_and_the_row_binds_the_successful_one(tmp_path):
    http = ContextFitHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    ev = MintingEvidence()
    _fit_run(tmp_path, http, ev)
    n_items = len(CELLS) * len(ITEMS)
    assert len(http.posts) == len(ev.calls) == len(http.headers) == 2 * n_items      # two attempts per (cell, item)
    # every record IS the body the transport was handed for THAT attempt, with the headers it went out under
    for (payload, headers, _cid), (_url, sent), h in zip(ev.calls, http.posts, http.headers):
        assert payload == sent and headers == h
    by_key = {}
    for payload, headers, cid in ev.calls:
        by_key.setdefault((payload["model"], payload["seed"]), []).append((payload, headers, cid))
    assert len(by_key) == n_items
    for (p1, h1, c1), (p2, h2, c2) in by_key.values():
        assert p1["max_tokens"] == WIDE_CAP and c1 is None                            # the refused attempt: as sent, no completion
        assert p2["max_tokens"] == FITTED_CAP and c2 == "cmpl-1"                      # the fitted attempt: the REDUCED body, bound
        assert h1["X-Tinycap-Request-Id"] != h2["X-Tinycap-Request-Id"]                # a retry is a new attempt: its own id
        assert h1["X-Tinycap-Sample"] == h2["X-Tinycap-Sample"] and h1["X-Tinycap-Task"] == h2["X-Tinycap-Task"]
        assert {k: v for k, v in p1.items() if k != "max_tokens"} == {k: v for k, v in p2.items() if k != "max_tokens"}
    rows = [r for p in sorted((tmp_path / "fit").glob("*/responses.jsonl")) for r in _rows(p)]
    assert len(rows) == n_items
    for r in rows:
        _p2, h2, c2 = by_key[(r["virtual_model"], r["seed"])][-1]
        assert r["request_id"] == h2["X-Tinycap-Request-Id"] and r["completion_id"] == c2
        assert r["max_tokens_effective"] == FITTED_CAP and r["seed_base"] == 1
        assert r["sampling"] == {k: QWEN_VENDOR_ROW[k] for k in r["sampling"]} and set(r["sampling"]) == set(R.SAMPLING_KEYS)


def test_post_with_retries_stamps_the_successful_attempts_id_and_nothing_without_a_hook():
    http = ContextFitHttp(["petri__base"], identity=QWEN_IDENTITY)
    payload = R.build_payload("petri__base", [{"role": "user", "content": "hi"}], max_tokens=WIDE_CAP,
                              sampling=QWEN_VENDOR_ROW, seed=7)
    resp = R._post_with_retries(http, "http://127.0.0.1:1/v1/chat/completions", payload, max_retries=0, what="w")
    assert resp["_max_tokens_effective"] == FITTED_CAP and "_request_id" not in resp and http.headers == [None, None]
    assert [p["max_tokens"] for _u, p in http.posts] == [WIDE_CAP, FITTED_CAP]
    ev = MintingEvidence()
    http2 = ContextFitHttp(["petri__base"], identity=QWEN_IDENTITY)
    resp2 = R._post_with_retries(http2, "http://127.0.0.1:1/v1/chat/completions", payload, max_retries=0, what="w",
                                 mint_headers=functools.partial(ev.headers_for, family="agentic", behavior="leaking",
                                                                item_id="it", sample_idx=0),
                                 record=ev.record)
    assert resp2["_request_id"] == ev.calls[-1][1]["X-Tinycap-Request-Id"] == "req-it-0-002"
    assert ev.calls[-1][0] == http2.posts[-1][1]                                        # the recorded body equals the successful send


def test_a_hook_that_returns_the_same_request_id_for_two_attempts_is_refused(tmp_path):
    # KNOWN-BAD: FakeEvidence mints one id per (item, sample) — fine for one attempt, refused on the retry
    http = ContextFitHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    with pytest.raises(R.MisalignRunnerError, match=r"returned request id 'req-.*' for two attempts"):
        _fit_run(tmp_path, http, FakeEvidence())


# --------------------------------------------------------------------------- the default policy's meta.json, byte for byte
@pytest.mark.parametrize("kw", [{}, {"sampler": "vendor"}], ids=["t1_default", "vendor_none"])
def test_the_historical_invocation_writes_meta_json_byte_for_byte_as_the_runner_before_ws1c(tmp_path, monkeypatch, kw):
    """Under the historical invocation (``seed_policy none``, no base, no evidence hook) the wire AND
    ``meta.json`` are the runner's before this change — the REAL runner at :data:`RUNNER_BEFORE_WS1C`, same inputs, the clock
    frozen (``started`` / ``finished`` are meta's only clock reads). KNOWN-BAD for the check: the lane's ``per_response``
    run does carry the five keys, so the equality is not vacuous."""
    monkeypatch.setattr(time, "time", lambda: FROZEN_CLOCK)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    before = runner_before_ws1c(tmp_path)
    http_b = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http_b, out="before", mod=before, **kw)
    http_a = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http_a, out="after", **kw)
    assert [p for _u, p in http_a.posts] == [p for _u, p in http_b.posts]           # the wire, unchanged
    assert _metas(tmp_path / "after") == _metas(tmp_path / "before")                 # meta.json, byte for byte
    assert not set(META_SEED_KEYS) & set(json.loads(_metas(tmp_path / "after")["none"]))
    http_s = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _seeded(tmp_path, http_s, 2, out="seeded")
    seeded = json.loads(_metas(tmp_path / "seeded")["none"])
    assert set(META_SEED_KEYS) <= set(seeded) and seeded["seed_policy"] == "per_response" and seeded["seed_base"] == 2
    assert seeded["decode_policy_id"] == VENDOR_REF and seeded["seed_rule"] == R.SEED_RULE_PER_RESPONSE
    assert _metas(tmp_path / "seeded") != _metas(tmp_path / "before")


def test_a_legacy_tree_resumed_under_the_default_policy_is_rewritten_exactly_as_the_legacy_runner_would(tmp_path, monkeypatch):
    """Resuming a tree the earlier runner wrote, under the default policy, rewrites ``meta.json`` byte for
    byte as that runner's own resume would — the seed keys it never had are not written back."""
    monkeypatch.setattr(time, "time", lambda: FROZEN_CLOCK)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    before = runner_before_ws1c(tmp_path)
    http_o = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http_o, out="oracle", mod=before, sampler="vendor")
    _run(tmp_path, http_o, out="oracle", mod=before, sampler="vendor")                # the legacy runner's own resume
    http_n = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http_n, out="legacy", mod=before, sampler="vendor")
    n = len(http_n.posts)
    _run(tmp_path, http_n, out="legacy", sampler="vendor")                             # THIS runner resumes it
    assert len(http_n.posts) == n                                                       # complete: nothing re-drawn
    assert _metas(tmp_path / "legacy") == _metas(tmp_path / "oracle")
    assert not set(META_SEED_KEYS) & set(json.loads(_metas(tmp_path / "legacy")["none"]))
