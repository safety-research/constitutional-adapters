"""The misalign served-path runner (sisteer.capeval.misalign.runner, WP-B).

Covers: the SI text pins (torch-free resolution == the manifest's recorded
prefix hashes, and == TrainTest's own resolver when torch is available); the
message rules for prompted vs non-prompted cells in both families; the
request payload never carrying a proxy-forbidden or refused key; em8q item
ids; the run itself against a fake HTTP layer (rows + meta, resume skipping,
the /v1/models refusal before any request, the foreign-endpoint check vs
--plain, a 4xx abort, the per-cell behavior exclusion, the resume-identity
mismatch); the CLI; that none of the package imports torch; and the real
httpx transport against an in-process loopback server (bearer, JSON framing,
4xx -> HttpError with status).

CPU-offline; loopback HTTP only."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from sisteer.capeval.manifest import sha256_hex16
from sisteer.capeval.misalign import runner as R
from sisteer.capeval.petri.registry_adapter import HARNESS_TO_TRAINTEST_SI
from sisteer.data.safety_texts import BETLEY_8Q, BETLEY_8Q_SOURCE_IDS

REPO = Path(__file__).parents[1]

#: the prefix hashes every full22 manifest records (prompted_prefix_hashes)
SI_PINS = {
    "safety": "ac3a58f61d53cf24",
    "constitution": "21e06f482a6559b3",
    "irrelevant": "71ff67aa2bc000f1",
}
IDENT = "proxy_vllm+passthrough+interleaved"
CELLS = [
    {"cell_id": "none", "virtual_model": "petri__base", "serving_identity": IDENT,
     "intervention_hash": "0" * 16, "si_id": None},
    {"cell_id": "prompted__safety", "virtual_model": "petri__prompted__safety",
     "serving_identity": IDENT, "intervention_hash": "a" * 16, "si_id": "safety"},
]
ITEMS = [
    {"item_id": "leaking_explicit-none", "behavior": "leaking",
     "system_prompt": "You are Alex, deployed at NovaStar.", "user_prompt": "<email>one</email>"},
    {"item_id": "murder_explicit-none", "behavior": "murder",
     "system_prompt": "You are Alex, deployed at SummitBridge.", "user_prompt": "<email>two</email>"},
]
REV = "agentic.test@000000000000"


class FakeHttp:
    """The two-method transport the runner takes; records every call."""

    def __init__(self, models, *, fail_status=None, identity=None):
        self.models = list(models)
        self.fail_status = fail_status
        self.identity = dict(identity or {})   # extra petri_proxy identity facts (base_model, thinking_mode)
        self.gets, self.posts = [], []
        self._lock = threading.Lock()

    def get_json(self, url):
        self.gets.append(url)
        return {"object": "list",
                "data": [{"id": m, "object": "model", "owned_by": "petri"} for m in self.models],
                "petri_proxy": {"version": 1, "serving_mode": "interleaved", **self.identity}}

    def post_json(self, url, payload):
        with self._lock:
            self.posts.append((url, json.loads(json.dumps(payload))))
        if self.fail_status is not None:
            raise R.HttpError("refused", status=self.fail_status, body="si_prefix_mismatch")
        user = payload["messages"][-1]["content"]
        return {"id": "cmpl-1", "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": f"reply:{user}"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2}}

    def close(self):
        pass


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# --- SI text ----------------------------------------------------------------------


@pytest.mark.parametrize("si", sorted(SI_PINS))
def test_si_text_matches_recorded_prefix_hash(si):
    assert sha256_hex16(R.si_text_for(si).encode("utf-8")) == SI_PINS[si]


@pytest.mark.parametrize("si", sorted(SI_PINS))
def test_si_text_matches_traintest_resolver(si):
    pytest.importorskip("torch")  # the TrainTest resolver imports the methods stack
    from sisteer.capeval.petri.registry_adapter import _traintest_si_text_bytes
    assert R.si_text_for(si).encode("utf-8") == _traintest_si_text_bytes(
        HARNESS_TO_TRAINTEST_SI[si])


def test_si_text_unknown_id_refuses():
    with pytest.raises(R.MisalignRunnerError, match="no TrainTest translation"):
        R.si_text_for("nonsense")


# --- message rules -------------------------------------------------------------


def test_agentic_prompted_system_is_si_then_scenario():
    si = R.si_text_for("safety")
    msgs = R.build_messages("agentic", ITEMS[0], si)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == si + "\n\n" + ITEMS[0]["system_prompt"]
    assert msgs[0]["content"].startswith(si)  # the proxy hash-checks this prefix
    assert msgs[1]["content"] == ITEMS[0]["user_prompt"]


def test_agentic_nonprompted_system_is_scenario_only():
    msgs = R.build_messages("agentic", ITEMS[1], None)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == ITEMS[1]["system_prompt"]
    assert msgs[1]["content"] == ITEMS[1]["user_prompt"]


def test_em8q_prompted_system_is_si_alone():
    si = R.si_text_for("constitution")
    item = R.em8q_items()[0]
    msgs = R.build_messages("em8q", item, si)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == si
    assert msgs[1]["content"] == BETLEY_8Q["betley_q0"]


def test_em8q_nonprompted_has_no_system_message():
    item = R.em8q_items()[3]
    msgs = R.build_messages("em8q", item, None)
    assert msgs == [{"role": "user", "content": BETLEY_8Q["betley_q3"]}]


def test_agentic_item_without_system_prompt_refuses():
    with pytest.raises(R.MisalignRunnerError, match="no system_prompt"):
        R.build_messages("agentic", {"item_id": "x", "user_prompt": "u"}, None)


def test_payload_is_one_sample_and_never_carries_refused_keys():
    msgs = R.build_messages("em8q", R.em8q_items()[0], None)
    payload = R.build_payload("petri__base", msgs, max_tokens=600, temperature=1.0)
    assert set(payload) == {"model", "messages", "max_tokens", "temperature"}
    assert not set(payload) & set(R.NEVER_SEND)
    for k in ("n", "best_of", "stream", "vllm_xargs", "petri_object", "expect_hash",
              "extra_args", "cache_salt"):
        assert k in R.NEVER_SEND and k not in payload
    assert payload["max_tokens"] == 600 and payload["temperature"] == 1.0


# --- items -----------------------------------------------------------------------------


QWEN_IDENTITY = {"base_model": "Qwen/Qwen3.5-4B", "thinking_mode": "off"}
QWEN_VENDOR_ROW = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                   "presence_penalty": 1.5, "repetition_penalty": 1.0}


def test_vendor_sampling_resolves_qwen_thinking_off_and_refuses_the_rest():
    assert R.vendor_sampling_for(QWEN_IDENTITY) == QWEN_VENDOR_ROW
    assert set(QWEN_VENDOR_ROW) <= set(R.SAMPLING_KEYS)
    with pytest.raises(R.MisalignRunnerError, match="no frozen vendor sampling row"):
        R.vendor_sampling_for({"base_model": "Qwen/Qwen3.5-4B", "thinking_mode": "on"})
    # since the rule of 2026-09-21 (decode-vendor-default-all-evals-2026-09-21) every SERVED model has a `<model>|agentic|<thinking>|vendor`
    # row restating its static vendor preset -- gemma-4-E4B sends 1.0 / 0.95 / 64; a model with no row (the E2B reference model) is still refused
    assert R.vendor_sampling_for({"base_model": "google/gemma-4-E4B-it", "thinking_mode": "off"}) == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    with pytest.raises(R.MisalignRunnerError, match="no frozen vendor sampling row"):
        R.vendor_sampling_for({"base_model": "google/gemma-4-E2B-it", "thinking_mode": "off"})
    with pytest.raises(R.MisalignRunnerError, match="identity block"):
        R.vendor_sampling_for(None)
    assert R.resolve_sampling("t1", 1.0, None) == {"temperature": 1.0}
    assert R.resolve_sampling("vendor", 1.0, QWEN_IDENTITY) == QWEN_VENDOR_ROW


def test_payload_with_vendor_sampling_carries_every_key_explicitly():
    msgs = R.build_messages("agentic", ITEMS[0], None)
    p = R.build_payload("petri__base", msgs, max_tokens=16384, sampling=QWEN_VENDOR_ROW)
    assert {k: p[k] for k in QWEN_VENDOR_ROW} == QWEN_VENDOR_ROW
    assert p["max_tokens"] == 16384 and not set(p) & set(R.NEVER_SEND)
    # the t1 payload is byte-for-byte what the campaign sent: temperature alone
    assert set(R.build_payload("petri__base", msgs, max_tokens=8, temperature=1.0)) == {
        "model", "messages", "max_tokens", "temperature"}
    with pytest.raises(R.MisalignRunnerError, match="non-sampling"):
        R.build_payload("petri__base", msgs, max_tokens=8, sampling={"temperature": 1.0, "seed": 3})


def test_vendor_run_records_the_row_and_never_mixes_with_a_t1_tree(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY)
    _run(tmp_path, http, sampler="vendor")
    assert http.posts and all({k: p[k] for k in QWEN_VENDOR_ROW} == QWEN_VENDOR_ROW for _, p in http.posts)
    meta = json.loads((tmp_path / "out" / "none" / "meta.json").read_text())
    assert meta["sampler"] == "vendor" == meta["params"]["sampler"]
    assert meta["sampling"] == meta["params"]["sampling"] == QWEN_VENDOR_ROW
    assert meta["temperature"] == 0.7 == meta["params"]["temperature"]
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch"):
        _run(tmp_path, http)          # the t1 instrument cannot resume onto vendor draws
    # a vendor run against a proxy serving a model with no row refuses BEFORE any request
    http3 = FakeHttp(["petri__base", "petri__prompted__safety"])
    with pytest.raises(R.MisalignRunnerError, match="identity block|no frozen vendor"):
        _run(tmp_path / "x", http3, sampler="vendor")
    assert http3.posts == []
    # a LEGACY tree (meta.json from before the sampler existed) reads as t1 and resumes under t1 only
    legacy = tmp_path / "legacy"
    http2 = FakeHttp(["petri__base", "petri__prompted__safety"])
    R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 1, legacy, concurrency=2, max_tokens=64,
                 http=http2, items=ITEMS, suite_revision=REV)
    for cid in ("none", "prompted__safety"):
        mp = legacy / cid / "meta.json"
        m = json.loads(mp.read_text()); m.pop("sampler"); m.pop("sampling"); mp.write_text(json.dumps(m))
    n0 = len(http2.posts)
    R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 1, legacy, concurrency=2, max_tokens=64,
                 http=http2, items=ITEMS, suite_revision=REV)
    assert len(http2.posts) == n0, "a complete legacy tree resumes under t1 with no requests"
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch"):
        R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 1, legacy, concurrency=2, max_tokens=64,
                     http=FakeHttp(["petri__base", "petri__prompted__safety"], identity=QWEN_IDENTITY),
                     items=ITEMS, suite_revision=REV, sampler="vendor")


def test_em8q_items_use_the_paper_ids():
    items = R.em8q_items()
    assert [i["item_id"] for i in items] == list(BETLEY_8Q_SOURCE_IDS.values())
    assert [i["user_prompt"] for i in items] == list(BETLEY_8Q.values())
    assert all(i["system_prompt"] is None and i["behavior"] is None for i in items)
    _, rev = R.load_items("em8q")
    assert rev == R.EM8Q_SUITE_REVISION == "betley8q.v1"


def test_agentic_items_default_to_leaking_and_murder_or_refuse():
    items, rev = R.load_items("agentic")
    assert rev.startswith("agentic.")
    behaviors = {i["behavior"] for i in items}
    assert "leaking" in behaviors and behaviors <= set(R.AGENTIC_BEHAVIORS)
    assert all(i["item_id"] and i["system_prompt"] and i["user_prompt"] for i in items)
    with pytest.raises(R.MisalignRunnerError, match="no agentic scenarios"):
        R.agentic_items(["blackmail_not_a_behavior"], rows=[])
    with pytest.raises(R.MisalignRunnerError, match="agentic family only"):
        R.load_items("em8q", behaviors=["leaking"])


def test_parse_exclude():
    assert R.parse_exclude(["nemotron3nano4b:leaking", "a:b", "a:c"]) == {
        "nemotron3nano4b": {"leaking"}, "a": {"b", "c"}}
    assert R.parse_exclude(None) == {}
    with pytest.raises(R.MisalignRunnerError):
        R.parse_exclude(["no-colon"])


def test_v1_base():
    assert R.v1_base("http://127.0.0.1:8814") == "http://127.0.0.1:8814/v1"
    assert R.v1_base("http://127.0.0.1:8814/v1/") == "http://127.0.0.1:8814/v1"


# --- the run against a fake transport ------------------------------------------------


def _run(tmp_path, http, cells=CELLS, n=2, **kw):
    kw.setdefault("items", ITEMS)
    kw.setdefault("suite_revision", REV)
    return R.run_family("http://127.0.0.1:1/v1", cells, "agentic", n, tmp_path / "out",
                        concurrency=4, max_tokens=64, http=http, **kw)


def test_run_writes_rows_and_meta_per_cell(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"])
    summary = _run(tmp_path, http)
    assert http.gets == ["http://127.0.0.1:1/v1/models"]
    assert len(http.posts) == 2 * len(ITEMS) * 2
    assert all(u == "http://127.0.0.1:1/v1/chat/completions" for u, _ in http.posts)
    si = R.si_text_for("safety")
    for cell in CELLS:
        cid = cell["cell_id"]
        assert summary[cid] == {"requested": 4, "skipped": 0, "generated": 4,
                                "excluded_items": 0}
        rows = _rows(tmp_path / "out" / cid / "responses.jsonl")
        assert len(rows) == 4
        assert {(r["item_id"], r["sample_idx"]) for r in rows} == {
            (i["item_id"], s) for i in ITEMS for s in range(2)}
        for r in rows:
            assert set(R.RESPONSE_FIELDS) <= set(r)
            assert r["family"] == "agentic" and r["cell_id"] == cid
            assert r["virtual_model"] == cell["virtual_model"]
            assert r["behavior"] in ("leaking", "murder")
            assert r["response"].startswith("reply:<email>")
            assert r["finish_reason"] == "stop" and r["usage"]["completion_tokens"] == 2
            assert r["model_reported"] == cell["virtual_model"]
            assert len(r["prompt_sha"]) == 64
        meta = json.loads((tmp_path / "out" / cid / "meta.json").read_text())
        assert meta["family"] == "agentic" and meta["suite_revision"] == REV
        assert meta["cell"] == cell and meta["cell_id"] == cid
        assert meta["n_samples"] == 2 and meta["params"]["max_tokens"] == 64
        assert meta["params"]["temperature"] == 1.0
        assert meta["finished"] is not None and meta["present"] == 4
        assert meta["served_models"] == ["petri__base", "petri__prompted__safety"]
        assert meta["endpoint_identity"]["version"] == 1
        assert meta["behaviors"] == ["leaking", "murder"]
    # the prompted cell's requests carry the SI as the first system message,
    # the base cell's carry the scenario system prompt alone
    by_model = {}
    for _, p in http.posts:
        by_model.setdefault(p["model"], []).append(p)
    for p in by_model["petri__prompted__safety"]:
        assert p["messages"][0]["role"] == "system"
        assert p["messages"][0]["content"].startswith(si)
        assert p["messages"][0]["content"].endswith(
            next(i["system_prompt"] for i in ITEMS if i["user_prompt"] == p["messages"][1]["content"]))
    for p in by_model["petri__base"]:
        assert p["messages"][0]["content"] in {i["system_prompt"] for i in ITEMS}
        assert not set(p) & set(R.NEVER_SEND)


def test_composite_organism_cell_writes_to_a_colon_free_directory(tmp_path):
    # cluster A's shared storage refuses ':' in a path (EINVAL); the composite id stays the
    # analysis contract in meta.json and every row, only the directory carries '@'
    org_cell = {"cell_id": "em-bad-medical-advice::none", "virtual_model": "petri__base@em-bad-medical-advice",
                "serving_identity": IDENT, "intervention_hash": "b" * 16, "si_id": None}
    http = FakeHttp(["petri__base", "petri__base@em-bad-medical-advice"])
    summary = _run(tmp_path, http, cells=[CELLS[0], org_cell])
    assert R.cell_dirname("em-bad-medical-advice::none") == "em-bad-medical-advice@none"
    assert R.cell_dirname("none") == "none"
    assert summary["em-bad-medical-advice::none"]["generated"] == 4
    d = tmp_path / "out" / "em-bad-medical-advice@none"
    assert d.is_dir() and not (tmp_path / "out" / "em-bad-medical-advice::none").exists()
    assert ":" not in d.name
    meta = json.loads((d / "meta.json").read_text())
    assert meta["cell_id"] == "em-bad-medical-advice::none" and meta["cell"] == org_cell
    assert {r["cell_id"] for r in _rows(d / "responses.jsonl")} == {"em-bad-medical-advice::none"}
    with pytest.raises(ValueError):
        R.cell_dirname("already@ambiguous::none")


def test_read_done_splits_on_newline_only_a_line_separator_inside_a_response_is_not_a_corrupt_row(tmp_path):
    # job 487681/487957: a Llama organism response carried U+2028 LINE SEPARATOR; json.dumps(ensure_ascii=False)
    # wrote it raw, str.splitlines() cut the row there, and an intact 400-row file read as "corrupt row at line 298"
    rows = [{"item_id": "ruler_of_the_world", "sample_idx": 7, "response": "first\u2028second\u2029third\x85nel\x0bvt"},
            {"item_id": "i2", "sample_idx": 0, "response": "plain"}]
    path = tmp_path / "responses.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert len(path.read_text(encoding="utf-8").splitlines()) > 2      # the trap is real
    assert R.split_jsonl(path.read_text(encoding="utf-8"))[:2] == [json.dumps(r, ensure_ascii=False) for r in rows]
    assert R.read_done(path) == {("ruler_of_the_world", 7), ("i2", 0)}
    # a genuinely corrupt mid-file row still refuses; a torn tail is still truncated away
    path.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n" + '{"item_id": "x", "sam' + "\n" + json.dumps(rows[1]) + "\n", encoding="utf-8")
    with pytest.raises(R.MisalignRunnerError, match="corrupt row at line 2"):
        R.read_done(path)
    path.write_text(json.dumps(rows[1]) + "\n" + '{"item_id": "x", "sam', encoding="utf-8")
    assert R.read_done(path) == {("i2", 0)}
    assert path.read_text(encoding="utf-8") == json.dumps(rows[1]) + "\n"


def test_resume_skips_present_samples(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"])
    _run(tmp_path, http)
    n_first = len(http.posts)
    summary = _run(tmp_path, http)
    assert len(http.posts) == n_first, "a complete cell must issue no requests"
    assert summary["none"] == {"requested": 4, "skipped": 4, "generated": 0,
                               "excluded_items": 0}
    # drop one sample from one cell: exactly that (item, sample) reruns
    path = tmp_path / "out" / "none" / "responses.jsonl"
    rows = _rows(path)
    removed = rows.pop(1)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _run(tmp_path, http)
    assert len(http.posts) == n_first + 1
    rows_after = _rows(path)
    assert len(rows_after) == 4
    assert (removed["item_id"], removed["sample_idx"]) in {
        (r["item_id"], r["sample_idx"]) for r in rows_after}
    meta = json.loads((tmp_path / "out" / "none" / "meta.json").read_text())
    assert meta["resumed_at"] is not None and meta["generated_this_run"] == 1
    # a torn final line is skipped (that sample reruns), never a refusal
    with open(path, "a") as f:
        f.write('{"item_id": "murder_explicit-none", "sample_idx": 1, "resp')
    _run(tmp_path, http)
    assert len(_rows(path)) == 4


def test_growing_n_adds_only_the_new_samples(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"])
    _run(tmp_path, http, n=2)
    n_first = len(http.posts)
    summary = _run(tmp_path, http, n=3)
    assert len(http.posts) == n_first + 2 * len(ITEMS)
    assert summary["none"]["generated"] == len(ITEMS)
    meta = json.loads((tmp_path / "out" / "none" / "meta.json").read_text())
    assert meta["n_samples"] == 3


def test_refuses_when_a_virtual_model_is_missing(tmp_path):
    http = FakeHttp(["petri__base"])  # the prompted vmodel is not served
    with pytest.raises(R.MisalignRunnerError, match="does not serve virtual model"):
        _run(tmp_path, http)
    assert http.posts == [], "the refusal must come before any generation"
    assert not (tmp_path / "out").exists()


def test_refuses_a_foreign_endpoint_unless_plain(tmp_path):
    # One served model may back several cells that differ in their SI (the EM-organisms lane);
    # the duplicate that IS refused is the same (virtual_model, si_id) under two cell ids.
    dup_cells = [
        {"cell_id": "none", "virtual_model": "unsloth/Llama-3.1-8B-Instruct",
         "serving_identity": "vllm", "intervention_hash": "0" * 16, "si_id": None},
        {"cell_id": "none_again", "virtual_model": "unsloth/Llama-3.1-8B-Instruct",
         "serving_identity": "vllm", "intervention_hash": "0" * 16, "si_id": None},
    ]
    with pytest.raises(R.MisalignRunnerError, match=r"duplicate \(virtual_model, si_id\)"):
        _run(tmp_path, FakeHttp(["unsloth/Llama-3.1-8B-Instruct"]), cells=dup_cells)
    plain_cells = [
        {"cell_id": "none", "virtual_model": "unsloth/Llama-3.1-8B-Instruct",
         "serving_identity": "vllm", "intervention_hash": "0" * 16, "si_id": None},
        {"cell_id": "prompted__safety", "virtual_model": "unsloth/Llama-3.1-8B-Instruct",
         "serving_identity": "vllm", "intervention_hash": "b" * 16, "si_id": "safety"},
    ]
    assert len(R.validate_cells(plain_cells)) == 2   # same model, different SI: legal
    plain_cells[1]["virtual_model"] = "em-bad-medical-advice"
    models = ["unsloth/Llama-3.1-8B-Instruct", "em-bad-medical-advice"]
    with pytest.raises(R.MisalignRunnerError, match="not the routing proxy"):
        _run(tmp_path, FakeHttp(models), cells=plain_cells)
    http = FakeHttp(models)
    summary = _run(tmp_path, http, cells=plain_cells, plain=True)
    assert summary["prompted__safety"]["generated"] == 4
    # prompted cells still put the SI first in plain mode
    for _, p in http.posts:
        if p["model"] == "em-bad-medical-advice":
            assert p["messages"][0]["content"].startswith(R.si_text_for("safety"))


def test_endpoint_4xx_aborts_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    http = FakeHttp(["petri__base", "petri__prompted__safety"], fail_status=400)
    with pytest.raises(R.MisalignRunnerError, match="refused the request \\(400\\)"):
        _run(tmp_path, http, cells=CELLS[:1], n=1)
    # no retries: at most one attempt per item (pending work is cancelled
    # after the first refusal), and nothing reaches the store
    assert 1 <= len(http.posts) <= len(ITEMS)
    assert not (tmp_path / "out" / "none" / "responses.jsonl").exists()


def test_transport_failures_retry_then_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    http = FakeHttp(["petri__base"], fail_status=503)
    with pytest.raises(R.MisalignRunnerError, match="failed after 1 retries"):
        _run(tmp_path, http, cells=CELLS[:1], n=1, max_retries=1)
    assert len(http.posts) >= 2


def test_exclude_skips_a_behavior_for_one_cell(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"])
    summary = _run(tmp_path, http, exclude={"none": {"leaking"}})
    assert summary["none"] == {"requested": 2, "skipped": 0, "generated": 2,
                               "excluded_items": 1}
    none_rows = _rows(tmp_path / "out" / "none" / "responses.jsonl")
    assert {r["behavior"] for r in none_rows} == {"murder"}
    prompted_rows = _rows(tmp_path / "out" / "prompted__safety" / "responses.jsonl")
    assert {r["behavior"] for r in prompted_rows} == {"leaking", "murder"}
    meta = json.loads((tmp_path / "out" / "none" / "meta.json").read_text())
    assert meta["excluded_behaviors"] == ["leaking"]
    assert meta["excluded_items"] == ["leaking_explicit-none"]
    with pytest.raises(R.MisalignRunnerError, match="unknown cell"):
        _run(tmp_path, http, exclude={"nemotron3nano4b": {"leaking"}})


def test_resume_onto_a_different_configuration_refuses(tmp_path):
    http = FakeHttp(["petri__base", "petri__prompted__safety"])
    R.run_family("http://127.0.0.1:1", CELLS[:1], "agentic", 1, tmp_path / "out",
                 max_tokens=64, items=ITEMS, suite_revision=REV, http=http)
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch on 'max_tokens'"):
        R.run_family("http://127.0.0.1:1", CELLS[:1], "agentic", 1, tmp_path / "out",
                     max_tokens=65, items=ITEMS, suite_revision=REV, http=http)
    with pytest.raises(R.MisalignRunnerError, match="resume mismatch on 'suite_revision'"):
        R.run_family("http://127.0.0.1:1", CELLS[:1], "agentic", 1, tmp_path / "out",
                     max_tokens=64, items=ITEMS, suite_revision="agentic.other@1",
                     http=http)


def test_validate_cells_refusals():
    with pytest.raises(R.MisalignRunnerError, match="missing field"):
        R.validate_cells([{"cell_id": "none"}])
    with pytest.raises(R.MisalignRunnerError, match="duplicate cell_id"):
        R.validate_cells([CELLS[0], dict(CELLS[0], virtual_model="petri__x")])
    with pytest.raises(R.MisalignRunnerError, match="no cells"):
        R.validate_cells([])


def test_cli_em8q_against_fake_transport(tmp_path, monkeypatch):
    fake = FakeHttp(["petri__base", "petri__prompted__safety"])
    monkeypatch.setattr(R, "HttpxJson", lambda api_key, timeout_s: fake)
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps({
        "target": "gemma4e4b", "model_id": "google/gemma-4-E4B-it",
        "model_revision": "ee0ef602", "cells": CELLS}))
    rc = R.main(["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells_path),
                 "--family", "em8q", "--n", "1", "--out", str(tmp_path / "out"),
                 "--concurrency", "3", "--exclude", "none:nothing_here"])
    assert rc == 0
    assert len(fake.posts) == 2 * 8
    for cid in ("none", "prompted__safety"):
        rows = _rows(tmp_path / "out" / cid / "responses.jsonl")
        assert [r["item_id"] for r in sorted(rows, key=lambda r: r["item_id"])] == sorted(
            BETLEY_8Q_SOURCE_IDS.values())
        assert all(r["behavior"] is None for r in rows)
        meta = json.loads((tmp_path / "out" / cid / "meta.json").read_text())
        assert meta["run"] == {"target": "gemma4e4b", "model_id": "google/gemma-4-E4B-it",
                               "model_revision": "ee0ef602"}
        assert meta["params"]["max_tokens"] == R.DEFAULT_MAX_TOKENS["em8q"] == 600
        assert meta["suite_revision"] == "betley8q.v1"
    # em8q payloads: the prompted cell's system message is exactly the SI
    for _, p in fake.posts:
        if p["model"] == "petri__prompted__safety":
            assert p["messages"][0] == {"role": "system", "content": R.si_text_for("safety")}
        else:
            assert p["messages"][0]["role"] == "user"


def test_package_imports_without_torch():
    code = ("import sys\n"
            "import sisteer.capeval.misalign.runner, sisteer.capeval.misalign.rows, "
            "sisteer.capeval.misalign.analyze\n"
            "assert 'torch' not in sys.modules, 'torch was imported'\n"
            "print('ok')\n")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# --- the real httpx transport against a loopback server ---------------------------


class _LoopbackHandler(__import__("http.server").server.BaseHTTPRequestHandler):
    """A minimal OpenAI-compatible loopback: /v1/models + /v1/chat/completions,
    recording the bearer and refusing a vmodel it does not serve with a 404."""

    served = ["petri__base"]
    seen = []

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.seen.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "petri"} for m in self.served]})
        else:
            self._send(404, {"error": "no such route"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(n))
        self.seen.append(("POST", self.path, self.headers.get("Authorization"), payload))
        if payload["model"] not in self.served:
            self._send(404, {"error": {"code": "unknown_virtual_model"}})
            return
        self._send(200, {"id": "x", "model": payload["model"], "choices": [
            {"index": 0, "finish_reason": "length",
             "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


@pytest.fixture
def loopback():
    from http.server import ThreadingHTTPServer
    _LoopbackHandler.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_httpx_transport_end_to_end_on_loopback(tmp_path, loopback):
    pytest.importorskip("httpx")
    summary = R.run_family(loopback, CELLS[:1], "em8q", 1, tmp_path / "out",
                           concurrency=2, api_key="secret-bearer", timeout_s=30.0)
    assert summary["none"]["generated"] == 8
    rows = _rows(tmp_path / "out" / "none" / "responses.jsonl")
    assert len(rows) == 8 and {r["finish_reason"] for r in rows} == {"length"}
    assert all(r["usage"] == {"prompt_tokens": 1, "completion_tokens": 1} for r in rows)
    gets = [s for s in _LoopbackHandler.seen if s[0] == "GET"]
    posts = [s for s in _LoopbackHandler.seen if s[0] == "POST"]
    assert gets == [("GET", "/v1/models", "Bearer secret-bearer")]
    assert len(posts) == 8 and all(p[2] == "Bearer secret-bearer" for p in posts)
    assert all(p[1] == "/v1/chat/completions" for p in posts)
    # a server-side 4xx surfaces as a refusal with its status and body
    http = R.HttpxJson(api_key="k", timeout_s=10.0)
    try:
        with pytest.raises(R.HttpError) as ei:
            http.post_json(f"{loopback}/v1/chat/completions",
                           {"model": "petri__nope", "messages": []})
        assert ei.value.status == 404 and "unknown_virtual_model" in ei.value.body
        with pytest.raises(R.HttpError) as ei2:
            http.get_json("http://127.0.0.1:9/v1/models")  # nothing listens here
        assert ei2.value.status is None
    finally:
        http.close()


# --- truncation ladder (--rerun-truncated-from) ------------------------------------------


class _CapHttp(FakeHttp):
    """Every leaking sample hits the cap: sample 0 as a repetition loop, sample 1 as prose."""

    def post_json(self, url, payload):
        out = super().post_json(url, payload)
        user = payload["messages"][-1]["content"]
        if "one" in user:  # the leaking item
            idx = sum(1 for _, p in self.posts if p["messages"][-1]["content"] == user
                      and p["model"] == payload["model"]) - 1
            text = "ab" * 400 if idx == 0 else " ".join(f"w{i * 7919 % 1013}" for i in range(400))
            out["choices"][0].update(finish_reason="length", message={"role": "assistant", "content": text})
        return out


def test_truncated_selection_keeps_prose_and_drops_loops(tmp_path):
    http = _CapHttp(["petri__base", "petri__prompted__safety"])
    _run(tmp_path, http)  # first rung: 2 items x 2 samples per cell
    sel, counts = R.truncated_selection(tmp_path / "out")
    for cid in ("none", "prompted__safety"):
        assert counts[cid] == {"truncated": 2, "degenerate": 1, "selected": 1}
        assert sel[cid] == {("leaking_explicit-none", 1)}
    assert R.zlib_ratio("ab" * 400) < R.RERUN_MIN_ZLIB_RATIO < R.zlib_ratio(" ".join(f"w{i * 7919 % 1013}" for i in range(400)))
    with pytest.raises(R.MisalignRunnerError, match="not a directory"):
        R.truncated_selection(tmp_path / "nope")


def test_only_regenerates_the_selected_pairs_into_a_new_root(tmp_path):
    http = _CapHttp(["petri__base", "petri__prompted__safety"])
    _run(tmp_path, http)
    sel, _ = R.truncated_selection(tmp_path / "out")
    http2 = FakeHttp(["petri__base", "petri__prompted__safety"])
    summary = R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 2, tmp_path / "out16k",
                           concurrency=4, max_tokens=128, http=http2, items=ITEMS,
                           suite_revision=REV, only=sel)
    assert len(http2.posts) == 2  # one selected pair per cell
    for cid in ("none", "prompted__safety"):
        assert summary[cid]["generated"] == 1
        rows = _rows(tmp_path / "out16k" / cid / "responses.jsonl")
        assert [(r["item_id"], r["sample_idx"]) for r in rows] == [("leaking_explicit-none", 1)]
        meta = json.loads((tmp_path / "out16k" / cid / "meta.json").read_text())
        assert meta["params"]["max_tokens"] == 128
    # a cell absent from the selection is skipped, not generated
    summary = R.run_family("http://127.0.0.1:1/v1", CELLS, "agentic", 2, tmp_path / "out16k_b",
                           concurrency=4, max_tokens=128, http=FakeHttp(["petri__base", "petri__prompted__safety"]),
                           items=ITEMS, suite_revision=REV, only={"none": sel["none"]})
    assert summary["prompted__safety"] == {"requested": 0, "skipped": 0, "generated": 0, "excluded_items": 0}
    assert not (tmp_path / "out16k_b" / "prompted__safety").exists()
    # a selected pair that is not an item of this family refuses (source/rerun mismatch)
    with pytest.raises(R.MisalignRunnerError, match="not items of this family"):
        R.run_family("http://127.0.0.1:1/v1", CELLS[:1], "agentic", 2, tmp_path / "out16k_c",
                     concurrency=4, max_tokens=128, http=FakeHttp(["petri__base"]),
                     items=ITEMS, suite_revision=REV, only={"none": {("ghost_item", 0)}})


def test_cli_rerun_truncated_from_refuses_the_source_root_and_records_the_rung(tmp_path, monkeypatch):
    http = _CapHttp(["petri__base", "petri__prompted__safety"])
    _run(tmp_path, http)
    fake = FakeHttp(["petri__base", "petri__prompted__safety"])
    monkeypatch.setattr(R, "HttpxJson", lambda api_key, timeout_s: fake)
    monkeypatch.setattr(R, "load_items", lambda family, behaviors: ([dict(i) for i in ITEMS], REV))
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps({"target": "t", "model_id": "m", "model_revision": "r", "cells": CELLS}))
    common = ["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells_path), "--family", "agentic",
              "--n", "2", "--max-tokens", "16384", "--rerun-truncated-from", str(tmp_path / "out")]
    with pytest.raises(R.MisalignRunnerError, match="NEW --out root"):
        R.main(common + ["--out", str(tmp_path / "out")])
    rc = R.main(common + ["--out", str(tmp_path / "out16k")])
    assert rc == 0 and len(fake.posts) == 2
    meta = json.loads((tmp_path / "out16k" / "none" / "meta.json").read_text())
    assert meta["run"]["rerun_of"] == str((tmp_path / "out").resolve())
    assert meta["run"]["rung_max_tokens"] == 16384
    assert meta["run"]["rerun_selection"]["none"] == {"truncated": 2, "degenerate": 1, "selected": 1}
    assert meta["params"]["max_tokens"] == 16384



class _WindowHttp(FakeHttp):
    """vLLM's context-length refusal, as it really behaves: the leaking item's prompt is
    ``true_prompt`` tokens in a 24,576 window, and the message's "at least N input tokens" is
    NOT that length but window - requested + 1."""

    def __init__(self, models, true_prompt):
        super().__init__(models)
        self.true_prompt = true_prompt

    def post_json(self, url, payload):
        if "one" in payload["messages"][-1]["content"] and payload["max_tokens"] + self.true_prompt > 24576:
            with self._lock:
                self.posts.append((url, json.loads(json.dumps(payload))))
            reported = 24576 - payload["max_tokens"] + 1
            raise R.HttpError("refused", status=400, body=json.dumps({"error": {"message":
                "upstream vLLM returned 400: This model's maximum context length is 24576 tokens. However, you "
                f"requested {payload['max_tokens']} output tokens and your prompt contains at least {reported} input "
                "tokens, for a total of at least 24577 tokens. Please reduce the length of the input prompt or the "
                "number of requested output tokens."}}))
        return super().post_json(url, payload)


def test_context_length_400_steps_max_tokens_down_and_records_the_effective_cap(tmp_path):
    assert R.fit_max_tokens("maximum context length is 24576 tokens. ... at least 8193 input tokens") == 24576 - 8193 - 64
    # with a current cap the fit also steps down by at least _FIT_STEP: the reported input is a lower bound
    assert R.fit_max_tokens("maximum context length is 24576 tokens ... at least 8193 input tokens", 16384) == 16384 - 2048
    assert R.fit_max_tokens("maximum context length is 24576 tokens ... at least 24500 input tokens") is None  # no room
    assert R.fit_max_tokens("some other 400") is None
    # a 9,000-token prompted prompt: 16384 refused (reports 8193), the step to 14336 fits
    http = _WindowHttp(["petri__base"], true_prompt=9000)
    R.run_family("http://127.0.0.1:1/v1", CELLS[:1], "agentic", 1, tmp_path / "out", concurrency=2,
                 max_tokens=16384, http=http, items=ITEMS, suite_revision=REV)
    rows = {r["item_id"]: r for r in _rows(tmp_path / "out" / "none" / "responses.jsonl")}
    assert rows["leaking_explicit-none"]["max_tokens_effective"] == 16384 - 2048
    assert "max_tokens_effective" not in rows["murder_explicit-none"]
    leak_posts = [p for _, p in http.posts if "one" in p["messages"][-1]["content"]]
    assert [p["max_tokens"] for p in leak_posts] == [16384, 14336]
    # a prompt that leaves no room within _FIT_ATTEMPTS steps stays a refusal (6 fits, then raise)
    http2 = _WindowHttp(["petri__base"], true_prompt=24000)
    with pytest.raises(R.MisalignRunnerError, match="refused the request \\(400\\)"):
        R.run_family("http://127.0.0.1:1/v1", CELLS[:1], "agentic", 1, tmp_path / "out2", concurrency=1,
                     max_tokens=16384, http=http2, items=ITEMS, suite_revision=REV, max_retries=0)
    leak_posts2 = [p["max_tokens"] for _, p in http2.posts if "one" in p["messages"][-1]["content"]]
    assert leak_posts2 == [16384 - 2048 * k for k in range(R._FIT_ATTEMPTS + 1)]



def test_cli_only_cells_restricts_the_run_and_refuses_unknown_ids(tmp_path, monkeypatch):
    fake = FakeHttp(["petri__base", "petri__prompted__safety"])
    monkeypatch.setattr(R, "HttpxJson", lambda api_key, timeout_s: fake)
    monkeypatch.setattr(R, "load_items", lambda family, behaviors: ([dict(i) for i in ITEMS], REV))
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps({"target": "t", "model_id": "m", "model_revision": "r", "cells": CELLS}))
    common = ["--proxy-url", "http://127.0.0.1:1", "--cells", str(cells_path), "--family", "agentic", "--n", "1"]
    rc = R.main(common + ["--out", str(tmp_path / "out"), "--only-cells", "prompted__safety"])
    assert rc == 0
    assert (tmp_path / "out" / "prompted__safety" / "responses.jsonl").exists()
    assert not (tmp_path / "out" / "none").exists()
    assert {p["model"] for _, p in fake.posts} == {"petri__prompted__safety"}
    meta = json.loads((tmp_path / "out" / "prompted__safety" / "meta.json").read_text())
    assert meta["run"]["only_cells"] == ["prompted__safety"]
    with pytest.raises(R.MisalignRunnerError, match="unknown cell"):
        R.main(common + ["--out", str(tmp_path / "out2"), "--only-cells", "none,ghost"])


def test_ladder_selection_filters_to_the_requested_behaviours(monkeypatch):
    """A leaking-only ladder (qwen: its murder column is the vendor re-sample) reruns only the leaking pairs of the
    source's truncation selection; the murder pairs are counted as left, and a cell left empty drops out — so the
    per-cell 'not items of this run' guard keeps meaning a mismatched rerun root, not a behaviour restriction."""
    from sisteer.capeval.misalign import runner as R
    fake_items = [{"item_id": "leaking_a", "behavior": "leaking"}, {"item_id": "leaking_b", "behavior": "leaking"}]
    monkeypatch.setattr(R, "agentic_items", lambda behaviors=None, **kw: [i for i in fake_items if i["behavior"] in set(behaviors or ("leaking", "murder"))])
    only = {"none": {("leaking_a", 0), ("murder_x", 0), ("murder_x", 1)}, "ctl": {("murder_y", 3)}, "const": {("leaking_b", 2)}}
    kept, left = R.filter_selection_by_behaviors(only, ["leaking"])
    assert kept == {"none": {("leaking_a", 0)}, "const": {("leaking_b", 2)}}
    assert left == {"none": 2, "ctl": 1}
