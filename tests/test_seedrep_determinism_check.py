"""The Qwen seed-replicate battery's determinism spot-check on synthetic roots with an injected replay:
the 20 lowest-sha256 items are selected, their retained wire bodies are held to the records' digests and replayed
byte-identically, texts compare byte-for-byte, a miss records its first divergence (char + token, basis stated), the
engine must be DRAINED (or `--allow-busy` recorded), multi-turn replays turn 0 only, and every missing input refuses.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import determinism_check as dc  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402


def _faithful_post(raw: bytes, headers):
    body = json.loads(raw)
    item = body["messages"][0]["content"][len("prompt "):]
    return {"id": "chatcmpl-replay", "choices": [{"message": {"content": f"response for {item} under {body['model']}"}, "finish_reason": "stop"}]}


@pytest.fixture(autouse=True)
def _hooks(monkeypatch):
    monkeypatch.setattr(dc, "POST_FN", _faithful_post)
    monkeypatch.setattr(dc, "METRICS_FN", lambda: {"num_requests_running": 0, "num_requests_waiting": 0, "generation_tokens_total": 10})
    monkeypatch.setattr(dc, "TOKENIZE_FN", None)
    yield


def _root(tmp_path: Path, leg="pasttense", n_items=25):
    root = fx.make_root(tmp_path / "stores", leg=leg, view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=n_items)
    return root, ac.read_attempt_root(root.path)


def test_selection_is_the_20_lowest_sha256_items_and_replay_reproduces(tmp_path: Path):
    root, ar = _root(tmp_path)
    rec = dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=False, petri_behaviours=[])
    all_items = [f"g{k}" for k in range(25)]
    want = sorted(all_items, key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:20]
    assert rec["arms"][0]["items"] == want
    assert rec["n_items"] == 20 and rec["n_requests"] == 40 and rec["identical"] == 40 and rec["reproduces"] is True
    assert rec["misses"] == [] and rec["engine_batch_state"]["drained"] is True and rec["replay_bypasses_ledger"] is True
    assert rec["replay_target"] == "http://127.0.0.1:18080/v1" and rec["S"] == 45 and rec["L"] == 11520 and rec["W"] == fx.W_FIXTURE
    assert rec["first_turn_only"] is False and rec["token_basis"] == "whitespace"


def test_divergence_recorded_per_miss_with_stated_token_basis(tmp_path: Path, monkeypatch):
    root, ar = _root(tmp_path)
    all_items = sorted([f"g{k}" for k in range(25)], key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    victim = all_items[0]

    def post(raw, headers):
        out = _faithful_post(raw, headers)
        body = json.loads(raw)
        if body["messages"][0]["content"] == f"prompt {victim}/1":
            out["choices"][0]["message"]["content"] = f"response for {victim}/1 under DIFFERENT engine"
        return out

    monkeypatch.setattr(dc, "POST_FN", post)
    rec = dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=False, petri_behaviours=[])
    assert rec["identical"] == 39 and rec["reproduces"] is False and len(rec["misses"]) == 1
    m = rec["misses"][0]
    assert m["item_id"] == victim and m["sample_idx"] == 1 and m["arm"] == "none"
    orig = f"response for {victim}/1 under petri__none"
    assert m["first_divergence_char"] == len("response for ") + len(f"{victim}/1 under ")
    assert m["first_divergence_token"] == 4 and m["token_basis"] == "whitespace"
    assert m["original_sha256"] == hashlib.sha256(orig.encode()).hexdigest() and m["replay_sha256"] != m["original_sha256"]
    # a served tokenizer, when injected, is the stated basis
    monkeypatch.setattr(dc, "TOKENIZE_FN", lambda text: [ord(c) for c in text])
    rec2 = dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                        allow_busy=False, petri_behaviours=[])
    assert rec2["misses"][0]["token_basis"] == "served_tokenizer" and rec2["misses"][0]["first_divergence_token"] == m["first_divergence_char"]


def test_busy_engine_refuses_unless_allowed(tmp_path: Path, monkeypatch):
    root, ar = _root(tmp_path, n_items=20)
    monkeypatch.setattr(dc, "METRICS_FN", lambda: {"num_requests_running": 3, "num_requests_waiting": 0})
    with pytest.raises(ac.AuditRefusal, match="still has 3 request"):
        dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                     allow_busy=False, petri_behaviours=[])
    rec = dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=True, petri_behaviours=[])
    assert rec["engine_batch_state"]["drained"] is False and rec["engine_batch_state"]["allow_busy"] is True


def test_missing_bodies_or_tampered_body_refuses(tmp_path: Path):
    """The wire bytes are read from the request-evidence RECORD's additive `body_b64` (held to the record's own
    body_sha256 and body_bytes) — no `request_bodies.jsonl` side file exists or is read. A tampered body → "not the wire
    bytes"; a record without the field (what the recorder writes today: digest + length) → the replay REFUSES by name,
    never reconstructs a body."""
    root, ar = _root(tmp_path, n_items=20)
    ev = root.path / ac.EVIDENCE_FILE
    assert not (root.path / "request_bodies.jsonl").exists()
    recs = ac.read_jsonl(ev)
    victim = next(r["request_id"] for r in recs if r["kind"] == "task")
    fx.rewrite_jsonl(ev, lambda r: {**r, ac.REQUEST_BODY_B64_FIELD: r[ac.REQUEST_BODY_B64_FIELD][:-4] + "QUJD"} if r["request_id"] == victim else r)
    with pytest.raises(ac.AuditRefusal, match="not the wire bytes"):
        dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                     allow_busy=False, petri_behaviours=[])
    fx.rewrite_jsonl(ev, lambda r: {k: v for k, v in r.items() if k != ac.REQUEST_BODY_B64_FIELD})
    with pytest.raises(ac.AuditRefusal, match="retains no wire bytes \\(body_b64; it records body_sha256 and body_bytes, a digest and a length\\)"):
        dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                     allow_busy=False, petri_behaviours=[])


def test_text_less_store_without_the_response_hash_refuses(tmp_path: Path):
    """GSM8K stores hold no text: the original is the response record's additive `text_sha256`; a response record WITHOUT it
    (the recorder today) leaves nothing to compare against and the replay REFUSES by name — the hash is consumed when
    present, never invented."""
    root = fx.make_root(tmp_path / "stores", leg="gsm8k", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=20)
    ar = ac.read_attempt_root(root.path)
    for p in root.pipeline_files(ac.RESPONSE_EVIDENCE_FILE):  # a GSM8K row's response records live per child store (fz2)
        fx.rewrite_jsonl(p, lambda r: {k: v for k, v in r.items() if k != ac.RESPONSE_TEXT_SHA_FIELD})
    with pytest.raises(ac.AuditRefusal, match="no text_sha256 on the response record of attempt"):
        dc.run_check(ar, leg="gsm8k", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                     allow_busy=False, petri_behaviours=[])


def test_fewer_items_than_n_refuses(tmp_path: Path):
    root, ar = _root(tmp_path, n_items=5)
    with pytest.raises(ac.AuditRefusal, match="files 5 item"):
        dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                     allow_busy=False, petri_behaviours=[])


def test_unknown_arm_refuses(tmp_path: Path):
    root, ar = _root(tmp_path, n_items=20)
    with pytest.raises(ac.AuditRefusal, match="no pasttense store for arm 'nope'"):
        dc.run_check(ar, leg="pasttense", arms=["nope"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                     allow_busy=False, petri_behaviours=[])


def test_multiturn_replays_first_turn_only(tmp_path: Path):
    root, ar = _root(tmp_path, leg="pair", n_items=20)
    rec = dc.run_check(ar, leg="pair", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=False, petri_behaviours=[])
    assert rec["first_turn_only"] is True and rec["n_requests"] == 20 and rec["identical"] == 20


def test_inspect_leg_compares_by_response_sha(tmp_path: Path, monkeypatch):
    """GSM8K stores hold no text: the original is the `response_evidence.jsonl` text_sha256; a miss has no char index and says so."""
    root = fx.make_root(tmp_path / "stores", leg="gsm8k", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=20)
    ar = ac.read_attempt_root(root.path)

    def post(raw, headers):
        body = json.loads(raw)
        item = body["messages"][0]["content"][len("q "):]
        return {"id": "x", "choices": [{"message": {"content": f"response for {item}/0 under {body['model']}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(dc, "POST_FN", post)
    rec = dc.run_check(ar, leg="gsm8k", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=False, petri_behaviours=[])
    assert rec["identical"] == 20
    monkeypatch.setattr(dc, "POST_FN", lambda raw, h: {"id": "x", "choices": [{"message": {"content": "other"}, "finish_reason": "stop"}]})
    rec2 = dc.run_check(ar, leg="gsm8k", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                        allow_busy=False, petri_behaviours=[])
    assert rec2["identical"] == 0 and rec2["misses"][0]["first_divergence_char"] is None
    assert rec2["misses"][0]["token_basis"].startswith("unavailable")


def test_cli_writes_record_and_copy(tmp_path: Path):
    root, ar = _root(tmp_path, n_items=20)
    out = root.path / ac.DETERMINISM_FILE
    copy = tmp_path / "audits" / "determinism" / "pasttense.json"
    rc = ac.run_main(dc.main, ["--root", str(root.path), "--leg", "pasttense", "--n", "20", "--out", str(out), "--copy-to", str(copy)])
    assert rc == 0 and out.is_file() and copy.is_file()
    assert json.loads(out.read_text())["identical"] == 40 and json.loads(copy.read_text()) == json.loads(out.read_text())
    assert ac.run_main(dc.main, ["--root", str(root.path), "--leg", "pasttense", "--n", "0", "--out", str(out)]) == ac.EXIT_REFUSED


def test_integration_no_credential_file_flag_the_proxy_bearer_rides_the_environment(tmp_path: Path, monkeypatch):
    """Integration fix (W gate 2, main's `tests/test_no_credential_files.py::test_no_source_file_takes_a_key_file_flag`): the repo's vault
    rule forbids a flag that takes a credential FILE; `--api-key-file` is gone and the proxy bearer is read from `$VLLM_API_KEY`
    (`vllm_auth.TOKEN_ENV`, the serving stack's one bearer name — how seedrep_petri_drive.py reads a pool bearer). KNOWN-BAD first: the
    parser refuses the old flag; then the env value reaches `run_check` as `api_key`, and an unset / blank env is None."""
    import re
    src = Path(dc.__file__).read_text()
    assert re.search(r"""["']--?(?:api-)?(?:anthropic-)?key-file["']""", src) is None, "a credential FILE flag is back"
    assert dc.PROXY_BEARER_ENV == "VLLM_API_KEY"
    root, ar = _root(tmp_path, n_items=20)
    out = root.path / ac.DETERMINISM_FILE
    with pytest.raises(SystemExit):
        dc.main(["--root", str(root.path), "--leg", "pasttense", "--n", "20", "--out", str(out), "--api-key-file", str(tmp_path / "k")])
    seen = {}
    real = dc.run_check

    def spy(root_, **kw):
        seen["api_key"] = kw.get("api_key")
        return real(root_, **kw)
    monkeypatch.setattr(dc, "run_check", spy)
    monkeypatch.setenv(dc.PROXY_BEARER_ENV, " tok-123 ")
    assert ac.run_main(dc.main, ["--root", str(root.path), "--leg", "pasttense", "--n", "20", "--out", str(out)]) == 0 and seen["api_key"] == "tok-123"
    monkeypatch.setenv(dc.PROXY_BEARER_ENV, "")
    assert ac.run_main(dc.main, ["--root", str(root.path), "--leg", "pasttense", "--n", "20", "--out", str(out)]) == 0 and seen["api_key"] is None
    monkeypatch.delenv(dc.PROXY_BEARER_ENV)
    assert ac.run_main(dc.main, ["--root", str(root.path), "--leg", "pasttense", "--n", "20", "--out", str(out)]) == 0 and seen["api_key"] is None


# --------------------------------------------------------------------------- F-U: the smoke id form parses through the ONE parser
KNOWN_BAD_SMOKE_ID = "smoke-S3__misalign__s1__base__shard00"     # the exact id every smoke row's determinism check REFUSED


def _smoke_root(tmp_path: Path, smoke_id: str = "S3", leg: str = "pasttense"):
    """The synthetic root re-keyed the way `seedrep_launch_plan.smoke_row` keys a smoke row: ATTEMPT.json's `launch_plan_row` = the
    production standup id under the `smoke-<S>__` prefix (the control key stays the six axes under the smoke wave)."""
    root = fx.make_root(tmp_path / "stores", leg=leg, view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=25)
    ap = root.path / ac.ATTEMPT_FILE
    att = ac.read_json(ap)
    att["launch_plan_row"] = f"smoke-{smoke_id}__{root.standup_id}"
    ap.write_text(json.dumps(att, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def test_the_known_bad_smoke_id_parses_to_its_axes_and_smoke_id():
    assert ac.split_standup_id(KNOWN_BAD_SMOKE_ID) == ("S3", "misalign", "s1", "base", "shard00")
    assert ac.parse_standup_id(KNOWN_BAD_SMOKE_ID) == ("misalign", "s1", "base", "shard00")
    assert ac.smoke_standup_id_of("S3", "misalign", "s1", "base", 0) == KNOWN_BAD_SMOKE_ID
    # a production id is unchanged: no smoke id, the same four axes
    assert ac.split_standup_id("misalign__s1__base__shard00") == (None, "misalign", "s1", "base", "shard00")
    assert ac.parse_standup_id("pasttense__s3__8192__shard07") == ("pasttense", "s3", "8192", "shard07")
    assert ac.standup_id_of("pasttense", "s3", "8192", 7) == "pasttense__s3__8192__shard07"


@pytest.mark.parametrize("bad", [
    "smoke-S3__misalign__s1__base",                   # smoke prefix, three axes
    "misalign__s1__base__shard00__extra",             # five parts, no smoke prefix
    "probe-S3__misalign__s1__base__shard00",          # a prefix that is not the plan's
    "smoke-__misalign__s1__base__shard00",            # an empty smoke id
    "smoke-S3__misalign__s1__base__shardXX",          # a shard not spelled shard<NN>
    "smoke-S3__S3__misalign__s1__base__shard00",      # six parts
    "misalign__s1__base",
])
def test_an_id_of_neither_form_is_refused_by_name(bad):
    with pytest.raises(ac.AuditRefusal, match=r"is neither <leg>__<view>__<stage>__shard<NN> \(production\) or smoke-<S>__"):
        ac.split_standup_id(bad)
    with pytest.raises(ac.AuditRefusal):
        ac.parse_standup_id(bad)


def test_the_parser_round_trips_the_plans_own_smoke_row_composer():
    """The plan composes the smoke id (`seedrep_launch_plan.smoke_row`); the audit parses it. One spelling, pinned end to end: a drift in
    either side fails here rather than in a container at hour twenty."""
    repo = Path(__file__).resolve().parents[1]
    for _p in (repo / "src", repo / "scripts", repo / "scripts" / "modal"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import seedrep_launch_plan as lp  # noqa: E402
    W7, wu = fx.W_FIXTURE[:7], "12345678-1234-4123-8123-123456789abc"
    prod_wave = "87654321-4321-4321-8321-cba987654321"
    row = {"standup_id": lp.standup_id("misalign", "s1", "base", 0), "leg": "misalign", "engine": "s1", "stage": "base", "shard": 0,
           "adapters": {}, "control_key": lp.control_key(W7, prod_wave, "misalign", "s1", "base", 0), "children": [{"arm": "none"}],
           "modal": {"timeout_s": 3600}}
    smoke = lp.smoke_row(row, {"smoke_id": "S3", "wave_uuid": wu, "W7": W7, "W": fx.W_FIXTURE}, max_children=5, max_timeout_s=1800)
    assert smoke["standup_id"] == KNOWN_BAD_SMOKE_ID
    assert ac.split_standup_id(smoke["standup_id"]) == ("S3", "misalign", "s1", "base", "shard00")
    assert ac.smoke_standup_id_of("S3", "misalign", "s1", "base", 0) == smoke["standup_id"]
    # the audit's production spelling is the plan's (already pinned by the coverage tests; restated beside the smoke form)
    assert ac.standup_id_of("misalign", "s1", "base", 0) == row["standup_id"] == smoke["smoke"]["production_standup_id"]


def test_a_smoke_root_reads_with_its_smoke_id_and_the_check_writes_its_record(tmp_path: Path):
    """F-U end to end: the known-bad id form on ATTEMPT.json's `launch_plan_row` → `read_attempt_root` resolves the axes AND the smoke id;
    the REAL `main` writes determinism.json carrying `smoke_id` beside leg / view / stage / shard, rc 0."""
    root = _smoke_root(tmp_path, smoke_id="S3")
    ar = ac.read_attempt_root(root.path)
    assert (ar.smoke_id, ar.leg, ar.view, ar.stage, ar.shard) == ("S3", "pasttense", "s1", "base", "shard00")
    assert ar.standup_id == "smoke-S3__pasttense__s1__base__shard00" and ar.production_standup_id == "pasttense__s1__base__shard00"
    out = root.path / "audits" / "determinism" / "pasttense.json"
    rc = dc.main(["--root", str(root.path), "--leg", "pasttense", "--proxy-url", "http://127.0.0.1:18080/v1",
                  "--metrics-url", "http://127.0.0.1:8000/metrics", "--out", str(out), "--copy-to", str(root.path / ac.DETERMINISM_FILE)])
    assert rc == ac.EXIT_OK and out.is_file()
    rec = json.loads(out.read_text())
    assert rec["smoke_id"] == "S3" and rec["standup_id"] == "smoke-S3__pasttense__s1__base__shard00"
    assert (rec["leg"], rec["view"], rec["stage"], rec["shard"]) == ("pasttense", "s1", "base", "shard00")
    assert rec["attempt_id"] == ar.attempt_id and rec["reproduces"] is True and rec["n_items"] == 20
    assert (root.path / ac.DETERMINISM_FILE).read_bytes() == out.read_bytes()


def test_a_production_root_records_a_null_smoke_id_and_its_production_id(tmp_path: Path):
    root, ar = _root(tmp_path)
    assert ar.smoke_id is None and ar.standup_id == ar.production_standup_id == "pasttense__s1__base__shard00"
    rec = dc.run_check(ar, leg="pasttense", arms=["none"], n=20, proxy_url=None, api_key=None, metrics_url=None, tokenize_url=None,
                       allow_busy=False, petri_behaviours=[])
    assert rec["smoke_id"] is None and rec["standup_id"] == "pasttense__s1__base__shard00"
    assert (rec["leg"], rec["view"], rec["stage"], rec["shard"]) == ("pasttense", "s1", "base", "shard00")


def test_a_smoke_root_whose_sources_name_two_smoke_ids_is_refused(tmp_path: Path):
    """A dict-shaped `launch_plan_row` (read_attempt's shape) carrying `smoke.smoke_id` must agree with the id's own prefix."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    ap = root.path / ac.ATTEMPT_FILE
    att = ac.read_json(ap)
    att["launch_plan_row"] = {"standup_id": f"smoke-S3__{root.standup_id}", "leg": "pasttense", "view": "s1", "stage": "base", "shard": 0,
                              "smoke": {"smoke_id": "S1b"}}
    ap.write_text(json.dumps(att) + "\n", encoding="utf-8")
    with pytest.raises(ac.AuditRefusal, match="smoke id 'S1b' \\(launch_plan_row.smoke.smoke_id\\) disagrees with 'S3'"):
        ac.read_attempt_root(root.path)
    att["launch_plan_row"]["smoke"]["smoke_id"] = "S3"
    ap.write_text(json.dumps(att) + "\n", encoding="utf-8")
    assert ac.read_attempt_root(root.path).smoke_id == "S3"


def test_a_malformed_id_on_attempt_json_still_refuses_the_root_by_name(tmp_path: Path):
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    ap = root.path / ac.ATTEMPT_FILE
    att = ac.read_json(ap)
    att["launch_plan_row"] = "probe-S3__pasttense__s1__base__shard00"
    ap.write_text(json.dumps(att) + "\n", encoding="utf-8")
    with pytest.raises(ac.AuditRefusal, match="is neither"):
        ac.read_attempt_root(root.path)


def test_the_audits_smoke_id_regex_is_the_plans_regex_pattern_and_flags():
    """The F-U delta: `audit_common.SMOKE_ID_RE` MIRRORS `seedrep_launch_plan.SMOKE_ID_RE` (the audit imports no plan module) — pin
    pattern AND flags equal so the mirror cannot drift unnoticed; `SMOKE_STANDUP_PREFIX` likewise equals the plan's `smoke_row` prefix."""
    import re as _re
    import seedrep_launch_plan as lp  # noqa: E402
    assert ac.SMOKE_ID_RE.pattern == lp.SMOKE_ID_RE.pattern
    assert ac.SMOKE_ID_RE.flags == lp.SMOKE_ID_RE.flags
    # the composer's spelling: the plan keys a smoke row `smoke-<smoke_id>__<standup_id>`
    assert ac.SMOKE_STANDUP_PREFIX == "smoke-"
    assert _re.match(ac.SMOKE_ID_RE, "S3") and not _re.match(ac.SMOKE_ID_RE, "")
