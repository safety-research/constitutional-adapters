"""The two standup checks of the sdfbase-llama8b wave: the REPRESENTATIVE LOAD CHECK that
makes the provisional budget usable (S concurrent requests at the leg's p99 prompt + cap, every
finish_reason in {stop, length}, no 400 / 5xx, the server alive) and the PRELAUNCH LENGTH CHECK of a
prompted view (system = the full text + each leg's longest prompt must fit prompt + cap <= L at every
cap the leg runs, else the leg is REFUSED — a clamp is a defect for this cell, not a recovery).
Both against fakes: a local HTTP server that answers like the proxy, a word-count tokenizer.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

import load_check as lc
import prompted_window_check as pwc


class _WordTok:
    """One token per whitespace word; the served render adds 17 scaffolding tokens (the measured
    [system, minimal user] + generation-header overhead of the pinned Llama-3.1 template)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        n = sum(len(m["content"].split()) for m in messages)
        return [1] * (n + 17)


# --------------------------------------------------------------------------- the load check

class _Fake:
    """A proxy-shaped chat endpoint: 200 + finish_reason stop, except the request indices in `fail`
    (a 400) and `length_for` (finish_reason length, which is FINE)."""

    def __init__(self, fail=(), length_for=()):
        self.fail, self.length_for, self.seen = set(fail), set(length_for), []
        self.httpd = None

    def start(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.seen.append(body)
                i = int(body.get("seed", 0))
                if i in fake.fail:
                    self.send_response(400); self.end_headers()
                    self.wfile.write(b'{"error": "maximum context length exceeded"}'); return
                fr = "length" if i in fake.length_for else "stop"
                out = {"choices": [{"message": {"content": "ok"}, "finish_reason": fr}],
                       "usage": {"prompt_tokens": len(" ".join(m["content"] for m in body["messages"]).split()),
                                 "completion_tokens": 3}}
                data = json.dumps(out).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def stop(self):
        self.httpd.shutdown()


FULL_HASH = "82f88c5532e4939e"     # sha256_hex16 of the registered constitution_full prefix bytes


def _endpoint(tmp_path, url, S=5, L=45056):
    # the standup's routing table beside endpoint.json: `none` and the prompted constitution route
    # (a prefixed probe resolves ITS route from here, by registered prefix hash — never `none`)
    table = tmp_path / "routing_table.json"
    table.write_text(json.dumps({
        "version": 1, "base_model": "m", "serving_mode": "interleaved", "exact_replay_strategy": None,
        "routes": [
            {"virtual_model": "petri__base", "cell_id": "none", "cell_class": "base", "serving_identity": "x",
             "intervention_hash": "0" * 16},
            {"virtual_model": "petri__prompted__constitution_full", "cell_id": "prompted__constitution_full",
             "cell_class": "prompted", "serving_identity": "x", "intervention_hash": "1" * 16,
             "si_prefix_hash": FULL_HASH}],
        "prompted_prefixes": [{"virtual_model": "petri__prompted__constitution_full", "byte_length": 183701,
                               "sha16": FULL_HASH}]}))
    ep = {"proxy_url": url, "max_model_len": L, "max_num_seqs": S, "model": "m", "model_revision": "r",
          "served_models": ["petri__base", "petri__prompted__constitution_full"], "pid": None,
          "routing_table": str(table)}
    p = tmp_path / "endpoint.json"
    p.write_text(json.dumps(ep))
    return p


@pytest.fixture()
def tok(monkeypatch):
    import sisteer.models as sm
    monkeypatch.setattr(sm, "load_tokenizer", lambda name, revision=None, **kw: _WordTok())
    # the geometry table's entry for the leg: p99 prompt 762, requested cap 4,096 (families_served:short)
    monkeypatch.setattr(lc, "table_entry",
                        lambda ev: SimpleNamespace(basis=SimpleNamespace(p99_prompt=762, requested_max_tokens=4096)))
    return _WordTok()


def test_load_check_passes_when_every_request_finishes_and_records_it(tmp_path, tok):
    fake = _Fake(length_for={2})
    url = fake.start()
    try:
        ep = _endpoint(tmp_path, url, S=5)
        rc = lc.main(["--endpoint", str(ep), "--eval", "families_served:short", "--timeout-s", "30"])
    finally:
        fake.stop()
    assert rc == 0
    rec = json.loads((tmp_path / "load_check.json").read_text())
    assert rec["ok"] and rec["S"] == 5 == rec["n_requests"] == rec["n_ok"] and rec["L"] == 45056 and rec["cap"] == 4096
    assert rec["finish_reasons"] == {"stop": 4, "length": 1} and rec["n_bad"] == 0
    assert rec["probe"]["target_prompt_tokens"] == 762 and 700 <= rec["max_prompt_tokens"] <= 762
    assert rec["probe"]["cap_source"] == "table:basis.requested_max_tokens"     # no --config, no --cap: the table's
    assert len(fake.seen) == 5 and all(b["model"] == "petri__base" and b["max_tokens"] == 4096 for b in fake.seen)


def test_load_check_refuses_on_one_400_and_names_the_limit(tmp_path, tok):
    fake = _Fake(fail={3})
    url = fake.start()
    try:
        ep = _endpoint(tmp_path, url, S=5)
        rc = lc.main(["--endpoint", str(ep), "--eval", "families_served:short", "--timeout-s", "30"])
    finally:
        fake.stop()
    assert rc == 3
    rec = json.loads((tmp_path / "load_check.json").read_text())
    assert rec["ok"] is False and rec["n_bad"] == 1 and rec["bad"][0]["status"] == 400
    assert "MEASURED LIMIT" in rec["verdict"] and "never shrink S" in rec["verdict"]


def test_load_check_sends_the_full_prefix_as_the_system_message_and_the_cap(tmp_path, tok):
    fake = _Fake()
    url = fake.start()
    try:
        ep = _endpoint(tmp_path, url, S=2)
        rc = lc.main(["--endpoint", str(ep), "--eval", "families_served:short", "--cap", "8192",
                      "--system-si", "si_constitution_full", "--timeout-s", "30"])
    finally:
        fake.stop()
    assert rc == 0
    from sisteer.data.instructions import BY_ID
    b = fake.seen[0]
    assert b["messages"][0] == {"role": "system", "content": BY_ID["si_constitution_full"].text}
    assert b["max_tokens"] == 8192 and b["temperature"] == 0.0
    rec = json.loads((tmp_path / "load_check.json").read_text())
    assert rec["system_si"] == "si_constitution_full" and rec["max_prompt_tokens"] >= 28977
    assert rec["vmodel"] == "petri__prompted__constitution_full"          # the prompted route, never `none`
    assert all(x["model"] == "petri__prompted__constitution_full" for x in fake.seen)


def test_load_check_refuses_an_endpoint_record_without_the_geometry(tmp_path, tok):
    (tmp_path / "endpoint.json").write_text(json.dumps({"proxy_url": "http://x/v1"}))
    with pytest.raises(SystemExit, match="lacks 'max_model_len'"):
        lc.main(["--endpoint", str(tmp_path / "endpoint.json"), "--eval", "families_served:short"])


# --------------------------------------------------------------------------- the window check

def _goals(texts):
    return [SimpleNamespace(goal_id=f"g{i}", text=t) for i, t in enumerate(texts)]


def test_window_check_passes_the_measured_full_constitution_inputs_at_45056_and_refuses_a_short_window():
    tok = _WordTok()
    system = "w " * 33_980           # the raw full-constitution token count
    static = {fam: ("s " * 434, "g7") for fam in pwc.STATIC_FAMILIES}   # grok420's longest prompt: 434
    bon = _goals(["b " * 60])
    caps = {"gsm8k": "q " * 300, "mmlu": "m " * 250}
    rec = pwc.check(tok=tok, system=system, max_model_len=45056, legs=("static", "bon", "capability"),
                    static_prompts=static, bon_goals=bon, cap_prompts=caps, extra_rungs=(16384,))
    assert rec["ok"] and rec["problems"] == [] and rec["system_tokens_rendered"] == 33_980 + 17
    needs = {(r["leg"], r["family"], r["cap"]): r["need"] for r in rec["rows"]}
    assert needs[("static", "grok420", 4096)] == 33_980 + 434 + 17 + 4096       # 38,527 <= 45,056
    assert needs[("static", "grok420", 8192)] == 33_980 + 434 + 17 + 8192       # 42,623 <= 45,056
    assert needs[("bon", "bon", 512)] == 33_980 + 120 + 17 + 512
    assert all(r["fits"] for r in rec["rows"] if r["cap"] != 16384)
    # the 16,384 rung is REPORTED as not fitting at 45,056 (34,431 + 16,384 = 50,815) — it stands up at its own window
    assert rec["extra_rungs"] == [{"cap": 16384, "fits_at_L": False}]
    # a window that cannot hold the static cap refuses the leg, naming the tokens it needs
    short = pwc.check(tok=tok, system=system, max_model_len=36_864, legs=("static", "bon", "capability"),
                      static_prompts=static, bon_goals=bon, cap_prompts=caps)
    assert not short["ok"] and any("static/grok420 @ cap 4096" in p for p in short["problems"])
    assert any("@ cap 8192" in p for p in short["problems"])
    # bon / capability at 512 still fit there, so the refusal is specific to the leg that does not
    assert all(r["fits"] for r in short["rows"] if r["leg"] in ("bon", "capability"))


def test_window_check_real_tokenizer_render_of_the_full_constitution():
    """The plan's measured figures on the PINNED tokenizer (33,980 raw; 33,997 rendered as [system,
    minimal user] + the generation header) — skipped where the snapshot is not cached offline."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sisteer.data.instructions import BY_ID
    from sisteer.models import load_tokenizer
    try:
        tok = load_tokenizer("NousResearch/Meta-Llama-3.1-8B-Instruct",
                             revision="d10aef7999a2b5ba950ab3974312feeedbfe0b77")
    except Exception as e:  # noqa: BLE001 — recorded, never swallowed
        pytest.skip(f"pinned tokenizer not cached offline: {type(e).__name__}")
    text = BY_ID["si_constitution_full"].text
    assert len(tok(text, add_special_tokens=False)["input_ids"]) == 33_980
    # the plan's "minimal user turn" is a 2-token one ("Hello."): 33,980 + 15 scaffolding + 2 = 33,997;
    # an EMPTY user turn renders to 33,995 (measured 2026-09-10 on the cached snapshot) — the window
    # check itself renders each leg's real longest prompt, so this pin is about the scaffold, not a leg
    assert pwc.rendered_tokens(tok, text, "") == 33_995
    assert pwc.rendered_tokens(tok, text, "Hello.") == 33_997


# --------------------------------------------------------------------------- the wiring

def test_common_sh_runs_the_load_check_after_every_standup_and_the_window_check_before_a_prompted_one():
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    i_up = common.index('f22_log "target up:')
    i_lc = common.index("scripts/sdfbase/full22/load_check.py --endpoint")
    assert i_up < i_lc, "the load check runs AFTER the attestation checks and the target-up log"
    assert 'touch "$out/STOP"; return 1' in common[i_lc:i_lc + 1200]      # a failed load check refuses traffic
    assert "MEASURED LIMIT" in common or "measured limit" in common
    assert '${F22_LOAD_CHECK:-1}' in common and "F22_LOAD_CHECK_CAP" in common and "F22_LOAD_CHECK_SYSTEM_SI" in common
    assert "f22_window_check(){" in common and "prompted_window_check.py --config" in common
    assert 'roster_view.py "$KEY" "$VIEW" prompted_only' in common
    for sb in ("scripts/sdfbase/full22/target_static.sbatch", "scripts/sdfbase/full22/sampled/target_sampled.sbatch"):
        src = (REPO / sb).read_text()
        # every standup of a prompted-only view is preceded by the window check at the window it stands up at
        assert 'f22_window_check "$SERVE_DIR/static"' in src, sb
        assert src.index('f22_window_check "$SERVE_DIR/static"') < src.index('f22_standup attacks "$SERVE_DIR/static"'), sb
        assert 'F22_LOAD_CHECK_CAP="${F22_LOAD_CHECK_CAP:-8192}"' in src, sb     # the fullconst smoke: 5 x full prefix at 8192
        assert "F22_LOAD_CHECK_SYSTEM_SI" in src, sb
