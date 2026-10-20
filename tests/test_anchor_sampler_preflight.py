"""scripts/sdfbase/seedrep/anchor_sampler_preflight.py: the standup's ENGINE-RESOLVED
sampler against the reused anchor's — never the keys a client sent.

The NEGATIVE case the tool exists for: both clients omit every sampling key, but the anchor's server resolved an omitted
key to vLLM's neutral defaults (no generation_config.json, no override) while the standup's server resolved it to the
Qwen vendor row (0.7 / 0.8 / 20) -> DIVERGED, row marker non_comparable(...), exit 1. The same two
clients on two servers that both resolve to the neutral tuple -> AGREES, comparable, exit 0. A missing anchor, a
client-only anchor record with no way to resolve it, a mixed client class -> UNKNOWN (exit 2), never "assumed equal";
``sampler_check(None, None, not_sent=True)`` is asserted NOT to be the verdict source.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import test_record_resolved_sampler as R  # noqa: E402  (the synthetic vLLM log-line fixtures)


def _load():
    spec = importlib.util.spec_from_file_location("_anchor_preflight", REPO / "scripts/sdfbase/seedrep/anchor_sampler_preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NEUTRAL = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
VENDOR = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
CAP = 1024


def _line(i, *, cap=CAP, seed="None", **kw):
    """A vLLM request_logger line for request chatcmpl-<i>; a Petri-shaped request sends no seed (`seed=None`)."""
    return R._sp(i, cap=cap, seed=0, **kw).replace("seed=0", f"seed={seed}")


def _standup(tmp_path, name, lines, *, policy="contract_override", override=None):
    d = tmp_path / name
    d.mkdir()
    (d / "vllm_server.log").write_text("\n".join([R._STARTED, R._ATTEST_INFO, *lines]) + "\n")
    (d / "endpoint.json").write_text(json.dumps({"server_default_policy": policy, "override_generation_config": override,
                                                 "engine": "lora_native", "model": "Qwen/Qwen3.5-4B", "model_revision": "0" * 40}))
    return d


def test_negative_both_clients_omit_params_but_the_servers_resolve_differently_diverges(tmp_path):
    m = _load()
    # the standup: a Petri-shaped probe (no sampling key sent) that the engine resolved to the VENDOR row under contract_override
    su = _standup(tmp_path, "standup", [_line(7, **R._VENDOR)], override='{"temperature":0.7,"top_k":20,"top_p":0.8}')
    # the anchor: its client record states what it SENT (nothing) — resolvable only by naming the anchor server's policy
    anchor = tmp_path / "petri_anchor_client.json"
    anchor.write_text(json.dumps({"target_decode": {}, "note": "petri exposes no sampler; harness-drawn"}))
    rec = m.run("petri", "s1", su, anchor=anchor, anchor_server_policy="engine_defaults", request_id="chatcmpl-7",
                out_root=tmp_path / "out")
    assert rec["verdict"] == "DIVERGED"
    assert rec["row_marker"].startswith("non_comparable(") and "temperature: 0.7 vs anchor 1.0" in rec["row_marker"]
    assert {d["key"] for d in rec["diff"]} == {"temperature", "top_p", "top_k", "presence_penalty"}
    assert rec["effective"]["server_default_policy"] == "contract_override"
    assert rec["anchor"]["anchor_server_policy"] == "engine_defaults" and rec["anchor"]["sent"] == {}
    assert (tmp_path / "out" / "petri_s1.json").is_file()
    assert m.EXIT[rec["verdict"]] == 1


def test_positive_the_same_omitted_params_on_two_neutral_servers_agree(tmp_path):
    m = _load()
    su = _standup(tmp_path, "standup", [_line(3, **R._NEUTRAL)], policy="engine_defaults")
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({"target_decode": {}}))
    rec = m.run("petri", "s1", su, anchor=anchor, anchor_server_policy="engine_defaults", request_id="chatcmpl-3", out_root=tmp_path / "out")
    assert rec["verdict"] == "AGREES" and rec["row_marker"] == "comparable" and rec["diff"] == []
    assert rec["effective"]["server_default_policy"] == "engine_defaults"
    # explicit params equal to the anchor's resolved tuple agree too (the multi-turn spec: T 1.0 / top_p 1.0 / top_k -1)
    anchor2 = tmp_path / "anchor2.json"
    anchor2.write_text(json.dumps({"resolved_sampler": {**NEUTRAL, "top_k": -1}, "source": "anchor4096r2 server log (synthetic)"}))
    rec2 = m.run("crescendo", "s1", su, anchor=anchor2, request_id="chatcmpl-3", out_root=tmp_path / "out")
    assert rec2["verdict"] == "AGREES", rec2   # top_k -1 and 0 both mean "off"


def test_a_request_id_isolates_the_probe_from_the_load_checks_greedy_traffic_at_the_same_cap(tmp_path):
    m = _load()
    greedy = [_line(i, seed=str(i), **R._GREEDY) for i in range(3)]          # the load check: temperature 0.0 at the run's cap
    su = _standup(tmp_path, "standup", greedy + [_line(9, **R._NEUTRAL)], policy="engine_defaults")
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({"resolved_sampler": NEUTRAL, "source": "synthetic"}))
    # without the request id the client class is MIXED (greedy + sampled at one cap) -> UNKNOWN, never AGREES
    rec = m.run("misalign", "s2", su, anchor=anchor, client_caps=[CAP], out_root=tmp_path / "o1")
    assert rec["verdict"] == "UNKNOWN" and "distinct resolved tuples" in rec["row_marker"]
    # with it, exactly the probe is read
    rec = m.run("misalign", "s2", su, anchor=anchor, client_caps=[CAP], request_id="chatcmpl-9", out_root=tmp_path / "o2")
    assert rec["verdict"] == "AGREES" and rec["effective"]["n_requests"] == 1


def test_a_missing_anchor_is_unknown_never_assumed_equal(tmp_path):
    m = _load()
    su = _standup(tmp_path, "standup", [_line(1, **R._NEUTRAL)], policy="engine_defaults")
    rec = m.run("petri", "s3", su, anchor=tmp_path / "nowhere.json", request_id="chatcmpl-1", out_root=tmp_path / "out")
    assert rec["verdict"] == "UNKNOWN" and m.EXIT["UNKNOWN"] == 2
    assert any("does not exist" in e for e in rec["errors"])
    assert rec["row_marker"].startswith("non_comparable(unknown:")
    assert json.loads((tmp_path / "out" / "petri_s3.json").read_text())["verdict"] == "UNKNOWN"


def test_a_client_only_anchor_record_without_a_server_policy_is_unknown(tmp_path):
    m = _load()
    su = _standup(tmp_path, "standup", [_line(1, **R._NEUTRAL)], policy="engine_defaults")
    anchor = tmp_path / "bon_served_run.json"
    anchor.write_text(json.dumps({"sampler": {"temperature": 1.0}, "n_samples": 75}))
    rec = m.run("bon", "s1", su, anchor=anchor, request_id="chatcmpl-1", out_root=tmp_path / "out")
    assert rec["verdict"] == "UNKNOWN" and any("CLIENT record" in e for e in rec["errors"])
    # ... and resolves under an explicit policy: contract_override:<model> fills the contracts' row (presence_penalty stays
    # 0.0 — vLLM never lifts it from a generation config), engine_defaults the neutral tuple
    rec_co = m.run("bon", "s1", su, anchor=anchor, anchor_server_policy="contract_override:Qwen/Qwen3.5-4B",
                   request_id="chatcmpl-1", out_root=tmp_path / "o2")
    assert rec_co["anchor"]["sampler"] == {"temperature": 1.0, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                           "presence_penalty": 0.0, "repetition_penalty": 1.0}
    assert rec_co["verdict"] == "DIVERGED"
    rec_ed = m.run("bon", "s1", su, anchor=anchor, anchor_server_policy="engine_defaults", request_id="chatcmpl-1", out_root=tmp_path / "o3")
    assert rec_ed["verdict"] == "AGREES"
    with pytest.raises(m.PreflightError, match="pin no server-default row"):
        m._resolve_sent({}, policy="contract_override:nobody/NoModel")


def test_anchor_kinds_resolved_record_and_vllm_log(tmp_path):
    m = _load()
    import record_resolved_sampler as rrs
    su = _standup(tmp_path, "standup", [_line(1, **R._NEUTRAL)], policy="engine_defaults")
    # the anchor as a record_resolved_sampler record (one distinct client tuple)
    alog = tmp_path / "anchor_vllm_server.log"
    alog.write_text("\n".join([R._STARTED, R._ATTEST_INFO, _line(11, **R._NEUTRAL), _line(12, **R._NEUTRAL)]) + "\n")
    arec = tmp_path / "anchor_resolved_sampler.json"
    assert rrs.main(["--vllm-log", str(alog), "--out", str(arec), "--client-cap", str(CAP)]) == 0
    rec = m.run("petri", "s1", su, anchor=arec, request_id="chatcmpl-1", out_root=tmp_path / "o1")
    assert rec["verdict"] == "AGREES" and rec["anchor"]["kind"] == "resolved_record"
    # the anchor as its raw server log
    rec = m.run("petri", "s1", su, anchor=alog, anchor_client_caps=[CAP], request_id="chatcmpl-1", out_root=tmp_path / "o2")
    assert rec["verdict"] == "AGREES" and rec["anchor"]["kind"] == "vllm_log" and rec["anchor"]["n_requests"] == 2
    # a record whose client class holds two distinct tuples is not one sampler -> UNKNOWN
    mixed = tmp_path / "mixed.log"
    mixed.write_text("\n".join([R._STARTED, R._ATTEST_INFO, _line(11, **R._NEUTRAL), _line(12, **R._VENDOR)]) + "\n")
    mrec = tmp_path / "mixed_resolved.json"
    rrs.main(["--vllm-log", str(mixed), "--out", str(mrec), "--client-cap", str(CAP)])
    rec = m.run("petri", "s1", su, anchor=mrec, request_id="chatcmpl-1", out_root=tmp_path / "o3")
    assert rec["verdict"] == "UNKNOWN" and any("2 distinct" in e for e in rec["errors"])


def test_the_standup_side_refuses_an_unlogged_probe(tmp_path):
    m = _load()
    su = _standup(tmp_path, "standup", [_line(1, **R._NEUTRAL)], policy="engine_defaults")
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({"resolved_sampler": NEUTRAL, "source": "synthetic"}))
    rec = m.run("petri", "s1", su, anchor=anchor, request_id="chatcmpl-404", out_root=tmp_path / "out")
    assert rec["verdict"] == "UNKNOWN" and any("was not logged" in e for e in rec["errors"])
    # no log at all: the standup ran without --enable-log-requests
    bare = tmp_path / "bare"; bare.mkdir()
    rec = m.run("petri", "s1", bare, anchor=anchor, request_id="chatcmpl-1", out_root=tmp_path / "out2")
    assert rec["verdict"] == "UNKNOWN" and any("enable-log-requests" in e for e in rec["errors"])


def test_not_sent_is_not_a_verdict_source(tmp_path):
    """``anchor_pairing.sampler_check(None, None, not_sent=True)`` (relu-w6a's paired_vs_anchor) accepts ABSENT client
    params without comparing engine-resolved values; this tool never calls it, never imports the pairing helpers, and
    reads two unreadable / empty / partially stated sides as UNKNOWN — never AGREES."""
    import ast
    m = _load()
    tree = ast.parse((REPO / "scripts/sdfbase/seedrep/anchor_sampler_preflight.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not [c for c in calls if (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) == "sampler_check"]
    assert not [kw for c in calls for kw in c.keywords if kw.arg == "not_sent"]
    imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
              {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any("anchor_pairing" in i or "paired_vs_anchor" in i for i in imports), imports
    assert m.verdict_of(None, None, error=None)["verdict"] == "UNKNOWN"
    assert m.verdict_of({}, {}, error=None)["verdict"] == "UNKNOWN"            # two EMPTY tuples state nothing
    partial = {**NEUTRAL}; del partial["presence_penalty"]
    v = m.verdict_of(partial, NEUTRAL, error=None)
    assert v["verdict"] == "UNKNOWN" and "presence_penalty" in v["row_marker"]   # a key one side does not state
    assert m.verdict_of(NEUTRAL, {**NEUTRAL, "top_k": -1}, error=None)["verdict"] == "AGREES"   # off is off
    assert set(m.SAMPLER_KEYS) == {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"}
    with pytest.raises(m.PreflightError, match="lacks"):
        m.anchor_sampler(_write(tmp_path / "x.json", {"resolved_sampler": {"temperature": 1.0}}), kind="auto", server_policy=None,
                         request_id=None, client_caps=[], probe_caps=None)


def _write(p, doc):
    p.write_text(json.dumps(doc))
    return p


def test_the_cli_exit_codes_follow_the_verdict(tmp_path):
    import subprocess
    su = _standup(tmp_path, "standup", [_line(5, **R._VENDOR)])
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({"target_decode": {}}))
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    cmd = [sys.executable, str(REPO / "scripts/sdfbase/seedrep/anchor_sampler_preflight.py"), "petri", "s1", str(su),
           "--anchor", str(anchor), "--request-id", "chatcmpl-5", "--out-root", str(tmp_path / "rs")]
    r = subprocess.run(cmd + ["--anchor-server-policy", "engine_defaults"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "DIVERGED" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)   # a client record with no policy: UNKNOWN
    assert r.returncode == 2 and "UNKNOWN" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(cmd + ["--anchor-server-policy", "contract_override:Qwen/Qwen3.5-4B"], capture_output=True, text=True, env=env)
    # the standup resolved presence_penalty 1.5 (the vendor row sent per request); the pin never lifts it -> DIVERGED on that key alone
    assert r.returncode == 1 and "presence_penalty: 1.5 vs anchor 0.0" in r.stdout, r.stdout
