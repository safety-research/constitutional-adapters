"""The shared attacker pool (sisteer.eval.attacks.attacker_pool) and its three
surfaces — CPU-only, no server, no cluster, no Modal.

Covered:
- endpoint.json schema: required keys, routability of the url, identity subset;
- refusal paths, each its own typed error: absent / torn file, stale Modal
  server identity, unreachable /health, an UNGUARDED pool, bearer rejected, model /
  snapshot / window mismatch, an unreadable (world-readable) key file;
- the bearer: env wins over the key file; write_api_key holds 0600;
- sizing: the Qwen3.6-27B memory model's literals, the identity fit at the
  measured window, the refusal of an S that does not fit (naming the largest
  that would), chunked prefill buying more slots, the log coverage attestation;
- the client: the bearer rides the session header; a 401 is typed and never
  retried; the once-only max_tokens clamp on a context-length 400 and the
  typed AttackerPromptOverWindow when the prompt no longer fits;
- jailbreak_run.py plumbing: --attacker-pool parses; it rewrites the engine /
  endpoint / model, refuses a conflicting engine, refuses (SystemExit) BEFORE
  the target loads, threads the key into the vLLM client, and the manifest
  records the pool identity; the flag is a no-op when absent;
- grokmt_served.py plumbing: _driver_cmd's pool branch, _attacker_for_run
  (pool -> no local attacker spawned; None -> spawned), the provenance record,
  the pooled function's signature, the 2-GPU default untouched;
- the Modal Server: fixed one-H200 geometry, unconditional live coverage gate,
  bearer, endpoint identity, and a canary that refuses same-container evidence;
- the two sbatch files: retained but impossible to launch;
- attacker_pool_profile.py: the sizing rule on synthetic rows;
- the measured literals match the in-repo doc.
Synthetic data only: placeholder hosts (192.0.2.10), invented job ids.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.eval.attacks import attacker_pool as ap
from sisteer.eval.attacks import vllm_attacker as va
from sisteer.eval.attacks.pyrit_adapters import ATTACKER_MODEL, ATTACKER_SNAPSHOT
from sisteer.capeval.serving import vllm_auth

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import jailbreak_run  # noqa: E402

URL = "http://192.0.2.10:8000"
MODAL_URL = "https://test--attacker-pool.modal.run"
MODAL_LIVENESS = {
    "kind": "modal", "app_name": "attacker-pool-qwen36-27b",
    "app_id": "ap-test", "server_name": "qwen36-27b-abliterated-vllm",
    "server_id": "sv-test", "call_id": "fc-test", "container_id": "ta-pool",
}


def _attested_block(*, quantization="none", prefix=True) -> dict:
    """An engine-attested ``serving`` block, the way a pool's standup writes it:
    a shared pool's record MUST carry one (a consumer refuses a record without)."""
    from sisteer.eval.attacks import attacker_serving as serving
    opts = serving.AttackerServingOptions(quantization, prefix)
    log = ("INFO Initializing a V1 LLM engine (v0.25.1) with config: model='m', dtype=torch.bfloat16, "
           f"max_seq_len=19200, quantization={'None' if quantization == 'none' else quantization}, "
           f"kv_cache_dtype=auto, enable_prefix_caching={prefix}\n"
           + ("INFO Mamba cache mode is set to 'align'\n" if prefix else ""))
    engine = serving.parse_engine_log(log)
    return serving.attested_serving_record(
        {"schema": serving.MANIFEST_SCHEMA, "requested": opts.as_dict(), "engine": engine,
         "engine_verified": True}, manifest_path="/pool/serving.json")


def _ep_dict(**over) -> dict:
    d = {"url": URL, "model": ATTACKER_MODEL, "snapshot": ATTACKER_SNAPSHOT,
         "max_model_len": 19200, "max_num_seqs": 37, "max_num_batched_tokens": 19200 * 37,
         "api_key_env_name": "VLLM_API_KEY", "job_id": "424242", "node": "compute-250",
         "started_at": "2026-09-04T00:00:00Z", "pool_name": "t", "prefix_caching": True,
         "gpu_memory_utilization": 0.9, "vllm_version": "0.25.1",
         "attacker_quantization": "none", "serving": _attested_block()}
    d.update(over)
    return d


def _modal_ep_dict(**over) -> dict:
    return _ep_dict(url=MODAL_URL, job_id="fc-test", node="ta-pool",
                    liveness=dict(MODAL_LIVENESS), runtime="modal_server", **over)


def _write_pool(tmp_path: Path, key: str = "k" * 43, **over) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    d = _ep_dict(pool_dir=str(pool), api_key_file=str(pool / "api_key"), **over)
    (pool / "endpoint.json").write_text(json.dumps(d))
    ap.write_api_key(pool / "api_key", key)
    return pool


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")


class _PoolSession:
    """A fake pool: /health, /v1/models with the guard, /v1/chat/completions."""

    def __init__(self, *, key="k" * 43, model=ATTACKER_MODEL, root=None, max_model_len=19200,
                 health=200, guard=True, chat_status=200, chat_body=None, chat_text=""):
        self.key, self.model, self.health, self.guard = key, model, health, guard
        self.root = root if root is not None else f"/cache/snapshots/{ATTACKER_SNAPSHOT}"
        self.L = max_model_len
        self.chat_status, self.chat_body, self.chat_text = chat_status, chat_body, chat_text
        self.gets, self.posts = [], []
        self.headers = {}

    def _auth(self, headers):
        h = dict(self.headers)
        h.update({k: v for k, v in (headers or {}).items() if v is not None})
        for k in list(h):
            if (headers or {}).get(k, "x") is None:
                h.pop(k, None)
        return h.get("Authorization") == f"Bearer {self.key}"

    def get(self, url, headers=None, timeout=None):
        self.gets.append((url, headers))
        if url.endswith("/health"):
            return _Resp(self.health)
        if url.endswith("/v1/models"):
            if self.guard and not self._auth(headers):
                return _Resp(401, {"error": "Unauthorized"})
            return _Resp(200, {"object": "list", "data": [
                {"id": self.model, "object": "model", "root": self.root, "max_model_len": self.L}]})
        return _Resp(404)

    def post(self, url, json=None, timeout=None, headers=None):
        self.posts.append({"url": url, "json": json})
        if url.endswith("/tokenize"):
            return _Resp(200, {"count": 18_900, "tokens": []})
        if self.guard and not self._auth(headers):
            return _Resp(401, {"error": "Unauthorized"})
        if self.chat_status != 200:
            return _Resp(self.chat_status, self.chat_body, self.chat_text)
        return _Resp(200, {"choices": [{"message": {"content": "reply"}}]})


def _squeue(state):
    def runner(cmd, **kw):
        assert cmd[:3] == ["squeue", "-h", "-j"]
        return SimpleNamespace(stdout=(state + "\n") if state else "", returncode=0)
    return runner


def _modal_lookup(*, server_id="sv-test", url=MODAL_URL, seen=None):
    def lookup(app_name, server_name):
        if seen is not None:
            seen.append((app_name, server_name))
        class Server:
            object_id = server_id
            def hydrate(self):
                return self
            def get_url(self):
                return url
        return Server()
    return lookup


def _modal_app_lookup(*, app_id="ap-test", seen=None):
    def lookup(app_name):
        if seen is not None:
            seen.append(app_name)
        return SimpleNamespace(app_id=app_id)
    return lookup


def _modal_call_live(*, live=True, seen=None):
    def check(call_id):
        if seen is not None:
            seen.append(call_id)
        return live
    return check


# --------------------------------------------------------------------------- #
# schema                                                                       #
# --------------------------------------------------------------------------- #

def test_constants_agree_with_the_served_lane_and_the_adapters():
    assert ap.API_KEY_ENV == vllm_auth.TOKEN_ENV == "VLLM_API_KEY"
    assert (ap.ATTACKER_MODEL, ap.ATTACKER_SNAPSHOT) == (ATTACKER_MODEL, ATTACKER_SNAPSHOT)


def test_endpoint_from_dict_and_identity():
    ep = ap.PoolEndpoint.from_dict(_ep_dict())
    assert ep.url == URL and ep.job_id == "424242" and ep.max_num_batched_tokens == 19200 * 37
    assert ep.extra["vllm_version"] == "0.25.1"
    ident = ep.identity()
    assert set(ident) >= {"model", "snapshot", "max_model_len", "max_num_seqs", "url",
                          "job_id", "node", "started_at", "prefix_caching"}
    assert ident["snapshot"] == ATTACKER_SNAPSHOT and "api_key" not in json.dumps(ident)
    modal_ident = ap.PoolEndpoint.from_dict(_modal_ep_dict()).identity()
    assert modal_ident["liveness"]["server_id"] == "sv-test"
    back = ap.PoolEndpoint.from_dict(ep.as_dict())
    assert back == ep


@pytest.mark.parametrize("bad", [
    {"url": None}, {"job_id": ""}, {"max_model_len": 0}, {"max_num_seqs": True},
    {"url": "http://127.0.0.1:8000"}, {"url": "http://172.19.0.2:8000"}, {"url": "192.0.2.9:8000"},
])
def test_endpoint_refuses_missing_or_unroutable(bad):
    d = _ep_dict()
    d.update(bad)
    with pytest.raises(ap.PoolAbsent):
        ap.PoolEndpoint.from_dict(d)
    d2 = _ep_dict()
    del d2["snapshot"]
    with pytest.raises(ap.PoolAbsent):
        ap.PoolEndpoint.from_dict(d2)


def test_read_endpoint_absent_and_torn(tmp_path):
    with pytest.raises(ap.PoolAbsent) as ei:
        ap.read_endpoint(tmp_path / "nope")
    assert "scripts/modal/attacker_pool.py" in str(ei.value) and "NOT falling back" in str(ei.value)
    p = tmp_path / "endpoint.json"
    p.write_text('{"url": "http://192.0.2.10:8000", "model": ')
    with pytest.raises(ap.PoolAbsent):
        ap.read_endpoint(p)
    pool = _write_pool(tmp_path)
    ep = ap.read_endpoint(pool)
    assert ep.pool_dir == str(pool) and ap.endpoint_path(pool) == pool / "endpoint.json"


def test_bare_url_is_refused_instead_of_fabricating_geometry():
    for url in ("http://192.0.2.10:8000/", "https://test--pool.modal.run"):
        with pytest.raises(ap.PoolAbsent) as ei:
            ap.resolve_pool(url)
        assert "endpoint.json" in str(ei.value) and "URL-only" in str(ei.value)
    with pytest.raises(ap.PoolAbsent):
        ap.resolve_pool("http://127.0.0.1:8000")


# --------------------------------------------------------------------------- #
# liveness + bearer                                                            #
# --------------------------------------------------------------------------- #

def test_job_liveness_refusals():
    ep = ap.PoolEndpoint.from_dict(_ep_dict())
    assert ap.check_job_live(ep, runner=_squeue("RUNNING")) == "RUNNING"
    for state in ("", "COMPLETING", "PENDING"):
        with pytest.raises(ap.PoolStale) as ei:
            ap.check_job_live(ep, runner=_squeue(state))
        assert "424242" in str(ei.value)

    def boom(cmd, **kw):
        raise FileNotFoundError("squeue")
    with pytest.raises(ap.PoolStale):
        ap.check_job_live(ep, runner=boom)


def test_modal_liveness_pins_app_call_server_object_and_url():
    ep = ap.PoolEndpoint.from_dict(_modal_ep_dict())
    server_seen, app_seen, call_seen = [], [], []
    live = ap.check_modal_live(
        ep, lookup=_modal_lookup(seen=server_seen),
        app_lookup=_modal_app_lookup(seen=app_seen),
        call_live=_modal_call_live(seen=call_seen))
    assert live["state"] == "deployed" and live["server_id"] == "sv-test"
    assert live["app_id"] == "ap-test" and live["call_state"] == "running"
    assert server_seen == [(MODAL_LIVENESS["app_name"], MODAL_LIVENESS["server_name"])]
    assert app_seen == [MODAL_LIVENESS["app_name"]]
    assert call_seen == [MODAL_LIVENESS["call_id"]]
    with pytest.raises(ap.PoolStale):
        ap.check_modal_live(ep, lookup=_modal_lookup(server_id="sv-new"))
    with pytest.raises(ap.PoolStale):
        ap.check_modal_live(ep, lookup=_modal_lookup(url="https://new.modal.run"))
    with pytest.raises(ap.PoolStale, match="now has id"):
        ap.check_modal_live(
            ep, lookup=_modal_lookup(), app_lookup=_modal_app_lookup(app_id="ap-new"))
    with pytest.raises(ap.PoolStale, match="has completed"):
        ap.check_modal_live(
            ep, lookup=_modal_lookup(), call_live=_modal_call_live(live=False))
    bad = ap.PoolEndpoint.from_dict(_ep_dict(liveness={"kind": "modal"}))
    with pytest.raises(ap.PoolStale):
        ap.check_modal_live(bad, lookup=_modal_lookup())


def test_api_key_env_wins_then_private_file_then_refusal(tmp_path):
    pool = _write_pool(tmp_path, key="filekey")
    ep = ap.read_endpoint(pool)
    assert ap.load_api_key(ep, env={"VLLM_API_KEY": "envkey"}) == "envkey"
    assert ap.load_api_key(ep, env={}) == "filekey"
    os.chmod(pool / "api_key", 0o644)
    with pytest.raises(ap.PoolKeyUnreadable) as ei:
        ap.load_api_key(ep, env={})
    assert "644" in str(ei.value)
    (pool / "api_key").unlink()
    with pytest.raises(ap.PoolKeyUnreadable):
        ap.load_api_key(ep, env={})
    assert ap.load_api_key(ep, env={}, require=False) is None


def test_write_api_key_is_private_and_never_returned_in_cli_output(tmp_path, capsys):
    p = ap.write_api_key(tmp_path / "d" / "api_key", "s3cret-token")
    assert oct(p.stat().st_mode & 0o777) == "0o600" and p.read_text() == "s3cret-token"
    rc = ap.main(["mint", "--out", str(tmp_path / "k2")])
    out = capsys.readouterr().out
    assert rc == 0 and out.strip() == str(tmp_path / "k2")
    key = (tmp_path / "k2").read_text()
    assert len(key) >= 40 and key not in out


# --------------------------------------------------------------------------- #
# verify_pool: every refusal typed                                             #
# --------------------------------------------------------------------------- #

def test_verify_ok_evidence():
    ep = ap.PoolEndpoint.from_dict(_ep_dict())
    s = _PoolSession()
    ev = ap.verify_pool(ep, "k" * 43, session=s)
    assert ev["ok"] and ev["health_status"] == 200
    assert ev["unauthenticated_models_status"] == 401 and ev["authenticated_models_status"] == 200
    assert ev["models"] == [ATTACKER_MODEL] and ATTACKER_SNAPSHOT in ev["root"]
    assert ev["max_model_len"] == 19200
    # three GETs, the first two without our bearer
    assert [u.split("/", 3)[-1] for u, _ in s.gets] == ["health", "v1/models", "v1/models"]


def test_verify_unreachable_and_unguarded_and_unauthorized():
    ep = ap.PoolEndpoint.from_dict(_ep_dict())
    with pytest.raises(ap.PoolUnreachable):
        ap.verify_pool(ep, "k" * 43, session=_PoolSession(health=503))

    class _Dead:
        def get(self, *a, **k):
            raise ConnectionError("refused")
    with pytest.raises(ap.PoolUnreachable):
        ap.verify_pool(ep, "k" * 43, session=_Dead())
    with pytest.raises(ap.PoolUnguarded):
        ap.verify_pool(ep, "k" * 43, session=_PoolSession(guard=False))
    with pytest.raises(ap.PoolUnauthorized) as ei:
        ap.verify_pool(ep, "wrong-key", session=_PoolSession())
    assert "VLLM_API_KEY" in str(ei.value)


def test_verify_mismatches():
    ep = ap.PoolEndpoint.from_dict(_ep_dict())
    with pytest.raises(ap.PoolMismatch) as ei:
        ap.verify_pool(ep, "k" * 43, session=_PoolSession(model="some/other-model"))
    assert ATTACKER_MODEL in str(ei.value)
    with pytest.raises(ap.PoolMismatch) as ei:
        ap.verify_pool(ep, "k" * 43, session=_PoolSession(root="/cache/snapshots/" + "0" * 40))
    assert ATTACKER_SNAPSHOT in str(ei.value)
    with pytest.raises(ap.PoolMismatch) as ei:
        ap.verify_pool(ep, "k" * 43, session=_PoolSession(max_model_len=49152))
    assert "49152" in str(ei.value) and "19200" in str(ei.value)
    with pytest.raises(ap.PoolMismatch) as ei:
        ap.verify_pool(ep, "k" * 43, session=_PoolSession(max_model_len=None))
    assert "omits max_model_len" in str(ei.value)
    # an expected model the run names overrides the endpoint's
    with pytest.raises(ap.PoolMismatch):
        ap.verify_pool(ep, "k" * 43, session=_PoolSession(), expect_model="x/y")


def test_connect_end_to_end_defaults_slurm_off_and_checks_modal(tmp_path):
    pool = _write_pool(tmp_path, **_modal_ep_dict())
    def no_squeue(*a, **k):
        raise AssertionError("squeue must not run for a copied Modal endpoint")
    ep, key, ev = ap.connect(
        str(pool), session=_PoolSession(), runner=no_squeue, env={},
        modal_lookup=_modal_lookup())
    assert ep.url == MODAL_URL and key == "k" * 43 and ev["job_state"] == "modal:deployed"
    assert ev["liveness"]["server_id"] == "sv-test" and ev["ok"]
    with pytest.raises(ap.PoolStale):
        ap.connect(str(pool), session=_PoolSession(), env={},
                   modal_lookup=_modal_lookup(server_id="sv-new"))
    # Legacy Slurm is now explicit opt-in, retained only for old records.
    legacy = _write_pool(tmp_path / "legacy")
    _, _, old = ap.connect(str(legacy), session=_PoolSession(), squeue=True,
                           runner=_squeue("RUNNING"), env={})
    assert old["job_state"] == "RUNNING"
    assert "k" * 43 not in json.dumps(ev)


# --------------------------------------------------------------------------- #
# sizing                                                                       #
# --------------------------------------------------------------------------- #

def test_memory_model_literals():
    a = ap.QWEN36_27B
    assert a.n_full_attention == 16 and a.n_linear == 48
    assert a.kv_bytes_per_token == 2 * 16 * 4 * 256 * 2 == 65_536
    rec = 48 * 128 * 128 * 4
    conv = 3 * (2 * 16 * 128 + 48 * 128) * 2
    assert a.state_bytes_per_slot == 48 * (rec + conv) == 153_944_064
    assert a.weights_bytes == 55_562_855_904
    # the profile-activation coefficient is the MEASURED 2026-09-04 OOM, not a
    # single MLP intermediate: 137.04 GiB held (51.08 of it weights) when a
    # further 23.04 GiB failed, at 710,400 batched tokens
    assert ap.PROFILE_ACTIVATION_BYTES_PER_TOKEN == int((137.04 - 51.08 + 23.04) * ap.GIB / 710_400)
    assert a.profile_activation_bytes(710_400, 37) >= 109 * ap.GIB
    # ... and it is ~4.7x the term the first model used (one MLP intermediate)
    assert ap.PROFILE_ACTIVATION_BYTES_PER_TOKEN > 4 * a.intermediate_size * a.kv_dtype_bytes


def test_pool_geometry_chunked_fits_and_the_identity_is_refused():
    # the launched geometry: chunked prefill at a 16,384-token batch, S = 37 at L = 19,200
    g = ap.resolve_pool_geometry(19200, 37, chunked_prefill=True, chunk_tokens=16_384)
    assert (g.max_model_len, g.max_num_seqs, g.max_num_batched_tokens) == (19200, 37, 16_384)
    assert g.fits and g.chunked_prefill and g.seqs_source == "explicit"
    assert g.kv_bytes == 37 * 19200 * ap.QWEN36_27B.kv_bytes_per_token     # KV covers S x L
    assert g.headroom_bytes > 10 * ap.GIB                                  # not a razor-thin fit
    assert g.max_seqs_fitting >= 37
    # the largest chunked S the model allows is derived, never pinned
    kv_fit = ap.resolve_pool_geometry(19200, chunked_prefill=True, chunk_tokens=16_384)
    assert kv_fit.seqs_source == "kv_fit" and kv_fit.max_num_seqs == g.max_seqs_fitting >= 37
    # the no-chunk identity at the same S x L does NOT fit: the measured profile
    # peak (>= 109 GiB at 710,400 tokens) dwarfs the KV it would protect
    ident = ap.fit(19200, 37)
    assert not ident.fits and ident.max_num_batched_tokens == 710_400
    assert ident.activation_bytes >= 109 * ap.GIB and ident.headroom_bytes < 0
    assert 0 < ident.max_seqs_fitting < 37
    with pytest.raises(ap.PoolGeometryUnfit) as ei:
        ap.resolve_pool_geometry(19200, 37)
    assert f"max_num_seqs <= {ident.max_seqs_fitting}" in str(ei.value)
    # a 128-slot pool is refused in both modes; the ceiling too
    for chunked in (False, True):
        with pytest.raises(ap.PoolGeometryUnfit):
            ap.resolve_pool_geometry(19200, 128, chunked_prefill=chunked)
    with pytest.raises(ap.PoolGeometryUnfit):
        ap.resolve_pool_geometry(19200, ap.MAX_SEQS_CEILING + 1)
    # the other measured windows, chunked: 37 fits at Crescendo's and the
    # cover-max window; not at the per-job 49,152
    assert ap.fit(ap.CRESCENDO_MAX_MODEL_LEN, 37, chunked_prefill=True).fits
    assert ap.fit(ap.COVER_MAX_MAX_MODEL_LEN, 37, chunked_prefill=True).fits
    assert not ap.fit(49152, 37, chunked_prefill=True).fits
    with pytest.raises(ValueError):
        ap.fit(19201, 1)                            # not a block multiple
    d = g.as_dict()
    assert d["kv_gib"] > 40 and "FITS" in g.describe() and "chunked prefill" in g.describe()


def test_size_cli_sh_output(capsys):
    assert ap.main(["size", "--chunked-prefill", "--max-num-seqs", "37", "--sh"]) == 0
    out = capsys.readouterr().out
    assert "POOL_MAX_MODEL_LEN=19200\nPOOL_MAX_NUM_SEQS=37\n" in out
    assert "POOL_MAX_NUM_BATCHED_TOKENS=16384" in out and "POOL_GEOMETRY_JSON='{" in out
    # the bare identity default no longer reaches 37 at this window
    assert ap.main(["size", "--sh"]) == 0
    out = capsys.readouterr().out
    s = int(re.search(r"POOL_MAX_NUM_SEQS=(\d+)", out).group(1))
    assert 0 < s < 37 and f"POOL_MAX_NUM_BATCHED_TOKENS={s * 19200}" in out
    assert ap.main(["size", "--max-num-seqs", "128", "--sh"]) == 3
    assert "REFUSED" in capsys.readouterr().err


def test_coverage_and_logstats():
    log = ("INFO 09-04 10:00:00 [kv_cache_utils.py:1] GPU KV cache size: 1,234,560 tokens\n"
           "INFO 09-04 10:00:00 [kv_cache_utils.py:2] Maximum concurrency for 19,200 tokens per request: 37.20x\n"
           "INFO 09-04 10:05:00 [loggers.py:1] Engine 000: Avg prompt throughput: 812.0 tokens/s, "
           "Avg generation throughput: 210.5 tokens/s, Running: 12 reqs, Waiting: 0 reqs, "
           "GPU KV cache usage: 31.0%, Prefix cache hit rate: 78.5%\n"
           "INFO 09-04 10:15:00 [loggers.py:1] Engine 000: Avg prompt throughput: 0.0 tokens/s, "
           "Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, "
           "GPU KV cache usage: 0.0%, Prefix cache hit rate: 78.5%\n")
    cov = ap.kv_coverage_from_log(log)
    assert cov == {"kv_tokens_reported": 1_234_560, "max_concurrency_reported": 37.2,
                   "concurrency_window": 19200}
    assert ap.coverage_problem(cov, max_num_seqs=37, max_model_len=19200) is None
    assert "below the launched" in ap.coverage_problem(cov, max_num_seqs=40, max_model_len=19200)
    assert "not the launched window" in ap.coverage_problem(cov, max_num_seqs=37, max_model_len=20480)
    assert "unattested" in ap.coverage_problem({}, max_num_seqs=37, max_model_len=19200)
    st = ap.log_stats(log)
    assert st["n_stats_lines"] == 2 and st["running_max"] == 12 and st["running_mean"] == 6.0
    assert st["running_mean_while_active"] == 12.0 and st["gen_tps_max"] == 210.5
    assert st["prefix_cache_hit_rate_last"] == 78.5
    assert ap.log_stats(log, since="09-04 10:10:00")["n_stats_lines"] == 1
    assert ap.log_stats(log, until="09-04 09:00:00") == {"n_stats_lines": 0}


def test_routable_refusals():
    assert ap.routable_problem("192.0.2.9") is None
    for ip in ("127.0.0.1", "172.19.0.2", "172.31.255.1", "0.0.0.0", ""):
        assert "hostname -I" in ap.routable_problem(ip)
    assert ap.main(["routable", "172.19.0.2"]) == 3


def test_endpoint_cli_writes_atomically_and_live_job(tmp_path, monkeypatch):
    pool = tmp_path / "p"
    pool.mkdir()
    rc = ap.main(["endpoint", "--pool-name", "p", "--pool-dir", str(pool), "--url", URL,
                  "--model", ATTACKER_MODEL, "--snapshot", ATTACKER_SNAPSHOT,
                  "--api-key-file", str(pool / "api_key"), "--job-id", "7", "--node", "compute-250",
                  "--node-ip", "192.0.2.10", "--max-model-len", "19200", "--max-num-seqs", "37",
                  "--max-num-batched-tokens", "710400", "--extra", "chunked_prefill=0",
                  "--coverage-json", json.dumps({"max_concurrency_reported": 37.2})])
    assert rc == 0
    ep = ap.read_endpoint(pool)
    assert ep.job_id == "7" and ep.extra["chunked_prefill"] == "0"
    assert ep.extra["kv_coverage"]["max_concurrency_reported"] == 37.2
    assert not (pool / "endpoint.json.tmp").exists()
    # live-job: 1 = no file, 0 = RUNNING, 2 = stale
    assert ap.main(["live-job", "--pool", str(tmp_path / "none")]) == 1
    monkeypatch.setattr(ap.subprocess, "run", _squeue("RUNNING"))
    assert ap.main(["live-job", "--pool", str(pool)]) == 0
    monkeypatch.setattr(ap.subprocess, "run", _squeue(""))
    assert ap.main(["live-job", "--pool", str(pool)]) == 2
    # a bad record (loopback url) never lands on disk
    with pytest.raises(ap.PoolAbsent):
        ap.main(["endpoint", "--pool-name", "p", "--pool-dir", str(pool), "--url", "http://127.0.0.1:1",
                 "--model", "m", "--snapshot", "s", "--api-key-file", "k", "--job-id", "8",
                 "--node", "n", "--node-ip", "127.0.0.1", "--max-model-len", "16", "--max-num-seqs", "1",
                 "--max-num-batched-tokens", "16"])
    assert ap.read_endpoint(pool).job_id == "7"


# --------------------------------------------------------------------------- #
# the client: bearer, 401, the once-only clamp                                 #
# --------------------------------------------------------------------------- #

def test_client_sends_bearer_and_types_a_401():
    s = _PoolSession()
    c = va.VLLMAttackerClient(URL, ATTACKER_MODEL, session=s, api_key="k" * 43, max_retries=2)
    assert s.headers["Authorization"] == "Bearer " + "k" * 43 and c.api_key_set
    assert c.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9,
                  max_new_tokens=16) == "reply"
    bad = _PoolSession()
    c2 = va.VLLMAttackerClient(URL, ATTACKER_MODEL, session=bad, api_key="nope", max_retries=2,
                               backoff=0)
    with pytest.raises(va.VLLMEndpointUnauthorized) as ei:
        c2.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=16)
    assert len(bad.posts) == 1                         # never retried
    assert isinstance(ei.value, va.VLLMEndpointUnreachable) and "attacker-pool" in str(ei.value)
    # no key -> no header, as before
    plain = _PoolSession(guard=False)
    va.VLLMAttackerClient(URL, ATTACKER_MODEL, session=plain)
    assert "Authorization" not in plain.headers


# the pinned vLLM 0.25.1 rejection texts served_window parses (its regexes are the contract)
_OVERFLOW = ("This model's maximum context length is 19200 tokens. However, you requested 1024 output "
             "tokens and the prompt contains 18676 input tokens, for a total of 19700 tokens. Please "
             "reduce the length of the input messages or completion.")


def test_client_clamps_once_on_a_context_400_then_serves():
    class _Sess(_PoolSession):
        def post(self, url, json=None, timeout=None, headers=None):
            self.posts.append({"url": url, "json": json})
            if json.get("max_tokens") == 1024:
                return _Resp(400, {"error": {"message": _OVERFLOW}})
            return _Resp(200, {"choices": [{"message": {"content": "clamped reply"}}]})
    s = _Sess(guard=False)
    c = va.VLLMAttackerClient(URL, ATTACKER_MODEL, session=s, max_retries=2, backoff=0)
    out = c.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=1024)
    assert out == "clamped reply" and len(s.posts) == 2
    assert s.posts[1]["json"]["max_tokens"] == 19200 - 18676 == 524
    assert c.n_clamped == 1 and c.last_meta["clamped_max_tokens"] == 524
    assert c.last_meta["prompt_tokens"] == 18676 and c.last_meta["window"] == 19200


def test_client_prompt_over_window_is_typed_not_retried():
    over = ("This model's maximum context length is 19200 tokens. However, you requested 1024 output "
            "tokens and the prompt contains 19200 input tokens, for a total of 20224 tokens.")

    class _Sess(_PoolSession):
        def post(self, url, json=None, timeout=None, headers=None):
            self.posts.append({"url": url, "json": json})
            return _Resp(400, {"error": {"message": over}})
    s = _Sess(guard=False)
    c = va.VLLMAttackerClient(URL, ATTACKER_MODEL, session=s, max_retries=2, backoff=0)
    with pytest.raises(va.AttackerPromptOverWindow) as ei:
        c.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=1024)
    assert len(s.posts) == 1 and c.n_prompt_over_window == 1
    assert isinstance(ei.value, va.VLLMEndpointUnreachable) and "19200" in str(ei.value)
    # a floor-only rejection resolves the exact prompt through /tokenize
    floor = "prompt contains at least 18177 input tokens, longer than the maximum model length of 19200"

    class _Floor(_PoolSession):
        def post(self, url, json=None, timeout=None, headers=None):
            self.posts.append({"url": url, "json": json})
            if url.endswith("/tokenize"):
                return _Resp(200, {"count": 18_900})
            if json.get("max_tokens") == 1024:
                return _Resp(400, {"error": {"message": "returned 400: " + floor}})
            return _Resp(200, {"choices": [{"message": {"content": "ok"}}]})
    s2 = _Floor(guard=False)
    c2 = va.VLLMAttackerClient(URL, ATTACKER_MODEL, session=s2, max_retries=2, backoff=0)
    assert c2.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9,
                   max_new_tokens=1024) == "ok"
    assert [p["url"].rsplit("/", 1)[-1] for p in s2.posts] == ["completions", "tokenize", "completions"]
    assert s2.posts[-1]["json"]["max_tokens"] == 19200 - 18900
    assert c2.last_meta["prompt_tokens_source"] == "tokenize_endpoint"
    # a non-context 400 stays the generic (retried) failure
    s3 = _PoolSession(guard=False, chat_status=400, chat_body={"error": {"message": "bad param"}})
    c3 = va.VLLMAttackerClient(URL, ATTACKER_MODEL, session=s3, max_retries=1, backoff=0)
    with pytest.raises(va.VLLMEndpointUnreachable) as ei:
        c3.chat([{"role": "user", "content": "x"}], temperature=1.0, top_p=0.9, max_new_tokens=16)
    assert not isinstance(ei.value, va.AttackerPromptOverWindow) and len(s3.posts) == 2


# --------------------------------------------------------------------------- #
# jailbreak_run.py plumbing                                                    #
# --------------------------------------------------------------------------- #

def test_attacker_pool_flag_parses_and_defaults_off():
    p = argparse.ArgumentParser()
    jailbreak_run._add_attacker_engine_args(p)
    ns = p.parse_args([])
    assert ns.attacker_pool is None and ns.attacker_engine == "vllm"
    ns = p.parse_args(["--attacker-pool", "/x/pool"])
    assert ns.attacker_pool == "/x/pool"
    # no pool asked for -> the resolver is a no-op and touches nothing
    args = SimpleNamespace(attacker_pool=None, attacker_engine="hf", attacker_endpoint="http://x")
    assert jailbreak_run._resolve_attacker_pool(args, {"attacker": {}}) is None
    assert args.attacker_engine == "hf" and not hasattr(args, "_attacker_pool")


def _fake_connect(tmp_path, **kw):
    pool = _write_pool(tmp_path, **_modal_ep_dict())

    def connect(spec, squeue=True, expect_model=None, expect_snapshot=None):
        assert squeue is False
        return ap.connect(spec, session=_PoolSession(), squeue=squeue, env={},
                          modal_lookup=_modal_lookup(), expect_model=expect_model,
                          expect_snapshot=expect_snapshot)
    return pool, connect


def test_resolve_attacker_pool_rewrites_args_and_records(tmp_path, capsys):
    pool, connect = _fake_connect(tmp_path)
    cfg = {"attacker": {"model": ATTACKER_MODEL, "snapshot": ATTACKER_SNAPSHOT}}
    args = SimpleNamespace(attacker_pool=str(pool), attacker_engine="vllm",
                           attacker_endpoint="http://127.0.0.1:8000", attacker_model=None)
    rec = jailbreak_run._resolve_attacker_pool(args, cfg, connect=connect)
    assert args.attacker_engine == "vllm" and args.attacker_endpoint == MODAL_URL
    assert args.attacker_model == ATTACKER_MODEL and args._attacker_pool_key == "k" * 43
    assert rec["identity"]["job_id"] == "fc-test" and rec["evidence"]["ok"]
    assert "k" * 43 not in capsys.readouterr().out          # the token is never printed
    # the manifest records the pool (and never the key)
    jailbreak_run._write_engine_manifest(tmp_path / "m", cfg, args, "crescendo")
    m = json.loads((tmp_path / "m" / "attacker_engine.json").read_text())
    assert m["attacker_endpoint"] == MODAL_URL and m["attacker_pool"]["identity"]["snapshot"] == ATTACKER_SNAPSHOT
    assert m["attacker_pool"]["evidence"]["unauthenticated_models_status"] == 401
    assert "k" * 43 not in json.dumps(m)
    # without a pool the manifest field is None (an existing per-job run)
    jailbreak_run._write_engine_manifest(tmp_path / "n", cfg,
                                         SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://x"),
                                         "pair")
    assert json.loads((tmp_path / "n" / "attacker_engine.json").read_text())["attacker_pool"] is None


def test_resolve_attacker_pool_refusals(tmp_path):
    pool, connect = _fake_connect(tmp_path)
    cfg = {"attacker": {"model": ATTACKER_MODEL, "snapshot": ATTACKER_SNAPSHOT}}
    # conflicting engine
    args = SimpleNamespace(attacker_pool=str(pool), attacker_engine="hf", attacker_endpoint="x")
    with pytest.raises(SystemExit) as ei:
        jailbreak_run._resolve_attacker_pool(args, cfg, connect=connect)
    assert "pool IS a vLLM attacker" in str(ei.value)
    # a stale pool: typed refusal -> SystemExit naming the type, no fallback
    def stale(spec, squeue=True, expect_model=None, expect_snapshot=None):
        assert squeue is False
        return ap.connect(spec, session=_PoolSession(), env={},
                          modal_lookup=_modal_lookup(server_id="sv-new"))
    args = SimpleNamespace(attacker_pool=str(pool), attacker_engine="vllm", attacker_endpoint="x",
                           attacker_model=None)
    with pytest.raises(SystemExit) as ei:
        jailbreak_run._resolve_attacker_pool(args, cfg, connect=stale)
    assert "PoolStale" in str(ei.value) and "NOT falling back" in str(ei.value)
    assert args.attacker_endpoint == "x"                    # untouched on refusal
    # the config's snapshot is what the run expects: a pool at another snapshot refuses
    def other_snapshot(spec, squeue=True, expect_model=None, expect_snapshot=None):
        assert squeue is False
        return ap.connect(spec, session=_PoolSession(root="/c/" + "0" * 40), env={},
                          modal_lookup=_modal_lookup(), expect_model=expect_model,
                          expect_snapshot=expect_snapshot)
    with pytest.raises(SystemExit) as ei:
        jailbreak_run._resolve_attacker_pool(args, cfg, connect=other_snapshot)
    assert "PoolMismatch" in str(ei.value)


def test_cmd_generate_refuses_pool_before_any_target_load(monkeypatch, tmp_path):
    import sisteer.models as models_mod

    def boom(*a, **k):
        raise AssertionError("load_model_and_tokenizer must not run when the pool refuses")
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", boom)
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    calls = []

    def refusing_connect(spec, squeue=True, expect_model=None, expect_snapshot=None):
        calls.append((spec, squeue))
        raise ap.PoolAbsent("no pool at " + spec)
    monkeypatch.setattr(ap, "connect", refusing_connect)
    cfg = {"attacker": {"model": ATTACKER_MODEL, "snapshot": ATTACKER_SNAPSHOT}, "model": "m",
           "dtype": "bfloat16", "attacks": {}, "private_run": "r"}
    args = SimpleNamespace(attack="crescendo", defense="none", attacker_pool="/pool/x",
                           attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000",
                           attacker_model=None, target_engine="hf", xstest=False, confirmatory=False,
                           shard=0, nshards=1, private_root=str(tmp_path), positions=None, scale=1.0)
    with pytest.raises(SystemExit) as ei:
        jailbreak_run.cmd_generate(args, cfg)
    assert calls == [("/pool/x", False)] and "PoolAbsent" in str(ei.value)


def test_build_seams_threads_the_pool_key_into_the_vllm_client(monkeypatch):
    made = []

    class _Cap(va.VLLMAttackerClient):
        def __init__(self, endpoint, model, **kw):
            made.append((endpoint, model, kw.get("api_key")))
            super().__init__(endpoint, model, session=_PoolSession(guard=False), **kw)

        def check_health(self):
            pass
    monkeypatch.setattr(va, "VLLMAttackerClient", _Cap)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    cfg = {"attacker": {"model": ATTACKER_MODEL}, "attacks": {"attacker_max_new_tokens": 64}}
    args = SimpleNamespace(attack="pair", attacker_engine="vllm", attacker_endpoint=URL,
                           attacker_model=None, scorer_endpoint=None,
                           _attacker_pool_key="k" * 43, _attacker_pool={"identity": {}})
    jailbreak_run._build_attacker_seams(cfg, args)
    assert made == [(URL, ATTACKER_MODEL, "k" * 43)]
    # a separate scorer endpoint is NOT the pool: it gets the env key only (none here)
    made.clear()
    args.scorer_endpoint = "http://192.0.2.11:8000"
    args.scorer_model = None
    jailbreak_run._build_attacker_seams(cfg, args)
    assert made[0][2] == "k" * 43 and made[1][2] is None
    # no pool, an env key: the per-job guarded server case
    made.clear()
    monkeypatch.setenv("VLLM_API_KEY", "envk")
    args2 = SimpleNamespace(attack="pair", attacker_engine="vllm", attacker_endpoint=URL,
                            attacker_model=None, scorer_endpoint=None)
    jailbreak_run._build_attacker_seams(cfg, args2)
    assert made == [(URL, ATTACKER_MODEL, "envk")]


# --------------------------------------------------------------------------- #
# grokmt_served.py plumbing                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def grokmt():
    from test_grokmt_served_resume import _stub_modal
    sys.modules.setdefault("modal", _stub_modal())
    spec = importlib.util.spec_from_file_location("_grokmt_attacker_pool",
                                                  REPO / "scripts/modal/grokmt_served.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grokmt_driver_cmd_pool_branch(grokmt, tmp_path):
    pj = tmp_path / "attacker_pool.json"
    pj.write_text(json.dumps(_ep_dict()))
    kw = dict(attack="crescendo", cell="none", sub=0, subshards=3, ids_path="ids", private_root="pr",
              proxy_url="http://127.0.0.1:1/v1", rt="rt.json", attacker_port=1234, target_window=8704)
    cmd = grokmt._driver_cmd("/tree", arm="qwen", attacker_pool=str(pj), **kw)
    assert cmd[cmd.index("--attacker-endpoint") + 1] == URL
    assert cmd[cmd.index("--attacker-pool") + 1] == str(pj) and "1234" not in " ".join(cmd)
    assert cmd[cmd.index("--attacker-engine") + 1] == "vllm"
    grok = grokmt._driver_cmd("/tree", arm="grok", attacker_pool=str(pj), **kw)
    assert grok[grok.index("--scorer-endpoint") + 1] == URL and "--attacker-pool" not in grok
    # the default (no pool) is byte-for-byte the per-job loopback command
    legacy = grokmt._driver_cmd("/tree", arm="qwen", **kw)
    assert legacy[legacy.index("--attacker-endpoint") + 1] == "http://127.0.0.1:1234"
    assert "--attacker-pool" not in legacy


def test_grokmt_attacker_for_run_pool_vs_local(grokmt, tmp_path, monkeypatch):
    spawned = []
    proc, pj, rec = grokmt._attacker_for_run(None, port=5, log=tmp_path / "l", run_local=tmp_path,
                                             start_attacker=lambda port, log: spawned.append(port) or "PROC")
    assert proc == "PROC" and pj is None and rec is None and spawned == [5]
    # pooled: NO local attacker; verified with the bearer from the env; endpoint copied for the drivers
    monkeypatch.setenv("VLLM_API_KEY", "k" * 43)
    seen = {}

    def connect(ep, **kw):
        seen.update(**kw)
        return ep, "k" * 43, {"ok": True, "health_status": 200}
    proc, pj, rec = grokmt._attacker_for_run(_modal_ep_dict(), port=5, log=tmp_path / "l", run_local=tmp_path,
                                             start_attacker=lambda *a: pytest.fail("spawned"),
                                             connect=connect)
    assert proc is None and json.loads(pj.read_text())["url"] == MODAL_URL
    assert seen["squeue"] is False and seen["env"] is os.environ
    assert seen["expect_model"] == ATTACKER_MODEL
    assert seen["expect_snapshot"] == ATTACKER_SNAPSHOT
    assert rec["identity"]["job_id"] == "fc-test" and rec["evidence"]["ok"]
    # a refusal propagates typed (run_smoke turns it into ATTACKER_POOL_REFUSED)
    def refuse(ep, **kw):
        raise ap.PoolUnauthorized("nope")
    with pytest.raises(ap.PoolUnauthorized):
        grokmt._attacker_for_run(_modal_ep_dict(), port=5, log=tmp_path / "l", run_local=tmp_path,
                                 start_attacker=lambda *a: pytest.fail("spawned"), connect=refuse)


def test_grokmt_provenance_and_signatures(grokmt):
    import inspect
    geom = grokmt.resolve_target_geometry(None)
    base = dict(run_id="r", sha="s" * 40, geom=geom, goal_ids=["g1"], attempts=[{"attempt": 1}],
                now_iso="2026-09-04T00:00:00Z", resume=False, summary={})
    per_job = grokmt._provenance(pool_rec=None, **base)
    assert per_job["attacker_pool"] is None and per_job["attacker_qwen"]["max_model_len"] == grokmt.ATTACKER_MAX_MODEL_LEN
    assert "2xH200" in per_job["topology"]
    ep = ap.PoolEndpoint.from_dict(_modal_ep_dict())
    pooled = grokmt._provenance(pool_rec={"identity": ep.identity(), "evidence": {"ok": True}}, **base)
    assert pooled["attacker_pool"]["identity"]["snapshot"] == ATTACKER_SNAPSHOT
    assert pooled["attacker_qwen"]["source"] == "shared attacker pool"
    assert pooled["attacker_qwen"]["max_model_len"] == 19200 and "1xH200" in pooled["topology"]
    assert MODAL_URL in pooled["topology"]
    # the per-job function keeps its signature (2xH200 default); the pooled twin takes the record
    for fn in (grokmt.run_smoke, grokmt.run_smoke_pooled, grokmt.main):
        params = inspect.signature(fn).parameters
        assert params["target_seqs"].default == 0 and params["target_key"].default == "gemma4e4b"
    assert "attacker_pool" not in inspect.signature(grokmt.run_smoke).parameters
    assert inspect.signature(grokmt.run_smoke_pooled).parameters["attacker_pool"].default is inspect.Parameter.empty
    assert inspect.signature(grokmt.main).parameters["attacker_pool"].default == ""
    src = (REPO / "scripts/modal/grokmt_served.py").read_text()
    assert 'gpu="H200:1"' in src and 'gpu="H200:2"' in src
    assert src.count("target_key=target_key") >= 3
    # the full endpoint record is read at `modal run` time; URL-only refuses
    assert grokmt._pool_spec_from_arg("") is None
    with pytest.raises(ap.PoolAbsent):
        grokmt._pool_spec_from_arg(URL)


# --------------------------------------------------------------------------- #
# the Modal pool app                                                           #
# --------------------------------------------------------------------------- #

class _ModalImageChain:
    def __getattr__(self, _name):
        return lambda *a, **k: self


def _modal_app_module():
    old = sys.modules.get("modal")
    fake = types.ModuleType("modal")
    class FakeApp:
        def __init__(self, name=None, **kw):
            self.name, self.app_id = name, "ap-local"
        def function(self, **kw):
            return lambda f: f
        def server(self, **kw):
            return lambda cls: cls
        def local_entrypoint(self, **kw):
            return lambda f: f
    fake.App = FakeApp
    fake.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(
        commit=lambda: None, reload=lambda: None))
    fake.Secret = types.SimpleNamespace(from_dict=lambda d: d)
    fake.Image = types.SimpleNamespace(
        from_registry=lambda *a, **k: _ModalImageChain(),
        debian_slim=lambda *a, **k: _ModalImageChain())
    fake.is_local = lambda: False
    fake.enter = lambda *a, **k: (lambda f: f)
    fake.exit = lambda *a, **k: (lambda f: f)
    fake.current_function_call_id = lambda: "fc-local"
    sys.modules["modal"] = fake
    old_key = os.environ.get("VLLM_API_KEY")
    os.environ["VLLM_API_KEY"] = "test-only-placeholder"
    try:
        spec = importlib.util.spec_from_file_location(
            "_modal_attacker_pool", REPO / "scripts/modal/attacker_pool.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if old is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = old
        if old_key is None:
            os.environ.pop("VLLM_API_KEY", None)
        else:
            os.environ["VLLM_API_KEY"] = old_key


def test_modal_canary_refuses_unproved_or_same_container_before_work():
    mod = _modal_app_module()
    ep = _modal_ep_dict()
    assert mod._assert_separate_container(ep, "ta-canary") == "ta-pool"
    with pytest.raises(RuntimeError, match="same-container green"):
        mod._assert_separate_container(ep, "ta-pool")
    with pytest.raises(RuntimeError, match="separation cannot be proved"):
        mod._assert_separate_container({**ep, "liveness": {}}, "ta-canary")
    with pytest.raises(RuntimeError, match="separation cannot be proved"):
        mod._assert_separate_container(ep, "")


def test_modal_coverage_gate_is_fatal_in_both_prefill_modes():
    mod = _modal_app_module()
    good = ("GPU KV cache size: 1,234,560 tokens\n"
            "Maximum concurrency for 19,200 tokens per request: 37.20x\n")
    assert mod._attest_coverage(good)["max_concurrency_reported"] == 37.2
    for chunked in (False, True):
        with pytest.raises(RuntimeError, match="Refusing to advertise"):
            mod._attest_coverage("server ready but no coverage line", chunked_prefill=chunked)
        low = "Maximum concurrency for 19,200 tokens per request: 36.99x\n"
        with pytest.raises(RuntimeError, match="below the launched"):
            mod._attest_coverage(low, chunked_prefill=chunked)


def _pool_engine_log(*, quantization="None", prefix="True", kv="auto", mamba=True) -> str:
    return ("INFO Initializing a V1 LLM engine (v0.25.1) with config: model='m', dtype=torch.bfloat16, "
            f"max_seq_len=19200, quantization={quantization}, kv_cache_dtype={kv}, "
            f"enable_prefix_caching={prefix}\n"
            + ("WARNING Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default "
               "when prefix caching is enabled\n" if mamba else "")
            + "INFO GPU KV cache size: 1,234,560 tokens\n"
              "Maximum concurrency for 19,200 tokens per request: 37.20x\n")


def test_modal_pool_serving_attestation_kills_a_contradicting_engine_and_derives_the_record(tmp_path):
    from sisteer.eval.attacks import attacker_serving as serving
    mod = _modal_app_module()
    opts = serving.AttackerServingOptions(mod.POOL_QUANTIZATION, mod.POOL_PREFIX_CACHING)
    manifest_path = tmp_path / "serving.json"
    # the engine came up FP8 although bf16 was requested: the standup fails, no manifest
    with pytest.raises(RuntimeError, match="contradicts the requested serving options"):
        mod._attest_serving(_pool_engine_log(quantization="fp8"), opts, manifest_path=manifest_path,
                            log_path="serve.log", endpoint="http://127.0.0.1:8000")
    assert not manifest_path.exists()
    # prefix caching silently off, or an FP8 KV cache: same refusal
    for bad in (_pool_engine_log(prefix="False", mamba=False), _pool_engine_log(kv="fp8")):
        with pytest.raises(RuntimeError, match="contradicts"):
            mod._attest_serving(bad, opts, manifest_path=manifest_path,
                                log_path="serve.log", endpoint="http://127.0.0.1:8000")
    # the agreeing engine: the record fields come from the LINE (and the manifest is written)
    manifest, fields = mod._attest_serving(_pool_engine_log(), opts, manifest_path=manifest_path,
                                           log_path="serve.log", endpoint="http://127.0.0.1:8000")
    assert manifest_path.exists() and manifest["engine_verified"] is True
    assert fields["attacker_quantization"] == "none" and fields["prefix_caching"] is True
    assert fields["serving"]["attested"] is True and fields["serving"]["mamba_cache_mode"] == "align"
    assert fields["serving"]["engine"]["vllm_version"] == "0.25.1"
    # a pool REQUESTING fp8 advertises fp8 because its engine line says so -- not a literal
    fp8 = serving.AttackerServingOptions("fp8", True)
    _, f2 = mod._attest_serving(_pool_engine_log(quantization="fp8"), fp8, manifest_path=tmp_path / "s2.json",
                                log_path="serve.log", endpoint="http://127.0.0.1:8000")
    assert f2["attacker_quantization"] == "fp8" and f2["serving"]["attacker_quantization"] == "fp8"
    # and the consumer copies the attested record, refuses the bare declaration
    copied = serving.pool_driver_fields(f2["serving"], declared_quantization="fp8", url=URL)
    assert copied["attacker_quantization"] == "fp8" and copied["attacker_serving_source"] == "attacker_pool"
    with pytest.raises(serving.AttackerServingError, match="refusing to copy a declaration"):
        serving.pool_driver_fields(None, declared_quantization="none", url=URL)


def test_contract_endpoint_cli_embeds_the_verified_manifest(tmp_path, capsys):
    from sisteer.eval.attacks import attacker_serving as serving
    opts = serving.AttackerServingOptions("none", True)
    engine = serving.parse_engine_log(_pool_engine_log())
    mpath = tmp_path / "serve.log.serving.json"
    serving.write_manifest(mpath, opts=opts, sources={"quantization": "flag", "prefix_caching": "flag"},
                           engine=engine, model="/snap", served_name=ATTACKER_MODEL,
                           endpoint=URL, log_path="serve.log", argv=opts.vllm_argv())
    pool_dir = tmp_path / "pool"; pool_dir.mkdir()
    rc = ap.main(["endpoint", "--pool-name", "t", "--pool-dir", str(pool_dir), "--url", URL,
                  "--model", ATTACKER_MODEL, "--snapshot", ATTACKER_SNAPSHOT, "--model-path", "/snap",
                  "--max-model-len", "19200", "--max-num-seqs", "37", "--max-num-batched-tokens", "16384",
                  "--api-key-file", str(tmp_path / "k"), "--job-id", "1", "--node", "n", "--node-ip", "192.0.2.10",
                  "--serving-manifest", str(mpath)])
    assert rc == 0
    rec = json.loads((pool_dir / "endpoint.json").read_text())
    assert rec["attacker_quantization"] == "none" and rec["prefix_caching"] is True
    assert rec["serving"]["attested"] is True and rec["serving"]["manifest"] == str(mpath)
    ep = ap.PoolEndpoint.from_dict(rec)
    assert ep.extra["serving"]["engine"]["enable_prefix_caching"] is True
    # a manifest that no longer verifies is refused at write time (the contract's typed refusal)
    bad = json.loads(mpath.read_text()); bad["engine"]["quantization"] = "fp8"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ap.AttackerPoolError, match="no longer verifies"):
        ap.main(["endpoint", "--pool-name", "t", "--pool-dir", str(pool_dir), "--url", URL,
                 "--model", ATTACKER_MODEL, "--snapshot", ATTACKER_SNAPSHOT, "--model-path", "/snap",
                 "--max-model-len", "19200", "--max-num-seqs", "37", "--max-num-batched-tokens", "16384",
                 "--api-key-file", str(tmp_path / "k"), "--job-id", "1", "--node", "n", "--node-ip", "192.0.2.10",
                 "--serving-manifest", str(tmp_path / "bad.json")])


def test_modal_app_contract_and_canary_order_are_pinned():
    txt = (REPO / "scripts/modal/attacker_pool.py").read_text()
    assert 'APP_NAME = "attacker-pool-qwen36-27b"' in txt
    assert 'name=SERVER_NAME' in txt and 'gpu="H200"' in txt
    assert 'tools = modal.App' in txt
    assert '@tools.function' in txt and '@tools.local_entrypoint()' in txt
    assert 'min_containers=1' in txt and 'max_containers=1' in txt
    assert 'target_concurrency=MAX_NUM_SEQS' in txt
    assert 'MAX_MODEL_LEN = 19_200' in txt and 'MAX_NUM_SEQS = 37' in txt
    # chunked prefill at a bounded batch; bf16 with NO weight offload and NO
    # eager mode (the unfittable identity forced both; it is gone); text-only
    assert 'MAX_NUM_BATCHED_TOKENS = 16_384' in txt and 'CHUNKED_PREFILL = True' in txt
    assert '"--enable-chunked-prefill"' in txt and '"--max-num-batched-tokens"' in txt
    assert 'chunk_tokens=MAX_NUM_BATCHED_TOKENS' in txt
    assert '"--cpu-offload-gb"' not in txt and 'CPU_OFFLOAD_GB' not in txt
    assert '"--enforce-eager"' not in txt
    assert 'LIMIT_MM_PER_PROMPT = \'{"image": 0, "video": 0}\'' in txt
    assert '"--limit-mm-per-prompt"' in txt
    # the serving identity is DERIVED from the engine line, never a literal in the record
    assert '"attacker_quantization": "none"' not in txt and '"prefix_caching": True' not in txt
    assert '**serving_fields' in txt and '"cpu_offload_gb": 0' in txt
    assert 'POOL_QUANTIZATION = "none"' in txt and 'POOL_PREFIX_CACHING = True' in txt
    assert '*opts.vllm_argv(model_role="attacker")' in txt      # flags from the one seam
    assert '"--enable-prefix-caching"' not in txt                # ... not a literal here
    assert '"ninja==1.13.0"' in txt
    assert '"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"' in txt
    assert 'for tool in ("vllm", "ninja")' in txt
    assert 'unauthenticated=True' in txt and 'secrets=[pool_secret]' in txt
    assert 'weights from /vol/hf' not in txt  # no prose-only substitute for the actual path
    assert 'MODEL_REPO_DIR = f"{VOL_PATH}/hf/hub/' in txt
    # Coverage, serving attestation and guard happen before the atomic advertisement.
    assert txt.index("coverage = _attest_coverage") < txt.index("_write_endpoint(record)")
    assert txt.index("_, serving_fields = _attest_serving(") < txt.index("_write_endpoint(record)")
    assert txt.index("unauth = _http_status") < txt.index("_write_endpoint(record)")
    # In the canary body, separation is established before connect or chat.
    canary_src = txt[txt.index("def canary("):]
    assert canary_src.index("_wait_endpoint_file") < canary_src.index("_assert_separate_container")
    assert canary_src.index("_assert_separate_container") < canary_src.index("connect_endpoint")
    assert canary_src.index("_assert_separate_container") < canary_src.index("client.chat")
    assert "squeue=False" in canary_src
    # Commands are explicit app names; no unnamed modal run/deploy example.
    assert "--name attacker-pool-qwen36-27b-stage" in txt
    assert "--name attacker-pool-qwen36-27b scripts/modal/attacker_pool.py::app" in txt
    assert "scripts/modal/attacker_pool.py::tools.main --action canary" in txt
    assert 'if modal.is_local()' in txt and 'SOURCE_DIR = Path("/root/sisteer")' in txt


# --------------------------------------------------------------------------- #
# the sbatch files                                                             #
# --------------------------------------------------------------------------- #

SBATCH_DIR = REPO / "scripts" / "sbatch"


def _code_lines(txt: str) -> list[str]:
    return [l for l in txt.splitlines() if not l.lstrip().startswith("#")]


def _hard_rules(txt: str) -> None:
    code = "\n".join(_code_lines(txt))
    assert not re.search(r"^#SBATCH --export", txt, re.M)          # rule 1
    assert "--export" not in code                                  # ... nor in the body
    assert not re.search(r"^#SBATCH --mem=0", txt, re.M)           # rule 2
    assert not re.search(r"^\s*(export\s+)?CUDA_VISIBLE_DEVICES=", txt, re.M)   # rule 3: read, never set
    assert not re.search(r"^#SBATCH --qos=dev", txt, re.M)
    assert re.search(r"^#SBATCH --time=\d-\d\d:\d\d:\d\d", txt, re.M)
    assert "#SBATCH --qos=high" in txt and "#SBATCH --open-mode=append" in txt
    assert "#SBATCH --no-requeue" in txt
    assert "hostname -I" not in code                               # the bridge-address bug, comments only


def test_pool_sbatch_contract():
    txt = (SBATCH_DIR / "attacker_pool.sbatch").read_text()
    _hard_rules(txt)
    assert "#SBATCH --gres=gpu:1" in txt and "#SBATCH --mem=128G" in txt
    assert re.search(r"^#SBATCH --time=1-00:00:00", txt, re.M)
    # positional sha; commit-addressed checkout; HEAD check
    assert 'POOL_SHA="${1:?' in txt and "git rev-parse HEAD" in txt
    # the routable address: the node NAME's resolution, refused if bridge/loopback
    assert 'getent hosts "$NODE"' in txt and "routable" in txt
    # the bearer: minted into a 0600 file, exported into the env, never on argv
    assert "mint --out" in txt and 'VLLM_API_KEY="$(cat "$KEY_FILE")"; export VLLM_API_KEY' in txt
    assert "--api-key" not in txt.replace("--api-key-file", "")
    assert "umask 077" in txt
    # geometry through the memory model; the identity via max_num_batched_tokens; prefix caching stated
    assert "size --max-model-len" in txt and "--max-num-batched-tokens" in txt
    assert "--enable-prefix-caching" in txt
    assert "--no-enable-chunked-prefill" not in "\n".join(_code_lines(txt))   # the composer's way: S x L
    assert "coverage --log" in txt
    assert 'if [ "$cov_rc" != 0 ]; then' in txt
    # the launcher's verified serving manifest is required and passed to the endpoint writer
    assert "VLLM_SERVING_MANIFEST=" in txt and '--serving-manifest "$SERVING_MANIFEST"' in txt
    assert "refusing to advertise an unattested pool" in txt
    assert '&& [ "$CHUNKED" != "1" ]' not in txt
    # refuse to overwrite a live pool; endpoint.json atomic via the CLI; trap removes it
    assert "live-job --pool" in txt and "endpoint --pool-name" in txt
    assert "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in txt and 'rm -f "$POOL_DIR/endpoint.json"' in txt
    # the guard verified from the routable address before advertising
    assert "want 401" in txt and "STOP" in txt
    # the documented options
    for opt in ("POOL_MAX_MODEL_LEN", "POOL_MAX_NUM_SEQS", "POOL_CHUNKED_PREFILL", "POOL_REPLICAS",
                "sbatch $CLUSTER_SBATCH_EXTRA scripts/sbatch/attacker_pool.sbatch"):
        assert opt in txt, opt
    assert not re.search(r"--exclude=\S", txt), "node exclusions are site config (CLUSTER_SBATCH_EXTRA), never spelled in the script"
    assert 'REPLICAS" = "1" ]' in txt                     # refused above 1
    assert "19200" in txt
    assert "retired launcher" in txt and "exit 64" in txt
    # runs the contract module from the vllm venv via PYTHONPATH (stdlib-only)
    assert 'export PYTHONPATH="$REPO_DIR/src' in txt and "sisteer.eval.attacks.attacker_pool" in txt
    # bash syntax
    subprocess.run(["bash", "-n", str(SBATCH_DIR / "attacker_pool.sbatch")], check=True)


def test_canary_sbatch_contract():
    txt = (SBATCH_DIR / "attacker_pool_canary.sbatch").read_text()
    _hard_rules(txt)
    assert "#SBATCH --gres=gpu:1" in txt                  # the target only
    assert "vllm_attacker_serve.sh" not in txt            # stands up nothing
    assert 'CANARY_SHA="${1:?' in txt
    # the order of assertions: reachability, guard, driver, pool log
    i1 = txt.index("ASSERT 1 OK"); i2 = txt.index("ASSERT 2 OK"); i3 = txt.index("ASSERT 3 OK")
    assert i1 < i2 < i3 < txt.index("logstats --log")
    assert "/health" in txt and "want 401" in txt and "verify --pool" in txt
    assert "--attacker-pool" in txt and "attacker_engine.json" in txt and "attacker_calls.jsonl" in txt
    assert "--attacker-endpoint" not in txt               # the pool sets it, never the canary
    assert "uv sync --frozen --extra cu128" in txt        # the drivers' venv, dedicated to the sha
    assert "crescendo" in txt and "prompted__safety_refusal" in txt and "CANARY_NSHARDS:-52" in txt
    assert "retired canary" in txt and "exit 64" in txt
    assert "NOTE: the canary landed" not in txt
    assert "fail \"the canary landed on the pool's own node" in txt
    subprocess.run(["bash", "-n", str(SBATCH_DIR / "attacker_pool_canary.sbatch")], check=True)


def test_slurm_network_path_is_only_unreachable_historical_code():
    """The retained scripts exit before any vanished-cluster network lookup."""
    for name in ("attacker_pool.sbatch", "attacker_pool_canary.sbatch"):
        txt = (SBATCH_DIR / name).read_text()
        assert txt.index("exit 64") < txt.index("getent hosts" if name == "attacker_pool.sbatch" else "POOL_DIR=")
    doc = (REPO / "docs" / "attacker_pool.md").read_text()
    assert "retired" in doc and "exit 64" in doc
    assert "modal.Server" in doc and "bare URL is refused" in doc


# --------------------------------------------------------------------------- #
# the profile script + the pinned measurement                                  #
# --------------------------------------------------------------------------- #

def _profile():
    spec = importlib.util.spec_from_file_location("_attacker_pool_profile",
                                                  REPO / "scripts/attacker_pool_profile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_profile_sizing_rule_on_synthetic_rows():
    pr = _profile()
    assert pr.pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5) == 5 and pr.pct(list(range(1, 101)), 0.99) == 99
    # branch 1: short completions -> p99 total; branch 2: every call at the cap -> prompt + cap
    # prompts spread 1k..20k with short completions: the clamp fires only on the tail (~4%)
    rows1 = [{"prompt_tokens": p, "completion_tokens": 400, "latency_s": 1.0} for p in range(1000, 20000, 10)]
    s1 = pr.summarize(rows1, 1024)
    assert s1["sized_by"] == "p99_total" and s1["L"] == pr.next_multiple(s1["p99_total"])
    # branch 2 (the real PAIR shape): prompts clustered just under the p99 with short
    # completions, so prompt + the 1,024 cap overshoots the total-p99 window on most calls
    rows2 = [{"prompt_tokens": p, "completion_tokens": 100, "latency_s": 1.0} for p in range(4000, 4400, 2)]
    s2 = pr.summarize(rows2, 1024)
    assert s2["clamp_rate_at_total_sizing"] > 0.05 and s2["sized_by"] == "p99_prompt_plus_max_tokens"
    assert s2["L"] == pr.next_multiple(s2["p99_prompt"] + 1024) and s2["n_at_cap"] == 0
    rows3 = [{"prompt_tokens": 100, "completion_tokens": 1024, "latency_s": 1.0}] * 20
    assert pr.summarize(rows3, 1024)["n_at_cap"] == 20

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True, **kw):
            assert kw.get("enable_thinking") is False
            return [0] * sum(len(m["content"]) for m in msgs)

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * len(text)}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "pair" / "cell" / "shard00"
        sd.mkdir(parents=True)
        (sd / "attacker_calls.jsonl").write_text(
            json.dumps({"role": "attacker", "messages": [{"role": "user", "content": "x" * 50}],
                        "reply": "y" * 20, "latency_s": 2.0, "params": {"max_new_tokens": 1024}}) + "\n"
            + json.dumps({"role": "attacker", "messages": [{"role": "user", "content": "x" * 70}],
                          "error": "boom", "latency_s": 1.0}) + "\n")
        (sd / "scorer_calls.jsonl").write_text(
            json.dumps({"role": "scorer", "messages": [{"role": "user", "content": "r" * 30}],
                        "reply": "7", "latency_s": 0.1, "params": {"max_new_tokens": 8}}) + "\n")
        grouped, n = pr.tokenize_rows([sd / "attacker_calls.jsonl", sd / "scorer_calls.jsonl"], _Tok(),
                                      max_tokens_by_role={"attacker": 1024, "scorer": 8})
        rep = pr.build_report(grouped, max_tokens_by_role={"attacker": 1024, "scorer": 8}, n_files=n,
                              tokenizer="t")
    assert rep["rows"]["attacker/pair"]["n_calls"] == 2 and rep["rows"]["attacker/pair"]["n_errors"] == 1
    assert rep["rows"]["attacker/pair"]["max_completion"] == 21          # reply + EOS
    assert rep["rows"]["scorer/pair"]["max_completion"] == 8             # the cap stands in
    assert rep["pool_max_model_len"] == max(v["L"] for v in rep["pooled"].values())
    md = pr.markdown(rep)
    assert "attacker (ALL attacks)" in md and "x" * 50 not in md          # never prompt text


def test_measured_literals_match_the_doc():
    """Every measured literal the window rests on is carried by the pool doc's window-sizing table, so a change to either
    side is a visible change to both (the measurement date lives in the code alone)."""
    assert ap.MEASUREMENT_DOC == "docs/attacker_pool.md"
    doc = (REPO / ap.MEASUREMENT_DOC).read_text()
    for lit in (ap.MEASURED_ATTACKER_P99_TOTAL, ap.MEASURED_ATTACKER_P99_PROMPT,
                ap.MEASURED_ATTACKER_MAX_TOTAL, ap.MEASURED_ATTACKER_N_CALLS,
                ap.DEFAULT_MAX_MODEL_LEN, ap.CRESCENDO_MAX_MODEL_LEN, ap.COVER_MAX_MAX_MODEL_LEN):
        assert f"{lit:,}" in doc, lit
    assert f"{100 * ap.MEASURED_CLAMP_RATE_AT_L:.2f}%" in doc
    assert f"{100 * ap.MEASURED_FRAC_OVER_L:.2f}%" in doc
    assert "CRESCENDO_MAX_MODEL_LEN" in doc and "COVER_MAX_MAX_MODEL_LEN" in doc and "attacker_pool_profile.py" in doc
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", ap.MEASURED_ON)


def test_measured_literals_obey_the_sizing_rule():
    # the rule: 18,973 -> 19,200 is the next 256-multiple, and the clamp rate is under the ceiling
    assert -(-ap.MEASURED_ATTACKER_P99_TOTAL // 256) * 256 == ap.DEFAULT_MAX_MODEL_LEN
    assert ap.MEASURED_CLAMP_RATE_AT_L < 0.05
    # the tail: the share of calls over L is within the clamp rate; the alternatives are 256-multiples that bracket it,
    # and the cover window really covers the largest observed call
    assert 0 < ap.MEASURED_FRAC_OVER_L <= ap.MEASURED_CLAMP_RATE_AT_L
    assert ap.MEASURED_ATTACKER_P99_PROMPT < ap.MEASURED_ATTACKER_P99_TOTAL <= ap.DEFAULT_MAX_MODEL_LEN
    assert ap.DEFAULT_MAX_MODEL_LEN < ap.CRESCENDO_MAX_MODEL_LEN < ap.COVER_MAX_MAX_MODEL_LEN
    assert ap.CRESCENDO_MAX_MODEL_LEN % 256 == 0 and ap.COVER_MAX_MAX_MODEL_LEN % 256 == 0
    assert ap.MEASURED_ATTACKER_MAX_TOTAL <= ap.COVER_MAX_MAX_MODEL_LEN < ap.MEASURED_ATTACKER_MAX_TOTAL + 1024
    # no personal paths in the pool doc (a per-user shared root included)
    me = os.environ.get("USER", "")
    txt = (REPO / "docs" / "attacker_pool.md").read_text()
    assert "/home/" not in txt
    if me:
        assert f"/{me}/" not in txt and f"/{me}\n" not in txt


def test_connect_endpoint_refuses_an_unattested_serving_record_before_any_liveness_lookup():
    from sisteer.eval.attacks import attacker_serving as serving

    def never(*a, **k):
        pytest.fail("a liveness lookup ran for a record the driver would refuse")

    # the first deployed pool's literal-only record: refused, network untouched
    bare = ap.PoolEndpoint.from_dict(_modal_ep_dict(attacker_quantization="none", serving=None))
    with pytest.raises(ap.PoolUnattested, match="refusing to copy a declaration"):
        ap.connect_endpoint(bare, modal_lookup=never, modal_app_lookup=never, modal_call_live=never)
    # an unattested block
    unattested = ap.PoolEndpoint.from_dict(_modal_ep_dict(attacker_quantization=None,
                                                          serving={"attacker_quantization": "none"}))
    with pytest.raises(ap.PoolUnattested, match="not engine-attested"):
        ap.connect_endpoint(unattested, modal_lookup=never, modal_app_lookup=never, modal_call_live=never)
    # an attested block that contradicts its own engine line
    opts = serving.AttackerServingOptions("none", True)
    engine = serving.parse_engine_log(_pool_engine_log())
    attested = serving.attested_serving_record(
        {"schema": serving.MANIFEST_SCHEMA, "requested": opts.as_dict(), "engine": engine,
         "engine_verified": True})
    tampered = dict(attested, attacker_quantization="fp8")
    with pytest.raises(ap.PoolUnattested, match="contradict"):
        ap.connect_endpoint(ap.PoolEndpoint.from_dict(_modal_ep_dict(attacker_quantization=None,
                                                                     serving=tampered)),
                            modal_lookup=never, modal_app_lookup=never, modal_call_live=never)
    # a genuinely attested record passes the serving check (its fields ride as evidence)
    good = ap.PoolEndpoint.from_dict(_modal_ep_dict(serving=attested, attacker_quantization="none"))
    assert ap.check_serving_record(good)["attacker_quantization"] == "none"
    # a record advertising NOTHING about its serving is refused too: a shared pool must
    # attest, so a block-less record is stale, stripped or malformed -- never 'unrecorded'
    nothing = ap.PoolEndpoint.from_dict(_modal_ep_dict(attacker_quantization=None, serving=None))
    with pytest.raises(ap.PoolUnattested, match="no engine-attested serving record at all"):
        ap.connect_endpoint(nothing, modal_lookup=never, modal_app_lookup=never, modal_call_live=never)
