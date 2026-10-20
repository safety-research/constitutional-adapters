"""scripts/modal/seedrep_attacker_pool.py -- the seedrep battery's parameterised attacker pools:
the env-driven parameters and their refusals, the per-card (tensor-parallel) sizing through the SHARED memory
model, the vLLM argv, the driver-count pool roster and the deterministic pin, the lifetime check, the
target-vs-attacker request routing against the vendor row read by EXACT registry ref (a target body carrying
the attacker's parameters is REFUSED), the health gate, and the import-time bindings under a stub `modal`.
Every check has a known-bad twin."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from seedrep_history_fixtures import pinned_commit_or_skip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/modal/seedrep_attacker_pool.py"
ORIGINAL = REPO / "scripts/modal/attacker_pool.py"
BASE_SHA = "0c192511"
VENDOR_ROW = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
POOL_ENV = {"SEEDREP_ATTACKER_POOL_INDEX": "0", "SEEDREP_ATTACKER_ATTACK": "crescendo"}


class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self


def _fake_modal():
    fake = types.ModuleType("modal")

    class FakeApp:
        instances: list = []

        def __init__(self, name=None, **kw):
            self.name, self.app_id = name, "ap-local"
            self.server_kwargs = None
            FakeApp.instances.append(self)

        def function(self, **kw):
            return lambda f: f

        def server(self, **kw):
            self.server_kwargs = kw
            return lambda cls: cls

        def local_entrypoint(self, **kw):
            return lambda f: f

    fake.App = FakeApp
    fake.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    fake.Secret = types.SimpleNamespace(from_dict=lambda d: d)
    fake.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    fake.is_local = lambda: False
    fake.enter = lambda *a, **k: (lambda f: f)
    fake.exit = lambda *a, **k: (lambda f: f)
    fake.current_function_call_id = lambda: "fc-local"
    return fake


def _load(env_overrides: dict | None = None, name: str = "_seedrep_attacker_pool"):
    """Import the pool module under the stub `modal` with the two REQUIRED env names (+ overrides)."""
    old_modal = sys.modules.get("modal")
    saved = {k: os.environ.get(k) for k in list(os.environ) if k.startswith("SEEDREP_ATTACKER_")}
    saved["VLLM_API_KEY"] = os.environ.get("VLLM_API_KEY")
    for k in saved:
        os.environ.pop(k, None)
    env = {**POOL_ENV, "VLLM_API_KEY": "test-only-placeholder", **(env_overrides or {})}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    fake = _fake_modal()
    sys.modules["modal"] = fake
    try:
        spec = importlib.util.spec_from_file_location(name, SRC)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # dataclasses resolve string annotations through sys.modules[cls.__module__]
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        mod._fake_modal = fake
        return mod
    finally:
        if old_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = old_modal
        for k in list(os.environ):
            if k.startswith("SEEDREP_ATTACKER_") or k == "VLLM_API_KEY":
                os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@pytest.fixture(scope="module")
def pool():
    return _load()


# --------------------------------------------------------------------------- parameters

def test_pool_params_defaults_are_the_plan_s_and_every_bad_value_is_refused(pool):
    p = pool.pool_params(POOL_ENV)
    # ONE card by default (H200:2 was the F-M era's identity-rule sizing); a spec with a count still parses (below)
    assert (p.pool_index, p.attack, p.gpu, p.gpu_kind, p.tensor_parallel) == (0, "crescendo", "H200", "H200", 1)
    assert (p.max_model_len, p.chunk_tokens, p.max_num_seqs, p.gpu_memory_utilization, p.lifetime_h) == (65_536, 16_384, None, 0.90, 24.0)
    assert p.app_name == "seedrep-attacker-0" and p.pool_name == "seedrep-0" and p.pool_dir == "/vol/attacker-pool/seedrep-0"
    assert (p.cpu, p.memory_mib) == (16, 131_072)
    assert pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H200:2"}).tensor_parallel == 2
    # the env block the images bake carries every param under the SEEDREP_ATTACKER_ prefix and round-trips
    env = p.as_env()
    assert env["SEEDREP_ATTACKER_GPU"] == "H200" and env["SEEDREP_ATTACKER_MAX_MODEL_LEN"] == "65536"
    assert "SEEDREP_ATTACKER_MAX_NUM_SEQS" not in env
    assert pool.pool_params(env) == p
    # a lower window on ONE card parses; an explicit S rides through (whether either FITS is pool_geometry's call, below)
    q = pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H200", "SEEDREP_ATTACKER_MAX_MODEL_LEN": "32768",
                          "SEEDREP_ATTACKER_MAX_NUM_SEQS": "12", "SEEDREP_ATTACKER_POOL_INDEX": "3", "SEEDREP_ATTACKER_ATTACK": "pair"})
    assert (q.gpu, q.tensor_parallel, q.max_model_len, q.max_num_seqs, q.app_name, q.cpu, q.memory_mib) == ("H200", 1, 32_768, 12, "seedrep-attacker-3", 16, 131_072)
    assert pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_W7": "abcdef0", "SEEDREP_ATTACKER_WAVE_UUID8": "01234567"}).w7 == "abcdef0"
    # KNOWN-BAD, one per rule
    bad = [
        ({"SEEDREP_ATTACKER_ATTACK": "crescendo"}, "POOL_INDEX is required"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_POOL_INDEX": "-1"}, "below the minimum"),
        ({"SEEDREP_ATTACKER_POOL_INDEX": "0"}, "ATTACK=''"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_ATTACK": "tap"}, "ATTACK='tap'"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "A100"}, "not <kind>"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H100:2"}, "calibrated on H200"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H200:3"}, "tensor-parallel size"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_MAX_MODEL_LEN": "65537"}, "multiple of the vLLM block size"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_MAX_MODEL_LEN": "4096"}, "below the minimum 8192"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_CHUNK_TOKENS": "131072"}, "exceeds the window"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_GPU_MEM_UTIL": "0.3"}, "outside"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_GPU_MEM_UTIL": "high"}, "not a number"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_LIFETIME_H": "0"}, "must be positive"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_W7": "abc"}, "not 7 hex"),
        ({**POOL_ENV, "SEEDREP_ATTACKER_MAX_NUM_SEQS": "0"}, "below the minimum 1"),
    ]
    for env, needle in bad:
        with pytest.raises(pool.PoolParamError, match=needle):
            pool.pool_params(env)


def test_parse_gpu_and_the_per_card_arch_split_weights_and_heads_by_the_tp_size(pool):
    from sisteer.eval.attacks import attacker_pool as c
    assert pool.parse_gpu("H200") == ("H200", 1) and pool.parse_gpu("H200:2") == ("H200", 2) and pool.parse_gpu("H200:8") == ("H200", 8)
    a2 = pool.per_card_arch(c.QWEN36_27B, 2)
    assert a2.weights_bytes == 27_781_427_952 == -(-c.QWEN36_27B.weights_bytes // 2)
    assert (a2.num_key_value_heads, a2.linear_num_key_heads, a2.linear_num_value_heads) == (2, 8, 24)
    assert a2.kv_bytes_per_token == c.QWEN36_27B.kv_bytes_per_token // 2 == 32_768
    assert a2.state_bytes_per_slot < c.QWEN36_27B.state_bytes_per_slot
    assert a2.vocab_size == c.QWEN36_27B.vocab_size          # the logits row stays whole (over-estimate, safe)
    assert pool.per_card_arch(c.QWEN36_27B, 1) is c.QWEN36_27B
    a8 = pool.per_card_arch(c.QWEN36_27B, 8)
    assert a8.num_key_value_heads == 1 and a8.linear_num_key_heads == 2 and a8.linear_num_value_heads == 6
    with pytest.raises(pool.PoolParamError):
        pool.per_card_arch(c.QWEN36_27B, 3)


def test_pool_geometry_is_the_shared_resolver_on_the_per_card_arch_pinned_numbers(pool):
    from sisteer.eval.attacks import attacker_pool as c
    # ONE H200 at 65,536, chunked 16,384, the PEAK_MEAN rule (the Slurm precedent): S = the production peak
    # (51 pinned children x 1.25 = 64), the KV checked at the MEAN sequence (crescendo mean prompt 4,756 + the 1,024 reply cap =
    # 5,780; 64 x 5,780 = 369,920 KV tokens, 22.6 GiB, 37.8 GiB headroom), vLLM's scheduler queuing the tail. The p99 fit rides
    # BESIDE it (p99_fit_S 24), never as the bound; the identity would have given 16.
    g1 = pool.pool_geometry(pool.pool_params(POOL_ENV))
    assert (g1.max_model_len, g1.max_num_seqs, g1.max_num_batched_tokens, g1.seqs_source, g1.fits, g1.chunked_prefill) == (65_536, 64, 16_384, "peak_mean", True, True)
    assert (g1.kv_rule, g1.kv_seq_tokens) == ("peak_mean", 5_780) and g1.kv_bytes == 64 * 5_780 * 65_536
    assert g1.weights_bytes == c.QWEN36_27B.weights_bytes == 55_562_855_904 and g1.reserve_bytes == c.RESERVE_BYTES
    assert "S_peak 64 = 51 pinned children x 1.25" in g1.note and "S_kv_mean 139" in g1.note and "p99_fit_S 24" in g1.note and "identity would give 16" in g1.note
    # the per-attack derivation the default S comes from, pinned with its inputs
    d = pool.derive_pool_sizing("crescendo", 65_536)
    assert (d["kv_rule"], d["S"], d["S_peak"], d["S_kv_mean"], d["mean_seq_tokens"], d["slots_cover_peak"]) == ("peak_mean", 64, 64, 139, 5_780, True)
    assert (d["p99_prompt_tokens"], d["p99_seq_tokens"], d["p99_seq_tokens_ceiling"], d["p99_fit_S"], d["S_kv_p99"], d["identity_S"]) == (41_206, 42_230, 51_024, 24, 24, 16)
    d2 = pool.derive_pool_sizing("actorattack", 65_536)
    assert (d2["S"], d2["S_kv_mean"], d2["mean_seq_tokens"], d2["p99_fit_S"], d2["p99_seq_tokens"]) == (158, 269, 1_864, 149, 5_214)
    d3 = pool.derive_pool_sizing("pair", 65_536)
    assert (d3["S"], d3["S_kv_mean"], d3["mean_seq_tokens"], d3["p99_fit_S"], d3["S_kv_p99"], d3["p99_seq_tokens"]) == (64, 225, 2_691, 64, 110, 7_842)
    # a tensor-parallel spec is still accepted and applies the same rule PER CARD with the F-M reserve
    g2 = pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H200:2"}))
    assert (g2.max_num_seqs, g2.seqs_source, g2.weights_bytes, g2.kv_rule) == (64, "peak_mean", 27_781_427_952, "peak_mean")
    assert g2.reserve_bytes == c.RESERVE_BYTES + pool.TP_RESERVE_BYTES == 5 * c.GIB and "45.07x" in g2.note and "tp 2" in g2.note
    # an explicit S that fits is kept and labelled explicit (the two-card era's 44 fits at the mean); one past the mean bound is
    # REFUSED BY NAME under the peak_mean rule, naming the largest that fits; a window that cannot hold the attack's p99 sequence
    # is refused as the wrong WINDOW, not a smaller S
    ge = pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_MAX_NUM_SEQS": "44"}))
    assert (ge.max_num_seqs, ge.seqs_source, ge.kv_rule, ge.kv_seq_tokens) == (44, "explicit", "peak_mean", 5_780)
    with pytest.raises(c.PoolGeometryUnfit, match=r"peak_mean rule \(S x 5780 sequence tokens.*max_num_seqs <= 139"):
        pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_MAX_NUM_SEQS": "140"}))
    with pytest.raises(pool.PoolParamError, match="the window, not S, is wrong"):
        pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_MAX_MODEL_LEN": "32768"}))
    # pair's p99 sequence (7,842) fits a 32,768 window; its S is the peak there too
    assert pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_ATTACK": "pair", "SEEDREP_ATTACKER_MAX_MODEL_LEN": "32768"})).max_num_seqs == 64


def test_vllm_argv_carries_the_geometry_and_tensor_parallel_only_above_one_card(pool):
    from sisteer.eval.attacks import attacker_serving as serving
    opts = serving.AttackerServingOptions(pool.POOL_QUANTIZATION, pool.POOL_PREFIX_CACHING)
    p2 = pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H200:2"})
    argv = pool.vllm_argv(p2, pool.pool_geometry(p2), opts)
    assert argv[:3] == ["vllm", "serve", pool.MODEL_PATH]
    assert argv[argv.index("--max-model-len") + 1] == "65536" and argv[argv.index("--max-num-seqs") + 1] == "64"
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "16384" and "--enable-chunked-prefill" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--served-model-name") + 1] == "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
    assert "--enable-prefix-caching" in argv and argv[argv.index("--limit-mm-per-prompt") + 1] == '{"image": 0, "video": 0}'
    assert "--quantization" not in argv
    p1 = pool.pool_params(POOL_ENV)                     # the default: ONE card, no --tensor-parallel-size, S 64 (peak_mean)
    argv1 = pool.vllm_argv(p1, pool.pool_geometry(p1), opts)
    assert "--tensor-parallel-size" not in argv1 and argv1[argv1.index("--max-num-seqs") + 1] == "64"
    assert pool.MODEL_PATH.endswith("/snapshots/27502c8717fd5a2f8c0c77188c10c243fd4f672e") and pool.ATTACKER_SNAPSHOT.startswith("27502c87")


# --------------------------------------------------------------------------- pool count + pinning + lifetime

def test_pool_roster_is_sized_by_drivers_with_the_plan_s_four_as_a_floor(pool):
    assert pool.pool_roster() == ((0, "crescendo"), (1, "crescendo"), (2, "actorattack"), (3, "pair"))
    assert pool.pools_needed(0, 100) == 0 and pool.pools_needed(1, 100) == 1 and pool.pools_needed(250, 100) == 3 and pool.pools_needed(200, 100) == 2
    # S4's measured drivers-per-pool raises an attack's count; the floor holds the others
    assert pool.size_pool_plan({"crescendo": 260, "actorattack": 90, "pair": 40}, 100) == (("crescendo", 3), ("actorattack", 1), ("pair", 1))
    assert pool.size_pool_plan({}, 100) == pool.DEFAULT_POOL_PLAN
    roster = pool.pool_roster(pool.size_pool_plan({"crescendo": 260, "actorattack": 90, "pair": 40}, 100))
    assert roster == ((0, "crescendo"), (1, "crescendo"), (2, "crescendo"), (3, "actorattack"), (4, "pair"))
    # the pin is deterministic round-robin over the attack's pools
    assert [pool.pool_for_driver(roster, "crescendo", i) for i in range(5)] == [0, 1, 2, 0, 1]
    assert pool.pool_for_driver(roster, "pair", 7) == 4
    # KNOWN-BAD: an attack with no pool, a plan without every attack, an unknown attack, a zero count
    with pytest.raises(ValueError, match="no pool serves attack"):
        pool.pool_for_driver(((0, "crescendo"),), "pair", 0)
    with pytest.raises(ValueError, match="no pool for"):
        pool.pool_roster((("crescendo", 2), ("actorattack", 1)))
    with pytest.raises(ValueError, match="unknown attack"):
        pool.size_pool_plan({"tap": 10}, 100)
    with pytest.raises(ValueError, match="at least one"):
        pool.pool_roster((("crescendo", 0), ("actorattack", 1), ("pair", 1)))
    with pytest.raises(ValueError):
        pool.pools_needed(10, 0)


def test_pool_outlives_the_longest_driver_plus_its_queue_wait_or_refuses(pool):
    assert pool.pool_outlives(24.0, 10.0, 2.0) == 12.0
    with pytest.raises(pool.PoolLifetimeShort, match="margin 0.00 h"):
        pool.pool_outlives(10.0, 10.0)
    with pytest.raises(pool.PoolLifetimeShort, match="queue wait"):
        pool.pool_outlives(10.0, 9.0, 2.0)          # 9 h of driver behind 2 h of queue: the pool dies first
    with pytest.raises(ValueError):
        pool.pool_outlives(10.0, 0.0)


# --------------------------------------------------------------------------- routing: the vendor row by exact ref

def _fixture_policy_file(tmp_path: Path, *, superseded: bool = False) -> Path:
    """The real export at this tree + the three vendor rows the lane needs (absent on the base tree), each a
    copy of the IFEval vendor row under its own ref. Returns the fixture's path."""
    data = json.loads((REPO / "configs/registry/decode_policy.json").read_text())
    ifeval = next(p for p in data["policies"] if p["policy_id"] == "qwen35_4b_it|ifeval|off|vendor_preset@v1")
    # main cc994f18 (#216) landed the three vendor rows in the export: the fixture's copies REPLACE them (appending beside them made
    # `by_ref` refuse "names 2 rows"), so every test below keeps the premise it was written against
    own = {f"qwen35_4b_it|{cat}|off|vendor_preset@v1" for cat in ("multi_turn", "petri", "bon")}
    data["policies"] = [p for p in data["policies"] if p["policy_id"] not in own]
    for cat, cap in (("multi_turn", "per-reply cap 4096"), ("petri", "1024 per turn"), ("bon", "4096 per reply")):
        row = dict(ifeval)
        row.update(policy_id=f"qwen35_4b_it|{cat}|off|vendor_preset@v1", policy_ref=f"qwen35_4b_it|{cat}|off|vendor_preset",
                   eval_id=None, eval_category=cat if cat != "bon" else "best_of_n", max_new_tokens_rule=cap, is_default=1,
                   same_values_as="qwen35_4b_it|ifeval|off|vendor_preset@v1", seed_rule="per_turn" if cat != "bon" else "per_draw",
                   superseded_by_policy_id=("qwen35_4b_it|multi_turn|off|vendor_preset@v2" if superseded and cat == "multi_turn" else None))
        data["policies"].append(row)
    out = tmp_path / ("decode_policy_fixture_superseded.json" if superseded else "decode_policy_fixture.json")
    out.write_text(json.dumps(data))
    return out


def test_target_row_resolves_the_exact_ref_and_never_a_default_lookup(pool, tmp_path):
    from sisteer.registry.decode_policy import NoPolicy, PolicyFile
    pf = PolicyFile.load(str(_fixture_policy_file(tmp_path)))
    row = pool.target_row(pf, "qwen35_4b", "multi_turn")
    assert row.policy_id == "qwen35_4b_it|multi_turn|off|vendor_preset@v1" and row.superseded_by_policy_id is None
    assert row.request_params() == VENDOR_ROW
    assert pool.target_row(pf, "qwen35_4b", "petri").policy_id == "qwen35_4b_it|petri|off|vendor_preset@v1"
    # KNOWN-BAD 1: a registry with no multi_turn vendor row -> NoPolicy propagates, nothing is invented. (The base export had none;
    # main cc994f18 / #216 landed it, so the absence is built here by removing the row from a copy of the committed export — and the
    # committed export itself now resolves the exact ref.)
    without = json.loads((REPO / "configs/registry/decode_policy.json").read_text())
    without["policies"] = [p for p in without["policies"] if p["policy_id"] != "qwen35_4b_it|multi_turn|off|vendor_preset@v1"]
    (tmp_path / "no_mt.json").write_text(json.dumps(without))
    with pytest.raises(NoPolicy, match=r"qwen35_4b_it\|multi_turn\|off\|vendor_preset"):
        pool.target_row(PolicyFile.load(str(tmp_path / "no_mt.json")), "qwen35_4b", "multi_turn")
    real = pool.target_row(PolicyFile.load(str(REPO / "configs/registry/decode_policy.json")), "qwen35_4b", "multi_turn")
    assert real.policy_id == "qwen35_4b_it|multi_turn|off|vendor_preset@v1" and real.superseded_by_policy_id is None
    # KNOWN-BAD 2: a superseded row fails closed
    with pytest.raises(pool.RoutingRefused, match="superseded"):
        pool.target_row(PolicyFile.load(str(_fixture_policy_file(tmp_path, superseded=True))), "qwen35_4b", "multi_turn")
    # the known-WRONG lookup the plan forbids: resolve(mode=) ignores the eval and returns the IFEval row -- never consulted
    wrong = pf.resolve("qwen35_4b", eval_id="crescendo", mode="vendor_preset")
    assert wrong.policy_id == "qwen35_4b_it|ifeval|off|vendor_preset@v1"
    src = SRC.read_text()
    # the grep rule: no default `resolve(` / `find(` on a decode-policy object; exact `by_ref(` only
    assert ".by_ref(" in src and "policy_file.resolve(" not in src and "policy_file.find(" not in src and ".resolve(mode" not in src


def test_route_request_sends_attacker_bodies_to_the_pool_and_holds_target_bodies_to_the_vendor_row(pool, tmp_path):
    from sisteer.registry.decode_policy import PolicyFile
    pf = PolicyFile.load(str(_fixture_policy_file(tmp_path)))
    row = pool.target_row(pf, "qwen35_4b", "multi_turn")
    table = pool.RoutingTable(attacker_models=frozenset({pool.ATTACKER_MODEL}), attacker_endpoint="https://pool-0.modal.run",
                              target_vmodels=frozenset({"petri__none", "si_constv2_mix_full__psr_ce_all"}),
                              target_endpoint="http://127.0.0.1:18984", target_params=row.request_params(), policy_ref=row.policy_id)
    # the attacker decodes at the ATTACK's parameters: routed as-is to the pool
    attacker_body = {"model": pool.ATTACKER_MODEL, "messages": [], "temperature": 1.0, "top_p": 1.0, "max_tokens": 1024}
    assert pool.route_request(attacker_body, table) == ("attacker", "https://pool-0.modal.run")
    # the target carries exactly the six vendor keys + seed: routed to the proxy
    target_body = {"model": "petri__none", "messages": [], **VENDOR_ROW, "max_tokens": 4096, "seed": 4057}
    assert pool.route_request(target_body, table) == ("target", "http://127.0.0.1:18984")
    assert pool.check_target_body(target_body, row.request_params()) == VENDOR_ROW
    # KNOWN-BAD 1: a TARGET body carrying the ATTACKER's parameters is refused, every departure named
    with pytest.raises(pool.RoutingRefused) as ei:
        pool.route_request({"model": "petri__none", "messages": [], "temperature": 1.0, "top_p": 1.0, "max_tokens": 4096, "seed": 1}, table)
    msg = str(ei.value)
    for needle in ("temperature: body sends 1.0, the row says 0.7", "top_p: body sends 1.0, the row says 0.8",
                   "top_k: absent", "min_p: absent", "repetition_penalty: absent", "presence_penalty: absent"):
        assert needle in msg, msg
    # KNOWN-BAD 2: an unseeded target draw
    with pytest.raises(pool.RoutingRefused, match="seed: absent"):
        pool.route_request({"model": "petri__none", "messages": [], **VENDOR_ROW}, table)
    with pytest.raises(pool.RoutingRefused, match="seed: absent"):
        pool.route_request({"model": "petri__none", **VENDOR_ROW, "seed": True}, table)
    # KNOWN-BAD 3: a key the server would fill (top_k omitted -> 20) is not a SENT key
    with pytest.raises(pool.RoutingRefused, match="top_k: absent from the body"):
        pool.route_request({"model": "petri__none", **{k: v for k, v in VENDOR_ROW.items() if k != "top_k"}, "seed": 3}, table)
    # KNOWN-BAD 4: presence penalty at the server-default 0.0 instead of the row's 1.5
    with pytest.raises(pool.RoutingRefused, match="presence_penalty: body sends 0.0, the row says 1.5"):
        pool.route_request({"model": "petri__none", **VENDOR_ROW, "presence_penalty": 0.0, "seed": 3}, table)
    # KNOWN-BAD 5: a model neither side serves; no model; an ambiguous table
    with pytest.raises(pool.RoutingRefused, match="neither"):
        pool.route_request({"model": "Qwen/Qwen3.5-4B", **VENDOR_ROW, "seed": 3}, table)
    with pytest.raises(pool.RoutingRefused, match="names no model"):
        pool.route_request({**VENDOR_ROW, "seed": 3}, table)
    with pytest.raises(pool.RoutingRefused, match="ambiguous"):
        pool.classify_request({"model": "x"}, attacker_models={"x"}, target_vmodels={"x"})


# --------------------------------------------------------------------------- health gate

def test_endpoint_health_requires_health_200_bare_401_and_bearer_200(pool):
    rec = {"url": "https://pool-0.modal.run/", "pool_name": "seedrep-0", "pool_index": 0, "attack": "crescendo"}
    good = {("https://pool-0.modal.run/health", None): 200, ("https://pool-0.modal.run/v1/models", None): 401,
            ("https://pool-0.modal.run/v1/models", "k"): 200}
    v = pool.endpoint_health(rec, "k", get=lambda u, b: good[(u, b)])
    assert (v["health"], v["models_unauthenticated"], v["models_authenticated"], v["pool_index"], v["attack"]) == (200, 401, 200, 0, "crescendo")
    # KNOWN-BAD: the guard is void (bare 200); the engine is down (health 503); the bearer is refused (auth 401)
    with pytest.raises(RuntimeError, match="bearer guard is void"):
        pool.endpoint_health(rec, "k", get=lambda u, b: {**good, ("https://pool-0.modal.run/v1/models", None): 200}[(u, b)])
    with pytest.raises(RuntimeError, match="/health=503"):
        pool.endpoint_health(rec, "k", get=lambda u, b: {**good, ("https://pool-0.modal.run/health", None): 503}[(u, b)])
    with pytest.raises(RuntimeError, match="authenticated /v1/models=401"):
        pool.endpoint_health(rec, "k", get=lambda u, b: {**good, ("https://pool-0.modal.run/v1/models", "k"): 401}[(u, b)])
    with pytest.raises(RuntimeError, match="no url"):
        pool.endpoint_health({}, "k", get=lambda u, b: 200)


# --------------------------------------------------------------------------- the module under `modal deploy`

def test_the_module_binds_the_app_and_the_server_from_the_env_and_the_endpoint_record_validates(pool, monkeypatch, tmp_path):
    assert pool.APP_NAME == "seedrep-attacker-0" and pool.POOL_DIR == "/vol/attacker-pool/seedrep-0"
    assert pool.ENDPOINT_PATH == "/vol/attacker-pool/seedrep-0/endpoint.json" and pool.SERVING_MANIFEST == "/vol/attacker-pool/seedrep-0/serving.json"
    assert (pool.MAX_MODEL_LEN, pool.MAX_NUM_SEQS, pool.MAX_NUM_BATCHED_TOKENS) == (65_536, 64, 16_384)
    apps = {a.name: a for a in pool._fake_modal.App.instances}
    assert set(apps) >= {"seedrep-attacker-0", "seedrep-attacker-0-tools"}
    kw = apps["seedrep-attacker-0"].server_kwargs
    assert kw["gpu"] == "H200" and kw["target_concurrency"] == 64 and kw["cpu"] == 16 and kw["memory"] == 131_072
    assert (kw["min_containers"], kw["max_containers"], kw["startup_timeout"], kw["port"], kw["name"]) == (1, 1, 3600, 8000, "qwen36-27b-abliterated-vllm")
    assert apps["seedrep-attacker-0-tools"].server_kwargs is None      # the tools app never declares a GPU server
    assert pool._IMAGE_ENV["SEEDREP_ATTACKER_POOL_INDEX"] == "0" and pool._IMAGE_ENV["SEEDREP_ATTACKER_ATTACK"] == "crescendo"
    assert pool._IMAGE_ENV["SEEDREP_ATTACKER_GPU"] == "H200" and pool._IMAGE_ENV["HF_HOME"] == "/vol/hf"
    # the endpoint record: the contract's required keys + the lane's fields; validated by the contract's reader
    from sisteer.eval.attacks import attacker_pool as c
    rec = pool.endpoint_record_fields(
        pool.P, pool.GEOM, coverage={"kv_tokens_reported": 1_060_000, "concurrency_window": 65_536, "max_concurrency_reported": 16.17},
        serving_fields={"attacker_quantization": "none", "prefix_caching": True, "serving": {"x": 1}},
        url="https://seedrep-attacker-0.modal.run", liveness={"kind": "modal", "app_name": "seedrep-attacker-0", "app_id": "ap-1",
                                                               "server_name": "qwen36-27b-abliterated-vllm", "server_id": "sv-1",
                                                               "call_id": None, "container_id": "ta-1"},
        started="2026-09-18T00:00:00Z", call_id=None, server_id="sv-1", container_id="ta-1", source_sha="deadbeef")
    ep = c.PoolEndpoint.from_dict(rec)
    assert (ep.url, ep.model, ep.snapshot, ep.max_model_len, ep.max_num_seqs, ep.max_num_batched_tokens, ep.job_id, ep.node) == (
        "https://seedrep-attacker-0.modal.run", pool.ATTACKER_MODEL, pool.ATTACKER_SNAPSHOT, 65_536, 64, 16_384, "sv-1", "ta-1")
    assert (rec["lane"], rec["pool_index"], rec["attack"], rec["gpu"], rec["tensor_parallel_size"], rec["expected_lifetime_h"]) == ("seedrep", 0, "crescendo", "H200", 1, 24.0)
    assert rec["params"]["max_model_len"] == 65_536 and rec["api_key_file"] is None and rec["api_key_env_name"] == "VLLM_API_KEY"
    assert rec["geometry"]["seqs_source"] == "peak_mean" and rec["geometry"]["kv_rule"] == "peak_mean" and rec["geometry"]["kv_seq_tokens"] == 5_780 and rec["chunked_prefill"] is True
    assert (rec["sizing"]["kv_rule"], rec["sizing"]["S"], rec["sizing"]["S_peak"], rec["sizing"]["S_kv_mean"], rec["sizing"]["p99_fit_S"], rec["sizing"]["slots_cover_peak"]) == ("peak_mean", 64, 64, 139, 24, True)
    assert rec["sizing"]["S"] == rec["max_num_seqs"] == rec["geometry"]["max_num_seqs"] and rec["sizing"]["recommended"]["S"] == 64 and rec["sizing"]["explicit_override"] is False
    assert [m["metric"] for m in rec["watch_metrics"]] == list(pool.WATCH_METRIC_NAMES) and rec["sizing"]["caveat"] == pool.PEAK_MEAN_CAVEAT
    # the lane key file is per lane (shared by every seedrep pool), never the campaign pool's -- and under the tmpfs runtime dir,
    # while the RECORDS stay under the state dir (an isolated empty site: the neutral default); the legacy key location is known
    # only to refuse it
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/0")
    _empty_site(monkeypatch, tmp_path)
    assert pool._local_key_path() == Path("/run/user/0/seedrep-battery/attacker-pool/seedrep/api_key")
    assert pool._legacy_key_path() == Path.home() / ".local/state/sisteer/attacker-pool-modal/seedrep/api_key"
    assert pool._local_endpoint_path() == Path.home() / ".local/state/sisteer/attacker-pool-modal/seedrep/endpoint_0.json"
    # a site that names the directory moves the records (and the legacy location) with it
    monkeypatch.setenv("SISTEER_ATTACKER_POOL_STATE_DIR", str(tmp_path / "pool-state"))
    assert pool._local_endpoint_path() == tmp_path / "pool-state/seedrep/endpoint_0.json"
    assert pool._legacy_key_path() == tmp_path / "pool-state/seedrep/api_key"


def test_a_second_pool_index_binds_its_own_names_and_a_missing_required_env_refuses_the_import():
    m3 = _load({"SEEDREP_ATTACKER_POOL_INDEX": "3", "SEEDREP_ATTACKER_ATTACK": "pair", "SEEDREP_ATTACKER_GPU": "H200",
                "SEEDREP_ATTACKER_MAX_MODEL_LEN": "32768"}, name="_seedrep_attacker_pool_3")
    assert m3.APP_NAME == "seedrep-attacker-3" and m3.POOL_DIR == "/vol/attacker-pool/seedrep-3" and m3.P.attack == "pair"
    assert (m3.MAX_MODEL_LEN, m3.MAX_NUM_SEQS, m3.GEOM.seqs_source) == (32_768, 64, "peak_mean")     # pair: the production peak
    kw = {a.name: a for a in m3._fake_modal.App.instances}["seedrep-attacker-3"].server_kwargs
    assert kw["gpu"] == "H200" and kw["target_concurrency"] == 64 and kw["cpu"] == 16
    # KNOWN-BAD: no pool index -> the import refuses (a nameless deploy would advertise nothing drivers can pin)
    with pytest.raises(SystemExit, match="POOL_INDEX is required"):
        _load({"SEEDREP_ATTACKER_POOL_INDEX": None}, name="_seedrep_attacker_pool_bad")
    with pytest.raises(SystemExit, match="ATTACK"):
        _load({"SEEDREP_ATTACKER_ATTACK": None}, name="_seedrep_attacker_pool_bad2")



def _code_of(text: str, *, except_functions: tuple = ()) -> str:
    """A module's code, comments and docstrings aside: its syntax tree with every docstring removed, dumped. A top-level
    function named in ``except_functions`` is left out (a change that was made to the original on purpose, named here)."""
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in except_functions)]
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def test_the_original_pool_is_untouched_and_the_copy_carries_no_credential_and_names_its_own_apps():
    src = SRC.read_text()                                              # history-independent first: outside the skipped portion
    assert "sk-ant" not in src and "SISTEER_SECRET_CMD" not in src   # the pool has no vault credential; its bearer is self-minted
    assert 'LANE = "seedrep"' in src and "attacker-pool-qwen36-27b" not in src
    assert "--name seedrep-attacker-0 scripts/modal/seedrep_attacker_pool.py::app" in src
    assert "modal app stop --yes seedrep-attacker-0" in src
    # the docstring's deploy line passes the two REQUIRED names inline
    assert "SEEDREP_ATTACKER_POOL_INDEX=0 SEEDREP_ATTACKER_ATTACK=crescendo" in src
    # the original pool's file, unchanged since the workstream base (the object store; the base sits on every seedrep-ws* branch)
    pinned_commit_or_skip(BASE_SHA, "seedrep-ws2f", "the workstream base for scripts/modal/attacker_pool.py", repo=REPO, anchor=Path(__file__))
    old = subprocess.run(["git", "show", f"{BASE_SHA}:{ORIGINAL.relative_to(REPO)}"], cwd=str(REPO), capture_output=True,
                         text=True, check=True).stdout
    # one deliberate exception: the release moved the grading host's state directory behind the site settings
    # (`_local_state_dir`, sisteer.site.state_path), the same change this module's copy carries
    released = ("_local_state_dir",)
    assert _code_of(old, except_functions=released) == _code_of(ORIGINAL.read_text(), except_functions=released), \
        "scripts/modal/attacker_pool.py's code must not be edited by this workstream (its comments and docstrings may change wording)"


# --------------------------------------------------------------------------- the self-minted bearer lives ONLY in the tmpfs runtime dir
def _empty_site(monkeypatch, tmp_path):
    """An isolated, empty site config and none of the state keys in the environment: the state dir is the neutral default."""
    cfg = tmp_path / "site" / "config.env"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text("")
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    for k in ("SISTEER_ATTACKER_POOL_STATE_DIR", "SISTEER_STATE_ROOT"):
        monkeypatch.delenv(k, raising=False)


def _grading_host(pool, monkeypatch, tmp_path, *, runtime=True, rotate=False):
    """The pool module as the grading host runs it (``modal.is_local()`` True): HOME and the runtime dir under ``tmp_path``,
    and an isolated empty site."""
    monkeypatch.setattr(pool.modal, "is_local", lambda: True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    _empty_site(monkeypatch, tmp_path)
    if runtime:
        rd = tmp_path / "run"
        rd.mkdir(exist_ok=True)
        rd.chmod(0o700)
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(rd))
    else:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    if rotate:
        monkeypatch.setenv("SEEDREP_ATTACKER_ROTATE_KEY", "1")
    else:
        monkeypatch.delenv("SEEDREP_ATTACKER_ROTATE_KEY", raising=False)
    return home, (tmp_path / "run")


def _modes_below(root: Path, path: Path) -> list:
    out, cur = [], root
    for part in path.relative_to(root).parts:
        cur = cur / part
        out.append(cur.stat().st_mode & 0o777)
    return out


def test_d77_the_bearer_is_minted_0600_under_0700_dirs_in_the_runtime_dir_stable_across_reads_and_rotated_on_request(pool, monkeypatch, tmp_path):
    """The self-minted transport bearer is rendered under ``$XDG_RUNTIME_DIR`` (tmpfs,
    RAM-only) -- ``seedrep-battery/attacker-pool/<lane>/api_key``, every directory 0700, the file 0600 -- read back stable, and
    re-minted only under ``ROTATE_KEY=1``; nothing is written under ``~/.local/state``."""
    home, rd = _grading_host(pool, monkeypatch, tmp_path)
    key = pool._load_local_api_key()
    kf = pool._local_key_path()
    assert kf == rd / "seedrep-battery" / "attacker-pool" / "seedrep" / "api_key" and kf.is_file()
    assert kf.read_text().strip() == key and len(key) >= 32
    assert _modes_below(rd, kf) == [0o700, 0o700, 0o700, 0o600]
    assert pool._load_local_api_key() == key                                             # stable: read, not re-minted
    assert not (home / ".local").exists()                                                # nothing under ~/.local/state
    monkeypatch.setenv("SEEDREP_ATTACKER_ROTATE_KEY", "1")
    rotated = pool._load_local_api_key()
    assert rotated != key and kf.read_text().strip() == rotated and (kf.stat().st_mode & 0o777) == 0o600
    # the container side is unchanged: the bearer arrives through the Modal secret's env, never a file
    monkeypatch.setattr(pool.modal, "is_local", lambda: False)
    monkeypatch.setenv(pool.API_KEY_ENV, "from-the-secret")
    assert pool._load_local_api_key() == "from-the-secret"


def test_d77_the_runtime_dir_unset_refuses_the_mint_rather_than_falling_back_to_the_state_dir(pool, monkeypatch, tmp_path):
    home, _ = _grading_host(pool, monkeypatch, tmp_path, runtime=False)
    with pytest.raises(SystemExit, match="XDG_RUNTIME_DIR is unset"):
        pool._local_key_path()
    with pytest.raises(SystemExit, match="refusing to fall back to ~/.local/state"):
        pool._load_local_api_key()
    assert not (home / ".local").exists()
    monkeypatch.setenv("XDG_RUNTIME_DIR", "")
    with pytest.raises(SystemExit, match="XDG_RUNTIME_DIR is unset"):
        pool._load_local_api_key()


def test_d77_a_pre_d77_bearer_under_the_state_dir_refuses_by_name_and_only_rotate_key_discards_it(pool, monkeypatch, tmp_path):
    """KNOWN-BAD: a bearer file at the legacy location is never migrated (silently or otherwise): its presence refuses the mint,
    naming ``SEEDREP_ATTACKER_ROTATE_KEY=1``; under that flag it is discarded and a fresh bearer is minted in tmpfs."""
    home, rd = _grading_host(pool, monkeypatch, tmp_path)
    legacy = pool._legacy_key_path()
    assert legacy == home / ".local/state/sisteer/attacker-pool-modal/seedrep/api_key"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy-bearer-not-to-be-read\n")
    legacy.chmod(0o600)
    with pytest.raises(SystemExit, match=r"NOT migrated -- re-run with SEEDREP_ATTACKER_ROTATE_KEY=1"):
        pool._load_local_api_key()
    assert legacy.read_text() == "legacy-bearer-not-to-be-read\n"                       # untouched
    assert not (rd / "seedrep-battery").exists()                                          # nothing minted
    monkeypatch.setenv("SEEDREP_ATTACKER_ROTATE_KEY", "1")
    fresh = pool._load_local_api_key()
    assert fresh != "legacy-bearer-not-to-be-read" and not legacy.exists()
    assert pool._local_key_path().read_text().strip() == fresh


def test_d77_a_group_accessible_runtime_component_refuses_and_the_endpoint_record_names_the_runtime_dir_bearer(pool, monkeypatch, tmp_path):
    home, rd = _grading_host(pool, monkeypatch, tmp_path)
    (rd / "seedrep-battery").mkdir(mode=0o750)
    (rd / "seedrep-battery").chmod(0o750)
    with pytest.raises(RuntimeError, match="is not private"):
        pool._load_local_api_key()
    assert not pool._local_key_path().exists()
    # the grading-host-side `--action endpoint` stamps the record's api_key_file with THIS path (the drive's reader then holds it to the runtime dir)
    src = SRC.read_text()
    assert 'rec["api_key_file"] = str(_local_key_path())' in src
    assert src.index("def _runtime_key_dir") < src.index("def _local_key_path") and "~/.local/state" not in src.split("def _runtime_key_dir")[1].split("def _local_key_path")[0].replace("refusing to fall back to ~/.local/state", "")
    assert pool.endpoint_record_fields.__doc__ and "api_key_file" in src.split("def endpoint_record_fields")[1].split("def ")[0]


# --------------------------------------------------------------------------- F-M: allocator config follows the card count

def test_a_multi_card_pool_bakes_no_expandable_segments_and_refuses_one_that_carries_it(pool):
    """F-M (2026-09-18): the four H200:2 Servers loaded (25.17 GiB/card), compiled (333 s) and warmed up (268 s), then
    died at the FIRST CUDA-graph capture -- ``Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid
    argument'`` (``get_graph_buffer_ipc_meta``: cudaIpcGetMemHandle on the torch-allocated all-reduce buffers), the
    EngineCore dead, Modal restarting the min_containers=1 Server for 50 minutes with no endpoint.json. The pool image had
    inherited the one-card campaign pool's ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``, whose cuMemMap allocations
    the legacy IPC export refuses; harmless at tp 1 (no all-reduce), fatal at tp 2."""
    # the pure rule: one card keeps the campaign pool's setting; every multi-card size bakes NOTHING
    assert pool.alloc_conf_env(1) == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    assert pool.alloc_conf_env(2) == {} and pool.alloc_conf_env(4) == {} and pool.alloc_conf_env(8) == {}
    with pytest.raises(pool.PoolParamError, match="must be one of"):
        pool.alloc_conf_env(3)
    # the fixture's pool IS the plan's ONE card -> its baked env carries the campaign pool's setting; a two-card pool bakes none
    assert pool.P.tensor_parallel == 1 and pool._IMAGE_ENV["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    one_card = pool
    two_card = _load({"SEEDREP_ATTACKER_GPU": "H200:2"}, name="_seedrep_attacker_pool_tp2_alloc")
    assert two_card.P.tensor_parallel == 2 and "PYTORCH_CUDA_ALLOC_CONF" not in two_card._IMAGE_ENV
    # the Server's enter guard: KNOWN-BAD env at tp 2 is a named refusal pointing at the failing call; tp 1 and clean envs pass
    bad = pool.serve_env_problem({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, 2)
    assert bad and "cudaIpcGetMemHandle" in bad and "custom_all_reduce.cuh:455" in bad and "tensor-parallel 2" in bad
    assert pool.serve_env_problem({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512, Expandable_Segments:TRUE"}, 2)
    assert pool.serve_env_problem({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, 1) is None
    assert pool.serve_env_problem({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}, 2) is None
    assert pool.serve_env_problem({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}, 2) is None
    assert pool.serve_env_problem({}, 2) is None and two_card.serve_env_problem(two_card._IMAGE_ENV, two_card.P.tensor_parallel) is None
    assert one_card.serve_env_problem(one_card._IMAGE_ENV, one_card.P.tensor_parallel) is None
    # the endpoint record states the allocator config the engine ran under (None above one card), and still validates
    from sisteer.eval.attacks import attacker_pool as c
    live = {"kind": "modal", "app_name": "seedrep-attacker-0", "app_id": "ap-1", "server_name": "qwen36-27b-abliterated-vllm",
            "server_id": "sv-1", "call_id": "fc-1", "container_id": "ta-1"}
    rec = two_card.endpoint_record_fields(
        two_card.P, two_card.GEOM, coverage={"kv_tokens_reported": 3_014_656, "concurrency_window": 65_536, "max_concurrency_reported": 46.0},
        serving_fields={"attacker_quantization": "none", "prefix_caching": True, "serving": {"x": 1}},
        url="https://seedrep-attacker-0.modal.run", liveness=live, started="2026-09-18T06:00:00Z", call_id="fc-1",
        server_id="sv-1", container_id="ta-1", source_sha="a" * 40)
    assert rec["pytorch_cuda_alloc_conf"] is None and rec["tensor_parallel_size"] == 2
    assert c.PoolEndpoint.from_dict(rec).max_num_seqs == 64
    # the carries downstream: identity() -- the projection run manifests keep -- names the deployment shape, and the Server
    # hands the allocator config it ran under to serving.json's extra block
    ident = c.PoolEndpoint.from_dict(rec).identity()
    assert ident["pytorch_cuda_alloc_conf"] is None and ident["tensor_parallel_size"] == 2 and ident["gpu"] == "H200:2" and ident["vllm_version"] == pool.VLLM_VERSION
    import inspect
    assert '"pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")' in inspect.getsource(pool.SeedrepAttackerPoolServer.start)
    rec1 = one_card.endpoint_record_fields(
        one_card.P, one_card.GEOM, coverage={"kv_tokens_reported": 1_060_000, "concurrency_window": 65_536,
                                             "max_concurrency_reported": 16.17},
        serving_fields={"attacker_quantization": "none", "prefix_caching": True, "serving": {"x": 1}},
        url="https://seedrep-attacker-0.modal.run", liveness=live, started="2026-09-18T06:00:00Z", call_id="fc-1",
        server_id="sv-1", container_id="ta-1", source_sha="a" * 40)
    assert rec1["pytorch_cuda_alloc_conf"] == "expandable_segments:True" and rec1["tensor_parallel_size"] == 1 and rec1["gpu"] == "H200"
    ident1 = c.PoolEndpoint.from_dict(rec1).identity()
    assert ident1["gpu"] == "H200" and ident1["tensor_parallel_size"] == 1 and ident1["max_num_seqs"] == 64 and ident1["pytorch_cuda_alloc_conf"] == "expandable_segments:True"


# --------------------------------------------------------------------------- F-M (2): the launched S sits inside the LIVE room

def test_launched_s_sits_at_or_below_the_live_certified_room_via_the_tensor_parallel_reserve(pool):
    """Pool 0's first advertised-or-refused run (2026-09-18 06:37Z, after the all-reduce fix): vLLM reported room for 45.07
    sequences of 65,536 tokens; the one-card model had launched S 46 (0.1 GiB headroom) and the Server's coverage check --
    unchanged here, it did its job -- refused to advertise. The per-card reserve for the tensor-parallel communicators and
    graphs makes the model sit inside the measured room; the measurement and the reserve are recorded. The DEFAULT pool
    is one card and the rule is the p99 fit, so this history is exercised on an explicit H200:2 params object."""
    import math
    from sisteer.eval.attacks import attacker_pool as c
    p2 = pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H200:2"})
    g2 = pool.pool_geometry(p2)
    key = (p2.gpu_kind, p2.tensor_parallel, p2.max_model_len)
    live = pool.LIVE_ROOM_MEASURED[key]
    assert live == 45.07 and key == ("H200", 2, 65_536)
    # the identity relation the deploy proved necessary (S <= floor(room)) held at S 44; under the peak_mean rule the live KV
    # (45.07 x 65,536 = 2,953,700 tokens) holds the two-card S 64 x 5,780 and the check agrees; it also holds 64 x 42,230 p99
    cov = {"kv_tokens_reported": int(live * 65_536), "concurrency_window": 65_536, "max_concurrency_reported": live}
    assert c.coverage_problem(cov, max_num_seqs=44, max_model_len=65_536) is None
    assert c.coverage_problem(cov, max_num_seqs=g2.max_num_seqs, max_model_len=65_536, kv_seq_tokens=g2.kv_seq_tokens, kv_rule=g2.kv_rule) is None
    assert c.coverage_fraction(cov, max_num_seqs=64, max_model_len=65_536, seq_tokens=42_230) >= 1.0
    # KNOWN-BAD: the pre-reserve S 46 is exactly what the identity check refused live; 70 p99 sequences do not fit the live room either
    assert "below the launched max_num_seqs 46" in c.coverage_problem(cov, max_num_seqs=46, max_model_len=65_536)
    assert "relaunch at S <= 69" in c.coverage_problem(cov, max_num_seqs=70, max_model_len=65_536, kv_seq_tokens=42_230, kv_rule="p99_fit")
    # the reserve: one-card 2 GiB, +3 GiB per card above one card; recorded in the geometry and the endpoint record
    assert pool.pool_reserve_bytes(1) == c.RESERVE_BYTES == 2 * c.GIB and pool.pool_reserve_bytes(2) == 5 * c.GIB
    with pytest.raises(pool.PoolParamError, match="must be one of"):
        pool.pool_reserve_bytes(3)
    assert g2.reserve_bytes == 5 * c.GIB and "45.07x" in g2.describe() and math.floor(live) >= 44
    # a window WITHOUT a measurement says so in the note and still carries the reserve (the coverage check decides live)
    g48 = pool.pool_geometry(pool.pool_params({**POOL_ENV, "SEEDREP_ATTACKER_GPU": "H200:2", "SEEDREP_ATTACKER_MAX_MODEL_LEN": "49152"}))
    assert g48.reserve_bytes == 5 * c.GIB and "unmeasured" in g48.note
    # ONE card (the default): the campaign pool's reserve; the note carries the one-card arithmetic, no tensor-parallel clause
    g1 = pool.pool_geometry(pool.pool_params(POOL_ENV))
    assert g1.reserve_bytes == c.RESERVE_BYTES and "tp " not in g1.note and g1.max_num_seqs == 64 and pool.GEOM.max_num_seqs == 64
    assert c.resolve_pool_geometry(19200, 37, chunked_prefill=True, chunk_tokens=16_384).reserve_bytes == c.RESERVE_BYTES
    with pytest.raises(ValueError, match="reserve_bytes"):
        c.fit(65_536, 1, reserve_bytes=-1)
    # the plan's pool sizing (drivers per pool, S4_DEFAULTS.pools.s_att) is a LOWER bound on the served S; the plan
    # carries no served S of its own (pool_rows: gpu, gpus, window, chunk, s_att), so S0's gates stay consistent
    spec = importlib.util.spec_from_file_location("_seedrep_launch_plan_for_pool_test", REPO / "scripts/modal/seedrep_launch_plan.py")
    plan = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plan          # its dataclasses resolve string annotations through sys.modules[cls.__module__]
    try:
        spec.loader.exec_module(plan)
    finally:
        sys.modules.pop(spec.name, None)
    assert pool.MAX_NUM_SEQS >= int(plan.S4_DEFAULTS["pools"]["s_att"]) == 10
    assert plan.S4_DEFAULTS["pools"]["gpu"] == pool.P.gpu == "H200" and plan.S4_DEFAULTS["pools"]["max_model_len"] == pool.MAX_MODEL_LEN
    rows = plan.pool_rows({**plan.S4_DEFAULTS, "pools": plan.S4_DEFAULTS["pools"]}, "7bd7257", "0123456789abcdef")
    assert all("max_num_seqs" not in r for r in rows) and rows[0]["gpus"] == 1 and rows[0]["s_att"] == 10
