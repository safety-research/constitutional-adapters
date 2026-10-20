"""``scripts/modal/tinycap_compile_cache.py``:
the key changes with every row-shape field; ``restore`` reports ``restored`` and never ``hit``; the verdict is
``hit`` ONLY with the matching ``Using cache directory`` hash AND a ``Directly load the compiled graph(s)`` line
(fixture excerpts in vLLM 0.25.1's real format for cold / warm-matching / warm-mismatching / restored-but-no-load /
disabled); ``persist`` refuses before readiness, after a restore, under ``--no-cache-persist`` and for the smoke /
negative-control shapes; the directory never holds more than 8 inodes at ANY instant (the transient ``.tmp``
included), evicting least-recently-used tarballs; the smoke's served set yields a different key than the full row's.
Everything runs on a temp "volume"; no Modal call, no real cache."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts/modal/tinycap_compile_cache.py"


@pytest.fixture(scope="module")
def cc():
    spec = importlib.util.spec_from_file_location("_tinycap_compile_cache_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# fixtures: the row shapes and the log excerpts                                #
# --------------------------------------------------------------------------- #

#: The qwen s1 full row as the b64 solo logs stated it (`non-default args` of a real Modal standup): 16384 × 31,
#: `max_num_batched_tokens` 507904 = S × L, rank 1, `o_proj`; here with the 8 fitted cells + the zero adapter.
QWEN_ROW = dict(model_key="qwen35_4b", vllm_pin="0.25.1", max_loras=9, max_lora_rank=1,
                max_model_len=16384, max_num_seqs=31, lora_target_modules=["o_proj"])
HASH = "d26cc59120"

#: Real-format lines (vLLM 0.25.1, `(EngineCore pid=…) INFO … [backends.py:N] …`); the digits are synthetic.
def _line(src: str, msg: str) -> str:
    return f"(EngineCore pid=104) INFO 09-28 01:33:43 [{src}] {msg}"


def cold_log(root="/tmp/compile_cache/vllm", h=HASH) -> str:
    return "\n".join([
        "[sisteer] vllm _MAX_TEMP 0.01 -> 0.001 (BFCL protocol, D13)",
        "(APIServer pid=28) INFO 09-28 01:32:47 [api_utils.py:273] non-default args: {'max_loras': 9, 'max_lora_rank': 1}",
        _line("backends.py:1089", f"Using cache directory: {root}/torch_compile_cache/{h}/rank_0_0/backbone for vLLM's torch.compile"),
        _line("backends.py:378", "Cache the graph of compile range (1, 507904) for later use"),
        _line("backends.py:393", "Compiling a graph for compile range (1, 507904) takes 122.40 s"),
        "(EngineCore pid=104) INFO 09-28 01:46:57 [gpu_model_runner.py:6707] Graph capturing finished in 10 secs, took 0.30 GiB",
        "",
    ])


def warm_log(root="/tmp/compile_cache/vllm", h=HASH) -> str:
    return "\n".join([
        _line("backends.py:1089", f"Using cache directory: {root}/torch_compile_cache/{h}/rank_0_0/backbone for vLLM's torch.compile"),
        _line("backends.py:293", "Directly load the compiled graph(s) for compile range (1, 507904) from the cache, took 2.317 s"),
        "(EngineCore pid=104) INFO 09-28 01:36:57 [gpu_model_runner.py:6707] Graph capturing finished in 10 secs, took 0.30 GiB",
        "",
    ])


def disabled_log() -> str:
    return _line("backends.py:1087", "vLLM's torch.compile cache is disabled.") + "\n"


def _local_cache(root: Path, h: str = HASH, *, payload: bytes = b"graph-bytes") -> Path:
    """A node-local cache as vLLM + Triton + FlashInfer would leave it after a cold standup."""
    d = root / "vllm" / "torch_compile_cache" / h / "rank_0_0" / "backbone"
    d.mkdir(parents=True, exist_ok=True)
    (d / "computation_graph.py").write_text("# graph")
    (d / "artifact.bin").write_bytes(payload)
    (root / "triton" / "abc").mkdir(parents=True, exist_ok=True)
    (root / "triton" / "abc" / "kernel.cubin").write_bytes(b"cubin")
    (root / "flashinfer" / "jit").mkdir(parents=True, exist_ok=True)
    (root / "flashinfer" / "jit" / "gdn.so").write_bytes(b"so")
    return d


def _persist_cold(cc, key, *, local, vol, h=HASH, **kw):
    _local_cache(local, h)
    return cc.compile_cache_persist(key, local_root=local, vol_root=vol, vllm_server_log=cold_log(str(local / "vllm"), h),
                                    restored=False, ready=True, **kw)


# --------------------------------------------------------------------------- #
# the key                                                                      #
# --------------------------------------------------------------------------- #

def test_the_key_is_stable_readable_and_filesystem_safe(cc):
    k1, k2 = cc.cache_key(**QWEN_ROW), cc.cache_key(**QWEN_ROW)
    assert k1 == k2
    assert k1.startswith("qwen35_4b__vllm0.25.1__base__")
    assert len(k1.rsplit("__", 1)[1]) == 16
    assert "/" not in k1 and " " not in k1
    # a model key with characters a file name must not carry is slugged, not passed through
    assert "/" not in cc.cache_key(**{**QWEN_ROW, "model_key": "Qwen/Qwen3.5-4B@851bf6e8"})


@pytest.mark.parametrize("field, other", [
    ("max_loras", 8),                 # the negative-control smoke: one adapter fewer
    ("max_lora_rank", 4),             # lora_config.compute_hash factor
    ("max_num_seqs", 15),             # S: the capture list and max_num_batched_tokens
    ("max_model_len", 8192),          # L: a model_config factor and max_num_batched_tokens
    ("vllm_pin", "0.26.0"),           # __version__ is the first VllmConfig factor
    ("model_key", "gemma4e4b"),
    ("lora_target_modules", ["o_proj", "q_proj"]),
])
def test_the_key_changes_with_every_shape_field(cc, field, other):
    assert cc.cache_key(**{**QWEN_ROW, field: other}) != cc.cache_key(**QWEN_ROW)


def test_the_key_changes_with_image_tag_batched_tokens_and_extra(cc):
    base = cc.cache_key(**QWEN_ROW)
    assert cc.cache_key(**QWEN_ROW, image_tag="nemotron") != base
    assert cc.cache_key(**QWEN_ROW, max_num_batched_tokens=8192) != base
    assert cc.cache_key(**QWEN_ROW, extra={"cudagraph_mode": "PIECEWISE"}) != base


def test_max_num_batched_tokens_defaults_to_s_times_l_as_the_lora_native_launch_composes_it(cc):
    inputs = cc.cache_key_inputs(**QWEN_ROW)
    assert inputs["max_num_batched_tokens"] == 31 * 16384 == 507904  # the real qwen standup's value
    assert cc.cache_key(**QWEN_ROW) == cc.cache_key(**QWEN_ROW, max_num_batched_tokens=507904)
    assert inputs["key_version"] == cc.KEY_VERSION and inputs["lora_target_modules"] == ["o_proj"]


def test_the_smoke_served_set_yields_a_different_key_than_the_full_row(cc):
    """The five-item negative-control smoke drops one fitted cell: 7 adapters + the zero adapter = max_loras 8 against
    the 8-cell row's 9 — a different vLLM config hash, so a different tarball by construction."""
    full = cc.cache_key(**{**QWEN_ROW, "model_key": "gemma4e4b", "max_loras": 8 + 1})
    smoke = cc.cache_key(**{**QWEN_ROW, "model_key": "gemma4e4b", "max_loras": 7 + 1})
    assert full != smoke


@pytest.mark.parametrize("bad", [
    {"max_loras": 0}, {"max_lora_rank": -1}, {"max_num_seqs": None}, {"max_model_len": None},
    {"max_loras": True}, {"max_num_seqs": "31"}, {"model_key": ""}, {"vllm_pin": ""},
])
def test_a_half_known_shape_never_mints_a_key(cc, bad):
    with pytest.raises(ValueError):
        cc.cache_key(**{**QWEN_ROW, **bad})


def test_key_inputs_disagreements_read_the_ready_endpoint(cc):
    inputs = cc.cache_key_inputs(**QWEN_ROW)
    ep = {"lora_config": {"max_loras": 9, "max_lora_rank": 1, "max_cpu_loras": 9, "target_modules": ["o_proj"]}}
    assert cc.key_inputs_disagreements(inputs, ep) == []
    assert cc.key_inputs_disagreements(inputs, {"lora_config": {"max_loras": 8, "max_lora_rank": 1}}) == \
        ["max_loras: key 9 != endpoint 8"]
    assert len(cc.key_inputs_disagreements(inputs, {"lora_config": {"max_loras": 8, "max_lora_rank": 4,
                                                                    "target_modules": ["q_proj"]}})) == 3
    assert cc.key_inputs_disagreements(inputs, {}) == []  # no lora_config: nothing to contradict


# --------------------------------------------------------------------------- #
# the env hook                                                                 #
# --------------------------------------------------------------------------- #

def test_the_env_hook_points_the_three_caches_at_node_local_dirs_and_creates_them(cc, tmp_path):
    local = tmp_path / "cc"
    env = cc.compile_cache_env(local)
    assert env == {"VLLM_CACHE_ROOT": str(local / "vllm"), "TRITON_CACHE_DIR": str(local / "triton"),
                   "FLASHINFER_WORKSPACE_BASE": str(local / "flashinfer")}
    assert all((local / sub).is_dir() for sub in ("vllm", "triton", "flashinfer"))
    assert set(env) == set(cc.CACHE_ENV_VARS)


def test_the_default_local_root_is_under_tmp_never_the_volume(cc):
    assert cc.LOCAL_ROOT_DEFAULT.startswith("/tmp/")
    for v in cc.compile_cache_env(cc.LOCAL_ROOT_DEFAULT, mkdir=False).values():
        assert v.startswith("/tmp/") and not v.startswith("/vol")


def test_with_compile_cache_env_copies_the_standup_env_and_never_mutates_it(cc, tmp_path):
    env = {"PYTHONPATH": "/tmp/tree/src", "SISTEER_LENS_STACKED_REDUCTIONS": "fold"}
    out = cc.with_compile_cache_env(env, tmp_path / "cc")
    assert out["PYTHONPATH"] == "/tmp/tree/src" and out["VLLM_CACHE_ROOT"] == str(tmp_path / "cc" / "vllm")
    assert "VLLM_CACHE_ROOT" not in env


def _contracts_roots() -> dict:
    """``VOL_TINYCAP_ROOT`` / ``VOL_CACHE_ROOT`` read off THIS tree's ``contracts.py`` by AST (the two module-level
    assignments evaluated in order) — not imported: the shared venv's editable ``sisteer`` may resolve to another
    worktree, and the module under test deliberately imports no ``sisteer``."""
    import ast
    src = (REPO / "src/sisteer/capeval/tinycap/contracts.py").read_text()
    ns: dict = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("VOL_TINYCAP_ROOT", "VOL_CACHE_ROOT"):
            exec(ast.get_source_segment(src, node), {}, ns)
    assert set(ns) == {"VOL_TINYCAP_ROOT", "VOL_CACHE_ROOT"}, ns
    return ns


def test_the_volume_root_is_the_contracts_cache_root_plus_compile(cc):
    C = _contracts_roots()
    assert cc.DEFAULT_VOL_ROOT == f"{C['VOL_CACHE_ROOT']}/compile" == "/vol/tinycap/cache/compile"
    assert cc.DEFAULT_VOL_ROOT.startswith(C["VOL_TINYCAP_ROOT"] + "/")


# --------------------------------------------------------------------------- #
# evidence: the two log lines                                                  #
# --------------------------------------------------------------------------- #

def test_the_regexes_match_the_lines_vllm_0_25_1_prints(cc):
    """The literal formats of backends.py 1089–1092 / 292–297 / 377–380 / 392–397 / 1086."""
    assert cc.CACHE_DIR_RE.search(cold_log()).group("dir") == f"/tmp/compile_cache/vllm/torch_compile_cache/{HASH}/rank_0_0/backbone"
    assert cc.CACHE_HASH_RE.search(f"/root/.cache/vllm/torch_compile_cache/{HASH}/rank_0_0/backbone").group("hash") == HASH
    assert cc.CACHE_HASH_RE.search(f"/root/.cache/vllm/torch_compile_cache/{HASH}").group("hash") == HASH
    m = cc.GRAPH_LOAD_RE.search(warm_log())
    assert m.group("range") == "(1, 507904)" and m.group("s") == "2.317"
    m = cc.GRAPH_COMPILE_RE.search(cold_log())
    assert m.group("range") == "(1, 507904)" and m.group("s") == "122.40"
    assert cc.GRAPH_STORE_RE.search(cold_log()) and cc.CACHE_DISABLED_RE.search(disabled_log())
    assert not cc.GRAPH_LOAD_RE.search(cold_log()) and not cc.GRAPH_COMPILE_RE.search(warm_log())


def test_parse_reads_the_facts_and_decides_nothing(cc):
    facts = cc.parse_vllm_log(cold_log())
    assert facts["observed_hash"] == HASH and facts["cache_root"] == "/tmp/compile_cache/vllm"
    assert facts["graphs_loaded"] == 0 and facts["graphs_compiled"] == 1 and facts["graphs_stored"] == 1
    assert facts["compile_s"] == 122.4 and facts["cache_disabled"] is False
    facts = cc.parse_vllm_log(warm_log())
    assert facts["graphs_loaded"] == 1 and facts["load_s"] == 2.317 and facts["graphs_compiled"] == 0
    assert cc.parse_vllm_log("")["observed_hash"] is None and cc.parse_vllm_log(None)["graphs_loaded"] == 0


def test_a_cold_start_is_never_a_hit_even_though_it_prints_using_cache_directory(cc):
    v = cc.compile_cache_evidence(cold_log(), "k", restored=False, expected_hash=HASH)
    assert v["hit"] is False and v["class"] == "cold" and v["reason"] == "not_restored"
    assert v["observed_hash"] == HASH and v["graphs_loaded"] == 0 and v["graphs_compiled"] == 1


def test_restored_with_matching_hash_and_a_graph_load_line_is_a_hit(cc):
    v = cc.compile_cache_evidence(warm_log(), "k", restored=True, expected_hash=HASH)
    assert v["hit"] is True and v["class"] == "hit" and v["reason"] == "hit"
    assert v["graphs_loaded"] == 1 and v["load_s"] == 2.317 and v["observed_hash"] == v["expected_hash"] == HASH
    assert v["cache_dir"].endswith(f"/torch_compile_cache/{HASH}/rank_0_0/backbone")


def test_restored_but_a_different_hash_is_a_miss_named_hash_mismatch(cc):
    """The tarball unpacked, but vLLM computed another config hash (a shape field the key did not fold in, a moved
    code hash) — the loaded-graph line is for a directory that came from THIS container's compile, not ours."""
    v = cc.compile_cache_evidence(warm_log(h="0123456789"), "k", restored=True, expected_hash=HASH)
    assert v["hit"] is False and v["class"] == "restored_miss" and v["reason"] == "hash_mismatch"
    assert v["observed_hash"] == "0123456789" and v["graphs_loaded"] == 1


def test_restored_with_matching_hash_but_no_graph_load_line_is_a_miss(cc):
    v = cc.compile_cache_evidence(cold_log(), "k", restored=True, expected_hash=HASH)
    assert v["hit"] is False and v["class"] == "restored_miss" and v["reason"] == "no_graph_load_evidence"
    assert v["graphs_compiled"] == 1


def test_a_disabled_cache_and_a_log_without_the_line_are_named(cc):
    assert cc.compile_cache_evidence(disabled_log(), "k", restored=True, expected_hash=HASH)["reason"] == "cache_disabled"
    v = cc.compile_cache_evidence("(EngineCore pid=1) INFO nothing about caches\n", "k", restored=True, expected_hash=HASH)
    assert v["hit"] is False and v["reason"] == "no_cache_dir_line" and v["observed_hash"] is None


def test_restored_without_a_recorded_hash_cannot_be_attested_a_hit(cc):
    """An adopted orphan tarball (index lost) restores but has no recorded hash: the graphs may load, the verdict
    stays honest — ``no_expected_hash``, with ``graphs_loaded`` on record for the analysis."""
    v = cc.compile_cache_evidence(warm_log(), "k", restored=True, expected_hash=None)
    assert v["hit"] is False and v["reason"] == "no_expected_hash" and v["graphs_loaded"] == 1


def test_evidence_reads_a_log_path_and_looks_the_expected_hash_up_in_the_index(cc, tmp_path):
    vol, local = tmp_path / "vol", tmp_path / "local"
    key = cc.cache_key(**QWEN_ROW)
    assert _persist_cold(cc, key, local=local, vol=vol)["persisted"]
    log = tmp_path / "vllm_server.log"
    log.write_text(warm_log(str(local / "vllm")))
    v = cc.compile_cache_evidence(log, key, restored=True, vol_root=vol, local_root=local)
    assert v["hit"] is True and v["expected_hash"] == HASH and v["root_matches"] is True
    # the same by str path, and a root that is NOT the hook's says so
    assert cc.compile_cache_evidence(str(log), key, restored=True, vol_root=vol)["hit"] is True
    v = cc.compile_cache_evidence(warm_log("/root/.cache/vllm"), key, restored=True, vol_root=vol, local_root=local)
    assert v["hit"] is True and v["root_matches"] is False  # the hash matched but VLLM_CACHE_ROOT was not honoured


# --------------------------------------------------------------------------- #
# restore / persist round trip                                                 #
# --------------------------------------------------------------------------- #

def test_restore_on_an_empty_volume_reports_a_miss_and_still_hands_out_the_env(cc, tmp_path):
    vol, local = tmp_path / "vol", tmp_path / "local"
    r = cc.compile_cache_restore("k", vol_root=vol, local_root=local)
    assert r["restored"] is False and r["reason"] == "no_index" and "hit" not in r
    assert r["env"]["VLLM_CACHE_ROOT"] == str(local / "vllm") and (local / "vllm").is_dir()
    assert cc.inode_count(vol) == 0


def test_persist_then_restore_round_trips_the_three_caches_and_the_index_records_the_hash_dir(cc, tmp_path):
    vol, local, local2 = tmp_path / "vol", tmp_path / "local", tmp_path / "local2"
    key = cc.cache_key(**QWEN_ROW)
    inputs = cc.cache_key_inputs(**QWEN_ROW)
    p = _persist_cold(cc, key, local=local, vol=vol, inputs=inputs, bring_up_s=822.0,
                      endpoint={"lora_config": {"max_loras": 9, "max_lora_rank": 1}})
    assert p["persisted"] is True and p["skipped"] is None and p["files"] == 4 and p["vllm_hash_dir"] == HASH
    assert p["inodes"] == 2 == cc.inode_count(vol)  # index.json + one tarball, nothing else
    assert sorted(x.name for x in vol.iterdir()) == ["index.json", f"{key}.tar.gz"]
    entry = cc.index_entry(vol, key)
    assert entry["vllm_hash_dir"] == HASH and entry["sha256"] == p["sha256"] and entry["inputs"] == inputs
    assert entry["bring_up_s_cold"] == 822.0 and entry["graphs_compiled"] == 1 and entry["compile_s"] == 122.4
    # the tarball's members are the three roots, relative
    with tarfile.open(p["tarball"]) as t:
        names = t.getnames()
    assert f"vllm/torch_compile_cache/{HASH}/rank_0_0/backbone/artifact.bin" in names
    assert "triton/abc/kernel.cubin" in names and "flashinfer/jit/gdn.so" in names
    assert not any(n.startswith("/") or ".." in n for n in names)

    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local2)
    assert r["restored"] is True and r["reason"] == "restored" and r["sha256"] == p["sha256"]
    assert r["bytes"] == p["bytes"] > 0 and r["files"] == 4 and r["expected_vllm_hash_dir"] == HASH
    assert "hit" not in r  # restored is not hit
    assert (local2 / "vllm" / "torch_compile_cache" / HASH / "rank_0_0" / "backbone" / "artifact.bin").read_bytes() == b"graph-bytes"
    assert (local2 / "triton" / "abc" / "kernel.cubin").exists() and (local2 / "flashinfer" / "jit" / "gdn.so").exists()
    assert r["env"]["VLLM_CACHE_ROOT"] == str(local2 / "vllm")
    # the whole warm path: the standup then prints the recorded hash and loads its graphs
    v = cc.compile_cache_evidence(warm_log(str(local2 / "vllm")), key, restored=r["restored"], vol_root=vol)
    assert v["hit"] is True


def test_a_tampered_tarball_is_refused_by_sha256_and_nothing_is_extracted(cc, tmp_path):
    vol, local, local2 = tmp_path / "vol", tmp_path / "local", tmp_path / "local2"
    key = cc.cache_key(**QWEN_ROW)
    p = _persist_cold(cc, key, local=local, vol=vol)
    Path(p["tarball"]).write_bytes(b"not a tarball")
    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local2)
    assert r["restored"] is False and r["reason"] == "sha256_mismatch" and r["expected_sha256"] == p["sha256"]
    assert not any((local2 / "vllm").iterdir())


def test_an_index_entry_whose_tarball_vanished_and_a_corrupt_index_are_named(cc, tmp_path):
    vol, local, local2 = tmp_path / "vol", tmp_path / "local", tmp_path / "local2"
    key = cc.cache_key(**QWEN_ROW)
    p = _persist_cold(cc, key, local=local, vol=vol)
    Path(p["tarball"]).unlink()
    assert cc.compile_cache_restore(key, vol_root=vol, local_root=local2)["reason"] == "tarball_missing"
    assert cc.compile_cache_restore("other", vol_root=vol, local_root=local2)["reason"] == "no_tarball"
    (vol / "index.json").write_text("{not json")
    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local2)
    assert r["restored"] is False and r["reason"] == "index_unreadable"
    assert cc.index_entry(vol, key) is None


def test_a_tarball_member_that_escapes_the_local_root_is_refused(cc, tmp_path):
    vol, local = tmp_path / "vol", tmp_path / "local"
    vol.mkdir()
    key = "evil"
    tb = vol / f"{key}.tar.gz"
    with tarfile.open(tb, "w:gz") as t:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        t.addfile(info)
    sha = cc._sha256_file(tb)
    (vol / "index.json").write_text(json.dumps({"version": 1, "entries": {key: {"tarball": tb.name, "sha256": sha, "vllm_hash_dir": HASH}}}))
    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local)
    assert r["restored"] is False and r["reason"] == "tarball_unsafe"
    assert not (tmp_path / "escape.txt").exists()


def test_restore_touches_the_lru_clock_and_persist_after_a_restore_refuses(cc, tmp_path):
    vol, local, local2 = tmp_path / "vol", tmp_path / "local", tmp_path / "local2"
    key = cc.cache_key(**QWEN_ROW)
    _persist_cold(cc, key, local=local, vol=vol)
    before = cc.index_entry(vol, key)
    time.sleep(0.01)
    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local2)
    after = cc.index_entry(vol, key)
    assert after["last_used_at"] > before["last_used_at"] and after["written_at"] == before["written_at"]
    # a hit never rewrites (and a restored-but-miss is a finding, not a rewrite)
    p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=warm_log(str(local2 / "vllm")),
                                 restored=r["restored"], ready=True)
    assert p["persisted"] is False and p["skipped"] == "restored"
    assert cc.index_entry(vol, key)["written_at"] == before["written_at"] and cc.inode_count(vol) == 2


# --------------------------------------------------------------------------- #
# the no-persist rule                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kw, reason", [
    (dict(cache_persist=False), "no_cache_persist"),        # --no-cache-persist
    (dict(inspect_limit=5), "smoke_shape"),                 # any --inspect-limit run
    (dict(negative_control={"cell": "x", "flip": "last_hex"}), "negative_control"),
    (dict(ready=False), "not_ready"),                       # readiness not attested
    (dict(restored=True), "restored"),                      # a hit never rewrites
])
def test_persist_refusal_names_the_first_reason(cc, kw, reason):
    base = dict(restored=False, ready=True)
    assert cc.persist_refusal(**{**base, **kw}) == reason
    assert cc.persist_refusal(**base) is None
    assert cc.persist_refusal(**base, inspect_limit=0) is None  # 0 / None = the full run


@pytest.mark.parametrize("kw, reason", [
    (dict(cache_persist=False), "no_cache_persist"),
    (dict(inspect_limit=5), "smoke_shape"),
    (dict(negative_control={"cell": "constv2ctl2-full__lora_r4_ce", "flip": "last_hex"}), "negative_control"),
    (dict(ready=False), "not_ready"),
])
def test_a_smoke_or_negative_control_or_unready_row_writes_nothing_to_the_volume(cc, tmp_path, kw, reason):
    vol, local = tmp_path / "vol", tmp_path / "local"
    _local_cache(local)
    p = cc.compile_cache_persist("k", local_root=local, vol_root=vol, vllm_server_log=cold_log(str(local / "vllm")),
                                 restored=False, **{**dict(ready=True), **kw})
    assert p["persisted"] is False and p["skipped"] == reason
    assert not vol.exists() or cc.inode_count(vol) == 0


def test_persist_refuses_when_the_log_names_no_cache_dir_or_the_cache_is_disabled(cc, tmp_path):
    vol, local = tmp_path / "vol", tmp_path / "local"
    _local_cache(local)
    p = cc.compile_cache_persist("k", local_root=local, vol_root=vol, vllm_server_log="no such line\n", restored=False)
    assert p["skipped"] == "no_cache_dir_line" and p["persisted"] is False
    p = cc.compile_cache_persist("k", local_root=local, vol_root=vol, vllm_server_log=disabled_log(), restored=False)
    assert p["skipped"] == "cache_disabled"
    assert cc.inode_count(vol) == 0


def test_persist_refuses_when_the_named_hash_dir_is_not_under_the_hooks_root(cc, tmp_path):
    """The log says vLLM wrote to `/root/.cache/vllm/...` (the hook's VLLM_CACHE_ROOT was not honoured): the
    node-local root holds no such directory, so a tarball of it would be empty — refused and named."""
    vol, local = tmp_path / "vol", tmp_path / "local"
    cc.compile_cache_env(local)  # empty dirs, as after a miss whose standup ignored the env
    p = cc.compile_cache_persist("k", local_root=local, vol_root=vol, vllm_server_log=cold_log("/root/.cache/vllm"), restored=False)
    assert p["persisted"] is False and p["skipped"] == "local_cache_dir_missing"
    assert "/root/.cache/vllm" in p["refusal_detail"] and cc.inode_count(vol) == 0


def test_persist_refuses_when_the_ready_endpoint_contradicts_the_key(cc, tmp_path):
    vol, local = tmp_path / "vol", tmp_path / "local"
    key, inputs = cc.cache_key(**QWEN_ROW), cc.cache_key_inputs(**QWEN_ROW)
    p = _persist_cold(cc, key, local=local, vol=vol, inputs=inputs,
                      endpoint={"lora_config": {"max_loras": 8, "max_lora_rank": 1}})
    assert p["persisted"] is False and p["skipped"] == "key_inputs_disagree"
    assert p["refusal_detail"] == ["max_loras: key 9 != endpoint 8"] and cc.inode_count(vol) == 0


def test_persist_on_a_corrupt_index_refuses_rather_than_guessing(cc, tmp_path):
    vol, local = tmp_path / "vol", tmp_path / "local"
    vol.mkdir()
    (vol / "index.json").write_text("{not json")
    p = _persist_cold(cc, "k", local=local, vol=vol)
    assert p["persisted"] is False and p["skipped"] == "index_unreadable"
    assert sorted(x.name for x in vol.iterdir()) == ["index.json"]


# --------------------------------------------------------------------------- #
# the inode bound                                                              #
# --------------------------------------------------------------------------- #

def test_the_bound_constants(cc):
    assert cc.MAX_INODES == 8 and cc.MAX_TARBALLS == 6  # index + 6 tarballs + ONE transient .tmp = 8


def test_twelve_persists_never_take_the_volume_past_eight_inodes_at_any_instant(cc, tmp_path, monkeypatch):
    """Twelve distinct row shapes persisted in turn: steady state ≤ 7 (index + 6), and the count observed at EVERY
    ``os.replace`` (the moment the transient ``.tmp`` coexists with the rest) is ≤ 8."""
    vol = tmp_path / "vol"
    seen = []
    real_replace = os.replace

    def counting_replace(src, dst):
        seen.append(cc.inode_count(vol))
        return real_replace(src, dst)

    monkeypatch.setattr(cc.os, "replace", counting_replace)
    keys = []
    for i in range(12):
        key = cc.cache_key(**{**QWEN_ROW, "max_num_seqs": 10 + i})
        keys.append(key)
        p = _persist_cold(cc, key, local=tmp_path / f"local{i}", vol=vol, h=f"{i:010x}")
        assert p["persisted"] is True
        assert cc.inode_count(vol) <= cc.MAX_INODES - 1, f"steady state after persist {i}: {cc.inode_count(vol)}"
    assert seen and max(seen) <= cc.MAX_INODES, f"transient inode counts at os.replace: {seen}"
    entries = cc.read_index(vol)["entries"]
    assert len(entries) == cc.MAX_TARBALLS == 6 and set(entries) == set(keys[-6:])
    assert sorted(x.name for x in vol.iterdir()) == sorted(["index.json", *(f"{k}.tar.gz" for k in keys[-6:])])
    assert not any(x.name.endswith(".tmp") for x in vol.iterdir())


def test_eviction_is_least_recently_used_and_a_restore_counts_as_a_use(cc, tmp_path):
    vol = tmp_path / "vol"
    keys = [cc.cache_key(**{**QWEN_ROW, "max_num_seqs": 10 + i}) for i in range(6)]
    for i, k in enumerate(keys):
        assert _persist_cold(cc, k, local=tmp_path / f"local{i}", vol=vol, h=f"{i:010x}")["persisted"]
        time.sleep(0.005)
    # touch the OLDEST by restoring it: it becomes the most recently used
    assert cc.compile_cache_restore(keys[0], vol_root=vol, local_root=tmp_path / "r0")["restored"]
    time.sleep(0.005)
    k7 = cc.cache_key(**{**QWEN_ROW, "max_num_seqs": 99})
    p = _persist_cold(cc, k7, local=tmp_path / "local7", vol=vol, h="aaaaaaaaaa")
    assert p["persisted"] and p["evicted"] == [keys[1]]  # the least recently USED, not the oldest written
    entries = cc.read_index(vol)["entries"]
    assert keys[0] in entries and keys[1] not in entries and k7 in entries and len(entries) == 6
    assert not (vol / f"{keys[1]}.tar.gz").exists()


def test_re_persisting_an_existing_key_replaces_in_place_without_eviction(cc, tmp_path):
    vol = tmp_path / "vol"
    keys = [cc.cache_key(**{**QWEN_ROW, "max_num_seqs": 10 + i}) for i in range(6)]
    for i, k in enumerate(keys):
        _persist_cold(cc, k, local=tmp_path / f"local{i}", vol=vol, h=f"{i:010x}")
    p = _persist_cold(cc, keys[2], local=tmp_path / "again", vol=vol, h="0000000002")
    assert p["persisted"] and p["evicted"] == [] and cc.inode_count(vol) == 7
    assert set(cc.read_index(vol)["entries"]) == set(keys)


def test_orphan_tarballs_and_stale_tmps_are_reconciled_into_the_bound(cc, tmp_path):
    """A lost index update (two row containers persisting at once) must never leave an uncounted inode: an
    unindexed tarball is adopted (restorable, never a hit until re-persisted) and counts toward the bound; a stale
    ``.tmp`` from a crashed write is removed; an entry whose tarball is gone is dropped."""
    vol = tmp_path / "vol"
    keys = [cc.cache_key(**{**QWEN_ROW, "max_num_seqs": 10 + i}) for i in range(6)]
    for i, k in enumerate(keys):
        _persist_cold(cc, k, local=tmp_path / f"local{i}", vol=vol, h=f"{i:010x}")
    # drop two entries from the index (a lost update), leave their tarballs; add a stale tmp; delete one tarball
    idx = cc.read_index(vol)
    for k in keys[:2]:
        del idx["entries"][k]
    (vol / "index.json").write_text(json.dumps(idx))
    (vol / "stale.tar.gz.tmp").write_bytes(b"x")
    (vol / f"{keys[5]}.tar.gz").unlink()
    assert cc.inode_count(vol) == 7  # index + 5 remaining tarballs + the stale tmp: the state a crash leaves
    k7 = cc.cache_key(**{**QWEN_ROW, "max_num_seqs": 77})
    p = _persist_cold(cc, k7, local=tmp_path / "local7", vol=vol, h="bbbbbbbbbb")
    assert p["persisted"]
    rec = p["reconciled"]
    assert rec["removed_tmp"] == ["stale.tar.gz.tmp"] and sorted(rec["adopted"]) == sorted(keys[:2]) and rec["dropped"] == [keys[5]]
    entries = cc.read_index(vol)["entries"]
    assert len(entries) <= 6 and cc.inode_count(vol) <= 7 and k7 in entries
    adopted_left = [k for k in keys[:2] if k in entries]
    for k in adopted_left:
        assert entries[k]["adopted"] is True and entries[k]["vllm_hash_dir"] is None
    # an adopted entry restores, but its evidence can never be a hit until it is re-persisted with a hash
    if adopted_left:
        r = cc.compile_cache_restore(adopted_left[0], vol_root=vol, local_root=tmp_path / "ra")
        assert r["restored"] is True and r["expected_vllm_hash_dir"] is None
        assert cc.compile_cache_evidence(warm_log(), adopted_left[0], restored=True, vol_root=vol)["reason"] == "no_expected_hash"


def test_nothing_under_the_volume_root_is_ever_an_extracted_tree(cc, tmp_path):
    """Rule 3 (caches never file-for-file on the volume): after a persist and a restore the volume directory holds
    files only — no directory, ever."""
    vol, local, local2 = tmp_path / "vol", tmp_path / "local", tmp_path / "local2"
    key = cc.cache_key(**QWEN_ROW)
    _persist_cold(cc, key, local=local, vol=vol)
    cc.compile_cache_restore(key, vol_root=vol, local_root=local2)
    assert all(x.is_file() for x in vol.iterdir())
    assert cc.inode_count(vol) == 2


# --------------------------------------------------------------------------- #
# the env_hash factor (warm_cache_miss_diagnosis 2026-09-29): VLLM_API_KEY      #
# --------------------------------------------------------------------------- #

#: A ``cache_key_factors.json`` ``env`` block as vLLM 0.25.1 writes it (``backends.py`` 1116–1127: the raw
#: ``envs.compile_factors()`` dict) — a handful of the 273 factors, synthetic values, the token among them.
def _env_block(token, **over) -> dict:
    return {"VLLM_API_KEY": token, "VLLM_USE_V2_MODEL_RUNNER": True, "VLLM_BATCH_INVARIANT": False,
            "VLLM_ATTENTION_BACKEND": None, "VLLM_FLASHINFER_WORKSPACE_BASE": None, "TORCH_CUDA_ARCH_LIST": None,
            "VLLM_XLA_CACHE_PATH": "/tmp/compile_cache/vllm/xla_cache", **over}


def test_two_env_blocks_differing_only_in_the_api_key_hash_equal_once_the_worker_has_dropped_it(cc):
    """The miss, modelled: vLLM 0.25.1's ``env_hash`` is ``hash_factors(compile_factors())`` — a sha256 over the
    sorted JSON of every known env var's VALUE minus an ignore list that does NOT contain ``VLLM_API_KEY`` — and
    the launcher mints a fresh random token per launch, so the probe's and the row's blocks differed in exactly
    that entry and hashed apart (``485efd7135`` vs ``71450296d1``). With the worker extension's import popping the
    variable before the first compile (``native_worker``, fix F1) the getter returns ``None`` in both, and the two
    blocks hash EQUAL; a real factor difference still hashes apart."""
    probe, row = _env_block("tokA-" + "a" * 38), _env_block("tokB-" + "b" * 38)
    assert probe != row and cc.hash_env_factors(probe) != cc.hash_env_factors(row)   # the 0.25.1 miss
    scrubbed = [cc.env_factors_as_the_worker_compiles(b) for b in (probe, row)]
    assert scrubbed[0] == scrubbed[1] and scrubbed[0]["VLLM_API_KEY"] is None
    assert cc.hash_env_factors(scrubbed[0]) == cc.hash_env_factors(scrubbed[1])
    # a block whose token was already absent (post-F1) hashes the same as a scrubbed one
    assert cc.hash_env_factors(_env_block(None)) == cc.hash_env_factors(scrubbed[0])
    # the scrub touches ONLY the token: a real factor difference is still a different hash
    other = cc.env_factors_as_the_worker_compiles(_env_block("tokA", VLLM_USE_V2_MODEL_RUNNER=False))
    assert cc.hash_env_factors(other) != cc.hash_env_factors(scrubbed[0])
    # the digest is vLLM's own (config/utils.py 365–367): sha256 of json.dumps(items, sort_keys=True)
    import hashlib
    assert cc.hash_env_factors(probe) == hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()


def _serving_module_constant(rel: str, name: str):
    """A module-level assignment read off THIS tree by AST (the module under test imports no ``sisteer``): a
    literal's value, or — for an alias of another module's constant — its dotted source (``"<alias>.<attr>"``)."""
    import ast
    src = (REPO / "src/sisteer/capeval/serving" / rel).read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            if isinstance(node.value, ast.Attribute):
                return f"{ast.unparse(node.value.value)}.{node.value.attr}"
            return ast.literal_eval(node.value)
    raise AssertionError(f"{rel} defines no {name}")


def test_the_api_key_env_name_is_the_one_the_launcher_injects_and_the_worker_drops(cc):
    """Three spellings, one variable: the launcher's / guards' ``vllm_auth.TOKEN_ENV`` (a literal), the worker's
    ``native_worker.TOKEN_ENV`` (an alias OF that literal — so it cannot drift), and this module's."""
    assert cc.VLLM_API_KEY_ENV == _serving_module_constant("vllm_auth.py", "TOKEN_ENV") == "VLLM_API_KEY"
    assert _serving_module_constant("native_worker.py", "TOKEN_ENV").endswith("vllm_auth.TOKEN_ENV")
    assert cc.VLLM_API_KEY_ENV in cc.CREDENTIAL_ENV_FACTORS


# --------------------------------------------------------------------------- #
# the key follows the SERVED geometry (diagnosis gap 2)                        #
# --------------------------------------------------------------------------- #

#: The qwen s1 row as the c9513c2 records state it: the key minted BEFORE Popen from the CONFIGURED shape
#: (S 45, 45 × 16384 = 737,280 batched) while the composer served the kernel-ceiling-lowered shape (S 31,
#: 507,904), recorded in ``geometry.adjustments`` — equal across the probe and the row, so not THAT miss's cause;
#: the step is a pure function of the key's fields, so the key names the served graph iff the record explains it.
QWEN_CONFIGURED = dict(QWEN_ROW, max_num_seqs=45, max_loras=7, max_lora_rank=8)
QWEN_SERVED_ENDPOINT = {
    "geometry": {"eval": None, "max_model_len": 16384, "max_num_seqs": 31, "batched_tokens": 507904,
                 "budget": 741888, "model_key": "qwen35_4b",
                 "adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 45, "to_S": 31, "D": 4096,
                                  "ceiling": 524287}]},
    "lora_config": {"fully_sharded_loras": False, "lora_dtype": "torch.bfloat16", "max_cpu_loras": 7,
                    "max_lora_rank": 8, "max_loras": 7, "target_modules": ["o_proj"]},
    "attested_max_num_seqs": 31, "attested_max_model_len": 16384,
}


def test_key_version_is_v2_so_no_landing_1_tarball_answers_a_new_key(cc):
    """The fix changes vLLM's ``env_hash`` (the token is no longer a factor), so every tarball persisted before it
    recorded a ``vllm_hash_dir`` no post-fix standup can print — a restore of one is a ``restored_miss`` by
    construction, and a restore REFUSES the persist (``restored``), so under the old key the shape could never be
    re-warmed. A v1 key never equals a v2 key for the same inputs: the four v1 tarballs are unreachable."""
    assert cc.KEY_VERSION == "v2"
    inputs = cc.cache_key_inputs(**QWEN_ROW)
    assert inputs["key_version"] == "v2"
    assert cc.cache_key_of({**inputs, "key_version": "v1"}) != cc.cache_key_of(inputs)
    v1_qwen = cc.cache_key_of({**cc.cache_key_inputs(**QWEN_CONFIGURED), "key_version": "v1"})
    assert v1_qwen != cc.cache_key(**QWEN_CONFIGURED)


def test_key_inputs_disagreements_hold_the_key_to_the_served_geometry_through_the_recorded_adjustment(cc):
    """The endpoint's ``geometry`` block is what the composer LAUNCHED after the LoRA kernel ceiling step — the
    ``SchedulerConfig`` / ``CompilationConfig`` / ``ModelConfig`` factors vLLM hashed. The key is minted before
    Popen from the configured shape, and the step is a pure function of the key's fields (the model's pinned D, L,
    S), so the served shape AGREES with the key when the endpoint's recorded ``adjustments`` chain leads from the
    key's S to it (the c9513c2 qwen row: 45 -> 31, 737,280 -> 507,904); anything else is named per field."""
    configured = cc.cache_key_inputs(**QWEN_CONFIGURED)
    assert cc.key_inputs_disagreements(configured, QWEN_SERVED_ENDPOINT) == []
    served_key = cc.cache_key_inputs(**{**QWEN_CONFIGURED, "max_num_seqs": 31})
    # a key minted from the served shape itself agrees (no chain needed when nothing moved) ...
    flat = {**QWEN_SERVED_ENDPOINT, "geometry": {**QWEN_SERVED_ENDPOINT["geometry"], "adjustments": []}}
    assert cc.key_inputs_disagreements(served_key, flat) == []
    # ... but the configured key against a served 31 with NO recorded step is a disagreement: a KV-derived S, or a
    # lowering the composer did not record, is not a function of the key
    assert cc.key_inputs_disagreements(configured, flat) == [
        "max_num_seqs: key 45 != endpoint 31 (no recorded adjustment from the key's shape)",
        "max_num_batched_tokens: key 737280 != endpoint 507904 (no recorded adjustment from the key's shape)",
    ]
    # a served S the recorded chain does not reach
    g27 = {**QWEN_SERVED_ENDPOINT["geometry"], "max_num_seqs": 27, "batched_tokens": 27 * 16384}
    assert cc.key_inputs_disagreements(configured, {**QWEN_SERVED_ENDPOINT, "geometry": g27}) == [
        "max_num_seqs: key 45 != endpoint 27 (recorded adjustment lora_kernel_index_ceiling: S 45 -> 31 does not explain it)",
        "max_num_batched_tokens: key 737280 != endpoint 442368 (recorded adjustment lora_kernel_index_ceiling: S 45 -> 31 does not explain it)",
    ]
    # a chain that does not START at the key's S explains nothing
    g_other = {**QWEN_SERVED_ENDPOINT["geometry"],
               "adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 40, "to_S": 31, "D": 4096}]}
    assert len(cc.key_inputs_disagreements(configured, {**QWEN_SERVED_ENDPOINT, "geometry": g_other})) == 2
    # two recorded steps chain, each from where the last left off
    g_two = {**QWEN_SERVED_ENDPOINT["geometry"],
             "adjustments": [{"kind": "kv_derived_seqs", "from_S": 45, "to_S": 40},
                             {"kind": "lora_kernel_index_ceiling", "from_S": 40, "to_S": 31, "D": 4096}]}
    assert cc.key_inputs_disagreements(configured, {**QWEN_SERVED_ENDPOINT, "geometry": g_two}) == []
    # a key whose budget is not the composed S x L cannot be explained by an S step
    explicit = cc.cache_key_inputs(**QWEN_CONFIGURED, max_num_batched_tokens=600000)
    assert [d.split(":")[0] for d in cc.key_inputs_disagreements(explicit, QWEN_SERVED_ENDPOINT)] == \
        ["max_num_seqs", "max_num_batched_tokens"]
    # L is never adjusted: a served L that differs is plainly named; vLLM's spelling of the budget is read too
    ep = {"geometry": {"max_model_len": 8192, "max_num_seqs": 31, "max_num_batched_tokens": 507904}}
    assert cc.key_inputs_disagreements(served_key, ep) == ["max_model_len: key 16384 != endpoint 8192"]
    # a geometry block without the fields, or no geometry at all, contradicts nothing
    assert cc.key_inputs_disagreements(configured, {"geometry": {"budget": 1}}) == []
    assert cc.key_inputs_disagreements(configured, {"lora_config": {"max_loras": 7, "max_lora_rank": 8}}) == []
    # the LoRA fields still count, and come first
    ep = {**QWEN_SERVED_ENDPOINT, "lora_config": {**QWEN_SERVED_ENDPOINT["lora_config"], "max_loras": 8}}
    assert cc.key_inputs_disagreements(configured, ep) == ["max_loras: key 7 != endpoint 8"]
    assert cc.key_inputs_disagreements(configured, {**ep, "geometry": g27})[0] == "max_loras: key 7 != endpoint 8"


def test_served_geometry_reads_the_endpoint_block_and_nothing_else(cc):
    assert cc.served_geometry(QWEN_SERVED_ENDPOINT) == {
        "max_num_seqs": 31, "max_num_batched_tokens": 507904, "max_model_len": 16384,
        "adjustments": [{"kind": "lora_kernel_index_ceiling", "from_S": 45, "to_S": 31, "D": 4096, "ceiling": 524287}]}
    assert cc.served_geometry({"geometry": {"max_num_seqs": 31}}) == \
        {"max_num_seqs": 31, "max_num_batched_tokens": None, "max_model_len": None, "adjustments": []}
    assert cc.served_geometry({"lora_config": {}}) is None and cc.served_geometry({}) is None


def test_persist_files_the_configured_key_when_the_served_shape_is_its_recorded_adjustment_and_refuses_otherwise(cc, tmp_path):
    """The c9513c2 qwen key (configured S 45) against its own served endpoint (S 31 under the recorded ceiling
    step) persists — the index entry carries the key's ``inputs`` AND the ``served_geometry`` the graph was
    compiled for; the same key against an endpoint serving a shape the record does not explain is refused
    ``key_inputs_disagree`` by name and the volume is unchanged."""
    vol, local = tmp_path / "vol", tmp_path / "local"
    key, inputs = cc.cache_key(**QWEN_CONFIGURED), cc.cache_key_inputs(**QWEN_CONFIGURED)
    p = _persist_cold(cc, key, local=local, vol=vol, inputs=inputs, endpoint=QWEN_SERVED_ENDPOINT)
    assert p["persisted"] is True and p["skipped"] is None
    assert p["served_geometry"] == cc.served_geometry(QWEN_SERVED_ENDPOINT)
    entry = cc.index_entry(vol, key)
    assert entry["served_geometry"] == p["served_geometry"] and entry["inputs"]["max_num_seqs"] == 45
    g27 = {**QWEN_SERVED_ENDPOINT["geometry"], "max_num_seqs": 27, "batched_tokens": 27 * 16384}
    p = _persist_cold(cc, key, local=tmp_path / "local2", vol=vol, h="0000000027", inputs=inputs,
                      endpoint={**QWEN_SERVED_ENDPOINT, "geometry": g27})
    assert p["persisted"] is False and p["skipped"] == "key_inputs_disagree"
    assert p["refusal_detail"][0].startswith("max_num_seqs: key 45 != endpoint 27 (recorded adjustment")
    assert cc.index_entry(vol, key)["vllm_hash_dir"] == HASH and cc.inode_count(vol) == 2
    # a persist without an endpoint records no served geometry, and is not refused for lacking one
    p = _persist_cold(cc, "k2", local=tmp_path / "local3", vol=vol, h="0000000002")
    assert p["persisted"] is True and p["served_geometry"] is None and cc.index_entry(vol, "k2")["served_geometry"] is None


# --------------------------------------------------------------------------- #
# the tarball never carries a credential (diagnosis gap 1)                     #
# --------------------------------------------------------------------------- #

def _factors_json(token, **over) -> str:
    return json.dumps({"env": _env_block(token, **over), "config_hash": "c" * 64, "code_hash": "d" * 64,
                       "compiler_hash": "e" * 64}, indent=2, sort_keys=True)


def test_persist_redacts_the_api_key_in_cache_key_factors_json_and_restore_still_hits(cc, tmp_path):
    """vLLM 0.25.1 writes the RAW env factors to ``<local_cache_dir>/cache_key_factors.json`` (``backends.py``
    1116–1127) — the per-launch token among them — and never reads the file back (its only reference in the
    tree is that write, guarded by ``if not os.path.exists``; a hit is decided from ``vllm_compile_cache.py`` and
    the inductor artifacts, ``CompilerManager.initialize_cache`` / ``load``). So the tarball's copy carries the
    key NAME with a redaction marker, the token's bytes are nowhere in the tarball, every other factor and file
    is byte-identical, and the restored tree still hits."""
    vol, local, local2 = tmp_path / "vol", tmp_path / "local", tmp_path / "local2"
    token = "probe-token-" + "x" * 31
    d = _local_cache(local)
    (d / cc.FACTORS_FILE).write_text(_factors_json(token))
    (d / "vllm_compile_cache.py").write_text("{((1, 507904), 0, 'inductor'): {'graph_handle': ('h', 'p'), 'cache_key': 'k'}}")
    key, inputs = cc.cache_key(**QWEN_ROW), cc.cache_key_inputs(**QWEN_ROW)
    p = cc.compile_cache_persist(key, local_root=local, vol_root=vol, vllm_server_log=cold_log(str(local / "vllm")),
                                 restored=False, ready=True, inputs=inputs)
    assert p["persisted"] is True and p["files"] == 6
    member = f"vllm/torch_compile_cache/{HASH}/rank_0_0/backbone/{cc.FACTORS_FILE}"
    assert p["redacted"] == {member: ["VLLM_API_KEY"]}
    assert cc.index_entry(vol, key)["redacted"] == {member: ["VLLM_API_KEY"]}
    with tarfile.open(p["tarball"]) as t:
        shipped = json.loads(t.extractfile(member).read())
        raw = b"".join(t.extractfile(m).read() for m in t.getmembers() if m.isfile())
    assert token.encode() not in raw
    assert shipped["env"]["VLLM_API_KEY"] == cc.REDACTED and "VLLM_API_KEY" in shipped["env"]
    assert {k: v for k, v in shipped["env"].items() if k != "VLLM_API_KEY"} == \
        {k: v for k, v in _env_block(token).items() if k != "VLLM_API_KEY"}
    assert shipped["config_hash"] == "c" * 64 and shipped["code_hash"] == "d" * 64
    # the on-disk original is untouched (the standup's own diagnostics keep the raw file)
    assert json.loads((d / cc.FACTORS_FILE).read_text())["env"]["VLLM_API_KEY"] == token
    # round trip: the redacted copy lands, the cache manifest is byte-identical, the warm path hits
    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local2)
    assert r["restored"] is True and r["files"] == 6
    d2 = local2 / "vllm" / "torch_compile_cache" / HASH / "rank_0_0" / "backbone"
    assert json.loads((d2 / cc.FACTORS_FILE).read_text())["env"]["VLLM_API_KEY"] == cc.REDACTED
    assert (d2 / "vllm_compile_cache.py").read_bytes() == (d / "vllm_compile_cache.py").read_bytes()
    assert cc.compile_cache_evidence(warm_log(str(local2 / "vllm")), key, restored=True, vol_root=vol)["hit"] is True


def test_redaction_covers_every_credential_factor_leaves_a_null_token_alone_and_drops_an_unparseable_file(cc, tmp_path):
    """Post-F1 the compiling process writes ``VLLM_API_KEY: null`` — nothing to redact, and the copy stays a
    faithful record that the worker saw no token. The other two credential getters of 0.25.1's ``envs.py``
    (``S3_ACCESS_KEY_ID`` / ``S3_SECRET_ACCESS_KEY``, the RunAI streamer's) are redacted the same way. A factors
    file that is not JSON is not shipped at all (vLLM rewrites it when absent) and the persist record says so."""
    assert set(cc.CREDENTIAL_ENV_FACTORS) == {"VLLM_API_KEY", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"}
    out, names = cc.redact_factors(_factors_json(None).encode())
    assert names == [] and json.loads(out)["env"]["VLLM_API_KEY"] is None
    out, names = cc.redact_factors(_factors_json("tok", S3_SECRET_ACCESS_KEY="s3s", S3_ACCESS_KEY_ID="").encode())
    env = json.loads(out)["env"]
    assert names == ["S3_SECRET_ACCESS_KEY", "VLLM_API_KEY"]                     # sorted; an empty string is not a secret
    assert env["VLLM_API_KEY"] == env["S3_SECRET_ACCESS_KEY"] == cc.REDACTED and env["S3_ACCESS_KEY_ID"] == ""
    assert cc.redact_factors(b"{not json") == (None, ["unparseable"])
    assert cc.redact_factors(b'{"config_hash": "c"}') == (b'{"config_hash": "c"}', [])   # no env block: as is

    vol, local = tmp_path / "vol", tmp_path / "local"
    d = _local_cache(local)
    (d / cc.FACTORS_FILE).write_text("{not json")
    p = cc.compile_cache_persist("k", local_root=local, vol_root=vol, vllm_server_log=cold_log(str(local / "vllm")),
                                 restored=False, ready=True)
    member = f"vllm/torch_compile_cache/{HASH}/rank_0_0/backbone/{cc.FACTORS_FILE}"
    assert p["persisted"] is True and p["files"] == 4 and p["redacted"] == {member: ["unparseable"]}
    with tarfile.open(p["tarball"]) as t:
        assert member not in t.getnames()


# --------------------------------------------------------------------------- #
# a restored_miss re-persists: no key is a trap                               #
# --------------------------------------------------------------------------- #

STALE, FRESH = HASH, "0000000fee"


def _restored_then_recompiled(cc, tmp_path, *, key, token=None, fresh_payload=b"fresh-graph"):
    """Persist a cold tarball at hash STALE, restore it into a second node-local root, then leave what vLLM leaves
    after a ``restored_miss``: the restored ``STALE/`` tree beside a freshly compiled ``FRESH/`` tree (the
    ``Using cache directory`` hash differs from the record because a factor moved — an image / vLLM / env bump).
    Returns (vol, local2, the pre-replacement index entry)."""
    vol, local1, local2 = tmp_path / "vol", tmp_path / "local1", tmp_path / "local2"
    assert _persist_cold(cc, key, local=local1, vol=vol, h=STALE)["persisted"]
    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local2)
    assert r["restored"] is True and r["expected_vllm_hash_dir"] == STALE
    d = local2 / "vllm" / "torch_compile_cache" / FRESH / "rank_0_0" / "backbone"
    d.mkdir(parents=True)
    (d / "computation_graph.py").write_text("# fresh graph")
    (d / "artifact.bin").write_bytes(fresh_payload)
    if token is not None:
        (d / cc.FACTORS_FILE).write_text(_factors_json(token))
    return vol, local2, cc.index_entry(vol, key)


def test_a_restored_miss_with_a_completed_compile_replaces_the_stale_tarball_under_the_same_key(cc, tmp_path):
    """The trap this closes: a tarball whose recorded ``vllm_hash_dir`` can no longer be printed (the token left
    ``env_hash``; an image or vLLM bump) restores on every row, misses, and — because a restore refused the persist
    — never re-persists. Now: restored AND ``restored_miss`` / ``hash_mismatch`` AND a completed compile → the fresh
    tree REPLACES the stale tarball under the same key (``.tmp`` + ``os.replace``), the index entry carries the new
    hash, ``served_geometry`` and a ``replaced_stale`` block naming the old hash / sha256 / age, the stale
    ``<old hash>/`` subtree the restore left node-local is NOT shipped, the redaction still applies, the bound
    holds, and the next row hits."""
    key, inputs = cc.cache_key(**QWEN_CONFIGURED), cc.cache_key_inputs(**QWEN_CONFIGURED)
    token = "probe-token-" + "y" * 31
    vol, local2, before = _restored_then_recompiled(cc, tmp_path, key=key, token=token)
    time.sleep(0.01)
    p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=cold_log(str(local2 / "vllm"), FRESH),
                                 restored=True, ready=True, inputs=inputs, endpoint=QWEN_SERVED_ENDPOINT)
    assert p["persisted"] is True and p["skipped"] is None and p["vllm_hash_dir"] == FRESH
    stale = p["replaced_stale"]
    assert stale["vllm_hash_dir"] == STALE and stale["sha256"] == before["sha256"] and stale["reason"] == "hash_mismatch"
    assert stale["written_at"] == before["written_at"] and stale["age_s"] > 0 and stale["observed_hash"] == FRESH
    entry = cc.index_entry(vol, key)
    assert entry["vllm_hash_dir"] == FRESH and entry["sha256"] == p["sha256"] != before["sha256"]
    assert entry["replaced_stale"] == stale and entry["served_geometry"] == cc.served_geometry(QWEN_SERVED_ENDPOINT)
    assert entry["written_at"] > before["written_at"] and entry["inputs"] == inputs
    # the tarball: the fresh tree in, the stale subtree out, the old bytes gone, the token redacted
    with tarfile.open(p["tarball"]) as t:
        names = t.getnames()
        raw = b"".join(t.extractfile(m).read() for m in t.getmembers() if m.isfile())
    assert f"vllm/torch_compile_cache/{FRESH}/rank_0_0/backbone/artifact.bin" in names
    assert not any(f"/torch_compile_cache/{STALE}" in n for n in names), names
    assert b"graph-bytes" not in raw and b"fresh-graph" in raw and token.encode() not in raw
    assert "triton/abc/kernel.cubin" in names  # the restored Triton / FlashInfer caches are content-addressed: kept
    assert p["redacted"] == {f"vllm/torch_compile_cache/{FRESH}/rank_0_0/backbone/{cc.FACTORS_FILE}": ["VLLM_API_KEY"]}
    # the bound: a replacement is in place — one tarball for the key, no eviction, 2 inodes
    assert p["evicted"] == [] and cc.inode_count(vol) == 2 and len(cc.read_index(vol)["entries"]) <= cc.MAX_TARBALLS
    assert not any(x.name.endswith(".tmp") for x in vol.iterdir())
    # the next row: restore → vLLM prints the NEW hash and loads → hit
    local3 = tmp_path / "local3"
    r = cc.compile_cache_restore(key, vol_root=vol, local_root=local3)
    assert r["restored"] and r["expected_vllm_hash_dir"] == FRESH and r["sha256"] == p["sha256"]
    assert not (local3 / "vllm" / "torch_compile_cache" / STALE).exists()
    assert cc.compile_cache_evidence(warm_log(str(local3 / "vllm"), FRESH), key, restored=True, vol_root=vol)["hit"] is True


def test_a_restored_miss_replacement_holds_the_bound_when_the_volume_is_full(cc, tmp_path):
    vol = tmp_path / "vol"
    keys = [cc.cache_key(**{**QWEN_ROW, "max_num_seqs": 10 + i}) for i in range(6)]
    for i, k in enumerate(keys):
        assert _persist_cold(cc, k, local=tmp_path / f"local{i}", vol=vol, h=f"{i:010x}")["persisted"]
    local2 = tmp_path / "r2"
    assert cc.compile_cache_restore(keys[2], vol_root=vol, local_root=local2)["restored"]
    d = local2 / "vllm" / "torch_compile_cache" / FRESH / "rank_0_0" / "backbone"
    d.mkdir(parents=True)
    (d / "artifact.bin").write_bytes(b"fresh")
    p = cc.compile_cache_persist(keys[2], local_root=local2, vol_root=vol, vllm_server_log=cold_log(str(local2 / "vllm"), FRESH),
                                 restored=True, ready=True)
    assert p["persisted"] and p["replaced_stale"]["vllm_hash_dir"] == "0000000002" and p["evicted"] == []
    assert set(cc.read_index(vol)["entries"]) == set(keys) and cc.inode_count(vol) == 7 == cc.MAX_TARBALLS + 1


def test_a_restored_hit_keeps_refusing_the_persist(cc, tmp_path):
    """A hit never rewrites: restored, the recorded hash printed, a graph loaded → ``restored`` as today, the
    refusal naming the verdict, the index untouched."""
    key = cc.cache_key(**QWEN_ROW)
    vol, local2, before = _restored_then_recompiled(cc, tmp_path, key=key)
    p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=warm_log(str(local2 / "vllm"), STALE),
                                 restored=True, ready=True)
    assert p["persisted"] is False and p["skipped"] == "restored" and p["replaced_stale"] is None
    assert "hit" in p["refusal_detail"]
    assert cc.index_entry(vol, key) == before and cc.inode_count(vol) == 2


def test_a_restored_miss_whose_compile_did_not_complete_does_not_replace(cc, tmp_path):
    """vLLM printed another ``Using cache directory`` hash but no ``Compiling a graph … takes`` line (the standup
    died or was torn down mid-compile): the fresh tree is not a compiled graph — refused ``restored``, named."""
    key = cc.cache_key(**QWEN_ROW)
    vol, local2, before = _restored_then_recompiled(cc, tmp_path, key=key)
    log = "\n".join([_line("backends.py:1089", f"Using cache directory: {local2 / 'vllm'}/torch_compile_cache/{FRESH}/rank_0_0/backbone for vLLM's torch.compile"), ""])
    p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=log, restored=True, ready=True)
    assert p["persisted"] is False and p["skipped"] == "restored" and p["replaced_stale"] is None
    assert "compile did not complete" in p["refusal_detail"] and "hash_mismatch" in p["refusal_detail"]
    assert cc.index_entry(vol, key) == before and cc.inode_count(vol) == 2
    # a matching hash without a graph-load line (restored_miss / no_graph_load_evidence) is a finding, not a rewrite
    p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=cold_log(str(local2 / "vllm"), STALE),
                                 restored=True, ready=True)
    assert p["skipped"] == "restored" and "no_graph_load_evidence" in p["refusal_detail"]
    # an adopted orphan (no recorded hash) cannot be judged stale either
    idx = cc.read_index(vol)
    idx["entries"][key]["vllm_hash_dir"] = None
    (vol / "index.json").write_text(json.dumps(idx))
    p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=cold_log(str(local2 / "vllm"), FRESH),
                                 restored=True, ready=True)
    assert p["skipped"] == "restored" and "no_expected_hash" in p["refusal_detail"]


def test_a_restored_miss_replacement_is_still_subject_to_the_other_refusals(cc, tmp_path):
    """The first reason still wins (``--no-cache-persist`` / smoke / negative control / not ready), and the served
    geometry must still agree with the key."""
    key, inputs = cc.cache_key(**QWEN_CONFIGURED), cc.cache_key_inputs(**QWEN_CONFIGURED)
    vol, local2, before = _restored_then_recompiled(cc, tmp_path, key=key)
    log = cold_log(str(local2 / "vllm"), FRESH)
    for kw, reason in [(dict(cache_persist=False), "no_cache_persist"), (dict(inspect_limit=3), "smoke_shape"),
                       (dict(negative_control={"cell": "x"}), "negative_control"), (dict(ready=False), "not_ready")]:
        p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=log, restored=True,
                                     **{**dict(ready=True), **kw})
        assert p["persisted"] is False and p["skipped"] == reason and p["replaced_stale"] is None
    g27 = {**QWEN_SERVED_ENDPOINT["geometry"], "max_num_seqs": 27, "batched_tokens": 27 * 16384}
    p = cc.compile_cache_persist(key, local_root=local2, vol_root=vol, vllm_server_log=log, restored=True, ready=True,
                                 inputs=inputs, endpoint={**QWEN_SERVED_ENDPOINT, "geometry": g27})
    assert p["persisted"] is False and p["skipped"] == "key_inputs_disagree"
    assert cc.index_entry(vol, key) == before and cc.inode_count(vol) == 2
