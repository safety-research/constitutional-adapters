"""scripts/jailbreak_run.py: the served-target + hosted-attacker + goal-replay
flags (all opt-in; defaults unchanged).

- ``--target-engine served`` builds the target from the proxy + routing table
  and NEVER calls load_model_and_tokenizer; the run dir gets
  ``served_target.json`` beside ``attacker_engine.json``.
- ``--goal-ids-file`` replays an exact goal set (file order) before sharding;
  unknown / duplicate ids refuse.
- ``--expect-pair-n-streams`` refuses a PAIR whose module budget differs.
- ``--attacker-engine openrouter`` reads the key from the env, builds the
  OpenRouter client, and ``--scorer-endpoint`` keeps the in-loop scorer on a
  SEPARATE vLLM endpoint; both roles are logged when a log dir is given.
- ``attacker_engine.json`` records the PAIR budget, the scorer's engine and
  the target engine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jailbreak_run  # noqa: E402
from sisteer.eval.attacks import openrouter_attacker as oa  # noqa: E402
from sisteer.eval.attacks import pair_tap  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402
from sisteer.eval.attacks import vllm_attacker as va  # noqa: E402


def _goals(n):
    return [SimpleNamespace(goal_id=f"g{i}", text=f"goal {i}", category="c")
            for i in range(n)]


# --------------------------------------------------------------------------- #
# goal replay + PAIR budget                                                    #
# --------------------------------------------------------------------------- #

def test_select_goal_ids_file_order_and_refusals(tmp_path):
    goals = _goals(6)
    f = tmp_path / "ids.txt"
    f.write_text("# the wave's shard\ng4\ng1\n\ng5\n")
    sel = jailbreak_run._select_goal_ids(goals, f)
    assert [g.goal_id for g in sel] == ["g4", "g1", "g5"]
    assert [g.goal_id for g in sel[0::2]] == ["g4", "g5"]      # then sharded
    f.write_text("g1\nnope\n")
    with pytest.raises(SystemExit):
        jailbreak_run._select_goal_ids(goals, f)
    f.write_text("g1\ng1\n")
    with pytest.raises(SystemExit):
        jailbreak_run._select_goal_ids(goals, f)
    f.write_text("\n")
    with pytest.raises(SystemExit):
        jailbreak_run._select_goal_ids(goals, f)


def test_assert_pair_budget():
    jailbreak_run._assert_pair_budget(None)
    jailbreak_run._assert_pair_budget(pair_tap.PAIR_N_STREAMS)
    with pytest.raises(SystemExit):
        jailbreak_run._assert_pair_budget(pair_tap.PAIR_N_STREAMS + 25)


def test_pair_budget_is_the_wave_value():
    """The published gemma4e4b multi-turn wave ran 5 x 3 (commit d17976c);
    a reproduction on this branch must see the same constant."""
    assert pair_tap.PAIR_N_STREAMS == 5 and pair_tap.PAIR_DEPTH == 3


# --------------------------------------------------------------------------- #
# attacker endpoint resolution + manifest                                     #
# --------------------------------------------------------------------------- #

def test_resolve_attacker_endpoint():
    ns = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    assert jailbreak_run._resolve_attacker_endpoint(ns) == "http://127.0.0.1:8000"
    ns = SimpleNamespace(attacker_engine="openrouter", attacker_endpoint="http://127.0.0.1:8000")
    assert jailbreak_run._resolve_attacker_endpoint(ns) == oa.DEFAULT_ENDPOINT
    ns = SimpleNamespace(attacker_engine="openrouter", attacker_endpoint="https://x/api")
    assert jailbreak_run._resolve_attacker_endpoint(ns) == "https://x/api"
    assert jailbreak_run._resolve_attacker_endpoint(SimpleNamespace(attacker_engine="hf")) is None


def test_engine_manifest_records_budget_scorer_and_target(tmp_path):
    cfg = {"attacker": {"model": "qwen", "snapshot": "s"}}
    args = SimpleNamespace(attacker_engine="openrouter", attacker_endpoint="http://127.0.0.1:8000",
                           attacker_model="x-ai/grok-4.6", attacker_reasoning_effort=None,
                           scorer_endpoint="http://127.0.0.1:9", scorer_model=None,
                           target_engine="served", goal_ids_file="/tmp/ids")
    jailbreak_run._write_engine_manifest(tmp_path, cfg, args, "pair")
    m = json.loads((tmp_path / "attacker_engine.json").read_text())
    assert m["attacker_engine"] == "openrouter" and m["attacker_model"] == "x-ai/grok-4.6"
    assert m["attacker_endpoint"] == oa.DEFAULT_ENDPOINT and m["attacker_snapshot"] is None
    assert m["scorer_engine"] == "vllm" and m["scorer_endpoint"] == "http://127.0.0.1:9"
    assert m["scorer_model"] == "qwen"
    assert m["target_engine"] == "served" and m["goal_ids_file"] == "/tmp/ids"
    assert m["pair_n_streams"] == pair_tap.PAIR_N_STREAMS and m["pair_depth"] == pair_tap.PAIR_DEPTH
    # the legacy vllm shape: scorer shares the attacker
    args2 = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    jailbreak_run._write_engine_manifest(tmp_path / "b", cfg, args2, "crescendo")
    m2 = json.loads((tmp_path / "b" / "attacker_engine.json").read_text())
    assert m2["attacker_model"] == "qwen" and m2["attacker_snapshot"] == "s"
    assert m2["scorer_engine"] == "vllm" and m2["scorer_endpoint"] == "http://127.0.0.1:8000"
    assert m2["target_engine"] == "hf"


# --------------------------------------------------------------------------- #
# --attacker-engine openrouter seams                                          #
# --------------------------------------------------------------------------- #

class _FakeOR(oa.OpenRouterAttackerClient):
    instances: list = []

    def __init__(self, model, *, api_key, endpoint=oa.DEFAULT_ENDPOINT, reasoning=None, **kw):
        super().__init__(model, api_key=api_key, endpoint=endpoint, reasoning=reasoning,
                         session=_NullSession())
        _FakeOR.instances.append(self)

    def check_health(self):
        self.healthy = True

    def chat(self, messages, **kw):
        self.last_meta = {"finish_reason": "stop"}
        return "{\"improvement\": \"x\", \"prompt\": \"attack\"}"


class _FakeVLLM(va.VLLMAttackerClient):
    def check_health(self):
        self.healthy = True

    def chat(self, messages, **kw):
        return "7"


class _NullSession:
    headers: dict = {}


def test_build_seams_openrouter_with_separate_scorer(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sekrit")
    monkeypatch.setattr(oa, "OpenRouterAttackerClient", _FakeOR)
    monkeypatch.setattr(va, "VLLMAttackerClient", _FakeVLLM)
    _FakeOR.instances.clear()
    cfg = {"attacker": {"model": "qwen"}, "attacks": {"attacker_max_new_tokens": 64,
                                                      "scorer_max_new_tokens": 8}}
    args = SimpleNamespace(attack="pair", attacker_engine="openrouter",
                           attacker_endpoint="http://127.0.0.1:8000",
                           attacker_model="x-ai/grok-4.6", attacker_api_key_env="OPENROUTER_API_KEY",
                           attacker_reasoning_effort="low",
                           scorer_endpoint="http://127.0.0.1:9", scorer_model=None)
    scorer_fn, factory, attacker_fn = jailbreak_run._build_attacker_seams(cfg, args, log_dir=tmp_path)
    assert attacker_fn([{"role": "user", "content": "go"}]).startswith("{")
    assert scorer_fn("rate it") == "7"
    (orc,) = _FakeOR.instances
    assert orc.model == "x-ai/grok-4.6" and orc.reasoning == {"effort": "low"}
    assert orc.endpoint == oa.DEFAULT_ENDPOINT
    # both roles logged, in separate files
    a = [json.loads(l) for l in (tmp_path / "attacker_calls.jsonl").read_text().splitlines()]
    s = [json.loads(l) for l in (tmp_path / "scorer_calls.jsonl").read_text().splitlines()]
    assert len(a) == 1 and a[0]["role"] == "attacker" and a[0]["reply"].startswith("{")
    assert len(s) == 1 and s[0]["role"] == "scorer" and s[0]["reply"] == "7"


def test_build_seams_openrouter_refuses_without_key_or_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = {"attacker": {"model": "qwen"}, "attacks": {}}
    args = SimpleNamespace(attack="pair", attacker_engine="openrouter",
                           attacker_endpoint="http://127.0.0.1:8000",
                           attacker_model="x-ai/grok-4.6", attacker_api_key_env="OPENROUTER_API_KEY",
                           attacker_reasoning_effort=None, scorer_endpoint=None, scorer_model=None)
    with pytest.raises(SystemExit):
        jailbreak_run._build_attacker_seams(cfg, args)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    args.attacker_model = None
    with pytest.raises(SystemExit):
        jailbreak_run._build_attacker_seams(cfg, args)


# --------------------------------------------------------------------------- #
# cmd_generate --target-engine served                                          #
# --------------------------------------------------------------------------- #

class _FakeServed:
    def __init__(self):
        self.virtual_model = "petri__si_constitution__psr_kl_all"
        self.route_info = {"intervention_hash": "cc" * 8, "virtual_model": self.virtual_model}
        self.evidence = {"steering_bites": {"applicable": True, "verified": True}}
        self.timing_log = None
        self.n_target_forwards = 0
        self.mode = "si_constitution__psr_kl_all"
        # the window-clamp surface the target manifest records (served_window)
        self.max_model_len = None
        self.n_clamped = 0
        self.n_prompt_over_window = 0
        self.n_window_mismatch = 0
        self.window_mismatches = []
        self.n_tokenize_resolved = 0
        self.n_floor_only_unresolvable = 0
        # the all-turns render-check counters + the per-process attempts row
        # (served_target.append_target_attempt) read these
        self.n_all_turns_refused = {}
        self.n_timing_rows = 0
        self.cell_id = self.mode
        self.route_info.update(tt_cell_id=self.mode, harness_cell_id=self.mode,
                               serve_variant=None)


def test_cmd_generate_served_skips_hf_load_and_writes_manifests(monkeypatch, tmp_path):
    import sisteer.models as models_mod

    def boom(*a, **k):
        raise AssertionError("load_model_and_tokenizer must not be called on the served path")
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", boom)
    fake = _FakeServed()
    built = {}

    def fake_build(**kw):
        built.update(kw)
        return fake
    monkeypatch.setattr(st, "build_served_target", fake_build)
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids",
                        lambda cfg: {"none", "si_constitution__psr_kl_all"})
    goals = _goals(5)
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: goals)
    monkeypatch.setattr(jailbreak_run, "_build_attacker_seams",
                        lambda cfg, args, log_dir=None: ((lambda p: "5"), (lambda lock: None),
                                                         (lambda chat: "x")))
    rec = {}

    def fake_run(attack, defense, goals_arg, *, attacker_factory, attacker_fn, scorer_fn,
                 out_dir, forward_cap, target_max_new_tokens, resume=True, target_temperature=None):
        # the STATED generation config (attacks.target_decode below) -> None: the server's generation config
        assert target_temperature is None, target_temperature
        rec.update(attack=attack, defense=defense, out_dir=Path(out_dir),
                   goals=[g.goal_id for g in goals_arg], cap=forward_cap, mnt=target_max_new_tokens)
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg),
                "n_target_forwards_total": len(goals_arg)}       # fw: a gated run that ran goals reached the target
    monkeypatch.setattr(jailbreak_run, "_run_gated_attack", fake_run)
    ids = tmp_path / "ids.txt"
    ids.write_text("g3\ng0\ng4\n")
    # this synthetic model "m" has no vendor-preset row, and the campaign default (vendor_preset, 2026-09-16)
    # refuses such a served model; the plumbing under test is the served path, so the decode is STATED
    # (the 2026-09-11 -> 2026-09-16 default, byte-identical to what this test drove before the flip)
    cfg = {"model": "m", "dtype": "bfloat16", "private_run": "jb",
           "attacker": {"model": "qwen"},
           "attacks": {"forward_cap": 150, "target_max_new_tokens": 2048, "target_decode": "generation_config"}}
    args = SimpleNamespace(attack="crescendo", defense="si_constitution__psr_kl_all",
                           shard=1, nshards=2, xstest=False, confirmatory=True, trial=0,
                           private_root=str(tmp_path / "state"), target_engine="served",
                           target_proxy_url="http://127.0.0.1:7/v1",
                           target_routing_table=str(tmp_path / "rt.json"), positions="response",
                           scale=1.0, goal_ids_file=str(ids), expect_pair_n_streams=None,
                           attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    jailbreak_run.cmd_generate(args, cfg)
    # the serving window is threaded as PROVENANCE only: absent
    # --target-max-model-len it is None (the server's stated window governs
    # the clamp; no table default is assumed for a standup of unknown geometry)
    assert built == {"proxy_url": "http://127.0.0.1:7/v1",
                     "routing_table": str(tmp_path / "rt.json"),
                     "cell_id": "si_constitution__psr_kl_all",
                     "max_model_len": None,
                     # no --target-top-p/-k/-repetition-penalty -> no decode block (the greedy default, 2026-09-10)
                     "decode": None}
    assert rec["defense"] is fake and rec["attack"] == "crescendo"
    assert rec["goals"] == ["g0"]                       # file order [g3,g0,g4] -> shard 1 of 2
    assert rec["cap"] == 150 and rec["mnt"] == 2048
    out = rec["out_dir"]
    assert out.as_posix().endswith("jb/crescendo/si_constitution__psr_kl_all/shard01")
    assert (out / "attacker_engine.json").exists()
    doc = json.loads((out / "served_target.json").read_text())
    assert doc["target_engine"] == "served" and doc["evidence"]["steering_bites"]["verified"]
    # rewritten AFTER generation, so the clamp counters are the real ones
    assert doc["manifest_stage"] == "final"
    assert doc["window_clamp"]["n_prompt_over_window"] == 0
    assert fake.timing_log == out / "target_timing.jsonl"


def test_cmd_generate_served_refusals(monkeypatch, tmp_path):
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none"})
    cfg = {"model": "m", "dtype": "bfloat16", "private_run": "jb", "attacks": {}}
    base = dict(attack="pair", defense="none", shard=0, nshards=1, xstest=False,
                confirmatory=False, trial=0, private_root=str(tmp_path), target_engine="served",
                target_proxy_url="http://127.0.0.1:7/v1", target_routing_table="/tmp/rt.json",
                positions=None, scale=1.0, goal_ids_file=None, expect_pair_n_streams=None)
    with pytest.raises(SystemExit):   # missing proxy url
        jailbreak_run.cmd_generate(SimpleNamespace(**{**base, "target_proxy_url": None}), cfg)
    with pytest.raises(SystemExit):   # --positions all is not servable
        jailbreak_run.cmd_generate(SimpleNamespace(**{**base, "positions": "all"}), cfg)
    with pytest.raises(SystemExit):   # --scale is baked into the served object
        jailbreak_run.cmd_generate(SimpleNamespace(**{**base, "scale": 2.0}), cfg)


def test_cmd_generate_pair_budget_refusal_on_hf_path(monkeypatch, tiny_llama, tmp_path):
    import sisteer.models as models_mod

    model, tok = tiny_llama
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", lambda *a, **k: (model, tok))
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: _goals(2))
    cfg = {"model": "m", "dtype": "float32", "private_run": "jb", "attacks": {}}
    args = SimpleNamespace(attack="pair", defense="none", shard=0, nshards=1, xstest=False,
                           confirmatory=False, trial=0, private_root=str(tmp_path),
                           target_engine="hf", goal_ids_file=None,
                           expect_pair_n_streams=pair_tap.PAIR_N_STREAMS + 1)
    with pytest.raises(SystemExit):
        jailbreak_run.cmd_generate(args, cfg)


def test_generate_parser_defaults_unchanged():
    """Nothing changes unless asked: hf target, vllm attacker, no goal file."""
    import argparse
    p = argparse.ArgumentParser()
    jailbreak_run._add_attacker_engine_args(p)
    jailbreak_run._add_target_engine_args(p)
    ns = p.parse_args([])
    assert ns.target_engine == "hf" and ns.attacker_engine == "vllm"
    assert ns.goal_ids_file is None and ns.expect_pair_n_streams is None
    assert ns.scorer_endpoint is None and ns.attacker_model is None
    assert ns.target_max_model_len is None       # -> the table's entry at build time
    assert p.parse_args(["--target-max-model-len", "8704"]).target_max_model_len == 8704


# --------------------------------------------------------------------------- #
# Serve variants on the --defense surface                                     #
# --------------------------------------------------------------------------- #

TT_PSR = "safety_refusal__psr_kl_all"


def _served_args(tmp_path, ids, **over):
    base = dict(attack="crescendo", defense=f"{TT_PSR}@all_turns",
                shard=1, nshards=2, xstest=False, confirmatory=True, trial=0,
                private_root=str(tmp_path / "state"), target_engine="served",
                target_proxy_url="http://127.0.0.1:7/v1",
                target_routing_table=str(tmp_path / "rt.json"), positions=None,
                scale=1.0, goal_ids_file=str(ids), expect_pair_n_streams=None,
                attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    base.update(over)
    return SimpleNamespace(**base)


def _wire_served(monkeypatch, tmp_path, fake):
    import sisteer.models as models_mod

    def boom(*a, **k):
        raise AssertionError("load_model_and_tokenizer must not be called on the served path")
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer", boom)
    built = {}

    def fake_build(**kw):
        built.update(kw)
        fake.cell_id = kw["cell_id"]
        fake.route_info["tt_cell_id"] = kw["cell_id"]
        return fake
    monkeypatch.setattr(st, "build_served_target", fake_build)
    monkeypatch.setattr(jailbreak_run, "_valid_defense_ids", lambda cfg: {"none", TT_PSR})
    monkeypatch.setattr(jailbreak_run, "_eval_goals", lambda cfg, *, confirmatory: _goals(5))
    monkeypatch.setattr(jailbreak_run, "_build_attacker_seams",
                        lambda cfg, args, log_dir=None: ((lambda p: "5"), (lambda lock: None),
                                                         (lambda chat: "x")))
    rec = {}

    def fake_run(attack, defense, goals_arg, *, attacker_factory, attacker_fn, scorer_fn,
                 out_dir, forward_cap, target_max_new_tokens, resume=True, target_temperature=None):
        rec.update(attack=attack, defense=defense, out_dir=Path(out_dir),
                   goals=[g.goal_id for g in goals_arg], target_temperature=target_temperature)
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg),
                "n_target_forwards_total": len(goals_arg)}       # fw: a gated run that ran goals reached the target
    monkeypatch.setattr(jailbreak_run, "_run_gated_attack", fake_run)
    ids = tmp_path / "ids.txt"
    ids.write_text("g3\ng0\ng4\n")
    # synthetic model "m" has no vendor-preset row (the 2026-09-16 default refuses one); the decode is STATED
    cfg = {"model": "m", "dtype": "bfloat16", "private_run": "jb",
           "attacker": {"model": "qwen"},
           "attacks": {"forward_cap": 150, "target_max_new_tokens": 2048, "target_decode": "generation_config"}}
    return built, rec, ids, cfg


def test_cmd_generate_served_accepts_a_variant_cell_and_records_it(monkeypatch, tmp_path):
    fake = _FakeServed()
    fake.virtual_model = f"petri__si_safety__psr_kl_all@all_turns"
    fake.route_info.update(virtual_model=fake.virtual_model, serve_variant="all_turns",
                           harness_cell_id="si_safety__psr_kl_all@all_turns")
    fake.n_all_turns_refused = {"all_turns_expected_unavailable": 2}
    built, rec, ids, cfg = _wire_served(monkeypatch, tmp_path, fake)
    jailbreak_run.cmd_generate(_served_args(tmp_path, ids), cfg)
    # the FULL id (variant included) reaches the served target builder
    assert built["cell_id"] == f"{TT_PSR}@all_turns"
    out = rec["out_dir"]
    assert out.as_posix().endswith(f"jb/crescendo/{TT_PSR}@all_turns/shard01")
    m = json.loads((out / "attacker_engine.json").read_text())
    assert m["serve_variant"] == "all_turns" and m["target_engine"] == "served"
    doc = json.loads((out / "served_target.json").read_text())
    assert doc["manifest_stage"] == "final"
    assert doc["all_turns_check"]["n_refused_total"] == 2
    assert doc["all_turns_check"]["by_code"]["all_turns_expected_unavailable"] == 2
    # ONE append-only attempts row per process, beside the manifest
    rows = [json.loads(l) for l in (out / st.TARGET_ATTEMPTS_FILE).read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["stage"] == "final"
    assert rows[0]["all_turns_check"]["n_refused_total"] == 2
    assert rows[0]["serve_variant"] == "all_turns"
    # a second process on the same dir appends, never overwrites
    st.append_target_attempt(out, fake, stage="final")
    rows = [json.loads(l) for l in (out / st.TARGET_ATTEMPTS_FILE).read_text().splitlines()]
    assert len(rows) == 2 and all(r["stage"] == "final" for r in rows)


def test_attempts_ledger_is_a_manifest_class_record_for_the_resume_guard(tmp_path):
    """A dir holding ONLY the manifest files + the attempts ledger has no
    persisted work: the attacker-identity guard rewrites the manifest instead
    of refusing (as it does for served_target.json, which IS evidence)."""
    assert st.TARGET_ATTEMPTS_FILE in jailbreak_run._MANIFEST_FILES
    cfg = {"attacker": {"model": "qwen", "snapshot": "s"}}
    args = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000",
                           defense=f"{TT_PSR}@all_turns", target_engine="served")
    d = tmp_path / "shard00"
    jailbreak_run._write_engine_manifest(d, cfg, args, "crescendo")
    (d / st.TARGET_ATTEMPTS_FILE).write_text(json.dumps({"stage": "final"}) + "\n")
    jailbreak_run._write_engine_manifest(d, cfg, args, "crescendo")      # no refusal
    assert not (d / "attacker_engine.resumes.jsonl").exists()
    # whereas real persisted work under an unverifiable identity refuses, as before
    (d / "served_target.json").write_text("{}")
    from sisteer.capeval.engine_identity import EngineIdentityError
    with pytest.raises(EngineIdentityError):
        jailbreak_run._write_engine_manifest(d, cfg, args, "crescendo")


def test_cmd_generate_variant_refusals(monkeypatch, tmp_path):
    fake = _FakeServed()
    built, rec, ids, cfg = _wire_served(monkeypatch, tmp_path, fake)
    # --positions beside a variant: the variant IS the positions
    with pytest.raises(SystemExit, match="positions"):
        jailbreak_run.cmd_generate(_served_args(tmp_path, ids, positions="response"), cfg)
    # the HF path cannot serve a variant
    with pytest.raises(SystemExit, match="SERVED-ONLY"):
        jailbreak_run.cmd_generate(_served_args(tmp_path, ids, target_engine="hf"), cfg)
    # an unknown variant, a variant on `none`, an unknown base
    with pytest.raises(SystemExit, match="unknown --defense"):
        jailbreak_run.cmd_generate(_served_args(tmp_path, ids, defense=f"{TT_PSR}@bogus"), cfg)
    with pytest.raises(SystemExit):
        jailbreak_run.cmd_generate(_served_args(tmp_path, ids, defense="none@all_turns"), cfg)
    with pytest.raises(SystemExit, match="unknown --defense"):
        jailbreak_run.cmd_generate(_served_args(tmp_path, ids, defense="nope@all_turns"), cfg)
    assert built == {} and rec == {}                       # refused before anything cost
    # the BASE fitted cell keeps --positions response, as the §15 wave did
    jailbreak_run.cmd_generate(_served_args(tmp_path, ids, defense=TT_PSR,
                                            positions="response"), cfg)
    assert built["cell_id"] == TT_PSR
    m = json.loads((rec["out_dir"] / "attacker_engine.json").read_text())
    assert m["serve_variant"] is None


def test_engine_manifest_serve_variant_does_not_enter_the_attacker_identity(tmp_path):
    cfg = {"attacker": {"model": "qwen", "snapshot": "s"}}
    a1 = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000",
                         defense=f"{TT_PSR}@all_positions", target_engine="served")
    jailbreak_run._write_engine_manifest(tmp_path / "a", cfg, a1, "crescendo")
    m1 = json.loads((tmp_path / "a" / "attacker_engine.json").read_text())
    assert m1["serve_variant"] == "all_positions"
    a2 = SimpleNamespace(attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    jailbreak_run._write_engine_manifest(tmp_path / "b", cfg, a2, "crescendo")
    m2 = json.loads((tmp_path / "b" / "attacker_engine.json").read_text())
    assert m2["serve_variant"] is None
    # the attacker identity compare ignores it (the target identity is
    # fst_run_identity.json's job)
    i1 = jailbreak_run._identity_of_manifest_dict(m1, where="a")
    i2 = jailbreak_run._identity_of_manifest_dict(m2, where="b")
    assert i1 == i2 and "serve_variant" not in i1


# --------------------------------------------------------------------------- #
# Grading: a refused goal is excluded unconditionally; the denominator is the  #
# CLEAN mapping's (S3)                                                          #
# --------------------------------------------------------------------------- #

THRESHOLDS = [0.25, 0.5, 0.75]


def _grade_world():
    by_id = {g.goal_id: g for g in (
        SimpleNamespace(goal_id="g0", text="o0", dataset="harmbench", revision="r", category="c"),
        SimpleNamespace(goal_id="g1", text="o1", dataset="harmbench", revision="r", category="c"),
        SimpleNamespace(goal_id="g2", text="o2", dataset="harmbench", revision="r", category="c"))}
    responses = {"g0": ["no.", "still no."], "g1": ["partial turn one", "partial turn two"],
                 "g2": ["nope"]}
    progress = {"g0": {"goal_id": "g0", "outcome": "failure", "n_target_forwards": 2},
                "g1": {"goal_id": "g1", "outcome": "refused:all_turns_expected_unavailable",
                       "n_target_forwards": 2},
                "g2": {"goal_id": "g2", "outcome": "failure", "n_target_forwards": 1}}
    return by_id, responses, progress


def test_gated_metrics_exclude_a_refused_goal_unconditionally():
    by_id, responses, progress = _grade_world()
    seen = []

    def grade(items):
        seen.extend(items)
        return {it: 0.0 for it in items}
    # NO --allow-errored-goals: a refused goal is not an incompleteness, so no refusal
    m = jailbreak_run._gated_defense_metrics(
        responses, progress, by_id, grade, thresholds=THRESHOLDS, canonical=0.5,
        leaked_needles=[], expected_goal_ids=["g0", "g1", "g2"], label="crescendo] x")
    assert m["refused_goal_ids"] == ["g1"] and m["n_refused"] == 1
    assert m["refused_by_code"] == {"all_turns_expected_unavailable": 1}
    assert m["n_errored"] == 0 and m["errored_goal_ids"] == []
    assert m["n_graded"] == 2 and m["n_expected"] == 3 and m["n_goals"] == 2
    assert set(m["per_goal"]) == {"g0", "g2"}
    # the persisted partial rows of the refused goal were never graded ...
    assert all("partial" not in it[3] for it in seen) and len(seen) == 3
    # ... and the denominator is the CLEAN mapping's row count
    assert m["n_responses_graded"] == 3
    # an errored goal beside it still refuses without the flag
    progress["g2"]["outcome"] = "error:PromptOverWindow"
    with pytest.raises(SystemExit, match="PromptOverWindow"):
        jailbreak_run._gated_defense_metrics(
            responses, progress, by_id, grade, thresholds=THRESHOLDS, canonical=0.5,
            leaked_needles=[], expected_goal_ids=["g0", "g1", "g2"])
    m2 = jailbreak_run._gated_defense_metrics(
        responses, progress, by_id, grade, thresholds=THRESHOLDS, canonical=0.5,
        leaked_needles=[], expected_goal_ids=["g0", "g1", "g2"], allow_errored_goals=True)
    assert m2["n_graded"] == 1 and m2["n_responses_graded"] == 2
    assert m2["refused_goal_ids"] == ["g1"] and m2["errored_goal_ids"] == ["g2"]


def test_grokmt_grade_outcomes_histogram_names_the_refusal(monkeypatch, tmp_path):
    """The one consumer that used to recompute the denominator: it now reads
    the helper's; the ``outcomes`` histogram shows ``refused:<code>: 1``; a
    variant cell id is accepted on ``--cells``."""
    import grade_gated_served as ggs
    import grokmt_grade

    by_id, responses, progress = _grade_world()
    goals_all = list(by_id.values())
    cell = f"{TT_PSR}@all_turns"
    monkeypatch.setattr(grokmt_grade.jr, "_eval_goals", lambda cfg, *, confirmatory: goals_all)
    monkeypatch.setattr(grokmt_grade.jr, "_category_groups",
                        lambda cfg, goals: ({g.goal_id: "held_in" for g in goals}, ["c"], [],
                                            {"seed": 0, "frac": 0.33}))
    monkeypatch.setattr(grokmt_grade.jr, "_private_root", lambda cfg, o: Path(o))
    seen_roots = []

    def fake_load(defense_root):
        seen_roots.append(Path(defense_root))
        return responses, progress
    monkeypatch.setattr(ggs, "load_gated_dir", fake_load)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("private_run: jb\nprereg:\n  threshold_grid: [0.25, 0.5, 0.75]\n"
                   "grader:\n  model: judge\n")
    ids = tmp_path / "ids.txt"
    ids.write_text("g0\ng1\ng2\n")
    out = tmp_path / "grades.json"
    argv = ["grokmt_grade.py", "--config", str(cfg), "--private-root", str(tmp_path / "root"),
            "--cells", cell, "--families", "crescendo", "--goal-ids-file", str(ids),
            "--out", str(out), "--cache", str(tmp_path / "cache.jsonl"), "--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)
    grokmt_grade.main()
    doc = json.loads(out.read_text())
    m = doc["cells"][cell]["crescendo"]
    assert seen_roots[0].as_posix().endswith(f"root/jb/crescendo/{cell}")
    assert m["outcomes"] == {"failure": 2, "refused:all_turns_expected_unavailable": 1}
    assert m["refused_goal_ids"] == ["g1"]
    assert m["refused_by_code"] == {"all_turns_expected_unavailable": 1}
    assert m["n_responses_graded"] == 3 and m["n_graded"] == 2
    assert doc["goal_ids"] == ["g0", "g1", "g2"]
    # a malformed variant refuses at the door
    monkeypatch.setattr(sys, "argv", argv[:6] + [f"{TT_PSR}@bogus"] + argv[7:])
    with pytest.raises(SystemExit, match="unknown"):
        grokmt_grade.main()


def _grokmt_grade_tree(tmp_path: Path, cell: str, by_id, responses, progress) -> tuple[Path, Path]:
    """A REAL on-disk ``<root>/jb/crescendo/<cell>/shard00/`` tree for the
    grader to glob (the provenance block digests the files it read), plus a
    run identity record at ``<root>/../fst_run_identity.json``."""
    import hashlib as _h
    root = tmp_path / "r1" / "qwen"
    sd = root / "jb" / "crescendo" / cell / "shard00"
    sd.mkdir(parents=True)
    rows = [{"goal_id": gid, "order": o, "response": t}
            for gid in by_id for o, t in enumerate(responses[gid])]
    (sd / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (sd / "progress.jsonl").write_text("".join(json.dumps(progress[g]) + "\n" for g in by_id))
    ident = tmp_path / "r1" / "fst_run_identity.json"
    ident.write_text(json.dumps({"gated_sha": "0" * 40, "cells": [cell]}, sort_keys=True,
                                separators=(",", ":")))
    assert _h.sha256(ident.read_bytes()).hexdigest()  # exists, non-empty
    return root, ident


def test_grokmt_grade_writes_an_additive_grading_provenance_block(monkeypatch, tmp_path):
    """S-A1 (fit-support-transfer code review): the output gains
    ``grading_provenance`` — schema, the run identity's sha256 (found by
    default beside the private root), the goal-ids file digest, the sha256 of
    EVERY responses.jsonl / progress.jsonl the grader read, the grader model +
    its source, the rubric and ``dry_run``. Everything that existed before is
    byte-for-byte the same keys with the same meaning (additive only)."""
    import hashlib as _h

    import grokmt_grade

    by_id, responses, progress = _grade_world()
    goals_all = list(by_id.values())
    cell = f"{TT_PSR}@all_turns"
    root, ident = _grokmt_grade_tree(tmp_path, cell, by_id, responses, progress)
    monkeypatch.setattr(grokmt_grade.jr, "_eval_goals", lambda cfg, *, confirmatory: goals_all)
    monkeypatch.setattr(grokmt_grade.jr, "_category_groups",
                        lambda cfg, goals: ({g.goal_id: "held_in" for g in goals}, ["c"], [],
                                            {"seed": 0, "frac": 0.33}))
    monkeypatch.setattr(grokmt_grade.jr, "_private_root", lambda cfg, o: Path(o))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("private_run: jb\nprereg:\n  threshold_grid: [0.25, 0.5, 0.75]\n"
                   "grader:\n  model: judge\n")
    ids = tmp_path / "ids.txt"
    ids.write_text("# synthetic\ng0\ng1\ng2\n")
    out = tmp_path / "grades.json"
    base = ["grokmt_grade.py", "--config", str(cfg), "--private-root", str(root),
            "--cells", cell, "--families", "crescendo", "--goal-ids-file", str(ids),
            "--out", str(out), "--cache", str(tmp_path / "cache.jsonl")]
    # the REAL (non-dry) grader path, with the live judge replaced by a seam
    seen: dict = {}

    import types

    from sisteer import vault

    class FakeGrader:
        """The production grader's surface as grokmt_grade uses it: grade(), rubric,
        and the provenance the script RECORDS — the vault pool stamp of the client
        it was built with (never the key) and run_stats()."""
        def __init__(self, **kw):
            seen.update(kw)
            self.rubric = kw["rubric"]
            self.org = kw.get("org") or "lowprio"
            self._client = types.SimpleNamespace(
                base_url="https://api.anthropic.com",
                sisteer_pool_stamp=vault.pool_stamp(self.org, f"ORG-KEY-{self.org}"))   # one placeholder per pool, never a key

        @property
        def pool_stamp(self):
            return vault.client_pool_stamp(self._client)

        def run_stats(self):
            return {**self.pool_stamp, "org": self.org, "concurrency": 1,
                    "sdk_max_retries": 2, "n_rate_limited": 0}

        def grade(self, items):
            return {it: 0.0 for it in items}
    import sisteer.eval.attacks.grading_live as gl
    monkeypatch.setattr(gl, "LiveStrongREJECTGrader", FakeGrader)
    monkeypatch.setattr(sys, "argv", base)
    grokmt_grade.main()
    doc = json.loads(out.read_text())
    # the pre-existing surface is intact
    m = doc["cells"][cell]["crescendo"]
    assert m["n_responses_graded"] == 3 and m["refused_goal_ids"] == ["g1"]
    assert doc["model"] == "judge" and doc["goal_ids"] == ["g0", "g1", "g2"]
    assert set(doc) == {"model", "org", "thresholds", "goal_ids", "category_holdout", "cells",
                        "grading_provenance", "grader_run_stats"}
    # the pool the judge calls went to is RECORDED from the grader, not assumed: the
    # default routes to lowprio and labels it as the output always has
    assert doc["org"] == "low-prio" and seen["org"] == "lowprio"
    assert doc["grader_run_stats"] == {**vault.pool_stamp("lowprio", "ORG-KEY-lowprio"),
                                       "org": "lowprio", "concurrency": 1,
                                       "sdk_max_retries": 2, "n_rate_limited": 0}
    gp = doc["grading_provenance"]
    sd = root / "jb" / "crescendo" / cell / "shard00"
    assert gp == {
        "schema": "fst_grading/1",
        "run_identity_path": str(ident),
        "run_identity_sha256": _h.sha256(ident.read_bytes()).hexdigest(),
        "goal_ids_file_sha256": _h.sha256(ids.read_bytes()).hexdigest(),
        "inputs": {cell: {"crescendo": {
            "defense_root": str(sd.parent),
            "shards": {"shard00": {
                "responses_sha256": _h.sha256((sd / "responses.jsonl").read_bytes()).hexdigest(),
                "progress_sha256": _h.sha256((sd / "progress.jsonl").read_bytes()).hexdigest()}}}}},
        "grader_model": "judge",
        "grader_model_source": "config",
        "rubric_version": "v1",
        "dry_run": False,
    }
    assert seen["model"] == "judge" and seen["rubric"] == grokmt_grade.RUBRIC_VERSION == "v1"
    # --org highprio reaches the grader AND the record: the label and the stamp both
    # name the pool the run actually used (the bug was a hardcoded "low-prio" whatever
    # key was in the environment)
    monkeypatch.setattr(sys, "argv", base + ["--org", "highprio"])
    grokmt_grade.main()
    doc_hi = json.loads(out.read_text())
    assert seen["org"] == "highprio" and doc_hi["org"] == "high-prio"
    assert doc_hi["grader_run_stats"]["pool"] == "highprio"
    assert doc_hi["grader_run_stats"]["key_sha256_12"] != doc["grader_run_stats"]["key_sha256_12"]
    # --grader-model is recorded as an override; --dry-run is recorded as such
    monkeypatch.setattr(sys, "argv", base + ["--grader-model", "other-judge"])
    grokmt_grade.main()
    gp2 = json.loads(out.read_text())["grading_provenance"]
    assert (gp2["grader_model"], gp2["grader_model_source"]) == ("other-judge", "override")
    monkeypatch.setattr(sys, "argv", base + ["--dry-run"])
    grokmt_grade.main()
    doc3 = json.loads(out.read_text())
    gp3 = doc3["grading_provenance"]
    assert doc3["model"] == "DRY-RUN-no-api" and gp3["dry_run"] is True
    assert doc3["grader_run_stats"] is None   # no judge was built, so no pool to record
    assert (gp3["grader_model"], gp3["grader_model_source"], gp3["rubric_version"]) == \
        ("DRY-RUN-no-api", "dry_run", None)
    assert gp3["inputs"] == gp["inputs"] and gp3["run_identity_sha256"] == gp["run_identity_sha256"]
    # no identity record beside the private root -> null, never a guess; an
    # explicit --run-identity that does not exist refuses
    ident.unlink()
    monkeypatch.setattr(sys, "argv", base + ["--dry-run"])
    grokmt_grade.main()
    gp4 = json.loads(out.read_text())["grading_provenance"]
    assert gp4["run_identity_sha256"] is None and gp4["run_identity_path"] is None
    monkeypatch.setattr(sys, "argv", base + ["--dry-run", "--run-identity", str(ident)])
    with pytest.raises(SystemExit, match="no such file"):
        grokmt_grade.main()
    # the digest helper reads the same glob load_gated_dir does; an absent
    # file is null (a cell whose shard never persisted a response)
    (sd / "responses.jsonl").unlink()
    assert grokmt_grade.shard_digests(sd.parent) == {"shard00": {
        "responses_sha256": None,
        "progress_sha256": _h.sha256((sd / "progress.jsonl").read_bytes()).hexdigest()}}
    assert grokmt_grade.shard_digests(tmp_path / "nowhere") == {}


def test_target_cap_and_timeout_flags_default_to_the_config_and_override_when_given():
    """The 4,096-cap multi-turn re-run (2026-09-11): the target cap lived only in the
    config (attacks.target_max_new_tokens 512); the flag overrides it, refuses nonsense, and the
    timeout / note flags default to None (engine default, no key) so nothing changes unless asked."""
    import argparse
    p = argparse.ArgumentParser()
    jailbreak_run._add_target_engine_args(p)
    ns = p.parse_args([])
    assert ns.target_max_new_tokens is None and ns.target_timeout_s is None and ns.target_decode_note is None
    cfg = {"attacks": {"target_max_new_tokens": 512}}
    assert jailbreak_run._target_max_new_tokens(cfg) == 512
    assert jailbreak_run._target_max_new_tokens(cfg, ns) == 512
    ns2 = p.parse_args(["--target-max-new-tokens", "4096", "--target-timeout-s", "3600",
                        "--target-decode-note", "vllm_defaults"])
    assert jailbreak_run._target_max_new_tokens(cfg, ns2) == 4096
    assert ns2.target_timeout_s == 3600.0 and ns2.target_decode_note == "vllm_defaults"
    assert jailbreak_run._target_max_new_tokens({}, None) == 512          # the pre-full22 default
    assert jailbreak_run._target_max_new_tokens({"bon": {"max_new_tokens": 256}}) == 256
    with pytest.raises(SystemExit):
        jailbreak_run._target_max_new_tokens(cfg, p.parse_args(["--target-max-new-tokens", "0"]))


# --- the CLI's engine ordering against an EXISTING lens store ----------------------------------------------------------


def _digest(d: Path) -> dict:
    import hashlib
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


def _lens_store(out_dir: Path) -> dict:
    """A lens-era served store: a covered goal, a lens row and a lens manifest (neither records an engine — lens by
    construction) and the target's timing / call logs — every file this CLI path rewrites (the gated caller adds the
    attacker manifest it wrote)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "progress.jsonl").write_text(json.dumps({"goal_id": "g0", "outcome": "done"}) + "\n")
    (out_dir / "responses.jsonl").write_text(json.dumps({"goal_id": "g0", "sample_idx": 0, "response": "r",
                                                         "engine": "served"}) + "\n")
    (out_dir / "served_target.json").write_text(json.dumps({"target_engine": "served", "manifest_stage": "final",
                                                            "decode": None}))
    (out_dir / "target_timing.jsonl").write_text("")
    (out_dir / "attacker_calls.jsonl").write_text("")
    return _digest(out_dir)


def test_cmd_generate_native_target_onto_a_lens_store_refuses_before_any_write_on_both_paths(monkeypatch, tmp_path):
    """Review r1 (e), the ACTUAL CLI ordering: a served target on the NATIVE engine pointed at an existing LENS store
    refuses before the CLI writes a byte — on the gated (multi-turn) path AND the static/replayable path — and every
    file is byte-identical afterwards; the attack never runs. Before the fix both paths called the decode compare
    WITHOUT the engine and stamped a native ``served_target.json`` over the lens manifest (then again in the
    ``finally``): lens responses under a native manifest, and an ordinary lens resume then refused its own store. The
    lens twin of each run RESUMES the store and stamps the manifest with its engine."""
    NATIVE, LENS = st.SERVING_ENGINE_LORA_NATIVE, st.SERVING_ENGINE_LENS
    fake = _FakeServed()
    fake.serving_engine = NATIVE
    built, rec, ids, cfg = _wire_served(monkeypatch, tmp_path, fake)
    root = jailbreak_run._private_root(cfg, str(tmp_path / "state"))

    # the gated path (crescendo)
    # the co-resident HF attacker (the default): its identity is RECORDED (attacker_quantization 'none'), so the lens
    # twin's resume passes the attacker-identity gate; a vLLM attacker with no serving record is 'unrecorded' and
    # would refuse every resume of a store with work, before the engine question is even reached
    args = _served_args(tmp_path, ids, defense=TT_PSR, attacker_engine="hf", attacker_endpoint=None)
    out_dir = jailbreak_run._gated_shard_dir(cfg, root, "crescendo", TT_PSR, 1)
    jailbreak_run._write_engine_manifest(out_dir, cfg, args, "crescendo")   # the lens-era attacker manifest, as written
    before = _lens_store(out_dir)
    with pytest.raises(st.ServedTargetError, match="lens by construction"):
        jailbreak_run.cmd_generate(args, cfg)
    assert _digest(out_dir) == before, "the native run rewrote a file of the lens store"
    assert rec == {}, "the attack ran against the other engine's store"
    fake.serving_engine = LENS
    jailbreak_run.cmd_generate(args, cfg)
    doc = json.loads((out_dir / "served_target.json").read_text())
    assert doc[st.SERVING_ENGINE_FIELD] == LENS and doc["manifest_stage"] == "final" and rec["attack"] == "crescendo"

    # the static / replayable path (pasttense, greedy by the explicit opt-in; the family itself is faked)
    ran = {}

    def fake_family(attack, defense, goals_arg, cfg_in, out_dir_in, *, trial=0, resume=True, **extra):
        ran.update(attack=attack, out_dir=Path(out_dir_in), goals=[g.goal_id for g in goals_arg])
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg), "n_target_forwards_total": 0}

    monkeypatch.setattr(jailbreak_run, "_run_replayable_family", fake_family)
    fake.serving_engine = NATIVE
    sargs = _served_args(tmp_path, ids, attack="pasttense", defense=TT_PSR, shard=0, nshards=1, greedy=True,
                         attacker_engine="hf", attacker_endpoint=None)
    sdir = jailbreak_run._gated_shard_dir(cfg, root, "pasttense", TT_PSR, 0)
    sbefore = _lens_store(sdir)
    with pytest.raises(st.ServedTargetError, match="lens by construction"):
        jailbreak_run.cmd_generate(sargs, cfg)
    assert _digest(sdir) == sbefore, "the native static run rewrote a file of the lens store"
    assert ran == {}
    fake.serving_engine = LENS
    jailbreak_run.cmd_generate(sargs, cfg)
    sdoc = json.loads((sdir / "served_target.json").read_text())
    assert sdoc[st.SERVING_ENGINE_FIELD] == LENS and sdoc["manifest_stage"] == "final" and ran["attack"] == "pasttense"
    assert ran["out_dir"] == sdir
