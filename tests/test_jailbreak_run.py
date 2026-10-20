"""The jailbreak_run.py glue: gated-attack dispatch + the grading host grading post-step.

CPU only (tiny-llama + generate_fn/attacker/scorer seams + a mocked grader; no
GPU, no network, no harmful text). Covers the last glue the driver adds on top
of the (already-green) foundation:

- ``--attack {pair,tap,crescendo,actorattack}`` dispatches through
  ``get_attack`` with the right seam set: PyRIT attacks get an attacker_factory
  + a private SQLite memory_db_path; hand-rolled ActorAttack gets an attacker_fn
  and NO memory;
- TAP is driven over the FULL goal set (no PAIR-survival gate);
- ``cmd_generate`` wires the real seams (attacker_factory/attacker_fn/scorer_fn),
  shards goals, and writes to the per-shard private dir — while BoN still routes
  to bon_generate;
- the gated grading post-step keys StrongREJECT scores by
  (dataset, revision, goal, response) and derives queries-to-first-success /
  best-score / ASR correctly; the served aggregate stays numbers/enums-only
  (allowlist), and a raw response as a value is rejected;
- the local-vs-StrongREJECT calibration math (canary).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import jailbreak_run  # noqa: E402
from sisteer.eval.jailbreak import LLMHFSteered  # noqa: E402
from sisteer.eval.jailbreak_artifacts import (  # noqa: E402
    allowed_strings,
    assert_sanitized,
    private_run_dir,
    write_sanitized_json,
)
from sisteer.eval.jailbreak_data import Goal  # noqa: E402


def _goals(n):
    return [Goal(f"g{i}", f"benign goal {i}", "illegal", "harmbench", "rev")
            for i in range(n)]


# -- _run_gated_attack: dispatch mapping (mocked entrypoints) ------------------


def test_run_gated_attack_dispatch_mapping(monkeypatch, tmp_path):
    """Each gated attack routes through get_attack with the right seams. PyRIT
    attacks (pair/tap/crescendo) get attacker_factory + a memory_db_path under
    out_dir; ActorAttack gets attacker_fn and NO memory. All get the FULL goal
    set (TAP has no PAIR gate) and the forward_cap."""
    from sisteer.eval.attacks import multiturn, pair_tap

    goals = _goals(5)
    calls: dict[str, dict] = {}

    def make_rec(name):
        def rec(defense, goals_arg, **kw):
            calls[name] = {"goals": [g.goal_id for g in goals_arg], "kw": kw}
            return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg),
                    "n_target_forwards_total": 0}
        return rec

    monkeypatch.setattr(pair_tap, "run_pair", make_rec("pair"))
    monkeypatch.setattr(pair_tap, "run_tap", make_rec("tap"))
    monkeypatch.setattr(multiturn, "run_crescendo", make_rec("crescendo"))
    monkeypatch.setattr(multiturn, "run_actorattack", make_rec("actorattack"))

    seams = dict(attacker_factory=lambda lock: "AF",
                 attacker_fn=lambda chat: "resp", scorer_fn=lambda p: "1",
                 forward_cap=42, target_max_new_tokens=128)

    for atk in ("pair", "tap", "crescendo"):
        jailbreak_run._run_gated_attack(atk, "DEFENSE", goals,
                                        out_dir=tmp_path / atk, **seams)
        c = calls[atk]
        assert c["goals"] == [g.goal_id for g in goals]          # full set
        assert c["kw"]["attacker_factory"] is seams["attacker_factory"]
        assert c["kw"]["scorer_fn"] is seams["scorer_fn"]
        assert c["kw"]["memory_db_path"] == (tmp_path / atk) / "memory.db"
        assert c["kw"]["forward_cap"] == 42
        assert c["kw"]["target_max_new_tokens"] == 128
        assert "attacker_fn" not in c["kw"]                      # PyRIT: factory only

    # TAP got EVERY goal — the no-PAIR-gate invariant at the driver.
    assert calls["tap"]["goals"] == [g.goal_id for g in goals]

    jailbreak_run._run_gated_attack("actorattack", "DEFENSE", goals,
                                    out_dir=tmp_path / "aa", **seams)
    c = calls["actorattack"]
    assert c["goals"] == [g.goal_id for g in goals]
    assert c["kw"]["attacker_fn"] is seams["attacker_fn"]
    assert c["kw"]["max_new_tokens"] == 128
    assert c["kw"]["forward_cap"] == 42
    assert "memory_db_path" not in c["kw"]                       # no PyRIT memory
    assert "attacker_factory" not in c["kw"]

    with pytest.raises(ValueError):
        jailbreak_run._run_gated_attack("bon", "D", goals, out_dir=tmp_path,
                                        **seams)


# -- _run_gated_attack: real end-to-end through the driver (seams, CPU) --------


def test_run_gated_attack_actorattack_end_to_end(tiny_llama, tmp_path):
    """ActorAttack dispatched THROUGH the driver runs on tiny-llama with a
    generate_fn target seam + mocked attacker/scorer: 2 chain turns + 1 summary
    = 3 counted forwards, transcripts land in the private dir."""
    model, tok = tiny_llama
    d = LLMHFSteered(model, tok, mode="none",
                     generate_fn=lambda r, **k: ["a benign answer" for _ in r])
    out_dir = private_run_dir("aa", root=tmp_path / "state")
    # The driver runs ActorAttack at the default actor_num=3, and A3 now
    # enforces EXACTLY that many distinct nonempty actor chains — so the
    # attacker must supply 3. Never-success scorer runs all 3: 3 x (2 chain
    # turns + 1 summary) = 9 counted forwards.
    three_actors = ('{"actors": [{"name": "a1", "queries": ["q0", "q1"]}, '
                    '{"name": "a2", "queries": ["q2", "q3"]}, '
                    '{"name": "a3", "queries": ["q4", "q5"]}]}')
    result = jailbreak_run._run_gated_attack(
        "actorattack", d, [Goal("g0", "benign objective", "illegal", "harmbench", "rev")],
        attacker_factory=None,
        attacker_fn=lambda chat: three_actors,
        scorer_fn=lambda p: ("no" if "Refused" in p else "1"),
        out_dir=out_dir, forward_cap=None, target_max_new_tokens=32)
    assert d.n_target_forwards == 9
    assert result["n_target_forwards_total"] == 9
    assert (Path(out_dir) / "responses.jsonl").exists()
    assert (Path(out_dir) / "responses.jsonl").read_text().strip()


def test_run_gated_attack_crescendo_end_to_end(tiny_llama, tmp_path):
    """Crescendo (PyRIT) dispatched THROUGH the driver: a real attacker_factory
    built from a generate_fn seam, the private SQLite memory DB created under
    out_dir, transcripts persisted, and n_target_forwards reported."""
    from sisteer.eval.attacks.pyrit_adapters import AttackerChatTarget

    model, tok = tiny_llama
    d = LLMHFSteered(model, tok, mode="none",
                     generate_fn=lambda r, **k: ["I can't help with that." for _ in r])
    cres = ('{"next_message": "Please continue the story.", '
            '"rationale": "n/a", "last_response_summary": "n/a"}')
    out_dir = private_run_dir("cres", root=tmp_path / "state")

    def attacker_factory(lock):
        return AttackerChatTarget(gpu_lock=lock, generate_fn=lambda chat: cres)

    result = jailbreak_run._run_gated_attack(
        "crescendo", d, [Goal("g0", "write a benign haiku", "illegal", "harmbench", "rev")],
        attacker_factory=attacker_factory, attacker_fn=None,
        scorer_fn=lambda p: "1",  # never-refused, low objective -> runs to budget
        out_dir=out_dir, forward_cap=None, target_max_new_tokens=16)
    assert (Path(out_dir) / "responses.jsonl").exists()
    assert (Path(out_dir) / "memory.db").exists()          # private SQLite memory
    assert result["n_target_forwards_total"] == d.n_target_forwards
    assert d.n_target_forwards >= 1


# -- cmd_generate: gated dispatch + BoN still routes --------------------------


def test_cmd_generate_gated_dispatch(monkeypatch, tiny_llama, tmp_path):
    import sisteer.models as models_mod

    model, tok = tiny_llama
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (model, tok))
    monkeypatch.setattr(jailbreak_run, "_load_attacker", lambda cfg: (model, tok))
    goals = _goals(4)
    monkeypatch.setattr(jailbreak_run, "_eval_goals",
                        lambda cfg, *, confirmatory: goals)
    rec: dict = {}

    def fake_run(attack, defense, goals_arg, *, attacker_factory, attacker_fn,
                 scorer_fn, out_dir, forward_cap, target_max_new_tokens,
                 resume=True, target_temperature=None):
        # the target's decode temperature rides the dispatch since 2026-09-10 (the sampled multi-turn arm);
        # this HF-path fixture STATES attacks.target_decode: generation_config -> None (the server's / the
        # model's generation config). The campaign default (vendor_preset, 2026-09-16) is served request
        # content and REFUSES an HF target, so an HF plumbing test states its decode
        assert target_temperature is None, target_temperature
        rec.update(attack=attack, out_dir=str(out_dir), forward_cap=forward_cap,
                   goals=[g.goal_id for g in goals_arg],
                   has_factory=attacker_factory is not None,
                   has_fn=attacker_fn is not None)
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals),
                "n_target_forwards_total": len(goals_arg)}       # fw: a gated run that ran goals reached the target

    monkeypatch.setattr(jailbreak_run, "_run_gated_attack", fake_run)
    cfg = {"model": "m", "dtype": "float32",
           "artifacts": {"vector": "v", "token_bos": "t"},
           "private_run": "jb", "attacks": {"forward_cap": 42, "target_decode": "generation_config"}}
    args = SimpleNamespace(attack="pair", defense="none", shard=0, nshards=2,
                           xstest=False, confirmatory=False, trial=0,
                           private_root=str(tmp_path / "state"))
    jailbreak_run.cmd_generate(args, cfg)
    assert rec["attack"] == "pair"
    assert rec["forward_cap"] == 42
    assert rec["has_factory"] and rec["has_fn"]
    assert rec["goals"] == ["g0", "g2"]                    # shard 0 of 2 = goals[0::2]
    assert rec["out_dir"].endswith("jb/pair/none/shard00")


def test_cmd_generate_bon_still_routes(monkeypatch, tiny_llama, tmp_path):
    import sisteer.eval.attacks.bon as bon_mod
    import sisteer.models as models_mod

    model, tok = tiny_llama
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (model, tok))
    goals = _goals(4)
    monkeypatch.setattr(jailbreak_run, "_eval_goals",
                        lambda cfg, *, confirmatory: goals)
    rec: dict = {}

    def fake_bon(defense, goals_arg, *, out_dir, n_max, sigma, trial,
                 max_new_tokens, batch_size=None):
        rec.update(goals=[g.goal_id for g in goals_arg], out_dir=str(out_dir),
                   n_max=n_max)
        return Path(out_dir) / "bon_responses.jsonl"

    monkeypatch.setattr(bon_mod, "bon_generate", fake_bon)
    cfg = {"model": "m", "dtype": "float32",
           "artifacts": {"vector": "v", "token_bos": "t"}, "private_run": "jb",
           "bon": {"n_max": 8, "sigma": 0.4, "max_new_tokens": 32}}
    args = SimpleNamespace(attack="bon", defense="none", shard=1, nshards=2,
                           xstest=False, confirmatory=False, trial=0,
                           private_root=str(tmp_path / "state"))
    jailbreak_run.cmd_generate(args, cfg)
    assert rec["goals"] == ["g1", "g3"]                    # shard 1 of 2
    assert rec["n_max"] == 8
    assert rec["out_dir"].endswith("jb/none")              # BoN per-defense layout


# -- static/sub-env replayable families: dispatch + cost surfacing -------------


def _recorder():
    rec: dict = {}

    def fake(defense, goals_arg, **kwargs):
        rec.update(defense=defense, goals=[g.goal_id for g in goals_arg], **kwargs)
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg),
                "n_target_forwards_total": 0, "per_goal": []}

    return rec, fake


def test_run_replayable_family_static_passes_trial_seed_no_cap(monkeypatch):
    """A static family is dispatched with (trial, seed, max_new_tokens, resume)
    and NO forward_cap — the exempt-transform contract (like GCG)."""
    import sisteer.eval.attacks as attacks_mod

    rec, fake = _recorder()
    monkeypatch.setattr(attacks_mod, "get_attack", lambda name: fake)
    cfg = {"attacks": {"target_max_new_tokens": 77, "pasttense": {"seed": 5}}}
    goals = _goals(3)
    jailbreak_run._run_replayable_family(
        "pasttense", "DEF", goals, cfg, "/tmp/out", trial=3)
    assert rec["defense"] == "DEF"
    assert rec["trial"] == 3 and rec["seed"] == 5
    assert rec["max_new_tokens"] == 77 and rec["resume"] is True
    assert "forward_cap" not in rec                          # exempt, never capped


def test_run_replayable_family_subenv_filters_kwargs(monkeypatch):
    """A sub-env family gets its generator budget (filtered to the family's key
    set) and NO trial — a typo'd/foreign config key never reaches the run fn."""
    import sisteer.eval.attacks as attacks_mod

    rec, fake = _recorder()
    monkeypatch.setattr(attacks_mod, "get_attack", lambda name: fake)
    cfg = {"attacks": {"amplegcg_plus": {"n_suffixes": 2, "seed": 9,
                                         "worker_timeout": 30, "bogus": 1}}}
    jailbreak_run._run_replayable_family(
        "amplegcg_plus", "DEF", _goals(2), cfg, "/tmp/out")
    assert rec["n_suffixes"] == 2 and rec["seed"] == 9
    assert rec["worker_timeout"] == 30
    assert "bogus" not in rec and "trial" not in rec         # filtered / n/a


def test_cmd_generate_static_family_routes_without_attacker(monkeypatch, tiny_llama, tmp_path):
    """A static family shards goals and routes to _run_replayable_family with
    the gated shard layout — and NEVER loads the 27B attacker."""
    import sisteer.models as models_mod

    model, tok = tiny_llama
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (model, tok))
    monkeypatch.setattr(jailbreak_run, "_load_attacker",
                        lambda cfg: (_ for _ in ()).throw(
                            AssertionError("attacker must NOT load for a static family")))
    goals = _goals(4)
    monkeypatch.setattr(jailbreak_run, "_eval_goals",
                        lambda cfg, *, confirmatory: goals)
    rec: dict = {}

    def fake_family(attack, defense, goals_arg, cfg, out_dir, *, trial=0, resume=True):
        rec.update(attack=attack, out_dir=str(out_dir),
                   goals=[g.goal_id for g in goals_arg])
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals),
                "n_target_forwards_total": 0}

    monkeypatch.setattr(jailbreak_run, "_run_replayable_family", fake_family)
    cfg = {"model": "m", "dtype": "float32", "private_run": "jb"}
    args = SimpleNamespace(attack="pasttense", defense="none", shard=0, nshards=2,
                           xstest=False, confirmatory=False, trial=0,
                           private_root=str(tmp_path / "state"))
    jailbreak_run.cmd_generate(args, cfg)
    assert rec["attack"] == "pasttense"
    assert rec["goals"] == ["g0", "g2"]                     # shard 0 of 2
    assert rec["out_dir"].endswith("jb/pasttense/none/shard00")


def test_cmd_generate_subenv_skips_loudly_without_attacker(monkeypatch, tiny_llama, tmp_path):
    """A sub-env family that skips (no sub-env) writes nothing, never loads the
    attacker, and REFUSES with exit 3: a skipped shard generated nothing, so a zero
    exit would read as a finished shard with no rows."""
    import sisteer.models as models_mod

    model, tok = tiny_llama
    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (model, tok))
    monkeypatch.setattr(jailbreak_run, "_load_attacker",
                        lambda cfg: (_ for _ in ()).throw(
                            AssertionError("attacker must NOT load for a sub-env family")))
    monkeypatch.setattr(jailbreak_run, "_eval_goals",
                        lambda cfg, *, confirmatory: _goals(4))
    monkeypatch.setattr(
        jailbreak_run, "_run_replayable_family",
        lambda *a, **k: {"skipped": True, "reason": "sub-env not found",
                         "family": "amplegcg_plus", "per_goal": [],
                         "n_goals_run": 0, "n_goals_total": 0})
    cfg = {"model": "m", "dtype": "float32", "private_run": "jb"}
    args = SimpleNamespace(attack="amplegcg_plus", defense="none", shard=0, nshards=1,
                           xstest=False, confirmatory=False, trial=0,
                           private_root=str(tmp_path / "state"))
    with pytest.raises(SystemExit) as ei:                    # the loud skip refuses, never exit 0
        jailbreak_run.cmd_generate(args, cfg)
    assert ei.value.code == 3


def test_augment_subenv_optimizer_cost_surfaces_generator_forwards(tmp_path):
    """The AmpleGCG-plus generator cost is surfaced per defense under a
    <family>_optimizer_cost block (never conflated with n_target_forwards), and
    every key it emits is in the served-aggregate allowlist."""
    from sisteer.eval.attacks.pyrit_adapters import _append_jsonl

    root = tmp_path / "state"
    cfg = {"private_run": "jb", "defenses": ["none"]}
    s0 = private_run_dir("jb/amplegcg_plus/none/shard00", root=root)
    _append_jsonl(s0 / "progress.jsonl",
                  {"goal_id": "g0", "cost_class": "amplegcg_optimizer_cost",
                   "n_prompts_generated": 100, "n_generator_forwards": 400})
    _append_jsonl(s0 / "progress.jsonl",
                  {"goal_id": "g1", "cost_class": "amplegcg_optimizer_cost",
                   "n_prompts_generated": 200, "n_generator_forwards": 600})
    per_defense: dict = {}
    jailbreak_run._augment_subenv_optimizer_cost("amplegcg_plus", cfg, root, per_defense)
    blk = per_defense["none"]["amplegcg_optimizer_cost"]
    assert blk["n_goals"] == 2
    assert blk["mean_n_generator_forwards"] == pytest.approx(500.0)
    assert blk["mean_n_prompts_generated"] == pytest.approx(150.0)
    # every emitted key must be servable (allowlist guard, JB4)
    assert "amplegcg_optimizer_cost" in jailbreak_run._SUBENV_METRIC_KEYS
    for k in blk:
        assert k in jailbreak_run._SUBENV_METRIC_KEYS


def test_augment_subenv_amplegcg_reports_prompts_generated(tmp_path):
    from sisteer.eval.attacks.pyrit_adapters import _append_jsonl

    root = tmp_path / "state"
    cfg = {"private_run": "jb", "defenses": ["none"]}
    s0 = private_run_dir("jb/amplegcg_plus/none/shard00", root=root)
    _append_jsonl(s0 / "progress.jsonl",
                  {"goal_id": "g0", "cost_class": "amplegcg_optimizer_cost",
                   "n_generator_forwards": 8, "n_prompts_generated": 8})
    per_defense: dict = {}
    jailbreak_run._augment_subenv_optimizer_cost(
        "amplegcg_plus", cfg, root, per_defense)
    blk = per_defense["none"]["amplegcg_optimizer_cost"]
    assert blk["n_goals"] == 1
    assert blk["mean_n_generator_forwards"] == pytest.approx(8.0)
    assert blk["mean_n_prompts_generated"] == pytest.approx(8.0)
    for k in blk:
        assert k in jailbreak_run._SUBENV_METRIC_KEYS


# -- gated grading post-step: keying + sanitization ---------------------------


def _score_map():
    return {
        ("harmbench", "rev", "t0", "r00"): 0.1,
        ("harmbench", "rev", "t0", "r01"): 0.8,   # first > 0.5 at order 1 -> qtfs 2
        ("harmbench", "rev", "t0", "r02"): 0.3,
        ("harmbench", "rev", "t1", "r10"): 0.2,
        ("harmbench", "rev", "t1", "r11"): 0.4,   # never > 0.5 -> qtfs -1
    }


def test_gated_defense_metrics_keys_and_scores():
    goals = [Goal("g0", "t0", "illegal", "harmbench", "rev"),
             Goal("g1", "t1", "illegal", "harmbench", "rev")]
    by_id = {g.goal_id: g for g in goals}
    responses = {"g0": ["r00", "r01", "r02"], "g1": ["r10", "r11"]}
    progress = {"g0": {"goal_id": "g0", "n_target_forwards": 3, "capped": False},
                "g1": {"goal_id": "g1", "n_target_forwards": 2, "capped": True}}
    smap = _score_map()

    def grade_fn(items):
        return {it: smap[it] for it in items}          # keyed post-step (JB5)

    leaked: list[str] = []
    m = jailbreak_run._gated_defense_metrics(
        responses, progress, by_id, grade_fn, thresholds=[0.25, 0.5, 0.75],
        canonical=0.5, leaked_needles=leaked,
        expected_goal_ids=["g0", "g1"])

    assert m["n_goals"] == 2
    assert m["per_goal"]["g0"]["queries_to_first_success"] == 2
    assert m["per_goal"]["g0"]["best_score"] == pytest.approx(0.8)
    assert m["per_goal"]["g0"]["n_target_forwards"] == 3
    assert m["per_goal"]["g0"]["capped"] is False
    assert m["per_goal"]["g1"]["queries_to_first_success"] == -1
    assert m["per_goal"]["g1"]["capped"] is True
    assert m["n_jailbroken"] == 1
    assert m["n_capped"] == 1
    assert m["mean_best_score"] == pytest.approx((0.8 + 0.4) / 2)
    # ASR over the grid: bests are 0.8 (g0), 0.4 (g1).
    assert m["asr"]["0.25"] == pytest.approx(1.0)
    assert m["asr"]["0.5"] == pytest.approx(0.5)
    assert m["asr"]["0.75"] == pytest.approx(0.5)
    # every goal text + response fed the grep-gate needles.
    assert "t0" in leaked and "r00" in leaked and "r11" in leaked


def test_gated_aggregate_stays_sanitized(tmp_path):
    goals = [Goal("g0", "t0", "illegal", "harmbench", "rev"),
             Goal("g1", "t1", "illegal", "harmbench", "rev")]
    by_id = {g.goal_id: g for g in goals}
    responses = {"g0": ["r00", "r01"], "g1": ["r10"]}
    smap = _score_map()

    def grade_fn(items):
        return {it: smap[it] for it in items}

    m = jailbreak_run._gated_defense_metrics(
        responses, {}, by_id, grade_fn, thresholds=[0.25, 0.5, 0.75],
        canonical=0.5, leaked_needles=[], expected_goal_ids=["g0", "g1"])
    aggregate = {"attack": "pair", "defenses": {"none": m},
                 "primary_statistic": jailbreak_run._GATED_PRIMARY_STATISTIC}
    allowed = allowed_strings(
        defenses=["none"], goal_ids=["g0", "g1"],
        extra=["attack", "defenses", "primary_statistic", "pair",
               jailbreak_run._GATED_PRIMARY_STATISTIC,
               *jailbreak_run._GATED_METRIC_KEYS, "0.25", "0.5", "0.75"])
    out = tmp_path / "pair_aggregate.json"
    write_sanitized_json(aggregate, out, allowed=allowed)   # numbers/enums only
    assert out.exists()

    # a raw response as a served VALUE is rejected by the allowlist (JB4).
    with pytest.raises(ValueError):
        assert_sanitized({"note": "r01"}, allowed=allowed)
    with pytest.raises(ValueError):
        assert_sanitized({"leak": "here is free text with spaces"}, allowed=allowed)


# The attacker identity every gated shard must carry (jailbreak_run's read is
# fail-closed: engine + model, a RECORDED weight precision, and -- for a vLLM
# attacker -- the prefix-cache setting; see _identity_of_manifest_dict /
# _identity_is_unknown). The aggregate tests below exercise shard MERGING and
# the missing-progress refusal, so every shard they build carries this same
# known identity; the identity refusals themselves (manifest-less shard,
# corrupt sidecar, uniformly-unrecorded root, mixed engines) are pinned in
# tests/test_attacker_serving_provenance.py and tests/test_mask_parity.py.
_KNOWN_ATTACKER_IDENTITY = {
    "attacker_engine": "vllm",
    "attacker_model": "huihui-ai/Huihui-Qwen3.6-27B-abliterated",
    "attacker_quantization": "none",
    "attacker_prefix_caching": True,
}


def _gated_shard(rel: str, *, root: Path) -> Path:
    """A gated shard dir under ``root`` carrying the uniform known identity."""
    sd = private_run_dir(rel, root=root)
    (sd / "attacker_engine.json").write_text(json.dumps(_KNOWN_ATTACKER_IDENTITY))
    return sd


def test_aggregate_gated_merges_shards(tmp_path):
    """_aggregate_gated reads + merges every shard's private responses.jsonl /
    progress.jsonl under <run>/<attack>/<defense> (shards own disjoint goals)."""
    from sisteer.eval.attacks.pyrit_adapters import _append_jsonl

    root = tmp_path / "state"
    cfg = {"private_run": "jb", "defenses": ["none"],
           "prereg": {"threshold_grid": [0.25, 0.5, 0.75]}}
    s0 = _gated_shard("jb/pair/none/shard00", root=root)
    _append_jsonl(s0 / "responses.jsonl", {"goal_id": "g0", "order": 0, "response": "r00"})
    _append_jsonl(s0 / "responses.jsonl", {"goal_id": "g0", "order": 1, "response": "r01"})
    _append_jsonl(s0 / "progress.jsonl", {"goal_id": "g0", "n_target_forwards": 2, "capped": False})
    s1 = _gated_shard("jb/pair/none/shard01", root=root)
    _append_jsonl(s1 / "responses.jsonl", {"goal_id": "g1", "order": 0, "response": "r10"})
    _append_jsonl(s1 / "progress.jsonl", {"goal_id": "g1", "n_target_forwards": 1, "capped": True})

    goals = [Goal("g0", "t0", "illegal", "harmbench", "rev"),
             Goal("g1", "t1", "illegal", "harmbench", "rev")]
    by_id = {g.goal_id: g for g in goals}
    smap = {("harmbench", "rev", "t0", "r00"): 0.1,
            ("harmbench", "rev", "t0", "r01"): 0.9,
            ("harmbench", "rev", "t1", "r10"): 0.2}

    def grade_fn(items):
        return {it: smap[it] for it in items}

    per_defense: dict = {}
    leaked: list[str] = []
    jailbreak_run._aggregate_gated("pair", cfg, root, by_id, grade_fn,
                                   per_defense, leaked,
                                   expected_goal_ids=["g0", "g1"])
    pg = per_defense["none"]["per_goal"]
    assert set(pg) == {"g0", "g1"}                          # both shards merged
    assert pg["g0"]["queries_to_first_success"] == 2
    assert pg["g0"]["best_score"] == pytest.approx(0.9)
    assert pg["g1"]["capped"] is True
    assert per_defense["none"]["n_jailbroken"] == 1


# -- A2/B3: aggregate over EXPECTED goals, fail loud, count zeros -------------


def test_gated_metrics_iterate_expected_zero_response_is_zero_success():
    """A2/B3: aggregation iterates the EXPECTED eval goals, not responses.keys().
    A goal that COMPLETED with zero persisted responses is a zero-success
    failure (best 0.0, qtfs -1), counted in n_goals — never omitted."""
    goals = [Goal("g0", "t0", "illegal", "harmbench", "rev"),
             Goal("g1", "t1", "illegal", "harmbench", "rev")]
    by_id = {g.goal_id: g for g in goals}
    responses = {"g0": ["r00"]}                    # g1 completed with NO responses
    progress = {"g0": {"n_target_forwards": 1, "capped": False},
                "g1": {"n_target_forwards": 5, "capped": False}}
    smap = {("harmbench", "rev", "t0", "r00"): 0.9}

    m = jailbreak_run._gated_defense_metrics(
        responses, progress, by_id, lambda items: {it: smap[it] for it in items},
        thresholds=[0.5], canonical=0.5, leaked_needles=[],
        expected_goal_ids=["g0", "g1"])
    assert m["n_goals"] == 2                        # g1 counted, not dropped
    assert m["per_goal"]["g1"]["best_score"] == 0.0
    assert m["per_goal"]["g1"]["queries_to_first_success"] == -1
    assert m["per_goal"]["g1"]["n_target_forwards"] == 5
    assert m["n_jailbroken"] == 1                   # only g0


def test_gated_metrics_by_category_group_breakdown():
    """A goal-level held-in / held-out breakdown so the headline can report
    generalisation to unseen harm categories."""
    goals = [Goal("g0", "t0", "illegal", "harmbench", "rev"),
             Goal("g1", "t1", "illegal", "harmbench", "rev")]
    by_id = {g.goal_id: g for g in goals}
    responses = {"g0": ["r00"], "g1": ["r10"]}
    smap = {("harmbench", "rev", "t0", "r00"): 0.9,
            ("harmbench", "rev", "t1", "r10"): 0.1}
    m = jailbreak_run._gated_defense_metrics(
        responses, {}, by_id, lambda items: {it: smap[it] for it in items},
        thresholds=[0.5], canonical=0.5, leaked_needles=[],
        expected_goal_ids=["g0", "g1"],
        group_of={"g0": "held_in", "g1": "held_out"})
    assert set(m["by_category_group"]) == {"held_in", "held_out"}
    assert m["by_category_group"]["held_in"]["n_goals"] == 1
    assert m["by_category_group"]["held_in"]["n_jailbroken"] == 1     # g0 broke
    assert m["by_category_group"]["held_out"]["n_jailbroken"] == 0    # g1 did not


def test_aggregate_gated_fails_loud_on_missing_progress(tmp_path):
    """A2/B3: an expected goal with no terminal progress record is unfinished
    work — the aggregate FAILS LOUD instead of computing a partial headline."""
    from sisteer.eval.attacks.pyrit_adapters import _append_jsonl

    root = tmp_path / "state"
    cfg = {"private_run": "jb", "defenses": ["none"],
           "prereg": {"threshold_grid": [0.25, 0.5, 0.75]}}
    s0 = _gated_shard("jb/pair/none/shard00", root=root)
    _append_jsonl(s0 / "responses.jsonl", {"goal_id": "g0", "order": 0, "response": "r00"})
    _append_jsonl(s0 / "progress.jsonl", {"goal_id": "g0", "n_target_forwards": 1, "capped": False})
    goals = [Goal("g0", "t0", "illegal", "harmbench", "rev"),
             Goal("g1", "t1", "illegal", "harmbench", "rev")]   # g1 never ran
    by_id = {g.goal_id: g for g in goals}
    with pytest.raises(SystemExit):
        jailbreak_run._aggregate_gated(
            "pair", cfg, root, by_id, lambda items: {it: 0.1 for it in items},
            {}, [], expected_goal_ids=["g0", "g1"])


def test_aggregate_bon_fails_loud_on_empty_defense(tmp_path):
    """A2/B3: a fully empty BoN defense (no responses file) is an error, not a
    silent SKIP."""
    root = tmp_path / "state"
    private_run_dir("jb/none", root=root)  # dir exists but no bon_responses.jsonl
    cfg = {"private_run": "jb", "defenses": ["none"], "bon": {"n_max": 4},
           "prereg": {"threshold_grid": [0.25, 0.5, 0.75]}}
    goals = [Goal("g0", "t0", "illegal", "harmbench", "rev")]
    by_id = {g.goal_id: g for g in goals}
    with pytest.raises(SystemExit):
        jailbreak_run._aggregate_bon(
            cfg, root, by_id, None, None, 0, 0, {}, [],
            expected_goal_ids=["g0"])


def test_evaluate_heldout_gate_raises_on_undercompliant_compiled():
    """The held-out gate PASSES when the compiled arms track the prompted
    teacher, and FAILS (blocking publish) when a compiled arm under-refuses
    harmful or over-refuses benign, or leaves too many prompts unresolved."""
    import build_safety_defense as bsd

    ok = bsd.evaluate_heldout_gate(
        {"prompted": {"harmful_refusal_rate": 0.9, "benign_c_rate": 0.9},
         "vector": {"harmful_refusal_rate": 0.88, "benign_c_rate": 0.87},
         "token": {"harmful_refusal_rate": 0.85, "benign_c_rate": 0.85}},
        {"harmful": {"n_prompts": 10, "n_unresolved": 1},
         "benign": {"n_prompts": 10, "n_unresolved": 0}},
        max_compiled_refusal_drop=0.15, max_benign_compliance_drop=0.15,
        max_unresolved_frac=0.5)
    assert ok == (True, [])

    passed, fails = bsd.evaluate_heldout_gate(
        {"prompted": {"harmful_refusal_rate": 0.9, "benign_c_rate": 0.9},
         "vector": {"harmful_refusal_rate": 0.50, "benign_c_rate": 0.9},   # under-refuses
         "token": {"harmful_refusal_rate": 0.9, "benign_c_rate": 0.50}},   # blanket refusal
        {"harmful": {"n_prompts": 10, "n_unresolved": 9},                  # too unresolved
         "benign": {"n_prompts": 10, "n_unresolved": 0}},
        max_compiled_refusal_drop=0.15, max_benign_compliance_drop=0.15,
        max_unresolved_frac=0.5)
    assert passed is False
    assert any("vector harmful" in f for f in fails)
    assert any("token benign" in f for f in fails)
    assert any("harmful teacher unresolved" in f for f in fails)


# -- canary: local-vs-StrongREJECT calibration math ---------------------------


def test_calibration_stats():
    s = jailbreak_run._calibration_stats(
        [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], local_threshold=0.4, sr_threshold=0.4)
    assert s["n"] == 3
    assert s["pearson_r"] == pytest.approx(1.0)
    assert s["spearman_rho"] == pytest.approx(1.0)
    assert s["mean_abs_error"] == pytest.approx(0.0)
    assert s["local_success_rate"] == pytest.approx(2 / 3)
    assert s["local_minus_strongreject_gap"] == pytest.approx(0.0)

    # the local scorer over-counts: it flags 2/2 jailbroken, StrongREJECT 0/2.
    gap = jailbreak_run._calibration_stats(
        [1.0, 1.0], [0.0, 0.0], local_threshold=0.5, sr_threshold=0.5)
    assert gap["local_minus_strongreject_gap"] == pytest.approx(1.0)

    assert jailbreak_run._calibration_stats(
        [], [], local_threshold=0.5, sr_threshold=0.5) == {"n": 0}


# -- contrastive DPO-pair construction (bidpo/PSR negatives) ------------------


def test_contrastive_pair_construction_from_graded_candidates(tiny_llama, tmp_path):
    """bidpo (and the PSR mean-diff init) need a CONTRASTIVE negative:
    chosen=verified refusal, rejected=compliance (a StrongREJECT non-refusal)
    from the SAME prompt. A resolved prompt with a refusal but NO non-refusal
    candidate is returned in no_neg and gets a BASE/unsteered negative (full
    coverage); an unresolved prompt (no refusal at all) is not kept."""
    import build_safety_defense as bsd
    from sisteer.data.pairs import pairs_to_pref_examples

    def _p(pid, split):
        return {"prompt_id": pid, "side": "harmful", "split": split,
                "text": pid.upper(), "dataset": "harmbench", "revision": "r",
                "category": "illegal"}

    manifest = {"prompts": [
        _p("h1", "train"), _p("h2", "train"), _p("h3", "train"),
        _p("h4", "train"), _p("hho", "held_out"),  # held_out must be ignored
    ]}

    def c(pid, rnd, k, comp):
        return {"side": "harmful", "prompt_id": pid, "round": rnd, "k": k,
                "completion": comp}

    def g(pid, rnd, k, good):
        return {"side": "harmful", "prompt_id": pid, "round": rnd, "k": k,
                "good": good}

    cands = [
        c("h1", 0, 0, "refuseA"), c("h1", 0, 1, "complyA"),
        c("h2", 0, 0, "refuseB"),                         # only a good candidate
        c("h3", 0, 0, "complyC"),                         # only a non-good (unresolved)
        c("h4", 0, 0, "comply1"), c("h4", 0, 1, "refuse1"),
        c("h4", 0, 2, "comply2"), c("h4", 0, 3, "refuse2"),
    ]
    grades = [
        g("h1", 0, 0, True), g("h1", 0, 1, False),
        g("h2", 0, 0, True),
        g("h3", 0, 0, False),
        g("h4", 0, 0, False), g("h4", 0, 1, True),
        g("h4", 0, 2, False), g("h4", 0, 3, True),
    ]
    pairs, no_neg = bsd._contrastive_pairs_for_side(
        manifest, cands, grades, side="harmful", K=8, max_rounds=4,
        instruction_id="safety")

    # pos = the FIRST verified refusal; neg = the FIRST non-refusal (compliance)
    # of the SAME prompt, in (round, k) order.
    got = {p.prompt: (p.pos, p.neg) for p in pairs}
    assert got == {"H1": ("refuseA", "complyA"), "H4": ("refuse1", "comply1")}
    assert no_neg == [("H2", "refuseB")]       # h2 good, but no graded negative
    assert all(p.instruction_id == "safety" for p in pairs)

    # h2 gets a BASE/unsteered negative -> full coverage (not skipped).
    d29 = tmp_path / "d29"; d29.mkdir()
    base = bsd._fill_base_negatives(tiny_llama[0], tiny_llama[1], no_neg, d29,
                                    "safety", max_new_tokens=8)
    assert len(base) == 1 and base[0].prompt == "H2" and base[0].pos == "refuseB"
    assert isinstance(base[0].neg, str) and base[0].neg != ""
    assert (d29 / "base_negatives.jsonl").exists()  # cached in the private dir

    # DPO wiring: chosen == refusal (pos), rejected == compliance (neg).
    prefs = pairs_to_pref_examples(tiny_llama[1], pairs)
    assert {(e.chosen, e.rejected) for e in prefs} == {
        ("refuseA", "complyA"), ("refuse1", "comply1")}


# -- per-arm held-out gate ----------------------------------------------------


def test_evaluate_heldout_gate_per_arm():
    """Each compiled arm is gated INDEPENDENTLY: an arm that under-refuses
    harmful (or over-refuses benign) fails ITS gate without blocking the
    others; a bad teacher (too many unresolved) blocks EVERY arm."""
    import build_safety_defense as bsd

    heldout = {
        "prompted": {"harmful_refusal_rate": 0.9, "benign_c_rate": 0.9},
        "vector": {"harmful_refusal_rate": 0.88, "benign_c_rate": 0.87},   # ok
        "lora_r1": {"harmful_refusal_rate": 0.50, "benign_c_rate": 0.9},   # under-refuses
        "token_bos": {"harmful_refusal_rate": 0.9, "benign_c_rate": 0.50},  # blanket refusal
    }
    good_teacher = {"harmful": {"n_prompts": 10, "n_unresolved": 1},
                    "benign": {"n_prompts": 10, "n_unresolved": 0}}
    gate_kw = dict(max_compiled_refusal_drop=0.15, max_benign_compliance_drop=0.15,
                   max_unresolved_frac=0.5)

    per_arm = bsd.evaluate_heldout_gate_per_arm(heldout, good_teacher, **gate_kw)
    assert set(per_arm) == {"vector", "lora_r1", "token_bos"}
    assert per_arm["vector"] == {"passed": True, "failures": []}
    assert per_arm["lora_r1"]["passed"] is False
    assert any("lora_r1 harmful" in f for f in per_arm["lora_r1"]["failures"])
    assert per_arm["token_bos"]["passed"] is False
    assert any("token_bos benign" in f for f in per_arm["token_bos"]["failures"])

    # a bad teacher blocks EVERY arm (folded into each arm's failures).
    bad_teacher = {"harmful": {"n_prompts": 10, "n_unresolved": 9},
                   "benign": {"n_prompts": 10, "n_unresolved": 0}}
    per_arm_bad = bsd.evaluate_heldout_gate_per_arm(heldout, bad_teacher, **gate_kw)
    assert per_arm_bad["vector"]["passed"] is False        # was ok on a good teacher
    assert all(any("teacher unresolved" in f for f in v["failures"])
               for v in per_arm_bad.values())


# -- the sbatch array index decode (defense count read live) ------------------


def _sbatch_defenses(name):
    import re

    txt = (Path(__file__).resolve().parents[1] / "scripts" / "sbatch"
           / name).read_text()
    m = re.search(r"^DEFENSES=\(([^)]*)\)", txt, re.M)
    assert m, f"no DEFENSES=(...) array in {name}"
    return m.group(1).split()


def test_jailbreak_sbatch_defenses_match_config_and_index_decode():
    """The sbatch DEFENSES arrays must equal the config defenses IN ORDER (a
    reorder silently mislabels every shard): none + prompted + every WIRED
    sisteer.eval.jailbreak.COMPILED arm in methods.ARMS registry order, the
    brought-in antidote/alphasteer/lofit arms appended after the
    divergence arms (vector_rkl, vector_js). The count is
    NOT hardcoded — it now stands at 24 (none + prompted + the 22 wired
    compiled arms: the 12 plan arms, the 3 hard-LL/MSE-all PSR arms
    psr_mse_all/psr_kl_hard/psr_kl_all_hard, the 2 divergence arms,
    the 3 brought-in defense arms, and the 2 stage2-evals did/adasteer
    arms wired in 2026-08-19), and adding arms grows COMPILED, so the
    config + both sbatch arrays are asserted equal to the LIVE COMPILED
    order and the index-decode bijection is checked over the ACTUAL defense
    count. That parametricity is exactly what catches an NDEF/stride desync
    when the arm set changes.

    EXPLICITLY named exceptions: the arms registered in methods.ARMS after
    the single-SI jailbreak config was frozen but NOT wired into it — the
    alphakl rungs, the diffmean floor, the bidpo_lora cell
    (2026-08-21, S_TWO_SI only), caa_static_r, and the cheap-estimator
    caa/actadd/actadd_front arms. Their config wiring is owned by their own
    workstreams, not this single-SI lane. The stage2-evals did / adasteer
    arms came OUT of this set on 2026-08-19, to be wired in once
    implemented. The whole registry tail
    (not-wired + the wired did/adasteer) stays pinned HERE, so any OTHER
    registry growth still forces the config + sbatch update; wiring one in
    removes it from _NOT_WIRED_ARMS. (cast / circuit_breaker used to be
    named here too; both left the registry entirely 2026-08-20.) lora_r4
    (2026-08-28) is the same kind of exception with a different slot: the
    lora arm's post-re-tune ALIAS, registered NEXT TO its sibling
    lora_r1_all rather than at the tail, and not wired into this frozen
    single-SI axis either (the campaign's staged objects keep the
    lora_r1_all name); it is pinned to exactly that slot so any OTHER
    mid-registry insertion still fails here."""
    import yaml

    from sisteer.eval.jailbreak import COMPILED

    _NOT_WIRED_ARMS = ("alphakl_null", "alphakl_soft", "alphakl_fisher",
                       "diffmean", "bidpo_lora", "caa_static_r",
                       # cheap-estimator arms (IFEval-only; appended last
                       # at the stage2-evals merge): never jailbreak
                       # defenses
                       "caa", "actadd", "actadd_front",
                       # psr_xfer_* transferred-PSR arms (merged 2026-09-02):
                       # serving/eval-lane only, trainer refuses in-repo —
                       # never jailbreak defenses, in NO DEFENSES array
                       "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                       "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
                       # sdf-base-steering document-CE arms (2026-09-01,
                       # S_TWO_SI only — the bidpo_lora kind of exception;
                       # appended LAST at the sdfbase-mt merge, after
                       # psr_xfer_*): never single-SI jailbreak defenses
                       "psr_ce_all", "lora_r4_ce",
                       # sdf-base-steering DERIVED arms (2026-09-06,
                       # appended after the CE arms they decompose): the
                       # corpus fit MINUS its register-matched control fit.
                       # Serving-lane only, and not even trainable —
                       # methods.ARMS refuses both their build and their
                       # train — so they are in NO DEFENSES array, for the
                       # same reason psr_xfer_* are not.
                       "psr_ce_all_minus_ctl_distilled",
                       "lora_r4_ce_minus_ctl",
                       # ... and the UNDISTILLED paired write (kind
                       # psr_paired_write; relu-differenced-native-qwen,
                       # 2026-09-17), appended LAST: the same DERIVED,
                       # serving-lane-only arm (methods.DERIVED_ARMS names
                       # all three), so in NO DEFENSES array either
                       "psr_ce_all_minus_ctl")
    # did/adasteer were wired 2026-08-19 (appending to the defense axis, no
    # pre-existing defense's positional index shifted); the cheap-estimator arms now
    # sit at the registry tail after them, psr_xfer_* after those, the
    # document-CE arms next and the DERIVED arms last
    assert COMPILED[-20:] == ("alphakl_null", "alphakl_soft",
                              "alphakl_fisher", "diffmean", "bidpo_lora",
                              "caa_static_r", "did", "adasteer",
                              "caa", "actadd", "actadd_front",
                              # psr_xfer_* appended (merged 2026-09-02)
                              "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                              "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
                              # the document-CE arms after them (sdfbase-mt merge)
                              "psr_ce_all", "lora_r4_ce",
                              # and the DERIVED arms after those (2026-09-06)
                              "psr_ce_all_minus_ctl_distilled",
                              "lora_r4_ce_minus_ctl",
                              # then the UNDISTILLED paired write (2026-09-17)
                              "psr_ce_all_minus_ctl")
    # The registry tail in RELATIVE order (each arm strictly before the
    # next) — a literal tail slice would move every time an arm is appended,
    # and appending is the rule; the derived arm is asserted separately as
    # the FINAL registered arm and the final unwired one (2026-09-10).
    _TAIL_ORDER = ("alphakl_null", "alphakl_soft",
                   "alphakl_fisher", "diffmean", "bidpo_lora",
                   "caa_static_r", "did", "adasteer",
                   "caa", "actadd", "actadd_front",
                   # psr_xfer_* appended after the cheap-estimator arms (merged 2026-09-02)
                   "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                   "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
                   # the sdf-base DERIVED LoRA arm (2026-09-10), then the
                   # UNDISTILLED paired write appended LAST (2026-09-17)
                   "lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl")
    _idx = [COMPILED.index(a) for a in _TAIL_ORDER]
    assert _idx == sorted(_idx) and len(set(_idx)) == len(_idx)
    assert COMPILED[-2:] == ("lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl") == _NOT_WIRED_ARMS[-2:]
    # the rename alias sits exactly one past its sibling, nowhere else
    assert COMPILED[COMPILED.index("lora_r1_all") + 1] == "lora_r4"
    _UNWIRED_ARMS = set(_NOT_WIRED_ARMS) | {"lora_r4"}
    wired = [a for a in COMPILED if a not in _UNWIRED_ARMS]

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "stage2_jailbreak.yaml").read_text())
    expected = ["none", "prompted", *wired]
    # the two-SI bring-in APPENDS after the divergence arms,
    # and the did/adasteer arms wired 2026-08-19 append after those
    assert expected[-7:] == ["vector_rkl", "vector_js",
                             "antidote", "alphasteer", "lofit",
                             "did", "adasteer"]
    assert cfg["defenses"] == expected
    assert _sbatch_defenses("jailbreak_attacks.sbatch") == expected
    assert _sbatch_defenses("jailbreak.sbatch") == expected

    # Parametric on the LIVE defense count — never a stale literal, so adding
    # arms can't quietly desync the array math from the config.
    ndef, nshards, n_attacks = len(expected), 4, 4

    # attacks.sbatch: idx = attack*(NDEF*NSHARDS) + defense*NSHARDS + shard.
    seen = set()
    for idx in range(n_attacks * ndef * nshards):
        per_atk = ndef * nshards
        a, rem = idx // per_atk, idx % per_atk
        d, s = rem // nshards, rem % nshards
        assert 0 <= a < n_attacks and 0 <= d < ndef and 0 <= s < nshards
        seen.add((a, d, s))
    assert len(seen) == n_attacks * ndef * nshards  # bijection over the product

    # jailbreak.sbatch (BoN): idx = defense*NSHARDS + shard.
    seen_bon = set()
    for idx in range(ndef * nshards):
        seen_bon.add((idx // nshards, idx % nshards))
    assert len(seen_bon) == ndef * nshards
