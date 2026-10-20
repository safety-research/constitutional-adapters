"""The VENDOR PRESET is the multi-turn TARGET's default decode (the user's instruction, 2026-09-16).

Before this change `attacks.target_decode` defaulted to `generation_config` (the request carried NO
sampling key; the served model's generation config governed; 2026-09-11) and `--multiturn-target-decode
auto` was vendor only with a served target AND --model-key. Now:

  * `attacks.target_decode` defaults to `vendor_preset` (`target_decode.DEFAULT_CONFIG_MODE`); an
    explicit `generation_config` / `greedy` / temperature keeps its old behaviour exactly.
  * `auto` follows the config-level mode: a served run with no flag and no config key decodes at the
    served model's vendor preset from configs/sdfbase_sampled_decode.yaml -- --model-key naming the row
    when given, else the config's model / model_revision pins resolving it (`sampled_decode.
    preset_key_for_model`, the static families' own rule). An HF target or a served model with no row
    REFUSES (a temperature nobody vouched for is never a default).
  * An explicit `--multiturn-target-decode vendor` / `greedy` behaves exactly as before (the in-flight
    waves pass `vendor` explicitly), and the served_target.json decode block a default run writes is
    byte-identical to the explicit switch's, so the two are one sampler and resume each other.
  * attacker_engine.json's `target_decode` records the RESOLVED sampler (mode vendor_preset, the preset,
    the model key and its source) -- before, it recorded the config-level mode even on a vendor run.
  * The substantive point: gemma / llama ship a generation_config.json equal to their vendor row, so for
    them only the wire changes; Qwen3.5-4B ships none and its row carries presence_penalty 1.5, which no
    vLLM server default can carry (contracts.SERVER_DEFAULTABLE_SAMPLING_KEYS), so under the old default
    Qwen multi-turn decoded at presence_penalty 0.0 and under the vendor preset it decodes at 1.5.
    Nemotron's YAML row is a DECLARED departure from its shipped file (top_p 1.0 vs 0.95), pinned here
    so the split is a recorded fact, not a surprise.

  * THE LEG-LEVEL, SELF-RETIRING OPT-OUT (`attacks.target_decode_pin` / `--target-decode-pin-*`): a leg pairing
    against an anchor already drawn under the old decode declares the anchor and the decode it must RECORD; the
    pin applies while the anchor's own records agree, retires (the default applies, recorded `applied: false`)
    once the anchor is re-drawn under the vendor preset, refuses when the anchor's decode cannot be established
    or when the leg states its decode twice, and is recorded in both manifests beside what was actually sent.

Synthetic worlds only (the served-driver tests' fake target); the model pins are the repo's own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jailbreak_run as jr  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.eval import sampled_decode as sdd  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402
from sisteer.eval.attacks import target_decode as td  # noqa: E402

from test_jailbreak_run_served import _FakeServed, _goals  # noqa: E402
from test_wave_target_decode import _StubChat  # noqa: E402

YAML = yaml.safe_load(jr.VENDOR_PRESET_FILE.read_text(encoding="utf-8"))["presets"]
QWEN = ("Qwen/Qwen3.5-4B", YAML["qwen35_4b"]["revision"])
LLAMA = ("NousResearch/Meta-Llama-3.1-8B-Instruct", YAML["llama8b"]["revision"])
QWEN_PRESET = dict(YAML["qwen35_4b"]["preset"])
LLAMA_PRESET = dict(YAML["llama8b"]["preset"])


def _ns(*argv):
    p = argparse.ArgumentParser()
    jr._add_target_engine_args(p)
    return p.parse_args(list(argv))


def _cfg(model, revision, **attacks):
    return {"model": model, "model_revision": revision, "attacks": {"target_max_new_tokens": 4096, **attacks}}


# --------------------------------------------------------------------------- #
# the config-level default                                                      #
# --------------------------------------------------------------------------- #


def test_the_config_level_default_is_the_vendor_preset_and_explicit_modes_keep_their_behaviour():
    ns = _ns()
    assert td.DEFAULT_CONFIG_MODE == td.VENDOR_PRESET == "vendor_preset"
    assert jr._target_decode_mode({}, ns) == "vendor_preset"
    assert jr._target_decode_mode({"attacks": {}}, ns) == "vendor_preset"
    # the explicit old default resolves to the old behaviour: temperature None, no sampling key
    gen = {"attacks": {"target_decode": "generation_config"}}
    assert jr._target_decode_mode(gen, ns) == "generation_config" and jr._target_temperature(gen, ns) is None
    assert jr._target_decode_record(gen, ns) == {"mode": "generation_config", "temperature": None, "do_sample": True,
                                                 "temperature_source": "server_generation_config"}
    assert jr._target_temperature({"attacks": {"target_decode": "greedy"}}, ns) == 0.0
    assert jr._target_temperature({}, _ns("--target-greedy")) == 0.0
    assert jr._target_temperature({}, _ns("--target-temperature", "0.9")) == 0.9
    # the vendor preset has NO wire-level temperature until the preset is resolved: a read here is a caller bug
    with pytest.raises(ValueError, match="vendor preset"):
        jr._target_temperature({}, ns)
    # the bare manifest record under the default names the mode and says the numbers are not resolved here
    assert jr._target_decode_record({}, ns) == {"mode": "vendor_preset", "temperature": None, "do_sample": True,
                                                "temperature_source": "vendor_preset", "resolved": False}
    with pytest.raises(SystemExit, match="vendor_preset"):
        jr._target_decode_mode({"attacks": {"target_decode": "vendor"}}, ns)


# --------------------------------------------------------------------------- #
# the resolver: how the two knobs meet                                         #
# --------------------------------------------------------------------------- #


def test_served_default_resolves_the_vendor_preset_from_the_config_pins_without_model_key():
    r = jr._resolve_multiturn_target_decode(_cfg(*LLAMA), SimpleNamespace(attack="crescendo"), served_target=True)
    assert r["cli"] == "auto" and r["config_mode"] == "vendor_preset" and r["mode"] == "vendor"
    assert r["model_key"] == "llama8b" and r["model_key_source"] == "config_pins"
    assert r["vendor"]["decode"] == LLAMA_PRESET and r["vendor"]["seed_base"] == 20260909
    assert r["temperature"] == 0.6
    rec = r["record"]
    assert rec["mode"] == "vendor_preset" and rec["temperature_source"] == "vendor_preset" and rec["do_sample"] is True
    assert rec["preset"] == LLAMA_PRESET and rec["model_key"] == "llama8b" and rec["model_key_source"] == "config_pins"
    assert rec["multiturn_target_decode"] == "auto" and rec["multiturn_target_seed"] == "per_turn"
    assert rec["source_kind"] == "generation_config" and rec["seed_base"] == 20260909


def test_served_default_with_model_key_uses_it_and_qwen_gains_presence_penalty():
    r = jr._resolve_multiturn_target_decode(_cfg(*QWEN), SimpleNamespace(attack="pair", model_key="qwen35_4b"),
                                            served_target=True)
    assert r["model_key_source"] == "--model-key" and r["vendor"]["decode"] == QWEN_PRESET
    assert r["vendor"]["decode"]["presence_penalty"] == 1.5 and r["record"]["preset"]["presence_penalty"] == 1.5
    # the same row from the pins alone
    r2 = jr._resolve_multiturn_target_decode(_cfg(*QWEN), SimpleNamespace(attack="pair"), served_target=True)
    assert r2["model_key"] == "qwen35_4b" and r2["vendor"]["decode"] == QWEN_PRESET


def test_explicit_generation_config_in_the_config_is_honoured_under_auto_even_with_model_key():
    cfg = _cfg(*QWEN, target_decode="generation_config")
    r = jr._resolve_multiturn_target_decode(cfg, SimpleNamespace(attack="pair", model_key="qwen35_4b"), served_target=True)
    assert r["mode"] == "generation_config" and r["temperature"] is None and r["vendor"] is None
    assert r["record"] == {"mode": "generation_config", "temperature": None, "do_sample": True,
                           "temperature_source": "server_generation_config",
                           "multiturn_target_decode": "auto", "config_mode": "generation_config", "pin": None}
    # an explicit greedy in the config, or --target-greedy on the CLI, is greedy -- not the vendor preset
    r = jr._resolve_multiturn_target_decode(_cfg(*QWEN, target_decode="greedy"), SimpleNamespace(attack="pair"),
                                            served_target=True)
    assert r["mode"] == "greedy" and r["temperature"] == 0.0 and r["vendor"] is None
    r = jr._resolve_multiturn_target_decode(_cfg(*QWEN), _ns("--target-greedy", "--model-key", "qwen35_4b"),
                                            served_target=True)
    assert r["mode"] == "greedy" and r["temperature"] == 0.0 and r["vendor"] is None
    # an explicit temperature is explicit
    r = jr._resolve_multiturn_target_decode(_cfg(*QWEN), _ns("--target-temperature", "0.9"), served_target=True)
    assert r["mode"] == "explicit" and r["temperature"] == 0.9 and r["record"]["temperature_source"] == "request"


def test_the_explicit_cli_switch_wins_over_the_config_exactly_as_before():
    # --multiturn-target-decode vendor: the preset, --model-key required, whatever the config says
    r = jr._resolve_multiturn_target_decode(_cfg(*QWEN, target_decode="greedy"),
                                            _ns("--multiturn-target-decode", "vendor", "--model-key", "qwen35_4b"),
                                            served_target=True)
    assert r["mode"] == "vendor" and r["vendor"]["decode"] == QWEN_PRESET and r["model_key_source"] == "--model-key"
    assert r["record"]["multiturn_target_decode"] == "vendor" and r["record"]["config_mode"] == "greedy"
    with pytest.raises(SystemExit, match="needs --model-key"):
        jr._resolve_multiturn_target_decode(_cfg(*QWEN), _ns("--multiturn-target-decode", "vendor"), served_target=True)
    with pytest.raises(SystemExit, match="needs --target-engine"):
        jr._resolve_multiturn_target_decode(_cfg(*QWEN), _ns("--multiturn-target-decode", "vendor", "--model-key", "qwen35_4b"),
                                            served_target=False)
    # --multiturn-target-decode greedy: 0.0 whatever the config says
    r = jr._resolve_multiturn_target_decode(_cfg(*QWEN), _ns("--multiturn-target-decode", "greedy"), served_target=True)
    assert r["mode"] == "greedy" and r["temperature"] == 0.0 and r["vendor"] is None
    assert r["record"]["mode"] == "greedy" and r["record"]["multiturn_target_decode"] == "greedy"
    # --multiturn-target-seed none under the default: seed_base None in the block AND the record
    r = jr._resolve_multiturn_target_decode(_cfg(*QWEN), _ns("--multiturn-target-seed", "none"), served_target=True)
    assert r["vendor"]["seed_base"] is None and r["record"]["seed_base"] is None and r["record"]["multiturn_target_seed"] == "none"


def test_the_default_refuses_an_hf_target_and_a_served_model_with_no_row_naming_the_ways_out():
    with pytest.raises(SystemExit) as ei:
        jr._resolve_multiturn_target_decode(_cfg(*QWEN), SimpleNamespace(attack="pair"), served_target=False)
    msg = str(ei.value)
    assert "VENDOR PRESET" in msg and "--target-greedy" in msg and "generation_config" in msg and "HF" in msg
    with pytest.raises(SystemExit) as ei:
        jr._resolve_multiturn_target_decode({"model": "m", "attacks": {}}, SimpleNamespace(attack="pair"), served_target=True)
    msg = str(ei.value)
    assert "no row" in msg and "--model-key" in msg and "--target-greedy" in msg and "generation_config" in msg
    # the stated old default on the HF path still resolves as it always did
    r = jr._resolve_multiturn_target_decode({"model": "m", "attacks": {"target_decode": "generation_config"}},
                                            SimpleNamespace(attack="pair"), served_target=False)
    assert r["mode"] == "generation_config" and r["temperature"] is None


# --------------------------------------------------------------------------- #
# the wire: what the preset changes for Qwen and does not change for llama      #
# --------------------------------------------------------------------------- #


def test_qwen_presence_penalty_rides_only_under_the_preset_never_under_the_generation_config(tmp_path):
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", timing_log=tmp_path / "t.jsonl")
    # the OLD default: no sampling key at all -> presence_penalty is whatever the engine defaults to (0.0)
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8, do_sample=True, temperature=None)
    assert set(chat.payloads[0]) == {"model", "messages", "max_tokens"}
    assert t.last_decode["temperature_source"] == "server_generation_config"
    # the NEW default: exactly the preset's keys, presence_penalty 1.5 among them
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8, do_sample=True, temperature=0.7,
           decode=QWEN_PRESET, sampling_seed=7)
    body = chat.payloads[1]
    assert body["presence_penalty"] == 1.5 and body["temperature"] == 0.7 and body["top_p"] == 0.8
    assert body["top_k"] == 20 and body["min_p"] == 0.0 and body["repetition_penalty"] == 1.0 and body["seed"] == 7
    assert t.last_decode["temperature_source"] == "vendor_preset" and t.last_decode["preset"] == QWEN_PRESET
    # and the server can never default presence_penalty: the composed override omits it
    assert "presence_penalty" not in tc.SERVER_DEFAULTABLE_SAMPLING_KEYS
    override = json.loads(tc.server_default_generation_config(tc.SERVER_DEFAULT_SAMPLING_BY_MODEL["Qwen/Qwen3.5-4B"]))
    assert "presence_penalty" not in override
    assert override == {k: v for k, v in QWEN_PRESET.items() if k != "presence_penalty"}


def _cfg_for(key):
    return _cfg(YAML[key]["model"], YAML[key]["revision"])


def test_yaml_preset_equals_the_contracts_vendor_row_except_where_the_yaml_declares_a_departure():
    """The divergence this test first recorded (2026-09-16 morning) and the truth it now pins: gemma / llama / qwen:
    the YAML row IS contracts' vendor_preset row. nemotron: the YAML row runs top_p 1.0 against the shipped 0.95 that
    contracts (and the IFEval instrument) carry -- a DECLARED departure (`deliberate_departure`). The multi-turn
    DEFAULT resolution now equals contracts' vendor_preset numbers for ALL FOUR models (nemotron pinned to the shipped
    file), while the explicit `--multiturn-target-decode vendor` switch still sends the YAML row as it stands -- so the
    two nemotron behaviours are both pinned here, by number."""
    for key in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"):
        row = tc.sampling_for(key, tc.SAMPLING_VENDOR_PRESET)
        row_keys = {k: getattr(row, k) for k in sdd.SAMPLER_KEYS if getattr(row, k, None) is not None}
        preset = dict(YAML[key]["preset"])
        departures = set((YAML[key].get("deliberate_departure") or {}))
        agree = {k: v for k, v in preset.items() if k not in departures}
        assert agree == {k: v for k, v in row_keys.items() if k not in departures}, key
        for k in departures:
            assert preset[k] != row_keys[k] and YAML[key]["shipped_generation_config"][k] == row_keys[k], key
        # THE DEFAULT PATH sends contracts' vendor_preset numbers for every paper model (nemotron: the shipped 0.95)
        default = jr._resolve_multiturn_target_decode(_cfg_for(key), SimpleNamespace(attack="pair"), served_target=True)
        assert default["vendor"]["decode"] == row_keys, (key, default["vendor"]["decode"], row_keys)
        # THE EXPLICIT SWITCH sends the YAML row as it stands (nemotron: 1.0), as every explicit-vendor store did
        explicit = jr._resolve_multiturn_target_decode(_cfg_for(key), _ns("--multiturn-target-decode", "vendor", "--model-key", key),
                                                       served_target=True)
        assert explicit["vendor"]["decode"] == preset, key
        assert explicit["record"]["preset_basis"] == jr.PRESET_BASIS_YAML_ROW and explicit["record"]["shipped_pin"] is None, key
        assert (default["record"]["preset_basis"] == jr.PRESET_BASIS_SHIPPED_PIN) is bool(departures), key
    assert set(YAML["nemotron3nano4b"]["deliberate_departure"]) == {"top_p"}
    assert YAML["nemotron3nano4b"]["preset"]["top_p"] == 1.0 and tc.sampling_for("nemotron3nano4b", tc.SAMPLING_VENDOR_PRESET).top_p == 0.95
    assert set(YAML["gemma4e4b"].get("deliberate_departure") or {}) == set() == set(YAML["llama8b"].get("deliberate_departure") or {})
    assert QWEN_PRESET["presence_penalty"] == 1.5 and "presence_penalty" not in LLAMA_PRESET


def test_nemotron_default_is_pinned_to_the_shipped_file_and_the_explicit_switch_is_not():
    """Six nemotron 4,096 multi-turn roots record top_p 0.95, so the DEFAULT sends the shipped file's values; the
    explicit switch keeps the YAML row's 1.0. Known-bad: the earlier code sent 1.0 on the default path too (this test
    fails there), and a decorative record would not name which of the two ran."""
    ns = SimpleNamespace(attack="crescendo")
    r = jr._resolve_multiturn_target_decode(_cfg_for("nemotron3nano4b"), ns, served_target=True)
    assert r["vendor"]["decode"] == {"temperature": 1.0, "top_p": 0.95} and r["temperature"] == 1.0
    rec = r["record"]
    assert rec["preset"] == {"temperature": 1.0, "top_p": 0.95} and rec["preset_basis"] == "shipped_generation_config_pin"
    assert rec["yaml_preset"] == {"temperature": 1.0, "top_p": 1.0}
    assert rec["departure_from_yaml"] == {"top_p": {"yaml": 1.0, "sent": 0.95}}
    assert rec["shipped_pin"]["state"] == "present" and rec["shipped_pin"]["sampler_keys"] == {"temperature": 1.0, "top_p": 0.95}
    assert rec["shipped_pin"]["revision"] == YAML["nemotron3nano4b"]["revision"] and rec["shipped_pin"]["sha256"]
    assert rec["source_kind"] == "generation_config" and "2026-09-16" in rec["source"] and "0.95" in rec["source"]
    # the explicit switch: the YAML row as it stands, recorded as such
    e = jr._resolve_multiturn_target_decode(_cfg_for("nemotron3nano4b"),
                                            _ns("--multiturn-target-decode", "vendor", "--model-key", "nemotron3nano4b"), served_target=True)
    assert e["vendor"]["decode"] == {"temperature": 1.0, "top_p": 1.0} and e["record"]["preset_basis"] == "yaml_vendor_row"
    assert e["record"]["departure_from_yaml"] == {} and e["record"]["source_kind"] == "vendor_recommendation"
    # the two are DIFFERENT samplers and must never resume each other: their served_target.json decode identities differ
    def _block(res):
        v = res["vendor"]
        return {"do_sample": True, **v["decode"], "cap": 4096, "arm": "vendor_recommended", "seed_base": v["seed_base"],
                "trial": 0, "seed_rule": "sha256(...)", "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}
    assert st._decode_identity(_block(r)) != st._decode_identity(_block(e))
    assert st._decode_identity(_block(r))["top_p"] == 0.95 and st._decode_identity(_block(e))["top_p"] == 1.0
    # the other three paper models are unmoved by the pin (their rows equal their files, or nothing ships)
    for key, state in (("gemma4e4b", "present"), ("llama8b", "present"), ("qwen35_4b", "ships_none")):
        rr = jr._resolve_multiturn_target_decode(_cfg_for(key), ns, served_target=True)
        assert rr["vendor"]["decode"] == dict(YAML[key]["preset"]) and rr["record"]["preset_basis"] == "yaml_vendor_row", key
        assert rr["record"]["departure_from_yaml"] == {} and rr["record"]["shipped_pin"]["state"] == state, key


def test_the_shipped_file_pin_refuses_inconsistent_or_unestablished_data(tmp_path):
    """Known-bad pins: a table that disagrees with the YAML row's own copy of the file, an undeclared difference, a
    revision the table pins differently, a model it does not name, an unread (`unknown`) state. None may decide a leg."""
    nemo = YAML["nemotron3nano4b"]
    cfg = _cfg_for("nemotron3nano4b")
    block = jr.vendor_preset_for("nemotron3nano4b", cfg)
    assert block["shipped_generation_config"] == {"temperature": 1.0, "top_p": 0.95} and set(block["deliberate_departure"]) == {"top_p"}

    def pin_table(**row):
        p = tmp_path / f"pin_{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps({"models": {nemo["model"]: {"revision": nemo["revision"], "state": "present",
                                                           "sampler_keys": {"temperature": 1.0, "top_p": 0.95}, "sha256": "ab" * 32,
                                                           **row}}}))
        return p
    ok = jr._pin_vendor_preset_to_shipped_file(dict(block), cfg, attack="pair", path=pin_table())
    assert ok["decode"] == {"temperature": 1.0, "top_p": 0.95} and ok["preset_basis"] == "shipped_generation_config_pin"
    with pytest.raises(SystemExit, match="two records of one file disagree"):
        jr._pin_vendor_preset_to_shipped_file(dict(block), cfg, attack="pair", path=pin_table(sampler_keys={"temperature": 1.0, "top_p": 0.9}))
    with pytest.raises(SystemExit, match="without declaring the departure"):
        jr._pin_vendor_preset_to_shipped_file({**block, "shipped_generation_config": None, "deliberate_departure": None}, cfg,
                                              attack="pair", path=pin_table())
    with pytest.raises(SystemExit, match="pins .* @"):
        jr._pin_vendor_preset_to_shipped_file(dict(block), cfg, attack="pair", path=pin_table(revision="0" * 40))
    with pytest.raises(SystemExit, match="not read from the snapshot"):
        jr._pin_vendor_preset_to_shipped_file(dict(block), cfg, attack="pair", path=pin_table(state="unknown"))
    with pytest.raises(SystemExit, match="pins no shipped generation_config"):
        jr._pin_vendor_preset_to_shipped_file(dict(block), {**cfg, "model": "no/such-model"}, attack="pair", path=pin_table())
    # a model that ships none rides its row whole
    none = tmp_path / "none.json"
    none.write_text(json.dumps({"models": {nemo["model"]: {"revision": nemo["revision"], "state": "ships_none"}}}))
    rides = jr._pin_vendor_preset_to_shipped_file(dict(block), cfg, attack="pair", path=none)
    assert rides["decode"] == {"temperature": 1.0, "top_p": 1.0} and rides["preset_basis"] == "yaml_vendor_row"


# --------------------------------------------------------------------------- #
# end to end through cmd_generate on the served path                            #
# --------------------------------------------------------------------------- #


def _wire(monkeypatch, tmp_path):
    import sisteer.models as models_mod

    monkeypatch.setattr(models_mod, "load_model_and_tokenizer",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HF load on the served path")))
    fake = _FakeServed()
    monkeypatch.setattr(st, "build_served_target", lambda **kw: fake)
    monkeypatch.setattr(jr, "_valid_defense_ids", lambda cfg: {"none", "si_constitution__psr_kl_all"})
    monkeypatch.setattr(jr, "_eval_goals", lambda cfg, *, confirmatory: _goals(3))
    monkeypatch.setattr(jr, "_build_attacker_seams",
                        lambda cfg, args, log_dir=None: ((lambda p: "5"), (lambda lock: None), (lambda chat: "x")))
    got = {}

    def fake_run(attack, defense, goals_arg, **kw):
        got.update(kw, attack=attack)
        return {"n_goals_run": len(goals_arg), "n_goals_total": len(goals_arg), "n_target_forwards_total": len(goals_arg)}   # fw
    monkeypatch.setattr(jr, "_run_gated_attack", fake_run)
    args = SimpleNamespace(attack="crescendo", defense="si_constitution__psr_kl_all", shard=0, nshards=1,
                           xstest=False, confirmatory=True, trial=0, private_root=str(tmp_path / "state"),
                           target_engine="served", target_proxy_url="http://127.0.0.1:7/v1",
                           target_routing_table=str(tmp_path / "rt.json"), positions="response", scale=1.0,
                           goal_ids_file=None, expect_pair_n_streams=None,
                           attacker_engine="vllm", attacker_endpoint="http://127.0.0.1:8000")
    return fake, got, args


def test_cmd_generate_served_default_threads_the_qwen_preset_and_records_it_everywhere(monkeypatch, tmp_path):
    fake, got, args = _wire(monkeypatch, tmp_path)
    cfg = {"model": QWEN[0], "model_revision": QWEN[1], "dtype": "bfloat16", "private_run": "jb",
           "attacker": {"model": "qwen"}, "attacks": {"forward_cap": 150, "target_max_new_tokens": 4096}}
    jr.cmd_generate(args, cfg)
    # the attack receives the preset (presence_penalty 1.5 included), the seed base and the preset's temperature
    assert got["target_decode"] == QWEN_PRESET and got["seed_base"] == 20260909 and got["trial"] == 0
    assert got["target_temperature"] == 0.7
    out = Path(got["out_dir"])
    eng = json.loads((out / "attacker_engine.json").read_text())["target_decode"]
    assert eng["mode"] == "vendor_preset" and eng["temperature_source"] == "vendor_preset"
    assert eng["preset"] == QWEN_PRESET and eng["preset"]["presence_penalty"] == 1.5
    assert eng["model_key"] == "qwen35_4b" and eng["model_key_source"] == "config_pins"
    assert eng["multiturn_target_decode"] == "auto" and eng["config_mode"] == "vendor_preset"
    doc = json.loads((out / "served_target.json").read_text())
    blk = doc["decode"]                       # write_target_manifest's `decode` block (the sampler the store was drawn under)
    # the served_target.json block is the explicit switch's, byte for byte (one sampler, one identity):
    # exactly the keys an explicit --multiturn-target-decode vendor run wrote before 2026-09-16
    assert blk == {"do_sample": True, **QWEN_PRESET, "cap": 4096, "arm": "vendor_recommended", "seed_base": 20260909,
                   "trial": 0, "seed_rule": "sha256(f'{seed_base}|{trial}|{attack}|{goal_id}|{turn}')[:4]",
                   "source": YAML["qwen35_4b"]["source"], "source_kind": "model_card",
                   "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}
    assert doc["target_decode"] is None       # the target's CLI extras block: none given
    assert fake.allow_unseeded_decode is False


def test_cmd_generate_served_explicit_generation_config_still_sends_no_sampling_key(monkeypatch, tmp_path):
    fake, got, args = _wire(monkeypatch, tmp_path)
    cfg = {"model": QWEN[0], "model_revision": QWEN[1], "dtype": "bfloat16", "private_run": "jb",
           "attacker": {"model": "qwen"},
           "attacks": {"forward_cap": 150, "target_max_new_tokens": 4096, "target_decode": "generation_config"}}
    jr.cmd_generate(args, cfg)
    assert got["target_temperature"] is None and "target_decode" not in got and "seed_base" not in got
    out = Path(got["out_dir"])
    eng = json.loads((out / "attacker_engine.json").read_text())["target_decode"]
    assert eng["mode"] == "generation_config" and eng["temperature"] is None
    assert eng["temperature_source"] == "server_generation_config" and eng["config_mode"] == "generation_config"
    doc = json.loads((out / "served_target.json").read_text())
    assert doc.get("target_decode") is None and doc.get("decode") is None


def test_cmd_generate_served_default_refuses_a_model_with_no_row_before_writing(monkeypatch, tmp_path):
    fake, got, args = _wire(monkeypatch, tmp_path)
    cfg = {"model": "m", "dtype": "bfloat16", "private_run": "jb", "attacker": {"model": "qwen"},
           "attacks": {"forward_cap": 150, "target_max_new_tokens": 4096}}
    with pytest.raises(SystemExit, match="no row"):
        jr.cmd_generate(args, cfg)
    assert not got
    assert not list((tmp_path / "state").rglob("attacker_engine.json"))


def test_the_canary_records_the_greedy_it_states():
    assert jr._decode_record_for_temperature(0.0) == {"mode": "greedy", "temperature": 0.0, "do_sample": False,
                                                       "temperature_source": "request"}
    src = (REPO / "scripts" / "jailbreak_run.py").read_text()
    canary = src[src.index("def cmd_canary("):src.index("def cmd_canary_calibrate(")]
    assert "target_decode_record=_decode_record_for_temperature(0.0)" in canary and "target_temperature=0.0" in canary


# --------------------------------------------------------------------------- #
# the leg-level, SELF-RETIRING opt-out (attacks.target_decode_pin / --target-decode-pin-*)          #
#                                                                                                   #
# Built on the REAL record shapes of the corpus (the 2026-09-16 sweeps: 5,117 shard stores), laid   #
# out as the stores are: <root>/<view>/<attack>/<cell>/shard<NN>/{served_target.json,                #
# attacker_engine.json}. Two defects found by peers on real data are pinned here: the first chain    #
# was INERT (no rung matched the anchors' records); its successor was CREDULOUS (it re-sent the      #
# anchors' bytes over a server whose fill of the omitted keys has since moved).                       #
# --------------------------------------------------------------------------- #

#: the four canonical Qwen 4,096 anchors' records, field for field: attacker_engine.json has NO `mode` key; the
#: served manifest's last_request_decode says a bare temperature 1.0 was sent (the note names the era's engine fill)
QWEN_ANCHOR_AE = {"temperature": 1.0, "do_sample": True, "note": "qwen35_4b_ships_no_generation_config.json;vllm_defaults_govern_top_p_top_k"}
QWEN_ANCHOR_SENT = {"do_sample": True, "temperature": 1.0, "seed": None, "top_p": None, "top_k": None, "repetition_penalty": None,
                    "note": "qwen35_4b_ships_no_generation_config.json;vllm_defaults_govern_top_p_top_k"}
#: gemma's 4,096 anchors: the shipped file's values sent explicitly (temperature 1.0, top_p 0.95, top_k 64)
GEMMA_ANCHOR_AE = {"temperature": 1.0, "do_sample": True, "top_p": 0.95, "top_k": 64, "note": "shipped_generation_config.json@ee0ef602"}
GEMMA_ANCHOR_SENT = {"do_sample": True, "temperature": 1.0, "seed": None, "top_p": 0.95, "top_k": 64, "repetition_penalty": None,
                     "note": "shipped_generation_config.json@ee0ef602"}
#: nemotron's mt4karr roots: the shipped file (1.0 / 0.95), no top_k
NEMO_ANCHOR_AE = {"temperature": 1.0, "do_sample": True, "top_p": 0.95, "note": "shipped_generation_config.json@dfaf35de;no_top_k_shipped"}
NEMO_ANCHOR_SENT = {"do_sample": True, "temperature": 1.0, "seed": None, "top_p": 0.95, "top_k": None, "repetition_penalty": None}
#: llama's mt4kasr roots: a `mode`-keyed record (since 0cd39504) at the shipped file (0.6 / 0.9)
LLAMA_ANCHOR_AE = {"mode": "explicit", "temperature": 0.6, "do_sample": True, "temperature_source": "request", "top_p": 0.9}
LLAMA_ANCHOR_SENT = {"do_sample": True, "temperature": 0.6, "temperature_source": "request", "seed": None, "top_p": 0.9,
                     "top_k": None, "repetition_penalty": None}
GREEDY_AE = {"temperature": 0.0, "do_sample": False}
GREEDY_SENT = {"do_sample": False, "temperature": 0.0, "seed": None, "top_p": None, "top_k": None, "repetition_penalty": None}
VENDOR_BLOCK = {"do_sample": True, **QWEN_PRESET, "cap": 4096, "arm": "vendor_recommended", "seed_base": 20260909,
                "trial": 0, "seed_rule": "sha256(...)", "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}
MODELS = {"qwen": ("Qwen/Qwen3.5-4B", YAML["qwen35_4b"]["revision"]), "gemma": ("google/gemma-4-E4B-it", YAML["gemma4e4b"]["revision"]),
          "nemo": (YAML["nemotron3nano4b"]["model"], YAML["nemotron3nano4b"]["revision"]),
          "llama": ("NousResearch/Meta-Llama-3.1-8B-Instruct", YAML["llama8b"]["revision"])}
SHAPES = {  # kind -> (model, attacker_engine target_decode, served last_request_decode, served decode block)
    "qwen_anchor": ("qwen", QWEN_ANCHOR_AE, QWEN_ANCHOR_SENT, None),
    "gemma_anchor": ("gemma", GEMMA_ANCHOR_AE, GEMMA_ANCHOR_SENT, None),
    "nemo_anchor": ("nemo", NEMO_ANCHOR_AE, NEMO_ANCHOR_SENT, None),
    "llama_anchor": ("llama", LLAMA_ANCHOR_AE, LLAMA_ANCHOR_SENT, None),
    "greedy": ("qwen", GREEDY_AE, GREEDY_SENT, None),
    "vendor": ("qwen", QWEN_ANCHOR_AE, None, VENDOR_BLOCK),        # a pre-2026-09-16 explicit-vendor store: STALE engine record
    "recordless": ("qwen", None, None, None),                       # a pre-2026-09-10 store: no decode record anywhere
}


def _store(root: Path, kind: str, *, view="sdfbase_full22_x", attack="pair", cell="none", n=3, **over) -> Path:
    """A synthetic anchor STORE in the corpus layout, every shard shaped like the real records of ``kind``."""
    model_key, ae_td, sent, blk = SHAPES[kind]
    model = MODELS[model_key][0]
    store = root / view / attack / cell
    for i in range(n):
        d = store / f"shard{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        ae = {"attack": attack, "target_engine": "served"}
        if ae_td is not None:
            ae["target_decode"] = dict(ae_td)
        st = {"target_engine": "served", "base_model": model, "manifest_stage": "final", "target_decode": None}
        if sent is not None:
            st["last_request_decode"] = dict(sent)
        if blk is not None:
            st["decode"] = dict(blk)
        st.update(over)
        (d / "attacker_engine.json").write_text(json.dumps(ae))
        (d / "served_target.json").write_text(json.dumps(st))
    return root


def _cfg_m(model_key, **attacks):
    m, rev = MODELS[model_key]
    return {"model": m, "model_revision": rev, "attacks": {"target_max_new_tokens": 4096, **attacks}}


def _pin_cfg(model_key, anchor, expect, **attacks):
    return _cfg_m(model_key, target_decode_pin={"anchor": str(anchor), "expect": expect,
                                                 "reason": "user ruling: Phase F pairs against anchors drawn under the old decode",
                                                 "owner": "Scale"}, **attacks)


NS = SimpleNamespace(attack="pair", shard=0, defense="cellX")


def test_a_leg_with_no_declaration_resolves_to_the_vendor_preset():
    """The new default (known-bad input: the same leg WITH a pin resolves differently, below)."""
    r = jr._resolve_multiturn_target_decode(_cfg_m("qwen"), NS, served_target=True)
    assert r["mode"] == "vendor" and r["record"]["mode"] == "vendor_preset" and r["record"]["pin"] is None
    assert jr._target_decode_pin(_cfg_m("qwen"), NS) is None


def test_the_server_defaultable_key_set_is_contracts():
    """The fill rule names the keys a vLLM server can fill; contracts is the authority for that set."""
    assert tuple(jr._SERVER_DEFAULTABLE_KEYS) == tuple(tc.SERVER_DEFAULTABLE_SAMPLING_KEYS)
    assert "presence_penalty" not in jr._SERVER_DEFAULTABLE_KEYS


def test_the_qwen_anchor_shape_is_established_from_values_but_the_pin_refuses_the_unrecorded_fill(tmp_path):
    """The two peer findings on the SAME shape. (1) The first chain refused it ('records no target decode': no `mode`
    key, no `decode` block); the values establish `explicit` temperature 1.0 uniquely -- for Qwen (no shipped file,
    vendor row 0.7 / 0.8 / 20 / 1.5) only `explicit` reproduces the sent bytes. (2) Its successor then APPLIED and
    re-sent temperature 1.0 alone -- but the anchors OMITTED top_p / top_k / min_p / repetition_penalty, the standup
    of the time filled them (vLLM's defaults, at a code sha before e631261d), the store records only that they were
    omitted, and today's standup fills them from --override-generation-config. Known-bad: the previous commit applied
    this pin; it must REFUSE, naming the omitted keys and the hand-passed spec."""
    root = _store(tmp_path / "transcripts_anchor4096_qwen35_4b", "qwen_anchor", view="sdfbase_full22_qwen35_4b_it", n=4)
    shard = jr._anchor_recorded_decode(root / "sdfbase_full22_qwen35_4b_it" / "pair" / "none" / "shard00", attack="pair",
                                       model=MODELS["qwen"][0])
    assert shard["kind"] == "values" and shard["mode"] == "explicit" and shard["ambiguous"] is False
    assert shard["temperature"] == 1.0 and shard["extras"] is None and shard["candidates_matching"] == ["explicit"]
    assert shard["read_from"].startswith("served_target.json last_request_decode") and "agrees" in shard["read_from"]
    # the fill verdict: every server-defaultable key it omitted, none pinned by a shipped file (Qwen ships none)
    assert shard["fill_reproducible"] is False and shard["omitted_unpinned"] == ["top_p", "top_k", "min_p", "repetition_penalty"]
    assert shard["shipped_pin_state"] == "ships_none"
    store = jr._anchor_store_decode(root, attack="pair", cell="none", model=MODELS["qwen"][0])
    assert store["n_shards"] == 4 and store["mode"] == "explicit" and store["fill_reproducible"] is False
    with pytest.raises(SystemExit) as ei:
        jr._target_decode_pin(_pin_cfg("qwen", root, "explicit"), NS)
    msg = str(ei.value)
    assert "OMITTED ['top_p', 'top_k', 'min_p', 'repetition_penalty']" in msg and "does not record" in msg
    assert "REQUEST, not the engine values" in msg and "STATE every key by hand" in msg and "e631261d" in msg
    # a pin naming another expectation still retires (the store establishes explicit, not generation_config) -- the fill
    # check comes after the mode check, so a retiring pin never claims a fill
    p2 = jr._target_decode_pin(_pin_cfg("qwen", root, "generation_config"), NS)
    assert p2["applied"] is False and p2["retired"] is True
    # the CLI form refuses the same way
    ns = _ns("--target-decode-pin-anchor", str(root), "--target-decode-pin-expect", "explicit", "--target-decode-pin-reason", "r")
    ns.attack, ns.shard, ns.defense = "pair", 0, "cellX"
    with pytest.raises(SystemExit, match="OMITTED"):
        jr._target_decode_pin(_cfg_m("qwen"), ns)


def test_a_pin_applies_only_when_every_omitted_key_is_pinned_by_the_shipped_file(tmp_path):
    """The one shape a pin may reproduce byte-for-byte AND value-for-value: an anchor that sent, or whose shipped
    file states, every server-defaultable key. No real store has it (none states min_p / repetition_penalty), so this
    is the mechanism's contract for stores that will record their fill; the near-miss (one key omitted and unpinned)
    refuses. Known-bad: the previous commit applied both."""
    hot = {"mode": "explicit", "temperature": 0.9, "do_sample": True, "top_p": 0.5, "top_k": 7, "repetition_penalty": 1.05}
    sent = {"do_sample": True, "temperature": 0.9, "seed": None, "top_p": 0.5, "top_k": 7, "repetition_penalty": 1.05}
    SHAPES["llama_full"] = ("llama", hot, sent, None)
    root = _store(tmp_path / "transcripts_full_llama8b", "llama_full", n=2)
    shard = jr._anchor_recorded_decode(root / "sdfbase_full22_x" / "pair" / "none" / "shard00", attack="pair", model=MODELS["llama"][0])
    assert shard["mode"] == "explicit" and shard["omitted_unpinned"] == ["min_p"]      # min_p: not sendable as an extra, no file states it
    with pytest.raises(SystemExit, match=r"OMITTED \['min_p'\]"):
        jr._target_decode_pin(_pin_cfg("llama", root, "explicit"), NS)
    # a synthetic model whose shipped pin states min_p too would reproduce -- exercise the rule at the function level
    ctx = {"shipped": {"temperature": 0.6, "top_p": 0.9, "min_p": 0.0, "repetition_penalty": 1.0}, "shipped_state": "present",
           "yaml": None, "default": None}
    est = jr._establish_from_wire({"temperature": 0.9, "top_p": 0.5, "top_k": 7, "repetition_penalty": 1.05}, ctx)
    assert est["mode"] == "explicit" and est["fill_reproducible"] is True and est["omitted_unpinned"] == []
    assert est["extras"] == {"top_p": 0.5, "top_k": 7, "repetition_penalty": 1.05}
    # greedy sends only temperature 0.0: its omitted keys are as unrecorded as anyone's
    g = _store(tmp_path / "transcripts_cap4096greedy_qwen35_4b", "greedy", cell="constv2-mix-full__lora_r4_ce_minus_ctl", n=2)
    with pytest.raises(SystemExit, match="OMITTED"):
        jr._target_decode_pin(_cfg_m("qwen", target_decode_pin={"anchor": str(g), "expect": "greedy", "reason": "r",
                                                                 "anchor_cell": "constv2-mix-full__lora_r4_ce_minus_ctl"}), NS)


def test_values_name_a_mode_only_where_the_candidates_differ_gemma_llama_nemotron_at_their_file_are_ambiguous(tmp_path):
    """The peer's rule: gemma's and llama's vendor rows ARE their shipped files, so an anchor decoded at the file is
    reproduced by generation_config, by an explicit temperature with the file's extras AND by the vendor preset --
    no value read separates them, and the pin REFUSES as AMBIGUOUS naming the coinciding candidates. Nemotron at its
    file coincides with generation_config and with the shipped-pinned DEFAULT (the yaml row's 1.0 is out). Known-bad:
    a resolver that picks `explicit` here (the wire mode) would reintroduce the inference."""
    for kind, model_key, must_name in (("gemma_anchor", "gemma", "vendor_preset (the yaml row == the DEFAULT"),
                                       ("llama_anchor", "llama", "vendor_preset (the yaml row == the DEFAULT"),
                                       ("nemo_anchor", "nemo", "vendor_preset (the DEFAULT: the yaml row pinned")):
        root = _store(tmp_path / f"transcripts_{kind}", kind, n=2)
        rec = jr._anchor_recorded_decode(root / "sdfbase_full22_x" / "pair" / "none" / "shard00", attack="pair", model=MODELS[model_key][0])
        assert rec["kind"] == "values" and rec["ambiguous"] is True and rec["mode"] is None, kind
        assert "generation_config" in rec["candidates_matching"] and "explicit" in rec["candidates_matching"], kind
        assert any(m.startswith(must_name) for m in rec["candidates_matching"]), (kind, rec["candidates_matching"])
        if kind == "nemo_anchor":   # the yaml row (1.0 / 1.0) is NOT a match: nemotron discriminates against it
            assert not any("explicit --multiturn-target-decode vendor" in m for m in rec["candidates_matching"])
        with pytest.raises(SystemExit) as ei:
            jr._target_decode_pin(_pin_cfg(model_key, root, "explicit"), NS)
        msg = str(ei.value)
        assert "AMBIGUOUS" in msg and "refused rather than guessed" in msg and "drop the pin" in msg, kind
        with pytest.raises(SystemExit, match="AMBIGUOUS"):
            jr._target_decode_pin(_pin_cfg(model_key, root, "generation_config"), NS)
    rec = jr._anchor_recorded_decode(_store(tmp_path / "t_llama2", "llama_anchor") / "sdfbase_full22_x" / "pair" / "none" / "shard00",
                                     attack="pair", model=MODELS["llama"][0])
    assert rec["declared_mode"] == "explicit" and rec["ambiguous"] is True


def test_the_pin_retires_when_the_anchor_store_is_a_vendor_draw_even_with_the_declaration_present(tmp_path):
    """Known-bad input: the pin is declared, but the store records a vendor draw (served_target.json decode block)
    -- the STALE engine manifest still says temperature 1.0 with no mode, exactly the shape a value-only reader
    would pin. The served block wins; the pin retires; the default applies and both say so."""
    root = _store(tmp_path / "transcripts_vendor_qwen35_4b", "vendor", n=3)
    rec = jr._anchor_recorded_decode(root / "sdfbase_full22_x" / "pair" / "none" / "shard01", attack="pair", model=MODELS["qwen"][0])
    assert rec["kind"] == "vendor" and rec["read_from"] == "served_target.json decode" and rec["block"]["presence_penalty"] == 1.5
    cfg = _pin_cfg("qwen", root, "explicit")
    pin = jr._target_decode_pin(cfg, NS)
    assert pin["applied"] is False and pin["retired"] is True and pin["anchor_decode"]["mode"] == "vendor_preset"
    r = jr._resolve_multiturn_target_decode(cfg, NS, served_target=True)
    assert r["mode"] == "vendor" and r["vendor"]["decode"] == QWEN_PRESET and r["record"]["pin"]["applied"] is False
    assert jr._target_decode_mode(cfg, NS) == "vendor_preset"


def test_the_pin_refuses_rather_than_guessing(tmp_path):
    root = _store(tmp_path / "transcripts_anchor4096_qwen35_4b", "qwen_anchor", view="sdfbase_full22_qwen35_4b_it")
    # a single SHARD directory is a sample, not a store
    with pytest.raises(SystemExit, match="ONE SHARD"):
        jr._target_decode_pin(_pin_cfg("qwen", root / "sdfbase_full22_qwen35_4b_it" / "pair" / "none" / "shard00", "explicit"), NS)
    # the shards must AGREE
    (root / "sdfbase_full22_qwen35_4b_it" / "pair" / "none" / "shard02" / "served_target.json").write_text(json.dumps(
        {"target_engine": "served", "base_model": MODELS["qwen"][0], "manifest_stage": "final",
         "last_request_decode": {**QWEN_ANCHOR_SENT, "temperature": 0.7}}))
    (root / "sdfbase_full22_qwen35_4b_it" / "pair" / "none" / "shard02" / "attacker_engine.json").write_text(json.dumps(
        {"target_decode": {**QWEN_ANCHOR_AE, "temperature": 0.7}}))
    with pytest.raises(SystemExit, match="DISAGREE"):
        jr._target_decode_pin(_pin_cfg("qwen", root, "explicit"), NS)
    # configured vs sent disagreeing inside ONE shard
    bad = _store(tmp_path / "transcripts_bad", "qwen_anchor", n=1)
    (bad / "sdfbase_full22_x" / "pair" / "none" / "shard00" / "attacker_engine.json").write_text(json.dumps({"target_decode": {**QWEN_ANCHOR_AE, "temperature": 0.5}}))
    with pytest.raises(SystemExit, match="records disagree"):
        jr._target_decode_pin(_pin_cfg("qwen", bad, "explicit"), NS)
    # a record-less store (pre-2026-09-10) cannot be established
    with pytest.raises(SystemExit, match="records no target decode"):
        jr._target_decode_pin(_pin_cfg("qwen", _store(tmp_path / "transcripts_rest", "recordless"), "greedy"), NS)
    with pytest.raises(SystemExit, match="not a directory"):
        jr._target_decode_pin(_pin_cfg("qwen", tmp_path / "nowhere", "explicit"), NS)
    # the leg's attack / cell must exist under the root; several views need naming
    with pytest.raises(SystemExit, match="holds no <view>/crescendo/none"):
        jr._target_decode_pin(_pin_cfg("qwen", root, "explicit"), SimpleNamespace(attack="crescendo", shard=0, defense="x"))
    two = _store(tmp_path / "transcripts_two", "qwen_anchor", view="v1", n=1)
    _store(tmp_path / "transcripts_two", "qwen_anchor", view="v2", n=1)
    with pytest.raises(SystemExit, match="several views"):
        jr._target_decode_pin(_pin_cfg("qwen", two, "explicit"), NS)
    # a named view / the leg's own config stem selects the store (and then the fill refusal fires, after the view resolved)
    with pytest.raises(SystemExit, match="OMITTED"):
        jr._target_decode_pin(_cfg_m("qwen", target_decode_pin={"anchor": str(two), "expect": "explicit", "reason": "r", "anchor_view": "v2"}), NS)
    with pytest.raises(SystemExit, match="OMITTED"):
        jr._target_decode_pin(_pin_cfg("qwen", two, "explicit"), SimpleNamespace(attack="pair", shard=0, defense="x", config="/some/where/v1.yaml"))
    # never the vendor preset; never beside another statement; never twice; never half a CLI declaration
    with pytest.raises(SystemExit, match="never name the vendor preset"):
        jr._target_decode_pin(_pin_cfg("qwen", root, "vendor_preset"), NS)
    for extra in ({"target_decode": "generation_config"}, {"target_temperature": 0.5}):
        with pytest.raises(SystemExit, match="ONCE"):
            jr._target_decode_pin(_pin_cfg("qwen", root, "explicit", **extra), NS)
    for argv in (["--target-greedy"], ["--multiturn-target-decode", "vendor", "--model-key", "qwen35_4b"], ["--target-top-p", "0.9"]):
        ns = _ns(*argv); ns.attack, ns.shard, ns.defense = "pair", 0, "cellX"
        with pytest.raises(SystemExit, match="ONCE"):
            jr._target_decode_pin(_pin_cfg("qwen", root, "explicit"), ns)
    ns_both = _ns("--target-decode-pin-anchor", str(root), "--target-decode-pin-expect", "explicit", "--target-decode-pin-reason", "r")
    ns_both.attack, ns_both.shard, ns_both.defense = "pair", 0, "cellX"
    with pytest.raises(SystemExit, match="declare it once"):
        jr._target_decode_pin(_pin_cfg("qwen", root, "explicit"), ns_both)
    ns_half = _ns("--target-decode-pin-anchor", str(root)); ns_half.attack = "pair"
    with pytest.raises(SystemExit, match="go together"):
        jr._target_decode_pin(_cfg_m("qwen"), ns_half)
    with pytest.raises(SystemExit, match="mapping with anchor, expect, reason"):
        jr._target_decode_pin({**_cfg_m("qwen"), "attacks": {"target_decode_pin": {"anchor": "x"}}}, NS)


def test_the_recorded_resolution_matches_what_is_actually_sent(tmp_path):
    """The record is checked against the WIRE, not the declaration: each resolution is fed into a real
    ServedChatTarget and the body + its per-request decode block are compared with the record."""
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", timing_log=tmp_path / "t.jsonl")
    # (1) a retired pin (vendor anchor): the record says vendor_preset and the body carries exactly the preset
    v = _store(tmp_path / "transcripts_vendor", "vendor")
    r = jr._resolve_multiturn_target_decode(_pin_cfg("qwen", v, "explicit"), NS, served_target=True)
    assert r["record"]["mode"] == "vendor_preset" and r["record"]["pin"]["applied"] is False
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=8, do_sample=True, temperature=r["temperature"],
           decode=r["vendor"]["decode"], sampling_seed=3)
    body = chat.payloads[-1]
    assert {k: body[k] for k in QWEN_PRESET} == QWEN_PRESET == r["record"]["preset"] and body["seed"] == 3
    assert t.last_decode["temperature_source"] == r["record"]["temperature_source"] == "vendor_preset"
    # (2) the hand-passed spec the fill refusal names: explicit temperature + extras, sent as given -- top_k 0 (the
    # wheel's disable value, contracts.PROTOCOL_TOP_K) is accepted by the driver's validator since 2026-09-16
    stated = jr._resolve_multiturn_target_decode(_cfg_m("qwen"), _ns("--target-temperature", "1.0", "--target-top-p", "1.0", "--target-top-k", "0"),
                                                 served_target=True)
    assert stated["mode"] == "explicit" and stated["record"]["top_p"] == 1.0 and stated["record"]["top_k"] == 0
    t2 = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", timing_log=tmp_path / "t2.jsonl",
                             decode={"top_p": 1.0, "top_k": 0})
    t2.chat([{"role": "user", "content": "hi"}], max_new_tokens=8, do_sample=True, temperature=1.0)
    body = chat.payloads[-1]
    assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 1.0, 0) and "repetition_penalty" not in body
    assert t2.last_decode["temperature_source"] == stated["record"]["temperature_source"] == "request"
    # (3) no declaration: the same vendor wire, record pin None
    r = jr._resolve_multiturn_target_decode(_cfg_m("qwen"), NS, served_target=True)
    assert r["record"]["pin"] is None and r["record"]["preset"] == QWEN_PRESET


def test_cmd_generate_served_refuses_a_pin_with_unrecorded_fill_before_writing_and_retires_a_vendor_anchor(monkeypatch, tmp_path):
    fake, got, args = _wire(monkeypatch, tmp_path)
    args.attack, args.shard = "crescendo", 0
    root = _store(tmp_path / "transcripts_anchor4096_qwen35_4b", "qwen_anchor", view="sdfbase_full22_qwen35_4b_it", attack="crescendo")
    cfg = {**_pin_cfg("qwen", root, "explicit"), "dtype": "bfloat16", "private_run": "jb", "attacker": {"model": "qwen"}}
    cfg["attacks"]["forward_cap"] = 150
    with pytest.raises(SystemExit, match="OMITTED"):
        jr.cmd_generate(args, cfg)
    assert not got and not list((tmp_path / "state").rglob("attacker_engine.json"))       # refused before any write
    # a vendor anchor: the pin retires, the preset rides, both manifests say so
    fake2, got2, args2 = _wire(monkeypatch, tmp_path / "again")
    args2.attack, args2.shard = "crescendo", 0
    v = _store(tmp_path / "transcripts_vendor", "vendor", attack="crescendo")
    cfg2 = {**_pin_cfg("qwen", v, "explicit"), "dtype": "bfloat16", "private_run": "jb", "attacker": {"model": "qwen"}}
    cfg2["attacks"]["forward_cap"] = 150
    jr.cmd_generate(args2, cfg2)
    assert got2["target_decode"] == QWEN_PRESET and got2["target_temperature"] == 0.7
    eng2 = json.loads((Path(got2["out_dir"]) / "attacker_engine.json").read_text())["target_decode"]
    assert eng2["mode"] == "vendor_preset" and eng2["pin"]["applied"] is False and eng2["pin"]["retired"] is True
    assert eng2["pin"]["anchor_decode"]["kind"] == "vendor" and eng2["pin"]["anchor_decode"]["n_shards"] == 3
    doc2 = json.loads((Path(got2["out_dir"]) / "served_target.json").read_text())
    assert doc2["decode"]["multiturn_target_decode"] == "vendor" and doc2["target_decode_resolution"] == eng2


def test_every_rung_of_the_resolver_can_be_true_of_existing_data_or_is_labelled_future_only():
    """The failure mode (2026-09-16): a chain whose rungs were each reasonable and jointly matched nothing that
    exists. The real shapes above are the existing data; each must be established by a rung, and the one rung
    that is future-only (target_decode_resolution) must say so in the docstring."""
    doc = jr._anchor_recorded_decode.__doc__
    assert "FUTURE data only" in doc and "last_request_decode" in doc and "mode-less shape" in doc
    src = (REPO / "scripts" / "sdfbase" / "full22" / "anchor_decode_sweep.py").read_text()
    assert "PROTECTED_QWEN_ANCHOR_ROOTS" in src and "transcripts_anchor4096_qwen35_4b" in src and "FILL UNRECORDED" in src


# --------------------------------------------------------------------------- #
# follow-on 2: the two top_k validators were INVERTED                          #
# --------------------------------------------------------------------------- #


def test_top_k_disable_values_agree_between_contracts_and_the_served_target_validator():
    """contracts.PROTOCOL_TOP_K is 0 (the wheel's disable value; -1 its tolerated legacy alias) and the served
    target's normalize_decode REFUSED 0 while accepting -1 -- a peer ran both through the validator. Known-bad:
    on the previous commit normalize_decode({"top_k": 0}) raised. Reconciled ON THE ENGINE'S CODE (read from the
    installed wheels at both served versions, 0.25.1+cu129 and 0.26.0+cu129; citations beside
    served_target.TOP_K_DISABLE_VALUES): the OpenAI protocol passes a provided top_k through untouched,
    SamplingParams._verify_args refuses only < -1 and never rewrites -1, and v1's gpu_input_batch maps every top_k
    outside (0, vocab_size) -- 0 and -1 alike -- to vocab_size, which apply_top_k_top_p reads as no top-k. So both
    disable values pass, sent as given; anything below -1 still refuses; contracts names the driver's validator."""
    assert tc.PROTOCOL_TOP_K == 0 and tc.PROTOCOL_TOP_K in st.TOP_K_DISABLE_VALUES and -1 in st.TOP_K_DISABLE_VALUES
    assert st.normalize_decode({"top_k": tc.PROTOCOL_TOP_K}) == {"top_k": 0}
    assert st.normalize_decode({"top_k": -1}) == {"top_k": -1} and st.normalize_decode({"top_k": 5}) == {"top_k": 5}
    with pytest.raises(st.ServedTargetError, match="disable value"):
        st.normalize_decode({"top_k": -2})
    # the CLI extras path is the same validator: --target-top-k 0 reaches the served target
    ns = _ns("--target-temperature", "1.0", "--target-top-k", "0")
    assert jr._target_decode({}, ns) == {"top_k": 0} and st.normalize_decode(jr._target_decode({}, ns)) == {"top_k": 0}
    # each side names the other
    import inspect
    assert "PROTOCOL_TOP_K" in inspect.getsource(st.normalize_decode) or "PROTOCOL_TOP_K" in inspect.getsource(st)
    csrc = (REPO / "src" / "sisteer" / "capeval" / "tinycap" / "contracts.py").read_text()
    assert "normalize_decode" in csrc and "TOP_K_DISABLE_VALUES" in csrc
