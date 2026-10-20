"""The Qwen seed-replicate battery's multi-turn TARGET — its seed base = the literal training seed.

The vendor preset is already the multi-turn target's default (`target_decode.DEFAULT_CONFIG_MODE`); its per-(goal, turn)
seed `multiturn_seed(seed_base, trial, attack, goal_id, turn)` took its base from the preset FILE's campaign value
(20260909) with no CLI knob. Two ADDITIVE flags, both defaulting to today's behaviour byte for byte:

  * `--multiturn-target-seed-base <s>`: the base becomes the served view's LITERAL training seed (1 / 2 / 3); recorded as
    `seed_base` in served_target.json's decode block (a TARGET_DECODE_IDENTITY_KEYS member -> a seed-1 store and a
    campaign store never resume each other) and in attacker_engine.json target_decode (+ `seed_base_source`);
    REFUSED with `--multiturn-target-seed none` and on a non-vendor decode (a base with nowhere to go).
  * `--multiturn-target-policy-ref <ref>`: ASSERT the registry row (must equal the derived
    `qwen35_4b_it|multi_turn|off|vendor_preset@v1`); wrong ref / unversioned ref REFUSE.
  * EITHER flag puts the run on the REGISTRY PATH: the row `<model_id>|multi_turn|off|vendor_preset@v1` is
    derived from --model-key and resolved by exact by_ref (not superseded, all six keys sent) BEFORE and INSTEAD OF the
    preset file -- `vendor_preset_for` is never called (a monkeypatched sentinel proves it), the row's numbers ride even
    when the yaml drifts, its identity is recorded (`decode_policy`), and absent / superseded rows REFUSE with the seed
    base ALONE. Its `4096 per reply` is checked against the cap the replies are DISPATCHED with
    (`_multiturn_target_max_new_tokens(cfg, attack, args)`), both mismatch directions + CLI precedence.
    A seeded (per_turn) registry run must state its base; `none` admits an unseeded one.
  * target_attacks.sbatch forwards F22_MT_TARGET_SEED_BASE / F22_MT_TARGET_POLICY_REF only when set.

Pinned here against the FIXTURE policy file of tests/test_bon_vendor_preset.py: the resolver; cmd_generate end to end
(both manifests, the identity); the WIRE — every target request carries the six keys (presence 1.5), `max_tokens 4096`
and `seed = multiturn_seed(s, trial, attack, goal_id, turn)` (the literal s), an unseeded preset REFUSES; the ATTACKER's
request body is untouched (its own algorithm parameters); the sbatch render; and the request-evidence seam
(off by default, rows byte-identical). Synthetic worlds only; the model pins are the repo's own."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.eval.attacks import multiturn
from sisteer.eval.attacks import pyrit_adapters as pa
from sisteer.eval.attacks import served_target as st
from sisteer.eval.attacks import target_decode as td
from sisteer.eval.attacks import vllm_attacker as va
from sisteer.registry import decode_policy as dp

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import jailbreak_run as jr  # noqa: E402
from test_bon_vendor_preset import BON_REF, MT_REF, VENDOR_SIX, policy_file_with_vendor_rows  # noqa: E402
from test_multiturn_vendor_default import QWEN, QWEN_PRESET, YAML, _cfg, _wire  # noqa: E402
from test_wave_target_decode import _Defense, _StubChat  # noqa: E402

CAMPAIGN_BASE = 20260909
SEED_RULE = "sha256(f'{seed_base}|{trial}|{attack}|{goal_id}|{turn}')[:4]"
#: the EXACT key set of a vendor multi-turn target request body
BODY_KEYS = frozenset({"model", "messages", "max_tokens", "temperature", "seed",
                       "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"})


def _ns(**over):
    base = dict(attack="crescendo", multiturn_target_decode="vendor", model_key="qwen35_4b",
                multiturn_target_seed="per_turn", trial=0)
    base.update(over)
    return SimpleNamespace(**base)


def _golden_block(seed_base):
    """The served_target.json decode block an explicit vendor run writes (test_multiturn_vendor_default's golden), at a base."""
    return {"do_sample": True, **QWEN_PRESET, "cap": 4096, "arm": "vendor_recommended", "seed_base": seed_base,
            "trial": 0, "seed_rule": SEED_RULE, "source": YAML["qwen35_4b"]["source"], "source_kind": "model_card",
            "multiturn_target_decode": "vendor", "multiturn_target_seed": "per_turn"}


def _registry_block(seed_base):
    """The served_target.json decode block a REGISTRY-path run writes: the yaml block's keys and numbers, the provenance
    naming the row (source = the exact policy_id; source_kind decode_policy_registry) -- neither is an identity key."""
    return {**_golden_block(seed_base), "source": MT_REF, "source_kind": "decode_policy_registry"}


def _never_yaml(*a, **k):
    """The sentinel the registry path must never reach: vendor_preset_for (the preset yaml resolver)."""
    raise AssertionError("vendor_preset_for (the preset yaml) must not be consulted on the registry path")


def _assert_registry_provenance(r, *, ref_source, cap=4096):
    """The provenance every registry-path resolution records: the row (exact ref, not superseded), how the ref was named,
    the dispatched cap the row's rule was checked against, and source / source_kind / preset_basis naming the registry."""
    pol = r["record"]["decode_policy"]
    assert pol["policy_id"] == MT_REF and pol["superseded_by_policy_id"] is None
    assert pol["policy_ref_source"] == ref_source and pol["target_cap"] == cap == pol["cap_named_by_rule"]
    assert r["record"]["source"] == MT_REF == r["vendor"]["source"]
    assert r["record"]["source_kind"] == jr.SOURCE_KIND_REGISTRY_ROW == r["vendor"]["source_kind"]
    assert r["record"]["preset_basis"] == jr.PRESET_BASIS_REGISTRY_ROW == r["vendor"]["preset_basis"]
    for k in ("yaml_preset", "shipped_pin", "departure_from_yaml"):          # the yaml basis's keys are not invented here
        assert k not in r["record"] and k not in r["vendor"]


# --------------------------------------------------------------------------- #
# the resolver                                                                  #
# --------------------------------------------------------------------------- #


def test_the_resolver_takes_the_literal_base_and_records_nothing_new_when_omitted(monkeypatch, tmp_path):
    cfg = _cfg(*QWEN)
    assert QWEN_PRESET == VENDOR_SIX                                       # the file's row IS the six-key vendor row
    default = jr._resolve_multiturn_target_decode(cfg, _ns(), served_target=True)
    bare = jr._resolve_multiturn_target_decode(cfg, SimpleNamespace(attack="crescendo", multiturn_target_decode="vendor",
                                                                    model_key="qwen35_4b"), served_target=True)
    assert default["vendor"]["seed_base"] == CAMPAIGN_BASE and default["record"]["seed_base"] == CAMPAIGN_BASE
    for key in ("seed_base_source", "multiturn_target_seed_base", "decode_policy"):
        assert key not in default["record"] and key not in default["vendor"]
    assert default["record"]["source_kind"] == "model_card" and default["record"]["preset_basis"] == jr.PRESET_BASIS_YAML_ROW
    # a namespace WITHOUT the new attributes and one with them at None: byte-identical records
    assert json.dumps(default["record"], sort_keys=True) == json.dumps(bare["record"], sort_keys=True)
    # the LITERAL base puts the run on the REGISTRY PATH: the row is derived from --model-key and resolved
    # by exact ref; the yaml resolver is NEVER reached (sentinel); decode_policy is recorded
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="rows.json"))
    monkeypatch.setattr(jr, "vendor_preset_for", _never_yaml)
    for s in (1, 2, 3):
        r = jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_seed_base=s), served_target=True)
        assert r["vendor"]["seed_base"] == s and r["vendor"]["decode"] == VENDOR_SIX and r["temperature"] == 0.7
        rec = r["record"]
        assert rec["seed_base"] == s and rec["seed_base_source"] == "--multiturn-target-seed-base" and rec["multiturn_target_seed_base"] == s
        assert rec["preset"] == VENDOR_SIX and rec["preset"]["presence_penalty"] == 1.5 and rec["mode"] == "vendor_preset"
        assert rec["multiturn_target_seed"] == "per_turn"
        _assert_registry_provenance(r, ref_source="derived_from_model_key")
        # the recorded strings, pinned once by value (the other tests read the constants)
        assert rec["source_kind"] == "decode_policy_registry" and rec["preset_basis"] == "decode_policy_registry_row"
        # each basis carries its own keys: every default-record key but the yaml basis's three is present, and everything
        # but the base and the provenance equals the default's
        yaml_only = {"yaml_preset", "shipped_pin", "departure_from_yaml"}
        assert set(default["record"]) - yaml_only <= set(rec)
        moved = {"seed_base", "source", "source_kind", "preset_basis"} | yaml_only
        assert {k: v for k, v in rec.items() if k in default["record"] and k not in moved} == \
               {k: v for k, v in default["record"].items() if k not in moved}
    # under `auto` (the CLI default, the config's vendor default) the base rides the same way -- still the registry row
    r = jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_decode="auto", multiturn_target_seed_base=2), served_target=True)
    assert r["cli"] == "auto" and r["mode"] == "vendor" and r["vendor"]["seed_base"] == 2 and r["record"]["seed_base"] == 2
    _assert_registry_provenance(r, ref_source="derived_from_model_key")
    # KNOWN-BAD: a base with nowhere to go (pure argument conflicts -- refused before any export is read)
    with pytest.raises(SystemExit, match="unseeded draw has no base to take"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_seed="none", multiturn_target_seed_base=1), served_target=True)
    with pytest.raises(SystemExit, match="non-negative int"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_seed_base=-1), served_target=True)
    with pytest.raises(SystemExit, match="needs the vendor preset"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_decode="greedy", multiturn_target_seed_base=1), served_target=True)
    gen = _cfg(*QWEN, target_decode="generation_config")
    with pytest.raises(SystemExit, match="needs the vendor preset"):
        jr._resolve_multiturn_target_decode(gen, _ns(multiturn_target_decode="auto", multiturn_target_seed_base=1), served_target=True)
    # the greedy / generation_config paths themselves are untouched
    assert jr._resolve_multiturn_target_decode(gen, _ns(multiturn_target_decode="auto"), served_target=True)["temperature"] is None
    assert jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_decode="greedy"), served_target=True)["temperature"] == 0.0
    # the parser: type int, default None, beside --multiturn-target-seed
    p = __import__("argparse").ArgumentParser()
    jr._add_target_engine_args(p)
    ns = p.parse_args(["--multiturn-target-decode", "vendor", "--multiturn-target-seed-base", "3", "--model-key", "qwen35_4b"])
    assert ns.multiturn_target_seed_base == 3 and ns.multiturn_target_policy_ref is None
    assert p.parse_args([]).multiturn_target_seed_base is None and p.parse_args([]).multiturn_target_policy_ref is None
    with pytest.raises(SystemExit):
        p.parse_args(["--multiturn-target-seed-base", "one"])


def test_the_seed_base_alone_resolves_the_registry_row_and_refuses_an_absent_or_superseded_one(monkeypatch, tmp_path):
    """`--multiturn-target-seed-base` ALONE -- no --multiturn-target-policy-ref -- is the registry path's
    opt-in. Before the fix it succeeded against an export lacking the row with seed_base 1 and NO decode_policy, sending
    the preset file's numbers. Now the row `qwen35_4b_it|multi_turn|off|vendor_preset@v1` is derived from --model-key and
    resolved by exact ref: an export lacking it, or carrying it superseded, REFUSES before a byte lands; the yaml resolver
    is never reached; a yaml whose numbers drift from the row changes nothing (the row is the only authority). The absent
    and superseded worlds are SYNTHETIC fixtures (`drop=` / `superseded=`): the committed export carries the row (main since
    525607e3), so this test never reads the export's own state."""
    cfg = _cfg(*QWEN)
    monkeypatch.setattr(jr, "vendor_preset_for", _never_yaml)
    # an export with the row DROPPED, and one with it SUPERSEDED: refused by name, seed base alone
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, drop=[MT_REF], name="absent.json"))
    with pytest.raises(SystemExit, match=r"\[crescendo\] --multiturn-target-seed-base: .*has no row 'qwen35_4b_it\|multi_turn\|off\|vendor_preset@v1'"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_seed_base=1), served_target=True)
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, superseded=[MT_REF], name="sup.json"))
    with pytest.raises(SystemExit, match="--multiturn-target-seed-base: .*is SUPERSEDED by"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_seed_base=1), served_target=True)
    # a row that leaves presence_penalty to the server is not the vendor row
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(
        tmp_path, edit={MT_REF: {"params_sent": ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"]}}, name="torn.json"))
    with pytest.raises(SystemExit, match=r"does not SEND \['presence_penalty'\]"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_seed_base=1), served_target=True)
    # the yaml drifts from the row: the ROW rides (before: the ref path refused "two authorities"; the seed-base path
    # sent the yaml) -- the sentinel proves the yaml was never read
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, edit={MT_REF: {"temperature": 0.6}}, name="drift.json"))
    r = jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_seed_base=1), served_target=True)
    assert r["vendor"]["decode"] == {**VENDOR_SIX, "temperature": 0.6} == r["record"]["preset"] and r["temperature"] == 0.6
    assert r["record"]["decode_policy"]["policy_id"] == MT_REF and r["record"]["seed_base"] == 1
    # the typed ref is an ASSERTION on top of the derived row, not the switch: alone under per_turn the row resolves and
    # then the seeded run must state its base (the campaign base lives in the file this path does not read) ...
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="ok.json"))
    with pytest.raises(SystemExit, match=r"--multiturn-target-policy-ref resolves the registry row .*pass --multiturn-target-seed-base <s>"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF), served_target=True)
    # ... while under `none` it is admitted unseeded, on the row
    r = jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF, multiturn_target_seed="none"), served_target=True)
    assert r["vendor"]["seed_base"] is None and r["record"]["seed_base"] is None and "seed_base_source" not in r["record"]
    assert r["vendor"]["decode"] == VENDOR_SIX
    _assert_registry_provenance(r, ref_source="--multiturn-target-policy-ref")
    # the DEFAULT path (neither flag) still reads the yaml: the sentinel fires -- so the two paths are disjoint
    with pytest.raises(AssertionError, match="must not be consulted"):
        jr._resolve_multiturn_target_decode(cfg, _ns(), served_target=True)


# --------------------------------------------------------------------------- #
# cmd_generate end to end: both manifests, the identity                          #
# --------------------------------------------------------------------------- #


def _served_cfg():
    return {"model": QWEN[0], "model_revision": QWEN[1], "dtype": "bfloat16", "private_run": "jb",
            "attacker": {"model": "qwen"}, "attacks": {"forward_cap": 150, "target_max_new_tokens": 4096}}


def test_cmd_generate_records_the_base_in_both_manifests_and_the_identity_separates_seeds(monkeypatch, tmp_path):
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="rows.json"))
    fake, got, args = _wire(monkeypatch, tmp_path)
    args.multiturn_target_decode, args.model_key, args.multiturn_target_seed = "vendor", "qwen35_4b", "per_turn"
    args.multiturn_target_seed_base = 2
    with monkeypatch.context() as m:                                       # the seed base alone = the registry path
        m.setattr(jr, "vendor_preset_for", _never_yaml)
        jr.cmd_generate(args, _served_cfg())
    assert got["seed_base"] == 2 and got["target_decode"] == VENDOR_SIX and got["trial"] == 0 and got["target_temperature"] == 0.7
    out = Path(got["out_dir"])
    eng = json.loads((out / "attacker_engine.json").read_text())["target_decode"]
    assert eng["seed_base"] == 2 and eng["seed_base_source"] == "--multiturn-target-seed-base" and eng["multiturn_target_seed_base"] == 2
    assert eng["preset"] == VENDOR_SIX and eng["mode"] == "vendor_preset" and eng["multiturn_target_decode"] == "vendor"
    assert eng["decode_policy"]["policy_id"] == MT_REF and eng["source_kind"] == "decode_policy_registry"
    doc = json.loads((out / "served_target.json").read_text())
    blk = doc["decode"]
    assert blk == _registry_block(2)                                      # the explicit block at base 2, provenance = the row
    assert doc["target_decode_resolution"]["seed_base"] == 2 and doc["target_decode_resolution"]["seed_base_source"]
    assert doc["target_decode_resolution"]["decode_policy"]["policy_id"] == MT_REF
    assert fake.allow_unseeded_decode is False
    # the identity: three seeds are three stores; none resumes the campaign-base store. Provenance (source / source_kind)
    # is NOT an identity key: the registry block and the yaml block at one base and the same numbers are ONE sampler
    ids = {s: st._decode_identity(_golden_block(s)) for s in (1, 2, 3, CAMPAIGN_BASE)}
    assert len({json.dumps(v, sort_keys=True) for v in ids.values()}) == 4
    assert st._decode_identity(blk) == ids[2] != ids[CAMPAIGN_BASE]
    assert st._decode_identity(_registry_block(2)) == st._decode_identity(_golden_block(2))
    assert "seed_base" in st.TARGET_DECODE_IDENTITY_KEYS and "source" not in st.TARGET_DECODE_IDENTITY_KEYS
    # a store drawn under base 2 refuses a resume under base 1 (covered goals + the recorded block)
    (out / "progress.jsonl").write_text(json.dumps({"goal_id": "g0", "outcome": "success"}) + "\n")
    with pytest.raises(st.ServedTargetError):
        st.assert_target_decode_compatible(out, _golden_block(1))
    st.assert_target_decode_compatible(out, _golden_block(2))              # its own base resumes
    # the DEFAULT run (no flag) still writes the campaign block and no seed_base_source (byte-identical) -- through the yaml
    fake2, got2, args2 = _wire(monkeypatch, tmp_path / "d")
    args2.multiturn_target_decode, args2.model_key = "vendor", "qwen35_4b"
    jr.cmd_generate(args2, _served_cfg())
    assert got2["seed_base"] == CAMPAIGN_BASE
    eng2 = json.loads((Path(got2["out_dir"]) / "attacker_engine.json").read_text())["target_decode"]
    assert "seed_base_source" not in eng2 and "decode_policy" not in eng2 and eng2["seed_base"] == CAMPAIGN_BASE
    assert json.loads((Path(got2["out_dir"]) / "served_target.json").read_text())["decode"] == _golden_block(CAMPAIGN_BASE)


# --------------------------------------------------------------------------- #
# the wire: every target request                                               #
# --------------------------------------------------------------------------- #


def test_every_target_request_carries_the_six_keys_and_the_per_turn_seed_from_the_literal_s(tmp_path):
    from sisteer.eval.jailbreak_data import Goal
    goals = [Goal("g0", "benign objective", "illegal", "harmbench", "rev")]
    plan = '{"actors": [{"name": "a1", "queries": ["q0", "q1"]}]}'
    calls_by_base = {}
    for s in (1, 2, CAMPAIGN_BASE):
        d = _Defense()
        multiturn.run_actorattack(d, goals, attacker_fn=lambda chat: plan, scorer_fn=lambda p: "1",
                                  out_dir=tmp_path / f"aa_{s}", resume=False, actor_num=1, discovery_retries=0,
                                  max_new_tokens=4096, target_decode=VENDOR_SIX, seed_base=s, trial=0)
        assert d.calls and all(c["do_sample"] and c["temperature"] == 0.7 and c["decode"] == VENDOR_SIX for c in d.calls)
        assert all(c["decode"]["presence_penalty"] == 1.5 for c in d.calls)
        seeds = [c["seed"] for c in d.calls]
        assert seeds == [pa.multiturn_seed(s, 0, "actorattack", "g0", i) for i in range(len(seeds))]
        calls_by_base[s] = seeds
    assert len(calls_by_base[1]) == len(calls_by_base[2]) == len(calls_by_base[CAMPAIGN_BASE]) >= 3
    assert calls_by_base[1] != calls_by_base[2] != calls_by_base[CAMPAIGN_BASE]      # only s moves between replicates
    # the rule, spelled out: the static families' closure over the LITERAL s, the turn as the sample index
    assert pa.multiturn_seed(1, 0, "crescendo", "g7", 3) == int.from_bytes(hashlib.sha256(b"1|0|crescendo|g7|3").digest()[:4], "big")
    assert pa.multiturn_seed(1, 0, "crescendo", "g7", 3) != pa.multiturn_seed(2, 0, "crescendo", "g7", 3)
    with pytest.raises(ValueError, match="seed_base"):
        pa.multiturn_seed(None, 0, "pair", "g1", 0)                                  # seed missing -> fails
    # the REAL served target's body: EXACTLY the ten keys, the row's values, the cap, the seed
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x", serving_engine="lora_native")
    seed = pa.multiturn_seed(1, 0, "crescendo", "g7", 3)
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=4096, do_sample=True, temperature=0.7,
           sampling_seed=seed, decode=VENDOR_SIX)
    body = chat.payloads[0]
    assert frozenset(body) == BODY_KEYS, sorted(body)
    for k, v in VENDOR_SIX.items():
        assert body[k] == v, (k, body[k])
    assert body["seed"] == seed and body["max_tokens"] == 4096 and body["model"] == "petri__x"
    assert t.last_decode["seed"] == seed and t.last_decode["preset"] == VENDOR_SIX and t.last_decode["temperature_source"] == "vendor_preset"
    # KNOWN-BAD: a preset draw with no seed refuses (the safety net every multi-turn request passes through)
    with pytest.raises(st.ServedTargetError, match="sampling_seed"):
        t.chat([{"role": "user", "content": "hi"}], max_new_tokens=4096, do_sample=True, temperature=0.7, decode=VENDOR_SIX)
    assert len(chat.payloads) == 1


def test_the_pyrit_target_seeds_each_turn_from_the_literal_s(tmp_path):
    from pyrit.models import Message
    asyncio.run(pa.init_pyrit_memory_async(tmp_path / "memory.db"))

    async def one_turn(tgt):
        out = await tgt._send_prompt_to_target_async(
            normalized_conversation=[Message.from_prompt(prompt="hi", role="user")])
        return out[0].get_value()

    d = _Defense()
    gid = {"g": "g7"}
    tgt = pa.SteeredPromptTarget(defense=d, target_decode=VENDOR_SIX,
                                 seed_fn=lambda turn: pa.multiturn_seed(3, 0, "pair", gid["g"], turn))
    tgt.begin_goal()
    for turn in range(3):
        assert asyncio.run(one_turn(tgt)) == "ok"
        assert d.calls[-1]["seed"] == pa.multiturn_seed(3, 0, "pair", "g7", turn) and d.calls[-1]["decode"] == VENDOR_SIX
        assert d.calls[-1]["do_sample"] is True and d.calls[-1]["temperature"] == 0.7


# --------------------------------------------------------------------------- #
# the attacker's request body is untouched                                     #
# --------------------------------------------------------------------------- #


def test_attacker_request_bodies_are_untouched_by_the_target_flags():
    class _Client:
        def __init__(self):
            self.calls = []

        def chat(self, messages, *, temperature, top_p, max_new_tokens):
            self.calls.append({"messages": messages, "temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens})
            return "x"
    cfg = _served_cfg()
    for attack in ("crescendo", "pair", "actorattack"):
        samp = jr._attacker_sampling(cfg, attack)                          # reads the CONFIG only — no args, no target flag
        assert samp == jr._DEFAULT_ATTACKER_SAMPLING[attack] and set(samp) >= {"temperature", "top_p"}
        c = _Client()
        fn = va.make_attacker_fn_vllm(c, temperature=samp["temperature"], top_p=samp["top_p"],
                                      max_new_tokens=jr._attacker_max_new_tokens(cfg))
        fn([{"role": "user", "content": "plan"}])
        assert c.calls == [{"messages": [{"role": "user", "content": "plan"}], "temperature": samp["temperature"],
                            "top_p": samp["top_p"], "max_new_tokens": 1024}]
    # the attacker client's body: its own algorithm parameters — no seed, no presence_penalty, no vendor row
    src = (REPO / "src/sisteer/eval/attacks/vllm_attacker.py").read_text()
    body = src[src.index("        payload = {"):src.index("        self.last_meta = {}")]
    assert '"temperature": temperature' in body and '"top_p": top_p' in body and '"max_tokens": max_new_tokens' in body
    assert "seed" not in body and "presence_penalty" not in body and "vendor" not in body
    # and the seams builder never reads the two target flags
    jr_src = (REPO / "scripts/jailbreak_run.py").read_text()
    seams = jr_src[jr_src.index("def _build_attacker_seams("):jr_src.index("def _attacker_serving_provenance(")]
    assert "multiturn_target_seed_base" not in seams and "multiturn_target_policy_ref" not in seams
    assert "multiturn_target" not in seams


# --------------------------------------------------------------------------- #
# the registry pin                                                             #
# --------------------------------------------------------------------------- #


def test_the_registry_pin_holds_the_preset_to_the_multi_turn_row_and_refuses_every_known_bad(monkeypatch, tmp_path):
    cfg = _cfg(*QWEN)
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="mt.json"))
    monkeypatch.setattr(jr, "vendor_preset_for", _never_yaml)             # the registry path never reads the yaml
    r = jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF, multiturn_target_seed_base=1), served_target=True)
    assert r["vendor"]["decode"] == VENDOR_SIX and r["record"]["preset"] == VENDOR_SIX and r["record"]["seed_base"] == 1
    _assert_registry_provenance(r, ref_source="--multiturn-target-policy-ref")
    pol = r["record"]["decode_policy"]
    assert pol["max_new_tokens_rule"] == "4096 per reply" and pol["policy_file"].endswith("mt.json") and pol["mode"] == "vendor_preset"
    # KNOWN-BAD, each refused before anything is written
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, drop=[MT_REF], name="absent.json"))
    with pytest.raises(SystemExit, match="has no row"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF), served_target=True)
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, superseded=[MT_REF], name="sup.json"))
    with pytest.raises(SystemExit, match="SUPERSEDED"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF), served_target=True)
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="ok.json"))
    with pytest.raises(SystemExit, match="is not the served model's multi_turn vendor row"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=BON_REF), served_target=True)
    with pytest.raises(SystemExit, match="is not the served model's multi_turn vendor row"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref="qwen35_4b_it|multi_turn|off|vendor_preset"), served_target=True)
    # a yaml that drifts from the row is NOT a second authority: the row's numbers ride (and the yaml is never read)
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, edit={MT_REF: {"temperature": 0.6}}, name="drift.json"))
    drift = jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF, multiturn_target_seed_base=1), served_target=True)
    assert drift["vendor"]["decode"]["temperature"] == 0.6 == drift["temperature"] and QWEN_PRESET["temperature"] == 0.7
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="cap.json"))
    with pytest.raises(SystemExit, match="names 4096 per reply but this run's target cap is 2048"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF, target_max_new_tokens=2048), served_target=True)
    with pytest.raises(SystemExit, match="needs the vendor preset"):
        jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_decode="greedy", multiturn_target_policy_ref=MT_REF), served_target=True)
    # a row whose rule names no per-reply cap is not cross-checked (the instrument's cap governs)
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, edit={MT_REF: {"max_new_tokens_rule": "per instrument"}}, name="norule.json"))
    r2 = jr._resolve_multiturn_target_decode(cfg, _ns(multiturn_target_policy_ref=MT_REF, multiturn_target_seed_base=1,
                                                      target_max_new_tokens=2048), served_target=True)
    assert r2["record"]["decode_policy"]["cap_named_by_rule"] is None and r2["record"]["decode_policy"]["target_cap"] == 2048


def test_the_cap_check_uses_the_selector_dispatch_uses_in_both_directions_and_the_cli_wins(monkeypatch, tmp_path):
    """The row's `4096 per reply` is checked against the cap the replies are DISPATCHED with —
    `_multiturn_target_max_new_tokens(cfg, attack, args)`, the selector cmd_generate hands _run_gated_attack — never
    `_target_max_new_tokens(cfg, args)` alone, which ignores `attacks.multiturn_target_max_new_tokens`. Both mismatch
    directions are pinned against the OLD selector's answer, and the CLI flag's precedence over the config's map."""
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="cap.json"))
    ns = _ns(multiturn_target_policy_ref=MT_REF, multiturn_target_seed_base=1)
    # (a) target 4096 but the per-family map dispatches Crescendo at 512: the OLD selector certified 4096 (a pass);
    #     the dispatched cap is 512 -> REFUSED
    cfg_a = _cfg(*QWEN, multiturn_target_max_new_tokens={"crescendo": 512, "pair": 4096, "actorattack": 512})
    assert jr._target_max_new_tokens(cfg_a, ns) == 4096                          # the old check's answer (known-bad)
    assert jr._multiturn_target_max_new_tokens(cfg_a, "crescendo", ns) == 512    # what the replies are dispatched with
    with pytest.raises(SystemExit, match=r"\[crescendo\].*names 4096 per reply but this run's target cap is 512"):
        jr._resolve_multiturn_target_decode(cfg_a, ns, served_target=True)
    # ... while PAIR, whose map entry is 4096, is admitted under the SAME config (the family is what the check reads)
    pair = jr._resolve_multiturn_target_decode(cfg_a, _ns(attack="pair", multiturn_target_policy_ref=MT_REF,
                                                          multiturn_target_seed_base=1), served_target=True)
    _assert_registry_provenance(pair, ref_source="--multiturn-target-policy-ref")
    # (b) target 2048 but a flat multi-turn cap of 4096: the OLD selector refused a VALID run; the dispatched cap is
    #     4096 -> ADMITTED, and the recorded target_cap is the dispatched one
    cfg_b = _cfg(*QWEN, target_max_new_tokens=2048, multiturn_target_max_new_tokens=4096)
    assert jr._target_max_new_tokens(cfg_b, ns) == 2048                          # the old check's answer (known-bad)
    ok = jr._resolve_multiturn_target_decode(cfg_b, ns, served_target=True)
    _assert_registry_provenance(ok, ref_source="--multiturn-target-policy-ref")
    assert ok["record"]["decode_policy"]["target_cap"] == jr._multiturn_target_max_new_tokens(cfg_b, "crescendo", ns)
    # (c) CLI precedence: an explicit --target-max-new-tokens wins over the config's map in BOTH directions
    cli_ok = _ns(multiturn_target_policy_ref=MT_REF, multiturn_target_seed_base=1, target_max_new_tokens=4096)
    r = jr._resolve_multiturn_target_decode(cfg_a, cli_ok, served_target=True)   # map says 512, the flag says 4096
    _assert_registry_provenance(r, ref_source="--multiturn-target-policy-ref")
    assert r["record"]["decode_policy"]["target_cap"] == jr._multiturn_target_max_new_tokens(cfg_a, "crescendo", cli_ok)
    cli_bad = _ns(multiturn_target_policy_ref=MT_REF, multiturn_target_seed_base=1, target_max_new_tokens=2048)
    with pytest.raises(SystemExit, match="names 4096 per reply but this run's target cap is 2048"):
        jr._resolve_multiturn_target_decode(cfg_b, cli_bad, served_target=True)  # flat cap says 4096, the flag says 2048
    # a per-family map that names no cap for the running family refuses by name (never another family's cap)
    with pytest.raises(SystemExit, match="names no cap for 'crescendo'"):
        jr._resolve_multiturn_target_decode(_cfg(*QWEN, multiturn_target_max_new_tokens={"pair": 4096}), ns, served_target=True)


def test_cmd_generate_records_the_registry_pin_in_both_manifests(monkeypatch, tmp_path):
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="e2e.json"))
    fake, got, args = _wire(monkeypatch, tmp_path)
    args.multiturn_target_decode, args.model_key = "vendor", "qwen35_4b"
    args.multiturn_target_seed_base, args.multiturn_target_policy_ref = 3, MT_REF
    jr.cmd_generate(args, _served_cfg())
    assert got["seed_base"] == 3 and got["target_decode"] == VENDOR_SIX
    out = Path(got["out_dir"])
    eng = json.loads((out / "attacker_engine.json").read_text())["target_decode"]
    assert eng["decode_policy"]["policy_id"] == MT_REF and eng["seed_base"] == 3
    doc = json.loads((out / "served_target.json").read_text())
    assert doc["decode"] == _registry_block(3)                              # the block's identity is the numbers + the base
    assert st._decode_identity(doc["decode"]) == st._decode_identity(_golden_block(3))
    assert doc["target_decode_resolution"]["decode_policy"]["policy_id"] == MT_REF
    assert got["target_max_new_tokens"] == 4096 == eng["decode_policy"]["target_cap"] == doc["decode"]["cap"]


def test_cmd_generate_records_one_cap_in_the_dispatch_the_registry_check_and_the_served_manifest(monkeypatch, tmp_path):
    """`served_target.json.decode.cap` (an identity key) was computed WITHOUT `args`, so a config map of 512
    under an explicit `--target-max-new-tokens 4096` dispatched 4096 and recorded `decode_policy.target_cap` 4096 while the
    served manifest said 512 -- a permanently mislabeled store. The three now read ONE selector with `args`: the dispatch
    kwarg, the registry check's `target_cap` and the manifest's `cap` all equal the CLI's 4096, and the block IS the
    registry block at cap 4096 (a store drawn under the map's 512 would not resume it)."""
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="cap-e2e.json"))
    fake, got, args = _wire(monkeypatch, tmp_path)
    args.multiturn_target_decode, args.model_key = "vendor", "qwen35_4b"
    args.multiturn_target_seed_base, args.multiturn_target_policy_ref = 1, MT_REF
    args.target_max_new_tokens = 4096                                       # the canonical 4,096-cap sbatch passes it
    cfg = _served_cfg()
    cfg["attacks"]["multiturn_target_max_new_tokens"] = {"crescendo": 512, "pair": 4096, "actorattack": 512}
    assert jr._multiturn_target_max_new_tokens(cfg, "crescendo") == 512    # the known-bad answer the manifest used to record
    jr.cmd_generate(args, cfg)
    out = Path(got["out_dir"])
    eng = json.loads((out / "attacker_engine.json").read_text())["target_decode"]
    doc = json.loads((out / "served_target.json").read_text())
    assert got["target_max_new_tokens"] == 4096 == eng["decode_policy"]["target_cap"] == doc["decode"]["cap"]
    assert doc["decode"] == _registry_block(1) and "cap" in st.TARGET_DECODE_IDENTITY_KEYS
    assert st._decode_identity(doc["decode"]) != st._decode_identity({**_registry_block(1), "cap": 512})
    # the same config WITHOUT the CLI flag: dispatch, check and manifest all follow the map's 512 -- and the row's
    # `4096 per reply` then refuses before a byte lands (the P2-3 check), the manifest never written
    fake2, got2, args2 = _wire(monkeypatch, tmp_path / "nocli")
    args2.multiturn_target_decode, args2.model_key = "vendor", "qwen35_4b"
    args2.multiturn_target_seed_base, args2.multiturn_target_policy_ref = 1, MT_REF
    with pytest.raises(SystemExit, match="names 4096 per reply but this run's target cap is 512"):
        jr.cmd_generate(args2, cfg)
    assert "out_dir" not in got2


# --------------------------------------------------------------------------- #
# the sbatch                                                                    #
# --------------------------------------------------------------------------- #

SBATCH = REPO / "scripts/sdfbase/full22/target_attacks.sbatch"
GOLDEN_TAIL = ["scripts/jailbreak_run.py", "generate", "--config", "/cfg.yaml", "--attack", "crescendo", "--defense", "none",
               "--confirmatory", "--shard", "0", "--nshards", "9", "--private-root", "/priv",
               "--attacker-engine", "vllm", "--attacker-endpoint", "http://a:8000",
               # 31d (060c2c919): the bound member first, then the other live members, so a driver re-binds when its member retires
               "--attacker-endpoints", "http://a:8000",
               "--target-engine", "served", "--target-proxy-url", "http://p/v1", "--target-routing-table", "/rt.json",
               "--target-max-model-len", "49152", "--expect-pair-n-streams", "5",
               "--target-max-new-tokens", "4096", "--target-timeout-s", "3600",
               "--multiturn-target-decode", "vendor", "--multiturn-target-seed", "per_turn", "--model-key", "qwen35_4b"]


def _render_driver_line(tmp_path, env):
    src = SBATCH.read_text()
    # the driver line carries two per-driver arrays, both declared EMPTY here the way the loop declares them when it has
    # nothing to hand over: the low-dose liveness record rides in FRONT of the interpreter as an `env` prefix (LIVE_ENV,
    # the attacks leg since 2026-09-18) and the attacker member's serving manifest after its endpoint (ASERV,
    # --attacker-serving, so the store records a real attacker_quantization). Empty = the rendered tail is byte-identical.
    i = src.index('( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" ${LIVE_ENV[@]+"${LIVE_ENV[@]}"} "$MPY" scripts/jailbreak_run.py generate')
    lines, out = src[i:].splitlines(), []
    for line in lines:
        out.append(line)
        if line.rstrip().endswith("2>&1 &"):
            break
    snippet = "\n".join(out)
    tmp_path.mkdir(parents=True, exist_ok=True)
    mpy = tmp_path / "mpy.sh"
    mpy.write_text('#!/bin/bash\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    mpy.chmod(0o755)
    (tmp_path / "drivers").mkdir(exist_ok=True)
    vars_ = (f"REPO_DIR={shlex.quote(str(tmp_path))}; MPY={shlex.quote(str(mpy))}; CFG=/cfg.yaml; atk=crescendo; cell=none; s=0; "
             f"NSHARDS=9; PRIVATE_ROOT=/priv; url=http://a:8000; PROXY_URL=http://p/v1; TABLE=/rt.json; TARGET_L=49152; "
             f"KEY=qwen35_4b; SERVE_DIR={shlex.quote(str(tmp_path))}; tag=x; "
             "DRIVER_CAP_FLAGS=(--target-max-new-tokens 4096 --target-timeout-s 3600); ASERV=(); LIVE_ENV=(); "
             "ATTACKERS=(http://a:8000); ")   # the live-members array 31d's --attacker-endpoints iterates: the one bound member here
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env.items())
    r = subprocess.run(["bash", "-c", "set -u; " + vars_ + exports + snippet + "\nwait"], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return (tmp_path / "drivers" / "x.log").read_text().splitlines()


def test_the_sbatch_forwards_the_two_scalars_only_when_set(tmp_path):
    src = SBATCH.read_text()
    assert '--multiturn-target-seed "${F22_MT_TARGET_SEED:-per_turn}"' in src                  # the standing pin
    assert _render_driver_line(tmp_path / "unset", {}) == GOLDEN_TAIL                                 # byte-identical when unset
    both = _render_driver_line(tmp_path / "both", {"F22_MT_TARGET_SEED_BASE": "1", "F22_MT_TARGET_POLICY_REF": MT_REF})
    assert both == GOLDEN_TAIL + ["--multiturn-target-seed-base", "1", "--multiturn-target-policy-ref", MT_REF]
    assert _render_driver_line(tmp_path / "seed", {"F22_MT_TARGET_SEED_BASE": "2"}) == GOLDEN_TAIL + ["--multiturn-target-seed-base", "2"]
    assert _render_driver_line(tmp_path / "ref", {"F22_MT_TARGET_POLICY_REF": MT_REF}) == GOLDEN_TAIL + ["--multiturn-target-policy-ref", MT_REF]
    # the driver's own parser accepts the rendered tail (the target-engine group)
    p = __import__("argparse").ArgumentParser()
    jr._add_target_engine_args(p)
    ns, _rest = p.parse_known_args(both[2:])
    assert (ns.multiturn_target_decode, ns.multiturn_target_seed, ns.model_key) == ("vendor", "per_turn", "qwen35_4b")
    assert ns.multiturn_target_seed_base == 1 and ns.multiturn_target_policy_ref == MT_REF
    # the env names are read as SCALARS and never assigned by the job; no --export= anywhere
    import re
    for name in ("F22_MT_TARGET_SEED_BASE", "F22_MT_TARGET_POLICY_REF", "F22_MT_TARGET_SEED", "F22_MT_TARGET_DECODE"):
        assert re.search(r"\$\{" + name + r"[:}]", src), name
        assert not re.search(r"^\s*" + name + "=", src, flags=re.M), name
    assert not any("--export=" in l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True).returncode == 0


# --------------------------------------------------------------------------- #
# the request-evidence seam                                                   #
# --------------------------------------------------------------------------- #


class _SeamDefense(_Defense):
    """A served-shaped defense that EXPOSES the seam the shared client will add to ServedChatTarget."""

    def __init__(self):
        super().__init__()
        self.contexts = []
        self.last_request_id = None
        self.last_completion_id = None

    def set_request_context(self, **ctx):
        self.contexts.append(dict(ctx))

    def chat(self, messages, **kw):
        n = len(self.calls)
        self.last_request_id, self.last_completion_id = f"rid-{n}", f"cmpl-{n}"
        return super().chat(messages, **kw)


def test_the_request_evidence_seam_is_inert_today_and_binds_rows_when_a_target_exposes_it(tmp_path):
    from sisteer.eval.jailbreak_data import Goal
    goals = [Goal("g0", "benign objective", "illegal", "harmbench", "rev")]
    plan = '{"actors": [{"name": "a1", "queries": ["q0", "q1"]}]}'
    # inert: a target without the seam -> rows carry neither id (byte-identical to before)
    d = _Defense()
    multiturn.run_actorattack(d, goals, attacker_fn=lambda chat: plan, scorer_fn=lambda p: "1", out_dir=tmp_path / "plain",
                              resume=False, actor_num=1, discovery_retries=0, max_new_tokens=64,
                              target_decode=VENDOR_SIX, seed_base=1, trial=0)
    rows = [json.loads(l) for l in (tmp_path / "plain" / "responses.jsonl").read_text().splitlines() if l.strip()]
    assert rows and all("request_id" not in r and "completion_id" not in r for r in rows)
    # armed: the context names every target call (attack, goal, turn, trial) and the rows bind to the attempt ids
    sd = _SeamDefense()
    multiturn.run_actorattack(sd, goals, attacker_fn=lambda chat: plan, scorer_fn=lambda p: "1", out_dir=tmp_path / "seam",
                              resume=False, actor_num=1, discovery_retries=0, max_new_tokens=64,
                              target_decode=VENDOR_SIX, seed_base=1, trial=0)
    assert sd.contexts == [{"kind": "multi_turn", "attack": "actorattack", "goal_id": "g0", "turn": i, "trial": 0}
                           for i in range(len(sd.calls))]
    rows = [json.loads(l) for l in (tmp_path / "seam" / "responses.jsonl").read_text().splitlines() if l.strip()]
    assert [r["request_id"] for r in rows] == [f"rid-{i}" for i in range(len(rows))]
    assert [r["completion_id"] for r in rows] == [f"cmpl-{i}" for i in range(len(rows))]
    # the PyRIT target: the same context, set inside the lock right before the counted forward
    from pyrit.models import Message
    asyncio.run(pa.init_pyrit_memory_async(tmp_path / "memory.db"))

    async def one_turn(tgt):
        out = await tgt._send_prompt_to_target_async(normalized_conversation=[Message.from_prompt(prompt="hi", role="user")])
        return out[0].get_value()
    sd2 = _SeamDefense()
    tgt = pa.SteeredPromptTarget(defense=sd2, target_decode=VENDOR_SIX, seed_fn=lambda turn: 100 + turn,
                                 request_context_fn=lambda turn: {"kind": "multi_turn", "attack": "pair", "goal_id": "g1", "turn": turn, "trial": 0})
    tgt.begin_goal()
    assert asyncio.run(one_turn(tgt)) == "ok" and asyncio.run(one_turn(tgt)) == "ok"
    assert sd2.contexts == [{"kind": "multi_turn", "attack": "pair", "goal_id": "g1", "turn": 0, "trial": 0},
                            {"kind": "multi_turn", "attack": "pair", "goal_id": "g1", "turn": 1, "trial": 0}]
    assert [c["seed"] for c in sd2.calls] == [100, 101]
    plain = pa.SteeredPromptTarget(defense=_Defense(), target_decode=VENDOR_SIX, seed_fn=lambda turn: 7)
    assert plain._request_context_fn is None and asyncio.run(one_turn(plain)) == "ok"          # no fn, no context, no change
