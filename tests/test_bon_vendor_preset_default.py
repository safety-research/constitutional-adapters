"""Best-of-N decodes at EACH SERVED MODEL'S VENDOR PRESET, replacing the
grandfathered model-independent temperature 1.0.

What is pinned, and what failed before the change:

* the registry: `<model>|bon|off|vendor_preset` is the DEFAULT row for `strongreject_bon` on every served
  model, `same_values_as` the model's static vendor row (nemotron: the shipped 0.95, ruling
  decode-nemotron-vendor-preset-0.95-2026-09-17), and the grandfathered rows are SUPERSEDED but readable by
  name — before, the grandfathered rows were the defaults and no vendor_preset BoN row existed;
* the library: `bon_served.bon_sampler(model)` is the model's row (Qwen's presence_penalty 1.5 included), a BoN
  request body carries exactly those keys + seed + max_tokens, every row and the manifest record the label AND
  the registry policy id, a nameless call refuses — before, `bon_sampler` did not exist and `_sent_sampler(None)`
  was a constant {temperature: 1.0} for every model;
* the climb: `bon_climb.store_sampler` reads the sampler a store was drawn under and the climb follows it (a
  T=1 store climbs at T=1, a vendor-preset store at its preset, a legacy store names the grandfathered row, a
  mixture refuses) — before, `bon_climb.CLIMB_TEMPERATURE` was the constant 1.0;
* the seed rule and the augmentation are untouched by the sampler: the attack's diversity is the augmentation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.eval.attacks import bon as bon_mod
from sisteer.eval.attacks import bon_climb as bc
from sisteer.eval.attacks import bon_served as bs
from sisteer.registry import decode_policy as DP

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
EXPORT = REPO / "configs" / "registry" / "decode_policy.json"

#: the four paper families (legacy key -> model_id) and the request-body keys their vendor row sends
FAMILIES = {
    "gemma4e4b": ("gemma4e4b_it", {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
    "qwen35_4b": ("qwen35_4b_it", {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                   "presence_penalty": 1.5, "repetition_penalty": 1.0}),
    "nemotron3nano4b": ("nemotron3nano4b", {"temperature": 1.0, "top_p": 0.95}),
    "llama8b": ("llama8b_it", {"temperature": 0.6, "top_p": 0.9}),
}
T1 = {"temperature": 1.0}


# --------------------------------------------------------------------------- the registry

def test_every_family_has_a_default_vendor_preset_bon_row_and_the_grandfathered_row_is_live_and_non_default():
    pf = DP.load(str(EXPORT))
    for key, (mid, want) in FAMILIES.items():
        p = pf.resolve(key, eval_id="strongreject_bon", thinking_mode="off")
        assert p.policy_id == f"{mid}|bon|off|vendor_preset@v1" and p.mode == "vendor_preset" and p.is_default == 1
        assert p.request_params() == want, key
        assert p.params_sent == (*[k for k in bs.SAMPLER_KEYS if k in want], "seed") or set(p.params_sent) == set(want) | {"seed"}
        # the same numbers the sampled static lane sends (nemotron: the SHIPPED 0.95 row, @v2, by the 2026-09-17 ruling)
        static_ref = "nemotron3nano4b|static|off|vendor_preset@v2" if key == "nemotron3nano4b" else f"{mid}|static|off|vendor_preset"
        assert pf.by_ref(static_ref).numbers() == p.numbers(), key
        old = pf.by_ref(f"{mid}|bon|off|grandfathered_T1")
        # superseded by the vendor row from 2026-09-17 to 2026-10-14; LIVE and non-default since the rule of 2026-10-14
        # (decode-bon-canonical-T1-live-non-default-2026-10-14: canonical best-of-N stays the T = 1 instrument, so the row its draws send
        # stands beside the default instead of under it) -- resolve() still returns the vendor row, find() lists both
        assert old.superseded_by_policy_id is None and old.is_default == 0 and old.request_params() == T1
        assert old in pf.find(mid, eval_id="strongreject_bon") and p in pf.find(mid, eval_id="strongreject_bon")     # both live, one default
        assert p.ruling_ref == "decode-bon-vendor-preset-2026-09-17"
    assert pf.resolve("nemotron3nano4b", eval_id="strongreject_bon").request_params()["top_p"] == 0.95
    assert pf.resolve("qwen35_4b", eval_id="strongreject_bon").request_params()["presence_penalty"] == 1.5


# --------------------------------------------------------------------------- the library

def test_bon_sampler_is_the_models_vendor_row_and_the_grandfathered_row_only_by_name():
    for key, (mid, want) in FAMILIES.items():
        assert bs.bon_sampler(key) == want and bs.bon_sampler(mid) == want
        assert bs.bon_policy(key).policy_id == f"{mid}|bon|off|vendor_preset@v1"
        assert bs.bon_sampler(key, bs.BON_HARNESS_DEFAULT) == T1
        s = bs.resolve_bon_sampler(None, None, key)
        assert (s.decode, s.label, s.policy_id) == (want, bs.BON_VENDOR_PRESET, f"{mid}|bon|off|vendor_preset@v1")
    # the HF repo names the same row; Qwen's presence penalty rides the request
    assert bs.bon_sampler("Qwen/Qwen3.5-4B")["presence_penalty"] == 1.5
    # no model, no block: refused (no model-independent default); the grandfathered row alone may resolve nameless
    with pytest.raises(bs.BonServedError, match="no model was named"):
        bs.resolve_bon_sampler(None, None, None)
    assert bs.resolve_bon_sampler(None, bs.BON_HARNESS_DEFAULT, None).decode == T1
    # an unknown model, an explicit block wearing a registry label, a non-registry label without a block: refused
    with pytest.raises(bs.BonServedError, match="names no model"):
        bs.bon_sampler("no_such_model")
    with pytest.raises(bs.BonServedError, match="registry's to resolve"):
        bs.resolve_bon_sampler(T1, bs.BON_VENDOR_PRESET, "gemma4e4b")
    with pytest.raises(bs.BonServedError, match="not a registry label"):
        bs.resolve_bon_sampler(None, "vendor", "gemma4e4b")
    # an explicit block is the caller's: label `vendor`, no policy id
    e = bs.resolve_bon_sampler({"temperature": 0.6, "top_p": 0.9}, None, None)
    assert (e.decode, e.label, e.policy_id) == ({"temperature": 0.6, "top_p": 0.9}, bs.BON_EXPLICIT_PRESET, None)
    # the constants of the old model-independent default are gone
    assert not hasattr(bs, "HARNESS_DEFAULT_SAMPLER") and not hasattr(bs, "HARNESS_DEFAULT_TEMPERATURE")


def test_a_bon_request_carries_exactly_the_familys_vendor_keys_plus_seed_and_max_tokens():
    sd = bs.ServedDefense(lambda p: None, virtual_model="petri__base", cell_id="none")
    for key, (_, want) in FAMILIES.items():
        req = sd._request("variant text", sampling_seed=1234, max_new_tokens=512, decode=bs.bon_sampler(key))
        assert {k: req[k] for k in want} == want and req["seed"] == 1234 and req["max_tokens"] == 512
        assert set(req) == {"model", "messages", "max_tokens", "seed", *want}, key
    q = sd._request("v", sampling_seed=1, max_new_tokens=8, decode=bs.bon_sampler("qwen35_4b"))
    assert q["presence_penalty"] == 1.5 and q["temperature"] == 0.7


def _world(tmp_path):
    import test_bon_served as tb
    backend = tb.RecordingBackend()
    proxy, routes = tb._proxy(tmp_path, backend)
    return tb, backend, routes, bs.ServedDefense.over_proxy(proxy, virtual_model="petri__base", cell_id="none")


def test_bon_generate_served_draws_the_named_models_vendor_preset_and_records_the_policy(tmp_path):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(2)
    for key, (mid, want) in FAMILIES.items():
        out = tmp_path / key
        jsonl = bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=2, max_concurrency=4, model=key)
        rows = tb._read(jsonl)
        assert len(rows) == 4
        for r in rows:
            assert r["decode"] == want and r["bon_decode_preset"] == bs.BON_VENDOR_PRESET
            assert r["decode_policy_id"] == f"{mid}|bon|off|vendor_preset@v1"
            # the seed rule is untouched by the sampler: the per-draw bon_seed and its rule, row by row
            assert r["seed_rule"] == bon_mod.BON_SEED_RULE and r["sampling_seed"] == bon_mod.bon_seed(r["goal_id"], r["sample_idx"], r["trial"])
        man = json.loads((out / bs.BON_SERVED_MANIFEST).read_text())
        assert man["decode"] == want and man["bon_decode_preset"] == bs.BON_VENDOR_PRESET
        assert man["decode_policy_id"] == f"{mid}|bon|off|vendor_preset@v1"
        sent = [p for p in backend.payloads if p.get("max_tokens")]
        assert sent and all({k: p[k] for k in want} == want and "seed" in p for p in sent[-4:]), key
        backend.payloads.clear()
    # the augmentation is the attack's diversity, not the temperature: the same (goal, sample) is the same variant
    # under every family's sampler
    variants = {key: sorted((r["goal_id"], r["sample_idx"], r["variant"]) for r in tb._read(tmp_path / key / "bon_responses.jsonl"))
                for key in FAMILIES}
    assert len({json.dumps(v) for v in variants.values()}) == 1
    # a vendor-preset store is never resumed under another family's preset, nor under the grandfathered row
    with pytest.raises(bs.BonServedError, match="drawn under a different sampler"):
        bs.bon_generate_served(defense, goals, out_dir=tmp_path / "gemma4e4b", routes=routes, n_max=2, max_concurrency=4, model="llama8b")
    with pytest.raises(bs.BonServedError, match="drawn under a different sampler"):
        bs.bon_generate_served(defense, goals, out_dir=tmp_path / "gemma4e4b", routes=routes, n_max=2, max_concurrency=4,
                               decode_label=bs.BON_HARNESS_DEFAULT)


# --------------------------------------------------------------------------- the climb follows its store

def _store(tmp_path, rows):
    """A COMPLETE 512 store (f4dc4b8e1 / d0d1b8eb4): the rows, the durable manifest's `units` deciding every planned slot, and the
    completion marker the 512 wave writes — has_base_store -> store_state reads STORE_COMPLETE only with all three; a bare rows file
    is a store IN PROGRESS (a writer that died before its marker), which the climb refuses to read."""
    d = tmp_path / "cell"; d.mkdir(parents=True, exist_ok=True)
    (d / bc.BON_RESPONSES).write_text("".join(json.dumps(r) + "\n" for r in rows))
    units = {"n_planned": len(rows), "n_done": len(rows), "n_nofit": 0, "n_undecided": 0}
    (d / bc.BON_STORE_MANIFEST_UNITS).write_text(json.dumps({"units": units}))
    (d / bc.BON_STORE_MANIFEST).write_text(json.dumps({"complete": True, "units": units}))
    return d


def _row(gid, idx, sampler=None, label=None, pid=None):
    r = {"goal_id": gid, "sample_idx": idx, "trial": 0, "variant": f"v{idx}", "response": "text " * 40, "finish_reason": "length"}
    if sampler is not None:
        r.update({"decode": sampler, "bon_decode_preset": label, "decode_policy_id": pid})
    return r


def test_the_climb_re_draws_at_the_sampler_its_store_records_never_a_constant(tmp_path):
    assert bc.CLIMB_DECODE == "as_drawn" and not hasattr(bc, "CLIMB_TEMPERATURE")
    gem = FAMILIES["gemma4e4b"][1]
    # a vendor-preset store: its rows name the row; the manifest carries it row by row
    d = _store(tmp_path / "vp", [_row("g0", 0, gem, bs.BON_VENDOR_PRESET, "gemma4e4b_it|bon|off|vendor_preset@v1"),
                                 _row("g0", 1, gem, bs.BON_VENDOR_PRESET, "gemma4e4b_it|bon|off|vendor_preset@v1")])
    assert bc.store_sampler(d) == {"decode": gem, "bon_decode_preset": bs.BON_VENDOR_PRESET,
                                   "decode_policy_id": "gemma4e4b_it|bon|off|vendor_preset@v1"}
    rows = bc.climb_manifest_rows(d, lane="gemma4e4b", cell_id="none")
    assert all(r["decode"] == gem and r["bon_decode_preset"] == bs.BON_VENDOR_PRESET for r in rows)
    # a T=1 store (every existing store): climbs at T=1
    d1 = _store(tmp_path / "t1", [_row("g0", 0, T1, "harness_default", "gemma4e4b_it|bon|off|grandfathered_T1@v1")])
    assert bc.store_sampler(d1)["decode"] == T1 and bc.climb_manifest_rows(d1, lane="l", cell_id="c")[0]["decode"] == T1
    # a legacy store (rows record nothing): None, and the manifest names the grandfathered row for the driver
    dl = _store(tmp_path / "legacy", [_row("g0", 0)])
    assert bc.store_sampler(dl) is None
    lr = bc.climb_manifest_rows(dl, lane="l", cell_id="c")[0]
    assert lr["decode"] is None and lr["bon_decode_preset"] == bc.LEGACY_STORE_PRESET == bs.BON_HARNESS_DEFAULT
    # a mixture is never a store: two samplers, or recorded beside unrecorded rows
    dm = _store(tmp_path / "mix", [_row("g0", 0, gem, bs.BON_VENDOR_PRESET, None), _row("g0", 1, T1, "harness_default", None)])
    with pytest.raises(bc.BonClimbError, match="different samplers"):
        bc.store_sampler(dm)
    dm2 = _store(tmp_path / "mix2", [_row("g0", 0, gem, bs.BON_VENDOR_PRESET, None), _row("g0", 1)])
    with pytest.raises(bc.BonClimbError, match="recorded and legacy"):
        bc.store_sampler(dm2)
    # the planner's rows say the rule
    assert bc.plan_cells([{"cell_id": "c", "store_dir": d}])[0]["decode"] == "as_drawn"


def test_the_roster_and_every_wave_config_state_the_as_drawn_rule_not_a_temperature():
    import yaml
    from full22_wave_configs import wave_configs
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
    assert roster["bon_climb_decode"] == bc.CLIMB_DECODE and "bon_climb_temperature" not in roster
    n = 0
    for path in wave_configs(REPO):
        bon = yaml.safe_load(path.read_text())["bon"]
        assert bon["climb_decode"] == "as_drawn" and "climb_temperature" not in bon, path.name
        n += 1
    assert n >= 7
    drv = (REPO / "scripts/sdfbase/full22/bon_served_run.py").read_text()
    assert 'default="vendor_preset"' in drv
    # the served model is DERIVED from the config the driver loads (cfg["model"]); --model-key is an optional cross-check,
    # so every existing invocation (the wave seams) parses unchanged
    assert "served_model_of(cfg, table, args.model_key" in drv and '"--model-key", default=None' in drv
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    import bon_served_run as br
    ns = br.build_parser().parse_args(["--config", "c", "--defense", "d", "--private-root", "r", "--proxy-url", "u",
                                       "--routing-table", "t", "--max-concurrency", "4"])
    assert ns.bon_decode_preset == bs.BON_VENDOR_PRESET and ns.model_key is None
    assert br.served_model_of({"model": "google/gemma-4-E4B-it"}, {"base_model": "google/gemma-4-E4B-it"}, None, tag="t") == "google/gemma-4-E4B-it"
    assert br.served_model_of({"model": "google/gemma-4-E4B-it"}, {}, "gemma4e4b", tag="t") == "google/gemma-4-E4B-it"
    with pytest.raises(SystemExit, match="names no served"):
        br.served_model_of({}, {}, None, tag="t")
    with pytest.raises(SystemExit, match="disagrees with the config"):
        br.served_model_of({"model": "google/gemma-4-E4B-it"}, {}, "llama8b", tag="t")
    with pytest.raises(SystemExit, match="routing table serves base_model"):
        br.served_model_of({"model": "google/gemma-4-E4B-it"}, {"base_model": "Qwen/Qwen3.5-4B"}, None, tag="t")
    sb = (REPO / "scripts/sdfbase/full22/target_static.sbatch").read_text()
    assert '${F22_BON_DECODE_PRESET:-vendor_preset}' in sb
