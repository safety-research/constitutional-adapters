"""The vendor-preset decode DEFAULT (`sisteer.eval.sampled_decode`):
the data file `configs/sdfbase_sampled_decode.yaml` validates under the drivers' own reader (the
Llama-3.1-8B-Instruct row included, equal to the repo's pinned card row and to the llama8b wave's
row byte for byte; the GLM row kept for provenance); the Qwen row IS the model card and
agrees with main's vLLM server default (`--override-generation-config`) on every field the
server can default, `presence_penalty` riding per request only; the block the drivers inject
equals the generator's rendering; the resolver injects / refuses naming `--greedy` / records the
opt-in; the three sampler-key vocabularies are one; and, end to end on the controls driver, a
greedy run opts in and records it while an unopted run on a model with no row refuses before any
store exists and a model WITH a row draws under the preset.

Synthetic worlds only (test_served_capability's canonical served world); no real recorded data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_sampled_configs as msc  # noqa: E402
import served_capability_run as srun  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.eval import cell_controls as cc  # noqa: E402
from sisteer.eval import sampled_decode as sdd  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402

from test_served_capability import BASE_MODEL, REV, LEGACY_GEN_CONTROLS, world, _cell_dir  # noqa: E402,F401

PRESETS = sdd.load_decode_presets()
WAVE_LLAMA_ROW = {
    "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
    "preset": {"temperature": 0.6, "top_p": 0.9},
    "source_kind": "generation_config",
    "source": ("NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79: generation_config.json (do_sample: true; "
               "temperature 0.6, top_p 0.9); no top_k, no penalty term; read 2026-09-09"),
}


# --------------------------------------------------------------------------- #
# the data file                                                                #
# --------------------------------------------------------------------------- #


def test_the_data_file_validates_under_the_drivers_reader_and_the_generators():
    # + the paper-companion-repo FRONT-DOOR row (2026-09-29): qwen35_4b_demo (configured, not run), read from its shipped
    # generation_config.json (tests/test_registry_seeds_test_models.py pins its value); the drivers match a row by model + revision
    assert set(PRESETS["presets"]) == {"gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b", "qwen35_4b_demo"}
    assert msc.load_presets() == PRESETS                       # the generator's reader agrees
    assert PRESETS["arm"] == "vendor_recommended" and PRESETS["applies_to"] == ["static_families", "controls"]
    for key, p in PRESETS["presets"].items():
        assert set(p["preset"]) <= set(sdd.SAMPLER_KEYS) and p["preset"]["temperature"] > 0, key


def test_llama_row_is_the_waves_row_and_the_repos_pinned_card_row():
    """The sdfbase-llama8b wave's row, byte for byte (so its rebase merges clean), equal to the
    repo's pinned card row (contracts._LLAMA_CARD: 0.6 / 0.9 / no top_k) — the values of
    meta-llama/Llama-3.1-8B-Instruct's published generation_config.json. Neither cached snapshot
    on this box carries the file, so the row is verified against the repo's pins, not a fresh read."""
    row = PRESETS["presets"]["llama8b"]
    assert {k: row[k] for k in WAVE_LLAMA_ROW} == WAVE_LLAMA_ROW
    card = tc.SAMPLING_BY_MODEL_KEY["llama8b"]
    assert (card.temperature, card.top_p, card.top_k) == (0.6, 0.9, None)
    assert row["preset"] == {"temperature": card.temperature, "top_p": card.top_p}
    # a row for a target the roster does not (yet) name is legal: the generator's pin check admits it
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
    if "llama8b" not in roster["targets"]:
        msc.check_pins(PRESETS, roster)                        # no SystemExit
        assert "llama8b" not in msc.SAMPLED_VIEWS               # rendered configs stay the four targets'
    with pytest.raises(SystemExit, match="names a target the roster does not have"):
        msc.check_pins({**PRESETS, "presets": {**PRESETS["presets"], "gemma4e4b": row}},
                       {**roster, "targets": {k: v for k, v in roster["targets"].items() if k != "gemma4e4b"}})


def test_glm_row_is_kept_for_provenance_and_marked_not_in_the_paper():
    text = (REPO / "configs/sdfbase_sampled_decode.yaml").read_text()
    assert "glm46vflash" in PRESETS["presets"]
    assert "NOT IN THE PAPER (GLM dropped 2026-09-11)" in text and "kept for the completed sampled run's provenance" in text
    assert PRESETS["presets"]["glm46vflash"]["preset"] == {"temperature": 0.8, "top_p": 0.6, "top_k": 2,
                                                            "repetition_penalty": 1.1}


def test_qwen_row_is_the_model_card_and_agrees_with_the_servers_default_except_presence_penalty():
    """The vLLM server pins Qwen3.5's vendor instruct row as its default
    (`--override-generation-config`), so an omit-params request decodes at the same
    temperature / top_p / top_k / min_p / repetition_penalty the yaml preset names;
    `presence_penalty` cannot be a server default (the OpenAI field defaults to 0.0) and
    rides PER REQUEST — the yaml row carries it, the composed override does not, and both
    drivers send it explicitly."""
    row = PRESETS["presets"]["qwen35_4b"]
    assert row["source_kind"] == "model_card"
    assert row["preset"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                             "presence_penalty": 1.5, "repetition_penalty": 1.0}
    server = tc.SERVER_DEFAULT_SAMPLING_BY_MODEL["Qwen/Qwen3.5-4B"]
    assert row["model"] == "Qwen/Qwen3.5-4B"
    for k in tc.SERVER_DEFAULTABLE_SAMPLING_KEYS:
        assert getattr(server, k) == row["preset"][k], k
    assert "presence_penalty" not in tc.SERVER_DEFAULTABLE_SAMPLING_KEYS
    assert server.presence_penalty == row["preset"]["presence_penalty"] == 1.5
    composed = json.loads(tc.server_default_generation_config(server))
    assert set(composed) == {"temperature", "top_p", "top_k", "min_p", "repetition_penalty"}
    assert "presence_penalty" not in composed
    # the drivers send presence_penalty on the request: the controls payload and the static preset
    payload = srun._chat_payload("petri__x", "hi", system=None, max_tokens=512, decode=row["preset"], seed=7)
    assert payload["presence_penalty"] == 1.5 and payload["seed"] == 7 and "n" not in payload
    assert [k for k in srun.SAMPLED_DECODE_SAMPLER_KEYS if k in payload] == list(row["preset"])
    assert bs.sampler_preset(row["preset"])["presence_penalty"] == 1.5


def test_the_sampler_key_vocabulary_is_one():
    assert tuple(sdd.SAMPLER_KEYS) == tuple(bs.SAMPLER_KEYS) == tuple(cc.SAMPLED_DECODE_SAMPLER_KEYS) \
        == tuple(msc.ALLOWED_SAMPLER_KEYS)


# --------------------------------------------------------------------------- #
# the resolver                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", sorted(msc.SAMPLED_VIEWS))
def test_injected_block_equals_the_generators_rendering(key):
    """What the drivers inject IS the block make_sampled_configs renders for the same
    key/trial (minus the per-config `escalated_from` a rung carries) — one contract."""
    from sisteer.eval import capability_omissions as co
    for trial in (0, 1):
        want = msc.expected_block(PRESETS, key, trial, False)
        got = sdd.default_sampled_block(PRESETS, key, trial=trial, static_cap=msc.STATIC_CAP_BASE,
                                        controls_cap=msc.CONTROLS_CAP)
        # what the drivers inject = the shared builder's block + the lane overlay on its readouts (the roster's
        # capability omissions: llama8b's gsm8k), applied right after resolve_default_decode
        got["readouts"] = co.readouts_with_omissions(got["readouts"], key)
        assert got == want, key


def test_resolver_injects_refuses_and_records():
    cfg = {"model": "Qwen/Qwen3.5-4B", "model_revision": PRESETS["presets"]["qwen35_4b"]["revision"],
           "attacks": {"static_max_new_tokens": 4096, "target_max_new_tokens": 512}}
    out, rec = sdd.resolve_default_decode(cfg, greedy=False, trial=1, static_cap=4096, controls_cap=512)
    assert "sampled_decode" not in cfg and out["sampled_decode"]["trial"] == 1
    assert out["sampled_decode"]["temperature"] == 0.7 and out["sampled_decode"]["presence_penalty"] == 1.5
    assert rec["decode_default"] == sdd.DEFAULT_VENDOR_PRESET and rec["preset_key"] == "qwen35_4b"
    assert rec["presets_file"] == "configs/sdfbase_sampled_decode.yaml" and len(rec["presets_sha256"]) == 64
    # the worked example's row (qwen35_4b_demo) serves the SAME checkpoint with a byte-identical preset and source_kind: identical rows for one
    # (model, revision) collapse to the FIRST key in file order (the paper's), never an ambiguity; a row that differs in one number still refuses
    demo, paper = PRESETS["presets"]["qwen35_4b_demo"], PRESETS["presets"]["qwen35_4b"]
    assert (demo["model"], demo["revision"], demo["preset"], demo["source_kind"]) == (paper["model"], paper["revision"], paper["preset"], paper["source_kind"])
    assert sdd.preset_key_for_model(PRESETS, "Qwen/Qwen3.5-4B", paper["revision"]) == "qwen35_4b" == sdd.preset_key_for_model(PRESETS, "Qwen/Qwen3.5-4B")
    import copy
    differing = copy.deepcopy(PRESETS); differing["presets"]["qwen35_4b_demo"]["preset"]["temperature"] = 0.9
    with pytest.raises(sdd.DecodePresetUnavailable, match="2 presets name served model 'Qwen/Qwen3.5-4B'.*ambiguous"):
        sdd.preset_key_for_model(differing, "Qwen/Qwen3.5-4B", paper["revision"])
    # ... and so do two rows whose preset values agree but whose REVISION (or recorded shipped generation config) differs: a
    # preset read from one snapshot never stands in for another's, so the collapse is keyed on the whole row, not the numbers
    other_rev = copy.deepcopy(PRESETS); other_rev["presets"]["qwen35_4b_demo"]["revision"] = "0" * 40
    with pytest.raises(sdd.DecodePresetUnavailable, match="2 presets name served model 'Qwen/Qwen3.5-4B'.*ambiguous"):
        sdd.preset_key_for_model(other_rev, "Qwen/Qwen3.5-4B")
    other_cfg = copy.deepcopy(PRESETS); other_cfg["presets"]["qwen35_4b_demo"]["shipped_generation_config"] = {"temperature": 0.6}
    with pytest.raises(sdd.DecodePresetUnavailable, match="ambiguous"):
        sdd.preset_key_for_model(other_cfg, "Qwen/Qwen3.5-4B", paper["revision"])
    # both drivers' validators admit the injected block
    assert cc.controls_gen_config(out)["do_sample"] is True and cc.controls_gen_config(out)["max_new_tokens"] == 512
    import jailbreak_run as jr
    assert jr._sampled_decode(out, 1)["decode"]["arm"] == "vendor_recommended"
    # the revision must agree when both name one
    with pytest.raises(sdd.DecodePresetUnavailable, match="serves revision"):
        sdd.resolve_default_decode({**cfg, "model_revision": "0" * 40}, greedy=False, trial=0)
    # no row -> refuse, naming the flag; never a default temperature
    with pytest.raises(sdd.DecodePresetUnavailable, match="--greedy"):
        sdd.resolve_default_decode({"model": "org/no-such-model"}, greedy=False, trial=0)
    with pytest.raises(sdd.DecodePresetUnavailable, match="names no served model"):
        sdd.resolve_default_decode({}, greedy=False, trial=0)
    # the opt-in leaves the config untouched and says so
    same, rec = sdd.resolve_default_decode({"model": "org/no-such-model"}, greedy=True, trial=0)
    assert "sampled_decode" not in same and rec["decode_default"] == sdd.DEFAULT_GREEDY_OPT_IN
    assert "--greedy" in rec["reason"] and rec["preset_key"] is None
    # a config carrying its own block is the contract; nothing injected, whatever the flag
    own = {"model": "org/no-such-model", "sampled_decode": {"arm": "x"}}
    for greedy in (False, True):
        kept, rec = sdd.resolve_default_decode(own, greedy=greedy, trial=0)
        assert kept["sampled_decode"] == {"arm": "x"} and rec["decode_default"] == "config_block"
    # the served model wins over the config's when the caller names it (the controls driver)
    out, rec = sdd.resolve_default_decode({"model": "org/other"}, greedy=False, trial=0, controls_cap=512,
                                          model="google/gemma-4-E4B-it")
    assert rec["preset_key"] == "gemma4e4b" and out["sampled_decode"]["top_k"] == 64


def test_the_worked_examples_readouts_render_as_the_paper_keys():
    """qwen35_4b_demo serves the paper's Qwen3.5-4B checkpoint, so its GSM8K readout is the 4B's: the data file carries the
    vendor's THINKING row under both keys (verbatim), and both renderers — the shared builder and the generator — write one
    readout block for the two keys (thinking on, the card's thinking-mode sampler). Without the demo row the demo key fell
    back to its non-thinking preset with a 'no thinking-specific vendor row' source while the same checkpoint under
    qwen35_4b rendered the thinking row (review finding, 2026-10-08)."""
    g = PRESETS["readouts"]["gsm8k"]["presets"]
    assert g["qwen35_4b_demo"] == g["qwen35_4b"]
    assert sdd._readouts_for(PRESETS, "qwen35_4b_demo") == sdd._readouts_for(PRESETS, "qwen35_4b")
    # the generator renders against the wave ROSTER, which has no demo target (the worked example is configured, not run):
    # shown on a roster that carries the demo key as the 4B's twin
    import copy
    roster = copy.deepcopy(msc.load_roster())
    roster["targets"]["qwen35_4b_demo"] = copy.deepcopy(roster["targets"]["qwen35_4b"])
    assert msc.expected_readouts(PRESETS, "qwen35_4b_demo", roster) == msc.expected_readouts(PRESETS, "qwen35_4b", roster)
    r = msc.expected_readouts(PRESETS, "qwen35_4b_demo", roster)["gsm8k"]
    assert r["thinking"] == "on" and r["preset"]["temperature"] == 1.0 and r["preset"]["top_p"] == 0.95


def test_the_reader_refuses_a_malformed_table(tmp_path):
    bad = dict(PRESETS)
    bad["presets"] = {"k": {**PRESETS["presets"]["gemma4e4b"], "preset": {"temperature": 1.0, "n": 4}}}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(sdd.DecodePresetsMalformed, match="non-sampler keys"):
        sdd.load_decode_presets(p)


# --------------------------------------------------------------------------- #
# end to end on the controls driver                                            #
# --------------------------------------------------------------------------- #


def _without_greedy(argv):
    return [a for a in argv if a != "--greedy"]


def test_greedy_opt_in_is_recorded_and_keeps_the_greedy_store(world):
    srun.main(world.argv)                       # the world opts in
    gs = json.loads((_cell_dir(world.out_root) / "gen_stats.json").read_text())
    rec = gs["decode_resolution"]
    assert rec["decode_default"] == sdd.DEFAULT_GREEDY_OPT_IN and rec["served_model"] == BASE_MODEL
    assert rec["served_revision"] == REV and rec["preset_key"] is None
    assert all(req["temperature"] == 0.0 for req in world.posted)          # greedy bodies, as ever
    for control in LEGACY_GEN_CONTROLS.split(","):
        assert "decode" not in gs[control]                                   # no sampled additions


def test_without_the_opt_in_a_model_with_no_row_refuses_before_any_store(world):
    with pytest.raises(SystemExit, match="--greedy"):
        srun.main(_without_greedy(world.argv))
    assert not (world.out_root / "cells").exists() and world.posted == []


def test_without_the_opt_in_a_model_with_a_row_draws_under_its_preset(world, monkeypatch, tmp_path):
    """The served model of the synthetic world has no vendor row; give it one (a
    temporary data file naming the world's model + revision) and the driver injects it: the
    bodies carry the preset keys and a per-item seed, the store re-keys, the sidecar is
    written, and gen_stats says how the decode was chosen."""
    table = dict(PRESETS)
    table["presets"] = {"synthetic": {"model": BASE_MODEL, "revision": REV,
                                      "preset": {"temperature": 0.6, "top_p": 0.9},
                                      "source_kind": "generation_config",
                                      "source": "synthetic row for the test world (test bytes)"}}
    # the readouts block, minus the per-key thinking map (it names llama8b, which this synthetic table lacks)
    table["readouts"] = json.loads(json.dumps(PRESETS["readouts"]))
    for k in ("thinking_by_key", "thinking_by_key_source"):
        table["readouts"]["gsm8k"].pop(k, None)
    path = tmp_path / "decode.yaml"
    path.write_text(yaml.safe_dump(table))
    monkeypatch.setattr(sdd, "DECODE_PRESETS_PATH", path)
    # a SAMPLED draw's response must carry `usage` (its at-cap status and length are read off it;
    # the driver refuses a draw without them, by design) — the greedy world's canned poster sends
    # none, so wrap it: the same bodies, plus the token counts a real server returns
    import contextlib
    inner = srun._proxy_poster

    @contextlib.asynccontextmanager
    async def poster_with_usage(proxy_url):
        async with inner(proxy_url) as post:
            async def post_u(req):
                resp = await post(req)
                if not req.get("continue_final_message") and isinstance(resp, dict):
                    resp = {**resp, "usage": {"completion_tokens": 7, "prompt_tokens": 11}}
                return resp
            yield post_u

    monkeypatch.setattr(srun, "_proxy_poster", poster_with_usage)
    srun.main(_without_greedy(world.argv))
    cell_dir = _cell_dir(world.out_root)
    gs = json.loads((cell_dir / "gen_stats.json").read_text())
    rec = gs["decode_resolution"]
    assert rec["decode_default"] == sdd.DEFAULT_VENDOR_PRESET and rec["preset_key"] == "synthetic"
    assert rec["preset"] == {"temperature": 0.6, "top_p": 0.9} and rec["presets_file"] == str(path)
    assert world.posted and all(req["temperature"] == 0.6 and req["top_p"] == 0.9 and "seed" in req
                                for req in world.posted)
    from sisteer.eval.cell_controls import controls_cap
    cap = controls_cap(yaml.safe_load(world.cfg_path.read_text()))   # the cap the driver reads
    for control in LEGACY_GEN_CONTROLS.split(","):
        assert gs[control]["decode"]["arm"] == "vendor_recommended" and gs[control]["cap"] == cap
        assert gs[control]["client_timeout_s"] == srun.AIOHTTP_TOTAL_TIMEOUT_S
    assert list((cell_dir / "private").glob("*.meta.jsonl"))                # the sidecar
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert manifest["resolved_config"]["gen"]["do_sample"] is True          # a sampled store key


# --------------------------------------------------------------------------- #
# the union file: the wave's thinking_by_key block; BoN deliberately excluded    #
# --------------------------------------------------------------------------- #


def test_thinking_by_key_block_is_the_waves_and_validates():
    """The data file is the UNION of the three copies (this branch, sdfbase-llama8b-wave, em-misalign):
    the wave's `readouts.gsm8k.thinking_by_key` block rides here verbatim so those branches' rebases are
    a no-op for the file; both readers validate it (keys are preset keys, values "on"/"off", a reason
    per key), and the per-key readout mode flows through the generator's and the drivers' renderings."""
    g = PRESETS["readouts"]["gsm8k"]
    assert g["thinking_by_key"] == {"llama8b": "off"}
    assert "no thinking mode" in g["thinking_by_key_source"]["llama8b"]
    # the per-key mode flows through BOTH renderings — shown on a roster that does NOT omit gsm8k for llama8b (the
    # wave roster does: targets.llama8b.capability_omitted, since the paper has ONE GSM8K instrument — under the real
    # roster both renderers write that omission instead, and still agree)
    import copy
    roster = msc.load_roster()
    no_omit = copy.deepcopy(roster)
    no_omit["targets"]["llama8b"].pop("capability_controls", None)
    no_omit["targets"]["llama8b"].pop("capability_omitted", None)
    assert msc.expected_readouts(PRESETS, "llama8b", no_omit)["gsm8k"]["thinking"] == "off"
    assert msc.expected_readouts(PRESETS, "qwen35_4b")["gsm8k"]["thinking"] == "on"
    # the shared builder (sampled_decode, under the sampled driver's VENDOR_CONTRACT) knows nothing of the roster: the
    # drivers apply the roster overlay capability_omissions right after the default-decode resolution, and the overlaid
    # rendering equals the generator's, on both a roster without the omission and the wave roster with it
    from sisteer.eval import capability_omissions as co
    assert sdd._readouts_for(PRESETS, "llama8b")["gsm8k"]["thinking"] == "off"
    assert co.readouts_with_omissions(sdd._readouts_for(PRESETS, "llama8b"), "llama8b", no_omit) == msc.expected_readouts(PRESETS, "llama8b", no_omit)
    omitted = msc.expected_readouts(PRESETS, "llama8b")            # the wave roster: the omission, on both sides
    assert omitted["gsm8k"]["omitted"] is True and "owned by the evaluation lane" in omitted["gsm8k"]["reason"]
    assert co.readouts_with_omissions(sdd._readouts_for(PRESETS, "llama8b"), "llama8b") == omitted \
        == co.readouts_with_omissions(sdd._readouts_for(PRESETS, "llama8b"), "llama8b", roster)
    assert co.gsm8k_omission("llama8b") == omitted["gsm8k"]["reason"] and co.gsm8k_omission("qwen35_4b") is None
    assert co.gsm8k_omission("llama8b", roster="/nonexistent/roster.yaml") is None   # no roster file: no omission
    # the driver-side path: a config with no block gets the default injected, then the overlay
    cfg0 = {"model": PRESETS["presets"]["llama8b"]["model"], "model_revision": PRESETS["presets"]["llama8b"]["revision"]}
    cfg1, rec = sdd.resolve_default_decode(cfg0, greedy=False, trial=0, static_cap=4096)
    assert rec["preset_key"] == "llama8b" and cfg1["sampled_decode"]["readouts"]["gsm8k"].get("omitted") is None
    cfg2 = co.apply_to_config(cfg1, rec["preset_key"])
    assert cfg2["sampled_decode"]["readouts"] == omitted and cfg2["sampled_decode"]["arm"] == cfg1["sampled_decode"]["arm"]
    assert co.apply_to_config(cfg0, "llama8b") == cfg0            # no block: an equal copy
    for bad, match in (({"nope": "off"}, "which has no preset"), ({"llama8b": True}, '"on" or "off"')):
        table = json.loads(json.dumps(PRESETS))
        table["readouts"]["gsm8k"]["thinking_by_key"] = bad
        with pytest.raises(sdd.DecodePresetsMalformed, match=match):
            sdd.load_decode_presets(_write_yaml(table))
    table = json.loads(json.dumps(PRESETS))
    del table["readouts"]["gsm8k"]["thinking_by_key_source"]
    with pytest.raises(sdd.DecodePresetsMalformed, match="thinking_by_key_source"):
        sdd.load_decode_presets(_write_yaml(table))


def _write_yaml(table):
    import tempfile
    p = Path(tempfile.mkdtemp(prefix="decode-presets-")) / "decode.yaml"
    p.write_text(yaml.safe_dump(table))
    return p


def test_bon_is_deliberately_outside_the_preset_default():
    """BoN draws at each served model's vendor preset too, but reads it from the
    results REGISTRY (`<model>|bon|off|vendor_preset`, same_values_as this file's static rows), never from this
    data file or its resolver: one authority for the numbers, and no path by which the yaml's default could
    reach a BoN request unannounced. The driver's independence from the yaml is pinned so nobody 'fixes' it;
    its dependence on the registry is pinned beside it."""
    src = (REPO / "scripts/sdfbase/full22/bon_served_run.py").read_text()
    for needle in ("sampled_decode", "sdfbase_sampled_decode", "sampled_decode.yaml", "resolve_default_decode",
                   "DECODE_PRESETS_PATH", "--greedy"):
        assert needle not in src, needle
    assert "bs.resolve_bon_sampler(" in src and "vendor_preset" in src      # the registry's per-model row
    header = (REPO / "configs/sdfbase_sampled_decode.yaml").read_text()
    assert "BoN (scripts/sdfbase/full22/bon_served_run.py) is DELIBERATELY EXCLUDED from THIS FILE's default" in header
    assert "`<model>|bon|off|vendor_preset`" in header
