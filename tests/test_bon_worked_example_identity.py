"""The worked example's best-of-N: its key resolves and files under ITS OWN rows on the paper's checkpoint, and the public recipe
draws the sampler of the canonical instrument it registers under (2026-10-09).

Finding 1. `reproduce` submits the worked example's 512 store through target_static.sbatch with `--model-key qwen35_4b_demo`, while the
generated config can only spell the served repo `Qwen/Qwen3.5-4B`, which the decode-policy export resolves to the PAPER's row (the paper's
checkpoint owns its spellings, vocab.model_spelling_owner). The store driver compared the two model ids and refused before a draw, so the
demo's seeded rows and instruments could not be used. Now `bon_served.served_model_identity` compares CHECKPOINTS: a key naming another
model_id of the same `hf_repo`, whose pinned `hf_revision` (exported since this change) agrees with the config's `model_revision` and the
routing table's `base_revision`, is the identity the sampler resolves for -- the demo draws and files the demo's rows; a key of a different
checkpoint, a disagreeing revision, or a table serving another checkpoint still refuses. The climb driver takes the same `--model-key`
(bon_climb.sbatch forwards the launcher's key), reads the store's own recorded key when none is given, and refuses to climb a store under
another model's registry row.

Finding 2. The planner's canonical best-of-N instrument for the demo is the paper's T=1.0 climb (`bon_512store_4096climb_T1`), but the
sbatch default `--bon-decode-preset vendor_preset` sends the Qwen vendor row (0.7 / 0.8 / 20 / 1.5), so the native registration would have
attested the T=1.0 instrument for vendor draws. The recipe now declares `F22_BON_DECODE_PRESET=canonical` on its two best-of-N units: the
driver resolves, at launch and before the proxy is touched, the registry label whose sampler reproduces the canonical instrument's stated
decode (`bon_served.canonical_bon_label` over the checkout's seeds -- the same `jobplan.canonical_instrument` the start-of-job registration
binds), refusing by name when no label does. The paper's and the demo's instrument resolves to `harness_default`; the two pipeline-test
models keep `vendor_preset` (their vendor rows send the instrument's temperature 1.0); the paper's canonical selection is untouched.
Rejected alternative, recorded: a demo-only canonical vendor-preset instrument. A new seed row with the vendor row's fields collides in
identity with the existing sensitivity instrument's paper record (dimensions.build_records refuses a cross-row collision), and binding the
demo's vendor twin inside the T=1.0 seed row would mint a record whose hashed `ladder_promotion_rule` prose names temperature 1.0 beside
vendor numbers. Everything synthetic: recording transports, invented goals, no server; the registry's committed bytes are read as data."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", REPO / "tests", REPO / "reproduce" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import bon_climb_run as bcr  # noqa: E402
import bon_served_run as br  # noqa: E402
import test_bon_vendor_preset as tv  # noqa: E402  (the store driver's end-to-end wiring: stubbed proxy, recorded fan-out)
import test_target_static_bon_argv as ta  # noqa: E402  (the sbatch launch line rendered under bash)
from sisteer.eval.attacks import bon_climb as bc  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402
from sisteer.registry import decode_policy as DP  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402

F22 = REPO / "scripts" / "sdfbase" / "full22"
SEEDS = REPO / "configs" / "registry"
EXPORT = SEEDS / "decode_policy.json"
DEMO, PAPER = "qwen35_4b_demo", "qwen35_4b_it"
HF, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GEMMA = "google/gemma-4-E4B-it"
T1_SEED = "bon_512store_4096climb_T1"
T1_REF, VENDOR_REF = f"{DEMO}|bon|off|grandfathered_T1@v1", f"{DEMO}|bon|off|vendor_preset@v1"
PIPELINE_TEST_MODELS = ("qwen35_35b_a3b", "gemma4_31b")


@pytest.fixture(scope="module")
def pf() -> DP.PolicyFile:
    return DP.PolicyFile.load(str(EXPORT))


@pytest.fixture(scope="module")
def seed_index() -> JP.SeedIndex:
    return JP.load_seed_index(str(SEEDS))


def _synthetic_pf(models: dict, spellings: dict) -> DP.PolicyFile:
    return DP.PolicyFile({"export_version": DP.EXPORT_VERSION, "policies": [], "sampling_mode_policy": [], "models": models,
                          "model_spellings": spellings, "evals": {}, "eval_spellings": {}, "header": {}})


# --------------------------------------------------------------------------------------------
# the served identity is a checkpoint comparison, with the revision exported and compared
# --------------------------------------------------------------------------------------------


def test_the_export_pins_each_models_revision_and_the_two_keys_share_the_checkpoint(pf):
    assert pf.models[DEMO]["hf_revision"] == pf.models[PAPER]["hf_revision"] == REV == pf.hf_revision(DEMO) == pf.hf_revision("qwen35_4b")
    assert all(("hf_revision" in m) for m in pf.models.values())            # every model entry carries its pin (or `unverified`)
    assert pf.model_id(HF) == PAPER and pf.model_id(DEMO) == DEMO            # the repo spelling is the paper's; the key is the demo's own
    assert bs.same_checkpoint(pf, DEMO, PAPER) and bs.same_checkpoint(pf, PAPER, DEMO) and bs.same_checkpoint(pf, DEMO, DEMO)
    assert bs.same_checkpoint(pf, DEMO, PAPER, declared_revisions=(REV, None, "unverified"))
    assert not bs.same_checkpoint(pf, DEMO, PAPER, declared_revisions=("0" * 40,))          # a declared revision that disagrees refutes it
    assert not bs.same_checkpoint(pf, PAPER, "gemma4e4b_it") and not bs.same_checkpoint(pf, DEMO, "qwen35_4b_base")
    # unknown pins compare to nothing; a repo nobody states is never shared
    sp = _synthetic_pf({"a": {"hf_repo": "org/m", "hf_revision": "unverified"}, "b": {"hf_repo": "org/m", "hf_revision": "a" * 40},
                        "c": {"hf_repo": None, "hf_revision": None}, "d": {"hf_repo": None, "hf_revision": None}}, {"org/m": "a"})
    assert bs.same_checkpoint(sp, "a", "b") and bs.same_checkpoint(sp, "a", "b", declared_revisions=("a" * 40,))
    assert not bs.same_checkpoint(sp, "a", "b", declared_revisions=("b" * 40,)) and not bs.same_checkpoint(sp, "c", "d")


def test_the_served_identity_is_the_keys_model_id_only_for_a_key_of_the_same_checkpoint(pf):
    table = {"base_model": HF, "base_revision": REV}
    # the worked example: the config can only spell the paper's repo; the key names the demo; one checkpoint -> the demo's identity
    assert bs.served_model_identity(HF, HF, DEMO, model_revision=REV, base_revision=REV, pf=pf) == DEMO
    assert bs.served_model_identity(HF, None, DEMO, pf=pf) == DEMO                      # no revision declared anywhere: the pins agree
    assert br.served_model_of({"model": HF, "model_revision": REV}, table, DEMO, tag="t") == DEMO
    assert bcr.served_model_of({"model": HF, "model_revision": REV}, table, DEMO, tag="t") == DEMO
    # every existing invocation is byte-identical: no key, or a key naming the config's own model, returns the config's spelling
    for key in (None, "qwen35_4b", PAPER, HF):
        assert bs.served_model_identity(HF, HF, key, model_revision=REV, base_revision=REV, pf=pf) == HF
        assert br.served_model_of({"model": HF}, {"base_model": HF}, key, tag="t") == HF
    assert bcr.served_model_of({"model": GEMMA}, {"base_model": GEMMA}, tag="t") == GEMMA   # the climb's old signature still parses
    # the same key is REFUSED the moment a declared revision disagrees with the key's pin, on either side
    with pytest.raises(bs.BonServedError, match="disagrees with the config's model"):
        bs.served_model_identity(HF, HF, DEMO, model_revision="0" * 40, base_revision=REV, pf=pf)
    with pytest.raises(SystemExit, match="disagrees with the config's model"):
        br.served_model_of({"model": HF, "model_revision": REV}, {"base_model": HF, "base_revision": "f" * 40}, DEMO, tag="t")
    # a key of another checkpoint, an unknown key, an unknown config model, a table serving another checkpoint: refused as before
    with pytest.raises(SystemExit, match="disagrees with the config"):
        br.served_model_of({"model": GEMMA}, {}, "llama8b", tag="t")
    with pytest.raises(SystemExit, match="disagrees with the config"):
        br.served_model_of({"model": GEMMA}, {}, "nonsuch", tag="t")
    with pytest.raises(SystemExit, match="names no served"):
        br.served_model_of({}, {}, None, tag="t")
    with pytest.raises(SystemExit, match="not a model the decode-policy export knows"):
        br.served_model_of({"model": "org/nonsuch"}, {}, None, tag="t")
    with pytest.raises(SystemExit, match="routing table serves base_model"):
        br.served_model_of({"model": GEMMA}, {"base_model": HF}, None, tag="t")
    with pytest.raises(SystemExit, match="routing table serves base_model"):
        bcr.served_model_of({"model": HF}, {"base_model": GEMMA}, DEMO, tag="t")
    # under the demo's identity the sampler is the DEMO's row (the twin's numbers), never the paper's id
    for label, ref, numbers in ((bs.BON_VENDOR_PRESET, VENDOR_REF, tv.VENDOR_SIX), (bs.BON_HARNESS_DEFAULT, T1_REF, {"temperature": 1.0})):
        s = bs.resolve_bon_sampler(None, label, DEMO)
        assert s.policy_id == ref and dict(s.decode) == numbers, (label, s)
        assert dict(bs.resolve_bon_sampler(None, label, HF).decode) == numbers                 # the paper's row states the same numbers


# --------------------------------------------------------------------------------------------
# the recipe draws the sampler of the canonical instrument it registers under
# --------------------------------------------------------------------------------------------


def _stated(payload: dict) -> dict:
    return {k: payload.get(f"decode_{k}") for k in DP.PARAM_KEYS if payload.get(f"decode_{k}") is not None}


def test_the_canonical_label_reproduces_the_planners_instrument_and_the_paper_selection_is_unchanged(seed_index):
    for mid in (PAPER, DEMO):
        payload, extra = JP.canonical_instrument(seed_index, "strongreject_bon", mid, "lora_native")
        assert extra["seed_instrument_id"] == T1_SEED and _stated(payload) == {"temperature": 1.0}, mid
        label, facts = bs.canonical_bon_label(mid, "lora_native", seeds_dir=str(SEEDS))
        assert label == bs.BON_HARNESS_DEFAULT and facts["instrument_id"] == payload["instrument_id"]
        assert facts["seed_instrument_id"] == T1_SEED and facts["decode_source"] == "grandfathered_T1" and facts["decode"]["temperature"] == 1.0
        sent = dict(bs.resolve_bon_sampler(None, label, mid).decode)
        assert {k: v for k, v in sent.items() if k in _stated(payload)} == _stated(payload)     # the stated numbers are the sent ones
        # the sbatch default would NOT have: the vendor row sends 0.7, and the instrument states 1.0 -- the mislabel, now named
        assert bs.resolve_bon_sampler(None, bs.BON_VENDOR_PRESET, mid).decode["temperature"] != 1.0
    paper_id = JP.canonical_instrument(seed_index, "strongreject_bon", PAPER, "lora_native")[0]["instrument_id"]
    demo_id = JP.canonical_instrument(seed_index, "strongreject_bon", DEMO, "lora_native")[0]["instrument_id"]
    assert paper_id == demo_id                                                                  # one record, both keys (the mirror)
    # the pipeline-test models: their vendor rows send the instrument's temperature 1.0, so the default label reproduces it
    for mid in PIPELINE_TEST_MODELS:
        label, facts = bs.canonical_bon_label(mid, "lora_native", seeds_dir=str(SEEDS))
        assert label == bs.BON_VENDOR_PRESET and facts["seed_instrument_id"] == T1_SEED, mid
    # the rule itself: the vendor preset first, the grandfathered row second, a refusal by name when neither reproduces the numbers
    assert bs.label_for_decode(DEMO, {"temperature": 1.0}) == bs.BON_HARNESS_DEFAULT
    assert bs.label_for_decode(DEMO, {"temperature": 0.7, "presence_penalty": 1.5}) == bs.BON_VENDOR_PRESET
    assert bs.label_for_decode("gemma4_31b", {"temperature": 1.0, "top_p": None}) == bs.BON_VENDOR_PRESET
    with pytest.raises(bs.BonServedError, match="no registry best-of-N label reproduces"):
        bs.label_for_decode(DEMO, {"temperature": 0.3})
    with pytest.raises(bs.BonServedError, match="states no decode number"):
        bs.label_for_decode(DEMO, {k: None for k in DP.PARAM_KEYS})
    with pytest.raises(bs.BonServedError, match="canonical best-of-N"):
        bs.canonical_bon_label("org/nonsuch", "lora_native", seeds_dir=str(SEEDS))


def _demo_argv(tmp_path, rt, **over):
    base = {"--model-key": DEMO, "--bon-decode-preset": br.BON_DECODE_PRESET_CANONICAL, "--decode-seed-policy": None, "--decode-seed-base": None}
    base.update(over)
    return tv._argv(tmp_path, rt, **base)


def _with_base_model(monkeypatch):
    from sisteer.capeval.serving import proxy as px
    stubbed = px.load_routing_table                     # _wire_main's stub: the synthetic routes; the table here also names the served checkpoint
    monkeypatch.setattr(px, "load_routing_table", lambda p: {**stubbed(p), "base_model": HF, "base_revision": REV})


def test_the_store_driver_draws_the_demo_under_its_key_at_the_canonical_instruments_sampler(monkeypatch, tmp_path, seed_index):
    monkeypatch.setenv(DP.ENV_VAR, str(EXPORT))
    rt, got = tv._wire_main(monkeypatch, tmp_path)
    _with_base_model(monkeypatch)
    assert br.main(_demo_argv(tmp_path, rt)) == 0
    rec = json.loads((Path(got["out_dir"]) / bc.BON_STORE_MANIFEST).read_text())
    payload, extra = JP.canonical_instrument(seed_index, "strongreject_bon", DEMO, "lora_native")
    assert rec["served_model"] == DEMO == got["model"] and rec["model_key"] == DEMO
    assert rec["bon_decode_preset"] == bs.BON_HARNESS_DEFAULT == got["decode_label"] and rec["bon_decode_preset_declared"] == "canonical"
    assert rec["decode"] == {"temperature": 1.0} == _stated(payload) and rec["decode_policy_id"] == T1_REF and got["decode"] is None
    assert rec["canonical_instrument"]["instrument_id"] == payload["instrument_id"] and rec["canonical_instrument"]["seed_instrument_id"] == T1_SEED
    assert rec["canonical_instrument"]["decode_source"] == payload["decode_source"] == "grandfathered_T1"
    assert rec["canonical_instrument"]["serving_path"] == "lora_native" and rec["canonical_instrument"]["seeds_revision"] == seed_index.seeds_revision
    # the explicit labels resolve under the demo's identity too, and add no key to the record (every existing record byte-identical)
    got.clear()
    assert br.main(_demo_argv(tmp_path / "v", rt, **{"--bon-decode-preset": "vendor_preset"})) == 0
    rec_v = json.loads((Path(got["out_dir"]) / bc.BON_STORE_MANIFEST).read_text())
    assert rec_v["served_model"] == DEMO and rec_v["decode_policy_id"] == VENDOR_REF and rec_v["decode"] == tv.VENDOR_SIX
    assert "canonical_instrument" not in rec_v and "bon_decode_preset_declared" not in rec_v
    # a key of a different checkpoint still refuses before the proxy client is built
    tv._StubClient.made = []
    with pytest.raises(SystemExit, match="disagrees with the config's model"):
        br.main(_demo_argv(tmp_path / "x", rt, **{"--model-key": "llama8b"}))
    assert tv._StubClient.made == []
    # `canonical` is a marker in the preset table, resolved per served model at launch; nothing is read at import
    assert br.BON_DECODE_PRESETS[br.BON_DECODE_PRESET_CANONICAL] == br.BON_DECODE_FROM_CANONICAL == "canonical_instrument"
    assert br.resolve_bon_decode(br.BON_DECODE_PRESET_CANONICAL, None) == (None, None)
    assert "canonical" in br.build_parser().parse_args(["--config", "c", "--defense", "d", "--private-root", "r", "--proxy-url", "u",
                                                        "--routing-table", "t", "--max-concurrency", "4", "--bon-decode-preset", "canonical"]).bon_decode_preset


def test_the_climb_driver_carries_the_stores_key_and_refuses_another_models_row(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    assert bcr.climb_model_key(None, None, tag="t") is None and bcr.climb_model_key(DEMO, None, tag="t") == DEMO
    assert bcr.climb_model_key(None, store, tag="t") is None                        # no store record yet: nothing to read
    (store / bc.BON_STORE_MANIFEST).write_text(json.dumps({"model_key": DEMO, "served_model": DEMO, "complete": True}))
    assert bcr.climb_model_key(None, store, tag="t") == DEMO                        # the store's own declaration
    assert bcr.climb_model_key(DEMO, store, tag="t") == DEMO
    with pytest.raises(SystemExit, match="drawn under --model-key"):
        bcr.climb_model_key("qwen35_4b", store, tag="t")                            # another model's key over the demo's store
    # the climb re-draws at the store's recorded sampler, resolved for the store's model: the demo's row, by id
    rec0 = {"decode": {"temperature": 1.0}, "bon_decode_preset": bs.BON_HARNESS_DEFAULT, "decode_policy_id": T1_REF}
    s = bcr.climb_sampler(rec0, DEMO, tag="t")
    assert s.policy_id == T1_REF and dict(s.decode) == {"temperature": 1.0} and s.label == bs.BON_HARNESS_DEFAULT
    with pytest.raises(SystemExit, match="record registry row"):
        bcr.climb_sampler(rec0, HF, tag="t")                                        # the paper's id would be filed on the demo's climb
    legacy = bcr.climb_sampler({"decode": None, "bon_decode_preset": bc.LEGACY_STORE_PRESET, "decode_policy_id": None}, DEMO, tag="t")
    assert legacy.policy_id == T1_REF
    explicit = bcr.climb_sampler({"decode": {"temperature": 1.0, "top_p": 1.0}, "bon_decode_preset": "anchor_match", "decode_policy_id": None}, DEMO, tag="t")
    assert explicit.policy_id is None and explicit.label == "anchor_match"
    with pytest.raises(SystemExit, match="refusing to climb it under another"):
        bcr.climb_sampler({"decode": {"temperature": 0.3}, "bon_decode_preset": bs.BON_HARNESS_DEFAULT, "decode_policy_id": None}, DEMO, tag="t")
    ns = bcr.build_parser().parse_args(["--config", "c", "--defense", "d", "--shard", "0", "--nshards", "4", "--out", "o", "--proxy-url", "u",
                                        "--routing-table", "t", "--max-concurrency", "2", "--model-key", DEMO, "--bon-decode-preset", "canonical"])
    assert ns.model_key == DEMO and ns.bon_decode_preset == "canonical"


# --------------------------------------------------------------------------------------------
# the recipe: both best-of-N units declare `canonical`; both sbatch lines forward it and the key to the drivers
# --------------------------------------------------------------------------------------------


def test_the_recipe_declares_the_canonical_preset_on_both_best_of_n_units(monkeypatch):
    monkeypatch.setenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", "1")        # the plan of a steered selection reads generated artifacts a worktree lacks
    import stage as ST
    assert ST.BON_DECODE_PRESET == br.BON_DECODE_PRESET_CANONICAL and ST.BON_DECODE_PRESET_ENV == "F22_BON_DECODE_PRESET"
    rows = ST.unit_plan(DEMO, "battery", families=["static", "bon"], sha="a" * 40)   # the 512 store is placed by the static stage family
    by_unit = {r["unit"]: r for r in rows if r["family"] == "bon"}
    assert set(by_unit) == {"store512", "climb"}
    assert by_unit["store512"]["script"].endswith("target_static.sbatch") and by_unit["climb"]["script"].endswith("bon_climb.sbatch")
    for r in by_unit.values():
        assert r["env"]["F22_BON_DECODE_PRESET"] == "canonical" and r["env"]["F22_REGISTRY_EVALS"] == "strongreject_bon", r["unit"]
    others = [r for r in ST.unit_plan(DEMO, "battery", families=["static", "controls", "mmlu"], sha="a" * 40) if r["family"] != "bon"]
    assert others and all("F22_BON_DECODE_PRESET" not in r["env"] for r in others), [r["unit"] for r in others]   # no other unit declares one


def test_the_store_sbatch_forwards_the_canonical_preset_and_the_drivers_parser_accepts_it():
    src = (F22 / "target_static.sbatch").read_text()
    snippet = ta._snippet(src, 'f22_queue_launch "bon__${cell}"')
    assert ta._render(snippet, {}) == ta.GOLDEN_WAVE                                                  # unset: byte for byte as before
    rendered = ta._render(snippet, {"F22_BON_DECODE_PRESET": "canonical"})
    assert rendered == ta.GOLDEN_WAVE[:-4] + ["--bon-decode-preset", "canonical", "--model-key", "qwen35_4b"]
    assert br.build_parser().parse_args(rendered[2:]).bon_decode_preset == "canonical"


CLIMB_VARS = ('MPY=/venv/bin/python; CFG=configs/sdfbase_full22_qwen35_4b_demo_battery.yaml; cell=constitution_lora; PRIVATE_ROOT=/w/transcripts_results; '
              'MANIFEST_JSONL=/w/bon_climb/t/plan/climb_manifest.jsonl; SHARD=1; NSHARDS=4; TAG=t; OUT_ROOT=/w/bon_climb/t/out; '
              'PROXY_URL=http://127.0.0.1:4321/v1; TABLE=/w/serving/routing_table.json; TARGET_L=4864; CONC=8; CAP=4096; TARGET_ENGINE=lora_native; '
              'SERVE_DIR=/w/serving; KEY=qwen35_4b_demo; MODE=climb; ')


def _render_climb(src: str, env: dict) -> list:
    """The climb launch line rendered under bash with the sbatch's own preset gate (the `case` on F22_BON_DECODE_PRESET) in front of it."""
    gate = re.search(r'^BON_CLIMB_PRESET=""\n(?:.*\n)*?.*esac\n', src, flags=re.M)
    assert gate, "the preset gate is not in the sbatch"
    launch = ta._snippet(src, 'f22_queue_launch "bonclimb__${cell}"')
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env.items())
    r = subprocess.run(["bash", "-c", "set -u; " + CLIMB_VARS + ta.STUBS + exports + gate.group(0) + launch],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def test_the_climb_sbatch_forwards_the_key_and_only_a_label_the_climb_driver_accepts():
    src = (F22 / "bon_climb.sbatch").read_text()
    assert subprocess.run(["bash", "-n", str(F22 / "bon_climb.sbatch")], capture_output=True).returncode == 0
    unset = _render_climb(src, {})
    assert unset[-2:] == ["--model-key", "qwen35_4b_demo"] and "--bon-decode-preset" not in unset
    ns = bcr.build_parser().parse_args(unset[2:])
    assert ns.model_key == "qwen35_4b_demo" and ns.bon_decode_preset == bs.BON_VENDOR_PRESET
    canonical = _render_climb(src, {"F22_BON_DECODE_PRESET": "canonical"})
    assert canonical == unset + ["--bon-decode-preset", "canonical"]
    assert bcr.build_parser().parse_args(canonical[2:]).bon_decode_preset == "canonical"
    assert _render_climb(src, {"F22_BON_DECODE_PRESET": "harness_default"}) == unset + ["--bon-decode-preset", "harness_default"]
    # the seed-replicate declarations are the store driver's explicit blocks, not the climb's: not forwarded (the climb names no block)
    for declared in ("anchor_match", "vendor"):
        assert _render_climb(src, {"F22_BON_DECODE_PRESET": declared}) == unset
    assert not any("--export=" in l for l in src.splitlines() if not l.lstrip().startswith("#"))
