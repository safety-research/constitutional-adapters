"""The served driver states the request sampler of every legacy control it runs (decode-recording lane, section 1.3 (a''), 2026-10-13):
``manifest["request_samplers"]`` -- top-level and UNHASHED, keyed by producer spelling, holding exactly the sampler keys the request body
names: ``mmlu`` the one scoring object ``MMLU_SCORING_SAMPLER`` (spread into ``_mmlu_scoring_payload``'s body -- one object, so the
manifest and the request cannot be two numbers), a sampled over-refusal control ``_decode_of(gen)``, ``{}`` under ``--greedy``. The store
identity is byte-identical: ``resolved_config`` carries no such key, ``harness_config_hash`` is its hash (the pre-change goldens of
``tests/fixtures/tinycap_identity_v1`` / ``tinycap_row_engine`` pin the chash across the change); a LEGACY manifest written without the
block RESUMES under the new driver onto the same cell dir and GAINS the block through ``write_cell_manifest``'s whole-document rewrite.
The world is
``test_served_capability``'s offline one; the sampled world is ``test_sampled_default``'s temporary vendor row."""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import served_capability_run as srun  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.manifest import canonical_json  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.eval import sampled_decode as sdd  # noqa: E402
from sisteer.eval.sampled_controls import _decode_of  # noqa: E402
from sisteer.registry import decode_evidence as DE  # noqa: E402
from test_sampled_default import PRESETS, _without_greedy  # noqa: E402
from test_served_capability import BASE_MODEL, LEGACY_GEN_CONTROLS, MMLU_ROWS, REV, _cell_dir, _mmlu_argv, world  # noqa: E402,F401


def _manifest(world):
    cell_dir = _cell_dir(world.out_root)
    return cell_dir, json.loads((cell_dir / "manifest.json").read_text())


def test_the_constant_is_the_object_the_scoring_body_spreads():
    assert srun.MMLU_SCORING_SAMPLER == {"temperature": 0.0}
    body = srun._mmlu_scoring_payload("vm", "Q", system=None, top_logprobs=64, response_prefill=False)
    assert body["temperature"] is srun.MMLU_SCORING_SAMPLER["temperature"] and body["max_tokens"] == 1 and body["seed"] == 0
    # the block builder: keyed by the legacy controls requested, exactly the sampler keys the request names
    gen = {"temperature": 0.6, "top_p": 0.9, "do_sample": True, "max_new_tokens": 512, "arm": "vendor_recommended", "seed_base": 1, "trial": 0, "batch_size": 8}
    assert srun.request_samplers_block(["mmlu", "xstest", "or_bench_hard_1k", "gsm8k", "humaneval_plus"], gen) == {"mmlu": {"temperature": 0.0}, "xstest": {"temperature": 0.6, "top_p": 0.9}, "or_bench_hard_1k": {"temperature": 0.6, "top_p": 0.9}}
    # the greedy opt-in (Major 3): the greedy body SENDS temperature 0.0 -- GREEDY_CHAT_SAMPLER is the object it spreads
    # and the block states it; `{}` would attest a request shape the body never had
    assert srun.GREEDY_CHAT_SAMPLER == {"temperature": 0.0} == DE.GREEDY_CHAT_SAMPLER
    greedy = srun._chat_payload("vm", "Q", system=None, max_tokens=512)
    assert greedy["temperature"] is srun.GREEDY_CHAT_SAMPLER["temperature"] and greedy["seed"] == 0 and greedy["max_tokens"] == 512
    assert {k: greedy[k] for k in srun.SAMPLED_DECODE_SAMPLER_KEYS if k in greedy} == srun.GREEDY_CHAT_SAMPLER
    assert srun.request_samplers_block(["xstest"], {"do_sample": False, "max_new_tokens": 512}) == {"xstest": {"temperature": 0.0}}
    assert srun.request_samplers_block(["xstest"], {"do_sample": False})["xstest"] is not srun.GREEDY_CHAT_SAMPLER
    assert srun.request_samplers_block(["gsm8k"], gen) == {} and srun.request_samplers_block(["mmlu"], {"do_sample": False}) == {"mmlu": srun.MMLU_SCORING_SAMPLER}
    assert srun.request_samplers_block(["mmlu"], gen)["mmlu"] is not srun.MMLU_SCORING_SAMPLER       # a copy on the manifest, never the object itself
    assert tc.MANIFEST_KEY_REQUEST_SAMPLERS == DE.MANIFEST_KEY_REQUEST_SAMPLERS == "request_samplers" and "MANIFEST_KEY_REQUEST_SAMPLERS" in tc.__all__


def test_the_greedy_controls_manifest_states_the_greedy_constant_every_posted_body_sends_it_and_the_identity_is_untouched(world):
    """Review round 1, Major 3: under `--greedy` every judged-control body the driver posts carries temperature 0.0 and no other sampler
    key, so the manifest states {temperature: 0.0} per control (round 0 stated `{}`, a request shape the body never had) and the registry's
    reader files it from the manifest; the store identity is byte-identical."""
    srun.main(world.argv)                                            # --controls xstest,or_bench_hard_1k --greedy
    cell_dir, manifest = _manifest(world)
    assert manifest[tc.MANIFEST_KEY_REQUEST_SAMPLERS] == {"xstest": {"temperature": 0.0}, "or_bench_hard_1k": {"temperature": 0.0}}
    resolved = manifest["resolved_config"]
    assert resolved["gen"]["do_sample"] is False
    assert "request_sampler" not in canonical_json(resolved) and "request_sampler" not in canonical_json(manifest["row_constants"])
    assert cs.harness_config_hash(resolved) == manifest["harness_config_hash"] and manifest["harness_config_hash"][:8] in cell_dir.name
    # the posted bodies: temperature 0.0 on every one, no other sampler key, the judged legs seeded 0 -- the statement is the request
    bodies = [r for r in world.posted if not r.get("continue_final_message")]
    assert bodies and all(r["temperature"] == 0.0 for r in bodies)
    assert not any(k in r for r in bodies for k in srun.SAMPLED_DECODE_SAMPLER_KEYS if k != "temperature")
    assert any(r.get("seed") == 0 and r.get("max_tokens") == resolved["gen"]["max_new_tokens"] for r in bodies)      # the judged legs: seed 0 at the controls cap
    # the registry's reader over this manifest, per control: the greedy constant, from the manifest's statement, the typed column beside it
    for sp in ("xstest", "or_bench_hard_1k"):
        ev = DE.from_store_manifest(manifest, sp, store_dir=str(cell_dir))
        assert ev.recorded() and ev.sent == {"temperature": 0.0} and ev.source == f"{cell_dir / 'manifest.json'}#request_samplers.{sp}"
        assert ev.sha256 == DE.sha256_file(str(cell_dir / "manifest.json")) and ev.witnesses["gen_do_sample"] is False
        assert ev.columns() == {"decode_sent_json": {"temperature": 0.0}, "decode_temperature_sent": 0.0}


def test_the_mmlu_manifest_states_the_scoring_constant_and_every_posted_request_sends_it(world):
    srun.main(_mmlu_argv(world))
    cell_dir, manifest = _manifest(world)
    assert manifest["controls"] == ["mmlu"] and manifest[tc.MANIFEST_KEY_REQUEST_SAMPLERS] == {"mmlu": {"temperature": 0.0}}
    assert "request_sampler" not in canonical_json(manifest["resolved_config"]) and cs.harness_config_hash(manifest["resolved_config"]) == manifest["harness_config_hash"]
    scoring = [r for r in world.posted if r.get("continue_final_message")]
    assert len(scoring) == len(MMLU_ROWS) and all(r["temperature"] == srun.MMLU_SCORING_SAMPLER["temperature"] for r in scoring)
    ev = DE.from_store_manifest(manifest, "mmlu", store_dir=str(cell_dir))
    assert ev.recorded() and ev.sent == {"temperature": 0.0} and ev.source == f"{cell_dir / 'manifest.json'}#request_samplers.mmlu"
    assert ev.columns() == {"decode_sent_json": {"temperature": 0.0}, "decode_temperature_sent": 0.0}


def test_a_legacy_mmlu_manifest_resumes_onto_the_same_cell_and_gains_the_block_in_place(world):
    """The login host's shape, synthetic: an mmlu cell written by a producer before this change (no `request_samplers`) resumed under the new driver --
    `validate_resume` passes (the block is top-level, never an identity field), the cell key is unchanged, `rows.jsonl` is untouched (every
    item already filed), and the rewritten manifest carries the block."""
    srun.main(_mmlu_argv(world))
    cell_dir, manifest = _manifest(world)
    rows_before = (cell_dir / "rows.jsonl").read_bytes()
    chash_before = manifest["harness_config_hash"]
    legacy = {k: v for k, v in manifest.items() if k != tc.MANIFEST_KEY_REQUEST_SAMPLERS}
    (cell_dir / "manifest.json").write_text(json.dumps(legacy, indent=2, sort_keys=True))
    assert tc.MANIFEST_KEY_REQUEST_SAMPLERS not in json.loads((cell_dir / "manifest.json").read_text())
    world.posted.clear()
    srun.main(_mmlu_argv(world))                                      # the resume
    cell_dir2, manifest2 = _manifest(world)
    assert cell_dir2 == cell_dir and manifest2["harness_config_hash"] == chash_before and manifest2["intervention_hash"] == manifest["intervention_hash"]
    assert manifest2[tc.MANIFEST_KEY_REQUEST_SAMPLERS] == {"mmlu": {"temperature": 0.0}}
    assert (cell_dir / "rows.jsonl").read_bytes() == rows_before
    assert not [r for r in world.posted if r.get("continue_final_message")]      # every item on file: no scoring request re-sent
    assert {k: v for k, v in manifest2.items() if k not in (tc.MANIFEST_KEY_REQUEST_SAMPLERS, tc.MANIFEST_KEY_RESUME_ATTEMPTS, "request_evidence")} == \
           {k: v for k, v in legacy.items() if k not in (tc.MANIFEST_KEY_RESUME_ATTEMPTS, "request_evidence")}


def test_a_sampled_controls_manifest_states_decode_of_gen_per_sampled_control(world, monkeypatch, tmp_path):
    """The vendor-preset default (test_sampled_default's temporary row for the world's model): the manifest's block carries `_decode_of(gen)`
    for each sampled control, equal to the keys every posted body sent, and the registry's reader files them from the block first."""
    table = dict(PRESETS)
    table["presets"] = {"synthetic": {"model": BASE_MODEL, "revision": REV, "preset": {"temperature": 0.6, "top_p": 0.9},
                                      "source_kind": "generation_config", "source": "synthetic row for the test world (test bytes)"}}
    table["readouts"] = json.loads(json.dumps(PRESETS["readouts"]))
    for k in ("thinking_by_key", "thinking_by_key_source"):
        table["readouts"]["gsm8k"].pop(k, None)
    path = tmp_path / "decode.yaml"; path.write_text(yaml.safe_dump(table))
    monkeypatch.setattr(sdd, "DECODE_PRESETS_PATH", path)
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
    cell_dir, manifest = _manifest(world)
    gen = manifest["resolved_config"]["gen"]
    assert gen["do_sample"] is True and _decode_of(gen) == {"temperature": 0.6, "top_p": 0.9}
    assert manifest[tc.MANIFEST_KEY_REQUEST_SAMPLERS] == {"xstest": {"temperature": 0.6, "top_p": 0.9}, "or_bench_hard_1k": {"temperature": 0.6, "top_p": 0.9}}
    assert world.posted and all(req["temperature"] == 0.6 and req["top_p"] == 0.9 for req in world.posted)
    assert cs.harness_config_hash(manifest["resolved_config"]) == manifest["harness_config_hash"] and "request_sampler" not in canonical_json(manifest["resolved_config"])
    for sp in ("xstest", "or_bench_hard_1k"):
        ev = DE.from_store_manifest(manifest, sp, store_dir=str(cell_dir))
        assert ev.recorded() and ev.sent == {"temperature": 0.6, "top_p": 0.9} and ev.source.endswith(f"manifest.json#request_samplers.{sp}")
        assert ev.witnesses["gen_stats_decode_agrees"] is True and ev.witnesses["request_samplers_present"] is True
