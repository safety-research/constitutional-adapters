"""full22 SAMPLED re-run — the configs, sbatch and submit printer + the instruct-direct views and the 16384 rung
configs + the strength frontier's per-dose views (2026-09-10). Synthetic data only: placeholder shas, tmp
paths, a stub ssh. Pins: the 405 generated configs (66 TAGs x 2 trials + the 3 seed-replicate TAGs x 1 trial, each x base / cap8192 /
cap16384 — one trial per seed view, the three seeds are the replication) are byte-identical to the
generator and the 16 main-view files the live jobs run are byte-identical to their frozen shas; every sampled config is its
wave config AS DATA except the permitted keys; the presets equal configs/sdfbase_sampled_decode.yaml and the pinned table
(DESIGN.md + the 2026-09-09 amendments; llama8b's row is origin/main's, byte for byte); only Qwen and GLM carry a
penalty key; every rung config is chained from the rung before it; the scope is ONE list (the generator's SAMPLED_VIEWS: the
shell gate asks it, the printer / grader / grade planner import it) and an out-of-scope view still refuses everywhere — llama8b's
twelve dose views are IN it and its `main` / `it` views are OUT of it, 2026-09-13; the dose
views run the campaign's ladder (no per-TAG base
cap); the sbatch carries the wave's rules, never the forbidden flags, and takes the view as its 6th
positional; the ladder fold applies the exit rule; the submit printer routes / splits / holds as documented, plans every
non-main TAG as its own unit on cluster B, and runs nothing."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))

import f22_roster as fr  # noqa: E402
import make_sampled_configs as msc  # noqa: E402
import submit_sampled as ss  # noqa: E402
import write_ladder as wl  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_site as _cs  # noqa: E402

# Every printed command takes its cluster's login host, eval-repo root, log dir and QoS from the site config: one synthetic
# two-cluster site for the module (site labels of its own), and the values the printers must read from it.
import tempfile  # noqa: E402
_SITE_ROOT = Path(tempfile.mkdtemp(prefix="sampled-site-"))
_SITE_CFG = _cs.write_site(_SITE_ROOT)
A, B = _cs.expect(_SITE_ROOT, "a"), _cs.expect(_SITE_ROOT, "b")


@pytest.fixture(autouse=True)
def _cluster_site(monkeypatch):
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))

F22 = REPO / "scripts/sdfbase/full22"
SAMPLED = F22 / "sampled"
PRESETS = msc.load_presets()
ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
DESIGN = (SAMPLED / "DESIGN.md").read_text()
KEYS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash")
IT_TAGS = ("gemma4e4b_it", "qwen35_4b_it")           # the instruct-direct views (lane F): TAG = <key>_it
#: llama8b's four TAGs (2026-09-10 / 2026-09-12): its main view (the bare key), its instruct-direct view, the
#: FULL-constitution appendix view and the SHORT constitution-SI arm's view — a group of their own in submit_sampled.py
LLAMA_TAGS = ("llama8b", "llama8b_it", "llama8b_fullconst", "llama8b_shortconst")
#: the strength frontier's per-dose views (2026-09-10; the 24-view list): `d<dose>` with
#: 0.25 -> 0p25, 1.5 -> 1p5, -1 -> m1, `d1` the unscaled same-pass anchor; the `it_` prefix marks the instruct-direct lane
DOSE_TOKENS = ("m1", "0p1", "0p25", "0p5", "1", "1p5")
#: gemma / qwen carry BOTH ladder families since 2026-09-15: `it_d*` (instruct-direct, batch-64 refit lineage) and bare
#: `d*` (the paper's BASE-FIT objects transferred onto the instruct target — constitution, control, subtraction)
#: ... and gemma4e4b's base->instruct ladder is EXTENDED to doses 2 and 4 (2026-09-16: every dose from -1 to 1.5
#: read flat on that lane) — two more bare TAGs on gemma only, after its d1p5
HIGH_DOSE_TOKENS = {"gemma4e4b": ("2", "4")}
#: ... and qwen35_4b's ReLU (PSR) DIFFERENCE ladder: six `d<dose>_relu` TAGs — the difference ReLU
#: object served base->instruct at the same six doses, its own view namespace (`object_family: relu`) and its own dose UNIT
#: (the direction norm, not the LoRA write-basis multiplier) — in the scope beside the LoRA ladders, after qwen's bare d*
#: (make_sampled_configs.RELU_DOSE_VIEWS)
RELU_DOSE_TAGS = tuple(f"qwen35_4b_d{t}_relu" for t in DOSE_TOKENS)
DOSE_TAGS = (tuple(f"{k}_{fam}d{t}" for k in ("gemma4e4b", "qwen35_4b") for fam in ("it_", "") for t in DOSE_TOKENS)
             + tuple(f"gemma4e4b_d{t}" for t in HIGH_DOSE_TOKENS["gemma4e4b"])
             + RELU_DOSE_TAGS
             + tuple(f"{k}_d{t}" for k in ("nemotron3nano4b", "glm46vflash") for t in DOSE_TOKENS))
#: the twelve llama8b dose TAGs (2026-09-13): BOTH ladder families on the ONE served instruct model — `d<dose>`
#: (the base-family LoRA transferred onto the instruct target) and `it_d<dose>` (the instruct-direct LoRA); `d1` / `it_d1`
#: are the ladder's own byte-identical re-serve anchors. Since 2026-09-14 llama8b's `main` / `it` / `fullconst` /
#: `shortconst` TAGs are in the same scope (LLAMA_TAGS), so llama8b is a served model
#: whose bare key IS a TAG here.
LLAMA_DOSE_TAGS = tuple(f"llama8b_d{t}" for t in DOSE_TOKENS) + tuple(f"llama8b_it_d{t}" for t in DOSE_TOKENS)
ALL_DOSE_TAGS = DOSE_TAGS + LLAMA_DOSE_TAGS          # every per-dose TAG in the scope (56: 48 + gemma's d2 / d4 + qwen's six relu)
#: qwen-base-seed-replicates: the three seed-replicate views of qwen35_4b (promoted into the scope 2026-09-15 with their roster views)
SEEDREP_TAGS = ("qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3")
LLAMA_CELLS = ["none", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]   # every llama8b dose view
#: every TAG in the scope, in the generator's order: the 6 main/it TAGs + the 38 LoRA dose views (the 24, the twelve
#: base->instruct views on gemma / qwen, gemma's d2 / d4) + qwen's 6 relu dose views + qwen's 3 seed-replicate views + the
#: 4 llama8b view TAGs + llama8b's 12 dose views (69), read from the generator rather than spelled out — a narrower literal
#: would silently drop a view from every scope assertion here.
TAGS = tuple(msc.sampled_tags())
#: the FRONT DOOR's TAGs the production roster carries today (paper-companion-repo: none until 05_stage_roster.sh --step roster lands the test
#: model's block; msc.FRONT_DOOR_VIEWS through the roster) — every count below is the paper's 69 / 405 PLUS these, so stage 05 on the real
#: site does not turn this suite red (the fix-1 owner follow-up 10); PAPER_TAGS is the paper's scope byte for byte
FRONT_DOOR_TAGS = tuple(t for t in TAGS if msc.split_tag(t)[0] in msc.FRONT_DOOR_VIEWS)
PAPER_TAGS = tuple(t for t in TAGS if t not in FRONT_DOOR_TAGS)
#: per TAG its trial count (the campaign's 2; the three seed-replicate TAGs 1 — `n_trials_by_view`) x (base, cap8192, cap16384) = 405
N_OUTPUTS = sum(msc.n_trials_of_tag(PRESETS, tag) * 3 for tag in TAGS)
N_FRONT_DOOR_OUTPUTS = sum(msc.n_trials_of_tag(PRESETS, tag) * 3 for tag in FRONT_DOOR_TAGS)
#: every preset ROW in the data file: the four generated targets + llama8b (the sdfbase-llama8b wave's row,
#: added 2026-09-11 so the drivers' vendor-preset default covers Llama-3.1-8B-Instruct; its roster target and
#: generated configs land with that wave, so it is a row here, not a TAG). Orthogonal to TAGS: it is the
#: preset DATA FILE's row set, which is why main's addition survives this merge unchanged.
ALL_KEYS = KEYS + ("llama8b",)
#: lane F's RECORD (wave_manifest_sampled.json it_views.cells): the cells the sampled it-view wave RAN -- the pre-itdoc64 five; the
#: record moves when the lane re-runs the two new cells, not when the roster gains them
IT_CELLS_LANE_F_RECORD = ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "prompted__safety_refusal", "prompted__si_constitution"]
#: the batch-64 instruct-direct lane's two cells -- the control LoRA fit and the LoRA subtraction -- ADDED beside the frozen itdoc__
#: objects on the `it` views (qwen35_4b 2026-09-13, gemma4e4b the same day; mixing batch-8 and batch-64 is allowed for now).
#: Since then the it view's explicit_cells are not one hard-coded list but each KEY's roster (it_cells).
ITDOC64_CELLS = {"constv2ctl2-full__lora_r4_ce": "itdoc64__constv2ctl2-full__lora_r4_ce",
                 "constv2-mix-full__lora_r4_ce_minus_ctl": "derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl"}
ITDOC64_KEYS = ("qwen35_4b", "gemma4e4b")            # the keys whose `it` view carries the pair
#: the pair's object pins (configs/sdfbase_full22_pins.json, per KEY -- each target's re-fit is its own object): the generated it
#: configs' headers carry them the way the base pins are
ITDOC64_OBJECT_SHA256 = {
    "qwen35_4b": {"itdoc64__constv2ctl2-full__lora_r4_ce": "bace45e26a6d1dc262e6fbb63851f7546b1327d7561cc3744e2bcf2adf881415",
                  "derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl": "5e9b0009eb24eaf66842b4a2a9f647fced95699c7aae748eaab5333b05a4c6ce"},
    "gemma4e4b": {"itdoc64__constv2ctl2-full__lora_r4_ce": "04a90de8edd863345bf182adad717801b391096b350253c72bcb077b97885ed8",
                  "derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl": "8b7d10301a1fd04fbe4c71f21ae8912bae60ffc56df34acd10fa07a960b2080e"},
}
PINS = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())


def it_cells(key: str) -> list:
    """The instruct-direct view's explicit_cells for KEY: none + the roster's served `it` cells, in roster order (the ONE source)."""
    return ["none"] + list(fr.served_cells(ROSTER, key, "it"))
RUNGS = (8192, 16384)
SHA = "ab" * 20
PY = sys.executable

#: THE PINNED PRESETS — DESIGN.md's table for gemma / qwen; the third amendment (2026-09-09, from the vendor
#: recommendations) for nemotron (top_p 1.0, not the shipped 0.95) and glm (+ repetition_penalty 1.1)
PINNED = {
    "gemma4e4b": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "qwen35_4b": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    "nemotron3nano4b": {"temperature": 1.0, "top_p": 1.0},
    "glm46vflash": {"temperature": 0.8, "top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1},
    # llama8b: the shipped generation_config.json at the pinned revision (== contracts._LLAMA_CARD)
    "llama8b": {"temperature": 0.6, "top_p": 0.9},
    # the reproduction pipeline's worked example (its TAGs enter the scope once the roster carries a view): it serves the paper's
    # Qwen3.5-4B checkpoint under its own key, so its row is the qwen35_4b row's, verbatim (the model card's settings; the
    # checkpoint ships no generation_config.json)
    "qwen35_4b_demo": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
}
SOURCE_KIND = {"gemma4e4b": "generation_config", "qwen35_4b": "model_card",
               "nemotron3nano4b": "vendor_recommendation", "glm46vflash": "model_card",
               "llama8b": "generation_config"}
#: DESIGN.md's source cells, verbatim, that each preset's `source` string must carry (gemma / qwen unchanged);
#: the amendment's quotes for nemotron / glm
SOURCE_MUST_CONTAIN = {
    "gemma4e4b": ["generation_config.json (do_sample: true); no penalty term"],
    "qwen35_4b": ["NO generation_config.json ships (.no_exist marker)", 'model card README "Best Practices"',
                  "instruct/non-thinking mode, GENERAL tasks"],
    "nemotron3nano4b": ["the shipped generation_config.json @ dfaf35de carries top_p 0.95",
                        '"temperature=1.0 and top_p=0.95 are recommended for reasoning tasks, while temperature=0.6 and top_p=0.95 are recommended for tool calling"',
                        "shows top_p 1.0 only in a curl example", "T=1.0, top_p=1.0"],
    "glm46vflash": ['"Evaluation Settings"', "top_p: 0.6, top_k: 2, temperature: 0.8, repetition_penalty: 1.1, max_generate_tokens: 16K",
                    "shipped generation_config.json @ 411bb4d7 which carries the first three", "16K", "8192"],
    "llama8b": ["generation_config.json (do_sample: true; temperature 0.6, top_p 0.9)", "no top_k, no penalty term", "d10aef79"],
}
PENALTY_KEYS = {"presence_penalty", "repetition_penalty"}
OUTPUTS = msc.outputs(PRESETS)
OUTPUT_IDS = [p.name for p, *_ in OUTPUTS]
#: THE 16 MAIN-VIEW FILES, frozen by sha256 at LIVE_SHA — the bytes the sampled jobs of that commit run. History: the
#: live main-view jobs (cluster B, 2026-09-09) ran the files at W7_SHA (their `it` views and the 16384 rung moved nothing;
#: the second amendment APPENDED the `readouts` sub-block behind one marker comment, so a pin is the
#: sha256 of everything BEFORE that marker); main then inherited four wave keys none of the instrument reads
#: (W7_TO_MAIN_INHERITANCE); and on 2026-09-12 the CONSTITUTION-SI PROMPTED BASELINE joined gemma4e4b/main,
#: qwen35_4b/main and nemotron3nano4b/main (explicit_cells gained prompted__si_constitution at its roster position;
#: glm46vflash unchanged), a deliberate wave-config change, so the pin moved to 7dde4613 and the W7_SHA fixture stays
#: beside the new one as the record the live jobs ran — the delta between the two is pinned EXACTLY below.
#: Refresh this table ONLY with a deliberate wave-config change (a roster edit on a main view), never to make the test pass.
MAIN16_SHA256_LIVE = {
    "sdfbase_full22_gemma4e4b_sampled_t0.yaml": "0ecaa3002001ec24c81e67eeb55af7bb49824a0bec4d51ffe6004b08a7e69f0e",
    "sdfbase_full22_gemma4e4b_sampled_t0_cap8192.yaml": "f9ff8a4f19c048623a9a349c35bbf0d1465fa2e6d6b9f3ce88390be74b9963db",
    "sdfbase_full22_gemma4e4b_sampled_t1.yaml": "554c42771d3cf28bd28ca33f6636f47ddb361fdd694261eeda84d649df6bc718",
    "sdfbase_full22_gemma4e4b_sampled_t1_cap8192.yaml": "029170c43c10d0802f3fb022c614fca6fd69d84ccb0747dcd26b177d024241de",
    "sdfbase_full22_qwen35_4b_sampled_t0.yaml": "ef4d83d127507c12dec2052c3d90beb1dd377551e14c216c282e3cf9e8f93845",
    "sdfbase_full22_qwen35_4b_sampled_t0_cap8192.yaml": "183ee7059ee8a250ba1a9a9cc0ec9e3766060b31a3c5152812d7bbeb30651db8",
    "sdfbase_full22_qwen35_4b_sampled_t1.yaml": "b716921a99e763eb1b3eb38be3f07c2c0ba721bc64e59aa7df71a42d5aa7fe52",
    "sdfbase_full22_qwen35_4b_sampled_t1_cap8192.yaml": "32ae7e7ef68ef7a09c160e077f88d7cd95c1d062982271c15a91b5b8dd5995d5",
    "sdfbase_full22_nemotron3nano4b_sampled_t0.yaml": "33dc13817f03ae40b3701b4373f90abe9b7437e719f568778ab93c7fde46db7e",
    "sdfbase_full22_nemotron3nano4b_sampled_t0_cap8192.yaml": "8a5e265c7b19c3008a6e5e5fc76042feb813a02a9f325a7a8821c169cc975fea",
    "sdfbase_full22_nemotron3nano4b_sampled_t1.yaml": "8e2fa5da3afbb96abb43ac8ea11b654ccdc1c83b37cfd36da4ba7c20f0d6bd91",
    "sdfbase_full22_nemotron3nano4b_sampled_t1_cap8192.yaml": "119f955422f23cd24e6a89c0f96bec39000e6dd6d2506ee55e9af4340cfe4773",
    "sdfbase_full22_glm46vflash_sampled_t0.yaml": "fb9500902a917bcb6f18f330de6a2dd43c2269b7b8428354870f7826e541336f",
    "sdfbase_full22_glm46vflash_sampled_t0_cap8192.yaml": "d7cb54ceb72858251df61098541ea00c57e9c2cb7e1fe706161fa924408c00f3",
    "sdfbase_full22_glm46vflash_sampled_t1.yaml": "c0c6b277b7fe37e66905b461536621b7b8ba54157fc7dd86021f47c89e62334f",
    "sdfbase_full22_glm46vflash_sampled_t1_cap8192.yaml": "54ca8cc0d7f31bccc57e492d0242ed90be487a5f0a3a456fab75995662572660",
}


def _run_py(*args, **kw):
    return subprocess.run([PY, *map(str, args)], capture_output=True, text=True, cwd=REPO, **kw)


# --- 1: the generated configs -----------------------------------------------------------------------

def test_generated_configs_are_byte_identical_to_the_generator():
    rc = _run_py(SAMPLED / "make_sampled_configs.py", "--check")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert rc.stdout.count("ok: ") == N_OUTPUTS == 405 + N_FRONT_DOOR_OUTPUTS and "DRIFT" not in rc.stdout and "STALE" not in rc.stdout


def test_the_405_outputs_are_the_tags_x_their_trial_counts_x_base_and_two_rungs():
    names = {p.name for p, *_ in OUTPUTS}
    want = {f"sdfbase_full22_{tag}_sampled_t{t}{s}.yaml" for tag in TAGS for t in range(msc.n_trials_of_tag(PRESETS, tag)) for s in ("", "_cap8192", "_cap16384")}
    assert names == want and len(OUTPUTS) == N_OUTPUTS == 405 + N_FRONT_DOOR_OUTPUTS and 405 == 66 * 2 * 3 + 3 * 1 * 3 and all(p.is_file() for p, *_ in OUTPUTS)
    assert PRESETS["n_trials"] == 2 and PRESETS["seed_base"] == 20260909 and PRESETS["arm"] == "vendor_recommended"
    # the three seed-replicate TAGs render ONE trial (n_trials_by_view), every other TAG the campaign's two; no `_t1` file of theirs exists
    # ... + the six seeded control TAGs at 1 (SEEDED_LANE_VIEWS; rendered only under its private roster, so
    # they are rows in the data file and NOT in the production scope's TAGS)
    assert PRESETS["n_trials_by_view"] == {t: 1 for t in (*SEEDREP_TAGS, *msc.lane_replicate_tags())} and all(msc.n_trials_of_tag(PRESETS, t) == 1 for t in SEEDREP_TAGS)
    assert not (set(msc.lane_replicate_tags()) & set(TAGS)) and msc.replicate_tags() == [*SEEDREP_TAGS, *msc.lane_replicate_tags()]
    assert all(msc.n_trials_of_tag(PRESETS, t) == 2 for t in TAGS if t not in SEEDREP_TAGS)
    assert not any((REPO / "configs").glob("sdfbase_full22_qwen35_4b_s*_sampled_t1*.yaml")) and msc.stale_outputs(PRESETS) == []
    assert PRESETS["applies_to"] == ["static_families", "controls"]
    # THE ONE SCOPE LIST: DESIGN.md's four main views + the two instruct-direct views + the strength frontier's
    # 24 per-dose views (6 doses x 4 targets; the `it_` lane on gemma / qwen, the base lane on nemotron / glm) + llama8b's
    # twelve dose views (2026-09-13: BOTH lanes on the one served instruct model; its main / it views are NOT here) + the three
    # qwen35_4b seed-replicate views (qwen-base-seed-replicates, promoted 2026-09-15 with their roster views) — explicit,
    # NEVER the roster's view list (the halfdose views are out and stay out)
    dose, dose_it = ("dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5"), ("it_dm1", "it_d0p1", "it_d0p25", "it_d0p5", "it_d1", "it_d1p5")
    assert msc.DOSE_TOKENS == DOSE_TOKENS and msc.DOSE_VIEWS == dose and msc.DOSE_VIEWS_IT == dose_it
    # gemma / qwen carry both ladder families since 2026-09-15: the instruct-direct it_d* and the base->instruct d*;
    # gemma's base->instruct ladder is extended to doses 2 and 4 (2026-09-16) — on gemma only, after d1p5, own d2__ / d4__ objects;
    # qwen35_4b also carries the three seed-replicate views s1 / s2 / s3 LAST, after its dose views (qwen-base-seed-replicates, 2026-09-15)
    hi = ("d2", "d4")
    assert msc.HIGH_DOSE_VIEWS == {"gemma4e4b": hi}
    # ... and qwen carries the ReLU difference ladder (2026-09-16): six `d*_relu` views, on qwen only, after its bare d*
    relu = ("dm1_relu", "d0p1_relu", "d0p25_relu", "d0p5_relu", "d1_relu", "d1p5_relu")
    assert msc.RELU_DOSE_VIEWS == {"qwen35_4b": relu} and RELU_DOSE_TAGS == tuple(f"qwen35_4b_{v}" for v in relu)
    assert msc.SAMPLED_VIEWS == {"gemma4e4b": ("main", "it", *dose_it, *dose, *hi), "qwen35_4b": ("main", "it", *dose_it, *dose, *relu, "s1", "s2", "s3"),
                                 "nemotron3nano4b": ("main", *dose), "glm46vflash": ("main", *dose),
                                 "llama8b": ("main", "it", "fullconst", "shortconst", *dose, *dose_it)}
    assert msc.sampled_tags() == [
        "gemma4e4b", "gemma4e4b_it", "gemma4e4b_it_dm1", "gemma4e4b_it_d0p1", "gemma4e4b_it_d0p25", "gemma4e4b_it_d0p5", "gemma4e4b_it_d1", "gemma4e4b_it_d1p5", "gemma4e4b_dm1", "gemma4e4b_d0p1", "gemma4e4b_d0p25", "gemma4e4b_d0p5", "gemma4e4b_d1", "gemma4e4b_d1p5", "gemma4e4b_d2", "gemma4e4b_d4",
        "qwen35_4b", "qwen35_4b_it", "qwen35_4b_it_dm1", "qwen35_4b_it_d0p1", "qwen35_4b_it_d0p25", "qwen35_4b_it_d0p5", "qwen35_4b_it_d1", "qwen35_4b_it_d1p5", "qwen35_4b_dm1", "qwen35_4b_d0p1", "qwen35_4b_d0p25", "qwen35_4b_d0p5", "qwen35_4b_d1", "qwen35_4b_d1p5",
        "qwen35_4b_dm1_relu", "qwen35_4b_d0p1_relu", "qwen35_4b_d0p25_relu", "qwen35_4b_d0p5_relu", "qwen35_4b_d1_relu", "qwen35_4b_d1p5_relu",
        "qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3",
        "nemotron3nano4b", "nemotron3nano4b_dm1", "nemotron3nano4b_d0p1", "nemotron3nano4b_d0p25", "nemotron3nano4b_d0p5", "nemotron3nano4b_d1", "nemotron3nano4b_d1p5",
        "glm46vflash", "glm46vflash_dm1", "glm46vflash_d0p1", "glm46vflash_d0p25", "glm46vflash_d0p5", "glm46vflash_d1", "glm46vflash_d1p5",
        "llama8b", "llama8b_it", "llama8b_fullconst", "llama8b_shortconst",
        "llama8b_dm1", "llama8b_d0p1", "llama8b_d0p25", "llama8b_d0p5", "llama8b_d1", "llama8b_d1p5",
        "llama8b_it_dm1", "llama8b_it_d0p1", "llama8b_it_d0p25", "llama8b_it_d0p5", "llama8b_it_d1", "llama8b_it_d1p5"] + list(FRONT_DOOR_TAGS)
    # ... the front door's TAGs LAST, in the table's order, exactly those the production roster carries (none until stage 05 lands the block)
    assert list(FRONT_DOOR_TAGS) == [msc.tag_of(k, v) for k, vs in msc.FRONT_DOOR_VIEWS.items() for v in vs if v in ((msc.load_roster()["targets"].get(k) or {}).get("views") or {})]
    assert len(PAPER_TAGS) == 69 and set(PAPER_TAGS) == set(KEYS) | set(IT_TAGS) | set(DOSE_TAGS) | set(LLAMA_TAGS) | set(LLAMA_DOSE_TAGS) | set(SEEDREP_TAGS) \
        and len(DOSE_TAGS) == 44 and len(RELU_DOSE_TAGS) == 6 and len(LLAMA_DOSE_TAGS) == 12 and TAGS == PAPER_TAGS + FRONT_DOOR_TAGS
    assert PAPER_TAGS[-12:] == LLAMA_DOSE_TAGS   # the lane's four rows come first, then the frontier's twelve dose views
    assert msc.STATIC_RUNGS == RUNGS and msc.previous_rung(8192) == (4096, False) and msc.previous_rung(16384) == (8192, 8192)
    for key, views in msc.SAMPLED_VIEWS.items():
        for view in views:
            tag = msc.tag_of(key, view)
            assert msc.split_tag(tag) == (key, view) and msc.key_of(tag) == key and msc.in_scope(key, view)
            assert tag == (key if view == "main" else f"{key}_{view}")
    for tag in IT_TAGS:
        assert msc.key_of(tag) == tag[:-3]
    # out of scope: the halfdose view, a dose view on the WRONG lane (gemma's doses are `it_d*`, nemotron's `d*`), an `it`
    # view of a base-lane target, llama8b's `halfdose` (never a view of that target), an unknown key — every one refuses,
    # none is silently mapped (llama8b's `main` / `it` ARE in scope since 2026-09-14)
    for key, view in (("nemotron3nano4b", "halfdose"), ("gemma4e4b", "d0p75"), ("nemotron3nano4b", "it_d0p25"),
                      ("glm46vflash", "it"), ("llama8b", "halfdose"), ("llama3b", "d1")):
        assert not msc.in_scope(key, view), (key, view)
        with pytest.raises(SystemExit, match="not a \\(target, view\\) in the sampled scope"):
            msc.key_of(msc.tag_of(key, view))
    with pytest.raises(SystemExit, match="not one of the static rungs"):
        msc.rung_cap(4096)
    assert msc.rung_cap(True) == 8192 and msc.rung_cap(False) is None and msc.rung_cap(16384) == 16384


#: the commit whose bytes the pin records (2026-09-12) and its fixture — the sixteen
#: files parsed and hashed at that sha, written once from `git show` (the test carries its own evidence: the Modal suite
#: runs in a shallow exact-sha clone with no history, so no test-time git call may be load-bearing).
LIVE_SHA = "7dde46131ef18a1d825706d2c69937cfbbf13954"
LIVE_FIXTURE = REPO / "tests" / "fixtures" / "sampled_main16_live_7dde4613.json"
#: the record the live main-view jobs ran (the tip of 2026-09-09), kept as data
W7_SHA = "6a162dba75a7ce29548952772a55d55d6e3144bd"
W7_FIXTURE = REPO / "tests" / "fixtures" / "sampled_main16_live_6a162dba.json"
#: what main ADDED to every sampled config by inheritance from its wave config between W7_SHA and LIVE_SHA -- main's
#: multi-turn 4,096 cap (attacks.target_max_new_tokens), the canonical best-of-N climb (bon.climb_*) and the judged
#: controls' OWN cap key (attacks.controls_max_new_tokens, 512 = the live run's target_max_new_tokens, so the
#: controls' gen block and store key did not move). None is read by the sampled instrument: `(was at W7_SHA, is now)`,
#: None = absent.
W7_TO_MAIN_INHERITANCE = {("attacks", "target_max_new_tokens"): (512, 4096),
                          ("attacks", "controls_max_new_tokens"): (None, 512),
                          ("bon", "climb_cap"): (None, 4096), ("bon", "climb_decode"): (None, "as_drawn")}
#: ... and what main changed by inheritance between LIVE_SHA and today INSIDE a section: the climb's wave-wide
#: `bon.climb_temperature: 1.0` (inherited at 7dde4613) was RENAMED `bon.climb_decode: as_drawn` on 2026-09-17
#: (best-of-N draws at each served model's vendor preset, and a climb re-draws at the sampler its 512 store
#: records -- no wave-wide scalar can say that). Not read by the sampled instrument: `(was at LIVE_SHA, is now)`.
LIVE_TO_MAIN_INHERITANCE = {("bon", "climb_temperature"): (1.0, None), ("bon", "climb_decode"): (None, "as_drawn")}
#: ... and the ONE deliberate wave-config change: the constitution-SI prompted baseline (2026-09-12) inserted into
#: explicit_cells at its roster position -- gemma / qwen: before the derived pair (index 7 of the 10-cell list);
#: nemotron: after prompted__safety_refusal (index 6 of 9); glm: not served (its list is unchanged)
W7_TO_MAIN_NEW_CELL = "prompted__si_constitution"
W7_TO_MAIN_NEW_CELL_INDEX = {"gemma4e4b": 7, "qwen35_4b": 7, "nemotron3nano4b": 6, "glm46vflash": None}
#: ... and the RETIREMENT of the two obsolete benign controls (decided 2026-09-11, landed on main 2026-09-12 AFTER
#: the constitution-SI jobs ran): alpaca_eval and wmdp leave every wave config, so the generator inherits the shortened
#: top-level `controls` list. Both records (the W7_SHA fixture and the 7dde4613 live pin) carry the long list; the working
#: tree carries the short one. Top-level (not section-scoped) inheritance, named for both comparisons.
_CONTROLS_LIVE = ["xstest", "or_bench_hard_1k", "wmdp", "mmlu", "gsm8k", "alpaca_eval"]
_CONTROLS_NOW = ["xstest", "or_bench_hard_1k", "mmlu", "gsm8k"]
W7_TO_MAIN_TOPLEVEL = {"controls": (_CONTROLS_LIVE, _CONTROLS_NOW)}
LIVE_TO_MAIN_TOPLEVEL = {"controls": (_CONTROLS_LIVE, _CONTROLS_NOW)}


def _fixture(path, sha) -> dict:
    fx = json.loads(path.read_text())
    assert fx["source_sha"] == sha and fx["readouts_marker"] == msc.READOUTS_MARKER
    assert set(fx["files"]) == set(MAIN16_SHA256_LIVE) and len(fx["files"]) == 16
    return fx
def test_the_dose_view_configs_follow_the_tag_under_the_keys_preset_and_the_campaigns_ladder():
    """The per-dose TAGs (the 24; twelve more on the base->instruct families; gemma's d2 / d4; qwen's six `d*_relu` of
    the ReLU difference ladder — ALL_DOSE_TAGS): six files each (2 trials x base / cap8192 / cap16384) generated
    from the view's wave config; the preset is the KEY's (the served model's, as for the `it` views); the cells, the bank view
    (`defenses_dir` = the (key, view) view dir) and the Petri manifest follow the TAG; the ladder is the CAMPAIGN's — base
    4096, rungs 8192 / 16384, controls 512 — not a per-TAG base cap; and the `bon.max_new_tokens` override on the dose
    views' wave configs rides through UNTOUCHED: the `bon` block is the wave
    config's on both sides, so `_verify` / `f22s_check_config` (which compare the two AS DATA) are unaffected by it."""
    for tag in ALL_DOSE_TAGS:
        key, view = msc.split_tag(tag)
        # the lane is the target's: `it_d*` only on the instruct lanes (gemma / qwen / llama); bare `d*` on nemotron / glm
        # AND on gemma / qwen / llama too (the base->instruct family riding the one served instruct model)
        assert view in msc.DOSE_VIEWS + msc.DOSE_VIEWS_IT + msc.HIGH_DOSE_VIEWS.get(key, ()) + msc.RELU_DOSE_VIEWS.get(key, ())   # + gemma's d2 / d4, qwen's relu (2026-09-16)
        assert (not view.startswith("it_")) or key in ("gemma4e4b", "qwen35_4b", "llama8b")
        wave = yaml.safe_load(msc.wave_config_path(tag).read_text())
        assert wave["private_run"] == f"sdfbase_full22_{tag}" and wave["explicit_cells"][0] == "none" and len(wave["explicit_cells"]) >= 2
        assert (REPO / f"configs/capeval/manifests/petri_full22_{tag}.yaml").is_file()
        for trial in (0, 1):
            for esc in (False, 8192, 16384):
                path = msc.output_path(tag, trial, esc)
                assert path.is_file(), path
                cfg = yaml.safe_load(path.read_text())
                assert cfg["private_run"] == msc.sampled_name(tag, trial, esc) == f"sdfbase_full22_{tag}_sampled_t{trial}" + (f"_cap{esc}" if esc else "")
                assert cfg["explicit_cells"] == wave["explicit_cells"] and cfg["defenses_dir"] == f"out/full22/{key}/cells_view/{view}"
                assert cfg["model"] == ROSTER["targets"][key]["model"] == PRESETS["presets"][key]["model"]
                sd = cfg["sampled_decode"]
                assert {k: v for k, v in sd.items() if k in msc.ALLOWED_SAMPLER_KEYS} == PINNED[key] and sd["source"] == PRESETS["presets"][key]["source"]
                assert sd["seed_base"] == 20260909 and sd["trial"] == trial and sd["n_trials"] == 2   # the same per-item seed rule
                assert sd["static_cap"] == cfg["attacks"]["static_max_new_tokens"] == (esc or 4096) and sd["controls_cap"] == cfg["attacks"]["controls_max_new_tokens"] == 512
                assert cfg.get("bon") == wave.get("bon"), tag       # the bon override is inherited, never a permitted-key difference
    # the halfdose view (the greedy 0.5-dose precedent) is still out: no config, no scope entry (the 0.5 point is
    # re-run as the d0p5 view under THIS instrument instead)
    assert not msc.in_scope("nemotron3nano4b", "halfdose") and not list((REPO / "configs").glob("*halfdose*_sampled_*"))


def _live_fixture() -> dict:
    fx = _fixture(LIVE_FIXTURE, LIVE_SHA)
    assert {n: f["sha256_pre_marker"] for n, f in fx["files"].items()} == MAIN16_SHA256_LIVE   # the fixture IS the pinned record
    return fx


#: a site placeholder (src/sisteer/site.py): the public release wrote one where a record spelled the site's literal path
_SITE_PLACEHOLDER = re.compile(r"\$\{?(SHARED|HF_HOME|STATE|STAGE|WORK|GRADE|WANDB_PROJECT)\}?")


def _site_equal(record, today) -> bool:
    """Equal as data, where a site placeholder in one of today's values stands for the literal text the record spelled
    there (each placeholder matches a non-empty run; everything around it must match exactly)."""
    if isinstance(record, str) and isinstance(today, str) and _SITE_PLACEHOLDER.search(today):
        parts = _SITE_PLACEHOLDER.split(today)
        pattern = "".join(re.escape(x) if i % 2 == 0 else ".+" for i, x in enumerate(parts))
        return re.fullmatch(pattern, record, re.S) is not None
    if isinstance(record, dict) and isinstance(today, dict):
        return set(record) == set(today) and all(_site_equal(record[k], today[k]) for k in record)
    if isinstance(record, list) and isinstance(today, list):
        return len(record) == len(today) and all(_site_equal(a, b) for a, b in zip(record, today))
    return record == today


def _prose_head(data):
    """`data` with every `source` string under its `sampled_decode` block cut to its model-and-revision head (the text
    before the first colon): the public release reworded that provenance prose after the snapshots were taken, and the
    sampled instrument reads none of it (the controls gen block and the static decode leave `source` out)."""
    def cut(x, under):
        if isinstance(x, dict):
            return {k: (v.split(":", 1)[0] if under and k == "source" and isinstance(v, str) else cut(v, under or k == "sampled_decode"))
                    for k, v in x.items()}
        if isinstance(x, list):
            return [cut(v, under) for v in x]
        return x
    return cut(data, False)


def test_the_sixteen_main_view_files_are_the_pinned_instrument_as_data_and_differ_from_w7_only_by_the_named_keys():
    """The working-tree main-view files are the pinned record (LIVE_FIXTURE at LIVE_SHA -- the sha the constitution-SI jobs
    ran under; its pre-marker bytes hash to MAIN16_SHA256_LIVE inside the fixture) up to EXACTLY one named top-level
    inheritance, the retirement of the two obsolete benign controls that landed on main after those jobs ran
    (LIVE_TO_MAIN_TOPLEVEL). Against the earlier record the first live jobs ran (W7_FIXTURE), the files differ AS DATA in
    EXACTLY the four inherited wave keys (W7_TO_MAIN_INHERITANCE), that same controls retirement (W7_TO_MAIN_TOPLEVEL),
    plus the constitution-SI prompted cell inserted into explicit_cells at its roster position (W7_TO_MAIN_NEW_CELL_INDEX;
    glm unchanged) -- and in nothing the
    sampled instrument reads: `sampled_decode`, the static cap and the controls' gen block (the store key) are the live
    run's, so a driver at the new sha resuming over that earlier store derives the same per-row decode block. Both fixtures are
    self-contained; their faithfulness to their commits is the next test's, run where the commits exist."""
    from sisteer.eval.cell_controls import controls_gen_config
    fx, w7 = _live_fixture(), _fixture(W7_FIXTURE, W7_SHA)
    for n in MAIN16_SHA256_LIVE:
        text = (REPO / "configs" / n).read_text()
        assert text.count(msc.READOUTS_MARKER) == 1 and text.rstrip("\n").splitlines()[-1].startswith("      note: ")   # the block is LAST
        now = yaml.safe_load(text)
        # the 7dde4613 record vs the working tree: identical as data once the ONE named top-level inheritance is popped
        pin, now_ = dict(fx["files"][n]["data"]), dict(now)
        for key_, (was, is_now) in LIVE_TO_MAIN_TOPLEVEL.items():
            assert pin.pop(key_, None) == was and now_.pop(key_, None) == is_now, (n, key_)
        pin["bon"], now_["bon"] = dict(pin["bon"]), dict(now_["bon"])
        for (sect, key_), (was, is_now) in LIVE_TO_MAIN_INHERITANCE.items():
            assert pin[sect].pop(key_, None) == was and now_[sect].pop(key_, None) == is_now, (n, sect, key_)
        pin, now_ = _prose_head(pin), _prose_head(now_)
        assert _site_equal(pin, now_), (n, {k for k in set(pin) | set(now_) if not _site_equal(pin.get(k), now_.get(k))})
        live, now = _prose_head(w7["files"][n]["data"]), _prose_head(now)
        key = next(k for k in KEYS if n.startswith(f"sdfbase_full22_{k}_sampled_"))
        # the instrument a driver derives is the earlier run's: the same block, and -- on the base trial, the config the
        # controls driver runs under -- the same controls gen block, i.e. the same store key
        assert now["sampled_decode"] == live["sampled_decode"] and now["sampled_decode"]["controls_cap"] == 512, n
        assert now["attacks"]["controls_max_new_tokens"] == 512, n
        if "escalated_from" not in now["sampled_decode"]:
            assert controls_gen_config(live) == controls_gen_config(now) and controls_gen_config(now)["max_new_tokens"] == 512, n
        for (sect, key_), (was, is_now) in W7_TO_MAIN_INHERITANCE.items():
            assert live[sect].pop(key_, None) == was and now[sect].pop(key_, None) == is_now, (n, sect, key_)
        idx = W7_TO_MAIN_NEW_CELL_INDEX[key]
        cells_now, cells_w7 = list(now.pop("explicit_cells")), list(live.pop("explicit_cells"))
        if idx is None:
            assert cells_now == cells_w7 and W7_TO_MAIN_NEW_CELL not in cells_now, n
        else:
            assert cells_now[idx] == W7_TO_MAIN_NEW_CELL and cells_now[:idx] + cells_now[idx + 1:] == cells_w7, n
            assert cells_now.count(W7_TO_MAIN_NEW_CELL) == 1 and cells_now[0] == "none", n
        for key_, (was, is_now) in W7_TO_MAIN_TOPLEVEL.items():
            assert live.pop(key_, None) == was and now.pop(key_, None) == is_now, (n, key_)
        assert _site_equal(live, now), (n, {k for k in set(live) | set(now) if not _site_equal(live.get(k), now.get(k))})
    assert len(MAIN16_SHA256_LIVE) == 16 and set(MAIN16_SHA256_LIVE) == {msc.output_path(k, t, e).name for k in KEYS for t in (0, 1) for e in (False, 8192)}


@pytest.mark.parametrize("sha,fixture", [(LIVE_SHA, LIVE_FIXTURE), (W7_SHA, W7_FIXTURE)], ids=["live-7dde4613", "w7-6a162dba"])
def test_the_fixture_is_the_frozen_commits_bytes_where_the_commit_is_present(sha, fixture):
    """The secondary check, against git itself: where the commit is in the clone (a full checkout; NOT the Modal suite's
    shallow exact-sha clone, which skips here by name), every file's text at that commit parses to the fixture's data and
    hashes whole to the fixture's sha256_text (and, for the live pin, before the marker to MAIN16_SHA256_LIVE) -- so a
    fixture cannot drift from the record it stands for."""
    import subprocess
    probe = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f"commit {sha[:8]} is not present in this checkout (a shallow exact-sha clone); the fixture stands")
    fx = _fixture(fixture, sha)
    for n, rec in fx["files"].items():
        text = subprocess.run(["git", "show", f"{sha}:configs/{n}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
        assert text.count(msc.READOUTS_MARKER) == 1
        assert hashlib.sha256(text[:text.index(msc.READOUTS_MARKER)].encode("utf-8")).hexdigest() == rec["sha256_pre_marker"], n
        if sha == LIVE_SHA:
            assert rec["sha256_pre_marker"] == MAIN16_SHA256_LIVE[n], n
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == rec["sha256_text"], n
        # the two hashes above pin the commit's text byte for byte; its data is the fixture's up to _commit_equal's two
        # readings (the `source` prose, and a site placeholder where the commit spelled the literal path)
        assert _commit_equal(yaml.safe_load(text), rec["data"]), n


#: a site placeholder in a fixture value (src/sisteer/site.py), as the history check reads it: one absolute path the
#: commit spelled there, a run of path characters from "/" with no whitespace, quote or second placeholder in it
_COMMIT_PLACEHOLDER = re.compile(r"\$\{?(SHARED|HF_HOME|STATE|STAGE|WORK|GRADE)\}?")
_COMMIT_PATH = r"/[^\s'\"$]+"


def _commit_equal(commit, fixture) -> bool:
    """A frozen commit's parsed config against the fixture's data: equal everywhere, with two readings. A `source` value
    (prose) is a string on both sides but not compared, since the site labels in that prose were renamed after the commit
    and the working-tree tests compare it there. A fixture string that spells a site placeholder matches when each
    placeholder stands for one absolute path in the commit's string and everything around it is exact."""
    if isinstance(commit, dict) and isinstance(fixture, dict):
        return set(commit) == set(fixture) and all(
            (isinstance(commit[k], str) and isinstance(fixture[k], str)) if k == "source" else _commit_equal(commit[k], fixture[k])
            for k in commit)
    if isinstance(commit, list) and isinstance(fixture, list):
        return len(commit) == len(fixture) and all(_commit_equal(a, b) for a, b in zip(commit, fixture))
    if isinstance(commit, str) and isinstance(fixture, str) and _COMMIT_PLACEHOLDER.search(fixture):
        parts = _COMMIT_PLACEHOLDER.split(fixture)
        pattern = "".join(re.escape(x) if i % 2 == 0 else _COMMIT_PATH for i, x in enumerate(parts))
        return re.fullmatch(pattern, commit) is not None
    return commit == fixture


def test_the_history_check_reads_a_placeholder_as_one_path_and_masks_only_source_prose():
    """_commit_equal's two readings, and nothing wider: a placeholder stands for one absolute path (not an empty run, not a
    relative path, not text with whitespace, not a different tail); a `source` value is skipped only as a string on both
    sides; every other value must be exactly equal."""
    lit = "/data/acct/exp/jailbreak/contrast/benign.json"
    assert _commit_equal({"p": lit}, {"p": "$SHARED/exp/jailbreak/contrast/benign.json"})
    assert _commit_equal({"p": lit}, {"p": "${SHARED}/exp/jailbreak/contrast/benign.json"})
    for wrong in ("/exp/jailbreak/contrast/benign.json", "data/acct/exp/jailbreak/contrast/benign.json",
                  "/data/a b/exp/jailbreak/contrast/benign.json", "/data/acct/exp/jailbreak/contrast/harmful.json"):
        assert not _commit_equal({"p": wrong}, {"p": "$SHARED/exp/jailbreak/contrast/benign.json"}), wrong
    assert not _commit_equal({"p": lit}, {"p": "$UNKNOWN/exp/jailbreak/contrast/benign.json"})
    assert _commit_equal({"a": {"source": "read on site one"}, "b": [1, 2]}, {"a": {"source": "read on site two"}, "b": [1, 2]})
    assert not _commit_equal({"a": {"source": "x"}}, {"a": {"source": ["x"]}})
    assert not _commit_equal({"a": {"source": "x"}, "b": 1}, {"a": {"source": "x"}, "b": 2})
    assert not _commit_equal({"a": 1}, {"a": 1, "source": "x"})


def test_every_sampled_config_carries_the_readouts_sub_block_and_the_drivers_ignore_it():
    """The `readouts` sub-block: `sampled_decode.readouts.gsm8k` (cap 8192, thinking "on", the vendor's THINKING
    preset for Qwen — the card's thinking row — and the single preset for Gemma / Nemotron / GLM with a `source` that says no
    thinking row exists) and `readouts.mmlu` (letter_logprob_argmax, sampler-moot) on all 36 configs; recorded as data in
    configs/sdfbase_sampled_decode.yaml. The static validator (jailbreak_run._sampled_decode) and the controls one
    (cell_controls.controls_gen_config) ADMIT the key and derive the same decode / gen as without it; an unknown key still refuses."""
    sys.path.insert(0, str(REPO / "scripts"))
    import jailbreak_run as jr
    from sisteer.eval.cell_controls import controls_gen_config, SampledDecodeError, SAMPLED_DECODE_BLOCK_KEYS
    r = PRESETS["readouts"]
    assert set(r) == {"gsm8k", "mmlu"} and r["gsm8k"]["cap"] == 8192 and r["gsm8k"]["thinking"] == "on"
    # the thinking row is stated under the paper's qwen35_4b AND its identical twin qwen35_4b_demo (the worked example serves
    # the same Qwen3.5-4B checkpoint; 51609a681): exactly these two keys, the twin's row the 4B's verbatim — a THIRD key, or
    # a twin that drifts from the row it mirrors, fails here
    assert set(r["gsm8k"]["presets"]) == {"qwen35_4b", "qwen35_4b_demo"}
    assert r["gsm8k"]["presets"]["qwen35_4b_demo"] == r["gsm8k"]["presets"]["qwen35_4b"]
    # llama8b (2026-09-13): the data file carries origin/main's `thinking_by_key: {llama8b: "off"}` (+ its source) — Llama-3.1 has
    # no thinking switch. THIS sha's generator does not read the key (main's does): every rendered config keeps the block's
    # "on", and that is what the FROZEN job-side belt (common_sampled.sh f22s_check_config, an invariant of this instrument)
    # REQUIRES on every sampled config — so "on" is what launches here. The sub-block is inert for the dose screens (the static
    # families and the judged controls ignore it); closing the gap is the generator's + the belt's change together, on rebase.
    assert r["gsm8k"]["thinking_by_key"] == {"llama8b": "off"} and "has no thinking mode" in r["gsm8k"]["thinking_by_key_source"]["llama8b"]
    # closed on main (2026-09-13/14): the generator READS thinking_by_key (readouts.gsm8k.thinking_by_key -> the config's mode);
    # llama8b's GSM8K readout is OMITTED by its roster (capability_omitted), so no llama sampled config renders a thinking mode
    assert "thinking_by_key" in (SAMPLED / "make_sampled_configs.py").read_text()
    assert '.get("thinking") == "on"' in (SAMPLED / "common_sampled.sh").read_text()             # the frozen belt's requirement
    qwen_think = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
    assert r["gsm8k"]["presets"]["qwen35_4b"]["preset"] == qwen_think and "THINKING mode" in r["gsm8k"]["presets"]["qwen35_4b"]["source"]
    assert r["mmlu"] == {"instrument": "letter_logprob_argmax",
                         "note": "sampler-moot: teacher-forced letter-logprob argmax; the sampled run measures server-to-server logprob nondeterminism only"}
    assert "readouts" in jr._SAMPLED_DECODE_META_KEYS and "readouts" in SAMPLED_DECODE_BLOCK_KEYS
    for path, tag, trial, esc in OUTPUTS:
        key = msc.key_of(tag)
        cfg = yaml.safe_load(path.read_text())
        sd = cfg["sampled_decode"]
        assert list(sd)[-1] == "readouts" and sd["readouts"] == msc.expected_readouts(PRESETS, key)
        g = sd["readouts"]["gsm8k"]
        if key == "llama8b":
            # the sdfbase llama8b preset REMOVED gsm8k (a single GSM8K instrument, owned elsewhere in the roster;
            # roster capability_omitted): the readout block records the OMISSION with its reason — never a runnable gsm8k block
            assert g == {"omitted": True, "reason": g["reason"]} and "owned by the evaluation lane" in g["reason"]
        else:
            # every other key: thinking "on" at cap 8192 (quoted: a bare on/off is a YAML bool)
            assert g["cap"] == 8192 and g["thinking"] == "on" and g["thinking"] not in (True, False)
            if key in r["gsm8k"]["presets"]:
                # a key the DATA FILE states a THINKING row for — exactly the paper's qwen35_4b and its identical twin qwen35_4b_demo,
                # pinned above — renders that row (the card's THINKING-mode source), never the single preset its PINNED row mirrors. The
                # branch keys on the file's rows, not on a key literal: the worked example's sampled configs joined OUTPUTS with its roster
                # block (demo-import c7e73cd04) and a `== "qwen35_4b"` literal here demanded the single preset of the twin the file gives a
                # thinking row — a red the configs were right about (they equal expected_readouts, asserted above)
                assert g["preset"] == qwen_think and g["preset"] != PINNED[key] and "THINKING mode" in g["source"] and PRESETS["presets"][key]["model"] in g["source"]
            else:
                assert g["preset"] == PINNED[key] and "no thinking-specific vendor row exists" in g["source"] and PRESETS["presets"][key]["model"] in g["source"]
        assert sd["readouts"]["mmlu"]["instrument"] == "letter_logprob_argmax" and "sampler-moot" in sd["readouts"]["mmlu"]["note"]
        # the static validator: admitted, ignored — the decode is arm + the top-level preset, exactly as before the sub-block
        v = jr._sampled_decode(cfg, trial)
        assert v["decode"] == {"arm": "vendor_recommended", **PINNED[key]} and "readouts" not in v["decode"] and v["trial"] == trial
        # the controls validator: admitted, ignored — no readouts key in the gen block the stores are keyed on. A STATIC rung
        # config is refused there as before (a controls driver never runs under one: the escalated_from shape, not the sub-block)
        if msc.rung_cap(esc) is None:
            gen = controls_gen_config(cfg)
            assert gen["do_sample"] is True and gen["max_new_tokens"] == 512 and "readouts" not in gen and gen["temperature"] == PINNED[key]["temperature"]
        else:
            with pytest.raises(SampledDecodeError, match="not the controls ladder"):
                controls_gen_config(cfg)
    # a key outside the closed sets still refuses on both paths
    cfg = yaml.safe_load(OUTPUTS[0][0].read_text())
    cfg["sampled_decode"]["bogus"] = 1
    with pytest.raises(SystemExit, match="outside the allowed set"):
        jr._sampled_decode(cfg, 0)
    with pytest.raises(SampledDecodeError, match="outside the allowed set"):
        controls_gen_config(cfg)
    # the data file refuses a thinking preset identical to the single preset, a bare-bool thinking, a stray readout
    import copy
    for mut, needle in ((lambda d: d["readouts"]["gsm8k"]["presets"].__setitem__("gemma4e4b", {"preset": dict(PINNED["gemma4e4b"]), "source": "x"}), "equals the single preset"),
                        (lambda d: d["readouts"]["gsm8k"].__setitem__("thinking", True), "must be the string"),
                        (lambda d: d["readouts"].__setitem__("gsm8k_plus", {}), "exactly")):
        bad = copy.deepcopy(PRESETS); mut(bad)
        f = REPO / "configs" / "_tmp_bad_presets.yaml"
        try:
            f.write_text(yaml.safe_dump(bad))
            with pytest.raises(SystemExit, match=needle):
                msc.load_presets(f)
        finally:
            f.unlink(missing_ok=True)


@pytest.mark.parametrize("path,tag,trial,esc", OUTPUTS, ids=OUTPUT_IDS)
def test_sampled_config_is_its_wave_config_as_data_except_the_permitted_keys(path, tag, trial, esc):
    key, cap = msc.key_of(tag), msc.rung_cap(esc)
    cfg = yaml.safe_load(path.read_text())
    wave = yaml.safe_load(msc.wave_config_path(tag).read_text())
    name = msc.sampled_name(tag, trial, esc)
    changed = {k for k in set(cfg) | set(wave) if cfg.get(k) != wave.get(k)}
    assert changed == {"out_dir", "private_run", "sampled_decode"} | ({"attacks"} if cap else set()), changed
    assert cfg["out_dir"] == f"out/{name}" and cfg["private_run"] == name
    assert cfg["explicit_cells"] == wave["explicit_cells"]
    import f22_roster as _fr                                             # (the llama8b branch below binds `fr` locally; a distinct name here)
    view = tag[len(key) + 1:] if tag != key else "main"
    if _fr.steered_wave_view(view, key):   # a steered-wave view's rungs enumerate no unsteered leg — `none` is served implicitly, listed nowhere
        assert cfg["explicit_cells"] == list(_fr.view_cell_universe(ROSTER, key, view)) and "none" not in cfg["explicit_cells"] and cfg["explicit_cells"], cfg["explicit_cells"]
    else:
        assert cfg["explicit_cells"][0] == "none"
    if tag in IT_TAGS[:2]:   # lane F's instruct-direct views: none + the KEY's roster `it` cells (the itdoc pair, the batch-64 pair where declared, the prompted baselines)
        assert cfg["explicit_cells"] == it_cells(key) and cfg["defenses_dir"].endswith(f"/{key}/cells_view/it")
        assert all(c in cfg["explicit_cells"] for c in ITDOC64_CELLS) is (key in ITDOC64_KEYS), (key, cfg["explicit_cells"])
    elif tag.startswith("llama8b"):   # llama8b: the roster's served cells (its fits are pending_fit until staged)
        import f22_roster as fr
        view = tag[len("llama8b_"):] if "_" in tag else "main"
        assert cfg["explicit_cells"] == ["none"] + list(fr.served_cells(ROSTER, "llama8b", view))
        assert cfg["defenses_dir"].endswith(f"/llama8b/cells_view/{view}")
    a, b = dict(cfg["attacks"]), dict(wave["attacks"])
    assert a.pop("static_max_new_tokens") == (cap or 4096) and b.pop("static_max_new_tokens") == 4096
    # the multi-turn target's cap is the roster's 4,096 (2026-09-11); the judged controls' cap is their OWN
    # key, 512 -- the sampled ladder's root -- inherited from the wave config, never the target's
    assert a == b and a["target_max_new_tokens"] == 4096 and a["controls_max_new_tokens"] == 512
    sd = cfg["sampled_decode"]
    assert sd == msc.expected_block(PRESETS, tag, trial, esc)
    # the campaign base for every TAG but the three seed-replicate views, which draw under THEIR view's base (seed_base_by_view)
    assert sd["trial"] == trial and sd["n_trials"] == msc.n_trials_of_tag(PRESETS, tag) == (1 if tag in SEEDREP_TAGS else 2)
    assert sd["seed_base"] == msc.seed_base_of_tag(PRESETS, tag) and sd["arm"] == "vendor_recommended"
    assert (sd["seed_base"] == 20260909) == (tag not in SEEDREP_TAGS)
    assert sd["static_cap"] == cfg["attacks"]["static_max_new_tokens"]
    assert sd["controls_cap"] == cfg["attacks"]["controls_max_new_tokens"] == 512 != cfg["attacks"]["target_max_new_tokens"]
    assert sd["applies_to"] == ["static_families", "controls"]
    # a rung is chained from the rung before it: 8192 from the base trial, 16384 from the 8192 rung
    if cap == 8192:
        assert sd["escalated_from"] == {"kind": "static", "cap": 4096, "private_run": msc.sampled_name(tag, trial, False)}
    elif cap == 16384:
        assert sd["escalated_from"] == {"kind": "static", "cap": 8192, "private_run": msc.sampled_name(tag, trial, 8192)}
    else:
        assert "escalated_from" not in sd
    # the block's key ORDER is DESIGN.md's: head, the preset keys in the allowed order, then the tail; the preset is the KEY's
    keys = list(sd)
    sampler = [k for k in keys if k in msc.ALLOWED_SAMPLER_KEYS]
    assert keys[:4] == ["arm", "trial", "n_trials", "seed_base"]
    assert keys[4:4 + len(sampler)] == sampler == [k for k in msc.ALLOWED_SAMPLER_KEYS if k in PINNED[key]]
    assert {k: sd[k] for k in sampler} == PINNED[key]
    assert keys[4 + len(sampler):4 + len(sampler) + 5] == ["source", "source_kind", "applies_to", "static_cap", "controls_cap"]
    assert keys[-1] == "readouts" and keys[4 + len(sampler) + 5:] == (["escalated_from", "readouts"] if cap else ["readouts"])
    # the header names the generator and the source config's sha; the wave header follows byte-for-byte
    text = path.read_text()
    src_bytes = msc.wave_config_path(tag).read_bytes()
    import hashlib
    assert f"GENERATED by {msc.GENERATOR}" in text and hashlib.sha256(src_bytes).hexdigest() in text
    assert src_bytes.decode("utf-8").replace(f"out_dir: out/sdfbase_full22_{tag}\n", f"out_dir: out/{name}\n") \
        .replace(f"private_run: sdfbase_full22_{tag}\n", f"private_run: {name}\n") \
        .replace("  static_max_new_tokens: 4096\n", f"  static_max_new_tokens: {cap or 4096}\n") in text
    # the Petri manifest is the wave's for THIS (key, view) (the sbatch pairs them); no manifest is generated here
    assert f"configs/capeval/manifests/petri_full22_{tag}.yaml (the sbatch pairs them)" in text
    assert (REPO / f"configs/capeval/manifests/petri_full22_{tag}.yaml").is_file()
    assert not (REPO / f"configs/capeval/manifests/petri_full22_{tag}_sampled_t{trial}.yaml").exists()


#: the reproduction pipeline's FRONT-DOOR preset: a row for a target the roster does not name (legal, as llama8b's once was); the
#: worked example's row is the qwen35_4b row's, verbatim (pinned in PINNED and asserted against the 4B row below)
FRONT_DOOR_KEYS = ("qwen35_4b_demo",)


def test_presets_match_the_data_file_and_the_pinned_table():
    assert set(PRESETS["presets"]) == set(ALL_KEYS) | set(FRONT_DOOR_KEYS)
    for key in ALL_KEYS:
        p = PRESETS["presets"][key]
        assert p["preset"] == PINNED[key], key
        assert list(p["preset"]) == [k for k in msc.ALLOWED_SAMPLER_KEYS if k in PINNED[key]], key   # the allowed order
        assert p["source_kind"] == SOURCE_KIND[key], key
        for needle in SOURCE_MUST_CONTAIN[key]:
            assert needle in p["source"], (key, needle)
        if key in ROSTER["targets"]:
            assert (p["model"], p["revision"]) == (ROSTER["targets"][key]["model"], ROSTER["targets"][key]["model_revision"])
        else:
            # llama8b: no roster target on main yet (the wave's); the row still names its own pins
            assert key == "llama8b" and p["model"] == "NousResearch/Meta-Llama-3.1-8B-Instruct"
            assert p["revision"] == "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
        assert p["revision"][:8] in p["source"]
        # the block carries exactly the preset's sampler keys (a model without top_k has no top_k key):
        # the generator's rendering for a rendered target, the drivers' default builder for a
        # driver-side row (llama8b: no view rendered here — the sdfbase-llama8b wave's)
        if key in msc.SAMPLED_VIEWS:
            block = msc.expected_block(PRESETS, key, 0, False)
        else:
            from sisteer.eval import sampled_decode as sdd
            block = sdd.default_sampled_block(PRESETS, key, trial=0, static_cap=msc.STATIC_CAP_BASE,
                                              controls_cap=msc.CONTROLS_CAP)
        assert {k for k in block if k in msc.ALLOWED_SAMPLER_KEYS} == set(PINNED[key])
    for key in FRONT_DOOR_KEYS:          # the worked example's row: the pinned table's values, in the allowed order
        p = PRESETS["presets"][key]
        assert p["preset"] == PINNED[key], key
        assert list(p["preset"]) == [k for k in msc.ALLOWED_SAMPLER_KEYS if k in PINNED[key]], key
    # nemotron: the preset that RUNS differs from the shipped file, and the file's values sit beside it
    n = PRESETS["presets"]["nemotron3nano4b"]
    assert n["shipped_generation_config"] == {"temperature": 1.0, "top_p": 0.95} and n["preset"]["top_p"] == 1.0
    assert "shipped_generation_config" not in msc.expected_block(PRESETS, "nemotron3nano4b", 0, False)   # the block's key set is DESIGN's
    for key in ("gemma4e4b", "qwen35_4b", "glm46vflash", "llama8b"):
        assert "shipped_generation_config" not in PRESETS["presets"][key]
    # gemma / qwen are DESIGN.md's table verbatim (the amendments touched nemotron and glm only)
    assert "temperature 1.0, top_p 0.95, top_k 64" in DESIGN and "presence_penalty 1.5, repetition_penalty 1.0" in DESIGN


def test_only_qwen_and_glm_carry_a_penalty_key():
    from sisteer.eval import sampled_decode as sdd
    with_penalty = {k for k, p in PRESETS["presets"].items() if PENALTY_KEYS & set(p["preset"])}
    # qwen35_4b_demo is the 4B's identical twin (the worked example serves the same checkpoint and carries its preset row
    # verbatim), so it carries the 4B's penalty key too — and the preset join collapses the pair to the paper's key
    # (sampled_decode.preset_key_for_model, b19cb3b9f); a twin whose row drifted would not collapse, and any OTHER key with a
    # penalty term still fails the set below
    assert with_penalty == {"qwen35_4b", "qwen35_4b_demo", "glm46vflash"}
    q, demo = PRESETS["presets"]["qwen35_4b"], PRESETS["presets"]["qwen35_4b_demo"]
    assert (demo["model"], demo["revision"], demo["preset"], demo["source_kind"]) == (q["model"], q["revision"], q["preset"], q["source_kind"])
    assert sdd.preset_key_for_model(PRESETS, q["model"], q["revision"]) == "qwen35_4b"
    assert PRESETS["presets"]["qwen35_4b"]["preset"]["presence_penalty"] == 1.5
    assert PRESETS["presets"]["glm46vflash"]["preset"]["repetition_penalty"] == 1.1
    assert "presence_penalty" not in PRESETS["presets"]["glm46vflash"]["preset"]
    for key in ("gemma4e4b", "nemotron3nano4b", "llama8b"):
        assert not PENALTY_KEYS & set(PRESETS["presets"][key]["preset"]), key
    for key in ("nemotron3nano4b", "llama8b"):
        assert "top_k" not in PRESETS["presets"][key]["preset"], key


def test_escalation_configs_chain_from_the_rung_before_them():
    """8192 <- the base trial (4096); 16384 <- the 8192 rung (the ceiling, 1% at-cap trigger; the
    rung LOGIC lands later — only the configs exist here). Every TAG, every trial IT renders (the campaign's two; the three
    seed-replicate TAGs' one — `n_trials_by_view`)."""
    for tag in TAGS:
        key = msc.key_of(tag)
        for trial in range(msc.n_trials_of_tag(PRESETS, tag)):
            base = yaml.safe_load(msc.output_path(tag, trial, False).read_text())
            esc = yaml.safe_load(msc.output_path(tag, trial, True).read_text())          # the bool API = the 8192 rung
            esc16 = yaml.safe_load(msc.output_path(tag, trial, 16384).read_text())
            assert msc.output_path(tag, trial, True) == msc.output_path(tag, trial, 8192)
            assert "escalated_from" not in base["sampled_decode"]
            assert esc["sampled_decode"]["escalated_from"] == {"kind": "static", "cap": 4096, "private_run": base["private_run"]}
            assert esc16["sampled_decode"]["escalated_from"] == {"kind": "static", "cap": 8192, "private_run": esc["private_run"]}
            assert esc["private_run"] == base["private_run"] + "_cap8192" and esc16["private_run"] == base["private_run"] + "_cap16384"
            assert esc["sampled_decode"]["static_cap"] == 8192 == esc["attacks"]["static_max_new_tokens"]
            assert esc16["sampled_decode"]["static_cap"] == 16384 == esc16["attacks"]["static_max_new_tokens"]
            assert base["sampled_decode"]["static_cap"] == 4096 == base["attacks"]["static_max_new_tokens"]
            assert base["sampled_decode"]["controls_cap"] == esc["sampled_decode"]["controls_cap"] == esc16["sampled_decode"]["controls_cap"] == 512
            # a rung's sampler keys are the base's (the decoder is not the moving part of a rung); the preset is the KEY's
            samp = lambda c: {k: v for k, v in c["sampled_decode"].items() if k in msc.ALLOWED_SAMPLER_KEYS}
            assert samp(esc) == samp(esc16) == samp(base) == PINNED[key]
            # the 16384 header says what it is chained from
            assert "names the 8192 rung " + esc["private_run"] in msc.output_path(tag, trial, 16384).read_text()


def test_it_view_configs_serve_the_roster_cells_under_the_models_preset():
    """The instruct-direct views: TAG <key>_it, the `_it` wave config (none + the KEY's roster `it` cells: the two itdoc
    objects, the batch-64 control LoRA and LoRA subtraction where the roster declares them (qwen35_4b and gemma4e4b),
    then the prompted safety and constitution baselines LAST, with the nemotron / GLM grammar), the KEY's preset, the
    `_it` Petri manifest paired by the sbatch. The roster is the ONE source: the prompted cells are served (config, manifest) and never
    staged (bank view, pins); the batch-64 pair is staged under its own itdoc64__ / derived_itdoc64__ ids beside the frozen
    itdoc__ objects (never a repoint), and the generated headers carry the pair's object pins the way the base pins are."""
    sys.path.insert(0, str(F22))
    import f22_roster as fr
    for key in ("gemma4e4b", "qwen35_4b"):
        tag = f"{key}_it"
        cells = it_cells(key)
        served = fr.served_cells(ROSTER, key, "it")
        assert list(served) == cells[1:] and served["prompted__safety_refusal"] is None and served["prompted__si_constitution"] is None
        assert cells[-2:] == ["prompted__safety_refusal", "prompted__si_constitution"]     # the prompted baselines follow every object
        staged = fr.staged_cells(ROSTER, key, "it")
        assert not any(c.startswith("prompted__") for c in staged)
        if key in ITDOC64_KEYS:
            assert len(staged) == 4 and {c: staged[c] for c in ITDOC64_CELLS} == ITDOC64_CELLS
            assert {sid: PINS[key][sid]["object_sha256"] for sid in ITDOC64_CELLS.values()} == ITDOC64_OBJECT_SHA256[key]
            assert staged["constv2-mix-full__lora_r4_ce"] == "itdoc__constv2-mix-full__lora_r4_ce"      # the constitution parent stays batch-8
        else:
            assert len(staged) == 2 and not any(c in staged for c in ITDOC64_CELLS)
        wave = yaml.safe_load(msc.wave_config_path(tag).read_text())
        assert wave["explicit_cells"] == cells and wave["private_run"] == f"sdfbase_full22_{tag}"
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{tag}.yaml").read_text())["petri"]
        # the two prompted legs, in roster order (safety, then the constitution baseline of 2026-09-12), each H8-hashed
        assert man["cells"][-2:] == ["prompted__safety", "prompted__constitution"] and man["prompted_prefix_hashes"]["safety"] == "ac3a58f61d53cf24"
        assert man["prompted_prefix_hashes"]["constitution"] == "21e06f482a6559b3"
        for trial in (0, 1):
            for esc in (False, 8192, 16384):
                cfg = yaml.safe_load(msc.output_path(tag, trial, esc).read_text())
                assert cfg["explicit_cells"] == cells and cfg["private_run"].startswith(f"sdfbase_full22_{tag}_sampled_t{trial}")
                text = msc.output_path(tag, trial, esc).read_text()
                for cid, sid in ITDOC64_CELLS.items():      # the header pins the pair's objects where the roster declares them, and only there
                    assert (f"<- {sid}\n" in text) is (key in ITDOC64_KEYS), (tag, trial, esc, cid)
                    assert (f"object_sha256 {ITDOC64_OBJECT_SHA256[key][sid]}\n" in text) is (key in ITDOC64_KEYS), (tag, trial, esc, sid)
                assert {k: v for k, v in cfg["sampled_decode"].items() if k in msc.ALLOWED_SAMPLER_KEYS} == PINNED[key]
                assert cfg["sampled_decode"]["source"] == PRESETS["presets"][key]["source"]
                assert cfg["model"] == ROSTER["targets"][key]["model"] and cfg["defenses_dir"] == f"out/full22/{key}/cells_view/it"
    # the roster's main views did not gain the prompted cell (gemma / qwen main serve the FITTED safety SI; the sampled
    # main-view jobs are live on the unchanged configs)
    for key in ("gemma4e4b", "qwen35_4b"):
        assert "prompted__safety_refusal" not in fr.served_cells(ROSTER, key, "main")


def test_generator_refuses_a_non_sampler_key_a_bad_source_kind_and_a_pin_mismatch(tmp_path):
    import copy
    bad = copy.deepcopy(PRESETS)
    bad["presets"]["gemma4e4b"]["preset"]["n"] = 2
    f = tmp_path / "p.yaml"; f.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="non-sampler keys"):
        msc.load_presets(f)
    bad = copy.deepcopy(PRESETS); bad["presets"]["gemma4e4b"]["source_kind"] = "guess"
    f.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="source_kind"):
        msc.load_presets(f)
    # a vendor recommendation must record the shipped file's values beside it
    bad = copy.deepcopy(PRESETS); del bad["presets"]["nemotron3nano4b"]["shipped_generation_config"]
    f.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="shipped_generation_config"):
        msc.load_presets(f)
    bad = copy.deepcopy(PRESETS); bad["presets"]["qwen35_4b"]["revision"] = "0" * 40
    with pytest.raises(SystemExit, match="roster serves"):
        msc.check_pins(bad, ROSTER)


# --- 2: the sbatch + the helpers -------------------------------------------------------------------

SHELL = [SAMPLED / "target_sampled.sbatch", SAMPLED / "common_sampled.sh", SAMPLED / "submit_sampled.sh"]


@pytest.mark.parametrize("script", SHELL, ids=lambda p: p.name)
def test_shell_scripts_parse_and_carry_the_wave_rules(script):
    rc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    src = script.read_text()
    for line in src.splitlines():
        if "--export=" in line:
            assert "never" in line.lower(), line    # only the rule's own statement may name the flag
        # CUDA_VISIBLE_DEVICES is READ (TARGET_GPU="$CUDA_VISIBLE_DEVICES"), never assigned, outside common.sh's own patterns
        if re.search(r"(?<![\w$\"{])CUDA_VISIBLE_DEVICES=", line):
            pytest.fail(f"{script.name} assigns CUDA_VISIBLE_DEVICES: {line}")
        if "scancel" in line.lower():
            low = line.lower()
            assert "by name" in low or "--name" in low or " -n " in low, line   # never a bare scancel
    assert "--mem=0" not in src
    if script.suffix == ".sbatch":
        for m in ("#SBATCH --partition=general", "#SBATCH --qos=high", "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=32",
                  "#SBATCH --mem=200G", "#SBATCH --time=", "#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out",
                  "#SBATCH --open-mode=append", "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120"):
            assert m in src, m
        assert "positional only" in src and "never --export=" in src
        assert "f22_stop_requested" in src and "f22s_stop_requested" in src
        assert "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in src and "f22_result" in src
        assert 'TARGET_GPU="$CUDA_VISIBLE_DEVICES"' in src
        mem = [l for l in src.splitlines() if l.startswith("#SBATCH --mem=")][0]
        assert int(mem.split("=")[1].rstrip("G")) <= 200
        # sourced from the WAVE CHECKOUT on either cluster, positionally
        assert '|| source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/sampled/common_sampled.sh"' in src
        assert 'source "/workspace' not in src
        assert 'KEY="${1:?key}"; PHASE="${2:?phase}"; WAVE_SHA="${3:?wave sha}"; NSHARDS="${4:-9}"; CELLS_ARG="${5:-}"; VIEW="${6:-main}"' in src
        # the view (lane F) rides as the 6th positional into the resolver; KEY stays the bare model key everywhere else
        assert 'f22s_resolve_config "$KEY" "$TRIAL" "$VIEW"' in src and "[cells=all] [view=main]" in src
        assert src.count('--model-key "$KEY"') == 0 and 'f22s_rung_window' in src    # --model-key is common.sh f22_standup's ($KEY)
    if script.name == "common_sampled.sh":
        assert 'source "$F22S_HERE/../common.sh"' in src   # reused, never copied
        for fn in ("f22_init", "f22_resolve_config", "f22_bank_view", "f22_standup", "f22_queue_launch", "f22_queue_wait", "f22_positive_control"):
            assert f"{fn}()" not in src, f"{fn} redefined in common_sampled.sh (must reuse common.sh's)"


def test_sbatch_phases_request_the_contracts_geometry_and_flags():
    src = (SAMPLED / "target_sampled.sbatch").read_text()
    common = (SAMPLED / "common_sampled.sh").read_text()
    # static base: the wave's 4864 x S window — or the VIEW's window when the roster sets one (llama8b/fullconst 45,056 for
    # every standup; llama8b/shortconst its PER-LEG static window 10,496, 2026-09-12 — f22s_base_window reads the roster block
    # the sbatch and the printer share); static esc: the same eval at the rung window (view-aware, common_sampled.sh)
    assert 'f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$(f22s_base_window static)" || { RC=1; exit 1; }' in src
    assert 'f22_window_check "$SERVE_DIR/static" "$(f22s_base_window static)" static "$( [ -n "$VIEW_WINDOW" ] || echo base )"' in src
    # every standup that can serve a prompted cell puts the SI text's FULL sha256 on the record beside the routing table (the
    # constitution-SI arm, 2026-09-12: common.sh f22_record_si_prefix, also called by f22_standup itself)
    for sub in ("static", "static_esc", "controls_512", "controls_${RUNG}"):
        assert f'f22s_record_si_prefix "$SERVE_DIR/{sub}" || {{ RC=1; exit 1; }}' in src, sub
    assert "f22s_record_si_prefix(){ f22_record_si_prefix" in common and "f22s_base_window(){" in common
    assert "gsm8k_t[01])       PHASE_KIND=gsm8k" in common and 'f22s_view_has_leg capability' in src
    assert 'f22s_view_has_leg controls || f22_fatal' in src        # no over-refusal leg -> no controls ladder, ever
    assert "16384) if [ \"$VIEW_WINDOW\" -ge 51200 ]; then echo \"$VIEW_WINDOW\"; else echo 51200; fi;;" in common
    assert 'f22_standup attacks "$SERVE_DIR/static_esc" "" families_served:short "$(f22s_rung_window 8192)" || { RC=1; exit 1; }' in src
    # controls: 512 at capability:short 2048 with --max-logprobs from the driver (a view's controls window when the roster sets
    # one: fullconst 45,056, shortconst 6,656); the rungs at 4096 / 10240
    assert 'f22_standup attacks "$SERVE_DIR/controls_512" "" capability:short "$(f22s_base_window controls)" "" "$CTL_TOP_LOGPROBS" ||' in src
    # the sampled constitution-SI assertions, kept beside these (the two window mechanisms coexist: common_sampled.sh)
    assert "f22_standup_extra_reset --enable-log-requests" in src and src.count('f22s_record_resolved_sampler "$SERVE_DIR/') == 4   # readback per phase
    assert "F22_STANDUP_EXTRA=(" not in src        # an inherited launch-time hook survives (common.sh f22_standup_extra_reset, 2026-09-16)
    assert 'f22s_detect_long_si "${CELLS[@]}"' in src and "F22S_LONG_SI_CELLS=(prompted__si_constitution prompted__si_irrelevant)" in common
    assert 'f22_standup attacks "$SERVE_DIR/controls_${RUNG}" "" capability:short "$WINDOW" "" "$CTL_TOP_LOGPROBS" ||' in src
    assert 'WINDOW=$(f22s_rung_window "$RUNG")' in src
    # per-model rung windows: nemotron3nano4b 4864 / 8704 (its hybrid KV cache held 165 slots at 4096 < identity S 192, job
    # 84848); every other key 4096 / 10240 as DESIGN.md
    import subprocess
    # ... and under F22S_LONG_SI=1 (a long-SI prompted cell among the job's cells) the max-prompt-sized table, model-independent:
    # 8192 / 14848 / 23040 (the constitution-SI arm, 2026-09-12; the derivation in common_sampled.sh)
    for key, rung, want in (("nemotron3nano4b", 2048, "4864"), ("nemotron3nano4b", 8192, "8704"), ("qwen35_4b", 2048, "4096"),
                            ("gemma4e4b", 8192, "10240"), ("glm46vflash", 8192, "10240"),
                            ("gemma4e4b", "2048 long", "8192"), ("nemotron3nano4b", "8192 long", "14848"), ("qwen35_4b", "16384 long", "23040")):
        long_si = "F22S_LONG_SI=1; " if isinstance(rung, str) and rung.endswith(" long") else ""
        rung = int(str(rung).split()[0])
        got = subprocess.run(["bash", "-c", f'f22_fatal(){{ echo FATAL "$@" >&2; exit 1; }}; KEY={key}; {long_si}'
                              f'eval "$(sed -n "/^f22s_rung_window()/,/^}}/p" {SAMPLED}/common_sampled.sh)"; f22s_rung_window {rung}'],
                             capture_output=True, text=True)
        assert got.stdout.strip() == want, (key, rung, got.stdout, got.stderr)
    assert "capability:short 2048 256" not in src    # the launcher's strict identity: only the window is passed
    assert "MMLU_TOP_LOGPROBS" in src               # ONE number for the server cap and the driver request
    # the drivers: the contract's exact flags
    assert "scripts/jailbreak_run.py generate" in src and '--confirmatory --trial "$TRIAL" --shard "$s" --nshards "$NSHARDS"' in src
    assert '--target-engine served --target-proxy-url "$PROXY_URL"' in src and '--target-routing-table "$TABLE" --target-max-model-len "$TARGET_L"' in src
    assert '"$F22S_DIR/escalate_static.py"' in src and '--config "$CFG_ESC" --attack "$fam" --defense "$cell" --private-root "$PRIVATE_ROOT"' in src
    assert '--proxy-url "$PROXY_URL" --routing-table "$TABLE" --max-model-len "$TARGET_L" --concurrency "$F22S_ESC_CONC"' in src
    assert "F22S_ESC_CONC=32" in common and "MAX_DRIVERS=$(( (TARGET_S + F22S_ESC_CONC - 1) / F22S_ESC_CONC ))" in src
    assert "scripts/served_capability_run.py generate" in src and '--controls "$CTL_LIST" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS"' in src
    assert '--controls-cap "$RUNG" --escalate-from-root "$PREV_ROOT" --floor-sample "$F22S_FLOOR_SAMPLE_K"' in src
    assert "F22S_FLOOR_SAMPLE_K=200" in common and "F22S_CONTROLS=(xstest or_bench_hard_1k)" in common
    assert "F22S_FAMILIES=(pasttense codeattack renellm grok420)" in common
    # the static drivers default to the server width S (F22S_DRIVER_CAP caps it;
    # the campaign's 96 is gone from the source) — tests/test_seedrep_ws1c_sbatch_flags.py evaluates the arithmetic
    assert 'DRIVER_CAP="${F22S_DRIVER_CAP:-$TARGET_S}"' in src and "DRIVER_CAP=96" not in src
    assert "MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))" in src
    assert "MAX_DRIVERS=$((NCELLS + 1))" in src and "MAX_DRIVERS=$((NRUNG + 1))" in src
    # the stores: new names, rung roots, never the greedy roots
    assert 'PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"' in src
    assert 'ROOT_OF(){ echo "$WAVE_ROOT/sampled/controls_cap$1/${TAG}_t${TRIAL}"; }' in src
    assert '"$WAVE_ROOT/controls/' not in src and 'transcripts_results/sdfbase_full22_$' not in src
    assert 'LADDER_DIR="$WAVE_ROOT/sampled/ladders/${TAG}_t${TRIAL}"' in src
    assert 'f22s_write_ladder controls "$LADDER_DIR/ladder.json"' in src and 'f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"' in src
    # the at-cap gates refuse rather than guess; the escalation phase gates BEFORE any standup
    assert "at_cap_static.py" in common and "at_cap_controls.py" in common and "return 2" in common
    assert "REFUSING: the base run's at-cap count" in src and "REFUSING: the ladder gate at rung" in src   # the gate
    assert src.index("no standup") < src.index('f22_standup attacks "$SERVE_DIR/static_esc"')
    # renellm's frozen corpus, exactly as the wave
    assert "SISTEER_HELDOUT_CORPUS_DIR" in src and "heldout_corpora" in src
    # one geometry at a time: every standup is stopped before the next (static base, static esc, the static 16,384 rung
    # [gate-based], controls 512 + rungs, the extend's static / controls / gsm8k 16384 standups, the capability
    # battery's gsm8k thinking-ON + mmlu, the gsm8k readout phase [llama8b] = eleven)
    assert src.count("f22_standup attacks") == 11 and src.count("f22_standup_stop") == 11
    # the sampled config check runs in the job before the bank view
    assert ('f22_driver_venv; f22_serving_venv; f22s_assert_vllm_mamba_kernels || f22_fatal "nemotron fused Mamba2 kernels '
            'missing in the serving venv (see log)"; f22s_check_config "$MPY"; f22_bank_view "$MPY"') in src


def _bash(snippet: str, **env) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; {snippet}'], capture_output=True, text=True,
                          cwd=REPO, env={**os.environ, **env})


@pytest.mark.parametrize("phase,want", [
    ("static_base_t0", "static_base 0 pasttense codeattack renellm grok420 static_base_t0"),
    ("static_base_t1:renellm,grok420", "static_base 1 renellm grok420 static_base_t1_renellm-grok420"),
    ("static_esc_t1", "static_esc 1 pasttense codeattack renellm grok420 static_esc_t1"),
    ("static_esc_t0:pasttense", "static_esc 0 pasttense static_esc_t0_pasttense"),
    ("controls_t0", "controls 0  controls_t0"),
    ("controls_t1", "controls 1  controls_t1"),
])
def test_phase_grammar_parses(phase, want):
    rc = _bash(f'f22s_parse_phase "{phase}"; echo "$PHASE_KIND $TRIAL ${{FAMS[*]:-}} $PHASE_TAG"')
    assert rc.returncode == 0 and rc.stdout.strip() == want, rc.stdout + rc.stderr


@pytest.mark.parametrize("phase,want", [
    # the controls LADDER over an existing COMPLETE 512 root (2026-09-13): no 512 pass; an optional `:<ceiling>` is the LAST rung this
    # job climbs (the rungs above stay a priced decision for ladder_extend_t<T>; 16k is stopped and priced, never
    # run in-job); no ceiling = the full ladder, exactly as controls_t<T>
    ("controls_esc_t0", "controls 1 0 controls_esc_t0 512 2048 8192 16384"),
    ("controls_esc_t1:8192", "controls 1 1 controls_esc_t1_8192 512 2048 8192"),
    ("controls_esc_t0:2048", "controls 1 0 controls_esc_t0_2048 512 2048"),
    ("controls_esc_t1:16384", "controls 1 1 controls_esc_t1_16384 512 2048 8192 16384"),
    ("controls_t0", "controls  0 controls_t0 512 2048 8192 16384"),                 # the plain phase: the flag empty
])
def test_controls_esc_phase_parses_with_an_optional_ceiling_rung(phase, want):
    # ONE dispatch arm: PHASE_KIND stays `controls` (the sbatch's chain and the review tests' marker), the mode is F22S_CONTROLS_ESC
    rc = _bash(f'f22s_parse_phase "{phase}"; echo "$PHASE_KIND ${{F22S_CONTROLS_ESC:-}} $TRIAL $PHASE_TAG ${{F22S_CONTROLS_RUNGS[*]}}"')
    assert rc.returncode == 0 and rc.stdout.strip() == want, rc.stdout + rc.stderr


@pytest.mark.parametrize("phase,above", [
    # a `:<ceiling>` BELOW the campaign ceiling records the real next rung: the post-loop gate then marks a pair still > 1 % at the
    # top rung ESCALATING to it (due for a later controls_esc_t<T>:<next> job) — never exit_ceiling, never a refusal, never the
    # 'transport_refused: no hop timeouts supplied' the live endpoint path produced (llama8b_it t0/t1 under :8192, 2026-09-14)
    ("controls_esc_t1:8192", "16384"), ("controls_esc_t0:2048", "8192"), ("controls_esc_t1:16384", ""), ("controls_esc_t0", ""), ("controls_t0", ""),
])
def test_controls_esc_ceiling_below_the_campaign_ceiling_records_the_rung_above(phase, above):
    rc = _bash(f'f22s_parse_phase "{phase}"; echo "[${{F22S_CONTROLS_CEILING_ABOVE:-}}]"')
    assert rc.returncode == 0 and rc.stdout.strip() == f"[{above}]", rc.stdout + rc.stderr


@pytest.mark.parametrize("phase", ["controls_esc_t0:4096", "controls_esc_t1:pasttense", "controls_esc_t0:2048,8192", "controls_esc_t2"])
def test_controls_esc_refuses_a_ceiling_that_is_not_a_ladder_rung(phase):
    rc = _bash(f'f22s_parse_phase "{phase}"')
    out = rc.stdout + rc.stderr                                    # f22_fatal logs through f22_log (stdout) before exiting
    assert rc.returncode != 0 and ("ceiling rung" in out or "unknown phase" in out), out


def test_standup_endpoint_prefers_the_live_record_then_the_ready_copy_and_never_invents_one(tmp_path):
    """mt_serving_up.py UNLINKS endpoint.json at teardown, and the controls loop gates the previous rung's stores AFTER
    f22_standup_stop — the llama8b controls_t<T> jobs read the unlinked file and every (cell, control) pair left the ladder
    `transport_refused` on absent evidence. The helper returns the live file while the server is up, else the byte-identical
    READY copy f22_standup keeps, else the live path (so the gate still refuses on an absent record — never a guessed hop)."""
    d = tmp_path / "controls_512"; d.mkdir()
    assert _bash(f'f22s_standup_endpoint "{d}"').stdout.strip() == str(d / "endpoint.json")            # nothing: the live path, refused downstream
    (d / "endpoint.recorded.json").write_text("{}")
    assert _bash(f'f22s_standup_endpoint "{d}"').stdout.strip() == str(d / "endpoint.recorded.json")   # torn down: the READY copy
    (d / "endpoint.json").write_text("{}")
    assert _bash(f'f22s_standup_endpoint "{d}"').stdout.strip() == str(d / "endpoint.json")            # up: the live record


def test_prev_endpoint_capture_is_exactly_the_path_even_when_the_record_is_reconstructed(tmp_path):
    """Job 100037 (2026-09-13): f22s_prev_endpoint RECONSTRUCTED the 512 standup's record and logged it — to STDOUT, inside the caller's
    EP_512=$(…) capture, so the variable was the log line + a newline + the path, ladder_gate.py --endpoint read a non-path, and every
    pair left the ladder `transport_refused` ("no hop timeouts supplied") with the record sitting right there. The note now goes to
    stderr (the job log merges it); the capture is ONE line, the path; the live and recorded branches never logged and stay one line."""
    wave = tmp_path / "wave"; sd = wave / "sampled" / "serving" / "controls_t0_llama8b_it_95768" / "controls_512"; sd.mkdir(parents=True)
    (sd / "standup_jobs.jsonl").write_text('{"job_id": 95768}\n')
    fake_py = tmp_path / "fake_mpy"; fake_py.write_text("#!/bin/bash\n# stands in for $MPY reconstruct_prev_endpoint.py --standup-dir D: writes the record, prints its path\n"
                                                       'd=${@: -1}; echo "{}" > "$d/endpoint.reconstructed.json"; echo "$d/endpoint.reconstructed.json"\n'); fake_py.chmod(0o755)
    env = dict(WAVE_ROOT=str(wave), TAG="llama8b_it", MPY=str(fake_py), REPO_DIR=str(tmp_path), F22S_DIR=str(tmp_path), F22S_LOG=str(tmp_path / "log"))
    rc = _bash('f22_log(){ echo "[t] $*"; }; out=$(f22s_prev_endpoint controls_t0 controls_512); echo "LINES=$(printf "%s\n" "$out" | wc -l)"; echo "OUT=$out"', **env)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert "LINES=1" in rc.stdout and f"OUT={sd}/endpoint.reconstructed.json" in rc.stdout, rc.stdout + rc.stderr
    assert "prev endpoint RECONSTRUCTED" in rc.stderr and "RECONSTRUCTED" not in rc.stdout.split("OUT=")[1]
    # the recorded-copy branch: one line, no note
    (sd / "endpoint.recorded.json").write_text("{}")
    rc = _bash('out=$(f22s_prev_endpoint controls_t0 controls_512); echo "LINES=$(printf "%s\n" "$out" | wc -l)"; echo "OUT=$out"', **env)
    assert "LINES=1" in rc.stdout and f"OUT={sd}/endpoint.recorded.json" in rc.stdout, rc.stdout + rc.stderr


def test_the_controls_ladder_gates_through_a_surviving_endpoint_record_and_controls_esc_resumes_a_complete_512_root():
    src = (SAMPLED / "target_sampled.sbatch").read_text()
    assert 'elif [ "$PHASE_KIND" = controls ]; then' in src and 'elif [ "$PHASE_KIND" = controls_esc ]' not in src   # one arm, the mode a flag
    assert 'if [ -z "${F22S_CONTROLS_ESC:-}" ]; then' in src
    # the gate never reads the previous rung's live endpoint.json after its server is stopped
    assert 'f22s_gate_pairs "$PREV_RUNG" "$PREV_ROOT" "$EP_PREV" "$RUNG"' in src
    assert 'f22s_gate_pairs "$PREV_RUNG" "$PREV_ROOT" "$SERVE_DIR/controls_${PREV_RUNG}/endpoint.json" "$RUNG"' not in src
    assert 'if [ "$PREV_RUNG" = 512 ]; then EP_PREV="$EP_512"; else EP_PREV="$(f22s_standup_endpoint "$SERVE_DIR/controls_${PREV_RUNG}")"; fi' in src
    # the POST-loop gate (the top rung ran) reads the top rung's endpoint through the same surviving record — the live path made every
    # pair still > 1 % at the top read 'transport_refused' (2026-09-14) — and under a `:<ceiling>` below 16,384 it gates toward the
    # real next rung (ESCALATE recorded, nothing climbed) instead of treating the truncated list's last rung as the ceiling
    assert 'f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$SERVE_DIR/controls_${LAST_RAN}/endpoint.json" ""' not in src
    assert 'EP_LAST="$(f22s_standup_endpoint "$SERVE_DIR/controls_${LAST_RAN}")"' in src
    assert 'f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$EP_LAST" "$F22S_CONTROLS_CEILING_ABOVE"' in src
    assert 'f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$EP_LAST" ""' in src
    # controls_t<T>: its own 512 standup's record through the helper (the READY copy once the server is stopped)
    assert 'EP_512="$(f22s_standup_endpoint "$SERVE_DIR/controls_512")"' in src
    # controls_esc_t<T>: every job cell's 512 store must be COMPLETE, the 512 record comes from the earlier job (live / recorded /
    # reconstructed), and the phase never stands the 512 server up
    assert 'EP_512=$(f22s_prev_endpoint "controls_t$TRIAL" controls_512) || f22_fatal "controls_esc_t$TRIAL: no endpoint record' in src
    assert 'no COMPLETE 512 store for cell $cell under $(ROOT_OF 512)' in src and "gen_stats.json" in src
    esc_start = src.index('  else' + chr(10) + '    # ---- controls_esc_t<T>')
    esc = src[esc_start:src.index('  fi' + chr(10), esc_start)]
    assert "f22_standup " not in esc and "f22_queue_launch" not in esc
    # the 512 pass is controls_t<T>'s alone
    ctl = src[src.index('  if [ -z "${F22S_CONTROLS_ESC:-}" ]; then'):esc_start]
    assert 'f22_standup attacks "$SERVE_DIR/controls_512"' in ctl and 'f22_standup_stop' in ctl


@pytest.mark.parametrize("phase", ["controls_t1:xstest", "static_base_t2", "static_base_t0:bogus", "static_t0", "controls", ""])
def test_phase_grammar_refuses(phase):
    rc = _bash(f'f22s_parse_phase "{phase}"; echo "reached"')
    assert "FATAL" in rc.stdout + rc.stderr and "reached" not in rc.stdout, phase


def test_sampled_resolver_reuses_the_wave_resolver_and_keeps_the_wave_manifest():
    rc = _bash('f22s_resolve_config qwen35_4b 1 >/dev/null; echo "$KEY|$VIEW|$TAG|$WAVE_CFG|$CFG|$CFG_ESC|$MANIFEST|$PRIVATE_RUN_BASE|$PRIVATE_RUN_ESC|${GPU_MEM_UTIL:-unset}"')
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert rc.stdout.strip() == ("qwen35_4b|main|qwen35_4b|configs/sdfbase_full22_qwen35_4b.yaml|"
                                 "configs/sdfbase_full22_qwen35_4b_sampled_t1.yaml|configs/sdfbase_full22_qwen35_4b_sampled_t1_cap8192.yaml|"
                                 "configs/capeval/manifests/petri_full22_qwen35_4b.yaml|sdfbase_full22_qwen35_4b_sampled_t1|"
                                 "sdfbase_full22_qwen35_4b_sampled_t1_cap8192|unset")
    # the instruct-direct view (lane F): TAG = <key>_it; the `_it` wave config, sampled configs, Petri manifest and private_runs
    # follow the TAG; KEY stays the bare model key (budgets / hybrid facts / rung windows / --model-key are per model)
    rc = _bash('f22s_resolve_config gemma4e4b 0 it >/dev/null; echo "$KEY|$VIEW|$TAG|$WAVE_CFG|$CFG|$CFG_ESC|$MANIFEST|$PRIVATE_RUN_BASE|$PRIVATE_RUN_ESC"; '
               'echo "$(f22s_rung_window 2048) $(f22s_rung_window 8192)"')
    assert rc.returncode == 0, rc.stdout + rc.stderr
    assert rc.stdout.strip().splitlines() == [
        "gemma4e4b|it|gemma4e4b_it|configs/sdfbase_full22_gemma4e4b_it.yaml|configs/sdfbase_full22_gemma4e4b_it_sampled_t0.yaml|"
        "configs/sdfbase_full22_gemma4e4b_it_sampled_t0_cap8192.yaml|configs/capeval/manifests/petri_full22_gemma4e4b_it.yaml|"
        "sdfbase_full22_gemma4e4b_it_sampled_t0|sdfbase_full22_gemma4e4b_it_sampled_t0_cap8192",
        "4096 10240"]
    # the default view is main (the five-positional call of the live jobs is unchanged); a view outside the scope refuses
    rc = _bash('f22s_resolve_config qwen35_4b 1 >/dev/null; echo "$VIEW|$TAG"')
    assert rc.stdout.strip() == "main|qwen35_4b"
    rc = _bash('f22s_resolve_config nemotron3nano4b 0 halfdose; echo reached')   # f22_fatal logs to stdout
    assert "FATAL" in rc.stdout + rc.stderr and "not in the sampled scope" in rc.stdout + rc.stderr and "reached" not in rc.stdout
    # the job-side config check holds on the `_it` configs too (they are the `_it` wave config as data)
    rc = _bash(f'f22s_resolve_config qwen35_4b 1 it >/dev/null; TRIAL=1; f22s_check_config "{PY}"')
    assert rc.returncode == 0 and "sampled configs OK" in rc.stdout and "qwen35_4b_it_sampled_t1" in rc.stdout, rc.stdout + rc.stderr
    # glm keeps the wave's GPU_MEM_UTIL belt through the reused resolver
    rc = _bash('f22s_resolve_config glm46vflash 0 >/dev/null; echo "${GPU_MEM_UTIL:-unset}"')
    assert rc.stdout.strip() == "0.92", rc.stdout + rc.stderr
    # the job-side config check passes on the generated configs, and refuses a trial mismatch
    rc = _bash(f'f22s_resolve_config gemma4e4b 0 >/dev/null; TRIAL=0; f22s_check_config "{PY}"')
    assert rc.returncode == 0 and "sampled configs OK" in rc.stdout, rc.stdout + rc.stderr
    rc = _bash(f'f22s_resolve_config gemma4e4b 0 >/dev/null; TRIAL=1; f22s_check_config "{PY}"; echo reached')
    assert "FATAL" in rc.stdout + rc.stderr and "reached" not in rc.stdout


def test_the_scope_is_one_list_the_shell_gate_asks_the_generator():
    """common_sampled.sh used to hold its own `case main|it` view list beside the generator's SAMPLED_VIEWS; two lists drift
    (the 24 dose views would have had to land in both). Now the shell ASKS the generator — `make_sampled_configs.py
    --is-in-scope <key> <view>`: 0 in scope, 3 not, anything else = the query itself failed — driven here for EVERY (key, view)
    of SAMPLED_VIEWS and for out-of-scope pairs, under the system python3 (the sbatch resolves the config before the driver venv
    exists) and under an explicit MPY, so the shell gate and the python scope cannot disagree. A broken interpreter is reported
    as a FAILED QUERY, never as "out of scope". Every other consumer imports the list rather than restating it."""
    common = (SAMPLED / "common_sampled.sh").read_text()
    assert 'case "$view" in main|it)' not in common and not re.search(r"main\|it\)", common), "a second view list survives in the shell"
    assert "f22s_view_in_scope()" in common and "--is-in-scope" in common and 'f22s_view_in_scope "$key" "$view"' in common
    rc = _run_py(SAMPLED / "make_sampled_configs.py", "--tags")
    assert rc.returncode == 0 and tuple(rc.stdout.split()) == tuple(msc.sampled_tags()) == TAGS
    for env in ({}, {"MPY": PY}):
        for key, views in msc.SAMPLED_VIEWS.items():
            for view in views:
                rc = _bash(f'f22s_view_in_scope {key} {view}; echo "rc=$?"', **env)
                assert rc.stdout.strip() == "rc=0", (key, view, env, rc.stdout, rc.stderr)
        for key, view in (("nemotron3nano4b", "halfdose"), ("gemma4e4b", "d0p75"), ("nemotron3nano4b", "it_d0p25"),
                          ("glm46vflash", "it"), ("llama8b", "halfdose"), ("gemma4e4b", "")):
            rc = _bash(f'f22s_view_in_scope {key} "{view}"; echo "rc=$?"', **env)
            assert rc.stdout.strip() == "rc=3" and "not in the sampled scope" in rc.stderr, (key, view, env, rc.stdout, rc.stderr)
            assert not msc.in_scope(key, view)
    # the resolver: an out-of-scope view refuses with the scope message; a broken interpreter refuses with the QUERY message
    rc = _bash('f22s_resolve_config gemma4e4b 0 d0p75; echo reached')   # a dose token no ladder minted (gemma's d0p25 is in scope)
    assert "FATAL" in rc.stdout and "not in the sampled scope" in rc.stdout and "reached" not in rc.stdout
    rc = _bash('f22s_view_in_scope gemma4e4b main; echo "rc=$?"', MPY="/nonexistent/python")
    assert rc.stdout.strip() not in ("rc=0", "rc=3"), rc.stdout
    rc = _bash('f22s_resolve_config gemma4e4b 0 main; echo reached', MPY="/nonexistent/python")
    assert "FATAL: the sampled scope query failed" in rc.stdout and "not in the sampled scope" not in rc.stdout and "reached" not in rc.stdout, rc.stdout + rc.stderr
    # the consumers import the ONE list: the printer, the grader (its --target choices), the grade planner
    import grade_static_sampled as gss
    assert gss.KEYS == tuple(msc.sampled_tags()) and ss.TARGETS_ACCEPTED == KEYS + ss.VIEW_TAGS and set(ss.VIEW_TAGS) == set(TAGS) - set(KEYS)
    assert "import make_sampled_configs as msc" in (SAMPLED / "grade_sampled.sh").read_text()
    for name in ("grade_static_sampled.py", "submit_sampled.py", "grade_sampled.sh"):
        assert '"gemma4e4b_it", "qwen35_4b_it")' not in (SAMPLED / name).read_text(), f"{name} restates the scope"


def test_the_dose_views_run_the_campaigns_ladder_not_a_per_tag_base_cap():
    """NO `--base-cap` knob (8,192 from the outset). The instrument the dose views run is the campaign's
    4096 -> 8192 -> 16384 static ladder — base 4096 at the 4864 `families_served:short` window, the 8192 rung at
    f22s_rung_window 8192 — so the frontier's dose points are measured by the SAME instrument as the `main` / `it` / re-served
    `d1` points, and the paper reads ONE root shape everywhere (the `_cap8192` final rung). Pinned: one module-wide
    base cap in the generator, the shell's cap constants equal to it, every base config at 4096 and every TAG with the same
    three files per trial, and the sbatch's base standup at the short window — the geometry under which an 8192 base would be a
    served clamp, which is WHY the knob was not added."""
    common = (SAMPLED / "common_sampled.sh").read_text(); sbatch = (SAMPLED / "target_sampled.sbatch").read_text()
    assert msc.STATIC_CAP_BASE == 4096 and msc.STATIC_CAP_ESC == 8192 and msc.STATIC_RUNGS == (8192, 16384) and msc.CONTROLS_CAP == 512
    assert "F22S_STATIC_CAP=4096; F22S_STATIC_CAP_ESC=8192" in common and "F22S_CEILING=16384; F22S_STATIC_CAP_EXT=16384" in common
    # the base standup at the VIEW's base window — f22s_base_window static = the 4864 short window for main / it / dose views
    # (the llama8b shortconst view alone widens it for its SI prefix, roster serving.windows)
    # the base standup's window is f22s_base_window: the short 4,864 window unless a long-SI cell (prompted__si_constitution,
    # ~6k prompt tokens) is served, when it is the long-SI base window -- never a per-TAG base cap
    assert 'f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$(f22s_base_window static)" || { RC=1; exit 1; }' in sbatch
    assert 'f22_standup attacks "$SERVE_DIR/static_esc" "" families_served:short "$(f22s_rung_window 8192)"' in sbatch
    per_tag = {}
    for path, tag, trial, esc in OUTPUTS:
        cap = msc.rung_cap(esc)
        cfg = yaml.safe_load(path.read_text())
        assert cfg["attacks"]["static_max_new_tokens"] == cfg["sampled_decode"]["static_cap"] == (cap or 4096), path.name
        assert cfg["attacks"]["controls_max_new_tokens"] == cfg["sampled_decode"]["controls_cap"] == 512, path.name
        assert ("escalated_from" in cfg["sampled_decode"]) == bool(cap)
        per_tag.setdefault(tag, set()).add((trial, cap))
    # every TAG carries the same three files PER TRIAL it renders (two trials; ONE on the three seed-replicate TAGs — n_trials_by_view)
    assert set(per_tag) == set(TAGS) and all(v == {(t, c) for t in range(msc.n_trials_of_tag(PRESETS, tag)) for c in (None, 8192, 16384)} for tag, v in per_tag.items())
    assert {tag: len(v) for tag, v in per_tag.items() if tag in SEEDREP_TAGS} == {tag: 3 for tag in SEEDREP_TAGS}


def test_stop_helper_reads_the_wave_sentinel_or_the_jobs_own(tmp_path):
    (tmp_path / "wave").mkdir(); (tmp_path / "job").mkdir()
    base = f'WAVE_ROOT="{tmp_path}/wave"; JOB_STOP="{tmp_path}/job/STOP"; '
    assert _bash(base + 'f22s_stop_requested && echo yes || echo no').stdout.strip() == "no"
    (tmp_path / "job/STOP").touch()
    assert _bash(base + 'f22s_stop_requested && echo yes || echo no').stdout.strip() == "yes"
    (tmp_path / "job/STOP").unlink(); (tmp_path / "wave/STOP").touch()
    assert _bash(base + 'f22s_stop_requested && echo yes || echo no').stdout.strip() == "yes"


# --- 3: the ladder fold -----------------------------------------------------------------------------

def test_write_ladder_applies_the_exit_rule_records_the_floor_sample_and_merges(tmp_path):
    ev = tmp_path / "events.jsonl"; out = tmp_path / "ladder.json"
    rows = [
        {"kind": "controls", "cell": "none", "control": "xstest", "rung": 512, "n_at_cap": 0, "root": "/r512"},
        {"kind": "controls", "cell": "none", "control": "or_bench_hard_1k", "rung": 512, "n_at_cap": 3, "root": "/r512"},
        {"kind": "controls", "cell": "none", "control": "or_bench_hard_1k", "rung": 2048, "n_at_cap": 0, "root": "/r2048"},
        {"kind": "controls", "cell": "c1", "control": "xstest", "rung": 512, "n_at_cap": 40, "root": "/r512"},
        {"kind": "controls", "cell": "c1", "control": "xstest", "rung": 2048, "n_at_cap": 5, "root": "/r2048"},
        {"kind": "controls", "cell": "c1", "control": "xstest", "rung": 8192, "n_at_cap": 2, "root": "/r8192"},
        {"kind": "controls", "cell": "c1", "control": "or_bench_hard_1k", "rung": 512, "n_at_cap": 7, "root": "/r512"},   # no 2048 count: incomplete
    ]
    ev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc = _run_py(SAMPLED / "write_ladder.py", "--events", ev, "--out", out, "--kind", "controls", "--tag", "gemma4e4b", "--trial", 0)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    lad = json.loads(out.read_text())
    assert lad["rungs"] == [512, 2048, 8192, 16384] and lad["floor_sample_k"] == 200 and lad["tag"] == "gemma4e4b" and lad["trial"] == 0
    assert "floor sample never keeps a cell on the ladder" in lad["exit_rule"]
    c = lad["cells"]
    assert c["none"]["xstest"]["reached_zero_at"] == "512" and c["none"]["xstest"]["residual"] == 0
    assert c["none"]["or_bench_hard_1k"]["reached_zero_at"] == "2048" and c["none"]["or_bench_hard_1k"]["n_at_cap"] == {"512": 3, "2048": 0}
    # the RAW count history: with the 16,384 continuation a non-zero 8192 count is not the top any more
    assert c["c1"]["xstest"]["reached_zero_at"] == "incomplete" and c["c1"]["xstest"]["residual"] is None
    assert c["c1"]["xstest"]["left_ladder_at"] == {"status": "undecided"}     # no gate event in these rows
    assert c["c1"]["or_bench_hard_1k"]["reached_zero_at"] == "incomplete" and c["c1"]["or_bench_hard_1k"]["residual"] is None
    assert c["c1"]["xstest"]["roots"] == {"512": "/r512", "2048": "/r2048", "8192": "/r8192"}
    assert lad["summary"] == {"n_pairs": 4, "left_at": {"512": 1, "2048": 1, "8192": 0, "16384": 0}, "residual_at_top": 0, "incomplete": 2,
                              "exit_kind": {"exit_zero": 0, "exit_le_1pct_with_residual": 0, "exit_ceiling": 0, "promoted": 0, "undecided": 4},
                              "exits": {"exit_le_1pct": 0, "exit_ceiling": 0, "transport_refused": 0, "escalating": 0, "undecided": 4}}
    assert lad["extend_kind"] == "controls"   # lane G2: a fold names the extend kind it folded (the gsm8k ladder is its own file)
    # a later fold MERGES (the 2048 count for c1/or_bench arrives): the record completes, others untouched
    ev.write_text(json.dumps({"kind": "controls", "cell": "c1", "control": "or_bench_hard_1k", "rung": 2048, "n_at_cap": 0, "root": "/r2048"}) + "\n")
    rc = _run_py(SAMPLED / "write_ladder.py", "--events", ev, "--out", out, "--kind", "controls", "--tag", "gemma4e4b", "--trial", 0)
    lad2 = json.loads(out.read_text())
    assert lad2["cells"]["c1"]["or_bench_hard_1k"]["reached_zero_at"] == "2048" and lad2["cells"]["c1"]["xstest"] == lad["cells"]["c1"]["xstest"]
    assert lad2["summary"]["incomplete"] == 1
    # a different (kind, tag, trial) never merges into it
    rc = _run_py(SAMPLED / "write_ladder.py", "--events", ev, "--out", out, "--kind", "controls", "--tag", "gemma4e4b", "--trial", 1)
    assert rc.returncode != 0 and "refusing to merge" in rc.stderr
    # a torn event line refuses rather than folding a partial ladder
    ev.write_text('{"kind": "controls", "cell": "c1"\n')
    rc = _run_py(SAMPLED / "write_ladder.py", "--events", ev, "--out", tmp_path / "x.json", "--kind", "controls", "--tag", "g", "--trial", 0)
    assert rc.returncode != 0 and "torn event line" in rc.stderr


def test_write_ladder_static_reads_the_escalation_record():
    events = [
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 4096, "n_at_cap": 0},
        {"kind": "static", "attack": "renellm", "cell": "c1", "rung": 4096, "n_at_cap": 12},
        {"kind": "static_escalation", "attack": "renellm", "cell": "c1",
         "escalation": {"n_rows": 300, "n_at_cap_before": 12, "n_regenerated": 12, "n_at_cap_after": 4, "cap_from": 4096, "cap_to": 8192}, "problem": None},
        {"kind": "static", "attack": "grok420", "cell": "c1", "rung": 4096, "n_at_cap": 2},
        {"kind": "static_escalation", "attack": "grok420", "cell": "c1", "escalation": None, "problem": "escalation.json not found under /x"},
    ]
    lad = wl.fold(events, "static", wl.RUNGS["static"], None, tag="qwen35_4b", trial=1)
    assert "floor_sample_k" not in lad and lad["rungs"] == [4096, 8192, 16384]      # the 16,384 ceiling rung (round 2 P1)
    assert lad["cells"]["none"]["pasttense"]["reached_zero_at"] == "4096"
    r = lad["cells"]["c1"]["renellm"]
    # the RAW count history: with the 16,384 rung in the ladder a non-zero 8192 count is not the top any more
    assert r["n_at_cap"] == {"4096": 12, "8192": 4} and r["reached_zero_at"] == "incomplete" and r["escalation"]["n_regenerated"] == 12
    assert r["escalations"] == {"8192": r["escalation"]} and r["left_ladder_at"] == {"status": "undecided"}
    g = lad["cells"]["c1"]["grok420"]
    assert g["reached_zero_at"] == "incomplete" and g["problem"].startswith("escalation.json not found")
    # the other kind's events are SKIPPED and counted, never refused: the controls and static jobs of one
    # (target, trial) share LADDER_DIR and therefore one events.jsonl (canary 84812, 2026-09-09)
    mixed = wl.fold([{"kind": "controls", "cell": "c", "control": "xstest", "rung": 512, "n_at_cap": 1}], "static", wl.RUNGS["static"], None, tag="t", trial=0)
    assert mixed["skipped_other_kind_events"] == 1 and mixed["cells"] == {}


# --- 4: the submit printer --------------------------------------------------------------------------

def _plan(cluster_of=None):
    return ss.plan(ROSTER, list(KEYS), {**ss.DEFAULT_CLUSTER, **(cluster_of or {})}, 9)


def test_submit_plan_routes_splits_and_times_the_jobs_as_documented():
    p = _plan()
    names = [j.name for j in p["jobs"]]
    assert names == ["f22s-gemma4e4b-sb0", "f22s-gemma4e4b-sb1", "f22s-gemma4e4b-ctl0", "f22s-gemma4e4b-ctl1",
                     "f22s-qwen35_4b-sb0a", "f22s-qwen35_4b-sb0b", "f22s-qwen35_4b-sb1a", "f22s-qwen35_4b-sb1b",
                     "f22s-qwen35_4b-ctl0", "f22s-qwen35_4b-ctl1",
                     "f22s-nemotron3nano4b-sb0", "f22s-nemotron3nano4b-sb1", "f22s-nemotron3nano4b-ctl0", "f22s-nemotron3nano4b-ctl1",
                     "f22s-glm46vflash-sb0a", "f22s-glm46vflash-sb0b", "f22s-glm46vflash-sb1a", "f22s-glm46vflash-sb1b",
                     "f22s-glm46vflash-ctl0", "f22s-glm46vflash-ctl1"]
    assert [j.name for j in p["esc"]] == [f"f22s-{k}-se{t}" for k in KEYS for t in (0, 1)]
    by = {j.name: j for j in p["jobs"] + p["esc"]}
    for k in ("gemma4e4b", "qwen35_4b"):
        assert all(j.cluster == "cluster_a" for j in by.values() if j.key == k)
    for k in ("nemotron3nano4b", "glm46vflash"):
        assert all(j.cluster == "cluster_b" for j in by.values() if j.key == k)
    assert by["f22s-qwen35_4b-sb0a"].phase == "static_base_t0:pasttense,codeattack" and by["f22s-glm46vflash-sb1b"].phase == "static_base_t1:renellm,grok420"
    assert by["f22s-gemma4e4b-sb1"].phase == "static_base_t1" and by["f22s-nemotron3nano4b-ctl0"].phase == "controls_t0"
    assert by["f22s-glm46vflash-se1"].phase == "static_esc_t1" and by["f22s-glm46vflash-se1"].block == "esc"
    assert {j.time for j in by.values() if "-sb" in j.name} == {"16:00:00"}
    assert {j.time for j in by.values() if "-ctl" in j.name} == {"24:00:00"}
    assert {j.time for j in by.values() if "-se" in j.name} == {"16:00:00"}
    assert all(j.block == "base" for j in p["jobs"]) and len(p["jobs"]) == 20 and len(p["esc"]) == 8
    # the THIRD block (lane G2): per target and trial the capability battery FIRST, then the 16,384 extend
    assert [j.name for j in p["ext"]] == [f"f22s-{k}-{kind}{t}" for k in KEYS for t in (0, 1) for kind in ("cap", "le")]
    assert all(j.block == "ext" for j in p["ext"]) and len(p["ext"]) == 16
    by_ext = {j.name: j for j in p["ext"]}
    assert by_ext["f22s-qwen35_4b-cap1"].phase == "capability_t1" and by_ext["f22s-qwen35_4b-cap1"].time == "12:00:00"
    assert by_ext["f22s-glm46vflash-le0"].phase == "ladder_extend_t0" and by_ext["f22s-glm46vflash-le0"].time == "16:00:00"
    assert by_ext["f22s-glm46vflash-le0"].args == "glm46vflash ladder_extend_t0 W 9" and by_ext["f22s-glm46vflash-le0"].cluster == "cluster_b"
    # geometry per target from the measured budget table (the windows DESIGN.md names)
    g = {s["key"]: s["geom"] for s in p["served"]}
    assert g["gemma4e4b"] == {"static_base": (4864, 121), "static_esc": (10240, 57), "controls_512": (2048, 288), "controls_2048": (4096, 144), "controls_8192": (10240, 57)}
    # the hybrid targets: S = min(256, budget // L) from the LIVE budget table (geometry.BUDGET_BY_MODEL —
    # main re-measured qwen35_4b's budget after the run, 2026-09-10; the run's own S is in README's table)
    from sisteer.capeval.serving.geometry import BUDGET_BY_MODEL
    for key in ("qwen35_4b", "nemotron3nano4b"):
        b = BUDGET_BY_MODEL[key]
        assert b.hybrid and g[key]["controls_512"] == (2048, min(256, b.batched_token_budget // 2048)), key
    assert g["glm46vflash"] == {"static_base": (4864, 161), "static_esc": (10240, 76), "controls_512": (2048, 384), "controls_2048": (4096, 192), "controls_8192": (10240, 76)}
    # --cluster re-routes a target wholesale
    p2 = _plan({"glm46vflash": "cluster_a"})
    assert all(j.cluster == "cluster_a" for j in p2["jobs"] + p2["esc"] if j.key == "glm46vflash")
    # the sbatch line: the site's qos, partition and log file on every cluster, --time, the positional args, --exclude passthrough
    # scoped by cluster
    line = ss.sbatch_line(by["f22s-glm46vflash-ctl1"], SHA, {"cluster_b": "gpu-x3"})
    assert line == (f"ssh {B['host']} 'cd {B['eval']}/{SHA} && sbatch {B['sbatch']} "
                    f"--time=24:00:00 --exclude=gpu-x3 -J f22s-glm46vflash-ctl1 "
                    f"scripts/sdfbase/full22/sampled/target_sampled.sbatch glm46vflash controls_t1 {SHA} 9'")
    line = ss.sbatch_line(by["f22s-qwen35_4b-sb0b"], SHA, {"cluster_b": "gpu-x3"})
    assert line == (f"ssh {A['host']} 'cd {A['eval']}/{SHA} && sbatch {A['sbatch']} --time=16:00:00 "
                    f"-J f22s-qwen35_4b-sb0b scripts/sdfbase/full22/sampled/target_sampled.sbatch qwen35_4b static_base_t0:renellm,grok420 {SHA} 9'")
    assert ss.parse_excludes(["compute-7"]) == {"cluster_a": "compute-7", "cluster_b": "compute-7"}
    assert ss.parse_excludes(["cluster_a:compute-7,compute-9", "compute-7"]) == {"cluster_a": "compute-7,compute-9", "cluster_b": "compute-7"}


def test_submit_plan_for_the_it_views_is_one_static_job_per_trial_on_cluster_b():
    """The instruct-direct TAGs are their OWN units: the KEY's roster `it` cells (seven on both since the batch-64 pair),
    so ONE static base job per trial (never the
    a/b split), controls per trial, escalation per trial; names f22s-<TAG>-sb<T> / -ctl<T> / -se<T>; the printed sbatch
    line passes the bare model key and `9 "" it` as the trailing positionals; cluster B like the rest of the run."""
    assert ss.IT_TAGS == IT_TAGS and ss.DOSE_TAGS == tuple(t for t in TAGS if t in DOSE_TAGS) and len(ss.DOSE_TAGS) == 44   # 38 LoRA + qwen's 6 relu
    # the seed-replicate views are their own group (never a dose view): the frontier's 44 stay exactly the frontier's
    assert ss.SEEDREP_TAGS == SEEDREP_TAGS == tuple(t for t in TAGS if t in SEEDREP_TAGS) and not set(ss.SEEDREP_TAGS) & set(ss.DOSE_TAGS)
    # submit_sampled groups EVERY llama8b view TAG under LLAMA_TAGS (the lane's four + the frontier's twelve dose views)
    assert ss.LLAMA_TAGS == tuple(t for t in TAGS if t in LLAMA_TAGS + LLAMA_DOSE_TAGS) and len(ss.LLAMA_TAGS) == 16
    assert ss.VIEW_TAGS == tuple(t for t in TAGS if t not in KEYS) and ss.TARGETS_ACCEPTED == KEYS + ss.VIEW_TAGS and len(ss.TARGETS_ACCEPTED) == 69 + len(FRONT_DOOR_TAGS)
    assert ss.FRONT_DOOR_TAGS == FRONT_DOOR_TAGS and not set(ss.FRONT_DOOR_TAGS) & set(ss.DOSE_TAGS)     # the front door is its own group, never a dose view
    assert ss.split_target("gemma4e4b_it") == ("gemma4e4b", "it") and ss.split_target("gemma4e4b") == ("gemma4e4b", "main")
    assert ss.split_target("llama8b_fullconst") == ("llama8b", "fullconst") and ss.split_target("llama8b") == ("llama8b", "main")
    p = ss.plan(ROSTER, list(IT_TAGS[:2]), {**ss.DEFAULT_CLUSTER}, 9)
    assert [j.name for j in p["jobs"]] == ["f22s-gemma4e4b_it-sb0", "f22s-gemma4e4b_it-sb1", "f22s-gemma4e4b_it-ctl0", "f22s-gemma4e4b_it-ctl1",
                                           "f22s-qwen35_4b_it-sb0", "f22s-qwen35_4b_it-sb1", "f22s-qwen35_4b_it-ctl0", "f22s-qwen35_4b_it-ctl1"]
    assert [j.name for j in p["esc"]] == ["f22s-gemma4e4b_it-se0", "f22s-gemma4e4b_it-se1", "f22s-qwen35_4b_it-se0", "f22s-qwen35_4b_it-se1"]
    by = {j.name: j for j in p["jobs"] + p["esc"]}
    assert all(j.cluster == "cluster_b" and j.view == "it" for j in by.values())
    assert by["f22s-qwen35_4b_it-sb0"].phase == "static_base_t0" and by["f22s-qwen35_4b_it-sb0"].model_key == "qwen35_4b"   # never split
    assert by["f22s-gemma4e4b_it-sb1"].args == 'gemma4e4b static_base_t1 W 9 "" it'
    assert by["f22s-gemma4e4b_it-ctl0"].args == 'gemma4e4b controls_t0 W 9 "" it' and by["f22s-qwen35_4b_it-se1"].args == 'qwen35_4b static_esc_t1 W 9 "" it'
    assert {j.time for j in by.values() if "-sb" in j.name} == {"16:00:00"} and {j.time for j in by.values() if "-ctl" in j.name} == {"24:00:00"}
    served = {s["key"]: s for s in p["served"]}
    assert served["gemma4e4b_it"]["n"] == 7 == len(it_cells("gemma4e4b"))   # none + two itdoc objects + the batch-64 control LoRA fit and LoRA subtraction (itdoc64__ / derived_itdoc64__) + the prompted safety AND constitution baselines
    assert served["qwen35_4b_it"]["n"] == 7 == len(it_cells("qwen35_4b"))   # the same seven, the batch-64 pair included
    assert served["gemma4e4b_it"]["model_key"] == "gemma4e4b" and served["gemma4e4b_it"]["view"] == "it"
    from sisteer.capeval.serving.geometry import BUDGET_BY_MODEL as _B
    # the MODEL's budget; a target WITHOUT a caps block carries no gate-based 16,384 rows (its third block is the
    # capability + extend one) and no gsm8k readout row (that phase is llama8b's, and its roster omits gsm8k)
    assert served["gemma4e4b_it"]["geom"] == {"static_base": (4864, 121), "static_esc": (10240, 57), "controls_512": (2048, 288),
                                              "controls_2048": (4096, 144), "controls_8192": (10240, 57)}
    assert _B["gemma4e4b"].seqs_at(18432) == 32   # the extend's 16,384 window buys 32 (common_sampled.sh f22s_rung_window)
    assert "not measured" in by["f22s-gemma4e4b_it-sb0"].est_h
    line = ss.sbatch_line(by["f22s-gemma4e4b_it-sb0"], SHA, {})
    assert line == (f"ssh {B['host']} 'cd {B['eval']}/{SHA} && sbatch {B['sbatch']} "
                    f"--time=16:00:00 -J f22s-gemma4e4b_it-sb0 "
                    f"scripts/sdfbase/full22/sampled/target_sampled.sbatch gemma4e4b static_base_t0 {SHA} 9 \"\" it'")
    # the main-view args are byte-identical to before (no trailing positionals)
    assert _plan()["jobs"][0].args == "gemma4e4b static_base_t0 W 9" and all(j.view == "main" for j in _plan()["jobs"])
    text = "\n".join(ss.render(p, ss.layout(p, {"cluster_b": 96}), SHA, {}, offline=True))
    assert "gemma4e4b/it=7 cells, qwen35_4b/it=7 cells" in text and text.count(f"\nssh {B['host']} ") == 8 + 4 + 8   # the roster's counts (seven each since the itdoc64 pair, 2026-09-13); still ONE static job per trial
    assert by["f22s-gemma4e4b_it-sb1"].args == 'gemma4e4b static_base_t1 W 9 "" it'
    ext = {j.name: j for j in p["ext"]}
    assert ext["f22s-qwen35_4b_it-le1"].args == 'qwen35_4b ladder_extend_t1 W 9 "" it' and ext["f22s-gemma4e4b_it-cap0"].args == 'gemma4e4b capability_t0 W 9 "" it'
    # the default --targets stays the four main views (their jobs were live when the views were added)
    assert ss.KEYS == KEYS


def test_submit_plan_for_a_dose_view_is_its_own_unit_on_cluster_b_with_the_view_positional(tmp_path):
    """A per-dose TAG is planned like an `it` TAG: its own unit with the roster's cell count for
    that view, ONE static base job per trial (no a/b split even on glm), controls + escalation per trial, routed to cluster B
    by default (its 96-GPU budget; cluster A's 16 is the main views'), the bare model key + `9 "" <view>` as the
    trailing positionals, the (key, view) bank view probed. Pins the exact sbatch line for nemotron3nano4b/d0p25 static_base_t0."""
    assert all(ss.DEFAULT_CLUSTER[t] == "cluster_b" for t in ALL_DOSE_TAGS + IT_TAGS) and ss.DEFAULT_CLUSTER["gemma4e4b"] == "cluster_a"
    assert ss.split_target("nemotron3nano4b_d0p25") == ("nemotron3nano4b", "d0p25") and ss.split_target("gemma4e4b_it_d1") == ("gemma4e4b", "it_d1")
    with pytest.raises(SystemExit, match="not a \\(target, view\\) in the sampled scope"):
        ss.split_target("nemotron3nano4b_halfdose")
    p = ss.plan(ROSTER, ["nemotron3nano4b_d0p25", "glm46vflash_d1p5", "gemma4e4b_it_d1"], {**ss.DEFAULT_CLUSTER}, 9)
    assert [j.name for j in p["jobs"]] == [
        "f22s-nemotron3nano4b_d0p25-sb0", "f22s-nemotron3nano4b_d0p25-sb1", "f22s-nemotron3nano4b_d0p25-ctl0", "f22s-nemotron3nano4b_d0p25-ctl1",
        "f22s-glm46vflash_d1p5-sb0", "f22s-glm46vflash_d1p5-sb1", "f22s-glm46vflash_d1p5-ctl0", "f22s-glm46vflash_d1p5-ctl1",
        "f22s-gemma4e4b_it_d1-sb0", "f22s-gemma4e4b_it_d1-sb1", "f22s-gemma4e4b_it_d1-ctl0", "f22s-gemma4e4b_it_d1-ctl1"]
    assert [j.name for j in p["esc"]] == [f"f22s-{t}-se{T}" for t in ("nemotron3nano4b_d0p25", "glm46vflash_d1p5", "gemma4e4b_it_d1") for T in (0, 1)]
    by = {j.name: j for j in p["jobs"] + p["esc"] + p["ext"]}
    assert all(j.cluster == "cluster_b" for j in by.values())
    j = by["f22s-nemotron3nano4b_d0p25-sb0"]
    assert (j.view, j.model_key, j.phase, j.time, j.args) == ("d0p25", "nemotron3nano4b", "static_base_t0", "16:00:00", 'nemotron3nano4b static_base_t0 W 9 "" d0p25')
    assert by["f22s-glm46vflash_d1p5-sb1"].args == 'glm46vflash static_base_t1 W 9 "" d1p5' and "-sb1a" not in by      # never the a/b split
    assert by["f22s-gemma4e4b_it_d1-se0"].args == 'gemma4e4b static_esc_t0 W 9 "" it_d1' and by["f22s-nemotron3nano4b_d0p25-ctl1"].args == 'nemotron3nano4b controls_t1 W 9 "" d0p25'
    served = {s["key"]: s for s in p["served"]}
    assert served["nemotron3nano4b_d0p25"]["n"] == 1 + len(fr.served_cells(ROSTER, "nemotron3nano4b", "d0p25")) >= 2
    assert (served["nemotron3nano4b_d0p25"]["model_key"], served["nemotron3nano4b_d0p25"]["view"]) == ("nemotron3nano4b", "d0p25")
    assert served["nemotron3nano4b_d0p25"]["geom"]["controls_512"] == (2048, 256) and served["glm46vflash_d1p5"]["geom"]["static_esc"] == (10240, 76)   # the MODEL's budget
    assert "not measured" in j.est_h
    assert ss.sbatch_line(j, SHA, {}) == (
        f"ssh {B['host']} 'cd {B['eval']}/{SHA} && sbatch {B['sbatch']} "
        f"--time=16:00:00 -J f22s-nemotron3nano4b_d0p25-sb0 "
        f"scripts/sdfbase/full22/sampled/target_sampled.sbatch nemotron3nano4b static_base_t0 {SHA} 9 \"\" d0p25'")
    # the printer end to end (offline): every line on cluster_b ending in the view positional; the probed path asks for the
    # (key, view) bank view; a dose view on a key that does not carry it refuses like the halfdose view
    rc = _run_printer(tmp_path, "down", "--offline", "--targets", "nemotron3nano4b_d0p25")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    subs = [l for l in rc.stdout.splitlines() if l.startswith("ssh ")]
    assert len(subs) == 4 + 2 + 4 and all(l.startswith(f"ssh {B['host']} ") and l.endswith(f'{SHA} 9 "" d0p25\'') for l in subs)
    assert "nemotron3nano4b/d0p25=" in rc.stdout and sum(1 for l in subs if " -J f22s-nemotron3nano4b_d0p25-sb0 " in l) == 1
    stub2, captured = tmp_path / "ssh-capture", tmp_path / "probe.sh"
    stub2.write_text(STUB.replace("cat >/dev/null", f"cat > {captured}", 1)); stub2.chmod(stub2.stat().st_mode | stat.S_IXUSR)
    rc = subprocess.run(["bash", str(SAMPLED / "submit_sampled.sh"), SHA, "--targets", "glm46vflash_d1p5"], capture_output=True, text=True,
                        env={**os.environ, "F22_SSH": str(stub2), "F22_STUB_MODE": "ok", "F22_PYTHON": PY}, cwd=REPO)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    remote = captured.read_text()
    assert "make_bank_view.py --check --target glm46vflash --view d1p5" in remote and 'echo "CHECK view glm46vflash/d1p5 OK' in remote and "--view main" not in remote
    for bad in ("gemma4e4b_d0p75", "nemotron3nano4b_it_d0p25", "nemotron3nano4b_halfdose"):
        rc = _run_printer(tmp_path, "ok", "--offline", "--targets", bad)
        assert rc.returncode == 2 and "REFUSING" in rc.stdout and "not in the sampled scope" in rc.stdout, bad


def test_llama8b_dose_views_are_in_scope_and_launchable_beside_the_lanes_own_four_views(tmp_path):
    """2026-09-13 (branch strength-phaseA): the strength frontier's twelve llama8b dose views join the sampled scope — and ONLY
    those. Pinned end to end: SAMPLED_VIEWS names both ladder families on the one served instruct model (`d*` the base-family
    LoRA transferred onto the instruct target, `it_d*` the instruct-direct LoRA) and neither `main` nor `it` (the
    sdfbase-llama8b lane's rows, run at its own sha; `d1` / `it_d1` are the frontier's own 1.0 anchors); the preset is
    origin/main's llama8b row byte for byte and agrees with the roster's pins; the 72 generated files are the twelve wave
    configs' text with the row's block (two cells + `none` each, the (llama8b, view) bank view, the campaign's ladder); NO
    llama8b main / it sampled config exists; the FROZEN shell (common_sampled.sh: the gate, the resolver, the job-side belt,
    the default rung windows) admits a dose view and refuses main / it under the same bytes it ran the other 30 TAGs with; and
    the printer plans a llama dose TAG as its own unit on cluster B with the view positional and an UNMEASURED estimate (no
    greedy pass of this target exists to anchor on), while the bare key and `llama8b_it` refuse."""
    # the llama8b scope holds its four views AND the twelve dose views
    assert msc.SAMPLED_VIEWS["llama8b"] == ("main", "it", "fullconst", "shortconst") + msc.DOSE_VIEWS + msc.DOSE_VIEWS_IT
    assert len(msc.SAMPLED_VIEWS["llama8b"]) == 16
    # ... plus the all-models one-seed views s1 / s1p, Modal-only and OUTSIDE the sampled scope
    assert set(ROSTER["targets"]["llama8b"]["views"]) == {"main", "it", "fullconst", "shortconst", "s1", "s1p", *LLAMA_DOSE_TAGS_VIEWS()}
    assert not {"s1", "s1p"} & set(msc.SAMPLED_VIEWS["llama8b"])
    p = PRESETS["presets"]["llama8b"]
    assert (p["model"], p["revision"]) == (ROSTER["targets"]["llama8b"]["model"], ROSTER["targets"]["llama8b"]["model_revision"]) \
        == ("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77")
    assert p["preset"] == {"temperature": 0.6, "top_p": 0.9} and p["source_kind"] == "generation_config" and "read 2026-09-09" in p["source"]
    # the 72 files: 12 TAGs x 2 trials x (base, cap8192, cap16384); every one the view's wave config + the row's block
    for tag in LLAMA_DOSE_TAGS:
        key, view = msc.split_tag(tag)
        assert key == "llama8b" and tag == f"llama8b_{view}" and msc.in_scope("llama8b", view)
        wave = yaml.safe_load(msc.wave_config_path(tag).read_text())
        assert wave["explicit_cells"] == ["none"] + list(fr.served_cells(ROSTER, "llama8b", view))
        # the it family keeps its two arms; the base family carries the control LoRA as its third
        assert wave["explicit_cells"] == (LLAMA_CELLS if view.startswith("it_") else
                                          ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"])
        for trial in (0, 1):
            for esc in (False, 8192, 16384):
                cfg = yaml.safe_load(msc.output_path(tag, trial, esc).read_text())
                assert cfg["model"] == "NousResearch/Meta-Llama-3.1-8B-Instruct"
                assert cfg["explicit_cells"] == (LLAMA_CELLS if view.startswith("it_") else
                                                 ["none", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"])
                assert cfg["defenses_dir"] == f"out/full22/llama8b/cells_view/{view}" and cfg["private_run"] == msc.sampled_name(tag, trial, esc)
                assert {k: v for k, v in cfg["sampled_decode"].items() if k in msc.ALLOWED_SAMPLER_KEYS} == PINNED["llama8b"]
                assert cfg["sampled_decode"]["static_cap"] == (esc or 4096) and cfg["sampled_decode"]["controls_cap"] == 512
    assert len([q for q in (REPO / "configs").glob("sdfbase_full22_llama8b_*d*_sampled_*.yaml")
                if msc.split_tag(q.name[len("sdfbase_full22_"):].split("_sampled_")[0])[1] in LLAMA_DOSE_TAGS_VIEWS()]) == 72
    # the FROZEN shell: the gate, the resolver + the job-side belt, the default rung windows; a non-view refuses
    for view, want in (("it_d0p25", "rc=0"), ("d1", "rc=0"), ("main", "rc=0"), ("it", "rc=0"), ("halfdose", "rc=3")):
        rc = _bash(f'f22s_view_in_scope llama8b {view}; echo "rc=$?"')
        assert rc.stdout.strip() == want, (view, rc.stdout, rc.stderr)
    rc = _bash(f'f22s_resolve_config llama8b 0 it_d0p25 >/dev/null; echo "$KEY|$VIEW|$TAG|$WAVE_CFG|$CFG|$MANIFEST"; '
               f'echo "$(f22s_rung_window 2048) $(f22s_rung_window 8192) $(f22s_rung_window 16384)"; TRIAL=0; f22s_check_config "{PY}"')
    assert rc.returncode == 0, rc.stdout + rc.stderr
    lines = rc.stdout.strip().splitlines()
    assert lines[0] == ("llama8b|it_d0p25|llama8b_it_d0p25|configs/sdfbase_full22_llama8b_it_d0p25.yaml|"
                        "configs/sdfbase_full22_llama8b_it_d0p25_sampled_t0.yaml|configs/capeval/manifests/petri_full22_llama8b_it_d0p25.yaml")
    assert lines[1] == "4096 10240 18432" and lines[-1].startswith("sampled configs OK: configs/sdfbase_full22_llama8b_it_d0p25_sampled_t0.yaml")
    rc = _bash('f22s_resolve_config llama8b 0 halfdose; echo reached')
    assert "FATAL" in rc.stdout and "not in the sampled scope" in rc.stdout and "reached" not in rc.stdout, rc.stdout
    # the printer: the plan
    assert ss.split_target("llama8b_it_d0p25") == ("llama8b", "it_d0p25") and ss.split_target("llama8b_d1") == ("llama8b", "d1")
    assert ss.split_target("llama8b") == ("llama8b", "main") and ss.split_target("llama8b_it") == ("llama8b", "it")
    with pytest.raises(SystemExit, match="not a \\(target, view\\) in the sampled scope"):
        ss.split_target("llama8b_halfdose")
    # submit_sampled files every llama8b view TAG under LLAMA_TAGS (the lane's four + the frontier's twelve dose views)
    assert "llama8b" in ss.TARGETS_ACCEPTED and set(LLAMA_DOSE_TAGS) <= set(ss.LLAMA_TAGS) and all(ss.DEFAULT_CLUSTER[t] == "cluster_b" for t in LLAMA_DOSE_TAGS)
    assert ss.GREEDY_STATIC_HOURS["llama8b"] is None and ss.KEYS == KEYS       # no greedy anchor; the default --targets is unchanged
    p = ss.plan(ROSTER, ["llama8b_it_d0p25", "llama8b_d1"], {**ss.DEFAULT_CLUSTER}, 9)
    assert [j.name for j in p["jobs"]] == ["f22s-llama8b_it_d0p25-sb0", "f22s-llama8b_it_d0p25-sb1", "f22s-llama8b_it_d0p25-ctl0", "f22s-llama8b_it_d0p25-ctl1",
                                           "f22s-llama8b_d1-sb0", "f22s-llama8b_d1-sb1", "f22s-llama8b_d1-ctl0", "f22s-llama8b_d1-ctl1"]
    assert [j.name for j in p["esc"]] == [f"f22s-{t}-se{T}" for t in ("llama8b_it_d0p25", "llama8b_d1") for T in (0, 1)]
    by = {j.name: j for j in p["jobs"] + p["esc"] + p["ext"]}
    assert all(j.cluster == "cluster_b" and j.model_key == "llama8b" for j in by.values()) and "-sb0a" not in by
    j = by["f22s-llama8b_it_d0p25-sb0"]
    assert (j.view, j.phase, j.time, j.args) == ("it_d0p25", "static_base_t0", "16:00:00", 'llama8b static_base_t0 W 9 "" it_d0p25')
    assert j.est_h.startswith("UNMEASURED") and by["f22s-llama8b_d1-se1"].args == 'llama8b static_esc_t1 W 9 "" d1'
    served = {s["key"]: s for s in p["served"]}
    assert served["llama8b_it_d0p25"]["n"] == 3 and served["llama8b_d1"]["n"] == 4 and served["llama8b_d1"]["view"] == "d1"   # base family: none + 3 arms
    # the MODEL's certified budget: main's measured row for llama8b (475,136 = 29 x 16384, the 2026-09-10 serving gate) —
    # S = budget // L: 97 static, 232 controls-512, 116 controls-2048, 46 at the 10,240 rung windows
    assert served["llama8b_d1"]["geom"] == {"static_base": (4864, 97), "static_esc": (10240, 46), "controls_512": (2048, 232),
                                            "controls_2048": (4096, 116), "controls_8192": (10240, 46),
                                            "controls_16384": (18432, 25), "static_esc16": (18432, 25)}   # main's 16,384 rungs
    assert ss.sbatch_line(j, SHA, {}) == (
        f"ssh {B['host']} 'cd {B['eval']}/{SHA} && sbatch {B['sbatch']} "
        f"--time=16:00:00 -J f22s-llama8b_it_d0p25-sb0 "
        f"scripts/sdfbase/full22/sampled/target_sampled.sbatch llama8b static_base_t0 {SHA} 9 \"\" it_d0p25'")
    # the printer end to end (offline): the acceptance — a llama dose TAG PRINTS beside the lane's own views
    rc = _run_printer(tmp_path, "down", "--offline", "--targets", "llama8b_it_d0p25")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    subs = [l for l in rc.stdout.splitlines() if l.startswith("ssh ")]
    # llama8b carries main's caps block, so its third block is the GATE-based static 16,384 rung
    # x2 (static_esc16_t<T>), not the ext battery x2 + extend x2 the caps-less targets get (4 + 2 + 4 for nemotron above)
    assert len(subs) == 4 + 2 + 2 and all(l.startswith(f"ssh {B['host']} ") and l.endswith(f'{SHA} 9 "" it_d0p25\'') for l in subs)
    assert sum(1 for l in subs if "static_esc16_t" in l) == 2 and not any("gsm8k" in l or "extend" in l for l in subs)
    assert "llama8b/it_d0p25=3 cells" in rc.stdout and "UNMEASURED" in rc.stdout and sum(1 for l in subs if " -J f22s-llama8b_it_d0p25-sb0 " in l) == 1
    # the bare key and `llama8b_it` are the lane's OWN views on main (in scope; test_submit_plan_for_llama8b_it... prints them);
    # only the halfdose forms stay out, on either family
    for bad in ("llama8b_halfdose", "llama8b_it_halfdose"):
        rc = _run_printer(tmp_path, "ok", "--offline", "--targets", bad)
        assert rc.returncode == 2 and "REFUSING" in rc.stdout and "not in the sampled scope" in rc.stdout, (bad, rc.stdout)
        assert not any(l.startswith("ssh ") for l in rc.stdout.splitlines()), bad


def LLAMA_DOSE_TAGS_VIEWS():
    return tuple(t[len("llama8b_"):] for t in LLAMA_DOSE_TAGS)


def test_submit_layout_holds_the_jobs_over_a_clusters_free_budget():
    p = _plan()
    lay = ss.layout(p, {"cluster_a": 3, "cluster_b": 96})
    assert [j.name for j in lay["cluster_a"]["submit"]] == ["f22s-gemma4e4b-sb0", "f22s-gemma4e4b-sb1", "f22s-gemma4e4b-ctl0"]
    assert len(lay["cluster_a"]["hold"]) == 7 and lay["cluster_a"]["hold"][0].name == "f22s-gemma4e4b-ctl1"
    assert len(lay["cluster_b"]["submit"]) == 10 and lay["cluster_b"]["hold"] == []
    assert [j.name for j in lay["cluster_a"]["esc"]] == ["f22s-gemma4e4b-se0", "f22s-gemma4e4b-se1", "f22s-qwen35_4b-se0", "f22s-qwen35_4b-se1"]
    lay0 = ss.layout(p, {"cluster_a": 0, "cluster_b": 0})
    assert lay0["cluster_a"]["submit"] == [] and len(lay0["cluster_a"]["hold"]) == 10
    text = "\n".join(ss.render(p, lay, SHA, {}, offline=True))
    assert text.count(f"\nssh {A['host']} ") == 3 + 4 + 8 and text.count("# HOLD (over the free budget on cluster_a") == 7   # + the third block
    assert "7 HELD" in text and "second-block jobs" in text and "No --dependency on purpose" in text


STUB = r'''#!/bin/bash
# stub ssh for the printer test: consumes the remote script on stdin, answers per F22_STUB_MODE
cat >/dev/null
host="${@: -3:1}"
[ "$F22_STUB_MODE" = down ] && exit 255
echo "CHECK checkout - OK /shared/sisteer-eval-repo/x"
for k in gemma4e4b qwen35_4b nemotron3nano4b glm46vflash; do
  if [ "$F22_STUB_MODE" = nocache ] && [ "$host" = login-b ] && [ "$k" = glm46vflash ]; then echo "CHECK cache $k PROBLEM no snapshot at /hub/x"; else echo "CHECK cache $k OK /hub/$k"; fi
  if [ "$F22_STUB_MODE" = short ] && [ "$host" = login-a ] && [ "$k" = qwen35_4b ]; then echo "CHECK view $k PROBLEM view /v/main is SHORT: the roster names 8 cells, 6 are present"; else echo "CHECK view $k OK ok: /v/main (8 entries verified)"; fi
done
if [ "$host" = login-b ]; then echo "CHECK budget - OK used=4 cap=16"; else echo "CHECK budget - OK used=7 cap=16"; fi
[ "$F22_STUB_MODE" = stop ] && [ "$host" = login-a ] && echo "CHECK stop - PROBLEM wave STOP sentinel present at /x/STOP" || echo "CHECK stop - OK no wave STOP"
'''


def _run_printer(tmp_path, mode, *extra):
    stub = tmp_path / "ssh"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    env = {**os.environ, "F22_SSH": str(stub), "F22_STUB_MODE": mode, "F22_PYTHON": PY}
    return subprocess.run(["bash", str(SAMPLED / "submit_sampled.sh"), SHA, *extra], capture_output=True, text=True, env=env, cwd=REPO)


def test_submit_printer_prints_the_plan_only_when_every_precondition_holds(tmp_path):
    rc = _run_printer(tmp_path, "ok", "--exclude", "cluster_a:gpu-x7")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    lines = rc.stdout.splitlines()
    assert "# CHECK configs - OK" in rc.stdout and "# CHECK sampled-configs - OK" in rc.stdout
    # the probed budgets drive the layout: cluster_a 16 - 7 = 9 free -> 10 first-block jobs, 1 HELD; cluster_b 96 - 4 = 92
    assert "# CHECK budget cluster_a OK free=9 (cap 16 - used 7)" in rc.stdout and "# CHECK budget cluster_b OK free=92 (cap 96 - used 4)" in rc.stdout
    subs = [l for l in lines if l.startswith("ssh ")]
    assert len(subs) == 9 + 10 + 8 + 16    # first block minus the held one, plus the second block, plus the third (cap + le)
    held = [l for l in lines if l.startswith("# HOLD")]
    assert len(held) == 1 and "-J f22s-qwen35_4b-ctl1 " in held[0]
    for name in [j.name for j in _plan()["jobs"] + _plan()["esc"] + _plan()["ext"]]:
        assert sum(1 for l in subs + held if f" -J {name} " in l) == 1, name   # one submission each
    for l in subs:
        assert "--export=" not in l and "--mem=0" not in l and "target_sampled.sbatch" in l and f" {SHA} 9'" in l
        if l.startswith(f"ssh {A['host']} "):
            assert f"sbatch {A['sbatch']} --time=" in l and "--exclude=gpu-x7" in l       # cluster A names the site's log dir too
        else:
            assert l.startswith(f"ssh {B['host']} ") and f"sbatch {B['sbatch']} --time=" in l and "--exclude" not in l
    # the second block follows the first and carries no Slurm dependency
    i_first = rc.stdout.index("# --- 1. FIRST BLOCK"); i_second = rc.stdout.index("# --- 2. SECOND BLOCK")
    assert i_first < i_second and not any("--dependency" in l for l in subs + held)   # no Slurm dependency on any submission
    assert all(rc.stdout.index(f" -J f22s-{k}-se{t} ") > i_second for k in KEYS for t in (0, 1))
    # the THIRD block follows the second: cap<T> printed before le<T> for every target and trial, no dependency
    i_third = rc.stdout.index("# --- 3. THIRD BLOCK")
    assert i_second < i_third < rc.stdout.index("# --- watchers")
    assert all(i_second < rc.stdout.index(f" -J f22s-{k}-se{t} ") < i_third for k in KEYS for t in (0, 1))
    for k in KEYS:
        for t in (0, 1):
            assert i_third < rc.stdout.index(f" -J f22s-{k}-cap{t} ") < rc.stdout.index(f" -J f22s-{k}-le{t} ")
            assert f"target_sampled.sbatch {k} capability_t{t} {SHA} 9'" in rc.stdout and f"target_sampled.sbatch {k} ladder_extend_t{t} {SHA} 9'" in rc.stdout
    assert "no greedy arm, S-D7" in rc.stdout and "nothing past" in rc.stdout and "16,384 — S-D4" in rc.stdout
    assert f"watch-job.sh --slurm <jobid> --host {A['host']}" in rc.stdout and f"watch-job.sh --slurm <jobid> --host {B['host']}" in rc.stdout
    # it runs nothing: no sbatch invoked, the stub only ever read stdin
    src = (SAMPLED / "submit_sampled.py").read_text()
    assert not any(l.strip().startswith("sbatch ") for l in src.splitlines()) and "subprocess.run" in src
    assert "from submit_newmodels import parse_checks, probe" in src   # the wave's read-only probe, reused not copied


@pytest.mark.parametrize("mode,needle", [("short", "is SHORT"), ("nocache", "no snapshot"), ("down", "CHECK ssh - PROBLEM"), ("stop", "STOP sentinel present")])
def test_submit_printer_refuses_a_short_view_a_missing_snapshot_a_stop_or_an_unreachable_cluster(tmp_path, mode, needle):
    rc = _run_printer(tmp_path, mode)
    assert rc.returncode == 2, rc.stdout + rc.stderr
    assert "REFUSING TO PRINT SUBMISSIONS" in rc.stdout and needle in rc.stdout
    assert not any(l.startswith("ssh ") for l in rc.stdout.splitlines()), mode   # nothing submittable
    if mode == "short":
        assert "make_bank_view.py --target qwen35_4b --view main" in rc.stdout   # the rebuild command, as a comment


def test_submit_printer_offline_and_its_flags(tmp_path):
    rc = _run_printer(tmp_path, "down", "--offline", "--cluster-a-budget", "4", "--targets", "gemma4e4b,glm46vflash", "--cluster", "glm46vflash=cluster_a")
    assert rc.returncode == 0, rc.stdout + rc.stderr    # offline never probes (the stub would be 'down')
    assert "UNVERIFIED (--offline)" in rc.stdout
    subs = [l for l in rc.stdout.splitlines() if l.startswith("ssh ")]
    assert all(l.startswith(f"ssh {A['host']} ") for l in subs) and f"ssh {B['host']}" not in rc.stdout
    assert len(subs) == 4 + 4 + 8 and rc.stdout.count("# HOLD") == 6    # 10 first-block jobs, 4 fit, 6 held; 4 esc jobs; 8 third-block jobs
    assert "qwen35_4b" not in "\n".join(subs) and "nemotron3nano4b" not in "\n".join(subs)
    for bad in (["--targets", "nemotron3nano4b_halfdose"], ["--cluster", "qwen35_4b=mars"], ["--exclude", "mars:compute-1"]):
        rc = _run_printer(tmp_path, "ok", "--offline", *bad)
        assert rc.returncode == 2 and "REFUSING" in rc.stdout, bad
    # the instruct-direct TAGs print (lane F): 8 first-block + 4 second-block + 4 third-block lines, every one on cluster_b with `9 "" it`
    rc = _run_printer(tmp_path, "down", "--offline", "--targets", "gemma4e4b_it,qwen35_4b_it")
    assert rc.returncode == 0, rc.stdout + rc.stderr
    subs = [l for l in rc.stdout.splitlines() if l.startswith("ssh ")]
    assert len(subs) == 8 + 4 + 8 and all(l.startswith(f"ssh {B['host']} ") and l.endswith(f'{SHA} 9 "" it\'') for l in subs)
    assert sum(1 for l in subs if " -J f22s-gemma4e4b_it-sb0 " in l) == 1 and not any("-sb0a" in l for l in subs)
    # the probed (stub) path checks the (key, it) bank view: the remote script asks for `--view it` and echoes `CHECK view key/it`
    stub2, captured = tmp_path / "ssh-capture", tmp_path / "probe.sh"
    stub2.write_text(STUB.replace("cat >/dev/null", f"cat > {captured}", 1)); stub2.chmod(stub2.stat().st_mode | stat.S_IXUSR)
    rc = subprocess.run(["bash", str(SAMPLED / "submit_sampled.sh"), SHA, "--targets", "gemma4e4b_it"], capture_output=True, text=True,
                        env={**os.environ, "F22_SSH": str(stub2), "F22_STUB_MODE": "ok", "F22_PYTHON": PY}, cwd=REPO)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    remote = captured.read_text()
    assert "make_bank_view.py --check --target gemma4e4b --view it" in remote and 'echo "CHECK view gemma4e4b/it OK' in remote
    assert "--view main" not in remote and "CHECK cache gemma4e4b_it" in remote
    rc = subprocess.run(["bash", str(SAMPLED / "submit_sampled.sh"), "notasha", "--offline"], capture_output=True, text=True, env={**os.environ, "F22_PYTHON": PY}, cwd=REPO)
    assert rc.returncode == 2 and "40-hex" in rc.stdout


# --- 5: the docs + the wave's globs -----------------------------------------------------------------

def test_readme_carries_the_contract_sections():
    src = (SAMPLED / "README.md").read_text()
    for m in ("## Presets", "Seed rule", "20260909", "## Stores", "never in place", "## Running it",
              "## The ladder", "--floor-sample 200", "no Slurm dependencies", "f22s-<key>-sb<T>", "f22s-<key>-ctl<T>", "f22s-<key>-se<T>",
              "second block", "## Grading", "two_si_<family>_aggregate.json", "F22_WORKTREE", "served_capability_judge.py",
              "CAPEVAL_JUDGE_CACHE", "run from a clone", "$WAVE_ROOT/STOP"):
        assert m in src, m
    for key in KEYS:
        assert key in src
    assert "top_p 1.0" in src and "repetition_penalty 1.1" in src and "shipped_generation_config" in src
    # lane F: the instruct-direct views, their exact usages, the two report notes, the 16384 ceiling
    for m in ("## The instruct-direct views", "gemma4e4b_it", "qwen35_4b_it", 'target_sampled.sbatch gemma4e4b static_base_t0 <sha> 9 "" it',
              "f22s-<TAG>-sb<T>", "--targets gemma4e4b_it,qwen35_4b_it", "greedy_baseline_it.json", "by_cell", "first sampled-vs-greedy comparison of prompting",
              "rank statistic", "16,384", "1% at-cap", "## Capability readouts", "readouts.gsm8k", "letter_logprob_argmax", "thinking-on",
              "temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5", "read the top-level preset"):
        assert m in src, m
    design = (SAMPLED / "DESIGN.md").read_text()
    assert "16,384" in design
    man = json.loads((SAMPLED / "wave_manifest_sampled.json").read_text())
    assert set(man["it_views"]["cells"]) == set(IT_CELLS_LANE_F_RECORD) and man["it_views"]["tags"] == list(IT_TAGS[:2])   # lane F's record, as run


def test_wave_manifest_records_the_final_state_of_the_sampled_lane():
    """The manifest closes the run's record: pass 2 complete on all 12 tag-trials with the it lanes
    restricted and the four constitution cells held, the supplementary per-(cell, control) pass with its re-attempt flags, the GLM
    reclassify re-derivation with its `.pre-S-D26` backups named, the report's location (v2 / v3) and the figures mirror, and the
    check statuses the re-run relies on. The file stays the same JSON shape (indent 1, ASCII-escaped) every earlier block was written in."""
    raw = (SAMPLED / "wave_manifest_sampled.json").read_bytes()
    man = json.loads(raw)
    assert (json.dumps(man, indent=1) + "\n").encode() == raw                  # one writer shape: nothing hand-edited out of format
    fs = man["final_state_2026_09_11"]
    for k in ("at", "pass2", "supplementary_pass2_S_D24_S_D25", "S_D26_glm_reclassify_rederived", "ladder", "report", "checker_fixes_92b3f35f"):
        assert k in fs, k
    assert "12 tag-trials" in fs["pass2"] and "prompted__safety_refusal" in fs["pass2"] and "are HELD - their pass 2" in fs["pass2"]
    sup = fs["supplementary_pass2_S_D24_S_D25"]
    assert "34 stores" in sup["mirror"] and sup["units"].startswith("12") and sup["why"].startswith("The flip pass was per CELL") and "Separately, two pass-2 blocks" in sup["why"]
    assert "--reattempt-abstentions=transport_refusal" in sup["judge_args"] and "--reattempt-after=2026-09-10T18:00:00Z" in sup["judge_args"] and "--no-resume-ok" in sup["judge_args"]
    assert "350" in sup["result"] and "refused_abstention_ceiling" in sup["why"] and "now file ok" in sup["result"]
    # the record says what RAN (the then-default 200 per unit, a configured 1,000 on Low Prio), not what was intended
    assert sup["configured_concurrency"].startswith("VIOLATION (the lane's own)") and "200" in sup["configured_concurrency"] and "1,000" in sup["configured_concurrency"] and "UNMEASURED" in sup["configured_concurrency"]
    assert sup["concurrency"].startswith("INTENDED")
    fx2 = [v for k, v in fs.items() if k.startswith("fixes_") and k != "checker_fixes_92b3f35f"]
    assert len(fx2) == 1 and "_yerr" in fx2[0]["flip_rates_figure"] and "flip_rates_<TAG>.png" in fx2[0]["flip_rates_figure"]
    assert "--allow-over-ceiling" in fx2[0]["grade_sampled_lane_ceiling"] and "100" in fx2[0]["grade_sampled_lane_ceiling"] and "tests/test_sampled_grading.py" in fx2[0]["tests"]
    assert "--allow-over-ceiling" in DESIGN and "ceiling: org=<org> parallel=<p> judge_concurrency=<j> configured=<p*j> limit=<500|100>" in DESIGN
    d26 = fs["S_D26_glm_reclassify_rederived"]
    assert "seven" in d26["found"].lower() or "7 " in d26["found"]
    assert "copy-forward" in d26["cause"] and "complete" in d26["cause"] and "exit_le_1pct" in d26["done"] and "no_rung" in d26["done"]
    assert sorted(Path(b).name for b in d26["backups"]) == sorted(["reclassify.pre-S-D26-20260911T023528Z.json", "reclassify.pre-S-D26-20260911T023542Z.json",
                                                                  "decision_16384.pre-S-D26.json", "decision_16384.pre-S-D26.json"])
    assert all(b.startswith("DEST/") for b in d26["backups"])                     # DEST, never an absolute home path
    assert "8,192" in fs["ladder"] and "zero promotions" in fs["ladder"] and "12 tag-trials" in fs["ladder"]
    rep = fs["report"]
    assert rep["location"].startswith("DEST/report/REPORT.md") and "analysis/" in rep["location"] and "figs/" in rep["location"]
    assert "48" in rep["v2"] and "16,384" in rep["v2"] and "v3" in rep and "not_reached 48" in rep["v3"] and "final_rung.json" in rep["v3"] and "byte-identical" in rep["v3"]
    assert rep["figures_mirror"].startswith("~/figs/f22-sampled/") and "served from the maintainer's figure host" in rep["figures_mirror"]
    fx = fs["checker_fixes_92b3f35f"]
    assert "decision.json" in fx["reclassify_rows"] and "complete" in fx["reclassify_rows"] and "never trusted" in fx["reclassify_rows"]
    assert "not_reached" in fx["not_reached"] and "decision_16384.json" in fx["not_reached"] and "--allow-pending" in fx["not_reached"]
    assert "tests/test_analysis_sampled.py" in fx["tests"]
    assert "92b3f35f" in man["analysis_tool"]["branch"] and isinstance(man["analysis_tool"]["tests"], int) and man["analysis_tool"]["tests"] >= 33
    assert any(c.startswith("92b3f35f") for c in man["analysis_tool"]["commits"]) and any(c.startswith("439ff3ad") for c in man["analysis_tool"]["commits"])
    # the earlier blocks stand untouched in shape
    assert man["ladder_closed_at_8192"]["consequence"] == "no 16,384 rung is generated anywhere; the cap-8192 roots are final"
    assert man["it_views"]["tags"] == list(IT_TAGS[:2])   # lane F's record: the two canonical it views (llama8b_it is the sdfbase-llama8b wave's)


def test_the_wave_tests_globs_are_untouched():
    # the wave test globs F22/*.sbatch (non-recursive) and parametrises configs from the roster's views, so nothing
    # added under sampled/ or as configs/sdfbase_full22_*_sampled_* enters its invariants
    assert not list(F22.glob("*sampled*.sbatch")) and (SAMPLED / "target_sampled.sbatch").is_file()
    roster_configs = {v["config"] for t in ROSTER["targets"].values() for v in t["views"].values()}
    assert not any("_sampled_" in c for c in roster_configs)
    assert (REPO / "scripts/sdfbase/full22/wave_manifest.json").read_text().count("sampled") == 0


def test_nemotron_fused_kernel_assertion_runs_before_any_standup():
    """The Nemotron target must refuse, inside the job, when the SERVING venv lacks vLLM's fused Mamba2 ops:
    generation never runs on a silent slow path. The helper exists, names the four vLLM modules
    + the custom op, and the sbatch calls it after the serving venv is built and BEFORE the first standup."""
    helpers = (REPO / "scripts/sdfbase/full22/sampled/common_sampled.sh").read_text()
    sbatch = (REPO / "scripts/sdfbase/full22/sampled/target_sampled.sbatch").read_text()
    assert "f22s_assert_vllm_mamba_kernels()" in helpers
    for m in ("vllm.model_executor.models.nemotron_h", "vllm.model_executor.layers.mamba.ops.ssd_combined",
              "vllm.model_executor.layers.mamba.ops.causal_conv1d", "selective_scan_fwd"):
        assert m in helpers, m
    call = sbatch.index("f22s_assert_vllm_mamba_kernels")
    assert sbatch.index("f22_serving_venv") < call < sbatch.index("f22_standup attacks"), "after the venv, before any standup"
    assert 'case "$KEY" in nemotron3nano4b)' in helpers, "a no-op for the non-Mamba targets"


def test_sbatch_reads_mmlu_top_logprobs_with_a_module_registered_exec():
    """The controls phase reads the driver's MMLU_TOP_LOGPROBS by exec'ing scripts/served_capability_run.py from a
    file spec. Python's dataclasses need sys.modules[cls.__module__] to exist, so a module-level @dataclass in the
    driver (lane B) killed the wave's unregistered one-liner in 76 s (canary 84788, 2026-09-09):
    AttributeError: 'NoneType' object has no attribute '__dict__'. Run the sbatch's EXACT recipe against the real
    driver and require the constant back."""
    import re, subprocess, sys
    sbatch = (REPO / "scripts/sdfbase/full22/sampled/target_sampled.sbatch").read_text()
    m = re.search(r"CTL_TOP_LOGPROBS=\$\(\"\$MPY\" -c '([^']+)' scripts/served_capability_run.py\)", sbatch)
    assert m, "the sbatch no longer reads MMLU_TOP_LOGPROBS with the -c recipe this test pins"
    recipe = m.group(1)
    assert 'sys.modules["scr"]=m' in recipe, "the exec'd module must be registered before exec_module (dataclasses)"
    out = subprocess.run([sys.executable, "-c", recipe, str(REPO / "scripts/served_capability_run.py")],
                         capture_output=True, text=True, cwd=str(REPO),
                         env={"PYTHONPATH": f"{REPO}/src:{REPO}/scripts", "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr[-800:]
    assert int(out.stdout.strip()) == 64
    wave = (REPO / "scripts/sdfbase/full22/target_static.sbatch").read_text()
    assert 'sys.modules["scr"]=m' in wave, "the wave's controls phase reads the same driver and needs the same fix"


def test_write_ladder_skips_the_other_kinds_events_in_the_shared_events_file(tmp_path):
    """The controls job and the static jobs of one (target, trial) share LADDER_DIR, so events.jsonl carries both
    kinds; a static fold must skip controls events (and vice versa), never refuse (canary 84812, 2026-09-09)."""
    import json, subprocess, sys
    ev = tmp_path / "events.jsonl"
    ev.write_text("\n".join(json.dumps(e) for e in [
        {"kind": "controls", "cell": "c", "control": "xstest", "rung": 512, "n_at_cap": 7, "root": "/r512", "at": "t"},
        {"kind": "static", "attack": "pasttense", "cell": "c", "rung": 4096, "n_at_cap": 3, "root": "/s", "at": "t"},
        {"kind": "controls", "cell": "c", "control": "xstest", "rung": 2048, "n_at_cap": 0, "root": "/r2048", "at": "t"},
    ]) + "\n")
    script = REPO / "scripts/sdfbase/full22/sampled/write_ladder.py"
    for kind, out, expect_cell_readout, skipped in (("static", "static_ladder_t0.json", ("c", "pasttense"), 2),
                                                    ("controls", "ladder.json", ("c", "xstest"), 1)):
        r = subprocess.run([sys.executable, str(script), "--events", str(ev), "--out", str(tmp_path / out), "--kind", kind,
                            "--tag", "qwen35_4b", "--trial", "0", "--floor-sample-k", "200"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-600:]
        lad = json.loads((tmp_path / out).read_text())
        assert lad["kind"] == kind and expect_cell_readout[1] in lad["cells"][expect_cell_readout[0]]
        assert lad["skipped_other_kind_events"] == skipped


def test_dedupe_raced_store_keeps_first_row_per_key_and_parks_the_rest(tmp_path):
    """Two writers on one shard (canary 84812 + sb0a 84829): the first row per (goal_id, sample_idx, trial) and the first
    progress record per goal survive; removed rows are parked beside the file; a text-differing pair is counted; a
    recently modified file (active writer), a greedy-shaped row, or two decode blocks refuse."""
    import json, os, subprocess, sys, time
    shard = tmp_path / "pasttense" / "cell" / "shard00"; shard.mkdir(parents=True)
    dec = {"do_sample": True, "temperature": 0.7, "cap": 4096, "arm": "vendor_recommended", "seed_base": 1, "trial": 0}
    def row(g, i, text):
        return {"goal_id": g, "sample_idx": i, "trial": 0, "decode": dec, "sampling_seed": 5, "finish_reason": "stop",
                "response": text, "family": "pasttense"}
    rows = [row("g1", 0, "a"), row("g1", 1, "b"), row("g2", 0, "c"), row("g1", 0, "a"), row("g1", 1, "B2"), row("g2", 1, "d")]
    (shard / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (shard / "progress.jsonl").write_text("".join(json.dumps({"goal_id": g}) + "\n" for g in ("g1", "g2", "g1")))
    old = time.time() - 3600
    for f in ("responses.jsonl", "progress.jsonl"):
        os.utime(shard / f, (old, old))
    script = REPO / "scripts/sdfbase/full22/sampled/dedupe_raced_store.py"
    r = subprocess.run([sys.executable, str(script), "--cell-root", str(shard.parent), "--quiet-s", "600"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    kept = [json.loads(l) for l in (shard / "responses.jsonl").read_text().splitlines()]
    assert [(k["goal_id"], k["sample_idx"], k["response"]) for k in kept] == [("g1", 0, "a"), ("g1", 1, "b"), ("g2", 0, "c"), ("g2", 1, "d")]
    assert len((shard / "progress.jsonl").read_text().splitlines()) == 2
    rec = json.loads((shard / "dedupe_record.json").read_text())
    assert rec["n_duplicate_rows"] == 2 and rec["n_duplicate_pairs_text_differs"] == 1 and rec["n_duplicate_pairs_text_identical"] == 1
    parked = list(shard.glob("responses.jsonl.duplicates.*.jsonl"))
    assert len(parked) == 1 and len(parked[0].read_text().splitlines()) == 2
    # idempotent
    r2 = subprocess.run([sys.executable, str(script), "--cell-root", str(shard.parent), "--quiet-s", "0"], capture_output=True, text=True)
    assert r2.returncode == 0 and '"changed": false' in r2.stdout
    # an active writer refuses
    (shard / "responses.jsonl").touch()
    r3 = subprocess.run([sys.executable, str(script), "--cell-root", str(shard.parent), "--quiet-s", "600"], capture_output=True, text=True)
    assert r3.returncode != 0 and "writer may still be active" in r3.stderr
    # a greedy-shaped row refuses
    (shard / "responses.jsonl").write_text(json.dumps({"goal_id": "g9", "sample_idx": 0, "trial": 0, "response": "x"}) + "\n")
    for f in ("responses.jsonl", "progress.jsonl"):   # both files must look quiet for the row check to be reached
        os.utime(shard / f, (old, old))
    r4 = subprocess.run([sys.executable, str(script), "--cell-root", str(shard.parent), "--quiet-s", "600"], capture_output=True, text=True)
    assert r4.returncode != 0 and "not a sampled store" in r4.stderr


# --------------------------------------------------------------------------- #
# LOW-DOSE LIVENESS in the launcher (common_sampled.sh f22s_low_dose_liveness / f22s_launch_cell / f22s_positive_control):
# the writer once per steered cell per standup, ONE log line per cell, and a live_numeric record handed ONLY to that cell's
# processes as an `env` prefix inside the launch line — bash-level, against a fake writer that renders its summary through the
# REAL lora_liveness.py --summary, so the asserted log line is the line a job prints.                                         #
# --------------------------------------------------------------------------- #

LIVENESS_ENV = "SISTEER_LOW_DOSE_LIVENESS_RECORD"

FAKE_WRITER = """#!/usr/bin/env python
import json, subprocess, sys
from pathlib import Path
argv = sys.argv[1:]
if argv[:1] == ["--summary"]:
    sys.exit(subprocess.call([sys.executable, {real!r}, "--summary", argv[1]]))   # the REAL renderer of the launcher's line
opts, i = {{}}, 0
while i < len(argv):
    if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
        opts[argv[i]] = argv[i + 1]; i += 2
    else:
        opts[argv[i]] = True; i += 1
with open({calls!r}, "a") as f:
    f.write(json.dumps(opts) + "\\n")
plan = {plan!r}
verdict, rc = plan[opts["--cell"]]
if verdict is None:
    print("boom: the writer crashed before writing anything", file=sys.stderr); sys.exit(rc)
rec = {{"schema": "low_dose_liveness/1", "verdict": verdict, "cell": opts["--cell"], "routing_table": opts["--routing-table"],
       "tag": opts.get("--tag"), "view": opts.get("--view"), "nominal_dose": 0.1, "intervention_hash_distinct": True,
       "continuous_effect": {{"diag": {{"steered_divergence_median": 0.0421, "noise_ceiling": 0.0011, "bar": 0.0022}}}},
       "text_divergence_observation": {{"n_probes": 5, "n_differing": 0}}}}
Path(opts["--out"]).parent.mkdir(parents=True, exist_ok=True)
Path(opts["--out"]).write_text(json.dumps(rec))
sys.exit(rc)
"""

LIVENESS_PLAN = {"cellA": ("live_numeric", 0), "cellB": ("not_live", 2), "cellC": ("unevaluable", 3), "cellD": (None, 1)}


def _liveness_job(tmp_path: Path, *, writer: bool = True) -> Path:
    """A fake job: REPO_DIR = tmp (with or without the writer at the sha's path), one standup dir holding the routing table."""
    (tmp_path / "serve" / "standup").mkdir(parents=True, exist_ok=True)
    (tmp_path / "serve" / "standup" / "routing_table.json").write_text("{}")
    if writer:
        fake = tmp_path / "scripts/sdfbase/full22/sampled/lora_liveness.py"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text(FAKE_WRITER.format(real=str(SAMPLED / "lora_liveness.py"), calls=str(tmp_path / "calls.jsonl"), plan=LIVENESS_PLAN))
    return tmp_path


def _liveness_bash(snippet: str, tmp_path: Path, **env) -> subprocess.CompletedProcess:
    pre = (f'REPO_DIR="{tmp_path}"; SERVE_DIR="{tmp_path}/serve"; TABLE="{tmp_path}/serve/standup/routing_table.json"; '
           f'PROXY_URL=http://127.0.0.1:9/v1; TAG=llama8b_it_d0p1; VIEW=it_d0p1; MPY="{PY}"; MAX_DRIVERS=64; '
           'f22_log(){ echo "$*"; }; '
           'f22_queue_launch(){ local tag="$1"; shift; printf "LAUNCH %s |" "$tag"; printf " %s" "$@"; echo; }; ')
    return _bash(pre + snippet, **env)


def test_every_steered_driver_launch_of_the_sbatch_names_its_cell_and_the_smoke_runs_after_the_writer():
    src = (SAMPLED / "target_sampled.sbatch").read_text()
    # The sampled lane's RESOLVED vocabulary, not one file: common_sampled.sh sources common.sh, and
    # f22s_low_dose_liveness / f22s_launch_cell now live in the shared file because the ATTACKS leg calls
    # the liveness writer and never sources common_sampled.sh. One definition serves both legs; asserting
    # on the concatenation keeps every count below exact (each symbol still appears exactly once across
    # the pair) while no longer pinning which of the two files holds it.
    common = (SAMPLED / "common_sampled.sh").read_text() + "\n" + (F22 / "common.sh").read_text()
    # no bare f22_queue_launch is left in the sbatch: the 11 driver launches (static shards, the 8192 / 16384 static escalations
    # + main's item-filtered esc16, controls 512 / rung / 16384, gsm8k 8192 (per arm, main's sampled + greedy comparator) / 16384,
    # mmlu) all go through the per-cell launcher with the loop's cell (the liveness record reaches EVERY steered driver)
    assert "f22_queue_launch " not in src
    launches = re.findall(r'f22s_launch_cell "([^\n]*?)" "\$cell" "\$MPY" ', src)
    assert len(launches) == 11, launches
    assert {l.split("__")[0] for l in launches} == {"${fam}", "esc", "esc16", "controls512", "controls${RUNG}", "ext16384", "controls16384",
                                                    "gsm8k16384", "gsm8k", "gsm8k_${arm}", "mmlu"}
    assert src.count("scripts/served_capability_run.py generate") == 7 and src.count('"$F22S_DIR/escalate_static.py"') == 3
    assert src.count("scripts/jailbreak_run.py generate") == 1
    # the prefix is `env VAR=record` in front of the interpreter, per invocation; nothing exports it into the job
    assert f'f22_queue_launch "$tag" env "{LIVENESS_ENV}=$rec" "$@"' in common
    assert all(l.lstrip().startswith("#") for l in src.splitlines() if LIVENESS_ENV in l)   # the sbatch names it in prose only
    assert common.count(f"export {LIVENESS_ENV}") == 1 and "( export " in common   # the smoke's subshell, and only there
    # every standup's positive control runs the writer FIRST, then the smoke (with the probed cell's record)
    assert src.count("f22s_positive_control ") == 11 == src.count("f22_standup attacks")   # + main's esc16 + gsm8k-arm standups
    body = common[common.index("f22s_positive_control(){"):]
    body = body[:body.index("\n}")]
    assert body.index("f22s_low_dose_liveness") < body.index("f22_positive_control ")
    # the writer never fails the job, never blanks the GPU var (the sampled shell files may not assign it), records per standup
    fn = common[common.index("f22s_low_dose_liveness(){"):]
    fn = fn[:fn.index("\n}")]
    assert fn.count("return 0") == 2 and "return 1" not in fn and "exit" not in fn
    assert 'out="$dir/liveness_low_dose_${cell}.json"' in fn and 'dir="$(dirname "$TABLE")"' in fn
    assert '--proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell"' in fn and '${F22_THINKING_ON:+--thinking-on}' in fn
    assert 'f22_log "liveness $cell: $line"' in fn


def test_the_writer_runs_once_per_steered_cell_and_only_a_live_numeric_record_reaches_that_cells_launch_line(tmp_path):
    job = _liveness_job(tmp_path)
    cells = "CELLS=(none cellA cellB cellC cellD prompted__safety_refusal)"
    launch_all = ('for c in "${CELLS[@]}"; do f22s_launch_cell "ctl__$c" "$c" "$MPY" scripts/served_capability_run.py generate '
                  '--cell "$c" --routing-table "$TABLE"; done; ')
    r = _liveness_bash(cells + '; f22s_low_dose_liveness; echo "rc=$?"; echo "records=${!F22S_LIVENESS_RECORD[*]}"; ' + launch_all +
                       'echo "env_in_job=${' + LIVENESS_ENV + ':-unset}"', job)
    assert r.returncode == 0, (r.stdout, r.stderr)
    out = r.stdout.splitlines()
    standup = f"{job}/serve/standup"
    # ONE line per steered cell — the REAL renderer's shape — none for `none` or the prompted baseline
    live = [l for l in out if l.startswith("liveness ")]
    assert live[:3] == [
        "liveness cellA: live_numeric median=0.0421 bar=0.0022 n_differing=0/5 hash_distinct=True noise=0.0011 nominal_dose=0.1",
        "liveness cellB: not_live median=0.0421 bar=0.0022 n_differing=0/5 hash_distinct=True noise=0.0011 nominal_dose=0.1",
        "liveness cellC: unevaluable median=0.0421 bar=0.0022 n_differing=0/5 hash_distinct=True noise=0.0011 nominal_dose=0.1"]
    assert len(live) == 4 and live[3].startswith(f"liveness cellD: writer failed rc=1, no record at {standup}/liveness_low_dose_cellD.json")
    assert "boom: the writer crashed" in live[3]                   # the writer's own last words ride the line
    assert "rc=0" in out and "records=cellA" in out and "env_in_job=unset" in out   # a failed writer never fails the job; nothing leaks
    # the writer was called once per steered cell, bound to THIS standup's proxy + table, into the standup dir
    calls = [json.loads(l) for l in (job / "calls.jsonl").read_text().splitlines()]
    assert [c["--cell"] for c in calls] == ["cellA", "cellB", "cellC", "cellD"]
    for c in calls:
        assert c["--proxy-url"] == "http://127.0.0.1:9/v1" and c["--routing-table"] == f"{standup}/routing_table.json"
        assert c["--out"] == f"{standup}/liveness_low_dose_{c['--cell']}.json" and c["--tag"] == "llama8b_it_d0p1" and c["--view"] == "it_d0p1"
        assert "--thinking-on" not in c
    assert sorted(p.name for p in (job / "serve/standup").glob("liveness_low_dose_*.json")) == [
        "liveness_low_dose_cellA.json", "liveness_low_dose_cellB.json", "liveness_low_dose_cellC.json"]
    assert (job / "serve/drivers/liveness_cellD.log").is_file()
    # THE LAUNCH LINES: the env prefix in front of the interpreter for cellA ONLY; every other cell's line is byte-identical to
    # a bare f22_queue_launch (not_live, unevaluable, a crashed writer, `none`, the prompted baseline)
    launches = [l for l in out if l.startswith("LAUNCH ")]
    bare = lambda c: f"LAUNCH ctl__{c} | {PY} scripts/served_capability_run.py generate --cell {c} --routing-table {standup}/routing_table.json"  # noqa: E731
    assert launches == [bare("none"),
                        f"LAUNCH ctl__cellA | env {LIVENESS_ENV}={standup}/liveness_low_dose_cellA.json {PY} scripts/served_capability_run.py "
                        f"generate --cell cellA --routing-table {standup}/routing_table.json",
                        bare("cellB"), bare("cellC"), bare("cellD"), bare("prompted__safety_refusal")]
    # the record the launch line names is one the policy ADMITS for that cell on that table (and refuses for another cell)
    sys.path.insert(0, str(F22))
    import low_dose_liveness_policy as pol
    rec_a = f"{standup}/liveness_low_dose_cellA.json"
    assert pol.consult("cellA", f"{standup}/routing_table.json", env={LIVENESS_ENV: rec_a})[0] is not None
    assert pol.consult("cellB", f"{standup}/routing_table.json", env={LIVENESS_ENV: rec_a})[0] is None
    assert pol.consult("cellB", f"{standup}/routing_table.json", env={LIVENESS_ENV: f"{standup}/liveness_low_dose_cellB.json"})[0] is None
    # a thinking-ON standup passes the flag through (the proxy requires petri_thinking_on on lens-route requests)
    (job / "calls.jsonl").unlink()
    r = _liveness_bash("F22_THINKING_ON=1; CELLS=(none cellA); f22s_low_dose_liveness", job)
    assert r.returncode == 0, r.stderr
    assert [json.loads(l).get("--thinking-on") for l in (job / "calls.jsonl").read_text().splitlines()] == [True]
    # a NEW standup resets the map: a record of the previous standup never rides into the next one's drivers
    r = _liveness_bash('CELLS=(none cellA); f22s_low_dose_liveness; echo "before=${!F22S_LIVENESS_RECORD[*]}"; '
                       'CELLS=(none cellB); f22s_low_dose_liveness; echo "after=${!F22S_LIVENESS_RECORD[*]:-}"; '
                       'f22s_launch_cell t cellA "$MPY" x', job)
    assert r.returncode == 0, r.stderr
    assert "before=cellA" in r.stdout and "after=" in r.stdout and f"LAUNCH t | {PY} x" in r.stdout


def test_a_job_at_a_sha_without_the_writer_is_byte_identical_to_before(tmp_path):
    job = _liveness_job(tmp_path, writer=False)
    r = _liveness_bash('CELLS=(none cellA); f22s_low_dose_liveness; echo "rc=$? records=${!F22S_LIVENESS_RECORD[*]:-}"; '
                       'f22s_launch_cell "controls512__cellA" cellA "$MPY" scripts/served_capability_run.py generate --cell cellA', job)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "liveness: no scripts/sdfbase/full22/sampled/lora_liveness.py at this sha — no numeric record for any cell; the text-divergence gate is unchanged",
        "rc=0 records=",
        f"LAUNCH controls512__cellA | {PY} scripts/served_capability_run.py generate --cell cellA"]
    assert not (job / "serve/drivers").exists()


def test_the_smoke_gets_the_record_of_the_cell_it_probes_in_its_own_process_only(tmp_path):
    job = _liveness_job(tmp_path, writer=False)
    stub_writer = ('f22s_low_dose_liveness(){ F22S_LIVENESS_RECORD=(); F22S_LIVENESS_RECORD[cellA]=/rec/a.json; '
                   'F22S_LIVENESS_RECORD[cellB]=/rec/b.json; }; ')
    stub_smoke = 'f22_positive_control(){ echo "smoke cell=$1 tag=$2 env=${' + LIVENESS_ENV + ':-unset}"; return "${SMOKE_RC:-0}"; }; '
    r = _liveness_bash(
        stub_writer + stub_smoke +
        'CELLS=(none cellA cellB); f22s_positive_control static_base; echo "rc=$? env_in_job=${' + LIVENESS_ENV + ':-unset}"; '
        'CELLS=(none prompted__safety_refusal cellB cellA); f22s_positive_control mmlu; echo "rc=$?"; '       # the FIRST object cell
        'SMOKE_RC=1; CELLS=(cellA); f22s_positive_control controls_512; echo "rc=$?"; '                        # a refusal still refuses
        'SMOKE_RC=0; f22s_low_dose_liveness(){ F22S_LIVENESS_RECORD=(); }; CELLS=(none cellA); f22s_positive_control controls_2048; echo "rc=$?"; '
        'CELLS=(none prompted__safety_refusal); f22s_positive_control gsm8k_thinking_on; echo "rc=$?"', job)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "smoke cell=cellA tag=static_base env=/rec/a.json",
        "rc=0 env_in_job=unset",
        "smoke cell=cellB tag=mmlu env=/rec/b.json",
        "rc=0",
        "smoke cell=cellA tag=controls_512 env=/rec/a.json",
        "rc=1",
        "smoke cell=cellA tag=controls_2048 env=unset",     # no admitting record: the smoke's process is byte-identical to before
        "rc=0",
        "positive control (gsm8k_thinking_on): no object cell among none prompted__safety_refusal — nothing steered to prove; skipped",
        "rc=0"]


def test_capability_mmlu_block_owns_its_concurrency_when_gsm8k_is_omitted():
    """The capability leg runs MMLU after an OPTIONAL GSM8K block (skipped whenever the roster omits gsm8k — every
    strength-frontier dose view, llama8b/shortconst). Under `set -u` the MMLU block may reference no variable that only the
    GSM8K block assigns: on 2026-09-16 an MMLU-only leg died at the MMLU log line with `CONC: unbound variable` after its
    standup and positive controls had passed. The MMLU block computes MMLU_CONC itself and the driver call uses it."""
    src = (SAMPLED / "target_sampled.sbatch").read_text()
    start = src.index('f22s_positive_control mmlu || { RC=1; exit 1; }')
    end = src.index('f22_log "phase CAPABILITY mmlu done', start)
    block = src[start:end]
    assert 'MMLU_CONC=$(( TARGET_S / NCELLS )); [ "$MMLU_CONC" -ge 8 ] || MMLU_CONC=8' in block
    assert '--controls mmlu --concurrency "$MMLU_CONC" --top-logprobs "$CTL_TOP_LOGPROBS"' in block
    assert "$CONC" not in block and "${CONC}" not in block, "the MMLU block must not read the GSM8K block's CONC"
    assert block.index("MMLU_CONC=$((") < block.index('concurrency/cell=$MMLU_CONC') < block.index('--concurrency "$MMLU_CONC"')
    # the GSM8K block keeps its own CONC (unchanged), and it still sits inside the RUN_GSM8K conditional above the MMLU block
    gs = src.index('CONC=$(( TARGET_S / NCELLS )); [ "$CONC" -ge 8 ] || CONC=8', src.index("PHASE_KIND\" = capability"))
    assert gs < start and 'fi   # RUN_GSM8K' in src[gs:start]


# --------------------------------------------------------------------------- #
# qwen-base-seed-replicates: the per-view seed base                                                                        #
# --------------------------------------------------------------------------- #
def test_the_replicate_views_draw_under_their_own_seed_base_and_every_existing_view_keeps_the_campaigns(tmp_path, monkeypatch):
    """`seed_base_by_view` (configs/sdfbase_sampled_decode.yaml) names the three replicate TAGs at eval_seed's bases; `expected_block`
    reads `seed_base_of_tag(presets, tag)` — the campaign base for every other TAG in the scope and the view's own base for a
    replicate TAG (the 405 committed configs are byte-identical: `--check`). The views are DECLARED (SEEDREP_VIEWS) and PROMOTED
    into SAMPLED_VIEWS with their roster views (reseed 2026-09-15): the committed replicate configs carry their bases, the
    render is rehearsed over a SYNTHETIC wave config too, and the guards that force the promotion (a roster view the scope does
    not render) and pin the declaration (a wrong or stray base) refuse by name."""
    import copy
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "seedrep"))
    import eval_seed as es
    assert msc.SEEDREP_VIEWS == {"qwen35_4b": ("s1", "s2", "s3")} and msc.seedrep_tags() == ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"]
    # the literal training seed -- the three s-views' rows, + the six seeded control TAGs (SEEDED_LANE_VIEWS)
    # at THEIR training seed, which the rule table carries as literal data rows and the data file must equal
    LANE_ROWS = {f"qwen35_4b_dm1_{cls}s{k}": k for k in (1, 2, 3) for cls in ("ctllora64", "ctlrelu64")}
    assert msc.SEEDED_LANE_VIEWS == {"qwen35_4b": tuple(f"dm1_{cls}s{k}" for k in (1, 2, 3) for cls in ("ctllora64", "ctlrelu64"))}
    assert {t: msc.SEED_BASE_RULE_BY_TAG[t] for t in es.SEED_BASE_BY_TAG} == {"qwen35_4b_s1": 1, "qwen35_4b_s2": 2, "qwen35_4b_s3": 3} == es.SEED_BASE_BY_TAG
    assert PRESETS["seed_base_by_view"] == {**es.SEED_BASE_BY_TAG, **LANE_ROWS} == msc.SEED_BASE_RULE_BY_TAG
    assert all(msc.seed_of_lane_view(v) == k for k in (1, 2, 3) for v in (f"dm1_ctllora64s{k}", f"dm1_ctlrelu64s{k}"))
    with pytest.raises(ValueError, match="not a seeded lane view"):
        msc.seed_of_lane_view("dm1_ctllora64")
    assert all(msc.in_scope("qwen35_4b", v) for v in ("s1", "s2", "s3"))              # PROMOTED with their roster views
    # every other TAG in the rendered scope reads the CAMPAIGN base and every committed config says so; the 9 replicate configs
    # (3 views x 1 trial x base / cap8192 / cap16384) carry THEIR view's base = the literal training seed
    assert all(msc.seed_base_of_tag(PRESETS, tag) == 20260909 for tag in TAGS if tag not in SEEDREP_TAGS)
    for path, tag, trial, esc in OUTPUTS:
        base = es.seed_base(es.seed_of_view(tag.rsplit("_", 1)[1])) if tag in SEEDREP_TAGS else 20260909
        assert yaml.safe_load(path.read_text())["sampled_decode"]["seed_base"] == base, path.name
        assert f"seed_base {base}, trial {trial} of {msc.n_trials_of_tag(PRESETS, tag)}." in path.read_text()          # the header's provenance line
    assert sum(tag in SEEDREP_TAGS for _p, tag, *_ in OUTPUTS) == 9
    # a replicate TAG reads ITS base through the one lookup
    for s in (1, 2, 3):
        assert msc.seed_base_of_tag(PRESETS, f"qwen35_4b_s{s}") == es.seed_base(s) != 20260909
    # the promotion has landed: the scope carries the three views after the dose views (the ReLU ladder's included); the render is rehearsed over a synthetic
    # s1 wave config too (the qwen main config renamed) — the block and the header carry the replicate base, everything else is
    # the source config byte for byte (the generator's own _verify)
    promoted = dict(msc.SAMPLED_VIEWS)
    assert promoted["qwen35_4b"] == ("main", "it", *msc.DOSE_VIEWS_IT, *msc.DOSE_VIEWS, *msc.RELU_DOSE_VIEWS["qwen35_4b"], *msc.SEEDREP_VIEWS["qwen35_4b"])
    src = (REPO / "configs/sdfbase_full22_qwen35_4b.yaml").read_text()
    for s in (1, 2, 3):
        tag = f"qwen35_4b_s{s}"
        (tmp_path / f"sdfbase_full22_{tag}.yaml").write_text(
            src.replace("out_dir: out/sdfbase_full22_qwen35_4b\n", f"out_dir: out/sdfbase_full22_{tag}\n")
               .replace("private_run: sdfbase_full22_qwen35_4b\n", f"private_run: sdfbase_full22_{tag}\n"))
    monkeypatch.setattr(msc, "wave_config_path", lambda tag: tmp_path / f"sdfbase_full22_{tag}.yaml" if tag.startswith("qwen35_4b_s") else REPO / "configs" / f"sdfbase_full22_{tag}.yaml")
    assert msc.key_of("qwen35_4b_s2") == "qwen35_4b" and msc.in_scope("qwen35_4b", "s2")
    for s in (1, 2, 3):
        tag = f"qwen35_4b_s{s}"
        for trial, esc in ((0, False), (0, 8192), (0, 16384)):           # trial 0 is the ONE trial of a seed view
            block = msc.expected_block(PRESETS, tag, trial, esc)
            assert block["seed_base"] == es.seed_base(s) and block["trial"] == trial and block["arm"] == "vendor_recommended"
            assert {k: block[k] for k in PINNED["qwen35_4b"]} == PINNED["qwen35_4b"]          # the KEY's preset, not a per-view one
            text = msc.render(PRESETS, tag, trial, esc)                                          # _verify inside: the wave config as data + the block
            assert f"seed_base {es.seed_base(s)}, trial {trial}" in text and yaml.safe_load(text)["sampled_decode"] == block
            assert yaml.safe_load(text)["out_dir"] == f"out/{msc.sampled_name(tag, trial, esc)}"
        # the s-view's base differs from the main view's under the same key, trial and cap
        assert msc.expected_block(PRESETS, "qwen35_4b", 0)["seed_base"] == 20260909 != msc.expected_block(PRESETS, tag, 0)["seed_base"]
    # the GUARD: a roster that names a replicate view the scope does not render refuses (the promotion is forced, never forgotten):
    # under the UN-promoted scope today's roster (which names s1/s2/s3) refuses, and so does a roster naming s1 alone
    monkeypatch.setattr(msc, "SAMPLED_VIEWS", {k: tuple(v) for k, v in promoted.items() if True} | {"qwen35_4b": ("main", "it", *msc.DOSE_VIEWS_IT, *msc.DOSE_VIEWS, *msc.RELU_DOSE_VIEWS["qwen35_4b"])})
    with pytest.raises(SystemExit, match="promote it"):
        msc.check_pins(PRESETS, ROSTER)
    roster2 = copy.deepcopy(ROSTER)
    for v in ("s1", "s2", "s3"):
        roster2["targets"]["qwen35_4b"]["views"].pop(v, None)
    roster2["targets"]["qwen35_4b"]["views"]["s1"] = {"config": "x", "manifest": "y", "cells": {}}
    with pytest.raises(SystemExit, match="promote it"):
        msc.check_pins(PRESETS, roster2)
    monkeypatch.setattr(msc, "SAMPLED_VIEWS", promoted)
    msc.check_pins(PRESETS, ROSTER)                                                               # the promoted scope renders every roster view: no refusal
    # the DECLARATION is pinned to the rule: a wrong base, a stray TAG, a missing TAG, a base equal to the campaign's — each refuses
    def presets_with(sbv):
        d = copy.deepcopy(yaml.safe_load(msc.PRESETS.read_text()))
        if sbv is None:
            d.pop("seed_base_by_view")
        else:
            d["seed_base_by_view"] = sbv
        f = tmp_path / f"p_{abs(hash(json.dumps(sbv, sort_keys=True)))}.yaml"
        f.write_text(yaml.safe_dump(d))
        return f
    good = dict(PRESETS["seed_base_by_view"])
    with pytest.raises(SystemExit, match="must equal the rule"):
        msc.load_presets(presets_with({**good, "qwen35_4b_s2": good["qwen35_4b_s2"] + 1}))
    with pytest.raises(SystemExit, match="no strays"):
        msc.load_presets(presets_with({**good, "qwen35_4b_it": 7}))
    with pytest.raises(SystemExit, match="no strays"):
        msc.load_presets(presets_with({k: v for k, v in good.items() if k != "qwen35_4b_s3"}))
    with pytest.raises(SystemExit, match="must equal the rule"):
        msc.load_presets(presets_with({**good, "qwen35_4b_s1": 20260909}))
    # a preset file WITHOUT the map: legal (no replicate view declared in data) but check_pins then refuses the declared views
    no_map = msc.load_presets(presets_with(None))
    assert "seed_base_by_view" not in no_map and msc.seed_base_of_tag(no_map, "qwen35_4b_s1") == 20260909
    with pytest.raises(SystemExit, match="carries no seed_base_by_view"):
        msc.check_pins(no_map, ROSTER)
