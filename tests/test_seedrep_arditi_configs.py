"""qwen-seedrep-native-battery — the LANE's Arditi client configs.

Pinned: the committed outputs are byte-identical to the renderer (`--check` rc 0) and `--check` refuses a
drifted byte; the source copies are the upstream files at a sha >= 0f3576f3 (recorded, sha256-held); every
client config carries `seed_base ∈ {1,2,3}` (the LITERAL training seed), `n_trials 1`, `trial 0`, the six
sampler keys read from the registry rows, a `private_run` / `out_dir` distinct per seed AND per rung and
disjoint from the upstream namespaces, `bon.max_new_tokens 4096`, the rung lineage in the same seed's
namespace, and is otherwise the upstream standup config AS DATA; the manifest carries exactly the declared
`target_decode` block and the roster's `serving_impl_commit`, everything else upstream's; the seed closure
reads the block's base.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "seedrep"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))
import render_arditi_seedrep_configs as ra  # noqa: E402
import make_sampled_configs as ms  # noqa: E402

from sisteer.eval.attacks.static_common import static_sampling_seed  # noqa: E402
from sisteer.eval import sampled_controls as sc  # noqa: E402

SIX = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
VENDOR = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_check_is_green_on_the_committed_outputs_and_refuses_a_drifted_byte(tmp_path, monkeypatch, capsys):
    assert ra.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert out.count("ok: ") == 14 and "DRIFT" not in out       # standup copy + manifest + 9 client configs + 3 wave configs
    # drift: a copy of the lane dir with ONE byte changed in one client config
    lane = tmp_path / "configs" / "seedrep"
    lane.mkdir(parents=True)
    for p in ra.LANE_DIR.iterdir():
        if p.is_file():
            (lane / p.name).write_bytes(p.read_bytes())
    src = lane / "source"
    (src / "configs" / "capeval" / "manifests").mkdir(parents=True)
    for rel in ra.SOURCE_FILES:
        (src / rel).write_bytes((ra.SOURCE_DIR / rel).read_bytes())
    (src / "SOURCE.json").write_bytes(ra.SOURCE_JSON.read_bytes())
    victim = lane / f"{ra.sampled_name(2, 8192)}.yaml"
    victim.write_text(victim.read_text().replace("seed_base: 2", "seed_base: 20260909"))
    monkeypatch.setattr(ra, "LANE_DIR", lane)
    monkeypatch.setattr(ra, "SOURCE_DIR", src)
    monkeypatch.setattr(ra, "SOURCE_JSON", src / "SOURCE.json")
    assert ra.main(["--check"]) == 1
    assert f"DRIFT: {victim}" in capsys.readouterr().out
    # an edited SOURCE copy refuses outright (the verbatim copy is held to its recorded sha256)
    (src / ra.STANDUP_SOURCE).write_bytes((src / ra.STANDUP_SOURCE).read_bytes() + b"\n# edited\n")
    with pytest.raises(SystemExit, match="does not match SOURCE.json"):
        ra.main(["--check"])


def test_source_copies_are_scales_files_at_a_sha_at_or_after_0f3576f3():
    doc = json.loads(ra.SOURCE_JSON.read_text())
    assert doc["source_sha"].startswith(ra.MIN_SOURCE_SHA) and doc["min_source_sha"] == doc["source_sha"]
    assert len(doc["source_sha"]) == 40
    for rel, spec in doc["files"].items():
        data = (ra.SOURCE_DIR / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == spec["sha256"]
    # the three digests are upstream's (identical on 0f3576f3, arditi-qwen-all-positions and arditi-allpos-reseed-c0p3)
    assert doc["files"][ra.STANDUP_SOURCE]["sha256"].startswith("1c5d8b33ef355bdf")
    assert doc["files"][ra.SAMPLED_SOURCE]["sha256"].startswith("e6a7da072adde67f")
    assert doc["files"][ra.MANIFEST_SOURCE]["sha256"].startswith("6898b5bae7f6d7db")
    # the standup lane copy is byte-identical to the source (the engine is stood up from the lane path)
    assert (ra.LANE_DIR / ra.SOURCE_FILES[ra.STANDUP_SOURCE]).read_bytes() == (ra.SOURCE_DIR / ra.STANDUP_SOURCE).read_bytes()
    # when git can see the sha, it descends from MIN_SOURCE_SHA (a refresh from an older sha refuses)
    rc = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{doc['source_sha']}^{{commit}}"], capture_output=True).returncode
    if rc == 0:
        assert subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", ra.MIN_SOURCE_SHA, doc["source_sha"]]).returncode == 0
        with pytest.raises(SystemExit, match="does not descend"):
            ra.refresh_source(f"{ra.MIN_SOURCE_SHA}~1")


@pytest.mark.parametrize("s", ra.SEEDS)
@pytest.mark.parametrize("esc", [False, 8192, 16384])
def test_each_client_config_is_scales_standup_config_plus_the_lane_keys(s, esc):
    src = _load(ra.SOURCE_DIR / ra.STANDUP_SOURCE)
    got = _load(ra.output_path(s, esc))
    cap = ms.rung_cap(esc)
    name = ra.sampled_name(s, esc)
    block = got["sampled_decode"]
    assert (block["arm"], block["trial"], block["n_trials"], block["seed_base"]) == ("vendor_recommended", 0, 1, s)
    assert {k: block[k] for k in SIX} == VENDOR
    assert (block["source_kind"], block["applies_to"]) == ("model_card", ["static_families", "controls"])
    assert block["static_cap"] == (cap or 4096) == got["attacks"]["static_max_new_tokens"]
    assert block["controls_cap"] == 512 == got["attacks"]["controls_max_new_tokens"]
    assert block["readouts"]["gsm8k"]["cap"] == 8192 and block["readouts"]["gsm8k"]["thinking"] == "on"
    assert block["readouts"]["mmlu"]["instrument"] == "letter_logprob_argmax"
    if cap:
        prev_cap = 4096 if cap == 8192 else 8192
        assert block["escalated_from"] == {"kind": "static", "cap": prev_cap,
                                           "private_run": ra.sampled_name(s, False if cap == 8192 else 8192)}
        assert block["escalated_from"]["private_run"].startswith(f"{ra.LANE_TAG}_s{s}_")   # the SAME seed's predecessor
    else:
        assert "escalated_from" not in block
    assert got["private_run"] == name and got["out_dir"] == f"out/{name}"
    assert got["bon"]["max_new_tokens"] == 4096 and src["bon"]["max_new_tokens"] == 512
    # everything else is upstream's, as data
    changed = {k for k in set(got) | set(src) if got.get(k) != src.get(k)}
    assert changed == {"out_dir", "private_run", "sampled_decode", "bon"} | ({"attacks"} if cap else set())
    b, sb = dict(got["bon"]), dict(src["bon"])
    b.pop("max_new_tokens"); sb.pop("max_new_tokens")
    assert b == sb
    a, sa = dict(got["attacks"]), dict(src["attacks"])
    a.pop("static_max_new_tokens"); sa.pop("static_max_new_tokens")
    assert a == sa
    assert got["explicit_cells"] == ["none", "arditi_allpos"] and got["c"] == 0.3 and got["dose_unit"] == "raw_coefficient"
    # the block equals make_sampled_configs' shape for a qwen35_4b TAG except the three lane differences
    ref = ms.expected_block(ms.load_presets(), "qwen35_4b_s1", 0, esc)
    for k in ("n_trials", "seed_base", "escalated_from"):
        ref.pop(k, None);
    lane = dict(block)
    for k in ("n_trials", "seed_base", "escalated_from"):
        lane.pop(k, None)
    assert lane == ref


def test_the_nine_namespaces_are_distinct_per_seed_and_rung_and_disjoint_from_scales():
    names = [_load(p)["private_run"] for p, _s, _e in ra.outputs()]
    dirs = [_load(p)["out_dir"] for p, _s, _e in ra.outputs()]
    assert len(names) == 9 == len(set(names)) and len(set(dirs)) == 9
    assert sorted(names) == sorted(f"{ra.LANE_TAG}_s{s}_sampled_t0{suf}" for s in (1, 2, 3) for suf in ("", "_cap8192", "_cap16384"))
    scale = {_load(ra.SOURCE_DIR / ra.SAMPLED_SOURCE)["private_run"], _load(ra.SOURCE_DIR / ra.STANDUP_SOURCE)["private_run"]}
    assert scale == {"arditi_allpos_qwen35_4b_ar_c0p3_sampled_t0", "arditi_allpos_qwen35_4b_ar_c0p3"}
    assert not (set(names) & scale)
    # per seed: the three rungs share the seed prefix and differ only by rung
    for s in ra.SEEDS:
        mine = {n for n in names if n.startswith(f"{ra.LANE_TAG}_s{s}_")}
        assert len(mine) == 3
    # the grader twin keys stores by private_run — three seed replicates, three keys
    assert len({_load(ra.output_path(s))["private_run"] for s in ra.SEEDS}) == 3


def test_the_seed_closure_reads_the_blocks_base():
    """The per-item seed the static families / controls draw is the rule over the block's `seed_base` and
    `trial` — the LITERAL training seed s, one trial (S2 recomputes every request's seed this way)."""
    for s in ra.SEEDS:
        block = _load(ra.output_path(s))["sampled_decode"]
        assert static_sampling_seed(block["seed_base"], block["trial"], "pasttense", "hb_0001", 0) == static_sampling_seed(s, 0, "pasttense", "hb_0001", 0)
        assert sc.sampling_seed(block["seed_base"], block["trial"], "xstest", "xstest_aaaa1111") == sc.sampling_seed(s, 0, "xstest", "xstest_aaaa1111")
    seeds = {static_sampling_seed(s, 0, "pasttense", "hb_0001", 0) for s in ra.SEEDS}
    assert len(seeds) == 3 and static_sampling_seed(20260909, 0, "pasttense", "hb_0001", 0) not in seeds


def test_the_registry_rows_and_the_presets_yaml_agree_on_the_six_keys():
    presets = ms.load_presets()
    assert ra.vendor_preset(presets) == VENDOR
    for ref in (ra.STATIC_POLICY_REF, ra.CONTROLS_POLICY_REF):
        assert {k: float(v) for k, v in ra.registry_preset(ref).items()} == {k: float(v) for k, v in VENDOR.items()}
    # a disagreeing yaml row refuses (two statements of the vendor row cannot both be true)
    bad = json.loads(json.dumps(presets))
    bad["presets"]["qwen35_4b"]["preset"]["top_k"] = 40
    with pytest.raises(SystemExit, match="disagree"):
        ra.vendor_preset(bad)


def test_the_manifest_carries_ws1b_target_decode_and_the_rosters_serving_impl_commit_and_nothing_else_changes():
    src = _load(ra.SOURCE_DIR / ra.MANIFEST_SOURCE)["petri"]
    got = _load(ra.LANE_DIR / ra.SOURCE_FILES[ra.MANIFEST_SOURCE])["petri"]
    assert got["target_decode"] == {"mode": "vendor", "policy": "qwen35_4b_it|petri|off|vendor_preset@v1"} == ra.PETRI_TARGET_DECODE_BLOCK
    assert "seed_base" not in got["target_decode"] and "target_seed_base" not in got       # a run knob, never here
    roster = yaml.safe_load(ra.ROSTER.read_text())
    assert got["serving_impl_commit"] == roster["serving_impl_commit"] != src["serving_impl_commit"]
    g, s = dict(got), dict(src)
    g.pop("target_decode"); g.pop("serving_impl_commit"); s.pop("serving_impl_commit")
    # the THIRD permitted change (PETRI_FALLBACK_CHAINS): the chains are the lane's, by name
    for key, want in ra.PETRI_FALLBACK_CHAINS.items():
        assert g.pop(key) == want == ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"], key
        s.pop(key, None)
    assert g == s
    assert got["audit"] == "arditi_allpos_qwen35_4b_ar_c0p3" and got["cells"] == ["none", "arditi_allpos"]
    assert len(got["behaviors"]) == 6 and got["bloom_options"]["epochs"] == 1
    # the block sits where it is rendered on the s-view manifests: after prompted_prefix_hashes, before bloom_options
    text = (ra.LANE_DIR / ra.SOURCE_FILES[ra.MANIFEST_SOURCE]).read_text()
    assert text.index("  prompted_prefix_hashes:") < text.index("  target_decode:") < text.index("  bloom_options:")
    assert "    policy: 'qwen35_4b_it|petri|off|vendor_preset@v1'\n" in text


def _harness_inputs(cm, spec):
    """`petri_harness_config_inputs` exactly as the standup calls it when it writes each cell's harness_config_hash into
    cells_served.json: the FULL behaviour inventory (synthetic BEHAVIOR.md bytes — hashed per file, never read), the REAL
    auditor/judge prompt bytes (the manifest's `override@<hex>` provenance is VERIFIED against them; synthetic bytes would
    refuse), and the Arditi lane's serving identity in the manifest's own mode (lora_native rides the proxy's `lora` path)."""
    behaviors = {k: f"# synthetic BEHAVIOR.md for {k}\n".encode("utf-8") for k in cm.petri_expected_behavior_keys(spec)}
    prompts = {role: (REPO / "configs" / "capeval" / "petri" / "prompts" / f"{role}.md").read_bytes()
               for role in ("auditor", "judge")}
    return cm.petri_harness_config_inputs(spec, behavior_files=behaviors, prompt_files=prompts,
                                          serving_identity=cm.serving_identity("proxy_vllm", "lora", spec.serving_mode))


def test_the_manifests_target_decode_hashes_as_ws1b_asserts_when_the_loader_knows_the_block():
    """When manifest.py defines PetriTargetDecode, the lane manifest's harness hash inputs carry the
    resolved identity `--target-decode vendor` asserts; without that field the loader has none and this
    test states so instead of pretending. The block is hashed on BOTH sides (standup and client), so what
    is pinned here is (1) the identity the one authority resolves for the lane manifest, (2) that it enters
    the hash inputs under `target_decode`, (3) the known-bad/additive half — the same manifest WITHOUT the
    block (the upstream harness world) carries no such key, every other input equal, a different hash."""
    import dataclasses

    from sisteer.capeval import manifest as cm

    if not hasattr(cm, "PetriTargetDecode"):
        pytest.skip("manifest.PetriTargetDecode is not on this tree yet (WS1-B lands it); the merged reseed exercises the hash")
    lane = cm.load_petri_spec(ra.LANE_DIR / ra.SOURCE_FILES[ra.MANIFEST_SOURCE])
    assert lane.target_decode is not None and lane.target_decode.mode == "vendor"
    assert lane.target_decode.policy == ra.PETRI_POLICY_REF
    # (1) the resolved identity — by_ref on the exact versioned ref, not superseded, the six keys AS SENT, the 1024 per-turn cap
    ident = cm.petri_target_decode_identity(lane)
    assert ident["mode"] == "vendor" and ident["policy_id"] == ra.PETRI_POLICY_REF and ident["max_tokens"] == 1024
    assert {k: float(v) for k, v in ident["request_params"].items()} == {k: float(v) for k, v in VENDOR.items()}
    assert ident["seed_rule"] == cm.PETRI_TARGET_SEED_RULE == "per_scenario_turn"
    # (2) ... and it ENTERS the harness hash inputs, keyed `target_decode`, as the standup and the client both hash it
    inputs = _harness_inputs(cm, lane)
    assert inputs["target_decode"] == ident
    # (3) known-bad / additive: the block removed (upstream's harness world) → no key, everything else byte-equal, another hash
    harness = dataclasses.replace(lane, target_decode=None)
    assert cm.petri_target_decode_identity(harness) is None
    h_inputs = _harness_inputs(cm, harness)
    assert "target_decode" not in h_inputs
    assert {k: v for k, v in inputs.items() if k != "target_decode"} == h_inputs
    sha = lambda i: cm.sha256_hex16(cm.canonical_json(i).encode("utf-8"))  # noqa: E731
    assert sha(inputs) != sha(h_inputs)
    # the SOURCE manifest loads as harness: the lane copy is the ONLY one declaring vendor
    src = cm.load_petri_spec(ra.SOURCE_DIR / ra.MANIFEST_SOURCE)
    assert src.target_decode is None and cm.petri_target_decode_identity(src) is None


# --------------------------------------------------------------------------- #
# the per-seed NON-sampled WAVE configs — BoN / multi-turn namespaces          #
# --------------------------------------------------------------------------- #

WS2D_PLAN = next((d for d in (REPO / "scripts" / "modal" / "seedrep_launch_plan.py",   # this tree first (W)
                             REPO.parent / "seedrep-ws2d" / "scripts" / "modal" / "seedrep_launch_plan.py") if d.is_file()),
                 REPO / "scripts" / "modal" / "seedrep_launch_plan.py")
WS5I_GRADER = next((d for d in (REPO / "scripts" / "sdfbase" / "seedrep" / "grade_full22_native.py",   # this tree first (W)
                               REPO.parent / "seedrep-ws5i" / "scripts" / "sdfbase" / "seedrep" / "grade_full22_native.py") if d.is_file()),
                   REPO / "scripts" / "sdfbase" / "seedrep" / "grade_full22_native.py")


def _ast_constants(path: Path, names):
    """Module-level string / tuple constants read off a peer's file by `ast` (their spellings, never re-typed; no import
    of a heavy script). Missing file -> skip; a missing name -> None."""
    import ast
    if not path.is_file():
        pytest.skip(f"{path} is not beside this tree")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {n: None for n in names}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in out:
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, "attr", None) == "compile" and v.args and isinstance(v.args[0], ast.Constant):
                out[node.targets[0].id] = v.args[0].value          # re.compile(r"...") -> the pattern
            else:
                try:
                    out[node.targets[0].id] = ast.literal_eval(v)
                except ValueError:
                    pass
    return out


@pytest.mark.parametrize("s", [1, 2, 3])
def test_each_wave_config_is_scales_standup_config_with_its_own_namespace_and_the_d20_cap_and_nothing_else(s):
    src = _load(ra.SOURCE_DIR / ra.STANDUP_SOURCE)
    path = ra.wave_output_path(s)
    got = _load(path)
    name = ra.wave_name(s)
    assert path == ra.LANE_DIR / f"{ra.LANE_TAG}_s{s}.yaml" and path.stem == name == got["private_run"]
    assert got["out_dir"] == f"out/{name}"
    # the 4096 cap, one trial (upstream's), no sampled block (this is the WAVE config, not the static chain's)
    assert got["bon"]["max_new_tokens"] == 4096 and src["bon"]["max_new_tokens"] == 512
    assert got["bon"]["trials"] == 1 == src["bon"]["trials"]
    assert "sampled_decode" not in got
    # everything else is upstream's, as data: only out_dir / private_run / bon.max_new_tokens differ
    changed = {k for k in set(got) | set(src) if got.get(k) != src.get(k)}
    assert changed == {"out_dir", "private_run", "bon"}
    b, sb = dict(got["bon"]), dict(src["bon"])
    b.pop("max_new_tokens"); sb.pop("max_new_tokens")
    assert b == sb
    assert got["attacks"] == src["attacks"] and got["explicit_cells"] == ["none", "arditi_allpos"] and got["c"] == 0.3
    # the seed base is the drivers' argv, stated in the header (never a key nothing reads); the source sha is recorded
    text = path.read_text()
    assert f"--decode-seed-base {s}" in text and f"--multiturn-target-seed-base {s}" in text
    assert f"source sha {ra.load_source()[0]['source_sha'][:12]}" in text and "seed_base:" not in text
    # byte-identity beyond the header: the source text follows verbatim except the three substituted lines
    src_text = (ra.SOURCE_DIR / ra.STANDUP_SOURCE).read_text()
    body = text[text.index("#\n", text.index("The Petri manifest")) + 2:]
    diff = [(a, b) for a, b in zip(src_text.splitlines(), body.splitlines()) if a != b]
    assert diff == [(f"  max_new_tokens: 512", f"  max_new_tokens: 4096"),
                    (f"out_dir: out/{ra.RUN_TAG}", f"out_dir: out/{name}"),
                    (f"private_run: {ra.RUN_TAG}", f"private_run: {name}")]
    assert len(src_text.splitlines()) == len(body.splitlines())


def test_the_three_wave_namespaces_are_distinct_and_disjoint_from_every_sampled_static_and_scale_namespace():
    waves = {s: _load(ra.wave_output_path(s))["private_run"] for s in ra.SEEDS}
    assert len(set(waves.values())) == 3
    sampled = {_load(p)["private_run"] for p, _s, _e in ra.outputs()}
    scale = {_load(ra.SOURCE_DIR / ra.STANDUP_SOURCE)["private_run"], _load(ra.SOURCE_DIR / ra.SAMPLED_SOURCE)["private_run"],
             _load(ra.LANE_DIR / ra.SOURCE_FILES[ra.STANDUP_SOURCE])["private_run"]}
    assert len(sampled) == 9 and not (set(waves.values()) & sampled) and not (set(waves.values()) & scale)
    # the discovery rule in grade_full22_native.seedrep_arditi_targets (tip >= 8427e4c4): the stem matches
    # ^(?P<base>.+)_seedrep_s(?P<s>[0-9]+)$, private_run == the stem, not a sampled namespace, one namespace per seed
    consts = _ast_constants(WS5I_GRADER, ["ARDITI_WAVE_RE"])
    pattern = consts["ARDITI_WAVE_RE"] or r"^(?P<base>.+)_seedrep_s(?P<s>[0-9]+)$"
    import re as _re
    for s, run in waves.items():
        m = _re.match(pattern, ra.wave_output_path(s).stem)
        assert m and int(m.group("s")) == s and m.group("base") == ra.RUN_TAG and run == ra.wave_output_path(s).stem
        assert not any(_re.match(pattern, p.stem) for p, _s, _e in ra.outputs())    # the sampled files are never wave targets
    # the renderer refuses a wave config whose private_run is not its stem / collides with a sampled namespace
    src_text = (ra.SOURCE_DIR / ra.STANDUP_SOURCE).read_text()
    with pytest.raises(SystemExit, match="not renamed"):
        ra._verify_wave(ra.render_wave(src_text, "0" * 40, 1).replace(f"private_run: {ra.wave_name(1)}", f"private_run: {ra.sampled_name(1)}"), src_text, ra.wave_name(1))
    with pytest.raises(SystemExit, match="carries no sampled_decode block"):
        ra._verify_wave(ra.render_wave(src_text, "0" * 40, 1) + "\nsampled_decode: {}\n", src_text, ra.wave_name(1))
    with pytest.raises(SystemExit, match="keys differing"):
        ra._verify_wave(ra.render_wave(src_text, "0" * 40, 1) + "\nwandb_project: other\n", src_text, ra.wave_name(1))
    with pytest.raises(SystemExit, match="beyond max_new_tokens"):
        ra._verify_wave(ra.render_wave(src_text, "0" * 40, 1).replace("  sigma: 0.4", "  sigma: 0.5"), src_text, ra.wave_name(1))


def test_the_wave_and_client_paths_are_exactly_ws2ds_expected_spellings():
    consts = _ast_constants(WS2D_PLAN, ["ARDITI_WAVE_CONFIG_REL", "ARDITI_CLIENT_CONFIG_REL", "ARDITI_SEEDS", "ARDITI_CONFIG_REL", "ARDITI_MANIFEST_REL"])
    assert consts["ARDITI_WAVE_CONFIG_REL"] and consts["ARDITI_CLIENT_CONFIG_REL"], consts
    seeds = tuple(consts["ARDITI_SEEDS"] or (1, 2, 3))
    assert seeds == ra.SEEDS
    for s in seeds:
        assert (REPO / consts["ARDITI_WAVE_CONFIG_REL"].format(s=s)) == ra.wave_output_path(s)
        for esc, suffix in ((False, ""), (8192, "_cap8192"), (16384, "_cap16384")):
            assert (REPO / consts["ARDITI_CLIENT_CONFIG_REL"].format(s=s, suffix=suffix)) == ra.output_path(s, esc)
    if consts["ARDITI_CONFIG_REL"]:
        assert (REPO / consts["ARDITI_CONFIG_REL"]) == ra.LANE_DIR / ra.SOURCE_FILES[ra.STANDUP_SOURCE]
    if consts["ARDITI_MANIFEST_REL"]:
        assert (REPO / consts["ARDITI_MANIFEST_REL"]) == ra.LANE_DIR / ra.SOURCE_FILES[ra.MANIFEST_SOURCE]
    # every expected file exists and is what the renderer renders (--check's contract)
    for s in seeds:
        assert ra.wave_output_path(s).is_file()


def test_the_fallback_chains_are_an_explicit_rule_over_an_immutable_source():
    """The fallback chains are a generator RULE, not a hand edit: the archived source copy still spells the
    abliterated-Qwen third auditor leg and no judge chain (it is verbatim upstream, sha256-held); the
    rendered manifest carries ``[grok-4.6]`` for BOTH roles by PETRI_FALLBACK_CHAINS -- so a regeneration
    can never re-instate the Qwen leg (which the launch found: the drive passes no Qwen endpoint, so the
    archived chain refuses at leg start) nor drop the judge chain; a source whose chain block is missing
    or already spells a judge chain refuses."""
    src = _load(ra.SOURCE_DIR / ra.MANIFEST_SOURCE)["petri"]
    assert "openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated" in src["auditor_fallback_models"]
    assert "judge_fallback_models" not in src
    got = _load(ra.LANE_DIR / ra.SOURCE_FILES[ra.MANIFEST_SOURCE])["petri"]
    assert got["auditor_fallback_models"] == got["judge_fallback_models"] == ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"]
    text = (ra.LANE_DIR / ra.SOURCE_FILES[ra.MANIFEST_SOURCE]).read_text()
    assert "Huihui" not in text.split("\n  judge_model:")[0].split("auditor_fallback_models:")[1]
    assert text.index("  auditor_fallback_models:") < text.index("  judge_fallback_models:") < text.index("  judge_model:")
    rendered = ra.render_fallback_chains((ra.SOURCE_DIR / ra.MANIFEST_SOURCE).read_text())
    assert yaml.safe_load(rendered)["petri"]["auditor_fallback_models"] == ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"]
    with pytest.raises(SystemExit, match="already spells judge_fallback_models"):
        ra.render_fallback_chains(rendered)
    with pytest.raises(SystemExit, match="exactly one"):
        ra.render_fallback_chains(rendered.replace("  auditor_fallback_models:", "  x_models:"))
