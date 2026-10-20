"""objects/DEMO.json (reproduce/tools/demo_objects.py) and the lighter checkouts' path lists (reproduce/tools/sparse_paths.py).

The committed list names, for each of the seven Qwen3.5-4B interventions the README's demo checkout carries, one object:
the six fitted interventions at ONE training seed, the seed whose difference LoRA the paper's multi-turn figure rates best
(the argmax of the recorded per-seed gains, unrounded, ties to the lower seed), and the refusal direction at its operating
coefficient. Every listed object is a manifest row of that label and fit model, its directory is in the tree, its seed and
dose read back from the sidecars through `sisteer.objects`, each gain is the mean of its two attacks' deltas, and the one
seed makes the set self-consistent: each difference's derived_meta.json names the chosen constitution and control as its
parents, under one model key. The selection itself (`demo_objects.choose`) is pinned on synthetic candidates: the
difference seed wins even where another intervention's own best seed differs, a fitted intervention with no fit at that
seed is refused by name, and a tie goes to the lower seed. The list is current: the digests it records of its inputs (the
notebook's code cells, the reader library, the snapshot, the manifest, the tool) match the tree, so a change to any of
them fails here until the list is recomputed. The path helper prints exactly the demo's object directories plus the code,
works from the committed list when the work tree has none (a sparse clone after its first two commands), and the README's
three checkout blocks spell the commands the helper is designed for. The full recomputation (`--check`, a two-minute run
of the notebook's load chain) is opt-in: SISTEER_SLOW_TESTS=1."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.objects import load_index, resolve

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "objects" / "DEMO.json"
MANIFEST = REPO / "objects" / "MANIFEST.json"
TOOLS = REPO / "reproduce" / "tools"
sys.path.insert(0, str(TOOLS))
import demo_objects  # noqa: E402  (stdlib at import; pandas only inside compute())
import sparse_paths  # noqa: E402

FITTED = {"control_relu", "control_lora", "constitution_relu", "constitution_lora", "difference_relu", "difference_lora"}
BASELINE = "arditi_allpos"
EVALS = ["crescendo", "actorattack"]
#: The one training seed the committed list names (the seed of the best difference LoRA, 41.06 pp against 24.07 for seed 1
#: and 25.03 for seed 2) and the seven objects at it; the four control and constitution objects are exactly the two
#: differences' parents, model key qwen35_4b_base_eb64_s3.
TRAINING_SEED = 3
MODEL_KEY = "qwen35_4b_base_eb64_s3"
EXPECTED_OBJECTS = {
    "arditi_allpos": "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2",
    "control_relu": "c5686c7316b7f79ba31edeea4ecf5d3b237724e399cafb41295478b457471c34",
    "control_lora": "33e7b327049aa28605818fe02970d918a86a2c6eea8f6bbf38a01779b84f9d32",
    "constitution_relu": "cc1ec068380ed8fe51de1e91e8b8fe6b32c95fa9a00bcf381a536007c075bf09",
    "constitution_lora": "a9934f6a796c238d58f4da10615c9f1c211d8b067afcf1330241cdcff967ce42",
    "difference_relu": "0718a3f9edfb033ad84c3322f717dc62b32b6298fe48f0a6c42e9f0a7c031647",
    "difference_lora": "8952728fa0f7fb6a0b48557e9c491073c5080f63178fc64c4f113088e1f2f974",
}
PARENTS = {"difference_lora": ("constitution_lora", "control_lora"), "difference_relu": ("constitution_relu", "control_relu")}


@pytest.fixture(scope="module")
def demo() -> dict:
    return json.loads(DEMO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return {r["object_id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["objects"]}


def test_the_seven_interventions_each_have_one_entry(demo):
    assert demo["schema"] == "demo_objects/2"
    labels = [e["paper_label"] for e in demo["objects"]]
    assert sorted(labels) == sorted(FITTED | {BASELINE}) and len(labels) == 7
    assert demo["metric"]["evals"] == EVALS and demo["metric"]["category"] == "multi-turn"
    assert demo["metric"]["served_model_id"] == "qwen35_4b_it"
    assert demo["source"]["snapshot_id"] == (REPO / "paper" / "snapshot" / "CURRENT").read_text().strip()
    assert (demo["training_seed"], demo["seed_chosen_by"], demo["model_key"]) == (TRAINING_SEED, "difference_lora", MODEL_KEY)
    assert "One training seed" in demo["title"] and "difference_lora" in demo["rule"] and "self-consistent" in demo["rule"]


def test_the_committed_list_names_the_seven_objects_at_the_one_seed(demo):
    """The pin: the committed choice as object ids. A recomputation that moves any of them is a change to talk about."""
    assert {e["paper_label"]: e["object_id"] for e in demo["objects"]} == EXPECTED_OBJECTS
    assert {e["paper_label"]: e["seed"] for e in demo["objects"]} == {BASELINE: None, **{k: TRAINING_SEED for k in FITTED}}


def test_the_inputs_behind_the_committed_list_are_unchanged(demo):
    """The fast currency check: DEMO.json records a sha256 of everything its numbers come from; the tree must still match.
    A mismatch means the list may be stale, and names the command that recomputes it."""
    recorded = demo["source"]["inputs_sha256"]
    now = demo_objects.inputs_sha256(REPO)
    changed = sorted(k for k in set(recorded) | set(now) if recorded.get(k) != now.get(k))
    assert not changed, ("the inputs of objects/DEMO.json changed since it was written: " + ", ".join(changed)
                         + "; recompute it with `python reproduce/tools/demo_objects.py` (notebook environment, ~2 min) "
                         "and commit the result")
    assert set(now) == {k if not k.endswith(".ipynb") else k + "#code" for k in
                        (i.replace("<CURRENT>", demo["source"]["snapshot_id"]) for i in demo_objects.INPUTS)}


def test_every_listed_object_is_a_manifest_row_present_in_the_tree(demo, manifest):
    for e in demo["objects"]:
        row = manifest[e["object_id"]]                                    # KeyError = not a committed object
        assert (row["paper_label"], row["model_id"], row["dir"], row["object_kind"], row["derivation"]) == \
            (e["paper_label"], e["model_id"], e["dir"], e["object_kind"], e["derivation"])
        assert (REPO / "objects" / e["dir"] / "object.pt").is_file()
        for c in e["candidates"]:
            assert manifest[c["object_id"]]["paper_label"] == e["paper_label"]


def test_seed_and_dose_read_back_from_the_sidecars_and_the_address_resolves(demo):
    idx = load_index(REPO)
    for e in demo["objects"]:
        ent = idx.get(e["object_id"])
        assert ent.present and ent.seed_read
        if e["paper_label"] in FITTED:
            assert isinstance(e["seed"], int) and ent.seed == e["seed"] and e["dose"] == 1.0 and ent.dose == 1.0
            assert ent.effective_batch == 64 == e["effective_batch"]      # the batch-64 replicates the figures average
            assert resolve(e["model_id"], e["paper_label"], seed=e["seed"], repo_root=REPO) == e["object_id"]
        else:
            assert e["seed"] is None and ent.seed is None
            assert ent.dose == e["dose"] == demo["metric"]["baseline_coefficient"]
            assert resolve(e["model_id"], e["paper_label"], dose=e["dose"], repo_root=REPO) == e["object_id"]


def test_the_one_seed_is_the_difference_loras_argmax_and_every_fitted_entry_carries_it(demo):
    seed = demo["training_seed"]
    by_label = {e["paper_label"]: e for e in demo["objects"]}
    # the recorded gains are unrounded, so difference_lora's argmax (ties to the lower seed) is exact from the file alone
    anchor = by_label[demo["seed_chosen_by"]]["candidates"]
    best = min(anchor, key=lambda c: (-c["multi_turn_gain_pp"], c["seed"]))
    assert best["seed"] == seed and best["chosen"]
    assert all(c["multi_turn_gain_pp"] < best["multi_turn_gain_pp"] or c is best or c["seed"] > seed for c in anchor)
    for e in demo["objects"]:
        cands = e["candidates"]
        chosen = [c for c in cands if c["chosen"]]
        assert len(chosen) == 1 and chosen[0]["object_id"] == e["object_id"] and chosen[0]["seed"] == e["seed"]
        assert chosen[0]["multi_turn_gain_pp"] == e["multi_turn_gain_pp"]
        if e["paper_label"] in FITTED:
            assert e["seed"] == seed, f"{e['paper_label']}: seed {e['seed']} is not the one training seed {seed}"
            assert sorted(c["seed"] for c in cands) == [1, 2, 3]            # the three batch-64 replicates, all kept
            assert len({c["object_id"] for c in cands}) == 3
        else:
            assert e["seed"] is None and len(cands) == 1 and cands[0]["seed"] is None
    # the rule is not "best per intervention": at this snapshot two interventions' own best seeds differ from the one seed
    own_best = {lab: min(by_label[lab]["candidates"], key=lambda c: (-c["multi_turn_gain_pp"], c["seed"]))["seed"]
                for lab in FITTED}
    assert own_best == {"control_relu": 1, "control_lora": 2, "constitution_relu": 2, "constitution_lora": 1,
                        "difference_relu": 3, "difference_lora": 3}


def test_the_differences_parents_are_the_chosen_control_and_constitution(demo):
    """Why one seed: with every fitted object at the same seed, the four control and constitution objects are exactly the
    two differences' parents. The file records it (`parents`) and the sidecars on disk agree."""
    by_label = {e["paper_label"]: e for e in demo["objects"]}
    for diff, (pos, neg) in PARENTS.items():
        e = by_label[diff]
        assert e["parents"] == {pos: by_label[pos]["object_id"], neg: by_label[neg]["object_id"]}
        dm = json.loads((REPO / "objects" / e["dir"] / "derived_meta.json").read_text(encoding="utf-8"))
        assert (dm["parent_pos"]["object_sha256"], dm["parent_neg"]["object_sha256"]) == (e["parents"][pos], e["parents"][neg])
        assert dm["model_key"] == dm["parent_pos"]["model_key"] == dm["parent_neg"]["model_key"] == demo["model_key"]
    assert all("parents" not in e for e in demo["objects"] if e["paper_label"] not in PARENTS)


def _cands(gains_by_label: dict[str, dict[int | None, float]]) -> dict[str, list[dict]]:
    """Synthetic candidates: label -> {seed: multi_turn_gain_pp}, as the tool's first pass builds them."""
    return {lab: [dict(seed=s, multi_turn_gain_pp=g, object_id=f"{lab}-s{s}") for s, g in sorted(gains.items(), key=str)]
            for lab, gains in gains_by_label.items()}


def _full(diff_lora: dict[int, float], **others: dict[int, float]) -> dict[str, dict]:
    base = {lab: {1: 1.0, 2: 2.0, 3: 3.0} for lab in FITTED}
    base["difference_lora"] = diff_lora
    base.update(others)
    base[BASELINE] = {None: 7.5}
    return base


def test_choose_takes_the_difference_loras_seed_for_every_fitted_intervention():
    # control_relu's own best is seed 1 and constitution_lora's seed 2; difference_lora's best (seed 3) wins for all
    cands = _cands(_full({1: 24.0, 2: 25.0, 3: 41.0}, control_relu={1: 5.1, 2: 2.8, 3: 3.4},
                         constitution_lora={1: 15.0, 2: 15.8, 3: 12.5}))
    seed, chosen = demo_objects.choose(cands)
    assert seed == 3 and demo_objects.shared_seed(cands) == 3
    assert {lab: c["seed"] for lab, c in chosen.items()} == {BASELINE: None, **{lab: 3 for lab in FITTED}}
    assert chosen["control_relu"]["object_id"] == "control_relu-s3" and chosen[BASELINE]["object_id"] == f"{BASELINE}-sNone"
    for lab, cs in cands.items():
        assert [c["chosen"] for c in cs].count(True) == 1 and next(c for c in cs if c["chosen"]) is chosen[lab]
    # the seed follows difference_lora alone: move its best and every fitted choice moves with it
    seed, chosen = demo_objects.choose(_cands(_full({1: 44.0, 2: 25.0, 3: 41.0})))
    assert seed == 1 and all(chosen[lab]["seed"] == 1 for lab in FITTED)


def test_choose_refuses_by_name_a_fitted_intervention_with_no_fit_at_the_one_seed():
    cands = _cands(_full({1: 24.0, 2: 25.0, 3: 41.0}, constitution_relu={1: 13.0, 2: 18.3}))
    with pytest.raises(SystemExit) as ei:
        demo_objects.choose(cands)
    msg = str(ei.value)
    assert msg.startswith("REFUSED") and "constitution_relu" in msg and "seed 3" in msg and "difference_lora" in msg
    assert "s1, s2" in msg                                                   # the seeds it does carry
    with pytest.raises(SystemExit, match="no difference_lora candidate"):
        demo_objects.choose(_cands({lab: {1: 1.0} for lab in FITTED - {"difference_lora"}}))


def test_choose_breaks_a_tie_to_the_lower_seed():
    seed, _ = demo_objects.choose(_cands(_full({1: 30.0, 2: 41.0, 3: 41.0})))
    assert seed == 2
    seed, _ = demo_objects.choose(_cands(_full({1: 41.0, 2: 41.0, 3: 41.0})))
    assert seed == 1
    # unrounded: a difference in the last place decides, whatever the display rounding would say
    seed, _ = demo_objects.choose(_cands(_full({1: 41.0, 2: 41.0 + 1e-12, 3: 41.0})))
    assert seed == 2


def test_each_gain_is_the_mean_of_the_two_attacks_deltas(demo):
    for e in demo["objects"]:
        for c in e["candidates"]:
            assert list(c["per_eval"]) == EVALS
            gains = [c["per_eval"][ev]["gain_pp"] for ev in EVALS]
            assert abs(sum(gains) / len(gains) - c["multi_turn_gain_pp"]) < 1e-6         # full precision: float noise only
            assert c["lo"] <= c["multi_turn_gain_pp"] <= c["hi"]
            for ev in EVALS:
                p = c["per_eval"][ev]
                assert p["n"] > 0 and p["draws"] >= 1 and p["anchor_runs"] >= 1
                assert abs((p["anchor_asr"] - p["asr"]) - p["gain_pp"]) < 1e-6         # defense gain = anchor ASR − steered ASR
                if "k" in p:
                    assert 0 <= p["k"] <= p["n"] and abs(100.0 * p["k"] / p["n"] - p["asr"]) < 1e-6


def test_sparse_paths_name_directories_that_exist(demo):
    figs = sparse_paths.paths("figures")
    assert figs == ["paper/snapshot", "paper/notebooks", "src", "reproduce", "corpus_generator", "LICENSES"]
    dem = sparse_paths.paths("demo")
    assert dem == [f"objects/{e['dir']}" for e in demo["objects"]] + ["src", "reproduce/lib", "corpus_generator",
                                                                       "system_instructions", "LICENSES"]
    for p in figs + dem:
        assert (REPO / p).is_dir(), p
    # the two the stage script and the snapshot reader open, by name
    assert (REPO / "paper" / "snapshot" / "CURRENT").is_file() and (REPO / "reproduce" / "10_figures.sh").is_file()


def test_sparse_paths_cli_runs_on_a_bare_interpreter_and_falls_back_to_git_show(tmp_path, demo):
    # stdlib only: an interpreter with no site packages (-I also drops the environment and the script's directory from sys.path)
    r = subprocess.run([sys.executable, "-I", str(TOOLS / "sparse_paths.py"), "demo"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == sparse_paths.paths("demo")
    # a sparse clone after `git sparse-checkout add reproduce/tools` has objects/DEMO.json only in HEAD, not in the tree
    clone = tmp_path / "clone"
    (clone / "objects").mkdir(parents=True)
    (clone / "objects" / "DEMO.json").write_text(DEMO.read_text(encoding="utf-8"), encoding="utf-8")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid")
    for cmd in (["git", "init", "-q"], ["git", "add", "objects/DEMO.json"], ["git", "commit", "-q", "-m", "demo list"]):
        subprocess.run(cmd, cwd=clone, check=True, env=env, capture_output=True)
    (clone / "objects" / "DEMO.json").unlink()
    (clone / "objects").rmdir()
    r = subprocess.run([sys.executable, "-I", str(TOOLS / "sparse_paths.py"), "demo", "--root", str(clone)],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == sparse_paths.paths("demo")
    r = subprocess.run([sys.executable, "-I", str(TOOLS / "sparse_paths.py"), "demo", "--root", str(tmp_path / "empty")],
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode != 0 and "REFUSED" in r.stderr


def test_readme_spells_the_three_checkouts_and_reads_the_demo_list(demo):
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "git clone https://github.com/safety-research/constitutional-adapters.git" in readme
    assert readme.count("git clone --filter=blob:none --sparse https://github.com/safety-research/constitutional-adapters.git") == 2
    assert readme.count("git sparse-checkout add reproduce/tools") == 2
    # the path helper runs through uv, so the sparse blocks need no Python besides the one uv supplies for `uv sync`
    assert "git sparse-checkout add $(uv run --no-project python reproduce/tools/sparse_paths.py figures)" in readme
    assert "git sparse-checkout add $(uv run --no-project python reproduce/tools/sparse_paths.py demo)" in readme
    assert "python3 " not in readme
    for m in re.finditer(r"^uv sync .*$", readme, flags=re.M):
        assert "--frozen" in m.group(0), m.group(0)
    # the adapter snippet loops over the demo list, so it runs in the demo checkout
    assert 'json.loads(Path("objects/DEMO.json").read_text())["objects"]' in readme
    assert any(e["paper_label"] == "difference_lora" for e in demo["objects"])
    assert "DEMO.json" in (REPO / "objects" / "README.md").read_text(encoding="utf-8")


@pytest.mark.skipif(os.environ.get("SISTEER_SLOW_TESTS") != "1",
                    reason="recomputes the deltas through the notebook's load chain (~2 min, 2 GB cache) — SISTEER_SLOW_TESTS=1 opts in")
def test_check_recomputes_the_committed_list():
    env = dict(os.environ, PYTHONPATH=str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, str(TOOLS / "demo_objects.py"), "--check"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "is current" in r.stdout
