"""Petri Bloom Phase 2: frozen-scenario management
over SYNTHETIC scenario files — no `bloom scenarios` run, no inspect_ai /
petri_bloom imports.

Covers: freeze + content hash + no-regeneration on rerun; hash-mismatch
refusals (verify, materialize, lock); permission-only drift refusals
(bundle AND materialization — a chmod that changes no hash still voids
the 0700/read-only containment); read-only materialization of the
canonical + PROMPTED_* variant dirs from the ONE bundle with the
variant/canonical scenario-id bijection, verified over the FULL frozen
tree (understanding.md included, extra/missing files refused); the
durable-bundle-root policy (the cluster's shared storage, never a
MooseFS bulk mount — enforced at freeze_lock, with an explicit
test-only seam); the hashes-only seed-config lock incl. the M14
concurrency-cap record; and the committed BEHAVIOR.md definitions
(frontmatter shape, modality, and the EXACT SI-prefix bytes
in the prompted variants).
"""

import os
import stat
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import scenarios as psc
from sisteer.capeval.tiers import petri_runner as pr

REPO = Path(__file__).parents[1]
REPO_CONFIG = REPO / "configs" / "capeval" / "petri"
BEHAVIORS = ("self_preservation", "defer_to_users")
SIDS = ("scenario_alpha", "scenario_beta")


# --- synthetic builders ---------------------------------------------------------


def _make_source(root: Path, *, behaviors=BEHAVIORS, salt="") -> Path:
    src = root / f"generated{salt}"
    for b in behaviors:
        seeds = src / b / "scenarios" / "seeds"
        dims = src / b / "scenarios" / "dimensions"
        seeds.mkdir(parents=True, exist_ok=True)
        dims.mkdir(parents=True, exist_ok=True)
        for sid in SIDS:
            (seeds / f"{sid}.md").write_text(
                f"synthetic seed text {salt} for {b}/{sid}\n")
        (dims / f"{b}_propensity.md").write_text(
            f"synthetic rubric {salt} for {b}\n")
        (src / b / "scenarios" / "understanding.md").write_text(
            f"synthetic understanding {salt} for {b}\n")
    return src


def _make_config_root(root: Path, *, behaviors=BEHAVIORS) -> Path:
    for b in behaviors:
        (root / b).mkdir(parents=True, exist_ok=True)
        (root / b / "BEHAVIOR.md").write_text(
            f"---\nnum_scenarios: 2\nmodality: conversation\n---\n\n"
            f"synthetic behavior {b}\n")
        for v in psc.VARIANT_DIRS:
            (root / b / v).mkdir(exist_ok=True)
            (root / b / v / "BEHAVIOR.md").write_text(
                f"---\nnum_scenarios: 2\nmodality: conversation\n"
                f"target_sysprompt_prefix: |-\n  synthetic {v} prefix\n"
                f"---\n\nsynthetic behavior {b}\n")
    return root


def _write_lock(path: Path, *, behaviors=BEHAVIORS, cap=4,
                status="UNFROZEN", content_hash="UNFROZEN") -> Path:
    payload = {
        "version": 1,
        "status": status,
        "scenarios_model": "anthropic/claude-sonnet-5",
        "num_scenarios_per_behavior": 2,
        "generation": {"max_connections_cap": cap,
                       "actual_max_connections": None, "ran_at": None,
                       "variations": False, "realism_filter": False,
                       "enable_rollback": False},
        "bundle": {"location": None, "content_hash": content_hash},
        "behaviors": {b: "UNFROZEN" for b in behaviors},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _freeze(tmp_path, **kw):
    src = _make_source(tmp_path, **kw)
    return psc.freeze_scenarios(src, tmp_path / "bundles",
                                behaviors=kw.get("behaviors", BEHAVIORS))


def _writable(path: Path) -> bool:
    return bool(stat.S_IMODE(os.stat(path).st_mode) & 0o222)


# --- freeze: content-addressed, read-only, idempotent ----------------------------


def test_freeze_creates_content_addressed_readonly_bundle(tmp_path):
    bundle = _freeze(tmp_path)
    assert bundle.bundle_dir.name == bundle.content_hash
    assert len(bundle.content_hash) == cm.HASH_HEX_LEN
    # top-level 0700 (M12: containment); every file write-protected
    assert stat.S_IMODE(os.stat(bundle.bundle_dir).st_mode) == 0o700
    for b in BEHAVIORS:
        for rel in bundle.file_maps[b]:
            assert not _writable(bundle.bundle_dir / b / "scenarios" / rel)
        # understanding.md travels in the bundle (and is manifest-pinned)
        assert (bundle.bundle_dir / b / "scenarios" / "understanding.md").is_file()
        # the FULL per-behavior file map is retained on the handle so
        # materializations verify the whole tree (review 2026-08-18)
        assert "scenarios/understanding.md" in bundle.full_file_maps[b]
        assert set(bundle.full_file_maps[b]) > {
            f"scenarios/{rel}" for rel in bundle.file_maps[b]}
    # verification round-trips (and carries the full map too)
    again = psc.verify_bundle(bundle.bundle_dir,
                              expected_hash=bundle.content_hash)
    assert again.behavior_hashes == bundle.behavior_hashes
    assert again.full_file_maps == bundle.full_file_maps


def test_refreeze_is_a_verified_noop_not_a_regeneration(tmp_path):
    bundle = _freeze(tmp_path)
    marker = bundle.bundle_dir / psc.BUNDLE_MANIFEST_NAME
    before = os.stat(marker).st_mtime_ns
    again = _freeze(tmp_path)  # same content -> same hash -> verify only
    assert again.content_hash == bundle.content_hash
    assert os.stat(marker).st_mtime_ns == before  # nothing rewritten


def test_content_change_changes_the_hash(tmp_path):
    a = _freeze(tmp_path)
    b = _freeze(tmp_path, salt="-v2")
    assert a.content_hash != b.content_hash


def test_bundle_tamper_refuses(tmp_path):
    bundle = _freeze(tmp_path)
    victim = next(iter(bundle.file_maps[BEHAVIORS[0]]))
    path = bundle.bundle_dir / BEHAVIORS[0] / "scenarios" / victim
    os.chmod(path, 0o600)
    path.write_text("tampered\n")
    with pytest.raises(psc.ScenarioError, match="drift"):
        psc.verify_bundle(bundle.bundle_dir)


def test_bundle_permission_only_drift_refuses(tmp_path):
    # review 2026-08-18: a chmod changes NO hash but voids the M12
    # 0700/read-only containment — verification must refuse it, for the
    # root, a directory, and a file alike.
    victim_rel = "scenarios/understanding.md"  # an un-hashed bundled file

    bundle = _freeze(tmp_path)
    os.chmod(bundle.bundle_dir, 0o755)  # root must be EXACTLY 0700
    with pytest.raises(psc.ScenarioError, match="permission drift"):
        psc.verify_bundle(bundle.bundle_dir)
    os.chmod(bundle.bundle_dir, 0o700)
    psc.verify_bundle(bundle.bundle_dir)  # restored -> verifies again

    f = bundle.bundle_dir / BEHAVIORS[0] / victim_rel
    os.chmod(f, 0o644)  # world-readable + writable, content untouched
    with pytest.raises(psc.ScenarioError, match="permission drift"):
        psc.verify_bundle(bundle.bundle_dir)
    os.chmod(f, 0o400)

    d = bundle.bundle_dir / BEHAVIORS[0] / "scenarios" / "seeds"
    os.chmod(d, 0o755)
    with pytest.raises(psc.ScenarioError, match="permission drift"):
        psc.verify_bundle(bundle.bundle_dir)
    os.chmod(d, 0o500)
    psc.verify_bundle(bundle.bundle_dir)


def test_verify_refuses_wrong_expected_hash(tmp_path):
    bundle = _freeze(tmp_path)
    with pytest.raises(psc.ScenarioError, match="pinned to a different"):
        psc.verify_bundle(bundle.bundle_dir, expected_hash="ff" * 8)


def test_symlink_in_scenario_tree_refuses(tmp_path):
    src = _make_source(tmp_path)
    seeds = src / BEHAVIORS[0] / "scenarios" / "seeds"
    (seeds / "sneaky.md").symlink_to(seeds / f"{SIDS[0]}.md")
    with pytest.raises(psc.ScenarioError, match="symlink"):
        psc.freeze_scenarios(src, tmp_path / "bundles", behaviors=BEHAVIORS)


def test_scenario_ids_are_sorted_seed_stems(tmp_path):
    bundle = _freeze(tmp_path)
    ids = psc.scenario_ids(bundle.bundle_dir / BEHAVIORS[0] / "scenarios")
    assert ids == tuple(sorted(SIDS))


# --- the durable bundle root: the cluster's shared root, never a MooseFS root ------------------


def test_default_bundle_root_is_the_cluster_shared_root(monkeypatch):
    monkeypatch.setenv("SISTEER_SHARED_ROOT", "/shared/testuser")
    root = psc.default_bundle_root()
    assert root.as_posix() == "/shared/testuser/petri/scenarios"
    assert psc.validate_bundle_root(root) == root


def test_cluster_b_moosefs_root_refuses():
    # /workspace on cluster B is MooseFS: chmod is silently ignored, so
    # a "0700" bundle there would be world-readable (M12) — refuse.
    for bad in ("/workspace/testuser/petri/scenarios", "/workspace"):
        with pytest.raises(psc.ScenarioError, match="MooseFS"):
            psc.validate_bundle_root(bad)


def test_relative_bundle_root_refuses():
    with pytest.raises(psc.ScenarioError, match="absolute"):
        psc.validate_bundle_root("bundles/petri")


# --- the seed-config lock: hashes only; frozen blocks regeneration ---------------


def test_freeze_lock_records_hashes_only(tmp_path):
    bundle = _freeze(tmp_path)
    lock_path = _write_lock(tmp_path / "lock.yaml")
    updated = psc.freeze_lock(lock_path, bundle, actual_max_connections=4,
                              ran_at="2026-08-18T00:00:00Z",
                              durable_root=tmp_path)
    assert updated["status"] == "frozen"
    assert updated["bundle"]["content_hash"] == bundle.content_hash
    assert updated["behaviors"] == bundle.behavior_hashes
    text = lock_path.read_text()
    # HASHES ONLY: no seed/scenario text ever reaches the lock (M12)
    assert "synthetic seed text" not in text
    assert "synthetic rubric" not in text
    # idempotent on the same bundle
    psc.freeze_lock(lock_path, bundle, actual_max_connections=4,
                    ran_at="2026-08-18T00:00:00Z", durable_root=tmp_path)
    # a DIFFERENT bundle refuses — never an overwrite
    other = _freeze(tmp_path, salt="-v2")
    with pytest.raises(psc.ScenarioError, match="re-freeze"):
        psc.freeze_lock(lock_path, other, actual_max_connections=4,
                        ran_at="2026-08-18T00:00:00Z", durable_root=tmp_path)


def test_freeze_lock_requires_durable_root(tmp_path):
    # a production freeze may only RECORD a bundle under the cluster's
    # shared storage — a /tmp bundle marked frozen would vanish and
    # leave the audit unreproducible. Tests inject their root through the
    # EXPLICIT durable_root seam; production leaves it None.
    bundle = _freeze(tmp_path)
    lock_path = _write_lock(tmp_path / "lock.yaml")
    with pytest.raises(psc.ScenarioError, match="durable"):
        psc.freeze_lock(lock_path, bundle, actual_max_connections=4,
                        ran_at="2026-08-18T00:00:00Z")  # no seam -> refuse
    # the lock is untouched by the refusal
    assert psc.read_lock(lock_path)["status"] == psc.LOCK_STATUS_UNFROZEN
    # a seam root that does NOT contain the bundle refuses too
    with pytest.raises(psc.ScenarioError, match="durable root"):
        psc.freeze_lock(lock_path, bundle, actual_max_connections=4,
                        ran_at="2026-08-18T00:00:00Z",
                        durable_root=tmp_path / "elsewhere")


def test_frozen_lock_blocks_regeneration(tmp_path):
    bundle = _freeze(tmp_path)
    lock_path = _write_lock(tmp_path / "lock.yaml")
    psc.assert_generation_allowed(lock_path)  # unfrozen: generation may run
    psc.freeze_lock(lock_path, bundle, actual_max_connections=2,
                    ran_at="2026-08-18T00:00:00Z", durable_root=tmp_path)
    with pytest.raises(psc.ScenarioError, match="never regenerates"):
        psc.assert_generation_allowed(lock_path)


def test_lock_cap_exceeded_requires_coordination_post(tmp_path):
    # M14: `bloom scenarios` runs capped at <=4 connections; exceeding the
    # cap requires the coordination post FIRST — recorded, never silent.
    bundle = _freeze(tmp_path)
    lock_path = _write_lock(tmp_path / "lock.yaml", cap=4)
    with pytest.raises(psc.ScenarioError, match="coordination"):
        psc.freeze_lock(lock_path, bundle, actual_max_connections=8,
                        ran_at="2026-08-18T00:00:00Z", durable_root=tmp_path)
    updated = psc.freeze_lock(lock_path, bundle, actual_max_connections=8,
                              ran_at="2026-08-18T00:00:00Z",
                              coordination_posted=True,
                              durable_root=tmp_path)
    assert updated["generation"]["coordination_posted"] is True


def test_lock_behavior_set_mismatch_refuses(tmp_path):
    bundle = _freeze(tmp_path)
    lock_path = _write_lock(tmp_path / "lock.yaml",
                            behaviors=("self_preservation",))
    with pytest.raises(psc.ScenarioError, match="behaviors"):
        psc.freeze_lock(lock_path, bundle, actual_max_connections=4,
                        ran_at="2026-08-18T00:00:00Z", durable_root=tmp_path)


def test_repo_lock_parses_with_recorded_cap_and_behaviors():
    # the committed lock: UNFROZEN until the one `bloom scenarios` pass
    # (launch phase) and frozen-with-hashes after; either way the M14
    # cap is recorded and all six behaviors listed.
    lock = psc.read_lock(REPO / "configs" / "capeval" /
                         "petri-scenarios.lock.yaml")
    assert lock["generation"]["max_connections_cap"] == 4
    assert set(lock["behaviors"]) == set(cm.PETRI_BEHAVIORS)
    if lock["status"] == psc.LOCK_STATUS_FROZEN:
        ch = lock["bundle"]["content_hash"]
        assert isinstance(ch, str) and len(ch) == cm.HASH_HEX_LEN
    else:
        assert lock["status"] == psc.LOCK_STATUS_UNFROZEN
        assert lock["bundle"]["content_hash"] == cm.PETRI_UNFROZEN


# --- read-only materialization from the ONE bundle (M12/H6) ----------------------


def _materialize(tmp_path, bundle, **kw):
    config_root = _make_config_root(tmp_path / "configs")
    return psc.materialize(
        bundle.bundle_dir, tmp_path / "work" / "behaviors",
        expected_bundle_hash=kw.pop("expected_hash", bundle.content_hash),
        config_root=config_root, behaviors=BEHAVIORS, **kw), config_root


def test_materialize_builds_canonical_and_variant_dirs_readonly(tmp_path):
    bundle = _freeze(tmp_path)
    dirs, config_root = _materialize(tmp_path, bundle)
    expected_keys = {b for b in BEHAVIORS} | {
        f"{b}/{v}" for b in BEHAVIORS for v in psc.VARIANT_DIRS}
    assert set(dirs) == expected_keys  # the harness_config_hash dir keys
    canonical_ids = {b: psc.scenario_ids(dirs[b] / "scenarios")
                     for b in BEHAVIORS}
    for key, d in dirs.items():
        b = key.split("/", 1)[0]
        # BEHAVIOR.md comes from the version-controlled config dir
        src_md = (config_root / key / "BEHAVIOR.md").read_bytes()
        assert (d / "BEHAVIOR.md").read_bytes() == src_md
        # scenarios come from the ONE bundle: identical ids in every
        # variant dir — the variant/canonical bijection holds by
        # construction (H6)
        assert psc.scenario_ids(d / "scenarios") == canonical_ids[b]
        for rel in bundle.file_maps[b]:
            assert not _writable(d / "scenarios" / rel)  # read-only (M12)
        assert not _writable(d / "BEHAVIOR.md")


def test_materialize_refuses_wrong_bundle_hash(tmp_path):
    bundle = _freeze(tmp_path)
    with pytest.raises(psc.ScenarioError, match="pinned to a different"):
        _materialize(tmp_path, bundle, expected_hash="ee" * 8)


def test_materialize_is_idempotent_and_refuses_tamper(tmp_path):
    bundle = _freeze(tmp_path)
    dirs, config_root = _materialize(tmp_path, bundle)
    again = psc.materialize(
        bundle.bundle_dir, tmp_path / "work" / "behaviors",
        expected_bundle_hash=bundle.content_hash,
        config_root=config_root, behaviors=BEHAVIORS)
    assert again == dirs
    victim = dirs[BEHAVIORS[0]] / "scenarios" / "seeds" / f"{SIDS[0]}.md"
    os.chmod(victim.parent, 0o700)
    os.chmod(victim, 0o600)
    victim.write_text("tampered\n")
    with pytest.raises(psc.ScenarioError, match="materialization"):
        psc.materialize(
            bundle.bundle_dir, tmp_path / "work" / "behaviors",
            expected_bundle_hash=bundle.content_hash,
            config_root=config_root, behaviors=BEHAVIORS)


def test_materialize_refuses_partial_dest(tmp_path):
    bundle = _freeze(tmp_path)
    dirs, config_root = _materialize(tmp_path, bundle)
    import shutil
    doomed = dirs[f"{BEHAVIORS[0]}/{psc.VARIANT_DIRS[0]}"]
    os.chmod(doomed.parent, 0o700)
    for dirpath, _dirnames, _filenames in os.walk(doomed):
        os.chmod(dirpath, 0o700)  # unlink needs writable parent dirs
    shutil.rmtree(doomed)
    with pytest.raises(psc.ScenarioError, match="missing from an existing"):
        psc.materialize(
            bundle.bundle_dir, tmp_path / "work" / "behaviors",
            expected_bundle_hash=bundle.content_hash,
            config_root=config_root, behaviors=BEHAVIORS)


def test_materialize_verifies_the_full_frozen_tree(tmp_path):
    # review 2026-08-18: an existing materialization used to verify only
    # seeds/dimensions — editing understanding.md (an un-hashed bundled
    # file) was accepted. The FULL tree is compared now: modified
    # un-hashed files and extra files both refuse.
    bundle = _freeze(tmp_path)
    dirs, config_root = _materialize(tmp_path, bundle)

    def re_materialize():
        return psc.materialize(
            bundle.bundle_dir, tmp_path / "work" / "behaviors",
            expected_bundle_hash=bundle.content_hash,
            config_root=config_root, behaviors=BEHAVIORS)

    victim = dirs[BEHAVIORS[0]] / "scenarios" / "understanding.md"
    old = victim.read_bytes()
    os.chmod(victim, 0o600)
    victim.write_text("tampered understanding\n")
    with pytest.raises(psc.ScenarioError, match="materialization"):
        re_materialize()
    victim.write_bytes(old)
    os.chmod(victim, 0o400)
    re_materialize()  # restored -> verifies again

    extra_dir = dirs[BEHAVIORS[0]] / "scenarios"
    os.chmod(extra_dir, 0o700)
    (extra_dir / "smuggled.md").write_text("extra file\n")
    os.chmod(extra_dir / "smuggled.md", 0o400)
    os.chmod(extra_dir, 0o500)
    with pytest.raises(psc.ScenarioError, match="unexpected file"):
        re_materialize()


def test_materialize_permission_only_drift_refuses(tmp_path):
    # review 2026-08-18: permission drift with intact content refuses over
    # the WHOLE materialized tree — root, dirs, BEHAVIOR.md, and the
    # un-hashed understanding.md alike.
    bundle = _freeze(tmp_path)
    dirs, config_root = _materialize(tmp_path, bundle)
    dest_root = tmp_path / "work" / "behaviors"

    def re_materialize():
        return psc.materialize(
            bundle.bundle_dir, dest_root,
            expected_bundle_hash=bundle.content_hash,
            config_root=config_root, behaviors=BEHAVIORS)

    for victim, restore in [
            (dest_root, 0o700),                            # the 0700 root
            (dirs[BEHAVIORS[0]], 0o500),                   # a behavior dir
            (dirs[BEHAVIORS[0]] / "BEHAVIOR.md", 0o400),   # the eval file
            (dirs[BEHAVIORS[0]] / "scenarios" / "understanding.md", 0o400),
    ]:
        os.chmod(victim, 0o755 if restore != 0o400 else 0o644)
        with pytest.raises(psc.ScenarioError, match="permission drift"):
            re_materialize()
        os.chmod(victim, restore)
    re_materialize()  # all restored -> verifies again


def test_behavior_files_covers_canonical_plus_variants(tmp_path):
    config_root = _make_config_root(tmp_path / "configs")
    bf = psc.behavior_files(config_root, BEHAVIORS)
    assert set(bf) == {b for b in BEHAVIORS} | {
        f"{b}/{v}" for b in BEHAVIORS for v in psc.VARIANT_DIRS}
    assert all(isinstance(v, bytes) and v for v in bf.values())
    # a missing definition is a refusal, never a silently narrower hash
    (config_root / BEHAVIORS[0] / psc.VARIANT_DIRS[0] / "BEHAVIOR.md").unlink()
    with pytest.raises(psc.ScenarioError, match="missing"):
        psc.behavior_files(config_root, BEHAVIORS)


# --- the committed behavior definitions ---------------------------------


def test_behavior_defs_shape_and_modalities():
    # six version-controlled BEHAVIOR.md files: num_scenarios 10, modality
    # self_preservation -> agent, others -> conversation, no variations,
    # no SI prefix on the canonical dirs.
    assert psc.BEHAVIOR_MODALITIES["self_preservation"] == "agent"
    assert all(psc.BEHAVIOR_MODALITIES[b] == "conversation"
               for b in cm.PETRI_BEHAVIORS if b != "self_preservation")
    for b in cm.PETRI_BEHAVIORS:
        front, body = psc.parse_behavior_md(
            (REPO_CONFIG / b / "BEHAVIOR.md").read_bytes())
        assert front["num_scenarios"] == 10
        assert front["modality"] == psc.BEHAVIOR_MODALITIES[b]
        assert "variations" not in front
        assert "target_sysprompt_prefix" not in front
        # realism_filter / enable_rollback are bloom_audit TASK options,
        # pinned OFF at the eval call — never frontmatter
        assert "realism_filter" not in front
        assert "enable_rollback" not in front
        assert len(body.split()) > 50  # a real description, not a stub
    # the §6 defaults are pinned OFF in the COMMITTED manifest's
    # bloom_options — the manifest-owned, harness-hashed source the runner
    # passes to the eval call (H10; the old module-level constant is gone)
    spec = cm.load_petri_spec(REPO / "configs" / "capeval" / "manifests" /
                              "petri_twosi.yaml")
    bo = spec.bloom_options
    assert (bo.realism_filter, bo.enable_rollback, bo.enable_prefill) == \
        (False, False, False)
    # max_turns 15 = Petri's bloom_audit default and the plan's cost
    # bound
    assert pr.bloom_task_options(spec) == {
        "max_turns": 15, "enable_prefill": False, "enable_rollback": False,
        "compaction": False, "realism_filter": False}


def test_prompted_variants_carry_the_exact_si_bytes():
    # H8: the variant dirs' target_sysprompt_prefix is the EXACT SI byte
    # string — its sha256_hex16 is what the manifest records and the
    # proxy verifies per request. SAFETY_SI is the repo's canonical
    # 66-word instruction; CONSTITUTION_SI is the verified verbatim CC0
    # constitution assembly (two-SI spec: 4,896 words / 30,952 chars);
    # IRRELEVANT_SI (2026-08-28) is the stage-3 length-matched control
    # text (30,407 chars).
    from sisteer.data.instructions import SAFETY_SI

    assert len(SAFETY_SI.text.split()) == 66
    per_variant_texts: dict[str, set] = {v: set() for v in psc.VARIANT_DIRS}
    for b in cm.PETRI_BEHAVIORS:
        cfront, cbody = psc.parse_behavior_md(
            (REPO_CONFIG / b / "BEHAVIOR.md").read_bytes())
        for v in psc.VARIANT_DIRS:
            front, body = psc.parse_behavior_md(
                (REPO_CONFIG / b / v / "BEHAVIOR.md").read_bytes())
            assert front["num_scenarios"] == 10
            assert front["modality"] == cfront["modality"]
            assert "variations" not in front
            assert body == cbody  # only the prefix differs from canonical
            prefix = front["target_sysprompt_prefix"]
            assert isinstance(prefix, str) and prefix
            per_variant_texts[v].add(prefix)
    # one identical text per variant across all six behavior dirs,
    # matching the verified byte-exact assemblies
    expected = {
        "PROMPTED_SAFETY": (len(SAFETY_SI.text), "ac3a58f61d53cf24"),
        "PROMPTED_CONSTITUTION": (30952, "21e06f482a6559b3"),
        "PROMPTED_IRRELEVANT": (30407, "71ff67aa2bc000f1"),
        # the FULL constitution (sdfbase-llama8b, 2026-09-10): data/instructions.CONSTITUTION_FULL_PROMPT,
        # the vendored constitution_full.md byte-for-byte (its sha256 is asserted at import)
        "PROMPTED_CONSTITUTION_FULL": (182219, "82f88c5532e4939e"),
    }
    assert set(per_variant_texts) == set(expected)
    for v, (n_chars, sha) in expected.items():
        assert len(per_variant_texts[v]) == 1
        text = per_variant_texts[v].pop()
        assert len(text) == n_chars
        assert cm.sha256_hex16(text.encode("utf-8")) == sha
    assert expected["PROMPTED_SAFETY"][1] == cm.sha256_hex16(
        SAFETY_SI.text.encode("utf-8"))


def test_variant_dir_names_cover_all_sis():
    assert psc.SI_VARIANT_DIRS == {"safety": "PROMPTED_SAFETY",
                                   "constitution": "PROMPTED_CONSTITUTION",
                                   "irrelevant": "PROMPTED_IRRELEVANT",
                                   "constitution_full": "PROMPTED_CONSTITUTION_FULL"}
    # variant dirs exist for the PROMPTABLE SIs only; the object-only
    # pseudo-instructions (sdf-base-steering) have no prompted baseline
    assert set(psc.SI_VARIANT_DIRS) == set(cm.PETRI_PROMPTABLE_SI_IDS)
    assert set(cm.PETRI_SI_IDS) == (set(cm.PETRI_PROMPTABLE_SI_IDS)
                                    | set(cm.PETRI_OBJECT_ONLY_SI_IDS))
