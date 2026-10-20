"""The sdfbase-mt wave: the three
adaptive multi-turn attacks + the Petri audit on the full-corpus cells, on
Slurm, on the vLLM-served lens target.

Pins, in the order the wave needs them:
  1. the OBJECT-ONLY SI vocabulary — the two corpus pseudo-instructions are
     expressible as `si_<si>__<method>` harness ids and NOTHING else
  2. the harness <-> train/test translation round-trips for every wave cell
  3. the CE arms' object kinds are the real artifact classes' kinds
  4. `jailbreak_run._valid_defense_ids` accepts an explicit_cells config
  5. `traintest_cell_source` builds an explicit_cells registry (the Petri
     adapter path) over a staged bank
  6. the two `_mt1` transfer configs are the `_r3` configs plus exactly four
     keys, regenerate byte-identically from their generator + pins
  7. the Petri manifest loads, names exactly the config's cells, pins the
     config's target, the v1 bundle, and carries no prompted baseline.
QWEN ONLY: gemma4e4b_base is not part of this wave.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri import petri_stats  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.eval.defense_cell import (EXPLICIT_CELLS_KEY,  # noqa: E402
                                       arm_of, sha256_file)
from sisteer.methods import object_artifact_name  # noqa: E402

#: QWEN ONLY (since 2026-09-04): gemma4e4b_base is dropped from this wave
SLUGS = ("qwen35_4b",)
WAVE_CELLS = ["none",
              "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
              "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"]
HARNESS_CELLS = ["none",
                 "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce",
                 "si_constv2ctl2_full__psr_ce_all", "si_constv2ctl2_full__lora_r4_ce"]
V1_BUNDLE = "8812ba083bb5473c"


def _mt(slug):
    return yaml.safe_load((REPO / "configs" / f"sdfbase_transfer_{slug}_mt1.yaml").read_text())


def _r3(slug):
    return yaml.safe_load((REPO / "configs" / f"sdfbase_transfer_{slug}_r3.yaml").read_text())


# --- 1/2/3: vocabulary ---------------------------------------------------------

def test_object_only_sis_are_class_a_only():
    assert set(cm.PETRI_OBJECT_ONLY_SI_IDS) == {"constv2_mix_full", "constv2ctl2_full"}
    assert set(cm.PETRI_SI_IDS) == set(cm.PETRI_PROMPTABLE_SI_IDS) | set(cm.PETRI_OBJECT_ONLY_SI_IDS)
    assert set(cm.PETRI_SI_VARIANT_DIRS) == set(cm.PETRI_PROMPTABLE_SI_IDS)
    for hid in HARNESS_CELLS[1:]:
        p = dc.parse_cell_id(hid)
        assert p.cell_class == "class_a" and p.method in ("psr_ce_all", "lora_r4_ce")
        assert p.si_id in cm.PETRI_OBJECT_ONLY_SI_IDS
    for si in cm.PETRI_OBJECT_ONLY_SI_IDS:
        with pytest.raises(dc.DefenseCellError, match="OBJECT-ONLY"):
            dc.parse_cell_id(f"prompted__{si}")
        with pytest.raises(dc.DefenseCellError, match="OBJECT-ONLY"):
            dc.parse_cell_id(f"stacked__{si}__psr_ce_all")
    # the canonical vocabulary never lists a prompted cell over them
    assert not any(c.startswith("prompted__constv2")
                   for c in petri_stats.canonical_cell_ids())
    # the promptable vocabulary: the historical three plus the FULL constitution
    # (sdfbase-llama8b, 2026-09-10 — additive; the hash reads spec content, never the tuple)
    assert cm.PETRI_PROMPTABLE_SI_IDS == ("safety", "constitution", "irrelevant", "constitution_full")


def test_translation_round_trips_for_every_wave_cell():
    for tt_id, hid in zip(WAVE_CELLS, HARNESS_CELLS):
        assert ra.traintest_cell_id(hid) == tt_id
        assert ra.harness_cell_id(tt_id) == hid


def test_ce_arm_object_kinds_are_the_real_classes():
    from sisteer.extraction.multilayer import MultiLayerLoRAObject
    from sisteer.extraction.psr import PSRObject

    assert ra.ARM_OBJECT_KINDS["psr_ce_all"] == PSRObject.kind == "psr"
    assert ra.ARM_OBJECT_KINDS["lora_r4_ce"] == MultiLayerLoRAObject.kind == "lora_all"
    assert "psr_ce_all" in dc.CLASS_A_METHODS and "lora_r4_ce" in dc.CLASS_A_METHODS


def test_prompted_prefix_hashes_may_be_empty_only_without_prompted_cells():
    spec = cm.load_petri_spec(REPO / "configs/capeval/manifests/petri_sdfbase_qwen35_4b.yaml")
    assert spec.prompted_prefix_hashes == {}
    import dataclasses
    bad = dataclasses.replace(spec, cells=list(spec.cells) + ["prompted__safety"])
    with pytest.raises(cm.ManifestError, match="non-empty mapping when"):
        cm.validate_petri_spec(bad)


# --- 4: the driver's --defense surface ----------------------------------------

def test_valid_defense_ids_accepts_explicit_cells(monkeypatch):
    import jailbreak_run as jr
    from sisteer.eval import defense_cell as tt

    fake = [tt.DefenseCell(c, "baseline" if c == "none" else "A",
                           arm_of(c), None if c == "none" else c.split("__")[0],
                           None, ()) for c in WAVE_CELLS]
    seen = {}

    def _registry(cfg, *, repo_root=None):
        seen["cfg"] = cfg
        return fake

    monkeypatch.setattr(tt, "registry_from_cfg", _registry)
    cfg = {EXPLICIT_CELLS_KEY: list(WAVE_CELLS), "defenses_dir": "out/x/cells"}
    ids = jr._valid_defense_ids(cfg)
    assert set(WAVE_CELLS) <= ids
    assert seen["cfg"] is cfg


# --- 5: the Petri adapter over an explicit_cells config ------------------------

def _stage(bank: Path, cell_id: str) -> None:
    """A minimal LOADABLE torch blob of the arm's real kind (the adapter
    torch.loads it and checks `kind` against ARM_OBJECT_KINDS) + the budget
    sidecar the explicit registry requires."""
    import torch

    arm = arm_of(cell_id)
    kind = ra.ARM_OBJECT_KINDS[arm]
    p = bank / cell_id / object_artifact_name(arm)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "positions": "all", "layers": [0],
                "w": torch.zeros(4)}, p)
    (p.parent / "train_budget_meta.json").write_text(json.dumps(
        {"cell_id": cell_id, "completion_tokens": 512,
         "object_sha256": sha256_file(p)}))


def test_traintest_cell_source_builds_the_explicit_registry(tmp_path):
    bank = tmp_path / "cells"
    for c in WAVE_CELLS[1:]:
        _stage(bank, c)
    cfg = _mt("qwen35_4b")
    cfg["defenses_dir"] = str(bank)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    spec = cm.load_petri_spec(REPO / "configs/capeval/manifests/petri_sdfbase_qwen35_4b.yaml")
    src = ra.traintest_cell_source(spec, repo_root=REPO, config_path=cfg_path,
                                   defenses_dir=bank)
    assert set(src.cells) == set(WAVE_CELLS)
    # every manifest cell resolves to a registry entry through the adapter
    for hid in HARNESS_CELLS[1:]:
        entry = src.get(hid)
        assert entry is not None and entry.kind in ("psr", "lora_all"), hid
    assert src.get("none") is not None
    # a config whose explicit cells omit a manifest cell refuses as drift
    cfg2 = dict(cfg)
    cfg2[EXPLICIT_CELLS_KEY] = [c for c in WAVE_CELLS if "ctl2" not in c]
    (tmp_path / "cfg2.yaml").write_text(yaml.safe_dump(cfg2))
    with pytest.raises(ra.RegistryAdapterError, match="vocabulary drift"):
        ra.traintest_cell_source(spec, repo_root=REPO,
                                 config_path=tmp_path / "cfg2.yaml",
                                 defenses_dir=bank)


# --- 6: the configs ------------------------------------------------------------

@pytest.mark.parametrize("slug", SLUGS)
def test_mt1_config_is_r3_plus_exactly_four_keys(slug):
    r3, mt = _r3(slug), _mt(slug)
    assert mt[EXPLICIT_CELLS_KEY] == WAVE_CELLS
    assert mt["defenses_dir"] == f"out/transfer_{slug}_mt1/cells"
    assert mt["out_dir"] == f"out/sdfbase_transfer_{slug}_mt1"
    assert mt["private_run"] == f"sdfbase_transfer_{slug}_mt1"
    for k in ("si_ids", "class_a_arms", "class_b_arms", "stacked_cells"):
        assert k not in mt
    changed = {k for k in set(r3) | set(mt) if r3.get(k) != mt.get(k)}
    assert changed == {EXPLICIT_CELLS_KEY, "out_dir", "private_run", "defenses_dir"}
    # the attacker + PAIR budget the drivers assert
    assert mt["attacker"]["model"] == "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
    from sisteer.eval.attacks.pair_tap import PAIR_N_STREAMS
    assert PAIR_N_STREAMS == 5


def test_mt1_configs_regenerate_byte_identically():
    rc = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_mt_wave_configs.py"),
                         "--check"], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr


def test_mt1_pins_name_every_fitted_cell_with_a_fingerprint():
    pins = json.loads((REPO / "configs" / "sdfbase_mt_pins.json").read_text())
    for slug in SLUGS:
        for c in WAVE_CELLS[1:]:
            fp = pins[slug][c]["fit_fingerprint"]
            assert isinstance(fp, str) and len(fp) == 64, (slug, c)
            text = (REPO / "configs" / f"sdfbase_transfer_{slug}_mt1.yaml").read_text()
            assert fp in text


# --- 7: the Petri manifests ----------------------------------------------------

@pytest.mark.parametrize("slug", SLUGS)
def test_petri_manifest_names_the_configs_cells_and_target(slug):
    spec = cm.load_petri_spec(REPO / f"configs/capeval/manifests/petri_sdfbase_{slug}.yaml")
    cfg = _mt(slug)
    assert [ra.traintest_cell_id(c) for c in spec.cells] == cfg[EXPLICIT_CELLS_KEY]
    assert (spec.target_model, spec.target_revision) == (cfg["model"], cfg["model_revision"])
    assert spec.scenario_bundle_hash == V1_BUNDLE
    lock = yaml.safe_load((REPO / "configs/capeval/petri-scenarios.lock.yaml").read_text())
    assert lock["bundle"]["content_hash"] == V1_BUNDLE
    assert spec.prompted_prefix_hashes == {}
    assert cm.petri_expected_behavior_keys(spec) == sorted(spec.behaviors)
    assert spec.serving_impl_commit != cm.PETRI_UNFROZEN
    # the same audit WORLD as the completed gemma-4-E4B-it two-SI audit (bundle,
    # behaviors, auditor/judge, prompt bytes, bloom options) on the qwen target
    e4b = cm.load_petri_spec(REPO / "configs/capeval/manifests/petri_gemma4e4b.yaml")
    assert spec.behaviors == e4b.behaviors
    assert (spec.auditor_model, spec.judge_model, spec.prompt_provenance) == \
        (e4b.auditor_model, e4b.judge_model, e4b.prompt_provenance)
    assert spec.bloom_options == e4b.bloom_options
    assert spec.chat_template_hash == "a4aee8afcf2e0711"
    assert spec.tool_parser == "qwen3_xml"


# --- the registry writer: immutable snapshots that a re-export can still refresh ---

def test_write_immutable_is_idempotent_atomic_and_keeps_0440(tmp_path):
    import os
    import stat

    from sisteer.capeval.serving import lens_steer as ls

    p = tmp_path / "reg" / "cell" / "artifact.bin"
    assert ls.write_immutable(p, b"one") is True
    assert p.read_bytes() == b"one"
    assert stat.S_IMODE(os.stat(p).st_mode) == (stat.S_IRUSR | stat.S_IRGRP)
    # same bytes: nothing written (proven idempotent, not merely tolerated)
    assert ls.write_immutable(p, b"one") is False
    assert stat.S_IMODE(os.stat(p).st_mode) == (stat.S_IRUSR | stat.S_IRGRP)
    # different bytes: the directory entry is replaced despite the 0440 file
    assert ls.write_immutable(p, b"two") is True
    assert p.read_bytes() == b"two"
    assert stat.S_IMODE(os.stat(p).st_mode) == (stat.S_IRUSR | stat.S_IRGRP)
    assert not (p.parent / "artifact.bin.tmp").exists()


# --- the reduction-mode boundary (the STACKED reductions landing, 2026-09-05) -----------------

def test_standup_pins_the_reduction_mode_and_defaults_to_per_row():
    """mt1 is a per_row wave. The standup must pin SISTEER_LENS_STACKED_REDUCTIONS before the lens
    loads, default to per_row, put it in the launch attestation, and record it — so a later reader
    can tell which side of the stacked-reductions boundary any shard came from. These are greedy
    generations compared BETWEEN cells: pooling per_row and stacked shards inside one attack family
    would put an implementation difference inside the contrast."""
    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert 'REDUCTION_ENV_VAR = "SISTEER_LENS_STACKED_REDUCTIONS"' in src
    assert '"--reduction-mode", choices=["per_row", "stacked"], default="per_row"' in src
    # pinned into the process env BEFORE the server launch, and into the child's env
    assert "os.environ.update(reduction_env)" in src
    assert "extra_env=reduction_env" in src
    # into the attestation the proxy records
    assert "env={**dict(launch.env), **reduction_env}" in src
    # recorded in both artifacts a reader would consult
    assert src.count('"reduction_mode": args.reduction_mode') >= 2
    # and the SERVED mode is read back from the server's own attestation record — never from the
    # env the standup set itself — recorded next to the pin, and refused when it disagrees
    assert "served_reduction_mode(attestation)" in src
    assert "att_path = out / vs.STEERING_ATTESTATION_FILE" in src
    assert "refusing to serve the other side of the " in src and "stacked-reductions boundary" in src
    assert "LENS_REDUCTION_FIELD: lens_mode," in src            # endpoint.json
    assert 'cs[LENS_REDUCTION_FIELD] = lens_mode' in src          # cells_served.json
    assert "dict(launch.env).get(REDUCTION_ENV_VAR)" not in src   # the old self-attestation is gone


def _standup_module():
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import mt_serving_up  # noqa: E402  (imports its own deps off REPO/scripts)
    return mt_serving_up


def test_served_reduction_mode_reads_the_workers_field_or_records_absent():
    """`lens_reduction_mode` is the served-side field (the workers' install reports) that says
    which side of the stacked-reductions boundary a shard came from. Pre-landing workers name no
    mode: that is recorded as ABSENT, never rewritten as per_row by the standup."""
    m = _standup_module()
    assert m.LENS_REDUCTION_FIELD == "lens_reduction_mode"
    # the live mt1 attestation shape: worker reports without the field
    pre = {"worker_reports": [{"cells": ["a"], "layers": [1], "registration_count": 1}],
           "n_workers": 1}
    assert m.served_reduction_mode(pre) == (None, "absent")
    # post-landing: every worker names the mode
    post = {"worker_reports": [{"cells": ["a"], "lens_reduction_mode": "per_row"},
                               {"cells": ["a"], "lens_reduction_mode": "per_row"}]}
    assert m.served_reduction_mode(post) == ("per_row", "worker_reports")
    stacked = {"worker_reports": [{"lens_reduction_mode": "stacked"}]}
    assert m.served_reduction_mode(stacked) == ("stacked", "worker_reports")
    # a top-level stamp counts when no worker carries one
    assert m.served_reduction_mode({"lens_reduction_mode": "per_row", "worker_reports": [{}]}) \
        == ("per_row", "attestation_top")


@pytest.mark.parametrize("att,why", [
    ({"worker_reports": [{"lens_reduction_mode": "per_row"}, {"lens_reduction_mode": "stacked"}]},
     "disagree"),
    ({"worker_reports": [{"lens_reduction_mode": "per_row"}, {"cells": []}]}, "only 1/2"),
    ({"worker_reports": [{"lens_reduction_mode": "vectorized"}]}, "not one of"),
    ([], "not a mapping"),
])
def test_served_reduction_mode_refuses_inconsistent_attestations(att, why):
    """Workers that did not all install the same reduction (or name one outside the vocabulary)
    must not serve a wave pinned to one — a ValueError the standup turns into a refusal."""
    m = _standup_module()
    with pytest.raises(ValueError, match=why):
        m.served_reduction_mode(att)


@pytest.mark.parametrize("script,default_pos", [("attack_wave.sbatch", 'REDUCTION_MODE="${10:-per_row}"'),
                                                ("petri_proxy.sbatch", 'REDUCTION_MODE="${4:-per_row}"')])
def test_sbatch_scripts_export_and_pass_the_reduction_mode(script, default_pos):
    src = (REPO / "scripts/sdfbase/mt" / script).read_text()
    assert default_pos in src, script
    assert "export SISTEER_LENS_STACKED_REDUCTIONS=0" in src and \
        "export SISTEER_LENS_STACKED_REDUCTIONS=1" in src, script
    assert '--reduction-mode "$REDUCTION_MODE"' in src, script
    # an unknown mode refuses rather than serving something unrecorded
    assert 'FATAL: bad reduction mode' in src, script
