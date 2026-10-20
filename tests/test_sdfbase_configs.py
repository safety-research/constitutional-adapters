"""sdf-base-steering campaign configs are generator-pinned.

scripts/sdfbase/make_sdfbase_configs.py is the single source of the two
committed configs/sdfbase_*_fits.yaml files. These tests pin:

- the generator writes ONLY configs/sdfbase_*.yaml (hard path refusal) and
  running it leaves every configs/two_si_*.yaml byte-identical;
- the committed files byte-match a regeneration (expect_sha stability —
  the fits' --expect-config-sha is sha256 of these bytes);
- the per-cell caps table, the budget provenance schema
  (training_budget_kind: document_tokens, completion_tokens: null), and
  the MEMPROBE / HPO sentinels' int()/float()-crash behavior;
- the tune_arm seam: ARM_AXES / seed boxes / model slugs /
  CESoloAdapter registration and its config-shape refusals.

CPU-only; no model, no network.
"""

import hashlib
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO / "scripts"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: @dataclass under PEP-563 string annotations
    # resolves types via sys.modules[cls.__module__].
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


msc = _load_module("make_sdfbase_configs",
                   REPO / "scripts/sdfbase/make_sdfbase_configs.py")

FILES = ["sdfbase_gemma4e4b_base_fits.yaml",
         "sdfbase_qwen35_4b_base_fits.yaml",
         # GLM-4.6V-Flash (2026-09-02): the `_doc` lane — the CHAT model,
         # no base variant exists; both folds pending (all sentinels).
         "sdfbase_glm46vflash_doc_fits.yaml",
         # Nemotron-3-Nano-4B on the documents lane (2026-09-02): "_doc",
         # not "_base" — no base variant exists, the instruct checkpoint is
         # the documents-lane model; both folds pending (all sentinels).
         # INSTRUCT-DIRECT lanes (2026-09-03): the two instruct
         # checkpoints the battery serves, trained on the corpus directly;
         # mix-full only at cap 16 / patience 2; folds COPIED from the
         # same architecture's base lane.
         "sdfbase_gemma4e4b_it_doc_fits.yaml",
         "sdfbase_qwen35_4b_it_doc_fits.yaml",
         # FULL-SCALE lanes of the two chat checkpoints (2026-09-06, run
         # on Slurm): mix-full + ctl2-full at cap 16 / patience 2 in SIBLING
         # configs, so the frozen `_doc` bytes (their finished cells' fit
         # identity) never move; folds COPIED from the parent lane.
         "sdfbase_nemotron3nano4b_doc_full_fits.yaml",
         "sdfbase_glm46vflash_doc_full_fits.yaml",
         # GLM gated-arm REFIT (2026-09-07): the `_full` lane's config with
         # psr_ce_all's optimizer = HPO trial r2_t12's, as a SIBLING so the
         # `_full` bytes (two fingerprinted LoRA cells) never move.
         "sdfbase_glm46vflash_doc_full_r2t12_fits.yaml",
         # Llama-3.1-8B (2026-09-09): the ungated
         # NousResearch base and instruct mirrors, five cells each (three
         # working cells + the two full-scale production cells), effective
         # batch 64 per key; both folds pending (all sentinels).
         "sdfbase_llama8b_base_fits.yaml",
         "sdfbase_llama8b_it_doc_fits.yaml",
         # INSTRUCT-DIRECT BATCH-64 siblings (2026-09-11): the two
         # `_it_doc` checkpoints re-fit at effective batch 64, constitution
         # AND control twins, in SIBLING configs so the frozen `_it_doc`
         # bytes (four published cells' fit identity) never move; pools
         # shared with the parent lane (pool_key); both folds pending.
         "sdfbase_gemma4e4b_it_doc_eb64_fits.yaml",
         "sdfbase_qwen35_4b_it_doc_eb64_fits.yaml",
         # Qwen-BASE batch-64 SEED REPLICATES (2026-09-15): the frozen
         # qwen35_4b_base checkpoint re-fit
         # from scratch at effective batch 64 under seeds 1, 2, 3, one
         # SIBLING config per seed (train_kwargs.default.seed declared; own
         # /vol/pool/<key>/); memprobe folded at the spec default 2 / 2, the
         # HPO fold pending (the seed-1 search, one winner per arm).
         "sdfbase_qwen35_4b_base_eb64_s1_fits.yaml",
         "sdfbase_qwen35_4b_base_eb64_s2_fits.yaml",
         "sdfbase_qwen35_4b_base_eb64_s3_fits.yaml",
         # paper-companion-repo (2026-09-29; spec FD 4): the front door's
         # configured-not-run Qwen3.5-4B (worked example) lane — mix-full + ctl2-full at cap
         # 16 / patience 2, effective batch 64, gradient checkpointing, a
         # `fit_selection` block per arm; its folds pending (all sentinels)
         # until stage 02's memprobe / stage 03's search fold them.
         "sdfbase_qwen35_4b_demo_base_fits.yaml"]

#: generated file -> its MODELS key (the one spelling every per-model
#: assertion below resolves through — never a substring of the file name).
KEY_OF = {"sdfbase_gemma4e4b_base_fits.yaml": "gemma4e4b_base",
          "sdfbase_qwen35_4b_base_fits.yaml": "qwen35_4b_base",
          "sdfbase_glm46vflash_doc_fits.yaml": "glm46vflash_doc",
          "sdfbase_gemma4e4b_it_doc_fits.yaml": "gemma4e4b_it_doc",
          "sdfbase_qwen35_4b_it_doc_fits.yaml": "qwen35_4b_it_doc",
          "sdfbase_nemotron3nano4b_doc_full_fits.yaml":
              "nemotron3nano4b_doc_full",
          "sdfbase_glm46vflash_doc_full_fits.yaml": "glm46vflash_doc_full",
          "sdfbase_glm46vflash_doc_full_r2t12_fits.yaml":
              "glm46vflash_doc_full_r2t12",
          "sdfbase_llama8b_base_fits.yaml": "llama8b_base",
          "sdfbase_llama8b_it_doc_fits.yaml": "llama8b_it_doc",
          "sdfbase_gemma4e4b_it_doc_eb64_fits.yaml": "gemma4e4b_it_doc_eb64",
          "sdfbase_qwen35_4b_it_doc_eb64_fits.yaml": "qwen35_4b_it_doc_eb64",
          "sdfbase_qwen35_4b_base_eb64_s1_fits.yaml": "qwen35_4b_base_eb64_s1",
          "sdfbase_qwen35_4b_base_eb64_s2_fits.yaml": "qwen35_4b_base_eb64_s2",
          "sdfbase_qwen35_4b_base_eb64_s3_fits.yaml": "qwen35_4b_base_eb64_s3",
          "sdfbase_qwen35_4b_demo_base_fits.yaml": "qwen35_4b_demo_base"}


def _key_of(name: str) -> str:
    """generated file name -> generator model key, through KEY_OF and
    checked against the generator's table so a typo fails here."""
    key = KEY_OF[name]
    assert key in msc.MODELS, (key, sorted(msc.MODELS))
    return key

#: The plan-§6 per-cell caps table + the later rows:
#: mix-1m raised 12->24 (refit of the cap-limited 1M cells —
#: the config change moves every fit fingerprint, so the respawn retrains
#: them), mix-10m {8, 2}, mix-full {4, 1}.
EXPECTED_CAPS = {
    "constv2-mix-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-mix-10k": {"epochs": 120, "early_stop_patience": 12},
    "constv2-mix-100k": {"epochs": 60, "early_stop_patience": 6},
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 4, "early_stop_patience": 1},
    "constv2-adv-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-adv-100k": {"epochs": 60, "early_stop_patience": 6},
    "constv2ctl-100k": {"epochs": 60, "early_stop_patience": 6},
    # 2026-09-03: the register-matched control at the mix-full
    # budget, capped {16, 2} — budgeted for more epochs
    # than the constitutional full cells' original {4, 1}.
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}

EXPECTED_SI_IDS = ["constv2-mix-1k", "constv2-mix-10k", "constv2-mix-100k",
                   "constv2-mix-1m", "constv2-mix-10m", "constv2-mix-full",
                   "constv2-adv-1k", "constv2-adv-100k",
                   "constv2ctl-100k", "constv2ctl2-full"]

#: Workstream 3 (2026-09-02): the two chat lanes run the MIXED
#: curve {1k,10k,100k,1M} + ctl-100k only — no advice lane, no 10m/full,
#: no transfer battery — each cell at the base configs' caps.
EXPECTED_WS3_SI_IDS = ["constv2-mix-1k", "constv2-mix-10k",
                       "constv2-mix-100k", "constv2-mix-1m",
                       "constv2ctl-100k"]
WS3_KEYS = {"glm46vflash_doc"}   # the Nemotron ladder lane (nemotron3nano4b_doc) was retired 2026-10-08; its full-scale lane is the paper's

#: 2026-09-03: the INSTRUCT-DIRECT lanes train the
#: constitutional corpus at FULL scale only, directly on the instruct
#: checkpoints, at cap 16 / patience 2 (the "more epochs" rule — the same
#: row as constv2ctl2-full; NOT the base configs' cap-4 mix-full row, which
#: keeps its bytes). Every other cell is absent from these configs.
EXPECTED_IT_SI_IDS = ["constv2-mix-full"]
EXPECTED_IT_CAPS = {
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
}
IT_KEYS = {"gemma4e4b_it_doc", "qwen35_4b_it_doc"}

#: FULL-SCALE Slurm lanes (2026-09-06): the two chat checkpoints at
#: constv2-mix-full + constv2ctl2-full, both cells cap 16 / patience 2 (the
#: it_doc / ctl2-full "more epochs" row), in sibling configs keyed `_full`.
EXPECTED_FULL_SI_IDS = ["constv2-mix-full", "constv2ctl2-full"]
EXPECTED_FULL_CAPS = {
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}
FULL_KEYS = {"nemotron3nano4b_doc_full", "glm46vflash_doc_full"}
#: FULL_KEYS plus the GLM gated-arm REFIT sibling (2026-09-07): the same
#: full-scale cell scope, caps row and parent pool dir, but NOT the parent's
#: optimizer verbatim (psr_ce_all is HPO trial r2_t12's) — so it joins the
#: scope/caps/pool enumerations below and stays OUT of the verbatim-folds and
#: verbatim-train_kwargs parametrizations over FULL_KEYS.
FULL_SIBLING_KEYS = FULL_KEYS | {"glm46vflash_doc_full_r2t12"}
#: full-scale key -> the frozen `_doc` lane it is the full-scale sibling of
#: (same checkpoint; pools + sidecars + HPO winners + memprobe widths shared)
FULL_PARENT_OF = {"nemotron3nano4b_doc_full": "nemotron3nano4b_doc",
                  "glm46vflash_doc_full": "glm46vflash_doc",
                  "glm46vflash_doc_full_r2t12": "glm46vflash_doc"}
#: it-direct key -> the base lane whose HPO winners + memprobe widths it
#: reuses verbatim (same architecture, same objective).
IT_BASE_OF = {"gemma4e4b_it_doc": "gemma4e4b_base",
              "qwen35_4b_it_doc": "qwen35_4b_base",
              # the Llama lane's instruct-direct key reuses ITS base lane's
              # winners and widths under the same rule; its own contract
              # lives in tests/test_sdfbase_llama8b_lane.py because, unlike
              # the two above, it has no probe_only sibling and no r2
              # transfer config to pin against.
              "llama8b_it_doc": "llama8b_base"}

#: The Llama-3.1-8B lane (2026-09-09): five cells per key — three WORKING
#: cells (mix-1k identity gate, mix-1m memprobe, mix-10m search reference)
#: at the shared CAPS rows, plus the two full-scale production cells at the
#: cap-16 / patience-2 row — declared through the per-MODEL override so no
#: shared table moves. Effective batch 64, per key.
EXPECTED_LLAMA8B_SI_IDS = ["constv2-mix-1k", "constv2-mix-1m",
                           "constv2-mix-10m", "constv2-mix-full",
                           "constv2ctl2-full"]
EXPECTED_LLAMA8B_CAPS = {
    "constv2-mix-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}
LLAMA8B_KEYS = {"llama8b_base", "llama8b_it_doc"}

#: The instruct-direct BATCH-64 lanes (2026-09-11): four cells per
#: key — mix-1m (memprobe rows) and mix-10m (the LR search reference) as
#: WORKING cells at the shared CAPS rows, mix-full + ctl2-full as the
#: production twins at the cap-16 / patience-2 row — via the per-MODEL
#: override; effective batch 64 per key; pools shared with the parent
#: `_it_doc` lane (pool_key), whose frozen bytes stay untouched.
EXPECTED_IT_EB64_SI_IDS = ["constv2-mix-1m", "constv2-mix-10m",
                           "constv2-mix-full", "constv2ctl2-full"]
EXPECTED_IT_EB64_CAPS = {
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}
IT_EB64_KEYS = {"gemma4e4b_it_doc_eb64", "qwen35_4b_it_doc_eb64"}
IT_EB64_PARENT_OF = {"gemma4e4b_it_doc_eb64": "gemma4e4b_it_doc",
                     "qwen35_4b_it_doc_eb64": "qwen35_4b_it_doc"}

#: Qwen-BASE batch-64 SEED REPLICATES (2026-09-15; the qwen-base-seed-
#: replicates lane): one key per seed, the qwen35_4b_base pin
#: verbatim, effective batch 64, `train_kwargs.default.seed` = the seed
#: (a NEW config key, emitted ONLY when a MODELS entry declares "seed", so
#: every other config's bytes stay put). s1 carries three WORKING cells at
#: the shared CAPS rows (mix-1k identity gate, mix-1m memprobe, mix-10m
#: search reference) plus the two production cells; s2 / s3 the production
#: cells only. Both production cells at cap 16 / patience 2 (scratch fits
#: of BOTH corpora — never the published cap-4->16 continuations). No
#: pool_key: the held-out carve differs per seed, so each key owns
#: /vol/pool/<key>/.
SEEDREP_KEYS = ("qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
                "qwen35_4b_base_eb64_s3")
SEEDREP_SEED_OF = {"qwen35_4b_base_eb64_s1": 1, "qwen35_4b_base_eb64_s2": 2,
                   "qwen35_4b_base_eb64_s3": 3}
SEEDREP_PARENT = "qwen35_4b_base"
EXPECTED_SEEDREP_SI_IDS_S1 = ["constv2-mix-1k", "constv2-mix-1m",
                              "constv2-mix-10m", "constv2-mix-full",
                              "constv2ctl2-full"]
EXPECTED_SEEDREP_SI_IDS = ["constv2-mix-full", "constv2ctl2-full"]
EXPECTED_SEEDREP_CAPS = {
    "constv2-mix-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}

#: paper-companion-repo (2026-09-29): the front door's test-model lane — the
#: FULL scope + caps row (mix-full + ctl2-full at cap 16 / patience 2),
#: the paper's qwen35_4b_base lane under its own key (the same Qwen3.5-4B-Base
#: checkpoint: 32 hybrid blocks of hidden 2,560, 8 of them full attention),
#: its own /vol/pool/<key>/, and the plan-§3.7 `fit_selection` block per arm
#: (pinned in detail by tests/test_models_demo_front_door.py). The fit slug
#: carries `_base`; the served id is qwen35_4b_demo.
PIPELINE_TEST_KEYS = ("qwen35_4b_demo_base",)

#: sha16 (= the fits' expect_sha / manifest config_sha) of the four
#: configs that FINISHED or IN-FLIGHT cells were fingerprinted under:
#: wave A / A' (gemma, qwen — deploy_stamp.json), the chat lanes (glm,
#: nemotron — deploy_stamp_ws3.json). Any byte change here moves every one
#: of those fit fingerprints, so a lane that adds a model or a cell must
#: leave these bytes alone (the instruct-direct lanes did: their caps live
#: in a per-MODEL override, never in the shared CAPS row).
FROZEN_CONFIG_SHA16 = {
    # This tree carries ctl-full (constv2ctl2-full joined si_ids + CAPS):
    # the two base configs hash to the ctl2 bytes (3e1aa42e /
    # 17d7e404 — the fit_calls_ctl2.json config_sha), not the wave-A bytes
    # (6b2c47bc / a0e16665) the constitutional continuations still run
    # under on their own old-bytes apps. The merge constraint binds
    # trees that DEPLOY those apps; the sdfbase-mt wave deploys nothing on
    # Modal and fits nothing — it only serves published objects.
    "sdfbase_gemma4e4b_base_fits.yaml": "3e1aa42e66d9e857",
    "sdfbase_qwen35_4b_base_fits.yaml": "17d7e40443d59b50",
    "sdfbase_glm46vflash_doc_fits.yaml": "7fa1e3f2f204aa1d",
    # the two FULL-SCALE Slurm lanes (2026-09-06): cells have been fit
    # under both since 2026-09-07 (GLM: two LoRA cells, one COMPLETE and one
    # in flight when this pin was added), so their bytes are frozen too —
    # which is why the GLM gated-arm refit lives in a SIBLING config below.
    "sdfbase_nemotron3nano4b_doc_full_fits.yaml": "a2077080423b5714",
    "sdfbase_glm46vflash_doc_full_fits.yaml": "27dbe45c5a60321a",
}


@pytest.fixture(scope="module")
def built():
    return msc.build_all()


def _cfg_of(text: str) -> dict:
    return yaml.safe_load(text)


# --------------------------------------------------------------------------
# Generator pins
# --------------------------------------------------------------------------

def test_committed_files_match_generator(built):
    for name in FILES:
        got = (REPO / "configs" / name).read_text()
        assert got == built[name], (
            f"configs/{name} drifted from make_sdfbase_configs.py + "
            f"configs/{msc.FOLDS_FILENAME} — a bare regeneration must "
            "reproduce the committed bytes; regenerate and commit both")


def _isolated_fits_site(monkeypatch, tmp_path, text: str) -> None:
    """An isolated site config holding ``text``, with no W&B project in the environment."""
    from sisteer import site as S
    cfg = tmp_path / "site" / "config.env"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(text)
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    for k in ("SISTEER_WANDB_PROJECT_FITS", "WANDB_PROJECT", "SISTEER_WANDB_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    S._SITE_CACHE.clear()


def test_new_key_gets_the_site_fits_project(monkeypatch, tmp_path):
    """A key with no committed config (the lookup pointed at an empty directory: a new model) gets the site's fits
    project, resolved when the config is generated; with no site project, the neutral default."""
    empty = tmp_path / "configs"
    empty.mkdir()
    monkeypatch.setattr(msc, "COMMITTED_CONFIGS_DIR", empty)
    key = _key_of(FILES[0])
    _isolated_fits_site(monkeypatch, tmp_path, "SISTEER_WANDB_PROJECT_FITS=example-fits\n")
    cfg = msc.build_config(key)
    assert cfg["wandb_project"] == "example-fits"
    assert "\nwandb_project: example-fits\n" in msc.render_config(key, cfg)
    _isolated_fits_site(monkeypatch, tmp_path, "")
    assert msc.build_config(key)["wandb_project"] == "sisteer-fits"


def test_committed_keys_ignore_the_site_project(monkeypatch, tmp_path):
    """With a site (and an environment) naming other projects, a bare regeneration still reproduces every committed
    config byte for byte: the site's project reaches only keys with no committed config."""
    _isolated_fits_site(monkeypatch, tmp_path, "SISTEER_WANDB_PROJECT_FITS=example-fits\nSISTEER_WANDB_PROJECT=example-site\n")
    monkeypatch.setenv("SISTEER_WANDB_PROJECT_FITS", "example-fits")
    monkeypatch.setenv("WANDB_PROJECT", "example-env")
    built = msc.build_all()
    for name in FILES:
        assert (REPO / "configs" / name).read_text() == built[name], name


def test_expect_sha_stability(built):
    """Same inputs -> same bytes -> same expect_sha across regenerations
    (the fits' --expect-config-sha pins the config FILE bytes)."""
    again = msc.build_all()
    for name in FILES:
        assert built[name] == again[name]
        sha1 = hashlib.sha256(built[name].encode()).hexdigest()[:16]
        sha2 = hashlib.sha256(again[name].encode()).hexdigest()[:16]
        assert sha1 == sha2


def test_running_generator_leaves_two_si_bytes_untouched(built, tmp_path):
    """§14: the two_si_* instruct configs' bytes are untouched by running
    the generator — exercised on a full copy of configs/ so the test
    can run the REAL write path."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    before = {}
    for p in sorted((REPO / "configs").glob("*.yaml")):
        shutil.copy2(p, cfg_dir / p.name)
        before[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    for name in FILES:
        msc.guarded_write(cfg_dir / name, built[name], repo=tmp_path)
    for p in sorted(cfg_dir.glob("*.yaml")):
        if p.name in FILES:
            assert p.read_text() == built[p.name]
        else:
            assert hashlib.sha256(p.read_bytes()).hexdigest() == \
                before[p.name], f"{p.name} was MUTATED by the generator"


def test_write_guard_refuses_non_sdfbase_paths(tmp_path):
    (tmp_path / "configs").mkdir()
    bad = [
        tmp_path / "configs" / "two_si_llama_fits.yaml",
        tmp_path / "configs" / "two_si_gemma4e4b_fits.yaml",
        tmp_path / "sdfbase_escape.yaml",                # not configs/
        tmp_path / "configs" / "sdfbase_x.yml",          # wrong suffix
        tmp_path / "configs" / "sub" / "sdfbase_x.yaml",  # nested
        tmp_path / "configs" / ".." / "sdfbase_x.yaml",   # traversal
        tmp_path / "configs" / "sdfbase_other.json",      # not THE store
        tmp_path / "configs" / "two_si_folds.json",
    ]
    for p in bad:
        with pytest.raises(SystemExit):
            msc.guarded_write(p, "x: 1\n", repo=tmp_path)
    # The allowed shapes work: generated yaml + the one folds store.
    ok = tmp_path / "configs" / "sdfbase_ok.yaml"
    msc.guarded_write(ok, "x: 1\n", repo=tmp_path)
    assert ok.read_text() == "x: 1\n"
    folds = tmp_path / "configs" / msc.FOLDS_FILENAME
    msc.guarded_write(folds, "{}\n", repo=tmp_path)
    assert folds.read_text() == "{}\n"


def test_generator_never_imports_make_fit_configs():
    """The instruct generator is NEVER invoked (its lanes silently skip on
    absent INTERIM sources; a byte change to two_si_* invalidates finished
    cells) — pinned at the import level."""
    import ast

    tree = ast.parse((REPO / "scripts/sdfbase/make_sdfbase_configs.py")
                     .read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any("make_fit_configs" in a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert "make_fit_configs" not in (node.module or "")


# --------------------------------------------------------------------------
# Config contents
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", FILES)
def test_budget_schema_present(built, name):
    """§14: training_budget_kind: document_tokens + completion_tokens:
    null in EVERY generated config (never the legacy 512 default)."""
    cfg = _cfg_of(built[name])
    assert cfg["training_budget_kind"] == "document_tokens"
    assert cfg["completion_tokens"] is None
    assert cfg["sequence_max_len"] == 8192
    assert cfg["max_len"] == 8192
    assert cfg["data_mode"] == "documents"


@pytest.mark.parametrize("name", FILES)
def test_teacher_prefix_cache_false_at_config_level(built, name):
    """teacher_prefix_cache: false is emitted for BOTH models at CONFIG
    level — the key the driver reads (train_two_si_objects reads
    cfg["teacher_prefix_cache"], default True; a train_kwargs copy would
    read as set and behave as unset, and the CE shape guards refuse it
    there). Honest for the CE arms (no teacher exists) and REQUIRED for
    qwen, where gradient_checkpointing would refuse a defaulted-true
    cache (maybe_enable_gradient_checkpointing's incompatibility guard)."""
    cfg = _cfg_of(built[name])
    assert cfg["teacher_prefix_cache"] is False
    # Never in the train_kwargs blocks — that placement is the exact
    # reads-as-set-behaves-as-unset shape refuse_ce_misconfiguration and
    # tune_arm's CESoloAdapter refuse.
    assert "teacher_prefix_cache" not in cfg["train_kwargs"]["default"]
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert "teacher_prefix_cache" not in cfg["train_kwargs"][arm]


@pytest.mark.parametrize("name", FILES)
def test_caps_table_as_specced(built, name):
    """The base lanes carry every cell; the chat lanes carry the mixed
    curve + ctl-100k only, each cell at the SAME caps row as the base
    configs (like-for-like per cell across the four models)."""
    cfg = _cfg_of(built[name])
    key = _key_of(name)
    if key in WS3_KEYS:
        assert cfg["si_ids"] == EXPECTED_WS3_SI_IDS
        assert cfg["train_kwargs_by_si"] == {
            si: EXPECTED_CAPS[si] for si in EXPECTED_WS3_SI_IDS}
        for absent in ("constv2-adv-1k", "constv2-adv-100k",
                       "constv2-mix-10m", "constv2-mix-full"):
            assert absent not in cfg["si_ids"]
            assert absent not in cfg["train_kwargs_by_si"]
    elif key in IT_KEYS:
        # instruct-direct: mix-full ONLY, at the cap-16 override — not the
        # shared CAPS row (which stays 4 / 1 for the base lanes, byte-pinned).
        assert cfg["si_ids"] == EXPECTED_IT_SI_IDS
        assert cfg["train_kwargs_by_si"] == EXPECTED_IT_CAPS
        assert msc.CAPS["constv2-mix-full"] == EXPECTED_CAPS["constv2-mix-full"]
        assert msc.caps_for(key, "constv2-mix-full") == \
            EXPECTED_IT_CAPS["constv2-mix-full"]
        for absent in EXPECTED_SI_IDS:
            if absent != "constv2-mix-full":
                assert absent not in cfg["si_ids"]
                assert absent not in cfg["train_kwargs_by_si"]
    elif key in FULL_SIBLING_KEYS:
        # 2026-09-06: mix-full + ctl2-full ONLY, both at cap 16 / patience
        # 2 via the per-MODEL override; the shared CAPS mix-full row stays
        # 4 / 1 (byte-pinned base configs) and the ctl2-full CAPS row is
        # the ctl-full lane's {16, 2}.
        assert cfg["si_ids"] == EXPECTED_FULL_SI_IDS
        assert cfg["train_kwargs_by_si"] == EXPECTED_FULL_CAPS
        assert msc.CAPS["constv2-mix-full"] == EXPECTED_CAPS["constv2-mix-full"]
        assert msc.CAPS["constv2ctl2-full"] == \
            EXPECTED_FULL_CAPS["constv2ctl2-full"]
        for si in EXPECTED_FULL_SI_IDS:
            assert msc.caps_for(key, si) == EXPECTED_FULL_CAPS[si]
        # (reseed merge) main's ten-cell EXPECTED_SI_IDS carries constv2ctl2-full, which
        # these lanes legitimately scope — exclude every EXPECTED_FULL_SI_IDS cell, not
        # just mix-full (the LLAMA8B / IT_EB64 branches' pattern)
        for absent in EXPECTED_SI_IDS:
            if absent not in EXPECTED_FULL_SI_IDS:
                assert absent not in cfg["si_ids"]
                assert absent not in cfg["train_kwargs_by_si"]
    elif key in IT_EB64_KEYS:
        # 2026-09-11: four cells — the two working cells at
        # exactly the shared CAPS rows, the two production twins at cap 16
        # / patience 2 — via the per-MODEL override; the shared CAPS
        # mix-full row stays 4 / 1 (byte-pinned base configs).
        assert cfg["si_ids"] == EXPECTED_IT_EB64_SI_IDS
        assert cfg["train_kwargs_by_si"] == EXPECTED_IT_EB64_CAPS
        assert msc.CAPS["constv2-mix-full"] == EXPECTED_CAPS["constv2-mix-full"]
        for si in ("constv2-mix-1m", "constv2-mix-10m"):
            assert EXPECTED_IT_EB64_CAPS[si] == EXPECTED_CAPS[si] == msc.CAPS[si]
        for si in EXPECTED_IT_EB64_SI_IDS:
            assert msc.caps_for(key, si) == EXPECTED_IT_EB64_CAPS[si]
        for absent in EXPECTED_SI_IDS:
            if absent not in EXPECTED_IT_EB64_SI_IDS:
                assert absent not in cfg["si_ids"]
                assert absent not in cfg["train_kwargs_by_si"]
    elif key in SEEDREP_KEYS:
        # 2026-09-15: s1 = three working cells at exactly the shared CAPS
        # rows + the two production cells at cap 16 / patience 2 (BOTH
        # corpora, scratch); s2 / s3 = the two production cells only — all
        # via the per-MODEL override; the shared CAPS mix-full row stays
        # 4 / 1 (the byte-pinned base config this lane replicates).
        ids = (EXPECTED_SEEDREP_SI_IDS_S1 if SEEDREP_SEED_OF[key] == 1
               else EXPECTED_SEEDREP_SI_IDS)
        assert cfg["si_ids"] == ids
        assert cfg["train_kwargs_by_si"] == {
            si: EXPECTED_SEEDREP_CAPS[si] for si in ids}
        assert msc.CAPS["constv2-mix-full"] == EXPECTED_CAPS["constv2-mix-full"]
        for si in ("constv2-mix-1k", "constv2-mix-1m", "constv2-mix-10m"):
            assert EXPECTED_SEEDREP_CAPS[si] == EXPECTED_CAPS[si] == msc.CAPS[si]
        for si in ids:
            assert msc.caps_for(key, si) == EXPECTED_SEEDREP_CAPS[si]
        for absent in EXPECTED_SI_IDS:
            if absent not in ids:
                assert absent not in cfg["si_ids"]
                assert absent not in cfg["train_kwargs_by_si"]
    elif key in PIPELINE_TEST_KEYS:
        # 2026-10-08: the worked example's demo lane is the paper's qwen35_4b_base lane under its own key (the same Qwen3.5-4B-Base
        # checkpoint; no per-MODEL si_ids override): the base lane's cell scope and the shared CAPS rows, byte for byte
        base = _cfg_of(built[msc.config_filename("qwen35_4b_base")])
        assert cfg["si_ids"] == base["si_ids"] == EXPECTED_SI_IDS and "si_ids" not in msc.MODELS[key]
        assert cfg["train_kwargs_by_si"] == base["train_kwargs_by_si"]
        assert msc.CAPS["constv2-mix-full"] == EXPECTED_CAPS["constv2-mix-full"]
        for si in EXPECTED_SI_IDS:
            assert msc.caps_for(key, si) == msc.caps_for("qwen35_4b_base", si) == EXPECTED_CAPS[si]     # the base lane's shared CAPS rows, no per-MODEL override
    elif key in LLAMA8B_KEYS:
        # 2026-09-09: five cells — the three working cells at exactly the
        # shared CAPS rows, the two production cells at cap 16 / patience 2
        # — all via the per-MODEL override; the shared CAPS mix-full row
        # stays 4 / 1 (byte-pinned base configs).
        assert cfg["si_ids"] == EXPECTED_LLAMA8B_SI_IDS
        assert cfg["train_kwargs_by_si"] == EXPECTED_LLAMA8B_CAPS
        assert msc.CAPS["constv2-mix-full"] == EXPECTED_CAPS["constv2-mix-full"]
        for si in ("constv2-mix-1k", "constv2-mix-1m", "constv2-mix-10m"):
            assert EXPECTED_LLAMA8B_CAPS[si] == EXPECTED_CAPS[si] == msc.CAPS[si]
        for absent in EXPECTED_SI_IDS:
            if absent not in EXPECTED_LLAMA8B_SI_IDS:
                assert absent not in cfg["si_ids"]
                assert absent not in cfg["train_kwargs_by_si"]
    else:
        assert cfg["train_kwargs_by_si"] == EXPECTED_CAPS
        assert cfg["si_ids"] == EXPECTED_SI_IDS
    # the scope is exactly what the generator table declares
    assert cfg["si_ids"] == msc.si_ids_for(key)
    assert list(cfg["train_kwargs_by_si"]) == cfg["si_ids"]


def test_ws3_scope_is_declared_on_exactly_the_chat_lanes():
    # scoped lanes: the chat lanes + the instruct-direct lanes;
    # the two base lanes carry the full SI_IDS
    assert {k for k, m in msc.MODELS.items() if m.get("si_ids")} == \
        WS3_KEYS | IT_KEYS | FULL_SIBLING_KEYS | LLAMA8B_KEYS | IT_EB64_KEYS \
        | set(SEEDREP_KEYS)                                       # the demo lane (PIPELINE_TEST_KEYS) carries the base lanes' full SI_IDS: no scope
    assert msc.WS3_SI_IDS == EXPECTED_WS3_SI_IDS
    for key in WS3_KEYS:
        assert msc.MODELS[key]["si_ids"] is msc.WS3_SI_IDS
    # a scope naming a cell with no caps row refuses loudly
    saved = msc.MODELS["glm46vflash_doc"]["si_ids"]
    try:
        msc.MODELS["glm46vflash_doc"]["si_ids"] = ["constv2-mix-1k", "nope"]
        with pytest.raises(SystemExit, match="no CAPS row"):
            msc.si_ids_for("glm46vflash_doc")
    finally:
        msc.MODELS["glm46vflash_doc"]["si_ids"] = saved


@pytest.mark.parametrize("name", FILES)
def test_sentinels_crash_if_unfilled(name):
    """The fail-loud seams: resolve_batch_size int()s batch_size_by_arm;
    a wave consuming lr/warmup/clip/accum before the folds must crash.
    Exercised on a fully-UNFOLDED build (folds_path=None) — the committed
    configs carry the memprobe fold, so their bs/accum are real ints."""
    cfg = _cfg_of(msc.build_all(folds_path=None)[name])
    for arm in ("psr_ce_all", "lora_r4_ce"):
        with pytest.raises(ValueError):
            int(cfg["batch_size_by_arm"][arm])
        tk = cfg["train_kwargs"][arm]
        for key in ("lr", "warmup_frac", "max_grad_norm"):
            with pytest.raises(ValueError):
                float(tk[key])
        with pytest.raises(ValueError):
            int(tk["accum_steps"])


@pytest.mark.parametrize("name", FILES)
def test_committed_configs_keep_uncovered_sentinels(built, name):
    """Sentinel behavior is derived from the folds store: covered keys
    carry real folded numbers (memprobe widths where the "memprobe"
    section names the model+arm; HPO winners where the "hpo" section
    does), UNcovered keys keep their sentinels and int()/float()-crash.
    Coverage is read from the store so this pin survives each fold
    landing (qwen r1 winners folded 2026-09-02; gemma pends its r2;
    glm46vflash_doc pends BOTH folds — its memprobe has not run, so both
    its seams must still crash)."""
    cfg = _cfg_of(built[name])
    folds = msc.load_folds()
    key = _key_of(name)
    # The two base lanes are memprobe-folded (the store must say so — a
    # fold that silently dropped out would otherwise read as "pending").
    assert {"gemma4e4b_base", "qwen35_4b_base"} <= set(folds["memprobe"])
    for arm in ("psr_ce_all", "lora_r4_ce"):
        tk = cfg["train_kwargs"][arm]
        mp = folds["memprobe"].get(key, {}).get(arm)
        if mp is not None:  # covered: committed width is the store's
            bs = int(cfg["batch_size_by_arm"][arm])
            assert bs == mp
            # the KEY's effective batch (64 on the Llama lane, the module
            # constant 8 everywhere else), never the constant alone
            assert int(tk["accum_steps"]) == \
                -(-msc.effective_batch_for(key) // bs)
        else:               # uncovered: width AND accum must still crash
            with pytest.raises(ValueError):
                int(cfg["batch_size_by_arm"][arm])
            with pytest.raises(ValueError):
                int(tk["accum_steps"])
        hpo = folds.get("hpo", {}).get(key, {}).get(arm) or {}
        for hkey, fold_name in (("lr", "lr"), ("warmup_frac", "warmup_frac"),
                                ("max_grad_norm", "max_grad_norm")):
            # coverage is per WINNER KEY: 04's init-scheme stamp (FD 7) may leave a
            # row carrying only init_scheme on an arm whose search has not closed
            # (the LoRA arm of the front-door test-model lane is still open, FD 6a) —
            # its winner keys are uncovered and keep their sentinels
            if hpo.get(fold_name) is not None:  # covered: committed value is the store's, verbatim
                assert float(tk[hkey]) == pytest.approx(float(hpo[fold_name]))
            else:    # uncovered: sentinel must still crash
                with pytest.raises(ValueError):
                    float(tk[hkey])


def test_gradient_checkpointing_table():
    """qwen3_5 is the hybrid-attention architecture: checkpointing is
    an architecture requirement on both arms. The dense full-attention
    models (gemma-4-E4B base; GLM-4.6V-Flash, dense 10.3B) run without
    it — the table is pinned here so test_arm_blocks' per-file branch
    cannot drift with it."""
    assert msc.MODELS["qwen35_4b_base"]["gradient_checkpointing"] is True
    assert msc.MODELS["gemma4e4b_base"]["gradient_checkpointing"] is False
    assert msc.MODELS["glm46vflash_doc"]["gradient_checkpointing"] is False
    # nemotron_h: hybrid Mamba2/MLP/attention — checkpointing ON (and it is
    # what turns use_cache off for the training forward on this stack).
    assert msc.MODELS["nemotron3nano4b_doc_full"]["gradient_checkpointing"] is True
    # the instruct-direct lanes follow their ARCHITECTURE, not their lane
    assert msc.MODELS["qwen35_4b_it_doc"]["gradient_checkpointing"] is True
    assert msc.MODELS["gemma4e4b_it_doc"]["gradient_checkpointing"] is False
    # dense Llama-3.1-8B, both checkpoints: OFF (the memprobe may overrule)
    assert msc.MODELS["llama8b_base"]["gradient_checkpointing"] is False
    assert msc.MODELS["llama8b_it_doc"]["gradient_checkpointing"] is False
    # the qwen-base seed replicates (2026-09-15) are the qwen3_5 hybrid: ON
    for key in SEEDREP_KEYS:
        assert msc.MODELS[key]["gradient_checkpointing"] is True
    # paper-companion-repo (2026-09-29): ON for the front door's test-model
    # lane — the dense 31B (60 blocks of hidden 5,376 at 8,192 tokens; the
    # FD 23 probe's premise)
    for key in PIPELINE_TEST_KEYS:
        assert msc.MODELS[key]["gradient_checkpointing"] is True
        assert msc.PIPELINE_TEST_KEYS == PIPELINE_TEST_KEYS


@pytest.mark.parametrize("name", FILES)
def test_arm_blocks(built, name):
    cfg = _cfg_of(built[name])
    # gradient checkpointing follows the model table: ON for the hybrid
    # stacks (qwen3_5; nemotron_h, where it also turns use_cache off for a
    # cache transformers 5.14.1 cannot allocate), OFF for the dense
    # full-attention models (gemma-4-E4B base, GLM-4.6V-Flash).
    key = _key_of(name)
    checkpointed = msc.MODELS[key]["gradient_checkpointing"]
    assert checkpointed is (key in ("qwen35_4b_base",
                                    "qwen35_4b_it_doc",
                                    "nemotron3nano4b_doc_full",
                                    "qwen35_4b_it_doc_eb64", *SEEDREP_KEYS,
                                    *PIPELINE_TEST_KEYS))
    for arm in ("psr_ce_all", "lora_r4_ce"):
        tk = cfg["train_kwargs"][arm]
        # The SI-teacher floor is EXPLICITLY null (replaced, not silently
        # disabled); the CE-native bar carries the verdict.
        assert tk["floor_ratio_max"] is None
        assert tk["ce_floor_ratio_max"] == 1.1
        assert tk["weight_decay"] == 0.01
        assert tk["lr_schedule"] == "warmup_cosine"
        if checkpointed:
            assert tk["gradient_checkpointing"] is True
        else:
            assert "gradient_checkpointing" not in tk
    assert cfg["train_kwargs"]["psr_ce_all"]["gate_reg"] == 0.1
    assert "gate_reg" not in cfg["train_kwargs"]["lora_r4_ce"]
    # The codebase DEFAULT is the paper's rank 4 on o_proj (attention-out only). A model whose hpo row in configs/sdfbase_folds.json
    # carries an architecture pick (hpo_pick -> the fold) follows ITS row; every other model still asserts the attention-out
    # default. The expectation is fold-or-default, never "whatever the yaml says".
    fold_row = (msc.load_folds()["hpo"].get(key) or {}).get("lora_r4_ce") or {}
    exp_rank = fold_row.get("lora_rank", msc.PAPER_LORA_RANK)
    exp_modules = sorted(fold_row.get("lora_modules", msc.PAPER_LORA_MODULES))
    assert cfg["train_kwargs"]["lora_r4_ce"]["lora_rank"] == exp_rank
    if key not in PIPELINE_TEST_KEYS:
        assert (exp_rank, exp_modules) == (4, ["o_proj"]), f"{key}: a model without a folded pick sits on the paper default"
    sel = cfg["train_kwargs"]["lora_r4_ce"].get("fit_selection")
    if sel is not None:
        assert (sel["lora_rank"], sel["lora_modules"]) == (exp_rank, exp_modules)
        assert sel["init_scheme"] == fold_row.get("init_scheme", msc.INIT_SCHEME_DEFAULT)
    assert "lora_rank" not in cfg["train_kwargs"]["psr_ce_all"]
    # the plan-§3.7 fit_selection block rides EXACTLY the 2026-09-29 front-door
    # lane (declared "fit_selection"); every earlier lane's bytes have none
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert ("fit_selection" in cfg["train_kwargs"][arm]) is (key in PIPELINE_TEST_KEYS)
    assert bool(msc.MODELS[key].get("fit_selection")) is (key in PIPELINE_TEST_KEYS)


@pytest.mark.parametrize("name", FILES)
def test_default_block_and_heldout_pool(built, name):
    cfg = _cfg_of(built[name])
    d = cfg["train_kwargs"]["default"]
    key = _key_of(name)
    assert d["holdout_frac"] == 0.0
    # the `_full` lanes point at their parent lane's staged pool dir
    # (pool_key); every other key at its own
    assert d["heldout_pool"] == \
        f"/vol/pool/{msc.pool_key_for(key)}/pairs_constv2-heldout.jsonl"
    assert msc.pool_key_for(key) == \
        {**FULL_PARENT_OF, **IT_EB64_PARENT_OF}.get(key, key)
    assert d["positions"] == "all"
    assert d["probe_n"] is None
    assert d["early_stop_min_epochs"] == 1
    assert d["early_stop_min_delta"] == 0.0
    # a committed key re-emits the project its committed file records (the file's bytes are its fits' identity)
    assert cfg["wandb_project"] == _cfg_of((REPO / "configs" / name).read_text())["wandb_project"]
    assert cfg["wandb_fits"] is True


def test_model_pins_cross_consistent_with_model_specs(built):
    """The sdfbase configs and derive_model_config.MODEL_SPECS pin the
    SAME model/revision/layer (onboard_model_smoke resolves specs from
    these pins)."""
    from derive_model_config import MODEL_SPECS, SDFBASE_SLUGS

    # every sdfbase slug has a generated file, and every generated file a
    # slug — the two tables cannot drift apart in either direction
    assert set(SDFBASE_SLUGS) == set(msc.MODELS) == {_key_of(n) for n in FILES}
    for name in FILES:
        cfg = _cfg_of(built[name])
        key = _key_of(name)
        assert key in SDFBASE_SLUGS
        spec = MODEL_SPECS[key]
        assert cfg["model"] == spec.model
        assert cfg["model_revision"] == spec.revision
        assert cfg["layer"] == spec.layer == round(0.469 * spec.n_layers)
        assert cfg["dtype"] == "bfloat16"
        assert cfg["attn_implementation"] == "sdpa"
    # every generator key has a spec, every sdfbase spec has a generator key
    assert set(msc.MODELS) == set(SDFBASE_SLUGS) == set(KEY_OF.values())


def test_nemotron_doc_config_header_names_the_instruct_checkpoint(built):
    """The provenance header says what the checkpoint IS: the two base
    configs keep their historical "BASE model" line byte-for-byte (their
    expect_sha); the nemotron file — an instruct checkpoint with no base
    variant — wears the SAME chat_model wording as the GLM lane (one flag,
    one header form for every chat/instruct checkpoint on the doc lane)."""
    nemo = built["sdfbase_nemotron3nano4b_doc_full_fits.yaml"]
    assert msc.MODELS["nemotron3nano4b_doc_full"]["chat_model"] is True
    head = nemo.split("\nmodel:", 1)[0]
    assert "(CHAT model; no base variant of it exists — " in head and " — no chat template is" in head
    assert "BASE model" not in head
    for name in FILES:
        m = msc.MODELS[_key_of(name)]
        if m.get("header_kind"):
            # an entry with its own header_kind says exactly that
            assert f"({m['header_kind']} — no chat template is" in built[name]
        elif not m.get("chat_model"):
            assert "(BASE model — no chat template is" in built[name]
    cfg = _cfg_of(nemo)
    # the paper's full-scale lane shares the RETIRED data-scaling ladder's pool dir (make_sdfbase_configs.RETIRED_POOL_PARENTS,
    # 2026-10-08): the path is part of the committed config's bytes, the fit identity of the paper's 17 Nemotron objects
    assert cfg["train_kwargs"]["default"]["heldout_pool"] == \
        "/vol/pool/nemotron3nano4b_doc/pairs_constv2-heldout.jsonl"


def test_glm_doc_lane_is_the_chat_checkpoint(built):
    """glm46vflash_doc pins the SAME checkpoint as the two_si slug
    glm46vflash (no base variant of GLM-4.6V exists — the doc-CE fits
    target the chat model), and the generated header says so instead of
    calling it a base model; the base lanes' header wording is untouched
    (their bytes are pinned by test_committed_files_match_generator)."""
    from derive_model_config import MODEL_SPECS

    two_si, doc = MODEL_SPECS["glm46vflash"], MODEL_SPECS["glm46vflash_doc"]
    assert (two_si.model, two_si.revision) == (doc.model, doc.revision)
    assert (doc.n_layers, doc.hidden_size, doc.layer) == (40, 4096, 19)
    assert msc.MODELS["glm46vflash_doc"]["chat_model"] is True
    head = built["sdfbase_glm46vflash_doc_fits.yaml"].split("\nmodel:", 1)[0]
    assert "(CHAT model; no base variant of it exists — no chat template is" \
        in head
    assert "BASE model" not in head
    for name in ("sdfbase_gemma4e4b_base_fits.yaml",
                 "sdfbase_qwen35_4b_base_fits.yaml"):
        base_head = built[name].split("\nmodel:", 1)[0]
        assert "(BASE model — no chat template is" in base_head
        assert "CHAT model" not in base_head
    # the same flat-document rendering as every other lane — the flag is
    # header wording only
    cfg = _cfg_of(built["sdfbase_glm46vflash_doc_fits.yaml"])
    assert cfg["data_mode"] == "documents"
    assert cfg["train_kwargs"]["default"]["heldout_pool"] == \
        "/vol/pool/glm46vflash_doc/pairs_constv2-heldout.jsonl"


@pytest.mark.parametrize("slug", ["gemma4e4b_base", "qwen35_4b_base",
                                  "glm46vflash_doc",
                                  "gemma4e4b_it_doc", "qwen35_4b_it_doc",
                                  "llama8b_base", "llama8b_it_doc",
                                  "qwen35_4b_base_eb64_s1"])
def test_derive_model_config_refuses_base_slugs(tmp_path, slug):
    """derive_model_config must never write a two_si_<sdfbase slug>.yaml —
    these configs come from THIS generator (the GLM and Nemotron doc lanes
    included, even though their two_si siblings ARE derived there)."""
    import derive_model_config as dmc

    assert slug in dmc.SDFBASE_SLUGS
    counts = tmp_path / "counts.json"
    counts.write_text("{}")
    with pytest.raises(SystemExit, match="sdf-base-steering"):
        dmc.main(["--models", slug, "--counts", str(counts)])
    assert not (REPO / "configs" / f"two_si_{slug}.yaml").exists()


# --------------------------------------------------------------------------
# INSTRUCT-DIRECT lanes
# --------------------------------------------------------------------------


def _fingerprint_holds(name: str, data: bytes) -> bool:
    """A fingerprinted config's pinned sha16 still names ``data``: the bytes hash to it, or configs/SCRUB_MAP.tsv certifies
    that ``data`` is the public release's rule rewrite of the bytes the fits were fingerprinted under (the pin itself stays
    as recorded, since fits keep their fingerprints)."""
    from sisteer import scrub_map
    got = hashlib.sha256(data).hexdigest()
    return scrub_map.check_prefix(FROZEN_CONFIG_SHA16[name], got, path=f"configs/{name}",
                                  map_path=REPO / "configs" / "SCRUB_MAP.tsv") is not None


def test_frozen_configs_keep_their_fingerprinted_bytes():
    """The four configs finished / in-flight cells were fingerprinted under
    are byte-pinned by sha16 (the manifests' config_sha). A lane adding a
    model or a cell must not move them — the instruct-direct lanes keep
    their cap-16 row in a per-MODEL override for exactly this reason."""
    for name, sha16 in FROZEN_CONFIG_SHA16.items():
        got = hashlib.sha256((REPO / "configs" / name).read_bytes()).hexdigest()
        assert _fingerprint_holds(name, (REPO / "configs" / name).read_bytes()), (name, got[:16], sha16)


@pytest.mark.parametrize("key", sorted(IT_KEYS))
def test_it_direct_lane_is_the_battery_instruct_model_trained_directly(
        built, key):
    """Each instruct-direct lane pins EXACTLY the checkpoint the transfer
    battery serves (the r2 transfer config's model / model_revision), says
    so in its header (an INSTRUCT model with a base sibling — neither
    'BASE model' nor the chat lanes' 'no base variant' wording), keeps the
    doc lane's rendering (data_mode documents), and evaluates on its own
    staged copy of the shared held-out pool."""
    import yaml as _yaml

    slug = {"gemma4e4b_it_doc": "gemma4e4b",
            "qwen35_4b_it_doc": "qwen35_4b"}[key]
    transfer = _yaml.safe_load(
        (REPO / "configs" / f"sdfbase_transfer_{slug}_r2.yaml").read_text())
    text = built[msc.config_filename(key)]
    cfg = _cfg_of(text)
    assert cfg["model"] == transfer["model"] == msc.MODELS[key]["model"]
    assert cfg["model_revision"] == transfer["model_revision"] == \
        msc.MODELS[key]["revision"]
    assert cfg["model"].endswith(("-it", "Qwen3.5-4B"))  # the instruct repo
    head = text.split("\nmodel:", 1)[0]
    assert "INSTRUCT model, trained DIRECTLY (the instruct-direct arm" in head
    assert "BASE model" not in head
    assert "no base variant of it exists" not in head
    assert msc.MODELS[key]["chat_model"] is True
    assert cfg["data_mode"] == "documents"
    assert cfg["train_kwargs"]["default"]["heldout_pool"] == \
        f"/vol/pool/{key}/pairs_constv2-heldout.jsonl"
    assert cfg["layer"] == msc.MODELS[IT_BASE_OF[key]]["layer"]


@pytest.mark.parametrize("key", sorted(IT_KEYS))
def test_it_direct_lane_reuses_its_base_lanes_folds_verbatim(built, key):
    """HPO winners and memprobe widths are the same
    architecture's base lane's, copied into the folds store under the
    it-direct key — so the committed config carries REAL numbers (no
    sentinel seam left to crash) and every optimizer field equals the base
    config's, bit for bit; only model / revision / scope / caps differ."""
    base = IT_BASE_OF[key]
    folds = msc.load_folds()
    assert folds["hpo"][key] == folds["hpo"][base]
    assert folds["memprobe"][key] == folds["memprobe"][base]
    it_cfg = _cfg_of(built[msc.config_filename(key)])
    base_cfg = _cfg_of(built[msc.config_filename(base)])
    assert it_cfg["batch_size_by_arm"] == base_cfg["batch_size_by_arm"]
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert it_cfg["train_kwargs"][arm] == base_cfg["train_kwargs"][arm]
        assert not isinstance(it_cfg["train_kwargs"][arm]["lr"], str)
    # what differs is exactly the identity, the scope and the caps row
    differ = ("model", "model_revision", "si_ids", "train_kwargs_by_si",
              "train_kwargs", "leading_special_tokens")
    same = {k: v for k, v in it_cfg.items() if k not in differ}
    base_same = {k: v for k, v in base_cfg.items() if k not in differ}
    assert same == base_same
    assert it_cfg["train_kwargs"]["default"] != base_cfg["train_kwargs"]["default"]
    assert {k: v for k, v in it_cfg["train_kwargs"]["default"].items()
            if k != "heldout_pool"} == \
        {k: v for k, v in base_cfg["train_kwargs"]["default"].items()
         if k != "heldout_pool"}


#: it-direct key -> the leading-special-token policy its config declares =
#: its BASE sibling's template-free convention (gemma-4-E4B prepends <bos>;
#: Qwen3.5-4B-Base prepends nothing and has no BOS token).
EXPECTED_LEADING = {"gemma4e4b_it_doc": "bos", "qwen35_4b_it_doc": "none",
                    # the Llama lane (2026-09-09): MEASURED on both real
                    # tokenizers at the pinned revisions — both lead a bare
                    # document with BOS 128000, so "bos" on BOTH keys is a
                    # no-op installation that asserts the observed behaviour
                    # (verified-facts §15)
                    "llama8b_base": "bos", "llama8b_it_doc": "bos",
                    # the batch-64 siblings (2026-09-11) declare
                    # exactly their `_it_doc` parents' policy
                    "gemma4e4b_it_doc_eb64": "bos",
                    "qwen35_4b_it_doc_eb64": "none"}


def test_leading_special_tokens_declared_on_exactly_the_it_direct_lanes(built):
    """The doc-CE path lets the tokenizer's post-processor decide what leads
    a sequence, and gemma-4-E4B-it's adds NOTHING where the base's adds
    <bos>. The instruct-direct configs therefore DECLARE the base sibling's
    convention (asserted by the fit before any step), and the Llama lane
    declares its MEASURED convention on both keys; the frozen configs
    declare nothing (their bytes are pinned) and keep observing their
    tokenizers' own convention."""
    from sisteer.extraction.leading_tokens import LEADING_KEY, POLICIES

    assert msc.LEADING_TOKEN_KEY == LEADING_KEY
    assert msc.LEADING_TOKEN_POLICIES == POLICIES
    declared = {k for k, m in msc.MODELS.items() if m.get("leading_special_tokens")}
    assert declared == set(EXPECTED_LEADING)
    for name in FILES:
        key = _key_of(name)
        cfg = _cfg_of(built[name])
        text = built[name]
        if key in EXPECTED_LEADING:
            assert cfg[LEADING_KEY] == EXPECTED_LEADING[key]
            assert msc._leading_policy(key) == EXPECTED_LEADING[key]
            # emitted right after teacher_prefix_cache, before si_ids
            keys = list(cfg)
            assert keys.index(LEADING_KEY) == keys.index("teacher_prefix_cache") + 1
            assert f"# leading_special_tokens: {EXPECTED_LEADING[key]}" in \
                text.split("\nmodel:", 1)[0]
        else:
            assert LEADING_KEY not in cfg
            assert LEADING_KEY not in text
    # an unknown policy refuses at build time
    saved = msc.MODELS["gemma4e4b_it_doc"]["leading_special_tokens"]
    try:
        msc.MODELS["gemma4e4b_it_doc"]["leading_special_tokens"] = "eos"
        with pytest.raises(SystemExit, match="leading_special_tokens='eos'"):
            msc.build_config("gemma4e4b_it_doc")
    finally:
        msc.MODELS["gemma4e4b_it_doc"]["leading_special_tokens"] = saved


# --------------------------------------------------------------------------
# FULL-SCALE Slurm lanes (2026-09-06)
# --------------------------------------------------------------------------


#: the `_full` lanes whose frozen parent lane is STILL a MODELS row: the Nemotron ladder lane (`nemotron3nano4b_doc`) was retired
#: 2026-10-08 (its full-scale lane is the paper's; RETIRED_POOL_PARENTS keeps its pool dir resolvable), so the verbatim-parent pins
#: below run on the GLM pair alone and the retired parent is pinned by test_a_retired_parent_lane_keeps_its_full_lanes_pool_resolvable
FULL_KEYS_WITH_PARENT = sorted(k for k in FULL_KEYS if FULL_PARENT_OF[k] in msc.MODELS)


def test_a_retired_parent_lane_keeps_its_full_lanes_pool_resolvable():
    retired = sorted(k for k in FULL_KEYS if FULL_PARENT_OF[k] not in msc.MODELS)
    assert retired == ["nemotron3nano4b_doc_full"] and FULL_KEYS_WITH_PARENT == ["glm46vflash_doc_full"]
    parent = FULL_PARENT_OF["nemotron3nano4b_doc_full"]
    assert parent in msc.RETIRED_POOL_PARENTS and msc.MODELS["nemotron3nano4b_doc_full"]["pool_key"] == parent
    assert msc.pool_key_for("nemotron3nano4b_doc_full") == parent                                   # the held-out pool path still names the retired parent's dir
    assert not (REPO / "configs" / f"sdfbase_{parent}_fits.yaml").exists() and parent not in msc.load_folds()["hpo"]


@pytest.mark.parametrize("key", FULL_KEYS_WITH_PARENT)
def test_full_lane_is_its_parents_checkpoint_with_a_new_cell_scope(built,
                                                                    key):
    """A `_full` lane pins EXACTLY its frozen parent's checkpoint, layer and
    gradient-checkpointing choice — the only differences are the cell scope
    (mix-full + ctl2-full), the caps row and the header — and its config's
    bytes are NOT the parent's (a different expect_sha, by construction),
    while the parent's bytes are the frozen ones (pinned separately)."""
    parent = FULL_PARENT_OF[key]
    me, pa = msc.MODELS[key], msc.MODELS[parent]
    for field in ("model", "revision", "layer", "gradient_checkpointing"):
        assert me[field] == pa[field], field
    assert me["chat_model"] is True and "FULL-SCALE" in me["header_kind"]
    assert me["pool_key"] == parent
    cfg = _cfg_of(built[msc.config_filename(key)])
    pcfg = _cfg_of(built[msc.config_filename(parent)])
    for field in ("model", "model_revision", "layer", "dtype",
                  "attn_implementation", "max_len", "data_mode",
                  "training_budget_kind", "sequence_max_len",
                  "completion_tokens", "teacher_prefix_cache",
                  "batch_size_by_arm", "wandb_project", "wandb_fits"):
        assert cfg[field] == pcfg[field], field
    # same optimizer block per arm (the HPO winners and accum), same
    # default block INCLUDING the held-out pool path (the parent's dir)
    assert cfg["train_kwargs"] == pcfg["train_kwargs"]
    assert cfg["si_ids"] == EXPECTED_FULL_SI_IDS
    assert built[msc.config_filename(key)] != built[msc.config_filename(parent)]
    head = built[msc.config_filename(key)].split("\nmodel:", 1)[0]
    assert "FULL-SCALE lane of the frozen " + parent in head
    assert "leading_special_tokens" not in cfg   # the parent declares none


@pytest.mark.parametrize("key", FULL_KEYS_WITH_PARENT)
def test_full_lane_reuses_its_parents_folds_verbatim(key):
    folds = msc.load_folds()
    parent = FULL_PARENT_OF[key]
    for sec in ("memprobe", "hpo"):
        assert folds[sec][key] == folds[sec][parent], sec


def test_pool_key_validation():
    """pool_key must name a MODELS key of the SAME checkpoint and never
    chain — pools and tokcache sidecars are per checkpoint."""
    key = "nemotron3nano4b_doc_full"
    saved = dict(msc.MODELS[key])
    try:
        msc.MODELS[key]["pool_key"] = "nope"
        with pytest.raises(SystemExit, match="not a MODELS key"):
            msc.pool_key_for(key)
        msc.MODELS[key]["pool_key"] = "glm46vflash_doc"   # another checkpoint
        with pytest.raises(SystemExit, match="per checkpoint"):
            msc.pool_key_for(key)
        msc.MODELS[key]["pool_key"] = "glm46vflash_doc_full"  # chains
        with pytest.raises(SystemExit, match="never chains"):
            msc.pool_key_for(key)
    finally:
        msc.MODELS[key] = saved
    # the frozen keys have no pool_key: their own dir
    for k in ("glm46vflash_doc", "gemma4e4b_base"):
        assert msc.pool_key_for(k) == k
    # a RETIRED parent (the Nemotron data-scaling ladder, 2026-10-08) still resolves the full lane's pool dir, under the
    # same-checkpoint rule; a retired dir whose pins differ refuses by name
    assert "nemotron3nano4b_doc" not in msc.MODELS
    assert msc.pool_key_for("nemotron3nano4b_doc_full") == "nemotron3nano4b_doc"
    saved_retired = dict(msc.RETIRED_POOL_PARENTS)
    try:
        msc.RETIRED_POOL_PARENTS["nemotron3nano4b_doc"] = ("example/other", "0" * 40)
        with pytest.raises(SystemExit, match="retired pool dir"):
            msc.pool_key_for("nemotron3nano4b_doc_full")
    finally:
        msc.RETIRED_POOL_PARENTS.clear(); msc.RETIRED_POOL_PARENTS.update(saved_retired)


# test_ctl2_full_caps_row_moves_no_frozen_bytes (itdoc-ctl2) was dropped at the reseed merge: its premise
# (constv2ctl2-full absent from the four frozen _base/_doc configs) is false on main since the ten-cell SI_IDS;
# the FROZEN_CONFIG_SHA16 byte pins in test_frozen_configs_keep_their_fingerprinted_bytes remain the real guard.


def test_caps_override_is_scoped_and_sane():
    """A "caps" override may name only cells in the entry's si_ids (a row
    for a cell the config does not carry would silently mean nothing) and
    must be a complete, sane {epochs, early_stop_patience} pair; every
    other cell falls through to the shared CAPS row untouched."""
    key = "gemma4e4b_it_doc"
    saved = msc.MODELS[key]["caps"]
    try:
        msc.MODELS[key]["caps"] = {"constv2-mix-1m": {"epochs": 16,
                                                     "early_stop_patience": 2}}
        with pytest.raises(SystemExit, match="outside its si_ids scope"):
            msc.build_config(key)
        msc.MODELS[key]["caps"] = {"constv2-mix-full": {"epochs": 16}}
        with pytest.raises(SystemExit, match="exactly epochs"):
            msc.build_config(key)
        msc.MODELS[key]["caps"] = {"constv2-mix-full": {
            "epochs": 2, "early_stop_patience": 2}}
        with pytest.raises(SystemExit, match="patience < epochs"):
            msc.build_config(key)
        msc.MODELS[key]["caps"] = None
        assert msc.caps_for(key, "constv2-mix-full") == \
            msc.CAPS["constv2-mix-full"]
    finally:
        msc.MODELS[key]["caps"] = saved
    # a fresh dict every time: mutating the row cannot leak into CAPS
    row = msc.caps_for("gemma4e4b_base", "constv2-mix-full")
    row["epochs"] = 999
    assert msc.CAPS["constv2-mix-full"]["epochs"] == 4


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------

def test_fold_memprobe_fills_bs_and_accum():
    # folds_path=None isolates the test from the committed folds store.
    mp = {"gemma4e4b_base": {"psr_ce_all": 2, "lora_r4_ce": 8}}
    built = msc.build_all(memprobe=mp, folds_path=None)
    cfg = _cfg_of(built["sdfbase_gemma4e4b_base_fits.yaml"])
    assert cfg["batch_size_by_arm"] == {"psr_ce_all": 2, "lora_r4_ce": 8}
    assert cfg["train_kwargs"]["psr_ce_all"]["accum_steps"] == 4  # 8/2
    assert cfg["train_kwargs"]["lora_r4_ce"]["accum_steps"] == 1  # 8/8
    # The other model keeps its sentinels.
    other = _cfg_of(built["sdfbase_qwen35_4b_base_fits.yaml"])
    with pytest.raises(ValueError):
        int(other["batch_size_by_arm"]["psr_ce_all"])


def test_hpo_fold_vocabulary_is_the_winner_plus_the_guard_plus_the_selection_keys():
    """The committed store's hpo rows carry the winner + the guard and, since
    the stage-E front-door lanes, the
    selection keys; any other key refuses at build time. The sixteen paper
    lanes' rows carry no selection key (their bytes never move); a front-door
    test-model lane's folds are repo data in the same store."""
    assert msc.HPO_WINNER_KEYS == ("lr", "warmup_frac", "max_grad_norm")
    assert set(msc.HPO_FOLD_KEYS) == set(msc.HPO_WINNER_KEYS) | {"selection_guard"} | set(msc.HPO_SELECTION_KEYS)
    folds = msc.load_folds()
    used = {k for arms in folds["hpo"].values() for row in arms.values() for k in row}
    assert used <= set(msc.HPO_FOLD_KEYS)
    paper_rows = {k: arms for k, arms in folds["hpo"].items() if k not in PIPELINE_TEST_KEYS}
    assert not {k for arms in paper_rows.values() for row in arms.values() for k in row} & set(msc.HPO_SELECTION_KEYS)
    for key in set(folds["hpo"]) & set(PIPELINE_TEST_KEYS):
        for arm, row in folds["hpo"][key].items():
            if set(row) <= {"init_scheme"}:   # 04's init-scheme stamp alone (FD 7): the arm's search has not closed
                continue
            assert row.get("fit_lr_selected_by") in msc.FIT_LR_SELECTED_BY and row.get("fit_search_id") and "lr" in row, (key, arm)
    with pytest.raises(SystemExit, match="unknown key"):
        msc.build_all(hpo={"gemma4e4b_base": {"psr_ce_all": {
            "lr": 1e-4, "warmup_frac": 0.05, "max_grad_norm": 1.0, "epochs": 3}}},
            folds_path=None)


def test_fold_hpo_fills_winners_and_refuses_bad_input():
    win = {"qwen35_4b_base": {"psr_ce_all": {
        "lr": 2.2e-3, "warmup_frac": 0.15, "max_grad_norm": 0.19}}}
    built = msc.build_all(hpo=win, folds_path=None)
    tk = _cfg_of(built["sdfbase_qwen35_4b_base_fits.yaml"])[
        "train_kwargs"]["psr_ce_all"]
    assert tk["lr"] == pytest.approx(2.2e-3)
    assert tk["warmup_frac"] == 0.15
    assert tk["max_grad_norm"] == pytest.approx(0.19)
    # The unfolded sibling arm keeps its sentinel.
    tk2 = _cfg_of(built["sdfbase_qwen35_4b_base_fits.yaml"])[
        "train_kwargs"]["lora_r4_ce"]
    with pytest.raises(ValueError):
        float(tk2["lr"])
    with pytest.raises(SystemExit):
        msc.build_all(hpo={"nope": {}}, folds_path=None)
    with pytest.raises(SystemExit):
        msc.build_all(hpo={"gemma4e4b_base": {"psr_kl_all": {}}},
                      folds_path=None)
    with pytest.raises(SystemExit):
        msc.build_all(hpo={"gemma4e4b_base": {"psr_ce_all": {"lr": 1e-4}}},
                      folds_path=None)


# --------------------------------------------------------------------------
# The declarative folds store (configs/sdfbase_folds.json)
# --------------------------------------------------------------------------

def test_committed_folds_file_is_consumed_by_default():
    """Bare build_all() reads configs/sdfbase_folds.json: it must equal an
    explicit build from the file's own sections, and differ from an
    unfolded build (the fold is real, not a no-op)."""
    folds = msc.load_folds()
    assert folds["memprobe"], "committed folds file carries no memprobe fold"
    explicit = msc.build_all(memprobe=folds["memprobe"],
                             hpo=folds["hpo"], folds_path=None)
    assert msc.build_all() == explicit
    assert msc.build_all() != msc.build_all(folds_path=None)


def test_missing_folds_file_yields_sentinels(tmp_path):
    """No folds file -> every fold-fed key keeps its sentinel (a fresh
    lane before its memprobe must still fail loudly)."""
    nope = tmp_path / "sdfbase_folds.json"  # does not exist
    built = msc.build_all(folds_path=nope)
    assert built == msc.build_all(folds_path=None)
    cfg = _cfg_of(built["sdfbase_qwen35_4b_base_fits.yaml"])
    with pytest.raises(ValueError):
        int(cfg["batch_size_by_arm"]["lora_r4_ce"])


def test_folds_file_round_trips_deterministically():
    """dump_folds(load_folds()) reproduces the committed bytes — the store
    itself is regeneration-stable, like the configs it feeds."""
    committed = (REPO / "configs" / msc.FOLDS_FILENAME).read_bytes()
    assert msc.dump_folds(msc.load_folds()).encode() == committed


def test_load_folds_refuses_unknown_section(tmp_path):
    bad = tmp_path / "sdfbase_folds.json"
    bad.write_text('{"memprobe": {}, "typo_section": {}}')
    with pytest.raises(SystemExit, match="unknown fold section"):
        msc.load_folds(bad)


def test_update_folds_merges_per_model_per_arm(tmp_path):
    """--fold-* semantics: deep-merge into the store (a partial fold never
    drops other lanes' values), guarded write, consumed by build_all."""
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / msc.FOLDS_FILENAME
    msc.update_folds("memprobe",
                     {"gemma4e4b_base": {"psr_ce_all": 2}},
                     path=path, repo=tmp_path)
    msc.update_folds("memprobe",
                     {"gemma4e4b_base": {"lora_r4_ce": 8},
                      "qwen35_4b_base": {"psr_ce_all": 4}},
                     path=path, repo=tmp_path)
    folds = msc.load_folds(path)
    assert folds["memprobe"] == {
        "gemma4e4b_base": {"psr_ce_all": 2, "lora_r4_ce": 8},
        "qwen35_4b_base": {"psr_ce_all": 4}}
    built = msc.build_all(folds_path=path)
    cfg = _cfg_of(built["sdfbase_gemma4e4b_base_fits.yaml"])
    assert cfg["batch_size_by_arm"] == {"psr_ce_all": 2, "lora_r4_ce": 8}
    # HPO section untouched by the memprobe merge -> sentinels stand.
    with pytest.raises(ValueError):
        float(cfg["train_kwargs"]["psr_ce_all"]["lr"])
    # Refusals: unknown section / unknown arm, and nothing written by them.
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        msc.update_folds("typo", {}, path=path, repo=tmp_path)
    with pytest.raises(SystemExit):
        msc.update_folds("memprobe", {"gemma4e4b_base": {"psr_kl_all": 1}},
                         path=path, repo=tmp_path)
    assert path.read_bytes() == before


# --------------------------------------------------------------------------
# tune_arm seam: ARM_AXES / boxes / slugs / CESoloAdapter
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tune_arm():
    return _load_module("_sdfbase_tune_arm", REPO / "scripts/tune_arm.py")


def test_arm_axes_and_seed_boxes(tune_arm):
    # 2-D lr x clip Sobol (no extra axes) -> emit-trials defaults to 16.
    assert tune_arm.ARM_AXES["psr_ce_all"] == ()
    assert tune_arm.ARM_AXES["lora_r4_ce"] == ()
    boxes = tune_arm.SEED_BOXES_BY_MODEL
    wide = tune_arm.CLIP_BOX_WIDE
    assert boxes[("psr_ce_all", "gemma4e4b_base")] == \
        {"lr": (1e-5, 2e-2), "max_grad_norm": wide}
    assert boxes[("lora_r4_ce", "gemma4e4b_base")] == \
        {"lr": (3e-6, 5e-3), "max_grad_norm": wide}
    assert boxes[("psr_ce_all", "qwen35_4b_base")] == \
        {"lr": (5e-5, 5e-2), "max_grad_norm": wide}
    assert boxes[("lora_r4_ce", "qwen35_4b_base")] == \
        {"lr": (5e-5, 5e-2), "max_grad_norm": wide}


def test_model_slugs_route_base_and_instruct_apart(tune_arm):
    # Instruct configs keep their historical slugs byte-for-byte; only
    # the bare base repo ids resolve to the sdfbase slugs.
    assert tune_arm.model_slug("google/gemma-4-E4B-it") == "gemma4e4b"
    assert tune_arm.model_slug("google/gemma-4-E4B") == "gemma4e4b_base"
    assert tune_arm.model_slug("google/gemma-4-E2B-it") == "gemma4e2b"
    assert tune_arm.model_slug("Qwen/Qwen3.5-4B-Base") == "qwen35_4b_base"
    # 2026-09-11: the bare instruct id resolves to "qwen35_4b" for
    # the instruct-direct batch-64 CE rows (IT_EB64_CE_LR_BOX); it had NO
    # slug before, and no other (arm, qwen35_4b) row exists, so the only
    # behaviour that moved is that its CE arms now find a box
    assert tune_arm.model_slug("Qwen/Qwen3.5-4B") == "qwen35_4b"
    assert tune_arm.model_slug("Qwen/Qwen3.8-27B") == "qwen38_27b"
    # the Llama lane: BOTH Nous ids resolve to the EXISTING first-match
    # slug "llama" — its CE boxes are registered under that slug, no needle
    # is inserted ahead of it, and the SI-distillation lane's resolution of
    # the shared instruct id is byte-unchanged
    assert tune_arm.MODEL_SLUGS[0] == ("llama", "llama")
    assert tune_arm.model_slug("NousResearch/Meta-Llama-3.1-8B") == "llama"
    assert tune_arm.model_slug("NousResearch/Meta-Llama-3.1-8B-Instruct") \
        == "llama"


def test_ce_adapters_registered(tune_arm):
    for arm in ("psr_ce_all", "lora_r4_ce"):
        ad = tune_arm.make_adapter(arm)
        assert isinstance(ad, tune_arm.CESoloAdapter)
        assert ad.kind == "solo"
        assert ad.floor_ratio_max == 1.1
        assert ad.uses_kl_probe is False
        assert ad.probe_via_trainer is True
        assert ad.supports_response is False  # positions=all only
        assert ad.consumes_teacher_topk is False  # no teacher leg at all
        assert ad.effective_batch_target == 8
        assert ad.accum_for(2) == 4 and ad.accum_for(8) == 1
    with pytest.raises(SystemExit):
        tune_arm.CESoloAdapter("psr_kl_all")
    # Legacy arms keep the KL probe (default) — no behavior change.
    assert tune_arm.make_adapter("psr_mse_all").uses_kl_probe is True
    assert tune_arm.make_adapter("lora_r4").uses_kl_probe is True


def test_ce_config_shape_guard(tune_arm):
    ad = tune_arm.make_adapter("psr_ce_all")
    ok_cfg = yaml.safe_load(
        (REPO / "configs/sdfbase_gemma4e4b_base_fits.yaml").read_text())
    ad._refuse_si_shaped_config(ok_cfg)  # the committed config passes
    for mutate in (
        lambda c: c["train_kwargs"]["psr_ce_all"].update(temperature=1.0),
        lambda c: c["train_kwargs"]["psr_ce_all"].update(
            teacher_prefix_cache=False),
        lambda c: c["train_kwargs"]["psr_ce_all"].update(
            floor_ratio_max=1.0),
        lambda c: c.update(data_mode="pairs"),
    ):
        cfg = yaml.safe_load(
            (REPO / "configs/sdfbase_gemma4e4b_base_fits.yaml").read_text())
        mutate(cfg)
        with pytest.raises(SystemExit):
            ad._refuse_si_shaped_config(cfg)


def test_ce_floor_verdict_and_objective(tune_arm):
    ad = tune_arm.make_adapter("lora_r4_ce")
    diag = {"lora_r4_ce": {"best_heldout_ce": 2.2, "heldout_noop_ce": 2.0}}
    y, label = ad.heldout_objective(diag, {})
    assert (y, label) == (2.2, "heldout_ce")
    ok, ratio, sem = ad.floor_verdict(diag, {})
    assert ok is True and ratio == pytest.approx(1.1) and sem is None
    diag["lora_r4_ce"]["best_heldout_ce"] = 2.3
    ok, ratio, _ = ad.floor_verdict(diag, {})
    assert ok is False and ratio == pytest.approx(1.15)
    # Missing curve -> no verdict, never a fabricated one.
    assert ad.floor_verdict({}, {}) == (None, None, None)


def test_ce_fingerprint_pool_extra(tune_arm, tmp_path):
    held = tmp_path / "pairs_constv2-heldout.jsonl"
    held.write_text('{"prompt": "", "pos": "x", "neg": ""}\n')
    sha = hashlib.sha256(held.read_bytes()).hexdigest()[:16]
    ad = tune_arm.make_adapter("psr_ce_all")
    cfg = {"train_kwargs": {"default": {"heldout_pool": str(held)}}}
    assert ad.fingerprint_pool_extra(None, cfg) == f"|heldout_pool:{sha}"
    cfg2 = {"train_kwargs": {"default": {}}}
    assert ad.fingerprint_pool_extra(None, cfg2) == "|heldout_pool:MISSING"
    # Legacy arms contribute nothing — fingerprints unchanged.
    assert tune_arm.make_adapter("psr_kl_all").fingerprint_pool_extra(
        None, cfg) == ""


def test_ws3_chat_lanes_inherit_their_seed_boxes(tune_arm, built):
    """emit-trials on the two chat configs resolves the `_doc` slug
    from the config's repo id and draws the INHERITED box — GLM (dense)
    from the gemma lane, Nemotron (hybrid) from the qwen lane — as the
    very same dict objects, so the numbers live in one place."""
    for name, parent in (("sdfbase_glm46vflash_doc_fits.yaml",
                          "gemma4e4b_base"),
                         ("sdfbase_nemotron3nano4b_doc_full_fits.yaml",      # the Nemotron ladder lane (`_doc`) was retired 2026-10-08; its full-scale lane keeps the box
                          "qwen35_4b_base")):
        cfg = _cfg_of(built[name])
        slug = tune_arm.model_slug(cfg["model"])
        assert slug == (_key_of(name) if not name.endswith("_doc_full_fits.yaml") else "nemotron3nano4b_doc")   # model_slug keeps the box key; the config key is the lane's
        assert tune_arm.SEED_BOX_PARENT[slug] == parent
        for arm in ("psr_ce_all", "lora_r4_ce"):
            assert tune_arm.SEED_BOXES_BY_MODEL[(arm, slug)] is \
                tune_arm.SEED_BOXES_BY_MODEL[(arm, parent)]
    # the two_si siblings' repo ids resolve to the same `_doc` slug (no
    # other lane of either checkpoint has a box), and nothing else does
    assert tune_arm.model_slug("zai-org/GLM-4.6V-Flash") == "glm46vflash_doc"
    assert tune_arm.model_slug("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16") \
        == "nemotron3nano4b_doc"
    assert tune_arm.model_slug("google/gemma-4-E4B") == "gemma4e4b_base"
    assert tune_arm.model_slug("Qwen/Qwen3.5-4B-Base") == "qwen35_4b_base"


def test_ce_emit_trials_design_is_2d(tune_arm):
    payload = tune_arm.build_trials(
        arm="psr_ce_all", model="google/gemma-4-E4B",
        model_revision="411aa17b749aa952df1359d2dcea73917a544d9a",
        config_sha="deadbeefdeadbeef", position="all", round_no=1, n=16,
        box={"lr": (1e-5, 2e-2), "max_grad_norm": (0.02, 10.0)})
    assert payload["sobol_d"] == 2
    assert payload["axes"] == ["lr", "max_grad_norm"]
    assert len(payload["trials"]) == 16
    for t in payload["trials"]:
        assert set(t) == {"index", "lr", "warmup_frac", "max_grad_norm"}
        assert 1e-5 <= t["lr"] <= 2e-2
        assert t["warmup_frac"] in (0.05, 0.15)


# --------------------------------------------------------------------------
# The Modal app (source-level pins; modal may be absent from the test env)
# --------------------------------------------------------------------------

def test_sdfbase_fits_app_pins():
    src = (REPO / "scripts/modal/sdfbase_fits.py").read_text()
    # The App binds through resolve_app_name() (--app-name >
    # $SDFBASE_APP_NAME > the frozen wave-A default), never a literal.
    assert 'app = modal.App(APP_NAME)' in src
    assert 'DEFAULT_APP_NAME = "sdfbase-fits"' in src
    assert '"qwen35_4b_base"' in src and '"gemma4e4b_base"' in src
    assert '"glm46vflash_doc"' in src
    assert '"nemotron3nano4b_doc"' in src              # the Modal lane's Nemotron key (its config is the full-scale lane's since 2026-10-08)
    # probe_only instruct entry rides the qwen image; the nemotron
    # documents key rides ITS OWN image — image_nemotron, the hybrid-stack
    # recipe plus the pinned mamba-ssm fused kernels (2026-09-02) — never
    # the qwen one, whose key table is unchanged. GLM (dense, full
    # attention) is deliberately NOT on either kernel image.
    assert ('QWEN_IMAGE_KEYS = ("qwen35_4b_base", "qwen35_4b_it", '
            '"qwen35_4b_it_doc",\n                   "qwen35_4b_it_doc_eb64",\n'
            '                   # the qwen-base batch-64 seed replicates '
            '(2026-09-15):\n'
            '                   # the same hybrid checkpoint, so the same D66 '
            'kernels\n'
            '                   "qwen35_4b_base_eb64_s1", '
            '"qwen35_4b_base_eb64_s2",\n'
            '                   "qwen35_4b_base_eb64_s3")') in src
    assert 'NEMOTRON_IMAGE_KEYS = ("nemotron3nano4b_doc",)' in src
    # the two instruct-direct keys ride their ARCHITECTURE's image
    # (qwen_it_doc on the hybrid-stack image, gemma_it_doc on std)
    assert '"gemma4e4b_it_doc"' in src and '"qwen35_4b_it_doc"' in src
    assert "mamba-ssm==2.3.2.post1" in src
    assert 'Volume.from_name("twosi-fits"' in src
    # The two base pins, exactly the plan-§0 table, + the GLM and nemotron
    # pins.
    assert "411aa17b749aa952df1359d2dcea73917a544d9a" in src
    assert "1001bb4d826a52d1f399e183466143f4da7b741b" in src
    assert "411bb4d77144a3f03accbf4b780f5acb8b7cde4e" in src
    assert "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f" in src
    # + the Llama-3.1-8B lane's two ungated Nous pins (2026-09-09)
    assert '"llama8b_base"' in src and '"llama8b_it_doc"' in src
    assert "1f47e50cdbe801ad8a5174156ec3a0655108fb9f" in src
    assert "d10aef7999a2b5ba950ab3974312feeedbfe0b77" in src
    # + the instruct-direct pins (= the r2 transfer configs' model_revision)
    assert "ee0ef6023621cff504d758262d4e04895a5af4a2" in src
    assert "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" in src
    # The qwen image stanza rode over verbatim (re-pin + assertion).
    assert "pip install --no-deps triton==3.7.1" in src
    assert "import causal_conv1d, fla" in src
    # It never touches the shared deployed app.
    assert 'Function.from_name("twosi-fits"' not in src


def test_sdfbase_fits_never_redeploys_twosi():
    import ast

    tree = ast.parse((REPO / "scripts/modal/sdfbase_fits.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names]
                     + [getattr(node, "module", "") or ""])
            assert not any("twosi_fits" in n for n in names)


# --------------------------------------------------------------------------
# GLM gated-arm REFIT lane (2026-09-07)
# --------------------------------------------------------------------------

#: The `_full` lane's psr_ce_all cells diverged at full scale under the
#: pooled r1+2 HPO fold's raw-argmin winner (r2_t10: lr 5.32e-3,
#: max_grad_norm 89.93 — effectively unclipped; the fold's quadratic was
#: not positive-definite and the objective was flat, so the fall-through
#: picked a clip-box extreme by a margin inside noise). The correction is
#: trial r2_t12, whose values are folded VERBATIM from its own record —
#: twosi-fits volume out/hpo/glm46vflash_doc/psr_ce_all_all/r2_t12/
#: summary.json, sha256 649172a8c6d0a7ca676a67478bcb17efd67302a1fd24f0b5
#: 17c8edac0942d2d5 — and pinned here BY VALUE, so a retyped digit in the
#: folds store fails this test instead of entering a fit.
R2T12_KEY = "glm46vflash_doc_full_r2t12"
R2T12_PARENT = "glm46vflash_doc_full"
R2T12_PSR = {"lr": 0.000794883034896619,
             "max_grad_norm": 0.8505392956149449,
             "warmup_frac": 0.05}


def test_glm_r2t12_lane_is_the_full_lane_with_the_r2t12_gated_optimizer(
        built):
    """The `_full` config with exactly two value lines changed (psr_ce_all
    lr and max_grad_norm; warmup_frac happens to coincide) plus the header
    — same checkpoint, scope, caps, pool dir, default block, LoRA block."""
    me, pa = msc.MODELS[R2T12_KEY], msc.MODELS[R2T12_PARENT]
    for field in ("model", "revision", "layer", "gradient_checkpointing",
                  "chat_model", "si_ids", "caps", "pool_key"):
        assert me[field] == pa[field], field
    assert "REFIT" in me["header_kind"] and "r2_t12" in me["header_kind"]
    mine_txt = built[msc.config_filename(R2T12_KEY)]
    pa_txt = built[msc.config_filename(R2T12_PARENT)]
    cfg, pcfg = _cfg_of(mine_txt), _cfg_of(pa_txt)
    assert set(cfg) == set(pcfg)
    for k in cfg:
        if k != "train_kwargs":
            assert cfg[k] == pcfg[k], k
    tk, ptk = cfg["train_kwargs"], pcfg["train_kwargs"]
    assert set(tk) == set(ptk) == {"default", "psr_ce_all", "lora_r4_ce"}
    assert tk["default"] == ptk["default"]
    assert tk["lora_r4_ce"] == ptk["lora_r4_ce"]
    psr, ppsr = dict(tk["psr_ce_all"]), dict(ptk["psr_ce_all"])
    for k, v in R2T12_PSR.items():
        assert psr.pop(k) == v, k
        ppsr.pop(k)
    assert psr == ppsr   # schedule, weight_decay, gate_reg, accum, floors
    # the runaway's clip was effectively none; r2_t12's is a real guard
    assert tk["psr_ce_all"]["max_grad_norm"] < 1.0 < \
        ptk["psr_ce_all"]["max_grad_norm"]
    # body text: exactly the two value lines differ, in the psr block
    mine = mine_txt.split("\nmodel:", 1)[1].splitlines()
    theirs = pa_txt.split("\nmodel:", 1)[1].splitlines()
    assert len(mine) == len(theirs)
    changed = [(a, b) for a, b in zip(theirs, mine) if a != b]
    assert [a.split(":")[0].strip() for a, b in changed] == \
        ["lr", "max_grad_norm"]
    assert [float(b.split(":")[1]) for a, b in changed] == \
        [R2T12_PSR["lr"], R2T12_PSR["max_grad_norm"]]
    # and the parent's bytes are the frozen ones (pinned above as well)
    assert _fingerprint_holds("sdfbase_glm46vflash_doc_full_fits.yaml", pa_txt.encode())


def test_glm_r2t12_folds_are_the_parents_except_the_gated_winner():
    folds = msc.load_folds()
    assert folds["memprobe"][R2T12_KEY] == folds["memprobe"][R2T12_PARENT]
    hpo, phpo = folds["hpo"][R2T12_KEY], folds["hpo"][R2T12_PARENT]
    assert hpo["lora_r4_ce"] == phpo["lora_r4_ce"]
    assert hpo["psr_ce_all"] == R2T12_PSR
    # the parent's (runaway) winner stays on record, untouched
    assert phpo["psr_ce_all"]["max_grad_norm"] > 80
    assert R2T12_KEY not in FULL_KEYS   # those tests assert verbatim folds


# --------------------------------------------------------------------------
# Instruct-direct BATCH-64 siblings (2026-09-11)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key,parent", sorted(IT_EB64_PARENT_OF.items()))
def test_it_eb64_lane_is_the_it_doc_checkpoint_at_effective_batch_64(
        built, key, parent):
    """Each `_eb64` key is its frozen `_it_doc` parent's checkpoint (same
    model / revision / layer / gradient-checkpointing / leading-token
    policy) re-fit at effective batch 64 in a SIBLING config: the parent's
    bytes never move (its four published cells' fit identity), the pools
    are the parent's dir (pool_key), the accumulation resolves against 64
    (sentinel spelled EFF64 until the memprobe fold), and the HPO fold is
    NOT the parent's batch-8 winners — a batch change invalidates an
    inherited learning rate, so the seams stay sentinels until this lane's
    own search folds."""
    spec, par = msc.MODELS[key], msc.MODELS[parent]
    for f in ("model", "revision", "layer", "gradient_checkpointing",
              "leading_special_tokens", "chat_model"):
        assert spec[f] == par[f], f
    assert spec["pool_key"] == parent and "pool_key" not in par
    assert msc.pool_key_for(key) == parent
    assert msc.effective_batch_for(key) == 64
    assert msc.effective_batch_for(parent) == 8          # the parent is untouched
    cfg = _cfg_of(built[f"sdfbase_{key}_fits.yaml"])
    pcfg = _cfg_of(built[f"sdfbase_{parent}_fits.yaml"])
    assert (cfg["model"], cfg["model_revision"], cfg["layer"]) == \
        (pcfg["model"], pcfg["model_revision"], pcfg["layer"])
    assert cfg["leading_special_tokens"] == pcfg["leading_special_tokens"]
    assert cfg["train_kwargs"]["default"]["heldout_pool"] == \
        pcfg["train_kwargs"]["default"]["heldout_pool"]
    folds = msc.load_folds()
    text = built[f"sdfbase_{key}_fits.yaml"]
    if folds["memprobe"].get(key):     # header names the KEY's effective batch
        assert "accum_steps = ceil(64 / bs) — effective batch 64" in text
    else:                              # unfolded: the EFF64 sentinel, never EFF8
        assert "PENDING_MEMPROBE_EFF64_ACCUM" in text
        assert "EFF8" not in text
    for arm in ("psr_ce_all", "lora_r4_ce"):
        tk = cfg["train_kwargs"][arm]
        if folds["memprobe"].get(key, {}).get(arm) is None:
            assert tk["accum_steps"] == "PENDING_MEMPROBE_EFF64_ACCUM"
        else:
            bs = int(cfg["batch_size_by_arm"][arm])
            assert int(tk["accum_steps"]) == -(-64 // bs)
        hpo = folds.get("hpo", {}).get(key, {}).get(arm)
        phpo = folds["hpo"][parent][arm]
        if hpo is None:
            for hk in ("lr", "warmup_frac", "max_grad_norm"):
                with pytest.raises(ValueError):
                    float(tk[hk])
        else:
            # a folded winner is this lane's own, never the parent's
            # batch-8 learning rate copied across
            assert float(hpo["lr"]) != pytest.approx(float(phpo["lr"]))
    # the parent's committed bytes are exactly what the generator emits
    assert (REPO / "configs" / f"sdfbase_{parent}_fits.yaml").read_text() == \
        built[f"sdfbase_{parent}_fits.yaml"]


# --------------------------------------------------------------------------
# Qwen-BASE batch-64 SEED REPLICATES (2026-09-15)
# --------------------------------------------------------------------------

def test_seedrep_table_is_the_plan_s3_declaration():
    """The generator's SEEDREP_* block is the plan-§3 declaration, spelled
    once: three seeds, effective batch 64, s1's five-cell scope vs the
    two-cell production scope, the caps rows, the key naming."""
    assert msc.SEEDREP_PARENT == SEEDREP_PARENT == "qwen35_4b_base"
    assert msc.SEEDREP_SEEDS == (1, 2, 3)
    assert msc.SEEDREP_EFFECTIVE_BATCH == 64
    assert msc.SEEDREP_KEYS == SEEDREP_KEYS
    assert [msc.seedrep_key(s) for s in (1, 2, 3)] == list(SEEDREP_KEYS)
    assert msc.SEEDREP_SI_IDS_S1 == EXPECTED_SEEDREP_SI_IDS_S1
    assert msc.SEEDREP_SI_IDS == EXPECTED_SEEDREP_SI_IDS
    assert msc.SEEDREP_CAPS == EXPECTED_SEEDREP_CAPS
    for key in SEEDREP_KEYS:
        assert msc.config_filename(key) in FILES
        assert KEY_OF[msc.config_filename(key)] == key
        assert msc.MODELS[key]["si_ids"] is (
            msc.SEEDREP_SI_IDS_S1 if SEEDREP_SEED_OF[key] == 1
            else msc.SEEDREP_SI_IDS)
    # the shared CAPS table did not move (the frozen configs' rows)
    assert msc.CAPS["constv2-mix-full"] == {"epochs": 4,
                                            "early_stop_patience": 1}


@pytest.mark.parametrize("key", SEEDREP_KEYS)
def test_seedrep_key_is_the_base_pin_at_batch_64_with_its_own_seed_and_pool(
        built, key):
    """Each seed-replicate key pins EXACTLY the frozen qwen35_4b_base
    checkpoint (model / revision / layer / gradient checkpointing), renders
    `train_kwargs.default.seed: <s>` (the one new key, last in the default
    block), accumulates to 64 on both arms from the folded micro-batch 2,
    evaluates on ITS OWN pool dir (/vol/pool/<key>/ — no pool_key, the
    carve is per seed), carries the folded LR-search winners (the seed-1
    search of 2026-09-16, one winner per arm under all three keys — the
    spec's one-search-per-arm rule; until that fold the seams were PENDING
    sentinels that float()-crashed), says what it is in the header — and
    leaves the parent's bytes exactly where they were (FROZEN_CONFIG_SHA16)."""
    s = SEEDREP_SEED_OF[key]
    spec, par = msc.MODELS[key], msc.MODELS[SEEDREP_PARENT]
    for f in ("model", "revision", "layer", "gradient_checkpointing"):
        assert spec[f] == par[f], f
    assert spec["model"] == "Qwen/Qwen3.5-4B-Base"
    assert spec["revision"] == "1001bb4d826a52d1f399e183466143f4da7b741b"
    assert spec["seed"] == s and msc.seed_for(key) == s
    assert "pool_key" not in spec and msc.pool_key_for(key) == key
    assert "chat_model" not in spec              # a base checkpoint
    assert "leading_special_tokens" not in spec  # as the parent: undeclared
    assert msc.effective_batch_for(key) == 64
    assert msc.effective_batch_for(SEEDREP_PARENT) == 8   # untouched
    assert f"SEED REPLICATE s{s} of the qwen35_4b_base lane" in \
        spec["header_kind"]
    text = built[msc.config_filename(key)]
    cfg = _cfg_of(text)
    d = cfg["train_kwargs"]["default"]
    assert d["seed"] == s and isinstance(d["seed"], int)
    assert list(d) == ["holdout_frac", "heldout_pool", "early_stop_min_epochs",
                       "early_stop_min_delta", "positions", "probe_n", "seed"]
    assert d["heldout_pool"] == \
        f"/vol/pool/{key}/pairs_constv2-heldout.jsonl"
    # the spec's default micro-batch 2 / accum 32 on BOTH arms (the
    # memprobe confirms or corrects through --fold-memprobe)
    assert cfg["batch_size_by_arm"] == {"psr_ce_all": 2, "lora_r4_ce": 2}
    for arm in ("psr_ce_all", "lora_r4_ce"):
        tk = cfg["train_kwargs"][arm]
        assert tk["accum_steps"] == 32
        assert tk["gradient_checkpointing"] is True
        # the fold (2026-09-16): the seed-1 search's winner per arm, the
        # SAME values under every seed key (folds store == rendered config)
        folded = msc.load_folds()["hpo"][key][arm]
        for hk in ("lr", "warmup_frac", "max_grad_norm"):
            assert isinstance(tk[hk], float) and tk[hk] > 0, (key, arm, hk)
            assert tk[hk] == folded[hk], (key, arm, hk)
        assert {hk: tk[hk] for hk in ("lr", "warmup_frac", "max_grad_norm")} == \
            SEEDREP_HPO_WINNERS[arm], (key, arm)
        assert "seed" not in tk                   # the default block only
    head = text.split("\nmodel:", 1)[0]
    assert "#   accum_steps = ceil(64 / bs) — effective batch 64." in head
    assert "effective batch 8" not in head and "EFF8" not in text
    assert f"# seed: {s} (train_kwargs.default.seed)" in head
    assert (f"(BASE model, batch-64 SEED REPLICATE s{s} of the qwen35_4b_base "
            "lane") in head
    # identical to the frozen parent config in everything but identity of
    # the lane: scope, caps, the default block's pool + seed, the optimizer
    # seams (the parent's are folded winners) and the accumulation width
    pcfg = _cfg_of(built[msc.config_filename(SEEDREP_PARENT)])
    for f in ("model", "model_revision", "layer", "dtype",
              "attn_implementation", "max_len", "data_mode",
              "training_budget_kind", "sequence_max_len",
              "completion_tokens", "teacher_prefix_cache",
              "wandb_project", "wandb_fits"):
        assert cfg[f] == pcfg[f], f
    assert "seed" not in pcfg["train_kwargs"]["default"]
    assert {k: v for k, v in d.items() if k not in ("heldout_pool", "seed")} \
        == {k: v for k, v in pcfg["train_kwargs"]["default"].items()
            if k != "heldout_pool"}
    assert text != built[msc.config_filename(SEEDREP_PARENT)]
    # the parent's committed bytes: exactly the generator's, at the pin
    parent_txt = (REPO / "configs" /
                  msc.config_filename(SEEDREP_PARENT)).read_text()
    assert parent_txt == built[msc.config_filename(SEEDREP_PARENT)]
    assert _fingerprint_holds("sdfbase_qwen35_4b_base_fits.yaml", parent_txt.encode())


def _seed_normalised(text: str) -> str:
    """A seed-replicate config's text with its seed spelled N everywhere
    the seed appears (key name, header, pool dir, the seed value)."""
    text = re.sub(r"eb64_s[123]", "eb64_sN", text)
    text = re.sub(r"REPLICATE s[123] ", "REPLICATE sN ", text)
    return re.sub(r"seed: [123]\b", "seed: N", text)


def test_seedrep_configs_differ_only_in_seed_scope_and_pool(built):
    """s2 and s3 are the same config up to the seed; s1 is that config plus
    the three working cells (scope + caps rows), again up to the seed —
    nothing else (an optimizer field, a cap, a flag) may differ between
    replicates, or the triplet would not be a replicate."""
    t1, t2, t3 = (built[msc.config_filename(k)] for k in SEEDREP_KEYS)
    assert _seed_normalised(t2) == _seed_normalised(t3)
    assert _seed_normalised(t1) != _seed_normalised(t2)   # the working cells
    c1, c2 = _cfg_of(t1), _cfg_of(t2)
    assert c1["si_ids"] == EXPECTED_SEEDREP_SI_IDS_S1
    assert c2["si_ids"] == EXPECTED_SEEDREP_SI_IDS
    c1["si_ids"] = [si for si in c1["si_ids"] if si in c2["si_ids"]]
    c1["train_kwargs_by_si"] = {si: row for si, row in
                                c1["train_kwargs_by_si"].items()
                                if si in c2["si_ids"]}
    for c in (c1, c2):
        c["train_kwargs"]["default"]["seed"] = "N"
        c["train_kwargs"]["default"]["heldout_pool"] = "N"
    assert c1 == c2
    # and the three files hash apart (three distinct expect_shas)
    assert len({hashlib.sha256(t.encode()).hexdigest() for t in (t1, t2, t3)}) == 3


def test_seed_key_is_declared_and_emitted_on_exactly_the_seedrep_lanes(built):
    """`seed` is a NEW MODELS field emitted ONLY when declared: the three
    replicate configs carry `train_kwargs.default.seed` and the header
    line, and so does the demo lane (demo-import: the worked example fits
    at effective batch 64 under seed 3, the committed objects' recipe (the
    one-seed objects/DEMO.json, 2026-10-12), through these same fields —
    tests/test_models_demo_front_door.py pins the rest); EVERY other config
    — the frozen ones included — has no `seed` anywhere and no declaration
    (their fits keep the trainer's default 0, bytes untouched)."""
    seeded = {**SEEDREP_SEED_OF, "qwen35_4b_demo_base": 3}
    declared = {k for k, m in msc.MODELS.items() if "seed" in m}
    assert declared == set(seeded) == set(SEEDREP_KEYS) | set(PIPELINE_TEST_KEYS)
    for name in FILES:
        key = _key_of(name)
        cfg, text = _cfg_of(built[name]), built[name]
        blocks = cfg["train_kwargs"]
        if key in seeded:
            assert blocks["default"]["seed"] == seeded[key]
            assert msc.seed_for(key) == seeded[key]
            assert "# seed:" in text
        else:
            assert msc.seed_for(key) is None
            assert "seed" not in blocks["default"]
            assert "# seed:" not in text
            assert not re.search(r"^\s*seed:", text, re.M), name
        for arm in ("psr_ce_all", "lora_r4_ce"):
            assert "seed" not in blocks[arm]


def test_seed_declaration_is_validated_and_undeclared_emits_nothing():
    """A declared seed must be a non-negative int (a bool, a string, a
    float or a negative refuse loudly at build time); removing the
    declaration removes the key from the rendered default block — the
    frozen behaviour, by construction."""
    key = "qwen35_4b_base_eb64_s1"
    saved = msc.MODELS[key]["seed"]
    try:
        for bad in ("1", 1.0, True, -1, None):
            msc.MODELS[key]["seed"] = bad
            with pytest.raises(SystemExit, match="seed="):
                msc.build_config(key)
        del msc.MODELS[key]["seed"]
        assert msc.seed_for(key) is None
        cfg = msc.build_config(key)
        assert "seed" not in cfg["train_kwargs"]["default"]
        assert "# seed:" not in msc.render_config(key, cfg)
        msc.MODELS[key]["seed"] = 0                 # 0 is a legal declaration
        assert msc.build_config(key)["train_kwargs"]["default"]["seed"] == 0
    finally:
        msc.MODELS[key]["seed"] = saved
    assert msc.build_config(key)["train_kwargs"]["default"]["seed"] == saved


#: The folded winners (the seed-1 batch-64 search on constv2-mix-10m,
#: 2026-09-16; $STAGE/sdfbase/reseed_eb64/hpo/): psr = the fitted
#: argmin CONFIRMED in_interval (d 0.52 sigma); lora = the raw best trial
#: r1_t07 ADOPTED under ruling O8 (its fitted argmin was ALARMED
#: sampled_extreme:lr) and replicated in_interval (d -0.12 sigma). The SAME
#: dict under every seed key — the one-search-per-arm rule.
SEEDREP_HPO_WINNERS = {
    "psr_ce_all": {"lr": 0.004578836147882968, "warmup_frac": 0.15,
                   "max_grad_norm": 0.34},
    "lora_r4_ce": {"lr": 0.006681234631156978, "warmup_frac": 0.15,
                   "max_grad_norm": 0.12},
}


def test_seedrep_memprobe_fold_is_2_2_on_all_three_and_hpo_folded_identically():
    """The committed folds store carries the default micro-batch 2
    for BOTH arms under all three keys (--fold-memprobe; the memprobe
    measured psr 2 / lora 4 and folded min(measured, 2)) and, since the
    fold of 2026-09-16, the seed-1 search's one winner per arm under all
    three keys IDENTICALLY (SEEDREP_HPO_WINNERS); lora's entry carries an
    ALARMED selection guard acknowledged by ruling O8 (never silently)."""
    folds = msc.load_folds()
    for key in SEEDREP_KEYS:
        assert folds["memprobe"][key] == {"lora_r4_ce": 2, "psr_ce_all": 2}
        assert set(folds["hpo"][key]) == {"psr_ce_all", "lora_r4_ce"}
        for arm, want in SEEDREP_HPO_WINNERS.items():
            got = folds["hpo"][key][arm]
            assert {k: got[k] for k in want} == want, (key, arm)
            g = got["selection_guard"]
            assert g["selected"] == {"lr": want["lr"], "max_grad_norm": want["max_grad_norm"], "warmup_frac": want["warmup_frac"]}, (key, arm)
            if arm == "lora_r4_ce":
                assert g["alarmed"] is True and g["selected_from"] == "raw_best_trial"
                assert g["acknowledged"] and "O8" in g["acknowledged"]["by"]
            else:
                assert g["alarmed"] is False and g["selected_from"] == "fitted_argmin"
    # the parent's folds are untouched (its bytes are pinned regardless)
    assert folds["memprobe"][SEEDREP_PARENT]["psr_ce_all"] == 2
    assert set(folds["hpo"][SEEDREP_PARENT]) == {"psr_ce_all", "lora_r4_ce"}
