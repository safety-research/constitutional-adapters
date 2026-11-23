"""The sikl4 refit configs and task table (scripts/sikl4/make_fit_configs.py).

- every committed config and the task table equal a regeneration (``--check`` exits 0, and 1 on a stale file);
- each fit config equals the source (configs/two_si_qwen35_4b_fits.yaml) in every key except the declared deltas;
- the per-kind hyperparameters are the published recipe literally at learning-rate step 0, and x 2^(-1/2) / x 2^(-1) at steps 1 / 2;
- configs differ across seeds only in ``seed``, and across learning-rate steps only in the two learning rates;
- the smoke config is the seed-1 step-0 config with one epoch;
- the task table: 54 fit rows + 3 teacher rows + 3 smoke rows, a unique row id first (the job name), and every row's config,
  pool, teacher, output root and checkpoint cadence agree with its (mode, SI, kind, seed, step);
- the configs load through the trainer's own reader and pass its registry-statement check.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "sikl4" / "make_fit_configs.py"
_spec = importlib.util.spec_from_file_location("sikl4_make_fit_configs", GEN)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

SOURCE = yaml.safe_load((REPO / M.SOURCE).read_text())
LORA_LR = 0.00219324600173211
RELU_LR = 0.0021912898676553186


def _cfg(name: str) -> dict:
    return yaml.safe_load((REPO / name).read_text())


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict) and v and k not in ("registry_prompt_pool_ids",):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _tasks() -> list[dict]:
    lines = (REPO / M.TASKS).read_text().splitlines()
    header = lines[0].split("\t")
    # the fit-ops tools' column contract (row id first; the job name)
    assert tuple(header) == M.TASK_COLUMNS == ("row_id", "mode", "si", "arm", "seed", "lr_step", "config", "pool", "teacher",
                                               "out_root", "checkpoint_every")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def test_committed_files_equal_a_regeneration_and_check_catches_a_stale_one(tmp_path):
    r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    files = M.build_all()
    assert set(files) == {M.config_name(s, n) for s in M.SEEDS for n in M.LR_STEPS} | {M.SMOKE_CONFIG, M.TASKS}
    for name, text in files.items():
        assert (REPO / name).read_text() == text, name
    # a stale file fails --check (exercised on a copy of the generator rooted at a scratch tree)
    scratch = tmp_path / "repo"
    (scratch / "scripts" / "sikl4").mkdir(parents=True)
    (scratch / "configs" / "sikl4").mkdir(parents=True)
    (scratch / "scripts" / "sikl4" / "make_fit_configs.py").write_text(GEN.read_text())
    (scratch / M.SOURCE).write_text((REPO / M.SOURCE).read_text())
    gen = [sys.executable, str(scratch / "scripts" / "sikl4" / "make_fit_configs.py")]
    assert subprocess.run(gen, capture_output=True, text=True).returncode == 0
    assert subprocess.run(gen + ["--check"], capture_output=True, text=True).returncode == 0
    p = scratch / M.config_name(2, 1)
    p.write_text(p.read_text() + "# hand edit\n")
    bad = subprocess.run(gen + ["--check"], capture_output=True, text=True)
    assert bad.returncode == 1 and "STALE" in bad.stderr and M.config_name(2, 1) in bad.stderr


@pytest.mark.parametrize("seed", M.SEEDS)
@pytest.mark.parametrize("step", M.LR_STEPS)
def test_each_fit_config_is_the_source_plus_exactly_the_declared_deltas(seed, step):
    cfg = _cfg(M.config_name(seed, step))
    src, got = _flat(SOURCE), _flat(cfg)
    changed = {k for k in set(src) | set(got) if src.get(k, object()) != got.get(k, object())}
    # step 0 IS the source's learning rates (x 2^0), so the two lr keys differ only at steps 1 and 2
    lr_keys = {"train_kwargs.lora_r4.lr", "train_kwargs.psr_kl_all.lr"}
    want = set(M.DELTA_KEYS) - (lr_keys if step == 0 else set())
    assert changed == want, sorted(changed ^ want)
    assert cfg["si_ids"] == ["safety_refusal", "si_constitution", "C16"]
    assert cfg["max_len_by_si"] == {**SOURCE["max_len_by_si"], "C16": 8192}
    assert cfg["max_len_by_si"]["safety_refusal"] == 8192 and cfg["max_len_by_si"]["si_constitution"] == 16384
    d = cfg["train_kwargs"]["default"]
    assert d["positions"] == "all" and d["seed"] == seed and d["epochs"] == 10
    assert cfg["registry_paper_label"] == "other"
    assert cfg["registry_prompt_pool_ids"] == {si: f"qwen35_4b/pairs_{si}" for si in ("safety_refusal", "si_constitution", "C16")}
    assert cfg["wandb_project"] == SOURCE["wandb_project"]          # the job environment names the W&B project


@pytest.mark.parametrize("step,factor", [(0, 1.0), (1, 2 ** -0.5), (2, 0.5)])
def test_the_per_kind_hyperparameters_are_the_published_recipe_at_each_step(step, factor):
    for seed in M.SEEDS:
        tk = _cfg(M.config_name(seed, step))["train_kwargs"]
        lora, relu = tk["lora_r4"], tk["psr_kl_all"]
        if step == 0:
            assert lora["lr"] == LORA_LR and relu["lr"] == RELU_LR          # the published literals, bit for bit
        assert math.isclose(lora["lr"], LORA_LR * factor, rel_tol=1e-15)
        assert math.isclose(relu["lr"], RELU_LR * factor, rel_tol=1e-15)
        assert lora["max_grad_norm"] == 0.24882679700493568 and relu["max_grad_norm"] == 0.19294010750831866
        assert lora["lora_rank"] == 4 and lora["accum_steps"] == relu["accum_steps"] == 32
        assert lora["warmup_frac"] == relu["warmup_frac"] == 0.15
        assert lora["lr_schedule"] == relu["lr_schedule"] == "warmup_cosine"
        assert lora["weight_decay"] == relu["weight_decay"] == 0.01 and lora["temperature"] == relu["temperature"] == 1.0
        assert tk["default"]["early_stop_patience"] == 3 and tk["default"]["probe_n"] == 256
        assert tk["default"]["floor_ratio_max"] == 1.0 and tk["default"]["holdout_salt"] == "two_si_holdout"
    assert math.isclose(_cfg(M.config_name(1, 1))["train_kwargs"]["lora_r4"]["lr"], 1.5508e-3, rel_tol=1e-4)
    assert math.isclose(_cfg(M.config_name(1, 2))["train_kwargs"]["psr_kl_all"]["lr"], 1.0956e-3, rel_tol=1e-4)


def test_configs_differ_across_seeds_only_in_seed_and_across_steps_only_in_the_two_lrs():
    for step in M.LR_STEPS:
        base = _flat(_cfg(M.config_name(1, step)))
        for seed in M.SEEDS[1:]:
            other = _flat(_cfg(M.config_name(seed, step)))
            assert {k for k in base if base[k] != other[k]} == {"train_kwargs.default.seed"}
    for seed in M.SEEDS:
        base = _flat(_cfg(M.config_name(seed, 0)))
        for step in M.LR_STEPS[1:]:
            other = _flat(_cfg(M.config_name(seed, step)))
            assert {k for k in base if base[k] != other[k]} == {"train_kwargs.lora_r4.lr", "train_kwargs.psr_kl_all.lr"}


def test_the_smoke_config_is_seed_1_step_0_with_one_epoch():
    smoke, s1 = _flat(_cfg(M.SMOKE_CONFIG)), _flat(_cfg(M.config_name(1, 0)))
    assert {k for k in s1 if s1[k] != smoke[k]} == {"train_kwargs.default.epochs"}
    assert smoke["train_kwargs.default.epochs"] == 1 and smoke["train_kwargs.default.seed"] == 1
    assert set(M.SMOKE_DELTA_KEYS) - set(M.DELTA_KEYS) == {"train_kwargs.default.epochs"}


def test_the_task_table_rows_agree_with_their_identity():
    rows = _tasks()
    ids = [r["row_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 60
    by_mode = {m: [r for r in rows if r["mode"] == m] for m in ("fit", "teacher", "smoke")}
    assert (len(by_mode["fit"]), len(by_mode["teacher"]), len(by_mode["smoke"])) == (54, 3, 3)
    assert sum(len(v) for v in by_mode.values()) == len(rows)
    for r in rows:
        assert all(r[c] for c in M.TASK_COLUMNS)
        assert not any(r[c].startswith("/") for c in ("config", "pool", "teacher", "out_root")), r   # all paths relative
        assert (REPO / r["config"]).is_file(), r["row_id"]
        si = r["si"]
        assert r["pool"] == f"exp/sikl4/pools/qwen35_4b/pairs_{si}.jsonl"
        assert r["teacher"] == r["pool"] + ".teachertopk.safetensors"
    seen = set()
    for r in by_mode["fit"]:
        si, arm, k, n = r["si"], r["arm"], int(r["seed"]), int(r["lr_step"])
        seen.add((si, arm, k, n))
        assert r["row_id"] == (f"sikl-fit-{si}-{arm}-s{k}" + (f"-lr{n}" if n else ""))
        assert r["config"] == f"configs/sikl4_qwen35_4b_fits_s{k}_lr{n}.yaml"
        assert r["out_root"] == f"exp/objects/sikl4/qwen35_4b/{'fits' if n == 0 else f'fits_lr{n}'}/s{k}/{si}__{arm}"
        assert r["checkpoint_every"] == "100"
    assert seen == {(si, arm, k, n) for si in M.SIS for arm in M.ARMS for k in M.SEEDS for n in M.LR_STEPS}
    assert len({r["out_root"] for r in by_mode["fit"]}) == 54                  # no two fit jobs share a store
    for r in by_mode["teacher"]:
        assert r["row_id"] == f"sikl-teacher-{r['si']}" and r["arm"] == r["seed"] == r["lr_step"] == r["out_root"] == "-"
        assert r["config"] == M.config_name(1, 0) and r["checkpoint_every"] == "-"
    assert {r["si"] for r in by_mode["teacher"]} == set(M.SIS)
    for a, r in zip((1, 2, 3), by_mode["smoke"]):
        assert r["row_id"] == f"sikl-fit-smoke-a{a}" and r["out_root"] == f"exp/sikl4/smoke/a{a}"
        assert (r["si"], r["arm"], r["seed"], r["lr_step"]) == ("safety_refusal", "lora_r4", "1", "0")
        assert r["config"] == M.SMOKE_CONFIG and r["checkpoint_every"] == "10"
    # the smoke and production s1 safety-LoRA rows never collide: distinct ids and distinct output roots
    s1 = next(r for r in by_mode["fit"] if r["row_id"] == "sikl-fit-safety_refusal-lora_r4-s1")
    assert all(r["out_root"] != s1["out_root"] for r in by_mode["smoke"])


def test_the_configs_load_through_the_trainer_and_pass_its_statement_check():
    sys.path.insert(0, str(REPO / "scripts"))
    import train_two_si_objects as t2

    from sisteer.eval.train_budget import resolve_max_len

    for name in [M.config_name(s, n) for s in M.SEEDS for n in M.LR_STEPS] + [M.SMOKE_CONFIG]:
        cfg = t2._load_cfg(name)
        assert t2._config_sha(name) == hashlib.sha256((REPO / name).read_bytes()).hexdigest()[:16]
        for si in M.SIS:
            t2.check_registry_statement_cfg(cfg, si)
            assert resolve_max_len(cfg, si) == {"safety_refusal": 8192, "si_constitution": 16384, "C16": 8192}[si]
        assert t2._train_kwargs_for_si(cfg, "C16")["positions"] == "all"
    # the source itself states no registry statement (nothing changes for its fits)
    assert not any(k in SOURCE for k in ("registry_paper_label", "registry_prompt_pool_ids"))
