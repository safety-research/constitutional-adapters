"""memprobe_fold.py (memprobe-fix-2; fits 831981 / 831986 OOM at step 25): the memprobe record is keyed to what it MEASURED (arm, model, rank, module
set, the explicit per_expert_rank map, seq_len, and — memprobe-fix-3 — the pinned revision, gradient checkpointing and dtype by the trainer's
rule; omission of any keyed field is STALE; init recorded, not keyed), refused by name on another card
class, and folded as the largest rung within headroom. The stale record has the shape that killed two fits on 2026-10-02 (the paper's rank-4
o_proj probe, rung 2 at 91 % of the card) held against a config that selects a rank-16 MLP set with an explicit per-expert map — a SYNTHETIC
fits config under a synthetic MoE fit key (no model in the table carries that shape), in the generated shape make_sdfbase_configs writes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PICK = {"lora_rank": 16, "lora_modules": ["down_proj", "gate_proj", "up_proj"], "per_expert_rank": {"shared_expert": 16, "routed_expert": 2}}
FIT_KEY, SYNTH_MODEL, SYNTH_REV = "synth_moe_base", "synth-org/synth-moe-base", "0123456789abcdef0123456789abcdef01234567"
INIT = "random_lecun_1e-3"


def _cfg() -> dict:
    """The synthetic fit key's fits config in the generated shape: the picked selection at the TOP LEVEL of the LoRA arm block (what the
    trainer gets) with the fit_selection block beneath it, a pinned revision, the fit's runtime (checkpointing per arm, bfloat16)."""
    lora = {"lr": 0.001, "lora_rank": PICK["lora_rank"], "lora_modules": list(PICK["lora_modules"]), "per_expert_rank": dict(PICK["per_expert_rank"]),
            "init_scheme": INIT, "gradient_checkpointing": True,
            "fit_selection": {"fit_search_id": "search:synthetic", "lora_rank": PICK["lora_rank"], "lora_modules": list(PICK["lora_modules"]),
                              "per_expert_rank": dict(PICK["per_expert_rank"]), "fit_lr_selected_by": "search", "init_scheme": INIT}}
    psr = {"lr": 0.001, "init_scheme": INIT, "gradient_checkpointing": True, "fit_selection": {"fit_search_id": "search:synthetic", "init_scheme": INIT}}
    return {"model": SYNTH_MODEL, "model_revision": SYNTH_REV, "dtype": "bfloat16", "max_len": 8192, "data_mode": "documents",
            "batch_size_by_arm": {"psr_ce_all": "MEMPROBE", "lora_r4_ce": "MEMPROBE"},
            "train_kwargs": {"default": {"holdout_frac": 0.0, "positions": "all"}, "lora_r4_ce": lora, "psr_ce_all": psr}}


def _mod():
    spec = importlib.util.spec_from_file_location("memprobe_fold", REPO / "scripts" / "sdfbase" / "memprobe_fold.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _stale_record() -> dict:
    """The 2026-09-30 record: the paper's object at seq_len 8192 on an H200 — rung 1 69 %, rung 2 91 % (fits), rung 4 OOM; no per_expert_rank key."""
    return {"arm": "lora_r4_ce", "schema": 1, "seq_len": 8192, "steps": 3, "gpu_total_bytes": 150_120_759_296, "gpu_name": "NVIDIA H200",
            "gradient_checkpointing": True, "dtype": "bfloat16",
            "model": {"model": SYNTH_MODEL, "revision": SYNTH_REV, "model_override": False},
            "object_kwargs": {"init_scheme": "zero", "lora_modules": ["o_proj"], "lora_rank": 4},
            "rungs": [{"micro_batch": 1, "fits": True, "peak_alloc_bytes": 103_347_794_432, "s_per_step": 49.2, "error": None},
                      {"micro_batch": 2, "fits": True, "peak_alloc_bytes": 137_304_196_608, "s_per_step": 13.1, "error": None},
                      {"micro_batch": 4, "fits": False, "peak_alloc_bytes": 134_481_610_240, "s_per_step": None, "error": "OutOfMemoryError: …"}],
            "verdict": {"exit_code": 0, "fits_at_min_micro_batch": True, "largest_fitting_micro_batch": 2, "min_micro_batch": 1,
                        "replan_required": True, "s_per_step_at_min": 49.2}}


def _fresh_record() -> dict:
    """What the re-probe writes for the picked selection: the full identity, rung 1 within headroom, rung 2 above it."""
    r = _stale_record()
    r["object_kwargs"] = {"init_scheme": "random_lecun_1e-3", "lora_modules": PICK["lora_modules"], "lora_rank": 16, "per_expert_rank": PICK["per_expert_rank"]}
    return r


def _run(tmp_path, rec: dict, cfg: dict, arm="lora_r4_ce", **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    m = _mod(); r = tmp_path / "rec.json"; c = tmp_path / "cfg.yaml"; out = tmp_path / "fold.json"
    r.write_text(json.dumps(rec)); c.write_text(yaml.safe_dump(cfg))
    argv = [str(r), "--config", str(c), "--fit-key", FIT_KEY, "--arm", arm, "--out", str(out)]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    rc = m.main(argv)
    return rc, (json.loads(out.read_text()) if out.exists() else None)


def test_the_selection_source_is_the_trainers_precedence_and_the_fingerprint_follows_it():
    """config_selection reads the TOP-LEVEL arm kwargs first (what the trainer gets), fit_selection filling, defaults last; the unit
    fingerprint derives from the same tuple (+ the config's max_len)."""
    m = _mod(); cfg = _cfg()
    assert m.config_selection(cfg, "lora_r4_ce") == {**PICK, "init_scheme": INIT}
    assert m.config_selection(cfg, "psr_ce_all") == {"lora_rank": None, "lora_modules": None, "per_expert_rank": None, "init_scheme": INIT}
    tag = m.model_tag(cfg); assert len(tag) == 8
    assert m.fingerprint(cfg, "lora_r4_ce") == f"r16-down-gate-up-pe16x2-L8192-gc1-bf16-m{tag}" and m.fingerprint(cfg, "psr_ce_all") == f"psr-L8192-gc1-bf16-m{tag}"
    # a config whose fit_selection CONTRADICTS the top-level block: the top-level (trainer) shape wins on both sides of the fold
    contra = _cfg(); tk = contra["train_kwargs"]["lora_r4_ce"]
    tk["fit_selection"] = {**tk["fit_selection"], "lora_rank": 4, "lora_modules": ["o_proj"], "per_expert_rank": None}
    assert m.config_selection(contra, "lora_r4_ce")["lora_rank"] == 16 and m.fingerprint(contra, "lora_r4_ce") == m.fingerprint(cfg, "lora_r4_ce")
    paper = _cfg(); tk = paper["train_kwargs"]["lora_r4_ce"]
    tk["lora_rank"] = 4; tk["lora_modules"] = ["o_proj"]; tk["per_expert_rank"] = None
    assert m.fingerprint(paper, "lora_r4_ce") == f"r4-o-pe-none-L8192-gc1-bf16-m{tag}"


def test_the_real_stale_record_refuses_by_name_against_the_picked_selection_including_the_omitted_map(tmp_path, capsys):
    cfg = _cfg(); m = _mod()
    rc, fold = _run(tmp_path, _stale_record(), cfg)
    out = capsys.readouterr().out
    assert rc == m.EXIT_STALE and fold is None
    assert "lora_rank: record probed 4, the config selects 16" in out and "record probed ['o_proj'], the config selects ['down_proj', 'gate_proj', 'up_proj']" in out
    pe = out.split("per_expert_rank: the record omits it (it predates the keyed identity) — the config selects ", 1)
    assert len(pe) == 2 and "'shared_expert': 16" in pe[1].splitlines()[0] and "'routed_expert': 2" in pe[1].splitlines()[0], out
    assert "init_scheme" not in out.split("STALE", 1)[1].split(" — not folded")[0], "init is recorded, never a staleness reason"


def test_a_fresh_record_folds_within_headroom_and_every_keyed_omission_or_drift_is_stale(tmp_path, capsys):
    cfg = _cfg(); m = _mod()
    rc, fold = _run(tmp_path / "ok", _fresh_record(), cfg)
    out = capsys.readouterr().out
    assert rc == 0 and fold == {FIT_KEY: {"lora_r4_ce": 1}}, (rc, fold, out)       # rung 2 at 91.5 % is above the 85 % headroom
    assert "within_headroom(85%)=False" in out and "the record's largest fitting rung was 2" in out and "init=random_lecun_1e-3 (recorded, not keyed)" in out
    rc, fold = _run(tmp_path / "loose", _fresh_record(), cfg, headroom=0.95)                       # a looser headroom admits rung 2, explicitly
    assert rc == 0 and fold == {FIT_KEY: {"lora_r4_ce": 2}}
    rec = _fresh_record(); rec["object_kwargs"]["init_scheme"] = "zero"                           # init differs: recorded, NOT stale
    assert _run(tmp_path / "init", rec, cfg)[0] == 0
    rec = _fresh_record(); rec["seq_len"] = 4096                                                   # probed shorter than the fit trains at
    rc, _ = _run(tmp_path / "seq", rec, cfg); assert rc == m.EXIT_STALE and "seq_len: record probed 4096, the config's max_len is 8192" in capsys.readouterr().out
    rec = _fresh_record(); del rec["seq_len"]
    rc, _ = _run(tmp_path / "noseq", rec, cfg); assert rc == m.EXIT_STALE and "seq_len: the record omits it" in capsys.readouterr().out
    rec = _fresh_record(); rec["object_kwargs"]["per_expert_rank"] = {"shared_expert": 16, "routed_expert": 4}
    rc, _ = _run(tmp_path / "pe", rec, cfg); assert rc == m.EXIT_STALE and "per_expert_rank: record probed {'shared_expert': 16, 'routed_expert': 4}" in capsys.readouterr().out
    rec = _fresh_record(); rec["model"]["revision"] = "f" * 40
    rc, _ = _run(tmp_path / "rev", rec, cfg); assert rc == m.EXIT_STALE and "revision: record" in capsys.readouterr().out


def test_fix3_revision_checkpointing_and_dtype_are_keyed_and_a_drift_is_a_distinct_unit(tmp_path, capsys):
    """A record with NO revision passed a pinned config; the fingerprint (the unit id) omitted model/revision, so a revision
    drift would have re-minted the completed unit; the sbatch forced checkpointing on while the fold keyed neither it nor dtype."""
    import copy
    cfg = _cfg(); m = _mod()
    assert cfg.get("model_revision") and m.config_runtime(cfg, "lora_r4_ce") == {"gradient_checkpointing": True, "dtype": "bfloat16"}
    rec = _fresh_record(); del rec["model"]["revision"]                                            # omitted revision against a pinned config
    rc, _ = _run(tmp_path / "norev", rec, cfg); assert rc == m.EXIT_STALE and "revision: the record omits it — the config pins" in capsys.readouterr().out
    rec = _fresh_record(); rec["model"]["revision"] = None
    rc, _ = _run(tmp_path / "nullrev", rec, cfg); assert rc == m.EXIT_STALE and "revision: the record omits it" in capsys.readouterr().out
    drifted = copy.deepcopy(cfg); drifted["model_revision"] = "f" * 40                             # same selection, another revision: a NEW unit
    assert m.fingerprint(drifted, "lora_r4_ce") != m.fingerprint(cfg, "lora_r4_ce") and m.model_tag(drifted) != m.model_tag(cfg)
    assert m.fingerprint(drifted, "lora_r4_ce").startswith("r16-down-gate-up-pe16x2-L8192-gc1-bf16-m")
    assert m.staleness(_fresh_record(), drifted, "lora_r4_ce") == [f"revision: record {SYNTH_REV!r} vs config " + repr("f" * 40)]
    off = copy.deepcopy(cfg); off["train_kwargs"]["lora_r4_ce"]["gradient_checkpointing"] = False  # the fit runs WITHOUT checkpointing: an ON record priced another job
    assert m.config_runtime(off, "lora_r4_ce")["gradient_checkpointing"] is False, "per-arm over default (the trainer's rule)"
    rc, _ = _run(tmp_path / "gcoff", _fresh_record(), off); assert rc == m.EXIT_STALE and "gradient_checkpointing: record probed True, the fit runs with False" in capsys.readouterr().out
    assert m.fingerprint(off, "lora_r4_ce") == m.fingerprint(cfg, "lora_r4_ce").replace("-gc1-", "-gc0-")
    rec = _fresh_record(); rec["dtype"] = "float32"                                                 # dtype drift
    rc, _ = _run(tmp_path / "dtype", rec, cfg); assert rc == m.EXIT_STALE and "dtype: record probed 'float32', the fit runs with 'bfloat16'" in capsys.readouterr().out
    rec = _fresh_record(); del rec["gradient_checkpointing"]; del rec["dtype"]                      # omission of either is STALE
    rc, _ = _run(tmp_path / "omit", rec, cfg); out = capsys.readouterr().out
    assert rc == m.EXIT_STALE and "gradient_checkpointing: the record omits it — the fit runs with True" in out and "dtype: the record omits it — the fit runs with 'bfloat16'" in out
    rec = _fresh_record(); rec["gradient_checkpointing"] = None
    rc, _ = _run(tmp_path / "nullgc", rec, cfg); assert rc == m.EXIT_STALE and "gradient_checkpointing: the record omits it" in capsys.readouterr().out
    # the trainer's rule end to end: no per-arm key -> the default block; no default -> OFF
    bare = copy.deepcopy(cfg); bare["train_kwargs"]["lora_r4_ce"].pop("gradient_checkpointing"); bare["train_kwargs"]["default"]["gradient_checkpointing"] = False
    assert m.config_runtime(bare, "lora_r4_ce")["gradient_checkpointing"] is False
    bare["train_kwargs"]["default"].pop("gradient_checkpointing"); assert m.config_runtime(bare, "lora_r4_ce")["gradient_checkpointing"] is False
    nodt = copy.deepcopy(cfg); nodt.pop("dtype")
    with pytest.raises(SystemExit): m.config_runtime(nodt, "lora_r4_ce")
    # the fresh record under the current config still folds, its identity line naming the three fields
    rc, fold = _run(tmp_path / "ok", _fresh_record(), cfg); out = capsys.readouterr().out
    assert rc == 0 and fold == {FIT_KEY: {"lora_r4_ce": 1}} and f"gradient_checkpointing True, dtype bfloat16, revision {SYNTH_REV[:10]}" in out


def test_another_card_class_is_refused_by_name_and_nothing_within_headroom_or_a_does_not_fit_verdict_exit_5(tmp_path, capsys):
    cfg = _cfg(); m = _mod()
    rc, fold = _run(tmp_path / "hw", _fresh_record(), cfg, expect_gpu_name="NVIDIA H100")
    assert rc == m.EXIT_HARDWARE and fold is None and "HARDWARE: the record probed on 'NVIDIA H200'" in capsys.readouterr().out
    assert _run(tmp_path / "hwok", _fresh_record(), cfg, expect_gpu_name="h200")[0] == 0           # case-insensitive substring of the card name
    rec = _fresh_record(); rec["rungs"][0]["peak_alloc_bytes"] = int(0.9 * rec["gpu_total_bytes"])
    rc, fold = _run(tmp_path / "a", rec, cfg); assert rc == 5 and fold is None and "no rung fits WITH headroom" in capsys.readouterr().out
    rec = _fresh_record(); rec["verdict"].update({"exit_code": 5, "fits_at_min_micro_batch": False, "largest_fitting_micro_batch": None}); rec["rungs"][0]["fits"] = False
    rc, fold = _run(tmp_path / "b", rec, cfg); assert rc == 5 and fold is None and "DOES NOT FIT at micro-batch 1" in capsys.readouterr().out


def test_the_relu_arm_is_keyed_by_arm_model_and_seq_len_only(tmp_path, capsys):
    cfg = _cfg(); m = _mod()
    psr = _stale_record(); psr["arm"] = "psr_ce_all"; psr["object_kwargs"] = {"init_scheme": "zero", "lora_modules": None, "lora_rank": None}
    rc, fold = _run(tmp_path / "psr", psr, cfg, arm="psr_ce_all")
    assert rc == 0 and fold == {FIT_KEY: {"psr_ce_all": 1}}, capsys.readouterr().out   # init zero vs the config's lecun: recorded, not keyed; no expert map
    psr["seq_len"] = 2048
    assert _run(tmp_path / "psrseq", psr, cfg, arm="psr_ce_all")[0] == m.EXIT_STALE


PAPER_KEY, DEMO_KEY = "qwen35_4b_base", "qwen35_4b_demo_base"


def _gen():
    """The REAL generator (scripts/sdfbase/make_sdfbase_configs.py), loaded by path like memprobe_fold itself."""
    spec = importlib.util.spec_from_file_location("_mf_make_sdfbase_configs", REPO / "scripts" / "sdfbase" / "make_sdfbase_configs.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    return g


def _paper_lane_record(g) -> dict:
    """A record of the PAPER lane's selection — rank 4 / o_proj, no expert map, L8192, checkpointing, bf16, the pinned revision (what 02's
    probe writes for the demo key, whose selection is the paper's) — with the 2026-09-30 rungs: 1 at 69 %, 2 at 91 % (fits), 4 OOM."""
    r = _stale_record()
    spec = g.MODELS[DEMO_KEY]
    assert (spec["model"], spec["revision"]) == (g.MODELS[PAPER_KEY]["model"], g.MODELS[PAPER_KEY]["revision"])
    r["model"] = {"model": spec["model"], "revision": spec["revision"], "model_override": False}
    r["object_kwargs"] = {"init_scheme": "zero", "lora_modules": ["o_proj"], "lora_rank": 4, "per_expert_rank": None}
    return r


@pytest.mark.parametrize("headroom,bs", [(0.85, 1), (0.95, 2)])
def test_q4_a_record_folds_onto_the_eb8_and_eb64_renderings_alike_because_the_probe_measures_capacity_not_a_batch(tmp_path, capsys, headroom, bs):
    """Q4 (demo-import; RULED: LEAVE the fingerprint, no EFF cross-check in the fold): the memprobe runs every rung at accum_steps=1
    (memprobe_fit.py) and records CAPACITY and pace only — no effective batch — so the unit fingerprint is keyed on what the probe
    measured (arm, model@rev, selection, seq_len, checkpointing, dtype) and never on the lane's effective batch, the fold writes the
    micro-batch alone, and the GENERATOR derives accum_steps = ceil(effective_batch_for(key) / bs) at render time. Hence a record of the
    paper lane's selection folds onto the demo's eb-64 config unchanged: the same fingerprint (the string 02 keys the unit on) for both
    arms, no staleness against either config, the same chosen micro-batch whichever config it is folded against, the EFF64 sentinel
    standing in the demo's config before the fold, and post-fold accum 64 / bs beside the paper's 8 / bs — the two configs differ in
    batch and seed alone. The demo's first `02 --step memprobe` is simply its first probe; nothing is re-probed "at" a batch."""
    m = _mod(); g = _gen()
    bare = g.build_all(folds_path=None)                                       # every lane unfolded: EFF8 on the paper lane, EFF64 on the demo
    eb8 = yaml.safe_load(bare[g.config_filename(PAPER_KEY)]); eb64 = yaml.safe_load(bare[g.config_filename(DEMO_KEY)])
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert eb8["train_kwargs"][arm]["accum_steps"] == "PENDING_MEMPROBE_EFF8_ACCUM"
        assert eb64["train_kwargs"][arm]["accum_steps"] == "PENDING_MEMPROBE_EFF64_ACCUM"
        assert eb8["batch_size_by_arm"][arm] == eb64["batch_size_by_arm"][arm] == "MEMPROBE"
        assert m.fingerprint(eb8, arm) == m.fingerprint(eb64, arm), arm       # the batch is not in the key
    assert m.fingerprint(eb64, "lora_r4_ce") == f"r4-o-pe-none-L8192-gc1-bf16-m{m.model_tag(eb64)}" and m.model_tag(eb8) == m.model_tag(eb64)
    # the demo lane's declared fit seed: 3 since the one-seed objects/DEMO.json — every fitted Qwen3.5-4B intervention under ONE
    # training seed, the best difference LoRA's; the paper lane declares none
    assert "seed" not in eb8["train_kwargs"]["default"] and eb64["train_kwargs"]["default"]["seed"] == 3 == g.MODELS[DEMO_KEY]["seed"]
    rec = _paper_lane_record(g)
    assert m.staleness(rec, eb8, "lora_r4_ce") == [] == m.staleness(rec, eb64, "lora_r4_ce")
    assert m.record_identity(rec)["lora_modules"] == ["o_proj"] and m.config_selection(eb64, "lora_r4_ce")["lora_rank"] == 4
    # the REAL fold (the CLI 02 calls) against each config: the same micro-batch, the fold file carrying the micro-batch alone
    folds = {}
    for key, cfg in ((PAPER_KEY, eb8), (DEMO_KEY, eb64)):
        d = tmp_path / key; d.mkdir()
        (d / "rec.json").write_text(json.dumps(rec)); (d / "cfg.yaml").write_text(yaml.safe_dump(cfg))
        rc = m.main([str(d / "rec.json"), "--config", str(d / "cfg.yaml"), "--fit-key", key, "--arm", "lora_r4_ce", "--out", str(d / "fold.json"),
                     "--headroom", str(headroom)])
        fold_out = capsys.readouterr().out
        assert rc == 0, fold_out
        assert f"fold -> {d / 'fold.json'} (micro-batch {bs}:" in fold_out, fold_out
        folds[key] = json.loads((d / "fold.json").read_text())
        assert m.main(["--fingerprint", "--config", str(d / "cfg.yaml"), "--arm", "lora_r4_ce"]) == 0
        folds[key + ":fp"] = capsys.readouterr().out.strip().splitlines()[-1]
    assert folds[PAPER_KEY] == {PAPER_KEY: {"lora_r4_ce": bs}} and folds[DEMO_KEY] == {DEMO_KEY: {"lora_r4_ce": bs}}
    assert folds[PAPER_KEY + ":fp"] == folds[DEMO_KEY + ":fp"] == m.fingerprint(eb64, "lora_r4_ce")
    # rendered with that fold: the same micro-batch, accum ceil(8 / bs) on the paper lane and ceil(64 / bs) on the demo — the realised
    # batch (bs x accum, tune_arm's rule) is each lane's own
    mp = {PAPER_KEY: {"lora_r4_ce": bs, "psr_ce_all": bs}, DEMO_KEY: {"lora_r4_ce": bs, "psr_ce_all": bs}}
    folded = g.build_all(memprobe=mp, folds_path=None)
    c8 = yaml.safe_load(folded[g.config_filename(PAPER_KEY)]); c64 = yaml.safe_load(folded[g.config_filename(DEMO_KEY)])
    for arm in ("psr_ce_all", "lora_r4_ce"):
        assert c8["batch_size_by_arm"][arm] == c64["batch_size_by_arm"][arm] == bs
        assert c8["train_kwargs"][arm]["accum_steps"] == -(-8 // bs) and c64["train_kwargs"][arm]["accum_steps"] == -(-64 // bs)
        assert bs * c8["train_kwargs"][arm]["accum_steps"] == 8 and bs * c64["train_kwargs"][arm]["accum_steps"] == 64
        assert m.fingerprint(c8, arm) == m.fingerprint(c64, arm) == m.fingerprint(eb64, arm)     # folding moves nothing in the key
    assert "accum_steps = ceil(64 / bs) — effective batch 64." in folded[g.config_filename(DEMO_KEY)]
    assert "accum_steps = ceil(8 / bs) — effective batch 8." in folded[g.config_filename(PAPER_KEY)]


def test_02_folds_through_the_tool_with_the_card_class_archives_fail_closed_and_keys_the_unit_on_the_fingerprint():
    src = (REPO / "reproduce" / "02_prepare_model.sh").read_text()
    frag = src[src.index("if want memprobe; then"):src.index("if want register; then")]
    assert "scripts/sdfbase/memprobe_fold.py" in frag and '"${gpu_expect[@]}"' in frag and "CLUSTER_GPU_NAME" in frag
    assert 'dest="$(memprobe_archive_stale "$PRIMARY" "$STATE" "$arm")"' in frag and "nothing cleared, nothing submitted" in frag
    assert frag.index("memprobe_archive_stale") < frag.index('rm -f "$(repro_state_dir "$MODEL")/$STAGE/DONE.$unit"'), "the marker is cleared only after the archive read back"
    assert "--fingerprint" in frag and 'funit="memprobe.$arm.$fp"' in frag and 'lwid="$MODEL/$STAGE/memprobe/$arm/$fp"' in frag and 'slurm_submit --unit "$funit"' in frag
    assert 'slurm_mirror_clusters "$SHA" "$PRIMARY"' in frag, "the pinned sha is mirrored/verified before the probe's sbatch"
    assert "7) repro_fatal" in frag and "; true\"" not in frag
    assert frag.index("memprobe_fold.py") < frag.index('repro_done "$MODEL" "$STAGE" "$unit"')
    sb = (REPO / "scripts" / "sbatch" / "memprobe_fit.sbatch").read_text()
    assert "MEMPROBE_SEQ_LEN" not in sb.split("srun --ntasks=1", 1)[1], "the probe length is the config's, never an env override"
